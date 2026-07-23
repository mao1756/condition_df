"""Zero-residual stationarity diagnostics for the production D0 reverse kernel.

This module deliberately exercises the same elementary direct-Doob substep used
by generation.  It starts from the symmetric finite-grid Dirichlet reference,
sets the learned physical residual to zero, and compares the terminal law with
an independently sampled Dirichlet bank while refining the reference time step.

The fixed-grid stationarity decision and the stronger negligible-intervention
status are reported separately.  Neither decision is a spatial
Dirichlet--Ferguson refinement claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.diag_forward_noising import (
    expected_symmetric_dirichlet_entropy,
    make_rate_schedule,
    two_sample_state_diagnostics,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    edge_alpha_value,
    masked_reference_free_step_torch,
    natural_horizon,
)
from mnist.experiment12_d0 import _direct_doob_reverse_substep


@dataclass(frozen=True)
class ZeroResidualDiagnosticConfig:
    """Run and gate settings for the zero-residual reference diagnostic."""

    num_paths: int = 128
    sample_steps: int = 8
    substep_levels: tuple[int, ...] = (2, 4, 8)
    horizon: float | None = None
    tau_eff: float = 2e-4
    time_change_mode: str = "integral"
    rate_ramp: str = "none"
    rate_ramp_ratio: float = 1.0
    rate_min: float | None = None
    rate_max: float | None = None
    seed: int = 260715
    calibration_reps: int = 8
    stationarity_quantile_multiplier: float = 3.0
    stationarity_mmd_multiplier: float = 3.0
    stationarity_quantile_floor: float = 0.02
    stationarity_mmd_floor: float = 1e-4
    entropy_error_floor: float = 2e-3
    second_moment_error_floor: float = 2e-4
    max_standardized_moment_error: float = 4.0
    max_simplex_mass_error: float = 2e-6
    max_floor_correction_l1: float = 1e-8
    max_renorm_correction_l1: float = 1e-6
    refinement_contraction: float = 0.90
    refinement_absolute_tolerance: float = 1e-7
    intervention_nonincrease_tolerance: float = 0.05
    strict_raw_limiter_threshold: float = 5e-3
    strict_weighted_limiter_threshold: float = 5e-4
    include_forward_control: bool = False
    preflight_only: bool = False
    preflight_paths: int = 16
    preflight_reps: int = 1
    preflight_limiter_fractions: tuple[float, ...] = ()
    preflight_max_substeps: int = 64

    def __post_init__(self) -> None:
        levels = parse_substep_levels(self.substep_levels)
        object.__setattr__(self, "substep_levels", levels)
        if int(self.num_paths) <= 1:
            raise ValueError("num_paths must exceed one")
        if int(self.sample_steps) <= 0:
            raise ValueError("sample_steps must be positive")
        if int(self.calibration_reps) <= 0:
            raise ValueError("calibration_reps must be positive")
        limiter_fractions = parse_limiter_fractions(self.preflight_limiter_fractions)
        object.__setattr__(self, "preflight_limiter_fractions", limiter_fractions)
        if int(self.preflight_paths) <= 1:
            raise ValueError("preflight_paths must exceed one")
        if int(self.preflight_reps) <= 0:
            raise ValueError("preflight_reps must be positive")
        if self.preflight_only:
            if int(self.preflight_max_substeps) < max(levels):
                raise ValueError(
                    "preflight_max_substeps must be at least the finest requested level"
                )
            cap_ratio = int(self.preflight_max_substeps) // max(levels)
            if (
                int(self.preflight_max_substeps) % max(levels) != 0
                or cap_ratio & (cap_ratio - 1)
            ):
                raise ValueError(
                    "preflight_max_substeps must be the finest requested level times a power of two"
                )
        if not math.isfinite(float(self.tau_eff)) or float(self.tau_eff) <= 0.0:
            raise ValueError("tau_eff must be finite and positive")
        if self.horizon is not None and (
            not math.isfinite(float(self.horizon)) or float(self.horizon) <= 0.0
        ):
            raise ValueError("horizon must be finite and positive when supplied")


@dataclass
class ZeroResidualDiagnosticResult:
    """In-memory result with JSON-friendly summaries and optional state banks."""

    config: ZeroResidualDiagnosticConfig
    dynamics_config: DirectFluxMNISTConfig
    rate_schedule: np.ndarray
    reference_rate_integral: float
    dirichlet_alpha: float
    calibration: dict[str, float | int]
    levels: list[dict[str, float | int | str]]
    gate: dict[str, float | int | str]
    initial_states: np.ndarray
    states_by_level: dict[int, np.ndarray]
    reference_states: np.ndarray
    forward_control_levels: list[dict[str, float | int | str]] = field(default_factory=list)

    @property
    def states_finest(self) -> np.ndarray:
        return self.states_by_level[max(self.states_by_level)]

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, object]:
        out: dict[str, object] = {
            "config": asdict(self.config),
            "dynamics_config": asdict(self.dynamics_config),
            "rate_schedule": self.rate_schedule.tolist(),
            "reference_rate_integral": float(self.reference_rate_integral),
            "dirichlet_alpha": float(self.dirichlet_alpha),
            "calibration": dict(self.calibration),
            "levels": [dict(row) for row in self.levels],
            "gate": dict(self.gate),
            "forward_control_levels": [dict(row) for row in self.forward_control_levels],
        }
        if include_arrays:
            out["initial_states"] = self.initial_states
            out["states_finest"] = self.states_finest
            out["reference_states"] = self.reference_states
            out["states_by_level"] = dict(self.states_by_level)
        return out


@dataclass
class ZeroResidualPreflightResult:
    """Forecast-only intervention sweep with no stationarity or gate decision."""

    config: ZeroResidualDiagnosticConfig
    dynamics_config: DirectFluxMNISTConfig
    rate_schedule: np.ndarray
    reference_rate_integral: float
    levels_evaluated: tuple[int, ...]
    limiter_fractions: tuple[float, ...]
    metrics: list[dict[str, float | int | str]]
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "forecast-only-intervention-preflight",
            "decision_scope": "intervention forecast only; no law or acceptance decision",
            "preflight": {
                "paths": int(self.config.preflight_paths),
                "reps": int(self.config.preflight_reps),
                "requested_substeps": list(self.config.substep_levels),
                "max_substeps": int(self.config.preflight_max_substeps),
                "limiter_fractions": list(self.limiter_fractions),
                "seed": int(self.config.seed),
                "raw_intervention_threshold": float(
                    self.config.strict_raw_limiter_threshold
                ),
                "weighted_intervention_threshold": float(
                    self.config.strict_weighted_limiter_threshold
                ),
            },
            "dynamics_config": asdict(self.dynamics_config),
            "rate_schedule": self.rate_schedule.tolist(),
            "reference_rate_integral": float(self.reference_rate_integral),
            "levels_evaluated": list(self.levels_evaluated),
            "summary": dict(self.summary),
        }


def parse_substep_levels(value: str | Sequence[int]) -> tuple[int, ...]:
    """Return sorted unique refinement levels compatible with Brownian coupling."""

    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        levels = tuple(int(piece) for piece in pieces)
    else:
        levels = tuple(int(item) for item in value)
    levels = tuple(sorted(set(levels)))
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("substep levels must contain positive integers")
    finest = max(levels)
    if any(finest % level != 0 for level in levels):
        raise ValueError("each substep level must divide the finest level for coupling")
    return levels


def parse_limiter_fractions(value: str | Sequence[float]) -> tuple[float, ...]:
    """Return sorted unique limiter fractions for the forecast sweep."""

    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        fractions = tuple(float(piece) for piece in pieces)
    else:
        fractions = tuple(float(item) for item in value)
    fractions = tuple(sorted(set(fractions)))
    for fraction in fractions:
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("preflight limiter fractions must be finite and in (0, 1]")
    return fractions


def make_coupled_rate_schedule(
    sample_steps: int,
    *,
    horizon: float,
    tau_eff: float,
    time_change_mode: str = "integral",
    ramp: str = "none",
    ramp_ratio: float = 1.0,
    rate_min: float | None = None,
    rate_max: float | None = None,
) -> np.ndarray:
    """Build the faithful rate schedule shared by all temporal refinements."""

    if not math.isfinite(float(tau_eff)) or float(tau_eff) <= 0.0:
        raise ValueError("tau_eff must be finite and positive")

    schedule = make_rate_schedule(
        int(sample_steps),
        mode="faithful",
        tau_eff=float(tau_eff),
        constant_rate=None,
        ramp=str(ramp),
        ramp_ratio=float(ramp_ratio),
        rate_min=rate_min,
        rate_max=rate_max,
        horizon=float(horizon),
        time_change_mode=str(time_change_mode),
    )
    if not np.isfinite(schedule).all() or np.any(schedule < 0.0):
        raise ValueError("rate schedule must be finite and non-negative")
    return schedule.astype(np.float64, copy=False)


def _sample_symmetric_dirichlet(
    count: int,
    dim: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(count) <= 0 or int(dim) <= 0:
        raise ValueError("Dirichlet sample dimensions must be positive")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("Dirichlet alpha must be finite and positive")
    raw = rng.gamma(shape=float(alpha), scale=1.0, size=(int(count), int(dim)))
    raw = np.maximum(raw, np.finfo(np.float64).tiny)
    raw /= raw.sum(axis=1, keepdims=True)
    return raw.astype(np.float64, copy=False)


def _calibrate_dirichlet_null(
    dynamics_config: DirectFluxMNISTConfig,
    *,
    num_paths: int,
    alpha: float,
    seed: int,
    reps: int,
    quantile_multiplier: float,
    mmd_multiplier: float,
    quantile_floor: float,
    mmd_floor: float,
) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed))
    dim = int(dynamics_config.grid_size) ** 2
    quantiles: list[float] = []
    mmds: list[float] = []
    for _ in range(max(1, int(reps))):
        a = _sample_symmetric_dirichlet(num_paths, dim, alpha, rng)
        b = _sample_symmetric_dirichlet(num_paths, dim, alpha, rng)
        row = two_sample_state_diagnostics(
            a,
            b,
            dynamics_config,
            prefix="null",
            max_samples=int(num_paths),
        )
        quantiles.append(float(row["null_quantile_distance"]))
        mmds.append(float(row["null_feature_mmd"]))
    q_mean = float(np.mean(quantiles))
    mmd_mean = float(np.mean(mmds))
    return {
        "calibration_reps": int(reps),
        "calibration_samples": int(num_paths),
        "quantile_baseline_mean": q_mean,
        "feature_mmd_baseline_mean": mmd_mean,
        "quantile_threshold": max(float(quantile_floor), float(quantile_multiplier) * q_mean),
        "feature_mmd_threshold": max(float(mmd_floor), float(mmd_multiplier) * mmd_mean),
    }


def _observables(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(states, dtype=np.float64)
    safe = np.clip(x, np.finfo(np.float64).tiny, None)
    entropy = -(safe * np.log(safe)).sum(axis=1)
    second_moment = np.square(x).sum(axis=1)
    return entropy, second_moment


def _standard_error_units(
    values: np.ndarray,
    reference: np.ndarray,
    *,
    error_floor: float,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    delta = float(values.mean() - reference.mean())
    se = math.sqrt(
        float(values.var(ddof=1)) / max(values.size, 1)
        + float(reference.var(ddof=1)) / max(reference.size, 1)
    )
    units = abs(delta) / max(float(se), float(error_floor), 1e-30)
    return delta, se, float(units)


def _one_sample_standard_error_units(
    values: np.ndarray,
    *,
    expected: float,
    error_floor: float,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    delta = float(values.mean() - float(expected))
    se = float(values.std(ddof=1)) / math.sqrt(max(values.size, 1))
    units = abs(delta) / max(float(se), float(error_floor), 1e-30)
    return delta, se, float(units)


def _empty_accumulator() -> dict[str, float]:
    return {
        "limited_edges": 0.0,
        "proposed_edges": 0.0,
        "mobility_weight_sum": 0.0,
        "limited_mobility_weight_sum": 0.0,
        "noise_energy_sum": 0.0,
        "limited_noise_energy_sum": 0.0,
        "nonfinite_edges": 0.0,
        "floor_touched_pixels": 0.0,
        "floor_proposed_pixels": 0.0,
        "floor_correction_l1": 0.0,
        "renorm_correction_l1": 0.0,
        "max_simplex_mass_error": 0.0,
        "free_square_mean_sum": 0.0,
        "noise_square_mean_sum": 0.0,
        "step_count": 0.0,
    }


def _accumulate_direct_step(acc: dict[str, float], step) -> None:
    diagnostics = step.diagnostics
    for key in (
        "limited_edges",
        "proposed_edges",
        "mobility_weight_sum",
        "limited_mobility_weight_sum",
        "noise_energy_sum",
        "limited_noise_energy_sum",
        "nonfinite_edges",
        "floor_touched_pixels",
        "floor_proposed_pixels",
        "floor_correction_l1",
        "renorm_correction_l1",
    ):
        acc[key] += float(diagnostics.get(key, 0.0))
    acc["max_simplex_mass_error"] = max(
        acc["max_simplex_mass_error"],
        float(diagnostics.get("max_simplex_mass_error", 0.0)),
    )
    acc["free_square_mean_sum"] += float(step.free_delta.detach().double().square().mean().cpu())
    acc["noise_square_mean_sum"] += float(step.noise_delta.detach().double().square().mean().cpu())
    acc["step_count"] += 1.0


def _empty_device_accumulator(reference: Tensor) -> dict[str, Tensor]:
    """Create GPU/CPU tensor totals that can span many elementary steps."""

    zero = reference.new_zeros((), dtype=torch.float64)
    return {key: zero.clone() for key in _empty_accumulator()}


def _accumulate_direct_step_device(acc: dict[str, Tensor], step) -> None:
    diagnostics = step.diagnostics
    for key in (
        "limited_edges",
        "proposed_edges",
        "mobility_weight_sum",
        "limited_mobility_weight_sum",
        "noise_energy_sum",
        "limited_noise_energy_sum",
        "nonfinite_edges",
        "floor_touched_pixels",
        "floor_proposed_pixels",
        "floor_correction_l1",
        "renorm_correction_l1",
    ):
        value = diagnostics.get(key)
        if not isinstance(value, Tensor):
            raise TypeError("device diagnostic mode must return tensor totals")
        acc[key].add_(value.to(dtype=torch.float64))
    maximum = diagnostics.get("max_simplex_mass_error")
    if not isinstance(maximum, Tensor):
        raise TypeError("device diagnostic mode must return a tensor simplex error")
    acc["max_simplex_mass_error"] = torch.maximum(
        acc["max_simplex_mass_error"], maximum.to(dtype=torch.float64)
    )
    acc["free_square_mean_sum"].add_(
        step.free_delta.detach().float().square().mean().to(torch.float64)
    )
    acc["noise_square_mean_sum"].add_(
        step.noise_delta.detach().float().square().mean().to(torch.float64)
    )
    acc["step_count"].add_(1.0)


def _device_accumulator_to_host(acc: Mapping[str, Tensor]) -> dict[str, float]:
    """Synchronize a complete accumulator in one device-to-host transfer."""

    keys = tuple(_empty_accumulator())
    values = torch.stack([acc[key].detach().to(torch.float64) for key in keys])
    host = values.cpu().tolist()
    return {key: float(value) for key, value in zip(keys, host)}


def _accumulator_summary(acc: Mapping[str, float], *, num_paths: int) -> dict[str, float | int]:
    proposed = float(acc["proposed_edges"])
    mobility = float(acc["mobility_weight_sum"])
    noise_energy = float(acc["noise_energy_sum"])
    path_substeps = max(float(num_paths) * float(acc["step_count"]), 1.0)
    return {
        "limiter_fraction": 0.0 if proposed <= 0.0 else float(acc["limited_edges"]) / proposed,
        "mobility_weighted_limiter_fraction": (
            0.0 if mobility <= 0.0 else float(acc["limited_mobility_weight_sum"]) / mobility
        ),
        "noise_energy_weighted_limiter_fraction": (
            0.0 if noise_energy <= 0.0 else float(acc["limited_noise_energy_sum"]) / noise_energy
        ),
        "limited_edges": int(acc["limited_edges"]),
        "proposed_edges": int(acc["proposed_edges"]),
        "nonfinite_edges": int(acc["nonfinite_edges"]),
        "floor_touched_pixels": int(acc["floor_touched_pixels"]),
        "floor_proposed_pixels": int(acc["floor_proposed_pixels"]),
        "floor_correction_l1_per_path_substep": float(acc["floor_correction_l1"]) / path_substeps,
        "renorm_correction_l1_per_path_substep": float(acc["renorm_correction_l1"]) / path_substeps,
        "max_simplex_mass_error": float(acc["max_simplex_mass_error"]),
        "free_step_rms": math.sqrt(float(acc["free_square_mean_sum"]) / max(float(acc["step_count"]), 1.0)),
        "noise_step_rms": math.sqrt(float(acc["noise_square_mean_sum"]) / max(float(acc["step_count"]), 1.0)),
        "kernel_substeps_executed": int(acc["step_count"]),
    }


@torch.no_grad()
def _run_coupled_direct_levels(
    initial_states: np.ndarray,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    rate_schedule: np.ndarray,
    horizon: float,
    substep_levels: Sequence[int],
    seed: int,
    device: torch.device,
    progress_callback: Callable[[Mapping[str, int | str]], None] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float | int]]]:
    levels = parse_substep_levels(substep_levels)
    finest = max(levels)
    dtype = torch.float32
    initial = torch.as_tensor(initial_states, dtype=dtype, device=device)
    states = {level: initial.clone() for level in levels}
    accumulators = {level: _empty_device_accumulator(initial) for level in levels}
    buffers = {
        level: torch.zeros(
            (initial.shape[0], 2, int(dynamics_config.grid_size), int(dynamics_config.grid_size)),
            dtype=dtype,
            device=device,
        )
        for level in levels
    }
    try:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    except TypeError:
        generator = torch.Generator(device=device.type).manual_seed(int(seed))
    zero = torch.zeros_like(buffers[finest])
    sample_steps = int(np.asarray(rate_schedule).size)
    if sample_steps <= 0:
        raise ValueError("rate_schedule must be nonempty")
    dt_by_level = {level: float(horizon) / float(sample_steps * level) for level in levels}

    for outer_k in range(sample_steps - 1, -1, -1):
        rate = float(rate_schedule[outer_k])
        for fine_index in range(finest):
            fine_normal = torch.randn(
                zero.shape,
                dtype=zero.dtype,
                device=zero.device,
                generator=generator,
            )
            for level in levels:
                group = finest // level
                buffers[level].add_(fine_normal)
                if (fine_index + 1) % group != 0:
                    continue
                standard_normal = buffers[level] / math.sqrt(float(group))
                step = _direct_doob_reverse_substep(
                    states[level],
                    zero,
                    rate=rate,
                    dt=dt_by_level[level],
                    dynamics_config=dynamics_config,
                    standard_normal=standard_normal,
                    deterministic=False,
                    diagnostics_device=True,
                )
                states[level] = step.states
                _accumulate_direct_step_device(accumulators[level], step)
                buffers[level].zero_()
        if progress_callback is not None:
            completed_outer = sample_steps - outer_k
            progress_callback(
                {
                    "phase": "direct",
                    "outer_completed": int(completed_outer),
                    "outer_total": int(sample_steps),
                    "work_completed": int(completed_outer * sum(levels)),
                    "work_total": int(sample_steps * sum(levels)),
                }
            )

    states_np = {
        level: states[level].detach().cpu().numpy().astype(np.float64)
        for level in levels
    }
    summaries = {
        level: _accumulator_summary(
            _device_accumulator_to_host(accumulators[level]),
            num_paths=initial.shape[0],
        )
        for level in levels
    }
    return states_np, summaries


def _evaluate_level(
    terminal_states: np.ndarray,
    reference_states: np.ndarray,
    initial_states: np.ndarray,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    dirichlet_alpha: float,
    calibration: Mapping[str, float | int],
    entropy_error_floor: float,
    second_moment_error_floor: float,
) -> dict[str, float | int]:
    comparison = two_sample_state_diagnostics(
        terminal_states,
        reference_states,
        dynamics_config,
        prefix="stationarity",
        max_samples=min(len(terminal_states), len(reference_states)),
    )
    entropy, second = _observables(terminal_states)
    ref_entropy, ref_second = _observables(reference_states)
    initial_entropy, initial_second = _observables(initial_states)
    entropy_delta, entropy_se, entropy_units = _standard_error_units(
        entropy,
        ref_entropy,
        error_floor=float(entropy_error_floor),
    )
    second_delta, second_se, second_units = _standard_error_units(
        second,
        ref_second,
        error_floor=float(second_moment_error_floor),
    )
    expected_entropy = float(expected_symmetric_dirichlet_entropy(dynamics_config))
    dim = int(dynamics_config.grid_size) ** 2
    expected_second = (float(dirichlet_alpha) + 1.0) / (
        float(dim) * float(dirichlet_alpha) + 1.0
    )
    entropy_analytic_delta, entropy_analytic_se, entropy_analytic_units = (
        _one_sample_standard_error_units(
            entropy,
            expected=expected_entropy,
            error_floor=float(entropy_error_floor),
        )
    )
    second_analytic_delta, second_analytic_se, second_analytic_units = (
        _one_sample_standard_error_units(
            second,
            expected=expected_second,
            error_floor=float(second_moment_error_floor),
        )
    )
    entropy_drift_delta, entropy_drift_se, entropy_drift_units = (
        _one_sample_standard_error_units(
            entropy - initial_entropy,
            expected=0.0,
            error_floor=float(entropy_error_floor),
        )
    )
    second_drift_delta, second_drift_se, second_drift_units = (
        _one_sample_standard_error_units(
            second - initial_second,
            expected=0.0,
            error_floor=float(second_moment_error_floor),
        )
    )
    qdist = float(comparison["stationarity_quantile_distance"])
    mmd = float(comparison["stationarity_feature_mmd"])
    q_threshold = float(calibration["quantile_threshold"])
    mmd_threshold = float(calibration["feature_mmd_threshold"])
    return {
        "stationarity_samples": int(comparison["stationarity_samples"]),
        "stationarity_quantile_distance": qdist,
        "stationarity_quantile_threshold": q_threshold,
        "stationarity_quantile_ratio": qdist / max(q_threshold, 1e-30),
        "stationarity_feature_mmd": mmd,
        "stationarity_feature_mmd_threshold": mmd_threshold,
        "stationarity_feature_mmd_ratio": mmd / max(mmd_threshold, 1e-30),
        "stationarity_pass_quantile": int(qdist <= q_threshold),
        "stationarity_pass_feature_mmd": int(mmd <= mmd_threshold),
        "entropy_mean": float(entropy.mean()),
        "entropy_reference_mean": float(ref_entropy.mean()),
        "entropy_delta": float(entropy_delta),
        "entropy_standard_error": float(entropy_se),
        "entropy_standard_error_units": float(entropy_units),
        "entropy_analytic_expected": float(expected_entropy),
        "entropy_analytic_delta": float(entropy_analytic_delta),
        "entropy_analytic_standard_error": float(entropy_analytic_se),
        "entropy_analytic_standard_error_units": float(entropy_analytic_units),
        "entropy_paired_drift": float(entropy_drift_delta),
        "entropy_paired_drift_standard_error": float(entropy_drift_se),
        "entropy_paired_drift_standard_error_units": float(entropy_drift_units),
        "second_moment_mean": float(second.mean()),
        "second_moment_reference_mean": float(ref_second.mean()),
        "second_moment_delta": float(second_delta),
        "second_moment_standard_error": float(second_se),
        "second_moment_standard_error_units": float(second_units),
        "second_moment_analytic_expected": float(expected_second),
        "second_moment_analytic_delta": float(second_analytic_delta),
        "second_moment_analytic_standard_error": float(second_analytic_se),
        "second_moment_analytic_standard_error_units": float(second_analytic_units),
        "second_moment_paired_drift": float(second_drift_delta),
        "second_moment_paired_drift_standard_error": float(second_drift_se),
        "second_moment_paired_drift_standard_error_units": float(second_drift_units),
    }


def evaluate_refinement_gate(
    levels: Sequence[Mapping[str, float | int | str]],
    *,
    max_simplex_mass_error: float = 2e-6,
    max_floor_correction_l1: float = 1e-8,
    max_renorm_correction_l1: float = 1e-6,
    max_standardized_moment_error: float = 4.0,
    refinement_contraction: float = 0.90,
    refinement_absolute_tolerance: float = 1e-7,
    intervention_nonincrease_tolerance: float = 0.05,
    strict_raw_limiter_threshold: float = 5e-3,
    strict_weighted_limiter_threshold: float = 5e-4,
) -> dict[str, float | int | str]:
    """Apply the fixed-grid gate without promoting it to a strict-limit claim."""

    rows = sorted((dict(row) for row in levels), key=lambda row: int(row["substeps"]))
    if not rows:
        raise ValueError("at least one refinement level is required")
    finest = rows[-1]
    stationarity_ok = bool(
        float(finest["stationarity_quantile_distance"])
        <= float(finest["stationarity_quantile_threshold"])
        and float(finest["stationarity_feature_mmd"])
        <= float(finest["stationarity_feature_mmd_threshold"])
        and float(finest["entropy_standard_error_units"])
        <= float(max_standardized_moment_error)
        and float(finest["entropy_analytic_standard_error_units"])
        <= float(max_standardized_moment_error)
        and float(finest["entropy_paired_drift_standard_error_units"])
        <= float(max_standardized_moment_error)
        and float(finest["second_moment_standard_error_units"])
        <= float(max_standardized_moment_error)
        and float(finest["second_moment_analytic_standard_error_units"])
        <= float(max_standardized_moment_error)
        and float(finest["second_moment_paired_drift_standard_error_units"])
        <= float(max_standardized_moment_error)
    )
    numerical_ok = all(
        int(row.get("nonfinite_edges", 0)) == 0
        and float(row.get("max_simplex_mass_error", float("inf")))
        <= float(max_simplex_mass_error)
        and float(row.get("floor_correction_l1_per_path_substep", float("inf")))
        <= float(max_floor_correction_l1)
        and float(row.get("renorm_correction_l1_per_path_substep", float("inf")))
        <= float(max_renorm_correction_l1)
        for row in rows
    )

    coupled_values = [
        float(row.get("coupled_refinement_rms", float("nan")))
        for row in rows[1:]
    ]
    # A contraction claim needs two successive coupled discrepancies, hence
    # at least three temporal levels.  Missing/nonfinite values fail closed.
    coupled_ok = len(coupled_values) >= 2 and all(
        math.isfinite(value) for value in coupled_values
    )
    if coupled_ok:
        for previous, current in zip(coupled_values[:-1], coupled_values[1:]):
            if current > float(refinement_contraction) * previous + float(refinement_absolute_tolerance):
                coupled_ok = False
                break

    # Coarse levels are allowed to be biased, but their discrepancy must improve
    # as dt shrinks until it reaches the independently calibrated null band.
    distribution_metrics = (
        ("stationarity_quantile_ratio", 1.0),
        ("stationarity_feature_mmd_ratio", 1.0),
        ("entropy_standard_error_units", float(max_standardized_moment_error)),
        ("entropy_analytic_standard_error_units", float(max_standardized_moment_error)),
        ("entropy_paired_drift_standard_error_units", float(max_standardized_moment_error)),
        ("second_moment_standard_error_units", float(max_standardized_moment_error)),
        ("second_moment_analytic_standard_error_units", float(max_standardized_moment_error)),
        ("second_moment_paired_drift_standard_error_units", float(max_standardized_moment_error)),
    )
    distribution_refinement_ok = all(
        math.isfinite(float(row.get(key, float("nan"))))
        for row in rows
        for key, _ in distribution_metrics
    )
    if distribution_refinement_ok:
        for coarse, fine in zip(rows[:-1], rows[1:]):
            for key, null_limit in distribution_metrics:
                coarse_value = float(coarse[key])
                fine_value = float(fine[key])
                if coarse_value <= null_limit and fine_value <= null_limit:
                    continue
                if fine_value > coarse_value * (1.0 + float(intervention_nonincrease_tolerance)) + 1e-12:
                    distribution_refinement_ok = False

    intervention_keys = (
        "limiter_fraction",
        "mobility_weighted_limiter_fraction",
        "noise_energy_weighted_limiter_fraction",
    )
    interventions_ok = all(
        math.isfinite(float(row.get(key, float("nan"))))
        for row in rows
        for key in intervention_keys
    )
    if interventions_ok:
        for coarse, fine in zip(rows[:-1], rows[1:]):
            for key in intervention_keys:
                coarse_value = float(coarse[key])
                fine_value = float(fine[key])
                if fine_value > coarse_value * (1.0 + float(intervention_nonincrease_tolerance)) + 1e-12:
                    interventions_ok = False

    # A strict-limit evidence flag needs more than a small value on one tested
    # grid: each intervention metric must contract under every dt refinement
    # (or already be indistinguishable from zero at the absolute tolerance).
    strict_intervention_decay_ok = interventions_ok
    if strict_intervention_decay_ok:
        for coarse, fine in zip(rows[:-1], rows[1:]):
            for key in intervention_keys:
                coarse_value = float(coarse[key])
                fine_value = float(fine[key])
                if fine_value > (
                    float(refinement_contraction) * coarse_value
                    + float(refinement_absolute_tolerance)
                ):
                    strict_intervention_decay_ok = False

    fixed_grid_pass = bool(
        stationarity_ok
        and numerical_ok
        and coupled_ok
        and distribution_refinement_ok
        and interventions_ok
    )
    strict_limit_supported = bool(
        fixed_grid_pass
        and float(finest.get("limiter_fraction", float("inf")))
        <= float(strict_raw_limiter_threshold)
        and float(finest.get("mobility_weighted_limiter_fraction", float("inf")))
        <= float(strict_weighted_limiter_threshold)
        and float(finest.get("noise_energy_weighted_limiter_fraction", float("inf")))
        <= float(strict_weighted_limiter_threshold)
        and strict_intervention_decay_ok
    )
    return {
        "gate_pass_stationarity": int(stationarity_ok),
        "gate_pass_numerical_health": int(numerical_ok),
        "gate_pass_coupled_refinement": int(coupled_ok),
        "gate_pass_distributional_refinement": int(distribution_refinement_ok),
        "gate_pass_nonincreasing_interventions": int(interventions_ok),
        "gate_pass_strict_intervention_decay": int(strict_intervention_decay_ok),
        "fixed_grid_stationarity_pass": int(fixed_grid_pass),
        "strict_h_transform_limit_supported": int(strict_limit_supported),
        "strict_raw_limiter_threshold": float(strict_raw_limiter_threshold),
        "strict_weighted_limiter_threshold": float(strict_weighted_limiter_threshold),
        "refinement_levels_evaluated": int(len(rows)),
        "minimum_refinement_levels": 3,
        "finest_substeps": int(finest["substeps"]),
        "finest_limiter_fraction": float(finest.get("limiter_fraction", float("nan"))),
        "finest_mobility_weighted_limiter_fraction": float(
            finest.get("mobility_weighted_limiter_fraction", float("nan"))
        ),
        "finest_noise_energy_weighted_limiter_fraction": float(
            finest.get("noise_energy_weighted_limiter_fraction", float("nan"))
        ),
        "claim_scope": "fixed-grid temporal refinement only",
    }


def evaluate_forward_reference_control(
    levels: Sequence[Mapping[str, float | int | str]],
    *,
    max_simplex_mass_error: float = 2e-6,
    max_floor_correction_l1: float = 1e-8,
    max_renorm_correction_l1: float = 1e-6,
    max_standardized_moment_error: float = 4.0,
    intervention_nonincrease_tolerance: float = 0.05,
    strict_raw_limiter_threshold: float = 5e-3,
    strict_weighted_limiter_threshold: float = 5e-4,
) -> dict[str, float | int | str]:
    """Evaluate the forward cache/reference integrator without coupled-RMS claims."""

    rows = sorted((dict(row) for row in levels), key=lambda row: int(row["substeps"]))
    if not rows:
        return {
            "forward_reference_control_evaluated": 0,
            "forward_reference_control_pass": 0,
            "forward_control_levels_evaluated": 0,
            "forward_control_minimum_levels": 3,
            "forward_control_claim_scope": "forward reference fixed-grid temporal control",
        }
    finest = rows[-1]
    moment_keys = (
        "entropy_standard_error_units",
        "entropy_analytic_standard_error_units",
        "entropy_paired_drift_standard_error_units",
        "second_moment_standard_error_units",
        "second_moment_analytic_standard_error_units",
        "second_moment_paired_drift_standard_error_units",
    )
    stationarity_ok = bool(
        float(finest.get("stationarity_quantile_distance", float("inf")))
        <= float(finest.get("stationarity_quantile_threshold", float("-inf")))
        and float(finest.get("stationarity_feature_mmd", float("inf")))
        <= float(finest.get("stationarity_feature_mmd_threshold", float("-inf")))
        and all(
            math.isfinite(float(finest.get(key, float("nan"))))
            and float(finest[key]) <= float(max_standardized_moment_error)
            for key in moment_keys
        )
    )
    numerical_ok = all(
        int(row.get("nonfinite_edges", 0)) == 0
        and math.isfinite(float(row.get("max_simplex_mass_error", float("nan"))))
        and float(row["max_simplex_mass_error"]) <= float(max_simplex_mass_error)
        and math.isfinite(float(row.get("floor_correction_l1_per_path_substep", float("nan"))))
        and float(row["floor_correction_l1_per_path_substep"]) <= float(max_floor_correction_l1)
        and math.isfinite(float(row.get("renorm_correction_l1_per_path_substep", float("nan"))))
        and float(row["renorm_correction_l1_per_path_substep"]) <= float(max_renorm_correction_l1)
        for row in rows
    )
    distribution_metrics = (
        ("stationarity_quantile_ratio", 1.0),
        ("stationarity_feature_mmd_ratio", 1.0),
        *((key, float(max_standardized_moment_error)) for key in moment_keys),
    )
    distribution_ok = all(
        math.isfinite(float(row.get(key, float("nan"))))
        for row in rows
        for key, _ in distribution_metrics
    )
    if distribution_ok:
        for coarse, fine in zip(rows[:-1], rows[1:]):
            for key, pass_band in distribution_metrics:
                coarse_value = float(coarse[key])
                fine_value = float(fine[key])
                if coarse_value <= pass_band and fine_value <= pass_band:
                    continue
                if fine_value > coarse_value * (
                    1.0 + float(intervention_nonincrease_tolerance)
                ) + 1e-12:
                    distribution_ok = False
    intervention_keys = (
        "limiter_fraction",
        "mobility_weighted_limiter_fraction",
        "noise_energy_weighted_limiter_fraction",
    )
    interventions_ok = all(
        math.isfinite(float(row.get(key, float("nan"))))
        for row in rows
        for key in intervention_keys
    )
    if interventions_ok:
        for coarse, fine in zip(rows[:-1], rows[1:]):
            for key in intervention_keys:
                if float(fine[key]) > float(coarse[key]) * (
                    1.0 + float(intervention_nonincrease_tolerance)
                ) + 1e-12:
                    interventions_ok = False
    levels_complete = len(rows) >= 3
    strict_interventions = bool(
        float(finest.get("limiter_fraction", float("inf")))
        <= float(strict_raw_limiter_threshold)
        and float(finest.get("mobility_weighted_limiter_fraction", float("inf")))
        <= float(strict_weighted_limiter_threshold)
        and float(finest.get("noise_energy_weighted_limiter_fraction", float("inf")))
        <= float(strict_weighted_limiter_threshold)
    )
    passed = bool(
        levels_complete
        and stationarity_ok
        and numerical_ok
        and distribution_ok
        and interventions_ok
        and strict_interventions
    )
    return {
        "forward_reference_control_evaluated": 1,
        "forward_control_levels_evaluated": int(len(rows)),
        "forward_control_minimum_levels": 3,
        "forward_control_pass_levels_complete": int(levels_complete),
        "forward_control_pass_stationarity": int(stationarity_ok),
        "forward_control_pass_numerical_health": int(numerical_ok),
        "forward_control_pass_distributional_refinement": int(distribution_ok),
        "forward_control_pass_nonincreasing_interventions": int(interventions_ok),
        "forward_control_strict_intervention_thresholds_met": int(strict_interventions),
        "forward_reference_control_pass": int(passed),
        "forward_control_finest_substeps": int(finest["substeps"]),
        "forward_control_claim_scope": "forward reference fixed-grid temporal control",
    }


@torch.no_grad()
def _run_forward_control_level(
    initial_states: np.ndarray,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    rate_schedule: np.ndarray,
    horizon: float,
    substeps: int,
    seed: int,
    device: torch.device,
    diagnostics_device: bool = False,
    progress_callback: Callable[[Mapping[str, int | str]], None] | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    states = torch.as_tensor(initial_states, dtype=torch.float32, device=device)
    totals = _empty_accumulator()
    tensor_totals: dict[str, Tensor] | None = None
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        outer_dt = float(horizon) / float(len(rate_schedule))
        # This is the production forward cache/reference control, so it follows
        # the schedule in forward order.  The direct reverse kernel above uses
        # the opposite order by construction.
        for outer_k in range(len(rate_schedule)):
            rate = float(rate_schedule[outer_k])
            result = masked_reference_free_step_torch(
                states,
                outer_dt,
                dynamics_config,
                free_weight=rate,
                noise_weight=math.sqrt(max(rate, 0.0)),
                substeps=int(substeps),
                deterministic=False,
                collect_diagnostics=True,
                diagnostics_device=bool(diagnostics_device),
            )
            states = result.states
            if diagnostics_device:
                if result.device_diagnostics is None:
                    raise RuntimeError("forward control did not return device diagnostics")
                if tensor_totals is None:
                    tensor_totals = {
                        key: value.clone() for key, value in result.device_diagnostics.items()
                    }
                else:
                    for key, value in result.device_diagnostics.items():
                        if key == "max_simplex_mass_error":
                            tensor_totals[key] = torch.maximum(tensor_totals[key], value)
                        else:
                            tensor_totals[key].add_(value)
            else:
                totals["limited_edges"] += float(result.limited_edges)
                totals["proposed_edges"] += float(result.proposed_edges)
                totals["mobility_weight_sum"] += float(result.mobility_weight_sum)
                totals["limited_mobility_weight_sum"] += float(result.limited_mobility_weight_sum)
                totals["noise_energy_sum"] += float(result.noise_energy_sum)
                totals["limited_noise_energy_sum"] += float(result.limited_noise_energy_sum)
                totals["nonfinite_edges"] += float(result.nonfinite_edges)
                totals["floor_touched_pixels"] += float(result.floor_touched_pixels)
                totals["floor_proposed_pixels"] += float(states.numel() * int(substeps))
                totals["floor_correction_l1"] += float(result.floor_correction_l1)
                totals["renorm_correction_l1"] += float(result.renorm_correction_l1)
                totals["max_simplex_mass_error"] = max(
                    totals["max_simplex_mass_error"],
                    float((states.sum(dim=1) - 1.0).abs().max().detach().cpu()),
                )
            totals["step_count"] += float(substeps)
            if progress_callback is not None:
                completed_outer = outer_k + 1
                progress_callback(
                    {
                        "phase": "forward-control",
                        "level": int(substeps),
                        "outer_completed": int(completed_outer),
                        "outer_total": int(len(rate_schedule)),
                        "work_completed": int(completed_outer * int(substeps)),
                        "work_total": int(len(rate_schedule) * int(substeps)),
                    }
                )
    if tensor_totals is not None:
        keys = tuple(key for key in tensor_totals if key in totals)
        values = torch.stack([tensor_totals[key].detach().to(torch.float64) for key in keys])
        for key, value in zip(keys, values.cpu().tolist()):
            totals[key] = float(value)
    summary = _accumulator_summary(totals, num_paths=states.shape[0])
    # The control integrator does not expose its separated free/noise transfer
    # tensors, so reporting zero component RMS values would be misleading.
    summary.pop("free_step_rms", None)
    summary.pop("noise_step_rms", None)
    return states.detach().cpu().numpy().astype(np.float64), summary


def _preflight_candidate_levels(
    requested_levels: Sequence[int],
    *,
    max_substeps: int,
) -> tuple[int, ...]:
    requested = parse_substep_levels(requested_levels)
    levels = list(requested)
    current = max(requested)
    if int(max_substeps) < current:
        raise ValueError("preflight_max_substeps must be at least the finest requested level")
    cap_ratio = int(max_substeps) // current
    if int(max_substeps) % current != 0 or cap_ratio & (cap_ratio - 1):
        raise ValueError(
            "preflight_max_substeps must be the finest requested level times a power of two"
        )
    while current < int(max_substeps):
        current *= 2
        levels.append(current)
    return tuple(sorted(set(levels)))


def _intervention_thresholds_met(
    summary: Mapping[str, float | int],
    *,
    raw_threshold: float,
    weighted_threshold: float,
) -> bool:
    return bool(
        float(summary.get("limiter_fraction", float("inf"))) <= float(raw_threshold)
        and float(summary.get("mobility_weighted_limiter_fraction", float("inf")))
        <= float(weighted_threshold)
        and float(summary.get("noise_energy_weighted_limiter_fraction", float("inf")))
        <= float(weighted_threshold)
    )


def _preflight_fraction_meets_at_level(
    metrics: Sequence[Mapping[str, float | int | str]],
    *,
    limiter_fraction: float,
    substeps: int,
    reps: int,
) -> bool:
    rows = [
        row
        for row in metrics
        if int(row["substeps"]) == int(substeps)
        and math.isclose(
            float(row["limiter_fraction_setting"]),
            float(limiter_fraction),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ]
    return bool(
        len(rows) == int(reps) * 2
        and {str(row["kernel"]) for row in rows}
        == {"direct-reverse", "forward-reference"}
        and all(int(row["intervention_thresholds_met"]) == 1 for row in rows)
    )


@torch.no_grad()
def _run_direct_preflight_step(
    initial_states: np.ndarray,
    standard_normal: Tensor,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    rate: float,
    dt: float,
    device: torch.device,
) -> dict[str, float | int]:
    """Probe one production direct-reverse elementary step."""

    states = torch.as_tensor(initial_states, dtype=torch.float32, device=device)
    if tuple(standard_normal.shape) != (
        states.shape[0],
        2,
        int(dynamics_config.grid_size),
        int(dynamics_config.grid_size),
    ):
        raise ValueError("standard_normal has the wrong shape for the direct preflight")
    zero = torch.zeros_like(standard_normal)
    step = _direct_doob_reverse_substep(
        states,
        zero,
        rate=float(rate),
        dt=float(dt),
        dynamics_config=dynamics_config,
        standard_normal=standard_normal,
        deterministic=False,
    )
    accumulator = _empty_accumulator()
    _accumulate_direct_step(accumulator, step)
    return _accumulator_summary(accumulator, num_paths=states.shape[0])


@torch.no_grad()
def _run_forward_preflight_step(
    initial_states: np.ndarray,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    rate: float,
    dt: float,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Probe one production forward-reference elementary step."""

    states = torch.as_tensor(initial_states, dtype=torch.float32, device=device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        result = masked_reference_free_step_torch(
            states,
            float(dt),
            dynamics_config,
            free_weight=float(rate),
            noise_weight=math.sqrt(max(float(rate), 0.0)),
            substeps=1,
            deterministic=False,
            collect_diagnostics=True,
            diagnostics_device=True,
        )
    if result.device_diagnostics is None:
        raise RuntimeError("forward preflight did not return device diagnostics")
    accumulator = _empty_accumulator()
    for key, value in result.device_diagnostics.items():
        if key in accumulator:
            accumulator[key] = float(value.detach().cpu())
    accumulator["step_count"] = 1.0
    summary = _accumulator_summary(accumulator, num_paths=states.shape[0])
    # This integrator does not expose separated free/noise transfer tensors.
    summary.pop("free_step_rms", None)
    summary.pop("noise_step_rms", None)
    return summary


@torch.no_grad()
def run_zero_residual_preflight(
    *,
    dynamics_config: DirectFluxMNISTConfig,
    diagnostic_config: ZeroResidualDiagnosticConfig,
    device: torch.device,
    rate_schedule: np.ndarray | None = None,
) -> ZeroResidualPreflightResult:
    """Forecast limiter activity from one worst-rate elementary step.

    This cheap preflight does not roll out the horizon or evaluate a terminal
    law.  Each replicate reuses one Dirichlet bank and one random draw within
    each kernel across every limiter setting and candidate temporal level.
    """

    horizon = (
        float(diagnostic_config.horizon)
        if diagnostic_config.horizon is not None
        else float(natural_horizon(dynamics_config))
    )
    if rate_schedule is None:
        rate_schedule = make_coupled_rate_schedule(
            diagnostic_config.sample_steps,
            horizon=horizon,
            tau_eff=diagnostic_config.tau_eff,
            time_change_mode=diagnostic_config.time_change_mode,
            ramp=diagnostic_config.rate_ramp,
            ramp_ratio=diagnostic_config.rate_ramp_ratio,
            rate_min=diagnostic_config.rate_min,
            rate_max=diagnostic_config.rate_max,
        )
    schedule = np.asarray(rate_schedule, dtype=np.float64).reshape(-1)
    if schedule.size != int(diagnostic_config.sample_steps):
        raise ValueError("rate_schedule length must equal sample_steps")
    if not np.isfinite(schedule).all() or np.any(schedule < 0.0):
        raise ValueError("rate_schedule must be finite and non-negative")
    effective_time = float(schedule.sum()) * float(horizon) / float(schedule.size)
    if not math.isfinite(effective_time) or effective_time <= 0.0:
        raise ValueError("rate_schedule must have a finite positive reference-rate integral")

    limiter_fractions = diagnostic_config.preflight_limiter_fractions or (
        float(dynamics_config.limiter_fraction),
    )
    candidate_levels = _preflight_candidate_levels(
        diagnostic_config.substep_levels,
        max_substeps=diagnostic_config.preflight_max_substeps,
    )
    alpha = float(edge_alpha_value(dynamics_config))
    dim = int(dynamics_config.grid_size) ** 2
    forecast_rate_index = int(np.argmax(schedule))
    forecast_rate = float(schedule[forecast_rate_index])
    raw_threshold = float(diagnostic_config.strict_raw_limiter_threshold)
    weighted_threshold = float(diagnostic_config.strict_weighted_limiter_threshold)
    metrics: list[dict[str, float | int | str]] = []
    initial_by_rep: dict[int, np.ndarray] = {}
    random_seed_by_rep: dict[int, int] = {}
    normal_by_rep: dict[int, Tensor] = {}
    for rep in range(int(diagnostic_config.preflight_reps)):
        state_seed = int(diagnostic_config.seed) + 100_003 * rep
        initial_by_rep[rep] = _sample_symmetric_dirichlet(
            diagnostic_config.preflight_paths,
            dim,
            alpha,
            np.random.default_rng(state_seed),
        )
        random_seed = int(diagnostic_config.seed) + 20_027 + 100_003 * rep
        random_seed_by_rep[rep] = random_seed
        try:
            generator = torch.Generator(device=device).manual_seed(random_seed)
        except TypeError:
            generator = torch.Generator(device=device.type).manual_seed(random_seed)
        normal_by_rep[rep] = torch.randn(
            (
                int(diagnostic_config.preflight_paths),
                2,
                int(dynamics_config.grid_size),
                int(dynamics_config.grid_size),
            ),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

    evaluated: list[int] = []
    stop_level: int | None = None
    eligible_at_stop: list[float] = []
    requested_finest = max(diagnostic_config.substep_levels)
    for level in candidate_levels:
        for rep in range(int(diagnostic_config.preflight_reps)):
            initial_states = initial_by_rep[rep]
            dt = float(horizon) / float(diagnostic_config.sample_steps * int(level))
            for limiter_fraction in limiter_fractions:
                candidate_dynamics = replace(
                    dynamics_config,
                    limiter_fraction=float(limiter_fraction),
                )
                direct_summary = _run_direct_preflight_step(
                    initial_states,
                    normal_by_rep[rep],
                    dynamics_config=candidate_dynamics,
                    rate=forecast_rate,
                    dt=dt,
                    device=device,
                )
                direct_row: dict[str, float | int | str] = {
                    "mode": "forecast-only",
                    "kernel": "direct-reverse",
                    "rep": int(rep),
                    "state_seed": int(diagnostic_config.seed) + 100_003 * rep,
                    "random_seed": random_seed_by_rep[rep],
                    "substeps": int(level),
                    "dt": dt,
                    "forecast_rate": forecast_rate,
                    "forecast_rate_index": forecast_rate_index,
                    "forecast_elementary_steps": 1,
                    "limiter_fraction_setting": float(limiter_fraction),
                    "raw_intervention_threshold": raw_threshold,
                    "weighted_intervention_threshold": weighted_threshold,
                    "intervention_thresholds_met": int(
                        _intervention_thresholds_met(
                            direct_summary,
                            raw_threshold=raw_threshold,
                            weighted_threshold=weighted_threshold,
                        )
                    ),
                }
                direct_row.update(direct_summary)
                metrics.append(direct_row)

                forward_summary = _run_forward_preflight_step(
                    initial_states,
                    dynamics_config=candidate_dynamics,
                    rate=forecast_rate,
                    dt=dt,
                    seed=random_seed_by_rep[rep],
                    device=device,
                )
                forward_row: dict[str, float | int | str] = {
                    "mode": "forecast-only",
                    "kernel": "forward-reference",
                    "rep": int(rep),
                    "state_seed": int(diagnostic_config.seed) + 100_003 * rep,
                    "random_seed": random_seed_by_rep[rep],
                    "substeps": int(level),
                    "dt": dt,
                    "forecast_rate": forecast_rate,
                    "forecast_rate_index": forecast_rate_index,
                    "forecast_elementary_steps": 1,
                    "limiter_fraction_setting": float(limiter_fraction),
                    "raw_intervention_threshold": raw_threshold,
                    "weighted_intervention_threshold": weighted_threshold,
                    "intervention_thresholds_met": int(
                        _intervention_thresholds_met(
                            forward_summary,
                            raw_threshold=raw_threshold,
                            weighted_threshold=weighted_threshold,
                        )
                    ),
                }
                forward_row.update(forward_summary)
                metrics.append(forward_row)
        evaluated.append(int(level))
        if int(level) < int(requested_finest):
            continue
        eligible = [
            float(fraction)
            for fraction in limiter_fractions
            if _preflight_fraction_meets_at_level(
                metrics,
                limiter_fraction=float(fraction),
                substeps=int(level),
                reps=diagnostic_config.preflight_reps,
            )
        ]
        if eligible:
            stop_level = int(level)
            eligible_at_stop = eligible
            break

    first_met_by_fraction: dict[str, int | None] = {}
    for fraction in limiter_fractions:
        first_met_by_fraction[f"{float(fraction):.12g}"] = next(
            (
                int(level)
                for level in evaluated
                if _preflight_fraction_meets_at_level(
                    metrics,
                    limiter_fraction=float(fraction),
                    substeps=int(level),
                    reps=diagnostic_config.preflight_reps,
                )
            ),
            None,
        )

    summary: dict[str, object] = {
        "forecast_only": 1,
        "decision_scope": "intervention forecast only; no law or acceptance decision",
        "requested_substeps": list(diagnostic_config.substep_levels),
        "evaluated_substeps": list(evaluated),
        "auto_doubling_used": int(max(evaluated) > requested_finest),
        "configured_substep_cap": int(diagnostic_config.preflight_max_substeps),
        "cap_reached": int(max(evaluated) >= diagnostic_config.preflight_max_substeps),
        "any_configuration_met_thresholds": int(stop_level is not None),
        "first_joint_threshold_substeps": stop_level,
        "eligible_limiter_fractions_at_stop": eligible_at_stop,
        "first_joint_threshold_substeps_by_limiter_fraction": first_met_by_fraction,
        "kernels_evaluated": ["direct-reverse", "forward-reference"],
        "common_state_randomness_within_replicate": 1,
        "forecast_elementary_steps_per_configuration": 1,
        "forecast_schedule_rate": forecast_rate,
        "forecast_schedule_rate_index": forecast_rate_index,
        "forecast_dt_formula": "horizon / (sample_steps * substeps)",
        "raw_intervention_threshold": raw_threshold,
        "weighted_intervention_threshold": weighted_threshold,
        "reference_rate_integral": float(effective_time),
    }
    return ZeroResidualPreflightResult(
        config=diagnostic_config,
        dynamics_config=dynamics_config,
        rate_schedule=schedule,
        reference_rate_integral=effective_time,
        levels_evaluated=tuple(evaluated),
        limiter_fractions=tuple(float(value) for value in limiter_fractions),
        metrics=metrics,
        summary=summary,
    )


def run_zero_residual_diagnostic(
    *,
    dynamics_config: DirectFluxMNISTConfig,
    diagnostic_config: ZeroResidualDiagnosticConfig,
    device: torch.device,
    rate_schedule: np.ndarray | None = None,
    progress_callback: Callable[[Mapping[str, int | str]], None] | None = None,
) -> ZeroResidualDiagnosticResult:
    """Run coupled temporal refinements of the production zero-residual kernel."""

    horizon = (
        float(diagnostic_config.horizon)
        if diagnostic_config.horizon is not None
        else float(natural_horizon(dynamics_config))
    )
    if rate_schedule is None:
        rate_schedule = make_coupled_rate_schedule(
            diagnostic_config.sample_steps,
            horizon=horizon,
            tau_eff=diagnostic_config.tau_eff,
            time_change_mode=diagnostic_config.time_change_mode,
            ramp=diagnostic_config.rate_ramp,
            ramp_ratio=diagnostic_config.rate_ramp_ratio,
            rate_min=diagnostic_config.rate_min,
            rate_max=diagnostic_config.rate_max,
        )
    rate_schedule = np.asarray(rate_schedule, dtype=np.float64).reshape(-1)
    if rate_schedule.size != int(diagnostic_config.sample_steps):
        raise ValueError("rate_schedule length must equal sample_steps")
    if not np.isfinite(rate_schedule).all() or np.any(rate_schedule < 0.0):
        raise ValueError("rate_schedule must be finite and non-negative")
    effective_time = float(rate_schedule.sum()) * float(horizon) / float(rate_schedule.size)
    if not math.isfinite(effective_time) or effective_time <= 0.0:
        raise ValueError("rate_schedule must have a finite positive reference-rate integral")

    alpha = float(edge_alpha_value(dynamics_config))
    dim = int(dynamics_config.grid_size) ** 2
    rng = np.random.default_rng(int(diagnostic_config.seed))
    initial_states = _sample_symmetric_dirichlet(
        diagnostic_config.num_paths,
        dim,
        alpha,
        rng,
    )
    reference_states = _sample_symmetric_dirichlet(
        diagnostic_config.num_paths,
        dim,
        alpha,
        rng,
    )
    calibration = _calibrate_dirichlet_null(
        dynamics_config,
        num_paths=diagnostic_config.num_paths,
        alpha=alpha,
        seed=int(diagnostic_config.seed) + 1009,
        reps=diagnostic_config.calibration_reps,
        quantile_multiplier=diagnostic_config.stationarity_quantile_multiplier,
        mmd_multiplier=diagnostic_config.stationarity_mmd_multiplier,
        quantile_floor=diagnostic_config.stationarity_quantile_floor,
        mmd_floor=diagnostic_config.stationarity_mmd_floor,
    )
    states_by_level, kernel_summaries = _run_coupled_direct_levels(
        initial_states,
        dynamics_config=dynamics_config,
        rate_schedule=rate_schedule,
        horizon=horizon,
        substep_levels=diagnostic_config.substep_levels,
        seed=int(diagnostic_config.seed) + 2027,
        device=device,
        progress_callback=progress_callback,
    )
    levels: list[dict[str, float | int | str]] = []
    previous_states: np.ndarray | None = None
    for level in diagnostic_config.substep_levels:
        states = states_by_level[level]
        row: dict[str, float | int | str] = {
            "kernel": "direct-doob-zero-residual",
            "substeps": int(level),
            "dt": float(horizon) / float(diagnostic_config.sample_steps * level),
        }
        row.update(kernel_summaries[level])
        row.update(
            _evaluate_level(
                states,
                reference_states,
                initial_states,
                dynamics_config=dynamics_config,
                dirichlet_alpha=alpha,
                calibration=calibration,
                entropy_error_floor=diagnostic_config.entropy_error_floor,
                second_moment_error_floor=diagnostic_config.second_moment_error_floor,
            )
        )
        if previous_states is None:
            row["coupled_refinement_rms"] = float("nan")
        else:
            row["coupled_refinement_rms"] = float(
                np.sqrt(np.mean(np.square(states - previous_states), dtype=np.float64))
            )
        levels.append(row)
        previous_states = states

    gate = evaluate_refinement_gate(
        levels,
        max_simplex_mass_error=diagnostic_config.max_simplex_mass_error,
        max_floor_correction_l1=diagnostic_config.max_floor_correction_l1,
        max_renorm_correction_l1=diagnostic_config.max_renorm_correction_l1,
        max_standardized_moment_error=diagnostic_config.max_standardized_moment_error,
        refinement_contraction=diagnostic_config.refinement_contraction,
        refinement_absolute_tolerance=diagnostic_config.refinement_absolute_tolerance,
        intervention_nonincrease_tolerance=diagnostic_config.intervention_nonincrease_tolerance,
        strict_raw_limiter_threshold=diagnostic_config.strict_raw_limiter_threshold,
        strict_weighted_limiter_threshold=diagnostic_config.strict_weighted_limiter_threshold,
    )

    forward_control_levels: list[dict[str, float | int | str]] = []
    if diagnostic_config.include_forward_control:
        for level in diagnostic_config.substep_levels:
            control_states, control_summary = _run_forward_control_level(
                initial_states,
                dynamics_config=dynamics_config,
                rate_schedule=rate_schedule,
                horizon=horizon,
                substeps=level,
                seed=int(diagnostic_config.seed) + 4001 + int(level),
                device=device,
                diagnostics_device=True,
                progress_callback=progress_callback,
            )
            row = {
                "kernel": "forward-reference-control",
                "substeps": int(level),
                "dt": float(horizon) / float(diagnostic_config.sample_steps * level),
                "random_seed": int(diagnostic_config.seed) + 4001 + int(level),
            }
            row.update(control_summary)
            row.update(
                _evaluate_level(
                    control_states,
                    reference_states,
                    initial_states,
                    dynamics_config=dynamics_config,
                    dirichlet_alpha=alpha,
                    calibration=calibration,
                    entropy_error_floor=diagnostic_config.entropy_error_floor,
                    second_moment_error_floor=diagnostic_config.second_moment_error_floor,
                )
            )
            forward_control_levels.append(row)

    forward_gate = evaluate_forward_reference_control(
        forward_control_levels,
        max_simplex_mass_error=diagnostic_config.max_simplex_mass_error,
        max_floor_correction_l1=diagnostic_config.max_floor_correction_l1,
        max_renorm_correction_l1=diagnostic_config.max_renorm_correction_l1,
        max_standardized_moment_error=diagnostic_config.max_standardized_moment_error,
        intervention_nonincrease_tolerance=diagnostic_config.intervention_nonincrease_tolerance,
        strict_raw_limiter_threshold=diagnostic_config.strict_raw_limiter_threshold,
        strict_weighted_limiter_threshold=diagnostic_config.strict_weighted_limiter_threshold,
    )

    gate.update(
        {
            "expected_symmetric_dirichlet_entropy": float(
                expected_symmetric_dirichlet_entropy(dynamics_config)
            ),
            "reference_rate_integral": float(effective_time),
            "dirichlet_alpha": float(alpha),
            **forward_gate,
        }
    )
    return ZeroResidualDiagnosticResult(
        config=diagnostic_config,
        dynamics_config=dynamics_config,
        rate_schedule=rate_schedule,
        reference_rate_integral=effective_time,
        dirichlet_alpha=alpha,
        calibration=calibration,
        levels=levels,
        gate=gate,
        initial_states=initial_states,
        states_by_level=states_by_level,
        reference_states=reference_states,
        forward_control_levels=forward_control_levels,
    )


def run_zero_residual_refinement(
    *,
    dynamics_config: DirectFluxMNISTConfig,
    num_paths: int,
    sample_steps: int,
    substeps: Sequence[int],
    horizon: float,
    rate_schedule: np.ndarray,
    seed: int,
    device: torch.device,
    calibration_reps: int = 3,
) -> dict[str, object]:
    """Compatibility wrapper used by focused tests and notebook experiments."""

    schedule = np.asarray(rate_schedule, dtype=np.float64).reshape(-1)
    tau_eff = float(schedule.sum()) * float(horizon) / float(schedule.size)
    config = ZeroResidualDiagnosticConfig(
        num_paths=int(num_paths),
        sample_steps=int(sample_steps),
        substep_levels=parse_substep_levels(substeps),
        horizon=float(horizon),
        tau_eff=float(tau_eff),
        seed=int(seed),
        calibration_reps=int(calibration_reps),
    )
    result = run_zero_residual_diagnostic(
        dynamics_config=dynamics_config,
        diagnostic_config=config,
        device=device,
        rate_schedule=schedule,
    )
    return result.to_dict(include_arrays=True)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


_RUN_SCHEMA_VERSION = 2
_RUN_ALGORITHM_VERSION = 2
_IMPLEMENTATION_SOURCE_FILES = (
    "diag_d0_zero_residual.py",
    "diag_forward_noising.py",
    "experiment12_d0.py",
    "eulerian_flux_mnist.py",
)


def _implementation_source_hashes() -> dict[str, str]:
    """Fingerprint the production runner and both kernels it exercises."""

    source_root = Path(__file__).resolve().parent
    hashes: dict[str, str] = {}
    for filename in _IMPLEMENTATION_SOURCE_FILES:
        hashes[filename] = hashlib.sha256((source_root / filename).read_bytes()).hexdigest()
    return hashes


def _runtime_identity(device: torch.device) -> dict[str, str | None]:
    """Return runtime details whose changes make mixed-seed resume unsafe."""

    device_name = "cpu"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    return {
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "device_type": str(device.type),
        "device_name": str(device_name),
    }


def _atomic_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = _atomic_temp_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_jsonable)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, float | int | str]],
) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = _atomic_temp_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = _atomic_temp_path(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_plot(path: Path, writer: Callable[[Path], None]) -> None:
    temporary = _atomic_temp_path(path)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scientific_manifest(
    *,
    dynamics_config: DirectFluxMNISTConfig,
    diagnostic_config: ZeroResidualDiagnosticConfig,
    seeds: Sequence[int],
    forward_control_seeds: Sequence[int],
    rate_schedule: np.ndarray,
    mode: str,
    device: torch.device,
) -> dict[str, object]:
    diagnostic = asdict(diagnostic_config)
    if mode == "diagnostic":
        diagnostic.pop("seed", None)
        diagnostic.pop("include_forward_control", None)
        for key in tuple(diagnostic):
            if key.startswith("preflight_"):
                diagnostic.pop(key)
    return {
        "schema_version": _RUN_SCHEMA_VERSION,
        "algorithm": "d0-zero-residual-production-gate",
        "algorithm_version": _RUN_ALGORITHM_VERSION,
        "implementation_sha256": _implementation_source_hashes(),
        "runtime": _runtime_identity(device),
        "mode": str(mode),
        "dynamics_config": asdict(dynamics_config),
        "diagnostic_config": diagnostic,
        "seeds": sorted(int(seed) for seed in seeds),
        "forward_control_seeds": sorted(int(seed) for seed in forward_control_seeds),
        "resolved_horizon": float(
            diagnostic_config.horizon
            if diagnostic_config.horizon is not None
            else natural_horizon(dynamics_config)
        ),
        "rate_schedule": np.asarray(rate_schedule, dtype=np.float64).tolist(),
        "state_dtype": "float32",
    }


def _config_fingerprint(manifest: Mapping[str, object]) -> str:
    canonical = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        default=_jsonable,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_zero_residual_preflight(
    result: ZeroResidualPreflightResult,
    run_dir: str | Path,
) -> dict[str, str]:
    """Save forecast-only summary, per-probe metrics, and refinement plot."""

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "preflight_summary.json"
    metrics_path = root / "preflight_metrics.csv"
    plot_path = root / "preflight_refinement.png"
    _atomic_write_json(summary_path, result.to_dict())
    _atomic_write_csv(metrics_path, result.metrics)
    _atomic_save_plot(plot_path, lambda target: _save_preflight_plot(result, target))
    return {
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "plot_path": str(plot_path),
    }


def _save_preflight_plot(result: ZeroResidualPreflightResult, path: Path) -> None:
    """Plot worst-replicate raw and weighted intervention forecasts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    floor = 1e-12
    for kernel in ("direct-reverse", "forward-reference"):
        for limiter_fraction in result.limiter_fractions:
            x_values: list[int] = []
            raw_values: list[float] = []
            weighted_values: list[float] = []
            for level in result.levels_evaluated:
                rows = [
                    row
                    for row in result.metrics
                    if str(row["kernel"]) == kernel
                    and int(row["substeps"]) == int(level)
                    and math.isclose(
                        float(row["limiter_fraction_setting"]),
                        float(limiter_fraction),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ]
                if not rows:
                    continue
                x_values.append(int(level))
                raw_values.append(max(float(row["limiter_fraction"]) for row in rows))
                weighted_values.append(
                    max(
                        max(
                            float(row["mobility_weighted_limiter_fraction"]),
                            float(row["noise_energy_weighted_limiter_fraction"]),
                        )
                        for row in rows
                    )
                )
            label = f"{kernel}, limiter={float(limiter_fraction):.4g}"
            axes[0].plot(
                x_values,
                np.maximum(raw_values, floor),
                marker="o",
                label=label,
            )
            axes[1].plot(
                x_values,
                np.maximum(weighted_values, floor),
                marker="o",
                label=label,
            )

    thresholds = (
        float(result.config.strict_raw_limiter_threshold),
        float(result.config.strict_weighted_limiter_threshold),
    )
    titles = ("Raw limiter activity", "Worst weighted limiter activity")
    for axis, threshold, title in zip(axes, thresholds, titles):
        axis.axhline(
            max(threshold, floor),
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="forecast threshold",
        )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("candidate reference substeps")
        axis.set_ylabel("worst fraction across replicates")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("D0 one-step intervention preflight (forecast only)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_zero_residual_diagnostic(
    result: ZeroResidualDiagnosticResult,
    run_dir: str | Path,
) -> dict[str, str]:
    """Save the gate record, per-level rows, and the finest state bank."""

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    metrics_path = root / "refinement_metrics.csv"
    states_path = root / "states_finest.npz"
    plot_path = root / "stationarity_refinement.png"
    _atomic_write_json(summary_path, result.to_dict(include_arrays=False))
    rows = [*result.levels, *result.forward_control_levels]
    _atomic_write_csv(metrics_path, rows)
    _atomic_save_npz(
        states_path,
        initial_states=result.initial_states.astype(np.float32),
        terminal_states=result.states_finest.astype(np.float32),
        reference_states=result.reference_states.astype(np.float32),
        substep_levels=np.asarray(result.config.substep_levels, dtype=np.int64),
        rate_schedule=result.rate_schedule.astype(np.float64),
        horizon=np.asarray([
            result.config.horizon
            if result.config.horizon is not None
            else natural_horizon(result.dynamics_config)
        ], dtype=np.float64),
        dirichlet_alpha=np.asarray([result.dirichlet_alpha], dtype=np.float64),
    )
    _atomic_save_plot(plot_path, lambda target: _save_refinement_plot(result, target))
    return {
        "summary_path": str(summary_path),
        "metrics_path": str(metrics_path),
        "states_path": str(states_path),
        "plot_path": str(plot_path),
    }


def _save_refinement_plot(result: ZeroResidualDiagnosticResult, path: Path) -> None:
    """Save normalized law errors and intervention fractions versus refinement."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = result.levels
    substeps = np.asarray([int(row["substeps"]) for row in levels], dtype=np.float64)
    q_ratio = np.asarray([float(row["stationarity_quantile_ratio"]) for row in levels])
    mmd_ratio = np.asarray([float(row["stationarity_feature_mmd_ratio"]) for row in levels])
    entropy_ratio = np.asarray(
        [
            max(
                float(row["entropy_standard_error_units"]),
                float(row["entropy_analytic_standard_error_units"]),
                float(row["entropy_paired_drift_standard_error_units"]),
            )
            / max(float(result.config.max_standardized_moment_error), 1e-30)
            for row in levels
        ]
    )
    moment_ratio = np.asarray(
        [
            max(
                float(row["second_moment_standard_error_units"]),
                float(row["second_moment_analytic_standard_error_units"]),
                float(row["second_moment_paired_drift_standard_error_units"]),
            )
            / max(float(result.config.max_standardized_moment_error), 1e-30)
            for row in levels
        ]
    )
    raw = np.asarray([float(row["limiter_fraction"]) for row in levels])
    mobility = np.asarray(
        [float(row["mobility_weighted_limiter_fraction"]) for row in levels]
    )
    noise = np.asarray(
        [float(row["noise_energy_weighted_limiter_fraction"]) for row in levels]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    left, right = axes
    positive_floor = 1e-12
    left.plot(substeps, np.maximum(q_ratio, positive_floor), marker="o", label="quantile / null gate")
    left.plot(substeps, np.maximum(mmd_ratio, positive_floor), marker="o", label="feature MMD / null gate")
    left.plot(substeps, np.maximum(entropy_ratio, positive_floor), marker="o", label="worst entropy / tolerance")
    left.plot(substeps, np.maximum(moment_ratio, positive_floor), marker="o", label="worst second moment / tolerance")
    left.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="pass boundary")
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xlabel("reference substeps")
    left.set_ylabel("normalized discrepancy")
    left.set_title("Dirichlet-law diagnostics")
    left.grid(True, which="both", alpha=0.25)
    left.legend(fontsize=8)

    right.plot(substeps, np.maximum(raw, positive_floor), marker="o", label="raw")
    right.plot(substeps, np.maximum(mobility, positive_floor), marker="o", label="mobility weighted")
    right.plot(substeps, np.maximum(noise, positive_floor), marker="o", label="noise weighted")
    right.axhline(
        float(result.config.strict_raw_limiter_threshold),
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
        label="strict raw threshold",
    )
    right.axhline(
        float(result.config.strict_weighted_limiter_threshold),
        color="tab:orange",
        linestyle="--",
        linewidth=1.0,
        label="strict weighted threshold",
    )
    right.set_xscale("log", base=2)
    right.set_yscale("log")
    right.set_xlabel("reference substeps")
    right.set_ylabel("intervention fraction")
    right.set_title("Limiter refinement")
    right.grid(True, which="both", alpha=0.25)
    right.legend(fontsize=8)
    fig.suptitle("D0 zero-residual production-kernel gate")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the production D0 direct reverse reference at zero learned residual."
    )
    parser.add_argument("--runs-root", type=str, default="runs/experiment12_d0_zero_residual")
    parser.add_argument("--run-name", type=str, default="zero-residual")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=260715)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated replicate seeds. Each replicate gets its own frozen null calibration.",
    )
    parser.add_argument("--num-paths", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--substeps", type=str, default="2,4,8")
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--tau-eff", type=float, default=2e-4)
    parser.add_argument("--time-change-mode", choices=("integral", "rate"), default="integral")
    parser.add_argument("--rate-ramp", choices=("none", "geometric"), default="none")
    parser.add_argument("--rate-ramp-ratio", type=float, default=1.0)
    parser.add_argument("--rate-min", type=float, default=None)
    parser.add_argument("--rate-max", type=float, default=None)
    parser.add_argument("--edge-alpha-mode", choices=("legacy", "grid", "alpha_eff"), default="alpha_eff")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha-eff", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--mass-floor", type=float, default=1e-7)
    parser.add_argument("--limiter-fraction", type=float, default=0.25)
    parser.add_argument("--calibration-reps", type=int, default=8)
    parser.add_argument("--stationarity-quantile-multiplier", type=float, default=3.0)
    parser.add_argument("--stationarity-mmd-multiplier", type=float, default=3.0)
    parser.add_argument("--stationarity-quantile-floor", type=float, default=0.02)
    parser.add_argument("--stationarity-mmd-floor", type=float, default=1e-4)
    parser.add_argument("--entropy-error-floor", type=float, default=2e-3)
    parser.add_argument("--second-moment-error-floor", type=float, default=2e-4)
    parser.add_argument("--include-forward-control", action="store_true")
    parser.add_argument(
        "--forward-control-seeds",
        type=str,
        default="",
        help="Comma-separated subset of diagnostic seeds that run the forward reference control.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only the one-step worst-rate intervention forecast.",
    )
    parser.add_argument("--preflight-paths", type=int, default=16)
    parser.add_argument("--preflight-reps", type=int, default=1)
    parser.add_argument(
        "--preflight-limiter-fractions",
        type=str,
        default="",
        help="Comma-separated limiter settings; defaults to --limiter-fraction.",
    )
    parser.add_argument("--preflight-max-substeps", type=int, default=64)
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default="",
        help="Resume a compatible run directory, skipping seeds with complete atomic artifacts.",
    )
    parser.add_argument(
        "--require-gate",
        choices=("none", "fixed", "strict", "training-ready"),
        default="none",
        help="Exit nonzero after saving all artifacts when the requested evidence gate is unmet.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=8,
        help="Print and persist progress every N outer steps; zero disables progress updates.",
    )
    parser.add_argument("--max-simplex-mass-error", type=float, default=2e-6)
    parser.add_argument("--max-floor-correction-l1", type=float, default=1e-8)
    parser.add_argument("--max-renorm-correction-l1", type=float, default=1e-6)
    parser.add_argument("--max-standardized-moment-error", type=float, default=4.0)
    parser.add_argument("--refinement-contraction", type=float, default=0.90)
    parser.add_argument("--refinement-absolute-tolerance", type=float, default=1e-7)
    parser.add_argument("--intervention-nonincrease-tolerance", type=float, default=0.05)
    parser.add_argument("--strict-raw-limiter-threshold", type=float, default=5e-3)
    parser.add_argument("--strict-weighted-limiter-threshold", type=float, default=5e-4)
    return parser


def _parse_seed_csv(value: str, *, fallback: int | None = None) -> list[int]:
    seeds = [int(piece.strip()) for piece in str(value).split(",") if piece.strip()]
    if not seeds and fallback is not None:
        seeds = [int(fallback)]
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("diagnostic seeds must be unique")
    return seeds


def _resolve_forward_control_seeds(
    diagnostic_seeds: Sequence[int],
    *,
    include_all: bool,
    explicit: str,
) -> list[int]:
    if str(explicit).strip():
        selected = _parse_seed_csv(str(explicit))
        missing = sorted(set(selected) - set(int(seed) for seed in diagnostic_seeds))
        if missing:
            raise ValueError(
                "forward-control seeds must be a subset of diagnostic seeds; missing "
                + ",".join(str(seed) for seed in missing)
            )
        return selected
    return [int(seed) for seed in diagnostic_seeds] if include_all else []


def _replicate_paths(run_dir: Path) -> dict[str, str]:
    return {
        "summary_path": str(run_dir / "summary.json"),
        "metrics_path": str(run_dir / "refinement_metrics.csv"),
        "states_path": str(run_dir / "states_finest.npz"),
        "plot_path": str(run_dir / "stationarity_refinement.png"),
    }


def _load_completed_seed_record(
    *,
    seed: int,
    replicate_dir: Path,
    completed_seeds: set[int],
) -> dict[str, object] | None:
    paths = _replicate_paths(replicate_dir)
    if int(seed) not in completed_seeds or not all(Path(path).exists() for path in paths.values()):
        return None
    with Path(paths["summary_path"]).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if int(summary.get("config", {}).get("seed", -1)) != int(seed):
        return None
    return {"seed": int(seed), "gate": dict(summary["gate"]), "paths": paths, "resumed": 1}


def _required_gate_pass(aggregate: Mapping[str, object], required_gate: str) -> bool:
    required = str(required_gate)
    if required == "none":
        return True
    if required == "fixed":
        return bool(int(aggregate["fixed_grid_stationarity_pass"]))
    if required == "strict":
        return bool(int(aggregate["strict_h_transform_limit_supported"]))
    if required == "training-ready":
        return bool(int(aggregate["training_ready"]))
    raise ValueError(f"unknown required gate: {required}")


def _make_progress_callback(
    *,
    seed: int,
    every: int,
    status: dict[str, object],
    status_path: Path,
) -> Callable[[Mapping[str, int | str]], None] | None:
    if int(every) <= 0:
        return None
    phase_started = time.perf_counter()
    active_token: tuple[str, int | None] | None = None

    def report(event: Mapping[str, int | str]) -> None:
        nonlocal active_token, phase_started
        completed_outer = int(event["outer_completed"])
        total_outer = int(event["outer_total"])
        phase = str(event["phase"])
        level = event.get("level")
        token = (phase, None if level is None else int(level))
        if token != active_token:
            active_token = token
            phase_started = time.perf_counter()
        if completed_outer % int(every) != 0 and completed_outer != total_outer:
            return
        elapsed = max(time.perf_counter() - phase_started, 1e-9)
        fraction = completed_outer / max(total_outer, 1)
        eta = elapsed * (1.0 - fraction) / max(fraction, 1e-12)
        status.update(
            {
                "current_seed": int(seed),
                "current_phase": phase,
                "current_level": None if level is None else int(level),
                "current_outer_completed": completed_outer,
                "current_outer_total": total_outer,
                "elapsed_seconds_current_phase": float(elapsed),
                "eta_seconds_current_phase": float(eta),
                "updated_at": datetime.now().astimezone().isoformat(),
            }
        )
        _atomic_write_json(status_path, status)
        level_text = "" if level is None else f" level={int(level)}"
        print(
            f"[d0-gate] seed={seed} phase={phase}{level_text} "
            f"outer={completed_outer}/{total_outer} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.progress_every) < 0:
        raise ValueError("--progress-every must be non-negative")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n = int(args.grid_size)
    dynamics = DirectFluxMNISTConfig(
        grid_size=n,
        num_steps=int(args.sample_steps),
        source_lowfreq_size=min(7, n),
        source_blur_sigma=0.0,
        ot_lowres_size=min(7, n),
        ot_blur_sigma=0.0,
        edge_alpha_mode=str(args.edge_alpha_mode),
        alpha=float(args.alpha),
        alpha_eff=float(args.alpha_eff),
        beta=float(args.beta),
        mass_floor=float(args.mass_floor),
        limiter_fraction=float(args.limiter_fraction),
    )
    config = ZeroResidualDiagnosticConfig(
        num_paths=int(args.num_paths),
        sample_steps=int(args.sample_steps),
        substep_levels=parse_substep_levels(args.substeps),
        horizon=args.horizon,
        tau_eff=float(args.tau_eff),
        time_change_mode=str(args.time_change_mode),
        rate_ramp=str(args.rate_ramp),
        rate_ramp_ratio=float(args.rate_ramp_ratio),
        rate_min=args.rate_min,
        rate_max=args.rate_max,
        seed=int(args.seed),
        calibration_reps=int(args.calibration_reps),
        stationarity_quantile_multiplier=float(args.stationarity_quantile_multiplier),
        stationarity_mmd_multiplier=float(args.stationarity_mmd_multiplier),
        stationarity_quantile_floor=float(args.stationarity_quantile_floor),
        stationarity_mmd_floor=float(args.stationarity_mmd_floor),
        entropy_error_floor=float(args.entropy_error_floor),
        second_moment_error_floor=float(args.second_moment_error_floor),
        max_simplex_mass_error=float(args.max_simplex_mass_error),
        max_floor_correction_l1=float(args.max_floor_correction_l1),
        max_renorm_correction_l1=float(args.max_renorm_correction_l1),
        max_standardized_moment_error=float(args.max_standardized_moment_error),
        refinement_contraction=float(args.refinement_contraction),
        refinement_absolute_tolerance=float(args.refinement_absolute_tolerance),
        intervention_nonincrease_tolerance=float(args.intervention_nonincrease_tolerance),
        strict_raw_limiter_threshold=float(args.strict_raw_limiter_threshold),
        strict_weighted_limiter_threshold=float(args.strict_weighted_limiter_threshold),
        include_forward_control=bool(args.include_forward_control),
        preflight_only=bool(args.preflight_only),
        preflight_paths=int(args.preflight_paths),
        preflight_reps=int(args.preflight_reps),
        preflight_limiter_fractions=parse_limiter_fractions(args.preflight_limiter_fractions),
        preflight_max_substeps=int(args.preflight_max_substeps),
    )
    horizon = float(config.horizon if config.horizon is not None else natural_horizon(dynamics))
    rate_schedule = make_coupled_rate_schedule(
        config.sample_steps,
        horizon=horizon,
        tau_eff=config.tau_eff,
        time_change_mode=config.time_change_mode,
        ramp=config.rate_ramp,
        ramp_ratio=config.rate_ramp_ratio,
        rate_min=config.rate_min,
        rate_max=config.rate_max,
    )
    if config.preflight_only:
        if str(args.seeds).strip():
            raise ValueError("--seeds cannot be combined with --preflight-only; use --preflight-reps")
        if str(args.forward_control_seeds).strip() or bool(args.include_forward_control):
            raise ValueError("preflight already probes both kernels; forward-control seed options are invalid")
        if str(args.require_gate) != "none":
            raise ValueError("--preflight-only cannot satisfy a scientific gate; use --require-gate none")
        seeds = [int(config.seed)]
        forward_control_seeds: list[int] = []
        mode = "preflight"
    else:
        seeds = _parse_seed_csv(str(args.seeds), fallback=int(args.seed))
        forward_control_seeds = _resolve_forward_control_seeds(
            seeds,
            include_all=bool(args.include_forward_control),
            explicit=str(args.forward_control_seeds),
        )
        mode = "diagnostic"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if str(args.resume_run_dir).strip():
        run_dir = Path(args.resume_run_dir)
        if not run_dir.is_dir():
            raise ValueError(f"resume run directory does not exist: {run_dir}")
    else:
        run_dir = Path(args.runs_root) / f"{timestamp}_{args.run_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
    manifest = _scientific_manifest(
        dynamics_config=dynamics,
        diagnostic_config=config,
        seeds=seeds,
        forward_control_seeds=forward_control_seeds,
        rate_schedule=rate_schedule,
        mode=mode,
        device=device,
    )
    fingerprint = _config_fingerprint(manifest)
    run_config_path = run_dir / "run_config.json"
    status_path = run_dir / "run_status.json"
    now = datetime.now().astimezone().isoformat()
    if str(args.resume_run_dir).strip():
        if not run_config_path.exists():
            raise ValueError("resume directory has no run_config.json and is not safely resumable")
        with run_config_path.open("r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        if str(previous_manifest.get("config_fingerprint", "")) != fingerprint:
            raise ValueError("resume configuration fingerprint mismatch")
        status: dict[str, object] = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else {}
        )
        status["attempt_count"] = int(status.get("attempt_count", 1)) + 1
        status["resumed_at"] = now
    else:
        _atomic_write_json(
            run_config_path,
            {**manifest, "config_fingerprint": fingerprint},
        )
        status = {
            "schema_version": _RUN_SCHEMA_VERSION,
            "config_fingerprint": fingerprint,
            "mode": mode,
            "started_at": now,
            "attempt_count": 1,
            "completed_seeds": [],
        }
    status.update(
        {
            "status": "running",
            "outcome": "pending",
            "updated_at": now,
            "required_gate": str(args.require_gate),
            "requested_seeds": [int(seed) for seed in seeds],
            "requested_forward_control_seeds": [int(seed) for seed in forward_control_seeds],
            "run_dir": str(run_dir),
        }
    )
    _atomic_write_json(status_path, status)

    try:
        if config.preflight_only:
            result = run_zero_residual_preflight(
                dynamics_config=dynamics,
                diagnostic_config=config,
                device=device,
                rate_schedule=rate_schedule,
            )
            paths = save_zero_residual_preflight(result, run_dir)
            output = {
                "mode": "forecast-only-intervention-preflight",
                "summary": result.summary,
                "paths": paths,
                "run_dir": str(run_dir),
                "config_fingerprint": fingerprint,
            }
            status.update(
                {
                    "status": "complete",
                    "outcome": "preflight_complete",
                    "scientific_gate_evaluated": 0,
                    "artifact_paths": paths,
                    "completed_at": datetime.now().astimezone().isoformat(),
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
            )
            _atomic_write_json(status_path, status)
            print(json.dumps(output, indent=2, default=_jsonable))
            return

        records: list[dict[str, object]] = []
        completed_seeds = {int(seed) for seed in status.get("completed_seeds", [])}
        for seed in seeds:
            replicate_dir = run_dir if len(seeds) == 1 else run_dir / f"seed-{seed}"
            existing = _load_completed_seed_record(
                seed=int(seed),
                replicate_dir=replicate_dir,
                completed_seeds=completed_seeds,
            )
            if existing is not None:
                records.append(existing)
                continue
            include_forward = int(seed) in set(forward_control_seeds)
            replicate_config = replace(
                config,
                seed=int(seed),
                include_forward_control=bool(include_forward),
            )
            status.update(
                {
                    "current_seed": int(seed),
                    "current_phase": "starting",
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
            )
            _atomic_write_json(status_path, status)
            progress = _make_progress_callback(
                seed=int(seed),
                every=int(args.progress_every),
                status=status,
                status_path=status_path,
            )
            result = run_zero_residual_diagnostic(
                dynamics_config=dynamics,
                diagnostic_config=replicate_config,
                device=device,
                rate_schedule=rate_schedule,
                progress_callback=progress,
            )
            paths = save_zero_residual_diagnostic(result, replicate_dir)
            records.append(
                {
                    "seed": int(seed),
                    "forward_control_requested": int(include_forward),
                    "gate": result.gate,
                    "paths": paths,
                    "resumed": 0,
                }
            )
            completed_seeds.add(int(seed))
            status.update(
                {
                    "completed_seeds": sorted(completed_seeds),
                    "current_seed": None,
                    "current_phase": "seed-complete",
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
            )
            _atomic_write_json(status_path, status)

        fixed_pass = all(
            int(record["gate"]["fixed_grid_stationarity_pass"]) for record in records
        )
        strict_pass = all(
            int(record["gate"]["strict_h_transform_limit_supported"]) for record in records
        )
        forward_records = [
            record for record in records if int(record["seed"]) in set(forward_control_seeds)
        ]
        forward_evaluated = bool(forward_records) and all(
            int(record["gate"].get("forward_reference_control_evaluated", 0))
            for record in forward_records
        )
        forward_pass = bool(forward_evaluated) and all(
            int(record["gate"].get("forward_reference_control_pass", 0))
            for record in forward_records
        )
        aggregate_path = run_dir / "aggregate_summary.json"
        aggregate: dict[str, object] = {
            "seeds": [int(seed) for seed in seeds],
            "forward_control_seeds": [int(seed) for seed in forward_control_seeds],
            "replicates": records,
            "fixed_grid_stationarity_pass": int(fixed_pass),
            "strict_h_transform_limit_supported": int(strict_pass),
            "forward_reference_control_evaluated": int(forward_evaluated),
            "forward_reference_control_pass": int(forward_pass),
            "training_ready": int(strict_pass and forward_evaluated and forward_pass),
            "claim_scope": "fixed-grid temporal refinement across independent seeds",
            "aggregate_summary_path": str(aggregate_path),
            "run_dir": str(run_dir),
            "config_fingerprint": fingerprint,
        }
        required_pass = _required_gate_pass(aggregate, str(args.require_gate))
        aggregate["required_gate"] = str(args.require_gate)
        aggregate["required_gate_pass"] = int(required_pass)
        _atomic_write_json(aggregate_path, aggregate)
        status.update(
            {
                "status": "complete",
                "outcome": "gate_passed" if required_pass else "gate_failed",
                "required_gate_pass": int(required_pass),
                "aggregate_summary_path": str(aggregate_path),
                "completed_at": datetime.now().astimezone().isoformat(),
                "updated_at": datetime.now().astimezone().isoformat(),
                "current_seed": None,
                "current_phase": None,
            }
        )
        _atomic_write_json(status_path, status)
        print(json.dumps(aggregate, indent=2, default=_jsonable))
        if not required_pass:
            raise SystemExit(1)
    except KeyboardInterrupt:
        status.update(
            {
                "status": "interrupted",
                "outcome": "interrupted",
                "updated_at": datetime.now().astimezone().isoformat(),
            }
        )
        _atomic_write_json(status_path, status)
        raise
    except SystemExit:
        raise
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "outcome": "runtime_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "updated_at": datetime.now().astimezone().isoformat(),
            }
        )
        _atomic_write_json(status_path, status)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
