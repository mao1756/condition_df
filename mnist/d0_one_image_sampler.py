from __future__ import annotations

"""Restartable paired sampler for the Experiment-12 one-image gate.

The production gate compares the direct-Doob sampler at control strengths zero
and one.  This module deliberately owns the random stream, terminal selection,
restart state, and diagnostics so that the comparison cannot accidentally use
different terminal states or Brownian inputs in its two arms.
"""

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_one_image_gate import configure_exact_torch_backend
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, project_edge_flux_torch
from mnist.experiment12_d0 import (
    Experiment12D0Config,
    _direct_doob_reverse_substep,
    _normalize_d0_target_space,
    _validate_direct_doob_config,
)


_SCHEMA_VERSION = 1
_ARM_NAMES = ("strength0", "strength1")
_SUM_DIAGNOSTICS = (
    "limited_edges",
    "proposed_edges",
    "nonfinite_edges",
    "mobility_weight_sum",
    "limited_mobility_weight_sum",
    "noise_energy_sum",
    "limited_noise_energy_sum",
    "floor_touched_pixels",
    "floor_proposed_pixels",
    "floor_correction_l1",
    "renorm_correction_l1",
)


@dataclass(frozen=True)
class PairedSamplerConfig:
    sample_batch_size: int = 8
    checkpoint_every_outer_steps: int = 8
    time_bin_edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    deterministic: bool = False
    show_progress: bool = True

    def __post_init__(self) -> None:
        if int(self.sample_batch_size) <= 0:
            raise ValueError("sample_batch_size must be positive")
        if int(self.checkpoint_every_outer_steps) <= 0:
            raise ValueError("checkpoint_every_outer_steps must be positive")
        edges = tuple(float(value) for value in self.time_bin_edges)
        if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError("time_bin_edges must start at 0 and end at 1")
        if any(not math.isfinite(value) for value in edges) or any(
            right <= left for left, right in zip(edges, edges[1:])
        ):
            raise ValueError("time_bin_edges must be finite and strictly increasing")
        object.__setattr__(self, "time_bin_edges", edges)


@dataclass(frozen=True)
class PairedSeedSamplingResult:
    complete: bool
    eval_seed: int
    terminal_indices: np.ndarray
    labels: np.ndarray
    terminal_states: np.ndarray
    samples_strength0: np.ndarray
    samples_strength1: np.ndarray
    mixed_target: np.ndarray
    unmixed_target: np.ndarray | None
    per_sample_metrics: tuple[dict[str, float | int], ...]
    arm_summaries: dict[str, dict[str, float | int]]
    time_bin_metrics: tuple[dict[str, float | int | str], ...]
    checkpoint_path: str
    configuration_fingerprint: str
    outer_steps_completed: int


@dataclass(frozen=True)
class PairedSamplingResult:
    complete: bool
    eval_seeds: np.ndarray
    terminal_indices: np.ndarray
    labels: np.ndarray
    terminal_states: np.ndarray
    samples_strength0: np.ndarray
    samples_strength1: np.ndarray
    mixed_target: np.ndarray
    unmixed_target: np.ndarray | None
    per_sample_metrics: tuple[dict[str, float | int], ...]
    arm_summaries: dict[str, dict[str, float | int]]
    time_bin_metrics: tuple[dict[str, float | int | str], ...]
    seed_results: tuple[PairedSeedSamplingResult, ...]
    output_dir: str


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_jsonable)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent, delete=False
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


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    temporary = _atomic_temp_path(path)
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_or_create_terminal_assignments(
    path: str | Path,
    *,
    validation_terminal_indices: Sequence[int],
    eval_seeds: Sequence[int],
    samples_per_seed: int,
    selection_seed: int = 260718,
) -> dict[int, tuple[int, ...]]:
    """Persist a deterministic, disjoint assignment of validation terminals."""

    output_path = Path(path)
    available = tuple(int(value) for value in validation_terminal_indices)
    seeds = tuple(int(value) for value in eval_seeds)
    count = int(samples_per_seed)
    if count <= 0 or not seeds:
        raise ValueError("eval_seeds and samples_per_seed must be non-empty and positive")
    if len(set(available)) != len(available):
        raise ValueError("validation_terminal_indices must be unique")
    if len(set(seeds)) != len(seeds):
        raise ValueError("eval_seeds must be unique")
    if len(available) < len(seeds) * count:
        raise ValueError("not enough validation terminals for disjoint evaluation assignments")

    request = {
        "schema_version": _SCHEMA_VERSION,
        "validation_terminal_indices": list(available),
        "eval_seeds": list(seeds),
        "samples_per_seed": count,
        "selection_seed": int(selection_seed),
    }
    request_fingerprint = _canonical_digest(request)
    rng = np.random.default_rng(int(selection_seed))
    order = np.asarray(available, dtype=np.int64)[rng.permutation(len(available))]
    expected_assignments: dict[int, tuple[int, ...]] = {}
    for seed_index, seed in enumerate(seeds):
        offset = seed_index * count
        expected_assignments[seed] = tuple(
            int(value) for value in order[offset : offset + count]
        )
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if payload.get("request_fingerprint") != request_fingerprint:
            raise ValueError("persisted terminal assignment does not match the requested validation set")
        assignments = {
            int(seed): tuple(int(value) for value in payload["assignments"][str(seed)])
            for seed in seeds
        }
        if assignments != expected_assignments:
            raise ValueError(
                "persisted terminal assignments differ from the deterministic selection"
            )
    else:
        assignments = expected_assignments
        payload = {
            **request,
            "request_fingerprint": request_fingerprint,
            "assignments": {str(seed): list(assignments[seed]) for seed in seeds},
        }
        _atomic_write_json(output_path, payload)

    flat = [value for seed in seeds for value in assignments[seed]]
    if len(flat) != len(set(flat)) or any(value not in available for value in flat):
        raise ValueError("persisted terminal assignments are not a disjoint subset of validation terminals")
    return assignments


def _flatten_states(states: np.ndarray | Tensor, grid_size: int) -> np.ndarray:
    out = np.asarray(states.detach().cpu() if isinstance(states, Tensor) else states, dtype=np.float32)
    if out.ndim == 3 and out.shape[1:] == (grid_size, grid_size):
        out = out.reshape(out.shape[0], -1)
    if out.ndim != 2 or out.shape[1] != grid_size * grid_size:
        raise ValueError("terminal_states have an incompatible grid shape")
    if not np.isfinite(out).all():
        raise ValueError("terminal_states must be finite")
    return np.ascontiguousarray(out)


def _flatten_target(target: np.ndarray | Tensor, grid_size: int, name: str) -> np.ndarray:
    out = np.asarray(target.detach().cpu() if isinstance(target, Tensor) else target, dtype=np.float32).reshape(-1)
    if out.size != grid_size * grid_size or not np.isfinite(out).all():
        raise ValueError(f"{name} must be one finite grid-sized image")
    total = float(out.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError(f"{name} must have positive finite mass")
    return np.ascontiguousarray(out / total)


def _sampler_seed(eval_seed: int, batch_index: int, terminal_indices: Sequence[int]) -> int:
    # Strength is intentionally absent: both arms receive the same stream.
    payload = f"d0-one-image:{int(eval_seed)}:{int(batch_index)}:" + ",".join(
        str(int(value)) for value in terminal_indices
    )
    value = int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "little")
    return value % (2**63 - 1)


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _runtime_manifest(
    *,
    selected_states: np.ndarray,
    selected_labels: np.ndarray,
    terminal_indices: np.ndarray,
    mixed_target: np.ndarray,
    unmixed_target: np.ndarray | None,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    rate_schedule: np.ndarray,
    horizon: float,
    start_substep: int,
    physical_target_scale: float,
    eval_seed: int,
    sampler_config: PairedSamplerConfig,
    device: torch.device,
    fingerprints: Mapping[str, str],
    exact_backend: Mapping[str, object],
) -> dict[str, object]:
    sampler_semantics = asdict(sampler_config)
    sampler_semantics.pop("show_progress", None)
    return {
        "schema_version": _SCHEMA_VERSION,
        "fingerprints": {str(key): str(value) for key, value in sorted(fingerprints.items())},
        "terminal_indices": terminal_indices.tolist(),
        "terminal_states_sha256": _array_digest(selected_states),
        "terminal_labels_sha256": _array_digest(selected_labels),
        "mixed_target_sha256": _array_digest(mixed_target),
        "unmixed_target_sha256": None if unmixed_target is None else _array_digest(unmixed_target),
        "rate_schedule_sha256": _array_digest(rate_schedule),
        "horizon": float(horizon),
        "start_substep": int(start_substep),
        "physical_target_scale": float(physical_target_scale),
        "eval_seed": int(eval_seed),
        "sampler_config": sampler_semantics,
        "kernel": {
            "grid_size": int(dynamics_config.grid_size),
            "mass_floor": float(dynamics_config.mass_floor),
            "limiter_fraction": float(dynamics_config.limiter_fraction),
            "edge_alpha_mode": str(dynamics_config.edge_alpha_mode),
            "alpha_eff": float(dynamics_config.alpha_eff),
            "sample_steps": int(d0_config.sample_steps),
            "reference_substeps": int(d0_config.reference_substeps),
            "tau_eff": float(d0_config.tau_eff),
            "target_space": str(_normalize_d0_target_space(d0_config.d0_target_space)),
            "physical_target_normalization": str(d0_config.physical_target_normalization),
            "physical_sampler_noise_mode": str(d0_config.physical_sampler_noise_mode),
            "control_output_clip": float(d0_config.control_output_clip),
            "sample_project_learned_mean": bool(d0_config.sample_project_learned_mean),
        },
        "runtime": {
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
            "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
            "device_type": str(device.type),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "exact_backend": dict(exact_backend),
        },
    }


def _new_accumulator(device: torch.device, bins: int) -> dict[str, Tensor]:
    accumulator = {
        key: torch.zeros(bins, dtype=torch.float64, device=device) for key in _SUM_DIAGNOSTICS
    }
    accumulator.update(
        {
            "max_simplex_mass_error": torch.zeros(bins, dtype=torch.float64, device=device),
            "learned_sq_sum": torch.zeros(bins, dtype=torch.float64, device=device),
            "free_sq_sum": torch.zeros(bins, dtype=torch.float64, device=device),
            "noise_sq_sum": torch.zeros(bins, dtype=torch.float64, device=device),
            "edge_value_count": torch.zeros(bins, dtype=torch.float64, device=device),
            "substep_count": torch.zeros(bins, dtype=torch.float64, device=device),
            "path_substep_count": torch.zeros(bins, dtype=torch.float64, device=device),
        }
    )
    return accumulator


def _accumulator_to_cpu(accumulator: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in accumulator.items()}


def _accumulator_to_device(
    accumulator: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {key: value.to(device=device, dtype=torch.float64) for key, value in accumulator.items()}


def _update_accumulator(
    accumulator: dict[str, Tensor],
    *,
    bin_index: int,
    diagnostics: Mapping[str, float | Tensor],
    learned_delta: Tensor,
    free_delta: Tensor,
    noise_delta: Tensor,
) -> None:
    index = int(bin_index)
    for key in _SUM_DIAGNOSTICS:
        value = diagnostics.get(key, 0.0)
        tensor = value if isinstance(value, Tensor) else learned_delta.new_tensor(float(value))
        accumulator[key][index].add_(tensor.to(dtype=torch.float64))
    simplex = diagnostics.get("max_simplex_mass_error", 0.0)
    simplex_tensor = simplex if isinstance(simplex, Tensor) else learned_delta.new_tensor(float(simplex))
    accumulator["max_simplex_mass_error"][index] = torch.maximum(
        accumulator["max_simplex_mass_error"][index], simplex_tensor.to(dtype=torch.float64)
    )
    accumulator["learned_sq_sum"][index].add_(learned_delta.square().sum().to(torch.float64))
    accumulator["free_sq_sum"][index].add_(free_delta.square().sum().to(torch.float64))
    accumulator["noise_sq_sum"][index].add_(noise_delta.square().sum().to(torch.float64))
    accumulator["edge_value_count"][index].add_(float(learned_delta.numel()))
    accumulator["substep_count"][index].add_(1.0)
    accumulator["path_substep_count"][index].add_(float(learned_delta.shape[0]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def _summarize_accumulator(accumulator: Mapping[str, Tensor]) -> dict[str, float | int]:
    totals = {key: float(value.sum().item()) for key, value in accumulator.items()}
    edge_count = totals["edge_value_count"]
    learned_rms = math.sqrt(_safe_ratio(totals["learned_sq_sum"], edge_count))
    free_rms = math.sqrt(_safe_ratio(totals["free_sq_sum"], edge_count))
    noise_rms = math.sqrt(_safe_ratio(totals["noise_sq_sum"], edge_count))
    out: dict[str, float | int] = {
        key: totals[key] for key in _SUM_DIAGNOSTICS
    }
    out.update(
        {
            "limiter_fraction": _safe_ratio(totals["limited_edges"], totals["proposed_edges"]),
            "mobility_weighted_limiter_fraction": _safe_ratio(
                totals["limited_mobility_weight_sum"], totals["mobility_weight_sum"]
            ),
            "noise_energy_weighted_limiter_fraction": _safe_ratio(
                totals["limited_noise_energy_sum"], totals["noise_energy_sum"]
            ),
            "floor_touched_fraction": _safe_ratio(
                totals["floor_touched_pixels"], totals["floor_proposed_pixels"]
            ),
            "floor_correction_l1_per_path_substep": _safe_ratio(
                totals["floor_correction_l1"], totals["path_substep_count"]
            ),
            "renorm_correction_l1_per_path_substep": _safe_ratio(
                totals["renorm_correction_l1"], totals["path_substep_count"]
            ),
            "max_simplex_mass_error": float(accumulator["max_simplex_mass_error"].max().item()),
            "learned_step_rms": learned_rms,
            "free_step_rms": free_rms,
            "noise_step_rms": noise_rms,
            "learned_to_noise_ratio": _safe_ratio(learned_rms, noise_rms),
            "substep_count": int(round(totals["substep_count"])),
            "path_substep_count": int(round(totals["path_substep_count"])),
        }
    )
    out["nonfinite_edges"] = int(round(totals["nonfinite_edges"]))
    out["floor_touched_pixels"] = int(round(totals["floor_touched_pixels"]))
    out["floor_proposed_pixels"] = int(round(totals["floor_proposed_pixels"]))
    return out


def _time_bin_rows(
    accumulator: Mapping[str, Tensor],
    *,
    eval_seed: int,
    strength: int,
    edges: Sequence[float],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for index, (left, right) in enumerate(zip(edges, edges[1:])):
        values = {key: float(value[index].item()) for key, value in accumulator.items()}
        count = values["edge_value_count"]
        learned_rms = math.sqrt(_safe_ratio(values["learned_sq_sum"], count))
        noise_rms = math.sqrt(_safe_ratio(values["noise_sq_sum"], count))
        rows.append(
            {
                "eval_seed": int(eval_seed),
                "strength": int(strength),
                "tau_fraction_left": float(left),
                "tau_fraction_right": float(right),
                "right_closed": int(index == len(edges) - 2),
                "substep_count": int(round(values["substep_count"])),
                "path_substep_count": int(round(values["path_substep_count"])),
                "learned_step_rms": learned_rms,
                "free_step_rms": math.sqrt(_safe_ratio(values["free_sq_sum"], count)),
                "noise_step_rms": noise_rms,
                "learned_to_noise_ratio": _safe_ratio(learned_rms, noise_rms),
                "limiter_fraction": _safe_ratio(values["limited_edges"], values["proposed_edges"]),
                "mobility_weighted_limiter_fraction": _safe_ratio(
                    values["limited_mobility_weight_sum"], values["mobility_weight_sum"]
                ),
                "noise_energy_weighted_limiter_fraction": _safe_ratio(
                    values["limited_noise_energy_sum"], values["noise_energy_sum"]
                ),
                "nonfinite_edges": int(round(values["nonfinite_edges"])),
                "floor_touched_fraction": _safe_ratio(
                    values["floor_touched_pixels"], values["floor_proposed_pixels"]
                ),
                "floor_correction_l1_per_path_substep": _safe_ratio(
                    values["floor_correction_l1"], values["path_substep_count"]
                ),
                "renorm_correction_l1_per_path_substep": _safe_ratio(
                    values["renorm_correction_l1"], values["path_substep_count"]
                ),
                "max_simplex_mass_error": float(accumulator["max_simplex_mass_error"][index].item()),
            }
        )
    return rows


def _time_bin_index(fraction: float, edges: Sequence[float]) -> int:
    return min(max(int(np.searchsorted(np.asarray(edges), fraction, side="right") - 1), 0), len(edges) - 2)


def _learned_delta(
    model: torch.nn.Module,
    *,
    tau: Tensor,
    states: Tensor,
    labels: Tensor,
    grid_size: int,
    strength: float,
    physical_target_scale: float,
    bypass_zero: bool,
) -> Tensor:
    shape = (states.shape[0], 2, int(grid_size), int(grid_size))
    if bypass_zero and float(strength) == 0.0:
        return states.new_zeros(shape)
    raw = model(tau, states, labels, None)
    if raw.shape != shape:
        raise ValueError(f"model output must have shape {shape}")
    projected = project_edge_flux_torch(raw, grid_size=int(grid_size))
    return float(strength) * float(physical_target_scale) * projected


@torch.no_grad()
def verify_strength_zero_bypass_equivalence(
    model: torch.nn.Module,
    *,
    states: Tensor,
    tau: Tensor,
    labels: Tensor,
    rate: float,
    dt: float,
    dynamics_config: DirectFluxMNISTConfig,
    physical_target_scale: float,
    standard_normal: Tensor,
) -> dict[str, float | int]:
    """Compare the zero-arm bypass with the ordinary strength-zero path."""

    bypass_delta = _learned_delta(
        model,
        tau=tau,
        states=states,
        labels=labels,
        grid_size=int(dynamics_config.grid_size),
        strength=0.0,
        physical_target_scale=float(physical_target_scale),
        bypass_zero=True,
    )
    ordinary_delta = _learned_delta(
        model,
        tau=tau,
        states=states,
        labels=labels,
        grid_size=int(dynamics_config.grid_size),
        strength=0.0,
        physical_target_scale=float(physical_target_scale),
        bypass_zero=False,
    )
    bypass = _direct_doob_reverse_substep(
        states.clone(), bypass_delta, rate=rate, dt=dt, dynamics_config=dynamics_config,
        standard_normal=standard_normal, diagnostics_device=True,
    )
    ordinary = _direct_doob_reverse_substep(
        states.clone(), ordinary_delta, rate=rate, dt=dt, dynamics_config=dynamics_config,
        standard_normal=standard_normal, diagnostics_device=True,
    )
    state_error = float((bypass.states - ordinary.states).abs().max().cpu())
    delta_error = float((bypass_delta - ordinary_delta).abs().max().cpu())
    return {
        "pass": int(state_error == 0.0 and delta_error == 0.0),
        "max_state_error": state_error,
        "max_learned_delta_error": delta_error,
    }


def _correlation(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered_samples = samples - samples.mean(axis=1, keepdims=True)
    centered_target = target - target.mean()
    denominator = np.sqrt((centered_samples**2).sum(axis=1) * float((centered_target**2).sum()))
    numerator = (centered_samples * centered_target[None, :]).sum(axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(samples.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def _per_sample_rows(
    *,
    samples0: np.ndarray,
    samples1: np.ndarray,
    terminal_indices: np.ndarray,
    labels: np.ndarray,
    mixed_target: np.ndarray,
    unmixed_target: np.ndarray | None,
    eval_seed: int,
) -> list[dict[str, float | int]]:
    corr0 = _correlation(samples0, mixed_target)
    corr1 = _correlation(samples1, mixed_target)
    l10 = np.abs(samples0 - mixed_target[None, :]).sum(axis=1)
    l11 = np.abs(samples1 - mixed_target[None, :]).sum(axis=1)
    raw_corr0 = raw_corr1 = raw_l10 = raw_l11 = None
    if unmixed_target is not None:
        raw_corr0 = _correlation(samples0, unmixed_target)
        raw_corr1 = _correlation(samples1, unmixed_target)
        raw_l10 = np.abs(samples0 - unmixed_target[None, :]).sum(axis=1)
        raw_l11 = np.abs(samples1 - unmixed_target[None, :]).sum(axis=1)
    rows: list[dict[str, float | int]] = []
    for index in range(samples0.shape[0]):
        row: dict[str, float | int] = {
            "eval_seed": int(eval_seed),
            "sample_index_within_seed": int(index),
            "terminal_index": int(terminal_indices[index]),
            "label": int(labels[index]),
            "mixed_corr_strength0": float(corr0[index]),
            "mixed_corr_strength1": float(corr1[index]),
            "paired_corr_improvement": float(corr1[index] - corr0[index]),
            "mixed_l1_strength0": float(l10[index]),
            "mixed_l1_strength1": float(l11[index]),
            "paired_l1_reduction": float(l10[index] - l11[index]),
            "paired_relative_l1_reduction": _safe_ratio(float(l10[index] - l11[index]), float(l10[index])),
            "simplex_error_strength0": float(abs(samples0[index].sum() - 1.0)),
            "simplex_error_strength1": float(abs(samples1[index].sum() - 1.0)),
        }
        if unmixed_target is not None:
            assert raw_corr0 is not None and raw_corr1 is not None
            assert raw_l10 is not None and raw_l11 is not None
            row.update(
                {
                    "unmixed_corr_strength0": float(raw_corr0[index]),
                    "unmixed_corr_strength1": float(raw_corr1[index]),
                    "unmixed_l1_strength0": float(raw_l10[index]),
                    "unmixed_l1_strength1": float(raw_l11[index]),
                }
            )
        rows.append(row)
    return rows


def _load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def run_paired_seed_sampling(
    model: torch.nn.Module,
    *,
    terminal_states: np.ndarray | Tensor,
    terminal_labels: Sequence[int] | np.ndarray | Tensor,
    terminal_indices: Sequence[int],
    mixed_target: np.ndarray | Tensor,
    unmixed_target: np.ndarray | Tensor | None,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    rate_schedule: Sequence[float] | np.ndarray,
    horizon: float,
    physical_target_scale: float,
    eval_seed: int,
    device: torch.device,
    checkpoint_path: str | Path,
    fingerprints: Mapping[str, str],
    start_substep: int | None = None,
    sampler_config: PairedSamplerConfig = PairedSamplerConfig(),
    resume: bool = True,
    stop_after_outer_steps: int | None = None,
) -> PairedSeedSamplingResult:
    """Run (or exactly resume) one seed of a paired strength-zero/one rollout."""

    exact_backend = configure_exact_torch_backend(device)
    _validate_direct_doob_config(d0_config)
    if _normalize_d0_target_space(d0_config.d0_target_space) != "doob-physical-residual":
        raise ValueError("paired gate sampling requires doob-physical-residual")
    if not math.isfinite(float(horizon)) or float(horizon) <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(float(physical_target_scale)) or float(physical_target_scale) <= 0.0:
        raise ValueError("physical_target_scale must be finite and positive")
    configured_scale = float(d0_config.physical_target_scale)
    if configured_scale > 0.0 and not math.isclose(
        configured_scale,
        float(physical_target_scale),
        rel_tol=1e-7,
        abs_tol=1e-12,
    ):
        raise ValueError("physical_target_scale must match the frozen D0 training scale")
    n = int(dynamics_config.grid_size)
    all_states = _flatten_states(terminal_states, n)
    all_labels = np.asarray(
        terminal_labels.detach().cpu() if isinstance(terminal_labels, Tensor) else terminal_labels,
        dtype=np.int64,
    ).reshape(-1)
    if all_labels.size != all_states.shape[0]:
        raise ValueError("terminal_labels must align with terminal_states")
    selected_indices = np.asarray(tuple(int(value) for value in terminal_indices), dtype=np.int64)
    if selected_indices.size == 0 or len(set(selected_indices.tolist())) != selected_indices.size:
        raise ValueError("terminal_indices must be non-empty and unique")
    if selected_indices.min() < 0 or selected_indices.max() >= all_states.shape[0]:
        raise IndexError("terminal index is outside terminal_states")
    selected_states = np.ascontiguousarray(all_states[selected_indices])
    selected_labels = np.ascontiguousarray(all_labels[selected_indices])
    target_mixed = _flatten_target(mixed_target, n, "mixed_target")
    target_unmixed = None if unmixed_target is None else _flatten_target(unmixed_target, n, "unmixed_target")
    rates = np.ascontiguousarray(np.asarray(rate_schedule, dtype=np.float64).reshape(-1))
    if rates.size != int(d0_config.sample_steps) or not np.isfinite(rates).all() or np.any(rates < 0.0):
        raise ValueError("rate_schedule must be finite, non-negative, and match sample_steps")
    reference_substeps = int(d0_config.reference_substeps)
    total_substeps = int(rates.size * reference_substeps)
    start = total_substeps if start_substep is None else int(start_substep)
    if start <= 0 or start > total_substeps:
        raise ValueError("start_substep must be in [1, sample_steps * reference_substeps]")
    dt_sub = float(horizon) / float(total_substeps)
    manifest = _runtime_manifest(
        selected_states=selected_states,
        selected_labels=selected_labels,
        terminal_indices=selected_indices,
        mixed_target=target_mixed,
        unmixed_target=target_unmixed,
        dynamics_config=dynamics_config,
        d0_config=d0_config,
        rate_schedule=rates,
        horizon=float(horizon),
        start_substep=start,
        physical_target_scale=float(physical_target_scale),
        eval_seed=int(eval_seed),
        sampler_config=sampler_config,
        device=device,
        fingerprints=fingerprints,
        exact_backend=exact_backend,
    )
    configuration_fingerprint = _canonical_digest(manifest)
    checkpoint = Path(checkpoint_path)
    batch_size = int(sampler_config.sample_batch_size)
    batch_ranges = [
        (offset, min(offset + batch_size, selected_indices.size))
        for offset in range(0, selected_indices.size, batch_size)
    ]
    bins = len(sampler_config.time_bin_edges) - 1

    generator = _device_generator(device, 0)
    if checkpoint.exists() and resume:
        payload = _load_checkpoint(checkpoint)
        if int(payload.get("schema_version", -1)) != _SCHEMA_VERSION:
            raise ValueError("paired sampler checkpoint schema is incompatible")
        if payload.get("configuration_fingerprint") != configuration_fingerprint:
            raise ValueError("paired sampler checkpoint fingerprint mismatch")
        outputs = {
            name: torch.as_tensor(payload["outputs"][name], dtype=torch.float32)
            for name in _ARM_NAMES
        }
        accumulators = {
            name: _accumulator_to_device(payload["accumulators"][name], device)
            for name in _ARM_NAMES
        }
        batch_index = int(payload["batch_index"])
        branch_index = int(payload["branch_index"])
        q_next = int(payload["q_next"])
        current_states_payload = payload.get("current_states")
        current_states = (
            None
            if current_states_payload is None
            else torch.as_tensor(current_states_payload, dtype=torch.float32, device=device)
        )
        generator.set_state(torch.as_tensor(payload["generator_state"], dtype=torch.uint8))
        outer_steps_completed = int(payload["outer_steps_completed"])
        complete = bool(payload.get("complete", False))
    elif checkpoint.exists():
        raise FileExistsError(f"paired sampler checkpoint already exists: {checkpoint}")
    else:
        outputs = {
            name: torch.full((selected_indices.size, n * n), float("nan"), dtype=torch.float32)
            for name in _ARM_NAMES
        }
        accumulators = {name: _new_accumulator(device, bins) for name in _ARM_NAMES}
        batch_index = 0
        branch_index = 0
        q_next = start - 1
        begin, end = batch_ranges[0]
        current_states = torch.as_tensor(selected_states[begin:end], device=device)
        generator.manual_seed(_sampler_seed(int(eval_seed), batch_index, selected_indices[begin:end]))
        outer_steps_completed = 0
        complete = False

    def save_checkpoint() -> None:
        _atomic_torch_save(
            checkpoint,
            {
                "schema_version": _SCHEMA_VERSION,
                "configuration_fingerprint": configuration_fingerprint,
                "runtime_manifest": manifest,
                "complete": bool(complete),
                "batch_index": int(batch_index),
                "branch_index": int(branch_index),
                "q_next": int(q_next),
                "current_states": None if current_states is None else current_states.detach().cpu(),
                "generator_state": generator.get_state().cpu(),
                "outputs": outputs,
                "accumulators": {
                    name: _accumulator_to_cpu(accumulators[name]) for name in _ARM_NAMES
                },
                "outer_steps_completed": int(outer_steps_completed),
            },
        )

    model.eval()
    invocation_outer_steps = 0
    started_at = time.monotonic()
    groups_per_branch = (start - 1) // reference_substeps + 1
    planned_outer_steps = groups_per_branch * 2 * len(batch_ranges)
    checkpoint_interval = int(sampler_config.checkpoint_every_outer_steps)
    while not complete:
        assert current_states is not None
        begin, end = batch_ranges[batch_index]
        labels_t = torch.as_tensor(selected_labels[begin:end], dtype=torch.long, device=device)
        strength = float(branch_index)
        outer_index = int(q_next // reference_substeps)
        group_floor = int(outer_index * reference_substeps)
        while q_next >= group_floor:
            tau_value = max(float(horizon) - float(q_next + 1) * dt_sub, 0.0)
            tau = torch.full(
                (current_states.shape[0],), tau_value, dtype=current_states.dtype, device=device
            )
            learned_delta = _learned_delta(
                model,
                tau=tau,
                states=current_states,
                labels=labels_t,
                grid_size=n,
                strength=strength,
                physical_target_scale=float(physical_target_scale),
                bypass_zero=True,
            )
            noise_shape = (current_states.shape[0], 2, n, n)
            standard_normal = (
                torch.zeros(noise_shape, dtype=current_states.dtype, device=device)
                if sampler_config.deterministic
                else torch.randn(
                    noise_shape,
                    dtype=current_states.dtype,
                    device=device,
                    generator=generator,
                )
            )
            direct = _direct_doob_reverse_substep(
                current_states,
                learned_delta,
                rate=float(rates[outer_index]),
                dt=dt_sub,
                dynamics_config=dynamics_config,
                standard_normal=standard_normal,
                deterministic=bool(sampler_config.deterministic),
                diagnostics_device=True,
            )
            fraction = min(max(tau_value / float(horizon), 0.0), 1.0)
            _update_accumulator(
                accumulators[_ARM_NAMES[branch_index]],
                bin_index=_time_bin_index(fraction, sampler_config.time_bin_edges),
                diagnostics=direct.diagnostics,
                learned_delta=direct.learned_delta,
                free_delta=direct.free_delta,
                noise_delta=direct.noise_delta,
            )
            current_states = direct.states
            q_next -= 1

        invocation_outer_steps += 1
        outer_steps_completed += 1
        if q_next < 0:
            outputs[_ARM_NAMES[branch_index]][begin:end] = current_states.detach().cpu()
            if branch_index == 0:
                branch_index = 1
                q_next = start - 1
                current_states = torch.as_tensor(selected_states[begin:end], device=device)
                generator.manual_seed(
                    _sampler_seed(int(eval_seed), batch_index, selected_indices[begin:end])
                )
            else:
                batch_index += 1
                branch_index = 0
                if batch_index >= len(batch_ranges):
                    complete = True
                    q_next = -1
                    current_states = None
                else:
                    begin, end = batch_ranges[batch_index]
                    q_next = start - 1
                    current_states = torch.as_tensor(selected_states[begin:end], device=device)
                    generator.manual_seed(
                        _sampler_seed(int(eval_seed), batch_index, selected_indices[begin:end])
                    )

        should_checkpoint = (
            complete
            or outer_steps_completed % checkpoint_interval == 0
            or (
                stop_after_outer_steps is not None
                and invocation_outer_steps >= int(stop_after_outer_steps)
            )
        )
        if should_checkpoint:
            save_checkpoint()
            if sampler_config.show_progress:
                elapsed = max(time.monotonic() - started_at, 1e-9)
                rate_per_second = invocation_outer_steps / elapsed
                remaining = max(planned_outer_steps - outer_steps_completed, 0)
                eta = remaining / max(rate_per_second, 1e-12)
                print(
                    f"paired D0 seed={int(eval_seed)} outer={outer_steps_completed}/{planned_outer_steps} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        if (
            not complete
            and stop_after_outer_steps is not None
            and invocation_outer_steps >= int(stop_after_outer_steps)
        ):
            break

    samples0 = outputs["strength0"].numpy().astype(np.float32, copy=True)
    samples1 = outputs["strength1"].numpy().astype(np.float32, copy=True)
    per_sample = (
        _per_sample_rows(
            samples0=samples0,
            samples1=samples1,
            terminal_indices=selected_indices,
            labels=selected_labels,
            mixed_target=target_mixed,
            unmixed_target=target_unmixed,
            eval_seed=int(eval_seed),
        )
        if complete
        else []
    )
    arm_summaries = {
        str(index): _summarize_accumulator(accumulators[name])
        for index, name in enumerate(_ARM_NAMES)
    }
    time_rows = [
        row
        for index, name in enumerate(_ARM_NAMES)
        for row in _time_bin_rows(
            accumulators[name],
            eval_seed=int(eval_seed),
            strength=index,
            edges=sampler_config.time_bin_edges,
        )
    ]
    return PairedSeedSamplingResult(
        complete=bool(complete),
        eval_seed=int(eval_seed),
        terminal_indices=selected_indices.copy(),
        labels=selected_labels.copy(),
        terminal_states=selected_states.copy(),
        samples_strength0=samples0,
        samples_strength1=samples1,
        mixed_target=target_mixed.copy(),
        unmixed_target=None if target_unmixed is None else target_unmixed.copy(),
        per_sample_metrics=tuple(per_sample),
        arm_summaries=arm_summaries,
        time_bin_metrics=tuple(time_rows),
        checkpoint_path=str(checkpoint),
        configuration_fingerprint=configuration_fingerprint,
        outer_steps_completed=int(outer_steps_completed),
    )


def _combine_arm_summaries(
    seed_results: Sequence[PairedSeedSamplingResult], strength: int
) -> dict[str, float | int]:
    rows = [result.arm_summaries[str(strength)] for result in seed_results]
    if not rows:
        return {}
    sums = {key: sum(float(row.get(key, 0.0)) for row in rows) for key in _SUM_DIAGNOSTICS}
    learned_sq = sum(
        float(row["learned_step_rms"]) ** 2 * float(row["path_substep_count"])
        for row in rows
    )
    free_sq = sum(
        float(row["free_step_rms"]) ** 2 * float(row["path_substep_count"])
        for row in rows
    )
    noise_sq = sum(
        float(row["noise_step_rms"]) ** 2 * float(row["path_substep_count"])
        for row in rows
    )
    weight = sum(float(row["path_substep_count"]) for row in rows)
    learned_rms = math.sqrt(_safe_ratio(learned_sq, weight))
    noise_rms = math.sqrt(_safe_ratio(noise_sq, weight))
    out: dict[str, float | int] = dict(sums)
    out.update(
        {
            "limiter_fraction": _safe_ratio(sums["limited_edges"], sums["proposed_edges"]),
            "mobility_weighted_limiter_fraction": _safe_ratio(
                sums["limited_mobility_weight_sum"], sums["mobility_weight_sum"]
            ),
            "noise_energy_weighted_limiter_fraction": _safe_ratio(
                sums["limited_noise_energy_sum"], sums["noise_energy_sum"]
            ),
            "floor_touched_fraction": _safe_ratio(
                sums["floor_touched_pixels"], sums["floor_proposed_pixels"]
            ),
            "floor_correction_l1_per_path_substep": _safe_ratio(
                sums["floor_correction_l1"], weight
            ),
            "renorm_correction_l1_per_path_substep": _safe_ratio(
                sums["renorm_correction_l1"], weight
            ),
            "max_simplex_mass_error": max(float(row["max_simplex_mass_error"]) for row in rows),
            "learned_step_rms": learned_rms,
            "free_step_rms": math.sqrt(_safe_ratio(free_sq, weight)),
            "noise_step_rms": noise_rms,
            "learned_to_noise_ratio": _safe_ratio(learned_rms, noise_rms),
            "substep_count": int(sum(int(row["substep_count"]) for row in rows)),
            "path_substep_count": int(weight),
            "nonfinite_edges": int(round(sums["nonfinite_edges"])),
            "floor_touched_pixels": int(round(sums["floor_touched_pixels"])),
            "floor_proposed_pixels": int(round(sums["floor_proposed_pixels"])),
        }
    )
    return out


def _result_summary(result: PairedSamplingResult) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "complete": bool(result.complete),
        "num_samples": int(result.terminal_indices.size),
        "eval_seeds": sorted(set(int(value) for value in result.eval_seeds.tolist())),
        "arm_summaries": result.arm_summaries,
        "per_seed": [
            {
                "eval_seed": seed.eval_seed,
                "complete": seed.complete,
                "terminal_indices": seed.terminal_indices.tolist(),
                "checkpoint_path": seed.checkpoint_path,
                "configuration_fingerprint": seed.configuration_fingerprint,
                "outer_steps_completed": seed.outer_steps_completed,
                "arm_summaries": seed.arm_summaries,
            }
            for seed in result.seed_results
        ],
    }


@torch.no_grad()
def run_paired_d0_sampling(
    model: torch.nn.Module,
    *,
    terminal_states: np.ndarray | Tensor,
    terminal_labels: Sequence[int] | np.ndarray | Tensor,
    terminal_assignments: Mapping[int, Sequence[int]],
    mixed_target: np.ndarray | Tensor,
    unmixed_target: np.ndarray | Tensor | None,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    rate_schedule: Sequence[float] | np.ndarray,
    horizon: float,
    physical_target_scale: float,
    device: torch.device,
    output_dir: str | Path,
    fingerprints: Mapping[str, str],
    start_substep: int | None = None,
    sampler_config: PairedSamplerConfig = PairedSamplerConfig(),
    resume: bool = True,
    stop_after_outer_steps: int | None = None,
) -> PairedSamplingResult:
    """Run all persisted seed assignments and atomically write paired artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seed_results: list[PairedSeedSamplingResult] = []
    for eval_seed, indices in terminal_assignments.items():
        result = run_paired_seed_sampling(
            model,
            terminal_states=terminal_states,
            terminal_labels=terminal_labels,
            terminal_indices=indices,
            mixed_target=mixed_target,
            unmixed_target=unmixed_target,
            dynamics_config=dynamics_config,
            d0_config=d0_config,
            rate_schedule=rate_schedule,
            horizon=horizon,
            physical_target_scale=physical_target_scale,
            eval_seed=int(eval_seed),
            device=device,
            checkpoint_path=destination / "sampling_checkpoints" / f"seed-{int(eval_seed)}.pt",
            fingerprints=fingerprints,
            start_substep=start_substep,
            sampler_config=sampler_config,
            resume=resume,
            stop_after_outer_steps=stop_after_outer_steps,
        )
        seed_results.append(result)
        if not result.complete:
            break

    complete = len(seed_results) == len(terminal_assignments) and all(
        result.complete for result in seed_results
    )
    flatten = lambda attr, dtype: np.concatenate(  # noqa: E731
        [np.asarray(getattr(result, attr), dtype=dtype) for result in seed_results], axis=0
    ) if seed_results else np.asarray([], dtype=dtype)
    eval_seeds = np.concatenate(
        [np.full(result.terminal_indices.size, result.eval_seed, dtype=np.int64) for result in seed_results]
    ) if seed_results else np.asarray([], dtype=np.int64)
    target_mixed = _flatten_target(mixed_target, int(dynamics_config.grid_size), "mixed_target")
    target_unmixed = (
        None
        if unmixed_target is None
        else _flatten_target(unmixed_target, int(dynamics_config.grid_size), "unmixed_target")
    )
    combined = PairedSamplingResult(
        complete=bool(complete),
        eval_seeds=eval_seeds,
        terminal_indices=flatten("terminal_indices", np.int64),
        labels=flatten("labels", np.int64),
        terminal_states=flatten("terminal_states", np.float32),
        samples_strength0=flatten("samples_strength0", np.float32),
        samples_strength1=flatten("samples_strength1", np.float32),
        mixed_target=target_mixed,
        unmixed_target=target_unmixed,
        per_sample_metrics=tuple(
            row for result in seed_results for row in result.per_sample_metrics
        ),
        arm_summaries={
            str(strength): _combine_arm_summaries(seed_results, strength) for strength in (0, 1)
        },
        time_bin_metrics=tuple(
            row for result in seed_results for row in result.time_bin_metrics
        ),
        seed_results=tuple(seed_results),
        output_dir=str(destination),
    )
    _atomic_write_json(destination / "sampler_summary.json", _result_summary(combined))
    if complete:
        _atomic_save_npz(
            destination / "paired_samples.npz",
            eval_seeds=combined.eval_seeds,
            terminal_indices=combined.terminal_indices,
            labels=combined.labels,
            terminal_states=combined.terminal_states,
            samples_strength0=combined.samples_strength0,
            samples_strength1=combined.samples_strength1,
            mixed_target=combined.mixed_target,
            unmixed_target=(
                np.asarray([], dtype=np.float32)
                if combined.unmixed_target is None
                else combined.unmixed_target
            ),
        )
        _atomic_write_csv(destination / "per_sample_metrics.csv", combined.per_sample_metrics)
        _atomic_write_csv(destination / "sampling_time_bins.csv", combined.time_bin_metrics)
    return combined


__all__ = [
    "PairedSamplerConfig",
    "PairedSeedSamplingResult",
    "PairedSamplingResult",
    "resolve_or_create_terminal_assignments",
    "run_paired_seed_sampling",
    "run_paired_d0_sampling",
    "verify_strength_zero_bypass_equivalence",
]
