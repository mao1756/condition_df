from __future__ import annotations

r"""Experiment 12 / D0 Phase 0.8 forward-from-data reference diagnostics.

P0.8 keeps the P0.7 fast fixed-substep integrated-time semantics and
splits the MNIST-start D0 practical gate from Dirichlet-start symmetry checks:

* The default practical reference is the soft symmetric Dirichlet(alpha_eff)
  finite-volume process, with faithful time-change coupling
  ``w_free = rate`` and ``w_sigma = sqrt(rate)``.  The manuscript/grid
  ``alpha_h = beta h^d`` reference and independent legacy scaling are retained
  as explicit diagnostics only.
* ``--sweep-tau-eff`` can now mean the requested integrated effective
  time-change, rather than a raw per-unit-horizon rate.  In faithful mode the
  schedule is normalized so ``sum_k rate_k * dt ~= tau_eff``.
* The simulator no longer uses global clipping/retry or a min-endpoint
  stiffness precheck as a hidden dynamics modifier.  A fixed-substep,
  direction-aware limiter advances the feasible part of the deterministic drift
  and realized Gaussian increment, while masking limiter-touched edges for later
  innovation-regression losses.
* The MNIST/data-start gate no longer requires terminal samples to look
  Dirichlet.  It treats the terminal bank as empirical forward-from-data
  ``p_T`` for D0 reverse initialization.
* A separate ``--init-law dirichlet --phase0-gate-mode exact-stationary``
  diagnostic tests invariance of the symmetric reference from Dirichlet starts.
* Raw limiter counts are still logged, but D0 practical gating can use
  mobility- or noise-energy-weighted limiter fractions.
The script still does not create a model, does not train, and does not produce a
D0 cache.  It writes schedule diagnostics, noising preview grids, per-schedule
terminal banks, and a canonical Phase-1 prior bank only when the D0 practical gate passes.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    _lowres_features_np,
    _progress,
    checkerboard_energy_torch,
    edge_alpha_value,
    harmonic_mobility_channels,
    image_total_variation,
    load_mnist_measure_dataset,
    masked_reference_free_step_torch,
    natural_horizon,
)
from mnist.weighted_point_cloud import normalize_images_to_measures


@dataclass
class Phase0SingleResult:
    """Outputs for one forward-from-data noising schedule."""

    run_id: str
    config: DirectFluxMNISTConfig
    lambda_mix: float
    free_weight: float
    noise_weight: float
    reference_scale_mode: str
    reference_rate: float | None
    tau_eff: float | None
    time_change_mode: str
    init_law: str
    effective_time_integral: float | None
    rate_schedule: np.ndarray
    substeps: int
    sample_steps: int
    labels: np.ndarray
    source_indices: np.ndarray
    initial_states: np.ndarray
    terminal_states: np.ndarray
    metrics: list[dict[str, float | int | str]]
    summary: dict[str, float | int | str]
    checkpoint_states: dict[int, np.ndarray]


def _serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _parse_csv_floats(value: str | float | int | None) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (float, int)):
        return [float(value)]
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        return []
    return [float(piece) for piece in pieces]


def _parse_csv_ints(value: str | int | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        return []
    return [int(piece) for piece in pieces]


def _digamma_float(value: float) -> float:
    return float(torch.digamma(torch.tensor(float(value), dtype=torch.float64)).item())


def expected_symmetric_dirichlet_entropy(config: DirectFluxMNISTConfig) -> float:
    r"""Expected entropy of the symmetric Dirichlet grid reference law."""

    n = int(config.grid_size)
    cell_alpha = float(edge_alpha_value(config))
    total_alpha = float(n * n) * cell_alpha
    if cell_alpha <= 0.0 or total_alpha <= 0.0:
        return float("nan")
    return _digamma_float(total_alpha + 1.0) - _digamma_float(cell_alpha + 1.0)


def _reference_schedules(args: argparse.Namespace) -> list[dict[str, float | str | None]]:
    """Return reference scaling schedules for the CLI.

    In P0.7, explicit ``--sweep-tau-eff`` values are integrated effective
    times by default.  The deprecated ``--sweep-reference-rates`` fallback keeps
    literal raw-rate semantics so older commands continue to run.
    """

    mode = str(args.reference_scale_mode)
    schedules: list[dict[str, float | str | None]] = []
    if mode == "faithful":
        tau_values = _parse_csv_floats(getattr(args, "sweep_tau_eff", None))
        if tau_values:
            time_change_mode = str(getattr(args, "time_change_mode", "integral"))
            if time_change_mode not in {"integral", "rate"}:
                raise ValueError("time_change_mode must be 'integral' or 'rate'")
            for tau_eff in tau_values:
                if tau_eff < 0.0 or not math.isfinite(float(tau_eff)):
                    raise ValueError("tau_eff values must be finite and non-negative")
                schedules.append(
                    {
                        "reference_scale_mode": "faithful",
                        "time_change_mode": time_change_mode,
                        "tau_eff": float(tau_eff),
                        # Filled in after the config fixes the horizon.
                        "reference_rate": None if time_change_mode == "integral" else float(tau_eff),
                        "free_weight": float(tau_eff),
                        "noise_weight": math.sqrt(float(tau_eff)),
                    }
                )
            return schedules
        for rate in _parse_csv_floats(getattr(args, "sweep_reference_rates", "1e-6")):
            if rate < 0.0 or not math.isfinite(float(rate)):
                raise ValueError("faithful reference rates must be finite and non-negative")
            schedules.append(
                {
                    "reference_scale_mode": "faithful",
                    "time_change_mode": "rate",
                    "tau_eff": None,
                    "reference_rate": float(rate),
                    "free_weight": float(rate),
                    "noise_weight": math.sqrt(float(rate)),
                }
            )
        return schedules

    for free_weight in _parse_csv_floats(getattr(args, "sweep_free_weights", "0.03")):
        for noise_weight in _parse_csv_floats(getattr(args, "sweep_noise_weights", "0.005")):
            schedules.append(
                {
                    "reference_scale_mode": "independent",
                    "time_change_mode": "rate",
                    "tau_eff": None,
                    "reference_rate": None,
                    "free_weight": float(free_weight),
                    "noise_weight": float(noise_weight),
                }
            )
    return schedules

def _parse_checkpoint_fractions(value: str) -> list[float]:
    fractions = _parse_csv_floats(value)
    return sorted({min(1.0, max(0.0, float(frac))) for frac in fractions})


def _safe_tag(x: float | int | str) -> str:
    text = str(x).replace("-", "m").replace(".", "p").replace("+", "")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def make_phase0_run_dir(runs_root: Path, run_name: str | None) -> tuple[Path, dict[str, object]]:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    nickname = "d0-p08" if not run_name else str(run_name).strip().replace(" ", "-")
    run_dir = root / f"{timestamp}_{nickname}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp}_{nickname}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, {"timestamp": timestamp, "run_name": nickname, "run_dir": str(run_dir)}


def _make_dynamics_config(
    args: argparse.Namespace,
    *,
    sample_steps: int,
    free_weight: float,
    noise_weight: float,
    horizon_scale: float,
) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=int(args.grid_size),
        alpha=float(args.alpha),
        beta=float(args.beta),
        alpha_eff=float(args.alpha_eff),
        edge_alpha_mode=str(args.edge_alpha_mode),
        horizon_scale=float(horizon_scale),
        num_steps=int(sample_steps),
        limiter_fraction=float(args.stiffness_fraction),
        source_lowfreq_size=min(int(args.source_lowfreq_size), int(args.grid_size)),
        source_blur_sigma=float(args.source_blur_sigma),
        source_uniform_mix=float(args.source_uniform_mix),
        source_concentration=float(args.source_concentration),
        ot_cost_mode=str(args.ot_cost_mode),
        ot_lowres_size=min(int(args.ot_lowres_size), int(args.grid_size)),
        ot_blur_sigma=float(args.ot_blur_sigma),
        ot_com_weight=float(args.ot_com_weight),
        free_weight=float(free_weight),
        noise_weight=float(noise_weight),
        learned_weight=0.0,
        mass_floor=float(args.mass_floor),
        adaptive_sampling=False,
        clip_target=float(args.gate_max_masked_edge_fraction),
        max_substeps=int(args.substeps),
    )


def _synthetic_digit_measures(*, examples_per_class: int, grid_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic digit-like measures for tests and no-data smoke runs."""

    rng = np.random.default_rng(int(seed))
    n = int(grid_size)
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx + 0.5) / float(n)
    y = (yy + 0.5) / float(n)
    images: list[np.ndarray] = []
    labels: list[int] = []

    def blob(cx: float, cy: float, scale: float = 80.0) -> np.ndarray:
        return np.exp(-scale * ((x - cx) ** 2 + (y - cy) ** 2))

    def line_vertical(cx: float) -> np.ndarray:
        return np.exp(-160.0 * (x - cx) ** 2) * np.exp(-8.0 * (y - 0.50) ** 2)

    def line_horizontal(cy: float) -> np.ndarray:
        return np.exp(-160.0 * (y - cy) ** 2) * np.exp(-8.0 * (x - 0.50) ** 2)

    for label in range(10):
        for _rep in range(int(examples_per_class)):
            jitter = 0.015 * rng.normal(size=4)
            if label == 0:
                r = np.sqrt((x - 0.50 - jitter[0]) ** 2 + (y - 0.50 - jitter[1]) ** 2)
                img = np.exp(-250.0 * (r - 0.24) ** 2)
            elif label == 1:
                img = line_vertical(0.50 + jitter[0])
            elif label == 2:
                img = line_horizontal(0.30 + jitter[0]) + line_horizontal(0.70 + jitter[1])
                img += np.exp(-110.0 * (y - (1.0 - x + jitter[2])) ** 2)
            elif label == 3:
                img = blob(0.55 + jitter[0], 0.35 + jitter[1]) + blob(0.55 + jitter[2], 0.65 + jitter[3])
                img += line_vertical(0.68)
            elif label == 4:
                img = line_vertical(0.68 + jitter[0]) + line_horizontal(0.52 + jitter[1])
                img += np.exp(-120.0 * (y - x - jitter[2]) ** 2) * (x < 0.65)
            elif label == 5:
                img = line_horizontal(0.30 + jitter[0]) + line_horizontal(0.68 + jitter[1])
                img += blob(0.38 + jitter[2], 0.55 + jitter[3])
            elif label == 6:
                r = np.sqrt((x - 0.48 - jitter[0]) ** 2 + (y - 0.60 - jitter[1]) ** 2)
                img = np.exp(-220.0 * (r - 0.20) ** 2) + line_vertical(0.36 + jitter[2])
            elif label == 7:
                img = line_horizontal(0.30 + jitter[0])
                img += np.exp(-130.0 * (y - (1.15 - x) - jitter[1]) ** 2)
            elif label == 8:
                img = blob(0.50 + jitter[0], 0.35 + jitter[1]) + blob(0.50 + jitter[2], 0.67 + jitter[3])
            else:
                r = np.sqrt((x - 0.50 - jitter[0]) ** 2 + (y - 0.42 - jitter[1]) ** 2)
                img = np.exp(-250.0 * (r - 0.18) ** 2) + line_vertical(0.65 + jitter[2])
            images.append(np.maximum(img, 0.0))
            labels.append(label)
    measures = normalize_images_to_measures(np.asarray(images, dtype=np.float64))
    return np.asarray(measures, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def load_phase0_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if bool(args.synthetic_data):
        return _synthetic_digit_measures(
            examples_per_class=int(args.synthetic_examples_per_class),
            grid_size=int(args.grid_size),
            seed=int(args.seed),
        )
    dataset = load_mnist_measure_dataset(
        args.data_root,
        max_train=args.max_train,
        examples_per_class=args.examples_per_class,
        download=bool(args.download),
        seed=int(args.seed),
    )
    return np.asarray(dataset.train_images, dtype=np.float64), np.asarray(dataset.train_labels, dtype=np.int64)


def _sample_lambda_mixed_data(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    lambda_mix: float,
    grid_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(grid_size)
    flat = np.asarray(images, dtype=np.float64).reshape(images.shape[0], n * n)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    class_indices = [np.flatnonzero(labels_arr == digit) for digit in range(10)]
    requested = rng.integers(0, 10, size=int(count), dtype=np.int64)
    chosen: list[int] = []
    all_idx = np.arange(labels_arr.shape[0], dtype=np.int64)
    for digit in requested:
        candidates = class_indices[int(digit)]
        idx = int(rng.choice(all_idx if candidates.size == 0 else candidates))
        chosen.append(idx)
    source_indices = np.asarray(chosen, dtype=np.int64)
    states = flat[source_indices].copy()
    uniform = np.full((1, n * n), 1.0 / float(n * n), dtype=np.float64)
    states = (1.0 - float(lambda_mix)) * states + float(lambda_mix) * uniform
    states = np.maximum(states, 0.0)
    states /= np.maximum(states.sum(axis=1, keepdims=True), 1e-30)
    return states.astype(np.float32), requested.astype(np.int64), source_indices


def _sample_dirichlet_initial_states(
    *,
    count: int,
    grid_size: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a Dirichlet-start bank for the separate symmetry test."""

    n = int(grid_size)
    states = _dirichlet_samples(int(count), n * n, float(alpha), rng).astype(np.float32)
    # Labels are irrelevant for symmetry, but keeping the shape/type stable makes
    # previews, prior-bank files, and downstream readers compatible.
    labels = rng.integers(0, 10, size=int(count), dtype=np.int64)
    source_indices = np.full(int(count), -1, dtype=np.int64)
    return states, labels, source_indices


def two_sample_state_diagnostics(
    states_a: np.ndarray,
    states_b: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    prefix: str,
    max_samples: int = 256,
) -> dict[str, float | int]:
    """Compare two empirical state banks without assuming either is Dirichlet."""

    n = int(config.grid_size)
    dim = n * n
    a = np.asarray(states_a, dtype=np.float64).reshape(-1, dim)
    b = np.asarray(states_b, dtype=np.float64).reshape(-1, dim)
    m = min(int(max_samples), a.shape[0], b.shape[0])
    if m <= 0:
        return {f"{prefix}_samples": 0, f"{prefix}_quantile_distance": float("nan"), f"{prefix}_feature_mmd": float("nan")}
    a_eval = a[:m]
    b_eval = b[:m]
    mean_mass = 1.0 / float(dim)
    fa = _lowres_features_np(a_eval.reshape(-1, n, n), config)
    fb = _lowres_features_np(b_eval.reshape(-1, n, n), config)
    return {
        f"{prefix}_samples": int(m),
        f"{prefix}_quantile_distance": _quantile_distance(a_eval, b_eval, mean_mass=mean_mass),
        f"{prefix}_feature_mmd": _rbf_mmd2(fa, fb),
    }


def _checkpoint_steps(steps: int, fractions: Sequence[float]) -> list[int]:
    return sorted({int(round(float(frac) * int(steps))) for frac in fractions} | {0, int(steps)})


def _metric_steps(steps: int, bins: int, preview_fractions: Sequence[float]) -> list[int]:
    bins = max(1, int(bins))
    fractions = {float(k) / float(bins) for k in range(bins + 1)}
    fractions.update(float(x) for x in preview_fractions)
    return _checkpoint_steps(int(steps), sorted(fractions))


def _mean_pixel_correlation(states: np.ndarray, initial: np.ndarray) -> float:
    x = np.asarray(states, dtype=np.float64).reshape(states.shape[0], -1)
    y = np.asarray(initial, dtype=np.float64).reshape(initial.shape[0], -1)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    corr = np.sum(x * y, axis=1) / np.maximum(denom, 1e-30)
    return float(np.mean(corr))


def _dirichlet_samples(num: int, dim: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    samples = rng.gamma(shape=float(alpha), scale=1.0, size=(int(num), int(dim))).astype(np.float64)
    samples /= np.maximum(samples.sum(axis=1, keepdims=True), 1e-300)
    return samples


def _global_quantile_curve(samples: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    flat = np.asarray(samples, dtype=np.float64).reshape(-1)
    return np.quantile(flat, quantiles)


def _quantile_distance(a: np.ndarray, b: np.ndarray, *, mean_mass: float, num_quantiles: int = 101) -> float:
    qs = np.linspace(0.0, 1.0, int(num_quantiles))
    ca = _global_quantile_curve(a, qs)
    cb = _global_quantile_curve(b, qs)
    return float(np.sqrt(np.mean((ca - cb) ** 2)) / max(float(mean_mass), 1e-30))


def _rbf_mmd2(features_a: np.ndarray, features_b: np.ndarray) -> float:
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    pooled = np.concatenate([a, b], axis=0)
    # Robust median heuristic on a capped subset.
    cap = min(256, pooled.shape[0])
    sub = pooled[:cap]
    sq = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    positive = sq[sq > 0.0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    sigma2 = max(sigma2, 1e-12)

    def k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d2 = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * d2 / sigma2)

    return float(k(a, a).mean() + k(b, b).mean() - 2.0 * k(a, b).mean())


def stationarity_diagnostics(
    terminal_states: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    rng: np.random.Generator,
    max_samples: int = 256,
    calibration_reps: int = 3,
    quantile_multiplier: float = 3.0,
    mmd_multiplier: float = 3.0,
    quantile_floor: float = 0.02,
    mmd_floor: float = 1e-4,
) -> dict[str, float | int]:
    """Compare the terminal bank against exact Dirichlet(alpha_eff) samples."""

    n = int(config.grid_size)
    dim = n * n
    alpha = float(edge_alpha_value(config))
    states = np.asarray(terminal_states, dtype=np.float64).reshape(-1, dim)
    if states.shape[0] > int(max_samples):
        idx = rng.choice(states.shape[0], size=int(max_samples), replace=False)
        states_eval = states[idx]
    else:
        states_eval = states
    m = int(states_eval.shape[0])
    ref = _dirichlet_samples(m, dim, alpha, rng)
    mean_mass = 1.0 / float(dim)
    qdist = _quantile_distance(states_eval, ref, mean_mass=mean_mass)
    f_states = _lowres_features_np(states_eval.reshape(-1, n, n), config)
    f_ref = _lowres_features_np(ref.reshape(-1, n, n), config)
    mmd = _rbf_mmd2(f_states, f_ref)

    q_base: list[float] = []
    mmd_base: list[float] = []
    reps = max(1, int(calibration_reps))
    for _ in range(reps):
        a = _dirichlet_samples(m, dim, alpha, rng)
        b = _dirichlet_samples(m, dim, alpha, rng)
        q_base.append(_quantile_distance(a, b, mean_mass=mean_mass))
        fa = _lowres_features_np(a.reshape(-1, n, n), config)
        fb = _lowres_features_np(b.reshape(-1, n, n), config)
        mmd_base.append(_rbf_mmd2(fa, fb))
    q_base_mean = float(np.mean(q_base))
    mmd_base_mean = float(np.mean(mmd_base))
    q_threshold = max(float(quantile_floor), float(quantile_multiplier) * q_base_mean)
    mmd_threshold = max(float(mmd_floor), float(mmd_multiplier) * mmd_base_mean)
    q_pass = qdist <= q_threshold
    mmd_pass = mmd <= mmd_threshold
    return {
        "stationarity_alpha": float(alpha),
        "stationarity_eval_samples": int(m),
        "stationarity_quantile_distance": float(qdist),
        "stationarity_quantile_baseline_mean": float(q_base_mean),
        "stationarity_quantile_threshold": float(q_threshold),
        "stationarity_feature_mmd": float(mmd),
        "stationarity_feature_mmd_baseline_mean": float(mmd_base_mean),
        "stationarity_feature_mmd_threshold": float(mmd_threshold),
        "stationarity_pass_quantile": int(q_pass),
        "stationarity_pass_feature_mmd": int(mmd_pass),
        "stationarity_pass": int(q_pass and mmd_pass),
    }


def _support_escape_metrics(
    states_np: np.ndarray,
    initial_states_np: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    lambda_mix: float,
) -> dict[str, float]:
    """Diagnostics for whether mass actually leaves the initial digit support."""

    n = int(config.grid_size)
    dim = int(n * n)
    current = np.asarray(states_np, dtype=np.float64).reshape(-1, dim)
    initial = np.asarray(initial_states_np, dtype=np.float64).reshape(current.shape[0], dim)
    uniform_floor = float(lambda_mix) / float(dim)
    support_eps = max(10.0 * float(config.mass_floor), 1e-12)
    foreground = initial > (uniform_floor + support_eps)
    # Synthetic or heavily anti-aliased examples can make the lambda-floor rule
    # select no pixels.  Fall back to above-uniform pixels so all diagnostics are
    # finite and meaningful.
    empty = ~foreground.any(axis=1)
    if bool(np.any(empty)):
        foreground[empty] = initial[empty] > (1.0 / float(dim))
    background = ~foreground
    diff = current - initial

    fg_count = np.maximum(foreground.sum(axis=1), 1)
    bg_count = np.maximum(background.sum(axis=1), 1)
    initial_fg = (initial * foreground).sum(axis=1)
    terminal_fg = (current * foreground).sum(axis=1)
    initial_bg = (initial * background).sum(axis=1)
    terminal_bg = (current * background).sum(axis=1)
    bg_l1 = (np.abs(diff) * background).sum(axis=1)
    fg_l1 = (np.abs(diff) * foreground).sum(axis=1)
    bg_positive_gain = (np.maximum(diff, 0.0) * background).sum(axis=1)
    fg_mass_loss = np.maximum(initial_fg - terminal_fg, 0.0)
    change_threshold = max(10.0 * float(config.mass_floor), 0.05 * uniform_floor, 1e-9)
    changed = np.abs(diff) > change_threshold
    return {
        "support_uniform_floor": float(uniform_floor),
        "support_change_threshold": float(change_threshold),
        "foreground_pixel_fraction_mean": float(np.mean(fg_count / float(dim))),
        "initial_foreground_mass_mean": float(np.mean(initial_fg)),
        "terminal_foreground_mass_mean": float(np.mean(terminal_fg)),
        "initial_background_mass_mean": float(np.mean(initial_bg)),
        "terminal_background_mass_mean": float(np.mean(terminal_bg)),
        "foreground_mass_retention_mean": float(np.mean(terminal_fg / np.maximum(initial_fg, 1e-30))),
        "background_mass_gain_mean": float(np.mean(terminal_bg - initial_bg)),
        "support_escape_mass_mean": float(np.mean(bg_positive_gain)),
        "foreground_mass_loss_mean": float(np.mean(fg_mass_loss)),
        "background_l1_mean": float(np.mean(bg_l1)),
        "foreground_l1_mean": float(np.mean(fg_l1)),
        "background_l1_per_pixel_mean": float(np.mean(bg_l1 / bg_count)),
        "foreground_l1_per_pixel_mean": float(np.mean(fg_l1 / fg_count)),
        "fraction_pixels_changed_above_floor": float(np.mean(changed)),
        "background_fraction_pixels_changed_above_floor": float(np.mean((changed & background).sum(axis=1) / bg_count)),
        "foreground_fraction_pixels_changed_above_floor": float(np.mean((changed & foreground).sum(axis=1) / fg_count)),
    }


def _state_metrics(
    states: Tensor,
    initial_states_np: np.ndarray,
    initial_features: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    lambda_mix: float,
    theta_mask_min: float,
    step: int,
    horizon: float,
    dt: float,
    last_step_masked_edge_fraction: float,
    cumulative_masked_edge_fraction: float,
    max_time_bin_masked_edge_fraction: float,
    last_step_drift_limited_fraction: float = 0.0,
    last_step_noise_limited_fraction: float = 0.0,
    last_step_nonfinite_fraction: float = 0.0,
    last_step_floor_correction_l1: float = 0.0,
    last_step_renorm_correction_l1: float = 0.0,
    last_step_mobility_weighted_masked_edge_fraction: float = 0.0,
    cumulative_mobility_weighted_masked_edge_fraction: float = 0.0,
    max_time_bin_mobility_weighted_masked_edge_fraction: float = 0.0,
    last_step_noise_energy_weighted_masked_edge_fraction: float = 0.0,
    cumulative_noise_energy_weighted_masked_edge_fraction: float = 0.0,
    max_time_bin_noise_energy_weighted_masked_edge_fraction: float = 0.0,
    substeps: int,
) -> dict[str, float | int | str]:
    n = int(config.grid_size)
    states_np = states.detach().cpu().numpy().astype(np.float64).reshape(-1, n, n)
    flat_np = states_np.reshape(states_np.shape[0], -1)
    features = _lowres_features_np(states_np, config)
    feature_dist2 = np.sum((features - initial_features) ** 2, axis=1)
    entropy = -(flat_np * np.log(np.maximum(flat_np, 1e-30))).sum(axis=1)
    expected_entropy = expected_symmetric_dirichlet_entropy(config)
    theta = harmonic_mobility_channels(states, config)
    frozen = (theta <= float(theta_mask_min)).detach().float().mean()
    tv = image_total_variation(states, grid_size=n)
    checker = checkerboard_energy_torch(states, grid_size=n)
    support = _support_escape_metrics(states_np, initial_states_np, config, lambda_mix=float(lambda_mix))
    return {
        "step": int(step),
        "time": float(step) * float(dt),
        "tau": max(float(horizon) - float(step) * float(dt), 0.0),
        "feature_dist2_mean": float(np.mean(feature_dist2)),
        "feature_dist2_median": float(np.median(feature_dist2)),
        "pixel_corr_mean": _mean_pixel_correlation(flat_np, initial_states_np),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
        "entropy_fraction_of_uniform": float(np.mean(entropy) / math.log(n * n)),
        "expected_stationary_entropy": float(expected_entropy),
        "entropy_fraction_of_stationary": float(np.mean(entropy) / max(float(expected_entropy), 1e-30)),
        "total_variation_mean": float(tv.detach().cpu()),
        "checkerboard_energy": float(checker.detach().cpu()),
        "frozen_edge_fraction": float(frozen.detach().cpu()),
        "last_step_masked_edge_fraction": float(last_step_masked_edge_fraction),
        "cumulative_masked_edge_fraction": float(cumulative_masked_edge_fraction),
        "max_time_bin_masked_edge_fraction": float(max_time_bin_masked_edge_fraction),
        "last_step_drift_limited_fraction": float(last_step_drift_limited_fraction),
        "last_step_noise_limited_fraction": float(last_step_noise_limited_fraction),
        "last_step_nonfinite_fraction": float(last_step_nonfinite_fraction),
        "last_step_floor_correction_l1": float(last_step_floor_correction_l1),
        "last_step_renorm_correction_l1": float(last_step_renorm_correction_l1),
        "last_step_mobility_weighted_masked_edge_fraction": float(last_step_mobility_weighted_masked_edge_fraction),
        "cumulative_mobility_weighted_masked_edge_fraction": float(cumulative_mobility_weighted_masked_edge_fraction),
        "max_time_bin_mobility_weighted_masked_edge_fraction": float(max_time_bin_mobility_weighted_masked_edge_fraction),
        "last_step_noise_energy_weighted_masked_edge_fraction": float(last_step_noise_energy_weighted_masked_edge_fraction),
        "cumulative_noise_energy_weighted_masked_edge_fraction": float(cumulative_noise_energy_weighted_masked_edge_fraction),
        "max_time_bin_noise_energy_weighted_masked_edge_fraction": float(max_time_bin_noise_energy_weighted_masked_edge_fraction),
        # Deprecated compatibility columns: clipping is no longer used by P0.8.
        "last_step_clip_fraction": 0.0,
        "cumulative_clip_fraction": 0.0,
        "substeps": int(substeps),
        **support,
    }


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return image / max(float(image.max()), 1e-12)


def _balanced_preview_indices(labels: np.ndarray, max_images: int) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_arr.size <= int(max_images):
        return np.arange(labels_arr.size, dtype=np.int64)
    per_label = max(1, int(math.ceil(int(max_images) / 10.0)))
    chosen: list[int] = []
    for digit in range(10):
        idx = np.flatnonzero(labels_arr == digit)
        chosen.extend(int(i) for i in idx[:per_label])
    if not chosen:
        return np.arange(min(int(max_images), labels_arr.size), dtype=np.int64)
    return np.asarray(chosen[: int(max_images)], dtype=np.int64)


def save_noising_preview_panel(
    checkpoint_states: dict[int, np.ndarray],
    labels: np.ndarray,
    output_path: Path,
    *,
    grid_size: int,
    max_images: int,
) -> None:
    """Save a fixed-scale preview panel.

    Earlier diagnostics normalized every cell image by its own maximum, which can
    visually preserve digit contrast even when mass is flattening.  This panel
    uses one shared ``vmax`` across all selected checkpoints and samples.
    """

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to save a noising preview panel") from exc

    steps = sorted(checkpoint_states)
    idx = _balanced_preview_indices(labels, int(max_images))
    cols = int(idx.size)
    rows = len(steps)
    if cols <= 0 or rows <= 0:
        return
    selected_values: list[np.ndarray] = []
    for step in steps:
        arr = np.asarray(checkpoint_states[step], dtype=np.float64).reshape(-1, grid_size, grid_size)
        selected_values.append(arr[idx])
    stacked = np.concatenate([x.reshape(-1) for x in selected_values])
    vmax = max(float(np.quantile(stacked, 0.995)), float(stacked.max()), 1e-12)
    fig, axes = plt.subplots(rows, cols, figsize=(1.25 * cols, 1.35 * rows), squeeze=False)
    for row, step in enumerate(steps):
        arr = np.asarray(checkpoint_states[step], dtype=np.float64).reshape(-1, grid_size, grid_size)
        for col, sample_idx in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(arr[int(sample_idx)], cmap="gray", interpolation="nearest", vmin=0.0, vmax=vmax)
            if row == 0:
                ax.set_title(str(int(labels[int(sample_idx)])), fontsize=8)
            if col == 0:
                ax.set_ylabel(f"k={step}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.12)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_noising_delta_panel(
    checkpoint_states: dict[int, np.ndarray],
    labels: np.ndarray,
    output_path: Path,
    *,
    grid_size: int,
    max_images: int,
) -> None:
    """Save signed differences from k=0 so support movement is visible."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to save a noising delta panel") from exc

    steps = sorted(checkpoint_states)
    if 0 not in checkpoint_states:
        return
    idx = _balanced_preview_indices(labels, int(max_images))
    cols = int(idx.size)
    rows = len(steps)
    if cols <= 0 or rows <= 0:
        return
    initial = np.asarray(checkpoint_states[0], dtype=np.float64).reshape(-1, grid_size, grid_size)
    diffs: list[np.ndarray] = []
    for step in steps:
        arr = np.asarray(checkpoint_states[step], dtype=np.float64).reshape(-1, grid_size, grid_size)
        diffs.append(arr[idx] - initial[idx])
    flat_abs = np.concatenate([np.abs(x).reshape(-1) for x in diffs])
    vmax = max(float(np.quantile(flat_abs, 0.995)), 1e-12)
    fig, axes = plt.subplots(rows, cols, figsize=(1.25 * cols, 1.35 * rows), squeeze=False)
    for row, step in enumerate(steps):
        arr = np.asarray(checkpoint_states[step], dtype=np.float64).reshape(-1, grid_size, grid_size)
        diff = arr - initial
        for col, sample_idx in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(diff[int(sample_idx)], cmap="coolwarm", interpolation="nearest", vmin=-vmax, vmax=vmax)
            if row == 0:
                ax.set_title(str(int(labels[int(sample_idx)])), fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Δ k={step}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.12)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_rate_schedule(
    sample_steps: int,
    *,
    mode: str,
    tau_eff: float | None,
    constant_rate: float | None,
    ramp: str,
    ramp_ratio: float,
    rate_min: float | None,
    rate_max: float | None,
    horizon: float | None = None,
    time_change_mode: str = "integral",
) -> np.ndarray:
    """Build the faithful reference rate schedule.

    ``time_change_mode='integral'`` makes ``tau_eff`` a total effective time:
    with ``dt = horizon / sample_steps``, the schedule satisfies
    ``sum_k rate_k * dt = tau_eff``.  ``time_change_mode='rate'`` keeps the old
    P0.6 behavior where ``tau_eff`` is already a raw rate.  Explicit
    ``rate_min``/``rate_max`` are always interpreted as raw rates.
    """

    if str(mode) != "faithful":
        return np.full(int(sample_steps), float("nan"), dtype=np.float64)
    k = int(sample_steps)
    if k <= 0:
        raise ValueError("sample_steps must be positive")
    if rate_min is not None and rate_max is not None:
        if str(ramp) == "geometric":
            schedule = np.geomspace(float(rate_min), float(rate_max), k)
        else:
            schedule = np.linspace(float(rate_min), float(rate_max), k)
        return schedule.astype(np.float64)
    if tau_eff is not None:
        target = float(tau_eff)
        if target < 0.0 or not math.isfinite(target):
            raise ValueError("tau_eff must be finite and non-negative")
        if str(time_change_mode) == "integral":
            if horizon is None or float(horizon) <= 0.0 or not math.isfinite(float(horizon)):
                raise ValueError("horizon must be positive and finite for integrated time-change mode")
            target = target / float(horizon)
        elif str(time_change_mode) != "rate":
            raise ValueError("time_change_mode must be 'integral' or 'rate'")
    else:
        target = float(constant_rate if constant_rate is not None else 0.0)
        if target < 0.0 or not math.isfinite(target):
            raise ValueError("reference rates must be finite and non-negative")
    if str(ramp) == "none" or k == 1:
        return np.full(k, target, dtype=np.float64)
    ratio = max(float(ramp_ratio), 1.0 + 1e-12)
    raw = np.geomspace(1.0 / ratio, 1.0, k)
    raw /= max(float(raw.mean()), 1e-30)
    return (target * raw).astype(np.float64)


def rate_schedule_effective_time(rate_schedule: np.ndarray, horizon: float, *, mode: str) -> float:
    """Return ``sum_k rate_k * dt`` for a faithful schedule."""

    if str(mode) != "faithful":
        return float("nan")
    schedule = np.asarray(rate_schedule, dtype=np.float64).reshape(-1)
    if schedule.size == 0 or np.isnan(schedule).all():
        return float("nan")
    dt = float(horizon) / float(schedule.size)
    return float(np.nansum(schedule) * dt)


def effective_time_integral(rate_schedule: np.ndarray, *, dt: float) -> float:
    """Return the integrated faithful time-change ``sum_k rate_k * dt``."""

    rates = np.asarray(rate_schedule, dtype=np.float64).reshape(-1)
    if rates.size == 0 or not np.isfinite(rates).any():
        return float("nan")
    return float(np.nansum(rates) * float(dt))

def run_forward_noising_single(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    lambda_mix: float,
    free_weight: float,
    noise_weight: float,
    init_law: str = "data",
    sample_steps: int,
    num_paths: int,
    reference_scale_mode: str = "independent",
    reference_rate: float | None = None,
    tau_eff: float | None = None,
    time_change_mode: str = "rate",
    rate_schedule: np.ndarray | None = None,
    substeps: int | None = None,
    adaptive_substeps: bool = False,
    adaptive_min_substeps: int | None = None,
    adaptive_max_substeps: int | None = None,
    adaptive_substep_target: float = 0.20,
    adaptive_substep_quantile: float = 0.99,
    batch_size: int = 128,
    theta_mask_min: float = 1e-12,
    preview_fractions: Sequence[float] = (0.0, 0.5, 1.0),
    metric_bins: int = 16,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
    show_progress: bool = True,
    stationarity_max_samples: int = 256,
    stationarity_calibration_reps: int = 3,
    stationarity_quantile_multiplier: float = 3.0,
    stationarity_mmd_multiplier: float = 3.0,
    stationarity_quantile_floor: float = 0.02,
    stationarity_mmd_floor: float = 1e-4,
) -> Phase0SingleResult:
    """Run one P0.8 schedule and return metrics plus terminal/prior states."""

    if int(num_paths) <= 0:
        raise ValueError("num_paths must be positive")
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    n = int(config.grid_size)
    horizon = natural_horizon(config)
    dt = float(horizon) / float(sample_steps)
    fixed_substeps = int(config.max_substeps if substeps is None else substeps)
    if fixed_substeps <= 0:
        raise ValueError("substeps must be positive")
    # P0.7-fast disables the optional adaptive-substep controller.  It made
    # Phase-0 sweeps unusably slow by doing an extra full-edge diagnostic pass
    # every outer step and often raising the actual integrator substeps to the
    # configured cap.  Keep the arguments accepted for command compatibility,
    # but always use the explicit fixed --substeps value.
    adaptive_requested = bool(adaptive_substeps)
    adaptive_substeps = False
    adaptive_cap = int(fixed_substeps)
    if rate_schedule is None:
        rate_schedule_arr = np.full(int(sample_steps), float(reference_rate if reference_rate is not None else free_weight), dtype=np.float64)
    else:
        rate_schedule_arr = np.asarray(rate_schedule, dtype=np.float64).reshape(-1)
        if rate_schedule_arr.shape[0] != int(sample_steps):
            raise ValueError("rate_schedule must have length sample_steps")
    preview_steps = _checkpoint_steps(int(sample_steps), preview_fractions)
    metric_step_set = set(_metric_steps(int(sample_steps), int(metric_bins), preview_fractions))
    preview_step_set = set(preview_steps)

    init_mode = str(init_law).lower().replace("_", "-")
    if init_mode == "dirichlet":
        initial_np, requested_labels, source_indices = _sample_dirichlet_initial_states(
            count=int(num_paths),
            grid_size=n,
            alpha=float(edge_alpha_value(config)),
            rng=rng,
        )
    elif init_mode == "data":
        initial_np, requested_labels, source_indices = _sample_lambda_mixed_data(
            images,
            labels,
            count=int(num_paths),
            lambda_mix=float(lambda_mix),
            grid_size=n,
            rng=rng,
        )
    else:
        raise ValueError("init_law must be 'data' or 'dirichlet'")
    initial_features = _lowres_features_np(initial_np.reshape(-1, n, n), config)
    states = torch.as_tensor(initial_np, dtype=torch.float32, device=device)

    checkpoint_states: dict[int, np.ndarray] = {}
    metrics: list[dict[str, float | int | str]] = []
    total_masked = 0
    total_proposed = 0
    total_noise_stiff = 0
    total_drift_stiff = 0
    total_overflow = 0
    total_drift_limited = 0
    total_noise_limited = 0
    total_nonfinite = 0
    total_floor_correction_l1 = 0.0
    total_renorm_correction_l1 = 0.0
    total_mobility_weight = 0.0
    total_limited_mobility_weight = 0.0
    total_noise_energy = 0.0
    total_limited_noise_energy = 0.0
    last_masked_fraction = 0.0
    max_time_bin_masked = 0.0
    max_time_bin_mobility_weighted_masked = 0.0
    max_time_bin_noise_energy_weighted_masked = 0.0
    total_substeps_used = 0
    max_substeps_used = int(fixed_substeps)
    steps_at_substep_cap = 0
    adaptive_required_substeps_max = int(fixed_substeps)
    adaptive_drift_ratio_q_max = 0.0
    adaptive_noise_ratio_q_max = 0.0

    if 0 in preview_step_set:
        checkpoint_states[0] = states.detach().cpu().numpy().copy()
    if 0 in metric_step_set:
        row0 = _state_metrics(
            states,
            initial_np,
            initial_features,
            config,
            theta_mask_min=float(theta_mask_min),
            step=0,
            horizon=horizon,
            dt=dt,
            last_step_masked_edge_fraction=0.0,
            cumulative_masked_edge_fraction=0.0,
            max_time_bin_masked_edge_fraction=0.0,
            substeps=fixed_substeps,
            lambda_mix=float(lambda_mix),
        )
        row0.update(
            {
                "adaptive_substeps_enabled": 0,
                "adaptive_substeps_requested_ignored": int(bool(adaptive_requested)),
                "adaptive_step_substeps": int(fixed_substeps),
                "adaptive_mean_substeps_so_far": 0.0,
                "adaptive_required_substeps": float(fixed_substeps),
                "adaptive_drift_ratio_q": 0.0,
                "adaptive_noise_ratio_q": 0.0,
                "last_step_drift_limited_fraction": 0.0,
                "last_step_noise_limited_fraction": 0.0,
                "last_step_nonfinite_fraction": 0.0,
                "cumulative_drift_limited_edge_fraction": 0.0,
                "cumulative_noise_limited_edge_fraction": 0.0,
                "cumulative_nonfinite_edge_fraction": 0.0,
                "mean_floor_correction_l1_per_step": 0.0,
                "mean_renorm_correction_l1_per_step": 0.0,
                "last_step_mobility_weighted_masked_edge_fraction": 0.0,
                "cumulative_mobility_weighted_masked_edge_fraction": 0.0,
                "max_time_bin_mobility_weighted_masked_edge_fraction": 0.0,
                "last_step_noise_energy_weighted_masked_edge_fraction": 0.0,
                "cumulative_noise_energy_weighted_masked_edge_fraction": 0.0,
                "max_time_bin_noise_energy_weighted_masked_edge_fraction": 0.0,
                "valid_innovation_fraction": 1.0,
                "valid_innovation_mobility_fraction": 1.0,
                "valid_innovation_noise_energy_fraction": 1.0,
            }
        )
        metrics.append(row0)

    step_iter: Iterable[int] = range(1, int(sample_steps) + 1)
    bar = _progress(list(step_iter), total=int(sample_steps), desc="P0.8 forward noising", disable=not show_progress)
    for step in bar:
        next_chunks: list[Tensor] = []
        masked_step = 0
        proposed_step = 0
        noise_stiff_step = 0
        drift_stiff_step = 0
        overflow_step = 0
        drift_limited_step = 0
        noise_limited_step = 0
        nonfinite_step = 0
        floor_correction_l1_step = 0.0
        renorm_correction_l1_step = 0.0
        mobility_weight_step = 0.0
        limited_mobility_weight_step = 0.0
        noise_energy_step = 0.0
        limited_noise_energy_step = 0.0
        if str(reference_scale_mode) == "faithful":
            step_rate = float(rate_schedule_arr[int(step) - 1])
            step_free_weight = step_rate
            step_noise_weight = math.sqrt(max(step_rate, 0.0))
        else:
            step_rate = float("nan")
            step_free_weight = float(free_weight)
            step_noise_weight = float(noise_weight)

        step_substeps = int(fixed_substeps)
        adaptive_diag = {
            "chosen_substeps": float(step_substeps),
            "required_substeps_unclipped": float(step_substeps),
            "drift_ratio_q": 0.0,
            "noise_ratio_q": 0.0,
            "hit_substep_cap": 0.0,
        }
        total_substeps_used += int(step_substeps)
        max_substeps_used = max(max_substeps_used, int(step_substeps))
        adaptive_required_substeps_max = max(
            adaptive_required_substeps_max,
            int(math.ceil(float(adaptive_diag.get("required_substeps_unclipped", step_substeps)))),
        )
        adaptive_drift_ratio_q_max = max(adaptive_drift_ratio_q_max, float(adaptive_diag.get("drift_ratio_q", 0.0)))
        adaptive_noise_ratio_q_max = max(adaptive_noise_ratio_q_max, float(adaptive_diag.get("noise_ratio_q", 0.0)))
        for start in range(0, int(num_paths), int(batch_size)):
            stop = min(int(num_paths), start + int(batch_size))
            chunk = states[start:stop]
            step_result = masked_reference_free_step_torch(
                chunk,
                dt,
                config,
                free_weight=step_free_weight,
                noise_weight=step_noise_weight,
                substeps=int(step_substeps),
                stiffness_fraction=float(config.limiter_fraction),
                return_innovations=False,
            )
            next_chunks.append(step_result.states)
            masked_step += int(step_result.masked_edges)
            proposed_step += int(step_result.proposed_edges)
            noise_stiff_step += int(step_result.noise_stiff_edges)
            drift_stiff_step += int(step_result.drift_stiff_edges)
            overflow_step += int(step_result.overflow_edges)
            drift_limited_step += int(getattr(step_result, "drift_limited_edges", step_result.drift_stiff_edges))
            noise_limited_step += int(getattr(step_result, "noise_limited_edges", step_result.noise_stiff_edges))
            nonfinite_step += int(getattr(step_result, "nonfinite_edges", 0))
            floor_correction_l1_step += float(getattr(step_result, "floor_correction_l1", 0.0))
            renorm_correction_l1_step += float(getattr(step_result, "renorm_correction_l1", 0.0))
            mobility_weight_step += float(getattr(step_result, "mobility_weight_sum", 0.0))
            limited_mobility_weight_step += float(getattr(step_result, "limited_mobility_weight_sum", 0.0))
            noise_energy_step += float(getattr(step_result, "noise_energy_sum", 0.0))
            limited_noise_energy_step += float(getattr(step_result, "limited_noise_energy_sum", 0.0))
        states = torch.cat(next_chunks, dim=0)
        total_masked += int(masked_step)
        total_proposed += int(proposed_step)
        total_noise_stiff += int(noise_stiff_step)
        total_drift_stiff += int(drift_stiff_step)
        total_overflow += int(overflow_step)
        total_drift_limited += int(drift_limited_step)
        total_noise_limited += int(noise_limited_step)
        total_nonfinite += int(nonfinite_step)
        total_floor_correction_l1 += float(floor_correction_l1_step)
        total_renorm_correction_l1 += float(renorm_correction_l1_step)
        total_mobility_weight += float(mobility_weight_step)
        total_limited_mobility_weight += float(limited_mobility_weight_step)
        total_noise_energy += float(noise_energy_step)
        total_limited_noise_energy += float(limited_noise_energy_step)
        last_masked_fraction = 0.0 if proposed_step == 0 else float(masked_step) / float(proposed_step)
        last_mobility_weighted_masked = 0.0 if mobility_weight_step <= 0.0 else float(limited_mobility_weight_step) / float(mobility_weight_step)
        last_noise_energy_weighted_masked = 0.0 if noise_energy_step <= 0.0 else float(limited_noise_energy_step) / float(noise_energy_step)
        cumulative_masked = 0.0 if total_proposed == 0 else float(total_masked) / float(total_proposed)
        cumulative_mobility_weighted_masked = 0.0 if total_mobility_weight <= 0.0 else float(total_limited_mobility_weight) / float(total_mobility_weight)
        cumulative_noise_energy_weighted_masked = 0.0 if total_noise_energy <= 0.0 else float(total_limited_noise_energy) / float(total_noise_energy)
        max_time_bin_masked = max(max_time_bin_masked, last_masked_fraction)
        max_time_bin_mobility_weighted_masked = max(max_time_bin_mobility_weighted_masked, last_mobility_weighted_masked)
        max_time_bin_noise_energy_weighted_masked = max(max_time_bin_noise_energy_weighted_masked, last_noise_energy_weighted_masked)
        if hasattr(bar, "set_postfix"):
            with torch.no_grad():
                entropy = float((-(states.clamp_min(1e-30) * states.clamp_min(1e-30).log()).sum(dim=1)).mean().cpu())
            bar.set_postfix(k=int(step), mask=cumulative_masked, sub=int(step_substeps), H=entropy, rate=step_rate)
        if int(step) in preview_step_set:
            checkpoint_states[int(step)] = states.detach().cpu().numpy().copy()
        if int(step) in metric_step_set:
            row = _state_metrics(
                states,
                initial_np,
                initial_features,
                config,
                lambda_mix=float(lambda_mix),
                theta_mask_min=float(theta_mask_min),
                step=int(step),
                horizon=horizon,
                dt=dt,
                last_step_masked_edge_fraction=last_masked_fraction,
                cumulative_masked_edge_fraction=cumulative_masked,
                max_time_bin_masked_edge_fraction=max_time_bin_masked,
                last_step_mobility_weighted_masked_edge_fraction=last_mobility_weighted_masked,
                cumulative_mobility_weighted_masked_edge_fraction=cumulative_mobility_weighted_masked,
                max_time_bin_mobility_weighted_masked_edge_fraction=max_time_bin_mobility_weighted_masked,
                last_step_noise_energy_weighted_masked_edge_fraction=last_noise_energy_weighted_masked,
                cumulative_noise_energy_weighted_masked_edge_fraction=cumulative_noise_energy_weighted_masked,
                max_time_bin_noise_energy_weighted_masked_edge_fraction=max_time_bin_noise_energy_weighted_masked,
                substeps=int(step_substeps),
            )
            row.update(
                {
                    "adaptive_substeps_enabled": 0,
                    "adaptive_substeps_requested_ignored": int(bool(adaptive_requested)),
                    "adaptive_step_substeps": int(step_substeps),
                    "adaptive_mean_substeps_so_far": float(total_substeps_used) / float(max(1, int(step))),
                    "adaptive_required_substeps": float(adaptive_diag.get("required_substeps_unclipped", step_substeps)),
                    "adaptive_drift_ratio_q": float(adaptive_diag.get("drift_ratio_q", 0.0)),
                    "adaptive_noise_ratio_q": float(adaptive_diag.get("noise_ratio_q", 0.0)),
                    "last_step_drift_limited_fraction": 0.0 if proposed_step == 0 else float(drift_limited_step) / float(proposed_step),
                    "last_step_noise_limited_fraction": 0.0 if proposed_step == 0 else float(noise_limited_step) / float(proposed_step),
                    "last_step_nonfinite_fraction": 0.0 if proposed_step == 0 else float(nonfinite_step) / float(proposed_step),
                    "last_step_floor_correction_l1": float(floor_correction_l1_step),
                    "last_step_renorm_correction_l1": float(renorm_correction_l1_step),
                    "cumulative_drift_limited_edge_fraction": 0.0 if total_proposed == 0 else float(total_drift_limited) / float(total_proposed),
                    "cumulative_noise_limited_edge_fraction": 0.0 if total_proposed == 0 else float(total_noise_limited) / float(total_proposed),
                    "cumulative_nonfinite_edge_fraction": 0.0 if total_proposed == 0 else float(total_nonfinite) / float(total_proposed),
                    "mean_floor_correction_l1_per_step": float(total_floor_correction_l1) / float(max(1, int(step))),
                    "mean_renorm_correction_l1_per_step": float(total_renorm_correction_l1) / float(max(1, int(step))),
                    "last_step_mobility_weighted_masked_edge_fraction": float(last_mobility_weighted_masked),
                    "last_step_noise_energy_weighted_masked_edge_fraction": float(last_noise_energy_weighted_masked),
                    "cumulative_mobility_weighted_masked_edge_fraction": float(cumulative_mobility_weighted_masked),
                    "cumulative_noise_energy_weighted_masked_edge_fraction": float(cumulative_noise_energy_weighted_masked),
                    "max_time_bin_mobility_weighted_masked_edge_fraction": float(max_time_bin_mobility_weighted_masked),
                    "max_time_bin_noise_energy_weighted_masked_edge_fraction": float(max_time_bin_noise_energy_weighted_masked),
                    "valid_innovation_fraction": 0.0 if total_proposed == 0 else 1.0 - float(total_masked) / float(total_proposed),
                    "valid_innovation_mobility_fraction": 1.0 - float(cumulative_mobility_weighted_masked),
                    "valid_innovation_noise_energy_fraction": 1.0 - float(cumulative_noise_energy_weighted_masked),
                }
            )
            metrics.append(row)

    terminal_np = states.detach().cpu().numpy().astype(np.float64)
    final_row = metrics[-1] if metrics else {}
    stat = stationarity_diagnostics(
        terminal_np,
        config,
        rng=np.random.default_rng(int(seed) + 7919),
        max_samples=int(stationarity_max_samples),
        calibration_reps=int(stationarity_calibration_reps),
        quantile_multiplier=float(stationarity_quantile_multiplier),
        mmd_multiplier=float(stationarity_mmd_multiplier),
        quantile_floor=float(stationarity_quantile_floor),
        mmd_floor=float(stationarity_mmd_floor),
    )

    if str(reference_scale_mode) == "faithful":
        if tau_eff is not None:
            scale_tag = f"taueff{_safe_tag(float(tau_eff))}"
        else:
            scale_tag = f"rate{_safe_tag(float(reference_rate if reference_rate is not None else free_weight))}"
    else:
        scale_tag = f"wfree{_safe_tag(float(free_weight))}_wsigma{_safe_tag(float(noise_weight))}"
    ramp_tag = "ramp" if np.nanmax(rate_schedule_arr) > np.nanmin(rate_schedule_arr) else "const"
    adaptive_tag = f"sub{fixed_substeps}"
    init_tag = "dirichlet" if init_mode == "dirichlet" else f"lambda{_safe_tag(float(lambda_mix))}"
    run_id = f"K{int(sample_steps)}_{adaptive_tag}_{init_tag}_{scale_tag}_{ramp_tag}"
    eff_time = effective_time_integral(rate_schedule_arr, dt=dt)
    resolved_reference_rate = float(np.nanmean(rate_schedule_arr)) if str(reference_scale_mode) == "faithful" else (float(reference_rate) if reference_rate is not None else None)
    summary: dict[str, float | int | str] = {
        "run_id": run_id,
        "init_law": str(init_mode),
        "sample_steps": int(sample_steps),
        "substeps": int(fixed_substeps),
        "lambda_mix": float(lambda_mix),
        "free_weight": float(free_weight),
        "noise_weight": float(noise_weight),
        "reference_scale_mode": str(reference_scale_mode),
        "reference_rate": float(resolved_reference_rate) if resolved_reference_rate is not None else "",
        "tau_eff": float(tau_eff) if tau_eff is not None else "",
        "time_change_mode": str(time_change_mode),
        "effective_time_integral": float(eff_time),
        "tau_eff_error": float("nan") if tau_eff is None or str(time_change_mode) != "integral" or not math.isfinite(float(eff_time)) else float(eff_time) - float(tau_eff),
        "rate_schedule_min": float(np.nanmin(rate_schedule_arr)),
        "rate_schedule_max": float(np.nanmax(rate_schedule_arr)),
        "rate_schedule_mean": float(np.nanmean(rate_schedule_arr)),
        "horizon_scale": float(config.horizon_scale),
        "num_paths": int(num_paths),
        "horizon": float(horizon),
        "dt": float(dt),
        "edge_alpha_mode": str(config.edge_alpha_mode),
        "alpha_eff": float(config.alpha_eff),
        "edge_alpha_value": float(edge_alpha_value(config)),
        "mass_floor": float(config.mass_floor),
        "stiffness_fraction": float(config.limiter_fraction),
        "cumulative_masked_edge_fraction": 0.0 if total_proposed == 0 else float(total_masked) / float(total_proposed),
        "max_time_bin_masked_edge_fraction": float(max_time_bin_masked),
        "cumulative_mobility_weighted_masked_edge_fraction": 0.0 if total_mobility_weight <= 0.0 else float(total_limited_mobility_weight) / float(total_mobility_weight),
        "max_time_bin_mobility_weighted_masked_edge_fraction": float(max_time_bin_mobility_weighted_masked),
        "cumulative_noise_energy_weighted_masked_edge_fraction": 0.0 if total_noise_energy <= 0.0 else float(total_limited_noise_energy) / float(total_noise_energy),
        "max_time_bin_noise_energy_weighted_masked_edge_fraction": float(max_time_bin_noise_energy_weighted_masked),
        "valid_innovation_fraction": 0.0 if total_proposed == 0 else 1.0 - float(total_masked) / float(total_proposed),
        "valid_innovation_mobility_fraction": 1.0 if total_mobility_weight <= 0.0 else 1.0 - float(total_limited_mobility_weight) / float(total_mobility_weight),
        "valid_innovation_noise_energy_fraction": 1.0 if total_noise_energy <= 0.0 else 1.0 - float(total_limited_noise_energy) / float(total_noise_energy),
        "total_mobility_weight": float(total_mobility_weight),
        "total_limited_mobility_weight": float(total_limited_mobility_weight),
        "total_noise_energy": float(total_noise_energy),
        "total_limited_noise_energy": float(total_limited_noise_energy),
        "total_masked_edges": int(total_masked),
        "total_proposed_edges": int(total_proposed),
        "total_noise_stiff_edges": int(total_noise_stiff),
        "total_drift_stiff_edges": int(total_drift_stiff),
        "total_overflow_edges": int(total_overflow),
        "total_drift_limited_edges": int(total_drift_limited),
        "total_noise_limited_edges": int(total_noise_limited),
        "total_nonfinite_edges": int(total_nonfinite),
        "cumulative_drift_limited_edge_fraction": 0.0 if total_proposed == 0 else float(total_drift_limited) / float(total_proposed),
        "cumulative_noise_limited_edge_fraction": 0.0 if total_proposed == 0 else float(total_noise_limited) / float(total_proposed),
        "cumulative_nonfinite_edge_fraction": 0.0 if total_proposed == 0 else float(total_nonfinite) / float(total_proposed),
        "total_floor_correction_l1": float(total_floor_correction_l1),
        "total_renorm_correction_l1": float(total_renorm_correction_l1),
        "mean_floor_correction_l1_per_step": float(total_floor_correction_l1) / float(max(1, int(sample_steps))),
        "mean_renorm_correction_l1_per_step": float(total_renorm_correction_l1) / float(max(1, int(sample_steps))),
        "mean_floor_correction_l1_per_path_step": float(total_floor_correction_l1) / float(max(1, int(sample_steps) * int(num_paths))),
        "mean_renorm_correction_l1_per_path_step": float(total_renorm_correction_l1) / float(max(1, int(sample_steps) * int(num_paths))),
        "adaptive_substeps_enabled": 0,
        "adaptive_substeps_requested_ignored": int(bool(adaptive_requested)),
        "adaptive_min_substeps": int(fixed_substeps),
        "adaptive_max_substeps": int(fixed_substeps),
        "adaptive_substep_target": float(adaptive_substep_target),
        "adaptive_substep_quantile": float(adaptive_substep_quantile),
        "adaptive_required_substeps_max": int(adaptive_required_substeps_max),
        "adaptive_drift_ratio_q_max": float(adaptive_drift_ratio_q_max),
        "adaptive_noise_ratio_q_max": float(adaptive_noise_ratio_q_max),
        # Deprecated compatibility columns.
        "cumulative_clip_fraction": 0.0,
        "total_clipped_edges": 0,
        "mean_substeps": float(total_substeps_used) / float(max(1, int(sample_steps))),
        "max_substeps_used": int(max_substeps_used),
        "fraction_steps_at_max_substeps": float(steps_at_substep_cap) / float(max(1, int(sample_steps))),
    }
    for key, value in final_row.items():
        if key not in {"run_id"}:
            summary[f"final_{key}"] = value
    summary.update(two_sample_state_diagnostics(initial_np, terminal_np, config, prefix="initial_terminal"))
    summary.update(stat)
    summary["dirichlet_comparison_context"] = "dirichlet_start_invariance" if init_mode == "dirichlet" else "data_start_distance_to_dirichlet_report_only"
    return Phase0SingleResult(
        run_id=run_id,
        config=config,
        lambda_mix=float(lambda_mix),
        free_weight=float(free_weight),
        noise_weight=float(noise_weight),
        reference_scale_mode=str(reference_scale_mode),
        reference_rate=resolved_reference_rate,
        tau_eff=float(tau_eff) if tau_eff is not None else None,
        time_change_mode=str(time_change_mode),
        init_law=str(init_mode),
        effective_time_integral=float(eff_time),
        rate_schedule=rate_schedule_arr.astype(np.float64),
        substeps=int(fixed_substeps),
        sample_steps=int(sample_steps),
        labels=requested_labels,
        source_indices=source_indices,
        initial_states=initial_np.astype(np.float64),
        terminal_states=terminal_np,
        metrics=metrics,
        summary=summary,
        checkpoint_states=checkpoint_states,
    )

def _gate_summary(
    result: Phase0SingleResult,
    *,
    max_final_corr: float,
    max_clip_fraction: float | None = None,
    min_entropy_fraction: float | None = None,
    max_frozen_edge_fraction: float = 1.0,
    max_at_max_substeps_fraction: float | None = None,
    max_masked_edge_fraction: float | None = None,
    max_time_bin_masked_edge_fraction: float = 0.10,
    allow_label_matched_prior_corr_fallback: bool = False,
    label_matched_max_final_corr: float = 0.30,
    require_stationarity: bool = False,
    min_background_l1: float = 0.0,
    min_fraction_pixels_changed: float = 0.0,
    max_floor_correction_l1: float = float("inf"),
    max_renorm_correction_l1: float = float("inf"),
    phase0_gate_mode: str = "d0-practical",
    limiter_health_metric: str = "mobility_weighted",
    max_weighted_masked_edge_fraction: float = 0.10,
    max_weighted_time_bin_masked_edge_fraction: float = 0.20,
    require_raw_mask: bool = False,
) -> dict[str, float | int | str]:
    """Apply the P0.8 split gate.

    ``d0-practical`` is the MNIST/data-start gate used to decide whether an
    empirical forward terminal bank can seed the first reverse D0 cache/training
    patch.  It requires digit destruction and numerical health, but it does not
    require the terminal data-start law to look Dirichlet.

    ``exact-stationary`` is a separate simulator validation gate.  It should be
    run with ``--init-law dirichlet`` and checks Dirichlet-start invariance plus
    numerical health, not digit destruction.
    """

    summary = dict(result.summary)
    gate_mode = str(phase0_gate_mode).replace("_", "-").lower()
    if gate_mode not in {"d0-practical", "exact-stationary", "report-only"}:
        raise ValueError("phase0_gate_mode must be d0-practical, exact-stationary, or report-only")
    metric = str(limiter_health_metric).replace("-", "_").lower()
    metric_aliases = {
        "raw": "raw",
        "edge": "raw",
        "mobility": "mobility_weighted",
        "mobility_weighted": "mobility_weighted",
        "noise": "noise_energy_weighted",
        "noise_energy": "noise_energy_weighted",
        "noise_energy_weighted": "noise_energy_weighted",
    }
    if metric not in metric_aliases:
        raise ValueError("limiter_health_metric must be raw, mobility_weighted, or noise_energy_weighted")
    metric = metric_aliases[metric]

    final_corr = abs(float(summary.get("final_pixel_corr_mean", float("inf"))))
    raw_masked = float(summary.get("cumulative_masked_edge_fraction", summary.get("cumulative_clip_fraction", float("inf"))))
    raw_masked_bin = float(summary.get("max_time_bin_masked_edge_fraction", 0.0))
    mobility_masked = float(summary.get("cumulative_mobility_weighted_masked_edge_fraction", raw_masked))
    mobility_masked_bin = float(summary.get("max_time_bin_mobility_weighted_masked_edge_fraction", raw_masked_bin))
    noise_masked = float(summary.get("cumulative_noise_energy_weighted_masked_edge_fraction", raw_masked))
    noise_masked_bin = float(summary.get("max_time_bin_noise_energy_weighted_masked_edge_fraction", raw_masked_bin))
    if metric == "raw":
        limiter_value = raw_masked
        limiter_bin_value = raw_masked_bin
        limiter_threshold = float(max_masked_edge_fraction if max_masked_edge_fraction is not None else max_clip_fraction if max_clip_fraction is not None else 0.05)
        limiter_bin_threshold = float(max_time_bin_masked_edge_fraction)
    elif metric == "mobility_weighted":
        limiter_value = mobility_masked
        limiter_bin_value = mobility_masked_bin
        limiter_threshold = float(max_weighted_masked_edge_fraction)
        limiter_bin_threshold = float(max_weighted_time_bin_masked_edge_fraction)
    else:
        limiter_value = noise_masked
        limiter_bin_value = noise_masked_bin
        limiter_threshold = float(max_weighted_masked_edge_fraction)
        limiter_bin_threshold = float(max_weighted_time_bin_masked_edge_fraction)
    raw_threshold = float(max_masked_edge_fraction if max_masked_edge_fraction is not None else max_clip_fraction if max_clip_fraction is not None else 0.05)
    raw_bin_threshold = float(max_time_bin_masked_edge_fraction)

    frozen = float(summary.get("final_frozen_edge_fraction", 0.0))
    stationarity_pass = bool(int(summary.get("stationarity_pass", 0)))
    stationarity_valid_start = str(summary.get("init_law", "data")) == "dirichlet"
    background_l1 = float(summary.get("final_background_l1_mean", 0.0))
    pixels_changed = float(summary.get("final_fraction_pixels_changed_above_floor", 0.0))
    floor_correction = float(summary.get("mean_floor_correction_l1_per_path_step", summary.get("mean_floor_correction_l1_per_step", 0.0)))
    renorm_correction = float(summary.get("mean_renorm_correction_l1_per_path_step", summary.get("mean_renorm_correction_l1_per_step", 0.0)))

    pass_corr_strict = final_corr <= float(max_final_corr)
    pass_corr_fallback = bool(allow_label_matched_prior_corr_fallback) and final_corr <= float(label_matched_max_final_corr)
    pass_corr = pass_corr_strict or pass_corr_fallback
    pass_limiter = limiter_value <= limiter_threshold
    pass_limiter_bin = limiter_bin_value <= limiter_bin_threshold
    pass_raw_mask = raw_masked <= raw_threshold
    pass_raw_mask_bin = raw_masked_bin <= raw_bin_threshold
    pass_frozen = frozen <= float(max_frozen_edge_fraction)
    pass_support_escape = background_l1 >= float(min_background_l1)
    pass_pixels_changed = pixels_changed >= float(min_fraction_pixels_changed)
    pass_floor = floor_correction <= float(max_floor_correction_l1)
    pass_renorm = renorm_correction <= float(max_renorm_correction_l1)
    destruction_gate = pass_corr and pass_support_escape and pass_pixels_changed
    reference_health_gate = pass_limiter and pass_limiter_bin and pass_frozen and pass_floor and pass_renorm
    if bool(require_raw_mask):
        reference_health_gate = reference_health_gate and pass_raw_mask and pass_raw_mask_bin
    stationarity_gate = stationarity_pass and stationarity_valid_start

    if gate_mode == "d0-practical":
        gate_pass = destruction_gate and reference_health_gate
        stationarity_required_for_gate = False
    elif gate_mode == "exact-stationary":
        gate_pass = reference_health_gate and stationarity_gate
        stationarity_required_for_gate = True
    else:
        gate_pass = False
        stationarity_required_for_gate = bool(require_stationarity)

    summary.update(
        {
            "gate_pass": int(gate_pass),
            "phase0_gate_mode": gate_mode,
            "gate_pass_corr": int(pass_corr),
            "gate_pass_corr_strict": int(pass_corr_strict),
            "gate_pass_corr_label_matched_fallback": int(pass_corr_fallback),
            "gate_pass_mask": int(pass_limiter),
            "gate_pass_mask_time_bin": int(pass_limiter_bin),
            "gate_pass_raw_mask_warning": int(pass_raw_mask),
            "gate_pass_raw_mask_time_bin_warning": int(pass_raw_mask_bin),
            "gate_require_raw_mask": int(bool(require_raw_mask)),
            "gate_pass_stationarity": int(stationarity_pass),
            "gate_stationarity_valid_start": int(stationarity_valid_start),
            "gate_require_stationarity": int(stationarity_required_for_gate),
            "gate_pass_frozen": int(pass_frozen),
            "gate_pass_support_escape": int(pass_support_escape),
            "gate_pass_pixels_changed": int(pass_pixels_changed),
            "gate_pass_floor_correction": int(pass_floor),
            "gate_pass_renorm_correction": int(pass_renorm),
            "destruction_gate_pass": int(destruction_gate),
            "reference_health_gate_pass": int(reference_health_gate),
            "stationarity_gate_pass": int(stationarity_gate),
            "stationarity_gate_context": "requires_dirichlet_start" if gate_mode == "exact-stationary" else "report_only_for_data_start",
            "gate_limiter_health_metric": metric,
            "gate_limiter_health_value": float(limiter_value),
            "gate_limiter_health_time_bin_value": float(limiter_bin_value),
            "gate_max_limiter_health_fraction": float(limiter_threshold),
            "gate_max_limiter_health_time_bin_fraction": float(limiter_bin_threshold),
            "gate_raw_masked_edge_fraction": float(raw_masked),
            "gate_raw_max_masked_edge_fraction_warning_threshold": float(raw_threshold),
            "gate_raw_max_time_bin_masked_edge_fraction_warning_threshold": float(raw_bin_threshold),
            "gate_mobility_weighted_masked_edge_fraction": float(mobility_masked),
            "gate_noise_energy_weighted_masked_edge_fraction": float(noise_masked),
            "gate_max_final_corr": float(max_final_corr),
            "gate_label_matched_max_final_corr": float(label_matched_max_final_corr),
            "gate_allow_label_matched_prior_corr_fallback": int(bool(allow_label_matched_prior_corr_fallback)),
            "gate_max_masked_edge_fraction": float(raw_threshold),
            "gate_max_time_bin_masked_edge_fraction": float(max_time_bin_masked_edge_fraction),
            "gate_max_weighted_masked_edge_fraction": float(max_weighted_masked_edge_fraction),
            "gate_max_weighted_time_bin_masked_edge_fraction": float(max_weighted_time_bin_masked_edge_fraction),
            "gate_max_frozen_edge_fraction": float(max_frozen_edge_fraction),
            "gate_min_background_l1": float(min_background_l1),
            "gate_min_fraction_pixels_changed": float(min_fraction_pixels_changed),
            "gate_max_floor_correction_l1": float(max_floor_correction_l1),
            "gate_max_renorm_correction_l1": float(max_renorm_correction_l1),
            "gate_floor_correction_l1": float(floor_correction),
            "gate_renorm_correction_l1": float(renorm_correction),
            # Deprecated columns kept visible for old dashboards.
            "gate_pass_clip": int(pass_limiter),
            "gate_pass_entropy": 1,
            "gate_pass_substeps": 1,
            "final_entropy_fraction_for_diagnostic": float(summary.get("final_entropy_fraction_of_stationary", 0.0)),
        }
    )

    denom_corr = max(float(max_final_corr), 1e-12)
    denom_limiter = max(float(limiter_threshold), 1e-12)
    denom_limiter_bin = max(float(limiter_bin_threshold), 1e-12)
    denom_frozen = max(float(max_frozen_edge_fraction), 1e-12)
    score = 0.0
    if gate_mode in {"d0-practical", "report-only"}:
        if not pass_corr:
            score += max(0.0, final_corr - float(max_final_corr)) / denom_corr
        if not pass_support_escape:
            score += max(0.0, float(min_background_l1) - background_l1) / max(float(min_background_l1), 1e-12)
        if not pass_pixels_changed:
            score += max(0.0, float(min_fraction_pixels_changed) - pixels_changed) / max(float(min_fraction_pixels_changed), 1e-12)
    score += max(0.0, limiter_value - limiter_threshold) / denom_limiter
    score += max(0.0, limiter_bin_value - limiter_bin_threshold) / denom_limiter_bin
    if bool(require_raw_mask):
        score += max(0.0, raw_masked - raw_threshold) / max(raw_threshold, 1e-12)
        score += max(0.0, raw_masked_bin - raw_bin_threshold) / max(raw_bin_threshold, 1e-12)
    score += 0.25 * max(0.0, frozen - float(max_frozen_edge_fraction)) / denom_frozen
    if not pass_floor:
        score += max(0.0, floor_correction - float(max_floor_correction_l1)) / max(float(max_floor_correction_l1), 1e-12)
    if not pass_renorm:
        score += max(0.0, renorm_correction - float(max_renorm_correction_l1)) / max(float(max_renorm_correction_l1), 1e-12)
    if gate_mode == "exact-stationary" and not stationarity_gate:
        if not stationarity_valid_start:
            score += 1.0
        q = float(summary.get("stationarity_quantile_distance", 0.0))
        qt = float(summary.get("stationarity_quantile_threshold", 1.0))
        m = float(summary.get("stationarity_feature_mmd", 0.0))
        mt = float(summary.get("stationarity_feature_mmd_threshold", 1.0))
        score += max(0.0, q - qt) / max(qt, 1e-12)
        score += max(0.0, m - mt) / max(mt, 1e-12)
    summary["gate_violation_score"] = float(score)
    return summary

def _choose_best_result(summaries: Sequence[dict[str, float | int | str]]) -> dict[str, float | int | str] | None:
    if not summaries:
        return None
    passing = [row for row in summaries if int(row.get("gate_pass", 0)) == 1]
    if passing:
        return sorted(
            passing,
            key=lambda r: (
                abs(float(r.get("final_pixel_corr_mean", 0.0))),
                float(r.get("stationarity_quantile_distance", float("inf"))),
                float(r.get("gate_limiter_health_value", r.get("cumulative_masked_edge_fraction", float("inf")))),
            ),
        )[0]
    return sorted(
        summaries,
        key=lambda r: (
            float(r.get("gate_violation_score", float("inf"))),
            float(r.get("gate_limiter_health_value", r.get("cumulative_masked_edge_fraction", float("inf")))),
            abs(float(r.get("final_pixel_corr_mean", float("inf"))),),
        ),
    )[0]


def save_phase0_result(result: Phase0SingleResult, out_dir: Path, *, preview_images: int, save_previews: bool) -> dict[str, str]:
    run_dir = Path(out_dir) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    metrics_rows = [dict(row, run_id=result.run_id) for row in result.metrics]
    write_csv_rows(metrics_path, metrics_rows)

    prior_path = run_dir / "prior_bank.npz"
    gate_record_json = json.dumps(result.summary, default=_serializable)
    np.savez_compressed(
        prior_path,
        terminal_states=result.terminal_states.reshape(result.terminal_states.shape[0], -1),
        labels=result.labels.astype(np.int64),
        source_indices=result.source_indices.astype(np.int64),
        initial_states=result.initial_states.reshape(result.initial_states.shape[0], -1),
        lambda_mix=np.asarray([result.lambda_mix], dtype=np.float64),
        free_weight=np.asarray([result.free_weight], dtype=np.float64),
        noise_weight=np.asarray([result.noise_weight], dtype=np.float64),
        reference_scale_mode=np.asarray([result.reference_scale_mode]),
        reference_rate=np.asarray([float("nan") if result.reference_rate is None else result.reference_rate], dtype=np.float64),
        tau_eff=np.asarray([float("nan") if result.tau_eff is None else result.tau_eff], dtype=np.float64),
        time_change_mode=np.asarray([result.time_change_mode]),
        init_law=np.asarray([result.init_law]),
        effective_time_integral=np.asarray([float("nan") if result.effective_time_integral is None else result.effective_time_integral], dtype=np.float64),
        rate_schedule=result.rate_schedule.astype(np.float64),
        substeps=np.asarray([result.substeps], dtype=np.int64),
        sample_steps=np.asarray([result.sample_steps], dtype=np.int64),
        horizon=np.asarray([natural_horizon(result.config)], dtype=np.float64),
        grid_size=np.asarray([result.config.grid_size], dtype=np.int64),
        edge_alpha_mode=np.asarray([result.config.edge_alpha_mode]),
        alpha_eff=np.asarray([result.config.alpha_eff], dtype=np.float64),
        edge_alpha_value=np.asarray([edge_alpha_value(result.config)], dtype=np.float64),
        mass_floor=np.asarray([result.config.mass_floor], dtype=np.float64),
        cumulative_masked_edge_fraction=np.asarray([float(result.summary.get("cumulative_masked_edge_fraction", np.nan))], dtype=np.float64),
        max_time_bin_masked_edge_fraction=np.asarray([float(result.summary.get("max_time_bin_masked_edge_fraction", np.nan))], dtype=np.float64),
        cumulative_mobility_weighted_masked_edge_fraction=np.asarray([float(result.summary.get("cumulative_mobility_weighted_masked_edge_fraction", np.nan))], dtype=np.float64),
        cumulative_noise_energy_weighted_masked_edge_fraction=np.asarray([float(result.summary.get("cumulative_noise_energy_weighted_masked_edge_fraction", np.nan))], dtype=np.float64),
        valid_innovation_fraction=np.asarray([float(result.summary.get("valid_innovation_fraction", np.nan))], dtype=np.float64),
        valid_innovation_mobility_fraction=np.asarray([float(result.summary.get("valid_innovation_mobility_fraction", np.nan))], dtype=np.float64),
        valid_innovation_noise_energy_fraction=np.asarray([float(result.summary.get("valid_innovation_noise_energy_fraction", np.nan))], dtype=np.float64),
        gate_pass=np.asarray([int(result.summary.get("gate_pass", 0))], dtype=np.int64),
        gate_record_json=np.asarray([gate_record_json]),
        expected_stationary_entropy=np.asarray([expected_symmetric_dirichlet_entropy(result.config)], dtype=np.float64),
        support_escape_mass_mean=np.asarray([float(result.summary.get("final_support_escape_mass_mean", np.nan))], dtype=np.float64),
        background_l1_mean=np.asarray([float(result.summary.get("final_background_l1_mean", np.nan))], dtype=np.float64),
        foreground_l1_mean=np.asarray([float(result.summary.get("final_foreground_l1_mean", np.nan))], dtype=np.float64),
    )
    saved = {"metrics_path": str(metrics_path), "prior_bank_path": str(prior_path)}

    if save_previews:
        panel_path = run_dir / "forward_noising_panel.png"
        save_noising_preview_panel(
            result.checkpoint_states,
            result.labels,
            panel_path,
            grid_size=int(result.config.grid_size),
            max_images=int(preview_images),
        )
        saved["preview_panel_path"] = str(panel_path)
        delta_panel_path = run_dir / "forward_noising_delta_panel.png"
        save_noising_delta_panel(
            result.checkpoint_states,
            result.labels,
            delta_panel_path,
            grid_size=int(result.config.grid_size),
            max_images=int(preview_images),
        )
        saved["delta_panel_path"] = str(delta_panel_path)
    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(result.summary | saved, handle, indent=2, default=_serializable)
    saved["summary_path"] = str(summary_path)
    return saved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_phase0"))
    parser.add_argument("--run-name", type=str, default="d0-p08")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--examples-per-class", type=int, default=1000)
    parser.add_argument("--synthetic-data", action="store_true", help="Use generated digit-like blobs instead of MNIST; intended for smoke tests.")
    parser.add_argument("--synthetic-examples-per-class", type=int, default=8)
    parser.add_argument("--init-law", choices=("data", "dirichlet"), default="data", help="Initial law for Phase-0 trajectories.  Use dirichlet only for the separate symmetry/invariance diagnostic.")

    parser.add_argument("--grid-size", type=int, default=28)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha-eff", type=float, default=1.0)
    parser.add_argument("--sweep-alpha-eff", type=str, default=None, help="Comma-separated alpha_eff sweep values; defaults to --alpha-eff.")
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff", "legacy", "grid"), default="alpha_eff")
    parser.add_argument("--horizon-scale", type=float, default=1.0, help="Compatibility knob; P0.7 tau_eff sweeps normally keep this at 1.")
    parser.add_argument("--sweep-horizon-scales", type=str, default=None, help="Deprecated compatibility sweep values.")
    parser.add_argument(
        "--reference-scale-mode",
        choices=("faithful", "independent"),
        default="faithful",
        help="faithful ties drift/noise by free_weight=rate and noise_weight=sqrt(rate); independent is legacy diagnostics only.",
    )
    parser.add_argument("--sweep-tau-eff", type=str, default=None, help="Comma-separated faithful-mode integrated effective times by default; use --time-change-mode rate for legacy raw-rate semantics.")
    parser.add_argument("--sweep-reference-rates", type=str, default="1e-6", help="Deprecated faithful-mode alias used when --sweep-tau-eff is omitted.")
    parser.add_argument("--time-change-mode", choices=("integral", "rate"), default="integral", help="Interpret --sweep-tau-eff as integrated time-change (P0.7 default) or as a raw rate (legacy P0.6).")
    parser.add_argument("--rate-ramp", choices=("none", "geometric"), default="none")
    parser.add_argument("--rate-ramp-ratio", type=float, default=100.0)
    parser.add_argument("--rate-min", type=float, default=None)
    parser.add_argument("--rate-max", type=float, default=None)
    parser.add_argument("--limiter-fraction", type=float, default=0.25, help="Deprecated alias for --stiffness-fraction.")
    parser.add_argument("--stiffness-fraction", type=float, default=0.25)
    parser.add_argument("--mass-floor", type=float, default=1e-7)
    parser.add_argument("--substeps", type=int, default=8, help="Fixed substeps per outer step.")
    parser.add_argument("--max-substeps", type=int, default=None, help="Deprecated alias for --substeps.")
    parser.add_argument("--adaptive-substeps", action="store_true", help="Deprecated no-op: P0.7-fast always uses fixed --substeps.")
    parser.add_argument("--no-adaptive-substeps", dest="adaptive_substeps", action="store_false")
    parser.set_defaults(adaptive_substeps=False)
    parser.add_argument("--adaptive-min-substeps", type=int, default=None, help="Deprecated no-op; accepted for compatibility.")
    parser.add_argument("--adaptive-max-substeps", type=int, default=None, help="Deprecated no-op; accepted for compatibility.")
    parser.add_argument("--adaptive-substep-target", type=float, default=0.20, help="Deprecated no-op; accepted for compatibility.")
    parser.add_argument("--adaptive-substep-quantile", type=float, default=0.99, help="Deprecated no-op; accepted for compatibility.")
    parser.add_argument("--adaptive-sampling", action="store_true", help="Deprecated no-op alias for --adaptive-substeps.")
    parser.add_argument("--no-adaptive-sampling", dest="adaptive_sampling", action="store_false")
    parser.set_defaults(adaptive_sampling=False)
    parser.add_argument("--clip-target", type=float, default=0.03, help="Deprecated; P0.7 uses limiter/mask gates.")

    parser.add_argument("--source-lowfreq-size", type=int, default=7)
    parser.add_argument("--source-blur-sigma", type=float, default=1.0)
    parser.add_argument("--source-uniform-mix", type=float, default=0.15)
    parser.add_argument("--source-concentration", type=float, default=1.0)
    parser.add_argument("--ot-cost-mode", choices=("lowres", "pixel"), default="lowres")
    parser.add_argument("--ot-lowres-size", type=int, default=7)
    parser.add_argument("--ot-blur-sigma", type=float, default=1.0)
    parser.add_argument("--ot-com-weight", type=float, default=0.25)

    parser.add_argument("--num-paths", type=int, default=256)
    parser.add_argument("--cache-batch-size", type=int, default=128)
    parser.add_argument("--sweep-noise-weights", type=str, default="0.005", help="Legacy/independent mode only.")
    parser.add_argument("--sweep-free-weights", type=str, default="0.03", help="Legacy/independent mode only.")
    parser.add_argument("--sweep-lambdas", type=str, default="0.05")
    parser.add_argument("--sweep-sample-steps", type=str, default="256")
    parser.add_argument("--theta-mask-min", type=float, default=1e-12)
    parser.add_argument("--metric-bins", type=int, default=16)
    parser.add_argument("--preview-checkpoints", type=str, default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1")
    parser.add_argument("--preview-images", type=int, default=20)
    parser.add_argument("--skip-previews", action="store_true")
    parser.add_argument("--early-stop-corr-threshold", type=float, default=None, help="Diagnostic only in this patch; the full path is still simulated.")

    parser.add_argument("--gate-max-final-corr", type=float, default=0.10)
    parser.add_argument("--allow-label-matched-prior-corr-fallback", action="store_true")
    parser.add_argument("--gate-label-matched-max-final-corr", type=float, default=0.30)
    parser.add_argument("--phase0-gate-mode", choices=("d0-practical", "exact-stationary", "report-only"), default="d0-practical", help="d0-practical gates MNIST-start terminal banks; exact-stationary is for --init-law dirichlet symmetry tests.")
    parser.add_argument("--gate-limiter-health-metric", choices=("raw", "mobility_weighted", "noise_energy_weighted"), default="mobility_weighted", help="Limiter metric used by the reference-health gate. Raw counts remain logged as warnings.")
    parser.add_argument("--gate-require-raw-mask", action="store_true", help="Also require the raw edge-count mask thresholds, not just the selected weighted limiter metric.")
    parser.add_argument("--gate-max-masked-edge-fraction", type=float, default=0.05, help="Raw masked-edge warning threshold, and active threshold if --gate-limiter-health-metric raw.")
    parser.add_argument("--gate-max-time-bin-masked-edge-fraction", type=float, default=0.10, help="Raw masked-edge time-bin warning threshold, and active threshold if --gate-limiter-health-metric raw.")
    parser.add_argument("--gate-max-weighted-masked-edge-fraction", type=float, default=0.10, help="Active threshold for mobility/noise-energy weighted limiter health.")
    parser.add_argument("--gate-max-weighted-time-bin-masked-edge-fraction", type=float, default=0.20, help="Active time-bin threshold for mobility/noise-energy weighted limiter health.")
    parser.add_argument("--gate-max-clip-fraction", type=float, default=None, help="Deprecated alias for --gate-max-masked-edge-fraction.")
    parser.add_argument("--gate-min-final-entropy-fraction", type=float, default=0.0, help="Deprecated; entropy is logged only.")
    parser.add_argument("--gate-min-final-stationary-entropy-fraction", type=float, default=None, help="Deprecated; stationarity uses two-sample tests.")
    parser.add_argument("--gate-max-frozen-edge-fraction", type=float, default=1.0)
    parser.add_argument("--gate-min-background-l1", type=float, default=1e-3, help="Require L1 change on the initial background support; set 0 for legacy behavior.")
    parser.add_argument("--gate-min-fraction-pixels-changed", type=float, default=0.01, help="Require a minimum fraction of pixels to change above the lambda-floor threshold; set 0 for legacy behavior.")
    parser.add_argument("--gate-max-floor-correction-l1", type=float, default=1e-8, help="Maximum mean floor correction L1 per outer step for reference-health gate.")
    parser.add_argument("--gate-max-renorm-correction-l1", type=float, default=1e-6, help="Maximum mean renormalization correction L1 per outer step for reference-health gate.")
    parser.add_argument("--gate-max-at-max-substeps-fraction", type=float, default=None, help="Deprecated; retry logic was removed.")
    parser.add_argument("--stationarity-max-samples", type=int, default=256)
    parser.add_argument("--stationarity-calibration-reps", type=int, default=3)
    parser.add_argument("--gate-stationarity-quantile-multiplier", type=float, default=3.0)
    parser.add_argument("--gate-stationarity-mmd-multiplier", type=float, default=3.0)
    parser.add_argument("--gate-stationarity-quantile-floor", type=float, default=0.02)
    parser.add_argument("--gate-stationarity-mmd-floor", type=float, default=1e-4)
    parser.add_argument("--refinement-gate", dest="refinement_gate", action="store_true", default=True)
    parser.add_argument("--no-refinement-gate", dest="refinement_gate", action="store_false")
    parser.add_argument("--refinement-corr-tol", type=float, default=0.02)
    parser.add_argument("--refinement-mask-nonincrease-tol", type=float, default=0.05)
    args = parser.parse_args(argv)
    adaptive_requested = bool(getattr(args, "adaptive_sampling", False)) or bool(getattr(args, "adaptive_substeps", False))
    if args.max_substeps is not None:
        args.substeps = int(args.max_substeps)
    args.adaptive_substeps_requested = bool(adaptive_requested)
    args.adaptive_substeps = False
    args.adaptive_sampling = False
    args.adaptive_min_substeps = int(args.substeps)
    args.adaptive_max_substeps = int(args.substeps)
    if args.gate_max_clip_fraction is not None:
        args.gate_max_masked_edge_fraction = float(args.gate_max_clip_fraction)
    if args.limiter_fraction != 0.25 and args.stiffness_fraction == 0.25:
        args.stiffness_fraction = float(args.limiter_fraction)
    return args


def _run_one_from_summary(
    args: argparse.Namespace,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    summary: dict[str, float | int | str],
    substeps: int,
    device: torch.device,
    preview_fractions: Sequence[float],
    seed: int,
    show_progress: bool,
) -> Phase0SingleResult:
    sample_steps = int(summary["sample_steps"])
    lambda_mix = float(summary["lambda_mix"])
    tau_eff = None if summary.get("tau_eff", "") == "" else float(summary["tau_eff"])
    reference_scale_mode = str(summary.get("reference_scale_mode", "faithful"))
    reference_rate = None if summary.get("reference_rate", "") == "" else float(summary["reference_rate"])
    time_change_mode = str(summary.get("time_change_mode", args.time_change_mode))
    free_weight = float(summary.get("free_weight", 0.0))
    noise_weight = float(summary.get("noise_weight", 0.0))
    config = _make_dynamics_config(
        args,
        sample_steps=sample_steps,
        free_weight=free_weight,
        noise_weight=noise_weight,
        horizon_scale=float(summary.get("horizon_scale", 1.0)),
    )
    config = DirectFluxMNISTConfig(**{**config.__dict__, "max_substeps": int(substeps)})
    rate_schedule = make_rate_schedule(
        sample_steps,
        mode=reference_scale_mode,
        tau_eff=tau_eff,
        constant_rate=reference_rate if reference_rate is not None else free_weight,
        ramp=str(args.rate_ramp),
        ramp_ratio=float(args.rate_ramp_ratio),
        rate_min=args.rate_min,
        rate_max=args.rate_max,
        horizon=natural_horizon(config),
        time_change_mode=time_change_mode,
    )
    if reference_scale_mode == "faithful":
        realized_rate = float(np.nanmean(rate_schedule))
        free_weight = realized_rate
        noise_weight = math.sqrt(max(realized_rate, 0.0))
        config = _make_dynamics_config(
            args,
            sample_steps=sample_steps,
            free_weight=free_weight,
            noise_weight=noise_weight,
            horizon_scale=float(summary.get("horizon_scale", 1.0)),
        )
        config = DirectFluxMNISTConfig(**{**config.__dict__, "max_substeps": max(int(config.max_substeps), int(substeps))})
    return run_forward_noising_single(
        images=images,
        labels=labels,
        config=config,
        lambda_mix=lambda_mix,
        free_weight=free_weight,
        noise_weight=noise_weight,
        init_law=str(summary.get("init_law", getattr(args, "init_law", "data"))),
        sample_steps=sample_steps,
        num_paths=int(args.num_paths),
        reference_scale_mode=reference_scale_mode,
        reference_rate=reference_rate,
        tau_eff=tau_eff,
        time_change_mode=time_change_mode,
        rate_schedule=rate_schedule,
        substeps=int(substeps),
        adaptive_substeps=bool(getattr(args, "adaptive_substeps_requested", False)),
        adaptive_min_substeps=int(args.adaptive_min_substeps),
        adaptive_max_substeps=int(args.adaptive_max_substeps),
        adaptive_substep_target=float(args.adaptive_substep_target),
        adaptive_substep_quantile=float(args.adaptive_substep_quantile),
        batch_size=max(1, int(args.cache_batch_size)),
        theta_mask_min=float(args.theta_mask_min),
        preview_fractions=preview_fractions,
        metric_bins=int(args.metric_bins),
        device=device,
        seed=seed,
        show_progress=show_progress,
        stationarity_max_samples=int(args.stationarity_max_samples),
        stationarity_calibration_reps=int(args.stationarity_calibration_reps),
        stationarity_quantile_multiplier=float(args.gate_stationarity_quantile_multiplier),
        stationarity_mmd_multiplier=float(args.gate_stationarity_mmd_multiplier),
        stationarity_quantile_floor=float(args.gate_stationarity_quantile_floor),
        stationarity_mmd_floor=float(args.gate_stationarity_mmd_floor),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    show_progress = not bool(args.no_progress)
    run_dir, metadata = make_phase0_run_dir(args.runs_root, args.run_name)
    n = int(args.grid_size)
    if str(args.init_law) == "dirichlet":
        # Dirichlet-start symmetry tests do not need MNIST on disk.  Provide a
        # tiny compatible placeholder for code paths that expect image/label arrays.
        images = np.full((1, n, n), 1.0 / float(n * n), dtype=np.float64)
        labels = np.zeros((1,), dtype=np.int64)
    else:
        images, labels = load_phase0_dataset(args)
        if images.shape[1:] != (n, n):
            raise ValueError(f"dataset images have shape {images.shape[1:]}, but --grid-size={n}")
    metadata.update(
        {
            "experiment": "experiment12_d0_p08_gate_split_weighted_limiter",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "args": {key: _serializable(value) for key, value in vars(args).items()},
            "dataset_size": int(images.shape[0]),
            "theory_notes": ["p06_boundary_integrator_patch.md", "experiment12_d0_patch_plan.md", "d0_patch_theory.pdf", "eulerian_approx.tex"],
        }
    )
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=_serializable)

    reference_schedules = _reference_schedules(args)
    horizon_scales = _parse_csv_floats(args.sweep_horizon_scales) or [float(args.horizon_scale)]
    if args.sweep_horizon_scales is not None:
        print("WARNING: --sweep-horizon-scales is deprecated in P0.7; prefer integrated --sweep-tau-eff.")
    alpha_eff_values = _parse_csv_floats(args.sweep_alpha_eff) or [float(args.alpha_eff)]
    lambdas = _parse_csv_floats(args.sweep_lambdas) or [0.05]
    sample_steps_list = _parse_csv_ints(args.sweep_sample_steps) or [256]
    preview_fractions = _parse_checkpoint_fractions(args.preview_checkpoints)

    print(f"Experiment 12/D0 P0.8 on device={device}")
    print(f"Run directory: {run_dir}")
    print(f"Initial law: {args.init_law}; loaded measures: {images.shape[0]} examples, grid={n}x{n}")
    print(f"Reference mode: {args.reference_scale_mode}, time_change_mode={args.time_change_mode}, edge_alpha_mode={args.edge_alpha_mode}, alpha_eff sweep={alpha_eff_values}")
    if bool(getattr(args, "adaptive_substeps_requested", False)):
        print("WARNING: --adaptive-substeps/--adaptive-sampling is disabled in P0.7-fast; using fixed --substeps only.")
    print(f"Gate mode: {args.phase0_gate_mode}; limiter health metric={args.gate_limiter_health_metric}")
    if str(args.phase0_gate_mode) == "exact-stationary" and str(args.init_law) != "dirichlet":
        print("WARNING: exact-stationary gate is meaningful only with --init-law dirichlet.")
    if str(args.reference_scale_mode) == "independent":
        print("WARNING: independent reference mode decouples w_free and w_sigma; use only for diagnostics.")
    if str(args.edge_alpha_mode) == "grid":
        print("WARNING: grid alpha mode is the manuscript scaling and was unstable for Gaussian innovation Phase 0; use as diagnostics unless intended.")

    all_summaries: list[dict[str, float | int | str]] = []
    result_paths: dict[str, dict[str, str]] = {}
    run_counter = 0
    for sample_steps in sample_steps_list:
        for horizon_scale in horizon_scales:
            for alpha_eff in alpha_eff_values:
                args.alpha_eff = float(alpha_eff)
                for lambda_mix in lambdas:
                    for ref in reference_schedules:
                        reference_scale_mode = str(ref["reference_scale_mode"])
                        reference_rate = ref["reference_rate"]
                        tau_eff = None if ref.get("tau_eff") is None else float(ref["tau_eff"])
                        time_change_mode = str(ref.get("time_change_mode", args.time_change_mode))
                        raw_free = ref.get("free_weight", 0.0)
                        raw_noise = ref.get("noise_weight", 0.0)
                        free_weight = 0.0 if raw_free is None or not math.isfinite(float(raw_free)) else float(raw_free)
                        noise_weight = 0.0 if raw_noise is None or not math.isfinite(float(raw_noise)) else float(raw_noise)
                        config = _make_dynamics_config(
                            args,
                            sample_steps=int(sample_steps),
                            free_weight=float(free_weight),
                            noise_weight=float(noise_weight),
                            horizon_scale=float(horizon_scale),
                        )
                        rate_schedule = make_rate_schedule(
                            int(sample_steps),
                            mode=reference_scale_mode,
                            tau_eff=tau_eff,
                            constant_rate=float(reference_rate) if reference_rate is not None else free_weight,
                            ramp=str(args.rate_ramp),
                            ramp_ratio=float(args.rate_ramp_ratio),
                            rate_min=args.rate_min,
                            rate_max=args.rate_max,
                            horizon=natural_horizon(config),
                            time_change_mode=time_change_mode,
                        )
                        if reference_scale_mode == "faithful":
                            realized_rate = float(np.nanmean(rate_schedule))
                            free_weight = realized_rate
                            noise_weight = math.sqrt(max(realized_rate, 0.0))
                            reference_rate_for_run = realized_rate
                            config = _make_dynamics_config(
                                args,
                                sample_steps=int(sample_steps),
                                free_weight=free_weight,
                                noise_weight=noise_weight,
                                horizon_scale=float(horizon_scale),
                            )
                            eff_int = effective_time_integral(rate_schedule, dt=natural_horizon(config) / float(sample_steps))
                            scale_msg = f"tau_eff={tau_eff if tau_eff is not None else float('nan'):g} eff_int={eff_int:g} rate_mean={realized_rate:g} tc={time_change_mode}"
                        else:
                            reference_rate_for_run = float(reference_rate) if reference_rate is not None else None
                            scale_msg = f"w_free={free_weight:g} w_sigma={noise_weight:g}"
                        print(
                            "\nP0.8 schedule: "
                            f"K={sample_steps} substeps={args.substeps} adaptive=0 alpha_eff={alpha_eff:g} lambda={lambda_mix:g} "
                            f"Hscale={horizon_scale:g} {scale_msg}"
                        )
                        result = run_forward_noising_single(
                            images=images,
                            labels=labels,
                            config=config,
                            lambda_mix=float(lambda_mix),
                            free_weight=float(free_weight),
                            noise_weight=float(noise_weight),
                            init_law=str(args.init_law),
                            sample_steps=int(sample_steps),
                            num_paths=int(args.num_paths),
                            reference_scale_mode=reference_scale_mode,
                            reference_rate=reference_rate_for_run,
                            tau_eff=tau_eff,
                            time_change_mode=time_change_mode,
                            rate_schedule=rate_schedule,
                            substeps=int(args.substeps),
                            adaptive_substeps=bool(getattr(args, "adaptive_substeps_requested", False)),
                            adaptive_min_substeps=int(args.adaptive_min_substeps),
                            adaptive_max_substeps=int(args.adaptive_max_substeps),
                            adaptive_substep_target=float(args.adaptive_substep_target),
                            adaptive_substep_quantile=float(args.adaptive_substep_quantile),
                            batch_size=max(1, int(args.cache_batch_size)),
                            theta_mask_min=float(args.theta_mask_min),
                            preview_fractions=preview_fractions,
                            metric_bins=int(args.metric_bins),
                            device=device,
                            seed=int(args.seed) + run_counter * 1009,
                            show_progress=show_progress,
                            stationarity_max_samples=int(args.stationarity_max_samples),
                            stationarity_calibration_reps=int(args.stationarity_calibration_reps),
                            stationarity_quantile_multiplier=float(args.gate_stationarity_quantile_multiplier),
                            stationarity_mmd_multiplier=float(args.gate_stationarity_mmd_multiplier),
                            stationarity_quantile_floor=float(args.gate_stationarity_quantile_floor),
                            stationarity_mmd_floor=float(args.gate_stationarity_mmd_floor),
                        )
                        gated = _gate_summary(
                            result,
                            max_final_corr=float(args.gate_max_final_corr),
                            max_masked_edge_fraction=float(args.gate_max_masked_edge_fraction),
                            max_time_bin_masked_edge_fraction=float(args.gate_max_time_bin_masked_edge_fraction),
                            max_frozen_edge_fraction=float(args.gate_max_frozen_edge_fraction),
                            allow_label_matched_prior_corr_fallback=bool(args.allow_label_matched_prior_corr_fallback),
                            label_matched_max_final_corr=float(args.gate_label_matched_max_final_corr),
                            min_background_l1=float(args.gate_min_background_l1),
                            min_fraction_pixels_changed=float(args.gate_min_fraction_pixels_changed),
                            max_floor_correction_l1=float(args.gate_max_floor_correction_l1),
                            max_renorm_correction_l1=float(args.gate_max_renorm_correction_l1),
                            phase0_gate_mode=str(args.phase0_gate_mode),
                            limiter_health_metric=str(args.gate_limiter_health_metric),
                            max_weighted_masked_edge_fraction=float(args.gate_max_weighted_masked_edge_fraction),
                            max_weighted_time_bin_masked_edge_fraction=float(args.gate_max_weighted_time_bin_masked_edge_fraction),
                            require_raw_mask=bool(args.gate_require_raw_mask),
                        )
                        result.summary = gated
                        paths = save_phase0_result(
                            result,
                            run_dir,
                            preview_images=int(args.preview_images),
                            save_previews=not bool(args.skip_previews),
                        )
                        result_paths[result.run_id] = paths
                        gated.update(paths)
                        all_summaries.append(gated)
                        run_counter += 1
                        print(
                            "  corr={:.4g} raw_mask={:.4g}/{:.4g} weighted_mask={:.4g}/{:.4g} "
                            "stat_q={:.4g}/{:.4g} H/Hunif={:.4g} gate={}".format(
                                float(gated.get("final_pixel_corr_mean", float("nan"))),
                                float(gated.get("cumulative_masked_edge_fraction", float("nan"))),
                                float(gated.get("max_time_bin_masked_edge_fraction", float("nan"))),
                                float(gated.get("gate_limiter_health_value", float("nan"))),
                                float(gated.get("gate_limiter_health_time_bin_value", float("nan"))),
                                float(gated.get("stationarity_quantile_distance", float("nan"))),
                                float(gated.get("stationarity_quantile_threshold", float("nan"))),
                                float(gated.get("final_entropy_fraction_of_uniform", float("nan"))),
                                "PASS" if int(gated.get("gate_pass", 0)) else "FAIL",
                            )
                        )

    sweep_path = run_dir / "d0_p08_sweep.csv"
    write_csv_rows(sweep_path, all_summaries)
    # Compatibility aliases for earlier Phase-0.5/P0.6/P0.7 notebooks/scripts.
    write_csv_rows(run_dir / "d0_p07_sweep.csv", all_summaries)
    write_csv_rows(run_dir / "d0_p06_sweep.csv", all_summaries)
    write_csv_rows(run_dir / "d0_p05_sweep.csv", all_summaries)
    best = _choose_best_result(all_summaries)
    gate_pass_any_initial = bool(any(int(row.get("gate_pass", 0)) for row in all_summaries))
    best_gate_pass = False if best is None else bool(int(best.get("gate_pass", 0)))

    refinement: dict[str, object] = {"enabled": bool(args.refinement_gate), "ran": False, "pass": False}
    if bool(args.refinement_gate) and best is not None and best_gate_pass:
        print("\nRunning refinement gate at 2x substeps for best schedule...")
        refined = _run_one_from_summary(
            args,
            images,
            labels,
            summary=best,
            substeps=int(best.get("substeps", args.substeps)) * 2,
            device=device,
            preview_fractions=preview_fractions,
            seed=int(args.seed) + 99991,
            show_progress=show_progress,
        )
        refined_gated = _gate_summary(
            refined,
            max_final_corr=float(args.gate_max_final_corr),
            max_masked_edge_fraction=float(args.gate_max_masked_edge_fraction),
            max_time_bin_masked_edge_fraction=float(args.gate_max_time_bin_masked_edge_fraction),
            max_frozen_edge_fraction=float(args.gate_max_frozen_edge_fraction),
            allow_label_matched_prior_corr_fallback=bool(args.allow_label_matched_prior_corr_fallback),
            label_matched_max_final_corr=float(args.gate_label_matched_max_final_corr),
            min_background_l1=float(args.gate_min_background_l1),
            min_fraction_pixels_changed=float(args.gate_min_fraction_pixels_changed),
            max_floor_correction_l1=float(args.gate_max_floor_correction_l1),
            max_renorm_correction_l1=float(args.gate_max_renorm_correction_l1),
            phase0_gate_mode=str(args.phase0_gate_mode),
            limiter_health_metric=str(args.gate_limiter_health_metric),
            max_weighted_masked_edge_fraction=float(args.gate_max_weighted_masked_edge_fraction),
            max_weighted_time_bin_masked_edge_fraction=float(args.gate_max_weighted_time_bin_masked_edge_fraction),
            require_raw_mask=bool(args.gate_require_raw_mask),
        )
        refined.summary = refined_gated
        refined_paths = save_phase0_result(
            refined,
            run_dir,
            preview_images=int(args.preview_images),
            save_previews=not bool(args.skip_previews),
        )
        refined_gated.update(refined_paths)
        initial_corr = abs(float(best.get("final_pixel_corr_mean", float("inf"))))
        refined_corr = abs(float(refined_gated.get("final_pixel_corr_mean", float("inf"))))
        initial_mask = float(best.get("gate_limiter_health_value", best.get("cumulative_masked_edge_fraction", float("inf"))))
        refined_mask = float(refined_gated.get("gate_limiter_health_value", refined_gated.get("cumulative_masked_edge_fraction", float("inf"))))
        delta_corr = abs(refined_corr - initial_corr)
        mask_ok = refined_mask <= initial_mask * (1.0 + float(args.refinement_mask_nonincrease_tol)) + 1e-12
        refinement_pass = bool(
            int(refined_gated.get("gate_pass", 0))
            and delta_corr <= float(args.refinement_corr_tol)
            and mask_ok
        )
        refinement = {
            "enabled": True,
            "ran": True,
            "pass": refinement_pass,
            "run_id": refined.run_id,
            "summary": refined_gated,
            "delta_corr": float(delta_corr),
            "corr_tol": float(args.refinement_corr_tol),
            "initial_masked_edge_fraction": float(initial_mask),
            "refined_masked_edge_fraction": float(refined_mask),
            "mask_nonincrease_tol": float(args.refinement_mask_nonincrease_tol),
            "mask_nonincrease_pass": bool(mask_ok),
            "paths": refined_paths,
        }
        print(
            "  refinement delta_corr={:.4g} mask {:.4g}->{:.4g} pass={}".format(
                delta_corr, initial_mask, refined_mask, "PASS" if refinement_pass else "FAIL"
            )
        )

    gate_pass_after_refinement = bool(gate_pass_any_initial and (not bool(args.refinement_gate) or bool(refinement.get("pass", False))))
    d0_phase1_candidate = str(args.phase0_gate_mode) == "d0-practical" and str(args.init_law) == "data"
    usable_for_phase1 = bool(d0_phase1_candidate and best_gate_pass and gate_pass_after_refinement)
    decision = {
        "phase0_gate_mode": str(args.phase0_gate_mode),
        "init_law": str(args.init_law),
        "d0_phase1_candidate_mode": bool(d0_phase1_candidate),
        "gate_pass_any": bool(gate_pass_after_refinement),
        "gate_pass_any_initial": gate_pass_any_initial,
        "usable_for_phase1": usable_for_phase1,
        "best_run_id": None if best is None else best.get("run_id"),
        "best_gate_pass_initial": best_gate_pass,
        "best_summary": best,
        "refinement_gate": refinement,
        "sweep_csv": str(sweep_path),
    }
    if best is not None:
        selected_summary = best
        selected_paths: dict[str, object] = {}
        if bool(refinement.get("pass", False)):
            refined_summary = refinement.get("summary")
            refined_paths = refinement.get("paths")
            if isinstance(refined_summary, dict):
                selected_summary = refined_summary
            if isinstance(refined_paths, dict):
                selected_paths = refined_paths
        selected_prior_value = selected_paths.get("prior_bank_path", selected_summary.get("prior_bank_path", ""))
        selected_prior = Path(str(selected_prior_value))
        decision["selected_prior_run_id"] = selected_summary.get("run_id")
        decision["selected_prior_from_refinement"] = bool(refinement.get("pass", False))
        if selected_prior.exists():
            if usable_for_phase1:
                canonical_prior = run_dir / "d0_phase0_prior_bank.npz"
                canonical_prior.write_bytes(selected_prior.read_bytes())
                decision["canonical_prior_bank"] = str(canonical_prior)
                decision["canonical_prior_source"] = str(selected_prior)
            else:
                failed_prior = run_dir / ("best_validation_prior_bank.npz" if str(args.phase0_gate_mode) != "d0-practical" else "best_failed_prior_bank.npz")
                failed_prior.write_bytes(selected_prior.read_bytes())
                decision["best_noncanonical_prior_bank"] = str(failed_prior)
                decision["best_noncanonical_prior_source"] = str(selected_prior)
                if str(args.phase0_gate_mode) == "d0-practical":
                    decision["best_failed_prior_bank"] = str(failed_prior)
    with (run_dir / "d0_phase0_gate_decision.json").open("w") as handle:
        json.dump(decision, handle, indent=2, default=_serializable)

    print("\nP0.8 Phase 0 complete")
    print(f"Sweep CSV: {sweep_path}")
    print(f"Gate decision: {run_dir / 'd0_phase0_gate_decision.json'}")
    if best is not None:
        print(f"Best run: {best.get('run_id')} initial_gate={'PASS' if int(best.get('gate_pass', 0)) else 'FAIL'}")
        if decision.get("usable_for_phase1"):
            print(f"Phase-1 prior bank: {decision.get('canonical_prior_bank')}")
        else:
            print(f"Best noncanonical prior bank (not for Phase 1): {decision.get('best_noncanonical_prior_bank', decision.get('best_failed_prior_bank', best.get('prior_bank_path')))}")


if __name__ == "__main__":
    main()
