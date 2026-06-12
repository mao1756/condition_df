from __future__ import annotations

r"""Experiment 12 / D0 Phase 0 forward-from-data noising diagnostics.

This module implements the *pre-training gate* from the D0 patch plan.  It runs
only the free Eulerian finite-volume reference process, starting from
lambda-mixed data images, and records whether the chosen schedule destroys digit
structure without becoming limiter/clip dominated.

Typical real-data usage:

    python -m mnist.diag_forward_noising \
        --run-name d0-phase0 \
        --data-root mnist_data \
        --download \
        --examples-per-class 200 \
        --num-paths 256 \
        --edge-alpha-mode grid \
        --reference-scale-mode faithful \
        --sweep-reference-rates 1e-7,3e-7,1e-6,3e-6 \
        --sweep-horizon-scales 10,30,100 \
        --sweep-lambdas 0.05,0.10,0.20 \
        --sweep-sample-steps 1024,2048

The script does not create a model, does not train, and does not produce a D0
cache.  Its outputs are schedule diagnostics, forward-noising preview grids, and
terminal prior-bank ``.npz`` files.  The canonical Phase-1 prior bank is written
only if a schedule passes the gate; otherwise the best terminal bank is marked as
failed diagnostics only.
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
    edge_alpha_value,
    _progress,
    checkerboard_energy_torch,
    harmonic_mobility_channels,
    image_total_variation,
    load_mnist_measure_dataset,
    natural_horizon,
)
from mnist.experiment11_c0 import _reference_free_step_with_innovation
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
    horizon_scale: float
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


def _parse_csv_floats(value: str | float | int) -> list[float]:
    if isinstance(value, (float, int)):
        return [float(value)]
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        raise ValueError("expected at least one comma-separated float")
    return [float(piece) for piece in pieces]


def _parse_csv_ints(value: str | int) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    if not pieces:
        raise ValueError("expected at least one comma-separated integer")
    return [int(piece) for piece in pieces]


def _optional_csv_floats(value: str | float | int | None, fallback: str | float | int) -> list[float]:
    if value is None:
        return _parse_csv_floats(fallback)
    return _parse_csv_floats(value)


def _digamma_float(value: float) -> float:
    return float(torch.digamma(torch.tensor(float(value), dtype=torch.float64)).item())


def expected_symmetric_dirichlet_entropy(config: DirectFluxMNISTConfig) -> float:
    r"""Expected entropy of the symmetric Dirichlet grid reference law.

    For a symmetric Dirichlet vector with cell parameter ``alpha`` and total
    parameter ``alpha0 = N * alpha``,

        E[-sum_i S_i log S_i] = psi(alpha0 + 1) - psi(alpha + 1).

    This is the right entropy normalizer for the theory-faithful grid mode; the
    old uniform-entropy fraction is still logged separately.
    """

    n = int(config.grid_size)
    cell_alpha = float(edge_alpha_value(config))
    total_alpha = float(n * n) * cell_alpha
    if cell_alpha <= 0.0 or total_alpha <= 0.0:
        return float("nan")
    return _digamma_float(total_alpha + 1.0) - _digamma_float(cell_alpha + 1.0)


def _reference_schedules(args: argparse.Namespace) -> list[dict[str, float | str | None]]:
    mode = str(args.reference_scale_mode)
    if mode == "faithful":
        schedules: list[dict[str, float | str | None]] = []
        for rate in _parse_csv_floats(args.sweep_reference_rates):
            if rate < 0.0 or not math.isfinite(float(rate)):
                raise ValueError("faithful reference rates must be finite and non-negative")
            schedules.append(
                {
                    "reference_scale_mode": "faithful",
                    "reference_rate": float(rate),
                    "free_weight": float(rate),
                    "noise_weight": math.sqrt(float(rate)),
                }
            )
        return schedules
    schedules = []
    for free_weight in _parse_csv_floats(args.sweep_free_weights):
        for noise_weight in _parse_csv_floats(args.sweep_noise_weights):
            schedules.append(
                {
                    "reference_scale_mode": "independent",
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
    nickname = "d0-phase0" if not run_name else str(run_name).strip().replace(" ", "-")
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
        edge_alpha_mode=str(args.edge_alpha_mode),
        horizon_scale=float(horizon_scale),
        num_steps=int(sample_steps),
        limiter_fraction=float(args.limiter_fraction),
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
        adaptive_sampling=bool(args.adaptive_sampling),
        clip_target=float(args.clip_target),
        max_substeps=int(args.max_substeps),
    )


def _synthetic_digit_measures(
    *,
    examples_per_class: int,
    grid_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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
        for rep in range(int(examples_per_class)):
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
        if candidates.size == 0:
            idx = int(rng.choice(all_idx))
        else:
            idx = int(rng.choice(candidates))
        chosen.append(idx)
    source_indices = np.asarray(chosen, dtype=np.int64)
    states = flat[source_indices].copy()
    uniform = np.full((1, n * n), 1.0 / float(n * n), dtype=np.float64)
    states = (1.0 - float(lambda_mix)) * states + float(lambda_mix) * uniform
    states = np.maximum(states, 0.0)
    states /= np.maximum(states.sum(axis=1, keepdims=True), 1e-30)
    return states.astype(np.float32), requested.astype(np.int64), source_indices


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


def _state_metrics(
    states: Tensor,
    initial_states_np: np.ndarray,
    initial_features: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    theta_mask_min: float,
    step: int,
    horizon: float,
    dt: float,
    last_step_clip_fraction: float,
    cumulative_clip_fraction: float,
    last_step_mean_substeps: float = 1.0,
    mean_substeps: float = 1.0,
    max_substeps_used: int = 1,
    fraction_steps_at_max_substeps: float = 0.0,
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
        "last_step_clip_fraction": float(last_step_clip_fraction),
        "cumulative_clip_fraction": float(cumulative_clip_fraction),
        "last_step_mean_substeps": float(last_step_mean_substeps),
        "mean_substeps": float(mean_substeps),
        "max_substeps_used": int(max_substeps_used),
        "fraction_steps_at_max_substeps": float(fraction_steps_at_max_substeps),
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
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        raise RuntimeError("matplotlib is required to save a noising preview panel") from exc

    steps = sorted(checkpoint_states)
    idx = _balanced_preview_indices(labels, int(max_images))
    cols = int(idx.size)
    rows = len(steps)
    if cols <= 0 or rows <= 0:
        return
    fig, axes = plt.subplots(rows, cols, figsize=(1.25 * cols, 1.35 * rows), squeeze=False)
    for row, step in enumerate(steps):
        arr = np.asarray(checkpoint_states[step], dtype=np.float64).reshape(-1, grid_size, grid_size)
        for col, sample_idx in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(_normalize_for_display(arr[int(sample_idx)]), cmap="gray", interpolation="nearest")
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


def _reference_free_step_phase0_adaptive(
    states: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    free_weight: float,
    noise_weight: float,
) -> tuple[Tensor, int, int, int, float]:
    """One Phase-0 macro step with optional adaptive substepping.

    The private C0 helper performs a single explicit Euler step and reports how
    many edge proposals were clipped.  Phase 0 needs the limiter diagnostics to
    reflect the same substepping convention that later cache generation and
    sampling will use.  We therefore retry a macro step with doubled substeps
    until the accepted substep-level clip fraction is below ``clip_target`` or
    ``max_substeps`` is reached.

    This diagnostic path does not store innovations, so retrying with fresh noise
    is acceptable; Phase 1 cache generation should store the raw innovations from
    the accepted substeps only.
    """

    if not bool(config.adaptive_sampling):
        out, _, _, clipped, proposed, _, _ = _reference_free_step_with_innovation(
            states,
            float(dt),
            config,
            free_weight=float(free_weight),
            noise_weight=float(noise_weight),
        )
        clip_fraction = 0.0 if int(proposed) == 0 else float(clipped) / float(proposed)
        return out, int(clipped), int(proposed), 1, clip_fraction

    max_substeps = max(1, int(config.max_substeps))
    target = float(config.clip_target)
    substeps = 1
    best: tuple[Tensor, int, int, int, float] | None = None
    while True:
        candidate = states
        clipped_total = 0
        proposed_total = 0
        sub_dt = float(dt) / float(substeps)
        for _ in range(substeps):
            candidate, _, _, clipped, proposed, _, _ = _reference_free_step_with_innovation(
                candidate,
                sub_dt,
                config,
                free_weight=float(free_weight),
                noise_weight=float(noise_weight),
            )
            clipped_total += int(clipped)
            proposed_total += int(proposed)
        clip_fraction = 0.0 if proposed_total == 0 else float(clipped_total) / float(proposed_total)
        best = (candidate, clipped_total, proposed_total, substeps, clip_fraction)
        if clip_fraction <= target or substeps >= max_substeps:
            return best
        substeps = min(max_substeps, substeps * 2)


def run_forward_noising_single(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    lambda_mix: float,
    free_weight: float,
    noise_weight: float,
    sample_steps: int,
    num_paths: int,
    reference_scale_mode: str = "independent",
    reference_rate: float | None = None,
    batch_size: int,
    theta_mask_min: float,
    preview_fractions: Sequence[float],
    metric_bins: int,
    device: torch.device,
    seed: int,
    show_progress: bool,
) -> Phase0SingleResult:
    """Run one Phase-0 schedule and return metrics plus terminal prior states."""

    if int(num_paths) <= 0:
        raise ValueError("num_paths must be positive")
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    n = int(config.grid_size)
    horizon = natural_horizon(config)
    dt = float(horizon) / float(sample_steps)
    preview_steps = _checkpoint_steps(int(sample_steps), preview_fractions)
    all_metric_steps = _metric_steps(int(sample_steps), int(metric_bins), preview_fractions)
    metric_step_set = set(all_metric_steps)
    preview_step_set = set(preview_steps)

    initial_np, requested_labels, source_indices = _sample_lambda_mixed_data(
        images,
        labels,
        count=int(num_paths),
        lambda_mix=float(lambda_mix),
        grid_size=n,
        rng=rng,
    )
    initial_features = _lowres_features_np(initial_np.reshape(-1, n, n), config)
    states = torch.as_tensor(initial_np, dtype=torch.float32, device=device)

    checkpoint_states: dict[int, np.ndarray] = {}
    metrics: list[dict[str, float | int | str]] = []
    total_clipped = 0
    total_proposed = 0
    total_path_steps = 0
    total_substeps_path_sum = 0
    total_path_steps_at_max_substeps = 0
    max_substeps_used = 1
    last_clip_fraction = 0.0
    last_mean_substeps = 1.0
    fraction_steps_at_max_substeps = 0.0

    if 0 in preview_step_set:
        checkpoint_states[0] = states.detach().cpu().numpy().copy()
    if 0 in metric_step_set:
        metrics.append(
            _state_metrics(
                states,
                initial_np,
                initial_features,
                config,
                theta_mask_min=float(theta_mask_min),
                step=0,
                horizon=horizon,
                dt=dt,
                last_step_clip_fraction=0.0,
                cumulative_clip_fraction=0.0,
                last_step_mean_substeps=1.0,
                mean_substeps=1.0,
                max_substeps_used=1,
                fraction_steps_at_max_substeps=0.0,
            )
        )

    step_iter: Iterable[int] = range(1, int(sample_steps) + 1)
    bar = _progress(list(step_iter), total=int(sample_steps), desc="Phase 0 forward noising", disable=not show_progress)
    for step in bar:
        next_chunks: list[Tensor] = []
        clipped_step = 0
        proposed_step = 0
        step_substeps_path_sum = 0
        step_path_steps_at_max = 0
        for start in range(0, int(num_paths), int(batch_size)):
            stop = min(int(num_paths), start + int(batch_size))
            chunk = states[start:stop]
            chunk, clipped, proposed, used_substeps, _accepted_clip_fraction = _reference_free_step_phase0_adaptive(
                chunk,
                dt,
                config,
                free_weight=float(free_weight),
                noise_weight=float(noise_weight),
            )
            chunk_count = int(stop - start)
            next_chunks.append(chunk)
            clipped_step += int(clipped)
            proposed_step += int(proposed)
            step_substeps_path_sum += int(used_substeps) * chunk_count
            if int(used_substeps) >= int(config.max_substeps):
                step_path_steps_at_max += chunk_count
            max_substeps_used = max(max_substeps_used, int(used_substeps))
        states = torch.cat(next_chunks, dim=0)
        total_clipped += int(clipped_step)
        total_proposed += int(proposed_step)
        total_path_steps += int(num_paths)
        total_substeps_path_sum += int(step_substeps_path_sum)
        total_path_steps_at_max_substeps += int(step_path_steps_at_max)
        last_clip_fraction = 0.0 if proposed_step == 0 else float(clipped_step) / float(proposed_step)
        cumulative_clip_fraction = 0.0 if total_proposed == 0 else float(total_clipped) / float(total_proposed)
        last_mean_substeps = float(step_substeps_path_sum) / max(float(num_paths), 1.0)
        mean_substeps = float(total_substeps_path_sum) / max(float(total_path_steps), 1.0)
        fraction_steps_at_max_substeps = float(total_path_steps_at_max_substeps) / max(float(total_path_steps), 1.0)
        if hasattr(bar, "set_postfix"):
            with torch.no_grad():
                entropy = float((-(states.clamp_min(1e-30) * states.clamp_min(1e-30).log()).sum(dim=1)).mean().cpu())
            bar.set_postfix(k=int(step), clip=cumulative_clip_fraction, sub=last_mean_substeps, H=entropy)
        if int(step) in preview_step_set:
            checkpoint_states[int(step)] = states.detach().cpu().numpy().copy()
        if int(step) in metric_step_set:
            metrics.append(
                _state_metrics(
                    states,
                    initial_np,
                    initial_features,
                    config,
                    theta_mask_min=float(theta_mask_min),
                    step=int(step),
                    horizon=horizon,
                    dt=dt,
                    last_step_clip_fraction=last_clip_fraction,
                    cumulative_clip_fraction=cumulative_clip_fraction,
                    last_step_mean_substeps=last_mean_substeps,
                    mean_substeps=mean_substeps,
                    max_substeps_used=max_substeps_used,
                    fraction_steps_at_max_substeps=fraction_steps_at_max_substeps,
                )
            )

    terminal_np = states.detach().cpu().numpy().astype(np.float64)
    final_row = metrics[-1] if metrics else {}
    if str(reference_scale_mode) == "faithful" and reference_rate is not None:
        scale_tag = f"rate{_safe_tag(float(reference_rate))}"
    else:
        scale_tag = f"wfree{_safe_tag(float(free_weight))}_wsigma{_safe_tag(float(noise_weight))}"
    run_id = (
        f"K{int(sample_steps)}_H{_safe_tag(float(config.horizon_scale))}"
        f"_lambda{_safe_tag(float(lambda_mix))}_{scale_tag}"
    )
    summary: dict[str, float | int | str] = {
        "run_id": run_id,
        "sample_steps": int(sample_steps),
        "lambda_mix": float(lambda_mix),
        "free_weight": float(free_weight),
        "noise_weight": float(noise_weight),
        "reference_scale_mode": str(reference_scale_mode),
        "reference_rate": float(reference_rate) if reference_rate is not None else "",
        "horizon_scale": float(config.horizon_scale),
        "num_paths": int(num_paths),
        "horizon": float(horizon),
        "dt": float(dt),
        "adaptive_sampling": int(bool(config.adaptive_sampling)),
        "clip_target": float(config.clip_target),
        "max_substeps_config": int(config.max_substeps),
        "cumulative_clip_fraction": 0.0 if total_proposed == 0 else float(total_clipped) / float(total_proposed),
        "total_clipped_edges": int(total_clipped),
        "total_proposed_edges": int(total_proposed),
        "mean_substeps": float(total_substeps_path_sum) / max(float(total_path_steps), 1.0),
        "max_substeps_used": int(max_substeps_used),
        "fraction_steps_at_max_substeps": float(total_path_steps_at_max_substeps) / max(float(total_path_steps), 1.0),
    }
    for key, value in final_row.items():
        if key not in {"run_id"}:
            summary[f"final_{key}"] = value
    return Phase0SingleResult(
        run_id=run_id,
        config=config,
        lambda_mix=float(lambda_mix),
        free_weight=float(free_weight),
        noise_weight=float(noise_weight),
        reference_scale_mode=str(reference_scale_mode),
        reference_rate=float(reference_rate) if reference_rate is not None else None,
        horizon_scale=float(config.horizon_scale),
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
    max_clip_fraction: float,
    min_entropy_fraction: float,
    max_frozen_edge_fraction: float,
    max_at_max_substeps_fraction: float = 0.25,
) -> dict[str, float | int | str]:
    summary = dict(result.summary)
    final_corr = abs(float(summary.get("final_pixel_corr_mean", float("inf"))))
    final_entropy_fraction_uniform = float(summary.get("final_entropy_fraction_of_uniform", 0.0))
    final_entropy_fraction_stationary = float(summary.get("final_entropy_fraction_of_stationary", 0.0))
    clip_fraction = float(summary.get("cumulative_clip_fraction", float("inf")))
    frozen = float(summary.get("final_frozen_edge_fraction", 1.0))
    at_max_substeps = float(summary.get("fraction_steps_at_max_substeps", 0.0))
    pass_corr = final_corr <= float(max_final_corr)
    pass_clip = clip_fraction <= float(max_clip_fraction)
    pass_entropy = final_entropy_fraction_stationary >= float(min_entropy_fraction)
    pass_frozen = frozen <= float(max_frozen_edge_fraction)
    pass_substeps = at_max_substeps <= float(max_at_max_substeps_fraction)
    summary.update(
        {
            "gate_pass": int(pass_corr and pass_clip and pass_entropy and pass_frozen and pass_substeps),
            "gate_pass_corr": int(pass_corr),
            "gate_pass_clip": int(pass_clip),
            "gate_pass_entropy": int(pass_entropy),
            "gate_pass_frozen": int(pass_frozen),
            "gate_pass_substeps": int(pass_substeps),
            "gate_max_final_corr": float(max_final_corr),
            "gate_max_clip_fraction": float(max_clip_fraction),
            "gate_min_entropy_fraction": float(min_entropy_fraction),
            "gate_min_stationary_entropy_fraction": float(min_entropy_fraction),
            "gate_max_frozen_edge_fraction": float(max_frozen_edge_fraction),
            "gate_max_at_max_substeps_fraction": float(max_at_max_substeps_fraction),
            "final_entropy_fraction_for_gate": float(final_entropy_fraction_stationary),
            "final_entropy_fraction_uniform_for_diagnostic": float(final_entropy_fraction_uniform),
        }
    )
    denom_corr = max(float(max_final_corr), 1e-12)
    denom_clip = max(float(max_clip_fraction), 1e-12)
    denom_entropy = max(float(min_entropy_fraction), 1e-12)
    denom_frozen = max(float(max_frozen_edge_fraction), 1e-12)
    denom_substeps = max(float(max_at_max_substeps_fraction), 1e-12)
    score = 0.0
    score += max(0.0, final_corr - float(max_final_corr)) / denom_corr
    score += max(0.0, clip_fraction - float(max_clip_fraction)) / denom_clip
    score += max(0.0, float(min_entropy_fraction) - final_entropy_fraction_stationary) / denom_entropy
    score += 0.25 * max(0.0, frozen - float(max_frozen_edge_fraction)) / denom_frozen
    score += 0.50 * max(0.0, at_max_substeps - float(max_at_max_substeps_fraction)) / denom_substeps
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
                float(r.get("noise_weight", 0.0)),
                float(r.get("cumulative_clip_fraction", 0.0)),
                abs(float(r.get("final_pixel_corr_mean", 0.0))),
            ),
        )[0]
    return sorted(
        summaries,
        key=lambda r: (
            float(r.get("gate_violation_score", float("inf"))),
            float(r.get("cumulative_clip_fraction", float("inf"))),
            abs(float(r.get("final_pixel_corr_mean", float("inf")))),
        ),
    )[0]


def save_phase0_result(result: Phase0SingleResult, out_dir: Path, *, preview_images: int, save_previews: bool) -> dict[str, str]:
    run_dir = Path(out_dir) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    metrics_rows = [dict(row, run_id=result.run_id) for row in result.metrics]
    write_csv_rows(metrics_path, metrics_rows)

    prior_path = run_dir / "prior_bank.npz"
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
        horizon_scale=np.asarray([result.horizon_scale], dtype=np.float64),
        sample_steps=np.asarray([result.sample_steps], dtype=np.int64),
        horizon=np.asarray([natural_horizon(result.config)], dtype=np.float64),
        grid_size=np.asarray([result.config.grid_size], dtype=np.int64),
        edge_alpha_mode=np.asarray([result.config.edge_alpha_mode]),
        expected_stationary_entropy=np.asarray([expected_symmetric_dirichlet_entropy(result.config)], dtype=np.float64),
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
    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(result.summary | saved, handle, indent=2, default=_serializable)
    saved["summary_path"] = str(summary_path)
    return saved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_phase0"))
    parser.add_argument("--run-name", type=str, default="d0-phase0")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--examples-per-class", type=int, default=1000)
    parser.add_argument("--synthetic-data", action="store_true", help="Use generated digit-like blobs instead of MNIST; intended for smoke tests.")
    parser.add_argument("--synthetic-examples-per-class", type=int, default=8)

    parser.add_argument("--grid-size", type=int, default=28)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--edge-alpha-mode", choices=("legacy", "grid"), default="grid")
    parser.add_argument("--horizon-scale", type=float, default=1.0, help="Single horizon scale used if --sweep-horizon-scales is not set.")
    parser.add_argument("--sweep-horizon-scales", type=str, default=None, help="Comma-separated horizon_scale sweep values.")
    parser.add_argument(
        "--reference-scale-mode",
        choices=("faithful", "independent"),
        default="faithful",
        help="faithful ties free/noise by free_weight=rate and noise_weight=sqrt(rate); independent preserves legacy sweeps.",
    )
    parser.add_argument("--sweep-reference-rates", type=str, default="1e-6", help="Comma-separated faithful time-rescaling rates.")
    parser.add_argument("--limiter-fraction", type=float, default=0.25)
    parser.add_argument("--mass-floor", type=float, default=1e-12)
    parser.add_argument("--adaptive-sampling", action="store_true", default=True)
    parser.add_argument("--no-adaptive-sampling", dest="adaptive_sampling", action="store_false")
    parser.add_argument("--clip-target", type=float, default=0.03)
    parser.add_argument("--max-substeps", type=int, default=16)

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

    parser.add_argument("--gate-max-final-corr", type=float, default=0.10)
    parser.add_argument("--gate-max-clip-fraction", type=float, default=0.05)
    parser.add_argument("--gate-min-final-entropy-fraction", type=float, default=0.80, help="Deprecated alias; now interpreted against expected stationary Dirichlet entropy.")
    parser.add_argument("--gate-min-final-stationary-entropy-fraction", type=float, default=None)
    parser.add_argument("--gate-max-frozen-edge-fraction", type=float, default=0.25)
    parser.add_argument("--gate-max-at-max-substeps-fraction", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    show_progress = not bool(args.no_progress)
    run_dir, metadata = make_phase0_run_dir(args.runs_root, args.run_name)
    images, labels = load_phase0_dataset(args)
    n = int(args.grid_size)
    if images.shape[1:] != (n, n):
        raise ValueError(f"dataset images have shape {images.shape[1:]}, but --grid-size={n}")
    metadata.update(
        {
            "experiment": "experiment12_d0_phase0_forward_noising",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "args": {key: _serializable(value) for key, value in vars(args).items()},
            "dataset_size": int(images.shape[0]),
            "theory_notes": ["experiment12_d0_patch_plan.md", "experiment11_advisor_report.pdf", "eulerian_approx.tex"],
        }
    )
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=_serializable)

    reference_schedules = _reference_schedules(args)
    horizon_scales = _optional_csv_floats(args.sweep_horizon_scales, args.horizon_scale)
    lambdas = _parse_csv_floats(args.sweep_lambdas)
    sample_steps_list = _parse_csv_ints(args.sweep_sample_steps)
    preview_fractions = _parse_checkpoint_fractions(args.preview_checkpoints)
    min_stationary_entropy_fraction = (
        float(args.gate_min_final_stationary_entropy_fraction)
        if args.gate_min_final_stationary_entropy_fraction is not None
        else float(args.gate_min_final_entropy_fraction)
    )

    print(f"Experiment 12/D0 Phase 0 on device={device}")
    print(f"Run directory: {run_dir}")
    print(f"Loaded measures: {images.shape[0]} examples, grid={n}x{n}")
    print(f"Reference mode: {args.reference_scale_mode}, edge_alpha_mode={args.edge_alpha_mode}")
    if str(args.reference_scale_mode) == "independent":
        print(
            "WARNING: --reference-scale-mode independent decouples w_free and w_sigma. "
            "Use it only for legacy diagnostics; faithful mode is the default."
        )

    all_summaries: list[dict[str, float | int | str]] = []
    result_paths: dict[str, dict[str, str]] = {}
    for sample_steps in sample_steps_list:
        for horizon_scale in horizon_scales:
            for lambda_mix in lambdas:
                for ref in reference_schedules:
                    free_weight = float(ref["free_weight"])
                    noise_weight = float(ref["noise_weight"])
                    reference_scale_mode = str(ref["reference_scale_mode"])
                    reference_rate = ref["reference_rate"]
                    config = _make_dynamics_config(
                        args,
                        sample_steps=int(sample_steps),
                        free_weight=float(free_weight),
                        noise_weight=float(noise_weight),
                        horizon_scale=float(horizon_scale),
                    )
                    if reference_scale_mode == "faithful":
                        scale_msg = f"rate={float(reference_rate):g} => w_free={free_weight:g} w_sigma={noise_weight:g}"
                    else:
                        scale_msg = f"w_free={free_weight:g} w_sigma={noise_weight:g}"
                    print(
                        "\nPhase 0 schedule: "
                        f"K={sample_steps} Hscale={horizon_scale:g} lambda={lambda_mix:g} "
                        f"{scale_msg}"
                    )
                    result = run_forward_noising_single(
                        images=images,
                        labels=labels,
                        config=config,
                        lambda_mix=float(lambda_mix),
                        free_weight=float(free_weight),
                        noise_weight=float(noise_weight),
                        sample_steps=int(sample_steps),
                        num_paths=int(args.num_paths),
                        reference_scale_mode=reference_scale_mode,
                        reference_rate=float(reference_rate) if reference_rate is not None else None,
                        batch_size=max(1, int(args.cache_batch_size)),
                        theta_mask_min=float(args.theta_mask_min),
                        preview_fractions=preview_fractions,
                        metric_bins=int(args.metric_bins),
                        device=device,
                        seed=int(args.seed) + len(all_summaries) * 1009,
                        show_progress=show_progress,
                    )
                    gated = _gate_summary(
                        result,
                        max_final_corr=float(args.gate_max_final_corr),
                        max_clip_fraction=float(args.gate_max_clip_fraction),
                        min_entropy_fraction=min_stationary_entropy_fraction,
                        max_frozen_edge_fraction=float(args.gate_max_frozen_edge_fraction),
                        max_at_max_substeps_fraction=float(args.gate_max_at_max_substeps_fraction),
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
                    print(
                        "  final_corr={:.4g} H/Hstat={:.4g} H/Hunif={:.4g} clip={:.4g} "
                        "substeps={:.3g}/{:.3g} frozen={:.4g} gate={}".format(
                            float(gated.get("final_pixel_corr_mean", float("nan"))),
                            float(gated.get("final_entropy_fraction_of_stationary", float("nan"))),
                            float(gated.get("final_entropy_fraction_of_uniform", float("nan"))),
                            float(gated.get("cumulative_clip_fraction", float("nan"))),
                            float(gated.get("mean_substeps", float("nan"))),
                            float(gated.get("fraction_steps_at_max_substeps", float("nan"))),
                            float(gated.get("final_frozen_edge_fraction", float("nan"))),
                            "PASS" if int(gated.get("gate_pass", 0)) else "FAIL",
                        )
                    )

    sweep_path = run_dir / "d0_phase0_sweep.csv"
    write_csv_rows(sweep_path, all_summaries)
    best = _choose_best_result(all_summaries)
    gate_pass_any = bool(any(int(row.get("gate_pass", 0)) for row in all_summaries))
    best_gate_pass = False if best is None else bool(int(best.get("gate_pass", 0)))
    decision = {
        "gate_pass_any": gate_pass_any,
        "usable_for_phase1": bool(gate_pass_any and best_gate_pass),
        "best_run_id": None if best is None else best.get("run_id"),
        "best_gate_pass": best_gate_pass,
        "best_summary": best,
        "sweep_csv": str(sweep_path),
    }
    if best is not None:
        best_prior = Path(str(best.get("prior_bank_path", "")))
        if best_prior.exists():
            if gate_pass_any and best_gate_pass:
                canonical_prior = run_dir / "d0_phase0_prior_bank.npz"
                canonical_prior.write_bytes(best_prior.read_bytes())
                decision["canonical_prior_bank"] = str(canonical_prior)
            else:
                failed_prior = run_dir / "best_failed_prior_bank.npz"
                failed_prior.write_bytes(best_prior.read_bytes())
                decision["best_failed_prior_bank"] = str(failed_prior)
    with (run_dir / "d0_phase0_gate_decision.json").open("w") as handle:
        json.dump(decision, handle, indent=2, default=_serializable)

    print("\nPhase 0 complete")
    print(f"Sweep CSV: {sweep_path}")
    print(f"Gate decision: {run_dir / 'd0_phase0_gate_decision.json'}")
    if best is not None:
        print(f"Best run: {best.get('run_id')} gate={'PASS' if int(best.get('gate_pass', 0)) else 'FAIL'}")
        if decision.get("usable_for_phase1"):
            print(f"Phase-1 prior bank: {decision.get('canonical_prior_bank')}")
        else:
            print(f"Best failed prior bank (not for Phase 1): {decision.get('best_failed_prior_bank', best.get('prior_bank_path'))}")


if __name__ == "__main__":
    main()
