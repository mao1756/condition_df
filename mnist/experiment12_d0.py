from __future__ import annotations

r"""Experiment 12 / D0: forward-from-data reverse innovation matching.

This module is intentionally separate from ``experiment11_c0.py``.  It reuses
only the shared finite-volume reference integrator and the U-Net architecture.
The D0 cache starts from lambda-mixed data, stores raw Gaussian edge innovations
before limiting, keys slices at the later noised state, and trains an
unweighted masked MSE for the conditional block innovation mean.
"""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    DirectFluxUNet,
    _cuda_autocast,
    _disable_mkldnn_for_cpu_if_needed,
    _edge_classes_torch,
    _make_cuda_grad_scaler,
    _progress,
    checkerboard_energy_torch,
    compute_shape_statistics_np,
    edge_alpha_value,
    edge_noise_std_channels,
    free_drift_flux_torch,
    harmonic_mobility_channels,
    image_total_variation,
    load_mnist_measure_dataset,
    masked_reference_free_step_torch,
    natural_horizon,
    save_flux_samples_grid,
    temporary_ema_weights,
    update_ema_state,
)
from mnist.weighted_point_cloud import normalize_images_to_measures


@dataclass(frozen=True)
class Experiment12D0Config:
    """Experiment-level settings for the standalone D0 trainer."""

    train_steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    base_channels: int = 48
    cache_paths: int = 4096
    cache_batch_size: int = 64
    cache_refresh_every: int = 500
    time_slices_per_path: int = 4
    teacher_stride_substeps: int = 8
    eta_l2_weight: float = 1e-4
    theta_mask_min: float = 1e-12
    lambda_mix: float = 0.35
    sample_steps: int = 512
    reference_substeps: int = 64
    tau_eff: float = 5e-5
    time_change_mode: str = "integral"
    rate_ramp: str = "none"
    rate_ramp_ratio: float = 1.0
    reference_rate_min: float | None = None
    reference_rate_max: float | None = None
    prior_bank_path: str = ""
    prior_bank_label_mode: str = "label-matched"
    control_strength: float = 1.0
    control_output_clip: float = 0.0
    sample_control_strengths: str = ""
    sampling_weights: str = "ema"
    num_samples: int = 64
    sample_batch_size: int = 64
    sample_seed: int = 1
    deterministic_sampling: bool = False
    save_sampling_ablations: bool = False
    single_image_overfit: bool = False
    single_image_index: int = 0
    single_image_label: int | None = None
    train_cache_only: bool = False
    save_cache_previews: bool = False
    seed: int = 0
    use_amp: bool = True
    ema_decay: float = 0.999


@dataclass
class D0TrainingCache:
    """Slice-level D0 cache.

    ``states`` are the later states ``S_{k+r}``; ``innovations`` are
    ``r^{-1/2} sum_{q<r} xi_{k+q}``; ``masks`` are the blockwise valid-edge
    masks including the later-state mobility threshold.
    """

    states: Tensor
    tau: Tensor
    labels: Tensor
    innovations: Tensor
    masks: Tensor
    starts: Tensor
    path_indices: Tensor
    start_images: Tensor
    terminal_states: np.ndarray
    source_indices: np.ndarray
    requested_labels: np.ndarray
    rate_schedule: np.ndarray
    horizon: float
    dt_sub: float
    stride_substeps: int
    sample_steps: int
    reference_substeps: int
    lambda_mix: float
    raw_limited_fraction: float
    mobility_weighted_limited_fraction: float
    noise_energy_weighted_limited_fraction: float
    valid_innovation_fraction: float
    valid_innovation_mobility_fraction: float
    valid_innovation_noise_energy_fraction: float
    floor_correction_l1: float = 0.0
    renorm_correction_l1: float = 0.0
    teacher_mode: str = "d0-forward"

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class D0GenerationResult:
    samples: np.ndarray
    labels: np.ndarray
    trajectory: np.ndarray | None
    limiter_fraction: float
    mobility_weighted_limiter_fraction: float
    noise_energy_weighted_limiter_fraction: float
    learned_step_rms: float
    free_step_rms: float
    noise_step_rms: float
    learned_to_noise_ratio: float
    entropy: float
    total_variation: float
    checkerboard_energy: float


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


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
    return [float(piece) for piece in pieces]


def make_experiment12_run_dir(runs_root: Path, run_name: str | None) -> tuple[Path, dict[str, object]]:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    nickname = "d0" if not run_name else str(run_name).strip().replace(" ", "-")
    run_dir = root / f"{timestamp}_{nickname}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp}_{nickname}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, {"timestamp": timestamp, "run_name": nickname, "run_dir": str(run_dir)}


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_rate_schedule(
    sample_steps: int,
    *,
    tau_eff: float,
    horizon: float,
    time_change_mode: str = "integral",
    ramp: str = "none",
    ramp_ratio: float = 1.0,
    rate_min: float | None = None,
    rate_max: float | None = None,
) -> np.ndarray:
    """Build the faithful D0 reference rate schedule.

    In the default ``integral`` mode, ``tau_eff`` is the total effective
    time-change: ``sum_k rate_k * horizon / K == tau_eff``.  ``rate`` mode
    keeps the legacy interpretation that ``tau_eff`` is already a raw rate.
    """

    k = int(sample_steps)
    if k <= 0:
        raise ValueError("sample_steps must be positive")
    if rate_min is not None and rate_max is not None:
        if str(ramp) == "geometric":
            return np.geomspace(float(rate_min), float(rate_max), k).astype(np.float64)
        return np.linspace(float(rate_min), float(rate_max), k).astype(np.float64)
    target = float(tau_eff)
    if target < 0.0 or not math.isfinite(target):
        raise ValueError("tau_eff must be finite and non-negative")
    if str(time_change_mode) == "integral":
        if horizon <= 0.0 or not math.isfinite(float(horizon)):
            raise ValueError("horizon must be positive for integral time-change mode")
        target = target / float(horizon)
    elif str(time_change_mode) != "rate":
        raise ValueError("time_change_mode must be 'integral' or 'rate'")
    if str(ramp) == "none" or k == 1:
        return np.full(k, target, dtype=np.float64)
    ratio = max(float(ramp_ratio), 1.0 + 1e-12)
    raw = np.geomspace(1.0 / ratio, 1.0, k)
    raw /= max(float(raw.mean()), 1e-30)
    return (target * raw).astype(np.float64)


def effective_time_integral(rate_schedule: np.ndarray, *, horizon: float) -> float:
    rates = np.asarray(rate_schedule, dtype=np.float64).reshape(-1)
    if rates.size == 0:
        return float("nan")
    return float(np.nansum(rates) * float(horizon) / float(rates.size))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def synthetic_digit_measures(*, examples_per_class: int, grid_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic digit-like measures for D0 smoke tests."""

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


def _lambda_mixed_data_for_paths(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    lambda_mix: float,
    grid_size: int,
    rng: np.random.Generator,
    single_image_overfit: bool = False,
    single_image_index: int = 0,
    single_image_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(grid_size)
    flat = np.asarray(images, dtype=np.float64).reshape(images.shape[0], n * n)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if single_image_overfit:
        if single_image_label is not None:
            choices = np.flatnonzero(labels_arr == int(single_image_label))
            if choices.size == 0:
                raise ValueError(f"no example with label {single_image_label} is available")
            idx0 = int(choices[min(max(int(single_image_index), 0), choices.size - 1)])
        else:
            idx0 = int(np.clip(int(single_image_index), 0, labels_arr.size - 1))
        source_indices = np.full(int(count), idx0, dtype=np.int64)
        requested = np.full(int(count), int(labels_arr[idx0]), dtype=np.int64)
    else:
        class_indices = [np.flatnonzero(labels_arr == digit) for digit in range(10)]
        requested = rng.integers(0, 10, size=int(count), dtype=np.int64)
        all_idx = np.arange(labels_arr.shape[0], dtype=np.int64)
        chosen: list[int] = []
        for digit in requested:
            candidates = class_indices[int(digit)]
            chosen.append(int(rng.choice(all_idx if candidates.size == 0 else candidates)))
        source_indices = np.asarray(chosen, dtype=np.int64)
    states = flat[source_indices].copy()
    uniform = np.full((1, n * n), 1.0 / float(n * n), dtype=np.float64)
    states = (1.0 - float(lambda_mix)) * states + float(lambda_mix) * uniform
    states = np.maximum(states, 0.0)
    states /= np.maximum(states.sum(axis=1, keepdims=True), 1e-30)
    return states.astype(np.float32), requested.astype(np.int64), source_indices.astype(np.int64)


# ---------------------------------------------------------------------------
# D0 cache and loss
# ---------------------------------------------------------------------------


def _d0_dynamics_config_from_base(base: DirectFluxMNISTConfig, *, sample_steps: int, reference_substeps: int) -> DirectFluxMNISTConfig:
    data = asdict(base)
    data.update(
        {
            "num_steps": int(sample_steps),
            "condition_on_source": False,
            "flux_parameterization": "edge",
            "free_weight": 0.0,
            "noise_weight": 0.0,
            "learned_weight": 1.0,
            "adaptive_sampling": False,
            "max_substeps": int(reference_substeps),
        }
    )
    return DirectFluxMNISTConfig(**data)


def build_d0_training_cache(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    device: torch.device,
    rng: np.random.Generator,
    show_progress: bool = True,
) -> D0TrainingCache:
    """Build an unweighted forward-from-data D0 innovation cache."""

    n = int(dynamics_config.grid_size)
    sample_steps = int(d0_config.sample_steps)
    reference_substeps = int(d0_config.reference_substeps)
    stride = int(d0_config.teacher_stride_substeps)
    if sample_steps <= 0 or reference_substeps <= 0:
        raise ValueError("sample_steps and reference_substeps must be positive")
    total_substeps = sample_steps * reference_substeps
    if stride <= 0 or stride > total_substeps:
        raise ValueError("teacher_stride_substeps must be in [1, sample_steps * reference_substeps]")
    horizon = natural_horizon(dynamics_config)
    dt_outer = horizon / float(sample_steps)
    dt_sub = horizon / float(total_substeps)
    rate_schedule = make_rate_schedule(
        sample_steps,
        tau_eff=float(d0_config.tau_eff),
        horizon=horizon,
        time_change_mode=str(d0_config.time_change_mode),
        ramp=str(d0_config.rate_ramp),
        ramp_ratio=float(d0_config.rate_ramp_ratio),
        rate_min=d0_config.reference_rate_min,
        rate_max=d0_config.reference_rate_max,
    )
    slices_per_path = int(d0_config.time_slices_per_path)
    if slices_per_path <= 0:
        raise ValueError("time_slices_per_path must be positive")

    all_states: list[Tensor] = []
    all_tau: list[Tensor] = []
    all_labels: list[Tensor] = []
    all_innov: list[Tensor] = []
    all_masks: list[Tensor] = []
    all_starts: list[Tensor] = []
    all_path_indices: list[Tensor] = []
    all_start_images: list[Tensor] = []
    terminal_states: list[np.ndarray] = []
    source_indices_chunks: list[np.ndarray] = []
    requested_label_chunks: list[np.ndarray] = []

    total_masked_edges = 0
    total_proposed_edges = 0
    total_mobility_weight = 0.0
    total_limited_mobility_weight = 0.0
    total_noise_energy = 0.0
    total_limited_noise_energy = 0.0
    total_floor_correction_l1 = 0.0
    total_renorm_correction_l1 = 0.0

    cache_paths = int(d0_config.cache_paths)
    batch_size = int(d0_config.cache_batch_size)
    batches = list(range(0, cache_paths, batch_size))
    bar = _progress(batches, total=len(batches), desc="D0 cache", disable=not show_progress)
    path_offset = 0
    for start in bar:
        stop = min(cache_paths, int(start) + batch_size)
        bsz = stop - int(start)
        initial_np, labels_np, source_idx_np = _lambda_mixed_data_for_paths(
            dataset_images,
            dataset_labels,
            count=bsz,
            lambda_mix=float(d0_config.lambda_mix),
            grid_size=n,
            rng=rng,
            single_image_overfit=bool(d0_config.single_image_overfit),
            single_image_index=int(d0_config.single_image_index),
            single_image_label=d0_config.single_image_label,
        )
        states = torch.as_tensor(initial_np, dtype=torch.float32, device=device)
        labels_t = torch.as_tensor(labels_np, dtype=torch.long)
        starts_np = rng.integers(0, total_substeps - stride + 1, size=(bsz, slices_per_path), dtype=np.int64)
        starts_t = torch.as_tensor(starts_np, dtype=torch.long, device=device)
        ends_t = starts_t + int(stride) - 1
        accum = torch.zeros((bsz, slices_per_path, 2, n, n), dtype=torch.float32, device=device)
        masks = torch.ones((bsz, slices_per_path, 2, n, n), dtype=torch.bool, device=device)
        later_states = torch.empty((bsz, slices_per_path, n * n), dtype=torch.float32, device=device)
        later_tau = torch.empty((bsz, slices_per_path), dtype=torch.float32, device=device)
        filled = torch.zeros((bsz, slices_per_path), dtype=torch.bool, device=device)

        for outer_k in range(sample_steps):
            rate = float(rate_schedule[outer_k])
            result = masked_reference_free_step_torch(
                states,
                dt_outer,
                dynamics_config,
                free_weight=rate,
                noise_weight=math.sqrt(max(rate, 0.0)),
                substeps=reference_substeps,
                stiffness_fraction=float(dynamics_config.limiter_fraction),
                return_innovations=True,
                return_substep_states=True,
            )
            if result.raw_innovations is None or result.valid_edge_mask is None or result.substep_states is None:
                raise RuntimeError("reference integrator did not return innovations/substep states")
            raw = result.raw_innovations
            valid = result.valid_edge_mask
            sub_states = result.substep_states
            for q in range(reference_substeps):
                g = outer_k * reference_substeps + q
                active = (starts_t <= g) & (g < starts_t + int(stride))
                if bool(active.any()):
                    rows, cols = torch.nonzero(active, as_tuple=True)
                    accum[rows, cols] = accum[rows, cols] + raw[q].index_select(0, rows)
                    masks[rows, cols] = masks[rows, cols] & valid[q].index_select(0, rows)
                ending = ends_t == g
                if bool(ending.any()):
                    rows, cols = torch.nonzero(ending, as_tuple=True)
                    later_states[rows, cols] = sub_states[q].index_select(0, rows)
                    later_tau[rows, cols] = max(float(horizon) - float(g + 1) * float(dt_sub), 0.0)
                    filled[rows, cols] = True
            states = result.states
            total_masked_edges += int(result.masked_edges)
            total_proposed_edges += int(result.proposed_edges)
            total_mobility_weight += float(result.mobility_weight_sum)
            total_limited_mobility_weight += float(result.limited_mobility_weight_sum)
            total_noise_energy += float(result.noise_energy_sum)
            total_limited_noise_energy += float(result.limited_noise_energy_sum)
            total_floor_correction_l1 += float(result.floor_correction_l1)
            total_renorm_correction_l1 += float(result.renorm_correction_l1)
        if not bool(filled.all()):
            raise RuntimeError("internal D0 cache bug: some later-state slices were not filled")
        accum = accum / math.sqrt(float(stride))
        flat_later = later_states.reshape(bsz * slices_per_path, n * n)
        mobility_valid = harmonic_mobility_channels(flat_later, dynamics_config) > float(d0_config.theta_mask_min)
        masks_flat = masks.reshape(bsz * slices_per_path, 2, n, n) & mobility_valid

        all_states.append(flat_later.detach().cpu())
        all_tau.append(later_tau.reshape(-1).detach().cpu())
        all_labels.append(labels_t.repeat_interleave(slices_per_path).cpu())
        all_innov.append(accum.reshape(bsz * slices_per_path, 2, n, n).detach().cpu())
        all_masks.append(masks_flat.detach().cpu())
        all_starts.append(starts_t.reshape(-1).detach().cpu())
        local_paths = torch.arange(path_offset, path_offset + bsz, dtype=torch.long).repeat_interleave(slices_per_path)
        all_path_indices.append(local_paths)
        start_images_t = torch.as_tensor(initial_np, dtype=torch.float32).repeat_interleave(slices_per_path, dim=0)
        all_start_images.append(start_images_t)
        terminal_states.append(states.detach().cpu().numpy().reshape(bsz, n, n))
        source_indices_chunks.append(source_idx_np)
        requested_label_chunks.append(labels_np)
        path_offset += bsz
        if hasattr(bar, "set_postfix"):
            valid_frac = float(torch.cat(all_masks, dim=0).float().mean()) if all_masks else 0.0
            bar.set_postfix(valid=f"{valid_frac:.3f}")

    states_out = torch.cat(all_states, dim=0).float()
    tau_out = torch.cat(all_tau, dim=0).float()
    labels_out = torch.cat(all_labels, dim=0).long()
    innov_out = torch.cat(all_innov, dim=0).float()
    masks_out = torch.cat(all_masks, dim=0).bool()
    raw_limited = 0.0 if total_proposed_edges == 0 else float(total_masked_edges) / float(total_proposed_edges)
    mobility_limited = 0.0 if total_mobility_weight <= 0.0 else total_limited_mobility_weight / total_mobility_weight
    noise_limited = 0.0 if total_noise_energy <= 0.0 else total_limited_noise_energy / total_noise_energy
    return D0TrainingCache(
        states=states_out,
        tau=tau_out,
        labels=labels_out,
        innovations=innov_out,
        masks=masks_out,
        starts=torch.cat(all_starts, dim=0).long(),
        path_indices=torch.cat(all_path_indices, dim=0).long(),
        start_images=torch.cat(all_start_images, dim=0).float(),
        terminal_states=np.concatenate(terminal_states, axis=0).astype(np.float32),
        source_indices=np.concatenate(source_indices_chunks, axis=0).astype(np.int64),
        requested_labels=np.concatenate(requested_label_chunks, axis=0).astype(np.int64),
        rate_schedule=rate_schedule.astype(np.float64),
        horizon=float(horizon),
        dt_sub=float(dt_sub),
        stride_substeps=int(stride),
        sample_steps=int(sample_steps),
        reference_substeps=int(reference_substeps),
        lambda_mix=float(d0_config.lambda_mix),
        raw_limited_fraction=float(raw_limited),
        mobility_weighted_limited_fraction=float(mobility_limited),
        noise_energy_weighted_limited_fraction=float(noise_limited),
        valid_innovation_fraction=float(masks_out.float().mean().item()),
        valid_innovation_mobility_fraction=float(1.0 - mobility_limited),
        valid_innovation_noise_energy_fraction=float(1.0 - noise_limited),
        floor_correction_l1=float(total_floor_correction_l1),
        renorm_correction_l1=float(total_renorm_correction_l1),
    )


def sample_d0_cache_batch(cache: D0TrainingCache, batch_size: int, *, device: torch.device, rng: np.random.Generator) -> dict[str, Tensor]:
    idx_np = rng.integers(0, cache.size, size=int(batch_size), dtype=np.int64)
    idx = torch.as_tensor(idx_np, dtype=torch.long)
    return {
        "states": cache.states.index_select(0, idx).to(device),
        "tau": cache.tau.index_select(0, idx).to(device),
        "labels": cache.labels.index_select(0, idx).to(device),
        "innovations": cache.innovations.index_select(0, idx).to(device),
        "masks": cache.masks.index_select(0, idx).to(device),
    }


def d0_unweighted_innovation_loss(
    model: DirectFluxUNet,
    batch: dict[str, Tensor],
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
) -> tuple[Tensor, dict[str, float]]:
    """Unweighted masked D0 MSE for ``E[Xi | later state, label]``."""

    pred = model(batch["tau"], batch["states"], batch["labels"], None)
    if float(d0_config.control_output_clip) > 0.0:
        pred = pred.clamp(-float(d0_config.control_output_clip), float(d0_config.control_output_clip))
    target = batch["innovations"]
    mask = batch["masks"] & (harmonic_mobility_channels(batch["states"], dynamics_config) > float(d0_config.theta_mask_min))
    mask_f = mask.to(dtype=pred.dtype)
    denom = mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
    per_slice = ((pred - target).square() * mask_f).sum(dim=(1, 2, 3)) / denom
    loss_main = per_slice.mean()
    zero = (target.square() * mask_f).sum(dim=(1, 2, 3)) / denom
    zero_loss = zero.mean()
    pred_l2 = (pred.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    loss = loss_main + float(d0_config.eta_l2_weight) * pred_l2
    target_rms = torch.sqrt((target.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
    residual_rms = torch.sqrt(((pred - target).square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
    pred_rms = torch.sqrt((pred.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
    gain = 1.0 - float(loss_main.detach().cpu()) / max(float(zero_loss.detach().cpu()), 1e-12)
    diag = {
        "loss": float(loss.detach().cpu()),
        "loss_main": float(loss_main.detach().cpu()),
        "loss_zero": float(zero_loss.detach().cpu()),
        "innovation_gain": float(gain),
        "prediction_rms": float(pred_rms.detach().cpu()),
        "target_rms": float(target_rms.detach().cpu()),
        "residual_rms": float(residual_rms.detach().cpu()),
        "mask_fraction": float(mask_f.mean().detach().cpu()),
        "eta_l2": float(pred_l2.detach().cpu()),
        "batch_ess_fraction": 1.0,
    }
    return loss, diag


# ---------------------------------------------------------------------------
# Reverse sampler
# ---------------------------------------------------------------------------


def _budget_limiter(delta: Tensor, tail_budget: Tensor, head_budget: Tensor) -> tuple[Tensor, Tensor]:
    upper = tail_budget.clamp_min(0.0)
    lower = -head_budget.clamp_min(0.0)
    finite = torch.isfinite(delta)
    safe = torch.where(finite, delta, torch.zeros_like(delta))
    clipped = torch.minimum(torch.maximum(safe, lower), upper)
    limited = (~finite) | (clipped != safe)
    return clipped, limited


def _apply_oriented_edge_transfer(
    states: Tensor,
    delta_channels: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    limiter_fraction: float | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Apply forward-style conservative edge transfers with P0.8 budgets."""

    if states.ndim != 2 or delta_channels.ndim != 4 or delta_channels.shape[1] != 2:
        raise ValueError("states must be (B,N) and delta_channels must be (B,2,H,W)")
    n = int(config.grid_size)
    if states.shape[1] != n * n or delta_channels.shape[2:] != (n, n):
        raise ValueError("states and edge channels have incompatible grid sizes")
    stiff_c = float(config.limiter_fraction if limiter_fraction is None else limiter_fraction)
    base = states
    out_delta = torch.zeros_like(base)
    remaining = stiff_c * (base - float(config.mass_floor)).clamp_min(0.0)
    flat_delta = torch.cat([delta_channels[:, 0].reshape(states.shape[0], -1), delta_channels[:, 1].reshape(states.shape[0], -1)], dim=1)
    proposed = 0
    limited_count = 0
    mobility_weight_sum = 0.0
    limited_mobility_weight_sum = 0.0
    theta_flat = torch.cat(
        [
            harmonic_mobility_channels(states, config)[:, 0].reshape(states.shape[0], -1),
            harmonic_mobility_channels(states, config)[:, 1].reshape(states.shape[0], -1),
        ],
        dim=1,
    )
    for edge_class in _edge_classes_torch(n, states.device):
        tails = edge_class.tails
        heads = edge_class.heads
        idx = edge_class.flux_indices
        raw = flat_delta[:, idx]
        flux, limited = _budget_limiter(raw, remaining[:, tails], remaining[:, heads])
        remaining[:, tails] = (remaining[:, tails] - flux.clamp_min(0.0)).clamp_min(0.0)
        remaining[:, heads] = (remaining[:, heads] - (-flux).clamp_min(0.0)).clamp_min(0.0)
        out_delta[:, tails] = out_delta[:, tails] - flux
        out_delta[:, heads] = out_delta[:, heads] + flux
        proposed += int(limited.numel())
        limited_count += int(limited.count_nonzero().detach().cpu())
        weights = theta_flat[:, idx].detach().clamp_min(0.0)
        mobility_weight_sum += float(weights.sum().detach().cpu())
        limited_mobility_weight_sum += float(weights.masked_select(limited).sum().detach().cpu())
    out = (base + out_delta).clamp_min(0.0)
    out = out / out.sum(dim=1, keepdim=True).clamp_min(float(config.mass_floor))
    return out, {
        "limited_edges": float(limited_count),
        "proposed_edges": float(proposed),
        "mobility_weight_sum": float(mobility_weight_sum),
        "limited_mobility_weight_sum": float(limited_mobility_weight_sum),
    }


def _load_prior_bank(
    path: str | Path,
    *,
    fallback_rate_schedule: np.ndarray | None = None,
    fallback_horizon: float | None = None,
) -> dict[str, np.ndarray | float | int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Phase-0 prior bank not found: {p}")
    bank = np.load(p, allow_pickle=True)
    states = np.asarray(bank["terminal_states"], dtype=np.float32)
    labels = np.asarray(bank["labels"], dtype=np.int64).reshape(-1)
    if states.ndim == 3:
        states = states.reshape(states.shape[0], -1)
    rate_schedule = np.asarray(bank["rate_schedule"], dtype=np.float64) if "rate_schedule" in bank else fallback_rate_schedule
    if rate_schedule is None:
        raise ValueError("prior bank does not contain a rate_schedule and no fallback was supplied")
    horizon = float(np.asarray(bank["horizon"]).reshape(-1)[0]) if "horizon" in bank else float(fallback_horizon)
    sample_steps = int(np.asarray(bank["sample_steps"]).reshape(-1)[0]) if "sample_steps" in bank else int(len(rate_schedule))
    substeps = int(np.asarray(bank["substeps"]).reshape(-1)[0]) if "substeps" in bank else 1
    return {
        "terminal_states": states,
        "labels": labels,
        "rate_schedule": np.asarray(rate_schedule, dtype=np.float64),
        "horizon": float(horizon),
        "sample_steps": int(sample_steps),
        "substeps": int(substeps),
    }


@torch.no_grad()
def simulate_d0_reverse_generation(
    model: DirectFluxUNet,
    labels: Sequence[int] | np.ndarray | Tensor,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    prior_bank_path: str | Path,
    device: torch.device,
    seed: int,
    deterministic: bool = False,
    control_strength: float | None = None,
    show_progress: bool = True,
    save_trajectory: bool = False,
) -> D0GenerationResult:
    """Generate samples by reversing the Phase-0 forward noising law."""

    model.eval()
    rng = np.random.default_rng(int(seed))
    labels_np = np.asarray(labels.detach().cpu() if isinstance(labels, Tensor) else labels, dtype=np.int64).reshape(-1)
    fallback_horizon = natural_horizon(dynamics_config)
    fallback_rate = make_rate_schedule(
        int(d0_config.sample_steps),
        tau_eff=float(d0_config.tau_eff),
        horizon=fallback_horizon,
        time_change_mode=str(d0_config.time_change_mode),
        ramp=str(d0_config.rate_ramp),
        ramp_ratio=float(d0_config.rate_ramp_ratio),
        rate_min=d0_config.reference_rate_min,
        rate_max=d0_config.reference_rate_max,
    )
    bank = _load_prior_bank(prior_bank_path, fallback_rate_schedule=fallback_rate, fallback_horizon=fallback_horizon)
    bank_states = np.asarray(bank["terminal_states"], dtype=np.float32)
    bank_labels = np.asarray(bank["labels"], dtype=np.int64)
    chosen: list[int] = []
    for label in labels_np:
        matches = np.flatnonzero(bank_labels == int(label))
        if matches.size == 0:
            if str(d0_config.prior_bank_label_mode) == "strict":
                raise ValueError(f"prior bank has no terminal states for label {label}")
            matches = np.arange(bank_labels.shape[0], dtype=np.int64)
        chosen.append(int(rng.choice(matches)))
    states = torch.as_tensor(bank_states[np.asarray(chosen)], dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    rate_schedule = np.asarray(bank["rate_schedule"], dtype=np.float64).reshape(-1)
    sample_steps = int(bank["sample_steps"])
    reference_substeps = int(bank["substeps"])
    total_substeps = sample_steps * reference_substeps
    horizon = float(bank["horizon"])
    dt_sub = horizon / float(total_substeps)
    stride = max(1, int(d0_config.teacher_stride_substeps))
    strength = float(d0_config.control_strength if control_strength is None else control_strength)

    total_limited = 0.0
    total_proposed = 0.0
    total_mobility = 0.0
    total_limited_mobility = 0.0
    learned_sq = 0.0
    free_sq = 0.0
    noise_sq = 0.0
    step_count = 0
    traj: list[np.ndarray] = []
    if save_trajectory:
        traj.append(states.detach().cpu().numpy().copy())

    q_values = list(range(total_substeps - 1, -1, -stride))
    bar = _progress(q_values, total=len(q_values), desc="D0 reverse", disable=not show_progress)
    for q_start in bar:
        block_len = min(stride, int(q_start) + 1)
        tau = torch.full((states.shape[0],), max(horizon - float(q_start + 1) * dt_sub, 0.0), dtype=states.dtype, device=device)
        m_block = model(tau, states, labels_t, None)
        if float(d0_config.control_output_clip) > 0.0:
            m_block = m_block.clamp(-float(d0_config.control_output_clip), float(d0_config.control_output_clip))
        m_step = strength * m_block / math.sqrt(float(block_len))
        for local in range(block_len):
            q = int(q_start) - local
            outer_k = min(max(q // reference_substeps, 0), rate_schedule.size - 1)
            rate = float(rate_schedule[outer_k])
            free_delta = rate * free_drift_flux_torch(states, dynamics_config) * dt_sub
            noise_std = math.sqrt(max(rate, 0.0)) * edge_noise_std_channels(states, dt_sub, dynamics_config)
            fresh = torch.zeros_like(m_step) if bool(deterministic) else torch.randn_like(m_step)
            xi_hat = fresh + m_step
            forward_delta = free_delta + noise_std * xi_hat
            states, limited = _apply_oriented_edge_transfer(states, -forward_delta, dynamics_config)
            total_limited += float(limited["limited_edges"])
            total_proposed += float(limited["proposed_edges"])
            total_mobility += float(limited["mobility_weight_sum"])
            total_limited_mobility += float(limited["limited_mobility_weight_sum"])
            learned_sq += float((noise_std * m_step).detach().square().mean().cpu())
            free_sq += float(free_delta.detach().square().mean().cpu())
            noise_sq += float((noise_std * fresh).detach().square().mean().cpu())
            step_count += 1
        if save_trajectory:
            traj.append(states.detach().cpu().numpy().copy())
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(H=float((-(states.clamp_min(1e-30) * states.clamp_min(1e-30).log()).sum(dim=1)).mean().cpu()))

    samples = states.detach().cpu().numpy().astype(np.float32)
    shape = compute_shape_statistics_np(samples, grid_size=int(dynamics_config.grid_size))
    learned_rms = math.sqrt(learned_sq / max(step_count, 1))
    noise_rms = math.sqrt(noise_sq / max(step_count, 1))
    free_rms = math.sqrt(free_sq / max(step_count, 1))
    return D0GenerationResult(
        samples=samples,
        labels=labels_np.astype(np.int64),
        trajectory=None if not save_trajectory else np.stack(traj, axis=0).astype(np.float32),
        limiter_fraction=0.0 if total_proposed <= 0.0 else float(total_limited / total_proposed),
        mobility_weighted_limiter_fraction=0.0 if total_mobility <= 0.0 else float(total_limited_mobility / total_mobility),
        noise_energy_weighted_limiter_fraction=float("nan"),
        learned_step_rms=float(learned_rms),
        free_step_rms=float(free_rms),
        noise_step_rms=float(noise_rms),
        learned_to_noise_ratio=float(learned_rms / max(noise_rms, 1e-12)),
        entropy=float(np.mean(shape["entropy"])),
        total_variation=float(np.mean(shape["tv"])),
        checkerboard_energy=float(np.mean(shape["checkerboard"] ** 2)),
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def cache_summary(cache: D0TrainingCache) -> dict[str, float | int]:
    return {
        "cache_size": int(cache.size),
        "cache_paths": int(cache.terminal_states.shape[0]),
        "cache_sample_steps": int(cache.sample_steps),
        "cache_reference_substeps": int(cache.reference_substeps),
        "cache_stride_substeps": int(cache.stride_substeps),
        "cache_horizon": float(cache.horizon),
        "cache_dt_sub": float(cache.dt_sub),
        "cache_tau_min": float(cache.tau.min().item()) if cache.size else float("nan"),
        "cache_tau_mean": float(cache.tau.mean().item()) if cache.size else float("nan"),
        "cache_tau_max": float(cache.tau.max().item()) if cache.size else float("nan"),
        "cache_target_rms": float(torch.sqrt(cache.innovations.square().mean()).item()) if cache.size else 0.0,
        "cache_mask_fraction": float(cache.masks.float().mean().item()) if cache.size else 0.0,
        "cache_raw_limited_fraction": float(cache.raw_limited_fraction),
        "cache_mobility_weighted_limited_fraction": float(cache.mobility_weighted_limited_fraction),
        "cache_noise_energy_weighted_limited_fraction": float(cache.noise_energy_weighted_limited_fraction),
        "cache_valid_innovation_fraction": float(cache.valid_innovation_fraction),
        "cache_valid_innovation_mobility_fraction": float(cache.valid_innovation_mobility_fraction),
        "cache_valid_innovation_noise_energy_fraction": float(cache.valid_innovation_noise_energy_fraction),
        "cache_floor_correction_l1": float(cache.floor_correction_l1),
        "cache_renorm_correction_l1": float(cache.renorm_correction_l1),
        "cache_effective_time_integral": float(effective_time_integral(cache.rate_schedule, horizon=cache.horizon)),
    }


def save_d0_cache_npz(cache: D0TrainingCache, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=cache.states.numpy().astype(np.float32),
        tau=cache.tau.numpy().astype(np.float32),
        labels=cache.labels.numpy().astype(np.int64),
        innovations=cache.innovations.numpy().astype(np.float32),
        masks=cache.masks.numpy().astype(np.bool_),
        starts=cache.starts.numpy().astype(np.int64),
        path_indices=cache.path_indices.numpy().astype(np.int64),
        start_images=cache.start_images.numpy().astype(np.float32),
        terminal_states=cache.terminal_states.astype(np.float32),
        source_indices=cache.source_indices.astype(np.int64),
        requested_labels=cache.requested_labels.astype(np.int64),
        rate_schedule=cache.rate_schedule.astype(np.float64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
        dt_sub=np.asarray([cache.dt_sub], dtype=np.float64),
        stride_substeps=np.asarray([cache.stride_substeps], dtype=np.int64),
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        reference_substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
    )


def train_experiment12_d0(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    run_dir: Path,
    device: torch.device,
    show_progress: bool = True,
) -> dict[str, object]:
    """Train the standalone D0 innovation predictor."""

    _disable_mkldnn_for_cpu_if_needed(device)
    rng = np.random.default_rng(int(d0_config.seed))
    torch.manual_seed(int(d0_config.seed))
    model = DirectFluxUNet(dynamics_config, base_channels=int(d0_config.base_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(d0_config.learning_rate), weight_decay=float(d0_config.weight_decay))
    scaler = _make_cuda_grad_scaler(enabled=bool(d0_config.use_amp and device.type == "cuda"))
    ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    cache = build_d0_training_cache(
        dataset_images=dataset_images,
        dataset_labels=dataset_labels,
        dynamics_config=dynamics_config,
        d0_config=d0_config,
        device=device,
        rng=rng,
        show_progress=show_progress,
    )
    save_d0_cache_npz(cache, run_dir / "experiment12_d0_cache_initial.npz")
    summary = cache_summary(cache)
    with (run_dir / "experiment12_d0_cache_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_serializable)
    if bool(d0_config.train_cache_only) or int(d0_config.train_steps) <= 0:
        torch.save({"model_state_dict": model.state_dict(), "dynamics_config": asdict(dynamics_config), "d0_config": asdict(d0_config)}, run_dir / "experiment12_d0_model.pt")
        return {"model": model, "cache": cache, "history": [], "cache_summary": summary}

    history: list[dict[str, float | int]] = []
    amp_context = _cuda_autocast if device.type == "cuda" else lambda enabled: nullcontext()
    bar = _progress(list(range(1, int(d0_config.train_steps) + 1)), total=int(d0_config.train_steps), desc="D0 train", disable=not show_progress)
    last_refresh = time.perf_counter()
    for step in bar:
        if int(d0_config.cache_refresh_every) > 0 and step > 1 and (step - 1) % int(d0_config.cache_refresh_every) == 0:
            cache = build_d0_training_cache(
                dataset_images=dataset_images,
                dataset_labels=dataset_labels,
                dynamics_config=dynamics_config,
                d0_config=d0_config,
                device=device,
                rng=rng,
                show_progress=show_progress,
            )
            last_refresh = time.perf_counter()
        batch = sample_d0_cache_batch(cache, int(d0_config.batch_size), device=device, rng=rng)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(bool(d0_config.use_amp and device.type == "cuda")):
            loss, diag = d0_unweighted_innovation_loss(model, batch, dynamics_config, d0_config)
        scaler.scale(loss).backward()
        if float(d0_config.grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(d0_config.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        update_ema_state(ema_state, model, float(d0_config.ema_decay))
        row: dict[str, float | int] = {"step": int(step), **diag, **cache_summary(cache), "seconds_since_cache_refresh": float(time.perf_counter() - last_refresh)}
        history.append(row)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=float(diag["loss"]), gain=float(diag["innovation_gain"]), mask=float(diag["mask_fraction"]))
    write_csv_rows(run_dir / "experiment12_d0_train_metrics.csv", history)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state,
            "dynamics_config": asdict(dynamics_config),
            "d0_config": asdict(d0_config),
            "history": history,
        },
        run_dir / "experiment12_d0_model.pt",
    )
    return {"model": model, "ema_state": ema_state, "cache": cache, "history": history, "cache_summary": cache_summary(cache)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _make_dynamics_config(args: argparse.Namespace) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=int(args.grid_size),
        alpha=float(args.alpha),
        beta=float(args.beta),
        alpha_eff=float(args.alpha_eff),
        edge_alpha_mode=str(args.edge_alpha_mode),
        horizon_scale=float(args.horizon_scale),
        num_steps=int(args.sample_steps),
        limiter_fraction=float(args.limiter_fraction),
        source_lowfreq_size=min(int(args.source_lowfreq_size), int(args.grid_size)),
        source_blur_sigma=float(args.source_blur_sigma),
        source_uniform_mix=float(args.source_uniform_mix),
        source_concentration=float(args.source_concentration),
        condition_on_source=False,
        upsample_mode=str(args.upsample_mode),
        flux_parameterization="edge",
        ot_cost_mode=str(args.ot_cost_mode),
        ot_lowres_size=min(int(args.ot_lowres_size), int(args.grid_size)),
        ot_blur_sigma=float(args.ot_blur_sigma),
        ot_com_weight=float(args.ot_com_weight),
        free_weight=0.0,
        noise_weight=0.0,
        learned_weight=1.0,
        mass_floor=float(args.mass_floor),
        adaptive_sampling=False,
        max_substeps=int(args.reference_substeps),
        ema_decay=float(args.ema_decay),
    )


def _make_d0_config(args: argparse.Namespace) -> Experiment12D0Config:
    return Experiment12D0Config(
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        base_channels=int(args.base_channels),
        cache_paths=int(args.cache_paths),
        cache_batch_size=int(args.cache_batch_size),
        cache_refresh_every=int(args.cache_refresh_every),
        time_slices_per_path=int(args.time_slices_per_path),
        teacher_stride_substeps=int(args.teacher_stride_substeps),
        eta_l2_weight=float(args.eta_l2_weight),
        theta_mask_min=float(args.theta_mask_min),
        lambda_mix=float(args.lambda_mix),
        sample_steps=int(args.sample_steps),
        reference_substeps=int(args.reference_substeps),
        tau_eff=float(args.tau_eff),
        time_change_mode=str(args.time_change_mode),
        rate_ramp=str(args.rate_ramp),
        rate_ramp_ratio=float(args.rate_ramp_ratio),
        reference_rate_min=args.reference_rate_min,
        reference_rate_max=args.reference_rate_max,
        prior_bank_path=str(args.prior_bank_path),
        prior_bank_label_mode=str(args.prior_bank_label_mode),
        control_strength=float(args.control_strength),
        control_output_clip=float(args.control_output_clip),
        sample_control_strengths=str(args.sample_control_strengths),
        sampling_weights=str(args.sampling_weights),
        num_samples=int(args.num_samples),
        sample_batch_size=int(args.sample_batch_size),
        sample_seed=int(args.sample_seed),
        deterministic_sampling=bool(args.deterministic_sampling),
        save_sampling_ablations=bool(args.save_sampling_ablations),
        single_image_overfit=bool(args.single_image_overfit),
        single_image_index=int(args.single_image_index),
        single_image_label=args.single_image_label,
        train_cache_only=bool(args.train_cache_only),
        save_cache_previews=bool(args.save_cache_previews),
        seed=int(args.seed),
        use_amp=not bool(args.no_amp),
        ema_decay=float(args.ema_decay),
    )


def load_d0_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if bool(args.synthetic_data):
        return synthetic_digit_measures(
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_d0"))
    parser.add_argument("--run-name", type=str, default="d0")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--examples-per-class", type=int, default=1000)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-examples-per-class", type=int, default=8)

    parser.add_argument("--grid-size", type=int, default=28)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha-eff", type=float, default=1.0)
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff", "legacy", "grid"), default="alpha_eff")
    parser.add_argument("--horizon-scale", type=float, default=1.0)
    parser.add_argument("--limiter-fraction", type=float, default=0.25)
    parser.add_argument("--mass-floor", type=float, default=1e-8)
    parser.add_argument("--source-lowfreq-size", type=int, default=7)
    parser.add_argument("--source-blur-sigma", type=float, default=1.0)
    parser.add_argument("--source-uniform-mix", type=float, default=0.15)
    parser.add_argument("--source-concentration", type=float, default=1.0)
    parser.add_argument("--upsample-mode", choices=("transpose", "resize-conv"), default="resize-conv")
    parser.add_argument("--ot-cost-mode", choices=("lowres", "pixel"), default="lowres")
    parser.add_argument("--ot-lowres-size", type=int, default=7)
    parser.add_argument("--ot-blur-sigma", type=float, default=1.0)
    parser.add_argument("--ot-com-weight", type=float, default=0.25)

    parser.add_argument("--lambda-mix", type=float, default=0.35)
    parser.add_argument("--sample-steps", type=int, default=512)
    parser.add_argument("--reference-substeps", type=int, default=64)
    parser.add_argument("--tau-eff", type=float, default=5e-5)
    parser.add_argument("--time-change-mode", choices=("integral", "rate"), default="integral")
    parser.add_argument("--rate-ramp", choices=("none", "geometric"), default="none")
    parser.add_argument("--rate-ramp-ratio", type=float, default=1.0)
    parser.add_argument("--reference-rate-min", type=float, default=None)
    parser.add_argument("--reference-rate-max", type=float, default=None)
    parser.add_argument("--teacher-stride-substeps", type=int, default=8)
    parser.add_argument("--time-slices-per-path", type=int, default=4)
    parser.add_argument("--theta-mask-min", type=float, default=1e-12)

    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--cache-paths", type=int, default=4096)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--cache-refresh-every", type=int, default=500)
    parser.add_argument("--eta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--control-output-clip", type=float, default=0.0)
    parser.add_argument("--train-cache-only", action="store_true")
    parser.add_argument("--save-cache-previews", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--single-image-overfit", action="store_true")
    parser.add_argument("--single-image-index", type=int, default=0)
    parser.add_argument("--single-image-label", type=int, default=None)

    parser.add_argument("--prior-bank-path", type=str, default="")
    parser.add_argument("--prior-bank-label-mode", choices=("label-matched", "any", "strict"), default="label-matched")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--sampling-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--control-strength", type=float, default=1.0)
    parser.add_argument("--sample-control-strengths", type=str, default="")
    parser.add_argument("--deterministic-sampling", action="store_true")
    parser.add_argument("--save-sampling-ablations", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    show_progress = not bool(args.no_progress)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir, metadata = make_experiment12_run_dir(Path(args.runs_root), args.run_name)
    dynamics_config = _make_dynamics_config(args)
    d0_config = _make_d0_config(args)
    images, labels = load_d0_dataset(args)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": metadata,
                "dynamics_config": asdict(dynamics_config),
                "d0_config": asdict(d0_config),
                "device": str(device),
            },
            handle,
            indent=2,
            default=_serializable,
        )
    result = train_experiment12_d0(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=dynamics_config,
        d0_config=d0_config,
        run_dir=run_dir,
        device=device,
        show_progress=show_progress,
    )
    model: DirectFluxUNet = result["model"]
    ema_state = result.get("ema_state")
    cache: D0TrainingCache = result["cache"]

    # Save the current empirical terminal bank too.  This makes overfit smoke
    # runs self-contained even when a Phase-0 bank path was not supplied.
    generated_bank = run_dir / "experiment12_d0_training_terminal_bank.npz"
    np.savez_compressed(
        generated_bank,
        terminal_states=cache.terminal_states.reshape(cache.terminal_states.shape[0], -1),
        labels=cache.requested_labels.astype(np.int64),
        rate_schedule=cache.rate_schedule.astype(np.float64),
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
        lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
    )
    prior_path = Path(d0_config.prior_bank_path) if str(d0_config.prior_bank_path) else generated_bank
    if int(d0_config.num_samples) > 0 and not bool(d0_config.train_cache_only):
        sample_labels = np.arange(int(d0_config.num_samples), dtype=np.int64) % 10
        if bool(d0_config.single_image_overfit):
            sample_labels[:] = int(cache.requested_labels[0])
        strengths = _parse_csv_floats(d0_config.sample_control_strengths) or [float(d0_config.control_strength)]
        sample_rows: list[dict[str, float | int | str]] = []
        for strength in strengths:
            context = temporary_ema_weights(model, ema_state) if str(d0_config.sampling_weights) == "ema" and ema_state else nullcontext()
            with context:
                gen = simulate_d0_reverse_generation(
                    model,
                    sample_labels,
                    dynamics_config=dynamics_config,
                    d0_config=d0_config,
                    prior_bank_path=prior_path,
                    device=device,
                    seed=int(d0_config.sample_seed),
                    deterministic=bool(d0_config.deterministic_sampling),
                    control_strength=float(strength),
                    show_progress=show_progress,
                    save_trajectory=False,
                )
            tag = str(strength).replace(".", "p").replace("-", "m")
            npz_path = run_dir / f"experiment12_d0_samples_strength_{tag}.npz"
            np.savez_compressed(npz_path, samples=gen.samples, labels=gen.labels)
            png_path = run_dir / f"experiment12_d0_samples_strength_{tag}.png"
            save_flux_samples_grid(gen.samples, gen.labels, png_path, grid_size=int(dynamics_config.grid_size), max_images=min(64, int(d0_config.num_samples)))
            sample_rows.append(
                {
                    "control_strength": float(strength),
                    "limiter_fraction": float(gen.limiter_fraction),
                    "mobility_weighted_limiter_fraction": float(gen.mobility_weighted_limiter_fraction),
                    "learned_step_rms": float(gen.learned_step_rms),
                    "free_step_rms": float(gen.free_step_rms),
                    "noise_step_rms": float(gen.noise_step_rms),
                    "learned_to_noise_ratio": float(gen.learned_to_noise_ratio),
                    "entropy": float(gen.entropy),
                    "total_variation": float(gen.total_variation),
                    "checkerboard_energy": float(gen.checkerboard_energy),
                    "npz_path": str(npz_path),
                    "png_path": str(png_path),
                }
            )
        write_csv_rows(run_dir / "experiment12_d0_sample_metrics.csv", sample_rows)
    print(f"Experiment 12 D0 complete: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
