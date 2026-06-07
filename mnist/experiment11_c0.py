from __future__ import annotations

r"""Experiment 11/C0: weighted free-rollout innovation matching on MNIST.

This experiment implements the C0 recipe described in the accompanying
``experiment_c0_weighted_innovation`` note.  It reuses the Experiment 10
finite-volume Eulerian simulator and U-Net architecture, but changes the
training target.  Instead of a Poisson/OT flux teacher, Experiment 11 simulates
free reference trajectories, weights whole trajectories by a soft terminal
endpoint reward, and trains the network to predict the terminally tilted mean
of the edge Brownian innovation.

The network output is interpreted as an edge Brownian-shift field ``eta``.  At
sampling time it is converted to a physical conditioning flux by

    learned_flux_e = noise_weight * sqrt(2 * theta_e * n^2) * eta_e.

Run with, for example:

    python -m mnist.experiment11_c0 --run-name c0-debug --train-steps 200

All run artifacts are written below ``runs/experiment11/<timestamp>_<name>`` by
default, matching the Experiment 10 run-folder style.
"""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from mnist.eulerian_flux_mnist import (
    ClasswiseOTCache,
    DirectFluxMNISTConfig,
    DirectFluxUNet,
    TinyMNISTClassifier,
    _cuda_autocast,
    _disable_mkldnn_for_cpu_if_needed,
    _lowres_features_np,
    _make_cuda_grad_scaler,
    _mass_entropy_torch,
    _ot_coupled_target_indices,
    _progress,
    _sample_source_batch_torch,
    classifier_generation_metrics,
    edge_alpha_value,
    flux_divergence_torch,
    free_drift_flux_torch,
    harmonic_mobility_channels,
    image_total_variation,
    load_mnist_measure_dataset,
    natural_horizon,
    poisson_flux_from_velocity_torch,
    save_flux_samples_grid,
    source_batch_diagnostics,
    train_or_load_mnist_classifier,
    update_ema_state,
)


@dataclass(frozen=True)
class Experiment11C0Config:
    """Experiment-level settings not already stored in DirectFluxMNISTConfig."""

    train_steps: int = 10_000
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    base_channels: int = 48
    cache_paths: int = 4096
    cache_batch_size: int = 128
    cache_refresh_every: int = 500
    teacher_stride: int = 8
    time_slices_per_path: int = 4
    terminal_epsilon: float = 0.0
    terminal_ess_target: float = 0.25
    eta_l2_weight: float = 1e-4
    theta_mask_min: float = 1e-12
    loss_reweighting: str = "label-balanced"
    hybrid_loss_weight: float = 0.10
    hybrid_target_clip: float = 5.0
    hybrid_min_tau_fraction: float = 0.03
    model_output_mode: str = "innovation"
    control_strength: float = 1.0
    control_output_clip: float = 5.0
    reference_free_weight: float = 0.03
    reference_noise_weight: float = 0.005
    sample_steps: int = 256
    num_samples: int = 64
    sample_save_every: int = 0
    save_cache_previews: bool = False
    seed: int = 0
    sample_seed: int = 1
    use_amp: bool = True
    use_ema_for_sampling: bool = True
    ema_decay: float = 0.999


@dataclass
class C0TrainingCache:
    """A slice-level cache produced from weighted free reference rollouts."""

    states: Tensor
    tau: Tensor
    labels: Tensor
    sources: Tensor
    innovations: Tensor
    log_weights: Tensor
    masks: Tensor
    endpoints: Tensor
    terminal_dist2: Tensor
    path_indices: Tensor
    starts: Tensor
    epsilon: float
    ess_fraction: float
    clip_fraction: float
    weighted_terminal_dist2: float
    unweighted_terminal_dist2: float
    source_indices: np.ndarray | None = None
    source_labels: np.ndarray | None = None
    requested_labels: np.ndarray | None = None
    target_indices: np.ndarray | None = None
    target_labels: np.ndarray | None = None
    terminal_states: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


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


def make_experiment11_run_dir(runs_root: Path, run_name: str | None) -> tuple[Path, dict[str, object]]:
    """Create a timestamped Experiment 11 run directory."""

    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    nickname = "c0" if not run_name else str(run_name).strip().replace(" ", "-")
    run_dir = root / f"{timestamp}_{nickname}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp}_{nickname}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, {"timestamp": timestamp, "run_name": nickname, "run_dir": str(run_dir)}


def _make_dynamics_config(args: argparse.Namespace) -> DirectFluxMNISTConfig:
    allowed = {field.name for field in fields(DirectFluxMNISTConfig)}
    raw = {
        "grid_size": args.grid_size,
        "alpha": args.alpha,
        "beta": args.beta,
        "edge_alpha_mode": args.edge_alpha_mode,
        "horizon_scale": args.horizon_scale,
        "num_steps": args.sample_steps,
        "limiter_fraction": args.limiter_fraction,
        "source_mode": args.source_mode,
        "source_lowfreq_size": args.source_lowfreq_size,
        "source_blur_sigma": args.source_blur_sigma,
        "source_uniform_mix": args.source_uniform_mix,
        "source_concentration": args.source_concentration,
        "condition_on_source": not args.no_condition_on_source,
        "upsample_mode": args.upsample_mode,
        "flux_parameterization": "edge",
        "ot_cost_mode": args.ot_cost_mode,
        "ot_match_mode": args.ot_match_mode,
        "ot_nearest_top_k": args.ot_nearest_top_k,
        "ot_lowres_size": args.ot_lowres_size,
        "ot_blur_sigma": args.ot_blur_sigma,
        "ot_com_weight": args.ot_com_weight,
        "free_weight": args.reference_free_weight,
        "noise_weight": args.reference_noise_weight,
        "learned_weight": 1.0,
        "mass_floor": args.mass_floor,
        "adaptive_sampling": args.adaptive_sampling,
        "clip_target": args.clip_target,
        "max_substeps": args.max_substeps,
        "ema_decay": args.ema_decay,
    }
    return DirectFluxMNISTConfig(**{key: value for key, value in raw.items() if key in allowed})


def _choose_epsilon_for_ess(dist2: np.ndarray, target_fraction: float) -> float:
    """Choose terminal epsilon so ESS/M is close to the requested target."""

    values = np.asarray(dist2, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    if np.all(finite <= 0.0):
        return 1.0
    target = float(np.clip(target_fraction, 1.0 / max(finite.size, 1), 0.999))

    def ess_fraction(eps: float) -> float:
        log_w = -finite / (2.0 * max(eps, 1e-12) ** 2)
        log_w = log_w - float(np.max(log_w))
        w = np.exp(log_w)
        denom = float(np.sum(w * w))
        if denom <= 0.0:
            return 0.0
        return float(np.sum(w) ** 2 / (finite.size * denom))

    scale = math.sqrt(max(float(np.median(finite)), 1e-12))
    lo = max(scale * 1e-3, 1e-6)
    hi = max(scale * 1e3, 1.0)
    for _ in range(64):
        mid = math.sqrt(lo * hi)
        if ess_fraction(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _log_weight_stats(log_weights: np.ndarray) -> dict[str, float]:
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if lw.size == 0:
        return {"ess_fraction": 0.0, "log_weight_max_minus_median": float("nan")}
    stable = lw - float(np.max(lw))
    w = np.exp(stable)
    denom = float(np.sum(w * w))
    ess = 0.0 if denom <= 0.0 else float(np.sum(w) ** 2 / (lw.size * denom))
    return {
        "ess_fraction": ess,
        "log_weight_max_minus_median": float(np.max(lw) - np.median(lw)),
    }


def _histogram_dict(values: np.ndarray | None, *, prefix: str, num_classes: int = 10) -> dict[str, int]:
    """Return class-count diagnostics with stable keys."""

    out = {f"{prefix}_{digit}": 0 for digit in range(num_classes)}
    if values is None:
        return out
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return out
    arr = arr[(0 <= arr) & (arr < num_classes)]
    counts = np.bincount(arr, minlength=num_classes)
    for digit in range(num_classes):
        out[f"{prefix}_{digit}"] = int(counts[digit])
    return out


def _weighted_histogram_dict(
    labels: np.ndarray | None,
    log_weights: np.ndarray,
    *,
    prefix: str,
    num_classes: int = 10,
) -> dict[str, float]:
    """Return normalized weighted class-mass diagnostics."""

    out = {f"{prefix}_{digit}": 0.0 for digit in range(num_classes)}
    if labels is None:
        return out
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if lab.size == 0 or lw.size == 0:
        return out
    if lab.size != lw.size:
        raise ValueError("labels and log_weights must have the same length")
    stable = lw - float(np.max(lw))
    weights = np.exp(stable)
    total = float(weights.sum())
    if total <= 0.0:
        return out
    for digit in range(num_classes):
        out[f"{prefix}_{digit}"] = float(weights[lab == digit].sum() / total)
    return out


def _per_label_weight_stats(
    labels: np.ndarray | None,
    log_weights: np.ndarray,
    dist2: np.ndarray,
    *,
    prefix: str,
    num_classes: int = 10,
) -> dict[str, float]:
    """Per-label ESS and terminal-distance diagnostics for path-level weights."""

    out: dict[str, float] = {}
    for digit in range(num_classes):
        out[f"{prefix}_ess_fraction_{digit}"] = float("nan")
        out[f"{prefix}_weighted_dist2_{digit}"] = float("nan")
        out[f"{prefix}_unweighted_dist2_{digit}"] = float("nan")
    if labels is None:
        return out
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    d2 = np.asarray(dist2, dtype=np.float64).reshape(-1)
    if lab.size == 0 or lw.size != lab.size or d2.size != lab.size:
        return out
    for digit in range(num_classes):
        mask = lab == digit
        if not np.any(mask):
            continue
        lw_d = lw[mask]
        d2_d = d2[mask]
        stable = lw_d - float(np.max(lw_d))
        w = np.exp(stable)
        denom = float(np.sum(w * w))
        total = float(np.sum(w))
        out[f"{prefix}_ess_fraction_{digit}"] = 0.0 if denom <= 0.0 else float(total * total / (w.size * denom))
        out[f"{prefix}_weighted_dist2_{digit}"] = float(np.sum(w * d2_d) / max(total, 1e-12))
        out[f"{prefix}_unweighted_dist2_{digit}"] = float(np.mean(d2_d))
    return out


def _label_balanced_preview_order(log_weights: np.ndarray, labels: np.ndarray | None, *, per_label: int = 8) -> np.ndarray:
    """Choose high-weight preview indices, with each label represented when possible."""

    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if labels is None:
        return np.argsort(-lw)[: int(per_label) * 10]
    lab = np.asarray(labels, dtype=np.int64).reshape(-1)
    chosen: list[int] = []
    for digit in range(10):
        idx = np.flatnonzero(lab == digit)
        if idx.size == 0:
            continue
        chosen.extend(int(i) for i in idx[np.argsort(-lw[idx])[: int(per_label)]])
    if not chosen:
        return np.argsort(-lw)[: int(per_label) * 10]
    out = np.asarray(chosen, dtype=np.int64)
    return out[np.argsort(-lw[out])]


def _label_balanced_weights_torch(weights: Tensor, labels: Tensor, *, num_classes: int = 10) -> Tensor:
    """Normalize weights so each label present in the minibatch contributes equally."""

    labels = labels.long().clamp(0, int(num_classes) - 1)
    class_sums = torch.zeros(int(num_classes), dtype=weights.dtype, device=weights.device)
    class_sums.scatter_add_(0, labels, weights)
    return weights / class_sums.index_select(0, labels).clamp_min(1e-12)


def _weighted_mean_torch(values: Tensor, weights: Tensor) -> Tensor:
    return (weights * values).sum() / weights.sum().clamp_min(1e-12)


def _edge_flat_to_channels(flat: Tensor, grid_size: int) -> Tensor:
    n = int(grid_size)
    return torch.stack([flat[:, : n * n].reshape(-1, n, n), flat[:, n * n :].reshape(-1, n, n)], dim=1)


def _channels_to_edge_flat(channels: Tensor) -> Tensor:
    return torch.cat([channels[:, 0].reshape(channels.shape[0], -1), channels[:, 1].reshape(channels.shape[0], -1)], dim=1)


def _reference_free_step_with_innovation(
    states: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    free_weight: float,
    noise_weight: float,
) -> tuple[Tensor, Tensor, Tensor, int, int]:
    """One free reference step, returning raw Gaussian edge innovations.

    The returned ``xi_channels`` are the standard normal edge innovations used
    before limiter clipping.  ``unclipped_channels`` is false for edges whose
    proposed transfer was clipped by the four-color update.
    """

    if states.ndim != 2:
        raise ValueError("states must have shape (B, N)")
    n = int(config.grid_size)
    batch = int(states.shape[0])
    out = states.clone()
    tiny = float(config.mass_floor)
    inv_h2 = float(n * n)
    alpha = edge_alpha_value(config)
    xi_flat = torch.randn(batch, 2 * n * n, dtype=states.dtype, device=states.device)
    unclipped_flat = torch.ones(batch, 2 * n * n, dtype=torch.bool, device=states.device)
    clipped = 0
    proposed = 0

    # Local import avoids exporting the private edge-color helper.
    from mnist.eulerian_flux_mnist import _edge_classes_torch

    for edge_class in _edge_classes_torch(n, states.device):
        tails = edge_class.tails
        heads = edge_class.heads
        a = out[:, tails]
        b = out[:, heads]
        denom = a + b
        harmonic = torch.where(denom > tiny, a * b / denom.clamp_min(tiny), torch.zeros_like(denom))
        ratio = torch.where(denom > tiny, (a - b) / denom.clamp_min(tiny), torch.zeros_like(denom))
        theta = ((2.0 * alpha + 1.0) / alpha) * harmonic
        free_flux = (2.0 * alpha + 1.0) * inv_h2 * ratio
        xi = xi_flat[:, edge_class.flux_indices]
        noise_std = float(noise_weight) * torch.sqrt((2.0 * theta * inv_h2 * float(dt)).clamp_min(0.0))
        d_flux = float(free_weight) * free_flux * float(dt) + noise_std * xi
        pos_clip = d_flux > float(config.limiter_fraction) * a
        neg_clip = d_flux < -float(config.limiter_fraction) * b
        clipped_mask = pos_clip | neg_clip
        unclipped_flat[:, edge_class.flux_indices] = ~clipped_mask
        clipped += int(clipped_mask.count_nonzero().detach().cpu())
        proposed += int(d_flux.numel())
        d_flux = torch.minimum(d_flux, float(config.limiter_fraction) * a)
        d_flux = torch.maximum(d_flux, -float(config.limiter_fraction) * b)
        out[:, tails] = out[:, tails] - d_flux
        out[:, heads] = out[:, heads] + d_flux
        out = out.clamp_min(tiny)
        out = out / out.sum(dim=1, keepdim=True).clamp_min(tiny)
    return out, _edge_flat_to_channels(xi_flat, n), _edge_flat_to_channels(unclipped_flat, n), clipped, proposed


def _sample_cache_indices(cache: C0TrainingCache, batch_size: int, device: torch.device) -> Tensor:
    del device
    # Cache tensors live on CPU.  Keep the random indices on CPU too; otherwise
    # torch.index_select raises a device-mismatch error when training on CUDA.
    return torch.randint(0, cache.size, (int(batch_size),), dtype=torch.long)


def _cache_batch(cache: C0TrainingCache, idx: Tensor, device: torch.device) -> dict[str, Tensor]:
    # Be defensive: callers may pass CUDA indices from older code or notebooks.
    # ``index_select`` requires the index tensor to live on the same device as
    # the source tensor, so select from the CPU cache first, then move the
    # minibatch to the requested training device.
    idx_cpu = idx.to(device=cache.states.device, dtype=torch.long, non_blocking=False)
    return {
        "states": cache.states.index_select(0, idx_cpu).to(device),
        "tau": cache.tau.index_select(0, idx_cpu).to(device),
        "labels": cache.labels.index_select(0, idx_cpu).to(device),
        "sources": cache.sources.index_select(0, idx_cpu).to(device),
        "innovations": cache.innovations.index_select(0, idx_cpu).to(device),
        "log_weights": cache.log_weights.index_select(0, idx_cpu).to(device),
        "masks": cache.masks.index_select(0, idx_cpu).to(device),
        "endpoints": cache.endpoints.index_select(0, idx_cpu).to(device),
    }


def build_c0_training_cache(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    ot_cache: ClasswiseOTCache,
    dynamics_config: DirectFluxMNISTConfig,
    c0_config: Experiment11C0Config,
    device: torch.device,
    rng: np.random.Generator,
    show_progress: bool = True,
) -> C0TrainingCache:
    """Build a weighted free-rollout innovation cache."""

    n = int(dynamics_config.grid_size)
    num_pixels = n * n
    paths = int(c0_config.cache_paths)
    chunk_size = max(1, int(c0_config.cache_batch_size))
    steps = int(c0_config.sample_steps)
    stride = int(c0_config.teacher_stride)
    slices_per_path = int(c0_config.time_slices_per_path)
    if paths <= 0 or steps <= 0 or stride <= 0 or slices_per_path <= 0:
        raise ValueError("cache paths, steps, teacher stride, and slices per path must be positive")
    if stride > steps:
        raise ValueError("teacher stride cannot exceed sample_steps")
    horizon = natural_horizon(dynamics_config)
    dt = horizon / float(steps)
    dtype = torch.float32

    all_states: list[Tensor] = []
    all_tau: list[Tensor] = []
    all_labels: list[Tensor] = []
    all_sources: list[Tensor] = []
    all_innovations: list[Tensor] = []
    all_masks: list[Tensor] = []
    all_path_indices: list[Tensor] = []
    all_starts: list[Tensor] = []
    all_endpoints: list[Tensor] = []
    all_terminal_np: list[np.ndarray] = []
    all_target_indices: list[np.ndarray] = []
    all_target_labels: list[np.ndarray] = []
    all_requested_labels: list[np.ndarray] = []
    all_source_indices: list[np.ndarray] = []
    all_source_labels: list[np.ndarray] = []
    all_terminal_dist2: list[np.ndarray] = []
    total_clipped = 0
    total_proposed = 0

    chunk_starts = list(range(0, paths, chunk_size))
    bar = _progress(chunk_starts, total=len(chunk_starts), desc="build C0 cache", disable=not show_progress)
    global_path_offset = 0
    for start_idx in bar:
        current = min(chunk_size, paths - int(start_idx))
        labels_np = rng.integers(0, 10, size=current, dtype=np.int64)
        labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=device)
        source_batch = _sample_source_batch_torch(
            current,
            dynamics_config,
            device=device,
            dtype=dtype,
            label_tensor=labels_t,
            source_images=dataset_images,
            source_labels=dataset_labels,
            rng=rng,
            class_indices=ot_cache.class_indices,
        )
        states = source_batch.masses.to(device=device, dtype=dtype)
        sources = states.clone()
        source_np = states.detach().cpu().numpy().reshape(current, n, n)
        target_indices = _ot_coupled_target_indices(
            source_np,
            labels_np,
            dataset_images,
            dataset_labels,
            dynamics_config,
            rng=rng,
            ot_cache=ot_cache,
        )
        endpoints_np = np.asarray(dataset_images[target_indices], dtype=np.float64).reshape(current, n, n)
        target_labels_np = np.asarray(dataset_labels[target_indices], dtype=np.int64)
        if not np.array_equal(target_labels_np, labels_np):
            mismatch = int(np.count_nonzero(target_labels_np != labels_np))
            raise RuntimeError(f"target sampler returned {mismatch} labels that do not match requested labels")
        endpoints = torch.as_tensor(endpoints_np.reshape(current, num_pixels), dtype=dtype, device=device)

        starts_np = rng.integers(0, steps - stride + 1, size=(current, slices_per_path), dtype=np.int64)
        # Store source state at requested starts and accumulate stride-normalized innovations.
        slice_states = torch.empty(current, slices_per_path, num_pixels, dtype=dtype, device=device)
        slice_tau = torch.empty(current, slices_per_path, dtype=dtype, device=device)
        slice_innov = torch.zeros(current, slices_per_path, 2, n, n, dtype=dtype, device=device)
        slice_masks = torch.ones(current, slices_per_path, 2, n, n, dtype=torch.bool, device=device)
        starts_t = torch.as_tensor(starts_np, dtype=torch.long, device=device)

        for step in range(steps):
            step_mask = starts_t == int(step)
            if bool(step_mask.any()):
                # Assign current state to every slice whose block begins here.
                rows, cols = torch.where(step_mask)
                slice_states[rows, cols] = states.index_select(0, rows)
                tau_value = max(horizon - float(step) * dt, 0.0)
                slice_tau[rows, cols] = tau_value
            states, xi_channels, unclipped_channels, clipped, proposed = _reference_free_step_with_innovation(
                states,
                dt,
                dynamics_config,
                free_weight=float(c0_config.reference_free_weight),
                noise_weight=float(c0_config.reference_noise_weight),
            )
            total_clipped += int(clipped)
            total_proposed += int(proposed)
            # Accumulate the current innovation into all stride windows containing this step.
            lo = max(0, step - stride + 1)
            hi = min(step, steps - stride) + 1
            if hi > lo:
                active = (starts_t >= lo) & (starts_t <= hi)
                if bool(active.any()):
                    rows, cols = torch.where(active)
                    slice_innov[rows, cols] += xi_channels.index_select(0, rows)
                    slice_masks[rows, cols] &= unclipped_channels.index_select(0, rows)

        slice_innov = slice_innov / math.sqrt(float(stride))
        terminal_np = states.detach().cpu().numpy().reshape(current, n, n).astype(np.float64)
        terminal_features = _lowres_features_np(terminal_np, dynamics_config)
        endpoint_features = ot_cache.target_features[target_indices]
        dist2 = np.sum((terminal_features - endpoint_features) ** 2, axis=1).astype(np.float64)

        all_states.append(slice_states.reshape(current * slices_per_path, num_pixels).detach().cpu())
        all_tau.append(slice_tau.reshape(current * slices_per_path).detach().cpu())
        all_labels.append(labels_t.view(current, 1).expand(current, slices_per_path).reshape(-1).detach().cpu())
        all_sources.append(sources[:, None, :].expand(current, slices_per_path, num_pixels).reshape(-1, num_pixels).detach().cpu())
        all_innovations.append(slice_innov.reshape(current * slices_per_path, 2, n, n).detach().cpu())
        all_masks.append(slice_masks.reshape(current * slices_per_path, 2, n, n).detach().cpu())
        all_endpoints.append(endpoints[:, None, :].expand(current, slices_per_path, num_pixels).reshape(-1, num_pixels).detach().cpu())
        all_path_indices.append(
            torch.arange(global_path_offset, global_path_offset + current, dtype=torch.long)
            .view(current, 1)
            .expand(current, slices_per_path)
            .reshape(-1)
        )
        all_starts.append(torch.as_tensor(starts_np.reshape(-1), dtype=torch.long))
        all_terminal_np.append(terminal_np)
        all_target_indices.append(target_indices.astype(np.int64))
        all_target_labels.append(target_labels_np.astype(np.int64))
        all_requested_labels.append(labels_np.astype(np.int64))
        if source_batch.indices is not None:
            all_source_indices.append(np.asarray(source_batch.indices, dtype=np.int64))
        if source_batch.labels is not None:
            all_source_labels.append(np.asarray(source_batch.labels, dtype=np.int64))
        all_terminal_dist2.append(dist2)
        global_path_offset += current
        if hasattr(bar, "set_postfix"):
            partial_dist = np.concatenate(all_terminal_dist2, axis=0)
            eps_preview = float(c0_config.terminal_epsilon)
            if eps_preview <= 0.0:
                eps_preview = _choose_epsilon_for_ess(partial_dist, float(c0_config.terminal_ess_target))
            lw = -partial_dist / (2.0 * max(eps_preview, 1e-12) ** 2)
            stats = _log_weight_stats(lw)
            clip_frac = 0.0 if total_proposed == 0 else float(total_clipped) / float(total_proposed)
            bar.set_postfix(ess=stats["ess_fraction"], eps=eps_preview, clip=clip_frac)

    terminal_dist2 = np.concatenate(all_terminal_dist2, axis=0)
    epsilon = float(c0_config.terminal_epsilon)
    if epsilon <= 0.0:
        epsilon = _choose_epsilon_for_ess(terminal_dist2, float(c0_config.terminal_ess_target))
    path_log_weights = -terminal_dist2 / (2.0 * max(epsilon, 1e-12) ** 2)
    stats = _log_weight_stats(path_log_weights)
    stable_w = np.exp(path_log_weights - float(np.max(path_log_weights)))
    weighted_terminal_dist2 = float(np.sum(stable_w * terminal_dist2) / max(float(np.sum(stable_w)), 1e-12))
    unweighted_terminal_dist2 = float(np.mean(terminal_dist2))
    log_weight_slices = torch.cat(
        [
            torch.as_tensor(path_log_weights[chunk_paths.numpy()], dtype=torch.float32)
            for chunk_paths in all_path_indices
        ],
        dim=0,
    )

    return C0TrainingCache(
        states=torch.cat(all_states, dim=0).float(),
        tau=torch.cat(all_tau, dim=0).float(),
        labels=torch.cat(all_labels, dim=0).long(),
        sources=torch.cat(all_sources, dim=0).float(),
        innovations=torch.cat(all_innovations, dim=0).float(),
        log_weights=log_weight_slices.float(),
        masks=torch.cat(all_masks, dim=0).bool(),
        endpoints=torch.cat(all_endpoints, dim=0).float(),
        terminal_dist2=torch.as_tensor(np.repeat(terminal_dist2, slices_per_path), dtype=torch.float32),
        path_indices=torch.cat(all_path_indices, dim=0).long(),
        starts=torch.cat(all_starts, dim=0).long(),
        epsilon=epsilon,
        ess_fraction=float(stats["ess_fraction"]),
        clip_fraction=0.0 if total_proposed == 0 else float(total_clipped) / float(total_proposed),
        weighted_terminal_dist2=weighted_terminal_dist2,
        unweighted_terminal_dist2=unweighted_terminal_dist2,
        source_indices=np.concatenate(all_source_indices) if all_source_indices else None,
        source_labels=np.concatenate(all_source_labels) if all_source_labels else None,
        requested_labels=np.concatenate(all_requested_labels),
        target_indices=np.concatenate(all_target_indices),
        target_labels=np.concatenate(all_target_labels),
        terminal_states=np.concatenate(all_terminal_np, axis=0),
    )


def _hybrid_endpoint_poisson_target(
    batch: dict[str, Tensor],
    dynamics_config: DirectFluxMNISTConfig,
    c0_config: Experiment11C0Config,
    *,
    sqrt_dt_eff: float,
    theta: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return a short-time endpoint residual teacher in innovation units.

    This teacher is a stabilizer for C0.  It asks for a minimum-energy edge flux
    whose divergence would move the current state toward the stored endpoint
    over the remaining time, after subtracting the free drift.  It is converted
    to the same units as the network output in ``model_output_mode=innovation``.
    """

    if "endpoints" not in batch:
        raise KeyError("hybrid endpoint teacher requires batch['endpoints']")
    n = int(dynamics_config.grid_size)
    horizon = natural_horizon(dynamics_config)
    tau_floor = max(float(c0_config.hybrid_min_tau_fraction) * horizon, 1e-12)
    remaining = batch["tau"].reshape(-1, 1).clamp_min(tau_floor)

    velocity = (batch["endpoints"] - batch["states"]) / remaining
    velocity = velocity - velocity.mean(dim=1, keepdim=True)

    free_flux = float(c0_config.reference_free_weight) * free_drift_flux_torch(batch["states"], dynamics_config)
    free_velocity = flux_divergence_torch(free_flux).reshape(velocity.shape[0], -1)
    learned_velocity = velocity - free_velocity
    learned_velocity = learned_velocity - learned_velocity.mean(dim=1, keepdim=True)

    teacher_flux = poisson_flux_from_velocity_torch(learned_velocity, grid_size=n).to(dtype=batch["states"].dtype)
    sigma_rate = float(c0_config.reference_noise_weight) * torch.sqrt((2.0 * theta * float(n * n)).clamp_min(1e-20))
    teacher = float(sqrt_dt_eff) * teacher_flux / sigma_rate.clamp_min(1e-12)
    clip = float(c0_config.hybrid_target_clip)
    if clip > 0.0:
        teacher = teacher.clamp(min=-clip, max=clip)
    return torch.where(mask, teacher, torch.zeros_like(teacher))


def c0_weighted_innovation_loss(
    model: DirectFluxUNet,
    batch: dict[str, Tensor],
    dynamics_config: DirectFluxMNISTConfig,
    c0_config: Experiment11C0Config,
) -> tuple[Tensor, dict[str, float]]:
    """Return the weighted C0 innovation loss and detached diagnostics."""

    raw = model.forward(batch["tau"], batch["states"], batch["labels"], batch["sources"])
    dt_eff = natural_horizon(dynamics_config) / float(c0_config.sample_steps) * float(c0_config.teacher_stride)
    sqrt_dt_eff = math.sqrt(max(dt_eff, 1e-20))
    output_mode = str(c0_config.model_output_mode).lower()
    if output_mode == "innovation":
        predicted_innovation_mean = raw
        eta = raw / sqrt_dt_eff
    elif output_mode == "eta":
        eta = raw
        predicted_innovation_mean = sqrt_dt_eff * eta
    else:
        raise ValueError(f"unknown model_output_mode: {c0_config.model_output_mode!r}")

    theta = harmonic_mobility_channels(batch["states"], dynamics_config)
    mask = batch["masks"] & (theta > float(c0_config.theta_mask_min))
    mask_f = mask.to(dtype=predicted_innovation_mean.dtype)

    residual = predicted_innovation_mean - batch["innovations"]
    per_slice = (residual.square() * mask_f).sum(dim=(1, 2, 3)) / mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)

    logw = batch["log_weights"].float()
    raw_weights = torch.exp(logw - torch.max(logw.detach()))
    mode = str(c0_config.loss_reweighting).lower().replace("_", "-")
    if mode in {"label-balanced", "balanced", "label"}:
        weights = _label_balanced_weights_torch(raw_weights, batch["labels"])
    elif mode in {"global", "standard", "weighted"}:
        weights = raw_weights
    elif mode in {"none", "uniform", "unweighted"}:
        weights = torch.ones_like(raw_weights)
    else:
        raise ValueError(f"unknown loss_reweighting: {c0_config.loss_reweighting!r}")
    loss_innovation = _weighted_mean_torch(per_slice, weights)

    loss_hybrid = torch.zeros((), dtype=loss_innovation.dtype, device=loss_innovation.device)
    hybrid_target_rms = torch.zeros((), dtype=loss_innovation.dtype, device=loss_innovation.device)
    hybrid_prediction_rms = torch.zeros((), dtype=loss_innovation.dtype, device=loss_innovation.device)
    if float(c0_config.hybrid_loss_weight) > 0.0 and "endpoints" in batch:
        with torch.no_grad():
            hybrid_target = _hybrid_endpoint_poisson_target(
                batch,
                dynamics_config,
                c0_config,
                sqrt_dt_eff=sqrt_dt_eff,
                theta=theta,
                mask=mask,
            )
        hybrid_residual = predicted_innovation_mean - hybrid_target
        hybrid_per_slice = (hybrid_residual.square() * mask_f).sum(dim=(1, 2, 3)) / mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
        loss_hybrid = _weighted_mean_torch(hybrid_per_slice, weights)
        denom = mask_f.detach().float().sum().clamp_min(1.0)
        hybrid_target_rms = (hybrid_target.detach().float().square() * mask_f.detach().float()).sum().div(denom).sqrt()
        hybrid_prediction_rms = (predicted_innovation_mean.detach().float().square() * mask_f.detach().float()).sum().div(denom).sqrt()

    eta_l2 = predicted_innovation_mean.square().mean()
    loss = loss_innovation + float(c0_config.hybrid_loss_weight) * loss_hybrid + float(c0_config.eta_l2_weight) * eta_l2
    with torch.no_grad():
        raw_denom = raw_weights.square().sum().clamp_min(1e-12)
        raw_ess_frac = float((raw_weights.sum().square() / raw_denom / float(raw_weights.numel())).detach().cpu())
        denom = weights.square().sum().clamp_min(1e-12)
        ess_frac = float((weights.sum().square() / denom / float(weights.numel())).detach().cpu())
        diagnostics = {
            "loss": float(loss.detach().cpu()),
            "loss_main": float(loss_innovation.detach().cpu()),
            "loss_innovation": float(loss_innovation.detach().cpu()),
            "loss_hybrid": float(loss_hybrid.detach().cpu()),
            "loss_poisson": float(loss_hybrid.detach().cpu()),
            "eta_l2": float(eta_l2.detach().cpu()),
            "eta_rms": float(eta.detach().float().square().mean().sqrt().cpu()),
            "prediction_rms": float(predicted_innovation_mean.detach().float().square().mean().sqrt().cpu()),
            "target_rms": float(batch["innovations"].detach().float().square().mean().sqrt().cpu()),
            "hybrid_target_rms": float(hybrid_target_rms.detach().cpu()),
            "hybrid_prediction_rms": float(hybrid_prediction_rms.detach().cpu()),
            "poisson_target_rms": float(hybrid_target_rms.detach().cpu()),
            "poisson_prediction_rms": float(hybrid_prediction_rms.detach().cpu()),
            "mask_fraction": float(mask_f.mean().detach().cpu()),
            "batch_ess_fraction": ess_frac,
            "batch_raw_ess_fraction": raw_ess_frac,
        }
    return loss, diagnostics


@torch.no_grad()
def simulate_c0_generation(
    model: DirectFluxUNet,
    labels: Sequence[int] | Tensor | np.ndarray,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    c0_config: Experiment11C0Config,
    device: torch.device,
    seed: int,
    source_images: np.ndarray,
    source_labels: np.ndarray,
    deterministic: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    """Generate samples using the learned C0 Brownian-shift field."""

    _disable_mkldnn_for_cpu_if_needed(device)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model.to(device)
    model.eval()
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device).reshape(-1)
    batch_size = int(labels_t.shape[0])
    n = int(dynamics_config.grid_size)
    steps = int(c0_config.sample_steps)
    horizon = natural_horizon(dynamics_config)
    dt = horizon / float(steps)
    source_batch = _sample_source_batch_torch(
        batch_size,
        dynamics_config,
        device=device,
        dtype=torch.float32,
        label_tensor=labels_t,
        source_images=source_images,
        source_labels=source_labels,
        rng=rng,
    )
    states = source_batch.masses
    source_condition = states.clone()
    initial_states = states.detach().cpu().numpy().astype(np.float64)
    trajectory: list[np.ndarray] = []
    if int(c0_config.sample_save_every) > 0:
        trajectory.append(initial_states)
    clipped = 0
    proposed = 0
    learned_step_rms_sum = 0.0
    free_step_rms_sum = 0.0
    noise_step_rms_sum = 0.0
    count = 0
    amp_enabled = bool(c0_config.use_amp and device.type == "cuda")
    bar = _progress(range(steps), total=steps, desc="sample Experiment 11", disable=not show_progress)
    from mnist.eulerian_flux_mnist import eulerian_flux_step_torch

    for step in bar:
        tau_value = max(horizon - float(step) * dt, 0.0)
        tau = torch.full((batch_size,), tau_value, dtype=states.dtype, device=device)
        context = _cuda_autocast(enabled=True) if amp_enabled else nullcontext()
        with context:
            raw_control = model.forward(tau, states, labels_t, source_condition).float()
        if float(c0_config.control_output_clip) > 0.0:
            raw_control = raw_control.clamp(
                -float(c0_config.control_output_clip),
                float(c0_config.control_output_clip),
            )
        output_mode = str(c0_config.model_output_mode).lower()
        if output_mode == "innovation":
            dt_eff_train = natural_horizon(dynamics_config) / float(c0_config.sample_steps) * float(c0_config.teacher_stride)
            eta = raw_control / math.sqrt(max(dt_eff_train, 1e-20))
        elif output_mode == "eta":
            eta = raw_control
        else:
            raise ValueError(f"unknown model_output_mode: {c0_config.model_output_mode!r}")
        eta = float(c0_config.control_strength) * eta
        theta = harmonic_mobility_channels(states, dynamics_config)
        sigma_rate = float(c0_config.reference_noise_weight) * torch.sqrt(
            (2.0 * theta * float(n * n)).clamp_min(0.0)
        )
        learned_flux = sigma_rate * eta
        learned_step_rms_sum += float((learned_flux * dt).detach().float().square().mean().sqrt().cpu())
        free_step_rms_sum += float(
            (float(c0_config.reference_free_weight) * free_drift_flux_torch(states, dynamics_config) * dt)
            .detach()
            .float()
            .square()
            .mean()
            .sqrt()
            .cpu()
        )
        noise_step_rms_sum += float(
            (float(c0_config.reference_noise_weight) * torch.sqrt((2.0 * theta * float(n * n) * dt).clamp_min(0.0)))
            .detach()
            .float()
            .square()
            .mean()
            .sqrt()
            .cpu()
        )
        count += 1
        states, c_step, p_step = eulerian_flux_step_torch(
            states,
            learned_flux,
            dt,
            dynamics_config,
            deterministic=deterministic,
            free_weight=float(c0_config.reference_free_weight),
            noise_weight=float(c0_config.reference_noise_weight),
            learned_weight=1.0,
        )
        clipped += int(c_step)
        proposed += int(p_step)
        if hasattr(bar, "set_postfix"):
            ent = float(_mass_entropy_torch(states).mean().detach().cpu())
            clip = 0.0 if proposed == 0 else clipped / proposed
            bar.set_postfix(ent=ent, clip=clip)
        if int(c0_config.sample_save_every) > 0 and ((step + 1) % int(c0_config.sample_save_every) == 0 or step + 1 == steps):
            trajectory.append(states.detach().cpu().numpy().astype(np.float64))
    diagnostics = source_batch_diagnostics(
        initial_states,
        requested_labels=labels_t.detach().cpu().numpy(),
        source_indices=source_batch.indices,
        source_labels=source_batch.labels,
    )
    learned_rms = learned_step_rms_sum / max(count, 1)
    free_rms = free_step_rms_sum / max(count, 1)
    noise_rms = noise_step_rms_sum / max(count, 1)
    samples = states.detach().cpu().numpy().astype(np.float64)
    return {
        "samples": samples,
        "labels": labels_t.detach().cpu().numpy().astype(np.int64),
        "sources": initial_states,
        "trajectory": None if int(c0_config.sample_save_every) <= 0 else np.stack(trajectory, axis=0),
        "clipping_fraction": 0.0 if proposed == 0 else float(clipped) / float(proposed),
        "learned_step_rms": learned_rms,
        "free_step_rms": free_rms,
        "noise_step_rms": noise_rms,
        "free_to_learned_ratio": free_rms / max(learned_rms, 1e-12),
        "noise_to_learned_ratio": noise_rms / max(learned_rms, 1e-12),
        "sample_entropy": float(_mass_entropy_torch(states).mean().detach().cpu()),
        "sample_total_variation": float(image_total_variation(states, grid_size=n).detach().cpu()),
        **diagnostics,
    }


def _write_cache_diagnostics(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_experiment11_c0(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    c0_config: Experiment11C0Config,
    out_dir: Path,
    device: torch.device,
    seed: int,
    show_progress: bool = True,
) -> tuple[DirectFluxUNet, dict[str, list[float]], list[dict[str, float | int]]]:
    """Train the Experiment 11 C0 model."""

    torch.manual_seed(int(seed))
    np_rng = np.random.default_rng(int(seed))
    _disable_mkldnn_for_cpu_if_needed(device)
    ot_cache = ClasswiseOTCache(
        class_indices=tuple(np.flatnonzero(np.asarray(dataset_labels) == digit).astype(np.int64) for digit in range(10)),
        target_features=_lowres_features_np(dataset_images, dynamics_config),
        class_means=np.zeros((10, int(dynamics_config.grid_size), int(dynamics_config.grid_size)), dtype=np.float64),
    )
    # Use the public builder when available to also get class means.
    try:
        from mnist.eulerian_flux_mnist import build_classwise_ot_cache

        ot_cache = build_classwise_ot_cache(dataset_images, dataset_labels, dynamics_config)
    except Exception:
        pass

    model = DirectFluxUNet(dynamics_config, base_channels=int(c0_config.base_channels)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(c0_config.learning_rate),
        weight_decay=float(c0_config.weight_decay),
    )
    scaler = _make_cuda_grad_scaler(bool(c0_config.use_amp and device.type == "cuda"))
    ema_state: dict[str, Tensor] | None = {name: value.detach().clone() for name, value in model.state_dict().items()}
    history: dict[str, list[float]] = {
        "loss": [],
        "loss_main": [],
        "loss_innovation": [],
        "loss_hybrid": [],
        "loss_poisson": [],
        "eta_l2": [],
        "eta_rms": [],
        "prediction_rms": [],
        "target_rms": [],
        "hybrid_target_rms": [],
        "hybrid_prediction_rms": [],
        "poisson_target_rms": [],
        "poisson_prediction_rms": [],
        "mask_fraction": [],
        "batch_ess_fraction": [],
        "batch_raw_ess_fraction": [],
        "cache_ess_fraction": [],
        "cache_clip_fraction": [],
    }
    cache_rows: list[dict[str, float | int]] = []
    cache: C0TrainingCache | None = None

    def refresh_cache(step: int) -> C0TrainingCache:
        started = time.perf_counter()
        built = build_c0_training_cache(
            dataset_images=dataset_images,
            dataset_labels=dataset_labels,
            ot_cache=ot_cache,
            dynamics_config=dynamics_config,
            c0_config=c0_config,
            device=device,
            rng=np_rng,
            show_progress=show_progress,
        )
        elapsed = time.perf_counter() - started
        row: dict[str, float | int] = {
            "step": int(step),
            "cache_size": int(built.size),
            "epsilon": float(built.epsilon),
            "ess_fraction": float(built.ess_fraction),
            "clip_fraction": float(built.clip_fraction),
            "weighted_terminal_dist2": float(built.weighted_terminal_dist2),
            "unweighted_terminal_dist2": float(built.unweighted_terminal_dist2),
            "seconds": float(elapsed),
        }
        slices_per_path_diag = max(1, int(c0_config.time_slices_per_path))
        path_dist2_diag = built.terminal_dist2.numpy().reshape(-1)[::slices_per_path_diag]
        path_logw_diag = built.log_weights.numpy().reshape(-1)[::slices_per_path_diag]
        requested_diag = built.requested_labels
        target_diag = built.target_labels
        row.update(_histogram_dict(requested_diag, prefix="requested_label_count"))
        row.update(_histogram_dict(target_diag, prefix="target_label_count"))
        row.update(_weighted_histogram_dict(requested_diag, path_logw_diag, prefix="requested_weighted_label_mass"))
        row.update(_weighted_histogram_dict(target_diag, path_logw_diag, prefix="target_weighted_label_mass"))
        row.update(_per_label_weight_stats(requested_diag, path_logw_diag, path_dist2_diag, prefix="requested"))
        row.update(_per_label_weight_stats(target_diag, path_logw_diag, path_dist2_diag, prefix="target"))
        if requested_diag is not None and target_diag is not None:
            row["target_label_mismatch_fraction"] = float(np.mean(np.asarray(requested_diag) != np.asarray(target_diag)))
        cache_rows.append(row)
        np.savez_compressed(
            out_dir / f"experiment11_c0_cache_step{int(step):06d}_diagnostics.npz",
            terminal_dist2=built.terminal_dist2.numpy(),
            log_weights=built.log_weights.numpy(),
            starts=built.starts.numpy(),
            path_indices=built.path_indices.numpy(),
            requested_labels=np.asarray([] if built.requested_labels is None else built.requested_labels, dtype=np.int64),
            target_indices=np.asarray([] if built.target_indices is None else built.target_indices, dtype=np.int64),
            target_labels=np.asarray([] if built.target_labels is None else built.target_labels, dtype=np.int64),
            source_indices=np.asarray([] if built.source_indices is None else built.source_indices, dtype=np.int64),
            source_labels=np.asarray([] if built.source_labels is None else built.source_labels, dtype=np.int64),
            epsilon=np.asarray([built.epsilon], dtype=np.float64),
            ess_fraction=np.asarray([built.ess_fraction], dtype=np.float64),
        )
        if bool(c0_config.save_cache_previews) and built.terminal_states is not None:
            # Optional endpoint previews for debugging cache quality.  Use the
            # true requested/target labels; low-frequency sources do not carry
            # source labels, and earlier zero-label fallbacks made previews look
            # like a target sampler failure.
            try:
                slices_per_path = max(1, int(c0_config.time_slices_per_path))
                path_dist2 = built.terminal_dist2.numpy().reshape(-1)[::slices_per_path]
                path_lw = -path_dist2 / (2.0 * max(float(built.epsilon), 1e-12) ** 2)
                requested_labels = built.requested_labels if built.requested_labels is not None else np.zeros((built.terminal_states.shape[0],), dtype=np.int64)
                target_labels = built.target_labels if built.target_labels is not None else requested_labels

                save_flux_samples_grid(
                    built.terminal_states,
                    requested_labels,
                    out_dir / f"experiment11_c0_free_endpoints_step{int(step):06d}.png",
                    grid_size=int(dynamics_config.grid_size),
                    max_images=64,
                )

                global_order = np.argsort(-path_lw)[:64]
                balanced_order = _label_balanced_preview_order(path_lw, target_labels, per_label=8)[:80]

                save_flux_samples_grid(
                    built.terminal_states[global_order],
                    requested_labels[global_order],
                    out_dir / f"experiment11_c0_global_weighted_terminals_step{int(step):06d}.png",
                    grid_size=int(dynamics_config.grid_size),
                    max_images=64,
                )
                if built.target_indices is not None:
                    save_flux_samples_grid(
                        dataset_images[built.target_indices[global_order]],
                        target_labels[global_order],
                        out_dir / f"experiment11_c0_global_weighted_targets_step{int(step):06d}.png",
                        grid_size=int(dynamics_config.grid_size),
                        max_images=64,
                    )

                save_flux_samples_grid(
                    built.terminal_states[balanced_order],
                    requested_labels[balanced_order],
                    out_dir / f"experiment11_c0_weighted_terminals_step{int(step):06d}.png",
                    grid_size=int(dynamics_config.grid_size),
                    max_images=80,
                )
                if built.target_indices is not None:
                    save_flux_samples_grid(
                        dataset_images[built.target_indices[balanced_order]],
                        target_labels[balanced_order],
                        out_dir / f"experiment11_c0_weighted_targets_step{int(step):06d}.png",
                        grid_size=int(dynamics_config.grid_size),
                        max_images=80,
                    )
            except Exception as exc:
                print(f"Warning: could not save cache preview: {exc}")
        return built

    cache = refresh_cache(0)
    amp_enabled = bool(c0_config.use_amp and device.type == "cuda")
    bar = _progress(range(int(c0_config.train_steps)), total=int(c0_config.train_steps), desc="train Experiment 11", disable=not show_progress)
    for step in bar:
        if step > 0 and int(c0_config.cache_refresh_every) > 0 and step % int(c0_config.cache_refresh_every) == 0:
            cache = refresh_cache(step)
        assert cache is not None
        idx = _sample_cache_indices(cache, int(c0_config.batch_size), device)
        batch = _cache_batch(cache, idx, device)
        optimizer.zero_grad(set_to_none=True)
        context = _cuda_autocast(enabled=True) if amp_enabled else nullcontext()
        with context:
            loss, diagnostics = c0_weighted_innovation_loss(model, batch, dynamics_config, c0_config)
        scaler.scale(loss).backward()
        if float(c0_config.grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(c0_config.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        update_ema_state(ema_state, model, decay=float(c0_config.ema_decay))
        for key in [
            "loss",
            "loss_main",
            "loss_innovation",
            "loss_hybrid",
            "loss_poisson",
            "eta_l2",
            "eta_rms",
            "prediction_rms",
            "target_rms",
            "hybrid_target_rms",
            "hybrid_prediction_rms",
            "poisson_target_rms",
            "poisson_prediction_rms",
            "mask_fraction",
            "batch_ess_fraction",
            "batch_raw_ess_fraction",
        ]:
            history[key].append(float(diagnostics[key]))
        history["cache_ess_fraction"].append(float(cache.ess_fraction))
        history["cache_clip_fraction"].append(float(cache.clip_fraction))
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(
                loss=float(diagnostics["loss"]),
                pred=float(diagnostics["prediction_rms"]),
                eta=float(diagnostics["eta_rms"]),
                ess=float(diagnostics["batch_ess_fraction"]),
                poi=float(diagnostics["loss_poisson"]),
                cache=float(cache.ess_fraction),
            )
    if bool(c0_config.use_ema_for_sampling) and ema_state is not None:
        model.load_state_dict(ema_state, strict=False)
    model.eval()
    return model, history, cache_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--examples-per-class", type=int, default=1000)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment11"))
    parser.add_argument("--run-name", type=str, default="c0-weighted-innovation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--grid-size", type=int, default=28)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--edge-alpha-mode", choices=("legacy", "grid"), default="legacy")
    parser.add_argument("--horizon-scale", type=float, default=1.0)
    parser.add_argument("--mass-floor", type=float, default=1e-8)
    parser.add_argument("--limiter-fraction", type=float, default=0.25)

    parser.add_argument("--source-mode", type=str, default="lowfreq")
    parser.add_argument("--source-lowfreq-size", type=int, default=7)
    parser.add_argument("--source-blur-sigma", type=float, default=1.0)
    parser.add_argument("--source-uniform-mix", type=float, default=0.15)
    parser.add_argument("--source-concentration", type=float, default=1.0)
    parser.add_argument("--no-condition-on-source", action="store_true")
    parser.add_argument("--upsample-mode", choices=("transpose", "resize-conv"), default="resize-conv")

    parser.add_argument("--ot-cost-mode", choices=("lowres", "pixel"), default="lowres")
    parser.add_argument("--ot-match-mode", choices=("minibatch", "nearest", "topk"), default="nearest")
    parser.add_argument("--ot-nearest-top-k", type=int, default=1)
    parser.add_argument("--ot-lowres-size", type=int, default=7)
    parser.add_argument("--ot-blur-sigma", type=float, default=1.0)
    parser.add_argument("--ot-com-weight", type=float, default=0.25)

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--cache-paths", type=int, default=4096)
    parser.add_argument("--cache-batch-size", type=int, default=128)
    parser.add_argument("--cache-refresh-every", type=int, default=500)
    parser.add_argument("--teacher-stride", type=int, default=8)
    parser.add_argument("--time-slices-per-path", type=int, default=4)
    parser.add_argument("--terminal-epsilon", type=float, default=0.0, help="<=0 chooses epsilon by ESS calibration")
    parser.add_argument("--terminal-ess-target", type=float, default=0.25)
    parser.add_argument("--eta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--theta-mask-min", type=float, default=1e-12)
    parser.add_argument(
        "--loss-reweighting",
        choices=("label-balanced", "global", "none"),
        default="label-balanced",
        help="How terminal weights are normalized in the training loss; label-balanced prevents easy digits dominating.",
    )
    parser.add_argument("--no-label-balanced-loss", dest="loss_reweighting", action="store_const", const="global")
    parser.add_argument("--hybrid-loss-weight", type=float, default=0.10, help="Weight for the endpoint residual Poisson stabilizer; 0 disables")
    parser.add_argument("--hybrid-target-clip", type=float, default=5.0)
    parser.add_argument("--hybrid-min-tau-fraction", type=float, default=0.03)
    # Backward-compatible aliases used in the previous discussion.
    parser.add_argument("--hybrid-poisson-weight", dest="hybrid_loss_weight", type=float)
    parser.add_argument("--hybrid-poisson-tau-floor-fraction", dest="hybrid_min_tau_fraction", type=float)
    parser.add_argument("--hybrid-poisson-target-clip", dest="hybrid_target_clip", type=float)
    parser.add_argument(
        "--model-output-mode",
        choices=("innovation", "eta"),
        default="innovation",
        help="innovation trains the network to predict sqrt(dt_eff)*eta; eta keeps the old direct eta output",
    )
    parser.add_argument("--control-strength", type=float, default=1.0, help="Multiplier applied to learned control at generation")
    parser.add_argument("--control-output-clip", type=float, default=5.0, help="Clip raw control output at generation; <=0 disables")

    parser.add_argument("--reference-free-weight", type=float, default=0.03)
    parser.add_argument("--reference-noise-weight", type=float, default=0.005)
    parser.add_argument("--sample-steps", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-save-every", type=int, default=0)
    parser.add_argument("--save-cache-previews", action="store_true")
    parser.add_argument("--deterministic-sampling", action="store_true")
    parser.add_argument("--adaptive-sampling", action="store_true", default=True)
    parser.add_argument("--no-adaptive-sampling", dest="adaptive_sampling", action="store_false")
    parser.add_argument("--clip-target", type=float, default=0.03)
    parser.add_argument("--max-substeps", type=int, default=4)

    parser.add_argument("--use-classifier-diagnostics", action="store_true")
    parser.add_argument("--classifier-cache-path", type=Path, default=None)
    parser.add_argument("--classifier-train-epochs", type=int, default=2)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    return parser.parse_args(argv)


def _experiment_config_from_args(args: argparse.Namespace) -> Experiment11C0Config:
    return Experiment11C0Config(
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        base_channels=int(args.base_channels),
        cache_paths=int(args.cache_paths),
        cache_batch_size=int(args.cache_batch_size),
        cache_refresh_every=int(args.cache_refresh_every),
        teacher_stride=int(args.teacher_stride),
        time_slices_per_path=int(args.time_slices_per_path),
        terminal_epsilon=float(args.terminal_epsilon),
        terminal_ess_target=float(args.terminal_ess_target),
        eta_l2_weight=float(args.eta_l2_weight),
        theta_mask_min=float(args.theta_mask_min),
        loss_reweighting=str(args.loss_reweighting),
        hybrid_loss_weight=float(args.hybrid_loss_weight),
        hybrid_min_tau_fraction=float(args.hybrid_min_tau_fraction),
        hybrid_target_clip=float(args.hybrid_target_clip),
        model_output_mode=str(args.model_output_mode),
        control_strength=float(args.control_strength),
        control_output_clip=float(args.control_output_clip),
        reference_free_weight=float(args.reference_free_weight),
        reference_noise_weight=float(args.reference_noise_weight),
        sample_steps=int(args.sample_steps),
        num_samples=int(args.num_samples),
        sample_save_every=int(args.sample_save_every),
        save_cache_previews=bool(args.save_cache_previews),
        seed=int(args.seed),
        sample_seed=int(args.sample_seed),
        use_amp=not bool(args.no_amp),
        use_ema_for_sampling=True,
        ema_decay=float(args.ema_decay),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir, metadata = make_experiment11_run_dir(args.runs_root, args.run_name)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    show_progress = not bool(args.no_progress)
    dynamics_config = _make_dynamics_config(args)
    c0_config = _experiment_config_from_args(args)

    metadata.update(
        {
            "experiment": "experiment11_c0_weighted_innovation",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "args": {key: _serializable(value) for key, value in vars(args).items()},
            "dynamics_config": asdict(dynamics_config),
            "c0_config": asdict(c0_config),
            "theory_notes": ["main.tex", "eulerian_approx.tex", "experiment_c0_weighted_innovation.tex"],
        }
    )
    with (run_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=_serializable)

    print(f"Experiment 11/C0 weighted innovation on device={device}")
    print(f"Run directory: {run_dir}")
    dataset = load_mnist_measure_dataset(
        args.data_root,
        max_train=args.max_train,
        examples_per_class=args.examples_per_class,
        download=bool(args.download),
        seed=int(args.seed),
    )
    train_images = np.asarray(dataset.train_images, dtype=np.float64)
    train_labels = np.asarray(dataset.train_labels, dtype=np.int64)
    print(f"Loaded MNIST measures: {train_images.shape[0]} train examples")

    classifier: TinyMNISTClassifier | None = None
    if bool(args.use_classifier_diagnostics):
        cache_path = args.classifier_cache_path if args.classifier_cache_path is not None else run_dir / "experiment11_mnist_classifier.pt"
        classifier = train_or_load_mnist_classifier(
            train_images,
            train_labels,
            grid_size=int(dynamics_config.grid_size),
            cache_path=cache_path,
            train_epochs=int(args.classifier_train_epochs),
            batch_size=int(args.classifier_batch_size),
            lr=float(args.classifier_lr),
            device=device,
            seed=int(args.seed),
            show_progress=show_progress,
        )

    model, history, cache_rows = train_experiment11_c0(
        dataset_images=train_images,
        dataset_labels=train_labels,
        dynamics_config=dynamics_config,
        c0_config=c0_config,
        out_dir=run_dir,
        device=device,
        seed=int(args.seed),
        show_progress=show_progress,
    )
    _write_cache_diagnostics(run_dir / "experiment11_c0_cache_diagnostics.csv", cache_rows)

    generation: dict[str, object] | None = None
    metrics: dict[str, object] = {}
    if int(c0_config.num_samples) > 0:
        labels = np.arange(int(c0_config.num_samples), dtype=np.int64) % 10
        generation = simulate_c0_generation(
            model,
            labels,
            dynamics_config=dynamics_config,
            c0_config=c0_config,
            device=device,
            seed=int(args.sample_seed),
            source_images=train_images,
            source_labels=train_labels,
            deterministic=bool(args.deterministic_sampling),
            show_progress=show_progress,
        )
        metrics = {
            key: value for key, value in generation.items() if isinstance(value, (int, float, np.integer, np.floating))
        }
        if classifier is not None:
            metrics.update(
                classifier_generation_metrics(
                    generation["samples"],
                    generation["labels"],
                    classifier,
                    grid_size=int(dynamics_config.grid_size),
                    device=device,
                )
            )
    history_path = run_dir / "experiment11_c0_history.json"
    with history_path.open("w") as handle:
        json.dump(history, handle, indent=2)
    metrics_path = run_dir / "experiment11_c0_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump({key: _serializable(value) for key, value in metrics.items()}, handle, indent=2)

    ckpt_path = run_dir / "experiment11_c0_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dynamics_config": asdict(dynamics_config),
            "c0_config": asdict(c0_config),
            "history": history,
            "metrics": metrics,
        },
        ckpt_path,
    )
    samples_path = run_dir / "experiment11_c0_samples.npz"
    png_path = run_dir / "experiment11_c0_samples.png"
    if generation is not None:
        np.savez_compressed(
            samples_path,
            samples=np.asarray(generation["samples"], dtype=np.float64),
            labels=np.asarray(generation["labels"], dtype=np.int64),
            sources=np.asarray(generation["sources"], dtype=np.float64),
            trajectory=np.asarray(generation["trajectory"], dtype=np.float64) if generation["trajectory"] is not None else np.empty((0,)),
            metrics=np.asarray(json.dumps({key: _serializable(value) for key, value in metrics.items()})),
        )
        save_flux_samples_grid(
            np.asarray(generation["samples"], dtype=np.float64),
            np.asarray(generation["labels"], dtype=np.int64),
            png_path,
            grid_size=int(dynamics_config.grid_size),
            max_images=int(c0_config.num_samples),
        )
        if generation["trajectory"] is not None:
            np.savez_compressed(run_dir / "experiment11_c0_trajectory.npz", trajectory=generation["trajectory"])

    print("Experiment 11/C0 complete")
    print(f"  checkpoint: {ckpt_path}")
    if generation is not None:
        print(f"  samples:    {samples_path}")
        print(f"  preview:    {png_path}")
    print(f"  metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
