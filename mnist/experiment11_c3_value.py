from __future__ import annotations

r"""Experiment 11/C3: scalar heat-potential/value estimator for MNIST.

This is the next honest Experiment 11 variant after the C0/C1/C2 score-estimator
experiments.  It trains a scalar network

    ell_phi(t, s, y, z) ~= log E_s[g_h(S_T) | y, z]

from branch Monte Carlo value targets.  At sampling time the conditioning flux is
computed by differentiating this scalar with respect to the current mass state and
then applying the finite-volume h-transform formula

    J_e = value_flux_scale * (2 / h) * theta_e(s) * partial_e^h ell_phi(t, s).

The module is intentionally standalone so it can be run without changing the C0
score-experiment implementation:

    python -m mnist.experiment11_c3_value --run-name c3-value-smoke
"""

import argparse
import csv
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mnist.eulerian_flux_mnist import (
    ClasswiseOTCache,
    DirectFluxMNISTConfig,
    _cuda_autocast,
    _disable_mkldnn_for_cpu_if_needed,
    _lowres_features_np,
    _make_cuda_grad_scaler,
    _mass_entropy_torch,
    _ot_coupled_target_indices,
    _progress,
    _sample_source_batch_torch,
    build_classwise_ot_cache,
    classifier_generation_metrics,
    eulerian_flux_step_torch,
    harmonic_mobility_channels,
    image_total_variation,
    load_mnist_measure_dataset,
    natural_horizon,
    save_flux_samples_grid,
    sample_checkerboard_energy_torch,
    train_or_load_mnist_classifier,
    update_ema_state,
)


@dataclass(frozen=True)
class Experiment11C3Config:
    train_steps: int = 10_000
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    base_channels: int = 48
    cache_paths: int = 1024
    cache_batch_size: int = 64
    cache_refresh_every: int = 1000
    branch_count: int = 32
    branch_batch_size: int = 128
    endpoint_count_per_state: int = 16
    terminal_epsilon: float = 0.0
    terminal_ess_target: float = 0.25
    terminal_epsilon_mode: str = "branch-ess"  # branch-ess, global-ess, fixed
    terminal_feature_mode: str = "multiscale"  # lowres, multiscale
    terminal_ms_weight_7: float = 1.0
    terminal_ms_weight_14: float = 0.5
    terminal_ms_weight_28: float = 0.25
    terminal_ms_weight_com: float = 0.25
    terminal_ms_blur_28: float = 0.75
    value_center_by_label: bool = True
    value_loss_weight: float = 1.0
    value_l2_weight: float = 0.0
    value_target_clip: float = 40.0
    value_flux_scale: float = 1.0
    value_flux_clip: float = 50.0
    reference_free_weight: float = 0.03
    reference_noise_weight: float = 0.02
    sample_steps: int = 256
    num_samples: int = 64
    sampling_weights: str = "raw"  # raw, ema, both
    sample_value_flux_scales: str = "1,2,5,10"
    save_sampling_ablations: bool = False
    save_cache_previews: bool = False
    seed: int = 0
    sample_seed: int = 1
    use_amp: bool = True
    ema_decay: float = 0.999


@dataclass
class C3ValueCache:
    states: Tensor
    tau: Tensor
    labels: Tensor
    sources: Tensor
    value_targets: Tensor
    log_weights: Tensor
    target_images: np.ndarray
    terminal_states: np.ndarray
    requested_labels: np.ndarray
    target_labels: np.ndarray
    target_indices: np.ndarray
    epsilon: float
    value_target_mean: float
    value_target_std: float
    branch_ess_fraction: np.ndarray
    branch_value_log_mean: np.ndarray
    branch_unweighted_terminal_dist2: np.ndarray
    branch_weighted_terminal_dist2: np.ndarray
    clip_fraction: float

    @property
    def size(self) -> int:
        return int(self.states.shape[0])


def _serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def make_experiment11_c3_run_dir(runs_root: Path, run_name: str | None) -> tuple[Path, dict[str, object]]:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    nickname = "c3-value" if not run_name else str(run_name).strip().replace(" ", "-")
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


def _parse_label_sequence(text: str, num_samples: int) -> np.ndarray:
    if text == "cycle":
        return np.arange(num_samples, dtype=np.int64) % 10
    if text == "random":
        rng = np.random.default_rng(0)
        return rng.integers(0, 10, size=num_samples, dtype=np.int64)
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        vals = [0]
    return np.asarray([vals[i % len(vals)] for i in range(num_samples)], dtype=np.int64)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ValuePotentialUNet(nn.Module):
    """Small source/label/time-conditioned scalar potential network."""

    def __init__(self, config: DirectFluxMNISTConfig, base_channels: int = 48):
        super().__init__()
        self.config = config
        self.base_channels = int(base_channels)
        in_channels = 1 + 1 + 1 + 1 + 1 + 10
        c = int(base_channels)
        self.enc1 = _ConvBlock(in_channels, c)
        self.down1 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)
        self.enc2 = _ConvBlock(2 * c, 2 * c)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1)
        self.mid = _ConvBlock(4 * c, 4 * c)
        self.head_map = nn.Sequential(
            nn.Conv2d(4 * c, 2 * c, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv2d(2 * c, 1, 1),
        )
        self.head_vec = nn.Sequential(
            nn.Linear(4 * c + 10 + 1, 2 * c),
            nn.SiLU(),
            nn.Linear(2 * c, 1),
        )

    def _inputs(self, tau: Tensor, states: Tensor, labels: Tensor, sources: Tensor) -> Tensor:
        n = int(self.config.grid_size)
        b = int(states.shape[0])
        s_img = states.reshape(b, 1, n, n)
        z_img = sources.reshape(b, 1, n, n)
        tau_norm = (tau.reshape(b, 1, 1, 1) / max(float(natural_horizon(self.config)), 1e-12)).expand(b, 1, n, n)
        one_hot = F.one_hot(labels.long().clamp(0, 9), num_classes=10).float().reshape(b, 10, 1, 1).expand(b, 10, n, n)
        return torch.cat(
            [
                s_img * float(n * n),
                torch.log(s_img.clamp_min(float(self.config.mass_floor))),
                z_img * float(n * n),
                torch.log(z_img.clamp_min(float(self.config.mass_floor))),
                tau_norm,
                one_hot,
            ],
            dim=1,
        )

    def forward(self, tau: Tensor, states: Tensor, labels: Tensor, sources: Tensor) -> Tensor:
        x = self._inputs(tau, states, labels, sources)
        h = self.enc1(x)
        h = F.silu(self.down1(h))
        h = self.enc2(h)
        h = F.silu(self.down2(h))
        h = self.mid(h)
        pooled = h.mean(dim=(2, 3))
        tau_scalar = (tau.reshape(-1, 1) / max(float(natural_horizon(self.config)), 1e-12)).to(dtype=pooled.dtype)
        one_hot = F.one_hot(labels.long().clamp(0, 9), num_classes=10).to(dtype=pooled.dtype)
        value_vec = self.head_vec(torch.cat([pooled, one_hot, tau_scalar], dim=1)).squeeze(1)
        value_map = self.head_map(h).mean(dim=(1, 2, 3))
        return value_vec + value_map


def _periodic_blur_torch(x: Tensor, sigma: float) -> Tensor:
    if sigma <= 0.0:
        return x
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    c = int(x.shape[1])
    kh = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    y = F.pad(x, (radius, radius, 0, 0), mode="circular")
    y = F.conv2d(y, kh, groups=c)
    y = F.pad(y, (0, 0, radius, radius), mode="circular")
    y = F.conv2d(y, kv, groups=c)
    return y


def _terminal_features_np(images: np.ndarray, config: DirectFluxMNISTConfig, c3: Experiment11C3Config) -> np.ndarray:
    arr = np.asarray(images, dtype=np.float64)
    n = int(config.grid_size)
    if arr.ndim == 2:
        arr = arr.reshape(-1, n, n)
    if str(c3.terminal_feature_mode).lower() == "lowres":
        return _lowres_features_np(arr, config)
    with torch.no_grad():
        t = torch.as_tensor(arr[:, None], dtype=torch.float32)
        feats: list[np.ndarray] = []
        for size, weight in [(7, c3.terminal_ms_weight_7), (14, c3.terminal_ms_weight_14)]:
            if weight <= 0.0:
                continue
            y = _periodic_blur_torch(t, sigma=float(config.ot_blur_sigma)) if config.ot_blur_sigma > 0 else t
            y = F.interpolate(y, size=(size, size), mode="area")
            f = y.reshape(y.shape[0], -1).numpy().astype(np.float64)
            f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
            feats.append(math.sqrt(float(weight)) * f)
        if float(c3.terminal_ms_weight_28) > 0.0:
            y = _periodic_blur_torch(t, sigma=float(c3.terminal_ms_blur_28))
            f = y.reshape(y.shape[0], -1).numpy().astype(np.float64)
            f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
            feats.append(math.sqrt(float(c3.terminal_ms_weight_28)) * f)
        if float(c3.terminal_ms_weight_com) > 0.0:
            yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
            xx = (xx + 0.5) / float(n)
            yy = (yy + 0.5) / float(n)
            denom = np.maximum(arr.sum(axis=(1, 2)), 1e-12)
            com_x = (arr * xx).sum(axis=(1, 2)) / denom
            com_y = (arr * yy).sum(axis=(1, 2)) / denom
            feats.append(math.sqrt(float(c3.terminal_ms_weight_com)) * np.stack([com_x, com_y], axis=1))
    return np.concatenate(feats, axis=1).astype(np.float64)


def _choose_epsilon_for_ess(logit_dist2: np.ndarray, target_fraction: float) -> float:
    values = np.asarray(logit_dist2, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.max(finite) <= 0.0:
        return 1.0
    target = float(np.clip(target_fraction, 1.0 / max(finite.size, 1), 0.999))
    scale = math.sqrt(max(float(np.median(finite)), 1e-12))
    lo = max(scale * 1e-3, 1e-6)
    hi = max(scale * 1e3, 1.0)
    for _ in range(64):
        mid = math.sqrt(lo * hi)
        lw = -finite / (2.0 * mid * mid)
        w = np.exp(lw - float(np.max(lw)))
        ess = float(w.sum() ** 2 / max(float(finite.size) * float((w * w).sum()), 1e-12))
        if ess < target:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _branch_ess(log_weights: np.ndarray) -> np.ndarray:
    lw = np.asarray(log_weights, dtype=np.float64)
    stable = lw - lw.max(axis=1, keepdims=True)
    w = np.exp(stable)
    return (w.sum(axis=1) ** 2) / np.maximum(w.shape[1] * np.sum(w * w, axis=1), 1e-12)


def _select_endpoint_indices(
    source_np: np.ndarray,
    labels_np: np.ndarray,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    rng: np.random.Generator,
    ot_cache: ClasswiseOTCache,
    count: int,
) -> np.ndarray:
    out = []
    for _ in range(max(1, int(count))):
        idx = _ot_coupled_target_indices(
            source_np,
            labels_np,
            dataset_images,
            dataset_labels,
            config,
            rng=rng,
            ot_cache=ot_cache,
        )
        out.append(idx.astype(np.int64))
    return np.stack(out, axis=1)


@torch.no_grad()
def _free_step(states: Tensor, dt: float, config: DirectFluxMNISTConfig, free_weight: float, noise_weight: float) -> tuple[Tensor, int, int]:
    zero = torch.zeros(states.shape[0], 2, int(config.grid_size), int(config.grid_size), dtype=states.dtype, device=states.device)
    out, clipped, proposed = eulerian_flux_step_torch(
        states,
        zero,
        float(dt),
        config,
        deterministic=False,
        free_weight=float(free_weight),
        noise_weight=float(noise_weight),
        learned_weight=0.0,
    )
    return out, int(clipped), int(proposed)


def build_c3_value_cache(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    ot_cache: ClasswiseOTCache,
    dynamics_config: DirectFluxMNISTConfig,
    c3_config: Experiment11C3Config,
    device: torch.device,
    rng: np.random.Generator,
    show_progress: bool = True,
) -> C3ValueCache:
    n = int(dynamics_config.grid_size)
    num_pixels = n * n
    entries = int(c3_config.cache_paths)
    chunk_size = max(1, int(c3_config.cache_batch_size))
    branch_count = max(1, int(c3_config.branch_count))
    endpoint_count = max(1, int(c3_config.endpoint_count_per_state))
    branch_batch_size = max(branch_count, int(c3_config.branch_batch_size))
    entries_per_branch_batch = max(1, branch_batch_size // branch_count)
    steps = int(c3_config.sample_steps)
    horizon = natural_horizon(dynamics_config)
    dt = horizon / float(steps)
    dtype = torch.float32

    target_feature_all = _terminal_features_np(dataset_images.reshape(-1, n, n), dynamics_config, c3_config)
    states_out: list[Tensor] = []
    tau_out: list[Tensor] = []
    labels_out: list[Tensor] = []
    sources_out: list[Tensor] = []
    branch_dist_chunks: list[np.ndarray] = []
    branch_reward_base_chunks: list[np.ndarray] = []
    terminal_chunks: list[np.ndarray] = []
    endpoint_preview_chunks: list[np.ndarray] = []
    target_idx_chunks: list[np.ndarray] = []
    target_label_chunks: list[np.ndarray] = []
    requested_label_chunks: list[np.ndarray] = []
    total_clipped = 0
    total_proposed = 0

    starts = list(range(0, entries, chunk_size))
    bar = _progress(starts, total=len(starts), desc="build C3 value cache", disable=not show_progress)
    for start in bar:
        current = min(chunk_size, entries - int(start))
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
        target_indices = _select_endpoint_indices(
            source_np,
            labels_np,
            dataset_images,
            dataset_labels,
            dynamics_config,
            rng,
            ot_cache,
            endpoint_count,
        )
        target_labels = np.asarray(dataset_labels[target_indices], dtype=np.int64)
        if np.any(target_labels != labels_np[:, None]):
            raise RuntimeError("target sampler produced label mismatch in C3 cache")
        starts_np = rng.integers(0, steps, size=current, dtype=np.int64)
        max_start = int(starts_np.max(initial=0))
        starts_t = torch.as_tensor(starts_np, dtype=torch.long, device=device)
        prefix_states = torch.empty(current, num_pixels, dtype=dtype, device=device)
        for step in range(max_start + 1):
            active = starts_t == int(step)
            if bool(active.any()):
                prefix_states[active] = states[active]
            if step < max_start:
                states, clipped, proposed = _free_step(
                    states,
                    dt,
                    dynamics_config,
                    c3_config.reference_free_weight,
                    c3_config.reference_noise_weight,
                )
                total_clipped += clipped
                total_proposed += proposed
        states_out.append(prefix_states.detach().cpu())
        tau_out.append(torch.as_tensor(horizon - starts_np.astype(np.float64) * dt, dtype=torch.float32))
        labels_out.append(torch.as_tensor(labels_np, dtype=torch.long))
        sources_out.append(sources.detach().cpu())
        target_idx_chunks.append(target_indices[:, 0].copy())
        target_label_chunks.append(target_labels[:, 0].copy())
        requested_label_chunks.append(labels_np.copy())
        endpoint_preview_chunks.append(np.asarray(dataset_images[target_indices[:, 0]], dtype=np.float64).reshape(current, n, n))

        chunk_branch_dist2 = np.empty((current, branch_count, endpoint_count), dtype=np.float64)
        chunk_branch_reward_base = np.empty((current, branch_count), dtype=np.float64)
        chunk_terminal = np.empty((current, branch_count, n, n), dtype=np.float64)
        for local_start in range(0, current, entries_per_branch_batch):
            local = min(entries_per_branch_batch, current - local_start)
            ids = torch.arange(local_start, local_start + local, device=device)
            branch_states = prefix_states.index_select(0, ids).repeat_interleave(branch_count, dim=0)
            branch_starts = starts_t.index_select(0, ids).repeat_interleave(branch_count)
            max_remaining = int((steps - starts_np[local_start : local_start + local]).max(initial=0))
            for local_step in range(max_remaining):
                active = local_step < (steps - branch_starts)
                if not bool(active.any()):
                    continue
                active_idx = torch.nonzero(active, as_tuple=False).flatten()
                stepped, clipped, proposed = _free_step(
                    branch_states.index_select(0, active_idx),
                    dt,
                    dynamics_config,
                    c3_config.reference_free_weight,
                    c3_config.reference_noise_weight,
                )
                branch_states[active_idx] = stepped
                total_clipped += clipped
                total_proposed += proposed
            terminal_np = branch_states.detach().cpu().numpy().reshape(local, branch_count, n, n).astype(np.float64)
            terminal_feat = _terminal_features_np(terminal_np.reshape(local * branch_count, n, n), dynamics_config, c3_config)
            endpoint_feat = target_feature_all[target_indices[local_start : local_start + local].reshape(-1)]
            terminal_feat = terminal_feat.reshape(local, branch_count, 1, -1)
            endpoint_feat = endpoint_feat.reshape(local, 1, endpoint_count, -1)
            dist2 = np.sum((terminal_feat - endpoint_feat) ** 2, axis=3)
            chunk_branch_dist2[local_start : local_start + local] = dist2
            chunk_branch_reward_base[local_start : local_start + local] = np.min(dist2, axis=2)
            chunk_terminal[local_start : local_start + local] = terminal_np
        branch_dist_chunks.append(chunk_branch_dist2)
        branch_reward_base_chunks.append(chunk_branch_reward_base)
        terminal_chunks.append(chunk_terminal)
        if hasattr(bar, "set_postfix"):
            preview = np.concatenate([x.reshape(-1) for x in branch_reward_base_chunks], axis=0)
            eps = c3_config.terminal_epsilon if c3_config.terminal_epsilon > 0 else _choose_epsilon_for_ess(preview, c3_config.terminal_ess_target)
            bar.set_postfix(eps=float(eps))

    states = torch.cat(states_out, dim=0).float()
    tau = torch.cat(tau_out, dim=0).float()
    labels = torch.cat(labels_out, dim=0).long()
    sources = torch.cat(sources_out, dim=0).float()
    branch_dist2 = np.concatenate(branch_dist_chunks, axis=0)
    branch_reward_base = np.concatenate(branch_reward_base_chunks, axis=0)
    terminals = np.concatenate(terminal_chunks, axis=0)
    epsilon = float(c3_config.terminal_epsilon)
    if epsilon <= 0.0:
        if str(c3_config.terminal_epsilon_mode).lower() == "branch-ess":
            epsilon = _choose_epsilon_for_ess(branch_reward_base.reshape(-1), float(c3_config.terminal_ess_target))
        elif str(c3_config.terminal_epsilon_mode).lower() == "global-ess":
            epsilon = _choose_epsilon_for_ess(branch_dist2.reshape(-1), float(c3_config.terminal_ess_target))
        elif str(c3_config.terminal_epsilon_mode).lower() == "fixed":
            epsilon = 1.0
        else:
            raise ValueError(f"unknown terminal_epsilon_mode: {c3_config.terminal_epsilon_mode}")
    # Mixture terminal reward across endpoints for each branch.
    branch_endpoint_logw = -branch_dist2 / (2.0 * max(epsilon, 1e-12) ** 2)
    max_ep = branch_endpoint_logw.max(axis=2, keepdims=True)
    branch_log_reward = (max_ep[:, :, 0] + np.log(np.exp(branch_endpoint_logw - max_ep).mean(axis=2) + 1e-30))
    max_branch = branch_log_reward.max(axis=1, keepdims=True)
    branch_w = np.exp(branch_log_reward - max_branch)
    branch_probs = branch_w / np.maximum(branch_w.sum(axis=1, keepdims=True), 1e-12)
    value_targets = max_branch[:, 0] + np.log(branch_w.mean(axis=1) + 1e-30)
    if c3_config.value_target_clip > 0:
        value_targets = np.maximum(value_targets, -float(c3_config.value_target_clip))
    best = np.argmax(branch_log_reward, axis=1)
    best_terminals = terminals[np.arange(entries), best]
    branch_ess = _branch_ess(branch_log_reward)
    branch_unweighted = branch_reward_base.mean(axis=1)
    branch_weighted = np.sum(branch_probs * branch_reward_base, axis=1)
    target_images = np.concatenate(endpoint_preview_chunks, axis=0)
    target_indices = np.concatenate(target_idx_chunks, axis=0)
    target_labels = np.concatenate(target_label_chunks, axis=0)
    requested_labels = np.concatenate(requested_label_chunks, axis=0)
    if np.any(target_labels != requested_labels):
        raise RuntimeError("target label mismatch in C3 diagnostics")
    log_weights = np.zeros(entries, dtype=np.float32)
    return C3ValueCache(
        states=states,
        tau=tau,
        labels=labels,
        sources=sources,
        value_targets=torch.as_tensor(value_targets, dtype=torch.float32),
        log_weights=torch.as_tensor(log_weights, dtype=torch.float32),
        target_images=target_images,
        terminal_states=best_terminals,
        requested_labels=requested_labels,
        target_labels=target_labels,
        target_indices=target_indices,
        epsilon=float(epsilon),
        value_target_mean=float(np.mean(value_targets)),
        value_target_std=float(np.std(value_targets)),
        branch_ess_fraction=branch_ess.astype(np.float64),
        branch_value_log_mean=value_targets.astype(np.float64),
        branch_unweighted_terminal_dist2=branch_unweighted.astype(np.float64),
        branch_weighted_terminal_dist2=branch_weighted.astype(np.float64),
        clip_fraction=0.0 if total_proposed == 0 else float(total_clipped) / float(total_proposed),
    )


def _cache_batch(cache: C3ValueCache, batch_size: int, device: torch.device, rng: np.random.Generator) -> dict[str, Tensor]:
    idx_np = rng.integers(0, cache.size, size=int(batch_size), dtype=np.int64)
    idx = torch.as_tensor(idx_np, dtype=torch.long)
    return {
        "states": cache.states.index_select(0, idx).to(device),
        "tau": cache.tau.index_select(0, idx).to(device),
        "labels": cache.labels.index_select(0, idx).to(device),
        "sources": cache.sources.index_select(0, idx).to(device),
        "value_targets": cache.value_targets.index_select(0, idx).to(device),
        "log_weights": cache.log_weights.index_select(0, idx).to(device),
    }


def _center_by_label(pred: Tensor, target: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    pred_out = pred.clone()
    target_out = target.clone()
    for digit in range(10):
        mask = labels == digit
        if bool(mask.any()):
            pred_out[mask] = pred_out[mask] - pred_out[mask].mean()
            target_out[mask] = target_out[mask] - target_out[mask].mean()
    return pred_out, target_out


def value_loss(model: ValuePotentialUNet, batch: dict[str, Tensor], c3: Experiment11C3Config) -> tuple[Tensor, dict[str, float]]:
    pred = model(batch["tau"], batch["states"], batch["labels"], batch["sources"])
    target = batch["value_targets"]
    pred_loss, target_loss = (pred, target)
    if bool(c3.value_center_by_label):
        pred_loss, target_loss = _center_by_label(pred_loss, target_loss, batch["labels"])
    loss_value = F.mse_loss(pred_loss, target_loss)
    loss_l2 = pred.square().mean()
    loss = float(c3.value_loss_weight) * loss_value + float(c3.value_l2_weight) * loss_l2
    with torch.no_grad():
        vx = pred_loss.detach().float()
        vy = target_loss.detach().float()
        cov = ((vx - vx.mean()) * (vy - vy.mean())).mean()
        corr = cov / (vx.std(unbiased=False).clamp_min(1e-12) * vy.std(unbiased=False).clamp_min(1e-12))
        return loss, {
            "loss": float(loss.detach().cpu()),
            "loss_value": float(loss_value.detach().cpu()),
            "value_l2": float(loss_l2.detach().cpu()),
            "value_pred_mean": float(pred.detach().float().mean().cpu()),
            "value_pred_std": float(pred.detach().float().std(unbiased=False).cpu()),
            "value_target_mean": float(target.detach().float().mean().cpu()),
            "value_target_std": float(target.detach().float().std(unbiased=False).cpu()),
            "value_corr": float(corr.detach().cpu()),
        }


def _init_ema(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}


def _update_ema(ema: dict[str, Tensor], model: nn.Module, decay: float) -> None:
    if decay <= 0.0:
        return
    with torch.no_grad():
        state = model.state_dict()
        for k, v in state.items():
            if k in ema and torch.is_floating_point(v):
                ema[k].mul_(float(decay)).add_(v.detach(), alpha=1.0 - float(decay))


def _load_ema(model: nn.Module, ema: dict[str, Tensor]) -> dict[str, Tensor]:
    old = {k: v.detach().clone() for k, v in model.state_dict().items()}
    state = model.state_dict()
    new_state = dict(state)
    for k, v in ema.items():
        if k in new_state:
            new_state[k] = v.to(device=new_state[k].device, dtype=new_state[k].dtype)
    model.load_state_dict(new_state, strict=False)
    return old


def _edge_gradient_from_cell_grad(grad: Tensor, grid_size: int) -> Tensor:
    n = int(grid_size)
    g = grad.reshape(-1, n, n)
    horiz = (torch.roll(g, shifts=-1, dims=2) - g) * float(n)
    vert = (torch.roll(g, shifts=-1, dims=1) - g) * float(n)
    return torch.stack([horiz, vert], dim=1)


def value_conditioning_flux(
    model: ValuePotentialUNet,
    tau: Tensor,
    states: Tensor,
    labels: Tensor,
    sources: Tensor,
    config: DirectFluxMNISTConfig,
    c3: Experiment11C3Config,
) -> tuple[Tensor, dict[str, float]]:
    n = int(config.grid_size)
    states_req = states.detach().clone().requires_grad_(True)
    value = model(tau, states_req, labels, sources)
    grad = torch.autograd.grad(value.sum(), states_req, create_graph=False, retain_graph=False)[0]
    edge_grad = _edge_gradient_from_cell_grad(grad, n)
    theta = harmonic_mobility_channels(states_req.detach(), config)
    flux = float(c3.value_flux_scale) * 2.0 * float(n) * theta * edge_grad
    if float(c3.value_flux_clip) > 0.0:
        flux = flux.clamp(-float(c3.value_flux_clip), float(c3.value_flux_clip))
    with torch.no_grad():
        return flux.detach(), {
            "value_mean": float(value.detach().float().mean().cpu()),
            "value_std": float(value.detach().float().std(unbiased=False).cpu()),
            "value_gradient_rms": float(grad.detach().float().square().mean().sqrt().cpu()),
            "value_flux_rms": float(flux.detach().float().square().mean().sqrt().cpu()),
        }


def train_value_model(
    model: ValuePotentialUNet,
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    ot_cache: ClasswiseOTCache,
    dynamics_config: DirectFluxMNISTConfig,
    c3_config: Experiment11C3Config,
    device: torch.device,
    show_progress: bool,
    cache_dir: Path | None = None,
) -> tuple[list[dict[str, float]], dict[str, Tensor], C3ValueCache]:
    _disable_mkldnn_for_cpu_if_needed(device)
    rng = np.random.default_rng(int(c3_config.seed))
    torch.manual_seed(int(c3_config.seed))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c3_config.learning_rate), weight_decay=float(c3_config.weight_decay))
    scaler = _make_cuda_grad_scaler(enabled=bool(c3_config.use_amp and device.type == "cuda"))
    ema = _init_ema(model)
    history: list[dict[str, float]] = []
    cache: C3ValueCache | None = None
    bar = _progress(range(int(c3_config.train_steps)), total=int(c3_config.train_steps), desc="train C3 value", disable=not show_progress)
    for step in bar:
        if cache is None or (int(c3_config.cache_refresh_every) > 0 and step % int(c3_config.cache_refresh_every) == 0):
            cache = build_c3_value_cache(
                dataset_images=dataset_images,
                dataset_labels=dataset_labels,
                ot_cache=ot_cache,
                dynamics_config=dynamics_config,
                c3_config=c3_config,
                device=device,
                rng=rng,
                show_progress=show_progress,
            )
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_dir / f"experiment11_c3_value_cache_step{step:06d}.npz",
                    value_targets=cache.value_targets.numpy(),
                    requested_labels=cache.requested_labels,
                    target_labels=cache.target_labels,
                    branch_ess_fraction=cache.branch_ess_fraction,
                    branch_unweighted_terminal_dist2=cache.branch_unweighted_terminal_dist2,
                    branch_weighted_terminal_dist2=cache.branch_weighted_terminal_dist2,
                    terminal_states=cache.terminal_states,
                    target_images=cache.target_images,
                    epsilon=np.asarray(cache.epsilon),
                )
                try:
                    order = np.argsort(cache.branch_weighted_terminal_dist2)[:64]
                    save_flux_samples_grid(cache.terminal_states[order].reshape(-1, n * n), cache.requested_labels[order], cache_dir / f"experiment11_c3_value_weighted_terminals_step{step:06d}.png", grid_size=n)
                    save_flux_samples_grid(cache.target_images[order].reshape(-1, n * n), cache.target_labels[order], cache_dir / f"experiment11_c3_value_targets_step{step:06d}.png", grid_size=n)
                except Exception:
                    pass
        assert cache is not None
        batch = _cache_batch(cache, int(c3_config.batch_size), device, rng)
        opt.zero_grad(set_to_none=True)
        amp_context = _cuda_autocast(enabled=True) if bool(c3_config.use_amp and device.type == "cuda") else nullcontext()
        with amp_context:
            loss, diag = value_loss(model, batch, c3_config)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if float(c3_config.grad_clip) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), float(c3_config.grad_clip))
        scaler.step(opt)
        scaler.update()
        _update_ema(ema, model, float(c3_config.ema_decay))
        diag.update(
            {
                "step": float(step),
                "cache_epsilon": float(cache.epsilon),
                "cache_value_target_std": float(cache.value_target_std),
                "branch_ess_fraction_mean": float(np.mean(cache.branch_ess_fraction)),
                "branch_weighted_minus_unweighted_dist2": float(np.mean(cache.branch_weighted_terminal_dist2 - cache.branch_unweighted_terminal_dist2)),
                "cache_clip_fraction": float(cache.clip_fraction),
            }
        )
        history.append(diag)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=diag["loss"], corr=diag["value_corr"], std=diag["value_pred_std"])
    assert cache is not None
    return history, ema, cache


@torch.enable_grad()
def simulate_value_generation(
    model: ValuePotentialUNet,
    labels: np.ndarray,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    c3_config: Experiment11C3Config,
    device: torch.device,
    source_images: np.ndarray,
    source_labels: np.ndarray,
    seed: int,
    deterministic: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    n = int(dynamics_config.grid_size)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    source_batch = _sample_source_batch_torch(
        len(labels),
        dynamics_config,
        device=device,
        dtype=torch.float32,
        label_tensor=labels_t,
        source_images=source_images,
        source_labels=source_labels,
        rng=rng,
    )
    states = source_batch.masses.to(device=device)
    sources = states.clone()
    horizon = natural_horizon(dynamics_config)
    dt = horizon / float(c3_config.sample_steps)
    clipped = 0
    proposed = 0
    diag_sums = {"value_gradient_rms": 0.0, "value_flux_rms": 0.0, "value_mean": 0.0, "value_std": 0.0}
    bar = _progress(range(int(c3_config.sample_steps)), total=int(c3_config.sample_steps), desc="sample C3 value", disable=not show_progress)
    model.to(device).eval()
    for step in bar:
        tau = torch.full((len(labels),), max(horizon - float(step) * dt, 0.0), dtype=states.dtype, device=device)
        flux, d = value_conditioning_flux(model, tau, states, labels_t, sources, dynamics_config, c3_config)
        for k in diag_sums:
            diag_sums[k] += float(d[k])
        states, c, p = eulerian_flux_step_torch(
            states,
            flux,
            dt,
            dynamics_config,
            deterministic=bool(deterministic),
            free_weight=float(c3_config.reference_free_weight),
            noise_weight=float(c3_config.reference_noise_weight),
            learned_weight=1.0,
        )
        clipped += int(c)
        proposed += int(p)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(clip=0.0 if proposed == 0 else clipped / proposed, flux=d["value_flux_rms"])
    samples = states.detach().cpu().numpy().astype(np.float64)
    denom = max(int(c3_config.sample_steps), 1)
    diagnostics = {k: v / denom for k, v in diag_sums.items()}
    diagnostics.update(
        {
            "clipping_fraction": 0.0 if proposed == 0 else float(clipped) / float(proposed),
            "sample_entropy": float(_mass_entropy_torch(states).mean().detach().cpu()),
            "sample_total_variation": float(image_total_variation(states, grid_size=n).mean().detach().cpu()),
            "sample_checkerboard_energy": float(sample_checkerboard_energy_torch(states, grid_size=n).detach().cpu()),
        }
    )
    return {"samples": samples, "labels": np.asarray(labels, dtype=np.int64), "sources": sources.detach().cpu().numpy(), "diagnostics": diagnostics}


def _write_history(history: list[dict[str, float]], path: Path) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if history:
        keys = sorted({k for row in history for k in row})
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    p.add_argument("--runs-root", type=Path, default=Path("runs/experiment11"))
    p.add_argument("--run-name", type=str, default="c3-value")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--examples-per-class", type=int, default=1000)
    p.add_argument("--download", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--no-amp", action="store_true")

    p.add_argument("--grid-size", type=int, default=28)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--edge-alpha-mode", type=str, default="legacy")
    p.add_argument("--horizon-scale", type=float, default=1.0)
    p.add_argument("--limiter-fraction", type=float, default=0.25)
    p.add_argument("--mass-floor", type=float, default=1e-8)
    p.add_argument("--adaptive-sampling", action="store_true")
    p.add_argument("--clip-target", type=float, default=0.03)
    p.add_argument("--max-substeps", type=int, default=4)
    p.add_argument("--source-mode", type=str, default="lowfreq")
    p.add_argument("--source-lowfreq-size", type=int, default=7)
    p.add_argument("--source-blur-sigma", type=float, default=1.0)
    p.add_argument("--source-uniform-mix", type=float, default=0.15)
    p.add_argument("--source-concentration", type=float, default=1.0)
    p.add_argument("--no-condition-on-source", action="store_true")
    p.add_argument("--upsample-mode", type=str, default="resize-conv")
    p.add_argument("--ot-cost-mode", type=str, default="lowres")
    p.add_argument("--ot-match-mode", type=str, default="topk")
    p.add_argument("--ot-nearest-top-k", type=int, default=32)
    p.add_argument("--ot-lowres-size", type=int, default=7)
    p.add_argument("--ot-blur-sigma", type=float, default=1.0)
    p.add_argument("--ot-com-weight", type=float, default=0.25)

    for field in fields(Experiment11C3Config):
        name = "--" + field.name.replace("_", "-")
        default = getattr(Experiment11C3Config, field.name)
        if isinstance(default, bool):
            # Accept both the positive and negative spellings for every boolean
            # Experiment11C3Config field.  The initial C3 patch only exposed the
            # negative spelling for fields whose default is True, which made the
            # documented command fail on flags such as --value-center-by-label.
            p.add_argument(name, dest=field.name, action="store_true")
            p.add_argument("--no-" + field.name.replace("_", "-"), dest=field.name, action="store_false")
            p.set_defaults(**{field.name: default})
        else:
            arg_type = type(default) if default is not None else str
            p.add_argument(name, type=arg_type, default=default)
    p.add_argument("--labels", type=str, default="cycle")
    p.add_argument("--deterministic-sampling", action="store_true")
    p.add_argument("--use-classifier-diagnostics", action="store_true")
    return p.parse_args(argv)


def _c3_config_from_args(args: argparse.Namespace) -> Experiment11C3Config:
    data = {f.name: getattr(args, f.name) for f in fields(Experiment11C3Config)}
    data["use_amp"] = not bool(args.no_amp)
    return Experiment11C3Config(**data)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    show_progress = not bool(args.no_progress)
    dynamics_config = _make_dynamics_config(args)
    c3_config = _c3_config_from_args(args)
    run_dir, metadata = make_experiment11_c3_run_dir(Path(args.runs_root), args.run_name)
    print(f"Experiment 11/C3 value potential on device={device}")
    print(f"Run directory: {run_dir}")
    dataset = load_mnist_measure_dataset(
        args.data_root,
        max_train=args.max_train,
        examples_per_class=args.examples_per_class,
        download=bool(args.download),
        seed=int(c3_config.seed),
    )
    ot_cache = build_classwise_ot_cache(dataset.train_images, dataset.train_labels, dynamics_config)
    model = ValuePotentialUNet(dynamics_config, base_channels=int(c3_config.base_channels))
    metadata.update({"dynamics_config": asdict(dynamics_config), "c3_config": asdict(c3_config), "args": vars(args)})
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=_serializable), encoding="utf-8")

    history, ema, cache = train_value_model(
        model,
        dataset_images=dataset.train_images,
        dataset_labels=dataset.train_labels,
        ot_cache=ot_cache,
        dynamics_config=dynamics_config,
        c3_config=c3_config,
        device=device,
        show_progress=show_progress,
        cache_dir=run_dir if bool(c3_config.save_cache_previews) else None,
    )
    _write_history(history, run_dir / "experiment11_c3_value_history.json")
    ckpt = {
        "raw_model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema,
        "model_state_dict": model.state_dict(),
        "dynamics_config": asdict(dynamics_config),
        "c3_config": asdict(c3_config),
        "history": history,
        "last_cache_summary": {
            "epsilon": cache.epsilon,
            "value_target_mean": cache.value_target_mean,
            "value_target_std": cache.value_target_std,
            "branch_ess_fraction_mean": float(np.mean(cache.branch_ess_fraction)),
            "branch_weighted_minus_unweighted_dist2": float(np.mean(cache.branch_weighted_terminal_dist2 - cache.branch_unweighted_terminal_dist2)),
        },
    }
    torch.save(ckpt, run_dir / "experiment11_c3_value_model.pt")

    labels = _parse_label_sequence(str(args.labels), int(c3_config.num_samples))
    weight_modes = [str(c3_config.sampling_weights).lower()]
    if weight_modes == ["both"]:
        weight_modes = ["raw", "ema"]
    scales = [float(x) for x in str(c3_config.sample_value_flux_scales).split(",") if x.strip()]
    summary: dict[str, object] = {}
    for weight_mode in weight_modes:
        old_state = None
        if weight_mode == "ema":
            old_state = _load_ema(model, ema)
        elif weight_mode != "raw":
            raise ValueError(f"unknown sampling weight mode: {weight_mode}")
        for scale in scales:
            run_cfg = Experiment11C3Config(**{**asdict(c3_config), "value_flux_scale": scale})
            result = simulate_value_generation(
                model,
                labels,
                dynamics_config=dynamics_config,
                c3_config=run_cfg,
                device=device,
                source_images=dataset.train_images,
                source_labels=dataset.train_labels,
                seed=int(c3_config.sample_seed),
                deterministic=bool(args.deterministic_sampling),
                show_progress=show_progress,
            )
            tag = f"{weight_mode}_scale{scale:g}".replace(".", "p")
            np.savez_compressed(
                run_dir / f"experiment11_c3_value_samples_{tag}.npz",
                samples=result["samples"],
                labels=result["labels"],
                sources=result["sources"],
            )
            save_flux_samples_grid(result["samples"], result["labels"], run_dir / f"experiment11_c3_value_samples_{tag}.png", grid_size=int(dynamics_config.grid_size))
            metrics = dict(result["diagnostics"])
            if bool(args.use_classifier_diagnostics):
                classifier = train_or_load_mnist_classifier(
                    dataset.train_images,
                    dataset.train_labels,
                    grid_size=int(dynamics_config.grid_size),
                    cache_path=run_dir / "experiment11_c3_classifier.pt",
                    device=device,
                    seed=int(c3_config.seed) + 123,
                    show_progress=show_progress,
                )
                cls = classifier_generation_metrics(result["samples"], result["labels"], classifier, grid_size=int(dynamics_config.grid_size), device=device)
                metrics.update({k: v for k, v in cls.items() if not isinstance(v, np.ndarray)})
            (run_dir / f"experiment11_c3_value_metrics_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            summary[tag] = metrics
        if old_state is not None:
            model.load_state_dict(old_state, strict=True)
    (run_dir / "experiment11_c3_value_sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Experiment 11/C3 complete")


if __name__ == "__main__":
    main()
