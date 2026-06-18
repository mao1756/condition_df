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
from dataclasses import asdict, dataclass, replace
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
    poisson_flux_from_velocity_torch,
    project_edge_flux_torch,
    flux_curl_torch,
    flux_divergence_torch,
    edge_laplacian_energy_torch,
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
    # D0 rollouts are model-independent; rebuilding them during training is
    # expensive and disabled by default.  Set positive to resample periodically.
    cache_refresh_every: int = 0
    cache_build_mode: str = "outer"
    cache_time_sampling: str = "endpoint-mixture"
    cache_data_endpoint_prob: float = 0.50
    cache_terminal_endpoint_prob: float = 0.10
    cache_endpoint_window_fraction: float = 0.10
    time_slices_per_path: int = 4
    teacher_stride_substeps: int = 8
    eta_l2_weight: float = 1e-4
    d0_target_space: str = "projected-edge"
    # D0.5: realized-* targets are built from the actual limited edge
    # transfers applied by the forward integrator, not reconstructed Poisson
    # transfers from node displacements.
    # D0.6: physical targets are tiny in mass-transfer units; train normalized
    # physical outputs and rescale only inside the sampler/update.
    physical_target_normalization: str = "global-rms"
    physical_target_scale: float = 0.0
    physical_target_scale_floor: float = 1e-6
    physical_loss_mask: str = "all"
    debug_memorize_cache: int = 0
    # D0.7: optional true trajectory-window supervision.  The exact substep
    # cache stores the forward states reached one or more reverse blocks before
    # each cached later state, so repeated learned updates can be supervised
    # against the correct trajectory targets rather than one fixed endpoint.
    cache_store_trajectory_windows: bool = False
    rollout_target_blocks: str = "1,2,4,8,16"
    trajectory_rollout_loss_weight: float = 0.0
    trajectory_rollout_warmup_steps: int = 500
    trajectory_rollout_depths: str = "1,2,4"
    trajectory_rollout_batch_size: int = 32
    sample_cache_subset: str = "auto"
    sample_cache_rollout_depths: str = ""
    edge_innovation_loss_weight: float = 1.0
    state_delta_loss_weight: float = 0.0
    # D0.4: keep the previous rollout regularizer disabled by default.  It is
    # a stability heuristic, not a supervised multi-block target, until the
    # cache stores trajectory windows.
    rollout_loss_weight: float = 0.0
    rollout_loss_warmup_steps: int = 1000
    rollout_loss_blocks: int = 4
    rollout_loss_batch_size: int = 32
    invalid_output_l2_weight: float = 1e-3
    curl_loss_weight: float = 1e-3
    edge_laplacian_loss_weight: float = 0.0
    sample_project_learned_mean: bool = True
    physical_sampler_noise_mode: str = "none"
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
    deterministic_control_strengths: str = "-0.25,-0.1,0,0.1,0.25,0.5,1"
    stochastic_control_strengths: str = "0,0.25,0.5,1"
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
    # Safety/diagnostic flags for the overfit-one-image acceptance test.
    allow_prior_bank_mismatch: bool = False
    allow_stride_rounding: bool = False
    use_external_prior_bank_for_overfit: bool = False
    oracle_reverse_diagnostic: bool = True
    oracle_reverse_slices: int = 16
    learned_block_diagnostic: bool = True
    learned_block_slices: int = 32
    learned_rollout_diagnostic: bool = True
    learned_rollout_slices: int = 16
    learned_rollout_blocks: str = "1,4,16,64"
    sample_start_source: str = "terminal-bank"
    sample_start_tau_fractions: str = ""
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
    earlier_states: Tensor
    physical_transfers: Tensor
    physical_target_scale: float
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
    cache_build_mode: str = "outer"
    requested_stride_substeps: int = 0
    trajectory_window_states: Tensor | None = None
    trajectory_window_valid: Tensor | None = None
    trajectory_window_depths: Tensor | None = None

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
    physical_sampler_mode: str = "brownian-innovation"
    physical_sampler_noise_mode: str = "reference"
    learned_raw_rms: float = float("nan")
    learned_projected_rms: float = float("nan")
    learned_removed_curl_rms: float = float("nan")
    sample_start_corr_mean: float = float("nan")
    sample_start_corr_max: float = float("nan")
    sample_start_l1_mean: float = float("nan")
    physical_target_scale: float = 1.0
    stride_substeps: int = 1
    sample_steps: int = 0
    reference_substeps: int = 0
    prior_bank_path: str = ""
    sample_start_source: str = "terminal-bank"
    sample_start_substep: int = -1
    reverse_blocks_requested: int = -1
    reverse_blocks_executed: int = 0


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


def _parse_csv_ints(value: str | int | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    pieces = [piece.strip() for piece in str(value).split(",") if piece.strip()]
    return [int(piece) for piece in pieces]


def _resolve_cache_subset_indices(
    cache: "D0TrainingCache",
    d0_config: Experiment12D0Config,
    subset: str | None = None,
) -> np.ndarray:
    """Resolve all/memorized/heldout cache slices for diagnostics and sampling."""

    mode = str(d0_config.sample_cache_subset if subset is None else subset).strip().lower().replace("_", "-")
    memorized_count = min(max(int(d0_config.debug_memorize_cache), 0), int(cache.size))
    if mode == "auto":
        mode = "memorized" if memorized_count > 0 else "all"
    if mode == "all":
        return np.arange(int(cache.size), dtype=np.int64)
    if mode == "memorized":
        if memorized_count <= 0:
            raise ValueError("sample_cache_subset=memorized requires --debug-memorize-cache > 0")
        return np.arange(memorized_count, dtype=np.int64)
    if mode == "heldout":
        if memorized_count <= 0:
            raise ValueError("sample_cache_subset=heldout requires --debug-memorize-cache > 0")
        return np.arange(memorized_count, int(cache.size), dtype=np.int64)
    raise ValueError("sample_cache_subset must be auto, all, memorized, or heldout")


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


def _actual_cache_stride_substeps(d0_config: Experiment12D0Config) -> int:
    requested = max(1, int(d0_config.teacher_stride_substeps))
    reference_substeps = max(1, int(d0_config.reference_substeps))
    mode = str(d0_config.cache_build_mode).strip().lower().replace("_", "-")
    if mode in {"outer", "fast", "outer-fast"}:
        return int(math.ceil(float(requested) / float(reference_substeps)) * reference_substeps)
    return int(requested)


def _expected_rate_schedule(dynamics_config: DirectFluxMNISTConfig, d0_config: Experiment12D0Config) -> np.ndarray:
    return make_rate_schedule(
        int(d0_config.sample_steps),
        tau_eff=float(d0_config.tau_eff),
        horizon=natural_horizon(dynamics_config),
        time_change_mode=str(d0_config.time_change_mode),
        ramp=str(d0_config.rate_ramp),
        ramp_ratio=float(d0_config.rate_ramp_ratio),
        rate_min=d0_config.reference_rate_min,
        rate_max=d0_config.reference_rate_max,
    )




def _sample_cache_start_substeps(
    rng: np.random.Generator,
    *,
    shape: tuple[int, int],
    total_substeps: int,
    stride: int,
    d0_config: Experiment12D0Config,
) -> np.ndarray:
    """Sample cache slice start indices with optional endpoint bias.

    ``data`` endpoint means small forward time / high reverse tau, where the
    overfit signal is strongest. ``terminal`` endpoint means large forward time
    / low reverse tau, useful for sampler initialization diagnostics.
    """

    max_start = int(total_substeps) - int(stride)
    if max_start < 0:
        raise ValueError("stride cannot exceed total_substeps")
    mode = str(d0_config.cache_time_sampling).strip().lower().replace("_", "-")
    if mode == "uniform" or max_start == 0:
        return rng.integers(0, max_start + 1, size=shape, dtype=np.int64)
    if mode not in {"endpoint-mixture", "endpoint", "endpoint-biased", "data-biased"}:
        raise ValueError("cache_time_sampling must be uniform or endpoint-mixture")
    flat_count = int(np.prod(shape))
    data_prob = max(0.0, min(1.0, float(d0_config.cache_data_endpoint_prob)))
    terminal_prob = max(0.0, min(1.0 - data_prob, float(d0_config.cache_terminal_endpoint_prob)))
    window = max(1, int(round(float(total_substeps) * float(d0_config.cache_endpoint_window_fraction))))
    window = min(window, max_start + 1)
    u = rng.random(flat_count)
    starts = rng.integers(0, max_start + 1, size=flat_count, dtype=np.int64)
    data_mask = u < data_prob
    terminal_mask = (u >= data_prob) & (u < data_prob + terminal_prob)
    if data_mask.any():
        starts[data_mask] = rng.integers(0, window, size=int(data_mask.sum()), dtype=np.int64)
    if terminal_mask.any():
        lo = max(0, max_start - window + 1)
        starts[terminal_mask] = rng.integers(lo, max_start + 1, size=int(terminal_mask.sum()), dtype=np.int64)
    return starts.reshape(shape)


def _project_d0_edge_field(field: Tensor, dynamics_config: DirectFluxMNISTConfig, d0_config: Experiment12D0Config) -> Tensor:
    space = _normalize_d0_target_space(d0_config.d0_target_space)
    if space in {"projected-edge", "physical-total", "physical-residual", "node-delta"}:
        return project_edge_flux_torch(field, grid_size=int(dynamics_config.grid_size))
    if space in {"realized-physical", "realized-residual"}:
        return field
    return field




def _normalize_d0_target_space(value: str) -> str:
    space = str(value).strip().lower().replace("_", "-")
    if space in {"physical", "physical-edge", "state-edge", "physical-total", "total-physical", "poisson-physical"}:
        return "physical-total"
    if space in {"physical-residual", "residual-physical", "physical-free-residual", "poisson-residual"}:
        return "physical-residual"
    if space in {"realized", "realized-physical", "pathwise-physical", "realized-total"}:
        return "realized-physical"
    if space in {"realized-residual", "pathwise-residual", "realized-physical-residual"}:
        return "realized-residual"
    if space in {"node", "node-delta", "state-delta"}:
        return "node-delta"
    if space == "projected-edge":
        return "projected-edge"
    if space in {"raw", "edge"}:
        return "raw"
    if space in {"divergence", "node-divergence"}:
        return "divergence"
    raise ValueError("d0_target_space must be raw, projected-edge, divergence, physical-total, physical-residual, realized-physical, realized-residual, or node-delta")


def _d0_is_physical_space(value: str) -> bool:
    return _normalize_d0_target_space(value) in {"physical-total", "physical-residual", "realized-physical", "realized-residual", "node-delta"}


def _physical_sampler_noise_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("_", "-")
    if mode in {"none", "off", "deterministic"}:
        return "none"
    if mode in {"reference", "ref", "brownian"}:
        return "reference"
    raise ValueError("physical_sampler_noise_mode must be none or reference")

def _physical_target_normalization_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("_", "-")
    if mode in {"none", "off", "unscaled"}:
        return "none"
    if mode in {"global", "global-rms", "cache-rms"}:
        return "global-rms"
    if mode in {"per-slice", "per-slice-rms", "slice-rms"}:
        return "per-slice-rms"
    raise ValueError("physical_target_normalization must be none, global-rms, or per-slice-rms")


def _physical_loss_mask_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("_", "-")
    if mode in {"all", "finite"}:
        return mode
    if mode in {"innovation", "innovation-valid", "valid-innovation"}:
        return "innovation-valid"
    raise ValueError("physical_loss_mask must be all, finite, or innovation-valid")


def _physical_target_scale_from_tensor(target: Tensor, d0_config: Experiment12D0Config) -> Tensor:
    floor = float(d0_config.physical_target_scale_floor)
    if float(d0_config.physical_target_scale) > 0.0:
        return torch.as_tensor(float(d0_config.physical_target_scale), dtype=target.dtype, device=target.device)
    return torch.sqrt(target.detach().float().square().mean()).to(device=target.device, dtype=target.dtype).clamp_min(floor)


def _physical_edge_loss_mask(mask_f: Tensor, target: Tensor, d0_config: Experiment12D0Config) -> Tensor:
    mode = _physical_loss_mask_mode(d0_config.physical_loss_mask)
    if mode == "innovation-valid":
        return mask_f
    finite = torch.isfinite(target).to(dtype=mask_f.dtype)
    if mode == "finite":
        return finite
    return torch.ones_like(mask_f) * finite


def _cache_physical_target_scale(cache: "D0TrainingCache", d0_config: Experiment12D0Config) -> float:
    mode = _physical_target_normalization_mode(d0_config.physical_target_normalization)
    if mode == "none":
        return 1.0
    if float(d0_config.physical_target_scale) > 0.0:
        return float(d0_config.physical_target_scale)
    value = float(torch.sqrt(cache.physical_transfers.float().square().mean()).item()) if cache.size else 0.0
    return max(value, float(d0_config.physical_target_scale_floor))


def _with_cache_physical_scale(d0_config: Experiment12D0Config, cache: "D0TrainingCache") -> Experiment12D0Config:
    scale = _cache_physical_target_scale(cache, d0_config)
    return replace(d0_config, physical_target_scale=float(scale))


def _physical_training_scale_tensor(target: Tensor, d0_config: Experiment12D0Config) -> Tensor:
    mode = _physical_target_normalization_mode(d0_config.physical_target_normalization)
    if mode == "none":
        return torch.ones((), dtype=target.dtype, device=target.device)
    if mode == "global-rms":
        return _physical_target_scale_from_tensor(target, d0_config)
    # Per-slice normalization is diagnostic/training-only.  The sampler cannot
    # know a slice-specific training target scale, so use it only for probing.
    scale = torch.sqrt(target.detach().float().square().mean(dim=(1, 2, 3), keepdim=True)).to(device=target.device, dtype=target.dtype)
    return scale.clamp_min(float(d0_config.physical_target_scale_floor))

def _d0_masked_edge_mse(
    pred: Tensor,
    target: Tensor,
    mask_f: Tensor,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return main loss, zero loss, residual RMS, pred RMS, target RMS."""

    space = _normalize_d0_target_space(d0_config.d0_target_space)
    if space == "raw":
        pred_loss = pred
        target_loss = target
        denom = mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_slice = ((pred_loss - target_loss).square() * mask_f).sum(dim=(1, 2, 3)) / denom
        zero = (target_loss.square() * mask_f).sum(dim=(1, 2, 3)) / denom
        pred_rms = torch.sqrt((pred_loss.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        target_rms = torch.sqrt((target_loss.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        residual_rms = torch.sqrt(((pred_loss - target_loss).square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        return per_slice.mean(), zero.mean(), residual_rms, pred_rms, target_rms
    if space in {"projected", "projected-edge", "edge-projected"}:
        pred_loss = project_edge_flux_torch(pred, grid_size=int(dynamics_config.grid_size))
        target_loss = project_edge_flux_torch(target, grid_size=int(dynamics_config.grid_size))
        denom = mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_slice = ((pred_loss - target_loss).square() * mask_f).sum(dim=(1, 2, 3)) / denom
        zero = (target_loss.square() * mask_f).sum(dim=(1, 2, 3)) / denom
        pred_rms = torch.sqrt((pred_loss.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        target_rms = torch.sqrt((target_loss.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        residual_rms = torch.sqrt(((pred_loss - target_loss).square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        return per_slice.mean(), zero.mean(), residual_rms, pred_rms, target_rms
    if space == "divergence":
        pred_node = flux_divergence_torch(pred).reshape(pred.shape[0], -1)
        target_node = flux_divergence_torch(target).reshape(target.shape[0], -1)
        per_slice = (pred_node - target_node).square().mean(dim=1)
        zero = target_node.square().mean(dim=1)
        pred_rms = torch.sqrt(pred_node.square().mean())
        target_rms = torch.sqrt(target_node.square().mean())
        residual_rms = torch.sqrt((pred_node - target_node).square().mean())
        return per_slice.mean(), zero.mean(), residual_rms, pred_rms, target_rms
    raise ValueError("d0_target_space must be raw, projected-edge, or divergence in _d0_masked_edge_mse")


def _sample_start_reference_stats(samples: np.ndarray, reference: np.ndarray | None) -> dict[str, float]:
    if reference is None:
        return {"sample_start_corr_mean": float("nan"), "sample_start_corr_max": float("nan"), "sample_start_l1_mean": float("nan")}
    s = np.asarray(samples, dtype=np.float64).reshape(samples.shape[0], -1)
    r = np.asarray(reference, dtype=np.float64).reshape(1, -1)
    s0 = s - s.mean(axis=1, keepdims=True)
    r0 = r - r.mean(axis=1, keepdims=True)
    corr = (s0 * r0).sum(axis=1) / np.maximum(np.sqrt((s0 * s0).sum(axis=1) * float((r0 * r0).sum())), 1e-30)
    l1 = np.abs(s - r).sum(axis=1)
    return {
        "sample_start_corr_mean": float(np.mean(corr)),
        "sample_start_corr_max": float(np.max(corr)),
        "sample_start_l1_mean": float(np.mean(l1)),
    }


def _validate_prior_bank_compatibility(
    bank: dict[str, np.ndarray | float | int | str],
    *,
    d0_config: Experiment12D0Config,
    dynamics_config: DirectFluxMNISTConfig,
    prior_bank_path: str | Path,
) -> None:
    """Reject prior banks whose forward noising law differs from training.

    The overfit failure mode that motivated D0.1 was sampling from a generic
    Phase-0 bank with a different substep grid.  This guard keeps the reverse
    sampler from silently mixing time grids, lambda conventions, or schedules.
    """

    if bool(d0_config.allow_prior_bank_mismatch):
        return
    mismatches: list[str] = []
    expected_steps = int(d0_config.sample_steps)
    expected_substeps = int(d0_config.reference_substeps)
    expected_horizon = float(natural_horizon(dynamics_config))
    expected_stride = int(_actual_cache_stride_substeps(d0_config))
    expected_rate = _expected_rate_schedule(dynamics_config, d0_config)

    sample_steps = int(bank.get("sample_steps", expected_steps))
    substeps = int(bank.get("substeps", expected_substeps))
    horizon = float(bank.get("horizon", expected_horizon))
    if sample_steps != expected_steps:
        mismatches.append(f"sample_steps bank={sample_steps} expected={expected_steps}")
    if substeps != expected_substeps:
        mismatches.append(f"substeps bank={substeps} expected={expected_substeps}")
    if not math.isclose(horizon, expected_horizon, rel_tol=1e-6, abs_tol=1e-12):
        mismatches.append(f"horizon bank={horizon:.12g} expected={expected_horizon:.12g}")
    if "lambda_mix" in bank:
        bank_lambda = float(bank["lambda_mix"])
        if not math.isclose(bank_lambda, float(d0_config.lambda_mix), rel_tol=1e-6, abs_tol=1e-12):
            mismatches.append(f"lambda_mix bank={bank_lambda:.12g} expected={float(d0_config.lambda_mix):.12g}")
    if "stride_substeps" in bank:
        bank_stride = int(bank["stride_substeps"])
        if bank_stride != expected_stride:
            mismatches.append(f"stride_substeps bank={bank_stride} expected_actual_cache_stride={expected_stride}")
    if "edge_alpha_mode" in bank:
        mode = str(bank["edge_alpha_mode"])
        if mode != str(dynamics_config.edge_alpha_mode):
            mismatches.append(f"edge_alpha_mode bank={mode} expected={dynamics_config.edge_alpha_mode}")
    if "alpha_eff" in bank:
        bank_alpha_eff = float(bank["alpha_eff"])
        if not math.isclose(bank_alpha_eff, float(dynamics_config.alpha_eff), rel_tol=1e-6, abs_tol=1e-12):
            mismatches.append(f"alpha_eff bank={bank_alpha_eff:.12g} expected={float(dynamics_config.alpha_eff):.12g}")
    rate = np.asarray(bank.get("rate_schedule", expected_rate), dtype=np.float64).reshape(-1)
    if rate.shape != expected_rate.shape:
        mismatches.append(f"rate_schedule length bank={rate.shape[0]} expected={expected_rate.shape[0]}")
    elif not np.allclose(rate, expected_rate, rtol=1e-5, atol=1e-12):
        max_abs = float(np.max(np.abs(rate - expected_rate)))
        mismatches.append(f"rate_schedule differs; max_abs_diff={max_abs:.4g}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            "Incompatible D0 prior bank for this trained sampler: "
            f"{prior_bank_path}. Mismatches: {joined}. "
            "For a one-image overfit run, omit --prior-bank-path to use the current run's "
            "experiment12_d0_training_terminal_bank.npz. To bypass only for debugging, pass "
            "--allow-prior-bank-mismatch."
        )


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


def _build_d0_training_cache_substep_exact(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    device: torch.device,
    rng: np.random.Generator,
    show_progress: bool = True,
) -> D0TrainingCache:
    """Build the original exact substep-keyed D0 cache.

    This path is semantically precise but expensive because it stores and scans
    all substep states.  The public wrapper defaults to the faster outer-step
    cache below.
    """

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
    trajectory_depths = sorted({max(1, int(x)) for x in _parse_csv_ints(d0_config.rollout_target_blocks)}) if bool(d0_config.cache_store_trajectory_windows) else []

    all_states: list[Tensor] = []
    all_tau: list[Tensor] = []
    all_labels: list[Tensor] = []
    all_innov: list[Tensor] = []
    all_masks: list[Tensor] = []
    all_starts: list[Tensor] = []
    all_path_indices: list[Tensor] = []
    all_start_images: list[Tensor] = []
    all_earlier_states: list[Tensor] = []
    all_physical_transfers: list[Tensor] = []
    all_trajectory_window_states: list[Tensor] = []
    all_trajectory_window_valid: list[Tensor] = []
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
        starts_np = _sample_cache_start_substeps(
            rng,
            shape=(bsz, slices_per_path),
            total_substeps=total_substeps,
            stride=stride,
            d0_config=d0_config,
        )
        starts_t = torch.as_tensor(starts_np, dtype=torch.long, device=device)
        ends_t = starts_t + int(stride) - 1
        accum = torch.zeros((bsz, slices_per_path, 2, n, n), dtype=torch.float32, device=device)
        physical_accum = torch.zeros((bsz, slices_per_path, 2, n, n), dtype=torch.float32, device=device)
        masks = torch.ones((bsz, slices_per_path, 2, n, n), dtype=torch.bool, device=device)
        later_states = torch.empty((bsz, slices_per_path, n * n), dtype=torch.float32, device=device)
        earlier_states = torch.empty((bsz, slices_per_path, n * n), dtype=torch.float32, device=device)
        later_tau = torch.empty((bsz, slices_per_path), dtype=torch.float32, device=device)
        filled = torch.zeros((bsz, slices_per_path), dtype=torch.bool, device=device)
        start_filled = torch.zeros((bsz, slices_per_path), dtype=torch.bool, device=device)
        if trajectory_depths:
            depth_t = torch.as_tensor(trajectory_depths, dtype=torch.long, device=device)
            trajectory_target_steps = starts_t.unsqueeze(-1) - (depth_t.view(1, 1, -1) - 1) * int(stride)
            trajectory_valid = trajectory_target_steps >= 0
            trajectory_states = torch.zeros((bsz, slices_per_path, len(trajectory_depths), n * n), dtype=torch.float32, device=device)
            trajectory_filled = ~trajectory_valid.clone()
        else:
            trajectory_target_steps = None
            trajectory_valid = None
            trajectory_states = None
            trajectory_filled = None

        for outer_k in range(sample_steps):
            rate = float(rate_schedule[outer_k])
            states_before_outer = states
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
                return_realized_transfers=True,
                collect_diagnostics=False,
            )
            if (
                result.raw_innovations is None
                or result.valid_edge_mask is None
                or result.substep_states is None
                or result.realized_edge_transfers is None
            ):
                raise RuntimeError("reference integrator did not return innovations/substep states/realized transfers")
            raw = result.raw_innovations
            valid = result.valid_edge_mask
            realized = result.realized_edge_transfers
            sub_states = result.substep_states
            for q in range(reference_substeps):
                g = outer_k * reference_substeps + q
                before_q = states_before_outer if q == 0 else sub_states[q - 1]
                starting = starts_t == g
                if bool(starting.any()):
                    rows, cols = torch.nonzero(starting, as_tuple=True)
                    earlier_states[rows, cols] = before_q.index_select(0, rows)
                    start_filled[rows, cols] = True
                if trajectory_target_steps is not None and trajectory_states is not None and trajectory_filled is not None:
                    for depth_idx in range(len(trajectory_depths)):
                        matching = (trajectory_target_steps[:, :, depth_idx] == g) & (~trajectory_filled[:, :, depth_idx])
                        if bool(matching.any()):
                            rows, cols = torch.nonzero(matching, as_tuple=True)
                            trajectory_states[rows, cols, depth_idx] = before_q.index_select(0, rows)
                            trajectory_filled[rows, cols, depth_idx] = True
                active = (starts_t <= g) & (g < starts_t + int(stride))
                if bool(active.any()):
                    rows, cols = torch.nonzero(active, as_tuple=True)
                    accum[rows, cols] = accum[rows, cols] + raw[q].index_select(0, rows)
                    # D0.5 pathwise target: exact reverse physical transfer is
                    # the negative of the forward-oriented edge transfer applied
                    # by the limiter-aware reference substep.
                    physical_accum[rows, cols] = physical_accum[rows, cols] - realized[q].index_select(0, rows)
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
        if not bool(filled.all() and start_filled.all()):
            raise RuntimeError("internal D0 cache bug: some start/later-state slices were not filled")
        if trajectory_filled is not None and not bool(trajectory_filled.all()):
            raise RuntimeError("internal D0 cache bug: some valid trajectory-window targets were not filled")
        accum = accum / math.sqrt(float(stride))
        flat_later = later_states.reshape(bsz * slices_per_path, n * n)
        mobility_valid = harmonic_mobility_channels(flat_later, dynamics_config) > float(d0_config.theta_mask_min)
        masks_flat = masks.reshape(bsz * slices_per_path, 2, n, n) & mobility_valid

        all_states.append(flat_later.detach().cpu())
        all_tau.append(later_tau.reshape(-1).detach().cpu())
        all_labels.append(labels_t.repeat_interleave(slices_per_path).cpu())
        all_innov.append(accum.reshape(bsz * slices_per_path, 2, n, n).detach().cpu())
        all_physical_transfers.append(physical_accum.reshape(bsz * slices_per_path, 2, n, n).detach().cpu())
        all_masks.append(masks_flat.detach().cpu())
        all_starts.append(starts_t.reshape(-1).detach().cpu())
        local_paths = torch.arange(path_offset, path_offset + bsz, dtype=torch.long).repeat_interleave(slices_per_path)
        all_path_indices.append(local_paths)
        start_images_t = torch.as_tensor(initial_np, dtype=torch.float32).repeat_interleave(slices_per_path, dim=0)
        all_start_images.append(start_images_t)
        all_earlier_states.append(earlier_states.reshape(bsz * slices_per_path, n * n).detach().cpu())
        if trajectory_states is not None and trajectory_valid is not None:
            all_trajectory_window_states.append(trajectory_states.reshape(bsz * slices_per_path, len(trajectory_depths), n * n).detach().cpu())
            all_trajectory_window_valid.append(trajectory_valid.reshape(bsz * slices_per_path, len(trajectory_depths)).detach().cpu())
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
    physical_transfer_out = torch.cat(all_physical_transfers, dim=0).float()
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
        earlier_states=torch.cat(all_earlier_states, dim=0).float(),
        physical_transfers=physical_transfer_out,
        physical_target_scale=float(max(float(torch.sqrt(physical_transfer_out.float().square().mean()).item()), float(d0_config.physical_target_scale_floor))),
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
        cache_build_mode="substep",
        requested_stride_substeps=int(stride),
        trajectory_window_states=torch.cat(all_trajectory_window_states, dim=0).float() if all_trajectory_window_states else None,
        trajectory_window_valid=torch.cat(all_trajectory_window_valid, dim=0).bool() if all_trajectory_window_valid else None,
        trajectory_window_depths=torch.as_tensor(trajectory_depths, dtype=torch.long) if trajectory_depths else None,
    )


def _build_d0_training_cache_outer_fast(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    device: torch.device,
    rng: np.random.Generator,
    show_progress: bool = True,
) -> D0TrainingCache:
    """Build a fast outer-step D0 cache.

    Instead of sampling arbitrary elementary substep blocks, this cache samples
    blocks aligned to outer noising steps.  The target is the normalized sum of
    all raw Gaussian innovations over those outer steps and is keyed at the
    later outer state.  This preserves the D0 essentials—data-start rollouts,
    raw innovations before limiting, later-state keying, no weights/source—while
    avoiding the expensive substep-state tensor and Python loop over every
    elementary substep.
    """

    n = int(dynamics_config.grid_size)
    sample_steps = int(d0_config.sample_steps)
    reference_substeps = int(d0_config.reference_substeps)
    requested_stride = int(d0_config.teacher_stride_substeps)
    if sample_steps <= 0 or reference_substeps <= 0:
        raise ValueError("sample_steps and reference_substeps must be positive")
    if requested_stride <= 0:
        raise ValueError("teacher_stride_substeps must be positive")
    outer_stride = max(1, int(math.ceil(float(requested_stride) / float(reference_substeps))))
    if outer_stride > sample_steps:
        raise ValueError("teacher_stride_substeps is longer than the full forward noising run")
    block_substeps = int(outer_stride * reference_substeps)
    if bool(d0_config.single_image_overfit) and requested_stride != block_substeps and not bool(d0_config.allow_stride_rounding):
        raise ValueError(
            "cache_build_mode='outer' rounded teacher_stride_substeps "
            f"from {requested_stride} to {block_substeps}. For an overfit correctness run, "
            "use --cache-build-mode substep, set --teacher-stride-substeps to the rounded value, "
            "or pass --allow-stride-rounding only for speed/debugging."
        )
    horizon = natural_horizon(dynamics_config)
    dt_outer = horizon / float(sample_steps)
    dt_sub = horizon / float(sample_steps * reference_substeps)
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
    all_earlier_states: list[Tensor] = []
    all_physical_transfers: list[Tensor] = []
    terminal_states: list[np.ndarray] = []
    source_indices_chunks: list[np.ndarray] = []
    requested_label_chunks: list[np.ndarray] = []

    total_masked_edges = 0
    total_proposed_edges = 0
    total_floor_correction_l1 = 0.0
    total_renorm_correction_l1 = 0.0
    valid_sum = 0.0
    valid_count = 0.0

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
        outer_starts_np = _sample_cache_start_substeps(
            rng,
            shape=(bsz, slices_per_path),
            total_substeps=sample_steps,
            stride=outer_stride,
            d0_config=d0_config,
        )
        outer_starts = torch.as_tensor(outer_starts_np, dtype=torch.long, device=device)
        outer_ends = outer_starts + int(outer_stride) - 1
        starts_t = outer_starts * int(reference_substeps)
        accum = torch.zeros((bsz, slices_per_path, 2, n, n), dtype=torch.float32, device=device)
        masks = torch.ones((bsz, slices_per_path, 2, n, n), dtype=torch.bool, device=device)
        later_states = torch.empty((bsz, slices_per_path, n * n), dtype=torch.float32, device=device)
        earlier_states = torch.empty((bsz, slices_per_path, n * n), dtype=torch.float32, device=device)
        later_tau = torch.empty((bsz, slices_per_path), dtype=torch.float32, device=device)
        filled = torch.zeros((bsz, slices_per_path), dtype=torch.bool, device=device)
        start_filled = torch.zeros((bsz, slices_per_path), dtype=torch.bool, device=device)

        for outer_k in range(sample_steps):
            rate = float(rate_schedule[outer_k])
            states_before_outer = states
            starting = outer_starts == outer_k
            if bool(starting.any()):
                rows, cols = torch.nonzero(starting, as_tuple=True)
                earlier_states[rows, cols] = states_before_outer.index_select(0, rows)
                start_filled[rows, cols] = True
            result = masked_reference_free_step_torch(
                states,
                dt_outer,
                dynamics_config,
                free_weight=rate,
                noise_weight=math.sqrt(max(rate, 0.0)),
                substeps=reference_substeps,
                stiffness_fraction=float(dynamics_config.limiter_fraction),
                return_innovations=True,
                return_substep_states=False,
                collect_diagnostics=False,
            )
            if result.raw_innovations is None or result.valid_edge_mask is None:
                raise RuntimeError("reference integrator did not return innovations/masks")
            raw_sum = result.raw_innovations.sum(dim=0)
            valid_all = result.valid_edge_mask.all(dim=0)
            active = (outer_starts <= outer_k) & (outer_k < outer_starts + int(outer_stride))
            if bool(active.any()):
                rows, cols = torch.nonzero(active, as_tuple=True)
                accum[rows, cols] = accum[rows, cols] + raw_sum.index_select(0, rows)
                masks[rows, cols] = masks[rows, cols] & valid_all.index_select(0, rows)
            ending = outer_ends == outer_k
            if bool(ending.any()):
                rows, cols = torch.nonzero(ending, as_tuple=True)
                later_states[rows, cols] = result.states.index_select(0, rows)
                later_tau[rows, cols] = max(float(horizon) - float(outer_k + 1) * float(dt_outer), 0.0)
                filled[rows, cols] = True
            states = result.states
            total_masked_edges += int(result.masked_edges)
            total_proposed_edges += int(result.proposed_edges)
            total_floor_correction_l1 += float(result.floor_correction_l1)
            total_renorm_correction_l1 += float(result.renorm_correction_l1)
        if not bool(filled.all() and start_filled.all()):
            raise RuntimeError("internal D0 cache bug: some start/later-state slices were not filled")
        accum = accum / math.sqrt(float(block_substeps))
        flat_later = later_states.reshape(bsz * slices_per_path, n * n)
        mobility_valid = harmonic_mobility_channels(flat_later, dynamics_config) > float(d0_config.theta_mask_min)
        masks_flat = masks.reshape(bsz * slices_per_path, 2, n, n) & mobility_valid

        all_states.append(flat_later.detach().cpu())
        all_tau.append(later_tau.reshape(-1).detach().cpu())
        all_labels.append(labels_t.repeat_interleave(slices_per_path).cpu())
        all_innov.append(accum.reshape(bsz * slices_per_path, 2, n, n).detach().cpu())
        masks_cpu = masks_flat.detach().cpu()
        all_masks.append(masks_cpu)
        valid_sum += float(masks_cpu.count_nonzero().item())
        valid_count += float(masks_cpu.numel())
        all_starts.append(starts_t.reshape(-1).detach().cpu())
        local_paths = torch.arange(path_offset, path_offset + bsz, dtype=torch.long).repeat_interleave(slices_per_path)
        all_path_indices.append(local_paths)
        start_images_t = torch.as_tensor(initial_np, dtype=torch.float32).repeat_interleave(slices_per_path, dim=0)
        all_start_images.append(start_images_t)
        earlier_flat = earlier_states.reshape(bsz * slices_per_path, n * n)
        all_earlier_states.append(earlier_flat.detach().cpu())
        # Fast outer cache does not keep substep realized transfers.  Provide a
        # Poisson fallback target so legacy outer-mode smoke tests still work;
        # use substep cache for --d0-target-space realized-* acceptance runs.
        physical_fallback = poisson_flux_from_velocity_torch(earlier_flat - flat_later, grid_size=n)
        all_physical_transfers.append(physical_fallback.detach().cpu())
        terminal_states.append(states.detach().cpu().numpy().reshape(bsz, n, n))
        source_indices_chunks.append(source_idx_np)
        requested_label_chunks.append(labels_np)
        path_offset += bsz
        if hasattr(bar, "set_postfix"):
            valid_frac = valid_sum / max(valid_count, 1.0)
            bar.set_postfix(valid=f"{valid_frac:.3f}", stride=f"{block_substeps}")

    states_out = torch.cat(all_states, dim=0).float()
    tau_out = torch.cat(all_tau, dim=0).float()
    labels_out = torch.cat(all_labels, dim=0).long()
    innov_out = torch.cat(all_innov, dim=0).float()
    physical_transfer_out = torch.cat(all_physical_transfers, dim=0).float()
    masks_out = torch.cat(all_masks, dim=0).bool()
    raw_limited = 0.0 if total_proposed_edges == 0 else float(total_masked_edges) / float(total_proposed_edges)
    # Fast mode intentionally avoids per-edge mobility/noise-energy host syncs.
    # Use the raw limiter fraction as a conservative stand-in in cache summaries.
    mobility_limited = float(raw_limited)
    noise_limited = float(raw_limited)
    return D0TrainingCache(
        states=states_out,
        tau=tau_out,
        labels=labels_out,
        innovations=innov_out,
        masks=masks_out,
        starts=torch.cat(all_starts, dim=0).long(),
        path_indices=torch.cat(all_path_indices, dim=0).long(),
        start_images=torch.cat(all_start_images, dim=0).float(),
        earlier_states=torch.cat(all_earlier_states, dim=0).float(),
        physical_transfers=physical_transfer_out,
        physical_target_scale=float(max(float(torch.sqrt(physical_transfer_out.float().square().mean()).item()), float(d0_config.physical_target_scale_floor))),
        terminal_states=np.concatenate(terminal_states, axis=0).astype(np.float32),
        source_indices=np.concatenate(source_indices_chunks, axis=0).astype(np.int64),
        requested_labels=np.concatenate(requested_label_chunks, axis=0).astype(np.int64),
        rate_schedule=rate_schedule.astype(np.float64),
        horizon=float(horizon),
        dt_sub=float(dt_sub),
        stride_substeps=int(block_substeps),
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
        cache_build_mode="outer",
        requested_stride_substeps=int(requested_stride),
    )


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
    mode = str(getattr(d0_config, "cache_build_mode", "outer")).strip().lower().replace("_", "-")
    if mode in {"substep", "exact-substep", "exact"}:
        return _build_d0_training_cache_substep_exact(
            dataset_images=dataset_images,
            dataset_labels=dataset_labels,
            dynamics_config=dynamics_config,
            d0_config=d0_config,
            device=device,
            rng=rng,
            show_progress=show_progress,
        )
    if mode in {"outer", "fast", "outer-fast"}:
        if bool(d0_config.cache_store_trajectory_windows):
            raise ValueError("cache_store_trajectory_windows requires --cache-build-mode substep")
        return _build_d0_training_cache_outer_fast(
            dataset_images=dataset_images,
            dataset_labels=dataset_labels,
            dynamics_config=dynamics_config,
            d0_config=d0_config,
            device=device,
            rng=rng,
            show_progress=show_progress,
        )
    raise ValueError("cache_build_mode must be one of: outer, substep")


def sample_d0_cache_batch(
    cache: D0TrainingCache,
    batch_size: int,
    *,
    device: torch.device,
    rng: np.random.Generator,
    allowed_indices: np.ndarray | None = None,
) -> dict[str, Tensor]:
    if allowed_indices is None:
        idx_np = rng.integers(0, cache.size, size=int(batch_size), dtype=np.int64)
    else:
        pool = np.asarray(allowed_indices, dtype=np.int64).reshape(-1)
        if pool.size == 0:
            raise ValueError("allowed_indices for D0 cache sampling is empty")
        idx_np = rng.choice(pool, size=int(batch_size), replace=True).astype(np.int64)
    idx = torch.as_tensor(idx_np, dtype=torch.long)
    return {
        "states": cache.states.index_select(0, idx).to(device),
        "tau": cache.tau.index_select(0, idx).to(device),
        "labels": cache.labels.index_select(0, idx).to(device),
        "innovations": cache.innovations.index_select(0, idx).to(device),
        "masks": cache.masks.index_select(0, idx).to(device),
        "earlier_states": cache.earlier_states.index_select(0, idx).to(device),
        "physical_transfers": cache.physical_transfers.index_select(0, idx).to(device),
        "starts": cache.starts.index_select(0, idx).to(device),
        "path_indices": cache.path_indices.index_select(0, idx).to(device),
        "stride_substeps": torch.full((idx.shape[0],), int(cache.stride_substeps), dtype=torch.long, device=device),
        "reference_substeps": torch.full((idx.shape[0],), int(cache.reference_substeps), dtype=torch.long, device=device),
        "dt_sub": torch.full((idx.shape[0],), float(cache.dt_sub), dtype=torch.float32, device=device),
        "rate_schedule": torch.as_tensor(cache.rate_schedule, dtype=torch.float32, device=device),
        **({
            "trajectory_window_states": cache.trajectory_window_states.index_select(0, idx).to(device),
            "trajectory_window_valid": cache.trajectory_window_valid.index_select(0, idx).to(device),
            "trajectory_window_depths": cache.trajectory_window_depths.to(device),
        } if cache.trajectory_window_states is not None and cache.trajectory_window_valid is not None and cache.trajectory_window_depths is not None else {}),
        "cache_indices": idx.to(device),
    }



def _poisson_physical_edge_target_from_batch(batch: dict[str, Tensor], dynamics_config: DirectFluxMNISTConfig) -> Tensor:
    """Minimum-energy edge transfer whose divergence is earlier_state - later_state."""

    if "earlier_states" not in batch:
        raise ValueError("D0 physical/state losses require earlier_states in the cache batch")
    node_delta = batch["earlier_states"] - batch["states"]
    return poisson_flux_from_velocity_torch(node_delta, grid_size=int(dynamics_config.grid_size)).to(dtype=batch["states"].dtype)


def _physical_edge_target_from_batch(batch: dict[str, Tensor], dynamics_config: DirectFluxMNISTConfig, d0_config: Experiment12D0Config) -> Tensor:
    """Return the configured physical reverse edge target.

    ``realized-physical`` uses the exact reverse edge transfer cached from the
    limiter-aware forward rollout.  The older physical/poisson modes reconstruct
    a minimum-energy flux only from the node displacement.
    """

    space = _normalize_d0_target_space(d0_config.d0_target_space)
    if space in {"realized-physical", "realized-residual"}:
        if "physical_transfers" not in batch:
            raise ValueError("realized physical D0 targets require physical_transfers in the cache batch")
        return batch["physical_transfers"].to(dtype=batch["states"].dtype)
    return _poisson_physical_edge_target_from_batch(batch, dynamics_config)


def _node_delta_from_edge(edge: Tensor) -> Tensor:
    return flux_divergence_torch(edge).reshape(edge.shape[0], -1)




def _reverse_free_block_baseline_from_batch(
    batch: dict[str, Tensor],
    dynamics_config: DirectFluxMNISTConfig,
) -> Tensor:
    """Approximate the block reverse-free physical transfer for cached slices.

    This is the physical edge transfer obtained by applying only the analytic
    reverse free drift over the cached block, with coefficients evaluated at
    the later cached state.  It is intentionally a first-order approximation;
    the sampler uses the same convention for physical-residual mode.
    """

    if not {"starts", "stride_substeps", "reference_substeps", "dt_sub", "rate_schedule"}.issubset(batch):
        return torch.zeros((batch["states"].shape[0], 2, dynamics_config.grid_size, dynamics_config.grid_size), device=batch["states"].device, dtype=batch["states"].dtype)
    states = batch["states"]
    starts = batch["starts"].to(device=states.device, dtype=torch.long)
    stride_values = batch["stride_substeps"].to(device=states.device, dtype=torch.long)
    reference_values = batch["reference_substeps"].to(device=states.device, dtype=torch.long)
    dt_values = batch["dt_sub"].to(device=states.device, dtype=states.dtype)
    rate_schedule = batch["rate_schedule"].to(device=states.device, dtype=states.dtype)
    max_stride = int(stride_values.max().item()) if stride_values.numel() else 0
    out = torch.zeros((states.shape[0], 2, dynamics_config.grid_size, dynamics_config.grid_size), device=states.device, dtype=states.dtype)
    free_flux = free_drift_flux_torch(states, dynamics_config)
    for offset in range(max_stride):
        active = offset < stride_values
        if not bool(active.any()):
            continue
        q = (starts + int(offset)).clamp_min(0)
        outer = torch.div(q, reference_values.clamp_min(1), rounding_mode="floor").clamp(0, rate_schedule.numel() - 1)
        rate = rate_schedule.index_select(0, outer).view(-1, 1, 1, 1)
        dt = dt_values.view(-1, 1, 1, 1)
        contrib = -rate * free_flux * dt
        out = out + torch.where(active.view(-1, 1, 1, 1), contrib, torch.zeros_like(contrib))
    return out


def _physical_total_prediction_from_raw(
    pred_raw: Tensor,
    batch: dict[str, Tensor],
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
) -> Tensor:
    space = _normalize_d0_target_space(d0_config.d0_target_space)
    pred_edge = pred_raw if space in {"realized-physical", "realized-residual"} else project_edge_flux_torch(pred_raw, grid_size=int(dynamics_config.grid_size))
    if space in {"physical-residual", "realized-residual"}:
        return _reverse_free_block_baseline_from_batch(batch, dynamics_config) + pred_edge
    return pred_edge

def _trajectory_rollout_supervised_loss(
    model: DirectFluxUNet,
    batch: dict[str, Tensor],
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
) -> tuple[Tensor, dict[str, float | int]]:
    """Supervise repeated reverse blocks against cached forward-trajectory states.

    The cache stores the true state reached one or more reverse blocks before
    the later keyed state.  Unlike the older rollout heuristic, depth ``d`` is
    compared with its own trajectory target rather than repeatedly comparing
    every depth to the one-block earlier state.
    """

    required = {"trajectory_window_states", "trajectory_window_valid", "trajectory_window_depths"}
    if not required.issubset(batch):
        raise ValueError("trajectory rollout loss requires --cache-store-trajectory-windows")
    requested = sorted({max(1, int(x)) for x in _parse_csv_ints(d0_config.trajectory_rollout_depths)})
    stored_depths = [int(x) for x in batch["trajectory_window_depths"].detach().cpu().tolist()]
    selected = [depth for depth in requested if depth in stored_depths]
    if not selected:
        raise ValueError("trajectory_rollout_depths do not overlap cached rollout_target_blocks")
    max_rows = min(int(d0_config.trajectory_rollout_batch_size), int(batch["states"].shape[0]))
    states = batch["states"][:max_rows].clone()
    labels = batch["labels"][:max_rows]
    tau = batch["tau"][:max_rows].clone()
    starts = batch["starts"][:max_rows].clone()
    stride = int(batch["stride_substeps"][0].item())
    dt_sub = float(batch["dt_sub"][0].item())
    horizon = natural_horizon(dynamics_config)
    depth_to_index = {depth: idx for idx, depth in enumerate(stored_depths)}
    max_depth = max(selected)
    losses: list[Tensor] = []
    diagnostics: dict[str, float | int] = {
        "trajectory_rollout_active": 1,
        "trajectory_rollout_blocks_max": int(max_depth),
        "trajectory_rollout_rows": int(max_rows),
    }
    physical_space = _d0_is_physical_space(d0_config.d0_target_space)
    if not physical_space:
        raise ValueError("trajectory rollout supervision currently requires a physical/realized D0 target space")
    normalization = _physical_target_normalization_mode(d0_config.physical_target_normalization)
    physical_scale = 1.0 if normalization == "none" else float(d0_config.physical_target_scale)
    if physical_scale <= 0.0 and "physical_transfers" in batch:
        physical_scale = float(torch.sqrt(batch["physical_transfers"].detach().float().square().mean()).cpu())
    if physical_scale <= 0.0:
        physical_scale = float(d0_config.physical_target_scale_floor)
    space = _normalize_d0_target_space(d0_config.d0_target_space)
    for depth in range(1, max_depth + 1):
        pred_raw = model(tau, states, labels, None)
        if float(d0_config.control_output_clip) > 0.0:
            pred_raw = pred_raw.clamp(-float(d0_config.control_output_clip), float(d0_config.control_output_clip))
        pred_edge = pred_raw if space in {"realized-physical", "realized-residual"} else project_edge_flux_torch(pred_raw, grid_size=int(dynamics_config.grid_size))
        pred_edge = float(physical_scale) * pred_edge
        if space in {"physical-residual", "realized-residual"}:
            pseudo_batch = {
                "states": states,
                "starts": (starts - int(depth - 1) * stride).clamp_min(0),
                "stride_substeps": batch["stride_substeps"][:max_rows],
                "reference_substeps": batch["reference_substeps"][:max_rows],
                "dt_sub": batch["dt_sub"][:max_rows],
                "rate_schedule": batch["rate_schedule"],
            }
            pred_edge = pred_edge + _reverse_free_block_baseline_from_batch(pseudo_batch, dynamics_config)
        node_delta = _node_delta_from_edge(pred_edge)
        states = (states + node_delta).clamp_min(float(dynamics_config.mass_floor))
        states = states / states.sum(dim=1, keepdim=True).clamp_min(float(dynamics_config.mass_floor))
        tau = (tau + float(stride) * dt_sub).clamp_max(float(horizon))
        if depth in selected:
            idx = depth_to_index[depth]
            valid = batch["trajectory_window_valid"][:max_rows, idx]
            diagnostics[f"trajectory_rollout_depth{depth}_valid_fraction"] = float(valid.float().mean().detach().cpu())
            if bool(valid.any()):
                target = batch["trajectory_window_states"][:max_rows, idx][valid]
                current = states[valid]
                mse = (current - target).square().mean()
                l1 = (current - target).abs().sum(dim=1).mean()
                a = current - current.mean(dim=1, keepdim=True)
                b = target - target.mean(dim=1, keepdim=True)
                corr = (a * b).sum(dim=1) / (a.square().sum(dim=1).sqrt() * b.square().sum(dim=1).sqrt()).clamp_min(1e-30)
                losses.append(mse)
                diagnostics[f"trajectory_rollout_depth{depth}_loss"] = float(mse.detach().cpu())
                diagnostics[f"trajectory_rollout_depth{depth}_l1"] = float(l1.detach().cpu())
                diagnostics[f"trajectory_rollout_depth{depth}_corr"] = float(corr.mean().detach().cpu())
            else:
                diagnostics[f"trajectory_rollout_depth{depth}_loss"] = float("nan")
                diagnostics[f"trajectory_rollout_depth{depth}_l1"] = float("nan")
                diagnostics[f"trajectory_rollout_depth{depth}_corr"] = float("nan")
    if not losses:
        return states.new_tensor(0.0), diagnostics
    loss = torch.stack(losses).mean()
    diagnostics["trajectory_rollout_loss"] = float(loss.detach().cpu())
    diagnostics["trajectory_rollout_depths_used"] = int(len(losses))
    return loss, diagnostics


def d0_unweighted_innovation_loss(
    model: DirectFluxUNet,
    batch: dict[str, Tensor],
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    *,
    step: int | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Unweighted D0 loss.

    D0.3 adds physical/state-relevant losses.  ``raw`` and ``projected-edge``
    keep the Brownian-coordinate innovation objective; ``physical-edge`` asks
    the network to predict the minimum-energy reverse edge transfer whose
    divergence is the cached state delta; ``node-delta`` compares only node
    divergences.  Optional state/rollout terms can be mixed in for stability.
    """

    pred_raw = model(batch["tau"], batch["states"], batch["labels"], None)
    if float(d0_config.control_output_clip) > 0.0:
        pred_raw = pred_raw.clamp(-float(d0_config.control_output_clip), float(d0_config.control_output_clip))
    target_raw = batch["innovations"]
    mask = batch["masks"] & (harmonic_mobility_channels(batch["states"], dynamics_config) > float(d0_config.theta_mask_min))
    mask_f = mask.to(dtype=pred_raw.dtype)
    space = _normalize_d0_target_space(d0_config.d0_target_space)

    need_state_target = (
        space in {"physical-total", "physical-residual", "realized-physical", "realized-residual", "node-delta"}
        or float(d0_config.state_delta_loss_weight) > 0.0
        or float(d0_config.rollout_loss_weight) > 0.0
    )
    if need_state_target:
        target_physical_total = _physical_edge_target_from_batch(batch, dynamics_config, d0_config)
        target_node_delta = batch["earlier_states"] - batch["states"]
    else:
        target_physical_total = torch.zeros_like(pred_raw)
        target_node_delta = torch.zeros_like(batch["states"])
    reverse_free_baseline = _reverse_free_block_baseline_from_batch(batch, dynamics_config) if space in {"physical-residual", "realized-residual"} else torch.zeros_like(pred_raw)
    target_physical_residual = target_physical_total - reverse_free_baseline
    pred_projected = project_edge_flux_torch(pred_raw, grid_size=int(dynamics_config.grid_size))
    pred_primary_edge = pred_raw if space in {"realized-physical", "realized-residual"} else pred_projected
    pred_physical_total = pred_primary_edge + reverse_free_baseline if space in {"physical-residual", "realized-residual"} else pred_primary_edge
    physical_scale_for_loss = torch.ones((), dtype=pred_raw.dtype, device=pred_raw.device)
    physical_edge_mask_f = mask_f

    if space in {"physical-total", "physical-residual", "realized-physical", "realized-residual"}:
        target_primary = target_physical_residual if space in {"physical-residual", "realized-residual"} else target_physical_total
        physical_scale_for_loss = _physical_training_scale_tensor(target_primary, d0_config)
        target_primary_scaled = target_primary / physical_scale_for_loss
        physical_edge_mask_f = _physical_edge_loss_mask(mask_f, target_primary, d0_config)
        denom = physical_edge_mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
        per_slice = ((pred_primary_edge - target_primary_scaled).square() * physical_edge_mask_f).sum(dim=(1, 2, 3)) / denom
        zero = (target_primary_scaled.square() * physical_edge_mask_f).sum(dim=(1, 2, 3)) / denom
        pred_rms = torch.sqrt((pred_primary_edge.square() * physical_edge_mask_f).sum() / physical_edge_mask_f.sum().clamp_min(1.0))
        target_rms = torch.sqrt((target_primary_scaled.square() * physical_edge_mask_f).sum() / physical_edge_mask_f.sum().clamp_min(1.0))
        residual_rms = torch.sqrt(((pred_primary_edge - target_primary_scaled).square() * physical_edge_mask_f).sum() / physical_edge_mask_f.sum().clamp_min(1.0))
        edge_loss_main = per_slice.mean()
        edge_zero_loss = zero.mean()
        if space in {"physical-residual", "realized-residual"}:
            pred_physical_total = reverse_free_baseline + physical_scale_for_loss * pred_primary_edge
        else:
            pred_physical_total = physical_scale_for_loss * pred_primary_edge
    elif space == "node-delta":
        pred_node = _node_delta_from_edge(pred_physical_total)
        target_node = target_node_delta
        per_slice = (pred_node - target_node).square().mean(dim=1)
        zero = target_node.square().mean(dim=1)
        pred_rms = torch.sqrt(pred_node.square().mean())
        target_rms = torch.sqrt(target_node.square().mean())
        residual_rms = torch.sqrt((pred_node - target_node).square().mean())
        edge_loss_main = per_slice.mean()
        edge_zero_loss = zero.mean()
    else:
        edge_loss_main, edge_zero_loss, residual_rms, pred_rms, target_rms = _d0_masked_edge_mse(
            pred_raw,
            target_raw,
            mask_f,
            dynamics_config,
            d0_config,
        )

    pred_node_delta = _node_delta_from_edge(pred_physical_total if space in {"physical-total", "physical-residual", "realized-physical", "realized-residual", "node-delta"} else pred_projected)
    state_delta_per = (pred_node_delta - target_node_delta).square().mean(dim=1)
    state_delta_loss = state_delta_per.mean()
    state_delta_zero = target_node_delta.square().mean(dim=1).mean()
    state_delta_gain = 1.0 - float(state_delta_loss.detach().cpu()) / max(float(state_delta_zero.detach().cpu()), 1e-12)
    state_delta_rms = torch.sqrt(state_delta_loss)
    state_delta_target_rms = torch.sqrt(state_delta_zero)

    regularizer_mask_f = physical_edge_mask_f if space in {"physical-total", "physical-residual", "realized-physical", "realized-residual"} else mask_f
    pred_l2 = (pred_raw.square() * regularizer_mask_f).sum() / regularizer_mask_f.sum().clamp_min(1.0)
    if space in {"physical-total", "physical-residual", "realized-physical", "realized-residual"} and _physical_loss_mask_mode(d0_config.physical_loss_mask) != "innovation-valid":
        invalid_l2 = pred_raw.new_tensor(0.0)
    else:
        invalid_mask_f = (~mask).to(dtype=pred_raw.dtype)
        invalid_l2 = (pred_raw.square() * invalid_mask_f).sum() / invalid_mask_f.sum().clamp_min(1.0)
    curl_loss = flux_curl_torch(pred_raw).square().mean()
    edge_lap_loss = edge_laplacian_energy_torch(pred_raw)
    edge_weight = float(d0_config.edge_innovation_loss_weight)
    state_weight = float(d0_config.state_delta_loss_weight)
    loss = (
        edge_weight * edge_loss_main
        + state_weight * state_delta_loss
        + float(d0_config.eta_l2_weight) * pred_l2
        + float(d0_config.invalid_output_l2_weight) * invalid_l2
        + float(d0_config.curl_loss_weight) * curl_loss
        + float(d0_config.edge_laplacian_loss_weight) * edge_lap_loss
    )
    gain = 1.0 - float(edge_loss_main.detach().cpu()) / max(float(edge_zero_loss.detach().cpu()), 1e-12)

    # Optional short rollout stability loss.  This is deliberately lightweight:
    # it rolls the deterministic learned physical transfer repeatedly and asks
    # it not to drift away from the cached earlier state.
    rollout_loss = pred_raw.new_tensor(0.0)
    rollout_active = False
    if float(d0_config.rollout_loss_weight) > 0.0 and (step is None or int(step) >= int(d0_config.rollout_loss_warmup_steps)):
        rollout_active = True
        blocks = max(1, int(d0_config.rollout_loss_blocks))
        max_rows = min(int(d0_config.rollout_loss_batch_size), int(batch["states"].shape[0]))
        state_roll = batch["states"][:max_rows]
        label_roll = batch["labels"][:max_rows]
        tau_roll = batch["tau"][:max_rows]
        target_roll = batch["earlier_states"][:max_rows]
        for _ in range(blocks):
            pred_step = model(tau_roll, state_roll, label_roll, None)
            if float(d0_config.control_output_clip) > 0.0:
                pred_step = pred_step.clamp(-float(d0_config.control_output_clip), float(d0_config.control_output_clip))
            pred_step = project_edge_flux_torch(pred_step, grid_size=int(dynamics_config.grid_size))
            if _normalize_d0_target_space(d0_config.d0_target_space) in {"physical-total", "physical-residual", "realized-physical", "realized-residual"}:
                pred_step = _physical_training_scale_tensor(target_physical_total[:max_rows], d0_config) * pred_step
            # D0.4: this remains a one-block stability heuristic.  It follows
            # the selected physical semantics when enabled, but it is not a
            # supervised multi-block trajectory loss.
            if _normalize_d0_target_space(d0_config.d0_target_space) in {"physical-residual", "realized-residual"}:
                pseudo_batch = {
                    "states": state_roll,
                    "starts": batch.get("starts", torch.zeros_like(label_roll))[:max_rows],
                    "stride_substeps": batch.get("stride_substeps", torch.ones_like(label_roll))[:max_rows],
                    "reference_substeps": batch.get("reference_substeps", torch.ones_like(label_roll))[:max_rows],
                    "dt_sub": batch.get("dt_sub", torch.ones_like(tau_roll))[:max_rows],
                    "rate_schedule": batch.get("rate_schedule", torch.ones(1, device=state_roll.device, dtype=state_roll.dtype)),
                }
                pred_step = pred_step + _reverse_free_block_baseline_from_batch(pseudo_batch, dynamics_config)
            node_step = _node_delta_from_edge(pred_step)
            # Euler-with-renormalization surrogate; avoids non-differentiable limiter in the loss.
            state_roll = (state_roll + node_step).clamp_min(float(dynamics_config.mass_floor))
            state_roll = state_roll / state_roll.sum(dim=1, keepdim=True).clamp_min(float(dynamics_config.mass_floor))
        rollout_loss = (state_roll - target_roll).square().mean()
        loss = loss + float(d0_config.rollout_loss_weight) * rollout_loss

    trajectory_rollout_loss = pred_raw.new_tensor(0.0)
    trajectory_rollout_diag: dict[str, float | int] = {"trajectory_rollout_active": 0}
    if (
        float(d0_config.trajectory_rollout_loss_weight) > 0.0
        and (step is None or int(step) >= int(d0_config.trajectory_rollout_warmup_steps))
    ):
        trajectory_rollout_loss, trajectory_rollout_diag = _trajectory_rollout_supervised_loss(
            model,
            batch,
            dynamics_config,
            d0_config,
        )
        loss = loss + float(d0_config.trajectory_rollout_loss_weight) * trajectory_rollout_loss

    with torch.no_grad():
        target_projected = project_edge_flux_torch(target_raw, grid_size=int(dynamics_config.grid_size))
        projected_residual_rms = torch.sqrt(((pred_projected - target_projected).square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        raw_residual_rms = torch.sqrt(((pred_raw - target_raw).square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        removed_curl_rms = torch.sqrt((pred_raw - pred_projected).square().mean())
        pred_projected_rms = torch.sqrt((pred_projected.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))
        target_projected_rms = torch.sqrt((target_projected.square() * mask_f).sum() / mask_f.sum().clamp_min(1.0))

    diag = {
        "loss": float(loss.detach().cpu()),
        "loss_main": float(edge_loss_main.detach().cpu()),
        "loss_zero": float(edge_zero_loss.detach().cpu()),
        "innovation_gain": float(gain),
        "prediction_rms": float(pred_rms.detach().cpu()),
        "target_rms": float(target_rms.detach().cpu()),
        "residual_rms": float(residual_rms.detach().cpu()),
        "physical_target_scale": float(torch.as_tensor(physical_scale_for_loss).detach().float().mean().cpu()),
        "physical_target_normalization": str(d0_config.physical_target_normalization),
        "physical_loss_mask": str(d0_config.physical_loss_mask),
        "physical_prediction_rms": float(torch.sqrt((pred_physical_total.square() * physical_edge_mask_f).sum() / physical_edge_mask_f.sum().clamp_min(1.0)).detach().cpu()) if space in {"physical-total", "physical-residual", "realized-physical", "realized-residual"} else float("nan"),
        "physical_target_rms": float(torch.sqrt((target_physical_total.square() * physical_edge_mask_f).sum() / physical_edge_mask_f.sum().clamp_min(1.0)).detach().cpu()) if space in {"physical-total", "physical-residual", "realized-physical", "realized-residual"} else float("nan"),
        "physical_residual_rms": float(torch.sqrt(((pred_physical_total - target_physical_total).square() * physical_edge_mask_f).sum() / physical_edge_mask_f.sum().clamp_min(1.0)).detach().cpu()) if space in {"physical-total", "physical-residual", "realized-physical", "realized-residual"} else float("nan"),
        "state_delta_loss": float(state_delta_loss.detach().cpu()),
        "state_delta_zero": float(state_delta_zero.detach().cpu()),
        "state_delta_gain": float(state_delta_gain),
        "state_delta_rms": float(state_delta_rms.detach().cpu()),
        "state_delta_target_rms": float(state_delta_target_rms.detach().cpu()),
        "rollout_loss": float(rollout_loss.detach().cpu()),
        "rollout_loss_active": int(bool(rollout_active)),
        "raw_residual_rms": float(raw_residual_rms.detach().cpu()),
        "projected_residual_rms": float(projected_residual_rms.detach().cpu()),
        "prediction_projected_rms": float(pred_projected_rms.detach().cpu()),
        "target_projected_rms": float(target_projected_rms.detach().cpu()),
        "prediction_removed_curl_rms": float(removed_curl_rms.detach().cpu()),
        "mask_fraction": float(mask_f.mean().detach().cpu()),
        "eta_l2": float(pred_l2.detach().cpu()),
        "invalid_output_l2": float(invalid_l2.detach().cpu()),
        "curl_loss": float(curl_loss.detach().cpu()),
        "edge_laplacian_loss": float(edge_lap_loss.detach().cpu()),
        "batch_ess_fraction": 1.0,
        "d0_target_space": str(d0_config.d0_target_space),
        "d0_target_space_normalized": str(space),
        "physical_target_source": "realized" if space in {"realized-physical", "realized-residual"} else ("poisson" if space in {"physical-total", "physical-residual", "node-delta"} else "brownian"),
        "trajectory_rollout_loss_weight": float(d0_config.trajectory_rollout_loss_weight),
    }
    diag.update(trajectory_rollout_diag)
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


def _maybe_np_scalar(bank: np.lib.npyio.NpzFile, key: str) -> float | int | str | None:
    if key not in bank:
        return None
    arr = np.asarray(bank[key]).reshape(-1)
    if arr.size == 0:
        return None
    value = arr[0]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _load_prior_bank(
    path: str | Path,
    *,
    fallback_rate_schedule: np.ndarray | None = None,
    fallback_horizon: float | None = None,
) -> dict[str, np.ndarray | float | int | str]:
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
    out: dict[str, np.ndarray | float | int | str] = {
        "terminal_states": states,
        "labels": labels,
        "rate_schedule": np.asarray(rate_schedule, dtype=np.float64),
        "horizon": float(horizon),
        "sample_steps": int(sample_steps),
        "substeps": int(substeps),
    }
    for key in ("lambda_mix", "stride_substeps", "requested_stride_substeps", "cache_build_mode", "edge_alpha_mode", "alpha_eff", "start_substep", "physical_target_scale"):
        value = _maybe_np_scalar(bank, key)
        if value is not None:
            out[key] = value
    for key in ("start_image", "overfit_start_image"):
        if key in bank:
            out[key] = np.asarray(bank[key], dtype=np.float32).reshape(-1)
            break
    return out


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
    max_reverse_blocks: int | None = None,
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
    _validate_prior_bank_compatibility(
        bank,
        d0_config=d0_config,
        dynamics_config=dynamics_config,
        prior_bank_path=prior_bank_path,
    )
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
    start_substep = int(bank.get("start_substep", total_substeps))
    start_substep = max(1, min(int(start_substep), int(total_substeps)))
    horizon = float(bank["horizon"])
    dt_sub = horizon / float(total_substeps)
    stride = max(1, int(bank.get("stride_substeps", _actual_cache_stride_substeps(d0_config))))
    strength = float(d0_config.control_strength if control_strength is None else control_strength)
    target_space = _normalize_d0_target_space(d0_config.d0_target_space)
    physical_sampling = target_space in {"physical-total", "physical-residual", "realized-physical", "realized-residual", "node-delta"}
    physical_noise_mode = _physical_sampler_noise_mode(d0_config.physical_sampler_noise_mode)
    bank_scale = float(bank.get("physical_target_scale", 1.0))
    physical_scale_value = float(d0_config.physical_target_scale) if float(d0_config.physical_target_scale) > 0.0 else bank_scale
    if _physical_target_normalization_mode(d0_config.physical_target_normalization) == "none":
        physical_scale_value = 1.0

    total_limited = 0.0
    total_proposed = 0.0
    total_mobility = 0.0
    total_limited_mobility = 0.0
    learned_sq = 0.0
    learned_raw_sq = 0.0
    learned_projected_sq = 0.0
    learned_removed_curl_sq = 0.0
    free_sq = 0.0
    noise_sq = 0.0
    step_count = 0
    traj: list[np.ndarray] = []
    if save_trajectory:
        traj.append(states.detach().cpu().numpy().copy())

    q_values = list(range(start_substep - 1, -1, -stride))
    requested_blocks = -1 if max_reverse_blocks is None else max(0, int(max_reverse_blocks))
    if requested_blocks >= 0:
        q_values = q_values[:requested_blocks]
    bar = _progress(q_values, total=len(q_values), desc="D0 reverse", disable=not show_progress)
    for q_start in bar:
        block_len = min(stride, int(q_start) + 1)
        tau = torch.full((states.shape[0],), max(horizon - float(q_start + 1) * dt_sub, 0.0), dtype=states.dtype, device=device)
        m_block_raw = model(tau, states, labels_t, None)
        if float(d0_config.control_output_clip) > 0.0:
            m_block_raw = m_block_raw.clamp(-float(d0_config.control_output_clip), float(d0_config.control_output_clip))
        m_block_projected = project_edge_flux_torch(m_block_raw, grid_size=int(dynamics_config.grid_size))
        if target_space in {"realized-physical", "realized-residual"}:
            # Realized-transfer targets are pathwise edge transfers already in
            # the sampler's physical coordinates; do not replace them by the
            # minimum-energy projection at generation time.
            m_block = m_block_raw
        elif physical_sampling:
            # Poisson physical modes use the divergence-relevant projection.
            m_block = m_block_projected
        else:
            m_block = m_block_projected if bool(d0_config.sample_project_learned_mean) else m_block_raw
        m_step = strength * m_block / math.sqrt(float(block_len))
        learned_raw_sq += float((m_block_raw.detach().square().mean()).cpu())
        learned_projected_sq += float((m_block_projected.detach().square().mean()).cpu())
        learned_removed_curl_sq += float(((m_block_raw - m_block_projected).detach().square().mean()).cpu())
        for local in range(block_len):
            q = int(q_start) - local
            outer_k = min(max(q // reference_substeps, 0), rate_schedule.size - 1)
            rate = float(rate_schedule[outer_k])
            free_delta = rate * free_drift_flux_torch(states, dynamics_config) * dt_sub
            noise_std = math.sqrt(max(rate, 0.0)) * edge_noise_std_channels(states, dt_sub, dynamics_config)
            fresh = torch.zeros_like(m_step) if bool(deterministic) else torch.randn_like(m_step)
            if physical_sampling and physical_noise_mode == "none":
                noise_delta = torch.zeros_like(noise_std)
            else:
                noise_delta = noise_std * fresh
            if physical_sampling:
                learned_delta = strength * float(physical_scale_value) * m_block / float(block_len)
                if target_space in {"physical-residual", "realized-residual"}:
                    reverse_delta = -free_delta + learned_delta - noise_delta
                    sampler_mode = "realized-residual" if target_space == "realized-residual" else "physical-residual"
                else:
                    reverse_delta = learned_delta - noise_delta
                    sampler_mode = "realized-physical" if target_space == "realized-physical" else "physical-total"
                states, limited = _apply_oriented_edge_transfer(states, reverse_delta, dynamics_config)
                learned_sq += float(learned_delta.detach().square().mean().cpu())
            else:
                xi_hat = fresh + m_step
                forward_delta = free_delta + noise_std * xi_hat
                states, limited = _apply_oriented_edge_transfer(states, -forward_delta, dynamics_config)
                sampler_mode = "brownian-innovation"
                learned_sq += float((noise_std * m_step).detach().square().mean().cpu())
            total_limited += float(limited["limited_edges"])
            total_proposed += float(limited["proposed_edges"])
            total_mobility += float(limited["mobility_weight_sum"])
            total_limited_mobility += float(limited["limited_mobility_weight_sum"])
            free_sq += float(free_delta.detach().square().mean().cpu())
            noise_sq += float(noise_delta.detach().square().mean().cpu())
            step_count += 1
        if save_trajectory:
            traj.append(states.detach().cpu().numpy().copy())
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(H=float((-(states.clamp_min(1e-30) * states.clamp_min(1e-30).log()).sum(dim=1)).mean().cpu()))

    samples = states.detach().cpu().numpy().astype(np.float32)
    shape = compute_shape_statistics_np(samples, grid_size=int(dynamics_config.grid_size))
    learned_rms = math.sqrt(learned_sq / max(step_count, 1))
    learned_raw_rms = math.sqrt(learned_raw_sq / max(len(q_values), 1))
    learned_projected_rms = math.sqrt(learned_projected_sq / max(len(q_values), 1))
    learned_removed_curl_rms = math.sqrt(learned_removed_curl_sq / max(len(q_values), 1))
    noise_rms = math.sqrt(noise_sq / max(step_count, 1))
    free_rms = math.sqrt(free_sq / max(step_count, 1))
    start_reference = np.asarray(bank.get("start_image"), dtype=np.float32) if "start_image" in bank else None
    start_stats = _sample_start_reference_stats(samples, start_reference)
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
        physical_sampler_mode=str(sampler_mode if physical_sampling else "brownian-innovation"),
        physical_sampler_noise_mode=str(physical_noise_mode if physical_sampling else "reference"),
        physical_target_scale=float(physical_scale_value if physical_sampling else 1.0),
        learned_raw_rms=float(learned_raw_rms),
        learned_projected_rms=float(learned_projected_rms),
        learned_removed_curl_rms=float(learned_removed_curl_rms),
        sample_start_corr_mean=float(start_stats["sample_start_corr_mean"]),
        sample_start_corr_max=float(start_stats["sample_start_corr_max"]),
        sample_start_l1_mean=float(start_stats["sample_start_l1_mean"]),
        stride_substeps=int(stride),
        sample_steps=int(sample_steps),
        reference_substeps=int(reference_substeps),
        prior_bank_path=str(prior_bank_path),
        sample_start_source=str(bank.get("sample_start_source", "terminal-bank")),
        sample_start_substep=int(start_substep),
        reverse_blocks_requested=int(requested_blocks),
        reverse_blocks_executed=int(len(q_values)),
    )


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def cache_summary(cache: D0TrainingCache) -> dict[str, float | int | str]:
    valid_mask_fraction = float(cache.masks.float().mean().item()) if cache.size else 0.0
    summary: dict[str, float | int | str] = {
        "cache_size": int(cache.size),
        "cache_paths": int(cache.terminal_states.shape[0]),
        "cache_sample_steps": int(cache.sample_steps),
        "cache_reference_substeps": int(cache.reference_substeps),
        "cache_requested_stride_substeps": int(cache.requested_stride_substeps or cache.stride_substeps),
        "cache_stride_substeps": int(cache.stride_substeps),
        "cache_actual_stride_substeps": int(cache.stride_substeps),
        "cache_build_mode": str(cache.cache_build_mode),
        "cache_horizon": float(cache.horizon),
        "cache_dt_sub": float(cache.dt_sub),
        "cache_tau_min": float(cache.tau.min().item()) if cache.size else float("nan"),
        "cache_tau_mean": float(cache.tau.mean().item()) if cache.size else float("nan"),
        "cache_tau_max": float(cache.tau.max().item()) if cache.size else float("nan"),
        "cache_target_rms": float(torch.sqrt(cache.innovations.square().mean()).item()) if cache.size else 0.0,
        "cache_physical_transfer_rms": float(torch.sqrt(cache.physical_transfers.square().mean()).item()) if cache.size else 0.0,
        "cache_state_delta_rms": float(torch.sqrt((cache.earlier_states - cache.states).square().mean()).item()) if cache.size else 0.0,
        "cache_physical_target_scale": float(cache.physical_target_scale),
        "cache_valid_mask_fraction": valid_mask_fraction,
        "cache_invalid_mask_fraction": float(1.0 - valid_mask_fraction),
        # Historical alias; this is the valid mask fraction, not the invalid fraction.
        "cache_mask_fraction": valid_mask_fraction,
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
    if cache.trajectory_window_states is not None and cache.trajectory_window_valid is not None and cache.trajectory_window_depths is not None:
        summary["cache_trajectory_windows"] = 1
        summary["cache_trajectory_window_depths"] = ",".join(str(int(x)) for x in cache.trajectory_window_depths.tolist())
        for idx, depth in enumerate(cache.trajectory_window_depths.tolist()):
            summary[f"cache_trajectory_depth{int(depth)}_valid_fraction"] = float(cache.trajectory_window_valid[:, idx].float().mean().item())
    else:
        summary["cache_trajectory_windows"] = 0
        summary["cache_trajectory_window_depths"] = ""
    if cache.size:
        tau = cache.tau.detach().cpu()
        innov = cache.innovations.detach().cpu().reshape(cache.size, -1)
        mask = cache.masks.detach().cpu().reshape(cache.size, -1)
        # Five fixed bins from terminal-prior end (tau=0) to data end (tau=T).
        h = max(float(cache.horizon), 1e-30)
        tau_frac = (tau / h).clamp(0.0, 1.0)
        for idx in range(5):
            lo = float(idx) / 5.0
            hi = float(idx + 1) / 5.0
            sel = (tau_frac >= lo) & (tau_frac <= hi if idx == 4 else tau_frac < hi)
            key = f"cache_tau_bin{idx}"
            summary[f"{key}_lo"] = lo
            summary[f"{key}_hi"] = hi
            summary[f"{key}_count"] = int(sel.count_nonzero().item())
            if bool(sel.any()):
                vals = innov[sel]
                m = mask[sel]
                summary[f"{key}_valid_mask_fraction"] = float(m.float().mean().item())
                summary[f"{key}_target_rms"] = float(torch.sqrt((vals[m].float().square().mean()).clamp_min(0.0)).item()) if bool(m.any()) else float("nan")
            else:
                summary[f"{key}_valid_mask_fraction"] = float("nan")
                summary[f"{key}_target_rms"] = float("nan")
    return summary


def save_d0_cache_npz(cache: D0TrainingCache, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "states": cache.states.numpy().astype(np.float32),
        "tau": cache.tau.numpy().astype(np.float32),
        "labels": cache.labels.numpy().astype(np.int64),
        "innovations": cache.innovations.numpy().astype(np.float32),
        "masks": cache.masks.numpy().astype(np.bool_),
        "starts": cache.starts.numpy().astype(np.int64),
        "path_indices": cache.path_indices.numpy().astype(np.int64),
        "start_images": cache.start_images.numpy().astype(np.float32),
        "earlier_states": cache.earlier_states.numpy().astype(np.float32),
        "physical_transfers": cache.physical_transfers.numpy().astype(np.float32),
        "physical_target_scale": np.asarray([cache.physical_target_scale], dtype=np.float64),
        "terminal_states": cache.terminal_states.astype(np.float32),
        "source_indices": cache.source_indices.astype(np.int64),
        "requested_labels": cache.requested_labels.astype(np.int64),
        "rate_schedule": cache.rate_schedule.astype(np.float64),
        "horizon": np.asarray([cache.horizon], dtype=np.float64),
        "dt_sub": np.asarray([cache.dt_sub], dtype=np.float64),
        "stride_substeps": np.asarray([cache.stride_substeps], dtype=np.int64),
        "requested_stride_substeps": np.asarray([cache.requested_stride_substeps or cache.stride_substeps], dtype=np.int64),
        "sample_steps": np.asarray([cache.sample_steps], dtype=np.int64),
        "reference_substeps": np.asarray([cache.reference_substeps], dtype=np.int64),
        "lambda_mix": np.asarray([cache.lambda_mix], dtype=np.float64),
        "cache_build_mode": np.asarray([cache.cache_build_mode]),
    }
    if cache.trajectory_window_states is not None and cache.trajectory_window_valid is not None and cache.trajectory_window_depths is not None:
        payload["trajectory_window_states"] = cache.trajectory_window_states.numpy().astype(np.float32)
        payload["trajectory_window_valid"] = cache.trajectory_window_valid.numpy().astype(np.bool_)
        payload["trajectory_window_depths"] = cache.trajectory_window_depths.numpy().astype(np.int64)
    np.savez_compressed(path, **payload)


@torch.no_grad()
def d0_oracle_reverse_diagnostic(
    cache: D0TrainingCache,
    dynamics_config: DirectFluxMNISTConfig,
    *,
    max_slices: int = 16,
    device: torch.device | None = None,
) -> dict[str, float | int | str]:
    """Reverse cached slices with their stored target to catch sign/stride bugs.

    For stride 1 this is a direct one-step oracle.  For larger block strides the
    stored block innovation is distributed uniformly over the block, so the
    diagnostic is approximate; its main purpose is comparing the two possible
    innovation signs and checking that stride bookkeeping is self-consistent.
    """

    if cache.size == 0:
        return {"oracle_slices": 0, "oracle_status": "empty_cache"}
    dev = device if device is not None else torch.device("cpu")
    count = min(int(max_slices), int(cache.size))
    idx = torch.arange(count, dtype=torch.long)
    later = cache.states.index_select(0, idx).to(dev)
    earlier = cache.earlier_states.index_select(0, idx).to(dev)
    block_innov = cache.innovations.index_select(0, idx).to(dev)
    starts = cache.starts.index_select(0, idx).to(dev)
    stride = int(cache.stride_substeps)
    rate_schedule = np.asarray(cache.rate_schedule, dtype=np.float64).reshape(-1)
    dt_sub = float(cache.dt_sub)
    reference_substeps = int(cache.reference_substeps)

    def run(sign: float) -> tuple[Tensor, float, float]:
        states = later.clone()
        limited_total = 0.0
        proposed_total = 0.0
        xi_step = float(sign) * block_innov / math.sqrt(float(max(stride, 1)))
        for offset in range(stride - 1, -1, -1):
            q_global = starts + int(offset)
            # The smoke/overfit diagnostic usually uses common starts; handle
            # mixed starts by grouping the tiny diagnostic batch by outer step.
            next_states = states
            for outer_k in torch.unique(torch.div(q_global, reference_substeps, rounding_mode="floor")):
                mask = torch.div(q_global, reference_substeps, rounding_mode="floor") == outer_k
                rows = torch.nonzero(mask, as_tuple=True)[0]
                k = int(outer_k.item())
                rate = float(rate_schedule[min(max(k, 0), rate_schedule.size - 1)])
                chunk = states.index_select(0, rows)
                free_delta = rate * free_drift_flux_torch(chunk, dynamics_config) * dt_sub
                noise_std = math.sqrt(max(rate, 0.0)) * edge_noise_std_channels(chunk, dt_sub, dynamics_config)
                forward_delta = free_delta + noise_std * xi_step.index_select(0, rows)
                updated, limited = _apply_oriented_edge_transfer(chunk, -forward_delta, dynamics_config)
                next_states = next_states.clone()
                next_states[rows] = updated
                limited_total += float(limited["limited_edges"])
                proposed_total += float(limited["proposed_edges"])
            states = next_states
        return states, limited_total, proposed_total

    plus, plus_limited, plus_proposed = run(1.0)
    minus, minus_limited, minus_proposed = run(-1.0)
    plus_l1 = float((plus - earlier).abs().sum(dim=1).mean().cpu())
    minus_l1 = float((minus - earlier).abs().sum(dim=1).mean().cpu())
    plus_rms = float(torch.sqrt((plus - earlier).square().mean()).cpu())
    minus_rms = float(torch.sqrt((minus - earlier).square().mean()).cpu())
    return {
        "oracle_slices": int(count),
        "oracle_stride_substeps": int(stride),
        "oracle_mode": "exact" if stride == 1 else "block_constant_approx",
        "oracle_plus_l1": plus_l1,
        "oracle_minus_l1": minus_l1,
        "oracle_plus_rms": plus_rms,
        "oracle_minus_rms": minus_rms,
        "oracle_preferred_sign": "plus" if plus_l1 <= minus_l1 else "minus",
        "oracle_plus_limiter_fraction": 0.0 if plus_proposed <= 0.0 else float(plus_limited / plus_proposed),
        "oracle_minus_limiter_fraction": 0.0 if minus_proposed <= 0.0 else float(minus_limited / minus_proposed),
    }


@torch.no_grad()
def d0_realized_target_replay_diagnostic(
    cache: D0TrainingCache,
    dynamics_config: DirectFluxMNISTConfig,
    *,
    max_slices: int = 16,
    device: torch.device | None = None,
) -> dict[str, float | int | str]:
    """Replay cached physical edge targets against cached earlier states.

    The realized target should be the most faithful physical edge target because
    it is the exact negative of the forward-oriented edge transfer applied by
    the limiter-aware reference integrator.  We also replay the Poisson target
    for comparison, since that was the D0.3/D0.4 physical surrogate.
    """

    if cache.size == 0:
        return {"target_replay_slices": 0, "target_replay_status": "empty_cache"}
    dev = device if device is not None else torch.device("cpu")
    count = min(int(max_slices), int(cache.size))
    idx = torch.arange(count, dtype=torch.long)
    later = cache.states.index_select(0, idx).to(dev)
    earlier = cache.earlier_states.index_select(0, idx).to(dev)
    realized = cache.physical_transfers.index_select(0, idx).to(dev)
    poisson = poisson_flux_from_velocity_torch(earlier - later, grid_size=int(dynamics_config.grid_size)).to(dtype=later.dtype)

    def replay(name: str, edge_transfer: Tensor) -> dict[str, float]:
        states = later.clone()
        limited_total = 0.0
        proposed_total = 0.0
        step_transfer = edge_transfer / float(max(int(cache.stride_substeps), 1))
        for _ in range(max(int(cache.stride_substeps), 1)):
            states, limited = _apply_oriented_edge_transfer(states, step_transfer, dynamics_config)
            limited_total += float(limited["limited_edges"])
            proposed_total += float(limited["proposed_edges"])
        diff = states - earlier
        centered_a = states - states.mean(dim=1, keepdim=True)
        centered_b = earlier - earlier.mean(dim=1, keepdim=True)
        corr = (centered_a * centered_b).sum(dim=1) / (centered_a.square().sum(dim=1).sqrt() * centered_b.square().sum(dim=1).sqrt()).clamp_min(1e-30)
        return {
            f"target_replay_{name}_l1": float(diff.abs().sum(dim=1).mean().detach().cpu()),
            f"target_replay_{name}_rms": float(torch.sqrt(diff.square().mean()).detach().cpu()),
            f"target_replay_{name}_corr": float(corr.mean().detach().cpu()),
            f"target_replay_{name}_limiter_fraction": 0.0 if proposed_total <= 0.0 else float(limited_total / proposed_total),
            f"target_replay_{name}_edge_rms": float(torch.sqrt(edge_transfer.square().mean()).detach().cpu()),
        }

    realized_stats = replay("realized", realized)
    poisson_stats = replay("poisson", poisson)
    return {
        "target_replay_slices": int(count),
        "target_replay_stride_substeps": int(cache.stride_substeps),
        **realized_stats,
        **poisson_stats,
    }



@torch.no_grad()
def d0_learned_block_reverse_diagnostic(
    model: DirectFluxUNet,
    cache: D0TrainingCache,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    *,
    max_slices: int = 32,
    device: torch.device | None = None,
    allowed_indices: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """One-block learned reverse diagnostic against cached earlier states.

    This tests whether the learned mean can improve a single cached block before
    asking a full reverse trajectory to make a digit.
    """

    if cache.size == 0:
        return {"learned_block_slices": 0, "learned_block_status": "empty_cache"}
    dev = device if device is not None else torch.device("cpu")
    model_was_training = model.training
    model.eval()
    pool = np.arange(int(cache.size), dtype=np.int64) if allowed_indices is None else np.asarray(allowed_indices, dtype=np.int64).reshape(-1)
    count = min(int(max_slices), int(pool.size))
    if count <= 0:
        return {"learned_block_slices": 0, "learned_block_status": "empty_subset"}
    idx = torch.as_tensor(pool[:count], dtype=torch.long)
    later = cache.states.index_select(0, idx).to(dev)
    earlier = cache.earlier_states.index_select(0, idx).to(dev)
    tau = cache.tau.index_select(0, idx).to(dev)
    labels = cache.labels.index_select(0, idx).to(dev)
    starts = cache.starts.index_select(0, idx).to(dev)
    stride = int(cache.stride_substeps)
    reference_substeps = int(cache.reference_substeps)
    rate_schedule = np.asarray(cache.rate_schedule, dtype=np.float64).reshape(-1)
    dt_sub = float(cache.dt_sub)
    pred_raw = model(tau, later, labels, None)
    if float(d0_config.control_output_clip) > 0.0:
        pred_raw = pred_raw.clamp(-float(d0_config.control_output_clip), float(d0_config.control_output_clip))
    pred_projected = project_edge_flux_torch(pred_raw, grid_size=int(dynamics_config.grid_size))
    target_space = _normalize_d0_target_space(d0_config.d0_target_space)
    physical_sampling = target_space in {"physical-total", "physical-residual", "realized-physical", "realized-residual", "node-delta"}
    pred = pred_raw if target_space in {"realized-physical", "realized-residual"} else (pred_projected if (physical_sampling or bool(d0_config.sample_project_learned_mean)) else pred_raw)
    if physical_sampling:
        phys_scale = 1.0 if _physical_target_normalization_mode(d0_config.physical_target_normalization) == "none" else float(d0_config.physical_target_scale or cache.physical_target_scale)
        pred = float(phys_scale) * pred

    def run(mean_block: Tensor, sign: float, *, force_free_baseline: bool = False) -> tuple[Tensor, float, float]:
        states = later.clone()
        limited_total = 0.0
        proposed_total = 0.0
        xi_step = float(sign) * mean_block / math.sqrt(float(max(stride, 1)))
        edge_step = float(sign) * mean_block / float(max(stride, 1))
        for offset in range(stride - 1, -1, -1):
            q_global = starts + int(offset)
            next_states = states
            outer_indices = torch.div(q_global, reference_substeps, rounding_mode="floor")
            for outer_k in torch.unique(outer_indices):
                mask = outer_indices == outer_k
                rows = torch.nonzero(mask, as_tuple=True)[0]
                k = int(outer_k.item())
                rate = float(rate_schedule[min(max(k, 0), rate_schedule.size - 1)])
                chunk = states.index_select(0, rows)
                free_delta = rate * free_drift_flux_torch(chunk, dynamics_config) * dt_sub
                if physical_sampling:
                    learned = edge_step.index_select(0, rows)
                    if target_space in {"physical-residual", "realized-residual"} or bool(force_free_baseline):
                        reverse_delta = -free_delta + learned
                    else:
                        reverse_delta = learned
                    updated, limited = _apply_oriented_edge_transfer(chunk, reverse_delta, dynamics_config)
                else:
                    noise_std = math.sqrt(max(rate, 0.0)) * edge_noise_std_channels(chunk, dt_sub, dynamics_config)
                    forward_delta = free_delta + noise_std * xi_step.index_select(0, rows)
                    updated, limited = _apply_oriented_edge_transfer(chunk, -forward_delta, dynamics_config)
                next_states = next_states.clone()
                next_states[rows] = updated
                limited_total += float(limited["limited_edges"])
                proposed_total += float(limited["proposed_edges"])
            states = next_states
        return states, limited_total, proposed_total

    plus, plus_limited, plus_proposed = run(pred, 1.0)
    minus, minus_limited, minus_proposed = run(pred, -1.0)
    zero, zero_limited, zero_proposed = run(torch.zeros_like(pred), 1.0)
    free, free_limited, free_proposed = run(torch.zeros_like(pred), 1.0, force_free_baseline=True)

    def stats(name: str, value: Tensor, limited: float, proposed: float) -> dict[str, float]:
        diff = value - earlier
        centered_a = value - value.mean(dim=1, keepdim=True)
        centered_b = earlier - earlier.mean(dim=1, keepdim=True)
        corr = (centered_a * centered_b).sum(dim=1) / (centered_a.square().sum(dim=1).sqrt() * centered_b.square().sum(dim=1).sqrt()).clamp_min(1e-30)
        return {
            f"learned_block_{name}_l1": float(diff.abs().sum(dim=1).mean().detach().cpu()),
            f"learned_block_{name}_rms": float(torch.sqrt(diff.square().mean()).detach().cpu()),
            f"learned_block_{name}_corr": float(corr.mean().detach().cpu()),
            f"learned_block_{name}_limiter_fraction": 0.0 if proposed <= 0.0 else float(limited / proposed),
        }

    plus_stats = stats("plus", plus, plus_limited, plus_proposed)
    minus_stats = stats("minus", minus, minus_limited, minus_proposed)
    zero_stats = stats("zero", zero, zero_limited, zero_proposed)
    free_stats = stats("free", free, free_limited, free_proposed)
    if model_was_training:
        model.train()
    candidates = {"plus": plus_stats, "minus": minus_stats, "zero": zero_stats, "free": free_stats}
    best_name = min(
        tuple(candidates.keys()),
        key=lambda name: candidates[name][f"learned_block_{name}_l1"],
    )
    return {
        "learned_block_slices": int(count),
        "learned_block_stride_substeps": int(stride),
        "learned_block_sampler_semantics": str(target_space),
        "learned_block_pred_raw_rms": float(torch.sqrt(pred_raw.square().mean()).detach().cpu()),
        "learned_block_pred_projected_rms": float(torch.sqrt(pred_projected.square().mean()).detach().cpu()),
        "learned_block_removed_curl_rms": float(torch.sqrt((pred_raw - pred_projected).square().mean()).detach().cpu()),
        "learned_block_best": str(best_name),
        **plus_stats,
        **minus_stats,
        **zero_stats,
        **free_stats,
    }



@torch.no_grad()
def d0_learned_rollout_diagnostic(
    model: DirectFluxUNet,
    cache: D0TrainingCache,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    *,
    max_slices: int,
    block_counts: Sequence[int],
    device: torch.device,
    allowed_indices: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Run deterministic learned reverse rollouts from cached states.

    The diagnostic compares each partial rollout to the cached start image.  It
    is intentionally not an exact path-replay test; it measures exposure-bias
    and long-horizon stability under the same deterministic learned update used
    by the sampler.
    """

    pool = np.arange(int(cache.size), dtype=np.int64) if allowed_indices is None else np.asarray(allowed_indices, dtype=np.int64).reshape(-1)
    count = min(int(max_slices), int(pool.size))
    if count <= 0:
        return {"learned_rollout_slices": 0, "learned_rollout_status": "empty_subset"}
    idx = torch.as_tensor(pool[:count], dtype=torch.long)
    depths = sorted({max(1, int(x)) for x in block_counts})
    if not depths:
        return {"learned_rollout_slices": int(count), "learned_rollout_status": "no_depths"}
    max_depth = max(depths)
    states = cache.states.index_select(0, idx).to(device).clone()
    labels = cache.labels.index_select(0, idx).to(device)
    start_ref = cache.start_images.index_select(0, idx).to(device)
    tau_base = cache.tau.index_select(0, idx).to(device)
    starts_base = cache.starts.index_select(0, idx)
    stride = max(1, int(cache.stride_substeps))
    dt_sub = float(cache.dt_sub)
    target_space = _normalize_d0_target_space(d0_config.d0_target_space)
    physical_sampling = target_space in {"physical-total", "physical-residual", "realized-physical", "realized-residual", "node-delta"}
    rate_schedule = np.asarray(cache.rate_schedule, dtype=np.float64).reshape(-1)
    total_limited = 0.0
    total_proposed = 0.0
    out: dict[str, float | int | str] = {
        "learned_rollout_slices": int(count),
        "learned_rollout_stride_substeps": int(stride),
        "learned_rollout_depths": ",".join(str(int(x)) for x in depths),
        "learned_rollout_sampler_semantics": str(target_space),
    }

    def stats(prefix: str, current: Tensor) -> None:
        diff = current - start_ref
        a = current - current.mean(dim=1, keepdim=True)
        b = start_ref - start_ref.mean(dim=1, keepdim=True)
        corr = (a * b).sum(dim=1) / (a.square().sum(dim=1).sqrt() * b.square().sum(dim=1).sqrt()).clamp_min(1e-30)
        out[f"{prefix}_l1_to_start"] = float(diff.abs().sum(dim=1).mean().detach().cpu())
        out[f"{prefix}_mse_to_start"] = float(diff.square().mean().detach().cpu())
        out[f"{prefix}_corr_to_start"] = float(corr.mean().detach().cpu())

    stats("learned_rollout_depth0", states)
    for depth in range(1, max_depth + 1):
        # Use current normalized reverse time from cache tau, shifted by depth.
        tau_values = (tau_base + float(depth - 1) * float(stride) * dt_sub).clamp(max=float(cache.horizon))
        pred_raw = model(tau_values, states, labels, None)
        if float(d0_config.control_output_clip) > 0.0:
            pred_raw = pred_raw.clamp(-float(d0_config.control_output_clip), float(d0_config.control_output_clip))
        pred_proj = project_edge_flux_torch(pred_raw, grid_size=int(dynamics_config.grid_size))
        pred_block = pred_raw if target_space in {"realized-physical", "realized-residual"} else (pred_proj if (physical_sampling or bool(d0_config.sample_project_learned_mean)) else pred_raw)
        if physical_sampling:
            phys_scale = 1.0 if _physical_target_normalization_mode(d0_config.physical_target_normalization) == "none" else float(d0_config.physical_target_scale or cache.physical_target_scale)
            pred_block = float(phys_scale) * pred_block
        for local in range(stride):
            q_rows = (starts_base - int((depth - 1) * stride + local)).clamp_min(0)
            outer_rows = torch.div(q_rows, max(1, int(cache.reference_substeps)), rounding_mode="floor").clamp(0, rate_schedule.size - 1)
            next_states = states.clone()
            for outer_value in torch.unique(outer_rows):
                row_idx = torch.nonzero(outer_rows == outer_value, as_tuple=True)[0]
                rate = float(rate_schedule[int(outer_value.item())])
                chunk = states.index_select(0, row_idx)
                free_delta = rate * free_drift_flux_torch(chunk, dynamics_config) * dt_sub
                pred_chunk = pred_block.index_select(0, row_idx)
                if physical_sampling:
                    if target_space in {"physical-residual", "realized-residual"}:
                        reverse_delta = -free_delta + pred_chunk / float(stride)
                    else:
                        reverse_delta = pred_chunk / float(stride)
                else:
                    noise_std = math.sqrt(max(rate, 0.0)) * edge_noise_std_channels(chunk, dt_sub, dynamics_config)
                    reverse_delta = -(free_delta + noise_std * (pred_chunk / math.sqrt(float(stride))))
                updated, limited = _apply_oriented_edge_transfer(chunk, reverse_delta, dynamics_config)
                next_states[row_idx] = updated
                total_limited += float(limited["limited_edges"])
                total_proposed += float(limited["proposed_edges"])
            states = next_states
        if depth in depths:
            stats(f"learned_rollout_depth{depth}", states)
    out["learned_rollout_limiter_fraction"] = 0.0 if total_proposed <= 0.0 else float(total_limited / total_proposed)
    return out


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
    d0_config = _with_cache_physical_scale(d0_config, cache)
    save_d0_cache_npz(cache, run_dir / "experiment12_d0_cache_initial.npz")
    summary = cache_summary(cache)
    oracle_summary: dict[str, float | int | str] = {}
    if bool(d0_config.oracle_reverse_diagnostic):
        oracle_summary = d0_oracle_reverse_diagnostic(
            cache,
            dynamics_config,
            max_slices=int(d0_config.oracle_reverse_slices),
            device=device,
        )
        summary.update(oracle_summary)
        with (run_dir / "experiment12_d0_oracle_reverse_diagnostic.json").open("w", encoding="utf-8") as handle:
            json.dump(oracle_summary, handle, indent=2, default=_serializable)
    target_replay_summary = d0_realized_target_replay_diagnostic(
        cache,
        dynamics_config,
        max_slices=int(d0_config.oracle_reverse_slices),
        device=device,
    )
    summary.update(target_replay_summary)
    with (run_dir / "experiment12_d0_target_replay_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(target_replay_summary, handle, indent=2, default=_serializable)
    with (run_dir / "experiment12_d0_cache_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_serializable)
    if bool(d0_config.train_cache_only) or int(d0_config.train_steps) <= 0:
        torch.save({"model_state_dict": model.state_dict(), "dynamics_config": asdict(dynamics_config), "d0_config": asdict(d0_config), "cache_summary": summary}, run_dir / "experiment12_d0_model.pt")
        return {"model": model, "cache": cache, "history": [], "cache_summary": summary, "d0_config": d0_config}

    history: list[dict[str, float | int]] = []
    debug_indices: np.ndarray | None = None
    if int(d0_config.debug_memorize_cache) > 0:
        debug_count = min(int(d0_config.debug_memorize_cache), int(cache.size))
        debug_indices = np.arange(debug_count, dtype=np.int64)
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
            d0_config = _with_cache_physical_scale(d0_config, cache)
            last_refresh = time.perf_counter()
        batch = sample_d0_cache_batch(cache, int(d0_config.batch_size), device=device, rng=rng, allowed_indices=debug_indices)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(bool(d0_config.use_amp and device.type == "cuda")):
            loss, diag = d0_unweighted_innovation_loss(model, batch, dynamics_config, d0_config, step=int(step))
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
    if bool(d0_config.learned_block_diagnostic):
        learned_block_summary = d0_learned_block_reverse_diagnostic(
            model,
            cache,
            dynamics_config,
            d0_config,
            max_slices=int(d0_config.learned_block_slices),
            device=device,
        )
        with (run_dir / "experiment12_d0_learned_block_diagnostic.json").open("w", encoding="utf-8") as handle:
            json.dump(learned_block_summary, handle, indent=2, default=_serializable)
        if history:
            history[-1].update({f"diagnostic_{k}": v for k, v in learned_block_summary.items() if isinstance(v, (int, float))})
    if bool(d0_config.learned_rollout_diagnostic):
        rollout_blocks = [int(x) for x in _parse_csv_floats(d0_config.learned_rollout_blocks)]
        learned_rollout_summary = d0_learned_rollout_diagnostic(
            model,
            cache,
            dynamics_config,
            d0_config,
            max_slices=int(d0_config.learned_rollout_slices),
            block_counts=rollout_blocks,
            device=device,
        )
        with (run_dir / "experiment12_d0_learned_rollout_diagnostic.json").open("w", encoding="utf-8") as handle:
            json.dump(learned_rollout_summary, handle, indent=2, default=_serializable)
        if history:
            history[-1].update({f"diagnostic_{k}": v for k, v in learned_rollout_summary.items() if isinstance(v, (int, float))})
    if debug_indices is not None:
        subset_summary: dict[str, float | int | str] = {}
        heldout_indices = np.arange(int(debug_indices.size), int(cache.size), dtype=np.int64)
        for subset_name, subset_indices in (("memorized", debug_indices), ("heldout", heldout_indices)):
            if subset_indices.size == 0:
                subset_summary[f"{subset_name}_size"] = 0
                continue
            subset_summary[f"{subset_name}_size"] = int(subset_indices.size)
            block_diag = d0_learned_block_reverse_diagnostic(
                model,
                cache,
                dynamics_config,
                d0_config,
                max_slices=int(d0_config.learned_block_slices),
                device=device,
                allowed_indices=subset_indices,
            )
            for key, value in block_diag.items():
                subset_summary[f"{subset_name}_{key}"] = value
            if bool(d0_config.learned_rollout_diagnostic):
                rollout_diag = d0_learned_rollout_diagnostic(
                    model,
                    cache,
                    dynamics_config,
                    d0_config,
                    max_slices=int(d0_config.learned_rollout_slices),
                    block_counts=rollout_blocks,
                    device=device,
                    allowed_indices=subset_indices,
                )
                for key, value in rollout_diag.items():
                    subset_summary[f"{subset_name}_{key}"] = value
        with (run_dir / "experiment12_d0_subset_diagnostics.json").open("w", encoding="utf-8") as handle:
            json.dump(subset_summary, handle, indent=2, default=_serializable)
        if history:
            history[-1].update({f"subset_{k}": v for k, v in subset_summary.items() if isinstance(v, (int, float))})
    write_csv_rows(run_dir / "experiment12_d0_train_metrics.csv", history)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_state,
            "dynamics_config": asdict(dynamics_config),
            "d0_config": asdict(d0_config),
            "history": history,
            "cache_summary": cache_summary(cache),
        },
        run_dir / "experiment12_d0_model.pt",
    )
    return {"model": model, "ema_state": ema_state, "cache": cache, "history": history, "cache_summary": cache_summary(cache), "d0_config": d0_config}


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
        cache_build_mode=str(args.cache_build_mode),
        cache_time_sampling=str(args.cache_time_sampling),
        cache_data_endpoint_prob=float(args.cache_data_endpoint_prob),
        cache_terminal_endpoint_prob=float(args.cache_terminal_endpoint_prob),
        cache_endpoint_window_fraction=float(args.cache_endpoint_window_fraction),
        time_slices_per_path=int(args.time_slices_per_path),
        teacher_stride_substeps=int(args.teacher_stride_substeps),
        eta_l2_weight=float(args.eta_l2_weight),
        d0_target_space=str(args.d0_target_space),
        physical_target_normalization=str(args.physical_target_normalization),
        physical_target_scale=float(args.physical_target_scale),
        physical_target_scale_floor=float(args.physical_target_scale_floor),
        physical_loss_mask=str(args.physical_loss_mask),
        debug_memorize_cache=int(args.debug_memorize_cache),
        cache_store_trajectory_windows=bool(args.cache_store_trajectory_windows),
        rollout_target_blocks=str(args.rollout_target_blocks),
        trajectory_rollout_loss_weight=float(args.trajectory_rollout_loss_weight),
        trajectory_rollout_warmup_steps=int(args.trajectory_rollout_warmup_steps),
        trajectory_rollout_depths=str(args.trajectory_rollout_depths),
        trajectory_rollout_batch_size=int(args.trajectory_rollout_batch_size),
        sample_cache_subset=str(args.sample_cache_subset),
        sample_cache_rollout_depths=str(args.sample_cache_rollout_depths),
        edge_innovation_loss_weight=float(args.edge_innovation_loss_weight),
        state_delta_loss_weight=float(args.state_delta_loss_weight),
        rollout_loss_weight=float(args.rollout_loss_weight),
        rollout_loss_warmup_steps=int(args.rollout_loss_warmup_steps),
        rollout_loss_blocks=int(args.rollout_loss_blocks),
        rollout_loss_batch_size=int(args.rollout_loss_batch_size),
        invalid_output_l2_weight=float(args.invalid_output_l2_weight),
        curl_loss_weight=float(args.curl_loss_weight),
        edge_laplacian_loss_weight=float(args.edge_laplacian_loss_weight),
        sample_project_learned_mean=not bool(args.no_sample_project_learned_mean),
        physical_sampler_noise_mode=str(args.physical_sampler_noise_mode),
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
        deterministic_control_strengths=str(args.deterministic_control_strengths),
        stochastic_control_strengths=str(args.stochastic_control_strengths),
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
        allow_prior_bank_mismatch=bool(args.allow_prior_bank_mismatch),
        allow_stride_rounding=bool(args.allow_stride_rounding),
        use_external_prior_bank_for_overfit=bool(args.use_external_prior_bank_for_overfit),
        oracle_reverse_diagnostic=bool(args.oracle_reverse_diagnostic),
        oracle_reverse_slices=int(args.oracle_reverse_slices),
        learned_block_diagnostic=not bool(args.no_learned_block_diagnostic),
        learned_block_slices=int(args.learned_block_slices),
        learned_rollout_diagnostic=not bool(args.no_learned_rollout_diagnostic),
        learned_rollout_slices=int(args.learned_rollout_slices),
        learned_rollout_blocks=str(args.learned_rollout_blocks),
        sample_start_source=str(args.sample_start_source),
        sample_start_tau_fractions=str(args.sample_start_tau_fractions),
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


def _sampling_jobs(d0_config: Experiment12D0Config) -> list[tuple[str, bool, float]]:
    explicit = _parse_csv_floats(d0_config.sample_control_strengths)
    if bool(d0_config.save_sampling_ablations):
        det_strengths = explicit or (_parse_csv_floats(d0_config.deterministic_control_strengths) or [-0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0])
        stoch_strengths = explicit or (_parse_csv_floats(d0_config.stochastic_control_strengths) or [0.0, 0.25, 0.5, 1.0])
        jobs = [("det", True, float(s)) for s in det_strengths]
        if not (_d0_is_physical_space(d0_config.d0_target_space) and _physical_sampler_noise_mode(d0_config.physical_sampler_noise_mode) == "none"):
            jobs += [("stoch", False, float(s)) for s in stoch_strengths]
        return jobs
    strengths = explicit or [float(d0_config.control_strength)]
    mode = "det" if bool(d0_config.deterministic_sampling) else "stoch"
    return [(mode, bool(d0_config.deterministic_sampling), float(s)) for s in strengths]


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
    parser.add_argument("--teacher-stride-substeps", type=int, default=8, help="Requested elementary stride. In fast outer cache mode this is rounded up to a whole outer noising step.")
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
    parser.add_argument("--cache-refresh-every", type=int, default=0, help="Model-independent cache refresh interval. Default 0 disables refresh.")
    parser.add_argument("--cache-build-mode", choices=("outer", "substep"), default="outer", help="Fast outer-step cache by default; use 'substep' for the exact but slow elementary-substep cache.")
    parser.add_argument("--cache-time-sampling", choices=("uniform", "endpoint-mixture"), default="endpoint-mixture")
    parser.add_argument("--cache-data-endpoint-prob", type=float, default=0.50)
    parser.add_argument("--cache-terminal-endpoint-prob", type=float, default=0.10)
    parser.add_argument("--cache-endpoint-window-fraction", type=float, default=0.10)
    parser.add_argument("--eta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--d0-target-space", choices=("raw", "projected-edge", "divergence", "physical-edge", "physical-total", "physical-residual", "realized-physical", "realized-residual", "node-delta"), default="projected-edge")
    parser.add_argument("--physical-target-normalization", choices=("none", "global-rms", "per-slice-rms"), default="global-rms")
    parser.add_argument("--physical-target-scale", type=float, default=0.0, help="Override physical target scale used by normalized physical/realized modes. 0 means infer from cache.")
    parser.add_argument("--physical-target-scale-floor", type=float, default=1e-6)
    parser.add_argument("--physical-loss-mask", choices=("all", "finite", "innovation-valid"), default="all")
    parser.add_argument("--debug-memorize-cache", type=int, default=0, help="If positive, train only on the first N cache slices; useful for separating optimization bugs from conditional-mean difficulty.")
    parser.add_argument("--cache-store-trajectory-windows", action="store_true", help="Store true multi-block forward trajectory targets. Requires --cache-build-mode substep.")
    parser.add_argument("--rollout-target-blocks", type=str, default="1,2,4,8,16", help="Reverse block depths stored in each trajectory window.")
    parser.add_argument("--trajectory-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-rollout-warmup-steps", type=int, default=500)
    parser.add_argument("--trajectory-rollout-depths", type=str, default="1,2,4", help="Stored trajectory depths used by the supervised rollout loss.")
    parser.add_argument("--trajectory-rollout-batch-size", type=int, default=32)
    parser.add_argument("--sample-cache-subset", choices=("auto", "all", "memorized", "heldout"), default="auto", help="Cache-state/terminal subset used for sampling diagnostics. auto uses memorized when debug_memorize_cache>0.")
    parser.add_argument("--sample-cache-rollout-depths", type=str, default="", help="Optional maximum reverse block counts for cache-state sampling diagnostics, e.g. 1,2,4,8,16.")
    parser.add_argument("--edge-innovation-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--rollout-loss-warmup-steps", type=int, default=1000)
    parser.add_argument("--rollout-loss-blocks", type=int, default=4)
    parser.add_argument("--rollout-loss-batch-size", type=int, default=32)
    parser.add_argument("--invalid-output-l2-weight", type=float, default=1e-3)
    parser.add_argument("--curl-loss-weight", type=float, default=1e-3)
    parser.add_argument("--edge-laplacian-loss-weight", type=float, default=0.0)
    parser.add_argument("--no-sample-project-learned-mean", action="store_true")
    parser.add_argument("--physical-sampler-noise-mode", choices=("none", "reference"), default="none", help="Noise convention for physical-total/residual sampling. Default none keeps the overfit acceptance test deterministic unless stochastic reference noise is explicitly requested.")
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
    parser.add_argument("--deterministic-control-strengths", type=str, default="-0.25,-0.1,0,0.1,0.25,0.5,1")
    parser.add_argument("--stochastic-control-strengths", type=str, default="0,0.25,0.5,1")
    parser.add_argument("--deterministic-sampling", action="store_true")
    parser.add_argument("--save-sampling-ablations", action="store_true")
    parser.add_argument("--allow-prior-bank-mismatch", action="store_true", help="Allow sampling from a prior bank whose time grid/schedule differs from the trained D0 config. Debug only.")
    parser.add_argument("--allow-stride-rounding", action="store_true", help="Allow outer cache mode to round the requested elementary stride. Disabled by default for one-image overfit correctness runs.")
    parser.add_argument("--use-external-prior-bank-for-overfit", action="store_true", help="In single-image overfit mode, honor --prior-bank-path instead of automatically sampling from this run's terminal bank.")
    parser.add_argument("--no-oracle-reverse-diagnostic", dest="oracle_reverse_diagnostic", action="store_false")
    parser.set_defaults(oracle_reverse_diagnostic=True)
    parser.add_argument("--oracle-reverse-slices", type=int, default=16)
    parser.add_argument("--no-learned-block-diagnostic", action="store_true")
    parser.add_argument("--learned-block-slices", type=int, default=32)
    parser.add_argument("--no-learned-rollout-diagnostic", action="store_true")
    parser.add_argument("--learned-rollout-slices", type=int, default=16)
    parser.add_argument("--learned-rollout-blocks", type=str, default="1,4,16,64")
    parser.add_argument("--sample-start-source", choices=("terminal-bank", "cache-states", "both"), default="terminal-bank")
    parser.add_argument("--sample-start-tau-fractions", type=str, default="")
    return parser.parse_args(argv)


def _resolved_cache_subset_mode(d0_config: Experiment12D0Config) -> str:
    mode = str(d0_config.sample_cache_subset).strip().lower().replace("_", "-")
    if mode == "auto":
        return "memorized" if int(d0_config.debug_memorize_cache) > 0 else "all"
    return mode


def _save_training_prior_bank(
    path: Path,
    cache: D0TrainingCache,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    *,
    path_indices: np.ndarray,
    subset_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paths = np.asarray(path_indices, dtype=np.int64).reshape(-1)
    if paths.size == 0:
        raise ValueError(f"cannot save empty D0 prior bank subset: {subset_name}")
    np.savez_compressed(
        path,
        terminal_states=cache.terminal_states[paths].reshape(paths.size, -1),
        labels=cache.requested_labels[paths].astype(np.int64),
        rate_schedule=cache.rate_schedule.astype(np.float64),
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        stride_substeps=np.asarray([cache.stride_substeps], dtype=np.int64),
        requested_stride_substeps=np.asarray([cache.requested_stride_substeps or cache.stride_substeps], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
        lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
        edge_alpha_mode=np.asarray([str(dynamics_config.edge_alpha_mode)]),
        alpha_eff=np.asarray([float(dynamics_config.alpha_eff)], dtype=np.float64),
        cache_build_mode=np.asarray([cache.cache_build_mode]),
        cache_subset=np.asarray([str(subset_name)]),
        start_image=cache.start_images[:1].numpy().astype(np.float32),
        overfit_start_image=cache.start_images[:1].numpy().astype(np.float32),
        d0_target_space=np.asarray([str(d0_config.d0_target_space)]),
        physical_target_scale=np.asarray([float(d0_config.physical_target_scale)], dtype=np.float64),
        physical_target_normalization=np.asarray([str(d0_config.physical_target_normalization)]),
        physical_loss_mask=np.asarray([str(d0_config.physical_loss_mask)]),
    )


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
    d0_config = result.get("d0_config", d0_config)  # includes auto physical target scale
    with (run_dir / "experiment12_d0_effective_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"dynamics_config": asdict(dynamics_config), "d0_config": asdict(d0_config)}, handle, indent=2, default=_serializable)

    # Save the current empirical terminal bank too.  This makes overfit smoke
    # runs self-contained even when a Phase-0 bank path was not supplied.
    generated_bank = run_dir / "experiment12_d0_training_terminal_bank.npz"
    all_path_indices = np.arange(cache.terminal_states.shape[0], dtype=np.int64)
    _save_training_prior_bank(
        generated_bank,
        cache,
        dynamics_config,
        d0_config,
        path_indices=all_path_indices,
        subset_name="all",
    )
    sampling_enabled = int(d0_config.num_samples) > 0 and not bool(d0_config.train_cache_only)
    sample_subset_mode = _resolved_cache_subset_mode(d0_config) if sampling_enabled else "all"
    sample_subset_indices = _resolve_cache_subset_indices(cache, d0_config, sample_subset_mode)
    if sampling_enabled and sample_subset_indices.size == 0:
        raise ValueError(f"sample_cache_subset={sample_subset_mode} selected no cache slices")
    subset_path_indices = np.unique(cache.path_indices.index_select(0, torch.as_tensor(sample_subset_indices, dtype=torch.long)).numpy())
    subset_bank = generated_bank
    if sample_subset_mode != "all":
        subset_bank = run_dir / f"experiment12_d0_training_terminal_bank_{sample_subset_mode}.npz"
        _save_training_prior_bank(
            subset_bank,
            cache,
            dynamics_config,
            d0_config,
            path_indices=subset_path_indices,
            subset_name=sample_subset_mode,
        )
    explicit_prior_path = Path(d0_config.prior_bank_path) if str(d0_config.prior_bank_path) else None
    if bool(d0_config.single_image_overfit) and not bool(d0_config.use_external_prior_bank_for_overfit):
        prior_path = subset_bank
        prior_source = f"current_run_overfit_terminal_bank_{sample_subset_mode}"
    else:
        prior_path = explicit_prior_path if explicit_prior_path is not None else subset_bank
        prior_source = "external_prior_bank" if explicit_prior_path is not None else f"current_run_terminal_bank_{sample_subset_mode}"
    if int(d0_config.num_samples) > 0 and not bool(d0_config.train_cache_only):
        sample_labels = np.arange(int(d0_config.num_samples), dtype=np.int64) % 10
        if bool(d0_config.single_image_overfit):
            sample_labels[:] = int(cache.requested_labels[0])
        sample_rows: list[dict[str, float | int | str]] = []
        for sample_mode, deterministic_flag, strength in _sampling_jobs(d0_config):
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
                    deterministic=bool(deterministic_flag),
                    control_strength=float(strength),
                    show_progress=show_progress,
                    save_trajectory=False,
                )
            tag = str(strength).replace(".", "p").replace("-", "m")
            prefix = f"experiment12_d0_samples_{sample_mode}_strength_{tag}"
            npz_path = run_dir / f"{prefix}.npz"
            np.savez_compressed(npz_path, samples=gen.samples, labels=gen.labels)
            png_path = run_dir / f"{prefix}.png"
            save_flux_samples_grid(gen.samples, gen.labels, png_path, grid_size=int(dynamics_config.grid_size), max_images=min(64, int(d0_config.num_samples)))
            sample_rows.append(
                {
                    "sample_mode": str(sample_mode),
                    "deterministic": int(bool(deterministic_flag)),
                    "control_strength": float(strength),
                    "prior_bank_path": str(prior_path),
                    "prior_bank_source": str(prior_source),
                    "sample_cache_subset": str(sample_subset_mode),
                    "sampler_stride_substeps": int(gen.stride_substeps),
                    "sampler_sample_steps": int(gen.sample_steps),
                    "sampler_reference_substeps": int(gen.reference_substeps),
                    "physical_sampler_mode": str(gen.physical_sampler_mode),
                    "physical_sampler_noise_mode": str(gen.physical_sampler_noise_mode),
                    "physical_target_scale": float(gen.physical_target_scale),
                    "limiter_fraction": float(gen.limiter_fraction),
                    "mobility_weighted_limiter_fraction": float(gen.mobility_weighted_limiter_fraction),
                    "learned_step_rms": float(gen.learned_step_rms),
                    "free_step_rms": float(gen.free_step_rms),
                    "noise_step_rms": float(gen.noise_step_rms),
                    "learned_to_noise_ratio": float(gen.learned_to_noise_ratio),
                    "learned_raw_rms": float(gen.learned_raw_rms),
                    "learned_projected_rms": float(gen.learned_projected_rms),
                    "learned_removed_curl_rms": float(gen.learned_removed_curl_rms),
                    "sample_start_corr_mean": float(gen.sample_start_corr_mean),
                    "sample_start_corr_max": float(gen.sample_start_corr_max),
                    "sample_start_l1_mean": float(gen.sample_start_l1_mean),
                    "entropy": float(gen.entropy),
                    "total_variation": float(gen.total_variation),
                    "checkerboard_energy": float(gen.checkerboard_energy),
                    "npz_path": str(npz_path),
                    "png_path": str(png_path),
                }
            )
        write_csv_rows(run_dir / "experiment12_d0_sample_metrics.csv", sample_rows)

        if str(d0_config.sample_start_source) in {"cache-states", "both"}:
            fractions = _parse_csv_floats(d0_config.sample_start_tau_fractions) or [0.95, 0.8, 0.6, 0.4, 0.2, 0.0]
            cache_state_rows: list[dict[str, float | int | str]] = []
            cache_tau_np = cache.tau.numpy()
            candidate_indices = np.asarray(sample_subset_indices, dtype=np.int64)
            rollout_depths = sorted({max(1, int(x)) for x in _parse_csv_ints(d0_config.sample_cache_rollout_depths)})
            rollout_options: list[int | None] = [None] if not rollout_depths else [int(x) for x in rollout_depths]
            for frac in fractions:
                frac_clamped = max(0.0, min(1.0, float(frac)))
                target_tau = frac_clamped * float(cache.horizon)
                local_order = np.argsort(np.abs(cache_tau_np[candidate_indices] - target_tau))[: int(d0_config.num_samples)]
                order = candidate_indices[local_order]
                if order.size == 0:
                    continue
                start_substep = int(round((float(cache.horizon) - target_tau) / max(float(cache.dt_sub), 1e-30)))
                start_substep = max(1, min(int(start_substep), int(cache.sample_steps * cache.reference_substeps)))
                cache_prior = run_dir / f"experiment12_d0_cache_state_prior_tau_{str(frac_clamped).replace('.', 'p')}.npz"
                np.savez_compressed(
                    cache_prior,
                    terminal_states=cache.states.index_select(0, torch.as_tensor(order, dtype=torch.long)).numpy(),
                    labels=cache.labels.index_select(0, torch.as_tensor(order, dtype=torch.long)).numpy().astype(np.int64),
                    rate_schedule=cache.rate_schedule.astype(np.float64),
                    sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
                    substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
                    stride_substeps=np.asarray([cache.stride_substeps], dtype=np.int64),
                    requested_stride_substeps=np.asarray([cache.requested_stride_substeps or cache.stride_substeps], dtype=np.int64),
                    horizon=np.asarray([cache.horizon], dtype=np.float64),
                    lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
                    edge_alpha_mode=np.asarray([str(dynamics_config.edge_alpha_mode)]),
                    alpha_eff=np.asarray([float(dynamics_config.alpha_eff)], dtype=np.float64),
                    cache_build_mode=np.asarray([cache.cache_build_mode]),
                    start_substep=np.asarray([start_substep], dtype=np.int64),
                    sample_start_source=np.asarray(["cache-states"]),
                    start_image=cache.start_images[:1].numpy().astype(np.float32),
                )
                for rollout_blocks in rollout_options:
                    for sample_mode, deterministic_flag, strength in _sampling_jobs(d0_config):
                        context = temporary_ema_weights(model, ema_state) if str(d0_config.sampling_weights) == "ema" and ema_state else nullcontext()
                        with context:
                            gen = simulate_d0_reverse_generation(
                                model,
                                cache.labels.index_select(0, torch.as_tensor(order, dtype=torch.long)).numpy().astype(np.int64),
                                dynamics_config=dynamics_config,
                                d0_config=d0_config,
                                prior_bank_path=cache_prior,
                                device=device,
                                seed=int(d0_config.sample_seed),
                                deterministic=bool(deterministic_flag),
                                control_strength=float(strength),
                                show_progress=show_progress,
                                save_trajectory=False,
                                max_reverse_blocks=rollout_blocks,
                            )
                        depth_tag = "full" if rollout_blocks is None else str(int(rollout_blocks))
                        tag = f"cache_tau_{str(frac_clamped).replace('.', 'p')}_blocks_{depth_tag}_{sample_mode}_strength_{str(strength).replace('.', 'p').replace('-', 'm')}"
                        npz_path = run_dir / f"experiment12_d0_samples_{tag}.npz"
                        np.savez_compressed(npz_path, samples=gen.samples, labels=gen.labels)
                        png_path = run_dir / f"experiment12_d0_samples_{tag}.png"
                        save_flux_samples_grid(gen.samples, gen.labels, png_path, grid_size=int(dynamics_config.grid_size), max_images=min(64, int(d0_config.num_samples)))
                        cache_state_rows.append(
                            {
                                "sample_start_source": "cache-states",
                                "sample_cache_subset": str(sample_subset_mode),
                                "sample_start_tau_fraction": float(frac_clamped),
                                "sample_start_substep": int(start_substep),
                                "sample_rollout_blocks_requested": -1 if rollout_blocks is None else int(rollout_blocks),
                                "sample_rollout_blocks_executed": int(gen.reverse_blocks_executed),
                                "sample_mode": str(sample_mode),
                                "deterministic": int(bool(deterministic_flag)),
                                "control_strength": float(strength),
                                "physical_sampler_mode": str(gen.physical_sampler_mode),
                                "physical_sampler_noise_mode": str(gen.physical_sampler_noise_mode),
                                "physical_target_scale": float(gen.physical_target_scale),
                                "limiter_fraction": float(gen.limiter_fraction),
                                "mobility_weighted_limiter_fraction": float(gen.mobility_weighted_limiter_fraction),
                                "learned_step_rms": float(gen.learned_step_rms),
                                "free_step_rms": float(gen.free_step_rms),
                                "noise_step_rms": float(gen.noise_step_rms),
                                "learned_to_noise_ratio": float(gen.learned_to_noise_ratio),
                                "sample_start_corr_mean": float(gen.sample_start_corr_mean),
                                "sample_start_corr_max": float(gen.sample_start_corr_max),
                                "sample_start_l1_mean": float(gen.sample_start_l1_mean),
                                "entropy": float(gen.entropy),
                                "total_variation": float(gen.total_variation),
                                "checkerboard_energy": float(gen.checkerboard_energy),
                                "npz_path": str(npz_path),
                                "png_path": str(png_path),
                            }
                        )
            write_csv_rows(run_dir / "experiment12_d0_cache_state_sample_metrics.csv", cache_state_rows)
    print(f"Experiment 12 D0 complete: {run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
