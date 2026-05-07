from __future__ import annotations

r"""Target-conditioned score matching for MNIST-CP point clouds.

This module implements the Experiment 8b model discussed in the notebooks.  The
score field is conditioned on a target shape latent code and, for each current
particle, can depend on both the particle location and the full current empirical
measure:

    S_theta(tilde_x_i, tilde_X, tau, z),    z = f_phi(X_target).

The training target follows the finite-particle free diffusion used in the
Wasserstein h-transform notes.  With frozen masses ``s_i`` and Euclidean-cover
forward kernel

    tilde_x_i = x_i + sqrt(2 tau / s_i) eps_i,

we have the Wasserstein/fiber score

    D_i log q_tau(tilde_X | X, s) = (x_i - tilde_x_i) / (2 tau).

For numerical stability the network predicts the scaled score

    R_i = sqrt(2 tau s_i) S_i,

whose denoising target is simply ``-eps_i``.  The public ``forward`` method
returns the unscaled Wasserstein score, so it can be inserted directly into the
finite-dimensional SDE drift ``2 S_theta``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import copy
from contextlib import nullcontext
import hashlib
import math

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mnist.conditioned_diffusion import (
    GeneratedPointCloudSet,
    _resolve_device,
    project_positions,
    rasterize_weighted_point_clouds,
    sample_initial_positions,
)
from mnist.score_matching import (
    _SmallUNet2d,
    _rasterize_weighted_point_clouds_torch,
    _sample_feature_grid_at_positions,
    _tau_features,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ScoreCalibration:
    r"""Per-noise calibration for learned physical scores.

    ``physical_score_scale[j]`` rescales ``s_i S_theta`` at ``tau_levels[j]``.
    ``physical_norm_clip[j]`` optionally clips per-particle physical-score norms
    to an oracle percentile.  The sampler chooses the nearest level in log-tau.
    """

    tau_levels: FloatArray
    physical_score_scale: FloatArray
    physical_norm_clip: Optional[FloatArray] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class LatentBank:
    """Encoded latent bank with provenance metadata.

    The Experiment 8b notebook trains several models that can all encode a
    target contour.  Keeping source/model metadata with each bank prevents stale
    latents from a target-grid teacher being accidentally reused for latent-only
    generation with a student model.
    """

    latents: FloatArray
    labels: IntArray
    source: str
    model_hash: str
    metadata: Optional[dict[str, Any]] = None

    @property
    def latent_dim(self) -> int:
        return int(self.latents.shape[1])

    @property
    def num_samples(self) -> int:
        return int(self.latents.shape[0])

    def is_compatible(self, *, source: str, model_hash: Optional[str] = None) -> bool:
        if str(source) != self.source:
            return False
        if model_hash is not None and str(model_hash) != self.model_hash:
            return False
        return True


__all__ = [
    "ScoreCalibration",
    "LatentBank",
    "score_calibration_to_dict",
    "score_calibration_from_dict",
    "save_target_conditioned_experiment_checkpoint",
    "load_target_conditioned_experiment_checkpoint",
    "TargetConditionedScoreModel",
    "TargetPointCloudEncoder",
    "LatentRasterDecoder",
    "LatentShapeAutoencoder",
    "NoisySetEquivariantEncoder",
    "GaussianLatentPrior",
    "EmpiricalLatentPrior",
    "PCALatentPrior",
    "PCAGMMLatentPrior",
    "LatentGenerator",
    "LatentCritic",
    "make_sigma_tau_schedule",
    "tau_levels_from_sigma_levels",
    "perturb_target_conditioned_positions",
    "empirical_gaussian_mixture_physical_score",
    "empirical_mixture_scaled_score_target",
    "empirical_gaussian_mixture_scaled_score",
    "empirical_gaussian_mixture_score_target",
    "sample_direct_mixture_queries",
    "sample_oracle_replay_queries",
    "target_conditioned_score_matching_loss",
    "evaluate_target_conditioned_score_model",
    "evaluate_model_against_empirical_mixture_score",
    "evaluate_model_against_mixture_oracle",
    "evaluate_model_vs_mixture_oracle",
    "fit_score_calibration_against_mixture_oracle",
    "train_target_conditioned_score_model",
    "train_latent_shape_autoencoder",
    "evaluate_latent_shape_autoencoder",
    "copy_matching_state_dict",
    "initialize_score_model_from_latent_autoencoder",
    "train_latent_only_student_from_teacher",
    "evaluate_latent_sensitivity",
    "latent_collapse_diagnostics",
    "latent_vicreg_regularization",
    "model_state_hash",
    "latent_bank_to_dict",
    "latent_bank_from_dict",
    "encode_latent_bank",
    "validate_latent_bank",
    "encode_target_latents",
    "sample_target_conditioned_annealed_dynamics",
    "sample_empirical_mixture_oracle_annealed_dynamics",
    "sample_empirical_mixture_oracle_dynamics",
    "sample_oracle_mixture_annealed_dynamics",
    "evaluate_hybrid_oracle_neural_reconstruction",
    "reconstruct_target_conditioned_point_clouds",
    "fit_gaussian_latent_prior",
    "sample_gaussian_latent_prior",
    "layernorm_project_latents",
    "sample_empirical_latent_prior",
    "latent_nearest_neighbor_summary",
    "latent_nearest_neighbor_diagnostics",
    "fit_pca_latent_prior",
    "sample_pca_latent_prior",
    "fit_pca_gmm_latent_prior",
    "sample_pca_gmm_latent_prior",
    "reconstruct_target_conditioned_from_latents",
    "train_latent_wgan_gp",
    "sample_wgan_latent_prior",
    "paired_chamfer_reconstruction_metrics",
    "raster_topology_summary",
    "topology_diagnostics",
    "component_balanced_target_masses",
    "contour_thickness_diagnostics",
    "sample_points_from_decoded_raster",
    "coverage_reseed_positions",
    "corner_points_from_contour",
    "decoded_raster_topology_diagnostics",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _validate_probability_masses(masses: Tensor, *, name: str = "masses") -> None:
    if masses.ndim != 2:
        raise ValueError(f"{name} must have shape (B, K)")
    if not torch.isfinite(masses).all():
        raise ValueError(f"{name} contains non-finite values")
    if bool(torch.any(masses <= 0.0)):
        raise ValueError(f"{name} must be strictly positive")


def _validate_positions_tensor(positions: Tensor, masses: Tensor, *, name: str = "positions") -> None:
    if positions.ndim != 3 or positions.shape[:2] != masses.shape or positions.shape[2] != 2:
        raise ValueError(f"{name} must have shape (B, K, 2) and match masses")
    if not torch.isfinite(positions).all():
        raise ValueError(f"{name} contains non-finite values")


def _optional_tqdm(iterable: Any, *, enabled: bool = False, **kwargs: Any) -> Any:
    """Return a tqdm-wrapped iterable when requested, with a dependency-free fallback."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def _progress_range(n: int, *, enabled: bool = False, desc: Optional[str] = None, leave: bool = False) -> Any:
    return _optional_tqdm(range(int(n)), enabled=enabled, desc=desc, leave=leave)


def _prepare_tau_tensor(
    tau: Tensor | float,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if isinstance(tau, Tensor):
        tau_tensor = tau.to(device=device, dtype=dtype).reshape(-1)
    else:
        tau_tensor = torch.full((batch_size,), float(tau), device=device, dtype=dtype)
    if tau_tensor.numel() == 1:
        tau_tensor = tau_tensor.expand(batch_size)
    if tau_tensor.shape != (batch_size,):
        raise ValueError("tau must be a scalar or have shape (B,)")
    if bool(torch.any(tau_tensor <= 0.0)):
        raise ValueError("tau values must be positive")
    return tau_tensor


def _position_features(positions: Tensor, *, use_fourier: bool) -> Tensor:
    if not use_fourier:
        return positions
    angles = 2.0 * math.pi * positions
    return torch.cat([positions, torch.sin(angles), torch.cos(angles)], dim=-1)


def _mass_features(masses: Tensor) -> Tensor:
    return torch.cat([masses.unsqueeze(-1), torch.log(masses.clamp_min(1e-8)).unsqueeze(-1)], dim=-1)


def _weighted_mean_and_std(h: Tensor, masses: Tensor) -> tuple[Tensor, Tensor]:
    weights = masses.unsqueeze(-1)
    mean = torch.sum(weights * h, dim=1)
    second = torch.sum(weights * h.square(), dim=1)
    std = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-8)
    return mean, std


def _uniform_masses(num_samples: int, num_points: int, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    return np.full((int(num_samples), int(num_points)), 1.0 / float(num_points), dtype=dtype)


def _amp_autocast_context(device: torch.device, enabled: bool) -> Any:
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _make_grad_scaler(device: torch.device, enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled and device.type == "cuda"))
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=bool(enabled and device.type == "cuda"))


def _clip_vectors_by_norm(vectors: Tensor, max_norm: Optional[float | Tensor]) -> Tensor:
    """Clip the final-dimension norm of ``vectors`` without changing directions."""
    if max_norm is None:
        return vectors
    if isinstance(max_norm, Tensor):
        max_norm_tensor = max_norm.to(device=vectors.device, dtype=vectors.dtype)
        if max_norm_tensor.numel() == 1:
            max_norm_tensor = max_norm_tensor.reshape(1, 1, 1)
        elif max_norm_tensor.ndim == 1:
            max_norm_tensor = max_norm_tensor[:, None, None]
        elif max_norm_tensor.ndim == 2:
            max_norm_tensor = max_norm_tensor[:, :, None]
    else:
        if float(max_norm) <= 0.0:
            return vectors
        max_norm_tensor = torch.tensor(float(max_norm), device=vectors.device, dtype=vectors.dtype).reshape(1, 1, 1)
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors * torch.clamp(max_norm_tensor / norm, max=1.0)


def _as_score_calibration_dict(calibration: ScoreCalibration | dict[str, Any]) -> dict[str, Any]:
    if isinstance(calibration, ScoreCalibration):
        return {
            "tau_levels": calibration.tau_levels,
            "physical_score_scale": calibration.physical_score_scale,
            "physical_norm_clip": calibration.physical_norm_clip,
            "metadata": calibration.metadata,
        }
    return dict(calibration)


def _score_calibration_for_tau(
    calibration: Optional[ScoreCalibration | dict[str, Any]],
    tau_value: float,
) -> tuple[float, Optional[float]]:
    """Return nearest-neighbor physical-score scale/clip for a tau value."""
    if calibration is None:
        return 1.0, None
    data = _as_score_calibration_dict(calibration)
    levels = np.asarray(data.get("tau_levels"), dtype=np.float64).reshape(-1)
    if levels.size == 0:
        return 1.0, None
    scale_arr = np.asarray(data.get("physical_score_scale", data.get("scale", np.ones_like(levels))), dtype=np.float64).reshape(-1)
    if scale_arr.size == 1 and levels.size > 1:
        scale_arr = np.repeat(scale_arr, levels.size)
    if scale_arr.size != levels.size:
        raise ValueError("calibration physical_score_scale must match tau_levels")
    idx = int(np.argmin(np.abs(np.log(levels) - math.log(float(tau_value)))))
    scale = float(scale_arr[idx])
    clip_value: Optional[float] = None
    clip_arr_raw = data.get("physical_norm_clip", None)
    if clip_arr_raw is not None:
        clip_arr = np.asarray(clip_arr_raw, dtype=np.float64).reshape(-1)
        if clip_arr.size == 1 and levels.size > 1:
            clip_arr = np.repeat(clip_arr, levels.size)
        if clip_arr.size != levels.size:
            raise ValueError("calibration physical_norm_clip must match tau_levels")
        clip_candidate = float(clip_arr[idx])
        if math.isfinite(clip_candidate) and clip_candidate > 0.0:
            clip_value = clip_candidate
    if not math.isfinite(scale):
        scale = 1.0
    return scale, clip_value


def _explicit_clip_for_level(
    score_norm_clip: Optional[float | Sequence[float] | np.ndarray],
    level_index: int,
    num_levels: int,
) -> Optional[float]:
    if score_norm_clip is None:
        return None
    if isinstance(score_norm_clip, (float, int)):
        value = float(score_norm_clip)
        return value if value > 0.0 else None
    arr = np.asarray(score_norm_clip, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        value = float(arr[0])
    elif arr.size == num_levels:
        value = float(arr[int(level_index)])
    else:
        raise ValueError("score_norm_clip must be scalar or have one entry per tau level")
    return value if math.isfinite(value) and value > 0.0 else None


def tau_levels_from_sigma_levels(
    sigma_levels: Sequence[float] | np.ndarray,
    *,
    num_points: int,
) -> np.ndarray:
    r"""Convert common point-noise levels to finite-particle ``tau`` levels.

    For equal masses ``s_i = 1/K``, the free noising standard deviation is
    ``sigma^2 = 2 tau / s_i = 2 K tau``.  Hence ``tau = sigma^2 / (2K)``.
    The returned array is sorted from largest to smallest, as expected by the
    annealed samplers.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    sigmas = np.asarray(sigma_levels, dtype=np.float64).reshape(-1)
    if sigmas.size == 0 or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("sigma_levels must be non-empty, positive, and finite")
    taus = np.square(sigmas) / (2.0 * float(num_points))
    return np.sort(taus)[::-1].copy().astype(np.float64, copy=False)


def make_sigma_tau_schedule(
    *,
    num_points: int,
    num_levels: int = 10,
    sigma_max: float = 0.5,
    sigma_min: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return geometric ``sigma`` and theory ``tau`` levels for MNIST-CP.

    The default sigmas are in unit-square coordinates.  They correspond to
    ShapeGF's ``1.0 -> 0.01`` schedule in ``[-1, 1]^2`` after dividing by two.
    """
    if num_levels <= 0:
        raise ValueError("num_levels must be positive")
    if sigma_max <= 0.0 or sigma_min <= 0.0 or sigma_min > sigma_max:
        raise ValueError("sigma levels must satisfy 0 < sigma_min <= sigma_max")
    sigmas = np.geomspace(float(sigma_max), float(sigma_min), int(num_levels)).astype(np.float64)
    taus = tau_levels_from_sigma_levels(sigmas, num_points=num_points)
    return sigmas, taus


class _ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class _FiLMResidualMLPBlock(nn.Module):
    """Residual pointwise block modulated by the target/time/label context."""

    def __init__(self, dim: int, context_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Sequential(
            nn.Linear(context_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[0] != x.shape[0]:
            raise ValueError("context must have shape (B, C_context)")
        scale, shift = self.to_scale_shift(context).chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1.0 + 0.1 * torch.tanh(scale)[:, None, :]) + shift[:, None, :]
        return x + self.net(h)


class _FiLMScoreHead(nn.Module):
    """Pointwise score head with FiLM modulation in every residual block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_dim: int,
        residual_blocks: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList(
            [_FiLMResidualMLPBlock(hidden_dim, context_dim, dropout=dropout) for _ in range(int(residual_blocks))]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        h = self.input(x)
        for block in self.blocks:
            h = block(h, context)
        return self.output(h)


# ---------------------------------------------------------------------------
# Target encoder and measure-aware score architecture
# ---------------------------------------------------------------------------


class TargetPointCloudEncoder(nn.Module):
    """PointNet/DeepSets encoder ``f_phi(X_target) -> z`` for contour shapes."""

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        point_hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_fourier_features: bool = False,
        normalize_latent: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or point_hidden_dim <= 0:
            raise ValueError("latent_dim and point_hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        self.latent_dim = int(latent_dim)
        self.point_hidden_dim = int(point_hidden_dim)
        self.use_fourier_features = bool(use_fourier_features)

        point_dim = (6 if self.use_fourier_features else 2) + 2
        layers: list[nn.Module] = [nn.Linear(point_dim, point_hidden_dim), nn.GELU()]
        for _ in range(int(num_layers) - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(point_hidden_dim, point_hidden_dim),
                    nn.GELU(),
                ]
            )
        self.point_mlp = nn.Sequential(*layers)
        pooled_dim = 3 * point_hidden_dim
        self.latent_head = nn.Sequential(
            nn.Linear(pooled_dim, point_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(point_hidden_dim, latent_dim),
        )
        self.latent_norm = nn.LayerNorm(latent_dim) if normalize_latent else nn.Identity()

    def forward(self, masses: Tensor, positions: Tensor) -> Tensor:
        _validate_probability_masses(masses, name="target_masses")
        _validate_positions_tensor(positions, masses, name="target_positions")
        point_inputs = torch.cat(
            [_position_features(positions, use_fourier=self.use_fourier_features), _mass_features(masses)],
            dim=-1,
        )
        h = self.point_mlp(point_inputs)
        mean, std = _weighted_mean_and_std(h, masses)
        maximum = torch.max(h, dim=1).values
        z = self.latent_head(torch.cat([mean, std, maximum], dim=1))
        return self.latent_norm(z)


class _EquivariantContextBlock(nn.Module):
    """Permutation-equivariant update using point, pooled-measure and time context."""

    def __init__(self, hidden_dim: int, context_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.update = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim + context_dim),
            nn.Linear(3 * hidden_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: Tensor, masses: Tensor, context: Tensor) -> Tensor:
        batch_size, num_points, _ = h.shape
        mean, _ = _weighted_mean_and_std(h, masses)
        maximum = torch.max(h, dim=1).values
        global_context = torch.cat([mean, maximum, context], dim=1)
        repeated = global_context[:, None, :].expand(batch_size, num_points, global_context.shape[1])
        return h + self.update(torch.cat([h, repeated], dim=-1))


class NoisySetEquivariantEncoder(nn.Module):
    r"""Equivariant branch exposing dependence on the full current measure ``tilde X``."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        output_dim: int = 128,
        context_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.0,
        use_fourier_features: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or output_dim <= 0 or context_dim <= 0:
            raise ValueError("hidden_dim, output_dim and context_dim must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.use_fourier_features = bool(use_fourier_features)
        point_dim = (6 if self.use_fourier_features else 2) + 2
        self.input_projection = nn.Sequential(
            nn.Linear(point_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_EquivariantContextBlock(hidden_dim, context_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, masses: Tensor, positions: Tensor, context: Tensor) -> Tensor:
        _validate_probability_masses(masses)
        _validate_positions_tensor(positions, masses)
        if context.ndim != 2 or context.shape[0] != positions.shape[0]:
            raise ValueError("context must have shape (B, C)")
        batch_size, num_points = masses.shape
        context_points = context[:, None, :].expand(batch_size, num_points, context.shape[1])
        point_inputs = torch.cat(
            [
                _position_features(positions, use_fourier=self.use_fourier_features),
                _mass_features(masses),
                context_points,
            ],
            dim=-1,
        )
        h = self.input_projection(point_inputs)
        for block in self.blocks:
            h = block(h, masses, context)
        return self.output_projection(h)


class LatentRasterDecoder(nn.Module):
    """Decode a global target latent into a rasterized target-contour feature image.

    This is a lightweight bridge between the successful target-grid reconstruction
    path and true latent-only generation: at generation time we no longer have
    ``target_positions``, so the model can synthesize an approximate target raster
    from ``z`` and feed it through the same target-grid feature branch.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        grid_size: int,
        out_channels: int = 2,
        hidden_dim: int = 256,
        label_dim: int = 0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or grid_size <= 0 or out_channels <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim, grid_size, out_channels and hidden_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.grid_size = int(grid_size)
        self.out_channels = int(out_channels)
        self.hidden_dim = int(hidden_dim)
        self.label_dim = int(label_dim)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + self.label_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_channels * self.grid_size * self.grid_size),
        )

    def forward(self, latents: Tensor, label_features: Optional[Tensor] = None) -> Tensor:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(f"latents must have shape (B, {self.latent_dim})")
        if self.label_dim > 0:
            if label_features is None:
                raise ValueError("label_features are required for this latent raster decoder")
            if label_features.ndim != 2 or label_features.shape != (latents.shape[0], self.label_dim):
                raise ValueError(f"label_features must have shape (B, {self.label_dim})")
            inputs = torch.cat([latents, label_features.to(device=latents.device, dtype=latents.dtype)], dim=1)
        else:
            inputs = latents
        raster_logits = self.net(inputs)
        raster = torch.sigmoid(raster_logits.reshape(latents.shape[0], self.out_channels, self.grid_size, self.grid_size))
        return raster


class TargetConditionedScoreModel(nn.Module):
    r"""Measure-aware target-conditioned score model.

    The model predicts ``R_i = sqrt(2 tau s_i) S_i`` by default through
    :meth:`predict_scaled_score`.  :meth:`forward` divides by
    ``sqrt(2 tau s_i)`` and returns the Wasserstein score ``S_i`` used in the
    drift ``2 S_i``.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        target_encoder_hidden_dim: int = 256,
        target_encoder_layers: int = 3,
        grid_size: int = 64,
        base_channels: int = 48,
        grid_feature_dim: int = 96,
        set_feature_dim: int = 128,
        set_hidden_dim: int = 128,
        set_blocks: int = 3,
        score_hidden_dim: int = 256,
        score_residual_blocks: int = 2,
        time_dim: int = 64,
        context_dim: int = 128,
        num_classes: int = 10,
        condition_on_label: bool = False,
        tau_min: float = 1e-8,
        tau_max: float = 1e-4,
        dropout: float = 0.0,
        use_fourier_features: bool = False,
        include_occupancy_channel: bool = True,
        use_image_field: bool = True,
        use_measure_residual: bool = True,
        use_target_grid_conditioning: bool = False,
        target_grid_feature_dim: Optional[int] = None,
        target_grid_dropout_probability: float = 0.0,
        use_latent_raster_decoder: bool = False,
        latent_raster_hidden_dim: int = 256,
        measure_gate_init: float = -3.0,
        measure_gate_max: float = 1.0,
        score_conditioning_mode: str = "concat",
    ) -> None:
        super().__init__()
        if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
            raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")
        if grid_size <= 0 or base_channels <= 0 or grid_feature_dim <= 0:
            raise ValueError("image-field dimensions must be positive")
        if context_dim <= 0 or time_dim <= 0 or latent_dim <= 0:
            raise ValueError("context, time and latent dimensions must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if measure_gate_max <= 0.0:
            raise ValueError("measure_gate_max must be positive")
        if not (0.0 <= float(target_grid_dropout_probability) < 1.0):
            raise ValueError("target_grid_dropout_probability must be in [0, 1)")
        if score_conditioning_mode not in {"concat", "film"}:
            raise ValueError("score_conditioning_mode must be 'concat' or 'film'")

        self.latent_dim = int(latent_dim)
        self.grid_size = int(grid_size)
        self.grid_feature_dim = int(grid_feature_dim)
        self.context_dim = int(context_dim)
        self.num_classes = int(num_classes)
        self.condition_on_label = bool(condition_on_label)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.use_fourier_features = bool(use_fourier_features)
        self.include_occupancy_channel = bool(include_occupancy_channel)
        self.use_image_field = bool(use_image_field)
        self.use_measure_residual = bool(use_measure_residual)
        self.use_target_grid_conditioning = bool(use_target_grid_conditioning)
        self.target_grid_dropout_probability = float(target_grid_dropout_probability)
        self.use_latent_raster_decoder = bool(use_latent_raster_decoder)
        self.measure_gate_max = float(measure_gate_max)
        self.score_conditioning_mode = str(score_conditioning_mode)
        self.measure_residual_active = True
        self._target_raster_cache_enabled = False
        self._target_raster_cache_max_items = 2048
        self._target_raster_cache: dict[str, Tensor] = {}

        self.target_encoder = TargetPointCloudEncoder(
            latent_dim=latent_dim,
            point_hidden_dim=target_encoder_hidden_dim,
            num_layers=target_encoder_layers,
            dropout=dropout,
            use_fourier_features=use_fourier_features,
            normalize_latent=True,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(3, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        if self.condition_on_label:
            self.label_embedding = nn.Embedding(self.num_classes, time_dim)
            label_dim = time_dim
        else:
            self.label_embedding = None
            label_dim = 0

        raw_context_dim = latent_dim + time_dim + label_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(raw_context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
        )
        self.latent_label_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, max(32, min(256, context_dim))),
            nn.GELU(),
            nn.Linear(max(32, min(256, context_dim)), self.num_classes),
        )

        raster_channels = 2 if self.include_occupancy_channel else 1
        if self.use_image_field:
            self.feature_unet = _SmallUNet2d(
                in_channels=raster_channels + context_dim,
                out_channels=grid_feature_dim,
                base_channels=base_channels,
                padding_mode="zeros",
            )
            local_feature_dim = grid_feature_dim
        else:
            self.feature_unet = None
            local_feature_dim = 0
        if self.use_target_grid_conditioning:
            target_grid_dim = int(grid_feature_dim if target_grid_feature_dim is None else target_grid_feature_dim)
            if target_grid_dim <= 0:
                raise ValueError("target_grid_feature_dim must be positive")
            self.target_feature_unet = _SmallUNet2d(
                in_channels=raster_channels + context_dim,
                out_channels=target_grid_dim,
                base_channels=base_channels,
                padding_mode="zeros",
            )
            if self.use_latent_raster_decoder:
                self.latent_raster_decoder = LatentRasterDecoder(
                    latent_dim=latent_dim,
                    grid_size=grid_size,
                    out_channels=raster_channels,
                    hidden_dim=latent_raster_hidden_dim,
                    label_dim=0,
                )
            else:
                self.latent_raster_decoder = None
            target_local_feature_dim = target_grid_dim
        else:
            self.target_feature_unet = None
            self.latent_raster_decoder = None
            target_local_feature_dim = 0
        self.target_grid_feature_dim = int(target_local_feature_dim)
        self.set_encoder = NoisySetEquivariantEncoder(
            hidden_dim=set_hidden_dim,
            output_dim=set_feature_dim,
            context_dim=context_dim,
            num_blocks=set_blocks,
            dropout=dropout,
            use_fourier_features=use_fourier_features,
        )

        position_dim = 6 if self.use_fourier_features else 2
        self.point_score_input_dim = position_dim + 2 + target_local_feature_dim + context_dim
        self.measure_score_input_dim = position_dim + 2 + target_local_feature_dim + local_feature_dim + set_feature_dim + context_dim

        def _make_score_head(input_dim: int) -> nn.Module:
            if self.score_conditioning_mode == "film":
                return _FiLMScoreHead(input_dim, score_hidden_dim, context_dim, score_residual_blocks, dropout=dropout)
            score_layers: list[nn.Module] = [nn.Linear(input_dim, score_hidden_dim), nn.GELU()]
            for _ in range(int(score_residual_blocks)):
                score_layers.append(_ResidualMLPBlock(score_hidden_dim, dropout=dropout))
            score_layers.extend([nn.LayerNorm(score_hidden_dim), nn.Linear(score_hidden_dim, 2)])
            return nn.Sequential(*score_layers)

        if self.use_measure_residual:
            # A pointwise ShapeGF-like field leads the prediction, and the branch
            # that explicitly sees the current empirical measure enters as a
            # learnable residual.  The gate starts small, which helps prevent a
            # collapsed off-distribution current cloud from dominating early
            # sampling while preserving the theoretical dependence on \tilde X.
            self.point_score_head = _make_score_head(self.point_score_input_dim)
            self.measure_score_head = _make_score_head(self.measure_score_input_dim)
            self.measure_gate_logit = nn.Parameter(torch.tensor(float(measure_gate_init)))
            self.score_head = None
        else:
            self.point_score_head = None
            self.measure_score_head = None
            self.measure_gate_logit = None
            self.score_head = _make_score_head(self.measure_score_input_dim)

    def _measure_residual_gate_tensor(self) -> Tensor:
        if self.measure_gate_logit is None:
            return torch.tensor(1.0)
        gate = torch.sigmoid(self.measure_gate_logit) * float(self.measure_gate_max)
        if not self.measure_residual_active:
            gate = gate * 0.0
        return gate

    def measure_residual_gate(self) -> float:
        """Return the current scalar residual gate for logging."""
        if self.measure_gate_logit is None:
            return 1.0
        return float(self._measure_residual_gate_tensor().detach().cpu().item())

    def set_measure_residual_active(self, enabled: bool) -> None:
        """Temporarily enable/disable the full-measure residual branch."""
        self.measure_residual_active = bool(enabled)

    def _run_score_head(self, head: nn.Module, inputs: Tensor, context: Tensor) -> Tensor:
        if self.score_conditioning_mode == "film":
            return head(inputs, context)  # type: ignore[misc]
        return head(inputs)  # type: ignore[misc]

    def encode_target(self, target_masses: Tensor, target_positions: Tensor) -> Tensor:
        return self.target_encoder(target_masses, target_positions)

    def set_target_raster_cache(self, enabled: bool = True, *, max_items: int = 2048, clear: bool = False) -> None:
        """Enable/disable a small per-target raster cache.

        The cache avoids rebuilding fixed target rasters every epoch.  It is keyed
        by CPU bytes of one target's masses/positions and stores detached CPU
        tensors, so it is safe across devices but should be bounded.
        """
        self._target_raster_cache_enabled = bool(enabled)
        self._target_raster_cache_max_items = max(1, int(max_items))
        if clear or not enabled:
            self._target_raster_cache.clear()

    def clear_target_raster_cache(self) -> None:
        self._target_raster_cache.clear()

    def _target_raster_cache_key(self, target_masses: Tensor, target_positions: Tensor, index: int) -> str:
        masses_np = target_masses[index].detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        pos_np = target_positions[index].detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        h = hashlib.sha1()
        h.update(str(self.grid_size).encode())
        h.update(str(int(self.include_occupancy_channel)).encode())
        h.update(masses_np.tobytes())
        h.update(pos_np.tobytes())
        return h.hexdigest()

    def _rasterize_target_measure(self, target_masses: Tensor, target_positions: Tensor) -> Tensor:
        _validate_probability_masses(target_masses, name="target_masses")
        _validate_positions_tensor(target_positions, target_masses, name="target_positions")
        if not self._target_raster_cache_enabled or target_positions.shape[0] <= 0:
            return _rasterize_weighted_point_clouds_torch(
                target_masses,
                target_positions,
                grid_size=self.grid_size,
                periodic=False,
                include_occupancy=self.include_occupancy_channel,
            )
        rasters: list[Tensor] = []
        missing_indices: list[int] = []
        missing_keys: list[str] = []
        for i in range(int(target_positions.shape[0])):
            key = self._target_raster_cache_key(target_masses, target_positions, i)
            cached = self._target_raster_cache.get(key)
            if cached is None:
                missing_indices.append(i)
                missing_keys.append(key)
                rasters.append(target_positions.new_empty((0,)))
            else:
                rasters.append(cached.to(device=target_positions.device, dtype=target_positions.dtype))
        if missing_indices:
            idx = torch.as_tensor(missing_indices, device=target_positions.device, dtype=torch.long)
            computed = _rasterize_weighted_point_clouds_torch(
                target_masses.index_select(0, idx),
                target_positions.index_select(0, idx),
                grid_size=self.grid_size,
                periodic=False,
                include_occupancy=self.include_occupancy_channel,
            )
            for local, (batch_index, key) in enumerate(zip(missing_indices, missing_keys)):
                raster = computed[local].detach().cpu()
                if len(self._target_raster_cache) >= self._target_raster_cache_max_items:
                    self._target_raster_cache.pop(next(iter(self._target_raster_cache)))
                self._target_raster_cache[key] = raster
                rasters[batch_index] = raster.to(device=target_positions.device, dtype=target_positions.dtype)
        return torch.stack(rasters, dim=0)

    def predict_target_raster_from_latent(self, target_latents: Tensor) -> Tensor:
        """Decode a target raster from ``z`` when no target positions are available."""
        if self.latent_raster_decoder is None:
            raise RuntimeError("latent_raster_decoder is not enabled")
        return self.latent_raster_decoder(target_latents)

    def latent_raster_reconstruction_loss(
        self,
        target_latents: Tensor,
        target_masses: Tensor,
        target_positions: Tensor,
        *,
        loss: str = "bce_dice",
        positive_weight: float = 25.0,
        dice_weight: float = 1.0,
        blur_steps: int = 1,
    ) -> tuple[Tensor, dict[str, float]]:
        """Auxiliary loss that forces the global latent to encode target geometry.

        Thin contour rasters are mostly background, so plain MSE/BCE can be
        minimized by predicting nearly empty images.  The default ``bce_dice``
        option uses a positive-weighted BCE plus a soft Dice term on a lightly
        blurred target raster.
        """
        if self.latent_raster_decoder is None:
            zero = target_positions.new_tensor(0.0)
            return zero, {"latent_raster_loss": 0.0, "latent_raster_bce": 0.0, "latent_raster_dice": 0.0}
        if positive_weight <= 0.0 or dice_weight < 0.0 or blur_steps < 0:
            raise ValueError("positive_weight must be positive, dice_weight non-negative and blur_steps non-negative")
        true_raster = self._rasterize_target_measure(
            target_masses.to(device=target_positions.device, dtype=target_positions.dtype),
            target_positions,
        ).detach().clamp(0.0, 1.0)
        if blur_steps > 0:
            for _ in range(int(blur_steps)):
                true_raster = F.avg_pool2d(true_raster, kernel_size=3, stride=1, padding=1)
            max_value = torch.amax(true_raster.flatten(1), dim=1).clamp_min(1e-6)
            true_raster = (true_raster / max_value[:, None, None, None]).clamp(0.0, 1.0)
        pred_raster = self.predict_target_raster_from_latent(target_latents).clamp(1e-6, 1.0 - 1e-6)
        if loss == "mse":
            value = F.mse_loss(pred_raster, true_raster)
            return value, {"latent_raster_loss": float(value.detach().item()), "latent_raster_bce": 0.0, "latent_raster_dice": 0.0}
        if loss == "bce":
            bce = F.binary_cross_entropy(pred_raster, true_raster)
            return bce, {"latent_raster_loss": float(bce.detach().item()), "latent_raster_bce": float(bce.detach().item()), "latent_raster_dice": 0.0}
        if loss != "bce_dice":
            raise ValueError("loss must be 'mse', 'bce', or 'bce_dice'")
        weights = 1.0 + (float(positive_weight) - 1.0) * true_raster
        bce = F.binary_cross_entropy(pred_raster, true_raster, weight=weights)
        intersection = torch.sum(pred_raster * true_raster, dim=(1, 2, 3))
        denom = torch.sum(pred_raster + true_raster, dim=(1, 2, 3)).clamp_min(1e-6)
        dice = torch.mean(1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6))
        value = bce + float(dice_weight) * dice
        return value, {
            "latent_raster_loss": float(value.detach().item()),
            "latent_raster_bce": float(bce.detach().item()),
            "latent_raster_dice": float(dice.detach().item()),
        }

    def latent_classification_loss(self, target_latents: Tensor, labels: Optional[Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Auxiliary digit-class loss on z to discourage complete latent collapse."""
        if labels is None:
            zero = target_latents.new_tensor(0.0)
            return zero, {"latent_class_loss": 0.0, "latent_class_accuracy": float("nan")}
        label_tensor = labels.reshape(-1).to(device=target_latents.device, dtype=torch.long)
        logits = self.latent_label_head(target_latents)
        loss_value = F.cross_entropy(logits, label_tensor)
        accuracy = torch.mean((torch.argmax(logits, dim=1) == label_tensor).float())
        return loss_value, {
            "latent_class_loss": float(loss_value.detach().item()),
            "latent_class_accuracy": float(accuracy.detach().item()),
        }

    def _prepare_context(
        self,
        tau: Tensor,
        target_latents: Tensor,
        labels: Optional[Tensor],
    ) -> Tensor:
        if target_latents.ndim != 2 or target_latents.shape[0] != tau.shape[0]:
            raise ValueError("target_latents must have shape (B, latent_dim)")
        if target_latents.shape[1] != self.latent_dim:
            raise ValueError(f"target_latents must have second dimension {self.latent_dim}")
        time_context = self.time_mlp(_tau_features(tau, tau_min=self.tau_min, tau_max=self.tau_max))
        pieces = [target_latents, time_context]
        if self.condition_on_label:
            if labels is None:
                raise ValueError("labels are required when condition_on_label=True")
            label_tensor = labels.reshape(-1).to(device=tau.device, dtype=torch.long)
            if label_tensor.shape != tau.shape:
                raise ValueError("labels must have shape (B,)")
            pieces.append(self.label_embedding(label_tensor))
        return self.context_mlp(torch.cat(pieces, dim=1))

    def _resolve_latents(
        self,
        masses: Tensor,
        target_positions: Optional[Tensor],
        *,
        target_masses: Optional[Tensor],
        target_latents: Optional[Tensor],
    ) -> Tensor:
        if target_latents is not None:
            return target_latents.to(device=masses.device, dtype=masses.dtype)
        if target_positions is None:
            raise ValueError("pass target_positions or target_latents")
        target_masses_resolved = masses if target_masses is None else target_masses
        target_masses_resolved = target_masses_resolved.to(device=masses.device, dtype=masses.dtype)
        target_positions = target_positions.to(device=masses.device, dtype=masses.dtype)
        return self.encode_target(target_masses_resolved, target_positions)

    def predict_scaled_score(
        self,
        masses: Tensor,
        positions: Tensor,
        tau: Tensor | float,
        *,
        target_positions: Optional[Tensor] = None,
        target_masses: Optional[Tensor] = None,
        target_latents: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        _validate_probability_masses(masses)
        _validate_positions_tensor(positions, masses)
        batch_size, num_points = masses.shape
        tau_tensor = _prepare_tau_tensor(tau, batch_size, device=positions.device, dtype=positions.dtype)
        target_positions_tensor = target_positions.to(device=masses.device, dtype=masses.dtype) if target_positions is not None else None
        target_masses_tensor = target_masses.to(device=masses.device, dtype=masses.dtype) if target_masses is not None else None
        latents = self._resolve_latents(
            masses,
            target_positions_tensor,
            target_masses=target_masses_tensor,
            target_latents=target_latents,
        )
        context = self._prepare_context(tau_tensor, latents, labels)
        context_points = context[:, None, :].expand(batch_size, num_points, context.shape[1])

        base_point_features = [
            _position_features(positions, use_fourier=self.use_fourier_features),
            _mass_features(masses),
        ]
        if self.use_target_grid_conditioning:
            if self.target_feature_unet is None:
                raise RuntimeError("target_feature_unet is unexpectedly missing")
            if target_positions_tensor is not None:
                if target_positions_tensor.ndim != 3 or target_positions_tensor.shape[0] != batch_size or target_positions_tensor.shape[2] != 2:
                    raise ValueError("target_positions must have shape (B,M,2)")
                if target_masses_tensor is None:
                    target_num_points = int(target_positions_tensor.shape[1])
                    target_masses_for_grid = masses.new_full((batch_size, target_num_points), 1.0 / max(target_num_points, 1))
                else:
                    target_masses_for_grid = target_masses_tensor
                target_raster = self._rasterize_target_measure(target_masses_for_grid, target_positions_tensor)
            elif self.latent_raster_decoder is not None:
                target_raster = self.predict_target_raster_from_latent(latents)
            else:
                target_raster = None

            if target_raster is not None:
                context_grid = context[:, :, None, None].expand(batch_size, context.shape[1], self.grid_size, self.grid_size)
                target_feature_grid = self.target_feature_unet(torch.cat([target_raster, context_grid], dim=1))
                target_local_features = _sample_feature_grid_at_positions(target_feature_grid, positions, periodic=False)
                if self.training and self.target_grid_dropout_probability > 0.0:
                    keep = (
                        torch.rand(batch_size, 1, 1, device=positions.device, dtype=positions.dtype)
                        >= self.target_grid_dropout_probability
                    )
                    target_local_features = target_local_features * keep
            else:
                target_local_features = positions.new_zeros((batch_size, num_points, self.target_grid_feature_dim))
            base_point_features.append(target_local_features)

        feature_pieces = list(base_point_features)
        if self.use_image_field:
            if self.feature_unet is None:
                raise RuntimeError("feature_unet is unexpectedly missing")
            raster = _rasterize_weighted_point_clouds_torch(
                masses,
                positions,
                grid_size=self.grid_size,
                periodic=False,
                include_occupancy=self.include_occupancy_channel,
            )
            context_grid = context[:, :, None, None].expand(batch_size, context.shape[1], self.grid_size, self.grid_size)
            feature_grid = self.feature_unet(torch.cat([raster, context_grid], dim=1))
            feature_pieces.append(_sample_feature_grid_at_positions(feature_grid, positions, periodic=False))
        set_features = self.set_encoder(masses, positions, context)
        feature_pieces.extend([set_features, context_points])
        measure_inputs = torch.cat(feature_pieces, dim=-1)
        if self.use_measure_residual:
            if self.point_score_head is None or self.measure_score_head is None or self.measure_gate_logit is None:
                raise RuntimeError("measure-residual heads are unexpectedly missing")
            point_inputs = torch.cat([*base_point_features, context_points], dim=-1)
            point_score = self._run_score_head(self.point_score_head, point_inputs, context)
            residual_score = self._run_score_head(self.measure_score_head, measure_inputs, context)
            return point_score + self._measure_residual_gate_tensor().to(device=point_score.device, dtype=point_score.dtype) * residual_score
        if self.score_head is None:
            raise RuntimeError("score_head is unexpectedly missing")
        return self._run_score_head(self.score_head, measure_inputs, context)

    def forward(
        self,
        masses: Tensor,
        positions: Tensor,
        tau: Tensor | float,
        *,
        target_positions: Optional[Tensor] = None,
        target_masses: Optional[Tensor] = None,
        target_latents: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        scaled = self.predict_scaled_score(
            masses,
            positions,
            tau,
            target_positions=target_positions,
            target_masses=target_masses,
            target_latents=target_latents,
            labels=labels,
        )
        tau_tensor = _prepare_tau_tensor(tau, masses.shape[0], device=positions.device, dtype=positions.dtype)
        scale = torch.sqrt((2.0 * tau_tensor[:, None, None] * masses.unsqueeze(-1)).clamp_min(self.tau_min * 1e-12))
        return scaled / scale



def copy_matching_state_dict(
    target: nn.Module,
    source: nn.Module,
    *,
    include_prefixes: Optional[Sequence[str]] = None,
    exclude_prefixes: Sequence[str] = (),
) -> dict[str, int]:
    """Copy matching parameters/buffers by key and shape from source to target."""
    target_state = target.state_dict()
    source_state = source.state_dict()
    include = None if include_prefixes is None else tuple(str(prefix) for prefix in include_prefixes)
    exclude = tuple(str(prefix) for prefix in exclude_prefixes)
    updated = dict(target_state)
    copied = skipped_name = skipped_shape = 0
    for key, value in source_state.items():
        if include is not None and not key.startswith(include):
            continue
        if exclude and key.startswith(exclude):
            continue
        if key not in target_state:
            skipped_name += 1
            continue
        if tuple(target_state[key].shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        updated[key] = value.detach().clone()
        copied += 1
    target.load_state_dict(updated, strict=True)
    return {"copied": int(copied), "skipped_name": int(skipped_name), "skipped_shape": int(skipped_shape)}


class LatentShapeAutoencoder(nn.Module):
    """Standalone shape-latent pretrainer ``X -> z -> target raster``.

    This model deliberately does not have access to the score network's true
    target-grid branch.  Its purpose is to make the global latent code carry
    contour geometry before the latent-only score student is trained.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 256,
        encoder_hidden_dim: int = 256,
        encoder_layers: int = 3,
        grid_size: int = 64,
        out_channels: int = 2,
        decoder_hidden_dim: int = 256,
        num_classes: int = 10,
        condition_on_label: bool = True,
        dropout: float = 0.0,
        use_fourier_features: bool = False,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or grid_size <= 0 or out_channels <= 0:
            raise ValueError("latent_dim, grid_size and out_channels must be positive")
        self.latent_dim = int(latent_dim)
        self.grid_size = int(grid_size)
        self.out_channels = int(out_channels)
        self.num_classes = int(num_classes)
        self.condition_on_label = bool(condition_on_label)
        self.encoder = TargetPointCloudEncoder(
            latent_dim=latent_dim,
            point_hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
            dropout=dropout,
            use_fourier_features=use_fourier_features,
            normalize_latent=True,
        )
        label_dim = encoder_hidden_dim // 2 if self.condition_on_label else 0
        if self.condition_on_label:
            self.label_embedding = nn.Embedding(self.num_classes, label_dim)
        else:
            self.label_embedding = None
        self.decoder = LatentRasterDecoder(
            latent_dim=latent_dim,
            grid_size=grid_size,
            out_channels=out_channels,
            hidden_dim=decoder_hidden_dim,
            label_dim=label_dim,
        )
        self.label_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, max(32, min(256, encoder_hidden_dim))),
            nn.GELU(),
            nn.Linear(max(32, min(256, encoder_hidden_dim)), self.num_classes),
        )

    def encode(self, masses: Tensor, positions: Tensor) -> Tensor:
        return self.encoder(masses, positions)

    def decode(self, latents: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if self.condition_on_label:
            if labels is None:
                raise ValueError("labels are required when condition_on_label=True")
            label_features = self.label_embedding(labels.reshape(-1).to(device=latents.device, dtype=torch.long))
        else:
            label_features = None
        return self.decoder(latents, label_features)

    def forward(self, masses: Tensor, positions: Tensor, labels: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        z = self.encode(masses, positions)
        return z, self.decode(z, labels)

    def label_logits(self, latents: Tensor) -> Tensor:
        return self.label_head(latents)


def rasterize_target_for_latent_autoencoder(
    masses: Tensor,
    positions: Tensor,
    *,
    grid_size: int,
    include_occupancy: bool = True,
    blur_steps: int = 1,
) -> Tensor:
    """Rasterize and lightly blur a target contour for latent-shape pretraining."""
    target = _rasterize_weighted_point_clouds_torch(
        masses,
        positions,
        grid_size=grid_size,
        periodic=False,
        include_occupancy=include_occupancy,
    ).detach().clamp(0.0, 1.0)
    if blur_steps > 0:
        for _ in range(int(blur_steps)):
            target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        max_value = torch.amax(target.flatten(1), dim=1).clamp_min(1e-6)
        target = (target / max_value[:, None, None, None]).clamp(0.0, 1.0)
    return target


def latent_shape_autoencoder_loss(
    model: LatentShapeAutoencoder,
    masses: Tensor,
    positions: Tensor,
    labels: Optional[Tensor] = None,
    *,
    raster_loss: str = "bce_dice",
    positive_weight: float = 25.0,
    dice_weight: float = 1.0,
    blur_steps: int = 1,
    latent_variance_weight: float = 5.0,
    latent_covariance_weight: float = 0.1,
    latent_variance_target: float = 1.0,
    latent_classification_weight: float = 0.2,
) -> tuple[Tensor, dict[str, float]]:
    """Loss for standalone latent shape autoencoder pretraining."""
    if positive_weight <= 0.0 or dice_weight < 0.0:
        raise ValueError("positive_weight must be positive and dice_weight non-negative")
    z, pred = model(masses, positions, labels)
    target = rasterize_target_for_latent_autoencoder(
        masses,
        positions,
        grid_size=model.grid_size,
        include_occupancy=(model.out_channels > 1),
        blur_steps=blur_steps,
    ).to(device=positions.device, dtype=positions.dtype)
    pred = pred.clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(pred, target)
    dice = positions.new_tensor(0.0)
    if raster_loss == "mse":
        raster_value = F.mse_loss(pred, target)
    elif raster_loss == "bce":
        raster_value = bce
    elif raster_loss in {"bce_dice", "balanced_bce_dice"}:
        weights = 1.0 + (float(positive_weight) - 1.0) * target
        bce = F.binary_cross_entropy(pred, target, weight=weights)
        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        denom = torch.sum(pred + target, dim=(1, 2, 3)).clamp_min(1e-6)
        dice = torch.mean(1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6))
        raster_value = bce + float(dice_weight) * dice
    else:
        raise ValueError("raster_loss must be 'mse', 'bce', or 'bce_dice'")
    latent_reg, latent_reg_metrics = latent_vicreg_regularization(
        z,
        variance_target=latent_variance_target,
        variance_weight=latent_variance_weight,
        covariance_weight=latent_covariance_weight,
    )
    class_loss = positions.new_tensor(0.0)
    class_acc = float("nan")
    if latent_classification_weight > 0.0 and labels is not None:
        label_tensor = labels.reshape(-1).to(device=positions.device, dtype=torch.long)
        logits = model.label_logits(z)
        class_loss = F.cross_entropy(logits, label_tensor)
        class_acc = float(torch.mean((torch.argmax(logits, dim=1) == label_tensor).float()).detach().item())
    total = raster_value + latent_reg + float(latent_classification_weight) * class_loss
    return total, {
        "loss": float(total.detach().item()),
        "raster_loss": float(raster_value.detach().item()),
        "raster_bce": float(bce.detach().item()),
        "raster_dice": float(dice.detach().item()),
        "latent_var_loss": float(latent_reg_metrics["latent_var_loss"]),
        "latent_cov_loss": float(latent_reg_metrics["latent_cov_loss"]),
        "latent_mean_std": float(latent_reg_metrics["latent_mean_std"]),
        "latent_class_loss": float(class_loss.detach().item()),
        "latent_class_accuracy": class_acc,
    }


def train_latent_shape_autoencoder(
    model: LatentShapeAutoencoder,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    val_masses: Optional[np.ndarray] = None,
    val_positions: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    raster_loss: str = "bce_dice",
    positive_weight: float = 25.0,
    dice_weight: float = 1.0,
    blur_steps: int = 1,
    latent_variance_weight: float = 5.0,
    latent_covariance_weight: float = 0.1,
    latent_variance_target: float = 1.0,
    latent_classification_weight: float = 0.2,
    max_grad_norm: Optional[float] = 5.0,
    early_stopping_patience: Optional[int] = 20,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
    show_progress: bool = False,
    progress_desc: str = "latent shape AE",
    dataloader_num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    use_amp: bool = False,
) -> dict[str, list[float]]:
    """Pretrain ``X -> z -> raster(X)`` before latent-only score distillation."""
    if epochs <= 0 or batch_size <= 0 or lr <= 0.0:
        raise ValueError("epochs, batch_size and lr must be positive")
    model_device = _resolve_device(device)
    model = model.to(model_device)
    pin = bool(model_device.type == "cuda") if pin_memory is None else bool(pin_memory)
    loader = _make_tensor_loader(masses, positions, labels, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers, pin_memory=pin)
    val_loader = None
    if val_masses is not None and val_positions is not None:
        val_loader = _make_tensor_loader(val_masses, val_positions, val_labels, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers, pin_memory=pin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = _make_grad_scaler(model_device, use_amp)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_raster_loss": [],
        "train_latent_mean_std": [],
        "train_latent_class_accuracy": [],
        "val_loss": [],
        "val_raster_loss": [],
        "val_latent_mean_std": [],
        "val_latent_class_accuracy": [],
    }
    best_state: Optional[dict[str, Tensor]] = None
    best_metric = float("inf")
    stale = 0

    def _run(loader_obj: DataLoader, *, train: bool, epoch_label: str) -> dict[str, float]:
        model.train(train)
        totals = {"loss": 0.0, "raster_loss": 0.0, "latent_mean_std": 0.0, "latent_class_accuracy": 0.0}
        count = 0
        acc_count = 0
        iterator = _optional_tqdm(loader_obj, enabled=show_progress, desc=epoch_label, leave=False)
        for batch_masses, batch_positions, batch_labels in iterator:
            batch_masses = batch_masses.to(model_device)
            batch_positions = batch_positions.to(model_device)
            batch_labels = batch_labels.to(model_device)
            with _amp_autocast_context(model_device, use_amp):
                loss_value, metrics = latent_shape_autoencoder_loss(
                    model,
                    batch_masses,
                    batch_positions,
                    batch_labels,
                    raster_loss=raster_loss,
                    positive_weight=positive_weight,
                    dice_weight=dice_weight,
                    blur_steps=blur_steps,
                    latent_variance_weight=latent_variance_weight,
                    latent_covariance_weight=latent_covariance_weight,
                    latent_variance_target=latent_variance_target,
                    latent_classification_weight=latent_classification_weight,
                )
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_value).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
            bsz = int(batch_masses.shape[0])
            count += bsz
            totals["loss"] += float(metrics["loss"]) * bsz
            totals["raster_loss"] += float(metrics["raster_loss"]) * bsz
            totals["latent_mean_std"] += float(metrics["latent_mean_std"]) * bsz
            if math.isfinite(float(metrics["latent_class_accuracy"])):
                totals["latent_class_accuracy"] += float(metrics["latent_class_accuracy"]) * bsz
                acc_count += bsz
        return {
            "loss": totals["loss"] / max(count, 1),
            "raster_loss": totals["raster_loss"] / max(count, 1),
            "latent_mean_std": totals["latent_mean_std"] / max(count, 1),
            "latent_class_accuracy": totals["latent_class_accuracy"] / max(acc_count, 1),
        }

    epoch_iter = _optional_tqdm(range(int(epochs)), enabled=show_progress, desc=progress_desc)
    for epoch in epoch_iter:
        train_metrics = _run(loader, train=True, epoch_label=f"{progress_desc} train {epoch + 1}/{epochs}")
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = _run(val_loader, train=False, epoch_label=f"{progress_desc} val {epoch + 1}/{epochs}")
        else:
            val_metrics = {"loss": float("nan"), "raster_loss": float("nan"), "latent_mean_std": float("nan"), "latent_class_accuracy": float("nan")}
        for key in ("loss", "raster_loss", "latent_mean_std", "latent_class_accuracy"):
            history[f"train_{key}"].append(float(train_metrics[key]))
            history[f"val_{key}"].append(float(val_metrics[key]))
        metric = val_metrics["loss"] if np.isfinite(val_metrics["loss"]) else train_metrics["loss"]
        if metric < best_metric - 1e-6:
            best_metric = float(metric)
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if verbose and (epoch == 0 or epoch + 1 == epochs or (epoch + 1) % max(1, epochs // 5) == 0):
            print(
                f"[latent-shape-ae] epoch {epoch + 1:04d}/{epochs}: "
                f"train={train_metrics['loss']:.6g} val={val_metrics['loss']:.6g} "
                f"z_std={train_metrics['latent_mean_std']:.3f} cls={train_metrics['latent_class_accuracy']:.3f}"
            )
        if show_progress and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(loss=f"{train_metrics['loss']:.3g}", val=f"{val_metrics['loss']:.3g}", zstd=f"{train_metrics['latent_mean_std']:.2f}")
        if early_stopping_patience is not None and stale >= int(early_stopping_patience):
            if verbose:
                print(f"[latent-shape-ae] early stopping at epoch {epoch + 1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_latent_shape_autoencoder(
    model: LatentShapeAutoencoder,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    batch_size: int = 128,
    device: Optional[str | torch.device] = None,
) -> dict[str, float]:
    loader = _make_tensor_loader(masses, positions, labels, batch_size=batch_size, shuffle=False)
    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    totals = {"loss": 0.0, "raster_loss": 0.0, "latent_mean_std": 0.0, "latent_class_accuracy": 0.0}
    count = 0
    acc_count = 0
    latents: list[np.ndarray] = []
    for batch_masses, batch_positions, batch_labels in loader:
        batch_masses = batch_masses.to(model_device)
        batch_positions = batch_positions.to(model_device)
        batch_labels = batch_labels.to(model_device)
        loss_value, metrics = latent_shape_autoencoder_loss(model, batch_masses, batch_positions, batch_labels)
        z = model.encode(batch_masses, batch_positions)
        latents.append(z.detach().cpu().numpy().astype(np.float64))
        bsz = int(batch_masses.shape[0])
        count += bsz
        totals["loss"] += float(metrics["loss"]) * bsz
        totals["raster_loss"] += float(metrics["raster_loss"]) * bsz
        totals["latent_mean_std"] += float(metrics["latent_mean_std"]) * bsz
        if math.isfinite(float(metrics["latent_class_accuracy"])):
            totals["latent_class_accuracy"] += float(metrics["latent_class_accuracy"]) * bsz
            acc_count += bsz
    z_all = np.concatenate(latents, axis=0) if latents else np.empty((0, model.latent_dim), dtype=np.float64)
    stats = latent_collapse_diagnostics(z_all) if len(z_all) else {}
    result = {
        "loss": totals["loss"] / max(count, 1),
        "raster_loss": totals["raster_loss"] / max(count, 1),
        "latent_mean_std": totals["latent_mean_std"] / max(count, 1),
        "latent_class_accuracy": totals["latent_class_accuracy"] / max(acc_count, 1),
    }
    result.update({f"latent_{k}": float(v) for k, v in stats.items() if isinstance(v, (int, float, np.floating))})
    if was_training:
        model.train()
    return result


def initialize_score_model_from_latent_autoencoder(
    score_model: TargetConditionedScoreModel,
    autoencoder: LatentShapeAutoencoder,
    *,
    copy_encoder: bool = True,
    copy_raster_decoder: bool = True,
    copy_label_head: bool = True,
) -> dict[str, int]:
    """Copy compatible latent modules from a pretrained shape autoencoder."""
    copied = 0
    if copy_encoder:
        report = copy_matching_state_dict(score_model.target_encoder, autoencoder.encoder)
        copied += int(report["copied"])
    if copy_raster_decoder and getattr(score_model, "latent_raster_decoder", None) is not None:
        report = copy_matching_state_dict(score_model.latent_raster_decoder, autoencoder.decoder)
        copied += int(report["copied"])
    if copy_label_head:
        report = copy_matching_state_dict(score_model.latent_label_head, autoencoder.label_head)
        copied += int(report["copied"])
    return {"copied": copied}


def latent_vicreg_regularization(
    latents: Tensor,
    *,
    variance_target: float = 1.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.05,
    eps: float = 1e-4,
) -> tuple[Tensor, dict[str, float]]:
    """VICReg-style anti-collapse regularizer for target latents.

    The variance term penalizes dimensions whose batch standard deviation falls
    below ``variance_target``.  The covariance term discourages redundant latent
    coordinates.  This is a practical diagnostic/regularizer, not part of the
    Wasserstein h-transform itself.
    """
    if latents.ndim != 2:
        raise ValueError("latents must have shape (B, D)")
    if variance_target <= 0.0 or variance_weight < 0.0 or covariance_weight < 0.0 or eps <= 0.0:
        raise ValueError("invalid VICReg parameters")
    if latents.shape[0] <= 1:
        zero = latents.new_tensor(0.0)
        return zero, {"latent_var_loss": 0.0, "latent_cov_loss": 0.0, "latent_mean_std": 0.0}
    centered = latents - torch.mean(latents, dim=0, keepdim=True)
    std = torch.sqrt(torch.var(latents, dim=0, unbiased=False) + float(eps))
    var_loss = torch.mean(F.relu(float(variance_target) - std))
    cov = (centered.T @ centered) / max(latents.shape[0] - 1, 1)
    diag = torch.diag(cov)
    cov_offdiag = cov - torch.diag(diag)
    cov_loss = torch.sum(cov_offdiag.square()) / max(latents.shape[1] * (latents.shape[1] - 1), 1)
    loss_value = float(variance_weight) * var_loss + float(covariance_weight) * cov_loss
    return loss_value, {
        "latent_var_loss": float(var_loss.detach().item()),
        "latent_cov_loss": float(cov_loss.detach().item()),
        "latent_mean_std": float(torch.mean(std).detach().item()),
    }


def latent_collapse_diagnostics(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    max_samples: int = 512,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Report simple latent-collapse diagnostics for notebook gating."""
    z = np.asarray(latents, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError("latents must have shape (N, D)")
    rng = np.random.default_rng() if rng is None else rng
    if z.shape[0] > int(max_samples):
        idx = rng.choice(z.shape[0], size=int(max_samples), replace=False)
        z_used = z[idx]
        labels_used = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)[idx]
    else:
        z_used = z
        labels_used = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
    per_dim_std = np.std(z_used, axis=0)
    centered = z_used - np.mean(z_used, axis=0, keepdims=True)
    if len(z_used) > 1:
        gram = np.sum(centered * centered, axis=1)[:, None] + np.sum(centered * centered, axis=1)[None, :] - 2.0 * centered @ centered.T
        gram = np.maximum(gram, 0.0)
        triu = np.triu_indices(len(z_used), k=1)
        pairwise = np.sqrt(gram[triu]) if len(triu[0]) else np.asarray([0.0])
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        var = singular_values**2
        explained = var / max(float(np.sum(var)), 1e-12)
        cumulative = np.cumsum(explained)
        rank90 = int(np.searchsorted(cumulative, 0.90) + 1) if len(cumulative) else 0
    else:
        pairwise = np.asarray([0.0])
        explained = np.asarray([])
        rank90 = 0
    result: dict[str, Any] = {
        "num_samples": int(z_used.shape[0]),
        "latent_dim": int(z_used.shape[1]),
        "mean_per_dim_std": float(np.mean(per_dim_std)),
        "median_per_dim_std": float(np.median(per_dim_std)),
        "min_per_dim_std": float(np.min(per_dim_std)),
        "max_per_dim_std": float(np.max(per_dim_std)),
        "mean_pairwise_distance": float(np.mean(pairwise)),
        "median_pairwise_distance": float(np.median(pairwise)),
        "rank90": int(rank90),
        "top5_explained_variance": explained[:5].astype(np.float64),
        "collapsed_by_pairwise_distance": bool(float(np.mean(pairwise)) < 1e-3),
        "collapsed_by_std": bool(float(np.mean(per_dim_std)) < 1e-3),
    }
    if labels_used is not None:
        per_label: dict[int, dict[str, float | int]] = {}
        for label in np.unique(labels_used):
            mask = labels_used == int(label)
            per_label[int(label)] = {
                "count": int(np.sum(mask)),
                "mean_per_dim_std": float(np.mean(np.std(z_used[mask], axis=0))) if np.sum(mask) > 1 else 0.0,
            }
        result["per_label"] = per_label
    return result


# ---------------------------------------------------------------------------
# Denoising score matching
# ---------------------------------------------------------------------------


def perturb_target_conditioned_positions(
    masses: Tensor,
    clean_positions: Tensor,
    tau: Tensor | float,
    *,
    projection: str = "none",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Apply theory-consistent forward noising and return stable targets.

    Returns ``(noisy_positions, target_scaled_score, target_score)`` where
    ``target_scaled_score = -eps`` and
    ``target_score = (clean_positions - noisy_positions) / (2 tau)``.
    """
    _validate_probability_masses(masses)
    _validate_positions_tensor(clean_positions, masses, name="clean_positions")
    tau_tensor = _prepare_tau_tensor(tau, masses.shape[0], device=clean_positions.device, dtype=clean_positions.dtype)
    eps = torch.randn_like(clean_positions)
    sigma = torch.sqrt((2.0 * tau_tensor[:, None, None]) / masses.unsqueeze(-1))
    noisy = clean_positions + sigma * eps
    if projection != "none":
        noisy = project_positions(noisy, mode=projection)
    target_scaled_score = -eps
    target_score = (clean_positions - noisy) / (2.0 * tau_tensor[:, None, None])
    return noisy, target_scaled_score, target_score


def empirical_gaussian_mixture_physical_score(
    query_positions: Tensor,
    target_positions: Tensor,
    sigma: Tensor | float,
    *,
    target_masses: Optional[Tensor] = None,
    chunk_size: Optional[int] = 256,
    return_posterior_mean: bool = False,
    eps: float = 1e-12,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Exact empirical Gaussian-mixture physical score for a target contour.

    For target points ``x_j`` this computes
    ``grad_x log sum_j w_j N(x; x_j, sigma^2 I)``.  This is the oracle field
    used in ShapeGF-style diagnostics; multiplying a Wasserstein score by the
    particle mass gives the corresponding physical score.
    """
    if query_positions.ndim != 3 or query_positions.shape[2] != 2:
        raise ValueError("query_positions must have shape (B,K,2)")
    if target_positions.ndim != 3 or target_positions.shape[0] != query_positions.shape[0] or target_positions.shape[2] != 2:
        raise ValueError("target_positions must have shape (B,M,2) with matching batch size")
    batch_size, num_queries, _ = query_positions.shape
    device, dtype = query_positions.device, query_positions.dtype
    target_positions = target_positions.to(device=device, dtype=dtype)
    if isinstance(sigma, Tensor):
        sigma_tensor = sigma.to(device=device, dtype=dtype)
    else:
        sigma_tensor = torch.tensor(float(sigma), device=device, dtype=dtype)
    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.reshape(1, 1).expand(batch_size, num_queries)
    elif sigma_tensor.ndim == 1:
        if sigma_tensor.shape[0] != batch_size:
            raise ValueError("sigma with shape (B,) must match batch size")
        sigma_tensor = sigma_tensor[:, None].expand(batch_size, num_queries)
    elif sigma_tensor.ndim == 2:
        if sigma_tensor.shape != (batch_size, num_queries):
            raise ValueError("sigma with shape (B,K) must match query positions")
    else:
        raise ValueError("sigma must be scalar, (B,), or (B,K)")
    sigma_tensor = sigma_tensor.clamp_min(float(eps))
    if target_masses is None:
        log_target_masses = None
    else:
        if target_masses.ndim != 2 or target_masses.shape != target_positions.shape[:2]:
            raise ValueError("target_masses must have shape (B,M)")
        log_target_masses = torch.log(target_masses.to(device=device, dtype=dtype).clamp_min(float(eps)))

    chunk = num_queries if chunk_size is None else int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive")
    scores: list[Tensor] = []
    means: list[Tensor] = []
    for start in range(0, num_queries, chunk):
        stop = min(start + chunk, num_queries)
        q = query_positions[:, start:stop]
        sigma_chunk = sigma_tensor[:, start:stop]
        d2 = torch.cdist(q, target_positions).square()
        logits = -d2 / (2.0 * sigma_chunk.square().unsqueeze(-1))
        if log_target_masses is not None:
            logits = logits + log_target_masses[:, None, :]
        weights = torch.softmax(logits, dim=-1)
        posterior_mean = weights @ target_positions
        physical_score = (posterior_mean - q) / sigma_chunk.square().unsqueeze(-1)
        scores.append(physical_score)
        if return_posterior_mean:
            means.append(posterior_mean)
    score = torch.cat(scores, dim=1)
    if return_posterior_mean:
        return score, torch.cat(means, dim=1)
    return score


def empirical_mixture_scaled_score_target(
    query_positions: Tensor,
    target_positions: Tensor,
    masses: Tensor,
    tau: Tensor | float,
    *,
    target_masses: Optional[Tensor] = None,
    chunk_size: Optional[int] = 256,
    target_norm_clip: Optional[float] = None,
    return_physical_score: bool = False,
    component_balance: bool = False,
    component_balance_image_size: int = 64,
    component_balance_dilation: int = 1,
    component_balance_min_pixels: int = 4,
) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    r"""Return scaled Wasserstein-score target from the empirical mixture oracle."""
    _validate_probability_masses(masses)
    _validate_positions_tensor(query_positions, masses, name="query_positions")
    if target_positions.ndim != 3 or target_positions.shape[0] != query_positions.shape[0] or target_positions.shape[2] != 2:
        raise ValueError("target_positions must have shape (B,M,2) and match query batch size")
    tau_tensor = _prepare_tau_tensor(tau, masses.shape[0], device=query_positions.device, dtype=query_positions.dtype)
    sigma = torch.sqrt((2.0 * tau_tensor[:, None]) / masses).clamp_min(1e-12)
    target_masses_resolved = target_masses
    if component_balance:
        target_masses_resolved = _component_balanced_target_masses_torch(
            target_positions,
            image_size=component_balance_image_size,
            contour_dilation=component_balance_dilation,
            min_component_pixels=component_balance_min_pixels,
        )
    physical_score, posterior_mean = empirical_gaussian_mixture_physical_score(
        query_positions,
        target_positions,
        sigma,
        target_masses=target_masses_resolved,
        chunk_size=chunk_size,
        return_posterior_mean=True,
    )
    scaled_target = physical_score * sigma.unsqueeze(-1)
    if target_norm_clip is not None and target_norm_clip > 0.0:
        norm = torch.linalg.norm(scaled_target, dim=-1, keepdim=True).clamp_min(1e-12)
        scaled_target = scaled_target * torch.clamp(float(target_norm_clip) / norm, max=1.0)
    if return_physical_score:
        return scaled_target, physical_score, posterior_mean
    return scaled_target


def empirical_gaussian_mixture_scaled_score(
    masses: Tensor,
    query_positions: Tensor,
    target_positions: Tensor,
    tau: Tensor | float,
    *,
    target_masses: Optional[Tensor] = None,
    chunk_size: Optional[int] = None,
    target_norm_clip: Optional[float] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Return scaled score, Wasserstein score and posterior mean for the oracle field.

    The empirical Gaussian mixture naturally gives the physical score
    ``grad_x log q_sigma``.  The finite-particle Wasserstein/fiber score is
    ``D_i log q = grad_x log q / s_i``.
    """
    scaled, physical, posterior_mean = empirical_mixture_scaled_score_target(
        query_positions,
        target_positions,
        masses,
        tau,
        target_masses=target_masses,
        chunk_size=chunk_size,
        return_physical_score=True,
        target_norm_clip=target_norm_clip,
    )
    wasserstein_score = physical / masses.unsqueeze(-1).clamp_min(1e-12)
    return scaled, wasserstein_score, posterior_mean


empirical_gaussian_mixture_score_target = empirical_mixture_scaled_score_target


def _sample_direct_mixture_query_positions(
    clean_positions: Tensor,
    masses: Tensor,
    tau: Tensor,
    *,
    query_modes: Sequence[str],
    coordinate_range: tuple[float, float] = (0.0, 1.0),
    center_std: float = 0.35,
    projection: str = "none",
    component_balance_image_size: int = 64,
    component_balance_dilation: int = 1,
    component_balance_min_pixels: int = 4,
) -> Tensor:
    if not query_modes:
        raise ValueError("query_modes must be non-empty")
    allowed = {
        "noised_target",
        "uniform",
        "center_gaussian",
        "fixed_center",
        "component_noised_target",
        "component_center_gaussian",
        "hole_region_uniform",
        "corner_noised_target",
        "corner_region_uniform",
    }
    modes = tuple(str(mode) for mode in query_modes)
    unknown = set(modes) - allowed
    if unknown:
        raise ValueError(f"unknown direct mixture query modes: {sorted(unknown)}")
    low, high = float(coordinate_range[0]), float(coordinate_range[1])
    if not high > low:
        raise ValueError("coordinate_range must satisfy high > low")
    batch_size, num_points, _ = clean_positions.shape
    sigma = torch.sqrt((2.0 * tau[:, None, None]) / masses.unsqueeze(-1))
    noised = clean_positions + sigma * torch.randn_like(clean_positions)
    uniform = low + (high - low) * torch.rand_like(clean_positions)
    center = 0.5 * (low + high)
    centered = center + float(center_std) * (high - low) * torch.randn_like(clean_positions)
    fixed = torch.full_like(clean_positions, center)
    component_points = None
    corner_points = None
    hole_uniform = None
    if "component_noised_target" in modes or "component_center_gaussian" in modes:
        component_points = _sample_component_balanced_points_torch(
            clean_positions,
            image_size=component_balance_image_size,
            contour_dilation=component_balance_dilation,
            min_component_pixels=component_balance_min_pixels,
        )
    if "corner_noised_target" in modes or "corner_region_uniform" in modes:
        corner_points = _sample_corner_points_torch(
            clean_positions,
            image_size=component_balance_image_size,
            contour_dilation=component_balance_dilation,
            corner_quantile=0.70,
        )
    if "hole_region_uniform" in modes:
        hole_uniform = _sample_hole_uniform_points_torch(
            clean_positions,
            image_size=component_balance_image_size,
            hole_dilation=component_balance_dilation,
            fallback_std=center_std,
        )
    choices = torch.randint(0, len(modes), (batch_size,), device=clean_positions.device)
    out = torch.empty_like(clean_positions)
    for mode_index, mode in enumerate(modes):
        mask = choices == mode_index
        if not bool(torch.any(mask)):
            continue
        if mode == "noised_target":
            out[mask] = noised[mask]
        elif mode == "uniform":
            out[mask] = uniform[mask]
        elif mode == "center_gaussian":
            out[mask] = centered[mask]
        elif mode == "component_noised_target":
            if component_points is None:
                raise RuntimeError("component_points unexpectedly missing")
            out[mask] = component_points[mask] + sigma[mask] * torch.randn_like(component_points[mask])
        elif mode == "component_center_gaussian":
            if component_points is None:
                raise RuntimeError("component_points unexpectedly missing")
            out[mask] = component_points[mask] + float(center_std) * sigma[mask] * torch.randn_like(component_points[mask])
        elif mode == "hole_region_uniform":
            if hole_uniform is None:
                raise RuntimeError("hole_uniform unexpectedly missing")
            out[mask] = hole_uniform[mask]
        elif mode == "corner_noised_target":
            if corner_points is None:
                raise RuntimeError("corner_points unexpectedly missing")
            out[mask] = corner_points[mask] + sigma[mask] * torch.randn_like(corner_points[mask])
        elif mode == "corner_region_uniform":
            if corner_points is None:
                raise RuntimeError("corner_points unexpectedly missing")
            out[mask] = corner_points[mask] + 0.5 * sigma[mask] * torch.randn_like(corner_points[mask])
        else:
            out[mask] = fixed[mask]
    if projection != "none":
        out = project_positions(out, mode=projection)
    return out



def sample_direct_mixture_queries(
    clean_positions: Tensor,
    masses: Tensor,
    tau: Tensor,
    *,
    query_modes: Sequence[str] = ("noised_target", "uniform", "center_gaussian"),
    coordinate_range: tuple[float, float] = (0.0, 1.0),
    center_std: float = 0.35,
    projection: str = "none",
    component_balance_image_size: int = 64,
    component_balance_dilation: int = 1,
    component_balance_min_pixels: int = 4,
) -> Tensor:
    """Public wrapper for generation-like direct-mixture query sampling."""
    return _sample_direct_mixture_query_positions(
        clean_positions, masses, tau, query_modes=query_modes,
        coordinate_range=coordinate_range, center_std=center_std, projection=projection,
        component_balance_image_size=component_balance_image_size,
        component_balance_dilation=component_balance_dilation,
        component_balance_min_pixels=component_balance_min_pixels,
    )


@torch.no_grad()
def sample_oracle_replay_queries(
    target_positions: Tensor,
    masses: Tensor,
    tau_levels: Sequence[float] | np.ndarray | Tensor,
    *,
    target_masses: Optional[Tensor] = None,
    steps_per_level: int = 2,
    max_levels: Optional[int] = None,
    initial_query_modes: Sequence[str] = ("uniform", "center_gaussian", "fixed_center"),
    coordinate_range: tuple[float, float] = (0.0, 1.0),
    center_std: float = 0.35,
    projection: str = "none",
    langevin_alpha: float = 5e-5,
    diffusion_temperature: float = 1.0,
    score_scale: float = 1.0,
    mixture_chunk_size: Optional[int] = 256,
) -> tuple[Tensor, Tensor, int]:
    r"""Sample query states from successful empirical-mixture oracle trajectories.

    This supplies training points from the actual annealed-sampler path
    (uniform/blob -> coarse contour -> fine contour), rather than only from
    independent DSM perturbations around the target contour.
    """
    _validate_probability_masses(masses)
    if target_positions.ndim != 3 or target_positions.shape[0] != masses.shape[0] or target_positions.shape[2] != 2:
        raise ValueError("target_positions must have shape (B,M,2) and match masses batch")
    if steps_per_level <= 0:
        raise ValueError("steps_per_level must be positive")
    if langevin_alpha <= 0.0 or diffusion_temperature < 0.0 or score_scale <= 0.0:
        raise ValueError("invalid replay sampler scale/temperature parameters")
    device, dtype = target_positions.device, target_positions.dtype
    if isinstance(tau_levels, Tensor):
        levels = tau_levels.to(device=device, dtype=dtype).reshape(-1)
    else:
        levels = torch.as_tensor(np.asarray(tau_levels, dtype=np.float64).copy(), device=device, dtype=dtype).reshape(-1)
    if levels.numel() <= 0 or bool(torch.any(levels <= 0.0)):
        raise ValueError("tau_levels must be positive and non-empty")
    levels = torch.sort(levels, descending=True).values
    replay_count = int(levels.numel()) if max_levels is None else min(int(max_levels), int(levels.numel()))
    replay_count = max(replay_count, 1)
    level_index = int(torch.randint(0, replay_count, (), device=device).item())
    tau0 = levels[0].expand(masses.shape[0])
    positions = _sample_direct_mixture_query_positions(
        target_positions,
        masses,
        tau0,
        query_modes=initial_query_modes,
        coordinate_range=coordinate_range,
        center_std=center_std,
        projection=projection,
    )
    tau_min = levels[-1].clamp_min(torch.finfo(dtype).eps)
    target_masses_resolved = target_masses.to(device=device, dtype=dtype) if target_masses is not None else masses
    for current_level_index in range(level_index + 1):
        tau_level = levels[current_level_index]
        tau_batch = tau_level.expand(masses.shape[0])
        sigma = torch.sqrt((2.0 * tau_level) / masses).clamp_min(1e-12)
        sigma_min = torch.sqrt((2.0 * tau_min) / masses).clamp_min(1e-12)
        ratio = (sigma / sigma_min).unsqueeze(-1)
        for _ in range(int(steps_per_level)):
            if diffusion_temperature > 0.0:
                positions = positions + float(diffusion_temperature) * math.sqrt(float(langevin_alpha)) * ratio * torch.randn_like(positions)
                positions = project_positions(positions, mode=projection)
            _, oracle_physical, _ = empirical_mixture_scaled_score_target(
                positions,
                target_positions,
                masses,
                tau_batch,
                target_masses=target_masses_resolved,
                chunk_size=mixture_chunk_size,
                return_physical_score=True,
            )
            positions = positions + 0.5 * float(langevin_alpha) * ratio.square() * float(score_scale) * oracle_physical
            positions = project_positions(positions, mode=projection)
    tau_replay = levels[level_index].expand(masses.shape[0])
    return positions.detach(), tau_replay.detach(), level_index


def target_conditioned_score_matching_loss(
    predicted_scaled_score: Tensor,
    target_scaled_score: Tensor,
    masses: Tensor,
    tau: Tensor,
    *,
    time_weighting: str = "none",
) -> tuple[Tensor, dict[str, float]]:
    """Mass-weighted loss for the scaled-score target ``-eps``."""
    if predicted_scaled_score.shape != target_scaled_score.shape:
        raise ValueError("predicted and target scaled scores must have the same shape")
    if predicted_scaled_score.ndim != 3 or predicted_scaled_score.shape[2] != 2:
        raise ValueError("scaled scores must have shape (B, K, 2)")
    if masses.shape != predicted_scaled_score.shape[:2]:
        raise ValueError("masses must have shape (B, K)")
    if tau.shape != (predicted_scaled_score.shape[0],):
        raise ValueError("tau must have shape (B,)")
    if time_weighting not in {"none", "sigma2", "sqrt_sigma2"}:
        raise ValueError("time_weighting must be one of {'none', 'sigma2', 'sqrt_sigma2'}")

    point_sq_error = torch.sum((predicted_scaled_score - target_scaled_score).square(), dim=-1)
    sample_loss = torch.sum(masses * point_sq_error, dim=1)
    if time_weighting == "none":
        weights = torch.ones_like(tau)
    elif time_weighting == "sigma2":
        weights = 2.0 * tau
    else:
        weights = torch.sqrt(2.0 * tau)
    loss = torch.mean(weights * sample_loss)
    metrics = {
        "loss": float(loss.detach().item()),
        "sample_loss": float(torch.mean(sample_loss).detach().item()),
        "mean_tau": float(torch.mean(tau).detach().item()),
    }
    return loss, metrics


def _sample_tau_from_levels(
    batch_size: int,
    tau_levels: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    levels = torch.as_tensor(np.asarray(tau_levels, dtype=np.float64).copy(), device=device, dtype=dtype).reshape(-1)
    if levels.numel() <= 0:
        raise ValueError("tau_levels must be non-empty")
    idx = torch.randint(0, int(levels.numel()), (int(batch_size),), device=device)
    return levels[idx]


def _make_tensor_loader(
    masses: np.ndarray,
    positions: np.ndarray,
    labels: Optional[np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    masses_arr = np.asarray(masses, dtype=np.float32)
    positions_arr = np.asarray(positions, dtype=np.float32)
    if masses_arr.ndim != 2:
        raise ValueError("masses must have shape (N, K)")
    if positions_arr.shape != (*masses_arr.shape, 2):
        raise ValueError("positions must have shape (N, K, 2) and match masses")
    labels_arr = np.zeros((masses_arr.shape[0],), dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_arr.shape != (masses_arr.shape[0],):
        raise ValueError("labels must have shape (N,)")
    dataset = TensorDataset(
        torch.from_numpy(masses_arr),
        torch.from_numpy(positions_arr),
        torch.from_numpy(labels_arr),
    )
    workers = max(0, int(num_workers))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(pin_memory),
        persistent_workers=workers > 0,
    )



def _build_score_training_query_and_target(
    batch_masses: Tensor,
    batch_positions: Tensor,
    tau: Tensor,
    *,
    projection: str,
    tau_levels: Optional[Sequence[float] | np.ndarray] = None,
    direct_mixture_probability: float,
    oracle_replay_probability: float = 0.0,
    direct_query_modes: Sequence[str],
    direct_query_center_std: float,
    mixture_chunk_size: Optional[int],
    mixture_target_norm_clip: Optional[float] = None,
    component_balance_oracle: bool = False,
    component_balance_image_size: int = 64,
    component_balance_dilation: int = 1,
    component_balance_min_pixels: int = 4,
    oracle_replay_steps_per_level: int = 2,
    oracle_replay_max_levels: Optional[int] = None,
    oracle_replay_initial_modes: Sequence[str] = ("uniform", "center_gaussian", "fixed_center"),
    oracle_replay_langevin_alpha: float = 5e-5,
    oracle_replay_diffusion_temperature: float = 1.0,
    oracle_replay_score_scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, str]:
    """Return a query cloud, tau tensor and scaled-score target for one batch."""
    if direct_mixture_probability < 0.0 or oracle_replay_probability < 0.0:
        raise ValueError("training objective probabilities must be non-negative")
    if direct_mixture_probability + oracle_replay_probability > 1.0 + 1e-12:
        raise ValueError("direct_mixture_probability + oracle_replay_probability must be <= 1")
    draw = float(torch.rand((), device=batch_positions.device).item())
    if oracle_replay_probability > 0.0 and draw < float(oracle_replay_probability):
        if tau_levels is None:
            raise ValueError("tau_levels are required when oracle_replay_probability > 0")
        query, replay_tau, _ = sample_oracle_replay_queries(
            batch_positions,
            batch_masses,
            tau_levels,
            target_masses=batch_masses,
            steps_per_level=oracle_replay_steps_per_level,
            max_levels=oracle_replay_max_levels,
            initial_query_modes=oracle_replay_initial_modes,
            center_std=direct_query_center_std,
            projection=projection,
            langevin_alpha=oracle_replay_langevin_alpha,
            diffusion_temperature=oracle_replay_diffusion_temperature,
            score_scale=oracle_replay_score_scale,
            mixture_chunk_size=mixture_chunk_size,
        )
        target_scaled = empirical_mixture_scaled_score_target(
            query,
            batch_positions,
            batch_masses,
            replay_tau,
            target_masses=batch_masses,
            chunk_size=mixture_chunk_size,
            return_physical_score=False,
            target_norm_clip=mixture_target_norm_clip,
            component_balance=component_balance_oracle,
            component_balance_image_size=component_balance_image_size,
            component_balance_dilation=component_balance_dilation,
            component_balance_min_pixels=component_balance_min_pixels,
        )
        return query, target_scaled, replay_tau, "oracle_replay"

    if direct_mixture_probability > 0.0 and draw < float(oracle_replay_probability + direct_mixture_probability):
        query = _sample_direct_mixture_query_positions(
            batch_positions,
            batch_masses,
            tau,
            query_modes=direct_query_modes,
            center_std=direct_query_center_std,
            projection=projection,
            component_balance_image_size=component_balance_image_size,
            component_balance_dilation=component_balance_dilation,
            component_balance_min_pixels=component_balance_min_pixels,
        )
        target_scaled = empirical_mixture_scaled_score_target(
            query,
            batch_positions,
            batch_masses,
            tau,
            target_masses=batch_masses,
            chunk_size=mixture_chunk_size,
            return_physical_score=False,
            target_norm_clip=mixture_target_norm_clip,
            component_balance=component_balance_oracle,
            component_balance_image_size=component_balance_image_size,
            component_balance_dilation=component_balance_dilation,
            component_balance_min_pixels=component_balance_min_pixels,
        )
        return query, target_scaled, tau, "direct_mixture"

    noisy, target_scaled, _ = perturb_target_conditioned_positions(
        batch_masses,
        batch_positions,
        tau,
        projection=projection,
    )
    return noisy, target_scaled, tau, "paired_dsm"


@torch.no_grad()
def evaluate_model_vs_mixture_oracle(
    model: TargetConditionedScoreModel,
    masses: np.ndarray,
    target_positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    tau_levels: Sequence[float] | np.ndarray,
    query_modes: Sequence[str] = ("noised_target", "uniform", "center_gaussian"),
    max_samples: int = 128,
    batch_size: int = 32,
    mixture_chunk_size: Optional[int] = 256,
    projection: str = "none",
    device: Optional[str | torch.device] = None,
) -> list[dict[str, float | str]]:
    r"""Compare learned physical score ``s_i S_i`` with exact mixture score."""
    if batch_size <= 0 or max_samples <= 0:
        raise ValueError("batch_size and max_samples must be positive")
    tau_levels_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if tau_levels_arr.size == 0 or not np.all(np.isfinite(tau_levels_arr)) or np.any(tau_levels_arr <= 0.0):
        raise ValueError("tau_levels must be positive and finite")
    masses_arr = np.asarray(masses, dtype=np.float32)
    positions_arr = np.asarray(target_positions, dtype=np.float32)
    if masses_arr.ndim != 2 or positions_arr.shape != (*masses_arr.shape, 2):
        raise ValueError("masses and target_positions must have shapes (N,K), (N,K,2)")
    n = min(int(max_samples), int(masses_arr.shape[0]))
    labels_arr = np.zeros((masses_arr.shape[0],), dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_arr.shape != (masses_arr.shape[0],):
        raise ValueError("labels must have shape (N,)")

    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    rows: list[dict[str, float | str]] = []
    for tau_value in tau_levels_arr:
        for query_mode in query_modes:
            total_sq = total_oracle_sq = total_model_sq = total_dot = total_weight = 0.0
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                batch_masses = torch.from_numpy(masses_arr[start:stop]).to(model_device)
                batch_target = torch.from_numpy(positions_arr[start:stop]).to(model_device)
                batch_labels = torch.from_numpy(labels_arr[start:stop]).to(model_device)
                tau = torch.full((stop - start,), float(tau_value), device=model_device, dtype=batch_target.dtype)
                query = _sample_direct_mixture_query_positions(
                    batch_target,
                    batch_masses,
                    tau,
                    query_modes=(str(query_mode),),
                    projection=projection,
                )
                _, oracle_physical, _ = empirical_mixture_scaled_score_target(
                    query,
                    batch_target,
                    batch_masses,
                    tau,
                    target_masses=batch_masses,
                    chunk_size=mixture_chunk_size,
                    return_physical_score=True,
                )
                model_wasserstein = model(
                    batch_masses,
                    query,
                    tau,
                    target_positions=batch_target,
                    target_masses=batch_masses,
                    labels=batch_labels,
                )
                model_physical = batch_masses.unsqueeze(-1) * model_wasserstein
                weights = batch_masses.unsqueeze(-1)
                diff = model_physical - oracle_physical
                total_sq += float(torch.sum(weights * diff.square()).item())
                total_oracle_sq += float(torch.sum(weights * oracle_physical.square()).item())
                total_model_sq += float(torch.sum(weights * model_physical.square()).item())
                total_dot += float(torch.sum(weights * model_physical * oracle_physical).item())
                total_weight += float(torch.sum(weights).item()) * 2.0
            rows.append(
                {
                    "tau": float(tau_value),
                    "sigma_approx": float(math.sqrt(2.0 * float(tau_value) * masses_arr.shape[1])),
                    "query_mode": str(query_mode),
                    "physical_score_rmse": float(math.sqrt(total_sq / max(total_weight, 1e-12))),
                    "relative_rmse": float(math.sqrt(total_sq / max(total_oracle_sq, 1e-12))),
                    "cosine": float(total_dot / max(math.sqrt(total_model_sq * total_oracle_sq), 1e-12)),
                }
            )
    if was_training:
        model.train()
    return rows


@torch.no_grad()
def fit_score_calibration_against_mixture_oracle(
    model: TargetConditionedScoreModel,
    masses: np.ndarray,
    target_positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    tau_levels: Sequence[float] | np.ndarray,
    query_modes: Sequence[str] = ("uniform", "center_gaussian", "noised_target"),
    max_samples: int = 128,
    batch_size: int = 32,
    mixture_chunk_size: Optional[int] = 256,
    projection: str = "none",
    clip_quantile: Optional[float] = 0.99,
    min_scale: float = 0.0,
    max_scale: float = 10.0,
    device: Optional[str | torch.device] = None,
) -> ScoreCalibration:
    r"""Fit per-tau scalar calibration against the empirical-mixture oracle.

    The sampler uses the physical score ``s_i S_theta``.  This function fits
    one scalar per tau level by least squares against the exact empirical
    mixture physical score and optionally stores oracle norm percentiles for
    score clipping.
    """
    if batch_size <= 0 or max_samples <= 0:
        raise ValueError("batch_size and max_samples must be positive")
    if clip_quantile is not None and not (0.0 < float(clip_quantile) <= 1.0):
        raise ValueError("clip_quantile must be in (0, 1]")
    tau_levels_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if tau_levels_arr.size == 0 or not np.all(np.isfinite(tau_levels_arr)) or np.any(tau_levels_arr <= 0.0):
        raise ValueError("tau_levels must be positive and finite")
    tau_levels_arr = np.sort(tau_levels_arr)[::-1].astype(np.float64, copy=False)
    masses_arr = np.asarray(masses, dtype=np.float32)
    positions_arr = np.asarray(target_positions, dtype=np.float32)
    if masses_arr.ndim != 2 or positions_arr.shape != (*masses_arr.shape, 2):
        raise ValueError("masses and target_positions must have shapes (N,K), (N,K,2)")
    labels_arr = np.zeros((masses_arr.shape[0],), dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_arr.shape != (masses_arr.shape[0],):
        raise ValueError("labels must have shape (N,)")
    n = min(int(max_samples), int(masses_arr.shape[0]))
    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()

    scales: list[float] = []
    clips: list[float] = []
    mode_tuple = tuple(str(mode) for mode in query_modes)
    for tau_value in tau_levels_arr:
        numerator = denominator = 0.0
        norm_values: list[np.ndarray] = []
        for query_mode in mode_tuple:
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                batch_masses = torch.from_numpy(masses_arr[start:stop]).to(model_device)
                batch_target = torch.from_numpy(positions_arr[start:stop]).to(model_device)
                batch_labels = torch.from_numpy(labels_arr[start:stop]).to(model_device)
                tau = torch.full((stop - start,), float(tau_value), device=model_device, dtype=batch_target.dtype)
                query = _sample_direct_mixture_query_positions(
                    batch_target,
                    batch_masses,
                    tau,
                    query_modes=(query_mode,),
                    projection=projection,
                )
                _, oracle_physical, _ = empirical_mixture_scaled_score_target(
                    query,
                    batch_target,
                    batch_masses,
                    tau,
                    target_masses=batch_masses,
                    chunk_size=mixture_chunk_size,
                    return_physical_score=True,
                )
                model_wasserstein = model(
                    batch_masses,
                    query,
                    tau,
                    target_positions=batch_target,
                    target_masses=batch_masses,
                    labels=batch_labels,
                )
                model_physical = batch_masses.unsqueeze(-1) * model_wasserstein
                weights = batch_masses.unsqueeze(-1)
                numerator += float(torch.sum(weights * model_physical * oracle_physical).item())
                denominator += float(torch.sum(weights * model_physical.square()).item())
                if clip_quantile is not None:
                    norm_values.append(torch.linalg.norm(oracle_physical, dim=-1).detach().cpu().numpy().reshape(-1))
        scale = numerator / max(denominator, 1e-12)
        scale = float(np.clip(scale, float(min_scale), float(max_scale)))
        scales.append(scale if math.isfinite(scale) else 1.0)
        if clip_quantile is not None and norm_values:
            clips.append(float(np.quantile(np.concatenate(norm_values), float(clip_quantile))))
        else:
            clips.append(float("nan"))
    if was_training:
        model.train()
    clip_arr: Optional[np.ndarray]
    if clip_quantile is None:
        clip_arr = None
    else:
        clip_arr = np.asarray(clips, dtype=np.float64)
    return ScoreCalibration(
        tau_levels=tau_levels_arr.astype(np.float64),
        physical_score_scale=np.asarray(scales, dtype=np.float64),
        physical_norm_clip=clip_arr,
        metadata={
            "query_modes": mode_tuple,
            "max_samples": int(n),
            "clip_quantile": None if clip_quantile is None else float(clip_quantile),
            "projection": projection,
        },
    )


@torch.no_grad()
def evaluate_target_conditioned_score_model(
    model: TargetConditionedScoreModel,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    tau_levels: Sequence[float] | np.ndarray,
    batch_size: int = 128,
    projection: str = "none",
    time_weighting: str = "none",
    direct_mixture_probability: float = 0.0,
    oracle_replay_probability: float = 0.0,
    direct_query_modes: Sequence[str] = ("noised_target", "uniform", "center_gaussian"),
    direct_query_center_std: float = 0.35,
    mixture_chunk_size: Optional[int] = 256,
    mixture_target_norm_clip: Optional[float] = None,
    component_balance_oracle: bool = False,
    component_balance_image_size: int = 64,
    component_balance_dilation: int = 1,
    component_balance_min_pixels: int = 4,
    oracle_replay_steps_per_level: int = 2,
    oracle_replay_max_levels: Optional[int] = None,
    oracle_replay_initial_modes: Sequence[str] = ("uniform", "center_gaussian", "fixed_center"),
    oracle_replay_langevin_alpha: float = 5e-5,
    oracle_replay_diffusion_temperature: float = 1.0,
    oracle_replay_score_scale: float = 1.0,
    device: Optional[str | torch.device] = None,
) -> dict[str, float]:
    """Estimate validation scaled-score loss for paired or mixed training."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if direct_mixture_probability < 0.0 or oracle_replay_probability < 0.0:
        raise ValueError("training objective probabilities must be non-negative")
    if direct_mixture_probability + oracle_replay_probability > 1.0 + 1e-12:
        raise ValueError("direct_mixture_probability + oracle_replay_probability must be <= 1")
    tau_levels_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if tau_levels_arr.size == 0:
        raise ValueError("tau_levels must be non-empty")
    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    loader = _make_tensor_loader(masses, positions, labels, batch_size=batch_size, shuffle=False)

    total_loss = total_sample_loss = total_zero_loss = 0.0
    total_items = direct_items = replay_items = 0
    tau_means: list[float] = []
    for batch_masses, batch_positions, batch_labels in loader:
        batch_masses = batch_masses.to(model_device)
        batch_positions = batch_positions.to(model_device)
        batch_labels = batch_labels.to(model_device)
        tau = _sample_tau_from_levels(int(batch_masses.shape[0]), tau_levels_arr, device=model_device, dtype=batch_positions.dtype)
        query, target_scaled, tau_used, target_kind = _build_score_training_query_and_target(
            batch_masses,
            batch_positions,
            tau,
            projection=projection,
            tau_levels=tau_levels_arr,
            direct_mixture_probability=direct_mixture_probability,
            oracle_replay_probability=oracle_replay_probability,
            direct_query_modes=direct_query_modes,
            direct_query_center_std=direct_query_center_std,
            mixture_chunk_size=mixture_chunk_size,
            mixture_target_norm_clip=mixture_target_norm_clip,
            oracle_replay_steps_per_level=oracle_replay_steps_per_level,
            oracle_replay_max_levels=oracle_replay_max_levels,
            oracle_replay_initial_modes=oracle_replay_initial_modes,
            oracle_replay_langevin_alpha=oracle_replay_langevin_alpha,
            oracle_replay_diffusion_temperature=oracle_replay_diffusion_temperature,
            oracle_replay_score_scale=oracle_replay_score_scale,
        )
        pred_scaled = model.predict_scaled_score(
            batch_masses,
            query,
            tau_used,
            target_positions=batch_positions,
            target_masses=batch_masses,
            labels=batch_labels,
        )
        loss, metrics = target_conditioned_score_matching_loss(pred_scaled, target_scaled, batch_masses, tau_used, time_weighting=time_weighting)
        zero_loss, _ = target_conditioned_score_matching_loss(torch.zeros_like(target_scaled), target_scaled, batch_masses, tau_used, time_weighting=time_weighting)
        bsz = int(batch_masses.shape[0])
        total_loss += float(loss.item()) * bsz
        total_sample_loss += float(metrics["sample_loss"]) * bsz
        total_zero_loss += float(zero_loss.item()) * bsz
        total_items += bsz
        direct_items += bsz if target_kind == "direct_mixture" else 0
        replay_items += bsz if target_kind == "oracle_replay" else 0
        tau_means.append(float(metrics["mean_tau"]))
    if was_training:
        model.train()
    return {
        "loss": total_loss / max(total_items, 1),
        "sample_loss": total_sample_loss / max(total_items, 1),
        "zero_loss": total_zero_loss / max(total_items, 1),
        "loss_ratio_vs_zero": total_loss / max(total_zero_loss, 1e-12),
        "mean_tau": float(np.mean(tau_means)) if tau_means else float("nan"),
        "direct_fraction": float(direct_items / max(total_items, 1)),
        "replay_fraction": float(replay_items / max(total_items, 1)),
    }


def train_target_conditioned_score_model(
    model: TargetConditionedScoreModel,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    val_masses: Optional[np.ndarray] = None,
    val_positions: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    tau_levels: Sequence[float] | np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    encoder_lr: Optional[float] = None,
    weight_decay: float = 1e-4,
    projection: str = "none",
    time_weighting: str = "none",
    direct_mixture_probability: float = 0.0,
    oracle_replay_probability: float = 0.0,
    oracle_replay_weight: float = 1.0,
    direct_query_modes: Sequence[str] = ("noised_target", "uniform", "center_gaussian"),
    direct_query_center_std: float = 0.35,
    mixture_chunk_size: Optional[int] = 256,
    mixture_target_norm_clip: Optional[float] = None,
    component_balance_oracle: bool = False,
    component_balance_image_size: int = 64,
    component_balance_dilation: int = 1,
    component_balance_min_pixels: int = 4,
    oracle_replay_steps_per_level: int = 2,
    oracle_replay_max_levels: Optional[int] = None,
    oracle_replay_initial_modes: Sequence[str] = ("uniform", "center_gaussian", "fixed_center"),
    oracle_replay_langevin_alpha: float = 5e-5,
    oracle_replay_diffusion_temperature: float = 1.0,
    oracle_replay_score_scale: float = 1.0,
    measure_gate_regularization: float = 0.0,
    latent_raster_loss_weight: float = 0.0,
    latent_raster_loss: str = "bce_dice",
    latent_raster_positive_weight: float = 25.0,
    latent_raster_dice_weight: float = 1.0,
    latent_raster_blur_steps: int = 1,
    latent_variance_weight: float = 0.0,
    latent_covariance_weight: float = 0.0,
    latent_classification_weight: float = 0.0,
    latent_variance_target: float = 1.0,
    freeze_measure_branch_epochs: int = 0,
    early_stopping_patience: Optional[int] = None,
    lr_scheduler_patience: Optional[int] = 10,
    lr_scheduler_factor: float = 0.5,
    min_lr: float = 1e-5,
    restore_best: bool = True,
    max_grad_norm: Optional[float] = 5.0,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
    show_progress: bool = False,
    progress_desc: str = "target score training",
    dataloader_num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    use_amp: bool = False,
) -> dict[str, list[float]]:
    """Train ``f_phi`` and ``S_theta`` by target-conditioned mixed score matching.

    The optional oracle-replay component distills the empirical-mixture score on
    states generated by the successful oracle sampler, reducing the mismatch
    between local one-step DSM and global reconstruction from a prior.
    """
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if direct_mixture_probability < 0.0 or oracle_replay_probability < 0.0:
        raise ValueError("training objective probabilities must be non-negative")
    if direct_mixture_probability + oracle_replay_probability > 1.0 + 1e-12:
        raise ValueError("direct_mixture_probability + oracle_replay_probability must be <= 1")
    if oracle_replay_weight <= 0.0:
        raise ValueError("oracle_replay_weight must be positive")
    if measure_gate_regularization < 0.0:
        raise ValueError("measure_gate_regularization must be non-negative")
    if latent_raster_loss_weight < 0.0:
        raise ValueError("latent_raster_loss_weight must be non-negative")
    if latent_variance_weight < 0.0 or latent_covariance_weight < 0.0 or latent_classification_weight < 0.0:
        raise ValueError("latent regularization weights must be non-negative")
    tau_levels_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if tau_levels_arr.size == 0 or not np.all(np.isfinite(tau_levels_arr)) or np.any(tau_levels_arr <= 0.0):
        raise ValueError("tau_levels must be positive and finite")
    model_device = _resolve_device(device)
    model = model.to(model_device)
    model.train()
    pin = bool(model_device.type == "cuda") if pin_memory is None else bool(pin_memory)
    loader = _make_tensor_loader(masses, positions, labels, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers, pin_memory=pin)
    enc_lr = lr if encoder_lr is None else float(encoder_lr)
    optimizer = torch.optim.Adam(
        [
            {"params": model.target_encoder.parameters(), "lr": enc_lr},
            {"params": [p for name, p in model.named_parameters() if not name.startswith("target_encoder.")], "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    scaler = _make_grad_scaler(model_device, use_amp)
    scheduler = None
    if lr_scheduler_patience is not None and int(lr_scheduler_patience) >= 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(lr_scheduler_factor),
            patience=int(lr_scheduler_patience),
            min_lr=float(min_lr),
        )
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_sample_loss": [],
        "train_direct_fraction": [],
        "train_replay_fraction": [],
        "val_loss": [],
        "val_loss_ratio_vs_zero": [],
        "val_direct_fraction": [],
        "val_replay_fraction": [],
        "lr": [],
        "measure_gate": [],
        "latent_raster_loss": [],
        "latent_var_loss": [],
        "latent_cov_loss": [],
        "latent_class_loss": [],
        "latent_class_accuracy": [],
        "latent_mean_std": [],
    }
    best_monitor = float("inf")
    best_state: Optional[dict[str, Tensor]] = None
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(int(epochs)):
        if hasattr(model, "set_measure_residual_active"):
            model.set_measure_residual_active(epoch >= int(freeze_measure_branch_epochs))
        model.train()
        total_loss = total_sample_loss = total_latent_raster_loss = 0.0
        total_latent_var_loss = total_latent_cov_loss = total_latent_class_loss = total_latent_class_acc = total_latent_std = 0.0
        total_items = direct_items = replay_items = 0
        epoch_iter = _optional_tqdm(loader, enabled=show_progress, desc=f"{progress_desc} epoch {epoch + 1}/{epochs}", leave=False)
        for batch_masses, batch_positions, batch_labels in epoch_iter:
            batch_masses = batch_masses.to(model_device)
            batch_positions = batch_positions.to(model_device)
            batch_labels = batch_labels.to(model_device)
            tau = _sample_tau_from_levels(int(batch_masses.shape[0]), tau_levels_arr, device=model_device, dtype=batch_positions.dtype)
            query, target_scaled, tau_used, target_kind = _build_score_training_query_and_target(
                batch_masses,
                batch_positions,
                tau,
                projection=projection,
                tau_levels=tau_levels_arr,
                direct_mixture_probability=direct_mixture_probability,
                oracle_replay_probability=oracle_replay_probability,
                direct_query_modes=direct_query_modes,
                direct_query_center_std=direct_query_center_std,
                mixture_chunk_size=mixture_chunk_size,
                mixture_target_norm_clip=mixture_target_norm_clip,
                component_balance_oracle=component_balance_oracle,
                component_balance_image_size=component_balance_image_size,
                component_balance_dilation=component_balance_dilation,
                component_balance_min_pixels=component_balance_min_pixels,
                oracle_replay_steps_per_level=oracle_replay_steps_per_level,
                oracle_replay_max_levels=oracle_replay_max_levels,
                oracle_replay_initial_modes=oracle_replay_initial_modes,
                oracle_replay_langevin_alpha=oracle_replay_langevin_alpha,
                oracle_replay_diffusion_temperature=oracle_replay_diffusion_temperature,
                oracle_replay_score_scale=oracle_replay_score_scale,
            )
            with _amp_autocast_context(model_device, use_amp):
                pred_scaled = model.predict_scaled_score(
                    batch_masses,
                    query,
                    tau_used,
                    target_positions=batch_positions,
                    target_masses=batch_masses,
                    labels=batch_labels,
                )
                loss, metrics = target_conditioned_score_matching_loss(pred_scaled, target_scaled, batch_masses, tau_used, time_weighting=time_weighting)
                objective_loss = loss * (float(oracle_replay_weight) if target_kind == "oracle_replay" else 1.0)
            latent_raster_loss_value = batch_positions.new_tensor(0.0)
            latent_var_value = batch_positions.new_tensor(0.0)
            latent_cov_value = batch_positions.new_tensor(0.0)
            latent_class_value = batch_positions.new_tensor(0.0)
            latent_class_accuracy = float("nan")
            latent_mean_std = 0.0
            need_latents_for_aux = (
                (latent_raster_loss_weight > 0.0 and getattr(model, "latent_raster_decoder", None) is not None)
                or latent_variance_weight > 0.0
                or latent_covariance_weight > 0.0
                or latent_classification_weight > 0.0
            )
            if need_latents_for_aux:
                target_latents_for_aux = model.encode_target(batch_masses, batch_positions)
                if latent_raster_loss_weight > 0.0 and getattr(model, "latent_raster_decoder", None) is not None:
                    latent_raster_loss_value, _ = model.latent_raster_reconstruction_loss(
                        target_latents_for_aux,
                        batch_masses,
                        batch_positions,
                        loss=latent_raster_loss,
                        positive_weight=latent_raster_positive_weight,
                        dice_weight=latent_raster_dice_weight,
                        blur_steps=latent_raster_blur_steps,
                    )
                    objective_loss = objective_loss + float(latent_raster_loss_weight) * latent_raster_loss_value
                if latent_variance_weight > 0.0 or latent_covariance_weight > 0.0:
                    latent_reg, latent_reg_metrics = latent_vicreg_regularization(
                        target_latents_for_aux,
                        variance_target=latent_variance_target,
                        variance_weight=latent_variance_weight,
                        covariance_weight=latent_covariance_weight,
                    )
                    objective_loss = objective_loss + latent_reg
                    latent_var_value = target_latents_for_aux.new_tensor(float(latent_reg_metrics["latent_var_loss"]))
                    latent_cov_value = target_latents_for_aux.new_tensor(float(latent_reg_metrics["latent_cov_loss"]))
                    latent_mean_std = float(latent_reg_metrics["latent_mean_std"])
                if latent_classification_weight > 0.0:
                    latent_class_value, latent_class_metrics = model.latent_classification_loss(target_latents_for_aux, batch_labels)
                    objective_loss = objective_loss + float(latent_classification_weight) * latent_class_value
                    latent_class_accuracy = float(latent_class_metrics["latent_class_accuracy"])
            if measure_gate_regularization > 0.0 and getattr(model, "measure_gate_logit", None) is not None:
                gate = model._measure_residual_gate_tensor().to(device=objective_loss.device, dtype=objective_loss.dtype)
                objective_loss = objective_loss + float(measure_gate_regularization) * gate.square()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(objective_loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            bsz = int(batch_masses.shape[0])
            total_loss += float(objective_loss.detach().item()) * bsz
            total_sample_loss += float(metrics["sample_loss"]) * bsz
            total_latent_raster_loss += float(latent_raster_loss_value.detach().item()) * bsz
            total_latent_var_loss += float(latent_var_value.detach().item()) * bsz
            total_latent_cov_loss += float(latent_cov_value.detach().item()) * bsz
            total_latent_class_loss += float(latent_class_value.detach().item()) * bsz
            if math.isfinite(latent_class_accuracy):
                total_latent_class_acc += float(latent_class_accuracy) * bsz
            total_latent_std += float(latent_mean_std) * bsz
            total_items += bsz
            direct_items += bsz if target_kind == "direct_mixture" else 0
            replay_items += bsz if target_kind == "oracle_replay" else 0
        train_loss = total_loss / max(total_items, 1)
        history["train_loss"].append(train_loss)
        history["train_sample_loss"].append(total_sample_loss / max(total_items, 1))
        history["train_direct_fraction"].append(float(direct_items / max(total_items, 1)))
        history["train_replay_fraction"].append(float(replay_items / max(total_items, 1)))
        history["latent_raster_loss"].append(float(total_latent_raster_loss / max(total_items, 1)))
        history["latent_var_loss"].append(float(total_latent_var_loss / max(total_items, 1)))
        history["latent_cov_loss"].append(float(total_latent_cov_loss / max(total_items, 1)))
        history["latent_class_loss"].append(float(total_latent_class_loss / max(total_items, 1)))
        history["latent_class_accuracy"].append(float(total_latent_class_acc / max(total_items, 1)))
        history["latent_mean_std"].append(float(total_latent_std / max(total_items, 1)))

        if val_masses is not None and val_positions is not None:
            val_metrics = evaluate_target_conditioned_score_model(
                model,
                val_masses,
                val_positions,
                val_labels,
                tau_levels=tau_levels_arr,
                batch_size=batch_size,
                projection=projection,
                time_weighting=time_weighting,
                direct_mixture_probability=direct_mixture_probability,
                oracle_replay_probability=oracle_replay_probability,
                direct_query_modes=direct_query_modes,
                direct_query_center_std=direct_query_center_std,
                mixture_chunk_size=mixture_chunk_size,
                mixture_target_norm_clip=mixture_target_norm_clip,
                component_balance_oracle=component_balance_oracle,
                component_balance_image_size=component_balance_image_size,
                component_balance_dilation=component_balance_dilation,
                component_balance_min_pixels=component_balance_min_pixels,
                oracle_replay_steps_per_level=oracle_replay_steps_per_level,
                oracle_replay_max_levels=oracle_replay_max_levels,
                oracle_replay_initial_modes=oracle_replay_initial_modes,
                oracle_replay_langevin_alpha=oracle_replay_langevin_alpha,
                oracle_replay_diffusion_temperature=oracle_replay_diffusion_temperature,
                oracle_replay_score_scale=oracle_replay_score_scale,
                device=model_device,
            )
            val_loss = float(val_metrics["loss"])
            val_ratio = float(val_metrics["loss_ratio_vs_zero"])
            val_direct_fraction = float(val_metrics["direct_fraction"])
            val_replay_fraction = float(val_metrics["replay_fraction"])
        else:
            val_loss = float("nan")
            val_ratio = float("nan")
            val_direct_fraction = float("nan")
            val_replay_fraction = float("nan")
        history["val_loss"].append(val_loss)
        history["val_loss_ratio_vs_zero"].append(val_ratio)
        history["val_direct_fraction"].append(val_direct_fraction)
        history["val_replay_fraction"].append(val_replay_fraction)
        current_lr = float(min(group["lr"] for group in optimizer.param_groups))
        history["lr"].append(current_lr)
        history["measure_gate"].append(model.measure_residual_gate())
        model.train()

        monitor = val_loss if math.isfinite(val_loss) else train_loss
        if scheduler is not None:
            scheduler.step(monitor)
        if monitor < best_monitor - 1e-6:
            best_monitor = monitor
            best_epoch = epoch
            stale_epochs = 0
            if restore_best:
                best_state = copy.deepcopy(model.state_dict())
        else:
            stale_epochs += 1
        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch + 1 == epochs):
            print(
                f"epoch {epoch + 1:04d}/{epochs}: train_loss={train_loss:.6g} "
                f"val_loss={val_loss:.6g} val/zero={val_ratio:.4f} "
                f"direct={history['train_direct_fraction'][-1]:.2f} "
                f"replay={history['train_replay_fraction'][-1]:.2f} "
                f"gate={history['measure_gate'][-1]:.3f} lr={current_lr:.2e}"
            )
        if early_stopping_patience is not None and int(early_stopping_patience) >= 0:
            if stale_epochs >= int(early_stopping_patience):
                if verbose:
                    print(f"early stopping at epoch {epoch + 1}; best epoch was {best_epoch + 1}")
                break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    if hasattr(model, "set_measure_residual_active"):
        model.set_measure_residual_active(True)
    history["best_epoch"] = [float(best_epoch + 1)]
    history["best_monitor"] = [float(best_monitor)]
    return history



def _weighted_sample_loss(pred_scaled: Tensor, target_scaled: Tensor, masses: Tensor) -> Tensor:
    return torch.mean(torch.sum(masses * torch.sum((pred_scaled - target_scaled).square(), dim=-1), dim=1))


@torch.no_grad()
def evaluate_latent_sensitivity(
    model: TargetConditionedScoreModel,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: Optional[np.ndarray],
    *,
    tau_levels: Sequence[float] | np.ndarray,
    max_samples: int = 64,
    batch_size: int = 32,
    query_mode: str = "uniform",
    device: Optional[str | torch.device] = None,
) -> dict[str, float]:
    """Measure how much the score changes when target latents are permuted/zeroed."""
    masses_arr = np.asarray(masses, dtype=np.float32)[:max_samples]
    positions_arr = np.asarray(positions, dtype=np.float32)[:max_samples]
    if labels is None:
        labels_arr = np.zeros((len(masses_arr),), dtype=np.int64)
    else:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)[:max_samples]
    tau_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if len(tau_arr) == 0:
        raise ValueError("tau_levels must not be empty")
    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    loader = _make_tensor_loader(masses_arr, positions_arr, labels_arr, batch_size=batch_size, shuffle=False)
    true_norm = wrong_diff = zero_diff = 0.0
    total = 0
    for batch_masses, batch_positions, batch_labels in loader:
        batch_masses = batch_masses.to(model_device)
        batch_positions = batch_positions.to(model_device)
        batch_labels = batch_labels.to(model_device)
        tau = _sample_tau_from_levels(int(batch_masses.shape[0]), tau_arr, device=model_device, dtype=batch_positions.dtype)
        if query_mode == "noised_target":
            query, _, _ = perturb_target_conditioned_positions(batch_masses, batch_positions, tau, projection="none")
        elif query_mode == "center_gaussian":
            query = 0.5 + 0.25 * torch.randn_like(batch_positions)
        elif query_mode == "fixed_center":
            query = torch.full_like(batch_positions, 0.5)
        elif query_mode == "uniform":
            query = torch.rand_like(batch_positions)
        else:
            raise ValueError("query_mode must be one of {'uniform','center_gaussian','fixed_center','noised_target'}")
        z_true = model.encode_target(batch_masses, batch_positions)
        perm = torch.roll(torch.arange(z_true.shape[0], device=model_device), shifts=1)
        z_wrong = z_true[perm]
        z_zero = torch.zeros_like(z_true)
        s_true = model.predict_scaled_score(batch_masses, query, tau, target_latents=z_true, labels=batch_labels)
        s_wrong = model.predict_scaled_score(batch_masses, query, tau, target_latents=z_wrong, labels=batch_labels)
        s_zero = model.predict_scaled_score(batch_masses, query, tau, target_latents=z_zero, labels=batch_labels)
        weights = batch_masses.unsqueeze(-1)
        norm = torch.sqrt(torch.sum(weights * s_true.square(), dim=(1, 2)).clamp_min(1e-12))
        wrong = torch.sqrt(torch.sum(weights * (s_true - s_wrong).square(), dim=(1, 2)).clamp_min(0.0))
        zero = torch.sqrt(torch.sum(weights * (s_true - s_zero).square(), dim=(1, 2)).clamp_min(0.0))
        n = int(batch_masses.shape[0])
        true_norm += float(torch.sum(norm).item())
        wrong_diff += float(torch.sum(wrong / norm).item())
        zero_diff += float(torch.sum(zero / norm).item())
        total += n
    if was_training:
        model.train()
    return {
        "mean_scaled_score_norm": true_norm / max(total, 1),
        "wrong_latent_relative_change": wrong_diff / max(total, 1),
        "zero_latent_relative_change": zero_diff / max(total, 1),
        "num_samples": float(total),
    }


def train_latent_only_student_from_teacher(
    teacher: TargetConditionedScoreModel,
    student: TargetConditionedScoreModel,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    val_masses: Optional[np.ndarray] = None,
    val_positions: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    tau_levels: Sequence[float] | np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    query_modes: Sequence[str] = ("uniform", "center_gaussian", "noised_target"),
    direct_query_center_std: float = 0.35,
    latent_raster_loss_weight: float = 1.0,
    latent_raster_loss: str = "bce_dice",
    latent_raster_positive_weight: float = 25.0,
    latent_raster_dice_weight: float = 1.0,
    latent_raster_blur_steps: int = 1,
    latent_variance_weight: float = 1.0,
    latent_covariance_weight: float = 0.05,
    latent_classification_weight: float = 0.1,
    latent_variance_target: float = 1.0,
    posterior_mean_loss_weight: float = 0.0,
    posterior_mean_loss_mode: str = "position",
    posterior_mean_query_modes: Sequence[str] = ("noised_target", "center_gaussian"),
    posterior_mean_target_norm_clip: Optional[float] = None,
    component_balance_oracle: bool = False,
    component_balance_image_size: int = 64,
    component_balance_dilation: int = 1,
    component_balance_min_pixels: int = 4,
    teacher_temperature: float = 1.0,
    latent_autoencoder: Optional[LatentShapeAutoencoder] = None,
    initialize_from_latent_autoencoder: bool = False,
    freeze_latent_modules_epochs: int = 0,
    validation_chamfer_every: Optional[int] = None,
    validation_chamfer_count: int = 8,
    validation_sampler_kwargs: Optional[Mapping[str, Any]] = None,
    validation_chamfer_seed: int = 0,
    restore_best_chamfer: bool = True,
    max_grad_norm: Optional[float] = 5.0,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
    show_progress: bool = False,
    progress_desc: str = "latent student",
    dataloader_num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    use_amp: bool = False,
) -> dict[str, list[float]]:
    """Distill a latent-only student from a successful target-grid teacher.

    The teacher receives ``target_positions`` and can use target-grid features.
    The student encodes the same target into ``z`` but is called with
    ``target_latents`` only.  If the student has ``use_latent_raster_decoder``,
    its generated target raster is used during latent-only sampling.
    """
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if lr <= 0.0 or weight_decay < 0.0 or latent_raster_loss_weight < 0.0:
        raise ValueError("lr must be positive and regularization weights non-negative")
    if latent_variance_weight < 0.0 or latent_covariance_weight < 0.0 or latent_classification_weight < 0.0:
        raise ValueError("latent regularization weights must be non-negative")
    if posterior_mean_loss_weight < 0.0:
        raise ValueError("posterior_mean_loss_weight must be non-negative")
    if posterior_mean_loss_mode not in {"position", "sigma_normalized"}:
        raise ValueError("posterior_mean_loss_mode must be 'position' or 'sigma_normalized'")
    if freeze_latent_modules_epochs < 0:
        raise ValueError("freeze_latent_modules_epochs must be non-negative")
    if validation_chamfer_every is not None and int(validation_chamfer_every) <= 0:
        raise ValueError("validation_chamfer_every must be positive when provided")
    if validation_chamfer_count <= 0:
        raise ValueError("validation_chamfer_count must be positive")
    tau_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if tau_arr.size == 0 or not np.all(np.isfinite(tau_arr)) or np.any(tau_arr <= 0.0):
        raise ValueError("tau_levels must be positive and finite")
    model_device = _resolve_device(device)
    teacher = teacher.to(model_device)
    student = student.to(model_device)
    pin = bool(model_device.type == "cuda") if pin_memory is None else bool(pin_memory)
    if latent_autoencoder is not None:
        latent_autoencoder = latent_autoencoder.to(model_device)
        latent_autoencoder.eval()
        if initialize_from_latent_autoencoder:
            initialize_score_model_from_latent_autoencoder(student, latent_autoencoder)
    teacher.eval()
    student.train()
    loader = _make_tensor_loader(masses, positions, labels, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers, pin_memory=pin)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = _make_grad_scaler(model_device, use_amp)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_distill_loss": [],
        "latent_raster_loss": [],
        "latent_var_loss": [],
        "latent_cov_loss": [],
        "latent_class_loss": [],
        "latent_class_accuracy": [],
        "latent_mean_std": [],
        "posterior_mean_loss": [],
        "latent_modules_frozen": [],
        "val_distill_loss": [],
        "val_latent_only_chamfer": [],
    }
    best_chamfer = float("inf")
    best_chamfer_state: Optional[dict[str, Tensor]] = None

    def _set_latent_modules_trainable(trainable: bool) -> None:
        modules: list[nn.Module] = [student.target_encoder, student.latent_label_head]
        if getattr(student, "latent_raster_decoder", None) is not None:
            modules.append(student.latent_raster_decoder)
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(bool(trainable))

    def _batch_query(batch_masses: Tensor, batch_positions: Tensor, tau: Tensor) -> Tensor:
        mode = str(query_modes[int(torch.randint(0, len(query_modes), ()).item())])
        if mode == "noised_target":
            q, _, _ = perturb_target_conditioned_positions(batch_masses, batch_positions, tau, projection="none")
            return q
        if mode == "uniform":
            return torch.rand_like(batch_positions)
        if mode == "fixed_center":
            return torch.full_like(batch_positions, 0.5)
        if mode == "center_gaussian":
            return 0.5 + float(direct_query_center_std) * torch.randn_like(batch_positions)
        if mode in {"component_noised_target", "component_center_gaussian", "hole_region_uniform"}:
            return _sample_direct_mixture_query_positions(
                batch_positions,
                batch_masses,
                tau,
                query_modes=(mode,),
                center_std=direct_query_center_std,
                projection="none",
                component_balance_image_size=component_balance_image_size,
                component_balance_dilation=component_balance_dilation,
                component_balance_min_pixels=component_balance_min_pixels,
            )
        raise ValueError(f"unknown query mode {mode!r}")

    def _latent_only_validation_chamfer(epoch_index: int) -> float:
        if val_masses is None or val_positions is None:
            return float("nan")
        val_masses_arr = np.asarray(val_masses, dtype=np.float64)
        val_positions_arr = np.asarray(val_positions, dtype=np.float64)
        if val_masses_arr.ndim != 2 or val_positions_arr.shape != (*val_masses_arr.shape, 2):
            return float("nan")
        rng = np.random.default_rng(int(validation_chamfer_seed) + int(epoch_index))
        count = min(int(validation_chamfer_count), val_masses_arr.shape[0])
        idx = rng.choice(val_masses_arr.shape[0], size=count, replace=False)
        kwargs = dict(validation_sampler_kwargs or {})
        kwargs.setdefault("tau_levels", tau_arr)
        kwargs.setdefault("steps_per_level", 2)
        kwargs.setdefault("sampler_scheme", "shape_gf_langevin")
        kwargs.setdefault("initial_position_mode", "uniform")
        kwargs.setdefault("state_projection", "none")
        kwargs.setdefault("diffusion_temperature", 1.0)
        kwargs.setdefault("final_polish_steps", 0)
        kwargs.setdefault("langevin_alpha", 5e-5)
        kwargs.setdefault("batch_size", min(int(batch_size), count))
        kwargs.setdefault("rasterize", False)
        kwargs.setdefault("device", model_device)
        kwargs.setdefault("rng", rng)
        kwargs.setdefault("show_progress", False)
        labels_arr = None if val_labels is None else np.asarray(val_labels, dtype=np.int64).reshape(-1)[idx]
        was_training_student = student.training
        student.eval()
        try:
            validation_latents = encode_target_latents(
                student,
                val_masses_arr[idx],
                val_positions_arr[idx],
                batch_size=min(int(batch_size), count),
                device=model_device,
            )
            generated = reconstruct_target_conditioned_from_latents(
                student,
                validation_latents,
                labels=labels_arr,
                output_masses=val_masses_arr[idx],
                **kwargs,
            )
            metrics = paired_chamfer_reconstruction_metrics(
                generated.positions,
                val_positions_arr[idx],
                labels_arr,
                squared=True,
            )
            return float(metrics["mean_chamfer"])
        finally:
            if was_training_student:
                student.train()

    for epoch in range(int(epochs)):
        student.train()
        _set_latent_modules_trainable(epoch >= int(freeze_latent_modules_epochs))
        total = total_distill = total_raster = 0.0
        total_var = total_cov = total_class = total_class_acc = total_std = 0.0
        total_posterior = 0.0
        total_items = 0
        epoch_iter = _optional_tqdm(loader, enabled=show_progress, desc=f"{progress_desc} epoch {epoch + 1}/{epochs}", leave=False)
        for batch_masses, batch_positions, batch_labels in epoch_iter:
            batch_masses = batch_masses.to(model_device)
            batch_positions = batch_positions.to(model_device)
            batch_labels = batch_labels.to(model_device)
            tau = _sample_tau_from_levels(int(batch_masses.shape[0]), tau_arr, device=model_device, dtype=batch_positions.dtype)
            query = _batch_query(batch_masses, batch_positions, tau)
            with _amp_autocast_context(model_device, use_amp):
                with torch.no_grad():
                    teacher_scaled = teacher.predict_scaled_score(
                        batch_masses,
                        query,
                        tau,
                        target_positions=batch_positions,
                        target_masses=batch_masses,
                        labels=batch_labels,
                    )
                    if teacher_temperature != 1.0:
                        teacher_scaled = float(teacher_temperature) * teacher_scaled
                student_latents = student.encode_target(batch_masses, batch_positions)
                student_scaled = student.predict_scaled_score(
                    batch_masses,
                    query,
                    tau,
                    target_latents=student_latents,
                    labels=batch_labels,
                )
                distill_loss = _weighted_sample_loss(student_scaled, teacher_scaled, batch_masses)
            posterior_loss = batch_positions.new_tensor(0.0)
            if posterior_mean_loss_weight > 0.0:
                if len(posterior_mean_query_modes) == 0:
                    raise ValueError("posterior_mean_query_modes must be non-empty when posterior loss is enabled")
                posterior_query = _batch_query(batch_masses, batch_positions, tau)
                # Prefer modes that emphasize final contour snapping.  If the sampled
                # query mode is not in the allowed set, resample once from the allowed set.
                mode = str(posterior_mean_query_modes[int(torch.randint(0, len(posterior_mean_query_modes), ()).item())])
                if mode != "same_as_training":
                    original_modes = query_modes
                    try:
                        query_modes = (mode,)  # type: ignore[assignment]
                        posterior_query = _batch_query(batch_masses, batch_positions, tau)
                    finally:
                        query_modes = original_modes  # type: ignore[assignment]
                with torch.no_grad():
                    _, _, oracle_posterior_mean = empirical_mixture_scaled_score_target(
                        posterior_query,
                        batch_positions,
                        batch_masses,
                        tau,
                        target_masses=batch_masses,
                        chunk_size=256,
                        target_norm_clip=posterior_mean_target_norm_clip,
                        return_physical_score=True,
                        component_balance=component_balance_oracle,
                        component_balance_image_size=component_balance_image_size,
                        component_balance_dilation=component_balance_dilation,
                        component_balance_min_pixels=component_balance_min_pixels,
                    )
                posterior_pred_scaled = student.predict_scaled_score(
                    batch_masses,
                    posterior_query,
                    tau,
                    target_latents=student_latents,
                    labels=batch_labels,
                )
                sigma = torch.sqrt((2.0 * tau[:, None]) / batch_masses).clamp_min(1e-12)
                if posterior_mean_loss_mode == "sigma_normalized":
                    # Match (m_sigma(x;X)-x)/sigma instead of raw position MSE.
                    # This keeps low-sigma contour-snapping errors numerically visible.
                    oracle_scaled = (oracle_posterior_mean - posterior_query) / sigma.unsqueeze(-1)
                    posterior_loss = _weighted_sample_loss(posterior_pred_scaled, oracle_scaled, batch_masses)
                else:
                    predicted_posterior_mean = posterior_query + sigma.unsqueeze(-1) * posterior_pred_scaled
                    posterior_loss = torch.mean(torch.sum(batch_masses.unsqueeze(-1) * (predicted_posterior_mean - oracle_posterior_mean).square(), dim=(1, 2)))
            raster_loss = batch_positions.new_tensor(0.0)
            latent_var_value = batch_positions.new_tensor(0.0)
            latent_cov_value = batch_positions.new_tensor(0.0)
            latent_class_value = batch_positions.new_tensor(0.0)
            latent_class_accuracy = float("nan")
            latent_mean_std = 0.0
            if latent_raster_loss_weight > 0.0 and getattr(student, "latent_raster_decoder", None) is not None:
                raster_loss, _ = student.latent_raster_reconstruction_loss(
                    student_latents,
                    batch_masses,
                    batch_positions,
                    loss=latent_raster_loss,
                    positive_weight=latent_raster_positive_weight,
                    dice_weight=latent_raster_dice_weight,
                    blur_steps=latent_raster_blur_steps,
                )
            loss = distill_loss + float(latent_raster_loss_weight) * raster_loss + float(posterior_mean_loss_weight) * posterior_loss
            if latent_variance_weight > 0.0 or latent_covariance_weight > 0.0:
                latent_reg, latent_reg_metrics = latent_vicreg_regularization(
                    student_latents,
                    variance_target=latent_variance_target,
                    variance_weight=latent_variance_weight,
                    covariance_weight=latent_covariance_weight,
                )
                loss = loss + latent_reg
                latent_var_value = student_latents.new_tensor(float(latent_reg_metrics["latent_var_loss"]))
                latent_cov_value = student_latents.new_tensor(float(latent_reg_metrics["latent_cov_loss"]))
                latent_mean_std = float(latent_reg_metrics["latent_mean_std"])
            if latent_classification_weight > 0.0:
                latent_class_value, latent_class_metrics = student.latent_classification_loss(student_latents, batch_labels)
                loss = loss + float(latent_classification_weight) * latent_class_value
                latent_class_accuracy = float(latent_class_metrics["latent_class_accuracy"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            bsz = int(batch_masses.shape[0])
            total += float(loss.detach().item()) * bsz
            total_distill += float(distill_loss.detach().item()) * bsz
            total_raster += float(raster_loss.detach().item()) * bsz
            total_posterior += float(posterior_loss.detach().item()) * bsz
            total_var += float(latent_var_value.detach().item()) * bsz
            total_cov += float(latent_cov_value.detach().item()) * bsz
            total_class += float(latent_class_value.detach().item()) * bsz
            if math.isfinite(latent_class_accuracy):
                total_class_acc += float(latent_class_accuracy) * bsz
            total_std += float(latent_mean_std) * bsz
            total_items += bsz
        history["train_loss"].append(total / max(total_items, 1))
        history["train_distill_loss"].append(total_distill / max(total_items, 1))
        history["latent_raster_loss"].append(total_raster / max(total_items, 1))
        history["posterior_mean_loss"].append(total_posterior / max(total_items, 1))
        history["latent_var_loss"].append(total_var / max(total_items, 1))
        history["latent_cov_loss"].append(total_cov / max(total_items, 1))
        history["latent_class_loss"].append(total_class / max(total_items, 1))
        history["latent_class_accuracy"].append(total_class_acc / max(total_items, 1))
        history["latent_mean_std"].append(total_std / max(total_items, 1))
        history["latent_modules_frozen"].append(float(epoch < int(freeze_latent_modules_epochs)))

        if val_masses is not None and val_positions is not None:
            val_loader = _make_tensor_loader(val_masses, val_positions, val_labels, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers, pin_memory=pin)
            val_loss = 0.0
            val_items = 0
            student.eval()
            with torch.no_grad():
                for batch_masses, batch_positions, batch_labels in val_loader:
                    batch_masses = batch_masses.to(model_device)
                    batch_positions = batch_positions.to(model_device)
                    batch_labels = batch_labels.to(model_device)
                    tau = _sample_tau_from_levels(int(batch_masses.shape[0]), tau_arr, device=model_device, dtype=batch_positions.dtype)
                    query = _batch_query(batch_masses, batch_positions, tau)
                    teacher_scaled = teacher.predict_scaled_score(
                        batch_masses,
                        query,
                        tau,
                        target_positions=batch_positions,
                        target_masses=batch_masses,
                        labels=batch_labels,
                    )
                    z = student.encode_target(batch_masses, batch_positions)
                    pred = student.predict_scaled_score(batch_masses, query, tau, target_latents=z, labels=batch_labels)
                    batch_loss = _weighted_sample_loss(pred, teacher_scaled, batch_masses)
                    bsz = int(batch_masses.shape[0])
                    val_loss += float(batch_loss.item()) * bsz
                    val_items += bsz
            history["val_distill_loss"].append(val_loss / max(val_items, 1))
        else:
            history["val_distill_loss"].append(float("nan"))
        val_chamfer = float("nan")
        if validation_chamfer_every is not None and ((epoch + 1) % int(validation_chamfer_every) == 0 or epoch + 1 == int(epochs)):
            val_chamfer = _latent_only_validation_chamfer(epoch + 1)
            if math.isfinite(val_chamfer) and val_chamfer < best_chamfer:
                best_chamfer = float(val_chamfer)
                if restore_best_chamfer:
                    best_chamfer_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
        history["val_latent_only_chamfer"].append(val_chamfer)
        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch + 1 == epochs):
            print(
                f"[latent-student] epoch {epoch + 1:04d}/{epochs}: "
                f"loss={history['train_loss'][-1]:.6g} "
                f"distill={history['train_distill_loss'][-1]:.6g} "
                f"raster={history['latent_raster_loss'][-1]:.6g} "
                f"post={history['posterior_mean_loss'][-1]:.6g} "
                f"z_std={history['latent_mean_std'][-1]:.4f} "
                f"z_cls={history['latent_class_accuracy'][-1]:.3f} "
                f"val={history['val_distill_loss'][-1]:.6g} "
                f"val_chamfer={history['val_latent_only_chamfer'][-1]:.4g}"
            )
    _set_latent_modules_trainable(True)
    if restore_best_chamfer and best_chamfer_state is not None:
        student.load_state_dict(best_chamfer_state)
        student.to(model_device)
    student.train()
    return history


# ---------------------------------------------------------------------------
# Latent-bank provenance helpers
# ---------------------------------------------------------------------------


def model_state_hash(model: nn.Module, *, max_tensors: Optional[int] = None) -> str:
    """Return a stable short hash of a model state dict for latent-bank metadata."""
    digest = hashlib.sha256()
    count = 0
    for key, value in sorted(model.state_dict().items()):
        digest.update(str(key).encode("utf8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf8"))
        digest.update(str(tensor.dtype).encode("utf8"))
        digest.update(tensor.numpy().tobytes())
        count += 1
        if max_tensors is not None and count >= int(max_tensors):
            break
    return digest.hexdigest()[:16]


def latent_bank_to_dict(bank: LatentBank) -> dict[str, Any]:
    """Serialize a :class:`LatentBank` to a checkpoint-friendly dictionary."""
    return {
        "latents": np.asarray(bank.latents, dtype=np.float64),
        "labels": np.asarray(bank.labels, dtype=np.int64),
        "source": str(bank.source),
        "model_hash": str(bank.model_hash),
        "metadata": dict(bank.metadata or {}),
    }


def latent_bank_from_dict(payload: Mapping[str, Any]) -> LatentBank:
    """Restore a :class:`LatentBank` from ``latent_bank_to_dict`` output."""
    if not isinstance(payload, Mapping):
        raise TypeError("latent bank payload must be a mapping")
    return LatentBank(
        latents=np.asarray(payload["latents"], dtype=np.float64),
        labels=np.asarray(payload["labels"], dtype=np.int64),
        source=str(payload.get("source", "unknown")),
        model_hash=str(payload.get("model_hash", "unknown")),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def validate_latent_bank(
    bank: LatentBank,
    *,
    expected_source: str,
    expected_model_hash: Optional[str] = None,
    expected_num_samples: Optional[int] = None,
) -> bool:
    """Check whether a latent bank matches the current model/source request."""
    if not bank.is_compatible(source=expected_source, model_hash=expected_model_hash):
        return False
    if expected_num_samples is not None and bank.num_samples != int(expected_num_samples):
        return False
    if bank.latents.ndim != 2 or bank.labels.shape != (bank.latents.shape[0],):
        return False
    return bool(np.all(np.isfinite(bank.latents)))


@torch.no_grad()
def encode_latent_bank(
    model: nn.Module,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: Optional[np.ndarray],
    *,
    source: str,
    batch_size: int = 256,
    device: Optional[str | torch.device] = None,
    model_hash_value: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> LatentBank:
    """Encode a dataset and attach source/model provenance metadata."""
    latents = encode_target_latents(model, masses, positions, batch_size=batch_size, device=device)
    labels_array = np.zeros((latents.shape[0],), dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_array.shape != (latents.shape[0],):
        raise ValueError("labels must have shape (N,) and match encoded latents")
    if model_hash_value is None:
        model_hash_value = model_state_hash(model)
    return LatentBank(
        latents=latents.astype(np.float64, copy=False),
        labels=labels_array.astype(np.int64, copy=False),
        source=str(source),
        model_hash=str(model_hash_value),
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Encoding and target-conditioned sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_target_latents(
    model: TargetConditionedScoreModel,
    masses: np.ndarray,
    positions: np.ndarray,
    *,
    batch_size: int = 256,
    device: Optional[str | torch.device] = None,
) -> np.ndarray:
    """Encode a bank of target point clouds into latent codes."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    masses_arr = np.asarray(masses, dtype=np.float32)
    positions_arr = np.asarray(positions, dtype=np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(masses_arr), torch.from_numpy(positions_arr)), batch_size=batch_size)
    latents: list[np.ndarray] = []
    for batch_masses, batch_positions in loader:
        batch_masses = batch_masses.to(model_device)
        batch_positions = batch_positions.to(model_device)
        if hasattr(model, "encode_target"):
            z = model.encode_target(batch_masses, batch_positions)  # type: ignore[attr-defined]
        elif hasattr(model, "encode"):
            z = model.encode(batch_masses, batch_positions)  # type: ignore[attr-defined]
        else:
            raise AttributeError("model must define encode_target(...) or encode(...)")
        latents.append(z.detach().cpu().numpy().astype(np.float64))
    if was_training:
        model.train()
    return np.concatenate(latents, axis=0) if latents else np.empty((0, model.latent_dim), dtype=np.float64)


def _resolve_tau_levels_for_sampling(
    *,
    tau_levels: Optional[Sequence[float] | np.ndarray],
    sigma_levels: Optional[Sequence[float] | np.ndarray],
    num_points: int,
) -> np.ndarray:
    if tau_levels is not None:
        levels = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    elif sigma_levels is not None:
        levels = tau_levels_from_sigma_levels(sigma_levels, num_points=num_points)
    else:
        _, levels = make_sigma_tau_schedule(num_points=num_points)
    if levels.size == 0 or not np.all(np.isfinite(levels)) or np.any(levels <= 0.0):
        raise ValueError("tau levels must be positive and finite")
    return np.sort(levels)[::-1].copy().astype(np.float64, copy=False)


def _initial_positions_for_sampling(
    num_samples: int,
    num_points: int,
    *,
    mode: str,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "fixed_center":
        return np.full((num_samples, num_points, 2), 0.5, dtype=np.float64)
    return sample_initial_positions(num_samples, num_points, mode=mode, scale=scale, rng=rng)


@torch.no_grad()
def sample_target_conditioned_annealed_dynamics(
    model: TargetConditionedScoreModel,
    *,
    target_masses: Optional[np.ndarray] = None,
    target_positions: Optional[np.ndarray] = None,
    target_latents: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    output_masses: Optional[np.ndarray] = None,
    num_points: Optional[int] = None,
    tau_levels: Optional[Sequence[float] | np.ndarray] = None,
    sigma_levels: Optional[Sequence[float] | np.ndarray] = None,
    steps_per_level: int = 10,
    sampler_scheme: str = "shape_gf_langevin",
    initial_position_mode: str = "uniform",
    initial_position_scale: float = 0.20,
    state_projection: str = "none",
    score_scale: float = 1.0,
    diffusion_temperature: float = 1.0,
    final_polish_steps: int = 0,
    final_polish_temperature: float = 0.0,
    final_polish_alpha: Optional[float] = None,
    fine_level_count: int = 0,
    fine_level_step_multiplier: int = 1,
    decoded_raster_guidance_weight: float = 0.0,
    decoded_raster_guidance_start_level: Optional[int] = None,
    decoded_raster_guidance_num_points: Optional[int] = None,
    decoded_raster_guidance_channel: int = 0,
    decoded_raster_guidance_component_balance: bool = True,
    decoded_raster_guidance_threshold_quantile: float = 0.75,
    decoded_raster_guidance_sampler_mode: str = "uniform_fps",
    decoded_raster_guidance_corner_weight: float = 1.0,
    decoded_raster_guidance_jitter_scale: float = 0.35,
    coverage_reseed_fraction: float = 0.0,
    coverage_reseed_start_level: Optional[int] = None,
    coverage_reseed_every: int = 1,
    coverage_reseed_jitter: float = 0.01,
    langevin_alpha: float = 5e-5,
    score_calibration: Optional[ScoreCalibration | dict[str, Any]] = None,
    score_norm_clip: Optional[float | Sequence[float] | np.ndarray] = None,
    oracle_prefix_levels: int = 0,
    oracle_suffix_levels: int = 0,
    batch_size: int = 64,
    rasterize: bool = False,
    image_size: int = 28,
    return_trajectories: bool = False,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
    show_progress: bool = False,
    progress_desc: str = "sampling",
) -> GeneratedPointCloudSet:
    """Sample/reconstruct contours using target-conditioned annealed dynamics.

    ``shape_gf_langevin`` follows ShapeGF's noise-then-gradient update.  The
    model still returns the Wasserstein score ``S_i``; the practical point-cloud
    Langevin step uses the corresponding physical score ``s_i S_i``.  The
    ``oracle_prefix_levels``/``oracle_suffix_levels`` switches make hybrid
    bisection diagnostics possible without changing the schedule.
    """
    if steps_per_level <= 0:
        raise ValueError("steps_per_level must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if score_scale <= 0.0 or diffusion_temperature < 0.0:
        raise ValueError("score_scale must be positive and diffusion_temperature must be non-negative")
    if langevin_alpha <= 0.0:
        raise ValueError("langevin_alpha must be positive")
    if final_polish_temperature < 0.0:
        raise ValueError("final_polish_temperature must be non-negative")
    if final_polish_alpha is not None and final_polish_alpha <= 0.0:
        raise ValueError("final_polish_alpha must be positive when provided")
    if fine_level_count < 0 or fine_level_step_multiplier <= 0:
        raise ValueError("fine_level_count must be non-negative and fine_level_step_multiplier positive")
    if not (0.0 <= float(decoded_raster_guidance_weight) <= 1.0):
        raise ValueError("decoded_raster_guidance_weight must lie in [0, 1]")
    if decoded_raster_guidance_num_points is not None and int(decoded_raster_guidance_num_points) <= 0:
        raise ValueError("decoded_raster_guidance_num_points must be positive when provided")
    if not (0.0 <= float(coverage_reseed_fraction) <= 1.0):
        raise ValueError("coverage_reseed_fraction must lie in [0, 1]")
    if coverage_reseed_every <= 0:
        raise ValueError("coverage_reseed_every must be positive")
    if coverage_reseed_jitter < 0.0:
        raise ValueError("coverage_reseed_jitter must be non-negative")
    if oracle_prefix_levels < 0 or oracle_suffix_levels < 0:
        raise ValueError("oracle prefix/suffix levels must be non-negative")
    allowed_schemes = {"theory_euler", "bridge", "langevin", "shape_gf_langevin", "oracle_shape_gf_langevin"}
    if sampler_scheme not in allowed_schemes:
        raise ValueError(f"sampler_scheme must be one of {sorted(allowed_schemes)}")
    rng = np.random.default_rng() if rng is None else rng
    pure_oracle = sampler_scheme == "oracle_shape_gf_langevin"
    uses_any_oracle = pure_oracle or oracle_prefix_levels > 0 or oracle_suffix_levels > 0
    if uses_any_oracle and target_positions is None:
        raise ValueError("oracle or hybrid sampling requires target_positions")

    if target_latents is not None:
        latents_arr = np.asarray(target_latents, dtype=np.float32)
        if latents_arr.ndim != 2:
            raise ValueError("target_latents must have shape (N, latent_dim)")
        num_samples = int(latents_arr.shape[0])
    elif target_positions is not None:
        positions_arr = np.asarray(target_positions, dtype=np.float32)
        if positions_arr.ndim != 3 or positions_arr.shape[2] != 2:
            raise ValueError("target_positions must have shape (N, K, 2)")
        num_samples = int(positions_arr.shape[0])
        latents_arr = None
    else:
        raise ValueError("pass either target_positions or target_latents")

    if labels is None:
        labels_arr = np.zeros((num_samples,), dtype=np.int64)
    else:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.shape != (num_samples,):
            raise ValueError("labels must have shape (N,)")

    if output_masses is not None:
        masses_np = np.asarray(output_masses, dtype=np.float64)
        if masses_np.ndim != 2 or masses_np.shape[0] != num_samples:
            raise ValueError("output_masses must have shape (N, K)")
        num_points_resolved = int(masses_np.shape[1])
    elif target_masses is not None:
        masses_np = np.asarray(target_masses, dtype=np.float64)
        if masses_np.ndim != 2 or masses_np.shape[0] != num_samples:
            raise ValueError("target_masses must have shape (N, K)")
        num_points_resolved = int(masses_np.shape[1])
    else:
        if num_points is None:
            if target_positions is None:
                raise ValueError("num_points is required when sampling from target_latents without masses")
            num_points = int(np.asarray(target_positions).shape[1])
        num_points_resolved = int(num_points)
        masses_np = _uniform_masses(num_samples, num_points_resolved, dtype=np.float64)

    levels = _resolve_tau_levels_for_sampling(tau_levels=tau_levels, sigma_levels=sigma_levels, num_points=num_points_resolved)
    initial_positions_np = _initial_positions_for_sampling(
        num_samples, num_points_resolved, mode=initial_position_mode, scale=initial_position_scale, rng=rng
    )
    initial_positions_np = np.asarray(project_positions(initial_positions_np, mode=state_projection), dtype=np.float64)

    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    masses = torch.from_numpy(masses_np.astype(np.float32)).to(model_device)
    positions = torch.from_numpy(initial_positions_np.astype(np.float32)).to(model_device)
    label_tensor = torch.from_numpy(labels_arr).to(model_device)

    target_positions_tensor = torch.from_numpy(np.asarray(target_positions, dtype=np.float32)).to(model_device) if target_positions is not None else None
    target_masses_tensor = torch.from_numpy(np.asarray(target_masses, dtype=np.float32)).to(model_device) if target_masses is not None else None

    if latents_arr is not None:
        latents = torch.from_numpy(latents_arr).to(model_device)
    elif target_positions_tensor is not None:
        encode_masses = masses if target_masses_tensor is None else target_masses_tensor
        latent_batches: list[Tensor] = []
        for start_idx in range(0, num_samples, batch_size):
            stop = min(start_idx + batch_size, num_samples)
            latent_batches.append(model.encode_target(encode_masses[start_idx:stop], target_positions_tensor[start_idx:stop]))
        latents = torch.cat(latent_batches, dim=0)
    else:
        raise RuntimeError("unreachable latent resolution state")

    decoded_guidance_positions: Optional[Tensor] = None
    decoded_guidance_masses: Optional[Tensor] = None
    if decoded_raster_guidance_weight > 0.0:
        if getattr(model, "latent_raster_decoder", None) is None:
            raise ValueError("decoded_raster_guidance_weight > 0 requires a model with latent_raster_decoder")
        guidance_points = int(decoded_raster_guidance_num_points or num_points_resolved)
        decoded_guidance_positions, decoded_guidance_masses = _decoded_raster_pseudo_targets_torch(
            model,
            latents,
            num_points=guidance_points,
            channel=int(decoded_raster_guidance_channel),
            component_balance=bool(decoded_raster_guidance_component_balance),
            threshold_quantile=float(decoded_raster_guidance_threshold_quantile),
            sampler_mode=decoded_raster_guidance_sampler_mode,
            corner_weight=float(decoded_raster_guidance_corner_weight),
            jitter_scale=float(decoded_raster_guidance_jitter_scale),
        )
    if decoded_raster_guidance_start_level is None:
        decoded_guidance_start_index = max(0, len(levels) - max(int(fine_level_count), 4))
    elif int(decoded_raster_guidance_start_level) < 0:
        decoded_guidance_start_index = max(0, len(levels) + int(decoded_raster_guidance_start_level))
    else:
        decoded_guidance_start_index = min(int(decoded_raster_guidance_start_level), len(levels))
    if coverage_reseed_start_level is None:
        coverage_reseed_start_index = decoded_guidance_start_index
    elif int(coverage_reseed_start_level) < 0:
        coverage_reseed_start_index = max(0, len(levels) + int(coverage_reseed_start_level))
    else:
        coverage_reseed_start_index = min(int(coverage_reseed_start_level), len(levels))

    def _coverage_reseed_batch(batch_positions: Tensor, start_idx: int, stop: int, *, level_index: int, inner_step: int) -> Tensor:
        if decoded_guidance_positions is None or coverage_reseed_fraction <= 0.0:
            return batch_positions
        if level_index < coverage_reseed_start_index or (inner_step % int(coverage_reseed_every)) != 0:
            return batch_positions
        num_replace = int(round(float(coverage_reseed_fraction) * batch_positions.shape[1]))
        if num_replace <= 0:
            return batch_positions
        pseudo = decoded_guidance_positions[start_idx:stop].to(device=batch_positions.device, dtype=batch_positions.dtype)
        d2 = torch.cdist(pseudo, batch_positions).square()
        min_d2 = torch.min(d2, dim=2).values
        topk = torch.topk(min_d2, k=min(num_replace, pseudo.shape[1]), dim=1).indices
        replacement = torch.gather(pseudo, 1, topk.unsqueeze(-1).expand(-1, -1, 2))
        if coverage_reseed_jitter > 0.0:
            replacement = replacement + float(coverage_reseed_jitter) * torch.randn_like(replacement)
        replace_idx = torch.randint(0, batch_positions.shape[1], (batch_positions.shape[0], replacement.shape[1]), device=batch_positions.device)
        out = batch_positions.clone()
        out.scatter_(1, replace_idx.unsqueeze(-1).expand(-1, -1, 2), replacement)
        return project_positions(out, mode=state_projection)

    def _decoded_guidance_active(level_index: int) -> bool:
        return decoded_guidance_positions is not None and level_index >= decoded_guidance_start_index

    def _target_slice(start_idx: int, stop: int) -> tuple[Optional[Tensor], Optional[Tensor]]:
        return (
            target_positions_tensor[start_idx:stop] if target_positions_tensor is not None else None,
            target_masses_tensor[start_idx:stop] if target_masses_tensor is not None else None,
        )

    def _score_batch(
        batch_masses: Tensor,
        batch_positions: Tensor,
        tau: Tensor,
        start_idx: int,
        stop: int,
        *,
        oracle: bool,
        tau_value: float,
        level_index: int,
    ) -> Tensor:
        batch_target_positions, batch_target_masses = _target_slice(start_idx, stop)
        if oracle:
            if batch_target_positions is None:
                raise RuntimeError("oracle score requested without target_positions")
            _, oracle_wasserstein, _ = empirical_gaussian_mixture_scaled_score(
                batch_masses,
                batch_positions,
                batch_target_positions,
                tau,
                target_masses=batch_target_masses,
            )
            return oracle_wasserstein
        model_wasserstein = model(
            batch_masses,
            batch_positions,
            tau,
            target_positions=batch_target_positions,
            target_masses=batch_target_masses,
            target_latents=latents[start_idx:stop],
            labels=label_tensor[start_idx:stop],
        )
        physical_score = batch_masses.unsqueeze(-1) * model_wasserstein
        calibration_scale, calibration_clip = _score_calibration_for_tau(score_calibration, tau_value)
        physical_score = physical_score * float(calibration_scale)
        explicit_clip = _explicit_clip_for_level(score_norm_clip, level_index, len(levels))
        clip_value = explicit_clip if explicit_clip is not None else calibration_clip
        physical_score = _clip_vectors_by_norm(physical_score, clip_value)
        if _decoded_guidance_active(level_index):
            assert decoded_guidance_positions is not None and decoded_guidance_masses is not None
            pseudo_positions = decoded_guidance_positions[start_idx:stop]
            pseudo_masses = decoded_guidance_masses[start_idx:stop]
            _, pseudo_wasserstein, _ = empirical_gaussian_mixture_scaled_score(
                batch_masses,
                batch_positions,
                pseudo_positions,
                tau,
                target_masses=pseudo_masses,
            )
            pseudo_physical = batch_masses.unsqueeze(-1) * pseudo_wasserstein
            pseudo_physical = _clip_vectors_by_norm(pseudo_physical, clip_value)
            weight = float(decoded_raster_guidance_weight)
            physical_score = (1.0 - weight) * physical_score + weight * pseudo_physical
        return physical_score / batch_masses.unsqueeze(-1).clamp_min(1e-12)

    def _use_oracle_for_level(level_index: int) -> bool:
        if pure_oracle:
            return True
        if oracle_prefix_levels > 0 and level_index < int(oracle_prefix_levels):
            return True
        if oracle_suffix_levels > 0 and level_index >= len(levels) - int(oracle_suffix_levels):
            return True
        return False

    trajectory_snapshots: list[np.ndarray] = []
    if return_trajectories:
        trajectory_snapshots.append(positions.detach().cpu().numpy().astype(np.float64))

    if sampler_scheme == "bridge":
        update_pairs = [(float(levels[i]), float(levels[i + 1]) if i + 1 < len(levels) else 0.0) for i in range(len(levels))]
        for level_id, (tau_value, tau_next) in _optional_tqdm(list(enumerate(update_pairs)), enabled=show_progress, desc=f"{progress_desc}: bridge levels", leave=False):
            level_oracle = _use_oracle_for_level(level_id)
            for start_idx in range(0, num_samples, batch_size):
                stop = min(start_idx + batch_size, num_samples)
                batch_masses = masses[start_idx:stop]
                batch_positions = positions[start_idx:stop]
                tau = torch.full((stop - start_idx,), tau_value, device=model_device, dtype=batch_positions.dtype)
                score = _score_batch(batch_masses, batch_positions, tau, start_idx, stop, oracle=level_oracle, tau_value=tau_value, level_index=level_id)
                clean_estimate = batch_positions + 2.0 * float(score_scale) * tau[:, None, None] * score
                if tau_next > 0.0 and diffusion_temperature > 0.0:
                    noise_scale = torch.sqrt((2.0 * tau_next * diffusion_temperature) / batch_masses).unsqueeze(-1)
                    batch_positions = clean_estimate + noise_scale * torch.randn_like(clean_estimate)
                else:
                    batch_positions = clean_estimate
                batch_positions = project_positions(batch_positions, mode=state_projection)
                batch_positions = _coverage_reseed_batch(batch_positions, start_idx, stop, level_index=level_id, inner_step=int(polish_step))
                positions[start_idx:stop] = batch_positions
            if return_trajectories:
                trajectory_snapshots.append(positions.detach().cpu().numpy().astype(np.float64))
    elif sampler_scheme in {"shape_gf_langevin", "oracle_shape_gf_langevin"}:
        tau_min = float(levels[-1])
        for level_id, tau_level in _optional_tqdm(list(enumerate(levels)), enabled=show_progress, desc=f"{progress_desc}: sigma levels", leave=False):
            level_oracle = _use_oracle_for_level(level_id)
            level_steps = int(steps_per_level)
            if fine_level_count > 0 and level_id >= len(levels) - int(fine_level_count):
                level_steps *= int(fine_level_step_multiplier)
            for inner_step in _progress_range(level_steps, enabled=show_progress, desc=f"{progress_desc}: level {level_id + 1}/{len(levels)}", leave=False):
                for start_idx in range(0, num_samples, batch_size):
                    stop = min(start_idx + batch_size, num_samples)
                    batch_masses = masses[start_idx:stop]
                    batch_positions = positions[start_idx:stop]
                    tau = torch.full((stop - start_idx,), float(tau_level), device=model_device, dtype=batch_positions.dtype)
                    sigma = torch.sqrt((2.0 * float(tau_level)) / batch_masses).clamp_min(1e-12)
                    sigma_min = torch.sqrt((2.0 * tau_min) / batch_masses).clamp_min(1e-12)
                    ratio = (sigma / sigma_min).unsqueeze(-1)
                    if diffusion_temperature > 0.0:
                        batch_positions = batch_positions + float(diffusion_temperature) * math.sqrt(float(langevin_alpha)) * ratio * torch.randn_like(batch_positions)
                        batch_positions = project_positions(batch_positions, mode=state_projection)
                    score = _score_batch(batch_masses, batch_positions, tau, start_idx, stop, oracle=level_oracle, tau_value=float(tau_level), level_index=level_id)
                    physical_score = batch_masses.unsqueeze(-1) * score
                    batch_positions = batch_positions + 0.5 * float(langevin_alpha) * ratio.square() * float(score_scale) * physical_score
                    batch_positions = project_positions(batch_positions, mode=state_projection)
                    batch_positions = _coverage_reseed_batch(batch_positions, start_idx, stop, level_index=level_id, inner_step=int(inner_step))
                    positions[start_idx:stop] = batch_positions
            if return_trajectories:
                trajectory_snapshots.append(positions.detach().cpu().numpy().astype(np.float64))
    else:
        # Legacy theory-Euler or SDE-scaled Langevin sampler.
        for level_id, tau_level in _optional_tqdm(list(enumerate(levels)), enabled=show_progress, desc=f"{progress_desc}: levels", leave=False):
            level_oracle = _use_oracle_for_level(level_id)
            tau_next_level = float(levels[level_id + 1]) if level_id + 1 < len(levels) else 0.0
            level_steps = int(steps_per_level)
            if fine_level_count > 0 and level_id >= len(levels) - int(fine_level_count):
                level_steps *= int(fine_level_step_multiplier)
            dt = max((float(tau_level) - tau_next_level) / float(level_steps), float(tau_level) * 1e-4) if sampler_scheme == "theory_euler" else max(0.02 * float(tau_level), 1e-12)
            for inner in range(level_steps):
                tau_value = max(float(tau_level) - inner * dt, max(tau_next_level, 1e-12)) if sampler_scheme == "theory_euler" else float(tau_level)
                for start_idx in range(0, num_samples, batch_size):
                    stop = min(start_idx + batch_size, num_samples)
                    batch_masses = masses[start_idx:stop]
                    batch_positions = positions[start_idx:stop]
                    tau = torch.full((stop - start_idx,), tau_value, device=model_device, dtype=batch_positions.dtype)
                    score = _score_batch(batch_masses, batch_positions, tau, start_idx, stop, oracle=level_oracle, tau_value=tau_value, level_index=level_id)
                    noise_scale = torch.sqrt((2.0 * diffusion_temperature * dt) / batch_masses).unsqueeze(-1)
                    batch_positions = batch_positions + 2.0 * float(score_scale) * dt * score + noise_scale * torch.randn_like(batch_positions)
                    batch_positions = project_positions(batch_positions, mode=state_projection)
                    positions[start_idx:stop] = batch_positions
            if return_trajectories:
                trajectory_snapshots.append(positions.detach().cpu().numpy().astype(np.float64))

    if final_polish_steps > 0 and not pure_oracle:
        tau_value = float(levels[-1])
        level_id = len(levels) - 1
        for polish_step in _progress_range(int(final_polish_steps), enabled=show_progress, desc=f"{progress_desc}: final polish", leave=False):
            for start_idx in range(0, num_samples, batch_size):
                stop = min(start_idx + batch_size, num_samples)
                batch_masses = masses[start_idx:stop]
                batch_positions = positions[start_idx:stop]
                tau = torch.full((stop - start_idx,), tau_value, device=model_device, dtype=batch_positions.dtype)
                score = _score_batch(batch_masses, batch_positions, tau, start_idx, stop, oracle=_use_oracle_for_level(level_id), tau_value=tau_value, level_index=level_id)
                polish_alpha = float(final_polish_alpha) if final_polish_alpha is not None else float(langevin_alpha)
                if final_polish_temperature > 0.0:
                    sigma = torch.sqrt((2.0 * tau_value) / batch_masses).clamp_min(1e-12)
                    sigma_min = torch.sqrt((2.0 * float(levels[-1])) / batch_masses).clamp_min(1e-12)
                    ratio = (sigma / sigma_min).unsqueeze(-1)
                    batch_positions = batch_positions + float(final_polish_temperature) * math.sqrt(polish_alpha) * ratio * torch.randn_like(batch_positions)
                    batch_positions = project_positions(batch_positions, mode=state_projection)
                if sampler_scheme == "shape_gf_langevin":
                    physical_score = batch_masses.unsqueeze(-1) * score
                    batch_positions = batch_positions + 0.5 * polish_alpha * float(score_scale) * physical_score
                else:
                    dt = max(tau_value / float(max(final_polish_steps, 1)), 1e-12)
                    batch_positions = batch_positions + 2.0 * float(score_scale) * dt * score
                batch_positions = project_positions(batch_positions, mode=state_projection)
                positions[start_idx:stop] = batch_positions
        if return_trajectories:
            trajectory_snapshots.append(positions.detach().cpu().numpy().astype(np.float64))

    final_positions = positions.detach().cpu().numpy().astype(np.float64)
    images = None
    if rasterize:
        images = rasterize_weighted_point_clouds(
            masses_np,
            np.asarray(project_positions(final_positions, mode="clip"), dtype=np.float64),
            image_size=image_size,
        )
    if was_training:
        model.train()
    trajectories = np.stack(trajectory_snapshots, axis=0) if return_trajectories else None
    return GeneratedPointCloudSet(
        masses=masses_np.astype(np.float64),
        positions=final_positions,
        labels=labels_arr.astype(np.int64),
        images=images,
        trajectories=trajectories,
    )


@torch.no_grad()
def sample_empirical_mixture_oracle_annealed_dynamics(
    *,
    target_masses: np.ndarray,
    target_positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    model: Optional[TargetConditionedScoreModel] = None,
    **kwargs: Any,
) -> GeneratedPointCloudSet:
    """Run the ShapeGF-style sampler with the exact empirical mixture score."""
    kwargs = dict(kwargs)
    kwargs.pop("sampler_scheme", None)
    kwargs.pop("target_masses", None)
    kwargs.pop("target_positions", None)
    kwargs.pop("output_masses", None)
    if model is None:
        tau_levels = kwargs.get("tau_levels", None)
        if tau_levels is None:
            _, tau_levels_arr = make_sigma_tau_schedule(num_points=int(np.asarray(target_positions).shape[1]))
        else:
            tau_levels_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
        model = TargetConditionedScoreModel(
            latent_dim=8,
            target_encoder_hidden_dim=16,
            grid_size=8,
            base_channels=4,
            grid_feature_dim=4,
            set_feature_dim=4,
            set_hidden_dim=4,
            score_hidden_dim=8,
            time_dim=4,
            context_dim=4,
            num_classes=10,
            tau_min=float(np.min(tau_levels_arr)),
            tau_max=float(np.max(tau_levels_arr)),
            use_image_field=False,
            condition_on_label=False,
        )
    return sample_target_conditioned_annealed_dynamics(
        model,
        target_masses=target_masses,
        target_positions=target_positions,
        labels=labels,
        output_masses=target_masses,
        sampler_scheme="oracle_shape_gf_langevin",
        **kwargs,
    )


sample_empirical_mixture_oracle_dynamics = sample_empirical_mixture_oracle_annealed_dynamics
sample_oracle_mixture_annealed_dynamics = sample_empirical_mixture_oracle_annealed_dynamics


@torch.no_grad()
def reconstruct_target_conditioned_point_clouds(
    model: TargetConditionedScoreModel,
    target_masses: np.ndarray,
    target_positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> GeneratedPointCloudSet:
    """Convenience wrapper for target-conditioned autoencoding/reconstruction."""
    return sample_target_conditioned_annealed_dynamics(
        model,
        target_masses=target_masses,
        target_positions=target_positions,
        labels=labels,
        output_masses=target_masses,
        **kwargs,
    )


@torch.no_grad()
def reconstruct_target_conditioned_from_latents(
    model: TargetConditionedScoreModel,
    target_latents: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    output_masses: Optional[np.ndarray] = None,
    num_points: Optional[int] = None,
    **kwargs: Any,
) -> GeneratedPointCloudSet:
    r"""Reconstruct/sample using only encoded target latents.

    This diagnostic intentionally does not pass ``target_positions`` to the
    sampler.  If target-grid conditioning is enabled, the target-grid branch is
    zeroed and the sampler must rely on the global latent code plus the current
    empirical measure.
    """
    return sample_target_conditioned_annealed_dynamics(
        model,
        target_latents=target_latents,
        labels=labels,
        output_masses=output_masses,
        num_points=num_points,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Experiment checkpoint helpers
# ---------------------------------------------------------------------------


def score_calibration_to_dict(calibration: Optional[ScoreCalibration]) -> Optional[dict[str, Any]]:
    """Convert a :class:`ScoreCalibration` object to a torch-saveable dict."""
    if calibration is None:
        return None
    if not isinstance(calibration, ScoreCalibration):
        raise TypeError("calibration must be a ScoreCalibration or None")
    return {
        "tau_levels": np.asarray(calibration.tau_levels, dtype=np.float64),
        "physical_score_scale": np.asarray(calibration.physical_score_scale, dtype=np.float64),
        "physical_norm_clip": (
            None
            if calibration.physical_norm_clip is None
            else np.asarray(calibration.physical_norm_clip, dtype=np.float64)
        ),
        "metadata": copy.deepcopy(calibration.metadata),
    }


def score_calibration_from_dict(payload: Optional[Mapping[str, Any] | ScoreCalibration]) -> Optional[ScoreCalibration]:
    """Restore a :class:`ScoreCalibration` object from a checkpoint payload."""
    if payload is None:
        return None
    if isinstance(payload, ScoreCalibration):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError("calibration payload must be a mapping, ScoreCalibration, or None")
    return ScoreCalibration(
        tau_levels=np.asarray(payload["tau_levels"], dtype=np.float64),
        physical_score_scale=np.asarray(payload["physical_score_scale"], dtype=np.float64),
        physical_norm_clip=(
            None
            if payload.get("physical_norm_clip") is None
            else np.asarray(payload["physical_norm_clip"], dtype=np.float64)
        ),
        metadata=copy.deepcopy(payload.get("metadata")),
    )


def _state_dict_to_cpu(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Clone a model state dict onto CPU before checkpointing."""
    return {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }


def save_target_conditioned_experiment_checkpoint(
    path: str | Path,
    model: TargetConditionedScoreModel,
    *,
    model_config: Mapping[str, Any],
    score_history: Optional[Mapping[str, Sequence[float]]] = None,
    sigma_levels: Optional[Sequence[float] | np.ndarray] = None,
    tau_levels: Optional[Sequence[float] | np.ndarray] = None,
    score_calibration: Optional[ScoreCalibration] = None,
    score_norm_clip: Optional[Sequence[float] | np.ndarray] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    r"""Save the trained Experiment 8b score model and sampler metadata.

    The checkpoint stores CPU copies of the model weights plus the architecture
    kwargs needed to reconstruct ``TargetConditionedScoreModel``.  It also stores
    the multi-scale schedule and optional calibration/clipping arrays, so the
    notebook can resume from the reconstruction and latent-prior sections without
    rerunning score training.
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": 1,
        "model_class": "TargetConditionedScoreModel",
        "model_config": copy.deepcopy(dict(model_config)),
        "model_state_dict": _state_dict_to_cpu(model.state_dict()),
        "score_history": copy.deepcopy(dict(score_history or {})),
        "sigma_levels": None if sigma_levels is None else np.asarray(sigma_levels, dtype=np.float64),
        "tau_levels": None if tau_levels is None else np.asarray(tau_levels, dtype=np.float64),
        "score_calibration": score_calibration_to_dict(score_calibration),
        "score_norm_clip": None if score_norm_clip is None else np.asarray(score_norm_clip, dtype=np.float64),
        "metadata": copy.deepcopy(dict(metadata or {})),
    }
    if extra is not None:
        payload["extra"] = copy.deepcopy(dict(extra))
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_target_conditioned_experiment_checkpoint(
    path: str | Path,
    *,
    device: Optional[str | torch.device] = None,
    strict: bool = True,
) -> dict[str, Any]:
    r"""Load an Experiment 8b checkpoint and reconstruct the score model.

    Returns the raw payload with two normalized entries added/replaced:
    ``payload["model"]`` is the loaded model on ``device`` and
    ``payload["score_calibration"]`` is a :class:`ScoreCalibration` object or
    ``None``.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    resolved_device = _resolve_device(device)

    try:
        payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose the weights_only argument.
        payload = torch.load(checkpoint_path, map_location=resolved_device)

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    if payload.get("model_class", "TargetConditionedScoreModel") != "TargetConditionedScoreModel":
        raise ValueError(f"unsupported model_class={payload.get('model_class')!r}")
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise KeyError("checkpoint must contain model_config and model_state_dict")

    model = TargetConditionedScoreModel(**dict(payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    model.to(resolved_device)
    model.eval()

    restored = dict(payload)
    restored["model"] = model
    restored["score_calibration"] = score_calibration_from_dict(payload.get("score_calibration"))
    if restored.get("score_norm_clip") is not None:
        restored["score_norm_clip"] = np.asarray(restored["score_norm_clip"], dtype=np.float64)
    if restored.get("sigma_levels") is not None:
        restored["sigma_levels"] = np.asarray(restored["sigma_levels"], dtype=np.float64)
    if restored.get("tau_levels") is not None:
        restored["tau_levels"] = np.asarray(restored["tau_levels"], dtype=np.float64)
    return restored


# ---------------------------------------------------------------------------
# Latent priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianLatentPrior:
    """Class-conditional or unconditional Gaussian prior over encoded shape latents."""

    means: FloatArray
    covariances: FloatArray
    labels: Optional[IntArray]
    diagonal: bool
    eps: float

    @property
    def latent_dim(self) -> int:
        return int(self.means.shape[-1])

    @property
    def num_components(self) -> int:
        return int(self.means.shape[0])


@dataclass(frozen=True)
class EmpiricalLatentPrior:
    """Empirical distribution over encoded training latents."""

    latents: FloatArray
    labels: Optional[IntArray]

    @property
    def latent_dim(self) -> int:
        return int(self.latents.shape[1])

    @property
    def num_samples(self) -> int:
        return int(self.latents.shape[0])


@dataclass(frozen=True)
class LatentStandardizer:
    """Affine standardizer for encoded shape latents."""

    mean: FloatArray
    scale: FloatArray
    eps: float = 1e-6


@dataclass(frozen=True)
class PCALatentPrior:
    """Class-conditional PCA-shrink Gaussian prior over encoded latents."""

    means: FloatArray
    components: FloatArray
    variances: FloatArray
    labels: Optional[IntArray]
    pca_dim: int
    shrink: float
    eps: float
    layernorm_project: bool = False

    @property
    def latent_dim(self) -> int:
        return int(self.means.shape[-1])

    @property
    def num_components(self) -> int:
        return int(self.means.shape[0])

    @property
    def component_labels(self) -> Optional[IntArray]:
        return self.labels

    @property
    def shrinkage(self) -> float:
        return float(self.shrink)


@dataclass(frozen=True)
class PCAGMMLatentPrior:
    """Class-conditional diagonal GMM in per-class PCA coordinates."""

    class_means: FloatArray
    components: FloatArray
    mixture_weights: FloatArray
    mixture_means: FloatArray
    mixture_variances: FloatArray
    labels: Optional[IntArray]
    pca_dim: int
    components_per_class: int
    covariance_shrink: float
    eps: float
    layernorm_project: bool = False

    @property
    def latent_dim(self) -> int:
        return int(self.class_means.shape[-1])

    @property
    def num_components(self) -> int:
        return int(self.class_means.shape[0])


def _validate_latent_matrix(latents: np.ndarray, *, name: str = "latents") -> np.ndarray:
    z = np.asarray(latents, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, D) with N,D > 0")
    if not np.all(np.isfinite(z)):
        raise ValueError(f"{name} contains non-finite values")
    return z


def _resolve_latent_labels(latents: np.ndarray, labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if labels is None:
        return None
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels_arr.shape != (latents.shape[0],):
        raise ValueError("labels must have shape (N,)")
    return labels_arr


def _latent_groups(latents: np.ndarray, labels: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], list[np.ndarray]]:
    labels_arr = _resolve_latent_labels(latents, labels)
    if labels_arr is None:
        return None, [np.arange(latents.shape[0])]
    component_labels = np.unique(labels_arr).astype(np.int64)
    return component_labels, [np.flatnonzero(labels_arr == label) for label in component_labels]


def _component_indices_for_requested_labels(
    prior_labels: Optional[np.ndarray],
    requested_labels: Optional[np.ndarray],
    *,
    num_samples: Optional[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if requested_labels is None:
        if num_samples is None:
            raise ValueError("num_samples is required when labels are not provided")
        if prior_labels is None:
            return np.zeros(int(num_samples), dtype=np.int64), np.zeros(int(num_samples), dtype=np.int64)
        component_idx = rng.integers(0, len(prior_labels), size=int(num_samples), endpoint=False)
        return component_idx, np.asarray(prior_labels, dtype=np.int64)[component_idx]

    out_labels = np.asarray(requested_labels, dtype=np.int64).reshape(-1)
    if prior_labels is None:
        return np.zeros(len(out_labels), dtype=np.int64), out_labels
    label_to_component = {int(label): i for i, label in enumerate(np.asarray(prior_labels, dtype=np.int64))}
    component_idx = np.empty(len(out_labels), dtype=np.int64)
    for i, label in enumerate(out_labels):
        if int(label) not in label_to_component:
            raise ValueError(f"label {int(label)} is not available in the latent prior")
        component_idx[i] = label_to_component[int(label)]
    return component_idx, out_labels


def fit_latent_standardizer(latents: np.ndarray, *, eps: float = 1e-6) -> LatentStandardizer:
    """Fit an elementwise affine standardizer for latent-code diagnostics."""
    z = _validate_latent_matrix(latents)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    mean = np.mean(z, axis=0)
    scale = np.std(z, axis=0)
    scale = np.where(scale > float(eps), scale, 1.0)
    return LatentStandardizer(mean=mean.astype(np.float64), scale=scale.astype(np.float64), eps=float(eps))


def transform_latents(latents: np.ndarray, standardizer: LatentStandardizer) -> np.ndarray:
    """Apply a fitted ``LatentStandardizer``."""
    z = np.asarray(latents, dtype=np.float64)
    return (z - standardizer.mean[None, :]) / standardizer.scale[None, :]


def inverse_transform_latents(latents: np.ndarray, standardizer: LatentStandardizer) -> np.ndarray:
    """Undo ``transform_latents``."""
    z = np.asarray(latents, dtype=np.float64)
    return z * standardizer.scale[None, :] + standardizer.mean[None, :]


def layernorm_project_latents(
    latents: np.ndarray,
    *,
    eps: float = 1e-6,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> np.ndarray:
    """Project latent codes back to a per-sample LayerNorm-like shell."""
    z = _validate_latent_matrix(latents)
    centered = z - np.mean(z, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    return float(target_mean) + float(target_std) * centered / (scale + float(eps))


def sample_empirical_latent_prior(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    requested_labels: Optional[np.ndarray] = None,
    labels_requested: Optional[np.ndarray] = None,
    sample_labels: Optional[np.ndarray] = None,
    num_samples: Optional[int] = None,
    noise_scale: float = 0.0,
    noise_mode: str = "component_std",
    per_dim_noise: bool = True,
    layernorm_project: bool = False,
    project_layernorm: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample encoded training latents, optionally adding small local noise.

    ``requested_labels`` is the preferred argument.  ``labels_requested`` and
    ``sample_labels`` are accepted for compatibility with earlier notebook cells.
    """
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")
    if noise_mode not in {"component_std", "global_std", "none"}:
        raise ValueError("noise_mode must be one of {'component_std', 'global_std', 'none'}")
    if requested_labels is None:
        requested_labels = labels_requested
    if requested_labels is None:
        requested_labels = sample_labels
    if project_layernorm is not None:
        layernorm_project = bool(project_layernorm)
    z = _validate_latent_matrix(latents)
    labels_arr = _resolve_latent_labels(z, labels)
    rng = np.random.default_rng() if rng is None else rng
    if requested_labels is None:
        if num_samples is None:
            raise ValueError("num_samples is required when requested_labels is not provided")
        if labels_arr is None:
            requested = np.zeros(int(num_samples), dtype=np.int64)
            sample_indices = rng.integers(0, z.shape[0], size=int(num_samples), endpoint=False)
        else:
            label_values = np.unique(labels_arr).astype(np.int64)
            requested = label_values[rng.integers(0, len(label_values), size=int(num_samples), endpoint=False)]
            sample_indices = np.empty(int(num_samples), dtype=np.int64)
            for i, label in enumerate(requested):
                idx = np.flatnonzero(labels_arr == int(label))
                sample_indices[i] = int(rng.choice(idx))
    else:
        requested = np.asarray(requested_labels, dtype=np.int64).reshape(-1)
        sample_indices = np.empty(len(requested), dtype=np.int64)
        if labels_arr is None:
            sample_indices[:] = rng.integers(0, z.shape[0], size=len(requested), endpoint=False)
        else:
            for i, label in enumerate(requested):
                idx = np.flatnonzero(labels_arr == int(label))
                if len(idx) == 0:
                    raise ValueError(f"label {int(label)} is not available in the latent bank")
                sample_indices[i] = int(rng.choice(idx))
    samples = z[sample_indices].copy()
    if noise_scale > 0.0 and noise_mode != "none":
        if noise_mode == "global_std" or labels_arr is None:
            scale = np.std(z, axis=0 if per_dim_noise else None) + 1e-12
            samples += float(noise_scale) * rng.normal(size=samples.shape) * scale
        else:
            for label in np.unique(requested):
                mask = requested == int(label)
                idx = np.flatnonzero(labels_arr == int(label))
                scale = np.std(z[idx], axis=0 if per_dim_noise else None) + 1e-12
                samples[mask] += float(noise_scale) * rng.normal(size=samples[mask].shape) * scale
    if layernorm_project:
        samples = layernorm_project_latents(samples)
    if return_indices:
        return samples.astype(np.float64), requested.astype(np.int64), sample_indices.astype(np.int64)
    return samples.astype(np.float64), requested.astype(np.int64)


def sample_empirical_latents(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alias for :func:`sample_empirical_latent_prior`."""
    return sample_empirical_latent_prior(*args, **kwargs)


def latent_nearest_neighbor_summary(
    query_latents: np.ndarray,
    reference_latents: np.ndarray,
    *,
    query_labels: Optional[np.ndarray] = None,
    reference_labels: Optional[np.ndarray] = None,
    same_label: bool = True,
    baseline_latents: Optional[np.ndarray] = None,
    max_reference: Optional[int] = None,
    max_query: Optional[int] = None,
    chunk_size: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Summarize nearest-neighbor distances from query latent codes to a reference bank."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    query = _validate_latent_matrix(query_latents, name="query_latents")
    reference = _validate_latent_matrix(reference_latents, name="reference_latents")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference latent dimensions must match")
    q_labels = _resolve_latent_labels(query, query_labels)
    r_labels = _resolve_latent_labels(reference, reference_labels)
    rng = np.random.default_rng() if rng is None else rng
    if max_query is not None and query.shape[0] > int(max_query):
        idx = rng.choice(query.shape[0], size=int(max_query), replace=False)
        query = query[idx]
        q_labels = None if q_labels is None else q_labels[idx]
    if max_reference is not None and reference.shape[0] > int(max_reference):
        idx = rng.choice(reference.shape[0], size=int(max_reference), replace=False)
        reference = reference[idx]
        r_labels = None if r_labels is None else r_labels[idx]
    if same_label and (q_labels is None or r_labels is None):
        same_label = False

    distances = np.empty(query.shape[0], dtype=np.float64)
    indices = np.empty(query.shape[0], dtype=np.int64)
    all_ref_idx = np.arange(reference.shape[0])
    for i, q in enumerate(query):
        if same_label:
            ref_idx = np.flatnonzero(r_labels == q_labels[i])
            if len(ref_idx) == 0:
                ref_idx = all_ref_idx
        else:
            ref_idx = all_ref_idx
        ref = reference[ref_idx]
        d2 = np.sum((ref - q[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        distances[i] = float(math.sqrt(float(max(d2[j], 0.0))))
        indices[i] = int(ref_idx[j])
    summary: dict[str, Any] = {
        "num_queries": int(query.shape[0]),
        "mean_nn_distance": float(np.mean(distances)),
        "median_nn_distance": float(np.median(distances)),
        "q90_nn_distance": float(np.quantile(distances, 0.90)),
        "q95_nn_distance": float(np.quantile(distances, 0.95)),
        "p95_nn_distance": float(np.quantile(distances, 0.95)),
        "max_nn_distance": float(np.max(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "q95": float(np.quantile(distances, 0.95)),
        "same_label": bool(same_label),
        "nearest_indices": indices,
    }
    if baseline_latents is not None:
        base = latent_nearest_neighbor_summary(
            baseline_latents,
            reference,
            reference_labels=r_labels,
            same_label=False,
            chunk_size=chunk_size,
            rng=rng,
        )
        summary["baseline_mean_nn_distance"] = float(base["mean_nn_distance"])
        summary["baseline_median_nn_distance"] = float(base["median_nn_distance"])
        summary["mean_distance_ratio_to_baseline"] = float(summary["mean_nn_distance"] / max(float(base["mean_nn_distance"]), 1e-12))
        summary["median_distance_ratio_to_baseline"] = float(summary["median_nn_distance"] / max(float(base["median_nn_distance"]), 1e-12))
    if q_labels is not None:
        per_label: dict[int, dict[str, float | int]] = {}
        for label in np.unique(q_labels):
            mask = q_labels == int(label)
            per_label[int(label)] = {
                "count": int(np.sum(mask)),
                "mean_nn_distance": float(np.mean(distances[mask])),
                "median_nn_distance": float(np.median(distances[mask])),
                "q95_nn_distance": float(np.quantile(distances[mask], 0.95)),
            }
        summary["per_label"] = per_label
    return summary


def latent_nearest_neighbor_diagnostics(
    reference_latents: np.ndarray,
    queries: Mapping[str, np.ndarray] | np.ndarray,
    *,
    reference_labels: Optional[np.ndarray] = None,
    query_labels: Optional[Mapping[str, np.ndarray] | np.ndarray] = None,
    baseline_latents: Optional[np.ndarray] = None,
    same_label: bool = True,
    max_reference: Optional[int] = None,
    max_query: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Return nearest-neighbor summaries for one or several latent-prior samples."""
    if isinstance(queries, Mapping):
        rows: list[dict[str, Any]] = []
        for name, z in queries.items():
            labels_for_query = None
            if isinstance(query_labels, Mapping):
                labels_for_query = query_labels.get(name)
            summary = latent_nearest_neighbor_summary(
                z,
                reference_latents,
                query_labels=labels_for_query,
                reference_labels=reference_labels,
                same_label=same_label,
                max_reference=max_reference,
                max_query=max_query,
                rng=rng,
            )
            row = {k: v for k, v in summary.items() if k not in {"nearest_indices", "per_label"}}
            row["name"] = str(name)
            rows.append(row)
        return rows
    return latent_nearest_neighbor_summary(
        queries,
        reference_latents,
        query_labels=query_labels if not isinstance(query_labels, Mapping) else None,
        reference_labels=reference_labels,
        same_label=same_label,
        baseline_latents=baseline_latents,
        max_reference=max_reference,
        max_query=max_query,
        rng=rng,
    )


def _fit_group_pca(group: np.ndarray, pca_dim: int, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(group, axis=0)
    centered = group - mean
    latent_dim = group.shape[1]
    components = np.zeros((pca_dim, latent_dim), dtype=np.float64)
    variances = np.full((pca_dim,), float(eps), dtype=np.float64)
    if group.shape[0] > 1:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        p = min(int(pca_dim), int(vt.shape[0]))
        components[:p] = vt[:p]
        variances[:p] = singular_values[:p] ** 2 / max(group.shape[0] - 1, 1) + float(eps)
    return mean, components, variances


def fit_pca_latent_prior(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    pca_dim: int = 32,
    shrink: Optional[float] = None,
    shrinkage: Optional[float] = None,
    components_per_class: int = 1,
    eps: float = 1e-4,
    layernorm_project: bool = False,
    project_layernorm: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
    **_: Any,
) -> PCALatentPrior:
    """Fit a class-conditional PCA-shrink Gaussian latent prior."""
    del components_per_class, rng  # accepted for compatibility with older cells
    z = _validate_latent_matrix(latents)
    p = min(int(pca_dim), int(z.shape[1]))
    if p <= 0 or eps <= 0.0:
        raise ValueError("pca_dim and eps must be positive")
    shrink_value = 0.5 if shrink is None and shrinkage is None else (float(shrink) if shrink is not None else float(shrinkage))
    if shrink_value < 0.0:
        raise ValueError("shrink must be non-negative")
    if project_layernorm is not None:
        layernorm_project = bool(project_layernorm)
    component_labels, groups = _latent_groups(z, labels)
    means = []
    components = []
    variances = []
    for idx in groups:
        mean, comp, var = _fit_group_pca(z[idx], p, float(eps))
        means.append(mean)
        components.append(comp)
        variances.append(var)
    return PCALatentPrior(
        means=np.asarray(means, dtype=np.float64),
        components=np.asarray(components, dtype=np.float64),
        variances=np.asarray(variances, dtype=np.float64),
        labels=component_labels,
        pca_dim=p,
        shrink=float(shrink_value),
        eps=float(eps),
        layernorm_project=bool(layernorm_project),
    )


def sample_pca_latent_prior(
    prior: PCALatentPrior,
    *,
    labels: Optional[np.ndarray] = None,
    num_samples: Optional[int] = None,
    shrink: Optional[float] = None,
    shrinkage: Optional[float] = None,
    layernorm_project: Optional[bool] = None,
    project_layernorm: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample latent codes from a PCA-shrink Gaussian prior."""
    rng = np.random.default_rng() if rng is None else rng
    component_idx, out_labels = _component_indices_for_requested_labels(prior.labels, labels, num_samples=num_samples, rng=rng)
    shrink_value = prior.shrink if shrink is None and shrinkage is None else (float(shrink) if shrink is not None else float(shrinkage))
    if shrink_value < 0.0:
        raise ValueError("shrink must be non-negative")
    samples = np.empty((len(out_labels), prior.latent_dim), dtype=np.float64)
    for i, comp_idx in enumerate(component_idx):
        c = int(comp_idx)
        coeff = np.sqrt(np.maximum(prior.variances[c], 0.0) * shrink_value) * rng.normal(size=prior.pca_dim)
        samples[i] = prior.means[c] + coeff @ prior.components[c]
    project = prior.layernorm_project if layernorm_project is None else bool(layernorm_project)
    if project_layernorm is not None:
        project = bool(project_layernorm)
    if project:
        samples = layernorm_project_latents(samples)
    return samples.astype(np.float64), out_labels.astype(np.int64, copy=False)


def _simple_kmeans(x: np.ndarray, num_clusters: int, *, max_iter: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("x must have shape (N, D) with N > 0")
    k = max(1, min(int(num_clusters), int(x.shape[0])))
    centers = x[rng.choice(x.shape[0], size=k, replace=False)].copy()
    assignments = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max(1, int(max_iter))):
        d2 = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(d2, axis=1).astype(np.int64)
        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        for j in range(k):
            mask = assignments == j
            centers[j] = np.mean(x[mask], axis=0) if np.any(mask) else x[int(rng.integers(0, x.shape[0]))]
    return centers, assignments


def fit_pca_gmm_latent_prior(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    pca_dim: int = 32,
    components_per_class: int = 8,
    covariance_shrink: float = 0.5,
    covariance_shrinkage: Optional[float] = None,
    shrinkage: Optional[float] = None,
    eps: float = 1e-4,
    max_iter: int = 50,
    layernorm_project: bool = False,
    project_layernorm: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
) -> PCAGMMLatentPrior:
    """Fit a class-conditional diagonal GMM after per-class PCA projection."""
    z = _validate_latent_matrix(latents)
    if covariance_shrinkage is not None:
        covariance_shrink = float(covariance_shrinkage)
    if shrinkage is not None:
        covariance_shrink = float(shrinkage)
    if project_layernorm is not None:
        layernorm_project = bool(project_layernorm)
    if pca_dim <= 0 or components_per_class <= 0 or covariance_shrink < 0.0 or eps <= 0.0:
        raise ValueError("pca_dim/components_per_class/eps must be positive and covariance_shrink non-negative")
    rng = np.random.default_rng() if rng is None else rng
    component_labels, groups = _latent_groups(z, labels)
    p = min(int(pca_dim), int(z.shape[1]))
    m = int(components_per_class)
    class_means = []
    class_components = []
    mixture_weights = []
    mixture_means = []
    mixture_variances = []
    for idx in groups:
        group = z[idx]
        mean, comp, _ = _fit_group_pca(group, p, float(eps))
        scores = (group - mean) @ comp.T
        centers, assignments = _simple_kmeans(scores, m, max_iter=max_iter, rng=rng)
        weights = np.zeros(m, dtype=np.float64)
        means = np.zeros((m, p), dtype=np.float64)
        variances = np.full((m, p), float(eps), dtype=np.float64)
        for j in range(m):
            mask = assignments == j
            if np.any(mask):
                local = scores[mask]
                weights[j] = float(local.shape[0])
                means[j] = np.mean(local, axis=0)
                variances[j] = np.var(local, axis=0, ddof=0) + float(eps)
            else:
                weights[j] = 0.0
                means[j] = centers[min(j, centers.shape[0] - 1)]
        weights = np.ones(m, dtype=np.float64) / float(m) if float(np.sum(weights)) <= 0.0 else weights / float(np.sum(weights))
        class_means.append(mean)
        class_components.append(comp)
        mixture_weights.append(weights)
        mixture_means.append(means)
        mixture_variances.append(variances)
    return PCAGMMLatentPrior(
        class_means=np.asarray(class_means, dtype=np.float64),
        components=np.asarray(class_components, dtype=np.float64),
        mixture_weights=np.asarray(mixture_weights, dtype=np.float64),
        mixture_means=np.asarray(mixture_means, dtype=np.float64),
        mixture_variances=np.asarray(mixture_variances, dtype=np.float64),
        labels=component_labels,
        pca_dim=p,
        components_per_class=m,
        covariance_shrink=float(covariance_shrink),
        eps=float(eps),
        layernorm_project=bool(layernorm_project),
    )


def sample_pca_gmm_latent_prior(
    prior: PCAGMMLatentPrior,
    *,
    labels: Optional[np.ndarray] = None,
    num_samples: Optional[int] = None,
    covariance_shrink: Optional[float] = None,
    covariance_shrinkage: Optional[float] = None,
    shrinkage: Optional[float] = None,
    layernorm_project: Optional[bool] = None,
    project_layernorm: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample latent codes from a class-conditional PCA-space diagonal GMM."""
    rng = np.random.default_rng() if rng is None else rng
    component_idx, out_labels = _component_indices_for_requested_labels(prior.labels, labels, num_samples=num_samples, rng=rng)
    shrink_value = prior.covariance_shrink
    if covariance_shrink is not None:
        shrink_value = float(covariance_shrink)
    if covariance_shrinkage is not None:
        shrink_value = float(covariance_shrinkage)
    if shrinkage is not None:
        shrink_value = float(shrinkage)
    if shrink_value < 0.0:
        raise ValueError("covariance_shrink must be non-negative")
    samples = np.empty((len(out_labels), prior.latent_dim), dtype=np.float64)
    for i, class_idx in enumerate(component_idx):
        c = int(class_idx)
        weights = prior.mixture_weights[c] / max(float(np.sum(prior.mixture_weights[c])), 1e-12)
        j = int(rng.choice(prior.components_per_class, p=weights))
        coords = prior.mixture_means[c, j] + np.sqrt(np.maximum(prior.mixture_variances[c, j], 0.0) * shrink_value) * rng.normal(size=prior.pca_dim)
        samples[i] = prior.class_means[c] + coords @ prior.components[c]
    project = prior.layernorm_project if layernorm_project is None else bool(layernorm_project)
    if project_layernorm is not None:
        project = bool(project_layernorm)
    if project:
        samples = layernorm_project_latents(samples)
    return samples.astype(np.float64), out_labels.astype(np.int64, copy=False)


def fit_gaussian_latent_prior(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    diagonal: bool = True,
    eps: float = 1e-4,
) -> GaussianLatentPrior:
    """Fit a simple latent prior.  With labels, one Gaussian is fit per class."""
    z = _validate_latent_matrix(latents)
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    component_labels, groups = _latent_groups(z, labels)
    means = []
    covariances = []
    for idx in groups:
        group = z[idx]
        means.append(np.mean(group, axis=0))
        centered = group - means[-1]
        if diagonal:
            covariances.append(np.var(centered, axis=0) + float(eps))
        else:
            cov = np.eye(z.shape[1], dtype=np.float64) * float(eps) if len(group) <= 1 else np.cov(group, rowvar=False) + np.eye(z.shape[1], dtype=np.float64) * float(eps)
            covariances.append(cov)
    return GaussianLatentPrior(
        means=np.asarray(means, dtype=np.float64),
        covariances=np.asarray(covariances, dtype=np.float64),
        labels=component_labels,
        diagonal=bool(diagonal),
        eps=float(eps),
    )


def sample_gaussian_latent_prior(
    prior: GaussianLatentPrior,
    *,
    labels: Optional[np.ndarray] = None,
    num_samples: Optional[int] = None,
    covariance_scale: Optional[float] = None,
    covariance_shrinkage: Optional[float] = None,
    layernorm_project: bool = False,
    project_layernorm: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample latent codes and labels/components from a Gaussian prior."""
    rng = np.random.default_rng() if rng is None else rng
    scale = 1.0
    if covariance_scale is not None:
        scale = float(covariance_scale)
    if covariance_shrinkage is not None:
        scale = float(covariance_shrinkage)
    if project_layernorm is not None:
        layernorm_project = bool(project_layernorm)
    if scale < 0.0:
        raise ValueError("covariance scale must be non-negative")
    component_idx, out_labels = _component_indices_for_requested_labels(prior.labels, labels, num_samples=num_samples, rng=rng)
    samples = np.empty((len(out_labels), prior.latent_dim), dtype=np.float64)
    for i, comp in enumerate(component_idx):
        mean = prior.means[int(comp)]
        cov = prior.covariances[int(comp)]
        if prior.diagonal:
            samples[i] = mean + np.sqrt(np.maximum(cov, 0.0) * scale) * rng.normal(size=prior.latent_dim)
        else:
            samples[i] = rng.multivariate_normal(mean, cov * scale)
    if layernorm_project:
        samples = layernorm_project_latents(samples)
    return samples.astype(np.float64), out_labels.astype(np.int64, copy=False)

class LatentGenerator(nn.Module):
    """Small MLP generator for a WGAN-GP latent prior."""

    def __init__(
        self,
        *,
        noise_dim: int = 128,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (256, 256),
        num_classes: int = 10,
        conditional: bool = True,
        label_embedding_dim: int = 32,
        noise_std: float = 0.2,
    ) -> None:
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.latent_dim = int(latent_dim)
        self.conditional = bool(conditional)
        self.num_classes = int(num_classes)
        self.noise_std = float(noise_std)
        if self.noise_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("noise_dim and latent_dim must be positive")
        if self.conditional:
            self.label_embedding = nn.Embedding(self.num_classes, int(label_embedding_dim))
            input_dim = self.noise_dim + int(label_embedding_dim)
        else:
            self.label_embedding = None
            input_dim = self.noise_dim
        layers: list[nn.Module] = []
        dim = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(dim, int(hidden)), nn.ReLU()])
            dim = int(hidden)
        layers.append(nn.Linear(dim, self.latent_dim))
        self.net = nn.Sequential(*layers)

    def sample_noise(self, num_samples: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        return self.noise_std * torch.randn(int(num_samples), self.noise_dim, device=device, dtype=dtype)

    def forward(self, noise: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError("noise must have shape (B, noise_dim)")
        inputs = [noise]
        if self.conditional:
            if labels is None:
                raise ValueError("labels are required for a conditional latent generator")
            inputs.append(self.label_embedding(labels.reshape(-1).to(device=noise.device, dtype=torch.long)))
        return self.net(torch.cat(inputs, dim=1))


class LatentCritic(nn.Module):
    """Small MLP WGAN critic over latent codes."""

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (512, 512),
        num_classes: int = 10,
        conditional: bool = True,
        label_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.conditional = bool(conditional)
        self.num_classes = int(num_classes)
        if self.conditional:
            self.label_embedding = nn.Embedding(self.num_classes, int(label_embedding_dim))
            input_dim = self.latent_dim + int(label_embedding_dim)
        else:
            self.label_embedding = None
            input_dim = self.latent_dim
        layers: list[nn.Module] = []
        dim = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(dim, int(hidden)), nn.LeakyReLU(0.2)])
            dim = int(hidden)
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, latents: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError("latents must have shape (B, latent_dim)")
        inputs = [latents]
        if self.conditional:
            if labels is None:
                raise ValueError("labels are required for a conditional latent critic")
            inputs.append(self.label_embedding(labels.reshape(-1).to(device=latents.device, dtype=torch.long)))
        return self.net(torch.cat(inputs, dim=1)).reshape(-1)


def _gradient_penalty(critic: LatentCritic, real: Tensor, fake: Tensor, labels: Optional[Tensor], *, weight: float) -> Tensor:
    eps = torch.rand(real.shape[0], 1, device=real.device, dtype=real.dtype)
    interp = eps * real + (1.0 - eps) * fake
    interp.requires_grad_(True)
    scores = critic(interp, labels)
    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return float(weight) * torch.mean((grad.reshape(grad.shape[0], -1).norm(2, dim=1) - 1.0).square())


def train_latent_wgan_gp(
    generator: LatentGenerator,
    critic: LatentCritic,
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    epochs: int = 1000,
    batch_size: int = 128,
    lr: float = 1e-4,
    betas: tuple[float, float] = (0.5, 0.9),
    gradient_penalty_weight: float = 10.0,
    critic_steps: int = 5,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train a small WGAN-GP prior on encoded target-shape latents."""
    if epochs <= 0 or batch_size <= 0 or critic_steps <= 0:
        raise ValueError("epochs, batch_size and critic_steps must be positive")
    z = np.asarray(latents, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError("latents must have shape (N, D) with N > 0")
    if labels is None:
        labels_arr = np.zeros((z.shape[0],), dtype=np.int64)
    else:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.shape != (z.shape[0],):
            raise ValueError("labels must have shape (N,)")

    model_device = _resolve_device(device)
    generator = generator.to(model_device)
    critic = critic.to(model_device)
    generator.train()
    critic.train()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z), torch.from_numpy(labels_arr)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
    history = {"critic_loss": [], "generator_loss": [], "gradient_penalty": []}

    global_step = 0
    for epoch in range(int(epochs)):
        critic_losses: list[float] = []
        generator_losses: list[float] = []
        gp_losses: list[float] = []
        for real_z, batch_labels in loader:
            real_z = real_z.to(model_device)
            batch_labels = batch_labels.to(model_device)

            noise = generator.sample_noise(real_z.shape[0], device=model_device, dtype=real_z.dtype)
            with torch.no_grad():
                fake_z = generator(noise, batch_labels if generator.conditional else None)
            real_score = critic(real_z, batch_labels if critic.conditional else None)
            fake_score = critic(fake_z, batch_labels if critic.conditional else None)
            gp = _gradient_penalty(
                critic,
                real_z,
                fake_z,
                batch_labels if critic.conditional else None,
                weight=gradient_penalty_weight,
            )
            critic_loss = torch.mean(fake_score) - torch.mean(real_score) + gp
            opt_d.zero_grad(set_to_none=True)
            critic_loss.backward()
            opt_d.step()
            critic_losses.append(float(critic_loss.detach().item()))
            gp_losses.append(float(gp.detach().item()))

            if global_step % int(critic_steps) == 0:
                noise = generator.sample_noise(real_z.shape[0], device=model_device, dtype=real_z.dtype)
                fake_z = generator(noise, batch_labels if generator.conditional else None)
                gen_loss = -torch.mean(critic(fake_z, batch_labels if critic.conditional else None))
                opt_g.zero_grad(set_to_none=True)
                gen_loss.backward()
                opt_g.step()
                generator_losses.append(float(gen_loss.detach().item()))
            global_step += 1

        history["critic_loss"].append(float(np.mean(critic_losses)) if critic_losses else float("nan"))
        history["generator_loss"].append(float(np.mean(generator_losses)) if generator_losses else float("nan"))
        history["gradient_penalty"].append(float(np.mean(gp_losses)) if gp_losses else float("nan"))
        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch + 1 == epochs):
            print(
                f"latent WGAN epoch {epoch + 1:04d}/{epochs}: "
                f"D={history['critic_loss'][-1]:.6g} G={history['generator_loss'][-1]:.6g} "
                f"GP={history['gradient_penalty'][-1]:.6g}"
            )
    return history


@torch.no_grad()
def sample_wgan_latent_prior(
    generator: LatentGenerator,
    *,
    labels: Optional[np.ndarray] = None,
    num_samples: Optional[int] = None,
    device: Optional[str | torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample latent codes from a trained ``LatentGenerator``."""
    if labels is None:
        if num_samples is None:
            raise ValueError("num_samples is required when labels are not provided")
        labels_arr = np.random.default_rng().integers(0, generator.num_classes, size=int(num_samples), endpoint=False)
    else:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    model_device = _resolve_device(device)
    was_training = generator.training
    generator = generator.to(model_device)
    generator.eval()
    label_tensor = torch.from_numpy(labels_arr).to(model_device)
    noise = generator.sample_noise(len(labels_arr), device=model_device)
    z = generator(noise, label_tensor if generator.conditional else None)
    if was_training:
        generator.train()
    return z.detach().cpu().numpy().astype(np.float64), labels_arr.astype(np.int64, copy=False)


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------


def paired_chamfer_reconstruction_metrics(
    reconstructed_positions: np.ndarray,
    target_positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    squared: bool = True,
) -> dict[str, Any]:
    """Paired target-vs-reconstruction Chamfer summary."""
    from mnist.mnist_cp import chamfer_distance

    recon = np.asarray(reconstructed_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if recon.shape != target.shape or recon.ndim != 3 or recon.shape[2] != 2:
        raise ValueError("reconstructed_positions and target_positions must both have shape (N, K, 2)")
    values = np.asarray(
        [chamfer_distance(recon[i], target[i], squared=squared) for i in range(recon.shape[0])],
        dtype=np.float64,
    )
    out: dict[str, Any] = {
        "mean_chamfer": float(np.mean(values)),
        "median_chamfer": float(np.median(values)),
        "std_chamfer": float(np.std(values)),
        "squared_chamfer": bool(squared),
        "per_sample_chamfer": values,
    }
    if labels is not None:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.shape != (recon.shape[0],):
            raise ValueError("labels must have shape (N,)")
        per_label = {}
        for label in np.unique(labels_arr):
            mask = labels_arr == label
            per_label[int(label)] = {
                "mean_chamfer": float(np.mean(values[mask])),
                "median_chamfer": float(np.median(values[mask])),
                "count": int(np.sum(mask)),
            }
        out["per_label"] = per_label
    return out




def _binary_dilate_numpy(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(steps))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        dilated = np.zeros_like(out, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                dilated |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = dilated
    return out


def _rasterize_points_binary(points: np.ndarray, *, image_size: int = 64, dilation: int = 0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (K, 2)")
    image_size = int(image_size)
    if image_size <= 1:
        raise ValueError("image_size must be greater than one")
    pix = np.floor(np.clip(pts, 0.0, 1.0) * float(image_size - 1) + 0.5).astype(np.int64)
    mask = np.zeros((image_size, image_size), dtype=bool)
    mask[pix[:, 1], pix[:, 0]] = True
    if dilation > 0:
        mask = _binary_dilate_numpy(mask, dilation)
    return mask


def _connected_components_numpy(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    sizes: list[int] = []
    current = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
            sizes.append(size)
    return labels, sizes


def _hole_mask_numpy(contour_mask: np.ndarray, *, dilation: int = 1) -> np.ndarray:
    # Dilate the contour a little so flood fill does not leak through sparse pixel gaps.
    barrier = _binary_dilate_numpy(contour_mask, dilation)
    background = ~barrier
    height, width = background.shape
    outside = np.zeros_like(background, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(width):
        if background[0, x]:
            outside[0, x] = True
            stack.append((0, x))
        if background[height - 1, x]:
            outside[height - 1, x] = True
            stack.append((height - 1, x))
    for y in range(height):
        if background[y, 0] and not outside[y, 0]:
            outside[y, 0] = True
            stack.append((y, 0))
        if background[y, width - 1] and not outside[y, width - 1]:
            outside[y, width - 1] = True
            stack.append((y, width - 1))
    while stack:
        cy, cx = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < height and 0 <= nx < width and background[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    return background & (~outside)


def raster_topology_summary(
    positions: np.ndarray,
    *,
    image_size: int = 64,
    contour_dilation: int = 1,
    hole_dilation: int = 1,
    min_component_pixels: int = 4,
    min_hole_pixels: int = 4,
) -> dict[str, float]:
    """Raster topology summary for one MNIST-CP contour point cloud."""
    contour = _rasterize_points_binary(positions, image_size=image_size, dilation=contour_dilation)
    comp_labels, comp_sizes = _connected_components_numpy(contour)
    comp_sizes_kept = [size for size in comp_sizes if size >= int(min_component_pixels)]
    holes = _hole_mask_numpy(contour, dilation=hole_dilation)
    _, hole_sizes = _connected_components_numpy(holes)
    hole_sizes_kept = [size for size in hole_sizes if size >= int(min_hole_pixels)]
    return {
        "component_count": float(len(comp_sizes_kept)),
        "hole_count": float(len(hole_sizes_kept)),
        "largest_component_pixels": float(max(comp_sizes_kept) if comp_sizes_kept else 0),
        "hole_pixels": float(sum(hole_sizes_kept)),
        "contour_pixels": float(np.sum(contour)),
        "hole_fraction": float(np.mean(holes)),
        "occupied_fraction": float(np.mean(contour)),
    }


def _component_balanced_weights_numpy(
    points: np.ndarray,
    *,
    image_size: int = 64,
    contour_dilation: int = 1,
    min_component_pixels: int = 4,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    contour = _rasterize_points_binary(pts, image_size=image_size, dilation=contour_dilation)
    labels, sizes = _connected_components_numpy(contour)
    pix = np.floor(np.clip(pts, 0.0, 1.0) * float(image_size - 1) + 0.5).astype(np.int64)
    point_labels = labels[pix[:, 1], pix[:, 0]]
    kept = [idx + 1 for idx, size in enumerate(sizes) if size >= int(min_component_pixels)]
    if not kept:
        return np.full((pts.shape[0],), 1.0 / max(pts.shape[0], 1), dtype=np.float64)
    weights = np.zeros((pts.shape[0],), dtype=np.float64)
    kept_set = set(kept)
    for label in kept:
        mask = point_labels == label
        if np.any(mask):
            weights[mask] = 1.0 / (len(kept) * float(np.sum(mask)))
    if float(np.sum(weights)) <= 0.0:
        weights[:] = 1.0 / float(max(len(weights), 1))
    else:
        weights /= float(np.sum(weights))
    return weights


def component_balanced_target_masses(
    target_positions: np.ndarray,
    *,
    image_size: int = 64,
    contour_dilation: int = 1,
    min_component_pixels: int = 4,
) -> np.ndarray:
    """Return target mixture weights that assign equal mass to each contour component."""
    arr = np.asarray(target_positions, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError("target_positions must have shape (N, K, 2)")
    return np.stack(
        [
            _component_balanced_weights_numpy(
                cloud,
                image_size=image_size,
                contour_dilation=contour_dilation,
                min_component_pixels=min_component_pixels,
            )
            for cloud in arr
        ],
        axis=0,
    )


def _component_balanced_target_masses_torch(
    target_positions: Tensor,
    *,
    image_size: int = 64,
    contour_dilation: int = 1,
    min_component_pixels: int = 4,
) -> Tensor:
    weights = component_balanced_target_masses(
        target_positions.detach().cpu().numpy(),
        image_size=image_size,
        contour_dilation=contour_dilation,
        min_component_pixels=min_component_pixels,
    )
    return torch.as_tensor(weights, dtype=target_positions.dtype, device=target_positions.device)


def _sample_component_balanced_points_torch(
    target_positions: Tensor,
    *,
    image_size: int = 64,
    contour_dilation: int = 1,
    min_component_pixels: int = 4,
) -> Tensor:
    weights = _component_balanced_target_masses_torch(
        target_positions,
        image_size=image_size,
        contour_dilation=contour_dilation,
        min_component_pixels=min_component_pixels,
    ).clamp_min(1e-12)
    weights = weights / torch.sum(weights, dim=1, keepdim=True).clamp_min(1e-12)
    indices = torch.multinomial(weights, num_samples=target_positions.shape[1], replacement=True)
    return torch.gather(target_positions, 1, indices.unsqueeze(-1).expand(-1, -1, 2))


def _sample_hole_uniform_points_torch(
    target_positions: Tensor,
    *,
    image_size: int = 64,
    hole_dilation: int = 1,
    fallback_std: float = 0.25,
) -> Tensor:
    clouds = target_positions.detach().cpu().numpy()
    out = np.empty_like(clouds, dtype=np.float32)
    rng = np.random.default_rng()
    for i, cloud in enumerate(clouds):
        contour = _rasterize_points_binary(cloud, image_size=image_size, dilation=1)
        holes = _hole_mask_numpy(contour, dilation=hole_dilation)
        ys, xs = np.nonzero(holes)
        if len(xs) == 0:
            center = np.mean(cloud, axis=0)
            out[i] = center[None, :] + float(fallback_std) * rng.normal(size=cloud.shape)
        else:
            pick = rng.integers(0, len(xs), size=cloud.shape[0])
            jitter = rng.uniform(-0.35, 0.35, size=cloud.shape)
            coords = np.stack([xs[pick], ys[pick]], axis=1).astype(np.float64)
            out[i] = (coords + 0.5 + jitter) / float(image_size)
    return torch.as_tensor(np.clip(out, 0.0, 1.0), dtype=target_positions.dtype, device=target_positions.device)




def _sample_corner_points_torch(
    target_positions: Tensor,
    *,
    image_size: int = 64,
    contour_dilation: int = 1,
    corner_quantile: float = 0.75,
) -> Tensor:
    clouds = corner_points_from_contour(
        target_positions.detach().cpu().numpy(),
        num_points=int(target_positions.shape[1]),
        image_size=image_size,
        contour_dilation=contour_dilation,
        corner_quantile=corner_quantile,
    )
    return torch.as_tensor(clouds, dtype=target_positions.dtype, device=target_positions.device)


def _raster_topology_summary_from_mask(
    contour_mask: np.ndarray,
    *,
    hole_dilation: int = 1,
    min_component_pixels: int = 4,
    min_hole_pixels: int = 4,
) -> dict[str, float]:
    contour = np.asarray(contour_mask, dtype=bool)
    _, comp_sizes = _connected_components_numpy(contour)
    comp_sizes_kept = [size for size in comp_sizes if size >= int(min_component_pixels)]
    holes = _hole_mask_numpy(contour, dilation=hole_dilation)
    _, hole_sizes = _connected_components_numpy(holes)
    hole_sizes_kept = [size for size in hole_sizes if size >= int(min_hole_pixels)]
    return {
        "component_count": float(len(comp_sizes_kept)),
        "hole_count": float(len(hole_sizes_kept)),
        "largest_component_pixels": float(max(comp_sizes_kept) if comp_sizes_kept else 0),
        "hole_pixels": float(sum(hole_sizes_kept)),
        "contour_pixels": float(np.sum(contour)),
        "hole_fraction": float(np.mean(holes)),
        "occupied_fraction": float(np.mean(contour)),
    }



def _cornerness_from_probability_grid_numpy(grid: np.ndarray) -> np.ndarray:
    """Cheap high-curvature / turning-region proxy for a decoded contour raster."""
    prob = np.asarray(grid, dtype=np.float64)
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    if prob.ndim != 2:
        raise ValueError("grid must have shape (H, W)")
    if float(np.max(prob)) > 0.0:
        prob = prob / float(np.max(prob))
    gy, gx = np.gradient(prob)
    grad = np.sqrt(gx * gx + gy * gy)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    curvature = np.sqrt(gxx * gxx + gyy * gyy + 0.5 * (gxy * gxy + gyx * gyx))
    score = grad * curvature
    score = np.maximum(score, 0.0)
    if float(np.max(score)) > 0.0:
        score = score / float(np.max(score))
    return score


def _grid_indices_to_unit_points_numpy(ys: np.ndarray, xs: np.ndarray, *, height: int, width: int, rng: np.random.Generator, jitter_scale: float = 0.45) -> np.ndarray:
    jitter = rng.uniform(-float(jitter_scale), float(jitter_scale), size=(len(xs), 2))
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    points = (coords + 0.5 + jitter) / np.asarray([[max(width, 1), max(height, 1)]], dtype=np.float64)
    return np.clip(points, 0.0, 1.0).astype(np.float64)


def _farthest_point_sample_indices_numpy(points: np.ndarray, num_samples: int, *, rng: np.random.Generator, weights: Optional[np.ndarray] = None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("points must have shape (N, D) with N > 0")
    n = pts.shape[0]
    k = min(int(num_samples), n)
    if weights is None:
        first = int(rng.integers(0, n))
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = np.maximum(w, 0.0)
        if w.shape != (n,) or float(np.sum(w)) <= 0.0:
            first = int(rng.integers(0, n))
        else:
            first = int(rng.choice(n, p=w / float(np.sum(w))))
    selected = np.empty((k,), dtype=np.int64)
    selected[0] = first
    min_d2 = np.sum((pts - pts[first][None, :]) ** 2, axis=1)
    for i in range(1, k):
        # Prefer uncovered points, with a small probability boost for high-weight pixels.
        score = min_d2.copy()
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape == (n,) and float(np.max(w)) > 0.0:
                score = score * (1.0 + 0.25 * w / float(np.max(w)))
        selected[i] = int(np.argmax(score))
        d2 = np.sum((pts - pts[selected[i]][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    if int(num_samples) > n:
        extra = rng.choice(selected, size=int(num_samples) - n, replace=True)
        selected = np.concatenate([selected, extra.astype(np.int64)])
    return selected


def corner_points_from_contour(
    positions: np.ndarray,
    *,
    num_points: Optional[int] = None,
    image_size: int = 64,
    contour_dilation: int = 1,
    corner_quantile: float = 0.75,
    seed: int = 0,
) -> np.ndarray:
    """Sample points near high-curvature rasterized contour regions."""
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim == 2:
        pts = pts[None, :, :]
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError("positions must have shape (K,2) or (N,K,2)")
    rng = np.random.default_rng(int(seed))
    clouds: list[np.ndarray] = []
    count = int(num_points or pts.shape[1])
    for cloud in pts:
        mask = _rasterize_points_binary(cloud, image_size=image_size, dilation=contour_dilation).astype(np.float64)
        corner = _cornerness_from_probability_grid_numpy(mask)
        positive = corner[corner > 0.0]
        if positive.size == 0:
            # Fall back to component-balanced target samples.
            weights = _component_balanced_weights_numpy(cloud, image_size=image_size, contour_dilation=contour_dilation)
            idx = rng.choice(cloud.shape[0], size=count, replace=True, p=weights)
            clouds.append(cloud[idx])
            continue
        threshold = float(np.quantile(positive, float(np.clip(corner_quantile, 0.0, 1.0))))
        ys, xs = np.nonzero(corner >= max(threshold, 1e-8))
        if len(xs) == 0:
            ys, xs = np.nonzero(mask > 0.0)
        if len(xs) == 0:
            clouds.append(rng.uniform(0.0, 1.0, size=(count, 2)))
            continue
        candidates = np.stack([xs / max(image_size - 1, 1), ys / max(image_size - 1, 1)], axis=1)
        weights = corner[ys, xs]
        indices = _farthest_point_sample_indices_numpy(candidates, count, rng=rng, weights=weights)
        clouds.append(_grid_indices_to_unit_points_numpy(ys[indices], xs[indices], height=image_size, width=image_size, rng=rng, jitter_scale=0.30))
    return np.stack(clouds, axis=0) if positions.ndim == 3 else clouds[0]


def _sample_points_from_probability_grid_numpy(
    grid: np.ndarray,
    *,
    num_points: int,
    component_balance: bool = False,
    threshold_quantile: float = 0.75,
    min_component_pixels: int = 3,
    rng: Optional[np.random.Generator] = None,
    sampler_mode: str = "random",
    corner_weight: float = 0.0,
    jitter_scale: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample [0,1]^2 points from a raster probability grid.

    ``sampler_mode='uniform_fps'`` spreads points over the decoded contour by
    farthest-point sampling over high-probability raster pixels.  ``corner_weight``
    boosts high-curvature raster pixels, which helps populate stroke turns.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if sampler_mode not in {"random", "uniform_fps", "fps", "corner_fps"}:
        raise ValueError("sampler_mode must be one of {'random', 'uniform_fps', 'fps', 'corner_fps'}")
    if corner_weight < 0.0:
        raise ValueError("corner_weight must be non-negative")
    rng = np.random.default_rng() if rng is None else rng
    prob = np.asarray(grid, dtype=np.float64)
    if prob.ndim != 2:
        raise ValueError("grid must have shape (H, W)")
    height, width = prob.shape
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.maximum(prob, 0.0)
    if float(np.sum(prob)) <= 0.0:
        prob = np.ones_like(prob, dtype=np.float64)
    corner = _cornerness_from_probability_grid_numpy(prob)
    if corner_weight > 0.0:
        prob = prob * (1.0 + float(corner_weight) * corner)

    positive_values = prob[prob > 0.0]
    threshold = float(np.quantile(positive_values, float(np.clip(threshold_quantile, 0.0, 1.0)))) if positive_values.size else 0.0
    candidate_mask = prob >= max(threshold, 1e-8)
    if not np.any(candidate_mask):
        candidate_mask = prob > 0.0
    labels = None
    components: list[int] = []
    if component_balance:
        labels, sizes = _connected_components_numpy(candidate_mask)
        components = [i + 1 for i, size in enumerate(sizes) if size >= int(min_component_pixels)]

    if sampler_mode in {"uniform_fps", "fps", "corner_fps"}:
        if component_balance and components and labels is not None:
            ys_all: list[np.ndarray] = []
            xs_all: list[np.ndarray] = []
            counts = np.full((len(components),), int(num_points) // len(components), dtype=np.int64)
            counts[: int(num_points) % len(components)] += 1
            for count, label in zip(counts, components):
                comp_y, comp_x = np.nonzero(labels == label)
                if len(comp_x) == 0 or count <= 0:
                    continue
                candidates = np.stack([comp_x / max(width - 1, 1), comp_y / max(height - 1, 1)], axis=1)
                weights = prob[comp_y, comp_x]
                if sampler_mode == "corner_fps" or corner_weight > 0.0:
                    weights = weights * (1.0 + float(max(corner_weight, 1.0)) * corner[comp_y, comp_x])
                idx = _farthest_point_sample_indices_numpy(candidates, int(count), rng=rng, weights=weights)
                ys_all.append(comp_y[idx])
                xs_all.append(comp_x[idx])
            ys = np.concatenate(ys_all) if ys_all else np.empty((0,), dtype=np.int64)
            xs = np.concatenate(xs_all) if xs_all else np.empty((0,), dtype=np.int64)
        else:
            ys, xs = np.nonzero(candidate_mask)
            if len(xs) > 0:
                candidates = np.stack([xs / max(width - 1, 1), ys / max(height - 1, 1)], axis=1)
                weights = prob[ys, xs]
                if sampler_mode == "corner_fps" or corner_weight > 0.0:
                    weights = weights * (1.0 + float(max(corner_weight, 1.0)) * corner[ys, xs])
                idx = _farthest_point_sample_indices_numpy(candidates, int(num_points), rng=rng, weights=weights)
                ys, xs = ys[idx], xs[idx]
        if len(xs) == 0:
            flat = (prob / max(float(np.sum(prob)), 1e-12)).reshape(-1)
            picks = rng.choice(flat.size, size=int(num_points), replace=True, p=flat)
            ys, xs = np.divmod(picks, width)
        elif len(xs) < int(num_points):
            extra = rng.choice(len(xs), size=int(num_points) - len(xs), replace=True)
            ys = np.concatenate([ys, ys[extra]])
            xs = np.concatenate([xs, xs[extra]])
    elif component_balance and components and labels is not None:
        component_choices = rng.integers(0, len(components), size=int(num_points), endpoint=False)
        ys = np.empty((int(num_points),), dtype=np.int64)
        xs = np.empty((int(num_points),), dtype=np.int64)
        for comp_idx, label in enumerate(components):
            out_mask = component_choices == comp_idx
            count = int(np.sum(out_mask))
            if count == 0:
                continue
            comp_y, comp_x = np.nonzero(labels == label)
            comp_prob = prob[comp_y, comp_x].astype(np.float64)
            comp_prob = comp_prob / max(float(np.sum(comp_prob)), 1e-12)
            picks = rng.choice(len(comp_x), size=count, replace=True, p=comp_prob)
            ys[out_mask] = comp_y[picks]
            xs[out_mask] = comp_x[picks]
    else:
        flat = (prob / max(float(np.sum(prob)), 1e-12)).reshape(-1)
        picks = rng.choice(flat.size, size=int(num_points), replace=True, p=flat)
        ys, xs = np.divmod(picks, width)

    points = _grid_indices_to_unit_points_numpy(ys[: int(num_points)], xs[: int(num_points)], height=height, width=width, rng=rng, jitter_scale=jitter_scale)
    masses = np.full((int(num_points),), 1.0 / float(num_points), dtype=np.float64)
    return points.astype(np.float64), masses



def sample_points_from_decoded_raster(
    decoded_rasters: np.ndarray,
    *,
    num_points: int,
    channel: int = 0,
    component_balance: bool = True,
    threshold_quantile: float = 0.75,
    min_component_pixels: int = 3,
    sampler_mode: str = "random",
    corner_weight: float = 0.0,
    jitter_scale: float = 0.45,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample pseudo-target point clouds from decoded latent rasters."""
    rasters = np.asarray(decoded_rasters, dtype=np.float64)
    if rasters.ndim != 4:
        raise ValueError("decoded_rasters must have shape (N, C, H, W)")
    if channel < 0 or channel >= rasters.shape[1]:
        raise ValueError("channel out of range")
    rng = np.random.default_rng(int(seed))
    clouds: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    for raster in rasters:
        pts, ms = _sample_points_from_probability_grid_numpy(
            raster[int(channel)],
            num_points=int(num_points),
            component_balance=component_balance,
            threshold_quantile=threshold_quantile,
            min_component_pixels=min_component_pixels,
            rng=rng,
            sampler_mode=sampler_mode,
            corner_weight=corner_weight,
            jitter_scale=jitter_scale,
        )
        clouds.append(pts)
        masses.append(ms)
    return np.stack(clouds, axis=0), np.stack(masses, axis=0)


@torch.no_grad()
def _decoded_raster_pseudo_targets_torch(
    model: TargetConditionedScoreModel,
    target_latents: Tensor,
    *,
    num_points: int,
    channel: int = 0,
    component_balance: bool = True,
    threshold_quantile: float = 0.75,
    min_component_pixels: int = 3,
    sampler_mode: str = "random",
    corner_weight: float = 0.0,
    jitter_scale: float = 0.45,
) -> tuple[Tensor, Tensor]:
    if getattr(model, "latent_raster_decoder", None) is None:
        raise RuntimeError("decoded-raster guidance requires model.latent_raster_decoder")
    decoded = model.predict_target_raster_from_latent(target_latents).detach().cpu().numpy()
    pts, masses = sample_points_from_decoded_raster(
        decoded,
        num_points=int(num_points),
        channel=int(channel),
        component_balance=component_balance,
        threshold_quantile=float(threshold_quantile),
        min_component_pixels=int(min_component_pixels),
        sampler_mode=sampler_mode,
        corner_weight=float(corner_weight),
        jitter_scale=float(jitter_scale),
        seed=0,
    )
    return (
        torch.as_tensor(pts, dtype=target_latents.dtype, device=target_latents.device),
        torch.as_tensor(masses, dtype=target_latents.dtype, device=target_latents.device),
    )


@torch.no_grad()
def decoded_raster_topology_diagnostics(
    model: TargetConditionedScoreModel,
    masses: np.ndarray,
    target_positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    channel: int = 0,
    threshold: float = 0.5,
    max_samples: Optional[int] = None,
    batch_size: int = 64,
    device: Optional[str | torch.device] = None,
) -> dict[str, float]:
    """Compare decoded latent rasters against true target-contour topology."""
    masses_arr = np.asarray(masses, dtype=np.float32)
    positions_arr = np.asarray(target_positions, dtype=np.float32)
    if masses_arr.ndim != 2 or positions_arr.shape != (*masses_arr.shape, 2):
        raise ValueError("masses and target_positions must have shapes (N,K), (N,K,2)")
    labels_arr = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)
    n = masses_arr.shape[0]
    if max_samples is not None:
        n = min(n, int(max_samples))
        masses_arr = masses_arr[:n]
        positions_arr = positions_arr[:n]
        labels_arr = None if labels_arr is None else labels_arr[:n]
    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    decoded_summaries: list[dict[str, float]] = []
    target_summaries: list[dict[str, float]] = []
    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        batch_masses = torch.from_numpy(masses_arr[start:stop]).to(model_device)
        batch_positions = torch.from_numpy(positions_arr[start:stop]).to(model_device)
        z = model.encode_target(batch_masses, batch_positions)
        if getattr(model, "latent_raster_decoder", None) is None:
            raise RuntimeError("model must have latent_raster_decoder enabled")
        decoded = model.predict_target_raster_from_latent(z).detach().cpu().numpy()
        for local_index, raster in enumerate(decoded):
            contour = raster[int(channel)] >= float(threshold)
            decoded_summaries.append(_raster_topology_summary_from_mask(contour))
            target_summaries.append(raster_topology_summary(positions_arr[start + local_index], image_size=raster.shape[-1]))
    if was_training:
        model.train()
    def _mean(key: str, rows: list[dict[str, float]]) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else float("nan")
    hole_match = [float(a["hole_count"] == b["hole_count"]) for a, b in zip(decoded_summaries, target_summaries)]
    comp_match = [float(a["component_count"] == b["component_count"]) for a, b in zip(decoded_summaries, target_summaries)]
    return {
        "decoded_hole_count_mean": _mean("hole_count", decoded_summaries),
        "target_hole_count_mean": _mean("hole_count", target_summaries),
        "decoded_component_count_mean": _mean("component_count", decoded_summaries),
        "target_component_count_mean": _mean("component_count", target_summaries),
        "decoded_hole_count_accuracy": float(np.mean(hole_match)) if hole_match else float("nan"),
        "decoded_component_count_accuracy": float(np.mean(comp_match)) if comp_match else float("nan"),
        "decoded_occupied_fraction_mean": _mean("occupied_fraction", decoded_summaries),
        "target_occupied_fraction_mean": _mean("occupied_fraction", target_summaries),
        "num_samples": float(n),
    }


def topology_diagnostics(
    generated_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    image_size: int = 64,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> dict[str, float]:
    """Compare raster topology of generated and target MNIST-CP contours."""
    gen = np.asarray(generated_positions, dtype=np.float64)
    tgt = np.asarray(target_positions, dtype=np.float64)
    if gen.ndim != 3 or tgt.ndim != 3 or gen.shape[0] != tgt.shape[0]:
        raise ValueError("generated_positions and target_positions must have matching batches")
    rng = np.random.default_rng(int(seed))
    if max_samples is not None and gen.shape[0] > int(max_samples):
        idx = rng.choice(gen.shape[0], size=int(max_samples), replace=False)
        gen = gen[idx]
        tgt = tgt[idx]
    gen_holes: list[float] = []
    tgt_holes: list[float] = []
    gen_components: list[float] = []
    tgt_components: list[float] = []
    hole_match: list[float] = []
    comp_match: list[float] = []
    occupied_ratio: list[float] = []
    hole_leakage: list[float] = []
    for g, t in zip(gen, tgt):
        gs = raster_topology_summary(g, image_size=image_size)
        ts = raster_topology_summary(t, image_size=image_size)
        gen_holes.append(gs["hole_count"])
        tgt_holes.append(ts["hole_count"])
        gen_components.append(gs["component_count"])
        tgt_components.append(ts["component_count"])
        hole_match.append(float(gs["hole_count"] == ts["hole_count"]))
        comp_match.append(float(gs["component_count"] == ts["component_count"]))
        occupied_ratio.append(gs["occupied_fraction"] / max(ts["occupied_fraction"], 1e-12))
        target_contour = _rasterize_points_binary(t, image_size=image_size, dilation=1)
        target_holes = _hole_mask_numpy(target_contour, dilation=1)
        gen_contour = _rasterize_points_binary(g, image_size=image_size, dilation=1)
        hole_leakage.append(float(np.mean(gen_contour[target_holes])) if np.any(target_holes) else 0.0)
    return {
        "generated_hole_count_mean": float(np.mean(gen_holes)),
        "target_hole_count_mean": float(np.mean(tgt_holes)),
        "hole_count_accuracy": float(np.mean(hole_match)),
        "generated_component_count_mean": float(np.mean(gen_components)),
        "target_component_count_mean": float(np.mean(tgt_components)),
        "component_count_accuracy": float(np.mean(comp_match)),
        "occupied_fraction_ratio_mean": float(np.mean(occupied_ratio)),
        "hole_leakage_mean": float(np.mean(hole_leakage)),
        "num_samples": float(gen.shape[0]),
    }


def contour_thickness_diagnostics(
    positions: np.ndarray,
    *,
    masses: Optional[np.ndarray] = None,
    target_positions: Optional[np.ndarray] = None,
    target_masses: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
    image_size: int = 28,
    occupancy_threshold: float = 1e-6,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> dict[str, float]:
    """Contour-thickness diagnostics for MNIST-CP point clouds.

    Chamfer distance can say a cloud is close to the target while the generated
    particles are still too thick or fuzzy.  These diagnostics measure internal
    spacing, raster occupancy, and optionally the oracle projection error at a
    chosen small sigma.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[2] != 2:
        raise ValueError("positions must have shape (N,K,2)")
    rng = np.random.default_rng(int(seed))
    if max_samples is not None and pos.shape[0] > int(max_samples):
        idx = rng.choice(pos.shape[0], size=int(max_samples), replace=False)
        pos = pos[idx]
        if masses is not None:
            masses = np.asarray(masses, dtype=np.float64)[idx]
        if target_positions is not None:
            target_positions = np.asarray(target_positions, dtype=np.float64)[idx]
        if target_masses is not None:
            target_masses = np.asarray(target_masses, dtype=np.float64)[idx]
    if masses is None:
        masses_arr = _uniform_masses(pos.shape[0], pos.shape[1], dtype=np.float64)
    else:
        masses_arr = np.asarray(masses, dtype=np.float64)
        if masses_arr.shape != pos.shape[:2]:
            raise ValueError("masses must have shape (N,K)")

    nearest_values: list[np.ndarray] = []
    for cloud in pos:
        diff = cloud[:, None, :] - cloud[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(d2, np.inf)
        nearest_values.append(np.sqrt(np.min(d2, axis=1)))
    nearest = np.concatenate(nearest_values)
    raster = rasterize_weighted_point_clouds(
        masses_arr,
        np.asarray(project_positions(pos, mode="clip"), dtype=np.float64),
        image_size=int(image_size),
    )
    flat = raster.reshape(raster.shape[0], -1)
    occupied = np.mean(flat > float(occupancy_threshold), axis=1)
    out: dict[str, float] = {
        "self_nn_mean": float(np.mean(nearest)),
        "self_nn_std": float(np.std(nearest)),
        "self_nn_median": float(np.median(nearest)),
        "raster_occupied_fraction_mean": float(np.mean(occupied)),
        "raster_occupied_fraction_std": float(np.std(occupied)),
        "num_samples": float(pos.shape[0]),
    }

    if target_positions is not None and sigma is not None and float(sigma) > 0.0:
        target_pos = np.asarray(target_positions, dtype=np.float64)
        if target_pos.ndim != 3 or target_pos.shape[0] != pos.shape[0] or target_pos.shape[2] != 2:
            raise ValueError("target_positions must have shape (N,M,2) with matching batch")
        if target_masses is None:
            target_masses_arr = _uniform_masses(target_pos.shape[0], target_pos.shape[1], dtype=np.float64)
        else:
            target_masses_arr = np.asarray(target_masses, dtype=np.float64)
        with torch.no_grad():
            q = torch.tensor(pos, dtype=torch.float32)
            t = torch.tensor(target_pos, dtype=torch.float32)
            tm = torch.tensor(target_masses_arr, dtype=torch.float32)
            _, posterior = empirical_gaussian_mixture_physical_score(
                q,
                t,
                float(sigma),
                target_masses=tm,
                return_posterior_mean=True,
            )
            posterior_np = posterior.detach().cpu().numpy().astype(np.float64)
        projection_rmse = np.sqrt(np.mean(np.sum((posterior_np - pos) ** 2, axis=-1)))
        out["oracle_projection_rmse"] = float(projection_rmse)
        out["oracle_projection_mean_distance"] = float(np.mean(np.linalg.norm(posterior_np - pos, axis=-1)))
    return out


@torch.no_grad()
def evaluate_hybrid_oracle_neural_reconstruction(
    model: TargetConditionedScoreModel,
    target_masses: np.ndarray,
    target_positions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    tau_levels: Sequence[float] | np.ndarray,
    prefix_levels: Sequence[int] = (0, 1, 2, 3, 5),
    suffix_levels: Sequence[int] = (),
    max_samples: int = 32,
    steps_per_level: int = 10,
    sampler_scheme: str = "shape_gf_langevin",
    initial_position_mode: str = "uniform",
    state_projection: str = "none",
    score_scale: float = 1.0,
    diffusion_temperature: float = 1.0,
    langevin_alpha: float = 5e-5,
    score_calibration: Optional[ScoreCalibration | dict[str, Any]] = None,
    score_norm_clip: Optional[float | Sequence[float] | np.ndarray] = None,
    batch_size: int = 64,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
    squared_chamfer: bool = True,
) -> list[dict[str, float | int | str]]:
    """Run oracle/neural hybrid reconstructions to localize sampler failure.

    ``prefix_levels=m`` uses the oracle for the first ``m`` high-noise levels and
    the neural score afterwards.  ``suffix_levels=m`` does the reverse: neural
    first, oracle for the last ``m`` low-noise levels.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    masses_arr = np.asarray(target_masses, dtype=np.float64)
    positions_arr = np.asarray(target_positions, dtype=np.float64)
    if masses_arr.ndim != 2 or positions_arr.shape != (*masses_arr.shape, 2):
        raise ValueError("target_masses and target_positions must have shapes (N,K), (N,K,2)")
    n = min(int(max_samples), int(masses_arr.shape[0]))
    labels_arr = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1)[:n]
    levels = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if levels.size == 0:
        raise ValueError("tau_levels must be non-empty")
    levels = np.sort(levels)[::-1]
    rows: list[dict[str, float | int | str]] = []
    base_rng = np.random.default_rng() if rng is None else rng

    def _run(mode: str, count: int) -> None:
        count_clamped = max(0, min(int(count), len(levels)))
        child_rng = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
        kwargs: dict[str, Any] = {
            "target_masses": masses_arr[:n],
            "target_positions": positions_arr[:n],
            "labels": labels_arr,
            "tau_levels": levels,
            "steps_per_level": steps_per_level,
            "sampler_scheme": sampler_scheme,
            "initial_position_mode": initial_position_mode,
            "state_projection": state_projection,
            "score_scale": score_scale,
            "diffusion_temperature": diffusion_temperature,
            "langevin_alpha": langevin_alpha,
            "score_calibration": score_calibration,
            "score_norm_clip": score_norm_clip,
            "batch_size": batch_size,
            "device": device,
            "rng": child_rng,
        }
        if mode == "oracle_prefix":
            kwargs["oracle_prefix_levels"] = count_clamped
        elif mode == "oracle_suffix":
            kwargs["oracle_suffix_levels"] = count_clamped
        else:
            raise ValueError("unknown hybrid mode")
        recon = reconstruct_target_conditioned_point_clouds(model, **kwargs)
        metrics = paired_chamfer_reconstruction_metrics(
            recon.positions,
            positions_arr[:n],
            labels_arr,
            squared=squared_chamfer,
        )
        rows.append(
            {
                "mode": mode,
                "oracle_levels": int(count_clamped),
                "mean_chamfer": float(metrics["mean_chamfer"]),
                "median_chamfer": float(metrics["median_chamfer"]),
                "std_chamfer": float(metrics["std_chamfer"]),
                "num_samples": int(n),
            }
        )

    for count in prefix_levels:
        _run("oracle_prefix", int(count))
    for count in suffix_levels:
        _run("oracle_suffix", int(count))
    return rows


# Backward-compatible aliases used by the Experiment 8b notebook/tests.
evaluate_model_against_empirical_mixture_score = evaluate_model_vs_mixture_oracle
evaluate_model_against_mixture_oracle = evaluate_model_vs_mixture_oracle
