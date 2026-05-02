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
from typing import Any, Optional, Sequence

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

__all__ = [
    "TargetConditionedScoreModel",
    "TargetPointCloudEncoder",
    "NoisySetEquivariantEncoder",
    "GaussianLatentPrior",
    "LatentGenerator",
    "LatentCritic",
    "make_sigma_tau_schedule",
    "tau_levels_from_sigma_levels",
    "perturb_target_conditioned_positions",
    "target_conditioned_score_matching_loss",
    "train_target_conditioned_score_model",
    "evaluate_target_conditioned_score_model",
    "encode_target_latents",
    "sample_target_conditioned_annealed_dynamics",
    "reconstruct_target_conditioned_point_clouds",
    "fit_gaussian_latent_prior",
    "sample_gaussian_latent_prior",
    "train_latent_wgan_gp",
    "sample_wgan_latent_prior",
    "paired_chamfer_reconstruction_metrics",
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
        self.set_encoder = NoisySetEquivariantEncoder(
            hidden_dim=set_hidden_dim,
            output_dim=set_feature_dim,
            context_dim=context_dim,
            num_blocks=set_blocks,
            dropout=dropout,
            use_fourier_features=use_fourier_features,
        )

        position_dim = 6 if self.use_fourier_features else 2
        score_input_dim = position_dim + 2 + local_feature_dim + set_feature_dim + context_dim
        score_layers: list[nn.Module] = [nn.Linear(score_input_dim, score_hidden_dim), nn.GELU()]
        for _ in range(int(score_residual_blocks)):
            score_layers.append(_ResidualMLPBlock(score_hidden_dim, dropout=dropout))
        score_layers.extend([nn.LayerNorm(score_hidden_dim), nn.Linear(score_hidden_dim, 2)])
        self.score_head = nn.Sequential(*score_layers)

    def encode_target(self, target_masses: Tensor, target_positions: Tensor) -> Tensor:
        return self.target_encoder(target_masses, target_positions)

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
        latents = self._resolve_latents(
            masses,
            target_positions,
            target_masses=target_masses,
            target_latents=target_latents,
        )
        context = self._prepare_context(tau_tensor, latents, labels)

        feature_pieces = [
            _position_features(positions, use_fourier=self.use_fourier_features),
            _mass_features(masses),
        ]
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
        context_points = context[:, None, :].expand(batch_size, num_points, context.shape[1])
        feature_pieces.extend([set_features, context_points])
        point_inputs = torch.cat(feature_pieces, dim=-1)
        return self.score_head(point_inputs)

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    device: Optional[str | torch.device] = None,
) -> dict[str, float]:
    """Estimate validation scaled-score DSM loss."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    tau_levels_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if tau_levels_arr.size == 0:
        raise ValueError("tau_levels must be non-empty")

    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    loader = _make_tensor_loader(masses, positions, labels, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_sample_loss = 0.0
    total_zero_loss = 0.0
    total_items = 0
    tau_means: list[float] = []
    for batch_masses, batch_positions, batch_labels in loader:
        batch_masses = batch_masses.to(model_device)
        batch_positions = batch_positions.to(model_device)
        batch_labels = batch_labels.to(model_device)
        tau = _sample_tau_from_levels(
            int(batch_masses.shape[0]),
            tau_levels_arr,
            device=model_device,
            dtype=batch_positions.dtype,
        )
        noisy, target_scaled, _ = perturb_target_conditioned_positions(
            batch_masses,
            batch_positions,
            tau,
            projection=projection,
        )
        pred_scaled = model.predict_scaled_score(
            batch_masses,
            noisy,
            tau,
            target_positions=batch_positions,
            target_masses=batch_masses,
            labels=batch_labels,
        )
        loss, metrics = target_conditioned_score_matching_loss(
            pred_scaled,
            target_scaled,
            batch_masses,
            tau,
            time_weighting=time_weighting,
        )
        zero_loss, _ = target_conditioned_score_matching_loss(
            torch.zeros_like(target_scaled),
            target_scaled,
            batch_masses,
            tau,
            time_weighting=time_weighting,
        )
        bsz = int(batch_masses.shape[0])
        total_loss += float(loss.item()) * bsz
        total_sample_loss += float(metrics["sample_loss"]) * bsz
        total_zero_loss += float(zero_loss.item()) * bsz
        total_items += bsz
        tau_means.append(float(metrics["mean_tau"]))

    if was_training:
        model.train()
    return {
        "loss": total_loss / max(total_items, 1),
        "sample_loss": total_sample_loss / max(total_items, 1),
        "zero_loss": total_zero_loss / max(total_items, 1),
        "loss_ratio_vs_zero": (total_loss / max(total_zero_loss, 1e-12)),
        "mean_tau": float(np.mean(tau_means)) if tau_means else float("nan"),
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
    max_grad_norm: Optional[float] = 5.0,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train ``f_phi`` and ``S_theta`` by target-conditioned DSM."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    tau_levels_arr = np.asarray(tau_levels, dtype=np.float64).reshape(-1)
    if tau_levels_arr.size == 0 or not np.all(np.isfinite(tau_levels_arr)) or np.any(tau_levels_arr <= 0.0):
        raise ValueError("tau_levels must be positive and finite")

    model_device = _resolve_device(device)
    model = model.to(model_device)
    model.train()
    loader = _make_tensor_loader(masses, positions, labels, batch_size=batch_size, shuffle=True)

    enc_lr = lr if encoder_lr is None else float(encoder_lr)
    optimizer = torch.optim.Adam(
        [
            {"params": model.target_encoder.parameters(), "lr": enc_lr},
            {"params": [p for name, p in model.named_parameters() if not name.startswith("target_encoder.")], "lr": lr},
        ],
        weight_decay=weight_decay,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_sample_loss": [],
        "val_loss": [],
        "val_loss_ratio_vs_zero": [],
    }
    for epoch in range(int(epochs)):
        total_loss = 0.0
        total_sample_loss = 0.0
        total_items = 0
        for batch_masses, batch_positions, batch_labels in loader:
            batch_masses = batch_masses.to(model_device)
            batch_positions = batch_positions.to(model_device)
            batch_labels = batch_labels.to(model_device)
            tau = _sample_tau_from_levels(
                int(batch_masses.shape[0]),
                tau_levels_arr,
                device=model_device,
                dtype=batch_positions.dtype,
            )
            noisy, target_scaled, _ = perturb_target_conditioned_positions(
                batch_masses,
                batch_positions,
                tau,
                projection=projection,
            )
            pred_scaled = model.predict_scaled_score(
                batch_masses,
                noisy,
                tau,
                target_positions=batch_positions,
                target_masses=batch_masses,
                labels=batch_labels,
            )
            loss, metrics = target_conditioned_score_matching_loss(
                pred_scaled,
                target_scaled,
                batch_masses,
                tau,
                time_weighting=time_weighting,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            optimizer.step()

            bsz = int(batch_masses.shape[0])
            total_loss += float(loss.item()) * bsz
            total_sample_loss += float(metrics["sample_loss"]) * bsz
            total_items += bsz

        train_loss = total_loss / max(total_items, 1)
        train_sample_loss = total_sample_loss / max(total_items, 1)
        history["train_loss"].append(train_loss)
        history["train_sample_loss"].append(train_sample_loss)

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
                device=model_device,
            )
            val_loss = float(val_metrics["loss"])
            val_ratio = float(val_metrics["loss_ratio_vs_zero"])
        else:
            val_loss = float("nan")
            val_ratio = float("nan")
        history["val_loss"].append(val_loss)
        history["val_loss_ratio_vs_zero"].append(val_ratio)
        model.train()

        if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch + 1 == epochs):
            print(
                f"epoch {epoch + 1:04d}/{epochs}: "
                f"train_loss={train_loss:.6g} val_loss={val_loss:.6g} val/zero={val_ratio:.4f}"
            )
    return history


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
        z = model.encode_target(batch_masses.to(model_device), batch_positions.to(model_device))
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
    sampler_scheme: str = "theory_euler",
    initial_position_mode: str = "uniform",
    initial_position_scale: float = 0.20,
    state_projection: str = "clip",
    score_scale: float = 1.0,
    diffusion_temperature: float = 0.30,
    final_polish_steps: int = 5,
    batch_size: int = 64,
    rasterize: bool = False,
    image_size: int = 28,
    return_trajectories: bool = False,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedPointCloudSet:
    """Sample/reconstruct contours using target-conditioned annealed dynamics."""
    if steps_per_level <= 0:
        raise ValueError("steps_per_level must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if score_scale <= 0.0 or diffusion_temperature < 0.0:
        raise ValueError("score_scale must be positive and diffusion_temperature must be non-negative")
    if sampler_scheme not in {"theory_euler", "bridge", "langevin"}:
        raise ValueError("sampler_scheme must be one of {'theory_euler', 'bridge', 'langevin'}")
    rng = np.random.default_rng() if rng is None else rng

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

    levels = _resolve_tau_levels_for_sampling(
        tau_levels=tau_levels,
        sigma_levels=sigma_levels,
        num_points=num_points_resolved,
    )

    initial_positions_np = _initial_positions_for_sampling(
        num_samples,
        num_points_resolved,
        mode=initial_position_mode,
        scale=initial_position_scale,
        rng=rng,
    )
    initial_positions_np = np.asarray(project_positions(initial_positions_np, mode=state_projection), dtype=np.float64)

    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()

    masses = torch.from_numpy(masses_np.astype(np.float32)).to(model_device)
    positions = torch.from_numpy(initial_positions_np.astype(np.float32)).to(model_device)
    label_tensor = torch.from_numpy(labels_arr).to(model_device)

    if latents_arr is not None:
        latents = torch.from_numpy(latents_arr).to(model_device)
    else:
        target_masses_tensor = masses if target_masses is None else torch.from_numpy(np.asarray(target_masses, dtype=np.float32)).to(model_device)
        target_positions_tensor = torch.from_numpy(np.asarray(target_positions, dtype=np.float32)).to(model_device)
        latent_batches: list[Tensor] = []
        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            latent_batches.append(model.encode_target(target_masses_tensor[start:stop], target_positions_tensor[start:stop]))
        latents = torch.cat(latent_batches, dim=0)

    trajectory_snapshots: list[np.ndarray] = []
    if return_trajectories:
        trajectory_snapshots.append(positions.detach().cpu().numpy().astype(np.float64))

    if sampler_scheme == "bridge":
        update_pairs = [(float(levels[i]), float(levels[i + 1]) if i + 1 < len(levels) else 0.0) for i in range(len(levels))]
        for tau_value, tau_next in update_pairs:
            for start in range(0, num_samples, batch_size):
                stop = min(start + batch_size, num_samples)
                batch_masses = masses[start:stop]
                batch_positions = positions[start:stop]
                tau = torch.full((stop - start,), tau_value, device=model_device, dtype=batch_positions.dtype)
                score = model(
                    batch_masses,
                    batch_positions,
                    tau,
                    target_latents=latents[start:stop],
                    labels=label_tensor[start:stop],
                )
                clean_estimate = batch_positions + 2.0 * float(score_scale) * tau[:, None, None] * score
                if tau_next > 0.0 and diffusion_temperature > 0.0:
                    noise_scale = torch.sqrt((2.0 * tau_next * diffusion_temperature) / batch_masses).unsqueeze(-1)
                    batch_positions = clean_estimate + noise_scale * torch.randn_like(clean_estimate)
                else:
                    batch_positions = clean_estimate
                batch_positions = project_positions(batch_positions, mode=state_projection)
                positions[start:stop] = batch_positions
            if return_trajectories:
                trajectory_snapshots.append(positions.detach().cpu().numpy().astype(np.float64))
    else:
        for level_id, tau_level in enumerate(levels):
            tau_next_level = float(levels[level_id + 1]) if level_id + 1 < len(levels) else 0.0
            if sampler_scheme == "theory_euler":
                dt = max((float(tau_level) - tau_next_level) / float(steps_per_level), float(tau_level) * 1e-4)
            else:  # ShapeGF-style Langevin inner loop at fixed noise level.
                dt = max(0.02 * float(tau_level), 1e-12)
            for inner in range(int(steps_per_level)):
                if sampler_scheme == "theory_euler":
                    tau_value = max(float(tau_level) - inner * dt, max(tau_next_level, 1e-12))
                else:
                    tau_value = float(tau_level)
                for start in range(0, num_samples, batch_size):
                    stop = min(start + batch_size, num_samples)
                    batch_masses = masses[start:stop]
                    batch_positions = positions[start:stop]
                    tau = torch.full((stop - start,), tau_value, device=model_device, dtype=batch_positions.dtype)
                    score = model(
                        batch_masses,
                        batch_positions,
                        tau,
                        target_latents=latents[start:stop],
                        labels=label_tensor[start:stop],
                    )
                    noise_scale = torch.sqrt((2.0 * diffusion_temperature * dt) / batch_masses).unsqueeze(-1)
                    batch_positions = (
                        batch_positions
                        + 2.0 * float(score_scale) * dt * score
                        + noise_scale * torch.randn_like(batch_positions)
                    )
                    batch_positions = project_positions(batch_positions, mode=state_projection)
                    positions[start:stop] = batch_positions
            if return_trajectories:
                trajectory_snapshots.append(positions.detach().cpu().numpy().astype(np.float64))

    if final_polish_steps > 0:
        tau_value = float(levels[-1])
        dt = max(tau_value / float(max(final_polish_steps, 1)), 1e-12)
        for _ in range(int(final_polish_steps)):
            for start in range(0, num_samples, batch_size):
                stop = min(start + batch_size, num_samples)
                batch_masses = masses[start:stop]
                batch_positions = positions[start:stop]
                tau = torch.full((stop - start,), tau_value, device=model_device, dtype=batch_positions.dtype)
                score = model(
                    batch_masses,
                    batch_positions,
                    tau,
                    target_latents=latents[start:stop],
                    labels=label_tensor[start:stop],
                )
                batch_positions = batch_positions + 2.0 * float(score_scale) * dt * score
                batch_positions = project_positions(batch_positions, mode=state_projection)
                positions[start:stop] = batch_positions
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


def fit_gaussian_latent_prior(
    latents: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    diagonal: bool = True,
    eps: float = 1e-4,
) -> GaussianLatentPrior:
    """Fit a simple latent prior.  With labels, one Gaussian is fit per class."""
    z = np.asarray(latents, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] == 0:
        raise ValueError("latents must have shape (N, D) with N > 0")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if labels is None:
        component_labels = None
        groups = [np.arange(z.shape[0])]
    else:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_arr.shape != (z.shape[0],):
            raise ValueError("labels must have shape (N,)")
        component_labels = np.unique(labels_arr).astype(np.int64)
        groups = [np.flatnonzero(labels_arr == label) for label in component_labels]

    means = []
    covariances = []
    for idx in groups:
        if len(idx) == 0:
            raise ValueError("empty latent-prior component")
        group = z[idx]
        means.append(np.mean(group, axis=0))
        centered = group - means[-1]
        if diagonal:
            var = np.var(centered, axis=0) + float(eps)
            covariances.append(var)
        else:
            if len(group) <= 1:
                cov = np.eye(z.shape[1], dtype=np.float64) * float(eps)
            else:
                cov = np.cov(group, rowvar=False) + np.eye(z.shape[1], dtype=np.float64) * float(eps)
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
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample latent codes and their labels/components from a Gaussian prior."""
    rng = np.random.default_rng() if rng is None else rng
    if labels is None:
        if num_samples is None:
            raise ValueError("num_samples is required when labels are not provided")
        if prior.labels is None:
            component_idx = np.zeros(int(num_samples), dtype=np.int64)
            out_labels = np.zeros(int(num_samples), dtype=np.int64)
        else:
            component_idx = rng.integers(0, prior.num_components, size=int(num_samples), endpoint=False)
            out_labels = prior.labels[component_idx]
    else:
        out_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        component_idx = np.zeros(len(out_labels), dtype=np.int64)
        if prior.labels is not None:
            label_to_component = {int(label): i for i, label in enumerate(prior.labels)}
            for i, label in enumerate(out_labels):
                if int(label) not in label_to_component:
                    raise ValueError(f"label {int(label)} is not available in the latent prior")
                component_idx[i] = label_to_component[int(label)]
        elif prior.num_components != 1:
            raise ValueError("unlabeled prior must have exactly one component")
    samples = np.empty((len(out_labels), prior.latent_dim), dtype=np.float64)
    for i, comp in enumerate(component_idx):
        mean = prior.means[int(comp)]
        cov = prior.covariances[int(comp)]
        if prior.diagonal:
            samples[i] = mean + np.sqrt(cov) * rng.normal(size=prior.latent_dim)
        else:
            samples[i] = rng.multivariate_normal(mean, cov)
    return samples, out_labels.astype(np.int64, copy=False)


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
