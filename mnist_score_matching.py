from __future__ import annotations

r"""Score-matching generation for MNIST weighted point clouds.

This module mirrors the weighted-point-cloud representation used in Example 6,
but replaces the terminal ``h``-transform / classifier guidance with the
denoising score-matching procedure described in the companion note.

For a clean configuration

    X = (X_1, ..., X_n),

with frozen masses ``s = (s_1, ..., s_n)``, we perturb the positions by the free
particle semigroup

    Y_i = X_i + sqrt(2 tau / s_i) Z_i,

and learn the score of the noisy marginal

    D_y^(n) log rho_{tau,s}(y).

For ``projection='none'`` this uses the Euclidean-cover denoising target

    D_y^(n) log p_tau,s(y | X) = -(y - X) / (2 tau).

For ``projection='wrap'`` it uses the corresponding wrapped heat-kernel score
on the unit torus, evaluated by convergent image/Fourier series.
"""

from typing import Any, Optional, Sequence

import copy
import math

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mnist_conditioned_diffusion import (
    GeneratedPointCloudSet,
    WeightedPointCloudDataset,
    _resolve_device,
    draw_joint_mass_position_vectors_from_bank,
    draw_mass_vectors_from_bank,
    project_positions,
    rasterize_weighted_point_clouds,
    sample_initial_positions,
)
from mnist_experiment6_fixes import (
    draw_position_vectors_from_bank,
    sample_truncated_poisson_dirichlet_masses,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "ConditionalScoreSetNetwork",
    "ConditionalScoreSetTransformer",
    "ConditionalScoreImageFieldNetwork",
    "sample_score_matching_noise_levels",
    "add_forward_noise_to_positions",
    "torus_heat_kernel_score_target",
    "perturb_weighted_point_cloud_positions",
    "weighted_denoising_score_matching_loss",
    "evaluate_score_model",
    "evaluate_score_model_by_tau_bins",
    "train_score_model",
    "diagnose_score_prior_horizons",
    "recommend_score_prior_horizon",
    "bridge_reverse_step",
    "generate_score_matching_point_clouds",
    "generate_balanced_score_matching_dataset",
]


# ---------------------------------------------------------------------------
# Score network
# ---------------------------------------------------------------------------


def _tau_features(
    tau: Tensor,
    *,
    tau_min: float,
    tau_max: float,
    eps: float = 1e-12,
) -> Tensor:
    """Build a small time-conditioning feature vector.

    The score varies rapidly near ``tau = 0``, so we expose both linear and
    logarithmic features of the normalized time variable.
    """
    if tau.ndim != 1:
        raise ValueError("tau must have shape (B,)")
    if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
        raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")

    tau_clamped = tau.clamp_min(float(tau_min))
    denom = max(float(tau_max - tau_min), eps)
    tau_unit = ((tau_clamped - float(tau_min)) / denom).clamp(0.0, 1.0)
    log_tau = torch.log(tau_clamped / float(tau_min))
    log_denom = max(math.log(float(tau_max) / float(tau_min)), eps)
    log_tau_unit = (log_tau / log_denom).clamp(0.0, 1.0)
    sqrt_tau_unit = torch.sqrt(tau_unit.clamp_min(0.0))
    return torch.stack([tau_unit, sqrt_tau_unit, log_tau_unit], dim=1)




def _group_norm_groups(num_channels: int, max_groups: int = 8) -> int:
    """Choose a small GroupNorm group count that divides ``num_channels``."""
    for groups in reversed(range(1, max_groups + 1)):
        if groups <= num_channels and num_channels % groups == 0:
            return groups
    return 1


def _normalize_positions_for_grid(positions: Tensor, *, periodic: bool) -> Tensor:
    """Map particle positions into the unit square used by the raster grid."""
    if periodic:
        return torch.remainder(positions, 1.0)
    eps = max(torch.finfo(positions.dtype).eps, 1e-6)
    return positions.clamp(0.0, 1.0 - eps)


def _grid_bilinear_coordinates(
    positions: Tensor,
    *,
    height: int,
    width: int,
    periodic: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return bilinear neighbor indices and weights for unit-square positions."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    positions = _normalize_positions_for_grid(positions, periodic=periodic)

    x = positions[..., 0] * float(width) - 0.5
    y = positions[..., 1] * float(height) - 0.5

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    wx1 = x - x0
    wy1 = y - y0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    x0_idx = x0.to(dtype=torch.long)
    y0_idx = y0.to(dtype=torch.long)
    x1_idx = x1.to(dtype=torch.long)
    y1_idx = y1.to(dtype=torch.long)

    if periodic:
        x0_idx = torch.remainder(x0_idx, width)
        x1_idx = torch.remainder(x1_idx, width)
        y0_idx = torch.remainder(y0_idx, height)
        y1_idx = torch.remainder(y1_idx, height)
    else:
        x0_idx = x0_idx.clamp(0, width - 1)
        x1_idx = x1_idx.clamp(0, width - 1)
        y0_idx = y0_idx.clamp(0, height - 1)
        y1_idx = y1_idx.clamp(0, height - 1)

    w00 = wx0 * wy0
    w01 = wx1 * wy0
    w10 = wx0 * wy1
    w11 = wx1 * wy1
    return x0_idx, x1_idx, y0_idx, y1_idx, w00, w01, w10, w11


def _rasterize_weighted_point_clouds_torch(
    masses: Tensor,
    positions: Tensor,
    *,
    grid_size: int,
    periodic: bool,
    include_occupancy: bool = True,
) -> Tensor:
    """Rasterize weighted point clouds into density/occupancy image channels."""
    if masses.ndim != 2:
        raise ValueError("masses must have shape (B, K)")
    if positions.ndim != 3 or positions.shape[:2] != masses.shape or positions.shape[2] != 2:
        raise ValueError("positions must have shape (B, K, 2) and match masses")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    batch_size, num_points = masses.shape
    height = width = int(grid_size)
    num_cells = height * width

    x0_idx, x1_idx, y0_idx, y1_idx, w00, w01, w10, w11 = _grid_bilinear_coordinates(
        positions,
        height=height,
        width=width,
        periodic=periodic,
    )

    idx00 = y0_idx * width + x0_idx
    idx01 = y0_idx * width + x1_idx
    idx10 = y1_idx * width + x0_idx
    idx11 = y1_idx * width + x1_idx

    density = masses.new_zeros((batch_size, num_cells))
    density.scatter_add_(1, idx00, masses * w00)
    density.scatter_add_(1, idx01, masses * w01)
    density.scatter_add_(1, idx10, masses * w10)
    density.scatter_add_(1, idx11, masses * w11)
    density = density.reshape(batch_size, 1, height, width)

    if not include_occupancy:
        return density

    occupancy = masses.new_zeros((batch_size, num_cells))
    point_unit = masses.new_full((batch_size, num_points), 1.0 / max(num_points, 1))
    occupancy.scatter_add_(1, idx00, point_unit * w00)
    occupancy.scatter_add_(1, idx01, point_unit * w01)
    occupancy.scatter_add_(1, idx10, point_unit * w10)
    occupancy.scatter_add_(1, idx11, point_unit * w11)
    occupancy = occupancy.reshape(batch_size, 1, height, width)
    return torch.cat([density, occupancy], dim=1)


def _sample_feature_grid_at_positions(
    feature_grid: Tensor,
    positions: Tensor,
    *,
    periodic: bool,
) -> Tensor:
    """Sample a dense feature grid at particle locations by bilinear interpolation."""
    if feature_grid.ndim != 4:
        raise ValueError("feature_grid must have shape (B, C, H, W)")
    if positions.ndim != 3 or positions.shape[0] != feature_grid.shape[0] or positions.shape[2] != 2:
        raise ValueError("positions must have shape (B, K, 2) and match feature_grid batch size")

    batch_size, channels, height, width = feature_grid.shape
    _, num_points, _ = positions.shape

    x0_idx, x1_idx, y0_idx, y1_idx, w00, w01, w10, w11 = _grid_bilinear_coordinates(
        positions,
        height=height,
        width=width,
        periodic=periodic,
    )
    idx00 = y0_idx * width + x0_idx
    idx01 = y0_idx * width + x1_idx
    idx10 = y1_idx * width + x0_idx
    idx11 = y1_idx * width + x1_idx

    flat = feature_grid.reshape(batch_size, channels, height * width)

    def _gather(idx: Tensor) -> Tensor:
        gathered = flat.gather(2, idx[:, None, :].expand(batch_size, channels, num_points))
        return gathered.transpose(1, 2)

    sampled = (
        w00[..., None] * _gather(idx00)
        + w01[..., None] * _gather(idx01)
        + w10[..., None] * _gather(idx10)
        + w11[..., None] * _gather(idx11)
    )
    return sampled


class _ConvBlock2d(nn.Module):
    """Two-layer convolutional block used by the image-field U-Net."""

    def __init__(self, in_channels: int, out_channels: int, *, padding_mode: str) -> None:
        super().__init__()
        groups = _group_norm_groups(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _SmallUNet2d(nn.Module):
    """Small U-Net that turns rasterized clouds into dense local feature grids."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        padding_mode: str,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("U-Net channel counts must be positive")

        c1 = int(base_channels)
        c2 = 2 * c1
        c3 = 4 * c1

        self.enc1 = _ConvBlock2d(in_channels, c1, padding_mode=padding_mode)
        self.pool1 = nn.AvgPool2d(2)
        self.enc2 = _ConvBlock2d(c1, c2, padding_mode=padding_mode)
        self.pool2 = nn.AvgPool2d(2)
        self.bottleneck = _ConvBlock2d(c2, c3, padding_mode=padding_mode)

        self.up2 = nn.Conv2d(c3, c2, kernel_size=1)
        self.dec2 = _ConvBlock2d(c2 + c2, c2, padding_mode=padding_mode)
        self.up1 = nn.Conv2d(c2, c1, kernel_size=1)
        self.dec1 = _ConvBlock2d(c1 + c1, c1, padding_mode=padding_mode)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        up2 = F.interpolate(bottleneck, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.up2(up2)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(up1)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.out(dec1)


class ConditionalScoreImageFieldNetwork(nn.Module):
    r"""Hybrid image-field score network for weighted point clouds.

    The model first rasterizes the weighted point cloud into a small periodic
    density image, augments it with broadcast label/time conditioning, and runs
    a compact U-Net to produce dense local features on the canvas.  These local
    features are then bilinearly sampled back at the particle locations and
    passed through a lightweight particle-wise head together with the masses and
    conditioning variables.

    This is the "Option B" hybrid architecture:
        point cloud -> rasterized image -> U-Net features
                    -> sample local features at particles -> point head -> score
    """

    def __init__(
        self,
        *,
        grid_size: int = 32,
        base_channels: int = 32,
        grid_feature_dim: int = 64,
        point_hidden_dim: int = 256,
        conditioning_dim: int = 64,
        num_classes: int = 10,
        condition_on_label: bool = True,
        tau_min: float = 1e-6,
        tau_max: float = 1e-3,
        dropout: float = 0.0,
        use_torus_features: bool = True,
        score_output_scaling: str = "tau_mass",
        include_occupancy_channel: bool = True,
    ) -> None:
        super().__init__()
        if grid_size <= 0 or base_channels <= 0 or grid_feature_dim <= 0 or point_hidden_dim <= 0:
            raise ValueError("image-field widths must be positive")
        if conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
            raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if score_output_scaling not in {"tau", "tau_mass", "none"}:
            raise ValueError("score_output_scaling must be one of {'tau', 'tau_mass', 'none'}")

        self.grid_size = int(grid_size)
        self.base_channels = int(base_channels)
        self.grid_feature_dim = int(grid_feature_dim)
        self.point_hidden_dim = int(point_hidden_dim)
        self.num_classes = int(num_classes)
        self.condition_on_label = bool(condition_on_label)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.use_torus_features = bool(use_torus_features)
        self.score_output_scaling = str(score_output_scaling)
        self.include_occupancy_channel = bool(include_occupancy_channel)

        self.time_mlp = nn.Sequential(
            nn.Linear(3, conditioning_dim),
            nn.GELU(),
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.GELU(),
        )
        if self.condition_on_label:
            self.label_embedding = nn.Embedding(self.num_classes, conditioning_dim)
            label_dim = conditioning_dim
        else:
            self.label_embedding = None
            label_dim = 0
        cond_dim = conditioning_dim + label_dim

        raster_channels = 2 if self.include_occupancy_channel else 1
        self.feature_unet = _SmallUNet2d(
            in_channels=raster_channels + cond_dim,
            out_channels=self.grid_feature_dim,
            base_channels=self.base_channels,
            padding_mode="circular" if self.use_torus_features else "zeros",
        )

        position_dim = 6 if self.use_torus_features else 2
        point_input_dim = self.grid_feature_dim + position_dim + 2 + cond_dim
        self.point_head = nn.Sequential(
            nn.Linear(point_input_dim, self.point_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.point_hidden_dim, self.point_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.point_hidden_dim, 2),
        )

    def _prepare_tau(self, tau: Tensor | float, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if isinstance(tau, Tensor):
            tau_tensor = tau.to(device=device, dtype=dtype).reshape(-1)
        else:
            tau_tensor = torch.full((batch_size,), float(tau), device=device, dtype=dtype)
        if tau_tensor.numel() == 1:
            tau_tensor = tau_tensor.expand(batch_size)
        if tau_tensor.shape != (batch_size,):
            raise ValueError("tau must be a scalar or have shape (B,)")
        return tau_tensor

    def _prepare_conditioning(self, tau: Tensor, labels: Optional[Tensor]) -> Tensor:
        time_context = self.time_mlp(
            _tau_features(tau, tau_min=self.tau_min, tau_max=self.tau_max)
        )
        if not self.condition_on_label:
            return time_context
        if labels is None:
            raise ValueError("labels are required when condition_on_label=True")
        labels = labels.reshape(-1).to(device=tau.device, dtype=torch.long)
        if labels.shape != tau.shape:
            raise ValueError("labels must have shape (B,)")
        label_context = self.label_embedding(labels)
        return torch.cat([time_context, label_context], dim=1)

    def _position_features(self, positions: Tensor) -> Tensor:
        if not self.use_torus_features:
            return positions
        angles = 2.0 * math.pi * positions
        return torch.cat([positions, torch.sin(angles), torch.cos(angles)], dim=-1)

    def _score_scale(self, tau: Tensor, masses: Tensor) -> Tensor:
        if self.score_output_scaling == "none":
            return torch.ones((*masses.shape, 1), device=masses.device, dtype=masses.dtype)
        if self.score_output_scaling == "tau":
            return torch.rsqrt((2.0 * tau).clamp_min(self.tau_min))[:, None, None]
        return torch.rsqrt(
            (2.0 * tau[:, None, None] * masses.unsqueeze(-1)).clamp_min(self.tau_min * 1e-8)
        )

    def forward(
        self,
        masses: Tensor,
        positions: Tensor,
        tau: Tensor | float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if masses.ndim != 2:
            raise ValueError("masses must have shape (B, K)")
        if positions.ndim != 3 or positions.shape[:2] != masses.shape or positions.shape[2] != 2:
            raise ValueError("positions must have shape (B, K, 2) and match masses")

        batch_size, num_points = masses.shape
        tau_tensor = self._prepare_tau(
            tau,
            batch_size,
            device=positions.device,
            dtype=positions.dtype,
        )
        cond = self._prepare_conditioning(tau_tensor, labels)
        cond_points = cond[:, None, :].expand(batch_size, num_points, cond.shape[1])

        raster = _rasterize_weighted_point_clouds_torch(
            masses,
            positions,
            grid_size=self.grid_size,
            periodic=self.use_torus_features,
            include_occupancy=self.include_occupancy_channel,
        )
        cond_grid = cond[:, :, None, None].expand(batch_size, cond.shape[1], self.grid_size, self.grid_size)
        feature_grid = self.feature_unet(torch.cat([raster, cond_grid], dim=1))
        local_features = _sample_feature_grid_at_positions(
            feature_grid,
            positions,
            periodic=self.use_torus_features,
        )

        log_masses = torch.log(masses.clamp_min(1e-8))
        point_inputs = torch.cat(
            [
                local_features,
                self._position_features(positions),
                masses.unsqueeze(-1),
                log_masses.unsqueeze(-1),
                cond_points,
            ],
            dim=-1,
        )
        raw_scaled_score = self.point_head(point_inputs)
        return raw_scaled_score * self._score_scale(tau_tensor, masses)


class ConditionalScoreSetNetwork(nn.Module):
    r"""Permutation-equivariant score network for weighted point clouds.

    The network is class-conditional by default so notebook 7 can mirror the
    class-conditional setup of notebook 6.  Setting ``condition_on_label=False``
    recovers an unconditional score model.

    The architecture follows the same DeepSets spirit as the terminal classifier:
    per-point features are encoded, pooled into a global context, and then mapped
    back to a per-point vector field.  By default, the network predicts a
    noise-like quantity and returns it scaled by ``1 / sqrt(2 tau s_i)`` via
    ``score_output_scaling='tau_mass'``.  This matches the small-time scale of
    the weighted score target when the masses vary.  Set
    ``score_output_scaling='tau'`` to recover the previous ``1 / sqrt(2 tau)``
    scaling, or ``'none'`` to return the raw network output.  Torus Fourier
    features are enabled by default to make the wrapped 0/1 chart continuous to
    the network.
    """

    def __init__(
        self,
        *,
        point_feature_dim: int = 128,
        hidden_dim: int = 256,
        conditioning_dim: int = 64,
        num_classes: int = 10,
        condition_on_label: bool = True,
        tau_min: float = 1e-6,
        tau_max: float = 1e-3,
        dropout: float = 0.0,
        use_torus_features: bool = True,
        score_output_scaling: str = "tau_mass",
    ) -> None:
        super().__init__()
        if point_feature_dim <= 0 or hidden_dim <= 0 or conditioning_dim <= 0:
            raise ValueError("network widths must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
            raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if score_output_scaling not in {"tau", "tau_mass", "none"}:
            raise ValueError("score_output_scaling must be one of {'tau', 'tau_mass', 'none'}")

        self.num_classes = int(num_classes)
        self.condition_on_label = bool(condition_on_label)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.use_torus_features = bool(use_torus_features)
        self.score_output_scaling = str(score_output_scaling)

        self.time_mlp = nn.Sequential(
            nn.Linear(3, conditioning_dim),
            nn.GELU(),
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.GELU(),
        )
        if self.condition_on_label:
            self.label_embedding = nn.Embedding(self.num_classes, conditioning_dim)
            label_dim = conditioning_dim
        else:
            self.label_embedding = None
            label_dim = 0

        cond_dim = conditioning_dim + label_dim
        position_dim = 6 if self.use_torus_features else 2
        point_input_dim = position_dim + 2 + cond_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(point_input_dim, point_feature_dim),
            nn.GELU(),
            nn.Linear(point_feature_dim, point_feature_dim),
            nn.GELU(),
            nn.Linear(point_feature_dim, point_feature_dim),
            nn.GELU(),
        )
        global_dim = 3 * point_feature_dim + cond_dim
        self.output_head = nn.Sequential(
            nn.Linear(point_feature_dim + global_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _prepare_tau(self, tau: Tensor | float, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if isinstance(tau, Tensor):
            tau_tensor = tau.to(device=device, dtype=dtype).reshape(-1)
        else:
            tau_tensor = torch.full((batch_size,), float(tau), device=device, dtype=dtype)
        if tau_tensor.numel() == 1:
            tau_tensor = tau_tensor.expand(batch_size)
        if tau_tensor.shape != (batch_size,):
            raise ValueError("tau must be a scalar or have shape (B,)")
        return tau_tensor

    def _prepare_conditioning(self, tau: Tensor, labels: Optional[Tensor]) -> Tensor:
        time_context = self.time_mlp(
            _tau_features(tau, tau_min=self.tau_min, tau_max=self.tau_max)
        )
        if not self.condition_on_label:
            return time_context
        if labels is None:
            raise ValueError("labels are required when condition_on_label=True")
        labels = labels.reshape(-1).to(device=tau.device, dtype=torch.long)
        if labels.shape != tau.shape:
            raise ValueError("labels must have shape (B,)")
        label_context = self.label_embedding(labels)
        return torch.cat([time_context, label_context], dim=1)

    def _position_features(self, positions: Tensor) -> Tensor:
        if not self.use_torus_features:
            return positions
        angles = 2.0 * math.pi * positions
        return torch.cat([positions, torch.sin(angles), torch.cos(angles)], dim=-1)

    def _score_scale(self, tau: Tensor, masses: Tensor) -> Tensor:
        if self.score_output_scaling == "none":
            return torch.ones((*masses.shape, 1), device=masses.device, dtype=masses.dtype)
        if self.score_output_scaling == "tau":
            return torch.rsqrt((2.0 * tau).clamp_min(self.tau_min))[:, None, None]
        return torch.rsqrt(
            (2.0 * tau[:, None, None] * masses.unsqueeze(-1)).clamp_min(self.tau_min * 1e-8)
        )

    def forward(
        self,
        masses: Tensor,
        positions: Tensor,
        tau: Tensor | float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if masses.ndim != 2:
            raise ValueError("masses must have shape (B, K)")
        if positions.ndim != 3 or positions.shape[:2] != masses.shape or positions.shape[2] != 2:
            raise ValueError("positions must have shape (B, K, 2) and match masses")

        batch_size, num_points = masses.shape
        tau_tensor = self._prepare_tau(
            tau,
            batch_size,
            device=positions.device,
            dtype=positions.dtype,
        )
        cond = self._prepare_conditioning(tau_tensor, labels)
        cond_points = cond[:, None, :].expand(batch_size, num_points, cond.shape[1])

        log_masses = torch.log(masses.clamp_min(1e-8))
        point_inputs = torch.cat(
            [
                self._position_features(positions),
                masses.unsqueeze(-1),
                log_masses.unsqueeze(-1),
                cond_points,
            ],
            dim=-1,
        )
        h = self.point_mlp(point_inputs)

        weights = masses.unsqueeze(-1)
        mean = torch.sum(weights * h, dim=1)
        second = torch.sum(weights * h.square(), dim=1)
        std = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-8)
        maximum = torch.max(h, dim=1).values
        global_context = torch.cat([mean, std, maximum, cond], dim=1)
        repeated_context = global_context[:, None, :].expand(batch_size, num_points, global_context.shape[1])

        raw_scaled_score = self.output_head(torch.cat([h, repeated_context], dim=-1))
        return raw_scaled_score * self._score_scale(tau_tensor, masses)

class ConditionalScoreSetTransformer(nn.Module):
    r"""Self-attention score network for weighted point clouds.

    This is a stronger alternative to :class:`ConditionalScoreSetNetwork` for
    notebook 7.  It keeps the same public ``forward(masses, positions, tau,
    labels)`` interface but replaces the DeepSets encoder by several
    permutation-equivariant self-attention blocks.  The per-point inputs include
    torus Fourier coordinates by default, which avoids making the wrapped chart
    discontinuity at 0/1 look artificial to the network.

    ``score_output_scaling='tau_mass'`` makes the raw network output a
    noise-like quantity and returns ``raw / sqrt(2 tau s_i)``.  This matches the
    small-time scale of the weighted score target and is usually easier to train
    than predicting a raw ``sqrt(2 tau)``-scaled score when the masses vary.
    """

    def __init__(
        self,
        *,
        point_feature_dim: int = 192,
        hidden_dim: int = 384,
        conditioning_dim: int = 64,
        num_classes: int = 10,
        condition_on_label: bool = True,
        tau_min: float = 1e-6,
        tau_max: float = 1e-3,
        dropout: float = 0.05,
        num_attention_layers: int = 4,
        num_attention_heads: int = 8,
        feedforward_dim: Optional[int] = None,
        use_torus_features: bool = True,
        score_output_scaling: str = "tau_mass",
    ) -> None:
        super().__init__()
        if point_feature_dim <= 0 or hidden_dim <= 0 or conditioning_dim <= 0:
            raise ValueError("network widths must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
            raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if num_attention_layers <= 0:
            raise ValueError("num_attention_layers must be positive")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if point_feature_dim % num_attention_heads != 0:
            raise ValueError("point_feature_dim must be divisible by num_attention_heads")
        if score_output_scaling not in {"tau", "tau_mass", "none"}:
            raise ValueError("score_output_scaling must be one of {'tau', 'tau_mass', 'none'}")

        self.num_classes = int(num_classes)
        self.condition_on_label = bool(condition_on_label)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.use_torus_features = bool(use_torus_features)
        self.score_output_scaling = str(score_output_scaling)

        self.time_mlp = nn.Sequential(
            nn.Linear(3, conditioning_dim),
            nn.GELU(),
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.GELU(),
        )
        if self.condition_on_label:
            self.label_embedding = nn.Embedding(self.num_classes, conditioning_dim)
            label_dim = conditioning_dim
        else:
            self.label_embedding = None
            label_dim = 0

        cond_dim = conditioning_dim + label_dim
        position_dim = 6 if self.use_torus_features else 2
        point_input_dim = position_dim + 2 + cond_dim
        self.input_projection = nn.Sequential(
            nn.Linear(point_input_dim, point_feature_dim),
            nn.GELU(),
            nn.LayerNorm(point_feature_dim),
        )

        ff_dim = int(feedforward_dim) if feedforward_dim is not None else 4 * int(point_feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=point_feature_dim,
            nhead=num_attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_attention_layers))

        global_dim = 3 * point_feature_dim + cond_dim
        self.output_head = nn.Sequential(
            nn.Linear(point_feature_dim + global_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def _prepare_tau(self, tau: Tensor | float, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if isinstance(tau, Tensor):
            tau_tensor = tau.to(device=device, dtype=dtype).reshape(-1)
        else:
            tau_tensor = torch.full((batch_size,), float(tau), device=device, dtype=dtype)
        if tau_tensor.numel() == 1:
            tau_tensor = tau_tensor.expand(batch_size)
        if tau_tensor.shape != (batch_size,):
            raise ValueError("tau must be a scalar or have shape (B,)")
        return tau_tensor

    def _prepare_conditioning(self, tau: Tensor, labels: Optional[Tensor]) -> Tensor:
        time_context = self.time_mlp(
            _tau_features(tau, tau_min=self.tau_min, tau_max=self.tau_max)
        )
        if not self.condition_on_label:
            return time_context
        if labels is None:
            raise ValueError("labels are required when condition_on_label=True")
        labels = labels.reshape(-1).to(device=tau.device, dtype=torch.long)
        if labels.shape != tau.shape:
            raise ValueError("labels must have shape (B,)")
        label_context = self.label_embedding(labels)
        return torch.cat([time_context, label_context], dim=1)

    def _position_features(self, positions: Tensor) -> Tensor:
        if not self.use_torus_features:
            return positions
        angles = 2.0 * math.pi * positions
        return torch.cat([positions, torch.sin(angles), torch.cos(angles)], dim=-1)

    def _score_scale(self, tau: Tensor, masses: Tensor) -> Tensor:
        if self.score_output_scaling == "none":
            return torch.ones((*masses.shape, 1), device=masses.device, dtype=masses.dtype)
        if self.score_output_scaling == "tau":
            return torch.rsqrt((2.0 * tau).clamp_min(self.tau_min))[:, None, None]
        return torch.rsqrt(
            (2.0 * tau[:, None, None] * masses.unsqueeze(-1)).clamp_min(self.tau_min * 1e-8)
        )

    def forward(
        self,
        masses: Tensor,
        positions: Tensor,
        tau: Tensor | float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        if masses.ndim != 2:
            raise ValueError("masses must have shape (B, K)")
        if positions.ndim != 3 or positions.shape[:2] != masses.shape or positions.shape[2] != 2:
            raise ValueError("positions must have shape (B, K, 2) and match masses")

        batch_size, num_points = masses.shape
        tau_tensor = self._prepare_tau(
            tau,
            batch_size,
            device=positions.device,
            dtype=positions.dtype,
        )
        cond = self._prepare_conditioning(tau_tensor, labels)
        cond_points = cond[:, None, :].expand(batch_size, num_points, cond.shape[1])

        log_masses = torch.log(masses.clamp_min(1e-8))
        point_inputs = torch.cat(
            [
                self._position_features(positions),
                masses.unsqueeze(-1),
                log_masses.unsqueeze(-1),
                cond_points,
            ],
            dim=-1,
        )
        h = self.input_projection(point_inputs)
        h = self.encoder(h)

        weights = masses.unsqueeze(-1)
        mean = torch.sum(weights * h, dim=1)
        second = torch.sum(weights * h.square(), dim=1)
        std = torch.sqrt((second - mean.square()).clamp_min(0.0) + 1e-8)
        maximum = torch.max(h, dim=1).values
        global_context = torch.cat([mean, std, maximum, cond], dim=1)
        repeated_context = global_context[:, None, :].expand(batch_size, num_points, global_context.shape[1])

        raw_scaled_score = self.output_head(torch.cat([h, repeated_context], dim=-1))
        return raw_scaled_score * self._score_scale(tau_tensor, masses)


# ---------------------------------------------------------------------------
# Noising and loss
# ---------------------------------------------------------------------------




def add_forward_noise_to_positions(
    masses: np.ndarray,
    clean_positions: np.ndarray,
    tau: float | np.ndarray,
    *,
    projection: str = "reflect",
    rng: Optional[np.random.Generator] = None,
) -> FloatArray:
    r"""Apply the finite-particle forward noising kernel in NumPy.

    This is the distribution that a reverse score sampler should start from at
    horizon ``T`` when the clean point-cloud law is represented by
    ``(masses, clean_positions)``:

        Y_i = X_i + sqrt(2 T / s_i) Z_i.

    The mass-dependent factor is the important part: replacing it by a single
    jitter scale changes the start distribution for small-mass atoms.
    """
    masses_arr = np.asarray(masses, dtype=np.float64)
    positions_arr = np.asarray(clean_positions, dtype=np.float64)
    if masses_arr.ndim != 2:
        raise ValueError("masses must have shape (N, K)")
    if positions_arr.ndim != 3 or positions_arr.shape[:2] != masses_arr.shape or positions_arr.shape[2] != 2:
        raise ValueError("clean_positions must have shape (N, K, 2) and match masses")
    if not np.all(np.isfinite(masses_arr)) or np.any(masses_arr <= 0.0):
        raise ValueError("all masses must be positive and finite")
    if not np.all(np.isfinite(positions_arr)):
        raise ValueError("clean_positions must be finite")

    tau_arr = np.asarray(tau, dtype=np.float64)
    if tau_arr.ndim == 0:
        tau_vec = np.full((masses_arr.shape[0],), float(tau_arr), dtype=np.float64)
    else:
        tau_vec = tau_arr.reshape(-1)
    if tau_vec.shape != (masses_arr.shape[0],):
        raise ValueError("tau must be scalar or have shape (N,)")
    if not np.all(np.isfinite(tau_vec)) or np.any(tau_vec <= 0.0):
        raise ValueError("tau values must be positive and finite")

    rng = np.random.default_rng() if rng is None else rng
    sigma = np.sqrt((2.0 * tau_vec[:, None, None]) / masses_arr[:, :, None])
    noisy = positions_arr + sigma * rng.normal(size=positions_arr.shape)
    return np.asarray(project_positions(noisy, mode=projection), dtype=np.float64)


def _balanced_subsample_indices(
    labels: np.ndarray,
    max_samples: Optional[int],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    num_items = int(labels_arr.shape[0])
    if max_samples is None or max_samples >= num_items:
        return rng.permutation(num_items)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    classes = np.unique(labels_arr)
    per_class = max(1, int(max_samples) // max(len(classes), 1))
    chosen: list[np.ndarray] = []
    used = np.zeros(num_items, dtype=bool)
    for cls in classes:
        idx = np.flatnonzero(labels_arr == cls)
        take = min(per_class, len(idx))
        draw = rng.choice(idx, size=take, replace=False)
        chosen.append(draw)
        used[draw] = True

    selected = np.concatenate(chosen) if chosen else np.empty((0,), dtype=np.int64)
    remaining_needed = int(max_samples) - int(selected.shape[0])
    if remaining_needed > 0:
        remaining = np.flatnonzero(~used)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(remaining_needed, len(remaining)), replace=False)
            selected = np.concatenate([selected, extra])
    return rng.permutation(selected.astype(np.int64, copy=False))


def _stratified_train_test_split(
    labels: np.ndarray,
    train_fraction: float,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie strictly between 0 and 1")

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for cls in np.unique(labels_arr):
        idx = rng.permutation(np.flatnonzero(labels_arr == cls))
        if len(idx) <= 1:
            train_parts.append(idx)
            continue
        n_train = int(round(train_fraction * len(idx)))
        n_train = min(max(n_train, 1), len(idx) - 1)
        train_parts.append(idx[:n_train])
        test_parts.append(idx[n_train:])

    train_idx = np.concatenate(train_parts) if train_parts else np.empty((0,), dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.empty((0,), dtype=np.int64)
    if len(test_idx) == 0 and len(labels_arr) >= 2:
        shuffled = rng.permutation(len(labels_arr))
        split = max(1, min(len(labels_arr) - 1, int(round(train_fraction * len(labels_arr)))))
        train_idx, test_idx = shuffled[:split], shuffled[split:]
    return rng.permutation(train_idx), rng.permutation(test_idx)


def _linear_classifier_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    train_fraction: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: Optional[str | torch.device],
    rng: np.random.Generator,
) -> float:
    features_arr = np.asarray(features, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if features_arr.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    if labels_arr.shape != (features_arr.shape[0],):
        raise ValueError("labels must have shape (N,)")
    if epochs <= 0:
        return float("nan")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    classes = np.unique(labels_arr)
    if len(classes) <= 1 or len(labels_arr) < 4:
        return float("nan")
    class_to_index = {int(cls): i for i, cls in enumerate(classes)}
    mapped_labels = np.asarray([class_to_index[int(cls)] for cls in labels_arr], dtype=np.int64)

    train_idx, test_idx = _stratified_train_test_split(mapped_labels, train_fraction, rng=rng)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return float("nan")

    mean = features_arr[train_idx].mean(axis=0, keepdims=True)
    std = features_arr[train_idx].std(axis=0, keepdims=True)
    standardized = (features_arr - mean) / np.maximum(std, 1e-6)

    model_device = _resolve_device(device)
    x_train = torch.from_numpy(standardized[train_idx]).to(device=model_device, dtype=torch.float32)
    y_train = torch.from_numpy(mapped_labels[train_idx]).to(device=model_device, dtype=torch.long)
    x_test = torch.from_numpy(standardized[test_idx]).to(device=model_device, dtype=torch.float32)
    y_test = torch.from_numpy(mapped_labels[test_idx]).to(device=model_device, dtype=torch.long)

    linear = nn.Linear(features_arr.shape[1], len(classes)).to(model_device)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
    optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(int(epochs)):
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), int(batch_size)):
            stop = min(start + int(batch_size), len(order))
            batch_np = order[start:stop]
            batch = torch.as_tensor(batch_np, device=model_device, dtype=torch.long)
            logits = linear(x_train[batch])
            loss = torch.nn.functional.cross_entropy(logits, y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        pred = torch.argmax(linear(x_test), dim=1)
        accuracy = torch.mean((pred == y_test).to(torch.float32)).item()
    return float(accuracy)


def _diagnostic_features(
    masses: np.ndarray,
    positions: np.ndarray,
    *,
    feature_grid_size: int,
) -> np.ndarray:
    if feature_grid_size <= 1:
        raise ValueError("feature_grid_size must be at least 2")
    images = rasterize_weighted_point_clouds(masses, positions, image_size=feature_grid_size)
    return np.asarray(images, dtype=np.float32).reshape(images.shape[0], -1)


def diagnose_score_prior_horizons(
    masses: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    horizons: Sequence[float],
    *,
    projection: str = "reflect",
    prior_position_mode: str = "uniform",
    prior_position_scale: float = 0.12,
    max_samples: Optional[int] = 2048,
    feature_grid_size: int = 8,
    train_fraction: float = 0.7,
    classifier_epochs: int = 80,
    classifier_batch_size: int = 256,
    classifier_lr: float = 5e-2,
    classifier_weight_decay: float = 1e-4,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[dict[str, float]]:
    r"""Diagnose whether a candidate horizon is large enough for a simple prior.

    A uniform reverse start is theoretically appropriate only when the forward
    noised data law at ``tau=T`` is close to the chosen prior.  This helper
    checks that assumption without training the score model.  For each candidate
    horizon it forward-noises real weighted point clouds with the same
    mass-dependent kernel used by score matching, compares them to independent
    samples from the proposed initial-position prior, and reports two small
    classifier diagnostics:

    ``prior_accuracy``
        Test accuracy of a linear classifier distinguishing forward-noised data
        from prior samples.  Values near 0.5 mean the prior is hard to
        distinguish from the noised endpoint.

    ``label_accuracy``
        Test accuracy of a linear classifier predicting the MNIST label from
        the forward-noised point clouds.  Values near the chance baseline mean
        that label-specific spatial information has mostly been erased at that
        horizon.

    Good horizons should make both accuracies close to their baselines.  The
    helper also reports a mass-weighted marginal histogram TV distance, which is
    cheap and useful when ``classifier_epochs=0``.
    """
    masses_arr = np.asarray(masses, dtype=np.float64)
    positions_arr = np.asarray(positions, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if masses_arr.ndim != 2:
        raise ValueError("masses must have shape (N, K)")
    if positions_arr.ndim != 3 or positions_arr.shape[:2] != masses_arr.shape or positions_arr.shape[2] != 2:
        raise ValueError("positions must have shape (N, K, 2) and match masses")
    if labels_arr.shape != (masses_arr.shape[0],):
        raise ValueError("labels must have shape (N,)")
    if prior_position_mode not in {"uniform", "centered_gaussian"}:
        raise ValueError("prior_position_mode must be 'uniform' or 'centered_gaussian'")

    horizons_arr = np.asarray(list(horizons), dtype=np.float64).reshape(-1)
    if horizons_arr.size == 0:
        raise ValueError("horizons must contain at least one value")
    if not np.all(np.isfinite(horizons_arr)) or np.any(horizons_arr <= 0.0):
        raise ValueError("all horizons must be positive and finite")

    rng = np.random.default_rng() if rng is None else rng
    selected = _balanced_subsample_indices(labels_arr, max_samples, rng=rng)
    masses_sub = masses_arr[selected]
    positions_sub = positions_arr[selected]
    labels_sub = labels_arr[selected]
    num_samples, num_points = masses_sub.shape

    prior_positions = sample_initial_positions(
        num_samples,
        num_points,
        mode=prior_position_mode,
        scale=prior_position_scale,
        rng=rng,
    )
    prior_positions = np.asarray(project_positions(prior_positions, mode=projection), dtype=np.float64)
    prior_features = _diagnostic_features(
        masses_sub,
        prior_positions,
        feature_grid_size=feature_grid_size,
    )
    prior_mean = prior_features.mean(axis=0)
    label_chance = 1.0 / max(1, len(np.unique(labels_sub)))

    rows: list[dict[str, float]] = []
    for horizon in sorted(float(h) for h in horizons_arr):
        noised_positions = add_forward_noise_to_positions(
            masses_sub,
            positions_sub,
            horizon,
            projection=projection,
            rng=rng,
        )
        noised_features = _diagnostic_features(
            masses_sub,
            noised_positions,
            feature_grid_size=feature_grid_size,
        )
        noised_mean = noised_features.mean(axis=0)
        weighted_marginal_tv = 0.5 * float(np.sum(np.abs(noised_mean - prior_mean)))

        domain_features = np.concatenate([noised_features, prior_features], axis=0)
        domain_labels = np.concatenate(
            [
                np.zeros(num_samples, dtype=np.int64),
                np.ones(num_samples, dtype=np.int64),
            ]
        )
        prior_accuracy = _linear_classifier_accuracy(
            domain_features,
            domain_labels,
            train_fraction=train_fraction,
            epochs=classifier_epochs,
            batch_size=classifier_batch_size,
            lr=classifier_lr,
            weight_decay=classifier_weight_decay,
            device=device,
            rng=rng,
        )
        label_accuracy = _linear_classifier_accuracy(
            noised_features,
            labels_sub,
            train_fraction=train_fraction,
            epochs=classifier_epochs,
            batch_size=classifier_batch_size,
            lr=classifier_lr,
            weight_decay=classifier_weight_decay,
            device=device,
            rng=rng,
        )

        sigma = np.sqrt((2.0 * horizon) / masses_sub)
        rows.append(
            {
                "horizon": float(horizon),
                "num_samples": float(num_samples),
                "feature_grid_size": float(feature_grid_size),
                "weighted_marginal_tv": float(weighted_marginal_tv),
                "prior_accuracy": float(prior_accuracy),
                "prior_chance_accuracy": 0.5,
                "label_accuracy": float(label_accuracy),
                "label_chance_accuracy": float(label_chance),
                "median_noise_std": float(np.median(sigma)),
                "p90_noise_std": float(np.quantile(sigma, 0.9)),
                "median_noise_pixels": float(28.0 * np.median(sigma)),
                "p90_noise_pixels": float(28.0 * np.quantile(sigma, 0.9)),
            }
        )
    return rows


def recommend_score_prior_horizon(
    diagnostics: Sequence[dict[str, float]],
    *,
    max_prior_accuracy: float = 0.55,
    label_accuracy_slack: float = 0.05,
    max_weighted_marginal_tv: Optional[float] = None,
) -> dict[str, float]:
    """Pick the smallest candidate horizon that passes prior-mixing checks.

    If no candidate satisfies the thresholds, the returned row is the closest
    available candidate according to the same excess-accuracy diagnostics.  The
    return value is a copy of the selected diagnostic row with an additional
    ``recommendation_status`` field equal to ``1.0`` for a threshold pass and
    ``0.0`` for the fallback case.
    """
    rows = [dict(row) for row in diagnostics]
    if not rows:
        raise ValueError("diagnostics must contain at least one row")
    if max_prior_accuracy < 0.5 or max_prior_accuracy > 1.0:
        raise ValueError("max_prior_accuracy must lie in [0.5, 1.0]")
    if label_accuracy_slack < 0.0:
        raise ValueError("label_accuracy_slack must be non-negative")
    if max_weighted_marginal_tv is not None and max_weighted_marginal_tv < 0.0:
        raise ValueError("max_weighted_marginal_tv must be non-negative when provided")

    rows.sort(key=lambda row: float(row["horizon"]))

    def _is_finite(value: float) -> bool:
        return bool(np.isfinite(float(value)))

    for row in rows:
        chance = float(row.get("label_chance_accuracy", 0.1))
        prior_accuracy = float(row.get("prior_accuracy", float("nan")))
        label_accuracy = float(row.get("label_accuracy", float("nan")))
        weighted_tv = float(row.get("weighted_marginal_tv", float("nan")))
        prior_ok = (not _is_finite(prior_accuracy)) or prior_accuracy <= max_prior_accuracy
        label_ok = (not _is_finite(label_accuracy)) or label_accuracy <= chance + label_accuracy_slack
        tv_ok = (
            max_weighted_marginal_tv is None
            or ((not _is_finite(weighted_tv)) or weighted_tv <= max_weighted_marginal_tv)
        )
        if prior_ok and label_ok and tv_ok:
            row["recommendation_status"] = 1.0
            return row

    def _score(row: dict[str, float]) -> float:
        chance = float(row.get("label_chance_accuracy", 0.1))
        prior_accuracy = float(row.get("prior_accuracy", float("nan")))
        label_accuracy = float(row.get("label_accuracy", float("nan")))
        weighted_tv = float(row.get("weighted_marginal_tv", 0.0))
        prior_excess = max(0.0, (prior_accuracy if np.isfinite(prior_accuracy) else 0.5) - 0.5)
        label_excess = max(0.0, (label_accuracy if np.isfinite(label_accuracy) else chance) - chance)
        tv_excess = 0.0
        if max_weighted_marginal_tv is not None:
            tv_excess = max(0.0, weighted_tv - max_weighted_marginal_tv)
        return prior_excess + label_excess + 0.25 * tv_excess

    best = dict(min(rows, key=_score))
    best["recommendation_status"] = 0.0
    return best

def sample_score_matching_noise_levels(
    batch_size: int,
    *,
    tau_min: float,
    tau_max: float,
    sampling: str = "quadratic_bias_to_zero",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Sample diffusion times for score matching.

    Parameters
    ----------
    batch_size:
        Number of noise levels to sample.
    tau_min, tau_max:
        Closed sampling interval.  Using ``tau_min`` comparable to the reverse
        Euler step size avoids extrapolating the score model to smaller times.
    sampling:
        One of ``'uniform'``, ``'quadratic_bias_to_zero'``, or ``'log_uniform'``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
        raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")
    if sampling not in {"uniform", "quadratic_bias_to_zero", "log_uniform"}:
        raise ValueError("sampling must be 'uniform', 'quadratic_bias_to_zero', or 'log_uniform'")

    u = torch.rand(batch_size, device=device, dtype=dtype)
    if sampling == "uniform":
        return tau_min + (tau_max - tau_min) * u
    if sampling == "quadratic_bias_to_zero":
        return tau_min + (tau_max - tau_min) * u.square()
    if math.isclose(tau_min, tau_max):
        return torch.full((batch_size,), float(tau_min), device=device, dtype=dtype)
    log_ratio = math.log(float(tau_max) / float(tau_min))
    return tau_min * torch.exp(log_ratio * u)


def _centered_torus_difference(delta: Tensor) -> Tensor:
    """Return coordinate differences in the chart ``[-1/2, 1/2)``."""
    return torch.remainder(delta + 0.5, 1.0) - 0.5


@torch.no_grad()
def torus_heat_kernel_score_target(
    masses: Tensor,
    clean_positions: Tensor,
    wrapped_positions: Tensor,
    tau: Tensor,
    *,
    image_radius: int = 3,
    fourier_modes: int = 32,
    fourier_switch_time: float = 2e-2,
    eps: float = 1e-12,
) -> Tensor:
    r"""Weighted score of the wrapped finite-particle heat kernel on ``T^2``.

    For each particle, the forward kernel used with ``projection='wrap'`` is the
    heat kernel on the unit torus with ordinary Brownian time ``tau / s_i``.  The
    returned target is the weighted Wasserstein score

        D_i^(n) log q_{tau,s_i}(y_i | x_i) = (1 / s_i) nabla_y log q_{tau/s_i}(y_i - x_i).

    Numerically, the exact torus heat kernel is evaluated by a stable hybrid of
    two equivalent convergent representations: a periodized Gaussian image sum
    for small ``tau / s_i`` and a Fourier series for larger ``tau / s_i``.  The
    truncation parameters ``image_radius`` and ``fourier_modes`` control the
    approximation accuracy.
    """
    if masses.ndim != 2:
        raise ValueError("masses must have shape (B, K)")
    if clean_positions.ndim != 3 or clean_positions.shape[:2] != masses.shape or clean_positions.shape[2] != 2:
        raise ValueError("clean_positions must have shape (B, K, 2) and match masses")
    if wrapped_positions.ndim != 3 or wrapped_positions.shape != clean_positions.shape:
        raise ValueError("wrapped_positions must have shape (B, K, 2) and match clean_positions")
    if tau.ndim != 1 or tau.shape != (masses.shape[0],):
        raise ValueError("tau must have shape (B,)")
    if image_radius < 0:
        raise ValueError("image_radius must be non-negative")
    if fourier_modes < 0:
        raise ValueError("fourier_modes must be non-negative")
    if fourier_switch_time < 0.0:
        raise ValueError("fourier_switch_time must be non-negative")

    dtype = clean_positions.dtype
    device = clean_positions.device
    masses_safe = masses.clamp_min(float(eps))
    tau_safe = tau.clamp_min(float(eps))
    heat_time = tau_safe[:, None] / masses_safe
    heat_time_expanded = heat_time[:, :, None, None]

    # Work in the centered torus chart.  This improves the accuracy of the
    # truncated image sum while leaving the infinite sum unchanged.
    displacement = _centered_torus_difference(wrapped_positions - clean_positions)

    offsets = torch.arange(
        -int(image_radius),
        int(image_radius) + 1,
        device=device,
        dtype=dtype,
    )
    image_deltas = displacement.unsqueeze(-1) + offsets.view(1, 1, 1, -1)
    image_log_weights = -image_deltas.square() / (4.0 * heat_time_expanded.clamp_min(float(eps)))
    image_weights = torch.softmax(image_log_weights, dim=-1)
    image_mean_delta = torch.sum(image_weights * image_deltas, dim=-1)
    image_score = -image_mean_delta / (2.0 * tau_safe[:, None, None])

    if fourier_modes == 0:
        return image_score

    modes = torch.arange(1, int(fourier_modes) + 1, device=device, dtype=dtype)
    mode_sq = modes.square().view(1, 1, -1)
    decay = torch.exp(-4.0 * math.pi * math.pi * heat_time[:, :, None] * mode_sq)
    angles = 2.0 * math.pi * displacement.unsqueeze(-1) * modes.view(1, 1, 1, -1)
    decay_for_coords = decay[:, :, None, :]

    # 1D heat kernel: p_t(r)=1+2 sum_n exp(-4 pi^2 n^2 t) cos(2 pi n r).
    # Its ordinary derivative is p'_t(r)=-4 pi sum_n n exp(-4 pi^2 n^2 t) sin(2 pi n r).
    density = 1.0 + 2.0 * torch.sum(decay_for_coords * torch.cos(angles), dim=-1)
    derivative = -4.0 * math.pi * torch.sum(
        modes.view(1, 1, 1, -1) * decay_for_coords * torch.sin(angles),
        dim=-1,
    )
    density_safe = torch.where(
        density.abs() > float(eps),
        density,
        torch.full_like(density, float(eps)),
    )
    fourier_score = (derivative / density_safe) / masses_safe[:, :, None]

    use_fourier = (heat_time > float(fourier_switch_time))[:, :, None]
    return torch.where(use_fourier, fourier_score, image_score)


@torch.no_grad()
def perturb_weighted_point_cloud_positions(
    masses: Tensor,
    clean_positions: Tensor,
    tau: Tensor,
    *,
    projection: str = "none",
    score_target: str = "auto",
    torus_image_radius: int = 3,
    torus_fourier_modes: int = 32,
    torus_fourier_switch_time: float = 2e-2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply the free finite-dimensional noising kernel to the positions.

    Returns
    -------
    noisy_positions:
        Positions passed to the score network; projections are applied here.
    kernel_score_target:
        Denoising score target.  With ``score_target='auto'`` the Euclidean-cover
        target is used for ``projection!='wrap'`` and the torus heat-kernel target
        is used for ``projection='wrap'``.
    noise:
        The standard-normal perturbation used to build the noisy positions.
    """
    if masses.ndim != 2:
        raise ValueError("masses must have shape (B, K)")
    if clean_positions.ndim != 3 or clean_positions.shape[:2] != masses.shape or clean_positions.shape[2] != 2:
        raise ValueError("clean_positions must have shape (B, K, 2) and match masses")
    if tau.ndim != 1 or tau.shape != (masses.shape[0],):
        raise ValueError("tau must have shape (B,)")
    if score_target not in {"auto", "euclidean", "torus", "torus_exact"}:
        raise ValueError("score_target must be one of {'auto', 'euclidean', 'torus', 'torus_exact'}")

    sigma = torch.sqrt((2.0 * tau[:, None, None]) / masses.unsqueeze(-1))
    noise = torch.randn_like(clean_positions)
    noisy_unprojected = clean_positions + sigma * noise
    noisy_positions = project_positions(noisy_unprojected, mode=projection)

    resolved_target = score_target
    if resolved_target == "auto":
        resolved_target = "torus" if projection == "wrap" else "euclidean"
    if resolved_target in {"torus", "torus_exact"}:
        if projection != "wrap":
            raise ValueError("torus score targets require projection='wrap'")
        kernel_score_target = torus_heat_kernel_score_target(
            masses,
            clean_positions,
            noisy_positions,
            tau,
            image_radius=torus_image_radius,
            fourier_modes=torus_fourier_modes,
            fourier_switch_time=torus_fourier_switch_time,
        )
    else:
        kernel_score_target = -noise / torch.sqrt((2.0 * tau[:, None, None]) * masses.unsqueeze(-1))
    return noisy_positions, kernel_score_target, noise


def weighted_denoising_score_matching_loss(
    predicted_score: Tensor,
    target_score: Tensor,
    masses: Tensor,
    tau: Tensor,
    *,
    time_weighting: str = "sigma2",
) -> tuple[Tensor, dict[str, float]]:
    """Weighted denoising score-matching loss.

    The note proposes the weighted least-squares loss

        E[ sum_i s_i ||S_theta - target||^2 ].

    Multiplying by a positive factor that depends only on ``tau`` does not change
    the population minimizer at each noise level.  The default choice
    ``time_weighting='sigma2'`` multiplies each sample loss by ``2 tau`` and
    therefore stabilizes the small-time regime.
    """
    if predicted_score.shape != target_score.shape:
        raise ValueError("predicted_score and target_score must have the same shape")
    if predicted_score.ndim != 3 or predicted_score.shape[2] != 2:
        raise ValueError("scores must have shape (B, K, 2)")
    if masses.shape != predicted_score.shape[:2]:
        raise ValueError("masses must have shape (B, K)")
    if tau.shape != (predicted_score.shape[0],):
        raise ValueError("tau must have shape (B,)")
    if time_weighting not in {"none", "sigma2", "sqrt_sigma2"}:
        raise ValueError("time_weighting must be one of {'none', 'sigma2', 'sqrt_sigma2'}")

    point_sq_error = torch.sum((predicted_score - target_score).square(), dim=-1)
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


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _make_loader(
    masses: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    position_jitter_std: float,
    projection: str,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    dataset = WeightedPointCloudDataset(
        masses,
        positions,
        labels,
        position_jitter_std=position_jitter_std,
        projection=projection,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_score_model(
    model: ConditionalScoreSetNetwork,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 256,
    tau_min: float,
    tau_max: float,
    tau_sampling: str = "quadratic_bias_to_zero",
    projection: str = "none",
    score_target: str = "auto",
    torus_image_radius: int = 3,
    torus_fourier_modes: int = 32,
    torus_fourier_switch_time: float = 2e-2,
    time_weighting: str = "sigma2",
    device: Optional[str | torch.device] = None,
) -> dict[str, float]:
    """Estimate the denoising score-matching objective on a validation split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()

    loader = _make_loader(
        masses,
        positions,
        labels,
        batch_size=batch_size,
        shuffle=False,
        position_jitter_std=0.0,
        projection=projection,
    )

    total_loss = 0.0
    total_sample_loss = 0.0
    total_zero_loss = 0.0
    total_items = 0
    tau_means = []

    for batch_masses, batch_positions, batch_labels in loader:
        batch_masses = batch_masses.to(model_device)
        batch_positions = batch_positions.to(model_device)
        batch_labels = batch_labels.to(model_device)
        tau = sample_score_matching_noise_levels(
            int(batch_masses.shape[0]),
            tau_min=tau_min,
            tau_max=tau_max,
            sampling=tau_sampling,
            device=model_device,
            dtype=batch_positions.dtype,
        )
        noisy_positions, target_score, _ = perturb_weighted_point_cloud_positions(
            batch_masses,
            batch_positions,
            tau,
            projection=projection,
            score_target=score_target,
            torus_image_radius=torus_image_radius,
            torus_fourier_modes=torus_fourier_modes,
            torus_fourier_switch_time=torus_fourier_switch_time,
        )
        predicted_score = model(batch_masses, noisy_positions, tau, batch_labels)
        loss, metrics = weighted_denoising_score_matching_loss(
            predicted_score,
            target_score,
            batch_masses,
            tau,
            time_weighting=time_weighting,
        )
        zero_loss, _ = weighted_denoising_score_matching_loss(
            torch.zeros_like(target_score),
            target_score,
            batch_masses,
            tau,
            time_weighting=time_weighting,
        )
        batch_size_actual = int(batch_masses.shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_sample_loss += float(metrics["sample_loss"]) * batch_size_actual
        total_zero_loss += float(zero_loss.item()) * batch_size_actual
        total_items += batch_size_actual
        tau_means.append(metrics["mean_tau"])

    if was_training:
        model.train()

    loss_value = total_loss / max(total_items, 1)
    zero_value = total_zero_loss / max(total_items, 1)
    return {
        "loss": loss_value,
        "sample_loss": total_sample_loss / max(total_items, 1),
        "zero_predictor_loss": zero_value,
        "fraction_improved_over_zero": (1.0 - loss_value / zero_value) if zero_value > 0.0 else float("nan"),
        "mean_tau": float(np.mean(tau_means)) if tau_means else float("nan"),
    }


def evaluate_score_model_by_tau_bins(
    model: ConditionalScoreSetNetwork,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 256,
    tau_min: float,
    tau_max: float,
    num_bins: int = 8,
    tau_values: Optional[Sequence[float]] = None,
    projection: str = "none",
    score_target: str = "auto",
    torus_image_radius: int = 3,
    torus_fourier_modes: int = 32,
    torus_fourier_switch_time: float = 2e-2,
    time_weighting: str = "sigma2",
    image_size: int = 28,
    device: Optional[str | torch.device] = None,
) -> list[dict[str, float]]:
    """Evaluate the score loss at fixed noise levels.

    The aggregate DSM loss can hide whether the score is weak near the high-noise
    start of the sampler or near the final denoising regime.  This helper fixes
    ``tau`` to several bins and also reports the zero-score baseline computed
    with the same target and time weighting.
    """
    if tau_values is None:
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
            raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")
        if math.isclose(float(tau_min), float(tau_max)):
            tau_grid = np.asarray([float(tau_min)], dtype=np.float64)
        else:
            tau_grid = np.geomspace(float(tau_min), float(tau_max), int(num_bins), dtype=np.float64)
    else:
        tau_grid = np.asarray(list(tau_values), dtype=np.float64).reshape(-1)
        if tau_grid.size == 0:
            raise ValueError("tau_values must contain at least one value")
        if not np.all(np.isfinite(tau_grid)) or np.any(tau_grid <= 0.0):
            raise ValueError("tau_values must be positive and finite")

    masses_arr = np.asarray(masses, dtype=np.float64)
    if masses_arr.ndim != 2:
        raise ValueError("masses must have shape (N, K)")
    median_mass = float(np.median(masses_arr))
    num_points = int(masses_arr.shape[1])

    rows: list[dict[str, float]] = []
    for tau_value in tau_grid:
        metrics = evaluate_score_model(
            model,
            masses,
            positions,
            labels,
            batch_size=batch_size,
            tau_min=float(tau_value),
            tau_max=float(tau_value),
            tau_sampling="uniform",
            projection=projection,
            score_target=score_target,
            torus_image_radius=torus_image_radius,
            torus_fourier_modes=torus_fourier_modes,
            torus_fourier_switch_time=torus_fourier_switch_time,
            time_weighting=time_weighting,
            device=device,
        )
        zero_loss = float(metrics.get("zero_predictor_loss", float("nan")))
        loss_value = float(metrics["loss"])
        rows.append(
            {
                "tau": float(tau_value),
                "loss": loss_value,
                "sample_loss": float(metrics["sample_loss"]),
                "zero_predictor_loss": zero_loss,
                "fraction_improved_over_zero": (1.0 - loss_value / zero_loss) if zero_loss > 0.0 else float("nan"),
                "median_noise_std": float(math.sqrt(2.0 * float(tau_value) / median_mass)) if median_mass > 0.0 else float("nan"),
                "median_noise_pixels": float(image_size * math.sqrt(2.0 * float(tau_value) / median_mass)) if median_mass > 0.0 else float("nan"),
                "num_points": float(num_points),
            }
        )
    return rows


def train_score_model(
    model: ConditionalScoreSetNetwork,
    train_masses: np.ndarray,
    train_positions: np.ndarray,
    train_labels: np.ndarray,
    *,
    val_masses: Optional[np.ndarray] = None,
    val_positions: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    position_jitter_std: float = 0.0,
    tau_min: float = 1e-6,
    tau_max: float = 1e-3,
    tau_sampling: str = "quadratic_bias_to_zero",
    projection: str = "none",
    score_target: str = "auto",
    torus_image_radius: int = 3,
    torus_fourier_modes: int = 32,
    torus_fourier_switch_time: float = 2e-2,
    time_weighting: str = "sigma2",
    max_grad_norm: Optional[float] = 5.0,
    device: Optional[str | torch.device] = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train the score network with denoising score matching."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
        raise ValueError("tau_min and tau_max must satisfy 0 < tau_min <= tau_max")

    model.tau_min = float(tau_min)
    model.tau_max = float(tau_max)

    model_device = _resolve_device(device)
    model = model.to(model_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = _make_loader(
        train_masses,
        train_positions,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        position_jitter_std=position_jitter_std,
        projection=projection,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_sample_loss": [],
        "val_loss": [],
        "val_sample_loss": [],
    }

    best_state: Optional[dict[str, Tensor]] = None
    best_metric = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_sample_loss = 0.0
        running_items = 0

        for batch_masses, batch_positions, batch_labels in train_loader:
            batch_masses = batch_masses.to(model_device)
            batch_positions = batch_positions.to(model_device)
            batch_labels = batch_labels.to(model_device)

            tau = sample_score_matching_noise_levels(
                int(batch_masses.shape[0]),
                tau_min=tau_min,
                tau_max=tau_max,
                sampling=tau_sampling,
                device=model_device,
                dtype=batch_positions.dtype,
            )
            noisy_positions, target_score, _ = perturb_weighted_point_cloud_positions(
                batch_masses,
                batch_positions,
                tau,
                projection=projection,
                score_target=score_target,
                torus_image_radius=torus_image_radius,
                torus_fourier_modes=torus_fourier_modes,
                torus_fourier_switch_time=torus_fourier_switch_time,
            )
            predicted_score = model(batch_masses, noisy_positions, tau, batch_labels)
            loss, metrics = weighted_denoising_score_matching_loss(
                predicted_score,
                target_score,
                batch_masses,
                tau,
                time_weighting=time_weighting,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                if max_grad_norm <= 0.0:
                    raise ValueError("max_grad_norm must be positive when provided")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            batch_size_actual = int(batch_masses.shape[0])
            running_loss += float(loss.item()) * batch_size_actual
            running_sample_loss += float(metrics["sample_loss"]) * batch_size_actual
            running_items += batch_size_actual

        train_loss = running_loss / max(running_items, 1)
        train_sample_loss = running_sample_loss / max(running_items, 1)
        history["train_loss"].append(float(train_loss))
        history["train_sample_loss"].append(float(train_sample_loss))

        if val_masses is not None and val_positions is not None and val_labels is not None:
            val_metrics = evaluate_score_model(
                model,
                val_masses,
                val_positions,
                val_labels,
                batch_size=batch_size,
                tau_min=tau_min,
                tau_max=tau_max,
                tau_sampling=tau_sampling,
                projection=projection,
                score_target=score_target,
                torus_image_radius=torus_image_radius,
                torus_fourier_modes=torus_fourier_modes,
                torus_fourier_switch_time=torus_fourier_switch_time,
                time_weighting=time_weighting,
                device=model_device,
            )
            val_loss = float(val_metrics["loss"])
            val_sample_loss = float(val_metrics["sample_loss"])
            history["val_loss"].append(val_loss)
            history["val_sample_loss"].append(val_sample_loss)
            selection_metric = val_loss
        else:
            history["val_loss"].append(float("nan"))
            history["val_sample_loss"].append(float("nan"))
            selection_metric = train_loss

        if selection_metric < best_metric:
            best_metric = selection_metric
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            val_message = (
                f", val loss = {history['val_loss'][-1]:.4f}, val sample loss = {history['val_sample_loss'][-1]:.4f}"
                if np.isfinite(history["val_loss"][-1])
                else ""
            )
            print(
                f"[score] epoch {epoch + 1:03d}/{epochs:03d}: "
                f"train loss = {train_loss:.4f}, train sample loss = {train_sample_loss:.4f}{val_message}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------------------
# Reverse-time generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def bridge_reverse_step(
    model: ConditionalScoreSetNetwork,
    masses: Tensor,
    positions: Tensor,
    labels: Tensor,
    tau: float,
    tau_next: float,
    *,
    state_projection: str = "reflect",
    diffusion_temperature: float = 1.0,
    score_scale: float = 1.0,
) -> Tensor:
    """Single Tweedie / Brownian-bridge reverse step.

    Given a current state ``Y_tau`` at noise level ``tau``, this first forms the
    Tweedie-style clean estimate

        X0_hat = Y_tau + 2 tau score_scale * S_theta(tau, s, Y_tau),

    then samples the bridge marginal at the smaller noise level ``tau_next``.
    When ``tau_next == 0`` this returns ``X0_hat`` itself.  For
    ``state_projection='wrap'`` the interpolation uses centered torus
    differences, so the bridge follows the shortest wrapped displacement.
    """
    if tau <= 0.0 or not np.isfinite(tau):
        raise ValueError("tau must be positive and finite")
    if tau_next < 0.0 or not np.isfinite(tau_next):
        raise ValueError("tau_next must be non-negative and finite")
    if tau_next > tau:
        raise ValueError("tau_next must satisfy 0 <= tau_next <= tau")
    if diffusion_temperature < 0.0 or not np.isfinite(diffusion_temperature):
        raise ValueError("diffusion_temperature must be non-negative and finite")
    if score_scale <= 0.0 or not np.isfinite(score_scale):
        raise ValueError("score_scale must be positive and finite")
    if masses.ndim != 2:
        raise ValueError("masses must have shape (B, K)")
    if positions.ndim != 3 or positions.shape[:2] != masses.shape or positions.shape[2] != 2:
        raise ValueError("positions must have shape (B, K, 2) and match masses")
    if labels.ndim != 1 or labels.shape != (masses.shape[0],):
        raise ValueError("labels must have shape (B,)")

    batch_size = int(masses.shape[0])
    tau_tensor = torch.full(
        (batch_size,),
        float(tau),
        device=positions.device,
        dtype=positions.dtype,
    )
    score = model(masses, positions, tau_tensor, labels)
    x0_hat = positions + (2.0 * float(tau) * float(score_scale)) * score
    x0_hat = project_positions(x0_hat, mode=state_projection)

    if tau_next <= 0.0:
        return x0_hat

    ratio = float(tau_next) / float(tau)
    if state_projection == "wrap":
        delta = _centered_torus_difference(positions - x0_hat)
        mean = x0_hat + ratio * delta
    else:
        mean = x0_hat + ratio * (positions - x0_hat)
    mean = project_positions(mean, mode=state_projection)

    bridge_factor = max(1.0 - ratio, 0.0)
    bridge_variance = (
        2.0 * float(diffusion_temperature) * float(tau_next) * bridge_factor
    ) / masses.clamp_min(1e-8)
    noise_scale = torch.sqrt(bridge_variance.clamp_min(0.0)).unsqueeze(-1)
    if torch.any(noise_scale > 0.0):
        mean = mean + noise_scale * torch.randn_like(positions)
    return project_positions(mean, mode=state_projection)


@torch.no_grad()
def generate_score_matching_point_clouds(
    model: ConditionalScoreSetNetwork,
    mass_bank: Optional[np.ndarray],
    target_labels: np.ndarray,
    *,
    bank_labels: Optional[np.ndarray] = None,
    num_points: Optional[int] = None,
    mass_sampling_mode: str = "bank",
    class_conditional_mass_sampling: bool = True,
    poisson_dirichlet_beta: Optional[float] = None,
    poisson_dirichlet_max_terms: Optional[int] = None,
    horizon: float = 5e-4,
    step_size: float = 5e-6,
    initial_position_mode: str = "uniform",
    initial_position_scale: float = 0.12,
    initial_position_bank: Optional[np.ndarray] = None,
    initial_position_bank_labels: Optional[np.ndarray] = None,
    class_conditional_initial_positions: bool = False,
    joint_bank_sampling: bool = False,
    initial_position_jitter: float = 0.02,
    state_projection: str = "reflect",
    diffusion_temperature: float = 1.0,
    score_scale: float = 1.0,
    sampler_scheme: str = "euler",
    batch_size: int = 64,
    return_trajectories: bool = False,
    rasterize: bool = True,
    image_size: int = 28,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedPointCloudSet:
    """Sample point clouds with a reverse-time score-based sampler.

    ``sampler_scheme='euler'`` uses the original reverse-time Euler--Maruyama
    update

        X_{m+1} = X_m + 2 Δt S_theta(tau_m, s, X_m, y)
                         + sqrt(2 Δt / s) ξ_m.

    ``sampler_scheme='bridge'`` instead forms a Tweedie-style clean estimate

        X0_hat = Y_tau + 2 tau score_scale * S_theta(tau, s, Y_tau),

    then samples the Brownian-bridge marginal at the next smaller noise level.
    This often produces a less noisy final tail than Euler--Maruyama while
    still using the same learned score field.

    ``initial_position_mode='forward_noised_bank'`` is a diagnostic mode: it
    draws a clean empirical mass/position pair and applies the same
    mass-dependent forward noising kernel at ``tau=horizon`` before running the
    reverse sampler.  This should not be used as the final independent generator,
    but it is the most direct test of whether the learned reverse dynamics are
    consistent with the score-matching noising process.
    """
    if horizon <= 0.0 or not np.isfinite(horizon):
        raise ValueError("horizon must be positive and finite")
    if step_size <= 0.0 or not np.isfinite(step_size):
        raise ValueError("step_size must be positive and finite")
    ratio = horizon / step_size
    num_steps = int(round(ratio))
    if num_steps <= 0 or not np.isclose(ratio, num_steps, atol=1e-10, rtol=1e-10):
        raise ValueError("horizon / step_size must be an integer")
    if diffusion_temperature <= 0.0 or not np.isfinite(diffusion_temperature):
        raise ValueError("diffusion_temperature must be positive and finite")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if score_scale <= 0.0 or not np.isfinite(score_scale):
        raise ValueError("score_scale must be positive and finite")
    if sampler_scheme not in {"euler", "bridge"}:
        raise ValueError("sampler_scheme must be one of {'euler', 'bridge'}")
    if joint_bank_sampling and initial_position_mode != "bank":
        raise ValueError("joint_bank_sampling=True requires initial_position_mode='bank'")
    if mass_sampling_mode not in {"bank", "truncated_poisson_dirichlet"}:
        raise ValueError("mass_sampling_mode must be 'bank' or 'truncated_poisson_dirichlet'")
    allowed_initial_modes = {"bank", "uniform", "centered_gaussian", "forward_noised_bank"}
    if initial_position_mode not in allowed_initial_modes:
        raise ValueError(
            "initial_position_mode must be 'bank', 'uniform', 'centered_gaussian', "
            "or 'forward_noised_bank'"
        )
    if initial_position_mode == "forward_noised_bank" and mass_sampling_mode != "bank":
        raise ValueError(
            "initial_position_mode='forward_noised_bank' requires mass_sampling_mode='bank' "
            "so masses and clean positions can be drawn from the same empirical sample"
        )

    labels = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng() if rng is None else rng

    inferred_num_points = None
    if num_points is not None:
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        inferred_num_points = int(num_points)
    if mass_bank is not None:
        mass_bank_arr = np.asarray(mass_bank, dtype=np.float64)
        if mass_bank_arr.ndim != 2:
            raise ValueError("mass_bank must have shape (N, K)")
        if inferred_num_points is None:
            inferred_num_points = int(mass_bank_arr.shape[1])
        elif inferred_num_points != int(mass_bank_arr.shape[1]):
            raise ValueError("num_points and mass_bank disagree about K")
    else:
        mass_bank_arr = None
    if initial_position_bank is not None:
        position_bank_arr = np.asarray(initial_position_bank, dtype=np.float64)
        if position_bank_arr.ndim != 3 or position_bank_arr.shape[2] != 2:
            raise ValueError("initial_position_bank must have shape (N, K, 2)")
        if inferred_num_points is None:
            inferred_num_points = int(position_bank_arr.shape[1])
        elif inferred_num_points != int(position_bank_arr.shape[1]):
            raise ValueError("num_points and initial_position_bank disagree about K")
    else:
        position_bank_arr = None
    if inferred_num_points is None:
        raise ValueError("pass num_points explicitly or provide mass_bank / initial_position_bank")

    if initial_position_mode == "forward_noised_bank":
        if mass_bank_arr is None or position_bank_arr is None:
            raise ValueError(
                "initial_position_mode='forward_noised_bank' requires both mass_bank and initial_position_bank"
            )
        joint_bank_labels = bank_labels if bank_labels is not None else initial_position_bank_labels
        if class_conditional_mass_sampling and joint_bank_labels is None:
            raise ValueError(
                "bank_labels or initial_position_bank_labels are required for class-conditional "
                "forward_noised_bank starts"
            )
        masses_np, clean_positions_np = draw_joint_mass_position_vectors_from_bank(
            mass_bank_arr,
            position_bank_arr,
            labels,
            bank_labels=joint_bank_labels,
            class_conditional=class_conditional_mass_sampling,
            rng=rng,
        )
        initial_positions_np = add_forward_noise_to_positions(
            masses_np,
            clean_positions_np,
            horizon,
            projection=state_projection,
            rng=rng,
        )
    elif initial_position_mode == "bank" and joint_bank_sampling:
        if mass_sampling_mode != "bank":
            raise ValueError("joint_bank_sampling=True requires mass_sampling_mode='bank'")
        if mass_bank_arr is None or position_bank_arr is None:
            raise ValueError("joint_bank_sampling=True requires both mass_bank and initial_position_bank")
        joint_bank_labels = bank_labels if bank_labels is not None else initial_position_bank_labels
        masses_np, initial_positions_np = draw_joint_mass_position_vectors_from_bank(
            mass_bank_arr,
            position_bank_arr,
            labels,
            bank_labels=joint_bank_labels,
            class_conditional=class_conditional_mass_sampling,
            rng=rng,
        )
        if initial_position_jitter > 0.0:
            initial_positions_np = initial_positions_np + initial_position_jitter * rng.normal(
                size=initial_positions_np.shape
            )
    else:
        if mass_sampling_mode == "bank":
            if mass_bank_arr is None:
                raise ValueError("mass_bank is required when mass_sampling_mode='bank'")
            masses_np = draw_mass_vectors_from_bank(
                mass_bank_arr,
                labels,
                bank_labels=bank_labels,
                class_conditional=class_conditional_mass_sampling,
                rng=rng,
            )
        else:
            masses_np = sample_truncated_poisson_dirichlet_masses(
                len(labels),
                inferred_num_points,
                beta=poisson_dirichlet_beta,
                max_terms=poisson_dirichlet_max_terms,
                rng=rng,
            )

        num_samples, num_points_resolved = masses_np.shape
        if initial_position_mode == "bank":
            if position_bank_arr is None:
                raise ValueError("initial_position_bank is required when initial_position_mode='bank'")
            initial_positions_np = draw_position_vectors_from_bank(
                position_bank_arr,
                labels,
                bank_labels=initial_position_bank_labels,
                class_conditional=class_conditional_initial_positions,
                jitter_std=initial_position_jitter,
                projection=state_projection,
                rng=rng,
            )
        elif initial_position_mode in {"uniform", "centered_gaussian"}:
            initial_positions_np = sample_initial_positions(
                num_samples,
                num_points_resolved,
                mode=initial_position_mode,
                scale=initial_position_scale,
                rng=rng,
            )
        else:
            # The allowed modes are checked above.  This branch is kept as a
            # defensive guard in case new modes are added without an initializer.
            raise ValueError(f"unhandled initial_position_mode={initial_position_mode!r}")

    initial_positions_np = np.asarray(
        project_positions(initial_positions_np, mode=state_projection),
        dtype=np.float64,
    )
    num_samples, num_points_resolved = masses_np.shape

    model_device = _resolve_device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()

    masses = torch.from_numpy(masses_np).to(device=model_device, dtype=torch.float32)
    positions = torch.from_numpy(initial_positions_np).to(device=model_device, dtype=torch.float32)
    label_tensor = torch.from_numpy(labels).to(device=model_device, dtype=torch.long)

    if return_trajectories:
        trajectories = np.empty((num_steps + 1, num_samples, num_points_resolved, 2), dtype=np.float64)
        trajectories[0] = initial_positions_np
    else:
        trajectories = None

    times = np.linspace(0.0, horizon, num_steps + 1, dtype=np.float64)

    for step in range(num_steps):
        tau_value = float(horizon - times[step])
        tau_next = max(float(horizon - times[step + 1]), 0.0)
        for start in range(0, num_samples, batch_size):
            stop = min(start + batch_size, num_samples)
            batch_masses = masses[start:stop]
            batch_positions = positions[start:stop]
            batch_labels = label_tensor[start:stop]

            if sampler_scheme == "euler":
                tau = torch.full(
                    (stop - start,),
                    tau_value,
                    device=model_device,
                    dtype=batch_positions.dtype,
                )
                score = model(batch_masses, batch_positions, tau, batch_labels)
                drift = 2.0 * float(score_scale) * score
                noise_scale = torch.sqrt((2.0 * diffusion_temperature * step_size) / batch_masses).unsqueeze(-1)
                batch_positions = batch_positions + step_size * drift + noise_scale * torch.randn_like(batch_positions)
                batch_positions = project_positions(batch_positions, mode=state_projection)
            else:
                batch_positions = bridge_reverse_step(
                    model,
                    batch_masses,
                    batch_positions,
                    batch_labels,
                    tau_value,
                    tau_next,
                    state_projection=state_projection,
                    diffusion_temperature=diffusion_temperature,
                    score_scale=score_scale,
                )
            positions[start:stop] = batch_positions

        if trajectories is not None:
            trajectories[step + 1] = positions.detach().cpu().numpy().astype(np.float64)

    final_positions = positions.detach().cpu().numpy().astype(np.float64)
    final_images = None
    if rasterize:
        final_positions_for_raster = np.asarray(project_positions(final_positions, mode="clip"), dtype=np.float64)
        final_images = rasterize_weighted_point_clouds(
            masses_np,
            final_positions_for_raster,
            image_size=image_size,
        )

    if was_training:
        model.train()
    return GeneratedPointCloudSet(
        masses=masses_np.astype(np.float64),
        positions=final_positions,
        labels=labels.astype(np.int64),
        images=final_images,
        trajectories=trajectories,
    )


@torch.no_grad()
def generate_balanced_score_matching_dataset(
    model: ConditionalScoreSetNetwork,
    mass_bank: Optional[np.ndarray],
    *,
    bank_labels: Optional[np.ndarray] = None,
    num_points: Optional[int] = None,
    num_per_class: int,
    mass_sampling_mode: str = "bank",
    class_conditional_mass_sampling: bool = True,
    poisson_dirichlet_beta: Optional[float] = None,
    poisson_dirichlet_max_terms: Optional[int] = None,
    horizon: float = 5e-4,
    step_size: float = 5e-6,
    initial_position_mode: str = "uniform",
    initial_position_scale: float = 0.12,
    initial_position_bank: Optional[np.ndarray] = None,
    initial_position_bank_labels: Optional[np.ndarray] = None,
    class_conditional_initial_positions: bool = False,
    joint_bank_sampling: bool = False,
    initial_position_jitter: float = 0.02,
    state_projection: str = "reflect",
    diffusion_temperature: float = 1.0,
    score_scale: float = 1.0,
    sampler_scheme: str = "euler",
    batch_size: int = 64,
    rasterize: bool = True,
    image_size: int = 28,
    device: Optional[str | torch.device] = None,
    rng: Optional[np.random.Generator] = None,
) -> GeneratedPointCloudSet:
    """Balanced class-conditional wrapper for score-based generation."""
    labels = np.repeat(np.arange(model.num_classes, dtype=np.int64), num_per_class)
    return generate_score_matching_point_clouds(
        model,
        mass_bank,
        labels,
        bank_labels=bank_labels,
        num_points=num_points,
        mass_sampling_mode=mass_sampling_mode,
        class_conditional_mass_sampling=class_conditional_mass_sampling,
        poisson_dirichlet_beta=poisson_dirichlet_beta,
        poisson_dirichlet_max_terms=poisson_dirichlet_max_terms,
        horizon=horizon,
        step_size=step_size,
        initial_position_mode=initial_position_mode,
        initial_position_scale=initial_position_scale,
        initial_position_bank=initial_position_bank,
        initial_position_bank_labels=initial_position_bank_labels,
        class_conditional_initial_positions=class_conditional_initial_positions,
        joint_bank_sampling=joint_bank_sampling,
        initial_position_jitter=initial_position_jitter,
        state_projection=state_projection,
        diffusion_temperature=diffusion_temperature,
        score_scale=score_scale,
        sampler_scheme=sampler_scheme,
        batch_size=batch_size,
        return_trajectories=False,
        rasterize=rasterize,
        image_size=image_size,
        device=device,
        rng=rng,
    )
