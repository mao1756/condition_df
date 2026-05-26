from __future__ import annotations

r"""Example 10: MNIST generation with a directly learned Eulerian edge flux.

The manuscript's terminal conditioning formula adds the conservative edge flux

    J_e^u(t, s) = (2 / h) theta_e(s) partial_e^h log u_t^h(s)

on top of the free harmonic-mobility finite-volume dynamics.  Example 9 learns
``u_t^h`` and differentiates it.  This module implements the smaller and faster
Experiment 10 variant: learn the two edge-flux channels directly.

A state is a ``28 x 28`` probability vector.  The neural network sees the current
state, remaining time, and digit label, and returns two flux images:

* channel 0: horizontal edge flux from pixel ``(row, col)`` to ``(row, col + 1)``;
* channel 1: vertical edge flux from pixel ``(row, col)`` to ``(row + 1, col)``.

The simulator updates masses by conservative incidence: incoming flux minus
outgoing flux.  Therefore total mass is conserved exactly before the optional
small positivity floor/renormalization used to protect explicit Euler steps.

The training target is a cheap laptop-friendly proxy for the theoretical
conditioning flux: draw a real MNIST target image of the requested label, form a
blurred positive Gibbs terminal score ``g_h(s | x)``, compute the analytic
terminal log-score flux

    (2 / h) theta_e(s) partial_e^h log g_h(s | x),

and regress the U-Net directly onto that two-channel field.  Because the target
image is not an input, the population minimizer is the label-conditional average
conditioning flux.  This is intentionally a small proof-of-concept rather than a
large diffusion model.
"""

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mnist.weighted_point_cloud import load_mnist_arrays, normalize_images_to_measures

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = [
    "DirectFluxMNISTConfig",
    "MNISTMeasureDataset",
    "FluxTrainingBatch",
    "FluxGenerationResult",
    "DirectFluxUNet",
    "natural_horizon",
    "load_mnist_measure_dataset",
    "sample_flux_training_batch",
    "terminal_potential_and_log_gradient_torch",
    "terminal_conditioning_flux_torch",
    "flux_divergence_torch",
    "eulerian_flux_step_torch",
    "direct_flux_matching_loss",
    "train_direct_flux_model",
    "simulate_direct_flux_generation",
    "save_flux_samples_grid",
    "main",
]


@dataclass(frozen=True)
class DirectFluxMNISTConfig:
    """Configuration for the direct-flux MNIST experiment.

    Defaults are intentionally modest: they are meant to fit comfortably on an
    8 GB laptop GPU while still producing useful progress-bar ETAs.
    """

    grid_size: int = 28
    alpha: float = 1.0
    horizon_scale: float = 1.0
    num_steps: int = 192
    limiter_fraction: float = 0.25
    terminal_lambda: float = 3.0
    terminal_floor: float = 1e-3
    blur_sigmas: tuple[float, ...] = (0.75, 1.5)
    blur_weights: tuple[float, ...] = (0.5, 0.5)
    mass_floor: float = 1e-8
    source_concentration: float = 1.0
    source_uniform_mix: float = 0.15
    state_jitter_weight: float = 0.02
    bridge_power: float = 1.0
    flux_scale: float = 100.0
    target_flux_clip: float = 5.0
    divergence_loss_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.grid_size <= 1:
            raise ValueError("grid_size must be at least 2")
        if self.grid_size % 4 != 0:
            raise ValueError("grid_size must be divisible by 4 for the small U-Net")
        if self.grid_size % 2 != 0:
            raise ValueError("grid_size must be even for four-color edge splitting")
        if self.alpha <= 0.0 or not math.isfinite(self.alpha):
            raise ValueError("alpha must be positive and finite")
        if self.horizon_scale <= 0.0 or not math.isfinite(self.horizon_scale):
            raise ValueError("horizon_scale must be positive and finite")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not (0.0 < self.limiter_fraction <= 1.0):
            raise ValueError("limiter_fraction must be in (0, 1]")
        if self.terminal_lambda < 0.0 or not math.isfinite(self.terminal_lambda):
            raise ValueError("terminal_lambda must be non-negative and finite")
        if not (0.0 < self.terminal_floor < 1.0):
            raise ValueError("terminal_floor must be in (0, 1)")
        if len(self.blur_sigmas) != len(self.blur_weights):
            raise ValueError("blur_sigmas and blur_weights must have equal length")
        if len(self.blur_sigmas) == 0:
            raise ValueError("at least one blur scale is required")
        if any(sigma < 0.0 for sigma in self.blur_sigmas):
            raise ValueError("blur_sigmas must be non-negative")
        if any(weight < 0.0 for weight in self.blur_weights):
            raise ValueError("blur_weights must be non-negative")
        if sum(self.blur_weights) <= 0.0:
            raise ValueError("at least one blur weight must be positive")
        if self.mass_floor <= 0.0:
            raise ValueError("mass_floor must be positive")
        if self.source_concentration <= 0.0:
            raise ValueError("source_concentration must be positive")
        if not (0.0 <= self.source_uniform_mix < 1.0):
            raise ValueError("source_uniform_mix must be in [0, 1)")
        if not (0.0 <= self.state_jitter_weight < 1.0):
            raise ValueError("state_jitter_weight must be in [0, 1)")
        if self.bridge_power <= 0.0:
            raise ValueError("bridge_power must be positive")
        if self.flux_scale <= 0.0:
            raise ValueError("flux_scale must be positive")
        if self.target_flux_clip <= 0.0:
            raise ValueError("target_flux_clip must be positive")
        if self.divergence_loss_weight < 0.0:
            raise ValueError("divergence_loss_weight must be non-negative")


@dataclass(frozen=True)
class MNISTMeasureDataset:
    """Raster MNIST images normalized as probability measures."""

    train_images: FloatArray
    train_labels: IntArray
    test_images: FloatArray | None = None
    test_labels: IntArray | None = None

    def __post_init__(self) -> None:
        if self.train_images.ndim != 3:
            raise ValueError("train_images must have shape (N, H, W)")
        if self.train_labels.shape != (self.train_images.shape[0],):
            raise ValueError("train_labels must have shape (N,)")
        if self.test_images is not None and self.test_images.ndim != 3:
            raise ValueError("test_images must have shape (N, H, W)")
        if self.test_images is not None and self.test_labels is not None:
            if self.test_labels.shape != (self.test_images.shape[0],):
                raise ValueError("test_labels must have shape (N_test,)")


@dataclass(frozen=True)
class FluxTrainingBatch:
    """One direct-flux regression batch."""

    tau: Tensor
    states: Tensor
    labels: Tensor
    targets: Tensor


@dataclass(frozen=True)
class FluxGenerationResult:
    """Generated image measures and optional trajectory."""

    samples: FloatArray
    labels: IntArray
    trajectory: FloatArray | None
    clipping_fraction: float


@dataclass(frozen=True)
class _TorchEdgeClass:
    tails: Tensor
    heads: Tensor
    flux_indices: Tensor


def natural_horizon(config: DirectFluxMNISTConfig) -> float:
    """Return the fixed-grid bridge horizon used by the Eulerian simulator."""
    n = float(config.grid_size)
    return float(config.horizon_scale) / ((2.0 * float(config.alpha) + 1.0) * n * n)


# ---------------------------------------------------------------------------
# Progress bar with a no-dependency fallback
# ---------------------------------------------------------------------------


class _SimpleProgress:
    def __init__(self, iterable: Sequence[int], *, total: int, desc: str, disable: bool) -> None:
        self.iterable = iterable
        self.total = int(total)
        self.desc = desc
        self.disable = disable
        self.start = time.perf_counter()
        self.count = 0
        self.postfix = ""
        self.print_every = max(1, self.total // 50)

    def __iter__(self) -> Iterator[int]:
        for item in self.iterable:
            yield item
            self.update(1)
        if not self.disable:
            print()

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        if self.disable:
            return
        if self.count != self.total and self.count % self.print_every != 0:
            return
        elapsed = max(time.perf_counter() - self.start, 1e-12)
        rate = self.count / elapsed
        remaining = max(self.total - self.count, 0) / max(rate, 1e-12)
        print(
            f"\r{self.desc}: {self.count}/{self.total} "
            f"[{elapsed:6.1f}s elapsed, {remaining:6.1f}s ETA] {self.postfix}",
            end="",
            flush=True,
        )

    def set_postfix(self, **kwargs: float | str) -> None:
        pieces = []
        for key, value in kwargs.items():
            if isinstance(value, float):
                pieces.append(f"{key}={value:.4g}")
            else:
                pieces.append(f"{key}={value}")
        self.postfix = " ".join(pieces)




def _make_cuda_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover - older PyTorch signature.
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _cuda_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover - older PyTorch signature.
            return torch.amp.autocast(enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def _progress(iterable: Sequence[int], *, total: int, desc: str, disable: bool = False):
    if disable:
        return _SimpleProgress(iterable, total=total, desc=desc, disable=True)
    try:  # pragma: no cover - depends on optional local tqdm install.
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:  # pragma: no cover - fallback exercised when tqdm is absent.
        return _SimpleProgress(iterable, total=total, desc=desc, disable=False)


# ---------------------------------------------------------------------------
# MNIST loading
# ---------------------------------------------------------------------------


def _read_mnist_arff_measures(
    arff_path: str | Path,
    *,
    max_samples: int | None = None,
    per_class: int | None = None,
) -> tuple[FloatArray, IntArray]:
    """Read OpenML ``mnist_784.arff`` without depending on scipy/sklearn."""
    path = Path(arff_path)
    if not path.exists():
        raise FileNotFoundError(f"MNIST ARFF file not found: {path}")
    images: list[np.ndarray] = []
    labels: list[int] = []
    counts = {digit: 0 for digit in range(10)}
    in_data = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not in_data:
                if stripped.upper() == "@DATA":
                    in_data = True
                continue
            if not stripped:
                continue
            values = np.fromstring(stripped, sep=",")
            if values.size < 785:
                continue
            label = int(values[-1])
            if per_class is not None and counts[label] >= per_class:
                continue
            images.append(values[:784].reshape(28, 28).astype(np.float64) / 255.0)
            labels.append(label)
            counts[label] += 1
            if per_class is not None and all(counts[digit] >= per_class for digit in range(10)):
                break
            if per_class is None and max_samples is not None and len(images) >= max_samples:
                break
    if not images:
        raise RuntimeError(f"No MNIST examples were read from {path}")
    measures = normalize_images_to_measures(np.stack(images, axis=0))
    return np.asarray(measures, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _balanced_subset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    per_class: int | None,
    max_samples: int | None,
    seed: int,
) -> tuple[FloatArray, IntArray]:
    labels = np.asarray(labels, dtype=np.int64)
    if per_class is not None:
        rng = np.random.default_rng(seed)
        indices: list[int] = []
        for digit in range(10):
            cls = np.flatnonzero(labels == digit)
            if cls.size == 0:
                continue
            take = min(int(per_class), int(cls.size))
            indices.extend(rng.choice(cls, size=take, replace=False).tolist())
        rng.shuffle(indices)
        idx = np.asarray(indices, dtype=np.int64)
        return np.asarray(images[idx], dtype=np.float64), np.asarray(labels[idx], dtype=np.int64)
    if max_samples is not None and images.shape[0] > max_samples:
        return np.asarray(images[: int(max_samples)], dtype=np.float64), np.asarray(labels[: int(max_samples)], dtype=np.int64)
    return np.asarray(images, dtype=np.float64), labels


def load_mnist_measure_dataset(
    data_root: str | Path = "mnist_data",
    *,
    max_train: int | None = None,
    examples_per_class: int | None = 1000,
    download: bool = False,
    seed: int = 0,
) -> MNISTMeasureDataset:
    """Load MNIST images as probability measures.

    The fastest path for this repository copy is ``mnist_data/mnist_784.arff``.
    If that file is missing, the function falls back to the existing IDX loader
    in :mod:`mnist.weighted_point_cloud`.
    """
    root = Path(data_root)
    arff_path = root if root.suffix.lower() == ".arff" else root / "mnist_784.arff"
    if arff_path.exists():
        train_images, train_labels = _read_mnist_arff_measures(
            arff_path,
            max_samples=max_train,
            per_class=examples_per_class,
        )
        return MNISTMeasureDataset(train_images=train_images, train_labels=train_labels)

    arrays = load_mnist_arrays(root, download=download, normalize_to_measure=True)
    train_images, train_labels = _balanced_subset(
        np.asarray(arrays["train_images"], dtype=np.float64),
        np.asarray(arrays["train_labels"], dtype=np.int64),
        per_class=examples_per_class,
        max_samples=max_train,
        seed=seed,
    )
    return MNISTMeasureDataset(
        train_images=train_images,
        train_labels=train_labels,
        test_images=np.asarray(arrays["test_images"], dtype=np.float64),
        test_labels=np.asarray(arrays["test_labels"], dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Small direct-flux U-Net
# ---------------------------------------------------------------------------


def _num_groups(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DirectFluxUNet(nn.Module):
    """Small label-conditioned U-Net that predicts normalized edge fluxes."""

    def __init__(
        self,
        config: DirectFluxMNISTConfig,
        *,
        base_channels: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.config = config
        self.num_classes = int(num_classes)
        channels = int(base_channels)
        in_channels = 1 + 1 + 1 + self.num_classes
        self.enc1 = _ConvBlock(in_channels, channels)
        self.down1 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.enc2 = _ConvBlock(2 * channels, 2 * channels)
        self.down2 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.mid = _ConvBlock(4 * channels, 4 * channels)
        self.up2 = nn.ConvTranspose2d(4 * channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = _ConvBlock(4 * channels, 2 * channels)
        self.up1 = nn.ConvTranspose2d(2 * channels, channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = _ConvBlock(2 * channels, channels)
        self.out = nn.Conv2d(channels, 2, kernel_size=3, padding=1, padding_mode="circular")

    def _inputs(self, tau: Tensor | float, masses: Tensor, labels: Tensor) -> Tensor:
        if masses.ndim != 2:
            raise ValueError("masses must have shape (B, N)")
        batch_size = int(masses.shape[0])
        n = int(self.config.grid_size)
        if masses.shape[1] != n * n:
            raise ValueError("masses have the wrong number of pixels")
        labels = labels.to(device=masses.device, dtype=torch.long).reshape(batch_size)
        if torch.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError("labels are outside the configured class range")
        tau_tensor = torch.as_tensor(tau, dtype=masses.dtype, device=masses.device)
        if tau_tensor.ndim == 0:
            tau_tensor = tau_tensor.repeat(batch_size)
        if tau_tensor.shape != (batch_size,):
            raise ValueError("tau must be scalar or have shape (B,)")

        density = masses.reshape(batch_size, 1, n, n) * float(n * n)
        log_density = torch.log(density.clamp_min(float(self.config.mass_floor)))
        tau_channel = (tau_tensor / max(natural_horizon(self.config), 1e-12)).view(batch_size, 1, 1, 1)
        tau_channel = tau_channel.expand(batch_size, 1, n, n)
        label_planes = F.one_hot(labels, num_classes=self.num_classes).to(dtype=masses.dtype)
        label_planes = label_planes.view(batch_size, self.num_classes, 1, 1).expand(
            batch_size, self.num_classes, n, n
        )
        return torch.cat([density, log_density, tau_channel, label_planes], dim=1)

    def forward(self, tau: Tensor | float, masses: Tensor, labels: Tensor) -> Tensor:
        """Return normalized flux channels with shape ``(B, 2, H, W)``."""
        x1 = self.enc1(self._inputs(tau, masses, labels))
        x2 = self.enc2(F.silu(self.down1(x1)))
        x3 = self.mid(F.silu(self.down2(x2)))
        y2 = self.up2(x3)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        return self.out(y1)

    def predict_flux(self, tau: Tensor | float, masses: Tensor, labels: Tensor) -> Tensor:
        """Return physical flux rates, not normalized training targets."""
        return float(self.config.flux_scale) * self.forward(tau, masses, labels)


# ---------------------------------------------------------------------------
# Terminal proxy and direct flux target
# ---------------------------------------------------------------------------


def _gaussian_kernel1d_torch(sigma: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (offsets / float(sigma)) ** 2)
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


def _periodic_gaussian_blur_torch(images: Tensor, *, sigma: float) -> Tensor:
    if sigma <= 0.0:
        return images
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W)")
    kernel = _gaussian_kernel1d_torch(sigma, device=images.device, dtype=images.dtype)
    radius = int((kernel.numel() - 1) // 2)
    channels = int(images.shape[1])
    kernel_x = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    out = F.pad(images, (radius, radius, 0, 0), mode="circular")
    out = F.conv2d(out, kernel_x, groups=channels)
    kernel_y = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    out = F.pad(out, (0, 0, radius, radius), mode="circular")
    return F.conv2d(out, kernel_y, groups=channels)


def terminal_potential_and_log_gradient_torch(
    masses: Tensor,
    target_masses: Tensor,
    config: DirectFluxMNISTConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``Phi_h``, ``g_h``, and ``grad_s log g_h`` for the Gibbs terminal score."""
    if masses.ndim != 2 or target_masses.ndim != 2:
        raise ValueError("masses and target_masses must have shape (B, N)")
    if masses.shape != target_masses.shape:
        raise ValueError("masses and target_masses must have the same shape")
    n = int(config.grid_size)
    if masses.shape[1] != n * n:
        raise ValueError("masses have the wrong number of pixels")
    num_pixels = float(n * n)
    density = masses.reshape(-1, 1, n, n) * num_pixels
    target_density = target_masses.reshape(-1, 1, n, n) * num_pixels
    diff = density - target_density
    weight_sum = float(sum(config.blur_weights))
    phi = torch.zeros((masses.shape[0],), device=masses.device, dtype=masses.dtype)
    grad_density = torch.zeros_like(diff)
    for sigma, weight in zip(config.blur_sigmas, config.blur_weights):
        w = float(weight) / weight_sum
        blurred = _periodic_gaussian_blur_torch(diff, sigma=float(sigma))
        phi = phi + w * blurred.square().mean(dim=(1, 2, 3))
        grad_density = grad_density + w * (2.0 / num_pixels) * _periodic_gaussian_blur_torch(
            blurred, sigma=float(sigma)
        )
    grad_phi = (grad_density * num_pixels).reshape_as(masses)
    exp_part = torch.exp(-float(config.terminal_lambda) * phi)
    score = float(config.terminal_floor) + (1.0 - float(config.terminal_floor)) * exp_part
    log_factor = ((1.0 - float(config.terminal_floor)) * exp_part / score).view(-1, 1)
    grad_log_score = -float(config.terminal_lambda) * log_factor * grad_phi
    return phi, score, grad_log_score


def harmonic_mobility_channels(masses: Tensor, config: DirectFluxMNISTConfig) -> Tensor:
    """Return harmonic edge mobilities for horizontal and vertical edges."""
    if masses.ndim != 2:
        raise ValueError("masses must have shape (B, N)")
    n = int(config.grid_size)
    image = masses.reshape(-1, 1, n, n)
    a = image[:, 0]
    bx = torch.roll(a, shifts=-1, dims=-1)
    by = torch.roll(a, shifts=-1, dims=-2)
    tiny = float(config.mass_floor)
    hx = torch.where(a + bx > tiny, a * bx / (a + bx).clamp_min(tiny), torch.zeros_like(a))
    hy = torch.where(a + by > tiny, a * by / (a + by).clamp_min(tiny), torch.zeros_like(a))
    kappa = (2.0 * float(config.alpha) + 1.0) / float(config.alpha)
    return kappa * torch.stack([hx, hy], dim=1)


def terminal_conditioning_flux_torch(
    masses: Tensor,
    target_masses: Tensor,
    config: DirectFluxMNISTConfig,
) -> Tensor:
    """Analytic two-channel proxy for ``(2 / h) theta partial^h log u``.

    The proxy uses ``u_t^h`` replaced by the positive terminal score ``g_h``.
    """
    _, _, grad_log_score = terminal_potential_and_log_gradient_torch(masses, target_masses, config)
    n = int(config.grid_size)
    grad_img = grad_log_score.reshape(-1, 1, n, n)[:, 0]
    delta_x = torch.roll(grad_img, shifts=-1, dims=-1) - grad_img
    delta_y = torch.roll(grad_img, shifts=-1, dims=-2) - grad_img
    theta = harmonic_mobility_channels(masses, config)
    inv_h2 = float(n * n)
    return 2.0 * theta * inv_h2 * torch.stack([delta_x, delta_y], dim=1)


def flux_divergence_torch(flux: Tensor) -> Tensor:
    """Conservative divergence ``incoming - outgoing`` for two edge channels."""
    if flux.ndim != 4 or flux.shape[1] != 2:
        raise ValueError("flux must have shape (B, 2, H, W)")
    fx = flux[:, 0]
    fy = flux[:, 1]
    return torch.roll(fx, shifts=1, dims=-1) - fx + torch.roll(fy, shifts=1, dims=-2) - fy


# ---------------------------------------------------------------------------
# Training batches, loss, and simulator
# ---------------------------------------------------------------------------


def _sample_simplex_noise(
    batch_size: int,
    num_pixels: int,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    concentration = torch.full(
        (batch_size, num_pixels),
        float(config.source_concentration),
        device=device,
        dtype=dtype,
    )
    samples = torch.distributions.Gamma(concentration, torch.ones_like(concentration)).sample()
    samples = samples.clamp_min(float(config.mass_floor))
    samples = samples / samples.sum(dim=1, keepdim=True).clamp_min(float(config.mass_floor))
    if config.source_uniform_mix > 0.0:
        uniform = torch.full_like(samples, 1.0 / float(num_pixels))
        samples = (1.0 - float(config.source_uniform_mix)) * samples + float(
            config.source_uniform_mix
        ) * uniform
    return samples / samples.sum(dim=1, keepdim=True).clamp_min(float(config.mass_floor))


def sample_flux_training_batch(
    images: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    batch_size: int,
    device: str | torch.device,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> FluxTrainingBatch:
    """Sample states on noisy source-to-target bridges for direct flux regression."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = np.random.default_rng() if rng is None else rng
    n = int(config.grid_size)
    images_arr = np.asarray(images, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if images_arr.ndim != 3 or images_arr.shape[1:] != (n, n):
        raise ValueError(f"images must have shape (N, {n}, {n})")
    if labels_arr.shape != (images_arr.shape[0],):
        raise ValueError("labels must have shape (N,)")
    idx = rng.integers(0, images_arr.shape[0], size=int(batch_size))
    resolved_device = torch.device(device)
    target = torch.as_tensor(
        images_arr[idx].reshape(int(batch_size), n * n),
        dtype=dtype,
        device=resolved_device,
    )
    target = target.clamp_min(float(config.mass_floor))
    target = target / target.sum(dim=1, keepdim=True).clamp_min(float(config.mass_floor))
    batch_labels = torch.as_tensor(labels_arr[idx], dtype=torch.long, device=resolved_device)
    tau = torch.rand((int(batch_size),), dtype=dtype, device=resolved_device) * float(
        natural_horizon(config)
    )
    mix = (tau / max(natural_horizon(config), 1e-12)).pow(float(config.bridge_power)).view(-1, 1)
    source = _sample_simplex_noise(
        int(batch_size),
        n * n,
        config,
        device=resolved_device,
        dtype=dtype,
    )
    states = (1.0 - mix) * target + mix * source
    if config.state_jitter_weight > 0.0:
        jitter = _sample_simplex_noise(
            int(batch_size),
            n * n,
            config,
            device=resolved_device,
            dtype=dtype,
        )
        states = (1.0 - float(config.state_jitter_weight)) * states + float(
            config.state_jitter_weight
        ) * jitter
    states = states.clamp_min(float(config.mass_floor))
    states = states / states.sum(dim=1, keepdim=True).clamp_min(float(config.mass_floor))
    return FluxTrainingBatch(tau=tau, states=states, labels=batch_labels, targets=target)


def direct_flux_matching_loss(
    model: DirectFluxUNet,
    batch: FluxTrainingBatch,
) -> tuple[Tensor, dict[str, float]]:
    """Return the direct-flux regression loss and scalar diagnostics."""
    config = model.config
    pred_norm = model(batch.tau, batch.states, batch.labels)
    with torch.no_grad():
        target_flux = terminal_conditioning_flux_torch(batch.states, batch.targets, config)
        target_norm = (target_flux / float(config.flux_scale)).clamp(
            -float(config.target_flux_clip), float(config.target_flux_clip)
        )
    flux_loss = F.smooth_l1_loss(pred_norm, target_norm)
    pred_div = flux_divergence_torch(pred_norm)
    target_div = flux_divergence_torch(target_norm)
    div_loss = F.mse_loss(pred_div, target_div)
    loss = flux_loss + float(config.divergence_loss_weight) * div_loss
    with torch.no_grad():
        pred_rms = pred_norm.square().mean().sqrt()
        target_rms = target_norm.square().mean().sqrt()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "flux_loss": float(flux_loss.detach().cpu()),
        "div_loss": float(div_loss.detach().cpu()),
        "pred_rms": float(pred_rms.detach().cpu()),
        "target_rms": float(target_rms.detach().cpu()),
    }


def train_direct_flux_model(
    model: DirectFluxUNet,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    train_steps: int = 1200,
    batch_size: int = 256,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: str | torch.device | None = None,
    seed: int = 0,
    use_amp: bool = True,
    show_progress: bool = True,
) -> dict[str, list[float]]:
    """Train the direct-flux U-Net with an ETA progress bar."""
    if train_steps <= 0 or batch_size <= 0:
        raise ValueError("train_steps and batch_size must be positive")
    resolved_device = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else "cpu" if device is None else device
    )
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(resolved_device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    scaler = _make_cuda_grad_scaler(enabled=amp_enabled)
    history: dict[str, list[float]] = {
        "loss": [],
        "flux_loss": [],
        "div_loss": [],
        "pred_rms": [],
        "target_rms": [],
    }

    bar = _progress(range(int(train_steps)), total=int(train_steps), desc="train flux", disable=not show_progress)
    for _ in bar:
        batch = sample_flux_training_batch(
            images,
            labels,
            model.config,
            batch_size=int(batch_size),
            device=resolved_device,
            rng=rng,
        )
        optimizer.zero_grad(set_to_none=True)
        context = _cuda_autocast(enabled=True) if amp_enabled else nullcontext()
        with context:
            loss, metrics = direct_flux_matching_loss(model, batch)
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        for key in history:
            history[key].append(metrics[key])
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=metrics["loss"], flux=metrics["flux_loss"])
    return history


def _edge_classes_torch(grid_size: int, device: torch.device) -> list[_TorchEdgeClass]:
    n = int(grid_size)
    classes: list[list[tuple[int, int, int]]] = [[], [], [], []]
    for row in range(n):
        for col in range(n):
            tail = row * n + col
            horizontal_head = row * n + ((col + 1) % n)
            vertical_head = ((row + 1) % n) * n + col
            classes[col % 2].append((tail, horizontal_head, tail))
            classes[2 + (row % 2)].append((tail, vertical_head, n * n + tail))
    result: list[_TorchEdgeClass] = []
    for edges in classes:
        result.append(
            _TorchEdgeClass(
                tails=torch.tensor([edge[0] for edge in edges], dtype=torch.long, device=device),
                heads=torch.tensor([edge[1] for edge in edges], dtype=torch.long, device=device),
                flux_indices=torch.tensor([edge[2] for edge in edges], dtype=torch.long, device=device),
            )
        )
    return result


def eulerian_flux_step_torch(
    states: Tensor,
    conditioning_flux: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    deterministic: bool = False,
) -> tuple[Tensor, int, int]:
    """One conservative four-color Euler step with learned conditioning flux."""
    if states.ndim != 2:
        raise ValueError("states must have shape (B, N)")
    if conditioning_flux.ndim != 4 or conditioning_flux.shape[1] != 2:
        raise ValueError("conditioning_flux must have shape (B, 2, H, W)")
    if dt < 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be non-negative and finite")
    if dt == 0.0:
        return states.clone(), 0, 0
    n = int(config.grid_size)
    if states.shape[1] != n * n:
        raise ValueError("states have the wrong number of pixels")
    if conditioning_flux.shape[2:] != (n, n):
        raise ValueError("conditioning_flux has the wrong grid size")
    out = states.clone()
    inv_h2 = float(n * n)
    alpha = float(config.alpha)
    tiny = float(config.mass_floor)
    flat_flux = torch.cat(
        [conditioning_flux[:, 0].reshape(states.shape[0], -1), conditioning_flux[:, 1].reshape(states.shape[0], -1)],
        dim=1,
    )
    clipped = 0
    proposed = 0
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
        learned_flux = flat_flux[:, edge_class.flux_indices]
        d_flux = (free_flux + learned_flux) * float(dt)
        if not deterministic:
            noise_std = torch.sqrt((2.0 * theta * inv_h2 * float(dt)).clamp_min(0.0))
            d_flux = d_flux + noise_std * torch.randn_like(noise_std)
        pos_clip = d_flux > float(config.limiter_fraction) * a
        neg_clip = d_flux < -float(config.limiter_fraction) * b
        clipped += int(pos_clip.count_nonzero().detach().cpu())
        clipped += int(neg_clip.count_nonzero().detach().cpu())
        proposed += int(d_flux.numel())
        d_flux = torch.minimum(d_flux, float(config.limiter_fraction) * a)
        d_flux = torch.maximum(d_flux, -float(config.limiter_fraction) * b)
        out[:, tails] = out[:, tails] - d_flux
        out[:, heads] = out[:, heads] + d_flux
        out = out.clamp_min(tiny)
        out = out / out.sum(dim=1, keepdim=True).clamp_min(tiny)
    return out, clipped, proposed


@torch.no_grad()
def simulate_direct_flux_generation(
    model: DirectFluxUNet,
    labels: Sequence[int] | Tensor | np.ndarray,
    *,
    config: DirectFluxMNISTConfig | None = None,
    num_steps: int | None = None,
    save_every: int = 0,
    deterministic: bool = False,
    device: str | torch.device | None = None,
    seed: int = 0,
    use_amp: bool = True,
    show_progress: bool = True,
) -> FluxGenerationResult:
    """Generate MNIST-like image measures by simulating learned edge-flux dynamics."""
    cfg = model.config if config is None else config
    if cfg != model.config:
        raise ValueError("config must match model.config")
    if save_every < 0:
        raise ValueError("save_every must be non-negative")
    resolved_device = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else "cpu" if device is None else device
    )
    torch.manual_seed(seed)
    model.to(resolved_device)
    model.eval()
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=resolved_device).reshape(-1)
    batch_size = int(labels_t.shape[0])
    n = int(cfg.grid_size)
    steps = int(cfg.num_steps if num_steps is None else num_steps)
    if steps <= 0:
        raise ValueError("num_steps must be positive")
    horizon = natural_horizon(cfg)
    dt = horizon / float(steps)
    states = _sample_simplex_noise(
        batch_size,
        n * n,
        cfg,
        device=resolved_device,
        dtype=torch.float32,
    )
    trajectory: list[np.ndarray] = []
    if save_every > 0:
        trajectory.append(states.detach().cpu().numpy().astype(np.float64))
    clipped = 0
    proposed = 0
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    bar = _progress(range(steps), total=steps, desc="sample flux", disable=not show_progress)
    for step in bar:
        tau_value = max(horizon - float(step) * dt, 0.0)
        tau = torch.full((batch_size,), tau_value, dtype=states.dtype, device=resolved_device)
        context = _cuda_autocast(enabled=True) if amp_enabled else nullcontext()
        with context:
            flux = model.predict_flux(tau, states, labels_t)
        states, c_step, p_step = eulerian_flux_step_torch(
            states,
            flux.float(),
            dt,
            cfg,
            deterministic=deterministic,
        )
        clipped += c_step
        proposed += p_step
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(clip=0.0 if proposed == 0 else clipped / proposed)
        if save_every > 0 and ((step + 1) % int(save_every) == 0 or step + 1 == steps):
            trajectory.append(states.detach().cpu().numpy().astype(np.float64))
    return FluxGenerationResult(
        samples=states.detach().cpu().numpy().astype(np.float64),
        labels=labels_t.detach().cpu().numpy().astype(np.int64),
        trajectory=None if save_every <= 0 else np.stack(trajectory, axis=0),
        clipping_fraction=0.0 if proposed == 0 else float(clipped) / float(proposed),
    )


# ---------------------------------------------------------------------------
# Output helpers and CLI
# ---------------------------------------------------------------------------


def save_flux_samples_grid(
    samples: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    output_path: str | Path,
    *,
    grid_size: int = 28,
    max_images: int = 64,
) -> None:
    """Save a simple preview grid of generated probability-mass images."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        raise RuntimeError("matplotlib is required to save a sample grid") from exc

    arr = np.asarray(samples, dtype=np.float64).reshape(-1, grid_size, grid_size)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    count = min(int(max_images), arr.shape[0])
    cols = min(8, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.55 * rows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for idx in range(count):
        image = arr[idx]
        image = image / max(float(image.max()), 1e-12)
        ax = axes[idx // cols, idx % cols]
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(str(int(labels_arr[idx])), fontsize=8)
        ax.axis("off")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _parse_label_sequence(text: str, count: int) -> list[int]:
    if text == "cycle":
        return [idx % 10 for idx in range(count)]
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("label sequence is empty")
    return [values[idx % len(values)] for idx in range(count)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true", help="Download IDX MNIST if no ARFF file is present.")
    parser.add_argument("--examples-per-class", type=int, default=1000)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--labels", type=str, default="cycle", help="'cycle' or comma-separated labels, e.g. 0,1,2")
    parser.add_argument("--deterministic-sampling", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/experiment10_mnist_flux"))
    args = parser.parse_args(argv)

    config = DirectFluxMNISTConfig()
    device = torch.device(
        "cuda" if args.device is None and torch.cuda.is_available() else "cpu" if args.device is None else args.device
    )
    print(f"Experiment 10 direct-flux MNIST on device={device}")
    print(
        "Laptop-friendly defaults: "
        f"steps={args.train_steps}, batch={args.batch_size}, base_channels={args.base_channels}, "
        f"horizon={natural_horizon(config):.3e}, sample_steps={args.sample_steps or config.num_steps}"
    )
    dataset = load_mnist_measure_dataset(
        args.data_root,
        max_train=args.max_train,
        examples_per_class=args.examples_per_class,
        download=bool(args.download),
        seed=int(args.seed),
    )
    print(f"Loaded {dataset.train_images.shape[0]} training images")

    model = DirectFluxUNet(config, base_channels=int(args.base_channels))
    history = train_direct_flux_model(
        model,
        dataset.train_images,
        dataset.train_labels,
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed),
        use_amp=not bool(args.no_amp),
        show_progress=not bool(args.no_progress),
    )

    labels = _parse_label_sequence(args.labels, int(args.num_samples))
    result = simulate_direct_flux_generation(
        model,
        labels,
        num_steps=args.sample_steps,
        deterministic=bool(args.deterministic_sampling),
        device=device,
        seed=int(args.seed) + 1,
        use_amp=not bool(args.no_amp),
        show_progress=not bool(args.no_progress),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "experiment10_direct_flux_mnist.pt"
    samples_path = args.out_dir / "experiment10_samples.npz"
    png_path = args.out_dir / "experiment10_samples.png"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "history": history,
            "labels": result.labels,
            "clipping_fraction": result.clipping_fraction,
        },
        ckpt_path,
    )
    np.savez_compressed(
        samples_path,
        samples=result.samples,
        labels=result.labels,
        clipping_fraction=np.asarray([result.clipping_fraction], dtype=np.float64),
    )
    try:
        save_flux_samples_grid(result.samples, result.labels, png_path, grid_size=config.grid_size)
        print(f"Saved preview: {png_path}")
    except RuntimeError as exc:
        print(f"Skipping PNG preview: {exc}")
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved samples: {samples_path}")
    print(f"Final clipping fraction: {result.clipping_fraction:.4f}")


if __name__ == "__main__":
    main()
