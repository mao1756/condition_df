from __future__ import annotations

r"""Example 10/10b/10c/10d/10e/10g: MNIST generation with directly learned Eulerian edge fluxes.

The manuscript's fixed-grid h-transform adds a conservative edge flux

    J_e^u(t, s) = (2 / h) theta_e(s) partial_e^h log u_t^h(s)

on top of the free harmonic-mobility finite-volume dynamics.  Example 10 keeps
that Eulerian object, but learns the two edge-flux channels directly:

* channel 0: horizontal flux from pixel ``(row, col)`` to ``(row, col + 1)``;
* channel 1: vertical flux from pixel ``(row, col)`` to ``(row + 1, col)``.

The original terminal-score proxy is still available with
``--target-mode terminal-score``.  The default is now the Experiment 10e
``poisson-ot-flow`` setting: sample a source measure ``z``, choose a digit label,
match sources to same-label MNIST targets by a stable nearest-neighbour rule
in blurred low-resolution features, interpolate
``s_tau = (tau/T) z + (1 - tau/T) x``, and train the network to predict the
minimum-energy periodic edge flux whose conservative divergence equals
``(x - z) / T``.  On-policy batches instead use the residual correction
``(x - s_tau) / tau`` at states actually visited by the current sampler.  The
sampler gets the current mass image, bridge time, digit label, and by default
the initial source/latent mass as persistent conditioning.  It never receives
the target MNIST image at generation time.  Experiment 10e adds stable
nearest/top-k source-target matching, on-policy correction, limiter-aware
one-step losses, node-velocity losses, and adaptive sampling.  Experiment
10g adds stochastic-aware conditioning targets: the network can be trained to
predict the h-transform conditioning flux, i.e. the desired total transport
flux minus the free Dirichlet drift, while on-policy/step losses can use the
same free/noisy SDE weights as the sampler.  The default remains learned-only
for quick deterministic debugging; pass the SDE curriculum/free-aware flags to
train and sample with stochastic dynamics.
"""

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
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

TARGET_MODES = ("poisson-flow", "poisson-ot-flow", "class-mean-flow", "terminal-score")
SOURCE_MODES = ("lowfreq", "uniform-plus-lowfreq", "blurred-dirichlet", "dirichlet", "class-lowres-prior", "target-lowres-prior")
VELOCITY_TARGET_MODES = ("constant", "residual")
TAU_SAMPLING_MODES = ("uniform", "endpoint-mixture")
OT_COST_MODES = ("lowres", "pixel")
OT_MATCH_MODES = ("minibatch", "nearest", "topk")
EDGE_ALPHA_MODES = ("legacy", "grid")

__all__ = [
    "DirectFluxMNISTConfig",
    "MNISTMeasureDataset",
    "SourceBatch",
    "FluxTrainingBatch",
    "FluxGenerationResult",
    "ClasswiseOTCache",
    "OT_MATCH_MODES",
    "EDGE_ALPHA_MODES",
    "DirectFluxUNet",
    "edge_alpha_value",
    "natural_horizon",
    "load_mnist_measure_dataset",
    "sample_flux_training_batch",
    "terminal_potential_and_log_gradient_torch",
    "terminal_conditioning_flux_torch",
    "free_drift_flux_torch",
    "edge_noise_std_channels",
    "step_component_rms_torch",
    "poisson_flux_from_velocity_torch",
    "build_classwise_ot_cache",
    "training_target_flux_torch",
    "make_on_policy_training_batch",
    "flux_divergence_torch",
    "eulerian_flux_step_torch",
    "eulerian_flux_step_differentiable_torch",
    "direct_flux_matching_loss",
    "train_direct_flux_model",
    "simulate_direct_flux_generation",
    "simulate_teacher_flux_rollout",
    "source_batch_diagnostics",
    "nearest_class_mean_metrics",
    "save_flux_samples_grid",
    "save_flux_preview_panel",
    "source_diversity_metrics",
    "main",
]


@dataclass(frozen=True)
class DirectFluxMNISTConfig:
    """Configuration for the direct-flux MNIST experiment.

    Defaults are intentionally modest and designed for an 8 GB laptop GPU.  The
    default 10e path uses stable nearest-matched Poisson-flow, a low-frequency
    source, persistent source conditioning, constant source-target velocity,
    on-policy corrections, limiter-aware losses, and learned-only deterministic
    sampling so that the first serious run tests the learned conservative flux
    before reintroducing the free harmonic drift/noise.
    """

    grid_size: int = 28
    # Legacy experiments used ``alpha`` directly on every edge.  Theory uses
    # alpha_h = beta h^d = beta / grid_size^2 on the 2D MNIST grid.  Keep
    # legacy as the default for continuity and expose the grid mode explicitly.
    alpha: float = 1.0
    beta: float = 1.0
    edge_alpha_mode: str = "legacy"
    horizon_scale: float = 1.0
    num_steps: int = 256
    limiter_fraction: float = 0.25

    target_mode: str = "poisson-ot-flow"
    source_mode: str = "lowfreq"
    source_lowfreq_size: int = 7
    source_blur_sigma: float = 1.0
    condition_on_source: bool = True

    # Experiment 10e: the default target matching is stable across batches.
    # ``nearest`` chooses a same-label target by global low-resolution features;
    # ``topk`` samples among the nearest few; ``minibatch`` keeps the older 10c
    # classwise assignment.
    ot_cost_mode: str = "lowres"
    ot_match_mode: str = "nearest"
    ot_nearest_top_k: int = 1
    ot_lowres_size: int = 7
    ot_blur_sigma: float = 1.0
    ot_com_weight: float = 0.25
    mean_flow_prob: float = 0.15
    mean_flow_warmup_prob: float = 0.20
    mean_flow_warmup_steps: int = 1000

    # Generation starts at the source end, so the default time sampler spends
    # extra training mass near tau=T and a little near tau=0.
    tau_sampling: str = "endpoint-mixture"
    tau_source_prob: float = 0.35
    tau_data_prob: float = 0.15

    # Sampling weights for the full h-transform-style SDE.
    free_weight: float = 0.0
    noise_weight: float = 0.0
    learned_weight: float = 1.0

    # Experiment 10g: stochastic-aware training.  The Poisson teacher gives a
    # desired total transport flux.  When ``free_aware_target`` is true, the
    # network target is the conditioning flux ``J_total - w_free J_free`` so
    # that adding the free drift at sampling time recovers the intended total
    # edge transport.  ``train_*`` default to the sampling weights unless an
    # SDE curriculum is active.
    free_aware_target: bool = False
    train_free_weight: float | None = None
    train_noise_weight: float | None = None
    on_policy_use_free: bool = False
    on_policy_use_noise: bool = False
    stochastic_step_loss: bool = False
    same_noise_step_loss: bool = True
    sde_curriculum: bool = False
    sde_ramp_steps: int = 3000
    target_free_weight: float = 0.02
    target_noise_weight: float = 0.003

    terminal_lambda: float = 3.0
    terminal_floor: float = 1e-3
    blur_sigmas: tuple[float, ...] = (0.75, 1.5)
    blur_weights: tuple[float, ...] = (0.5, 0.5)

    mass_floor: float = 1e-8
    source_concentration: float = 1.0
    source_uniform_mix: float = 0.15
    state_jitter_weight: float = 0.0
    velocity_target: str = "constant"
    min_tau_fraction: float = 0.03
    bridge_power: float = 1.0
    flux_scale: float = 20.0
    target_flux_clip: float = 10.0
    divergence_loss_weight: float = 0.50
    node_loss_weight: float = 1.0
    step_loss_weight: float = 0.25

    on_policy_prob: float = 0.25
    on_policy_warmup_steps: int = 1500
    on_policy_prefix_steps: int = 16

    adaptive_sampling: bool = False
    clip_target: float = 0.03
    max_substeps: int = 4

    def __post_init__(self) -> None:
        if self.grid_size <= 1:
            raise ValueError("grid_size must be at least 2")
        if self.grid_size % 4 != 0:
            raise ValueError("grid_size must be divisible by 4 for the small U-Net")
        if self.grid_size % 2 != 0:
            raise ValueError("grid_size must be even for four-color edge splitting")
        if self.alpha <= 0.0 or not math.isfinite(self.alpha):
            raise ValueError("alpha must be positive and finite")
        if self.beta <= 0.0 or not math.isfinite(self.beta):
            raise ValueError("beta must be positive and finite")
        if self.edge_alpha_mode not in EDGE_ALPHA_MODES:
            raise ValueError(f"edge_alpha_mode must be one of {EDGE_ALPHA_MODES}")
        if self.horizon_scale <= 0.0 or not math.isfinite(self.horizon_scale):
            raise ValueError("horizon_scale must be positive and finite")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not (0.0 < self.limiter_fraction <= 1.0):
            raise ValueError("limiter_fraction must be in (0, 1]")
        if self.target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}")
        if self.source_mode not in SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {SOURCE_MODES}")
        if not (2 <= self.source_lowfreq_size <= self.grid_size):
            raise ValueError("source_lowfreq_size must be between 2 and grid_size")
        if self.source_blur_sigma < 0.0 or not math.isfinite(self.source_blur_sigma):
            raise ValueError("source_blur_sigma must be non-negative and finite")
        if self.ot_cost_mode not in OT_COST_MODES:
            raise ValueError(f"ot_cost_mode must be one of {OT_COST_MODES}")
        if self.ot_match_mode not in OT_MATCH_MODES:
            raise ValueError(f"ot_match_mode must be one of {OT_MATCH_MODES}")
        if self.ot_nearest_top_k <= 0:
            raise ValueError("ot_nearest_top_k must be positive")
        if not (2 <= self.ot_lowres_size <= self.grid_size):
            raise ValueError("ot_lowres_size must be between 2 and grid_size")
        if self.ot_blur_sigma < 0.0 or not math.isfinite(self.ot_blur_sigma):
            raise ValueError("ot_blur_sigma must be non-negative and finite")
        if self.ot_com_weight < 0.0 or not math.isfinite(self.ot_com_weight):
            raise ValueError("ot_com_weight must be non-negative and finite")
        if not (0.0 <= self.mean_flow_prob <= 1.0):
            raise ValueError("mean_flow_prob must be in [0, 1]")
        if not (0.0 <= self.mean_flow_warmup_prob <= 1.0):
            raise ValueError("mean_flow_warmup_prob must be in [0, 1]")
        if self.mean_flow_warmup_steps < 0:
            raise ValueError("mean_flow_warmup_steps must be non-negative")
        if self.tau_sampling not in TAU_SAMPLING_MODES:
            raise ValueError(f"tau_sampling must be one of {TAU_SAMPLING_MODES}")
        if not (0.0 <= self.tau_source_prob <= 1.0):
            raise ValueError("tau_source_prob must be in [0, 1]")
        if not (0.0 <= self.tau_data_prob <= 1.0):
            raise ValueError("tau_data_prob must be in [0, 1]")
        if self.tau_source_prob + self.tau_data_prob > 1.0:
            raise ValueError("tau_source_prob + tau_data_prob must be at most 1")
        if self.free_weight < 0.0 or not math.isfinite(self.free_weight):
            raise ValueError("free_weight must be non-negative and finite")
        if self.noise_weight < 0.0 or not math.isfinite(self.noise_weight):
            raise ValueError("noise_weight must be non-negative and finite")
        if self.learned_weight < 0.0 or not math.isfinite(self.learned_weight):
            raise ValueError("learned_weight must be non-negative and finite")
        if not isinstance(self.free_aware_target, bool):
            raise ValueError("free_aware_target must be a bool")
        if self.train_free_weight is not None and (self.train_free_weight < 0.0 or not math.isfinite(self.train_free_weight)):
            raise ValueError("train_free_weight must be non-negative and finite when set")
        if self.train_noise_weight is not None and (self.train_noise_weight < 0.0 or not math.isfinite(self.train_noise_weight)):
            raise ValueError("train_noise_weight must be non-negative and finite when set")
        if not isinstance(self.on_policy_use_free, bool):
            raise ValueError("on_policy_use_free must be a bool")
        if not isinstance(self.on_policy_use_noise, bool):
            raise ValueError("on_policy_use_noise must be a bool")
        if not isinstance(self.stochastic_step_loss, bool):
            raise ValueError("stochastic_step_loss must be a bool")
        if not isinstance(self.same_noise_step_loss, bool):
            raise ValueError("same_noise_step_loss must be a bool")
        if not isinstance(self.sde_curriculum, bool):
            raise ValueError("sde_curriculum must be a bool")
        if self.sde_ramp_steps < 0:
            raise ValueError("sde_ramp_steps must be non-negative")
        if self.target_free_weight < 0.0 or not math.isfinite(self.target_free_weight):
            raise ValueError("target_free_weight must be non-negative and finite")
        if self.target_noise_weight < 0.0 or not math.isfinite(self.target_noise_weight):
            raise ValueError("target_noise_weight must be non-negative and finite")
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
        if not isinstance(self.condition_on_source, bool):
            raise ValueError("condition_on_source must be a bool")
        if not (0.0 <= self.state_jitter_weight < 1.0):
            raise ValueError("state_jitter_weight must be in [0, 1)")
        if self.velocity_target not in VELOCITY_TARGET_MODES:
            raise ValueError(f"velocity_target must be one of {VELOCITY_TARGET_MODES}")
        if not (0.0 < self.min_tau_fraction <= 1.0):
            raise ValueError("min_tau_fraction must be in (0, 1]")
        if self.bridge_power <= 0.0:
            raise ValueError("bridge_power must be positive")
        if self.flux_scale <= 0.0:
            raise ValueError("flux_scale must be positive")
        if self.target_flux_clip <= 0.0:
            raise ValueError("target_flux_clip must be positive")
        if self.divergence_loss_weight < 0.0:
            raise ValueError("divergence_loss_weight must be non-negative")
        if self.node_loss_weight < 0.0:
            raise ValueError("node_loss_weight must be non-negative")
        if self.step_loss_weight < 0.0:
            raise ValueError("step_loss_weight must be non-negative")
        if not (0.0 <= self.on_policy_prob <= 1.0):
            raise ValueError("on_policy_prob must be in [0, 1]")
        if self.on_policy_warmup_steps < 0:
            raise ValueError("on_policy_warmup_steps must be non-negative")
        if self.on_policy_prefix_steps < 0:
            raise ValueError("on_policy_prefix_steps must be non-negative")
        if not isinstance(self.adaptive_sampling, bool):
            raise ValueError("adaptive_sampling must be a bool")
        if not (0.0 <= self.clip_target <= 1.0):
            raise ValueError("clip_target must be in [0, 1]")
        if self.max_substeps <= 0:
            raise ValueError("max_substeps must be positive")


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
class SourceBatch:
    """A sampled source/latent batch plus optional provenance."""

    masses: Tensor
    indices: IntArray | None = None
    labels: IntArray | None = None


@dataclass(frozen=True)
class FluxTrainingBatch:
    """One direct-flux regression batch."""

    tau: Tensor
    states: Tensor
    labels: Tensor
    targets: Tensor
    sources: Tensor
    source_indices: IntArray | None = None
    source_labels: IntArray | None = None
    target_indices: IntArray | None = None
    target_velocity_mode: str | None = None
    train_free_weight: float = 0.0
    train_noise_weight: float = 0.0


@dataclass(frozen=True)
class FluxGenerationResult:
    """Generated image measures and optional trajectory."""

    samples: FloatArray
    labels: IntArray
    trajectory: FloatArray | None
    clipping_fraction: float
    sources: FloatArray | None = None
    source_indices: IntArray | None = None
    source_labels: IntArray | None = None
    source_unique_count: int | None = None
    source_diversity_l2: float | None = None
    source_pair_l2: float | None = None
    source_label_match_rate: float | None = None
    learned_step_rms: float | None = None
    free_step_rms: float | None = None
    noise_step_rms: float | None = None
    free_to_learned_ratio: float | None = None
    noise_to_learned_ratio: float | None = None


@dataclass(frozen=True)
class _TorchEdgeClass:
    tails: Tensor
    heads: Tensor
    flux_indices: Tensor


def edge_alpha_value(config: DirectFluxMNISTConfig) -> float:
    """Return the edge Dirichlet parameter used by mobility/free SDE terms."""
    if config.edge_alpha_mode == "grid":
        n = float(config.grid_size)
        return float(config.beta) / (n * n)
    return float(config.alpha)


def natural_horizon(config: DirectFluxMNISTConfig) -> float:
    """Return the fixed-grid bridge horizon used by the Eulerian simulator."""
    n = float(config.grid_size)
    alpha_edge = edge_alpha_value(config)
    return float(config.horizon_scale) / ((2.0 * alpha_edge + 1.0) * n * n)


def effective_train_sde_weights(
    config: DirectFluxMNISTConfig,
    step_index: int | None = None,
) -> tuple[float, float]:
    """Return the free/noise weights used for stochastic-aware training."""
    if bool(config.sde_curriculum):
        if config.sde_ramp_steps <= 0:
            ramp = 1.0
        else:
            step = 0 if step_index is None else max(int(step_index), 0)
            ramp = min(1.0, float(step + 1) / float(config.sde_ramp_steps))
        return ramp * float(config.target_free_weight), ramp * float(config.target_noise_weight)
    free_w = float(config.free_weight if config.train_free_weight is None else config.train_free_weight)
    noise_w = float(config.noise_weight if config.train_noise_weight is None else config.train_noise_weight)
    return free_w, noise_w


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
        if bool(config.condition_on_source):
            in_channels += 2
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

    def _inputs(
        self,
        tau: Tensor | float,
        masses: Tensor,
        labels: Tensor,
        source_masses: Tensor | None = None,
    ) -> Tensor:
        if masses.ndim != 2:
            raise ValueError("masses must have shape (B, N)")
        batch_size = int(masses.shape[0])
        n = int(self.config.grid_size)
        if masses.shape[1] != n * n:
            raise ValueError("masses have the wrong number of pixels")
        if bool(self.config.condition_on_source):
            if source_masses is None:
                # Backwards-compatible fallback for direct calls.  Training and
                # generation pass the persistent initial source explicitly.
                source_masses = masses
            source_masses = source_masses.to(device=masses.device, dtype=masses.dtype).reshape(batch_size, n * n)
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
        pieces = [density, log_density]
        if bool(self.config.condition_on_source):
            assert source_masses is not None
            source_density = source_masses.reshape(batch_size, 1, n, n) * float(n * n)
            source_log_density = torch.log(source_density.clamp_min(float(self.config.mass_floor)))
            pieces.extend([source_density, source_log_density])
        pieces.extend([tau_channel, label_planes])
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        tau: Tensor | float,
        masses: Tensor,
        labels: Tensor,
        source_masses: Tensor | None = None,
    ) -> Tensor:
        """Return normalized flux channels with shape ``(B, 2, H, W)``."""
        x1 = self.enc1(self._inputs(tau, masses, labels, source_masses))
        x2 = self.enc2(F.silu(self.down1(x1)))
        x3 = self.mid(F.silu(self.down2(x2)))
        y2 = self.up2(x3)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        return self.out(y1)

    def predict_flux(
        self,
        tau: Tensor | float,
        masses: Tensor,
        labels: Tensor,
        source_masses: Tensor | None = None,
    ) -> Tensor:
        """Return physical flux rates, not normalized training targets."""
        return float(self.config.flux_scale) * self.forward(tau, masses, labels, source_masses)


# ---------------------------------------------------------------------------
# Terminal proxy, Poisson-flow targets, and direct flux target
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
    alpha_edge = edge_alpha_value(config)
    kappa = (2.0 * alpha_edge + 1.0) / alpha_edge
    return kappa * torch.stack([hx, hy], dim=1)


def free_drift_flux_torch(masses: Tensor, config: DirectFluxMNISTConfig) -> Tensor:
    """Return the raw free Dirichlet drift flux through each oriented edge.

    The sampler multiplies this by ``free_weight`` before conservative
    incidence.  A positive horizontal/vertical value moves mass to the right
    or down, matching the learned flux orientation.
    """
    if masses.ndim != 2:
        raise ValueError("masses must have shape (B, N)")
    n = int(config.grid_size)
    image = masses.reshape(-1, 1, n, n)[:, 0]
    bx = torch.roll(image, shifts=-1, dims=-1)
    by = torch.roll(image, shifts=-1, dims=-2)
    tiny = float(config.mass_floor)
    rx = torch.where(image + bx > tiny, (image - bx) / (image + bx).clamp_min(tiny), torch.zeros_like(image))
    ry = torch.where(image + by > tiny, (image - by) / (image + by).clamp_min(tiny), torch.zeros_like(image))
    alpha_edge = edge_alpha_value(config)
    inv_h2 = float(n * n)
    return (2.0 * alpha_edge + 1.0) * inv_h2 * torch.stack([rx, ry], dim=1)


def edge_noise_std_channels(masses: Tensor, dt: float, config: DirectFluxMNISTConfig) -> Tensor:
    """Return per-edge standard deviations for the free SDE flux increments."""
    if dt < 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be non-negative and finite")
    n = int(config.grid_size)
    theta = harmonic_mobility_channels(masses, config)
    return torch.sqrt((2.0 * theta * float(n * n) * float(dt)).clamp_min(0.0))


def step_component_rms_torch(
    states: Tensor,
    conditioning_flux: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    free_weight: float | None = None,
    noise_weight: float | None = None,
    learned_weight: float | None = None,
) -> dict[str, float]:
    """Return RMS sizes of learned/free/noise edge increments for diagnostics."""
    free_w = float(config.free_weight if free_weight is None else free_weight)
    noise_w = float(config.noise_weight if noise_weight is None else noise_weight)
    learned_w = float(config.learned_weight if learned_weight is None else learned_weight)
    learned_inc = learned_w * conditioning_flux * float(dt)
    free_inc = free_w * free_drift_flux_torch(states, config) * float(dt)
    noise_inc = noise_w * edge_noise_std_channels(states, dt, config)
    learned_rms = float(learned_inc.detach().float().square().mean().sqrt().cpu())
    free_rms = float(free_inc.detach().float().square().mean().sqrt().cpu())
    noise_rms = float(noise_inc.detach().float().square().mean().sqrt().cpu())
    denom = max(learned_rms, 1e-12)
    return {
        "learned_step_rms": learned_rms,
        "free_step_rms": free_rms,
        "noise_step_rms": noise_rms,
        "free_to_learned_ratio": free_rms / denom,
        "noise_to_learned_ratio": noise_rms / denom,
    }


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


def poisson_flux_from_velocity_torch(velocity: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Return minimum-energy periodic edge flux with ``div flux = velocity``.

    ``velocity`` may have shape ``(B, H * W)`` or ``(B, H, W)`` and must have
    approximately zero spatial mean.  The zero mode is removed explicitly.
    """
    if velocity.ndim == 2:
        if grid_size is None:
            side = int(round(math.sqrt(int(velocity.shape[1]))))
            if side * side != int(velocity.shape[1]):
                raise ValueError("grid_size is required when velocity is not square")
            grid_size = side
        v = velocity.reshape(-1, int(grid_size), int(grid_size))
    elif velocity.ndim == 3:
        v = velocity
        if grid_size is not None and v.shape[1:] != (int(grid_size), int(grid_size)):
            raise ValueError("velocity has the wrong grid size")
    else:
        raise ValueError("velocity must have shape (B, N) or (B, H, W)")
    h, w = int(v.shape[-2]), int(v.shape[-1])
    if h != w:
        raise ValueError("only square grids are supported")
    v = v - v.mean(dim=(-2, -1), keepdim=True)
    v_hat = torch.fft.rfft2(v)
    ky = torch.arange(h, device=v.device, dtype=v.dtype).view(h, 1)
    kx = torch.arange(w // 2 + 1, device=v.device, dtype=v.dtype).view(1, w // 2 + 1)
    two_pi = 2.0 * math.pi
    denom = 2.0 * torch.cos(two_pi * kx / float(w)) - 2.0
    denom = denom + 2.0 * torch.cos(two_pi * ky / float(h)) - 2.0
    safe_denom = denom.clamp(max=-1e-12)
    psi_hat = torch.zeros_like(v_hat)
    psi_hat[..., 0, 0] = 0.0
    psi_hat[..., 1:, :] = v_hat[..., 1:, :] / safe_denom[1:, :]
    if w // 2 + 1 > 1:
        psi_hat[..., 0, 1:] = v_hat[..., 0, 1:] / safe_denom[0, 1:]
    psi = torch.fft.irfft2(psi_hat, s=(h, w))
    fx = psi - torch.roll(psi, shifts=-1, dims=-1)
    fy = psi - torch.roll(psi, shifts=-1, dims=-2)
    return torch.stack([fx, fy], dim=1)


# ---------------------------------------------------------------------------
# Training batches, source distributions, loss, and simulator
# ---------------------------------------------------------------------------


def _renormalize_masses(samples: Tensor, *, floor: float) -> Tensor:
    samples = samples.clamp_min(float(floor))
    return samples / samples.sum(dim=1, keepdim=True).clamp_min(float(floor))


def _sample_dirichlet_source(
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
    samples = _renormalize_masses(samples, floor=float(config.mass_floor))
    if config.source_uniform_mix > 0.0:
        uniform = torch.full_like(samples, 1.0 / float(num_pixels))
        samples = (1.0 - float(config.source_uniform_mix)) * samples + float(config.source_uniform_mix) * uniform
    return _renormalize_masses(samples, floor=float(config.mass_floor))


def _sample_source_batch_torch(
    batch_size: int,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    label_tensor: Tensor | None = None,
    source_images: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    class_indices: tuple[NDArray[np.int64], ...] | None = None,
) -> SourceBatch:
    """Sample source measures for training/generation and keep provenance when available."""
    n = int(config.grid_size)
    num_pixels = n * n
    mode = str(config.source_mode)
    if mode in {"class-lowres-prior", "target-lowres-prior"}:
        if label_tensor is None:
            raise ValueError(f"{mode} source mode requires labels")
        return _sample_class_lowres_prior_batch_torch(
            label_tensor,
            source_images,
            source_labels,
            config,
            device=device,
            dtype=dtype,
            rng=rng,
            class_indices=class_indices,
        )
    if mode == "dirichlet":
        masses = _sample_dirichlet_source(batch_size, num_pixels, config, device=device, dtype=dtype)
        return SourceBatch(masses=masses)

    if mode == "blurred-dirichlet":
        flat = _sample_dirichlet_source(batch_size, num_pixels, config, device=device, dtype=dtype)
        image = flat.reshape(batch_size, 1, n, n)
        blurred = _periodic_gaussian_blur_torch(image, sigma=float(config.source_blur_sigma))
        masses = _renormalize_masses(blurred.reshape(batch_size, num_pixels), floor=float(config.mass_floor))
        return SourceBatch(masses=masses)

    k = int(config.source_lowfreq_size)
    coarse_conc = torch.full(
        (batch_size, 1, k, k),
        float(config.source_concentration),
        device=device,
        dtype=dtype,
    )
    coarse = torch.distributions.Gamma(coarse_conc, torch.ones_like(coarse_conc)).sample()
    upsampled = F.interpolate(coarse, size=(n, n), mode="bilinear", align_corners=False)
    if config.source_blur_sigma > 0.0:
        upsampled = _periodic_gaussian_blur_torch(upsampled, sigma=float(config.source_blur_sigma))
    samples = _renormalize_masses(upsampled.reshape(batch_size, num_pixels), floor=float(config.mass_floor))
    uniform = torch.full_like(samples, 1.0 / float(num_pixels))
    if mode == "uniform-plus-lowfreq":
        mix = max(float(config.source_uniform_mix), 0.65)
    else:
        mix = float(config.source_uniform_mix)
    samples = (1.0 - mix) * samples + mix * uniform
    return SourceBatch(masses=_renormalize_masses(samples, floor=float(config.mass_floor)))


def _sample_source_masses_torch(
    batch_size: int,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    label_tensor: Tensor | None = None,
    source_images: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    class_indices: tuple[NDArray[np.int64], ...] | None = None,
) -> Tensor:
    """Compatibility wrapper returning only source masses."""
    return _sample_source_batch_torch(
        batch_size,
        config,
        device=device,
        dtype=dtype,
        label_tensor=label_tensor,
        source_images=source_images,
        source_labels=source_labels,
        rng=rng,
        class_indices=class_indices,
    ).masses

def _compute_class_mean_measures(images: np.ndarray, labels: np.ndarray, grid_size: int) -> FloatArray:
    images_arr = np.asarray(images, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    if images_arr.ndim != 3 or images_arr.shape[1:] != (grid_size, grid_size):
        raise ValueError(f"images must have shape (N, {grid_size}, {grid_size})")
    means = np.zeros((10, grid_size, grid_size), dtype=np.float64)
    global_mean = images_arr.mean(axis=0)
    for digit in range(10):
        cls = images_arr[labels_arr == digit]
        means[digit] = cls.mean(axis=0) if cls.size else global_mean
    flat = means.reshape(10, -1)
    flat = np.maximum(flat, 1e-12)
    flat = flat / flat.sum(axis=1, keepdims=True)
    return flat.reshape(10, grid_size, grid_size).astype(np.float64)


@dataclass(frozen=True)
class ClasswiseOTCache:
    """Precomputed class indices and cheap features for mini-batch OT matching."""

    class_indices: tuple[NDArray[np.int64], ...]
    target_features: FloatArray
    class_means: FloatArray


def _class_indices(labels: np.ndarray) -> tuple[NDArray[np.int64], ...]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    return tuple(np.flatnonzero(labels_arr == digit).astype(np.int64) for digit in range(10))


def _lowres_features_np(images: np.ndarray, config: DirectFluxMNISTConfig) -> FloatArray:
    """Return low-resolution image features plus optional center-of-mass coordinates."""
    arr = np.asarray(images, dtype=np.float64)
    if arr.ndim == 2:
        side = int(config.grid_size)
        arr = arr.reshape(-1, side, side)
    if arr.ndim != 3:
        raise ValueError("images must have shape (B, H, W) or (B, H*W)")
    n = int(config.grid_size)
    if arr.shape[1:] != (n, n):
        raise ValueError(f"images must have shape (B, {n}, {n})")
    batch = int(arr.shape[0])
    if str(config.ot_cost_mode) == "pixel":
        feat_img = arr.reshape(batch, -1)
    else:
        with torch.no_grad():
            t = torch.as_tensor(arr[:, None], dtype=torch.float32, device="cpu")
            if config.ot_blur_sigma > 0.0:
                t = _periodic_gaussian_blur_torch(t, sigma=float(config.ot_blur_sigma))
            t = F.interpolate(t, size=(int(config.ot_lowres_size), int(config.ot_lowres_size)), mode="area")
            feat_img = t.reshape(batch, -1).cpu().numpy().astype(np.float64)
    # Normalize feature scale so the COM term has a predictable effect.
    feat_img = feat_img / np.maximum(np.linalg.norm(feat_img, axis=1, keepdims=True), 1e-12)
    if config.ot_com_weight > 0.0:
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        xx = (xx + 0.5) / float(n)
        yy = (yy + 0.5) / float(n)
        mass = arr.reshape(batch, n, n)
        denom = np.maximum(mass.sum(axis=(1, 2)), 1e-12)
        com_x = (mass * xx).sum(axis=(1, 2)) / denom
        com_y = (mass * yy).sum(axis=(1, 2)) / denom
        com = np.stack([com_x, com_y], axis=1) * math.sqrt(float(config.ot_com_weight))
        feat_img = np.concatenate([feat_img, com], axis=1)
    return np.asarray(feat_img, dtype=np.float64)


def build_classwise_ot_cache(images: np.ndarray, labels: np.ndarray, config: DirectFluxMNISTConfig) -> ClasswiseOTCache:
    """Precompute reusable OT features for the 10c classwise matching target."""
    return ClasswiseOTCache(
        class_indices=_class_indices(labels),
        target_features=_lowres_features_np(images, config),
        class_means=_compute_class_mean_measures(images, labels, int(config.grid_size)),
    )


def _linear_assignment(cost: np.ndarray) -> NDArray[np.int64]:
    """Return column assignment for each row, with a SciPy path and greedy fallback."""
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix")
    n = int(cost.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if n == 1:
        return np.zeros((1,), dtype=np.int64)
    try:  # pragma: no cover - depends on optional scipy install.
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        out = np.empty(n, dtype=np.int64)
        out[np.asarray(rows, dtype=np.int64)] = np.asarray(cols, dtype=np.int64)
        return out
    except Exception:
        remaining_rows = set(range(n))
        remaining_cols = set(range(n))
        out = np.empty(n, dtype=np.int64)
        while remaining_rows:
            best_row = -1
            best_col = -1
            best_value = float("inf")
            for row in remaining_rows:
                cols = np.fromiter(remaining_cols, dtype=np.int64)
                col_idx = int(cols[np.argmin(cost[row, cols])])
                value = float(cost[row, col_idx])
                if value < best_value:
                    best_value = value
                    best_row = row
                    best_col = col_idx
            out[best_row] = best_col
            remaining_rows.remove(best_row)
            remaining_cols.remove(best_col)
        return out


def _ot_coupled_target_indices(
    source_np: np.ndarray,
    batch_labels_np: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    rng: np.random.Generator,
    ot_cache: ClasswiseOTCache | None,
) -> NDArray[np.int64]:
    """Assign each source to a same-label target.

    ``minibatch`` keeps the 10c behavior: draw a same-size candidate set and
    solve a tiny assignment.  ``nearest`` and ``topk`` are the 10e defaults: the
    target is chosen from the full same-label pool using fixed low-resolution
    features, making the approximate map ``(source, label) -> target`` stable
    across batches.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    cache = ot_cache if ot_cache is not None else build_classwise_ot_cache(images, labels_arr, config)
    source_features = _lowres_features_np(source_np, config)
    assigned = np.empty((source_np.shape[0],), dtype=np.int64)
    mode = str(config.ot_match_mode)
    for digit in range(10):
        rows = np.flatnonzero(batch_labels_np == digit)
        if rows.size == 0:
            continue
        available = cache.class_indices[digit]
        if available.size == 0:
            available = np.arange(labels_arr.shape[0], dtype=np.int64)

        if mode == "minibatch":
            replace = bool(available.size < rows.size)
            candidates = rng.choice(available, size=rows.size, replace=replace).astype(np.int64)
            src_feat = source_features[rows]
            tgt_feat = cache.target_features[candidates]
            diff = src_feat[:, None, :] - tgt_feat[None, :, :]
            cost = np.sum(diff * diff, axis=2)
            assignment = _linear_assignment(cost)
            assigned[rows] = candidates[assignment]
            continue

        src_feat = source_features[rows]
        tgt_feat = cache.target_features[available]
        diff = src_feat[:, None, :] - tgt_feat[None, :, :]
        cost = np.sum(diff * diff, axis=2)
        if mode == "nearest" or int(config.ot_nearest_top_k) <= 1:
            assigned[rows] = available[np.argmin(cost, axis=1)]
        elif mode == "topk":
            k = min(int(config.ot_nearest_top_k), int(available.size))
            # argpartition is much cheaper than sorting the entire class pool.
            top_cols = np.argpartition(cost, kth=k - 1, axis=1)[:, :k]
            choices = rng.integers(0, k, size=rows.size)
            assigned[rows] = available[top_cols[np.arange(rows.size), choices]]
        else:  # Defensive guard; config validation should make this unreachable.
            raise ValueError(f"unknown ot_match_mode: {mode}")
    return assigned


def _sample_tau_torch(batch_size: int, config: DirectFluxMNISTConfig, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Sample bridge times, optionally biased toward source and data endpoints."""
    horizon = float(natural_horizon(config))
    if config.tau_sampling == "uniform":
        u = torch.rand((batch_size,), dtype=dtype, device=device)
    else:
        source_prob = float(config.tau_source_prob)
        data_prob = float(config.tau_data_prob)
        selector = torch.rand((batch_size,), dtype=dtype, device=device)
        uniform_u = torch.rand((batch_size,), dtype=dtype, device=device)
        data_u = torch.rand((batch_size,), dtype=dtype, device=device).square()
        source_u = 1.0 - torch.rand((batch_size,), dtype=dtype, device=device).square()
        u = torch.where(selector < source_prob, source_u, uniform_u)
        u = torch.where(selector > 1.0 - data_prob, data_u, u)
    return u.clamp(0.0, 1.0) * horizon


def _coarsen_images_to_source_torch(
    selected: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Convert full-resolution measures to coarse source/latent measures."""
    n = int(config.grid_size)
    arr = np.asarray(selected, dtype=np.float64).reshape(-1, n, n)
    with torch.no_grad():
        source = torch.as_tensor(arr[:, None], dtype=dtype, device=device)
        k = int(config.source_lowfreq_size)
        source = F.interpolate(source, size=(k, k), mode="area")
        source = F.interpolate(source, size=(n, n), mode="bilinear", align_corners=False)
        blur_sigma = max(float(config.source_blur_sigma), 1.0)
        source = _periodic_gaussian_blur_torch(source, sigma=blur_sigma)
        flat = _renormalize_masses(source.reshape(arr.shape[0], n * n), floor=float(config.mass_floor))
        uniform = torch.full_like(flat, 1.0 / float(n * n))
        mix = max(float(config.source_uniform_mix), 0.35)
        flat = (1.0 - mix) * flat + mix * uniform
        return _renormalize_masses(flat, floor=float(config.mass_floor))


def _source_batch_from_images_torch(
    selected: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    indices: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> SourceBatch:
    masses = _coarsen_images_to_source_torch(selected, config, device=device, dtype=dtype)
    idx_arr = None if indices is None else np.asarray(indices, dtype=np.int64).reshape(-1).copy()
    lab_arr = None if labels is None else np.asarray(labels, dtype=np.int64).reshape(-1).copy()
    return SourceBatch(masses=masses, indices=idx_arr, labels=lab_arr)


def _sample_class_lowres_prior_batch_torch(
    label_tensor: Tensor,
    images: np.ndarray | None,
    labels: np.ndarray | None,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rng: np.random.Generator | None,
    class_indices: tuple[NDArray[np.int64], ...] | None = None,
) -> SourceBatch:
    """Sample a coarse class-matched source image and record provenance."""
    if images is None or labels is None:
        raise ValueError("class-lowres-prior source mode requires source images and labels")
    rng = np.random.default_rng() if rng is None else rng
    labels_np = label_tensor.detach().cpu().numpy().astype(np.int64).reshape(-1)
    labels_arr = np.asarray(labels, dtype=np.int64)
    class_idx = _class_indices(labels_arr) if class_indices is None else class_indices
    chosen = np.empty(labels_np.shape[0], dtype=np.int64)
    for digit in range(10):
        rows = np.flatnonzero(labels_np == digit)
        if rows.size == 0:
            continue
        available = class_idx[digit]
        if available.size == 0:
            available = np.arange(labels_arr.shape[0], dtype=np.int64)
        chosen[rows] = rng.choice(available, size=rows.size, replace=True)
    selected = np.asarray(images, dtype=np.float64)[chosen]
    return _source_batch_from_images_torch(
        selected,
        config,
        device=device,
        dtype=dtype,
        indices=chosen,
        labels=labels_arr[chosen],
    )


def _sample_class_lowres_prior_torch(
    label_tensor: Tensor,
    images: np.ndarray | None,
    labels: np.ndarray | None,
    config: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rng: np.random.Generator | None,
    class_indices: tuple[NDArray[np.int64], ...] | None = None,
) -> Tensor:
    """Compatibility wrapper returning only source masses."""
    return _sample_class_lowres_prior_batch_torch(
        label_tensor,
        images,
        labels,
        config,
        device=device,
        dtype=dtype,
        rng=rng,
        class_indices=class_indices,
    ).masses


def sample_flux_training_batch(
    images: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    batch_size: int,
    device: str | torch.device,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    class_means: np.ndarray | None = None,
    ot_cache: ClasswiseOTCache | None = None,
    step_index: int | None = None,
    mean_flow_prob: float | None = None,
) -> FluxTrainingBatch:
    """Sample states on source-to-target bridges for direct flux regression.

    Experiment 10d keeps the 10c OT-coupled target but optionally gives the
    network persistent access to the initial source/latent.  In
    ``source_mode='target-lowres-prior'`` the source is a coarse version of the
    same target image, which is a diagnostic upper bound for the flux/sampler.
    """
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

    cache = ot_cache
    if cache is None and (
        config.target_mode == "poisson-ot-flow" or config.source_mode in {"class-lowres-prior", "target-lowres-prior"}
    ):
        cache = build_classwise_ot_cache(images_arr, labels_arr, config)
    means = _compute_class_mean_measures(images_arr, labels_arr, n) if class_means is None else class_means

    base_idx = rng.integers(0, images_arr.shape[0], size=int(batch_size))
    batch_labels_np = labels_arr[base_idx]
    resolved_device = torch.device(device)
    batch_labels = torch.as_tensor(batch_labels_np, dtype=torch.long, device=resolved_device)

    source_batch: SourceBatch | None = None
    if config.source_mode != "target-lowres-prior":
        source_batch = _sample_source_batch_torch(
            int(batch_size),
            config,
            device=resolved_device,
            dtype=dtype,
            label_tensor=batch_labels,
            source_images=images_arr,
            source_labels=labels_arr,
            rng=rng,
            class_indices=None if cache is None else cache.class_indices,
        )

    target_np = np.empty((int(batch_size), n, n), dtype=np.float64)
    target_indices = np.full((int(batch_size),), -1, dtype=np.int64)
    if config.target_mode == "class-mean-flow":
        target_np[:] = np.asarray(means, dtype=np.float64)[batch_labels_np]
    else:
        anchor_prob = 0.0
        if config.target_mode in {"poisson-flow", "poisson-ot-flow"}:
            if mean_flow_prob is not None:
                anchor_prob = float(mean_flow_prob)
            elif step_index is not None and int(step_index) < int(config.mean_flow_warmup_steps):
                anchor_prob = float(config.mean_flow_warmup_prob)
            else:
                anchor_prob = float(config.mean_flow_prob)
        mean_mask = rng.random(int(batch_size)) < anchor_prob
        if mean_mask.any():
            target_np[mean_mask] = np.asarray(means, dtype=np.float64)[batch_labels_np[mean_mask]]
        non_mean = np.flatnonzero(~mean_mask)
        if non_mean.size:
            if config.source_mode == "target-lowres-prior":
                # Coupled diagnostic: the latent source is a coarse version of the
                # same full target.  This tests whether the flux model/sampler can
                # refine a coarse latent into a clean digit.
                target_np[non_mean] = images_arr[base_idx[non_mean]]
                target_indices[non_mean] = base_idx[non_mean]
            elif config.target_mode == "poisson-ot-flow":
                assert source_batch is not None
                source_np = source_batch.masses.detach().cpu().numpy().astype(np.float64).reshape(
                    int(batch_size), n, n
                )
                assigned_idx = _ot_coupled_target_indices(
                    source_np[non_mean],
                    batch_labels_np[non_mean],
                    images_arr,
                    labels_arr,
                    config,
                    rng=rng,
                    ot_cache=cache,
                )
                target_np[non_mean] = images_arr[assigned_idx]
                target_indices[non_mean] = assigned_idx
            else:
                # ``poisson-flow`` and ``terminal-score`` keep the older independent
                # same-label target sampling behavior, except for target-lowres-prior above.
                target_np[non_mean] = images_arr[base_idx[non_mean]]
                target_indices[non_mean] = base_idx[non_mean]

    if config.source_mode == "target-lowres-prior":
        source_batch = _source_batch_from_images_torch(
            target_np,
            config,
            device=resolved_device,
            dtype=dtype,
            indices=target_indices,
            labels=batch_labels_np,
        )
    assert source_batch is not None
    source = source_batch.masses

    target = torch.as_tensor(target_np.reshape(int(batch_size), n * n), dtype=dtype, device=resolved_device)
    target = _renormalize_masses(target, floor=float(config.mass_floor))
    tau = _sample_tau_torch(int(batch_size), config, device=resolved_device, dtype=dtype)
    mix = (tau / max(natural_horizon(config), 1e-12)).pow(float(config.bridge_power)).view(-1, 1)
    states = (1.0 - mix) * target + mix * source
    if config.state_jitter_weight > 0.0:
        jitter = _sample_source_masses_torch(
            int(batch_size),
            config,
            device=resolved_device,
            dtype=dtype,
            label_tensor=batch_labels,
            source_images=images_arr,
            source_labels=labels_arr,
            rng=rng,
            class_indices=None if cache is None else cache.class_indices,
        )
        states = (1.0 - float(config.state_jitter_weight)) * states + float(config.state_jitter_weight) * jitter
    states = _renormalize_masses(states, floor=float(config.mass_floor))
    train_free_weight, train_noise_weight = effective_train_sde_weights(config, step_index)
    return FluxTrainingBatch(
        tau=tau,
        states=states,
        labels=batch_labels,
        targets=target,
        sources=source,
        source_indices=source_batch.indices,
        source_labels=source_batch.labels,
        target_indices=target_indices,
        train_free_weight=float(train_free_weight),
        train_noise_weight=float(train_noise_weight),
    )

def training_target_flux_torch(batch: FluxTrainingBatch, config: DirectFluxMNISTConfig) -> Tensor:
    """Return the physical two-channel target flux for the configured target mode."""
    if config.target_mode == "terminal-score":
        return terminal_conditioning_flux_torch(batch.states, batch.targets, config)
    horizon = max(natural_horizon(config), 1e-12)
    velocity_mode = batch.target_velocity_mode or config.velocity_target
    if velocity_mode == "residual":
        remaining = batch.tau.clamp_min(float(config.min_tau_fraction) * float(horizon)).view(-1, 1)
        velocity = (batch.targets - batch.states) / remaining
    elif velocity_mode == "constant":
        velocity = (batch.targets - batch.sources) / float(horizon)
    else:
        raise ValueError(f"unknown velocity target mode: {velocity_mode}")
    velocity = velocity - velocity.mean(dim=1, keepdim=True)
    total_flux = poisson_flux_from_velocity_torch(velocity, grid_size=int(config.grid_size))
    if bool(config.free_aware_target):
        total_flux = total_flux - float(batch.train_free_weight) * free_drift_flux_torch(batch.states, config)
    return total_flux


def direct_flux_matching_loss(
    model: DirectFluxUNet,
    batch: FluxTrainingBatch,
) -> tuple[Tensor, dict[str, float]]:
    """Return the direct-flux regression loss and scalar diagnostics."""
    config = model.config
    pred_norm = model(batch.tau, batch.states, batch.labels, batch.sources)
    with torch.no_grad():
        target_flux = training_target_flux_torch(batch, config)
        target_norm = (target_flux / float(config.flux_scale)).clamp(
            -float(config.target_flux_clip), float(config.target_flux_clip)
        )
    flux_loss = F.smooth_l1_loss(pred_norm, target_norm)
    pred_div = flux_divergence_torch(pred_norm)
    target_div = flux_divergence_torch(target_norm)
    div_loss = F.smooth_l1_loss(pred_div, target_div)
    node_loss = F.mse_loss(pred_div, target_div)

    step_loss = pred_norm.new_tensor(0.0)
    if float(config.step_loss_weight) > 0.0:
        dt = natural_horizon(config) / float(max(int(config.num_steps), 1))
        step_free_weight = float(batch.train_free_weight) if bool(config.stochastic_step_loss) else 0.0
        step_noise_weight = float(batch.train_noise_weight) if bool(config.stochastic_step_loss) else 0.0
        noise_delta = (
            _noise_delta_flat_torch(batch.states, dt, config, step_noise_weight)
            if bool(config.same_noise_step_loss)
            else None
        )
        pred_next = eulerian_flux_step_differentiable_torch(
            batch.states,
            pred_norm * float(config.flux_scale),
            dt,
            config,
            free_weight=step_free_weight,
            learned_weight=1.0,
            noise_delta_flat=noise_delta,
        )
        with torch.no_grad():
            target_next = eulerian_flux_step_differentiable_torch(
                batch.states,
                target_norm * float(config.flux_scale),
                dt,
                config,
                free_weight=step_free_weight,
                learned_weight=1.0,
                noise_delta_flat=noise_delta,
            )
        density_scale = float(config.grid_size * config.grid_size)
        step_loss = F.mse_loss(pred_next * density_scale, target_next * density_scale)

    loss = (
        flux_loss
        + float(config.divergence_loss_weight) * div_loss
        + float(config.node_loss_weight) * node_loss
        + float(config.step_loss_weight) * step_loss
    )
    with torch.no_grad():
        pred_rms = pred_norm.square().mean().sqrt()
        target_rms = target_norm.square().mean().sqrt()
        flat_pred = pred_div.flatten(1)
        flat_target = target_div.flatten(1)
        numerator = (flat_pred * flat_target).sum(dim=1)
        denominator = flat_pred.square().sum(dim=1).sqrt() * flat_target.square().sum(dim=1).sqrt()
        div_cos = (numerator / denominator.clamp_min(1e-12)).mean()
        comp = step_component_rms_torch(
            batch.states,
            pred_norm * float(config.flux_scale),
            natural_horizon(config) / float(max(int(config.num_steps), 1)),
            config,
            free_weight=float(batch.train_free_weight),
            noise_weight=float(batch.train_noise_weight),
            learned_weight=1.0,
        )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "flux_loss": float(flux_loss.detach().cpu()),
        "div_loss": float(div_loss.detach().cpu()),
        "node_loss": float(node_loss.detach().cpu()),
        "step_loss": float(step_loss.detach().cpu()),
        "div_cos": float(div_cos.detach().cpu()),
        "pred_rms": float(pred_rms.detach().cpu()),
        "target_rms": float(target_rms.detach().cpu()),
        "train_free_weight": float(batch.train_free_weight),
        "train_noise_weight": float(batch.train_noise_weight),
        "learned_step_rms": float(comp["learned_step_rms"]),
        "free_step_rms": float(comp["free_step_rms"]),
        "noise_step_rms": float(comp["noise_step_rms"]),
        "free_to_learned_ratio": float(comp["free_to_learned_ratio"]),
        "noise_to_learned_ratio": float(comp["noise_to_learned_ratio"]),
    }


@torch.no_grad()
def make_on_policy_training_batch(
    model: DirectFluxUNet,
    images: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    batch_size: int,
    device: str | torch.device,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    class_means: np.ndarray | None = None,
    ot_cache: ClasswiseOTCache | None = None,
    step_index: int | None = None,
) -> FluxTrainingBatch:
    """Sample a batch from states visited by the current sampler.

    The batch keeps the same assigned source/target pair as the ordinary teacher
    batch, but replaces the interpolated state by a short no-gradient prefix
    rollout of the current model.  Its target velocity is forced to residual so
    the model learns to correct its own off-path states.
    """
    rng = np.random.default_rng() if rng is None else rng
    resolved_device = torch.device(device)
    base = sample_flux_training_batch(
        images,
        labels,
        config,
        batch_size=int(batch_size),
        device=resolved_device,
        rng=rng,
        dtype=dtype,
        class_means=class_means,
        ot_cache=ot_cache,
        step_index=step_index,
    )
    max_prefix = min(int(config.on_policy_prefix_steps), int(config.num_steps) - 1)
    if max_prefix <= 0:
        return FluxTrainingBatch(
            tau=base.tau,
            states=base.states,
            labels=base.labels,
            targets=base.targets,
            sources=base.sources,
            source_indices=base.source_indices,
            source_labels=base.source_labels,
            target_indices=base.target_indices,
            target_velocity_mode="residual",
            train_free_weight=base.train_free_weight,
            train_noise_weight=base.train_noise_weight,
        )
    prefix_steps = int(rng.integers(1, max_prefix + 1))
    horizon = float(natural_horizon(config))
    dt = horizon / float(max(int(config.num_steps), 1))
    states = base.sources.clone()
    source_condition = base.sources.clone()
    model_was_training = bool(model.training)
    model.eval()
    for prefix_idx in range(prefix_steps):
        tau_value = max(horizon - float(prefix_idx) * dt, 0.0)
        tau = torch.full((int(batch_size),), tau_value, dtype=states.dtype, device=resolved_device)
        flux = model.predict_flux(tau, states, base.labels, source_condition)
        rollout_free = float(base.train_free_weight) if bool(config.on_policy_use_free) else 0.0
        rollout_noise = float(base.train_noise_weight) if bool(config.on_policy_use_noise) else 0.0
        states, _, _ = eulerian_flux_step_torch(
            states,
            flux.float(),
            dt,
            config,
            deterministic=not bool(config.on_policy_use_noise),
            free_weight=rollout_free,
            noise_weight=rollout_noise,
            learned_weight=1.0,
        )
    if model_was_training:
        model.train()
    tau_remaining = torch.full(
        (int(batch_size),),
        max(horizon - float(prefix_steps) * dt, 0.0),
        dtype=states.dtype,
        device=resolved_device,
    )
    return FluxTrainingBatch(
        tau=tau_remaining,
        states=states.detach(),
        labels=base.labels,
        targets=base.targets,
        sources=base.sources,
        source_indices=base.source_indices,
        source_labels=base.source_labels,
        target_indices=base.target_indices,
        target_velocity_mode="residual",
        train_free_weight=base.train_free_weight,
        train_noise_weight=base.train_noise_weight,
    )


def _mass_entropy_torch(states: Tensor) -> Tensor:
    states = states.clamp_min(1e-30)
    return -(states * states.log()).sum(dim=1)


def _label_sequence_tensor(count: int, *, device: torch.device) -> Tensor:
    return torch.arange(count, device=device, dtype=torch.long) % 10


def _disable_mkldnn_for_cpu_if_needed(device: torch.device) -> None:
    """Avoid rare CPU convolution backward hangs seen with MKL-DNN on sparse MNIST masses."""
    if device.type == "cpu" and hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


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
    preview_dir: str | Path | None = None,
    preview_every: int = 0,
    preview_sample_steps: int = 64,
    preview_num_samples: int = 16,
) -> dict[str, list[float]]:
    """Train the direct-flux U-Net with an ETA progress bar and optional previews."""
    if train_steps <= 0 or batch_size <= 0:
        raise ValueError("train_steps and batch_size must be positive")
    resolved_device = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else "cpu" if device is None else device
    )
    _disable_mkldnn_for_cpu_if_needed(resolved_device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.to(resolved_device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    scaler = _make_cuda_grad_scaler(enabled=amp_enabled)
    ot_cache = build_classwise_ot_cache(images, labels, model.config)
    class_means = ot_cache.class_means
    history: dict[str, list[float]] = {
        "loss": [],
        "flux_loss": [],
        "div_loss": [],
        "node_loss": [],
        "step_loss": [],
        "on_policy": [],
        "div_cos": [],
        "pred_rms": [],
        "target_rms": [],
        "train_free_weight": [],
        "train_noise_weight": [],
        "learned_step_rms": [],
        "free_step_rms": [],
        "noise_step_rms": [],
        "free_to_learned_ratio": [],
        "noise_to_learned_ratio": [],
    }

    preview_path = None if preview_dir is None or preview_every <= 0 else Path(preview_dir)
    if preview_path is not None:
        preview_path.mkdir(parents=True, exist_ok=True)

    bar = _progress(range(int(train_steps)), total=int(train_steps), desc="train flux", disable=not show_progress)
    for step_index in bar:
        use_on_policy = (
            int(step_index) >= int(model.config.on_policy_warmup_steps)
            and float(model.config.on_policy_prob) > 0.0
            and rng.random() < float(model.config.on_policy_prob)
        )
        if use_on_policy:
            batch = make_on_policy_training_batch(
                model,
                images,
                labels,
                model.config,
                batch_size=int(batch_size),
                device=resolved_device,
                rng=rng,
                class_means=class_means,
                ot_cache=ot_cache,
                step_index=int(step_index),
            )
        else:
            batch = sample_flux_training_batch(
                images,
                labels,
                model.config,
                batch_size=int(batch_size),
                device=resolved_device,
                rng=rng,
                class_means=class_means,
                ot_cache=ot_cache,
                step_index=int(step_index),
            )
        optimizer.zero_grad(set_to_none=True)
        context = _cuda_autocast(enabled=True) if amp_enabled else nullcontext()
        with context:
            loss, metrics = direct_flux_matching_loss(model, batch)
        metrics["on_policy"] = 1.0 if use_on_policy else 0.0
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        for key in history:
            history[key].append(metrics[key])
        if hasattr(bar, "set_postfix"):
            anchor_prob = (
                model.config.mean_flow_warmup_prob
                if int(step_index) < int(model.config.mean_flow_warmup_steps)
                else model.config.mean_flow_prob
            )
            bar.set_postfix(
                loss=metrics["loss"],
                div_cos=metrics["div_cos"],
                step_l=metrics.get("step_loss", 0.0),
                pred=metrics["pred_rms"],
                tgt=metrics["target_rms"],
                onp=metrics.get("on_policy", 0.0),
                free_r=metrics.get("free_to_learned_ratio", 0.0),
                noise_r=metrics.get("noise_to_learned_ratio", 0.0),
                mean_p=float(anchor_prob) if model.config.target_mode in {"poisson-flow", "poisson-ot-flow"} else 0.0,
            )

        step_num = int(step_index) + 1
        if preview_path is not None and (step_num % int(preview_every) == 0 or step_num == int(train_steps)):
            try:
                preview_batch = sample_flux_training_batch(
                    images,
                    labels,
                    model.config,
                    batch_size=int(preview_num_samples),
                    device=resolved_device,
                    rng=rng,
                    class_means=class_means,
                    ot_cache=ot_cache,
                    step_index=int(step_index),
                )
                preview = simulate_direct_flux_generation(
                    model,
                    preview_batch.labels,
                    num_steps=int(preview_sample_steps),
                    deterministic=True,
                    device=resolved_device,
                    seed=int(seed) + 1000 + step_num,
                    use_amp=use_amp,
                    show_progress=False,
                    initial_states=preview_batch.sources,
                    source_images=images,
                    source_labels=labels,
                )
                teacher = simulate_teacher_flux_rollout(
                    preview_batch.sources,
                    preview_batch.targets,
                    model.config,
                    num_steps=int(preview_sample_steps),
                    device=resolved_device,
                )
                cls_mean_refs = class_means[preview_batch.labels.detach().cpu().numpy().astype(np.int64)].reshape(
                    int(preview_num_samples), -1
                )
                save_flux_preview_panel(
                    preview.sources if preview.sources is not None else preview.samples,
                    preview.samples,
                    preview_batch.targets.detach().cpu().numpy().astype(np.float64),
                    preview.labels,
                    preview_path / f"preview_step_{step_num:06d}.png",
                    grid_size=int(model.config.grid_size),
                    teacher=teacher.detach().cpu().numpy().astype(np.float64),
                    class_means=cls_mean_refs,
                )
            except RuntimeError:
                pass
            finally:
                model.train()
    return history


_EDGE_CLASS_CACHE: dict[tuple[int, str], list[_TorchEdgeClass]] = {}


def _edge_classes_torch(grid_size: int, device: torch.device) -> list[_TorchEdgeClass]:
    n = int(grid_size)
    key = (n, str(device))
    cached = _EDGE_CLASS_CACHE.get(key)
    if cached is not None:
        return cached
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
    _EDGE_CLASS_CACHE[key] = result
    return result


def eulerian_flux_step_differentiable_torch(
    states: Tensor,
    conditioning_flux: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    free_weight: float = 0.0,
    learned_weight: float = 1.0,
    noise_delta_flat: Tensor | None = None,
) -> Tensor:
    """Differentiable four-color limited step used by step/rollout losses.

    It mirrors the production sampler's conservative edge incidence and edge
    clipping.  Unlike the old training helper, it can include the same free
    drift term as the stochastic sampler.  ``noise_delta_flat`` may contain a
    pre-sampled edge increment with shape ``(B, 2 * H * W)``; using the same
    tensor for predicted and teacher steps makes the noise cancel in the loss
    while preserving limiter effects.
    """
    if states.ndim != 2:
        raise ValueError("states must have shape (B, N)")
    if conditioning_flux.ndim != 4 or conditioning_flux.shape[1] != 2:
        raise ValueError("conditioning_flux must have shape (B, 2, H, W)")
    n = int(config.grid_size)
    if states.shape[1] != n * n or conditioning_flux.shape[2:] != (n, n):
        raise ValueError("states/flux have incompatible grid sizes")
    if dt < 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be non-negative and finite")
    if dt == 0.0:
        return states.clone()
    out = states
    flat_flux = torch.cat(
        [conditioning_flux[:, 0].reshape(states.shape[0], -1), conditioning_flux[:, 1].reshape(states.shape[0], -1)],
        dim=1,
    )
    if noise_delta_flat is not None:
        noise_delta_flat = noise_delta_flat.to(device=states.device, dtype=states.dtype)
        if noise_delta_flat.shape != flat_flux.shape:
            raise ValueError("noise_delta_flat must have shape (B, 2 * H * W)")
    tiny = float(config.mass_floor)
    for edge_class in _edge_classes_torch(n, states.device):
        tails = edge_class.tails
        heads = edge_class.heads
        a = out[:, tails]
        b = out[:, heads]
        learned_flux = flat_flux[:, edge_class.flux_indices]
        d_flux = float(learned_weight) * learned_flux * float(dt)
        if float(free_weight) != 0.0:
            free_flat = torch.cat(
                [
                    free_drift_flux_torch(out, config)[:, 0].reshape(out.shape[0], -1),
                    free_drift_flux_torch(out, config)[:, 1].reshape(out.shape[0], -1),
                ],
                dim=1,
            )
            d_flux = d_flux + float(free_weight) * free_flat[:, edge_class.flux_indices] * float(dt)
        if noise_delta_flat is not None:
            d_flux = d_flux + noise_delta_flat[:, edge_class.flux_indices]
        d_flux = torch.minimum(d_flux, float(config.limiter_fraction) * a)
        d_flux = torch.maximum(d_flux, -float(config.limiter_fraction) * b)
        delta = torch.zeros_like(out)
        batch_tails = tails.view(1, -1).expand(out.shape[0], -1)
        batch_heads = heads.view(1, -1).expand(out.shape[0], -1)
        delta.scatter_add_(1, batch_tails, -d_flux)
        delta.scatter_add_(1, batch_heads, d_flux)
        out = (out + delta).clamp_min(tiny)
        out = out / out.sum(dim=1, keepdim=True).clamp_min(tiny)
    return out


def _noise_delta_flat_torch(states: Tensor, dt: float, config: DirectFluxMNISTConfig, noise_weight: float) -> Tensor | None:
    """Sample a flat edge-noise increment for differentiable paired step losses."""
    if float(noise_weight) <= 0.0:
        return None
    std = float(noise_weight) * edge_noise_std_channels(states, dt, config)
    return torch.cat(
        [std[:, 0].reshape(states.shape[0], -1), std[:, 1].reshape(states.shape[0], -1)], dim=1
    ) * torch.randn(states.shape[0], 2 * int(config.grid_size) * int(config.grid_size), device=states.device, dtype=states.dtype)

def eulerian_flux_step_torch(
    states: Tensor,
    conditioning_flux: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    deterministic: bool = False,
    free_weight: float | None = None,
    noise_weight: float | None = None,
    learned_weight: float | None = None,
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
    free_w = float(config.free_weight if free_weight is None else free_weight)
    noise_w = float(config.noise_weight if noise_weight is None else noise_weight)
    learned_w = float(config.learned_weight if learned_weight is None else learned_weight)
    out = states.clone()
    inv_h2 = float(n * n)
    alpha = edge_alpha_value(config)
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
        d_flux = (free_w * free_flux + learned_w * learned_flux) * float(dt)
        if (not deterministic) and noise_w > 0.0:
            noise_std = noise_w * torch.sqrt((2.0 * theta * inv_h2 * float(dt)).clamp_min(0.0))
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
    initial_states: Tensor | np.ndarray | None = None,
    source_images: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
    free_weight: float | None = None,
    noise_weight: float | None = None,
    learned_weight: float | None = None,
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
    _disable_mkldnn_for_cpu_if_needed(resolved_device)
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
    sample_free_weight = float(cfg.free_weight if free_weight is None else free_weight)
    sample_noise_weight = float(cfg.noise_weight if noise_weight is None else noise_weight)
    sample_learned_weight = float(cfg.learned_weight if learned_weight is None else learned_weight)
    rng = np.random.default_rng(int(seed))
    source_indices: IntArray | None = None
    sampled_source_labels: IntArray | None = None
    if initial_states is None:
        source_batch = _sample_source_batch_torch(
            batch_size,
            cfg,
            device=resolved_device,
            dtype=torch.float32,
            label_tensor=labels_t,
            source_images=source_images,
            source_labels=source_labels,
            rng=rng,
        )
        states = source_batch.masses
        source_indices = source_batch.indices
        sampled_source_labels = source_batch.labels
    else:
        states = torch.as_tensor(initial_states, dtype=torch.float32, device=resolved_device).reshape(batch_size, n * n)
        states = _renormalize_masses(states, floor=float(cfg.mass_floor))
    source_condition = states.clone()
    initial_states = states.detach().cpu().numpy().astype(np.float64)
    trajectory: list[np.ndarray] = []
    if save_every > 0:
        trajectory.append(states.detach().cpu().numpy().astype(np.float64))
    clipped = 0
    proposed = 0
    component_sums = {
        "learned_step_rms": 0.0,
        "free_step_rms": 0.0,
        "noise_step_rms": 0.0,
        "free_to_learned_ratio": 0.0,
        "noise_to_learned_ratio": 0.0,
    }
    component_count = 0
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    bar = _progress(range(steps), total=steps, desc="sample flux", disable=not show_progress)
    for step in bar:
        tau_value = max(horizon - float(step) * dt, 0.0)

        def advance_with_substeps(start_states: Tensor, substeps: int) -> tuple[Tensor, int, int]:
            nonlocal component_count
            local_states = start_states
            local_clipped = 0
            local_proposed = 0
            sub_dt = dt / float(substeps)
            for sub_idx in range(int(substeps)):
                sub_tau_value = max(tau_value - float(sub_idx) * sub_dt, 0.0)
                tau = torch.full((batch_size,), sub_tau_value, dtype=local_states.dtype, device=resolved_device)
                context = _cuda_autocast(enabled=True) if amp_enabled else nullcontext()
                with context:
                    flux = model.predict_flux(tau, local_states, labels_t, source_condition)
                nonlocal_component = step_component_rms_torch(
                    local_states,
                    flux.float(),
                    sub_dt,
                    cfg,
                    free_weight=sample_free_weight,
                    noise_weight=sample_noise_weight,
                    learned_weight=sample_learned_weight,
                )
                for _key, _value in nonlocal_component.items():
                    component_sums[_key] += float(_value)
                component_count += 1
                local_states, c_step, p_step = eulerian_flux_step_torch(
                    local_states,
                    flux.float(),
                    sub_dt,
                    cfg,
                    deterministic=deterministic,
                    free_weight=sample_free_weight,
                    noise_weight=sample_noise_weight,
                    learned_weight=sample_learned_weight,
                )
                local_clipped += c_step
                local_proposed += p_step
            return local_states, local_clipped, local_proposed

        if bool(cfg.adaptive_sampling):
            substeps = 1
            while True:
                candidate, c_step, p_step = advance_with_substeps(states, substeps)
                local_clip = 0.0 if p_step == 0 else float(c_step) / float(p_step)
                if local_clip <= float(cfg.clip_target) or substeps >= int(cfg.max_substeps):
                    states = candidate
                    break
                substeps = min(int(cfg.max_substeps), substeps * 2)
        else:
            states, c_step, p_step = advance_with_substeps(states, 1)
        clipped += c_step
        proposed += p_step
        if hasattr(bar, "set_postfix"):
            ent = float(_mass_entropy_torch(states).mean().detach().cpu())
            max_mass = float(states.max(dim=1).values.mean().detach().cpu())
            bar.set_postfix(ent=ent, max=max_mass, clip=0.0 if proposed == 0 else clipped / proposed)
        if save_every > 0 and ((step + 1) % int(save_every) == 0 or step + 1 == steps):
            trajectory.append(states.detach().cpu().numpy().astype(np.float64))
    diagnostics = source_batch_diagnostics(
        initial_states,
        requested_labels=labels_t.detach().cpu().numpy().astype(np.int64),
        source_indices=source_indices,
        source_labels=sampled_source_labels,
    )
    if cfg.source_mode in {"class-lowres-prior", "target-lowres-prior"} and batch_size > 1:
        if int(diagnostics["source_unique_count"]) <= 1:
            raise RuntimeError(
                f"{cfg.source_mode} collapsed to a single source for a batch of {batch_size}; "
                "check source sampling/provenance."
            )
    component_means = {
        key: (float(value) / float(component_count) if component_count > 0 else 0.0)
        for key, value in component_sums.items()
    }
    if component_means["free_to_learned_ratio"] > 0.5 or component_means["noise_to_learned_ratio"] > 0.5:
        print(
            "Warning: stochastic increments are comparable to the learned step: "
            f"free/learned={component_means['free_to_learned_ratio']:.3f}, "
            f"noise/learned={component_means['noise_to_learned_ratio']:.3f}"
        )
    return FluxGenerationResult(
        samples=states.detach().cpu().numpy().astype(np.float64),
        labels=labels_t.detach().cpu().numpy().astype(np.int64),
        trajectory=None if save_every <= 0 else np.stack(trajectory, axis=0),
        clipping_fraction=0.0 if proposed == 0 else float(clipped) / float(proposed),
        sources=initial_states,
        source_indices=source_indices,
        source_labels=sampled_source_labels,
        source_unique_count=int(diagnostics["source_unique_count"]),
        source_diversity_l2=float(diagnostics["source_diversity_l2"]),
        source_pair_l2=float(diagnostics["source_pair_l2"]),
        source_label_match_rate=float(diagnostics["source_label_match_rate"]),
        learned_step_rms=component_means["learned_step_rms"],
        free_step_rms=component_means["free_step_rms"],
        noise_step_rms=component_means["noise_step_rms"],
        free_to_learned_ratio=component_means["free_to_learned_ratio"],
        noise_to_learned_ratio=component_means["noise_to_learned_ratio"],
    )


@torch.no_grad()
def simulate_teacher_flux_rollout(
    sources: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    num_steps: int | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Roll out the exact supervised teacher flux from source to assigned target.

    This is a diagnostic upper bound: if the teacher rollout cannot reach the
    assigned target, the problem is the flux scaling/limiter/timestep rather
    than the learned U-Net.
    """
    resolved_device = torch.device(
        "cuda" if device is None and torch.cuda.is_available() else "cpu" if device is None else device
    )
    n = int(config.grid_size)
    states = torch.as_tensor(sources, dtype=torch.float32, device=resolved_device).reshape(-1, n * n)
    target = torch.as_tensor(targets, dtype=torch.float32, device=resolved_device).reshape_as(states)
    states = _renormalize_masses(states, floor=float(config.mass_floor))
    source0 = states.clone()
    target = _renormalize_masses(target, floor=float(config.mass_floor))
    steps = int(config.num_steps if num_steps is None else num_steps)
    horizon = max(float(natural_horizon(config)), 1e-12)
    dt = horizon / float(steps)
    if config.target_mode == "terminal-score":
        for _ in range(steps):
            flux = terminal_conditioning_flux_torch(states, target, config)
            states, _, _ = eulerian_flux_step_torch(
                states,
                flux,
                dt,
                config,
                deterministic=True,
                free_weight=0.0,
                noise_weight=0.0,
                learned_weight=1.0,
            )
    else:
        if config.velocity_target == "constant":
            velocity = (target - source0) / horizon
            flux = poisson_flux_from_velocity_torch(velocity, grid_size=n)
            for _ in range(steps):
                states, _, _ = eulerian_flux_step_torch(
                    states,
                    flux,
                    dt,
                    config,
                    deterministic=True,
                    free_weight=0.0,
                    noise_weight=0.0,
                    learned_weight=1.0,
                )
        else:
            for step in range(steps):
                remaining = max(horizon - float(step) * dt, float(config.min_tau_fraction) * horizon)
                velocity = (target - states) / remaining
                velocity = velocity - velocity.mean(dim=1, keepdim=True)
                flux = poisson_flux_from_velocity_torch(velocity, grid_size=n)
                states, _, _ = eulerian_flux_step_torch(
                    states,
                    flux,
                    dt,
                    config,
                    deterministic=True,
                    free_weight=0.0,
                    noise_weight=0.0,
                    learned_weight=1.0,
                )
    return states


# ---------------------------------------------------------------------------
# Output helpers and CLI
# ---------------------------------------------------------------------------


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return image / max(float(image.max()), 1e-12)


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
        import matplotlib
        matplotlib.use("Agg", force=True)
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
        image = _normalize_for_display(arr[idx])
        ax = axes[idx // cols, idx % cols]
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(str(int(labels_arr[idx])), fontsize=8)
        ax.axis("off")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_flux_preview_panel(
    sources: np.ndarray,
    generated: np.ndarray,
    references: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    output_path: str | Path,
    *,
    grid_size: int = 28,
    max_images: int = 16,
    teacher: np.ndarray | None = None,
    class_means: np.ndarray | None = None,
) -> None:
    """Save source/generated/reference rows for early training diagnostics."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        raise RuntimeError("matplotlib is required to save a preview panel") from exc

    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    src = np.asarray(sources, dtype=np.float64).reshape(-1, grid_size, grid_size)
    gen = np.asarray(generated, dtype=np.float64).reshape(-1, grid_size, grid_size)
    ref = np.asarray(references, dtype=np.float64).reshape(-1, grid_size, grid_size)
    rows: list[tuple[str, np.ndarray]] = [("source", src), ("generated", gen), ("assigned target", ref)]
    if teacher is not None:
        rows.append(("teacher rollout", np.asarray(teacher, dtype=np.float64).reshape(-1, grid_size, grid_size)))
    if class_means is not None:
        rows.append(("class mean", np.asarray(class_means, dtype=np.float64).reshape(-1, grid_size, grid_size)))
    count = min(int(max_images), gen.shape[0])
    fig, axes = plt.subplots(len(rows), count, figsize=(1.15 * count, 1.25 * len(rows)), squeeze=False)
    for row_idx, (row_name, row) in enumerate(rows):
        for col_idx in range(count):
            ax = axes[row_idx, col_idx]
            ax.imshow(_normalize_for_display(row[col_idx]), cmap="gray", interpolation="nearest")
            if row_idx == 0:
                ax.set_title(str(int(labels_arr[col_idx])), fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
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


def _samples_stats(samples: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=np.float64)
    ent = -np.sum(arr * np.log(np.maximum(arr, 1e-30)), axis=1).mean()
    max_mass = arr.max(axis=1).mean()
    return float(ent), float(max_mass)


def source_diversity_metrics(
    sources: np.ndarray | None,
    *,
    requested_labels: Sequence[int] | np.ndarray | None = None,
    source_labels: Sequence[int] | np.ndarray | None = None,
    source_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Return cheap source/latent diversity and provenance diagnostics.

    The unique count prefers recorded dataset indices when they are available;
    otherwise it falls back to rounded source rows.  A zero diversity value is a
    red flag for source-prior diagnostics because every generated sample is
    starting from exactly the same latent/source image.
    """
    if sources is None:
        return {
            "source_unique_count": 0,
            "source_diversity_l2": 0.0,
            "source_pair_l2": 0.0,
            "source_label_match_rate": float("nan"),
            "source_index_unique_count": -1,
        }
    src = np.asarray(sources, dtype=np.float64).reshape(np.asarray(sources).shape[0], -1)
    if src.shape[0] == 0:
        return {
            "source_unique_count": 0,
            "source_diversity_l2": 0.0,
            "source_pair_l2": 0.0,
            "source_label_match_rate": float("nan"),
            "source_index_unique_count": -1,
        }
    rounded_unique_count = int(np.unique(np.round(src, decimals=12), axis=0).shape[0])
    source_index_unique_count: int | None = None
    if source_indices is not None:
        idx = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        valid = idx >= 0
        if np.any(valid):
            source_index_unique_count = int(np.unique(idx[valid]).size)
    unique_count = rounded_unique_count if source_index_unique_count is None else source_index_unique_count
    centered = src - src.mean(axis=0, keepdims=True)
    diversity_l2 = float(np.sqrt(np.sum(centered * centered, axis=1)).mean())
    pair_l2 = (
        float(np.sqrt(np.sum((src[1:] - src[:-1]) ** 2, axis=1)).mean())
        if src.shape[0] > 1
        else 0.0
    )
    match_rate: float = float("nan")
    if requested_labels is not None and source_labels is not None:
        req = np.asarray(requested_labels, dtype=np.int64).reshape(-1)
        lab = np.asarray(source_labels, dtype=np.int64).reshape(-1)
        if req.shape == lab.shape and req.size > 0:
            valid = lab >= 0
            match_rate = float(np.mean(req[valid] == lab[valid])) if np.any(valid) else float("nan")
    return {
        "source_unique_count": unique_count,
        "source_diversity_l2": diversity_l2,
        "source_pair_l2": pair_l2,
        "source_label_match_rate": match_rate,
        "source_index_unique_count": -1 if source_index_unique_count is None else source_index_unique_count,
    }


def source_batch_diagnostics(
    sources: np.ndarray | None,
    *,
    labels: Sequence[int] | np.ndarray | None = None,
    requested_labels: Sequence[int] | np.ndarray | None = None,
    source_indices: Sequence[int] | np.ndarray | None = None,
    source_labels: Sequence[int] | np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Backward-compatible wrapper for ``source_diversity_metrics``."""
    return source_diversity_metrics(
        sources,
        requested_labels=requested_labels if requested_labels is not None else labels,
        source_labels=source_labels,
        source_indices=source_indices,
    )


def nearest_class_mean_metrics(
    samples: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    class_means: np.ndarray,
) -> dict[str, float]:
    """Cheap label-conditioning diagnostics using nearest class-mean images."""
    raw = np.asarray(samples, dtype=np.float64)
    arr = raw.reshape(raw.shape[0], -1)
    means = np.asarray(class_means, dtype=np.float64).reshape(10, -1)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    diff = arr[:, None, :] - means[None, :, :]
    dist = np.mean(diff * diff, axis=2)
    nearest = np.argmin(dist, axis=1)
    correct_dist = dist[np.arange(arr.shape[0]), labels_arr]
    masked = dist.copy()
    masked[np.arange(arr.shape[0]), labels_arr] = np.inf
    nearest_wrong = np.min(masked, axis=1)
    margin = nearest_wrong - correct_dist
    return {
        "nearest_mean_acc": float(np.mean(nearest == labels_arr)),
        "correct_mean_dist": float(np.mean(correct_dist)),
        "wrong_mean_margin": float(np.mean(margin)),
    }


def _serializable_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true", help="Download IDX MNIST if no ARFF file is present.")
    parser.add_argument("--examples-per-class", type=int, default=1000)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=256)
    parser.add_argument("--labels", type=str, default="cycle", help="'cycle' or comma-separated labels, e.g. 0,1,2")
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="poisson-ot-flow")
    parser.add_argument("--source-mode", choices=SOURCE_MODES, default="lowfreq")
    parser.add_argument("--source-lowfreq-size", type=int, default=7)
    parser.add_argument("--source-blur-sigma", type=float, default=1.0)
    parser.add_argument("--source-uniform-mix", type=float, default=0.15)
    parser.add_argument("--condition-on-source", dest="condition_on_source", action="store_true", default=True)
    parser.add_argument("--no-condition-on-source", dest="condition_on_source", action="store_false")
    parser.add_argument("--ot-cost-mode", choices=OT_COST_MODES, default="lowres")
    parser.add_argument("--ot-match-mode", choices=OT_MATCH_MODES, default="nearest")
    parser.add_argument("--ot-nearest-top-k", type=int, default=1)
    parser.add_argument("--ot-lowres-size", type=int, default=7)
    parser.add_argument("--ot-blur-sigma", type=float, default=1.0)
    parser.add_argument("--ot-com-weight", type=float, default=0.25)
    parser.add_argument("--mean-flow-prob", type=float, default=0.15)
    parser.add_argument("--mean-flow-warmup-prob", type=float, default=0.20)
    parser.add_argument("--mean-flow-warmup-steps", type=int, default=1000)
    parser.add_argument("--tau-sampling", choices=TAU_SAMPLING_MODES, default="endpoint-mixture")
    parser.add_argument("--tau-source-prob", type=float, default=0.35)
    parser.add_argument("--tau-data-prob", type=float, default=0.15)
    parser.add_argument("--free-weight", type=float, default=None, help="Sampling free-drift weight. Defaults to target-free-weight under --sde-curriculum, otherwise 0.")
    parser.add_argument("--noise-weight", type=float, default=None, help="Sampling noise weight. Defaults to target-noise-weight under --sde-curriculum, otherwise 0.")
    parser.add_argument("--learned-weight", type=float, default=1.0)
    parser.add_argument("--free-aware-target", dest="free_aware_target", action="store_true", default=None)
    parser.add_argument("--no-free-aware-target", dest="free_aware_target", action="store_false")
    parser.add_argument("--train-free-weight", type=float, default=None)
    parser.add_argument("--train-noise-weight", type=float, default=None)
    parser.add_argument("--on-policy-use-free", dest="on_policy_use_free", action="store_true", default=None)
    parser.add_argument("--no-on-policy-use-free", dest="on_policy_use_free", action="store_false")
    parser.add_argument("--on-policy-use-noise", dest="on_policy_use_noise", action="store_true", default=None)
    parser.add_argument("--no-on-policy-use-noise", dest="on_policy_use_noise", action="store_false")
    parser.add_argument("--stochastic-step-loss", dest="stochastic_step_loss", action="store_true", default=None)
    parser.add_argument("--no-stochastic-step-loss", dest="stochastic_step_loss", action="store_false")
    parser.add_argument("--same-noise-step-loss", dest="same_noise_step_loss", action="store_true", default=True)
    parser.add_argument("--no-same-noise-step-loss", dest="same_noise_step_loss", action="store_false")
    parser.add_argument("--sde-curriculum", action="store_true")
    parser.add_argument("--sde-ramp-steps", type=int, default=3000)
    parser.add_argument("--target-free-weight", type=float, default=0.02)
    parser.add_argument("--target-noise-weight", type=float, default=0.003)
    parser.add_argument("--flux-scale", type=float, default=20.0)
    parser.add_argument("--target-flux-clip", type=float, default=10.0)
    parser.add_argument("--divergence-loss-weight", type=float, default=0.50)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--step-loss-weight", type=float, default=0.25)
    parser.add_argument("--state-jitter-weight", type=float, default=0.0)
    parser.add_argument("--velocity-target", choices=VELOCITY_TARGET_MODES, default="constant")
    parser.add_argument("--min-tau-fraction", type=float, default=0.03)
    parser.add_argument("--horizon-scale", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--edge-alpha-mode", choices=EDGE_ALPHA_MODES, default="legacy")
    parser.add_argument("--terminal-lambda", type=float, default=3.0)
    parser.add_argument("--on-policy-prob", type=float, default=0.25)
    parser.add_argument("--on-policy-warmup-steps", type=int, default=1500)
    parser.add_argument("--on-policy-prefix-steps", type=int, default=16)
    parser.add_argument("--adaptive-sampling", action="store_true")
    parser.add_argument("--clip-target", type=float, default=0.03)
    parser.add_argument("--max-substeps", type=int, default=4)
    parser.add_argument("--deterministic-sampling", action="store_true")
    parser.add_argument("--preview-every", type=int, default=500)
    parser.add_argument("--preview-sample-steps", type=int, default=64)
    parser.add_argument("--preview-num-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-ablation-samples", action="store_true", help="Save learned-only/free-only/noise-only/stochastic sample grids from the same sources.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/experiment10_mnist_flux"))
    args = parser.parse_args(argv)

    sample_free_weight = (
        float(args.target_free_weight)
        if args.free_weight is None and bool(args.sde_curriculum)
        else (0.0 if args.free_weight is None else float(args.free_weight))
    )
    sample_noise_weight = (
        float(args.target_noise_weight)
        if args.noise_weight is None and bool(args.sde_curriculum)
        else (0.0 if args.noise_weight is None else float(args.noise_weight))
    )
    free_aware_target = bool(args.sde_curriculum) if args.free_aware_target is None else bool(args.free_aware_target)
    on_policy_use_free = bool(args.sde_curriculum) if args.on_policy_use_free is None else bool(args.on_policy_use_free)
    on_policy_use_noise = bool(args.sde_curriculum) if args.on_policy_use_noise is None else bool(args.on_policy_use_noise)
    stochastic_step_loss = bool(args.sde_curriculum) if args.stochastic_step_loss is None else bool(args.stochastic_step_loss)

    config = DirectFluxMNISTConfig(
        alpha=float(args.alpha),
        beta=float(args.beta),
        edge_alpha_mode=str(args.edge_alpha_mode),
        horizon_scale=float(args.horizon_scale),
        num_steps=int(args.sample_steps),
        target_mode=str(args.target_mode),
        source_mode=str(args.source_mode),
        source_lowfreq_size=int(args.source_lowfreq_size),
        source_blur_sigma=float(args.source_blur_sigma),
        source_uniform_mix=float(args.source_uniform_mix),
        condition_on_source=bool(args.condition_on_source),
        ot_cost_mode=str(args.ot_cost_mode),
        ot_match_mode=str(args.ot_match_mode),
        ot_nearest_top_k=int(args.ot_nearest_top_k),
        ot_lowres_size=int(args.ot_lowres_size),
        ot_blur_sigma=float(args.ot_blur_sigma),
        ot_com_weight=float(args.ot_com_weight),
        mean_flow_prob=float(args.mean_flow_prob),
        mean_flow_warmup_prob=float(args.mean_flow_warmup_prob),
        mean_flow_warmup_steps=int(args.mean_flow_warmup_steps),
        tau_sampling=str(args.tau_sampling),
        tau_source_prob=float(args.tau_source_prob),
        tau_data_prob=float(args.tau_data_prob),
        free_weight=float(sample_free_weight),
        noise_weight=float(sample_noise_weight),
        learned_weight=float(args.learned_weight),
        free_aware_target=bool(free_aware_target),
        train_free_weight=None if args.train_free_weight is None else float(args.train_free_weight),
        train_noise_weight=None if args.train_noise_weight is None else float(args.train_noise_weight),
        on_policy_use_free=bool(on_policy_use_free),
        on_policy_use_noise=bool(on_policy_use_noise),
        stochastic_step_loss=bool(stochastic_step_loss),
        same_noise_step_loss=bool(args.same_noise_step_loss),
        sde_curriculum=bool(args.sde_curriculum),
        sde_ramp_steps=int(args.sde_ramp_steps),
        target_free_weight=float(args.target_free_weight),
        target_noise_weight=float(args.target_noise_weight),
        flux_scale=float(args.flux_scale),
        target_flux_clip=float(args.target_flux_clip),
        divergence_loss_weight=float(args.divergence_loss_weight),
        node_loss_weight=float(args.node_loss_weight),
        step_loss_weight=float(args.step_loss_weight),
        state_jitter_weight=float(args.state_jitter_weight),
        velocity_target=str(args.velocity_target),
        min_tau_fraction=float(args.min_tau_fraction),
        terminal_lambda=float(args.terminal_lambda),
        on_policy_prob=float(args.on_policy_prob),
        on_policy_warmup_steps=int(args.on_policy_warmup_steps),
        on_policy_prefix_steps=int(args.on_policy_prefix_steps),
        adaptive_sampling=bool(args.adaptive_sampling),
        clip_target=float(args.clip_target),
        max_substeps=int(args.max_substeps),
    )
    device = torch.device(
        "cuda" if args.device is None and torch.cuda.is_available() else "cpu" if args.device is None else args.device
    )
    print(f"Experiment 10g direct-flux MNIST on device={device}")
    print(
        "Laptop-friendly settings: "
        f"target_mode={config.target_mode}, source_mode={config.source_mode}, "
        f"ot={config.ot_cost_mode}/{config.ot_lowres_size}, match={config.ot_match_mode}, tau={config.tau_sampling}, "
        f"source_cond={config.condition_on_source}, velocity={config.velocity_target}, "
        f"free_aware={config.free_aware_target}, sde_curr={config.sde_curriculum}, edge_alpha={config.edge_alpha_mode}, "
        f"on_policy={config.on_policy_prob}, onp_sde=({config.on_policy_use_free},{config.on_policy_use_noise}), "
        f"step_loss={config.step_loss_weight}, stochastic_step={config.stochastic_step_loss}, adaptive={config.adaptive_sampling}, "
        f"train_steps={args.train_steps}, batch={args.batch_size}, base_channels={args.base_channels}, "
        f"horizon={natural_horizon(config):.3e}, sample_steps={args.sample_steps}, "
        f"weights=(free={config.free_weight}, noise={config.noise_weight}, learned={config.learned_weight})"
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
    final_class_means = _compute_class_mean_measures(dataset.train_images, dataset.train_labels, config.grid_size)
    preview_dir = args.out_dir / "previews" if int(args.preview_every) > 0 else None
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
        preview_dir=preview_dir,
        preview_every=int(args.preview_every),
        preview_sample_steps=int(args.preview_sample_steps),
        preview_num_samples=int(args.preview_num_samples),
    )

    print("Training complete; starting generation")
    labels = _parse_label_sequence(args.labels, int(args.num_samples))
    result = simulate_direct_flux_generation(
        model,
        labels,
        num_steps=int(args.sample_steps),
        deterministic=bool(args.deterministic_sampling),
        device=device,
        seed=int(args.seed) + 1,
        use_amp=not bool(args.no_amp),
        show_progress=not bool(args.no_progress),
        source_images=dataset.train_images,
        source_labels=dataset.train_labels,
    )

    print("Generation complete; saving artifacts")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "experiment10_direct_flux_mnist.pt"
    samples_path = args.out_dir / "experiment10_samples.npz"
    png_path = args.out_dir / "experiment10_samples.png"
    final_metrics = nearest_class_mean_metrics(result.samples, result.labels, final_class_means)
    source_metrics = source_batch_diagnostics(
        result.sources if result.sources is not None else result.samples,
        requested_labels=result.labels,
        source_indices=result.source_indices,
        source_labels=result.source_labels,
    )
    source_label_match_value = (
        float("nan")
        if source_metrics.get("source_label_match_rate") is None
        else float(source_metrics["source_label_match_rate"])
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "args": _serializable_args(args),
            "history": history,
            "labels": result.labels,
            "clipping_fraction": result.clipping_fraction,
            "final_metrics": final_metrics,
            "source_metrics": source_metrics,
            "component_metrics": {
                "learned_step_rms": result.learned_step_rms,
                "free_step_rms": result.free_step_rms,
                "noise_step_rms": result.noise_step_rms,
                "free_to_learned_ratio": result.free_to_learned_ratio,
                "noise_to_learned_ratio": result.noise_to_learned_ratio,
            },
        },
        ckpt_path,
    )
    np.savez_compressed(
        samples_path,
        samples=result.samples,
        labels=result.labels,
        sources=result.sources,
        source_indices=np.asarray([] if result.source_indices is None else result.source_indices, dtype=np.int64),
        source_labels=np.asarray([] if result.source_labels is None else result.source_labels, dtype=np.int64),
        source_unique_count=np.asarray([source_metrics["source_unique_count"]], dtype=np.float64),
        source_diversity_l2=np.asarray([source_metrics["source_diversity_l2"]], dtype=np.float64),
        source_pair_l2=np.asarray([source_metrics["source_pair_l2"]], dtype=np.float64),
        source_label_match_rate=np.asarray([source_label_match_value], dtype=np.float64),
        clipping_fraction=np.asarray([result.clipping_fraction], dtype=np.float64),
        nearest_mean_acc=np.asarray([final_metrics["nearest_mean_acc"]], dtype=np.float64),
        correct_mean_dist=np.asarray([final_metrics["correct_mean_dist"]], dtype=np.float64),
        wrong_mean_margin=np.asarray([final_metrics["wrong_mean_margin"]], dtype=np.float64),
        learned_step_rms=np.asarray([0.0 if result.learned_step_rms is None else result.learned_step_rms], dtype=np.float64),
        free_step_rms=np.asarray([0.0 if result.free_step_rms is None else result.free_step_rms], dtype=np.float64),
        noise_step_rms=np.asarray([0.0 if result.noise_step_rms is None else result.noise_step_rms], dtype=np.float64),
        free_to_learned_ratio=np.asarray([0.0 if result.free_to_learned_ratio is None else result.free_to_learned_ratio], dtype=np.float64),
        noise_to_learned_ratio=np.asarray([0.0 if result.noise_to_learned_ratio is None else result.noise_to_learned_ratio], dtype=np.float64),
    )
    try:
        save_flux_samples_grid(result.samples, result.labels, png_path, grid_size=config.grid_size)
        print(f"Saved preview: {png_path}")
    except RuntimeError as exc:
        print(f"Skipping PNG preview: {exc}")

    if bool(args.save_ablation_samples) and result.sources is not None:
        ablations = [
            ("learned_only", dict(free_weight=0.0, noise_weight=0.0, learned_weight=1.0, deterministic=True)),
            ("free_only", dict(free_weight=config.free_weight, noise_weight=0.0, learned_weight=0.0, deterministic=True)),
            ("noise_only", dict(free_weight=0.0, noise_weight=config.noise_weight, learned_weight=0.0, deterministic=False)),
        ]
        for name, overrides in ablations:
            if name == "noise_only" and config.noise_weight <= 0.0:
                continue
            ablation = simulate_direct_flux_generation(
                model,
                result.labels,
                num_steps=int(args.sample_steps),
                deterministic=bool(overrides.pop("deterministic")),
                device=device,
                seed=int(args.seed) + 100 + len(name),
                use_amp=not bool(args.no_amp),
                show_progress=False,
                initial_states=result.sources,
                free_weight=float(overrides["free_weight"]),
                noise_weight=float(overrides["noise_weight"]),
                learned_weight=float(overrides["learned_weight"]),
            )
            out_png = args.out_dir / f"experiment10_samples_{name}.png"
            try:
                save_flux_samples_grid(ablation.samples, ablation.labels, out_png, grid_size=config.grid_size)
                print(f"Saved ablation preview: {out_png}")
            except RuntimeError as exc:
                print(f"Skipping {name} ablation PNG: {exc}")

    ent, max_mass = _samples_stats(result.samples)
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved samples: {samples_path}")
    print(f"Final clipping fraction: {result.clipping_fraction:.4f}")
    print(f"Final sample entropy: {ent:.4f}; mean max pixel mass: {max_mass:.4f}")
    print(
        "Step component RMS: "
        f"learned={0.0 if result.learned_step_rms is None else result.learned_step_rms:.4g}, "
        f"free={0.0 if result.free_step_rms is None else result.free_step_rms:.4g}, "
        f"noise={0.0 if result.noise_step_rms is None else result.noise_step_rms:.4g}, "
        f"free/learned={0.0 if result.free_to_learned_ratio is None else result.free_to_learned_ratio:.3f}, "
        f"noise/learned={0.0 if result.noise_to_learned_ratio is None else result.noise_to_learned_ratio:.3f}"
    )
    print(
        "Source diagnostics: "
        f"unique={source_metrics['source_unique_count']:.0f}, "
        f"div_l2={source_metrics['source_diversity_l2']:.4g}, "
        f"pair_l2={source_metrics['source_pair_l2']:.4g}, "
        f"label_match={source_label_match_value:.3f}"
    )
    print(
        "Nearest class-mean diagnostics: "
        f"acc={final_metrics['nearest_mean_acc']:.3f}, "
        f"correct_dist={final_metrics['correct_mean_dist']:.4g}, "
        f"wrong_margin={final_metrics['wrong_mean_margin']:.4g}"
    )


if __name__ == "__main__":
    main()
