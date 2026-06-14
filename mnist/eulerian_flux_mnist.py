from __future__ import annotations

r"""Example 10/10b/10c/10d/10e/10g/10h/10i/10j: MNIST generation with directly learned Eulerian edge fluxes.

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
train and sample with stochastic dynamics.  Experiment 10h merges the rollout-sharpening machinery back into stochastic training, replaces transposed-convolution upsampling by resize-conv by default, and can project learned edge fluxes to their minimum-energy divergence-equivalent part to suppress checkerboard/curl artifacts.  Experiment 10i adds a replay on-policy cache, cheaper projected-flux losses, cached Poisson denominators, rollout-endpoint sharpening losses, timing diagnostics, and diffusion-process figures. Experiment 10j replaces independent replay rollouts with trajectory-snapshot replay, adds safe residual replay targets, and optionally uses EMA weights for cache building and sampling. Experiment 10k adds terminal classifier diagnostics/losses, terminal-biased replay snapshots, endpoint losses, optional classifier-based sample selection, and good/bad sample annotation analysis. Experiment 10l makes terminal/classifier losses safe by gating them to late-terminal states, separating classifier diagnostics from classifier training, and disabling unsafe endpoint TV by default. Experiment 10r attempted more aggressive terminal replay/targets. Experiment 10s reverts the unstable parts: terminal batches use the global/mixed flux teacher by default, trajectory replay de-duplicates terminal snapshots again, and rollout-to-zero is no longer triggered implicitly by terminal microbatches.
"""

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from contextlib import contextmanager, nullcontext
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
VELOCITY_TARGET_MODES = ("constant", "residual", "mixed", "safe-residual")
TAU_SAMPLING_MODES = ("uniform", "endpoint-mixture")
OT_COST_MODES = ("lowres", "pixel")
OT_MATCH_MODES = ("minibatch", "nearest", "topk")
EDGE_ALPHA_MODES = ("legacy", "grid", "alpha_eff")
ON_POLICY_PREFIX_MODES = ("short", "uniform", "late-biased")
ON_POLICY_MODES = ("online", "replay", "off")
ON_POLICY_CACHE_MODES = ("independent", "trajectory")
ON_POLICY_TARGET_MODES = ("residual", "mixed", "constant", "safe-residual")
UPSAMPLE_MODES = ("transpose", "resize-conv")
FLUX_PARAMETERIZATION_MODES = ("edge", "projected")
SAMPLE_SELECTION_METRICS = ("none", "classifier-confidence", "composite", "composite-local", "composite-gap")
CLASSIFIER_LOSS_MODES = ("off", "terminal", "low-confidence-terminal")
TERMINAL_LOSS_MODES = ("fixed", "near-terminal", "to-terminal")

__all__ = [
    "DirectFluxMNISTConfig",
    "MNISTMeasureDataset",
    "SourceBatch",
    "FluxTrainingBatch",
    "FluxGenerationResult",
    "make_experiment10_run_dir",
    "OnPolicyReplayCache",
    "ClasswiseOTCache",
    "OT_MATCH_MODES",
    "EDGE_ALPHA_MODES",
    "ON_POLICY_PREFIX_MODES",
    "ON_POLICY_MODES",
    "ON_POLICY_CACHE_MODES",
    "ON_POLICY_TARGET_MODES",
    "UPSAMPLE_MODES",
    "FLUX_PARAMETERIZATION_MODES",
    "SAMPLE_SELECTION_METRICS",
    "CLASSIFIER_LOSS_MODES",
    "TERMINAL_LOSS_MODES",
    "TinyMNISTClassifier",
    "train_or_load_mnist_classifier",
    "classifier_generation_metrics",
    "compute_shape_statistics_np",
    "compute_class_shape_statistics",
    "terminal_local_shape_loss_torch",
    "local_shape_metrics_np",
    "write_goodbad_sample_report",
    "select_generation_result_by_classifier",
    "analyze_goodbad_annotations",
    "DirectFluxUNet",
    "edge_alpha_value",
    "natural_horizon",
    "load_mnist_measure_dataset",
    "sample_flux_training_batch",
    "sample_terminal_flux_training_batch",
    "terminal_potential_and_log_gradient_torch",
    "terminal_conditioning_flux_torch",
    "free_drift_flux_torch",
    "edge_noise_std_channels",
    "reference_step_substep_diagnostics_torch",
    "choose_reference_substeps_torch",
    "MaskedReferenceStepResult",
    "masked_reference_free_step_torch",
    "step_component_rms_torch",
    "poisson_flux_from_velocity_torch",
    "project_edge_flux_torch",
    "flux_curl_torch",
    "sample_total_variation_torch",
    "sample_checkerboard_energy_torch",
    "build_classwise_ot_cache",
    "training_target_flux_torch",
    "make_on_policy_training_batch",
    "build_on_policy_replay_cache",
    "sample_on_policy_replay_batch",
    "update_ema_state",
    "temporary_ema_weights",
    "save_diffusion_process_figure",
    "save_diffusion_marginal_process_figure",
    "direct_flux_rollout_consistency_loss",
    "image_total_variation",
    "checkerboard_energy_torch",
    "flux_curl_torch",
    "apply_flux_parameterization_torch",
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
    "save_flux_process_figure",
    "source_diversity_metrics",
    "main",
]


@dataclass(frozen=True)
class DirectFluxMNISTConfig:
    """Configuration for the direct-flux MNIST experiment.

    Defaults are intentionally modest and designed for an 8 GB laptop GPU.  The
    default 10j path uses stable nearest-matched Poisson-flow, a low-frequency
    source, persistent source conditioning, stochastic-aware correction targets,
    trajectory-snapshot replay, safe residual on-policy targets, and optional
    EMA weights for cache construction and sampling.
    """

    grid_size: int = 28
    # Legacy experiments used ``alpha`` directly on every edge.  Theory uses
    # alpha_h = beta h^d = beta / grid_size^2 on the 2D MNIST grid.  D0/P0.5
    # uses the soft practical reference ``alpha_eff`` by default in its own CLI:
    # it is still a symmetric finite-dimensional Dirichlet(alpha_eff) reference,
    # but is interior-typical enough for Gaussian innovation regression.
    alpha: float = 1.0
    beta: float = 1.0
    alpha_eff: float = 1.0
    edge_alpha_mode: str = "legacy"
    horizon_scale: float = 1.0
    num_steps: int = 256
    limiter_fraction: float = 0.25

    target_mode: str = "poisson-ot-flow"
    source_mode: str = "lowfreq"
    source_lowfreq_size: int = 7
    source_blur_sigma: float = 1.0
    condition_on_source: bool = True
    upsample_mode: str = "resize-conv"
    flux_parameterization: str = "projected"
    # Experiment 10i: skip the expensive FFT projection for the main
    # teacher-forced losses. Projection is still used for sampler-consistency
    # losses and generation, where it matters for the actual dynamics.
    project_main_loss: bool = False

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
    target_free_weight: float = 0.015
    target_noise_weight: float = 0.002

    terminal_lambda: float = 3.0
    terminal_floor: float = 1e-3
    blur_sigmas: tuple[float, ...] = (0.75, 1.5)
    blur_weights: tuple[float, ...] = (0.5, 0.5)

    mass_floor: float = 1e-8
    source_concentration: float = 1.0
    source_uniform_mix: float = 0.15
    state_jitter_weight: float = 0.0
    velocity_target: str = "mixed"
    late_residual_fraction: float = 0.25
    late_residual_prob: float = 0.50
    min_tau_fraction: float = 0.03
    bridge_power: float = 1.0
    flux_scale: float = 20.0
    target_flux_clip: float = 10.0
    divergence_loss_weight: float = 0.50
    node_loss_weight: float = 1.0
    step_loss_weight: float = 0.25
    rollout_loss_weight: float = 0.15
    rollout_loss_steps: int = 6
    rollout_loss_batch_size: int = 64
    rollout_loss_warmup_steps: int = 1500
    rollout_loss_every: int = 2
    rollout_loss_prob: float = 1.0
    image_grad_loss_weight: float = 0.0
    rollout_image_grad_loss_weight: float = 0.03
    # Experiment 10l: endpoint/classifier losses are potentially dangerous away
    # from the terminal endpoint.  Keep them disabled by default and, when
    # enabled, gate them by terminal time below.
    rollout_endpoint_l2_weight: float = 0.0
    rollout_endpoint_bce_weight: float = 0.0
    rollout_endpoint_tv_weight: float = 0.0
    terminal_loss_tau_max_fraction: float = 0.06
    terminal_loss_ramp_steps: int = 3000
    terminal_loss_mode: str = "near-terminal"
    terminal_rollout_max_steps: int = 16
    terminal_loss_every: int = 4
    terminal_rollout_batch_size: int = 32
    terminal_target_mode: str = "mixed"
    terminal_batch_rollout_mode: str = "fixed"
    terminal_batch_prob: float = 0.25
    terminal_batch_size: int = 64
    terminal_tau_min_fraction: float = 0.00
    terminal_tau_max_fraction: float = 0.06
    hard_label_sampling: bool = False
    hard_labels: tuple[int, ...] = (2, 5, 6, 7, 9)
    hard_label_prob: float = 0.35
    target_tv_loss_weight: float = 0.0
    target_entropy_loss_weight: float = 0.0
    use_classifier_loss: bool = False
    classifier_loss_mode: str = "off"
    classifier_loss_confidence_threshold: float = 0.75
    classifier_loss_weight: float = 0.0
    classifier_confidence_loss_weight: float = 0.0
    classifier_loss_blur_sigma: float = 0.6
    terminal_shape_loss_weight: float = 0.0
    terminal_shape_entropy_weight: float = 1.0
    terminal_shape_tv_weight: float = 1.0
    terminal_shape_maxmass_weight: float = 0.5
    # Experiment 10o: local low-resolution terminal support/edge/gap losses.
    terminal_local_shape_loss_weight: float = 0.0
    terminal_target_support_weight: float = 1.0
    terminal_target_edge_weight: float = 0.5
    terminal_negative_space_weight: float = 0.0
    # Experiment 10p: safer one-sided/gap local losses and stricter negative-space diagnostics.
    terminal_negative_space_mode: str = "strict"
    terminal_negative_space_threshold: float = 0.08
    terminal_negative_space_temperature: float = 0.03
    terminal_gap_loss_weight: float = 0.0
    terminal_gap_threshold: float = 0.12
    terminal_gap_dilate_radius: int = 1
    terminal_missing_support_weight: float = 0.0
    terminal_extra_support_weight: float = 0.0
    terminal_extra_support_margin: float = 0.10
    # Experiment 10s: foreground recall and label-gated local topology losses.
    terminal_foreground_recall_weight: float = 0.0
    terminal_foreground_threshold: float = 0.18
    terminal_foreground_temperature: float = 0.04
    terminal_foreground_size: int = 14
    terminal_foreground_blur_sigma: float = 0.7
    terminal_gap_labels: tuple[int, ...] = (5, 9)
    terminal_extra_support_labels: tuple[int, ...] = (5, 9)
    terminal_foreground_labels: tuple[int, ...] = (2, 3, 6, 9)
    terminal_local_loss_max_ratio: float = 0.25
    terminal_local_shape_size: int = 14
    terminal_local_shape_blur_sigma: float = 0.7
    selection_classifier_weight: float = 1.0
    selection_entropy_weight: float = 0.5
    selection_tv_weight: float = 0.5
    selection_maxmass_weight: float = 0.25
    selection_checkerboard_weight: float = 0.25
    selection_local_support_weight: float = 0.5
    selection_local_edge_weight: float = 0.25
    selection_negative_space_weight: float = 0.5
    selection_gap_weight: float = 0.5
    selection_extra_support_weight: float = 0.25
    selection_foreground_weight: float = 0.5
    curl_loss_weight: float = 0.01
    edge_laplacian_loss_weight: float = 0.0
    checkerboard_loss_weight: float = 0.001

    on_policy_prob: float = 0.40
    on_policy_warmup_steps: int = 1500
    on_policy_prefix_steps: int = 16
    on_policy_prefix_mode: str = "uniform"
    on_policy_min_prefix_fraction: float = 0.05
    on_policy_max_prefix_fraction: float = 0.85
    on_policy_batch_size: int = 64
    on_policy_mode: str = "replay"
    on_policy_cache_size: int = 2048
    on_policy_cache_refresh_interval: int = 100
    on_policy_cache_rollout_batch_size: int = 128
    on_policy_cache_device: str = "cpu"
    on_policy_cache_mode: str = "trajectory"
    on_policy_cache_snapshots_per_traj: int = 16
    on_policy_cache_terminal_fraction: float = 0.50
    on_policy_cache_terminal_min_tau: float = 0.00
    on_policy_cache_terminal_max_tau: float = 0.08
    on_policy_target_mode: str = "safe-residual"
    on_policy_residual_max_ratio: float = 1.5
    ema_decay: float = 0.999
    use_ema_for_sampling: bool = True
    use_ema_for_cache: bool = True

    adaptive_sampling: bool = False
    clip_target: float = 0.03
    max_substeps: int = 4

    use_classifier_diagnostics: bool = False
    classifier_train_epochs: int = 2
    classifier_cache_path: str = ""
    classifier_batch_size: int = 256
    classifier_lr: float = 1e-3
    sample_rejection_factor: int = 1
    sample_selection_metric: str = "none"
    analyze_goodbad_file: bool = True

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
        if self.alpha_eff <= 0.0 or not math.isfinite(self.alpha_eff):
            raise ValueError("alpha_eff must be positive and finite")
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
        if self.upsample_mode not in UPSAMPLE_MODES:
            raise ValueError(f"upsample_mode must be one of {UPSAMPLE_MODES}")
        if self.flux_parameterization not in FLUX_PARAMETERIZATION_MODES:
            raise ValueError(f"flux_parameterization must be one of {FLUX_PARAMETERIZATION_MODES}")
        if not isinstance(self.project_main_loss, bool):
            raise ValueError("project_main_loss must be a bool")
        if not (0.0 <= self.state_jitter_weight < 1.0):
            raise ValueError("state_jitter_weight must be in [0, 1)")
        if self.velocity_target not in VELOCITY_TARGET_MODES:
            raise ValueError(f"velocity_target must be one of {VELOCITY_TARGET_MODES}")
        if not (0.0 < self.min_tau_fraction <= 1.0):
            raise ValueError("min_tau_fraction must be in (0, 1]")
        if not (0.0 <= self.late_residual_fraction <= 1.0):
            raise ValueError("late_residual_fraction must be in [0, 1]")
        if not (0.0 <= self.late_residual_prob <= 1.0):
            raise ValueError("late_residual_prob must be in [0, 1]")
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
        if self.image_grad_loss_weight < 0.0:
            raise ValueError("image_grad_loss_weight must be non-negative")
        if self.rollout_image_grad_loss_weight < 0.0:
            raise ValueError("rollout_image_grad_loss_weight must be non-negative")
        if self.rollout_endpoint_l2_weight < 0.0:
            raise ValueError("rollout_endpoint_l2_weight must be non-negative")
        if self.rollout_endpoint_bce_weight < 0.0:
            raise ValueError("rollout_endpoint_bce_weight must be non-negative")
        if self.rollout_endpoint_tv_weight < 0.0:
            raise ValueError("rollout_endpoint_tv_weight must be non-negative")
        if not (0.0 <= self.terminal_loss_tau_max_fraction <= 1.0):
            raise ValueError("terminal_loss_tau_max_fraction must be in [0, 1]")
        if self.terminal_loss_ramp_steps < 0:
            raise ValueError("terminal_loss_ramp_steps must be non-negative")
        if self.terminal_loss_mode not in TERMINAL_LOSS_MODES:
            raise ValueError(f"terminal_loss_mode must be one of {TERMINAL_LOSS_MODES}")
        if self.terminal_target_mode not in VELOCITY_TARGET_MODES:
            raise ValueError(f"terminal_target_mode must be one of {VELOCITY_TARGET_MODES}")
        if self.terminal_batch_rollout_mode not in {"fixed", "to-zero"}:
            raise ValueError("terminal_batch_rollout_mode must be 'fixed' or 'to-zero'")
        if self.terminal_rollout_max_steps <= 0:
            raise ValueError("terminal_rollout_max_steps must be positive")
        if self.terminal_loss_every <= 0:
            raise ValueError("terminal_loss_every must be positive")
        if self.terminal_rollout_batch_size <= 0:
            raise ValueError("terminal_rollout_batch_size must be positive")
        if not (0.0 <= self.terminal_batch_prob <= 1.0):
            raise ValueError("terminal_batch_prob must be in [0, 1]")
        if self.terminal_batch_size <= 0:
            raise ValueError("terminal_batch_size must be positive")
        if not (0.0 <= self.terminal_tau_min_fraction <= 1.0):
            raise ValueError("terminal_tau_min_fraction must be in [0, 1]")
        if not (0.0 <= self.terminal_tau_max_fraction <= 1.0):
            raise ValueError("terminal_tau_max_fraction must be in [0, 1]")
        if self.terminal_tau_min_fraction > self.terminal_tau_max_fraction:
            raise ValueError("terminal_tau_min_fraction must be <= max fraction")
        if not isinstance(self.hard_label_sampling, bool):
            raise ValueError("hard_label_sampling must be a bool")
        if not (0.0 <= self.hard_label_prob <= 1.0):
            raise ValueError("hard_label_prob must be in [0, 1]")
        if len(self.hard_labels) == 0 or any((int(label) < 0 or int(label) > 9) for label in self.hard_labels):
            raise ValueError("hard_labels must be a non-empty tuple of digits in [0, 9]")
        if self.target_tv_loss_weight < 0.0:
            raise ValueError("target_tv_loss_weight must be non-negative")
        if self.target_entropy_loss_weight < 0.0:
            raise ValueError("target_entropy_loss_weight must be non-negative")
        if not isinstance(self.use_classifier_loss, bool):
            raise ValueError("use_classifier_loss must be a bool")
        if self.classifier_loss_mode not in CLASSIFIER_LOSS_MODES:
            raise ValueError(f"classifier_loss_mode must be one of {CLASSIFIER_LOSS_MODES}")
        if not (0.0 <= self.classifier_loss_confidence_threshold <= 1.0):
            raise ValueError("classifier_loss_confidence_threshold must be in [0, 1]")
        if self.classifier_loss_weight < 0.0:
            raise ValueError("classifier_loss_weight must be non-negative")
        if self.classifier_confidence_loss_weight < 0.0:
            raise ValueError("classifier_confidence_loss_weight must be non-negative")
        if self.classifier_loss_blur_sigma < 0.0 or not math.isfinite(self.classifier_loss_blur_sigma):
            raise ValueError("classifier_loss_blur_sigma must be non-negative and finite")
        for name in (
            "terminal_shape_loss_weight",
            "terminal_shape_entropy_weight",
            "terminal_shape_tv_weight",
            "terminal_shape_maxmass_weight",
            "terminal_local_shape_loss_weight",
            "terminal_target_support_weight",
            "terminal_target_edge_weight",
            "terminal_negative_space_weight",
            "terminal_gap_loss_weight",
            "terminal_missing_support_weight",
            "terminal_extra_support_weight",
            "terminal_foreground_recall_weight",
            "terminal_foreground_threshold",
            "terminal_foreground_temperature",
            "terminal_foreground_blur_sigma",
            "terminal_local_loss_max_ratio",
            "terminal_local_shape_blur_sigma",
            "selection_classifier_weight",
            "selection_entropy_weight",
            "selection_tv_weight",
            "selection_maxmass_weight",
            "selection_checkerboard_weight",
            "selection_local_support_weight",
            "selection_local_edge_weight",
            "selection_negative_space_weight",
            "selection_gap_weight",
            "selection_extra_support_weight",
            "selection_foreground_weight",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be non-negative and finite")
        if self.terminal_negative_space_mode not in {"mean", "strict"}:
            raise ValueError("terminal_negative_space_mode must be 'mean' or 'strict'")
        if not (0.0 <= self.terminal_negative_space_threshold <= 1.0):
            raise ValueError("terminal_negative_space_threshold must be in [0, 1]")
        if self.terminal_negative_space_temperature <= 0.0 or not math.isfinite(self.terminal_negative_space_temperature):
            raise ValueError("terminal_negative_space_temperature must be positive and finite")
        if not (0.0 <= self.terminal_gap_threshold <= 1.0):
            raise ValueError("terminal_gap_threshold must be in [0, 1]")
        if self.terminal_gap_dilate_radius < 0:
            raise ValueError("terminal_gap_dilate_radius must be non-negative")
        if self.terminal_extra_support_margin < 0.0:
            raise ValueError("terminal_extra_support_margin must be non-negative")
        if self.terminal_foreground_size <= 0:
            raise ValueError("terminal_foreground_size must be positive")
        for name, digits in (
            ("terminal_gap_labels", self.terminal_gap_labels),
            ("terminal_extra_support_labels", self.terminal_extra_support_labels),
            ("terminal_foreground_labels", self.terminal_foreground_labels),
        ):
            if any((int(label) < 0 or int(label) > 9) for label in digits):
                raise ValueError(f"{name} must contain digits in [0, 9]")
        if self.rollout_loss_weight < 0.0:
            raise ValueError("rollout_loss_weight must be non-negative")
        if self.rollout_loss_steps < 0:
            raise ValueError("rollout_loss_steps must be non-negative")
        if self.rollout_loss_batch_size <= 0:
            raise ValueError("rollout_loss_batch_size must be positive")
        if self.rollout_loss_warmup_steps < 0:
            raise ValueError("rollout_loss_warmup_steps must be non-negative")
        if self.rollout_loss_every <= 0:
            raise ValueError("rollout_loss_every must be positive")
        if not (0.0 <= self.rollout_loss_prob <= 1.0):
            raise ValueError("rollout_loss_prob must be in [0, 1]")
        if self.curl_loss_weight < 0.0:
            raise ValueError("curl_loss_weight must be non-negative")
        if self.edge_laplacian_loss_weight < 0.0:
            raise ValueError("edge_laplacian_loss_weight must be non-negative")
        if self.checkerboard_loss_weight < 0.0:
            raise ValueError("checkerboard_loss_weight must be non-negative")
        if not (0.0 <= self.on_policy_prob <= 1.0):
            raise ValueError("on_policy_prob must be in [0, 1]")
        if self.on_policy_mode not in ON_POLICY_MODES:
            raise ValueError(f"on_policy_mode must be one of {ON_POLICY_MODES}")
        if self.on_policy_cache_size <= 0:
            raise ValueError("on_policy_cache_size must be positive")
        if self.on_policy_cache_refresh_interval <= 0:
            raise ValueError("on_policy_cache_refresh_interval must be positive")
        if self.on_policy_cache_rollout_batch_size <= 0:
            raise ValueError("on_policy_cache_rollout_batch_size must be positive")
        if self.on_policy_cache_device not in {"cpu", "cuda"}:
            raise ValueError("on_policy_cache_device must be 'cpu' or 'cuda'")
        if self.on_policy_cache_mode not in ON_POLICY_CACHE_MODES:
            raise ValueError(f"on_policy_cache_mode must be one of {ON_POLICY_CACHE_MODES}")
        if self.on_policy_cache_snapshots_per_traj <= 0:
            raise ValueError("on_policy_cache_snapshots_per_traj must be positive")
        if not (0.0 <= self.on_policy_cache_terminal_fraction <= 1.0):
            raise ValueError("on_policy_cache_terminal_fraction must be in [0, 1]")
        if not (0.0 <= self.on_policy_cache_terminal_min_tau <= 1.0):
            raise ValueError("on_policy_cache_terminal_min_tau must be in [0, 1]")
        if not (0.0 <= self.on_policy_cache_terminal_max_tau <= 1.0):
            raise ValueError("on_policy_cache_terminal_max_tau must be in [0, 1]")
        if self.on_policy_cache_terminal_min_tau > self.on_policy_cache_terminal_max_tau:
            raise ValueError("on_policy_cache_terminal_min_tau must be <= max_tau")
        if self.on_policy_target_mode not in ON_POLICY_TARGET_MODES:
            raise ValueError(f"on_policy_target_mode must be one of {ON_POLICY_TARGET_MODES}")
        if self.on_policy_residual_max_ratio <= 0.0 or not math.isfinite(self.on_policy_residual_max_ratio):
            raise ValueError("on_policy_residual_max_ratio must be positive and finite")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")
        if not isinstance(self.use_ema_for_sampling, bool):
            raise ValueError("use_ema_for_sampling must be a bool")
        if not isinstance(self.use_ema_for_cache, bool):
            raise ValueError("use_ema_for_cache must be a bool")
        if self.on_policy_warmup_steps < 0:
            raise ValueError("on_policy_warmup_steps must be non-negative")
        if self.on_policy_prefix_steps < 0:
            raise ValueError("on_policy_prefix_steps must be non-negative")
        if self.on_policy_prefix_mode not in ON_POLICY_PREFIX_MODES:
            raise ValueError(f"on_policy_prefix_mode must be one of {ON_POLICY_PREFIX_MODES}")
        if not (0.0 <= self.on_policy_min_prefix_fraction <= 1.0):
            raise ValueError("on_policy_min_prefix_fraction must be in [0, 1]")
        if not (0.0 <= self.on_policy_max_prefix_fraction <= 1.0):
            raise ValueError("on_policy_max_prefix_fraction must be in [0, 1]")
        if self.on_policy_min_prefix_fraction > self.on_policy_max_prefix_fraction:
            raise ValueError("on_policy_min_prefix_fraction must be <= max fraction")
        if self.on_policy_batch_size <= 0:
            raise ValueError("on_policy_batch_size must be positive")
        if not isinstance(self.adaptive_sampling, bool):
            raise ValueError("adaptive_sampling must be a bool")
        if not (0.0 <= self.clip_target <= 1.0):
            raise ValueError("clip_target must be in [0, 1]")
        if self.max_substeps <= 0:
            raise ValueError("max_substeps must be positive")
        if not isinstance(self.use_classifier_diagnostics, bool):
            raise ValueError("use_classifier_diagnostics must be a bool")
        if self.classifier_train_epochs < 0:
            raise ValueError("classifier_train_epochs must be non-negative")
        if self.classifier_batch_size <= 0:
            raise ValueError("classifier_batch_size must be positive")
        if self.classifier_lr <= 0.0 or not math.isfinite(self.classifier_lr):
            raise ValueError("classifier_lr must be positive and finite")
        if self.terminal_local_shape_size <= 1:
            raise ValueError("terminal_local_shape_size must be at least 2")
        if self.sample_rejection_factor <= 0:
            raise ValueError("sample_rejection_factor must be positive")
        if self.sample_selection_metric not in SAMPLE_SELECTION_METRICS:
            raise ValueError(f"sample_selection_metric must be one of {SAMPLE_SELECTION_METRICS}")
        if not isinstance(self.analyze_goodbad_file, bool):
            raise ValueError("analyze_goodbad_file must be a bool")


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
    step_index: int | None = None
    train_free_weight: float = 0.0
    train_noise_weight: float = 0.0
    is_terminal_batch: bool = False


@dataclass
class OnPolicyReplayCache:
    """Replay buffer of model-visited states for cheaper on-policy training."""

    batch: FluxTrainingBatch
    created_step: int
    refresh_seconds: float
    mode: str = "independent"
    tau_min: float = float("nan")
    tau_mean: float = float("nan")
    tau_max: float = float("nan")
    terminal_fraction: float = float("nan")
    terminal_requested_fraction: float = float("nan")
    terminal_actual_fraction: float = float("nan")
    terminal_snapshot_count: int = 0
    regular_snapshot_count: int = 0

    @property
    def size(self) -> int:
        return int(self.batch.states.shape[0])


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
    sample_entropy: float | None = None
    sample_total_variation: float | None = None
    sample_checkerboard_energy: float | None = None
    sample_highfreq_fraction: float | None = None


@dataclass(frozen=True)
class _TorchEdgeClass:
    tails: Tensor
    heads: Tensor
    flux_indices: Tensor


def edge_alpha_value(config: DirectFluxMNISTConfig) -> float:
    """Return the edge Dirichlet parameter used by mobility/free SDE terms.

    ``legacy`` uses the historical code parameter ``alpha``.  ``grid`` uses the
    manuscript scaling ``beta / grid_size**2`` on MNIST.  ``alpha_eff`` is the
    P0.5 soft practical reference: a symmetric Dirichlet(alpha_eff) grid law
    with faithful time-change coupling of drift and noise.
    """
    mode = str(config.edge_alpha_mode)
    if mode == "grid":
        n = float(config.grid_size)
        return float(config.beta) / (n * n)
    if mode == "alpha_eff":
        return float(config.alpha_eff)
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




class _UpsampleBlock(nn.Module):
    """Anti-checkerboard upsampling block.

    Transposed convolutions can inject a grid texture into generated edge fluxes.
    The resize-conv path upsamples by interpolation and then applies an ordinary
    circular convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, mode: str) -> None:
        super().__init__()
        if mode == "transpose":
            self.net = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        elif mode == "resize-conv":
            self.net = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            )
        else:
            raise ValueError(f"unknown upsample mode: {mode}")

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
        self.up2 = _UpsampleBlock(4 * channels, 2 * channels, str(config.upsample_mode))
        self.dec2 = _ConvBlock(4 * channels, 2 * channels)
        self.up1 = _UpsampleBlock(2 * channels, channels, str(config.upsample_mode))
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
        raw = float(self.config.flux_scale) * self.forward(tau, masses, labels, source_masses)
        return apply_flux_parameterization_torch(raw, masses, self.config)


class TinyMNISTClassifier(nn.Module):
    """Small LeNet-style classifier used for MNIST terminal diagnostics/losses.

    Inputs are density images scaled as ``grid_size ** 2 * mass`` with shape
    ``(B, 1, H, W)``.  The classifier is intentionally tiny so training it for a
    few epochs on the same local MNIST subset remains laptop-friendly.
    """

    def __init__(self, grid_size: int = 28, num_classes: int = 10) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (self.grid_size // 4) * (self.grid_size // 4), 128),
            nn.SiLU(),
            nn.Linear(128, self.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("classifier input must have shape (B, 1, H, W)")
        return self.net(x)


def _classifier_input_from_masses(masses: Tensor, grid_size: int) -> Tensor:
    if masses.ndim == 2:
        return masses.reshape(masses.shape[0], 1, int(grid_size), int(grid_size)) * float(grid_size * grid_size)
    if masses.ndim == 3:
        return masses[:, None, :, :] * float(grid_size * grid_size)
    if masses.ndim == 4:
        return masses * float(grid_size * grid_size)
    raise ValueError("masses must have shape (B,N), (B,H,W), or (B,1,H,W)")


def train_or_load_mnist_classifier(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    grid_size: int,
    cache_path: str | Path | None = None,
    train_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
    show_progress: bool = True,
) -> TinyMNISTClassifier:
    """Return a cached or freshly trained tiny MNIST classifier."""
    resolved_device = torch.device(device)
    model = TinyMNISTClassifier(grid_size=int(grid_size)).to(resolved_device)
    path = None if cache_path is None or str(cache_path) == "" else Path(cache_path)
    if path is not None and path.exists():
        payload = torch.load(path, map_location=resolved_device)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model
    if train_epochs <= 0:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model
    rng = np.random.default_rng(seed)
    x_np = np.asarray(images, dtype=np.float32).reshape(-1, int(grid_size), int(grid_size))
    y_np = np.asarray(labels, dtype=np.int64).reshape(-1)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=resolved_device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=resolved_device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    steps_per_epoch = max(1, int(math.ceil(x.shape[0] / max(1, int(batch_size)))))
    total_steps = int(train_epochs) * steps_per_epoch
    bar = _progress(range(total_steps), total=total_steps, desc="train MNIST classifier", disable=not show_progress)
    for step in bar:
        if step % steps_per_epoch == 0:
            perm_np = rng.permutation(x.shape[0])
            perm = torch.as_tensor(perm_np, dtype=torch.long, device=resolved_device)
        start = (step % steps_per_epoch) * int(batch_size)
        idx = perm[start : start + int(batch_size)]
        xb = _classifier_input_from_masses(x.index_select(0, idx), int(grid_size))
        yb = y.index_select(0, idx)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
        if hasattr(bar, "set_postfix"):
            pred = logits.argmax(dim=1)
            acc = (pred == yb).float().mean()
            bar.set_postfix(loss=float(loss.detach().cpu()), acc=float(acc.detach().cpu()))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "grid_size": int(grid_size)}, path)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def classifier_generation_metrics(
    samples: np.ndarray | Tensor,
    labels: np.ndarray | Tensor,
    classifier: TinyMNISTClassifier | None,
    *,
    grid_size: int,
    device: str | torch.device | None = None,
) -> dict[str, float | np.ndarray]:
    """Classify generated samples and return confidence/margin diagnostics."""
    if classifier is None:
        return {}
    resolved_device = next(classifier.parameters()).device if device is None else torch.device(device)
    classifier = classifier.to(resolved_device).eval()
    x = torch.as_tensor(samples, dtype=torch.float32, device=resolved_device)
    y = torch.as_tensor(labels, dtype=torch.long, device=resolved_device).reshape(-1)
    logits = classifier(_classifier_input_from_masses(x, int(grid_size)))
    probs = logits.softmax(dim=1)
    pred = probs.argmax(dim=1)
    target_probs = probs.gather(1, y.view(-1, 1)).squeeze(1)
    masked = probs.clone()
    masked.scatter_(1, y.view(-1, 1), -1.0)
    wrong_best = masked.max(dim=1).values
    margin = target_probs - wrong_best
    per_label = np.full(10, np.nan, dtype=np.float64)
    for digit in range(10):
        mask = y == digit
        if bool(mask.any()):
            per_label[digit] = float((pred[mask] == y[mask]).float().mean().detach().cpu())
    return {
        "classifier_acc": float((pred == y).float().mean().detach().cpu()),
        "classifier_confidence": float(target_probs.mean().detach().cpu()),
        "classifier_margin": float(margin.mean().detach().cpu()),
        "classifier_predictions": pred.detach().cpu().numpy().astype(np.int64),
        "classifier_target_probs": target_probs.detach().cpu().numpy().astype(np.float64),
        "classifier_margins": margin.detach().cpu().numpy().astype(np.float64),
        "per_label_classifier_acc": per_label,
    }



def compute_shape_statistics_np(samples: np.ndarray, *, grid_size: int = 28) -> dict[str, np.ndarray]:
    """Return per-sample entropy/TV/max-mass/checkerboard statistics."""
    arr = np.asarray(samples, dtype=np.float64).reshape(-1, int(grid_size) * int(grid_size))
    imgs = arr.reshape(arr.shape[0], int(grid_size), int(grid_size))
    entropy = -(arr.clip(1e-30) * np.log(arr.clip(1e-30))).sum(axis=1)
    tv = np.abs(np.roll(imgs, -1, axis=1) - imgs).sum(axis=(1, 2)) + np.abs(np.roll(imgs, -1, axis=2) - imgs).sum(axis=(1, 2))
    maxmass = arr.max(axis=1)
    yy, xx = np.mgrid[0:int(grid_size), 0:int(grid_size)]
    checker_pattern = ((yy + xx) % 2) * 2.0 - 1.0
    checkerboard = np.abs((imgs * checker_pattern[None]).sum(axis=(1, 2)))
    return {"entropy": entropy, "tv": tv, "maxmass": maxmass, "checkerboard": checkerboard}


def _gaussian_kernel1d_np(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    return kernel / max(float(kernel.sum()), 1e-12)


def _periodic_gaussian_blur_np(images: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.asarray(images, dtype=np.float64)
    arr = np.asarray(images, dtype=np.float64)
    kernel = _gaussian_kernel1d_np(float(sigma))
    out = arr.copy()
    tmp = np.zeros_like(out)
    radius = int((kernel.size - 1) // 2)
    for offset, weight in zip(range(-radius, radius + 1), kernel):
        tmp += float(weight) * np.roll(out, shift=offset, axis=-1)
    out2 = np.zeros_like(out)
    for offset, weight in zip(range(-radius, radius + 1), kernel):
        out2 += float(weight) * np.roll(tmp, shift=offset, axis=-2)
    return out2


def _local_downsample_np(images: np.ndarray, *, size: int) -> np.ndarray:
    arr = np.asarray(images, dtype=np.float64)
    h, w = int(arr.shape[-2]), int(arr.shape[-1])
    size = int(size)
    if h == size and w == size:
        return arr.copy()
    if h % size == 0 and w % size == 0:
        fh, fw = h // size, w // size
        return arr.reshape(arr.shape[0], size, fh, size, fw).mean(axis=(2, 4))
    # Fallback nearest-area binning for unusual sizes.
    ys = np.linspace(0, h, size + 1).round().astype(int)
    xs = np.linspace(0, w, size + 1).round().astype(int)
    out = np.zeros((arr.shape[0], size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            patch = arr[:, ys[i] : max(ys[i + 1], ys[i] + 1), xs[j] : max(xs[j + 1], xs[j] + 1)]
            out[:, i, j] = patch.mean(axis=(1, 2))
    return out


def _binary_dilate_np(mask: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """Periodic binary dilation for small low-resolution diagnostic masks."""
    arr = np.asarray(mask, dtype=bool)
    radius = int(radius)
    if radius <= 0:
        return arr.astype(np.float64)
    out = arr.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            out |= np.roll(np.roll(arr, dy, axis=-2), dx, axis=-1)
    return out.astype(np.float64)


def local_shape_maps_np(samples: np.ndarray, *, grid_size: int = 28, local_shape_size: int = 14, blur_sigma: float = 0.7) -> dict[str, np.ndarray]:
    """Low-resolution blurred support and edge maps for shape diagnostics."""
    n = int(grid_size)
    arr = np.asarray(samples, dtype=np.float64).reshape(-1, n, n)
    density = np.clip(arr * float(n * n), 0.0, 1.0)
    blurred = _periodic_gaussian_blur_np(density, sigma=float(blur_sigma))
    low = np.clip(_local_downsample_np(blurred, size=int(local_shape_size)), 0.0, 1.0)
    dx = np.roll(low, -1, axis=-1) - low
    dy = np.roll(low, -1, axis=-2) - low
    edge = np.sqrt(dx * dx + dy * dy + 1e-12)
    return {"local_support": low, "local_edge": edge}


def local_shape_metrics_np(
    samples: np.ndarray,
    labels: np.ndarray,
    shape_stats: dict[str, np.ndarray] | None,
    *,
    grid_size: int = 28,
    negative_space_mode: str = "strict",
    negative_space_threshold: float = 0.08,
    negative_space_temperature: float = 0.03,
    gap_threshold: float = 0.12,
    gap_dilate_radius: int = 1,
    extra_support_margin: float = 0.10,
    foreground_threshold: float = 0.18,
    foreground_temperature: float = 0.04,
) -> dict[str, np.ndarray]:
    """Per-sample local support/edge/gap/negative-space penalties."""
    count = int(np.asarray(samples).reshape(np.asarray(samples).shape[0], -1).shape[0])
    z = np.zeros((count,), dtype=np.float64)
    if not shape_stats or "local_support_mean" not in shape_stats:
        return {
            "local_support_loss": z,
            "local_edge_loss": z,
            "negative_space_mass": z,
            "strict_negative_space_mass": z,
            "gap_mass": z,
            "missing_support_loss": z,
            "extra_support_loss": z,
            "foreground_recall_loss": z,
            "foreground_mass_ratio": z,
        }
    support_ref_all = np.asarray(shape_stats["local_support_mean"], dtype=np.float64)
    size = int(support_ref_all.shape[-1])
    blur_sigma = float(shape_stats.get("local_blur_sigma", np.asarray([0.7]))[0]) if "local_blur_sigma" in shape_stats else 0.7
    maps = local_shape_maps_np(samples, grid_size=int(grid_size), local_shape_size=size, blur_sigma=blur_sigma)
    labs = np.asarray(labels, dtype=np.int64).reshape(-1).clip(0, 9)
    support_ref = support_ref_all[labs]
    edge_ref = np.asarray(shape_stats["local_edge_mean"], dtype=np.float64)[labs]
    neg_ref = np.asarray(shape_stats["local_negative_space"], dtype=np.float64)[labs]
    support_q90_all = np.asarray(shape_stats.get("local_support_q90", support_ref_all), dtype=np.float64)
    support_q90 = support_q90_all[labs]
    temp = max(float(negative_space_temperature), 1e-6)
    if str(negative_space_mode) == "strict":
        strict_neg = 1.0 / (1.0 + np.exp((support_q90 - float(negative_space_threshold)) / temp))
    else:
        strict_neg = neg_ref
    stroke = support_ref > float(gap_threshold)
    gap_mask = (support_ref < float(gap_threshold)).astype(np.float64) * _binary_dilate_np(stroke, radius=int(gap_dilate_radius))
    support_loss = np.mean(np.abs(maps["local_support"] - support_ref), axis=(1, 2))
    edge_loss = np.mean(np.abs(maps["local_edge"] - edge_ref), axis=(1, 2))
    negative_mass = np.mean(maps["local_support"] * neg_ref, axis=(1, 2))
    strict_negative_mass = np.mean(maps["local_support"] * strict_neg, axis=(1, 2))
    gap_mass = np.mean(maps["local_support"] * gap_mask, axis=(1, 2))
    missing = np.mean(np.maximum(0.0, support_ref - maps["local_support"]) ** 2, axis=(1, 2))
    extra = np.mean(np.maximum(0.0, maps["local_support"] - support_ref - float(extra_support_margin)) ** 2, axis=(1, 2))
    fg_temp = max(float(foreground_temperature), 1e-6)
    foreground = 1.0 / (1.0 + np.exp(-(support_ref - float(foreground_threshold)) / fg_temp))
    fg_denom = np.maximum(foreground.sum(axis=(1, 2)), 1e-6)
    foreground_recall = (foreground * np.maximum(0.0, support_ref - maps["local_support"]) ** 2).sum(axis=(1, 2)) / fg_denom
    foreground_mass = (foreground * maps["local_support"]).sum(axis=(1, 2)) / np.maximum((foreground * support_ref).sum(axis=(1, 2)), 1e-6)
    return {
        "local_support_loss": support_loss,
        "local_edge_loss": edge_loss,
        "negative_space_mass": negative_mass,
        "strict_negative_space_mass": strict_negative_mass,
        "gap_mass": gap_mass,
        "missing_support_loss": missing,
        "extra_support_loss": extra,
        "foreground_recall_loss": foreground_recall,
        "foreground_mass_ratio": foreground_mass,
    }

def compute_class_shape_statistics(images: np.ndarray, labels: np.ndarray, *, grid_size: int = 28, local_shape_size: int = 14, local_blur_sigma: float = 0.7) -> dict[str, np.ndarray]:
    """Classwise robust shape statistics for terminal-quality scoring/losses."""
    stats = compute_shape_statistics_np(images, grid_size=int(grid_size))
    labs = np.asarray(labels, dtype=np.int64).reshape(-1)
    out: dict[str, np.ndarray] = {}
    for name, values in stats.items():
        q25 = np.full(10, np.nan, dtype=np.float64)
        q50 = np.full(10, np.nan, dtype=np.float64)
        q75 = np.full(10, np.nan, dtype=np.float64)
        scale = np.full(10, np.nan, dtype=np.float64)
        for digit in range(10):
            mask = labs == digit
            if mask.any():
                vals = np.asarray(values[mask], dtype=np.float64)
                q25[digit] = float(np.quantile(vals, 0.25))
                q50[digit] = float(np.quantile(vals, 0.50))
                q75[digit] = float(np.quantile(vals, 0.75))
                scale[digit] = max(float(q75[digit] - q25[digit]), 1e-6)
        out[f"{name}_q25"] = q25
        out[f"{name}_median"] = q50
        out[f"{name}_q75"] = q75
        out[f"{name}_iqr"] = scale
    # Low-resolution local support/edge/negative-space references for 10o.
    local_maps = local_shape_maps_np(images, grid_size=int(grid_size), local_shape_size=int(local_shape_size), blur_sigma=float(local_blur_sigma))
    support_mean = np.zeros((10, int(local_shape_size), int(local_shape_size)), dtype=np.float64)
    support_q90 = np.zeros_like(support_mean)
    edge_mean = np.zeros_like(support_mean)
    negative = np.zeros_like(support_mean)
    for digit in range(10):
        mask = labs == digit
        if mask.any():
            support_mean[digit] = local_maps["local_support"][mask].mean(axis=0)
            support_q90[digit] = np.quantile(local_maps["local_support"][mask], 0.90, axis=0)
            edge_mean[digit] = local_maps["local_edge"][mask].mean(axis=0)
        negative[digit] = np.clip(1.0 - support_mean[digit], 0.0, 1.0)
    out["local_support_mean"] = support_mean
    out["local_support_q90"] = support_q90
    out["local_edge_mean"] = edge_mean
    out["local_negative_space"] = negative
    out["local_shape_size"] = np.asarray([int(local_shape_size)], dtype=np.int64)
    out["local_blur_sigma"] = np.asarray([float(local_blur_sigma)], dtype=np.float64)
    return out


def _shape_stats_to_torch(shape_stats: dict[str, np.ndarray] | None, *, device: torch.device, dtype: torch.dtype) -> dict[str, Tensor]:
    if not shape_stats:
        return {}
    return {key: torch.as_tensor(value, dtype=dtype, device=device) for key, value in shape_stats.items()}


def _per_sample_mass_entropy_torch(states: Tensor) -> Tensor:
    flat = states.reshape(states.shape[0], -1).clamp_min(1e-30)
    return -(flat * flat.log()).sum(dim=1)


def _per_sample_max_mass_torch(states: Tensor) -> Tensor:
    return states.reshape(states.shape[0], -1).amax(dim=1)


def terminal_shape_loss_torch(
    states: Tensor,
    labels: Tensor,
    shape_stats: dict[str, Tensor],
    *,
    grid_size: int,
    weights: Tensor,
    entropy_weight: float,
    tv_weight: float,
    maxmass_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Bounded terminal shape penalty for high-entropy / low-TV / low-peak samples."""
    if not shape_stats:
        z = states.new_tensor(0.0)
        return z, z, z, z
    y = labels.to(device=states.device, dtype=torch.long).reshape(-1).clamp(0, 9)
    ent = _per_sample_mass_entropy_torch(states)
    tv = image_total_variation_per_sample(states, grid_size=int(grid_size))
    maxmass = _per_sample_max_mass_torch(states)
    ent_q75 = shape_stats["entropy_q75"].index_select(0, y)
    ent_iqr = shape_stats["entropy_iqr"].index_select(0, y).clamp_min(1e-6)
    tv_q25 = shape_stats["tv_q25"].index_select(0, y)
    tv_iqr = shape_stats["tv_iqr"].index_select(0, y).clamp_min(1e-6)
    max_q25 = shape_stats["maxmass_q25"].index_select(0, y)
    max_iqr = shape_stats["maxmass_iqr"].index_select(0, y).clamp_min(1e-6)
    ent_loss_per = (F.relu((ent - ent_q75) / ent_iqr)).square()
    tv_loss_per = (F.relu((tv_q25 - tv) / tv_iqr)).square()
    max_loss_per = (F.relu((max_q25 - maxmass) / max_iqr)).square()
    ent_loss = _masked_mean_torch(ent_loss_per, weights)
    tv_loss = _masked_mean_torch(tv_loss_per, weights)
    max_loss = _masked_mean_torch(max_loss_per, weights)
    total = float(entropy_weight) * ent_loss + float(tv_weight) * tv_loss + float(maxmass_weight) * max_loss
    return total, ent_loss, tv_loss, max_loss

def _local_shape_maps_torch(
    states: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    size: int | None = None,
    blur_sigma: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Return blurred low-res support and edge maps for mass states."""
    n = int(config.grid_size)
    resolved_size = int(config.terminal_local_shape_size if size is None else size)
    resolved_sigma = float(config.terminal_local_shape_blur_sigma if blur_sigma is None else blur_sigma)
    density = (states.reshape(-1, 1, n, n) * float(n * n)).clamp(0.0, 1.0)
    if resolved_sigma > 0.0:
        density = _periodic_gaussian_blur_torch(density, sigma=resolved_sigma)
    if resolved_size != n:
        low = F.interpolate(density, size=(resolved_size, resolved_size), mode="area")
    else:
        low = density
    low = low.clamp(0.0, 1.0)
    dx = torch.roll(low, shifts=-1, dims=-1) - low
    dy = torch.roll(low, shifts=-1, dims=-2) - low
    edge = torch.sqrt(dx.square() + dy.square() + 1e-12)
    return low, edge


def _binary_dilate_torch(mask: Tensor, *, radius: int = 1) -> Tensor:
    """Periodic dilation for small support masks."""
    radius = int(radius)
    if radius <= 0:
        return mask.to(dtype=torch.float32)
    mask_f = mask.to(dtype=torch.float32)
    dilated = mask_f
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            dilated = torch.maximum(dilated, torch.roll(torch.roll(mask_f, dy, dims=-2), dx, dims=-1))
    return dilated


def _strict_negative_space_mask_torch(labels: Tensor, shape_stats: dict[str, Tensor] | None, config: DirectFluxMNISTConfig, fallback: Tensor) -> Tensor:
    """Classwise strict negative-space mask from high-quantile real support."""
    if shape_stats is None:
        return fallback
    y = labels.to(dtype=torch.long, device=fallback.device).reshape(-1).clamp(0, 9)
    if str(config.terminal_negative_space_mode) == "strict" and "local_support_q90" in shape_stats:
        support_q90 = shape_stats["local_support_q90"].to(device=fallback.device, dtype=fallback.dtype).index_select(0, y).unsqueeze(1)
        temp = max(float(config.terminal_negative_space_temperature), 1e-6)
        return torch.sigmoid((float(config.terminal_negative_space_threshold) - support_q90) / temp)
    if "local_negative_space" in shape_stats:
        return shape_stats["local_negative_space"].to(device=fallback.device, dtype=fallback.dtype).index_select(0, y).unsqueeze(1)
    return fallback




def _label_gate_torch(labels: Tensor, allowed: tuple[int, ...]) -> Tensor:
    """Return a float mask selecting labels in ``allowed``; empty means all labels."""
    if len(tuple(allowed)) == 0:
        return torch.ones_like(labels, dtype=torch.float32)
    y = labels.to(dtype=torch.long).reshape(-1)
    mask = torch.zeros_like(y, dtype=torch.bool)
    for label in tuple(int(x) for x in allowed):
        mask = mask | (y == int(label))
    return mask.to(dtype=torch.float32)


def terminal_foreground_recall_loss_torch(
    states: Tensor,
    targets: Tensor,
    labels: Tensor,
    config: DirectFluxMNISTConfig,
    *,
    weights: Tensor,
) -> Tensor:
    """Low-res one-sided recall loss on high-confidence target foreground.

    This focuses on missing/faded strokes and avoids symmetric support matching.
    It is label-gated by ``terminal_foreground_labels`` and terminal-gated by
    the caller-supplied weights.
    """
    pred_low, _ = _local_shape_maps_torch(
        states,
        config,
        size=int(config.terminal_foreground_size),
        blur_sigma=float(config.terminal_foreground_blur_sigma),
    )
    with torch.no_grad():
        target_low, _ = _local_shape_maps_torch(
            targets,
            config,
            size=int(config.terminal_foreground_size),
            blur_sigma=float(config.terminal_foreground_blur_sigma),
        )
        temp = max(float(config.terminal_foreground_temperature), 1e-6)
        foreground = torch.sigmoid((target_low - float(config.terminal_foreground_threshold)) / temp)
        denom = foreground.flatten(1).sum(dim=1).clamp_min(1e-6)
    recall_per = (foreground * F.relu(target_low - pred_low).square()).flatten(1).sum(dim=1) / denom
    label_gate = _label_gate_torch(labels, tuple(config.terminal_foreground_labels)).to(device=states.device, dtype=states.dtype)
    return _masked_mean_torch(recall_per, weights * label_gate)


def terminal_gap_shape_loss_torch(
    states: Tensor,
    targets: Tensor,
    labels: Tensor,
    shape_stats: dict[str, Tensor] | None,
    config: DirectFluxMNISTConfig,
    *,
    weights: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One-sided terminal local losses for gaps and support without exact matching."""
    pred_low, _pred_edge = _local_shape_maps_torch(states, config)
    with torch.no_grad():
        target_low, _target_edge = _local_shape_maps_torch(targets, config)
        stroke = (target_low > float(config.terminal_gap_threshold)).to(dtype=pred_low.dtype)
        near_stroke = _binary_dilate_torch(stroke, radius=int(config.terminal_gap_dilate_radius))
        gap_mask = (target_low < float(config.terminal_gap_threshold)).to(dtype=pred_low.dtype) * near_stroke
        strict_negative = _strict_negative_space_mask_torch(labels, shape_stats, config, fallback=(1.0 - target_low).detach())
    missing_per = F.relu(target_low - pred_low).square().flatten(1).mean(dim=1)
    extra_per = F.relu(pred_low - target_low - float(config.terminal_extra_support_margin)).square().flatten(1).mean(dim=1)
    gap_per = (pred_low * gap_mask).flatten(1).mean(dim=1)
    strict_negative_per = (pred_low * strict_negative).flatten(1).mean(dim=1)
    missing_gate = _label_gate_torch(labels, tuple(config.terminal_foreground_labels)).to(device=states.device, dtype=states.dtype)
    extra_gate = _label_gate_torch(labels, tuple(config.terminal_extra_support_labels)).to(device=states.device, dtype=states.dtype)
    gap_gate = _label_gate_torch(labels, tuple(config.terminal_gap_labels)).to(device=states.device, dtype=states.dtype)
    missing_loss = _masked_mean_torch(missing_per, weights * missing_gate)
    extra_loss = _masked_mean_torch(extra_per, weights * extra_gate)
    gap_loss = _masked_mean_torch(gap_per, weights * gap_gate)
    strict_negative_loss = _masked_mean_torch(strict_negative_per, weights)
    total = (
        float(config.terminal_missing_support_weight) * missing_loss
        + float(config.terminal_extra_support_weight) * extra_loss
        + float(config.terminal_gap_loss_weight) * gap_loss
        + float(config.terminal_negative_space_weight) * strict_negative_loss
    )
    return total, missing_loss, extra_loss, gap_loss, strict_negative_loss


def terminal_local_shape_loss_torch(
    states: Tensor,
    targets: Tensor,
    labels: Tensor,
    shape_stats: dict[str, Tensor] | None,
    config: DirectFluxMNISTConfig,
    *,
    weights: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Low-res terminal support/edge/negative-space loss for local stroke topology."""
    pred_low, pred_edge = _local_shape_maps_torch(states, config)
    with torch.no_grad():
        target_low, target_edge = _local_shape_maps_torch(targets, config)
    support_per = F.smooth_l1_loss(pred_low, target_low, reduction="none").flatten(1).mean(dim=1)
    edge_per = F.smooth_l1_loss(pred_edge, target_edge, reduction="none").flatten(1).mean(dim=1)
    if shape_stats is not None and "local_negative_space" in shape_stats:
        y = labels.to(device=states.device, dtype=torch.long).reshape(-1).clamp(0, 9)
        neg = shape_stats["local_negative_space"].to(device=states.device, dtype=states.dtype).index_select(0, y).unsqueeze(1)
    else:
        neg = (1.0 - target_low).detach()
    negative_per = (pred_low * neg).flatten(1).mean(dim=1)
    support_loss = _masked_mean_torch(support_per, weights)
    edge_loss = _masked_mean_torch(edge_per, weights)
    negative_loss = _masked_mean_torch(negative_per, weights)
    total = (
        float(config.terminal_target_support_weight) * support_loss
        + float(config.terminal_target_edge_weight) * edge_loss
        + float(config.terminal_negative_space_weight) * negative_loss
    )
    return total, support_loss, edge_loss, negative_loss


def _normalize_goodbad_token(raw: str) -> bool | None:
    token = re.sub(r"[^a-z0-9]+", "", raw.strip().lower())
    if not token:
        return None
    if token in {"g", "1", "true", "ok", "yes", "y"} or re.fullmatch(r"go+d", token):
        return True
    if token in {"b", "0", "false", "no", "n"} or re.fullmatch(r"ba+d", token):
        return False
    return None


def parse_goodbad_annotation_file(path: str | Path, *, expected_count: int | None = None, grid_cols: int = 8) -> tuple[np.ndarray, list[str]]:
    """Parse the first good/bad annotation block and return warnings.

    Only the first contiguous block of annotation rows is parsed. Blank lines
    or comment-like lines after the block stop parsing, so explanatory notes do
    not accidentally add extra ``bad`` tokens. Row token counts are checked
    against the expected grid width.
    """
    values: list[bool] = []
    warnings: list[str] = []
    started = False
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        if stripped.startswith("#"):
            if started:
                break
            continue
        row: list[bool] = []
        for raw in stripped.replace(",", " ").split():
            parsed = _normalize_goodbad_token(raw)
            if parsed is not None:
                row.append(parsed)
        if not row:
            if started:
                break
            continue
        started = True
        if grid_cols > 0 and len(row) != int(grid_cols):
            warnings.append(f"line {line_no}: expected {grid_cols} annotation tokens, found {len(row)}")
        values.extend(row)
        if expected_count is not None and len(values) >= int(expected_count):
            if len(values) > int(expected_count):
                warnings.append(f"annotation block has extra tokens; truncating {len(values)} to {expected_count}")
            values = values[: int(expected_count)]
            break
    if expected_count is not None and len(values) != int(expected_count):
        warnings.append(f"expected {expected_count} annotation tokens, parsed {len(values)}")
    return np.asarray(values, dtype=bool), warnings


def _parse_goodbad_tokens(path: str | Path, *, expected_count: int | None = None) -> np.ndarray:
    """Backward-compatible parser returning only the boolean annotations."""
    values, _warnings = parse_goodbad_annotation_file(path, expected_count=expected_count)
    return values


def analyze_goodbad_annotations(
    path: str | Path,
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    classifier_metrics: dict[str, float | np.ndarray] | None = None,
) -> dict[str, float | np.ndarray]:
    """Analyze optional human good/bad labels for a saved sample grid."""
    good, parse_warnings = parse_goodbad_annotation_file(path, expected_count=int(samples.shape[0]))
    count = min(int(good.size), int(samples.shape[0]))
    if count <= 0:
        return {}
    good = good[:count]
    samples = np.asarray(samples[:count], dtype=np.float64)
    labels = np.asarray(labels[:count], dtype=np.int64)
    entropy = -(samples.clip(1e-30) * np.log(samples.clip(1e-30))).sum(axis=1)
    n = int(round(math.sqrt(samples.shape[1])))
    imgs = samples.reshape(count, n, n)
    tv = np.abs(np.roll(imgs, -1, axis=1) - imgs).sum(axis=(1, 2)) + np.abs(np.roll(imgs, -1, axis=2) - imgs).sum(axis=(1, 2))
    good_by_label = np.full(10, np.nan, dtype=np.float64)
    bad_count_by_label = np.zeros(10, dtype=np.int64)
    for digit in range(10):
        mask = labels == digit
        if mask.any():
            good_by_label[digit] = float(good[mask].mean())
            bad_count_by_label[digit] = int((~good[mask]).sum())
    result: dict[str, float | np.ndarray] = {
        "human_good_rate": float(good.mean()),
        "human_good_count": float(good.sum()),
        "human_bad_count": float((~good).sum()),
        "human_good_by_label": good_by_label,
        "human_bad_count_by_label": bad_count_by_label,
        "annotation_token_count": float(good.size),
        "annotation_warning": "; ".join(parse_warnings),
        "annotation_parse_ok": float(0 if parse_warnings else 1),
        "entropy_good_mean": float(entropy[good].mean()) if good.any() else float("nan"),
        "entropy_bad_mean": float(entropy[~good].mean()) if (~good).any() else float("nan"),
        "tv_good_mean": float(tv[good].mean()) if good.any() else float("nan"),
        "tv_bad_mean": float(tv[~good].mean()) if (~good).any() else float("nan"),
    }
    if classifier_metrics is not None:
        target_probs = classifier_metrics.get("classifier_target_probs")
        margins = classifier_metrics.get("classifier_margins")
        if isinstance(target_probs, np.ndarray) and target_probs.shape[0] >= count:
            tp = target_probs[:count]
            result["classifier_conf_good_mean"] = float(tp[good].mean()) if good.any() else float("nan")
            result["classifier_conf_bad_mean"] = float(tp[~good].mean()) if (~good).any() else float("nan")
        if isinstance(margins, np.ndarray) and margins.shape[0] >= count:
            mg = margins[:count]
            result["classifier_margin_good_mean"] = float(mg[good].mean()) if good.any() else float("nan")
            result["classifier_margin_bad_mean"] = float(mg[~good].mean()) if (~good).any() else float("nan")
    return result


def write_goodbad_sample_report(
    output_path: str | Path,
    goodbad_path: str | Path,
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    classifier_metrics: dict[str, float | np.ndarray] | None = None,
    selection_scores: np.ndarray | None = None,
    grid_size: int = 28,
) -> None:
    """Write a per-sample CSV combining human annotations and diagnostics."""
    good, parse_warnings = parse_goodbad_annotation_file(goodbad_path, expected_count=int(samples.shape[0]))
    count = min(int(good.size), int(samples.shape[0]))
    if count <= 0:
        return
    stats = compute_shape_statistics_np(np.asarray(samples[:count]), grid_size=int(grid_size))
    labs = np.asarray(labels[:count], dtype=np.int64)
    preds = None
    probs = None
    margins = None
    if classifier_metrics is not None:
        preds = classifier_metrics.get("classifier_predictions")
        probs = classifier_metrics.get("classifier_target_probs")
        margins = classifier_metrics.get("classifier_margins")
    scores = None if selection_scores is None else np.asarray(selection_scores, dtype=np.float64).reshape(-1)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "row",
                "col",
                "label",
                "human_good",
                "classifier_pred",
                "classifier_prob",
                "classifier_margin",
                "entropy",
                "tv",
                "maxmass",
                "checkerboard",
                "selection_score",
            ],
        )
        writer.writeheader()
        for i in range(count):
            writer.writerow(
                {
                    "index": i,
                    "row": int(i // 8),
                    "col": int(i % 8),
                    "label": int(labs[i]),
                    "human_good": int(bool(good[i])),
                    "classifier_pred": "" if not isinstance(preds, np.ndarray) or preds.shape[0] <= i else int(preds[i]),
                    "classifier_prob": "" if not isinstance(probs, np.ndarray) or probs.shape[0] <= i else float(probs[i]),
                    "classifier_margin": "" if not isinstance(margins, np.ndarray) or margins.shape[0] <= i else float(margins[i]),
                    "entropy": float(stats["entropy"][i]),
                    "tv": float(stats["tv"][i]),
                    "maxmass": float(stats["maxmass"][i]),
                    "checkerboard": float(stats["checkerboard"][i]),
                    "selection_score": "" if scores is None or scores.shape[0] <= i else float(scores[i]),
                }
            )


def write_local_shape_report(
    output_path: str | Path,
    samples: np.ndarray,
    labels: np.ndarray,
    shape_stats: dict[str, np.ndarray] | None,
    *,
    classifier_metrics: dict[str, float | np.ndarray] | None = None,
    selection_scores: np.ndarray | None = None,
    grid_size: int = 28,
    negative_space_mode: str = "strict",
    negative_space_threshold: float = 0.08,
    negative_space_temperature: float = 0.03,
    gap_threshold: float = 0.12,
    gap_dilate_radius: int = 1,
    extra_support_margin: float = 0.10,
) -> None:
    """Write per-sample local shape diagnostics used by 10o/10p."""
    samples_arr = np.asarray(samples, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    stats = compute_shape_statistics_np(samples_arr, grid_size=int(grid_size))
    local = local_shape_metrics_np(
        samples_arr,
        labels_arr,
        shape_stats,
        grid_size=int(grid_size),
        negative_space_mode=str(negative_space_mode),
        negative_space_threshold=float(negative_space_threshold),
        negative_space_temperature=float(negative_space_temperature),
        gap_threshold=float(gap_threshold),
        gap_dilate_radius=int(gap_dilate_radius),
        extra_support_margin=float(extra_support_margin),
    )
    preds = classifier_metrics.get("classifier_predictions") if classifier_metrics else None
    probs = classifier_metrics.get("classifier_target_probs") if classifier_metrics else None
    margins = classifier_metrics.get("classifier_margins") if classifier_metrics else None
    scores = None if selection_scores is None else np.asarray(selection_scores, dtype=np.float64).reshape(-1)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "label",
                "classifier_pred",
                "classifier_prob",
                "classifier_margin",
                "entropy",
                "tv",
                "maxmass",
                "checkerboard",
                "local_support_loss",
                "local_edge_loss",
                "negative_space_mass",
                "strict_negative_space_mass",
                "gap_mass",
                "missing_support_loss",
                "extra_support_loss",
                "foreground_recall_loss",
                "foreground_mass_ratio",
                "selection_score",
            ],
        )
        writer.writeheader()
        for i in range(samples_arr.shape[0]):
            writer.writerow(
                {
                    "index": i,
                    "label": int(labels_arr[i]),
                    "classifier_pred": "" if not isinstance(preds, np.ndarray) or preds.shape[0] <= i else int(preds[i]),
                    "classifier_prob": "" if not isinstance(probs, np.ndarray) or probs.shape[0] <= i else float(probs[i]),
                    "classifier_margin": "" if not isinstance(margins, np.ndarray) or margins.shape[0] <= i else float(margins[i]),
                    "entropy": float(stats["entropy"][i]),
                    "tv": float(stats["tv"][i]),
                    "maxmass": float(stats["maxmass"][i]),
                    "checkerboard": float(stats["checkerboard"][i]),
                    "local_support_loss": float(local["local_support_loss"][i]),
                    "local_edge_loss": float(local["local_edge_loss"][i]),
                    "negative_space_mass": float(local["negative_space_mass"][i]),
                    "strict_negative_space_mass": float(local["strict_negative_space_mass"][i]),
                    "gap_mass": float(local["gap_mass"][i]),
                    "missing_support_loss": float(local["missing_support_loss"][i]),
                    "extra_support_loss": float(local["extra_support_loss"][i]),
                    "foreground_recall_loss": float(local["foreground_recall_loss"][i]),
                    "foreground_mass_ratio": float(local["foreground_mass_ratio"][i]),
                    "selection_score": "" if scores is None or scores.shape[0] <= i else float(scores[i]),
                }
            )


def selection_scores_for_candidates(
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    classifier_metrics: dict[str, float | np.ndarray] | None,
    shape_stats: dict[str, np.ndarray] | None,
    grid_size: int,
    classifier_weight: float,
    entropy_weight: float,
    tv_weight: float,
    maxmass_weight: float,
    checkerboard_weight: float,
    local_support_weight: float = 0.5,
    local_edge_weight: float = 0.25,
    negative_space_weight: float = 0.5,
    gap_weight: float = 0.5,
    extra_support_weight: float = 0.25,
    foreground_weight: float = 0.5,
    negative_space_mode: str = "strict",
    negative_space_threshold: float = 0.08,
    negative_space_temperature: float = 0.03,
    gap_threshold: float = 0.12,
    gap_dilate_radius: int = 1,
    extra_support_margin: float = 0.10,
    foreground_threshold: float = 0.18,
    foreground_temperature: float = 0.04,
) -> np.ndarray:
    """Score candidate samples; larger is better."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1).clip(0, 9)
    if metric == "none":
        return np.zeros(labels.shape[0], dtype=np.float64)
    cls_margin = np.zeros(labels.shape[0], dtype=np.float64)
    if classifier_metrics is not None and isinstance(classifier_metrics.get("classifier_margins"), np.ndarray):
        cls_margin = np.asarray(classifier_metrics["classifier_margins"], dtype=np.float64).reshape(-1)
    if metric == "classifier-confidence":
        if classifier_metrics is not None and isinstance(classifier_metrics.get("classifier_target_probs"), np.ndarray):
            return np.asarray(classifier_metrics["classifier_target_probs"], dtype=np.float64).reshape(-1)
        return cls_margin
    if metric not in {"composite", "composite-local", "composite-gap"}:
        raise ValueError(f"unknown sample selection metric: {metric}")
    stats = compute_shape_statistics_np(samples, grid_size=int(grid_size))
    score = float(classifier_weight) * cls_margin.copy()
    if shape_stats:
        def penalty_high(name: str, values: np.ndarray, quantile: str, scale_name: str) -> np.ndarray:
            ref = np.asarray(shape_stats[f"{name}_{quantile}"], dtype=np.float64)[labels]
            scale = np.asarray(shape_stats[f"{name}_{scale_name}"], dtype=np.float64)[labels]
            scale = np.maximum(scale, 1e-6)
            return np.maximum(0.0, values - ref) / scale
        def penalty_low(name: str, values: np.ndarray, quantile: str, scale_name: str) -> np.ndarray:
            ref = np.asarray(shape_stats[f"{name}_{quantile}"], dtype=np.float64)[labels]
            scale = np.asarray(shape_stats[f"{name}_{scale_name}"], dtype=np.float64)[labels]
            scale = np.maximum(scale, 1e-6)
            return np.maximum(0.0, ref - values) / scale
        entropy_pen = penalty_high("entropy", stats["entropy"], "q75", "iqr")
        tv_pen = penalty_low("tv", stats["tv"], "q25", "iqr")
        max_pen = penalty_low("maxmass", stats["maxmass"], "q25", "iqr")
    else:
        entropy_pen = stats["entropy"] / max(float(np.nanmedian(stats["entropy"])), 1e-6)
        tv_pen = np.maximum(0.0, np.nanmedian(stats["tv"]) - stats["tv"]) / max(float(np.nanmedian(stats["tv"])), 1e-6)
        max_pen = np.zeros_like(tv_pen)
    checker_scale = max(float(np.nanmedian(stats["checkerboard"])) * 4.0, 1e-4)
    checker_pen = stats["checkerboard"] / checker_scale
    score -= float(entropy_weight) * entropy_pen ** 2
    score -= float(tv_weight) * tv_pen ** 2
    score -= float(maxmass_weight) * max_pen ** 2
    score -= float(checkerboard_weight) * checker_pen ** 2
    if metric in {"composite-local", "composite-gap"}:
        local = local_shape_metrics_np(
            samples,
            labels,
            shape_stats,
            grid_size=int(grid_size),
            negative_space_mode=str(negative_space_mode),
            negative_space_threshold=float(negative_space_threshold),
            negative_space_temperature=float(negative_space_temperature),
            gap_threshold=float(gap_threshold),
            gap_dilate_radius=int(gap_dilate_radius),
            extra_support_margin=float(extra_support_margin),
            foreground_threshold=float(foreground_threshold),
            foreground_temperature=float(foreground_temperature),
        )
        if metric == "composite-local":
            score -= float(local_support_weight) * local["local_support_loss"]
            score -= float(local_edge_weight) * local["local_edge_loss"]
            score -= float(negative_space_weight) * local["negative_space_mass"]
        else:
            score -= float(local_support_weight) * local["missing_support_loss"]
            score -= float(extra_support_weight) * local["extra_support_loss"]
            score -= float(foreground_weight) * local["foreground_recall_loss"]
            score -= float(gap_weight) * local["gap_mass"]
            score -= float(negative_space_weight) * local["strict_negative_space_mass"]
            score -= float(local_edge_weight) * local["local_edge_loss"]
    return score


def select_generation_result_by_classifier(
    result: FluxGenerationResult,
    requested_labels: np.ndarray,
    *,
    factor: int,
    classifier: TinyMNISTClassifier,
    grid_size: int,
    device: str | torch.device,
    selection_metric: str = "classifier-confidence",
    shape_stats: dict[str, np.ndarray] | None = None,
    config: DirectFluxMNISTConfig | None = None,
    report_path: str | Path | None = None,
) -> FluxGenerationResult:
    """Select one candidate per requested label by classifier or composite quality."""
    factor = int(factor)
    if factor <= 1:
        return result
    labels = np.asarray(requested_labels, dtype=np.int64).reshape(-1)
    expected = labels.size * factor
    if result.samples.shape[0] != expected:
        raise ValueError("candidate result size does not equal len(labels) * factor")
    metrics = classifier_generation_metrics(result.samples, result.labels, classifier, grid_size=int(grid_size), device=device)
    if config is None:
        classifier_weight = 1.0
        entropy_weight = 0.5
        tv_weight = 0.5
        maxmass_weight = 0.25
        checkerboard_weight = 0.25
        local_support_weight = 0.5
        local_edge_weight = 0.25
        negative_space_weight = 0.5
        gap_weight = 0.5
        extra_support_weight = 0.25
        foreground_weight = 0.5
        neg_mode = "strict"
        neg_threshold = 0.08
        neg_temp = 0.03
        gap_threshold = 0.12
        gap_radius = 1
        extra_margin = 0.10
        fg_threshold = 0.18
        fg_temp = 0.04
    else:
        classifier_weight = float(config.selection_classifier_weight)
        entropy_weight = float(config.selection_entropy_weight)
        tv_weight = float(config.selection_tv_weight)
        maxmass_weight = float(config.selection_maxmass_weight)
        checkerboard_weight = float(config.selection_checkerboard_weight)
        local_support_weight = float(config.selection_local_support_weight)
        local_edge_weight = float(config.selection_local_edge_weight)
        negative_space_weight = float(config.selection_negative_space_weight)
        gap_weight = float(config.selection_gap_weight)
        extra_support_weight = float(config.selection_extra_support_weight)
        foreground_weight = float(config.selection_foreground_weight)
        neg_mode = str(config.terminal_negative_space_mode)
        neg_threshold = float(config.terminal_negative_space_threshold)
        neg_temp = float(config.terminal_negative_space_temperature)
        gap_threshold = float(config.terminal_gap_threshold)
        gap_radius = int(config.terminal_gap_dilate_radius)
        extra_margin = float(config.terminal_extra_support_margin)
        fg_threshold = float(config.terminal_foreground_threshold)
        fg_temp = float(config.terminal_foreground_temperature)
    scores = selection_scores_for_candidates(
        result.samples,
        result.labels,
        metric=str(selection_metric),
        classifier_metrics=metrics,
        shape_stats=shape_stats,
        grid_size=int(grid_size),
        classifier_weight=classifier_weight,
        entropy_weight=entropy_weight,
        tv_weight=tv_weight,
        maxmass_weight=maxmass_weight,
        checkerboard_weight=checkerboard_weight,
        local_support_weight=local_support_weight,
        local_edge_weight=local_edge_weight,
        negative_space_weight=negative_space_weight,
        gap_weight=gap_weight,
        extra_support_weight=extra_support_weight,
        foreground_weight=foreground_weight,
        negative_space_mode=neg_mode,
        negative_space_threshold=neg_threshold,
        negative_space_temperature=neg_temp,
        gap_threshold=gap_threshold,
        gap_dilate_radius=gap_radius,
        extra_support_margin=extra_margin,
        foreground_threshold=fg_threshold,
        foreground_temperature=fg_temp,
    )
    chosen: list[int] = []
    for i in range(labels.size):
        start = i * factor
        stop = start + factor
        local = int(np.argmax(scores[start:stop]))
        chosen.append(start + local)
    idx = np.asarray(chosen, dtype=np.int64)
    if report_path is not None:
        stats = compute_shape_statistics_np(result.samples[:expected], grid_size=int(grid_size))
        with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
            local = local_shape_metrics_np(
                result.samples[:expected],
                result.labels[:expected],
                shape_stats,
                grid_size=int(grid_size),
                negative_space_mode=neg_mode,
                negative_space_threshold=neg_threshold,
                negative_space_temperature=neg_temp,
                gap_threshold=gap_threshold,
                gap_dilate_radius=gap_radius,
                extra_support_margin=extra_margin,
            )
            writer = csv.DictWriter(handle, fieldnames=["slot", "candidate", "index", "label", "score", "classifier_prob", "classifier_margin", "entropy", "tv", "maxmass", "checkerboard", "local_support_loss", "local_edge_loss", "negative_space_mass", "strict_negative_space_mass", "gap_mass", "missing_support_loss", "extra_support_loss", "selected"])
            writer.writeheader()
            probs = np.asarray(metrics.get("classifier_target_probs", np.full(expected, np.nan)), dtype=np.float64)
            margins = np.asarray(metrics.get("classifier_margins", np.full(expected, np.nan)), dtype=np.float64)
            for slot in range(labels.size):
                for candidate in range(factor):
                    flat = slot * factor + candidate
                    writer.writerow({
                        "slot": slot,
                        "candidate": candidate,
                        "index": flat,
                        "label": int(result.labels[flat]),
                        "score": float(scores[flat]),
                        "classifier_prob": float(probs[flat]),
                        "classifier_margin": float(margins[flat]),
                        "entropy": float(stats["entropy"][flat]),
                        "tv": float(stats["tv"][flat]),
                        "maxmass": float(stats["maxmass"][flat]),
                        "checkerboard": float(stats["checkerboard"][flat]),
                        "local_support_loss": float(local["local_support_loss"][flat]),
                        "local_edge_loss": float(local["local_edge_loss"][flat]),
                        "negative_space_mass": float(local["negative_space_mass"][flat]),
                        "selected": int(flat == idx[slot]),
                    })
    def maybe_take(arr):
        return None if arr is None else np.asarray(arr)[idx]
    selected_samples = np.asarray(result.samples)[idx]
    selected_sources = maybe_take(result.sources)
    n = int(round(math.sqrt(selected_samples.shape[1])))
    shape = compute_shape_statistics_np(selected_samples, grid_size=n)
    return FluxGenerationResult(
        samples=selected_samples,
        labels=np.asarray(result.labels)[idx],
        trajectory=None,
        clipping_fraction=float(result.clipping_fraction),
        sources=selected_sources,
        source_indices=maybe_take(result.source_indices),
        source_labels=maybe_take(result.source_labels),
        source_unique_count=None,
        source_diversity_l2=None,
        source_pair_l2=None,
        source_label_match_rate=None,
        learned_step_rms=result.learned_step_rms,
        free_step_rms=result.free_step_rms,
        noise_step_rms=result.noise_step_rms,
        free_to_learned_ratio=result.free_to_learned_ratio,
        noise_to_learned_ratio=result.noise_to_learned_ratio,
        sample_entropy=float(np.mean(shape["entropy"])),
        sample_total_variation=float(np.mean(shape["tv"])),
        sample_checkerboard_energy=float(np.mean(shape["checkerboard"] ** 2)),
        sample_highfreq_fraction=None,
    )


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


@torch.no_grad()
def _finite_quantile_torch(values: Tensor, q: float) -> float:
    flat = values.detach().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return 0.0
    q_clamped = min(1.0, max(0.0, float(q)))
    if finite.numel() == 1:
        return float(finite[0].cpu())
    try:
        return float(torch.quantile(finite.float(), q_clamped).cpu())
    except Exception:  # pragma: no cover - compatibility with older torch builds.
        sorted_vals = torch.sort(finite.float()).values
        idx = int(round(q_clamped * float(sorted_vals.numel() - 1)))
        return float(sorted_vals[idx].cpu())


@torch.no_grad()
def reference_step_substep_diagnostics_torch(
    states: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    free_weight: float | None = None,
    noise_weight: float | None = None,
    substeps: int = 1,
    stiffness_fraction: float | None = None,
    quantile: float = 0.99,
) -> dict[str, float]:
    """Estimate explicit-Euler proposal size relative to directional budgets.

    The drift ratio is sign-aware: positive edge flux consumes tail mass and
    negative edge flux consumes head mass.  The stochastic ratio is conservative
    and uses the smaller endpoint budget because a Gaussian increment may have
    either sign.  These diagnostics are used by Phase 0.7 to pick a fixed
    substep count before calling the boundary-aware limiter.
    """

    if states.ndim != 2:
        raise ValueError("states must have shape (B, N)")
    if dt < 0.0 or not math.isfinite(float(dt)):
        raise ValueError("dt must be non-negative and finite")
    if int(substeps) <= 0:
        raise ValueError("substeps must be positive")
    n = int(config.grid_size)
    if states.shape[1] != n * n:
        raise ValueError("states have the wrong number of pixels")
    free_w = float(config.free_weight if free_weight is None else free_weight)
    noise_w = float(config.noise_weight if noise_weight is None else noise_weight)
    stiff_c = float(config.limiter_fraction if stiffness_fraction is None else stiffness_fraction)
    if not (0.0 < stiff_c <= 1.0 and math.isfinite(stiff_c)):
        raise ValueError("stiffness_fraction must be finite and in (0, 1]")

    sub_dt = float(dt) / float(int(substeps))
    inv_h2 = float(n * n)
    alpha = edge_alpha_value(config)
    tiny = float(config.mass_floor)
    drift_ratios: list[Tensor] = []
    noise_ratios: list[Tensor] = []
    endpoint_masses: list[Tensor] = []

    for edge_class in _edge_classes_torch(n, states.device):
        tails = edge_class.tails
        heads = edge_class.heads
        a0 = states[:, tails]
        b0 = states[:, heads]
        denom = a0 + b0
        harmonic = torch.where(denom > tiny, a0 * b0 / denom.clamp_min(tiny), torch.zeros_like(denom))
        ratio = torch.where(denom > tiny, (a0 - b0) / denom.clamp_min(tiny), torch.zeros_like(denom))
        theta = ((2.0 * alpha + 1.0) / alpha) * harmonic
        free_flux = (2.0 * alpha + 1.0) * inv_h2 * ratio
        drift_delta = free_w * free_flux * sub_dt
        noise_std = noise_w * torch.sqrt((2.0 * theta * inv_h2 * sub_dt).clamp_min(0.0))
        tail_budget = (stiff_c * (a0 - tiny).clamp_min(0.0)).clamp_min(0.0)
        head_budget = (stiff_c * (b0 - tiny).clamp_min(0.0)).clamp_min(0.0)
        drift_budget = torch.where(drift_delta >= 0.0, tail_budget, head_budget).clamp_min(tiny)
        noise_budget = torch.minimum(tail_budget, head_budget).clamp_min(tiny)
        drift_ratios.append((drift_delta.abs() / drift_budget).reshape(-1))
        noise_ratios.append((noise_std / noise_budget).reshape(-1))
        endpoint_masses.append(torch.minimum(a0, b0).reshape(-1))

    drift_all = torch.cat(drift_ratios) if drift_ratios else states.new_zeros(1)
    noise_all = torch.cat(noise_ratios) if noise_ratios else states.new_zeros(1)
    endpoint_all = torch.cat(endpoint_masses) if endpoint_masses else states.new_zeros(1)
    q = min(1.0, max(0.0, float(quantile)))
    return {
        "dt": float(dt),
        "sub_dt": float(sub_dt),
        "diagnostic_substeps": float(int(substeps)),
        "ratio_quantile": float(q),
        "drift_ratio_q": _finite_quantile_torch(drift_all, q),
        "drift_ratio_max": _finite_quantile_torch(drift_all, 1.0),
        "noise_ratio_q": _finite_quantile_torch(noise_all, q),
        "noise_ratio_max": _finite_quantile_torch(noise_all, 1.0),
        "min_endpoint_mass_q01": _finite_quantile_torch(endpoint_all, 0.01),
        "min_endpoint_mass_q50": _finite_quantile_torch(endpoint_all, 0.50),
    }


@torch.no_grad()
def choose_reference_substeps_torch(
    states: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    free_weight: float | None = None,
    noise_weight: float | None = None,
    base_substeps: int = 1,
    max_substeps: int = 512,
    target_drift_ratio: float = 0.25,
    target_noise_ratio: float = 0.25,
    stiffness_fraction: float | None = None,
    quantile: float = 0.99,
) -> tuple[int, dict[str, float]]:
    """Choose substeps from one-substep drift/noise budget ratios.

    Drift increments scale like ``1 / substeps`` and one-sigma stochastic
    increments scale like ``1 / sqrt(substeps)``.  The diagnostics keep the
    unclipped requirement so callers can see when the configured cap is
    dominating the dynamics.
    """

    base = max(1, int(base_substeps))
    cap = max(base, int(max_substeps))
    diag = reference_step_substep_diagnostics_torch(
        states,
        dt,
        config,
        free_weight=free_weight,
        noise_weight=noise_weight,
        substeps=1,
        stiffness_fraction=stiffness_fraction,
        quantile=quantile,
    )
    drift_target = max(float(target_drift_ratio), 1e-12)
    noise_target = max(float(target_noise_ratio), 1e-12)
    drift_q = float(diag["drift_ratio_q"])
    noise_q = float(diag["noise_ratio_q"])
    drift_required = 1 if drift_q <= drift_target else int(math.ceil(drift_q / drift_target))
    noise_required = 1 if noise_q <= noise_target else int(math.ceil((noise_q / noise_target) ** 2))
    required_unclipped = max(base, drift_required, noise_required)
    chosen = min(cap, required_unclipped)
    diag.update(
        {
            "base_substeps": float(base),
            "max_substeps": float(cap),
            "target_drift_ratio": float(drift_target),
            "target_noise_ratio": float(noise_target),
            "drift_required_substeps": float(drift_required),
            "noise_required_substeps": float(noise_required),
            "required_substeps_unclipped": float(required_unclipped),
            "chosen_substeps": float(chosen),
            "hit_substep_cap": float(required_unclipped > cap),
        }
    )
    return int(chosen), diag


@dataclass(frozen=True)
class MaskedReferenceStepResult:
    """Result of the boundary-correct free reference integrator.

    ``raw_innovations`` and ``valid_edge_mask`` are only populated when
    ``return_innovations=True``.  They have shape ``(substeps, B, 2, H, W)``.
    A true mask value means the raw Gaussian innovation on that edge/substep was
    used without limiter modification.  A false value means the deterministic or
    stochastic transfer touched the directional limiter and should be excluded
    from innovation-regression losses.

    The historical ``masked_*`` names are retained for dashboards.  In this
    implementation they count limiter-touched edges, not frozen/no-op edges: the
    integrator advances the feasible part of the transfer instead of dropping the
    whole edge.  The weighted masked fractions down-weight limited edges carrying
    negligible harmonic mobility or Brownian noise energy, which is the more
    relevant health signal for D0 innovation regression.
    """

    states: Tensor
    raw_innovations: Tensor | None
    valid_edge_mask: Tensor | None
    masked_edges: int
    proposed_edges: int
    substeps: int
    masked_fraction: float
    noise_stiff_edges: int
    drift_stiff_edges: int
    overflow_edges: int
    limited_edges: int = 0
    drift_limited_edges: int = 0
    noise_limited_edges: int = 0
    nonfinite_edges: int = 0
    floor_touched_pixels: int = 0
    floor_correction_l1: float = 0.0
    renorm_correction_l1: float = 0.0
    mobility_weight_sum: float = 0.0
    limited_mobility_weight_sum: float = 0.0
    noise_energy_sum: float = 0.0
    limited_noise_energy_sum: float = 0.0
    mobility_weighted_masked_fraction: float = 0.0
    noise_energy_weighted_masked_fraction: float = 0.0
    valid_innovation_mobility_fraction: float = 1.0
    valid_innovation_noise_energy_fraction: float = 1.0
    substep_states: Tensor | None = None


def masked_reference_free_step_torch(
    states: Tensor,
    dt: float,
    config: DirectFluxMNISTConfig,
    *,
    free_weight: float | None = None,
    noise_weight: float | None = None,
    substeps: int = 1,
    stiffness_fraction: float | None = None,
    deterministic: bool = False,
    return_innovations: bool = False,
    return_substep_states: bool = False,
) -> MaskedReferenceStepResult:
    """Boundary-correct free reference step with direction-aware limiting.

    The old P0.5 rule rejected an edge whenever ``abs(delta)`` or even the
    one-sigma noise scale was large relative to ``min(a, b)``.  That makes a
    zero or nearly-zero endpoint behave like an absorbing wall: the deterministic
    inward Dirichlet drift on a one-zero edge is also rejected because the
    threshold is zero.

    This version computes all coefficients from the beginning of each substep
    and uses per-node directional mass budgets.  A positive oriented edge
    transfer consumes tail mass; a negative transfer consumes head mass.  Mass
    that arrives during the current substep is not made available for another
    outgoing transfer until the next substep, avoiding edge-color cascade
    artifacts at zero pixels.

    Raw Gaussian innovations are stored before any limiter modification, and
    limiter-touched edges are masked for D0 innovation regression.
    """

    if states.ndim != 2:
        raise ValueError("states must have shape (B, N)")
    if dt < 0.0 or not math.isfinite(float(dt)):
        raise ValueError("dt must be non-negative and finite")
    if int(substeps) <= 0:
        raise ValueError("substeps must be positive")
    if dt == 0.0:
        raw = None
        mask = None
        sub_states = None
        if return_innovations:
            n0 = int(config.grid_size)
            raw = torch.empty((0, states.shape[0], 2, n0, n0), device=states.device, dtype=states.dtype)
            mask = torch.empty((0, states.shape[0], 2, n0, n0), device=states.device, dtype=torch.bool)
        if return_substep_states:
            n0 = int(config.grid_size)
            sub_states = torch.empty((0, states.shape[0], n0 * n0), device=states.device, dtype=states.dtype)
        return MaskedReferenceStepResult(states.clone(), raw, mask, 0, 0, int(substeps), 0.0, 0, 0, 0, substep_states=sub_states)

    n = int(config.grid_size)
    if states.shape[1] != n * n:
        raise ValueError("states have the wrong number of pixels")
    free_w = float(config.free_weight if free_weight is None else free_weight)
    noise_w = float(config.noise_weight if noise_weight is None else noise_weight)
    stiff_c = float(config.limiter_fraction if stiffness_fraction is None else stiffness_fraction)
    if not (0.0 < stiff_c <= 1.0 and math.isfinite(stiff_c)):
        raise ValueError("stiffness_fraction must be finite and in (0, 1]")
    substeps_i = int(substeps)
    sub_dt = float(dt) / float(substeps_i)
    inv_h2 = float(n * n)
    alpha = edge_alpha_value(config)
    tiny = float(config.mass_floor)
    out = states.clone()
    raw_chunks: list[Tensor] = []
    mask_chunks: list[Tensor] = []
    substep_state_chunks: list[Tensor] = []
    masked_edges = 0
    proposed_edges = 0
    drift_limited_edges = 0
    noise_limited_edges = 0
    nonfinite_edges = 0
    floor_touched_pixels = 0
    floor_correction_l1 = 0.0
    renorm_correction_l1 = 0.0
    mobility_weight_sum = 0.0
    limited_mobility_weight_sum = 0.0
    noise_energy_sum = 0.0
    limited_noise_energy_sum = 0.0

    def _budget_limiter(delta: Tensor, tail_budget: Tensor, head_budget: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        upper = tail_budget.clamp_min(0.0)
        lower = -head_budget.clamp_min(0.0)
        finite = torch.isfinite(delta)
        safe = torch.where(finite, delta, torch.zeros_like(delta))
        clipped = torch.minimum(torch.maximum(safe, lower), upper)
        limited = (~finite) | (clipped != safe)
        return clipped, limited, ~finite

    for _sub in range(substeps_i):
        base = out
        state_delta = torch.zeros_like(base)
        # Directional consumption budget from the start-of-substep state.  This
        # keeps exact zero endpoints viable while preventing mass received during
        # this substep from immediately cascading through other edges.
        remaining = float(stiff_c) * (base - float(tiny)).clamp_min(0.0)

        if return_innovations:
            raw_flat = torch.zeros((base.shape[0], 2 * n * n), device=base.device, dtype=base.dtype)
            mask_flat = torch.ones((base.shape[0], 2 * n * n), device=base.device, dtype=torch.bool)
        else:
            raw_flat = None
            mask_flat = None

        for edge_class in _edge_classes_torch(n, base.device):
            tails = edge_class.tails
            heads = edge_class.heads
            a0 = base[:, tails]
            b0 = base[:, heads]
            denom = a0 + b0
            harmonic = torch.where(denom > tiny, a0 * b0 / denom.clamp_min(tiny), torch.zeros_like(denom))
            ratio = torch.where(denom > tiny, (a0 - b0) / denom.clamp_min(tiny), torch.zeros_like(denom))
            theta = ((2.0 * alpha + 1.0) / alpha) * harmonic
            free_flux = (2.0 * alpha + 1.0) * inv_h2 * ratio
            drift_delta = free_w * free_flux * sub_dt
            noise_std = noise_w * torch.sqrt((2.0 * theta * inv_h2 * sub_dt).clamp_min(0.0))
            xi = torch.zeros_like(noise_std) if deterministic or noise_w <= 0.0 else torch.randn_like(noise_std)

            drift_flux, drift_limited, drift_nonfinite = _budget_limiter(
                drift_delta,
                remaining[:, tails],
                remaining[:, heads],
            )
            drift_tail_consume = drift_flux.clamp_min(0.0)
            drift_head_consume = (-drift_flux).clamp_min(0.0)
            remaining[:, tails] = (remaining[:, tails] - drift_tail_consume).clamp_min(0.0)
            remaining[:, heads] = (remaining[:, heads] - drift_head_consume).clamp_min(0.0)
            state_delta[:, tails] = state_delta[:, tails] - drift_flux
            state_delta[:, heads] = state_delta[:, heads] + drift_flux

            noise_delta = noise_std * xi
            noise_flux, noise_limited, noise_nonfinite = _budget_limiter(
                noise_delta,
                remaining[:, tails],
                remaining[:, heads],
            )
            noise_tail_consume = noise_flux.clamp_min(0.0)
            noise_head_consume = (-noise_flux).clamp_min(0.0)
            remaining[:, tails] = (remaining[:, tails] - noise_tail_consume).clamp_min(0.0)
            remaining[:, heads] = (remaining[:, heads] - noise_head_consume).clamp_min(0.0)
            state_delta[:, tails] = state_delta[:, tails] - noise_flux
            state_delta[:, heads] = state_delta[:, heads] + noise_flux

            invalid_for_regression = drift_limited | noise_limited | drift_nonfinite | noise_nonfinite
            # P0.8: raw edge counts overstate D0 damage when the limited edge
            # carries negligible mobility/noise.  Accumulate signal-weighted
            # limiter fractions using tensors already computed for the step; no
            # extra diagnostic pass is required.
            theta_weight = theta.detach().clamp_min(0.0)
            noise_energy_weight = noise_std.detach().square().clamp_min(0.0)
            mobility_weight_sum += float(theta_weight.sum().detach().cpu())
            limited_mobility_weight_sum += float(theta_weight.masked_select(invalid_for_regression).sum().detach().cpu())
            noise_energy_sum += float(noise_energy_weight.sum().detach().cpu())
            limited_noise_energy_sum += float(noise_energy_weight.masked_select(invalid_for_regression).sum().detach().cpu())
            proposed_edges += int(invalid_for_regression.numel())
            drift_limited_edges += int(drift_limited.count_nonzero().detach().cpu())
            noise_limited_edges += int(noise_limited.count_nonzero().detach().cpu())
            nonfinite_edges += int((drift_nonfinite | noise_nonfinite).count_nonzero().detach().cpu())
            masked_edges += int(invalid_for_regression.count_nonzero().detach().cpu())

            if return_innovations and raw_flat is not None and mask_flat is not None:
                raw_flat[:, edge_class.flux_indices] = xi
                mask_flat[:, edge_class.flux_indices] = ~invalid_for_regression

        out = base + state_delta
        before_floor = out
        # Do not inject a positive floor into true zero pixels.  The finite-volume
        # boundary rule is viable with exact zeros; this repair only removes
        # negative roundoff/overshoot if the directional budgets were touched by
        # floating-point error.
        floored = before_floor.clamp_min(0.0)
        floor_touched_pixels += int((before_floor < 0.0).count_nonzero().detach().cpu())
        floor_correction_l1 += float((floored - before_floor).abs().sum().detach().cpu())
        before_renorm = floored
        sums = before_renorm.sum(dim=1, keepdim=True).clamp_min(tiny)
        out = before_renorm / sums
        renorm_correction_l1 += float((out - before_renorm).abs().sum().detach().cpu())
        if return_substep_states:
            substep_state_chunks.append(out.clone())
        if return_innovations and raw_flat is not None and mask_flat is not None:
            raw_chunks.append(raw_flat.reshape(base.shape[0], 2, n, n))
            mask_chunks.append(mask_flat.reshape(base.shape[0], 2, n, n))

    raw_tensor = torch.stack(raw_chunks, dim=0) if raw_chunks else None
    mask_tensor = torch.stack(mask_chunks, dim=0) if mask_chunks else None
    substep_state_tensor = torch.stack(substep_state_chunks, dim=0) if substep_state_chunks else None
    masked_fraction = 0.0 if proposed_edges == 0 else float(masked_edges) / float(proposed_edges)
    mobility_masked_fraction = 0.0 if mobility_weight_sum <= 0.0 else float(limited_mobility_weight_sum) / float(mobility_weight_sum)
    noise_energy_masked_fraction = 0.0 if noise_energy_sum <= 0.0 else float(limited_noise_energy_sum) / float(noise_energy_sum)
    return MaskedReferenceStepResult(
        states=out,
        raw_innovations=raw_tensor,
        valid_edge_mask=mask_tensor,
        masked_edges=int(masked_edges),
        proposed_edges=int(proposed_edges),
        substeps=int(substeps_i),
        masked_fraction=float(masked_fraction),
        noise_stiff_edges=int(noise_limited_edges),
        drift_stiff_edges=int(drift_limited_edges),
        overflow_edges=int(masked_edges),
        limited_edges=int(masked_edges),
        drift_limited_edges=int(drift_limited_edges),
        noise_limited_edges=int(noise_limited_edges),
        nonfinite_edges=int(nonfinite_edges),
        floor_touched_pixels=int(floor_touched_pixels),
        floor_correction_l1=float(floor_correction_l1),
        renorm_correction_l1=float(renorm_correction_l1),
        mobility_weight_sum=float(mobility_weight_sum),
        limited_mobility_weight_sum=float(limited_mobility_weight_sum),
        noise_energy_sum=float(noise_energy_sum),
        limited_noise_energy_sum=float(limited_noise_energy_sum),
        mobility_weighted_masked_fraction=float(mobility_masked_fraction),
        noise_energy_weighted_masked_fraction=float(noise_energy_masked_fraction),
        valid_innovation_mobility_fraction=float(1.0 - mobility_masked_fraction),
        valid_innovation_noise_energy_fraction=float(1.0 - noise_energy_masked_fraction),
        substep_states=substep_state_tensor,
    )


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


def _autocast_disabled_for(device_type: str):
    """Return an autocast-disabled context for FFT/spectral linear algebra.

    CUDA autocast can cast tensors entering ``torch.fft`` to ``float16``.  cuFFT
    only supports half-precision FFTs for power-of-two signal sizes, so the
    28x28 MNIST Poisson projection must opt out of autocast and run in at least
    float32.  The helper is intentionally small and conservative so it works
    across PyTorch versions.
    """
    amp_mod = getattr(torch, "amp", None)
    autocast = getattr(amp_mod, "autocast", None)
    if autocast is not None:
        try:
            return autocast(device_type=device_type, enabled=False)
        except TypeError:
            try:
                return autocast(device_type, enabled=False)
            except TypeError:
                pass
    if device_type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


_POISSON_DENOM_CACHE: dict[tuple[int, int, str, torch.dtype], Tensor] = {}


def _poisson_denom_cached(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Cached spectral denominator for repeated periodic Poisson solves."""
    key = (int(h), int(w), str(device), dtype)
    cached = _POISSON_DENOM_CACHE.get(key)
    if cached is not None:
        return cached
    ky = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    kx = torch.arange(w // 2 + 1, device=device, dtype=dtype).view(1, w // 2 + 1)
    two_pi = 2.0 * math.pi
    denom = 2.0 * torch.cos(two_pi * kx / float(w)) - 2.0
    denom = denom + 2.0 * torch.cos(two_pi * ky / float(h)) - 2.0
    denom = denom.clamp(max=-1e-12)
    _POISSON_DENOM_CACHE[key] = denom
    return denom


def poisson_flux_from_velocity_torch(velocity: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Return minimum-energy periodic edge flux with ``div flux = velocity``.

    ``velocity`` may have shape ``(B, H * W)`` or ``(B, H, W)`` and must have
    approximately zero spatial mean.  The zero mode is removed explicitly.

    The spectral solve is always performed in at least float32 with autocast
    disabled.  This is required on CUDA for the 28x28 MNIST grid: cuFFT cannot
    compute half-precision FFTs on non-power-of-two sizes.  Half/bfloat16 inputs
    therefore return a float32 projected flux.
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

    if v.dtype == torch.float64:
        compute_dtype = torch.float64
        return_dtype = torch.float64
    else:
        compute_dtype = torch.float32
        return_dtype = torch.float32

    with _autocast_disabled_for(v.device.type):
        v_work = v.to(dtype=compute_dtype)
        v_work = v_work - v_work.mean(dim=(-2, -1), keepdim=True)
        v_hat = torch.fft.rfft2(v_work)
        safe_denom = _poisson_denom_cached(h, w, device=v_work.device, dtype=compute_dtype)
        psi_hat = torch.zeros_like(v_hat)
        psi_hat[..., 0, 0] = 0.0
        psi_hat[..., 1:, :] = v_hat[..., 1:, :] / safe_denom[1:, :]
        if w // 2 + 1 > 1:
            psi_hat[..., 0, 1:] = v_hat[..., 0, 1:] / safe_denom[0, 1:]
        psi = torch.fft.irfft2(psi_hat, s=(h, w))
        fx = psi - torch.roll(psi, shifts=-1, dims=-1)
        fy = psi - torch.roll(psi, shifts=-1, dims=-2)
        flux = torch.stack([fx, fy], dim=1)
    return flux.to(dtype=return_dtype)




def apply_flux_parameterization_torch(flux: Tensor, masses: Tensor | None, config: DirectFluxMNISTConfig) -> Tensor:
    """Apply the configured edge-flux structure.

    ``edge`` returns the raw two-channel field.  ``projected`` keeps the same
    conservative node velocity but replaces the field by the minimum-energy
    periodic flux.  This removes arbitrary curl/gauge components that can
    interact with noise/edge limiting and show up as checkerboard texture.
    """
    if str(config.flux_parameterization) == "edge":
        return flux
    if str(config.flux_parameterization) == "projected":
        return poisson_flux_from_velocity_torch(flux_divergence_torch(flux), grid_size=int(config.grid_size))
    raise ValueError(f"unknown flux_parameterization: {config.flux_parameterization}")


def flux_curl_torch(flux: Tensor) -> Tensor:
    """Periodic discrete curl of an edge flux field."""
    if flux.ndim != 4 or flux.shape[1] != 2:
        raise ValueError("flux must have shape (B, 2, H, W)")
    fx = flux[:, 0]
    fy = flux[:, 1]
    return torch.roll(fy, shifts=-1, dims=-1) - fy - (torch.roll(fx, shifts=-1, dims=-2) - fx)


def edge_laplacian_energy_torch(flux: Tensor) -> Tensor:
    """Small smoothness penalty on edge flux channels."""
    lap = (
        torch.roll(flux, shifts=1, dims=-1)
        + torch.roll(flux, shifts=-1, dims=-1)
        + torch.roll(flux, shifts=1, dims=-2)
        + torch.roll(flux, shifts=-1, dims=-2)
        - 4.0 * flux
    )
    return lap.square().mean()


def image_total_variation(states: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Mean anisotropic total variation of mass images."""
    if states.ndim == 2:
        n = int(grid_size or round(math.sqrt(int(states.shape[1]))))
        img = states.reshape(-1, n, n)
    elif states.ndim == 3:
        img = states
    else:
        raise ValueError("states must have shape (B,N) or (B,H,W)")
    dx = torch.roll(img, shifts=-1, dims=-1) - img
    dy = torch.roll(img, shifts=-1, dims=-2) - img
    return (dx.abs() + dy.abs()).flatten(1).sum(dim=1).mean()


def image_total_variation_per_sample(states: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Per-sample anisotropic total variation of mass/image tensors."""
    if states.ndim == 2:
        n = int(grid_size or round(math.sqrt(int(states.shape[1]))))
        img = states.reshape(-1, n, n)
    elif states.ndim == 3:
        img = states
    elif states.ndim == 4 and states.shape[1] == 1:
        img = states[:, 0]
    else:
        raise ValueError("states must have shape (B,N), (B,H,W), or (B,1,H,W)")
    dx = torch.roll(img, shifts=-1, dims=-1) - img
    dy = torch.roll(img, shifts=-1, dims=-2) - img
    return (dx.abs() + dy.abs()).flatten(1).sum(dim=1)


def _masked_mean_torch(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over samples selected by a fixed boolean/float mask; zero if empty."""
    values = values.reshape(-1)
    weights = mask.to(dtype=values.dtype, device=values.device).reshape(-1)
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0.0:
        return values.new_tensor(0.0)
    return (values * weights).sum() / denom.clamp_min(torch.finfo(values.dtype).eps)


def _safe_terminal_loss_scale(step_index: int | None, config: DirectFluxMNISTConfig, ref: Tensor) -> Tensor:
    ramp_steps = int(config.terminal_loss_ramp_steps)
    if ramp_steps <= 0 or step_index is None:
        scale = 1.0
    else:
        scale = min(1.0, max(0.0, float(step_index) / float(max(ramp_steps, 1))))
    return ref.new_tensor(scale)


def binary_cross_entropy_probs_per_sample_autocast_safe(input_prob: Tensor, target_prob: Tensor) -> Tensor:
    """Autocast-safe per-sample BCE for probability tensors."""
    with _autocast_disabled_for(input_prob.device.type):
        input32 = input_prob.float().clamp(1e-5, 1.0 - 1e-5)
        target32 = target_prob.float().clamp(0.0, 1.0)
        loss = -(target32 * torch.log(input32) + (1.0 - target32) * torch.log1p(-input32))
        return loss.flatten(1).mean(dim=1)


def _classifier_input_from_masses_for_loss(masses: Tensor, grid_size: int, *, blur_sigma: float) -> Tensor:
    """Classifier input for generator loss; optionally blurred to avoid grid exploits."""
    n = int(grid_size)
    if masses.ndim == 2:
        img = masses.reshape(masses.shape[0], 1, n, n)
    elif masses.ndim == 3:
        img = masses[:, None, :, :]
    elif masses.ndim == 4 and masses.shape[1] == 1:
        img = masses
    else:
        raise ValueError("masses must have shape (B,N), (B,H,W), or (B,1,H,W)")
    if float(blur_sigma) > 0.0:
        with _autocast_disabled_for(img.device.type):
            img = _periodic_gaussian_blur_torch(img.float(), sigma=float(blur_sigma)).to(dtype=masses.dtype)
    return img * float(n * n)


def image_gradient_mse_torch(a: Tensor, b: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Finite-difference image-gradient MSE for sharpening losses."""
    if a.ndim == 2:
        n = int(grid_size or round(math.sqrt(int(a.shape[1]))))
        ai = a.reshape(-1, n, n)
        bi = b.reshape(-1, n, n)
    else:
        ai = a
        bi = b
    adx = torch.roll(ai, shifts=-1, dims=-1) - ai
    ady = torch.roll(ai, shifts=-1, dims=-2) - ai
    bdx = torch.roll(bi, shifts=-1, dims=-1) - bi
    bdy = torch.roll(bi, shifts=-1, dims=-2) - bi
    return F.mse_loss(adx, bdx) + F.mse_loss(ady, bdy)


def binary_cross_entropy_probs_autocast_safe(input_prob: Tensor, target_prob: Tensor) -> Tensor:
    """Autocast-safe BCE for probability tensors.

    ``torch.nn.functional.binary_cross_entropy`` intentionally raises under
    CUDA autocast.  This helper keeps the endpoint BCE probability-space loss
    but evaluates the scalar expression manually in float32 with autocast
    disabled, so mixed-precision training can still be used for the expensive
    network and rollout computations.
    """
    with _autocast_disabled_for(input_prob.device.type):
        input32 = input_prob.float().clamp(1e-5, 1.0 - 1e-5)
        target32 = target_prob.float().clamp(0.0, 1.0)
        loss = -(target32 * torch.log(input32) + (1.0 - target32) * torch.log1p(-input32)).mean()
    return loss


def checkerboard_energy_torch(states: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Energy at the alternating parity pattern; high values flag checkerboard artifacts."""
    if states.ndim == 2:
        n = int(grid_size or round(math.sqrt(int(states.shape[1]))))
        img = states.reshape(-1, n, n)
    elif states.ndim == 3:
        img = states
        n = int(img.shape[-1])
    else:
        raise ValueError("states must have shape (B,N) or (B,H,W)")
    rows = torch.arange(n, device=img.device).view(n, 1)
    cols = torch.arange(n, device=img.device).view(1, n)
    pattern = ((rows + cols) % 2).to(dtype=img.dtype) * 2.0 - 1.0
    coeff = (img * pattern).flatten(1).sum(dim=1)
    return coeff.square().mean()


def highfreq_fraction_torch(states: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Fraction of FFT power outside the low-frequency central band."""
    if states.ndim == 2:
        n = int(grid_size or round(math.sqrt(int(states.shape[1]))))
        img = states.reshape(-1, n, n)
    else:
        img = states
        n = int(img.shape[-1])
    fft = torch.fft.fft2(img)
    power = fft.real.square() + fft.imag.square()
    ky = torch.fft.fftfreq(n, device=img.device).abs().view(n, 1)
    kx = torch.fft.fftfreq(n, device=img.device).abs().view(1, n)
    high = (kx > 0.25) | (ky > 0.25)
    return power[:, high].sum(dim=1).div(power.flatten(1).sum(dim=1).clamp_min(1e-30)).mean()


def project_edge_flux_torch(flux: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Project an arbitrary edge flux to the minimum-energy flux with the same divergence."""
    return poisson_flux_from_velocity_torch(flux_divergence_torch(flux), grid_size=grid_size or int(flux.shape[-1]))


def sample_total_variation_torch(states: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Alias used by diagnostics/tests for image total variation."""
    return image_total_variation(states, grid_size=grid_size)


def sample_checkerboard_energy_torch(states: Tensor, *, grid_size: int | None = None) -> Tensor:
    """Alias used by diagnostics/tests for alternating-pixel energy."""
    return checkerboard_energy_torch(states, grid_size=grid_size)

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
        step_index=None if step_index is None else int(step_index),
        train_free_weight=float(train_free_weight),
        train_noise_weight=float(train_noise_weight),
    )


def _with_terminal_tau_window(batch: FluxTrainingBatch, config: DirectFluxMNISTConfig, *, rng: np.random.Generator) -> FluxTrainingBatch:
    """Move a sampled bridge batch into a configurable near-terminal tau window."""
    horizon = max(float(natural_horizon(config)), 1e-12)
    lo = float(config.terminal_tau_min_fraction)
    hi = float(config.terminal_tau_max_fraction)
    if hi < lo:
        lo, hi = hi, lo
    u = rng.uniform(lo, hi, size=int(batch.states.shape[0])).astype(np.float32)
    tau = torch.as_tensor(u, dtype=batch.states.dtype, device=batch.states.device) * float(horizon)
    mix = (tau / horizon).pow(float(config.bridge_power)).view(-1, 1)
    states = (1.0 - mix) * batch.targets + mix * batch.sources
    states = _renormalize_masses(states, floor=float(config.mass_floor))
    return FluxTrainingBatch(
        tau=tau,
        states=states,
        labels=batch.labels,
        targets=batch.targets,
        sources=batch.sources,
        source_indices=batch.source_indices,
        source_labels=batch.source_labels,
        target_indices=batch.target_indices,
        # Keep the primary flux teacher on the global velocity target.  The
        # 10r hotfix found that forcing all near-terminal microbatches to
        # safe-residual made the learned correction shrink and let free/noise
        # dominate.  Dedicated terminal losses still see the near-terminal
        # states through the ordinary masking/gating path.
        target_velocity_mode=batch.target_velocity_mode,
        step_index=batch.step_index,
        train_free_weight=batch.train_free_weight,
        train_noise_weight=batch.train_noise_weight,
        is_terminal_batch=True,
    )


def sample_terminal_flux_training_batch(
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
    """Sample a dedicated near-terminal training microbatch.

    The pair/source assignment is the ordinary teacher assignment, but tau is
    forced into ``[terminal_tau_min_fraction, terminal_tau_max_fraction]``.
    Optional hard-label sampling restricts this microbatch to recurrent failure
    classes without biasing the entire training stream.
    """
    rng = np.random.default_rng() if rng is None else rng
    images_arr = np.asarray(images, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    chosen_images = images_arr
    chosen_labels = labels_arr
    local_ot_cache = ot_cache
    if bool(config.hard_label_sampling) and float(config.hard_label_prob) > 0.0 and rng.random() < float(config.hard_label_prob):
        hard = np.asarray(tuple(int(x) for x in config.hard_labels), dtype=np.int64)
        mask = np.isin(labels_arr, hard)
        if bool(mask.any()):
            chosen_images = images_arr[mask]
            chosen_labels = labels_arr[mask]
            local_ot_cache = None
    batch = sample_flux_training_batch(
        chosen_images,
        chosen_labels,
        config,
        batch_size=int(batch_size),
        device=device,
        rng=rng,
        dtype=dtype,
        class_means=None if chosen_images is not images_arr else class_means,
        ot_cache=local_ot_cache,
        step_index=step_index,
    )
    return _with_terminal_tau_window(batch, config, rng=rng)


def training_target_velocity_torch(batch: FluxTrainingBatch, config: DirectFluxMNISTConfig, mode: str | None = None) -> Tensor:
    """Node velocity teacher used before the Poisson edge-flux solve."""
    horizon = max(natural_horizon(config), 1e-12)
    velocity_mode = mode or batch.target_velocity_mode or config.velocity_target
    constant_velocity = (batch.targets - batch.sources) / float(horizon)
    remaining = batch.tau.clamp_min(float(config.min_tau_fraction) * float(horizon)).view(-1, 1)
    residual_velocity = (batch.targets - batch.states) / remaining
    if velocity_mode == "constant":
        velocity = constant_velocity
    elif velocity_mode == "residual":
        velocity = residual_velocity
    elif velocity_mode == "safe-residual":
        const_rms = constant_velocity.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        resid_rms = residual_velocity.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        max_rms = float(config.on_policy_residual_max_ratio) * const_rms
        scale = torch.minimum(torch.ones_like(resid_rms), max_rms / resid_rms)
        velocity = residual_velocity * scale
    elif velocity_mode == "mixed":
        tau_fraction = batch.tau / float(horizon)
        late_mask = tau_fraction <= float(config.late_residual_fraction)
        if float(config.late_residual_prob) < 1.0:
            # Deterministic pseudo-random gate so the total teacher flux is
            # identical for free-aware and non-free-aware configs with the same
            # batch.  This keeps regression tests and paired losses stable while
            # still mixing constant and residual late targets across examples.
            seed_feature = (batch.tau / float(horizon)) * 12.9898 + batch.states[:, 0] * 78.233
            gate = torch.frac(torch.sin(seed_feature) * 43758.5453).abs()
            late_mask = late_mask & (gate < float(config.late_residual_prob))
        velocity = torch.where(late_mask.view(-1, 1), residual_velocity, constant_velocity)
    else:
        raise ValueError(f"unknown velocity target mode: {velocity_mode}")
    return velocity - velocity.mean(dim=1, keepdim=True)


def training_target_flux_torch(batch: FluxTrainingBatch, config: DirectFluxMNISTConfig) -> Tensor:
    """Return the physical two-channel target flux for the configured target mode."""
    if config.target_mode == "terminal-score":
        return terminal_conditioning_flux_torch(batch.states, batch.targets, config)
    velocity = training_target_velocity_torch(batch, config)
    total_flux = poisson_flux_from_velocity_torch(velocity, grid_size=int(config.grid_size))
    if bool(config.free_aware_target):
        total_flux = total_flux - float(batch.train_free_weight) * free_drift_flux_torch(batch.states, config)
    return total_flux


def direct_flux_rollout_consistency_loss(
    model: DirectFluxUNet,
    batch: FluxTrainingBatch,
    *,
    max_items: int | None = None,
    steps: int | None = None,
    return_extra: bool = False,
    terminal_classifier: TinyMNISTClassifier | None = None,
    shape_stats: dict[str, Tensor] | None = None,
    enable_terminal_losses: bool = True,
) -> tuple[Tensor, ...]:
    """Multi-step sampler-consistency loss on a small microbatch.

    The prediction is unrolled through the same differentiable conservative
    clipped step used by the one-step loss.  The teacher branch is unrolled with
    the corresponding supervised conditioning flux and no gradients.  Experiment
    10l makes the optional terminal losses safe: they act only after the rollout
    endpoint is actually close to terminal time, and they are ramped in over the
    early training steps.  This avoids forcing mid-trajectory states to look like
    final MNIST digits, which produced checkerboard/classifier artifacts in 10k.
    """
    config = model.config
    rollout_steps = int(config.rollout_loss_steps if steps is None else steps)
    if rollout_steps <= 0:
        z = batch.states.new_tensor(0.0)
        return (z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z) if return_extra else (z, z)
    count = int(batch.states.shape[0])
    if max_items is None:
        max_items = int(config.rollout_loss_batch_size)
    count = min(count, int(max_items))
    if count <= 0:
        z = batch.states.new_tensor(0.0)
        return (z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z) if return_extra else (z, z)
    sl = slice(0, count)
    horizon = max(float(natural_horizon(config)), 1e-12)
    dt = horizon / float(max(int(config.num_steps), 1))
    pred_state = batch.states[sl]
    teacher_state = batch.states[sl].detach()
    labels = batch.labels[sl]
    sources = batch.sources[sl]
    targets = batch.targets[sl]
    tau = batch.tau[sl]
    if enable_terminal_losses and steps is None:
        # Only the explicit global to-terminal mode rolls all the way to tau=0.
        # The 10r terminal-batch-specific to-zero path over-constrained a small
        # near-terminal microbatch and collapsed the learned correction flux.
        should_roll_to_zero = str(config.terminal_loss_mode) == "to-terminal"
        if should_roll_to_zero:
            needed = int(torch.ceil((tau.max() / max(dt, 1e-12)).detach()).item())
            rollout_steps = max(1, min(needed, int(config.terminal_rollout_max_steps)))
    free_w = float(batch.train_free_weight) if bool(config.stochastic_step_loss) else 0.0
    for _ in range(rollout_steps):
        pred_flux = model.predict_flux(tau, pred_state, labels, sources)
        pred_state = eulerian_flux_step_differentiable_torch(
            pred_state,
            pred_flux,
            dt,
            config,
            free_weight=free_w,
            learned_weight=1.0,
        )
        with torch.no_grad():
            teacher_batch = FluxTrainingBatch(
                tau=tau,
                states=teacher_state,
                labels=labels,
                targets=targets,
                sources=sources,
                target_velocity_mode=batch.target_velocity_mode,
                step_index=batch.step_index,
                train_free_weight=batch.train_free_weight,
                train_noise_weight=batch.train_noise_weight,
                is_terminal_batch=batch.is_terminal_batch,
            )
            teacher_flux = training_target_flux_torch(teacher_batch, config)
            teacher_state = eulerian_flux_step_differentiable_torch(
                teacher_state,
                teacher_flux,
                dt,
                config,
                free_weight=free_w,
                learned_weight=1.0,
            )
            tau = (tau - dt).clamp_min(0.0)
    scale = float(config.grid_size * config.grid_size)
    pred_scaled = pred_state * scale
    target_scaled = targets * scale
    rollout_l2 = F.mse_loss(pred_scaled, teacher_state * scale)

    tau_frac_after = (tau / horizon).clamp(0.0, 1.0)
    effective_terminal_tau_max = float(config.terminal_loss_tau_max_fraction)
    if str(config.terminal_loss_mode) in {"near-terminal", "to-terminal"}:
        effective_terminal_tau_max = min(
            effective_terminal_tau_max,
            float(config.terminal_rollout_max_steps) / float(max(int(config.num_steps), 1)),
        )
    terminal_mask = (tau_frac_after <= effective_terminal_tau_max) if enable_terminal_losses else torch.zeros_like(tau_frac_after, dtype=torch.bool)
    terminal_active_fraction = terminal_mask.float().mean()
    terminal_tau_mean = _masked_mean_torch(tau_frac_after, terminal_mask)
    terminal_loss_scale = _safe_terminal_loss_scale(batch.step_index, config, pred_state)

    endpoint_l2_vec = (pred_scaled - target_scaled).square().flatten(1).mean(dim=1)
    endpoint_l2 = terminal_loss_scale * _masked_mean_torch(endpoint_l2_vec, terminal_mask)
    if not return_extra:
        return rollout_l2, endpoint_l2

    # Keep the rollout image-gradient loss as the stable 10j sharpening signal.
    # The more aggressive endpoint losses below are terminal-gated.
    rollout_image_grad = image_gradient_mse_torch(pred_scaled, target_scaled, grid_size=int(config.grid_size))

    endpoint_bce_vec = binary_cross_entropy_probs_per_sample_autocast_safe(pred_scaled, target_scaled)
    endpoint_bce = terminal_loss_scale * _masked_mean_torch(endpoint_bce_vec, terminal_mask)

    pred_tv_vec = image_total_variation_per_sample(pred_scaled, grid_size=int(config.grid_size))
    target_tv_vec = image_total_variation_per_sample(target_scaled, grid_size=int(config.grid_size)).detach()
    # Raw scalar TV matching rewarded alternating pixel patterns in 10k.  Use a
    # relative TV discrepancy and terminal gating so the loss cannot dominate.
    endpoint_tv_vec = ((pred_tv_vec - target_tv_vec) / target_tv_vec.clamp_min(1.0)).square()
    endpoint_tv_loss = terminal_loss_scale * _masked_mean_torch(endpoint_tv_vec, terminal_mask)
    target_tv_loss = endpoint_tv_loss

    cls_loss = pred_state.new_tensor(0.0)
    cls_conf_loss = pred_state.new_tensor(0.0)
    classifier_mode = str(config.classifier_loss_mode)
    if bool(config.use_classifier_loss) and classifier_mode == "off":
        classifier_mode = "terminal"
    if terminal_classifier is not None and classifier_mode != "off" and (
        float(config.classifier_loss_weight) > 0.0 or float(config.classifier_confidence_loss_weight) > 0.0
    ):
        terminal_classifier.eval()
        cls_input = _classifier_input_from_masses_for_loss(
            pred_state,
            int(config.grid_size),
            blur_sigma=float(config.classifier_loss_blur_sigma),
        )
        logits = terminal_classifier(cls_input)
        probs = logits.softmax(dim=1)
        target_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        classifier_mask = terminal_mask
        if classifier_mode == "low-confidence-terminal":
            classifier_mask = classifier_mask & (target_prob.detach() < float(config.classifier_loss_confidence_threshold))
        ce_vec = F.cross_entropy(logits, labels, reduction="none")
        cls_loss = terminal_loss_scale * _masked_mean_torch(ce_vec, classifier_mask)
        cls_conf_loss = terminal_loss_scale * _masked_mean_torch(1.0 - target_prob, classifier_mask)

    shape_loss = pred_state.new_tensor(0.0)
    shape_entropy_loss = pred_state.new_tensor(0.0)
    shape_tv_loss = pred_state.new_tensor(0.0)
    shape_maxmass_loss = pred_state.new_tensor(0.0)
    if shape_stats and float(config.terminal_shape_loss_weight) > 0.0:
        shape_loss, shape_entropy_loss, shape_tv_loss, shape_maxmass_loss = terminal_shape_loss_torch(
            pred_state,
            labels,
            shape_stats,
            grid_size=int(config.grid_size),
            weights=terminal_mask.to(dtype=pred_state.dtype) * terminal_loss_scale,
            entropy_weight=float(config.terminal_shape_entropy_weight),
            tv_weight=float(config.terminal_shape_tv_weight),
            maxmass_weight=float(config.terminal_shape_maxmass_weight),
        )

    local_shape_loss = pred_state.new_tensor(0.0)
    local_support_loss = pred_state.new_tensor(0.0)
    local_edge_loss = pred_state.new_tensor(0.0)
    negative_space_loss = pred_state.new_tensor(0.0)
    gap_shape_loss = pred_state.new_tensor(0.0)
    missing_support_loss = pred_state.new_tensor(0.0)
    extra_support_loss = pred_state.new_tensor(0.0)
    gap_loss = pred_state.new_tensor(0.0)
    strict_negative_space_loss = pred_state.new_tensor(0.0)
    foreground_recall_loss = pred_state.new_tensor(0.0)
    if float(config.terminal_local_shape_loss_weight) > 0.0:
        local_shape_loss, local_support_loss, local_edge_loss, negative_space_loss = terminal_local_shape_loss_torch(
            pred_state,
            targets,
            labels,
            shape_stats,
            config,
            weights=terminal_mask.to(dtype=pred_state.dtype) * terminal_loss_scale,
        )
    if (
        float(config.terminal_gap_loss_weight) > 0.0
        or float(config.terminal_missing_support_weight) > 0.0
        or float(config.terminal_extra_support_weight) > 0.0
        or float(config.terminal_foreground_recall_weight) > 0.0
    ):
        gap_shape_loss, missing_support_loss, extra_support_loss, gap_loss, strict_negative_space_loss = terminal_gap_shape_loss_torch(
            pred_state,
            targets,
            labels,
            shape_stats,
            config,
            weights=terminal_mask.to(dtype=pred_state.dtype) * terminal_loss_scale,
        )
        if float(config.terminal_foreground_recall_weight) > 0.0:
            foreground_recall_loss = terminal_foreground_recall_loss_torch(
                pred_state,
                targets,
                labels,
                config,
                weights=terminal_mask.to(dtype=pred_state.dtype) * terminal_loss_scale,
            )
            gap_shape_loss = gap_shape_loss + float(config.terminal_foreground_recall_weight) * foreground_recall_loss
    return (
        rollout_l2,
        endpoint_l2,
        rollout_image_grad,
        target_tv_loss,
        endpoint_bce,
        endpoint_tv_loss,
        cls_loss,
        cls_conf_loss,
        terminal_active_fraction.detach(),
        terminal_tau_mean.detach(),
        terminal_loss_scale.detach(),
        shape_loss,
        shape_entropy_loss,
        shape_tv_loss,
        shape_maxmass_loss,
        local_shape_loss,
        local_support_loss,
        local_edge_loss,
        negative_space_loss,
        gap_shape_loss,
        missing_support_loss,
        extra_support_loss,
        gap_loss,
        strict_negative_space_loss,
        foreground_recall_loss,
    )


def direct_flux_matching_loss(
    model: DirectFluxUNet,
    batch: FluxTrainingBatch,
    *,
    step_index: int | None = None,
    terminal_classifier: TinyMNISTClassifier | None = None,
    shape_stats: dict[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Return the direct-flux regression loss and scalar diagnostics."""
    config = model.config
    raw_pred_norm = model(batch.tau, batch.states, batch.labels, batch.sources)
    if bool(config.project_main_loss):
        pred_norm = apply_flux_parameterization_torch(raw_pred_norm, batch.states, config)
    else:
        # Projection keeps the same node velocity/divergence, so the main
        # losses can skip the FFT-based projection.  Sampler-consistency losses
        # and generation still use the configured parameterization.
        pred_norm = raw_pred_norm
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

    pred_step_norm = pred_norm
    if not bool(config.project_main_loss) and str(config.flux_parameterization) != "edge":
        needs_step_projection = (
            float(config.step_loss_weight) > 0.0
            or float(config.rollout_loss_weight) > 0.0
            or float(config.rollout_image_grad_loss_weight) > 0.0
            or float(config.rollout_endpoint_l2_weight) > 0.0
            or float(config.rollout_endpoint_bce_weight) > 0.0
            or float(config.rollout_endpoint_tv_weight) > 0.0
            or ((bool(config.use_classifier_loss) or str(config.classifier_loss_mode) != "off") and (float(config.classifier_loss_weight) > 0.0 or float(config.classifier_confidence_loss_weight) > 0.0))
            or float(config.terminal_shape_loss_weight) > 0.0
            or float(config.terminal_local_shape_loss_weight) > 0.0
            or float(config.terminal_gap_loss_weight) > 0.0
            or float(config.terminal_missing_support_weight) > 0.0
            or float(config.terminal_extra_support_weight) > 0.0
            or float(config.terminal_foreground_recall_weight) > 0.0
            or float(config.target_tv_loss_weight) > 0.0
            or float(config.target_entropy_loss_weight) > 0.0
        )
        if needs_step_projection:
            pred_step_norm = apply_flux_parameterization_torch(raw_pred_norm, batch.states, config)

    step_loss = pred_norm.new_tensor(0.0)
    image_grad_loss = pred_norm.new_tensor(0.0)
    target_entropy_loss = pred_norm.new_tensor(0.0)
    pred_next: Tensor | None = None
    target_next: Tensor | None = None
    if float(config.step_loss_weight) > 0.0 or float(config.image_grad_loss_weight) > 0.0 or float(config.target_entropy_loss_weight) > 0.0:
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
            pred_step_norm * float(config.flux_scale),
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
        if float(config.step_loss_weight) > 0.0:
            step_loss = F.mse_loss(pred_next * density_scale, target_next * density_scale)
        if float(config.image_grad_loss_weight) > 0.0:
            image_grad_loss = image_gradient_mse_torch(pred_next * density_scale, target_next * density_scale, grid_size=int(config.grid_size))
        if float(config.target_entropy_loss_weight) > 0.0:
            target_entropy_loss = (_mass_entropy_torch(pred_next).mean() - _mass_entropy_torch(batch.targets).mean()).square()

    rollout_loss = pred_norm.new_tensor(0.0)
    rollout_endpoint_l2 = pred_norm.new_tensor(0.0)
    rollout_image_grad_loss = pred_norm.new_tensor(0.0)
    rollout_endpoint_bce_loss = pred_norm.new_tensor(0.0)
    rollout_endpoint_tv_loss = pred_norm.new_tensor(0.0)
    classifier_loss = pred_norm.new_tensor(0.0)
    classifier_confidence_loss = pred_norm.new_tensor(0.0)
    terminal_shape_loss = pred_norm.new_tensor(0.0)
    terminal_shape_entropy_loss = pred_norm.new_tensor(0.0)
    terminal_shape_tv_loss = pred_norm.new_tensor(0.0)
    terminal_shape_maxmass_loss = pred_norm.new_tensor(0.0)
    terminal_local_shape_loss = pred_norm.new_tensor(0.0)
    terminal_local_support_loss = pred_norm.new_tensor(0.0)
    terminal_local_edge_loss = pred_norm.new_tensor(0.0)
    terminal_negative_space_loss = pred_norm.new_tensor(0.0)
    terminal_gap_shape_loss = pred_norm.new_tensor(0.0)
    terminal_missing_support_loss = pred_norm.new_tensor(0.0)
    terminal_extra_support_loss = pred_norm.new_tensor(0.0)
    terminal_gap_loss = pred_norm.new_tensor(0.0)
    terminal_strict_negative_space_loss = pred_norm.new_tensor(0.0)
    terminal_foreground_recall_loss = pred_norm.new_tensor(0.0)
    target_tv_loss = pred_norm.new_tensor(0.0)
    terminal_active_fraction = pred_norm.new_tensor(0.0)
    terminal_tau_mean = pred_norm.new_tensor(float("nan"))
    terminal_loss_scale = pred_norm.new_tensor(0.0)
    rollout_ready = batch.step_index is None or int(batch.step_index) >= int(config.rollout_loss_warmup_steps)
    rollout_due = True
    terminal_due = True
    if batch.step_index is not None:
        rollout_due = int(batch.step_index) % int(config.rollout_loss_every) == 0
        terminal_due = int(batch.step_index) % int(config.terminal_loss_every) == 0
        if rollout_due and float(config.rollout_loss_prob) < 1.0:
            # Deterministic gate avoids adding another RNG stream to the loss.
            gate = math.fmod(abs(math.sin(float(int(batch.step_index) + 1) * 12.9898) * 43758.5453), 1.0)
            rollout_due = gate < float(config.rollout_loss_prob)
    has_core_rollout_losses = (
        float(config.rollout_loss_weight) > 0.0
        or float(config.rollout_image_grad_loss_weight) > 0.0
    )
    has_terminal_losses = (
        float(config.rollout_endpoint_l2_weight) > 0.0
        or float(config.rollout_endpoint_bce_weight) > 0.0
        or float(config.rollout_endpoint_tv_weight) > 0.0
        or ((bool(config.use_classifier_loss) or str(config.classifier_loss_mode) != "off") and (float(config.classifier_loss_weight) > 0.0 or float(config.classifier_confidence_loss_weight) > 0.0))
        or float(config.terminal_shape_loss_weight) > 0.0
        or float(config.terminal_local_shape_loss_weight) > 0.0
        or float(config.terminal_gap_loss_weight) > 0.0
        or float(config.terminal_missing_support_weight) > 0.0
        or float(config.terminal_extra_support_weight) > 0.0
        or float(config.terminal_foreground_recall_weight) > 0.0
        or float(config.target_tv_loss_weight) > 0.0
    )
    should_rollout = (has_core_rollout_losses and rollout_due) or (has_terminal_losses and terminal_due)
    if rollout_ready and should_rollout and int(config.rollout_loss_steps) > 0:
        rollout_max_items = int(config.rollout_loss_batch_size)
        if terminal_due and has_terminal_losses and not (has_core_rollout_losses and rollout_due):
            rollout_max_items = int(config.terminal_rollout_batch_size)
        (
            rollout_loss,
            rollout_endpoint_l2,
            rollout_image_grad_loss,
            target_tv_loss,
            rollout_endpoint_bce_loss,
            rollout_endpoint_tv_loss,
            classifier_loss,
            classifier_confidence_loss,
            terminal_active_fraction,
            terminal_tau_mean,
            terminal_loss_scale,
            terminal_shape_loss,
            terminal_shape_entropy_loss,
            terminal_shape_tv_loss,
            terminal_shape_maxmass_loss,
            terminal_local_shape_loss,
            terminal_local_support_loss,
            terminal_local_edge_loss,
            terminal_negative_space_loss,
            terminal_gap_shape_loss,
            terminal_missing_support_loss,
            terminal_extra_support_loss,
            terminal_gap_loss,
            terminal_strict_negative_space_loss,
            terminal_foreground_recall_loss,
        ) = direct_flux_rollout_consistency_loss(
            model,
            batch,
            max_items=rollout_max_items,
            return_extra=True,
            terminal_classifier=terminal_classifier,
            shape_stats=shape_stats,
            enable_terminal_losses=terminal_due,
        )


    terminal_local_loss_cap_scale = pred_norm.new_tensor(1.0)
    if float(config.terminal_local_loss_max_ratio) > 0.0:
        local_uncapped = (float(config.terminal_local_shape_loss_weight) * terminal_local_shape_loss) + terminal_gap_shape_loss
        cap_ref = (float(config.rollout_loss_weight) * rollout_loss.detach()) + (float(config.terminal_shape_loss_weight) * terminal_shape_loss.detach())
        cap_value = float(config.terminal_local_loss_max_ratio) * cap_ref.clamp_min(1e-4)
        if bool((local_uncapped.detach() > cap_value).cpu()):
            terminal_local_loss_cap_scale = (cap_value / local_uncapped.detach().clamp_min(1e-12)).to(dtype=pred_norm.dtype)
            terminal_gap_shape_loss = terminal_gap_shape_loss * terminal_local_loss_cap_scale
            terminal_local_shape_loss = terminal_local_shape_loss * terminal_local_loss_cap_scale

    curl_loss = flux_curl_torch(raw_pred_norm).square().mean()
    edge_lap_loss = edge_laplacian_energy_torch(raw_pred_norm)
    checker_loss = checkerboard_energy_torch(pred_div, grid_size=int(config.grid_size))

    loss = (
        flux_loss
        + float(config.divergence_loss_weight) * div_loss
        + float(config.node_loss_weight) * node_loss
        + float(config.step_loss_weight) * step_loss
        + float(config.rollout_loss_weight) * rollout_loss
        + float(config.rollout_endpoint_l2_weight) * rollout_endpoint_l2
        + float(config.rollout_endpoint_bce_weight) * rollout_endpoint_bce_loss
        + float(config.rollout_endpoint_tv_weight) * rollout_endpoint_tv_loss
        + float(config.image_grad_loss_weight) * image_grad_loss
        + float(config.rollout_image_grad_loss_weight) * rollout_image_grad_loss
        + float(config.target_tv_loss_weight) * target_tv_loss
        + float(config.target_entropy_loss_weight) * target_entropy_loss
        + (float(config.classifier_loss_weight) * classifier_loss if (bool(config.use_classifier_loss) or str(config.classifier_loss_mode) != "off") else classifier_loss.new_tensor(0.0))
        + (float(config.classifier_confidence_loss_weight) * classifier_confidence_loss if (bool(config.use_classifier_loss) or str(config.classifier_loss_mode) != "off") else classifier_confidence_loss.new_tensor(0.0))
        + float(config.terminal_shape_loss_weight) * terminal_shape_loss
        + float(config.terminal_local_shape_loss_weight) * terminal_local_shape_loss
        + terminal_gap_shape_loss
        + float(config.curl_loss_weight) * curl_loss
        + float(config.edge_laplacian_loss_weight) * edge_lap_loss
        + float(config.checkerboard_loss_weight) * checker_loss
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
            pred_step_norm * float(config.flux_scale),
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
        "rollout_loss": float(rollout_loss.detach().cpu()),
        "rollout_endpoint_l2": float(rollout_endpoint_l2.detach().cpu()),
        "rollout_endpoint_bce_loss": float(rollout_endpoint_bce_loss.detach().cpu()),
        "rollout_endpoint_tv_loss": float(rollout_endpoint_tv_loss.detach().cpu()),
        "rollout_image_grad_loss": float(rollout_image_grad_loss.detach().cpu()),
        "target_tv_loss": float(target_tv_loss.detach().cpu()),
        "target_entropy_loss": float(target_entropy_loss.detach().cpu()),
        "classifier_loss": float(classifier_loss.detach().cpu()),
        "classifier_confidence_loss": float(classifier_confidence_loss.detach().cpu()),
        "terminal_shape_loss": float(terminal_shape_loss.detach().cpu()),
        "terminal_shape_entropy_loss": float(terminal_shape_entropy_loss.detach().cpu()),
        "terminal_shape_tv_loss": float(terminal_shape_tv_loss.detach().cpu()),
        "terminal_shape_maxmass_loss": float(terminal_shape_maxmass_loss.detach().cpu()),
        "terminal_local_shape_loss": float(terminal_local_shape_loss.detach().cpu()),
        "terminal_local_support_loss": float(terminal_local_support_loss.detach().cpu()),
        "terminal_local_edge_loss": float(terminal_local_edge_loss.detach().cpu()),
        "terminal_negative_space_loss": float(terminal_negative_space_loss.detach().cpu()),
        "terminal_gap_shape_loss": float(terminal_gap_shape_loss.detach().cpu()),
        "terminal_missing_support_loss": float(terminal_missing_support_loss.detach().cpu()),
        "terminal_extra_support_loss": float(terminal_extra_support_loss.detach().cpu()),
        "terminal_gap_loss": float(terminal_gap_loss.detach().cpu()),
        "terminal_strict_negative_space_loss": float(terminal_strict_negative_space_loss.detach().cpu()),
        "terminal_foreground_recall_loss": float(terminal_foreground_recall_loss.detach().cpu()),
        "terminal_local_loss_cap_scale": float(terminal_local_loss_cap_scale.detach().cpu()),
        "terminal_loss_active_fraction": float(terminal_active_fraction.detach().cpu()),
        "terminal_tau_mean": float(terminal_tau_mean.detach().cpu()),
        "terminal_loss_scale": float(terminal_loss_scale.detach().cpu()),
        "image_grad_loss": float(image_grad_loss.detach().cpu()),
        "curl_loss": float(curl_loss.detach().cpu()),
        "edge_laplacian_loss": float(edge_lap_loss.detach().cpu()),
        "checkerboard_loss": float(checker_loss.detach().cpu()),
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
    rollout of the current model.  Its target velocity uses ``config.on_policy_target_mode``; the default safe-residual target clips residual corrections relative to the constant source-target velocity.
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
    if str(config.on_policy_prefix_mode) == "short":
        min_prefix = 1
        max_prefix = min(int(config.on_policy_prefix_steps), int(config.num_steps) - 1)
    else:
        min_prefix = max(1, int(round(float(config.on_policy_min_prefix_fraction) * float(config.num_steps))))
        max_prefix = min(int(config.num_steps) - 1, int(round(float(config.on_policy_max_prefix_fraction) * float(config.num_steps))))
    if max_prefix <= 0 or max_prefix < min_prefix:
        return FluxTrainingBatch(
            tau=base.tau,
            states=base.states,
            labels=base.labels,
            targets=base.targets,
            sources=base.sources,
            source_indices=base.source_indices,
            source_labels=base.source_labels,
            target_indices=base.target_indices,
            target_velocity_mode=str(config.on_policy_target_mode),
            step_index=base.step_index,
            train_free_weight=base.train_free_weight,
            train_noise_weight=base.train_noise_weight,
        )
    if str(config.on_policy_prefix_mode) == "late-biased":
        u = float(rng.random())
        prefix_steps = int(round(min_prefix + (max_prefix - min_prefix) * math.sqrt(u)))
    else:
        prefix_steps = int(rng.integers(min_prefix, max_prefix + 1))
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
        target_velocity_mode=str(config.on_policy_target_mode),
        step_index=base.step_index,
        train_free_weight=base.train_free_weight,
        train_noise_weight=base.train_noise_weight,
    )


def _batch_to_device(batch: FluxTrainingBatch, device: torch.device, *, step_index: int | None = None) -> FluxTrainingBatch:
    return FluxTrainingBatch(
        tau=batch.tau.to(device),
        states=batch.states.to(device),
        labels=batch.labels.to(device),
        targets=batch.targets.to(device),
        sources=batch.sources.to(device),
        source_indices=None if batch.source_indices is None else np.asarray(batch.source_indices, dtype=np.int64).copy(),
        source_labels=None if batch.source_labels is None else np.asarray(batch.source_labels, dtype=np.int64).copy(),
        target_indices=None if batch.target_indices is None else np.asarray(batch.target_indices, dtype=np.int64).copy(),
        target_velocity_mode=batch.target_velocity_mode,
        step_index=batch.step_index if step_index is None else int(step_index),
        train_free_weight=float(batch.train_free_weight),
        train_noise_weight=float(batch.train_noise_weight),
        is_terminal_batch=bool(batch.is_terminal_batch),
    )


def _concat_optional_arrays(values: list[IntArray | None]) -> IntArray | None:
    if any(v is None for v in values):
        return None
    return np.concatenate([np.asarray(v, dtype=np.int64).reshape(-1) for v in values], axis=0)


def _concat_training_batches(batches: list[FluxTrainingBatch], *, device: torch.device, step_index: int | None = None) -> FluxTrainingBatch:
    if not batches:
        raise ValueError("at least one batch is required")
    mode = batches[0].target_velocity_mode
    return FluxTrainingBatch(
        tau=torch.cat([b.tau.to(device) for b in batches], dim=0),
        states=torch.cat([b.states.to(device) for b in batches], dim=0),
        labels=torch.cat([b.labels.to(device) for b in batches], dim=0),
        targets=torch.cat([b.targets.to(device) for b in batches], dim=0),
        sources=torch.cat([b.sources.to(device) for b in batches], dim=0),
        source_indices=_concat_optional_arrays([b.source_indices for b in batches]),
        source_labels=_concat_optional_arrays([b.source_labels for b in batches]),
        target_indices=_concat_optional_arrays([b.target_indices for b in batches]),
        target_velocity_mode=mode,
        step_index=step_index,
        train_free_weight=float(batches[0].train_free_weight),
        train_noise_weight=float(batches[0].train_noise_weight),
        is_terminal_batch=any(bool(b.is_terminal_batch) for b in batches),
    )


def _subset_training_batch(batch: FluxTrainingBatch, indices: np.ndarray, *, device: torch.device, step_index: int | None = None) -> FluxTrainingBatch:
    idx_np = np.asarray(indices, dtype=np.int64).reshape(-1)
    idx_t = torch.as_tensor(idx_np, dtype=torch.long, device=batch.states.device)
    def arr_subset(arr: IntArray | None) -> IntArray | None:
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.int64).reshape(-1)[idx_np].copy()
    return FluxTrainingBatch(
        tau=batch.tau.index_select(0, idx_t).to(device),
        states=batch.states.index_select(0, idx_t).to(device),
        labels=batch.labels.index_select(0, idx_t).to(device),
        targets=batch.targets.index_select(0, idx_t).to(device),
        sources=batch.sources.index_select(0, idx_t).to(device),
        source_indices=arr_subset(batch.source_indices),
        source_labels=arr_subset(batch.source_labels),
        target_indices=arr_subset(batch.target_indices),
        target_velocity_mode=batch.target_velocity_mode,
        step_index=step_index,
        train_free_weight=float(batch.train_free_weight),
        train_noise_weight=float(batch.train_noise_weight),
        is_terminal_batch=bool(batch.is_terminal_batch),
    )


def _on_policy_prefix_bounds(config: DirectFluxMNISTConfig) -> tuple[int, int]:
    if str(config.on_policy_prefix_mode) == "short":
        min_prefix = 1
        max_prefix = min(int(config.on_policy_prefix_steps), int(config.num_steps) - 1)
    else:
        min_prefix = max(1, int(round(float(config.on_policy_min_prefix_fraction) * float(config.num_steps))))
        max_prefix = min(int(config.num_steps) - 1, int(round(float(config.on_policy_max_prefix_fraction) * float(config.num_steps))))
    return min_prefix, max_prefix


def _trajectory_snapshot_steps(config: DirectFluxMNISTConfig) -> np.ndarray:
    """Return replay snapshot prefix steps, including optional terminal-biased states.

    ``tau`` decreases from one at the source to zero at the terminal digit. The
    stable schedule de-duplicates integer snapshot steps.  The 10r exact-fraction
    schedule over-filled the replay cache with terminal model-rollout states and
    made on-policy training much harder; for stability we keep terminal replay
    biased but not dominant.
    """
    min_prefix, max_prefix = _on_policy_prefix_bounds(config)
    if max_prefix <= 0 or max_prefix < min_prefix:
        return np.asarray([], dtype=np.int64)
    count = max(1, int(config.on_policy_cache_snapshots_per_traj))
    terminal_count = int(round(float(config.on_policy_cache_terminal_fraction) * float(count)))
    terminal_count = max(0, min(count, terminal_count))
    regular_count = max(0, count - terminal_count)
    pieces: list[np.ndarray] = []
    if regular_count > 0:
        if str(config.on_policy_prefix_mode) == "late-biased":
            u = np.linspace(0.0, 1.0, regular_count + 2, dtype=np.float64)[1:-1]
            values = min_prefix + (max_prefix - min_prefix) * np.sqrt(u)
        else:
            values = np.linspace(min_prefix, max_prefix, regular_count, dtype=np.float64)
        pieces.append(values)
    if terminal_count > 0:
        # Terminal tau fraction in [tau_min, tau_max] corresponds to prefix
        # fraction in [1 - tau_max, 1 - tau_min].  Exclude the exact tau=0
        # snapshot from replay caches: those states are produced by an imperfect
        # model rollout and are too off-policy early in training.
        nsteps = max(int(config.num_steps), 1)
        lo = int(round((1.0 - float(config.on_policy_cache_terminal_max_tau)) * float(nsteps)))
        hi = int(round((1.0 - float(config.on_policy_cache_terminal_min_tau)) * float(nsteps)))
        lo = max(min_prefix, min(max_prefix, lo))
        hi = max(lo, min(int(config.num_steps) - 1, hi))
        pieces.append(np.linspace(lo, hi, terminal_count, dtype=np.float64))
    if not pieces:
        return np.asarray([max_prefix], dtype=np.int64)
    steps = np.unique(np.clip(np.round(np.concatenate(pieces)).astype(np.int64), min_prefix, int(config.num_steps) - 1))
    if steps.size == 0:
        steps = np.asarray([max_prefix], dtype=np.int64)
    return steps


def _snapshot_terminal_stats(config: DirectFluxMNISTConfig, snapshot_steps: np.ndarray) -> tuple[float, int, int]:
    """Return requested fraction, terminal count, and regular count for a snapshot plan."""
    count = int(np.asarray(snapshot_steps).reshape(-1).size)
    if count <= 0:
        return float("nan"), 0, 0
    nsteps = max(int(config.num_steps), 1)
    terminal_lo = int(math.ceil((1.0 - float(config.on_policy_cache_terminal_max_tau)) * float(nsteps)))
    terminal_lo = max(1, min(nsteps, terminal_lo))
    terminal_count = int((np.asarray(snapshot_steps).reshape(-1) >= terminal_lo).sum())
    regular_count = count - terminal_count
    return float(terminal_count) / float(count), terminal_count, regular_count


def _repeat_optional_array(arr: IntArray | None, repeats: int) -> IntArray | None:
    if arr is None:
        return None
    return np.tile(np.asarray(arr, dtype=np.int64).reshape(-1), int(repeats))


@torch.no_grad()
def _build_trajectory_on_policy_replay_cache(
    model: DirectFluxUNet,
    images: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    cache_size: int,
    rollout_batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    dtype: torch.dtype,
    class_means: np.ndarray,
    ot_cache: ClasswiseOTCache,
    step_index: int,
) -> OnPolicyReplayCache:
    """Build replay states by rolling each trajectory once and storing multiple snapshots."""
    start = time.perf_counter()
    cache_device = torch.device("cuda" if config.on_policy_cache_device == "cuda" and torch.cuda.is_available() else "cpu")
    snapshot_steps = _trajectory_snapshot_steps(config)
    requested_terminal_fraction, terminal_snapshot_count, regular_snapshot_count = _snapshot_terminal_stats(config, snapshot_steps)
    if snapshot_steps.size == 0:
        batch = sample_flux_training_batch(
            images,
            labels,
            config,
            batch_size=int(cache_size),
            device=device,
            rng=rng,
            dtype=dtype,
            class_means=class_means,
            ot_cache=ot_cache,
            step_index=step_index,
        )
        batch = FluxTrainingBatch(
            tau=batch.tau,
            states=batch.states,
            labels=batch.labels,
            targets=batch.targets,
            sources=batch.sources,
            source_indices=batch.source_indices,
            source_labels=batch.source_labels,
            target_indices=batch.target_indices,
            target_velocity_mode=str(config.on_policy_target_mode),
            step_index=batch.step_index,
            train_free_weight=batch.train_free_weight,
            train_noise_weight=batch.train_noise_weight,
        )
        cached = _batch_to_device(batch, cache_device, step_index=int(step_index))
        tau_frac = cached.tau / max(float(natural_horizon(config)), 1e-12)
        return OnPolicyReplayCache(
            batch=cached,
            created_step=int(step_index),
            refresh_seconds=time.perf_counter() - start,
            mode="trajectory",
            tau_min=float(tau_frac.min().detach().cpu()),
            tau_mean=float(tau_frac.mean().detach().cpu()),
            tau_max=float(tau_frac.max().detach().cpu()),
            terminal_fraction=float((tau_frac <= float(config.on_policy_cache_terminal_max_tau)).float().mean().detach().cpu()),
            terminal_requested_fraction=float(requested_terminal_fraction),
            terminal_actual_fraction=float((tau_frac <= float(config.on_policy_cache_terminal_max_tau)).float().mean().detach().cpu()),
            terminal_snapshot_count=int(terminal_snapshot_count),
            regular_snapshot_count=int(regular_snapshot_count),
        )
    pieces: list[FluxTrainingBatch] = []
    produced = 0
    horizon = float(natural_horizon(config))
    dt = horizon / float(max(int(config.num_steps), 1))
    model_was_training = bool(model.training)
    model.eval()
    max_step = int(snapshot_steps.max())
    while produced < int(cache_size):
        remaining_items = int(cache_size) - produced
        traj_batch = min(int(rollout_batch_size), max(1, int(math.ceil(remaining_items / max(1, snapshot_steps.size)))))
        base = sample_flux_training_batch(
            images,
            labels,
            config,
            batch_size=traj_batch,
            device=device,
            rng=rng,
            dtype=dtype,
            class_means=class_means,
            ot_cache=ot_cache,
            step_index=step_index,
        )
        states = base.sources.clone()
        source_condition = base.sources.clone()
        saved_states: list[Tensor] = []
        saved_tau: list[Tensor] = []
        wanted_counts: dict[int, int] = {}
        for snapshot_step in snapshot_steps.tolist():
            key = int(snapshot_step)
            wanted_counts[key] = wanted_counts.get(key, 0) + 1
        rollout_free = float(base.train_free_weight) if bool(config.on_policy_use_free) else 0.0
        rollout_noise = float(base.train_noise_weight) if bool(config.on_policy_use_noise) else 0.0
        for prefix_idx in range(max_step):
            tau_value = max(horizon - float(prefix_idx) * dt, 0.0)
            tau = torch.full((traj_batch,), tau_value, dtype=states.dtype, device=device)
            flux = model.predict_flux(tau, states, base.labels, source_condition)
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
            completed = prefix_idx + 1
            repeat_count = int(wanted_counts.get(int(completed), 0))
            if repeat_count > 0:
                tau_remaining = torch.full(
                    (traj_batch,),
                    max(horizon - float(completed) * dt, 0.0),
                    dtype=states.dtype,
                    device=device,
                )
                for _ in range(repeat_count):
                    saved_states.append(states.detach().clone())
                    saved_tau.append(tau_remaining.clone())
        if saved_states:
            repeats = len(saved_states)
            chunk = FluxTrainingBatch(
                tau=torch.cat(saved_tau, dim=0),
                states=torch.cat(saved_states, dim=0),
                labels=base.labels.repeat(repeats),
                targets=base.targets.repeat(repeats, 1),
                sources=base.sources.repeat(repeats, 1),
                source_indices=_repeat_optional_array(base.source_indices, repeats),
                source_labels=_repeat_optional_array(base.source_labels, repeats),
                target_indices=_repeat_optional_array(base.target_indices, repeats),
                target_velocity_mode=str(config.on_policy_target_mode),
                step_index=base.step_index,
                train_free_weight=base.train_free_weight,
                train_noise_weight=base.train_noise_weight,
            )
            if produced + chunk.states.shape[0] > int(cache_size):
                keep = np.arange(int(cache_size) - produced, dtype=np.int64)
                chunk = _subset_training_batch(chunk, keep, device=device, step_index=int(step_index))
            pieces.append(_batch_to_device(chunk, cache_device, step_index=int(step_index)))
            produced += int(chunk.states.shape[0])
    if model_was_training:
        model.train()
    batch = _concat_training_batches(pieces, device=cache_device, step_index=int(step_index))
    tau_frac = batch.tau / max(float(natural_horizon(config)), 1e-12)
    return OnPolicyReplayCache(
        batch=batch,
        created_step=int(step_index),
        refresh_seconds=time.perf_counter() - start,
        mode="trajectory",
        tau_min=float(tau_frac.min().detach().cpu()),
        tau_mean=float(tau_frac.mean().detach().cpu()),
        tau_max=float(tau_frac.max().detach().cpu()),
        terminal_fraction=float((tau_frac <= float(config.on_policy_cache_terminal_max_tau)).float().mean().detach().cpu()),
        terminal_requested_fraction=float(requested_terminal_fraction),
        terminal_actual_fraction=float((tau_frac <= float(config.on_policy_cache_terminal_max_tau)).float().mean().detach().cpu()),
        terminal_snapshot_count=int(terminal_snapshot_count),
        regular_snapshot_count=int(regular_snapshot_count),
    )


@torch.no_grad()
def build_on_policy_replay_cache(
    model: DirectFluxUNet,
    images: np.ndarray,
    labels: np.ndarray,
    config: DirectFluxMNISTConfig,
    *,
    cache_size: int,
    rollout_batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    dtype: torch.dtype,
    class_means: np.ndarray,
    ot_cache: ClasswiseOTCache,
    step_index: int,
) -> OnPolicyReplayCache:
    """Refresh a replay buffer of model-visited states.

    In ``trajectory`` mode each sampled trajectory is rolled once and multiple
    snapshots are stored, which is much cheaper than independent source-to-prefix
    rollouts for every cache item.
    """
    if str(config.on_policy_cache_mode) == "trajectory":
        return _build_trajectory_on_policy_replay_cache(
            model,
            images,
            labels,
            config,
            cache_size=int(cache_size),
            rollout_batch_size=int(rollout_batch_size),
            device=device,
            rng=rng,
            dtype=dtype,
            class_means=class_means,
            ot_cache=ot_cache,
            step_index=int(step_index),
        )

    start = time.perf_counter()
    cache_device = torch.device("cuda" if config.on_policy_cache_device == "cuda" and torch.cuda.is_available() else "cpu")
    pieces: list[FluxTrainingBatch] = []
    remaining = int(cache_size)
    model_was_training = bool(model.training)
    model.eval()
    while remaining > 0:
        current = min(int(rollout_batch_size), remaining)
        piece = make_on_policy_training_batch(
            model,
            images,
            labels,
            config,
            batch_size=current,
            device=device,
            rng=rng,
            dtype=dtype,
            class_means=class_means,
            ot_cache=ot_cache,
            step_index=step_index,
        )
        pieces.append(_batch_to_device(piece, cache_device, step_index=int(step_index)))
        remaining -= current
    if model_was_training:
        model.train()
    batch = _concat_training_batches(pieces, device=cache_device, step_index=int(step_index))
    tau_frac = batch.tau / max(float(natural_horizon(config)), 1e-12)
    return OnPolicyReplayCache(
        batch=batch,
        created_step=int(step_index),
        refresh_seconds=time.perf_counter() - start,
        mode="independent",
        tau_min=float(tau_frac.min().detach().cpu()),
        tau_mean=float(tau_frac.mean().detach().cpu()),
        tau_max=float(tau_frac.max().detach().cpu()),
        terminal_fraction=float((tau_frac <= float(config.on_policy_cache_terminal_max_tau)).float().mean().detach().cpu()),
        terminal_requested_fraction=float("nan"),
        terminal_actual_fraction=float((tau_frac <= float(config.on_policy_cache_terminal_max_tau)).float().mean().detach().cpu()),
        terminal_snapshot_count=0,
        regular_snapshot_count=0,
    )

def sample_on_policy_replay_batch(
    cache: OnPolicyReplayCache,
    *,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    step_index: int,
) -> FluxTrainingBatch:
    count = min(int(batch_size), int(cache.size))
    indices = rng.choice(cache.size, size=count, replace=cache.size < count)
    return _subset_training_batch(cache.batch, indices, device=device, step_index=int(step_index))


def _mass_entropy_torch(states: Tensor) -> Tensor:
    states = states.clamp_min(1e-30)
    return -(states * states.log()).sum(dim=1)


def _label_sequence_tensor(count: int, *, device: torch.device) -> Tensor:
    return torch.arange(count, device=device, dtype=torch.long) % 10


def _disable_mkldnn_for_cpu_if_needed(device: torch.device) -> None:
    """Avoid rare CPU convolution backward hangs seen with MKL-DNN on sparse MNIST masses."""
    if device.type == "cpu" and hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False



def init_ema_state(model: nn.Module) -> dict[str, Tensor]:
    """Create a detached state-dict copy for exponential moving average weights."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def update_ema_state(ema_state: dict[str, Tensor], model: nn.Module, decay: float) -> None:
    """Update EMA tensors in-place from ``model``."""
    if not ema_state:
        return
    decay_f = float(decay)
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if name not in ema_state:
                ema_state[name] = value.detach().clone()
                continue
            target = ema_state[name]
            if torch.is_floating_point(value):
                target.mul_(decay_f).add_(value.detach(), alpha=1.0 - decay_f)
            else:
                target.copy_(value.detach())


@contextmanager
def temporary_ema_weights(model: nn.Module, ema_state: dict[str, Tensor] | None):
    """Temporarily evaluate a model with EMA weights, restoring live weights after."""
    if ema_state is None:
        yield model
        return
    live_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    try:
        model.load_state_dict(ema_state, strict=False)
        yield model
    finally:
        model.load_state_dict(live_state, strict=False)


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
    terminal_classifier: TinyMNISTClassifier | None = None,
    class_shape_stats: dict[str, np.ndarray] | None = None,
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
    ema_state = init_ema_state(model) if float(model.config.ema_decay) > 0.0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    amp_enabled = bool(use_amp and resolved_device.type == "cuda")
    scaler = _make_cuda_grad_scaler(enabled=amp_enabled)
    ot_cache = build_classwise_ot_cache(images, labels, model.config)
    class_means = ot_cache.class_means
    class_shape_stats_torch = _shape_stats_to_torch(class_shape_stats, device=resolved_device, dtype=torch.float32)
    history: dict[str, list[float]] = {
        "loss": [],
        "flux_loss": [],
        "div_loss": [],
        "node_loss": [],
        "step_loss": [],
        "rollout_loss": [],
        "rollout_endpoint_l2": [],
        "rollout_endpoint_bce_loss": [],
        "rollout_endpoint_tv_loss": [],
        "rollout_image_grad_loss": [],
        "target_tv_loss": [],
        "target_entropy_loss": [],
        "classifier_loss": [],
        "classifier_confidence_loss": [],
        "terminal_shape_loss": [],
        "terminal_shape_entropy_loss": [],
        "terminal_shape_tv_loss": [],
        "terminal_shape_maxmass_loss": [],
        "terminal_local_shape_loss": [],
        "terminal_local_support_loss": [],
        "terminal_local_edge_loss": [],
        "terminal_negative_space_loss": [],
        "terminal_gap_shape_loss": [],
        "terminal_missing_support_loss": [],
        "terminal_extra_support_loss": [],
        "terminal_gap_loss": [],
        "terminal_strict_negative_space_loss": [],
        "terminal_foreground_recall_loss": [],
        "terminal_local_loss_cap_scale": [],
        "terminal_loss_active_fraction": [],
        "terminal_tau_mean": [],
        "terminal_loss_scale": [],
        "image_grad_loss": [],
        "curl_loss": [],
        "edge_laplacian_loss": [],
        "checkerboard_loss": [],
        "on_policy": [],
        "terminal_batch": [],
        "terminal_batch_loss": [],
        "terminal_batch_gap_loss": [],
        "terminal_batch_missing_support_loss": [],
        "terminal_batch_extra_support_loss": [],
        "terminal_batch_entropy": [],
        "terminal_batch_tv": [],
        "terminal_batch_active_fraction": [],
        "on_policy_tau_frac": [],
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
        "sec_per_step": [],
        "examples_per_sec": [],
        "on_policy_cache_age": [],
        "cache_refresh_sec": [],
        "cache_tau_min": [],
        "cache_tau_mean": [],
        "cache_tau_max": [],
        "cache_terminal_fraction": [],
        "cache_terminal_requested_fraction": [],
        "cache_terminal_actual_fraction": [],
        "cache_terminal_snapshot_count": [],
        "cache_regular_snapshot_count": [],
        "off_loss": [],
        "off_div_cos": [],
        "off_target_rms": [],
        "off_pred_rms": [],
        "on_loss": [],
        "on_div_cos": [],
        "on_target_rms": [],
        "on_pred_rms": [],
    }

    replay_cache: OnPolicyReplayCache | None = None
    preview_path = None if preview_dir is None or preview_every <= 0 else Path(preview_dir)
    if preview_path is not None:
        preview_path.mkdir(parents=True, exist_ok=True)

    bar = _progress(range(int(train_steps)), total=int(train_steps), desc="train flux", disable=not show_progress)
    for step_index in bar:
        iter_start = time.perf_counter()
        cache_refresh_sec_this_step = 0.0
        use_terminal_batch = (
            float(model.config.terminal_batch_prob) > 0.0
            and int(step_index) >= int(model.config.rollout_loss_warmup_steps)
            and rng.random() < float(model.config.terminal_batch_prob)
        )
        use_on_policy = (
            not use_terminal_batch
            and str(model.config.on_policy_mode) != "off"
            and int(step_index) >= int(model.config.on_policy_warmup_steps)
            and float(model.config.on_policy_prob) > 0.0
            and rng.random() < float(model.config.on_policy_prob)
        )
        if use_terminal_batch:
            batch = sample_terminal_flux_training_batch(
                images,
                labels,
                model.config,
                batch_size=min(int(batch_size), int(model.config.terminal_batch_size)),
                device=resolved_device,
                rng=rng,
                class_means=class_means,
                ot_cache=ot_cache,
                step_index=int(step_index),
            )
        elif use_on_policy and str(model.config.on_policy_mode) == "replay":
            cache_stale = (
                replay_cache is None
                or int(step_index) - int(replay_cache.created_step) >= int(model.config.on_policy_cache_refresh_interval)
                or replay_cache.size < min(int(batch_size), int(model.config.on_policy_batch_size))
            )
            if cache_stale:
                cache_context = (
                    temporary_ema_weights(model, ema_state)
                    if bool(model.config.use_ema_for_cache) and ema_state is not None
                    else nullcontext(model)
                )
                with cache_context:
                    replay_cache = build_on_policy_replay_cache(
                        model,
                        images,
                        labels,
                        model.config,
                        cache_size=int(model.config.on_policy_cache_size),
                        rollout_batch_size=int(model.config.on_policy_cache_rollout_batch_size),
                        device=resolved_device,
                        rng=rng,
                        dtype=torch.float32,
                        class_means=class_means,
                        ot_cache=ot_cache,
                        step_index=int(step_index),
                    )
                cache_refresh_sec_this_step = float(replay_cache.refresh_seconds)
            assert replay_cache is not None
            batch = sample_on_policy_replay_batch(
                replay_cache,
                batch_size=min(int(batch_size), int(model.config.on_policy_batch_size)),
                device=resolved_device,
                rng=rng,
                step_index=int(step_index),
            )
        elif use_on_policy and str(model.config.on_policy_mode) == "online":
            batch = make_on_policy_training_batch(
                model,
                images,
                labels,
                model.config,
                batch_size=min(int(batch_size), int(model.config.on_policy_batch_size)),
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
            loss, metrics = direct_flux_matching_loss(
                model,
                batch,
                step_index=int(step_index),
                terminal_classifier=terminal_classifier,
                shape_stats=class_shape_stats_torch,
            )
        metrics["on_policy"] = 1.0 if use_on_policy else 0.0
        metrics["terminal_batch"] = 1.0 if use_terminal_batch else 0.0
        metrics["on_policy_tau_frac"] = float((batch.tau / max(natural_horizon(model.config), 1e-12)).mean().detach().cpu()) if use_on_policy else -1.0
        inactive = float("nan")
        if use_terminal_batch:
            metrics["terminal_batch_loss"] = metrics["loss"]
            metrics["terminal_batch_gap_loss"] = metrics.get("terminal_gap_loss", 0.0)
            metrics["terminal_batch_missing_support_loss"] = metrics.get("terminal_missing_support_loss", 0.0)
            metrics["terminal_batch_extra_support_loss"] = metrics.get("terminal_extra_support_loss", 0.0)
            metrics["terminal_batch_entropy"] = metrics.get("terminal_shape_entropy_loss", 0.0)
            metrics["terminal_batch_tv"] = metrics.get("terminal_shape_tv_loss", 0.0)
            metrics["terminal_batch_active_fraction"] = metrics.get("terminal_loss_active_fraction", 0.0)
        else:
            metrics["terminal_batch_loss"] = inactive
            metrics["terminal_batch_gap_loss"] = inactive
            metrics["terminal_batch_missing_support_loss"] = inactive
            metrics["terminal_batch_extra_support_loss"] = inactive
            metrics["terminal_batch_entropy"] = inactive
            metrics["terminal_batch_tv"] = inactive
            metrics["terminal_batch_active_fraction"] = inactive
        metrics["on_policy_cache_age"] = -1.0 if replay_cache is None else float(int(step_index) - int(replay_cache.created_step))
        metrics["cache_refresh_sec"] = float(cache_refresh_sec_this_step)
        metrics["cache_tau_min"] = float("nan") if replay_cache is None else float(replay_cache.tau_min)
        metrics["cache_tau_mean"] = float("nan") if replay_cache is None else float(replay_cache.tau_mean)
        metrics["cache_tau_max"] = float("nan") if replay_cache is None else float(replay_cache.tau_max)
        metrics["cache_terminal_fraction"] = float("nan") if replay_cache is None else float(replay_cache.terminal_fraction)
        metrics["cache_terminal_requested_fraction"] = float("nan") if replay_cache is None else float(replay_cache.terminal_requested_fraction)
        metrics["cache_terminal_actual_fraction"] = float("nan") if replay_cache is None else float(replay_cache.terminal_actual_fraction)
        metrics["cache_terminal_snapshot_count"] = float("nan") if replay_cache is None else float(replay_cache.terminal_snapshot_count)
        metrics["cache_regular_snapshot_count"] = float("nan") if replay_cache is None else float(replay_cache.regular_snapshot_count)
        if use_on_policy:
            metrics["on_loss"] = metrics["loss"]
            metrics["on_div_cos"] = metrics["div_cos"]
            metrics["on_target_rms"] = metrics["target_rms"]
            metrics["on_pred_rms"] = metrics["pred_rms"]
            metrics["off_loss"] = inactive
            metrics["off_div_cos"] = inactive
            metrics["off_target_rms"] = inactive
            metrics["off_pred_rms"] = inactive
        else:
            metrics["off_loss"] = metrics["loss"]
            metrics["off_div_cos"] = metrics["div_cos"]
            metrics["off_target_rms"] = metrics["target_rms"]
            metrics["off_pred_rms"] = metrics["pred_rms"]
            metrics["on_loss"] = inactive
            metrics["on_div_cos"] = inactive
            metrics["on_target_rms"] = inactive
            metrics["on_pred_rms"] = inactive
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        if ema_state is not None:
            update_ema_state(ema_state, model, float(model.config.ema_decay))
        elapsed = max(time.perf_counter() - iter_start, 1e-12)
        metrics["sec_per_step"] = float(elapsed)
        metrics["examples_per_sec"] = float(batch.states.shape[0]) / elapsed
        for key in history:
            history[key].append(metrics.get(key, 0.0))
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
                roll=metrics.get("rollout_loss", 0.0),
                rimg=metrics.get("rollout_image_grad_loss", 0.0),
                ep=metrics.get("rollout_endpoint_l2", 0.0),
                cls=metrics.get("classifier_loss", 0.0),
                t_act=metrics.get("terminal_loss_active_fraction", 0.0),
                t_tau=metrics.get("terminal_tau_mean", float("nan")),
                tv=metrics.get("target_tv_loss", 0.0),
                curl=metrics.get("curl_loss", 0.0),
                pred=metrics["pred_rms"],
                tgt=metrics["target_rms"],
                onp=metrics.get("on_policy", 0.0),
                tb=metrics.get("terminal_batch", 0.0),
                tbg=metrics.get("terminal_batch_gap_loss", 0.0),
                on_tau=metrics.get("on_policy_tau_frac", -1.0),
                free_r=metrics.get("free_to_learned_ratio", 0.0),
                noise_r=metrics.get("noise_to_learned_ratio", 0.0),
                sec=metrics.get("sec_per_step", 0.0),
                eps=metrics.get("examples_per_sec", 0.0),
                cache_age=metrics.get("on_policy_cache_age", -1.0),
                cache_s=metrics.get("cache_refresh_sec", 0.0),
                c_tau=metrics.get("cache_tau_mean", float("nan")),
                c_term=metrics.get("cache_terminal_fraction", float("nan")),
                c_req=metrics.get("cache_terminal_requested_fraction", float("nan")),
                on_cos=metrics.get("on_div_cos", float("nan")),
                off_cos=metrics.get("off_div_cos", float("nan")),
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
    if ema_state is not None and bool(model.config.use_ema_for_sampling):
        model.load_state_dict(ema_state, strict=False)
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
    sample_entropy = float(_mass_entropy_torch(states).mean().detach().cpu())
    sample_tv = float(image_total_variation(states, grid_size=int(cfg.grid_size)).detach().cpu())
    sample_checker = float(checkerboard_energy_torch(states, grid_size=int(cfg.grid_size)).detach().cpu())
    sample_highfreq = float(highfreq_fraction_torch(states, grid_size=int(cfg.grid_size)).detach().cpu())
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
        sample_entropy=sample_entropy,
        sample_total_variation=sample_tv,
        sample_checkerboard_energy=sample_checker,
        sample_highfreq_fraction=sample_highfreq,
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




def _select_trajectory_frames(trajectory: np.ndarray, frame_count: int) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 3:
        raise ValueError("trajectory must have shape (T, B, N)")
    if frame_count <= 1 or traj.shape[0] <= frame_count:
        return traj
    idx = np.linspace(0, traj.shape[0] - 1, int(frame_count)).round().astype(np.int64)
    return traj[idx]


def save_diffusion_process_figure(
    trajectory: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    output_path: str | Path,
    *,
    grid_size: int = 28,
    num_frames: int = 12,
    max_samples: int = 8,
) -> None:
    """Save rows of individual trajectories from source to terminal sample."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        raise RuntimeError("matplotlib is required to save a process figure") from exc
    frames = _select_trajectory_frames(trajectory, int(num_frames))
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    count = min(int(max_samples), frames.shape[1])
    cols = int(frames.shape[0])
    fig, axes = plt.subplots(count, cols, figsize=(1.05 * cols, 1.12 * count), squeeze=False)
    for row in range(count):
        for col in range(cols):
            ax = axes[row, col]
            img = frames[col, row].reshape(grid_size, grid_size)
            ax.imshow(_normalize_for_display(img), cmap="gray", interpolation="nearest")
            if row == 0:
                frac = col / max(cols - 1, 1)
                ax.set_title(f"{frac:.2f}", fontsize=7)
            if col == 0:
                ax.set_ylabel(str(int(labels_arr[row])), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.12)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_diffusion_marginal_process_figure(
    trajectory: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    output_path: str | Path,
    *,
    grid_size: int = 28,
    num_frames: int = 8,
    samples_per_frame: int = 8,
) -> None:
    """Save time-slice rows showing the evolving generated distribution."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        raise RuntimeError("matplotlib is required to save a process figure") from exc
    frames = _select_trajectory_frames(trajectory, int(num_frames))
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    cols = min(int(samples_per_frame), frames.shape[1])
    rows = int(frames.shape[0])
    fig, axes = plt.subplots(rows, cols, figsize=(1.05 * cols, 1.15 * rows), squeeze=False)
    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]
            ax.imshow(_normalize_for_display(frames[row, col].reshape(grid_size, grid_size)), cmap="gray", interpolation="nearest")
            if row == 0:
                ax.set_title(str(int(labels_arr[col])), fontsize=7)
            if col == 0:
                frac = row / max(rows - 1, 1)
                ax.set_ylabel(f"{frac:.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.12)
    fig.savefig(output, dpi=180)
    plt.close(fig)



# Backwards-compatible name used by earlier notes/tests.
def save_flux_process_figure(*args, **kwargs) -> None:
    save_diffusion_process_figure(*args, **kwargs)

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



def _sanitize_run_name(name: str | None) -> str:
    """Return a filesystem-safe, human-readable run nickname."""
    raw = "" if name is None else str(name).strip()
    if not raw:
        return ""
    chars: list[str] = []
    previous_dash = False
    for ch in raw:
        if ch.isalnum() or ch in {"_", "-", "."}:
            chars.append(ch)
            previous_dash = False
        else:
            if not previous_dash:
                chars.append("-")
                previous_dash = True
    safe = "".join(chars).strip("-._")
    return safe[:80]


def make_experiment10_run_dir(
    runs_root: Path,
    run_name: str | None = None,
    *,
    now: datetime | None = None,
) -> tuple[Path, dict[str, object]]:
    """Create and return a unique Experiment 10 run directory.

    Run outputs live under ``runs_root``. The folder name always starts with a
    timestamp and optionally includes a sanitized nickname, so repeated runs do
    not overwrite one another.
    """
    current = datetime.now() if now is None else now
    timestamp = current.strftime("%Y%m%d-%H%M%S")
    safe_name = _sanitize_run_name(run_name)
    base_name = timestamp if not safe_name else f"{timestamp}_{safe_name}"
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / base_name
    if candidate.exists():
        for idx in range(2, 1000):
            alt = root / f"{base_name}_{idx:02d}"
            if not alt.exists():
                candidate = alt
                break
        else:
            raise RuntimeError(f"could not allocate a unique run directory under {root}")
    candidate.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": candidate.name,
        "run_name": safe_name,
        "created_at": current.isoformat(timespec="seconds"),
        "runs_root": str(root),
        "out_dir": str(candidate),
    }
    return candidate, metadata

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
    parser.add_argument("--target-free-weight", type=float, default=0.015)
    parser.add_argument("--target-noise-weight", type=float, default=0.002)
    parser.add_argument("--flux-scale", type=float, default=20.0)
    parser.add_argument("--target-flux-clip", type=float, default=10.0)
    parser.add_argument("--divergence-loss-weight", type=float, default=0.50)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("--step-loss-weight", type=float, default=0.25)
    parser.add_argument("--image-grad-loss-weight", type=float, default=0.0)
    parser.add_argument("--rollout-loss-weight", type=float, default=0.15)
    parser.add_argument("--rollout-loss-steps", type=int, default=6)
    parser.add_argument("--rollout-loss-batch-size", type=int, default=64)
    parser.add_argument("--rollout-loss-warmup-steps", type=int, default=1500)
    parser.add_argument("--rollout-loss-every", type=int, default=2)
    parser.add_argument("--rollout-loss-prob", type=float, default=1.0)
    parser.add_argument("--rollout-image-grad-loss-weight", type=float, default=0.03)
    parser.add_argument("--rollout-endpoint-l2-weight", type=float, default=0.0)
    parser.add_argument("--rollout-endpoint-bce-weight", type=float, default=0.0)
    parser.add_argument("--rollout-endpoint-tv-weight", type=float, default=0.0)
    parser.add_argument("--terminal-loss-tau-max-fraction", type=float, default=0.06)
    parser.add_argument("--terminal-loss-ramp-steps", type=int, default=3000)
    parser.add_argument("--terminal-loss-mode", choices=TERMINAL_LOSS_MODES, default="near-terminal")
    parser.add_argument("--terminal-rollout-max-steps", type=int, default=16)
    parser.add_argument("--terminal-loss-every", type=int, default=4)
    parser.add_argument("--terminal-rollout-batch-size", type=int, default=32)
    parser.add_argument("--terminal-target-mode", choices=VELOCITY_TARGET_MODES, default="mixed")
    parser.add_argument("--terminal-batch-rollout-mode", choices=("fixed", "to-zero"), default="fixed")
    parser.add_argument("--terminal-batch-prob", type=float, default=0.25)
    parser.add_argument("--terminal-batch-size", type=int, default=64)
    parser.add_argument("--terminal-tau-min-fraction", type=float, default=0.0)
    parser.add_argument("--terminal-tau-max-fraction", type=float, default=0.06)
    parser.add_argument("--hard-label-sampling", dest="hard_label_sampling", action="store_true", default=False)
    parser.add_argument("--no-hard-label-sampling", dest="hard_label_sampling", action="store_false")
    parser.add_argument("--hard-labels", type=str, default="2,5,6,7,9")
    parser.add_argument("--hard-label-prob", type=float, default=0.35)
    parser.add_argument("--target-tv-loss-weight", type=float, default=0.0)
    parser.add_argument("--target-entropy-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-classifier-loss", dest="use_classifier_loss", action="store_true", default=False)
    parser.add_argument("--no-use-classifier-loss", dest="use_classifier_loss", action="store_false")
    parser.add_argument("--classifier-loss-mode", choices=CLASSIFIER_LOSS_MODES, default="off")
    parser.add_argument("--classifier-loss-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--classifier-loss-weight", type=float, default=0.0)
    parser.add_argument("--classifier-confidence-loss-weight", type=float, default=0.0)
    parser.add_argument("--classifier-loss-blur-sigma", type=float, default=0.6)
    parser.add_argument("--terminal-shape-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-shape-entropy-weight", type=float, default=1.0)
    parser.add_argument("--terminal-shape-tv-weight", type=float, default=1.0)
    parser.add_argument("--terminal-shape-maxmass-weight", type=float, default=0.5)
    parser.add_argument("--terminal-local-shape-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-target-support-weight", type=float, default=1.0)
    parser.add_argument("--terminal-target-edge-weight", type=float, default=0.5)
    parser.add_argument("--terminal-negative-space-weight", type=float, default=0.0)
    parser.add_argument("--terminal-negative-space-mode", choices=("mean", "strict"), default="strict")
    parser.add_argument("--terminal-negative-space-threshold", type=float, default=0.08)
    parser.add_argument("--terminal-negative-space-temperature", type=float, default=0.03)
    parser.add_argument("--terminal-gap-loss-weight", type=float, default=0.0)
    parser.add_argument("--terminal-gap-threshold", type=float, default=0.12)
    parser.add_argument("--terminal-gap-dilate-radius", type=int, default=1)
    parser.add_argument("--terminal-missing-support-weight", type=float, default=0.0)
    parser.add_argument("--terminal-extra-support-weight", type=float, default=0.0)
    parser.add_argument("--terminal-extra-support-margin", type=float, default=0.10)
    parser.add_argument("--terminal-foreground-recall-weight", type=float, default=0.0)
    parser.add_argument("--terminal-foreground-threshold", type=float, default=0.18)
    parser.add_argument("--terminal-foreground-temperature", type=float, default=0.04)
    parser.add_argument("--terminal-foreground-size", type=int, default=14)
    parser.add_argument("--terminal-foreground-blur-sigma", type=float, default=0.7)
    parser.add_argument("--terminal-gap-labels", type=str, default="5,9")
    parser.add_argument("--terminal-extra-support-labels", type=str, default="5,9")
    parser.add_argument("--terminal-foreground-labels", type=str, default="2,3,6,9")
    parser.add_argument("--terminal-local-loss-max-ratio", type=float, default=0.25)
    parser.add_argument("--terminal-local-shape-size", type=int, default=14)
    parser.add_argument("--terminal-local-shape-blur-sigma", type=float, default=0.7)
    parser.add_argument("--selection-classifier-weight", type=float, default=1.0)
    parser.add_argument("--selection-entropy-weight", type=float, default=0.5)
    parser.add_argument("--selection-tv-weight", type=float, default=0.5)
    parser.add_argument("--selection-maxmass-weight", type=float, default=0.25)
    parser.add_argument("--selection-checkerboard-weight", type=float, default=0.25)
    parser.add_argument("--selection-local-support-weight", type=float, default=0.5)
    parser.add_argument("--selection-local-edge-weight", type=float, default=0.25)
    parser.add_argument("--selection-negative-space-weight", type=float, default=0.5)
    parser.add_argument("--selection-gap-weight", type=float, default=0.5)
    parser.add_argument("--selection-extra-support-weight", type=float, default=0.25)
    parser.add_argument("--selection-foreground-weight", type=float, default=0.5)
    parser.add_argument("--use-classifier-diagnostics", action="store_true")
    parser.add_argument("--classifier-cache-path", type=Path, default=None)
    parser.add_argument("--classifier-train-epochs", type=int, default=2)
    parser.add_argument("--classifier-batch-size", type=int, default=256)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--sample-rejection-factor", type=int, default=1)
    parser.add_argument("--sample-selection-metric", choices=SAMPLE_SELECTION_METRICS, default="none")
    parser.add_argument("--eval-source-batch-path", type=Path, default=None, help="Optional NPZ with fixed labels and source masses for sample-by-sample evaluation.")
    parser.add_argument("--save-eval-source-batch", action="store_true", help="Save the source masses used by the final sample grid.")
    parser.add_argument("--eval-fixed-source-seed", type=int, default=None, help="Optional seed used only for final source sampling.")
    parser.add_argument("--analyze-goodbad-file", dest="analyze_goodbad_file", action="store_true", default=True)
    parser.add_argument("--no-analyze-goodbad-file", dest="analyze_goodbad_file", action="store_false")
    parser.add_argument("--project-main-loss", action="store_true", help="Project predicted flux before the main flux/divergence losses; slower but exact for projected mode.")
    parser.add_argument("--curl-loss-weight", type=float, default=0.01)
    parser.add_argument("--edge-laplacian-loss-weight", type=float, default=0.0)
    parser.add_argument("--checkerboard-loss-weight", type=float, default=0.001)
    parser.add_argument("--state-jitter-weight", type=float, default=0.0)
    parser.add_argument("--velocity-target", choices=VELOCITY_TARGET_MODES, default="mixed")
    parser.add_argument("--min-tau-fraction", type=float, default=0.03)
    parser.add_argument("--late-residual-fraction", type=float, default=0.25)
    parser.add_argument("--late-residual-prob", type=float, default=0.50)
    parser.add_argument("--horizon-scale", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha-eff", type=float, default=1.0)
    parser.add_argument("--edge-alpha-mode", choices=EDGE_ALPHA_MODES, default="legacy")
    parser.add_argument("--upsample-mode", choices=UPSAMPLE_MODES, default="resize-conv")
    parser.add_argument("--flux-parameterization", choices=FLUX_PARAMETERIZATION_MODES, default="projected")
    parser.add_argument("--terminal-lambda", type=float, default=3.0)
    parser.add_argument("--on-policy-prob", type=float, default=0.40)
    parser.add_argument("--on-policy-warmup-steps", type=int, default=1500)
    parser.add_argument("--on-policy-prefix-steps", type=int, default=16)
    parser.add_argument("--on-policy-prefix-mode", choices=ON_POLICY_PREFIX_MODES, default="uniform")
    parser.add_argument("--on-policy-min-prefix-fraction", type=float, default=0.05)
    parser.add_argument("--on-policy-max-prefix-fraction", type=float, default=0.85)
    parser.add_argument("--on-policy-batch-size", type=int, default=64)
    parser.add_argument("--on-policy-mode", choices=ON_POLICY_MODES, default="replay")
    parser.add_argument("--on-policy-cache-size", type=int, default=2048)
    parser.add_argument("--on-policy-cache-refresh-interval", type=int, default=100)
    parser.add_argument("--on-policy-cache-rollout-batch-size", type=int, default=128)
    parser.add_argument("--on-policy-cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--on-policy-cache-mode", choices=ON_POLICY_CACHE_MODES, default="trajectory")
    parser.add_argument("--on-policy-cache-snapshots-per-traj", type=int, default=16)
    parser.add_argument("--on-policy-cache-terminal-fraction", type=float, default=0.50)
    parser.add_argument("--on-policy-cache-terminal-min-tau", type=float, default=0.00)
    parser.add_argument("--on-policy-cache-terminal-max-tau", type=float, default=0.08)
    parser.add_argument("--on-policy-target-mode", choices=ON_POLICY_TARGET_MODES, default="safe-residual")
    parser.add_argument("--on-policy-residual-max-ratio", type=float, default=1.5)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-ema-for-sampling", dest="use_ema_for_sampling", action="store_true", default=True)
    parser.add_argument("--no-use-ema-for-sampling", dest="use_ema_for_sampling", action="store_false")
    parser.add_argument("--use-ema-for-cache", dest="use_ema_for_cache", action="store_true", default=True)
    parser.add_argument("--no-use-ema-for-cache", dest="use_ema_for_cache", action="store_false")
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
    parser.add_argument("--save-process-figure", action="store_true", help="Save trajectory figures from source to terminal samples.")
    parser.add_argument("--process-num-samples", type=int, default=8)
    parser.add_argument("--process-num-frames", type=int, default=12)
    parser.add_argument("--process-mode", choices=("full_stochastic", "free_plus_conditioning_no_noise", "conditioning_only"), default="full_stochastic")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment10"), help="Root directory for Experiment 10 run folders.")
    parser.add_argument("--run-name", type=str, default="", help="Optional nickname appended to the timestamped run folder name.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Deprecated: use --runs-root and --run-name. If provided, its final path component is used as the run nickname.")
    args = parser.parse_args(argv)

    legacy_out_dir = args.out_dir
    if legacy_out_dir is not None:
        if not str(args.run_name).strip():
            args.run_name = Path(legacy_out_dir).name
        print("Note: --out-dir is deprecated for Experiment 10; writing a fresh run under --runs-root instead.")
    args.legacy_out_dir = None if legacy_out_dir is None else str(legacy_out_dir)
    args.out_dir, run_metadata = make_experiment10_run_dir(args.runs_root, args.run_name)
    args.run_id = run_metadata["run_id"]
    args.run_created_at = run_metadata["created_at"]

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
        alpha_eff=float(args.alpha_eff),
        edge_alpha_mode=str(args.edge_alpha_mode),
        horizon_scale=float(args.horizon_scale),
        num_steps=int(args.sample_steps),
        target_mode=str(args.target_mode),
        source_mode=str(args.source_mode),
        source_lowfreq_size=int(args.source_lowfreq_size),
        source_blur_sigma=float(args.source_blur_sigma),
        source_uniform_mix=float(args.source_uniform_mix),
        condition_on_source=bool(args.condition_on_source),
        upsample_mode=str(args.upsample_mode),
        flux_parameterization=str(args.flux_parameterization),
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
        image_grad_loss_weight=float(args.image_grad_loss_weight),
        rollout_loss_weight=float(args.rollout_loss_weight),
        rollout_loss_steps=int(args.rollout_loss_steps),
        rollout_loss_batch_size=int(args.rollout_loss_batch_size),
        rollout_loss_warmup_steps=int(args.rollout_loss_warmup_steps),
        rollout_loss_every=int(args.rollout_loss_every),
        rollout_loss_prob=float(args.rollout_loss_prob),
        rollout_image_grad_loss_weight=float(args.rollout_image_grad_loss_weight),
        rollout_endpoint_l2_weight=float(args.rollout_endpoint_l2_weight),
        rollout_endpoint_bce_weight=float(args.rollout_endpoint_bce_weight),
        rollout_endpoint_tv_weight=float(args.rollout_endpoint_tv_weight),
        terminal_loss_tau_max_fraction=float(args.terminal_loss_tau_max_fraction),
        terminal_loss_ramp_steps=int(args.terminal_loss_ramp_steps),
        terminal_loss_mode=str(args.terminal_loss_mode),
        terminal_rollout_max_steps=int(args.terminal_rollout_max_steps),
        terminal_loss_every=int(args.terminal_loss_every),
        terminal_rollout_batch_size=int(args.terminal_rollout_batch_size),
        terminal_target_mode=str(args.terminal_target_mode),
        terminal_batch_rollout_mode=str(args.terminal_batch_rollout_mode),
        terminal_batch_prob=float(args.terminal_batch_prob),
        terminal_batch_size=int(args.terminal_batch_size),
        terminal_tau_min_fraction=float(args.terminal_tau_min_fraction),
        terminal_tau_max_fraction=float(args.terminal_tau_max_fraction),
        hard_label_sampling=bool(args.hard_label_sampling),
        hard_labels=tuple(int(x.strip()) for x in str(args.hard_labels).split(",") if x.strip()),
        hard_label_prob=float(args.hard_label_prob),
        target_tv_loss_weight=float(args.target_tv_loss_weight),
        target_entropy_loss_weight=float(args.target_entropy_loss_weight),
        use_classifier_loss=bool(args.use_classifier_loss),
        classifier_loss_mode=str(args.classifier_loss_mode),
        classifier_loss_confidence_threshold=float(args.classifier_loss_confidence_threshold),
        classifier_loss_weight=float(args.classifier_loss_weight),
        classifier_confidence_loss_weight=float(args.classifier_confidence_loss_weight),
        classifier_loss_blur_sigma=float(args.classifier_loss_blur_sigma),
        terminal_shape_loss_weight=float(args.terminal_shape_loss_weight),
        terminal_shape_entropy_weight=float(args.terminal_shape_entropy_weight),
        terminal_shape_tv_weight=float(args.terminal_shape_tv_weight),
        terminal_shape_maxmass_weight=float(args.terminal_shape_maxmass_weight),
        terminal_local_shape_loss_weight=float(args.terminal_local_shape_loss_weight),
        terminal_target_support_weight=float(args.terminal_target_support_weight),
        terminal_target_edge_weight=float(args.terminal_target_edge_weight),
        terminal_negative_space_weight=float(args.terminal_negative_space_weight),
        terminal_negative_space_mode=str(args.terminal_negative_space_mode),
        terminal_negative_space_threshold=float(args.terminal_negative_space_threshold),
        terminal_negative_space_temperature=float(args.terminal_negative_space_temperature),
        terminal_gap_loss_weight=float(args.terminal_gap_loss_weight),
        terminal_gap_threshold=float(args.terminal_gap_threshold),
        terminal_gap_dilate_radius=int(args.terminal_gap_dilate_radius),
        terminal_missing_support_weight=float(args.terminal_missing_support_weight),
        terminal_extra_support_weight=float(args.terminal_extra_support_weight),
        terminal_extra_support_margin=float(args.terminal_extra_support_margin),
        terminal_foreground_recall_weight=float(args.terminal_foreground_recall_weight),
        terminal_foreground_threshold=float(args.terminal_foreground_threshold),
        terminal_foreground_temperature=float(args.terminal_foreground_temperature),
        terminal_foreground_size=int(args.terminal_foreground_size),
        terminal_foreground_blur_sigma=float(args.terminal_foreground_blur_sigma),
        terminal_gap_labels=tuple(int(x.strip()) for x in str(args.terminal_gap_labels).split(",") if x.strip()),
        terminal_extra_support_labels=tuple(int(x.strip()) for x in str(args.terminal_extra_support_labels).split(",") if x.strip()),
        terminal_foreground_labels=tuple(int(x.strip()) for x in str(args.terminal_foreground_labels).split(",") if x.strip()),
        terminal_local_loss_max_ratio=float(args.terminal_local_loss_max_ratio),
        terminal_local_shape_size=int(args.terminal_local_shape_size),
        terminal_local_shape_blur_sigma=float(args.terminal_local_shape_blur_sigma),
        selection_classifier_weight=float(args.selection_classifier_weight),
        selection_entropy_weight=float(args.selection_entropy_weight),
        selection_tv_weight=float(args.selection_tv_weight),
        selection_maxmass_weight=float(args.selection_maxmass_weight),
        selection_checkerboard_weight=float(args.selection_checkerboard_weight),
        selection_local_support_weight=float(args.selection_local_support_weight),
        selection_local_edge_weight=float(args.selection_local_edge_weight),
        selection_negative_space_weight=float(args.selection_negative_space_weight),
        selection_gap_weight=float(args.selection_gap_weight),
        selection_extra_support_weight=float(args.selection_extra_support_weight),
        selection_foreground_weight=float(args.selection_foreground_weight),
        use_classifier_diagnostics=bool(args.use_classifier_diagnostics),
        classifier_train_epochs=int(args.classifier_train_epochs),
        classifier_cache_path="" if args.classifier_cache_path is None else str(args.classifier_cache_path),
        classifier_batch_size=int(args.classifier_batch_size),
        classifier_lr=float(args.classifier_lr),
        sample_rejection_factor=int(args.sample_rejection_factor),
        sample_selection_metric=str(args.sample_selection_metric),
        analyze_goodbad_file=bool(args.analyze_goodbad_file),
        project_main_loss=bool(args.project_main_loss),
        curl_loss_weight=float(args.curl_loss_weight),
        edge_laplacian_loss_weight=float(args.edge_laplacian_loss_weight),
        checkerboard_loss_weight=float(args.checkerboard_loss_weight),
        state_jitter_weight=float(args.state_jitter_weight),
        velocity_target=str(args.velocity_target),
        min_tau_fraction=float(args.min_tau_fraction),
        late_residual_fraction=float(args.late_residual_fraction),
        late_residual_prob=float(args.late_residual_prob),
        terminal_lambda=float(args.terminal_lambda),
        on_policy_prob=float(args.on_policy_prob),
        on_policy_warmup_steps=int(args.on_policy_warmup_steps),
        on_policy_prefix_steps=int(args.on_policy_prefix_steps),
        on_policy_prefix_mode=str(args.on_policy_prefix_mode),
        on_policy_min_prefix_fraction=float(args.on_policy_min_prefix_fraction),
        on_policy_max_prefix_fraction=float(args.on_policy_max_prefix_fraction),
        on_policy_batch_size=int(args.on_policy_batch_size),
        on_policy_mode=str(args.on_policy_mode),
        on_policy_cache_size=int(args.on_policy_cache_size),
        on_policy_cache_refresh_interval=int(args.on_policy_cache_refresh_interval),
        on_policy_cache_rollout_batch_size=int(args.on_policy_cache_rollout_batch_size),
        on_policy_cache_device=str(args.on_policy_cache_device),
        on_policy_cache_mode=str(args.on_policy_cache_mode),
        on_policy_cache_snapshots_per_traj=int(args.on_policy_cache_snapshots_per_traj),
        on_policy_cache_terminal_fraction=float(args.on_policy_cache_terminal_fraction),
        on_policy_cache_terminal_min_tau=float(args.on_policy_cache_terminal_min_tau),
        on_policy_cache_terminal_max_tau=float(args.on_policy_cache_terminal_max_tau),
        on_policy_target_mode=str(args.on_policy_target_mode),
        on_policy_residual_max_ratio=float(args.on_policy_residual_max_ratio),
        ema_decay=float(args.ema_decay),
        use_ema_for_sampling=bool(args.use_ema_for_sampling),
        use_ema_for_cache=bool(args.use_ema_for_cache),
        adaptive_sampling=bool(args.adaptive_sampling),
        clip_target=float(args.clip_target),
        max_substeps=int(args.max_substeps),
    )
    device = torch.device(
        "cuda" if args.device is None and torch.cuda.is_available() else "cpu" if args.device is None else args.device
    )
    print(f"Experiment 10k direct-flux MNIST on device={device}")
    print(f"Run directory: {args.out_dir}")
    if float(config.rollout_endpoint_tv_weight) > 0.0:
        print(
            "Warning: --rollout-endpoint-tv-weight is enabled. 10l gates and normalizes this loss, "
            "but raw TV matching can still encourage high-frequency texture; prefer 0 for baseline runs."
        )
    if (float(config.classifier_loss_weight) > 0.0 or float(config.classifier_confidence_loss_weight) > 0.0) and not bool(config.use_classifier_loss):
        print(
            "Note: classifier loss weights were provided, but --use-classifier-loss was not set. "
            "The classifier will be used only for diagnostics/sample selection."
        )
    print(
        "Laptop-friendly settings: "
        f"target_mode={config.target_mode}, source_mode={config.source_mode}, "
        f"ot={config.ot_cost_mode}/{config.ot_lowres_size}, match={config.ot_match_mode}, tau={config.tau_sampling}, "
        f"source_cond={config.condition_on_source}, upsample={config.upsample_mode}, flux_param={config.flux_parameterization}, velocity={config.velocity_target}, "
        f"free_aware={config.free_aware_target}, sde_curr={config.sde_curriculum}, edge_alpha={config.edge_alpha_mode}, "
        f"on_policy={config.on_policy_prob}, onp_mode={config.on_policy_mode}/{config.on_policy_cache_mode}, cache={config.on_policy_cache_size}/{config.on_policy_cache_refresh_interval}/snap{config.on_policy_cache_snapshots_per_traj}, onp_target={config.on_policy_target_mode}, ema={config.ema_decay}, onp_sde=({config.on_policy_use_free},{config.on_policy_use_noise}), "
        f"step_loss={config.step_loss_weight}, rollout={config.rollout_loss_weight}/{config.rollout_loss_steps}/every{config.rollout_loss_every}, rollout_img={config.rollout_image_grad_loss_weight}, terminal_tau<={config.terminal_loss_tau_max_fraction}, shape={config.terminal_shape_loss_weight}, local_shape={config.terminal_local_shape_loss_weight}, cls_loss={config.use_classifier_loss}/{config.classifier_loss_weight}, project_main={config.project_main_loss}, curl={config.curl_loss_weight}, stochastic_step={config.stochastic_step_loss}, adaptive={config.adaptive_sampling}, "
        f"train_steps={args.train_steps}, batch={args.batch_size}, base_channels={args.base_channels}, "
        f"horizon={natural_horizon(config):.3e}, sample_steps={args.sample_steps}, "
        f"weights=(free={config.free_weight}, noise={config.noise_weight}, learned={config.learned_weight})"
    )
    run_metadata["args"] = _serializable_args(args)
    run_metadata["config"] = asdict(config)
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
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
    classifier_needed = (
        bool(config.use_classifier_diagnostics)
        or ((bool(config.use_classifier_loss) or str(config.classifier_loss_mode) != "off") and (float(config.classifier_loss_weight) > 0.0 or float(config.classifier_confidence_loss_weight) > 0.0))
        or int(config.sample_rejection_factor) > 1
    )
    terminal_classifier: TinyMNISTClassifier | None = None
    if classifier_needed:
        cls_cache = Path(config.classifier_cache_path) if str(config.classifier_cache_path) else args.out_dir / "experiment10_mnist_classifier.pt"
        print(f"Preparing terminal MNIST classifier: {cls_cache}")
        terminal_classifier = train_or_load_mnist_classifier(
            dataset.train_images,
            dataset.train_labels,
            grid_size=int(config.grid_size),
            cache_path=cls_cache,
            train_epochs=int(config.classifier_train_epochs),
            batch_size=int(config.classifier_batch_size),
            lr=float(config.classifier_lr),
            device=device,
            seed=int(args.seed) + 17,
            show_progress=not bool(args.no_progress),
        )
    final_class_means = _compute_class_mean_measures(dataset.train_images, dataset.train_labels, config.grid_size)
    class_shape_stats = compute_class_shape_statistics(
        dataset.train_images,
        dataset.train_labels,
        grid_size=int(config.grid_size),
        local_shape_size=max(2, min(int(config.terminal_local_shape_size), int(config.grid_size))),
        local_blur_sigma=float(config.terminal_local_shape_blur_sigma),
    )
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
        terminal_classifier=terminal_classifier,
        class_shape_stats=class_shape_stats,
    )

    print("Training complete; starting generation")
    labels = _parse_label_sequence(args.labels, int(args.num_samples))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_initial_states: np.ndarray | None = None
    if args.eval_source_batch_path is not None and Path(args.eval_source_batch_path).exists():
        fixed = np.load(Path(args.eval_source_batch_path), allow_pickle=True)
        eval_initial_states = np.asarray(fixed["sources"], dtype=np.float32).reshape(-1, int(config.grid_size * config.grid_size))
        labels = np.asarray(fixed["labels"], dtype=np.int64).reshape(-1)
        print(f"Loaded fixed evaluation source batch: {args.eval_source_batch_path} ({labels.shape[0]} samples)")
    candidate_labels = labels
    candidate_initial_states = eval_initial_states
    selection_factor = int(config.sample_rejection_factor)
    if selection_factor > 1:
        if terminal_classifier is None or str(config.sample_selection_metric) not in {"classifier-confidence", "composite", "composite-local", "composite-gap"}:
            raise RuntimeError("sample rejection requires classifier diagnostics and classifier-confidence/composite/composite-local/composite-gap selection")
        candidate_labels = np.repeat(labels, selection_factor)
        if eval_initial_states is not None:
            candidate_initial_states = np.repeat(eval_initial_states, selection_factor, axis=0)
        print(f"Generating {selection_factor} {config.sample_selection_metric}-ranked candidates per requested sample")
    sample_seed = int(args.eval_fixed_source_seed) if args.eval_fixed_source_seed is not None else int(args.seed) + 1
    result = simulate_direct_flux_generation(
        model,
        candidate_labels,
        num_steps=int(args.sample_steps),
        deterministic=bool(args.deterministic_sampling),
        device=device,
        seed=sample_seed,
        use_amp=not bool(args.no_amp),
        show_progress=not bool(args.no_progress),
        initial_states=candidate_initial_states,
        source_images=dataset.train_images,
        source_labels=dataset.train_labels,
    )
    if selection_factor > 1:
        raw_png = args.out_dir / "experiment10_samples_raw.png"
        try:
            save_flux_samples_grid(result.samples, result.labels, raw_png, grid_size=config.grid_size)
            print(f"Saved raw candidate preview: {raw_png}")
        except RuntimeError as exc:
            print(f"Skipping raw candidate PNG: {exc}")
        assert terminal_classifier is not None
        result = select_generation_result_by_classifier(
            result,
            labels,
            factor=selection_factor,
            classifier=terminal_classifier,
            grid_size=int(config.grid_size),
            device=device,
            selection_metric=str(config.sample_selection_metric),
            shape_stats=class_shape_stats,
            config=config,
            report_path=args.out_dir / "experiment10_selection_report.csv",
        )

    if bool(args.save_eval_source_batch) and result.sources is not None:
        eval_path = args.out_dir / "experiment10_eval_source_batch.npz"
        np.savez_compressed(eval_path, labels=result.labels, sources=result.sources)
        print(f"Saved fixed evaluation source batch: {eval_path}")

    print("Generation complete; saving run outputs")
    ckpt_path = args.out_dir / "experiment10_direct_flux_mnist.pt"
    samples_path = args.out_dir / "experiment10_samples.npz"
    png_path = args.out_dir / "experiment10_samples.png"
    final_metrics = nearest_class_mean_metrics(result.samples, result.labels, final_class_means)
    classifier_metrics = classifier_generation_metrics(
        result.samples,
        result.labels,
        terminal_classifier,
        grid_size=int(config.grid_size),
        device=device,
    )
    local_final_metrics = local_shape_metrics_np(
        result.samples,
        result.labels,
        class_shape_stats,
        grid_size=int(config.grid_size),
        negative_space_mode=str(config.terminal_negative_space_mode),
        negative_space_threshold=float(config.terminal_negative_space_threshold),
        negative_space_temperature=float(config.terminal_negative_space_temperature),
        gap_threshold=float(config.terminal_gap_threshold),
        gap_dilate_radius=int(config.terminal_gap_dilate_radius),
        extra_support_margin=float(config.terminal_extra_support_margin),
    )
    goodbad_metrics: dict[str, float | np.ndarray] = {}
    goodbad_path = args.out_dir / "samples_goodbad.txt"
    if bool(config.analyze_goodbad_file) and goodbad_path.exists():
        goodbad_metrics = analyze_goodbad_annotations(
            goodbad_path,
            result.samples,
            result.labels,
            classifier_metrics=classifier_metrics,
        )
        serializable_goodbad = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in goodbad_metrics.items()
        }
        (args.out_dir / "samples_goodbad_analysis.json").write_text(
            json.dumps(serializable_goodbad, indent=2),
            encoding="utf-8",
        )
        write_goodbad_sample_report(
            args.out_dir / "experiment10_goodbad_report.csv",
            goodbad_path,
            result.samples,
            result.labels,
            classifier_metrics=classifier_metrics,
            grid_size=int(config.grid_size),
        )
    write_local_shape_report(
        args.out_dir / "experiment10_local_shape_report.csv",
        result.samples,
        result.labels,
        class_shape_stats,
        classifier_metrics=classifier_metrics,
        grid_size=int(config.grid_size),
        negative_space_mode=str(config.terminal_negative_space_mode),
        negative_space_threshold=float(config.terminal_negative_space_threshold),
        negative_space_temperature=float(config.terminal_negative_space_temperature),
        gap_threshold=float(config.terminal_gap_threshold),
        gap_dilate_radius=int(config.terminal_gap_dilate_radius),
        extra_support_margin=float(config.terminal_extra_support_margin),
    )
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
            "run_metadata": run_metadata,
            "history": history,
            "labels": result.labels,
            "clipping_fraction": result.clipping_fraction,
            "final_metrics": final_metrics,
            "classifier_metrics": {k: v for k, v in classifier_metrics.items() if not isinstance(v, np.ndarray)},
            "class_shape_stats": {k: v.tolist() for k, v in class_shape_stats.items()},
            "goodbad_metrics": {k: v for k, v in goodbad_metrics.items() if not isinstance(v, np.ndarray)},
            "source_metrics": source_metrics,
            "component_metrics": {
                "learned_step_rms": result.learned_step_rms,
                "free_step_rms": result.free_step_rms,
                "noise_step_rms": result.noise_step_rms,
                "free_to_learned_ratio": result.free_to_learned_ratio,
                "noise_to_learned_ratio": result.noise_to_learned_ratio,
            },
            "sample_quality_metrics": {
                "sample_entropy": result.sample_entropy,
                "sample_total_variation": result.sample_total_variation,
                "sample_checkerboard_energy": result.sample_checkerboard_energy,
                "sample_highfreq_fraction": result.sample_highfreq_fraction,
            },
        },
        ckpt_path,
    )
    np.savez_compressed(
        samples_path,
        run_id=np.asarray([str(args.run_id)]),
        run_name=np.asarray([str(run_metadata.get("run_name", ""))]),
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
        classifier_acc=np.asarray([float(classifier_metrics.get("classifier_acc", np.nan))], dtype=np.float64),
        classifier_confidence=np.asarray([float(classifier_metrics.get("classifier_confidence", np.nan))], dtype=np.float64),
        classifier_margin=np.asarray([float(classifier_metrics.get("classifier_margin", np.nan))], dtype=np.float64),
        classifier_predictions=np.asarray(classifier_metrics.get("classifier_predictions", []), dtype=np.int64),
        classifier_target_probs=np.asarray(classifier_metrics.get("classifier_target_probs", []), dtype=np.float64),
        classifier_margins=np.asarray(classifier_metrics.get("classifier_margins", []), dtype=np.float64),
        class_entropy_q75=np.asarray(class_shape_stats.get("entropy_q75", []), dtype=np.float64),
        class_tv_q25=np.asarray(class_shape_stats.get("tv_q25", []), dtype=np.float64),
        class_maxmass_q25=np.asarray(class_shape_stats.get("maxmass_q25", []), dtype=np.float64),
        local_support_loss=np.asarray(local_final_metrics.get("local_support_loss", []), dtype=np.float64),
        local_edge_loss=np.asarray(local_final_metrics.get("local_edge_loss", []), dtype=np.float64),
        negative_space_mass=np.asarray(local_final_metrics.get("negative_space_mass", []), dtype=np.float64),
        strict_negative_space_mass=np.asarray(local_final_metrics.get("strict_negative_space_mass", []), dtype=np.float64),
        gap_mass=np.asarray(local_final_metrics.get("gap_mass", []), dtype=np.float64),
        missing_support_loss=np.asarray(local_final_metrics.get("missing_support_loss", []), dtype=np.float64),
        extra_support_loss=np.asarray(local_final_metrics.get("extra_support_loss", []), dtype=np.float64),
        class_local_support_mean=np.asarray(class_shape_stats.get("local_support_mean", []), dtype=np.float64),
        class_local_support_q90=np.asarray(class_shape_stats.get("local_support_q90", []), dtype=np.float64),
        class_local_edge_mean=np.asarray(class_shape_stats.get("local_edge_mean", []), dtype=np.float64),
        class_local_negative_space=np.asarray(class_shape_stats.get("local_negative_space", []), dtype=np.float64),
        human_good_rate=np.asarray([float(goodbad_metrics.get("human_good_rate", np.nan))], dtype=np.float64),
        human_good_by_label=np.asarray(goodbad_metrics.get("human_good_by_label", []), dtype=np.float64),
        human_bad_count_by_label=np.asarray(goodbad_metrics.get("human_bad_count_by_label", []), dtype=np.int64),
        learned_step_rms=np.asarray([0.0 if result.learned_step_rms is None else result.learned_step_rms], dtype=np.float64),
        free_step_rms=np.asarray([0.0 if result.free_step_rms is None else result.free_step_rms], dtype=np.float64),
        noise_step_rms=np.asarray([0.0 if result.noise_step_rms is None else result.noise_step_rms], dtype=np.float64),
        free_to_learned_ratio=np.asarray([0.0 if result.free_to_learned_ratio is None else result.free_to_learned_ratio], dtype=np.float64),
        noise_to_learned_ratio=np.asarray([0.0 if result.noise_to_learned_ratio is None else result.noise_to_learned_ratio], dtype=np.float64),
        sample_entropy=np.asarray([0.0 if result.sample_entropy is None else result.sample_entropy], dtype=np.float64),
        sample_total_variation=np.asarray([0.0 if result.sample_total_variation is None else result.sample_total_variation], dtype=np.float64),
        sample_checkerboard_energy=np.asarray([0.0 if result.sample_checkerboard_energy is None else result.sample_checkerboard_energy], dtype=np.float64),
        sample_highfreq_fraction=np.asarray([0.0 if result.sample_highfreq_fraction is None else result.sample_highfreq_fraction], dtype=np.float64),
    )
    try:
        save_flux_samples_grid(result.samples, result.labels, png_path, grid_size=config.grid_size)
        print(f"Saved preview: {png_path}")
    except RuntimeError as exc:
        print(f"Skipping PNG preview: {exc}")

    if bool(args.save_process_figure) and result.sources is not None:
        process_count = min(int(args.process_num_samples), int(result.samples.shape[0]))
        frame_count = max(2, int(args.process_num_frames))
        save_every = max(1, int(args.sample_steps) // max(frame_count - 1, 1))
        if str(args.process_mode) == "conditioning_only":
            process_overrides = dict(free_weight=0.0, noise_weight=0.0, learned_weight=1.0, deterministic=True)
        elif str(args.process_mode) == "free_plus_conditioning_no_noise":
            process_overrides = dict(free_weight=config.free_weight, noise_weight=0.0, learned_weight=1.0, deterministic=True)
        else:
            process_overrides = dict(free_weight=config.free_weight, noise_weight=config.noise_weight, learned_weight=1.0, deterministic=bool(args.deterministic_sampling))
        process = simulate_direct_flux_generation(
            model,
            result.labels[:process_count],
            num_steps=int(args.sample_steps),
            save_every=save_every,
            deterministic=bool(process_overrides.pop("deterministic")),
            device=device,
            seed=int(args.seed) + 300,
            use_amp=not bool(args.no_amp),
            show_progress=False,
            initial_states=result.sources[:process_count],
            free_weight=float(process_overrides["free_weight"]),
            noise_weight=float(process_overrides["noise_weight"]),
            learned_weight=float(process_overrides["learned_weight"]),
        )
        if process.trajectory is not None:
            process_npz = args.out_dir / "experiment10_diffusion_process.npz"
            np.savez_compressed(process_npz, trajectory=process.trajectory, labels=process.labels, sources=process.sources, samples=process.samples)
            try:
                process_png = args.out_dir / "experiment10_diffusion_process.png"
                marginal_png = args.out_dir / "experiment10_diffusion_marginal_process.png"
                save_diffusion_process_figure(
                    process.trajectory,
                    process.labels,
                    process_png,
                    grid_size=config.grid_size,
                    num_frames=frame_count,
                    max_samples=process_count,
                )
                save_diffusion_marginal_process_figure(
                    process.trajectory,
                    process.labels,
                    marginal_png,
                    grid_size=config.grid_size,
                    num_frames=min(frame_count, 8),
                    samples_per_frame=process_count,
                )
                print(f"Saved diffusion process figures: {process_png}, {marginal_png}")
            except RuntimeError as exc:
                print(f"Skipping diffusion process PNGs: {exc}")

    if bool(args.save_ablation_samples) and result.sources is not None:
        ablations = [
            ("conditioning_only", dict(free_weight=0.0, noise_weight=0.0, learned_weight=1.0, deterministic=True)),
            ("free_plus_conditioning_no_noise", dict(free_weight=config.free_weight, noise_weight=0.0, learned_weight=1.0, deterministic=True)),
            ("full_stochastic", dict(free_weight=config.free_weight, noise_weight=config.noise_weight, learned_weight=1.0, deterministic=False)),
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
        "Sample artifact diagnostics: "
        f"TV={0.0 if result.sample_total_variation is None else result.sample_total_variation:.4g}, "
        f"checker={0.0 if result.sample_checkerboard_energy is None else result.sample_checkerboard_energy:.4g}, "
        f"highfreq={0.0 if result.sample_highfreq_fraction is None else result.sample_highfreq_fraction:.4g}"
    )
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
    if classifier_metrics:
        print(
            "Classifier diagnostics: "
            f"acc={float(classifier_metrics.get('classifier_acc', float('nan'))):.3f}, "
            f"conf={float(classifier_metrics.get('classifier_confidence', float('nan'))):.3f}, "
            f"margin={float(classifier_metrics.get('classifier_margin', float('nan'))):.3f}"
        )
    if goodbad_metrics:
        print(
            "Good/bad annotation diagnostics: "
            f"good_rate={float(goodbad_metrics.get('human_good_rate', float('nan'))):.3f}, "
            f"entropy_good={float(goodbad_metrics.get('entropy_good_mean', float('nan'))):.3f}, "
            f"entropy_bad={float(goodbad_metrics.get('entropy_bad_mean', float('nan'))):.3f}, "
            f"tv_good={float(goodbad_metrics.get('tv_good_mean', float('nan'))):.3f}, "
            f"tv_bad={float(goodbad_metrics.get('tv_bad_mean', float('nan'))):.3f}"
        )


if __name__ == "__main__":
    main()
