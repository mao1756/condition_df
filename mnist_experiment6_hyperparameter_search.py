from __future__ import annotations

"""Random-search utilities for Example 6 (MNIST weighted point-cloud generation).

The notebook in ``notebooks/example_6_mnist_weighted_point_cloud_generation.ipynb``
exposes many interacting hyperparameters:

* terminal-network architecture and noisy augmentation,
* h-transform simulation horizon / discretization,
* Monte Carlo sample count and guidance scale,
* Poisson--Dirichlet mass concentration,
* diffusion temperature and drift clipping.

This module adds a lightweight two-stage search procedure that keeps the user's
"random masses + random positions" start intact by default:

* masses are still sampled from a truncated Poisson--Dirichlet law,
* initial positions are still sampled uniformly on the unit square.

The search proceeds in two stages.

1. **Terminal stage.** Sample terminal-classifier hyperparameters, train on the
   real MNIST point clouds, and rank trials using a validation score that mixes
   clean and diffusion-noised accuracy.
2. **Generation stage.** For the best few terminal models, sample generation
   hyperparameters and rank them first with a cheap proxy (``g``-accuracy and
   target probability), then optionally re-evaluate the top few with the full
   Example-6 metrics.

The functions are notebook-friendly but also expose a small CLI.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import argparse
import copy
import json
import math
import random

import numpy as np
import pandas as pd
import torch

from mnist_weighted_point_cloud import (
    WeightedPointCloudBatch,
    images_to_weighted_point_clouds,
    load_mnist_arrays,
)
from mnist_conditioned_diffusion import (
    TerminalSetClassifier,
    evaluate_generation_metrics,
    evaluate_terminal_set_classifier,
    project_positions,
    terminal_g_accuracy,
)
from mnist_experiment6_fixes import (
    generate_balanced_synthetic_dataset_reparam,
    train_terminal_set_classifier_noisy,
)


SearchSpace = Mapping[str, Any]

__all__ = [
    "PreparedMnistExperiment6Data",
    "Experiment6SearchConfig",
    "Experiment6SearchResult",
    "prepare_mnist_experiment6_data",
    "default_terminal_search_space",
    "default_generation_search_space",
    "run_experiment6_random_search",
]


# ---------------------------------------------------------------------------
# Small containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedMnistExperiment6Data:
    """Real MNIST splits converted to weighted point clouds."""

    train_pc: WeightedPointCloudBatch
    val_pc: WeightedPointCloudBatch
    test_pc: WeightedPointCloudBatch
    real_test_images: np.ndarray
    real_test_labels: np.ndarray
    top_k: int
    mass_floor: float

    @property
    def num_classes(self) -> int:
        labels = self.train_pc.labels
        if labels is None:
            raise ValueError("train_pc.labels are required")
        return int(np.max(labels) + 1)


@dataclass(frozen=True)
class Experiment6SearchConfig:
    """Global knobs for the two-stage random search."""

    seed: int = 1234
    device: Optional[str] = None

    train_per_class: int = 1500
    val_per_class: int = 300
    test_per_class: int = 500
    top_k: int = 32
    mass_floor: float = 1e-4

    terminal_trials: int = 12
    keep_top_terminal: int = 3
    generation_trials_per_terminal: int = 8
    final_eval_top_k: int = 3

    synthetic_per_class_proxy: int = 64
    synthetic_per_class_final: int = 256

    terminal_noise_eval_repeats: int = 2
    terminal_noise_eval_projection: str = "reflect"
    terminal_noise_horizon_reference: float = 1e-4

    generation_batch_size: int = 64

    proxy_score_weight_g_accuracy: float = 0.75
    proxy_score_weight_target_probability: float = 0.25

    final_score_weight_g_accuracy: float = 0.40
    final_score_weight_target_probability: float = 0.15
    final_score_weight_cas: float = 0.25
    final_score_weight_coverage: float = 0.10
    final_score_weight_two_sample: float = 0.10

    cas_epochs_final: int = 8
    cas_batch_size: int = 128
    cas_lr: float = 1e-3

    sinkhorn_subsample_per_class_final: int = 32
    sinkhorn_epsilon: float = 0.02
    sinkhorn_iterations_final: int = 40

    verbose: bool = True


@dataclass
class Experiment6SearchResult:
    """Return object for the random search."""

    prepared_data: PreparedMnistExperiment6Data
    terminal_trials: pd.DataFrame
    generation_proxy_trials: pd.DataFrame
    generation_final_trials: pd.DataFrame
    best_terminal_params: dict[str, Any]
    best_generation_params: dict[str, Any]
    best_terminal_metrics: dict[str, Any]
    best_generation_metrics: dict[str, Any]
    best_terminal_model: TerminalSetClassifier


@dataclass
class _TerminalCandidate:
    trial_id: int
    seed: int
    params: dict[str, Any]
    metrics: dict[str, Any]
    score: float
    model: TerminalSetClassifier


@dataclass
class _GenerationCandidate:
    proxy_trial_id: int
    terminal_trial_id: int
    seed: int
    params: dict[str, Any]
    proxy_metrics: dict[str, Any]
    proxy_score: float


# ---------------------------------------------------------------------------
# Search-space helpers
# ---------------------------------------------------------------------------


def choice(*values: Any) -> dict[str, Any]:
    return {"type": "choice", "values": list(values)}


def uniform(low: float, high: float) -> dict[str, Any]:
    return {"type": "uniform", "low": float(low), "high": float(high)}


def loguniform(low: float, high: float) -> dict[str, Any]:
    return {"type": "loguniform", "low": float(low), "high": float(high)}


def randint(low: int, high: int) -> dict[str, Any]:
    return {"type": "randint", "low": int(low), "high": int(high)}


def default_terminal_search_space() -> dict[str, Any]:
    """Default random-search space for the terminal classifier."""
    return {
        "point_feature_dim": choice(96, 128, 160, 192),
        "hidden_dim": choice(192, 256, 320, 384),
        "dropout": choice(0.0, 0.05, 0.10, 0.15),
        "epochs": choice(12, 16, 20, 24),
        "batch_size": choice(128, 256),
        "lr": loguniform(5e-4, 3e-3),
        "weight_decay": loguniform(1e-6, 5e-4),
        "position_jitter_std": choice(1e-3, 2e-3, 3e-3, 5e-3),
        "max_tau_factor": choice(0.5, 1.0, 2.0),
        "tau_sampling": choice("uniform", "quadratic_bias_to_zero"),
    }


def default_generation_search_space(*, top_k: int) -> dict[str, Any]:
    """Default random-search space for the guided generator.

    The defaults deliberately preserve the notebook's random initialization:
    ``mass_sampling_mode='truncated_poisson_dirichlet'`` and
    ``initial_position_mode='uniform'`` are treated as fixed constants rather
    than search dimensions.
    """
    return {
        "horizon": choice(2.5e-5, 5.0e-5, 1.0e-4, 2.0e-4),
        "num_steps": choice(64, 96, 128, 192),
        "terminal_mc_samples": choice(64, 96, 128, 192, 256),
        "guidance_scale": uniform(2.0, 6.5),
        "diffusion_temperature": choice(0.25, 0.5, 0.75, 1.0),
        "drift_clip_pixels": choice(1.0, 2.0, 3.0, 4.0, 6.0),
        "poisson_dirichlet_beta_factor": choice(1.0, 2.0, 4.0, 8.0),
        "poisson_dirichlet_max_terms_factor": choice(4, 8, 12, 16),
        "mass_sampling_mode": "truncated_poisson_dirichlet",
        "class_conditional_mass_sampling": False,
        "initial_position_mode": "uniform",
        "initial_position_scale": 0.12,
        "initial_position_jitter": 0.0,
        "joint_bank_sampling": False,
        "state_projection": "reflect",
        "terminal_projection": "reflect",
        # Keeping K fixed by default makes the search cheaper and easier to compare.
        "num_points": int(top_k),
    }


def _sample_spec(spec: Any, rng: np.random.Generator) -> Any:
    if isinstance(spec, dict) and "type" in spec:
        kind = spec["type"]
        if kind == "choice":
            values = list(spec["values"])
            return values[int(rng.integers(0, len(values)))]
        if kind == "uniform":
            return float(rng.uniform(spec["low"], spec["high"]))
        if kind == "loguniform":
            lo = math.log(float(spec["low"]))
            hi = math.log(float(spec["high"]))
            return float(math.exp(rng.uniform(lo, hi)))
        if kind == "randint":
            return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
        raise ValueError(f"unknown search-space spec type: {kind!r}")
    if isinstance(spec, (list, tuple)):
        values = list(spec)
        return values[int(rng.integers(0, len(values)))]
    return spec


def _sample_params(space: SearchSpace, rng: np.random.Generator) -> dict[str, Any]:
    return {key: _sample_spec(value, rng) for key, value in space.items()}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _class_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(np.max(labels)) + 1)


def _balanced_take(
    images: np.ndarray,
    labels: np.ndarray,
    per_class: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    selected: list[np.ndarray] = []
    for label in range(int(np.max(labels)) + 1):
        idx = np.flatnonzero(labels == label)
        if len(idx) < per_class:
            raise ValueError(
                f"label {label} only has {len(idx)} examples, need {per_class}. "
                f"Max balanced choice is {int(_class_counts(labels).min())}."
            )
        selected.append(rng.choice(idx, size=per_class, replace=False))
    selected_idx = np.concatenate(selected)
    rng.shuffle(selected_idx)
    return images[selected_idx], labels[selected_idx]


def _balanced_train_val_split(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    for label in range(int(np.max(labels)) + 1):
        idx = np.flatnonzero(labels == label)
        need = train_per_class + val_per_class
        if len(idx) < need:
            raise ValueError(
                f"label {label} only has {len(idx)} examples, need {need}."
            )
        chosen = rng.choice(idx, size=need, replace=False)
        train_indices.append(chosen[:train_per_class])
        val_indices.append(chosen[train_per_class:])
    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def prepare_mnist_experiment6_data(
    *,
    data_root: str | Path,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    top_k: int,
    mass_floor: float,
    seed: int,
    download: bool = True,
) -> PreparedMnistExperiment6Data:
    """Load MNIST, build balanced splits, and convert them to point clouds."""
    mnist = load_mnist_arrays(data_root, download=download, normalize_to_measure=True)

    train_images, train_labels, val_images, val_labels = _balanced_train_val_split(
        np.asarray(mnist["train_images"], dtype=np.float64),
        np.asarray(mnist["train_labels"], dtype=np.int64),
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        seed=seed,
    )
    test_images, test_labels = _balanced_take(
        np.asarray(mnist["test_images"], dtype=np.float64),
        np.asarray(mnist["test_labels"], dtype=np.int64),
        test_per_class,
        seed=seed + 1,
    )

    train_pc = images_to_weighted_point_clouds(
        train_images,
        labels=train_labels,
        top_k=top_k,
        mass_floor=mass_floor,
        normalize_to_measure=True,
    )
    val_pc = images_to_weighted_point_clouds(
        val_images,
        labels=val_labels,
        top_k=top_k,
        mass_floor=mass_floor,
        normalize_to_measure=True,
    )
    test_pc = images_to_weighted_point_clouds(
        test_images,
        labels=test_labels,
        top_k=top_k,
        mass_floor=mass_floor,
        normalize_to_measure=True,
    )

    return PreparedMnistExperiment6Data(
        train_pc=train_pc,
        val_pc=val_pc,
        test_pc=test_pc,
        real_test_images=np.asarray(test_images, dtype=np.float64),
        real_test_labels=np.asarray(test_labels, dtype=np.int64),
        top_k=int(top_k),
        mass_floor=float(mass_floor),
    )


# ---------------------------------------------------------------------------
# Trial helpers
# ---------------------------------------------------------------------------


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return str(device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _flatten_trial_record(prefix: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            for subkey, subvalue in value.items():
                out[f"{prefix}{key}.{subkey}"] = subvalue
        else:
            out[f"{prefix}{key}"] = value
    return out


def _make_noisy_positions(
    masses: np.ndarray,
    positions: np.ndarray,
    *,
    tau: float,
    repeats: int,
    projection: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    masses_arr = np.asarray(masses, dtype=np.float64)
    positions_arr = np.asarray(positions, dtype=np.float64)
    if tau <= 0.0:
        return np.repeat(positions_arr, repeats, axis=0)
    sigma = np.sqrt((2.0 * tau) / masses_arr)[..., None]
    out = []
    for _ in range(repeats):
        noisy = positions_arr + sigma * rng.normal(size=positions_arr.shape)
        noisy = np.asarray(project_positions(noisy, mode=projection), dtype=np.float64)
        out.append(noisy)
    return np.concatenate(out, axis=0)


def _evaluate_terminal_with_noise(
    model: TerminalSetClassifier,
    masses: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    tau: float,
    repeats: int,
    projection: str,
    device: str,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    repeated_masses = np.repeat(np.asarray(masses, dtype=np.float64), repeats, axis=0)
    repeated_labels = np.repeat(np.asarray(labels, dtype=np.int64), repeats, axis=0)
    noisy_positions = _make_noisy_positions(
        masses,
        positions,
        tau=tau,
        repeats=repeats,
        projection=projection,
        rng=rng,
    )
    metrics = evaluate_terminal_set_classifier(
        model,
        repeated_masses,
        noisy_positions,
        repeated_labels,
        device=device,
    )
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
    }


def _terminal_score(clean_val_accuracy: float, noisy_val_accuracy: float) -> float:
    return 0.35 * float(clean_val_accuracy) + 0.65 * float(noisy_val_accuracy)


def _proxy_generation_score(
    *,
    g_accuracy: float,
    g_mean_target_probability: float,
    config: Experiment6SearchConfig,
) -> float:
    return (
        config.proxy_score_weight_g_accuracy * float(g_accuracy)
        + config.proxy_score_weight_target_probability * float(g_mean_target_probability)
    )


def _two_sample_target_score(one_nn_accuracy_macro: float) -> float:
    # 1.0 is best (exactly 0.5), 0.0 is worst (exactly 0 or 1).
    return max(0.0, 1.0 - abs(float(one_nn_accuracy_macro) - 0.5) / 0.5)


def _final_generation_score(metrics: Mapping[str, Any], config: Experiment6SearchConfig) -> float:
    return (
        config.final_score_weight_g_accuracy * float(metrics["g_accuracy"])
        + config.final_score_weight_target_probability * float(metrics["g_mean_target_probability"])
        + config.final_score_weight_cas * float(metrics["cas_accuracy"])
        + config.final_score_weight_coverage * float(metrics["coverage_macro"])
        + config.final_score_weight_two_sample * _two_sample_target_score(float(metrics["one_nn_accuracy_macro"]))
    )


def _realize_terminal_params(
    raw: Mapping[str, Any],
    *,
    config: Experiment6SearchConfig,
) -> dict[str, Any]:
    params = dict(raw)
    params["max_tau"] = float(params.pop("max_tau_factor")) * float(config.terminal_noise_horizon_reference)
    return params


def _realize_generation_params(
    raw: Mapping[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    params = dict(raw)
    horizon = float(params["horizon"])
    num_steps = int(params.pop("num_steps"))
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    params["step_size"] = float(horizon / num_steps)
    params["poisson_dirichlet_beta"] = float(params.pop("poisson_dirichlet_beta_factor")) * float(top_k)
    params["poisson_dirichlet_max_terms"] = int(params.pop("poisson_dirichlet_max_terms_factor")) * int(top_k)
    params["drift_clip_total_displacement"] = float(params.pop("drift_clip_pixels")) / 28.0
    params["drift_clip_norm"] = None
    params["num_points"] = int(params.get("num_points", top_k))
    return params


def _train_terminal_trial(
    *,
    prepared_data: PreparedMnistExperiment6Data,
    trial_id: int,
    trial_seed: int,
    params: Mapping[str, Any],
    config: Experiment6SearchConfig,
) -> _TerminalCandidate:
    _set_global_seed(trial_seed)
    device = _resolve_device(config.device)

    model = TerminalSetClassifier(
        point_feature_dim=int(params["point_feature_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        num_classes=prepared_data.num_classes,
        dropout=float(params["dropout"]),
    )
    history = train_terminal_set_classifier_noisy(
        model,
        prepared_data.train_pc.masses,
        prepared_data.train_pc.positions,
        np.asarray(prepared_data.train_pc.labels, dtype=np.int64),
        val_masses=prepared_data.val_pc.masses,
        val_positions=prepared_data.val_pc.positions,
        val_labels=np.asarray(prepared_data.val_pc.labels, dtype=np.int64),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        position_jitter_std=float(params["position_jitter_std"]),
        max_tau=float(params["max_tau"]),
        projection=config.terminal_noise_eval_projection,
        tau_sampling=str(params["tau_sampling"]),
        device=device,
        verbose=config.verbose,
    )

    clean_val = evaluate_terminal_set_classifier(
        model,
        prepared_data.val_pc.masses,
        prepared_data.val_pc.positions,
        np.asarray(prepared_data.val_pc.labels, dtype=np.int64),
        batch_size=int(params["batch_size"]),
        device=device,
    )
    noisy_val = _evaluate_terminal_with_noise(
        model,
        prepared_data.val_pc.masses,
        prepared_data.val_pc.positions,
        np.asarray(prepared_data.val_pc.labels, dtype=np.int64),
        tau=float(params["max_tau"]),
        repeats=int(config.terminal_noise_eval_repeats),
        projection=config.terminal_noise_eval_projection,
        device=device,
        seed=trial_seed + 17,
    )

    metrics = {
        "clean_val_loss": float(clean_val["loss"]),
        "clean_val_accuracy": float(clean_val["accuracy"]),
        "noisy_val_loss": float(noisy_val["loss"]),
        "noisy_val_accuracy": float(noisy_val["accuracy"]),
        "best_train_accuracy": float(np.max(history["train_accuracy"])),
        "best_val_accuracy": float(np.nanmax(history["val_accuracy"])),
    }
    score = _terminal_score(metrics["clean_val_accuracy"], metrics["noisy_val_accuracy"])
    return _TerminalCandidate(
        trial_id=trial_id,
        seed=trial_seed,
        params=dict(params),
        metrics=metrics,
        score=float(score),
        model=copy.deepcopy(model).cpu(),
    )


def _run_generation_proxy_trial(
    *,
    prepared_data: PreparedMnistExperiment6Data,
    terminal_candidate: _TerminalCandidate,
    proxy_trial_id: int,
    trial_seed: int,
    params: Mapping[str, Any],
    config: Experiment6SearchConfig,
) -> _GenerationCandidate:
    _set_global_seed(trial_seed)
    device = _resolve_device(config.device)
    model = copy.deepcopy(terminal_candidate.model).to(device)
    model.eval()

    synthetic = generate_balanced_synthetic_dataset_reparam(
        model,
        mass_bank=None,
        num_points=int(params["num_points"]),
        num_per_class=int(config.synthetic_per_class_proxy),
        mass_sampling_mode=str(params["mass_sampling_mode"]),
        class_conditional_mass_sampling=bool(params["class_conditional_mass_sampling"]),
        poisson_dirichlet_beta=float(params["poisson_dirichlet_beta"]),
        poisson_dirichlet_max_terms=int(params["poisson_dirichlet_max_terms"]),
        horizon=float(params["horizon"]),
        step_size=float(params["step_size"]),
        terminal_mc_samples=int(params["terminal_mc_samples"]),
        guidance_scale=float(params["guidance_scale"]),
        initial_position_mode=str(params["initial_position_mode"]),
        initial_position_scale=float(params["initial_position_scale"]),
        initial_position_bank=None,
        initial_position_bank_labels=None,
        class_conditional_initial_positions=False,
        joint_bank_sampling=bool(params["joint_bank_sampling"]),
        initial_position_jitter=float(params["initial_position_jitter"]),
        state_projection=str(params["state_projection"]),
        terminal_projection=str(params["terminal_projection"]),
        diffusion_temperature=float(params["diffusion_temperature"]),
        drift_clip_norm=params["drift_clip_norm"],
        drift_clip_total_displacement=float(params["drift_clip_total_displacement"]),
        batch_size=int(config.generation_batch_size),
        rasterize=True,
        image_size=28,
        device=device,
        rng=np.random.default_rng(trial_seed),
    )

    g_metrics = terminal_g_accuracy(
        model,
        synthetic.masses,
        synthetic.positions,
        synthetic.labels,
        batch_size=512,
        device=device,
    )
    proxy_metrics = {
        "g_accuracy": float(g_metrics["accuracy"]),
        "g_mean_target_probability": float(g_metrics["mean_target_probability"]),
    }
    proxy_score = _proxy_generation_score(
        g_accuracy=proxy_metrics["g_accuracy"],
        g_mean_target_probability=proxy_metrics["g_mean_target_probability"],
        config=config,
    )
    return _GenerationCandidate(
        proxy_trial_id=proxy_trial_id,
        terminal_trial_id=terminal_candidate.trial_id,
        seed=trial_seed,
        params=dict(params),
        proxy_metrics=proxy_metrics,
        proxy_score=float(proxy_score),
    )


def _run_generation_final_eval(
    *,
    prepared_data: PreparedMnistExperiment6Data,
    terminal_candidate: _TerminalCandidate,
    generation_candidate: _GenerationCandidate,
    config: Experiment6SearchConfig,
) -> dict[str, Any]:
    _set_global_seed(generation_candidate.seed)
    device = _resolve_device(config.device)
    model = copy.deepcopy(terminal_candidate.model).to(device)
    model.eval()

    synthetic = generate_balanced_synthetic_dataset_reparam(
        model,
        mass_bank=None,
        num_points=int(generation_candidate.params["num_points"]),
        num_per_class=int(config.synthetic_per_class_final),
        mass_sampling_mode=str(generation_candidate.params["mass_sampling_mode"]),
        class_conditional_mass_sampling=bool(generation_candidate.params["class_conditional_mass_sampling"]),
        poisson_dirichlet_beta=float(generation_candidate.params["poisson_dirichlet_beta"]),
        poisson_dirichlet_max_terms=int(generation_candidate.params["poisson_dirichlet_max_terms"]),
        horizon=float(generation_candidate.params["horizon"]),
        step_size=float(generation_candidate.params["step_size"]),
        terminal_mc_samples=int(generation_candidate.params["terminal_mc_samples"]),
        guidance_scale=float(generation_candidate.params["guidance_scale"]),
        initial_position_mode=str(generation_candidate.params["initial_position_mode"]),
        initial_position_scale=float(generation_candidate.params["initial_position_scale"]),
        initial_position_bank=None,
        initial_position_bank_labels=None,
        class_conditional_initial_positions=False,
        joint_bank_sampling=bool(generation_candidate.params["joint_bank_sampling"]),
        initial_position_jitter=float(generation_candidate.params["initial_position_jitter"]),
        state_projection=str(generation_candidate.params["state_projection"]),
        terminal_projection=str(generation_candidate.params["terminal_projection"]),
        diffusion_temperature=float(generation_candidate.params["diffusion_temperature"]),
        drift_clip_norm=generation_candidate.params["drift_clip_norm"],
        drift_clip_total_displacement=float(generation_candidate.params["drift_clip_total_displacement"]),
        batch_size=int(config.generation_batch_size),
        rasterize=True,
        image_size=28,
        device=device,
        rng=np.random.default_rng(generation_candidate.seed),
    )

    metrics = evaluate_generation_metrics(
        model,
        synthetic,
        prepared_data.test_pc,
        prepared_data.real_test_images,
        prepared_data.real_test_labels,
        cas_epochs=int(config.cas_epochs_final),
        cas_batch_size=int(config.cas_batch_size),
        cas_lr=float(config.cas_lr),
        sinkhorn_epsilon=float(config.sinkhorn_epsilon),
        sinkhorn_iterations=int(config.sinkhorn_iterations_final),
        sinkhorn_subsample_per_class=int(config.sinkhorn_subsample_per_class_final),
        device=device,
        rng=np.random.default_rng(generation_candidate.seed + 1009),
        verbose=config.verbose,
    )
    final_score = _final_generation_score(metrics, config)
    return {
        "metrics": metrics,
        "final_score": float(final_score),
    }


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------


def run_experiment6_random_search(
    *,
    prepared_data: Optional[PreparedMnistExperiment6Data] = None,
    data_root: str | Path = "mnist_data",
    config: Experiment6SearchConfig = Experiment6SearchConfig(),
    terminal_search_space: Optional[SearchSpace] = None,
    generation_search_space: Optional[SearchSpace] = None,
    output_dir: Optional[str | Path] = None,
) -> Experiment6SearchResult:
    """Run the two-stage random search used to tune Example 6.

    Parameters
    ----------
    prepared_data:
        Optional precomputed weighted point-cloud splits. When omitted, the
        function loads MNIST from ``data_root`` and constructs balanced splits.
    data_root:
        Directory containing the raw MNIST IDX files.
    config:
        Global search settings (number of trials, split sizes, evaluation sizes,
        scoring weights, ...).
    terminal_search_space, generation_search_space:
        Random-search spaces. Omit either to use the defaults.
    output_dir:
        Optional directory where CSV/JSON summaries and the best terminal-model
        weights are written.
    """
    if config.terminal_trials <= 0:
        raise ValueError("terminal_trials must be positive")
    if config.keep_top_terminal <= 0:
        raise ValueError("keep_top_terminal must be positive")
    if config.generation_trials_per_terminal <= 0:
        raise ValueError("generation_trials_per_terminal must be positive")
    if config.final_eval_top_k < 0:
        raise ValueError("final_eval_top_k must be non-negative")

    if prepared_data is None:
        prepared_data = prepare_mnist_experiment6_data(
            data_root=data_root,
            train_per_class=config.train_per_class,
            val_per_class=config.val_per_class,
            test_per_class=config.test_per_class,
            top_k=config.top_k,
            mass_floor=config.mass_floor,
            seed=config.seed,
            download=True,
        )

    rng = np.random.default_rng(config.seed)
    terminal_space = default_terminal_search_space() if terminal_search_space is None else dict(terminal_search_space)
    generation_space = (
        default_generation_search_space(top_k=prepared_data.top_k)
        if generation_search_space is None
        else dict(generation_search_space)
    )

    terminal_records: list[dict[str, Any]] = []
    terminal_candidates: list[_TerminalCandidate] = []
    for trial_id in range(config.terminal_trials):
        raw_params = _sample_params(terminal_space, rng)
        params = _realize_terminal_params(raw_params, config=config)
        trial_seed = int(rng.integers(0, 2**31 - 1))
        candidate = _train_terminal_trial(
            prepared_data=prepared_data,
            trial_id=trial_id,
            trial_seed=trial_seed,
            params=params,
            config=config,
        )
        terminal_candidates.append(candidate)
        terminal_records.append(
            {
                "trial_id": trial_id,
                "seed": trial_seed,
                "score": candidate.score,
                **_flatten_trial_record("param.", candidate.params),
                **_flatten_trial_record("metric.", candidate.metrics),
            }
        )
        if config.verbose:
            print(
                f"[search][terminal] trial {trial_id + 1}/{config.terminal_trials}: "
                f"score={candidate.score:.4f}, clean_val={candidate.metrics['clean_val_accuracy']:.4f}, "
                f"noisy_val={candidate.metrics['noisy_val_accuracy']:.4f}"
            )

    terminal_df = pd.DataFrame(terminal_records).sort_values("score", ascending=False).reset_index(drop=True)
    top_terminal_candidates = sorted(terminal_candidates, key=lambda item: item.score, reverse=True)[
        : min(config.keep_top_terminal, len(terminal_candidates))
    ]

    generation_proxy_records: list[dict[str, Any]] = []
    generation_candidates: list[_GenerationCandidate] = []
    proxy_trial_id = 0
    for terminal_candidate in top_terminal_candidates:
        for _ in range(config.generation_trials_per_terminal):
            raw_params = _sample_params(generation_space, rng)
            params = _realize_generation_params(raw_params, top_k=prepared_data.top_k)
            trial_seed = int(rng.integers(0, 2**31 - 1))
            candidate = _run_generation_proxy_trial(
                prepared_data=prepared_data,
                terminal_candidate=terminal_candidate,
                proxy_trial_id=proxy_trial_id,
                trial_seed=trial_seed,
                params=params,
                config=config,
            )
            generation_candidates.append(candidate)
            generation_proxy_records.append(
                {
                    "proxy_trial_id": candidate.proxy_trial_id,
                    "terminal_trial_id": candidate.terminal_trial_id,
                    "seed": candidate.seed,
                    "proxy_score": candidate.proxy_score,
                    **_flatten_trial_record("param.", candidate.params),
                    **_flatten_trial_record("metric.", candidate.proxy_metrics),
                }
            )
            if config.verbose:
                print(
                    f"[search][generation-proxy] terminal={candidate.terminal_trial_id} "
                    f"trial={candidate.proxy_trial_id}: proxy={candidate.proxy_score:.4f}, "
                    f"g_acc={candidate.proxy_metrics['g_accuracy']:.4f}, "
                    f"g_p={candidate.proxy_metrics['g_mean_target_probability']:.4f}"
                )
            proxy_trial_id += 1

    generation_proxy_df = (
        pd.DataFrame(generation_proxy_records).sort_values("proxy_score", ascending=False).reset_index(drop=True)
        if generation_proxy_records
        else pd.DataFrame()
    )

    top_generation_candidates = sorted(
        generation_candidates,
        key=lambda item: item.proxy_score,
        reverse=True,
    )[: min(config.final_eval_top_k, len(generation_candidates))]

    final_records: list[dict[str, Any]] = []
    best_generation_candidate: Optional[_GenerationCandidate] = None
    best_generation_metrics: Optional[dict[str, Any]] = None
    best_generation_score = -float("inf")

    for generation_candidate in top_generation_candidates:
        terminal_candidate = next(
            candidate
            for candidate in top_terminal_candidates
            if candidate.trial_id == generation_candidate.terminal_trial_id
        )
        final_eval = _run_generation_final_eval(
            prepared_data=prepared_data,
            terminal_candidate=terminal_candidate,
            generation_candidate=generation_candidate,
            config=config,
        )
        metrics = dict(final_eval["metrics"])
        final_score = float(final_eval["final_score"])
        final_records.append(
            {
                "proxy_trial_id": generation_candidate.proxy_trial_id,
                "terminal_trial_id": generation_candidate.terminal_trial_id,
                "seed": generation_candidate.seed,
                "proxy_score": generation_candidate.proxy_score,
                "final_score": final_score,
                **_flatten_trial_record("param.", generation_candidate.params),
                **_flatten_trial_record(
                    "metric.",
                    {key: value for key, value in metrics.items() if key not in {"per_label", "cas_details"}},
                ),
            }
        )
        if final_score > best_generation_score:
            best_generation_score = final_score
            best_generation_candidate = generation_candidate
            best_generation_metrics = metrics
        if config.verbose:
            print(
                f"[search][generation-final] proxy_trial={generation_candidate.proxy_trial_id}: "
                f"final={final_score:.4f}, g_acc={metrics['g_accuracy']:.4f}, "
                f"cas={metrics['cas_accuracy']:.4f}, one_nn={metrics['one_nn_accuracy_macro']:.4f}, "
                f"coverage={metrics['coverage_macro']:.4f}"
            )

    generation_final_df = (
        pd.DataFrame(final_records).sort_values("final_score", ascending=False).reset_index(drop=True)
        if final_records
        else pd.DataFrame()
    )

    best_terminal_candidate = top_terminal_candidates[0]
    if best_generation_candidate is None:
        # Fall back to the best proxy candidate if final evaluation was disabled.
        best_generation_candidate = sorted(generation_candidates, key=lambda item: item.proxy_score, reverse=True)[0]
        best_generation_metrics = dict(best_generation_candidate.proxy_metrics)
        best_generation_metrics["proxy_only"] = True

    result = Experiment6SearchResult(
        prepared_data=prepared_data,
        terminal_trials=terminal_df,
        generation_proxy_trials=generation_proxy_df,
        generation_final_trials=generation_final_df,
        best_terminal_params=dict(best_terminal_candidate.params),
        best_generation_params=dict(best_generation_candidate.params),
        best_terminal_metrics=dict(best_terminal_candidate.metrics),
        best_generation_metrics=dict(best_generation_metrics),
        best_terminal_model=copy.deepcopy(best_terminal_candidate.model).cpu(),
    )

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result.terminal_trials.to_csv(output_path / "terminal_trials.csv", index=False)
        result.generation_proxy_trials.to_csv(output_path / "generation_proxy_trials.csv", index=False)
        result.generation_final_trials.to_csv(output_path / "generation_final_trials.csv", index=False)
        torch.save(result.best_terminal_model.state_dict(), output_path / "best_terminal_model.pt")
        with (output_path / "best_config.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "search_config": asdict(config),
                    "best_terminal_params": result.best_terminal_params,
                    "best_terminal_metrics": result.best_terminal_metrics,
                    "best_generation_params": result.best_generation_params,
                    "best_generation_metrics": result.best_generation_metrics,
                },
                handle,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="mnist_data")
    parser.add_argument("--output-dir", type=str, default="tuning_runs/example6_random_search")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--train-per-class", type=int, default=1500)
    parser.add_argument("--val-per-class", type=int, default=300)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--terminal-trials", type=int, default=12)
    parser.add_argument("--keep-top-terminal", type=int, default=3)
    parser.add_argument("--generation-trials-per-terminal", type=int, default=8)
    parser.add_argument("--final-eval-top-k", type=int, default=3)
    parser.add_argument("--synthetic-per-class-proxy", type=int, default=64)
    parser.add_argument("--synthetic-per-class-final", type=int, default=256)
    parser.add_argument("--cas-epochs-final", type=int, default=8)
    parser.add_argument("--sinkhorn-subsample-per-class-final", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = Experiment6SearchConfig(
        seed=args.seed,
        device=args.device,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        top_k=args.top_k,
        terminal_trials=args.terminal_trials,
        keep_top_terminal=args.keep_top_terminal,
        generation_trials_per_terminal=args.generation_trials_per_terminal,
        final_eval_top_k=args.final_eval_top_k,
        synthetic_per_class_proxy=args.synthetic_per_class_proxy,
        synthetic_per_class_final=args.synthetic_per_class_final,
        cas_epochs_final=args.cas_epochs_final,
        sinkhorn_subsample_per_class_final=args.sinkhorn_subsample_per_class_final,
        verbose=not args.quiet,
    )
    result = run_experiment6_random_search(
        data_root=args.data_root,
        config=config,
        output_dir=args.output_dir,
    )
    summary = {
        "best_terminal_params": result.best_terminal_params,
        "best_terminal_metrics": result.best_terminal_metrics,
        "best_generation_params": result.best_generation_params,
        "best_generation_metrics": result.best_generation_metrics,
    }
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
