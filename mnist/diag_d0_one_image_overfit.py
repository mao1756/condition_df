from __future__ import annotations

"""Production one-image overfit gate for the strict Experiment 12 D0 model.

This command is deliberately narrower than :mod:`mnist.experiment12_d0`.  It
freezes the manuscript-aligned 28x28 direct-Doob kernel, verifies the passing
zero-residual run, isolates validation by whole forward path, selects an EMA
checkpoint without looking at generated samples, and only then runs a paired
stochastic reconstruction gate.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    LegacyArtifactError,
    OneImageGateThresholds,
    atomic_copy_file,
    atomic_write_csv,
    atomic_write_json,
    configure_exact_torch_backend,
    config_fingerprint,
    deterministic_path_split,
    evaluate_overfit_gates,
    evaluate_raw_and_ema_validation,
    file_fingerprint,
    freeze_training_target_scale,
    infer_training_target_scale,
    load_cache_bundle,
    load_training_checkpoint,
    save_cache_bundle,
    save_training_checkpoint,
    select_best_ema_checkpoint,
    source_fingerprint,
    terminal_target_abs_correlation,
    restore_training_checkpoint,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    DirectFluxUNet,
    _cuda_autocast,
    _disable_mkldnn_for_cpu_if_needed,
    _make_cuda_grad_scaler,
    edge_alpha_value,
    load_mnist_measure_dataset,
    save_flux_samples_grid,
    temporary_ema_weights,
    update_ema_state,
)
from mnist.experiment12_d0 import (
    Experiment12D0Config,
    _validate_direct_doob_config,
    build_d0_training_cache,
    cache_summary,
    d0_direct_doob_oracle_diagnostic,
    d0_unweighted_innovation_loss,
    sample_d0_cache_batch,
    synthetic_digit_measures,
)
from mnist.d0_one_image_sampler import (
    PairedSamplerConfig,
    resolve_or_create_terminal_assignments,
    run_paired_d0_sampling,
)


RUN_SCHEMA = "experiment12-d0-production-one-image"
RUN_SCHEMA_VERSION = 1
LATEST_CHECKPOINT_SCHEMA = "experiment12-d0-one-image-latest-checkpoint"
LATEST_CHECKPOINT_SCHEMA_VERSION = 1
CLAIM_SCOPE = "one-image reproduction for one frozen fixed-grid temporal kernel"
EXPECTED_KERNEL = {
    "grid_size": 28,
    "sample_steps": 512,
    "reference_substeps": 256,
    "tau_eff": 5e-5,
    "edge_alpha_mode": "alpha_eff",
    "alpha_eff": 1.0,
    "mass_floor": 1e-7,
    "limiter_fraction": 1.0,
}

# A required gate is the named production experiment, not a tunable exploratory
# run.  Exploratory/report-only callers may change these options with
# ``--require-gate none``; a positive scientific gate keeps the choices made in
# the patch plan fixed.  Runtime-only batching options remain configurable but
# are fingerprinted below so exact resume still rejects changes.
REQUIRED_GATE_DEFAULTS = {
    "grid_size": 28,
    "sample_steps": 512,
    "reference_substeps": 256,
    "tau_eff": 5e-5,
    "edge_alpha_mode": "alpha_eff",
    "alpha_eff": 1.0,
    "mass_floor": 1e-7,
    "limiter_fraction": 1.0,
    "lambda_mix": 0.35,
    "single_image_label": 3,
    "single_image_index": 0,
    "max_train": None,
    "examples_per_class": 1000,
    "cache_preflight_paths": 8,
    "cache_paths": 64,
    "cache_batch_size": 64,
    "time_slices_per_path": 16,
    "validation_paths": 16,
    "physical_target_scale_floor": 1e-6,
    "seed": 260718,
    "train_steps": 10_000,
    "batch_size": 128,
    "base_channels": 32,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "validation_every": 500,
    "checkpoint_every": 500,
    "validation_batch_size": 128,
    "overfit_eval_seeds": (260719, 260720),
    "samples_per_seed": 8,
    "sample_batch_size": 8,
    "sampling_checkpoint_every_outer_steps": 8,
    "no_amp": False,
}
REQUIRED_GATE_THRESHOLD_DEFAULTS = {
    "max_terminal_target_abs_corr": 0.10,
    "max_simplex_mass_error": 2e-6,
    "max_floor_correction_l1": 1e-8,
    "max_renorm_correction_l1": 1e-6,
    "max_raw_intervention": 0.005,
    "max_weighted_intervention": 0.0005,
    "min_mean_correlation": 0.90,
    "max_mean_l1": 0.20,
    "min_sample_correlation": 0.85,
    "min_good_sample_fraction": 0.80,
    "min_correlation_improvement": 0.20,
    "min_relative_l1_reduction": 0.25,
}


def _parse_csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        result = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    else:
        result = tuple(int(item) for item in value)
    if not result:
        raise ValueError("at least one integer is required")
    return result


def _json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _utcish_now() -> str:
    return datetime.now().astimezone().isoformat()


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
        args._resolved_run_dir = run_dir
        return run_dir, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = str(args.run_name).strip() or "one-image-direct-doob"
    run_dir = Path(args.runs_root) / f"{stamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    args._resolved_run_dir = run_dir
    return run_dir, False


def _runtime_record(
    device: torch.device, exact_backend: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_type": device.type,
        "exact_backend": dict(exact_backend),
    }
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
        result["device_count"] = int(torch.cuda.device_count())
    return result


def _semantic_close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def verify_zero_residual_run(
    path: str | Path,
    *,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
) -> dict[str, Any]:
    """Verify the upstream training-ready decision by semantic kernel fields."""

    run_dir = Path(path)
    aggregate_path = run_dir / "aggregate_summary.json"
    config_path = run_dir / "run_config.json"
    status_path = run_dir / "run_status.json"
    for required in (aggregate_path, config_path, status_path):
        if not required.is_file():
            raise FileNotFoundError(f"upstream zero-residual artifact is missing: {required}")
    aggregate = _json_load(aggregate_path)
    source_config = _json_load(config_path)
    status = _json_load(status_path)
    if int(aggregate.get("training_ready", 0)) != 1:
        raise ValueError("upstream zero-residual run did not pass training_ready")
    if int(status.get("required_gate_pass", 0)) != 1:
        raise ValueError("upstream zero-residual required gate did not pass")
    upstream_dynamics = dict(source_config.get("dynamics_config", {}))
    upstream_diag = dict(source_config.get("diagnostic_config", {}))
    levels = [int(value) for value in upstream_diag.get("substep_levels", [])]
    actual = {
        "grid_size": upstream_dynamics.get("grid_size"),
        "sample_steps": upstream_diag.get("sample_steps"),
        "reference_substeps": max(levels) if levels else None,
        "tau_eff": upstream_diag.get("tau_eff"),
        "edge_alpha_mode": upstream_dynamics.get("edge_alpha_mode"),
        "alpha_eff": upstream_dynamics.get("alpha_eff"),
        "mass_floor": upstream_dynamics.get("mass_floor"),
        "limiter_fraction": upstream_dynamics.get("limiter_fraction"),
    }
    current = {
        "grid_size": int(dynamics_config.grid_size),
        "sample_steps": int(d0_config.sample_steps),
        "reference_substeps": int(d0_config.reference_substeps),
        "tau_eff": float(d0_config.tau_eff),
        "edge_alpha_mode": str(dynamics_config.edge_alpha_mode),
        "alpha_eff": float(dynamics_config.alpha_eff),
        "mass_floor": float(dynamics_config.mass_floor),
        "limiter_fraction": float(dynamics_config.limiter_fraction),
    }
    mismatches: list[str] = []
    for key, frozen in EXPECTED_KERNEL.items():
        if not _semantic_close(actual.get(key), frozen):
            mismatches.append(f"upstream {key}={actual.get(key)!r}, expected={frozen!r}")
        if not _semantic_close(current.get(key), frozen):
            mismatches.append(f"current {key}={current.get(key)!r}, expected={frozen!r}")
    if mismatches:
        raise ValueError("zero-residual kernel mismatch: " + "; ".join(mismatches))
    fingerprint = str(
        aggregate.get("config_fingerprint")
        or status.get("config_fingerprint")
        or source_config.get("config_fingerprint")
        or file_fingerprint(config_path)
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "training_ready": 1,
        "required_gate_pass": 1,
        "config_fingerprint": fingerprint,
        "semantic_kernel": actual,
        "aggregate_sha256": file_fingerprint(aggregate_path),
        "run_config_sha256": file_fingerprint(config_path),
    }


def _load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
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
    return (
        np.asarray(dataset.train_images, dtype=np.float64),
        np.asarray(dataset.train_labels, dtype=np.int64),
    )


def _select_source_image(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    label: int,
    class_index: int,
    grid_size: int,
    lambda_mix: float,
) -> dict[str, Any]:
    candidates = np.flatnonzero(np.asarray(labels, dtype=np.int64) == int(label))
    if candidates.size == 0:
        raise ValueError(f"dataset has no image with label {label}")
    if int(class_index) < 0 or int(class_index) >= candidates.size:
        raise ValueError(
            f"single-image-index {class_index} is outside label-{label} choices [0,{candidates.size - 1}]"
        )
    source_index = int(candidates[int(class_index)])
    source_input = np.asarray(images[source_index], dtype=np.float64).reshape(-1)
    if source_input.size != int(grid_size) ** 2:
        raise ValueError("selected source image has the wrong grid size")
    # Match `_lambda_mixed_data_for_paths` exactly: mix the dataset measure in
    # float64, clamp/renormalize, and only then store float32 artifacts.
    mixed = (1.0 - float(lambda_mix)) * source_input + float(lambda_mix) / float(
        source_input.size
    )
    mixed = np.maximum(mixed, 0.0)
    mixed /= max(float(mixed.sum()), 1e-30)
    source = np.maximum(source_input, 0.0)
    source /= max(float(source.sum()), 1e-30)

    def digest(value: np.ndarray) -> str:
        array = np.ascontiguousarray(value.astype(np.float32, copy=False))
        h = hashlib.sha256()
        h.update(str(array.shape).encode("ascii"))
        h.update(array.tobytes())
        return h.hexdigest()

    return {
        "dataset_index": source_index,
        "class_index": int(class_index),
        "label": int(label),
        "image": source.astype(np.float32),
        "mixed_target": mixed.astype(np.float32),
        "image_sha256": digest(source),
        "mixed_target_sha256": digest(mixed),
    }


def _make_configs(args: argparse.Namespace) -> tuple[DirectFluxMNISTConfig, Experiment12D0Config]:
    dynamics = DirectFluxMNISTConfig(
        grid_size=int(args.grid_size),
        alpha=1.0,
        beta=1.0,
        alpha_eff=float(args.alpha_eff),
        edge_alpha_mode=str(args.edge_alpha_mode),
        horizon_scale=1.0,
        num_steps=int(args.sample_steps),
        limiter_fraction=float(args.limiter_fraction),
        mass_floor=float(args.mass_floor),
        source_lowfreq_size=min(7, int(args.grid_size)),
        source_blur_sigma=0.0,
        source_uniform_mix=0.15,
        source_concentration=1.0,
        condition_on_source=False,
        flux_parameterization="edge",
        ot_lowres_size=min(7, int(args.grid_size)),
        ot_blur_sigma=0.0,
    )
    d0 = Experiment12D0Config(
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        base_channels=int(args.base_channels),
        cache_paths=int(args.cache_paths),
        cache_batch_size=int(args.cache_batch_size),
        cache_refresh_every=0,
        cache_build_mode="substep",
        cache_time_sampling="endpoint-mixture",
        time_slices_per_path=int(args.time_slices_per_path),
        teacher_stride_substeps=1,
        eta_l2_weight=0.0,
        d0_target_space="doob-physical-residual",
        physical_target_normalization="global-rms",
        physical_target_scale=0.0,
        physical_target_scale_floor=float(args.physical_target_scale_floor),
        physical_loss_mask="all",
        edge_innovation_loss_weight=1.0,
        state_delta_loss_weight=0.0,
        rollout_loss_weight=0.0,
        trajectory_rollout_loss_weight=0.0,
        invalid_output_l2_weight=0.0,
        curl_loss_weight=0.0,
        edge_laplacian_loss_weight=0.0,
        physical_sampler_noise_mode="reference",
        control_output_clip=0.0,
        lambda_mix=float(args.lambda_mix),
        sample_steps=int(args.sample_steps),
        reference_substeps=int(args.reference_substeps),
        tau_eff=float(args.tau_eff),
        time_change_mode="integral",
        rate_ramp="none",
        rate_ramp_ratio=1.0,
        single_image_overfit=True,
        single_image_index=int(args.single_image_index),
        single_image_label=int(args.single_image_label),
        use_external_prior_bank_for_overfit=False,
        allow_prior_bank_mismatch=False,
        allow_stride_rounding=False,
        seed=int(args.seed),
        use_amp=not bool(args.no_amp),
        ema_decay=float(args.ema_decay),
        num_samples=0,
    )
    _validate_direct_doob_config(d0)
    return dynamics, d0


def _thresholds(args: argparse.Namespace) -> OneImageGateThresholds:
    return OneImageGateThresholds(
        terminal_target_abs_corr_max=float(args.max_terminal_target_abs_corr),
        max_simplex_mass_error=float(args.max_simplex_mass_error),
        floor_correction_l1_per_path_substep=float(args.max_floor_correction_l1),
        renorm_correction_l1_per_path_substep=float(args.max_renorm_correction_l1),
        raw_intervention_fraction=float(args.max_raw_intervention),
        weighted_intervention_fraction=float(args.max_weighted_intervention),
        reconstruction_mean_corr=float(args.min_mean_correlation),
        reconstruction_mean_l1=float(args.max_mean_l1),
        reconstruction_good_corr=float(args.min_sample_correlation),
        reconstruction_good_fraction=float(args.min_good_sample_fraction),
        paired_corr_improvement=float(args.min_correlation_improvement),
        relative_l1_reduction=float(args.min_relative_l1_reduction),
    )


def _validate_required_gate_defaults(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Keep named gate claims on the frozen production experiment.

    The options remain useful for diagnostics, so they are not removed from the
    CLI.  A run that changes them must be explicitly non-gating instead of
    silently minting a weaker or scientifically different production claim.
    """

    if str(args.require_gate) == "none":
        return
    if bool(args.synthetic_data):
        parser.error("a required one-image gate must use MNIST, not --synthetic-data")
    mismatches: list[str] = []
    for name, expected in {
        **REQUIRED_GATE_DEFAULTS,
        **REQUIRED_GATE_THRESHOLD_DEFAULTS,
    }.items():
        actual = getattr(args, name)
        if isinstance(expected, float):
            try:
                matches = math.isfinite(float(actual)) and float(actual) == expected
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            option = "--" + name.replace("_", "-")
            mismatches.append(f"{option}={actual!r} (required {expected!r})")
    if mismatches:
        parser.error(
            "required gates freeze the production defaults; use --require-gate none "
            "for exploratory settings: " + "; ".join(mismatches)
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache-preflight", "train", "evaluate", "all"), default="all")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_d0_one_image"))
    parser.add_argument("--run-name", default="production-one-image-direct-doob")
    parser.add_argument("--zero-residual-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--require-gate", choices=("none", "cache", "optimization", "reconstruction"), default="none")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--examples-per-class", type=int, default=1000)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-examples-per-class", type=int, default=8)

    parser.add_argument("--grid-size", type=int, default=28)
    parser.add_argument("--sample-steps", type=int, default=512)
    parser.add_argument("--reference-substeps", type=int, default=256)
    parser.add_argument("--tau-eff", type=float, default=5e-5)
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff",), default="alpha_eff")
    parser.add_argument("--alpha-eff", type=float, default=1.0)
    parser.add_argument("--mass-floor", type=float, default=1e-7)
    parser.add_argument("--limiter-fraction", type=float, default=1.0)
    parser.add_argument("--lambda-mix", type=float, default=0.35)
    parser.add_argument("--single-image-label", type=int, default=3)
    parser.add_argument("--single-image-index", type=int, default=0)
    parser.add_argument("--cache-preflight-paths", type=int, default=8)
    parser.add_argument("--cache-paths", type=int, default=64)
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--time-slices-per-path", type=int, default=16)
    parser.add_argument("--validation-paths", type=int, default=16)
    parser.add_argument("--physical-target-scale-floor", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=260718)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--validation-batch-size", type=int, default=128)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--overfit-eval-seeds", default="260719,260720")
    parser.add_argument("--samples-per-seed", type=int, default=8)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--sampling-checkpoint-every-outer-steps", type=int, default=8)

    parser.add_argument("--max-terminal-target-abs-corr", type=float, default=0.10)
    parser.add_argument("--max-simplex-mass-error", type=float, default=2e-6)
    parser.add_argument("--max-floor-correction-l1", type=float, default=1e-8)
    parser.add_argument("--max-renorm-correction-l1", type=float, default=1e-6)
    parser.add_argument("--max-raw-intervention", type=float, default=0.005)
    parser.add_argument("--max-weighted-intervention", type=float, default=0.0005)
    parser.add_argument("--min-mean-correlation", type=float, default=0.90)
    parser.add_argument("--max-mean-l1", type=float, default=0.20)
    parser.add_argument("--min-sample-correlation", type=float, default=0.85)
    parser.add_argument("--min-good-sample-fraction", type=float, default=0.80)
    parser.add_argument("--min-correlation-improvement", type=float, default=0.20)
    parser.add_argument("--min-relative-l1-reduction", type=float, default=0.25)
    args = parser.parse_args(argv)
    args.overfit_eval_seeds = _parse_csv_ints(args.overfit_eval_seeds)
    if args.checkpoint_path is not None and args.stage not in {"evaluate", "all"}:
        parser.error("--checkpoint-path is only valid for evaluate/all")
    if args.checkpoint_path is not None and args.require_gate != "none":
        parser.error("report-only --checkpoint-path evaluation cannot satisfy a required gate")
    if len(args.overfit_eval_seeds) != 2:
        parser.error("--overfit-eval-seeds must contain exactly two seeds")
    if int(args.samples_per_seed) <= 0 or int(args.sample_batch_size) <= 0:
        parser.error("sampling counts must be positive")
    if int(args.samples_per_seed) * len(args.overfit_eval_seeds) > int(args.validation_paths):
        parser.error("paired evaluation requires disjoint validation terminals")
    for name in ("validation_every", "checkpoint_every", "sampling_checkpoint_every_outer_steps"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    _validate_required_gate_defaults(args, parser)
    return args


def _source_fingerprint() -> tuple[str, list[str]]:
    here = Path(__file__).resolve()
    paths = [
        here,
        here.with_name("experiment12_d0.py"),
        here.with_name("d0_one_image_gate.py"),
        here.with_name("d0_one_image_sampler.py"),
        here.with_name("eulerian_flux_mnist.py"),
        here.with_name("weighted_point_cloud.py"),
    ]
    existing = [path for path in paths if path.is_file()]
    return source_fingerprint(existing), [str(path) for path in existing]


def _scientific_payload(
    args: argparse.Namespace,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    thresholds: OneImageGateThresholds,
    source_image: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "kernel": {
            "grid_size": int(dynamics_config.grid_size),
            "sample_steps": int(d0_config.sample_steps),
            "reference_substeps": int(d0_config.reference_substeps),
            "teacher_stride_substeps": int(d0_config.teacher_stride_substeps),
            "tau_eff": float(d0_config.tau_eff),
            "edge_alpha_mode": str(dynamics_config.edge_alpha_mode),
            "edge_alpha_value": float(edge_alpha_value(dynamics_config)),
            "alpha_eff": float(dynamics_config.alpha_eff),
            "mass_floor": float(dynamics_config.mass_floor),
            "limiter_fraction": float(dynamics_config.limiter_fraction),
            "lambda_mix": float(d0_config.lambda_mix),
            "integrator": "masked_reference_free_step_torch/direct_doob_reverse_substep",
        },
        "objective": {
            "target": str(d0_config.d0_target_space),
            "normalization": str(d0_config.physical_target_normalization),
            "loss_mask": str(d0_config.physical_loss_mask),
            "sampler_noise": str(d0_config.physical_sampler_noise_mode),
            "auxiliary_weights": {
                "eta_l2": float(d0_config.eta_l2_weight),
                "state_delta": float(d0_config.state_delta_loss_weight),
                "rollout": float(d0_config.rollout_loss_weight),
                "trajectory_rollout": float(d0_config.trajectory_rollout_loss_weight),
                "invalid_output": float(d0_config.invalid_output_l2_weight),
                "curl": float(d0_config.curl_loss_weight),
                "edge_laplacian": float(d0_config.edge_laplacian_loss_weight),
            },
            "control_output_clip": float(d0_config.control_output_clip),
            "physical_target_scale_floor": float(
                d0_config.physical_target_scale_floor
            ),
        },
        "cache": {
            "paths": int(args.cache_paths),
            "preflight_paths": int(args.cache_preflight_paths),
            "time_slices_per_path": int(args.time_slices_per_path),
            "validation_paths": int(args.validation_paths),
            "cache_batch_size": int(args.cache_batch_size),
            "split_seed": int(args.seed),
            "preflight_seed": int(args.seed) + 11,
            "full_cache_seed": int(args.seed) + 23,
        },
        "training": {
            "seed": int(args.seed),
            "batch_rng_seed": int(args.seed) + 101,
            "steps": int(args.train_steps),
            "batch_size": int(args.batch_size),
            "base_channels": int(args.base_channels),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "ema_decay": float(args.ema_decay),
            "validation_every": int(args.validation_every),
            "checkpoint_every": int(args.checkpoint_every),
            "validation_batch_size": int(args.validation_batch_size),
            "use_amp": bool(d0_config.use_amp),
        },
        "evaluation": {
            "seeds": list(args.overfit_eval_seeds),
            "samples_per_seed": int(args.samples_per_seed),
            "sample_batch_size": int(args.sample_batch_size),
            "checkpoint_every_outer_steps": int(
                args.sampling_checkpoint_every_outer_steps
            ),
        },
        "source_image": {
            "label": int(source_image["label"]),
            "class_index": int(source_image["class_index"]),
            "dataset_index": int(source_image["dataset_index"]),
            "image_sha256": str(source_image["image_sha256"]),
            "mixed_target_sha256": str(source_image["mixed_target_sha256"]),
        },
        "upstream_zero_residual_fingerprint": str(upstream["config_fingerprint"]),
        "thresholds": asdict(thresholds),
        "claim_scope": CLAIM_SCOPE,
    }


def _initial_manifest(
    *,
    scientific_payload: Mapping[str, Any],
    scientific_fingerprint: str,
    source_hash: str,
    source_paths: Sequence[str],
    runtime: Mapping[str, Any],
    upstream: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": _utcish_now(),
        "run_dir": str(run_dir.resolve()),
        "scientific_config": dict(scientific_payload),
        "scientific_fingerprint": scientific_fingerprint,
        "source_fingerprint": source_hash,
        "source_paths": list(source_paths),
        "runtime": dict(runtime),
        "runtime_fingerprint": config_fingerprint(runtime),
        "upstream_zero_residual": dict(upstream),
        "artifacts": {},
    }


def _load_or_create_manifest(
    run_dir: Path,
    *,
    resumed: bool,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not resumed:
        manifest = dict(candidate)
        atomic_write_json(path, manifest)
        return manifest
    if not path.is_file():
        raise ArtifactCompatibilityError("resume run has no run_manifest.json")
    manifest = _json_load(path)
    if manifest.get("schema") != RUN_SCHEMA or int(manifest.get("schema_version", -1)) != RUN_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("resume run manifest schema is incompatible")
    comparisons = {
        "scientific_fingerprint": candidate["scientific_fingerprint"],
        "source_fingerprint": candidate["source_fingerprint"],
        "runtime_fingerprint": candidate["runtime_fingerprint"],
    }
    changed = [key for key, value in comparisons.items() if manifest.get(key) != value]
    if changed:
        raise ArtifactCompatibilityError(
            "resume run fingerprint mismatch: " + ", ".join(changed)
        )
    return manifest


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    status = _json_load(path) if path.is_file() else {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "started_at": _utcish_now(),
    }
    status.update(updates)
    status["updated_at"] = _utcish_now()
    atomic_write_json(path, status)
    return status


def _artifact_fingerprints(manifest: Mapping[str, Any]) -> dict[str, str]:
    upstream = dict(manifest.get("upstream_zero_residual", {}))
    return {
        "scientific": str(manifest["scientific_fingerprint"]),
        "source": str(manifest["source_fingerprint"]),
        "runtime": str(manifest["runtime_fingerprint"]),
        "image": str(
            dict(manifest["scientific_config"])["source_image"]["image_sha256"]
        ),
        "upstream": str(upstream.get("config_fingerprint", "")),
    }


def _cache_numerical_metrics(cache: Any) -> dict[str, Any]:
    state_arrays = [
        cache.states.detach().cpu().numpy(),
        cache.earlier_states.detach().cpu().numpy(),
        cache.start_images.detach().cpu().numpy(),
        np.asarray(cache.terminal_states),
    ]
    edge_arrays = [
        cache.innovations.detach().cpu().numpy(),
        cache.physical_transfers.detach().cpu().numpy(),
    ]
    nonfinite = int(sum(np.size(value) - np.isfinite(value).sum() for value in state_arrays + edge_arrays))
    simplex_errors: list[float] = []
    for values in state_arrays:
        flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
        simplex_errors.append(float(np.max(np.abs(flat.sum(axis=1) - 1.0))))
    denominator = float(
        max(1, int(np.asarray(cache.terminal_states).shape[0]))
        * max(1, int(cache.sample_steps))
        * max(1, int(cache.reference_substeps))
    )
    return {
        "nonfinite_edges": nonfinite,
        "max_simplex_mass_error": max(simplex_errors, default=float("nan")),
        "floor_correction_l1_per_path_substep": float(cache.floor_correction_l1) / denominator,
        "renorm_correction_l1_per_path_substep": float(cache.renorm_correction_l1) / denominator,
    }


def _validate_cache_source(cache: Any, source_image: Mapping[str, Any]) -> None:
    expected_index = int(source_image["dataset_index"])
    expected_label = int(source_image["label"])
    if not np.all(np.asarray(cache.source_indices, dtype=np.int64) == expected_index):
        raise ArtifactCompatibilityError("cache source image index does not match the manifest")
    if not np.all(np.asarray(cache.requested_labels, dtype=np.int64) == expected_label):
        raise ArtifactCompatibilityError("cache label does not match the manifest")
    mixed = np.asarray(source_image["mixed_target"], dtype=np.float32).reshape(1, -1)
    starts = cache.start_images.detach().cpu().numpy().astype(np.float32, copy=False)
    if starts.shape[1] != mixed.shape[1] or not np.array_equal(
        starts, np.broadcast_to(mixed, starts.shape)
    ):
        raise ArtifactCompatibilityError(
            "cache start images do not exactly match the lambda-mixed manifest target"
        )


def _cache_gate_metrics(
    cache: Any,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    *,
    device: torch.device,
) -> dict[str, Any]:
    summary = cache_summary(cache, dynamics_config, d0_config)
    oracle = d0_direct_doob_oracle_diagnostic(
        cache,
        dynamics_config,
        max_slices=int(cache.size),
        device=device,
    )
    correlations = terminal_target_abs_correlation(cache)
    return {
        **summary,
        **oracle,
        **correlations,
        **_cache_numerical_metrics(cache),
        "cache_build_mode": str(cache.cache_build_mode),
        "cache_stride_substeps": int(cache.stride_substeps),
        "physical_target_scale": float(cache.physical_target_scale),
        "target_finite_fraction": float(summary["cache_target_finite_fraction"]),
        "raw_limited_fraction": float(cache.raw_limited_fraction),
        "mobility_weighted_limited_fraction": float(
            cache.mobility_weighted_limited_fraction
        ),
        "noise_energy_weighted_limited_fraction": float(
            cache.noise_energy_weighted_limited_fraction
        ),
        "floor_touched_pixels": int(cache.floor_touched_pixels),
    }


def _validate_preflight_binding(
    record: Mapping[str, Any],
    artifact: Any,
    *,
    recomputed_metrics: Mapping[str, Any],
    recomputed_gate: Mapping[str, Any],
) -> None:
    """Bind the preflight decision to the exact verified cache evidence."""

    cache_fingerprint = str(record.get("cache_fingerprint", ""))
    if not cache_fingerprint or cache_fingerprint != str(artifact.cache_fingerprint):
        raise ArtifactCompatibilityError(
            "cache preflight JSON is not bound to the verified cache content"
        )
    metrics_fingerprint = str(record.get("metrics_fingerprint", ""))
    gate_fingerprint = str(record.get("gate_fingerprint", ""))
    if not metrics_fingerprint or not gate_fingerprint:
        raise ArtifactCompatibilityError(
            "cache preflight record has no metrics/gate evidence fingerprints"
        )
    metadata = dict(getattr(artifact, "metadata", {}))
    if str(metadata.get("metrics_fingerprint", "")) != metrics_fingerprint:
        raise ArtifactCompatibilityError(
            "cache preflight metrics fingerprint is not bound into the cache artifact"
        )
    if str(metadata.get("gate_fingerprint", "")) != gate_fingerprint:
        raise ArtifactCompatibilityError(
            "cache preflight gate fingerprint is not bound into the cache artifact"
        )
    comparisons = {
        "saved metrics": config_fingerprint(dict(record.get("metrics", {}))),
        "recomputed metrics": config_fingerprint(dict(recomputed_metrics)),
    }
    changed_metrics = [
        name for name, actual in comparisons.items() if actual != metrics_fingerprint
    ]
    if changed_metrics:
        raise ArtifactCompatibilityError(
            "cache preflight metrics do not reproduce from the verified cache: "
            + ", ".join(changed_metrics)
        )
    gate_comparisons = {
        "saved gate": config_fingerprint(dict(record.get("gate", {}))),
        "recomputed gate": config_fingerprint(dict(recomputed_gate)),
    }
    changed_gates = [
        name for name, actual in gate_comparisons.items() if actual != gate_fingerprint
    ]
    if changed_gates:
        raise ArtifactCompatibilityError(
            "cache preflight gate does not reproduce from the verified metrics: "
            + ", ".join(changed_gates)
        )


def _build_cache(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    paths: int,
    device: torch.device,
    seed: int,
    show_progress: bool,
) -> Any:
    config = replace(d0_config, cache_paths=int(paths), physical_target_scale=0.0)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    return build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=dynamics_config,
        d0_config=config,
        device=device,
        rng=np.random.default_rng(int(seed)),
        show_progress=show_progress,
    )


def run_cache_preflight(
    *,
    run_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    source_image: Mapping[str, Any],
    thresholds: OneImageGateThresholds,
    device: torch.device,
    show_progress: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _write_status(run_dir, status="running", phase="cache-preflight")
    cache = _build_cache(
        images=images,
        labels=labels,
        dynamics_config=dynamics_config,
        d0_config=d0_config,
        paths=int(args.cache_preflight_paths),
        device=device,
        seed=int(args.seed) + 11,
        show_progress=show_progress,
    )
    _validate_cache_source(cache, source_image)
    all_rows = np.arange(int(cache.size), dtype=np.int64)
    scale = infer_training_target_scale(
        cache,
        dynamics_config,
        d0_config,
        all_rows,
        device=device,
        batch_size=int(args.validation_batch_size),
    )
    cache.physical_target_scale = float(scale)
    effective = replace(d0_config, physical_target_scale=float(scale))
    metrics = _cache_gate_metrics(
        cache, dynamics_config, effective, device=device
    )
    gate = evaluate_overfit_gates(
        cache_metrics=metrics,
        optimization_metrics=None,
        reconstruction_metrics=None,
        require_gate="cache",
        thresholds=thresholds,
    )["cache"]
    metrics_fingerprint = config_fingerprint(metrics)
    gate_fingerprint = config_fingerprint(gate)
    fingerprints = _artifact_fingerprints(manifest)
    artifact = save_cache_bundle(
        run_dir / "cache_preflight.npz",
        cache,
        metadata={
            "scope": "eight-path cache/kernel preflight",
            "path_count": int(args.cache_preflight_paths),
            "metrics_fingerprint": metrics_fingerprint,
            "gate_fingerprint": gate_fingerprint,
        },
        fingerprints=fingerprints,
    )
    payload = {
        "schema": "experiment12-d0-one-image-cache-preflight",
        "schema_version": 1,
        "scope": "cache/kernel validation only; no learned-model evidence",
        "metrics": metrics,
        "gate": gate,
        "cache_fingerprint": artifact.cache_fingerprint,
        "metrics_fingerprint": metrics_fingerprint,
        "gate_fingerprint": gate_fingerprint,
        "fingerprints": fingerprints,
    }
    atomic_write_json(run_dir / "cache_preflight.json", payload)
    _write_status(
        run_dir,
        status="running",
        phase="cache-preflight-complete",
        cache_preflight_pass=int(gate["passed"]),
    )
    return payload, gate


def _validate_committed_cache_identity(
    manifest: Mapping[str, Any],
    *,
    cache_fingerprint: str,
    split_fingerprint: str,
    physical_target_scale: float,
) -> None:
    """Reject replacement of a cache already committed by the run manifest."""

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ArtifactCompatibilityError("run manifest artifacts must be a mapping")
    recorded = artifacts.get("cache")
    if recorded is None:
        # Safe recovery for a crash after the atomic cache/split writes but
        # before the manifest commit.
        return
    if not isinstance(recorded, Mapping):
        raise ArtifactCompatibilityError("run manifest cache record is malformed")
    comparisons = {
        "cache_fingerprint": (
            str(recorded.get("cache_fingerprint", "")),
            str(cache_fingerprint),
        ),
        "split_fingerprint": (
            str(recorded.get("split_fingerprint", "")),
            str(split_fingerprint),
        ),
        "physical_target_scale": (
            recorded.get("physical_target_scale"),
            float(physical_target_scale),
        ),
    }
    changed: list[str] = []
    for name, (saved, current) in comparisons.items():
        if name == "physical_target_scale":
            try:
                matches = (
                    math.isfinite(float(saved))
                    and float(saved) == float(current)
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = bool(saved) and saved == current
        if not matches:
            changed.append(name)
    if changed:
        raise ArtifactCompatibilityError(
            "training cache differs from the identity committed in run_manifest.json: "
            + ", ".join(changed)
        )


def ensure_full_cache(
    *,
    run_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    source_image: Mapping[str, Any],
    device: torch.device,
    show_progress: bool,
) -> tuple[Any, Experiment12D0Config, Any, dict[str, str]]:
    cache_path = run_dir / "experiment12_d0_one_image_cache.npz"
    base_fingerprints = _artifact_fingerprints(manifest)
    split_path = run_dir / "path_split.json"
    if cache_path.is_file():
        artifact = load_cache_bundle(
            cache_path,
            expected_fingerprints=base_fingerprints,
            verify_content=True,
        )
        cache = artifact.cache
        _validate_cache_source(cache, source_image)
        split = deterministic_path_split(
            cache,
            validation_paths=int(args.validation_paths),
            seed=int(args.seed),
        )
        if str(artifact.metadata.get("split_fingerprint", "")) != split.fingerprint:
            raise ArtifactCompatibilityError(
                "training cache metadata has the wrong whole-path split fingerprint"
            )
        if artifact.metadata.get("target_scale_source") != "training paths only":
            raise ArtifactCompatibilityError(
                "training cache does not attest a training-path-only target scale"
            )
        if split_path.is_file():
            split_json = _json_load(split_path)
            if split_json != split.to_dict():
                raise ArtifactCompatibilityError("saved whole-path split fingerprint mismatch")
        else:
            # The cache is atomically complete and the split is deterministic;
            # recover the tiny sidecar after a crash between the two replaces.
            atomic_write_json(split_path, split.to_dict())
        scale = float(cache.physical_target_scale)
        effective = replace(d0_config, physical_target_scale=scale)
    else:
        _write_status(run_dir, status="running", phase="full-cache")
        cache = _build_cache(
            images=images,
            labels=labels,
            dynamics_config=dynamics_config,
            d0_config=d0_config,
            paths=int(args.cache_paths),
            device=device,
            seed=int(args.seed) + 23,
            show_progress=show_progress,
        )
        _validate_cache_source(cache, source_image)
        split = deterministic_path_split(
            cache,
            validation_paths=int(args.validation_paths),
            seed=int(args.seed),
        )
        effective = freeze_training_target_scale(
            cache,
            dynamics_config,
            d0_config,
            split,
            device=device,
            batch_size=int(args.validation_batch_size),
        )
        artifact = save_cache_bundle(
            cache_path,
            cache,
            metadata={
                "scope": "production one-image training cache",
                "target_scale_source": "training paths only",
                "split_fingerprint": split.fingerprint,
            },
            fingerprints=base_fingerprints,
        )
        atomic_write_json(split_path, split.to_dict())
    _validate_committed_cache_identity(
        manifest,
        cache_fingerprint=artifact.cache_fingerprint,
        split_fingerprint=split.fingerprint,
        physical_target_scale=float(effective.physical_target_scale),
    )
    strict_fingerprints = {
        **base_fingerprints,
        "cache": artifact.cache_fingerprint,
        "split": split.fingerprint,
    }
    manifest.setdefault("artifacts", {})["cache"] = {
        "path": str(cache_path),
        "cache_fingerprint": artifact.cache_fingerprint,
        "split_fingerprint": split.fingerprint,
        "physical_target_scale": float(effective.physical_target_scale),
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    return cache, effective, split, strict_fingerprints


def _flatten_validation_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        step = int(record["step"])
        for weights in ("raw", "ema"):
            metrics = dict(record[weights])
            rows.append({"step": step, "weights": weights, **metrics})
    return rows


def _validation_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    ema = dict(record["ema"])
    bins = [
        dict(row)
        for row in record.get("time_bins", [])
        if str(row.get("weights")) == "ema" and int(row.get("bin_index", -1)) == 4
    ]
    data_end = bins[0] if bins else {}
    return {
        "step": int(record["step"]),
        "primary_mse": float(ema.get("primary_mse", float("nan"))),
        "prediction_gain": float(ema.get("prediction_gain", float("nan"))),
        "zero_baseline_mse": float(ema.get("zero_baseline_mse", float("nan"))),
        "data_end_prediction_gain": float(
            data_end.get("prediction_gain", float("nan"))
        ),
        "data_end_primary_mse": float(data_end.get("primary_mse", float("nan"))),
        "data_end_slice_count": int(data_end.get("slice_count", 0)),
    }


def _save_validation_plot(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for the validation refinement plot") from exc
    steps = [int(record["step"]) for record in records]
    raw = [float(dict(record["raw"])["primary_mse"]) for record in records]
    ema = [float(dict(record["ema"])["primary_mse"]) for record in records]
    zero = [float(dict(record["ema"])["zero_baseline_mse"]) for record in records]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(steps, raw, marker="o", linewidth=1.2, label="raw validation MSE")
    axis.plot(steps, ema, marker="o", linewidth=1.5, label="EMA validation MSE")
    axis.plot(steps, zero, linestyle="--", linewidth=1.1, label="zero baseline MSE")
    axis.set_xlabel("training step")
    axis.set_ylabel("scaled direct-residual MSE")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    temporary = path.with_name(f".{path.name}.tmp.png")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, path)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _load_latest_training_checkpoint(
    latest_path: Path,
    checkpoints_dir: Path,
    *,
    map_location: str | torch.device,
    strict_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve an integrity-bound latest pointer for exact training resume."""

    latest = _json_load(latest_path)
    if (
        latest.get("schema") != LATEST_CHECKPOINT_SCHEMA
        or int(latest.get("schema_version", -1))
        != LATEST_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ArtifactCompatibilityError(
            "latest checkpoint pointer schema is incompatible with exact resume"
        )
    step = int(latest.get("step", -1))
    filename = str(latest.get("filename", ""))
    expected_filename = f"step-{step:08d}.pt" if step >= 0 else ""
    if not filename or Path(filename).name != filename or filename != expected_filename:
        raise ArtifactCompatibilityError(
            "latest checkpoint pointer contains an unsafe or inconsistent filename"
        )
    if dict(latest.get("fingerprints", {})) != dict(strict_fingerprints):
        raise ArtifactCompatibilityError(
            "latest checkpoint pointer fingerprint mapping differs from the run"
        )
    checkpoint_path = checkpoints_dir / filename
    if not checkpoint_path.is_file():
        raise ArtifactCompatibilityError("latest checkpoint file is missing")
    expected_sha256 = str(latest.get("checkpoint_sha256", ""))
    if not expected_sha256 or file_fingerprint(checkpoint_path) != expected_sha256:
        raise ArtifactCompatibilityError(
            "latest checkpoint file hash differs from its committed pointer"
        )
    payload = load_training_checkpoint(
        checkpoint_path,
        map_location=map_location,
        expected_fingerprints=strict_fingerprints,
    )
    if int(payload.get("step", -1)) != step:
        raise ArtifactCompatibilityError(
            "latest checkpoint payload step differs from its committed pointer"
        )
    return payload


def _recover_selected_best_checkpoint(
    checkpoints_dir: Path,
    best_validation: Mapping[str, Any],
    strict_fingerprints: Mapping[str, str],
) -> Path:
    """Rebuild ``best_ema.pt`` from its authoritative step checkpoint."""

    selected_step = int(best_validation["step"])
    step_path = checkpoints_dir / f"step-{selected_step:08d}.pt"
    if not step_path.is_file():
        raise ArtifactCompatibilityError(
            "selected EMA checkpoint is missing and cannot be recovered"
        )
    payload = load_training_checkpoint(
        step_path,
        map_location="cpu",
        expected_fingerprints=strict_fingerprints,
    )
    if int(payload.get("step", -1)) != selected_step:
        raise ArtifactCompatibilityError(
            "selected EMA step checkpoint has an inconsistent training step"
        )
    best_path = checkpoints_dir / "best_ema.pt"
    # Preserve the selected checkpoint's byte identity.  The paired sampler
    # fingerprints this file, so load/re-serialize recovery would make an exact
    # sampling resume spuriously incompatible (and can change CUDA storage tags).
    atomic_copy_file(step_path, best_path)
    return best_path


def train_one_image_model(
    *,
    run_dir: Path,
    cache: Any,
    split: Any,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    strict_fingerprints: Mapping[str, str],
    args: argparse.Namespace,
    device: torch.device,
    show_progress: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Train with whole-path validation and exact atomic resume."""

    _write_status(run_dir, status="running", phase="training")
    _disable_mkldnn_for_cpu_if_needed(device)
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) & 0xFFFFFFFF)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    train_rng = np.random.default_rng(int(args.seed) + 101)
    model = DirectFluxUNet(
        dynamics_config, base_channels=int(d0_config.base_channels)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(d0_config.learning_rate),
        weight_decay=float(d0_config.weight_decay),
    )
    scaler = _make_cuda_grad_scaler(
        enabled=bool(d0_config.use_amp and device.type == "cuda")
    )
    ema_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints_dir / "latest.json"
    best_path = checkpoints_dir / "best_ema.pt"
    history: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    time_bin_rows: list[dict[str, Any]] = []
    best_validation: dict[str, Any] | None = None
    completed_step = 0

    if latest_path.is_file():
        payload = _load_latest_training_checkpoint(
            latest_path,
            checkpoints_dir,
            map_location=device,
            strict_fingerprints=strict_fingerprints,
        )
        restored = restore_training_checkpoint(
            payload,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            numpy_rng=train_rng,
            restore_rng=True,
        )
        _optimizer_to_device(optimizer, device)
        completed_step = int(restored["step"])
        ema_state = dict(restored["ema_state"])
        history = list(restored["history"])
        best_validation = restored["best_validation"]
        extra = dict(restored.get("extra", {}))
        validation_records = list(extra.get("validation_records", []))
        time_bin_rows = list(extra.get("time_bin_rows", []))
        if best_validation is not None:
            # ``latest.json`` and ``best_ema.pt`` are separate atomic commits.
            # Always reconstruct the latter from the selected step checkpoint
            # on resume, covering a crash after latest advanced while an older
            # best file was still present.
            _recover_selected_best_checkpoint(
                checkpoints_dir,
                best_validation,
                strict_fingerprints,
            )

    def validate(step: int) -> dict[str, Any]:
        result = evaluate_raw_and_ema_validation(
            model,
            ema_state,
            cache,
            split.validation_slice_indices,
            dynamics_config,
            d0_config,
            device=device,
            batch_size=int(args.validation_batch_size),
            step=int(step),
        )
        return result

    def save_checkpoint(step: int, *, validation: Mapping[str, Any] | None) -> Path:
        nonlocal best_validation
        if validation is not None:
            candidate = _validation_candidate(validation)
            candidates = [item for item in (best_validation, candidate) if item is not None]
            best_validation = select_best_ema_checkpoint(candidates)
        checkpoint_path = checkpoints_dir / f"step-{int(step):08d}.pt"
        payload = save_training_checkpoint(
            checkpoint_path,
            model=model,
            ema_state=ema_state,
            optimizer=optimizer,
            scaler=scaler,
            step=int(step),
            history=history,
            best_validation=best_validation,
            fingerprints=strict_fingerprints,
            numpy_rng=train_rng,
            dynamics_config=dynamics_config,
            d0_config=d0_config,
            extra={
                "validation_records": validation_records,
                "time_bin_rows": time_bin_rows,
                "split_fingerprint": split.fingerprint,
            },
        )
        atomic_write_json(
            latest_path,
            {
                "schema": LATEST_CHECKPOINT_SCHEMA,
                "schema_version": LATEST_CHECKPOINT_SCHEMA_VERSION,
                "filename": checkpoint_path.name,
                "step": int(step),
                "fingerprints": dict(strict_fingerprints),
                "checkpoint_sha256": file_fingerprint(checkpoint_path),
            },
        )
        if best_validation is not None and int(best_validation["step"]) == int(step):
            atomic_copy_file(checkpoint_path, best_path)
        return checkpoint_path

    if not validation_records:
        initial_validation = validate(0)
        validation_records.append(initial_validation)
        time_bin_rows.extend(initial_validation["time_bins"])
        save_checkpoint(0, validation=initial_validation)

    amp_context = _cuda_autocast if device.type == "cuda" else lambda enabled: nullcontext()
    last_report = time.perf_counter()
    for step in range(int(completed_step) + 1, int(d0_config.train_steps) + 1):
        batch = sample_d0_cache_batch(
            cache,
            int(d0_config.batch_size),
            device=device,
            rng=train_rng,
            allowed_indices=split.train_slice_indices,
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_context(bool(d0_config.use_amp and device.type == "cuda")):
            loss, diagnostics = d0_unweighted_innovation_loss(
                model,
                batch,
                dynamics_config,
                d0_config,
                step=int(step),
            )
        scaler.scale(loss).backward()
        if float(d0_config.grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(d0_config.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        update_ema_state(ema_state, model, float(d0_config.ema_decay))
        history.append({"step": int(step), **diagnostics})

        validation: dict[str, Any] | None = None
        if step % int(args.validation_every) == 0 or step == int(d0_config.train_steps):
            validation = validate(step)
            validation_records.append(validation)
            time_bin_rows.extend(validation["time_bins"])
        if (
            validation is not None
            or step % int(args.checkpoint_every) == 0
            or step == int(d0_config.train_steps)
        ):
            save_checkpoint(step, validation=validation)
            elapsed = time.perf_counter() - last_report
            last_report = time.perf_counter()
            ema_mse = (
                float(validation["ema"]["primary_mse"])
                if validation is not None
                else float("nan")
            )
            _write_status(
                run_dir,
                status="running",
                phase="training",
                training_step=int(step),
                training_steps=int(d0_config.train_steps),
                last_interval_seconds=float(elapsed),
                latest_ema_validation_mse=ema_mse,
            )
            if show_progress:
                print(
                    f"D0 one-image train step {step}/{d0_config.train_steps} "
                    f"loss={float(diagnostics['loss']):.6g} ema_val_mse={ema_mse:.6g}",
                    flush=True,
                )

    atomic_write_csv(run_dir / "checkpoint_metrics.csv", _flatten_validation_rows(validation_records))
    atomic_write_csv(run_dir / "validation_time_bins.csv", time_bin_rows)
    _save_validation_plot(run_dir / "validation_refinement.png", validation_records)
    if best_validation is None or not best_path.is_file():
        raise RuntimeError("training produced no finite EMA validation checkpoint")
    selected_bins = [
        row
        for row in time_bin_rows
        if str(row.get("weights")) == "ema"
        and int(row.get("step", -1)) == int(best_validation["step"])
    ]
    selection = {
        "policy": "minimum finite held-out EMA primary residual MSE; earliest exact tie",
        "selected": best_validation,
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": file_fingerprint(best_path),
        "selected_time_bins": selected_bins,
        "validation_path_ids": split.validation_path_ids.tolist(),
        "validation_slice_count": int(split.validation_slice_indices.size),
    }
    atomic_write_json(run_dir / "checkpoint_selection.json", selection)
    optimization_metrics = {
        "selected_ema_validation_gain": float(best_validation["prediction_gain"]),
        "selected_ema_data_end_gain": float(best_validation["data_end_prediction_gain"]),
        "selected_ema_data_end_count": int(best_validation["data_end_slice_count"]),
        "selected_step": int(best_validation["step"]),
        "selected_primary_mse": float(best_validation["primary_mse"]),
    }
    _write_status(
        run_dir,
        status="running",
        phase="training-complete",
        training_step=int(d0_config.train_steps),
        selected_checkpoint_step=int(best_validation["step"]),
    )
    return best_path, selection, optimization_metrics


def _optimization_metrics_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    best = payload.get("best_validation")
    if not isinstance(best, Mapping):
        return {}
    return {
        "selected_ema_validation_gain": float(best.get("prediction_gain", float("nan"))),
        "selected_ema_data_end_gain": float(
            best.get("data_end_prediction_gain", float("nan"))
        ),
        "selected_ema_data_end_count": int(best.get("data_end_slice_count", 0)),
        "selected_step": int(best.get("step", payload.get("step", -1))),
        "selected_primary_mse": float(best.get("primary_mse", float("nan"))),
    }


def _validate_selected_checkpoint_binding(
    run_dir: Path,
    checkpoint_path: Path,
    payload: Mapping[str, Any],
    *,
    strict_fingerprints: Mapping[str, str],
    expected_training_step: int,
) -> None:
    """Prove the evaluated file is the EMA winner selected on validation."""

    selection_path = run_dir / "checkpoint_selection.json"
    if not selection_path.is_file():
        raise ArtifactCompatibilityError(
            "strict evaluation requires checkpoint_selection.json"
        )
    selection = _json_load(selection_path)
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ArtifactCompatibilityError("checkpoint selection has no selected EMA record")
    expected_sha256 = str(selection.get("checkpoint_sha256", ""))
    if not expected_sha256 or file_fingerprint(checkpoint_path) != expected_sha256:
        raise ArtifactCompatibilityError(
            "evaluated checkpoint differs from checkpoint_selection.json"
        )
    selected_step = int(selected.get("step", -1))
    if int(payload.get("step", -1)) != selected_step:
        raise ArtifactCompatibilityError(
            "evaluated checkpoint step differs from the selected EMA step"
        )
    payload_best = payload.get("best_validation")
    if not isinstance(payload_best, Mapping) or config_fingerprint(
        dict(payload_best)
    ) != config_fingerprint(dict(selected)):
        raise ArtifactCompatibilityError(
            "evaluated checkpoint validation record differs from the selection artifact"
        )
    authoritative_step = (
        run_dir / "checkpoints" / f"step-{selected_step:08d}.pt"
    )
    if (
        not authoritative_step.is_file()
        or file_fingerprint(authoritative_step) != expected_sha256
    ):
        raise ArtifactCompatibilityError(
            "selected EMA file is not the byte-identical authoritative step checkpoint"
        )
    latest_path = run_dir / "checkpoints" / "latest.json"
    if not latest_path.is_file():
        raise ArtifactCompatibilityError(
            "strict evaluation requires an exact latest checkpoint pointer"
        )
    latest_payload = _load_latest_training_checkpoint(
        latest_path,
        run_dir / "checkpoints",
        map_location="cpu",
        strict_fingerprints=strict_fingerprints,
    )
    if int(latest_payload.get("step", -1)) != int(expected_training_step):
        raise ArtifactCompatibilityError(
            "strict evaluation requires the configured training step budget to be complete"
        )
    latest_best = latest_payload.get("best_validation")
    if not isinstance(latest_best, Mapping) or config_fingerprint(
        dict(latest_best)
    ) != config_fingerprint(dict(selected)):
        raise ArtifactCompatibilityError(
            "checkpoint selection differs from the best validation record in latest.json"
        )


def _save_paired_grid(path: Path, result: Any, *, grid_size: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for the paired reconstruction grid") from exc
    count = int(result.terminal_indices.size)
    cols = min(8, max(count, 1))
    groups = int(math.ceil(count / cols))
    rows = 4 * groups
    fig, axes = plt.subplots(rows, cols, figsize=(1.35 * cols, 1.35 * rows), squeeze=False)
    for axis in axes.reshape(-1):
        axis.axis("off")
    target = np.asarray(result.mixed_target).reshape(grid_size, grid_size)
    terminal = np.asarray(result.terminal_states).reshape(count, grid_size, grid_size)
    strength0 = np.asarray(result.samples_strength0).reshape(count, grid_size, grid_size)
    strength1 = np.asarray(result.samples_strength1).reshape(count, grid_size, grid_size)
    labels = ("mixed target", "terminal", "strength 0", "strength 1")
    arrays = (
        np.repeat(target[None, :, :], count, axis=0),
        terminal,
        strength0,
        strength1,
    )
    for sample_index in range(count):
        group = sample_index // cols
        column = sample_index % cols
        for row_offset, (label, array) in enumerate(zip(labels, arrays)):
            axis = axes[4 * group + row_offset, column]
            image = array[sample_index]
            axis.imshow(image / max(float(image.max()), 1e-30), cmap="gray", interpolation="nearest")
            if column == 0:
                axis.set_ylabel(label, fontsize=8)
            if row_offset == 0:
                axis.set_title(f"t{int(result.terminal_indices[sample_index])}", fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout(pad=0.15)
    temporary = path.with_name(f".{path.name}.tmp.png")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, path)


def _save_flux_grid_atomic(
    path: Path,
    samples: np.ndarray,
    labels: np.ndarray,
    *,
    grid_size: int,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.png")
    save_flux_samples_grid(samples, labels, temporary, grid_size=grid_size)
    os.replace(temporary, path)


def _reconstruction_metrics(
    result: Any,
    *,
    thresholds: OneImageGateThresholds,
) -> dict[str, Any]:
    rows = list(result.per_sample_metrics)
    if not result.complete or not rows:
        return {"sample_count": len(rows), "complete": int(bool(result.complete))}
    corr0 = np.asarray([float(row["mixed_corr_strength0"]) for row in rows])
    corr1 = np.asarray([float(row["mixed_corr_strength1"]) for row in rows])
    l10 = np.asarray([float(row["mixed_l1_strength0"]) for row in rows])
    l11 = np.asarray([float(row["mixed_l1_strength1"]) for row in rows])
    metrics: dict[str, Any] = {
        "complete": 1,
        "sample_count": int(len(rows)),
        "strength_0_mean_corr": float(np.mean(corr0)),
        "strength_0_mean_l1": float(np.mean(l10)),
        "strength_1_mean_corr": float(np.mean(corr1)),
        "strength_1_mean_l1": float(np.mean(l11)),
        "strength_1_good_corr_fraction": float(
            np.mean(corr1 >= float(thresholds.reconstruction_good_corr))
        ),
        "paired_mean_corr_improvement": float(np.mean(corr1 - corr0)),
        "relative_l1_reduction": float(
            1.0 - float(np.mean(l11)) / max(float(np.mean(l10)), 1e-30)
        ),
        "strength_0": dict(result.arm_summaries["0"]),
        "strength_1": dict(result.arm_summaries["1"]),
        "gate_target": "lambda-mixed source image",
        "unmixed_digit_metrics_are_advisory": 1,
    }
    if "unmixed_corr_strength0" in rows[0]:
        metrics.update(
            {
                "advisory_unmixed_strength_0_mean_corr": float(
                    np.mean([float(row["unmixed_corr_strength0"]) for row in rows])
                ),
                "advisory_unmixed_strength_1_mean_corr": float(
                    np.mean([float(row["unmixed_corr_strength1"]) for row in rows])
                ),
                "advisory_unmixed_strength_0_mean_l1": float(
                    np.mean([float(row["unmixed_l1_strength0"]) for row in rows])
                ),
                "advisory_unmixed_strength_1_mean_l1": float(
                    np.mean([float(row["unmixed_l1_strength1"]) for row in rows])
                ),
            }
        )
    return metrics


def evaluate_one_image_model(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    report_only: bool,
    cache: Any,
    split: Any,
    source_image: Mapping[str, Any],
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    strict_fingerprints: Mapping[str, str],
    thresholds: OneImageGateThresholds,
    args: argparse.Namespace,
    device: torch.device,
    show_progress: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Load the selected checkpoint and run paired EMA reconstruction."""

    _write_status(run_dir, status="running", phase="paired-evaluation")
    if report_only:
        warnings.warn(
            "--checkpoint-path evaluation is report-only: cache/source compatibility is "
            "not promoted to a scientific gate even when the checkpoint uses the new schema.",
            RuntimeWarning,
            stacklevel=2,
        )
    payload = load_training_checkpoint(
        checkpoint_path,
        map_location=device,
        expected_fingerprints=None if report_only else strict_fingerprints,
        allow_legacy_report_only=bool(report_only),
    )
    legacy = bool(payload.get("_legacy_report_only", False))
    if not report_only:
        _validate_selected_checkpoint_binding(
            run_dir,
            checkpoint_path,
            payload,
            strict_fingerprints=strict_fingerprints,
            expected_training_step=int(d0_config.train_steps),
        )
    if report_only and not legacy:
        checkpoint_fingerprints = dict(payload.get("fingerprints", {}))
        if checkpoint_fingerprints != dict(strict_fingerprints):
            warnings.warn(
                "modern checkpoint fingerprints differ from the current image/cache/kernel; "
                "evaluation remains report-only and cannot satisfy a required gate",
                RuntimeWarning,
                stacklevel=2,
            )
    checkpoint_d0 = payload.get("d0_config")
    base_channels = int(
        checkpoint_d0.get("base_channels", d0_config.base_channels)
        if isinstance(checkpoint_d0, Mapping)
        else d0_config.base_channels
    )
    model = DirectFluxUNet(dynamics_config, base_channels=base_channels).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    ema_state = payload.get("ema_state_dict")
    if not isinstance(ema_state, Mapping):
        warnings.warn(
            "checkpoint has no EMA state; report-only evaluation uses raw weights",
            RuntimeWarning,
            stacklevel=2,
        )
        ema_state = model.state_dict()
    validation_paths = split.validation_path_ids.astype(np.int64)
    assignments = resolve_or_create_terminal_assignments(
        run_dir / "validation_terminal_assignments.json",
        validation_terminal_indices=validation_paths.tolist(),
        eval_seeds=args.overfit_eval_seeds,
        samples_per_seed=int(args.samples_per_seed),
        selection_seed=int(args.seed),
    )
    sampler_config = PairedSamplerConfig(
        sample_batch_size=int(args.sample_batch_size),
        checkpoint_every_outer_steps=int(args.sampling_checkpoint_every_outer_steps),
        deterministic=False,
        show_progress=show_progress,
    )
    with temporary_ema_weights(model, dict(ema_state)):
        result = run_paired_d0_sampling(
            model,
            terminal_states=cache.terminal_states,
            terminal_labels=cache.requested_labels,
            terminal_assignments=assignments,
            mixed_target=np.asarray(source_image["mixed_target"], dtype=np.float32),
            unmixed_target=np.asarray(source_image["image"], dtype=np.float32),
            dynamics_config=dynamics_config,
            d0_config=d0_config,
            rate_schedule=cache.rate_schedule,
            horizon=float(cache.horizon),
            physical_target_scale=float(d0_config.physical_target_scale),
            device=device,
            output_dir=run_dir,
            fingerprints={
                **dict(strict_fingerprints),
                "checkpoint": file_fingerprint(checkpoint_path),
            },
            sampler_config=sampler_config,
            resume=True,
        )
    if not result.complete:
        raise RuntimeError("paired sampler stopped before completing every requested seed")
    _save_paired_grid(
        run_dir / "paired_sample_grid.png",
        result,
        grid_size=int(dynamics_config.grid_size),
    )
    # Separate grids make downstream visual diffing convenient while the paired
    # four-row panel remains the primary acceptance artifact.
    _save_flux_grid_atomic(
        run_dir / "strength_0_grid.png",
        result.samples_strength0,
        result.labels,
        grid_size=int(dynamics_config.grid_size),
    )
    _save_flux_grid_atomic(
        run_dir / "strength_1_grid.png",
        result.samples_strength1,
        result.labels,
        grid_size=int(dynamics_config.grid_size),
    )
    reconstruction = _reconstruction_metrics(result, thresholds=thresholds)
    atomic_write_json(run_dir / "reconstruction_metrics.json", reconstruction)
    optimization = _optimization_metrics_from_payload(payload)
    _write_status(
        run_dir,
        status="running",
        phase="paired-evaluation-complete",
        paired_samples=int(result.terminal_indices.size),
        report_only=int(report_only),
        legacy_checkpoint=int(legacy),
    )
    return reconstruction, optimization, bool(not legacy and not report_only)


def _selection_optimization_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "checkpoint_selection.json"
    if not path.is_file():
        return {}
    selected = _json_load(path).get("selected")
    if not isinstance(selected, Mapping):
        return {}
    return {
        "selected_ema_validation_gain": float(
            selected.get("prediction_gain", float("nan"))
        ),
        "selected_ema_data_end_gain": float(
            selected.get("data_end_prediction_gain", float("nan"))
        ),
        "selected_ema_data_end_count": int(selected.get("data_end_slice_count", 0)),
        "selected_step": int(selected.get("step", -1)),
        "selected_primary_mse": float(selected.get("primary_mse", float("nan"))),
    }


def _finalize_gate(
    *,
    run_dir: Path,
    cache_metrics: Mapping[str, Any] | None,
    optimization_metrics: Mapping[str, Any] | None,
    reconstruction_metrics: Mapping[str, Any] | None,
    require_gate: str,
    thresholds: OneImageGateThresholds,
    strict_checkpoint_eligible: bool,
    skips: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = evaluate_overfit_gates(
        cache_metrics=cache_metrics,
        optimization_metrics=optimization_metrics,
        reconstruction_metrics=reconstruction_metrics,
        require_gate=require_gate,
        thresholds=thresholds,
    )
    report["strict_checkpoint_eligible"] = int(strict_checkpoint_eligible)
    report["stage_skips"] = [dict(item) for item in skips]
    report["artifacts_complete_before_exit"] = 1
    if require_gate != "none" and not strict_checkpoint_eligible:
        report["required_gate_pass"] = 0
        report["strict_eligibility_failure"] = (
            "legacy or report-only checkpoints cannot satisfy a required gate"
        )
    atomic_write_json(run_dir / "overfit_gate.json", report)
    return report


def _run(args: argparse.Namespace) -> int:
    show_progress = not bool(args.no_progress)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    exact_backend = configure_exact_torch_backend(device)
    dynamics_config, base_d0_config = _make_configs(args)
    thresholds = _thresholds(args)
    images, labels = _load_dataset(args)
    source_image = _select_source_image(
        images,
        labels,
        label=int(args.single_image_label),
        class_index=int(args.single_image_index),
        grid_size=int(args.grid_size),
        lambda_mix=float(args.lambda_mix),
    )
    upstream = verify_zero_residual_run(
        args.zero_residual_run_dir,
        dynamics_config=dynamics_config,
        d0_config=base_d0_config,
    )
    runtime = _runtime_record(device, exact_backend)
    source_hash, source_paths = _source_fingerprint()
    scientific = _scientific_payload(
        args,
        dynamics_config,
        base_d0_config,
        thresholds,
        source_image,
        upstream,
    )
    scientific_hash = config_fingerprint(scientific)
    run_dir, resumed = _make_run_dir(args)
    candidate_manifest = _initial_manifest(
        scientific_payload=scientific,
        scientific_fingerprint=scientific_hash,
        source_hash=source_hash,
        source_paths=source_paths,
        runtime=runtime,
        upstream=upstream,
        run_dir=run_dir,
    )
    manifest = _load_or_create_manifest(
        run_dir, resumed=resumed, candidate=candidate_manifest
    )
    previous_status = _json_load(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
    _write_status(
        run_dir,
        status="running",
        phase="initialization",
        stage=str(args.stage),
        require_gate=str(args.require_gate),
        attempt_count=int(previous_status.get("attempt_count", 0)) + 1,
        scientific_fingerprint=scientific_hash,
    )

    skips: list[dict[str, Any]] = []
    cache_metrics: dict[str, Any] | None = None
    optimization_metrics: dict[str, Any] | None = None
    reconstruction_metrics: dict[str, Any] | None = None
    strict_checkpoint_eligible = True

    preflight_path = run_dir / "cache_preflight.json"
    if preflight_path.is_file():
        preflight = _json_load(preflight_path)
        expected_preflight_fingerprints = _artifact_fingerprints(manifest)
        if dict(preflight.get("fingerprints", {})) != expected_preflight_fingerprints:
            raise ArtifactCompatibilityError(
                "cache preflight fingerprint mapping does not match the run manifest"
            )
        preflight_artifact = load_cache_bundle(
            run_dir / "cache_preflight.npz",
            expected_fingerprints=expected_preflight_fingerprints,
            verify_content=True,
        )
        _validate_cache_source(preflight_artifact.cache, source_image)
        preflight_config = replace(
            base_d0_config,
            physical_target_scale=float(
                preflight_artifact.cache.physical_target_scale
            ),
        )
        recomputed_metrics = _cache_gate_metrics(
            preflight_artifact.cache,
            dynamics_config,
            preflight_config,
            device=device,
        )
        recomputed_gate = evaluate_overfit_gates(
            cache_metrics=recomputed_metrics,
            optimization_metrics=None,
            reconstruction_metrics=None,
            require_gate="cache",
            thresholds=thresholds,
        )["cache"]
        _validate_preflight_binding(
            preflight,
            preflight_artifact,
            recomputed_metrics=recomputed_metrics,
            recomputed_gate=recomputed_gate,
        )
        cache_metrics = dict(recomputed_metrics)
        cache_gate = dict(recomputed_gate)
    else:
        preflight, cache_gate = run_cache_preflight(
            run_dir=run_dir,
            images=images,
            labels=labels,
            dynamics_config=dynamics_config,
            d0_config=base_d0_config,
            args=args,
            manifest=manifest,
            source_image=source_image,
            thresholds=thresholds,
            device=device,
            show_progress=show_progress,
        )
        cache_metrics = dict(preflight["metrics"])

    if args.stage == "cache-preflight":
        report = _finalize_gate(
            run_dir=run_dir,
            cache_metrics=cache_metrics,
            optimization_metrics=None,
            reconstruction_metrics=None,
            require_gate=str(args.require_gate),
            thresholds=thresholds,
            strict_checkpoint_eligible=True,
            skips=skips,
        )
        passed = bool(int(report["required_gate_pass"]))
        _write_status(
            run_dir,
            status="complete",
            phase="complete",
            outcome="gate_passed" if passed else "gate_failed",
            required_gate_pass=int(passed),
            stage_skips=skips,
            completed_at=_utcish_now(),
        )
        return 0 if passed else 2

    cache_pass = bool(int(cache_gate.get("passed", 0)))
    if not cache_pass and args.require_gate != "none":
        skips.append(
            {
                "stage": "train/evaluate",
                "reason": "cache preflight failed; expensive downstream work skipped",
            }
        )
        report = _finalize_gate(
            run_dir=run_dir,
            cache_metrics=cache_metrics,
            optimization_metrics=None,
            reconstruction_metrics=None,
            require_gate=str(args.require_gate),
            thresholds=thresholds,
            strict_checkpoint_eligible=True,
            skips=skips,
        )
        _write_status(
            run_dir,
            status="complete",
            phase="complete",
            outcome="gate_failed",
            required_gate_pass=0,
            stage_skips=skips,
            completed_at=_utcish_now(),
        )
        return 2

    cache, d0_config, split, strict_fingerprints = ensure_full_cache(
        run_dir=run_dir,
        images=images,
        labels=labels,
        dynamics_config=dynamics_config,
        d0_config=base_d0_config,
        args=args,
        manifest=manifest,
        source_image=source_image,
        device=device,
        show_progress=show_progress,
    )

    selected_checkpoint: Path | None = None
    if args.stage in {"train", "all"} and args.checkpoint_path is None:
        selected_checkpoint, _selection, optimization_metrics = train_one_image_model(
            run_dir=run_dir,
            cache=cache,
            split=split,
            dynamics_config=dynamics_config,
            d0_config=d0_config,
            strict_fingerprints=strict_fingerprints,
            args=args,
            device=device,
            show_progress=show_progress,
        )
    else:
        optimization_metrics = _selection_optimization_metrics(run_dir)

    if args.stage == "train":
        report = _finalize_gate(
            run_dir=run_dir,
            cache_metrics=cache_metrics,
            optimization_metrics=optimization_metrics,
            reconstruction_metrics=None,
            require_gate=str(args.require_gate),
            thresholds=thresholds,
            strict_checkpoint_eligible=True,
            skips=skips,
        )
        passed = bool(int(report["required_gate_pass"]))
        _write_status(
            run_dir,
            status="complete",
            phase="complete",
            outcome="gate_passed" if passed else "gate_failed",
            required_gate_pass=int(passed),
            stage_skips=skips,
            completed_at=_utcish_now(),
        )
        return 0 if passed else 2

    if args.stage in {"evaluate", "all"}:
        report_only = args.checkpoint_path is not None
        if args.checkpoint_path is not None:
            selected_checkpoint = Path(args.checkpoint_path)
        elif selected_checkpoint is None:
            selected_checkpoint = run_dir / "checkpoints" / "best_ema.pt"
        if not selected_checkpoint.is_file():
            raise FileNotFoundError(f"selected checkpoint does not exist: {selected_checkpoint}")
        optimization_probe = evaluate_overfit_gates(
            cache_metrics=cache_metrics,
            optimization_metrics=optimization_metrics,
            reconstruction_metrics=None,
            require_gate="optimization",
            thresholds=thresholds,
        )
        optimization_pass = bool(
            int(optimization_probe["cumulative_pass"]["optimization"])
        )
        if (
            not report_only
            and not optimization_pass
            and args.require_gate == "reconstruction"
        ):
            skips.append(
                {
                    "stage": "evaluate",
                    "reason": "held-out optimization gate failed; paired production sampling skipped",
                }
            )
        else:
            reconstruction_metrics, checkpoint_optimization, strict_checkpoint_eligible = evaluate_one_image_model(
                run_dir=run_dir,
                checkpoint_path=selected_checkpoint,
                report_only=report_only,
                cache=cache,
                split=split,
                source_image=source_image,
                dynamics_config=dynamics_config,
                d0_config=d0_config,
                strict_fingerprints=strict_fingerprints,
                thresholds=thresholds,
                args=args,
                device=device,
                show_progress=show_progress,
            )
            if checkpoint_optimization:
                optimization_metrics = checkpoint_optimization

    report = _finalize_gate(
        run_dir=run_dir,
        cache_metrics=cache_metrics,
        optimization_metrics=optimization_metrics,
        reconstruction_metrics=reconstruction_metrics,
        require_gate=str(args.require_gate),
        thresholds=thresholds,
        strict_checkpoint_eligible=strict_checkpoint_eligible,
        skips=skips,
    )
    passed = bool(int(report["required_gate_pass"]))
    outcome = "gate_passed" if passed else "gate_failed"
    if args.checkpoint_path is not None:
        outcome = "report_only_complete" if passed else "report_only_failed"
    manifest = _json_load(run_dir / "run_manifest.json")
    manifest.setdefault("artifacts", {}).update(
        {
            "overfit_gate": str(run_dir / "overfit_gate.json"),
            "run_status": str(run_dir / "run_status.json"),
            "checkpoint": None if selected_checkpoint is None else str(selected_checkpoint),
            "checkpoint_sha256": (
                None
                if selected_checkpoint is None or not selected_checkpoint.is_file()
                else file_fingerprint(selected_checkpoint)
            ),
        }
    )
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    _write_status(
        run_dir,
        status="complete",
        phase="complete",
        outcome=outcome,
        required_gate_pass=int(passed),
        stage_skips=skips,
        completed_at=_utcish_now(),
    )
    return 0 if passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir: Path | None = Path(args.resume_run_dir) if args.resume_run_dir is not None else None
    try:
        return _run(args)
    except Exception as exc:
        resolved_run_dir = getattr(args, "_resolved_run_dir", run_dir)
        run_dir = None if resolved_run_dir is None else Path(resolved_run_dir)
        if run_dir is not None and run_dir.is_dir():
            try:
                _write_status(
                    run_dir,
                    status="failed",
                    phase="error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
