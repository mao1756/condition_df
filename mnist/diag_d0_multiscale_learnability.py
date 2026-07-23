from __future__ import annotations

"""Optimization-only multiscale learnability gate for Experiment 12 D0.

The strict elementary direct-Doob workflow remains unchanged.  This command
constructs trajectory-summed temporal block residuals, trains one independent
predictor per block length and seed, and stops before sampling.  A block result
is evidence about a finite-step conditional first moment, not an exact reverse
kernel or an instantaneous Doob field.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    array_fingerprint,
    atomic_copy_file,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    capture_rng_state,
    configure_exact_torch_backend,
    config_fingerprint,
    file_fingerprint,
    restore_rng_state,
    source_fingerprint,
)
from mnist.d0_multiscale_cache import (
    D0MultiscaleCache,
    D0MultiscaleCacheIndex,
    D0MultiscaleCompatibilityError,
    D0MultiscaleShardRecord,
    MULTISCALE_TARGET_CONTRACT,
    block_residual_targets,
    build_multiscale_cache_shard,
    deterministic_three_way_path_split,
    evaluate_multiscale_cache_preflight,
    infer_training_block_scales,
    load_multiscale_cache_index,
    load_multiscale_cache_shard,
    make_multiscale_cache_index,
    make_stratified_anchor_plan,
    multiscale_cache_fingerprint,
    save_multiscale_cache_index,
    save_multiscale_cache_shard,
    slice_anchor_plan,
    validate_multiscale_cache,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    DirectFluxUNet,
    _cuda_autocast,
    _disable_mkldnn_for_cpu_if_needed,
    _make_cuda_grad_scaler,
    edge_alpha_value,
    load_mnist_measure_dataset,
    project_edge_flux_torch,
    temporary_ema_weights,
    update_ema_state,
)
from mnist.experiment12_d0 import (
    Experiment12D0Config,
    synthetic_digit_measures,
)
from mnist.d0_multiscale_gate import (
    MultiscaleGateThresholds,
    compute_multiscale_split_metrics,
    evaluate_multiscale_gates,
    evaluate_stride_pass,
    evaluate_teacher_control,
    write_multiscale_gate_artifacts,
)


RUN_SCHEMA = "experiment12-d0-multiscale-learnability"
RUN_SCHEMA_VERSION = 2
CHECKPOINT_SCHEMA = "experiment12-d0-multiscale-training-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
LATEST_SCHEMA = "experiment12-d0-multiscale-latest"
LATEST_SCHEMA_VERSION = 1
TARGET_CONTRACT = MULTISCALE_TARGET_CONTRACT
CLAIM_SCOPE = "finite-block conditional residual-mean learnability for one frozen fixed-grid Euler kernel"

EXPECTED_KERNEL = {
    "grid_size": 28,
    "sample_steps": 512,
    "reference_substeps": 256,
    "tau_eff": 5e-5,
    "edge_alpha_mode": "alpha_eff",
    "alpha_eff": 1.0,
    "mass_floor": 1e-7,
    "limiter_fraction": 1.0,
    "lambda_mix": 0.35,
}

PILOT_REQUIRED_DEFAULTS: dict[str, Any] = {
    **EXPECTED_KERNEL,
    "single_image_label": 3,
    "single_image_index": 0,
    "temporal_strides": (1, 16, 64, 256, 1024),
    "cache_paths": 64,
    "anchors_per_path": 32,
    "anchor_bin_counts": (4, 4, 4, 4, 16),
    "train_paths": 40,
    "selection_paths": 12,
    "audit_paths": 12,
    "preflight_paths": 4,
    "cache_shard_paths": 8,
    "cache_seed": 260721,
    "dataset_seed": 260718,
    "split_seed": 260722,
    "training_seeds": (260723, 260724, 260725),
    "bootstrap_seed": 260726,
    "teacher_seed": 260727,
    "bootstrap_reps": 10_000,
    "target_scale_floor": 1e-6,
    "base_channels": 32,
    "batch_size": 128,
    "train_steps": 3_000,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "validation_every": 250,
    "checkpoint_every": 250,
    "validation_batch_size": 128,
    "teacher_steps": 1_500,
    "teacher_min_gain": 0.90,
    "bootstrap_confidence": 0.90,
    "max_raw_intervention": 0.005,
    "max_weighted_intervention": 0.0005,
    "max_floor_correction_l1": 1e-8,
    "max_renorm_correction_l1": 1e-6,
    "max_simplex_mass_error": 2e-6,
    "no_amp": False,
}

CONFIRMATION_REQUIRED_DEFAULTS: dict[str, Any] = {
    **PILOT_REQUIRED_DEFAULTS,
    "cache_paths": 128,
    "train_paths": 80,
    "selection_paths": 24,
    "audit_paths": 24,
    "cache_seed": 260728,
    "split_seed": 260729,
    "training_seeds": (260730, 260731, 260732, 260733, 260734),
    "bootstrap_seed": 260735,
    "teacher_seed": 260736,
}

STUDY_PROFILES: dict[str, dict[str, Any]] = {
    "pilot": {
        "version": 1,
        "defaults": PILOT_REQUIRED_DEFAULTS,
        "confirmation_number": 0,
    },
    "confirmation": {
        "version": 1,
        "defaults": CONFIRMATION_REQUIRED_DEFAULTS,
        "confirmation_number": 1,
    },
}

# Backward-compatible public alias used by existing callers and tests.  New
# code should select defaults through STUDY_PROFILES.
REQUIRED_DEFAULTS = PILOT_REQUIRED_DEFAULTS


def _parse_csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        parsed = tuple(int(item) for item in value)
    if not parsed:
        raise ValueError("at least one integer is required")
    return parsed


def _json_load(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path_obj}")
    return value


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _semantic_close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _device_from_arg(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        path = Path(args.resume_run_dir)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(args.runs_root) / f"{stamp}_{args.run_name}"
    path.mkdir(parents=True, exist_ok=False)
    return path, False


def _write_status(path: Path, **updates: Any) -> dict[str, Any]:
    current = _json_load(path / "run_status.json") if (path / "run_status.json").is_file() else {}
    current.update(updates)
    current.setdefault("schema", RUN_SCHEMA)
    current.setdefault("schema_version", RUN_SCHEMA_VERSION)
    current["updated_at"] = _now()
    atomic_write_json(path / "run_status.json", current)
    return current


def _runtime_record(device: torch.device, backend: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_type": device.type,
        "exact_backend": dict(backend),
    }
    if device.type == "cuda":
        result.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_count": int(torch.cuda.device_count()),
            }
        )
    return result


def _source_fingerprint() -> tuple[str, list[str]]:
    here = Path(__file__).resolve()
    names = (
        here,
        here.with_name("d0_multiscale_cache.py"),
        here.with_name("d0_multiscale_gate.py"),
        here.with_name("d0_one_image_gate.py"),
        here.with_name("experiment12_d0.py"),
        here.with_name("eulerian_flux_mnist.py"),
        here.with_name("weighted_point_cloud.py"),
    )
    paths = [path for path in names if path.is_file()]
    return source_fingerprint(paths), [str(path) for path in paths]


def _make_dynamics(args: argparse.Namespace) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
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


def _load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if bool(args.synthetic_data):
        return synthetic_digit_measures(
            examples_per_class=int(args.synthetic_examples_per_class),
            grid_size=int(args.grid_size),
            seed=int(args.dataset_seed),
        )
    dataset = load_mnist_measure_dataset(
        args.data_root,
        max_train=args.max_train,
        examples_per_class=args.examples_per_class,
        download=bool(args.download),
        seed=int(args.dataset_seed),
    )
    return (
        np.asarray(dataset.train_images, dtype=np.float64),
        np.asarray(dataset.train_labels, dtype=np.int64),
    )


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value.astype(np.float32, copy=False))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _select_source_image(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    label: int,
    class_index: int,
    grid_size: int,
    lambda_mix: float,
) -> dict[str, Any]:
    choices = np.flatnonzero(np.asarray(labels, dtype=np.int64) == int(label))
    if choices.size <= int(class_index) or int(class_index) < 0:
        raise ValueError("selected single-image class occurrence is unavailable")
    dataset_index = int(choices[int(class_index)])
    source_input = np.asarray(images[dataset_index], dtype=np.float64).reshape(-1)
    if source_input.size != int(grid_size) ** 2:
        raise ValueError("selected image has the wrong grid size")
    # Preserve the parent one-image workflow's operation order exactly so the
    # frozen float32 image hashes remain identical across the two commands.
    mixed = (1.0 - float(lambda_mix)) * source_input + float(lambda_mix) / float(
        source_input.size
    )
    mixed = np.maximum(mixed, 0.0)
    mixed /= max(float(mixed.sum()), 1e-30)
    image = np.maximum(source_input, 0.0)
    image /= max(float(image.sum()), 1e-30)
    return {
        "dataset_index": dataset_index,
        "class_index": int(class_index),
        "label": int(label),
        "image": image.astype(np.float32),
        "mixed_target": mixed.astype(np.float32),
        "image_sha256": _array_digest(image),
        "mixed_target_sha256": _array_digest(mixed),
    }


def verify_zero_residual_run(path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(path)
    aggregate_path = run_dir / "aggregate_summary.json"
    config_path = run_dir / "run_config.json"
    status_path = run_dir / "run_status.json"
    for required in (aggregate_path, config_path, status_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    aggregate = _json_load(aggregate_path)
    config = _json_load(config_path)
    status = _json_load(status_path)
    if int(aggregate.get("training_ready", 0)) != 1 or int(status.get("required_gate_pass", 0)) != 1:
        raise ArtifactCompatibilityError("upstream zero-residual run did not pass training-ready")
    dynamics = dict(config.get("dynamics_config", {}))
    diagnostic = dict(config.get("diagnostic_config", {}))
    levels = [int(item) for item in diagnostic.get("substep_levels", [])]
    actual = {
        "grid_size": dynamics.get("grid_size"),
        "sample_steps": diagnostic.get("sample_steps"),
        "reference_substeps": max(levels) if levels else None,
        "tau_eff": diagnostic.get("tau_eff"),
        "edge_alpha_mode": dynamics.get("edge_alpha_mode"),
        "alpha_eff": dynamics.get("alpha_eff"),
        "mass_floor": dynamics.get("mass_floor"),
        "limiter_fraction": dynamics.get("limiter_fraction"),
    }
    for key in (
        "grid_size",
        "sample_steps",
        "reference_substeps",
        "tau_eff",
        "edge_alpha_mode",
        "alpha_eff",
        "mass_floor",
        "limiter_fraction",
    ):
        if not _semantic_close(actual.get(key), EXPECTED_KERNEL[key]):
            raise ArtifactCompatibilityError(
                f"upstream zero-residual {key}={actual.get(key)!r}, expected {EXPECTED_KERNEL[key]!r}"
            )
        if not _semantic_close(getattr(args, key), EXPECTED_KERNEL[key]):
            raise ArtifactCompatibilityError(
                f"current {key}={getattr(args, key)!r}, expected {EXPECTED_KERNEL[key]!r}"
            )
    fingerprint = str(
        aggregate.get("config_fingerprint")
        or status.get("config_fingerprint")
        or config.get("config_fingerprint")
        or file_fingerprint(config_path)
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "config_fingerprint": fingerprint,
        "semantic_kernel": actual,
        "aggregate_sha256": file_fingerprint(aggregate_path),
        "run_config_sha256": file_fingerprint(config_path),
        "run_status_sha256": file_fingerprint(status_path),
    }


def verify_parent_one_image_run(
    path: str | Path,
    *,
    source_image: Mapping[str, Any],
    upstream_fingerprint: str,
) -> dict[str, Any]:
    run_dir = Path(path)
    manifest_path = run_dir / "run_manifest.json"
    status_path = run_dir / "run_status.json"
    gate_path = run_dir / "overfit_gate.json"
    selection_path = run_dir / "checkpoint_selection.json"
    for required in (manifest_path, status_path, gate_path, selection_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = _json_load(manifest_path)
    status = _json_load(status_path)
    gate = _json_load(gate_path)
    if manifest.get("schema") != "experiment12-d0-production-one-image":
        raise ArtifactCompatibilityError("parent run is not the production one-image workflow")
    if status.get("status") != "complete" or status.get("outcome") != "gate_failed":
        raise ArtifactCompatibilityError("parent one-image run is not the completed failed gate")
    if int(dict(gate.get("cache", {})).get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("parent one-image cache gate did not pass")
    if int(dict(gate.get("optimization", {})).get("passed", 1)) != 0:
        raise ArtifactCompatibilityError("parent one-image optimization was not the recorded failure")
    scientific = dict(manifest.get("scientific_config", {}))
    parent_source = dict(scientific.get("source_image", {}))
    for key in ("label", "class_index", "dataset_index", "image_sha256", "mixed_target_sha256"):
        if parent_source.get(key) != source_image.get(key):
            raise ArtifactCompatibilityError(f"parent source-image mismatch for {key}")
    if scientific.get("upstream_zero_residual_fingerprint") != str(upstream_fingerprint):
        raise ArtifactCompatibilityError("parent and current upstream zero-residual fingerprints differ")
    kernel = dict(scientific.get("kernel", {}))
    for key, expected in EXPECTED_KERNEL.items():
        if key == "lambda_mix":
            actual = kernel.get(key)
        else:
            actual = kernel.get(key)
        if not _semantic_close(actual, expected):
            raise ArtifactCompatibilityError(f"parent kernel mismatch for {key}: {actual!r}")
    selection = _json_load(selection_path)
    parent_fingerprint = str(manifest.get("scientific_fingerprint", ""))
    cache_fingerprint = str(
        dict(manifest.get("artifacts", {}))
        .get("cache", {})
        .get("cache_fingerprint", "")
    )
    if not parent_fingerprint or not cache_fingerprint:
        raise ArtifactCompatibilityError(
            "parent one-image run is missing scientific or cache provenance"
        )
    return {
        "run_dir": str(run_dir.resolve()),
        "scientific_fingerprint": parent_fingerprint,
        "manifest_sha256": file_fingerprint(manifest_path),
        "gate_sha256": file_fingerprint(gate_path),
        "status_sha256": file_fingerprint(status_path),
        "selection_sha256": file_fingerprint(selection_path),
        "cache_fingerprint": cache_fingerprint,
        "selected_step": int(dict(selection.get("selected", {})).get("step", -1)),
        "selected_prediction_gain": float(dict(selection.get("selected", {})).get("prediction_gain", float("nan"))),
        "selected_data_end_gain": float(dict(selection.get("selected", {})).get("data_end_prediction_gain", float("nan"))),
        "source_image": parent_source,
    }


def _require_semantic_fields(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if not _semantic_close(actual_value, expected_value):
            raise ArtifactCompatibilityError(
                f"{context} mismatch for {key}: {actual_value!r}, expected {expected_value!r}"
            )


def verify_parent_multiscale_run(
    path: str | Path,
    *,
    source_image: Mapping[str, Any],
    upstream_fingerprint: str,
    parent_one_image_fingerprint: str,
) -> dict[str, Any]:
    """Verify the completed schema-v1 pilot used to authorize confirmation.

    The pilot's source-code hash is deliberately not compared with the current
    command: adding this verifier necessarily changes that hash.  Scientific
    configuration and committed artifact bytes are bound instead.
    """

    run_dir = Path(path)
    manifest_path = run_dir / "run_manifest.json"
    status_path = run_dir / "run_status.json"
    decision_path = run_dir / "learnability_decision.json"
    cache_gate_path = run_dir / "cache_gate.json"
    teacher_path = run_dir / "teacher_control.json"
    stride_seed_path = run_dir / "stride_seed_metrics.csv"
    task_failures_path = run_dir / "task_failures.json"
    for required in (
        manifest_path,
        status_path,
        decision_path,
        cache_gate_path,
        teacher_path,
        stride_seed_path,
        task_failures_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest = _json_load(manifest_path)
    status = _json_load(status_path)
    gate_report = _json_load(decision_path)
    cache_gate = _json_load(cache_gate_path)
    teacher = _json_load(teacher_path)
    task_failures = _json_load(task_failures_path)
    if manifest.get("schema") != RUN_SCHEMA:
        raise ArtifactCompatibilityError("parent multiscale run has the wrong schema")
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version not in (1, RUN_SCHEMA_VERSION):
        raise ArtifactCompatibilityError("parent multiscale run schema version is unsupported")
    scientific = dict(manifest.get("scientific_config", {}))
    profile_value = scientific.get("study_profile")
    if schema_version == 1:
        if profile_value not in (None, "pilot"):
            raise ArtifactCompatibilityError("legacy parent cannot declare a non-pilot profile")
        inferred_legacy_profile = 1
    else:
        profile_name = (
            str(dict(profile_value).get("name", ""))
            if isinstance(profile_value, Mapping)
            else str(profile_value or "")
        )
        if profile_name != "pilot":
            raise ArtifactCompatibilityError("parent multiscale run is not a pilot profile")
        inferred_legacy_profile = 0

    if status.get("status") != "complete" or status.get("outcome") != "gate_failed":
        raise ArtifactCompatibilityError("parent multiscale pilot is not a completed failed gate")
    if str(status.get("required_gate")) != "any-scale" or int(
        status.get("required_gate_pass", -1)
    ) != 0:
        raise ArtifactCompatibilityError("parent multiscale pilot did not fail the any-scale gate")
    if list(status.get("skips", [])):
        raise ArtifactCompatibilityError("parent multiscale pilot recorded skipped work")
    if int(task_failures.get("failure_count", -1)) != 0 or list(
        task_failures.get("failures", [])
    ):
        raise ArtifactCompatibilityError("parent multiscale pilot recorded task failures")
    decision = dict(gate_report.get("decision", {}))
    if str(decision.get("decision")) != "inconclusive":
        raise ArtifactCompatibilityError("parent multiscale pilot decision is not inconclusive")
    if str(gate_report.get("required_gate")) != "any-scale" or int(
        gate_report.get("required_gate_pass", -1)
    ) != 0:
        raise ArtifactCompatibilityError("parent multiscale gate report is not the failed any-scale gate")
    if int(cache_gate.get("passed", 0)) != 1 or int(
        dict(gate_report.get("cache", {})).get("passed", 0)
    ) != 1:
        raise ArtifactCompatibilityError("parent multiscale cache gate did not pass")
    if int(dict(teacher.get("gate", {})).get("passed", 0)) != 1 or int(
        dict(gate_report.get("teacher", {})).get("passed", 0)
    ) != 1:
        raise ArtifactCompatibilityError("parent multiscale teacher gate did not pass")
    stride_gates = dict(gate_report.get("strides", {}))
    expected_stride_keys = {str(value) for value in PILOT_REQUIRED_DEFAULTS["temporal_strides"]}
    if set(stride_gates) != expected_stride_keys or any(
        int(dict(gate).get("passed", 1)) != 0 for gate in stride_gates.values()
    ):
        raise ArtifactCompatibilityError(
            "parent multiscale pilot stride gates are not the frozen all-failed result"
        )
    if any(
        int(record.get("sampling_performed", 0)) != 0
        for record in (manifest, status, gate_report, decision)
    ):
        raise ArtifactCompatibilityError("parent multiscale pilot performed sampling")

    parent_source = dict(scientific.get("source_image", {}))
    for key in (
        "label",
        "class_index",
        "dataset_index",
        "image_sha256",
        "mixed_target_sha256",
    ):
        if parent_source.get(key) != source_image.get(key):
            raise ArtifactCompatibilityError(
                f"parent multiscale source-image mismatch for {key}"
            )
    if scientific.get("upstream_zero_residual_fingerprint") != str(
        upstream_fingerprint
    ):
        raise ArtifactCompatibilityError(
            "parent multiscale and current upstream zero-residual fingerprints differ"
        )
    if scientific.get("parent_one_image_fingerprint") != str(
        parent_one_image_fingerprint
    ):
        raise ArtifactCompatibilityError(
            "parent multiscale and current one-image parent fingerprints differ"
        )
    if scientific.get("target_contract") != TARGET_CONTRACT:
        raise ArtifactCompatibilityError("parent multiscale target contract differs")

    _require_semantic_fields(
        dict(scientific.get("kernel", {})),
        {
            **EXPECTED_KERNEL,
            "edge_alpha_value": 1.0,
            "integrator": "masked_reference_free_step_torch",
        },
        context="parent multiscale kernel",
    )
    pilot = PILOT_REQUIRED_DEFAULTS
    _require_semantic_fields(
        dict(scientific.get("cache", {})),
        {
            "temporal_strides": list(pilot["temporal_strides"]),
            "paths": pilot["cache_paths"],
            "anchors_per_path": pilot["anchors_per_path"],
            "anchor_bin_counts": list(pilot["anchor_bin_counts"]),
            "preflight_paths": pilot["preflight_paths"],
            "shard_paths": pilot["cache_shard_paths"],
            "cache_seed": pilot["cache_seed"],
            "dataset_seed": pilot["dataset_seed"],
            "split_seed": pilot["split_seed"],
            "train_paths": pilot["train_paths"],
            "selection_paths": pilot["selection_paths"],
            "audit_paths": pilot["audit_paths"],
            "target_scale_floor": pilot["target_scale_floor"],
        },
        context="parent multiscale cache",
    )
    _require_semantic_fields(
        dict(scientific.get("training", {})),
        {
            "seeds": list(pilot["training_seeds"]),
            "teacher_seed": pilot["teacher_seed"],
            "steps": pilot["train_steps"],
            "teacher_steps": pilot["teacher_steps"],
            "batch_size": pilot["batch_size"],
            "base_channels": pilot["base_channels"],
            "learning_rate": pilot["learning_rate"],
            "weight_decay": pilot["weight_decay"],
            "grad_clip": pilot["grad_clip"],
            "ema_decay": pilot["ema_decay"],
            "validation_every": pilot["validation_every"],
            "checkpoint_every": pilot["checkpoint_every"],
            "validation_batch_size": pilot["validation_batch_size"],
            "use_amp": not bool(pilot["no_amp"]),
        },
        context="parent multiscale training",
    )
    _require_semantic_fields(
        dict(scientific.get("gate", {})),
        {
            "bootstrap_seed": pilot["bootstrap_seed"],
            "bootstrap_reps": pilot["bootstrap_reps"],
            "bootstrap_confidence": pilot["bootstrap_confidence"],
            "teacher_min_gain": pilot["teacher_min_gain"],
            "max_raw_intervention": pilot["max_raw_intervention"],
            "max_weighted_intervention": pilot["max_weighted_intervention"],
            "max_floor_correction_l1": pilot["max_floor_correction_l1"],
            "max_renorm_correction_l1": pilot["max_renorm_correction_l1"],
            "max_simplex_mass_error": pilot["max_simplex_mass_error"],
        },
        context="parent multiscale gate",
    )

    with stride_seed_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_tasks = {
        (int(stride), int(seed))
        for stride in pilot["temporal_strides"]
        for seed in pilot["training_seeds"]
    }
    completed_tasks: set[tuple[int, int]] = set()
    task_rows_valid = True
    for row in rows:
        try:
            task = (int(row["stride"]), int(row["training_seed"]))
            valid = (
                int(row.get("complete", 0)) == 1
                and int(row.get("finite", 0)) == 1
                and int(row.get("selected_step", 0)) > 0
            )
        except (KeyError, TypeError, ValueError):
            task_rows_valid = False
            continue
        task_rows_valid = task_rows_valid and valid
        completed_tasks.add(task)
    if (
        len(rows) != len(expected_tasks)
        or len(completed_tasks) != len(rows)
        or completed_tasks != expected_tasks
        or not task_rows_valid
    ):
        raise ArtifactCompatibilityError(
            "parent multiscale pilot does not contain all 15 complete, finite, selected physical tasks"
        )

    scientific_fingerprint = str(manifest.get("scientific_fingerprint", ""))
    cache_semantic_fingerprint = str(
        manifest.get("cache_semantic_fingerprint", "")
    )
    if not scientific_fingerprint or not cache_semantic_fingerprint:
        raise ArtifactCompatibilityError(
            "parent multiscale pilot is missing scientific or cache provenance"
        )
    return {
        "run_dir": str(run_dir.resolve()),
        "schema_version": schema_version,
        "study_profile": "pilot",
        "study_profile_version": int(STUDY_PROFILES["pilot"]["version"]),
        "profile_inferred_from_legacy": inferred_legacy_profile,
        "scientific_fingerprint": scientific_fingerprint,
        "cache_semantic_fingerprint": cache_semantic_fingerprint,
        "cache_index_fingerprint": str(
            dict(manifest.get("artifacts", {})).get("cache_index_fingerprint", "")
        ),
        "manifest_sha256": file_fingerprint(manifest_path),
        "status_sha256": file_fingerprint(status_path),
        "decision_sha256": file_fingerprint(decision_path),
        "cache_gate_sha256": file_fingerprint(cache_gate_path),
        "teacher_control_sha256": file_fingerprint(teacher_path),
        "stride_seed_metrics_sha256": file_fingerprint(stride_seed_path),
        "task_failures_sha256": file_fingerprint(task_failures_path),
        "decision_path": str(decision_path.resolve()),
        "stride_seed_metrics_path": str(stride_seed_path.resolve()),
        "decision": "inconclusive",
        "task_count": len(completed_tasks),
        "source_image": parent_source,
        "seed_plan": {
            "dataset_seed": int(pilot["dataset_seed"]),
            "cache_seed": int(pilot["cache_seed"]),
            "split_seed": int(pilot["split_seed"]),
            "training_seeds": list(pilot["training_seeds"]),
            "bootstrap_seed": int(pilot["bootstrap_seed"]),
            "teacher_seed": int(pilot["teacher_seed"]),
        },
    }


def _confirmation_independence_provenance(
    args: argparse.Namespace,
    parent_multiscale: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if str(args.study_profile) != "confirmation":
        return {
            "required": 0,
            "passed": 1,
            "checks": {},
        }
    if parent_multiscale is None:
        raise ArtifactCompatibilityError(
            "confirmation profile is missing parent multiscale provenance"
        )
    pilot = dict(parent_multiscale.get("seed_plan", {}))
    current_training = {int(value) for value in args.training_seeds}
    pilot_training = {int(value) for value in pilot.get("training_seeds", [])}
    checks = {
        "dataset_seed_preserved": int(args.dataset_seed)
        == int(pilot.get("dataset_seed", -1)),
        "cache_seed_distinct": int(args.cache_seed)
        != int(pilot.get("cache_seed", args.cache_seed)),
        "split_seed_distinct": int(args.split_seed)
        != int(pilot.get("split_seed", args.split_seed)),
        "training_seeds_disjoint": not bool(current_training & pilot_training),
        "bootstrap_seed_distinct": int(args.bootstrap_seed)
        != int(pilot.get("bootstrap_seed", args.bootstrap_seed)),
        "teacher_seed_distinct": int(args.teacher_seed)
        != int(pilot.get("teacher_seed", args.teacher_seed)),
    }
    return {
        "required": 1,
        "passed": int(all(checks.values())),
        "checks": {key: int(value) for key, value in checks.items()},
        "pilot_seed_plan": pilot,
        "confirmation_seed_plan": {
            "dataset_seed": int(args.dataset_seed),
            "cache_seed": int(args.cache_seed),
            "split_seed": int(args.split_seed),
            "training_seeds": list(args.training_seeds),
            "bootstrap_seed": int(args.bootstrap_seed),
            "teacher_seed": int(args.teacher_seed),
        },
    }


def _scientific_payload(
    args: argparse.Namespace,
    dynamics: DirectFluxMNISTConfig,
    source_image: Mapping[str, Any],
    upstream: Mapping[str, Any],
    parent: Mapping[str, Any],
    parent_multiscale: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile = STUDY_PROFILES[str(args.study_profile)]
    independence = _confirmation_independence_provenance(args, parent_multiscale)
    payload = {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "study_profile": {
            "name": str(args.study_profile),
            "version": int(profile["version"]),
            "confirmation_number": int(profile["confirmation_number"]),
            "profile_conformant": int(bool(args.profile_conformant)),
        },
        "target_contract": TARGET_CONTRACT,
        "kernel": {
            "grid_size": int(args.grid_size),
            "sample_steps": int(args.sample_steps),
            "reference_substeps": int(args.reference_substeps),
            "tau_eff": float(args.tau_eff),
            "edge_alpha_mode": str(args.edge_alpha_mode),
            "edge_alpha_value": float(edge_alpha_value(dynamics)),
            "alpha_eff": float(args.alpha_eff),
            "mass_floor": float(args.mass_floor),
            "limiter_fraction": float(args.limiter_fraction),
            "lambda_mix": float(args.lambda_mix),
            "integrator": "masked_reference_free_step_torch",
        },
        "cache": {
            "temporal_strides": list(args.temporal_strides),
            "paths": int(args.cache_paths),
            "anchors_per_path": int(args.anchors_per_path),
            "anchor_bin_counts": list(args.anchor_bin_counts),
            "preflight_paths": int(args.preflight_paths),
            "shard_paths": int(args.cache_shard_paths),
            "cache_seed": int(args.cache_seed),
            "dataset_seed": int(args.dataset_seed),
            "split_seed": int(args.split_seed),
            "train_paths": int(args.train_paths),
            "selection_paths": int(args.selection_paths),
            "audit_paths": int(args.audit_paths),
            "target_scale_floor": float(args.target_scale_floor),
        },
        "training": {
            "seeds": list(args.training_seeds),
            "teacher_seed": int(args.teacher_seed),
            "steps": int(args.train_steps),
            "teacher_steps": int(args.teacher_steps),
            "batch_size": int(args.batch_size),
            "base_channels": int(args.base_channels),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "ema_decay": float(args.ema_decay),
            "validation_every": int(args.validation_every),
            "checkpoint_every": int(args.checkpoint_every),
            "validation_batch_size": int(args.validation_batch_size),
            "use_amp": not bool(args.no_amp),
        },
        "gate": {
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_reps": int(args.bootstrap_reps),
            "bootstrap_confidence": float(args.bootstrap_confidence),
            "teacher_min_gain": float(args.teacher_min_gain),
            "max_raw_intervention": float(args.max_raw_intervention),
            "max_weighted_intervention": float(args.max_weighted_intervention),
            "max_floor_correction_l1": float(args.max_floor_correction_l1),
            "max_renorm_correction_l1": float(args.max_renorm_correction_l1),
            "max_simplex_mass_error": float(args.max_simplex_mass_error),
        },
        "source_image": {
            key: source_image[key]
            for key in ("label", "class_index", "dataset_index", "image_sha256", "mixed_target_sha256")
        },
        "upstream_zero_residual_fingerprint": str(upstream["config_fingerprint"]),
        "parent_one_image_fingerprint": str(parent["scientific_fingerprint"]),
        "parent_multiscale_fingerprint": (
            None
            if parent_multiscale is None
            else str(parent_multiscale["scientific_fingerprint"])
        ),
        "upstream_zero_residual_provenance": {
            key: upstream[key]
            for key in (
                "config_fingerprint",
                "aggregate_sha256",
                "run_config_sha256",
                "run_status_sha256",
            )
        },
        "parent_one_image_provenance": {
            key: parent[key]
            for key in (
                "scientific_fingerprint",
                "manifest_sha256",
                "gate_sha256",
                "status_sha256",
                "selection_sha256",
                "cache_fingerprint",
                "selected_step",
                "selected_prediction_gain",
                "selected_data_end_gain",
            )
        },
        "parent_multiscale_provenance": (
            None
            if parent_multiscale is None
            else {
                key: parent_multiscale[key]
                for key in (
                    "schema_version",
                    "study_profile",
                    "study_profile_version",
                    "profile_inferred_from_legacy",
                    "scientific_fingerprint",
                    "cache_semantic_fingerprint",
                    "cache_index_fingerprint",
                    "manifest_sha256",
                    "status_sha256",
                    "decision_sha256",
                    "cache_gate_sha256",
                    "teacher_control_sha256",
                    "stride_seed_metrics_sha256",
                    "task_failures_sha256",
                    "decision",
                    "task_count",
                )
            }
        ),
        "independence_provenance": independence,
        "claim_scope": CLAIM_SCOPE,
        "sampling_performed": 0,
    }
    return payload


def _profile_mismatches(args: argparse.Namespace) -> list[str]:
    profile = STUDY_PROFILES[str(args.study_profile)]
    mismatches: list[str] = []
    for name, expected in dict(profile["defaults"]).items():
        actual = getattr(args, name)
        if isinstance(expected, float):
            matches = _semantic_close(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            mismatches.append(f"--{name.replace('_', '-')}={actual!r} (required {expected!r})")
    return mismatches


def _validate_required_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    mismatches = _profile_mismatches(args)
    args.profile_conformant = not bool(mismatches) and not bool(args.synthetic_data)
    if str(args.require_gate) == "none":
        return
    if bool(args.synthetic_data):
        parser.error("a required multiscale gate must use MNIST")
    if mismatches:
        parser.error(
            "required gates freeze the production learnability profile "
            f"({args.study_profile}); use --require-gate none for exploration: "
            + "; ".join(mismatches)
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = None if argv is None else list(argv)
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument(
        "--study-profile",
        choices=tuple(STUDY_PROFILES),
        default="pilot",
    )
    profile_args, _ = profile_parser.parse_known_args(raw_argv)
    selected_profile = str(profile_args.study_profile)
    profile_defaults = dict(STUDY_PROFILES[selected_profile]["defaults"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-profile",
        choices=tuple(STUDY_PROFILES),
        default=selected_profile,
    )
    parser.add_argument("--stage", choices=("cache-preflight", "cache", "train", "report", "all"), default="all")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_d0_multiscale"))
    parser.add_argument("--run-name", default="production-multiscale-learnability")
    parser.add_argument("--zero-residual-run-dir", type=Path, required=True)
    parser.add_argument("--parent-one-image-run-dir", type=Path, required=True)
    parser.add_argument("--parent-multiscale-run-dir", type=Path, default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--cache-run-dir", type=Path, default=None)
    parser.add_argument("--require-gate", choices=("none", "cache", "teacher", "any-scale", "elementary"), default="none")
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

    parser.add_argument("--temporal-strides", default="1,16,64,256,1024")
    parser.add_argument("--cache-paths", type=int, default=64)
    parser.add_argument("--anchors-per-path", type=int, default=32)
    parser.add_argument("--anchor-bin-counts", default="4,4,4,4,16")
    parser.add_argument("--train-paths", type=int, default=40)
    parser.add_argument("--selection-paths", type=int, default=12)
    parser.add_argument("--audit-paths", type=int, default=12)
    parser.add_argument("--preflight-paths", type=int, default=4)
    parser.add_argument("--cache-shard-paths", type=int, default=8)
    parser.add_argument("--cache-seed", type=int, default=260721)
    parser.add_argument("--dataset-seed", type=int, default=260718)
    parser.add_argument("--split-seed", type=int, default=260722)
    parser.add_argument("--target-scale-floor", type=float, default=1e-6)

    parser.add_argument("--training-seeds", default="260723,260724,260725")
    parser.add_argument("--bootstrap-seed", type=int, default=260726)
    parser.add_argument("--teacher-seed", type=int, default=260727)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=3_000)
    parser.add_argument("--teacher-steps", type=int, default=1_500)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--validation-batch-size", type=int, default=128)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--teacher-min-gain", type=float, default=0.90)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.90)
    parser.add_argument("--max-raw-intervention", type=float, default=0.005)
    parser.add_argument("--max-weighted-intervention", type=float, default=0.0005)
    parser.add_argument("--max-floor-correction-l1", type=float, default=1e-8)
    parser.add_argument("--max-renorm-correction-l1", type=float, default=1e-6)
    parser.add_argument("--max-simplex-mass-error", type=float, default=2e-6)

    parser.set_defaults(**profile_defaults)
    args = parser.parse_args(raw_argv)
    try:
        args.temporal_strides = tuple(sorted(set(_parse_csv_ints(args.temporal_strides))))
        args.anchor_bin_counts = _parse_csv_ints(args.anchor_bin_counts)
        args.training_seeds = _parse_csv_ints(args.training_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if any(item <= 0 for item in args.temporal_strides):
        parser.error("temporal strides must be positive")
    if any(item < 0 for item in args.anchor_bin_counts):
        parser.error("anchor-bin-counts must be non-negative")
    if len(args.anchor_bin_counts) != 5 or sum(args.anchor_bin_counts) != int(args.anchors_per_path):
        parser.error("anchor-bin-counts must contain five counts summing to anchors-per-path")
    if int(args.train_paths) + int(args.selection_paths) + int(args.audit_paths) != int(args.cache_paths):
        parser.error("train, selection, and audit path counts must sum to cache-paths")
    if len(args.training_seeds) <= 0 or len(args.training_seeds) % 2 == 0:
        parser.error("training-seeds must contain a non-empty odd number of seeds")
    if len(set(args.training_seeds)) != len(args.training_seeds):
        parser.error("training seeds must be distinct")
    if str(args.study_profile) == "confirmation":
        if args.parent_multiscale_run_dir is None:
            parser.error(
                "--parent-multiscale-run-dir is required for the confirmation profile"
            )
    elif args.parent_multiscale_run_dir is not None:
        parser.error(
            "--parent-multiscale-run-dir is only valid for the confirmation profile"
        )
    total = int(args.sample_steps) * int(args.reference_substeps)
    if max(args.temporal_strides) >= total:
        parser.error("maximum temporal stride must be shorter than the forward trajectory")
    if any(total % int(stride) != 0 for stride in args.temporal_strides):
        parser.error("every temporal stride must divide the elementary trajectory length")
    for name in (
        "cache_paths",
        "anchors_per_path",
        "preflight_paths",
        "cache_shard_paths",
        "batch_size",
        "train_steps",
        "teacher_steps",
        "validation_every",
        "checkpoint_every",
        "validation_batch_size",
        "bootstrap_reps",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    _validate_required_defaults(args, parser)
    return args


@dataclass(frozen=True)
class TaskArrays:
    """CPU tensors shared by one teacher or physical regression task."""

    states: Tensor
    tau: Tensor
    tau_fraction: Tensor
    labels: Tensor
    targets: Tensor
    path_ids: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.states.shape[0])
        if self.states.ndim != 2:
            raise ValueError("task states must have shape (rows, pixels)")
        if self.tau.shape != (count,) or self.tau_fraction.shape != (count,) or self.labels.shape != (count,):
            raise ValueError("task tau/label rows are inconsistent")
        if self.targets.ndim != 4 or int(self.targets.shape[0]) != count:
            raise ValueError("task targets must have shape (rows, 2, H, W)")
        if np.asarray(self.path_ids).shape != (count,):
            raise ValueError("task path ids are inconsistent")
        if not bool(
            torch.isfinite(self.states).all()
            and torch.isfinite(self.tau).all()
            and torch.isfinite(self.tau_fraction).all()
            and torch.isfinite(self.targets).all()
        ):
            raise ValueError("task arrays must be finite")


def _row_indices(path_ids: np.ndarray, selected_paths: Sequence[int]) -> np.ndarray:
    return np.flatnonzero(
        np.isin(
            np.asarray(path_ids, dtype=np.int64),
            np.asarray(selected_paths, dtype=np.int64),
        )
    ).astype(np.int64)


def _teacher_targets(states: Tensor, tau_fraction: Tensor, *, grid_size: int) -> Tensor:
    grid = states.reshape(-1, int(grid_size), int(grid_size))
    inverse_h = float(grid_size)
    horizontal = (torch.roll(grid, shifts=-1, dims=2) - grid) * inverse_h
    vertical = (torch.roll(grid, shifts=-1, dims=1) - grid) * inverse_h
    edge = torch.stack((horizontal, vertical), dim=1)
    factor = (0.5 + 0.5 * tau_fraction).reshape(-1, 1, 1, 1).to(edge)
    return project_edge_flux_torch(edge * factor, grid_size=int(grid_size))


def make_teacher_task(
    states: Tensor,
    tau: Tensor,
    tau_fraction: Tensor,
    labels: Tensor,
    path_ids: np.ndarray,
    *,
    train_path_ids: Sequence[int],
    grid_size: int,
    scale_floor: float,
) -> tuple[TaskArrays, float]:
    raw = _teacher_targets(states.float(), tau_fraction.float(), grid_size=int(grid_size))
    train_rows = _row_indices(path_ids, train_path_ids)
    if train_rows.size == 0:
        raise ValueError("teacher task has no training rows")
    train_tensor = raw.index_select(0, torch.as_tensor(train_rows, dtype=torch.long))
    scale = max(
        float(torch.sqrt(train_tensor.double().square().mean()).item()),
        float(scale_floor),
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("teacher target scale is not finite and positive")
    return (
        TaskArrays(
            states=states.detach().float().cpu().contiguous(),
            tau=tau.detach().float().cpu().contiguous(),
            tau_fraction=tau_fraction.detach().float().cpu().contiguous(),
            labels=labels.detach().long().cpu().contiguous(),
            targets=(raw / float(scale)).detach().float().cpu().contiguous(),
            path_ids=np.asarray(path_ids, dtype=np.int64).copy(),
        ),
        float(scale),
    )


def _prediction_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[0] == 0:
        raise ValueError("prediction summary expects non-empty matching matrices")
    finite = np.isfinite(target) & np.isfinite(prediction)
    if not bool(finite.all()):
        return {
            "slice_count": int(target.shape[0]),
            "finite_fraction": float(finite.mean()),
            "primary_mse": float("nan"),
            "zero_baseline_mse": float("nan"),
            "prediction_gain": float("nan"),
            "target_prediction_covariance": float("nan"),
        }
    target64 = target.astype(np.float64, copy=False)
    prediction64 = prediction.astype(np.float64, copy=False)
    model_error = float(np.square(prediction64 - target64).sum())
    zero_error = float(np.square(target64).sum())
    covariance = float(
        np.mean(
            (target64 - float(target64.mean()))
            * (prediction64 - float(prediction64.mean()))
        )
    )
    return {
        "slice_count": int(target.shape[0]),
        "finite_fraction": 1.0,
        "primary_mse": model_error / float(target64.size),
        "zero_baseline_mse": zero_error / float(target64.size),
        "prediction_gain": 1.0 - model_error / zero_error if zero_error > 0.0 else float("nan"),
        "target_prediction_covariance": covariance,
    }


@torch.no_grad()
def _predict_task(
    model: nn.Module,
    arrays: TaskArrays,
    dynamics: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    batch_size: int,
    row_indices: Sequence[int] | np.ndarray | Tensor | None = None,
) -> np.ndarray:
    was_training = bool(model.training)
    model.eval()
    pieces: list[np.ndarray] = []
    if row_indices is None:
        rows = torch.arange(int(arrays.states.shape[0]), dtype=torch.long)
    else:
        rows = torch.as_tensor(row_indices, dtype=torch.long).reshape(-1)
        if rows.numel() == 0:
            raise ValueError("prediction row indices must not be empty")
        if bool(((rows < 0) | (rows >= int(arrays.states.shape[0]))).any()):
            raise IndexError("prediction row index is outside the task arrays")
    try:
        for start in range(0, int(rows.numel()), max(1, int(batch_size))):
            stop = min(int(rows.numel()), start + max(1, int(batch_size)))
            selected = rows[start:stop]
            states = arrays.states.index_select(0, selected).to(device)
            tau = arrays.tau.index_select(0, selected).to(device)
            labels = arrays.labels.index_select(0, selected).to(device)
            prediction = project_edge_flux_torch(
                model(tau, states, labels, None),
                grid_size=int(dynamics.grid_size),
            )
            pieces.append(prediction.detach().float().cpu().reshape(stop - start, -1).numpy())
    finally:
        model.train(was_training)
    return np.concatenate(pieces, axis=0)


def _selection_metrics(
    target_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
    tau_fraction: np.ndarray,
    selection_rows: np.ndarray,
) -> dict[str, Any]:
    overall = _prediction_summary(
        target_matrix[selection_rows], prediction_matrix[selection_rows]
    )
    data_rows = selection_rows[tau_fraction[selection_rows] >= 0.8]
    data_end = (
        _prediction_summary(target_matrix[data_rows], prediction_matrix[data_rows])
        if data_rows.size
        else {
            "slice_count": 0,
            "finite_fraction": float("nan"),
            "primary_mse": float("nan"),
            "zero_baseline_mse": float("nan"),
            "prediction_gain": float("nan"),
            "target_prediction_covariance": float("nan"),
        }
    )
    return {
        **overall,
        "data_end_slice_count": int(data_end["slice_count"]),
        "data_end_primary_mse": float(data_end["primary_mse"]),
        "data_end_prediction_gain": float(data_end["prediction_gain"]),
        "data_end_target_prediction_covariance": float(
            data_end["target_prediction_covariance"]
        ),
    }


def _set_task_seed(seed: int) -> np.random.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return np.random.default_rng(int(seed) + 101)


def _checkpoint_fingerprints_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if dict(actual) != dict(expected):
        raise ArtifactCompatibilityError("multiscale task checkpoint fingerprints differ")


def _save_task_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema_state: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    history: Sequence[Mapping[str, Any]],
    validation_records: Sequence[Mapping[str, Any]],
    best_validation: Mapping[str, Any] | None,
    fingerprints: Mapping[str, Any],
    numpy_rng: np.random.Generator,
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": int(step),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "ema_state_dict": {
            name: value.detach().clone() for name, value in ema_state.items()
        },
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scaler_state_dict": copy.deepcopy(scaler.state_dict()),
        "history": copy.deepcopy([dict(row) for row in history]),
        "validation_records": copy.deepcopy(
            [dict(row) for row in validation_records]
        ),
        "best_validation": None
        if best_validation is None
        else copy.deepcopy(dict(best_validation)),
        "fingerprints": copy.deepcopy(dict(fingerprints)),
        "rng_state": capture_rng_state(numpy_rng),
    }
    atomic_torch_save(path, payload)
    return payload


def _load_task_checkpoint(
    path: Path,
    *,
    map_location: torch.device | str,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ArtifactCompatibilityError("multiscale checkpoint must be a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA or int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("legacy/foreign checkpoint is report-only and cannot resume")
    required = {
        "step",
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "history",
        "validation_records",
        "best_validation",
        "fingerprints",
        "rng_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ArtifactCompatibilityError(
            "multiscale checkpoint is incomplete: " + ", ".join(missing)
        )
    _checkpoint_fingerprints_match(payload["fingerprints"], fingerprints)
    return payload


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _best_validation(
    current: Mapping[str, Any] | None, candidate: Mapping[str, Any]
) -> dict[str, Any] | None:
    choices = [] if current is None else [dict(current)]
    choices.append(dict(candidate))
    finite = [
        item
        for item in choices
        if math.isfinite(float(item.get("primary_mse", float("nan"))))
    ]
    if not finite:
        return None
    return min(finite, key=lambda item: (float(item["primary_mse"]), int(item["step"])))


def load_completed_task_result(
    task_dir: Path,
    *,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load a completed task without touching model or RNG state."""

    result_path = task_dir / "task_result.json"
    status_path = task_dir / "task_status.json"
    if not result_path.is_file() or not status_path.is_file():
        return None
    status = _json_load(status_path)
    result = _json_load(result_path)
    if status.get("status") != "complete" or int(result.get("task_complete", 0)) != 1:
        return None
    _checkpoint_fingerprints_match(status.get("fingerprints", {}), fingerprints)
    _checkpoint_fingerprints_match(result.get("fingerprints", {}), fingerprints)
    checkpoint_path = Path(str(result.get("checkpoint_path", "")))
    if not checkpoint_path.is_file():
        raise ArtifactCompatibilityError("completed multiscale task checkpoint is missing")
    if file_fingerprint(checkpoint_path) != str(result.get("checkpoint_sha256", "")):
        raise ArtifactCompatibilityError("completed multiscale task checkpoint hash mismatch")
    return result


def train_task(
    *,
    task_dir: Path,
    task_name: str,
    arrays: TaskArrays,
    split_path_ids: Mapping[str, Sequence[int]],
    dynamics: DirectFluxMNISTConfig,
    training_seed: int,
    train_steps: int,
    args: argparse.Namespace,
    fingerprints: Mapping[str, Any],
    device: torch.device,
    show_progress: bool,
    stride: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train/resume one independent task and return selected EMA metrics."""

    task_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = task_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    completed_result = load_completed_task_result(
        task_dir, fingerprints=fingerprints
    )
    if completed_result is not None:
        if show_progress:
            print(f"{task_name}: completed task verified and skipped", flush=True)
        return completed_result, {"skipped_completed": 1, "validation_records": []}
    latest_path = checkpoints / "latest.json"
    best_path = checkpoints / "best_ema.pt"
    train_rows = _row_indices(arrays.path_ids, split_path_ids["train"])
    selection_rows = _row_indices(arrays.path_ids, split_path_ids["selection"])
    if train_rows.size == 0 or selection_rows.size == 0:
        raise ValueError("task split contains no train or selection rows")

    _disable_mkldnn_for_cpu_if_needed(device)
    batch_rng = _set_task_seed(int(training_seed))
    model = DirectFluxUNet(dynamics, base_channels=int(args.base_channels)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scaler = _make_cuda_grad_scaler(
        enabled=bool(not args.no_amp and device.type == "cuda")
    )
    ema_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    history: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    completed = 0

    if latest_path.is_file():
        latest = _json_load(latest_path)
        if latest.get("schema") != LATEST_SCHEMA or int(latest.get("schema_version", -1)) != LATEST_SCHEMA_VERSION:
            raise ArtifactCompatibilityError("task latest pointer is incompatible")
        _checkpoint_fingerprints_match(latest.get("fingerprints", {}), fingerprints)
        filename = str(latest.get("filename", ""))
        checkpoint_path = checkpoints / filename
        if Path(filename).name != filename or not checkpoint_path.is_file():
            raise ArtifactCompatibilityError("task latest pointer is unsafe or missing")
        if file_fingerprint(checkpoint_path) != str(latest.get("checkpoint_sha256", "")):
            raise ArtifactCompatibilityError("task latest checkpoint hash mismatch")
        payload = _load_task_checkpoint(
            checkpoint_path, map_location=device, fingerprints=fingerprints
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        scaler.load_state_dict(payload["scaler_state_dict"])
        ema_state = {
            name: value.detach().clone()
            for name, value in payload["ema_state_dict"].items()
        }
        history = [dict(row) for row in payload["history"]]
        validation_records = [dict(row) for row in payload["validation_records"]]
        best = (
            None
            if payload["best_validation"] is None
            else dict(payload["best_validation"])
        )
        completed = int(payload["step"])
        restore_rng_state(payload["rng_state"], batch_rng)
        if best is not None:
            authoritative = checkpoints / f"step-{int(best['step']):08d}.pt"
            if not authoritative.is_file():
                raise ArtifactCompatibilityError("selected task checkpoint is missing")
            atomic_copy_file(authoritative, best_path)

    target_matrix = arrays.targets.reshape(arrays.targets.shape[0], -1).numpy()
    tau_np = arrays.tau_fraction.numpy().astype(np.float64, copy=False)

    def validate(step: int) -> dict[str, Any]:
        raw_prediction = _predict_task(
            model,
            arrays,
            dynamics,
            device=device,
            batch_size=int(args.validation_batch_size),
            row_indices=selection_rows,
        )
        raw = _selection_metrics(
            target_matrix[selection_rows],
            raw_prediction,
            tau_np[selection_rows],
            np.arange(selection_rows.size, dtype=np.int64),
        )
        with temporary_ema_weights(model, dict(ema_state)):
            ema_prediction = _predict_task(
                model,
                arrays,
                dynamics,
                device=device,
                batch_size=int(args.validation_batch_size),
                row_indices=selection_rows,
            )
        ema = _selection_metrics(
            target_matrix[selection_rows],
            ema_prediction,
            tau_np[selection_rows],
            np.arange(selection_rows.size, dtype=np.int64),
        )
        return {"step": int(step), "raw": raw, "ema": ema}

    def checkpoint(step: int, validation: Mapping[str, Any] | None) -> None:
        nonlocal best
        if validation is not None:
            candidate = {"step": int(step), **dict(validation["ema"])}
            best = _best_validation(best, candidate)
        path = checkpoints / f"step-{int(step):08d}.pt"
        _save_task_checkpoint(
            path,
            model=model,
            ema_state=ema_state,
            optimizer=optimizer,
            scaler=scaler,
            step=int(step),
            history=history,
            validation_records=validation_records,
            best_validation=best,
            fingerprints=fingerprints,
            numpy_rng=batch_rng,
        )
        atomic_write_json(
            latest_path,
            {
                "schema": LATEST_SCHEMA,
                "schema_version": LATEST_SCHEMA_VERSION,
                "filename": path.name,
                "step": int(step),
                "checkpoint_sha256": file_fingerprint(path),
                "fingerprints": dict(fingerprints),
            },
        )
        if best is not None and int(best["step"]) == int(step):
            atomic_copy_file(path, best_path)
        atomic_write_json(
            task_dir / "task_status.json",
            {
                "status": "running",
                "task_name": str(task_name),
                "training_seed": int(training_seed),
                "stride": int(stride),
                "training_step": int(step),
                "selected_step": None if best is None else int(best["step"]),
                "fingerprints": dict(fingerprints),
                "sampling_performed": 0,
            },
        )

    if not validation_records:
        initial = validate(0)
        validation_records.append(initial)
        checkpoint(0, initial)

    amp_context: Callable[[bool], Any] = (
        _cuda_autocast if device.type == "cuda" else lambda enabled: nullcontext()
    )
    last_report = time.perf_counter()
    for step in range(completed + 1, int(train_steps) + 1):
        selected = batch_rng.choice(
            train_rows,
            size=int(args.batch_size),
            replace=True,
        ).astype(np.int64)
        idx = torch.as_tensor(selected, dtype=torch.long)
        states = arrays.states.index_select(0, idx).to(device)
        tau = arrays.tau.index_select(0, idx).to(device)
        labels = arrays.labels.index_select(0, idx).to(device)
        target = arrays.targets.index_select(0, idx).to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(bool(not args.no_amp and device.type == "cuda")):
            prediction = project_edge_flux_torch(
                model(tau, states, labels, None),
                grid_size=int(dynamics.grid_size),
            )
            loss = (prediction - target).square().mean()
            zero = target.square().mean()
        scaler.scale(loss).backward()
        if float(args.grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        update_ema_state(ema_state, model, float(args.ema_decay))
        history.append(
            {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "zero_baseline_mse": float(zero.detach().cpu()),
                "training_gain": 1.0
                - float(loss.detach().cpu()) / max(float(zero.detach().cpu()), 1e-30),
            }
        )
        validation: dict[str, Any] | None = None
        if step % int(args.validation_every) == 0 or step == int(train_steps):
            validation = validate(step)
            validation_records.append(validation)
        if validation is not None or step % int(args.checkpoint_every) == 0 or step == int(train_steps):
            checkpoint(step, validation)
            if show_progress:
                elapsed = time.perf_counter() - last_report
                last_report = time.perf_counter()
                ema_mse = (
                    float(validation["ema"]["primary_mse"])
                    if validation is not None
                    else float("nan")
                )
                print(
                    f"{task_name} step {step}/{train_steps} loss={float(loss):.6g} "
                    f"ema_selection_mse={ema_mse:.6g} interval={elapsed:.1f}s",
                    flush=True,
                )

    if best is None or not best_path.is_file():
        raise RuntimeError(f"task {task_name} produced no finite selected checkpoint")
    selected_payload = _load_task_checkpoint(
        best_path, map_location=device, fingerprints=fingerprints
    )
    model.load_state_dict(selected_payload["model_state_dict"], strict=True)
    selected_ema = {
        name: value.detach().clone()
        for name, value in selected_payload["ema_state_dict"].items()
    }
    with temporary_ema_weights(model, selected_ema):
        selected_predictions = _predict_task(
            model,
            arrays,
            dynamics,
            device=device,
            batch_size=int(args.validation_batch_size),
        )
    result = compute_multiscale_split_metrics(
        target_matrix,
        selected_predictions,
        tau_np,
        arrays.path_ids,
        split_path_ids,
        stride=int(stride),
        training_seed=int(training_seed),
        selected_step=int(best["step"]),
        complete=True,
    )
    result["task_name"] = str(task_name)
    result["checkpoint_path"] = str(best_path.resolve())
    result["checkpoint_sha256"] = file_fingerprint(best_path)
    result["task_complete"] = 1
    result["fingerprints"] = dict(fingerprints)
    atomic_write_json(task_dir / "task_result.json", result)
    atomic_write_csv(
        task_dir / "checkpoint_metrics.csv",
        [
            {"step": int(record["step"]), "weights": weights, **dict(record[weights])}
            for record in validation_records
            for weights in ("raw", "ema")
        ],
    )
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "status": "complete",
            "task_name": str(task_name),
            "training_seed": int(training_seed),
            "stride": int(stride),
            "training_step": int(train_steps),
            "selected_step": int(best["step"]),
            "fingerprints": dict(fingerprints),
            "sampling_performed": 0,
        },
    )
    return result, {"best": best, "validation_records": validation_records}


def _make_d0_config(
    args: argparse.Namespace,
    *,
    cache_paths: int,
    anchors_per_path: int,
) -> Experiment12D0Config:
    """Create the isolated finite-block cache configuration.

    This does not call or relax the strict direct-Doob validator.  The
    multiscale cache has its own target contract and is never accepted by the
    elementary sampler.
    """

    return Experiment12D0Config(
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        base_channels=int(args.base_channels),
        cache_paths=int(cache_paths),
        cache_batch_size=int(cache_paths),
        cache_build_mode="substep",
        time_slices_per_path=int(anchors_per_path),
        teacher_stride_substeps=1,
        eta_l2_weight=0.0,
        d0_target_space="doob-physical-residual",
        physical_target_normalization="global-rms",
        physical_target_scale=0.0,
        physical_target_scale_floor=float(args.target_scale_floor),
        physical_loss_mask="all",
        invalid_output_l2_weight=0.0,
        curl_loss_weight=0.0,
        edge_laplacian_loss_weight=0.0,
        state_delta_loss_weight=0.0,
        rollout_loss_weight=0.0,
        trajectory_rollout_loss_weight=0.0,
        control_output_clip=0.0,
        sample_project_learned_mean=True,
        physical_sampler_noise_mode="reference",
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
        seed=int(args.cache_seed),
        use_amp=bool(not args.no_amp),
        ema_decay=float(args.ema_decay),
    )


def _initial_manifest(
    *,
    run_dir: Path,
    scientific: Mapping[str, Any],
    scientific_fingerprint: str,
    cache_semantic_fingerprint: str,
    source_hash: str,
    source_paths: Sequence[str],
    runtime: Mapping[str, Any],
    upstream: Mapping[str, Any],
    parent: Mapping[str, Any],
    parent_multiscale: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    study_profile = dict(scientific.get("study_profile", {}))
    return {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": _now(),
        "run_dir": str(run_dir.resolve()),
        "study_profile": study_profile,
        "scientific_config": dict(scientific),
        "scientific_fingerprint": str(scientific_fingerprint),
        "cache_semantic_fingerprint": str(cache_semantic_fingerprint),
        "source_fingerprint": str(source_hash),
        "source_paths": list(source_paths),
        "runtime": dict(runtime),
        "runtime_fingerprint": config_fingerprint(runtime),
        "upstream_zero_residual": dict(upstream),
        "parent_one_image": dict(parent),
        "parent_multiscale": (
            None if parent_multiscale is None else dict(parent_multiscale)
        ),
        "artifacts": {},
        "sampling_performed": 0,
    }


def _load_or_create_manifest(
    run_dir: Path,
    *,
    resumed: bool,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not resumed:
        atomic_write_json(path, candidate)
        return dict(candidate)
    if not path.is_file():
        raise ArtifactCompatibilityError("resume run has no run_manifest.json")
    manifest = _json_load(path)
    if manifest.get("schema") != RUN_SCHEMA or int(manifest.get("schema_version", -1)) != RUN_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("resume run manifest schema is incompatible")
    keys = (
        "scientific_fingerprint",
        "cache_semantic_fingerprint",
        "source_fingerprint",
        "runtime_fingerprint",
    )
    changed = [key for key in keys if manifest.get(key) != candidate.get(key)]
    if changed:
        raise ArtifactCompatibilityError(
            "resume run fingerprint mismatch: " + ", ".join(changed)
        )
    return manifest


def _cache_semantic_payload(scientific: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(scientific)
    return {
        "study_profile": value["study_profile"],
        "target_contract": value["target_contract"],
        "kernel": value["kernel"],
        "cache": value["cache"],
        "source_image": value["source_image"],
        "upstream_zero_residual_fingerprint": value[
            "upstream_zero_residual_fingerprint"
        ],
        "parent_one_image_fingerprint": value["parent_one_image_fingerprint"],
        "parent_multiscale_fingerprint": value["parent_multiscale_fingerprint"],
        "upstream_zero_residual_provenance": value[
            "upstream_zero_residual_provenance"
        ],
        "parent_one_image_provenance": value["parent_one_image_provenance"],
        "parent_multiscale_provenance": value["parent_multiscale_provenance"],
        "independence_provenance": value["independence_provenance"],
    }


def _artifact_fingerprints(manifest: Mapping[str, Any]) -> dict[str, str]:
    scientific = dict(manifest["scientific_config"])
    result = {
        "scientific": str(manifest["scientific_fingerprint"]),
        "cache_semantic": str(manifest["cache_semantic_fingerprint"]),
        "source": str(manifest["source_fingerprint"]),
        "runtime": str(manifest["runtime_fingerprint"]),
        "image": str(dict(scientific["source_image"])["image_sha256"]),
        "upstream": str(scientific["upstream_zero_residual_fingerprint"]),
        "parent": str(scientific["parent_one_image_fingerprint"]),
    }
    parent_multiscale = scientific.get("parent_multiscale_fingerprint")
    if parent_multiscale is not None:
        result["parent_multiscale"] = str(parent_multiscale)
    result["study_profile"] = config_fingerprint(scientific["study_profile"])
    return result


def _derive_shard_seed(base_seed: int, shard_id: int, *, scope: str) -> int:
    payload = f"d0-multiscale-shard:{scope}:{int(base_seed)}:{int(shard_id)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little") % (2**31 - 1)


def _merge_cache_diagnostics(shards: Sequence[D0MultiscaleCache]) -> dict[str, Any]:
    diagnostics = [dict(cache.diagnostics) for cache in shards]

    def total(name: str) -> float:
        values = [float(item.get(name, float("nan"))) for item in diagnostics]
        return float(sum(values)) if values and all(math.isfinite(value) for value in values) else float("nan")

    def integral_total(name: str) -> int | float:
        value = total(name)
        return int(value) if math.isfinite(value) else float("nan")

    proposed = total("proposed_edges")
    masked = total("masked_edges")
    mobility = total("mobility_weight_sum")
    limited_mobility = total("limited_mobility_weight_sum")
    noise = total("noise_energy_sum")
    limited_noise = total("limited_noise_energy_sum")
    if not all(math.isfinite(value) for value in (proposed, masked, mobility, limited_mobility, noise, limited_noise)):
        # Version-one shards built before additive counters were introduced are
        # not eligible for a run-level gate.  Keeping NaNs makes preflight fail
        # closed rather than silently averaging percentages.
        raw_fraction = mobility_fraction = noise_fraction = float("nan")
    else:
        raw_fraction = 0.0 if proposed <= 0.0 else masked / proposed
        mobility_fraction = float("nan") if mobility <= 0.0 else limited_mobility / mobility
        noise_fraction = float("nan") if noise <= 0.0 else limited_noise / noise
    return {
        "raw_limited_fraction": float(raw_fraction),
        "mobility_weighted_limited_fraction": float(mobility_fraction),
        "noise_energy_weighted_limited_fraction": float(noise_fraction),
        "masked_edges": integral_total("masked_edges"),
        "proposed_edges": integral_total("proposed_edges"),
        "mobility_weight_sum": mobility,
        "limited_mobility_weight_sum": limited_mobility,
        "noise_energy_sum": noise,
        "limited_noise_energy_sum": limited_noise,
        "floor_correction_l1": total("floor_correction_l1"),
        "renorm_correction_l1": total("renorm_correction_l1"),
        "floor_touched_pixels": integral_total("floor_touched_pixels"),
        "floor_proposed_pixels": integral_total("floor_proposed_pixels"),
        "nonfinite_edges": integral_total("nonfinite_edges"),
        "path_substep_count": integral_total("path_substep_count"),
        "builder_seeds": [int(item.get("builder_seed", -1)) for item in diagnostics],
        "prefix_aggregation": int(all(int(item.get("prefix_aggregation", 0)) == 1 for item in diagnostics)),
        "shard_count": len(shards),
    }


def merge_multiscale_cache_shards(
    shards: Sequence[D0MultiscaleCache],
    *,
    expected_path_ids: Sequence[int],
) -> D0MultiscaleCache:
    if not shards:
        raise ValueError("cannot merge an empty multiscale cache")
    for cache in shards:
        validate_multiscale_cache(cache)
    first = shards[0]
    for cache in shards[1:]:
        same = (
            torch.equal(cache.strides, first.strides)
            and np.array_equal(cache.tau_fraction_edges, first.tau_fraction_edges)
            and np.array_equal(cache.rate_schedule, first.rate_schedule)
            and cache.sample_steps == first.sample_steps
            and cache.reference_substeps == first.reference_substeps
            and cache.horizon == first.horizon
            and cache.dt_sub == first.dt_sub
            and cache.lambda_mix == first.lambda_mix
            and cache.anchor_plan_fingerprint == first.anchor_plan_fingerprint
            and cache.target_contract == first.target_contract
        )
        if not same:
            raise D0MultiscaleCompatibilityError("multiscale shards have incompatible scalar configuration")
    merged = replace(
        first,
        path_ids=torch.cat([cache.path_ids for cache in shards], dim=0),
        later_states=torch.cat([cache.later_states for cache in shards], dim=0),
        tau=torch.cat([cache.tau for cache in shards], dim=0),
        labels=torch.cat([cache.labels for cache in shards], dim=0),
        end_substeps=torch.cat([cache.end_substeps for cache in shards], dim=0),
        anchor_strata=torch.cat([cache.anchor_strata for cache in shards], dim=0),
        start_images=torch.cat([cache.start_images for cache in shards], dim=0),
        earlier_states=torch.cat([cache.earlier_states for cache in shards], dim=1),
        reverse_transfers=torch.cat([cache.reverse_transfers for cache in shards], dim=1),
        reference_transfers=torch.cat([cache.reference_transfers for cache in shards], dim=1),
        innovations=torch.cat([cache.innovations for cache in shards], dim=1),
        masks=torch.cat([cache.masks for cache in shards], dim=1),
        terminal_states=np.concatenate([cache.terminal_states for cache in shards], axis=0),
        source_indices=np.concatenate([cache.source_indices for cache in shards], axis=0),
        requested_labels=np.concatenate([cache.requested_labels for cache in shards], axis=0),
        diagnostics=_merge_cache_diagnostics(shards),
    )
    expected = np.asarray(expected_path_ids, dtype=np.int64)
    actual = merged.path_ids.numpy().astype(np.int64, copy=False)
    if np.unique(actual).size != actual.size or set(actual.tolist()) != set(expected.tolist()):
        raise D0MultiscaleCompatibilityError("merged cache path coverage differs from its index")
    axes = np.asarray([int(np.flatnonzero(actual == value)[0]) for value in expected], dtype=np.int64)
    if not np.array_equal(axes, np.arange(axes.size)):
        index = torch.as_tensor(axes, dtype=torch.long)
        merged = replace(
            merged,
            path_ids=merged.path_ids.index_select(0, index),
            later_states=merged.later_states.index_select(0, index),
            tau=merged.tau.index_select(0, index),
            labels=merged.labels.index_select(0, index),
            end_substeps=merged.end_substeps.index_select(0, index),
            anchor_strata=merged.anchor_strata.index_select(0, index),
            start_images=merged.start_images.index_select(0, index),
            earlier_states=merged.earlier_states.index_select(1, index),
            reverse_transfers=merged.reverse_transfers.index_select(1, index),
            reference_transfers=merged.reference_transfers.index_select(1, index),
            innovations=merged.innovations.index_select(1, index),
            masks=merged.masks.index_select(1, index),
            terminal_states=np.ascontiguousarray(merged.terminal_states[axes]),
            source_indices=np.ascontiguousarray(merged.source_indices[axes]),
            requested_labels=np.ascontiguousarray(merged.requested_labels[axes]),
        )
    validate_multiscale_cache(merged)
    return merged


def _split_mapping(split: Any) -> dict[str, list[int]]:
    return {
        "train": [int(value) for value in split.train_path_ids.tolist()],
        "selection": [int(value) for value in split.validation_path_ids.tolist()],
        "audit": [int(value) for value in split.confirmation_path_ids.tolist()],
    }


def make_physical_task(
    cache: D0MultiscaleCache,
    dynamics: DirectFluxMNISTConfig,
    *,
    stride: int,
    scale: float,
    device: torch.device,
    batch_size: int,
) -> TaskArrays:
    anchors = int(cache.anchors_per_path)
    targets = block_residual_targets(
        cache,
        dynamics,
        stride=int(stride),
        scale=float(scale),
        device=device,
        batch_size=int(batch_size),
    )
    return TaskArrays(
        states=cache.later_states.reshape(-1, cache.grid_size * cache.grid_size).contiguous(),
        tau=cache.tau.reshape(-1).contiguous(),
        tau_fraction=(cache.tau.reshape(-1) / float(cache.horizon)).contiguous(),
        labels=cache.labels.repeat_interleave(anchors).contiguous(),
        targets=targets.contiguous(),
        path_ids=cache.path_ids.numpy().astype(np.int64, copy=False).repeat(anchors),
    )


def _validate_cache_source(
    cache: D0MultiscaleCache,
    source_image: Mapping[str, Any],
) -> None:
    expected_index = int(source_image["dataset_index"])
    expected_label = int(source_image["label"])
    if not np.all(np.asarray(cache.source_indices, dtype=np.int64) == expected_index):
        raise ArtifactCompatibilityError("multiscale cache source image index differs")
    if not np.all(np.asarray(cache.requested_labels, dtype=np.int64) == expected_label):
        raise ArtifactCompatibilityError("multiscale cache requested label differs")
    expected = torch.as_tensor(
        np.asarray(source_image["mixed_target"], dtype=np.float32).reshape(-1)
    )
    maximum = float((cache.start_images - expected.view(1, -1)).abs().max())
    if not math.isfinite(maximum) or maximum > 2e-7:
        raise ArtifactCompatibilityError("multiscale cache mixed source image differs")


def _teacher_gate_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    split_rows = list(result.get("split_metrics", []))
    time_rows = list(result.get("time_bin_metrics", []))
    overall = next(
        row for row in split_rows
        if str(row.get("split")) == "audit" and int(row.get("bin_index", -2)) == -1
    )
    data_end = next(
        row for row in time_rows
        if str(row.get("split")) == "audit" and int(row.get("bin_index", -2)) == 4
    )
    return {
        "complete": int(result.get("task_complete", 0)),
        "selected_step": int(result.get("selected_step", -1)),
        "finite_fraction": float(overall.get("finite_fraction", float("nan"))),
        "audit_overall_gain": float(overall.get("prediction_gain", float("nan"))),
        "audit_data_end_gain": float(data_end.get("prediction_gain", float("nan"))),
        "audit_data_end_slice_count": int(data_end.get("slice_count", 0)),
    }


def _gate_thresholds(args: argparse.Namespace) -> MultiscaleGateThresholds:
    return MultiscaleGateThresholds(
        expected_training_seeds=len(args.training_seeds),
        min_passing_seeds=len(args.training_seeds) // 2 + 1,
        min_data_end_slices=int(args.audit_paths) * int(args.anchor_bin_counts[-1]),
        expected_audit_paths=int(args.audit_paths),
        expected_data_end_slices_per_path=int(args.anchor_bin_counts[-1]),
        bootstrap_confidence=float(args.bootstrap_confidence),
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_seed=int(args.bootstrap_seed),
        teacher_min_gain=float(args.teacher_min_gain),
        memorization_train_gain=0.50,
        elementary_stride=1,
    )


def _write_or_verify_json(path: Path, value: Mapping[str, Any], *, context: str) -> None:
    if path.is_file():
        existing = _json_load(path)
        if config_fingerprint(existing) != config_fingerprint(value):
            raise ArtifactCompatibilityError(f"{context} differs from the frozen run plan")
        return
    atomic_write_json(path, value)


def _cache_preflight_report(
    cache: D0MultiscaleCache,
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    *,
    train_path_ids: Sequence[int],
    expected_bin_counts: Sequence[int],
    require_explicit_slow_sum: bool,
) -> dict[str, Any]:
    diagnostic = dict(cache.diagnostics)
    try:
        correction_bound = (
            float(diagnostic.get("floor_correction_l1", 0.0))
            + float(diagnostic.get("renorm_correction_l1", 0.0))
        )
    except (TypeError, ValueError):
        correction_bound = float("nan")
    replay_tolerance = (
        1e-6 + max(correction_bound, 0.0)
        if math.isfinite(correction_bound)
        else 1e-6
    )
    report = evaluate_multiscale_cache_preflight(
        cache,
        dynamics,
        train_path_ids=train_path_ids,
        scale_floor=float(args.target_scale_floor),
        max_replay_l1=float(replay_tolerance),
        max_simplex_mass_error=float(args.max_simplex_mass_error),
        max_raw_intervention=float(args.max_raw_intervention),
        max_weighted_intervention=float(args.max_weighted_intervention),
        max_floor_correction_l1_per_path_substep=float(args.max_floor_correction_l1),
        max_renorm_correction_l1_per_path_substep=float(args.max_renorm_correction_l1),
        device=_device_from_arg(args.device),
    )
    checks = dict(report["checks"])
    counts = [
        np.bincount(row, minlength=5).astype(np.int64).tolist()
        for row in cache.anchor_strata.numpy()
    ]
    counts_pass = all(row == list(expected_bin_counts) for row in counts)
    checks["prescribed_anchor_strata"] = {
        "value": counts,
        "operator": "== each",
        "threshold": list(expected_bin_counts),
        "passed": bool(counts_pass),
    }
    crossings: dict[str, int] = {}
    for stride in cache.strides.tolist():
        starts = cache.end_substeps - int(stride)
        start_outer = torch.div(starts, int(cache.reference_substeps), rounding_mode="floor")
        end_outer = torch.div(
            cache.end_substeps - 1,
            int(cache.reference_substeps),
            rounding_mode="floor",
        )
        crossings[str(int(stride))] = int((start_outer != end_outer).count_nonzero())
    boundary_pass = any(value > 0 for key, value in crossings.items() if int(key) > 1)
    checks["outer_schedule_boundary_coverage"] = {
        "value": crossings,
        "operator": "any r>1 count >",
        "threshold": 0,
        "passed": bool(boundary_pass),
    }
    checks["common_endpoint_tensor"] = {
        "value": int(cache.later_states.shape[0] * cache.later_states.shape[1]),
        "operator": "single shared tensor for all strides",
        "threshold": int(cache.path_count * cache.anchors_per_path),
        "passed": cache.later_states.ndim == 3,
    }
    checks["prefix_aggregation"] = {
        "value": int(dict(cache.diagnostics).get("prefix_aggregation", 0)),
        "operator": "==",
        "threshold": 1,
        "passed": int(dict(cache.diagnostics).get("prefix_aggregation", 0)) == 1,
    }
    if require_explicit_slow_sum:
        diagnostic = dict(cache.diagnostics)
        slow_errors = {
            key: float(diagnostic.get(key, float("nan")))
            for key in (
                "slow_reverse_max_abs_error",
                "slow_reference_max_abs_error",
                "slow_innovation_max_abs_error",
                "slow_target_max_abs_error",
            )
        }
        slow_mask_mismatch = int(diagnostic.get("slow_mask_mismatch_count", -1))
        slow_replay = float(diagnostic.get("slow_replay_l1_max", float("nan")))
        slow_pass = (
            int(diagnostic.get("slow_sum_verified", 0)) == 1
            and all(math.isfinite(value) and value <= 2e-6 for value in slow_errors.values())
            and slow_mask_mismatch == 0
            and math.isfinite(slow_replay)
            and slow_replay <= float(replay_tolerance)
        )
        checks["explicit_slow_sum_equality"] = {
            "value": {
                **slow_errors,
                "replay_l1_max": slow_replay,
                "mask_mismatch_count": slow_mask_mismatch,
            },
            "operator": "max errors/replay <= and mask mismatches ==",
            "threshold": {
                "max_abs_error": 2e-6,
                "replay_l1_max": float(replay_tolerance),
                "mask_mismatch_count": 0,
            },
            "passed": bool(slow_pass),
        }
    report["checks"] = checks
    report["passed"] = int(all(bool(item.get("passed", False)) for item in checks.values()))
    report["cache_fingerprint"] = multiscale_cache_fingerprint(cache)
    report["target_contract"] = str(cache.target_contract)
    report["sampling_performed"] = 0
    return report


def _preflight_cache(
    *,
    run_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    show_progress: bool,
) -> tuple[D0MultiscaleCache, dict[str, Any]]:
    root = run_dir / "cache_preflight"
    root.mkdir(parents=True, exist_ok=True)
    total_substeps = int(args.sample_steps) * int(args.reference_substeps)
    plan = make_stratified_anchor_plan(
        num_paths=int(args.preflight_paths),
        anchors_per_path=5,
        total_substeps=total_substeps,
        max_stride=max(args.temporal_strides),
        seed=_derive_shard_seed(int(args.cache_seed), 0, scope="preflight-anchor"),
        bin_counts=(1, 1, 1, 1, 1),
    )
    _write_or_verify_json(
        root / "anchor_plan.json", plan.to_dict(), context="preflight anchor plan"
    )
    path = root / "shard-00000.npz"
    report_path = run_dir / "cache_preflight.json"
    seed = _derive_shard_seed(int(args.cache_seed), 0, scope="preflight-rollout")
    cache: D0MultiscaleCache | None = None
    record: D0MultiscaleShardRecord | None = None
    if path.is_file() and report_path.is_file():
        try:
            previous_report = _json_load(report_path)
            record = D0MultiscaleShardRecord.from_dict(
                dict(previous_report["cache_shard_record"])
            )
            candidate = load_multiscale_cache_shard(
                path,
                expected_record=record,
                verify_hashes=True,
            )
            valid = (
                record.shard_id == 0
                and candidate.path_ids.tolist() == plan.path_ids.tolist()
                and candidate.anchor_plan_fingerprint == plan.fingerprint
                and int(dict(candidate.diagnostics).get("builder_seed", -1)) == seed
                and previous_report.get("cache_fingerprint")
                == record.cache_fingerprint
                and previous_report.get("anchor_plan_fingerprint")
                == plan.fingerprint
                and dict(previous_report.get("fingerprints", {}))
                == _artifact_fingerprints(manifest)
            )
            if valid:
                cache = candidate
        except (KeyError, OSError, TypeError, ValueError, D0MultiscaleCompatibilityError):
            cache = None
            record = None
    if cache is None:
        local_d0 = _make_d0_config(
            args,
            cache_paths=int(args.preflight_paths),
            anchors_per_path=5,
        )
        cache = build_multiscale_cache_shard(
            dataset_images=images,
            dataset_labels=labels,
            dynamics_config=dynamics,
            d0_config=local_d0,
            anchor_plan=plan,
            strides=args.temporal_strides,
            device=_device_from_arg(args.device),
            seed=seed,
            global_anchor_plan_fingerprint=plan.fingerprint,
            verify_slow_sums=True,
            show_progress=show_progress,
        )
        record = save_multiscale_cache_shard(
            path,
            cache,
            shard_id=0,
            metadata={
                "scope": "cache-preflight",
                "scientific_fingerprint": manifest["cache_semantic_fingerprint"],
                "builder_seed": seed,
            },
        )
    if record is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("preflight cache shard has no integrity record")
    report = _cache_preflight_report(
        cache,
        dynamics,
        args,
        train_path_ids=cache.path_ids.tolist(),
        expected_bin_counts=(1, 1, 1, 1, 1),
        require_explicit_slow_sum=True,
    )
    report["fingerprints"] = _artifact_fingerprints(manifest)
    report["anchor_plan_fingerprint"] = plan.fingerprint
    report["cache_path"] = str(path.resolve())
    report["cache_shard_record"] = record.to_dict()
    atomic_write_json(report_path, report)
    return cache, report


def _verify_preflight_cache_for_report(
    run_dir: Path,
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> D0MultiscaleCache:
    """Load the frozen preflight shard without repairing or rewriting it."""

    try:
        record = D0MultiscaleShardRecord.from_dict(
            dict(report["cache_shard_record"])
        )
        filename = Path(record.filename)
        if (
            filename.name != record.filename
            or filename.is_absolute()
            or record.shard_id != 0
        ):
            raise ArtifactCompatibilityError(
                "cache preflight shard record has an unsafe filename or ID"
            )
        if dict(report.get("fingerprints", {})) != _artifact_fingerprints(manifest):
            raise ArtifactCompatibilityError("cache preflight fingerprints differ")
        if report.get("cache_fingerprint") != record.cache_fingerprint:
            raise ArtifactCompatibilityError(
                "cache preflight report and shard fingerprint differ"
            )
        if report.get("anchor_plan_fingerprint") != record.anchor_plan_fingerprint:
            raise ArtifactCompatibilityError(
                "cache preflight report and anchor plan differ"
            )
        return load_multiscale_cache_shard(
            run_dir / "cache_preflight" / record.filename,
            expected_record=record,
            verify_hashes=True,
        )
    except ArtifactCompatibilityError:
        raise
    except (KeyError, OSError, TypeError, ValueError, D0MultiscaleCompatibilityError) as exc:
        raise ArtifactCompatibilityError(
            f"cache preflight artifacts cannot be verified: {exc}"
        ) from exc


def _make_full_plan_and_split(
    run_dir: Path,
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, list[int]]]:
    total_substeps = int(args.sample_steps) * int(args.reference_substeps)
    plan = make_stratified_anchor_plan(
        num_paths=int(args.cache_paths),
        anchors_per_path=int(args.anchors_per_path),
        total_substeps=total_substeps,
        max_stride=max(args.temporal_strides),
        seed=int(args.cache_seed),
        bin_counts=args.anchor_bin_counts,
    )
    split = deterministic_three_way_path_split(
        plan.path_ids,
        seed=int(args.split_seed),
        train_paths=int(args.train_paths),
        validation_paths=int(args.selection_paths),
        confirmation_paths=int(args.audit_paths),
    )
    plan_payload = plan.to_dict()
    split_payload = {
        **split.to_dict(),
        "selection_path_ids": split.validation_path_ids.tolist(),
        "audit_path_ids": split.confirmation_path_ids.tolist(),
        "split_names": {
            "train": "train_path_ids",
            "selection": "validation_path_ids",
            "audit": "confirmation_path_ids",
        },
    }
    _write_or_verify_json(run_dir / "anchor_plan.json", plan_payload, context="anchor plan")
    _write_or_verify_json(run_dir / "path_split.json", split_payload, context="path split")
    return plan, split, _split_mapping(split)


def _record_for_existing_shard(
    path: Path,
    cache: D0MultiscaleCache,
    *,
    shard_id: int,
) -> D0MultiscaleShardRecord:
    return D0MultiscaleShardRecord(
        shard_id=int(shard_id),
        filename=path.name,
        path_ids=tuple(int(value) for value in cache.path_ids.tolist()),
        cache_fingerprint=multiscale_cache_fingerprint(cache),
        file_sha256=file_fingerprint(path),
        file_size=int(path.stat().st_size),
        anchor_plan_fingerprint=str(cache.anchor_plan_fingerprint),
    )


def _validate_cache_index_binding(
    index: D0MultiscaleCacheIndex,
    *,
    cache_semantic_fingerprint: str,
    expected_anchor_fingerprint: str,
    expected_path_ids: Sequence[int],
    expected_source_fingerprint: str,
    expected_runtime_fingerprint: str,
    expected_split_fingerprint: str,
    expected_shard_paths: int,
) -> None:
    if index.scientific_fingerprint != str(cache_semantic_fingerprint):
        raise ArtifactCompatibilityError("cache index scientific fingerprint differs")
    if index.anchor_plan_fingerprint != str(expected_anchor_fingerprint):
        raise ArtifactCompatibilityError("cache index anchor plan differs")
    if str(index.metadata.get("source_fingerprint", "")) != str(expected_source_fingerprint):
        raise ArtifactCompatibilityError("cache index source fingerprint differs")
    if str(index.metadata.get("runtime_fingerprint", "")) != str(expected_runtime_fingerprint):
        raise ArtifactCompatibilityError("cache index runtime fingerprint differs")
    if str(index.metadata.get("split_fingerprint", "")) != str(expected_split_fingerprint):
        raise ArtifactCompatibilityError("cache index path split differs")
    try:
        recorded_shard_paths = int(index.metadata.get("shard_paths", -1))
    except (TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError("cache index shard size is invalid") from exc
    if recorded_shard_paths != int(expected_shard_paths):
        raise ArtifactCompatibilityError("cache index shard size differs")
    if str(index.metadata.get("target_contract", "")) != TARGET_CONTRACT:
        raise ArtifactCompatibilityError("cache index target contract differs")
    expected = [int(value) for value in expected_path_ids]
    if list(index.expected_path_ids) != expected:
        raise ArtifactCompatibilityError("cache index path IDs differ")
    expected_records = [
        (
            shard_id,
            f"shard-{shard_id:05d}.npz",
            tuple(expected[start : start + int(expected_shard_paths)]),
        )
        for shard_id, start in enumerate(
            range(0, len(expected), int(expected_shard_paths))
        )
    ]
    actual_records = [
        (int(record.shard_id), str(record.filename), tuple(record.path_ids))
        for record in index.records
    ]
    if actual_records != expected_records:
        raise ArtifactCompatibilityError("cache index shard layout differs")


def _load_external_cache(
    root: Path,
    *,
    cache_semantic_fingerprint: str,
    expected_anchor_fingerprint: str,
    expected_path_ids: Sequence[int],
    expected_source_fingerprint: str,
    expected_runtime_fingerprint: str,
    expected_split_fingerprint: str,
    expected_shard_paths: int,
) -> tuple[D0MultiscaleCacheIndex, D0MultiscaleCache, Path]:
    candidates = (root / "cache" / "cache_index.json", root / "cache_index.json")
    index_path = next((path for path in candidates if path.is_file()), None)
    if index_path is None:
        raise FileNotFoundError(candidates[0])
    index = load_multiscale_cache_index(index_path, verify_shards=True)
    _validate_cache_index_binding(
        index,
        cache_semantic_fingerprint=cache_semantic_fingerprint,
        expected_anchor_fingerprint=expected_anchor_fingerprint,
        expected_path_ids=expected_path_ids,
        expected_source_fingerprint=expected_source_fingerprint,
        expected_runtime_fingerprint=expected_runtime_fingerprint,
        expected_split_fingerprint=expected_split_fingerprint,
        expected_shard_paths=expected_shard_paths,
    )
    shards = [
        load_multiscale_cache_shard(
            index_path.parent / record.filename,
            expected_record=record,
            verify_hashes=True,
        )
        for record in index.records
    ]
    return index, merge_multiscale_cache_shards(shards, expected_path_ids=expected_path_ids), index_path


def _prevalidate_external_cache_root(
    root: str | Path,
    *,
    study_profile: str,
    cache_semantic_fingerprint: str,
    parent_multiscale: Mapping[str, Any] | None,
) -> None:
    """Reject cross-profile cache reuse before a destination status is touched."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    if parent_multiscale is not None:
        parent_root = Path(str(parent_multiscale["run_dir"]))
        if root_path.resolve() == parent_root.resolve():
            raise ArtifactCompatibilityError(
                "confirmation cannot reuse the pilot multiscale cache"
            )
    manifest_path = root_path / "run_manifest.json"
    if not manifest_path.is_file():
        if str(study_profile) == "confirmation":
            raise ArtifactCompatibilityError(
                "confirmation external cache must come from a versioned confirmation run"
            )
        return
    manifest = _json_load(manifest_path)
    if manifest.get("schema") != RUN_SCHEMA or int(
        manifest.get("schema_version", -1)
    ) != RUN_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            "external cache run manifest schema is incompatible"
        )
    profile_value = manifest.get("study_profile")
    profile_name = (
        str(dict(profile_value).get("name", ""))
        if isinstance(profile_value, Mapping)
        else str(profile_value or "")
    )
    if profile_name != str(study_profile):
        raise ArtifactCompatibilityError("external cache study profile differs")
    if str(manifest.get("cache_semantic_fingerprint", "")) != str(
        cache_semantic_fingerprint
    ):
        raise ArtifactCompatibilityError(
            "external cache semantic fingerprint differs"
        )


def _full_cache(
    *,
    run_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    plan: Any,
    split: Any,
    show_progress: bool,
    read_only: bool = False,
) -> tuple[D0MultiscaleCacheIndex, D0MultiscaleCache, dict[str, Any], dict[int, float], Path]:
    cache_semantic = str(manifest["cache_semantic_fingerprint"])
    if args.cache_run_dir is not None:
        index, cache, index_path = _load_external_cache(
            Path(args.cache_run_dir),
            cache_semantic_fingerprint=cache_semantic,
            expected_anchor_fingerprint=plan.fingerprint,
            expected_path_ids=plan.path_ids,
            expected_source_fingerprint=str(manifest["source_fingerprint"]),
            expected_runtime_fingerprint=str(manifest["runtime_fingerprint"]),
            expected_split_fingerprint=str(split.fingerprint),
            expected_shard_paths=int(args.cache_shard_paths),
        )
    elif read_only:
        index, cache, index_path = _load_external_cache(
            run_dir,
            cache_semantic_fingerprint=cache_semantic,
            expected_anchor_fingerprint=plan.fingerprint,
            expected_path_ids=plan.path_ids,
            expected_source_fingerprint=str(manifest["source_fingerprint"]),
            expected_runtime_fingerprint=str(manifest["runtime_fingerprint"]),
            expected_split_fingerprint=str(split.fingerprint),
            expected_shard_paths=int(args.cache_shard_paths),
        )
    else:
        cache_dir = run_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = cache_dir / "cache_index.json"
        frozen_index: D0MultiscaleCacheIndex | None = None
        if index_path.is_file():
            try:
                frozen_index = load_multiscale_cache_index(
                    index_path,
                    verify_shards=False,
                )
            except (OSError, ValueError, D0MultiscaleCompatibilityError):
                frozen_index = None
            else:
                _validate_cache_index_binding(
                    frozen_index,
                    cache_semantic_fingerprint=cache_semantic,
                    expected_anchor_fingerprint=plan.fingerprint,
                    expected_path_ids=plan.path_ids,
                    expected_source_fingerprint=str(manifest["source_fingerprint"]),
                    expected_runtime_fingerprint=str(manifest["runtime_fingerprint"]),
                    expected_split_fingerprint=str(split.fingerprint),
                    expected_shard_paths=int(args.cache_shard_paths),
                )
        frozen_records = (
            {int(record.shard_id): record for record in frozen_index.records}
            if frozen_index is not None
            else {}
        )
        records: list[D0MultiscaleShardRecord] = []
        shard_size = int(args.cache_shard_paths)
        for shard_id, start in enumerate(range(0, int(args.cache_paths), shard_size)):
            shard_paths = plan.path_ids[start : start + shard_size]
            local_plan = slice_anchor_plan(plan, shard_paths)
            path = cache_dir / f"shard-{shard_id:05d}.npz"
            seed = _derive_shard_seed(int(args.cache_seed), shard_id, scope="production-rollout")
            cache: D0MultiscaleCache | None = None
            if path.is_file():
                try:
                    candidate = load_multiscale_cache_shard(
                        path,
                        expected_record=frozen_records.get(shard_id),
                        verify_hashes=True,
                    )
                    valid = (
                        candidate.path_ids.tolist() == shard_paths.tolist()
                        and candidate.anchor_plan_fingerprint == plan.fingerprint
                        and int(dict(candidate.diagnostics).get("builder_seed", -1)) == seed
                    )
                    if valid:
                        cache = candidate
                except (OSError, ValueError, D0MultiscaleCompatibilityError):
                    cache = None
            if cache is None:
                if show_progress:
                    print(
                        f"cache shard {shard_id + 1}/{math.ceil(int(args.cache_paths) / shard_size)} "
                        f"paths {int(start)}:{int(start + len(shard_paths))}",
                        flush=True,
                    )
                cache = build_multiscale_cache_shard(
                    dataset_images=images,
                    dataset_labels=labels,
                    dynamics_config=dynamics,
                    d0_config=_make_d0_config(
                        args,
                        cache_paths=len(shard_paths),
                        anchors_per_path=int(args.anchors_per_path),
                    ),
                    anchor_plan=local_plan,
                    strides=args.temporal_strides,
                    device=_device_from_arg(args.device),
                    seed=seed,
                    global_anchor_plan_fingerprint=plan.fingerprint,
                    show_progress=show_progress,
                )
                record = save_multiscale_cache_shard(
                    path,
                    cache,
                    shard_id=shard_id,
                    metadata={
                        "scientific_fingerprint": cache_semantic,
                        "builder_seed": seed,
                        "path_range": [int(start), int(start + len(shard_paths))],
                    },
                )
            else:
                record = _record_for_existing_shard(path, cache, shard_id=shard_id)
            records.append(record)
        candidate_index = make_multiscale_cache_index(
            records,
            expected_path_ids=plan.path_ids,
            scientific_fingerprint=cache_semantic,
            anchor_plan_fingerprint=plan.fingerprint,
            metadata={
                "target_contract": TARGET_CONTRACT,
                "split_fingerprint": split.fingerprint,
                "shard_paths": int(args.cache_shard_paths),
                "source_fingerprint": str(manifest["source_fingerprint"]),
                "runtime_fingerprint": str(manifest["runtime_fingerprint"]),
            },
        )
        if frozen_index is not None and frozen_index.fingerprint == candidate_index.fingerprint:
            index = frozen_index
        else:
            save_multiscale_cache_index(index_path, candidate_index)
            index = candidate_index
        shards = [
            load_multiscale_cache_shard(
                index_path.parent / record.filename,
                expected_record=record,
                verify_hashes=True,
            )
            for record in index.records
        ]
        cache = merge_multiscale_cache_shards(shards, expected_path_ids=plan.path_ids)

    cache_gate = _cache_preflight_report(
        cache,
        dynamics,
        args,
        train_path_ids=split.train_path_ids,
        expected_bin_counts=args.anchor_bin_counts,
        require_explicit_slow_sum=False,
    )
    cache_gate["fingerprints"] = _artifact_fingerprints(manifest)
    cache_gate["cache_index_fingerprint"] = index.fingerprint
    cache_gate["cache_fingerprint"] = multiscale_cache_fingerprint(cache)
    cache_gate["cache_index_path"] = str(index_path.resolve())
    scales = infer_training_block_scales(
        cache,
        dynamics,
        split.train_path_ids,
        floor=float(args.target_scale_floor),
        device=_device_from_arg(args.device),
        batch_size=int(args.validation_batch_size),
    )
    scale_payload = {
        "schema": "experiment12-d0-multiscale-target-scales",
        "schema_version": 1,
        "scales": {str(key): float(value) for key, value in scales.items()},
        "training_path_ids": split.train_path_ids.tolist(),
        "split_fingerprint": split.fingerprint,
        "cache_index_fingerprint": index.fingerprint,
        "training_only": 1,
    }
    atomic_write_json(run_dir / "target_scales.json", scale_payload)
    atomic_write_json(run_dir / "cache_gate.json", cache_gate)
    return index, cache, cache_gate, scales, index_path


def _task_fingerprints(
    manifest: Mapping[str, Any],
    *,
    index: D0MultiscaleCacheIndex,
    cache: D0MultiscaleCache,
    split_fingerprint: str,
    arrays: TaskArrays,
    task_kind: str,
    stride: int,
    training_seed: int,
    target_scale: float,
    train_steps: int,
) -> dict[str, Any]:
    return {
        **_artifact_fingerprints(manifest),
        "cache_index": str(index.fingerprint),
        "cache_content": multiscale_cache_fingerprint(cache),
        "split": str(split_fingerprint),
        "task_kind": str(task_kind),
        "stride": int(stride),
        "training_seed": int(training_seed),
        "target_scale": float(target_scale),
        "train_steps": int(train_steps),
        "states": array_fingerprint(arrays.states),
        "tau": array_fingerprint(arrays.tau),
        "tau_fraction": array_fingerprint(arrays.tau_fraction),
        "labels": array_fingerprint(arrays.labels),
        "targets": array_fingerprint(arrays.targets),
        "path_ids": array_fingerprint(np.asarray(arrays.path_ids, dtype=np.int64)),
    }


def _collect_checkpoint_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "tasks").glob("**/checkpoint_metrics.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"task_path": str(path.parent.relative_to(run_dir)), **row})
    return rows


def _atomic_save_figure(path: Path, figure: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, dpi=160, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_plots(
    run_dir: Path,
    *,
    stride_gates: Mapping[int, Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    written: list[str] = []
    strides = sorted(int(value) for value in stride_gates)
    if strides:
        overall = [
            float(dict(stride_gates[stride].get("diagnostics", {})).get("median_audit_overall_gain", float("nan")))
            for stride in strides
        ]
        data_end = [
            float(dict(stride_gates[stride].get("diagnostics", {})).get("median_audit_data_end_gain", float("nan")))
            for stride in strides
        ]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.plot(strides, overall, marker="o", label="audit overall")
        ax.plot(strides, data_end, marker="s", label="audit data end")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("temporal stride r")
        ax.set_ylabel("median prediction gain")
        ax.set_title("D0 block-residual gain versus stride")
        ax.grid(alpha=0.25)
        ax.legend()
        path = run_dir / "gain_vs_stride.png"
        _atomic_save_figure(path, fig)
        plt.close(fig)
        written.append(str(path))
    if strides:
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for row in checkpoint_rows:
            if str(row.get("weights")) != "ema":
                continue
            try:
                key = (str(row["task_path"]), str(row["weights"]))
                groups.setdefault(key, []).append(
                    (int(row["step"]), float(row["prediction_gain"]))
                )
            except (KeyError, TypeError, ValueError):
                continue
        for (label, _), points in groups.items():
            points.sort()
            ax.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                alpha=0.55,
                linewidth=1.0,
                label=label,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("training step")
        ax.set_ylabel("selection gain")
        ax.set_title("Multiscale EMA selection learning curves")
        ax.grid(alpha=0.25)
        if groups and len(groups) <= 16:
            ax.legend(fontsize=6, ncol=2)
        elif not groups:
            ax.text(
                0.5,
                0.5,
                "No completed checkpoint metrics available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        path = run_dir / "learning_curves.png"
        _atomic_save_figure(path, fig)
        plt.close(fig)
        written.append(str(path))
    return written


def _pilot_confirmation_comparison(
    run_dir: Path,
    *,
    stride_gates: Mapping[int, Mapping[str, Any]],
    parent_multiscale: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write an advisory, never-pooled comparison with the bound pilot."""

    if parent_multiscale is None or not stride_gates:
        return {}
    decision_path = Path(str(parent_multiscale.get("decision_path", "")))
    if not decision_path.is_file():
        raise ArtifactCompatibilityError(
            "bound parent multiscale decision artifact is unavailable"
        )
    pilot_report = _json_load(decision_path)
    pilot_gates = {
        int(stride): dict(gate)
        for stride, gate in dict(pilot_report.get("strides", {})).items()
    }
    current_gates = {
        int(stride): dict(gate) for stride, gate in stride_gates.items()
    }
    strides = sorted(set(pilot_gates) | set(current_gates))

    def scalar(gate: Mapping[str, Any], section: str, key: str) -> float:
        try:
            return float(dict(gate.get(section, {})).get(key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    def bootstrap_lcb(gate: Mapping[str, Any], key: str) -> float:
        try:
            return float(
                dict(dict(gate.get("bootstrap", {})).get(key, {})).get(
                    "lower_bound", float("nan")
                )
            )
        except (TypeError, ValueError):
            return float("nan")

    def passing_seeds(gate: Mapping[str, Any]) -> int:
        try:
            return int(
                dict(dict(gate.get("subchecks", {})).get("passing_seed_count", {})).get(
                    "value", 0
                )
            )
        except (TypeError, ValueError):
            return 0

    rows: list[dict[str, Any]] = []
    for stride in strides:
        row: dict[str, Any] = {
            "stride": int(stride),
            "evidence_pooled": 0,
            "gate_use": "advisory-only",
        }
        for prefix, gates in (("pilot", pilot_gates), ("confirmation", current_gates)):
            gate = gates.get(stride, {})
            row.update(
                {
                    f"{prefix}_present": int(bool(gate)),
                    f"{prefix}_passed": int(gate.get("passed", 0)) if gate else 0,
                    f"{prefix}_passing_seed_count": passing_seeds(gate),
                    f"{prefix}_median_audit_overall_gain": scalar(
                        gate, "diagnostics", "median_audit_overall_gain"
                    ),
                    f"{prefix}_median_audit_data_end_gain": scalar(
                        gate, "diagnostics", "median_audit_data_end_gain"
                    ),
                    f"{prefix}_median_audit_gain_vs_tau_baseline": scalar(
                        gate, "diagnostics", "median_audit_gain_vs_tau_baseline"
                    ),
                    f"{prefix}_median_audit_target_prediction_covariance": scalar(
                        gate,
                        "diagnostics",
                        "median_audit_target_prediction_covariance",
                    ),
                    f"{prefix}_overall_bootstrap_lcb": bootstrap_lcb(
                        gate, "overall_vs_zero"
                    ),
                    f"{prefix}_data_end_bootstrap_lcb": bootstrap_lcb(
                        gate, "data_end_vs_zero"
                    ),
                }
            )
        rows.append(row)

    csv_path = run_dir / "pilot_confirmation_stride_comparison.csv"
    atomic_write_csv(csv_path, rows)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.axhline(0.0, color="black", linewidth=0.8)
    for prefix, label, style in (
        ("pilot", "pilot", "--"),
        ("confirmation", "confirmation", "-"),
    ):
        axis.plot(
            strides,
            [float(row[f"{prefix}_median_audit_overall_gain"]) for row in rows],
            linestyle=style,
            marker="o",
            label=f"{label} overall",
        )
        axis.plot(
            strides,
            [float(row[f"{prefix}_median_audit_data_end_gain"]) for row in rows],
            linestyle=style,
            marker="s",
            label=f"{label} data end",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("temporal stride r")
    axis.set_ylabel("median prediction gain")
    axis.set_title("Independent confirmation versus pilot (not pooled)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    plot_path = run_dir / "pilot_confirmation_gain.png"
    _atomic_save_figure(plot_path, figure)
    plt.close(figure)
    return {
        "pilot_confirmation_comparison": str(csv_path.resolve()),
        "pilot_confirmation_plot": str(plot_path.resolve()),
    }


def _write_report_artifacts(
    run_dir: Path,
    *,
    seed_results: Sequence[Mapping[str, Any]],
    stride_gates: Mapping[int, Mapping[str, Any]],
    gate_report: Mapping[str, Any],
    task_failures: Sequence[Mapping[str, Any]],
    parent_multiscale: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    written = write_multiscale_gate_artifacts(run_dir, seed_results, gate_report)
    split_rows = [dict(row) for result in seed_results for row in result.get("split_metrics", [])]
    time_rows = [dict(row) for result in seed_results for row in result.get("time_bin_metrics", [])]
    path_rows = [dict(row) for result in seed_results for row in result.get("per_path_metrics", [])]
    stride_seed_rows = [
        {"stride": int(stride), **dict(row)}
        for stride, gate in sorted(stride_gates.items())
        for row in gate.get("seed_summaries", [])
    ]
    checkpoint_rows = _collect_checkpoint_rows(run_dir)
    atomic_write_csv(run_dir / "checkpoint_metrics.csv", checkpoint_rows)
    atomic_write_csv(run_dir / "validation_split_metrics.csv", split_rows)
    atomic_write_csv(run_dir / "validation_time_bins.csv", time_rows)
    atomic_write_csv(run_dir / "per_path_loss_deltas.csv", path_rows)
    atomic_write_csv(run_dir / "stride_seed_metrics.csv", stride_seed_rows)
    atomic_write_json(
        run_dir / "task_failures.json",
        {
            "failures": [dict(value) for value in task_failures],
            "failure_count": len(task_failures),
            "sampling_performed": 0,
        },
    )
    plots = _write_plots(
        run_dir,
        stride_gates=stride_gates,
        checkpoint_rows=checkpoint_rows,
    )
    comparison = _pilot_confirmation_comparison(
        run_dir,
        stride_gates=stride_gates,
        parent_multiscale=parent_multiscale,
    )
    return {
        **{key: str(path.resolve()) for key, path in written.items()},
        **comparison,
        "checkpoint_metrics": str((run_dir / "checkpoint_metrics.csv").resolve()),
        "validation_time_bins": str((run_dir / "validation_time_bins.csv").resolve()),
        "per_path": str((run_dir / "per_path_loss_deltas.csv").resolve()),
        "stride_seed": str((run_dir / "stride_seed_metrics.csv").resolve()),
        "plots": [str(Path(path).resolve()) for path in plots],
    }


def _empty_teacher_gate(reason: str) -> dict[str, Any]:
    return {
        "gate": "teacher",
        "passed": 0,
        "teacher_pass": 0,
        "subchecks": {},
        "reason": str(reason),
        "claim_scope": "teacher control was not completed",
    }


def _finish_run(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    gate_report: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    phase: str,
    skips: Sequence[Mapping[str, Any]],
) -> int:
    manifest = dict(manifest)
    manifest["artifacts"] = {**dict(manifest.get("artifacts", {})), **dict(artifacts)}
    manifest["sampling_performed"] = 0
    decision = dict(gate_report.get("decision", {}))
    manifest["decision_summary"] = {
        "decision": str(decision.get("decision", "optimization_pipeline_invalid")),
        "authoritative_decision": int(gate_report.get("authoritative_decision", 0)),
        "profile_conformant": int(gate_report.get("profile_conformant", 0)),
        "confirmation_exhausted": int(gate_report.get("confirmation_exhausted", 0)),
        "repeat_same_profile_authorized": int(
            gate_report.get("repeat_same_profile_authorized", 0)
        ),
        "sampling_authorized": 0,
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    required_pass = int(gate_report.get("required_gate_pass", 0))
    outcome = "complete" if required_pass else "gate_failed"
    _write_status(
        run_dir,
        status="complete",
        outcome=outcome,
        phase=str(phase),
        required_gate=str(gate_report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=decision.get("decision", "optimization_pipeline_invalid"),
        profile_conformant=int(gate_report.get("profile_conformant", 0)),
        authoritative_decision=int(gate_report.get("authoritative_decision", 0)),
        confirmation_exhausted=int(gate_report.get("confirmation_exhausted", 0)),
        repeat_same_profile_authorized=int(
            gate_report.get("repeat_same_profile_authorized", 0)
        ),
        sampling_authorized=0,
        skips=[dict(value) for value in skips],
        sampling_performed=0,
    )
    return 0 if required_pass else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    if hasattr(args, "_active_run_dir"):
        delattr(args, "_active_run_dir")
    try:
        return _run_impl(args)
    except Exception as exc:
        active = getattr(args, "_active_run_dir", None)
        if active is not None and Path(active).is_dir():
            try:
                _write_status(
                    Path(active),
                    status="failed",
                    outcome="implementation_error",
                    exception_type=type(exc).__name__,
                    error=str(exc),
                    sampling_performed=0,
                )
            except Exception:
                pass
        raise


def _run_impl(args: argparse.Namespace) -> int:
    device = _device_from_arg(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    exact_backend = configure_exact_torch_backend(device)
    dynamics = _make_dynamics(args)
    images, labels = _load_dataset(args)
    source_image = _select_source_image(
        images,
        labels,
        label=int(args.single_image_label),
        class_index=int(args.single_image_index),
        grid_size=int(args.grid_size),
        lambda_mix=float(args.lambda_mix),
    )
    upstream = verify_zero_residual_run(args.zero_residual_run_dir, args)
    parent = verify_parent_one_image_run(
        args.parent_one_image_run_dir,
        source_image=source_image,
        upstream_fingerprint=str(upstream["config_fingerprint"]),
    )
    parent_multiscale: dict[str, Any] | None = None
    if str(args.study_profile) == "confirmation":
        parent_multiscale = verify_parent_multiscale_run(
            args.parent_multiscale_run_dir,
            source_image=source_image,
            upstream_fingerprint=str(upstream["config_fingerprint"]),
            parent_one_image_fingerprint=str(parent["scientific_fingerprint"]),
        )
    scientific = _scientific_payload(
        args,
        dynamics,
        source_image,
        upstream,
        parent,
        parent_multiscale,
    )
    scientific_hash = config_fingerprint(scientific)
    cache_semantic_hash = config_fingerprint(_cache_semantic_payload(scientific))
    if args.cache_run_dir is not None:
        _prevalidate_external_cache_root(
            args.cache_run_dir,
            study_profile=str(args.study_profile),
            cache_semantic_fingerprint=cache_semantic_hash,
            parent_multiscale=parent_multiscale,
        )
    runtime = _runtime_record(device, exact_backend)
    source_hash, source_paths = _source_fingerprint()
    run_dir, resumed = _make_run_dir(args)
    # A rejected resume must not overwrite the existing run's status.  New
    # runs can record initialization errors immediately; resumed runs become
    # active only after their frozen manifest has been accepted.
    if not resumed:
        setattr(args, "_active_run_dir", run_dir)
    candidate_manifest = _initial_manifest(
        run_dir=run_dir,
        scientific=scientific,
        scientific_fingerprint=scientific_hash,
        cache_semantic_fingerprint=cache_semantic_hash,
        source_hash=source_hash,
        source_paths=source_paths,
        runtime=runtime,
        upstream=upstream,
        parent=parent,
        parent_multiscale=parent_multiscale,
    )
    manifest = _load_or_create_manifest(
        run_dir, resumed=resumed, candidate=candidate_manifest
    )
    setattr(args, "_active_run_dir", run_dir)
    print(f"run_dir={run_dir.resolve()}", flush=True)
    atomic_write_json(
        run_dir / "parent_provenance.json",
        {
            "upstream_zero_residual": upstream,
            "parent_one_image": parent,
            "parent_multiscale": parent_multiscale,
            "study_profile": dict(
                scientific.get(
                    "study_profile",
                    {
                        "name": str(args.study_profile),
                        "version": int(
                            STUDY_PROFILES[str(args.study_profile)]["version"]
                        ),
                    },
                )
            ),
            "independence_provenance": dict(
                scientific.get("independence_provenance", {})
            ),
            "source_image": dict(
                scientific.get(
                    "source_image",
                    {
                        key: source_image[key]
                        for key in (
                            "label",
                            "class_index",
                            "dataset_index",
                            "image_sha256",
                            "mixed_target_sha256",
                        )
                    },
                )
            ),
            "cache_semantic_fingerprint": cache_semantic_hash,
            "sampling_performed": 0,
        },
    )
    previous_status = (
        _json_load(run_dir / "run_status.json")
        if (run_dir / "run_status.json").is_file()
        else {}
    )
    _write_status(
        run_dir,
        status="running",
        phase="initialization",
        stage=str(args.stage),
        require_gate=str(args.require_gate),
        study_profile=str(args.study_profile),
        study_profile_version=int(
            STUDY_PROFILES[str(args.study_profile)]["version"]
        ),
        attempt_count=int(previous_status.get("attempt_count", 0)) + 1,
        scientific_fingerprint=scientific_hash,
        sampling_performed=0,
    )
    show_progress = not bool(args.no_progress)
    thresholds = _gate_thresholds(args)
    skips: list[dict[str, Any]] = []

    if str(args.stage) == "report":
        preflight_path = run_dir / "cache_preflight.json"
        if not preflight_path.is_file():
            raise ArtifactCompatibilityError(
                "report stage requires the completed cache_preflight.json"
            )
        preflight_report = _json_load(preflight_path)
        preflight_cache = _verify_preflight_cache_for_report(
            run_dir,
            preflight_report,
            manifest,
        )
        _validate_cache_source(preflight_cache, source_image)
    else:
        _write_status(run_dir, phase="cache-preflight")
        preflight_cache, preflight_report = _preflight_cache(
            run_dir=run_dir,
            images=images,
            labels=labels,
            dynamics=dynamics,
            args=args,
            manifest=manifest,
            show_progress=show_progress,
        )
        _validate_cache_source(preflight_cache, source_image)

    if str(args.stage) == "cache-preflight" or int(preflight_report.get("passed", 0)) != 1:
        if int(preflight_report.get("passed", 0)) != 1:
            skips.append(
                {
                    "stage": "cache",
                    "reason": "cache preflight failed; no production cache or optimization was attempted",
                }
            )
        gate_report = evaluate_multiscale_gates(
            cache_gate=preflight_report,
            teacher_gate=_empty_teacher_gate("not reached"),
            stride_gates={},
            require_gate=str(args.require_gate),
            thresholds=thresholds,
            study_profile=str(args.study_profile),
            profile_conformant=bool(args.profile_conformant),
        )
        report_artifacts = _write_report_artifacts(
            run_dir,
            seed_results=[],
            stride_gates={},
            gate_report=gate_report,
            task_failures=[],
            parent_multiscale=parent_multiscale,
        )
        return _finish_run(
            run_dir=run_dir,
            manifest=manifest,
            gate_report=gate_report,
            artifacts={
                **report_artifacts,
                "cache_preflight": str((run_dir / "cache_preflight.json").resolve()),
                "parent_provenance": str((run_dir / "parent_provenance.json").resolve()),
            },
            phase="cache-preflight",
            skips=skips,
        )

    plan, split, split_path_ids = _make_full_plan_and_split(run_dir, args)
    if str(args.stage) == "report" and args.cache_run_dir is None:
        if not (run_dir / "cache" / "cache_index.json").is_file():
            raise ArtifactCompatibilityError(
                "report stage requires an existing cache index or --cache-run-dir"
            )
    _write_status(run_dir, phase="cache")
    index, cache, cache_gate, target_scales, index_path = _full_cache(
        run_dir=run_dir,
        images=images,
        labels=labels,
        dynamics=dynamics,
        args=args,
        manifest=manifest,
        plan=plan,
        split=split,
        show_progress=show_progress and str(args.stage) != "report",
        read_only=str(args.stage) == "report",
    )
    _validate_cache_source(cache, source_image)
    if str(args.stage) == "cache" or int(cache_gate.get("passed", 0)) != 1:
        if int(cache_gate.get("passed", 0)) != 1:
            skips.append(
                {
                    "stage": "train",
                    "reason": "production multiscale cache gate failed",
                }
            )
        gate_report = evaluate_multiscale_gates(
            cache_gate=cache_gate,
            teacher_gate=_empty_teacher_gate("not reached"),
            stride_gates={},
            require_gate=str(args.require_gate),
            thresholds=thresholds,
            study_profile=str(args.study_profile),
            profile_conformant=bool(args.profile_conformant),
        )
        report_artifacts = _write_report_artifacts(
            run_dir,
            seed_results=[],
            stride_gates={},
            gate_report=gate_report,
            task_failures=[],
            parent_multiscale=parent_multiscale,
        )
        return _finish_run(
            run_dir=run_dir,
            manifest=manifest,
            gate_report=gate_report,
            artifacts={
                **report_artifacts,
                "cache_preflight": str((run_dir / "cache_preflight.json").resolve()),
                "cache_gate": str((run_dir / "cache_gate.json").resolve()),
                "cache_index": str(index_path.resolve()),
                "cache_index_fingerprint": index.fingerprint,
                "target_scales": str((run_dir / "target_scales.json").resolve()),
                "parent_provenance": str((run_dir / "parent_provenance.json").resolve()),
            },
            phase="cache",
            skips=skips,
        )

    anchors = int(cache.anchors_per_path)
    shared_states = cache.later_states.reshape(-1, cache.grid_size * cache.grid_size)
    shared_tau = cache.tau.reshape(-1)
    shared_tau_fraction = shared_tau / float(cache.horizon)
    shared_labels = cache.labels.repeat_interleave(anchors)
    shared_path_ids = cache.path_ids.numpy().astype(np.int64, copy=False).repeat(anchors)
    teacher_arrays, teacher_scale = make_teacher_task(
        shared_states,
        shared_tau,
        shared_tau_fraction,
        shared_labels,
        shared_path_ids,
        train_path_ids=split_path_ids["train"],
        grid_size=cache.grid_size,
        scale_floor=float(args.target_scale_floor),
    )
    teacher_dir = run_dir / "tasks" / "teacher"
    teacher_fingerprints = _task_fingerprints(
        manifest,
        index=index,
        cache=cache,
        split_fingerprint=split.fingerprint,
        arrays=teacher_arrays,
        task_kind="deterministic-teacher",
        stride=1,
        training_seed=int(args.teacher_seed),
        target_scale=float(teacher_scale),
        train_steps=int(args.teacher_steps),
    )
    _write_status(run_dir, phase="teacher-control")
    teacher_failure: dict[str, Any] | None = None
    teacher_result: dict[str, Any] | None = None
    try:
        if str(args.stage) == "report":
            teacher_result = load_completed_task_result(
                teacher_dir, fingerprints=teacher_fingerprints
            )
            if teacher_result is None:
                raise RuntimeError("report stage has no completed teacher task")
        else:
            teacher_result, _ = train_task(
                task_dir=teacher_dir,
                task_name="teacher-control",
                arrays=teacher_arrays,
                split_path_ids=split_path_ids,
                dynamics=dynamics,
                training_seed=int(args.teacher_seed),
                train_steps=int(args.teacher_steps),
                args=args,
                fingerprints=teacher_fingerprints,
                device=device,
                show_progress=show_progress,
                stride=1,
            )
    except ArtifactCompatibilityError:
        raise
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        teacher_failure = {
            "task": "teacher-control",
            "training_seed": int(args.teacher_seed),
            "exception_type": type(exc).__name__,
            "reason": str(exc),
        }
        teacher_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            teacher_dir / "task_status.json",
            {
                "status": "failed",
                **teacher_failure,
                "fingerprints": teacher_fingerprints,
                "sampling_performed": 0,
            },
        )
    if teacher_result is None:
        teacher_metrics = {
            "complete": 0,
            "selected_step": -1,
            "finite_fraction": float("nan"),
            "audit_overall_gain": float("nan"),
            "audit_data_end_gain": float("nan"),
            "audit_data_end_slice_count": 0,
        }
        teacher_gate = _empty_teacher_gate(
            "teacher task failed or is incomplete"
        )
    else:
        teacher_metrics = _teacher_gate_metrics(teacher_result)
        teacher_gate = evaluate_teacher_control(teacher_metrics, thresholds)
    atomic_write_json(
        run_dir / "teacher_control.json",
        {
            "target_scale": float(teacher_scale),
            "training_only_scale": 1,
            "metrics": teacher_metrics,
            "gate": teacher_gate,
            "task_result": teacher_result,
            "failure": teacher_failure,
            "sampling_performed": 0,
        },
    )

    task_failures: list[dict[str, Any]] = (
        [] if teacher_failure is None else [teacher_failure]
    )
    by_stride: dict[int, list[dict[str, Any]]] = {
        int(stride): [] for stride in args.temporal_strides
    }
    skip_physical = int(teacher_gate.get("passed", 0)) != 1 and str(args.require_gate) != "none"
    if skip_physical:
        skips.append(
            {
                "stage": "physical-tasks",
                "reason": "teacher control failed under a required gate",
            }
        )
    elif str(args.stage) == "report":
        for stride in args.temporal_strides:
            physical_arrays = make_physical_task(
                cache,
                dynamics,
                stride=int(stride),
                scale=float(target_scales[int(stride)]),
                device=device,
                batch_size=int(args.validation_batch_size),
            )
            for seed in args.training_seeds:
                expected = _task_fingerprints(
                    manifest,
                    index=index,
                    cache=cache,
                    split_fingerprint=split.fingerprint,
                    arrays=physical_arrays,
                    task_kind="physical-block-residual",
                    stride=int(stride),
                    training_seed=seed,
                    target_scale=float(target_scales[int(stride)]),
                    train_steps=int(args.train_steps),
                )
                task_dir = (
                    run_dir
                    / "tasks"
                    / f"stride-{int(stride):04d}"
                    / f"seed-{int(seed)}"
                )
                result = load_completed_task_result(
                    task_dir, fingerprints=expected
                )
                if result is None:
                    task_failures.append(
                        {
                            "stride": int(stride),
                            "training_seed": int(seed),
                            "reason": "missing or incomplete task result",
                        }
                    )
                else:
                    by_stride[int(stride)].append(result)
    else:
        _write_status(run_dir, phase="physical-tasks")
        for stride in args.temporal_strides:
            physical_arrays = make_physical_task(
                cache,
                dynamics,
                stride=int(stride),
                scale=float(target_scales[int(stride)]),
                device=device,
                batch_size=int(args.validation_batch_size),
            )
            for seed in args.training_seeds:
                task_dir = (
                    run_dir
                    / "tasks"
                    / f"stride-{int(stride):04d}"
                    / f"seed-{int(seed)}"
                )
                fingerprints = _task_fingerprints(
                    manifest,
                    index=index,
                    cache=cache,
                    split_fingerprint=split.fingerprint,
                    arrays=physical_arrays,
                    task_kind="physical-block-residual",
                    stride=int(stride),
                    training_seed=int(seed),
                    target_scale=float(target_scales[int(stride)]),
                    train_steps=int(args.train_steps),
                )
                try:
                    result, _ = train_task(
                        task_dir=task_dir,
                        task_name=f"stride-{int(stride)}-seed-{int(seed)}",
                        arrays=physical_arrays,
                        split_path_ids=split_path_ids,
                        dynamics=dynamics,
                        training_seed=int(seed),
                        train_steps=int(args.train_steps),
                        args=args,
                        fingerprints=fingerprints,
                        device=device,
                        show_progress=show_progress,
                        stride=int(stride),
                    )
                    by_stride[int(stride)].append(result)
                except ArtifactCompatibilityError:
                    raise
                except (RuntimeError, ValueError, FloatingPointError) as exc:
                    failure = {
                        "stride": int(stride),
                        "training_seed": int(seed),
                        "exception_type": type(exc).__name__,
                        "reason": str(exc),
                    }
                    task_failures.append(failure)
                    task_dir.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(
                        task_dir / "task_status.json",
                        {
                            "status": "failed",
                            **failure,
                            "fingerprints": fingerprints,
                            "sampling_performed": 0,
                        },
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    stride_gates = {
        int(stride): evaluate_stride_pass(
            int(stride), by_stride.get(int(stride), []), thresholds
        )
        for stride in args.temporal_strides
    }
    gate_report = evaluate_multiscale_gates(
        cache_gate=cache_gate,
        teacher_gate=teacher_gate,
        stride_gates=stride_gates,
        require_gate=str(args.require_gate),
        thresholds=thresholds,
        study_profile=str(args.study_profile),
        profile_conformant=bool(args.profile_conformant),
    )
    passing = sorted(
        int(stride)
        for stride, gate in stride_gates.items()
        if int(gate.get("passed", 0)) == 1
    )
    decision = dict(gate_report.get("decision", {}))
    decision["selected_stride"] = passing[0] if passing else None
    decision["passing_strides"] = passing
    decision["sampling_performed"] = 0
    gate_report["decision"] = decision
    gate_report["sampling_performed"] = 0
    seed_results = [
        result
        for stride in args.temporal_strides
        for result in by_stride.get(int(stride), [])
    ]
    report_artifacts = _write_report_artifacts(
        run_dir,
        seed_results=seed_results,
        stride_gates=stride_gates,
        gate_report=gate_report,
        task_failures=task_failures,
        parent_multiscale=parent_multiscale,
    )
    return _finish_run(
        run_dir=run_dir,
        manifest=manifest,
        gate_report=gate_report,
        artifacts={
            **report_artifacts,
            "cache_preflight": str((run_dir / "cache_preflight.json").resolve()),
            "cache_gate": str((run_dir / "cache_gate.json").resolve()),
            "cache_index": str(index_path.resolve()),
            "cache_index_fingerprint": index.fingerprint,
            "target_scales": str((run_dir / "target_scales.json").resolve()),
            "teacher_control": str((run_dir / "teacher_control.json").resolve()),
            "parent_provenance": str((run_dir / "parent_provenance.json").resolve()),
        },
        phase="report",
        skips=skips,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
