from __future__ import annotations

"""Positive-time Dirichlet-form implicit-score gate for Experiment 12 D0.

The command learns the relative score of the one-image forward marginal through
the symmetric Dirichlet form.  It is deliberately optimization-only: no reverse
sampler is imported or called, and every terminal artifact records that sampling
was not performed.
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
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    D0ThreeWayPathSplit,
    load_multiscale_cache_index,
    validate_three_way_path_split,
)
from mnist.d0_dirichlet_score import (
    D0DirichletScorePotentialUNet,
    D0LinearSplinePotential,
    carre_du_champ_from_gradients,
    dirichlet_score_objective,
    edge_difference_channels,
    edge_ratio_channels,
    exact_generator_from_derivatives,
    fit_linear_spline_baseline,
    harmonic_mobility_exact,
    physical_flux_from_edge_score,
    physical_flux_from_potential,
    rademacher_edge_probes,
    run_operator_preflight,
    sample_teacher_dirichlet,
    teacher_edge_score,
)
from mnist.d0_dirichlet_score_gate import (
    DirichletScoreGateThresholds,
    bootstrap_whole_path_delta,
    evaluate_control_bundle,
    evaluate_dirichlet_score_gates,
    evaluate_null_control,
    evaluate_positive_teacher_control,
)
from mnist.d0_score_state_cache import (
    D0ScoreStateCache,
    D0ScoreStateCacheIndex,
    build_fresh_score_state_shards,
    load_score_state_cache_index,
    load_score_state_cache_shards,
    materialize_parent_score_state_shards,
    merge_score_state_caches,
    validate_score_state_cache,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    _disable_mkldnn_for_cpu_if_needed,
    edge_alpha_value,
    init_ema_state,
    natural_horizon,
    temporary_ema_weights,
    update_ema_state,
)
from mnist.diag_d0_multiscale_learnability import (
    _load_dataset,
    _make_dynamics,
    _select_source_image,
    verify_parent_one_image_run,
    verify_zero_residual_run,
)
from mnist.experiment12_d0 import Experiment12D0Config


RUN_SCHEMA = "experiment12-d0-dirichlet-score-learnability"
RUN_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "experiment12-d0-dirichlet-score-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
TASK_RESULT_SCHEMA = "experiment12-d0-dirichlet-score-task-result"
TASK_RESULT_SCHEMA_VERSION = 1
TASK_STATUS_SCHEMA = "experiment12-d0-dirichlet-score-task-status"
TASK_STATUS_SCHEMA_VERSION = 1
AUDIT_INTENT_SCHEMA = "experiment12-d0-dirichlet-score-audit-intent"
AUDIT_INTENT_SCHEMA_VERSION = 1
PROBE_PLAN_SCHEMA = "experiment12-d0-dirichlet-score-probe-plan"
PROBE_PLAN_SCHEMA_VERSION = 1
WITNESS_PLAN_SCHEMA = "experiment12-d0-dirichlet-score-witness-plan"
WITNESS_PLAN_SCHEMA_VERSION = 1
CONTROL_SPLIT_SCHEMA = "experiment12-d0-dirichlet-score-control-split"
CONTROL_SPLIT_SCHEMA_VERSION = 1
STEIN_LINEAR_WITNESSES = 32
STEIN_QUADRATIC_WITNESSES = 32
CONTROL_SPLIT_COUNTS = (40, 12, 12)
CLAIM_SCOPE = (
    "positive-time relative-score learnability beyond a frozen smooth "
    "state-linear baseline for one fixed-grid one-image marginal"
)

EXPECTED_KERNEL: dict[str, Any] = {
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

REQUIRED_DEFAULTS: dict[str, Any] = {
    **EXPECTED_KERNEL,
    "single_image_label": 3,
    "single_image_index": 0,
    "parent_train_paths": 80,
    "parent_selection_paths": 24,
    "fresh_audit_paths": 24,
    "preflight_paths": 4,
    "anchors_per_path": 32,
    "anchor_bin_counts": (4, 4, 4, 4, 16),
    "minimum_forward_substep": 1024,
    "cache_shard_paths": 8,
    "dataset_seed": 260718,
    "fresh_cache_seed": 260751,
    "fresh_anchor_seed": 260752,
    "training_seeds": (260753, 260754, 260755),
    "training_probe_seed": 260756,
    "selection_probe_seed": 260757,
    "audit_probe_a_seed": 260758,
    "audit_probe_b_seed": 260759,
    "bootstrap_seed": 260760,
    "positive_teacher_data_seed": 260761,
    "positive_teacher_train_seed": 260762,
    "null_data_seed": 260763,
    "null_train_seed": 260764,
    "stein_a_seed": 260765,
    "stein_b_seed": 260766,
    "base_channels": 32,
    "batch_size": 32,
    "validation_batch_size": 32,
    "train_steps": 5000,
    "control_steps": 2000,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "validation_every": 500,
    "checkpoint_every": 500,
    "train_probes": 1,
    "selection_probes": 16,
    "audit_probes": 32,
    "bootstrap_reps": 10_000,
    "bootstrap_confidence": 0.90,
    "teacher_min_score_gain": 0.90,
    "teacher_min_flux_cosine": 0.98,
    "teacher_min_bin_flux_cosine": 0.95,
    "teacher_max_flux_relative_l2": 0.15,
    "teacher_max_bin_flux_relative_l2": 0.20,
    "min_cross_seed_flux_cosine": 0.50,
    "min_cross_seed_flux_cosine_lcb": 0.25,
    "max_raw_intervention": 0.005,
    "max_weighted_intervention": 0.0005,
    "max_floor_correction_l1": 1e-8,
    "max_renorm_correction_l1": 1e-6,
    "max_simplex_mass_error": 2e-6,
    "gpu_peak_memory_limit_gib": 7.5,
}


def _parse_csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        values = tuple(int(item) for item in value)
    if not values:
        raise ValueError("at least one integer is required")
    return values


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
            return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _device(value: str | None) -> torch.device:
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


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


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    status_path = run_dir / "run_status.json"
    current = _json_load(status_path) if status_path.is_file() else {}
    current.update(updates)
    current.setdefault("schema", RUN_SCHEMA)
    current.setdefault("schema_version", RUN_SCHEMA_VERSION)
    current.setdefault("sampling_performed", 0)
    current["updated_at"] = _now()
    atomic_write_json(status_path, current)
    return current


def _source_record() -> tuple[str, list[str]]:
    here = Path(__file__).resolve()
    names = (
        here,
        here.with_name("d0_dirichlet_score.py"),
        here.with_name("d0_score_state_cache.py"),
        here.with_name("d0_dirichlet_score_gate.py"),
        here.with_name("d0_one_image_gate.py"),
        here.with_name("d0_multiscale_cache.py"),
        here.with_name("eulerian_flux_mnist.py"),
        here.with_name("experiment12_d0.py"),
        here.with_name("diag_d0_multiscale_learnability.py"),
    )
    paths = [path for path in names if path.is_file()]
    return source_fingerprint(paths), [str(path) for path in paths]


def _runtime_record(device: torch.device, backend: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_type": device.type,
        "exact_backend": dict(backend),
    }
    if device.type == "cuda":
        value["device_name"] = torch.cuda.get_device_name(device)
        value["device_count"] = int(torch.cuda.device_count())
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "cache", "controls", "train", "report", "all"), default="all")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_d0_dirichlet_score"))
    parser.add_argument("--run-name", default="production-one-image-dirichlet-score")
    parser.add_argument("--zero-residual-run-dir", type=Path, required=True)
    parser.add_argument("--parent-one-image-run-dir", type=Path, required=True)
    parser.add_argument("--parent-multiscale-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--cache-run-dir", type=Path, default=None)
    parser.add_argument("--require-gate", choices=("none", "preflight", "cache", "controls", "score"), default="none")
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

    parser.add_argument("--parent-train-paths", type=int, default=80)
    parser.add_argument("--parent-selection-paths", type=int, default=24)
    parser.add_argument("--fresh-audit-paths", type=int, default=24)
    parser.add_argument("--preflight-paths", type=int, default=4)
    parser.add_argument("--anchors-per-path", type=int, default=32)
    parser.add_argument("--anchor-bin-counts", default="4,4,4,4,16")
    parser.add_argument("--minimum-forward-substep", type=int, default=1024)
    parser.add_argument("--cache-shard-paths", type=int, default=8)

    parser.add_argument("--dataset-seed", type=int, default=260718)
    parser.add_argument("--fresh-cache-seed", type=int, default=260751)
    parser.add_argument("--fresh-anchor-seed", type=int, default=260752)
    parser.add_argument("--training-seeds", default="260753,260754,260755")
    parser.add_argument("--training-probe-seed", type=int, default=260756)
    parser.add_argument("--selection-probe-seed", type=int, default=260757)
    parser.add_argument("--audit-probe-a-seed", type=int, default=260758)
    parser.add_argument("--audit-probe-b-seed", type=int, default=260759)
    parser.add_argument("--bootstrap-seed", type=int, default=260760)
    parser.add_argument("--positive-teacher-data-seed", type=int, default=260761)
    parser.add_argument("--positive-teacher-train-seed", type=int, default=260762)
    parser.add_argument("--null-data-seed", type=int, default=260763)
    parser.add_argument("--null-train-seed", type=int, default=260764)
    parser.add_argument("--stein-a-seed", type=int, default=260765)
    parser.add_argument("--stein-b-seed", type=int, default=260766)

    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--control-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--train-probes", type=int, default=1)
    parser.add_argument("--selection-probes", type=int, default=16)
    parser.add_argument("--audit-probes", type=int, default=32)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.90)

    parser.add_argument("--teacher-min-score-gain", type=float, default=0.90)
    parser.add_argument("--teacher-min-flux-cosine", type=float, default=0.98)
    parser.add_argument("--teacher-min-bin-flux-cosine", type=float, default=0.95)
    parser.add_argument("--teacher-max-flux-relative-l2", type=float, default=0.15)
    parser.add_argument("--teacher-max-bin-flux-relative-l2", type=float, default=0.20)
    parser.add_argument("--min-cross-seed-flux-cosine", type=float, default=0.50)
    parser.add_argument("--min-cross-seed-flux-cosine-lcb", type=float, default=0.25)
    parser.add_argument("--max-raw-intervention", type=float, default=0.005)
    parser.add_argument("--max-weighted-intervention", type=float, default=0.0005)
    parser.add_argument("--max-floor-correction-l1", type=float, default=1e-8)
    parser.add_argument("--max-renorm-correction-l1", type=float, default=1e-6)
    parser.add_argument("--max-simplex-mass-error", type=float, default=2e-6)
    parser.add_argument("--gpu-peak-memory-limit-gib", type=float, default=7.5)

    args = parser.parse_args(argv)
    try:
        args.anchor_bin_counts = _parse_csv_ints(args.anchor_bin_counts)
        args.training_seeds = _parse_csv_ints(args.training_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.anchor_bin_counts) != 5 or sum(args.anchor_bin_counts) != int(args.anchors_per_path):
        parser.error("anchor-bin-counts must contain five entries summing to anchors-per-path")
    if len(args.training_seeds) != 3 or len(set(args.training_seeds)) != 3:
        parser.error("training-seeds must contain exactly three distinct seeds")
    for name in (
        "parent_train_paths", "parent_selection_paths", "fresh_audit_paths",
        "preflight_paths", "anchors_per_path", "minimum_forward_substep",
        "cache_shard_paths", "batch_size", "validation_batch_size", "train_steps",
        "control_steps", "validation_every", "checkpoint_every", "train_probes",
        "selection_probes", "audit_probes", "bootstrap_reps",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.minimum_forward_substep) >= int(args.sample_steps) * int(args.reference_substeps):
        parser.error("minimum-forward-substep must be shorter than the trajectory")
    if args.require_gate != "none":
        mismatches = []
        for key, expected in REQUIRED_DEFAULTS.items():
            if not _semantic_close(getattr(args, key), expected):
                mismatches.append(f"{key}={getattr(args, key)!r}, expected {expected!r}")
        if mismatches:
            parser.error("required production gate rejects overrides: " + "; ".join(mismatches))
    return args


def verify_authoritative_confirmation(path: str | Path) -> dict[str, Any]:
    run_dir = Path(path)
    required = {
        "manifest": run_dir / "run_manifest.json",
        "status": run_dir / "run_status.json",
        "decision": run_dir / "learnability_decision.json",
        "cache_gate": run_dir / "cache_gate.json",
        "teacher": run_dir / "teacher_control.json",
        "failures": run_dir / "task_failures.json",
        "metrics": run_dir / "stride_seed_metrics.csv",
        "split": run_dir / "path_split.json",
        "cache_index": run_dir / "cache" / "cache_index.json",
    }
    for item in required.values():
        if not item.is_file():
            raise FileNotFoundError(item)
    manifest = _json_load(required["manifest"])
    status = _json_load(required["status"])
    report = _json_load(required["decision"])
    decision = dict(report.get("decision", {}))
    cache_gate = _json_load(required["cache_gate"])
    teacher = _json_load(required["teacher"])
    failures = _json_load(required["failures"])
    if manifest.get("schema") != "experiment12-d0-multiscale-learnability" or int(manifest.get("schema_version", -1)) != 2:
        raise ArtifactCompatibilityError("parent multiscale confirmation schema is incompatible")
    profile = dict(dict(manifest.get("scientific_config", {})).get("study_profile", {}))
    if profile.get("name") != "confirmation" or int(profile.get("profile_conformant", 0)) != 1:
        raise ArtifactCompatibilityError("parent run is not a conformant confirmation profile")
    required_status = {
        "status": "complete", "outcome": "gate_failed", "required_gate": "any-scale",
        "required_gate_pass": 0, "authoritative_decision": 1,
        "confirmation_exhausted": 1, "repeat_same_profile_authorized": 0,
        "sampling_performed": 0,
    }
    for key, expected in required_status.items():
        if status.get(key) != expected:
            raise ArtifactCompatibilityError(f"parent status mismatch for {key}")
    required_decision = {
        "decision": "no_confirmed_conditional_signal", "authoritative_decision": 1,
        "confirmation_exhausted": 1, "repeat_same_profile_authorized": 0,
        "sampling_authorized": 0, "sampling_performed": 0,
    }
    for key, expected in required_decision.items():
        if decision.get(key) != expected:
            raise ArtifactCompatibilityError(f"parent decision mismatch for {key}")
    if int(cache_gate.get("passed", 0)) != 1 or int(dict(teacher.get("gate", {})).get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("parent cache or teacher control did not pass")
    if int(failures.get("failure_count", -1)) != 0 or list(failures.get("failures", [])):
        raise ArtifactCompatibilityError("parent confirmation recorded task failures")
    with required["metrics"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 25 or any(
        int(row.get("complete", 0)) != 1
        or int(row.get("finite", 0)) != 1
        or int(row.get("selected_step", 0)) <= 0
        for row in rows
    ):
        raise ArtifactCompatibilityError("parent confirmation lacks 25 complete finite tasks")
    index = load_multiscale_cache_index(required["cache_index"], verify_shards=True)
    split = _json_load(required["split"])
    if (
        split.get("schema") != "experiment12-d0-three-way-path-split"
        or int(split.get("schema_version", -1)) != 1
    ):
        raise ArtifactCompatibilityError("parent split schema is incompatible")
    validation_ids = split.get("validation_path_ids", [])
    confirmation_ids = split.get("confirmation_path_ids", [])
    if split.get("selection_path_ids") != validation_ids or split.get("audit_path_ids") != confirmation_ids:
        raise ArtifactCompatibilityError("parent split role aliases are inconsistent")
    frozen_split = D0ThreeWayPathSplit(
        train_path_ids=np.asarray(split.get("train_path_ids", []), dtype=np.int64),
        validation_path_ids=np.asarray(validation_ids, dtype=np.int64),
        confirmation_path_ids=np.asarray(confirmation_ids, dtype=np.int64),
        seed=int(split.get("seed", -1)),
        fingerprint=str(split.get("fingerprint", "")),
    )
    try:
        validate_three_way_path_split(frozen_split, index.expected_path_ids)
    except (ValueError, TypeError) as exc:
        raise ArtifactCompatibilityError(f"parent split is invalid: {exc}") from exc
    if tuple(map(len, (frozen_split.train_path_ids, frozen_split.validation_path_ids, frozen_split.confirmation_path_ids))) != (80, 24, 24):
        raise ArtifactCompatibilityError("parent split is not the frozen 80/24/24 split")
    manifest_artifacts = dict(manifest.get("artifacts", {}))
    cache_fingerprints = dict(cache_gate.get("fingerprints", {}))
    bindings = {
        "manifest cache-index": manifest_artifacts.get("cache_index_fingerprint") == index.fingerprint,
        "cache-gate cache-index": cache_gate.get("cache_index_fingerprint") == index.fingerprint,
        "manifest cache-semantic": manifest.get("cache_semantic_fingerprint") == index.scientific_fingerprint,
        "cache-gate cache-semantic": cache_fingerprints.get("cache_semantic") == index.scientific_fingerprint,
        "cache-index split": dict(index.metadata).get("split_fingerprint") == frozen_split.fingerprint,
    }
    failed_bindings = [name for name, passed in bindings.items() if not passed]
    if failed_bindings:
        raise ArtifactCompatibilityError(
            "parent confirmation cache binding failed: " + ", ".join(failed_bindings)
        )
    return {
        "run_dir": str(run_dir.resolve()),
        "scientific_fingerprint": str(manifest.get("scientific_fingerprint", "")),
        "cache_semantic_fingerprint": str(manifest.get("cache_semantic_fingerprint", "")),
        "cache_index_fingerprint": str(index.fingerprint),
        "path_split_fingerprint": str(frozen_split.fingerprint),
        "source_image": dict(dict(manifest.get("scientific_config", {})).get("source_image", {})),
        "kernel": dict(dict(manifest.get("scientific_config", {})).get("kernel", {})),
        "manifest_sha256": file_fingerprint(required["manifest"]),
        "status_sha256": file_fingerprint(required["status"]),
        "decision_sha256": file_fingerprint(required["decision"]),
        "cache_gate_sha256": file_fingerprint(required["cache_gate"]),
        "teacher_sha256": file_fingerprint(required["teacher"]),
        "split_sha256": file_fingerprint(required["split"]),
        "cache_index_sha256": file_fingerprint(required["cache_index"]),
        "cache_index_path": str(required["cache_index"].resolve()),
        "path_split_path": str(required["split"].resolve()),
        "train_path_ids": [int(value) for value in split["train_path_ids"]],
        "selection_path_ids": [int(value) for value in split["selection_path_ids"]],
        "excluded_audit_path_ids": [int(value) for value in split["audit_path_ids"]],
    }


def _scientific_payload(args: argparse.Namespace, parent: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "sampling_performed": 0,
        "kernel": {
            key: getattr(args, key) for key in EXPECTED_KERNEL
        },
        "source_image": dict(source),
        "parent_confirmation_fingerprint": str(parent["scientific_fingerprint"]),
        "cache": {
            "parent_train_paths": int(args.parent_train_paths),
            "parent_selection_paths": int(args.parent_selection_paths),
            "fresh_audit_paths": int(args.fresh_audit_paths),
            "preflight_paths": int(args.preflight_paths),
            "anchors_per_path": int(args.anchors_per_path),
            "anchor_bin_counts": list(args.anchor_bin_counts),
            "minimum_forward_substep": int(args.minimum_forward_substep),
            "shard_paths": int(args.cache_shard_paths),
        },
        "training": {
            "seeds": list(args.training_seeds),
            "base_channels": int(args.base_channels),
            "batch_size": int(args.batch_size),
            "validation_batch_size": int(args.validation_batch_size),
            "steps": int(args.train_steps),
            "control_steps": int(args.control_steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "ema_decay": float(args.ema_decay),
            "validation_every": int(args.validation_every),
            "checkpoint_every": int(args.checkpoint_every),
            "train_probes": int(args.train_probes),
            "selection_probes": int(args.selection_probes),
            "audit_probes_per_bank": int(args.audit_probes),
            "probe_assignment": "canonical-state-order-int8-rademacher-v1",
            "amp": False,
        },
        "controls": {
            "independent_cluster_count": int(sum(CONTROL_SPLIT_COUNTS)),
            "split_counts": list(CONTROL_SPLIT_COUNTS),
            "physical_path_ids_reused": 0,
        },
        "stein_witnesses": {
            "banks": 2,
            "linear_per_bank": STEIN_LINEAR_WITNESSES,
            "quadratic_per_bank": STEIN_QUADRATIC_WITNESSES,
            "standardization": "training-states-only-population-sd",
            "aggregation": "square-within-path-time-bin-then-average",
        },
        "seeds": {
            key: getattr(args, key) for key in (
                "dataset_seed", "fresh_cache_seed", "fresh_anchor_seed",
                "training_probe_seed", "selection_probe_seed", "audit_probe_a_seed",
                "audit_probe_b_seed", "bootstrap_seed", "positive_teacher_data_seed",
                "positive_teacher_train_seed", "null_data_seed", "null_train_seed",
                "stein_a_seed", "stein_b_seed",
            )
        },
        "gate": {
            key: getattr(args, key) for key in (
                "bootstrap_reps", "bootstrap_confidence", "teacher_min_score_gain",
                "teacher_min_flux_cosine", "teacher_min_bin_flux_cosine",
                "teacher_max_flux_relative_l2", "teacher_max_bin_flux_relative_l2",
                "min_cross_seed_flux_cosine", "min_cross_seed_flux_cosine_lcb",
                "max_raw_intervention", "max_weighted_intervention",
                "max_floor_correction_l1", "max_renorm_correction_l1",
                "max_simplex_mass_error", "gpu_peak_memory_limit_gib",
            )
        },
    }


def _initial_manifest(
    *,
    run_dir: Path,
    scientific: Mapping[str, Any],
    scientific_fingerprint: str,
    source_hash: str,
    source_paths: Sequence[str],
    runtime: Mapping[str, Any],
    runtime_fingerprint: str,
    zero_residual: Mapping[str, Any],
    parent_one_image: Mapping[str, Any],
    parent_confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": _now(),
        "run_dir": str(run_dir.resolve()),
        "scientific_config": dict(scientific),
        "scientific_fingerprint": str(scientific_fingerprint),
        "source_fingerprint": str(source_hash),
        "source_paths": list(source_paths),
        "runtime": dict(runtime),
        "runtime_fingerprint": str(runtime_fingerprint),
        "upstream_zero_residual": dict(zero_residual),
        "parent_one_image": dict(parent_one_image),
        "parent_confirmation": dict(parent_confirmation),
        "sampling_performed": 0,
        "artifacts": {},
    }


def _load_or_create_manifest(
    run_dir: Path,
    *,
    resumed: bool,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if resumed:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _json_load(path)
        for key in (
            "schema", "schema_version", "scientific_fingerprint",
            "source_fingerprint", "runtime_fingerprint",
        ):
            if actual.get(key) != expected.get(key):
                raise ArtifactCompatibilityError(f"resume manifest mismatch for {key}")
        for key in ("upstream_zero_residual", "parent_one_image", "parent_confirmation"):
            if config_fingerprint(actual.get(key)) != config_fingerprint(expected.get(key)):
                raise ArtifactCompatibilityError(f"resume provenance mismatch for {key}")
        return actual
    atomic_write_json(path, expected)
    return dict(expected)


def _update_manifest_artifacts(run_dir: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    manifest = _json_load(path)
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts.update(dict(updates))
    manifest["artifacts"] = artifacts
    manifest["updated_at"] = _now()
    atomic_write_json(path, manifest)
    return manifest


def _validate_parent_bindings(
    args: argparse.Namespace,
    *,
    source: Mapping[str, Any],
    zero_residual: Mapping[str, Any],
    one_image: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> None:
    for parent_name, parent_source in (
        ("one-image", one_image.get("source_image", {})),
        ("confirmation", confirmation.get("source_image", {})),
    ):
        for key in ("label", "class_index", "dataset_index", "image_sha256", "mixed_target_sha256"):
            if dict(parent_source).get(key) != source.get(key):
                raise ArtifactCompatibilityError(f"{parent_name} source mismatch for {key}")
    kernel = dict(confirmation.get("kernel", {}))
    expected_kernel = {
        **EXPECTED_KERNEL,
        "edge_alpha_value": 1.0,
        "integrator": "masked_reference_free_step_torch",
    }
    for key, expected in expected_kernel.items():
        if not _semantic_close(kernel.get(key), expected):
            raise ArtifactCompatibilityError(f"confirmation kernel mismatch for {key}")
    if one_image.get("scientific_fingerprint") != dict(
        _json_load(Path(args.parent_multiscale_run_dir) / "run_manifest.json")
        .get("parent_one_image", {})
    ).get("scientific_fingerprint"):
        # Older manifests expose the same binding under the scientific config.
        scientific = dict(
            _json_load(Path(args.parent_multiscale_run_dir) / "run_manifest.json")
            .get("scientific_config", {})
        )
        if one_image.get("scientific_fingerprint") != scientific.get("parent_one_image_fingerprint"):
            raise ArtifactCompatibilityError("confirmation and one-image parent fingerprints differ")
    if zero_residual.get("config_fingerprint") != dict(
        _json_load(Path(args.parent_multiscale_run_dir) / "run_manifest.json")
        .get("scientific_config", {})
    ).get("upstream_zero_residual_fingerprint"):
        raise ArtifactCompatibilityError("confirmation and zero-residual fingerprints differ")


def _write_failure_artifact(run_dir: Path, exc: BaseException) -> None:
    atomic_write_json(
        run_dir / "failure.json",
        {
            "schema": RUN_SCHEMA + "-failure",
            "schema_version": 1,
            "type": type(exc).__name__,
            "message": str(exc),
            "sampling_performed": 0,
            "at": _now(),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        run_dir = getattr(args, "_active_run_dir", None)
        run_dir = None if run_dir is None else Path(run_dir)
        if run_dir is not None and run_dir.is_dir():
            _write_failure_artifact(run_dir, exc)
            _write_status(run_dir, status="failed", outcome="implementation_error", error=str(exc))
        raise


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    setattr(args, "_active_run_dir", run_dir)
    if not resumed:
        _write_status(
            run_dir, status="running", outcome="running", phase="initialization",
            stage=str(args.stage), require_gate=str(args.require_gate), required_gate=str(args.require_gate),
        )
    device = _device(args.device)
    backend = configure_exact_torch_backend(device)
    _disable_mkldnn_for_cpu_if_needed(device)
    runtime = _runtime_record(device, backend)
    runtime_fingerprint = config_fingerprint(runtime)
    source_hash, source_paths = _source_record()

    dynamics = _make_dynamics(args)
    images, labels = _load_dataset(args)
    source = _select_source_image(
        images,
        labels,
        label=int(args.single_image_label),
        class_index=int(args.single_image_index),
        grid_size=int(args.grid_size),
        lambda_mix=float(args.lambda_mix),
    )
    zero_residual = verify_zero_residual_run(args.zero_residual_run_dir, args)
    one_image = verify_parent_one_image_run(
        args.parent_one_image_run_dir,
        source_image=source,
        upstream_fingerprint=str(zero_residual["config_fingerprint"]),
    )
    confirmation = verify_authoritative_confirmation(args.parent_multiscale_run_dir)
    _validate_parent_bindings(
        args,
        source=source,
        zero_residual=zero_residual,
        one_image=one_image,
        confirmation=confirmation,
    )
    source_public = {key: value for key, value in source.items() if key not in {"image", "mixed_target"}}
    scientific = _scientific_payload(args, confirmation, source_public)
    scientific_fingerprint = config_fingerprint(scientific)
    expected_manifest = _initial_manifest(
        run_dir=run_dir,
        scientific=scientific,
        scientific_fingerprint=scientific_fingerprint,
        source_hash=source_hash,
        source_paths=source_paths,
        runtime=runtime,
        runtime_fingerprint=runtime_fingerprint,
        zero_residual=zero_residual,
        parent_one_image=one_image,
        parent_confirmation=confirmation,
    )
    _load_or_create_manifest(run_dir, resumed=resumed, expected=expected_manifest)
    setattr(args, "_active_run_dir", run_dir)
    previous = _json_load(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
    _write_status(
        run_dir, status="running", outcome="running", phase="initialization",
        stage=str(args.stage), require_gate=str(args.require_gate), required_gate=str(args.require_gate),
        attempt_count=int(previous.get("attempt_count", 0)) + 1,
        scientific_fingerprint=scientific_fingerprint,
    )
    print(f"run_dir={run_dir.resolve()}", flush=True)
    parent_provenance_value = {
        "zero_residual": zero_residual,
        "one_image": one_image,
        "confirmation": confirmation,
    }
    frozen_artifacts = {
        run_dir / "parent_provenance.json": parent_provenance_value,
        run_dir / "source_image.json": source_public,
    }
    for path, expected_value in frozen_artifacts.items():
        if resumed:
            if not path.is_file():
                raise FileNotFoundError(path)
            if _json_load(path) != expected_value:
                raise ArtifactCompatibilityError(f"resume frozen provenance mismatch for {path.name}")
        else:
            atomic_write_json(path, expected_value)

    return _run_stages(
        args,
        run_dir=run_dir,
        device=device,
        dynamics=dynamics,
        source=source,
        dataset_images=images,
        dataset_labels=labels,
        scientific_fingerprint=scientific_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_hash,
        confirmation=confirmation,
    )


@dataclass(frozen=True)
class ScoreArrays:
    """Flat, path-addressable view of one score-state cache role."""

    states: Tensor
    tau: Tensor
    tau_fraction: Tensor
    labels: Tensor
    path_ids: np.ndarray
    strata: np.ndarray
    end_substeps: np.ndarray
    rates: Tensor
    horizon: float
    role: str

    def __post_init__(self) -> None:
        rows = int(self.states.shape[0])
        if self.states.ndim != 2 or rows <= 0:
            raise ValueError("score arrays require non-empty (rows,pixels) states")
        if any(value.shape != (rows,) for value in (self.tau, self.tau_fraction, self.labels, self.rates)):
            raise ValueError("score array tensor rows are inconsistent")
        if any(np.asarray(value).shape != (rows,) for value in (self.path_ids, self.strata, self.end_substeps)):
            raise ValueError("score array path/anchor rows are inconsistent")
        if not bool(torch.isfinite(self.states).all() and torch.isfinite(self.tau).all() and torch.isfinite(self.rates).all()):
            raise ValueError("score arrays contain non-finite values")
        if not bool((self.states > 0).all()):
            raise ValueError("score arrays must be strictly positive")


class _CombinedPotential(nn.Module):
    def __init__(self, baseline: nn.Module, residual: nn.Module) -> None:
        super().__init__()
        self.baseline = baseline
        self.residual = residual
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)

    def forward(self, tau: Tensor | float, states: Tensor, labels: Tensor) -> Tensor:
        return self.baseline(tau, states, labels) + self.residual(tau, states, labels)


class _ZeroPotential(nn.Module):
    def forward(self, tau: Tensor | float, states: Tensor, labels: Tensor) -> Tensor:
        del tau, labels
        return (states * 0.0).sum(dim=1)


def _make_d0_config(args: argparse.Namespace, *, seed: int) -> Experiment12D0Config:
    return Experiment12D0Config(
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        base_channels=int(args.base_channels),
        cache_build_mode="exact-substep",
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
        seed=int(seed),
        use_amp=False,
        ema_decay=float(args.ema_decay),
    )


def _thresholds(args: argparse.Namespace) -> DirichletScoreGateThresholds:
    return DirichletScoreGateThresholds(
        expected_model_seeds=len(args.training_seeds),
        min_passing_model_seeds=2,
        expected_audit_paths=int(args.fresh_audit_paths),
        expected_data_end_states_per_path=int(args.anchor_bin_counts[-1]),
        expected_data_end_states=int(args.fresh_audit_paths) * int(args.anchor_bin_counts[-1]),
        bootstrap_confidence=float(args.bootstrap_confidence),
        bootstrap_reps=int(args.bootstrap_reps),
        bootstrap_seed=int(args.bootstrap_seed),
        nonlinear_flux_cosine=float(args.min_cross_seed_flux_cosine),
        nonlinear_flux_cosine_lcb=float(args.min_cross_seed_flux_cosine_lcb),
        teacher_min_score_gain=float(args.teacher_min_score_gain),
        teacher_min_overall_flux_cosine=float(args.teacher_min_flux_cosine),
        teacher_min_bin_flux_cosine=float(args.teacher_min_bin_flux_cosine),
        teacher_max_overall_relative_flux_l2=float(args.teacher_max_flux_relative_l2),
        teacher_max_bin_relative_flux_l2=float(args.teacher_max_bin_flux_relative_l2),
    )


def _flatten_cache(cache: D0ScoreStateCache) -> ScoreArrays:
    validate_score_state_cache(cache)
    paths, anchors, pixels = cache.states.shape
    ends = cache.end_substeps.numpy().astype(np.int64, copy=False)
    outer = np.maximum((ends - 1) // int(cache.reference_substeps), 0)
    rates = torch.as_tensor(np.asarray(cache.rate_schedule)[outer], dtype=torch.float32).reshape(-1)
    return ScoreArrays(
        states=cache.states.reshape(paths * anchors, pixels).float().contiguous(),
        tau=cache.tau.reshape(-1).float().contiguous(),
        tau_fraction=(cache.tau.reshape(-1).float() / float(cache.horizon)).contiguous(),
        labels=cache.labels.repeat_interleave(anchors).long().contiguous(),
        path_ids=np.repeat(cache.path_ids.numpy().astype(np.int64, copy=False), anchors),
        strata=cache.anchor_strata.numpy().astype(np.int64, copy=False).reshape(-1),
        end_substeps=ends.reshape(-1),
        rates=rates,
        horizon=float(cache.horizon),
        role=str(cache.role),
    )


def _load_role_arrays(index_path: Path, role: str) -> tuple[D0ScoreStateCacheIndex, D0ScoreStateCache, ScoreArrays]:
    index, shards = load_score_state_cache_shards(index_path, verify_hashes=True)
    selected = [cache for cache in shards if cache.role == str(role)]
    if not selected:
        raise ArtifactCompatibilityError(f"score cache index has no {role!r} shards")
    cache = merge_score_state_caches(selected)
    return index, cache, _flatten_cache(cache)


def _row_indices(arrays: ScoreArrays, paths: Sequence[int]) -> np.ndarray:
    return np.flatnonzero(
        np.isin(arrays.path_ids, np.asarray(paths, dtype=np.int64))
    ).astype(np.int64)


def _cache_health(
    cache: D0ScoreStateCache,
    args: argparse.Namespace,
    *,
    require_rollout_diagnostics: bool = True,
) -> dict[str, Any]:
    diag = dict(cache.diagnostics)
    path_substeps = max(float(diag.get("path_substep_count", 0.0)), 1.0)
    missing_value = float("inf") if require_rollout_diagnostics else 0.0
    values = {
        "state_finite_fraction": float(torch.isfinite(cache.states).double().mean()),
        "state_min": float(cache.states.min()),
        "max_simplex_mass_error": float((cache.states.double().sum(dim=-1) - 1.0).abs().max()),
        "raw_limited_fraction": float(diag.get("raw_limited_fraction", missing_value)),
        "mobility_weighted_limited_fraction": float(diag.get("mobility_weighted_limited_fraction", missing_value)),
        "noise_energy_weighted_limited_fraction": float(diag.get("noise_energy_weighted_limited_fraction", missing_value)),
        "floor_correction_l1_per_path_substep": float(diag.get("floor_correction_l1", missing_value)) / path_substeps,
        "renorm_correction_l1_per_path_substep": float(diag.get("renorm_correction_l1", missing_value)) / path_substeps,
        "floor_touched_pixels": float(diag.get("floor_touched_pixels", missing_value)),
        "nonfinite_edges": float(diag.get("nonfinite_edges", missing_value)),
    }
    checks = {
        "finite_states": values["state_finite_fraction"] == 1.0,
        "strictly_positive_states": values["state_min"] > 0.0,
        "simplex": values["max_simplex_mass_error"] <= float(args.max_simplex_mass_error),
        "raw_intervention": values["raw_limited_fraction"] <= float(args.max_raw_intervention),
        "mobility_intervention": values["mobility_weighted_limited_fraction"] <= float(args.max_weighted_intervention),
        "noise_intervention": values["noise_energy_weighted_limited_fraction"] <= float(args.max_weighted_intervention),
        "floor_correction": values["floor_correction_l1_per_path_substep"] <= float(args.max_floor_correction_l1),
        "renorm_correction": values["renorm_correction_l1_per_path_substep"] <= float(args.max_renorm_correction_l1),
        "floor_touches": values["floor_touched_pixels"] == 0.0,
        "nonfinite_edges": values["nonfinite_edges"] == 0.0,
    }
    return {"passed": int(all(checks.values())), "values": values, "checks": {key: int(value) for key, value in checks.items()}}


def _frozen_cache_kernel_matches(metadata: Mapping[str, Any]) -> bool:
    expected = {
        "grid_size": 28,
        "sample_steps": 512,
        "reference_substeps": 256,
        "tau_eff": 5e-5,
        "edge_alpha_mode": "alpha_eff",
        "edge_alpha_value": 1.0,
        "mass_floor": 1e-7,
        "limiter_fraction": 1.0,
        "lambda_mix": 0.35,
        "integrator": "masked_reference_free_step_torch",
    }
    return all(_semantic_close(dict(metadata).get(key), value) for key, value in expected.items())


def _schedule_core(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: dict(metadata).get(key)
        for key in (
            "sample_steps", "reference_substeps", "total_substeps", "horizon",
            "dt_sub", "tau_eff", "rate_schedule_sha256",
        )
    }


def _preflight_cache_identity(
    *,
    index: D0ScoreStateCacheIndex,
    cache: D0ScoreStateCache,
    expected_path_ids: Sequence[int],
    scientific_fingerprint: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = dict(index.metadata)
    checks = {
        "index_path_ids": tuple(index.expected_path_ids) == tuple(map(int, expected_path_ids)),
        "cache_path_ids": tuple(map(int, cache.path_ids.tolist())) == tuple(map(int, expected_path_ids)),
        "path_count": cache.path_count == int(args.preflight_paths),
        "role": cache.role == "preflight" and metadata.get("role") == "preflight",
        "origin": cache.origin == "fresh-reference" and metadata.get("origin") == "fresh-reference",
        "scientific_fingerprint": index.scientific_fingerprint == scientific_fingerprint == cache.scientific_fingerprint,
        "anchors_per_path": cache.anchors_per_path == int(args.anchors_per_path) == int(metadata.get("anchors_per_path", -1)),
        "anchor_bin_counts": list(metadata.get("anchor_bin_counts", [])) == list(args.anchor_bin_counts),
        "minimum_forward_substep": cache.minimum_forward_substep == int(args.minimum_forward_substep) == int(metadata.get("minimum_forward_substep", -1)),
        "anchor_plan": cache.anchor_plan_fingerprint == str(metadata.get("anchor_plan_fingerprint", "")),
        "frozen_kernel": _frozen_cache_kernel_matches(cache.kernel_metadata),
        "frozen_schedule": (
            int(cache.schedule_metadata.get("sample_steps", -1)) == int(args.sample_steps)
            and int(cache.schedule_metadata.get("reference_substeps", -1)) == int(args.reference_substeps)
            and _semantic_close(cache.schedule_metadata.get("tau_eff"), float(args.tau_eff))
        ),
        "parent_audit_not_reused": int(cache.provenance.get("parent_audit_reused", -1)) == 0,
    }
    return {
        "passed": int(all(checks.values())),
        "checks": {key: int(value) for key, value in checks.items()},
        "index_fingerprint": index.fingerprint,
        "anchor_plan_fingerprint": cache.anchor_plan_fingerprint,
    }


def _run_operator_and_device_preflight(
    *,
    run_dir: Path,
    device: torch.device,
    dynamics: DirectFluxMNISTConfig,
    arrays: ScoreArrays,
    args: argparse.Namespace,
) -> dict[str, Any]:
    operator = run_operator_preflight(dynamics, device=device, hutchinson_probes=4096)
    model = D0DirichletScorePotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    batch = min(int(args.batch_size), int(arrays.states.shape[0]))
    states = arrays.states[:batch].to(device)
    tau = arrays.tau[:batch].to(device)
    labels = arrays.labels[:batch].to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device.type).manual_seed(int(args.training_probe_seed))
    probes = rademacher_edge_probes(
        1, batch, int(dynamics.grid_size), device=device, dtype=states.dtype, generator=generator
    )
    model.zero_grad(set_to_none=True)
    objective = dirichlet_score_objective(
        model, tau, states, labels, dynamics, probes, create_graph=True
    )
    objective.loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    peak_gib = (
        float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
        if device.type == "cuda"
        else 0.0
    )
    device_report = {
        "device": str(device),
        "batch_size": batch,
        "production_batch_size": int(args.batch_size),
        "production_batch_exercised": int(batch == int(args.batch_size)),
        "objective": float(objective.loss.detach().cpu()),
        "objective_finite": int(bool(torch.isfinite(objective.loss))),
        "gradients_finite": int(gradients_finite),
        "peak_memory_gib": peak_gib,
        "peak_memory_limit_gib": float(args.gpu_peak_memory_limit_gib),
        "passed": int(
            bool(torch.isfinite(objective.loss))
            and gradients_finite
            and batch == int(args.batch_size)
            and peak_gib <= float(args.gpu_peak_memory_limit_gib)
        ),
    }
    report = {
        "schema": RUN_SCHEMA + "-operator-preflight",
        "schema_version": 1,
        "operator": operator,
        "device_objective": device_report,
        "passed": int(bool(operator.get("passed")) and bool(device_report["passed"])),
        "sampling_performed": 0,
    }
    atomic_write_json(run_dir / "operator_preflight.json", report)
    return report


def _score_split_value(
    confirmation: Mapping[str, Any], *, audit_ids: Sequence[int], preflight_ids: Sequence[int]
) -> dict[str, Any]:
    value = {
        "schema": RUN_SCHEMA + "-path-split",
        "schema_version": 1,
        "train_path_ids": list(map(int, confirmation["train_path_ids"])),
        "selection_path_ids": list(map(int, confirmation["selection_path_ids"])),
        "excluded_parent_audit_path_ids": list(map(int, confirmation["excluded_audit_path_ids"])),
        "fresh_audit_path_ids": list(map(int, audit_ids)),
        "fresh_preflight_path_ids": list(map(int, preflight_ids)),
        "whole_path_isolation": 1,
        "parent_audit_paths_reused": 0,
        "fingerprint": "",
    }
    value["fingerprint"] = config_fingerprint({key: item for key, item in value.items() if key != "fingerprint"})
    return value


def _make_split_artifact(
    run_dir: Path,
    confirmation: Mapping[str, Any],
    *,
    audit_ids: Sequence[int],
    preflight_ids: Sequence[int],
) -> dict[str, Any]:
    value = _score_split_value(
        confirmation, audit_ids=audit_ids, preflight_ids=preflight_ids
    )
    atomic_write_json(run_dir / "path_split.json", value)
    return value


def _verify_split_artifact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    actual = _json_load(path)
    if actual.get("schema") != RUN_SCHEMA + "-path-split" or int(actual.get("schema_version", -1)) != 1:
        raise ArtifactCompatibilityError("score path split schema is incompatible")
    declared = str(actual.get("fingerprint", ""))
    recomputed = config_fingerprint(
        {key: value for key, value in actual.items() if key != "fingerprint"}
    )
    if declared != recomputed:
        raise ArtifactCompatibilityError("score path split fingerprint is invalid")
    if actual != dict(expected):
        raise ArtifactCompatibilityError("score path split differs from the frozen plan")
    groups = [
        tuple(map(int, actual[key]))
        for key in (
            "train_path_ids", "selection_path_ids", "excluded_parent_audit_path_ids",
            "fresh_audit_path_ids", "fresh_preflight_path_ids",
        )
    ]
    if any(len(group) != len(set(group)) for group in groups):
        raise ArtifactCompatibilityError("score path split contains duplicate path IDs")
    parent = set(groups[0]) | set(groups[1]) | set(groups[2])
    if len(parent) != sum(len(group) for group in groups[:3]):
        raise ArtifactCompatibilityError("score parent roles overlap")
    if (set(groups[3]) & parent) or (set(groups[4]) & parent) or (set(groups[3]) & set(groups[4])):
        raise ArtifactCompatibilityError("score fresh and parent roles overlap")
    if int(actual.get("whole_path_isolation", 0)) != 1 or int(actual.get("parent_audit_paths_reused", -1)) != 0:
        raise ArtifactCompatibilityError("score path split does not certify audit isolation")
    return actual


def _probe_plan_value(
    args: argparse.Namespace, *, scientific_fingerprint: str
) -> dict[str, Any]:
    value = {
        "schema": PROBE_PLAN_SCHEMA,
        "schema_version": PROBE_PLAN_SCHEMA_VERSION,
        "scientific_fingerprint": str(scientific_fingerprint),
        "edge_order": "right-then-down-periodic",
        "assignment": "canonical-state-order-int8-rademacher-v1",
        "training": {
            "seed": int(args.training_probe_seed),
            "probes_per_batch_state": int(args.train_probes),
            "stream_state_checkpointed": 1,
        },
        "selection": {
            "seed": int(args.selection_probe_seed),
            "probes_per_state": int(args.selection_probes),
        },
        "audit_a": {
            "seed": int(args.audit_probe_a_seed),
            "probes_per_state": int(args.audit_probes),
        },
        "audit_b": {
            "seed": int(args.audit_probe_b_seed),
            "probes_per_state": int(args.audit_probes),
        },
        "fingerprint": "",
        "sampling_performed": 0,
    }
    value["fingerprint"] = config_fingerprint(
        {key: item for key, item in value.items() if key != "fingerprint"}
    )
    return value


def _load_or_create_probe_plan(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    scientific_fingerprint: str,
    read_only: bool,
) -> dict[str, Any]:
    path = run_dir / "probe_plan.json"
    expected = _probe_plan_value(args, scientific_fingerprint=scientific_fingerprint)
    if path.is_file():
        actual = _json_load(path)
        if actual != expected:
            raise ArtifactCompatibilityError("frozen Hutchinson probe plan mismatch")
        return actual
    if read_only:
        raise FileNotFoundError(path)
    atomic_write_json(path, expected)
    return expected


def _write_gate_report(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    cache: Mapping[str, Any],
    controls: Mapping[str, Any],
    seed_results: Sequence[Mapping[str, Any]],
    cosines: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = evaluate_dirichlet_score_gates(
        preflight_gate=preflight,
        cache_gate=cache,
        controls_gate=controls,
        seed_results=seed_results,
        nonlinear_flux_cosines=cosines,
        require_gate=str(args.require_gate),
        thresholds=_thresholds(args),
    )
    atomic_write_json(run_dir / "learnability_decision.json", report)
    atomic_write_json(run_dir / "implicit_score_gate.json", dict(report["score"]))
    atomic_write_json(run_dir / "score_learnability_decision.json", report)
    return report


def _empty_gate(name: str, reason: str) -> dict[str, Any]:
    return {"gate": str(name), "passed": 0, "subchecks": {}, "reason": str(reason), "sampling_performed": 0}


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


def _terminal_artifact_registry(run_dir: Path) -> dict[str, Any]:
    fixed = (
        "parent_provenance.json", "source_image.json", "path_split.json",
        "probe_plan.json", "stein_witness_plan.json", "stein_witness_plan.npz",
        "control_path_split.json", "operator_preflight.json", "preflight_gate.json",
        "cache_gate.json", "cache_provenance.json", "controls_gate.json",
        "task_failures.json", "checkpoint_metrics.csv", "audit_path_score_risks.csv",
        "stein_path_metrics.csv", "score_time_bins.csv", "cross_seed_flux_cosines.csv",
        "score_seed_metrics.csv", "learnability_decision.json", "implicit_score_gate.json",
        "score_learnability_decision.json", "run_status.json", "failure.json",
        "baselines/physical_linear_spline.pt", "baselines/physical_linear_spline.json",
        "cache/preflight/cache_index.json", "cache/parent/cache_index.json",
        "cache/audit/cache_index.json",
    )
    records: dict[str, Any] = {}
    for relative in fixed:
        path = run_dir / relative
        if path.is_file():
            records[relative] = _artifact_record(path)
    for pattern in (
        "tasks/seed-*/*.json", "tasks/seed-*/*.npz",
        "tasks/seed-*/checkpoint_metrics.csv",
        "tasks/seed-*/checkpoints/latest.json", "tasks/seed-*/checkpoints/step-*.pt",
        "tasks/seed-*/checkpoints/best.json", "tasks/seed-*/checkpoints/best_ema.pt",
        "controls/**/*.json", "controls/**/*.pt", "controls/**/*.csv",
        "cache/**/anchor_plan.json",
        "*.png",
    ):
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file():
                records[path.relative_to(run_dir).as_posix()] = _artifact_record(path)
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "records": records,
    }


def _finish(
    run_dir: Path,
    report: Mapping[str, Any],
    *,
    phase: str,
    skips: Sequence[Mapping[str, Any]],
) -> int:
    decision = dict(report.get("decision", {}))
    required_pass = int(report.get("required_gate_pass", 0))
    _write_status(
        run_dir,
        status="complete",
        outcome="complete" if required_pass else "gate_failed",
        phase=str(phase),
        required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=str(decision.get("decision", "optimization_pipeline_invalid")),
        recommended_next_action=decision.get("recommended_next_action"),
        skips=[dict(value) for value in skips],
        sampling_authorized=0,
        sampling_performed=0,
    )
    registry = _terminal_artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    _update_manifest_artifacts(
        run_dir,
        {
            "operator_preflight": str((run_dir / "operator_preflight.json").resolve()) if (run_dir / "operator_preflight.json").is_file() else None,
            "cache_gate": str((run_dir / "cache_gate.json").resolve()) if (run_dir / "cache_gate.json").is_file() else None,
            "controls_gate": str((run_dir / "controls_gate.json").resolve()) if (run_dir / "controls_gate.json").is_file() else None,
            "learnability_decision": str((run_dir / "learnability_decision.json").resolve()),
            "implicit_score_gate": str((run_dir / "implicit_score_gate.json").resolve()),
            "score_learnability_decision": str((run_dir / "score_learnability_decision.json").resolve()),
            "artifact_registry": _artifact_record(run_dir / "artifact_registry.json"),
        },
    )
    return 0 if required_pass else 2


def _set_seed(seed: int) -> np.random.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return np.random.default_rng(int(seed) + 104729)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _load_or_fit_baseline(
    *,
    run_dir: Path,
    train: ScoreArrays,
    dynamics: DirectFluxMNISTConfig,
    scientific_fingerprint: str,
    cache_fingerprint: str,
    device: torch.device,
    name: str = "physical",
    read_only: bool = False,
) -> tuple[D0LinearSplinePotential, dict[str, Any]]:
    root = run_dir / "baselines"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_linear_spline.pt"
    fingerprints = {
        "scientific_fingerprint": str(scientific_fingerprint),
        "cache_fingerprint": str(cache_fingerprint),
        "role": str(train.role),
        "path_ids_sha256": hashlib.sha256(np.ascontiguousarray(train.path_ids).tobytes()).hexdigest(),
    }
    if path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            payload = torch.load(path, map_location="cpu")
        if (
            payload.get("schema") != RUN_SCHEMA + "-linear-baseline"
            or int(payload.get("schema_version", -1)) != 1
            or payload.get("fingerprints") != fingerprints
        ):
            raise ArtifactCompatibilityError(f"{name} linear baseline fingerprint mismatch")
        sidecar_path = root / f"{name}_linear_spline.json"
        if sidecar_path.is_file():
            sidecar = _json_load(sidecar_path)
            if sidecar.get("sha256") != file_fingerprint(path) or sidecar.get("fingerprints") != fingerprints:
                raise ArtifactCompatibilityError(f"{name} linear baseline sidecar mismatch")
        elif read_only:
            raise FileNotFoundError(sidecar_path)
        model = D0LinearSplinePotential(dynamics, payload["coefficients"].float())
        return model, dict(payload["fit"])
    if read_only:
        raise FileNotFoundError(path)
    fit = fit_linear_spline_baseline(
        train.states.to(device), train.tau.to(device), dynamics, tolerance=1e-10, max_iterations=2000
    )
    model = fit.model.to("cpu")
    record = {
        "iterations": int(fit.iterations),
        "relative_residual": float(fit.relative_residual),
        "converged": int(fit.converged),
    }
    if not fit.converged or not math.isfinite(fit.relative_residual):
        raise RuntimeError(f"{name} linear baseline CG did not converge")
    atomic_torch_save(
        path,
        {
            "schema": RUN_SCHEMA + "-linear-baseline",
            "schema_version": 1,
            "coefficients": model.coefficients.detach().cpu(),
            "fit": record,
            "fingerprints": fingerprints,
        },
    )
    atomic_write_json(root / f"{name}_linear_spline.json", {**record, "fingerprints": fingerprints, "path": str(path.resolve()), "sha256": file_fingerprint(path)})
    return model, record


def _risk_components(
    model: nn.Module,
    baseline: nn.Module,
    arrays: ScoreArrays,
    dynamics: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    batch_size: int,
    probes_per_state: int,
    probe_seed: int,
    rows: Sequence[int] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    selected = (
        np.arange(int(arrays.states.shape[0]), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64).reshape(-1)
    )
    if selected.size == 0:
        raise ValueError("risk evaluation requires at least one state")
    zero = _ZeroPotential().to(device)
    model_was_training = bool(model.training)
    baseline_was_training = bool(baseline.training)
    model.eval()
    baseline.eval()
    output = {key: [] for key in ("model", "linear", "zero", "energy", "trace", "drift")}
    # Build the complete bank in canonical selected-row order on CPU.  Slicing
    # this int8 tensor makes probe assignment independent of evaluation batch
    # size and device scheduling while avoiding a full float32 bank in memory.
    probe_generator = torch.Generator(device="cpu").manual_seed(int(probe_seed))
    probe_bank_cpu = torch.randint(
        0, 2,
        (
            int(probes_per_state), int(selected.size), 2,
            int(dynamics.grid_size), int(dynamics.grid_size),
        ),
        generator=probe_generator,
        dtype=torch.int8,
        device="cpu",
    )
    try:
        with torch.enable_grad():
            for start in range(0, selected.size, max(1, int(batch_size))):
                ids = torch.as_tensor(selected[start : start + max(1, int(batch_size))], dtype=torch.long)
                states = arrays.states.index_select(0, ids).to(device)
                tau = arrays.tau.index_select(0, ids).to(device)
                labels = arrays.labels.index_select(0, ids).to(device)
                stop = min(selected.size, start + max(1, int(batch_size)))
                probes = probe_bank_cpu[:, start:stop].to(device=device, dtype=states.dtype)
                probes.mul_(2.0).sub_(1.0)
                full = dirichlet_score_objective(
                    model, tau, states, labels, dynamics, probes, create_graph=False
                )
                linear = dirichlet_score_objective(
                    baseline, tau, states, labels, dynamics, probes, create_graph=False
                )
                null = dirichlet_score_objective(
                    zero, tau, states, labels, dynamics, probes, create_graph=False
                )
                for key, value in (
                    ("model", full.per_sample), ("linear", linear.per_sample),
                    ("zero", null.per_sample), ("energy", full.energy),
                    ("trace", full.trace), ("drift", full.drift),
                ):
                    output[key].append(value.detach().double().cpu().numpy())
    finally:
        model.train(model_was_training)
        baseline.train(baseline_was_training)
    return {key: np.concatenate(parts).astype(np.float64, copy=False) for key, parts in output.items()}


def _paired_risk_summary(components: Mapping[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, Any]:
    size = int(np.asarray(components["model"]).size)
    selected = np.ones(size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != (size,) or not selected.any():
        return {
            "state_count": 0, "finite_fraction": 0.0,
            "model_score_risk": None, "linear_score_risk": None, "zero_score_risk": None,
            "score_risk_delta_vs_linear": None, "score_risk_delta_vs_zero": None,
        }
    values = {key: np.asarray(value, dtype=np.float64)[selected] for key, value in components.items()}
    finite = np.logical_and.reduce([np.isfinite(value) for value in values.values()])
    if not finite.all():
        return {"state_count": int(selected.sum()), "finite_fraction": float(finite.mean()), "model_score_risk": None, "linear_score_risk": None, "zero_score_risk": None, "score_risk_delta_vs_linear": None, "score_risk_delta_vs_zero": None}
    model = float(values["model"].mean())
    linear = float(values["linear"].mean())
    zero = float(values["zero"].mean())
    return {
        "state_count": int(selected.sum()),
        "finite_fraction": 1.0,
        "model_score_risk": model,
        "linear_score_risk": linear,
        "zero_score_risk": zero,
        "score_risk_delta_vs_linear": linear - model,
        "score_risk_delta_vs_zero": zero - model,
        "energy": float(values["energy"].mean()),
        "trace": float(values["trace"].mean()),
        "drift": float(values["drift"].mean()),
    }


def _selection_record(components: Mapping[str, np.ndarray], data_end_mask: np.ndarray) -> dict[str, Any]:
    return {
        "overall": _paired_risk_summary(components),
        "data_end": _paired_risk_summary(components, np.asarray(data_end_mask, dtype=bool)),
    }


def _task_checkpoint_fingerprints(
    *,
    manifest_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint_value: str,
    cache_fingerprint: str,
    baseline_path: Path,
    seed: int,
    train_steps: int,
    witness_plan_fingerprint: str | None = None,
) -> dict[str, Any]:
    value = {
        "scientific_fingerprint": str(manifest_fingerprint),
        "runtime_fingerprint": str(runtime_fingerprint),
        "source_fingerprint": str(source_fingerprint_value),
        "cache_fingerprint": str(cache_fingerprint),
        "baseline_sha256": file_fingerprint(baseline_path),
        "training_seed": int(seed),
        "train_steps": int(train_steps),
    }
    if witness_plan_fingerprint is not None:
        value["witness_plan_fingerprint"] = str(witness_plan_fingerprint)
    return value


def _save_score_checkpoint(
    path: Path,
    *,
    residual: nn.Module,
    ema_state: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    step: int,
    history: Sequence[Mapping[str, Any]],
    validations: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any] | None,
    fingerprints: Mapping[str, Any],
    batch_rng: np.random.Generator,
    training_probe_generator: torch.Generator,
) -> None:
    atomic_torch_save(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": int(step),
            "model_state_dict": copy.deepcopy(residual.state_dict()),
            "ema_state_dict": {key: value.detach().clone() for key, value in ema_state.items()},
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "history": [dict(value) for value in history],
            "validation_records": copy.deepcopy(list(validations)),
            "best_validation": None if best is None else dict(best),
            "fingerprints": dict(fingerprints),
            "rng_state": capture_rng_state(batch_rng),
            "training_probe_generator_state": training_probe_generator.get_state().cpu(),
            "amp": False,
        },
    )


def _load_score_checkpoint(path: Path, *, device: torch.device, fingerprints: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        value = torch.load(path, map_location=device)
    if value.get("schema") != CHECKPOINT_SCHEMA or int(value.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactCompatibilityError("legacy/foreign score checkpoint is report-only")
    if dict(value.get("fingerprints", {})) != dict(fingerprints):
        raise ArtifactCompatibilityError("score checkpoint fingerprint mismatch")
    required = {"step", "model_state_dict", "ema_state_dict", "optimizer_state_dict", "history", "validation_records", "best_validation", "rng_state", "training_probe_generator_state"}
    if not required.issubset(value):
        raise ArtifactCompatibilityError("score checkpoint is incomplete")
    return value


def _train_potential_task(
    *,
    task_dir: Path,
    train: ScoreArrays,
    selection: ScoreArrays,
    baseline: D0LinearSplinePotential,
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    args: argparse.Namespace,
    training_seed: int,
    train_steps: int,
    fingerprints: Mapping[str, Any],
    show_progress: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    task_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = task_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints / "latest.json"
    best_path = checkpoints / "best_ema.pt"
    best_pointer_path = checkpoints / "best.json"
    batch_rng = _set_seed(int(training_seed))
    training_probe_generator = torch.Generator(device=device).manual_seed(
        int(args.training_probe_seed)
    )
    residual = D0DirichletScorePotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    baseline_device = copy.deepcopy(baseline).to(device)
    combined = _CombinedPotential(baseline_device, residual).to(device)
    optimizer = torch.optim.AdamW(
        residual.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    ema_state = init_ema_state(residual)
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    completed = 0
    if latest_path.is_file():
        latest = _json_load(latest_path)
        filename = str(latest.get("filename", ""))
        path = checkpoints / filename
        if Path(filename).name != filename or not path.is_file() or file_fingerprint(path) != latest.get("sha256"):
            raise ArtifactCompatibilityError("score latest checkpoint pointer is invalid")
        payload = _load_score_checkpoint(path, device=device, fingerprints=fingerprints)
        residual.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        ema_state = {key: value.detach().clone().to(device) for key, value in payload["ema_state_dict"].items()}
        history = [dict(value) for value in payload["history"]]
        validations = [dict(value) for value in payload["validation_records"]]
        best = None if payload["best_validation"] is None else dict(payload["best_validation"])
        completed = int(payload["step"])
        restore_rng_state(payload["rng_state"], batch_rng)
        training_probe_generator.set_state(
            torch.as_tensor(payload["training_probe_generator_state"], dtype=torch.uint8, device="cpu")
        )
        if best is not None:
            authoritative = checkpoints / f"step-{int(best['step']):08d}.pt"
            if not authoritative.is_file():
                raise ArtifactCompatibilityError("selected score checkpoint is missing")
            if not best_path.is_file() or file_fingerprint(best_path) != file_fingerprint(authoritative):
                atomic_copy_file(authoritative, best_path)
            expected_best_pointer = {
                "schema": CHECKPOINT_SCHEMA + "-best",
                "schema_version": 1,
                "selected_step": int(best["step"]),
                "authoritative_filename": authoritative.name,
                "authoritative_sha256": file_fingerprint(authoritative),
                "best_ema_filename": best_path.name,
                "best_ema_sha256": file_fingerprint(best_path),
                "fingerprints": dict(fingerprints),
            }
            if not best_pointer_path.is_file() or _json_load(best_pointer_path) != expected_best_pointer:
                atomic_write_json(best_pointer_path, expected_best_pointer)

    selection_data_end = selection.strata == 4

    def validate(step: int) -> dict[str, Any]:
        raw_components = _risk_components(
            combined, baseline_device, selection, dynamics, device=device,
            batch_size=int(args.validation_batch_size), probes_per_state=int(args.selection_probes),
            probe_seed=int(args.selection_probe_seed),
        )
        raw = _selection_record(raw_components, selection_data_end)
        with temporary_ema_weights(residual, ema_state):
            ema_components = _risk_components(
                combined, baseline_device, selection, dynamics, device=device,
                batch_size=int(args.validation_batch_size), probes_per_state=int(args.selection_probes),
                probe_seed=int(args.selection_probe_seed),
            )
        ema = _selection_record(ema_components, selection_data_end)
        return {"step": int(step), "raw": raw, "ema": ema}

    def checkpoint(step: int, validation: Mapping[str, Any] | None) -> None:
        nonlocal best
        if validation is not None and int(step) > 0:
            overall = dict(validation["ema"]["overall"])
            data_end = dict(validation["ema"]["data_end"])
            candidate = {
                "step": int(step),
                "primary_risk": overall.get("model_score_risk"),
                "data_end_risk": data_end.get("model_score_risk"),
                "selection_metrics": copy.deepcopy(validation["ema"]),
            }
            if all(isinstance(candidate[key], (int, float)) and math.isfinite(float(candidate[key])) for key in ("primary_risk", "data_end_risk")):
                candidate["selection_eligible"] = int(
                    all(
                        isinstance(scope.get(key), (int, float))
                        and math.isfinite(float(scope[key]))
                        and float(scope[key]) > 0.0
                        for scope in (overall, data_end)
                        for key in ("score_risk_delta_vs_linear", "score_risk_delta_vs_zero")
                    )
                )
                choices = [candidate] + ([] if best is None else [best])
                best = min(
                    choices,
                    key=lambda value: (
                        -int(value.get("selection_eligible", 0)),
                        float(value["primary_risk"]),
                        float(value["data_end_risk"]),
                        int(value["step"]),
                    ),
                )
        path = checkpoints / f"step-{int(step):08d}.pt"
        _save_score_checkpoint(
            path, residual=residual, ema_state=ema_state, optimizer=optimizer, step=step,
            history=history, validations=validations, best=best, fingerprints=fingerprints,
            batch_rng=batch_rng, training_probe_generator=training_probe_generator,
        )
        if best is not None:
            authoritative = checkpoints / f"step-{int(best['step']):08d}.pt"
            if not authoritative.is_file():
                raise ArtifactCompatibilityError("selected score checkpoint is missing during publication")
            if not best_path.is_file() or file_fingerprint(best_path) != file_fingerprint(authoritative):
                atomic_copy_file(authoritative, best_path)
            atomic_write_json(
                best_pointer_path,
                {
                    "schema": CHECKPOINT_SCHEMA + "-best",
                    "schema_version": 1,
                    "selected_step": int(best["step"]),
                    "authoritative_filename": authoritative.name,
                    "authoritative_sha256": file_fingerprint(authoritative),
                    "best_ema_filename": best_path.name,
                    "best_ema_sha256": file_fingerprint(best_path),
                    "fingerprints": dict(fingerprints),
                },
            )
        atomic_write_json(
            latest_path,
            {"schema": CHECKPOINT_SCHEMA + "-latest", "schema_version": 1, "filename": path.name, "step": int(step), "sha256": file_fingerprint(path), "fingerprints": dict(fingerprints)},
        )
        atomic_write_json(
            task_dir / "task_status.json",
            {"schema": TASK_STATUS_SCHEMA, "schema_version": TASK_STATUS_SCHEMA_VERSION, "status": "running", "training_seed": int(training_seed), "training_step": int(step), "selected_step": None if best is None else int(best["step"]), "fingerprints": dict(fingerprints), "sampling_performed": 0},
        )

    if not validations:
        initial = validate(0)
        validations.append(initial)
        checkpoint(0, initial)
    started = time.perf_counter()
    for step in range(completed + 1, int(train_steps) + 1):
        choices = batch_rng.integers(0, int(train.states.shape[0]), size=int(args.batch_size), dtype=np.int64)
        ids = torch.as_tensor(choices, dtype=torch.long)
        states = train.states.index_select(0, ids).to(device)
        tau = train.tau.index_select(0, ids).to(device)
        labels = train.labels.index_select(0, ids).to(device)
        probes = rademacher_edge_probes(
            int(args.train_probes), int(ids.numel()), int(dynamics.grid_size),
            device=device, dtype=states.dtype, generator=training_probe_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        objective = dirichlet_score_objective(
            combined, tau, states, labels, dynamics, probes, create_graph=True
        )
        if not bool(torch.isfinite(objective.loss)):
            raise FloatingPointError(f"non-finite score loss at step {step}")
        objective.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(residual.parameters(), float(args.grad_clip))
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError(f"non-finite score gradient at step {step}")
        optimizer.step()
        update_ema_state(ema_state, residual, float(args.ema_decay))
        history.append({"step": int(step), "loss": float(objective.loss.detach().cpu()), "energy": float(objective.energy.mean().detach().cpu()), "trace": float(objective.trace.mean().detach().cpu()), "drift": float(objective.drift.mean().detach().cpu()), "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu())})
        validation = None
        if step % int(args.validation_every) == 0 or step == int(train_steps):
            validation = validate(step)
            validations.append(validation)
        if validation is not None or step % int(args.checkpoint_every) == 0 or step == int(train_steps):
            checkpoint(step, validation)
        if show_progress and (step % 50 == 0 or step == int(train_steps)):
            elapsed = time.perf_counter() - started
            done = max(1, step - completed)
            eta = elapsed / done * max(0, int(train_steps) - step)
            print(f"{task_dir.name}: step {step}/{train_steps} loss={history[-1]['loss']:.6g} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    if best is None or not best_path.is_file():
        raise RuntimeError("score task produced no finite nonzero EMA checkpoint")
    best_pointer = _json_load(best_pointer_path)
    if (
        best_pointer.get("schema") != CHECKPOINT_SCHEMA + "-best"
        or int(best_pointer.get("selected_step", -1)) != int(best["step"])
        or best_pointer.get("best_ema_sha256") != file_fingerprint(best_path)
        or dict(best_pointer.get("fingerprints", {})) != dict(fingerprints)
    ):
        raise ArtifactCompatibilityError("selected score checkpoint pointer is invalid")
    selected = _load_score_checkpoint(best_path, device=device, fingerprints=fingerprints)
    residual.load_state_dict(selected["ema_state_dict"], strict=True)
    combined.eval()
    checkpoint_rows: list[dict[str, Any]] = []
    for record in validations:
        for weights in ("raw", "ema"):
            for scope in ("overall", "data_end"):
                checkpoint_rows.append({"step": int(record["step"]), "weights": weights, "scope": scope, **dict(record[weights][scope])})
    atomic_write_csv(task_dir / "checkpoint_metrics.csv", checkpoint_rows)
    summary = {
        "complete": 1,
        "selected_step": int(best["step"]),
        "selection_eligible": int(best.get("selection_eligible", 0)),
        "selection_metrics": dict(best["selection_metrics"]),
        "checkpoint_path": str(best_path.resolve()),
        "checkpoint_sha256": file_fingerprint(best_path),
        "best_pointer_path": str(best_pointer_path.resolve()),
        "best_pointer_sha256": file_fingerprint(best_pointer_path),
        "fingerprints": dict(fingerprints),
        "training_history_rows": len(history),
    }
    return combined, summary


def _cell_gradients(
    model: nn.Module,
    arrays: ScoreArrays,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    was_training = bool(model.training)
    model.eval()
    pieces: list[np.ndarray] = []
    try:
        with torch.enable_grad():
            for start in range(0, int(arrays.states.shape[0]), max(1, int(batch_size))):
                stop = min(int(arrays.states.shape[0]), start + max(1, int(batch_size)))
                states = arrays.states[start:stop].to(device).detach().requires_grad_(True)
                values = model(arrays.tau[start:stop].to(device), states, arrays.labels[start:stop].to(device))
                gradient = torch.autograd.grad(values.sum(), states, create_graph=False)[0]
                pieces.append(gradient.detach().float().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(pieces, axis=0)


def _smooth_witness_bank(
    grid_size: int, *, seed: int, count: int = STEIN_LINEAR_WITNESSES
) -> np.ndarray:
    """Generate deterministic centered low-pass periodic Fourier directions."""

    rng = np.random.default_rng(int(seed))
    values = torch.as_tensor(rng.standard_normal((int(count), int(grid_size), int(grid_size))), dtype=torch.float64)
    for _ in range(4):
        values = (
            4.0 * values
            + torch.roll(values, 1, 1) + torch.roll(values, -1, 1)
            + torch.roll(values, 1, 2) + torch.roll(values, -1, 2)
        ) / 8.0
    values -= values.mean(dim=(1, 2), keepdim=True)
    values /= values.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-30)
    return values.reshape(int(count), -1).numpy()


def _witness_training_statistics(
    patterns: np.ndarray, train_states: Tensor
) -> dict[str, np.ndarray]:
    states = np.asarray(train_states.detach().cpu(), dtype=np.float64)
    directions = np.asarray(patterns, dtype=np.float64)
    linear = states @ directions.T
    quadratic = 0.5 * np.square(linear)
    result = {
        "linear_mean": linear.mean(axis=0),
        "linear_scale": linear.std(axis=0, ddof=0),
        "quadratic_mean": quadratic.mean(axis=0),
        "quadratic_scale": quadratic.std(axis=0, ddof=0),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("non-finite training-only Stein witness statistics")
    if np.any(result["linear_scale"] <= 1e-12) or np.any(result["quadratic_scale"] <= 1e-12):
        raise FloatingPointError("degenerate training-only Stein witness scale")
    return {key: np.ascontiguousarray(value, dtype=np.float64) for key, value in result.items()}


def _witness_plan_binding(
    train: ScoreArrays, *, scientific_fingerprint: str, cache_fingerprint: str
) -> dict[str, Any]:
    return {
        "scientific_fingerprint": str(scientific_fingerprint),
        "cache_fingerprint": str(cache_fingerprint),
        "train_role": str(train.role),
        "train_states_sha256": array_fingerprint(np.asarray(train.states, dtype=np.float32)),
        "train_path_ids_sha256": array_fingerprint(np.asarray(train.path_ids, dtype=np.int64)),
        "train_state_count": int(train.states.shape[0]),
    }


def _load_or_create_witness_plan(
    *,
    run_dir: Path,
    train: ScoreArrays,
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    scientific_fingerprint: str,
    cache_fingerprint: str,
    read_only: bool,
) -> dict[str, Any]:
    json_path = run_dir / "stein_witness_plan.json"
    npz_path = run_dir / "stein_witness_plan.npz"
    binding = _witness_plan_binding(
        train,
        scientific_fingerprint=scientific_fingerprint,
        cache_fingerprint=cache_fingerprint,
    )
    if json_path.is_file() or npz_path.is_file():
        if not json_path.is_file() or not npz_path.is_file():
            raise ArtifactCompatibilityError("Stein witness plan is incomplete")
        metadata = _json_load(json_path)
        if metadata.get("schema") != WITNESS_PLAN_SCHEMA or int(metadata.get("schema_version", -1)) != WITNESS_PLAN_SCHEMA_VERSION:
            raise ArtifactCompatibilityError("Stein witness plan schema is incompatible")
        if dict(metadata.get("binding", {})) != binding:
            raise ArtifactCompatibilityError("Stein witness plan training/cache binding mismatch")
        if metadata.get("npz_sha256") != file_fingerprint(npz_path):
            raise ArtifactCompatibilityError("Stein witness plan file hash mismatch")
        declared = str(metadata.get("fingerprint", ""))
        if declared != config_fingerprint({key: value for key, value in metadata.items() if key != "fingerprint"}):
            raise ArtifactCompatibilityError("Stein witness plan fingerprint mismatch")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        required = {
            f"{bank}_{field}"
            for bank in ("stein_a", "stein_b")
            for field in ("patterns", "linear_mean", "linear_scale", "quadratic_mean", "quadratic_scale")
        }
        if set(arrays) != required:
            raise ArtifactCompatibilityError("Stein witness plan arrays are incomplete")
        if dict(metadata.get("array_sha256", {})) != {
            key: array_fingerprint(value) for key, value in arrays.items()
        }:
            raise ArtifactCompatibilityError("Stein witness plan array fingerprint mismatch")
        for bank in ("stein_a", "stein_b"):
            if arrays[f"{bank}_patterns"].shape != (
                STEIN_LINEAR_WITNESSES, int(dynamics.grid_size) ** 2
            ):
                raise ArtifactCompatibilityError("Stein witness pattern shape mismatch")
            if any(
                arrays[f"{bank}_{field}"].shape != (STEIN_LINEAR_WITNESSES,)
                for field in ("linear_mean", "linear_scale", "quadratic_mean", "quadratic_scale")
            ):
                raise ArtifactCompatibilityError("Stein witness statistic shape mismatch")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ArtifactCompatibilityError("Stein witness plan contains non-finite values")
        return {"metadata": metadata, "arrays": arrays}
    if read_only:
        raise FileNotFoundError(json_path)

    arrays: dict[str, np.ndarray] = {}
    for bank, seed in (("stein_a", int(args.stein_a_seed)), ("stein_b", int(args.stein_b_seed))):
        patterns = np.ascontiguousarray(
            _smooth_witness_bank(
                int(dynamics.grid_size), seed=seed, count=STEIN_LINEAR_WITNESSES
            ),
            dtype=np.float64,
        )
        statistics = _witness_training_statistics(patterns, train.states)
        arrays[f"{bank}_patterns"] = patterns
        for key, value in statistics.items():
            arrays[f"{bank}_{key}"] = value
    _atomic_save_npz(npz_path, **arrays)
    metadata = {
        "schema": WITNESS_PLAN_SCHEMA,
        "schema_version": WITNESS_PLAN_SCHEMA_VERSION,
        "binding": binding,
        "banks": {
            "stein_a": {"seed": int(args.stein_a_seed)},
            "stein_b": {"seed": int(args.stein_b_seed)},
        },
        "construction": "centered-unit-rms-periodic-low-pass-fourier-directions-v1",
        "standardization": "training-states-only-population-sd",
        "aggregation": "square-within-path-and-time-bin-then-equal-average",
        "linear_witnesses_per_bank": STEIN_LINEAR_WITNESSES,
        "quadratic_witnesses_per_bank": STEIN_QUADRATIC_WITNESSES,
        "time_bin_count": 5,
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": file_fingerprint(npz_path),
        "array_sha256": {key: array_fingerprint(value) for key, value in arrays.items()},
        "sampling_performed": 0,
        "fingerprint": "",
    }
    metadata["fingerprint"] = config_fingerprint(
        {key: value for key, value in metadata.items() if key != "fingerprint"}
    )
    atomic_write_json(json_path, metadata)
    return {"metadata": metadata, "arrays": arrays}


def _binwise_stein_discrepancy(
    residual_matrix: np.ndarray,
    *,
    path_mask: np.ndarray,
    strata: np.ndarray,
) -> tuple[float, list[int]]:
    means: list[np.ndarray] = []
    counts: list[int] = []
    for bin_index in range(5):
        selected = np.asarray(path_mask, dtype=bool) & (np.asarray(strata) == bin_index)
        count = int(selected.sum())
        if count <= 0:
            raise ArtifactCompatibilityError("Stein witness path is missing a time bin")
        counts.append(count)
        means.append(np.asarray(residual_matrix, dtype=np.float64)[selected].mean(axis=0))
    return float(np.square(np.stack(means, axis=0)).mean()), counts


def _stein_path_rows(
    *,
    full_gradients: np.ndarray,
    linear_gradients: np.ndarray,
    arrays: ScoreArrays,
    dynamics: DirectFluxMNISTConfig,
    model_seed: int,
    witness_plan: Mapping[str, Any],
    bank: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    states = arrays.states.to(device)
    full = torch.as_tensor(full_gradients, device=device, dtype=states.dtype)
    linear = torch.as_tensor(linear_gradients, device=device, dtype=states.dtype)
    ratio = edge_ratio_channels(states, int(dynamics.grid_size))
    theta = harmonic_mobility_exact(states, dynamics)
    rates = arrays.rates.to(device)
    plan_arrays = dict(witness_plan["arrays"])
    patterns = torch.as_tensor(plan_arrays[f"{bank}_patterns"], device=device, dtype=states.dtype)
    linear_scales = torch.as_tensor(plan_arrays[f"{bank}_linear_scale"], device=device, dtype=states.dtype)
    quadratic_scales = torch.as_tensor(plan_arrays[f"{bank}_quadratic_scale"], device=device, dtype=states.dtype)
    alpha = float(edge_alpha_value(dynamics))
    residual_full: list[Tensor] = []
    residual_linear: list[Tensor] = []
    full_edges = edge_difference_channels(full, int(dynamics.grid_size))
    linear_edges = edge_difference_channels(linear, int(dynamics.grid_size))
    n_squared = float(dynamics.grid_size**2)
    for witness_index, witness in enumerate(patterns):
        edge_direction = edge_difference_channels(
            witness.expand(states.shape[0], -1), int(dynamics.grid_size)
        )
        linear_edge_gradient = edge_direction / linear_scales[witness_index]
        linear_generator = n_squared * (
            (2.0 * alpha + 1.0) * ratio * linear_edge_gradient
        ).flatten(1).sum(dim=1) * rates
        linear_gamma_full = n_squared * (
            theta * full_edges * linear_edge_gradient
        ).flatten(1).sum(dim=1) * rates
        linear_gamma_baseline = n_squared * (
            theta * linear_edges * linear_edge_gradient
        ).flatten(1).sum(dim=1) * rates
        residual_full.append(linear_generator + linear_gamma_full)
        residual_linear.append(linear_generator + linear_gamma_baseline)

        coordinate = states @ witness
        quadratic_edge_gradient = (
            coordinate[:, None, None, None] * edge_direction
            / quadratic_scales[witness_index]
        )
        quadratic_edge_hessian = edge_direction.square() / quadratic_scales[witness_index]
        quadratic_generator = n_squared * (
            theta * quadratic_edge_hessian
            + (2.0 * alpha + 1.0) * ratio * quadratic_edge_gradient
        ).flatten(1).sum(dim=1) * rates
        quadratic_gamma_full = n_squared * (
            theta * full_edges * quadratic_edge_gradient
        ).flatten(1).sum(dim=1) * rates
        quadratic_gamma_baseline = n_squared * (
            theta * linear_edges * quadratic_edge_gradient
        ).flatten(1).sum(dim=1) * rates
        residual_full.append(quadratic_generator + quadratic_gamma_full)
        residual_linear.append(quadratic_generator + quadratic_gamma_baseline)
    full_matrix = torch.stack(residual_full, dim=1).detach().double().cpu().numpy()
    linear_matrix = torch.stack(residual_linear, dim=1).detach().double().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for path_id in sorted(set(map(int, arrays.path_ids.tolist()))):
        path_selected = arrays.path_ids == path_id
        full_disc, bin_counts = _binwise_stein_discrepancy(
            full_matrix, path_mask=path_selected, strata=arrays.strata
        )
        linear_disc, linear_bin_counts = _binwise_stein_discrepancy(
            linear_matrix, path_mask=path_selected, strata=arrays.strata
        )
        if linear_bin_counts != bin_counts:
            raise ArtifactCompatibilityError("Stein witness time-bin counts differ")
        rows.append(
            {
                "model_seed": int(model_seed), "path_id": int(path_id), "stein_bank": str(bank),
                "state_count": int(path_selected.sum()), "full_discrepancy": full_disc,
                "linear_discrepancy": linear_disc,
                "stein_discrepancy_improvement": linear_disc - full_disc,
                "finite_fraction": float(
                    np.isfinite(full_matrix[path_selected]).all()
                    and np.isfinite(linear_matrix[path_selected]).all()
                ),
                "time_bin_count": 5,
                "time_bin_state_counts": bin_counts,
                "linear_witness_count": STEIN_LINEAR_WITNESSES,
                "quadratic_witness_count": STEIN_QUADRATIC_WITNESSES,
                "witness_count": STEIN_LINEAR_WITNESSES + STEIN_QUADRATIC_WITNESSES,
                "aggregation": "square-within-path-and-time-bin-then-equal-average",
            }
        )
    return rows


def _physical_flux_rows(
    *,
    full_gradients: np.ndarray,
    linear_gradients: np.ndarray,
    arrays: ScoreArrays,
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
) -> np.ndarray:
    states = arrays.states.to(device)
    edge = edge_difference_channels(
        torch.as_tensor(full_gradients - linear_gradients, device=device, dtype=states.dtype),
        int(dynamics.grid_size),
    )
    flux = physical_flux_from_edge_score(
        edge, states, dynamics, time_change=arrays.rates.to(device)
    )
    return flux.detach().float().cpu().reshape(states.shape[0], -1).numpy()


def _prepare_physical_task_evaluation(
    *,
    model: nn.Module,
    baseline: D0LinearSplinePotential,
    train: ScoreArrays,
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[D0LinearSplinePotential, dict[str, Any]]:
    """Finish every fallible non-audit calculation before audit intent."""

    baseline_device = copy.deepcopy(baseline).to(device)
    train_components = _risk_components(
        model, baseline_device, train, dynamics, device=device,
        batch_size=int(args.validation_batch_size), probes_per_state=int(args.selection_probes),
        probe_seed=int(args.selection_probe_seed),
    )
    return baseline_device, _paired_risk_summary(train_components)


def _evaluate_physical_task(
    *,
    model: nn.Module,
    baseline_device: D0LinearSplinePotential,
    train_summary: Mapping[str, Any],
    audit: ScoreArrays,
    selection_summary: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    device: torch.device,
    model_seed: int,
    witness_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """Evaluate the frozen audit split only after an intent is committed."""

    audit_specs = (
        ("audit_a", int(args.audit_probe_a_seed)),
        ("audit_b", int(args.audit_probe_b_seed)),
    )
    audit_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    for bank, seed in audit_specs:
        components = _risk_components(
            model, baseline_device, audit, dynamics, device=device,
            batch_size=int(args.validation_batch_size), probes_per_state=int(args.audit_probes),
            probe_seed=seed,
        )
        for path_id in sorted(set(map(int, audit.path_ids.tolist()))):
            path_mask = audit.path_ids == path_id
            for scope, mask in (
                ("overall", path_mask),
                ("data_end", path_mask & (audit.strata == 4)),
            ):
                audit_rows.append(
                    {"model_seed": int(model_seed), "audit_bank": bank, "scope": scope, "path_id": int(path_id), **_paired_risk_summary(components, mask)}
                )
        for bin_index in range(5):
            mask = audit.strata == bin_index
            time_rows.append(
                {"model_seed": int(model_seed), "audit_bank": bank, "time_bin": int(bin_index), "tau_fraction_lo": float(bin_index) / 5.0, "tau_fraction_hi": float(bin_index + 1) / 5.0, **_paired_risk_summary(components, mask)}
            )
    full_grad = _cell_gradients(model, audit, device=device, batch_size=int(args.validation_batch_size))
    linear_grad = _cell_gradients(baseline_device, audit, device=device, batch_size=int(args.validation_batch_size))
    stein_rows = _stein_path_rows(
        full_gradients=full_grad, linear_gradients=linear_grad, arrays=audit,
        dynamics=dynamics, model_seed=int(model_seed), witness_plan=witness_plan,
        bank="stein_a", device=device,
    ) + _stein_path_rows(
        full_gradients=full_grad, linear_gradients=linear_grad, arrays=audit,
        dynamics=dynamics, model_seed=int(model_seed), witness_plan=witness_plan,
        bank="stein_b", device=device,
    )
    flux = _physical_flux_rows(
        full_gradients=full_grad, linear_gradients=linear_grad, arrays=audit,
        dynamics=dynamics, device=device,
    )
    selection_scopes = dict(selection_summary.get("selection_metrics", {}))
    finite = bool(np.isfinite(flux).all())
    finite = finite and bool(int(selection_summary.get("complete", 0)))
    finite = finite and int(selection_summary.get("selected_step", 0)) > 0
    finite = finite and all(
        float(dict(selection_scopes.get(scope, {})).get("finite_fraction", 0.0)) == 1.0
        for scope in ("overall", "data_end")
    )
    finite = finite and float(train_summary.get("finite_fraction", 0.0)) == 1.0
    finite = finite and all(float(row.get("finite_fraction", 0.0)) == 1.0 for row in audit_rows)
    finite = finite and all(float(row.get("finite_fraction", 0.0)) == 1.0 for row in time_rows)
    finite = finite and all(float(row.get("finite_fraction", 0.0)) == 1.0 for row in stein_rows)
    witness_metadata = dict(witness_plan["metadata"])
    result = {
        "schema": TASK_RESULT_SCHEMA,
        "schema_version": TASK_RESULT_SCHEMA_VERSION,
        **dict(selection_summary),
        "model_seed": int(model_seed),
        "training_seed": int(model_seed),
        "complete": int(finite),
        "finite": int(finite),
        "train_metrics": {"overall": dict(train_summary)},
        "audit_path_ids": sorted(set(map(int, audit.path_ids.tolist()))),
        "audit_path_ids_sha256": array_fingerprint(np.asarray(audit.path_ids, dtype=np.int64)),
        "audit_end_substeps_sha256": array_fingerprint(np.asarray(audit.end_substeps, dtype=np.int64)),
        "audit_path_metrics": audit_rows,
        "stein_path_metrics": stein_rows,
        "stein_witness_plan_fingerprint": witness_metadata["fingerprint"],
        "stein_witness_plan_sha256": witness_metadata["npz_sha256"],
        "time_bin_metrics": time_rows,
        "sampling_performed": 0,
    }
    return result, flux, time_rows


def _cross_seed_flux_cosines(
    seed_fluxes: Mapping[int, np.ndarray], audit: ScoreArrays
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seeds = sorted(seed_fluxes)
    for index, left in enumerate(seeds):
        for right in seeds[index + 1 :]:
            for path_id in sorted(set(map(int, audit.path_ids.tolist()))):
                mask = audit.path_ids == path_id
                first = np.asarray(seed_fluxes[left])[mask].reshape(-1).astype(np.float64)
                second = np.asarray(seed_fluxes[right])[mask].reshape(-1).astype(np.float64)
                denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
                cosine = float(np.dot(first, second) / denominator) if denominator > 0.0 else 0.0
                rows.append({"path_id": int(path_id), "seed_a": int(left), "seed_b": int(right), "cosine": float(np.clip(cosine, -1.0, 1.0))})
    return rows


def _remap_control_template(
    arrays: ScoreArrays,
    *,
    path_count: int,
    first_control_path_id: int,
    role: str,
) -> tuple[ScoreArrays, list[dict[str, int]]]:
    source_ids = list(dict.fromkeys(map(int, arrays.path_ids.tolist())))[: int(path_count)]
    if len(source_ids) != int(path_count):
        raise ArtifactCompatibilityError(f"{role} control template lacks whole paths")
    row_parts: list[np.ndarray] = []
    mappings: list[dict[str, int]] = []
    remapped_parts: list[np.ndarray] = []
    for offset, source_id in enumerate(source_ids):
        rows = np.flatnonzero(arrays.path_ids == source_id).astype(np.int64)
        if rows.size == 0:
            raise ArtifactCompatibilityError("control template selected an empty path")
        control_id = int(first_control_path_id) + offset
        row_parts.append(rows)
        remapped_parts.append(np.full(rows.size, control_id, dtype=np.int64))
        mappings.append({"source_path_id": int(source_id), "control_path_id": control_id})
    selected = np.concatenate(row_parts)
    path_ids = np.concatenate(remapped_parts)
    pixels = int(arrays.states.shape[1])
    placeholder = torch.full(
        (int(selected.size), pixels), 1.0 / float(pixels), dtype=torch.float32
    )
    ids = torch.as_tensor(selected, dtype=torch.long)
    result = ScoreArrays(
        states=placeholder,
        tau=arrays.tau.index_select(0, ids).clone(),
        tau_fraction=arrays.tau_fraction.index_select(0, ids).clone(),
        labels=arrays.labels.index_select(0, ids).clone(),
        path_ids=path_ids,
        strata=np.asarray(arrays.strata[selected], dtype=np.int64),
        end_substeps=np.asarray(arrays.end_substeps[selected], dtype=np.int64),
        rates=arrays.rates.index_select(0, ids).clone(),
        horizon=float(arrays.horizon),
        role=str(role),
    )
    return result, mappings


def _control_templates(
    *,
    run_dir: Path,
    train: ScoreArrays,
    selection: ScoreArrays,
    audit: ScoreArrays,
    scientific_fingerprint: str,
    cache_fingerprint: str,
) -> tuple[ScoreArrays, ScoreArrays, ScoreArrays, dict[str, Any]]:
    train_control, train_map = _remap_control_template(
        train, path_count=CONTROL_SPLIT_COUNTS[0], first_control_path_id=3_000_000, role="control-train"
    )
    selection_control, selection_map = _remap_control_template(
        selection, path_count=CONTROL_SPLIT_COUNTS[1], first_control_path_id=3_100_000, role="control-selection"
    )
    audit_control, audit_map = _remap_control_template(
        audit, path_count=CONTROL_SPLIT_COUNTS[2], first_control_path_id=3_200_000, role="control-audit"
    )
    value = {
        "schema": CONTROL_SPLIT_SCHEMA,
        "schema_version": CONTROL_SPLIT_SCHEMA_VERSION,
        "scientific_fingerprint": str(scientific_fingerprint),
        "cache_fingerprint": str(cache_fingerprint),
        "split_counts": list(CONTROL_SPLIT_COUNTS),
        "cluster_count": int(sum(CONTROL_SPLIT_COUNTS)),
        "anchors_per_cluster": int(train_control.states.shape[0] // CONTROL_SPLIT_COUNTS[0]),
        "train_mapping": train_map,
        "selection_mapping": selection_map,
        "audit_mapping": audit_map,
        "physical_state_values_reused": 0,
        "physical_path_ids_reused": 0,
        "whole_cluster_isolation": 1,
        "fingerprint": "",
        "sampling_performed": 0,
    }
    control_ids = [
        int(row["control_path_id"])
        for mapping in (train_map, selection_map, audit_map)
        for row in mapping
    ]
    if len(control_ids) != len(set(control_ids)):
        raise ArtifactCompatibilityError("control path split overlaps")
    value["fingerprint"] = config_fingerprint(
        {key: item for key, item in value.items() if key != "fingerprint"}
    )
    path = run_dir / "control_path_split.json"
    if path.is_file():
        if _json_load(path) != value:
            raise ArtifactCompatibilityError("control path split fingerprint mismatch")
    else:
        atomic_write_json(path, value)
    return train_control, selection_control, audit_control, value


def _verify_control_split_artifact(
    path: Path, *, scientific_fingerprint: str, cache_fingerprint: str
) -> dict[str, Any]:
    value = _json_load(path)
    if (
        value.get("schema") != CONTROL_SPLIT_SCHEMA
        or int(value.get("schema_version", -1)) != CONTROL_SPLIT_SCHEMA_VERSION
        or value.get("scientific_fingerprint") != scientific_fingerprint
        or value.get("cache_fingerprint") != cache_fingerprint
        or tuple(value.get("split_counts", ())) != CONTROL_SPLIT_COUNTS
        or int(value.get("cluster_count", -1)) != sum(CONTROL_SPLIT_COUNTS)
        or int(value.get("whole_cluster_isolation", 0)) != 1
        or int(value.get("physical_state_values_reused", -1)) != 0
        or int(value.get("physical_path_ids_reused", -1)) != 0
    ):
        raise ArtifactCompatibilityError("control path split is incompatible")
    if value.get("fingerprint") != config_fingerprint(
        {key: item for key, item in value.items() if key != "fingerprint"}
    ):
        raise ArtifactCompatibilityError("control path split fingerprint is invalid")
    mappings = [
        list(value.get(key, []))
        for key in ("train_mapping", "selection_mapping", "audit_mapping")
    ]
    if tuple(map(len, mappings)) != CONTROL_SPLIT_COUNTS:
        raise ArtifactCompatibilityError("control path split coverage is incomplete")
    ids = [int(row["control_path_id"]) for mapping in mappings for row in mapping]
    if len(ids) != len(set(ids)):
        raise ArtifactCompatibilityError("control path split overlaps")
    return value


def _synthetic_arrays(
    template: ScoreArrays,
    *,
    dynamics: DirectFluxMNISTConfig,
    seed: int,
    teacher: bool,
) -> ScoreArrays:
    fractions = template.tau_fraction.clone()
    if teacher:
        states = sample_teacher_dirichlet(
            fractions, int(dynamics.grid_size), seed=int(seed), device="cpu", dtype=torch.float32
        )
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        concentration = torch.full_like(template.states, float(edge_alpha_value(dynamics)))
        raw = torch._standard_gamma(concentration, generator=generator)
        states = raw / raw.sum(dim=1, keepdim=True)
    return ScoreArrays(
        states=states.float().contiguous(), tau=template.tau.clone(),
        tau_fraction=fractions, labels=template.labels.clone(),
        path_ids=template.path_ids.copy(), strata=template.strata.copy(),
        end_substeps=template.end_substeps.copy(), rates=template.rates.clone(),
        horizon=float(template.horizon), role=str(template.role),
    )


def _teacher_metrics(
    *,
    model: nn.Module,
    baseline: nn.Module,
    audit: ScoreArrays,
    dynamics: DirectFluxMNISTConfig,
    selected_step: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    full_grad = _cell_gradients(model, audit, device=device, batch_size=int(args.validation_batch_size))
    linear_grad = _cell_gradients(baseline, audit, device=device, batch_size=int(args.validation_batch_size))
    states = audit.states.to(device)
    full_edge = edge_difference_channels(torch.as_tensor(full_grad, device=device, dtype=states.dtype), int(dynamics.grid_size))
    linear_edge = edge_difference_channels(torch.as_tensor(linear_grad, device=device, dtype=states.dtype), int(dynamics.grid_size))
    true_edge = teacher_edge_score(states, audit.tau_fraction.to(device), reference_alpha=float(edge_alpha_value(dynamics)))
    theta = harmonic_mobility_exact(states, dynamics)
    rates = audit.rates.to(device)

    def scope(mask: np.ndarray) -> dict[str, float]:
        ids = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=device)
        target = true_edge.index_select(0, ids)
        prediction = full_edge.index_select(0, ids)
        linear_prediction = linear_edge.index_select(0, ids)
        weight = theta.index_select(0, ids)
        full_mse = float((weight * (prediction - target).square()).mean().detach().cpu())
        linear_mse = float((weight * (linear_prediction - target).square()).mean().detach().cpu())
        zero_mse = float((weight * target.square()).mean().detach().cpu())
        target_flux = physical_flux_from_edge_score(target, states.index_select(0, ids), dynamics, time_change=rates.index_select(0, ids))
        predicted_flux = physical_flux_from_edge_score(prediction, states.index_select(0, ids), dynamics, time_change=rates.index_select(0, ids))
        flat_target = target_flux.reshape(-1).double()
        flat_prediction = predicted_flux.reshape(-1).double()
        denom = torch.linalg.vector_norm(flat_target) * torch.linalg.vector_norm(flat_prediction)
        cosine = float((flat_target @ flat_prediction / denom.clamp_min(1e-30)).detach().cpu())
        relative = float((torch.linalg.vector_norm(flat_prediction - flat_target) / torch.linalg.vector_norm(flat_target).clamp_min(1e-30)).detach().cpu())
        return {"score_gain": 1.0 - full_mse / zero_mse, "linear_score_gain": 1.0 - linear_mse / zero_mse, "nonlinear_gain_vs_linear": linear_mse - full_mse, "flux_cosine": cosine, "flux_relative_l2": relative}

    fractions = audit.tau_fraction.numpy()
    overall = scope(np.ones(fractions.size, dtype=bool))
    data_end = scope(audit.strata == 4)
    bins = [scope(audit.strata == index) for index in range(5)]
    return {
        "complete": 1, "selected_step": int(selected_step),
        "audit_overall_score_gain": overall["score_gain"],
        "audit_data_end_score_gain": data_end["score_gain"],
        "overall_flux_cosine": overall["flux_cosine"],
        "time_bin_flux_cosines": [value["flux_cosine"] for value in bins],
        "overall_relative_flux_l2": overall["flux_relative_l2"],
        "time_bin_relative_flux_l2": [value["flux_relative_l2"] for value in bins],
        "nonlinear_gain_vs_linear": overall["nonlinear_gain_vs_linear"],
        "time_bins": bins,
        "sampling_performed": 0,
    }


def _complete_control_task(
    *,
    task_dir: Path,
    artifact_path: Path,
    artifact: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    training_seed: int,
    selected_step: int,
) -> None:
    atomic_write_json(artifact_path, artifact)
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "schema": TASK_STATUS_SCHEMA,
            "schema_version": TASK_STATUS_SCHEMA_VERSION,
            "status": "complete",
            "training_seed": int(training_seed),
            "selected_step": int(selected_step),
            "fingerprints": dict(fingerprints),
            "control_result_sha256": file_fingerprint(artifact_path),
            "sampling_performed": 0,
        },
    )


def _load_completed_control_task(
    *,
    task_dir: Path,
    artifact_path: Path,
    schema: str,
    fingerprints: Mapping[str, Any],
    training_seed: int,
    require_complete_status: bool = True,
) -> dict[str, Any]:
    status_path = task_dir / "task_status.json"
    if not status_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(status_path if not status_path.is_file() else artifact_path)
    status = _json_load(status_path)
    artifact = _json_load(artifact_path)
    if (
        status.get("schema") != TASK_STATUS_SCHEMA
        or int(status.get("schema_version", -1)) != TASK_STATUS_SCHEMA_VERSION
        or (require_complete_status and status.get("status") != "complete")
        or int(status.get("training_seed", -1)) != int(training_seed)
        or dict(status.get("fingerprints", {})) != dict(fingerprints)
        or (
            require_complete_status
            and status.get("control_result_sha256") != file_fingerprint(artifact_path)
        )
        or artifact.get("schema") != schema
        or int(artifact.get("schema_version", -1)) != 1
        or dict(artifact.get("fingerprints", {})) != dict(fingerprints)
        or int(artifact.get("sampling_performed", -1)) != 0
    ):
        raise ArtifactCompatibilityError("completed synthetic control task is incompatible")
    summary = dict(artifact.get("training_summary", {}))
    selected_step = int(summary.get("selected_step", -1))
    if selected_step <= 0 or selected_step != int(status.get("selected_step", -2)):
        raise ArtifactCompatibilityError("synthetic control selected-step binding mismatch")
    checkpoints = task_dir / "checkpoints"
    best_pointer_path = checkpoints / "best.json"
    best_path = checkpoints / "best_ema.pt"
    if not best_pointer_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(best_pointer_path if not best_pointer_path.is_file() else best_path)
    pointer = _json_load(best_pointer_path)
    if (
        int(pointer.get("selected_step", -1)) != selected_step
        or dict(pointer.get("fingerprints", {})) != dict(fingerprints)
        or pointer.get("best_ema_sha256") != file_fingerprint(best_path)
        or summary.get("checkpoint_sha256") != file_fingerprint(best_path)
        or summary.get("best_pointer_sha256") != file_fingerprint(best_pointer_path)
    ):
        raise ArtifactCompatibilityError("synthetic control best-checkpoint binding mismatch")
    return artifact


def _control_evidence_records(run_dir: Path) -> dict[str, Any]:
    paths = {
        "positive_teacher_result": run_dir / "controls" / "positive_teacher.json",
        "positive_teacher_status": run_dir / "controls" / "positive_teacher" / "task_status.json",
        "positive_teacher_best_pointer": run_dir / "controls" / "positive_teacher" / "checkpoints" / "best.json",
        "positive_teacher_best_checkpoint": run_dir / "controls" / "positive_teacher" / "checkpoints" / "best_ema.pt",
        "null_result": run_dir / "controls" / "null_control.json",
        "null_status": run_dir / "controls" / "null" / "task_status.json",
        "null_best_pointer": run_dir / "controls" / "null" / "checkpoints" / "best.json",
        "null_best_checkpoint": run_dir / "controls" / "null" / "checkpoints" / "best_ema.pt",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing control evidence: " + ", ".join(missing))
    return {name: _artifact_record(path) for name, path in paths.items()}


def _validate_completed_controls_gate(
    *,
    run_dir: Path,
    controls_gate: Mapping[str, Any],
    args: argparse.Namespace,
    scientific_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint_value: str,
    cache_fingerprint: str,
    operator_gate: Mapping[str, Any],
    require_evidence: bool,
) -> dict[str, Any]:
    """Validate a frozen controls decision and, when passed, all bound evidence."""

    control_split = _verify_control_split_artifact(
        run_dir / "control_path_split.json",
        scientific_fingerprint=scientific_fingerprint,
        cache_fingerprint=cache_fingerprint,
    )
    expected_binding = {
        "scientific_fingerprint": scientific_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "source_fingerprint": source_fingerprint_value,
        "cache_fingerprint": cache_fingerprint,
        "control_split_fingerprint": control_split["fingerprint"],
    }
    record = dict(controls_gate)
    if (
        record.get("schema") != RUN_SCHEMA + "-controls-gate"
        or int(record.get("schema_version", -1)) != 1
        or dict(record.get("binding", {})) != expected_binding
        or int(record.get("sampling_performed", -1)) != 0
    ):
        raise ArtifactCompatibilityError("completed controls gate is incompatible")
    if not require_evidence:
        return record

    controls_root = run_dir / "controls"
    control_binding = cache_fingerprint + ":control-split:" + str(control_split["fingerprint"])
    teacher_fp = _task_checkpoint_fingerprints(
        manifest_fingerprint=scientific_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value,
        cache_fingerprint=control_binding + ":teacher",
        baseline_path=controls_root / "baselines" / "positive_teacher_linear_spline.pt",
        seed=int(args.positive_teacher_train_seed),
        train_steps=int(args.control_steps),
    )
    null_fp = _task_checkpoint_fingerprints(
        manifest_fingerprint=scientific_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value,
        cache_fingerprint=control_binding + ":null",
        baseline_path=controls_root / "baselines" / "null_linear_spline.pt",
        seed=int(args.null_train_seed),
        train_steps=int(args.control_steps),
    )
    teacher_record = _load_completed_control_task(
        task_dir=controls_root / "positive_teacher",
        artifact_path=controls_root / "positive_teacher.json",
        schema=RUN_SCHEMA + "-positive-teacher-control",
        fingerprints=teacher_fp,
        training_seed=int(args.positive_teacher_train_seed),
    )
    null_record = _load_completed_control_task(
        task_dir=controls_root / "null",
        artifact_path=controls_root / "null_control.json",
        schema=RUN_SCHEMA + "-null-control",
        fingerprints=null_fp,
        training_seed=int(args.null_train_seed),
    )
    teacher_gate = dict(teacher_record.get("gate", {}))
    null_gate = dict(null_record.get("gate", {}))
    recomputed_teacher_gate = evaluate_positive_teacher_control(
        dict(teacher_record.get("metrics", {})), _thresholds(args)
    )
    recomputed_null_gate = evaluate_null_control(dict(null_record.get("metrics", {})))
    if teacher_gate != recomputed_teacher_gate or null_gate != recomputed_null_gate:
        raise ArtifactCompatibilityError("control metrics and stored task gate disagree")
    expected_bundle = evaluate_control_bundle(
        operator_gate=operator_gate,
        positive_teacher_gate=teacher_gate,
        null_control_gate=null_gate,
    )
    if (
        dict(record.get("teacher", {})) != teacher_gate
        or dict(record.get("null", {})) != null_gate
        or any(record.get(key) != value for key, value in expected_bundle.items())
        or dict(record.get("evidence", {})) != _control_evidence_records(run_dir)
    ):
        raise ArtifactCompatibilityError("passed controls gate evidence is incompatible")
    return record


def _run_controls(
    *,
    run_dir: Path,
    train: ScoreArrays,
    selection: ScoreArrays,
    audit: ScoreArrays,
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    device: torch.device,
    scientific_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint_value: str,
    cache_fingerprint: str,
    operator_gate: Mapping[str, Any],
    show_progress: bool,
) -> dict[str, Any]:
    controls_root = run_dir / "controls"
    controls_root.mkdir(parents=True, exist_ok=True)
    control_train, control_selection, control_audit, control_split = _control_templates(
        run_dir=run_dir, train=train, selection=selection, audit=audit,
        scientific_fingerprint=scientific_fingerprint,
        cache_fingerprint=cache_fingerprint,
    )
    control_binding = cache_fingerprint + ":control-split:" + str(control_split["fingerprint"])
    teacher_train = _synthetic_arrays(control_train, dynamics=dynamics, seed=int(args.positive_teacher_data_seed), teacher=True)
    teacher_selection = _synthetic_arrays(control_selection, dynamics=dynamics, seed=int(args.positive_teacher_data_seed) + 1, teacher=True)
    teacher_audit = _synthetic_arrays(control_audit, dynamics=dynamics, seed=int(args.positive_teacher_data_seed) + 2, teacher=True)
    teacher_baseline, _ = _load_or_fit_baseline(
        run_dir=controls_root, train=teacher_train, dynamics=dynamics,
        scientific_fingerprint=scientific_fingerprint, cache_fingerprint=control_binding + ":teacher",
        device=device, name="positive_teacher",
    )
    teacher_baseline_path = controls_root / "baselines" / "positive_teacher_linear_spline.pt"
    teacher_fp = _task_checkpoint_fingerprints(
        manifest_fingerprint=scientific_fingerprint, runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value, cache_fingerprint=control_binding + ":teacher",
        baseline_path=teacher_baseline_path, seed=int(args.positive_teacher_train_seed), train_steps=int(args.control_steps),
    )
    teacher_task_dir = controls_root / "positive_teacher"
    teacher_artifact_path = controls_root / "positive_teacher.json"
    teacher_status_path = teacher_task_dir / "task_status.json"
    if teacher_status_path.is_file() and _json_load(teacher_status_path).get("status") == "complete":
        teacher_record = _load_completed_control_task(
            task_dir=teacher_task_dir, artifact_path=teacher_artifact_path,
            schema=RUN_SCHEMA + "-positive-teacher-control", fingerprints=teacher_fp,
            training_seed=int(args.positive_teacher_train_seed),
        )
        teacher_values = dict(teacher_record["metrics"])
        teacher_gate = dict(teacher_record["gate"])
    elif teacher_artifact_path.is_file():
        teacher_record = _load_completed_control_task(
            task_dir=teacher_task_dir, artifact_path=teacher_artifact_path,
            schema=RUN_SCHEMA + "-positive-teacher-control", fingerprints=teacher_fp,
            training_seed=int(args.positive_teacher_train_seed), require_complete_status=False,
        )
        teacher_values = dict(teacher_record["metrics"])
        teacher_gate = dict(teacher_record["gate"])
        teacher_summary = dict(teacher_record["training_summary"])
        _complete_control_task(
            task_dir=teacher_task_dir, artifact_path=teacher_artifact_path,
            artifact=teacher_record, fingerprints=teacher_fp,
            training_seed=int(args.positive_teacher_train_seed),
            selected_step=int(teacher_summary["selected_step"]),
        )
    else:
        teacher_model, teacher_summary = _train_potential_task(
            task_dir=teacher_task_dir, train=teacher_train, selection=teacher_selection,
            baseline=teacher_baseline, dynamics=dynamics, device=device, args=args,
            training_seed=int(args.positive_teacher_train_seed), train_steps=int(args.control_steps),
            fingerprints=teacher_fp, show_progress=show_progress,
        )
        teacher_values = _teacher_metrics(
            model=teacher_model, baseline=teacher_baseline.to(device), audit=teacher_audit,
            dynamics=dynamics, selected_step=int(teacher_summary["selected_step"]), args=args, device=device,
        )
        teacher_gate = evaluate_positive_teacher_control(teacher_values, _thresholds(args))
        teacher_record = {
            "schema": RUN_SCHEMA + "-positive-teacher-control",
            "schema_version": 1,
            "metrics": teacher_values,
            "gate": teacher_gate,
            "training_summary": teacher_summary,
            "fingerprints": teacher_fp,
            "sampling_performed": 0,
        }
        _complete_control_task(
            task_dir=teacher_task_dir, artifact_path=teacher_artifact_path,
            artifact=teacher_record, fingerprints=teacher_fp,
            training_seed=int(args.positive_teacher_train_seed),
            selected_step=int(teacher_summary["selected_step"]),
        )
    recomputed_teacher_gate = evaluate_positive_teacher_control(
        teacher_values, _thresholds(args)
    )
    if teacher_gate != recomputed_teacher_gate:
        raise ArtifactCompatibilityError("positive-teacher metrics and stored gate disagree")
    teacher_gate = recomputed_teacher_gate

    null_train = _synthetic_arrays(control_train, dynamics=dynamics, seed=int(args.null_data_seed), teacher=False)
    null_selection = _synthetic_arrays(control_selection, dynamics=dynamics, seed=int(args.null_data_seed) + 1, teacher=False)
    null_audit = _synthetic_arrays(control_audit, dynamics=dynamics, seed=int(args.null_data_seed) + 2, teacher=False)
    null_baseline, _ = _load_or_fit_baseline(
        run_dir=controls_root, train=null_train, dynamics=dynamics,
        scientific_fingerprint=scientific_fingerprint, cache_fingerprint=control_binding + ":null",
        device=device, name="null",
    )
    null_baseline_path = controls_root / "baselines" / "null_linear_spline.pt"
    null_fp = _task_checkpoint_fingerprints(
        manifest_fingerprint=scientific_fingerprint, runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value, cache_fingerprint=control_binding + ":null",
        baseline_path=null_baseline_path, seed=int(args.null_train_seed), train_steps=int(args.control_steps),
    )
    null_task_dir = controls_root / "null"
    null_artifact_path = controls_root / "null_control.json"
    null_status_path = null_task_dir / "task_status.json"
    if null_status_path.is_file() and _json_load(null_status_path).get("status") == "complete":
        null_record = _load_completed_control_task(
            task_dir=null_task_dir, artifact_path=null_artifact_path,
            schema=RUN_SCHEMA + "-null-control", fingerprints=null_fp,
            training_seed=int(args.null_train_seed),
        )
        null_values = dict(null_record["metrics"])
        null_gate = dict(null_record["gate"])
    elif null_artifact_path.is_file():
        null_record = _load_completed_control_task(
            task_dir=null_task_dir, artifact_path=null_artifact_path,
            schema=RUN_SCHEMA + "-null-control", fingerprints=null_fp,
            training_seed=int(args.null_train_seed), require_complete_status=False,
        )
        null_values = dict(null_record["metrics"])
        null_gate = dict(null_record["gate"])
        null_summary = dict(null_record["training_summary"])
        _complete_control_task(
            task_dir=null_task_dir, artifact_path=null_artifact_path,
            artifact=null_record, fingerprints=null_fp,
            training_seed=int(args.null_train_seed), selected_step=int(null_summary["selected_step"]),
        )
    else:
        null_model, null_summary = _train_potential_task(
            task_dir=null_task_dir, train=null_train, selection=null_selection,
            baseline=null_baseline, dynamics=dynamics, device=device, args=args,
            training_seed=int(args.null_train_seed), train_steps=int(args.control_steps),
            fingerprints=null_fp, show_progress=show_progress,
        )
        null_components = _risk_components(
            null_model, null_baseline.to(device), null_audit, dynamics, device=device,
            batch_size=int(args.validation_batch_size), probes_per_state=int(args.audit_probes),
            probe_seed=int(args.audit_probe_a_seed),
        )
        null_rows = []
        for path_id in sorted(set(map(int, null_audit.path_ids.tolist()))):
            summary = _paired_risk_summary(null_components, null_audit.path_ids == path_id)
            null_rows.append({"model_seed": int(args.null_train_seed), "path_id": int(path_id), "value": float(summary["score_risk_delta_vs_linear"])})
        interval = bootstrap_whole_path_delta(
            null_rows, value_key="value", reps=int(args.bootstrap_reps),
            confidence=float(args.bootstrap_confidence), seed=int(args.bootstrap_seed),
            expected_model_seeds=[int(args.null_train_seed)],
            expected_path_ids=sorted(set(map(int, null_audit.path_ids.tolist()))),
        )
        null_values = {
            "complete": 1,
            "selected_step": int(null_summary["selected_step"]),
            "audit_improvement_lower_bound": interval["lower_bound"],
            "bootstrap": interval,
            "comparator": "frozen_training_only_linear_spline_step0",
            "control_split_fingerprint": control_split["fingerprint"],
            "sampling_performed": 0,
        }
        null_gate = evaluate_null_control(null_values)
        null_record = {
            "schema": RUN_SCHEMA + "-null-control",
            "schema_version": 1,
            "metrics": null_values,
            "gate": null_gate,
            "training_summary": null_summary,
            "fingerprints": null_fp,
            "sampling_performed": 0,
        }
        _complete_control_task(
            task_dir=null_task_dir, artifact_path=null_artifact_path,
            artifact=null_record, fingerprints=null_fp,
            training_seed=int(args.null_train_seed), selected_step=int(null_summary["selected_step"]),
        )
    recomputed_null_gate = evaluate_null_control(null_values)
    if null_gate != recomputed_null_gate:
        raise ArtifactCompatibilityError("null-control metrics and stored gate disagree")
    null_gate = recomputed_null_gate
    bundle = evaluate_control_bundle(
        operator_gate=operator_gate, positive_teacher_gate=teacher_gate, null_control_gate=null_gate
    )
    record = {
        "schema": RUN_SCHEMA + "-controls-gate",
        "schema_version": 1,
        "binding": {
            "scientific_fingerprint": scientific_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "source_fingerprint": source_fingerprint_value,
            "cache_fingerprint": cache_fingerprint,
            "control_split_fingerprint": control_split["fingerprint"],
        },
        "teacher": teacher_gate,
        "null": null_gate,
        "evidence": _control_evidence_records(run_dir),
        **bundle,
        "sampling_performed": 0,
    }
    atomic_write_json(run_dir / "controls_gate.json", record)
    return record


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _physical_audit_identity(audit: ScoreArrays) -> dict[str, Any]:
    return {
        "state_sha256": array_fingerprint(
            np.asarray(audit.states.detach().cpu(), dtype=np.float32)
        ),
        "tau_sha256": array_fingerprint(
            np.asarray(audit.tau.detach().cpu(), dtype=np.float32)
        ),
        "path_ids_sha256": array_fingerprint(np.asarray(audit.path_ids, dtype=np.int64)),
        "strata_sha256": array_fingerprint(np.asarray(audit.strata, dtype=np.int64)),
        "end_substeps_sha256": array_fingerprint(
            np.asarray(audit.end_substeps, dtype=np.int64)
        ),
        "state_count": int(audit.states.shape[0]),
        "role": str(audit.role),
    }


def _write_physical_audit_intent(
    *,
    task_dir: Path,
    fingerprints: Mapping[str, Any],
    audit: ScoreArrays,
    model_seed: int,
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit an at-most-once audit-access marker before touching audit states."""

    path = task_dir / "audit_intent.json"
    record = {
        "schema": AUDIT_INTENT_SCHEMA,
        "schema_version": AUDIT_INTENT_SCHEMA_VERSION,
        "status": "started",
        "training_seed": int(model_seed),
        "selected_step": int(training_summary["selected_step"]),
        "checkpoint_sha256": str(training_summary["checkpoint_sha256"]),
        "best_pointer_sha256": str(training_summary["best_pointer_sha256"]),
        "fingerprints": dict(fingerprints),
        "audit_identity": _physical_audit_identity(audit),
        "sampling_performed": 0,
    }
    if path.is_file():
        if _json_load(path) != record:
            raise ArtifactCompatibilityError("physical audit intent is incompatible")
    else:
        atomic_write_json(path, record)
    return record


def _validate_physical_audit_intent(
    *,
    task_dir: Path,
    fingerprints: Mapping[str, Any],
    audit: ScoreArrays,
    model_seed: int,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = task_dir / "audit_intent.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    record = _json_load(path)
    if (
        record.get("schema") != AUDIT_INTENT_SCHEMA
        or int(record.get("schema_version", -1)) != AUDIT_INTENT_SCHEMA_VERSION
        or record.get("status") != "started"
        or int(record.get("training_seed", -1)) != int(model_seed)
        or dict(record.get("fingerprints", {})) != dict(fingerprints)
        or dict(record.get("audit_identity", {})) != _physical_audit_identity(audit)
        or int(record.get("sampling_performed", -1)) != 0
    ):
        raise ArtifactCompatibilityError("physical audit intent is incompatible")
    if result is not None and (
        int(record.get("selected_step", -1)) != int(result.get("selected_step", -2))
        or record.get("checkpoint_sha256") != result.get("checkpoint_sha256")
        or record.get("best_pointer_sha256") != result.get("best_pointer_sha256")
    ):
        raise ArtifactCompatibilityError("physical audit result differs from its intent")
    return record


def _repair_best_checkpoint_publication(
    *, task_dir: Path, fingerprints: Mapping[str, Any]
) -> None:
    """Repair only the redundant best publication from its immutable step file."""

    result_path = task_dir / "task_result.json"
    if not result_path.is_file():
        return
    result = _json_load(result_path)
    if (
        result.get("schema") != TASK_RESULT_SCHEMA
        or int(result.get("schema_version", -1)) != TASK_RESULT_SCHEMA_VERSION
        or dict(result.get("fingerprints", {})) != dict(fingerprints)
    ):
        raise ArtifactCompatibilityError("cannot repair best pointer from an incompatible task result")
    step = int(result.get("selected_step", -1))
    checkpoints = task_dir / "checkpoints"
    authoritative = checkpoints / f"step-{step:08d}.pt"
    if step <= 0 or not authoritative.is_file():
        raise ArtifactCompatibilityError("cannot repair best pointer without its authoritative step checkpoint")
    best_path = checkpoints / "best_ema.pt"
    if not best_path.is_file() or file_fingerprint(best_path) != file_fingerprint(authoritative):
        atomic_copy_file(authoritative, best_path)
    pointer = {
        "schema": CHECKPOINT_SCHEMA + "-best",
        "schema_version": 1,
        "selected_step": step,
        "authoritative_filename": authoritative.name,
        "authoritative_sha256": file_fingerprint(authoritative),
        "best_ema_filename": best_path.name,
        "best_ema_sha256": file_fingerprint(best_path),
        "fingerprints": dict(fingerprints),
    }
    pointer_path = checkpoints / "best.json"
    if not pointer_path.is_file() or _json_load(pointer_path) != pointer:
        atomic_write_json(pointer_path, pointer)
    if (
        result.get("checkpoint_sha256") != file_fingerprint(best_path)
        or result.get("best_pointer_sha256") != file_fingerprint(pointer_path)
    ):
        raise ArtifactCompatibilityError("repaired best publication differs from the task-result binding")


def _load_completed_physical_task(
    *,
    task_dir: Path,
    fingerprints: Mapping[str, Any],
    audit: ScoreArrays,
    model_seed: int,
    require_complete_status: bool = True,
) -> tuple[dict[str, Any], np.ndarray]:
    status_path = task_dir / "task_status.json"
    result_path = task_dir / "task_result.json"
    flux_path = task_dir / "audit_nonlinear_flux.npz"
    for path in (status_path, result_path, flux_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    status = _json_load(status_path)
    if (
        status.get("schema") != TASK_STATUS_SCHEMA
        or int(status.get("schema_version", -1)) != TASK_STATUS_SCHEMA_VERSION
        or (require_complete_status and status.get("status") != "complete")
        or int(status.get("training_seed", -1)) != int(model_seed)
        or dict(status.get("fingerprints", {})) != dict(fingerprints)
    ):
        raise ArtifactCompatibilityError("completed score task status is incompatible")
    if require_complete_status and (
        status.get("task_result_sha256") != file_fingerprint(result_path)
        or status.get("flux_sha256") != file_fingerprint(flux_path)
    ):
        raise ArtifactCompatibilityError("completed score task artifact hash mismatch")
    result = _json_load(result_path)
    if (
        result.get("schema") != TASK_RESULT_SCHEMA
        or int(result.get("schema_version", -1)) != TASK_RESULT_SCHEMA_VERSION
        or int(result.get("model_seed", -1)) != int(model_seed)
        or int(result.get("selected_step", -1)) != int(status.get("selected_step", -2))
        or dict(result.get("fingerprints", {})) != dict(fingerprints)
        or result.get("stein_witness_plan_fingerprint") != fingerprints.get("witness_plan_fingerprint")
        or int(result.get("sampling_performed", -1)) != 0
    ):
        raise ArtifactCompatibilityError("completed score task result is incompatible")
    _validate_physical_audit_intent(
        task_dir=task_dir,
        fingerprints=fingerprints,
        audit=audit,
        model_seed=model_seed,
        result=result,
    )
    with np.load(flux_path, allow_pickle=False) as archive:
        if set(archive.files) != {"flux", "path_ids", "end_substeps"}:
            raise ArtifactCompatibilityError("score flux artifact fields are incompatible")
        flux = np.asarray(archive["flux"], dtype=np.float32)
        path_ids = np.asarray(archive["path_ids"], dtype=np.int64)
        end_substeps = np.asarray(archive["end_substeps"], dtype=np.int64)
    if (
        flux.ndim != 2
        or flux.shape[0] != int(audit.states.shape[0])
        or not np.array_equal(path_ids, np.asarray(audit.path_ids, dtype=np.int64))
        or not np.array_equal(end_substeps, np.asarray(audit.end_substeps, dtype=np.int64))
        or result.get("audit_path_ids_sha256") != array_fingerprint(np.asarray(audit.path_ids, dtype=np.int64))
        or result.get("audit_end_substeps_sha256") != array_fingerprint(np.asarray(audit.end_substeps, dtype=np.int64))
    ):
        raise ArtifactCompatibilityError("score flux audit-state identity mismatch")
    if int(result.get("finite", 0)) == 1 and not np.isfinite(flux).all():
        raise ArtifactCompatibilityError("score task claims finite metrics but flux is non-finite")

    checkpoints = task_dir / "checkpoints"
    best_pointer_path = checkpoints / "best.json"
    best_path = checkpoints / "best_ema.pt"
    if not best_pointer_path.is_file() or not best_path.is_file():
        raise FileNotFoundError(best_pointer_path if not best_pointer_path.is_file() else best_path)
    best_pointer = _json_load(best_pointer_path)
    authoritative_name = str(best_pointer.get("authoritative_filename", ""))
    authoritative = checkpoints / authoritative_name
    if (
        best_pointer.get("schema") != CHECKPOINT_SCHEMA + "-best"
        or int(best_pointer.get("schema_version", -1)) != 1
        or Path(authoritative_name).name != authoritative_name
        or not authoritative.is_file()
        or int(best_pointer.get("selected_step", -1)) != int(result["selected_step"])
        or dict(best_pointer.get("fingerprints", {})) != dict(fingerprints)
        or best_pointer.get("authoritative_sha256") != file_fingerprint(authoritative)
        or best_pointer.get("best_ema_sha256") != file_fingerprint(best_path)
        or result.get("checkpoint_sha256") != file_fingerprint(best_path)
        or result.get("best_pointer_sha256") != file_fingerprint(best_pointer_path)
    ):
        raise ArtifactCompatibilityError("completed score task best-checkpoint binding mismatch")
    return result, flux


def _cache_gate_report(
    *,
    args: argparse.Namespace,
    parent_index: D0ScoreStateCacheIndex,
    train_cache: D0ScoreStateCache,
    selection_cache: D0ScoreStateCache,
    audit_index: D0ScoreStateCacheIndex,
    audit_cache: D0ScoreStateCache,
    confirmation: Mapping[str, Any],
    split: Mapping[str, Any],
    scientific_fingerprint: str,
) -> dict[str, Any]:
    train_ids = set(map(int, train_cache.path_ids.tolist()))
    selection_ids = set(map(int, selection_cache.path_ids.tolist()))
    audit_ids = set(map(int, audit_cache.path_ids.tolist()))
    excluded = set(map(int, confirmation["excluded_audit_path_ids"]))
    expected_train = set(map(int, split["train_path_ids"]))
    expected_selection = set(map(int, split["selection_path_ids"]))
    expected_audit = set(map(int, split["fresh_audit_path_ids"]))
    health = {
        "train": _cache_health(train_cache, args, require_rollout_diagnostics=False),
        "selection": _cache_health(selection_cache, args, require_rollout_diagnostics=False),
        "audit": _cache_health(audit_cache, args),
    }
    parent_diag = dict(parent_index.metadata.get("parent_rollout_diagnostics", {}))
    parent_denominator = max(float(parent_diag.get("path_substep_count", 0.0)), 1.0)
    parent_values = {
        "raw_limited_fraction": float(parent_diag.get("raw_limited_fraction", float("inf"))),
        "mobility_weighted_limited_fraction": float(parent_diag.get("mobility_weighted_limited_fraction", float("inf"))),
        "noise_energy_weighted_limited_fraction": float(parent_diag.get("noise_energy_weighted_limited_fraction", float("inf"))),
        "floor_correction_l1_per_path_substep": float(parent_diag.get("floor_correction_l1", float("inf"))) / parent_denominator,
        "renorm_correction_l1_per_path_substep": float(parent_diag.get("renorm_correction_l1", float("inf"))) / parent_denominator,
        "floor_touched_pixels": float(parent_diag.get("floor_touched_pixels", float("inf"))),
        "nonfinite_edges": float(parent_diag.get("nonfinite_edges", float("inf"))),
    }
    parent_checks = {
        "raw_intervention": parent_values["raw_limited_fraction"] <= float(args.max_raw_intervention),
        "mobility_intervention": parent_values["mobility_weighted_limited_fraction"] <= float(args.max_weighted_intervention),
        "noise_intervention": parent_values["noise_energy_weighted_limited_fraction"] <= float(args.max_weighted_intervention),
        "floor_correction": parent_values["floor_correction_l1_per_path_substep"] <= float(args.max_floor_correction_l1),
        "renorm_correction": parent_values["renorm_correction_l1_per_path_substep"] <= float(args.max_renorm_correction_l1),
        "floor_touches": parent_values["floor_touched_pixels"] == 0.0,
        "nonfinite_edges": parent_values["nonfinite_edges"] == 0.0,
    }
    health["parent_rollout"] = {
        "passed": int(all(parent_checks.values())), "values": parent_values,
        "checks": {key: int(value) for key, value in parent_checks.items()},
    }
    checks = {
        "parent_train_count": len(train_ids) == int(args.parent_train_paths),
        "parent_selection_count": len(selection_ids) == int(args.parent_selection_paths),
        "fresh_audit_count": len(audit_ids) == int(args.fresh_audit_paths),
        "exact_role_path_ids": train_ids == expected_train and selection_ids == expected_selection and audit_ids == expected_audit,
        "whole_path_isolation": not train_ids.intersection(selection_ids | audit_ids) and not selection_ids.intersection(audit_ids),
        "parent_audit_excluded": not excluded.intersection(train_ids | selection_ids | audit_ids),
        "parent_roles": train_cache.role == "train" and selection_cache.role == "selection",
        "fresh_audit_role": audit_cache.role == "audit" and audit_cache.origin == "fresh-reference",
        "anchors_per_path": all(cache.anchors_per_path == int(args.anchors_per_path) for cache in (train_cache, selection_cache, audit_cache)),
        "positive_time_minimum": all(int(cache.end_substeps.min()) >= int(args.minimum_forward_substep) for cache in (train_cache, selection_cache, audit_cache)),
        "scientific_fingerprint": (
            parent_index.scientific_fingerprint
            == audit_index.scientific_fingerprint
            == train_cache.scientific_fingerprint
            == selection_cache.scientific_fingerprint
            == audit_cache.scientific_fingerprint
            == scientific_fingerprint
        ),
        "frozen_kernel": all(
            _frozen_cache_kernel_matches(cache.kernel_metadata)
            for cache in (train_cache, selection_cache, audit_cache)
        ),
        "common_kernel": len({config_fingerprint(dict(cache.kernel_metadata)) for cache in (train_cache, selection_cache, audit_cache)}) == 1,
        "common_schedule": len({config_fingerprint(_schedule_core(cache.schedule_metadata)) for cache in (train_cache, selection_cache, audit_cache)}) == 1,
        "parent_confirmation_bound": int(parent_index.metadata.get("parent_audit_paths_excluded", 0)) == 1,
        "parent_index_bound": str(parent_index.metadata.get("parent_cache_index_fingerprint", "")) == str(confirmation["cache_index_fingerprint"]),
        "parent_semantic_bound": str(parent_index.metadata.get("parent_scientific_fingerprint", "")) == str(confirmation["cache_semantic_fingerprint"]),
        "parent_split_bound": str(parent_index.metadata.get("parent_path_split_sha256", "")) == str(confirmation["split_sha256"]),
        "fresh_audit_not_reused": int(audit_cache.provenance.get("parent_audit_reused", -1)) == 0,
        "numerical_health": all(bool(value["passed"]) for value in health.values()),
    }
    return {
        "schema": RUN_SCHEMA + "-cache-gate", "schema_version": 1,
        "passed": int(all(checks.values())),
        "subchecks": {key: int(value) for key, value in checks.items()},
        "health": health,
        "parent_index_fingerprint": parent_index.fingerprint,
        "audit_index_fingerprint": audit_index.fingerprint,
        "claim_scope": "positive-time state samples only",
        "sampling_performed": 0,
    }


def _write_report_tables_and_plots(
    run_dir: Path,
    *,
    seed_results: Sequence[Mapping[str, Any]],
    cosines: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint_rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "tasks").glob("seed-*/checkpoint_metrics.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                checkpoint_rows.append({"task": path.parent.name, **row})
    audit_rows = [dict(row) for result in seed_results for row in result.get("audit_path_metrics", [])]
    stein_rows = [dict(row) for result in seed_results for row in result.get("stein_path_metrics", [])]
    time_rows = [dict(row) for result in seed_results for row in result.get("time_bin_metrics", [])]
    summary_rows = [
        {
            "model_seed": int(result.get("model_seed", -1)),
            "complete": int(result.get("complete", 0)),
            "finite": int(result.get("finite", 0)),
            "selected_step": int(result.get("selected_step", 0)),
            "selection_delta_vs_linear": dict(result.get("selection_metrics", {})).get("overall", {}).get("score_risk_delta_vs_linear"),
            "selection_data_end_delta_vs_linear": dict(result.get("selection_metrics", {})).get("data_end", {}).get("score_risk_delta_vs_linear"),
        }
        for result in seed_results
    ]
    atomic_write_csv(run_dir / "checkpoint_metrics.csv", checkpoint_rows)
    atomic_write_csv(run_dir / "audit_path_score_risks.csv", audit_rows)
    atomic_write_csv(run_dir / "stein_path_metrics.csv", stein_rows)
    atomic_write_csv(run_dir / "score_time_bins.csv", time_rows)
    atomic_write_csv(run_dir / "cross_seed_flux_cosines.csv", list(cosines))
    atomic_write_csv(run_dir / "score_seed_metrics.csv", summary_rows)
    plots: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if checkpoint_rows:
            figure, axis = plt.subplots(figsize=(8.2, 4.8))
            groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
            for row in checkpoint_rows:
                if str(row.get("weights")) != "ema" or str(row.get("scope")) != "overall":
                    continue
                try:
                    groups.setdefault((str(row["task"]), "ema"), []).append((int(row["step"]), float(row["score_risk_delta_vs_linear"])))
                except (KeyError, TypeError, ValueError):
                    continue
            axis.axhline(0.0, color="black", linewidth=0.8)
            for (name, _), values in groups.items():
                values.sort()
                axis.plot([value[0] for value in values], [value[1] for value in values], label=name)
            axis.set_xlabel("training step")
            axis.set_ylabel("selection risk improvement vs linear")
            axis.set_title("Implicit-score checkpoint selection")
            axis.grid(alpha=0.25)
            if groups:
                axis.legend()
            figure.tight_layout()
            path = run_dir / "score_learning_curves.png"
            temporary = path.with_suffix(".tmp.png")
            figure.savefig(temporary, dpi=160)
            plt.close(figure)
            os.replace(temporary, path)
            plots.append(str(path.resolve()))
        if time_rows:
            figure, axis = plt.subplots(figsize=(7.2, 4.6))
            for seed in sorted({int(row["model_seed"]) for row in time_rows}):
                points = [row for row in time_rows if int(row["model_seed"]) == seed and str(row["audit_bank"]) == "audit_a"]
                points.sort(key=lambda row: int(row["time_bin"]))
                axis.plot([0.1 + 0.2 * int(row["time_bin"]) for row in points], [float(row["score_risk_delta_vs_linear"]) for row in points], marker="o", label=str(seed))
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xlabel("reverse-time fraction tau/T")
            axis.set_ylabel("audit risk improvement vs linear")
            axis.set_title("Positive-time score signal by time bin")
            axis.grid(alpha=0.25)
            axis.legend(title="model seed")
            figure.tight_layout()
            path = run_dir / "score_gain_by_time.png"
            temporary = path.with_suffix(".tmp.png")
            figure.savefig(temporary, dpi=160)
            plt.close(figure)
            os.replace(temporary, path)
            plots.append(str(path.resolve()))
    except (ImportError, OSError, ValueError):
        pass
    return {
        "checkpoint_metrics": str((run_dir / "checkpoint_metrics.csv").resolve()),
        "audit_path_score_risks": str((run_dir / "audit_path_score_risks.csv").resolve()),
        "stein_path_metrics": str((run_dir / "stein_path_metrics.csv").resolve()),
        "score_time_bins": str((run_dir / "score_time_bins.csv").resolve()),
        "cross_seed_flux_cosines": str((run_dir / "cross_seed_flux_cosines.csv").resolve()),
        "score_seed_metrics": str((run_dir / "score_seed_metrics.csv").resolve()),
        "plots": plots,
    }


def _run_stages(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    device: torch.device,
    dynamics: DirectFluxMNISTConfig,
    source: Mapping[str, Any],
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    scientific_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint_value: str,
    confirmation: Mapping[str, Any],
) -> int:
    """Execute the fail-closed preflight, cache, controls, and score stages."""

    show_progress = not bool(args.no_progress)
    stage = str(args.stage)
    skips: list[dict[str, Any]] = []
    audit_ids = list(range(1_000_000, 1_000_000 + int(args.fresh_audit_paths)))
    preflight_ids = list(range(2_000_000, 2_000_000 + int(args.preflight_paths)))
    split_path = run_dir / "path_split.json"
    expected_split = _score_split_value(
        confirmation, audit_ids=audit_ids, preflight_ids=preflight_ids
    )
    if split_path.is_file():
        split = _verify_split_artifact(split_path, expected_split)
    elif stage == "report":
        raise FileNotFoundError(split_path)
    else:
        split = _make_split_artifact(run_dir, confirmation, audit_ids=audit_ids, preflight_ids=preflight_ids)
    probe_plan = _load_or_create_probe_plan(
        run_dir, args, scientific_fingerprint=scientific_fingerprint,
        read_only=stage == "report",
    )

    source_images = np.asarray(dataset_images, dtype=np.float64)
    source_labels = np.asarray(dataset_labels, dtype=np.int64)
    d0_config = _make_d0_config(args, seed=int(args.fresh_cache_seed))
    cache_owner = Path(args.cache_run_dir) if args.cache_run_dir is not None else run_dir
    if args.cache_run_dir is not None:
        owner_manifest = _json_load(cache_owner / "run_manifest.json")
        if owner_manifest.get("scientific_fingerprint") != scientific_fingerprint:
            raise ArtifactCompatibilityError("external score cache scientific fingerprint mismatch")
        if owner_manifest.get("source_fingerprint") != source_fingerprint_value:
            raise ArtifactCompatibilityError("external score cache source fingerprint mismatch")
        if owner_manifest.get("runtime_fingerprint") != runtime_fingerprint:
            raise ArtifactCompatibilityError("external score cache runtime fingerprint mismatch")
    preflight_root = cache_owner / "cache" / "preflight"
    preflight_index_path = preflight_root / "cache_index.json"
    _write_status(run_dir, phase="preflight", status="running")
    if args.cache_run_dir is not None or stage == "report":
        if not preflight_index_path.is_file():
            raise FileNotFoundError(preflight_index_path)
    else:
        build_fresh_score_state_shards(
            dataset_images=source_images, dataset_labels=source_labels,
            dynamics_config=dynamics, d0_config=d0_config, output_dir=preflight_root,
            path_ids=preflight_ids, role="preflight", device=device,
            seed=int(args.fresh_cache_seed) + 1000, anchor_seed=int(args.fresh_anchor_seed) + 1000,
            scientific_fingerprint=scientific_fingerprint,
            anchors_per_path=int(args.anchors_per_path), bin_counts=args.anchor_bin_counts,
            minimum_forward_substep=int(args.minimum_forward_substep),
            shard_paths=int(args.cache_shard_paths), resume=True,
            metadata={"claim_scope": CLAIM_SCOPE}, provenance={"parent_audit_reused": 0},
            show_progress=show_progress,
        )
    preflight_index, preflight_cache, preflight_arrays = _load_role_arrays(preflight_index_path, "preflight")
    preflight_health = _cache_health(preflight_cache, args)
    preflight_identity = _preflight_cache_identity(
        index=preflight_index, cache=preflight_cache,
        expected_path_ids=preflight_ids,
        scientific_fingerprint=scientific_fingerprint, args=args,
    )
    operator_report_path = run_dir / "operator_preflight.json"
    if stage == "report":
        if not operator_report_path.is_file():
            raise FileNotFoundError(operator_report_path)
        operator_report = _json_load(operator_report_path)
    else:
        operator_report = _run_operator_and_device_preflight(
            run_dir=run_dir, device=device, dynamics=dynamics, arrays=preflight_arrays, args=args
        )
    preflight_gate = {
        "gate": "preflight", "passed": int(bool(operator_report.get("passed")) and bool(preflight_health["passed"]) and bool(preflight_identity["passed"])),
        "operator": operator_report, "cache_health": preflight_health,
        "cache_identity": preflight_identity,
        "probe_plan_fingerprint": probe_plan["fingerprint"],
        "preflight_paths": int(preflight_cache.path_count), "sampling_performed": 0,
    }
    atomic_write_json(run_dir / "preflight_gate.json", preflight_gate)
    if stage == "preflight" or not bool(preflight_gate["passed"]):
        if not bool(preflight_gate["passed"]):
            skips.append({"stage": "cache", "reason": "operator/device or fresh-state preflight failed"})
        report = _write_gate_report(
            run_dir, args=args, preflight=preflight_gate,
            cache=_empty_gate("cache", "not reached"), controls=_empty_gate("controls", "not reached"),
            seed_results=[], cosines=[],
        )
        _write_report_tables_and_plots(run_dir, seed_results=[], cosines=[])
        return _finish(run_dir, report, phase="preflight", skips=skips)

    parent_root = cache_owner / "cache" / "parent"
    audit_root = cache_owner / "cache" / "audit"
    parent_index_path = parent_root / "cache_index.json"
    audit_index_path = audit_root / "cache_index.json"
    _write_status(run_dir, phase="cache", status="running")
    if args.cache_run_dir is not None or stage == "report":
        if not parent_index_path.is_file():
            raise FileNotFoundError(parent_index_path)
    else:
        materialize_parent_score_state_shards(
            confirmation["cache_index_path"], confirmation["path_split_path"], parent_root,
            scientific_fingerprint=scientific_fingerprint, roles=("train", "selection"),
            shard_paths=int(args.cache_shard_paths), resume=True,
            metadata={"claim_scope": CLAIM_SCOPE, "excluded_parent_audit_path_ids": list(map(int, confirmation["excluded_audit_path_ids"]))},
        )
    if args.cache_run_dir is not None or stage == "report":
        if not audit_index_path.is_file():
            raise FileNotFoundError(audit_index_path)
    else:
        build_fresh_score_state_shards(
            dataset_images=source_images, dataset_labels=source_labels,
            dynamics_config=dynamics, d0_config=d0_config, output_dir=audit_root,
            path_ids=audit_ids, role="audit", device=device,
            seed=int(args.fresh_cache_seed), anchor_seed=int(args.fresh_anchor_seed),
            scientific_fingerprint=scientific_fingerprint,
            anchors_per_path=int(args.anchors_per_path), bin_counts=args.anchor_bin_counts,
            minimum_forward_substep=int(args.minimum_forward_substep),
            shard_paths=int(args.cache_shard_paths), resume=True,
            metadata={"claim_scope": CLAIM_SCOPE},
            provenance={"parent_audit_reused": 0, "excluded_parent_audit_path_ids": list(map(int, confirmation["excluded_audit_path_ids"]))},
            show_progress=show_progress,
        )
    parent_index, train_cache, train_arrays = _load_role_arrays(parent_index_path, "train")
    _, selection_cache, selection_arrays = _load_role_arrays(parent_index_path, "selection")
    audit_index, audit_cache, audit_arrays = _load_role_arrays(audit_index_path, "audit")
    cache_gate = _cache_gate_report(
        args=args, parent_index=parent_index, train_cache=train_cache,
        selection_cache=selection_cache, audit_index=audit_index, audit_cache=audit_cache,
        confirmation=confirmation, split=split,
        scientific_fingerprint=scientific_fingerprint,
    )
    atomic_write_json(run_dir / "cache_gate.json", cache_gate)
    cache_fingerprint = config_fingerprint({"parent": parent_index.fingerprint, "audit": audit_index.fingerprint, "split": split["fingerprint"]})
    atomic_write_json(
        run_dir / "cache_provenance.json",
        {"cache_owner": str(cache_owner.resolve()), "external": int(args.cache_run_dir is not None), "parent_index": str(parent_index_path.resolve()), "audit_index": str(audit_index_path.resolve()), "cache_fingerprint": cache_fingerprint, "sampling_performed": 0},
    )
    if stage == "cache" or not bool(cache_gate["passed"]):
        if not bool(cache_gate["passed"]):
            skips.append({"stage": "controls", "reason": "positive-time cache gate failed"})
        report = _write_gate_report(
            run_dir, args=args, preflight=preflight_gate, cache=cache_gate,
            controls=_empty_gate("controls", "not reached"), seed_results=[], cosines=[],
        )
        _write_report_tables_and_plots(run_dir, seed_results=[], cosines=[])
        return _finish(run_dir, report, phase="cache", skips=skips)

    controls_path = run_dir / "controls_gate.json"
    _write_status(run_dir, phase="controls", status="running")
    controls_gate: dict[str, Any] | None = None
    if controls_path.is_file():
        existing_controls_gate = _validate_completed_controls_gate(
            run_dir=run_dir,
            controls_gate=_json_load(controls_path),
            args=args,
            scientific_fingerprint=scientific_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            source_fingerprint_value=source_fingerprint_value,
            cache_fingerprint=cache_fingerprint,
            operator_gate=operator_report,
            require_evidence=False,
        )
        if int(existing_controls_gate.get("passed", 0)) == 1:
            controls_gate = _validate_completed_controls_gate(
                run_dir=run_dir,
                controls_gate=existing_controls_gate,
                args=args,
                scientific_fingerprint=scientific_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
                source_fingerprint_value=source_fingerprint_value,
                cache_fingerprint=cache_fingerprint,
                operator_gate=operator_report,
                require_evidence=True,
            )
        elif stage == "report":
            # Report is strictly read-only. A failed gate is reportable, but a
            # mutating stage retries its exact task checkpoints below. If both
            # tasks were committed, report validates their metrics too.
            try:
                _control_evidence_records(run_dir)
            except FileNotFoundError:
                controls_gate = existing_controls_gate
            else:
                controls_gate = _validate_completed_controls_gate(
                    run_dir=run_dir,
                    controls_gate=existing_controls_gate,
                    args=args,
                    scientific_fingerprint=scientific_fingerprint,
                    runtime_fingerprint=runtime_fingerprint,
                    source_fingerprint_value=source_fingerprint_value,
                    cache_fingerprint=cache_fingerprint,
                    operator_gate=operator_report,
                    require_evidence=True,
                )
    elif stage == "report":
        raise FileNotFoundError(controls_path)
    if controls_gate is None:
        try:
            controls_gate = _run_controls(
                run_dir=run_dir, train=train_arrays, selection=selection_arrays,
                audit=audit_arrays, dynamics=dynamics, args=args, device=device,
                scientific_fingerprint=scientific_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
                source_fingerprint_value=source_fingerprint_value,
                cache_fingerprint=cache_fingerprint, operator_gate=operator_report,
                show_progress=show_progress,
            )
        except ArtifactCompatibilityError:
            raise
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            teacher_artifact = run_dir / "controls" / "positive_teacher.json"
            teacher_failure = (
                dict(_json_load(teacher_artifact).get("gate", {}))
                if teacher_artifact.is_file()
                else _empty_gate("positive_teacher", f"{type(exc).__name__}: {exc}")
            )
            null_failure = _empty_gate("null_control", "not reached after control task failure")
            controls_gate = {
                "schema": RUN_SCHEMA + "-controls-gate",
                "schema_version": 1,
                "binding": {
                    "scientific_fingerprint": scientific_fingerprint,
                    "runtime_fingerprint": runtime_fingerprint,
                    "source_fingerprint": source_fingerprint_value,
                    "cache_fingerprint": cache_fingerprint,
                    "control_split_fingerprint": (
                        _json_load(run_dir / "control_path_split.json").get("fingerprint")
                        if (run_dir / "control_path_split.json").is_file()
                        else None
                    ),
                },
                "teacher": teacher_failure,
                "null": null_failure,
                **evaluate_control_bundle(
                    operator_gate=operator_report,
                    positive_teacher_gate=teacher_failure,
                    null_control_gate=null_failure,
                ),
                "failure": {"type": type(exc).__name__, "reason": str(exc)},
                "sampling_performed": 0,
            }
            atomic_write_json(run_dir / "controls_gate.json", controls_gate)
    assert controls_gate is not None
    if stage == "controls" or not bool(int(controls_gate.get("passed", 0))):
        if not bool(int(controls_gate.get("passed", 0))):
            skips.append({"stage": "train", "reason": "synthetic optimization controls failed"})
        report = _write_gate_report(
            run_dir, args=args, preflight=preflight_gate, cache=cache_gate,
            controls=controls_gate, seed_results=[], cosines=[],
        )
        _write_report_tables_and_plots(run_dir, seed_results=[], cosines=[])
        return _finish(run_dir, report, phase="controls", skips=skips)

    _write_status(run_dir, phase="physical-training", status="running")
    baseline, baseline_fit = _load_or_fit_baseline(
        run_dir=run_dir, train=train_arrays, dynamics=dynamics,
        scientific_fingerprint=scientific_fingerprint, cache_fingerprint=cache_fingerprint,
        device=device, name="physical", read_only=stage == "report",
    )
    baseline_path = run_dir / "baselines" / "physical_linear_spline.pt"
    witness_plan = _load_or_create_witness_plan(
        run_dir=run_dir, train=train_arrays, dynamics=dynamics, args=args,
        scientific_fingerprint=scientific_fingerprint,
        cache_fingerprint=cache_fingerprint, read_only=stage == "report",
    )
    witness_fingerprint = str(witness_plan["metadata"]["fingerprint"])
    seed_results: list[dict[str, Any]] = []
    seed_fluxes: dict[int, np.ndarray] = {}
    failures: list[dict[str, Any]] = []
    for seed in args.training_seeds:
        task_dir = run_dir / "tasks" / f"seed-{int(seed)}"
        task_fp = _task_checkpoint_fingerprints(
            manifest_fingerprint=scientific_fingerprint, runtime_fingerprint=runtime_fingerprint,
            source_fingerprint_value=source_fingerprint_value, cache_fingerprint=cache_fingerprint,
            baseline_path=baseline_path, seed=int(seed), train_steps=int(args.train_steps),
            witness_plan_fingerprint=witness_fingerprint,
        )
        result_path = task_dir / "task_result.json"
        flux_path = task_dir / "audit_nonlinear_flux.npz"
        audit_intent_path = task_dir / "audit_intent.json"
        try:
            task_status_path = task_dir / "task_status.json"
            completed_task = (
                task_status_path.is_file()
                and _json_load(task_status_path).get("status") == "complete"
            )
            if completed_task:
                try:
                    if stage != "report":
                        _repair_best_checkpoint_publication(
                            task_dir=task_dir, fingerprints=task_fp
                        )
                    result, flux = _load_completed_physical_task(
                        task_dir=task_dir, fingerprints=task_fp, audit=audit_arrays,
                        model_seed=int(seed),
                    )
                except FileNotFoundError as exc:
                    raise ArtifactCompatibilityError(
                        f"completed score task is missing {exc.filename or exc}"
                    ) from exc
            elif result_path.is_file() or flux_path.is_file():
                if not result_path.is_file() or not flux_path.is_file() or not task_status_path.is_file():
                    raise ArtifactCompatibilityError(
                        "interrupted score audit has only a partial committed artifact set"
                    )
                if stage == "report":
                    raise ArtifactCompatibilityError(
                        "report mode refuses to finalize an interrupted score audit"
                    )
                _repair_best_checkpoint_publication(
                    task_dir=task_dir, fingerprints=task_fp
                )
                result, flux = _load_completed_physical_task(
                    task_dir=task_dir, fingerprints=task_fp, audit=audit_arrays,
                    model_seed=int(seed), require_complete_status=False,
                )
                atomic_write_json(
                    task_status_path,
                    {
                        "schema": TASK_STATUS_SCHEMA,
                        "schema_version": TASK_STATUS_SCHEMA_VERSION,
                        "status": "complete",
                        "training_seed": int(seed),
                        "selected_step": int(result["selected_step"]),
                        "fingerprints": task_fp,
                        "task_result_sha256": file_fingerprint(result_path),
                        "flux_sha256": file_fingerprint(flux_path),
                        "recovered_after_audit_commit": 1,
                        "sampling_performed": 0,
                    },
                )
            elif audit_intent_path.is_file():
                _validate_physical_audit_intent(
                    task_dir=task_dir,
                    fingerprints=task_fp,
                    audit=audit_arrays,
                    model_seed=int(seed),
                )
                raise ArtifactCompatibilityError(
                    "physical audit was started without a complete committed result; "
                    "refusing to access the untouched audit split twice"
                )
            elif stage == "report":
                raise FileNotFoundError(f"missing completed score task for seed {seed}")
            else:
                model, training_summary = _train_potential_task(
                    task_dir=task_dir, train=train_arrays, selection=selection_arrays,
                    baseline=baseline, dynamics=dynamics, device=device, args=args,
                    training_seed=int(seed), train_steps=int(args.train_steps),
                    fingerprints=task_fp, show_progress=show_progress,
                )
                baseline_device, train_summary = _prepare_physical_task_evaluation(
                    model=model,
                    baseline=baseline,
                    train=train_arrays,
                    dynamics=dynamics,
                    args=args,
                    device=device,
                )
                _write_physical_audit_intent(
                    task_dir=task_dir,
                    fingerprints=task_fp,
                    audit=audit_arrays,
                    model_seed=int(seed),
                    training_summary=training_summary,
                )
                result, flux, _ = _evaluate_physical_task(
                    model=model,
                    baseline_device=baseline_device,
                    train_summary=train_summary,
                    audit=audit_arrays,
                    selection_summary=training_summary, dynamics=dynamics, args=args,
                    device=device, model_seed=int(seed), witness_plan=witness_plan,
                )
                result["fingerprints"] = task_fp
                result["baseline_fit"] = baseline_fit
                atomic_write_json(result_path, result)
                _atomic_save_npz(
                    flux_path, flux=np.asarray(flux, dtype=np.float32),
                    path_ids=np.asarray(audit_arrays.path_ids, dtype=np.int64),
                    end_substeps=np.asarray(audit_arrays.end_substeps, dtype=np.int64),
                )
                atomic_write_json(task_dir / "task_status.json", {"schema": TASK_STATUS_SCHEMA, "schema_version": TASK_STATUS_SCHEMA_VERSION, "status": "complete", "training_seed": int(seed), "selected_step": int(result["selected_step"]), "fingerprints": task_fp, "task_result_sha256": file_fingerprint(result_path), "flux_sha256": file_fingerprint(flux_path), "sampling_performed": 0})
            seed_results.append(dict(result))
            if np.isfinite(flux).all():
                seed_fluxes[int(seed)] = np.asarray(flux)
        except ArtifactCompatibilityError:
            raise
        except (RuntimeError, ValueError, FloatingPointError, FileNotFoundError) as exc:
            if stage == "report":
                raise
            failures.append({"model_seed": int(seed), "type": type(exc).__name__, "reason": str(exc)})
            atomic_write_json(task_dir / "task_status.json", {"schema": TASK_STATUS_SCHEMA, "schema_version": TASK_STATUS_SCHEMA_VERSION, "status": "failed", "training_seed": int(seed), "reason": str(exc), "fingerprints": task_fp, "sampling_performed": 0})
    atomic_write_json(run_dir / "task_failures.json", {"failure_count": len(failures), "failures": failures, "sampling_performed": 0})
    cosines = _cross_seed_flux_cosines(seed_fluxes, audit_arrays) if len(seed_fluxes) == len(args.training_seeds) else []
    artifacts = _write_report_tables_and_plots(run_dir, seed_results=seed_results, cosines=cosines)
    _update_manifest_artifacts(run_dir, artifacts)
    report = _write_gate_report(
        run_dir, args=args, preflight=preflight_gate, cache=cache_gate,
        controls=controls_gate, seed_results=seed_results, cosines=cosines,
    )
    return _finish(run_dir, report, phase="report", skips=skips)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
