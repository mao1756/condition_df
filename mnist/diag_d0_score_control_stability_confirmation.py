"""Streamed stability confirmation for boundary-admissible D0 score controls.

This workflow is intentionally controls-only.  It binds the completed
optimizer-scale repair run, verifies exact Stein identities, selects a stable
learning-rate profile on fresh streamed synthetic states, and confirms that
profile with fresh selection and audit panels.  It never trains on physical
score states and never imports or invokes a sampler.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

import mnist.diag_d0_score_boundary_controls as boundary
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_score_boundary_control_gate import BoundaryControlThresholds
from mnist.d0_score_control_stability import (
    STREAM_SCHEMA,
    STREAM_DERIVATION_VERSION,
    StreamPlan,
    build_stream_plan,
    generate_stream_batch,
    run_stein_identity_preflight,
    stateless_probe_banks,
    stream_plan_record,
    stream_replay_record,
    verify_stream_replay,
)
from mnist.d0_score_control_stability_gate import (
    ProbeBankStatus,
    StabilityThresholds,
    evaluate_stein_identity_preflight,
    evaluate_stability_confirmation,
    evaluate_stability_pilot,
    evaluate_stability_workflow,
    select_stability_profile,
)
from mnist.d0_score_control_stability_provenance import (
    verify_parent_scale_repair_run,
)


RUN_SCHEMA = "experiment12-d0-score-control-stability-confirmation"
RUN_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
CLAIM_SCOPE = "streamed boundary-admissible synthetic implicit-score controls only"
FROZEN_IMPLICIT_LOSS_SCALE = 0.00266397028560976

DEFAULTS: dict[str, Any] = {
    "grid_size": 28,
    "sample_steps": 512,
    "reference_substeps": 256,
    "tau_eff": 5e-5,
    "edge_alpha_mode": "alpha_eff",
    "alpha_eff": 1.0,
    "mass_floor": 1e-7,
    "limiter_fraction": 1.0,
    "lambda_mix": 0.35,
    "root_seed": 260801,
    "pilot_learning_rates": (1e-4, 3e-5, 1e-5, 3e-6),
    "pilot_steps": 1000,
    "confirm_steps": 4000,
    "pilot_selection_paths": 16,
    "confirm_selection_paths": 32,
    "confirm_audit_paths": 32,
    "anchors_per_path": 32,
    "anchor_bin_counts": (4, 4, 4, 4, 16),
    "clusters_per_step": 2,
    "anchors_per_cluster": 32,
    "base_channels": 32,
    "batch_size": 64,
    "validation_batch_size": 64,
    "weight_decay": 1e-4,
    "ema_decay": 0.99,
    "grad_clip": 1.0,
    "clip_warmup_steps": 500,
    "training_probe_banks": 2,
    "training_probes_per_bank": 4,
    "selection_probes": 16,
    "audit_probes": 64,
    "bootstrap_reps": 10_000,
    "bootstrap_confidence": 0.90,
    "stein_paths": 128,
    "stein_bootstrap_confidence": 0.99,
    "stein_amplitudes": (0.0, 0.25, 0.5, 1.0, 2.0),
    "confirm_model_seeds": (260811, 260812, 260813),
    "implicit_loss_scale": FROZEN_IMPLICIT_LOSS_SCALE,
    "pilot_validation_steps": (0, 25, 50, 100, 150, 250, 500, 750, 1000),
    "confirm_dense_validation_steps": (0, 25, 50, 100, 150, 250),
    "confirm_validation_every": 250,
}


def _json_load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("expected finite comma-separated numbers")
    return result


def _semantic_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        try:
            actual_values = tuple(actual)
        except TypeError:
            return False
        return len(actual_values) == len(expected) and all(
            _semantic_equal(left, right) for left, right in zip(actual_values, expected)
        )
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "gate": str(name),
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "reason": str(reason),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def classify_probe_bank_status(
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
) -> ProbeBankStatus:
    if not teacher_results or not null_results:
        return ProbeBankStatus.NOT_EVALUATED
    complete = all(
        set(dict(dict(result.get("metrics", {})).get("audit_objective_banks", {})))
        == {"a", "b"}
        for result in [*teacher_results, *null_results]
    )
    if not complete:
        return ProbeBankStatus.NOT_EVALUATED
    return (
        ProbeBankStatus.AGREE
        if boundary._probe_banks_agree(
            teacher_results=teacher_results, null_results=null_results
        )
        else ProbeBankStatus.DISAGREE
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "pilot", "confirm", "report", "all"), default="all")
    parser.add_argument("--parent-scale-repair-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_d0_score_control_stability_confirmation"))
    parser.add_argument("--run-name", default="production-streamed-implicit-controls")
    parser.add_argument("--require-gate", choices=("none", "preflight", "pilot", "controls"), default="none")
    parser.add_argument("--root-seed", type=int, default=DEFAULTS["root_seed"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--grid-size", type=int, default=DEFAULTS["grid_size"])
    parser.add_argument("--sample-steps", type=int, default=DEFAULTS["sample_steps"])
    parser.add_argument("--reference-substeps", type=int, default=DEFAULTS["reference_substeps"])
    parser.add_argument("--tau-eff", type=float, default=DEFAULTS["tau_eff"])
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff",), default=DEFAULTS["edge_alpha_mode"])
    parser.add_argument("--alpha-eff", type=float, default=DEFAULTS["alpha_eff"])
    parser.add_argument("--mass-floor", type=float, default=DEFAULTS["mass_floor"])
    parser.add_argument("--limiter-fraction", type=float, default=DEFAULTS["limiter_fraction"])
    parser.add_argument("--lambda-mix", type=float, default=DEFAULTS["lambda_mix"])

    parser.add_argument("--pilot-learning-rates", type=_parse_csv_floats, default=DEFAULTS["pilot_learning_rates"])
    parser.add_argument("--pilot-steps", type=int, default=DEFAULTS["pilot_steps"])
    parser.add_argument("--confirm-steps", type=int, default=DEFAULTS["confirm_steps"])
    parser.add_argument("--pilot-selection-paths", type=int, default=DEFAULTS["pilot_selection_paths"])
    parser.add_argument("--confirm-selection-paths", type=int, default=DEFAULTS["confirm_selection_paths"])
    parser.add_argument("--confirm-audit-paths", type=int, default=DEFAULTS["confirm_audit_paths"])
    parser.add_argument("--anchors-per-path", type=int, default=DEFAULTS["anchors_per_path"])
    parser.add_argument("--anchor-bin-counts", type=_parse_csv_ints, default=DEFAULTS["anchor_bin_counts"])
    parser.add_argument("--clusters-per-step", type=int, default=DEFAULTS["clusters_per_step"])
    parser.add_argument("--anchors-per-cluster", type=int, default=DEFAULTS["anchors_per_cluster"])
    parser.add_argument("--base-channels", type=int, default=DEFAULTS["base_channels"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--validation-batch-size", type=int, default=DEFAULTS["validation_batch_size"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--ema-decay", type=float, default=DEFAULTS["ema_decay"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"])
    parser.add_argument("--clip-warmup-steps", type=int, default=DEFAULTS["clip_warmup_steps"])
    parser.add_argument("--training-probe-banks", type=int, default=DEFAULTS["training_probe_banks"])
    parser.add_argument("--training-probes-per-bank", type=int, default=DEFAULTS["training_probes_per_bank"])
    parser.add_argument("--selection-probes", type=int, default=DEFAULTS["selection_probes"])
    parser.add_argument("--audit-probes", type=int, default=DEFAULTS["audit_probes"])
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULTS["bootstrap_reps"])
    parser.add_argument("--bootstrap-confidence", type=float, default=DEFAULTS["bootstrap_confidence"])
    parser.add_argument("--stein-paths", type=int, default=DEFAULTS["stein_paths"])
    parser.add_argument("--stein-bootstrap-confidence", type=float, default=DEFAULTS["stein_bootstrap_confidence"])
    parser.add_argument("--stein-amplitudes", type=_parse_csv_floats, default=DEFAULTS["stein_amplitudes"])
    parser.add_argument("--confirm-model-seeds", type=_parse_csv_ints, default=DEFAULTS["confirm_model_seeds"])
    parser.add_argument("--implicit-loss-scale", type=float, default=DEFAULTS["implicit_loss_scale"])
    parser.add_argument("--pilot-validation-steps", type=_parse_csv_ints, default=DEFAULTS["pilot_validation_steps"])
    parser.add_argument("--confirm-dense-validation-steps", type=_parse_csv_ints, default=DEFAULTS["confirm_dense_validation_steps"])
    parser.add_argument("--confirm-validation-every", type=int, default=DEFAULTS["confirm_validation_every"])
    args = parser.parse_args(argv)

    positive = (
        "grid_size", "sample_steps", "reference_substeps", "pilot_steps", "confirm_steps",
        "pilot_selection_paths", "confirm_selection_paths", "confirm_audit_paths",
        "anchors_per_path", "clusters_per_step", "anchors_per_cluster", "base_channels",
        "batch_size", "validation_batch_size", "training_probe_banks",
        "training_probes_per_bank", "selection_probes", "audit_probes", "bootstrap_reps",
        "stein_paths", "confirm_validation_every",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if len(args.anchor_bin_counts) != 5 or sum(args.anchor_bin_counts) != int(args.anchors_per_path):
        parser.error("--anchor-bin-counts must have five entries summing to --anchors-per-path")
    derived_anchors_per_cluster = sum(int(value) for value in args.anchor_bin_counts)
    if int(args.anchors_per_cluster) != derived_anchors_per_cluster:
        parser.error("--anchors-per-cluster must equal the sum of --anchor-bin-counts")
    derived_batch_size = int(args.clusters_per_step) * derived_anchors_per_cluster
    if int(args.batch_size) != derived_batch_size:
        parser.error(
            "--batch-size must equal --clusters-per-step times the sum of "
            "--anchor-bin-counts"
        )
    if int(args.training_probe_banks) != 2:
        parser.error("--training-probe-banks must equal 2 (independent banks A and B)")
    if args.anchor_bin_counts != (4, 4, 4, 4, 16):
        parser.error("streamed training currently requires exact time strata 4,4,4,4,16")
    if len(args.confirm_model_seeds) != 3 or len(set(args.confirm_model_seeds)) != 3:
        parser.error("--confirm-model-seeds must contain three distinct seeds")
    if any(value <= 0.0 for value in args.pilot_learning_rates):
        parser.error("pilot learning rates must be positive")
    if tuple(sorted(set(args.pilot_validation_steps))) != tuple(args.pilot_validation_steps):
        parser.error("pilot validation steps must be sorted and unique")
    if args.pilot_validation_steps[0] != 0 or args.pilot_validation_steps[-1] != args.pilot_steps:
        parser.error("pilot validation steps must include zero and pilot-steps")
    if args.confirm_dense_validation_steps[0] != 0:
        parser.error("confirm dense validation steps must begin at zero")
    if not 0.0 < float(args.ema_decay) < 1.0:
        parser.error("--ema-decay must lie in (0,1)")
    for name in ("bootstrap_confidence", "stein_bootstrap_confidence"):
        if not 0.0 < float(getattr(args, name)) < 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in (0,1)")
    if args.stage == "confirm" and args.resume_run_dir is None:
        parser.error("--stage confirm requires --resume-run-dir from a passing pilot")
    if args.stage == "report" and args.resume_run_dir is None:
        parser.error("--stage report requires --resume-run-dir")
    if args.require_gate != "none":
        mismatches = [
            f"{key}={getattr(args, key)!r}, expected {expected!r}"
            for key, expected in DEFAULTS.items()
            if hasattr(args, key) and not _semantic_equal(getattr(args, key), expected)
        ]
        if mismatches:
            parser.error("required production gate rejects overrides: " + "; ".join(mismatches))
    return args


def _source_record() -> tuple[str, list[str]]:
    here = Path(__file__).resolve()
    names = (
        here.name,
        "d0_score_control_stability.py",
        "d0_score_control_stability_gate.py",
        "d0_score_control_stability_provenance.py",
        "diag_d0_score_boundary_controls.py",
        "d0_score_boundary_controls.py",
        "d0_score_boundary_control_gate.py",
        "d0_score_control_scale_repair_gate.py",
        "d0_score_optimizer_scale.py",
        "d0_dirichlet_score.py",
        "d0_one_image_gate.py",
        "eulerian_flux_mnist.py",
    )
    paths = [here.with_name(name) for name in names]
    existing = [path for path in paths if path.is_file()]
    return source_fingerprint(existing), [str(path) for path in existing]


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        path = args.resume_run_dir.resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path, True
    root = args.runs_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in str(args.run_name)).strip("-")
    path = root / f"{timestamp}_{safe_name or 'run'}"
    suffix = 1
    while path.exists():
        path = root / f"{timestamp}_{safe_name or 'run'}-{suffix}"
        suffix += 1
    path.mkdir(parents=False)
    return path, False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    current = _json_load(path) if path.is_file() else {}
    current.update(updates)
    current.setdefault("schema", RUN_SCHEMA)
    current.setdefault("schema_version", RUN_SCHEMA_VERSION)
    current["physical_training_performed"] = 0
    current["sampling_performed"] = 0
    current["updated_at"] = _now()
    atomic_write_json(path, current)
    return current


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path), "size": int(path.stat().st_size)
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "records": records,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _verify_terminal_registry(run_dir: Path) -> dict[str, Any]:
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    if not registry_path.is_file() or not status_path.is_file():
        raise ArtifactCompatibilityError("terminal run is missing status or artifact registry")
    status = _json_load(status_path)
    if (
        status.get("artifact_registry_sha256") != file_fingerprint(registry_path)
        or int(status.get("artifact_registry_size", -1)) != int(registry_path.stat().st_size)
    ):
        raise ArtifactCompatibilityError("terminal status does not bind artifact registry")
    registry = _json_load(registry_path)
    if registry.get("schema") != RUN_SCHEMA + "-artifact-registry":
        raise ArtifactCompatibilityError("artifact registry schema mismatch")
    excluded = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    records = dict(registry.get("records", {}))
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    if set(records) != actual:
        raise ArtifactCompatibilityError("artifact registry file set mismatch")
    for relative, raw_record in records.items():
        path = run_dir / relative
        record = dict(raw_record)
        if (
            not path.is_file()
            or record.get("sha256") != file_fingerprint(path)
            or int(record.get("size", -1)) != int(path.stat().st_size)
        ):
            raise ArtifactCompatibilityError(f"artifact registry hash mismatch: {relative}")
    return registry


def _scientific_config(args: argparse.Namespace, parent: Mapping[str, Any], thresholds: StabilityThresholds) -> dict[str, Any]:
    anchors_per_cluster = sum(int(value) for value in args.anchor_bin_counts)
    batch_size = int(args.clusters_per_step) * anchors_per_cluster
    value = {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "stream_schema": STREAM_SCHEMA,
        "model_schema": boundary.MODEL_SCHEMA,
        "model_schema_version": boundary.MODEL_SCHEMA_VERSION,
        "kernel": {key: getattr(args, key) for key in boundary.EXPECTED_KERNEL},
        "root_seed": int(args.root_seed),
        "stream": {
            "clusters_per_step": int(args.clusters_per_step),
            "anchors_per_cluster": anchors_per_cluster,
            "batch_size": batch_size,
            "anchor_bin_counts": list(args.anchor_bin_counts),
            "training_probe_banks": int(args.training_probe_banks),
            "training_probes_per_bank": int(args.training_probes_per_bank),
            "stateless_seed_tuple": ["root_seed", "phase", "law", "step", "bank"],
        },
        "pilot": {
            "learning_rates": list(args.pilot_learning_rates),
            "steps": int(args.pilot_steps),
            "selection_paths": int(args.pilot_selection_paths),
            "validation_steps": list(args.pilot_validation_steps),
            "audit_paths": 0,
        },
        "confirmation": {
            "steps": int(args.confirm_steps),
            "model_seeds": list(args.confirm_model_seeds),
            "selection_paths": int(args.confirm_selection_paths),
            "audit_paths": int(args.confirm_audit_paths),
            "dense_validation_steps": list(args.confirm_dense_validation_steps),
            "validation_every": int(args.confirm_validation_every),
        },
        "optimization": {
            "implicit_loss_scale": float(args.implicit_loss_scale),
            "loss_scale_source": "verified parent implicit calibration",
            "adaptive_loss_scaling": 0,
            "weight_decay": float(args.weight_decay),
            "ema_decay": float(args.ema_decay),
            "grad_clip": float(args.grad_clip),
            "clip_warmup_steps": int(args.clip_warmup_steps),
        },
        "bootstrap": {"reps": int(args.bootstrap_reps), "confidence": float(args.bootstrap_confidence)},
        "stein": {
            "paths": int(args.stein_paths),
            "confidence": float(args.stein_bootstrap_confidence),
            "amplitudes": list(args.stein_amplitudes),
        },
        "thresholds": thresholds.to_dict(),
        "parent_scientific_fingerprint": parent.get("scientific_fingerprint"),
        "parent_artifact_registry_sha256": parent.get("artifact_registry_sha256"),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    # Normalize tuple-valued dataclass fields to their persisted JSON shape so
    # an exact resume compares semantic bytes rather than Python container types.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _panel_binding(*, phase: str, law: str, role: str, args: argparse.Namespace, scientific_fingerprint: str) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-fixed-panel-binding",
        "schema_version": 1,
        "phase": phase,
        "law": law,
        "role": role,
        "root_seed": int(args.root_seed),
        "scientific_fingerprint": scientific_fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _prepare_fixed_panel(
    run_dir: Path,
    *,
    phase: str,
    law: str,
    role: str,
    path_count: int,
    first_path_id: int,
    seed: int,
    horizon: float,
    args: argparse.Namespace,
    scientific_fingerprint: str,
) -> boundary.ControlArrays:
    folder = run_dir / "panels" / phase
    path = folder / f"{law}-{role}.npz"
    binding = _panel_binding(
        phase=phase, law=law, role=role, args=args,
        scientific_fingerprint=scientific_fingerprint,
    )
    if path.is_file():
        arrays, _ = boundary._load_arrays(path, binding)
        return arrays
    arrays = boundary._build_control_arrays(
        role=f"{phase}_{role}", law=law, path_count=int(path_count),
        first_path_id=int(first_path_id), bin_counts=args.anchor_bin_counts,
        horizon=float(horizon), grid_size=int(args.grid_size), seed=int(seed),
    )
    boundary._save_arrays(path, arrays, binding)
    return arrays


def _task_fingerprints(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    law: str,
    model_seed: int,
    learning_rate: float,
    loss_scale: float,
    stream_plan: StreamPlan,
    selection: boundary.ControlArrays,
    audit: boundary.ControlArrays | None,
    selected_profile_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema": RUN_SCHEMA + "-task-fingerprints",
        "schema_version": 1,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "scientific_fingerprint": manifest["scientific_fingerprint"],
        "runtime_fingerprint": manifest["runtime_fingerprint"],
        "source_fingerprint": manifest["source_fingerprint"],
        "phase": phase,
        "law": law,
        "task_kind": "implicit_teacher" if law == "bounded_teacher" else "null",
        "model_seed": int(model_seed),
        "learning_rate": float(learning_rate),
        "loss_scale": float(loss_scale),
        "stream_plan_fingerprint": str(stream_plan.fingerprint),
        "selection_identity": boundary._arrays_identity(selection),
        "audit_identity": None if audit is None else boundary._arrays_identity(audit),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    if selected_profile_binding is not None:
        normalized_binding = json.loads(
            json.dumps(dict(selected_profile_binding), sort_keys=True, allow_nan=False)
        )
        value["selected_profile_binding"] = normalized_binding
        value["selected_profile_binding_fingerprint"] = config_fingerprint(
            normalized_binding
        )
    return value


def _stream_task_args(
    args: argparse.Namespace,
    *,
    phase: str,
    learning_rate: float,
) -> argparse.Namespace:
    values = vars(args).copy()
    values["learning_rate"] = float(learning_rate)
    values["train_steps"] = int(args.pilot_steps if phase == "pilot" else args.confirm_steps)
    values["validation_steps"] = (
        tuple(args.pilot_validation_steps)
        if phase == "pilot"
        else tuple(
            sorted(
                set(args.confirm_dense_validation_steps)
                | set(range(int(args.confirm_validation_every), int(args.confirm_steps) + 1, int(args.confirm_validation_every)))
                | {int(args.confirm_steps)}
            )
        )
    )
    values["checkpoint_steps"] = values["validation_steps"]
    values["training_probes"] = int(args.training_probes_per_bank)
    values["audit_probe_a_seed"] = boundary._derived_seed(int(args.root_seed), phase, "audit", "a")
    values["audit_probe_b_seed"] = boundary._derived_seed(int(args.root_seed), phase, "audit", "b")
    values["selection_probe_a_seed"] = boundary._derived_seed(int(args.root_seed), phase, "selection", "a")
    values["selection_probe_b_seed"] = boundary._derived_seed(int(args.root_seed), phase, "selection", "b")
    values["bootstrap_seed"] = boundary._derived_seed(int(args.root_seed), phase, "bootstrap")
    values["batch_index_seed"] = boundary._derived_seed(int(args.root_seed), phase, "batch-index-unused")
    return argparse.Namespace(**values)


def _stream_replay_registry(plan: StreamPlan, *, phase: str) -> dict[str, Any]:
    records = [
        stream_replay_record(plan, phase=phase, law=law, step=step)
        for law in ("bounded_teacher", "dirichlet_null")
        for step in (1, 17)
    ]
    return {
        "schema": RUN_SCHEMA + "-stream-replay-registry",
        "schema_version": 1,
        "phase": phase,
        "plan_fingerprint": plan.fingerprint,
        "records": records,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _freeze_selected_profile(path: Path, selected: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(selected), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _json_load(path) != normalized:
            raise ArtifactCompatibilityError(
                "frozen selected stability profile changed on resume"
            )
    else:
        atomic_write_json(path, normalized)
    return normalized


def _bind_confirmation_profile(
    run_dir: Path, selected: Mapping[str, Any]
) -> dict[str, Any]:
    selected_path = run_dir / "selected_stability_profile.json"
    normalized = json.loads(json.dumps(dict(selected), sort_keys=True, allow_nan=False))
    if not selected_path.is_file() or _json_load(selected_path) != normalized:
        raise ArtifactCompatibilityError(
            "confirmation selected profile differs from the frozen pilot selection"
        )
    pilot_gate_path = run_dir / "stability_pilot_gate.json"
    if not pilot_gate_path.is_file():
        raise ArtifactCompatibilityError("confirmation is missing its pilot gate")
    pilot_selected = dict(_json_load(pilot_gate_path).get("selected_profile", {}))
    if pilot_selected != normalized:
        raise ArtifactCompatibilityError(
            "frozen selected profile disagrees with the pilot gate"
        )
    record = {
        "schema": RUN_SCHEMA + "-confirmation-profile-binding",
        "schema_version": 1,
        "selected_profile": normalized,
        "selected_profile_sha256": file_fingerprint(selected_path),
        "pilot_gate_sha256": file_fingerprint(pilot_gate_path),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    binding_path = run_dir / "confirmation_profile_binding.json"
    if binding_path.is_file():
        if _json_load(binding_path) != record:
            raise ArtifactCompatibilityError(
                "confirmation profile binding changed during resume"
            )
    else:
        atomic_write_json(binding_path, record)
    return record


def _load_stream_checkpoint(
    path: Path,
    *,
    device: torch.device,
    task_kind: str,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older Torch
        value = torch.load(path, map_location=device)
    if (
        value.get("schema") != RUN_SCHEMA + "-stream-checkpoint"
        or int(value.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION
        or value.get("task_kind") != task_kind
        or dict(value.get("fingerprints", {})) != dict(fingerprints)
        or value.get("stream_derivation_version") != STREAM_DERIVATION_VERSION
    ):
        raise ArtifactCompatibilityError("legacy, foreign, or mismatched streamed checkpoint")
    required = {
        "step", "model_state_dict", "ema_state_dict", "optimizer_state_dict",
        "history", "validation_records", "checkpoint_selection", "rng_state",
        "stream_cursor", "stream_plan", "fingerprints",
    }
    if not required.issubset(value):
        raise ArtifactCompatibilityError("streamed checkpoint is incomplete")
    return value


def _save_stream_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_state: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    step: int,
    history: Sequence[Mapping[str, Any]],
    validations: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    stream_plan: StreamPlan,
    task_kind: str,
    rng: np.random.Generator,
) -> None:
    boundary.atomic_torch_save(
        path,
        {
            "schema": RUN_SCHEMA + "-stream-checkpoint",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_schema": boundary.MODEL_SCHEMA,
            "model_schema_version": boundary.MODEL_SCHEMA_VERSION,
            "task_kind": task_kind,
            "step": int(step),
            "stream_cursor": int(step),
            "stream_derivation_version": STREAM_DERIVATION_VERSION,
            "stream_plan": stream_plan_record(stream_plan),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "ema_state_dict": {
                key: value.detach().clone() for key, value in ema_state.items()
            },
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "history": [dict(value) for value in history],
            "validation_records": copy.deepcopy(list(validations)),
            "checkpoint_selection": copy.deepcopy(dict(selection)),
            "rng_state": boundary.capture_rng_state(rng),
            "fingerprints": dict(fingerprints),
            "scaler_state_dict": None,
            "amp": False,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )


def _selected_validation_banks(
    validations: Sequence[Mapping[str, Any]], selected_step: int
) -> dict[str, Any]:
    matches = [dict(value) for value in validations if int(value.get("step", -1)) == int(selected_step)]
    if len(matches) != 1:
        raise ArtifactCompatibilityError("selected validation record is missing or ambiguous")
    return copy.deepcopy(dict(matches[0].get("banks", {})))


def _clip_fraction(history: Sequence[Mapping[str, Any]], *, start: int, stop: int) -> float:
    rows = [
        value for value in history
        if int(start) <= int(value.get("step", -1)) <= int(stop)
    ]
    if not rows:
        rows = list(history)
    if not rows:
        return 0.0
    return float(np.mean([bool(int(value.get("clipped", 0))) for value in rows]))


def _clipping_bound_status(
    history: Sequence[Mapping[str, Any]],
    *,
    warmup_steps: int,
    total_steps: int,
    maximum_fraction: float,
) -> dict[str, Any]:
    """Report whether the final clipping bound is already impossible.

    Only post-warmup optimizer steps count.  Equality with the largest allowed
    integer count remains feasible; the run fails early only after that count
    is exceeded.
    """

    eligible_steps = max(0, int(total_steps) - int(warmup_steps))
    maximum_allowed = math.floor(
        float(maximum_fraction) * float(eligible_steps) + 1e-12
    )
    observed = sum(
        int(value.get("clipped", 0))
        for value in history
        if int(value.get("step", -1)) > int(warmup_steps)
    )
    return {
        "post_warmup_steps": int(eligible_steps),
        "maximum_allowed_clips": int(maximum_allowed),
        "observed_clips": int(observed),
        "mathematically_impossible": int(observed > maximum_allowed),
    }


def _scaled_backward_and_clip_checked(
    loss: Tensor,
    parameters: Any,
    *,
    loss_scale: float,
    grad_clip: float,
) -> Any:
    """Translate only Torch's explicit nonfinite-gradient error to evidence."""

    try:
        return boundary.scaled_backward_and_clip(
            loss, parameters, loss_scale=float(loss_scale),
            grad_clip=float(grad_clip),
        )
    except RuntimeError as exc:
        message = str(exc).lower()
        is_nonfinite_gradient_norm = (
            "total norm" in message
            and "gradient" in message
            and ("non-finite" in message or "nonfinite" in message)
        )
        if is_nonfinite_gradient_norm:
            raise FloatingPointError(
                f"nonfinite scaled pre-clip gradient norm: {exc}"
            ) from exc
        raise


def _load_completed_streamed_task(
    task_dir: Path,
    *,
    device: torch.device,
    task_kind: str,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any] | None:
    result_path = task_dir / "task_result.json"
    status_path = task_dir / "task_status.json"
    if not result_path.is_file() and not status_path.is_file():
        return None
    if not result_path.is_file() or not status_path.is_file():
        return None
    result = _json_load(result_path)
    status = _json_load(status_path)
    if status.get("status") != "complete":
        return None
    summary = dict(result.get("training_summary", {}))
    selected_step = int(summary.get("selected_step", -1))
    training_step = int(summary.get("training_step", -1))
    checkpoints = task_dir / "checkpoints"
    best_path = checkpoints / "best_ema.pt"
    best_pointer_path = checkpoints / "best.json"
    latest_path = checkpoints / "latest.json"
    if not all(path.is_file() for path in (best_path, best_pointer_path, latest_path)):
        raise ArtifactCompatibilityError("completed streamed task is missing checkpoint pointers")
    best_pointer = _json_load(best_pointer_path)
    latest = _json_load(latest_path)
    authoritative_name = str(best_pointer.get("authoritative_filename", ""))
    latest_name = str(latest.get("filename", ""))
    authoritative = checkpoints / authoritative_name
    latest_checkpoint = checkpoints / latest_name
    if (
        Path(authoritative_name).name != authoritative_name
        or Path(latest_name).name != latest_name
        or not authoritative.is_file()
        or not latest_checkpoint.is_file()
        or dict(status.get("fingerprints", {})) != dict(fingerprints)
        or dict(result.get("fingerprints", {})) != dict(fingerprints)
        or dict(best_pointer.get("fingerprints", {})) != dict(fingerprints)
        or dict(latest.get("fingerprints", {})) != dict(fingerprints)
        or status.get("task_result_sha256") != file_fingerprint(result_path)
        or int(status.get("selected_step", -2)) != selected_step
        or int(status.get("training_step", -2)) != training_step
        or int(best_pointer.get("selected_step", -2)) != selected_step
        or int(latest.get("step", -2)) != training_step
        or int(latest.get("stream_cursor", -2)) != training_step
        or best_pointer.get("authoritative_sha256") != file_fingerprint(authoritative)
        or best_pointer.get("best_ema_sha256") != file_fingerprint(best_path)
        or latest.get("sha256") != file_fingerprint(latest_checkpoint)
        or summary.get("checkpoint_sha256") != file_fingerprint(best_path)
        or summary.get("best_pointer_sha256") != file_fingerprint(best_pointer_path)
    ):
        raise ArtifactCompatibilityError("completed streamed task checkpoint chain mismatch")
    for value in (result, status, summary, best_pointer, latest):
        if (
            int(value.get("physical_training_performed", 0)) != 0
            or int(value.get("sampling_performed", 0)) != 0
        ):
            raise ArtifactCompatibilityError(
                "completed streamed task violates controls-only scope"
            )
    selected = _load_stream_checkpoint(
        best_path, device=device, task_kind=task_kind, fingerprints=fingerprints
    )
    authoritative_payload = _load_stream_checkpoint(
        authoritative, device=device, task_kind=task_kind, fingerprints=fingerprints
    )
    latest_payload = _load_stream_checkpoint(
        latest_checkpoint, device=device, task_kind=task_kind,
        fingerprints=fingerprints,
    )
    if (
        int(selected["step"]) != selected_step
        or int(authoritative_payload["step"]) != selected_step
        or int(latest_payload["step"]) != training_step
    ):
        raise ArtifactCompatibilityError("completed streamed checkpoint cursor mismatch")
    return result


def _failed_stream_task_result(
    task_dir: Path,
    *,
    task_kind: str,
    model_seed: int,
    fingerprints: Mapping[str, Any],
    exc: BaseException,
    gate_name: str = "pilot_selection_task",
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    failure = {"type": type(exc).__name__, "message": str(exc)}
    zero_banks = {
        bank: {
            scope: {"model_score_risk": 0.0, "lower_bound": 0.0}
            for scope in ("overall", "data_end")
        }
        for bank in ("a", "b")
    }
    metrics = {
        "complete": 0,
        "finite": 0,
        "boundary_admissible": 0,
        "selected_step": 0,
        "clip_fraction_steps_101_1000": 0.0,
        "final_200_clip_fraction": 0.0,
        "selection_objective_banks": zero_banks,
        "mean_dual_bank_selection_risk": 0.0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result = {
        "schema": RUN_SCHEMA + "-stream-task-result",
        "schema_version": 1,
        "task_kind": task_kind,
        "model_seed": int(model_seed),
        "evaluation_status": "evaluated",
        "metrics": metrics,
        "gate": not_evaluated_gate(
            gate_name, f"{type(exc).__name__}: {exc}"
        ),
        "failure": failure,
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(task_dir / "task_failure.json", result)
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "schema": RUN_SCHEMA + "-stream-task-status",
            "schema_version": 1,
            "status": "failed",
            "task_kind": task_kind,
            "model_seed": int(model_seed),
            "failure": failure,
            "fingerprints": dict(fingerprints),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return result


def run_streamed_control_task(
    *,
    task_dir: Path,
    task_kind: str,
    selection_arrays: boundary.ControlArrays,
    audit_arrays: boundary.ControlArrays | None,
    dynamics: Any,
    args: argparse.Namespace,
    device: torch.device,
    model_seed: int,
    learning_rate: float,
    loss_scale: float,
    stream_plan: StreamPlan,
    fingerprints: Mapping[str, Any],
    phase: str,
    show_progress: bool,
    thresholds: BoundaryControlThresholds,
    interrupt_after_checkpoint_step: int | None = None,
) -> dict[str, Any]:
    """Train one exact-resumable task from stateless fresh-state streams."""

    if task_kind not in {"implicit_teacher", "null"}:
        raise ValueError("streamed controls support only implicit_teacher and null")
    task_dir.mkdir(parents=True, exist_ok=True)
    result_path = task_dir / "task_result.json"
    status_path = task_dir / "task_status.json"
    completed_result = _load_completed_streamed_task(
        task_dir, device=device, task_kind=task_kind, fingerprints=fingerprints
    )
    if completed_result is not None:
        return completed_result

    checkpoints = task_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints / "latest.json"
    best_path = checkpoints / "best_ema.pt"
    best_pointer_path = checkpoints / "best.json"
    rng = boundary._set_seed(int(model_seed), boundary._derived_seed(model_seed, phase, "rng"))
    model = boundary.D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(args.weight_decay)
    )
    ema_state = boundary.init_ema_state(model)
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    completed = 0

    if latest_path.is_file():
        latest = _json_load(latest_path)
        filename = str(latest.get("filename", ""))
        checkpoint_path = checkpoints / filename
        if (
            Path(filename).name != filename
            or not checkpoint_path.is_file()
            or latest.get("sha256") != file_fingerprint(checkpoint_path)
            or dict(latest.get("fingerprints", {})) != dict(fingerprints)
        ):
            raise ArtifactCompatibilityError("streamed latest pointer is invalid")
        payload = _load_stream_checkpoint(
            checkpoint_path, device=device, task_kind=task_kind,
            fingerprints=fingerprints,
        )
        if dict(payload["stream_plan"]) != stream_plan_record(stream_plan):
            raise ArtifactCompatibilityError("streamed checkpoint plan mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        boundary._optimizer_to_device(optimizer, device)
        ema_state = {
            key: value.detach().clone().to(device)
            for key, value in dict(payload["ema_state_dict"]).items()
        }
        history = [dict(value) for value in payload["history"]]
        validations = [dict(value) for value in payload["validation_records"]]
        selection = dict(payload["checkpoint_selection"])
        completed = int(payload["step"])
        boundary.restore_rng_state(payload["rng_state"], rng)

    def validate(step: int) -> dict[str, Any]:
        with boundary.temporary_ema_weights(model, ema_state):
            return boundary._implicit_selection_record(
                model, selection_arrays, dynamics, step=int(step), args=args, device=device
            )

    def publish(step: int) -> None:
        nonlocal selection
        selection = boundary.select_dual_bank_checkpoint(validations)
        eligibility = {
            int(value["step"]): int(value["selection_eligible"])
            for value in selection.get("records", [])
        }
        for record in validations:
            record["selection_eligible"] = eligibility.get(int(record["step"]), 0)
        checkpoint_path = checkpoints / f"step-{int(step):08d}.pt"
        _save_stream_checkpoint(
            checkpoint_path, model=model, ema_state=ema_state, optimizer=optimizer,
            step=int(step), history=history, validations=validations,
            selection=selection, fingerprints=fingerprints, stream_plan=stream_plan,
            task_kind=task_kind, rng=rng,
        )
        atomic_write_json(
            latest_path,
            {
                "schema": RUN_SCHEMA + "-stream-latest",
                "schema_version": 1,
                "filename": checkpoint_path.name,
                "sha256": file_fingerprint(checkpoint_path),
                "step": int(step),
                "stream_cursor": int(step),
                "fingerprints": dict(fingerprints),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )

    if not validations:
        validations.append(validate(0))
        publish(0)

    validation_steps = set(int(value) for value in args.validation_steps)
    total_steps = int(args.train_steps)
    started = time.perf_counter()
    early_stop_reason: str | None = None
    if phase == "confirm" and completed > int(args.clip_warmup_steps):
        clipping_status = _clipping_bound_status(
            history,
            warmup_steps=int(args.clip_warmup_steps),
            total_steps=total_steps,
            maximum_fraction=float(
                thresholds.maximum_post_warmup_clip_fraction
            ),
        )
        if bool(int(clipping_status["mathematically_impossible"])):
            early_stop_reason = (
                "post-warmup clipping count makes the final 0.10 bound impossible"
            )
    step_iterator = (
        range(completed + 1, total_steps + 1)
        if early_stop_reason is None
        else ()
    )
    for step in step_iterator:
        law = "bounded_teacher" if task_kind == "implicit_teacher" else "dirichlet_null"
        batch = generate_stream_batch(
            stream_plan, phase=phase, law=law, step=int(step),
            device=device, dtype=torch.float32,
        )
        probes = stateless_probe_banks(
            stream_plan, phase=phase, law=law, step=int(step),
            batch_size=int(batch.states.shape[0]), device=device,
            dtype=batch.states.dtype,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective_a = boundary.dirichlet_score_objective(
            model, batch.tau, batch.states, batch.labels, dynamics, probes.a,
            create_graph=True,
        )
        objective_b = boundary.dirichlet_score_objective(
            model, batch.tau, batch.states, batch.labels, dynamics, probes.b,
            create_graph=True,
        )
        unscaled = 0.5 * (objective_a.loss + objective_b.loss)
        if not bool(torch.isfinite(unscaled)):
            raise FloatingPointError(f"non-finite {task_kind} stream loss at step {step}")
        gradient = _scaled_backward_and_clip_checked(
            unscaled, model.parameters(), loss_scale=float(loss_scale),
            grad_clip=float(args.grad_clip),
        )
        optimizer.step()
        boundary.update_ema_state(ema_state, model, float(args.ema_decay))
        energy = 0.5 * (objective_a.energy.mean() + objective_b.energy.mean())
        trace = 0.5 * (objective_a.trace.mean() + objective_b.trace.mean())
        drift = 0.5 * (objective_a.drift.mean() + objective_b.drift.mean())
        unscaled_value = float(unscaled.detach().cpu())
        energy_value = float(energy.detach().cpu())
        trace_value = float(trace.detach().cpu())
        drift_value = float(drift.detach().cpu())
        denominator = abs(energy_value) + abs(trace_value) + abs(drift_value)
        history.append(
            {
                "step": int(step),
                "loss": float(gradient.scaled_loss),
                "unscaled_loss": unscaled_value,
                "scaled_loss": float(gradient.scaled_loss),
                "loss_scale": float(gradient.loss_scale),
                "raw_gradient_norm": float(gradient.raw_gradient_norm),
                "scaled_preclip_gradient_norm": float(gradient.scaled_preclip_gradient_norm),
                "grad_norm": float(gradient.scaled_preclip_gradient_norm),
                "clipped": int(gradient.clipped),
                "energy": energy_value,
                "trace": trace_value,
                "drift": drift_value,
                "trace_bank_a": float(objective_a.trace.mean().detach().cpu()),
                "trace_bank_b": float(objective_b.trace.mean().detach().cpu()),
                "trace_bank_disagreement": float(
                    (objective_a.trace.mean() - objective_b.trace.mean()).abs().detach().cpu()
                ),
                "stream_batch_fingerprint": batch.fingerprint,
                "probe_bank_fingerprint": probes.fingerprint,
                "cancellation_ratio": abs(unscaled_value) / max(denominator, 1e-30),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
        if step in validation_steps:
            validations.append(validate(step))
            publish(step)
            if (
                interrupt_after_checkpoint_step is not None
                and int(step) == int(interrupt_after_checkpoint_step)
            ):
                raise RuntimeError(
                    f"injected interruption after streamed checkpoint {step}"
                )
        if phase == "confirm" and step > int(args.clip_warmup_steps):
            clipping_status = _clipping_bound_status(
                history,
                warmup_steps=int(args.clip_warmup_steps),
                total_steps=total_steps,
                maximum_fraction=float(
                    thresholds.maximum_post_warmup_clip_fraction
                ),
            )
            if bool(int(clipping_status["mathematically_impossible"])):
                early_stop_reason = (
                    "post-warmup clipping count makes the final 0.10 bound impossible"
                )
                if step not in validation_steps:
                    validations.append(validate(step))
                    publish(step)
                break
        if show_progress and (step % 50 == 0 or step == total_steps):
            elapsed = time.perf_counter() - started
            done = max(1, step - completed)
            eta = elapsed / done * max(0, total_steps - step)
            print(
                f"{phase}/{task_kind}: step {step}/{total_steps} "
                f"loss={history[-1]['loss']:.6g} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    final_step = int(history[-1]["step"]) if history else int(completed)
    if not latest_path.is_file() or int(_json_load(latest_path).get("step", -1)) != final_step:
        if not any(int(value.get("step", -1)) == final_step for value in validations):
            validations.append(validate(final_step))
        publish(final_step)
    selected_step = int(selection.get("selected_step", -1))
    authoritative = checkpoints / f"step-{selected_step:08d}.pt"
    if selected_step < 0 or not authoritative.is_file():
        raise ArtifactCompatibilityError("selected streamed checkpoint is missing")
    boundary.atomic_copy_file(authoritative, best_path)
    atomic_write_json(
        best_pointer_path,
        {
            "schema": RUN_SCHEMA + "-stream-best",
            "schema_version": 1,
            "selected_step": selected_step,
            "authoritative_filename": authoritative.name,
            "authoritative_sha256": file_fingerprint(authoritative),
            "best_ema_filename": best_path.name,
            "best_ema_sha256": file_fingerprint(best_path),
            "fingerprints": dict(fingerprints),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    selected = _load_stream_checkpoint(
        best_path, device=device, task_kind=task_kind, fingerprints=fingerprints
    )
    model.load_state_dict(selected["ema_state_dict"], strict=True)
    model.eval()
    diagnostics = boundary._history_diagnostics(
        history, warmup_steps=int(args.clip_warmup_steps), grad_clip=float(args.grad_clip)
    )
    stable_fraction = _clip_fraction(
        history, start=101, stop=min(1000, max(101, final_step))
    )
    final_window_start = max(1, final_step - 199)
    final_fraction = _clip_fraction(history, start=final_window_start, stop=final_step)
    summary = {
        "complete": 1,
        "finite": int(all(math.isfinite(float(value["loss"])) for value in history)),
        "task_kind": task_kind,
        "model_seed": int(model_seed),
        "selected_step": selected_step,
        "training_step": final_step,
        "target_training_steps": total_steps,
        "early_stopped": int(early_stop_reason is not None),
        "early_stop_reason": early_stop_reason,
        "checkpoint_selection": selection,
        "post_warmup_clip_fraction": diagnostics["post_warmup_clip_fraction"],
        "clip_fraction_steps_101_1000": stable_fraction,
        "final_200_clip_fraction": final_fraction,
        "optimization_diagnostics": diagnostics,
        "boundary_admissibility_certificate": boundary._model_boundary_certificate(model),
        "checkpoint_path": str(best_path.resolve()),
        "checkpoint_sha256": file_fingerprint(best_path),
        "best_pointer_sha256": file_fingerprint(best_pointer_path),
        "stream_plan": stream_plan_record(stream_plan),
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    selection_only = audit_arrays is None
    evaluation_arrays = selection_arrays if selection_only else audit_arrays
    assert evaluation_arrays is not None
    if selection_only:
        if task_kind == "implicit_teacher":
            raw = boundary._teacher_metrics(
                model=model, arrays=evaluation_arrays, dynamics=dynamics,
                summary=summary, args=args, device=device,
                include_objective_banks=False,
            )
            metrics = {
                "complete": raw["complete"],
                "finite": raw["finite"],
                "model_seed": raw["model_seed"],
                "selected_step": raw["selected_step"],
                "selection_overall_score_gain": raw["audit_overall_score_gain"],
                "selection_data_end_score_gain": raw["audit_data_end_score_gain"],
                "selection_overall_flux_cosine": raw["overall_flux_cosine"],
                "selection_data_end_flux_cosine": raw["time_bin_flux_cosines"][-1],
                "selection_time_bin_flux_cosines": raw["time_bin_flux_cosines"],
                "selection_overall_relative_flux_l2": raw["overall_relative_flux_l2"],
                "selection_data_end_relative_flux_l2": raw["time_bin_relative_flux_l2"][-1],
                "selection_time_bin_relative_flux_l2": raw["time_bin_relative_flux_l2"],
                "selection_overall": raw["overall"],
                "selection_data_end": raw["data_end"],
                "selection_time_bins": raw["time_bins"],
                "boundary_admissible": raw["boundary_admissible"],
                "post_warmup_clip_fraction": raw["post_warmup_clip_fraction"],
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        else:
            metrics = {
                "complete": 1,
                "finite": summary["finite"],
                "model_seed": int(model_seed),
                "selected_step": selected_step,
                "comparator": "analytic_zero",
                "boundary_admissible": int(
                    summary["boundary_admissibility_certificate"]["passed"]
                ),
                "post_warmup_clip_fraction": diagnostics["post_warmup_clip_fraction"],
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        gate = {
            "gate": "pilot_selection_task",
            "evaluation_status": "evaluated",
            "passed": int(bool(int(metrics["complete"])) and bool(int(metrics["finite"]))),
            "claim_scope": "train/selection-only stability qualification",
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    elif task_kind == "implicit_teacher":
        metrics = boundary._teacher_metrics(
            model=model, arrays=evaluation_arrays, dynamics=dynamics, summary=summary,
            args=args, device=device, include_objective_banks=True,
        )
        gate = boundary.evaluate_implicit_teacher_seed(metrics, thresholds)
    else:
        metrics = boundary._null_metrics(
            model=model, arrays=evaluation_arrays, dynamics=dynamics, summary=summary,
            args=args, device=device,
        )
        gate = boundary.evaluate_null_seed(metrics, thresholds)
    selection_banks = _selected_validation_banks(validations, selected_step)
    metrics["selection_objective_banks"] = selection_banks
    metrics["clip_fraction_steps_101_1000"] = stable_fraction
    metrics["final_200_clip_fraction"] = final_fraction
    metrics["post_warmup_clip_fraction"] = diagnostics["post_warmup_clip_fraction"]
    metrics["boundary_admissible"] = int(summary["boundary_admissibility_certificate"]["passed"])
    risk_values = [
        float(dict(dict(selection_banks[name])["overall"])["model_score_risk"])
        for name in ("a", "b")
    ]
    metrics["mean_dual_bank_selection_risk"] = float(np.mean(risk_values))
    path_rows = list(metrics.pop("audit_path_rows", []))
    boundary.atomic_write_csv(task_dir / "training_history.csv", history)
    checkpoint_rows = [
        {**dict(value), "physical_training_performed": 0, "sampling_performed": 0}
        for value in boundary._flatten_checkpoint_rows(validations)
    ]
    boundary.atomic_write_csv(task_dir / "checkpoint_metrics.csv", checkpoint_rows)
    selection_path_rows = [
        {**dict(value), "physical_training_performed": 0, "sampling_performed": 0}
        for value in boundary._flatten_selection_path_rows(validations)
    ]
    boundary.atomic_write_csv(
        task_dir / "selection_path_risks.csv", selection_path_rows
    )
    if path_rows:
        path_filename = (
            "selection_analytic_path_metrics.csv"
            if selection_only
            else "audit_path_risks.csv"
        )
        boundary.atomic_write_csv(
            task_dir / path_filename,
            [
                {**dict(value), "physical_training_performed": 0, "sampling_performed": 0}
                for value in path_rows
            ],
        )
    time_bin_key = "selection_time_bins" if selection_only else "time_bins"
    time_bin_rows = [
        {
            "time_bin": index,
            **dict(value),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        for index, value in enumerate(metrics.get(time_bin_key, []))
        if isinstance(value, Mapping)
    ]
    if time_bin_rows:
        filename = (
            "selection_time_bin_metrics.csv"
            if selection_only
            else "audit_time_bin_metrics.csv"
        )
        boundary.atomic_write_csv(task_dir / filename, time_bin_rows)
    atomic_write_json(task_dir / "training_summary.json", summary)
    result = {
        "schema": RUN_SCHEMA + "-stream-task-result",
        "schema_version": 1,
        "task_kind": task_kind,
        "model_seed": int(model_seed),
        "metrics": metrics,
        "gate": gate,
        "training_summary": summary,
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result = json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
    atomic_write_json(result_path, result)
    atomic_write_json(
        status_path,
        {
            "schema": RUN_SCHEMA + "-stream-task-status",
            "schema_version": 1,
            "status": "complete",
            "task_kind": task_kind,
            "model_seed": int(model_seed),
            "training_step": final_step,
            "selected_step": selected_step,
            "fingerprints": dict(fingerprints),
            "task_result_sha256": file_fingerprint(result_path),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return result


def _control_array_subset(
    arrays: boundary.ControlArrays, indices: np.ndarray, *, role: str
) -> boundary.ControlArrays:
    ids = torch.as_tensor(indices, dtype=torch.long)
    return boundary.ControlArrays(
        states=arrays.states.index_select(0, ids).contiguous(),
        tau=arrays.tau.index_select(0, ids).contiguous(),
        tau_fraction=arrays.tau_fraction.index_select(0, ids).contiguous(),
        labels=arrays.labels.index_select(0, ids).contiguous(),
        path_ids=np.asarray(arrays.path_ids)[indices].copy(),
        strata=np.asarray(arrays.strata)[indices].copy(),
        role=role,
        law=arrays.law,
        horizon=arrays.horizon,
    )


def _load_parent_panel(path: Path, *, role: str, law: str, horizon: float) -> boundary.ControlArrays:
    with np.load(path, allow_pickle=False) as value:
        return boundary.ControlArrays(
            states=torch.from_numpy(value["states"].copy()).float(),
            tau=torch.from_numpy(value["tau"].copy()).float(),
            tau_fraction=torch.from_numpy(value["tau_fraction"].copy()).float(),
            labels=torch.from_numpy(value["labels"].copy()).long(),
            path_ids=value["path_ids"].astype(np.int64, copy=True),
            strata=value["strata"].astype(np.int64, copy=True),
            role=role,
            law=law,
            horizon=float(horizon),
        )


def run_parent_forensic_replay(
    *,
    parent_run_dir: Path,
    output_dir: Path,
    dynamics: Any,
    device: torch.device,
    root_seed: int,
    probe_count: int = 16,
) -> dict[str, Any]:
    """Advisory raw/EMA replay on parent-train and fresh law-matched panels."""

    parent = Path(parent_run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    horizon = float(boundary.natural_horizon(dynamics))
    rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    for law, task_folder, array_name in (
        ("bounded_teacher", "implicit-teacher", "teacher_train"),
        ("dirichlet_null", "null", "null_train"),
    ):
        full = _load_parent_panel(
            parent / "synthetic_arrays" / f"{array_name}.npz",
            role="parent_train_forensic", law=law, horizon=horizon,
        )
        unique_paths = np.unique(full.path_ids)
        selected_paths = unique_paths[: min(8, unique_paths.size)]
        mask = np.isin(full.path_ids, selected_paths)
        parent_subset = _control_array_subset(
            full, np.flatnonzero(mask), role="parent_train_forensic_subset"
        )
        fresh = boundary._build_control_arrays(
            role="fresh_forensic", law=law, path_count=len(selected_paths),
            first_path_id=10_000_000 + (0 if law == "bounded_teacher" else 100_000),
            bin_counts=(4, 4, 4, 4, 16), horizon=horizon,
            grid_size=int(dynamics.grid_size),
            seed=boundary._derived_seed(int(root_seed), "forensic", law, "fresh"),
        )
        seed_dirs = sorted((parent / "tasks" / task_folder).glob("seed-*"))
        if not seed_dirs:
            raise FileNotFoundError(parent / "tasks" / task_folder)
        task_dir = seed_dirs[0]
        for step in (250, 4000):
            checkpoint_path = task_dir / "checkpoints" / f"step-{step:08d}.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            try:
                payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except TypeError:  # pragma: no cover
                payload = torch.load(checkpoint_path, map_location=device)
            checkpoint_records.append(
                {
                    "law": law,
                    "step": step,
                    "path": str(checkpoint_path.resolve()),
                    "sha256": file_fingerprint(checkpoint_path),
                    "task_kind": payload.get("task_kind"),
                    "model_seed": int(task_dir.name.removeprefix("seed-")),
                }
            )
            for weights_name, state_name in (
                ("raw", "model_state_dict"), ("ema", "ema_state_dict")
            ):
                model = boundary.D0BoundarySmoothPotentialUNet(
                    dynamics, base_channels=32
                ).to(device)
                model.load_state_dict(payload[state_name], strict=True)
                model.eval()
                panel_values: dict[str, dict[str, float]] = {}
                for panel_name, arrays in (
                    ("parent_train", parent_subset), ("fresh", fresh)
                ):
                    banks: dict[str, float] = {}
                    for bank_name in ("a", "b"):
                        components = boundary._risk_values(
                            model, arrays, dynamics, device=device,
                            batch_size=64, probes_per_state=int(probe_count),
                            probe_seed=boundary._derived_seed(
                                int(root_seed), "forensic", law, step,
                                weights_name, bank_name,
                            ),
                        )
                        banks[bank_name] = float(
                            np.asarray(components["model"], dtype=np.float64).mean()
                        )
                    panel_values[panel_name] = banks
                train_mean = float(np.mean(list(panel_values["parent_train"].values())))
                fresh_mean = float(np.mean(list(panel_values["fresh"].values())))
                row = {
                    "law": law,
                    "step": step,
                    "weights": weights_name,
                    "parent_train_risk_bank_a": panel_values["parent_train"]["a"],
                    "parent_train_risk_bank_b": panel_values["parent_train"]["b"],
                    "fresh_risk_bank_a": panel_values["fresh"]["a"],
                    "fresh_risk_bank_b": panel_values["fresh"]["b"],
                    "parent_train_mean_risk": train_mean,
                    "fresh_mean_risk": fresh_mean,
                    "fresh_minus_train_risk": fresh_mean - train_mean,
                    "parent_probe_disagreement": abs(
                        panel_values["parent_train"]["a"]
                        - panel_values["parent_train"]["b"]
                    ),
                    "fresh_probe_disagreement": abs(
                        panel_values["fresh"]["a"] - panel_values["fresh"]["b"]
                    ),
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
                rows.append(row)
    boundary.atomic_write_csv(output_dir / "forensic_risk_replay.csv", rows)
    record = {
        "schema": RUN_SCHEMA + "-parent-forensic-replay",
        "schema_version": 1,
        "role": "advisory_only",
        "eligible_for_gate": 0,
        "complete": 1,
        "finite": int(
            all(
                math.isfinite(float(value))
                for row in rows
                for key, value in row.items()
                if key not in {"law", "weights"}
            )
        ),
        "representative_steps": [250, 4000],
        "weights": ["raw", "ema"],
        "probe_count_per_bank": int(probe_count),
        "checkpoint_records": checkpoint_records,
        "rows": rows,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(output_dir / "forensic_replay.json", record)
    return record


def _run_pilot(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dynamics: Any,
    device: torch.device,
    thresholds: StabilityThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    horizon = float(boundary.natural_horizon(dynamics))
    panels = {
        law: _prepare_fixed_panel(
            run_dir, phase="pilot", law=law, role="selection",
            path_count=int(args.pilot_selection_paths),
            first_path_id=8_000_000 + index * 100_000,
            seed=boundary._derived_seed(int(args.root_seed), "pilot", law, "selection"),
            horizon=horizon, args=args,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
        )
        for index, law in enumerate(("bounded_teacher", "dirichlet_null"))
    }
    atomic_write_json(
        run_dir / "pilot_panel_registry.json",
        {
            "schema": RUN_SCHEMA + "-panel-registry",
            "schema_version": 1,
            "phase": "pilot",
            "selection": {
                law: boundary._arrays_identity(value) for law, value in panels.items()
            },
            "audit": None,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    stream_plan = build_stream_plan(
        root_seed=int(args.root_seed), grid_size=int(args.grid_size),
        horizon=float(horizon), clusters_per_step=int(args.clusters_per_step),
        bin_counts=tuple(args.anchor_bin_counts),
        probes_per_bank=int(args.training_probes_per_bank),
    )
    atomic_write_json(run_dir / "pilot_stream_plan.json", stream_plan_record(stream_plan))
    atomic_write_json(
        run_dir / "pilot_stream_replay_registry.json",
        _stream_replay_registry(stream_plan, phase="pilot"),
    )
    candidate_records: list[dict[str, Any]] = []
    task_failures: list[dict[str, Any]] = []
    pilot_seed = boundary._derived_seed(int(args.root_seed), "pilot", "shared-model")
    for candidate_index, learning_rate in enumerate(args.pilot_learning_rates):
        results: dict[str, Any] = {}
        for law in ("bounded_teacher", "dirichlet_null"):
            task_kind = "implicit_teacher" if law == "bounded_teacher" else "null"
            task_dir = run_dir / "pilot" / f"lr-{candidate_index:02d}" / task_kind
            fingerprints = _task_fingerprints(
                manifest=manifest, phase="pilot", law=law,
                model_seed=int(pilot_seed), learning_rate=float(learning_rate),
                loss_scale=float(args.implicit_loss_scale),
                stream_plan=stream_plan, selection=panels[law], audit=None,
            )
            try:
                results[task_kind] = run_streamed_control_task(
                    task_dir=task_dir, task_kind=task_kind,
                    selection_arrays=panels[law], audit_arrays=None,
                    dynamics=dynamics,
                    args=_stream_task_args(args, phase="pilot", learning_rate=float(learning_rate)),
                    device=device, model_seed=int(pilot_seed),
                    learning_rate=float(learning_rate),
                    loss_scale=float(args.implicit_loss_scale),
                    stream_plan=stream_plan, fingerprints=fingerprints,
                    phase="pilot", show_progress=not bool(args.no_progress),
                    thresholds=BoundaryControlThresholds(
                        bootstrap_confidence=float(args.bootstrap_confidence)
                    ),
                )
            except FloatingPointError as exc:
                results[task_kind] = _failed_stream_task_result(
                    task_dir, task_kind=task_kind, model_seed=int(pilot_seed),
                    fingerprints=fingerprints, exc=exc,
                )
                task_failures.append(
                    {
                        "candidate_index": candidate_index,
                        "learning_rate": float(learning_rate),
                        "task_kind": task_kind,
                        "model_seed": int(pilot_seed),
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
        candidate_records.append(
            {
                "candidate_index": candidate_index,
                "learning_rate": float(learning_rate),
                "teacher": results["implicit_teacher"],
                "null": results["null"],
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
    pilot_gate = evaluate_stability_pilot(candidate_records, thresholds)
    selected = select_stability_profile(candidate_records, thresholds)
    pilot_gate["selected_profile"] = selected
    atomic_write_json(
        run_dir / "pilot_candidate_registry.json",
        {
            "schema": RUN_SCHEMA + "-pilot-candidate-registry",
            "schema_version": 1,
            "candidates": candidate_records,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "pilot_task_failures.json",
        {
            "schema": RUN_SCHEMA + "-pilot-task-failures",
            "schema_version": 1,
            "failure_count": len(task_failures),
            "failures": task_failures,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(run_dir / "stability_pilot_gate.json", pilot_gate)
    if bool(int(pilot_gate.get("passed", 0))):
        _freeze_selected_profile(
            run_dir / "selected_stability_profile.json", selected
        )
    return pilot_gate, selected


def _run_confirmation(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dynamics: Any,
    device: torch.device,
    thresholds: StabilityThresholds,
    selected_profile: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(
        int(
            selected_profile.get(
                "selected",
                selected_profile.get("eligible", selected_profile.get("passed", 0)),
            )
        )
    ):
        raise ArtifactCompatibilityError("confirmation requires an eligible frozen stability profile")
    profile_value = selected_profile.get("profile", selected_profile)
    if not isinstance(profile_value, Mapping):
        raise ArtifactCompatibilityError("selected stability profile payload is missing")
    learning_rate = float(profile_value["learning_rate"])
    profile_binding = _bind_confirmation_profile(run_dir, selected_profile)
    horizon = float(boundary.natural_horizon(dynamics))
    panels: dict[str, dict[str, boundary.ControlArrays]] = {}
    for law_index, law in enumerate(("bounded_teacher", "dirichlet_null")):
        panels[law] = {}
        for role_index, (role, path_count) in enumerate(
            (("selection", args.confirm_selection_paths), ("audit", args.confirm_audit_paths))
        ):
            panels[law][role] = _prepare_fixed_panel(
                run_dir, phase="confirm", law=law, role=role,
                path_count=int(path_count),
                first_path_id=9_000_000 + law_index * 100_000 + role_index * 10_000,
                seed=boundary._derived_seed(int(args.root_seed), "confirm", law, role),
                horizon=horizon, args=args,
                scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            )
    atomic_write_json(
        run_dir / "confirmation_panel_registry.json",
        {
            "schema": RUN_SCHEMA + "-panel-registry",
            "schema_version": 1,
            "phase": "confirm",
            "panels": {
                law: {
                    role: boundary._arrays_identity(value)
                    for role, value in role_values.items()
                }
                for law, role_values in panels.items()
            },
            "whole_path_selection_audit_disjoint": int(
                all(
                    set(role_values["selection"].path_ids).isdisjoint(
                        set(role_values["audit"].path_ids)
                    )
                    for role_values in panels.values()
                )
            ),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    stream_plan = build_stream_plan(
        root_seed=int(args.root_seed), grid_size=int(args.grid_size),
        horizon=float(horizon), clusters_per_step=int(args.clusters_per_step),
        bin_counts=tuple(args.anchor_bin_counts),
        probes_per_bank=int(args.training_probes_per_bank),
    )
    atomic_write_json(
        run_dir / "confirmation_stream_plan.json", stream_plan_record(stream_plan)
    )
    atomic_write_json(
        run_dir / "confirmation_stream_replay_registry.json",
        _stream_replay_registry(stream_plan, phase="confirm"),
    )
    teacher_results: list[dict[str, Any]] = []
    null_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in args.confirm_model_seeds:
        for law, output in (("bounded_teacher", teacher_results), ("dirichlet_null", null_results)):
            task_kind = "implicit_teacher" if law == "bounded_teacher" else "null"
            task_dir = run_dir / "confirmation" / task_kind / f"seed-{int(seed)}"
            fingerprints = _task_fingerprints(
                manifest=manifest, phase="confirm", law=law,
                model_seed=int(seed), learning_rate=learning_rate,
                loss_scale=float(args.implicit_loss_scale),
                stream_plan=stream_plan, selection=panels[law]["selection"],
                audit=panels[law]["audit"],
                selected_profile_binding=profile_binding,
            )
            try:
                result = run_streamed_control_task(
                    task_dir=task_dir, task_kind=task_kind,
                    selection_arrays=panels[law]["selection"],
                    audit_arrays=panels[law]["audit"], dynamics=dynamics,
                    args=_stream_task_args(args, phase="confirm", learning_rate=learning_rate),
                    device=device, model_seed=int(seed), learning_rate=learning_rate,
                    loss_scale=float(args.implicit_loss_scale), stream_plan=stream_plan,
                    fingerprints=fingerprints, phase="confirm",
                    show_progress=not bool(args.no_progress),
                    thresholds=BoundaryControlThresholds(
                        bootstrap_confidence=float(args.bootstrap_confidence)
                    ),
                )
            except FloatingPointError as exc:
                result = _failed_stream_task_result(
                    task_dir, task_kind=task_kind, model_seed=int(seed),
                    fingerprints=fingerprints, exc=exc,
                    gate_name="confirmation_task",
                )
                failures.append(
                    {
                        "task_kind": task_kind, "model_seed": int(seed),
                        "type": type(exc).__name__, "message": str(exc),
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
            output.append(result)
    probe_status = classify_probe_bank_status(teacher_results, null_results)
    confirmation = evaluate_stability_confirmation(
        teacher_results, null_results, thresholds, probe_bank_status=probe_status
    )
    confirmation.update(
        {
            "teacher_results": teacher_results,
            "null_results": null_results,
            "selected_profile": dict(selected_profile),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    )
    atomic_write_json(
        run_dir / "implicit_teacher_confirmation.json",
        {
            "task_results": teacher_results,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "null_confirmation.json",
        {
            "task_results": null_results,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "confirmation_task_failures.json",
        {
            "failure_count": len(failures), "failures": failures,
            "physical_training_performed": 0, "sampling_performed": 0,
        },
    )
    atomic_write_json(run_dir / "boundary_control_stability_gate.json", confirmation)
    return confirmation


def _workflow_report(
    *,
    provenance: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pilot: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    require_gate: str,
    thresholds: StabilityThresholds,
) -> dict[str, Any]:
    del thresholds
    report = evaluate_stability_workflow(
        provenance=provenance,
        stein=preflight,
        pilot=pilot,
        confirmation=confirmation,
        require_gate=require_gate,
    )
    report.setdefault("schema", RUN_SCHEMA + "-gate-report")
    report.setdefault("schema_version", 1)
    report["physical_training_performed"] = 0
    report["sampling_performed"] = 0
    return report


def _save_report(run_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / "control_stability_decision.json", dict(report.get("decision", {})))
    atomic_write_json(run_dir / "stability_confirmation_report.json", dict(report))


def _atomic_save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format=path.suffix.removeprefix("."), dpi=150, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_plot_artifacts(run_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    written: list[str] = []
    for phase, glob_pattern, filename in (
        ("pilot", "pilot/**/training_history.csv", "pilot_learning_curves.png"),
        ("confirm", "confirmation/**/training_history.csv", "confirmation_learning_curves.png"),
    ):
        paths = sorted(run_dir.glob(glob_pattern))
        if not paths:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue
            steps = [int(value["step"]) for value in rows]
            losses = [float(value["scaled_loss"]) for value in rows]
            gradients = [float(value["scaled_preclip_gradient_norm"]) for value in rows]
            label = path.parent.relative_to(run_dir).as_posix()
            axes[0].plot(steps, losses, linewidth=0.9, label=label)
            axes[1].plot(steps, gradients, linewidth=0.9, label=label)
        axes[0].set_title(f"{phase} scaled objective")
        axes[0].set_xlabel("step")
        axes[1].set_title(f"{phase} scaled pre-clip gradient")
        axes[1].set_xlabel("step")
        axes[1].set_yscale("log")
        for axis in axes:
            axis.grid(alpha=0.25)
        if len(paths) <= 10:
            axes[1].legend(fontsize=6)
        figure.tight_layout()
        output = run_dir / filename
        _atomic_save_figure(figure, output)
        plt.close(figure)
        written.append(output.name)

    pilot_gate_path = run_dir / "stability_pilot_gate.json"
    if pilot_gate_path.is_file():
        pilot = _json_load(pilot_gate_path)
        candidates = [dict(value) for value in pilot.get("candidate_gates", [])]
        if candidates:
            figure, axes = plt.subplots(1, 2, figsize=(9, 4))
            learning_rates = [float(value["learning_rate"]) for value in candidates]
            risks = [float(value.get("mean_teacher_selection_risk", math.nan)) for value in candidates]
            clipping = [float(value.get("maximum_clip_fraction_observed", math.nan)) for value in candidates]
            axes[0].plot(learning_rates, risks, marker="o")
            axes[1].plot(learning_rates, clipping, marker="o")
            axes[1].axhline(0.10, color="red", linestyle="--", linewidth=1)
            for axis in axes:
                axis.set_xscale("log")
                axis.grid(alpha=0.25)
                axis.set_xlabel("learning rate")
            axes[0].set_title("teacher selection risk")
            axes[1].set_title("maximum clipping fraction")
            figure.tight_layout()
            output = run_dir / "selected_stability_profile.png"
            _atomic_save_figure(figure, output)
            plt.close(figure)
            written.append(output.name)
    return written


def _write_summary_csvs(run_dir: Path) -> list[str]:
    written: list[str] = []
    pilot_path = run_dir / "stability_pilot_gate.json"
    if pilot_path.is_file():
        pilot = _json_load(pilot_path)
        rows = [
            {
                "candidate_index": index,
                "learning_rate": value.get("learning_rate"),
                "passed": value.get("passed"),
                "mean_teacher_selection_risk": value.get("mean_teacher_selection_risk"),
                "maximum_clip_fraction_observed": value.get("maximum_clip_fraction_observed"),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
            for index, value in enumerate(
                dict(item) for item in pilot.get("candidate_gates", [])
            )
        ]
        if rows:
            boundary.atomic_write_csv(run_dir / "pilot_profile_summary.csv", rows)
            written.append("pilot_profile_summary.csv")
    confirmation_path = run_dir / "boundary_control_stability_gate.json"
    if confirmation_path.is_file():
        confirmation = _json_load(confirmation_path)
        rows = []
        for study_name, result_key in (
            ("implicit_teacher", "teacher_results"), ("null", "null_results")
        ):
            for result in confirmation.get(result_key, []):
                metrics = dict(dict(result).get("metrics", {}))
                rows.append(
                    {
                        "task_kind": study_name,
                        "model_seed": result.get("model_seed"),
                        "complete": metrics.get("complete"),
                        "finite": metrics.get("finite"),
                        "selected_step": metrics.get("selected_step"),
                        "post_warmup_clip_fraction": metrics.get("post_warmup_clip_fraction"),
                        "audit_overall_score_gain": metrics.get("audit_overall_score_gain"),
                        "audit_data_end_score_gain": metrics.get("audit_data_end_score_gain"),
                        "overall_flux_cosine": metrics.get("overall_flux_cosine"),
                        "overall_relative_flux_l2": metrics.get("overall_relative_flux_l2"),
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
        if rows:
            boundary.atomic_write_csv(run_dir / "confirmation_seed_metrics.csv", rows)
            written.append("confirmation_seed_metrics.csv")
    return written


def _finish(
    run_dir: Path,
    *,
    report: Mapping[str, Any],
    stage: str,
    phase: str,
    execution_failed: bool = False,
    skips: Sequence[Mapping[str, Any]] = (),
) -> int:
    final_skips = [dict(value) for value in skips]
    try:
        _write_summary_csvs(run_dir)
        _write_plot_artifacts(run_dir)
    except Exception as exc:
        execution_failed = True
        final_skips.append(
            {"stage": "plot_artifacts", "reason": f"{type(exc).__name__}: {exc}"}
        )
        atomic_write_json(
            run_dir / "plot_artifact_failure.json",
            {
                "type": type(exc).__name__, "message": str(exc),
                "physical_training_performed": 0, "sampling_performed": 0,
            },
        )
    required_pass = 0 if execution_failed else int(report.get("required_gate_pass", 0))
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    decision = dict(report.get("decision", {}))
    _write_status(
        run_dir,
        status="failed" if execution_failed else "complete",
        outcome="implementation_error" if execution_failed else ("complete" if required_pass else "gate_failed"),
        phase=phase,
        stage=stage,
        required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=decision.get("decision", "controls_not_run"),
        recommended_next_action=decision.get("recommended_next_action"),
        probe_bank_status=decision.get("probe_bank_status", ProbeBankStatus.NOT_EVALUATED.value),
        physical_training_authorized=int(decision.get("physical_training_authorized", 0)) if not execution_failed else 0,
        physical_training_performed=0,
        sampling_authorized=0,
        sampling_performed=0,
        skips=final_skips,
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_size=int(registry_path.stat().st_size),
    )
    return 2 if execution_failed or not required_pass else 0


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    print(f"stability-confirmation run directory: {run_dir.resolve()}", flush=True)
    thresholds = StabilityThresholds(
        stein_bootstrap_confidence=float(args.stein_bootstrap_confidence),
        stein_paths=int(args.stein_paths),
        stein_teacher_amplitudes=tuple(args.stein_amplitudes),
        pilot_learning_rates=tuple(args.pilot_learning_rates),
        pilot_steps=int(args.pilot_steps),
        maximum_clip_fraction=0.10,
        confirmation_teacher_seeds=len(args.confirm_model_seeds),
        minimum_passing_teacher_seeds=2,
        confirmation_null_seeds=len(args.confirm_model_seeds),
    )
    mutation_started = False
    try:
        device = boundary._device(args.device)
        backend = boundary.configure_exact_torch_backend(device)
        runtime = boundary._runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
        source_hash, source_paths = _source_record()
        parent = verify_parent_scale_repair_run(args.parent_scale_repair_run_dir)
        if not math.isclose(
            float(parent.get("implicit_loss_scale", float("nan"))),
            float(args.implicit_loss_scale),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ArtifactCompatibilityError(
                "configured implicit loss scale does not exactly match the verified parent calibration"
            )
        provenance = {
            "schema": RUN_SCHEMA + "-parent-provenance",
            "schema_version": 1,
            **dict(parent),
            "verifier_source_fingerprint": source_hash,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        provenance_path = run_dir / "parent_provenance.json"
        if provenance_path.is_file():
            if _json_load(provenance_path) != provenance:
                raise ArtifactCompatibilityError("resume parent provenance mismatch")
        elif resumed:
            raise ArtifactCompatibilityError("resume is missing parent provenance")
        else:
            atomic_write_json(provenance_path, provenance)

        scientific = _scientific_config(args, parent, thresholds)
        scientific_fingerprint = config_fingerprint(scientific)
        manifest = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "created_at": _now(),
            "run_dir": str(run_dir.resolve()),
            "scientific_config": scientific,
            "scientific_fingerprint": scientific_fingerprint,
            "runtime": runtime,
            "runtime_fingerprint": runtime_fingerprint,
            "source_fingerprint": source_hash,
            "source_paths": source_paths,
            "parent_provenance_sha256": file_fingerprint(provenance_path),
            "claim_scope": CLAIM_SCOPE,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            existing = _json_load(manifest_path)
            for key in (
                "schema", "schema_version", "scientific_config", "scientific_fingerprint",
                "runtime", "runtime_fingerprint", "source_fingerprint", "source_paths",
                "parent_provenance_sha256", "claim_scope",
            ):
                if existing.get(key) != manifest.get(key):
                    raise ArtifactCompatibilityError(f"resume manifest mismatch for {key}")
            manifest = existing
        elif resumed:
            raise ArtifactCompatibilityError("resume is missing frozen manifest")
        else:
            atomic_write_json(manifest_path, manifest)

        previous = _json_load(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
        if resumed and str(previous.get("status", "")) in {"complete", "failed"}:
            _verify_terminal_registry(run_dir)
        _write_status(
            run_dir, status="running", phase="provenance", stage=str(args.stage),
            require_gate=str(args.require_gate), attempt_count=int(previous.get("attempt_count", 0)) + 1,
        )
        mutation_started = True

        if args.stage == "report":
            preflight = _json_load(run_dir / "stability_preflight_gate.json")
            pilot = (
                _json_load(run_dir / "stability_pilot_gate.json")
                if (run_dir / "stability_pilot_gate.json").is_file()
                else not_evaluated_gate("stability_pilot", "pilot was not run")
            )
            confirmation = (
                _json_load(run_dir / "boundary_control_stability_gate.json")
                if (run_dir / "boundary_control_stability_gate.json").is_file()
                else not_evaluated_gate("stability_confirmation", "confirmation was not run")
            )
            report = _workflow_report(
                provenance=provenance, preflight=preflight, pilot=pilot,
                confirmation=confirmation, require_gate=str(args.require_gate),
                thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(run_dir, report=report, stage="report", phase="report")

        dynamics = boundary._make_dynamics(args)
        stream_plan = build_stream_plan(
            root_seed=int(args.root_seed), grid_size=int(args.grid_size),
            horizon=float(boundary.natural_horizon(dynamics)),
            clusters_per_step=int(args.clusters_per_step),
            bin_counts=tuple(args.anchor_bin_counts),
            probes_per_bank=int(args.training_probes_per_bank),
        )
        atomic_write_json(run_dir / "stream_plan.json", stream_plan_record(stream_plan))
        _write_status(run_dir, status="running", phase="preflight")
        replay_rows: list[dict[str, Any]] = []
        for replay_phase in ("pilot", "confirm"):
            for replay_law in ("bounded_teacher", "dirichlet_null"):
                for replay_step in (1, 2):
                    replay = stream_replay_record(
                        stream_plan, phase=replay_phase, law=replay_law,
                        step=replay_step,
                    )
                    verification = verify_stream_replay(stream_plan, replay)
                    replay_rows.append(
                        {"record": replay, "verification": verification}
                    )
        replay_preflight = {
            "schema": RUN_SCHEMA + "-stream-replay-preflight",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": int(
                all(
                    bool(int(value["verification"].get("passed", 0)))
                    for value in replay_rows
                )
            ),
            "plan_fingerprint": stream_plan.fingerprint,
            "records": replay_rows,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        atomic_write_json(
            run_dir / "stream_replay_preflight.json", replay_preflight
        )
        stein = run_stein_identity_preflight(
            dynamics, device=device, root_seed=int(args.root_seed),
            path_count=int(args.stein_paths), anchors_per_path=int(args.anchors_per_path),
            bootstrap_reps=int(args.bootstrap_reps),
            confidence=float(args.stein_bootstrap_confidence),
            teacher_scales=tuple(args.stein_amplitudes),
        )
        atomic_write_json(run_dir / "stein_identity_preflight.json", stein)
        stein_gate = evaluate_stein_identity_preflight(stein, thresholds)
        atomic_write_json(run_dir / "stein_identity_preflight_gate.json", stein_gate)
        preflight = {
            "schema": RUN_SCHEMA + "-preflight-gate",
            "schema_version": 1,
            "gate": "stability_preflight",
            "evaluation_status": (
                "evaluated"
                if bool(int(replay_preflight.get("passed", 0)))
                else "not_evaluated"
            ),
            "passed": int(
                bool(int(provenance.get("passed", 0)))
                and bool(int(stein_gate.get("passed", 0)))
                and bool(int(replay_preflight.get("passed", 0)))
            ),
            "provenance": provenance,
            "stein_identity": stein,
            "stein_identity_gate": stein_gate,
            "stream_replay": replay_preflight,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        atomic_write_json(run_dir / "stability_preflight_gate.json", preflight)
        try:
            forensic = run_parent_forensic_replay(
                parent_run_dir=args.parent_scale_repair_run_dir,
                output_dir=run_dir / "forensics", dynamics=dynamics, device=device,
                root_seed=int(args.root_seed), probe_count=16,
            )
            atomic_write_json(run_dir / "parent_forensic_replay.json", forensic)
        except Exception as exc:
            atomic_write_json(
                run_dir / "parent_forensic_replay_warning.json",
                {
                    "role": "advisory_only", "type": type(exc).__name__,
                    "message": str(exc), "physical_training_performed": 0,
                    "sampling_performed": 0,
                },
            )

        pilot = not_evaluated_gate("stability_pilot", "pilot was not run")
        confirmation = not_evaluated_gate("stability_confirmation", "confirmation was not run")
        if args.stage == "preflight" or not bool(int(preflight["passed"])):
            report = _workflow_report(
                provenance=provenance, preflight=preflight, pilot=pilot,
                confirmation=confirmation, require_gate=str(args.require_gate), thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(
                run_dir, report=report, stage=str(args.stage), phase="preflight",
                skips=[] if preflight["passed"] else [{"stage": "pilot_and_confirmation", "reason": "stability preflight failed"}],
            )

        if args.stage in {"pilot", "all"}:
            _write_status(run_dir, status="running", phase="pilot")
            pilot, selected = _run_pilot(
                run_dir, args=args, manifest=manifest, dynamics=dynamics,
                device=device, thresholds=thresholds,
            )
            if args.stage == "pilot" or not bool(int(pilot.get("passed", 0))):
                report = _workflow_report(
                    provenance=provenance, preflight=preflight, pilot=pilot,
                    confirmation=confirmation, require_gate=str(args.require_gate), thresholds=thresholds,
                )
                _save_report(run_dir, report)
                return _finish(
                    run_dir, report=report, stage=str(args.stage), phase="pilot",
                    skips=[] if pilot.get("passed", 0) else [{"stage": "confirmation", "reason": "no pilot stability profile qualified"}],
                )
        else:
            pilot = _json_load(run_dir / "stability_pilot_gate.json")
            if not bool(int(pilot.get("passed", 0))):
                raise ArtifactCompatibilityError("confirmation requires a passing pilot gate")
            selected = _json_load(run_dir / "selected_stability_profile.json")

        _write_status(run_dir, status="running", phase="confirmation")
        confirmation = _run_confirmation(
            run_dir, args=args, manifest=manifest, dynamics=dynamics, device=device,
            thresholds=thresholds, selected_profile=selected,
        )
        report = _workflow_report(
            provenance=provenance, preflight=preflight, pilot=pilot,
            confirmation=confirmation, require_gate=str(args.require_gate), thresholds=thresholds,
        )
        _save_report(run_dir, report)
        return _finish(run_dir, report=report, stage=str(args.stage), phase="confirmation")
    except Exception as exc:
        if resumed and not mutation_started:
            if not args.no_progress:
                print(f"stability-confirmation resume rejected without mutation: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        atomic_write_json(
            run_dir / "failure.json",
            {
                "schema": RUN_SCHEMA + "-failure", "schema_version": 1,
                "type": type(exc).__name__, "message": str(exc), "stage": str(args.stage),
                "physical_training_performed": 0, "sampling_performed": 0,
            },
        )
        failed = not_evaluated_gate("workflow", f"{type(exc).__name__}: {exc}")
        report = _workflow_report(
            provenance=failed, preflight=failed, pilot=failed, confirmation=failed,
            require_gate=str(args.require_gate), thresholds=thresholds,
        )
        _save_report(run_dir, report)
        _finish(
            run_dir, report=report, stage=str(args.stage), phase="failure",
            execution_failed=True,
            skips=[{"stage": "remaining", "reason": f"{type(exc).__name__}: {exc}"}],
        )
        if not args.no_progress:
            print(f"stability confirmation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
