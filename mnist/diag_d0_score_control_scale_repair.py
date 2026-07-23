"""Versioned optimizer-scale confirmation for boundary-admissible D0 controls.

This command binds the completed boundary-control run whose exact bounded
teacher was learned but whose uncalibrated supervised loss clipped almost
every optimizer step.  It creates fresh synthetic states, calibrates separate
supervised and implicit loss multipliers from training states only, and then
reruns the supervised, implicit-teacher, and stationary-null controls.  It
never trains on physical score states and never imports or calls a sampler.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

import mnist.diag_d0_score_boundary_controls as boundary
from mnist.d0_dirichlet_score import dirichlet_score_objective
from mnist.d0_score_boundary_control_gate import (
    BoundaryControlThresholds,
    evaluate_implicit_teacher_seed,
    evaluate_implicit_teacher_study,
    evaluate_null_seed,
    evaluate_null_study,
    evaluate_supervised_teacher,
)
from mnist.d0_score_boundary_controls import D0BoundarySmoothPotentialUNet
from mnist.d0_score_control_scale_repair_gate import (
    ProbeBankStatus,
    classify_probe_bank_status,
    evaluate_loss_scale_calibration,
    evaluate_optimizer_task_health,
    evaluate_scale_repair_gates,
    not_evaluated_study,
    split_supervised_teacher_gate,
)
from mnist.d0_score_control_scale_repair_provenance import (
    verify_parent_boundary_control_run,
)
from mnist.d0_score_optimizer_scale import calibrate_initial_loss_scale
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    array_fingerprint,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.eulerian_flux_mnist import natural_horizon


RUN_SCHEMA = "experiment12-d0-score-control-scale-repair"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "boundary-admissible optimizer-scaled synthetic implicit-score controls only"
CALIBRATION_STATE_COUNT = 256

FRESH_DEFAULTS: dict[str, Any] = {
    **{
        key: value
        for key, value in boundary.DEFAULTS.items()
        if key
        not in {
            "teacher_seeds",
            "null_seeds",
            "supervised_seed",
            "teacher_data_seed",
            "null_data_seed",
            "calibration_seed",
            "training_probe_seed",
            "selection_probe_a_seed",
            "selection_probe_b_seed",
            "audit_probe_a_seed",
            "audit_probe_b_seed",
            "bootstrap_seed",
            "batch_index_seed",
        }
    },
    "teacher_data_seed": 260781,
    "null_data_seed": 260782,
    "calibration_seed": 260783,
    "supervised_seed": 260784,
    "teacher_seeds": (260785, 260786, 260787),
    "null_seeds": (260785, 260786, 260787),
    "training_probe_seed": 260788,
    "selection_probe_a_seed": 260789,
    "selection_probe_b_seed": 260790,
    "audit_probe_a_seed": 260791,
    "audit_probe_b_seed": 260792,
    "bootstrap_seed": 260793,
    "batch_index_seed": 260794,
    "supervised_initial_grad_target": 0.10,
    "implicit_initial_grad_target": 0.10,
    "teacher_path_base": 6_000_000,
    "null_path_base": 7_000_000,
}


def _json_load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "controls", "report", "all"), default="all")
    parser.add_argument("--parent-boundary-control-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_d0_score_control_scale_repair"))
    parser.add_argument("--run-name", default="production-boundary-control-scale-repair")
    parser.add_argument("--require-gate", choices=("none", "preflight", "controls"), default="none")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--grid-size", type=int, default=28)
    parser.add_argument("--sample-steps", type=int, default=512)
    parser.add_argument("--reference-substeps", type=int, default=256)
    parser.add_argument("--tau-eff", type=float, default=5e-5)
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff",), default="alpha_eff")
    parser.add_argument("--alpha-eff", type=float, default=1.0)
    parser.add_argument("--mass-floor", type=float, default=1e-7)
    parser.add_argument("--limiter-fraction", type=float, default=1.0)
    parser.add_argument("--lambda-mix", type=float, default=0.35)

    parser.add_argument("--train-paths", type=int, default=128)
    parser.add_argument("--selection-paths", type=int, default=32)
    parser.add_argument("--audit-paths", type=int, default=32)
    parser.add_argument("--anchors-per-path", type=int, default=32)
    parser.add_argument("--anchor-bin-counts", default="4,4,4,4,16")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--clip-warmup-steps", type=int, default=500)
    parser.add_argument("--training-probes", type=int, default=4)
    parser.add_argument("--selection-probes", type=int, default=16)
    parser.add_argument("--audit-probes", type=int, default=64)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.90)

    parser.add_argument("--teacher-data-seed", type=int, default=260781)
    parser.add_argument("--null-data-seed", type=int, default=260782)
    parser.add_argument("--calibration-seed", type=int, default=260783)
    parser.add_argument("--supervised-seed", type=int, default=260784)
    parser.add_argument("--teacher-seeds", default="260785,260786,260787")
    parser.add_argument("--null-seeds", default="260785,260786,260787")
    parser.add_argument("--training-probe-seed", type=int, default=260788)
    parser.add_argument("--selection-probe-a-seed", type=int, default=260789)
    parser.add_argument("--selection-probe-b-seed", type=int, default=260790)
    parser.add_argument("--audit-probe-a-seed", type=int, default=260791)
    parser.add_argument("--audit-probe-b-seed", type=int, default=260792)
    parser.add_argument("--bootstrap-seed", type=int, default=260793)
    parser.add_argument("--batch-index-seed", type=int, default=260794)
    parser.add_argument("--operator-hutchinson-probes", type=int, default=4096)
    parser.add_argument("--supervised-initial-grad-target", type=float, default=0.10)
    parser.add_argument("--implicit-initial-grad-target", type=float, default=0.10)

    args = parser.parse_args(argv)
    try:
        args.anchor_bin_counts = boundary._parse_csv_ints(args.anchor_bin_counts)
        args.teacher_seeds = boundary._parse_csv_ints(args.teacher_seeds)
        args.null_seeds = boundary._parse_csv_ints(args.null_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    args.teacher_path_base = int(FRESH_DEFAULTS["teacher_path_base"])
    args.null_path_base = int(FRESH_DEFAULTS["null_path_base"])
    if len(args.anchor_bin_counts) != 5 or sum(args.anchor_bin_counts) != int(args.anchors_per_path):
        parser.error("anchor-bin-counts must contain five values summing to anchors-per-path")
    if len(args.teacher_seeds) != 3 or len(set(args.teacher_seeds)) != 3:
        parser.error("teacher-seeds must contain three distinct seeds")
    if len(args.null_seeds) != 3 or len(set(args.null_seeds)) != 3:
        parser.error("null-seeds must contain three distinct seeds")
    if args.teacher_seeds != args.null_seeds:
        parser.error("teacher-seeds and null-seeds must be paired identically")
    positive = (
        "train_paths", "selection_paths", "audit_paths", "anchors_per_path",
        "base_channels", "batch_size", "validation_batch_size", "train_steps",
        "validation_every", "checkpoint_every", "training_probes",
        "selection_probes", "audit_probes", "bootstrap_reps",
        "operator_hutchinson_probes",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("supervised_initial_grad_target", "implicit_initial_grad_target"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0.0 < float(args.ema_decay) < 1.0:
        parser.error("ema-decay must be in (0,1)")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        parser.error("bootstrap-confidence must be in (0,1)")
    if args.require_gate != "none":
        mismatches = [
            f"{key}={getattr(args, key)!r}, expected {expected!r}"
            for key, expected in FRESH_DEFAULTS.items()
            if hasattr(args, key) and not _semantic_close(getattr(args, key), expected)
        ]
        if mismatches:
            parser.error("required production gate rejects overrides: " + "; ".join(mismatches))
    return args


def _source_record() -> tuple[str, list[str]]:
    here = Path(__file__).resolve()
    names = (
        here.name,
        "diag_d0_score_boundary_controls.py",
        "d0_score_control_scale_repair_gate.py",
        "d0_score_control_scale_repair_provenance.py",
        "d0_score_optimizer_scale.py",
        "d0_score_boundary_controls.py",
        "d0_score_boundary_control_gate.py",
        "d0_dirichlet_score.py",
        "d0_one_image_gate.py",
        "eulerian_flux_mnist.py",
    )
    paths = [here.with_name(name) for name in names]
    existing = [path for path in paths if path.is_file()]
    return source_fingerprint(existing), [str(path) for path in existing]


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    current = _json_load(path) if path.is_file() else {}
    current.update(updates)
    current.setdefault("schema", RUN_SCHEMA)
    current.setdefault("schema_version", RUN_SCHEMA_VERSION)
    current.setdefault("physical_training_performed", 0)
    current.setdefault("sampling_performed", 0)
    current["updated_at"] = _now()
    atomic_write_json(path, current)
    return current


def _scientific_config(
    args: argparse.Namespace,
    *,
    parent: Mapping[str, Any],
    thresholds: BoundaryControlThresholds,
) -> dict[str, Any]:
    return {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "model": {
            "schema": boundary.MODEL_SCHEMA,
            "schema_version": boundary.MODEL_SCHEMA_VERSION,
            "input_channels": [
                "N*s", "log1p(N*s)", "tau/T", "label", "periodic_coordinates"
            ],
            "raw_log_density_used": 0,
            "base_channels": int(args.base_channels),
        },
        "kernel": {key: getattr(args, key) for key in boundary.EXPECTED_KERNEL},
        "synthetic_data": {
            "train_paths": int(args.train_paths),
            "selection_paths": int(args.selection_paths),
            "audit_paths": int(args.audit_paths),
            "anchors_per_path": int(args.anchors_per_path),
            "anchor_bin_counts": list(args.anchor_bin_counts),
            "teacher_path_base": int(args.teacher_path_base),
            "null_path_base": int(args.null_path_base),
            "fresh_parent_states_reused": 0,
            "fresh_parent_path_ids_reused": 0,
            "teacher_epsilon": 0.5,
            "teacher_version": boundary.BOUNDED_TEACHER_VERSION,
            "teacher_and_null_states_independent": 1,
        },
        "optimization": {
            "base_channels": int(args.base_channels),
            "batch_size": int(args.batch_size),
            "validation_batch_size": int(args.validation_batch_size),
            "train_steps": int(args.train_steps),
            "validation_every": int(args.validation_every),
            "checkpoint_every": int(args.checkpoint_every),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "ema_decay": float(args.ema_decay),
            "grad_clip": float(args.grad_clip),
            "clip_warmup_steps": int(args.clip_warmup_steps),
            "training_probes": int(args.training_probes),
            "selection_probes": int(args.selection_probes),
            "audit_probes": int(args.audit_probes),
            "calibration_state_count": CALIBRATION_STATE_COUNT,
            "supervised_initial_grad_target": float(args.supervised_initial_grad_target),
            "implicit_initial_grad_target": float(args.implicit_initial_grad_target),
            "scale_formula": "min(1,target/raw_initial_gradient_norm)",
            "amp": False,
        },
        "seeds": {
            name: list(getattr(args, name)) if name in {"teacher_seeds", "null_seeds"} else int(getattr(args, name))
            for name in (
                "teacher_data_seed", "null_data_seed", "calibration_seed",
                "supervised_seed", "teacher_seeds", "null_seeds",
                "training_probe_seed", "selection_probe_a_seed",
                "selection_probe_b_seed", "audit_probe_a_seed",
                "audit_probe_b_seed", "bootstrap_seed", "batch_index_seed",
            )
        },
        "bootstrap": {
            "reps": int(args.bootstrap_reps),
            "confidence": float(args.bootstrap_confidence),
        },
        "preflight": {"operator_hutchinson_probes": int(args.operator_hutchinson_probes)},
        "thresholds": thresholds.to_dict(),
        "parent_boundary_run_status_sha256": dict(parent["artifacts"])["status"]["sha256"],
        "parent_scientific_fingerprint": parent["scientific_fingerprint"],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _calibration_binding(
    *,
    kind: str,
    arrays: boundary.ControlArrays,
    args: argparse.Namespace,
    scientific_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint_value: str,
    target: float,
) -> dict[str, Any]:
    model_initialization_seed = boundary._derived_seed(
        int(args.calibration_seed), str(kind), "model"
    )
    return {
        "objective_kind": str(kind),
        "scientific_fingerprint": scientific_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "source_fingerprint": source_fingerprint_value,
        "teacher_train_identity": boundary._arrays_identity(arrays),
        "calibration_seed": int(args.calibration_seed),
        "model_initialization_seed": int(model_initialization_seed),
        "training_probe_seed": int(args.training_probe_seed),
        "target_initial_gradient_norm": float(target),
        "calibration_state_count": min(CALIBRATION_STATE_COUNT, int(arrays.states.shape[0])),
    }


def _load_or_calibrate(
    path: Path,
    *,
    objective_kind: str,
    arrays: boundary.ControlArrays,
    dynamics: Any,
    args: argparse.Namespace,
    device: torch.device,
    binding: Mapping[str, Any],
    target: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    count = min(CALIBRATION_STATE_COUNT, int(arrays.states.shape[0]))
    if path.is_file():
        record = _json_load(path)
        if dict(record.get("binding", {})) != dict(binding):
            raise ArtifactCompatibilityError(f"{objective_kind} loss-scale calibration binding mismatch")
    else:
        model_seed = int(
            binding.get(
                "model_initialization_seed",
                boundary._derived_seed(int(args.calibration_seed), objective_kind, "model"),
            )
        )
        boundary._set_seed(model_seed, int(args.batch_index_seed))
        model = D0BoundarySmoothPotentialUNet(
            dynamics, base_channels=int(args.base_channels)
        ).to(device)
        chunk_size = min(int(args.batch_size), count)

        def objective_batches() -> Iterable[tuple[Tensor, int]]:
            for start in range(0, count, chunk_size):
                stop = min(count, start + chunk_size)
                states = arrays.states[start:stop].to(device)
                tau = arrays.tau[start:stop].to(device)
                labels = arrays.labels[start:stop].to(device)
                if objective_kind == "supervised_teacher":
                    loss, _ = boundary._supervised_loss(
                        model,
                        states,
                        tau,
                        arrays.tau_fraction[start:stop].to(device),
                        labels,
                        dynamics,
                    )
                elif objective_kind == "implicit_teacher":
                    probes = boundary._probe_bank(
                        probes=int(args.training_probes),
                        batch=stop - start,
                        grid_size=int(dynamics.grid_size),
                        seed=boundary._derived_seed(
                            int(args.training_probe_seed),
                            "scale-repair-calibration",
                            start,
                            stop,
                        ),
                        device=device,
                        dtype=states.dtype,
                    )
                    loss = dirichlet_score_objective(
                        model,
                        tau,
                        states,
                        labels,
                        dynamics,
                        probes,
                        create_graph=True,
                    ).loss
                else:
                    raise ValueError(f"unknown calibration objective {objective_kind!r}")
                yield loss, stop - start

        calibration = calibrate_initial_loss_scale(
            model,
            objective_batches,
            objective_kind=objective_kind,
            calibration_state_sha256=array_fingerprint(
                np.asarray(arrays.states[:count], dtype=np.float32)
            ),
            binding=binding,
            target_initial_gradient_norm=float(target),
            calibration_state_count=count,
            calibration_split="train",
        )
        record = calibration.to_record()
        record.update(
            {
                "complete": 1,
                "finite": 1,
                "shared_by_implicit_teacher_and_null": int(objective_kind == "implicit_teacher"),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
        atomic_write_json(path, record)
    gate = evaluate_loss_scale_calibration(
        record,
        expected_initial_grad_target=float(target),
        expected_state_count=count,
        expected_objective_kind=objective_kind,
    )
    return record, gate


def _record_calibration_failure(
    path: Path,
    *,
    objective_kind: str,
    arrays: boundary.ControlArrays,
    binding: Mapping[str, Any],
    target: float,
    exc: BaseException,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Commit numerical calibration failure as gate evidence, not a crash."""

    count = min(CALIBRATION_STATE_COUNT, int(arrays.states.shape[0]))
    record = {
        "schema": "d0-score-initial-gradient-loss-scale",
        "schema_version": 1,
        "objective_kind": str(objective_kind),
        "calibration_split": "train",
        "calibration_state_count": int(count),
        "calibration_state_sha256": array_fingerprint(
            np.asarray(arrays.states[:count], dtype=np.float32)
        ),
        "target_initial_gradient_norm": float(target),
        "unscaled_initial_gradient_norm": None,
        "scaled_initial_gradient_norm": None,
        "loss_scale": None,
        "unscaled_objective": None,
        "binding": dict(binding),
        "complete": 0,
        "finite": 0,
        "training_only": 1,
        "failure": {"type": type(exc).__name__, "message": str(exc)},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(path, record)
    gate = evaluate_loss_scale_calibration(
        record,
        expected_initial_grad_target=float(target),
        expected_state_count=count,
        expected_objective_kind=objective_kind,
    )
    return record, gate


def _task_fingerprints(
    *,
    task_kind: str,
    model_seed: int,
    loss_scale: float,
    train: boundary.ControlArrays,
    selection: boundary.ControlArrays,
    audit: boundary.ControlArrays,
    run_dir: Path,
    scientific_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint_value: str,
    calibration_path: Path,
) -> dict[str, Any]:
    value = boundary._task_fingerprints(
        scientific_fingerprint=scientific_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value,
        arrays=train,
        selection_arrays=selection,
        audit_arrays=audit,
        array_registry_sha256=file_fingerprint(run_dir / "synthetic_array_registry.json"),
        task_kind=task_kind,
        model_seed=int(model_seed),
        loss_scale=float(loss_scale),
    )
    value["scale_repair_schema"] = RUN_SCHEMA
    value["loss_scale_calibration"] = {
        "filename": calibration_path.name,
        "sha256": file_fingerprint(calibration_path),
    }
    return value


def _verify_saved_task_result(
    run_dir: Path,
    *,
    result: Mapping[str, Any],
    task_kind: str,
    model_seed: int,
    manifest: Mapping[str, Any],
    thresholds: BoundaryControlThresholds,
) -> dict[str, Any]:
    """Recompute a task gate and verify its checkpoint/fingerprint chain."""

    value = dict(result)
    if value.get("task_kind") != task_kind or int(value.get("model_seed", -1)) != int(model_seed):
        raise ArtifactCompatibilityError("saved task identity mismatch")
    metrics = dict(value.get("metrics", {}))
    if task_kind == "supervised_teacher":
        recomputed_gate = evaluate_supervised_teacher(metrics, thresholds)
        task_dir = run_dir / "tasks" / "supervised-teacher"
        calibration_path = run_dir / "supervised_loss_scale_calibration.json"
    elif task_kind == "implicit_teacher":
        recomputed_gate = evaluate_implicit_teacher_seed(metrics, thresholds)
        task_dir = run_dir / "tasks" / "implicit-teacher" / f"seed-{int(model_seed)}"
        calibration_path = run_dir / "implicit_loss_scale_calibration.json"
    elif task_kind == "null":
        recomputed_gate = evaluate_null_seed(metrics, thresholds)
        task_dir = run_dir / "tasks" / "null" / f"seed-{int(model_seed)}"
        calibration_path = run_dir / "implicit_loss_scale_calibration.json"
    else:
        raise ValueError(f"unknown task kind {task_kind!r}")
    if dict(value.get("gate", {})) != recomputed_gate:
        raise ArtifactCompatibilityError(f"saved {task_kind} gate does not recompute")
    if bool(int(metrics.get("complete", 0))):
        fingerprints = dict(value.get("fingerprints", {}))
        expected = {
            "scientific_fingerprint": manifest.get("scientific_fingerprint"),
            "runtime_fingerprint": manifest.get("runtime_fingerprint"),
            "source_fingerprint": manifest.get("source_fingerprint"),
            "task_kind": task_kind,
            "model_seed": int(model_seed),
            "scale_repair_schema": RUN_SCHEMA,
        }
        if any(fingerprints.get(key) != expected_value for key, expected_value in expected.items()):
            raise ArtifactCompatibilityError(f"saved {task_kind} fingerprint mismatch")
        calibration_record = _json_load(calibration_path)
        calibration_binding = dict(fingerprints.get("loss_scale_calibration", {}))
        if (
            calibration_binding.get("filename") != calibration_path.name
            or calibration_binding.get("sha256") != file_fingerprint(calibration_path)
            or not math.isclose(
                float(fingerprints.get("loss_scale", float("nan"))),
                float(calibration_record["loss_scale"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or fingerprints.get("array_registry_sha256")
            != file_fingerprint(run_dir / "synthetic_array_registry.json")
        ):
            raise ArtifactCompatibilityError(
                f"saved {task_kind} calibration or array fingerprint mismatch"
            )
        loaded = boundary._load_completed_task(task_dir, fingerprints=fingerprints)
        if loaded is None or loaded != value:
            raise ArtifactCompatibilityError(
                f"saved {task_kind} task/checkpoint chain is incomplete"
            )
    return value


def _aggregate_optimizer_health(
    results: Sequence[Mapping[str, Any]],
    thresholds: BoundaryControlThresholds,
    *,
    expected_count: int,
) -> dict[str, Any]:
    gates = [
        evaluate_optimizer_task_health(dict(result.get("metrics", {})), thresholds)
        for result in results
    ]
    complete = len(gates) == int(expected_count)
    attempted = bool(gates)
    passed = complete and all(bool(int(gate.get("passed", 0))) for gate in gates)
    return {
        "gate": "downstream_optimizer_health",
        # Once any downstream optimizer task has been attempted, an incomplete
        # task set is evidence of optimizer-pipeline failure rather than an
        # unevaluated scientific objective.  Zero attempts remain explicitly
        # not evaluated.
        "evaluation_status": "evaluated" if attempted else "not_evaluated",
        "passed": int(passed),
        "attempted": int(attempted),
        "complete_task_set": int(complete),
        "expected_task_count": int(expected_count),
        "task_count": len(gates),
        "task_gates": gates,
        "sampling_performed": 0,
    }


def _save_report(run_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / "boundary_control_gate.json", dict(report["controls"]))
    atomic_write_json(run_dir / "control_repair_decision.json", dict(report["decision"]))
    atomic_write_json(run_dir / "boundary_control_report.json", dict(report))


def _evaluate_saved_controls(
    run_dir: Path,
    *,
    provenance: Mapping[str, Any],
    preflight_gate: Mapping[str, Any],
    thresholds: BoundaryControlThresholds,
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest = _json_load(run_dir / "run_manifest.json")
    scientific_config = dict(manifest.get("scientific_config", {}))
    synthetic_config = dict(scientific_config.get("synthetic_data", {}))
    expected_calibration_count = min(
        CALIBRATION_STATE_COUNT,
        int(synthetic_config.get("train_paths", 0))
        * int(synthetic_config.get("anchors_per_path", 0)),
    )
    if expected_calibration_count <= 0:
        raise ArtifactCompatibilityError(
            "run manifest has an invalid calibration-state budget"
        )

    def calibration_gate(
        filename: str, target: float, objective_kind: str
    ) -> dict[str, Any]:
        path = run_dir / filename
        if not path.is_file():
            return not_evaluated_study(
                "loss_scale_calibration", f"{filename} was not produced"
            )
        record = _json_load(path)
        binding = dict(record.get("binding", {}))
        expected_model_seed = boundary._derived_seed(
            int(args.calibration_seed), objective_kind, "model"
        )
        expected_binding = {
            "scientific_fingerprint": manifest.get("scientific_fingerprint"),
            "runtime_fingerprint": manifest.get("runtime_fingerprint"),
            "source_fingerprint": manifest.get("source_fingerprint"),
            "objective_kind": objective_kind,
            "calibration_seed": int(args.calibration_seed),
            "model_initialization_seed": int(expected_model_seed),
            "training_probe_seed": int(args.training_probe_seed),
            "target_initial_gradient_norm": float(target),
            "calibration_state_count": expected_calibration_count,
        }
        if any(binding.get(key) != value for key, value in expected_binding.items()):
            raise ArtifactCompatibilityError(
                f"{objective_kind} calibration binding disagrees with the run manifest"
            )
        return evaluate_loss_scale_calibration(
            record,
            expected_initial_grad_target=float(target),
            expected_state_count=expected_calibration_count,
            expected_objective_kind=objective_kind,
        )

    supervised_calibration = calibration_gate(
        "supervised_loss_scale_calibration.json",
        float(args.supervised_initial_grad_target),
        "supervised_teacher",
    )
    implicit_calibration = calibration_gate(
        "implicit_loss_scale_calibration.json",
        float(args.implicit_initial_grad_target),
        "implicit_teacher",
    )
    if (run_dir / "supervised_teacher_control.json").is_file():
        supervised = _verify_saved_task_result(
            run_dir,
            result=_json_load(run_dir / "supervised_teacher_control.json"),
            task_kind="supervised_teacher",
            model_seed=int(args.supervised_seed),
            manifest=manifest,
            thresholds=thresholds,
        )
        split = split_supervised_teacher_gate(dict(supervised.get("gate", {})))
    else:
        split = {
            "optimizer": not_evaluated_study(
                "supervised_optimizer_health", "supervised task was not run"
            ),
            "representation": not_evaluated_study(
                "supervised_representation", "supervised task was not run"
            ),
        }
    teacher_study = (
        _json_load(run_dir / "implicit_teacher_study.json")
        if (run_dir / "implicit_teacher_study.json").is_file()
        else not_evaluated_study("implicit_teacher_study", "implicit teacher tasks were not run")
    )
    null_study = (
        _json_load(run_dir / "null_study.json")
        if (run_dir / "null_study.json").is_file()
        else not_evaluated_study("null_study", "null tasks were not run")
    )
    def verified_results(
        study: Mapping[str, Any], task_kind: str, expected_seeds: Sequence[int]
    ) -> list[dict[str, Any]]:
        raw = [dict(value) for value in study.get("task_results", [])]
        if len(raw) != len(expected_seeds):
            return raw
        observed = [int(value.get("model_seed", -1)) for value in raw]
        if set(observed) != set(int(seed) for seed in expected_seeds):
            raise ArtifactCompatibilityError(f"saved {task_kind} seed set mismatch")
        return [
            _verify_saved_task_result(
                run_dir,
                result=value,
                task_kind=task_kind,
                model_seed=int(value["model_seed"]),
                manifest=manifest,
                thresholds=thresholds,
            )
            for value in raw
        ]

    teacher_results = verified_results(
        teacher_study, "implicit_teacher", args.teacher_seeds
    )
    null_results = verified_results(null_study, "null", args.null_seeds)
    if teacher_study.get("evaluation_status", "evaluated") == "evaluated":
        recomputed_teacher = evaluate_implicit_teacher_study(
            [dict(result.get("metrics", {})) for result in teacher_results], thresholds
        )
        if any(teacher_study.get(key) != value for key, value in recomputed_teacher.items()):
            raise ArtifactCompatibilityError("implicit teacher study does not recompute")
        teacher_study = {
            **recomputed_teacher,
            "evaluation_status": "evaluated",
            "task_results": teacher_results,
        }
    if null_study.get("evaluation_status", "evaluated") == "evaluated":
        recomputed_null = evaluate_null_study(
            [dict(result.get("metrics", {})) for result in null_results], thresholds
        )
        if any(null_study.get(key) != value for key, value in recomputed_null.items()):
            raise ArtifactCompatibilityError("null study does not recompute")
        null_study = {
            **recomputed_null,
            "evaluation_status": "evaluated",
            "task_results": null_results,
        }

    def complete_probe_banks(result: Mapping[str, Any]) -> bool:
        banks = dict(dict(result.get("metrics", {})).get("audit_objective_banks", {}))
        if set(banks) != {"a", "b"}:
            return False
        try:
            return all(
                math.isfinite(float(dict(dict(banks[bank])[scope])["lower_bound"]))
                for bank in ("a", "b")
                for scope in ("overall", "data_end")
            )
        except (KeyError, TypeError, ValueError):
            return False

    studies_evaluated = (
        teacher_study.get("evaluation_status", "evaluated") == "evaluated"
        and null_study.get("evaluation_status", "evaluated") == "evaluated"
        and len(teacher_results) == len(args.teacher_seeds)
        and len(null_results) == len(args.null_seeds)
        and all(
            bool(int(dict(result.get("metrics", {})).get("complete", 0)))
            and bool(int(dict(result.get("metrics", {})).get("finite", 0)))
            for result in [*teacher_results, *null_results]
        )
        and all(complete_probe_banks(result) for result in [*teacher_results, *null_results])
    )
    downstream = _aggregate_optimizer_health(
        [*teacher_results, *null_results], thresholds,
        expected_count=len(args.teacher_seeds) + len(args.null_seeds),
    )
    banks_agree = (
        boundary._probe_banks_agree(
            teacher_results=teacher_results, null_results=null_results
        )
        if studies_evaluated
        else None
    )
    probe_status = classify_probe_bank_status(
        studies_evaluated=studies_evaluated, banks_agree=banks_agree
    )
    return evaluate_scale_repair_gates(
        provenance_pass=provenance,
        boundary_preflight=preflight_gate,
        supervised_calibration=supervised_calibration,
        implicit_calibration=implicit_calibration,
        supervised_optimizer=split["optimizer"],
        supervised_representation=split["representation"],
        downstream_optimizer=downstream,
        implicit_teacher_study=teacher_study,
        null_study=null_study,
        probe_bank_status=probe_status,
        require_gate=str(args.require_gate),
    )


def _run_controls(
    run_dir: Path,
    *,
    arrays: Mapping[str, boundary.ControlArrays],
    dynamics: Any,
    args: argparse.Namespace,
    device: torch.device,
    thresholds: BoundaryControlThresholds,
    scientific_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint_value: str,
    provenance: Mapping[str, Any],
    preflight_gate: Mapping[str, Any],
) -> dict[str, Any]:
    tasks_root = run_dir / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    supervised_calibration_path = run_dir / "supervised_loss_scale_calibration.json"
    implicit_calibration_path = run_dir / "implicit_loss_scale_calibration.json"
    supervised_binding = _calibration_binding(
        kind="supervised_teacher", arrays=arrays["teacher_train"], args=args,
        scientific_fingerprint=scientific_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value,
        target=float(args.supervised_initial_grad_target),
    )
    try:
        supervised_calibration_record, supervised_calibration_gate = _load_or_calibrate(
            supervised_calibration_path,
            objective_kind="supervised_teacher", arrays=arrays["teacher_train"],
            dynamics=dynamics, args=args, device=device, binding=supervised_binding,
            target=float(args.supervised_initial_grad_target),
        )
    except FloatingPointError as exc:
        supervised_calibration_record, supervised_calibration_gate = _record_calibration_failure(
            supervised_calibration_path,
            objective_kind="supervised_teacher",
            arrays=arrays["teacher_train"],
            binding=supervised_binding,
            target=float(args.supervised_initial_grad_target),
            exc=exc,
        )

    def write_calibration_gate(implicit_gate: Mapping[str, Any]) -> None:
        atomic_write_json(
            run_dir / "loss_scale_calibration_gate.json",
            {
                "schema": RUN_SCHEMA + "-loss-scale-calibration-gate",
                "schema_version": 1,
                "supervised": supervised_calibration_gate,
                "implicit_shared_teacher_null": dict(implicit_gate),
                "passed": int(
                    bool(int(supervised_calibration_gate["passed"]))
                    and bool(int(implicit_gate.get("passed", 0)))
                ),
                "sampling_performed": 0,
            },
        )

    implicit_not_evaluated = not_evaluated_study(
        "loss_scale_calibration", "awaiting a passing supervised control"
    )
    write_calibration_gate(implicit_not_evaluated)
    if not bool(int(supervised_calibration_gate["passed"])):
        supervised_result = boundary._failed_task_result(
            "supervised_teacher", int(args.supervised_seed),
            RuntimeError("supervised loss-scale calibration gate failed"),
        )
        atomic_write_json(run_dir / "supervised_teacher_control.json", supervised_result)
        teacher_study = not_evaluated_study("implicit_teacher_study", "supervised loss-scale calibration failed")
        null_study = not_evaluated_study("null_study", "supervised loss-scale calibration failed")
        atomic_write_json(run_dir / "implicit_teacher_study.json", teacher_study)
        atomic_write_json(run_dir / "null_study.json", null_study)
        atomic_write_json(
            run_dir / "task_failures.json",
            {
                "failure_count": 0,
                "failures": [],
                "skips": [
                    {
                        "stage": "supervised_and_implicit_controls",
                        "reason": "supervised loss-scale calibration failed",
                    }
                ],
            },
        )
        return _evaluate_saved_controls(
            run_dir, provenance=provenance, preflight_gate=preflight_gate,
            thresholds=thresholds, args=args,
        )

    supervised_scale = float(supervised_calibration_record["loss_scale"])
    supervised_fp = _task_fingerprints(
        task_kind="supervised_teacher", model_seed=int(args.supervised_seed),
        loss_scale=supervised_scale, train=arrays["teacher_train"],
        selection=arrays["teacher_selection"], audit=arrays["teacher_audit"],
        run_dir=run_dir, scientific_fingerprint=scientific_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value,
        calibration_path=supervised_calibration_path,
    )
    try:
        supervised_result = boundary._run_control_task(
            task_dir=tasks_root / "supervised-teacher",
            task_kind="supervised_teacher", train=arrays["teacher_train"],
            selection_arrays=arrays["teacher_selection"], audit=arrays["teacher_audit"],
            dynamics=dynamics, args=args, device=device,
            model_seed=int(args.supervised_seed), loss_scale=supervised_scale,
            fingerprints=supervised_fp, show_progress=not bool(args.no_progress),
            thresholds=thresholds,
        )
    except FloatingPointError as exc:
        supervised_result = boundary._failed_task_result(
            "supervised_teacher", int(args.supervised_seed), exc
        )
        failures.append({"task_kind": "supervised_teacher", **dict(supervised_result["failure"])})
        task_dir = tasks_root / "supervised-teacher"
        task_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(task_dir / "task_failure.json", supervised_result)
    atomic_write_json(run_dir / "supervised_teacher_control.json", supervised_result)
    supervised_split = split_supervised_teacher_gate(dict(supervised_result.get("gate", {})))
    if not (
        bool(int(supervised_split["optimizer"]["passed"]))
        and bool(int(supervised_split["representation"]["passed"]))
    ):
        reason = (
            "supervised optimizer health failed"
            if not bool(int(supervised_split["optimizer"]["passed"]))
            else "supervised analytic representation failed"
        )
        teacher_study = not_evaluated_study("implicit_teacher_study", reason)
        null_study = not_evaluated_study("null_study", reason)
        atomic_write_json(run_dir / "implicit_teacher_study.json", teacher_study)
        atomic_write_json(run_dir / "null_study.json", null_study)
        atomic_write_json(
            run_dir / "task_failures.json",
            {"failure_count": len(failures), "failures": failures, "skips": [{"stage": "implicit_controls", "reason": reason}]},
        )
        return _evaluate_saved_controls(
            run_dir, provenance=provenance, preflight_gate=preflight_gate,
            thresholds=thresholds, args=args,
        )

    implicit_binding = _calibration_binding(
        kind="implicit_teacher", arrays=arrays["teacher_train"], args=args,
        scientific_fingerprint=scientific_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value,
        target=float(args.implicit_initial_grad_target),
    )
    try:
        implicit_calibration_record, implicit_calibration_gate = _load_or_calibrate(
            implicit_calibration_path,
            objective_kind="implicit_teacher", arrays=arrays["teacher_train"],
            dynamics=dynamics, args=args, device=device, binding=implicit_binding,
            target=float(args.implicit_initial_grad_target),
        )
    except FloatingPointError as exc:
        implicit_calibration_record, implicit_calibration_gate = _record_calibration_failure(
            implicit_calibration_path,
            objective_kind="implicit_teacher",
            arrays=arrays["teacher_train"],
            binding=implicit_binding,
            target=float(args.implicit_initial_grad_target),
            exc=exc,
        )
    write_calibration_gate(implicit_calibration_gate)
    if not bool(int(implicit_calibration_gate["passed"])):
        reason = "implicit loss-scale calibration failed"
        teacher_study = not_evaluated_study("implicit_teacher_study", reason)
        null_study = not_evaluated_study("null_study", reason)
        atomic_write_json(run_dir / "implicit_teacher_study.json", teacher_study)
        atomic_write_json(run_dir / "null_study.json", null_study)
        atomic_write_json(
            run_dir / "task_failures.json",
            {
                "failure_count": len(failures),
                "failures": failures,
                "skips": [{"stage": "implicit_controls", "reason": reason}],
            },
        )
        return _evaluate_saved_controls(
            run_dir, provenance=provenance, preflight_gate=preflight_gate,
            thresholds=thresholds, args=args,
        )
    implicit_scale = float(implicit_calibration_record["loss_scale"])

    advisory_binding = {
        "scientific_fingerprint": scientific_fingerprint,
        "null_train_identity": boundary._arrays_identity(arrays["null_train"]),
        "null_audit_identity": boundary._arrays_identity(arrays["null_audit"]),
        "probe_seed": int(args.audit_probe_a_seed),
        "role": "advisory_only",
    }
    try:
        boundary._null_linear_advisory(
            run_dir,
            train=arrays["null_train"],
            audit=arrays["null_audit"],
            dynamics=dynamics,
            args=args,
            device=device,
            binding=advisory_binding,
        )
    except Exception as exc:
        atomic_write_json(
            run_dir / "advisory" / "null_linear_spline_warning.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "role": "advisory_only",
                "eligible_for_gate": 0,
                "sampling_performed": 0,
            },
        )

    teacher_results: list[dict[str, Any]] = []
    null_results: list[dict[str, Any]] = []
    for task_kind, seeds, train_name, selection_name, audit_name, output in (
        (
            "implicit_teacher", args.teacher_seeds, "teacher_train",
            "teacher_selection", "teacher_audit", teacher_results,
        ),
        (
            "null", args.null_seeds, "null_train",
            "null_selection", "null_audit", null_results,
        ),
    ):
        for seed in seeds:
            task_dir = tasks_root / ("implicit-teacher" if task_kind == "implicit_teacher" else "null") / f"seed-{int(seed)}"
            fingerprints = _task_fingerprints(
                task_kind=task_kind, model_seed=int(seed), loss_scale=implicit_scale,
                train=arrays[train_name], selection=arrays[selection_name],
                audit=arrays[audit_name], run_dir=run_dir,
                scientific_fingerprint=scientific_fingerprint,
                runtime_fingerprint=runtime_fingerprint,
                source_fingerprint_value=source_fingerprint_value,
                calibration_path=implicit_calibration_path,
            )
            try:
                result = boundary._run_control_task(
                    task_dir=task_dir, task_kind=task_kind,
                    train=arrays[train_name], selection_arrays=arrays[selection_name],
                    audit=arrays[audit_name], dynamics=dynamics, args=args, device=device,
                    model_seed=int(seed), loss_scale=implicit_scale,
                    fingerprints=fingerprints, show_progress=not bool(args.no_progress),
                    thresholds=thresholds,
                )
            except FloatingPointError as exc:
                result = boundary._failed_task_result(task_kind, int(seed), exc)
                failures.append({"task_kind": task_kind, "model_seed": int(seed), **dict(result["failure"])})
                task_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(task_dir / "task_failure.json", result)
            output.append(result)

    teacher_study = evaluate_implicit_teacher_study(
        [dict(result["metrics"]) for result in teacher_results], thresholds
    )
    null_study = evaluate_null_study(
        [dict(result["metrics"]) for result in null_results], thresholds
    )
    teacher_study.update(
        {"evaluation_status": "evaluated", "task_results": teacher_results}
    )
    null_study.update(
        {"evaluation_status": "evaluated", "task_results": null_results}
    )
    atomic_write_json(run_dir / "implicit_teacher_study.json", teacher_study)
    atomic_write_json(run_dir / "null_study.json", null_study)
    atomic_write_json(
        run_dir / "task_failures.json",
        {"failure_count": len(failures), "failures": failures, "skips": []},
    )
    return _evaluate_saved_controls(
        run_dir, provenance=provenance, preflight_gate=preflight_gate,
        thresholds=thresholds, args=args,
    )


def _pending_preflight_report(
    *,
    provenance: Mapping[str, Any],
    preflight_gate: Mapping[str, Any],
    require_gate: str,
) -> dict[str, Any]:
    passed = bool(int(provenance.get("passed", 0))) and bool(int(preflight_gate.get("passed", 0)))
    required_pass = True if require_gate == "none" else (passed if require_gate == "preflight" else False)
    return {
        "schema": RUN_SCHEMA + "-gate-report",
        "schema_version": 1,
        "required_gate": str(require_gate),
        "required_gate_pass": int(required_pass),
        "preflight_pass": int(passed),
        "controls": not_evaluated_study("scale_repair_controls", "controls stage not run"),
        "decision": {
            "decision": "controls_not_run",
            "recommended_next_action": "resume this run with --stage all",
            "probe_bank_status": ProbeBankStatus.NOT_EVALUATED.value,
            "physical_training_authorized": 0,
            "sampling_authorized": 0,
            "sampling_performed": 0,
        },
        "sampling_performed": 0,
    }


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(run_dir).as_posix(): boundary._artifact_record(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "records": records,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "sampling_performed": 0,
    }


def _verify_terminal_registry(run_dir: Path) -> dict[str, Any]:
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    if not registry_path.is_file() or not status_path.is_file():
        raise ArtifactCompatibilityError("report requires a terminal registry and status")
    status = _json_load(status_path)
    if (
        status.get("artifact_registry_sha256") != file_fingerprint(registry_path)
        or int(status.get("artifact_registry_size", -1)) != int(registry_path.stat().st_size)
    ):
        raise ArtifactCompatibilityError("terminal status does not bind the artifact registry")
    registry = _json_load(registry_path)
    if registry.get("schema") != RUN_SCHEMA + "-artifact-registry":
        raise ArtifactCompatibilityError("artifact registry schema mismatch")
    records = dict(registry.get("records", {}))
    excluded = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    if set(records) != actual:
        raise ArtifactCompatibilityError("artifact registry file set mismatch")
    for relative, record_value in records.items():
        path = run_dir / relative
        record = dict(record_value)
        if (
            not path.is_file()
            or record.get("sha256") != file_fingerprint(path)
            or int(record.get("size", -1)) != int(path.stat().st_size)
        ):
            raise ArtifactCompatibilityError(f"artifact registry hash mismatch: {relative}")
    return registry


def _finish(
    run_dir: Path,
    *,
    report: Mapping[str, Any],
    phase: str,
    stage: str,
    skips: Sequence[Mapping[str, Any]],
    execution_failed: bool = False,
) -> int:
    final_skips = [dict(value) for value in skips]
    required_pass = 0 if execution_failed else int(report.get("required_gate_pass", 0))
    try:
        boundary._write_report_artifacts(run_dir)
    except Exception as exc:
        execution_failed = True
        required_pass = 0
        final_skips.append({"stage": "report_artifacts", "reason": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(
            run_dir / "report_artifact_failure.json",
            {"type": type(exc).__name__, "message": str(exc), "sampling_performed": 0},
        )
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _json_load(manifest_path)
        manifest["artifacts"] = {
            "preflight": str((run_dir / "boundary_preflight_gate.json").resolve()) if (run_dir / "boundary_preflight_gate.json").is_file() else None,
            "calibrations": str((run_dir / "loss_scale_calibration_gate.json").resolve()) if (run_dir / "loss_scale_calibration_gate.json").is_file() else None,
            "controls": str((run_dir / "boundary_control_gate.json").resolve()) if (run_dir / "boundary_control_gate.json").is_file() else None,
            "decision": str((run_dir / "control_repair_decision.json").resolve()) if (run_dir / "control_repair_decision.json").is_file() else None,
            "artifact_registry": str((run_dir / "artifact_registry.json").resolve()),
        }
        atomic_write_json(manifest_path, manifest)
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    decision = dict(report.get("decision", {}))
    _write_status(
        run_dir,
        status="failed" if execution_failed else "complete",
        outcome="implementation_error" if execution_failed else ("complete" if required_pass else "gate_failed"),
        phase=str(phase),
        stage=str(stage),
        required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=int(required_pass),
        decision=str(decision.get("decision", "controls_not_run")),
        recommended_next_action=decision.get("recommended_next_action"),
        probe_bank_status=decision.get("probe_bank_status", ProbeBankStatus.NOT_EVALUATED.value),
        skips=final_skips,
        physical_training_authorized=0 if execution_failed else int(decision.get("physical_training_authorized", 0)),
        physical_training_performed=0,
        sampling_authorized=0,
        sampling_performed=0,
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_size=int(registry_path.stat().st_size),
    )
    return 2 if execution_failed else (0 if required_pass else 2)


def _failure_report(require_gate: str) -> dict[str, Any]:
    skipped = not_evaluated_study("study", "workflow failed before controls")
    return evaluate_scale_repair_gates(
        provenance_pass=False,
        boundary_preflight=False,
        supervised_calibration=False,
        implicit_calibration=False,
        supervised_optimizer=False,
        supervised_representation=False,
        downstream_optimizer=False,
        implicit_teacher_study=skipped,
        null_study=skipped,
        probe_bank_status=ProbeBankStatus.NOT_EVALUATED,
        require_gate=require_gate,
    )


def _failed_preflight_report(
    *,
    provenance: Mapping[str, Any],
    preflight_gate: Mapping[str, Any],
    require_gate: str,
) -> dict[str, Any]:
    skipped = not_evaluated_study("study", "boundary/operator preflight failed")
    return evaluate_scale_repair_gates(
        provenance_pass=provenance,
        boundary_preflight=preflight_gate,
        supervised_calibration=False,
        implicit_calibration=False,
        supervised_optimizer=False,
        supervised_representation=False,
        downstream_optimizer=False,
        implicit_teacher_study=skipped,
        null_study=skipped,
        probe_bank_status=ProbeBankStatus.NOT_EVALUATED,
        require_gate=require_gate,
    )


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = boundary._make_run_dir(args)
    print(f"scale-repair run directory: {run_dir.resolve()}", flush=True)
    thresholds = BoundaryControlThresholds(
        bootstrap_confidence=float(args.bootstrap_confidence)
    )
    mutation_started = False
    try:
        device = boundary._device(args.device)
        backend = boundary.configure_exact_torch_backend(device)
        runtime = boundary._runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
        source_hash, source_paths = _source_record()
        parent = verify_parent_boundary_control_run(args.parent_boundary_control_run_dir)
        provenance = {
            "schema": RUN_SCHEMA + "-parent-provenance",
            "schema_version": 1,
            **parent,
            "verifier_source_fingerprint": source_hash,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        provenance_path = run_dir / "parent_provenance.json"
        if provenance_path.is_file():
            if _json_load(provenance_path) != provenance:
                raise ArtifactCompatibilityError("resume parent provenance mismatch")
        else:
            if resumed:
                raise ArtifactCompatibilityError("resume is missing parent provenance")
            atomic_write_json(provenance_path, provenance)

        scientific = _scientific_config(args, parent=parent, thresholds=thresholds)
        scientific_fingerprint = config_fingerprint(scientific)
        manifest_value = {
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
                "schema", "schema_version", "scientific_config",
                "scientific_fingerprint", "runtime", "runtime_fingerprint",
                "source_fingerprint", "source_paths", "parent_provenance_sha256",
                "claim_scope",
            ):
                if existing.get(key) != manifest_value.get(key):
                    raise ArtifactCompatibilityError(f"resume manifest mismatch for {key}")
        else:
            if resumed:
                raise ArtifactCompatibilityError("resume is missing its frozen manifest")
            atomic_write_json(manifest_path, manifest_value)

        previous = _json_load(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
        if resumed and (
            str(previous.get("status", "")) in {"complete", "failed"}
            or str(args.stage) == "report"
        ):
            # A terminal resume first proves every prior artifact byte.  This
            # prevents an `all` resume from silently re-blessing a modified
            # preflight, task CSV, checkpoint, or report artifact.
            _verify_terminal_registry(run_dir)
        _write_status(
            run_dir, status="running", phase="provenance", stage=str(args.stage),
            require_gate=str(args.require_gate), attempt_count=int(previous.get("attempt_count", 0)) + 1,
        )
        mutation_started = True

        preflight_path = run_dir / "boundary_preflight_gate.json"
        if str(args.stage) == "report":
            preflight_gate = _json_load(preflight_path)
            if not bool(int(preflight_gate.get("passed", 0))):
                report = _failed_preflight_report(
                    provenance=provenance,
                    preflight_gate=preflight_gate,
                    require_gate=str(args.require_gate),
                )
            elif not (run_dir / "supervised_teacher_control.json").is_file():
                report = _pending_preflight_report(
                    provenance=provenance,
                    preflight_gate=preflight_gate,
                    require_gate=str(args.require_gate),
                )
            else:
                report = _evaluate_saved_controls(
                    run_dir, provenance=provenance, preflight_gate=preflight_gate,
                    thresholds=thresholds, args=args,
                )
            _save_report(run_dir, report)
            return _finish(run_dir, report=report, phase="report", stage="report", skips=[])

        dynamics = boundary._make_dynamics(args)
        if not math.isclose(
            float(parent["schedule_metadata"]["horizon"]),
            float(natural_horizon(dynamics)), rel_tol=1e-12, abs_tol=1e-18,
        ):
            raise ArtifactCompatibilityError("parent and scale-repair model horizons differ")
        preflight_binding = {
            "scientific_fingerprint": scientific_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "source_fingerprint": source_hash,
            "failed_run_status_sha256": dict(parent["artifacts"])["status"]["sha256"],
        }
        _write_status(run_dir, status="running", phase="preflight")
        _, preflight_gate = boundary._run_preflight(
            run_dir, dynamics=dynamics, args=args, device=device,
            binding=preflight_binding, thresholds=thresholds,
        )
        if str(args.stage) == "preflight" or not bool(int(preflight_gate.get("passed", 0))):
            report = _pending_preflight_report(
                provenance=provenance, preflight_gate=preflight_gate,
                require_gate=str(args.require_gate),
            )
            if not bool(int(preflight_gate.get("passed", 0))):
                report = _failed_preflight_report(
                    provenance=provenance,
                    preflight_gate=preflight_gate,
                    require_gate=str(args.require_gate),
                )
            _save_report(run_dir, report)
            skips = [] if bool(int(preflight_gate.get("passed", 0))) else [{"stage": "controls", "reason": "boundary/operator preflight failed"}]
            return _finish(run_dir, report=report, phase="preflight", stage=str(args.stage), skips=skips)

        _write_status(run_dir, status="running", phase="synthetic_data")
        arrays, plans = boundary._prepare_control_arrays(
            run_dir, args=args, parent=parent,
            scientific_fingerprint=scientific_fingerprint, resumed=resumed,
        )
        atomic_write_json(
            run_dir / "control_plan_registry.json",
            {
                "schema": RUN_SCHEMA + "-control-plan-registry",
                "schema_version": 1,
                "time_plan_fingerprint": plans["time"]["fingerprint"],
                "split_plan_fingerprint": plans["split"]["fingerprint"],
                "probe_plan_fingerprint": plans["probes"]["fingerprint"],
                "synthetic_array_registry_sha256": file_fingerprint(run_dir / "synthetic_array_registry.json"),
                "fresh_parent_states_reused": 0,
                "sampling_performed": 0,
            },
        )
        _write_status(run_dir, status="running", phase="controls")
        report = _run_controls(
            run_dir, arrays=arrays, dynamics=dynamics, args=args, device=device,
            thresholds=thresholds, scientific_fingerprint=scientific_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            source_fingerprint_value=source_hash, provenance=provenance,
            preflight_gate=preflight_gate,
        )
        _save_report(run_dir, report)
        return _finish(run_dir, report=report, phase="controls", stage=str(args.stage), skips=[])
    except Exception as exc:
        if resumed and not mutation_started:
            if not bool(args.no_progress):
                print(
                    f"scale-repair resume rejected without mutation: {type(exc).__name__}: {exc}",
                    file=sys.stderr, flush=True,
                )
            return 2
        atomic_write_json(
            run_dir / "failure.json",
            {
                "schema": RUN_SCHEMA + "-failure", "schema_version": 1,
                "type": type(exc).__name__, "message": str(exc),
                "stage": str(args.stage), "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )
        report = _failure_report(str(args.require_gate))
        _save_report(run_dir, report)
        _finish(
            run_dir, report=report, phase="failure", stage=str(args.stage),
            skips=[{"stage": "remaining", "reason": f"{type(exc).__name__}: {exc}"}],
            execution_failed=True,
        )
        if not bool(args.no_progress):
            print(f"scale repair failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
