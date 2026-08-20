"""Read-only adjudication of an absolute-coordinate Jacobi/RB hypothesis.

The workflow verifies the immutable terminal directional-result archive and
the independent physical coarse-witness run, audits the frozen predictor's
translation symmetry, and evaluates one Panel-A-sealed frequency-one
coordinate direction on Panel B.  It creates no paths or transitions and
performs no optimization, confirmation, controller execution, reconstruction,
or sampling.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_absolute_coordinate import (
    ABSOLUTE_COORDINATE_VERSION,
    COORDINATE_COMPONENTS,
    DEFAULT_BOOTSTRAP_CHUNK_SIZE,
    DEFAULT_BOOTSTRAP_NAMESPACE,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    AbsoluteCoordinateError,
    CoordinatePanel,
    build_coordinate_lattice,
    decompose_cross_panel_signal,
    evaluate_panel_b_linear,
    model_translation_equivariance_record,
    one_sided_b_linear_max_t,
    phase_predictor_architecture_contract,
    scaled_signed_cross_bounds,
    seal_panel_a_directions,
    synthetic_coordinate_fixture,
    synthetic_model_inputs,
)
from mnist.d0_jacobi_rb_absolute_coordinate_gate import (
    decide_absolute_coordinate,
    decision_exit_code,
    safety_record,
)
from mnist.d0_jacobi_rb_absolute_coordinate_provenance import (
    AbsoluteCoordinateProvenanceError,
    snapshot_absolute_coordinate_parents,
    verify_absolute_coordinate_parent_immutability,
    verify_absolute_coordinate_parents,
)
from mnist.d0_jacobi_rb_learnability import PhaseConditionedLocalAffineCNN


RUN_SCHEMA = "experiment12-d0-jacobi-rb-absolute-coordinate-adjudication"
STAGES = ("preflight", "replay", "symmetry", "decompose", "report", "all")
REQUIRED_GATES = ("none", "preflight", "replay", "symmetry", "decompose")
EXPECTED_COARSE_POINT = 0.000648424870102139
COARSE_POINT_TOLERANCE = 5.0e-18
PROJECTION_RECONSTRUCTION_TOLERANCE = 5.0e-15
MODEL_EQUIVARIANCE_TOLERANCE = 2.0e-6
PRIMARY_COMPONENT = "frequency1"
PRIMARY_FAMILY_NAMES = tuple(f"q{quartile}.{PRIMARY_COMPONENT}" for quartile in range(4))

NO_WORK = {
    "new_transitions_generated": 0,
    "new_path_ids": 0,
    "fresh_physical_labels_opened": 0,
    "optimizer_updates_performed": 0,
    "physical_training_performed": 0,
    "new_learner_training_performed": 0,
    "new_checkpoints_created": 0,
    "fresh_fit_performed": 0,
    "fresh_calibration_performed": 0,
    "fresh_selection_performed": 0,
    "confirmation_performed": 0,
    "confirmation_evidence_accessed": 0,
    "controller_trajectories_executed": 0,
    "reconstructions_created": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "parent_files_modified": 0,
}


class AbsoluteCoordinateWorkflowError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "absolute_coordinate_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = failure_domain
        self.failure_code = failure_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope() -> dict[str, int]:
    return {**NO_WORK, **safety_record()}


def _scope_arrays() -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value, dtype=np.int8) for name, value in _scope().items()
    }


def _semantic(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _verify_semantic(record: Mapping[str, Any], description: str) -> None:
    body = dict(record)
    observed = body.pop("semantic_sha256", None)
    if observed != config_fingerprint(body):
        raise ArtifactCompatibilityError(f"{description} semantic hash changed")


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact: {target}") from exc
    if not isinstance(value, dict):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {target}")
    return value


def _optional_json(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / name
    return _load_json(path) if path.is_file() else None


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                **{name: np.ascontiguousarray(value) for name, value in arrays.items()},
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": target.name,
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _load_panel(run_dir: Path, role: str) -> CoordinatePanel:
    if role not in {"a", "b"}:
        raise AbsoluteCoordinateWorkflowError("unknown witness panel role")
    path = run_dir / "panels" / role / "cell_means.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            cells = np.array(archive["cell_means"], dtype=np.float64, copy=True, order="C")
            path_ids = np.array(archive["path_ids"], dtype=np.int64, copy=True, order="C")
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactCompatibilityError(f"cannot load immutable panel {role}") from exc
    return CoordinatePanel(
        role=f"physical-panel-{role}", path_ids=path_ids, cell_means=cells
    )


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _gate(schema: str, checks: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, int] = {}
    for name, value in checks.items():
        if not isinstance(value, (bool, int)) or int(value) not in (0, 1):
            raise AbsoluteCoordinateWorkflowError(f"gate check {name} is not a bit")
        normalized[str(name)] = int(value)
    return {
        "schema": schema,
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "checks": normalized,
        "passed": int(bool(normalized) and all(normalized.values())),
        **_scope(),
    }


def _not_evaluated_gate(name: str) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + f"-{name}-gate",
        "schema_version": 1,
        "evaluation_status": "not_evaluated",
        "passed": 0,
        **_scope(),
    }


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    message: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
    scientific_evidence_complete: int = 0,
) -> None:
    atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": RUN_SCHEMA + "-status",
            "schema_version": 1,
            "state": state,
            "stage": stage,
            "decision": decision,
            "message": message,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "scientific_evidence_complete": int(scientific_evidence_complete),
            "updated_at": _now(),
            **_scope(),
        },
    )


def _seal_stage(run_dir: Path, names: Sequence[str], seal_name: str) -> dict[str, Any]:
    artifacts = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"stage artifact is missing: {name}")
        artifacts.append(
            {"path": name, "size": int(path.stat().st_size), "sha256": file_fingerprint(path)}
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-stage-seal",
            "schema_version": 1,
            "artifacts": artifacts,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / seal_name, record)
    return record


def _verify_stage_seal(run_dir: Path, seal_name: str) -> None:
    seal = _load_json(run_dir / seal_name)
    _verify_semantic(seal, seal_name)
    for row in seal.get("artifacts", []):
        path = run_dir / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size"])
            or file_fingerprint(path) != row["sha256"]
        ):
            raise ArtifactCompatibilityError(f"sealed artifact changed: {path}")


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(run_dir.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name == "artifact_registry.json" or ".tmp" in path.name:
            continue
        artifacts.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-artifact-registry",
            "schema_version": 1,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_registered_prefix(run_dir: Path) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    registry = _load_json(path)
    _verify_semantic(registry, "child artifact registry")
    for row in registry.get("artifacts", []):
        target = run_dir / str(row["path"])
        if (
            not target.is_file()
            or target.stat().st_size != int(row["size"])
            or file_fingerprint(target) != row["sha256"]
        ):
            raise ArtifactCompatibilityError(f"registered child artifact changed: {target}")


def _source_set() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parent
    return tuple(
        sorted(
            (
                Path(__file__).resolve(),
                root / "d0_jacobi_artifacts.py",
                root / "d0_jacobi_rb_absolute_coordinate.py",
                root / "d0_jacobi_rb_absolute_coordinate_gate.py",
                root / "d0_jacobi_rb_absolute_coordinate_provenance.py",
                root / "d0_jacobi_rb_learnability.py",
            ),
            key=lambda value: value.as_posix(),
        )
    )


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    body = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "absolute_coordinate_version": ABSOLUTE_COORDINATE_VERSION,
        "portable_directional_archive": str(args.parent_directional_result_archive),
        "coarse_witness_run_dir": str(args.parent_coarse_witness_run_dir),
        "coordinate_components": list(COORDINATE_COMPONENTS),
        "primary_component": PRIMARY_COMPONENT,
        "primary_family_names": list(PRIMARY_FAMILY_NAMES),
        "confidence": 0.99,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_namespace": int(DEFAULT_BOOTSTRAP_NAMESPACE),
        "bootstrap_quantile_interpolation": "higher",
        "bootstrap_unit": "whole_panel_b_path_jointly_across_quartiles",
        "panel_a_direction_sealed_before_panel_b_inference": 1,
        "negative_estimates_truncated": 0,
        "projection_reconstruction_tolerance": PROJECTION_RECONSTRUCTION_TOLERANCE,
        "model_equivariance_tolerance": MODEL_EQUIVARIANCE_TOLERANCE,
        "historical_post_hoc_evidence_authorizing": 0,
        "device": args.device,
        **_scope(),
    }
    return _semantic(body)


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.runs_root / f"{stamp}_{args.run_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, False


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    config = _scientific_config(args)
    sources = _source_set()
    source_hash = source_fingerprint(sources)
    if resumed:
        if _load_json(run_dir / "scientific_config.json") != config:
            raise ArtifactCompatibilityError("resume scientific configuration changed")
        manifest = _load_json(run_dir / "run_manifest.json")
        if (
            manifest.get("source_fingerprint") != source_hash
            or manifest.get("scientific_config_sha256") != config["semantic_sha256"]
        ):
            raise ArtifactCompatibilityError("resume source or configuration binding changed")
        _verify_registered_prefix(run_dir)
        return
    atomic_write_json(run_dir / "scientific_config.json", config)
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema": RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "created_by": "mnist.diag_d0_jacobi_rb_absolute_coordinate_adjudication",
            "source_fingerprint": source_hash,
            "source_paths": [str(path) for path in sources],
            "scientific_config_sha256": config["semantic_sha256"],
            "stage_contract": list(STAGES[:-1]),
            "required_gates": list(REQUIRED_GATES),
            "device": args.device,
            **_scope(),
        },
    )


def _hypothesis_plan(run_dir: Path, lattice_record: Mapping[str, Any]) -> dict[str, Any]:
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-hypothesis-plan",
            "schema_version": 1,
            "frozen_before_panel_array_loading": 1,
            "question": (
                "does a Panel-A-sealed periodic frequency-one absolute-edge direction "
                "transfer to independent Panel B in q0 through q3"
            ),
            "coordinate_components": list(COORDINATE_COMPONENTS),
            "primary_component": PRIMARY_COMPONENT,
            "primary_family_names": list(PRIMARY_FAMILY_NAMES),
            "panel_a_role": "physical-panel-a",
            "panel_b_role": "physical-panel-b",
            "confidence": 0.99,
            "bootstrap_replicates": int(
                _load_json(run_dir / "scientific_config.json")["bootstrap_replicates"]
            ),
            "bootstrap_seed": int(
                _load_json(run_dir / "scientific_config.json")["bootstrap_seed"]
            ),
            "quantile_interpolation": "higher",
            "lattice_semantic_sha256": lattice_record["semantic_sha256"],
            "negative_estimates_truncated": 0,
            "historical_post_hoc_evidence_authorizing": 0,
            "claim_restriction": (
                "feature-family hypothesis only; neither unique architecture nor "
                "impossibility of content-derived position is identified"
            ),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "coordinate_hypothesis_plan.json", record)
    return record


def _completed_stage(run_dir: Path, gate_name: str, seal_name: str) -> bool:
    gate_path = run_dir / gate_name
    seal_path = run_dir / seal_name
    if not gate_path.is_file() and not seal_path.is_file():
        return False
    if not gate_path.is_file():
        raise ArtifactCompatibilityError(f"incomplete committed stage: {gate_name}")
    gate = _load_json(gate_path)
    if gate.get("evaluation_status") == "evaluated":
        if not seal_path.is_file():
            raise ArtifactCompatibilityError(f"evaluated stage lacks its seal: {gate_name}")
        _verify_stage_seal(run_dir, seal_name)
        return True
    if gate.get("evaluation_status") != "execution_failed":
        raise ArtifactCompatibilityError(f"malformed stage gate: {gate_name}")
    stage = {
        "preflight_gate.json": "preflight",
        "replay_gate.json": "replay",
        "symmetry_gate.json": "symmetry",
        "decomposition_gate.json": "decompose",
    }[gate_name]
    attempts_root = run_dir / "execution_attempts" / stage
    attempts = sorted(attempts_root.glob("attempt-*")) if attempts_root.is_dir() else []
    attempt = attempts_root / f"attempt-{len(attempts) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    archived: list[dict[str, Any]] = []
    retry_names = (
        f"{stage}_execution_failure.json",
        gate_name,
        seal_name,
        "absolute_coordinate_decision_evidence.json",
        "absolute_coordinate_adjudication_decision.json",
        "workflow_gate.json",
    )
    for name in retry_names:
        source = run_dir / name
        if source.is_file():
            target = attempt / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            archived.append({"path": name, "sha256": file_fingerprint(target)})
            source.unlink()
    atomic_write_json(
        attempt / "retry_authorization.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-execution-retry",
                "schema_version": 1,
                "stage": stage,
                "attempt": len(attempts) + 1,
                "archived": archived,
                "execution_failure_reopened": 1,
                "scientific_gate_reopened": 0,
                **_scope(),
            }
        ),
    )
    (run_dir / "artifact_registry.json").unlink(missing_ok=True)
    return False


def _parent_snapshots_from_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshots = record.get("parent_snapshots")
    if not isinstance(snapshots, Mapping):
        raise ArtifactCompatibilityError("parent snapshot binding is missing")
    return snapshots


def _verify_parent_immutability(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    provenance = _load_json(run_dir / "parent_provenance.json")
    result = verify_absolute_coordinate_parent_immutability(
        portable_zip_path=args.parent_directional_result_archive,
        coarse_witness_run_dir=args.parent_coarse_witness_run_dir,
        snapshots=_parent_snapshots_from_record(provenance),
    )
    if int(result.get("passed", result.get("parent_files_modified", 1) == 0)) != 1:
        raise ArtifactCompatibilityError("immutable parent changed after preflight")
    return result


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "preflight_gate.json", "preflight_artifact_seal.json"):
        return
    lattice = build_coordinate_lattice()
    lattice_record = lattice.to_record()
    lattice_body = dict(lattice_record)
    lattice_body.pop("semantic_sha256", None)
    lattice_artifact = _semantic({**lattice_body, **_scope()})
    atomic_write_json(run_dir / "coordinate_lattice.json", lattice_artifact)
    plan = _hypothesis_plan(run_dir, lattice_artifact)
    snapshots = snapshot_absolute_coordinate_parents(
        portable_zip_path=args.parent_directional_result_archive,
        coarse_witness_run_dir=args.parent_coarse_witness_run_dir,
    )
    provenance = verify_absolute_coordinate_parents(
        portable_zip_path=args.parent_directional_result_archive,
        coarse_witness_run_dir=args.parent_coarse_witness_run_dir,
        snapshots=snapshots,
    )
    provenance_body = dict(provenance)
    provenance_body.pop("semantic_sha256", None)
    provenance = _semantic(
        {**provenance_body, "parent_snapshots": snapshots, **_scope()}
    )
    atomic_write_json(run_dir / "parent_provenance.json", provenance)
    atomic_write_json(
        run_dir / "parent_immutability_report.json",
        {
            "schema": RUN_SCHEMA + "-parent-immutability",
            "schema_version": 1,
            "portable_archive_snapshot": snapshots.get("portable_directional"),
            "coarse_witness_snapshot": snapshots.get("coarse_witness"),
            "parents_mutated": 0,
            **_scope(),
        },
    )
    input_contract = {
        "schema": RUN_SCHEMA + "-input-contract",
        "schema_version": 1,
        "permitted_conditioning": [
            "later_full_state",
            "reverse_time",
            "phase",
            "color",
            "duration",
            "label",
            "output_edge_identity_for_diagnostic_only",
        ],
        "forbidden_inputs": [
            "earlier_state",
            "outer_step_audit_field",
            "path_identity",
            "random_bits",
            "certificate_metadata",
            "oracle_quantity",
        ],
        "absolute_edge_is_output_site_coarsening": 1,
        "historical_diagnostic_only": 1,
        **_scope(),
    }
    atomic_write_json(run_dir / "model_input_contract.json", input_contract)
    checks = {
        "control_provenance_valid": int(provenance.get("provenance_valid", 0)),
        "portable_directional_parent_valid": int(
            provenance.get("portable_directional_parent_valid", 0)
        ),
        "coarse_witness_parent_valid": int(provenance.get("coarse_witness_parent_valid", 0)),
        "coordinate_hypothesis_plan_valid": int(
            plan.get("frozen_before_panel_array_loading", 0) == 1
            and plan.get("primary_family_names") == list(PRIMARY_FAMILY_NAMES)
        ),
        "coordinate_lattice_valid": int(
            lattice.maximum_gram_error <= PROJECTION_RECONSTRUCTION_TOLERANCE
        ),
        "read_only_scope_valid": int(all(value == 0 for value in NO_WORK.values())),
    }
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "checks": checks,
        "lattice_basis_sha256": lattice.basis_sha256,
        "maximum_gram_error": lattice.maximum_gram_error,
        **_scope(),
    }
    gate = _gate(RUN_SCHEMA + "-preflight-gate", checks)
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "coordinate_lattice.json",
            "coordinate_hypothesis_plan.json",
            "parent_provenance.json",
            "parent_immutability_report.json",
            "model_input_contract.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )


def _replay_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "replay_gate.json", "replay_artifact_seal.json"):
        return
    _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
    immutability = _verify_parent_immutability(run_dir, args)
    analysis = _load_json(args.parent_coarse_witness_run_dir / "physical_coarse_signal_analysis.json")
    decision = _load_json(args.parent_coarse_witness_run_dir / "physical_coarse_signal_decision.json")
    bootstrap = analysis.get("bootstrap", {})
    classification = analysis.get("classification", {})
    point = float(bootstrap.get("point_estimate", float("nan")))
    checks = {
        "immutable_parents_unchanged": int(
            immutability.get("passed", int(immutability.get("parent_files_modified", 1) == 0))
        ),
        "coarse_decision_reproduced": int(
            decision.get("decision") == "exact_physical_coarse_signal_detected"
            and classification.get("decision") == "exact_physical_coarse_signal_detected"
        ),
        "coarse_point_reproduced": int(
            np.isfinite(point) and abs(point - EXPECTED_COARSE_POINT) <= COARSE_POINT_TOLERANCE
        ),
        "coarse_lower_bound_positive": int(
            float(bootstrap.get("lower_bound", float("nan"))) > 0.0
        ),
        "panel_shapes_bound": int(
            analysis.get("left_panel", {}).get("shape") == [64, 4, 7, 392]
            and analysis.get("right_panel", {}).get("shape") == [64, 4, 7, 392]
        ),
        "panel_roles_disjoint": int(
            not set(analysis.get("left_panel", {}).get("path_ids", ()))
            .intersection(analysis.get("right_panel", {}).get("path_ids", ()))
        ),
        "panel_arrays_unopened_by_replay": 1,
    }
    record = {
        "schema": RUN_SCHEMA + "-coarse-witness-replay",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "point_estimate": point,
        "expected_point_estimate": EXPECTED_COARSE_POINT,
        "point_error": abs(point - EXPECTED_COARSE_POINT),
        "bootstrap_lower_bound": bootstrap.get("lower_bound"),
        "bootstrap_upper_bound": bootstrap.get("upper_bound"),
        "panel_a_path_count": analysis.get("left_panel", {}).get("path_count"),
        "panel_b_path_count": analysis.get("right_panel", {}).get("path_count"),
        "panel_arrays_deserialized": 0,
        "checks": checks,
        **_scope(),
    }
    gate = _gate(RUN_SCHEMA + "-replay-gate", checks)
    atomic_write_json(run_dir / "coarse_witness_replay.json", record)
    atomic_write_json(run_dir / "replay_metrics.json", record)
    atomic_write_json(run_dir / "replay_gate.json", gate)
    _seal_stage(
        run_dir,
        ("coarse_witness_replay.json", "replay_metrics.json", "replay_gate.json"),
        "replay_artifact_seal.json",
    )


def _synthetic_symmetry_controls(lattice: Any) -> dict[str, Any]:
    fixture = synthetic_coordinate_fixture(path_count=16, noise_scale=0.0, lattice=lattice)
    decomposition = decompose_cross_panel_signal(fixture.left, fixture.right, lattice=lattice)
    swapped = decompose_cross_panel_signal(fixture.right, fixture.left, lattice=lattice)
    permuted = CoordinatePanel(
        role="synthetic-left-permuted",
        path_ids=fixture.left.path_ids.copy(),
        cell_means=fixture.left.cell_means[::-1].copy(),
    )
    permuted_decomposition = decompose_cross_panel_signal(
        permuted, fixture.right, lattice=lattice
    )
    zero_amplitudes = {name: (0.0, 0.0, 0.0, 0.0) for name in COORDINATE_COMPONENTS}
    null = synthetic_coordinate_fixture(
        path_count=16,
        noise_scale=0.0,
        component_amplitudes=zero_amplitudes,
        seed=261_364,
        lattice=lattice,
    )
    null_seal = seal_panel_a_directions(
        null.left, lattice=lattice, components=(PRIMARY_COMPONENT,)
    )
    null_evidence = evaluate_panel_b_linear(null_seal, null.right, lattice=lattice)
    null_inference = one_sided_b_linear_max_t(
        null_evidence,
        replicates=1_000,
        seed=261_365,
        chunk_size=250,
    )
    return {
        "schema": RUN_SCHEMA + "-synthetic-symmetry-controls",
        "schema_version": 1,
        "exact_component_fixture_reconstruction_error": decomposition.maximum_reconstruction_error,
        "a_b_swap_maximum_error": float(
            np.max(
                np.abs(
                    decomposition.component_point_energies
                    - swapped.component_point_energies
                )
            )
        ),
        "path_permutation_maximum_error": float(
            np.max(
                np.abs(
                    decomposition.component_point_energies
                    - permuted_decomposition.component_point_energies
                )
            )
        ),
        "stationary_null_maximum_absolute_point": float(
            np.max(np.abs(null_inference.point_estimates))
        ),
        "stationary_null_maximum_lower_bound": float(
            np.max(null_inference.lower_bounds)
        ),
        "negative_values_truncated": 0,
        **_scope(),
    }


def _symmetry_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "symmetry_gate.json", "symmetry_artifact_seal.json"):
        return
    _verify_stage_seal(run_dir, "replay_artifact_seal.json")
    immutability = _verify_parent_immutability(run_dir, args)
    lattice = build_coordinate_lattice()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(261_362)
        model = PhaseConditionedLocalAffineCNN(width=32).to(device="cpu")
    architecture = phase_predictor_architecture_contract(model)
    inputs = synthetic_model_inputs(batch_size=7, seed=261_362)
    equivariance = [
        model_translation_equivariance_record(
            model,
            inputs,
            row_shift=row_shift,
            column_shift=column_shift,
            tolerance=MODEL_EQUIVARIANCE_TOLERANCE,
        )
        for row_shift, column_shift in ((2, 0), (0, 2), (2, 2), (-2, 2))
    ]
    controls = _synthetic_symmetry_controls(lattice)
    atomic_write_json(run_dir / "predictor_architecture_contract.json", {**architecture, **_scope()})
    atomic_write_json(
        run_dir / "translation_equivariance_audit.json",
        {
            "schema": RUN_SCHEMA + "-translation-equivariance-audit",
            "schema_version": 1,
            "records": equivariance,
            "shifted_states_are_scientific_evidence": 0,
            "off_support_caveat": (
                "the one-image law is not translation invariant; this audit establishes "
                "model symmetry, not a distributional counterfactual"
            ),
            **_scope(),
        },
    )
    atomic_write_json(run_dir / "synthetic_coordinate_controls.json", controls)
    checks = {
        "immutable_parents_unchanged": int(
            immutability.get("passed", int(immutability.get("parent_files_modified", 1) == 0))
        ),
        "coordinate_free_architecture_verified": int(architecture.get("passed", 0)),
        "translation_equivariance_verified": int(
            all(int(row.get("passed", 0)) == 1 for row in equivariance)
        ),
        "projection_fixture_reconstructs": int(
            controls["exact_component_fixture_reconstruction_error"]
            <= PROJECTION_RECONSTRUCTION_TOLERANCE
        ),
        "a_b_swap_invariant": int(
            controls["a_b_swap_maximum_error"] <= PROJECTION_RECONSTRUCTION_TOLERANCE
        ),
        "path_permutation_invariant": int(
            controls["path_permutation_maximum_error"]
            <= PROJECTION_RECONSTRUCTION_TOLERANCE
        ),
        "stationary_null_clean": int(
            controls["stationary_null_maximum_absolute_point"] == 0.0
            and controls["stationary_null_maximum_lower_bound"] == 0.0
        ),
    }
    metrics = {
        "schema": RUN_SCHEMA + "-symmetry-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "checks": checks,
        "maximum_translation_equivariance_error": max(
            float(row["maximum_translation_equivariance_error"]) for row in equivariance
        ),
        **_scope(),
    }
    gate = _gate(RUN_SCHEMA + "-symmetry-gate", checks)
    atomic_write_json(run_dir / "symmetry_metrics.json", metrics)
    atomic_write_json(run_dir / "symmetry_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "predictor_architecture_contract.json",
            "translation_equivariance_audit.json",
            "synthetic_coordinate_controls.json",
            "symmetry_metrics.json",
            "symmetry_gate.json",
        ),
        "symmetry_artifact_seal.json",
    )


def _decompose_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "decomposition_gate.json", "decomposition_artifact_seal.json"):
        return
    _verify_stage_seal(run_dir, "symmetry_artifact_seal.json")
    immutability = _verify_parent_immutability(run_dir, args)
    plan = _load_json(run_dir / "coordinate_hypothesis_plan.json")
    _verify_semantic(plan, "coordinate hypothesis plan")
    lattice = build_coordinate_lattice()

    # Panel A is the only numerical panel opened before the direction seal is
    # durably committed.  Panel B is loaded only after this mini-seal exists.
    panel_a = _load_panel(args.parent_coarse_witness_run_dir, "a")
    a_seal = seal_panel_a_directions(
        panel_a, lattice=lattice, components=(PRIMARY_COMPONENT,)
    )
    a_npz = _atomic_npz(
        run_dir / "panel_a_frequency1_directions.npz",
        panel_a_path_ids=a_seal.panel_a_path_ids,
        directions=a_seal.directions,
        direction_norms=a_seal.direction_norms,
        direction_active_mask=a_seal.direction_active_mask.astype(np.uint8),
        **_scope_arrays(),
    )
    a_record = {
        **a_seal.to_record(),
        "direction_npz": a_npz,
        "sealed_before_panel_b_array_loading": 1,
        **_scope(),
    }
    atomic_write_json(run_dir / "panel_a_direction_seal.json", a_record)
    _seal_stage(
        run_dir,
        ("panel_a_frequency1_directions.npz", "panel_a_direction_seal.json"),
        "panel_a_direction_artifact_seal.json",
    )

    panel_b_parent_path = (
        args.parent_coarse_witness_run_dir / "panels" / "b" / "cell_means.npz"
    )
    b_intent = _semantic(
        {
            "schema": RUN_SCHEMA + "-panel-b-opening-intent",
            "schema_version": 1,
            "committed_before_panel_b_array_loading": 1,
            "panel_a_direction_artifact_seal_sha256": file_fingerprint(
                run_dir / "panel_a_direction_artifact_seal.json"
            ),
            "panel_b_parent_file_sha256": file_fingerprint(panel_b_parent_path),
            "permitted_operation": "linear evaluation of frozen A directions",
            "direction_or_sign_optimization_on_panel_b_permitted": 0,
            "historical_panel_authorizing": 0,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "panel_b_opening_intent.json", b_intent)
    _seal_stage(
        run_dir,
        ("panel_b_opening_intent.json",),
        "panel_b_opening_intent_seal.json",
    )
    panel_b = _load_panel(args.parent_coarse_witness_run_dir, "b")
    b_opening = _semantic(
        {
            "schema": RUN_SCHEMA + "-panel-b-opening",
            "schema_version": 1,
            "opened_after_panel_a_direction_artifact_seal": 1,
            "panel_a_direction_seal_sha256": file_fingerprint(
                run_dir / "panel_a_direction_artifact_seal.json"
            ),
            "panel_b_role": panel_b.role,
            "panel_b_path_ids": panel_b.path_ids.tolist(),
            "panel_b_fingerprint": panel_b.fingerprint,
            "linear_evaluation_commit_count": 1,
            "opening_intent_sha256": file_fingerprint(
                run_dir / "panel_b_opening_intent_seal.json"
            ),
            "historical_panel_authorizing": 0,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "panel_b_opening_record.json", b_opening)

    decomposition = decompose_cross_panel_signal(panel_a, panel_b, lattice=lattice)
    b_evidence = evaluate_panel_b_linear(a_seal, panel_b, lattice=lattice)
    inference = one_sided_b_linear_max_t(
        b_evidence,
        confidence=0.99,
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        namespace=DEFAULT_BOOTSTRAP_NAMESPACE,
        chunk_size=DEFAULT_BOOTSTRAP_CHUNK_SIZE,
    )
    scaled_bounds = scaled_signed_cross_bounds(a_seal, inference)
    primary_cross = decomposition.component_point_energies[
        COORDINATE_COMPONENTS.index(PRIMARY_COMPONENT)
    ]
    direct_cross_error = float(
        np.max(np.abs(primary_cross - b_evidence.signed_cross_energies))
    )
    component_sum_error = float(
        np.max(
            np.abs(
                np.sum(decomposition.component_point_energies, axis=0, dtype=np.float64)
                - decomposition.full_point_energies
            )
        )
    )
    _atomic_npz(
        run_dir / "coordinate_decomposition.npz",
        panel_a_path_ids=panel_a.path_ids,
        panel_b_path_ids=panel_b.path_ids,
        component_kernels=decomposition.component_kernels,
        component_point_energies=decomposition.component_point_energies,
        full_point_energies=decomposition.full_point_energies,
        **_scope_arrays(),
    )
    _atomic_npz(
        run_dir / "panel_b_linear_evidence.npz",
        panel_b_path_ids=b_evidence.path_ids,
        path_values=b_evidence.path_values,
        point_estimates=b_evidence.point_estimates,
        signed_cross_energies=b_evidence.signed_cross_energies,
        standard_errors=inference.standard_errors,
        lower_bounds=inference.lower_bounds,
        scaled_signed_cross_lower_bounds=scaled_bounds,
        bootstrap_maxima=inference.bootstrap_maxima,
        **_scope_arrays(),
    )
    decomposition_record = {
        **decomposition.to_record(),
        "direct_frequency1_cross_error": direct_cross_error,
        "component_sum_point_error": component_sum_error,
        "panel_a_fingerprint": panel_a.fingerprint,
        "panel_b_fingerprint": panel_b.fingerprint,
        "coordinate_decomposition_npz_sha256": file_fingerprint(
            run_dir / "coordinate_decomposition.npz"
        ),
        **_scope(),
    }
    b_record = {
        **b_evidence.to_record(),
        "inference": inference.to_record(),
        "scaled_signed_cross_lower_bounds": scaled_bounds.tolist(),
        "panel_b_linear_evidence_npz_sha256": file_fingerprint(
            run_dir / "panel_b_linear_evidence.npz"
        ),
        **_scope(),
    }
    atomic_write_json(run_dir / "coordinate_decomposition.json", decomposition_record)
    atomic_write_json(run_dir / "panel_b_linear_inference.json", b_record)

    decomposition_rows = []
    for component_index, component in enumerate(COORDINATE_COMPONENTS):
        for quartile in range(4):
            decomposition_rows.append(
                {
                    "quartile": f"q{quartile}",
                    "component": component,
                    "signed_cross_panel_energy": float(
                        decomposition.component_point_energies[component_index, quartile]
                    ),
                    "full_cross_panel_energy": float(
                        decomposition.full_point_energies[quartile]
                    ),
                    "negative_values_truncated": 0,
                    "historical_post_hoc_evidence_authorizing": 0,
                    **_scope(),
                }
            )
    inference_rows = []
    for quartile, name in enumerate(inference.family_names):
        inference_rows.append(
            {
                "family": name,
                "quartile": f"q{quartile}",
                "panel_a_direction_norm": float(a_seal.direction_norms[quartile]),
                "b_linear_point_estimate": float(inference.point_estimates[quartile]),
                "b_linear_standard_error": float(inference.standard_errors[quartile]),
                "b_linear_simultaneous_lower_bound": float(inference.lower_bounds[quartile]),
                "signed_cross_panel_energy": float(b_evidence.signed_cross_energies[quartile]),
                "signed_cross_simultaneous_lower_bound": float(scaled_bounds[quartile]),
                "simultaneously_positive": int(scaled_bounds[quartile] > 0.0),
                "negative_values_truncated": 0,
                "historical_post_hoc_evidence_authorizing": 0,
                **_scope(),
            }
        )
    atomic_write_csv(run_dir / "coordinate_component_energies.csv", decomposition_rows)
    atomic_write_csv(run_dir / "frequency1_heldout_inference.csv", inference_rows)

    q_positive = {f"q{quartile}": bool(scaled_bounds[quartile] > 0.0) for quartile in range(4)}
    inference_valid = bool(
        tuple(inference.family_names) == PRIMARY_FAMILY_NAMES
        and np.isfinite(inference.point_estimates).all()
        and np.isfinite(inference.standard_errors).all()
        and np.all(inference.standard_errors > 0.0)
        and np.isfinite(inference.lower_bounds).all()
        and inference.replicates == int(args.bootstrap_replicates)
        and inference.seed == int(args.bootstrap_seed)
    )
    algebra_valid = bool(
        lattice.maximum_gram_error <= PROJECTION_RECONSTRUCTION_TOLERANCE
        and decomposition.maximum_reconstruction_error
        <= PROJECTION_RECONSTRUCTION_TOLERANCE
        and direct_cross_error <= PROJECTION_RECONSTRUCTION_TOLERANCE
        and component_sum_error <= PROJECTION_RECONSTRUCTION_TOLERANCE
    )
    checks = {
        "immutable_parents_unchanged": int(
            immutability.get("passed", int(immutability.get("parent_files_modified", 1) == 0))
        ),
        "panel_a_direction_sealed_before_b": int(
            b_opening["opened_after_panel_a_direction_artifact_seal"] == 1
        ),
        "panels_disjoint": int(np.intersect1d(panel_a.path_ids, panel_b.path_ids).size == 0),
        "coordinate_projection_algebra_valid": int(algebra_valid),
        "coordinate_inference_valid": int(inference_valid),
        "negative_values_untruncated": 1,
        "historical_evidence_nonauthorizing": 1,
    }
    metrics = {
        "schema": RUN_SCHEMA + "-decomposition-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "checks": checks,
        "maximum_gram_error": lattice.maximum_gram_error,
        "maximum_reconstruction_error": decomposition.maximum_reconstruction_error,
        "component_sum_point_error": component_sum_error,
        "direct_frequency1_cross_error": direct_cross_error,
        "critical_value": inference.critical_value,
        "quartile_positive": q_positive,
        **_scope(),
    }
    evidence = {
        "schema": RUN_SCHEMA + "-decision-evidence",
        "schema_version": 1,
        "control_provenance_valid": 1,
        "portable_directional_parent_valid": 1,
        "coarse_witness_parent_valid": 1,
        "coordinate_hypothesis_plan_valid": 1,
        "coarse_witness_replay_valid": 1,
        "translation_symmetry_audit_valid": 1,
        "coordinate_projection_algebra_valid": int(algebra_valid),
        "coordinate_inference_valid": int(inference_valid),
        "q0_positive_control": int(q_positive["q0"]),
        "later_quartile_positive": {
            quartile: int(q_positive[quartile]) for quartile in ("q1", "q2", "q3")
        },
        "historical_post_hoc_evidence_authorizing": 0,
        **_scope(),
    }
    gate = _gate(RUN_SCHEMA + "-decomposition-gate", checks)
    atomic_write_json(run_dir / "decomposition_metrics.json", metrics)
    atomic_write_json(run_dir / "absolute_coordinate_decision_evidence.json", evidence)
    atomic_write_json(run_dir / "decomposition_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "panel_a_frequency1_directions.npz",
            "panel_a_direction_seal.json",
            "panel_a_direction_artifact_seal.json",
            "panel_b_opening_intent.json",
            "panel_b_opening_intent_seal.json",
            "panel_b_opening_record.json",
            "coordinate_decomposition.npz",
            "panel_b_linear_evidence.npz",
            "coordinate_decomposition.json",
            "panel_b_linear_inference.json",
            "coordinate_component_energies.csv",
            "frequency1_heldout_inference.csv",
            "decomposition_metrics.json",
            "absolute_coordinate_decision_evidence.json",
            "decomposition_gate.json",
        ),
        "decomposition_artifact_seal.json",
    )


def _decision_evidence(run_dir: Path) -> dict[str, Any]:
    evidence = _optional_json(run_dir, "absolute_coordinate_decision_evidence.json")
    if evidence is not None:
        return evidence
    preflight = _optional_json(run_dir, "preflight_gate.json")
    replay = _optional_json(run_dir, "replay_gate.json")
    symmetry = _optional_json(run_dir, "symmetry_gate.json")
    decomposition = _optional_json(run_dir, "decomposition_gate.json")
    return {
        "control_provenance_valid": int(_passed(preflight)),
        "portable_directional_parent_valid": int(_passed(preflight)),
        "coarse_witness_parent_valid": int(_passed(preflight)),
        "coordinate_hypothesis_plan_valid": int(_passed(preflight)),
        "coarse_witness_replay_valid": int(_passed(replay)),
        "translation_symmetry_audit_valid": int(_passed(symmetry)),
        "coordinate_projection_algebra_valid": int(_passed(decomposition)),
        "coordinate_inference_valid": int(_passed(decomposition)),
        **_scope(),
    }


def _readiness_decision(run_dir: Path) -> str:
    if (run_dir / "absolute_coordinate_decision_evidence.json").is_file():
        candidate = decide_absolute_coordinate(_decision_evidence(run_dir))
        if int(candidate.get("invalid_evidence", 0)) == 1:
            return str(candidate["decision"])
    if _passed(_optional_json(run_dir, "decomposition_gate.json")):
        return str(decide_absolute_coordinate(_decision_evidence(run_dir))["decision"])
    if _passed(_optional_json(run_dir, "symmetry_gate.json")):
        return "ready_for_decompose"
    if _passed(_optional_json(run_dir, "replay_gate.json")):
        return "ready_for_symmetry"
    if _passed(_optional_json(run_dir, "preflight_gate.json")):
        return "ready_for_replay"
    return "preflight_not_passed"


def _failure_decision_evidence(stage: str, exc: BaseException) -> dict[str, Any]:
    values: dict[str, Any] = {
        "control_provenance_valid": 1,
        "portable_directional_parent_valid": 1,
        "coarse_witness_parent_valid": 1,
        "coordinate_hypothesis_plan_valid": 1,
        "coarse_witness_replay_valid": 1,
        "translation_symmetry_audit_valid": 1,
        "coordinate_projection_algebra_valid": 1,
        "coordinate_inference_valid": 1,
        **_scope(),
    }
    message = str(exc).lower()
    if stage in {"initialize", "preflight"}:
        if isinstance(exc, AbsoluteCoordinateProvenanceError):
            if any(token in message for token in ("portable", "archive", "zip")):
                values["portable_directional_parent_valid"] = 0
            else:
                values["control_provenance_valid"] = 0
                values["coarse_witness_parent_valid"] = 0
        else:
            values["coordinate_hypothesis_plan_valid"] = 0
    elif stage == "replay":
        values["coarse_witness_replay_valid"] = 0
    elif stage == "symmetry":
        values["translation_symmetry_audit_valid"] = 0
    elif stage == "decompose":
        if any(token in message for token in ("bootstrap", "max-t", "inference", "student")):
            values["coordinate_inference_valid"] = 0
        else:
            values["coordinate_projection_algebra_valid"] = 0
    else:
        values["coordinate_inference_valid"] = 0
    return {
        "schema": RUN_SCHEMA + "-failure-decision-evidence",
        "schema_version": 1,
        **values,
    }


def _required_gate_record(run_dir: Path, require_gate: str) -> dict[str, Any]:
    gate_files = {
        "preflight": "preflight_gate.json",
        "replay": "replay_gate.json",
        "symmetry": "symmetry_gate.json",
        "decompose": "decomposition_gate.json",
    }
    if require_gate == "none":
        passed = 1
        gate = None
    else:
        gate = _optional_json(run_dir, gate_files[require_gate])
        passed = int(_passed(gate))
    decision = (
        {**decide_absolute_coordinate(_decision_evidence(run_dir)), **_scope()}
        if (run_dir / "absolute_coordinate_decision_evidence.json").is_file()
        else None
    )
    record = {
        "schema": RUN_SCHEMA + "-workflow-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_gate_pass": passed,
        "required_gate_record": gate,
        "decision": decision,
        "current_decision": _readiness_decision(run_dir),
        **_scope(),
    }
    atomic_write_json(run_dir / "workflow_gate.json", record)
    if decision is not None:
        atomic_write_json(run_dir / "absolute_coordinate_adjudication_decision.json", decision)
    return record


def _write_report(run_dir: Path) -> dict[str, Any]:
    _verify_stage_seal(run_dir, "decomposition_artifact_seal.json")
    decision = {**decide_absolute_coordinate(_decision_evidence(run_dir)), **_scope()}
    inference = _load_json(run_dir / "panel_b_linear_inference.json")
    decomposition = _load_json(run_dir / "coordinate_decomposition.json")
    rows = []
    scaled = inference["scaled_signed_cross_lower_bounds"]
    points = inference["signed_cross_energies"]
    for quartile in range(4):
        rows.append(
            f"| q{quartile} | {float(points[quartile]):.10g} | "
            f"{float(scaled[quartile]):.10g} | {int(float(scaled[quartile]) > 0.0)} |"
        )
    report = "\n".join(
        [
            "# Exact Jacobi/RB absolute-coordinate adjudication",
            "",
            f"Decision: `{decision['decision']}`.",
            "",
            "This is a read-only, historical, post-hoc representation diagnostic. ",
            "It creates no new scientific evidence and grants no training, controller, ",
            "reconstruction, or sampling authority.",
            "",
            "## Held-out frequency-one result",
            "",
            "| Quartile | Signed cross energy | Simultaneous 99% lower bound | Positive |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            f"The joint max-T critical value was {float(inference['inference']['critical_value']):.10g}. ",
            f"The projection reconstruction error was {float(decomposition['maximum_reconstruction_error']):.3g}.",
            "",
            "## Interpretation",
            "",
            str(decision["recommended_next_action"]) + ".",
            "",
            "A positive result supports a finite absolute-coordinate feature-family hypothesis; ",
            "it does not identify a unique architecture. The fixed-image law is not translation ",
            "invariant, so the symmetry audit cannot prove that the old CNN was unable to infer ",
            "position indirectly from state content.",
            "",
            "The exact Jacobi transition law and raw Rao-Blackwell target were not evaluated or ",
            "changed by this workflow.",
            "",
        ]
    )
    target = run_dir / "REPORT.md"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, target)
    atomic_write_json(run_dir / "absolute_coordinate_adjudication_decision.json", decision)
    return decision


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "replay", "symmetry", "decompose", "report")
    return (stage,)


def _require_prerequisite(run_dir: Path, stage: str) -> None:
    prerequisites = {
        "replay": "preflight_gate.json",
        "symmetry": "replay_gate.json",
        "decompose": "symmetry_gate.json",
        "report": "decomposition_gate.json",
    }
    gate_name = prerequisites.get(stage)
    if gate_name is not None and not _passed(_optional_json(run_dir, gate_name)):
        raise ArtifactCompatibilityError(f"{stage} requires passing {gate_name}")


def _commit_execution_failure(
    run_dir: Path, *, stage: str, failure: Mapping[str, Any]
) -> None:
    gate_names = {
        "preflight": "preflight_gate.json",
        "replay": "replay_gate.json",
        "symmetry": "symmetry_gate.json",
        "decompose": "decomposition_gate.json",
    }
    gate_name = gate_names.get(stage)
    if gate_name is None:
        return
    gate = {
        "schema": RUN_SCHEMA + f"-{stage}-gate",
        "schema_version": 1,
        "evaluation_status": "execution_failed",
        "passed": 0,
        "failure_domain": failure.get("failure_domain"),
        "failure_code": failure.get("failure_code"),
        "error": failure.get("error"),
        **_scope(),
    }
    atomic_write_json(run_dir / gate_name, gate)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-directional-result-archive", type=Path, required=True)
    parser.add_argument("--parent-coarse-witness-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_absolute_coordinate_adjudication"),
    )
    parser.add_argument("--run-name", default="production-read-only-absolute-coordinate-adjudication")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in (
        "parent_directional_result_archive",
        "parent_coarse_witness_run_dir",
        "resume_run_dir",
        "runs_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.resume_run_dir is None and args.stage in {
        "replay",
        "symmetry",
        "decompose",
        "report",
    }:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    expected = {
        "preflight": "preflight",
        "replay": "replay",
        "symmetry": "symmetry",
        "decompose": "decompose",
        "report": "none",
        "all": "decompose",
    }[args.stage]
    if args.require_gate not in {"none", expected}:
        parser.error(f"--stage {args.stage} cannot require only {args.require_gate}")
    if args.device != "cpu" and not args.test_only:
        parser.error("this read-only production adjudication requires --device cpu")
    if (
        args.bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES
        or args.bootstrap_seed != DEFAULT_BOOTSTRAP_SEED
    ) and not args.test_only:
        parser.error("production bootstrap configuration is frozen")
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    initialized = False
    active_stage = "initialize"
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"absolute-coordinate adjudication run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            _status(run_dir, state="running", stage=stage)
            _require_prerequisite(run_dir, stage)
            if stage == "preflight":
                _preflight_stage(run_dir, args)
            elif stage == "replay":
                _replay_stage(run_dir, args)
            elif stage == "symmetry":
                _symmetry_stage(run_dir, args)
            elif stage == "decompose":
                _decompose_stage(run_dir, args)
            elif stage == "report":
                _write_report(run_dir)
            gate_by_stage = {
                "preflight": "preflight_gate.json",
                "replay": "replay_gate.json",
                "symmetry": "symmetry_gate.json",
                "decompose": "decomposition_gate.json",
            }
            gate_name = gate_by_stage.get(stage)
            if gate_name is not None and not _passed(_optional_json(run_dir, gate_name)):
                if not (run_dir / "absolute_coordinate_decision_evidence.json").is_file():
                    atomic_write_json(
                        run_dir / "absolute_coordinate_decision_evidence.json",
                        _failure_decision_evidence(
                            stage,
                            AbsoluteCoordinateWorkflowError(
                                f"{stage} evaluated gate failed",
                                failure_domain="scientific_integrity_gate",
                                failure_code=f"{stage}_gate_failed",
                            ),
                        ),
                    )
                break
        workflow = _required_gate_record(run_dir, args.require_gate)
        current = str(workflow["current_decision"])
        required_pass = int(workflow["required_gate_pass"]) == 1
        final = workflow.get("decision")
        final_scientific = isinstance(final, Mapping) and int(
            final.get("scientific_evidence_complete", 0)
        ) == 1
        final_valid_stop = isinstance(final, Mapping) and int(
            final.get("valid_scientific_stop", 0)
        ) == 1
        state = (
            "valid_scientific_stop"
            if required_pass and final_valid_stop
            else "complete"
            if required_pass
            else "gate_failed"
        )
        _status(
            run_dir,
            state=state,
            stage=args.stage,
            decision=current,
            failure_domain=None if required_pass else "required_gate",
            failure_code=None if required_pass else f"{args.require_gate}_gate_failed",
            scientific_evidence_complete=int(final_scientific),
        )
        _artifact_registry(run_dir)
        print(f"absolute-coordinate adjudication decision: {current}", flush=True)
        if not required_pass:
            return 2
        if final is not None:
            return decision_exit_code(final)
        return 0
    except KeyboardInterrupt:
        if run_dir is not None and initialized:
            _status(
                run_dir,
                state="interrupted",
                stage=active_stage,
                message="interrupted; resume the same run directory",
                failure_domain="interruption",
                failure_code="keyboard_interrupt",
            )
            _artifact_registry(run_dir)
        raise
    except Exception as exc:
        if run_dir is not None and initialized:
            failure = {
                "schema": RUN_SCHEMA + "-execution-failure",
                "schema_version": 1,
                "evaluation_status": "execution_failed",
                "stage": active_stage,
                "failure_domain": str(getattr(exc, "failure_domain", "workflow_execution")),
                "failure_code": str(
                    getattr(exc, "failure_code", "absolute_coordinate_execution_failed")
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                **_scope(),
            }
            atomic_write_json(run_dir / f"{active_stage}_execution_failure.json", failure)
            _commit_execution_failure(run_dir, stage=active_stage, failure=failure)
            if not (run_dir / "absolute_coordinate_decision_evidence.json").is_file():
                atomic_write_json(
                    run_dir / "absolute_coordinate_decision_evidence.json",
                    _failure_decision_evidence(active_stage, exc),
                )
            _required_gate_record(run_dir, "none")
            _status(
                run_dir,
                state="execution_failed",
                stage=active_stage,
                decision=_readiness_decision(run_dir),
                message=str(exc),
                failure_domain=failure["failure_domain"],
                failure_code=failure["failure_code"],
            )
            _artifact_registry(run_dir)
        print(f"absolute-coordinate adjudication error: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
