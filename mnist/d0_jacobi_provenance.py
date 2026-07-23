"""Immutable provenance and report-only adjudication for the Jacobi pivot.

The immediate parent contains a known workflow-reporting defect.  Its terminal
registry is nevertheless immutable and internally bound, so this module checks
the exact registry and reconstructs the factual pilot outcome without changing
any parent artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)


PARENT_RUN_SCHEMA = "experiment12-d0-score-density-ratio-h1-gradient-control-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 277
PARENT_REGISTRY_SHA256 = "0341f1defa29029fce03c638d86b15db1565c2f4d488b7fce8413fa140dc71ab"
PARENT_RAW_DECISION = "selection_false_discovery"
READJUDICATED_DECISION = "h1_strength_grid_unresolved"
EXPECTED_RATIOS = (0.0, 0.1, 0.3, 1.0)
EXPECTED_LINEAGE_COUNTS = (301, 263, 123, 125, 332, 222, 381)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read Jacobi parent artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"Jacobi parent artifact is not an object: {path}")
    return dict(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _one(value: Any) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _verify_registry(run_dir: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / "artifact_registry.json"
    registry = _load(path)
    digest = file_fingerprint(path)
    _require(digest == PARENT_REGISTRY_SHA256, "parent is not the frozen 277-record run")
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and int(registry.get("schema_version", -1)) == 1,
        "parent terminal registry schema is incompatible",
    )
    _require(
        status.get("artifact_registry_sha256") == digest
        and int(status.get("artifact_registry_size", -1)) == int(path.stat().st_size),
        "parent status does not bind its terminal registry",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    _require(exclusions == {"artifact_registry.json", "run_status.json"}, "parent registry exclusions changed")
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "parent registry records are invalid")
    records = dict(raw_records)
    _require(len(records) == PARENT_REGISTRY_RECORD_COUNT, "parent registry record count changed")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.relative_to(run_dir).as_posix() not in exclusions
    }
    _require(actual == set(records), "parent registry is incomplete or contains stale records")
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record: {relative}")
        artifact = run_dir / relative
        _require(
            artifact.is_file()
            and raw.get("sha256") == file_fingerprint(artifact)
            and int(raw.get("size", -1)) == int(artifact.stat().st_size),
            f"registered parent artifact changed: {relative}",
        )
    return registry


def _verify_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1,
        "parent manifest schema is incompatible",
    )
    scientific = manifest.get("scientific_config")
    _require(isinstance(scientific, Mapping), "parent scientific config is missing")
    _require(
        config_fingerprint(dict(scientific)) == manifest.get("scientific_fingerprint"),
        "parent scientific fingerprint is inconsistent",
    )
    source_paths = [Path(value) for value in manifest.get("source_paths", [])]
    _require(bool(source_paths) and all(path.is_file() for path in source_paths), "parent source paths are unavailable")
    _require(
        source_fingerprint(source_paths) == manifest.get("source_fingerprint"),
        "parent source fingerprint changed",
    )
    provenance = run_dir / "parent_provenance.json"
    _require(
        provenance.is_file()
        and file_fingerprint(provenance) == manifest.get("parent_provenance_sha256"),
        "parent transitive provenance binding changed",
    )


def _verify_transitive_registry_chain(
    first_provenance: Mapping[str, Any],
) -> list[int]:
    """Verify the registry/status bindings of every recorded ancestor.

    This deliberately avoids importing any historical trainer.  Each parent
    provenance record freezes the next run directory and registry digest; the
    status in that run independently binds the same immutable registry.
    """

    provenance = dict(first_provenance)
    observed: list[int] = []
    for expected_count in EXPECTED_LINEAGE_COUNTS:
        parent_dir = Path(str(provenance.get("run_dir", ""))).resolve()
        _require(parent_dir.is_dir(), f"transitive parent run is unavailable: {parent_dir}")
        registry_path = parent_dir / "artifact_registry.json"
        status_path = parent_dir / "run_status.json"
        _require(registry_path.is_file() and status_path.is_file(), "transitive registry/status is missing")
        artifact_records = provenance.get("artifacts", {})
        registry_record = (
            dict(artifact_records).get("registry", {})
            if isinstance(artifact_records, Mapping)
            else {}
        )
        expected_digest = str(
            provenance.get("artifact_registry_sha256")
            or (registry_record.get("sha256") if isinstance(registry_record, Mapping) else "")
        )
        expected_size = int(
            provenance.get("artifact_registry_size")
            or (registry_record.get("size") if isinstance(registry_record, Mapping) else -1)
        )
        digest = file_fingerprint(registry_path)
        _require(digest == expected_digest, "transitive registry digest changed")
        _require(
            expected_size == int(registry_path.stat().st_size),
            "transitive registry size changed",
        )
        registry = _load(registry_path)
        records = registry.get("records", {})
        _require(isinstance(records, Mapping), "transitive registry records are invalid")
        count = len(dict(records))
        _require(count == int(expected_count), f"transitive registry count changed: {count}")
        status = _load(status_path)
        _require(
            status.get("artifact_registry_sha256") == digest
            and int(status.get("artifact_registry_size", -1)) == expected_size,
            "transitive status no longer binds its registry",
        )
        observed.append(count)
        if len(observed) < len(EXPECTED_LINEAGE_COUNTS):
            next_path = parent_dir / "parent_provenance.json"
            _require(next_path.is_file(), "transitive parent provenance is missing")
            provenance = _load(next_path)
    return observed


def verify_and_readjudicate_gradient_parent(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"parent run does not exist: {root}")
    required = {
        name: root / relative
        for name, relative in {
            "manifest": "run_manifest.json",
            "status": "run_status.json",
            "decision": "h1_gradient_control_decision.json",
            "pilot_gate": "h1_gradient_control_pilot_gate.json",
            "candidates": "gradient_control_pilot_candidates.json",
            "failures": "pilot_task_failures.json",
            "null_gate": "pilot_null_family_gate.json",
            "null_max_t": "pilot_null_b_max_t.json",
        }.items()
    }
    for name, path in required.items():
        _require(path.is_file(), f"parent lacks {name}: {path.name}")
    manifest = _load(required["manifest"])
    status = _load(required["status"])
    registry = _verify_registry(root, status)
    _verify_manifest(root, manifest)

    decision = _load(required["decision"])
    pilot = _load(required["pilot_gate"])
    candidates_record = _load(required["candidates"])
    failures = _load(required["failures"])
    null_gate = _load(required["null_gate"])
    null_max_t = _load(required["null_max_t"])

    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and status.get("stage") == "pilot"
        and status.get("require_gate") == "pilot"
        and status.get("required_gate") == "pilot"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == PARENT_RAW_DECISION,
        "parent terminal status is not the expected failed pilot",
    )
    _require(
        _zero(status.get("physical_training_authorized", 0))
        and _zero(status.get("sampling_authorized", 0))
        and status.get("skips") == [
            {"reason": "gradient-control pilot failed", "stage": "confirmation"}
        ],
        "parent terminal authorization or skip semantics changed",
    )
    _require(decision.get("decision") == PARENT_RAW_DECISION, "parent raw decision changed")
    for record in (status, decision, pilot, candidates_record, failures, null_gate, null_max_t):
        _require(
            _zero(record.get("physical_training_performed", 0))
            and _zero(record.get("sampling_performed", 0)),
            "parent records physical training or sampling",
        )

    selected = pilot.get("selected_profile", {})
    nominated = pilot.get("nominated_profile", {})
    _require(
        _zero(selected.get("selected", -1))
        and _zero(nominated.get("selected", -1))
        and _one(pilot.get("all_nulls_select_analytic_zero"))
        and _one(pilot.get("optimizer_and_controller_health_pass"))
        and _zero(pilot.get("selection_false_discovery")),
        "parent pilot evidence does not support report-only re-adjudication",
    )
    _require(
        null_gate.get("evaluation_status") == "not_evaluated"
        and null_max_t.get("evaluation_status") == "not_evaluated"
        and int(null_max_t.get("family_size", -1)) == 0,
        "parent panel-B/null family was unexpectedly evaluated",
    )
    _require(
        int(failures.get("count", -1)) == 0 and not failures.get("failures"),
        "parent contains task failures",
    )
    candidates = candidates_record.get("candidates", [])
    _require(isinstance(candidates, list) and len(candidates) == 4, "parent must contain four ratio candidates")
    ratios = tuple(float(item.get("target_ratio")) for item in candidates)
    _require(ratios == EXPECTED_RATIOS, "parent ratio grid changed")
    for item in candidates:
        _require(
            all(_one(item.get(key)) for key in (
                "complete", "finite", "boundary_admissible", "optimizer_health_pass",
                "controller_health_pass", "teacher_complete", "teacher_finite",
                "teacher_boundary_admissible", "null_complete", "null_finite",
                "null_boundary_admissible", "null_optimizer_health_pass",
            )),
            f"parent candidate {item.get('target_ratio')} is not healthy",
        )
        _require(
            float(item.get("maximum_clip_fraction_observed", -1.0)) == 0.0
            and int(item.get("null_selected_step", -1)) == 0
            and int(item.get("panel_b_evaluation_count", -1)) == 0,
            f"parent candidate {item.get('target_ratio')} conflicts with re-adjudication",
        )

    stored_transitive = _load(root / "parent_provenance.json")
    lineage = [PARENT_REGISTRY_RECORD_COUNT, *_verify_transitive_registry_chain(stored_transitive)]
    expected_lineage = [277, *EXPECTED_LINEAGE_COUNTS]
    _require(lineage == expected_lineage, f"transitive registry lineage changed: observed {lineage}")

    return {
        "schema": "d0-jacobi-parent-readjudication",
        "schema_version": 1,
        "parent_run_dir": str(root),
        "parent_run_schema": PARENT_RUN_SCHEMA,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": int((root / "artifact_registry.json").stat().st_size),
        "parent_artifact_record_count": len(dict(registry["records"])),
        "parent_scientific_fingerprint": manifest.get("scientific_fingerprint"),
        "parent_source_fingerprint": manifest.get("source_fingerprint"),
        "lineage_registry_record_counts": lineage,
        "saved_decision": PARENT_RAW_DECISION,
        "readjudicated_decision": READJUDICATED_DECISION,
        "readjudication_basis": {
            "candidate_count": len(candidates),
            "ratios": list(ratios),
            "all_tasks_complete_finite_boundary_optimizer_healthy": 1,
            "maximum_clip_fraction": 0.0,
            "all_nulls_selected_step_zero": 1,
            "nonzero_profile_nominated": 0,
            "panel_b_evaluation_count": 0,
            "null_family_evaluation_status": "not_evaluated",
            "null_family_size": 0,
            "task_failure_count": 0,
        },
        "readjudication_valid": 1,
        "parent_mutated": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


__all__ = [
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_RAW_DECISION",
    "READJUDICATED_DECISION",
    "verify_and_readjudicate_gradient_parent",
]
