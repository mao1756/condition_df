"""Immutable provenance for the failed Dynkin ID-fix preflight.

The parent run is retained as evidence of a numerical observer defect.  This
module verifies its complete terminal registry and the exact failure record;
it never rewrites or resumes that directory.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)


PARENT_RUN_BASENAME = (
    "20260724-184842_production-dynkin-strang-power-idfix"
)
PARENT_RUN_SCHEMA = "experiment12-d0-jacobi-rb-dynkin-power-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 18
PARENT_REGISTRY_SHA256 = (
    "57daf61e9686c2257d6579c84130e5e4b4a400a8435916a62ac408c87ad6072d"
)
PARENT_SOURCE_COUNT = 21
PARENT_SOURCE_FINGERPRINT = (
    "5fcf9af561f40c0f7dd0f41c4b886fe332b46e65c8dce993a5850f534a7b1a9e"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "ca846d930a521780d1de5946d6f31cff41f250ed6a680ec821e7d78a55c001c9"
)
PARENT_DECISION = "dynkin_estimator_numerically_unresolved"
PARENT_FAILURE_CODE = "preflight_execution_exception"
PARENT_FAILURE_DOMAIN = "numerical_execution"
PARENT_FAILURE_TYPE = "ValueError"
PARENT_FAILURE_MESSAGE = "nonzero degenerate whole-path statistic"
PARENT_RE_ADJUDICATION = "tower_observer_roundoff_invalid"
PARENT_PATH_ID_PLAN_VERSION = "d0-jacobi-rb-dynkin-path-id-v1"
PARENT_PATH_ID_PLAN_SHA256 = (
    "3b63bde010f8301a8ba52d6f8bcd7ca70c0d338f7d719b85693fc9e484b0f66e"
)

_REQUIRED_FILES = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "scientific_config": "scientific_config.json",
    "decision": "dynkin_power_decision.json",
    "workflow": "dynkin_workflow_gate.json",
    "preflight_gate": "dynkin_preflight_gate.json",
    "failure": "preflight_failure.json",
    "path_plan": "path_id_plan.json",
    "path_preflight": "path_id_plan_preflight.json",
    "legacy_replay": "legacy_k512_dynkin_observer_replay.json",
    "phase_moments": "phase_moment_oracle.csv",
    "sealed_panels": "sealed_panel_registry.json",
    "panel_a_plan": "panel_a_plan.json",
    "panel_b_plan": "panel_b_plan.json",
    "parent_provenance": "parent_provenance.json",
}

_PATH_PLAN_CHECKS = {
    "canonical_id_smoke_pass",
    "cross_level_alias_plan_pass",
    "future_production_reserved_pass",
    "packed_id_43_bit_pass",
    "path_id_20_bit_pass",
    "path_major_order_pass",
    "plan_frozen_pass",
    "plan_hash_pass",
    "right_endpoint_alias_pass",
    "role_disjoint_pass",
    "tower_chunking_pass",
}

_LEGACY_REPLAY_CHECKS = {
    "certificate_hash_invariance_pass",
    "legacy_k512_replay_pass",
    "no_unrequested_checkpoint_pass",
    "observer_state_hash_invariance_pass",
    "transition_and_target_hash_invariance_pass",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read phase-observer parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"parent artifact is not an object: {path}")
    return dict(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _no_work(record: Mapping[str, Any], description: str) -> None:
    _require(
        _zero(record.get("physical_training_performed", 0))
        and _zero(record.get("sampling_performed", 0))
        and _zero(record.get("reverse_sampling_performed", 0)),
        f"{description} records physical training or reverse sampling",
    )


def _verify_registry(
    root: Path,
    path: Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    registry = _load(path)
    digest = file_fingerprint(path)
    _require(digest == PARENT_REGISTRY_SHA256, "parent registry SHA-256 changed")
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and registry.get("schema_version") == 1,
        "parent registry schema changed",
    )
    raw_records = registry.get("records")
    _require(isinstance(raw_records, Mapping), "parent registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == PARENT_REGISTRY_RECORD_COUNT,
        f"parent registry must contain {PARENT_REGISTRY_RECORD_COUNT} records",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", ()))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "parent registry exclusions changed",
    )
    actual = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file()
        and artifact.relative_to(root).as_posix() not in exclusions
    }
    _require(actual == set(records), "parent registry file set changed")
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry row: {relative}")
        artifact = root / str(relative)
        _require(
            artifact.is_file()
            and raw.get("sha256") == file_fingerprint(artifact)
            and raw.get("size") == artifact.stat().st_size,
            f"registered parent artifact changed: {relative}",
        )
    _require(
        status.get("artifact_registry_sha256") == digest
        and status.get("artifact_registry_record_count")
        == PARENT_REGISTRY_RECORD_COUNT
        and status.get("artifact_registry_size") == path.stat().st_size,
        "parent status does not bind its terminal registry",
    )
    _no_work(registry, "parent registry")
    return registry


def _verify_phase_moments(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ArtifactCompatibilityError(
            f"cannot read parent phase-moment evidence: {exc}"
        ) from exc
    _require(len(rows) == 8, "parent phase-moment evidence changed")
    matchings: set[tuple[int, float]] = set()
    for row in rows:
        try:
            matching_index = int(row["matching_index"])
            duration = float(row["duration_fraction"])
            float64_error = float(row["maximum_float64_error"])
            cuda_error = float(row["maximum_cuda_error"])
            radius = float(row["maximum_error_radius"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactCompatibilityError(
                "parent phase-moment evidence is invalid"
            ) from exc
        _require(
            matching_index in range(4)
            and duration in {0.5, 1.0}
            and all(
                math.isfinite(value) and value >= 0.0
                for value in (float64_error, cuda_error, radius)
            )
            and float64_error <= 1.0e-10
            and cuda_error <= 2.0e-6,
            "parent phase-moment oracle no longer passes",
        )
        matchings.add((matching_index, duration))
    _require(
        matchings == {(index, duration) for index in range(4) for duration in (0.5, 1.0)},
        "parent phase-moment color/duration coverage changed",
    )
    return len(rows)


def verify_tower_observer_roundoff_parent(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Verify the immutable failed preflight and re-adjudicate its defect."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"phase-observer parent does not exist: {root}")
    _require(
        root.name == PARENT_RUN_BASENAME,
        f"parent basename must be {PARENT_RUN_BASENAME}",
    )
    paths = {name: root / relative for name, relative in _REQUIRED_FILES.items()}
    for name, path in paths.items():
        _require(path.is_file(), f"parent lacks {name}: {path.name}")

    status = _load(paths["status"])
    registry = _verify_registry(root, paths["registry"], status)
    manifest = _load(paths["manifest"])
    config = _load(paths["scientific_config"])
    decision = _load(paths["decision"])
    workflow = _load(paths["workflow"])
    gate = _load(paths["preflight_gate"])
    failure = _load(paths["failure"])
    path_plan = _load(paths["path_plan"])
    path_preflight = _load(paths["path_preflight"])
    replay = _load(paths["legacy_replay"])
    sealed = _load(paths["sealed_panels"])
    panel_a = _load(paths["panel_a_plan"])
    panel_b = _load(paths["panel_b_plan"])
    transitive = _load(paths["parent_provenance"])

    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent manifest/source/config binding changed",
    )
    raw_sources = manifest.get("source_paths")
    _require(
        isinstance(raw_sources, list) and len(raw_sources) == PARENT_SOURCE_COUNT,
        "parent immutable source set changed",
    )
    sources = [Path(str(value)) for value in raw_sources]
    _require(
        all(path.is_file() for path in sources)
        and source_fingerprint(sources) == PARENT_SOURCE_FINGERPRINT,
        "one of the twenty-one immutable parent sources changed",
    )
    _require(
        config_fingerprint(config) == PARENT_SCIENTIFIC_CONFIG_SHA256
        and config.get("schema") == PARENT_RUN_SCHEMA + "-scientific-config"
        and config.get("schema_version") == 1
        and config.get("root_seed") == 261161
        and config.get("tower_panel_clusters") == 128
        and config.get("pilot_panel_paths") == 8
        and config.get("test_only_reduced_workload") == 0
        and config.get("path_id_plan_version") == PARENT_PATH_ID_PLAN_VERSION
        and config.get("path_id_plan_sha256") == PARENT_PATH_ID_PLAN_SHA256,
        "parent scientific configuration changed",
    )
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA
        and status.get("schema_version") == 1
        and status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "preflight"
        and status.get("required_gate") == "preflight"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == PARENT_DECISION
        and status.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and status.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent terminal status changed",
    )
    _require(
        decision == workflow
        and decision.get("decision") == PARENT_DECISION
        and _zero(decision.get("production_refinement_patch_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "parent failed decision changed",
    )
    _require(
        failure.get("schema") == PARENT_RUN_SCHEMA + "-stage-failure"
        and failure.get("stage") == "preflight"
        and failure.get("error_type") == PARENT_FAILURE_TYPE
        and failure.get("error") == PARENT_FAILURE_MESSAGE
        and failure.get("failure_code") == PARENT_FAILURE_CODE
        and failure.get("failure_domain") == PARENT_FAILURE_DOMAIN,
        "parent preflight failure changed",
    )
    _require(
        gate.get("evaluation_status") == "execution_failed"
        and gate.get("failed_stage") == "preflight"
        and gate.get("failure") == failure
        and gate.get("failure_code") == PARENT_FAILURE_CODE
        and gate.get("failure_domain") == PARENT_FAILURE_DOMAIN
        and _zero(gate.get("passed"))
        and _zero(gate.get("stage_execution_valid"))
        and _zero(gate.get("scientific_evidence_complete")),
        "parent execution-failure gate changed",
    )

    checks = path_preflight.get("checks")
    _require(
        path_plan.get("version") == PARENT_PATH_ID_PLAN_VERSION
        and path_plan.get("path_id_plan_sha256") == PARENT_PATH_ID_PLAN_SHA256
        and path_preflight.get("path_id_plan_version")
        == PARENT_PATH_ID_PLAN_VERSION
        and path_preflight.get("path_id_plan_sha256")
        == PARENT_PATH_ID_PLAN_SHA256
        and _one(path_preflight.get("passed"))
        and isinstance(checks, Mapping)
        and set(checks) == _PATH_PLAN_CHECKS
        and all(_one(value) for value in checks.values()),
        "parent canonical path-ID controls changed",
    )
    _require(
        replay.get("path_id_plan_version") == PARENT_PATH_ID_PLAN_VERSION
        and replay.get("path_id_plan_sha256") == PARENT_PATH_ID_PLAN_SHA256
        and replay.get("path_ids") == list(range(20_000, 20_008))
        and all(_one(replay.get(name)) for name in _LEGACY_REPLAY_CHECKS),
        "parent legacy K=512 replay changed",
    )
    phase_moment_rows = _verify_phase_moments(paths["phase_moments"])

    _require(
        sealed.get("path_id_plan_version") == PARENT_PATH_ID_PLAN_VERSION
        and sealed.get("path_id_plan_sha256") == PARENT_PATH_ID_PLAN_SHA256
        and _one(sealed.get("sealed_before_panel_a"))
        and _one(sealed.get("panels_disjoint"))
        and _one(sealed.get("future_production_namespace_disjoint"))
        and panel_a.get("panel") == "a"
        and panel_b.get("panel") == "b"
        and _one(panel_a.get("sealed_before_execution"))
        and _one(panel_b.get("sealed_before_execution")),
        "parent sealed panel plans changed",
    )
    _require(
        _one(transitive.get("passed"))
        and transitive.get("parent_artifact_record_count") == 1308
        and transitive.get("parent_artifact_registry_sha256")
        == "734c93e1e7d0be29041e1d567b36cbd8ea7aac50df7996d5f8c41fbddef8e632",
        "parent transitive provenance changed",
    )

    records = set(dict(registry["records"]))
    _require(
        not any(
            relative.startswith("tower_")
            or relative.startswith("pilot_")
            or "selected_dynkin_design" in relative
            or relative.startswith("dynkin_shards/pilot/")
            for relative in records
        ),
        "parent contains tower inference or pilot execution evidence",
    )
    for description, record in (
        ("manifest", manifest),
        ("status", status),
        ("registry", registry),
        ("scientific config", config),
        ("decision", decision),
        ("workflow", workflow),
        ("preflight gate", gate),
        ("failure", failure),
        ("path plan", path_plan),
        ("path preflight", path_preflight),
        ("legacy replay", replay),
        ("sealed panels", sealed),
        ("panel A plan", panel_a),
        ("panel B plan", panel_b),
        ("transitive provenance", transitive),
    ):
        _no_work(record, f"parent {description}")

    return {
        "schema": "d0-jacobi-rb-dynkin-phase-observer-parent-provenance",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": paths["registry"].stat().st_size,
        "parent_artifact_record_count": PARENT_REGISTRY_RECORD_COUNT,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_source_count": PARENT_SOURCE_COUNT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent_decision": PARENT_DECISION,
        "parent_failure_code": PARENT_FAILURE_CODE,
        "parent_failure_domain": PARENT_FAILURE_DOMAIN,
        "parent_failure_type": PARENT_FAILURE_TYPE,
        "parent_failure_message": PARENT_FAILURE_MESSAGE,
        "parent_path_id_plan_pass": 1,
        "parent_legacy_k512_replay_pass": 1,
        "parent_phase_moment_oracle_pass": 1,
        "parent_phase_moment_row_count": phase_moment_rows,
        "parent_tower_inference_performed": 0,
        "parent_pilot_performed": 0,
        "parent_re_adjudication": PARENT_RE_ADJUDICATION,
        "tower_observer_roundoff_failure_pass": 1,
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "parent_mutated": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


__all__ = [
    "PARENT_DECISION",
    "PARENT_FAILURE_CODE",
    "PARENT_FAILURE_DOMAIN",
    "PARENT_FAILURE_MESSAGE",
    "PARENT_FAILURE_TYPE",
    "PARENT_PATH_ID_PLAN_SHA256",
    "PARENT_PATH_ID_PLAN_VERSION",
    "PARENT_RE_ADJUDICATION",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_RUN_BASENAME",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_COUNT",
    "PARENT_SOURCE_FINGERPRINT",
    "verify_tower_observer_roundoff_parent",
]
