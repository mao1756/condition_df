"""Immutable parent binding for the eager-prefix complete-pipeline pilot.

The parent experiment established exact eager/adaptive output equivalence but
missed its *base-only* runtime forecast by less than one percent.  This module
binds that terminal evidence without importing the transition kernel, trainer,
controller, or sampler.  A successful verification authorizes only a fresh,
pre-registered complete-pipeline timing experiment.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-eager-pipeline-provenance"
SCHEMA_VERSION = 1

PARENT_RUN_BASENAME = (
    "20260803-021405_production-eager-prefix-boundary-tangent-schedule-arbfix-v2"
)
PARENT_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-prefix-schedule-confirmation-v1"
)
PARENT_REGISTRY_COUNT = 33
PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "b3a8a4f7187f49877257ae18b3e94cd66f09bf0b0c9b3b03773c181ad7a01086"
)
PARENT_REGISTRY_FILE_SHA256 = (
    "4ce586988c1dd768ea90e9a2d11d003c8bcf894156dbac47f0abe3056de43631"
)
PARENT_SOURCE_FINGERPRINT = (
    "e30e301ffa330108c986dfb80f32ac2d4d17f648b422a9b047ef5e807e826547"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "165a716e92d74ad9f78f2439d5eb96cf27f8f5756839e34c26c9d84073a9fd0b"
)

PARENT_MANIFEST_FILE_SHA256 = (
    "d8c92055faf4eb54a3b13e5eb2dbf4d7e02e817ccc14a3065a532a04b2b1b6db"
)
PARENT_SCIENTIFIC_CONFIG_FILE_SHA256 = (
    "79c34a4ef70f58251ad091933cd87c1f044d16033884cde36bc4601b1bca3ef6"
)
PARENT_STATUS_FILE_SHA256 = (
    "b3c7abcc593249bd889b0ced9ad1099028aa4968d776a0135b05ad32ef6ac8d7"
)
PARENT_WORKFLOW_FILE_SHA256 = (
    "6c781df999f2c65ec008b0c9162b8fc29dd13a84f6f061f0981a00735c2b03b9"
)
PARENT_DECISION_FILE_SHA256 = (
    "c0395692cde3925e45d0f38f2507d84b134bd5ef249f87c15daf12210892958f"
)
PARENT_PREFLIGHT_GATE_FILE_SHA256 = (
    "70e70f16266b69f1691d933392f840fd021c9d642defb6c63255157e9fa7df41"
)
PARENT_PROFILE_GATE_FILE_SHA256 = (
    "0fc82967da7abd8e2a94589c3d6f4bc5d9669d0f6a83a47fc4f5d284e0bbf598"
)

PARENT_DECISION = "eager_prefix_profile_computationally_infeasible"
PARENT_FAILURE_DOMAIN = "resource_gate"
PARENT_PROJECTED_TRANSITIONS = 337_182_720
PARENT_PROJECTED_SECONDS = 109_060.84962575609
PARENT_PROJECTED_HOURS = PARENT_PROJECTED_SECONDS / 3_600.0
PARENT_PROJECTED_RATE = 3_091.693500986353
PARENT_MAXIMUM_SECONDS = 108_000.0
PARENT_MINIMUM_RATE = PARENT_PROJECTED_TRANSITIONS / PARENT_MAXIMUM_SECONDS
PARENT_PROFILE_NAME = "eager_prefix_128_tpb128"
PARENT_PROFILE_REPEATS = 3
PARENT_CERTIFICATE_FRACTION = 1.0
PARENT_FORBIDDEN_EVENT_COUNT = 0
PARENT_CONSERVATIVE_SPEEDUP = 1.329134034459093
PARENT_SAVED_SECONDS = 19_377.45067184879

_REGISTRY_SEMANTICS = {
    "snapshot_kind": "terminal-exact-with-restartable-pilot-extras",
    "excluded_paths": [
        "artifact_registry.json",
        "prefix_schedule_decision.json",
        "run_status.json",
        "workflow_gate.json",
    ],
    "restartable_extras_must_match_frozen_pilot_layout": 1,
}
_EXCLUDED_FILE_SHA256 = {
    "artifact_registry.json": PARENT_REGISTRY_FILE_SHA256,
    "prefix_schedule_decision.json": PARENT_DECISION_FILE_SHA256,
    "run_status.json": PARENT_STATUS_FILE_SHA256,
    "workflow_gate.json": PARENT_WORKFLOW_FILE_SHA256,
}
_SOURCE_RELATIVE_PATHS = (
    "mnist/d0_jacobi_artifacts.py",
    "mnist/d0_jacobi_rb_boundary_tangent_cache.py",
    "mnist/d0_jacobi_rb_boundary_tangent_eager_prefix_provenance.py",
    "mnist/d0_jacobi_rb_boundary_tangent_prefix_fallback.py",
    "mnist/d0_jacobi_rb_boundary_tangent_prefix_schedule.py",
    "mnist/d0_jacobi_rb_boundary_tangent_prefix_schedule_gate.py",
    "mnist/d0_jacobi_rb_boundary_tangent_schedule.py",
    "mnist/d0_jacobi_rb_boundary_tangent_schedule_gate.py",
    "mnist/d0_jacobi_rb_boundary_tangent_schedule_provenance.py",
    "mnist/d0_jacobi_rb_coarse_residual.py",
    "mnist/d0_jacobi_rb_controls.py",
    "mnist/d0_jacobi_rb_cuda.py",
    "mnist/d0_jacobi_rb_cuda_certificate.py",
    "mnist/d0_jacobi_rb_cuda_controls.py",
    "mnist/d0_jacobi_rb_cuda_fused.py",
    "mnist/d0_jacobi_rb_cuda_multipath.py",
    "mnist/d0_jacobi_rb_learnability.py",
    "mnist/d0_jacobi_rb_reverse_controller.py",
    "mnist/d0_jacobi_rb_spectral.py",
    "mnist/d0_jacobi_source_compat.py",
    "mnist/diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation.py",
    "mnist/diag_d0_jacobi_rb_boundary_tangent_schedule_feasibility.py",
)

NO_WORK_FIELDS = (
    "physical_training_performed",
    "controller_control_trajectory_performed",
    "full_reverse_path_performed",
    "image_sampling_performed",
    "sampling_performed",
    "reverse_sampling_performed",
    "reconstruction_performed",
)
NO_AUTHORIZATION_FIELDS = (
    "cache_generation_authorized",
    "physical_training_authorized",
    "training_authorized",
    "controller_control_trajectory_authorized",
    "controller_trajectory_authorized",
    "image_sampling_authorized",
    "sampling_authorized",
    "reverse_sampling_authorized",
    "reconstruction_authorized",
    "schedule_integration_authorized",
)


class EagerPipelineProvenanceError(ArtifactCompatibilityError):
    """The immutable prefix parent or its exact live sources changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EagerPipelineProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EagerPipelineProvenanceError(
            f"invalid {description}: {path}"
        ) from exc
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


def _semantic_body(record: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(record)
    body.pop("semantic_sha256", None)
    return body


def _assert_semantic(record: Mapping[str, Any], description: str) -> None:
    _require(
        record.get("semantic_sha256") == config_fingerprint(_semantic_body(record)),
        f"{description} semantic hash changed",
    )


def _assert_zero_claims(
    record: Mapping[str, Any], description: str, *, require_work_fields: bool = True
) -> None:
    for field in NO_WORK_FIELDS:
        if require_work_fields:
            _require(field in record, f"{description} omits {field}")
        if field in record:
            _require(int(record[field]) == 0, f"{description} records {field}")
    for field in NO_AUTHORIZATION_FIELDS:
        if field in record:
            _require(int(record[field]) == 0, f"{description} authorizes {field}")


def _safe_registry_path(root: Path, value: Any) -> tuple[str, Path]:
    _require(isinstance(value, str) and bool(value), "registry path is invalid")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe registry path: {value!r}",
    )
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise EagerPipelineProvenanceError(
            f"registry path escapes parent: {value!r}"
        ) from exc
    return relative.as_posix(), target


def _verify_registry(root: Path) -> dict[str, Any]:
    path = root / "artifact_registry.json"
    _require(
        path.is_file() and file_fingerprint(path) == PARENT_REGISTRY_FILE_SHA256,
        "prefix parent registry file hash changed",
    )
    registry = _load_json(path, "prefix parent artifact registry")
    artifacts = registry.get("artifacts")
    semantics = registry.get("registry_semantics")
    _require(
        registry.get("schema") == PARENT_RUN_SCHEMA + "-artifact-registry"
        and registry.get("schema_version") == 1
        and isinstance(artifacts, list)
        and len(artifacts) == PARENT_REGISTRY_COUNT
        and int(registry.get("artifact_count", -1)) == PARENT_REGISTRY_COUNT
        and semantics == _REGISTRY_SEMANTICS
        and registry.get("semantic_sha256") == PARENT_REGISTRY_SEMANTIC_SHA256
        and config_fingerprint({"artifacts": artifacts, "registry_semantics": semantics})
        == PARENT_REGISTRY_SEMANTIC_SHA256,
        "prefix parent terminal registry changed",
    )
    registered: set[str] = set()
    for raw in artifacts:
        _require(isinstance(raw, Mapping), "registry row is malformed")
        relative, target = _safe_registry_path(root, raw.get("path"))
        _require(relative not in registered, "registry path is duplicated")
        _require(
            target.is_file()
            and int(raw.get("size", -1)) == target.stat().st_size
            and raw.get("sha256") == file_fingerprint(target),
            f"prefix parent artifact changed: {relative}",
        )
        registered.add(relative)

    excluded = set(_REGISTRY_SEMANTICS["excluded_paths"])
    actual_registered = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.relative_to(root).as_posix() not in excluded
    }
    _require(actual_registered == registered, "prefix parent terminal file set changed")
    for relative, expected_sha256 in _EXCLUDED_FILE_SHA256.items():
        target = root / relative
        _require(
            target.is_file() and file_fingerprint(target) == expected_sha256,
            f"excluded terminal artifact changed: {relative}",
        )
    _assert_zero_claims(registry, "prefix parent registry")
    return registry


def _resolve_live_sources(manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    raw = manifest.get("source_paths")
    _require(
        isinstance(raw, list)
        and len(raw) == len(_SOURCE_RELATIVE_PATHS)
        and all(isinstance(item, str) and item for item in raw),
        "prefix parent source list changed",
    )
    repository_root = Path(__file__).resolve().parents[1]
    paths = tuple(
        sorted(
            Path(item).resolve()
            if Path(item).is_absolute()
            else (repository_root / item).resolve()
            for item in raw
        )
    )
    expected = tuple(
        sorted((repository_root / relative).resolve() for relative in _SOURCE_RELATIVE_PATHS)
    )
    _require(paths == expected, "prefix parent live source set changed")
    _require(all(path.is_file() for path in paths), "prefix parent source is missing")
    _require(
        source_fingerprint(paths) == PARENT_SOURCE_FINGERPRINT,
        "prefix parent live source fingerprint changed",
    )
    return paths


def _failed_checks(gate: Mapping[str, Any]) -> set[str]:
    checks = gate.get("checks")
    _require(isinstance(checks, Mapping), "profile gate checks changed")
    failed: set[str] = set()
    for name, raw in checks.items():
        _require(isinstance(raw, Mapping), f"profile check is malformed: {name}")
        if int(raw.get("passed", 0)) != 1:
            failed.add(str(name))
    return failed


def verify_eager_pipeline_parent(
    *, parent_prefix_run_dir: str | Path
) -> dict[str, Any]:
    """Verify the exact 33-artifact eager-prefix resource-only failure."""

    root = Path(parent_prefix_run_dir).resolve()
    _require(root.is_dir(), f"prefix parent does not exist: {root}")
    _require(root.name == PARENT_RUN_BASENAME, "wrong prefix parent basename")
    registry = _verify_registry(root)

    manifest = _load_json(root / "run_manifest.json", "prefix parent manifest")
    config = _load_json(root / "scientific_config.json", "prefix parent config")
    status = _load_json(root / "run_status.json", "prefix parent status")
    preflight = _load_json(root / "preflight_gate.json", "prefix preflight gate")
    profile = _load_json(root / "profile_gate.json", "prefix profile gate")
    workflow = _load_json(root / "workflow_gate.json", "prefix workflow")
    decision = _load_json(root / "prefix_schedule_decision.json", "prefix decision")
    selected = _load_json(
        root / "selected_eager_prefix_profile.json", "selected prefix profile"
    )
    qualification = _load_json(
        root / "prefix_profile_qualification.json", "prefix qualification"
    )
    path_plan = _load_json(root / "path_id_plan.json", "prefix path plan")

    for relative, expected in {
        "run_manifest.json": PARENT_MANIFEST_FILE_SHA256,
        "scientific_config.json": PARENT_SCIENTIFIC_CONFIG_FILE_SHA256,
        "run_status.json": PARENT_STATUS_FILE_SHA256,
        "preflight_gate.json": PARENT_PREFLIGHT_GATE_FILE_SHA256,
        "profile_gate.json": PARENT_PROFILE_GATE_FILE_SHA256,
        "workflow_gate.json": PARENT_WORKFLOW_FILE_SHA256,
        "prefix_schedule_decision.json": PARENT_DECISION_FILE_SHA256,
    }.items():
        _require(file_fingerprint(root / relative) == expected, f"{relative} changed")

    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA + "-manifest"
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "prefix parent manifest binding changed",
    )
    _resolve_live_sources(manifest)
    _assert_semantic(config, "prefix parent scientific config")
    _require(
        config.get("semantic_sha256") == PARENT_SCIENTIFIC_CONFIG_SHA256
        and config.get("profile_repeats", config.get("repeat_count"))
        == PARENT_PROFILE_REPEATS
        and config.get("projected_total_transitions") == PARENT_PROJECTED_TRANSITIONS
        and float(config.get("maximum_projected_exact_cache_hours")) == 30.0
        and float(config.get("minimum_effective_projected_rate"))
        == PARENT_MINIMUM_RATE,
        "prefix parent scientific configuration changed",
    )
    _require(
        status.get("state") == "gate_failed"
        and status.get("stage") == "profile"
        and status.get("decision") == PARENT_DECISION
        and status.get("failure_domain") == PARENT_FAILURE_DOMAIN
        and int(status.get("scientific_evidence_complete", 0)) == 1,
        "prefix parent terminal status changed",
    )
    _require(
        int(preflight.get("passed", 0)) == 1
        and preflight.get("evaluation_status") == "evaluated"
        and int(preflight.get("stage_execution_valid", 0)) == 1
        and int(preflight.get("scientific_evidence_complete", 0)) == 1,
        "prefix parent preflight did not pass",
    )
    _require(
        profile.get("evaluation_status") == "evaluated"
        and int(profile.get("passed", 1)) == 0
        and profile.get("failure_domain") == PARENT_FAILURE_DOMAIN
        and int(profile.get("stage_execution_valid", 0)) == 1
        and int(profile.get("numerically_valid", 0)) == 1
        and int(profile.get("resource_valid", 1)) == 0
        and int(profile.get("resource_only_failure", 0)) == 1
        and int(profile.get("scientific_evidence_complete", 0)) == 1
        and _failed_checks(profile)
        == {
            "projected_effective_transitions_per_second",
            "projected_elapsed_seconds",
        },
        "prefix parent profile is not the exact resource-only failure",
    )
    checks = profile["checks"]
    _require(
        float(checks["projected_elapsed_seconds"]["value"])
        == PARENT_PROJECTED_SECONDS
        and float(checks["projected_effective_transitions_per_second"]["value"])
        == PARENT_PROJECTED_RATE,
        "prefix parent failed runtime values changed",
    )
    _require(
        workflow.get("required_gate") == "profile"
        and int(workflow.get("required_gate_pass", 1)) == 0
        and workflow.get("decision", {}).get("decision") == PARENT_DECISION
        and decision.get("decision") == PARENT_DECISION,
        "prefix parent workflow decision changed",
    )
    _require(
        int(selected.get("selected", 1)) == 0
        and selected.get("profile") is None
        and selected.get("profile_name") is None
        and selected.get("evaluation_status") == "not_selected",
        "prefix parent unexpectedly selected a profile",
    )
    _assert_semantic(selected, "selected prefix profile")
    _require(
        qualification.get("profile_name") == PARENT_PROFILE_NAME
        and int(qualification.get("passed", 1)) == 0
        and int(qualification.get("numerically_clean", 0)) == 1
        and len(qualification.get("records", [])) == PARENT_PROFILE_REPEATS
        and float(qualification.get("projected_elapsed_seconds"))
        == PARENT_PROJECTED_SECONDS
        and float(qualification.get("projected_effective_transitions_per_second"))
        == PARENT_PROJECTED_RATE
        and float(qualification.get("conservative_authorizer_speedup"))
        == PARENT_CONSERVATIVE_SPEEDUP
        and float(qualification.get("conservative_saved_seconds"))
        == PARENT_SAVED_SECONDS,
        "prefix parent qualification changed",
    )
    _require(
        int(path_plan.get("collision_free", 0)) == 1
        and path_plan.get("roles", {}).get("cache_p10")
        == list(range(0xEE000, 0xEE00A))
        and path_plan.get("roles", {}).get("cache_p6")
        == list(range(0xEE010, 0xEE016))
        and path_plan.get("roles", {}).get("stream_p10")
        == list(range(0xEE100, 0xEE10A))
        and path_plan.get("roles", {}).get("stream_p4")
        == list(range(0xEE110, 0xEE114)),
        "prefix parent pilot namespace plan changed",
    )
    _require(
        not (root / "pilot").exists()
        and not any(path.startswith("pilot/") for path in (
            str(row.get("path")) for row in registry["artifacts"]
        )),
        "prefix parent pilot namespace was opened",
    )
    for description, record in {
        "prefix parent manifest": manifest,
        "prefix parent status": status,
        "prefix preflight gate": preflight,
        "prefix profile gate": profile,
        "prefix workflow": workflow,
        "prefix decision": decision,
        "prefix qualification": qualification,
    }.items():
        _assert_zero_claims(record, description, require_work_fields=False)

    provenance: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_registry": {
            "artifact_count": PARENT_REGISTRY_COUNT,
            "semantic_sha256": PARENT_REGISTRY_SEMANTIC_SHA256,
            "file_sha256": PARENT_REGISTRY_FILE_SHA256,
        },
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent_decision": PARENT_DECISION,
        "parent_failure_domain": PARENT_FAILURE_DOMAIN,
        "parent_stage_execution_valid": 1,
        "parent_numerically_valid": 1,
        "parent_resource_valid": 0,
        "parent_resource_only_failure": 1,
        "parent_scientific_evidence_complete": 1,
        "parent_profile_name": PARENT_PROFILE_NAME,
        "parent_profile_repeat_count": PARENT_PROFILE_REPEATS,
        "parent_projected_transition_count": PARENT_PROJECTED_TRANSITIONS,
        "parent_projected_seconds": PARENT_PROJECTED_SECONDS,
        "parent_projected_hours": PARENT_PROJECTED_HOURS,
        "parent_projected_effective_rate": PARENT_PROJECTED_RATE,
        "parent_maximum_seconds": PARENT_MAXIMUM_SECONDS,
        "parent_minimum_effective_rate": PARENT_MINIMUM_RATE,
        "only_runtime_checks_failed": 1,
        "pilot_namespaces_unopened": 1,
        "complete_pipeline_timing_authorized": 1,
        "parent_artifacts_mutated": 0,
        "production_cache_generation_performed": 0,
        **{field: 0 for field in NO_WORK_FIELDS},
        **{field: 0 for field in NO_AUTHORIZATION_FIELDS},
    }
    provenance["semantic_sha256"] = config_fingerprint(provenance)
    return provenance


def build_eager_pipeline_parent_readjudication(
    provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify the base-only projection as inconclusive, without a pass."""

    _require(int(provenance.get("passed", 0)) == 1, "parent provenance did not pass")
    _require(
        provenance.get("parent_decision") == PARENT_DECISION
        and int(provenance.get("only_runtime_checks_failed", 0)) == 1,
        "parent evidence cannot be readjudicated",
    )
    record: dict[str, Any] = {
        "schema": SCHEMA + "-parent-readjudication",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "historical_decision": PARENT_DECISION,
        "historical_failure_domain": PARENT_FAILURE_DOMAIN,
        "readjudicated_decision": "base_only_projection_inconclusive",
        "readjudicated_failure_domain": "resource_forecast",
        "historical_gate_mutated": 0,
        "parent_artifacts_mutated": 0,
        "parent_projected_seconds": PARENT_PROJECTED_SECONDS,
        "maximum_projected_seconds": PARENT_MAXIMUM_SECONDS,
        "complete_pipeline_timing_authorized": 1,
        "scientific_evidence_complete": 1,
        "production_cache_generation_performed": 0,
        **{field: 0 for field in NO_WORK_FIELDS},
        **{field: 0 for field in NO_AUTHORIZATION_FIELDS},
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


__all__ = [
    "EagerPipelineProvenanceError",
    "PARENT_DECISION",
    "PARENT_FAILURE_DOMAIN",
    "PARENT_MAXIMUM_SECONDS",
    "PARENT_MINIMUM_RATE",
    "PARENT_PROJECTED_RATE",
    "PARENT_PROJECTED_SECONDS",
    "PARENT_PROJECTED_TRANSITIONS",
    "PARENT_REGISTRY_COUNT",
    "PARENT_REGISTRY_FILE_SHA256",
    "PARENT_REGISTRY_SEMANTIC_SHA256",
    "PARENT_RUN_BASENAME",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_FINGERPRINT",
    "SCHEMA",
    "SCHEMA_VERSION",
    "build_eager_pipeline_parent_readjudication",
    "verify_eager_pipeline_parent",
]
