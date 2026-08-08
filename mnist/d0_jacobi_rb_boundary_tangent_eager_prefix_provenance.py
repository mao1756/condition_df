"""Immutable parent binding for the eager-prefix schedule experiment.

This additive module verifies the completed fused-lane scheduling pilot that
failed only its frozen resource gate.  It deliberately imports no transition
kernel, model, trainer, controller, or sampler.  The parent directory and all
source files fingerprinted by that parent remain immutable.
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


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-eager-prefix-provenance"
SCHEMA_VERSION = 1

PARENT_RUN_BASENAME = (
    "20260802-174811_production-fused-boundary-tangent-schedule"
)
PARENT_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-schedule-feasibility-v1"
)
PARENT_REGISTRY_COUNT = 614
PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "3a9372dee33287c2bc3d2e2752d6b206a44cd070c91305dc9d921bb2521e688e"
)
PARENT_REGISTRY_FILE_SHA256 = (
    "23ae4a0b3448b57b4982c9d61eca2ea5df1834057cc7035dc1084b0c28cf6ea9"
)
PARENT_SOURCE_FINGERPRINT = (
    "c562b8e39a07bbc19a9f65b9a3187c49d97cb64a277bf9492f4cb7fb92c9b2ee"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "0820d8e5051adae996d8c79bc1e933c8725e958aba62ff6c4bed1525a6cc6845"
)
PARENT_MANIFEST_FILE_SHA256 = (
    "7e5aecdeb4410f3223f71431444251013840857cfd623f6e842d74aafa4ce651"
)
PARENT_SCIENTIFIC_CONFIG_FILE_SHA256 = (
    "7e15295fef672dd8adb15ca98a43522ea871e02a7c5a3b13704782ece036d1ef"
)
PARENT_STATUS_FILE_SHA256 = (
    "bc66bd698cdf881106911720ac16c6f05a5ec5100e22f00fb51a66d137d0240e"
)
PARENT_WORKFLOW_FILE_SHA256 = (
    "0cb92d4eb61efb31f0d8af7dbc4a98882074823d4d9f2347f706fa2ed32ce509"
)
PARENT_DECISION_FILE_SHA256 = (
    "c50af39752f8d053854567fd207e02ab99076f99e8dbda294ceac846528f9b36"
)
PARENT_PREFLIGHT_GATE_FILE_SHA256 = (
    "b21c42fb5958ffbed6964bdb792bc760f63c3c56961d17608656c1bc5236c902"
)
PARENT_PILOT_GATE_FILE_SHA256 = (
    "40617c27ac5450348e3062b66dc5798d462a8152b842cb96f1896c557868206f"
)

PARENT_DECISION = "boundary_tangent_schedule_computationally_infeasible"
PARENT_FAILURE_DOMAIN = "resource_gate"
PARENT_PROJECTED_TRANSITIONS = 337_182_720
PARENT_EXECUTED_PILOT_TRANSITIONS = 23_708_160
PARENT_PROJECTED_SECONDS = 128_438.30029760487
PARENT_PROJECTED_HOURS = 35.67730563822358
PARENT_PROJECTED_RATE = 2_625.2505616993735
PARENT_MAXIMUM_SECONDS = 108_000.0
PARENT_MAXIMUM_HOURS = 30.0
PARENT_MINIMUM_RATE = 3_122.0622222222223

_REGISTRY_SEMANTICS = {
    "snapshot_kind": "terminal-exact-with-restartable-pilot-extras",
    "excluded_paths": [
        "artifact_registry.json",
        "run_status.json",
        "schedule_decision.json",
        "workflow_gate.json",
    ],
    "restartable_extras_must_match_frozen_pilot_layout": 1,
}
_EXCLUDED_FILE_SHA256 = {
    "artifact_registry.json": PARENT_REGISTRY_FILE_SHA256,
    "run_status.json": PARENT_STATUS_FILE_SHA256,
    "schedule_decision.json": PARENT_DECISION_FILE_SHA256,
    "workflow_gate.json": PARENT_WORKFLOW_FILE_SHA256,
}

_SOURCE_RELATIVE_PATHS = (
    "mnist/d0_jacobi_artifacts.py",
    "mnist/d0_jacobi_rb_boundary_tangent_cache.py",
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
    "controller_control_trajectory_authorized",
    "image_sampling_authorized",
    "sampling_authorized",
    "reverse_sampling_authorized",
    "reconstruction_authorized",
)


class EagerPrefixScheduleProvenanceError(ArtifactCompatibilityError):
    """The immutable fused-schedule parent or its live sources changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EagerPrefixScheduleProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EagerPrefixScheduleProvenanceError(
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
    record: Mapping[str, Any],
    description: str,
    *,
    require_work_fields: bool = True,
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
        raise EagerPrefixScheduleProvenanceError(
            f"registry path escapes parent: {value!r}"
        ) from exc
    return relative.as_posix(), target


def _verify_registry(root: Path) -> dict[str, Any]:
    path = root / "artifact_registry.json"
    _require(
        path.is_file() and file_fingerprint(path) == PARENT_REGISTRY_FILE_SHA256,
        "fused-schedule registry file hash changed",
    )
    registry = _load_json(path, "fused-schedule artifact registry")
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
        and config_fingerprint(
            {"artifacts": artifacts, "registry_semantics": semantics}
        )
        == PARENT_REGISTRY_SEMANTIC_SHA256,
        "fused-schedule terminal registry changed",
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
            f"fused-schedule artifact changed: {relative}",
        )
        registered.add(relative)

    excluded = set(_REGISTRY_SEMANTICS["excluded_paths"])
    actual_registered = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.relative_to(root).as_posix() not in excluded
    }
    _require(
        actual_registered == registered,
        "fused-schedule terminal file set changed",
    )
    for relative, expected_sha256 in _EXCLUDED_FILE_SHA256.items():
        target = root / relative
        _require(
            target.is_file() and file_fingerprint(target) == expected_sha256,
            f"excluded terminal artifact changed: {relative}",
        )
    _assert_zero_claims(registry, "fused-schedule registry")
    return registry


def _resolve_live_sources(manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    raw = manifest.get("source_paths")
    _require(
        isinstance(raw, list)
        and len(raw) == len(_SOURCE_RELATIVE_PATHS)
        and all(isinstance(item, str) and item for item in raw),
        "fused-schedule source list changed",
    )
    repository_root = Path(__file__).resolve().parents[1]
    paths = tuple(
        sorted(
            (
                Path(item).resolve()
                if Path(item).is_absolute()
                else (repository_root / item).resolve()
            )
            for item in raw
        )
    )
    expected = tuple(
        sorted((repository_root / value).resolve() for value in _SOURCE_RELATIVE_PATHS)
    )
    _require(paths == expected, "fused-schedule live source set changed")
    _require(all(path.is_file() for path in paths), "fused-schedule source is missing")
    _require(
        source_fingerprint(paths) == PARENT_SOURCE_FINGERPRINT,
        "fused-schedule live source fingerprint changed",
    )
    return paths


def _failed_checks(gate: Mapping[str, Any]) -> set[str]:
    checks = gate.get("checks")
    _require(isinstance(checks, Mapping), "pilot gate checks changed")
    failed: set[str] = set()
    for name, raw in checks.items():
        _require(isinstance(raw, Mapping), f"pilot check is malformed: {name}")
        if int(raw.get("passed", 0)) != 1:
            failed.add(str(name))
    return failed


def verify_eager_prefix_schedule_parent(
    *, parent_schedule_run_dir: str | Path
) -> dict[str, Any]:
    """Verify the exact 614-artifact resource-only scheduling failure.

    A passing result authorizes only an additive eager-prefix scheduling
    feasibility experiment.  It does not authorize cache generation, model
    training, a controller trajectory, reconstruction, or sampling.
    """

    root = Path(parent_schedule_run_dir).resolve()
    _require(root.is_dir(), f"fused-schedule parent does not exist: {root}")
    _require(root.name == PARENT_RUN_BASENAME, "wrong fused-schedule parent basename")
    registry = _verify_registry(root)

    manifest = _load_json(root / "run_manifest.json", "parent manifest")
    config = _load_json(root / "scientific_config.json", "parent config")
    status = _load_json(root / "run_status.json", "parent status")
    preflight = _load_json(root / "preflight_gate.json", "parent preflight gate")
    pilot = _load_json(root / "pilot_gate.json", "parent pilot gate")
    metrics = _load_json(root / "pilot_metrics.json", "parent pilot metrics")
    projection = _load_json(root / "schedule_projection.json", "parent projection")
    workflow = _load_json(root / "workflow_gate.json", "parent workflow")
    decision = _load_json(root / "schedule_decision.json", "parent decision")

    _require(
        file_fingerprint(root / "run_manifest.json")
        == PARENT_MANIFEST_FILE_SHA256
        and file_fingerprint(root / "scientific_config.json")
        == PARENT_SCIENTIFIC_CONFIG_FILE_SHA256
        and file_fingerprint(root / "preflight_gate.json")
        == PARENT_PREFLIGHT_GATE_FILE_SHA256
        and file_fingerprint(root / "pilot_gate.json")
        == PARENT_PILOT_GATE_FILE_SHA256,
        "fused-schedule key artifact hash changed",
    )
    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA + "-manifest"
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "fused-schedule manifest binding changed",
    )
    _resolve_live_sources(manifest)
    _require(
        config.get("schema") == PARENT_RUN_SCHEMA + "-scientific-config"
        and config.get("schema_version") == 1
        and config.get("semantic_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256
        and config.get("maximum_projected_exact_cache_hours")
        == PARENT_MAXIMUM_HOURS
        and int(config.get("projected_total_transitions", -1))
        == PARENT_PROJECTED_TRANSITIONS
        and int(config.get("test_only", 1)) == 0,
        "fused-schedule scientific configuration changed",
    )
    _assert_semantic(config, "fused-schedule scientific configuration")

    _require(
        preflight.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-schedule-gate-preflight-gate"
        and preflight.get("evaluation_status") == "evaluated"
        and int(preflight.get("passed", 0)) == 1
        and int(preflight.get("stage_execution_valid", 0)) == 1
        and int(preflight.get("numerically_valid", 0)) == 1
        and int(preflight.get("resource_valid", 0)) == 1
        and int(preflight.get("scientific_evidence_complete", 0)) == 1,
        "fused-schedule preflight did not pass exactly",
    )
    _require(
        pilot.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-schedule-gate-pilot-gate"
        and pilot.get("evaluation_status") == "evaluated"
        and int(pilot.get("passed", 1)) == 0
        and int(pilot.get("stage_execution_valid", 0)) == 1
        and int(pilot.get("numerically_valid", 0)) == 1
        and int(pilot.get("resource_valid", 1)) == 0
        and int(pilot.get("resource_only_failure", 0)) == 1
        and int(pilot.get("scientific_evidence_complete", 0)) == 1
        and pilot.get("failure_domain") == PARENT_FAILURE_DOMAIN
        and _failed_checks(pilot)
        == {
            "projected_effective_transitions_per_second",
            "projected_elapsed_seconds",
            "projected_exact_cache_hours",
        },
        "fused-schedule pilot is not the exact resource-only failure",
    )
    _require(
        metrics.get("schema") == PARENT_RUN_SCHEMA + "-pilot-metrics"
        and int(metrics.get("all_profiles_complete", 0)) == 1
        and int(metrics.get("pilot_repeats", -1)) == 3
        and int(metrics.get("completed_shard_count", -1)) == 96
        and int(metrics.get("pilot_total_executed_transition_count", -1))
        == PARENT_EXECUTED_PILOT_TRANSITIONS
        and int(metrics.get("projected_transition_count", -1))
        == PARENT_PROJECTED_TRANSITIONS
        and metrics.get("projected_elapsed_seconds") == PARENT_PROJECTED_SECONDS
        and metrics.get("projected_effective_transitions_per_second")
        == PARENT_PROJECTED_RATE
        and metrics.get("certificate_fraction") == 1.0
        and int(metrics.get("repeat_hash_mismatch_count", -1)) == 0
        and int(metrics.get("forbidden_total", 0)) == 0,
        "fused-schedule pilot metrics changed",
    )
    _require(
        projection.get("schema")
        == "d0-jacobi-rb-boundary-tangent-fused-schedule-v1-projection"
        and int(projection.get("passed", 1)) == 0
        and set(projection.get("failed_checks", ()))
        == {"projected_seconds", "projected_effective_rate"}
        and int(projection.get("projected_total_transitions", -1))
        == PARENT_PROJECTED_TRANSITIONS
        and projection.get("projected_seconds") == PARENT_PROJECTED_SECONDS
        and projection.get("projected_hours") == PARENT_PROJECTED_HOURS
        and projection.get("projected_effective_rate") == PARENT_PROJECTED_RATE
        and projection.get("maximum_projected_exact_cache_hours")
        == PARENT_MAXIMUM_HOURS
        and int(projection.get("forbidden_total", -1)) == 0,
        "fused-schedule projection changed",
    )
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA + "-status"
        and status.get("state") == "gate_failed"
        and status.get("stage") == "pilot"
        and status.get("decision") == PARENT_DECISION
        and status.get("failure_domain") == PARENT_FAILURE_DOMAIN
        and status.get("failure_code") == "pilot_gate_failed"
        and int(status.get("scientific_evidence_complete", 0)) == 1,
        "fused-schedule terminal status changed",
    )
    workflow_decision = workflow.get("decision")
    _require(
        workflow.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-schedule-gate-workflow"
        and workflow.get("evaluation_status") == "evaluated"
        and workflow.get("required_gate") == "pilot"
        and int(workflow.get("required_gate_pass", 1)) == 0
        and isinstance(workflow_decision, Mapping)
        and workflow_decision.get("decision") == PARENT_DECISION,
        "fused-schedule workflow changed",
    )
    _require(
        decision.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-schedule-gate-decision"
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == PARENT_DECISION
        and int(decision.get("schedule_integration_authorized", 1)) == 0,
        "fused-schedule decision changed",
    )
    for description, record in (
        ("manifest", manifest),
        ("scientific configuration", config),
        ("status", status),
        ("preflight", preflight),
        ("pilot", pilot),
        ("workflow", workflow),
        ("decision", decision),
    ):
        _assert_zero_claims(record, f"fused-schedule {description}")

    prohibited_prefixes = (
        "cache/",
        "checkpoints/",
        "confirmation/",
        "controller_trajectory/",
        "reconstruction/",
        "samples/",
        "training/",
    )
    registered_paths = tuple(str(item["path"]) for item in registry["artifacts"])
    _require(
        not any(
            relative.startswith(prefix)
            for relative in registered_paths
            for prefix in prohibited_prefixes
        ),
        "fused-schedule parent contains prohibited downstream work",
    )

    result: dict[str, Any] = {
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
        "parent_projected_transition_count": PARENT_PROJECTED_TRANSITIONS,
        "parent_projected_seconds": PARENT_PROJECTED_SECONDS,
        "parent_projected_exact_cache_hours": PARENT_PROJECTED_HOURS,
        "parent_projected_effective_rate": PARENT_PROJECTED_RATE,
        "maximum_projected_seconds": PARENT_MAXIMUM_SECONDS,
        "maximum_projected_exact_cache_hours": PARENT_MAXIMUM_HOURS,
        "minimum_projected_effective_rate": PARENT_MINIMUM_RATE,
        "parent_artifacts_mutated": 0,
        "eager_prefix_schedule_feasibility_authorized": 1,
        **{field: 0 for field in NO_WORK_FIELDS},
        **{field: 0 for field in NO_AUTHORIZATION_FIELDS},
    }
    result["semantic_sha256"] = config_fingerprint(result)
    return result


__all__ = [
    "EagerPrefixScheduleProvenanceError",
    "PARENT_DECISION",
    "PARENT_REGISTRY_COUNT",
    "PARENT_REGISTRY_FILE_SHA256",
    "PARENT_REGISTRY_SEMANTIC_SHA256",
    "PARENT_RUN_BASENAME",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_FINGERPRINT",
    "SCHEMA",
    "SCHEMA_VERSION",
    "verify_eager_prefix_schedule_parent",
]
