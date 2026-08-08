"""Immutable provenance and frozen plans for the fused-lane schedule gate.

The admissible historical run failed only its 30-hour cache projection.  This
module verifies that immutable record, re-verifies its successful coarse
learner transitively, and allocates fresh benchmark namespaces.  It contains
no transition kernel, trainer, controller, or sampler.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_provenance import (
    EXPECTED_COARSE_RESIDUAL_REGISTRY_COUNT,
    EXPECTED_COARSE_RESIDUAL_REGISTRY_FILE_SHA256,
    EXPECTED_COARSE_RESIDUAL_REGISTRY_SEMANTIC_SHA256,
    EXPECTED_COARSE_RESIDUAL_RUN_BASENAME,
    EXPECTED_COARSE_RESIDUAL_SCIENTIFIC_CONFIG_SHA256,
    EXPECTED_COARSE_RESIDUAL_SOURCE_FINGERPRINT,
    verify_boundary_tangent_parents,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-schedule-provenance"
SCHEMA_VERSION = 1
PATH_PLAN_SCHEMA = SCHEMA + "-path-id-plan"
PATH_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-schedule-path-ids-v1"
COHORT_PLAN_SCHEMA = SCHEMA + "-cohort-plan"
COHORT_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-fused-cohorts-v1"
TIMING_PLAN_SCHEMA = SCHEMA + "-timing-plan"
TIMING_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-timing-v1"

FAILED_RUN_BASENAME = (
    "20260802-140158_production-boundary-tangent-rb-controller"
)
FAILED_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-controller-v1"
)
FAILED_REGISTRY_COUNT = 14
FAILED_REGISTRY_SEMANTIC_SHA256 = (
    "1171249fe72f8584b7f12857cb458957b23afc9ddbeec4f7f281e158f3699238"
)
FAILED_REGISTRY_FILE_SHA256 = (
    "64bfea0b4323081dc621ed5a19351e5209e6c89dfc8fea8499f45982e4c1217f"
)
FAILED_SOURCE_FINGERPRINT = (
    "cc46891556ca4ea00ab4f72b545d2452ccfb7ff9d47d467170f00b6396fad441"
)
FAILED_SCIENTIFIC_CONFIG_SHA256 = (
    "92b13421dd6838f30f898b1e3fc00535403213550ee35b04ea1e5e08cdbc048e"
)
FAILED_PARENT_PROVENANCE_SHA256 = (
    "3fa216445055636a690fbbebef939d5553146960b98063164f9ebd8b570ac859"
)
HISTORICAL_DECISION = "boundary_tangent_design_infeasible"
READJUDICATED_DECISION = "eight_path_cache_schedule_resource_infeasible"
HISTORICAL_TRANSITION_COUNT = 337_182_720
HISTORICAL_RATE = 2_864.1744357592156
HISTORICAL_PROJECTED_HOURS = 32.70117402672768
HISTORICAL_MAXIMUM_HOURS = 30.0

ROOT_SEED = 261_321
PATH_ID_BITS = 20
PATH_ID_LIMIT = 1 << PATH_ID_BITS
PATH_ROLE_RANGES: dict[str, tuple[int, int]] = {
    "cache_p10": (0xEE000, 0xEE00A),
    "cache_p6": (0xEE010, 0xEE016),
    "stream_p10": (0xEE100, 0xEE10A),
    "stream_p4": (0xEE110, 0xEE114),
    "cuda_warmup": (0xEE200, 0xEE20A),
}
PROFILE_PATH_ROLES = ("cache_p10", "cache_p6", "stream_p10", "stream_p4")
PROFILE_ORDER = PROFILE_PATH_ROLES
TIMING_WINDOW_STARTS = (0, 128, 256, 384)
TIMING_WINDOW_OUTER_STEPS = 16
TIMING_BRANCH_STEPS = (15, 143, 271, 399)
PRODUCTION_CACHE_GROUP_SIZES = (10,) * 9 + (6,)
PRODUCTION_STREAM_GROUP_SIZES = (10,) * 6 + (4,)

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
    "sampling_authorized",
    "reverse_sampling_authorized",
    "reconstruction_authorized",
    "reconstruction_claim_authorized",
    "one_image_reconstruction_control_planning_authorized",
)


class BoundaryTangentScheduleProvenanceError(ArtifactCompatibilityError):
    """An immutable parent or fresh schedule binding changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryTangentScheduleProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundaryTangentScheduleProvenanceError(
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


def _assert_no_work(record: Mapping[str, Any], description: str) -> None:
    for field in NO_WORK_FIELDS:
        _require(int(record.get(field, 0)) == 0, f"{description} records {field}")
    for field in NO_AUTHORIZATION_FIELDS:
        if field in record:
            _require(
                int(record.get(field, 0)) == 0,
                f"{description} authorizes {field}",
            )


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
        raise BoundaryTangentScheduleProvenanceError(
            f"registry path escapes parent: {value!r}"
        ) from exc
    return relative.as_posix(), target


def _verify_registry(root: Path) -> dict[str, Any]:
    path = root / "artifact_registry.json"
    _require(
        file_fingerprint(path) == FAILED_REGISTRY_FILE_SHA256,
        "failed boundary-tangent registry file hash changed",
    )
    registry = _load_json(path, "failed boundary-tangent registry")
    artifacts = registry.get("artifacts")
    _require(
        registry.get("schema") == FAILED_RUN_SCHEMA + "-artifact-registry"
        and registry.get("schema_version") == 1
        and isinstance(artifacts, list)
        and int(registry.get("artifact_count", -1)) == FAILED_REGISTRY_COUNT
        and len(artifacts) == FAILED_REGISTRY_COUNT
        and registry.get("semantic_sha256") == FAILED_REGISTRY_SEMANTIC_SHA256
        and config_fingerprint({"artifacts": artifacts})
        == FAILED_REGISTRY_SEMANTIC_SHA256,
        "failed boundary-tangent terminal registry changed",
    )
    registered: set[str] = set()
    for raw in artifacts:
        _require(isinstance(raw, Mapping), "failed registry row is malformed")
        relative, target = _safe_registry_path(root, raw.get("path"))
        _require(relative not in registered, "failed registry path is duplicated")
        _require(
            target.is_file()
            and int(raw.get("size", -1)) == target.stat().st_size
            and raw.get("sha256") == file_fingerprint(target),
            f"failed boundary-tangent artifact changed: {relative}",
        )
        registered.add(relative)
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.name not in {"artifact_registry.json", "run_status.json"}
    }
    _require(actual == registered, "failed boundary-tangent terminal file set changed")
    _assert_no_work(registry, "failed boundary-tangent registry")
    return registry


def _resolve_live_sources(manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    raw = manifest.get("source_paths")
    _require(
        isinstance(raw, list)
        and len(raw) == 6
        and all(isinstance(item, str) and item for item in raw),
        "failed boundary-tangent source list changed",
    )
    repository_root = Path(__file__).resolve().parents[1]
    paths = tuple(
        sorted(
            {
                (
                    Path(item).resolve()
                    if Path(item).is_absolute()
                    else (repository_root / item).resolve()
                )
                for item in raw
            },
            key=lambda item: item.as_posix(),
        )
    )
    _require(
        len(paths) == 6 and all(item.is_file() for item in paths),
        "failed boundary-tangent live source set changed",
    )
    _require(
        source_fingerprint(paths) == FAILED_SOURCE_FINGERPRINT,
        "failed boundary-tangent live source fingerprint changed",
    )
    return paths


def _failed_check_names(checks: Mapping[str, Any]) -> set[str]:
    return {
        str(name)
        for name, value in checks.items()
        if not isinstance(value, (bool, int)) or int(value) != 1
    }


def verify_and_readjudicate_boundary_tangent_schedule_parents(
    *,
    failed_boundary_tangent_run_dir: str | Path,
    parent_coarse_residual_run_dir: str | Path,
) -> dict[str, Any]:
    """Verify the exact resource-only failure and successful coarse parent."""

    root = Path(failed_boundary_tangent_run_dir).resolve()
    coarse_root = Path(parent_coarse_residual_run_dir).resolve()
    _require(root.is_dir(), f"failed boundary-tangent run does not exist: {root}")
    _require(root.name == FAILED_RUN_BASENAME, "wrong failed run basename")
    _require(
        coarse_root.name == EXPECTED_COARSE_RESIDUAL_RUN_BASENAME,
        "wrong successful coarse-residual parent basename",
    )
    registry = _verify_registry(root)
    manifest = _load_json(root / "run_manifest.json", "failed run manifest")
    status = _load_json(root / "run_status.json", "failed run status")
    config = _load_json(root / "scientific_config.json", "failed run config")
    decision = _load_json(root / "controller_decision.json", "failed decision")
    workflow = _load_json(root / "workflow_gate.json", "failed workflow gate")
    preflight = _load_json(root / "preflight_gate.json", "failed preflight gate")
    resource = _load_json(root / "resource_projection.json", "resource projection")
    parent_provenance = _load_json(
        root / "parent_provenance.json", "failed run parent provenance"
    )

    _require(
        manifest.get("schema") == FAILED_RUN_SCHEMA + "-manifest"
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == FAILED_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == FAILED_SCIENTIFIC_CONFIG_SHA256
        and manifest.get("parent_provenance_sha256")
        == FAILED_PARENT_PROVENANCE_SHA256,
        "failed boundary-tangent manifest binding changed",
    )
    _resolve_live_sources(manifest)
    _require(
        config.get("schema") == FAILED_RUN_SCHEMA + "-scientific-config"
        and config.get("schema_version") == 1
        and config.get("semantic_sha256") == FAILED_SCIENTIFIC_CONFIG_SHA256,
        "failed boundary-tangent scientific configuration changed",
    )
    _assert_semantic(config, "failed boundary-tangent scientific configuration")
    _require(
        status.get("schema") == FAILED_RUN_SCHEMA + "-status"
        and status.get("state") == "gate_failed"
        and status.get("stage") == "preflight"
        and status.get("decision") == HISTORICAL_DECISION
        and status.get("failure_domain") == "scientific_gate"
        and status.get("failure_code") == "preflight_gate_failed",
        "failed boundary-tangent historical status changed",
    )
    _require(
        decision.get("schema") == "d0-jacobi-rb-boundary-tangent-gate-v1-decision"
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == HISTORICAL_DECISION,
        "failed boundary-tangent historical decision changed",
    )
    _require(
        preflight.get("schema") == FAILED_RUN_SCHEMA + "-preflight-gate"
        and preflight.get("evaluation_status") == "evaluated"
        and int(preflight.get("passed", 1)) == 0
        and int(preflight.get("numerically_valid", 0)) == 1
        and int(preflight.get("resource_valid", 1)) == 0
        and _failed_check_names(dict(preflight.get("checks", {})))
        == {"resource_projection"},
        "failed preflight is not a numerical pass with one resource failure",
    )
    _require(
        resource.get("schema") == FAILED_RUN_SCHEMA + "-resource-projection"
        and int(resource.get("passed", 1)) == 0
        and _failed_check_names(dict(resource.get("checks", {})))
        == {"projected_runtime"}
        and int(resource.get("projected_transition_count", -1))
        == HISTORICAL_TRANSITION_COUNT
        and resource.get("transitions_per_second") == HISTORICAL_RATE
        and resource.get("projected_exact_cache_hours")
        == HISTORICAL_PROJECTED_HOURS
        and resource.get("certificate_fraction") == 1.0
        and resource.get("fallback_fraction") == 0.0,
        "failed resource projection changed",
    )
    stages = workflow.get("stage_gates")
    _require(
        workflow.get("required_gate") == "preflight"
        and int(workflow.get("required_gate_pass", 1)) == 0
        and isinstance(stages, Mapping)
        and dict(stages.get("preflight", {})).get("evaluation_status")
        == "evaluated"
        and all(
            dict(stages.get(name, {})).get("evaluation_status") == "not_evaluated"
            for name in ("cache", "train", "confirm", "control")
        ),
        "failed run unexpectedly opened a downstream stage",
    )
    _require(
        not any(
            (root / name).exists()
            for name in (
                "cache_gate.json",
                "train_gate.json",
                "confirm_gate.json",
                "control_gate.json",
                "cache_index.json",
                "selected_model.pt",
                "confirmation_open.json",
            )
        ),
        "failed run contains downstream evidence",
    )
    for description, record in (
        ("manifest", manifest),
        ("status", status),
        ("decision", decision),
        ("workflow", workflow),
        ("preflight", preflight),
        ("resource projection", resource),
    ):
        _assert_no_work(record, f"failed boundary-tangent {description}")

    _require(
        parent_provenance.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-provenance"
        and int(parent_provenance.get("passed", 0)) == 1
        and parent_provenance.get("semantic_sha256")
        == FAILED_PARENT_PROVENANCE_SHA256,
        "failed run transitive parent record changed",
    )
    _assert_semantic(parent_provenance, "failed run parent provenance")
    parents = parent_provenance.get("parents")
    _require(isinstance(parents, Mapping), "failed run parent map changed")
    failed_affine = dict(parents.get("failed_affine_reverse_controller", {}))
    coarse_saved = dict(parents.get("successful_coarse_residual", {}))
    _require(
        Path(str(coarse_saved.get("run_dir", ""))).resolve() == coarse_root
        and int(dict(coarse_saved.get("registry", {})).get("artifact_count", -1))
        == EXPECTED_COARSE_RESIDUAL_REGISTRY_COUNT
        and dict(coarse_saved.get("registry", {})).get("semantic_sha256")
        == EXPECTED_COARSE_RESIDUAL_REGISTRY_SEMANTIC_SHA256
        and dict(coarse_saved.get("registry", {})).get("file_sha256")
        == EXPECTED_COARSE_RESIDUAL_REGISTRY_FILE_SHA256
        and coarse_saved.get("source_fingerprint")
        == EXPECTED_COARSE_RESIDUAL_SOURCE_FINGERPRINT
        and coarse_saved.get("scientific_config_sha256")
        == EXPECTED_COARSE_RESIDUAL_SCIENTIFIC_CONFIG_SHA256,
        "failed run does not bind the exact successful coarse parent",
    )
    verified_transitive = verify_boundary_tangent_parents(
        coarse_residual_run_dir=coarse_root,
        failed_controller_run_dir=failed_affine.get("run_dir"),
    )
    _require(
        verified_transitive.get("semantic_sha256")
        == FAILED_PARENT_PROVENANCE_SHA256
        and config_fingerprint(_semantic_body(verified_transitive))
        == FAILED_PARENT_PROVENANCE_SHA256,
        "live transitive parent verification changed",
    )

    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "failed_run_dir": str(root),
        "failed_run_basename": FAILED_RUN_BASENAME,
        "failed_registry": {
            "artifact_count": int(registry["artifact_count"]),
            "semantic_sha256": FAILED_REGISTRY_SEMANTIC_SHA256,
            "file_sha256": FAILED_REGISTRY_FILE_SHA256,
        },
        "failed_source_fingerprint": FAILED_SOURCE_FINGERPRINT,
        "failed_scientific_config_sha256": FAILED_SCIENTIFIC_CONFIG_SHA256,
        "coarse_parent": {
            "run_dir": str(coarse_root),
            "basename": EXPECTED_COARSE_RESIDUAL_RUN_BASENAME,
            "registry_count": EXPECTED_COARSE_RESIDUAL_REGISTRY_COUNT,
            "registry_semantic_sha256": (
                EXPECTED_COARSE_RESIDUAL_REGISTRY_SEMANTIC_SHA256
            ),
            "registry_file_sha256": EXPECTED_COARSE_RESIDUAL_REGISTRY_FILE_SHA256,
            "source_fingerprint": EXPECTED_COARSE_RESIDUAL_SOURCE_FINGERPRINT,
            "scientific_config_sha256": (
                EXPECTED_COARSE_RESIDUAL_SCIENTIFIC_CONFIG_SHA256
            ),
        },
        "historical_decision": HISTORICAL_DECISION,
        "historical_failure_domain": "scientific_gate",
        "readjudicated_decision": READJUDICATED_DECISION,
        "readjudicated_failure_domain": "resource_gate",
        "scientific_evidence_complete": 1,
        "numerically_valid": 1,
        "resource_valid": 0,
        "only_failed_check": "projected_runtime",
        "projected_transition_count": HISTORICAL_TRANSITION_COUNT,
        "measured_transitions_per_second": HISTORICAL_RATE,
        "projected_exact_cache_hours": HISTORICAL_PROJECTED_HOURS,
        "maximum_projected_exact_cache_hours": HISTORICAL_MAXIMUM_HOURS,
        "parent_artifacts_mutated": 0,
        "fused_schedule_feasibility_authorized": 1,
        **{field: 0 for field in NO_WORK_FIELDS},
    }
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _hashed_record(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def build_schedule_path_plan() -> dict[str, Any]:
    """Return the frozen fresh benchmark path namespaces."""

    return _hashed_record(
        {
            "schema": PATH_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "path_id_plan_version": PATH_PLAN_VERSION,
            "root_seed": ROOT_SEED,
            "canonical_path_id_bits": PATH_ID_BITS,
            "roles": {
                name: list(range(start, stop))
                for name, (start, stop) in PATH_ROLE_RANGES.items()
            },
            "role_slots": {
                name: {
                    "start": start,
                    "stop_exclusive": stop,
                    "start_hex": f"0x{start:05X}",
                    "stop_exclusive_hex": f"0x{stop:05X}",
                }
                for name, (start, stop) in PATH_ROLE_RANGES.items()
            },
            "profile_order": list(PROFILE_ORDER),
            "fresh_path_count": sum(stop - start for start, stop in PATH_ROLE_RANGES.values()),
            "collision_free": 1,
            "silent_remapping_performed": 0,
        }
    )


def validate_schedule_path_plan(
    plan: Mapping[str, Any],
    *,
    claimed_ids: Mapping[str, Iterable[int]] | Iterable[int] | None = None,
) -> dict[str, Any]:
    expected = build_schedule_path_plan()
    _require(dict(plan) == expected, "schedule path plan changed")
    active: set[int] = set()
    intervals = list(PATH_ROLE_RANGES.items())
    for index, (role, (start, stop)) in enumerate(intervals):
        _require(0 <= start < stop <= PATH_ID_LIMIT, f"{role} path slot is invalid")
        for other, (other_start, other_stop) in intervals[index + 1 :]:
            _require(
                stop <= other_start or other_stop <= start,
                f"path roles overlap: {role} and {other}",
            )
        active.update(range(start, stop))
    if claimed_ids is None:
        claims: Mapping[str, Iterable[int]] = {}
    elif isinstance(claimed_ids, Mapping):
        claims = claimed_ids
    else:
        claims = {"external": claimed_ids}
    collisions: list[dict[str, Any]] = []
    for source, values in claims.items():
        for raw in values:
            _require(
                isinstance(raw, int) and not isinstance(raw, bool),
                "external path claim is not an integer",
            )
            value = int(raw)
            _require(0 <= value < PATH_ID_LIMIT, "external path claim is out of bounds")
            if value in active:
                collisions.append({"source": str(source), "path_id": value})
    _require(not collisions, "schedule path IDs collide with semantic claims")
    return {
        "schema": PATH_PLAN_SCHEMA + "-validation",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "path_id_plan_sha256": expected["semantic_sha256"],
        "path_count": len(active),
        "collision_count": 0,
        "twenty_bit_bounds_pass": 1,
        "role_disjointness_pass": 1,
    }


def _cohort_rows(role: str, sizes: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for index, size in enumerate(sizes):
        start = cursor
        cursor += size
        rows.append(
            {
                "cohort_index": index,
                "size": size,
                "logical_path_start": start,
                "logical_path_stop_exclusive": cursor,
                "evidence_role": role,
                "maximum_phase_launch_lanes": size * 392,
            }
        )
    return rows


def build_schedule_cohort_plan() -> dict[str, Any]:
    """Return frozen future-production cohorts and benchmark profiles."""

    return _hashed_record(
        {
            "schema": COHORT_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "cohort_plan_version": COHORT_PLAN_VERSION,
            "maximum_launch_lanes": 4096,
            "edges_per_matching": 392,
            "restart_outer_steps": 8,
            "production": {
                "train_validation": {
                    "path_count": 96,
                    "group_sizes": list(PRODUCTION_CACHE_GROUP_SIZES),
                    "cohorts": _cohort_rows(
                        "train_validation_cache", PRODUCTION_CACHE_GROUP_SIZES
                    ),
                    "role_slices": {
                        "train": [0, 64],
                        "validation": [64, 96],
                    },
                    "artifact_commit_role_isolation": 1,
                },
                "confirmation": {
                    "path_count": 64,
                    "group_sizes": list(PRODUCTION_STREAM_GROUP_SIZES),
                    "cohorts": _cohort_rows(
                        "confirmation_stream", PRODUCTION_STREAM_GROUP_SIZES
                    ),
                    "role_slices": {"confirmation": [0, 64]},
                    "artifact_commit_role_isolation": 1,
                },
            },
            "pilot_profiles": {
                "cache_p10": {"path_count": 10, "mode": "cache"},
                "cache_p6": {"path_count": 6, "mode": "cache"},
                "stream_p10": {"path_count": 10, "mode": "stream"},
                "stream_p4": {"path_count": 4, "mode": "stream"},
            },
            "cross_role_training_access": 0,
        }
    )


def validate_schedule_cohort_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_schedule_cohort_plan()
    _require(dict(plan) == expected, "schedule cohort plan changed")
    maximum = max(
        row["maximum_phase_launch_lanes"]
        for role in expected["production"].values()
        for row in role["cohorts"]
    )
    _require(maximum <= 4096, "production cohort exceeds the CUDA lane cap")
    return {
        "schema": COHORT_PLAN_SCHEMA + "-validation",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "cohort_plan_sha256": expected["semantic_sha256"],
        "train_validation_path_count": 96,
        "confirmation_path_count": 64,
        "maximum_phase_launch_lanes": maximum,
        "role_isolation_pass": 1,
    }


def build_schedule_timing_plan() -> dict[str, Any]:
    """Return the sealed four-window, three-repeat timing design."""

    return _hashed_record(
        {
            "schema": TIMING_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "timing_plan_version": TIMING_PLAN_VERSION,
            "root_seed": ROOT_SEED,
            "window_start_outer_steps": list(TIMING_WINDOW_STARTS),
            "outer_steps_per_window": TIMING_WINDOW_OUTER_STEPS,
            "branch_outer_steps": list(TIMING_BRANCH_STEPS),
            "time_quartiles": [0, 1, 2, 3],
            "profile_order": list(PROFILE_ORDER),
            "repeat_count": 3,
            "repeat_profile_orders": [
                list(PROFILE_ORDER[offset:] + PROFILE_ORDER[:offset])
                for offset in range(3)
            ],
            "slowest_repeat_authorizes": 1,
            "repeat_averaging_performed": 0,
            "posthoc_timing_allowance": 0,
        }
    )


def validate_schedule_timing_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_schedule_timing_plan()
    _require(dict(plan) == expected, "schedule timing plan changed")
    _require(
        all(
            start <= branch < start + TIMING_WINDOW_OUTER_STEPS
            for start, branch in zip(TIMING_WINDOW_STARTS, TIMING_BRANCH_STEPS)
        ),
        "timing window does not contain its frozen branch step",
    )
    return {
        "schema": TIMING_PLAN_SCHEMA + "-validation",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "timing_plan_sha256": expected["semantic_sha256"],
        "window_count": 4,
        "repeat_count": 3,
        "branch_coverage_pass": 1,
    }


def schedule_source_paths(
    paths: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    values = (Path(__file__),) if paths is None else tuple(Path(value) for value in paths)
    resolved = tuple(sorted({item.resolve() for item in values}, key=lambda p: p.as_posix()))
    _require(bool(resolved), "schedule source set is empty")
    _require(all(item.is_file() for item in resolved), "schedule source is missing")
    return resolved


def schedule_source_fingerprint(
    paths: Iterable[str | Path] | None = None,
) -> str:
    return source_fingerprint(schedule_source_paths(paths))


def verify_schedule_resume_compatibility(
    run_dir: str | Path,
    *,
    source_fingerprint_value: str,
    scientific_config_sha256: str,
    parent_provenance_sha256: str,
    path_plan_sha256: str,
    cohort_plan_sha256: str,
    timing_plan_sha256: str,
) -> dict[str, Any]:
    """Verify every frozen binding before a resumed directory is mutated."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"resume run does not exist: {root}")
    manifest = _load_json(root / "run_manifest.json", "resume manifest")
    config = _load_json(root / "scientific_config.json", "resume config")
    records = {
        "parent_provenance": _load_json(
            root / "parent_provenance.json", "resume parent provenance"
        ),
        "path_plan": _load_json(root / "path_id_plan.json", "resume path plan"),
        "cohort_plan": _load_json(root / "cohort_plan.json", "resume cohort plan"),
        "timing_plan": _load_json(root / "timing_plan.json", "resume timing plan"),
    }
    expected = {
        "source_fingerprint": source_fingerprint_value,
        "scientific_config_sha256": scientific_config_sha256,
        "parent_provenance_sha256": parent_provenance_sha256,
        "path_plan_sha256": path_plan_sha256,
        "cohort_plan_sha256": cohort_plan_sha256,
        "timing_plan_sha256": timing_plan_sha256,
    }
    _require(
        all(manifest.get(name) == value for name, value in expected.items()),
        "resume manifest compatibility changed",
    )
    _require(
        config.get("semantic_sha256") == scientific_config_sha256,
        "resume scientific configuration changed",
    )
    _assert_semantic(config, "resume scientific configuration")
    hashes = {
        "parent_provenance": parent_provenance_sha256,
        "path_plan": path_plan_sha256,
        "cohort_plan": cohort_plan_sha256,
        "timing_plan": timing_plan_sha256,
    }
    for name, record in records.items():
        _require(
            record.get("semantic_sha256") == hashes[name],
            f"resume {name.replace('_', ' ')} changed",
        )
        _assert_semantic(record, f"resume {name.replace('_', ' ')}")
    return {
        "schema": SCHEMA + "-resume-compatibility",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "run_dir": str(root),
        **expected,
    }


__all__ = [
    "BoundaryTangentScheduleProvenanceError",
    "COHORT_PLAN_SCHEMA",
    "COHORT_PLAN_VERSION",
    "FAILED_REGISTRY_COUNT",
    "FAILED_REGISTRY_FILE_SHA256",
    "FAILED_REGISTRY_SEMANTIC_SHA256",
    "FAILED_RUN_BASENAME",
    "FAILED_SCIENTIFIC_CONFIG_SHA256",
    "FAILED_SOURCE_FINGERPRINT",
    "HISTORICAL_DECISION",
    "HISTORICAL_PROJECTED_HOURS",
    "HISTORICAL_RATE",
    "HISTORICAL_TRANSITION_COUNT",
    "PATH_ID_BITS",
    "PATH_ID_LIMIT",
    "PATH_PLAN_SCHEMA",
    "PATH_PLAN_VERSION",
    "PATH_ROLE_RANGES",
    "PRODUCTION_CACHE_GROUP_SIZES",
    "PRODUCTION_STREAM_GROUP_SIZES",
    "PROFILE_ORDER",
    "READJUDICATED_DECISION",
    "ROOT_SEED",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TIMING_BRANCH_STEPS",
    "TIMING_PLAN_SCHEMA",
    "TIMING_PLAN_VERSION",
    "TIMING_WINDOW_OUTER_STEPS",
    "TIMING_WINDOW_STARTS",
    "build_schedule_cohort_plan",
    "build_schedule_path_plan",
    "build_schedule_timing_plan",
    "schedule_source_fingerprint",
    "schedule_source_paths",
    "validate_schedule_cohort_plan",
    "validate_schedule_path_plan",
    "validate_schedule_timing_plan",
    "verify_and_readjudicate_boundary_tangent_schedule_parents",
    "verify_schedule_resume_compatibility",
]
