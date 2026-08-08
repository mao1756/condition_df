"""Immutable provenance for the boundary-tangent Jacobi/RB repair.

The workflow starts from two historical facts:

* the coarse-residual learner completed and passed fresh confirmation; and
* its frozen affine reverse controller failed before opening any scientific
  reverse-law panel because a learned edge transfer left ``[0, 1]``.

This module verifies those runs byte-for-byte, allocates fresh path IDs, and
provides the small compatibility checks needed by a restartable additive
workflow.  It deliberately imports neither a trainer nor a sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-provenance"
SCHEMA_VERSION = 1
PATH_PLAN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-path-id-plan"
PATH_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-path-ids-v1"
PATH_ID_BITS = 20
PATH_ID_LIMIT = 1 << PATH_ID_BITS

SOURCE_IMAGE_SHA256 = (
    "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
)
MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
SELECTED_CHECKPOINT_SHA256 = (
    "24a0893daa31196815463a7396220542003e7dc2557689950ba4dd0eeaa9c914"
)
SELECTED_STATE_DICT_SHA256 = (
    "df479e979cf6dd99580bd918377405b665791a4608f45f6cae326cc10e5e6ad9"
)
FROZEN_COARSE_BASELINE_SHA256 = (
    "5d4e73153c36a59e26403439befd4e13b7f4fe096f7cbf9af6b77ac26565a9df"
)


class BoundaryTangentProvenanceError(ArtifactCompatibilityError):
    """A parent, namespace, source set, or resume binding changed."""


@dataclass(frozen=True)
class ParentSpec:
    role: str
    basename: str
    run_schema: str
    status_schema: str
    config_schema: str
    decision_file: str
    decision_schema: str
    decision: str
    terminal_state: str
    terminal_stage: str
    registry_count: int
    registry_semantic_sha256: str
    registry_file_sha256: str
    source_fingerprint: str
    scientific_config_sha256: str


COARSE_RESIDUAL_PARENT_SPEC = ParentSpec(
    role="successful_coarse_residual",
    basename="20260731-140333_production-exact-k512-coarse-residual-one-image",
    run_schema="experiment12-d0-jacobi-rb-coarse-residual-learnability-manifest",
    status_schema="experiment12-d0-jacobi-rb-coarse-residual-learnability-status",
    config_schema=(
        "experiment12-d0-jacobi-rb-coarse-residual-learnability-"
        "scientific-config"
    ),
    decision_file="coarse_residual_decision.json",
    decision_schema="experiment12-d0-jacobi-rb-coarse-residual-gate-decision",
    decision="exact_rb_coarse_residual_learnable",
    terminal_state="complete",
    terminal_stage="confirm",
    registry_count=3_471,
    registry_semantic_sha256=(
        "308c6452158c1198fd6bb0b7996eeb97a708913baa36b91531fd8e5e3af2c291"
    ),
    registry_file_sha256=(
        "45408753658b575d2bab52e3a1e991d97c79082923b5698c4057e1893e0d930a"
    ),
    source_fingerprint=(
        "42b7129b8850d5e4036137e1781799fe0d9b37d8ee98867d3e1a7b7a57b7906c"
    ),
    scientific_config_sha256=(
        "b49f50be2f414b5c5a7c402850ac40aa2e6d129325116e13033acd3884de3378"
    ),
)

FAILED_CONTROLLER_PARENT_SPEC = ParentSpec(
    role="failed_affine_reverse_controller",
    basename="20260802-040147_production-exact-rb-reverse-controller-control",
    run_schema="experiment12-d0-jacobi-rb-reverse-controller-control-manifest",
    status_schema="experiment12-d0-jacobi-rb-reverse-controller-control-status",
    config_schema=(
        "experiment12-d0-jacobi-rb-reverse-controller-control-scientific-config"
    ),
    decision_file="controller_decision.json",
    decision_schema="experiment12-d0-jacobi-rb-reverse-controller-control-decision",
    decision="controller_boundary_or_conservation_failed",
    terminal_state="gate_failed",
    terminal_stage="preflight",
    registry_count=26,
    registry_semantic_sha256=(
        "2b1c7dc65fa715a6996571bf2cadf8cdbada15eedfa5ef1b8ffa5b5d9c18be8b"
    ),
    registry_file_sha256=(
        "f58c3380a5bc54fb21d249cd0604917dbd18fe19092bbdb15e4266013c0ebfb5"
    ),
    source_fingerprint=(
        "54cfa6896de2ce7da3cd4190a01d113a04aee328dad0bc62e7d6a8f1aaa3a215"
    ),
    scientific_config_sha256=(
        "85f2773601f2759e6bf6b4405f0508462ae2fabaf2b7bf80b383d54b7efc907c"
    ),
)

EXPECTED_COARSE_RESIDUAL_RUN_BASENAME = COARSE_RESIDUAL_PARENT_SPEC.basename
EXPECTED_COARSE_RESIDUAL_REGISTRY_COUNT = COARSE_RESIDUAL_PARENT_SPEC.registry_count
EXPECTED_COARSE_RESIDUAL_REGISTRY_SEMANTIC_SHA256 = (
    COARSE_RESIDUAL_PARENT_SPEC.registry_semantic_sha256
)
EXPECTED_COARSE_RESIDUAL_REGISTRY_FILE_SHA256 = (
    COARSE_RESIDUAL_PARENT_SPEC.registry_file_sha256
)
EXPECTED_COARSE_RESIDUAL_SOURCE_FINGERPRINT = (
    COARSE_RESIDUAL_PARENT_SPEC.source_fingerprint
)
EXPECTED_COARSE_RESIDUAL_SCIENTIFIC_CONFIG_SHA256 = (
    COARSE_RESIDUAL_PARENT_SPEC.scientific_config_sha256
)
EXPECTED_FAILED_CONTROLLER_RUN_BASENAME = FAILED_CONTROLLER_PARENT_SPEC.basename
EXPECTED_FAILED_CONTROLLER_REGISTRY_COUNT = FAILED_CONTROLLER_PARENT_SPEC.registry_count
EXPECTED_FAILED_CONTROLLER_REGISTRY_SEMANTIC_SHA256 = (
    FAILED_CONTROLLER_PARENT_SPEC.registry_semantic_sha256
)
EXPECTED_FAILED_CONTROLLER_REGISTRY_FILE_SHA256 = (
    FAILED_CONTROLLER_PARENT_SPEC.registry_file_sha256
)
EXPECTED_FAILED_CONTROLLER_SOURCE_FINGERPRINT = (
    FAILED_CONTROLLER_PARENT_SPEC.source_fingerprint
)
EXPECTED_FAILED_CONTROLLER_SCIENTIFIC_CONFIG_SHA256 = (
    FAILED_CONTROLLER_PARENT_SPEC.scientific_config_sha256
)

PARENT_SPECS = {
    COARSE_RESIDUAL_PARENT_SPEC.role: COARSE_RESIDUAL_PARENT_SPEC,
    FAILED_CONTROLLER_PARENT_SPEC.role: FAILED_CONTROLLER_PARENT_SPEC,
}

# Half-open slots.  The first four are fresh evidence.  The final block is a
# reservation only and is intentionally not expanded into a 65,536-entry JSON
# array.
PATH_ROLE_RANGES: dict[str, tuple[int, int]] = {
    "preflight_benchmark": (0xEC000, 0xEC008),
    "train": (0xEC100, 0xEC140),
    "validation": (0xEC200, 0xEC220),
    "confirmation": (0xED000, 0xED040),
    "future_production_reserved": (0xF0000, 0x100000),
}
ACTIVE_PATH_ROLES = (
    "preflight_benchmark",
    "train",
    "validation",
    "confirmation",
)

NO_WORK_FIELDS = (
    "sampling_performed",
    "reverse_sampling_performed",
    "reconstruction_performed",
    "image_sampling_performed",
    "full_reverse_path_performed",
)
NO_AUTHORIZATION_FIELDS = (
    "sampling_authorized",
    "reverse_sampling_authorized",
    "reconstruction_authorized",
    "reconstruction_claim_authorized",
    "known_prior_claim_authorized",
    "full_dataset_training_authorized",
    "unsplit_generator_claim_authorized",
    "spatial_dirichlet_ferguson_claim_authorized",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryTangentProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundaryTangentProvenanceError(
            f"invalid {description}: {path}"
        ) from exc
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


def _semantic_body(record: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(record)
    body.pop("semantic_sha256", None)
    return body


def _assert_semantic_hash(record: Mapping[str, Any], description: str) -> None:
    _require(
        record.get("semantic_sha256") == config_fingerprint(_semantic_body(record)),
        f"{description} semantic hash changed",
    )


def _assert_no_sampling(record: Mapping[str, Any], description: str) -> None:
    for field in NO_WORK_FIELDS:
        _require(
            int(record.get(field, 0)) == 0,
            f"{description} unexpectedly records {field}",
        )
    for field in NO_AUTHORIZATION_FIELDS:
        if field in record:
            _require(
                int(record.get(field, 0)) == 0,
                f"{description} unexpectedly authorizes {field}",
            )


def _safe_registry_path(root: Path, value: Any) -> tuple[str, Path]:
    _require(isinstance(value, str) and value, "registry path is invalid")
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
        raise BoundaryTangentProvenanceError(
            f"registry path escapes parent: {value!r}"
        ) from exc
    return relative.as_posix(), target


def _verify_registry(
    root: Path, spec: ParentSpec
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = root / "artifact_registry.json"
    _require(
        file_fingerprint(path) == spec.registry_file_sha256,
        f"{spec.role} registry file hash changed",
    )
    registry = _load_json(path, f"{spec.role} registry")
    artifacts = registry.get("artifacts")
    _require(
        registry.get("schema") == spec.run_schema.removesuffix("-manifest")
        + "-artifact-registry"
        and registry.get("schema_version") == 1
        and isinstance(artifacts, list)
        and int(registry.get("artifact_count", -1)) == spec.registry_count
        and len(artifacts) == spec.registry_count
        and registry.get("semantic_sha256") == spec.registry_semantic_sha256
        and config_fingerprint({"artifacts": artifacts})
        == spec.registry_semantic_sha256,
        f"{spec.role} terminal registry binding changed",
    )
    by_path: dict[str, dict[str, Any]] = {}
    for raw in artifacts:
        _require(isinstance(raw, Mapping), f"{spec.role} registry row is malformed")
        record = dict(raw)
        relative, target = _safe_registry_path(root, record.get("path"))
        _require(relative not in by_path, f"{spec.role} registry path is duplicated")
        _require(
            target.is_file()
            and int(record.get("size", -1)) == target.stat().st_size
            and record.get("sha256") == file_fingerprint(target),
            f"{spec.role} artifact changed: {relative}",
        )
        by_path[relative] = record
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.name not in {"artifact_registry.json", "run_status.json"}
    }
    _require(actual == set(by_path), f"{spec.role} terminal file set changed")
    _assert_no_sampling(registry, f"{spec.role} registry")
    return registry, by_path


def _resolve_source_paths(manifest: Mapping[str, Any], role: str) -> tuple[Path, ...]:
    raw = manifest.get("source_paths")
    _require(
        isinstance(raw, list)
        and raw
        and all(isinstance(item, str) and item for item in raw),
        f"{role} source path list changed",
    )
    repository_root = Path(__file__).resolve().parents[1]
    resolved = tuple(
        sorted(
            {
                (
                    Path(item).resolve()
                    if Path(item).is_absolute()
                    else (repository_root / Path(item)).resolve()
                )
                for item in raw
            },
            key=lambda item: item.as_posix(),
        )
    )
    _require(
        len(resolved) == len(raw) and all(item.is_file() for item in resolved),
        f"{role} live source set changed",
    )
    return resolved


def _verify_common_parent(root_value: str | Path, spec: ParentSpec) -> dict[str, Any]:
    root = Path(root_value).resolve()
    _require(root.is_dir(), f"{spec.role} run does not exist: {root}")
    _require(root.name == spec.basename, f"wrong {spec.role} run basename")
    registry, registered = _verify_registry(root, spec)
    manifest = _load_json(root / "run_manifest.json", f"{spec.role} manifest")
    status = _load_json(root / "run_status.json", f"{spec.role} status")
    config = _load_json(root / "scientific_config.json", f"{spec.role} config")
    decision = _load_json(root / spec.decision_file, f"{spec.role} decision")
    _require(
        manifest.get("schema") == spec.run_schema
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == spec.source_fingerprint
        and manifest.get("scientific_config_sha256")
        == spec.scientific_config_sha256,
        f"{spec.role} manifest binding changed",
    )
    live_sources = _resolve_source_paths(manifest, spec.role)
    _require(
        source_fingerprint(live_sources) == spec.source_fingerprint,
        f"{spec.role} live source fingerprint changed",
    )
    _require(
        config.get("schema") == spec.config_schema
        and config.get("schema_version") == 1
        and config.get("semantic_sha256") == spec.scientific_config_sha256,
        f"{spec.role} scientific configuration changed",
    )
    _assert_semantic_hash(config, f"{spec.role} scientific configuration")
    _require(
        status.get("schema") == spec.status_schema
        and status.get("schema_version") == 1
        and status.get("state") == spec.terminal_state
        and status.get("stage") == spec.terminal_stage
        and status.get("decision") == spec.decision,
        f"{spec.role} terminal status changed",
    )
    _require(
        decision.get("schema") == spec.decision_schema
        and decision.get("schema_version") == 1
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == spec.decision,
        f"{spec.role} decision changed",
    )
    for description, record in (
        ("manifest", manifest),
        ("status", status),
        ("config", config),
        ("decision", decision),
    ):
        _assert_no_sampling(record, f"{spec.role} {description}")
    return {
        "role": spec.role,
        "run_dir": str(root),
        "basename": root.name,
        "registry": {
            "artifact_count": int(registry["artifact_count"]),
            "semantic_sha256": spec.registry_semantic_sha256,
            "file_sha256": spec.registry_file_sha256,
        },
        "source_fingerprint": spec.source_fingerprint,
        "scientific_config_sha256": spec.scientific_config_sha256,
        "terminal": {
            "state": spec.terminal_state,
            "stage": spec.terminal_stage,
            "decision": spec.decision,
        },
        "registered_paths": registered,
        "manifest": manifest,
        "status_record": status,
        "decision_record": decision,
        "verified": 1,
    }


def _passed_gate(root: Path, name: str, schema: str) -> dict[str, Any]:
    gate = _load_json(root / name, name)
    _require(
        gate.get("schema") == schema
        and gate.get("schema_version") == 1
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1,
        f"required parent gate changed: {name}",
    )
    _assert_no_sampling(gate, name)
    return gate


def _verify_coarse_specific(binding: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(binding["run_dir"]))
    schema = "experiment12-d0-jacobi-rb-coarse-residual-gate"
    for name in (
        "preflight_gate.json",
        "cache_gate.json",
        "train_gate.json",
        "confirmation_cache_gate.json",
        "confirmation_gate.json",
    ):
        _passed_gate(root, name, schema)
    _passed_gate(root, "workflow_gate.json", schema + "-workflow")
    status = binding["status_record"]
    decision = binding["decision_record"]
    selected = _load_json(root / "selected_model.json", "selected coarse model")
    confirmation_open = _load_json(
        root / "confirmation_open.json", "coarse confirmation-open record"
    )
    _require(
        int(status.get("physical_training_performed", 0)) == 1
        and int(decision.get("reverse_controller_planning_authorized", 0)) == 1
        and int(selected.get("nonzero_residual_selected", 0)) == 1
        and int(selected.get("confirmation_inspected", -1)) == 0
        and selected.get("selected_model_sha256") == SELECTED_CHECKPOINT_SHA256
        and dict(selected.get("candidate", {})).get("state_sha256")
        == SELECTED_STATE_DICT_SHA256
        and file_fingerprint(root / "selected_model.pt")
        == SELECTED_CHECKPOINT_SHA256
        and int(confirmation_open.get("open_count", -1)) == 1,
        "successful coarse-residual checkpoint/confirmation binding changed",
    )
    return {
        "physical_training_performed": 1,
        "confirmation_performed": 1,
        "selected_checkpoint_sha256": SELECTED_CHECKPOINT_SHA256,
        "selected_state_dict_sha256": SELECTED_STATE_DICT_SHA256,
        "confirmation_opened_count": 1,
        "all_required_gates_passed": 1,
    }


def _not_evaluated_gate(root: Path, name: str, expected_gate: str) -> None:
    gate = _load_json(root / name, name)
    _require(
        gate.get("evaluation_status") == "not_evaluated"
        and int(gate.get("passed", 1)) == 0
        and gate.get("gate") == expected_gate
        and gate.get("reason") == "skipped_after_failed_preflight_gate",
        f"failed controller unexpectedly evaluated {name}",
    )
    _assert_no_sampling(gate, name)


def _verify_failed_controller_specific(
    binding: Mapping[str, Any], coarse: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(str(binding["run_dir"]))
    status = binding["status_record"]
    decision = binding["decision_record"]
    preflight = _load_json(root / "preflight_gate.json", "controller preflight gate")
    rejection = preflight.get("boundary_rejection")
    _require(
        preflight.get("schema")
        == "experiment12-d0-jacobi-rb-reverse-controller-control-preflight-gate"
        and preflight.get("evaluation_status") == "evaluated"
        and int(preflight.get("passed", 1)) == 0
        and isinstance(rejection, Mapping)
        and rejection.get("failure_code") == "controller_boundary_step_rejected"
        and status.get("failure_domain") == "controller_boundary"
        and status.get("failure_code") == "controller_boundary_step_rejected"
        and int(status.get("controller_control_trajectory_performed", -1)) == 0
        and int(status.get("maximum_control_trajectory_phase_count", -1)) == 0
        and int(decision.get("controller_control_trajectory_performed", -1)) == 0
        and int(decision.get("maximum_control_trajectory_phase_count", -1)) == 0,
        "failed controller boundary scope changed",
    )
    for name, gate in (
        ("cache_gate.json", "cache"),
        ("oracle_gate.json", "oracle"),
        ("control_gate.json", "control"),
    ):
        _not_evaluated_gate(root, name, gate)
    decide_gate = _load_json(root / "decide_gate.json", "controller decide gate")
    _require(
        decide_gate.get("evaluation_status") == "evaluated"
        and decide_gate.get("gate") == "decide"
        and int(decide_gate.get("passed", 1)) == 0
        and decide_gate.get("decision") == FAILED_CONTROLLER_PARENT_SPEC.decision,
        "failed controller terminal decide gate changed",
    )
    _assert_no_sampling(decide_gate, "controller decide gate")
    registered = set(binding["registered_paths"])
    _require(
        not any(
            "confirmation" in path.lower()
            or path.startswith("forward/")
            or path.startswith("control/")
            for path in registered
        ),
        "failed controller opened a forbidden downstream evidence stage",
    )
    parent = _load_json(root / "parent_provenance.json", "controller parent binding")
    selected = _load_json(
        root / "selected_checkpoint_binding.json", "controller checkpoint binding"
    )
    baseline = _load_json(
        root / "frozen_baseline_binding.json", "controller baseline binding"
    )
    _require(
        int(parent.get("passed", 0)) == 1
        and Path(str(parent.get("parent_run_dir", ""))).resolve()
        == Path(str(coarse["run_dir"])).resolve()
        and int(parent.get("registry_count", -1))
        == int(coarse["registry"]["artifact_count"])
        and parent.get("registry_file_sha256")
        == coarse["registry"]["file_sha256"]
        and parent.get("registry_semantic_sha256")
        == coarse["registry"]["semantic_sha256"]
        and parent.get("source_fingerprint") == coarse["source_fingerprint"]
        and parent.get("scientific_config_sha256")
        == coarse["scientific_config_sha256"]
        and selected.get("checkpoint_sha256") == SELECTED_CHECKPOINT_SHA256
        and selected.get("state_dict_sha256") == SELECTED_STATE_DICT_SHA256
        and int(selected.get("seed", -1)) == 261254
        and int(selected.get("update", -1)) == 3000
        and baseline.get("values_c_order_sha256")
        == FROZEN_COARSE_BASELINE_SHA256
        and int(baseline.get("refit_performed", -1)) == 0,
        "failed controller transitive coarse-parent binding changed",
    )
    return {
        "preflight_evaluation_status": "evaluated",
        "preflight_passed": 0,
        "failure_domain": "controller_boundary",
        "failure_code": "controller_boundary_step_rejected",
        "controller_control_trajectory_performed": 0,
        "maximum_control_trajectory_phase_count": 0,
        "downstream_scientific_stages_opened": 0,
        "transitive_coarse_parent_binding_passed": 1,
    }


def verify_boundary_tangent_parents(
    *,
    coarse_residual_run_dir: str | Path | None = None,
    failed_controller_run_dir: str | Path | None = None,
    parent_coarse_residual_run_dir: str | Path | None = None,
    parent_failed_controller_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the successful learner and failed affine-controller evidence."""

    coarse_value = (
        coarse_residual_run_dir
        if coarse_residual_run_dir is not None
        else parent_coarse_residual_run_dir
    )
    failed_value = (
        failed_controller_run_dir
        if failed_controller_run_dir is not None
        else parent_failed_controller_run_dir
    )
    _require(coarse_value is not None, "successful coarse-residual parent is missing")
    _require(failed_value is not None, "failed controller parent is missing")
    coarse = _verify_common_parent(
        coarse_value, COARSE_RESIDUAL_PARENT_SPEC
    )
    failed = _verify_common_parent(
        failed_value, FAILED_CONTROLLER_PARENT_SPEC
    )
    coarse_specific = _verify_coarse_specific(coarse)
    failed_specific = _verify_failed_controller_specific(failed, coarse)
    for record in (coarse, failed):
        record.pop("registered_paths", None)
        record.pop("manifest", None)
        record.pop("status_record", None)
        record.pop("decision_record", None)
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parents": {
            "successful_coarse_residual": coarse,
            "failed_affine_reverse_controller": failed,
        },
        "successful_parent_evidence": coarse_specific,
        "failed_controller_evidence": failed_specific,
        "same_selected_checkpoint_pass": 1,
        "transitive_parent_binding_pass": 1,
        "all_artifact_hashes_pass": 1,
        "parents_immutable_pass": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "reconstruction_performed": 0,
    }
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def build_boundary_tangent_path_plan() -> dict[str, Any]:
    """Return the frozen fresh-evidence namespace and its semantic hash."""

    roles = {
        role: list(range(*PATH_ROLE_RANGES[role])) for role in ACTIVE_PATH_ROLES
    }
    record = {
        "schema": PATH_PLAN_SCHEMA,
        "schema_version": 1,
        "path_id_plan_version": PATH_PLAN_VERSION,
        "canonical_path_id_bits": PATH_ID_BITS,
        "roles": roles,
        "role_slots": {
            role: {
                "start": bounds[0],
                "stop_exclusive": bounds[1],
                "start_hex": f"0x{bounds[0]:05X}",
                "stop_exclusive_hex": f"0x{bounds[1]:05X}",
            }
            for role, bounds in PATH_ROLE_RANGES.items()
        },
        "reserved_roles": {
            "future_production": {
                "start": PATH_ROLE_RANGES["future_production_reserved"][0],
                "stop_exclusive": PATH_ROLE_RANGES["future_production_reserved"][1],
            }
        },
        "fresh_path_count": sum(len(values) for values in roles.values()),
        "collision_free": 1,
        "silent_remapping_performed": 0,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _claimed_id_map(
    claimed_ids: Mapping[str, Iterable[int]] | Iterable[int] | None,
) -> dict[str, tuple[int, ...]]:
    if claimed_ids is None:
        return {}
    if isinstance(claimed_ids, Mapping):
        return {
            str(name): tuple(int(value) for value in values)
            for name, values in claimed_ids.items()
        }
    return {"external": tuple(int(value) for value in claimed_ids)}


def validate_boundary_tangent_path_plan(
    plan: Mapping[str, Any],
    *,
    claimed_ids: Mapping[str, Iterable[int]] | Iterable[int] | None = None,
) -> dict[str, Any]:
    """Validate the exact plan and optional external semantic claims."""

    expected = build_boundary_tangent_path_plan()
    _require(dict(plan) == expected, "boundary-tangent path plan changed")
    intervals = list(PATH_ROLE_RANGES.items())
    for index, (role, (start, stop)) in enumerate(intervals):
        _require(
            isinstance(start, int)
            and isinstance(stop, int)
            and 0 <= start < stop <= PATH_ID_LIMIT,
            f"{role} path slot lies outside the 20-bit namespace",
        )
        for other, (other_start, other_stop) in intervals[index + 1 :]:
            _require(
                stop <= other_start or other_stop <= start,
                f"path slots overlap: {role} and {other}",
            )
    active = {
        value
        for role in ACTIVE_PATH_ROLES
        for value in range(*PATH_ROLE_RANGES[role])
    }
    reserved = set(range(*PATH_ROLE_RANGES["future_production_reserved"]))
    collisions: list[dict[str, Any]] = []
    for source, values in _claimed_id_map(claimed_ids).items():
        for value in values:
            _require(0 <= value < PATH_ID_LIMIT, "external path claim is out of bounds")
            if value in active or value in reserved:
                collisions.append({"source": source, "path_id": value})
    _require(not collisions, "boundary-tangent path IDs collide with semantic claims")
    return {
        "schema": PATH_PLAN_SCHEMA + "-validation",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "path_id_plan_version": PATH_PLAN_VERSION,
        "path_id_plan_sha256": expected["semantic_sha256"],
        "active_path_count": len(active),
        "reserved_path_count": len(reserved),
        "collision_count": 0,
        "twenty_bit_bounds_pass": 1,
        "role_disjointness_pass": 1,
    }


def build_failed_controller_readjudication(
    parent_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the immutable failure without changing its historical record."""

    _require(
        parent_binding.get("schema") == SCHEMA
        and int(parent_binding.get("passed", 0)) == 1,
        "verified boundary-tangent parent binding is required",
    )
    failed = dict(parent_binding.get("failed_controller_evidence", {}))
    _require(
        failed.get("failure_code") == "controller_boundary_step_rejected"
        and int(failed.get("downstream_scientific_stages_opened", -1)) == 0,
        "failed controller evidence is not the frozen boundary rejection",
    )
    record = {
        "schema": SCHEMA + "-failed-controller-readjudication",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "historical_decision_preserved": "controller_boundary_or_conservation_failed",
        "corrected_adjudication": "frozen_affine_controller_boundary_domain_invalid",
        "failure_domain": "controller_boundary",
        "failure_code": "controller_boundary_step_rejected",
        "scientific_evidence_complete": 0,
        "exact_jacobi_reference_invalidated": 0,
        "exact_rao_blackwell_target_invalidated": 0,
        "controller_control_trajectory_performed": 0,
        "downstream_scientific_stages_opened": 0,
        "parent_artifacts_mutated": 0,
        "recommended_next_action": (
            "retrain a boundary-tangent predictor against the unchanged exact "
            "Rao-Blackwell label before another controller gate"
        ),
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "reconstruction_performed": 0,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _normalize_source_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    resolved = tuple(
        sorted({Path(value).resolve() for value in paths}, key=lambda p: p.as_posix())
    )
    _require(bool(resolved), "boundary-tangent source set is empty")
    _require(all(path.is_file() for path in resolved), "boundary-tangent source is missing")
    return resolved


def boundary_tangent_source_paths(
    paths: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Normalize a workflow source set; default to this additive module."""

    return _normalize_source_paths((Path(__file__),) if paths is None else paths)


def boundary_tangent_source_fingerprint(
    paths: Iterable[str | Path] | None = None,
) -> str:
    """Hash a normalized boundary-tangent workflow source set."""

    return source_fingerprint(boundary_tangent_source_paths(paths))


def verify_resume_compatibility(
    run_dir: str | Path,
    *,
    scientific_config_sha256: str,
    path_plan_sha256: str,
    source_fingerprint_value: str | None = None,
    source_fingerprint: str | None = None,
    parent_provenance_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail before mutation when a resumed run has a different frozen binding."""

    expected_source = (
        source_fingerprint_value
        if source_fingerprint_value is not None
        else source_fingerprint
    )
    _require(bool(expected_source), "expected source fingerprint is missing")
    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"resume run does not exist: {root}")
    manifest = _load_json(root / "run_manifest.json", "resume manifest")
    config = _load_json(root / "scientific_config.json", "resume config")
    plan = _load_json(root / "path_id_plan.json", "resume path plan")
    _require(
        manifest.get("source_fingerprint") == expected_source
        and manifest.get("scientific_config_sha256")
        == scientific_config_sha256
        and manifest.get("path_plan_sha256") == path_plan_sha256,
        "resume manifest compatibility changed",
    )
    _require(
        config.get("semantic_sha256") == scientific_config_sha256,
        "resume scientific configuration changed",
    )
    _assert_semantic_hash(config, "resume scientific configuration")
    _require(
        plan.get("semantic_sha256") == path_plan_sha256,
        "resume path-ID plan changed",
    )
    _assert_semantic_hash(plan, "resume path-ID plan")
    if parent_provenance_sha256 is not None:
        provenance = _load_json(root / "parent_provenance.json", "resume provenance")
        _require(
            provenance.get("semantic_sha256") == parent_provenance_sha256,
            "resume parent provenance changed",
        )
        _assert_semantic_hash(provenance, "resume parent provenance")
    return {
        "schema": SCHEMA + "-resume-compatibility",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "run_dir": str(root),
        "source_fingerprint": expected_source,
        "scientific_config_sha256": scientific_config_sha256,
        "path_plan_sha256": path_plan_sha256,
        "parent_provenance_sha256": parent_provenance_sha256,
    }


__all__ = [
    "ACTIVE_PATH_ROLES",
    "BoundaryTangentProvenanceError",
    "COARSE_RESIDUAL_PARENT_SPEC",
    "EXPECTED_COARSE_RESIDUAL_REGISTRY_COUNT",
    "EXPECTED_COARSE_RESIDUAL_REGISTRY_FILE_SHA256",
    "EXPECTED_COARSE_RESIDUAL_REGISTRY_SEMANTIC_SHA256",
    "EXPECTED_COARSE_RESIDUAL_RUN_BASENAME",
    "EXPECTED_COARSE_RESIDUAL_SCIENTIFIC_CONFIG_SHA256",
    "EXPECTED_COARSE_RESIDUAL_SOURCE_FINGERPRINT",
    "EXPECTED_FAILED_CONTROLLER_REGISTRY_COUNT",
    "EXPECTED_FAILED_CONTROLLER_REGISTRY_FILE_SHA256",
    "EXPECTED_FAILED_CONTROLLER_REGISTRY_SEMANTIC_SHA256",
    "EXPECTED_FAILED_CONTROLLER_RUN_BASENAME",
    "EXPECTED_FAILED_CONTROLLER_SCIENTIFIC_CONFIG_SHA256",
    "EXPECTED_FAILED_CONTROLLER_SOURCE_FINGERPRINT",
    "FAILED_CONTROLLER_PARENT_SPEC",
    "FROZEN_COARSE_BASELINE_SHA256",
    "MIXED_TARGET_SHA256",
    "PARENT_SPECS",
    "PATH_ID_BITS",
    "PATH_ID_LIMIT",
    "PATH_PLAN_SCHEMA",
    "PATH_PLAN_VERSION",
    "PATH_ROLE_RANGES",
    "ParentSpec",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SELECTED_CHECKPOINT_SHA256",
    "SELECTED_STATE_DICT_SHA256",
    "SOURCE_IMAGE_SHA256",
    "boundary_tangent_source_fingerprint",
    "boundary_tangent_source_paths",
    "build_boundary_tangent_path_plan",
    "build_failed_controller_readjudication",
    "validate_boundary_tangent_path_plan",
    "verify_boundary_tangent_parents",
    "verify_resume_compatibility",
]
