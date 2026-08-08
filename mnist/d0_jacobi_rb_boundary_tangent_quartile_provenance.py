"""Immutable provenance and namespace plans for quartile specialists.

The quartile-specialist workflow is additive.  Historical validation evidence
may justify its design, but must never choose a production checkpoint, gain,
or audit result.  This module therefore has two deliberately separate jobs:

* verify the terminal time-local adjudication and all three parents it binds;
* freeze fresh path, cohort, seed, source, and role-opening contracts.

All helpers are read-only with respect to parent run directories.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import json

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    v3_transitive_source_paths,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local_provenance import (
    BAYES_PARENT_BASENAME,
    MEMORY_PARENT_BASENAME,
    WITNESS_PARENT_BASENAME,
    snapshot_parent_run,
    verify_parent_immutability_snapshot,
    verify_time_local_adjudication_parents,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-quartile-provenance"
SCHEMA_VERSION = 1

TIME_LOCAL_PARENT_BASENAME = (
    "20260807-005609_production-v3-time-local-adjudication"
)
TIME_LOCAL_PARENT_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-v3-time-local-adjudication"
)
TIME_LOCAL_PARENT_DECISION = "exact_rb_high_reverse_time_only_signal"
TIME_LOCAL_PARENT_REGISTRY_COUNT = 29
TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "b25256d606f1fea2c9ef78ab5f14a7b8ccd67bc6f5c234bd2ed2a1a0086fd9f5"
)
TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256 = (
    "15220d3f4ee3e7a4740fd5fae2695e1da1d0b1ea91ee05c70357c3a152569a64"
)
TIME_LOCAL_PARENT_SOURCE_FINGERPRINT = (
    "55f259f30ecb1eb47915a44d3ba67a353bac87abd87743f917c55bbcb06a0123"
)
TIME_LOCAL_PARENT_CONFIG_SHA256 = (
    "faf395317449a842e63de0807d39102f68d7afa49c7700e9cd6c94e0d381b009"
)

_TIME_LOCAL_TERMINAL_HASHES = {
    "artifact_registry.json": TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256,
    "run_status.json": (
        "324b9ab39fbb71c258f45f009a63c1bf95d516054f8bd4451926858121e37a41"
    ),
    "time_local_adjudication_decision.json": (
        "29de314f8fd315d5d38ca1272c9a2779128fb794aee0789719019d7ddaef7e72"
    ),
    "workflow_gate.json": (
        "e5f19e309dbe5d07490c4f7f01fe8bab29fe9b22b00d74fd7cf927f581a65bf2"
    ),
}

PATH_ID_BITS = 20
PATH_ID_LIMIT = 1 << PATH_ID_BITS
PRODUCTION_RESERVATION = (0xF0000, 0x100000)
PATH_PLAN_SCHEMA = f"{SCHEMA}-path-id-plan"
PATH_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-quartile-path-ids-v1"
PATH_ROLE_RANGES: dict[str, tuple[int, int]] = {
    "preflight_seam": (0xF3000, 0xF3008),
    "physical_fit": (0xF4000, 0xF4040),
    "gain_calibration": (0xF4100, 0xF4120),
    "training_rank": (0xF4200, 0xF4220),
    "fresh_selection": (0xF5000, 0xF5180),
    "untouched_confirmation": (0xF7000, 0xF7180),
}

# These historical allocations are not available to the new workflow even
# when the corresponding scientific role was never opened.
HISTORICAL_PATH_RANGES: dict[str, tuple[int, int]] = {
    "v3_preflight_seam": (0xF0000, 0xF0008),
    "v3_train": (0xF1000, 0xF1040),
    "v3_validation": (0xF1100, 0xF1120),
    "v3_unopened_confirmation": (0xF2000, 0xF2040),
}

COHORT_PLAN_SCHEMA = f"{SCHEMA}-cohort-plan"
COHORT_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-quartile-cohorts-v1"
ROLE_COHORT_SIZES: dict[str, tuple[int, ...]] = {
    "preflight_seam": (8,),
    "physical_fit": (10,) * 6 + (4,),
    "gain_calibration": (10,) * 3 + (2,),
    "training_rank": (10,) * 3 + (2,),
    "fresh_selection": (10,) * 38 + (4,),
    "untouched_confirmation": (10,) * 38 + (4,),
}

ROOT_SEED = 261_331
PHYSICAL_MODEL_SEEDS: dict[int, tuple[int, int, int]] = {
    0: (261_332, 261_333, 261_334),
    1: (261_335, 261_336, 261_337),
    2: (261_338, 261_339, 261_340),
    3: (261_341, 261_342, 261_343),
}
SELECTION_BOOTSTRAP_SEED = 261_350
SELECTION_BOOTSTRAP_NAMESPACE = 0x51545331
CONFIRMATION_BOOTSTRAP_SEED = 261_351
CONFIRMATION_BOOTSTRAP_NAMESPACE = 0x51544331
SYNTHETIC_CONTROL_SEEDS = (261_352, 261_353, 261_354, 261_355)
EXACT_NULL_CONTROL_ROOT_SEED = 261_356
RESERVED_FUTURE_CONTROL_SEED = 261_357

ROLE_OPEN_ORDER = (
    "physical_fit",
    "gain_calibration",
    "training_rank",
    "fresh_selection",
    "untouched_confirmation",
)


class QuartileSpecialistProvenanceError(ArtifactCompatibilityError):
    """An immutable parent or frozen production plan changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuartileSpecialistProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuartileSpecialistProvenanceError(
            f"invalid {description}: {path}"
        ) from exc
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


def _semantic_body(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.pop("semantic_sha256", None)
    return result


def _assert_semantic(record: Mapping[str, Any], description: str) -> None:
    _require(
        record.get("semantic_sha256") == config_fingerprint(_semantic_body(record)),
        f"{description} semantic hash changed",
    )


def _hashed(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _safe_relative(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "artifact path is invalid")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe artifact path: {value!r}",
    )
    return relative.as_posix()


def _verify_time_local_registry(root: Path) -> dict[str, Any]:
    _require(
        root.is_dir() and root.name == TIME_LOCAL_PARENT_BASENAME,
        "wrong authoritative time-local parent basename",
    )
    registry_path = root / "artifact_registry.json"
    _require(registry_path.is_file(), "missing time-local artifact registry")
    _require(
        file_fingerprint(registry_path) == TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256,
        "time-local registry file hash changed",
    )
    registry = _load_json(registry_path, "time-local artifact registry")
    artifacts = registry.get("artifacts")
    _require(
        registry.get("schema") == f"{TIME_LOCAL_PARENT_SCHEMA}-artifact-registry"
        and registry.get("schema_version") == 1
        and registry.get("artifact_count") == TIME_LOCAL_PARENT_REGISTRY_COUNT
        and isinstance(artifacts, list)
        and len(artifacts) == TIME_LOCAL_PARENT_REGISTRY_COUNT
        and registry.get("semantic_sha256")
        == TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256,
        "time-local registry binding changed",
    )
    _assert_semantic(registry, "time-local artifact registry")
    registered: set[str] = set()
    for item in artifacts:
        _require(isinstance(item, Mapping), "time-local registry row is malformed")
        relative = _safe_relative(item.get("path"))
        _require(relative not in registered, "time-local registry path is duplicated")
        path = root / relative
        _require(
            path.is_file()
            and path.stat().st_size == item.get("size")
            and file_fingerprint(path) == item.get("sha256"),
            f"time-local registered artifact changed: {relative}",
        )
        registered.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    _require(
        observed == registered | set(_TIME_LOCAL_TERMINAL_HASHES),
        "time-local terminal file set changed",
    )
    for relative, expected in _TIME_LOCAL_TERMINAL_HASHES.items():
        _require(
            file_fingerprint(root / relative) == expected,
            f"time-local terminal artifact changed: {relative}",
        )
    return registry


def _zero_scope(record: Mapping[str, Any], description: str) -> None:
    for field in (
        "confirmation_evidence_accessed",
        "confirmation_performed",
        "controller_control_trajectory_performed",
        "physical_training_performed",
        "reconstruction_performed",
        "reverse_sampling_performed",
        "sampling_performed",
    ):
        _require(int(record.get(field, 0)) == 0, f"{description} records {field}")


def _same_resolved_path(value: Any, expected: Path, description: str) -> None:
    _require(
        isinstance(value, str) and Path(value).resolve() == expected,
        f"{description} path binding changed",
    )


def verify_quartile_specialist_parents(
    *,
    time_local_run_dir: str | Path,
    memory_v3_run_dir: str | Path,
    coarse_witness_run_dir: str | Path,
    bayes_power_run_dir: str | Path,
    verify_external_cache: bool = True,
) -> dict[str, Any]:
    """Verify the authoritative child and its three transitive parents."""

    time_local = Path(time_local_run_dir).resolve()
    memory = Path(memory_v3_run_dir).resolve()
    witness = Path(coarse_witness_run_dir).resolve()
    bayes = Path(bayes_power_run_dir).resolve()
    _require(memory.name == MEMORY_PARENT_BASENAME, "wrong memory-v3 parent basename")
    _require(witness.name == WITNESS_PARENT_BASENAME, "wrong witness parent basename")
    _require(bayes.name == BAYES_PARENT_BASENAME, "wrong Bayes parent basename")

    before = snapshot_parent_run(time_local)
    registry = _verify_time_local_registry(time_local)
    manifest = _load_json(time_local / "run_manifest.json", "time-local manifest")
    config = _load_json(time_local / "scientific_config.json", "time-local config")
    status = _load_json(time_local / "run_status.json", "time-local status")
    decision = _load_json(
        time_local / "time_local_adjudication_decision.json",
        "time-local decision",
    )
    parent_record = _load_json(
        time_local / "parent_provenance.json", "time-local parent provenance"
    )
    _assert_semantic(config, "time-local scientific config")
    _assert_semantic(parent_record, "time-local parent provenance")
    _require(
        manifest.get("schema") == f"{TIME_LOCAL_PARENT_SCHEMA}-manifest"
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint")
        == TIME_LOCAL_PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == TIME_LOCAL_PARENT_CONFIG_SHA256,
        "time-local manifest binding changed",
    )
    _require(
        config.get("schema") == f"{TIME_LOCAL_PARENT_SCHEMA}-scientific-config"
        and config.get("schema_version") == 1
        and config.get("semantic_sha256") == TIME_LOCAL_PARENT_CONFIG_SHA256,
        "time-local scientific configuration changed",
    )
    _same_resolved_path(
        config.get("parent_memory_v3_run_dir"), memory, "memory-v3 config"
    )
    _same_resolved_path(
        config.get("parent_coarse_witness_run_dir"), witness, "witness config"
    )
    _same_resolved_path(
        config.get("parent_bayes_power_run_dir"), bayes, "Bayes config"
    )
    _require(
        status.get("schema") == f"{TIME_LOCAL_PARENT_SCHEMA}-status"
        and status.get("state") == "complete"
        and status.get("stage") == "decompose"
        and status.get("decision") == TIME_LOCAL_PARENT_DECISION
        and int(status.get("scientific_evidence_complete", 0)) == 1,
        "time-local terminal status changed",
    )
    _require(
        decision.get("decision") == TIME_LOCAL_PARENT_DECISION
        and decision.get("evaluation_status") == "evaluated"
        and int(decision.get("scientific_evidence_complete", 0)) == 1
        and int(decision.get("fresh_quartile_specialist_planning_authorized", 0))
        == 1
        and int(decision.get("new_training_authorized", 0)) == 0,
        "time-local terminal decision changed",
    )
    for row, description in (
        (manifest, "time-local manifest"),
        (registry, "time-local registry"),
        (status, "time-local status"),
        (decision, "time-local decision"),
        (parent_record, "time-local parent provenance"),
    ):
        _zero_scope(row, description)

    transitive = verify_time_local_adjudication_parents(
        memory_v3_run_dir=memory,
        coarse_witness_run_dir=witness,
        bayes_power_run_dir=bayes,
        verify_external_cache=verify_external_cache,
    )
    _require(int(transitive.get("passed", 0)) == 1, "transitive parents failed")
    recorded = parent_record.get("parents")
    _require(isinstance(recorded, Mapping), "time-local parent table is malformed")
    for role, expected in (
        ("memory_safe_v3_selection", memory),
        ("physical_coarse_signal_witness", witness),
        ("bayes_power_calibration", bayes),
    ):
        row = recorded.get(role)
        _require(isinstance(row, Mapping), f"missing transitive parent role: {role}")
        _same_resolved_path(row.get("run_dir"), expected, role)
        _require(int(row.get("verified", 0)) == 1, f"unverified parent role: {role}")

    after = verify_parent_immutability_snapshot(time_local, before)
    return _hashed(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "authoritative_parent": {
                "run_dir": str(time_local),
                "basename": time_local.name,
                "decision": TIME_LOCAL_PARENT_DECISION,
                "registry_count": TIME_LOCAL_PARENT_REGISTRY_COUNT,
                "registry_semantic_sha256": (
                    TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256
                ),
                "registry_file_sha256": TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256,
                "source_fingerprint": TIME_LOCAL_PARENT_SOURCE_FINGERPRINT,
                "scientific_config_sha256": TIME_LOCAL_PARENT_CONFIG_SHA256,
                "tree_sha256": after["tree_sha256"],
            },
            "transitive_parents": transitive["parents"],
            "transitive_provenance": transitive["transitive_provenance"],
            "all_four_parent_registries_verified": 1,
            "all_registered_artifact_hashes_verified": 1,
            "all_checkpoint_hashes_verified": 1,
            "historical_design_evidence_authorizing": 0,
            "historical_gain_or_checkpoint_reuse_authorized": 0,
            "confirmation_namespace_opened": 0,
            "parents_mutated": 0,
        }
    )


def build_path_id_plan() -> dict[str, Any]:
    roles = {
        role: list(range(start, stop))
        for role, (start, stop) in PATH_ROLE_RANGES.items()
    }
    return _hashed(
        {
            "schema": PATH_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "path_id_plan_version": PATH_PLAN_VERSION,
            "canonical_path_id_bits": PATH_ID_BITS,
            "allocator_reservation": {
                "start": PRODUCTION_RESERVATION[0],
                "stop_exclusive": PRODUCTION_RESERVATION[1],
            },
            "roles": roles,
            "role_slots": {
                role: {
                    "start": start,
                    "stop_exclusive": stop,
                    "path_count": stop - start,
                    "persistence": (
                        "diagnostic_only"
                        if role == "preflight_seam"
                        else "streamed_derived_only"
                        if role in {"fresh_selection", "untouched_confirmation"}
                        else "role_specific_cache"
                    ),
                    "opened": 0,
                }
                for role, (start, stop) in PATH_ROLE_RANGES.items()
            },
            "historical_forbidden_roles": {
                role: {"start": start, "stop_exclusive": stop}
                for role, (start, stop) in HISTORICAL_PATH_RANGES.items()
            },
            "historical_confirmation_range_reuse_authorized": 0,
            "collision_free": 1,
        }
    )


def _claim_values(value: Any) -> set[int]:
    if isinstance(value, Mapping):
        if "start" in value and "stop_exclusive" in value:
            start, stop = value["start"], value["stop_exclusive"]
            _require(
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(stop, int)
                and not isinstance(stop, bool)
                and start <= stop,
                "path claim interval is malformed",
            )
            value = range(start, stop)
        elif "path_ids" in value:
            value = value["path_ids"]
    elif (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and value[0] <= value[1]
    ):
        value = range(value[0], value[1])
    try:
        raw_values = tuple(value)
    except TypeError as exc:
        raise QuartileSpecialistProvenanceError(
            "path claim is not iterable"
        ) from exc
    result: set[int] = set()
    for raw in raw_values:
        _require(
            isinstance(raw, int) and not isinstance(raw, bool),
            "path claim is not an integer",
        )
        _require(0 <= raw < PATH_ID_LIMIT, "path claim is outside 20-bit bounds")
        _require(raw not in result, "path claim contains a duplicate ID")
        result.add(raw)
    return result


def validate_path_id_plan(
    plan: Mapping[str, Any],
    *,
    claimed_ids: Mapping[str, Any] | Iterable[int] | None = None,
) -> dict[str, Any]:
    expected = build_path_id_plan()
    _require(dict(plan) == expected, "quartile-specialist path plan changed")
    active_by_role = {
        role: set(range(start, stop))
        for role, (start, stop) in PATH_ROLE_RANGES.items()
    }
    roles = tuple(active_by_role)
    active = set().union(*active_by_role.values())
    for role, values in active_by_role.items():
        start, stop = PATH_ROLE_RANGES[role]
        _require(
            PRODUCTION_RESERVATION[0] <= start < stop <= PRODUCTION_RESERVATION[1]
            and len(values) == stop - start,
            f"{role} path range is invalid",
        )
    for index, role in enumerate(roles):
        for other in roles[index + 1 :]:
            _require(
                active_by_role[role].isdisjoint(active_by_role[other]),
                f"path roles overlap: {role} and {other}",
            )
    historical = {
        value
        for start, stop in HISTORICAL_PATH_RANGES.values()
        for value in range(start, stop)
    }
    _require(active.isdisjoint(historical), "new path IDs collide with history")

    if claimed_ids is None:
        claims: Mapping[str, Any] = {}
    elif isinstance(claimed_ids, Mapping):
        claims = claimed_ids
    else:
        claims = {"external": claimed_ids}
    allocator = set(range(*PRODUCTION_RESERVATION))
    allowed_allocator_claims: list[str] = []
    collisions: list[dict[str, Any]] = []
    for source, raw in claims.items():
        values = _claim_values(raw)
        if "reserv" in str(source).lower() and values == allocator:
            allowed_allocator_claims.append(str(source))
            continue
        collisions.extend(
            {"source": str(source), "path_id": value}
            for value in sorted(active & values)
        )
    _require(not collisions, "quartile-specialist path claim collision")
    return _hashed(
        {
            "schema": f"{PATH_PLAN_SCHEMA}-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "path_id_plan_sha256": expected["semantic_sha256"],
            "active_path_count": len(active),
            "role_disjointness_pass": 1,
            "historical_disjointness_pass": 1,
            "twenty_bit_bounds_pass": 1,
            "collision_count": 0,
            "allowed_allocator_claims": sorted(allowed_allocator_claims),
        }
    )


def _partition(role: str, values: Sequence[int], sizes: Sequence[int]) -> list[dict[str, Any]]:
    _require(sum(sizes) == len(values), f"{role} cohort sizes do not partition IDs")
    result: list[dict[str, Any]] = []
    cursor = 0
    for index, size in enumerate(sizes):
        path_ids = tuple(values[cursor : cursor + size])
        cursor += size
        result.append(
            {
                "index": index,
                "role": role,
                "size": size,
                "path_ids": list(path_ids),
            }
        )
    return result


def build_cohort_plan(
    path_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = build_path_id_plan() if path_plan is None else dict(path_plan)
    validate_path_id_plan(plan)
    return _hashed(
        {
            "schema": COHORT_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "cohort_plan_version": COHORT_PLAN_VERSION,
            "path_id_plan_sha256": plan["semantic_sha256"],
            "maximum_exact_generation_cohort_size": 10,
            "roles": {
                role: _partition(role, plan["roles"][role], ROLE_COHORT_SIZES[role])
                for role in PATH_ROLE_RANGES
            },
            "cross_role_cohorts": 0,
            "selection_cohort_count": 39,
            "confirmation_cohort_count": 39,
        }
    )


def validate_cohort_plan(
    cohort_plan: Mapping[str, Any],
    *,
    path_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_path = build_path_id_plan() if path_plan is None else dict(path_plan)
    expected = build_cohort_plan(expected_path)
    _require(dict(cohort_plan) == expected, "quartile-specialist cohort plan changed")
    seen: set[int] = set()
    for role in PATH_ROLE_RANGES:
        flattened: list[int] = []
        for index, cohort in enumerate(expected["roles"][role]):
            _require(
                cohort["index"] == index
                and cohort["role"] == role
                and cohort["size"] == len(cohort["path_ids"])
                and 1 <= cohort["size"] <= 10,
                f"{role} cohort is malformed",
            )
            flattened.extend(cohort["path_ids"])
        _require(
            flattened == expected_path["roles"][role],
            f"{role} cohort order changed",
        )
        _require(seen.isdisjoint(flattened), "cohort roles overlap")
        seen.update(flattened)
    return _hashed(
        {
            "schema": f"{COHORT_PLAN_SCHEMA}-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "cohort_plan_sha256": expected["semantic_sha256"],
            "maximum_cohort_size": 10,
            "role_ordering_valid": 1,
            "cross_role_cohorts": 0,
        }
    )


def build_seed_plan() -> dict[str, Any]:
    return _hashed(
        {
            "schema": f"{SCHEMA}-seed-plan",
            "schema_version": SCHEMA_VERSION,
            "root_seed": ROOT_SEED,
            "physical_model_seeds": {
                f"q{quartile}": list(seeds)
                for quartile, seeds in PHYSICAL_MODEL_SEEDS.items()
            },
            "selection_bootstrap": {
                "seed": SELECTION_BOOTSTRAP_SEED,
                "namespace": SELECTION_BOOTSTRAP_NAMESPACE,
            },
            "confirmation_bootstrap": {
                "seed": CONFIRMATION_BOOTSTRAP_SEED,
                "namespace": CONFIRMATION_BOOTSTRAP_NAMESPACE,
            },
            "synthetic_control_seeds": list(SYNTHETIC_CONTROL_SEEDS),
            "exact_null_control_root_seed": EXACT_NULL_CONTROL_ROOT_SEED,
            "reserved_future_control_seed": RESERVED_FUTURE_CONTROL_SEED,
        }
    )


def validate_seed_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_seed_plan()
    _require(dict(plan) == expected, "quartile-specialist seed plan changed")
    seeds = [ROOT_SEED]
    seeds.extend(seed for values in PHYSICAL_MODEL_SEEDS.values() for seed in values)
    seeds.extend((SELECTION_BOOTSTRAP_SEED, CONFIRMATION_BOOTSTRAP_SEED))
    seeds.extend(SYNTHETIC_CONTROL_SEEDS)
    seeds.extend((EXACT_NULL_CONTROL_ROOT_SEED, RESERVED_FUTURE_CONTROL_SEED))
    _require(len(seeds) == len(set(seeds)), "quartile-specialist seeds collide")
    _require(
        SELECTION_BOOTSTRAP_NAMESPACE != CONFIRMATION_BOOTSTRAP_NAMESPACE,
        "bootstrap namespaces collide",
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-seed-plan-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "seed_plan_sha256": expected["semantic_sha256"],
            "unique_seed_count": len(seeds),
            "bootstrap_namespace_disjointness": 1,
        }
    )


def build_role_firewall() -> dict[str, Any]:
    return _hashed(
        {
            "schema": f"{SCHEMA}-role-firewall",
            "schema_version": SCHEMA_VERSION,
            "role_open_order": list(ROLE_OPEN_ORDER),
            "requirements": {
                "physical_fit": ["prelabel_controls_passed"],
                "gain_calibration": ["physical_training_complete"],
                "training_rank": ["gain_calibration_seal_exists"],
                "fresh_selection": ["selected_system_seal_exists"],
                "untouched_confirmation": ["selection_passed"],
            },
            "selection_and_confirmation_raw_cache_authorized": 0,
            "cross_role_cohort_authorized": 0,
            "historical_design_label_reuse_authorized": 0,
        }
    )


def validate_role_firewall(record: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_role_firewall()
    _require(dict(record) == expected, "quartile-specialist role firewall changed")
    return _hashed(
        {
            "schema": f"{SCHEMA}-role-firewall-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "role_firewall_sha256": expected["semantic_sha256"],
        }
    )


def validate_role_open_order(
    opened_roles: Iterable[str],
    *,
    prerequisite_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = tuple(str(value) for value in opened_roles)
    _require(len(values) == len(set(values)), "a role was opened more than once")
    _require(
        all(value in ROLE_OPEN_ORDER for value in values),
        "unknown role-open event",
    )
    indices = tuple(ROLE_OPEN_ORDER.index(value) for value in values)
    _require(indices == tuple(sorted(indices)), "role-open ordering changed")
    # Opening a later evidence role implies every prerequisite role was opened.
    _require(
        not indices or indices == tuple(range(indices[-1] + 1)),
        "role-open sequence skipped a prerequisite role",
    )
    prerequisites = {
        "physical_fit": "prelabel_controls_passed",
        "gain_calibration": "physical_training_complete",
        "training_rank": "gain_calibration_seal_exists",
        "fresh_selection": "selected_system_seal_exists",
        "untouched_confirmation": "selection_passed",
    }
    flags = dict(prerequisite_flags or {})
    if prerequisite_flags is not None:
        for role in values:
            field = prerequisites[role]
            _require(int(flags.get(field, 0)) == 1, f"missing role prerequisite: {field}")
    return _hashed(
        {
            "schema": f"{SCHEMA}-role-open-order-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "opened_roles": list(values),
            "next_role": (
                ROLE_OPEN_ORDER[len(values)] if len(values) < len(ROLE_OPEN_ORDER) else None
            ),
            "confirmation_opened": int("untouched_confirmation" in values),
            "prerequisite_flags_checked": int(prerequisite_flags is not None),
        }
    )


def quartile_source_paths(
    entry_points: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Return the transitive local source closure for manifest binding."""

    if entry_points is None:
        package = Path(__file__).resolve().parent
        names = (
            "d0_jacobi_rb_boundary_tangent_quartile_specialist.py",
            "d0_jacobi_rb_boundary_tangent_quartile_selection.py",
            "d0_jacobi_rb_boundary_tangent_quartile_provenance.py",
            "d0_jacobi_rb_boundary_tangent_quartile_gate.py",
            "diag_d0_jacobi_rb_boundary_tangent_quartile_specialist.py",
        )
        entry_points = tuple(package / name for name in names)
    paths = tuple(Path(path).resolve() for path in entry_points)
    _require(paths and all(path.is_file() for path in paths), "source entry point missing")
    return v3_transitive_source_paths(paths)


def quartile_source_fingerprint(
    entry_points: Iterable[str | Path] | None = None,
) -> str:
    return source_fingerprint(quartile_source_paths(entry_points))


def scientific_config_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the semantic fingerprint, ignoring an existing self-hash."""

    return config_fingerprint(_semantic_body(record))


def validate_semantic_config(
    record: Mapping[str, Any],
    *,
    expected_schema: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a scientific config without prescribing its CLI-only fields."""

    _assert_semantic(record, "quartile-specialist scientific config")
    if expected_schema is not None:
        _require(record.get("schema") == expected_schema, "scientific config schema changed")
    if expected_sha256 is not None:
        _require(
            record.get("semantic_sha256") == expected_sha256,
            "scientific config fingerprint changed",
        )
    return _hashed(
        {
            "schema": f"{SCHEMA}-scientific-config-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "scientific_config_sha256": record["semantic_sha256"],
        }
    )


def verify_resume_compatibility(
    run_dir: str | Path,
    *,
    expected_bindings: Mapping[str, Any],
    artifact_bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Check immutable manifest and semantic artifacts before any resume write."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"resume run does not exist: {root}")
    manifest = _load_json(root / "run_manifest.json", "resume manifest")
    _require(
        all(manifest.get(key) == value for key, value in expected_bindings.items()),
        "resume manifest compatibility changed",
    )
    verified: dict[str, str] = {}
    for relative, expected in (artifact_bindings or {}).items():
        record = _load_json(root / relative, f"resume artifact {relative}")
        _assert_semantic(record, f"resume artifact {relative}")
        _require(
            record.get("semantic_sha256") == expected,
            f"resume artifact changed: {relative}",
        )
        verified[relative] = expected
    return _hashed(
        {
            "schema": f"{SCHEMA}-resume-compatibility",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "run_dir": str(root),
            "expected_bindings": dict(expected_bindings),
            "verified_artifacts": verified,
        }
    )


# Readable and compatibility aliases for workflow callers.
build_quartile_path_plan = build_path_id_plan
validate_quartile_path_plan = validate_path_id_plan
build_quartile_cohort_plan = build_cohort_plan
validate_quartile_cohort_plan = validate_cohort_plan
build_quartile_specialist_path_plan = build_path_id_plan
validate_quartile_specialist_path_plan = validate_path_id_plan
build_quartile_specialist_cohort_plan = build_cohort_plan
validate_quartile_specialist_cohort_plan = validate_cohort_plan
build_quartile_specialist_seed_plan = build_seed_plan
validate_quartile_specialist_seed_plan = validate_seed_plan
quartile_specialist_source_paths = quartile_source_paths
quartile_specialist_source_fingerprint = quartile_source_fingerprint
verify_quartile_specialist_resume_compatibility = verify_resume_compatibility
verify_parent_runs = verify_quartile_specialist_parents


__all__ = [
    "BAYES_PARENT_BASENAME",
    "COHORT_PLAN_SCHEMA",
    "COHORT_PLAN_VERSION",
    "CONFIRMATION_BOOTSTRAP_NAMESPACE",
    "CONFIRMATION_BOOTSTRAP_SEED",
    "EXACT_NULL_CONTROL_ROOT_SEED",
    "HISTORICAL_PATH_RANGES",
    "MEMORY_PARENT_BASENAME",
    "PATH_ID_BITS",
    "PATH_ID_LIMIT",
    "PATH_PLAN_SCHEMA",
    "PATH_PLAN_VERSION",
    "PATH_ROLE_RANGES",
    "PHYSICAL_MODEL_SEEDS",
    "PRODUCTION_RESERVATION",
    "QuartileSpecialistProvenanceError",
    "RESERVED_FUTURE_CONTROL_SEED",
    "ROLE_COHORT_SIZES",
    "ROLE_OPEN_ORDER",
    "ROOT_SEED",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SELECTION_BOOTSTRAP_NAMESPACE",
    "SELECTION_BOOTSTRAP_SEED",
    "SYNTHETIC_CONTROL_SEEDS",
    "TIME_LOCAL_PARENT_BASENAME",
    "TIME_LOCAL_PARENT_CONFIG_SHA256",
    "TIME_LOCAL_PARENT_DECISION",
    "TIME_LOCAL_PARENT_REGISTRY_COUNT",
    "TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256",
    "TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256",
    "TIME_LOCAL_PARENT_SOURCE_FINGERPRINT",
    "WITNESS_PARENT_BASENAME",
    "build_cohort_plan",
    "build_path_id_plan",
    "build_quartile_cohort_plan",
    "build_quartile_path_plan",
    "build_quartile_specialist_cohort_plan",
    "build_quartile_specialist_path_plan",
    "build_quartile_specialist_seed_plan",
    "build_role_firewall",
    "build_seed_plan",
    "quartile_source_fingerprint",
    "quartile_source_paths",
    "quartile_specialist_source_fingerprint",
    "quartile_specialist_source_paths",
    "scientific_config_fingerprint",
    "validate_cohort_plan",
    "validate_path_id_plan",
    "validate_quartile_cohort_plan",
    "validate_quartile_path_plan",
    "validate_quartile_specialist_cohort_plan",
    "validate_quartile_specialist_path_plan",
    "validate_quartile_specialist_seed_plan",
    "validate_role_open_order",
    "validate_role_firewall",
    "validate_seed_plan",
    "validate_semantic_config",
    "verify_parent_runs",
    "verify_quartile_specialist_parents",
    "verify_quartile_specialist_resume_compatibility",
    "verify_resume_compatibility",
]
