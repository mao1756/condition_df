"""Read-only recovery helpers for the interrupted certified Haar power pilot.

The original nested panel-A shards are immutable evidence.  This module
validates and replays those commits without invoking a scheduler or modifying
the parent run.  It also provides the additive antithetic-panel entry point
needed after the recovered nested panel is adjudicated as ineligible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_haar_gate import (
    ANTITHETIC_HAAR_PROFILE,
    NESTED_HAAR_PROFILE,
    nominate_haar_power_design,
)
from mnist.d0_jacobi_rb_haar_power import (
    HAAR_POWER_VERSION,
    NO_WORK,
    _antithetic_pair,
    _branch_checkpoints,
    _execution_record,
    _flatten_differences,
    _initial_states,
    _ordered_level_observables,
    _panel_evidence_flags,
    _payload,
    build_power_candidates,
    verify_certified_haar_power_panel_evidence,
)
from mnist.d0_jacobi_rb_haar_scheduler import (
    ADJACENT_LEVEL_PAIRS,
    HaarShardIdentity,
    NestedHaarSchedule,
    PairwiseHaarAntitheticSchedule,
    expected_haar_shard_input_sha256,
    initialize_nested_branch_states,
    load_committed_haar_shard,
)


HAAR_POWER_RECOVERY_VERSION = "d0-jacobi-rb-haar-power-recovery-v1"
CANONICAL_SCHEDULE_BINDING_VERSION = (
    HAAR_POWER_RECOVERY_VERSION + "-canonical-schedule-v1"
)
FROZEN_ROOT_SEED = 261181
FROZEN_PARENT_ARTIFACT_COUNT = 197
FROZEN_PARENT_REGISTRY_SHA256 = (
    "4bf1dab4c0905533fe0df885521fb3309ed6344e13f1fd67faad7fa9ae11abfe"
)
FROZEN_PARENT_SOURCE_FINGERPRINT = (
    "300bcdab17d9cac5605311bf0b513a5c476e88011662fb1e51ac69ca4f431c39"
)
FROZEN_PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "cb26ea614d20f7695b02fa063aafca41d0a229ff34a26e9bbb0610bdda1352cf"
)
INHERITED_COUPLING_PEAK_MEMORY_FRACTION = 0.006818991116410677
EXPECTED_NESTED_TRANSITIONS = 120_823_808
EXPECTED_NESTED_FALLBACKS = 38
EXPECTED_NESTED_FALLBACK_COST_FRACTION = 0.0005989668245920508
EXPECTED_NESTED_MASS_ERROR = 1.3322676295501878e-15
EXPECTED_NESTED_CONSERVATIVE_RATE = 4202.429019551445


class HaarPowerRecoveryError(RuntimeError):
    """The immutable recovery evidence failed a closed validation."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        **diagnostics: Any,
    ) -> None:
        super().__init__(message)
        self.failure_code = str(failure_code)
        self.failure_domain = "haar_power_recovery"
        self.diagnostics = dict(diagnostics)


def _fail(message: str, failure_code: str, **diagnostics: Any) -> None:
    raise HaarPowerRecoveryError(
        message,
        failure_code=failure_code,
        **diagnostics,
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HaarPowerRecoveryError(
            "a schedule is not canonical JSON",
            failure_code="panel_schedule_binding_invalid",
        ) from exc


def _schedule_object(value: Mapping[str, Any]) -> Any:
    profile = value.get("profile_name")
    try:
        if profile == NESTED_HAAR_PROFILE:
            return NestedHaarSchedule(
                pool=str(value["pool"]),
                role=str(value["role"]),
                levels=tuple(int(item) for item in value["levels"]),
            )
        if profile == ANTITHETIC_HAAR_PROFILE:
            return PairwiseHaarAntitheticSchedule(
                coarse_steps=int(value["coarse_steps"]),
                fine_steps=int(value["fine_steps"]),
                role=str(value["role"]),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise HaarPowerRecoveryError(
            "the canonical shard schedule is malformed",
            failure_code="panel_schedule_binding_invalid",
        ) from exc
    _fail(
        "the canonical shard schedule has an unknown profile",
        "panel_schedule_binding_invalid",
        profile_name=profile,
    )


def canonical_schedule(
    record: Mapping[str, Any],
    *,
    expected_profile: str | None = None,
    expected_pool: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Return the versioned canonical schedule bound by ``identity.schedule``.

    Historical shard records legitimately omit a top-level ``schedule``.
    When a top-level copy is present it is accepted only if it is canonically
    identical to ``identity.schedule``.  The identity copy itself must be the
    exact record emitted by the frozen scheduler dataclass.
    """

    if not isinstance(record, Mapping):
        _fail("the shard record is not a mapping", "panel_schedule_binding_invalid")
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        _fail(
            "the shard identity is missing",
            "panel_schedule_binding_invalid",
        )
    bound = identity.get("schedule")
    if not isinstance(bound, Mapping):
        _fail(
            "identity.schedule is missing",
            "panel_schedule_binding_invalid",
        )
    schedule = _schedule_object(bound)
    canonical = schedule.to_record()
    if _canonical_json(dict(bound)) != _canonical_json(canonical):
        _fail(
            "identity.schedule is not the frozen canonical schedule",
            "panel_schedule_binding_invalid",
        )
    if "schedule" in record:
        top_level = record.get("schedule")
        if (
            not isinstance(top_level, Mapping)
            or _canonical_json(dict(top_level)) != _canonical_json(canonical)
        ):
            _fail(
                "top-level schedule conflicts with identity.schedule",
                "panel_schedule_binding_invalid",
            )
    if expected_profile is not None and canonical["profile_name"] != expected_profile:
        _fail(
            "the shard schedule has the wrong profile",
            "panel_schedule_binding_invalid",
            expected_profile=expected_profile,
            actual_profile=canonical["profile_name"],
        )
    if expected_pool is not None and canonical.get("pool") != expected_pool:
        _fail(
            "the shard schedule has the wrong pool",
            "panel_schedule_binding_invalid",
            expected_pool=expected_pool,
            actual_pool=canonical.get("pool"),
        )
    if expected_role is not None and canonical.get("role") != expected_role:
        _fail(
            "the shard schedule has the wrong role",
            "panel_schedule_binding_invalid",
            expected_role=expected_role,
            actual_role=canonical.get("role"),
        )
    return dict(canonical)


def execution_record_with_canonical_schedules(
    records: Sequence[Mapping[str, Any]],
    *,
    peak_memory_fraction: float,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    """Aggregate committed diagnostics after resolving canonical schedules."""

    normalized: list[dict[str, Any]] = []
    for record in records:
        schedule = canonical_schedule(
            record,
            expected_profile=expected_profile,
        )
        normalized.append({**dict(record), "schedule": schedule})
    return _execution_record(
        normalized,
        peak_memory_fraction=float(peak_memory_fraction),
    )


def _load_json(path: Path, failure_code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HaarPowerRecoveryError(
            f"{path.name} is unreadable",
            failure_code=failure_code,
        ) from exc
    if not isinstance(value, Mapping):
        _fail(f"{path.name} is not a JSON object", failure_code)
    return dict(value)


def _safe_registered_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail(
            "the artifact registry contains an invalid path",
            "control_provenance_invalid",
        )
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        _fail(
            "the artifact registry escapes the immutable parent",
            "control_provenance_invalid",
            relative_path=relative,
        )
    return path


def _validated_registry(parent: Path) -> dict[str, Any]:
    registry_path = parent / "artifact_registry.json"
    if (
        not registry_path.is_file()
        or file_fingerprint(registry_path) != FROZEN_PARENT_REGISTRY_SHA256
    ):
        _fail(
            "the immutable parent artifact registry changed",
            "control_provenance_invalid",
        )
    registry = _load_json(registry_path, "control_provenance_invalid")
    records = registry.get("records")
    if not isinstance(records, Mapping) or len(records) != FROZEN_PARENT_ARTIFACT_COUNT:
        _fail(
            "the immutable parent artifact count changed",
            "control_provenance_invalid",
            expected_count=FROZEN_PARENT_ARTIFACT_COUNT,
            actual_count=(len(records) if isinstance(records, Mapping) else None),
        )
    return registry


def _verify_registered_file(
    parent: Path,
    registry: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(parent).as_posix()
    except ValueError:
        _fail(
            "a requested artifact is outside the immutable parent",
            "control_provenance_invalid",
            path=str(path),
        )
    records = registry.get("records")
    binding = records.get(relative) if isinstance(records, Mapping) else None
    if not isinstance(binding, Mapping):
        _fail(
            "an immutable shard is absent from the parent registry",
            "control_provenance_invalid",
            relative_path=relative,
        )
    registered = _safe_registered_path(parent, relative)
    if (
        registered != path.resolve()
        or not path.is_file()
        or int(binding.get("size", -1)) != path.stat().st_size
        or binding.get("sha256") != file_fingerprint(path)
    ):
        _fail(
            "an immutable shard failed its registry binding",
            "control_provenance_invalid",
            relative_path=relative,
        )
    return {
        "relative_path": relative,
        "sha256": str(binding["sha256"]),
        "size": int(binding["size"]),
    }


def _validated_parent_context(
    parent_run_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    parent = Path(parent_run_dir).resolve()
    if not parent.is_dir():
        _fail(
            "the immutable parent run does not exist",
            "control_provenance_invalid",
            parent_run_dir=str(parent),
        )
    registry = _validated_registry(parent)
    manifest_path = parent / "run_manifest.json"
    config_path = parent / "scientific_config.json"
    provenance_path = parent / "parent_provenance.json"
    path_plan_path = parent / "haar_path_id_plan.json"
    for path in (manifest_path, config_path, provenance_path, path_plan_path):
        _verify_registered_file(parent, registry, path)
    manifest = _load_json(manifest_path, "control_provenance_invalid")
    config = _load_json(config_path, "control_provenance_invalid")
    provenance = _load_json(provenance_path, "control_provenance_invalid")
    path_plan = _load_json(path_plan_path, "control_provenance_invalid")
    if (
        manifest.get("source_count") != 35
        or manifest.get("source_fingerprint")
        != FROZEN_PARENT_SOURCE_FINGERPRINT
        or manifest.get("scientific_config_sha256")
        != FROZEN_PARENT_SCIENTIFIC_CONFIG_SHA256
        or config.get("root_seed") != FROZEN_ROOT_SEED
    ):
        _fail(
            "the immutable parent scientific binding changed",
            "control_provenance_invalid",
        )
    upstream_value = provenance.get("parent_run_dir")
    if not isinstance(upstream_value, str):
        _fail(
            "the source-image parent binding is missing",
            "control_provenance_invalid",
        )
    upstream = Path(upstream_value).resolve()
    upstream_registry_path = upstream / "artifact_registry.json"
    if (
        not upstream_registry_path.is_file()
        or file_fingerprint(upstream_registry_path)
        != provenance.get("parent_artifact_registry_sha256")
    ):
        _fail(
            "the transitive source parent registry changed",
            "control_provenance_invalid",
        )
    upstream_registry = _load_json(
        upstream_registry_path,
        "control_provenance_invalid",
    )
    upstream_records = upstream_registry.get("records")
    if (
        not isinstance(upstream_records, Mapping)
        or len(upstream_records) != int(provenance.get("parent_artifact_record_count", -1))
    ):
        _fail(
            "the transitive source parent artifact count changed",
            "control_provenance_invalid",
        )
    for name in ("source_image.json", "source_image.npz"):
        path = upstream / name
        binding = upstream_records.get(name)
        if (
            not isinstance(binding, Mapping)
            or not path.is_file()
            or int(binding.get("size", -1)) != path.stat().st_size
            or binding.get("sha256") != file_fingerprint(path)
        ):
            _fail(
                "the transitive source-image binding changed",
                "control_provenance_invalid",
                relative_path=name,
            )
    return parent, registry, path_plan, upstream, provenance


def _nested_pool_replay(
    *,
    parent: Path,
    registry: Mapping[str, Any],
    upstream: Path,
    path_plan: Mapping[str, Any],
    pool: str,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    schedule = NestedHaarSchedule(pool=pool, role="nested_a")
    levels = tuple(int(value) for value in schedule.levels or ())
    try:
        plan = path_plan["profiles"][NESTED_HAAR_PROFILE]["a"]["roles"][pool]
        path_ids = tuple(int(value) for value in plan["root_path_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HaarPowerRecoveryError(
            "the frozen nested path plan is malformed",
            failure_code="control_provenance_invalid",
        ) from exc
    initial, _ = _initial_states(upstream, len(path_ids), torch.device("cpu"))
    states = initialize_nested_branch_states(initial, schedule)
    accumulators: dict[int, Any] = {level: None for level in levels}
    raw: dict[int, dict[int, np.ndarray]] = {level: {} for level in levels}
    dynkin: dict[int, dict[int, np.ndarray]] = {level: {} for level in levels}
    records: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    expected_stems: set[str] = set()
    shard_root = (
        parent
        / "haar_power_shards"
        / NESTED_HAAR_PROFILE
        / "a"
        / pool
    )
    profile = JacobiRBCudaProfile()
    for first in range(0, schedule.coarsest_steps, 8):
        identity = HaarShardIdentity(
            schedule=schedule,
            path_ids=path_ids,
            coarsest_start_step=first,
            root_seed=FROZEN_ROOT_SEED,
            panel_namespace=f"haar-power:a:{pool}",
        )
        expected_stems.add(identity.fingerprint)
        metadata_path = shard_root / f"{identity.fingerprint}.json"
        archive_path = shard_root / f"{identity.fingerprint}.npz"
        metadata_binding = _verify_registered_file(
            parent,
            registry,
            metadata_path,
        )
        archive_binding = _verify_registered_file(
            parent,
            registry,
            archive_path,
        )
        metadata = _load_json(
            metadata_path,
            "nested_panel_replay_invalid",
        )
        if "schedule" in metadata:
            _fail(
                "the historical shard unexpectedly has a top-level schedule",
                "panel_schedule_binding_invalid",
                relative_path=metadata_binding["relative_path"],
            )
        bound_schedule = canonical_schedule(
            metadata,
            expected_profile=NESTED_HAAR_PROFILE,
            expected_pool=pool,
            expected_role="nested_a",
        )
        try:
            resumed = load_committed_haar_shard(
                shard_root,
                expected_identity=identity,
                device="cpu",
            )
        except Exception as exc:
            raise HaarPowerRecoveryError(
                "an immutable nested shard failed internal verification",
                failure_code="nested_panel_replay_invalid",
                relative_path=metadata_binding["relative_path"],
                error_type=type(exc).__name__,
            ) from exc
        named_states = {f"k{level}": states[level] for level in levels}
        named_accumulators = {
            f"k{level}": accumulators[level] for level in levels
        }
        expected_input = expected_haar_shard_input_sha256(
            identity,
            named_states,
            named_accumulators,
            profile,
        )
        if resumed.metadata.get("input_sha256") != expected_input:
            _fail(
                "an immutable nested shard breaks the predecessor chain",
                "nested_panel_replay_invalid",
                relative_path=metadata_binding["relative_path"],
            )
        states = {
            level: resumed.states[f"k{level}"]
            for level in levels
        }
        accumulators = {
            level: resumed.accumulators[f"k{level}"]
            for level in levels
        }
        for level in levels:
            name = f"k{level}"
            completed = resumed.metadata["branches"][name]["completed_steps"]
            _branch_checkpoints(
                completed,
                resumed.raw_observables[name],
                sample_steps=level,
                destination=raw[level],
            )
            _branch_checkpoints(
                completed,
                resumed.dynkin_observables[name],
                sample_steps=level,
                destination=dynkin[level],
            )
        records.append(dict(resumed.metadata))
        audit.append(
            {
                "pool": pool,
                "coarsest_start_step": first,
                "identity_sha256": identity.fingerprint,
                "input_sha256": str(resumed.metadata["input_sha256"]),
                "output_sha256": str(resumed.metadata["output_sha256"]),
                "metadata": metadata_binding,
                "archive": archive_binding,
                "schedule_source": "identity.schedule",
                "canonical_schedule": bound_schedule,
                "canonical_schedule_binding_version": (
                    CANONICAL_SCHEDULE_BINDING_VERSION
                ),
                "internal_archive_hash_pass": 1,
                "predecessor_chain_pass": 1,
            }
        )
    actual_files = {
        path.stem
        for path in shard_root.iterdir()
        if path.is_file() and path.suffix in {".json", ".npz"}
    }
    if actual_files != expected_stems:
        _fail(
            "the immutable nested shard set changed",
            "nested_panel_replay_invalid",
            pool=pool,
            expected_count=len(expected_stems),
            actual_count=len(actual_files),
        )
    return (
        {
            level: _ordered_level_observables(raw[level], level)
            for level in levels
        },
        {
            level: _ordered_level_observables(dynkin[level], level)
            for level in levels
        },
        records,
        audit,
    )


def _path_isolation(path_plan: Mapping[str, Any], profile: str, panel: str) -> bool:
    try:
        roles = path_plan["profiles"][profile][panel]["roles"]
        reserved = path_plan["reserved_production_slot"]
        values = [
            int(path_id)
            for role in roles.values()
            for path_id in role["root_path_ids"]
        ]
        return bool(
            len(reserved) == 2
            and len(values) == len(set(values))
            and all(
                not int(reserved[0]) <= path_id < int(reserved[1])
                for path_id in values
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _assert_frozen_nested_result(
    execution: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    nomination: Mapping[str, Any],
) -> None:
    exact = {
        "shard_count": 80,
        "transition_count": EXPECTED_NESTED_TRANSITIONS,
        "certified_count": EXPECTED_NESTED_TRANSITIONS,
        "fallback_count": EXPECTED_NESTED_FALLBACKS,
    }
    if any(int(execution.get(name, -1)) != value for name, value in exact.items()):
        _fail(
            "the recovered nested execution counts changed",
            "nested_panel_replay_invalid",
        )
    if (
        float(execution.get("certificate_fraction", 0.0)) != 1.0
        or any(int(execution.get(name, -1)) != 0 for name in (
            "uncertified_count",
            "resource_cap_count",
            "invalid_density_count",
            "approximation_count",
            "correction_count",
            "floor_count",
            "limiter_count",
            "projection_count",
            "renormalization_count",
            "nonfinite_count",
        ))
        or not math.isclose(
            float(execution.get("fallback_fraction", math.inf)),
            EXPECTED_NESTED_FALLBACKS / EXPECTED_NESTED_TRANSITIONS,
            rel_tol=0.0,
            abs_tol=1.0e-20,
        )
        or not math.isclose(
            float(execution.get("fallback_cost_fraction", math.inf)),
            EXPECTED_NESTED_FALLBACK_COST_FRACTION,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            float(execution.get("mass_error", math.inf)),
            EXPECTED_NESTED_MASS_ERROR,
            rel_tol=0.0,
            abs_tol=1.0e-24,
        )
        or not math.isclose(
            float(execution.get("conservative_rate", 0.0)),
            EXPECTED_NESTED_CONSERVATIVE_RATE,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        _fail(
            "the recovered nested diagnostics changed",
            "nested_panel_replay_invalid",
        )
    required_flags = (
        "panel_complete_pass",
        "panel_finite_pass",
        "panel_certification_pass",
        "panel_numerical_health_pass",
        "mass_conservation_pass",
        "shard_chain_pass",
        "pilot_production_isolation_pass",
        "pilot_means_excluded_pass",
        "raw_endpoint_authorizing_pass",
        "dynkin_advisory_only_pass",
    )
    if (
        len(candidates) != 4
        or any(
            not all(int(row.get(name, 0)) == 1 for name in required_flags)
            for row in candidates
        )
        or nomination.get("selection_status") != "panel_a_no_eligible_design"
        or nomination.get("selected") is not None
        or int(nomination.get("eligible_candidate_count", -1)) != 0
    ):
        _fail(
            "the recovered nested nomination changed",
            "nested_panel_replay_invalid",
        )


def replay_nested_panel_a(
    parent_run_dir: str | Path,
    *,
    replay_peak_memory_fraction: float = 0.0,
) -> dict[str, Any]:
    """Reconstruct the sealed nested panel A entirely from committed shards.

    The returned mapping is JSON-friendly except for ``observable_arrays``,
    whose six values are NumPy arrays ready for an atomic NPZ write.
    """

    try:
        measured_peak = float(replay_peak_memory_fraction)
    except (TypeError, ValueError) as exc:
        raise HaarPowerRecoveryError(
            "replay peak memory is invalid",
            failure_code="nested_panel_replay_invalid",
        ) from exc
    if not math.isfinite(measured_peak) or not 0.0 <= measured_peak <= 1.0:
        _fail(
            "replay peak memory is invalid",
            "nested_panel_replay_invalid",
        )
    parent, registry, path_plan, upstream, provenance = _validated_parent_context(
        parent_run_dir
    )
    main_raw, main_dynkin, main_records, main_audit = _nested_pool_replay(
        parent=parent,
        registry=registry,
        upstream=upstream,
        path_plan=path_plan,
        pool="main",
    )
    reference_raw, reference_dynkin, reference_records, reference_audit = (
        _nested_pool_replay(
            parent=parent,
            registry=registry,
            upstream=upstream,
            path_plan=path_plan,
            pool="reference",
        )
    )
    records = [*main_records, *reference_records]
    peak = max(
        measured_peak,
        INHERITED_COUPLING_PEAK_MEMORY_FRACTION,
    )
    execution = execution_record_with_canonical_schedules(
        records,
        peak_memory_fraction=peak,
        expected_profile=NESTED_HAAR_PROFILE,
    )
    execution.update(
        {
            "timing_coverage_pass": 1,
            "committed_timing_coverage_pass": 1,
            "canonical_schedule_timing_groups": 2,
            "canonical_schedule_binding_pass": 1,
            "canonical_schedule_binding_version": (
                CANONICAL_SCHEDULE_BINDING_VERSION
            ),
            "replay_peak_memory_fraction": measured_peak,
            "inherited_coupling_peak_memory_fraction": (
                INHERITED_COUPLING_PEAK_MEMORY_FRACTION
            ),
        }
    )
    raw_d1 = main_raw[128] - main_raw[256]
    raw_d2 = main_raw[256] - main_raw[512]
    raw_d3 = main_raw[512] - main_raw[1024]
    raw_d4 = reference_raw[1024] - reference_raw[2048]
    dynkin_d1 = main_dynkin[128] - main_dynkin[256]
    dynkin_d2 = main_dynkin[256] - main_dynkin[512]
    dynkin_d3 = main_dynkin[512] - main_dynkin[1024]
    dynkin_d4 = reference_dynkin[1024] - reference_dynkin[2048]
    arrays = {
        "raw_main_differences": _flatten_differences(
            (raw_d1, raw_d2, raw_d3)
        ),
        "raw_d3_differences": _flatten_differences((raw_d3,)),
        "raw_d4_differences": _flatten_differences((raw_d4,)),
        "dynkin_main_differences": _flatten_differences(
            (dynkin_d1, dynkin_d2, dynkin_d3)
        ),
        "dynkin_d3_differences": _flatten_differences((dynkin_d3,)),
        "dynkin_d4_differences": _flatten_differences((dynkin_d4,)),
    }
    finite = all(np.isfinite(value).all() for value in arrays.values())
    isolation = _path_isolation(path_plan, NESTED_HAAR_PROFILE, "a")
    flags = _panel_evidence_flags(
        execution,
        complete=True,
        finite=finite,
        pilot_production_isolation=isolation,
    )
    candidates = build_power_candidates(
        profile=NESTED_HAAR_PROFILE,
        main_differences=arrays["raw_main_differences"],
        d3_differences=arrays["raw_d3_differences"],
        d4_differences=arrays["raw_d4_differences"],
        execution=execution,
        evidence_flags=flags,
    )
    nomination = nominate_haar_power_design(
        profile=NESTED_HAAR_PROFILE,
        panel_role="a",
        candidates=candidates,
    )
    _assert_frozen_nested_result(execution, candidates, nomination)
    audit = [*main_audit, *reference_audit]
    return {
        "schema": HAAR_POWER_RECOVERY_VERSION + "-nested-panel-replay",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "profile": NESTED_HAAR_PROFILE,
        "panel": "a",
        "root_seed": FROZEN_ROOT_SEED,
        "parent_run_dir": str(parent),
        "parent_artifact_record_count": FROZEN_PARENT_ARTIFACT_COUNT,
        "parent_artifact_registry_sha256": FROZEN_PARENT_REGISTRY_SHA256,
        "parent_source_fingerprint": FROZEN_PARENT_SOURCE_FINGERPRINT,
        "parent_scientific_config_sha256": (
            FROZEN_PARENT_SCIENTIFIC_CONFIG_SHA256
        ),
        "source_parent_run_dir": str(upstream),
        "source_parent_artifact_registry_sha256": provenance.get(
            "parent_artifact_registry_sha256"
        ),
        "canonical_schedule_binding_version": (
            CANONICAL_SCHEDULE_BINDING_VERSION
        ),
        "schedule_bindings": audit,
        "schedule_binding_count": len(audit),
        "schedule_binding_pass": 1,
        "shard_audit": {
            "main_shard_count": len(main_records),
            "reference_shard_count": len(reference_records),
            "total_shard_count": len(records),
            "registry_hash_pass": 1,
            "archive_hash_pass": 1,
            "predecessor_chain_pass": 1,
            "parent_mutated": 0,
            "nested_gpu_recomputation_performed": 0,
        },
        "execution": execution,
        "candidates": candidates,
        "nomination": nomination,
        "recovery_decision": "panel_a_no_eligible_design",
        "observable_arrays": arrays,
        "complete": 1,
        "finite": int(finite),
        "production_authorizing": int(all(flags.values()) and isolation),
        **flags,
        **NO_WORK,
    }


def _outside_parent(run_dir: Path, parent: Path) -> None:
    if run_dir == parent or run_dir.is_relative_to(parent):
        _fail(
            "recovery output must be outside the immutable parent",
            "control_provenance_invalid",
            run_dir=str(run_dir),
            parent_run_dir=str(parent),
        )


def run_recovery_antithetic_panel(
    *,
    run_dir: str | Path,
    parent_haar_run_dir: str | Path,
    panel: str,
    device: str | torch.device,
) -> dict[str, Any]:
    """Run or verify one antithetic panel while writing only to ``run_dir``.

    The returned record is compatible with the original sealed Haar panel
    evidence schema.  Existing evidence is hash-verified and returned without
    opening the scheduler again.
    """

    if panel not in {"a", "b"}:
        raise ValueError("panel must be a or b")
    parent, _, path_plan, upstream, _ = _validated_parent_context(
        parent_haar_run_dir
    )
    root = Path(run_dir).resolve()
    _outside_parent(root, parent)
    payload_path = root / f"{ANTITHETIC_HAAR_PROFILE}_panel_{panel}_observables.npz"
    evidence_path = root / f"{ANTITHETIC_HAAR_PROFILE}_panel_{panel}_evidence.json"
    if evidence_path.is_file():
        return verify_certified_haar_power_panel_evidence(
            run_dir=root,
            evidence=_load_json(
                evidence_path,
                "antithetic_scheduler_invalid",
            ),
            expected_profile=ANTITHETIC_HAAR_PROFILE,
            expected_panel=panel,
        )
    device_value = torch.device(device)
    if device_value.type != "cuda":
        _fail(
            "production antithetic panels require CUDA",
            "antithetic_coupling_computationally_infeasible",
            device=str(device_value),
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_value)
    try:
        plan = path_plan["profiles"][ANTITHETIC_HAAR_PROFILE][panel]["roles"]
    except (KeyError, TypeError) as exc:
        raise HaarPowerRecoveryError(
            "the frozen antithetic path plan is malformed",
            failure_code="antithetic_scheduler_invalid",
        ) from exc
    started = time.perf_counter()
    raw_values: list[np.ndarray] = []
    dynkin_values: list[np.ndarray] = []
    pair_path_ids: list[tuple[int, ...]] = []
    all_records: list[dict[str, Any]] = []
    source_metadata: dict[str, Any] = {}
    jacobi_profile = JacobiRBCudaProfile()
    role = f"antithetic_{panel}"
    for coarse, fine in ADJACENT_LEVEL_PAIRS:
        try:
            pair_plan = plan[f"{coarse}-{fine}"]
            paths = tuple(int(value) for value in pair_plan["root_path_ids"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HaarPowerRecoveryError(
                "the frozen antithetic pair plan is malformed",
                failure_code="antithetic_scheduler_invalid",
                pair=f"{coarse}-{fine}",
            ) from exc
        initial, source_metadata = _initial_states(
            upstream,
            len(paths),
            device_value,
        )
        raw_difference, dynkin_difference, records = _antithetic_pair(
            run_dir=root,
            panel=panel,
            role=role,
            path_ids=paths,
            initial=initial,
            root_seed=FROZEN_ROOT_SEED,
            coarse_steps=coarse,
            fine_steps=fine,
            jacobi_profile=jacobi_profile,
        )
        raw_values.append(raw_difference)
        dynkin_values.append(dynkin_difference)
        pair_path_ids.append(paths)
        all_records.extend(records)
    elapsed_outer = time.perf_counter() - started
    measured_peak = (
        torch.cuda.max_memory_allocated(device_value)
        / max(
            float(torch.cuda.get_device_properties(device_value).total_memory),
            1.0,
        )
    )
    peak = max(
        measured_peak,
        INHERITED_COUPLING_PEAK_MEMORY_FRACTION,
    )
    execution = execution_record_with_canonical_schedules(
        all_records,
        peak_memory_fraction=peak,
        expected_profile=ANTITHETIC_HAAR_PROFILE,
    )
    committed_elapsed = float(execution["elapsed_seconds"])
    authorizing_elapsed = max(committed_elapsed, elapsed_outer)
    execution.update(
        {
            "outer_wall_seconds": elapsed_outer,
            "committed_timing_coverage_pass": int(
                committed_elapsed + 1.0e-9 >= elapsed_outer * 0.95
            ),
            "complete_wall_upper_seconds": authorizing_elapsed,
            "elapsed_seconds": authorizing_elapsed,
            "outer_wall_rate": (
                float(execution["transition_count"]) / authorizing_elapsed
            ),
            "timing_coverage_pass": 1,
            "canonical_schedule_timing_groups": len(ADJACENT_LEVEL_PAIRS),
            "canonical_schedule_binding_pass": 1,
            "canonical_schedule_binding_version": (
                CANONICAL_SCHEDULE_BINDING_VERSION
            ),
            "measured_peak_memory_fraction": measured_peak,
            "inherited_coupling_peak_memory_fraction": (
                INHERITED_COUPLING_PEAK_MEMORY_FRACTION
            ),
        }
    )
    execution["conservative_rate"] = min(
        float(execution["conservative_rate"]),
        float(execution["outer_wall_rate"]),
    )
    raw_d1, raw_d2, raw_d3, raw_d4 = raw_values
    dynkin_d1, dynkin_d2, dynkin_d3, dynkin_d4 = dynkin_values
    arrays = {
        "raw_main_differences": _flatten_differences(
            (raw_d1, raw_d2, raw_d3)
        ),
        "raw_d3_differences": _flatten_differences((raw_d3,)),
        "raw_d4_differences": _flatten_differences((raw_d4,)),
        "dynkin_main_differences": _flatten_differences(
            (dynkin_d1, dynkin_d2, dynkin_d3)
        ),
        "dynkin_d3_differences": _flatten_differences((dynkin_d3,)),
        "dynkin_d4_differences": _flatten_differences((dynkin_d4,)),
    }
    path_id_pools = {
        f"{coarse}-{fine}": list(paths)
        for (coarse, fine), paths in zip(
            ADJACENT_LEVEL_PAIRS,
            pair_path_ids,
        )
    }
    finite = all(np.isfinite(value).all() for value in arrays.values())
    isolation = _path_isolation(path_plan, ANTITHETIC_HAAR_PROFILE, panel)
    flags = _panel_evidence_flags(
        execution,
        complete=True,
        finite=finite,
        pilot_production_isolation=isolation,
    )
    candidates = build_power_candidates(
        profile=ANTITHETIC_HAAR_PROFILE,
        main_differences=arrays["raw_main_differences"],
        d3_differences=arrays["raw_d3_differences"],
        d4_differences=arrays["raw_d4_differences"],
        execution=execution,
        evidence_flags=flags,
    )
    root.mkdir(parents=True, exist_ok=True)
    payload_record = _payload(
        payload_path,
        arrays,
        require_existing=False,
    )
    record = {
        "schema": HAAR_POWER_VERSION + "-panel",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "profile": ANTITHETIC_HAAR_PROFILE,
        "panel": panel,
        "root_seed": FROZEN_ROOT_SEED,
        "path_id_plan_sha256": path_plan["path_id_plan_sha256"],
        "main_path_ids": list(pair_path_ids[0]),
        "reference_path_ids": list(pair_path_ids[-1]),
        "path_id_pools": path_id_pools,
        "cluster_count": int(arrays["raw_main_differences"].shape[0]),
        "main_feature_count": 120,
        "reference_feature_count": 80,
        "source_npz_sha256": source_metadata.get("source_npz_sha256"),
        "observable_payload": payload_record,
        "execution": execution,
        "candidates": candidates,
        "complete": 1,
        "finite": int(finite),
        "raw_endpoint_authorizing": 1,
        "dynkin_advisory_only": 1,
        "production_authorizing_pass": int(all(flags.values()) and isolation),
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
        "independent_pool_variance_pass": int(isolation),
        "richardson_formula_pass": int(
            all(
                float(row.get("independent_pool_covariance", math.inf)) == 0.0
                for row in candidates
            )
        ),
        **flags,
        "panel_means_excluded_from_refinement": 1,
        "production_authorizing": int(all(flags.values()) and isolation),
        "canonical_schedule_binding_version": (
            CANONICAL_SCHEDULE_BINDING_VERSION
        ),
        "schedule_binding_count": len(all_records),
        "schedule_binding_pass": 1,
        **NO_WORK,
    }
    normalized = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    atomic_write_json(evidence_path, normalized)
    return normalized


__all__ = [
    "CANONICAL_SCHEDULE_BINDING_VERSION",
    "EXPECTED_NESTED_CONSERVATIVE_RATE",
    "EXPECTED_NESTED_FALLBACKS",
    "EXPECTED_NESTED_FALLBACK_COST_FRACTION",
    "EXPECTED_NESTED_MASS_ERROR",
    "EXPECTED_NESTED_TRANSITIONS",
    "FROZEN_PARENT_ARTIFACT_COUNT",
    "FROZEN_PARENT_REGISTRY_SHA256",
    "FROZEN_PARENT_SCIENTIFIC_CONFIG_SHA256",
    "FROZEN_PARENT_SOURCE_FINGERPRINT",
    "FROZEN_ROOT_SEED",
    "HAAR_POWER_RECOVERY_VERSION",
    "HaarPowerRecoveryError",
    "INHERITED_COUPLING_PEAK_MEMORY_FRACTION",
    "canonical_schedule",
    "execution_record_with_canonical_schedules",
    "replay_nested_panel_a",
    "run_recovery_antithetic_panel",
]
