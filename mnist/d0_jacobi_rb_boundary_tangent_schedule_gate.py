"""Pure fail-closed gates for fused-lane boundary-tangent scheduling.

The gate changes execution packing only.  It authorizes neither cache
generation nor training, controller trajectories, reconstruction, or sampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping, Sequence


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-schedule-gate"
SCHEMA_VERSION = 1
CLAIM_SCOPE = (
    "exact CUDA execution-schedule feasibility for the unchanged 64/32/64 "
    "boundary-tangent Jacobi/Rao-Blackwell workflow"
)
NO_WORK = {
    "physical_training_performed": 0,
    "controller_control_trajectory_performed": 0,
    "full_reverse_path_performed": 0,
    "image_sampling_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "reconstruction_performed": 0,
}
NO_AUTHORIZATION = {
    "cache_generation_authorized": 0,
    "physical_training_authorized": 0,
    "controller_control_trajectory_authorized": 0,
    "reconstruction_authorized": 0,
    "sampling_authorized": 0,
    "reverse_sampling_authorized": 0,
}


class BoundaryTangentScheduleGateError(ValueError):
    """Evidence violates the frozen schedule-gate contract."""


class BoundaryTangentScheduleDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    BOUNDARY_TANGENT_SCHEDULE_ALGEBRA_INVALID = (
        "boundary_tangent_schedule_algebra_invalid"
    )
    BOUNDARY_TANGENT_SCHEDULE_EQUIVALENCE_INVALID = (
        "boundary_tangent_schedule_equivalence_invalid"
    )
    BOUNDARY_TANGENT_SCHEDULE_EXECUTION_INVALID = (
        "boundary_tangent_schedule_execution_invalid"
    )
    BOUNDARY_TANGENT_SCHEDULE_COMPUTATIONALLY_INFEASIBLE = (
        "boundary_tangent_schedule_computationally_infeasible"
    )
    EXACT_BOUNDARY_TANGENT_SCHEDULE_FEASIBLE = (
        "exact_boundary_tangent_schedule_feasible"
    )


@dataclass(frozen=True)
class BoundaryTangentScheduleThresholds:
    """Frozen production design and resource thresholds."""

    failed_parent_record_count: int = 14
    root_seed: int = 261_321
    train_paths: int = 64
    validation_paths: int = 32
    confirmation_paths: int = 64
    cache_group_sizes: tuple[int, ...] = (10,) * 9 + (6,)
    stream_group_sizes: tuple[int, ...] = (10,) * 6 + (4,)
    edge_count_per_phase: int = 392
    phase_count: int = 7
    midpoint_count: int = 8
    sample_steps: int = 512
    restart_outer_steps: int = 8
    timing_window_starts: tuple[int, ...] = (0, 128, 256, 384)
    timing_window_outer_steps: int = 16
    timing_branch_steps: tuple[int, ...] = (15, 143, 271, 399)
    pilot_repeats: int = 3
    pilot_completed_shard_count: int = 96
    pilot_total_executed_transition_count: int = 23_708_160
    maximum_launch_lanes: int = 4096
    cache_p10_transitions_per_repeat: int = 2_634_240
    cache_p6_transitions_per_repeat: int = 1_580_544
    stream_p10_transitions_per_repeat: int = 2_634_240
    stream_p4_transitions_per_repeat: int = 1_053_696
    base_transition_count: int = 224_788_480
    midpoint_transition_count: int = 112_394_240
    projected_transition_count: int = 337_182_720
    maximum_projected_exact_cache_hours: float = 30.0
    maximum_projected_elapsed_seconds: float = 108_000.0
    minimum_projected_effective_transitions_per_second: float = (
        337_182_720 / 108_000.0
    )
    minimum_profile_transitions_per_second: float = 1_300.0
    maximum_mass_error: float = 2.0e-12
    maximum_peak_memory_fraction: float = 0.80
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_time_fraction: float = 0.10
    maximum_persisted_bytes: int = 5 * 1024**3 // 4

    def __post_init__(self) -> None:
        expected = BoundaryTangentScheduleThresholds.__dataclass_fields__
        for name, field in expected.items():
            if field.default is not field.default_factory:  # pragma: no cover - sentinel
                default = field.default
            else:  # pragma: no cover - no default factories in this dataclass
                default = field.default_factory()
            if getattr(self, name) != default:
                raise BoundaryTangentScheduleGateError(
                    f"{name} is frozen at {default}"
                )

    @property
    def profile_transition_counts(self) -> dict[str, int]:
        return {
            "cache_p10": self.cache_p10_transitions_per_repeat,
            "cache_p6": self.cache_p6_transitions_per_repeat,
            "stream_p10": self.stream_p10_transitions_per_repeat,
            "stream_p4": self.stream_p4_transitions_per_repeat,
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for name in (
            "cache_group_sizes",
            "stream_group_sizes",
            "timing_window_starts",
            "timing_branch_steps",
        ):
            result[name] = list(result[name])
        result["profile_transition_counts"] = self.profile_transition_counts
        return result


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _status(gate: Mapping[str, Any] | None) -> str:
    if not isinstance(gate, Mapping):
        return "not_evaluated"
    return str(gate.get("evaluation_status", "not_evaluated"))


def _passed(gate: bool | int | Mapping[str, Any] | None) -> bool:
    if isinstance(gate, Mapping):
        return _status(gate) == "evaluated" and _one(gate.get("passed"))
    return gate is True or (isinstance(gate, int) and not isinstance(gate, bool) and gate == 1)


def _check(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _eq_one(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", 1, _one(value))


def _eq_zero(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", 0, _zero(value))


def _equal(metrics: Mapping[str, Any], name: str, threshold: Any) -> dict[str, Any]:
    value = metrics.get(name)
    return _check(value, "==", threshold, value == threshold)


def _sequence_equal(
    metrics: Mapping[str, Any], name: str, threshold: Sequence[int]
) -> dict[str, Any]:
    value = metrics.get(name)
    valid = isinstance(value, (list, tuple)) and tuple(value) == tuple(threshold)
    return _check(value, "==", list(threshold), valid)


def _le(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    valid = _finite(value) and 0.0 <= float(value) <= float(threshold)
    return _check(value, "<=", threshold, valid)


def _ge(metrics: Mapping[str, Any], name: str, threshold: float) -> dict[str, Any]:
    value = metrics.get(name)
    valid = _finite(value) and float(value) >= float(threshold)
    return _check(value, ">=", threshold, valid)


def _mapping_exact(
    value: Any, expected: Mapping[str, Any]
) -> bool:
    return isinstance(value, Mapping) and dict(value) == dict(expected)


def _gate(
    name: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_status: str = "evaluated",
    failure_domain: str | None = None,
    scientific_evidence_complete: bool = False,
    stage_execution_valid: bool = True,
    numerically_valid: bool = True,
    resource_valid: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in checks.items()}
    passed = bool(
        evaluation_status == "evaluated"
        and normalized
        and all(_one(value.get("passed")) for value in normalized.values())
    )
    return {
        "schema": SCHEMA + f"-{name}-gate",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": evaluation_status,
        "gate": name,
        "claim_scope": CLAIM_SCOPE,
        "checks": normalized,
        "passed": int(passed),
        "failure_domain": None if passed else failure_domain,
        "scientific_evidence_complete": int(scientific_evidence_complete),
        "stage_execution_valid": int(stage_execution_valid),
        "numerically_valid": int(numerically_valid),
        "resource_valid": int(resource_valid),
        **NO_AUTHORIZATION,
        **NO_WORK,
        **extra,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    result = _gate(
        name,
        {},
        evaluation_status="not_evaluated",
        failure_domain=None,
        scientific_evidence_complete=False,
        stage_execution_valid=False,
        numerically_valid=False,
        resource_valid=False,
    )
    result["reason"] = str(reason)
    return result


_PREFLIGHT_FLAGS = (
    "provenance_valid",
    "readjudication_valid",
    "path_plan_valid",
    "cohort_plan_valid",
    "timing_plan_valid",
    "initial_states_valid",
    "launch_plan_valid",
    "path_collision_free",
    "base_equivalence_valid",
    "fused_branch_equivalence_valid",
    "cross_role_isolation_valid",
    "no_work_valid",
    "parent_sources_immutable",
    "canonical_id_uniqueness_valid",
    "canonical_id_order_invariance_valid",
    "p10_singleton_equivalence_valid",
    "path_permutation_invariance_valid",
    "chunk_invariance_valid",
    "repeat_rotation_valid",
    "atomic_commit_plan_valid",
)
_PREFLIGHT_PROVENANCE = frozenset(
    {
        "provenance_valid",
        "readjudication_valid",
        "parent_sources_immutable",
        "failed_parent_record_count",
    }
)
_PREFLIGHT_ALGEBRA = frozenset(
    {
        "path_plan_valid",
        "cohort_plan_valid",
        "timing_plan_valid",
        "initial_states_valid",
        "launch_plan_valid",
        "path_collision_free",
        "canonical_id_uniqueness_valid",
        "repeat_rotation_valid",
        "transition_count_algebra",
        "root_seed",
        "profile_transition_counts",
        "cache_group_sizes",
        "stream_group_sizes",
        "maximum_launch_lanes",
        "maximum_observed_launch_lanes",
        "timing_window_starts",
        "timing_branch_steps",
        "timing_window_outer_steps",
        "pilot_repeats",
        "restart_outer_steps",
        "base_transition_count",
        "midpoint_transition_count",
        "projected_transition_count",
        "maximum_projected_exact_cache_hours",
    }
)
_PREFLIGHT_EQUIVALENCE = frozenset(
    {
        "base_equivalence_valid",
        "fused_branch_equivalence_valid",
        "cross_role_isolation_valid",
        "canonical_id_order_invariance_valid",
        "p10_singleton_equivalence_valid",
        "path_permutation_invariance_valid",
        "chunk_invariance_valid",
        "scientific_target_changed",
    }
)


def evaluate_schedule_preflight(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentScheduleThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate immutable binding, schedule algebra, and exact parity controls."""

    t = thresholds or BoundaryTangentScheduleThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _gate(
            "preflight",
            {"stage_execution": _check(0, "==", 1, False)},
            evaluation_status="execution_failed",
            failure_domain="execution",
            stage_execution_valid=False,
            numerically_valid=False,
            resource_valid=False,
            scientific_evidence_complete=False,
            failure_code=str(metrics.get("failure_code", "preflight_execution_failed")),
        )
    checks = {name: _eq_one(metrics, name) for name in _PREFLIGHT_FLAGS}
    counts = t.profile_transition_counts
    checks.update(
        {
            "failed_parent_record_count": _equal(
                metrics,
                "failed_parent_record_count",
                t.failed_parent_record_count,
            ),
            "root_seed": _equal(metrics, "root_seed", t.root_seed),
            "cache_group_sizes": _sequence_equal(
                metrics, "cache_group_sizes", t.cache_group_sizes
            ),
            "stream_group_sizes": _sequence_equal(
                metrics, "stream_group_sizes", t.stream_group_sizes
            ),
            "timing_window_starts": _sequence_equal(
                metrics, "timing_window_starts", t.timing_window_starts
            ),
            "timing_branch_steps": _sequence_equal(
                metrics, "timing_branch_steps", t.timing_branch_steps
            ),
            "timing_window_outer_steps": _equal(
                metrics,
                "timing_window_outer_steps",
                t.timing_window_outer_steps,
            ),
            "pilot_repeats": _equal(metrics, "pilot_repeats", t.pilot_repeats),
            "maximum_launch_lanes": _le(
                metrics, "maximum_launch_lanes", float(t.maximum_launch_lanes)
            ),
            "maximum_observed_launch_lanes": _le(
                metrics,
                "maximum_observed_launch_lanes",
                float(t.maximum_launch_lanes),
            ),
            "profile_transition_counts": _check(
                metrics.get("profile_transition_counts"),
                "==",
                counts,
                _mapping_exact(metrics.get("profile_transition_counts"), counts),
            ),
            "base_transition_count": _equal(
                metrics, "base_transition_count", t.base_transition_count
            ),
            "midpoint_transition_count": _equal(
                metrics, "midpoint_transition_count", t.midpoint_transition_count
            ),
            "projected_transition_count": _equal(
                metrics, "projected_transition_count", t.projected_transition_count
            ),
            "restart_outer_steps": _equal(
                metrics, "restart_outer_steps", t.restart_outer_steps
            ),
            "maximum_projected_exact_cache_hours": _equal(
                metrics,
                "maximum_projected_exact_cache_hours",
                t.maximum_projected_exact_cache_hours,
            ),
            "transition_count_algebra": _eq_one(
                metrics, "transition_count_algebra"
            ),
            "scientific_target_changed": _eq_zero(
                metrics, "scientific_target_changed"
            ),
            "production_cache_generated": _eq_zero(
                metrics, "production_cache_generated"
            ),
            **{name: _eq_zero(metrics, name) for name in NO_WORK},
        }
    )
    failed = {
        name for name, check in checks.items() if not _one(check.get("passed"))
    }
    provenance_valid = not bool(failed & _PREFLIGHT_PROVENANCE)
    algebra_valid = not bool(failed & _PREFLIGHT_ALGEBRA) and provenance_valid
    equivalence_valid = not bool(failed & _PREFLIGHT_EQUIVALENCE) and algebra_valid
    execution_valid = not failed
    if failed & _PREFLIGHT_PROVENANCE:
        domain = "provenance"
    elif failed & _PREFLIGHT_ALGEBRA:
        domain = "schedule_algebra"
    elif failed & _PREFLIGHT_EQUIVALENCE:
        domain = "schedule_equivalence"
    elif failed:
        domain = "execution"
    else:
        domain = None
    result = _gate(
        "preflight",
        checks,
        failure_domain=domain,
        stage_execution_valid=execution_valid,
        numerically_valid=equivalence_valid,
        resource_valid=True,
        scientific_evidence_complete=not failed,
        provenance_valid=int(provenance_valid),
        schedule_algebra_valid=int(algebra_valid),
        schedule_equivalence_valid=int(equivalence_valid),
        thresholds=t.to_dict(),
    )
    return result


_PROFILE_NAMES = ("cache_p10", "cache_p6", "stream_p10", "stream_p4")
_PILOT_PASS_FLAGS = (
    "all_profiles_complete",
    "repeat_hashes_identical",
    "output_hashes_identical",
    "final_state_hashes_identical",
    "certificate_hashes_identical",
    "atomic_shard_chains_valid",
    "resume_replay_valid",
    "completed_repeat_skipping_valid",
    "permitted_input_conversion_valid",
    "raw_label_conversion_valid",
    "cache_commit_valid",
    "predictor_forward_valid",
    "gpu_risk_accumulation_valid",
    "stream_commit_valid",
    "cross_role_isolation_valid",
    "slowest_repeat_selection_valid",
    "repeat_averaging_not_used",
    "posthoc_allowance_not_used",
)
_PILOT_ZERO_COUNTS = (
    "uncertified_count",
    "cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "projection_count",
    "renormalization_count",
    "nonfinite_count",
    "boundary_rejection_count",
    "transition_id_collision_count",
    "repeat_hash_mismatch_count",
)
_PILOT_RESOURCE_NAMES = frozenset(
    {
        "fallback_fraction",
        "fallback_time_fraction",
        "peak_memory_fraction",
        "projected_persisted_bytes",
        "projected_elapsed_seconds",
        "projected_exact_cache_hours",
        "projected_effective_transitions_per_second",
        *{f"{name}_rate" for name in _PROFILE_NAMES},
    }
)


def _positive_repeat_times(value: Any, repeats: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != repeats:
        return None
    if any(not _finite(item) or float(item) <= 0.0 for item in value):
        return None
    return tuple(float(item) for item in value)


def evaluate_schedule_pilot(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentScheduleThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the sealed complete-pipeline four-profile timing panel."""

    t = thresholds or BoundaryTangentScheduleThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _gate(
            "pilot",
            {"stage_execution": _check(0, "==", 1, False)},
            evaluation_status="execution_failed",
            failure_domain="execution",
            stage_execution_valid=False,
            numerically_valid=False,
            resource_valid=False,
            scientific_evidence_complete=False,
            failure_code=str(metrics.get("failure_code", "pilot_execution_failed")),
        )
    checks = {name: _eq_one(metrics, name) for name in _PILOT_PASS_FLAGS}
    checks.update({name: _eq_zero(metrics, name) for name in _PILOT_ZERO_COUNTS})
    checks.update({name: _eq_zero(metrics, name) for name in NO_WORK})
    checks.update(
        {
            "certificate_fraction": _equal(metrics, "certificate_fraction", 1.0),
            "maximum_mass_error": _le(
                metrics, "maximum_mass_error", t.maximum_mass_error
            ),
            "fallback_fraction": _le(
                metrics, "fallback_fraction", t.maximum_fallback_fraction
            ),
            "fallback_time_fraction": _le(
                metrics,
                "fallback_time_fraction",
                t.maximum_fallback_time_fraction,
            ),
            "peak_memory_fraction": _le(
                metrics, "peak_memory_fraction", t.maximum_peak_memory_fraction
            ),
            "projected_persisted_bytes": _le(
                metrics, "projected_persisted_bytes", float(t.maximum_persisted_bytes)
            ),
            "projected_transition_count": _equal(
                metrics, "projected_transition_count", t.projected_transition_count
            ),
            "base_transition_count": _equal(
                metrics, "base_transition_count", t.base_transition_count
            ),
            "midpoint_transition_count": _equal(
                metrics, "midpoint_transition_count", t.midpoint_transition_count
            ),
            "profile_transition_counts": _check(
                metrics.get("profile_transition_counts"),
                "==",
                t.profile_transition_counts,
                _mapping_exact(
                    metrics.get("profile_transition_counts"),
                    t.profile_transition_counts,
                ),
            ),
            "restart_outer_steps": _equal(
                metrics, "restart_outer_steps", t.restart_outer_steps
            ),
            "pilot_repeats": _equal(
                metrics, "pilot_repeats", t.pilot_repeats
            ),
            "completed_shard_count": _equal(
                metrics,
                "completed_shard_count",
                t.pilot_completed_shard_count,
            ),
            "pilot_total_executed_transition_count": _equal(
                metrics,
                "pilot_total_executed_transition_count",
                t.pilot_total_executed_transition_count,
            ),
            "maximum_observed_launch_lanes": _le(
                metrics,
                "maximum_observed_launch_lanes",
                float(t.maximum_launch_lanes),
            ),
            "production_cache_generated": _eq_zero(
                metrics, "production_cache_generated"
            ),
        }
    )
    raw_times = metrics.get("profile_elapsed_seconds")
    times: dict[str, tuple[float, ...]] = {}
    times_valid = isinstance(raw_times, Mapping)
    if times_valid:
        for name in _PROFILE_NAMES:
            value = _positive_repeat_times(raw_times.get(name), t.pilot_repeats)
            if value is None:
                times_valid = False
                break
            times[name] = value
    checks["profile_elapsed_seconds"] = _check(
        raw_times,
        "valid three-repeat mapping",
        list(_PROFILE_NAMES),
        bool(times_valid),
    )
    slowest: dict[str, float] = {}
    rates: dict[str, float] = {}
    projected = math.inf
    effective = 0.0
    if times_valid:
        slowest = {name: max(values) for name, values in times.items()}
        rates = {
            name: t.profile_transition_counts[name] / slowest[name]
            for name in _PROFILE_NAMES
        }
        projected = 8.0 * (
            9.0 * slowest["cache_p10"]
            + slowest["cache_p6"]
            + 6.0 * slowest["stream_p10"]
            + slowest["stream_p4"]
        )
        effective = t.projected_transition_count / projected
    reported_projected = metrics.get("projected_elapsed_seconds")
    reported_effective = metrics.get("projected_effective_transitions_per_second")
    projection_match = (
        times_valid
        and _finite(reported_projected)
        and math.isclose(float(reported_projected), projected, rel_tol=0.0, abs_tol=1e-9)
        and _finite(reported_effective)
        and math.isclose(float(reported_effective), effective, rel_tol=1e-12, abs_tol=1e-9)
    )
    checks["projection_formula"] = _check(
        {
            "reported_elapsed_seconds": reported_projected,
            "computed_elapsed_seconds": projected,
            "reported_effective_rate": reported_effective,
            "computed_effective_rate": effective,
        },
        "==",
        "frozen slowest-repeat projection",
        projection_match,
    )
    checks["projected_elapsed_seconds"] = _check(
        projected,
        "<=",
        t.maximum_projected_elapsed_seconds,
        _finite(projected) and projected <= t.maximum_projected_elapsed_seconds,
    )
    hours = projected / 3600.0
    checks["projected_exact_cache_hours"] = _check(
        hours,
        "<=",
        t.maximum_projected_exact_cache_hours,
        _finite(hours) and hours <= t.maximum_projected_exact_cache_hours,
    )
    checks["projected_effective_transitions_per_second"] = _check(
        effective,
        ">=",
        t.minimum_projected_effective_transitions_per_second,
        _finite(effective)
        and effective >= t.minimum_projected_effective_transitions_per_second,
    )
    for name in _PROFILE_NAMES:
        rate = rates.get(name, 0.0)
        checks[f"{name}_rate"] = _check(
            rate,
            ">=",
            t.minimum_profile_transitions_per_second,
            _finite(rate) and rate >= t.minimum_profile_transitions_per_second,
        )
    failed = {
        name for name, check in checks.items() if not _one(check.get("passed"))
    }
    nonresource = failed - _PILOT_RESOURCE_NAMES
    numerical_valid = not nonresource
    resource_valid = not bool(failed & _PILOT_RESOURCE_NAMES)
    resource_only = bool(failed) and numerical_valid and not resource_valid
    if resource_only:
        domain = "resource_gate"
        evidence_complete = True
    elif failed:
        domain = "execution"
        evidence_complete = False
    else:
        domain = None
        evidence_complete = True
    return _gate(
        "pilot",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=evidence_complete,
        stage_execution_valid=True,
        numerically_valid=numerical_valid,
        resource_valid=resource_valid,
        resource_only_failure=int(resource_only),
        profile_slowest_elapsed_seconds=slowest,
        profile_slowest_transitions_per_second=rates,
        computed_projected_elapsed_seconds=projected,
        computed_projected_exact_cache_hours=hours,
        computed_projected_effective_transitions_per_second=effective,
        thresholds=t.to_dict(),
    )


def _failed_names(gate: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(gate, Mapping):
        return set()
    checks = gate.get("checks")
    if not isinstance(checks, Mapping):
        return set()
    return {
        str(name)
        for name, value in checks.items()
        if not isinstance(value, Mapping) or not _one(value.get("passed"))
    }


def decide_schedule_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply the closed schedule-feasibility decision partition."""

    if not _passed(provenance):
        decision = BoundaryTangentScheduleDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable failed-run and coarse-parent binding"
    elif _status(preflight_gate) == "not_evaluated":
        return {
            "schema": SCHEMA + "-decision",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "not_evaluated",
            "decision": "ready_for_preflight",
            "recommended_next_action": "run the exact schedule preflight",
            "schedule_integration_authorized": 0,
            **NO_AUTHORIZATION,
            **NO_WORK,
        }
    elif not _passed(preflight_gate):
        domain = str((preflight_gate or {}).get("failure_domain", ""))
        failed = _failed_names(preflight_gate)
        if domain == "provenance" or failed & _PREFLIGHT_PROVENANCE:
            decision = BoundaryTangentScheduleDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable parent/source binding"
        elif domain == "schedule_algebra" or failed & _PREFLIGHT_ALGEBRA:
            decision = (
                BoundaryTangentScheduleDecision.BOUNDARY_TANGENT_SCHEDULE_ALGEBRA_INVALID
            )
            action = "repair cohort, lane, ID, or transition-count algebra"
        elif domain == "schedule_equivalence" or failed & _PREFLIGHT_EQUIVALENCE:
            decision = (
                BoundaryTangentScheduleDecision.BOUNDARY_TANGENT_SCHEDULE_EQUIVALENCE_INVALID
            )
            action = "repair fused/legacy equality or evidence-role isolation"
        else:
            decision = (
                BoundaryTangentScheduleDecision.BOUNDARY_TANGENT_SCHEDULE_EXECUTION_INVALID
            )
            action = "repair exact preflight execution and resume semantics"
    elif _status(pilot_gate) == "not_evaluated":
        return {
            "schema": SCHEMA + "-decision",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "not_evaluated",
            "decision": "ready_for_pilot",
            "recommended_next_action": "run the sealed complete-pipeline pilot",
            "schedule_integration_authorized": 0,
            **NO_AUTHORIZATION,
            **NO_WORK,
        }
    elif not _passed(pilot_gate):
        if (
            str((pilot_gate or {}).get("failure_domain", "")) == "resource_gate"
            and _one((pilot_gate or {}).get("numerically_valid"))
            and _one((pilot_gate or {}).get("stage_execution_valid"))
        ):
            decision = (
                BoundaryTangentScheduleDecision.BOUNDARY_TANGENT_SCHEDULE_COMPUTATIONALLY_INFEASIBLE
            )
            action = "retain exact evidence and investigate another scheduling-only repair"
        else:
            decision = (
                BoundaryTangentScheduleDecision.BOUNDARY_TANGENT_SCHEDULE_EXECUTION_INVALID
            )
            action = "repair fused execution, certification, or atomic replay"
    else:
        decision = BoundaryTangentScheduleDecision.EXACT_BOUNDARY_TANGENT_SCHEDULE_FEASIBLE
        action = "integrate the frozen fused schedule into a fresh v2 workflow"
    feasible = (
        decision
        is BoundaryTangentScheduleDecision.EXACT_BOUNDARY_TANGENT_SCHEDULE_FEASIBLE
    )
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "decision": decision.value,
        "recommended_next_action": action,
        "schedule_integration_authorized": int(feasible),
        **NO_AUTHORIZATION,
        **NO_WORK,
    }


def evaluate_schedule_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    """Combine gates while preserving artifacts-before-failure semantics."""

    if require_gate not in {"none", "preflight", "pilot"}:
        raise BoundaryTangentScheduleGateError(
            f"unknown required gate: {require_gate}"
        )
    components = {
        "preflight": dict(
            preflight_gate or not_evaluated_gate("preflight", "not run")
        ),
        "pilot": dict(pilot_gate or not_evaluated_gate("pilot", "not run")),
    }
    required = {
        "none": (),
        "preflight": ("preflight",),
        "pilot": ("preflight", "pilot"),
    }[require_gate]
    required_pass = bool(
        _passed(provenance) and all(_passed(components[name]) for name in required)
    )
    decision = decide_schedule_workflow(
        provenance=provenance,
        preflight_gate=components["preflight"],
        pilot_gate=components["pilot"],
    )
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "components": components,
        "decision": decision,
        "thresholds": BoundaryTangentScheduleThresholds().to_dict(),
        **NO_AUTHORIZATION,
        **NO_WORK,
    }


__all__ = [
    "BoundaryTangentScheduleDecision",
    "BoundaryTangentScheduleGateError",
    "BoundaryTangentScheduleThresholds",
    "CLAIM_SCOPE",
    "NO_AUTHORIZATION",
    "NO_WORK",
    "SCHEMA",
    "SCHEMA_VERSION",
    "decide_schedule_workflow",
    "evaluate_schedule_pilot",
    "evaluate_schedule_preflight",
    "evaluate_schedule_workflow",
    "not_evaluated_gate",
]
