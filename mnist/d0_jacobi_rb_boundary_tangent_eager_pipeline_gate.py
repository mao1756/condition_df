"""Fail-closed gates for the exact eager-prefix complete-pipeline pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping

from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import (
    BoundaryTangentScheduleThresholds,
    evaluate_schedule_pilot,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-eager-pipeline-gate"
SCHEMA_VERSION = 1
PROJECTED_BASE_TRANSITIONS = 224_788_480
PROJECTED_MIDPOINT_TRANSITIONS = 112_394_240
PROJECTED_TOTAL_TRANSITIONS = 337_182_720
PROFILE_NAMES = ("cache_p10", "cache_p6", "stream_p10", "stream_p4")

NO_WORK = {
    "production_cache_generation_performed": 0,
    "physical_training_performed": 0,
    "controller_control_trajectory_performed": 0,
    "reconstruction_performed": 0,
    "image_sampling_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "full_reverse_path_performed": 0,
}
NO_AUTHORIZATION = {
    "cache_generation_authorized": 0,
    "training_authorized": 0,
    "controller_trajectory_authorized": 0,
    "reconstruction_authorized": 0,
    "sampling_authorized": 0,
    "reverse_sampling_authorized": 0,
}


class BoundaryTangentEagerPipelineGateError(ValueError):
    """Evidence does not satisfy the frozen complete-pipeline gate schema."""


class BoundaryTangentEagerPipelineDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    EAGER_PIPELINE_DESIGN_INVALID = "eager_pipeline_design_invalid"
    EAGER_PREFIX_POLICY_INVALID = "eager_prefix_policy_invalid"
    EAGER_PIPELINE_EXECUTION_INVALID = "eager_pipeline_execution_invalid"
    EAGER_PIPELINE_NUMERICALLY_UNRESOLVED = (
        "eager_pipeline_numerically_unresolved"
    )
    EAGER_PIPELINE_COMPUTATIONALLY_INFEASIBLE = (
        "eager_pipeline_computationally_infeasible"
    )
    EXACT_BOUNDARY_TANGENT_EAGER_PIPELINE_FEASIBLE = (
        "exact_boundary_tangent_eager_pipeline_feasible"
    )


@dataclass(frozen=True)
class BoundaryTangentEagerPipelineThresholds:
    parent_record_count: int = 33
    repeat_count: int = 3
    profile_count: int = 4
    candidate_modes: int = 128
    maximum_projected_seconds: float = 108_000.0
    minimum_effective_rate: float = PROJECTED_TOTAL_TRANSITIONS / 108_000.0
    minimum_individual_profile_rate: float = 1_300.0
    maximum_launch_lanes: int = 4_096
    required_certificate_fraction: float = 1.0
    maximum_fallback_fraction: float = 1.0e-4
    maximum_fallback_time_fraction: float = 0.10
    maximum_mass_error: float = 2.0e-12
    maximum_peak_memory_fraction: float = 0.80
    maximum_persisted_cache_gib: float = 1.25
    projected_base_transitions: int = PROJECTED_BASE_TRANSITIONS
    projected_midpoint_transitions: int = PROJECTED_MIDPOINT_TRANSITIONS
    projected_total_transitions: int = PROJECTED_TOTAL_TRANSITIONS

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if getattr(self, name) != field.default:
                raise BoundaryTangentEagerPipelineGateError(
                    f"{name} is frozen at {field.default}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _at_most(value: Any, threshold: float) -> bool:
    return _finite(value) and float(value) <= float(threshold)


def _at_least(value: Any, threshold: float) -> bool:
    return _finite(value) and float(value) >= float(threshold)


def _exact_float(value: Any, expected: float) -> bool:
    return _finite(value) and float(value) == float(expected)


def _passed(value: bool | int | Mapping[str, Any] | None) -> bool:
    if isinstance(value, Mapping):
        return _one(value.get("passed"))
    return _one(value)


def _check(value: Any, operator: str, threshold: Any, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": int(bool(passed)),
    }


def _gate(
    name: str,
    checks: Mapping[str, Mapping[str, Any]],
    *,
    failure_domain: str | None,
    scientific_evidence_complete: bool,
    **extra: Any,
) -> dict[str, Any]:
    passed = all(_one(value.get("passed")) for value in checks.values())
    return {
        "schema": SCHEMA + f"-{name}",
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "evaluation_status": "evaluated",
        "checks": {key: dict(value) for key, value in checks.items()},
        "passed": int(passed),
        "failure_domain": None if passed else failure_domain,
        "stage_execution_valid": 1,
        "scientific_evidence_complete": int(scientific_evidence_complete),
        **extra,
        **NO_WORK,
    }


def _execution_failed_gate(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA + f"-{name}",
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "evaluation_status": "execution_failed",
        "checks": {"stage_execution": _check(0, "==", 1, False)},
        "passed": 0,
        "failure_domain": str(metrics.get("failure_domain") or "execution"),
        "failure_code": str(metrics.get("failure_code") or f"{name}_execution_failed"),
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        **NO_WORK,
    }


def not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA + f"-{name}",
        "schema_version": SCHEMA_VERSION,
        "gate": name,
        "evaluation_status": "not_evaluated",
        "reason": str(reason),
        "passed": 0,
        "failure_domain": None,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        **NO_WORK,
    }


_PREFLIGHT_FLAGS = (
    "provenance_valid",
    "readjudication_valid",
    "parent_registry_valid",
    "parent_profile_gate_valid",
    "parent_resource_only_failure",
    "parent_stage_execution_valid",
    "parent_numerically_valid",
    "parent_scientific_evidence_complete",
    "only_runtime_checks_failed",
    "eager_profile_frozen",
    "pilot_namespaces_unopened",
    "path_plan_valid",
    "timing_plan_valid",
    "transition_counts_valid",
    "schedule_frozen",
    "cross_role_isolation_valid",
    "output_contract_valid",
    "resume_plan_valid",
    "runtime_contract_valid",
)


def evaluate_eager_pipeline_preflight(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentEagerPipelineThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate parent binding and the frozen complete-pipeline design."""

    t = thresholds or BoundaryTangentEagerPipelineThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate("preflight", metrics)
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in _PREFLIGHT_FLAGS
    }
    for name in NO_WORK:
        checks[name] = _check(metrics.get(name), "==", 0, _zero(metrics.get(name)))
    checks.update(
        {
            "parent_record_count": _check(
                metrics.get("parent_record_count"),
                "==",
                t.parent_record_count,
                metrics.get("parent_record_count") == t.parent_record_count,
            ),
            "repeat_count": _check(
                metrics.get("repeat_count"),
                "==",
                t.repeat_count,
                metrics.get("repeat_count") == t.repeat_count,
            ),
            "profile_count": _check(
                metrics.get("profile_count"),
                "==",
                t.profile_count,
                metrics.get("profile_count") == t.profile_count,
            ),
            "projected_base_transitions": _check(
                metrics.get("projected_base_transitions"),
                "==",
                t.projected_base_transitions,
                metrics.get("projected_base_transitions")
                == t.projected_base_transitions,
            ),
            "projected_midpoint_transitions": _check(
                metrics.get("projected_midpoint_transitions"),
                "==",
                t.projected_midpoint_transitions,
                metrics.get("projected_midpoint_transitions")
                == t.projected_midpoint_transitions,
            ),
            "projected_total_transitions": _check(
                metrics.get("projected_total_transitions"),
                "==",
                t.projected_total_transitions,
                metrics.get("projected_total_transitions")
                == t.projected_total_transitions,
            ),
            "maximum_launch_lanes": _check(
                metrics.get("maximum_launch_lanes"),
                "<=",
                t.maximum_launch_lanes,
                isinstance(metrics.get("maximum_launch_lanes"), int)
                and int(metrics["maximum_launch_lanes"]) <= t.maximum_launch_lanes,
            ),
        }
    )
    failed = {name for name, value in checks.items() if not _one(value["passed"])}
    if failed & {
        "provenance_valid",
        "parent_registry_valid",
        "parent_record_count",
        "parent_profile_gate_valid",
    }:
        domain = "provenance"
    elif failed & {"eager_profile_frozen", "output_contract_valid"}:
        domain = "prefix_policy"
    else:
        domain = "design"
    return _gate(
        "preflight",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=not bool(failed),
        thresholds=t.to_dict(),
    )


_PILOT_POLICY_CHECKS = frozenset(
    {
        "eager_prefix_policy_applied",
        "eager_base_prefix_schedule_valid",
        "eager_branch_prefix_schedule_valid",
        "candidate_modes_unchanged",
    }
)
_PILOT_DESIGN_CHECKS = frozenset(
    {
        "slowest_repeat_selection_valid",
        "repeat_averaging_not_used",
        "posthoc_allowance_not_used",
        "cross_role_isolation_valid",
        "profile_transition_counts",
        "restart_outer_steps",
        "pilot_repeats",
        "completed_shard_count",
        "pilot_total_executed_transition_count",
        "projected_transition_count",
        "base_transition_count",
        "midpoint_transition_count",
        "projection_formula",
    }
)
_PILOT_NUMERICAL_CHECKS = frozenset(
    {
        "repeat_hashes_identical",
        "output_hashes_identical",
        "final_state_hashes_identical",
        "certificate_hashes_identical",
        "certificate_fraction",
        "maximum_mass_error",
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
    }
)
_PILOT_RESOURCE_CHECKS = frozenset(
    {
        "fallback_fraction",
        "fallback_time_fraction",
        "peak_memory_fraction",
        "projected_persisted_bytes",
        "projected_elapsed_seconds",
        "projected_exact_cache_hours",
        "projected_effective_transitions_per_second",
        "maximum_observed_launch_lanes",
        *{f"{name}_rate" for name in PROFILE_NAMES},
    }
)


def evaluate_eager_pipeline_pilot(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentEagerPipelineThresholds | None = None,
) -> dict[str, Any]:
    """Apply the unchanged schedule pilot gate plus eager-policy checks.

    This continuation uses fresh pilot paths, so equality to a parent pilot
    hash is deliberately not part of the gate.  Equality across the three new
    repeats remains mandatory through the wrapped schedule gate.
    """

    t = thresholds or BoundaryTangentEagerPipelineThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate("pilot", metrics)
    base = evaluate_schedule_pilot(
        metrics, thresholds=BoundaryTangentScheduleThresholds()
    )
    checks = dict(base.get("checks", {}))
    checks.update(
        {
            "eager_prefix_policy_applied": _check(
                metrics.get("eager_prefix_policy_applied"),
                "==",
                1,
                _one(metrics.get("eager_prefix_policy_applied")),
            ),
            "eager_base_prefix_schedule_valid": _check(
                metrics.get("eager_base_prefix_schedule_valid"),
                "==",
                1,
                _one(metrics.get("eager_base_prefix_schedule_valid")),
            ),
            "eager_branch_prefix_schedule_valid": _check(
                metrics.get("eager_branch_prefix_schedule_valid"),
                "==",
                1,
                _one(metrics.get("eager_branch_prefix_schedule_valid")),
            ),
            "candidate_modes_unchanged": _check(
                metrics.get("candidate_modes"),
                "==",
                t.candidate_modes,
                metrics.get("candidate_modes") == t.candidate_modes,
            ),
        }
    )
    # The base gate checks its no-work fields.  This continuation also names
    # production cache generation explicitly in every terminal artifact.
    checks["production_cache_generation_performed"] = _check(
        metrics.get("production_cache_generation_performed"),
        "==",
        0,
        _zero(metrics.get("production_cache_generation_performed")),
    )
    failed = {
        name for name, value in checks.items() if not _one(value.get("passed"))
    }
    if failed & _PILOT_POLICY_CHECKS:
        domain = "prefix_policy"
        evidence_complete = False
    elif failed & _PILOT_DESIGN_CHECKS:
        domain = "design"
        evidence_complete = False
    elif failed & _PILOT_NUMERICAL_CHECKS:
        domain = "numerical"
        evidence_complete = True
    elif failed and failed <= _PILOT_RESOURCE_CHECKS:
        domain = "resource_gate"
        evidence_complete = True
    elif failed:
        domain = "execution"
        evidence_complete = False
    else:
        domain = None
        evidence_complete = True
    numerical_valid = not bool(failed & _PILOT_NUMERICAL_CHECKS)
    resource_valid = not bool(failed & _PILOT_RESOURCE_CHECKS)
    resource_only_failure = bool(
        failed and failed <= _PILOT_RESOURCE_CHECKS and numerical_valid
    )
    return _gate(
        "pilot",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=evidence_complete,
        numerically_valid=int(numerical_valid),
        resource_valid=int(resource_valid),
        resource_only_failure=int(resource_only_failure),
        legacy_schedule_gate=base,
        thresholds=t.to_dict(),
    )


def _status(gate: Mapping[str, Any] | None) -> str:
    return str((gate or {}).get("evaluation_status", "missing"))


def _ready(stage: str, action: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA + "-decision",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "not_evaluated",
        "decision": f"ready_for_{stage}",
        "recommended_next_action": action,
        "schedule_integration_authorized": 0,
        **NO_AUTHORIZATION,
        **NO_WORK,
    }


def decide_eager_pipeline_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the closed workflow decision and narrowly scoped authorization."""

    if not _passed(provenance):
        decision = BoundaryTangentEagerPipelineDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 33-artifact eager-prefix parent binding"
    elif _status(preflight_gate) == "not_evaluated":
        return _ready("preflight", "run the complete-pipeline preflight")
    elif not _passed(preflight_gate):
        domain = str((preflight_gate or {}).get("failure_domain", ""))
        if domain == "provenance":
            decision = BoundaryTangentEagerPipelineDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable eager-prefix parent binding"
        elif domain == "prefix_policy":
            decision = BoundaryTangentEagerPipelineDecision.EAGER_PREFIX_POLICY_INVALID
            action = "repair the frozen exact eager-prefix policy binding"
        elif domain == "execution":
            decision = BoundaryTangentEagerPipelineDecision.EAGER_PIPELINE_EXECUTION_INVALID
            action = "repair complete-pipeline preflight execution"
        else:
            decision = BoundaryTangentEagerPipelineDecision.EAGER_PIPELINE_DESIGN_INVALID
            action = "repair the frozen complete-pipeline design"
    elif _status(pilot_gate) == "not_evaluated":
        return _ready("pilot", "run the sealed complete-pipeline pilot")
    elif not _passed(pilot_gate):
        domain = str((pilot_gate or {}).get("failure_domain", ""))
        if domain == "resource_gate":
            decision = BoundaryTangentEagerPipelineDecision.EAGER_PIPELINE_COMPUTATIONALLY_INFEASIBLE
            action = "retain exact evidence and plan unresolved-lane compaction"
        elif domain == "numerical":
            decision = BoundaryTangentEagerPipelineDecision.EAGER_PIPELINE_NUMERICALLY_UNRESOLVED
            action = "repair exact numerical or certificate execution"
        elif domain == "prefix_policy":
            decision = BoundaryTangentEagerPipelineDecision.EAGER_PREFIX_POLICY_INVALID
            action = "repair the frozen eager-prefix policy"
        elif domain == "design":
            decision = BoundaryTangentEagerPipelineDecision.EAGER_PIPELINE_DESIGN_INVALID
            action = "repair the complete-pipeline timing design"
        else:
            decision = BoundaryTangentEagerPipelineDecision.EAGER_PIPELINE_EXECUTION_INVALID
            action = "repair complete-pipeline execution or replay"
    else:
        decision = BoundaryTangentEagerPipelineDecision.EXACT_BOUNDARY_TANGENT_EAGER_PIPELINE_FEASIBLE
        action = "integrate the frozen eager-prefix scheduler into a fresh v2 workflow"
    feasible = (
        decision
        is BoundaryTangentEagerPipelineDecision.EXACT_BOUNDARY_TANGENT_EAGER_PIPELINE_FEASIBLE
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


def evaluate_eager_pipeline_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    if require_gate not in {"none", "preflight", "pilot"}:
        raise BoundaryTangentEagerPipelineGateError(
            f"unknown required gate: {require_gate}"
        )
    components = {
        "preflight": dict(preflight_gate or not_evaluated_gate("preflight", "not run")),
        "pilot": dict(pilot_gate or not_evaluated_gate("pilot", "not run")),
    }
    required = {
        "none": (),
        "preflight": ("preflight",),
        "pilot": ("preflight", "pilot"),
    }[require_gate]
    required_pass = _passed(provenance) and all(
        _passed(components[name]) for name in required
    )
    decision = decide_eager_pipeline_workflow(
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
        "thresholds": BoundaryTangentEagerPipelineThresholds().to_dict(),
        **NO_AUTHORIZATION,
        **NO_WORK,
    }


__all__ = [
    "BoundaryTangentEagerPipelineDecision",
    "BoundaryTangentEagerPipelineGateError",
    "BoundaryTangentEagerPipelineThresholds",
    "NO_AUTHORIZATION",
    "NO_WORK",
    "PROFILE_NAMES",
    "PROJECTED_BASE_TRANSITIONS",
    "PROJECTED_MIDPOINT_TRANSITIONS",
    "PROJECTED_TOTAL_TRANSITIONS",
    "decide_eager_pipeline_workflow",
    "evaluate_eager_pipeline_pilot",
    "evaluate_eager_pipeline_preflight",
    "evaluate_eager_pipeline_workflow",
    "not_evaluated_gate",
]
