"""Fail-closed gates for exact eager-prefix certificate scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Mapping

from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import (
    BoundaryTangentScheduleThresholds,
    evaluate_schedule_pilot,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-prefix-schedule-gate"
SCHEMA_VERSION = 1
NO_WORK = {
    "physical_training_performed": 0,
    "controller_control_trajectory_performed": 0,
    "reconstruction_performed": 0,
    "image_sampling_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "full_reverse_path_performed": 0,
}
NO_AUTHORIZATION = {
    "training_authorized": 0,
    "controller_trajectory_authorized": 0,
    "reconstruction_authorized": 0,
    "sampling_authorized": 0,
    "reverse_sampling_authorized": 0,
}


class BoundaryTangentPrefixScheduleGateError(ValueError):
    """Evidence does not satisfy the frozen eager-prefix gate schema."""


class BoundaryTangentPrefixScheduleDecision(str, Enum):
    CONTROL_PROVENANCE_INVALID = "control_provenance_invalid"
    EAGER_PREFIX_RNG_CONTRACT_INVALID = "eager_prefix_rng_contract_invalid"
    EAGER_PREFIX_EQUIVALENCE_INVALID = "eager_prefix_equivalence_invalid"
    EAGER_PREFIX_CERTIFICATE_INVALID = "eager_prefix_certificate_invalid"
    EAGER_PREFIX_PROFILE_COMPUTATIONALLY_INFEASIBLE = (
        "eager_prefix_profile_computationally_infeasible"
    )
    EAGER_PREFIX_SCHEDULE_EXECUTION_INVALID = (
        "eager_prefix_schedule_execution_invalid"
    )
    EAGER_PREFIX_SCHEDULE_COMPUTATIONALLY_INFEASIBLE = (
        "eager_prefix_schedule_computationally_infeasible"
    )
    EXACT_BOUNDARY_TANGENT_EAGER_PREFIX_SCHEDULE_FEASIBLE = (
        "exact_boundary_tangent_eager_prefix_schedule_feasible"
    )


@dataclass(frozen=True)
class BoundaryTangentPrefixScheduleThresholds:
    parent_record_count: int = 614
    profile_repeats: int = 3
    candidate_modes: int = 128
    candidate_bisection_steps: int = 56
    threads_per_block: int = 128
    initial_eager_prefix_bits: int = 128
    maximum_prefix_bits: int = 1024
    maximum_projected_seconds: float = 108_000.0
    minimum_effective_rate: float = 337_182_720 / 108_000.0

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            if getattr(self, name) != field.default:
                raise BoundaryTangentPrefixScheduleGateError(
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
    failure_domain: str | None = None,
    scientific_evidence_complete: bool = True,
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
        "failure_code": str(
            metrics.get("failure_code") or f"{name}_execution_failed"
        ),
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
    "parent_resource_only_failure",
    "parent_scientific_evidence_complete",
    "parent_registry_valid",
    "candidate_unchanged",
    "thread_geometry_unchanged",
    "same_philox_key_and_counter",
    "same_infinite_dyadic_uniform",
    "second_word_revealed_earlier_only",
    "eager_prefix_policy_observed",
    "prefix_interval_nesting_valid",
    "base_scientific_output_equivalent",
    "branch_scientific_output_equivalent",
    "final_state_hashes_equal",
    "canonical_ids_equal",
    "permutation_invariant",
    "chunk_invariant",
    "resume_invariant",
    "eager_certificate_fraction_one",
    "arb_oracle_valid",
    "eager_prefix_arb_fallback_valid",
    "facet_zero_mass_duration_valid",
    "path_plan_valid",
    "cohort_plan_valid",
    "timing_plan_valid",
    "path_collision_free",
    "initial_states_valid",
    "launch_plan_valid",
    "cross_role_isolation_valid",
)


def evaluate_prefix_schedule_preflight(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentPrefixScheduleThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentPrefixScheduleThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate("preflight", metrics)
    checks = {
        name: _check(metrics.get(name), "==", 1, _one(metrics.get(name)))
        for name in _PREFLIGHT_FLAGS
    }
    for name in NO_WORK:
        checks[name] = _check(metrics.get(name), "==", 0, _zero(metrics.get(name)))
    checks["parent_record_count"] = _check(
        metrics.get("parent_record_count"),
        "==",
        t.parent_record_count,
        metrics.get("parent_record_count") == t.parent_record_count,
    )
    checks["candidate_modes"] = _check(
        metrics.get("candidate_modes"), "==", t.candidate_modes,
        metrics.get("candidate_modes") == t.candidate_modes,
    )
    checks["candidate_bisection_steps"] = _check(
        metrics.get("candidate_bisection_steps"), "==", t.candidate_bisection_steps,
        metrics.get("candidate_bisection_steps") == t.candidate_bisection_steps,
    )
    checks["threads_per_block"] = _check(
        metrics.get("threads_per_block"), "==", t.threads_per_block,
        metrics.get("threads_per_block") == t.threads_per_block,
    )
    checks["maximum_prefix_bits"] = _check(
        metrics.get("maximum_prefix_bits"), "==", t.maximum_prefix_bits,
        metrics.get("maximum_prefix_bits") == t.maximum_prefix_bits,
    )
    checks["initial_eager_prefix_bits"] = _check(
        metrics.get("initial_eager_prefix_bits"),
        "==",
        t.initial_eager_prefix_bits,
        metrics.get("initial_eager_prefix_bits") == t.initial_eager_prefix_bits,
    )
    checks["forbidden_event_count"] = _check(
        metrics.get("forbidden_event_count"), "==", 0,
        _zero(metrics.get("forbidden_event_count")),
    )
    failed = {name for name, value in checks.items() if not _one(value["passed"])}
    if failed & {"provenance_valid", "parent_registry_valid", "parent_record_count"}:
        domain = "provenance"
    elif failed & {
        "same_philox_key_and_counter", "same_infinite_dyadic_uniform",
        "second_word_revealed_earlier_only", "prefix_interval_nesting_valid",
    }:
        domain = "rng_contract"
    else:
        domain = "equivalence"
    return _gate(
        "preflight",
        checks,
        failure_domain=domain,
        scientific_evidence_complete=not bool(failed),
        thresholds=t.to_dict(),
    )


def evaluate_prefix_schedule_profile(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentPrefixScheduleThresholds | None = None,
) -> dict[str, Any]:
    t = thresholds or BoundaryTangentPrefixScheduleThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate("profile", metrics)
    checks = {
        "profile_complete": _check(metrics.get("profile_complete"), "==", 1, _one(metrics.get("profile_complete"))),
        "profile_repeat_count": _check(
            metrics.get("profile_repeat_count"), "==", t.profile_repeats,
            metrics.get("profile_repeat_count") == t.profile_repeats,
        ),
        "scientific_outputs_equal": _check(metrics.get("scientific_outputs_equal"), "==", 1, _one(metrics.get("scientific_outputs_equal"))),
        "eager_prefix_policy_observed": _check(
            metrics.get("eager_prefix_policy_observed"),
            "==",
            1,
            _one(metrics.get("eager_prefix_policy_observed")),
        ),
        "certificate_fraction": _check(metrics.get("certificate_fraction"), "==", 1.0, metrics.get("certificate_fraction") == 1.0),
        "forbidden_event_count": _check(metrics.get("forbidden_event_count"), "==", 0, _zero(metrics.get("forbidden_event_count"))),
        "projected_elapsed_seconds": _check(
            metrics.get("projected_elapsed_seconds"), "<=", t.maximum_projected_seconds,
            _finite(metrics.get("projected_elapsed_seconds"))
            and float(metrics["projected_elapsed_seconds"]) <= t.maximum_projected_seconds,
        ),
        "projected_effective_transitions_per_second": _check(
            metrics.get("projected_effective_transitions_per_second"), ">=", t.minimum_effective_rate,
            _finite(metrics.get("projected_effective_transitions_per_second"))
            and float(metrics["projected_effective_transitions_per_second"]) >= t.minimum_effective_rate,
        ),
    }
    for name in NO_WORK:
        checks[name] = _check(metrics.get(name), "==", 0, _zero(metrics.get(name)))
    failed = {name for name, value in checks.items() if not _one(value["passed"])}
    execution_names = {
        "profile_complete", "profile_repeat_count", "scientific_outputs_equal",
        "eager_prefix_policy_observed",
        "certificate_fraction", "forbidden_event_count",
    } | set(NO_WORK)
    resource_only = bool(failed) and not bool(failed & execution_names)
    return _gate(
        "profile",
        checks,
        failure_domain=("resource_gate" if resource_only else "execution"),
        scientific_evidence_complete=resource_only or not bool(failed),
        numerically_valid=int(not bool(failed & execution_names)),
        resource_valid=int(not bool(failed - execution_names)),
        resource_only_failure=int(resource_only),
        thresholds=t.to_dict(),
    )


def evaluate_prefix_schedule_pilot(
    metrics: Mapping[str, Any],
    *,
    thresholds: BoundaryTangentPrefixScheduleThresholds | None = None,
) -> dict[str, Any]:
    """Apply the unchanged parent pilot gate plus eager-policy checks."""

    t = thresholds or BoundaryTangentPrefixScheduleThresholds()
    if metrics.get("evaluation_status") == "execution_failed":
        return _execution_failed_gate("pilot", metrics)
    base = evaluate_schedule_pilot(
        metrics, thresholds=BoundaryTangentScheduleThresholds()
    )
    checks = dict(base.get("checks", {}))
    checks.update(
        {
            "eager_prefix_policy_applied": _check(
                metrics.get("eager_prefix_policy_applied"), "==", 1,
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
                metrics.get("candidate_modes"), "==", t.candidate_modes,
                metrics.get("candidate_modes") == t.candidate_modes,
            ),
            "scientific_hashes_match_parent": _check(
                metrics.get("scientific_hashes_match_parent"), "==", 1,
                _one(metrics.get("scientific_hashes_match_parent")),
            ),
        }
    )
    failed = {name for name, value in checks.items() if not _one(value.get("passed"))}
    prefix_failed = failed & {
        "eager_prefix_policy_applied", "eager_base_prefix_schedule_valid",
        "eager_branch_prefix_schedule_valid", "candidate_modes_unchanged",
        "scientific_hashes_match_parent",
    }
    if prefix_failed:
        domain = "execution"
        evidence_complete = False
    else:
        domain = base.get("failure_domain")
        evidence_complete = _one(base.get("scientific_evidence_complete"))
    return _gate(
        "pilot",
        checks,
        failure_domain=None if not failed else str(domain or "execution"),
        scientific_evidence_complete=evidence_complete,
        numerically_valid=int(_one(base.get("numerically_valid")) and not prefix_failed),
        resource_valid=int(_one(base.get("resource_valid"))),
        resource_only_failure=int(_one(base.get("resource_only_failure")) and not prefix_failed),
        legacy_schedule_gate=base,
        thresholds=t.to_dict(),
    )


def _status(gate: Mapping[str, Any] | None) -> str:
    return str((gate or {}).get("evaluation_status", "missing"))


def decide_prefix_schedule_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    profile_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not _passed(provenance):
        decision = BoundaryTangentPrefixScheduleDecision.CONTROL_PROVENANCE_INVALID
        action = "repair the immutable 614-artifact parent binding"
    elif _status(preflight_gate) == "not_evaluated":
        return _ready("preflight", "run the exact eager-prefix preflight")
    elif not _passed(preflight_gate):
        domain = str((preflight_gate or {}).get("failure_domain", ""))
        if domain == "provenance":
            decision = BoundaryTangentPrefixScheduleDecision.CONTROL_PROVENANCE_INVALID
            action = "repair the immutable 614-artifact parent binding"
        elif domain == "rng_contract":
            decision = BoundaryTangentPrefixScheduleDecision.EAGER_PREFIX_RNG_CONTRACT_INVALID
            action = "repair the exact stateless-Philox prefix contract"
        elif domain == "certificate_execution":
            decision = BoundaryTangentPrefixScheduleDecision.EAGER_PREFIX_CERTIFICATE_INVALID
            action = "repair the exact eager-prefix certificate fallback"
        elif domain in {"schedule_execution", "execution"}:
            decision = BoundaryTangentPrefixScheduleDecision.EAGER_PREFIX_SCHEDULE_EXECUTION_INVALID
            action = "repair exact eager-prefix preflight execution"
        else:
            decision = BoundaryTangentPrefixScheduleDecision.EAGER_PREFIX_EQUIVALENCE_INVALID
            action = "repair exact eager/adaptive output equivalence"
    elif _status(profile_gate) == "not_evaluated":
        return _ready("profile", "run the sealed eager-prefix qualification")
    elif not _passed(profile_gate):
        if str((profile_gate or {}).get("failure_domain")) == "resource_gate":
            decision = BoundaryTangentPrefixScheduleDecision.EAGER_PREFIX_PROFILE_COMPUTATIONALLY_INFEASIBLE
            action = "retain evidence and investigate exact authorizer geometry"
        else:
            decision = BoundaryTangentPrefixScheduleDecision.EAGER_PREFIX_SCHEDULE_EXECUTION_INVALID
            action = "repair eager-prefix profile execution"
    elif _status(pilot_gate) == "not_evaluated":
        return _ready("pilot", "run the full three-repeat complete-pipeline pilot")
    elif not _passed(pilot_gate):
        if str((pilot_gate or {}).get("failure_domain")) == "resource_gate":
            decision = BoundaryTangentPrefixScheduleDecision.EAGER_PREFIX_SCHEDULE_COMPUTATIONALLY_INFEASIBLE
            action = "retain exact evidence and plan another authorizer-only repair"
        else:
            decision = BoundaryTangentPrefixScheduleDecision.EAGER_PREFIX_SCHEDULE_EXECUTION_INVALID
            action = "repair exact eager-prefix execution or replay"
    else:
        decision = BoundaryTangentPrefixScheduleDecision.EXACT_BOUNDARY_TANGENT_EAGER_PREFIX_SCHEDULE_FEASIBLE
        action = "integrate the frozen eager-prefix profile into a fresh v2 workflow"
    feasible = decision is BoundaryTangentPrefixScheduleDecision.EXACT_BOUNDARY_TANGENT_EAGER_PREFIX_SCHEDULE_FEASIBLE
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


def evaluate_prefix_schedule_workflow(
    *,
    provenance: bool | int | Mapping[str, Any],
    preflight_gate: Mapping[str, Any] | None,
    profile_gate: Mapping[str, Any] | None,
    pilot_gate: Mapping[str, Any] | None,
    require_gate: str,
) -> dict[str, Any]:
    if require_gate not in {"none", "preflight", "profile", "pilot"}:
        raise BoundaryTangentPrefixScheduleGateError(f"unknown required gate: {require_gate}")
    components = {
        "preflight": dict(preflight_gate or not_evaluated_gate("preflight", "not run")),
        "profile": dict(profile_gate or not_evaluated_gate("profile", "not run")),
        "pilot": dict(pilot_gate or not_evaluated_gate("pilot", "not run")),
    }
    required = {
        "none": (),
        "preflight": ("preflight",),
        "profile": ("preflight", "profile"),
        "pilot": ("preflight", "profile", "pilot"),
    }[require_gate]
    required_pass = _passed(provenance) and all(_passed(components[name]) for name in required)
    decision = decide_prefix_schedule_workflow(
        provenance=provenance,
        preflight_gate=components["preflight"],
        profile_gate=components["profile"],
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
        "thresholds": BoundaryTangentPrefixScheduleThresholds().to_dict(),
        **NO_AUTHORIZATION,
        **NO_WORK,
    }


__all__ = [
    "BoundaryTangentPrefixScheduleDecision",
    "BoundaryTangentPrefixScheduleGateError",
    "BoundaryTangentPrefixScheduleThresholds",
    "decide_prefix_schedule_workflow",
    "evaluate_prefix_schedule_pilot",
    "evaluate_prefix_schedule_preflight",
    "evaluate_prefix_schedule_profile",
    "evaluate_prefix_schedule_workflow",
    "not_evaluated_gate",
]
