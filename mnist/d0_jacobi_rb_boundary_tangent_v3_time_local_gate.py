"""Fail-closed gates for immutable v3 time-local signal adjudication."""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA = "d0-jacobi-rb-boundary-tangent-v3-time-local-gate-v1"
REQUIRED_GATES = ("none", "preflight", "replay", "decompose")

PREFLIGHT_FLAGS = (
    "memory_parent_valid",
    "coarse_witness_parent_valid",
    "bayes_power_parent_valid",
    "selection_seal_valid",
    "checkpoint_bindings_valid",
    "confirmation_namespace_unopened",
    "parent_immutability_valid",
)
REPLAY_FLAGS = (
    "candidate_table_shape_valid",
    "candidate_grid_valid",
    "critical_value_reproduced",
    "zero_eligible_candidates_reproduced",
    "logical_update_zero_reproduced",
    "partial_discovery_census_reproduced",
    "nominee_tuples_reproduced",
    "coarse_witness_reproduced",
)
DECOMPOSITION_FLAGS = (
    "nominee_checkpoint_hashes_valid",
    "input_label_join_valid",
    "sealed_validation_risk_replay_valid",
    "batch_limit_valid",
    "finite_outputs",
    "quadratic_identity_valid",
    "resource_limit_valid",
    "confirmation_firewall_valid",
)

DECISIONS = (
    "control_provenance_invalid",
    "sealed_selection_replay_invalid",
    "coarse_witness_replay_invalid",
    "quadratic_risk_decomposition_invalid",
    "no_learned_time_local_signal",
    "multiplicity_only_underpowered",
    "mixed_time_local_signal_inconclusive",
    "exact_rb_high_reverse_time_only_signal",
)


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def not_evaluated_gate(name: str, reason: str = "not evaluated") -> dict[str, Any]:
    return {
        "schema": SCHEMA + f"-{name}",
        "schema_version": 1,
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "reason": str(reason),
    }


def _evaluate_flags(
    name: str,
    metrics: Mapping[str, Any],
    flags: tuple[str, ...],
) -> dict[str, Any]:
    status = str(metrics.get("evaluation_status", "not_evaluated"))
    checks = {flag: int(metrics.get(flag, 0)) for flag in flags}
    passed = int(status == "evaluated" and all(checks.values()))
    result: dict[str, Any] = {
        "schema": SCHEMA + f"-{name}",
        "schema_version": 1,
        "evaluation_status": status,
        "passed": passed,
        "checks": checks,
    }
    for key in ("failure_domain", "failure_code", "error"):
        if key in metrics:
            result[key] = metrics[key]
    return result


def evaluate_preflight_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_flags("preflight", metrics, PREFLIGHT_FLAGS)


def evaluate_replay_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_flags("replay", metrics, REPLAY_FLAGS)


def evaluate_decomposition_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_flags("decomposition", metrics, DECOMPOSITION_FLAGS)


def decide_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    replay_gate: Mapping[str, Any] | None,
    decomposition_gate: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return one closed decision without changing the historical selection."""

    if not _passed(preflight_gate):
        decision = "control_provenance_invalid"
        complete = 0
    elif not isinstance(replay_gate, Mapping) or replay_gate.get(
        "evaluation_status"
    ) == "not_evaluated":
        return {
            "schema": SCHEMA + "-decision",
            "schema_version": 1,
            "evaluation_status": "not_evaluated",
            "decision": "ready_for_replay",
            "scientific_evidence_complete": 0,
            "historical_decision_changed": 0,
            "confirmation_authorized": 0,
            "controller_execution_authorized": 0,
            "sampling_authorized": 0,
            "next_action": "run the immutable sealed-selection replay",
        }
    elif not _passed(replay_gate):
        failure = str((replay_gate or {}).get("failure_code", ""))
        decision = (
            "coarse_witness_replay_invalid"
            if "coarse_witness" in failure
            else "sealed_selection_replay_invalid"
        )
        complete = 0
    elif not isinstance(decomposition_gate, Mapping) or decomposition_gate.get(
        "evaluation_status"
    ) == "not_evaluated":
        return {
            "schema": SCHEMA + "-decision",
            "schema_version": 1,
            "evaluation_status": "not_evaluated",
            "decision": "ready_for_decompose",
            "scientific_evidence_complete": 0,
            "historical_decision_changed": 0,
            "confirmation_authorized": 0,
            "controller_execution_authorized": 0,
            "sampling_authorized": 0,
            "next_action": "run the frozen-checkpoint quadratic decomposition",
        }
    elif not _passed(decomposition_gate):
        decision = "quadratic_risk_decomposition_invalid"
        complete = 0
    else:
        row = dict(evidence or {})
        q0_bounds = tuple(float(value) for value in row.get("q0_nominee_lower_bounds", ()))
        q0_counts = tuple(int(value) for value in row.get("q0_positive_fine_counts", ()))
        high_only = bool(
            len(q0_bounds) == 3
            and all(value > 0.0 for value in q0_bounds)
            and len(q0_counts) == 3
            and all(value >= 51 for value in q0_counts)
            and int(row.get("later_adjusted_positive_count", -1)) == 0
            and int(row.get("all_point_positive_candidate_count", -1)) == 0
            and float(row.get("coarse_witness_overall_energy", 0.0)) > 0.0
        )
        if high_only:
            decision = "exact_rb_high_reverse_time_only_signal"
        else:
            positive_points = int(row.get("later_positive_point_count", 0))
            if not q0_bounds or not any(value > 0.0 for value in q0_bounds):
                decision = "no_learned_time_local_signal"
            elif (
                int(row.get("positive_adjusted_component_count", -1)) == 0
                and int(row.get("all_point_positive_candidate_count", 0)) > 0
            ):
                decision = "multiplicity_only_underpowered"
            elif positive_points > 0:
                decision = "mixed_time_local_signal_inconclusive"
            else:
                decision = "no_learned_time_local_signal"
        complete = 1

    return {
        "schema": SCHEMA + "-decision",
        "schema_version": 1,
        "evaluation_status": "evaluated" if complete else "not_evaluated",
        "decision": decision,
        "scientific_evidence_complete": complete,
        "historical_decision_changed": 0,
        "confirmation_authorized": 0,
        "controller_execution_authorized": 0,
        "sampling_authorized": 0,
        "next_action": (
            "plan a fresh quartile-specialized exact-RB learner"
            if decision == "exact_rb_high_reverse_time_only_signal"
            else "repair or interpret the named failed evidence gate"
        ),
    }


def evaluate_required_gate(
    *,
    preflight_gate: Mapping[str, Any] | None,
    replay_gate: Mapping[str, Any] | None,
    decomposition_gate: Mapping[str, Any] | None,
    decision: Mapping[str, Any],
    require_gate: str,
) -> dict[str, Any]:
    if require_gate not in REQUIRED_GATES:
        raise ValueError(f"unknown required gate: {require_gate}")
    selected = {
        "none": True,
        "preflight": _passed(preflight_gate),
        "replay": _passed(preflight_gate) and _passed(replay_gate),
        "decompose": (
            _passed(preflight_gate)
            and _passed(replay_gate)
            and _passed(decomposition_gate)
        ),
    }[require_gate]
    return {
        "schema": SCHEMA + "-workflow",
        "schema_version": 1,
        "require_gate": require_gate,
        "required_gate_pass": int(selected),
        "decision": dict(decision),
    }


__all__ = [
    "DECISIONS",
    "DECOMPOSITION_FLAGS",
    "PREFLIGHT_FLAGS",
    "REPLAY_FLAGS",
    "REQUIRED_GATES",
    "decide_workflow",
    "evaluate_decomposition_gate",
    "evaluate_preflight_gate",
    "evaluate_replay_gate",
    "evaluate_required_gate",
    "not_evaluated_gate",
]
