"""Fail-closed gates for the immutable quartile-direction adjudication.

This module is pure: it validates metrics and applies the closed decision
table.  Every gate and decision explicitly denies cache generation, training,
fresh evidence, controller execution, reconstruction, and sampling.
"""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA = "d0-jacobi-rb-boundary-tangent-quartile-direction-adjudication-gate-v1"
REQUIRED_GATES = ("none", "preflight", "replay", "decompose", "adjudicate")

PREFLIGHT_FLAGS = (
    "parent_provenance_valid",
    "parent_registry_valid",
    "parent_terminal_negative_valid",
    "checkpoint_grid_valid",
    "cache_bindings_valid",
    "role_open_history_valid",
    "selection_confirmation_absent",
    "scientific_contract_valid",
    "resource_projection_valid",
    "parent_snapshot_valid",
)
REPLAY_FLAGS = (
    "gain_table_replayed",
    "rank_table_replayed",
    "rank_path_table_replayed",
    "eligibility_replayed",
    "terminal_decision_replayed",
    "no_parent_write",
)
DECOMPOSE_FLAGS = (
    "all_960_candidate_role_jobs_complete",
    "finite_reductions",
    "batch_limit_valid",
    "memory_limit_valid",
    "gain_table_C_P_replayed",
    "rank_direct_reconstruction_valid",
    "q0_positive_control_valid",
    "algebra_controls_valid",
    "no_raw_predictions_persisted",
    "no_new_evidence_opened",
)
ADJUDICATE_FLAGS = (
    "all_mechanism_records_complete",
    "all_thresholds_frozen",
    "path_forecasts_valid",
    "classifications_deterministic",
    "parent_unchanged",
)
DECOMPOSITION_FLAGS = DECOMPOSE_FLAGS
ADJUDICATION_FLAGS = ADJUDICATE_FLAGS
STAGE_FLAGS: dict[str, tuple[str, ...]] = {
    "preflight": PREFLIGHT_FLAGS,
    "replay": REPLAY_FLAGS,
    "decompose": DECOMPOSE_FLAGS,
    "adjudicate": ADJUDICATE_FLAGS,
}

DECISIONS = (
    "quartile_direction_adjudication_parent_provenance_invalid",
    "quartile_direction_adjudication_table_replay_invalid",
    "quartile_direction_adjudication_decomposition_invalid",
    "quartile_direction_adjudication_classification_invalid",
    "no_later_quartile_direction_detectable_under_current_class",
    "partial_later_quartile_direction_only",
    "later_quartile_failure_mechanism_localized",
    "powered_fresh_later_quartile_design_justified",
)
INVALID_DECISIONS = frozenset(DECISIONS[:4])
HARD_STOP_DECISION = DECISIONS[4]
COMPLETED_DIAGNOSTIC_DECISIONS = frozenset(DECISIONS[5:])
PENDING_DECISIONS = (
    "ready_for_preflight",
    "ready_for_replay",
    "ready_for_decompose",
    "ready_for_adjudicate",
    "ready_for_report",
)

ZERO_AUTHORIZATION_FIELDS = (
    "cache_generation_authorized",
    "physical_training_authorized",
    "fresh_selection_authorized",
    "confirmation_authorized",
    "confirmation_reuse_authorized",
    "controller_control_planning_authorized",
    "controller_execution_authorized",
    "reconstruction_authorized",
    "sampling_authorized",
)
ZERO_WORK_FIELDS = (
    "new_transitions_generated",
    "new_physical_labels_opened",
    "optimizer_updates_performed",
    "checkpoints_created_or_modified",
    "fresh_selection_paths_opened",
    "confirmation_paths_opened",
    "controller_or_sampling_work_performed",
    "parent_files_modified",
    "historical_design_evidence_authorizing",
)


class DirectionAdjudicationGateError(ValueError):
    """A gate or decision lay outside the frozen workflow schema."""


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def safety_record() -> dict[str, int]:
    return {
        **{name: 0 for name in ZERO_AUTHORIZATION_FIELDS},
        **{name: 0 for name in ZERO_WORK_FIELDS},
    }


def _status(gate: Mapping[str, Any] | None) -> str:
    return str((gate or {}).get("evaluation_status", "not_evaluated"))


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return _status(gate) == "evaluated" and _one((gate or {}).get("passed"))


def not_evaluated_gate(stage: str, reason: str = "not evaluated") -> dict[str, Any]:
    if stage not in STAGE_FLAGS:
        raise DirectionAdjudicationGateError(f"unknown stage gate: {stage}")
    return {
        "schema": f"{SCHEMA}-{stage}",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "reason": str(reason),
        **safety_record(),
    }


def evaluate_stage_gate(stage: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact stage flag set and reject hidden work/authority."""

    if stage not in STAGE_FLAGS:
        raise DirectionAdjudicationGateError(f"unknown stage gate: {stage}")
    if not isinstance(metrics, Mapping):
        raise DirectionAdjudicationGateError("gate metrics must be a mapping")
    status = str(metrics.get("evaluation_status", "not_evaluated"))
    checks = {
        name: int(_one(metrics.get(name))) for name in STAGE_FLAGS[stage]
    }
    for name in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        checks[name] = int(_zero(metrics.get(name, 0)))
    execution_valid = _one(metrics.get("stage_execution_valid", 1))
    passed = int(status == "evaluated" and execution_valid and all(checks.values()))
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}-{stage}",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": status,
        "passed": passed,
        "stage_execution_valid": int(status == "evaluated" and execution_valid),
        "scientific_evidence_complete": int(
            bool(passed) and stage == "adjudicate"
        ),
        "checks": checks,
        **safety_record(),
    }
    for key in (
        "failure_domain",
        "failure_code",
        "error",
        "per_quartile_diagnostics",
    ):
        if key in metrics:
            result[key] = metrics[key]
    return result


def evaluate_preflight_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("preflight", metrics)


def evaluate_replay_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("replay", metrics)


def evaluate_decompose_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("decompose", metrics)


def evaluate_decomposition_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_decompose_gate(metrics)


def evaluate_adjudicate_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("adjudicate", metrics)


def evaluate_adjudication_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_adjudicate_gate(metrics)


def _decision_record(
    decision: str,
    *,
    complete: bool,
    terminal: bool = True,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if decision not in DECISIONS and decision not in PENDING_DECISIONS:
        raise DirectionAdjudicationGateError(f"unknown closed decision: {decision}")
    hard_stop = decision == HARD_STOP_DECISION
    invalid = decision in INVALID_DECISIONS
    pending = decision in PENDING_DECISIONS
    next_actions = {
        "ready_for_preflight": "verify the immutable quartile-specialist parent",
        "ready_for_replay": "replay the sealed gain and training-rank tables",
        "ready_for_decompose": "reduce the frozen checkpoints on the already-open roles",
        "ready_for_adjudicate": "classify the frozen directional diagnostics",
        "ready_for_report": "write and verify the immutable diagnostic report",
        "quartile_direction_adjudication_parent_provenance_invalid": (
            "repair or restore the immutable parent evidence before continuing"
        ),
        "quartile_direction_adjudication_table_replay_invalid": (
            "repair the exact sealed-table replay before continuing"
        ),
        "quartile_direction_adjudication_decomposition_invalid": (
            "repair the binary64 C/P decomposition before continuing"
        ),
        "quartile_direction_adjudication_classification_invalid": (
            "repair the deterministic mechanism classification before continuing"
        ),
        HARD_STOP_DECISION: (
            "stop the current width-32/later-state representation repair loop; "
            "do not add paths, weaken screens, or rerun the same recipe"
        ),
        "partial_later_quartile_direction_only": (
            "retain the partial historical diagnosis; authorize no fresh execution"
        ),
        "later_quartile_failure_mechanism_localized": (
            "retain the localized historical diagnosis; authorize no fresh execution"
        ),
        "powered_fresh_later_quartile_design_justified": (
            "draft exactly one separate fresh-learner plan targeted at the diagnosed mechanism"
        ),
    }
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}-decision",
        "schema_version": 1,
        "evaluation_status": "not_evaluated" if pending else "evaluated",
        "decision": decision,
        "terminal": int(terminal and not pending),
        "scientific_evidence_complete": int(complete),
        "valid_scientific_negative": int(hard_stop),
        "invalid_evidence": int(invalid),
        "fresh_learner_plan_drafting_recommended": int(
            decision == "powered_fresh_later_quartile_design_justified"
        ),
        "next_action": next_actions[decision],
        **safety_record(),
    }
    if evidence is not None:
        result["per_quartile_diagnostics"] = dict(evidence)
    return result


def _parent_integrity_failure(gates: tuple[Mapping[str, Any] | None, ...]) -> bool:
    for gate in gates:
        if not isinstance(gate, Mapping) or _passed(gate):
            continue
        checks = gate.get("checks") if isinstance(gate.get("checks"), Mapping) else {}
        if (
            gate.get("failure_domain") in {"provenance", "parent_immutability"}
            or "provenance" in str(gate.get("failure_code", ""))
            or "parent_unchanged" in checks
            and not _one(checks.get("parent_unchanged"))
            or "no_parent_write" in checks
            and not _one(checks.get("no_parent_write"))
        ):
            return True
    return False


def _quartile_evidence(evidence: Mapping[str, Any] | None) -> dict[int, dict[str, Any]] | None:
    if not isinstance(evidence, Mapping):
        return None
    nested = evidence.get("per_quartile_diagnostics", evidence.get("quartiles"))
    stable_map = evidence.get("cross_role_stable_candidate_counts")
    power_map = evidence.get("power_only_evidence_by_quartile")
    localized_map = evidence.get("mechanism_localized_by_quartile")
    result: dict[int, dict[str, Any]] = {}
    for quartile in (1, 2, 3):
        keys = (quartile, f"q{quartile}", str(quartile))
        row: Mapping[str, Any] | None = None
        if isinstance(nested, Mapping):
            row = next(
                (nested[key] for key in keys if key in nested),
                None,
            )
        local = dict(row or {})
        if "cross_role_stable_candidate_count" not in local and isinstance(
            stable_map, Mapping
        ):
            local["cross_role_stable_candidate_count"] = next(
                (stable_map[key] for key in keys if key in stable_map), None
            )
        if "cross_role_stable_candidate_count" not in local:
            for name in (
                f"q{quartile}_cross_role_stable_candidate_count",
                f"q{quartile}_stable_candidate_count",
            ):
                if name in evidence:
                    local["cross_role_stable_candidate_count"] = evidence[name]
                    break
        if "power_only_evidence" not in local and isinstance(power_map, Mapping):
            local["power_only_evidence"] = next(
                (power_map[key] for key in keys if key in power_map), None
            )
        if "power_only_evidence" not in local:
            local["power_only_evidence"] = evidence.get(
                f"q{quartile}_power_only_evidence", 0
            )
        if "mechanism_localized" not in local and isinstance(localized_map, Mapping):
            local["mechanism_localized"] = next(
                (localized_map[key] for key in keys if key in localized_map), None
            )
        if "mechanism_localized" not in local and (
            f"q{quartile}_mechanism_localized" in evidence
        ):
            local["mechanism_localized"] = evidence[
                f"q{quartile}_mechanism_localized"
            ]
        stable = local.get(
            "cross_role_stable_candidate_count",
            local.get("stable_candidate_count"),
        )
        if (
            not isinstance(stable, int)
            or isinstance(stable, bool)
            or stable < 0
        ):
            return None
        power = local.get("power_only_evidence", local.get("power_only"))
        if not isinstance(power, (bool, int)) or int(power) not in (0, 1):
            return None
        localized = local.get("mechanism_localized")
        if localized is None:
            flags = local.get("mechanism_flags", local)
            if isinstance(flags, Mapping):
                flag_names = (
                    "conditional_direction_absent",
                    "direction_present_but_role_unstable",
                    "phase_midpoint_cancellation",
                    "gain_transfer_failure",
                    "optimization_time_rotation",
                    "strictly_positive_but_too_small",
                )
                present = [flags.get(name) for name in flag_names if name in flags]
                localized = int(bool(present) and any(_one(value) for value in present))
        if localized is None:
            localized = 0
        if not isinstance(localized, (bool, int)) or int(localized) not in (0, 1):
            return None
        result[quartile] = {
            **local,
            "cross_role_stable_candidate_count": stable,
            "power_only_evidence": int(power),
            "mechanism_localized": int(localized),
        }
    return result


def decide_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    replay_gate: Mapping[str, Any] | None,
    decompose_gate: Mapping[str, Any] | None = None,
    adjudicate_gate: Mapping[str, Any] | None = None,
    decomposition_gate: Mapping[str, Any] | None = None,
    adjudication_gate: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen precedence and later-quartile decision partition."""

    if decompose_gate is None:
        decompose_gate = decomposition_gate
    if adjudicate_gate is None:
        adjudicate_gate = adjudication_gate
    gates = (preflight_gate, replay_gate, decompose_gate, adjudicate_gate)
    if _parent_integrity_failure(gates):
        return _decision_record(
            "quartile_direction_adjudication_parent_provenance_invalid",
            complete=False,
        )
    if _status(preflight_gate) == "not_evaluated":
        return _decision_record("ready_for_preflight", complete=False, terminal=False)
    if not _passed(preflight_gate):
        return _decision_record(
            "quartile_direction_adjudication_parent_provenance_invalid",
            complete=False,
        )
    if _status(replay_gate) == "not_evaluated":
        return _decision_record("ready_for_replay", complete=False, terminal=False)
    if not _passed(replay_gate):
        return _decision_record(
            "quartile_direction_adjudication_table_replay_invalid",
            complete=False,
        )
    if _status(decompose_gate) == "not_evaluated":
        return _decision_record("ready_for_decompose", complete=False, terminal=False)
    if not _passed(decompose_gate):
        return _decision_record(
            "quartile_direction_adjudication_decomposition_invalid",
            complete=False,
        )
    if _status(adjudicate_gate) == "not_evaluated":
        return _decision_record("ready_for_adjudicate", complete=False, terminal=False)
    if not _passed(adjudicate_gate):
        return _decision_record(
            "quartile_direction_adjudication_classification_invalid",
            complete=False,
        )
    quartiles = _quartile_evidence(evidence)
    if quartiles is None:
        return _decision_record(
            "quartile_direction_adjudication_classification_invalid",
            complete=False,
        )
    stable_quartiles = {
        quartile
        for quartile, row in quartiles.items()
        if int(row["cross_role_stable_candidate_count"]) > 0
    }
    if not stable_quartiles:
        decision = HARD_STOP_DECISION
    elif stable_quartiles != {1, 2, 3}:
        decision = "partial_later_quartile_direction_only"
    elif all(int(row["power_only_evidence"]) == 1 for row in quartiles.values()):
        decision = "powered_fresh_later_quartile_design_justified"
    elif all(int(row["mechanism_localized"]) == 1 for row in quartiles.values()):
        decision = "later_quartile_failure_mechanism_localized"
    else:
        decision = "quartile_direction_adjudication_classification_invalid"
    return _decision_record(
        decision,
        complete=decision not in INVALID_DECISIONS,
        evidence={f"q{key}": value for key, value in quartiles.items()},
    )


def evaluate_required_gate(
    *,
    preflight_gate: Mapping[str, Any] | None,
    replay_gate: Mapping[str, Any] | None,
    decompose_gate: Mapping[str, Any] | None = None,
    adjudicate_gate: Mapping[str, Any] | None = None,
    decomposition_gate: Mapping[str, Any] | None = None,
    adjudication_gate: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    require_gate: str,
) -> dict[str, Any]:
    """Evaluate a cumulative gate while allowing valid partial-stage runs."""

    if require_gate not in REQUIRED_GATES:
        raise DirectionAdjudicationGateError(
            f"unknown required gate: {require_gate}"
        )
    if decompose_gate is None:
        decompose_gate = decomposition_gate
    if adjudicate_gate is None:
        adjudicate_gate = adjudication_gate
    gates = {
        "preflight": preflight_gate,
        "replay": replay_gate,
        "decompose": decompose_gate,
        "adjudicate": adjudicate_gate,
    }
    if require_gate == "none":
        passed = True
    else:
        index = REQUIRED_GATES.index(require_gate)
        passed = all(
            _passed(gates[name]) for name in REQUIRED_GATES[1 : index + 1]
        )
    active_decision = dict(
        decision
        or decide_workflow(
            preflight_gate=preflight_gate,
            replay_gate=replay_gate,
            decompose_gate=decompose_gate,
            adjudicate_gate=adjudicate_gate,
        )
    )
    return {
        "schema": f"{SCHEMA}-workflow",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "require_gate": require_gate,
        "required_gate_pass": int(passed),
        "required_gate_exit_code": 0 if passed else 1,
        "decision": active_decision,
        **safety_record(),
    }


def decision_exit_code(decision: Mapping[str, Any]) -> int:
    name = str(decision.get("decision", ""))
    if name == HARD_STOP_DECISION:
        return 2
    if name in INVALID_DECISIONS or name not in set(DECISIONS) | set(PENDING_DECISIONS):
        return 1
    return 0


__all__ = [
    "ADJUDICATE_FLAGS",
    "ADJUDICATION_FLAGS",
    "COMPLETED_DIAGNOSTIC_DECISIONS",
    "DECISIONS",
    "DECOMPOSE_FLAGS",
    "DECOMPOSITION_FLAGS",
    "DirectionAdjudicationGateError",
    "HARD_STOP_DECISION",
    "INVALID_DECISIONS",
    "PENDING_DECISIONS",
    "PREFLIGHT_FLAGS",
    "REPLAY_FLAGS",
    "REQUIRED_GATES",
    "SCHEMA",
    "STAGE_FLAGS",
    "ZERO_AUTHORIZATION_FIELDS",
    "ZERO_WORK_FIELDS",
    "decide_workflow",
    "decision_exit_code",
    "evaluate_adjudicate_gate",
    "evaluate_adjudication_gate",
    "evaluate_decompose_gate",
    "evaluate_decomposition_gate",
    "evaluate_preflight_gate",
    "evaluate_replay_gate",
    "evaluate_required_gate",
    "evaluate_stage_gate",
    "not_evaluated_gate",
    "safety_record",
]
