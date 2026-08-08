"""Fail-closed gates and decisions for the quartile-specialist workflow."""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA = "d0-jacobi-rb-boundary-tangent-quartile-gate-v1"
REQUIRED_GATES = (
    "none",
    "preflight",
    "cache",
    "controls",
    "train",
    "calibrate",
    "select",
    "confirm",
)

PREFLIGHT_FLAGS = (
    "parent_provenance_valid",
    "scientific_contract_valid",
    "path_plan_valid",
    "seed_plan_valid",
    "cohort_plan_valid",
    "role_firewall_valid",
    "bootstrap_count_plans_sealed",
    "exact_backend_seam_valid",
    "resource_projection_valid",
)
CACHE_FLAGS = (
    "fit_cache_valid",
    "gain_cache_valid",
    "rank_cache_valid",
    "path_row_transition_counts_valid",
    "cache_hashes_valid",
    "cache_role_isolation_valid",
    "all_labels_unopened",
    "selection_confirmation_evidence_absent",
)
CONTROLS_FLAGS = (
    "source_backend_seam_valid",
    "zero_initialization_valid",
    "quartile_dispatch_valid",
    "synthetic_teacher_valid",
    "gain_algebra_valid",
    "exact_model_null_valid",
    "input_firewall_valid",
    "resource_guard_valid",
    "physical_labels_opened_zero",
)
TRAIN_FLAGS = (
    "fit_labels_only",
    "target_scales_training_only",
    "twelve_trajectories_complete",
    "four_hundred_ninety_two_checkpoints_complete",
    "finite_outputs",
    "batch_limit_valid",
    "memory_limit_valid",
    "downstream_labels_unopened",
)
CALIBRATE_FLAGS = (
    "gain_label_open_order_valid",
    "gain_table_valid",
    "gain_calibration_seal_valid",
    "rank_label_open_order_valid",
    "rank_rule_valid",
    "selected_system_complete",
    "selected_system_sealed",
)
SELECT_FLAGS = (
    "selected_system_seal_valid",
    "fresh_selection_paths_valid",
    "one_system_only",
    "six_family_inference_valid",
    "all_six_lower_bounds_positive",
    "all_local_screens_pass",
    "raw_selection_cache_absent",
)
CONFIRM_FLAGS = (
    "selected_system_unchanged",
    "untouched_confirmation_paths_valid",
    "confirmation_open_once",
    "six_family_inference_valid",
    "all_six_lower_bounds_positive",
    "all_local_screens_pass",
    "no_fitting_or_mutation",
    "raw_confirmation_cache_absent",
)

STAGE_FLAGS: dict[str, tuple[str, ...]] = {
    "preflight": PREFLIGHT_FLAGS,
    "cache": CACHE_FLAGS,
    "controls": CONTROLS_FLAGS,
    "train": TRAIN_FLAGS,
    "calibrate": CALIBRATE_FLAGS,
    "select": SELECT_FLAGS,
    "confirm": CONFIRM_FLAGS,
}

DECISIONS = (
    "quartile_specialist_parent_provenance_invalid",
    "quartile_specialist_scientific_contract_invalid",
    "quartile_specialist_path_or_resource_plan_invalid",
    "quartile_specialist_exact_cache_invalid",
    "quartile_specialist_prelabel_controls_failed",
    "quartile_specialist_physical_training_invalid",
    "quartile_specialist_gain_calibration_invalid",
    "no_training_only_quartile_system",
    "quartile_specialist_selection_inference_invalid",
    "no_fresh_quartile_specialist_system",
    "quartile_specialist_confirmation_invalid",
    "quartile_specialist_time_local_signal_not_confirmed",
    "exact_rb_quartile_specialist_time_local_signal_confirmed",
)
VALID_SCIENTIFIC_NEGATIVES = frozenset(
    {
        "no_training_only_quartile_system",
        "no_fresh_quartile_specialist_system",
        "quartile_specialist_time_local_signal_not_confirmed",
    }
)
PENDING_DECISIONS = (
    "ready_for_cache",
    "ready_for_controls",
    "ready_for_train",
    "ready_for_calibrate",
    "ready_for_select",
    "ready_for_confirm",
)


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _not_evaluated(gate: Mapping[str, Any] | None) -> bool:
    return not isinstance(gate, Mapping) or gate.get(
        "evaluation_status", "not_evaluated"
    ) == "not_evaluated"


def _valid_negative(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 0
        and int(gate.get("valid_scientific_negative", 0)) == 1
        and int(gate.get("stage_execution_valid", 1)) == 1
        and int(gate.get("inference_valid", 1)) == 1
    )


def not_evaluated_gate(name: str, reason: str = "not evaluated") -> dict[str, Any]:
    if name not in STAGE_FLAGS:
        raise ValueError(f"unknown gate name: {name}")
    return {
        "schema": f"{SCHEMA}-{name}",
        "schema_version": 1,
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "valid_scientific_negative": 0,
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
    valid_negative = int(
        status == "evaluated"
        and not passed
        and name in {"calibrate", "select", "confirm"}
        and int(metrics.get("valid_scientific_negative", 0)) == 1
        and int(metrics.get("stage_execution_valid", 1)) == 1
        and int(metrics.get("inference_valid", 1)) == 1
    )
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}-{name}",
        "schema_version": 1,
        "evaluation_status": status,
        "passed": passed,
        "valid_scientific_negative": valid_negative,
        "stage_execution_valid": int(metrics.get("stage_execution_valid", 1)),
        "inference_valid": int(metrics.get("inference_valid", 1)),
        "checks": checks,
        "confirmation_authorized": int(name == "select" and bool(passed)),
        "reverse_controller_control_planning_authorized": int(
            name == "confirm" and bool(passed)
        ),
        "controller_execution_authorized": 0,
        "sampling_authorized": 0,
        "reconstruction_authorized": 0,
    }
    for key in (
        "failure_domain",
        "failure_code",
        "error",
        "scientific_negative_reason",
        "per_quartile_diagnostics",
    ):
        if key in metrics:
            result[key] = metrics[key]
    return result


def evaluate_stage_gate(stage: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if stage not in STAGE_FLAGS:
        raise ValueError(f"unknown stage gate: {stage}")
    return _evaluate_flags(stage, metrics, STAGE_FLAGS[stage])


def evaluate_preflight_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("preflight", metrics)


def evaluate_cache_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("cache", metrics)


def evaluate_controls_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("controls", metrics)


def evaluate_train_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("train", metrics)


def evaluate_calibrate_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("calibrate", metrics)


def evaluate_select_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("select", metrics)


def evaluate_confirm_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("confirm", metrics)


def _decision_record(
    decision: str,
    *,
    complete: bool,
    valid_scientific_negative: bool = False,
    per_quartile_diagnostics: Any = None,
) -> dict[str, Any]:
    success = decision == "exact_rb_quartile_specialist_time_local_signal_confirmed"
    pending = decision in PENDING_DECISIONS
    next_actions = {
        "ready_for_cache": "generate only the frozen fit/gain/rank caches",
        "ready_for_controls": "run all prelabel controls",
        "ready_for_train": "open only fit labels and train the 12 trajectories",
        "ready_for_calibrate": (
            "open gain labels, seal gains, then open training-rank labels"
        ),
        "ready_for_select": "audit the one sealed system on fresh selection paths",
        "ready_for_confirm": "open the one untouched confirmation audit",
        "no_training_only_quartile_system": (
            "close the run; do not open fresh selection paths"
        ),
        "no_fresh_quartile_specialist_system": (
            "close the run; do not open confirmation paths"
        ),
        "quartile_specialist_time_local_signal_not_confirmed": (
            "close the run and retain the confirmation as a valid negative"
        ),
        "exact_rb_quartile_specialist_time_local_signal_confirmed": (
            "plan a separate reverse-controller control milestone"
        ),
    }
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}-decision",
        "schema_version": 1,
        "evaluation_status": "not_evaluated" if pending else "evaluated",
        "decision": decision,
        "terminal": int(not pending),
        "scientific_evidence_complete": int(complete),
        "valid_scientific_negative": int(valid_scientific_negative),
        "reverse_controller_control_planning_authorized": int(success),
        "controller_execution_authorized": 0,
        "sampling_authorized": 0,
        "reconstruction_authorized": 0,
        "confirmation_reuse_authorized": 0,
        "cache_authorized": int(decision == "ready_for_cache"),
        "controls_authorized": int(decision == "ready_for_controls"),
        "physical_training_authorized": int(decision == "ready_for_train"),
        "gain_and_rank_authorized": int(decision == "ready_for_calibrate"),
        "fresh_selection_authorized": int(decision == "ready_for_select"),
        "confirmation_authorized": int(decision == "ready_for_confirm"),
        "next_action": next_actions.get(
            decision, "repair or interpret the named closed gate"
        ),
    }
    if per_quartile_diagnostics is not None:
        result["per_quartile_diagnostics"] = per_quartile_diagnostics
    return result


def _preflight_failure(gate: Mapping[str, Any]) -> str:
    checks = gate.get("checks") if isinstance(gate.get("checks"), Mapping) else {}
    code = str(gate.get("failure_code", ""))
    domain = str(gate.get("failure_domain", ""))
    if (
        int(checks.get("parent_provenance_valid", 0)) == 0
        or "provenance" in code
        or domain == "provenance"
    ):
        return "quartile_specialist_parent_provenance_invalid"
    if (
        int(checks.get("scientific_contract_valid", 0)) == 0
        or "scientific_contract" in code
        or domain == "scientific_contract"
    ):
        return "quartile_specialist_scientific_contract_invalid"
    return "quartile_specialist_path_or_resource_plan_invalid"


def decide_workflow(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    controls_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    calibrate_gate: Mapping[str, Any] | None,
    select_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen precedence and distinguish valid scientific negatives."""

    diagnostics = dict(evidence or {}).get("per_quartile_diagnostics")
    all_gates = (
        preflight_gate,
        cache_gate,
        controls_gate,
        train_gate,
        calibrate_gate,
        select_gate,
        confirm_gate,
    )
    if any(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") in {"evaluated", "execution_failed"}
        and not _passed(gate)
        and (
            gate.get("failure_domain") == "provenance"
            or "provenance" in str(gate.get("failure_code", ""))
        )
        for gate in all_gates
    ):
        return _decision_record(
            "quartile_specialist_parent_provenance_invalid", complete=False
        )
    if not _passed(preflight_gate):
        if _not_evaluated(preflight_gate):
            decision = "quartile_specialist_parent_provenance_invalid"
        else:
            decision = _preflight_failure(dict(preflight_gate or {}))
        return _decision_record(decision, complete=False)
    stages = (
        ("cache", cache_gate, "quartile_specialist_exact_cache_invalid"),
        ("controls", controls_gate, "quartile_specialist_prelabel_controls_failed"),
        ("train", train_gate, "quartile_specialist_physical_training_invalid"),
        (
            "calibrate",
            calibrate_gate,
            "quartile_specialist_gain_calibration_invalid",
        ),
        (
            "select",
            select_gate,
            "quartile_specialist_selection_inference_invalid",
        ),
        ("confirm", confirm_gate, "quartile_specialist_confirmation_invalid"),
    )
    ready = {
        "cache": "ready_for_cache",
        "controls": "ready_for_controls",
        "train": "ready_for_train",
        "calibrate": "ready_for_calibrate",
        "select": "ready_for_select",
        "confirm": "ready_for_confirm",
    }
    negatives = {
        "calibrate": "no_training_only_quartile_system",
        "select": "no_fresh_quartile_specialist_system",
        "confirm": "quartile_specialist_time_local_signal_not_confirmed",
    }
    for stage, gate, invalid_decision in stages:
        if _not_evaluated(gate):
            return _decision_record(ready[stage], complete=False)
        if _passed(gate):
            continue
        if stage in negatives and _valid_negative(gate):
            return _decision_record(
                negatives[stage],
                complete=True,
                valid_scientific_negative=True,
                per_quartile_diagnostics=(
                    gate.get("per_quartile_diagnostics", diagnostics)
                    if isinstance(gate, Mapping)
                    else diagnostics
                ),
            )
        return _decision_record(
            invalid_decision,
            complete=False,
            per_quartile_diagnostics=(
                gate.get("per_quartile_diagnostics", diagnostics)
                if isinstance(gate, Mapping)
                else diagnostics
            ),
        )
    return _decision_record(
        "exact_rb_quartile_specialist_time_local_signal_confirmed",
        complete=True,
        per_quartile_diagnostics=diagnostics,
    )


def evaluate_required_gate(
    *,
    preflight_gate: Mapping[str, Any] | None,
    cache_gate: Mapping[str, Any] | None,
    controls_gate: Mapping[str, Any] | None,
    train_gate: Mapping[str, Any] | None,
    calibrate_gate: Mapping[str, Any] | None,
    select_gate: Mapping[str, Any] | None,
    confirm_gate: Mapping[str, Any] | None,
    decision: Mapping[str, Any],
    require_gate: str,
) -> dict[str, Any]:
    if require_gate not in REQUIRED_GATES:
        raise ValueError(f"unknown required gate: {require_gate}")
    gates = {
        "preflight": preflight_gate,
        "cache": cache_gate,
        "controls": controls_gate,
        "train": train_gate,
        "calibrate": calibrate_gate,
        "select": select_gate,
        "confirm": confirm_gate,
    }
    if require_gate == "none":
        passed = True
    else:
        index = REQUIRED_GATES.index(require_gate)
        passed = all(_passed(gates[name]) for name in REQUIRED_GATES[1 : index + 1])
    return {
        "schema": f"{SCHEMA}-workflow",
        "schema_version": 1,
        "require_gate": require_gate,
        "required_gate_pass": int(passed),
        "decision": dict(decision),
    }


def decision_exit_code(decision: Mapping[str, Any]) -> int:
    name = str(decision.get("decision", ""))
    if name in PENDING_DECISIONS or name == (
        "exact_rb_quartile_specialist_time_local_signal_confirmed"
    ):
        return 0
    if name in VALID_SCIENTIFIC_NEGATIVES:
        return 2
    return 1


__all__ = [
    "CACHE_FLAGS",
    "CALIBRATE_FLAGS",
    "CONFIRM_FLAGS",
    "CONTROLS_FLAGS",
    "DECISIONS",
    "PENDING_DECISIONS",
    "PREFLIGHT_FLAGS",
    "REQUIRED_GATES",
    "SCHEMA",
    "SELECT_FLAGS",
    "STAGE_FLAGS",
    "TRAIN_FLAGS",
    "VALID_SCIENTIFIC_NEGATIVES",
    "decide_workflow",
    "decision_exit_code",
    "evaluate_cache_gate",
    "evaluate_calibrate_gate",
    "evaluate_confirm_gate",
    "evaluate_controls_gate",
    "evaluate_preflight_gate",
    "evaluate_required_gate",
    "evaluate_select_gate",
    "evaluate_stage_gate",
    "evaluate_train_gate",
    "not_evaluated_gate",
]
