from __future__ import annotations

from copy import deepcopy

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate_gate import (
    CACHE_FLAGS,
    CONFIRM_FLAGS,
    CONTROLS_FLAGS,
    DECISION_ORDER,
    FINAL_DECISION,
    PREFLIGHT_CONTRACT_FLAGS,
    PREFLIGHT_FLAGS,
    PREFLIGHT_PLAN_FLAGS,
    PREFLIGHT_PROVENANCE_FLAGS,
    REQUIRED_GATES,
    SELECT_FLAGS,
    TRAIN_FLAGS,
    VALID_SCIENTIFIC_NEGATIVES,
    decide_workflow,
    decision_exit_code,
    evaluate_cache_gate,
    evaluate_confirm_gate,
    evaluate_controls_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    evaluate_select_gate,
    evaluate_train_gate,
    not_evaluated_gate,
    validate_resource_metrics,
)


def _metrics(flags: tuple[str, ...], **updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "evaluation_status": "evaluated",
        "stage_execution_valid": 1,
        "inference_valid": 1,
        **{name: 1 for name in flags},
    }
    result.update(updates)
    return result


def _passing_gates() -> dict[str, dict[str, object]]:
    return {
        "preflight_gate": evaluate_preflight_gate(_metrics(PREFLIGHT_FLAGS)),
        "cache_gate": evaluate_cache_gate(_metrics(CACHE_FLAGS)),
        "controls_gate": evaluate_controls_gate(_metrics(CONTROLS_FLAGS)),
        "train_gate": evaluate_train_gate(
            _metrics(TRAIN_FLAGS, physical_training_performed=1)
        ),
        "select_gate": evaluate_select_gate(
            _metrics(SELECT_FLAGS, validation_selection_performed=1)
        ),
        "confirm_gate": evaluate_confirm_gate(
            _metrics(CONFIRM_FLAGS, confirmation_performed=1)
        ),
    }


def test_closed_decision_order_and_required_gates_are_frozen() -> None:
    assert DECISION_ORDER == (
        "frequency1_coordinate_parent_provenance_invalid",
        "frequency1_coordinate_contract_invalid",
        "frequency1_coordinate_path_or_resource_plan_invalid",
        "frequency1_coordinate_exact_cache_invalid",
        "frequency1_coordinate_prelabel_controls_failed",
        "frequency1_coordinate_physical_training_invalid",
        "frequency1_coordinate_validation_inference_invalid",
        "no_frequency1_coordinate_validation_candidate",
        "frequency1_coordinate_fresh_confirmation_invalid",
        "frequency1_coordinate_signal_not_confirmed",
        "exact_rb_frequency1_coordinate_boundary_tangent_time_local_signal_confirmed",
    )
    assert REQUIRED_GATES == (
        "none",
        "preflight",
        "cache",
        "controls",
        "train",
        "select",
        "confirm",
        "terminal",
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (PREFLIGHT_PROVENANCE_FLAGS, "frequency1_coordinate_parent_provenance_invalid"),
        (PREFLIGHT_CONTRACT_FLAGS, "frequency1_coordinate_contract_invalid"),
        (PREFLIGHT_PLAN_FLAGS, "frequency1_coordinate_path_or_resource_plan_invalid"),
    ],
)
def test_preflight_failure_domains_map_by_precedence(
    flags: tuple[str, ...], expected: str
) -> None:
    metrics = _metrics(PREFLIGHT_FLAGS)
    metrics[flags[0]] = 0
    preflight = evaluate_preflight_gate(metrics)
    decision = decide_workflow(
        preflight_gate=preflight,
        cache_gate=None,
        controls_gate=None,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )
    assert decision["decision"] == expected
    assert decision["scientific_evidence_complete"] == 0
    assert decision_exit_code(decision) == 1


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("cache", "frequency1_coordinate_exact_cache_invalid"),
        ("controls", "frequency1_coordinate_prelabel_controls_failed"),
        ("train", "frequency1_coordinate_physical_training_invalid"),
        ("select", "frequency1_coordinate_validation_inference_invalid"),
        ("confirm", "frequency1_coordinate_fresh_confirmation_invalid"),
    ],
)
def test_each_stage_integrity_failure_is_closed(stage: str, expected: str) -> None:
    gates = _passing_gates()
    flags = {
        "cache": CACHE_FLAGS,
        "controls": CONTROLS_FLAGS,
        "train": TRAIN_FLAGS,
        "select": SELECT_FLAGS,
        "confirm": CONFIRM_FLAGS,
    }[stage]
    evaluator = {
        "cache": evaluate_cache_gate,
        "controls": evaluate_controls_gate,
        "train": evaluate_train_gate,
        "select": evaluate_select_gate,
        "confirm": evaluate_confirm_gate,
    }[stage]
    metrics = _metrics(flags)
    metrics[flags[0]] = 0
    gates[f"{stage}_gate"] = evaluator(metrics)
    decision = decide_workflow(**gates)
    assert decision["decision"] == expected
    assert decision_exit_code(decision) == 1
    assert decision["physical_training_performed"] == int(
        stage in {"train", "select", "confirm"}
    )
    assert decision["validation_selection_performed"] == int(
        stage in {"select", "confirm"}
    )
    assert decision["confirmation_performed"] == int(stage == "confirm")


def test_clean_no_candidate_is_terminal_valid_negative_and_never_opens_confirmation() -> None:
    gates = _passing_gates()
    metrics = _metrics(
        SELECT_FLAGS,
        all_228_simultaneous_lower_bounds_positive=0,
        no_validation_candidate=1,
        validation_selection_performed=1,
    )
    gates["select_gate"] = evaluate_select_gate(metrics)
    gates["confirm_gate"] = not_evaluated_gate("confirm")
    decision = decide_workflow(**gates)
    assert decision["decision"] == "no_frequency1_coordinate_validation_candidate"
    assert decision["scientific_evidence_complete"] == 1
    assert decision["valid_scientific_negative"] == 1
    assert decision["confirmation_authorized"] == 0
    assert decision_exit_code(decision) == 0
    workflow = evaluate_required_gate(**gates, require_gate="terminal", decision=decision)
    assert workflow["required_gate_pass"] == 1
    assert evaluate_required_gate(**gates, require_gate="select")["required_gate_pass"] == 0


def test_valid_selection_negative_precedes_any_unopened_confirmation_record() -> None:
    gates = _passing_gates()
    gates["select_gate"] = evaluate_select_gate(
        _metrics(
            SELECT_FLAGS,
            all_228_simultaneous_lower_bounds_positive=0,
            no_validation_candidate=1,
        )
    )
    confirm_metrics = _metrics(CONFIRM_FLAGS, failure_domain="parent_provenance")
    confirm_metrics[CONFIRM_FLAGS[0]] = 0
    bogus_confirm = evaluate_confirm_gate(confirm_metrics)
    # A confirmation record after a clean no-candidate terminal is itself an
    # integrity defect, and global provenance precedence must still fail closed.
    gates["confirm_gate"] = bogus_confirm
    assert decide_workflow(**gates)["decision"] == (
        "frequency1_coordinate_parent_provenance_invalid"
    )


def test_clean_confirmation_negative_and_positive_partition() -> None:
    gates = _passing_gates()
    metrics = _metrics(
        CONFIRM_FLAGS,
        all_228_simultaneous_lower_bounds_positive=0,
        signal_not_confirmed=1,
        confirmation_performed=1,
    )
    gates["confirm_gate"] = evaluate_confirm_gate(metrics)
    negative = decide_workflow(**gates)
    assert negative["decision"] == "frequency1_coordinate_signal_not_confirmed"
    assert negative["decision"] in VALID_SCIENTIFIC_NEGATIVES
    assert negative["controller_control_patch_planning_authorized"] == 0
    assert decision_exit_code(negative) == 0

    gates = _passing_gates()
    positive = decide_workflow(**gates)
    assert positive["decision"] == FINAL_DECISION
    assert positive["controller_control_patch_planning_authorized"] == 1
    assert positive["controller_execution_authorized"] == 0
    assert positive["sampling_authorized"] == 0
    assert evaluate_required_gate(**gates, require_gate="terminal")["required_gate_pass"] == 1


def test_terminal_required_gate_rejects_integrity_failure() -> None:
    gates = _passing_gates()
    bad = _metrics(CACHE_FLAGS)
    bad[CACHE_FLAGS[0]] = 0
    gates["cache_gate"] = evaluate_cache_gate(bad)
    workflow = evaluate_required_gate(**gates, require_gate="terminal")
    assert workflow["required_gate_pass"] == 0
    assert workflow["required_gate_exit_code"] == 1


def test_no_candidate_requires_complete_inference_not_just_a_flag() -> None:
    metrics = _metrics(
        SELECT_FLAGS,
        validation_inference_valid=0,
        all_228_simultaneous_lower_bounds_positive=0,
        no_validation_candidate=1,
    )
    gate = evaluate_select_gate(metrics)
    assert gate["valid_scientific_negative"] == 0
    gates = _passing_gates()
    gates["select_gate"] = gate
    assert decide_workflow(**gates)["decision"] == (
        "frequency1_coordinate_validation_inference_invalid"
    )


def test_resource_thresholds_use_exact_boundaries() -> None:
    metrics = {
        "transition_throughput": 1300.0,
        "peak_cuda_memory_fraction": 0.8,
        "projected_persisted_bytes": 3 * 1024**3,
        "projected_exact_capture_seconds": 160 * 3600,
        "forward_batch_size": 32,
        "target_batch_size": 32,
        "max_t_working_bytes": 64 * 1024**2 - 1,
    }
    assert validate_resource_metrics(metrics)["passed"] == 1
    changed = deepcopy(metrics)
    changed["max_t_working_bytes"] = 64 * 1024**2
    assert validate_resource_metrics(changed)["passed"] == 0
    changed = deepcopy(metrics)
    changed["transition_throughput"] = 1299.999
    assert validate_resource_metrics(changed)["passed"] == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("peak_cuda_memory_fraction", -0.01),
        ("projected_persisted_bytes", -1),
        ("projected_exact_capture_seconds", -1),
        ("forward_batch_size", 0),
        ("target_batch_size", 0),
        ("max_t_working_bytes", -1),
    ],
)
def test_resource_metrics_reject_impossible_negative_values(
    name: str, value: float
) -> None:
    metrics = {
        "transition_throughput": 1300.0,
        "peak_cuda_memory_fraction": 0.8,
        "projected_persisted_bytes": 3 * 1024**3,
        "projected_exact_capture_seconds": 160 * 3600,
        "forward_batch_size": 32,
        "target_batch_size": 32,
        "max_t_working_bytes": 64 * 1024**2 - 1,
    }
    metrics[name] = value
    assert validate_resource_metrics(metrics)["passed"] == 0


def test_input_gate_metrics_are_not_mutated() -> None:
    metrics = _metrics(PREFLIGHT_FLAGS)
    original = deepcopy(metrics)
    evaluate_preflight_gate(metrics)
    assert metrics == original
