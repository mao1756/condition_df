from __future__ import annotations

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_quartile_gate import (
    DECISIONS,
    PENDING_DECISIONS,
    REQUIRED_GATES,
    STAGE_FLAGS,
    VALID_SCIENTIFIC_NEGATIVES,
    decide_workflow,
    decision_exit_code,
    evaluate_required_gate,
    evaluate_stage_gate,
    not_evaluated_gate,
)


def _metrics(stage: str) -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        **{flag: 1 for flag in STAGE_FLAGS[stage]},
    }


def _passing() -> dict[str, dict[str, object]]:
    return {stage: evaluate_stage_gate(stage, _metrics(stage)) for stage in STAGE_FLAGS}


def _decide(gates: dict[str, dict[str, object]], evidence=None):
    return decide_workflow(
        preflight_gate=gates["preflight"],
        cache_gate=gates["cache"],
        controls_gate=gates["controls"],
        train_gate=gates["train"],
        calibrate_gate=gates["calibrate"],
        select_gate=gates["select"],
        confirm_gate=gates["confirm"],
        evidence=evidence,
    )


def test_every_stage_gate_is_fail_closed() -> None:
    for stage, flags in STAGE_FLAGS.items():
        assert evaluate_stage_gate(stage, _metrics(stage))["passed"] == 1
        failed = _metrics(stage)
        failed[flags[-1]] = 0
        assert evaluate_stage_gate(stage, failed)["passed"] == 0
        execution = _metrics(stage)
        execution["evaluation_status"] = "execution_failed"
        assert evaluate_stage_gate(stage, execution)["passed"] == 0
        assert evaluate_stage_gate(stage, execution)["valid_scientific_negative"] == 0


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (
            "parent_provenance_valid",
            "quartile_specialist_parent_provenance_invalid",
        ),
        (
            "scientific_contract_valid",
            "quartile_specialist_scientific_contract_invalid",
        ),
        (
            "resource_projection_valid",
            "quartile_specialist_path_or_resource_plan_invalid",
        ),
    ],
)
def test_preflight_failure_precedence(flag: str, expected: str) -> None:
    gates = _passing()
    metrics = _metrics("preflight")
    metrics[flag] = 0
    gates["preflight"] = evaluate_stage_gate("preflight", metrics)
    decision = _decide(gates)
    assert decision["decision"] == expected
    assert decision["decision"] in DECISIONS
    assert decision_exit_code(decision) == 1


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("cache", "quartile_specialist_exact_cache_invalid"),
        ("controls", "quartile_specialist_prelabel_controls_failed"),
        ("train", "quartile_specialist_physical_training_invalid"),
        ("calibrate", "quartile_specialist_gain_calibration_invalid"),
        ("select", "quartile_specialist_selection_inference_invalid"),
        ("confirm", "quartile_specialist_confirmation_invalid"),
    ],
)
def test_invalid_stage_maps_to_frozen_precedence(stage: str, expected: str) -> None:
    gates = _passing()
    failed = _metrics(stage)
    failed[STAGE_FLAGS[stage][-1]] = 0
    gates[stage] = evaluate_stage_gate(stage, failed)
    decision = _decide(gates)
    assert decision["decision"] == expected
    assert decision_exit_code(decision) == 1


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("calibrate", "no_training_only_quartile_system"),
        ("select", "no_fresh_quartile_specialist_system"),
        ("confirm", "quartile_specialist_time_local_signal_not_confirmed"),
    ],
)
def test_valid_scientific_negatives_close_and_forbid_later_stages(
    stage: str, expected: str
) -> None:
    gates = _passing()
    metrics = _metrics(stage)
    metrics[STAGE_FLAGS[stage][-1]] = 0
    metrics.update(
        {
            "valid_scientific_negative": 1,
            "stage_execution_valid": 1,
            "inference_valid": 1,
            "scientific_negative_reason": "frozen scientific gate missed",
            "per_quartile_diagnostics": {
                "q2": {"reason": "no_admissible_gain"}
            },
        }
    )
    gates[stage] = evaluate_stage_gate(stage, metrics)
    decision = _decide(gates)
    assert decision["decision"] == expected
    assert decision["decision"] in VALID_SCIENTIFIC_NEGATIVES
    assert decision["valid_scientific_negative"] == 1
    assert decision["terminal"] == 1
    assert decision["per_quartile_diagnostics"]["q2"]["reason"] == (
        "no_admissible_gain"
    )
    assert decision["fresh_selection_authorized"] == 0
    assert decision["confirmation_authorized"] == 0
    assert decision["controller_execution_authorized"] == 0
    assert decision["sampling_authorized"] == 0
    assert decision_exit_code(decision) == 2


def test_valid_negative_requires_valid_execution_and_inference() -> None:
    for field in ("stage_execution_valid", "inference_valid"):
        gates = _passing()
        metrics = _metrics("select")
        metrics[STAGE_FLAGS["select"][-1]] = 0
        metrics["valid_scientific_negative"] = 1
        metrics[field] = 0
        gates["select"] = evaluate_stage_gate("select", metrics)
        decision = _decide(gates)
        assert decision["decision"] == "quartile_specialist_selection_inference_invalid"
        assert decision_exit_code(decision) == 1


def test_pending_stage_authorizes_only_immediate_next_stage() -> None:
    gates = _passing()
    order = tuple(STAGE_FLAGS)
    authorization_field = {
        "cache": "cache_authorized",
        "controls": "controls_authorized",
        "train": "physical_training_authorized",
        "calibrate": "gain_and_rank_authorized",
        "select": "fresh_selection_authorized",
        "confirm": "confirmation_authorized",
    }
    for index, stage in enumerate(order[1:], start=1):
        local = dict(gates)
        local[stage] = not_evaluated_gate(stage)
        for later in order[index + 1 :]:
            local[later] = not_evaluated_gate(later)
        decision = _decide(local)
        assert decision["decision"] == f"ready_for_{stage}"
        assert decision["decision"] in PENDING_DECISIONS
        assert decision[authorization_field[stage]] == 1
        assert decision["terminal"] == 0
        assert decision_exit_code(decision) == 0


def test_success_authorizes_only_controller_control_planning() -> None:
    decision = _decide(_passing())
    assert decision["decision"] == (
        "exact_rb_quartile_specialist_time_local_signal_confirmed"
    )
    assert decision["reverse_controller_control_planning_authorized"] == 1
    assert decision["controller_execution_authorized"] == 0
    assert decision["sampling_authorized"] == 0
    assert decision["reconstruction_authorized"] == 0
    assert decision["confirmation_reuse_authorized"] == 0
    assert decision_exit_code(decision) == 0


def test_required_gates_are_exact_and_cumulative() -> None:
    assert REQUIRED_GATES == (
        "none",
        "preflight",
        "cache",
        "controls",
        "train",
        "calibrate",
        "select",
        "confirm",
    )
    gates = _passing()
    decision = _decide(gates)
    for required in REQUIRED_GATES:
        workflow = evaluate_required_gate(
            preflight_gate=gates["preflight"],
            cache_gate=gates["cache"],
            controls_gate=gates["controls"],
            train_gate=gates["train"],
            calibrate_gate=gates["calibrate"],
            select_gate=gates["select"],
            confirm_gate=gates["confirm"],
            decision=decision,
            require_gate=required,
        )
        assert workflow["required_gate_pass"] == 1
    failed = dict(gates)
    failed["controls"] = not_evaluated_gate("controls")
    assert evaluate_required_gate(
        preflight_gate=failed["preflight"],
        cache_gate=failed["cache"],
        controls_gate=failed["controls"],
        train_gate=failed["train"],
        calibrate_gate=failed["calibrate"],
        select_gate=failed["select"],
        confirm_gate=failed["confirm"],
        decision=decision,
        require_gate="confirm",
    )["required_gate_pass"] == 0
    with pytest.raises(ValueError):
        evaluate_required_gate(
            preflight_gate=gates["preflight"],
            cache_gate=gates["cache"],
            controls_gate=gates["controls"],
            train_gate=gates["train"],
            calibrate_gate=gates["calibrate"],
            select_gate=gates["select"],
            confirm_gate=gates["confirm"],
            decision=decision,
            require_gate="report",
        )

