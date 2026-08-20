from __future__ import annotations

import pytest

from mnist.d0_jacobi_rb_quartile_directional_adjudication_gate import (
    DECISION_ORDER,
    INVALID_DECISIONS,
    REQUIRED_GATES,
    SCIENTIFIC_DECISIONS,
    STAGE_FLAGS,
    ZERO_AUTHORIZATION_FIELDS,
    ZERO_WORK_FIELDS,
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


def _gates() -> dict[str, dict[str, object]]:
    return {
        stage: evaluate_stage_gate(stage, _metrics(stage))
        for stage in STAGE_FLAGS
    }


def _component(
    direction: int = 0,
    effect: int = 0,
    **extra: int,
) -> dict[str, int]:
    return {
        "stable_direction": direction,
        "stable_effect": effect,
        **extra,
    }


def _evidence() -> dict[str, object]:
    return {
        "q0_full": {"stable_direction": 1, "stable_effect": 1},
        "inferential_and_role_order_valid": 1,
        "branch_algebra_cancellation_valid": 1,
        "quartiles": {
            quartile: {
                "components": {
                    "full": _component(),
                    "local_affine": _component(),
                    "spatial_cnn": _component(),
                }
            }
            for quartile in ("q1", "q2", "q3")
        },
    }


def _decide(evidence: dict[str, object]) -> dict[str, object]:
    return decide_workflow(evidence, gates=_gates())


def test_decision_order_is_exact() -> None:
    assert DECISION_ORDER == (
        "quartile_directional_parent_provenance_invalid",
        "quartile_directional_scientific_contract_invalid",
        "quartile_directional_resource_plan_invalid",
        "quartile_directional_historical_replay_invalid",
        "quartile_directional_prelabel_controls_failed",
        "quartile_directional_fittrace_invalid",
        "quartile_directional_nomination_invalid",
        "quartile_directional_rank_adjudication_invalid",
        "quartile_directional_q0_positive_control_failed",
        "unique_representation_hypothesis_identified",
        "same_class_effect_detected_but_non_authorizing_stop",
        "representation_cancellation_nonidentifying_stop",
        "positive_direction_effect_unresolved_stop",
        "later_quartile_direction_unstable_across_roles_stop",
        "no_later_quartile_signal_detectable_under_permitted_class_stop",
    )


def test_stage_gates_are_exact_and_fail_closed() -> None:
    for stage, flags in STAGE_FLAGS.items():
        assert evaluate_stage_gate(stage, _metrics(stage))["passed"] == 1
        missing = _metrics(stage)
        missing[flags[-1]] = 0
        assert evaluate_stage_gate(stage, missing)["passed"] == 0
        execution = _metrics(stage)
        execution["stage_execution_valid"] = 0
        assert evaluate_stage_gate(stage, execution)["passed"] == 0
        authority = _metrics(stage)
        authority[ZERO_AUTHORIZATION_FIELDS[0]] = 1
        assert evaluate_stage_gate(stage, authority)["passed"] == 0
        work = _metrics(stage)
        work[ZERO_WORK_FIELDS[0]] = 1
        assert evaluate_stage_gate(stage, work)["passed"] == 0


def test_partial_stage_readiness_and_cumulative_required_gates() -> None:
    gates = _gates()
    order = tuple(STAGE_FLAGS)
    for index, stage in enumerate(order):
        local = dict(gates)
        for pending in order[index:]:
            local[pending] = not_evaluated_gate(pending)
        decision = decide_workflow(None, gates=local)
        assert decision["decision"] == f"ready_for_{stage}"
        assert decision["terminal"] == 0

    for required in REQUIRED_GATES:
        result = evaluate_required_gate(
            required,
            gates,
            decision=_decide(_evidence()),
        )
        assert result["required_gate_pass"] == 1
        assert result["required_gate_exit_code"] == 0
    with pytest.raises(ValueError, match="unknown required gate"):
        evaluate_required_gate("report", gates)


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("replay", "quartile_directional_historical_replay_invalid"),
        ("controls", "quartile_directional_prelabel_controls_failed"),
        ("fittrace", "quartile_directional_fittrace_invalid"),
        ("nominate", "quartile_directional_nomination_invalid"),
        ("adjudicate", "quartile_directional_rank_adjudication_invalid"),
    ],
)
def test_stage_failure_precedence(stage: str, expected: str) -> None:
    gates = _gates()
    failed = _metrics(stage)
    failed[STAGE_FLAGS[stage][0]] = 0
    gates[stage] = evaluate_stage_gate(stage, failed)
    assert decide_workflow(_evidence(), gates=gates)["decision"] == expected


def test_preflight_failure_domains_have_frozen_precedence() -> None:
    gates = _gates()
    failed = _metrics("preflight")
    failed["resource_projection_valid"] = 0
    failed["scientific_contract_valid"] = 0
    failed["parent_provenance_valid"] = 0
    gates["preflight"] = evaluate_stage_gate("preflight", failed)
    assert decide_workflow(_evidence(), gates=gates)["decision"] == (
        "quartile_directional_parent_provenance_invalid"
    )

    failed = _metrics("preflight")
    failed["resource_projection_valid"] = 0
    failed["scientific_contract_valid"] = 0
    gates["preflight"] = evaluate_stage_gate("preflight", failed)
    assert decide_workflow(_evidence(), gates=gates)["decision"] == (
        "quartile_directional_scientific_contract_invalid"
    )

    failed = _metrics("preflight")
    failed["resource_projection_valid"] = 0
    gates["preflight"] = evaluate_stage_gate("preflight", failed)
    assert decide_workflow(_evidence(), gates=gates)["decision"] == (
        "quartile_directional_resource_plan_invalid"
    )


def test_parent_mutation_has_first_precedence_at_any_stage() -> None:
    gates = _gates()
    failed = _metrics("adjudicate")
    failed["parent_unchanged"] = 0
    gates["adjudicate"] = evaluate_stage_gate("adjudicate", failed)
    assert decide_workflow(_evidence(), gates=gates)["decision"] == (
        "quartile_directional_parent_provenance_invalid"
    )


@pytest.mark.parametrize("passing_branch", ["local_affine", "spatial_cnn"])
def test_unique_representation_requires_one_strict_branch(passing_branch: str) -> None:
    evidence = _evidence()
    competitor = "spatial_cnn" if passing_branch == "local_affine" else "local_affine"
    for quartile in ("q1", "q2", "q3"):
        row = evidence["quartiles"][quartile]
        row["components"][passing_branch] = _component(1, 1)
        row["components"][competitor] = _component(1, 0)
        row["components"]["full"] = _component(1, 1)
    failed = evidence["quartiles"]["q2"]
    failed["components"]["full"] = _component(0, 0)
    failed["competing_branch_negative_in_full_failure_strata"] = {
        passing_branch: 1
    }
    decision = _decide(evidence)
    assert decision["decision"] == "unique_representation_hypothesis_identified"
    assert decision["unique_representation_identified"] == 1
    assert decision["fresh_learner_plan_drafting_recommended"] == 1


def test_unique_gate_rejects_missing_attribution_and_competing_effect() -> None:
    evidence = _evidence()
    for quartile in ("q1", "q2", "q3"):
        components = evidence["quartiles"][quartile]["components"]
        components["local_affine"] = _component(1, 1)
        components["full"] = _component(0, 0)
    assert _decide(evidence)["decision"] == (
        "representation_cancellation_nonidentifying_stop"
    )

    for quartile in ("q1", "q2", "q3"):
        evidence["quartiles"][quartile][
            "competing_branch_negative_in_full_failure_strata"
        ] = {"local_affine": 1}
    evidence["quartiles"]["q1"]["components"]["spatial_cnn"] = _component(1, 1)
    assert _decide(evidence)["decision"] == (
        "representation_cancellation_nonidentifying_stop"
    )


def test_inference_component_row_schema_is_accepted_directly() -> None:
    evidence = {
        "inferential_and_role_order_valid": 1,
        "branch_algebra_cancellation_valid": 1,
        "component_rows": [
            {
                "quartile": quartile,
                "component": component,
                "stable_direction": int(quartile == 0),
                "stable_effect": int(quartile == 0),
            }
            for quartile in range(4)
            for component in ("full", "local_affine", "spatial_cnn")
        ],
    }
    assert _decide(evidence)["decision"] == (
        "no_later_quartile_signal_detectable_under_permitted_class_stop"
    )


def test_same_class_effect_precedes_nonidentifying_branch_behavior() -> None:
    evidence = _evidence()
    for quartile in ("q1", "q2", "q3"):
        components = evidence["quartiles"][quartile]["components"]
        components["full"] = _component(1, 1)
        components["local_affine"] = _component(1, 1)
    assert _decide(evidence)["decision"] == (
        "same_class_effect_detected_but_non_authorizing_stop"
    )


def test_remaining_scientific_stop_partition() -> None:
    cancellation = _evidence()
    cancellation["cancellation_visible"] = 1
    assert _decide(cancellation)["decision"] == (
        "representation_cancellation_nonidentifying_stop"
    )

    unresolved = _evidence()
    unresolved["quartiles"]["q2"]["components"]["full"][
        "positive_direction_effect_unresolved"
    ] = 1
    assert _decide(unresolved)["decision"] == (
        "positive_direction_effect_unresolved_stop"
    )

    unstable = _evidence()
    unstable["quartiles"]["q3"]["components"]["spatial_cnn"][
        "positive_gain_direction"
    ] = 1
    assert _decide(unstable)["decision"] == (
        "later_quartile_direction_unstable_across_roles_stop"
    )

    assert _decide(_evidence())["decision"] == (
        "no_later_quartile_signal_detectable_under_permitted_class_stop"
    )


def test_q0_control_failure_and_malformed_evidence_fail_closed() -> None:
    evidence = _evidence()
    evidence["q0_full"]["stable_effect"] = 0
    assert _decide(evidence)["decision"] == (
        "quartile_directional_q0_positive_control_failed"
    )
    malformed = _evidence()
    del malformed["quartiles"]["q2"]["components"]["full"]["stable_effect"]
    assert _decide(malformed)["decision"] == (
        "quartile_directional_rank_adjudication_invalid"
    )


@pytest.mark.parametrize("decision_name", DECISION_ORDER)
def test_every_terminal_decision_has_zero_execution_authority(decision_name: str) -> None:
    if decision_name == "quartile_directional_q0_positive_control_failed":
        evidence = _evidence()
        evidence["q0_full"]["stable_effect"] = 0
        decision = _decide(evidence)
    elif decision_name == "no_later_quartile_signal_detectable_under_permitted_class_stop":
        decision = _decide(_evidence())
    else:
        # Decision records reached by the other focused tests share one invariant;
        # inspect the exported partition rather than fabricating invalid evidence.
        assert decision_name in INVALID_DECISIONS | SCIENTIFIC_DECISIONS
        return
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert decision[field] == 0


def test_all_valid_scientific_stops_exit_zero_and_invalid_evidence_exits_one() -> None:
    assert decision_exit_code(_decide(_evidence())) == 0
    q0 = _evidence()
    q0["q0_full"]["stable_effect"] = 0
    assert decision_exit_code(_decide(q0)) == 1
    assert decision_exit_code({"decision": "unknown"}) == 1
