from __future__ import annotations

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_gate import (
    DECISIONS,
    HARD_STOP_DECISION,
    REQUIRED_GATES,
    STAGE_FLAGS,
    ZERO_AUTHORIZATION_FIELDS,
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
    return {
        stage: evaluate_stage_gate(stage, _metrics(stage))
        for stage in STAGE_FLAGS
    }


def _evidence(
    stable: tuple[int, int, int],
    *,
    power: tuple[int, int, int] = (0, 0, 0),
    localized: tuple[int, int, int] = (1, 1, 1),
) -> dict[str, object]:
    return {
        "quartiles": {
            f"q{quartile}": {
                "cross_role_stable_candidate_count": stable[quartile - 1],
                "power_only_evidence": power[quartile - 1],
                "mechanism_localized": localized[quartile - 1],
            }
            for quartile in (1, 2, 3)
        }
    }


def _decide(
    gates: dict[str, dict[str, object]], evidence: dict[str, object] | None = None
):
    return decide_workflow(
        preflight_gate=gates["preflight"],
        replay_gate=gates["replay"],
        decompose_gate=gates["decompose"],
        adjudicate_gate=gates["adjudicate"],
        evidence=evidence,
    )


def test_exact_stage_flags_are_fail_closed_and_execution_must_be_valid() -> None:
    assert REQUIRED_GATES == (
        "none",
        "preflight",
        "replay",
        "decompose",
        "adjudicate",
    )
    for stage, flags in STAGE_FLAGS.items():
        assert evaluate_stage_gate(stage, _metrics(stage))["passed"] == 1
        failed = _metrics(stage)
        failed[flags[-1]] = 0
        assert evaluate_stage_gate(stage, failed)["passed"] == 0
        failed = _metrics(stage)
        failed["stage_execution_valid"] = 0
        assert evaluate_stage_gate(stage, failed)["passed"] == 0
        failed = _metrics(stage)
        failed[ZERO_AUTHORIZATION_FIELDS[0]] = 1
        assert evaluate_stage_gate(stage, failed)["passed"] == 0


def test_partial_stages_are_nonterminal_and_required_gates_are_cumulative() -> None:
    gates = _passing()
    order = ("preflight", "replay", "decompose", "adjudicate")
    for index, stage in enumerate(order):
        local = dict(gates)
        local[stage] = not_evaluated_gate(stage)
        for later in order[index + 1 :]:
            local[later] = not_evaluated_gate(later)
        decision = _decide(local)
        assert decision["decision"] == f"ready_for_{stage}"
        assert decision["terminal"] == 0
        assert decision_exit_code(decision) == 0

    for required in REQUIRED_GATES:
        workflow = evaluate_required_gate(
            preflight_gate=gates["preflight"],
            replay_gate=gates["replay"],
            decompose_gate=gates["decompose"],
            adjudicate_gate=gates["adjudicate"],
            decision=_decide(gates, _evidence((0, 0, 0))),
            require_gate=required,
        )
        assert workflow["required_gate_pass"] == 1
    with pytest.raises(ValueError, match="unknown required gate"):
        evaluate_required_gate(
            preflight_gate=gates["preflight"],
            replay_gate=gates["replay"],
            decompose_gate=gates["decompose"],
            adjudicate_gate=gates["adjudicate"],
            require_gate="report",
        )


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("preflight", "quartile_direction_adjudication_parent_provenance_invalid"),
        ("replay", "quartile_direction_adjudication_table_replay_invalid"),
        ("decompose", "quartile_direction_adjudication_decomposition_invalid"),
        ("adjudicate", "quartile_direction_adjudication_classification_invalid"),
    ],
)
def test_invalid_stage_decision_precedence(stage: str, expected: str) -> None:
    gates = _passing()
    metrics = _metrics(stage)
    metrics[STAGE_FLAGS[stage][0]] = 0
    gates[stage] = evaluate_stage_gate(stage, metrics)
    decision = _decide(gates, _evidence((1, 1, 1), power=(1, 1, 1)))
    assert decision["decision"] == expected
    assert decision["decision"] in DECISIONS
    assert decision_exit_code(decision) == 1


def test_parent_change_has_first_precedence_even_at_adjudication() -> None:
    gates = _passing()
    metrics = _metrics("adjudicate")
    metrics["parent_unchanged"] = 0
    gates["adjudicate"] = evaluate_stage_gate("adjudicate", metrics)
    decision = _decide(gates, _evidence((1, 1, 1)))
    assert decision["decision"] == (
        "quartile_direction_adjudication_parent_provenance_invalid"
    )


@pytest.mark.parametrize(
    ("evidence", "expected", "exit_code"),
    [
        (
            _evidence((0, 0, 0)),
            HARD_STOP_DECISION,
            2,
        ),
        (
            _evidence((1, 0, 1)),
            "partial_later_quartile_direction_only",
            0,
        ),
        (
            _evidence((1, 1, 1)),
            "later_quartile_failure_mechanism_localized",
            0,
        ),
        (
            _evidence((1, 1, 1), power=(1, 1, 1)),
            "powered_fresh_later_quartile_design_justified",
            0,
        ),
    ],
)
def test_closed_scientific_decisions_and_exit_semantics(
    evidence: dict[str, object], expected: str, exit_code: int
) -> None:
    decision = _decide(_passing(), evidence)
    assert decision["decision"] == expected
    assert decision_exit_code(decision) == exit_code
    for field in ZERO_AUTHORIZATION_FIELDS:
        assert decision[field] == 0
    if expected == HARD_STOP_DECISION:
        assert decision["valid_scientific_negative"] == 1
        assert "do not add paths" in decision["next_action"]


def test_incomplete_or_inconsistent_classification_fails_closed() -> None:
    invalid = _evidence((1, 1, 1), localized=(0, 1, 1))
    decision = _decide(_passing(), invalid)
    assert decision["decision"] == (
        "quartile_direction_adjudication_classification_invalid"
    )
    assert decision_exit_code(decision) == 1
    assert _decide(_passing(), None)["decision"] == (
        "quartile_direction_adjudication_classification_invalid"
    )
