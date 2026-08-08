from __future__ import annotations

import pytest

from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local_gate import (
    DECISIONS,
    DECOMPOSITION_FLAGS,
    PREFLIGHT_FLAGS,
    REPLAY_FLAGS,
    decide_workflow,
    evaluate_decomposition_gate,
    evaluate_preflight_gate,
    evaluate_replay_gate,
    evaluate_required_gate,
)


def _metrics(flags: tuple[str, ...], *, status: str = "evaluated") -> dict[str, object]:
    return {"evaluation_status": status, **{name: 1 for name in flags}}


def _passing() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        evaluate_preflight_gate(_metrics(PREFLIGHT_FLAGS)),
        evaluate_replay_gate(_metrics(REPLAY_FLAGS)),
        evaluate_decomposition_gate(_metrics(DECOMPOSITION_FLAGS)),
    )


def test_each_gate_is_fail_closed() -> None:
    for evaluator, flags in (
        (evaluate_preflight_gate, PREFLIGHT_FLAGS),
        (evaluate_replay_gate, REPLAY_FLAGS),
        (evaluate_decomposition_gate, DECOMPOSITION_FLAGS),
    ):
        assert evaluator(_metrics(flags))["passed"] == 1
        failed = _metrics(flags)
        failed[flags[-1]] = 0
        assert evaluator(failed)["passed"] == 0
        assert evaluator(_metrics(flags, status="execution_failed"))["passed"] == 0


def test_exact_high_reverse_time_decision_and_authority_are_restricted() -> None:
    preflight, replay, decomposition = _passing()
    decision = decide_workflow(
        preflight_gate=preflight,
        replay_gate=replay,
        decomposition_gate=decomposition,
        evidence={
            "q0_nominee_lower_bounds": [0.1, 0.2, 0.3],
            "q0_positive_fine_counts": [51, 55, 56],
            "later_adjusted_positive_count": 0,
            "all_point_positive_candidate_count": 0,
            "coarse_witness_overall_energy": 0.001,
        },
    )
    assert decision["decision"] == "exact_rb_high_reverse_time_only_signal"
    assert decision["scientific_evidence_complete"] == 1
    assert decision["confirmation_authorized"] == 0
    assert decision["controller_execution_authorized"] == 0
    assert decision["sampling_authorized"] == 0
    contaminated = decide_workflow(
        preflight_gate=preflight,
        replay_gate=replay,
        decomposition_gate=decomposition,
        evidence={
            "q0_nominee_lower_bounds": [0.1, 0.2, 0.3],
            "q0_positive_fine_counts": [51, 55, 56],
            "later_adjusted_positive_count": 1,
            "all_point_positive_candidate_count": 0,
            "coarse_witness_overall_energy": 0.001,
            "later_positive_point_count": 1,
        },
    )
    assert contaminated["decision"] == "mixed_time_local_signal_inconclusive"


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda p, r, d: ({**p, "passed": 0}, r, d), "control_provenance_invalid"),
        (lambda p, r, d: (p, {**r, "passed": 0}, d), "sealed_selection_replay_invalid"),
        (
            lambda p, r, d: (
                p,
                {**r, "passed": 0, "failure_code": "coarse_witness_replay_invalid"},
                d,
            ),
            "coarse_witness_replay_invalid",
        ),
        (
            lambda p, r, d: (p, r, {**d, "passed": 0}),
            "quadratic_risk_decomposition_invalid",
        ),
    ],
)
def test_invalid_evidence_maps_to_closed_decisions(mutator, expected: str) -> None:
    gates = mutator(*_passing())
    decision = decide_workflow(
        preflight_gate=gates[0],
        replay_gate=gates[1],
        decomposition_gate=gates[2],
        evidence={},
    )
    assert decision["decision"] == expected
    assert decision["decision"] in DECISIONS


def test_required_gate_boundaries() -> None:
    preflight, replay, decomposition = _passing()
    decision = decide_workflow(
        preflight_gate=preflight,
        replay_gate=replay,
        decomposition_gate=decomposition,
        evidence={
            "q0_nominee_lower_bounds": [1.0, 1.0, 1.0],
            "q0_positive_fine_counts": [56, 56, 56],
            "later_adjusted_positive_count": 0,
            "all_point_positive_candidate_count": 0,
            "coarse_witness_overall_energy": 1.0,
        },
    )
    for required in ("none", "preflight", "replay", "decompose"):
        workflow = evaluate_required_gate(
            preflight_gate=preflight,
            replay_gate=replay,
            decomposition_gate=decomposition,
            decision=decision,
            require_gate=required,
        )
        assert workflow["required_gate_pass"] == 1
    with pytest.raises(ValueError):
        evaluate_required_gate(
            preflight_gate=preflight,
            replay_gate=replay,
            decomposition_gate=decomposition,
            decision=decision,
            require_gate="confirm",
        )


def test_required_gates_are_cumulative_and_pending_is_not_a_failure_decision() -> None:
    preflight, replay, decomposition = _passing()
    failed_preflight = {**preflight, "passed": 0}
    pending = {"evaluation_status": "not_evaluated", "passed": 0}
    decision = decide_workflow(
        preflight_gate=preflight,
        replay_gate=pending,
        decomposition_gate=pending,
        evidence=None,
    )
    assert decision["decision"] == "ready_for_replay"
    assert decision["scientific_evidence_complete"] == 0
    workflow = evaluate_required_gate(
        preflight_gate=failed_preflight,
        replay_gate=replay,
        decomposition_gate=decomposition,
        decision=decision,
        require_gate="decompose",
    )
    assert workflow["required_gate_pass"] == 0
