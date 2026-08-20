from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

from mnist.d0_jacobi_rb_absolute_coordinate_gate import (
    DECISION_ORDER,
    INTEGRITY_DECISIONS,
    SCIENTIFIC_DECISIONS,
    ZERO_AUTHORIZATION_FIELDS,
    ZERO_CLAIM_FIELDS,
    ZERO_WORK_FIELDS,
    decide_absolute_coordinate,
    decision_exit_code,
)


def _evidence(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "provenance_valid": 1,
        "portable_directional_parent_valid": 1,
        "coarse_witness_parent_valid": 1,
        "hypothesis_plan_valid": 1,
        "coarse_witness_replay_valid": 1,
        "translation_symmetry_audit_valid": 1,
        "coordinate_projection_algebra_valid": 1,
        "coordinate_inference_valid": 1,
        "q0_positive_control": 1,
        "later_quartile_positive": {"q1": 1, "q2": 1, "q3": 1},
    }
    result.update(updates)
    return result


def _assert_safe(decision: dict[str, object]) -> None:
    for name in (*ZERO_AUTHORIZATION_FIELDS, *ZERO_WORK_FIELDS, *ZERO_CLAIM_FIELDS):
        assert decision[name] == 0


def test_decision_vocabulary_and_precedence_are_frozen() -> None:
    assert DECISION_ORDER == (
        "control_provenance_invalid",
        "portable_directional_parent_invalid",
        "coordinate_hypothesis_plan_invalid",
        "coarse_witness_replay_invalid",
        "translation_symmetry_audit_invalid",
        "coordinate_projection_algebra_invalid",
        "coordinate_inference_invalid",
        "coarse_signal_nonreplicating_stop",
        "absolute_coordinate_signal_not_detected_stop",
        "absolute_coordinate_signal_partial_stop",
        "absolute_coordinate_representation_hypothesis_supported",
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("provenance_valid", "control_provenance_invalid"),
        ("portable_directional_parent_valid", "portable_directional_parent_invalid"),
        ("coarse_witness_parent_valid", "control_provenance_invalid"),
        ("hypothesis_plan_valid", "coordinate_hypothesis_plan_invalid"),
        ("coarse_witness_replay_valid", "coarse_witness_replay_invalid"),
        ("translation_symmetry_audit_valid", "translation_symmetry_audit_invalid"),
        ("coordinate_projection_algebra_valid", "coordinate_projection_algebra_invalid"),
        ("coordinate_inference_valid", "coordinate_inference_invalid"),
    ],
)
def test_each_integrity_failure_is_closed(field: str, expected: str) -> None:
    evidence = _evidence()
    evidence[field] = 0
    decision = decide_absolute_coordinate(evidence)
    assert decision["decision"] == expected
    assert decision["scientific_evidence_complete"] == 0
    assert decision["invalid_evidence"] == 1
    assert decision_exit_code(decision) == 1
    _assert_safe(decision)


def test_integrity_precedence_is_fail_closed() -> None:
    evidence = _evidence(
        provenance_valid=0,
        portable_directional_parent_valid=0,
        coordinate_inference_valid=0,
    )
    assert decide_absolute_coordinate(evidence)["decision"] == "control_provenance_invalid"

    evidence["provenance_valid"] = 1
    evidence["coarse_witness_parent_valid"] = 1
    assert decide_absolute_coordinate(evidence)["decision"] == (
        "portable_directional_parent_invalid"
    )


def test_parent_specific_provenance_can_replace_composite_bit() -> None:
    evidence = _evidence()
    del evidence["provenance_valid"]
    assert decide_absolute_coordinate(evidence)["decision"] == (
        "absolute_coordinate_representation_hypothesis_supported"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence.pop("q0_positive_control"),
        lambda evidence: evidence.__setitem__("q0_positive_control", "1"),
        lambda evidence: evidence.__setitem__("later_quartile_positive", None),
        lambda evidence: evidence.__setitem__(
            "later_quartile_positive", {"q1": 1, "q2": 1}
        ),
        lambda evidence: evidence.__setitem__(
            "later_quartile_positive", {"q1": 1, "q2": 1, "q3": "1"}
        ),
        lambda evidence: evidence.__setitem__(
            "later_quartile_positive", {"q1": 1, "q2": 1, "q3": 1, "q4": 0}
        ),
    ],
)
def test_malformed_scientific_bits_are_inference_invalid(
    mutation: Callable[[dict[str, object]], object],
) -> None:
    evidence = _evidence()
    mutation(evidence)
    decision = decide_absolute_coordinate(evidence)
    assert decision["decision"] == "coordinate_inference_invalid"
    assert decision["scientific_evidence_complete"] == 0
    _assert_safe(decision)


def test_q0_failure_precedes_later_quartile_results() -> None:
    evidence = _evidence(
        q0_positive_control=0,
        later_quartile_positive={"q1": 1, "q2": 1, "q3": 1},
    )
    decision = decide_absolute_coordinate(evidence)
    assert decision["decision"] == "coarse_signal_nonreplicating_stop"
    assert decision["valid_scientific_stop"] == 1
    assert decision["fresh_coordinate_learner_plan_drafting_recommended"] == 0
    _assert_safe(decision)


def test_zero_later_quartile_positives_stops_as_not_detected() -> None:
    decision = decide_absolute_coordinate(
        _evidence(later_quartile_positive={"q1": 0, "q2": 0, "q3": 0})
    )
    assert decision["decision"] == "absolute_coordinate_signal_not_detected_stop"
    assert decision["later_quartile_positive_count"] == 0
    assert decision["fresh_coordinate_learner_plan_drafting_recommended"] == 0
    _assert_safe(decision)


@pytest.mark.parametrize(
    "later",
    [
        {"q1": 1, "q2": 0, "q3": 0},
        {"q1": 0, "q2": 1, "q3": 1},
    ],
)
def test_one_or_two_later_quartile_positives_stop_as_partial(
    later: dict[str, int],
) -> None:
    decision = decide_absolute_coordinate(_evidence(later_quartile_positive=later))
    assert decision["decision"] == "absolute_coordinate_signal_partial_stop"
    assert decision["later_quartile_positive_count"] == sum(later.values())
    assert decision["fresh_coordinate_learner_plan_drafting_recommended"] == 0
    _assert_safe(decision)


def test_all_later_quartiles_support_only_plan_drafting_recommendation() -> None:
    decision = decide_absolute_coordinate(_evidence())
    assert decision["decision"] == (
        "absolute_coordinate_representation_hypothesis_supported"
    )
    assert decision["scientific_evidence_complete"] == 1
    assert decision["invalid_evidence"] == 0
    assert decision["valid_scientific_stop"] == 0
    assert decision["absolute_coordinate_representation_hypothesis_supported"] == 1
    assert decision["fresh_coordinate_learner_plan_drafting_recommended"] == 1
    assert "do not train or sample" in str(decision["recommended_next_action"])
    assert decision_exit_code(decision) == 0
    _assert_safe(decision)


def test_every_scientific_outcome_is_terminal_complete_and_safe() -> None:
    fixtures = (
        _evidence(q0_positive_control=0),
        _evidence(later_quartile_positive={"q1": 0, "q2": 0, "q3": 0}),
        _evidence(later_quartile_positive={"q1": 1, "q2": 0, "q3": 0}),
        _evidence(),
    )
    observed = set()
    for evidence in fixtures:
        decision = decide_absolute_coordinate(evidence)
        observed.add(decision["decision"])
        assert decision["terminal"] == 1
        assert decision["scientific_evidence_complete"] == 1
        assert decision_exit_code(decision) == 0
        _assert_safe(decision)
    assert observed == set(SCIENTIFIC_DECISIONS)


def test_input_is_not_mutated() -> None:
    evidence = _evidence()
    original = deepcopy(evidence)
    decide_absolute_coordinate(evidence)
    assert evidence == original


def test_nonmapping_input_fails_closed() -> None:
    decision = decide_absolute_coordinate(None)
    assert decision["decision"] == INTEGRITY_DECISIONS[0]
    assert decision_exit_code(decision) == 1
    _assert_safe(decision)
