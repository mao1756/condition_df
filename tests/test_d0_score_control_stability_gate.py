from __future__ import annotations

import copy
import json

import pytest

from mnist.d0_score_control_stability_gate import (
    ProbeBankStatus,
    StabilityDecision,
    StabilityThresholds,
    classify_probe_bank_status,
    decide_stability_confirmation,
    evaluate_pilot_candidate,
    evaluate_stability_confirmation,
    evaluate_stability_pilot,
    evaluate_stability_workflow,
    evaluate_stein_identity_preflight,
    not_evaluated_gate,
    select_stability_profile,
)


def _banks(lower: float, *, risk: float = 1.0) -> dict[str, object]:
    return {
        name: {
            scope: {"lower_bound": lower, "model_score_risk": risk}
            for scope in ("overall", "data_end")
        }
        for name in ("a", "b")
    }


def _pilot_teacher(*, risk: float = 1.0, clip: float = 0.05) -> dict[str, object]:
    return {
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "selected_step": 250,
        "clip_fraction_steps_101_1000": clip,
        "final_200_clip_fraction": clip,
        "selection_objective_banks": _banks(0.01, risk=risk),
        "mean_dual_bank_selection_risk": risk,
        "selection_overall_score_gain": 0.1,
        "selection_data_end_score_gain": 0.05,
        "selection_overall_flux_cosine": 0.2,
        "selection_data_end_flux_cosine": 0.1,
        "selection_overall_relative_flux_l2": 0.9,
        "selection_data_end_relative_flux_l2": 0.95,
    }


def _pilot_null(*, clip: float = 0.05) -> dict[str, object]:
    return {
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "selected_step": 0,
        "clip_fraction_steps_101_1000": clip,
        "final_200_clip_fraction": clip,
        "selection_objective_banks": _banks(0.0),
    }


def _candidate(lr: float, *, risk: float = 1.0, clip: float = 0.05) -> dict[str, object]:
    return {
        "learning_rate": lr,
        "teacher": {"metrics": _pilot_teacher(risk=risk, clip=clip)},
        "null": {"metrics": _pilot_null(clip=clip)},
    }


def _teacher(seed: int, *, clip: float = 0.05) -> dict[str, object]:
    return {
        "model_seed": seed,
        "complete": 1,
        "finite": 1,
        "selected_step": 500,
        "audit_overall_score_gain": 0.92,
        "audit_data_end_score_gain": 0.91,
        "overall_flux_cosine": 0.99,
        "time_bin_flux_cosines": [0.96] * 5,
        "overall_relative_flux_l2": 0.12,
        "time_bin_relative_flux_l2": [0.18] * 5,
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": clip,
        "audit_objective_banks": _banks(0.01),
    }


def _null(seed: int, *, clip: float = 0.05) -> dict[str, object]:
    return {
        "model_seed": seed,
        "complete": 1,
        "finite": 1,
        "selected_step": 0,
        "comparator": "analytic_zero",
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": clip,
        "audit_objective_banks": _banks(0.0),
    }


def _stein() -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "path_count": 128,
        "bootstrap_confidence": 0.99,
        "null_identity": {"finite": 1, "bootstrap_interval": [-0.01, 0.02]},
        "teacher_identities": [
            {"a": value, "finite": 1, "bootstrap_interval": [-0.01, 0.01]}
            for value in (0.0, 0.25, 0.5, 1.0, 2.0)
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def test_stein_identity_requires_every_frozen_amplitude_and_interval() -> None:
    gate = evaluate_stein_identity_preflight(_stein())
    assert gate["passed"] == 1
    changed = copy.deepcopy(_stein())
    changed["teacher_identities"][2]["bootstrap_interval"] = [0.001, 0.1]  # type: ignore[index]
    bad = evaluate_stein_identity_preflight(changed)
    assert bad["passed"] == 0
    assert bad["subchecks"]["teacher_identity_intervals"]["passed"] == 0


def test_stein_gate_recomputes_the_serialized_core_preflight_schema() -> None:
    def identity(name: str) -> dict[str, object]:
        return {
            "name": name,
            "finite": 1,
            "passed": 1,
            "measured_minus_predicted": {
                "lower_bound": -1e-10,
                "upper_bound": 2e-10,
                "finite": 1,
            },
        }

    core = {
        "schema": "d0-score-control-stream-v1-stein-preflight",
        "schema_version": 1,
        "passed": 1,
        "finite": 1,
        "path_count_per_law": 128,
        "anchors_per_path": 32,
        "confidence": 0.99,
        "teacher_scales": [0.0, 0.25, 0.5, 1.0, 2.0],
        "null_identity": identity("dirichlet_null"),
        "teacher_identities": [
            {**identity(f"teacher-{scale}"), "scale": scale}
            for scale in (0.0, 0.25, 0.5, 1.0, 2.0)
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }

    recomputed = evaluate_stein_identity_preflight(core)

    assert recomputed["passed"] == 1
    assert recomputed["operator_identity_pass"] == 1
    json.dumps(recomputed, allow_nan=False)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda row: row["teacher"]["metrics"].__setitem__("final_200_clip_fraction", 0.100001), "final_clip_fractions"),
        (lambda row: row["teacher"]["metrics"].__setitem__("selected_step", 0), "teacher_selected_nonzero"),
        (lambda row: row["null"]["metrics"]["selection_objective_banks"]["b"]["data_end"].__setitem__("lower_bound", 1e-12), "null_no_positive_bank"),
    ],
)
def test_pilot_candidate_fails_closed_at_stability_boundaries(mutation, failed_check) -> None:
    candidate = _candidate(1e-5)
    assert evaluate_pilot_candidate(candidate)["passed"] == 1
    mutation(candidate)
    gate = evaluate_pilot_candidate(candidate)
    assert gate["passed"] == 0
    assert gate["subchecks"][failed_check]["passed"] == 0
    json.dumps(gate, allow_nan=False)


def test_pilot_selects_by_risk_then_clipping_then_smaller_learning_rate() -> None:
    candidates = [
        _candidate(1e-4, risk=2.0),
        _candidate(3e-5, risk=1.0, clip=0.08),
        _candidate(1e-5, risk=1.0, clip=0.04),
        _candidate(3e-6, risk=1.0, clip=0.04),
    ]
    pilot = evaluate_stability_pilot(candidates)
    assert pilot["passed"] == 1
    selection = select_stability_profile(candidates)
    assert selection["selected"] == 1
    assert selection["profile"]["learning_rate"] == pytest.approx(3e-6)


def test_confirmation_preserves_teacher_null_thresholds_and_optimizer_health() -> None:
    teachers = [_teacher(seed) for seed in (11, 12, 13)]
    nulls = [_null(seed) for seed in (11, 12, 13)]
    passing = evaluate_stability_confirmation(
        teachers, nulls, probe_bank_status=ProbeBankStatus.AGREE
    )
    assert passing["passed"] == 1
    assert passing["optimizer_health_pass"] == 1

    clipped = copy.deepcopy(teachers)
    clipped[0]["post_warmup_clip_fraction"] = 0.1000001
    failed = evaluate_stability_confirmation(
        clipped, nulls, probe_bank_status=ProbeBankStatus.AGREE
    )
    assert failed["passed"] == 0
    assert failed["optimizer_health_pass"] == 0


def test_workflow_decision_precedence_and_required_gates() -> None:
    thresholds = StabilityThresholds()
    pilot = evaluate_stability_pilot(
        [_candidate(value) for value in thresholds.pilot_learning_rates]
    )
    confirmation = evaluate_stability_confirmation(
        [_teacher(seed) for seed in (11, 12, 13)],
        [_null(seed) for seed in (11, 12, 13)],
        probe_bank_status=ProbeBankStatus.AGREE,
    )
    report = evaluate_stability_workflow(
        provenance={"passed": 1},
        stein=evaluate_stein_identity_preflight(_stein()),
        pilot=pilot,
        confirmation=confirmation,
        require_gate="controls",
    )
    assert report["required_gate_pass"] == 1
    assert report["decision"]["decision"] == StabilityDecision.CONTROL_PIPELINE_REPAIRED.value
    assert report["decision"]["physical_training_authorized"] == 1
    assert report["sampling_performed"] == 0

    clipped = copy.deepcopy(confirmation)
    clipped["passed"] = 0
    clipped["optimizer_health_pass"] = 0
    failed = evaluate_stability_workflow(
        provenance={"passed": 1}, stein={"passed": 1}, pilot=pilot,
        confirmation=clipped, require_gate="controls",
    )
    assert failed["decision"]["decision"] == StabilityDecision.OPTIMIZER_STABILITY_INVALID.value

    disagree = copy.deepcopy(confirmation)
    disagree["passed"] = 0
    disagree["probe_bank_status"] = "disagree"
    result = evaluate_stability_workflow(
        provenance={"passed": 1}, stein={"passed": 1}, pilot=pilot,
        confirmation=disagree,
    )
    assert result["decision"]["decision"] == StabilityDecision.TRACE_ESTIMATOR_INCONCLUSIVE.value


def test_probe_status_and_skips_are_explicit() -> None:
    assert classify_probe_bank_status(studies_evaluated=False, banks_agree=True) is ProbeBankStatus.NOT_EVALUATED
    assert classify_probe_bank_status(studies_evaluated=True, banks_agree=True) is ProbeBankStatus.AGREE
    skipped = not_evaluated_gate("confirmation", "pilot failed")
    assert skipped["evaluation_status"] == "not_evaluated"
    assert skipped["passed"] == 0
    with pytest.raises(ValueError, match="reason"):
        not_evaluated_gate("confirmation", "")


def test_skipped_and_evaluated_failed_pilots_have_distinct_next_actions() -> None:
    common = {
        "provenance": {"passed": 1},
        "stein": {"evaluation_status": "evaluated", "passed": 1},
        "confirmation": not_evaluated_gate("confirmation", "pilot prerequisite"),
    }
    skipped = evaluate_stability_workflow(
        **common,
        pilot=not_evaluated_gate("stability_pilot", "pilot was not run"),
    )
    assert (
        skipped["decision"]["decision"]
        == StabilityDecision.OPTIMIZER_STABILITY_UNRESOLVED.value
    )
    assert (
        skipped["decision"]["recommended_next_action"]
        == "run the train/selection-only stability pilot"
    )

    evaluated_failure = evaluate_stability_workflow(
        **common,
        pilot={
            "gate": "stability_pilot",
            "evaluation_status": "evaluated",
            "passed": 0,
        },
    )
    assert (
        evaluated_failure["decision"]["decision"]
        == StabilityDecision.OPTIMIZER_STABILITY_UNRESOLVED.value
    )
    assert "coercivity" in evaluated_failure["decision"]["recommended_next_action"]


@pytest.mark.parametrize(
    ("provenance", "stein", "pilot", "confirmation", "expected"),
    [
        (
            {"passed": 0},
            {"evaluation_status": "evaluated", "passed": 1},
            {"evaluation_status": "evaluated", "passed": 1},
            {"evaluation_status": "evaluated", "passed": 1},
            StabilityDecision.CONTROL_PROVENANCE_INVALID,
        ),
        (
            {"passed": 1},
            {"evaluation_status": "not_evaluated", "passed": 0},
            {"evaluation_status": "not_evaluated", "passed": 0},
            {"evaluation_status": "not_evaluated", "passed": 0},
            StabilityDecision.STABILITY_PREFLIGHT_INVALID,
        ),
        (
            {"passed": 1},
            {"evaluation_status": "evaluated", "passed": 0},
            {"evaluation_status": "not_evaluated", "passed": 0},
            {"evaluation_status": "not_evaluated", "passed": 0},
            StabilityDecision.OPERATOR_IDENTITY_INVALID,
        ),
        (
            {"passed": 1},
            {"evaluation_status": "evaluated", "passed": 1},
            {"evaluation_status": "evaluated", "passed": 0},
            {"evaluation_status": "not_evaluated", "passed": 0},
            StabilityDecision.OPTIMIZER_STABILITY_UNRESOLVED,
        ),
        (
            {"passed": 1},
            {"evaluation_status": "evaluated", "passed": 1},
            {"evaluation_status": "evaluated", "passed": 1},
            {
                "evaluation_status": "evaluated",
                "passed": 0,
                "optimizer_health_pass": 1,
                "probe_bank_status": "agree",
                "implicit_teacher_study": {"passed": 0},
                "null_study": {"passed": 1},
            },
            StabilityDecision.IMPLICIT_OBJECTIVE_UNSTABLE,
        ),
    ],
)
def test_all_remaining_closed_decision_states(
    provenance, stein, pilot, confirmation, expected
) -> None:
    result = decide_stability_confirmation(
        provenance=provenance,
        stein=stein,
        pilot=pilot,
        confirmation=confirmation,
    )
    assert result["decision"] == expected.value
    assert result["physical_training_performed"] == 0
    assert result["sampling_performed"] == 0
