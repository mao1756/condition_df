from __future__ import annotations

import copy

import pytest

from mnist.d0_score_density_ratio_h1_trust_gate import (
    H1TrustDecision,
    H1TrustThresholds,
    decide_h1_workflow,
    evaluate_h1_calibration,
    evaluate_h1_operator_preflight,
    evaluate_h1_pilot,
    evaluate_h1_pilot_candidate,
    evaluate_h1_workflow,
    normalize_h1_candidate,
    not_evaluated_gate,
)


def _operator(**updates):
    value = {
        "complete": 1,
        "finite": 1,
        "gamma_symmetry_pass": 1,
        "gamma_positivity_pass": 1,
        "orientation_invariance_pass": 1,
        "analytic_agreement_pass": 1,
        "identical_anchor_zero_pass": 1,
        "identical_anchor_gradient_zero_pass": 1,
        "constant_shift_l2_detection_pass": 1,
        "stopped_anchor_pass": 1,
        "boundary_finite_pass": 1,
        "cuda_second_order_backward_pass": 1,
        "lambda_zero_regression_pass": 1,
        "stateless_stream_replay_pass": 1,
        "candidate_order_invariance_pass": 1,
        "teacher_null_namespace_isolation_pass": 1,
    }
    value.update(updates)
    return value


def _calibration(**updates):
    value = {
        "complete": 1,
        "finite": 1,
        "value_scale": 0.2,
        "energy_scale": 0.3,
        "lambda_base": 0.4,
        "training_only": 1,
        "evidence_overlap_path_count": 0,
        "shared_teacher_null": 1,
        "deterministic_replay_pass": 1,
    }
    value.update(updates)
    return value


def _candidate(multiplier: float, **updates):
    value = {
        "evaluation_status": "evaluated",
        "multiplier": multiplier,
        "learning_rate": 3e-5,
        "accumulation_steps": 8,
        "base_channels": 32,
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "optimizer_health_pass": 1,
        "h1_health_pass": 1,
        "maximum_clip_fraction_observed": 0.0,
        "teacher_complete": 1,
        "teacher_finite": 1,
        "teacher_boundary_admissible": 1,
        "teacher_selected_step": 250,
        "teacher_panel_b_confirmed": 1,
        "teacher_panel_b_lower_bounds": [0.01, 0.01],
        "teacher_panel_b_bce": 0.67,
        "teacher_score_gain_overall": 0.95,
        "teacher_score_gain_data_end": 0.95,
        "teacher_flux_cosine_overall": 0.99,
        "teacher_time_bin_flux_cosines": [0.96] * 5,
        "teacher_relative_flux_l2_overall": 0.10,
        "teacher_relative_flux_l2_data_end": 0.10,
        "teacher_time_bin_relative_flux_l2": [0.15] * 5,
        "null_complete": 1,
        "null_finite": 1,
        "null_boundary_admissible": 1,
        "null_optimizer_health_pass": 1,
        "null_selected_step": 0,
        "null_panel_b_rejected": 1,
        "null_panel_b_lower_bounds": [-0.01, -0.01],
    }
    value.update(updates)
    return value


def _baseline(**updates):
    value = _candidate(
        0.0,
        teacher_flux_cosine_overall=0.90,
        teacher_time_bin_flux_cosines=[0.85] * 5,
        teacher_relative_flux_l2_overall=0.30,
        teacher_relative_flux_l2_data_end=0.30,
        teacher_time_bin_relative_flux_l2=[0.35] * 5,
    )
    value.update(updates)
    return value


def _passed_gate():
    return {"evaluation_status": "evaluated", "passed": 1}


def _failed_gate():
    return {"evaluation_status": "evaluated", "passed": 0}


def test_frozen_thresholds_reject_changes():
    assert H1TrustThresholds().multipliers == (0.0, 0.1, 0.3, 1.0)
    with pytest.raises(ValueError):
        H1TrustThresholds(multipliers=(0.0, 0.2))
    with pytest.raises(ValueError):
        H1TrustThresholds(minimum_relative_l2_reduction=0.05)


def test_operator_and_calibration_fail_closed():
    assert evaluate_h1_operator_preflight(_operator())["passed"] == 1
    assert (
        evaluate_h1_operator_preflight(
            _operator(identical_anchor_gradient_zero_pass=0)
        )["passed"]
        == 0
    )
    assert evaluate_h1_calibration(_calibration())["passed"] == 1
    assert evaluate_h1_calibration(_calibration(training_only=0))["passed"] == 0
    assert evaluate_h1_calibration(_calibration(lambda_base=0.0))["passed"] == 0


def test_null_marginals_are_discovery_only_under_simultaneous_family():
    baseline = _baseline(null_panel_b_lower_bounds=[0.001, -0.002])
    candidate = _candidate(
        0.3,
        null_panel_b_lower_bounds=[0.002, -0.001],
    )
    assert evaluate_h1_pilot_candidate(
        candidate, baseline=baseline
    )["null_pass"] == 1


def test_nested_candidate_normalization():
    nested = {
        "multiplier": 0.3,
        "learning_rate": 3e-5,
        "accumulation_steps": 8,
        "base_channels": 32,
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "optimizer_health_pass": 1,
        "h1_health_pass": 1,
        "maximum_clip_fraction_observed": 0.0,
        "teacher": {
            "metrics": {
                "complete": 1,
                "finite": 1,
                "boundary_admissible": 1,
                "selection": {
                    "selected_step": 500,
                    "confirmation": {
                        "accepted": 1,
                        "panel_b_lower_bounds": [0.01, 0.02],
                        "panel_b_overall_bce": 0.66,
                    },
                },
                "score_gain_overall": 0.95,
                "score_gain_data_end": 0.94,
                "overall_flux_cosine": 0.99,
                "time_bin_flux_cosines": [0.96] * 5,
                "overall_relative_flux_l2": 0.1,
                "data_end_relative_flux_l2": 0.1,
                "time_bin_relative_flux_l2": [0.15] * 5,
            }
        },
        "null": {
            "metrics": {
                "complete": 1,
                "finite": 1,
                "boundary_admissible": 1,
                "optimizer_health_pass": 1,
                "selection": {
                    "selected_step": 0,
                    "confirmation": {
                        "accepted": 0,
                        "panel_b_lower_bounds": [-0.01, -0.02],
                    },
                },
            }
        },
    }
    value = normalize_h1_candidate(nested)
    assert value["teacher_selected_step"] == 500
    assert value["teacher_panel_b_bce"] == 0.66
    assert value["null_panel_b_rejected"] == 1
    assert value["teacher_time_bin_flux_cosines"] == [0.96] * 5


def test_nonzero_candidate_requires_full_derivatives_and_two_reductions():
    baseline = _baseline()
    gate = evaluate_h1_pilot_candidate(_candidate(0.1), baseline=baseline)
    assert gate["passed"] == 1
    assert gate["derivative_pass"] == 1
    assert gate["relative_l2_reduction_pass"] == 1

    bad_bin = evaluate_h1_pilot_candidate(
        _candidate(0.1, teacher_time_bin_flux_cosines=[0.949] + [0.99] * 4),
        baseline=baseline,
    )
    assert bad_bin["passed"] == 0
    assert bad_bin["derivative_pass"] == 0

    bad_end_reduction = evaluate_h1_pilot_candidate(
        _candidate(0.1, teacher_relative_flux_l2_data_end=0.29),
        baseline=baseline,
    )
    assert bad_end_reduction["passed"] == 0
    assert bad_end_reduction["relative_l2_reduction_pass"] == 0


def test_baseline_does_not_claim_derivative_pass_but_must_be_healthy():
    gate = evaluate_h1_pilot_candidate(_baseline())
    assert gate["passed"] == 1
    assert gate["is_baseline"] == 1
    assert gate["derivative_pass"] == 0
    unhealthy = evaluate_h1_pilot_candidate(_baseline(optimizer_health_pass=0))
    assert unhealthy["passed"] == 0


def test_pilot_ranking_uses_l2_then_cosine_then_bce_then_multiplier():
    baseline = _baseline()
    arm_01 = _candidate(
        0.1,
        teacher_relative_flux_l2_overall=0.12,
        teacher_relative_flux_l2_data_end=0.12,
        teacher_time_bin_relative_flux_l2=[0.14] * 5,
    )
    arm_03 = _candidate(
        0.3,
        teacher_relative_flux_l2_overall=0.11,
        teacher_relative_flux_l2_data_end=0.11,
        teacher_time_bin_relative_flux_l2=[0.12] * 5,
        teacher_panel_b_bce=0.69,
    )
    arm_10 = _candidate(
        1.0,
        teacher_relative_flux_l2_overall=0.11,
        teacher_relative_flux_l2_data_end=0.11,
        teacher_time_bin_relative_flux_l2=[0.12] * 5,
        teacher_time_bin_flux_cosines=[0.951] * 5,
        teacher_panel_b_bce=0.60,
    )
    gate = evaluate_h1_pilot(
        [arm_10, baseline, arm_03, arm_01],
        panel_power=_passed_gate(),
        null_family=_passed_gate(),
    )
    assert gate["passed"] == 1
    # 0.3 and 1.0 tie on max L2; 0.3 wins on minimum cosine before BCE.
    assert gate["selected_profile"]["selected_multiplier"] == 0.3


def test_pilot_reports_optimizer_overregularization_and_null_failures():
    rows = [_baseline()] + [_candidate(value) for value in (0.1, 0.3, 1.0)]
    optimizer = copy.deepcopy(rows)
    optimizer[2]["maximum_clip_fraction_observed"] = 0.11
    gate = evaluate_h1_pilot(
        optimizer, panel_power=_passed_gate(), null_family=_passed_gate()
    )
    assert gate["passed"] == 0
    assert gate["optimizer_health_pass"] == 0

    over = copy.deepcopy(rows)
    for value in over[1:]:
        value["teacher_panel_b_confirmed"] = 0
        value["teacher_panel_b_lower_bounds"] = [-0.01, -0.01]
    gate = evaluate_h1_pilot(
        over, panel_power=_passed_gate(), null_family=_passed_gate()
    )
    assert gate["overregularized"] == 1

    null = evaluate_h1_pilot(
        rows, panel_power=_passed_gate(), null_family=_failed_gate()
    )
    assert null["passed"] == 0
    assert null["null_family_pass"] == 0


def _decision(**updates):
    values = {
        "provenance": _passed_gate(),
        "operator": _passed_gate(),
        "calibration": _passed_gate(),
        "pilot_panel_power": _passed_gate(),
        "pilot": {
            "evaluation_status": "evaluated",
            "passed": 1,
            "optimizer_health_pass": 1,
            "overregularized": 0,
            "null_family_pass": 1,
            "null_family": _passed_gate(),
        },
        "confirmation_panel_power": _passed_gate(),
        "controls": {
            "evaluation_status": "evaluated",
            "passed": 1,
            "optimizer_health_pass": 1,
            "teacher_study": {
                "passed": 1,
                "optimizer_health_pass": 1,
                "classification_passing_seed_count": 3,
                "derivative_passing_seed_count": 3,
                "panel_disagreement": 0,
            },
            "null_family": {
                "passed": 1,
                "optimizer_health_pass": 1,
                "familywise_false_discovery": 0,
                "selection_false_discovery": 0,
            },
        },
    }
    values.update(updates)
    return decide_h1_workflow(**values)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"provenance": _failed_gate()}, H1TrustDecision.CONTROL_PROVENANCE_INVALID.value),
        ({"operator": _failed_gate()}, H1TrustDecision.H1_OPERATOR_INVALID.value),
        ({"calibration": _failed_gate()}, H1TrustDecision.H1_CALIBRATION_INVALID.value),
        ({"pilot_panel_power": _failed_gate()}, H1TrustDecision.EVIDENCE_PANEL_UNDERPOWERED.value),
        (
            {
                "pilot": {
                    "evaluation_status": "evaluated",
                    "passed": 0,
                    "optimizer_health_pass": 0,
                    "null_family_pass": 1,
                    "null_family": _passed_gate(),
                }
            },
            H1TrustDecision.H1_OPTIMIZER_INVALID.value,
        ),
        (
            {
                "pilot": {
                    "evaluation_status": "evaluated",
                    "passed": 0,
                    "optimizer_health_pass": 1,
                    "overregularized": 1,
                    "null_family_pass": 1,
                    "null_family": _passed_gate(),
                }
            },
            H1TrustDecision.H1_OVERREGULARIZED.value,
        ),
    ],
)
def test_closed_decision_boundaries(updates, expected):
    assert _decision(**updates)["decision"] == expected


def test_confirmation_decisions_value_only_audit_null_and_repaired():
    assert _decision()["decision"] == H1TrustDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value

    value_only = _decision(
        controls={
            "evaluation_status": "evaluated",
            "passed": 0,
            "optimizer_health_pass": 1,
            "teacher_study": {
                "passed": 0,
                "optimizer_health_pass": 1,
                "classification_passing_seed_count": 3,
                "derivative_passing_seed_count": 0,
                "panel_disagreement": 0,
            },
            "null_family": {"passed": 1, "optimizer_health_pass": 1},
        }
    )
    assert value_only["decision"] == H1TrustDecision.H1_DENSITY_RATIO_VALUE_ONLY.value

    false = _decision(
        controls={
            "evaluation_status": "evaluated",
            "passed": 0,
            "optimizer_health_pass": 1,
            "teacher_study": {"classification_passing_seed_count": 3},
            "null_family": {
                "passed": 0,
                "optimizer_health_pass": 1,
                "familywise_false_discovery": 1,
            },
        }
    )
    assert false["decision"] == H1TrustDecision.SELECTION_FALSE_DISCOVERY.value

    audit = _decision(
        controls={
            "evaluation_status": "evaluated",
            "passed": 0,
            "optimizer_health_pass": 1,
            "teacher_study": {
                "classification_passing_seed_count": 3,
                "panel_disagreement": 1,
            },
            "null_family": {"passed": 1, "optimizer_health_pass": 1},
        }
    )
    assert audit["decision"] == H1TrustDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE.value


def test_required_gates_and_no_sampling_or_physical_work():
    pilot = {"evaluation_status": "evaluated", "passed": 1}
    report = evaluate_h1_workflow(
        provenance=_passed_gate(),
        operator=_passed_gate(),
        calibration=_passed_gate(),
        preflight=_passed_gate(),
        pilot_panel_power=_passed_gate(),
        pilot=pilot,
        confirmation_panel_power=_passed_gate(),
        teacher_study={
            "evaluation_status": "evaluated",
            "passed": 1,
            "optimizer_health_pass": 1,
        },
        null_family={
            "evaluation_status": "evaluated",
            "passed": 1,
            "optimizer_health_pass": 1,
        },
        require_gate="controls",
    )
    assert report["required_gate_pass"] == 1
    assert report["physical_training_performed"] == 0
    assert report["sampling_performed"] == 0
    with pytest.raises(ValueError):
        evaluate_h1_workflow(
            provenance=1,
            operator=1,
            calibration=1,
            preflight=1,
            pilot_panel_power=1,
            pilot=pilot,
            confirmation_panel_power=1,
            teacher_study={},
            null_family={},
            require_gate="bad",
        )


def test_not_evaluated_is_fail_closed():
    gate = not_evaluated_gate("x", "not run")
    assert gate["evaluation_status"] == "not_evaluated"
    assert gate["passed"] == 0
