from __future__ import annotations

import copy

import pytest

from mnist.d0_score_density_ratio_h1_gradient_control_gate import (
    H1GradientControlDecision,
    H1GradientControlThresholds,
    decide_gradient_control_workflow,
    evaluate_gradient_control_confirmation,
    evaluate_gradient_control_pilot,
    evaluate_gradient_control_pilot_candidate,
    evaluate_gradient_controller_preflight,
    evaluate_gradient_control_workflow,
)


def _pass():
    return {"evaluation_status": "evaluated", "passed": 1}


def _fail():
    return {"evaluation_status": "evaluated", "passed": 0}


def _controller_preflight(**updates):
    value = {
        "complete": 1,
        "finite": 1,
        "exact_target_ratio_algebra_pass": 1,
        "stopped_coefficient_pass": 1,
        "positive_rescaling_invariance_pass": 1,
        "ramp_endpoints_pass": 1,
        "floor_branches_pass": 1,
        "fixed_point_pass": 1,
        "rho_zero_regression_pass": 1,
        "cuda_second_order_backward_pass": 1,
        "boundary_admissibility_pass": 1,
        "candidate_order_invariance_pass": 1,
        "stateless_stream_replay_pass": 1,
        "interruption_replay_pass": 1,
        "no_sampler_import_pass": 1,
        "no_physical_state_training_pass": 1,
    }
    value.update(updates)
    return value


def _health(ratio: float):
    return {
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "optimizer_health_pass": 1,
        "controller_health_pass": 1,
        "fixed_endpoint_step": 4000,
        "controller_active_fraction": 1.0 if ratio else 0.0,
        "maximum_ratio_relative_error": 1e-8,
        "post_ramp_h1_floor_hit_count": 0,
        "nonfinite_coefficient_count": 0,
        "clipping_windows": {"post_warmup": 0.0, "final_500": 0.0, "final_200": 0.0},
    }


def _panel(*, l2: float, cosine: float = 0.99, bce: float = 0.60):
    return {
        "evaluation_status": "evaluated",
        "opened": 1,
        "evaluation_count": 1,
        "confirmed": 1,
        "bce_improvement_lower_bounds": [0.01, 0.01],
        "bce": bce,
        "score_gain_overall": 0.95,
        "score_gain_data_end": 0.95,
        "flux_cosine_overall": cosine,
        "time_bin_flux_cosines": [0.96] * 5,
        "relative_flux_l2_overall": l2,
        "relative_flux_l2_data_end": l2,
        "time_bin_relative_flux_l2": [l2] * 5,
    }


def _candidate(ratio: float, *, l2: float, bce: float = 0.60, b_count: int = 0):
    teacher = {
        **_health(ratio),
        "panels": {
            "a": _panel(l2=l2, bce=bce),
            "b": _panel(l2=l2, bce=bce),
        },
    }
    null = {
        **_health(ratio),
        "selected_step": 0,
        "selection": {"confirmation": {"accepted": 0}},
    }
    return {
        "evaluation_status": "evaluated",
        "target_ratio": ratio,
        "learning_rate": 3e-5,
        "accumulation_steps": 8,
        "base_channels": 32,
        "teacher": teacher,
        "null": null,
        "panel_b_evaluation_count": b_count,
        "matched_effects": {
            "a": {
                "point_reductions": [0.50, 0.50],
                "simultaneous_lower_bounds": [0.01, 0.01],
            },
            "b": {
                "point_reductions": [0.50, 0.50],
                "simultaneous_lower_bounds": [0.01, 0.01],
            },
        },
    }


def _pilot_rows():
    baseline = _candidate(0.0, l2=0.30, b_count=1)
    # The baseline is not required to meet the derivative thresholds.
    for panel in baseline["teacher"]["panels"].values():
        panel.update(
            {
                "flux_cosine_overall": 0.90,
                "time_bin_flux_cosines": [0.90] * 5,
                "time_bin_relative_flux_l2": [0.30] * 5,
            }
        )
    rows = [
        baseline,
        _candidate(0.1, l2=0.14, bce=0.61, b_count=0),
        _candidate(0.3, l2=0.12, bce=0.62, b_count=1),
        _candidate(1.0, l2=0.13, bce=0.59, b_count=0),
    ]
    return rows


def _null_family(size: int):
    return {
        "evaluation_status": "evaluated",
        "passed": 1,
        "family_size": size,
        "familywise_false_discovery": 0,
    }


def _positive_family(size: int):
    return {
        "evaluation_status": "evaluated",
        "passed": 1,
        "family_size": size,
        "all_simultaneous_lower_bounds_positive": 1,
    }


def test_thresholds_are_frozen():
    value = H1GradientControlThresholds()
    assert value.target_ratios == (0.0, 0.1, 0.3, 1.0)
    assert value.fixed_endpoint_step == 4000
    with pytest.raises(ValueError):
        H1GradientControlThresholds(target_ratios=(0.0, 0.2))
    with pytest.raises(ValueError):
        H1GradientControlThresholds(maximum_ratio_relative_error=1e-3)


def test_controller_preflight_fails_closed():
    assert evaluate_gradient_controller_preflight(_controller_preflight())["passed"] == 1
    assert (
        evaluate_gradient_controller_preflight(
            _controller_preflight(stopped_coefficient_pass=0)
        )["passed"]
        == 0
    )
    assert (
        evaluate_gradient_controller_preflight(
            _controller_preflight(no_sampler_import_pass=None)
        )["passed"]
        == 0
    )


def test_fixed_endpoint_controller_and_matched_effect_are_required():
    baseline, candidate = _pilot_rows()[:2]
    gate = evaluate_gradient_control_pilot_candidate(candidate, baseline=baseline)
    assert gate["passed"] == 1
    assert gate["derivative_pass"] == 1
    assert gate["matched_relative_l2_reduction_pass"] == 1

    wrong_step = copy.deepcopy(candidate)
    wrong_step["teacher"]["fixed_endpoint_step"] = 3999
    assert (
        evaluate_gradient_control_pilot_candidate(wrong_step, baseline=baseline)[
            "optimizer_and_controller_health_pass"
        ]
        == 0
    )

    bad_tracking = copy.deepcopy(candidate)
    bad_tracking["null"]["maximum_ratio_relative_error"] = 1.01e-4
    assert (
        evaluate_gradient_control_pilot_candidate(bad_tracking, baseline=baseline)[
            "optimizer_and_controller_health_pass"
        ]
        == 0
    )

    no_effect = copy.deepcopy(candidate)
    no_effect["matched_effects"]["a"]["simultaneous_lower_bounds"] = [0.0, 0.01]
    assert evaluate_gradient_control_pilot_candidate(
        no_effect, baseline=baseline
    )["passed"] == 0


def test_strict_inherited_derivative_boundaries():
    baseline, candidate = _pilot_rows()[:2]
    bad = copy.deepcopy(candidate)
    bad["teacher"]["panels"]["a"]["time_bin_flux_cosines"][0] = 0.949999
    gate = evaluate_gradient_control_pilot_candidate(bad, baseline=baseline)
    assert gate["passed"] == 0
    assert gate["derivative_pass"] == 0

    bad = copy.deepcopy(candidate)
    bad["teacher"]["panels"]["a"]["relative_flux_l2_overall"] = 0.150001
    gate = evaluate_gradient_control_pilot_candidate(bad, baseline=baseline)
    assert gate["derivative_pass"] == 0


def test_pilot_ranks_on_a_then_opens_only_nominee_and_baseline_b():
    rows = _pilot_rows()
    gate = evaluate_gradient_control_pilot(
        rows,
        panel_power=_pass(),
        null_family=_null_family(8),
    )
    assert gate["passed"] == 1
    assert gate["selected_profile"]["selected_ratio"] == 0.3
    assert gate["nominee_panel_b_gate"]["panel_role"] == "b"

    leaked = copy.deepcopy(rows)
    leaked[1]["panel_b_evaluation_count"] = 1
    assert evaluate_gradient_control_pilot(
        leaked, panel_power=_pass(), null_family=_null_family(8)
    )["passed"] == 0


def _confirmation_seed(seed: int, ratio: float = 0.3):
    selected = {**_health(ratio), "panels": {role: _panel(l2=0.12) for role in "bcd"}}
    baseline = {**_health(0.0), "panels": {role: _panel(l2=0.30) for role in "bcd"}}
    null = {**_health(ratio)}
    return {
        "evaluation_status": "evaluated",
        "seed": seed,
        "teacher": selected,
        "baseline": baseline,
        "null": null,
        "matched_effects": {
            role: {"point_reductions": [0.60, 0.60]} for role in "bcd"
        },
    }


def test_confirmation_requires_nine_healthy_tasks_and_two_teacher_seeds():
    seeds = [_confirmation_seed(seed) for seed in (1, 2, 3)]
    gate = evaluate_gradient_control_confirmation(
        seeds,
        selected_ratio=0.3,
        panel_power=_pass(),
        matched_effect_family=_positive_family(18),
        null_family=_null_family(18),
    )
    assert gate["passed"] == 1
    assert gate["passing_teacher_seed_count"] == 3

    bad = copy.deepcopy(seeds)
    bad[0]["baseline"]["finite"] = 0
    failed = evaluate_gradient_control_confirmation(
        bad,
        selected_ratio=0.3,
        panel_power=_pass(),
        matched_effect_family=_positive_family(18),
        null_family=_null_family(18),
    )
    assert failed["passed"] == 0
    assert failed["optimizer_and_controller_health_pass"] == 0


def _decision(**updates):
    values = {
        "provenance": _pass(),
        "controller_preflight": _pass(),
        "pilot_panel_power": _pass(),
        "pilot": {
            "evaluation_status": "evaluated",
            "passed": 1,
            "optimizer_and_controller_health_pass": 1,
            "null_family_pass": 1,
            "null_positive_roles": [],
        },
        "confirmation_panel_power": _pass(),
        "confirmation": {
            "evaluation_status": "evaluated",
            "passed": 1,
            "optimizer_and_controller_health_pass": 1,
            "matched_effect_family_pass": 1,
            "null_family_pass": 1,
            "null_positive_roles": [],
            "classification_passing_seed_count": 3,
        },
    }
    values.update(updates)
    return decide_gradient_control_workflow(**values)


@pytest.mark.parametrize(
    ("updates", "decision"),
    [
        ({"provenance": _fail()}, H1GradientControlDecision.CONTROL_PROVENANCE_INVALID.value),
        ({"controller_preflight": _fail()}, H1GradientControlDecision.H1_GRADIENT_CONTROLLER_INVALID.value),
        ({"pilot_panel_power": _fail()}, H1GradientControlDecision.EVIDENCE_PANEL_UNDERPOWERED.value),
        (
            {
                "pilot": {
                    "evaluation_status": "evaluated",
                    "passed": 0,
                    "optimizer_and_controller_health_pass": 0,
                }
            },
            H1GradientControlDecision.H1_CONTROLLER_OPTIMIZER_INVALID.value,
        ),
        (
            {
                "pilot": {
                    "evaluation_status": "evaluated",
                    "passed": 0,
                    "optimizer_and_controller_health_pass": 1,
                    "null_family_pass": 1,
                    "overregularized": 1,
                }
            },
            H1GradientControlDecision.H1_CONTROLLER_OVERREGULARIZED.value,
        ),
        (
            {
                "pilot": {
                    "evaluation_status": "evaluated",
                    "passed": 0,
                    "optimizer_and_controller_health_pass": 1,
                    "null_family_pass": 1,
                    "matched_effect_unconfirmed": 1,
                }
            },
            H1GradientControlDecision.H1_CAUSAL_EFFECT_UNCONFIRMED.value,
        ),
    ],
)
def test_closed_pilot_decisions(updates, decision):
    assert _decision(**updates)["decision"] == decision


def test_confirmation_decisions_and_authorization():
    repaired = _decision()
    assert repaired["decision"] == H1GradientControlDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value
    assert repaired["physical_training_authorized"] == 1
    assert repaired["sampling_authorized"] == 0

    audit = _decision(
        confirmation={
            "evaluation_status": "evaluated",
            "passed": 0,
            "optimizer_and_controller_health_pass": 1,
            "matched_effect_family_pass": 1,
            "null_family_pass": 0,
            "null_positive_roles": ["c"],
        }
    )
    assert audit["decision"] == H1GradientControlDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE.value

    effect = _decision(
        confirmation={
            "evaluation_status": "evaluated",
            "passed": 0,
            "optimizer_and_controller_health_pass": 1,
            "matched_effect_family_pass": 0,
            "matched_effect_failed_roles": ["d"],
            "null_family_pass": 1,
            "null_positive_roles": [],
        }
    )
    assert effect["decision"] == H1GradientControlDecision.H1_EFFECT_AUDIT_INCONCLUSIVE.value

    value = _decision(
        confirmation={
            "evaluation_status": "evaluated",
            "passed": 0,
            "optimizer_and_controller_health_pass": 1,
            "matched_effect_family_pass": 1,
            "null_family_pass": 1,
            "null_positive_roles": [],
            "classification_passing_seed_count": 2,
        }
    )
    assert value["decision"] == H1GradientControlDecision.H1_DENSITY_RATIO_VALUE_ONLY.value


def test_required_gates_and_controls_only_claim():
    report = evaluate_gradient_control_workflow(
        provenance=_pass(),
        controller_preflight=_pass(),
        preflight=_pass(),
        pilot_panel_power=_pass(),
        pilot={"evaluation_status": "evaluated", "passed": 1},
        confirmation_panel_power=_pass(),
        confirmation={
            "evaluation_status": "evaluated",
            "passed": 1,
            "optimizer_and_controller_health_pass": 1,
            "matched_effect_family_pass": 1,
            "null_positive_roles": [],
        },
        require_gate="controls",
    )
    assert report["required_gate_pass"] == 1
    assert report["physical_training_performed"] == 0
    assert report["sampling_performed"] == 0
    blocked = evaluate_gradient_control_workflow(
        provenance=_fail(),
        controller_preflight=_pass(),
        preflight=_pass(),
        pilot_panel_power=_pass(),
        pilot={"evaluation_status": "evaluated", "passed": 1},
        confirmation_panel_power=_pass(),
        confirmation={"evaluation_status": "evaluated", "passed": 1},
        require_gate="controls",
    )
    assert blocked["required_gate_pass"] == 0
    with pytest.raises(ValueError):
        evaluate_gradient_control_workflow(
            provenance=1,
            controller_preflight=1,
            preflight=1,
            pilot_panel_power=1,
            pilot={},
            confirmation_panel_power=1,
            confirmation={},
            require_gate="wrong",
        )
