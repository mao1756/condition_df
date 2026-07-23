from __future__ import annotations

import copy
import json

import pytest

from mnist.d0_score_density_ratio_stability_gate import (
    RatioStabilityDecision,
    RatioStabilityThresholds,
    decide_ratio_stability,
    evaluate_null_seed,
    evaluate_null_study,
    evaluate_paired_ratio_preflight,
    evaluate_ratio_stability_controls,
    evaluate_ratio_stability_workflow,
    evaluate_stability_pilot,
    evaluate_stability_pilot_candidate,
    evaluate_teacher_seed,
    evaluate_teacher_study,
)


def _scope(risk: float, lower: float) -> dict[str, float]:
    return {"risk": risk, "objective_improvement_lower_bound": lower}


def _checkpoint(
    step: int,
    *,
    a_risk: float = 0.60,
    a_lower: float = 0.01,
    b_risk: float = 0.61,
    b_lower: float = 0.01,
) -> dict[str, object]:
    return {
        "step": step,
        "finite": 1,
        "ema": 1,
        "panels": {
            "a": {
                "confidence": 0.90,
                "overall": _scope(a_risk, a_lower),
                "data_end": _scope(a_risk, a_lower),
            },
            "b": {
                "confidence": 0.90,
                "overall": _scope(b_risk, b_lower),
                "data_end": _scope(b_risk, b_lower),
            },
        },
    }


def _audit_panel(*, lower: float = 0.01) -> dict[str, object]:
    return {
        "finite": 1,
        "path_count": 32,
        "anchors_per_path": 32,
        "confidence": 0.90,
        "classification_improvement": {
            "overall": {"objective_improvement_lower_bound": lower},
            "data_end": {"objective_improvement_lower_bound": lower},
        },
        "audit_overall_score_gain": 0.90,
        "audit_data_end_score_gain": 0.90,
        "overall_flux_cosine": 0.98,
        "time_bin_flux_cosines": [0.95] * 5,
        "overall_relative_flux_l2": 0.15,
        "time_bin_relative_flux_l2": [0.20] * 5,
    }


def _analytic_pilot() -> dict[str, object]:
    return {
        "overall": {
            "score_gain": 0.10,
            "flux_cosine": 0.20,
            "flux_relative_l2": 0.90,
        },
        "data_end": {
            "score_gain": 0.15,
            "flux_cosine": 0.25,
            "flux_relative_l2": 0.80,
        },
    }


def _clips(value: float = 0.10) -> dict[str, float]:
    return {
        "post_warmup_clip_fraction": value,
        "final_500_clip_fraction": value,
        "final_200_clip_fraction": value,
    }


def _teacher(seed: int = 1, *, a_risk: float = 0.50) -> dict[str, object]:
    return {
        "model_seed": seed,
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        **_clips(),
        "checkpoints": [
            _checkpoint(0),
            _checkpoint(100, a_risk=a_risk, b_risk=0.51, b_lower=0.01),
        ],
        "selected_analytic_metrics": _analytic_pilot(),
        "audit_panels": {"c": _audit_panel(), "d": _audit_panel()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _null(seed: int = 1) -> dict[str, object]:
    return {
        "model_seed": seed,
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        **_clips(),
        "comparator": "analytic_zero",
        "checkpoints": [
            _checkpoint(0),
            _checkpoint(
                100,
                a_risk=0.69,
                a_lower=0.0,
                b_risk=0.70,
                b_lower=0.0,
            ),
        ],
        "audit_panels": {
            "c": _audit_panel(lower=0.0),
            "d": _audit_panel(lower=0.0),
        },
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _candidate(
    level: int,
    lr: float,
    *,
    healthy: bool = True,
    teacher_risk: float = 0.50,
) -> dict[str, object]:
    teacher = _teacher(a_risk=teacher_risk)
    if not healthy:
        teacher["final_200_clip_fraction"] = 0.100001
    return {
        "evaluation_status": "evaluated",
        "accumulation_steps": level,
        "learning_rate": lr,
        "teacher": teacher,
        "null": _null(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _preflight() -> dict[str, object]:
    result: dict[str, object] = {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "preflight_paths": 128,
        "preflight_confidence": 0.99,
        "loss_algebra_max_error": 1e-13,
        "expanded_loss_max_error": 1e-8,
        "expanded_gradient_max_error": 1e-7,
        "accumulation_gradient_max_error": 1e-7,
        "parent_loss_scale_reused": 1,
        "adaptive_loss_scaling": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    for name in (
        "parent_provenance_pass",
        "mixture_coefficients_pass",
        "dirichlet_marginals_pass",
        "common_gamma_coupling_pass",
        "time_strata_pass",
        "class_balance_pass",
        "stream_replay_pass",
        "candidate_order_invariance_pass",
        "fresh_panel_isolation_pass",
        "simultaneous_loss_interval_contains_zero",
        "simultaneous_directional_gradient_intervals_contain_zero",
        "boundary_operator_pass",
        "device_smoke_pass",
    ):
        result[name] = 1
    return result


def test_frozen_defaults_include_hierarchy_and_all_clip_windows() -> None:
    thresholds = RatioStabilityThresholds()
    assert thresholds.accumulation_levels == (2, 4, 8)
    assert thresholds.pilot_learning_rates == (3e-5, 1e-5)
    assert thresholds.final_clip_windows == (500, 200)
    assert thresholds.maximum_clip_fraction == 0.10
    assert thresholds.density_ratio.teacher.teacher_min_score_gain == 0.90
    assert thresholds.density_ratio.teacher.teacher_min_overall_flux_cosine == 0.98
    assert thresholds.density_ratio.teacher.teacher_max_overall_relative_flux_l2 == 0.15


def test_preflight_requires_every_exactness_and_isolation_check() -> None:
    metrics = _preflight()
    gate = evaluate_paired_ratio_preflight(metrics)
    assert gate["passed"] == 1
    assert gate["variance_forensics_gate_eligible"] == 0
    for key in (
        "common_gamma_coupling_pass",
        "simultaneous_directional_gradient_intervals_contain_zero",
        "parent_loss_scale_reused",
    ):
        broken = copy.deepcopy(metrics)
        broken[key] = 0
        assert evaluate_paired_ratio_preflight(broken)["passed"] == 0


@pytest.mark.parametrize(
    "window",
    [
        "post_warmup_clip_fraction",
        "final_500_clip_fraction",
        "final_200_clip_fraction",
    ],
)
def test_pilot_clipping_is_inclusive_at_point_one_and_fails_above(window: str) -> None:
    candidate = _candidate(2, 3e-5)
    assert evaluate_stability_pilot_candidate(candidate)["passed"] == 1
    candidate["teacher"][window] = 0.10000001
    gate = evaluate_stability_pilot_candidate(candidate)
    assert gate["passed"] == 0
    assert gate["subchecks"][f"teacher_{window.replace('_clip_fraction', '')}_clip_fraction"]["passed"] == 0


def test_hierarchy_stops_at_first_passing_level_and_rejects_later_peeking() -> None:
    level_two = [_candidate(2, 3e-5), _candidate(2, 1e-5)]
    gate = evaluate_stability_pilot(level_two)
    assert gate["passed"] == 1
    assert gate["selected_accumulation_level"] == 2

    peeked = [*level_two, _candidate(4, 3e-5), _candidate(4, 1e-5)]
    gate = evaluate_stability_pilot(peeked)
    assert gate["passed"] == 0
    assert gate["subchecks"]["hierarchical_order"]["passed"] == 0


def test_hierarchy_escalates_only_after_a_complete_failed_level() -> None:
    candidates = [
        _candidate(2, 3e-5, healthy=False),
        _candidate(2, 1e-5, healthy=False),
        _candidate(4, 3e-5, healthy=True, teacher_risk=0.49),
        _candidate(4, 1e-5, healthy=True, teacher_risk=0.48),
    ]
    gate = evaluate_stability_pilot(candidates)
    assert gate["passed"] == 1
    assert gate["selected_accumulation_level"] == 4
    assert gate["selected_profile"]["profile"]["learning_rate"] == pytest.approx(1e-5)

    incomplete = candidates[1:]
    gate = evaluate_stability_pilot(incomplete)
    assert gate["passed"] == 0
    assert gate["subchecks"]["hierarchical_order"]["passed"] == 0


def test_all_three_failed_levels_is_terminal_unresolved_pilot() -> None:
    candidates = [
        _candidate(level, lr, healthy=False)
        for level in (2, 4, 8)
        for lr in (3e-5, 1e-5)
    ]
    gate = evaluate_stability_pilot(candidates)
    assert gate["evaluation_status"] == "evaluated"
    assert gate["passed"] == 0
    assert gate["all_levels_complete"] == 1
    assert gate["selected_profile"]["selected"] == 0


def test_confirmation_preserves_strict_science_and_all_clip_windows() -> None:
    passing = [_teacher(seed) for seed in (10, 11)]
    science_failure = _teacher(12)
    science_failure["audit_panels"]["c"]["overall_flux_cosine"] = 0.979999
    science_failure["audit_panels"]["d"]["overall_flux_cosine"] = 0.979999
    assert evaluate_teacher_seed(passing[0])["passed"] == 1
    study = evaluate_teacher_study([*passing, science_failure])
    assert study["passed"] == 1
    assert study["passing_seed_count"] == 2

    unhealthy = copy.deepcopy(passing[0])
    unhealthy["final_500_clip_fraction"] = 0.100001
    assert evaluate_teacher_seed(unhealthy)["passed"] == 0
    # All teacher optimizers must be healthy even when two scientific seeds pass.
    assert evaluate_teacher_study([passing[0], passing[1], unhealthy])["passed"] == 0


def test_null_remains_independent_state_false_discovery_control() -> None:
    assert evaluate_null_seed(_null())["passed"] == 1
    bad = _null()
    bad["audit_panels"]["d"]["classification_improvement"]["overall"][
        "objective_improvement_lower_bound"
    ] = 1e-12
    gate = evaluate_null_seed(bad)
    assert gate["passed"] == 0
    assert gate["false_discovery"] == 1
    assert evaluate_null_study([_null(seed) for seed in (1, 2, 3)])["passed"] == 1


def test_closed_decisions_and_authorization() -> None:
    provenance = {"evaluation_status": "evaluated", "passed": 1}
    preflight = {"evaluation_status": "evaluated", "passed": 1}
    pilot = {"evaluation_status": "evaluated", "passed": 1}
    base = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "optimizer_health_pass": 1,
        "teacher_study": {
            "classification_passing_seed_count": 0,
            "panel_disagreement": 0,
        },
        "null_study": {"false_discovery_count": 0},
    }
    assert decide_ratio_stability(
        provenance={"evaluation_status": "evaluated", "passed": 0},
        preflight=preflight,
        pilot=pilot,
        controls=base,
    )["decision"] == RatioStabilityDecision.CONTROL_PROVENANCE_INVALID.value
    assert decide_ratio_stability(
        provenance=provenance,
        preflight={"evaluation_status": "evaluated", "passed": 0},
        pilot=pilot,
        controls=base,
    )["decision"] == RatioStabilityDecision.PAIRED_RATIO_ESTIMATOR_INVALID.value
    assert decide_ratio_stability(
        provenance=provenance,
        preflight=preflight,
        pilot={"evaluation_status": "evaluated", "passed": 0},
        controls=base,
    )["decision"] == RatioStabilityDecision.CLASSIFICATION_VARIANCE_REDUCTION_UNRESOLVED.value

    false = copy.deepcopy(base)
    false["null_study"]["false_discovery_count"] = 1
    assert decide_ratio_stability(
        provenance=provenance, preflight=preflight, pilot=pilot, controls=false
    )["decision"] == RatioStabilityDecision.SELECTION_FALSE_DISCOVERY.value

    value_only = copy.deepcopy(base)
    value_only["teacher_study"]["classification_passing_seed_count"] = 2
    assert decide_ratio_stability(
        provenance=provenance, preflight=preflight, pilot=pilot, controls=value_only
    )["decision"] == RatioStabilityDecision.DENSITY_RATIO_VALUE_ONLY.value

    audit_disagreement = copy.deepcopy(base)
    audit_disagreement["teacher_study"]["panel_disagreement"] = 1
    assert decide_ratio_stability(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        controls=audit_disagreement,
    )["decision"] == RatioStabilityDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE.value

    assert decide_ratio_stability(
        provenance=provenance, preflight=preflight, pilot=pilot, controls=base
    )["decision"] == RatioStabilityDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL.value

    repaired = copy.deepcopy(base)
    repaired["passed"] = 1
    decision = decide_ratio_stability(
        provenance=provenance, preflight=preflight, pilot=pilot, controls=repaired
    )
    assert decision["decision"] == RatioStabilityDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value
    assert decision["physical_training_authorized"] == 1
    assert decision["sampling_authorized"] == decision["sampling_performed"] == 0


def test_not_evaluated_stages_are_pending_not_scientific_failures() -> None:
    provenance = {"evaluation_status": "evaluated", "passed": 1}
    preflight = {"evaluation_status": "evaluated", "passed": 1}
    pilot = {"evaluation_status": "evaluated", "passed": 1}
    pending = {"evaluation_status": "not_evaluated", "passed": 0}

    before_preflight = decide_ratio_stability(
        provenance=provenance,
        preflight=pending,
        pilot=pending,
        controls=pending,
    )
    assert before_preflight["decision"] == "paired_ratio_preflight_not_evaluated"
    assert before_preflight["closed_terminal_scientific_outcome"] == 0

    after_preflight = decide_ratio_stability(
        provenance=provenance,
        preflight=preflight,
        pilot=pending,
        controls=pending,
    )
    assert after_preflight["decision"] == "paired_ratio_preflight_passed"
    assert after_preflight["interim_stage_success"] == 1

    after_pilot = decide_ratio_stability(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        controls=pending,
    )
    assert after_pilot["decision"] == "paired_ratio_pilot_passed"
    assert after_pilot["interim_stage_success"] == 1
    assert after_pilot["physical_training_authorized"] == 0

    evaluated_unhealthy = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "optimizer_health_pass": 0,
        "teacher_study": {},
        "null_study": {},
    }
    failed_confirmation = decide_ratio_stability(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        controls=evaluated_unhealthy,
    )
    assert failed_confirmation["decision"] == (
        RatioStabilityDecision.CLASSIFICATION_OPTIMIZER_INVALID.value
    )
    assert failed_confirmation["closed_terminal_scientific_outcome"] == 1


def test_required_gate_and_all_artifacts_are_strict_json() -> None:
    candidates = [_candidate(2, 3e-5), _candidate(2, 1e-5)]
    pilot = evaluate_stability_pilot(candidates)
    report = evaluate_ratio_stability_workflow(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight=evaluate_paired_ratio_preflight(_preflight()),
        pilot=pilot,
        teacher_results=[_teacher(seed) for seed in (1, 2, 3)],
        null_results=[_null(seed) for seed in (1, 2, 3)],
        require_gate="controls",
    )
    assert report["required_gate_pass"] == 1
    json.dumps(report, allow_nan=False)


def test_confirmation_requires_paired_teacher_null_seed_sets() -> None:
    controls = evaluate_ratio_stability_controls(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight={"evaluation_status": "evaluated", "passed": 1},
        pilot={"evaluation_status": "evaluated", "passed": 1},
        teacher_results=[_teacher(seed) for seed in (1, 2, 3)],
        null_results=[_null(seed) for seed in (1, 2, 4)],
    )
    assert controls["paired_teacher_null_seed_set_pass"] == 0
    assert controls["subchecks"]["paired_teacher_null_seed_set"]["passed"] == 0
    assert controls["passed"] == 0
