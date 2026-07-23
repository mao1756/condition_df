from __future__ import annotations

import copy
import json

import pytest

from mnist.d0_score_boundary_control_gate import (
    BoundaryControlDecision,
    BoundaryControlThresholds,
    checkpoint_is_dual_bank_eligible,
    decide_control_repair,
    evaluate_boundary_control_gate,
    evaluate_boundary_control_gates,
    evaluate_boundary_preflight,
    evaluate_implicit_teacher_seed,
    evaluate_implicit_teacher_study,
    evaluate_null_seed,
    evaluate_null_study,
    evaluate_supervised_teacher,
    select_dual_bank_checkpoint,
)


def _thresholds(**overrides: object) -> BoundaryControlThresholds:
    values: dict[str, object] = {
        "expected_implicit_teacher_seeds": 3,
        "minimum_passing_implicit_teacher_seeds": 2,
        "expected_null_seeds": 3,
    }
    values.update(overrides)
    return BoundaryControlThresholds(**values)


def _selection_bank(*, lower_bound: float, risk: float) -> dict[str, object]:
    return {
        "overall": {
            "lower_bound": lower_bound,
            "model_score_risk": risk,
        },
        "data_end": {
            "lower_bound": lower_bound,
            "model_score_risk": risk + 0.25,
        },
    }


def _checkpoint(
    step: int,
    *,
    lower_a: float = 0.2,
    lower_b: float = 0.3,
    risk_a: float = -2.0,
    risk_b: float = -1.0,
) -> dict[str, object]:
    return {
        "step": step,
        "finite": 1,
        "banks": {
            "a": _selection_bank(lower_bound=lower_a, risk=risk_a),
            "b": _selection_bank(lower_bound=lower_b, risk=risk_b),
        },
    }


def _teacher_metrics(seed: int = 101) -> dict[str, object]:
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
        "post_warmup_clip_fraction": 0.05,
        "audit_objective_banks": {
            "a": {
                "overall": {"lower_bound": 0.2},
                "data_end": {"lower_bound": 0.1},
            },
            "b": {
                "overall": {"lower_bound": 0.15},
                "data_end": {"lower_bound": 0.05},
            },
        },
    }


def _null_metrics(seed: int = 201) -> dict[str, object]:
    return {
        "model_seed": seed,
        "complete": 1,
        "finite": 1,
        "selected_step": 0,
        "comparator": "analytic_zero_step0",
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": 0.05,
        "audit_objective_banks": {
            "a": {
                "overall": {"lower_bound": -0.2},
                "data_end": {"lower_bound": 0.0},
            },
            "b": {
                "overall": {"lower_bound": -0.1},
                "data_end": {"lower_bound": -0.05},
            },
        },
    }


def _preflight_metrics() -> dict[str, object]:
    return {
        "potential_finite": 1,
        "gradient_finite": 1,
        "hvp_finite": 1,
        "generator_finite": 1,
        "energy_finite": 1,
        "incident_flux_loglog_slope": 0.95,
        "incident_flux_endpoint_ratio": 5e-4,
        "legacy_barrier_rejected": 1,
        "legacy_coefficient_relative_error": 0.01,
        "operator_pass": 1,
    }


def test_dual_bank_checkpoint_selection_includes_step_zero_and_uses_both_banks() -> None:
    step_zero = {"step": 0, "finite": 1, "mean_selection_risk": 0.0}
    one_bank_failure = _checkpoint(250, lower_b=0.0, risk_a=-100.0, risk_b=-100.0)
    passing = _checkpoint(500, risk_a=-4.0, risk_b=-2.0)
    tied_later = _checkpoint(750, risk_a=-4.0, risk_b=-2.0)

    assert checkpoint_is_dual_bank_eligible(step_zero)
    assert not checkpoint_is_dual_bank_eligible(one_bank_failure)
    assert checkpoint_is_dual_bank_eligible(passing)
    selected = select_dual_bank_checkpoint(
        [tied_later, one_bank_failure, step_zero, passing]
    )
    assert selected["selected_step"] == 500
    assert selected["comparator"] == "analytic_zero_step0"
    records = {int(row["step"]): row for row in selected["records"]}
    assert records[0]["selection_eligible"] == 1
    assert records[250]["selection_eligible"] == 0

    with pytest.raises(ValueError, match="step zero"):
        select_dual_bank_checkpoint([passing])


def test_nonzero_checkpoint_requires_strictly_positive_bounds_in_every_scope() -> None:
    for bank in ("a", "b"):
        for scope in ("overall", "data_end"):
            value = _checkpoint(500)
            value["banks"][bank][scope]["lower_bound"] = 0.0  # type: ignore[index]
            assert not checkpoint_is_dual_bank_eligible(value)
    assert not checkpoint_is_dual_bank_eligible({"step": 500, "finite": 0})
    assert not checkpoint_is_dual_bank_eligible({"step": -1})


def test_boundary_preflight_checks_smooth_flux_decay_and_legacy_rejection() -> None:
    gate = evaluate_boundary_preflight(_preflight_metrics(), _thresholds())
    assert gate["passed"] == 1
    assert gate["subchecks"]["incident_flux_slope"]["passed"] == 1
    assert gate["subchecks"]["legacy_barrier_rejected"]["passed"] == 1

    no_decay = _preflight_metrics()
    no_decay["incident_flux_loglog_slope"] = 0.89
    assert evaluate_boundary_preflight(no_decay, _thresholds())["passed"] == 0
    legacy_accepted = _preflight_metrics()
    legacy_accepted["legacy_barrier_rejected"] = 0
    assert evaluate_boundary_preflight(legacy_accepted, _thresholds())["passed"] == 0


def test_supervised_and_implicit_teacher_gates_aggregate_all_time_bins_and_banks() -> None:
    thresholds = _thresholds()
    metrics = _teacher_metrics()
    assert evaluate_supervised_teacher(metrics, thresholds)["passed"] == 1
    assert evaluate_implicit_teacher_seed(metrics, thresholds)["passed"] == 1

    missing_bin = copy.deepcopy(metrics)
    missing_bin["time_bin_flux_cosines"] = [0.96] * 4
    assert evaluate_supervised_teacher(missing_bin, thresholds)["passed"] == 0

    failed_bank = copy.deepcopy(metrics)
    failed_bank["audit_objective_banks"]["b"]["data_end"]["lower_bound"] = 0.0  # type: ignore[index]
    implicit = evaluate_implicit_teacher_seed(failed_bank, thresholds)
    assert implicit["passed"] == 0
    assert implicit["subchecks"]["audit_objective_bank_b"]["passed"] == 0
    # The exact-score supervised representation check intentionally does not
    # depend on stochastic objective banks.
    assert evaluate_supervised_teacher(failed_bank, thresholds)["passed"] == 1


def test_implicit_teacher_study_requires_distinct_seed_count_and_two_passes() -> None:
    thresholds = _thresholds()
    rows = [_teacher_metrics(seed) for seed in (11, 12, 13)]
    rows[2]["audit_overall_score_gain"] = 0.0
    study = evaluate_implicit_teacher_study(rows, thresholds)
    assert study["passed"] == 1
    assert study["passing_seed_count"] == 2

    rows[1]["audit_overall_score_gain"] = 0.0
    assert evaluate_implicit_teacher_study(rows, thresholds)["passed"] == 0
    duplicate = [_teacher_metrics(11), _teacher_metrics(11), _teacher_metrics(13)]
    assert evaluate_implicit_teacher_study(duplicate, thresholds)["passed"] == 0


def test_null_uses_analytic_zero_and_requires_step_zero_in_both_audit_banks() -> None:
    metrics = _null_metrics()
    assert evaluate_null_seed(metrics)["passed"] == 1

    legacy_linear = copy.deepcopy(metrics)
    legacy_linear["comparator"] = "frozen_training_only_linear_spline_step0"
    assert evaluate_null_seed(legacy_linear)["passed"] == 0
    nonzero = copy.deepcopy(metrics)
    nonzero["selected_step"] = 500
    assert evaluate_null_seed(nonzero)["passed"] == 0
    false_positive = copy.deepcopy(metrics)
    false_positive["audit_objective_banks"]["a"]["overall"]["lower_bound"] = 1e-12  # type: ignore[index]
    assert evaluate_null_seed(false_positive)["passed"] == 0

    study = evaluate_null_study(
        [_null_metrics(seed) for seed in (21, 22, 23)], _thresholds()
    )
    assert study["passed"] == 1
    failed_rows = [_null_metrics(seed) for seed in (21, 22, 23)]
    failed_rows[1]["selected_step"] = 500
    assert evaluate_null_study(failed_rows, _thresholds())["passed"] == 0


@pytest.mark.parametrize(
    ("values", "agree", "expected"),
    [
        ((False, True, True, True, True), True, BoundaryControlDecision.CONTROL_PROVENANCE_INVALID),
        ((True, False, True, True, True), True, BoundaryControlDecision.BOUNDARY_DOMAIN_INVALID),
        ((True, True, False, True, True), True, BoundaryControlDecision.REPRESENTATION_INVALID),
        ((True, True, True, True, True), False, BoundaryControlDecision.TRACE_ESTIMATOR_INCONCLUSIVE),
        ((True, True, True, False, True), True, BoundaryControlDecision.IMPLICIT_OBJECTIVE_UNSTABLE),
        ((True, True, True, True, False), True, BoundaryControlDecision.IMPLICIT_OBJECTIVE_UNSTABLE),
        ((True, True, True, True, True), True, BoundaryControlDecision.CONTROL_PIPELINE_REPAIRED),
    ],
)
def test_closed_control_repair_decisions(
    values: tuple[bool, bool, bool, bool, bool],
    agree: bool,
    expected: BoundaryControlDecision,
) -> None:
    provenance, preflight, supervised, implicit, null = values
    result = decide_control_repair(
        provenance_pass=provenance,
        boundary_preflight=preflight,
        supervised_teacher=supervised,
        implicit_teacher_study=implicit,
        null_study=null,
        probe_banks_agree=agree,
    )
    assert result["decision"] == expected.value
    assert result["physical_training_authorized"] == int(
        expected is BoundaryControlDecision.CONTROL_PIPELINE_REPAIRED
    )
    assert result["sampling_authorized"] == 0
    assert result["sampling_performed"] == 0


def test_required_gates_are_cumulative_and_fail_closed() -> None:
    components = {
        "provenance_pass": {"passed": 1},
        "boundary_preflight": {"passed": 1},
        "supervised_teacher": {"passed": 1},
        "implicit_teacher_study": {"passed": 1},
        "null_study": {"passed": 1},
    }
    assert evaluate_boundary_control_gate(**components)["passed"] == 1
    report = evaluate_boundary_control_gates(
        **components, require_gate="controls"
    )
    assert report["required_gate_pass"] == 1
    assert report["preflight_pass"] == 1
    assert report["decision"]["decision"] == BoundaryControlDecision.CONTROL_PIPELINE_REPAIRED.value
    json.dumps(report, allow_nan=False)

    failed = dict(components)
    failed["supervised_teacher"] = {"passed": 0}
    preflight_only = evaluate_boundary_control_gates(
        **failed, require_gate="preflight"
    )
    assert preflight_only["required_gate_pass"] == 1
    controls = evaluate_boundary_control_gates(
        **failed, require_gate="controls"
    )
    assert controls["required_gate_pass"] == 0
    assert controls["decision"]["decision"] == BoundaryControlDecision.REPRESENTATION_INVALID.value

    with pytest.raises(ValueError, match="require_gate"):
        evaluate_boundary_control_gates(**components, require_gate="sampling")


def test_threshold_defaults_and_validation_are_frozen() -> None:
    thresholds = BoundaryControlThresholds()
    assert thresholds.expected_implicit_teacher_seeds == 3
    assert thresholds.minimum_passing_implicit_teacher_seeds == 2
    assert thresholds.expected_null_seeds == 3
    assert thresholds.maximum_post_warmup_clip_fraction == pytest.approx(0.10)
    assert thresholds.boundary_min_flux_slope == pytest.approx(0.90)
    assert thresholds.boundary_max_flux_ratio == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="passing implicit-teacher"):
        BoundaryControlThresholds(
            expected_implicit_teacher_seeds=2,
            minimum_passing_implicit_teacher_seeds=3,
        )
    with pytest.raises(ValueError, match="bootstrap_confidence"):
        BoundaryControlThresholds(bootstrap_confidence=1.0)
