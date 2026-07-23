from __future__ import annotations

import copy
import json

import pytest

from mnist.d0_score_boundary_control_gate import (
    BoundaryControlThresholds,
    evaluate_supervised_teacher,
)
from mnist.d0_score_control_scale_repair_gate import (
    ProbeBankStatus,
    ScaleRepairDecision,
    classify_probe_bank_status,
    decide_scale_repair,
    evaluate_loss_scale_calibration,
    evaluate_optimizer_task_health,
    evaluate_scale_repair_gates,
    not_evaluated_study,
    split_supervised_teacher_gate,
)


def _teacher_metrics() -> dict[str, object]:
    return {
        "complete": 1,
        "finite": 1,
        "selected_step": 3750,
        "audit_overall_score_gain": 0.99849,
        "audit_data_end_score_gain": 0.99885,
        "overall_flux_cosine": 0.99925,
        "time_bin_flux_cosines": [0.9989, 0.9990, 0.9991, 0.9992, 0.9994],
        "overall_relative_flux_l2": 0.03875,
        "time_bin_relative_flux_l2": [0.047, 0.043, 0.040, 0.037, 0.034],
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": 0.05,
    }


def _calibration(raw_norm: float = 8.0, target: float = 0.1) -> dict[str, object]:
    scale = min(1.0, target / raw_norm)
    return {
        "complete": 1,
        "finite": 1,
        "training_only": 1,
        "state_count": 256,
        "unscaled_initial_grad_norm": raw_norm,
        "initial_grad_target": target,
        "loss_scale": scale,
        "scaled_initial_gradient_norm": raw_norm * scale,
        "objective_kind": "supervised_teacher",
        "calibration_split": "train",
        "calibration_state_sha256": "states-sha256",
        "binding": {"scientific_fingerprint": "science"},
    }


def _evaluated(passed: bool = True) -> dict[str, object]:
    return {"evaluation_status": "evaluated", "passed": int(passed)}


def _passing_components() -> dict[str, object]:
    return {
        "provenance_pass": _evaluated(),
        "boundary_preflight": _evaluated(),
        "supervised_calibration": _evaluated(),
        "implicit_calibration": _evaluated(),
        "supervised_optimizer": _evaluated(),
        "supervised_representation": _evaluated(),
        "downstream_optimizer": _evaluated(),
        "implicit_teacher_study": _evaluated(),
        "null_study": _evaluated(),
        "probe_bank_status": ProbeBankStatus.AGREE,
    }


def test_calibration_checks_exact_training_only_formula() -> None:
    gate = evaluate_loss_scale_calibration(
        _calibration(), expected_initial_grad_target=0.1
    )
    assert gate["passed"] == 1
    assert gate["expected_loss_scale"] == pytest.approx(0.0125)

    validation_leak = _calibration()
    validation_leak["training_only"] = 0
    assert (
        evaluate_loss_scale_calibration(
            validation_leak, expected_initial_grad_target=0.1
        )["passed"]
        == 0
    )

    wrong_formula = _calibration()
    wrong_formula["loss_scale"] = 0.02
    bad = evaluate_loss_scale_calibration(
        wrong_formula, expected_initial_grad_target=0.1
    )
    assert bad["passed"] == 0
    assert bad["subchecks"]["loss_scale_formula"]["passed"] == 0


def test_clipping_only_failure_is_optimizer_scale_invalid_not_representation() -> None:
    metrics = _teacher_metrics()
    metrics["post_warmup_clip_fraction"] = 0.989714
    legacy = evaluate_supervised_teacher(metrics)
    assert legacy["passed"] == 0

    split = split_supervised_teacher_gate(legacy)
    assert split["optimizer"]["passed"] == 0
    assert split["representation"]["passed"] == 1

    components = _passing_components()
    components["supervised_optimizer"] = split["optimizer"]
    components["supervised_representation"] = split["representation"]
    decision = decide_scale_repair(**components)
    assert decision["decision"] == ScaleRepairDecision.OPTIMIZER_SCALE_INVALID.value
    assert decision["physical_training_authorized"] == 0


def test_healthy_optimizer_with_bad_scientific_fit_is_representation_invalid() -> None:
    metrics = _teacher_metrics()
    metrics["overall_flux_cosine"] = 0.97
    split = split_supervised_teacher_gate(evaluate_supervised_teacher(metrics))
    assert split["optimizer"]["passed"] == 1
    assert split["representation"]["passed"] == 0

    components = _passing_components()
    components["supervised_optimizer"] = split["optimizer"]
    components["supervised_representation"] = split["representation"]
    assert (
        decide_scale_repair(**components)["decision"]
        == ScaleRepairDecision.REPRESENTATION_INVALID.value
    )

    # The shared implicit calibration is intentionally deferred until the
    # supervised representation passes; its absence must not mask this
    # scientific representation outcome as an optimizer failure.
    components["implicit_calibration"] = not_evaluated_study(
        "implicit_calibration", "awaiting supervised representation"
    )
    assert (
        decide_scale_repair(**components)["decision"]
        == ScaleRepairDecision.REPRESENTATION_INVALID.value
    )


def test_invalid_supervised_or_implicit_calibration_is_optimizer_scale_invalid() -> None:
    components = _passing_components()
    components["supervised_calibration"] = _evaluated(False)
    assert (
        decide_scale_repair(**components)["decision"]
        == ScaleRepairDecision.OPTIMIZER_SCALE_INVALID.value
    )

    components = _passing_components()
    components["implicit_calibration"] = _evaluated(False)
    assert (
        decide_scale_repair(**components)["decision"]
        == ScaleRepairDecision.OPTIMIZER_SCALE_INVALID.value
    )


def test_split_fails_closed_when_a_required_legacy_subcheck_is_missing() -> None:
    legacy = evaluate_supervised_teacher(_teacher_metrics())
    missing_optimizer = copy.deepcopy(legacy)
    del missing_optimizer["subchecks"]["finite"]
    assert split_supervised_teacher_gate(missing_optimizer)["optimizer"]["passed"] == 0

    missing_science = copy.deepcopy(legacy)
    del missing_science["subchecks"]["overall_flux_cosine"]
    assert (
        split_supervised_teacher_gate(missing_science)["representation"]["passed"]
        == 0
    )


def test_optimizer_health_uses_unchanged_clip_threshold_and_skips_fail_closed() -> None:
    threshold = BoundaryControlThresholds().maximum_post_warmup_clip_fraction
    assert threshold == pytest.approx(0.10)
    assert (
        evaluate_optimizer_task_health(
            {
                "evaluation_status": "evaluated",
                "complete": 1,
                "finite": 1,
                "post_warmup_clip_fraction": threshold,
            }
        )["passed"]
        == 1
    )
    assert (
        evaluate_optimizer_task_health(
            {
                "evaluation_status": "evaluated",
                "complete": 1,
                "finite": 1,
                "post_warmup_clip_fraction": threshold + 1e-12,
            }
        )["passed"]
        == 0
    )
    skipped = evaluate_optimizer_task_health(
        {
            "evaluation_status": "not_evaluated",
            "complete": 1,
            "finite": 1,
            "post_warmup_clip_fraction": 0.0,
        }
    )
    assert skipped["evaluation_status"] == "not_evaluated"
    assert skipped["passed"] == 0


@pytest.mark.parametrize(
    ("evaluated", "agree", "expected"),
    [
        (False, None, ProbeBankStatus.NOT_EVALUATED),
        (False, True, ProbeBankStatus.NOT_EVALUATED),
        (True, None, ProbeBankStatus.NOT_EVALUATED),
        (True, True, ProbeBankStatus.AGREE),
        (True, False, ProbeBankStatus.DISAGREE),
    ],
)
def test_probe_bank_status_is_explicit_tri_state(
    evaluated: bool, agree: bool | None, expected: ProbeBankStatus
) -> None:
    assert (
        classify_probe_bank_status(
            studies_evaluated=evaluated, banks_agree=agree
        )
        is expected
    )


def test_skipped_studies_and_default_probe_status_fail_closed() -> None:
    components = _passing_components()
    components["implicit_teacher_study"] = not_evaluated_study(
        "implicit_teacher_study", "supervised prerequisite failed"
    )
    components["null_study"] = not_evaluated_study(
        "null_study", "supervised prerequisite failed"
    )
    components["downstream_optimizer"] = not_evaluated_study(
        "downstream_optimizer", "no downstream task ran"
    )
    components["probe_bank_status"] = ProbeBankStatus.NOT_EVALUATED

    report = evaluate_scale_repair_gates(
        **components, require_gate="controls"
    )
    assert report["required_gate_pass"] == 0
    assert report["controls"]["passed"] == 0
    assert report["controls"]["probe_bank_status"] == "not_evaluated"
    assert (
        report["decision"]["decision"]
        == ScaleRepairDecision.IMPLICIT_OBJECTIVE_UNSTABLE.value
    )
    assert report["decision"]["studies_evaluated"] == 0
    assert report["decision"]["physical_training_authorized"] == 0
    json.dumps(report, allow_nan=False)


def test_completed_disagreeing_banks_are_inconclusive_and_agreement_repairs() -> None:
    components = _passing_components()
    components["probe_bank_status"] = ProbeBankStatus.DISAGREE
    disagreement = evaluate_scale_repair_gates(
        **components, require_gate="controls"
    )
    assert disagreement["required_gate_pass"] == 0
    assert (
        disagreement["decision"]["decision"]
        == ScaleRepairDecision.TRACE_ESTIMATOR_INCONCLUSIVE.value
    )

    components["probe_bank_status"] = ProbeBankStatus.AGREE
    repaired = evaluate_scale_repair_gates(
        **components, require_gate="controls"
    )
    assert repaired["schema_version"] == 2
    assert repaired["required_gate_pass"] == 1
    assert (
        repaired["decision"]["decision"]
        == ScaleRepairDecision.CONTROL_PIPELINE_REPAIRED.value
    )
    assert repaired["decision"]["physical_training_authorized"] == 1
    assert repaired["sampling_performed"] == 0


def test_completed_downstream_clipping_failure_is_optimizer_scale_invalid() -> None:
    components = _passing_components()
    components["downstream_optimizer"] = _evaluated(False)
    result = decide_scale_repair(**components)
    assert result["decision"] == ScaleRepairDecision.OPTIMIZER_SCALE_INVALID.value


def test_preflight_requirement_does_not_promote_unevaluated_controls() -> None:
    components = _passing_components()
    components["implicit_teacher_study"] = not_evaluated_study(
        "implicit_teacher_study", "not run in preflight stage"
    )
    components["null_study"] = not_evaluated_study(
        "null_study", "not run in preflight stage"
    )
    components["probe_bank_status"] = ProbeBankStatus.NOT_EVALUATED
    report = evaluate_scale_repair_gates(
        **components, require_gate="preflight"
    )
    assert report["required_gate_pass"] == 1
    assert report["controls"]["passed"] == 0
    assert report["decision"]["physical_training_authorized"] == 0

    with pytest.raises(ValueError, match="require_gate"):
        evaluate_scale_repair_gates(**components, require_gate="sampling")
    with pytest.raises(ValueError, match="probe_bank_status"):
        evaluate_scale_repair_gates(
            **{**components, "probe_bank_status": "missing"}, require_gate="none"
        )
