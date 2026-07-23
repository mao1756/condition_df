from __future__ import annotations

import copy
import json

import pytest

from mnist.d0_score_density_ratio_gate import (
    DensityRatioDecision,
    DensityRatioThresholds,
    decide_density_ratio_controls,
    evaluate_density_ratio_pilot,
    evaluate_null_seed,
    evaluate_null_study,
    evaluate_ratio_preflight,
    evaluate_teacher_seed,
    evaluate_teacher_study,
    nominate_checkpoint_on_a,
    select_density_ratio_checkpoint,
)


def _scope(risk: float, lower: float) -> dict[str, float]:
    return {
        "risk": risk,
        "objective_improvement_lower_bound": lower,
    }


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


def _audit_panel(*, classification_lower: float = 0.01) -> dict[str, object]:
    return {
        "finite": 1,
        "path_count": 32,
        "anchors_per_path": 32,
        "confidence": 0.90,
        "classification_improvement": {
            "overall": {"objective_improvement_lower_bound": classification_lower},
            "data_end": {"objective_improvement_lower_bound": classification_lower},
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


def _teacher_metrics(seed: int = 1) -> dict[str, object]:
    return {
        "model_seed": seed,
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": 0.10,
        "checkpoints": [
            _checkpoint(0),
            _checkpoint(100, a_risk=0.50, b_risk=0.51, b_lower=0.01),
        ],
        "selected_analytic_metrics": _analytic_pilot(),
        "audit_panels": {"c": _audit_panel(), "d": _audit_panel()},
    }


def _null_metrics(seed: int = 1) -> dict[str, object]:
    return {
        "model_seed": seed,
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": 0.10,
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
            "c": _audit_panel(classification_lower=0.0),
            "d": _audit_panel(classification_lower=0.0),
        },
    }


def _pilot_candidate(lr: float, *, healthy: bool = True) -> dict[str, object]:
    teacher = _teacher_metrics()
    null = _null_metrics()
    if not healthy:
        teacher["post_warmup_clip_fraction"] = 0.11
    return {"learning_rate": lr, "teacher": teacher, "null": null}


def _preflight_metrics() -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "analytic_logit_max_error": 1e-9,
        "analytic_score_max_error": 1e-9,
        "analytic_flux_max_error": 1e-9,
        "teacher_normalization_interval": {"lower": 0.99, "upper": 1.01},
        "teacher_bce_improvement_lower_bounds": {
            "overall": {"objective_improvement_lower_bound": 0.01},
            "data_end": {"objective_improvement_lower_bound": 0.01},
        },
        "oracle_bootstrap_confidence": 0.99,
        "null_bce_error": 0.0,
        "null_score_max_abs": 0.0,
        "null_flux_max_abs": 0.0,
        "class_balance_pass": 1,
        "time_strata_pass": 1,
        "null_exchangeability_pass": 1,
        "independent_class_namespaces": 1,
        "stream_replay_pass": 1,
        "panel_isolation_pass": 1,
        "boundary_admissible": 1,
        "operator_preflight_pass": 1,
        "device_smoke_pass": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def test_frozen_defaults_match_the_protocol() -> None:
    thresholds = DensityRatioThresholds()
    assert thresholds.pilot_learning_rates == (3e-4, 1e-4, 3e-5, 1e-5)
    assert thresholds.oracle_confidence == 0.99
    assert thresholds.confirm_confidence == thresholds.audit_confidence == 0.90


def test_panel_a_nomination_is_isolated_from_b_c_and_d() -> None:
    checkpoints = [_checkpoint(0), _checkpoint(100, a_risk=0.4), _checkpoint(200, a_risk=0.5)]
    expected = nominate_checkpoint_on_a(checkpoints)
    mutated = copy.deepcopy(checkpoints)
    for row in mutated:
        row["panels"]["b"] = {"arbitrary": "changed"}
        row["panels"]["c"] = {"leak": -1e9}
        row["panels"]["d"] = {"leak": 1e9}
    assert nominate_checkpoint_on_a(mutated) == expected
    assert expected["nominee_step"] == 100


def test_panel_b_tests_only_the_a_nominee_and_cannot_switch_checkpoints() -> None:
    checkpoints = [
        _checkpoint(0),
        _checkpoint(100, a_risk=0.40, b_risk=0.80, b_lower=-0.01),
        _checkpoint(200, a_risk=0.50, b_risk=0.20, b_lower=0.20),
    ]
    selected = select_density_ratio_checkpoint(checkpoints)
    assert selected["nominee_step"] == 100
    assert selected["confirmation"]["accepted"] == 0
    assert selected["selected_step"] == 0


def test_confirmation_lower_bound_is_strict_but_confidence_is_inclusive() -> None:
    checkpoints = [_checkpoint(0), _checkpoint(100, b_lower=0.0)]
    assert select_density_ratio_checkpoint(checkpoints)["selected_step"] == 0
    checkpoints[1]["panels"]["b"]["overall"]["objective_improvement_lower_bound"] = 1e-12
    checkpoints[1]["panels"]["b"]["data_end"]["objective_improvement_lower_bound"] = 1e-12
    assert select_density_ratio_checkpoint(checkpoints)["selected_step"] == 100


def test_preflight_requires_operator_and_production_device_smoke() -> None:
    metrics = _preflight_metrics()
    assert evaluate_ratio_preflight(metrics)["passed"] == 1
    metrics["operator_preflight_pass"] = 0
    assert evaluate_ratio_preflight(metrics)["passed"] == 0
    metrics["operator_preflight_pass"] = 1
    metrics["device_smoke_pass"] = 0
    assert evaluate_ratio_preflight(metrics)["passed"] == 0


def test_preflight_oracle_uses_exact_99_percent_confidence() -> None:
    metrics = _preflight_metrics()
    metrics["oracle_bootstrap_confidence"] = 0.90
    gate = evaluate_ratio_preflight(metrics)
    assert gate["passed"] == 0
    assert gate["subchecks"]["oracle_bootstrap_confidence"]["passed"] == 0


def test_pilot_allows_bad_candidates_when_one_profile_qualifies() -> None:
    candidates = [
        _pilot_candidate(3e-4, healthy=False),
        _pilot_candidate(1e-4, healthy=False),
        _pilot_candidate(3e-5, healthy=True),
        _pilot_candidate(1e-5, healthy=False),
    ]
    gate = evaluate_density_ratio_pilot(candidates)
    assert gate["passed"] == 1
    assert gate["selected_profile"]["profile"]["learning_rate"] == pytest.approx(3e-5)
    assert gate["selected_profile"]["profile"]["teacher_mean_ab_bce"] == pytest.approx(0.505)


def test_pilot_profile_ranks_mean_a_b_risk_before_clipping_and_lr() -> None:
    first = _pilot_candidate(3e-4)
    first["teacher"]["checkpoints"][1]["panels"]["a"]["overall"]["risk"] = 0.60
    first["teacher"]["checkpoints"][1]["panels"]["b"]["overall"]["risk"] = 0.10
    second = _pilot_candidate(1e-4)
    second["teacher"]["checkpoints"][1]["panels"]["a"]["overall"]["risk"] = 0.20
    second["teacher"]["checkpoints"][1]["panels"]["b"]["overall"]["risk"] = 0.30
    gate = evaluate_density_ratio_pilot(
        [
            first,
            second,
            _pilot_candidate(3e-5, healthy=False),
            _pilot_candidate(1e-5, healthy=False),
        ]
    )
    profile = gate["selected_profile"]["profile"]
    assert profile["learning_rate"] == pytest.approx(1e-4)
    assert profile["teacher_mean_ab_bce"] == pytest.approx(0.25)
    assert profile["teacher_panel_b_bce"] == pytest.approx(0.30)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("score_gain", 0.0),
        ("flux_cosine", 0.0),
        ("flux_relative_l2", 1.0),
    ],
)
def test_pilot_teacher_analytic_qualification_is_strict(key: str, value: float) -> None:
    candidate = _pilot_candidate(3e-4)
    candidate["teacher"]["selected_analytic_metrics"]["overall"][key] = value
    gate = evaluate_density_ratio_pilot(
        [
            candidate,
            _pilot_candidate(1e-4, healthy=False),
            _pilot_candidate(3e-5, healthy=False),
            _pilot_candidate(1e-5, healthy=False),
        ]
    )
    assert gate["passed"] == 0


def test_null_pilot_requires_nonpositive_a_and_b_bounds() -> None:
    candidate = _pilot_candidate(3e-4)
    candidate["null"]["checkpoints"][1]["panels"]["a"]["overall"][
        "objective_improvement_lower_bound"
    ] = 1e-12
    candidate["null"]["checkpoints"][1]["panels"]["a"]["data_end"][
        "objective_improvement_lower_bound"
    ] = 1e-12
    gate = evaluate_density_ratio_pilot(
        [
            candidate,
            _pilot_candidate(1e-4, healthy=False),
            _pilot_candidate(3e-5, healthy=False),
            _pilot_candidate(1e-5, healthy=False),
        ]
    )
    assert gate["passed"] == 0


def test_teacher_threshold_boundaries_are_inclusive_and_two_of_three_pass() -> None:
    passing = [_teacher_metrics(seed) for seed in (10, 11)]
    failing = _teacher_metrics(12)
    failing["audit_panels"]["c"]["overall_relative_flux_l2"] = 0.150001
    failing["audit_panels"]["d"]["overall_relative_flux_l2"] = 0.150001
    assert evaluate_teacher_seed(passing[0])["passed"] == 1
    study = evaluate_teacher_study([*passing, failing])
    assert study["passed"] == 1
    assert study["passing_seed_count"] == 2


def test_null_zero_bound_passes_and_positive_bound_is_false_discovery() -> None:
    assert evaluate_null_seed(_null_metrics())["passed"] == 1
    bad = _null_metrics()
    bad["audit_panels"]["d"]["classification_improvement"]["overall"][
        "objective_improvement_lower_bound"
    ] = 1e-12
    gate = evaluate_null_seed(bad)
    assert gate["passed"] == 0
    assert gate["false_discovery"] == 1
    assert evaluate_null_study([_null_metrics(seed) for seed in (1, 2, 3)])["passed"] == 1


def _decision(controls: dict[str, object], *, pilot: dict[str, object] | None = None) -> str:
    return decide_density_ratio_controls(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight={"evaluation_status": "evaluated", "passed": 1},
        pilot=pilot or {"evaluation_status": "evaluated", "passed": 1},
        controls=controls,
    )["decision"]


def test_closed_decision_precedence() -> None:
    base = {
        "passed": 0,
        "optimizer_health_pass": 1,
        "teacher_study": {
            "classification_passing_seed_count": 0,
            "panel_disagreement": 0,
        },
        "null_study": {"false_discovery_count": 0},
    }
    assert _decision(base, pilot={"evaluation_status": "evaluated", "passed": 0}) == (
        DensityRatioDecision.CLASSIFICATION_OPTIMIZER_UNRESOLVED.value
    )
    unhealthy = copy.deepcopy(base)
    unhealthy["optimizer_health_pass"] = 0
    assert _decision(unhealthy) == DensityRatioDecision.CLASSIFICATION_OPTIMIZER_INVALID.value
    false = copy.deepcopy(base)
    false["null_study"]["false_discovery_count"] = 1
    assert _decision(false) == DensityRatioDecision.SELECTION_FALSE_DISCOVERY.value
    disagree = copy.deepcopy(base)
    disagree["teacher_study"]["panel_disagreement"] = 1
    assert _decision(disagree) == DensityRatioDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE.value
    value_only = copy.deepcopy(base)
    value_only["teacher_study"]["classification_passing_seed_count"] = 2
    assert _decision(value_only) == DensityRatioDecision.DENSITY_RATIO_VALUE_ONLY.value
    assert _decision(base) == DensityRatioDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL.value
    repaired = copy.deepcopy(base)
    repaired["passed"] = 1
    decision = decide_density_ratio_controls(
        provenance=1, preflight=1, pilot=1, controls=repaired
    )
    assert decision["decision"] == DensityRatioDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value
    assert decision["physical_training_authorized"] == 1
    assert decision["sampling_authorized"] == decision["sampling_performed"] == 0


def test_gate_artifacts_are_strict_json() -> None:
    artifacts = [
        evaluate_ratio_preflight(_preflight_metrics()),
        evaluate_density_ratio_pilot(
            [_pilot_candidate(lr) for lr in DensityRatioThresholds().pilot_learning_rates]
        ),
        evaluate_teacher_seed(_teacher_metrics()),
        evaluate_null_seed(_null_metrics()),
    ]
    for artifact in artifacts:
        json.dumps(artifact, allow_nan=False)
