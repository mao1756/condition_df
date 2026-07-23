from __future__ import annotations

import copy
import json

import pytest

import mnist.d0_score_density_ratio_head_gate as gate_module
from mnist.d0_score_density_ratio_head_gate import (
    HeadCoordinateDecision,
    HeadCoordinateThresholds,
    decide_head_coordinate,
    evaluate_head_pilot,
    evaluate_head_pilot_candidate,
    evaluate_head_workflow,
    evaluate_normalized_head_preflight,
    select_head_profile,
)


def _preflight() -> dict[str, object]:
    value: dict[str, object] = {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "parent_provenance_pass": 1,
        "grid_cells": 784,
        "base_channels": 32,
        "preflight_paths": 128,
        "preflight_confidence": 0.99,
        "loss_algebra_max_error": 1e-13,
        "expanded_loss_max_error": 1e-8,
        "expanded_gradient_max_error": 1e-7,
        "accumulation_gradient_max_error": 1e-7,
        "cuda_logit_max_abs_error": 2e-6,
        "cuda_bce_max_abs_error": 2e-6,
        "float64_logit_max_abs_error": 1e-9,
        "float64_bce_max_abs_error": 1e-9,
        "state_gradient_relative_error": 2e-6,
        "edge_score_relative_error": 2e-6,
        "flux_relative_error": 2e-6,
        "head_gradient_scale_relative_error": 2e-6,
        "backbone_gradient_relative_error": 2e-6,
        "adamw_coordinate_max_relative_error": 2e-6,
        "ema_coordinate_max_relative_error": 2e-6,
        "median_legacy_head_squared_gradient_share": 0.95,
        "parent_loss_scale_reused": 1,
        "adaptive_loss_scaling": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    for name in (
        "head_gradient_one_over_n_pass",
        "backbone_gradient_unchanged_pass",
        "adamw_group_learning_rate_pass",
        "adamw_group_epsilon_pass",
        "adamw_group_weight_decay_pass",
        "legacy_checkpoint_report_only_pass",
        "boundary_operator_pass",
        "finite_device_backward_pass",
        "stream_replay_pass",
        "paired_estimator_pass",
        "mixture_coefficients_pass",
        "dirichlet_marginals_pass",
        "common_gamma_coupling_pass",
        "exact_seed_namespaces_pass",
        "null_pool_swap_structure_pass",
        "time_strata_pass",
        "class_balance_pass",
        "candidate_order_invariance_pass",
        "nested_accumulation_prefix_pass",
        "parent_forensic_finite",
    ):
        value[name] = 1
    return value


def _candidate_gate(lr: float, *, passed: int = 1, risk: float = 0.6, clip: float = 0.05) -> dict[str, object]:
    return {
        "gate": "normalized_head_pilot_candidate",
        "evaluation_status": "evaluated",
        "passed": passed,
        "subchecks": {},
        "accumulation_steps": 8,
        "learning_rate": lr,
        "teacher_mean_ab_bce": risk,
        "teacher_panel_b_bce": risk + 0.01,
        "maximum_clip_fraction_observed": clip,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def test_thresholds_freeze_width_grid_accumulation_and_science() -> None:
    thresholds = HeadCoordinateThresholds()
    assert thresholds.grid_cells == 784
    assert thresholds.base_channels == 32
    assert thresholds.accumulation_steps == 8
    assert thresholds.pilot_learning_rates == (3e-5, 1e-5)
    assert thresholds.stability.density_ratio.teacher.teacher_min_score_gain == 0.90
    with pytest.raises(ValueError, match="base_channels"):
        HeadCoordinateThresholds(base_channels=16)


@pytest.mark.parametrize(
    "name",
    [
        "cuda_logit_max_abs_error",
        "float64_bce_max_abs_error",
        "state_gradient_relative_error",
        "head_gradient_scale_relative_error",
        "adamw_coordinate_max_relative_error",
        "ema_coordinate_max_relative_error",
    ],
)
def test_preflight_thresholds_are_inclusive_and_fail_above(name: str) -> None:
    metrics = _preflight()
    assert evaluate_normalized_head_preflight(metrics)["passed"] == 1
    metrics[name] = float(metrics[name]) * 1.000001
    result = evaluate_normalized_head_preflight(metrics)
    assert result["passed"] == 0
    assert result["subchecks"][name]["passed"] == 0


def test_preflight_head_share_is_gate_but_width_ablation_is_advisory() -> None:
    metrics = _preflight()
    metrics["width_ablation"] = {"16": 1, "24": 1, "32": 0}
    result = evaluate_normalized_head_preflight(metrics)
    assert result["passed"] == 1
    assert result["width_ablation_gate_eligible"] == 0
    metrics["median_legacy_head_squared_gradient_share"] = 0.949999
    assert evaluate_normalized_head_preflight(metrics)["passed"] == 0


def test_candidate_requires_fixed_accumulation_and_lr(monkeypatch) -> None:
    monkeypatch.setattr(
        gate_module,
        "evaluate_frozen_pilot_candidate",
        lambda candidate, thresholds: {
            "gate": "paired_ratio_stability_pilot_candidate",
            "evaluation_status": "evaluated",
            "passed": 1,
            "teacher_mean_ab_bce": 0.6,
            "teacher_panel_b_bce": 0.61,
            "maximum_clip_fraction_observed": 0.05,
            "optimizer_health_pass": 1,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    candidate = {"accumulation_steps": 8, "learning_rate": 3e-5}
    result = evaluate_head_pilot_candidate(candidate)
    assert result["passed"] == 1
    assert result["head_learning_rate"] == pytest.approx(784 * 3e-5)
    candidate["accumulation_steps"] = 4
    assert evaluate_head_pilot_candidate(candidate)["passed"] == 0


def test_profile_ranking_and_exact_candidate_set() -> None:
    gates = [
        _candidate_gate(3e-5, risk=0.60, clip=0.02),
        _candidate_gate(1e-5, risk=0.60, clip=0.01),
    ]
    selected = select_head_profile(gates)
    assert selected["selected"] == 1
    assert selected["profile"]["body_learning_rate"] == pytest.approx(1e-5)
    assert selected["profile"]["head_learning_rate"] == pytest.approx(784e-5)
    pilot = evaluate_head_pilot(gates)
    assert pilot["passed"] == 1
    assert pilot["selected_profile"]["profile"]["body_learning_rate"] == pytest.approx(1e-5)
    assert evaluate_head_pilot(gates[:1])["passed"] == 0


def test_decisions_are_closed_and_fail_closed() -> None:
    passed = {"evaluation_status": "evaluated", "passed": 1}
    pending = {"evaluation_status": "not_evaluated", "passed": 0}
    controls = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "optimizer_health_pass": 1,
        "teacher_study": {"classification_passing_seed_count": 0, "panel_disagreement": 0},
        "null_study": {"false_discovery_count": 0},
    }
    assert decide_head_coordinate(
        provenance={"evaluation_status": "evaluated", "passed": 0},
        preflight=passed,
        pilot=passed,
        controls=controls,
    )["decision"] == HeadCoordinateDecision.CONTROL_PROVENANCE_INVALID.value
    assert decide_head_coordinate(
        provenance=passed, preflight={"evaluation_status": "evaluated", "passed": 0}, pilot=passed, controls=controls
    )["decision"] == HeadCoordinateDecision.NORMALIZED_HEAD_COORDINATE_INVALID.value
    assert decide_head_coordinate(
        provenance=passed, preflight=passed, pilot={"evaluation_status": "evaluated", "passed": 0}, controls=controls
    )["decision"] == HeadCoordinateDecision.CLASSIFICATION_COORDINATE_REPAIR_UNRESOLVED.value
    assert decide_head_coordinate(
        provenance=passed, preflight=passed, pilot=passed, controls=pending
    )["decision"] == "normalized_head_pilot_passed"

    optimizer = copy.deepcopy(controls)
    optimizer["optimizer_health_pass"] = 0
    assert decide_head_coordinate(
        provenance=passed, preflight=passed, pilot=passed, controls=optimizer
    )["decision"] == HeadCoordinateDecision.CLASSIFICATION_OPTIMIZER_INVALID.value
    false = copy.deepcopy(controls)
    false["null_study"]["false_discovery_count"] = 1
    assert decide_head_coordinate(
        provenance=passed, preflight=passed, pilot=passed, controls=false
    )["decision"] == HeadCoordinateDecision.SELECTION_FALSE_DISCOVERY.value
    disagree = copy.deepcopy(controls)
    disagree["teacher_study"]["panel_disagreement"] = 1
    assert decide_head_coordinate(
        provenance=passed, preflight=passed, pilot=passed, controls=disagree
    )["decision"] == HeadCoordinateDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE.value
    value_only = copy.deepcopy(controls)
    value_only["teacher_study"]["classification_passing_seed_count"] = 2
    assert decide_head_coordinate(
        provenance=passed, preflight=passed, pilot=passed, controls=value_only
    )["decision"] == HeadCoordinateDecision.DENSITY_RATIO_VALUE_ONLY.value
    repaired = copy.deepcopy(controls)
    repaired["passed"] = 1
    decision = decide_head_coordinate(
        provenance=passed, preflight=passed, pilot=passed, controls=repaired
    )
    assert decision["decision"] == HeadCoordinateDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value
    assert decision["physical_training_authorized"] == 1
    assert decision["sampling_authorized"] == 0


def test_required_gate_is_strict_json() -> None:
    report = evaluate_head_workflow(
        provenance={"evaluation_status": "evaluated", "passed": 1},
        preflight=evaluate_normalized_head_preflight(_preflight()),
        pilot=evaluate_head_pilot(
            [_candidate_gate(3e-5), _candidate_gate(1e-5)]
        ),
        teacher_results=[],
        null_results=[],
        require_gate="pilot",
    )
    assert report["required_gate_pass"] == 1
    assert report["physical_training_performed"] == report["sampling_performed"] == 0
    json.dumps(report, allow_nan=False)
