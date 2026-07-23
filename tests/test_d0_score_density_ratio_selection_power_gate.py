from __future__ import annotations

import copy
import json

import pytest

import mnist.d0_score_density_ratio_selection_power_gate as gate_module
from mnist.d0_score_density_ratio_selection_power_gate import (
    SelectionPowerDecision,
    SelectionPowerThresholds,
    analyze_null_multiplicity,
    decide_selection_power,
    evaluate_oracle_calibration,
    evaluate_oracle_panel_set,
    evaluate_power_pilot,
    evaluate_power_teacher_seed,
    evaluate_saved_oracle_forensic,
    evaluate_selection_power_preflight,
    evaluate_selection_power_workflow,
)


def _panel(
    *,
    paths: int = 128,
    confidence: float = 0.90,
    lower: float = 0.01,
) -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "path_count": paths,
        "anchors_per_path": 32,
        "bootstrap_replicates": 10_000,
        "confidence": confidence,
        "overall": {"objective_improvement_lower_bound": lower},
        "data_end": {"objective_improvement_lower_bound": lower},
    }


def _panel_set(roles: tuple[str, ...] = ("a", "b")) -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "panels": {role: _panel() for role in roles},
        "pairwise_disjoint": 1,
        "calibration_overlap_path_count": 0,
        "frozen_before_training": 1,
        "optimizer_steps_before_oracle_gate": 0,
        "regenerated_after_inspection": 0,
    }


def _forensic() -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "path_count": 16,
        "anchors_per_path": 32,
        "saved_panel_hashes_verified": 1,
        "panel_a": {"lower_bounds": [0.00872031148, 0.00517841895]},
        "panel_b": {"lower_bounds": [-0.00913743689, -0.00565862046]},
    }


def _calibration() -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "full": _panel(paths=256, confidence=0.99),
        "halves": [_panel(), _panel()],
        "predetermined_split": 1,
        "halves_disjoint": 1,
        "evaluation_overlap_path_count": 0,
        "panel_frozen_before_inspection": 1,
        "regenerated_after_inspection": 0,
    }


def _candidate_gate(
    lr: float,
    *,
    passed: int = 1,
    risk: float = 0.60,
    clip: float = 0.0,
) -> dict[str, object]:
    return {
        "gate": "selection_power_pilot_candidate",
        "evaluation_status": "evaluated",
        "passed": passed,
        "subchecks": {},
        "learning_rate": lr,
        "accumulation_steps": 8,
        "teacher_mean_ab_bce": risk,
        "maximum_clip_fraction_observed": clip,
        "optimizer_health_pass": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _a_only_candidate() -> dict[str, object]:
    base = {
        "gate": "density_ratio_pilot_candidate",
        "evaluation_status": "evaluated",
        "passed": 0,
        "subchecks": {
            "null_panel_a_lower_bounds": {"passed": 0},
            "null_panel_b_lower_bounds": {"passed": 1},
            "null_b_rejected": {"passed": 1},
            "null_selected_zero": {"passed": 1},
            "teacher_b_confirmed": {"passed": 1},
        },
        "null_selection": {
            "selected_step": 0,
            "confirmation": {
                "accepted": 0,
                "panel_b_lower_bounds": [-0.001, -0.002],
            },
        },
    }
    return {
        "gate": "selection_power_pilot_candidate",
        "evaluation_status": "evaluated",
        "passed": 0,
        "subchecks": {},
        "learning_rate": 3e-5,
        "accumulation_steps": 8,
        "optimizer_health_pass": 1,
        "frozen_normalized_head_gate": {"nested": {"base": base}},
    }


def _passed() -> dict[str, object]:
    return {"evaluation_status": "evaluated", "passed": 1}


def _pending() -> dict[str, object]:
    return {"evaluation_status": "not_evaluated", "passed": 0}


def test_thresholds_freeze_power_repair_without_changing_science() -> None:
    thresholds = SelectionPowerThresholds()
    assert thresholds.calibration_paths == 256
    assert thresholds.evidence_panel_paths == 128
    assert thresholds.bootstrap_replicates == 10_000
    assert thresholds.head.base_channels == 32
    assert thresholds.head.stability.density_ratio.teacher.teacher_min_score_gain == 0.90
    with pytest.raises(ValueError, match="evidence_panel_paths"):
        SelectionPowerThresholds(evidence_panel_paths=64)


def test_saved_forensic_reproduces_inspected_a_b_pattern() -> None:
    gate = evaluate_saved_oracle_forensic(_forensic())
    assert gate["passed"] == 1
    assert gate["panel_a_lower_bounds"][0] > 0
    assert gate["panel_b_lower_bounds"][0] < 0

    broken = _forensic()
    broken["panel_b"]["lower_bounds"][0] += 1e-3
    assert evaluate_saved_oracle_forensic(broken)["passed"] == 0


def test_full_256_and_both_predetermined_halves_are_required() -> None:
    assert evaluate_oracle_calibration(_calibration())["passed"] == 1
    broken = _calibration()
    broken["halves"][1]["data_end"]["objective_improvement_lower_bound"] = 0.0
    gate = evaluate_oracle_calibration(broken)
    assert gate["passed"] == 0
    assert gate["half_panel_gates"][1]["passed"] == 0

    leaked = _calibration()
    leaked["evaluation_overlap_path_count"] = 1
    assert evaluate_oracle_calibration(leaked)["passed"] == 0


def test_actual_panel_set_fails_before_any_optimizer_step_or_regeneration() -> None:
    gate = evaluate_oracle_panel_set(_panel_set(), expected_roles=("a", "b"))
    assert gate["passed"] == 1

    stepped = _panel_set()
    stepped["optimizer_steps_before_oracle_gate"] = 1
    assert evaluate_oracle_panel_set(stepped, expected_roles=("a", "b"))["passed"] == 0

    regenerated = _panel_set()
    regenerated["regenerated_after_inspection"] = 1
    assert evaluate_oracle_panel_set(regenerated, expected_roles=("a", "b"))["passed"] == 0

    wrong_roles = _panel_set(("a", "b", "c"))
    assert evaluate_oracle_panel_set(wrong_roles, expected_roles=("a", "b"))["passed"] == 0


def test_preflight_combines_inherited_coordinate_and_oracle_power() -> None:
    gate = evaluate_selection_power_preflight(
        normalized_head_preflight=_passed(),
        saved_forensic=_forensic(),
        calibration=_calibration(),
    )
    assert gate["passed"] == 1
    broken = evaluate_selection_power_preflight(
        normalized_head_preflight={"evaluation_status": "evaluated", "passed": 0},
        saved_forensic=_forensic(),
        calibration=_calibration(),
    )
    assert broken["passed"] == 0


def test_power_pilot_keeps_normalized_head_ranking_and_requires_panel_power() -> None:
    candidates = [
        _candidate_gate(3e-5, risk=0.60),
        _candidate_gate(1e-5, risk=0.59),
    ]
    gate = evaluate_power_pilot(candidates, panel_power=_passed())
    assert gate["passed"] == 1
    assert gate["selected_profile"]["profile"]["body_learning_rate"] == pytest.approx(1e-5)
    assert evaluate_power_pilot(candidates, panel_power={"evaluation_status": "evaluated", "passed": 0})["passed"] == 0


def test_a_only_null_excursion_is_advisory_and_never_authorizes() -> None:
    analysis = analyze_null_multiplicity([_a_only_candidate()])
    assert analysis["authorizing"] == 0
    assert analysis["a_only_explains_failure"] == 1

    other = copy.deepcopy(_a_only_candidate())
    base = gate_module._find_gate(other, "density_ratio_pilot_candidate")
    # _find_gate returns a copy, so edit the nested source explicitly.
    del base
    other["frozen_normalized_head_gate"]["nested"]["base"]["subchecks"][
        "teacher_b_confirmed"
    ]["passed"] = 0
    analysis = analyze_null_multiplicity([other])
    assert analysis["a_only_explains_failure"] == 0


def test_confirmation_adapts_only_old_path_metadata(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_seed(value, _thresholds):
        observed.update(value)
        assert value["audit_panels"]["c"]["path_count"] == 32
        assert value["audit_panels"]["d"]["path_count"] == 32
        return {
            "gate": "normalized_head_teacher_seed",
            "evaluation_status": "evaluated",
            "passed": 1,
            "model_seed": 260941,
            "optimizer_health_pass": 1,
            "classification_pass": 1,
            "derivative_pass": 1,
            "panel_disagreement": 0,
        }

    monkeypatch.setattr(gate_module, "evaluate_frozen_teacher_seed", fake_seed)
    raw = {
        "evaluation_status": "evaluated",
        "model_seed": 260941,
        "audit_panels": {
            "c": {"path_count": 128, "anchors_per_path": 32},
            "d": {"path_count": 128, "anchors_per_path": 32},
        },
    }
    gate = evaluate_power_teacher_seed(raw)
    assert gate["passed"] == 1
    assert raw["audit_panels"]["c"]["path_count"] == 128
    wrong = copy.deepcopy(raw)
    wrong["audit_panels"]["d"]["path_count"] = 127
    assert evaluate_power_teacher_seed(wrong)["passed"] == 0


def test_closed_decisions_distinguish_power_multiplicity_optimizer_and_h1() -> None:
    controls = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "optimizer_health_pass": 1,
        "teacher_study": {"classification_passing_seed_count": 0, "panel_disagreement": 0},
        "null_study": {"false_discovery_count": 0},
    }
    pilot = {"evaluation_status": "evaluated", "passed": 0, "optimizer_health_pass": 1}
    kwargs = {
        "provenance": _passed(),
        "preflight": _passed(),
        "pilot_panel_power": _passed(),
        "pilot": pilot,
        "confirmation_panel_power": _pending(),
        "controls": controls,
    }
    decision = decide_selection_power(**kwargs)
    assert decision["decision"] == SelectionPowerDecision.CLASSIFICATION_POWER_CONFIRMATION_UNRESOLVED.value
    assert decision["h1_function_step_patch_authorized"] == 1
    assert decision["physical_training_authorized"] == 0

    underpowered = dict(kwargs)
    underpowered["pilot_panel_power"] = {"evaluation_status": "evaluated", "passed": 0}
    assert decide_selection_power(**underpowered)["decision"] == SelectionPowerDecision.EVIDENCE_PANEL_UNDERPOWERED.value

    oracle = dict(kwargs)
    oracle["preflight"] = {"evaluation_status": "evaluated", "passed": 0}
    assert decide_selection_power(**oracle)["decision"] == SelectionPowerDecision.ORACLE_POWER_INVALID.value

    multiplicity = copy.deepcopy(kwargs)
    multiplicity["pilot"]["null_multiplicity_analysis"] = {"a_only_explains_failure": 1}
    assert decide_selection_power(**multiplicity)["decision"] == SelectionPowerDecision.NULL_GATE_MULTIPLICITY_INCONCLUSIVE.value

    optimizer = copy.deepcopy(kwargs)
    optimizer["pilot"]["optimizer_health_pass"] = 0
    assert decide_selection_power(**optimizer)["decision"] == SelectionPowerDecision.CLASSIFICATION_OPTIMIZER_INVALID.value


def test_confirmation_decisions_retain_strict_authorization() -> None:
    controls = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "optimizer_health_pass": 1,
        "teacher_study": {"classification_passing_seed_count": 0, "panel_disagreement": 0},
        "null_study": {"false_discovery_count": 0},
    }
    kwargs = {
        "provenance": _passed(),
        "preflight": _passed(),
        "pilot_panel_power": _passed(),
        "pilot": _passed(),
        "confirmation_panel_power": _passed(),
        "controls": controls,
    }
    assert decide_selection_power(**kwargs)["decision"] == SelectionPowerDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL.value

    false = copy.deepcopy(kwargs)
    false["controls"]["null_study"]["false_discovery_count"] = 1
    assert decide_selection_power(**false)["decision"] == SelectionPowerDecision.SELECTION_FALSE_DISCOVERY.value

    value_only = copy.deepcopy(kwargs)
    value_only["controls"]["teacher_study"]["classification_passing_seed_count"] = 2
    assert decide_selection_power(**value_only)["decision"] == SelectionPowerDecision.DENSITY_RATIO_VALUE_ONLY.value

    repaired = copy.deepcopy(kwargs)
    repaired["controls"]["passed"] = 1
    decision = decide_selection_power(**repaired)
    assert decision["decision"] == SelectionPowerDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value
    assert decision["physical_training_authorized"] == 1
    assert decision["sampling_authorized"] == 0


def test_required_gate_fails_closed_and_report_is_strict_json() -> None:
    pilot = evaluate_power_pilot(
        [_candidate_gate(3e-5), _candidate_gate(1e-5)],
        panel_power=_passed(),
    )
    report = evaluate_selection_power_workflow(
        provenance=_passed(),
        preflight=_passed(),
        pilot_panel_power=_passed(),
        pilot=pilot,
        confirmation_panel_power=_pending(),
        teacher_results=[],
        null_results=[],
        require_gate="pilot",
    )
    assert report["required_gate_pass"] == 1
    assert report["physical_training_performed"] == 0
    assert report["sampling_performed"] == 0
    json.dumps(report, allow_nan=False)

    blocked = copy.deepcopy(report)
    del blocked
    with pytest.raises(ValueError, match="require_gate"):
        evaluate_selection_power_workflow(
            provenance=_passed(),
            preflight=_passed(),
            pilot_panel_power=_passed(),
            pilot=pilot,
            confirmation_panel_power=_pending(),
            teacher_results=[],
            null_results=[],
            require_gate="science",
        )
