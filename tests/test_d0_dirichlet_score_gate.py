from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import mnist.d0_dirichlet_score_gate as score_gate
from mnist.d0_dirichlet_score_gate import (
    DirichletScoreGateThresholds,
    ImplicitScoreDecision,
    bootstrap_cross_seed_cosine,
    bootstrap_whole_path_delta,
    decide_score_learnability,
    evaluate_control_bundle,
    evaluate_dirichlet_score_gates,
    evaluate_null_control,
    evaluate_positive_teacher_control,
    evaluate_score_seed,
    evaluate_score_study,
    median_of_means_whole_path_delta,
)


def _thresholds(**overrides: object) -> DirichletScoreGateThresholds:
    values: dict[str, object] = {
        "expected_model_seeds": 2,
        "min_passing_model_seeds": 2,
        "expected_audit_paths": 4,
        "expected_data_end_states_per_path": 2,
        "expected_data_end_states": 8,
        "bootstrap_reps": 300,
        "bootstrap_seed": 1234,
        "median_of_means_groups": 2,
    }
    values.update(overrides)
    return DirichletScoreGateThresholds(**values)


def _seed_result(
    seed: int,
    *,
    linear_delta: float = 0.4,
    zero_delta: float = 0.6,
    stein_delta: float = 0.3,
    bank_linear: dict[str, float] | None = None,
    selection_delta: float | None = None,
    train_delta: float = 0.5,
) -> dict[str, object]:
    paths = [10, 20, 30, 40]
    selection = linear_delta if selection_delta is None else selection_delta
    audit_rows: list[dict[str, object]] = []
    for bank in score_gate.AUDIT_BANKS:
        for scope in score_gate.AUDIT_SCOPES:
            for path in paths:
                local = (bank_linear or {}).get(bank, linear_delta)
                audit_rows.append(
                    {
                        "audit_bank": bank,
                        "scope": scope,
                        "path_id": path,
                        "state_count": 2 if scope == "data_end" else 4,
                        "score_risk_delta_vs_linear": local,
                        "score_risk_delta_vs_zero": zero_delta,
                        "finite_fraction": 1.0,
                    }
                )
    stein_rows = [
        {
            "stein_bank": bank,
            "path_id": path,
            "state_count": 4,
            "stein_discrepancy_improvement": stein_delta,
            "finite_fraction": 1.0,
            "time_bin_count": 5,
            "time_bin_state_counts": [1, 1, 1, 1, 2],
            "linear_witness_count": 32,
            "quadratic_witness_count": 32,
            "witness_count": 64,
            "aggregation": "square-within-path-and-time-bin-then-equal-average",
        }
        for bank in score_gate.STEIN_BANKS
        for path in paths
    ]
    return {
        "model_seed": seed,
        "complete": 1,
        "selected_step": 500,
        "selection_metrics": {
            scope: {
                "score_risk_delta_vs_linear": selection,
                "score_risk_delta_vs_zero": zero_delta,
                "finite_fraction": 1.0,
            }
            for scope in score_gate.AUDIT_SCOPES
        },
        "train_metrics": {
            "overall": {
                "score_risk_delta_vs_linear": train_delta,
                "score_risk_delta_vs_zero": zero_delta,
            }
        },
        "audit_path_ids": paths,
        "audit_path_metrics": audit_rows,
        "stein_path_metrics": stein_rows,
    }


def _cosines(value: float = 0.8) -> list[dict[str, object]]:
    return [
        {"path_id": path, "seed_a": 101, "seed_b": 102, "cosine": value}
        for path in (10, 20, 30, 40)
    ]


def _path_seed_rows() -> list[dict[str, object]]:
    return [
        {"path_id": path, "model_seed": seed, "value": path / 100.0 + seed / 1000.0}
        for path in (10, 20, 30, 40)
        for seed in (1, 2)
    ]


def test_whole_path_bootstrap_keeps_model_seeds_fixed_and_is_order_invariant() -> None:
    rows = _path_seed_rows()
    first = bootstrap_whole_path_delta(
        rows,
        reps=300,
        confidence=0.9,
        seed=77,
        expected_model_seeds=(1, 2),
        expected_path_ids=(10, 20, 30, 40),
    )
    second = bootstrap_whole_path_delta(
        list(reversed(rows)),
        reps=300,
        confidence=0.9,
        seed=77,
        expected_model_seeds=(1, 2),
        expected_path_ids=(10, 20, 30, 40),
    )
    assert first == second
    assert first["cluster_unit"] == "whole_path_id"
    assert first["fixed_factors"] == ["model_seed"]
    assert first["model_seed_count"] == 2
    assert first["point_delta"] == pytest.approx(0.2515)

    missing = rows[:-1]
    with pytest.raises(ValueError, match="every fixed model seed"):
        bootstrap_whole_path_delta(missing)
    duplicate = rows + [dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate"):
        bootstrap_whole_path_delta(duplicate)


def test_median_of_means_is_path_clustered_balanced_and_order_invariant() -> None:
    rows = _path_seed_rows()
    first = median_of_means_whole_path_delta(rows, groups=2, seed=99)
    second = median_of_means_whole_path_delta(list(reversed(rows)), groups=2, seed=99)
    assert first == second
    assert sorted(len(group) for group in first["group_path_ids"]) == [2, 2]
    assert first["median_of_means"] == pytest.approx(0.2515)
    with pytest.raises(ValueError, match="groups"):
        median_of_means_whole_path_delta(rows, groups=5)


def test_cross_seed_cosine_requires_every_pair_on_every_path() -> None:
    rows = [
        {
            "path_id": path,
            "seed_a": first,
            "seed_b": second,
            "cosine": 0.75,
        }
        for path in (1, 2)
        for first, second in ((3, 4), (3, 5), (4, 5))
    ]
    summary = bootstrap_cross_seed_cosine(
        rows,
        reps=100,
        seed=5,
        expected_model_seeds=(3, 4, 5),
        expected_path_ids=(1, 2),
    )
    assert summary["point_median"] == pytest.approx(0.75)
    assert summary["lower_bound"] == pytest.approx(0.75)
    assert summary["pair_count"] == 3
    with pytest.raises(ValueError, match="every fixed cross-seed pair"):
        bootstrap_cross_seed_cosine(rows[:-1])
    bad = copy.deepcopy(rows)
    bad[0]["cosine"] = 1.01
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        bootstrap_cross_seed_cosine(bad)


def test_seed_gate_requires_two_banks_two_scopes_and_exact_audit_coverage() -> None:
    thresholds = _thresholds()
    result = _seed_result(101)
    passing = evaluate_score_seed(result, thresholds)
    assert passing["passed"] == 1
    assert passing["subchecks"]["data_end_states_per_bank"]["value"] == {
        "audit_a": 8,
        "audit_b": 8,
    }
    assert len(passing["canonical_audit_path_metrics"]) == 16
    assert len(passing["canonical_stein_path_metrics"]) == 8

    missing_bank_scope = copy.deepcopy(result)
    missing_bank_scope["audit_path_metrics"] = [
        row
        for row in missing_bank_scope["audit_path_metrics"]
        if not (row["audit_bank"] == "audit_b" and row["scope"] == "data_end")
    ]
    failed = evaluate_score_seed(missing_bank_scope, thresholds)
    assert failed["passed"] == 0
    assert failed["subchecks"]["audit_bank_scope_coverage"]["passed"] == 0

    redistributed = copy.deepcopy(result)
    rows = [
        row
        for row in redistributed["audit_path_metrics"]
        if row["audit_bank"] == "audit_a" and row["scope"] == "data_end"
    ]
    rows[0]["state_count"] = 1
    rows[1]["state_count"] = 3
    failed = evaluate_score_seed(redistributed, thresholds)
    assert failed["passed"] == 0
    assert failed["subchecks"]["data_end_states_per_path"]["passed"] == 0


def test_seed_gate_fails_closed_for_duplicates_nonfinite_and_zero_boundaries() -> None:
    thresholds = _thresholds()
    duplicate = _seed_result(101)
    duplicate["audit_path_metrics"].append(copy.deepcopy(duplicate["audit_path_metrics"][0]))
    failed = evaluate_score_seed(duplicate, thresholds)
    assert failed["passed"] == 0
    assert failed["subchecks"]["parse"]["passed"] == 0

    nonfinite = _seed_result(101)
    nonfinite["audit_path_metrics"][0]["score_risk_delta_vs_linear"] = float("nan")
    failed = evaluate_score_seed(nonfinite, thresholds)
    assert failed["passed"] == 0
    assert json.dumps(failed, allow_nan=False)

    zero = _seed_result(101, linear_delta=0.0)
    failed = evaluate_score_seed(zero, thresholds)
    assert failed["passed"] == 0
    assert failed["subchecks"]["audit_beats_linear_and_zero"]["passed"] == 0


def test_risk_sums_are_converted_to_paired_mean_deltas() -> None:
    result = _seed_result(101)
    for row in result["audit_path_metrics"]:
        linear = row.pop("score_risk_delta_vs_linear")
        zero = row.pop("score_risk_delta_vs_zero")
        count = row["state_count"]
        row["full_risk_sum"] = 2.0 * count
        row["linear_risk_sum"] = row["full_risk_sum"] + linear * count
        row["zero_risk_sum"] = row["full_risk_sum"] + zero * count
    gate = evaluate_score_seed(result, _thresholds())
    assert gate["passed"] == 1
    assert gate["audit_mean_deltas"]["audit_a"]["overall"]["linear"] == pytest.approx(0.4)


def test_complete_score_study_passes_all_robustness_checks() -> None:
    thresholds = _thresholds()
    results = [_seed_result(101), _seed_result(102)]
    gate = evaluate_score_study(results, _cosines(), thresholds)
    assert gate["passed"] == 1
    assert gate["subchecks"]["passing_model_seed_count"]["value"] == 2
    assert gate["subchecks"]["audit_bootstrap_lower_bounds"]["passed"] == 1
    assert gate["subchecks"]["four_group_median_of_means"]["passed"] == 1
    assert gate["subchecks"]["stein_bootstrap_lower_bounds"]["passed"] == 1
    assert gate["subchecks"]["nonlinear_flux_cosine_median"]["passed"] == 1
    assert gate["subchecks"]["nonlinear_flux_cosine_bootstrap_lcb"]["passed"] == 1
    interval = gate["audit_statistics"]["audit_a"]["overall"]["linear"]["bootstrap"]
    assert interval["fixed_factors"] == ["model_seed"]
    assert interval["path_count"] == 4
    assert interval["seed"] == thresholds.bootstrap_seed
    assert (
        gate["audit_statistics"]["audit_b"]["data_end"]["zero"]["bootstrap"][
            "seed"
        ]
        == interval["seed"]
    )
    assert gate["stein_statistics"]["stein_a"]["bootstrap"]["seed"] == interval["seed"]
    assert gate["nonlinear_flux_cosine"]["seed"] == interval["seed"]
    first_groups = gate["audit_statistics"]["audit_a"]["overall"]["linear"][
        "median_of_means"
    ]["group_path_ids"]
    assert (
        gate["audit_statistics"]["audit_b"]["data_end"]["linear"][
            "median_of_means"
        ]["group_path_ids"]
        == first_groups
    )
    json.dumps(gate, allow_nan=False)


def test_study_fails_when_trace_banks_disagree() -> None:
    thresholds = _thresholds()
    results = [
        _seed_result(
            seed,
            bank_linear={"audit_a": 0.4, "audit_b": -0.2},
        )
        for seed in (101, 102)
    ]
    gate = evaluate_score_study(results, _cosines(), thresholds)
    assert gate["passed"] == 0
    assert gate["diagnostics"]["trace_bank_sign_disagreement"] == 1
    decision = decide_score_learnability(
        preflight_gate=True,
        cache_gate=True,
        controls_gate=True,
        score_gate=gate,
    )
    assert decision["decision"] == ImplicitScoreDecision.TRACE_ESTIMATOR_INCONCLUSIVE.value


def test_study_requires_all_fixed_model_seeds_and_cross_seed_pairs() -> None:
    thresholds = _thresholds()
    missing_seed = evaluate_score_study([_seed_result(101)], [], thresholds)
    assert missing_seed["passed"] == 0
    assert missing_seed["subchecks"]["model_seed_count"]["passed"] == 0

    missing_pair = evaluate_score_study(
        [_seed_result(101), _seed_result(102)], _cosines()[:-1], thresholds
    )
    assert missing_pair["passed"] == 0
    assert missing_pair["subchecks"]["nonlinear_flux_cosine_median"]["passed"] == 0


def test_positive_teacher_and_null_control_boundaries() -> None:
    thresholds = _thresholds()
    teacher = {
        "complete": 1,
        "selected_step": 500,
        "audit_overall_score_gain": 0.90,
        "audit_data_end_score_gain": 0.90,
        "overall_flux_cosine": 0.98,
        "time_bin_flux_cosines": [0.95] * 5,
        "overall_relative_flux_l2": 0.15,
        "time_bin_relative_flux_l2": [0.20] * 5,
        "nonlinear_gain_vs_linear": 1e-12,
    }
    assert evaluate_positive_teacher_control(teacher, thresholds)["passed"] == 1
    bad_teacher = dict(teacher, nonlinear_gain_vs_linear=0.0)
    assert evaluate_positive_teacher_control(bad_teacher, thresholds)["passed"] == 0
    bad_bins = dict(teacher, time_bin_flux_cosines=[])
    assert evaluate_positive_teacher_control(bad_bins, thresholds)["passed"] == 0
    too_few_bins = dict(teacher, time_bin_flux_cosines=[0.95] * 4)
    assert evaluate_positive_teacher_control(too_few_bins, thresholds)["passed"] == 0

    assert evaluate_null_control(
        {"complete": 1, "audit_improvement_lower_bound": 0.0, "comparator": "frozen_training_only_linear_spline_step0"}
    )["passed"] == 1
    assert evaluate_null_control(
        {"complete": 1, "audit_improvement_lower_bound": 1e-12, "comparator": "frozen_training_only_linear_spline_step0"}
    )["passed"] == 0
    assert evaluate_null_control(
        {"complete": 1, "audit_improvement_lower_bound": 0.0}
    )["passed"] == 0

    controls = evaluate_control_bundle(
        operator_gate={"passed": 1},
        positive_teacher_gate={"passed": 1},
        null_control_gate={"passed": 1},
    )
    assert controls["passed"] == 1
    assert evaluate_control_bundle(
        operator_gate=True,
        positive_teacher_gate=True,
        null_control_gate=False,
    )["passed"] == 0


@pytest.mark.parametrize(
    ("preflight", "cache", "controls", "passed", "diagnostics", "expected"),
    [
        (False, True, True, 0, {}, ImplicitScoreDecision.OPERATOR_INVALID),
        (True, False, True, 0, {}, ImplicitScoreDecision.CACHE_INVALID),
        (True, True, False, 0, {}, ImplicitScoreDecision.OPTIMIZATION_PIPELINE_INVALID),
        (
            True,
            True,
            True,
            0,
            {"task_set_complete": 0},
            ImplicitScoreDecision.OPTIMIZATION_PIPELINE_INVALID,
        ),
        (
            True,
            True,
            True,
            1,
            {"task_set_complete": 1},
            ImplicitScoreDecision.IMPLICIT_SCORE_SIGNAL,
        ),
        (
            True,
            True,
            True,
            0,
            {"task_set_complete": 1, "trace_bank_sign_disagreement": 1},
            ImplicitScoreDecision.TRACE_ESTIMATOR_INCONCLUSIVE,
        ),
        (
            True,
            True,
            True,
            0,
            {
                "task_set_complete": 1,
                "objective_robust_signal": 1,
                "stein_robust_signal": 0,
            },
            ImplicitScoreDecision.OBJECTIVE_ONLY_SIGNAL,
        ),
        (
            True,
            True,
            True,
            0,
            {
                "task_set_complete": 1,
                "linear_point_signal": 1,
                "stein_point_signal": 1,
                "cosine_point_signal": 1,
                "bootstrap_and_mom_pass": 0,
            },
            ImplicitScoreDecision.BOUNDARY_OR_OUTLIER_ARTIFACT,
        ),
        (
            True,
            True,
            True,
            0,
            {
                "task_set_complete": 1,
                "zero_point_signal": 1,
                "linear_point_signal": 0,
            },
            ImplicitScoreDecision.LINEAR_SPATIOTEMPORAL_ONLY,
        ),
        (
            True,
            True,
            True,
            0,
            {
                "task_set_complete": 1,
                "selection_point_seed_count": 2,
                "train_point_seed_count": 2,
                "linear_point_signal": 0,
            },
            ImplicitScoreDecision.PATH_MEMORIZATION_ONLY,
        ),
        (
            True,
            True,
            True,
            0,
            {"task_set_complete": 1},
            ImplicitScoreDecision.NO_DETECTABLE_IMPLICIT_SCORE,
        ),
    ],
)
def test_closed_decision_state_machine(
    preflight: bool,
    cache: bool,
    controls: bool,
    passed: int,
    diagnostics: dict[str, int],
    expected: ImplicitScoreDecision,
) -> None:
    report = decide_score_learnability(
        preflight_gate=preflight,
        cache_gate=cache,
        controls_gate=controls,
        score_gate={"passed": passed, "diagnostics": diagnostics},
    )
    assert report["decision"] == expected.value
    assert report["sampling_performed"] == 0
    assert report["sampling_authorized"] == 0
    assert report["recommended_next_action"]


def test_final_aggregation_is_cumulative_and_require_gate_is_fail_closed() -> None:
    thresholds = _thresholds()
    results = [_seed_result(101), _seed_result(102)]
    report = evaluate_dirichlet_score_gates(
        preflight_gate={"passed": 1},
        cache_gate={"passed": 1},
        controls_gate={"passed": 1},
        seed_results=results,
        nonlinear_flux_cosines=_cosines(),
        require_gate="score",
        thresholds=thresholds,
    )
    assert report["required_gate_pass"] == 1
    assert report["cumulative_pass"] == {
        "preflight": 1,
        "cache": 1,
        "controls": 1,
        "score": 1,
    }
    assert report["decision"]["decision"] == ImplicitScoreDecision.IMPLICIT_SCORE_SIGNAL.value
    assert report["sampling_performed"] == 0
    json.dumps(report, allow_nan=False)

    failed = evaluate_dirichlet_score_gates(
        preflight_gate={"passed": 1},
        cache_gate={"passed": 0},
        controls_gate={"passed": 1},
        seed_results=results,
        nonlinear_flux_cosines=_cosines(),
        require_gate="score",
        thresholds=thresholds,
    )
    assert failed["required_gate_pass"] == 0
    assert failed["cumulative_pass"]["controls"] == 0
    assert failed["decision"]["decision"] == ImplicitScoreDecision.CACHE_INVALID.value

    with pytest.raises(ValueError, match="require_gate"):
        evaluate_dirichlet_score_gates(
            preflight_gate=True,
            cache_gate=True,
            controls_gate=True,
            seed_results=results,
            nonlinear_flux_cosines=_cosines(),
            require_gate="sampling",
            thresholds=thresholds,
        )


def test_threshold_validation_and_default_production_contract() -> None:
    defaults = DirichletScoreGateThresholds()
    assert defaults.expected_model_seeds == 3
    assert defaults.min_passing_model_seeds == 2
    assert defaults.expected_audit_paths == 24
    assert defaults.expected_data_end_states == 384
    assert defaults.bootstrap_reps == 10_000
    assert defaults.bootstrap_confidence == 0.90
    assert defaults.bootstrap_seed == 260760
    assert defaults.nonlinear_flux_cosine == 0.50
    assert defaults.nonlinear_flux_cosine_lcb == 0.25

    with pytest.raises(ValueError, match="min_passing"):
        DirichletScoreGateThresholds(
            expected_model_seeds=2, min_passing_model_seeds=3
        )
    with pytest.raises(ValueError, match="paths times"):
        DirichletScoreGateThresholds(expected_data_end_states=383)
    with pytest.raises(ValueError, match="bootstrap_confidence"):
        DirichletScoreGateThresholds(bootstrap_confidence=1.0)


def test_gate_module_has_no_sampler_dependency() -> None:
    source = Path(score_gate.__file__).read_text(encoding="utf-8")
    assert "d0_one_image_sampler" not in source
    assert "run_paired" not in source
