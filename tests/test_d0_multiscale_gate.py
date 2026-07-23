from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import mnist.d0_multiscale_gate as multiscale
from mnist.d0_multiscale_gate import (
    LearnabilityDecision,
    MultiscaleGateThresholds,
    bootstrap_whole_path_gain,
    compute_multiscale_split_metrics,
    decide_learnability,
    evaluate_multiscale_gates,
    evaluate_stride_pass,
    evaluate_teacher_control,
    fit_tau_bin_baseline,
    predict_tau_bin_baseline,
    write_multiscale_gate_artifacts,
)


def _pilot_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    paths = np.arange(6, dtype=np.int64)
    path_ids = np.repeat(paths, 5)
    tau = np.tile(np.asarray([0.1, 0.3, 0.5, 0.7, 0.9]), paths.size)
    features = []
    for path in paths.tolist():
        # Each split has one negative- and one positive-offset path.  The
        # train-only time mean sees the time profile but not this state signal.
        offset = -0.5 if path % 2 == 0 else 0.5
        for value in (0.1, 0.3, 0.5, 0.7, 0.9):
            features.append([1.0 + value + offset, 0.5 - value - offset])
    targets = np.asarray(features, dtype=np.float64)
    predictions = 0.9 * targets
    splits = {"train": [0, 1], "selection": [2, 3], "audit": [4, 5]}
    return targets, predictions, tau, path_ids, splits


def _seed_result(seed: int, *, stride: int = 16) -> dict[str, object]:
    targets, predictions, tau, path_ids, splits = _pilot_arrays()
    return compute_multiscale_split_metrics(
        targets,
        predictions,
        tau,
        path_ids,
        splits,
        stride=stride,
        training_seed=seed,
        selected_step=250,
    )


def _production_coverage_seed_result(
    seed: int,
    *,
    stride: int = 16,
) -> dict[str, object]:
    paths = np.arange(36, dtype=np.int64)
    path_ids = np.repeat(paths, 16)
    tau = np.full(path_ids.size, 0.9, dtype=np.float64)
    targets: list[list[float]] = []
    for path in paths.tolist():
        offset = -0.4 if path % 2 == 0 else 0.4
        for anchor in range(16):
            wobble = 0.01 * (anchor - 7.5)
            targets.append([1.5 + offset + wobble, -0.75 - offset + wobble])
    target = np.asarray(targets, dtype=np.float64)
    prediction = 0.9 * target
    return compute_multiscale_split_metrics(
        target,
        prediction,
        tau,
        path_ids,
        {
            "train": paths[:12].tolist(),
            "selection": paths[12:24].tolist(),
            "audit": paths[24:].tolist(),
        },
        stride=int(stride),
        training_seed=int(seed),
        selected_step=250,
    )


def _confirmation_coverage_seed_result(
    seed: int,
    *,
    stride: int = 16,
) -> dict[str, object]:
    paths = np.arange(128, dtype=np.int64)
    path_ids = np.repeat(paths, 16)
    tau = np.full(path_ids.size, 0.9, dtype=np.float64)
    targets: list[list[float]] = []
    for path in paths.tolist():
        offset = -0.4 if path % 2 == 0 else 0.4
        for anchor in range(16):
            wobble = 0.01 * (anchor - 7.5)
            targets.append([1.5 + offset + wobble, -0.75 - offset + wobble])
    target = np.asarray(targets, dtype=np.float64)
    return compute_multiscale_split_metrics(
        target,
        0.9 * target,
        tau,
        path_ids,
        {
            "train": paths[:80].tolist(),
            "selection": paths[80:104].tolist(),
            "audit": paths[104:].tolist(),
        },
        stride=int(stride),
        training_seed=int(seed),
        selected_step=250,
    )


def _thresholds(**changes: object) -> MultiscaleGateThresholds:
    values: dict[str, object] = {
        "expected_training_seeds": 3,
        "min_passing_seeds": 2,
        "min_data_end_slices": 2,
        "bootstrap_reps": 100,
        "bootstrap_confidence": 0.90,
    }
    values.update(changes)
    return MultiscaleGateThresholds(**values)


def test_tau_bin_baseline_fits_training_rows_and_has_deterministic_empty_fallback() -> None:
    targets = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float64)
    tau = np.asarray([0.1, 0.9], dtype=np.float64)
    baseline = fit_tau_bin_baseline(targets, tau)

    assert baseline.counts == (1, 0, 0, 0, 1)
    assert np.array_equal(baseline.means[0], targets[0])
    assert np.array_equal(baseline.means[4], targets[1])
    assert np.array_equal(baseline.means[2], targets.mean(axis=0))
    predicted = predict_tau_bin_baseline(baseline, [0.1, 0.5, 1.0])
    assert np.array_equal(predicted[0], targets[0])
    assert np.array_equal(predicted[1], targets.mean(axis=0))
    assert np.array_equal(predicted[2], targets[1])


def test_split_metrics_are_whole_path_isolated_binned_and_train_only() -> None:
    targets, predictions, tau, path_ids, splits = _pilot_arrays()
    first = compute_multiscale_split_metrics(
        targets,
        predictions,
        tau,
        path_ids,
        splits,
        stride=16,
        training_seed=11,
        selected_step=50,
    )
    changed_targets = targets.copy()
    changed_targets[np.isin(path_ids, splits["audit"])] += 1000.0
    second = compute_multiscale_split_metrics(
        changed_targets,
        predictions,
        tau,
        path_ids,
        splits,
        stride=16,
        training_seed=11,
        selected_step=50,
    )

    assert first["tau_baseline"] == second["tau_baseline"]
    assert len(first["split_metrics"]) == 3
    assert len(first["time_bin_metrics"]) == 15
    assert len(first["per_path_metrics"]) == 6 * 6
    audit_paths = {
        row["path_id"]
        for row in first["per_path_metrics"]
        if row["split"] == "audit"
    }
    assert audit_paths == {4, 5}
    data_end = next(
        row
        for row in first["time_bin_metrics"]
        if row["split"] == "audit" and row["bin_index"] == 4
    )
    assert data_end["slice_count"] == 2
    assert data_end["finite_fraction"] == 1.0
    assert data_end["prediction_gain"] > 0.0
    assert data_end["prediction_gain_vs_tau_baseline"] > 0.0


def test_split_metrics_reject_overlap_omission_and_nonfinite_tau() -> None:
    targets, predictions, tau, path_ids, splits = _pilot_arrays()
    overlapping = copy.deepcopy(splits)
    overlapping["audit"] = [3, 4, 5]
    with pytest.raises(ValueError, match="overlap"):
        compute_multiscale_split_metrics(
            targets,
            predictions,
            tau,
            path_ids,
            overlapping,
            stride=1,
            training_seed=1,
            selected_step=1,
        )
    omitted = copy.deepcopy(splits)
    omitted["audit"] = [4]
    with pytest.raises(ValueError, match="cover"):
        compute_multiscale_split_metrics(
            targets,
            predictions,
            tau,
            path_ids,
            omitted,
            stride=1,
            training_seed=1,
            selected_step=1,
        )
    bad_tau = tau.copy()
    bad_tau[0] = np.nan
    with pytest.raises(ValueError, match="tau_fractions"):
        compute_multiscale_split_metrics(
            targets,
            predictions,
            bad_tau,
            path_ids,
            splits,
            stride=1,
            training_seed=1,
            selected_step=1,
        )


def test_whole_path_bootstrap_is_order_invariant_and_clusters_seed_rows() -> None:
    rows = [
        {"path_id": path, "squared_error_sum": 1.0, "zero_squared_error_sum": 4.0}
        for path in (7, 3, 7, 3)
    ]
    first = bootstrap_whole_path_gain(
        rows,
        baseline_error_key="zero_squared_error_sum",
        reps=100,
        confidence=0.9,
        seed=99,
    )
    second = bootstrap_whole_path_gain(
        list(reversed(rows)),
        baseline_error_key="zero_squared_error_sum",
        reps=100,
        confidence=0.9,
        seed=99,
    )
    assert first == second
    assert first["point_gain"] == pytest.approx(0.75)
    assert first["lower_bound"] == pytest.approx(0.75)
    assert first["path_count"] == 2


def test_whole_path_bootstrap_rejects_bad_or_zero_baselines() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        bootstrap_whole_path_gain(
            [{"path_id": 1, "squared_error_sum": -1, "zero_squared_error_sum": 1}],
            baseline_error_key="zero_squared_error_sum",
            reps=10,
            confidence=0.9,
            seed=1,
        )
    with pytest.raises(ValueError, match="positive"):
        bootstrap_whole_path_gain(
            [{"path_id": 1, "squared_error_sum": 0, "zero_squared_error_sum": 0}],
            baseline_error_key="zero_squared_error_sum",
            reps=10,
            confidence=0.9,
            seed=1,
        )


def test_stride_pass_accepts_reproducible_audit_signal() -> None:
    results = [_seed_result(seed) for seed in (10, 11, 12)]
    gate = evaluate_stride_pass(16, results, _thresholds())

    assert gate["passed"] == 1
    assert gate["subchecks"]["passing_seed_count"]["value"] == 3
    assert gate["bootstrap"]["overall_vs_zero"]["path_count"] == 2
    assert gate["diagnostics"]["audit_point_signal"] == 1
    assert gate["diagnostics"]["train_point_signal"] == 1


def test_stride_pass_fails_closed_for_incomplete_nonfinite_or_sparse_results() -> None:
    base = [_seed_result(seed) for seed in (10, 11, 12)]

    incomplete = copy.deepcopy(base)
    incomplete[0]["task_complete"] = 0
    incomplete_gate = evaluate_stride_pass(16, incomplete, _thresholds())
    assert incomplete_gate["passed"] == 0
    assert incomplete_gate["diagnostics"]["all_tasks_complete"] == 0
    assert incomplete_gate["diagnostics"]["audit_point_signal"] == 0

    missing_seed_gate = evaluate_stride_pass(16, base[:1], _thresholds())
    assert missing_seed_gate["diagnostics"]["all_tasks_complete"] == 0
    assert missing_seed_gate["diagnostics"]["audit_point_signal"] == 0
    decision = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={16: missing_seed_gate},
    )
    assert decision["decision"] == LearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID.value

    nonfinite = copy.deepcopy(base)
    audit = next(
        row
        for row in nonfinite[0]["split_metrics"]
        if row["split"] == "audit" and row["bin_index"] == -1
    )
    audit["prediction_gain"] = float("nan")
    assert evaluate_stride_pass(16, nonfinite, _thresholds())["passed"] == 0

    sparse_thresholds = _thresholds(min_data_end_slices=3)
    sparse = evaluate_stride_pass(16, base, sparse_thresholds)
    assert sparse["passed"] == 0
    assert sparse["subchecks"]["data_end_count"]["passed"] == 0


def test_stride_pass_fails_when_selection_gain_is_negative() -> None:
    results = [_seed_result(seed) for seed in (10, 11, 12)]
    for result in results:
        selection = next(
            row
            for row in result["split_metrics"]
            if row["split"] == "selection" and row["bin_index"] == -1
        )
        selection_data_end = next(
            row
            for row in result["time_bin_metrics"]
            if row["split"] == "selection" and row["bin_index"] == 4
        )
        selection["prediction_gain"] = -1e-6
        selection_data_end["prediction_gain"] = -1e-6

    gate = evaluate_stride_pass(16, results, _thresholds())
    assert gate["passed"] == 0
    assert gate["subchecks"]["median_selection_overall_gain"]["passed"] == 0
    assert gate["subchecks"]["median_selection_data_end_gain"]["passed"] == 0
    assert gate["subchecks"]["passing_seed_count"]["value"] == 0


def test_stride_pass_fails_when_median_audit_covariance_is_negative() -> None:
    results = [_seed_result(seed) for seed in (10, 11, 12)]
    for result in results:
        audit = next(
            row
            for row in result["split_metrics"]
            if row["split"] == "audit" and row["bin_index"] == -1
        )
        audit["target_prediction_covariance"] = -1e-12

    gate = evaluate_stride_pass(16, results, _thresholds())
    assert gate["passed"] == 0
    check = gate["subchecks"]["median_audit_target_prediction_covariance"]
    assert check["passed"] == 0
    assert check["value"] == pytest.approx(-1e-12)


def test_stride_pass_requires_the_declared_seed_count_and_unique_seeds() -> None:
    two = [_seed_result(seed) for seed in (10, 11)]
    assert evaluate_stride_pass(16, two, _thresholds())["passed"] == 0
    duplicate = [_seed_result(seed) for seed in (10, 10, 12)]
    gate = evaluate_stride_pass(16, duplicate, _thresholds())
    assert gate["passed"] == 0
    assert gate["subchecks"]["seed_result_count"]["passed"] == 0


def test_production_gate_requires_exact_twelve_by_sixteen_audit_coverage() -> None:
    thresholds = _thresholds(
        min_data_end_slices=192,
        expected_audit_paths=12,
        expected_data_end_slices_per_path=16,
    )
    results = [_production_coverage_seed_result(seed) for seed in (10, 11, 12)]
    passing = evaluate_stride_pass(16, results, thresholds)
    assert passing["passed"] == 1
    assert passing["subchecks"]["data_end_count"]["value"] == [192, 192, 192]
    assert passing["subchecks"]["audit_path_coverage"]["value"] == [12, 12, 12]

    # Preserve the aggregate 192 count while making the per-path allocation
    # 15+17 instead of the frozen 16+16.  Aggregate-only validation would miss
    # this incomplete/duplicated path coverage.
    corrupted = copy.deepcopy(results)
    for result in corrupted:
        rows = [
            row
            for row in result["per_path_metrics"]
            if row["split"] == "audit" and row["bin_index"] == 4
        ]
        rows[0]["slice_count"] = 15
        rows[1]["slice_count"] = 17
    failed = evaluate_stride_pass(16, corrupted, thresholds)
    assert failed["passed"] == 0
    assert failed["subchecks"]["audit_path_coverage"]["passed"] == 0


def test_confirmation_gate_requires_exact_twenty_four_by_sixteen_coverage() -> None:
    thresholds = _thresholds(
        expected_training_seeds=5,
        min_passing_seeds=3,
        min_data_end_slices=384,
        expected_audit_paths=24,
        expected_data_end_slices_per_path=16,
    )
    results = [
        _confirmation_coverage_seed_result(seed)
        for seed in (260730, 260731, 260732, 260733, 260734)
    ]
    passing = evaluate_stride_pass(16, results, thresholds)
    assert passing["passed"] == 1
    assert passing["subchecks"]["data_end_count"]["value"] == [384] * 5
    assert passing["subchecks"]["audit_path_coverage"]["value"] == [24] * 5
    assert passing["bootstrap"]["overall_vs_zero"]["path_count"] == 24
    assert passing["bootstrap"]["data_end_vs_zero"]["path_count"] == 24

    redistributed = copy.deepcopy(results)
    for result in redistributed:
        rows = [
            row
            for row in result["per_path_metrics"]
            if row["split"] == "audit" and row["bin_index"] == 4
        ]
        rows[0]["slice_count"] = 15
        rows[1]["slice_count"] = 17
    redistributed_gate = evaluate_stride_pass(16, redistributed, thresholds)
    assert redistributed_gate["passed"] == 0
    assert redistributed_gate["subchecks"]["audit_path_coverage"]["passed"] == 0

    duplicate = copy.deepcopy(results)
    for result in duplicate:
        row = next(
            row
            for row in result["per_path_metrics"]
            if row["split"] == "audit" and row["bin_index"] == 4
        )
        result["per_path_metrics"].append(copy.deepcopy(row))
    duplicate_gate = evaluate_stride_pass(16, duplicate, thresholds)
    assert duplicate_gate["passed"] == 0
    assert duplicate_gate["subchecks"]["audit_path_coverage"]["passed"] == 0

    missing = copy.deepcopy(results)
    for result in missing:
        result["split_path_ids"]["audit"].pop()
    missing_gate = evaluate_stride_pass(16, missing, thresholds)
    assert missing_gate["passed"] == 0
    assert missing_gate["subchecks"]["audit_path_coverage"]["passed"] == 0


def test_tau_bootstrap_is_advisory_and_unavailability_does_not_fail_gate() -> None:
    thresholds = _thresholds(
        min_data_end_slices=192,
        expected_audit_paths=12,
        expected_data_end_slices_per_path=16,
    )
    results = [_production_coverage_seed_result(seed) for seed in (10, 11, 12)]
    for result in results:
        for row in result["per_path_metrics"]:
            if row["split"] == "audit" and row["bin_index"] == -1:
                row["tau_baseline_squared_error_sum"] = 0.0

    gate = evaluate_stride_pass(16, results, thresholds)
    assert gate["passed"] == 1
    assert gate["subchecks"]["overall_path_bootstrap_lcb"]["passed"] == 1
    assert gate["subchecks"]["data_end_path_bootstrap_lcb"]["passed"] == 1
    assert "tau_baseline_path_bootstrap_lcb" not in gate["subchecks"]
    tau_bootstrap = gate["bootstrap"]["overall_vs_tau_mean"]
    assert tau_bootstrap["status"] == "unavailable"
    assert tau_bootstrap["gated"] == 0
    assert "baseline error must be positive" in tau_bootstrap["reason"]


def test_per_seed_pass_does_not_require_positive_tau_baseline_gain() -> None:
    thresholds = _thresholds(
        min_data_end_slices=192,
        expected_audit_paths=12,
        expected_data_end_slices_per_path=16,
    )
    results = [_production_coverage_seed_result(seed) for seed in (10, 11, 12)]
    first_audit = next(
        row
        for row in results[0]["split_metrics"]
        if row["split"] == "audit" and row["bin_index"] == -1
    )
    first_audit["prediction_gain_vs_tau_baseline"] = -10.0

    gate = evaluate_stride_pass(16, results, thresholds)
    first_summary = next(
        row for row in gate["seed_summaries"] if row["training_seed"] == 10
    )
    assert first_summary["audit_gain_vs_tau_baseline"] == -10.0
    assert first_summary["seed_signal_pass"] == 1
    assert gate["subchecks"]["passing_seed_count"]["value"] == 3
    assert gate["passed"] == 1  # The cross-seed median tau comparison still passes.


def test_no_detectable_requires_absence_of_every_positive_audit_estimate() -> None:
    positive_point = {
        "passed": 0,
        "diagnostics": {
            "all_tasks_complete": 1,
            "audit_point_signal": 1,
            "median_train_overall_gain": 0.1,
            "max_train_overall_gain": 0.1,
        },
    }
    positive_report = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={1: positive_point},
    )
    assert positive_report["decision"] == LearnabilityDecision.INCONCLUSIVE.value

    no_positive_point = copy.deepcopy(positive_point)
    no_positive_point["diagnostics"]["audit_point_signal"] = 0
    absent_report = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={1: no_positive_point},
    )
    assert absent_report["decision"] == (
        LearnabilityDecision.NO_DETECTABLE_CONDITIONAL_SIGNAL.value
    )


def test_confirmation_no_pass_is_terminal_even_with_positive_audit_hint() -> None:
    positive_point = {
        "passed": 0,
        "diagnostics": {
            "all_tasks_complete": 1,
            "audit_point_signal": 1,
            "median_train_overall_gain": 0.1,
            "max_train_overall_gain": 0.1,
        },
    }
    pilot = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={1024: positive_point},
    )
    assert pilot["decision"] == LearnabilityDecision.INCONCLUSIVE.value

    confirmation = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={1024: positive_point},
        study_profile="confirmation",
        profile_conformant=True,
    )
    assert confirmation["decision"] == (
        LearnabilityDecision.NO_CONFIRMED_CONDITIONAL_SIGNAL.value
    )
    assert confirmation["confirmation_profile"] == 1
    assert confirmation["confirmation_exhausted"] == 1
    assert confirmation["repeat_same_profile_authorized"] == 0
    assert confirmation["sampling_authorized"] == 0

    exploratory = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={1024: positive_point},
        study_profile="confirmation",
        profile_conformant=False,
    )
    assert exploratory["decision"] == LearnabilityDecision.INCONCLUSIVE.value
    assert exploratory["confirmation_exhausted"] == 0


def test_confirmation_path_memorization_precedes_terminal_no_signal() -> None:
    report = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={
            1024: {
                "passed": 0,
                "diagnostics": {
                    "all_tasks_complete": 1,
                    "audit_point_signal": 1,
                    "median_train_overall_gain": 0.50,
                },
            }
        },
        study_profile="confirmation",
    )
    assert report["decision"] == LearnabilityDecision.PATH_MEMORIZATION_ONLY.value
    assert report["confirmation_exhausted"] == 1


def test_teacher_control_boundaries_and_nonfinite_values_fail_closed() -> None:
    thresholds = _thresholds(teacher_min_gain=0.9)
    passing = {
        "complete": 1,
        "selected_step": 1,
        "finite_fraction": 1.0,
        "audit_overall_gain": 0.9,
        "audit_data_end_gain": 0.9,
        "audit_data_end_slice_count": 2,
    }
    assert evaluate_teacher_control(passing, thresholds)["passed"] == 1
    bad = dict(passing, audit_overall_gain=float("nan"))
    assert evaluate_teacher_control(bad, thresholds)["passed"] == 0
    impossible = dict(passing, audit_data_end_gain=1.01)
    assert evaluate_teacher_control(impossible, thresholds)["passed"] == 0


@pytest.mark.parametrize(
    ("cache", "teacher", "gates", "expected"),
    [
        (False, True, {}, LearnabilityDecision.CACHE_INVALID),
        (True, False, {}, LearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID),
        (
            True,
            True,
            {1: {"passed": 1, "diagnostics": {"all_tasks_complete": 1}}},
            LearnabilityDecision.ELEMENTARY_SIGNAL,
        ),
        (
            True,
            True,
            {
                1: {"passed": 0, "diagnostics": {"all_tasks_complete": 1}},
                16: {"passed": 1, "diagnostics": {"all_tasks_complete": 1}},
            },
            LearnabilityDecision.COARSE_ONLY_SIGNAL,
        ),
        (
            True,
            True,
            {
                1: {
                    "passed": 0,
                    "diagnostics": {
                        "all_tasks_complete": 1,
                        "audit_point_signal": 1,
                        "train_point_signal": 1,
                    },
                }
            },
            LearnabilityDecision.INCONCLUSIVE,
        ),
        (
            True,
            True,
            {
                1: {
                    "passed": 0,
                    "diagnostics": {
                        "all_tasks_complete": 1,
                        "audit_point_signal": 0,
                        "train_point_signal": 1,
                        "median_train_overall_gain": 0.50,
                    },
                }
            },
            LearnabilityDecision.PATH_MEMORIZATION_ONLY,
        ),
        (
            True,
            True,
            {
                1: {
                    "passed": 0,
                    "diagnostics": {
                        "all_tasks_complete": 1,
                        "audit_point_signal": 0,
                        "train_point_signal": 0,
                        "median_train_overall_gain": 0.49,
                    },
                }
            },
            LearnabilityDecision.NO_DETECTABLE_CONDITIONAL_SIGNAL,
        ),
        (
            True,
            True,
            {1: {"passed": 0, "diagnostics": {"all_tasks_complete": 0}}},
            LearnabilityDecision.OPTIMIZATION_PIPELINE_INVALID,
        ),
    ],
)
def test_decision_state_machine(
    cache: bool,
    teacher: bool,
    gates: dict[int, dict[str, object]],
    expected: LearnabilityDecision,
) -> None:
    report = decide_learnability(
        cache_gate=cache,
        teacher_gate=teacher,
        stride_gates=gates,
    )
    assert report["decision"] == expected.value
    assert report["sampling_performed"] == 0
    assert report["sampling_authorized"] == 0


def test_decision_actions_match_the_approved_next_workflow() -> None:
    elementary = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={
            1: {"passed": 1, "diagnostics": {"all_tasks_complete": 1}}
        },
    )
    assert elementary["recommended_next_action"] == (
        "launch a fresh strict r=1 reconstruction run"
    )

    coarse = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={
            1: {"passed": 0, "diagnostics": {"all_tasks_complete": 1}},
            16: {"passed": 1, "diagnostics": {"all_tasks_complete": 1}},
        },
    )
    assert coarse["recommended_next_action"] == (
        "plan a separately named coarse sampler plus conditional-noise calibration; "
        "this workflow performs no sampling"
    )


def test_memorization_decision_requires_training_gain_at_least_one_half() -> None:
    def report(train_gain: float) -> dict[str, object]:
        return decide_learnability(
            cache_gate=True,
            teacher_gate=True,
            stride_gates={
                1: {
                    "passed": 0,
                    "diagnostics": {
                        "all_tasks_complete": 1,
                        "audit_point_signal": 0,
                        "median_train_overall_gain": train_gain,
                    },
                }
            },
        )

    assert report(0.50)["decision"] == LearnabilityDecision.PATH_MEMORIZATION_ONLY.value
    assert report(0.499999)["decision"] == (
        LearnabilityDecision.NO_DETECTABLE_CONDITIONAL_SIGNAL.value
    )
    any_seed_memorizes = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={
            1: {
                "passed": 0,
                "diagnostics": {
                    "all_tasks_complete": 1,
                    "audit_point_signal": 0,
                    "median_train_overall_gain": 0.10,
                    "max_train_overall_gain": 0.50,
                },
            }
        },
    )
    assert any_seed_memorizes["decision"] == (
        LearnabilityDecision.PATH_MEMORIZATION_ONLY.value
    )
    positive_audit = decide_learnability(
        cache_gate=True,
        teacher_gate=True,
        stride_gates={
            1: {
                "passed": 0,
                "diagnostics": {
                    "all_tasks_complete": 1,
                    "audit_point_signal": 1,
                    "median_train_overall_gain": 0.99,
                },
            }
        },
    )
    assert positive_audit["decision"] == LearnabilityDecision.INCONCLUSIVE.value


def test_cumulative_gate_semantics_never_authorize_sampling() -> None:
    elementary = {"passed": 1, "diagnostics": {"all_tasks_complete": 1}}
    report = evaluate_multiscale_gates(
        cache_gate={"passed": 1},
        teacher_gate={"passed": 1},
        stride_gates={1: elementary},
        require_gate="elementary",
        thresholds=_thresholds(),
    )
    assert report["required_gate_pass"] == 1
    assert report["cumulative_pass"] == {
        "cache": 1,
        "teacher": 1,
        "any-scale": 1,
        "elementary": 1,
    }
    assert report["sampling_authorized"] == 0
    assert report["schema_version"] == 2
    assert report["study_profile"] == "pilot"
    assert report["profile_conformant"] == 1
    assert report["authoritative_decision"] == 1

    broken = evaluate_multiscale_gates(
        cache_gate={"passed": 0},
        teacher_gate={"passed": 1},
        stride_gates={1: elementary},
        require_gate="elementary",
        thresholds=_thresholds(),
    )
    assert broken["required_gate_pass"] == 0
    with pytest.raises(ValueError, match="require_gate"):
        evaluate_multiscale_gates(
            cache_gate=True,
            teacher_gate=True,
            stride_gates={},
            require_gate="sampling",
            thresholds=_thresholds(),
        )


def test_final_gate_marks_exploratory_and_confirmation_reports_fail_closed() -> None:
    passing_stride = {"passed": 1, "diagnostics": {"all_tasks_complete": 1}}
    exploratory = evaluate_multiscale_gates(
        cache_gate={"passed": 1},
        teacher_gate={"passed": 1},
        stride_gates={16: passing_stride},
        require_gate="any-scale",
        thresholds=_thresholds(),
        study_profile="confirmation",
        profile_conformant=False,
    )
    assert exploratory["cumulative_pass"]["any-scale"] == 1
    assert exploratory["required_gate_pass"] == 0
    assert exploratory["authoritative_decision"] == 0
    assert exploratory["decision"]["authoritative_decision"] == 0

    no_required_gate = evaluate_multiscale_gates(
        cache_gate={"passed": 1},
        teacher_gate={"passed": 1},
        stride_gates={16: passing_stride},
        require_gate="none",
        thresholds=_thresholds(),
        study_profile="confirmation",
        profile_conformant=True,
    )
    assert no_required_gate["required_gate_pass"] == 1
    assert no_required_gate["authoritative_decision"] == 0

    terminal = evaluate_multiscale_gates(
        cache_gate={"passed": 1},
        teacher_gate={"passed": 1},
        stride_gates={
            1024: {
                "passed": 0,
                "diagnostics": {
                    "all_tasks_complete": 1,
                    "audit_point_signal": 1,
                    "median_train_overall_gain": 0.1,
                },
            }
        },
        require_gate="any-scale",
        thresholds=_thresholds(),
        study_profile="confirmation",
        profile_conformant=True,
    )
    assert terminal["schema_version"] == 2
    assert terminal["required_gate_pass"] == 0
    assert terminal["authoritative_decision"] == 1
    assert terminal["confirmation_exhausted"] == 1
    assert terminal["decision"]["decision"] == (
        LearnabilityDecision.NO_CONFIRMED_CONDITIONAL_SIGNAL.value
    )


def test_profile_context_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="study_profile"):
        decide_learnability(
            cache_gate=True,
            teacher_gate=True,
            stride_gates={},
            study_profile="future",
        )
    with pytest.raises(ValueError, match="profile_conformant"):
        evaluate_multiscale_gates(
            cache_gate=True,
            teacher_gate=True,
            stride_gates={},
            profile_conformant=2,
        )
    with pytest.raises(ValueError, match="authoritative"):
        decide_learnability(
            cache_gate=True,
            teacher_gate=True,
            stride_gates={},
            profile_conformant=False,
            authoritative_decision=True,
        )


def test_atomic_artifact_writer_flattens_every_table(tmp_path: Path) -> None:
    results = [_seed_result(seed) for seed in (10, 11, 12)]
    stride_gate = evaluate_stride_pass(16, results, _thresholds())
    report = evaluate_multiscale_gates(
        cache_gate={"passed": 1},
        teacher_gate={"passed": 1},
        stride_gates={16: stride_gate},
        require_gate="any-scale",
        thresholds=_thresholds(),
    )
    paths = write_multiscale_gate_artifacts(tmp_path / "nested", results, report)

    assert all(path.is_file() for path in paths.values())
    persisted = json.loads(paths["decision"].read_text(encoding="utf-8"))
    assert persisted["required_gate_pass"] == 1
    with paths["split_metrics"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9


def test_thresholds_validate_domains() -> None:
    with pytest.raises(ValueError, match="min_passing_seeds"):
        MultiscaleGateThresholds(
            expected_training_seeds=2,
            min_passing_seeds=3,
        )
    with pytest.raises(ValueError, match="bootstrap_confidence"):
        MultiscaleGateThresholds(bootstrap_confidence=1.0)
    with pytest.raises(ValueError, match="teacher_min_gain"):
        MultiscaleGateThresholds(teacher_min_gain=float("nan"))
    with pytest.raises(ValueError, match="memorization_train_gain"):
        MultiscaleGateThresholds(memorization_train_gain=1.01)


def test_module_has_no_sampler_dependency() -> None:
    source = Path(multiscale.__file__).read_text(encoding="utf-8")
    assert "d0_one_image_sampler" not in source
    assert "run_paired" not in source
