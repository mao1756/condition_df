from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_COUNT,
    SELECTED_OUTER_STEPS,
)
from mnist.d0_jacobi_rb_physical_coarse_signal import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COARSE_CELL_COUNT,
    PANEL_A_PATH_IDS,
    PANEL_B_PATH_IDS,
    PHYSICAL_COARSE_SIGNAL_VERSION,
    RESOLUTION_TARGET,
    PhysicalCoarsePanel,
    PhysicalCoarsePathPlan,
    PhysicalCoarseSignalError,
    PhysicalCoarseStatisticPlan,
    analyze_cross_panel_signal,
    bayes_control_split_pairs,
    classify_cross_panel_signal,
    coarse_cell_path_means,
    cross_panel_path_kernel,
    cross_panel_point_estimate,
    evaluate_bayes_control_replay,
    frozen_path_plan,
    frozen_statistic_plan,
    welch_delta_cross_panel_bounds,
    whole_path_cross_panel_bootstrap,
)


def _panel(role: str, path_base: int, path_values: list[float]) -> PhysicalCoarsePanel:
    values = np.asarray(path_values, dtype=np.float64)
    cells = np.broadcast_to(
        values[:, None, None, None],
        (values.size, 4, PHASE_COUNT, EDGES_PER_PHASE),
    ).copy()
    return PhysicalCoarsePanel(
        role=role,
        path_ids=np.arange(path_base, path_base + values.size, dtype=np.int64),
        cell_means=cells,
    )


def _method(
    *,
    point: float = 0.0,
    lower: float = -1.0e-4,
    upper: float = 1.0e-4,
) -> dict[str, float]:
    return {
        "point_estimate": point,
        "lower_bound": lower,
        "upper_bound": upper,
        "confidence": 0.99,
    }


def test_frozen_path_and_statistic_plans() -> None:
    paths = frozen_path_plan()
    statistic = frozen_statistic_plan()
    assert isinstance(paths, PhysicalCoarsePathPlan)
    assert isinstance(statistic, PhysicalCoarseStatisticPlan)
    assert paths.panel_a == PANEL_A_PATH_IDS
    assert paths.panel_b == PANEL_B_PATH_IDS
    assert len(set(paths.panel_a).intersection(paths.panel_b)) == 0
    assert len(paths.fingerprint) == 64
    assert statistic.bootstrap_replicates == BOOTSTRAP_REPLICATES
    assert statistic.bootstrap_seed == BOOTSTRAP_SEED
    assert statistic.resolution_target == RESOLUTION_TARGET
    assert statistic.to_record()["coarse_cell_count"] == COARSE_CELL_COUNT
    assert PHYSICAL_COARSE_SIGNAL_VERSION in statistic.version


def test_frozen_plans_reject_changes_and_invalid_path_types() -> None:
    with pytest.raises(PhysicalCoarseSignalError, match="constants changed"):
        PhysicalCoarseStatisticPlan(bootstrap_replicates=49_999)
    with pytest.raises(PhysicalCoarseSignalError, match="integers"):
        PhysicalCoarsePathPlan(
            panel_a=tuple(float(value) for value in PANEL_A_PATH_IDS)  # type: ignore[arg-type]
        )
    with pytest.raises(PhysicalCoarseSignalError, match="20-bit"):
        PhysicalCoarsePathPlan(
            panel_a=(*PANEL_A_PATH_IDS[:-1], 1 << 20)
        )
    with pytest.raises(PhysicalCoarseSignalError, match="overlap"):
        PhysicalCoarsePathPlan(panel_b=PANEL_A_PATH_IDS)


def test_coarse_cell_means_equal_explicit_slow_sum() -> None:
    paths = np.repeat(np.asarray([19, 23], dtype=np.int64), 32 * PHASE_COUNT)
    steps = np.tile(
        np.repeat(np.asarray(SELECTED_OUTER_STEPS, dtype=np.int64), PHASE_COUNT),
        2,
    )
    phases = np.tile(
        np.tile(np.arange(PHASE_COUNT, dtype=np.int64), 32), 2
    )
    edge = np.arange(EDGES_PER_PHASE, dtype=np.float64)
    target = np.empty((paths.size, EDGES_PER_PHASE), dtype=np.float64)
    for row in range(paths.size):
        target[row] = (
            0.01 * float(paths[row])
            + 0.001 * float(steps[row])
            + 0.1 * float(phases[row])
            + 1.0e-5 * edge
        )
    permutation = np.random.default_rng(11).permutation(paths.size)
    panel = coarse_cell_path_means(
        target[permutation],
        paths[permutation],
        steps[permutation],
        phases[permutation],
        role="fixture",
    )
    assert panel.path_ids.tolist() == [19, 23]
    assert panel.cell_means.shape == (2, 4, PHASE_COUNT, EDGES_PER_PHASE)
    for path_index, path in enumerate((19, 23)):
        for quartile in range(4):
            quartile_steps = [
                value
                for value in SELECTED_OUTER_STEPS
                if value // 128 == quartile
            ]
            for phase in range(PHASE_COUNT):
                expected = np.asarray(
                    [
                        math.fsum(
                            0.01 * path
                            + 0.001 * step
                            + 0.1 * phase
                            + 1.0e-5 * edge_index
                            for step in quartile_steps
                        )
                        / 8.0
                        for edge_index in range(EDGES_PER_PHASE)
                    ],
                    dtype=np.float64,
                )
                np.testing.assert_allclose(
                    panel.cell_means[path_index, quartile, phase],
                    expected,
                    rtol=0.0,
                    atol=4.0e-16,
                )


def test_coarse_cell_means_reject_incomplete_or_duplicate_design() -> None:
    path = np.repeat(np.asarray([19], dtype=np.int64), 32 * PHASE_COUNT)
    step = np.repeat(np.asarray(SELECTED_OUTER_STEPS, dtype=np.int64), PHASE_COUNT)
    phase = np.tile(np.arange(PHASE_COUNT, dtype=np.int64), 32)
    target = np.zeros((path.size, EDGES_PER_PHASE), dtype=np.float64)
    with pytest.raises(PhysicalCoarseSignalError, match="row count"):
        coarse_cell_path_means(
            target[:-1], path[:-1], step[:-1], phase[:-1], role="bad"
        )
    bad_step = step.copy()
    bad_step[0] = 0
    with pytest.raises(PhysicalCoarseSignalError, match="unselected"):
        coarse_cell_path_means(target, path, bad_step, phase, role="bad")
    duplicate_step = step.copy()
    duplicate_phase = phase.copy()
    duplicate_step[0] = duplicate_step[1]
    duplicate_phase[0] = duplicate_phase[1]
    with pytest.raises(PhysicalCoarseSignalError, match="exactly once"):
        coarse_cell_path_means(
            target, path, duplicate_step, duplicate_phase, role="bad"
        )


def test_cross_panel_kernel_and_point_match_explicit_formula() -> None:
    left = _panel("panel-a", 100, [1.0, 3.0])
    right = _panel("panel-b", 200, [2.0, 4.0, 5.0])
    expected = np.outer([1.0, 3.0], [2.0, 4.0, 5.0])
    np.testing.assert_array_equal(cross_panel_path_kernel(left, right), expected)
    assert cross_panel_point_estimate(left, right) == pytest.approx(
        float(expected.mean()), rel=0.0, abs=0.0
    )
    slow = np.mean(
        left.cell_means.mean(axis=0) * right.cell_means.mean(axis=0)
    )
    assert cross_panel_point_estimate(left, right) == pytest.approx(slow)


def test_whole_path_bootstrap_is_deterministic_and_preserves_negative_values() -> None:
    left = _panel("panel-a", 100, [1.0, 2.0, 3.0])
    right = _panel("panel-b", 200, [-1.0, -2.0, -3.0])
    first = whole_path_cross_panel_bootstrap(
        left,
        right,
        seed=7,
        replicates=257,
        namespace=9,
        chunk_size=31,
    )
    replay = whole_path_cross_panel_bootstrap(
        left,
        right,
        seed=7,
        replicates=257,
        namespace=9,
        chunk_size=31,
    )
    assert first == replay
    different_chunking = whole_path_cross_panel_bootstrap(
        left,
        right,
        seed=7,
        replicates=257,
        namespace=9,
        chunk_size=73,
    )
    assert different_chunking["point_estimate"] == first["point_estimate"]
    assert different_chunking["lower_bound"] == first["lower_bound"]
    assert different_chunking["upper_bound"] == first["upper_bound"]
    assert (
        different_chunking["central_99_lower_bound"]
        == first["central_99_lower_bound"]
    )
    assert (
        different_chunking["central_99_upper_bound"]
        == first["central_99_upper_bound"]
    )
    assert first["point_estimate"] == -4.0
    assert first["upper_bound"] < 0.0
    assert first["negative_values_truncated"] == 0
    assert first["bootstrap_unit"] == "whole_path_independently_within_panel"


def test_bootstrap_uses_independent_panel_namespaces() -> None:
    # Identical path values would yield identical resampled panel means if the
    # same path-index stream were incorrectly reused for both panels.  The
    # independent streams leave nonzero product variation.
    left = _panel("panel-a", 100, [-3.0, -1.0, 1.0, 3.0])
    right = _panel("panel-b", 200, [-3.0, -1.0, 1.0, 3.0])
    record = whole_path_cross_panel_bootstrap(
        left, right, seed=13, replicates=1000, namespace=3, chunk_size=79
    )
    assert record["lower_bound"] < 0.0 < record["upper_bound"]


def test_welch_delta_matches_hand_calculation() -> None:
    left = _panel("panel-a", 100, [1.0, 3.0])
    right = _panel("panel-b", 200, [2.0, 4.0])
    record = welch_delta_cross_panel_bounds(left, right)
    point = 6.0
    left_component = np.var([3.0, 9.0], ddof=1) / 2.0
    right_component = np.var([4.0, 8.0], ddof=1) / 2.0
    variance = left_component + right_component
    degrees = variance**2 / (
        left_component**2 / 1.0 + right_component**2 / 1.0
    )
    critical = stats.t.ppf(0.99, degrees)
    central_critical = stats.t.ppf(0.995, degrees)
    assert record["point_estimate"] == point
    assert record["left_variance_component"] == left_component
    assert record["right_variance_component"] == right_component
    assert record["degrees_of_freedom"] == pytest.approx(degrees)
    assert record["lower_bound"] == pytest.approx(
        point - critical * math.sqrt(variance)
    )
    assert record["upper_bound"] == pytest.approx(
        point + critical * math.sqrt(variance)
    )
    assert record["central_99_lower_bound"] == pytest.approx(
        point - central_critical * math.sqrt(variance)
    )
    assert record["central_99_upper_bound"] == pytest.approx(
        point + central_critical * math.sqrt(variance)
    )
    assert sum(record["left_influence"]) == pytest.approx(0.0)
    assert sum(record["right_influence"]) == pytest.approx(0.0)


def test_zero_variance_welch_bounds_are_exact() -> None:
    left = _panel("panel-a", 100, [2.0, 2.0])
    right = _panel("panel-b", 200, [3.0, 3.0])
    record = welch_delta_cross_panel_bounds(left, right)
    assert record["point_estimate"] == 6.0
    assert record["lower_bound"] == 6.0
    assert record["upper_bound"] == 6.0
    assert record["central_99_lower_bound"] == 6.0
    assert record["central_99_upper_bound"] == 6.0
    assert record["standard_error"] == 0.0
    assert math.isinf(record["degrees_of_freedom"])


def test_classification_detection_resolution_and_inconclusive_boundaries() -> None:
    smallest_positive = np.nextafter(0.0, math.inf)
    threshold_above = np.nextafter(RESOLUTION_TARGET, math.inf)
    detected = classify_cross_panel_signal(
        _method(point=1.0e-4, lower=smallest_positive, upper=2.0e-4),
        _method(point=1.0e-4, lower=smallest_positive, upper=2.0e-4),
    )
    assert detected["decision"] == "exact_physical_coarse_signal_detected"

    resolved = classify_cross_panel_signal(
        _method(point=0.0, lower=-1.0e-4, upper=RESOLUTION_TARGET),
        _method(point=0.0, lower=0.0, upper=RESOLUTION_TARGET),
    )
    assert (
        resolved["decision"]
        == "coarse_signal_below_preregistered_resolution"
    )

    disagreement = classify_cross_panel_signal(
        _method(point=1.0e-4, lower=smallest_positive, upper=3.0e-4),
        _method(point=1.0e-4, lower=0.0, upper=3.0e-4),
    )
    assert disagreement["decision"] == "physical_coarse_signal_inconclusive"

    unresolved = classify_cross_panel_signal(
        _method(point=0.0, lower=-1.0e-4, upper=threshold_above),
        _method(point=0.0, lower=-1.0e-4, upper=RESOLUTION_TARGET),
    )
    assert unresolved["decision"] == "physical_coarse_signal_inconclusive"


def test_classification_fails_closed_on_invalid_or_inconsistent_methods() -> None:
    invalid = classify_cross_panel_signal(
        _method(lower=1.0, upper=-1.0), _method()
    )
    assert invalid["decision"] == "physical_coarse_signal_estimator_invalid"
    mismatched = classify_cross_panel_signal(
        _method(point=0.0), _method(point=1.0e-3)
    )
    assert mismatched["decision"] == "physical_coarse_signal_estimator_invalid"
    nonfinite = classify_cross_panel_signal(
        _method(point=math.nan), _method(point=math.nan)
    )
    assert nonfinite["decision"] == "physical_coarse_signal_estimator_invalid"


def test_full_analysis_uses_both_authorizing_methods() -> None:
    left = _panel("panel-a", 100, [1.0, 1.0, 1.0])
    right = _panel("panel-b", 200, [2.0, 2.0, 2.0])
    result = analyze_cross_panel_signal(
        left, right, seed=5, replicates=100, chunk_size=13
    )
    assert result["bootstrap"]["point_estimate"] == 2.0
    assert result["welch_delta"]["point_estimate"] == 2.0
    assert (
        result["classification"]["decision"]
        == "exact_physical_coarse_signal_detected"
    )
    assert result["conditional_mean_identically_zero_proven"] == 0
    assert result["physical_training_performed"] == 0


def test_bayes_control_pairs_and_teacher_null_semantics() -> None:
    teacher = {
        "train": _panel("teacher-train", 100, [1.0, 1.0]),
        "validation": _panel("teacher-validation", 200, [1.0, 1.0]),
        "confirmation": _panel("teacher-confirmation", 300, [1.0, 1.0]),
    }
    null = {
        "train": _panel("null-train", 400, [0.0, 0.0]),
        "validation": _panel("null-validation", 500, [0.0, 0.0]),
        "confirmation": _panel("null-confirmation", 600, [0.0, 0.0]),
    }
    pairs = bayes_control_split_pairs(teacher)
    assert [(left, right) for left, right, _, _ in pairs] == [
        ("train", "validation"),
        ("train", "confirmation"),
        ("validation", "confirmation"),
    ]
    result = evaluate_bayes_control_replay(
        teacher_panels=teacher,
        null_panels=null,
        seed=17,
        replicates=101,
        chunk_size=19,
    )
    assert result["passed"] == 1
    assert result["pair_count"] == 6
    assert result["teacher_pair_count"] == 3
    assert result["null_pair_count"] == 3
    assert all(row["passed"] == 1 for row in result["rows"])
    assert all(
        row["bootstrap_central_99_lower_bound"]
        <= 0.0
        <= row["bootstrap_central_99_upper_bound"]
        for row in result["rows"]
        if row["law"] == "null"
    )
    assert result["source"] == "immutable_noisy_label_caches_not_oracle_means"


def test_bayes_controls_fail_closed_on_missing_split_or_false_null() -> None:
    teacher = {
        "train": _panel("teacher-train", 100, [1.0, 1.0]),
        "validation": _panel("teacher-validation", 200, [1.0, 1.0]),
    }
    with pytest.raises(PhysicalCoarseSignalError, match="exactly"):
        bayes_control_split_pairs(teacher)

    complete_teacher = {
        **teacher,
        "confirmation": _panel("teacher-confirmation", 300, [1.0, 1.0]),
    }
    false_null = {
        "train": _panel("null-train", 400, [1.0, 1.0]),
        "validation": _panel("null-validation", 500, [1.0, 1.0]),
        "confirmation": _panel("null-confirmation", 600, [1.0, 1.0]),
    }
    result = evaluate_bayes_control_replay(
        teacher_panels=complete_teacher,
        null_panels=false_null,
        seed=17,
        replicates=31,
        chunk_size=7,
    )
    assert result["passed"] == 0
    assert any(
        row["law"] == "null" and row["passed"] == 0
        for row in result["rows"]
    )


def test_panels_reject_overlap_noncanonical_order_and_bad_shape() -> None:
    with pytest.raises(PhysicalCoarseSignalError, match="canonical"):
        PhysicalCoarsePanel(
            role="bad",
            path_ids=np.asarray([2, 1], dtype=np.int64),
            cell_means=np.zeros(
                (2, 4, PHASE_COUNT, EDGES_PER_PHASE), dtype=np.float64
            ),
        )
    with pytest.raises(PhysicalCoarseSignalError, match="shape"):
        PhysicalCoarsePanel(
            role="bad",
            path_ids=np.asarray([1, 2], dtype=np.int64),
            cell_means=np.zeros((2, COARSE_CELL_COUNT), dtype=np.float64),
        )
    left = _panel("left", 100, [1.0, 2.0])
    overlap = _panel("right", 101, [1.0, 2.0])
    with pytest.raises(PhysicalCoarseSignalError, match="disjoint"):
        cross_panel_path_kernel(left, overlap)
