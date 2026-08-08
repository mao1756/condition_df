from __future__ import annotations

import numpy as np
import pytest

from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE, PHASE_COUNT
from mnist.d0_jacobi_rb_zero_signal_diagnostic import (
    coarse_cell_path_means,
    cross_split_coarse_signal,
    diagnostic_conclusion,
    path_decomposition_rows,
    quadratic_risk_decomposition,
    stratified_decomposition_rows,
    whole_path_bootstrap_interval,
)


def test_quadratic_risk_identity_and_zero_comparison() -> None:
    target = np.full((3, EDGES_PER_PHASE), 2.0, dtype=np.float64)
    prediction = np.full_like(target, 0.5)
    metadata = np.full_like(target, 1.0)
    result = quadratic_risk_decomposition(prediction, target, metadata)
    assert result.target_energy == 4.0
    assert result.prediction_energy == 0.25
    assert result.target_prediction_inner_product == 1.0
    assert result.model_mse == 2.25
    assert result.zero_minus_model_mse == 1.75
    assert result.model_beats_zero
    assert result.model_identity_abs_error <= 1.0e-15
    assert result.metadata_identity_abs_error <= 1.0e-15


def test_prediction_energy_can_dominate_alignment() -> None:
    target = np.ones((2, EDGES_PER_PHASE), dtype=np.float64)
    prediction = np.full_like(target, -0.1)
    result = quadratic_risk_decomposition(
        prediction, target, np.zeros_like(target)
    )
    assert result.zero_minus_model_mse < 0.0
    assert not result.covariance_exceeds_prediction_cost


def _structured_fixture(paths: tuple[int, ...] = (11, 12)) -> tuple[np.ndarray, ...]:
    rows = [
        (path, quartile * 128 + offset, phase)
        for path in paths
        for quartile in range(4)
        for offset in range(8)
        for phase in range(PHASE_COUNT)
    ]
    path_id = np.asarray([row[0] for row in rows], dtype=np.int64)
    outer_step = np.asarray([row[1] for row in rows], dtype=np.int64)
    phase = np.asarray([row[2] for row in rows], dtype=np.int64)
    target = np.empty((len(rows), EDGES_PER_PHASE), dtype=np.float64)
    for index, (path, step, phase_index) in enumerate(rows):
        target[index] = (
            0.01 * path
            + 0.001 * (step // 128)
            + 0.0001 * phase_index
            + np.arange(EDGES_PER_PHASE, dtype=np.float64) * 1.0e-6
        )
    return target, path_id, outer_step, phase


def test_path_and_stratified_rows_cover_frozen_partition() -> None:
    target, path_id, outer_step, phase = _structured_fixture()
    prediction = 0.5 * target
    metadata = 0.25 * target
    path_rows = path_decomposition_rows(
        prediction, target, metadata, path_id, split="validation"
    )
    strata = stratified_decomposition_rows(
        prediction,
        target,
        metadata,
        outer_step,
        phase,
        split="validation",
    )
    assert [row["path_id"] for row in path_rows] == [11, 12]
    assert len(strata) == 1 + PHASE_COUNT + 4 + 4 * PHASE_COUNT
    assert all(row["model_beats_zero"] == 1 for row in strata)


def test_whole_path_bootstrap_is_deterministic() -> None:
    rows = [
        {"zero_minus_model_mse": value}
        for value in (-0.3, -0.1, 0.2, 0.4)
    ]
    first = whole_path_bootstrap_interval(
        rows, field="zero_minus_model_mse", seed=17, replicates=1_000
    )
    second = whole_path_bootstrap_interval(
        rows, field="zero_minus_model_mse", seed=17, replicates=1_000
    )
    assert first == second
    assert first["bootstrap_unit"] == "whole_path"
    assert first["non_authorizing_posthoc"] == 1


def test_coarse_path_cells_and_cross_split_bootstrap() -> None:
    left_target, left_paths, steps, phases = _structured_fixture((11, 12))
    right_target, right_paths, right_steps, right_phases = _structured_fixture(
        (21, 22)
    )
    _, left = coarse_cell_path_means(left_target, left_paths, steps, phases)
    _, right = coarse_cell_path_means(
        right_target, right_paths, right_steps, right_phases
    )
    assert left.shape == (2, 4, PHASE_COUNT, EDGES_PER_PHASE)
    record = cross_split_coarse_signal(
        left,
        right,
        left_split="train",
        right_split="confirmation",
        seed=19,
        replicates=200,
        chunk_size=17,
    )
    replay = cross_split_coarse_signal(
        left,
        right,
        left_split="train",
        right_split="confirmation",
        seed=19,
        replicates=200,
        chunk_size=17,
    )
    assert record == replay
    assert record["observations_per_path_cell"] == 8
    assert record["coarse_cell_count"] == 4 * PHASE_COUNT * EDGES_PER_PHASE
    assert record["cross_split_coarse_signal"] > 0.0


def test_coarse_cell_counts_fail_closed() -> None:
    target, path_id, outer_step, phase = _structured_fixture((11,))
    with pytest.raises(ValueError, match="exactly eight"):
        coarse_cell_path_means(
            target[:-1], path_id[:-1], outer_step[:-1], phase[:-1]
        )


def test_conclusion_limits_claim_scope() -> None:
    summaries = {
        "validation": {"zero_minus_model_mse": -1.0e-5},
        "confirmation": {"zero_minus_model_mse": -2.0e-5},
    }
    coarse = [{"interval_contains_zero": 1}] * 3
    record = diagnostic_conclusion(summaries, coarse)
    assert record["conclusion"] == "frozen_model_does_not_beat_zero"
    assert (
        record["coarse_conditional_signal_conclusion"]
        == "coarse_conditional_signal_inconclusive"
    )
    assert record["conditional_mean_identically_zero_proven"] == 0
    assert record["population_signal_absence_proven"] == 0
