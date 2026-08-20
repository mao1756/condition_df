from __future__ import annotations

import inspect

import numpy as np
import pytest

from mnist import diag_d0_jacobi_rb_rollout_reweighted as workflow
from mnist.d0_jacobi_rb_learnability import MODEL_INPUT_FIELDS, PHASE_DURATIONS, PHASE_MATCHINGS, STATE_SIZE
from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time
from mnist.d0_jacobi_rb_rollout_reweight import (
    ACTIVE_OUTER_STEPS, PROTECTED_CONFIRMATION_PATH_IDS, TRAIN_PATH_IDS,
    VALIDATION_PATH_IDS, RolloutReweightError, augment_mapping,
    build_rollout_reweighting, validate_cache_role_path_ids,
)


Q_NEAREST_ENDPOINT = 15.0 / 16.0
ALL_MIDPOINTS = tuple((2 * index + 1) / 16.0 for index in range(8))


def _state(delta: float = 0.0) -> np.ndarray:
    value = np.full(STATE_SIZE, 1.0 / STATE_SIZE, dtype=np.float64)
    value[0] += delta
    value[1] -= delta
    return value.astype(np.float32)


def _inputs(rows: list[tuple[int, int, float, float]]) -> dict[str, np.ndarray]:
    outer_steps = np.asarray([row[0] for row in rows], dtype=np.int64)
    phases = np.asarray([row[1] for row in rows], dtype=np.int8)
    fractions = np.asarray([row[2] for row in rows], dtype=np.float64)
    times = [
        internal_reverse_time(int(step), int(phase), float(fraction))
        for step, phase, fraction in zip(outer_steps, phases, fractions, strict=True)
    ]
    return {
        "later_full_state": np.stack([_state(row[3]) for row in rows]),
        "reverse_time": np.asarray(times, dtype=np.float64),
        "phase": phases,
        "color": np.asarray(PHASE_MATCHINGS, dtype=np.int8)[phases],
        "duration": np.asarray(PHASE_DURATIONS, dtype=np.float64)[phases],
        "label": np.full(len(rows), 3, dtype=np.int64),
    }


def _complete_rows(deltas: tuple[float, ...], *, include_ineligible: bool = False) -> list[tuple[int, int, float, float]]:
    rows = [
        (outer_step, phase, Q_NEAREST_ENDPOINT, delta)
        for outer_step in ACTIVE_OUTER_STEPS
        for phase in range(7)
        for delta in deltas
    ]
    if include_ineligible:
        rows.extend([(287, 0, Q_NEAREST_ENDPOINT, 0.0), (303, 0, 13.0 / 16.0, 0.0)])
    return rows


def _all_midpoint_rows(delta: float = 0.001) -> list[tuple[int, int, float, float]]:
    return [
        (outer_step, phase, fraction, delta)
        for outer_step in ACTIVE_OUTER_STEPS for phase in range(7) for fraction in ALL_MIDPOINTS
    ]


@pytest.fixture
def reference_boundaries() -> np.ndarray:
    return np.full((65, STATE_SIZE), 1.0 / STATE_SIZE, dtype=np.float64)


def _assert_same_plan(left: object, right: object) -> None:
    for name in (
        "train_distances", "validation_distances", "threshold_outer_steps", "thresholds",
        "train_duplicate_indices", "validation_duplicate_indices",
        "train_augmented_indices", "validation_augmented_indices",
    ):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_builder_is_target_and_audit_blind_and_rejects_extra_fields(
    reference_boundaries: np.ndarray,
) -> None:
    assert tuple(inspect.signature(build_rollout_reweighting).parameters) == (
        "train_inputs", "validation_inputs", "stage_e_boundary_states",
    )
    assert MODEL_INPUT_FIELDS == (
        "later_full_state", "reverse_time", "phase", "color", "duration", "label",
    )
    train = _inputs(_complete_rows((0.0, 0.001)))
    validation = _inputs(_complete_rows((0.0005,)))
    first = build_rollout_reweighting(train, validation, reference_boundaries)

    target = np.arange(len(train["phase"]) * 392, dtype=np.float64).reshape(-1, 392)
    certificate = np.arange(len(train["phase"]), dtype=np.int64)
    target[:] = target[::-1]
    certificate[:] = -certificate
    second = build_rollout_reweighting(train, validation, reference_boundaries)
    _assert_same_plan(first, second)

    for name, value in (("denoising_target", target), ("sample_key", certificate)):
        with pytest.raises(RolloutReweightError):
            build_rollout_reweighting({**train, name: value}, validation, reference_boundaries)


def test_train_medians_include_ties_and_validation_cannot_choose_thresholds(
    reference_boundaries: np.ndarray,
    tmp_path,
) -> None:
    train = _inputs(_complete_rows((0.0, 0.001, 0.001, 0.002), include_ineligible=True))
    validation = _inputs(_complete_rows((0.0005, 0.0015), include_ineligible=True))
    result = build_rollout_reweighting(train, validation, reference_boundaries)

    expected = float(np.sum((_state(0.001).astype(np.float64) - 1.0 / STATE_SIZE) ** 2))
    np.testing.assert_allclose(result.thresholds, expected, rtol=0.0, atol=0.0)
    (tmp_path / "reweighting").mkdir()
    workflow._persist_reweighting(tmp_path, result)
    workflow._verify_reweighting(tmp_path, result.record)

    assert result.train_duplicate_indices.size == 14 * 7 * 3
    assert result.validation_duplicate_indices.size == 14 * 7
    assert all(np.all(np.diff(value) > 0) for value in (result.train_duplicate_indices, result.validation_duplicate_indices))

    assert np.isnan(result.train_distances[-2:]).all()
    assert np.isnan(result.validation_distances[-2:]).all()
    ineligible = [len(train["phase"]) - 2, len(train["phase"]) - 1]
    assert not np.isin(ineligible, result.train_duplicate_indices).any()

    changed_validation = {name: value.copy() for name, value in validation.items()}
    changed_validation["later_full_state"] = np.stack(
        [_state(0.01) for _ in range(len(validation["phase"]))]
    )
    changed = build_rollout_reweighting(train, changed_validation, reference_boundaries)
    np.testing.assert_array_equal(changed.thresholds, result.thresholds)
    np.testing.assert_array_equal(changed.train_duplicate_indices, result.train_duplicate_indices)
    assert changed.validation_duplicate_indices.size == 0


def test_distance_uses_exact_pre_step_boundary_511_minus_k(
    reference_boundaries: np.ndarray,
) -> None:
    rows = _complete_rows((0.0,))
    for index, (outer_step, phase, fraction, _) in enumerate(rows):
        if outer_step == 303:
            rows[index] = (outer_step, phase, fraction, 0.003)
    boundaries = reference_boundaries.copy()
    boundaries[26] = _state(0.003).astype(np.float64)
    boundaries[25] = _state(0.006).astype(np.float64)
    boundaries[27] = _state(0.009).astype(np.float64)
    values = _inputs(rows)
    result = build_rollout_reweighting(values, values, boundaries)
    steps = np.asarray([row[0] for row in rows])
    assert np.all(result.train_distances[steps == 303] == 0.0)


def test_augmented_index_contract_retains_every_original_then_exact_copies(
    reference_boundaries: np.ndarray,
) -> None:
    train = _inputs(_all_midpoint_rows())
    validation = _inputs(_all_midpoint_rows(0.0005))
    result = build_rollout_reweighting(train, validation, reference_boundaries)
    n = len(train["phase"])

    assert n == 14 * 7 * 8
    assert result.train_duplicate_indices.size == 14 * 7
    np.testing.assert_array_equal(result.train_augmented_indices[:n], np.arange(n))
    np.testing.assert_array_equal(result.train_augmented_indices[n:], result.train_duplicate_indices)
    augmented = augment_mapping(train, result.train_augmented_indices)
    target = np.arange(n * 392, dtype=np.float64).reshape(n, 392)
    augmented_target = target[result.train_augmented_indices]
    for name, original in train.items():
        np.testing.assert_array_equal(augmented[name][:n], original)
        np.testing.assert_array_equal(augmented[name][n:], original[result.train_duplicate_indices])
    np.testing.assert_array_equal(augmented_target[:n], target)
    np.testing.assert_array_equal(augmented_target[n:], target[result.train_duplicate_indices])
    recovered_q = np.asarray([ALL_MIDPOINTS[index % 8] for index in result.train_duplicate_indices])
    assert np.all(recovered_q == Q_NEAREST_ENDPOINT)

    repeated = np.concatenate((np.arange(n, dtype=np.int64), [0, 0]))
    with pytest.raises(RolloutReweightError, match="ascending"):
        augment_mapping(train, repeated)


def test_repeated_identical_inputs_preserve_empirical_conditional_target_mean(
    reference_boundaries: np.ndarray,
) -> None:
    rows = _complete_rows((0.001, 0.001))
    train = _inputs(rows)
    result = build_rollout_reweighting(train, train, reference_boundaries)
    target = np.asarray([
        1000.0 * row[0] + 10.0 * row[1] + (-1.0 if index % 2 == 0 else 1.0)
        for index, row in enumerate(rows)
    ], dtype=np.float64)
    augmented_target = target[result.train_augmented_indices]
    for cell in range(14 * 7):
        original = slice(2 * cell, 2 * cell + 2)
        augmented_positions = np.flatnonzero(np.isin(result.train_augmented_indices, [2 * cell, 2 * cell + 1]))
        assert np.mean(target[original]) == np.mean(augmented_target[augmented_positions])


def test_cache_roles_are_exact_disjoint_and_confirmation_stays_sealed() -> None:
    assert validate_cache_role_path_ids("train", TRAIN_PATH_IDS) == tuple(TRAIN_PATH_IDS)
    assert validate_cache_role_path_ids("validation", VALIDATION_PATH_IDS) == tuple(VALIDATION_PATH_IDS)
    assert set(TRAIN_PATH_IDS).isdisjoint(VALIDATION_PATH_IDS)
    assert set(TRAIN_PATH_IDS).isdisjoint(PROTECTED_CONFIRMATION_PATH_IDS)
    assert set(VALIDATION_PATH_IDS).isdisjoint(PROTECTED_CONFIRMATION_PATH_IDS)
    with pytest.raises(RolloutReweightError):
        validate_cache_role_path_ids("train", tuple(reversed(TRAIN_PATH_IDS)))
    with pytest.raises(RolloutReweightError):
        validate_cache_role_path_ids("validation", PROTECTED_CONFIRMATION_PATH_IDS[:32])
