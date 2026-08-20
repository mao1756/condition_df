"""Target-blind rollout-proximity reweighting for the Stage E pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_rb_learnability import (
    MODEL_INPUT_FIELDS,
    PHASE_COUNT,
    STATE_SIZE,
    semantic_sha256,
)
from mnist.d0_jacobi_rb_reverse_controller import fractional_coordinate


SCHEMA = "d0-jacobi-rb-rollout-reweight-v1"
TRAIN_PATH_IDS = tuple(range(0xF8100, 0xF8140))
VALIDATION_PATH_IDS = tuple(range(0xF8200, 0xF8220))
PROTECTED_CONFIRMATION_PATH_IDS = tuple(range(0xF9000, 0xF9040))
ACTIVE_OUTER_STEPS = tuple(range(303, 512, 16))
ELIGIBLE_MIDPOINT_FRACTION = 15.0 / 16.0
ACTIVE_OUTER_STEP_MINIMUM = 296
STAGE_E_COMPLETED_STEPS = tuple(range(0, 513, 8))


class RolloutReweightError(ValueError):
    """The frozen reweighting contract was violated."""


@dataclass(frozen=True)
class RolloutReweighting:
    train_distances: np.ndarray
    validation_distances: np.ndarray
    threshold_outer_steps: np.ndarray
    thresholds: np.ndarray
    train_duplicate_indices: np.ndarray
    validation_duplicate_indices: np.ndarray
    train_augmented_indices: np.ndarray
    validation_augmented_indices: np.ndarray
    hashes: Mapping[str, str]
    record: Mapping[str, Any]


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return semantic_sha256(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )


def _model_input_sha256(values: Mapping[str, np.ndarray]) -> str:
    return semantic_sha256(
        {name: _array_sha256(values[name]) for name in MODEL_INPUT_FIELDS}
    )


def _as_model_input_arrays(values: Mapping[str, np.ndarray], *, role: str) -> dict[str, np.ndarray]:
    expected = set(MODEL_INPUT_FIELDS)
    if set(values) != expected:
        raise RolloutReweightError(f"{role} model-input fields must be exactly {sorted(expected)}; got {sorted(values)}")
    arrays = {name: np.asarray(values[name]) for name in MODEL_INPUT_FIELDS}
    state = arrays["later_full_state"]
    if state.ndim != 2 or state.shape[1] != STATE_SIZE or state.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise RolloutReweightError(f"{role} later_full_state has the wrong shape or dtype")
    count = int(state.shape[0])
    for name in MODEL_INPUT_FIELDS[1:]:
        if arrays[name].shape != (count,):
            raise RolloutReweightError(f"{role} {name} has the wrong shape")
    for names, kinds in ((('reverse_time', 'duration'), 'f'), (('phase', 'color', 'label'), 'iu')):
        if any(arrays[name].dtype.kind not in kinds for name in names):
            raise RolloutReweightError(f"{role} model-input dtypes changed")
    if not np.isfinite(state).all():
        raise RolloutReweightError(f"{role} later_full_state is nonfinite")
    return arrays


def validate_cache_role_path_ids(role: str, path_ids: Sequence[int]) -> tuple[int, ...]:
    """Require the sealed train/validation allocation without opening confirmation."""

    try:
        values = tuple(operator.index(value) for value in path_ids)
    except TypeError as exc:
        raise RolloutReweightError("cache path IDs must be integers") from exc
    protected = sorted(set(values) & set(PROTECTED_CONFIRMATION_PATH_IDS))
    if protected:
        raise RolloutReweightError(
            f"protected confirmation path IDs are forbidden: {protected}"
        )
    expected = {"train": TRAIN_PATH_IDS, "validation": VALIDATION_PATH_IDS}.get(role)
    if expected is None:
        raise RolloutReweightError("cache role must be train or validation")
    if values != expected:
        raise RolloutReweightError(f"{role} cache path IDs changed")
    return values


def _coordinates(values: Mapping[str, np.ndarray], *, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        coordinate = fractional_coordinate(
            torch.as_tensor(values["reverse_time"], device="cpu"),
            torch.as_tensor(values["phase"], device="cpu"),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RolloutReweightError(f"{role} fractional coordinates are invalid") from exc
    outer_step = coordinate.outer_step.cpu().numpy().astype(np.int64, copy=False)
    fraction = coordinate.within_phase_fraction.cpu().numpy().astype(np.float64, copy=False)
    phase = np.asarray(values["phase"], dtype=np.int64)
    eligible = (fraction == ELIGIBLE_MIDPOINT_FRACTION) & (outer_step >= ACTIVE_OUTER_STEP_MINIMUM)
    return outer_step, phase, eligible


def _distances(
    states: np.ndarray,
    outer_step: np.ndarray,
    eligible: np.ndarray,
    stage_e_boundary_states: np.ndarray,
) -> np.ndarray:
    result = np.full(len(states), np.nan, dtype=np.float64)
    for step in sorted(set(int(value) for value in outer_step[eligible])):
        boundary_step = 511 - step
        if boundary_step < 0 or boundary_step > 512 or boundary_step % 16:
            raise RolloutReweightError(f"outer step {step} has no frozen 16-step Stage E boundary")
        boundary_index = boundary_step // 8
        if STAGE_E_COMPLETED_STEPS[boundary_index] != boundary_step:
            raise RolloutReweightError("Stage E boundary indexing changed")
        indices = np.flatnonzero(eligible & (outer_step == step))
        later = np.asarray(states[indices], dtype=np.float64)
        delta = later - stage_e_boundary_states[boundary_index]
        result[indices] = np.sum(np.square(delta, dtype=np.float64), axis=1, dtype=np.float64)
    return np.ascontiguousarray(result)


def _duplicate_indices(
    distances: np.ndarray,
    outer_step: np.ndarray,
    phase: np.ndarray,
    eligible: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    selected = np.zeros(len(distances), dtype=bool)
    step_to_row = {step: index for index, step in enumerate(ACTIVE_OUTER_STEPS)}
    for index in np.flatnonzero(eligible):
        step = int(outer_step[index])
        threshold_row = step_to_row.get(step)
        if threshold_row is None:
            raise RolloutReweightError(f"eligible outer step {step} is not frozen")
        threshold = thresholds[threshold_row, int(phase[index])]
        if not np.isfinite(threshold):
            raise RolloutReweightError(f"training threshold is missing for cell ({step}, {int(phase[index])})")
        selected[index] = bool(distances[index] <= threshold)
    return np.flatnonzero(selected).astype(np.int64, copy=False)


def _augmented_indices(count: int, duplicates: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate((np.arange(count, dtype=np.int64), duplicates)))


def _role_record(distances: np.ndarray, eligible: np.ndarray, duplicates: np.ndarray, augmented: np.ndarray, **extra: Any) -> dict[str, Any]:
    return {
        "original_row_count": len(distances),
        "eligible_row_count": int(np.count_nonzero(eligible)),
        "duplicate_row_count": len(duplicates),
        "augmented_row_count": len(augmented),
        **extra,
    }


def build_rollout_reweighting(
    train_inputs: Mapping[str, np.ndarray],
    validation_inputs: Mapping[str, np.ndarray],
    stage_e_boundary_states: np.ndarray,
) -> RolloutReweighting:
    """Build the frozen target-blind integer reweighting for train and validation."""

    train = _as_model_input_arrays(train_inputs, role="train")
    validation = _as_model_input_arrays(validation_inputs, role="validation")
    boundaries = np.asarray(stage_e_boundary_states)
    if boundaries.shape != (len(STAGE_E_COMPLETED_STEPS), STATE_SIZE):
        raise RolloutReweightError("Stage E boundary states have the wrong shape")
    if boundaries.dtype != np.dtype(np.float64):
        raise RolloutReweightError("Stage E boundary states must be binary64")
    if not np.isfinite(boundaries).all():
        raise RolloutReweightError("Stage E boundary states are nonfinite")

    train_step, train_phase, train_eligible = _coordinates(train, role="train")
    validation_step, validation_phase, validation_eligible = _coordinates(validation, role="validation")
    train_distances = _distances(train["later_full_state"], train_step, train_eligible, boundaries)
    validation_distances = _distances(validation["later_full_state"], validation_step, validation_eligible, boundaries)

    thresholds = np.full((len(ACTIVE_OUTER_STEPS), PHASE_COUNT), np.nan, np.float64)
    for step_index, step in enumerate(ACTIVE_OUTER_STEPS):
        for phase in range(PHASE_COUNT):
            cell = train_eligible & (train_step == step) & (train_phase == phase)
            if np.any(cell):
                thresholds[step_index, phase] = np.median(train_distances[cell], overwrite_input=False)
    train_duplicates = _duplicate_indices(train_distances, train_step, train_phase, train_eligible, thresholds)
    validation_duplicates = _duplicate_indices(validation_distances, validation_step, validation_phase, validation_eligible, thresholds)
    train_augmented = _augmented_indices(len(train_distances), train_duplicates)
    validation_augmented = _augmented_indices(len(validation_distances), validation_duplicates)
    threshold_outer_steps = np.asarray(ACTIVE_OUTER_STEPS, dtype=np.int64)

    arrays = {
        "stage_e_boundary_states": boundaries,
        "train_distances": train_distances,
        "validation_distances": validation_distances,
        "threshold_outer_steps": threshold_outer_steps,
        "thresholds": thresholds,
        "train_duplicate_indices": train_duplicates,
        "validation_duplicate_indices": validation_duplicates,
        "train_augmented_indices": train_augmented,
        "validation_augmented_indices": validation_augmented,
    }
    hashes = {
        "train_model_inputs_sha256": _model_input_sha256(train),
        "validation_model_inputs_sha256": _model_input_sha256(validation),
        **{f"{name}_sha256": _array_sha256(value) for name, value in arrays.items()},
    }
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "eligibility": {
            "midpoint_fraction_numerator": 15,
            "midpoint_fraction_denominator": 16,
            "active_outer_step_minimum": ACTIVE_OUTER_STEP_MINIMUM,
            "active_outer_steps": list(ACTIVE_OUTER_STEPS),
            "boundary_rule": "completed_reverse_steps=511-outer_step",
            "stage_e_completed_steps": list(STAGE_E_COMPLETED_STEPS),
        },
        "train": _role_record(train_distances, train_eligible, train_duplicates, train_augmented),
        "validation": _role_record(validation_distances, validation_eligible, validation_duplicates, validation_augmented, threshold_source="train"),
        "finite_threshold_cell_count": int(np.count_nonzero(np.isfinite(thresholds))),
        "hashes": hashes,
    }
    record = {**body, "semantic_sha256": semantic_sha256(body)}
    return RolloutReweighting(
        train_distances=train_distances,
        validation_distances=validation_distances,
        threshold_outer_steps=threshold_outer_steps,
        thresholds=np.ascontiguousarray(thresholds),
        train_duplicate_indices=np.ascontiguousarray(train_duplicates),
        validation_duplicate_indices=np.ascontiguousarray(validation_duplicates),
        train_augmented_indices=train_augmented,
        validation_augmented_indices=validation_augmented,
        hashes=hashes,
        record=record,
    )


def augment_mapping(values: Mapping[str, np.ndarray], augmented_indices: Sequence[int] | np.ndarray) -> dict[str, np.ndarray]:
    """Materialize exact model-input copies in the supplied augmented row order."""

    arrays = _as_model_input_arrays(values, role="augmentation")
    indices = np.asarray(augmented_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise RolloutReweightError("augmented indices must be a one-dimensional integer array")
    indices = indices.astype(np.int64, copy=False)
    count = len(arrays["later_full_state"])
    if len(indices) < count or not np.array_equal(indices[:count], np.arange(count, dtype=np.int64)):
        raise RolloutReweightError("augmented indices must retain every original once first")
    duplicates = indices[count:]
    if np.any((duplicates < 0) | (duplicates >= count)) or np.any(duplicates[1:] <= duplicates[:-1]):
        raise RolloutReweightError("duplicate indices must be in-range and ascending")
    return {name: np.ascontiguousarray(array[indices]) for name, array in arrays.items()}


__all__ = [
    "ACTIVE_OUTER_STEP_MINIMUM",
    "ACTIVE_OUTER_STEPS",
    "ELIGIBLE_MIDPOINT_FRACTION",
    "PROTECTED_CONFIRMATION_PATH_IDS",
    "RolloutReweightError",
    "RolloutReweighting",
    "SCHEMA",
    "STAGE_E_COMPLETED_STEPS",
    "TRAIN_PATH_IDS",
    "VALIDATION_PATH_IDS",
    "augment_mapping",
    "build_rollout_reweighting",
    "validate_cache_role_path_ids",
]
