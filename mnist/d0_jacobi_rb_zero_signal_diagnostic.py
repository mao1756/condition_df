"""Post-hoc diagnostics for the sealed Jacobi/RB learnability result.

The helpers in this module do not fit a model, select a checkpoint, generate
states, or change the completed one-image gate.  They only decompose the
quadratic risk of one already-frozen predictor:

    MSE(p, z) - MSE(0, z) = E[p^2] - 2 E[p z].

This identity distinguishes prediction energy from target alignment and makes
the negative confirmation result interpretable without reopening it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from torch import Tensor

from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_COUNT,
    stable_sum,
)


DIAGNOSTIC_VERSION = "d0-jacobi-rb-zero-signal-diagnostic-v1"
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 261_221


class ZeroSignalDiagnosticError(ValueError):
    """Raised when a sealed diagnostic input or statistic is invalid."""


def _array(value: np.ndarray | Tensor, *, name: str) -> np.ndarray:
    if isinstance(value, Tensor):
        result = value.detach().to(device="cpu", dtype=value.dtype).numpy()
    else:
        result = np.asarray(value)
    result = np.asarray(result, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ZeroSignalDiagnosticError(f"{name} must be finite and nonempty")
    return result


def _mean_product(left: np.ndarray, right: np.ndarray) -> float:
    return stable_sum(left * right) / left.size


@dataclass(frozen=True)
class QuadraticRiskDecomposition:
    """Exact binary64 summary of one predictor/target comparison."""

    row_count: int
    element_count: int
    target_energy: float
    prediction_energy: float
    metadata_energy: float
    target_prediction_inner_product: float
    target_metadata_inner_product: float
    prediction_metadata_inner_product: float
    zero_mse: float
    model_mse: float
    metadata_mse: float
    zero_minus_model_mse: float
    metadata_minus_model_mse: float
    metadata_minus_zero_mse: float
    prediction_minus_metadata_mse: float
    relative_zero_gain: float
    relative_metadata_improvement: float
    metadata_harm_removed_fraction: float
    alignment_to_prediction_cost_ratio: float
    target_rms: float
    prediction_rms: float
    metadata_rms: float
    target_prediction_cosine: float
    target_metadata_cosine: float
    prediction_metadata_cosine: float
    model_identity_abs_error: float
    metadata_identity_abs_error: float

    @property
    def model_beats_zero(self) -> bool:
        return self.zero_minus_model_mse > 0.0

    @property
    def covariance_exceeds_prediction_cost(self) -> bool:
        return (
            2.0 * self.target_prediction_inner_product
            > self.prediction_energy
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "model_beats_zero": int(self.model_beats_zero),
                "covariance_exceeds_prediction_cost": int(
                    self.covariance_exceeds_prediction_cost
                ),
                "risk_identity": (
                    "zero_mse-model_mse="
                    "2*target_prediction_inner_product-prediction_energy"
                ),
            }
        )
        return record


def quadratic_risk_decomposition(
    prediction: np.ndarray | Tensor,
    target: np.ndarray | Tensor,
    metadata_prediction: np.ndarray | Tensor,
) -> QuadraticRiskDecomposition:
    """Compute the zero/model/metadata quadratic-risk decomposition."""

    prediction_array = _array(prediction, name="prediction")
    target_array = _array(target, name="target")
    metadata_array = _array(metadata_prediction, name="metadata_prediction")
    if (
        prediction_array.shape != target_array.shape
        or metadata_array.shape != target_array.shape
        or target_array.ndim != 2
        or target_array.shape[1] != EDGES_PER_PHASE
    ):
        raise ZeroSignalDiagnosticError(
            "prediction, target, and metadata must have equal [N,392] shapes"
        )

    target_energy = _mean_product(target_array, target_array)
    prediction_energy = _mean_product(prediction_array, prediction_array)
    metadata_energy = _mean_product(metadata_array, metadata_array)
    target_prediction = _mean_product(target_array, prediction_array)
    target_metadata = _mean_product(target_array, metadata_array)
    prediction_metadata = _mean_product(prediction_array, metadata_array)

    model_difference = prediction_array - target_array
    metadata_difference = metadata_array - target_array
    prediction_metadata_difference = prediction_array - metadata_array
    model_mse = _mean_product(model_difference, model_difference)
    metadata_mse = _mean_product(metadata_difference, metadata_difference)
    prediction_minus_metadata_mse = _mean_product(
        prediction_metadata_difference, prediction_metadata_difference
    )

    expanded_model = target_energy + prediction_energy - 2.0 * target_prediction
    expanded_metadata = target_energy + metadata_energy - 2.0 * target_metadata

    def cosine(inner: float, left_energy: float, right_energy: float) -> float:
        denominator = math.sqrt(left_energy * right_energy)
        return inner / denominator if denominator > 0.0 else 0.0

    return QuadraticRiskDecomposition(
        row_count=int(target_array.shape[0]),
        element_count=int(target_array.size),
        target_energy=float(target_energy),
        prediction_energy=float(prediction_energy),
        metadata_energy=float(metadata_energy),
        target_prediction_inner_product=float(target_prediction),
        target_metadata_inner_product=float(target_metadata),
        prediction_metadata_inner_product=float(prediction_metadata),
        zero_mse=float(target_energy),
        model_mse=float(model_mse),
        metadata_mse=float(metadata_mse),
        zero_minus_model_mse=float(target_energy - model_mse),
        metadata_minus_model_mse=float(metadata_mse - model_mse),
        metadata_minus_zero_mse=float(metadata_mse - target_energy),
        prediction_minus_metadata_mse=float(prediction_minus_metadata_mse),
        relative_zero_gain=float(
            (target_energy - model_mse) / target_energy
            if target_energy > 0.0
            else 0.0
        ),
        relative_metadata_improvement=float(
            (metadata_mse - model_mse) / metadata_mse
            if metadata_mse > 0.0
            else 0.0
        ),
        metadata_harm_removed_fraction=float(
            (metadata_mse - model_mse) / (metadata_mse - target_energy)
            if metadata_mse != target_energy
            else 0.0
        ),
        alignment_to_prediction_cost_ratio=float(
            2.0 * target_prediction / prediction_energy
            if prediction_energy > 0.0
            else 0.0
        ),
        target_rms=float(math.sqrt(target_energy)),
        prediction_rms=float(math.sqrt(prediction_energy)),
        metadata_rms=float(math.sqrt(metadata_energy)),
        target_prediction_cosine=float(
            cosine(target_prediction, target_energy, prediction_energy)
        ),
        target_metadata_cosine=float(
            cosine(target_metadata, target_energy, metadata_energy)
        ),
        prediction_metadata_cosine=float(
            cosine(prediction_metadata, prediction_energy, metadata_energy)
        ),
        model_identity_abs_error=float(abs(model_mse - expanded_model)),
        metadata_identity_abs_error=float(abs(metadata_mse - expanded_metadata)),
    )


def path_decomposition_rows(
    prediction: np.ndarray | Tensor,
    target: np.ndarray | Tensor,
    metadata_prediction: np.ndarray | Tensor,
    path_id: np.ndarray | Tensor,
    *,
    split: str,
) -> list[dict[str, Any]]:
    prediction_array = _array(prediction, name="prediction")
    target_array = _array(target, name="target")
    metadata_array = _array(metadata_prediction, name="metadata_prediction")
    paths = np.asarray(
        path_id.detach().cpu().numpy() if isinstance(path_id, Tensor) else path_id,
        dtype=np.int64,
    )
    if paths.shape != (target_array.shape[0],):
        raise ZeroSignalDiagnosticError("path_id must have one value per cache row")
    rows: list[dict[str, Any]] = []
    for value in sorted(int(item) for item in np.unique(paths)):
        mask = paths == value
        decomposition = quadratic_risk_decomposition(
            prediction_array[mask], target_array[mask], metadata_array[mask]
        )
        rows.append(
            {
                "split": str(split),
                "path_id": value,
                **decomposition.to_record(),
            }
        )
    return rows


def stratified_decomposition_rows(
    prediction: np.ndarray | Tensor,
    target: np.ndarray | Tensor,
    metadata_prediction: np.ndarray | Tensor,
    outer_step: np.ndarray | Tensor,
    phase: np.ndarray | Tensor,
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Return overall, phase, quartile, and phase-by-quartile summaries."""

    prediction_array = _array(prediction, name="prediction")
    target_array = _array(target, name="target")
    metadata_array = _array(metadata_prediction, name="metadata_prediction")
    steps = np.asarray(
        outer_step.detach().cpu().numpy()
        if isinstance(outer_step, Tensor)
        else outer_step,
        dtype=np.int64,
    )
    phases = np.asarray(
        phase.detach().cpu().numpy() if isinstance(phase, Tensor) else phase,
        dtype=np.int64,
    )
    if steps.shape != (target_array.shape[0],) or phases.shape != steps.shape:
        raise ZeroSignalDiagnosticError(
            "outer_step and phase must have one value per cache row"
        )
    if ((steps < 0) | (steps >= 512)).any():
        raise ZeroSignalDiagnosticError("outer_step lies outside the frozen chain")
    if ((phases < 0) | (phases >= PHASE_COUNT)).any():
        raise ZeroSignalDiagnosticError("phase lies outside the frozen chain")

    rows: list[dict[str, Any]] = []

    def append(stratum: str, value: str, mask: np.ndarray) -> None:
        if not bool(mask.any()):
            raise ZeroSignalDiagnosticError(f"empty diagnostic stratum {stratum}:{value}")
        rows.append(
            {
                "split": str(split),
                "stratum": stratum,
                "stratum_value": value,
                **quadratic_risk_decomposition(
                    prediction_array[mask],
                    target_array[mask],
                    metadata_array[mask],
                ).to_record(),
            }
        )

    append("overall", "all", np.ones(steps.shape, dtype=bool))
    for phase_index in range(PHASE_COUNT):
        append("phase", str(phase_index), phases == phase_index)
    quartiles = steps // 128
    for quartile in range(4):
        append("time_quartile", str(quartile), quartiles == quartile)
    for quartile in range(4):
        for phase_index in range(PHASE_COUNT):
            append(
                "phase_time",
                f"q{quartile}_p{phase_index}",
                (quartiles == quartile) & (phases == phase_index),
            )
    return rows


def whole_path_bootstrap_interval(
    path_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = 0.99,
) -> dict[str, Any]:
    """Descriptive percentile interval over whole paths.

    This interval is deliberately non-authorizing: the confirmation panel has
    already been opened and inspected.
    """

    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ZeroSignalDiagnosticError("bootstrap configuration is invalid")
    values = np.asarray([float(row[field]) for row in path_rows], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ZeroSignalDiagnosticError("bootstrap path statistic is invalid")
    generator = np.random.Generator(np.random.Philox(int(seed)))
    indices = generator.integers(
        0, values.size, size=(int(replicates), values.size), dtype=np.int64
    )
    estimates = np.mean(values[indices], axis=1, dtype=np.float64)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        estimates,
        [0.5 * alpha, 1.0 - 0.5 * alpha],
        method="linear",
    )
    return {
        "field": str(field),
        "path_count": int(values.size),
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "path_mean": float(np.mean(values, dtype=np.float64)),
        "lower": float(lower),
        "upper": float(upper),
        "bootstrap_unit": "whole_path",
        "non_authorizing_posthoc": 1,
    }


def coarse_cell_path_means(
    target: np.ndarray | Tensor,
    path_id: np.ndarray | Tensor,
    outer_step: np.ndarray | Tensor,
    phase: np.ndarray | Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-path means on the frozen quartile/phase/edge partition.

    The result has shape ``[paths,4,7,392]``.  Production caches contain
    exactly eight selected outer steps per quartile, so every path/cell mean
    uses the same number of observations.
    """

    target_array = _array(target, name="target")
    paths = np.asarray(
        path_id.detach().cpu().numpy() if isinstance(path_id, Tensor) else path_id,
        dtype=np.int64,
    )
    steps = np.asarray(
        outer_step.detach().cpu().numpy()
        if isinstance(outer_step, Tensor)
        else outer_step,
        dtype=np.int64,
    )
    phases = np.asarray(
        phase.detach().cpu().numpy() if isinstance(phase, Tensor) else phase,
        dtype=np.int64,
    )
    if (
        target_array.ndim != 2
        or target_array.shape[1] != EDGES_PER_PHASE
        or paths.shape != (target_array.shape[0],)
        or steps.shape != paths.shape
        or phases.shape != paths.shape
    ):
        raise ZeroSignalDiagnosticError("coarse cell arrays have invalid shapes")
    unique_paths = np.unique(paths)
    result = np.empty(
        (unique_paths.size, 4, PHASE_COUNT, EDGES_PER_PHASE),
        dtype=np.float64,
    )
    for path_index, path_value in enumerate(unique_paths):
        for quartile in range(4):
            for phase_index in range(PHASE_COUNT):
                mask = (
                    (paths == path_value)
                    & (steps // 128 == quartile)
                    & (phases == phase_index)
                )
                if int(mask.sum()) != 8:
                    raise ZeroSignalDiagnosticError(
                        "every path/coarse cell must contain exactly eight rows"
                    )
                result[path_index, quartile, phase_index] = np.mean(
                    target_array[mask], axis=0, dtype=np.float64
                )
    return np.ascontiguousarray(unique_paths), np.ascontiguousarray(result)


def cross_split_coarse_signal(
    left_path_cell_means: np.ndarray,
    right_path_cell_means: np.ndarray,
    *,
    left_split: str,
    right_split: str,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = 0.99,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Cross-split coarse conditional-mean-energy estimate.

    Paths are independently resampled within each split.  The statistic is
    post-hoc and non-authorizing because the sealed confirmation result was
    already inspected before this workflow was defined.
    """

    left = np.asarray(left_path_cell_means, dtype=np.float64)
    right = np.asarray(right_path_cell_means, dtype=np.float64)
    expected_tail = (4, PHASE_COUNT, EDGES_PER_PHASE)
    if (
        left.ndim != 4
        or right.ndim != 4
        or left.shape[1:] != expected_tail
        or right.shape[1:] != expected_tail
        or left.shape[0] <= 0
        or right.shape[0] <= 0
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ZeroSignalDiagnosticError("coarse path-cell means are invalid")
    if replicates <= 0 or chunk_size <= 0 or not 0.0 < confidence < 1.0:
        raise ZeroSignalDiagnosticError("coarse bootstrap configuration is invalid")

    left_mean = np.mean(left, axis=0, dtype=np.float64)
    right_mean = np.mean(right, axis=0, dtype=np.float64)
    point = _mean_product(left_mean, right_mean)
    generator = np.random.Generator(np.random.Philox(int(seed)))
    estimates = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), int(chunk_size)):
        stop = min(int(replicates), start + int(chunk_size))
        count = stop - start
        left_indices = generator.integers(
            0, left.shape[0], size=(count, left.shape[0]), dtype=np.int64
        )
        right_indices = generator.integers(
            0, right.shape[0], size=(count, right.shape[0]), dtype=np.int64
        )
        left_boot = np.mean(left[left_indices], axis=1, dtype=np.float64)
        right_boot = np.mean(right[right_indices], axis=1, dtype=np.float64)
        estimates[start:stop] = np.mean(
            left_boot * right_boot,
            axis=(1, 2, 3),
            dtype=np.float64,
        )
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        estimates,
        [0.5 * alpha, 1.0 - 0.5 * alpha],
        method="linear",
    )
    return {
        "left_split": str(left_split),
        "right_split": str(right_split),
        "left_path_count": int(left.shape[0]),
        "right_path_count": int(right.shape[0]),
        "coarse_cell_count": int(np.prod(expected_tail)),
        "observations_per_path_cell": 8,
        "observations_per_split_cell": int(8 * left.shape[0]),
        "cross_split_coarse_signal": float(point),
        "confidence": float(confidence),
        "lower": float(lower),
        "upper": float(upper),
        "interval_contains_zero": int(lower <= 0.0 <= upper),
        "replicates": int(replicates),
        "seed": int(seed),
        "bootstrap_unit": "whole_path_independently_within_split",
        "estimand": "E[(E[target|time_quartile,phase,edge])^2]",
        "lower_bound_on_full_conditional_mean_energy": 1,
        "non_authorizing_posthoc": 1,
    }


def diagnostic_conclusion(
    split_summaries: Mapping[str, Mapping[str, Any]],
    coarse_signal_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify the frozen predictor without making a population-zero claim."""

    for split in ("validation", "confirmation"):
        if split not in split_summaries:
            raise ZeroSignalDiagnosticError(f"missing {split} diagnostic summary")
    validation_margin = float(
        split_summaries["validation"]["zero_minus_model_mse"]
    )
    confirmation_margin = float(
        split_summaries["confirmation"]["zero_minus_model_mse"]
    )
    if confirmation_margin <= 0.0:
        conclusion = "frozen_model_does_not_beat_zero"
    elif validation_margin <= 0.0:
        conclusion = "held_out_splits_disagree"
    else:
        conclusion = "posthoc_positive_fixed_model_signal"
    coarse_rows = tuple(coarse_signal_rows or ())
    coarse_inconclusive = bool(coarse_rows) and all(
        int(row.get("interval_contains_zero", 0)) == 1 for row in coarse_rows
    )
    coarse_positive = bool(coarse_rows) and all(
        float(row.get("lower", -math.inf)) > 0.0 for row in coarse_rows
    )
    return {
        "conclusion": conclusion,
        "coarse_conditional_signal_conclusion": (
            "coarse_conditional_signal_inconclusive"
            if coarse_inconclusive
            else (
                "coarse_conditional_signal_posthoc_positive"
                if coarse_positive
                else (
                    "coarse_conditional_signal_mixed"
                    if coarse_rows
                    else "not_evaluated"
                )
            )
        ),
        "validation_zero_minus_model_mse": validation_margin,
        "confirmation_zero_minus_model_mse": confirmation_margin,
        "conditional_mean_identically_zero_proven": 0,
        "population_signal_absence_proven": 0,
        "new_scientific_gate_authorized": 0,
        "interpretation_scope": (
            "the already-frozen selected model on the already-opened validation "
            "and confirmation caches"
        ),
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "DIAGNOSTIC_VERSION",
    "QuadraticRiskDecomposition",
    "ZeroSignalDiagnosticError",
    "coarse_cell_path_means",
    "cross_split_coarse_signal",
    "diagnostic_conclusion",
    "path_decomposition_rows",
    "quadratic_risk_decomposition",
    "stratified_decomposition_rows",
    "whole_path_bootstrap_interval",
]
