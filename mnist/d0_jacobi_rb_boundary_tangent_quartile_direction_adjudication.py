"""Pure directional diagnostics for the frozen quartile-specialist family.

The functions in this module reduce predictions that have already been
evaluated on the already-open ``gain_calibration`` and ``training_rank``
roles.  They do not load checkpoints, open roles, generate labels, select a
model, or authorize any physical work.  In particular, every scalar gain
reported here is an algebraic diagnostic of the unchanged raw prediction.

All scalar reductions iterate in canonical C order and use :func:`math.fsum`.
This is intentionally a little slower than vectorized ``numpy.mean``: the
sealed rank replay requires deterministic binary64 reductions rather than a
platform-dependent summation tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from mnist.d0_jacobi_rb_boundary_tangent_quartile_specialist import (
    FINE_CELL_SHAPE,
    MINIMUM_POSITIVE_FINE_CELLS,
    MODEL_SEEDS_BY_QUARTILE,
    NONZERO_CANDIDATE_IDENTITIES,
    Q1_SENTINEL,
)


SCHEMA = "d0-jacobi-rb-boundary-tangent-quartile-direction-adjudication-v1"
ROLE_ORDER = ("gain_calibration", "training_rank")
CANDIDATE_ORDER = tuple(candidate.key for candidate in NONZERO_CANDIDATE_IDENTITIES)
CANDIDATE_COUNT = 480
ROLE_COUNT = 2
PATH_COUNT = 32
PHASE_COUNT, MIDPOINT_COUNT = FINE_CELL_SHAPE
FINE_CELL_COUNT = PHASE_COUNT * MIDPOINT_COUNT
IDENTITY_TOLERANCE = 5.0e-15
CRITICAL_VALUE = 7.1588810358178305
MAXIMUM_PREDICTION_BATCH_SIZE = 32
POWER_ONLY_MAXIMUM_ROUNDED_PATH_COUNT = 384


class DirectionAdjudicationError(ValueError):
    """A frozen direction-adjudication arithmetic contract was violated."""

    def __init__(self, message: str, *, failure_code: str = "classification_invalid"):
        super().__init__(message)
        self.failure_code = str(failure_code)


def _readonly(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DirectionAdjudicationError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise DirectionAdjudicationError(f"{name} must be a finite scalar")
    return result


def _quartile(value: Any) -> int:
    if isinstance(value, bool):
        raise DirectionAdjudicationError("quartile must be an integer in [0,4)")
    try:
        result = int(value.__index__())
    except (AttributeError, TypeError) as exc:
        raise DirectionAdjudicationError(
            "quartile must be an integer in [0,4)"
        ) from exc
    if not 0 <= result < 4:
        raise DirectionAdjudicationError("quartile must be an integer in [0,4)")
    return result


def _strict_integer_array(value: Any, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.integer):
        raise DirectionAdjudicationError(f"{name} must be a one-dimensional integer array")
    return np.ascontiguousarray(source, dtype=np.int64)


def _cell_arrays(values: Any, counts: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(values, dtype=np.float64)
    if cells.shape not in {(PHASE_COUNT, MIDPOINT_COUNT)} and not (
        cells.ndim == 3 and cells.shape[1:] == (PHASE_COUNT, MIDPOINT_COUNT)
    ):
        raise DirectionAdjudicationError("cell values must have shape [7,8] or [path,7,8]")
    if not np.isfinite(cells).all():
        raise DirectionAdjudicationError("cell values must be finite")
    if counts is None:
        weights = np.ones(cells.shape, dtype=np.int64)
    else:
        source = np.asarray(counts)
        if source.shape != cells.shape or not np.issubdtype(source.dtype, np.integer):
            raise DirectionAdjudicationError("cell counts must be an integral matching array")
        weights = np.asarray(source, dtype=np.int64)
        if np.any(weights < 0):
            raise DirectionAdjudicationError("cell counts must be nonnegative")
    return np.ascontiguousarray(cells), np.ascontiguousarray(weights)


def _weighted_mean(values: np.ndarray, weights: np.ndarray, *, name: str) -> float:
    flat_values = np.ascontiguousarray(values, dtype=np.float64).ravel(order="C")
    flat_weights = np.ascontiguousarray(weights, dtype=np.int64).ravel(order="C")
    total = math.fsum(float(weight) for weight in flat_weights)
    if total <= 0.0:
        raise DirectionAdjudicationError(f"{name} has no rows")
    numerator = math.fsum(
        float(value) * float(weight)
        for value, weight in zip(flat_values, flat_weights, strict=True)
    )
    return numerator / total


def quadratic_improvement(cross_term: Any, prediction_energy: Any, gain: float) -> Any:
    """Return ``2 * gain * C - gain**2 * P`` without clipping ``gain``."""

    active_gain = _finite_float(gain, "gain")
    cross = np.asarray(cross_term, dtype=np.float64)
    energy = np.asarray(prediction_energy, dtype=np.float64)
    if cross.shape != energy.shape or not np.isfinite(cross).all() or not np.isfinite(energy).all():
        raise DirectionAdjudicationError("C and P must be finite matching arrays")
    if np.any(energy < 0.0):
        raise DirectionAdjudicationError("prediction energy cannot be negative")
    result = 2.0 * active_gain * cross - active_gain * active_gain * energy
    if result.ndim == 0:
        return float(result)
    return np.ascontiguousarray(result, dtype=np.float64)


def scalar_optimum(cross_term: float, prediction_energy: float) -> dict[str, Any]:
    """Return the unprojected diagnostic optimum ``C/P`` and ceiling ``C**2/P``."""

    cross = _finite_float(cross_term, "cross_term")
    energy = _finite_float(prediction_energy, "prediction_energy")
    if energy < 0.0:
        raise DirectionAdjudicationError("prediction energy cannot be negative")
    usable = cross > 0.0 and energy > 0.0
    return {
        "cross_term": cross,
        "prediction_energy": energy,
        "lambda_star": cross / energy if usable else None,
        "optimal_improvement": cross * cross / energy if usable else None,
        "positive_direction": int(usable),
        "gain_clipped_or_projected": 0,
        "authorizing": 0,
    }


@dataclass(frozen=True)
class CandidateRoleDecomposition:
    """One candidate's deterministic path/phase/midpoint reduction on one role."""

    path_ids: np.ndarray
    cross_term: np.ndarray
    prediction_energy: np.ndarray
    raw_improvement: np.ndarray
    parent_gain_improvement: np.ndarray
    diagnostic_gain_improvement: np.ndarray
    fine_cell_row_count: np.ndarray
    parent_gain: float
    diagnostic_gain: float
    maximum_raw_identity_error: float
    maximum_parent_gain_identity_error: float

    def __post_init__(self) -> None:
        paths = np.asarray(self.path_ids)
        expected = (paths.size, PHASE_COUNT, MIDPOINT_COUNT)
        floating = (
            self.cross_term,
            self.prediction_energy,
            self.raw_improvement,
            self.parent_gain_improvement,
            self.diagnostic_gain_improvement,
        )
        if (
            paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or paths.size == 0
            or np.unique(paths).size != paths.size
            or any(np.asarray(value).shape != expected for value in floating)
            or any(np.asarray(value).dtype != np.dtype(np.float64) for value in floating)
            or any(not np.isfinite(np.asarray(value)).all() for value in floating)
        ):
            raise DirectionAdjudicationError("candidate-role decomposition is malformed")
        counts = np.asarray(self.fine_cell_row_count)
        if (
            counts.dtype != np.dtype(np.int64)
            or counts.shape != expected
            or np.any(counts <= 0)
            or np.any(np.asarray(self.prediction_energy) < 0.0)
        ):
            raise DirectionAdjudicationError("candidate-role row counts are malformed")
        parent_gain = _finite_float(self.parent_gain, "parent_gain")
        diagnostic_gain = _finite_float(self.diagnostic_gain, "diagnostic_gain")
        raw_error = _finite_float(self.maximum_raw_identity_error, "raw identity error")
        parent_error = _finite_float(
            self.maximum_parent_gain_identity_error, "parent-gain identity error"
        )
        if (
            raw_error < 0.0
            or parent_error < 0.0
            or raw_error > IDENTITY_TOLERANCE
            or parent_error > IDENTITY_TOLERANCE
        ):
            raise DirectionAdjudicationError(
                "direct and reconstructed quadratic improvements disagree",
                failure_code="quadratic_identity_invalid",
            )
        object.__setattr__(self, "path_ids", _readonly(paths, dtype=np.dtype(np.int64)))
        for name in (
            "cross_term",
            "prediction_energy",
            "raw_improvement",
            "parent_gain_improvement",
            "diagnostic_gain_improvement",
        ):
            object.__setattr__(
                self, name, _readonly(getattr(self, name), dtype=np.dtype(np.float64))
            )
        object.__setattr__(
            self,
            "fine_cell_row_count",
            _readonly(counts, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(self, "parent_gain", parent_gain)
        object.__setattr__(self, "diagnostic_gain", diagnostic_gain)

    @property
    def maximum_identity_error(self) -> float:
        return max(
            float(self.maximum_raw_identity_error),
            float(self.maximum_parent_gain_identity_error),
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "path_ids": np.array(self.path_ids, copy=True),
            "cross_term": np.array(self.cross_term, copy=True),
            "prediction_energy": np.array(self.prediction_energy, copy=True),
            "raw_improvement": np.array(self.raw_improvement, copy=True),
            "parent_gain_improvement": np.array(
                self.parent_gain_improvement, copy=True
            ),
            "diagnostic_gain_improvement": np.array(
                self.diagnostic_gain_improvement, copy=True
            ),
            "fine_cell_row_count": np.array(self.fine_cell_row_count, copy=True),
        }


def reduce_quadratic_cells(
    *,
    targets: Any,
    predictions: Any,
    row_path_ids: Any,
    phases: Any,
    midpoint_indices: Any,
    expected_path_ids: Any | None = None,
    parent_gain: float = 1.0,
    diagnostic_gain: float = 1.0,
    identity_tolerance: float = IDENTITY_TOLERANCE,
) -> CandidateRoleDecomposition:
    """Reduce raw target/prediction rows to path-by-7-by-8 ``C/P/I`` cells.

    ``targets`` and ``predictions`` may be ``[row]`` or ``[row, component...]``.
    They are converted to binary64 before any product.  ``fine_cell_row_count``
    counts cache rows (not flattened model-output components).
    """

    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if target.ndim == 1:
        target = target[:, None]
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if (
        target.ndim < 2
        or target.shape != prediction.shape
        or target.shape[0] == 0
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        raise DirectionAdjudicationError(
            "targets and predictions must be finite, nonempty, and equal-shaped",
            failure_code="finite_reductions_invalid",
        )
    target = np.ascontiguousarray(target.reshape(target.shape[0], -1))
    prediction = np.ascontiguousarray(prediction.reshape(prediction.shape[0], -1))
    paths = _strict_integer_array(row_path_ids, "row_path_ids")
    phase = _strict_integer_array(phases, "phases")
    midpoint = _strict_integer_array(midpoint_indices, "midpoint_indices")
    if paths.size != target.shape[0] or phase.size != paths.size or midpoint.size != paths.size:
        raise DirectionAdjudicationError("row identities do not match prediction rows")
    if np.any((phase < 0) | (phase >= PHASE_COUNT)):
        raise DirectionAdjudicationError("phase indices must lie in [0,7)")
    if np.any((midpoint < 0) | (midpoint >= MIDPOINT_COUNT)):
        raise DirectionAdjudicationError("midpoint indices must lie in [0,8)")
    if expected_path_ids is None:
        _, first = np.unique(paths, return_index=True)
        canonical_paths = paths[np.sort(first)]
    else:
        canonical_paths = _strict_integer_array(expected_path_ids, "expected_path_ids")
    if canonical_paths.size == 0 or np.unique(canonical_paths).size != canonical_paths.size:
        raise DirectionAdjudicationError("expected path IDs must be nonempty and unique")
    if set(int(value) for value in paths) != set(int(value) for value in canonical_paths):
        raise DirectionAdjudicationError("row path IDs do not equal the expected path set")

    active_parent_gain = _finite_float(parent_gain, "parent_gain")
    active_diagnostic_gain = _finite_float(diagnostic_gain, "diagnostic_gain")
    tolerance = _finite_float(identity_tolerance, "identity_tolerance")
    if tolerance < 0.0:
        raise DirectionAdjudicationError("identity tolerance cannot be negative")
    shape = (canonical_paths.size, PHASE_COUNT, MIDPOINT_COUNT)
    cross = np.empty(shape, dtype=np.float64)
    energy = np.empty(shape, dtype=np.float64)
    raw_direct = np.empty(shape, dtype=np.float64)
    parent_direct = np.empty(shape, dtype=np.float64)
    counts = np.empty(shape, dtype=np.int64)

    for path_index, path_id in enumerate(canonical_paths):
        for phase_index in range(PHASE_COUNT):
            for midpoint_index in range(MIDPOINT_COUNT):
                indices = np.flatnonzero(
                    (paths == path_id)
                    & (phase == phase_index)
                    & (midpoint == midpoint_index)
                )
                if indices.size == 0:
                    raise DirectionAdjudicationError(
                        "every path/phase/midpoint cell must contain a row",
                        failure_code="fine_cell_coverage_invalid",
                    )
                selected_target = np.ascontiguousarray(target[indices]).ravel(order="C")
                selected_prediction = np.ascontiguousarray(prediction[indices]).ravel(
                    order="C"
                )
                scalar_count = selected_target.size
                cross_value = math.fsum(
                    float(z) * float(m)
                    for z, m in zip(
                        selected_target, selected_prediction, strict=True
                    )
                ) / scalar_count
                energy_value = math.fsum(
                    float(m) * float(m) for m in selected_prediction
                ) / scalar_count
                raw_value = math.fsum(
                    float(z) * float(z) - (float(z) - float(m)) ** 2
                    for z, m in zip(
                        selected_target, selected_prediction, strict=True
                    )
                ) / scalar_count
                parent_value = math.fsum(
                    float(z) * float(z)
                    - (float(z) - active_parent_gain * float(m)) ** 2
                    for z, m in zip(
                        selected_target, selected_prediction, strict=True
                    )
                ) / scalar_count
                cross[path_index, phase_index, midpoint_index] = cross_value
                energy[path_index, phase_index, midpoint_index] = energy_value
                raw_direct[path_index, phase_index, midpoint_index] = raw_value
                parent_direct[path_index, phase_index, midpoint_index] = parent_value
                counts[path_index, phase_index, midpoint_index] = indices.size

    raw_reconstructed = quadratic_improvement(cross, energy, 1.0)
    parent_reconstructed = quadratic_improvement(
        cross, energy, active_parent_gain
    )
    diagnostic = quadratic_improvement(cross, energy, active_diagnostic_gain)
    raw_error = float(np.max(np.abs(raw_direct - raw_reconstructed)))
    parent_error = float(np.max(np.abs(parent_direct - parent_reconstructed)))
    if raw_error > tolerance or parent_error > tolerance:
        raise DirectionAdjudicationError(
            "direct and reconstructed quadratic improvements disagree",
            failure_code="quadratic_identity_invalid",
        )
    return CandidateRoleDecomposition(
        path_ids=canonical_paths,
        cross_term=cross,
        prediction_energy=energy,
        raw_improvement=raw_direct,
        parent_gain_improvement=parent_direct,
        diagnostic_gain_improvement=diagnostic,
        fine_cell_row_count=counts,
        parent_gain=active_parent_gain,
        diagnostic_gain=active_diagnostic_gain,
        maximum_raw_identity_error=raw_error,
        maximum_parent_gain_identity_error=parent_error,
    )


# Descriptive alias used by the decomposition workflow.
aggregate_quadratic_decomposition = reduce_quadratic_cells


def pooled_cell_map(values: Any, counts: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pool a path-by-cell table without pairing paths across evidence roles."""

    cells, weights = _cell_arrays(values, counts)
    if cells.ndim == 2:
        return (
            np.ascontiguousarray(cells, dtype=np.float64),
            np.ascontiguousarray(weights, dtype=np.int64),
        )
    pooled = np.empty((PHASE_COUNT, MIDPOINT_COUNT), dtype=np.float64)
    pooled_counts = np.sum(weights, axis=0, dtype=np.int64)
    for phase in range(PHASE_COUNT):
        for midpoint in range(MIDPOINT_COUNT):
            pooled[phase, midpoint] = _weighted_mean(
                cells[:, phase, midpoint],
                weights[:, phase, midpoint],
                name=f"phase{phase}.midpoint{midpoint}",
            )
    return pooled, np.ascontiguousarray(pooled_counts)


def summarize_cell_map(values: Any, counts: Any | None = None) -> dict[str, Any]:
    """Return pooled, phase, midpoint, cell-count, and q1-sentinel reductions."""

    cells, weights = pooled_cell_map(values, counts)
    pooled = _weighted_mean(cells, weights, name="pooled map")
    phase_values = tuple(
        _weighted_mean(cells[index], weights[index], name=f"phase{index}")
        for index in range(PHASE_COUNT)
    )
    midpoint_values = tuple(
        _weighted_mean(
            cells[:, index], weights[:, index], name=f"midpoint{index}"
        )
        for index in range(MIDPOINT_COUNT)
    )
    return {
        "pooled": pooled,
        "phase_marginals": phase_values,
        "midpoint_marginals": midpoint_values,
        "fine_cells": tuple(tuple(float(value) for value in row) for row in cells),
        "positive_fine_cell_count": int(np.count_nonzero(cells > 0.0)),
        "phase4_midpoint7": float(cells[Q1_SENTINEL]),
        "fine_cell_row_count": tuple(
            tuple(int(value) for value in row) for row in weights
        ),
    }


def directional_compatibility_screen(
    values: Any,
    *,
    quartile: int,
    counts: Any | None = None,
) -> dict[str, Any]:
    """Apply the frozen local directional ``C`` screen to one role."""

    active_quartile = _quartile(quartile)
    summary = summarize_cell_map(values, counts)
    checks = {
        "pooled_cross_term_positive": int(summary["pooled"] > 0.0),
        "all_phase_marginals_positive": int(
            all(float(value) > 0.0 for value in summary["phase_marginals"])
        ),
        "all_midpoint_marginals_positive": int(
            all(float(value) > 0.0 for value in summary["midpoint_marginals"])
        ),
        "at_least_51_of_56_fine_cells_positive": int(
            int(summary["positive_fine_cell_count"])
            >= MINIMUM_POSITIVE_FINE_CELLS
        ),
        "q1_phase4_midpoint7_positive": int(
            active_quartile != 1 or float(summary["phase4_midpoint7"]) > 0.0
        ),
    }
    if not checks["pooled_cross_term_positive"]:
        reason = "pooled_cross_term_nonpositive"
    elif not checks["all_phase_marginals_positive"]:
        reason = "phase_marginal_nonpositive"
    elif not checks["all_midpoint_marginals_positive"]:
        reason = "midpoint_marginal_nonpositive"
    elif not checks["at_least_51_of_56_fine_cells_positive"]:
        reason = "positive_fine_cells_below_51"
    elif not checks["q1_phase4_midpoint7_positive"]:
        reason = "q1_phase4_midpoint7_nonpositive"
    else:
        reason = "directionally_compatible"
    passed = int(all(checks.values()))
    return {
        "schema": SCHEMA + "-directional-screen",
        "quartile": active_quartile,
        **summary,
        "checks": checks,
        "passed": passed,
        "directional_screen_passed": passed,
        "reason_code": reason,
        "authorizing": 0,
    }


directional_screen = directional_compatibility_screen


def path_stability_diagnostics(
    cross_terms: Any,
    *,
    counts: Any | None = None,
    transferred_improvements: Any | None = None,
    transferred_counts: Any | None = None,
) -> dict[str, Any]:
    """Compute within-role path signs, LOO minima, SD, and standard error."""

    cross, cross_counts = _cell_arrays(cross_terms, counts)
    if cross.ndim != 3 or cross.shape[0] < 2:
        raise DirectionAdjudicationError("path stability requires at least two paths")
    path_values = tuple(
        _weighted_mean(cross[index], cross_counts[index], name=f"path{index}")
        for index in range(cross.shape[0])
    )
    pooled = _weighted_mean(cross, cross_counts, name="all paths")
    loo = tuple(
        _weighted_mean(
            np.delete(cross, index, axis=0),
            np.delete(cross_counts, index, axis=0),
            name=f"leave-path-{index}-out",
        )
        for index in range(cross.shape[0])
    )
    mean_path = math.fsum(path_values) / len(path_values)
    variance = math.fsum(
        (float(value) - mean_path) ** 2 for value in path_values
    ) / (len(path_values) - 1)
    deviation = math.sqrt(max(0.0, variance))
    result: dict[str, Any] = {
        "path_count": len(path_values),
        "pooled_cross_term": pooled,
        "path_cross_terms": path_values,
        "positive_cross_term_path_count": sum(value > 0.0 for value in path_values),
        "leave_one_path_out_cross_terms": loo,
        "minimum_leave_one_path_out_cross_term": min(loo),
        "all_leave_one_path_out_cross_terms_positive": int(min(loo) > 0.0),
        "path_standard_deviation": deviation,
        "path_standard_error": deviation / math.sqrt(len(path_values)),
        "authorizing": 0,
    }
    if transferred_improvements is None:
        result.update(
            {
                "transferred_path_improvements": None,
                "positive_transferred_improvement_path_count": None,
                "leave_one_path_out_transferred_improvements": None,
                "minimum_leave_one_path_out_transferred_improvement": None,
            }
        )
        return result
    transferred, active_counts = _cell_arrays(
        transferred_improvements,
        cross_counts if transferred_counts is None else transferred_counts,
    )
    if transferred.shape != cross.shape:
        raise DirectionAdjudicationError(
            "transferred improvement must match the cross-term path table"
        )
    transferred_paths = tuple(
        _weighted_mean(
            transferred[index], active_counts[index], name=f"transferred path{index}"
        )
        for index in range(transferred.shape[0])
    )
    transferred_loo = tuple(
        _weighted_mean(
            np.delete(transferred, index, axis=0),
            np.delete(active_counts, index, axis=0),
            name=f"transferred leave-path-{index}-out",
        )
        for index in range(transferred.shape[0])
    )
    result.update(
        {
            "transferred_path_improvements": transferred_paths,
            "positive_transferred_improvement_path_count": sum(
                value > 0.0 for value in transferred_paths
            ),
            "leave_one_path_out_transferred_improvements": transferred_loo,
            "minimum_leave_one_path_out_transferred_improvement": min(
                transferred_loo
            ),
        }
    )
    return result


def evaluate_cross_role_directional_stability(
    *,
    gain_screen: Mapping[str, Any],
    rank_screen: Mapping[str, Any],
    gain_path_stability: Mapping[str, Any],
    rank_path_stability: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the independent-role screen without pairing disjoint path IDs."""

    checks = {
        "gain_directional_screen_passed": int(gain_screen.get("passed", 0) == 1),
        "rank_directional_screen_passed": int(rank_screen.get("passed", 0) == 1),
        "gain_at_least_24_positive_paths": int(
            int(gain_path_stability.get("positive_cross_term_path_count", -1)) >= 24
        ),
        "rank_at_least_24_positive_paths": int(
            int(rank_path_stability.get("positive_cross_term_path_count", -1)) >= 24
        ),
        "gain_all_leave_one_path_out_positive": int(
            float(
                gain_path_stability.get(
                    "minimum_leave_one_path_out_cross_term", -math.inf
                )
            )
            > 0.0
        ),
        "rank_all_leave_one_path_out_positive": int(
            float(
                rank_path_stability.get(
                    "minimum_leave_one_path_out_cross_term", -math.inf
                )
            )
            > 0.0
        ),
    }
    passed = int(all(checks.values()))
    return {
        "schema": SCHEMA + "-cross-role-stability",
        "checks": checks,
        "passed": passed,
        "cross_role_directionally_stable": passed,
        "paths_paired_across_roles": 0,
        "authorizing": 0,
    }


def cross_role_directionally_stable(**kwargs: Any) -> bool:
    return bool(evaluate_cross_role_directional_stability(**kwargs)["passed"])


def gain_transfer_diagnostics(
    *,
    gain_cross_term: float,
    gain_prediction_energy: float,
    rank_cross_term: float,
    rank_prediction_energy: float,
    gain_permitted: bool = True,
) -> dict[str, Any]:
    """Compare the gain-role optimum with the independent rank-role optimum."""

    gain = scalar_optimum(gain_cross_term, gain_prediction_energy)
    rank = scalar_optimum(rank_cross_term, rank_prediction_energy)
    transferred: float | None = None
    if gain["lambda_star"] is not None:
        transferred = quadratic_improvement(
            rank["cross_term"], rank["prediction_energy"], gain["lambda_star"]
        )
    rank_ceiling = rank["optimal_improvement"]
    efficiency = (
        transferred / rank_ceiling
        if transferred is not None and rank_ceiling is not None and rank_ceiling > 0.0
        else None
    )
    return {
        "C_gain": gain["cross_term"],
        "P_gain": gain["prediction_energy"],
        "lambda_gain_star": gain["lambda_star"],
        "I_gain_star": gain["optimal_improvement"],
        "C_rank": rank["cross_term"],
        "P_rank": rank["prediction_energy"],
        "lambda_rank_star": rank["lambda_star"],
        "I_rank_at_lambda_gain": transferred,
        "I_rank_at_lambda_rank": rank_ceiling,
        "transfer_efficiency": efficiency,
        "gain_positive_and_permitted": int(
            gain["lambda_star"] is not None and bool(gain_permitted)
        ),
        "gain_permitted": int(bool(gain_permitted)),
        "transfer_efficiency_clipped": 0,
        "authorizing": 0,
    }


def cancellation_diagnostics(values: Any, counts: Any | None = None) -> dict[str, Any]:
    """Summarize local cancellation and the fixed balanced two-way decomposition."""

    cells, weights = pooled_cell_map(values, counts)
    summary = summarize_cell_map(cells, weights)
    absolute_mean = _weighted_mean(np.abs(cells), weights, name="absolute C map")
    ratio = (
        0.0
        if absolute_mean == 0.0
        else 1.0 - abs(float(summary["pooled"])) / absolute_mean
    )
    # The phase/midpoint grid is a fixed balanced 7x8 design.  Row-count
    # weights govern pooled C and cancellation; the two-way SS decomposition
    # is the uniquely defined balanced map decomposition.
    grand = math.fsum(float(value) for value in cells.ravel(order="C")) / FINE_CELL_COUNT
    phase_means = np.asarray(
        [
            math.fsum(float(value) for value in cells[index]) / MIDPOINT_COUNT
            for index in range(PHASE_COUNT)
        ],
        dtype=np.float64,
    )
    midpoint_means = np.asarray(
        [
            math.fsum(float(value) for value in cells[:, index]) / PHASE_COUNT
            for index in range(MIDPOINT_COUNT)
        ],
        dtype=np.float64,
    )
    phase_ss = MIDPOINT_COUNT * math.fsum(
        (float(value) - grand) ** 2 for value in phase_means
    )
    midpoint_ss = PHASE_COUNT * math.fsum(
        (float(value) - grand) ** 2 for value in midpoint_means
    )
    interaction = np.empty_like(cells)
    for phase in range(PHASE_COUNT):
        for midpoint in range(MIDPOINT_COUNT):
            interaction[phase, midpoint] = (
                cells[phase, midpoint]
                - phase_means[phase]
                - midpoint_means[midpoint]
                + grand
            )
    interaction_ss = math.fsum(
        float(value) * float(value) for value in interaction.ravel(order="C")
    )
    total_ss = math.fsum(
        (float(value) - grand) ** 2 for value in cells.ravel(order="C")
    )
    return {
        **summary,
        "minimum_phase_marginal": min(summary["phase_marginals"]),
        "minimum_midpoint_marginal": min(summary["midpoint_marginals"]),
        "weighted_mean_absolute_cross_term": absolute_mean,
        "weighted_cancellation_ratio": ratio,
        "phase_sum_of_squares": phase_ss,
        "midpoint_sum_of_squares": midpoint_ss,
        "interaction_sum_of_squares": interaction_ss,
        "total_sum_of_squares": total_ss,
        "two_way_identity_error": abs(
            total_ss - phase_ss - midpoint_ss - interaction_ss
        ),
        "authorizing": 0,
    }


def compare_direction_maps(
    first: Any,
    second: Any,
    *,
    first_counts: Any | None = None,
    second_counts: Any | None = None,
    first_prediction_energy: Any | None = None,
    second_prediction_energy: Any | None = None,
) -> dict[str, Any]:
    """Compare two frozen 56-cell directions without selecting either map."""

    left, left_weights = pooled_cell_map(first, first_counts)
    right, right_weights = pooled_cell_map(second, second_counts)
    left_flat = left.ravel(order="C")
    right_flat = right.ravel(order="C")
    left_norm = math.sqrt(math.fsum(float(value) ** 2 for value in left_flat))
    right_norm = math.sqrt(math.fsum(float(value) ** 2 for value in right_flat))
    cosine = None
    if left_norm > 0.0 and right_norm > 0.0:
        dot = math.fsum(
            float(a) * float(b)
            for a, b in zip(left_flat, right_flat, strict=True)
        )
        cosine = dot / (left_norm * right_norm)
        # Roundoff can put a mathematical cosine a few ulps outside [-1,1].
        cosine = min(1.0, max(-1.0, cosine))
    sign_flips = math.fsum(
        1.0 if np.signbit(float(a)) != np.signbit(float(b)) else 0.0
        for a, b in zip(left_flat, right_flat, strict=True)
    ) / FINE_CELL_COUNT
    left_pooled = _weighted_mean(left, left_weights, name="first direction map")
    right_pooled = _weighted_mean(right, right_weights, name="second direction map")
    lambda_change: float | None = None
    first_lambda: float | None = None
    second_lambda: float | None = None
    if first_prediction_energy is not None or second_prediction_energy is not None:
        if first_prediction_energy is None or second_prediction_energy is None:
            raise DirectionAdjudicationError("both prediction-energy maps are required")
        first_energy, first_energy_counts = pooled_cell_map(
            first_prediction_energy, first_counts
        )
        second_energy, second_energy_counts = pooled_cell_map(
            second_prediction_energy, second_counts
        )
        first_optimum = scalar_optimum(
            left_pooled,
            _weighted_mean(first_energy, first_energy_counts, name="first energy map"),
        )
        second_optimum = scalar_optimum(
            right_pooled,
            _weighted_mean(second_energy, second_energy_counts, name="second energy map"),
        )
        first_lambda = first_optimum["lambda_star"]
        second_lambda = second_optimum["lambda_star"]
        if first_lambda is not None and second_lambda is not None:
            lambda_change = second_lambda - first_lambda
    return {
        "cosine_similarity": cosine,
        "cell_sign_flip_fraction": sign_flips,
        "first_pooled_cross_term": left_pooled,
        "second_pooled_cross_term": right_pooled,
        "pooled_cross_term_change": right_pooled - left_pooled,
        "first_lambda_star": first_lambda,
        "second_lambda_star": second_lambda,
        "lambda_star_change": lambda_change,
        "checkpoint_selected": 0,
        "authorizing": 0,
    }


def summarize_optimization_rotation(
    maps_by_seed: Mapping[int, Mapping[int, Any]],
    *,
    counts_by_seed: Mapping[int, Mapping[int, Any]] | None = None,
    energy_by_seed: Mapping[int, Mapping[int, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the frozen adjacent-update and same-update cross-seed rotation rule."""

    if not isinstance(maps_by_seed, Mapping) or not maps_by_seed:
        raise DirectionAdjudicationError("rotation diagnostics require seed/update maps")
    seeds = tuple(sorted(int(seed) for seed in maps_by_seed))
    if len(seeds) != len(maps_by_seed):
        raise DirectionAdjudicationError("rotation seed identities are malformed")
    adjacent_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in seeds:
        update_maps = maps_by_seed[seed]
        updates = tuple(sorted(int(update) for update in update_maps))
        if len(updates) < 2 or len(updates) != len(update_maps):
            raise DirectionAdjudicationError("each seed requires unique adjacent updates")
        local_rows: list[dict[str, Any]] = []
        for first_update, second_update in zip(updates, updates[1:]):
            if second_update - first_update != 100:
                raise DirectionAdjudicationError(
                    "optimization rotation compares adjacent 100-update checkpoints"
                )
            kwargs: dict[str, Any] = {}
            if counts_by_seed is not None:
                kwargs.update(
                    first_counts=counts_by_seed[seed][first_update],
                    second_counts=counts_by_seed[seed][second_update],
                )
            if energy_by_seed is not None:
                kwargs.update(
                    first_prediction_energy=energy_by_seed[seed][first_update],
                    second_prediction_energy=energy_by_seed[seed][second_update],
                )
            comparison = compare_direction_maps(
                update_maps[first_update], update_maps[second_update], **kwargs
            )
            row = {
                "seed": seed,
                "first_update": first_update,
                "second_update": second_update,
                **comparison,
            }
            local_rows.append(row)
            adjacent_rows.append(row)
        cosines = [
            float(row["cosine_similarity"])
            for row in local_rows
            if row["cosine_similarity"] is not None
        ]
        flips = [float(row["cell_sign_flip_fraction"]) for row in local_rows]
        median_cosine = statistics.median(cosines) if cosines else None
        median_flips = statistics.median(flips)
        seed_summaries.append(
            {
                "seed": seed,
                "median_adjacent_update_cosine": median_cosine,
                "median_adjacent_update_sign_flip_fraction": median_flips,
                "rotating": int(
                    median_cosine is not None
                    and median_cosine < 0.5
                    and median_flips >= 0.25
                ),
            }
        )
    cross_seed_rows: list[dict[str, Any]] = []
    for first_seed, second_seed in combinations(seeds, 2):
        common_updates = sorted(
            set(maps_by_seed[first_seed]).intersection(maps_by_seed[second_seed])
        )
        for update in common_updates:
            kwargs = {}
            if counts_by_seed is not None:
                kwargs.update(
                    first_counts=counts_by_seed[first_seed][update],
                    second_counts=counts_by_seed[second_seed][update],
                )
            comparison = compare_direction_maps(
                maps_by_seed[first_seed][update],
                maps_by_seed[second_seed][update],
                **kwargs,
            )
            cross_seed_rows.append(
                {
                    "first_seed": first_seed,
                    "second_seed": second_seed,
                    "update": int(update),
                    **comparison,
                }
            )
    cross_cosines = [
        float(row["cosine_similarity"])
        for row in cross_seed_rows
        if row["cosine_similarity"] is not None
    ]
    median_cross_seed = statistics.median(cross_cosines) if cross_cosines else None
    rotating_seed_count = sum(int(row["rotating"]) for row in seed_summaries)
    flag = bool(
        rotating_seed_count >= 2
        or (median_cross_seed is not None and median_cross_seed < 0.5)
    )
    return {
        "adjacent_update_comparisons": tuple(adjacent_rows),
        "seed_summaries": tuple(seed_summaries),
        "same_update_cross_seed_comparisons": tuple(cross_seed_rows),
        "rotating_seed_count": rotating_seed_count,
        "median_same_update_cross_seed_cosine": median_cross_seed,
        "optimization_time_rotation": int(flag),
        "three_seed_grid_present": int(len(seeds) == 3),
        "checkpoint_selected": 0,
        "authorizing": 0,
    }


def forecast_required_paths(
    transferred_improvements: Any,
    *,
    quartile: int,
    counts: Any | None = None,
    critical_value: float = CRITICAL_VALUE,
) -> dict[str, Any]:
    """Forecast whole paths only after every original point screen passes."""

    active_quartile = _quartile(quartile)
    values, weights = _cell_arrays(transferred_improvements, counts)
    if values.ndim != 3 or values.shape[0] < 2:
        raise DirectionAdjudicationError("path forecast requires whole-path cell tables")
    screen = directional_compatibility_screen(
        values, quartile=active_quartile, counts=weights
    )
    path_values = tuple(
        _weighted_mean(values[index], weights[index], name=f"forecast path{index}")
        for index in range(values.shape[0])
    )
    point = math.fsum(path_values) / len(path_values)
    deviation = math.sqrt(
        math.fsum((float(value) - point) ** 2 for value in path_values)
        / (len(path_values) - 1)
    )
    critical = _finite_float(critical_value, "critical_value")
    if critical != CRITICAL_VALUE:
        raise DirectionAdjudicationError(
            f"critical_value is frozen at {CRITICAL_VALUE}"
        )
    if not screen["passed"]:
        return {
            "quartile": active_quartile,
            "point_screen_passed": 0,
            "point_screen_reason": screen["reason_code"],
            "path_count": len(path_values),
            "path_point_estimate": point,
            "path_standard_deviation": deviation,
            "critical_value": critical,
            "n_raw": None,
            "n_rounded": None,
            "required_path_count": math.inf,
            "required_path_count_is_infinite": 1,
            "reason": "negative_or_incompatible_point_effect",
            "planning_forecast_only": 1,
            "authorizing": 0,
        }
    raw_value = (critical * deviation / point) ** 2
    if not math.isfinite(raw_value):
        n_raw: int | None = None
        n_rounded: int | None = None
        required: int | float = math.inf
        reason = "nonfinite_power_forecast"
    else:
        n_raw = int(math.ceil(raw_value))
        n_rounded = 32 * int(math.ceil(n_raw / 32.0))
        required = n_rounded
        reason = "finite_power_forecast"
    return {
        "quartile": active_quartile,
        "point_screen_passed": 1,
        "point_screen_reason": screen["reason_code"],
        "path_count": len(path_values),
        "path_point_estimate": point,
        "path_standard_deviation": deviation,
        "critical_value": critical,
        "n_raw": n_raw,
        "n_rounded": n_rounded,
        "required_path_count": required,
        "required_path_count_is_infinite": int(not math.isfinite(float(required))),
        "reason": reason,
        "planning_forecast_only": 1,
        "authorizing": 0,
    }


forecast_required_path_count = forecast_required_paths


def _record_value(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


MECHANISM_FLAG_NAMES = (
    "conditional_direction_absent",
    "direction_present_but_role_unstable",
    "phase_midpoint_cancellation",
    "gain_transfer_failure",
    "optimization_time_rotation",
    "strictly_positive_but_too_small",
)


def classify_mechanism_flags(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    optimization_time_rotation: bool = False,
) -> dict[str, Any]:
    """Compute the six frozen, nonexclusive mechanism flags for one quartile.

    Candidate records use the canonical keys emitted by this module's summary
    helpers.  A few descriptive aliases are accepted to keep CSV/JSON callers
    lossless (for example ``C_gain`` and ``gain_pooled_cross_term``).
    """

    records = tuple(candidate_records)
    if not records or any(not isinstance(record, Mapping) for record in records):
        raise DirectionAdjudicationError("mechanism classification needs candidates")
    normalized: list[dict[str, Any]] = []
    for record in records:
        gain_c = _finite_float(
            _record_value(record, "C_gain", "gain_pooled_cross_term"), "C_gain"
        )
        rank_c = _finite_float(
            _record_value(record, "C_rank", "rank_pooled_cross_term"), "C_rank"
        )
        gain_screen_value = _record_value(
            record, "gain_directional_screen_passed", "gain_screen_passed", default=0
        )
        rank_screen_value = _record_value(
            record, "rank_directional_screen_passed", "rank_screen_passed", default=0
        )
        if isinstance(gain_screen_value, Mapping):
            gain_screen_value = gain_screen_value.get("passed", 0)
        if isinstance(rank_screen_value, Mapping):
            rank_screen_value = rank_screen_value.get("passed", 0)
        stable = bool(
            _record_value(
                record,
                "cross_role_directionally_stable",
                "cross_role_stable",
                default=0,
            )
        )
        lambda_gain = _record_value(record, "lambda_gain_star", "gain_lambda_star")
        if lambda_gain is not None:
            lambda_gain = _finite_float(lambda_gain, "lambda_gain_star")
        rank_at_gain = _record_value(
            record, "I_rank_at_lambda_gain", "rank_improvement_at_gain"
        )
        if rank_at_gain is not None:
            rank_at_gain = _finite_float(rank_at_gain, "I_rank_at_lambda_gain")
        margin = _record_value(record, "fixed_design_margin", "rank_fixed_design_margin")
        if margin is not None:
            margin = _finite_float(margin, "fixed_design_margin")
        normalized.append(
            {
                "gain_c": gain_c,
                "rank_c": rank_c,
                "gain_screen": bool(gain_screen_value),
                "rank_screen": bool(rank_screen_value),
                "stable": stable,
                "lambda_gain": lambda_gain,
                "gain_permitted": bool(record.get("gain_permitted", 0)),
                "rank_at_gain": rank_at_gain,
                "point_screen": bool(
                    _record_value(
                        record,
                        "transferred_rank_point_screen_passed",
                        "point_screen_passed",
                        default=0,
                    )
                ),
                "margin": margin,
            }
        )
    any_positive_direction = any(
        row["gain_c"] > 0.0 or row["rank_c"] > 0.0 for row in normalized
    )
    any_stable = any(row["stable"] for row in normalized)
    cancellation = any(
        row["gain_c"] > 0.0
        and row["rank_c"] > 0.0
        and (not row["gain_screen"] or not row["rank_screen"])
        for row in normalized
    )
    transfer_failure = any(
        row["lambda_gain"] is not None
        and row["lambda_gain"] > 0.0
        and row["gain_permitted"]
        and row["rank_at_gain"] is not None
        and row["rank_at_gain"] <= 0.0
        for row in normalized
    )
    too_small = any(
        row["stable"]
        and row["point_screen"]
        and row["margin"] is not None
        and row["margin"] <= 0.0
        for row in normalized
    )
    flags = {
        "conditional_direction_absent": int(not any_positive_direction),
        "direction_present_but_role_unstable": int(
            any_positive_direction and not any_stable
        ),
        "phase_midpoint_cancellation": int(cancellation),
        "gain_transfer_failure": int(transfer_failure),
        "optimization_time_rotation": int(bool(optimization_time_rotation)),
        "strictly_positive_but_too_small": int(too_small),
    }
    return {
        "schema": SCHEMA + "-mechanism-classification",
        **flags,
        "mechanism_flag_count": sum(flags.values()),
        "mechanism_localized": int(any(flags.values())),
        "candidate_count": len(normalized),
        "cross_role_stable_candidate_count": sum(
            int(row["stable"]) for row in normalized
        ),
        "historical_design_evidence_only": 1,
        "authorizing": 0,
    }


def classify_power_only_evidence(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Require qualifying finite forecasts in at least two of three seeds."""

    seeds = tuple(int(value) for value in expected_seeds) if expected_seeds is not None else tuple(
        sorted({int(record["seed"]) for record in candidate_records})
    )
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise DirectionAdjudicationError("power-only classification requires three seeds")
    qualifying: list[int] = []
    for seed in seeds:
        passes = False
        for record in candidate_records:
            if int(record.get("seed", -1)) != seed:
                continue
            stable = bool(
                _record_value(
                    record,
                    "cross_role_directionally_stable",
                    "cross_role_stable",
                    default=0,
                )
            )
            rounded = _record_value(record, "n_rounded", "required_path_count")
            if rounded is None:
                continue
            try:
                finite_rounded = float(rounded)
            except (TypeError, ValueError):
                continue
            if (
                stable
                and math.isfinite(finite_rounded)
                and finite_rounded <= POWER_ONLY_MAXIMUM_ROUNDED_PATH_COUNT
            ):
                passes = True
                break
        if passes:
            qualifying.append(seed)
    return {
        "seed_count": len(seeds),
        "qualifying_seed_count": len(qualifying),
        "qualifying_seeds": tuple(qualifying),
        "maximum_n_rounded": POWER_ONLY_MAXIMUM_ROUNDED_PATH_COUNT,
        "power_only_evidence": int(len(qualifying) >= 2),
        "historical_design_evidence_only": 1,
        "authorizing": 0,
    }


def canonical_candidate_order() -> tuple[str, ...]:
    """Return the sealed 480 nonzero candidates in quartile/seed/update order."""

    if len(CANDIDATE_ORDER) != CANDIDATE_COUNT:
        raise DirectionAdjudicationError("canonical candidate grid changed")
    return CANDIDATE_ORDER


__all__ = [
    "CANDIDATE_COUNT",
    "CANDIDATE_ORDER",
    "CRITICAL_VALUE",
    "CandidateRoleDecomposition",
    "DirectionAdjudicationError",
    "FINE_CELL_COUNT",
    "IDENTITY_TOLERANCE",
    "MAXIMUM_PREDICTION_BATCH_SIZE",
    "MECHANISM_FLAG_NAMES",
    "MIDPOINT_COUNT",
    "PATH_COUNT",
    "PHASE_COUNT",
    "POWER_ONLY_MAXIMUM_ROUNDED_PATH_COUNT",
    "ROLE_COUNT",
    "ROLE_ORDER",
    "SCHEMA",
    "aggregate_quadratic_decomposition",
    "cancellation_diagnostics",
    "canonical_candidate_order",
    "classify_mechanism_flags",
    "classify_power_only_evidence",
    "compare_direction_maps",
    "cross_role_directionally_stable",
    "directional_compatibility_screen",
    "directional_screen",
    "evaluate_cross_role_directional_stability",
    "forecast_required_path_count",
    "forecast_required_paths",
    "gain_transfer_diagnostics",
    "path_stability_diagnostics",
    "pooled_cell_map",
    "quadratic_improvement",
    "reduce_quadratic_cells",
    "scalar_optimum",
    "summarize_cell_map",
    "summarize_optimization_rotation",
]
