"""Path-level risk aggregation for boundary-tangent confirmation.

This module is deliberately independent of the trainer and transition
scheduler.  It converts separated, sealed midpoint-cache evidence into the
frozen 228-component whole-path table consumed by
``one_sided_whole_path_max_t``.  Rows are paths; edges, selected outer steps,
phases, and midpoint branches are never bootstrap units.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np

from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    MIDPOINT_COUNT,
    SELECTED_OUTER_STEPS,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    COMBINED_VS_ZERO_FAMILY_SIZE,
    CONFIRMATION_FAMILY_NAMES,
    CONFIRMATION_FAMILY_SIZE,
    CONTROLLER_FAMILY_SIZE,
    DEFAULT_CONTROLLER_BOOTSTRAP_SEED,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_SIMULTANEOUS_CONFIDENCE,
    PHASE_COUNT,
    TIME_QUARTILES,
)
from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE


SCHEMA = "d0-jacobi-rb-boundary-tangent-confirmation-risk-v1"
_CONTROLLER_BOOTSTRAP_NAMESPACE = 0x42544354


class BoundaryTangentConfirmationError(ValueError):
    """Raised when sealed confirmation rows violate the frozen design."""


def _integer_vector(value: Any, name: str, rows: int | None = None) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise BoundaryTangentConfirmationError(f"{name} must be an integer vector")
    result = np.asarray(source, dtype=np.int64)
    if rows is not None and result.shape != (rows,):
        raise BoundaryTangentConfirmationError(f"{name} has the wrong row count")
    return np.ascontiguousarray(result)


def _float_edge_table(value: Any, name: str, rows: int) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype != np.dtype(np.float64):
        raise BoundaryTangentConfirmationError(f"{name} must be binary64")
    if source.shape != (rows, EDGES_PER_PHASE) or not np.isfinite(source).all():
        raise BoundaryTangentConfirmationError(
            f"{name} must be finite [{rows},{EDGES_PER_PHASE}]"
        )
    return np.ascontiguousarray(source)


def _float_vector(value: Any, name: str, rows: int) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype != np.dtype(np.float64):
        raise BoundaryTangentConfirmationError(f"{name} must be binary64")
    if source.shape != (rows,) or not np.isfinite(source).all():
        raise BoundaryTangentConfirmationError(
            f"{name} must be a finite [{rows}] vector"
        )
    return np.ascontiguousarray(source)


def _canonical_paths(value: Any) -> np.ndarray:
    paths = _integer_vector(value, "expected_path_ids")
    if paths.size < 8 or np.any(paths < 0) or np.any(paths >= 1 << 20):
        raise BoundaryTangentConfirmationError(
            "expected path IDs must contain at least eight 20-bit IDs"
        )
    if np.unique(paths).size != paths.size:
        raise BoundaryTangentConfirmationError("expected path IDs are not unique")
    return np.sort(paths, kind="stable")


def _canonical_steps(value: Sequence[int]) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value)
    if (
        not steps
        or len(set(steps)) != len(steps)
        or tuple(sorted(steps)) != steps
        or any(step < 0 or step >= 512 for step in steps)
        or any(step // 128 not in range(TIME_QUARTILES) for step in steps)
    ):
        raise BoundaryTangentConfirmationError("selected outer steps are invalid")
    counts = np.bincount(
        np.asarray([step // 128 for step in steps], dtype=np.int64),
        minlength=TIME_QUARTILES,
    )
    if np.any(counts <= 0):
        raise BoundaryTangentConfirmationError(
            "selected outer steps must populate every forward quartile"
        )
    return steps


@dataclass(frozen=True)
class ConfirmationRiskTable:
    """Canonical path-level table for the frozen 228-member family."""

    path_ids: np.ndarray
    path_values: np.ndarray
    cell_counts: np.ndarray
    sample_key_sha256: str
    combined_vs_zero_row_count: int
    combined_vs_baseline_row_count: int

    def __post_init__(self) -> None:
        paths = np.ascontiguousarray(self.path_ids)
        values = np.ascontiguousarray(self.path_values)
        counts = np.ascontiguousarray(self.cell_counts)
        if (
            paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or values.dtype != np.dtype(np.float64)
            or values.shape != (paths.size, CONFIRMATION_FAMILY_SIZE)
            or counts.dtype != np.dtype(np.int64)
            or counts.shape != values.shape
            or not np.isfinite(values).all()
            or np.any(counts <= 0)
        ):
            raise BoundaryTangentConfirmationError(
                "confirmation risk table is malformed"
            )
        if (
            np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths, kind="stable"))
        ):
            raise BoundaryTangentConfirmationError("path table contains duplicates")
        if len(self.sample_key_sha256) != 64:
            raise BoundaryTangentConfirmationError("sample-key hash is malformed")
        paths.setflags(write=False)
        values.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "path_ids", paths)
        object.__setattr__(self, "path_values", values)
        object.__setattr__(self, "cell_counts", counts)

    def to_record(self) -> dict[str, Any]:
        point = np.mean(self.path_values, axis=0, dtype=np.float64)
        return {
            "schema": SCHEMA,
            "schema_version": 1,
            "family_names": list(CONFIRMATION_FAMILY_NAMES),
            "family_size": CONFIRMATION_FAMILY_SIZE,
            "path_ids": self.path_ids.tolist(),
            "path_count": int(self.path_ids.size),
            "sample_key_sha256": self.sample_key_sha256,
            "combined_vs_zero_row_count": int(
                self.combined_vs_zero_row_count
            ),
            "combined_vs_baseline_row_count": int(
                self.combined_vs_baseline_row_count
            ),
            "minimum_cell_count": int(np.min(self.cell_counts)),
            "maximum_cell_count": int(np.max(self.cell_counts)),
            "point_estimates": {
                name: float(value)
                for name, value in zip(
                    CONFIRMATION_FAMILY_NAMES, point, strict=True
                )
            },
            "bootstrap_unit": "whole_path",
            "negative_values_truncated": 0,
            "target_transformed": 0,
            "quotient_target_formed": 0,
        }


def aggregate_confirmation_improvements(
    *,
    sample_keys: Any,
    row_path_ids: Any,
    outer_steps: Any,
    phases: Any,
    midpoint_indices: Any,
    combined_vs_zero_improvements: Any,
    combined_vs_baseline_improvements: Any,
    expected_path_ids: Any,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
) -> ConfirmationRiskTable:
    """Aggregate precomputed scalar raw-risk improvements by whole path.

    The two improvement vectors must already be edge-averaged per cache row.
    This is the streaming entry point: callers can discard raw targets and
    predictions after sealing each row's two direct-MSE contrasts.
    """

    keys = _integer_vector(sample_keys, "sample_keys")
    rows = int(keys.size)
    paths = _integer_vector(row_path_ids, "row_path_ids", rows)
    steps = _integer_vector(outer_steps, "outer_steps", rows)
    occurrences = _integer_vector(phases, "phases", rows)
    midpoints = _integer_vector(midpoint_indices, "midpoint_indices", rows)
    zero_improvement = _float_vector(
        combined_vs_zero_improvements,
        "combined_vs_zero_improvements",
        rows,
    )
    baseline_improvement = _float_vector(
        combined_vs_baseline_improvements,
        "combined_vs_baseline_improvements",
        rows,
    )
    expected_paths = _canonical_paths(expected_path_ids)
    expected_steps = _canonical_steps(selected_outer_steps)
    expected_step_set = set(expected_steps)
    if np.unique(keys).size != rows:
        raise BoundaryTangentConfirmationError("sample keys are not unique")
    if set(np.unique(paths).tolist()) != set(expected_paths.tolist()):
        raise BoundaryTangentConfirmationError("confirmation path set changed")
    if set(np.unique(steps).tolist()) != expected_step_set:
        raise BoundaryTangentConfirmationError("selected outer-step set changed")
    if np.any((occurrences < 0) | (occurrences >= PHASE_COUNT)):
        raise BoundaryTangentConfirmationError("phase occurrence is invalid")
    if np.any((midpoints < 0) | (midpoints >= MIDPOINT_COUNT)):
        raise BoundaryTangentConfirmationError("midpoint index is invalid")
    expected_rows = (
        expected_paths.size
        * len(expected_steps)
        * PHASE_COUNT
        * MIDPOINT_COUNT
    )
    if rows != expected_rows:
        raise BoundaryTangentConfirmationError(
            f"confirmation row count {rows} != {expected_rows}"
        )
    expected_keys = np.fromiter(
        (
            midpoint_sample_key(int(path), int(step), int(phase), int(midpoint))
            for path, step, phase, midpoint in zip(
                paths, steps, occurrences, midpoints, strict=True
            )
        ),
        dtype=np.int64,
        count=rows,
    )
    if not np.array_equal(keys, expected_keys):
        raise BoundaryTangentConfirmationError(
            "sample keys do not match path/time/phase/midpoint identity"
        )
    identity = np.stack((paths, steps, occurrences, midpoints), axis=1)
    if np.unique(identity, axis=0).shape[0] != rows:
        raise BoundaryTangentConfirmationError("confirmation identity repeats")

    path_index = np.searchsorted(expected_paths, paths)
    if np.any(path_index >= expected_paths.size) or not np.array_equal(
        expected_paths[path_index], paths
    ):
        raise BoundaryTangentConfirmationError("confirmation path indexing failed")
    quartiles = steps // 128
    zero_column = (
        quartiles * (PHASE_COUNT * MIDPOINT_COUNT)
        + occurrences * MIDPOINT_COUNT
        + midpoints
    )
    baseline_column = COMBINED_VS_ZERO_FAMILY_SIZE + quartiles
    values = np.zeros(
        (expected_paths.size, CONFIRMATION_FAMILY_SIZE), dtype=np.float64
    )
    counts = np.zeros_like(values, dtype=np.int64)
    np.add.at(values, (path_index, zero_column), zero_improvement)
    np.add.at(counts, (path_index, zero_column), 1)
    np.add.at(values, (path_index, baseline_column), baseline_improvement)
    np.add.at(counts, (path_index, baseline_column), 1)
    if np.any(counts <= 0):
        raise BoundaryTangentConfirmationError("a confirmation family cell is empty")
    values /= counts
    key_digest = hashlib.sha256(
        np.ascontiguousarray(np.sort(keys, kind="stable")).tobytes(order="C")
    ).hexdigest()
    return ConfirmationRiskTable(
        path_ids=np.ascontiguousarray(expected_paths),
        path_values=np.ascontiguousarray(values),
        cell_counts=np.ascontiguousarray(counts),
        sample_key_sha256=key_digest,
        combined_vs_zero_row_count=rows,
        combined_vs_baseline_row_count=rows,
    )


def aggregate_confirmation_risks(
    *,
    sample_keys: Any,
    row_path_ids: Any,
    outer_steps: Any,
    phases: Any,
    midpoint_indices: Any,
    targets: Any,
    combined_predictions: Any,
    baseline_predictions: Any,
    expected_path_ids: Any,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
) -> ConfirmationRiskTable:
    """Aggregate direct raw-MSE improvements on the whole-path unit.

    ``combined_vs_zero`` is ``R(0)-R(combined)`` in every
    quartile/phase/midpoint cell.  ``combined_vs_baseline`` is
    ``R(baseline)-R(combined)`` pooled within each forward quartile.  The
    function never forms a quotient target, clips an improvement, or exposes
    the historical ambiguous ``data_end`` name.
    """

    keys = _integer_vector(sample_keys, "sample_keys")
    rows = int(keys.size)
    paths = _integer_vector(row_path_ids, "row_path_ids", rows)
    steps = _integer_vector(outer_steps, "outer_steps", rows)
    occurrences = _integer_vector(phases, "phases", rows)
    midpoints = _integer_vector(midpoint_indices, "midpoint_indices", rows)
    target = _float_edge_table(targets, "targets", rows)
    combined = _float_edge_table(
        combined_predictions, "combined_predictions", rows
    )
    baseline = _float_edge_table(
        baseline_predictions, "baseline_predictions", rows
    )
    zero_improvement = np.mean(
        target * target - (target - combined) ** 2,
        axis=1,
        dtype=np.float64,
    )
    baseline_improvement = np.mean(
        (target - baseline) ** 2 - (target - combined) ** 2,
        axis=1,
        dtype=np.float64,
    )
    if not np.isfinite(zero_improvement).all() or not np.isfinite(
        baseline_improvement
    ).all():
        raise BoundaryTangentConfirmationError("risk improvement is nonfinite")

    return aggregate_confirmation_improvements(
        sample_keys=keys,
        row_path_ids=paths,
        outer_steps=steps,
        phases=occurrences,
        midpoint_indices=midpoints,
        combined_vs_zero_improvements=np.ascontiguousarray(zero_improvement),
        combined_vs_baseline_improvements=np.ascontiguousarray(
            baseline_improvement
        ),
        expected_path_ids=expected_path_ids,
        selected_outer_steps=selected_outer_steps,
    )


def normalized_controller_trajectory_max_t(
    *,
    numerators: Any,
    forward_changes: Any,
    path_ids: Any,
    names: Sequence[str],
    confidence: float = DEFAULT_SIMULTANEOUS_CONFIDENCE,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_CONTROLLER_BOOTSTRAP_SEED,
    namespace: int = 0,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Return the frozen two-sided 784-family controller-law max-T record.

    Each feature is normalized by the RMS forward change on the same whole
    paths.  The RMS denominator is recomputed inside every bootstrap draw;
    treating it as fixed would understate uncertainty.  Rows are sorted by
    path ID before the stateless Philox bootstrap, so input ordering cannot
    affect the result.
    """

    numerator_source = np.asarray(numerators)
    denominator_source = np.asarray(forward_changes)
    if (
        numerator_source.dtype != np.dtype(np.float64)
        or denominator_source.dtype != np.dtype(np.float64)
    ):
        raise BoundaryTangentConfirmationError(
            "controller trajectory arrays must be binary64"
        )
    if (
        numerator_source.ndim != 2
        or numerator_source.shape != denominator_source.shape
        or numerator_source.shape[0] < 8
        or numerator_source.shape[1] != CONTROLLER_FAMILY_SIZE
        or not np.isfinite(numerator_source).all()
        or not np.isfinite(denominator_source).all()
    ):
        raise BoundaryTangentConfirmationError(
            "controller trajectory arrays must be finite [at-least-8,784]"
        )
    family_names = tuple(str(name) for name in names)
    if (
        len(family_names) != CONTROLLER_FAMILY_SIZE
        or len(set(family_names)) != CONTROLLER_FAMILY_SIZE
        or any(not name for name in family_names)
    ):
        raise BoundaryTangentConfirmationError(
            "controller trajectory family names must be 784 unique strings"
        )
    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates <= 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or not isinstance(namespace, int)
        or isinstance(namespace, bool)
        or namespace < 0
        or not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
        or not 0.5 < float(confidence) < 1.0
    ):
        raise BoundaryTangentConfirmationError(
            "controller max-T configuration is invalid"
        )

    paths = _integer_vector(path_ids, "path_ids", numerator_source.shape[0])
    if (
        np.any(paths < 0)
        or np.any(paths >= 1 << 20)
        or np.unique(paths).size != paths.size
    ):
        raise BoundaryTangentConfirmationError(
            "controller path IDs must be unique 20-bit integers"
        )
    order = np.argsort(paths, kind="stable")
    paths = np.ascontiguousarray(paths[order])
    numerator = np.ascontiguousarray(numerator_source[order])
    denominator = np.ascontiguousarray(denominator_source[order])
    path_count = int(paths.size)

    def _estimate(
        numerator_table: np.ndarray,
        denominator_table: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # The path axis is second-to-last for both [P,F] and [B,P,F].
        scale = np.sqrt(
            np.mean(denominator_table * denominator_table, axis=-2, dtype=np.float64)
        )
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise BoundaryTangentConfirmationError(
                "controller forward-change RMS is degenerate/nonfinite"
            )
        mean_numerator = np.mean(numerator_table, axis=-2, dtype=np.float64)
        point = mean_numerator / scale
        influence = (
            numerator_table / scale[..., None, :]
            - point[..., None, :]
            - mean_numerator[..., None, :]
            * (
                denominator_table * denominator_table
                - scale[..., None, :] * scale[..., None, :]
            )
            / (2.0 * scale[..., None, :] ** 3)
        )
        standard_error = (
            np.std(influence, axis=-2, ddof=1, dtype=np.float64)
            / math.sqrt(path_count)
        )
        if (
            not np.isfinite(point).all()
            or not np.isfinite(standard_error).all()
            or np.any(standard_error <= 0.0)
        ):
            raise BoundaryTangentConfirmationError(
                "controller trajectory studentization is degenerate/nonfinite"
            )
        return point, standard_error

    point, standard_error = _estimate(numerator, denominator)
    generator = np.random.Generator(
        np.random.Philox(
            [int(seed), int(namespace), _CONTROLLER_BOOTSTRAP_NAMESPACE]
        )
    )
    maxima = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), int(chunk_size)):
        stop = min(int(replicates), start + int(chunk_size))
        indices = generator.integers(
            0,
            path_count,
            size=(stop - start, path_count),
            dtype=np.int64,
        )
        draw_point, draw_error = _estimate(
            numerator[indices], denominator[indices]
        )
        maxima[start:stop] = np.max(
            np.abs(draw_point - point[None, :]) / draw_error,
            axis=1,
        )
    critical = float(
        np.quantile(maxima, float(confidence), method="higher")
    )
    if not math.isfinite(critical):
        raise BoundaryTangentConfirmationError(
            "controller bootstrap critical value is nonfinite"
        )
    upper = np.abs(point) + critical * standard_error
    return {
        "schema": SCHEMA + "-normalized-controller-trajectory-max-t",
        "schema_version": 1,
        "method": "whole_path_rms_normalized_two_sided_studentized_max_t",
        "bootstrap_unit": "whole_path",
        "family_resampling": "joint_across_all_784_components",
        "denominator_recomputed_per_resample": 1,
        "quantile_method": "higher",
        "family_names": list(family_names),
        "family_size": CONTROLLER_FAMILY_SIZE,
        "point_estimates": {
            name: float(value)
            for name, value in zip(family_names, point, strict=True)
        },
        "standard_errors": {
            name: float(value)
            for name, value in zip(family_names, standard_error, strict=True)
        },
        "simultaneous_upper_absolute": {
            name: float(value)
            for name, value in zip(family_names, upper, strict=True)
        },
        "critical_value": critical,
        "path_ids": paths.tolist(),
        "path_count": path_count,
        "confidence": float(confidence),
        "replicates": int(replicates),
        "seed": int(seed),
        "namespace": int(namespace),
        "negative_values_truncated": 0,
    }


__all__ = [
    "BoundaryTangentConfirmationError",
    "ConfirmationRiskTable",
    "SCHEMA",
    "aggregate_confirmation_improvements",
    "aggregate_confirmation_risks",
    "normalized_controller_trajectory_max_t",
]
