"""Pure forensic statistics for the boundary-tangent false discovery.

The historical eager boundary-tangent run is immutable.  This module only
validates and aggregates already-sealed path evidence, replays the historical
selection rule, and performs deterministic whole-path inference.  It imports
neither a trainer nor a transition/sampling implementation.

Two statistical families are deliberately distinct:

* the omitted baseline-versus-zero contrast has 229 members (224 fine cells,
  four forward-time quartiles, and one overall member) and is adjudicated by
  a two-sided max-|T| bootstrap; and
* the searched residual family has 480 members (120 candidates by four
  quartiles) and is adjudicated by a one-sided max-T bootstrap.

Rows are whole paths throughout.  Candidate and path inputs are canonicalized
before arithmetic so replay is invariant to their presentation order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    MIDPOINT_COUNT,
    SELECTED_OUTER_STEPS,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_confirmation import (
    ConfirmationRiskTable,
    aggregate_confirmation_improvements,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    COMBINED_VS_BASELINE_FAMILY_SIZE,
    COMBINED_VS_ZERO_FAMILY_SIZE,
    CONFIRMATION_FAMILY_NAMES,
    CONFIRMATION_FAMILY_SIZE,
    PHASE_COUNT,
    TIME_QUARTILES,
)


SCHEMA = "d0-jacobi-rb-boundary-tangent-false-discovery-v1"
SCHEMA_VERSION = 1

THREE_CONTRAST_IDENTITY_TOLERANCE = 5.0e-15
BASELINE_FAMILY_SIZE = COMBINED_VS_ZERO_FAMILY_SIZE + TIME_QUARTILES + 1
SEARCHED_CANDIDATE_COUNT = 120
SEARCHED_RESIDUAL_FAMILY_SIZE = (
    SEARCHED_CANDIDATE_COUNT * COMBINED_VS_BASELINE_FAMILY_SIZE
)
HISTORICAL_SEEDS = (261_312, 261_313, 261_314)
HISTORICAL_NONZERO_UPDATES = tuple(range(100, 4_001, 100))
HISTORICAL_UPDATES = (0,) + HISTORICAL_NONZERO_UPDATES
HISTORICAL_SELECTED_SEED = 261_314
HISTORICAL_SELECTED_UPDATE = 800

DEFAULT_BOOTSTRAP_SEED = 261_319
DEFAULT_BOOTSTRAP_REPLICATES = 50_000
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.995
_BASELINE_BOOTSTRAP_NAMESPACE = 0x42544644
_SEARCH_BOOTSTRAP_NAMESPACE = 0x53414644


def baseline_family_names() -> tuple[str, ...]:
    fine = tuple(
        f"baseline_vs_zero.q{quartile}.phase{phase}.midpoint{midpoint}"
        for quartile in range(TIME_QUARTILES)
        for phase in range(PHASE_COUNT)
        for midpoint in range(MIDPOINT_COUNT)
    )
    quartiles = tuple(
        f"baseline_vs_zero.q{quartile}" for quartile in range(TIME_QUARTILES)
    )
    names = fine + quartiles + ("baseline_vs_zero.overall",)
    if len(names) != BASELINE_FAMILY_SIZE or len(set(names)) != len(names):
        raise AssertionError("baseline family construction is malformed")
    return names


BASELINE_FAMILY_NAMES = baseline_family_names()


class FalseDiscoveryEvidenceError(ValueError):
    """Base class for typed, fail-closed forensic evidence errors."""

    default_failure_domain = "forensic_evidence"
    default_failure_code = "forensic_evidence_invalid"

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str | None = None,
        failure_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_domain = failure_domain or self.default_failure_domain
        self.failure_code = failure_code or self.default_failure_code

    def to_record(self) -> dict[str, Any]:
        return {
            "error_type": type(self).__name__,
            "failure_domain": self.failure_domain,
            "failure_code": self.failure_code,
            "message": str(self),
        }


class ThreeContrastEvidenceError(FalseDiscoveryEvidenceError):
    default_failure_domain = "sealed_confirmation_replay"
    default_failure_code = "sealed_three_contrast_evidence_invalid"


class MaxTInferenceError(FalseDiscoveryEvidenceError):
    default_failure_domain = "paired_risk_inference"
    default_failure_code = "paired_risk_inference_invalid"


class CandidateAuditError(FalseDiscoveryEvidenceError):
    default_failure_domain = "candidate_selection_audit"
    default_failure_code = "candidate_audit_invalid"


class HistoricalSelectionReplayError(FalseDiscoveryEvidenceError):
    default_failure_domain = "historical_selection_replay"
    default_failure_code = "historical_selection_replay_invalid"


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _integer_vector(value: Any, name: str, rows: int | None = None) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise ThreeContrastEvidenceError(f"{name} must be an integer vector")
    result = np.asarray(source, dtype=np.int64)
    if rows is not None and result.shape != (rows,):
        raise ThreeContrastEvidenceError(f"{name} has the wrong row count")
    return np.ascontiguousarray(result)


def _binary64_vector(value: Any, name: str, rows: int) -> np.ndarray:
    source = np.asarray(value)
    if (
        source.dtype != np.dtype(np.float64)
        or source.shape != (rows,)
        or not np.isfinite(source).all()
    ):
        raise ThreeContrastEvidenceError(
            f"{name} must be a finite binary64 [{rows}] vector"
        )
    return np.ascontiguousarray(source)


def _canonical_paths(value: Any, *, minimum: int = 8) -> np.ndarray:
    paths = _integer_vector(value, "expected_path_ids")
    if (
        paths.size < int(minimum)
        or np.any(paths < 0)
        or np.any(paths >= 1 << 20)
        or np.unique(paths).size != paths.size
    ):
        raise ThreeContrastEvidenceError(
            "expected path IDs must be unique 20-bit integers"
        )
    return np.sort(paths, kind="stable")


def _canonical_steps(value: Sequence[int]) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value)
    if (
        not steps
        or len(set(steps)) != len(steps)
        or tuple(sorted(steps)) != steps
        or any(step < 0 or step >= 512 for step in steps)
    ):
        raise ThreeContrastEvidenceError("selected outer steps are invalid")
    quartiles = np.asarray(steps, dtype=np.int64) // 128
    if not np.array_equal(
        np.unique(quartiles), np.arange(TIME_QUARTILES, dtype=np.int64)
    ):
        raise ThreeContrastEvidenceError(
            "selected outer steps must populate every forward quartile"
        )
    return steps


@dataclass(frozen=True)
class ValidatedThreeContrastRows:
    """Canonical sealed row evidence for the three direct-risk contrasts."""

    sample_keys: np.ndarray
    path_ids: np.ndarray
    outer_steps: np.ndarray
    phases: np.ndarray
    midpoint_indices: np.ndarray
    combined_vs_zero: np.ndarray
    combined_vs_baseline: np.ndarray
    baseline_vs_zero: np.ndarray
    expected_path_ids: np.ndarray
    selected_outer_steps: tuple[int, ...]
    maximum_identity_error: float
    sample_key_sha256: str

    def __post_init__(self) -> None:
        rows = int(np.asarray(self.sample_keys).size)
        integer_names = (
            "sample_keys",
            "path_ids",
            "outer_steps",
            "phases",
            "midpoint_indices",
        )
        for name in integer_names:
            value = np.asarray(getattr(self, name))
            if value.dtype != np.dtype(np.int64) or value.shape != (rows,):
                raise ThreeContrastEvidenceError(
                    f"validated {name} is malformed"
                )
            object.__setattr__(self, name, _readonly(value))
        for name in (
            "combined_vs_zero",
            "combined_vs_baseline",
            "baseline_vs_zero",
        ):
            value = np.asarray(getattr(self, name))
            if (
                value.dtype != np.dtype(np.float64)
                or value.shape != (rows,)
                or not np.isfinite(value).all()
            ):
                raise ThreeContrastEvidenceError(
                    f"validated {name} is malformed"
                )
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "expected_path_ids", _readonly(self.expected_path_ids))

    @property
    def row_count(self) -> int:
        return int(self.sample_keys.size)


def validate_three_contrast_rows(
    *,
    sample_keys: Any,
    row_path_ids: Any,
    outer_steps: Any,
    phases: Any,
    midpoint_indices: Any,
    combined_vs_zero: Any,
    combined_vs_baseline: Any,
    baseline_vs_zero: Any,
    expected_path_ids: Any,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    identity_tolerance: float = THREE_CONTRAST_IDENTITY_TOLERANCE,
) -> ValidatedThreeContrastRows:
    """Validate, sort, and freeze sealed three-contrast confirmation rows."""

    if (
        not math.isfinite(float(identity_tolerance))
        or float(identity_tolerance) < 0.0
        or float(identity_tolerance) > THREE_CONTRAST_IDENTITY_TOLERANCE
    ):
        raise ThreeContrastEvidenceError("three-contrast identity tolerance changed")
    keys = _integer_vector(sample_keys, "sample_keys")
    rows = int(keys.size)
    paths = _integer_vector(row_path_ids, "row_path_ids", rows)
    steps = _integer_vector(outer_steps, "outer_steps", rows)
    phase = _integer_vector(phases, "phases", rows)
    midpoint = _integer_vector(midpoint_indices, "midpoint_indices", rows)
    zero = _binary64_vector(combined_vs_zero, "combined_vs_zero", rows)
    residual = _binary64_vector(
        combined_vs_baseline, "combined_vs_baseline", rows
    )
    baseline = _binary64_vector(baseline_vs_zero, "baseline_vs_zero", rows)
    expected_paths = _canonical_paths(expected_path_ids)
    expected_steps = _canonical_steps(selected_outer_steps)

    expected_rows = (
        expected_paths.size * len(expected_steps) * PHASE_COUNT * MIDPOINT_COUNT
    )
    if rows != expected_rows:
        raise ThreeContrastEvidenceError(
            f"sealed row count {rows} != {expected_rows}"
        )
    if np.unique(keys).size != rows:
        raise ThreeContrastEvidenceError("sample keys are not unique")
    if set(np.unique(paths).tolist()) != set(expected_paths.tolist()):
        raise ThreeContrastEvidenceError("sealed confirmation path set changed")
    if set(np.unique(steps).tolist()) != set(expected_steps):
        raise ThreeContrastEvidenceError("selected outer-step set changed")
    if np.any((phase < 0) | (phase >= PHASE_COUNT)):
        raise ThreeContrastEvidenceError("phase occurrence is invalid")
    if np.any((midpoint < 0) | (midpoint >= MIDPOINT_COUNT)):
        raise ThreeContrastEvidenceError("midpoint index is invalid")
    expected_keys = np.fromiter(
        (
            midpoint_sample_key(int(path), int(step), int(p), int(mid))
            for path, step, p, mid in zip(
                paths, steps, phase, midpoint, strict=True
            )
        ),
        dtype=np.int64,
        count=rows,
    )
    if not np.array_equal(keys, expected_keys):
        raise ThreeContrastEvidenceError(
            "sample keys do not match path/time/phase/midpoint identity"
        )
    identity = np.stack((paths, steps, phase, midpoint), axis=1)
    if np.unique(identity, axis=0).shape[0] != rows:
        raise ThreeContrastEvidenceError("sealed row identity repeats")

    error = np.abs(zero - (baseline + residual))
    maximum_error = float(np.max(error, initial=0.0))
    if maximum_error > float(identity_tolerance):
        raise ThreeContrastEvidenceError(
            "rowwise three-contrast identity exceeds 5e-15",
            failure_code="sealed_three_contrast_identity_invalid",
        )

    digest = hashlib.sha256(
        np.ascontiguousarray(np.sort(keys, kind="stable")).tobytes(order="C")
    ).hexdigest()
    order = np.lexsort((midpoint, phase, steps, paths))
    keys = np.ascontiguousarray(keys[order])
    return ValidatedThreeContrastRows(
        sample_keys=keys,
        path_ids=np.ascontiguousarray(paths[order]),
        outer_steps=np.ascontiguousarray(steps[order]),
        phases=np.ascontiguousarray(phase[order]),
        midpoint_indices=np.ascontiguousarray(midpoint[order]),
        combined_vs_zero=np.ascontiguousarray(zero[order]),
        combined_vs_baseline=np.ascontiguousarray(residual[order]),
        baseline_vs_zero=np.ascontiguousarray(baseline[order]),
        expected_path_ids=np.ascontiguousarray(expected_paths),
        selected_outer_steps=expected_steps,
        maximum_identity_error=maximum_error,
        sample_key_sha256=digest,
    )


@dataclass(frozen=True)
class BaselineRiskTable:
    """Canonical 229-component whole-path baseline-versus-zero table."""

    path_ids: np.ndarray
    path_values: np.ndarray
    cell_counts: np.ndarray
    sample_key_sha256: str

    def __post_init__(self) -> None:
        paths = np.asarray(self.path_ids)
        values = np.asarray(self.path_values)
        counts = np.asarray(self.cell_counts)
        if (
            paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths, kind="stable"))
            or values.dtype != np.dtype(np.float64)
            or values.shape != (paths.size, BASELINE_FAMILY_SIZE)
            or not np.isfinite(values).all()
            or counts.dtype != np.dtype(np.int64)
            or counts.shape != values.shape
            or np.any(counts <= 0)
            or len(self.sample_key_sha256) != 64
        ):
            raise ThreeContrastEvidenceError("baseline path table is malformed")
        object.__setattr__(self, "path_ids", _readonly(paths))
        object.__setattr__(self, "path_values", _readonly(values))
        object.__setattr__(self, "cell_counts", _readonly(counts))

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    def to_record(self) -> dict[str, Any]:
        point = np.mean(self.path_values, axis=0, dtype=np.float64)
        return {
            "schema": SCHEMA + "-baseline-path-table",
            "schema_version": SCHEMA_VERSION,
            "family_names": list(BASELINE_FAMILY_NAMES),
            "family_size": BASELINE_FAMILY_SIZE,
            "path_ids": self.path_ids.tolist(),
            "path_count": self.path_count,
            "sample_key_sha256": self.sample_key_sha256,
            "minimum_cell_count": int(np.min(self.cell_counts)),
            "maximum_cell_count": int(np.max(self.cell_counts)),
            "point_estimates": {
                name: float(value)
                for name, value in zip(BASELINE_FAMILY_NAMES, point, strict=True)
            },
            "posthoc_non_authorizing": 1,
            "old_confirmation_paths_burned": 1,
            "controller_planning_authorized": 0,
        }


@dataclass(frozen=True)
class ThreeContrastRiskTables:
    confirmation: ConfirmationRiskTable
    baseline: BaselineRiskTable
    maximum_identity_error: float


def aggregate_validated_three_contrasts(
    rows: ValidatedThreeContrastRows,
) -> ThreeContrastRiskTables:
    """Aggregate the existing 228 family and omitted 229 baseline family."""

    if not isinstance(rows, ValidatedThreeContrastRows):
        raise ThreeContrastEvidenceError(
            "aggregation requires ValidatedThreeContrastRows"
        )
    confirmation = aggregate_confirmation_improvements(
        sample_keys=rows.sample_keys,
        row_path_ids=rows.path_ids,
        outer_steps=rows.outer_steps,
        phases=rows.phases,
        midpoint_indices=rows.midpoint_indices,
        combined_vs_zero_improvements=rows.combined_vs_zero,
        combined_vs_baseline_improvements=rows.combined_vs_baseline,
        expected_path_ids=rows.expected_path_ids,
        selected_outer_steps=rows.selected_outer_steps,
    )

    paths = rows.expected_path_ids
    path_index = np.searchsorted(paths, rows.path_ids)
    quartile = rows.outer_steps // 128
    fine_column = (
        quartile * (PHASE_COUNT * MIDPOINT_COUNT)
        + rows.phases * MIDPOINT_COUNT
        + rows.midpoint_indices
    )
    quartile_column = COMBINED_VS_ZERO_FAMILY_SIZE + quartile
    overall_column = BASELINE_FAMILY_SIZE - 1
    values = np.zeros((paths.size, BASELINE_FAMILY_SIZE), dtype=np.float64)
    counts = np.zeros_like(values, dtype=np.int64)
    for columns in (fine_column, quartile_column):
        np.add.at(values, (path_index, columns), rows.baseline_vs_zero)
        np.add.at(counts, (path_index, columns), 1)
    np.add.at(values, (path_index, overall_column), rows.baseline_vs_zero)
    np.add.at(counts, (path_index, overall_column), 1)
    if np.any(counts <= 0):
        raise ThreeContrastEvidenceError("a baseline family cell is empty")
    values /= counts
    baseline = BaselineRiskTable(
        path_ids=np.ascontiguousarray(paths),
        path_values=np.ascontiguousarray(values),
        cell_counts=np.ascontiguousarray(counts),
        sample_key_sha256=rows.sample_key_sha256,
    )
    return ThreeContrastRiskTables(
        confirmation=confirmation,
        baseline=baseline,
        maximum_identity_error=float(rows.maximum_identity_error),
    )


def require_exact_confirmation_replay(
    replay: ConfirmationRiskTable,
    *,
    parent_path_ids: Any,
    parent_path_values: Any,
    parent_cell_counts: Any | None = None,
) -> None:
    """Fail closed unless re-aggregated 228 evidence matches the parent."""

    paths = np.asarray(parent_path_ids)
    values = np.asarray(parent_path_values)
    if (
        paths.dtype != np.dtype(np.int64)
        or values.dtype != np.dtype(np.float64)
        or not np.array_equal(replay.path_ids, paths)
        or not np.array_equal(replay.path_values, values)
    ):
        raise ThreeContrastEvidenceError(
            "re-aggregated 228-component evidence differs from parent",
            failure_code="parent_confirmation_reaggregation_mismatch",
        )
    if parent_cell_counts is not None and not np.array_equal(
        replay.cell_counts, np.asarray(parent_cell_counts)
    ):
        raise ThreeContrastEvidenceError(
            "re-aggregated confirmation cell counts differ from parent",
            failure_code="parent_confirmation_reaggregation_mismatch",
        )


def _canonical_inference_table(
    values: Any,
    path_ids: Any,
    names: Sequence[str],
    *,
    expected_size: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    source = np.asarray(values)
    raw_paths = np.asarray(path_ids)
    family_names = tuple(str(name) for name in names)
    if (
        source.dtype != np.dtype(np.float64)
        or source.ndim != 2
        or source.shape[0] < 8
        or source.shape[1] != int(expected_size)
        or not np.isfinite(source).all()
        or raw_paths.ndim != 1
        or raw_paths.shape[0] != source.shape[0]
        or raw_paths.dtype.kind not in "iu"
        or len(family_names) != int(expected_size)
        or len(set(family_names)) != int(expected_size)
    ):
        raise MaxTInferenceError("max-T path family is malformed")
    paths = np.asarray(raw_paths, dtype=np.int64)
    if np.any(paths < 0) or np.unique(paths).size != paths.size:
        raise MaxTInferenceError("max-T path IDs are invalid")
    order = np.argsort(paths, kind="stable")
    return (
        np.ascontiguousarray(paths[order]),
        np.ascontiguousarray(source[order]),
        family_names,
    )


def _validate_bootstrap_configuration(
    *,
    confidence: float,
    replicates: int,
    seed: int,
    namespace: int,
    chunk_size: int,
    component_block_size: int,
) -> None:
    if (
        not 0.5 < float(confidence) < 1.0
        or not isinstance(replicates, int)
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
        or not isinstance(component_block_size, int)
        or isinstance(component_block_size, bool)
        or component_block_size <= 0
    ):
        raise MaxTInferenceError("max-T bootstrap configuration is invalid")


def _studentized_max_t(
    *,
    path_ids: np.ndarray,
    values: np.ndarray,
    names: tuple[str, ...],
    two_sided: bool,
    confidence: float,
    replicates: int,
    seed: int,
    namespace: int,
    chunk_size: int,
    component_block_size: int,
) -> dict[str, Any]:
    _validate_bootstrap_configuration(
        confidence=confidence,
        replicates=replicates,
        seed=seed,
        namespace=namespace,
        chunk_size=chunk_size,
        component_block_size=component_block_size,
    )
    path_count, family_size = values.shape
    point = np.mean(values, axis=0, dtype=np.float64)
    standard_error = np.std(values, axis=0, ddof=1, dtype=np.float64) / math.sqrt(
        path_count
    )
    if (
        not np.isfinite(point).all()
        or not np.isfinite(standard_error).all()
        or np.any(standard_error <= 0.0)
    ):
        raise MaxTInferenceError(
            "max-T family has degenerate/nonfinite studentization",
            failure_code="max_t_studentization_invalid",
        )

    stream_namespace = (
        _BASELINE_BOOTSTRAP_NAMESPACE if two_sided else _SEARCH_BOOTSTRAP_NAMESPACE
    )
    generator = np.random.Generator(
        np.random.Philox([int(seed), int(namespace), int(stream_namespace)])
    )
    maxima = np.full(int(replicates), -np.inf, dtype=np.float64)
    for start in range(0, int(replicates), int(chunk_size)):
        stop = min(int(replicates), start + int(chunk_size))
        indices = generator.integers(
            0,
            path_count,
            size=(stop - start, path_count),
            dtype=np.int64,
        )
        chunk_maximum = np.full(stop - start, -np.inf, dtype=np.float64)
        for left in range(0, family_size, int(component_block_size)):
            right = min(family_size, left + int(component_block_size))
            sampled = values[indices, left:right]
            draw_mean = np.mean(sampled, axis=1, dtype=np.float64)
            draw_error = np.std(sampled, axis=1, ddof=1, dtype=np.float64) / math.sqrt(
                path_count
            )
            if not np.isfinite(draw_error).all() or np.any(draw_error <= 0.0):
                raise MaxTInferenceError(
                    "bootstrap produced degenerate/nonfinite studentization",
                    failure_code="max_t_bootstrap_studentization_invalid",
                )
            statistic = (draw_mean - point[None, left:right]) / draw_error
            if two_sided:
                statistic = np.abs(statistic)
            chunk_maximum = np.maximum(
                chunk_maximum, np.max(statistic, axis=1)
            )
        maxima[start:stop] = chunk_maximum
    if not np.isfinite(maxima).all():
        raise MaxTInferenceError("max-T bootstrap is nonfinite")
    critical = float(np.quantile(maxima, float(confidence), method="higher"))
    if not math.isfinite(critical):
        raise MaxTInferenceError("max-T critical value is nonfinite")
    lower = point - critical * standard_error
    upper = point + critical * standard_error if two_sided else None
    return {
        "schema": SCHEMA + ("-two-sided-max-abs-t" if two_sided else "-one-sided-max-t"),
        "schema_version": SCHEMA_VERSION,
        "method": (
            "centered_whole_path_two_sided_studentized_max_abs_t"
            if two_sided
            else "centered_whole_path_one_sided_studentized_max_t"
        ),
        "bootstrap_unit": "whole_path_jointly_across_family",
        "quantile_method": "higher",
        "family_names": list(names),
        "family_size": family_size,
        "point_estimates": {
            name: float(value) for name, value in zip(names, point, strict=True)
        },
        "standard_errors": {
            name: float(value)
            for name, value in zip(names, standard_error, strict=True)
        },
        "lower_bounds": {
            name: float(value) for name, value in zip(names, lower, strict=True)
        },
        **(
            {
                "upper_bounds": {
                    name: float(value)
                    for name, value in zip(names, upper, strict=True)
                }
            }
            if upper is not None
            else {}
        ),
        "critical_value": critical,
        "path_ids": path_ids.tolist(),
        "path_count": path_count,
        "confidence": float(confidence),
        "bootstrap_replicates": int(replicates),
        "seed": int(seed),
        "namespace": int(namespace),
        "component_block_size": int(component_block_size),
        "negative_values_truncated": 0,
    }


def two_sided_baseline_max_abs_t(
    table: BaselineRiskTable,
    *,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    namespace: int = 0,
    chunk_size: int = 256,
    component_block_size: int = 64,
) -> dict[str, Any]:
    """Two-sided 99.5% whole-path max-|T| interval for 229 components."""

    if not isinstance(table, BaselineRiskTable):
        raise MaxTInferenceError("baseline inference requires BaselineRiskTable")
    paths, values, names = _canonical_inference_table(
        table.path_values,
        table.path_ids,
        BASELINE_FAMILY_NAMES,
        expected_size=BASELINE_FAMILY_SIZE,
    )
    result = _studentized_max_t(
        path_ids=paths,
        values=values,
        names=names,
        two_sided=True,
        confidence=confidence,
        replicates=replicates,
        seed=seed,
        namespace=namespace,
        chunk_size=chunk_size,
        component_block_size=component_block_size,
    )
    result.update(
        {
            "posthoc_non_authorizing": 1,
            "old_confirmation_paths_burned": 1,
            "controller_planning_authorized": 0,
        }
    )
    return result


def classify_sealed_baseline(record: Mapping[str, Any] | None) -> str:
    """Classify the sealed 229-family baseline evidence, fail closed."""

    if not isinstance(record, Mapping):
        return "sealed_baseline_evidence_invalid"
    try:
        names = tuple(record["family_names"])
        lower_map = record["lower_bounds"]
        upper_map = record["upper_bounds"]
        if names != BASELINE_FAMILY_NAMES:
            return "sealed_baseline_evidence_invalid"
        lower = np.asarray([lower_map[name] for name in names], dtype=np.float64)
        upper = np.asarray([upper_map[name] for name in names], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return "sealed_baseline_evidence_invalid"
    if (
        lower.shape != (BASELINE_FAMILY_SIZE,)
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > upper)
    ):
        return "sealed_baseline_evidence_invalid"
    if np.all(lower > 0.0):
        return "sealed_baseline_advantage_confirmed"
    coarse = np.r_[
        np.arange(COMBINED_VS_ZERO_FAMILY_SIZE, BASELINE_FAMILY_SIZE - 1),
        BASELINE_FAMILY_SIZE - 1,
    ]
    if np.all(upper[coarse] < 0.0):
        return "sealed_baseline_harm_confirmed"
    return "sealed_baseline_not_established"


@dataclass(frozen=True)
class CandidateValidationTable:
    """Canonical 120-candidate by path by 228-component validation table."""

    seeds: np.ndarray
    updates: np.ndarray
    path_ids: np.ndarray
    path_values: np.ndarray

    def __post_init__(self) -> None:
        seeds = np.asarray(self.seeds)
        updates = np.asarray(self.updates)
        paths = np.asarray(self.path_ids)
        values = np.asarray(self.path_values)
        candidates = int(seeds.size)
        if (
            seeds.dtype != np.dtype(np.int64)
            or seeds.shape != (candidates,)
            or updates.dtype != np.dtype(np.int64)
            or updates.shape != (candidates,)
            or paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or paths.size < 8
            or np.unique(paths).size != paths.size
            or np.any(paths < 0)
            or values.dtype != np.dtype(np.float64)
            or values.shape != (candidates, paths.size, CONFIRMATION_FAMILY_SIZE)
            or not np.isfinite(values).all()
            or np.unique(np.stack((seeds, updates), axis=1), axis=0).shape[0]
            != candidates
            or np.any(updates <= 0)
        ):
            raise CandidateAuditError("candidate validation table is malformed")
        if candidates != SEARCHED_CANDIDATE_COUNT:
            raise CandidateAuditError(
                f"candidate family has {candidates} entries, expected 120"
            )
        candidate_order = np.lexsort((updates, seeds))
        path_order = np.argsort(paths, kind="stable")
        object.__setattr__(self, "seeds", _readonly(seeds[candidate_order]))
        object.__setattr__(self, "updates", _readonly(updates[candidate_order]))
        object.__setattr__(self, "path_ids", _readonly(paths[path_order]))
        object.__setattr__(
            self,
            "path_values",
            _readonly(values[candidate_order][:, path_order, :]),
        )

    @property
    def candidate_count(self) -> int:
        return int(self.seeds.size)

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    def candidate_index(self, seed: int, update: int) -> int:
        matches = np.flatnonzero(
            (self.seeds == int(seed)) & (self.updates == int(update))
        )
        if matches.size != 1:
            raise CandidateAuditError("selected candidate is absent or duplicated")
        return int(matches[0])


def build_candidate_validation_table(
    *,
    seeds: Any,
    updates: Any,
    path_ids: Any,
    path_values: Any,
    forbidden_path_ids: Any | None = None,
) -> CandidateValidationTable:
    """Build a canonical candidate table and enforce the confirmation firewall."""

    seed_array = np.asarray(seeds)
    update_array = np.asarray(updates)
    path_array = np.asarray(path_ids)
    value_array = np.asarray(path_values)
    if seed_array.dtype.kind not in "iu" or update_array.dtype.kind not in "iu":
        raise CandidateAuditError("candidate seeds and updates must be integers")
    if path_array.dtype.kind not in "iu":
        raise CandidateAuditError("candidate path IDs must be integers")
    canonical_paths = np.asarray(path_array, dtype=np.int64)
    if forbidden_path_ids is not None:
        forbidden = np.asarray(forbidden_path_ids)
        if forbidden.ndim != 1 or forbidden.dtype.kind not in "iu":
            raise CandidateAuditError("forbidden path IDs are malformed")
        if np.intersect1d(canonical_paths, np.asarray(forbidden, dtype=np.int64)).size:
            raise CandidateAuditError(
                "confirmation path IDs entered candidate evaluation",
                failure_code="confirmation_path_firewall_violated",
            )
    return CandidateValidationTable(
        seeds=np.asarray(seed_array, dtype=np.int64),
        updates=np.asarray(update_array, dtype=np.int64),
        path_ids=canonical_paths,
        path_values=value_array,
    )


def candidate_directional_screens(
    table: CandidateValidationTable,
) -> list[dict[str, Any]]:
    """Return nonauthorizing exact-family point-direction screens."""

    if not isinstance(table, CandidateValidationTable):
        raise CandidateAuditError("directional screens require candidate table")
    point = np.mean(table.path_values, axis=1, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index in range(table.candidate_count):
        fine = point[index, :COMBINED_VS_ZERO_FAMILY_SIZE]
        residual = point[index, COMBINED_VS_ZERO_FAMILY_SIZE:]
        rows.append(
            {
                "seed": int(table.seeds[index]),
                "update": int(table.updates[index]),
                "all_224_combined_vs_zero_point_estimates_positive": int(
                    np.all(fine > 0.0)
                ),
                "all_4_combined_vs_baseline_point_estimates_positive": int(
                    np.all(residual > 0.0)
                ),
                "all_228_point_estimates_positive": int(
                    np.all(point[index] > 0.0)
                ),
                "positive_combined_vs_zero_count": int(np.count_nonzero(fine > 0.0)),
                "positive_combined_vs_baseline_count": int(
                    np.count_nonzero(residual > 0.0)
                ),
                "minimum_combined_vs_zero_point_estimate": float(np.min(fine)),
                "minimum_combined_vs_baseline_point_estimate": float(
                    np.min(residual)
                ),
            }
        )
    return rows


def search_aware_candidate_max_t(
    table: CandidateValidationTable,
    *,
    confidence: float = DEFAULT_BOOTSTRAP_CONFIDENCE,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    namespace: int = 1,
    chunk_size: int = 256,
    component_block_size: int = 64,
) -> dict[str, Any]:
    """One-sided max-T over 120 candidates by four residual quartiles."""

    if not isinstance(table, CandidateValidationTable):
        raise CandidateAuditError("search inference requires candidate table")
    residual = table.path_values[:, :, COMBINED_VS_ZERO_FAMILY_SIZE:]
    # [candidate,path,quartile] -> [path,candidate*quartile]
    values = np.ascontiguousarray(
        residual.transpose(1, 0, 2).reshape(table.path_count, -1)
    )
    names = tuple(
        f"seed{int(seed_value)}.update{int(update):04d}.combined_vs_baseline.q{q}"
        for seed_value, update in zip(table.seeds, table.updates, strict=True)
        for q in range(TIME_QUARTILES)
    )
    if len(names) != SEARCHED_RESIDUAL_FAMILY_SIZE:
        raise CandidateAuditError("searched residual family size changed")
    result = _studentized_max_t(
        path_ids=table.path_ids,
        values=values,
        names=names,
        two_sided=False,
        confidence=confidence,
        replicates=replicates,
        seed=seed,
        namespace=namespace,
        chunk_size=chunk_size,
        component_block_size=component_block_size,
    )
    lower = np.asarray([result["lower_bounds"][name] for name in names])
    lower = lower.reshape(table.candidate_count, TIME_QUARTILES)
    screens = candidate_directional_screens(table)
    candidate_rows: list[dict[str, Any]] = []
    for index, screen in enumerate(screens):
        candidate_rows.append(
            {
                **screen,
                "residual_lower_bounds": lower[index].tolist(),
                "selection_resolved_residual_signal": int(
                    np.all(lower[index] > 0.0)
                ),
                "selection_resolved_and_directionally_compatible": int(
                    np.all(lower[index] > 0.0)
                    and int(screen["all_228_point_estimates_positive"]) == 1
                ),
            }
        )
    selected_index = table.candidate_index(
        HISTORICAL_SELECTED_SEED, HISTORICAL_SELECTED_UPDATE
    )
    resolved_count = int(
        sum(row["selection_resolved_residual_signal"] for row in candidate_rows)
    )
    compatible_count = int(
        sum(row["all_228_point_estimates_positive"] for row in candidate_rows)
    )
    qualified_count = int(
        sum(
            row["selection_resolved_and_directionally_compatible"]
            for row in candidate_rows
        )
    )
    result.update(
        {
            "candidate_count": table.candidate_count,
            "candidate_rows": candidate_rows,
            "selection_resolved_candidate_count": resolved_count,
            "directionally_compatible_candidate_count": compatible_count,
            "fully_qualified_candidate_count": qualified_count,
            # Gate-facing aliases make the closed decision schema compact
            # without discarding the more descriptive forensic field names.
            "residual_resolved_candidate_count": resolved_count,
            "direction_compatible_candidate_count": compatible_count,
            "qualifying_candidate_count": qualified_count,
            "selected_seed": HISTORICAL_SELECTED_SEED,
            "selected_update": HISTORICAL_SELECTED_UPDATE,
            "selected_update_residual_resolved": int(
                candidate_rows[selected_index][
                    "selection_resolved_residual_signal"
                ]
            ),
            "selected_update_directionally_compatible": int(
                candidate_rows[selected_index]["all_228_point_estimates_positive"]
            ),
            "controller_planning_authorized": 0,
        }
    )
    return result


def classify_candidate_audit(
    record: Mapping[str, Any],
    *,
    selected_seed: int = HISTORICAL_SELECTED_SEED,
    selected_update: int = HISTORICAL_SELECTED_UPDATE,
) -> str:
    """Apply the closed candidate-audit classification with fixed precedence."""

    try:
        rows = [dict(row) for row in record["candidate_rows"]]
        selected = next(
            row
            for row in rows
            if int(row["seed"]) == int(selected_seed)
            and int(row["update"]) == int(selected_update)
        )
    except (KeyError, TypeError, ValueError, StopIteration):
        return "implementation_or_replay_defect"
    if len(rows) != SEARCHED_CANDIDATE_COUNT:
        return "implementation_or_replay_defect"
    if any(int(row.get("selection_resolved_and_directionally_compatible", 0)) == 1 for row in rows):
        return "current_candidate_family_residual_signal_resolved"
    if any(int(row.get("selection_resolved_residual_signal", 0)) == 1 for row in rows):
        return "residual_signal_directionally_incompatible_with_zero"
    if int(selected.get("selection_resolved_residual_signal", 0)) == 0:
        return "selected_update_below_resolution"
    return "selection_audit_inconclusive"


def _finite_number(record: Mapping[str, Any], name: str) -> float:
    try:
        value = float(record[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise HistoricalSelectionReplayError(
            f"candidate field {name!r} is missing or invalid"
        ) from exc
    if not math.isfinite(value):
        raise HistoricalSelectionReplayError(
            f"candidate field {name!r} is nonfinite"
        )
    return value


def _sha256_field(record: Mapping[str, Any], name: str) -> str:
    value = str(record.get(name, ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HistoricalSelectionReplayError(f"candidate {name} is malformed")
    return value


def replay_historical_selection(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int] = HISTORICAL_SEEDS,
    expected_updates: Sequence[int] = HISTORICAL_UPDATES,
    strict_artifacts: bool = True,
    metric_tolerance: float = THREE_CONTRAST_IDENTITY_TOLERANCE,
) -> dict[str, Any]:
    """Replay the v2 point-selection rule from all 3 x 41 records.

    The historical rule first selects within each seed from nonzero candidates
    beating the baseline overall and at high reverse time (falling back to
    update zero), then selects across seed winners by validation MSE, update,
    and seed.  Stored ``eligible_nonzero`` flags are checked but never trusted.
    """

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise HistoricalSelectionReplayError("candidate records must be a sequence")
    seeds = tuple(int(seed) for seed in expected_seeds)
    updates = tuple(int(update) for update in expected_updates)
    if len(set(seeds)) != len(seeds) or len(set(updates)) != len(updates):
        raise HistoricalSelectionReplayError("expected candidate grid is malformed")
    expected = {(seed, update) for seed in seeds for update in updates}
    canonical: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise HistoricalSelectionReplayError("candidate record is not a mapping")
        try:
            seed = int(raw["seed"])
            update = int(raw["update"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HistoricalSelectionReplayError(
                "candidate seed/update is invalid"
            ) from exc
        key = (seed, update)
        if key not in expected or key in canonical:
            raise HistoricalSelectionReplayError(
                "candidate grid contains an extra or duplicate record"
            )
        finite = int(raw.get("finite", -1))
        if finite != 1:
            raise HistoricalSelectionReplayError("candidate is not complete and finite")
        validation = _finite_number(raw, "validation_mse")
        high = _finite_number(raw, "validation_high_reverse_time_mse")
        baseline = _finite_number(raw, "baseline_validation_mse")
        baseline_high = _finite_number(raw, "baseline_high_reverse_time_mse")
        zero = _finite_number(raw, "zero_validation_mse")
        zero_high = _finite_number(raw, "zero_high_reverse_time_mse")
        residual = baseline - validation
        residual_high = baseline_high - high
        versus_zero = zero - validation
        versus_zero_high = zero_high - high
        for name, derived in (
            ("combined_vs_baseline", residual),
            ("combined_vs_baseline_high_reverse_time", residual_high),
            ("combined_vs_zero", versus_zero),
            ("combined_vs_zero_high_reverse_time", versus_zero_high),
        ):
            stored = _finite_number(raw, name)
            if abs(stored - derived) > float(metric_tolerance):
                raise HistoricalSelectionReplayError(
                    f"candidate derived metric {name} does not replay"
                )
        eligible = int(update > 0 and validation < baseline and high < baseline_high)
        if int(raw.get("eligible_nonzero", -1)) != eligible:
            raise HistoricalSelectionReplayError(
                "stored candidate eligibility does not replay"
            )
        if strict_artifacts:
            _sha256_field(raw, "state_sha256")
            _sha256_field(raw, "checkpoint_file_sha256")
            path = str(raw.get("checkpoint_path", ""))
            if not path or not path.endswith(f"update-{update:04d}.pt"):
                raise HistoricalSelectionReplayError(
                    "candidate checkpoint path is malformed"
                )
        canonical[key] = {
            **dict(raw),
            "seed": seed,
            "update": update,
            "eligible_nonzero_replayed": eligible,
        }
    if set(canonical) != expected:
        missing = sorted(expected - set(canonical))
        raise HistoricalSelectionReplayError(
            f"candidate grid is incomplete; missing {missing[:3]}"
        )

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        candidates = [canonical[(seed, update)] for update in updates]
        eligible = [row for row in candidates if row["eligible_nonzero_replayed"] == 1]
        if not eligible:
            eligible = [row for row in candidates if int(row["update"]) == 0]
        selected = min(
            eligible,
            key=lambda row: (
                float(row["validation_mse"]),
                int(row["update"]),
                int(row["seed"]),
            ),
        )
        per_seed.append(selected)
    nonzero = [
        row
        for row in per_seed
        if int(row["update"]) > 0
        and int(row["eligible_nonzero_replayed"]) == 1
    ]
    selected = min(
        nonzero if nonzero else per_seed,
        key=lambda row: (
            float(row["validation_mse"]),
            int(row["update"]),
            int(row["seed"]),
        ),
    )
    digest_rows = [
        {
            key: row[key]
            for key in sorted(row)
            if isinstance(row[key], (str, int, float, bool)) or row[key] is None
        }
        for row in (canonical[key] for key in sorted(canonical))
    ]
    digest = hashlib.sha256(
        json.dumps(
            digest_rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": SCHEMA + "-historical-selection-replay",
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(canonical),
        "candidate_grid_sha256": digest,
        "per_seed_selected": [
            {
                "seed": int(row["seed"]),
                "update": int(row["update"]),
                "validation_mse": float(row["validation_mse"]),
            }
            for row in per_seed
        ],
        "selected_seed": int(selected["seed"]),
        "selected_update": int(selected["update"]),
        "selected_state_sha256": str(selected.get("state_sha256", "")),
        "historical_selection_reproduced": int(
            int(selected["seed"]) == HISTORICAL_SELECTED_SEED
            and int(selected["update"]) == HISTORICAL_SELECTED_UPDATE
        ),
        "controller_planning_authorized": 0,
    }


def corrected_point_candidate_eligible(record: Mapping[str, Any]) -> bool:
    """Necessary pointwise v3 screen: beat baseline and zero in both scopes."""

    try:
        return bool(
            int(record["update"]) > 0
            and int(record["finite"]) == 1
            and float(record["combined_vs_baseline"]) > 0.0
            and float(record["combined_vs_baseline_high_reverse_time"]) > 0.0
            and float(record["combined_vs_zero"]) > 0.0
            and float(record["combined_vs_zero_high_reverse_time"]) > 0.0
        )
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "BASELINE_FAMILY_NAMES",
    "BASELINE_FAMILY_SIZE",
    "CandidateAuditError",
    "CandidateValidationTable",
    "DEFAULT_BOOTSTRAP_CONFIDENCE",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "FalseDiscoveryEvidenceError",
    "HISTORICAL_NONZERO_UPDATES",
    "HISTORICAL_SEEDS",
    "HISTORICAL_SELECTED_SEED",
    "HISTORICAL_SELECTED_UPDATE",
    "HISTORICAL_UPDATES",
    "HistoricalSelectionReplayError",
    "MaxTInferenceError",
    "SCHEMA",
    "SEARCHED_CANDIDATE_COUNT",
    "SEARCHED_RESIDUAL_FAMILY_SIZE",
    "THREE_CONTRAST_IDENTITY_TOLERANCE",
    "ThreeContrastEvidenceError",
    "ThreeContrastRiskTables",
    "ValidatedThreeContrastRows",
    "aggregate_validated_three_contrasts",
    "baseline_family_names",
    "build_candidate_validation_table",
    "candidate_directional_screens",
    "classify_candidate_audit",
    "classify_sealed_baseline",
    "corrected_point_candidate_eligible",
    "replay_historical_selection",
    "require_exact_confirmation_replay",
    "search_aware_candidate_max_t",
    "two_sided_baseline_max_abs_t",
    "validate_three_contrast_rows",
]
