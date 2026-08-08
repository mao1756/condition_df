"""Read-only analysis core for the sealed zero-baseline v3 validation run.

This module deliberately contains no cache generation, optimization, confirmation,
or sampler code.  It validates the committed candidate/path table, reconstructs the
search-aware max-T result, exposes descriptive resolution pooling, and provides the
quadratic-risk arithmetic used by the time-local adjudication workflow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mnist.d0_jacobi_rb_boundary_tangent_v3_selection import (
    DEFAULT_CONFIDENCE,
    V3_CANDIDATE_COUNT,
    V3_COMPONENT_COUNT,
    V3_FAMILY_NAMES,
    V3_MODEL_SEEDS,
    CandidateValidationTableV3,
    NumericMaxTResult,
    V3SelectionError,
    aggregate_zero_baseline_improvements,
    build_candidate_validation_table_v3,
    rank_validation_nominee,
)


SCHEMA = "d0-jacobi-rb-boundary-tangent-v3-time-local-v1"
FINE_COMPONENT_COUNT = 224
TIME_QUARTILES = 4
PHASE_COUNT = 7
MIDPOINT_COUNT = 8
Q0_POOLED_COMPONENT = 224
Q1_PHASE4_MIDPOINT7_COMPONENT = 56 + 4 * MIDPOINT_COUNT + 7
CONFIRMATION_PATH_IDS = tuple(range(0xF2000, 0xF2040))

PRODUCTION_CRITICAL_VALUE = 7.1588810358178305
PRODUCTION_DISCOVERED_COMPONENTS = (6, 7, 15, 224)
PRODUCTION_POSITIVE_COMPONENT_COUNT = 28
PRODUCTION_DISCOVERING_CANDIDATE_COUNT = 24
PRODUCTION_Q0_NOMINEES = (
    (261_312, 900, 0.0007141942787250211, 0.0003539919961197451, 55),
    (261_313, 1_600, 0.00133972226314771, 0.000663622214106495, 55),
    (261_314, 3_900, 0.0006320899845075137, 0.00023917961459697846, 54),
)


class TimeLocalAdjudicationError(ValueError):
    """The immutable validation evidence or adjudication arithmetic is invalid."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str = "sealed_selection_replay_invalid",
    ) -> None:
        super().__init__(message)
        self.failure_code = str(failure_code)


def _readonly(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _float_equal(left: float, right: float, *, tolerance: float = 5e-15) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


@dataclass(frozen=True)
class Q0Nominee:
    seed: int
    update: int
    candidate_index: int
    point_estimate: float
    adjusted_lower_bound: float
    positive_fine_cell_count: int

    def to_record(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "update": int(self.update),
            "candidate_index": int(self.candidate_index),
            "pooled_q0_point_estimate": float(self.point_estimate),
            "pooled_q0_adjusted_lower_bound": float(self.adjusted_lower_bound),
            "positive_q0_fine_cell_count": int(self.positive_fine_cell_count),
            "q0_fine_cell_count": PHASE_COUNT * MIDPOINT_COUNT,
        }


def recover_q0_nominees(
    table: CandidateValidationTableV3,
    result: NumericMaxTResult,
) -> tuple[Q0Nominee, ...]:
    """Recover one frozen q0 nominee per seed using adjusted pooled-q0 evidence."""

    if not isinstance(table, CandidateValidationTableV3) or not isinstance(
        result, NumericMaxTResult
    ):
        raise TimeLocalAdjudicationError("q0 nominee recovery requires canonical evidence")
    if (
        result.candidate_count != table.candidate_count
        or not np.array_equal(result.path_ids, table.path_ids)
    ):
        raise TimeLocalAdjudicationError("q0 nominee evidence does not match the table")
    nominees: list[Q0Nominee] = []
    for seed in V3_MODEL_SEEDS:
        indices = np.flatnonzero(table.seeds == int(seed))
        if indices.size == 0:
            raise TimeLocalAdjudicationError(f"model seed {seed} is absent")
        best_lower = float(np.max(result.lower_bounds[indices, Q0_POOLED_COMPONENT]))
        tied = indices[
            result.lower_bounds[indices, Q0_POOLED_COMPONENT] == best_lower
        ]
        selected = min(
            tied.tolist(),
            key=lambda index: (int(table.updates[index]), int(index)),
        )
        nominees.append(
            Q0Nominee(
                seed=int(seed),
                update=int(table.updates[selected]),
                candidate_index=int(selected),
                point_estimate=float(
                    result.point_estimates[selected, Q0_POOLED_COMPONENT]
                ),
                adjusted_lower_bound=float(best_lower),
                positive_fine_cell_count=int(
                    np.count_nonzero(result.point_estimates[selected, :56] > 0.0)
                ),
            )
        )
    return tuple(nominees)


@dataclass(frozen=True)
class SealedSelectionReplay:
    table: CandidateValidationTableV3
    result: NumericMaxTResult
    nominees: tuple[Q0Nominee, ...]
    positive_component_count: int
    discovering_candidate_count: int
    discovered_component_indices: tuple[int, ...]
    candidate_with_all_point_estimates_positive: bool
    q1_phase4_midpoint7_maximum_point_estimate: float

    def to_record(self) -> dict[str, Any]:
        ranking = rank_validation_nominee(self.table, self.result)
        return {
            "schema": SCHEMA + "-sealed-selection-replay",
            "schema_version": 1,
            "table_shape": list(self.table.path_values.shape),
            "path_ids": self.table.path_ids.tolist(),
            "candidate_count": int(self.table.candidate_count),
            "component_count": V3_COMPONENT_COUNT,
            "critical_value": float(self.result.critical_value),
            "confidence": float(self.result.confidence),
            "replicates": int(self.result.maxima.size),
            "positive_simultaneous_candidate_components": int(
                self.positive_component_count
            ),
            "candidates_with_positive_simultaneous_component": int(
                self.discovering_candidate_count
            ),
            "discovered_component_indices": list(self.discovered_component_indices),
            "discovered_component_names": [
                V3_FAMILY_NAMES[index] for index in self.discovered_component_indices
            ],
            "candidate_with_all_228_point_estimates_positive": int(
                self.candidate_with_all_point_estimates_positive
            ),
            "q1_phase4_midpoint7_maximum_point_estimate": float(
                self.q1_phase4_midpoint7_maximum_point_estimate
            ),
            "eligible_candidate_count": int(ranking["eligible_candidate_count"]),
            "logical_update_zero_selected": int(
                ranking["logical_update_zero_selected"]
            ),
            "q0_nominees": [nominee.to_record() for nominee in self.nominees],
            "confirmation_authorized": 0,
            "authorizing_reinterpretation_performed": 0,
        }


def replay_sealed_selection(
    table: CandidateValidationTableV3,
    result: NumericMaxTResult,
    *,
    require_production_fixture: bool = False,
) -> SealedSelectionReplay:
    """Validate and summarize the already-committed v3 validation search."""

    if not isinstance(table, CandidateValidationTableV3) or not isinstance(
        result, NumericMaxTResult
    ):
        raise TimeLocalAdjudicationError("sealed replay requires canonical evidence")
    if (
        table.path_values.shape != (32, 120, 228)
        or result.point_estimates.shape != (120, 228)
        or result.standard_errors.shape != (120, 228)
        or result.lower_bounds.shape != (120, 228)
        or not np.array_equal(result.path_ids, table.path_ids)
    ):
        raise TimeLocalAdjudicationError("sealed validation shapes or paths changed")
    if np.intersect1d(table.path_ids, np.asarray(CONFIRMATION_PATH_IDS)).size:
        raise TimeLocalAdjudicationError(
            "confirmation paths entered sealed validation evidence",
            failure_code="confirmation_path_firewall_violated",
        )

    point = np.mean(table.path_values, axis=0, dtype=np.float64)
    standard_error = np.std(
        table.path_values, axis=0, ddof=1, dtype=np.float64
    ) / math.sqrt(table.path_count)
    critical = float(
        np.quantile(result.maxima, float(result.confidence), method="higher")
    )
    lower = point - critical * standard_error
    if (
        not np.array_equal(point, result.point_estimates)
        or not np.array_equal(standard_error, result.standard_errors)
        or not np.array_equal(lower, result.lower_bounds)
        or critical != float(result.critical_value)
    ):
        raise TimeLocalAdjudicationError("stored max-T arrays do not replay exactly")

    ranking = rank_validation_nominee(table, result)
    if (
        ranking["decision"] != "no_validation_candidate"
        or int(ranking["eligible_candidate_count"]) != 0
        or int(ranking["logical_update_zero_selected"]) != 1
    ):
        raise TimeLocalAdjudicationError("sealed no-candidate decision did not replay")

    positive = result.lower_bounds > 0.0
    discovered = tuple(int(value) for value in np.flatnonzero(np.any(positive, axis=0)))
    nominees = recover_q0_nominees(table, result)
    replay = SealedSelectionReplay(
        table=table,
        result=result,
        nominees=nominees,
        positive_component_count=int(np.count_nonzero(positive)),
        discovering_candidate_count=int(np.count_nonzero(np.any(positive, axis=1))),
        discovered_component_indices=discovered,
        candidate_with_all_point_estimates_positive=bool(
            np.any(np.all(result.point_estimates > 0.0, axis=1))
        ),
        q1_phase4_midpoint7_maximum_point_estimate=float(
            np.max(result.point_estimates[:, Q1_PHASE4_MIDPOINT7_COMPONENT])
        ),
    )
    if require_production_fixture:
        _require_production_replay(replay)
    return replay


def build_sealed_selection_replay(
    *,
    seeds: Any,
    updates: Any,
    path_ids: Any,
    path_values: Any,
    point_estimates: Any,
    standard_errors: Any,
    lower_bounds: Any,
    maxima: Any,
    confidence: float = DEFAULT_CONFIDENCE,
    require_production_fixture: bool = False,
) -> SealedSelectionReplay:
    """Construct canonical evidence from the two committed NPZ payloads."""

    try:
        table = build_candidate_validation_table_v3(
            seeds=seeds,
            updates=updates,
            path_ids=path_ids,
            path_values=path_values,
            forbidden_path_ids=np.asarray(CONFIRMATION_PATH_IDS, dtype=np.int64),
        )
    except V3SelectionError as exc:
        raise TimeLocalAdjudicationError(
            str(exc), failure_code=str(exc.failure_code)
        ) from exc
    maxima_array = np.asarray(maxima)
    if maxima_array.dtype != np.dtype(np.float64) or maxima_array.ndim != 1:
        raise TimeLocalAdjudicationError("stored bootstrap maxima are malformed")
    critical = float(np.quantile(maxima_array, float(confidence), method="higher"))
    result = NumericMaxTResult(
        path_ids=table.path_ids,
        point_estimates=np.asarray(point_estimates),
        standard_errors=np.asarray(standard_errors),
        lower_bounds=np.asarray(lower_bounds),
        maxima=maxima_array,
        critical_value=critical,
        confidence=float(confidence),
    )
    return replay_sealed_selection(
        table,
        result,
        require_production_fixture=require_production_fixture,
    )


def _require_production_replay(replay: SealedSelectionReplay) -> None:
    if not _float_equal(replay.result.critical_value, PRODUCTION_CRITICAL_VALUE):
        raise TimeLocalAdjudicationError("production critical value changed")
    if (
        replay.positive_component_count != PRODUCTION_POSITIVE_COMPONENT_COUNT
        or replay.discovering_candidate_count
        != PRODUCTION_DISCOVERING_CANDIDATE_COUNT
        or replay.discovered_component_indices != PRODUCTION_DISCOVERED_COMPONENTS
        or replay.candidate_with_all_point_estimates_positive
        or replay.q1_phase4_midpoint7_maximum_point_estimate > 0.0
    ):
        raise TimeLocalAdjudicationError("production partial-discovery census changed")
    observed = tuple(
        (
            nominee.seed,
            nominee.update,
            nominee.point_estimate,
            nominee.adjusted_lower_bound,
            nominee.positive_fine_cell_count,
        )
        for nominee in replay.nominees
    )
    for actual, expected in zip(observed, PRODUCTION_Q0_NOMINEES, strict=True):
        if (
            actual[:2] != expected[:2]
            or not _float_equal(actual[2], expected[2])
            or not _float_equal(actual[3], expected[3])
            or actual[4] != expected[4]
        ):
            raise TimeLocalAdjudicationError("production q0 nominees changed")


@dataclass(frozen=True)
class ResolutionLevel:
    level: str
    names: tuple[str, ...]
    path_values: np.ndarray
    point_estimates: np.ndarray
    standard_errors: np.ndarray
    descriptive_lower_bounds: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.path_values)
        point = np.asarray(self.point_estimates)
        error = np.asarray(self.standard_errors)
        lower = np.asarray(self.descriptive_lower_bounds)
        if (
            values.dtype != np.dtype(np.float64)
            or values.ndim != 3
            or values.shape[2] != len(self.names)
            or point.shape != values.shape[1:]
            or error.shape != point.shape
            or lower.shape != point.shape
            or not np.isfinite(values).all()
            or not np.isfinite(point).all()
            or not np.isfinite(error).all()
            or not np.isfinite(lower).all()
        ):
            raise TimeLocalAdjudicationError("resolution-ladder level is malformed")
        object.__setattr__(self, "path_values", _readonly(values))
        object.__setattr__(self, "point_estimates", _readonly(point))
        object.__setattr__(self, "standard_errors", _readonly(error))
        object.__setattr__(self, "descriptive_lower_bounds", _readonly(lower))


def _resolution_level(
    level: str,
    names: Sequence[str],
    values: np.ndarray,
    critical_value: float,
) -> ResolutionLevel:
    point = np.mean(values, axis=0, dtype=np.float64)
    error = np.std(values, axis=0, ddof=1, dtype=np.float64) / math.sqrt(
        values.shape[0]
    )
    lower = point - float(critical_value) * error
    return ResolutionLevel(
        level=str(level),
        names=tuple(str(name) for name in names),
        path_values=np.ascontiguousarray(values),
        point_estimates=np.ascontiguousarray(point),
        standard_errors=np.ascontiguousarray(error),
        descriptive_lower_bounds=np.ascontiguousarray(lower),
    )


def build_resolution_ladder(
    table: CandidateValidationTableV3,
    *,
    critical_value: float,
    pooled_identity_tolerance: float = 5e-15,
) -> tuple[ResolutionLevel, ...]:
    """Pool the sealed fine risks at preregistered descriptive resolutions."""

    if not isinstance(table, CandidateValidationTableV3):
        raise TimeLocalAdjudicationError("resolution ladder requires canonical evidence")
    if not math.isfinite(float(critical_value)) or float(critical_value) < 0.0:
        raise TimeLocalAdjudicationError("resolution critical value is invalid")
    fine = np.ascontiguousarray(
        table.path_values[:, :, :FINE_COMPONENT_COUNT].reshape(
            table.path_count,
            table.candidate_count,
            TIME_QUARTILES,
            PHASE_COUNT,
            MIDPOINT_COUNT,
        )
    )
    derived_quartile = np.mean(fine, axis=(3, 4), dtype=np.float64)
    stored_quartile = table.path_values[:, :, FINE_COMPONENT_COUNT:]
    if float(np.max(np.abs(derived_quartile - stored_quartile))) > float(
        pooled_identity_tolerance
    ):
        raise TimeLocalAdjudicationError("stored quartile pools do not match fine cells")

    values_and_names: tuple[tuple[str, np.ndarray, tuple[str, ...]], ...] = (
        (
            "quartile_phase_midpoint",
            fine.reshape(table.path_count, table.candidate_count, -1),
            tuple(
                f"q{q}.phase{phase}.midpoint{midpoint}"
                for q in range(TIME_QUARTILES)
                for phase in range(PHASE_COUNT)
                for midpoint in range(MIDPOINT_COUNT)
            ),
        ),
        (
            "quartile_phase",
            np.mean(fine, axis=4, dtype=np.float64).reshape(
                table.path_count, table.candidate_count, -1
            ),
            tuple(
                f"q{q}.phase{phase}"
                for q in range(TIME_QUARTILES)
                for phase in range(PHASE_COUNT)
            ),
        ),
        (
            "quartile_midpoint",
            np.mean(fine, axis=3, dtype=np.float64).reshape(
                table.path_count, table.candidate_count, -1
            ),
            tuple(
                f"q{q}.midpoint{midpoint}"
                for q in range(TIME_QUARTILES)
                for midpoint in range(MIDPOINT_COUNT)
            ),
        ),
        (
            "quartile",
            derived_quartile,
            tuple(f"q{q}" for q in range(TIME_QUARTILES)),
        ),
        (
            "phase",
            np.mean(fine, axis=(2, 4), dtype=np.float64),
            tuple(f"phase{phase}" for phase in range(PHASE_COUNT)),
        ),
        (
            "midpoint",
            np.mean(fine, axis=(2, 3), dtype=np.float64),
            tuple(f"midpoint{midpoint}" for midpoint in range(MIDPOINT_COUNT)),
        ),
        (
            "overall",
            np.mean(fine, axis=(2, 3, 4), dtype=np.float64)[:, :, None],
            ("overall",),
        ),
    )
    return tuple(
        _resolution_level(level, names, np.ascontiguousarray(values), critical_value)
        for level, values, names in values_and_names
    )


@dataclass(frozen=True)
class QuadraticRiskDecomposition:
    path_ids: np.ndarray
    candidate_labels: tuple[str, ...]
    cross_terms: np.ndarray
    prediction_energies: np.ndarray
    direct_improvements: np.ndarray
    reconstructed_improvements: np.ndarray
    maximum_identity_error: float

    def __post_init__(self) -> None:
        paths = np.asarray(self.path_ids)
        arrays = tuple(
            np.asarray(value)
            for value in (
                self.cross_terms,
                self.prediction_energies,
                self.direct_improvements,
                self.reconstructed_improvements,
            )
        )
        expected = (paths.size, len(self.candidate_labels), V3_COMPONENT_COUNT)
        if (
            paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or np.unique(paths).size != paths.size
            or not self.candidate_labels
            or any(value.dtype != np.dtype(np.float64) for value in arrays)
            or any(value.shape != expected for value in arrays)
            or any(not np.isfinite(value).all() for value in arrays)
            or not math.isfinite(float(self.maximum_identity_error))
        ):
            raise TimeLocalAdjudicationError(
                "quadratic decomposition is malformed",
                failure_code="quadratic_risk_decomposition_invalid",
            )
        object.__setattr__(self, "path_ids", _readonly(paths))
        object.__setattr__(self, "cross_terms", _readonly(arrays[0]))
        object.__setattr__(self, "prediction_energies", _readonly(arrays[1]))
        object.__setattr__(self, "direct_improvements", _readonly(arrays[2]))
        object.__setattr__(self, "reconstructed_improvements", _readonly(arrays[3]))


def aggregate_quadratic_risk_decomposition(
    *,
    sample_keys: Any,
    row_path_ids: Any,
    outer_steps: Any,
    phases: Any,
    midpoint_indices: Any,
    targets: Any,
    predictions: Any,
    expected_path_ids: Any,
    candidate_labels: Sequence[str],
    identity_tolerance: float = 5e-15,
) -> QuadraticRiskDecomposition:
    """Aggregate ``I = 2C - P`` without changing the raw RB target."""

    target = np.asarray(targets)
    prediction = np.asarray(predictions)
    labels = tuple(str(value) for value in candidate_labels)
    if target.dtype != np.dtype(np.float64) or target.ndim != 2:
        raise TimeLocalAdjudicationError(
            "targets must be a binary64 row/edge table",
            failure_code="quadratic_risk_decomposition_invalid",
        )
    if prediction.ndim == 2:
        prediction = prediction[:, None, :]
    if (
        prediction.dtype != np.dtype(np.float64)
        or prediction.ndim != 3
        or prediction.shape[0] != target.shape[0]
        or prediction.shape[2] != target.shape[1]
        or prediction.shape[1] != len(labels)
        or len(set(labels)) != len(labels)
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        raise TimeLocalAdjudicationError(
            "predictions or candidate labels are malformed",
            failure_code="quadratic_risk_decomposition_invalid",
        )

    target_expanded = target[:, None, :]
    cross_rows = np.mean(target_expanded * prediction, axis=2, dtype=np.float64)
    energy_rows = np.mean(prediction * prediction, axis=2, dtype=np.float64)
    direct_rows = np.mean(
        target_expanded * target_expanded - (target_expanded - prediction) ** 2,
        axis=2,
        dtype=np.float64,
    )
    reconstructed_rows = 2.0 * cross_rows - energy_rows

    component_tables: dict[str, list[np.ndarray]] = {
        "cross": [],
        "energy": [],
        "direct": [],
        "reconstructed": [],
    }
    canonical_paths: np.ndarray | None = None
    for candidate in range(len(labels)):
        for name, rows in (
            ("cross", cross_rows[:, candidate]),
            ("energy", energy_rows[:, candidate]),
            ("direct", direct_rows[:, candidate]),
            ("reconstructed", reconstructed_rows[:, candidate]),
        ):
            aggregated = aggregate_zero_baseline_improvements(
                sample_keys=sample_keys,
                row_path_ids=row_path_ids,
                outer_steps=outer_steps,
                phases=phases,
                midpoint_indices=midpoint_indices,
                model_vs_zero_improvements=np.ascontiguousarray(rows),
                expected_path_ids=expected_path_ids,
            )
            if canonical_paths is None:
                canonical_paths = aggregated.path_ids
            elif not np.array_equal(canonical_paths, aggregated.path_ids):
                raise TimeLocalAdjudicationError(
                    "quadratic path aggregation changed order",
                    failure_code="quadratic_risk_decomposition_invalid",
                )
            component_tables[name].append(aggregated.path_values)
    assert canonical_paths is not None
    arrays = {
        name: np.ascontiguousarray(np.stack(values, axis=1))
        for name, values in component_tables.items()
    }
    identity_error = float(
        np.max(np.abs(arrays["direct"] - arrays["reconstructed"]))
    )
    if identity_error > float(identity_tolerance):
        raise TimeLocalAdjudicationError(
            "direct and reconstructed quadratic risks disagree",
            failure_code="quadratic_risk_decomposition_invalid",
        )
    return QuadraticRiskDecomposition(
        path_ids=canonical_paths,
        candidate_labels=labels,
        cross_terms=arrays["cross"],
        prediction_energies=arrays["energy"],
        direct_improvements=arrays["direct"],
        reconstructed_improvements=arrays["reconstructed"],
        maximum_identity_error=identity_error,
    )


def advisory_scalar_calibration(cross_term: float, prediction_energy: float) -> dict[str, Any]:
    """Return the nonauthorizing scalar optimum and directional risk ceiling."""

    cross = float(cross_term)
    energy = float(prediction_energy)
    if not math.isfinite(cross) or not math.isfinite(energy) or energy < 0.0:
        raise TimeLocalAdjudicationError(
            "scalar-calibration inputs are invalid",
            failure_code="quadratic_risk_decomposition_invalid",
        )
    usable = cross > 0.0 and energy > 0.0
    return {
        "cross_term": cross,
        "prediction_energy": energy,
        "scalar_optimum": float(cross / energy) if usable else None,
        "directional_ceiling": float(cross * cross / energy) if usable else None,
        "finite_positive_direction": int(usable),
        "authorizing": 0,
        "checkpoint_or_prediction_modified": 0,
    }


def classify_quartile_signal(
    *,
    nominee_cross_terms: Sequence[float],
    nominee_prediction_energies: Sequence[float],
    nominee_adjusted_lower_bounds: Sequence[float],
) -> str:
    """Classify one quartile from the three frozen nominees."""

    cross = np.asarray(nominee_cross_terms, dtype=np.float64)
    energy = np.asarray(nominee_prediction_energies, dtype=np.float64)
    lower = np.asarray(nominee_adjusted_lower_bounds, dtype=np.float64)
    if (
        cross.shape != (3,)
        or energy.shape != (3,)
        or lower.shape != (3,)
        or not np.isfinite(cross).all()
        or not np.isfinite(energy).all()
        or np.any(energy < 0.0)
    ):
        raise TimeLocalAdjudicationError(
            "quartile-classification inputs are invalid",
            failure_code="quadratic_risk_decomposition_invalid",
        )
    median_cross = float(np.median(cross))
    median_improvement = float(np.median(2.0 * cross - energy))
    if np.all(lower > 0.0):
        return "resolved"
    if median_cross <= 0.0:
        return "directional_alignment_missing"
    if median_improvement <= 0.0:
        return "prediction_energy_dominates"
    return "positive_but_underpowered"


def forecast_required_path_count(
    *,
    point_estimate: float,
    path_standard_deviation: float,
    critical_value: float,
) -> int | None:
    """Forecast paths needed for ``point - critical * sd/sqrt(n) > 0``.

    ``None`` is the JSON-safe representation of an infinite requirement and is
    mandatory for nonpositive effects.
    """

    point = float(point_estimate)
    deviation = float(path_standard_deviation)
    critical = float(critical_value)
    if (
        not math.isfinite(point)
        or not math.isfinite(deviation)
        or deviation < 0.0
        or not math.isfinite(critical)
        or critical < 0.0
    ):
        raise TimeLocalAdjudicationError("path-count forecast inputs are invalid")
    if point <= 0.0:
        return None
    if deviation == 0.0 or critical == 0.0:
        return 2
    boundary = (critical * deviation / point) ** 2
    if not math.isfinite(boundary):
        return None
    required = max(2, int(math.floor(boundary)) + 1)
    while point - critical * deviation / math.sqrt(required) <= 0.0:
        required += 1
    return required


def classify_adjudication_decision(
    replay: SealedSelectionReplay,
    *,
    witness_quartile_energies: Sequence[float],
) -> str:
    """Return the closed scientific decision from immutable replay evidence."""

    if not isinstance(replay, SealedSelectionReplay):
        raise TimeLocalAdjudicationError("decision requires a sealed replay")
    witness = np.asarray(witness_quartile_energies, dtype=np.float64)
    if witness.shape != (4,) or not np.isfinite(witness).all():
        raise TimeLocalAdjudicationError(
            "coarse witness replay is invalid",
            failure_code="coarse_witness_replay_invalid",
        )
    witness_positive = bool(np.mean(witness, dtype=np.float64) > 0.0)
    exact_q0 = (
        all(nominee.adjusted_lower_bound > 0.0 for nominee in replay.nominees)
        and all(
            nominee.positive_fine_cell_count >= math.ceil(0.9 * 56)
            for nominee in replay.nominees
        )
        and all(index < 56 or index == 224 for index in replay.discovered_component_indices)
        and not replay.candidate_with_all_point_estimates_positive
        and witness_positive
    )
    if exact_q0:
        return "exact_rb_high_reverse_time_only_signal"
    if (
        replay.positive_component_count == 0
        and not np.any(replay.result.point_estimates > 0.0)
    ):
        return "no_learned_time_local_signal"
    if (
        replay.positive_component_count == 0
        and replay.candidate_with_all_point_estimates_positive
        and witness_positive
    ):
        return "multiplicity_only_underpowered"
    return "mixed_time_local_signal_inconclusive"
