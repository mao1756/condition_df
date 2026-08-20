"""Frozen inference for the read-only quartile directional adjudication.

The helpers in this module operate only on already-reduced historical path
tables.  They do not load role labels, checkpoints, caches, or models.  The
joint family contains the pooled rank direction and the gain-fixed rank effect
for every quartile/component/seed stream.  Constant path statistics are
handled analytically instead of receiving an artificial standard-error floor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mnist.d0_jacobi_artifacts import config_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_quartile_specialist import (
    CHECKPOINT_UPDATES,
    MINIMUM_POSITIVE_FINE_CELLS,
    MODEL_SEEDS_BY_QUARTILE,
    Q1_SENTINEL,
)


SCHEMA = "d0-jacobi-rb-quartile-directional-adjudication-inference-v1"
COMPONENTS = ("full", "local_affine", "spatial_cnn")
STATISTICS = ("direction", "effect")
QUARTILE_COUNT = 4
SEED_COUNT = 3
PHASE_COUNT = 7
MIDPOINT_COUNT = 8
FINE_CELL_COUNT = PHASE_COUNT * MIDPOINT_COUNT
STREAM_COUNT = QUARTILE_COUNT * len(COMPONENTS) * SEED_COUNT
FAMILY_SIZE = STREAM_COUNT * len(STATISTICS)

DEFAULT_CONFIDENCE = 0.995
DEFAULT_REPLICATES = 50_000
DEFAULT_BOOTSTRAP_SEED = 261_352
DEFAULT_BOOTSTRAP_NAMESPACE = 0x51444149
DEFAULT_BOOTSTRAP_CHUNK_SIZE = 1_000
MAXIMUM_POWER_FORECAST_PATHS = 384


class DirectionalInferenceError(ValueError):
    """The frozen directional inference contract was violated."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str = "quartile_directional_rank_adjudication_invalid",
    ) -> None:
        super().__init__(message)
        self.failure_code = str(failure_code)


@dataclass(frozen=True, order=True)
class StreamIdentity:
    """One quartile/component/seed diagnostic stream."""

    quartile: int
    component: str
    seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.quartile, bool)
            or not isinstance(self.quartile, (int, np.integer))
            or not 0 <= int(self.quartile) < QUARTILE_COUNT
        ):
            raise DirectionalInferenceError("stream quartile is invalid")
        if self.component not in COMPONENTS:
            raise DirectionalInferenceError("stream component is invalid")
        if int(self.seed) not in MODEL_SEEDS_BY_QUARTILE[int(self.quartile)]:
            raise DirectionalInferenceError("stream seed is assigned to another quartile")

    @property
    def key(self) -> str:
        return f"q{int(self.quartile)}.{self.component}.seed{int(self.seed)}"

    def family_name(self, statistic: str) -> str:
        if statistic not in STATISTICS:
            raise DirectionalInferenceError("stream statistic is invalid")
        return f"{self.key}.{statistic}"

    def to_record(self) -> dict[str, Any]:
        return {
            "quartile": int(self.quartile),
            "component": self.component,
            "seed": int(self.seed),
            "stream_key": self.key,
        }


def canonical_stream_identities() -> tuple[StreamIdentity, ...]:
    """Return the frozen quartile/component/seed order."""

    streams = tuple(
        StreamIdentity(quartile, component, seed)
        for quartile in range(QUARTILE_COUNT)
        for component in COMPONENTS
        for seed in MODEL_SEEDS_BY_QUARTILE[quartile]
    )
    if len(streams) != STREAM_COUNT or len({stream.key for stream in streams}) != len(
        streams
    ):
        raise AssertionError("the frozen directional stream family is malformed")
    return streams


STREAM_IDENTITIES = canonical_stream_identities()


def direction_effect_family_names() -> tuple[str, ...]:
    """Return the frozen 72-member quartile/component/seed/statistic family."""

    names = tuple(
        stream.family_name(statistic)
        for stream in STREAM_IDENTITIES
        for statistic in STATISTICS
    )
    if len(names) != FAMILY_SIZE or len(set(names)) != FAMILY_SIZE:
        raise AssertionError("the frozen directional max-T family is malformed")
    return names


FAMILY_NAMES = direction_effect_family_names()
FAMILY_NAMES_SHA256 = config_fingerprint(list(FAMILY_NAMES))


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DirectionalInferenceError(f"{name} must be a finite scalar") from exc
    if not math.isfinite(result):
        raise DirectionalInferenceError(f"{name} must be a finite scalar")
    return result


def _readonly(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PositiveRayMoment:
    """Exact positive-ray summary for target and prediction moments."""

    target_energy: float
    cross_term: float
    prediction_energy: float
    rho: float
    lambda_plus: float
    directional_ceiling: float
    positive_direction: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "target_energy": float(self.target_energy),
            "cross_term": float(self.cross_term),
            "prediction_energy": float(self.prediction_energy),
            "rho": float(self.rho),
            "lambda_plus": float(self.lambda_plus),
            "directional_ceiling": float(self.directional_ceiling),
            "positive_direction": int(self.positive_direction),
            "gain_clipped": 0,
            "target_transformed": 0,
        }


def positive_ray_moment(
    target_energy: Any,
    cross_term: Any,
    prediction_energy: Any,
) -> PositiveRayMoment:
    """Return ``rho``, the best positive scalar, and its exact MSE ceiling."""

    target = _finite_float(target_energy, "target_energy")
    cross = _finite_float(cross_term, "cross_term")
    energy = _finite_float(prediction_energy, "prediction_energy")
    if target < 0.0 or energy < 0.0:
        raise DirectionalInferenceError("target and prediction energies cannot be negative")
    if energy == 0.0 and cross != 0.0:
        raise DirectionalInferenceError("P=0 with C!=0 is algebraically impossible")
    if (target == 0.0 or energy == 0.0) and cross != 0.0:
        raise DirectionalInferenceError("a degenerate energy has nonzero cross moment")
    rho = 0.0 if target == 0.0 or energy == 0.0 else cross / math.sqrt(target * energy)
    if not math.isfinite(rho):
        raise DirectionalInferenceError("scale-free alignment is nonfinite")
    positive = cross > 0.0 and energy > 0.0
    return PositiveRayMoment(
        target_energy=target,
        cross_term=cross,
        prediction_energy=energy,
        rho=rho,
        lambda_plus=cross / energy if positive else 0.0,
        directional_ceiling=cross * cross / energy if positive else 0.0,
        positive_direction=positive,
    )


def quadratic_improvement(cross_term: Any, prediction_energy: Any, gain: Any) -> Any:
    """Evaluate ``2*gain*C-gain**2*P`` without clipping or projection."""

    cross = np.asarray(cross_term, dtype=np.float64)
    energy = np.asarray(prediction_energy, dtype=np.float64)
    active_gain = _finite_float(gain, "gain")
    if (
        cross.shape != energy.shape
        or not np.isfinite(cross).all()
        or not np.isfinite(energy).all()
        or np.any(energy < 0.0)
    ):
        raise DirectionalInferenceError("C and P must be finite matching moments")
    result = 2.0 * active_gain * cross - active_gain * active_gain * energy
    if result.ndim == 0:
        return float(result)
    return np.ascontiguousarray(result)


def _record_stream(record: Mapping[str, Any]) -> StreamIdentity:
    if not isinstance(record, Mapping):
        raise DirectionalInferenceError("candidate moment row is malformed")
    return StreamIdentity(
        quartile=record.get("quartile"),
        component=str(record.get("component")),
        seed=record.get("seed"),
    )


def select_direction_nominees(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    require_complete_grid: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Nominate one gain-role checkpoint per stream by ``D_plus``, then update.

    Candidate rows must contain ``quartile``, ``component``, ``seed``,
    ``update``, ``target_energy``, ``cross_term``, and ``prediction_energy``.
    Rank-role values are intentionally neither accepted nor consulted.
    """

    grouped: dict[StreamIdentity, list[tuple[int, PositiveRayMoment]]] = {
        stream: [] for stream in STREAM_IDENTITIES
    }
    seen: set[tuple[StreamIdentity, int]] = set()
    for record in candidate_records:
        stream = _record_stream(record)
        update_value = record.get("update")
        if (
            isinstance(update_value, bool)
            or not isinstance(update_value, (int, np.integer))
            or int(update_value) not in CHECKPOINT_UPDATES
        ):
            raise DirectionalInferenceError("candidate update is outside the frozen grid")
        update = int(update_value)
        identity = (stream, update)
        if identity in seen:
            raise DirectionalInferenceError("candidate stream/update is duplicated")
        seen.add(identity)
        moment = positive_ray_moment(
            record.get("target_energy"),
            record.get("cross_term"),
            record.get("prediction_energy"),
        )
        supplied_ceiling = record.get("directional_ceiling")
        if supplied_ceiling is not None and float(supplied_ceiling) != moment.directional_ceiling:
            raise DirectionalInferenceError("supplied directional ceiling changed")
        grouped[stream].append((update, moment))

    nominees: list[dict[str, Any]] = []
    for stream in STREAM_IDENTITIES:
        rows = grouped[stream]
        if require_complete_grid and tuple(sorted(update for update, _ in rows)) != tuple(
            CHECKPOINT_UPDATES
        ):
            raise DirectionalInferenceError("gain-role candidate grid is incomplete")
        positive = [(update, moment) for update, moment in rows if moment.positive_direction]
        if not positive:
            nominees.append(
                {
                    **stream.to_record(),
                    "status": "no_positive_gain_direction",
                    "nominee_present": 0,
                    "update": None,
                    "target_energy_gain": None,
                    "C_gain": None,
                    "P_gain": None,
                    "lambda_gain": None,
                    "D_plus_gain": None,
                    "candidate_count": len(rows),
                    "positive_gain_direction_count": 0,
                    "rank_evidence_used": 0,
                    "historical_design_evidence": 1,
                    "authorizing": 0,
                }
            )
            continue
        selected_update, selected_moment = min(
            positive,
            key=lambda item: (-item[1].directional_ceiling, item[0]),
        )
        nominees.append(
            {
                **stream.to_record(),
                "status": "gain_direction_nominee",
                "nominee_present": 1,
                "update": selected_update,
                "target_energy_gain": selected_moment.target_energy,
                "C_gain": selected_moment.cross_term,
                "P_gain": selected_moment.prediction_energy,
                "rho_gain": selected_moment.rho,
                "lambda_gain": selected_moment.lambda_plus,
                "D_plus_gain": selected_moment.directional_ceiling,
                "candidate_count": len(rows),
                "positive_gain_direction_count": len(positive),
                "ranking_rule": ["largest_D_plus_gain", "earlier_update"],
                "rank_evidence_used": 0,
                "historical_design_evidence": 1,
                "authorizing": 0,
            }
        )
    return tuple(nominees)


def local_compatibility_screen(values: Any, *, quartile: int) -> dict[str, Any]:
    """Apply the unchanged phase, midpoint, 51/56, and q1 sentinel screen."""

    cells = np.asarray(values, dtype=np.float64)
    if cells.shape != (PHASE_COUNT, MIDPOINT_COUNT) or not np.isfinite(cells).all():
        raise DirectionalInferenceError("local screen requires a finite [7,8] table")
    if isinstance(quartile, bool) or not isinstance(quartile, (int, np.integer)):
        raise DirectionalInferenceError("quartile is invalid")
    active_quartile = int(quartile)
    if not 0 <= active_quartile < QUARTILE_COUNT:
        raise DirectionalInferenceError("quartile is invalid")
    phase = np.mean(cells, axis=1, dtype=np.float64)
    midpoint = np.mean(cells, axis=0, dtype=np.float64)
    positive_count = int(np.count_nonzero(cells > 0.0))
    sentinel = float(cells[Q1_SENTINEL])
    phase_pass = bool(np.all(phase > 0.0))
    midpoint_pass = bool(np.all(midpoint > 0.0))
    cell_pass = positive_count >= MINIMUM_POSITIVE_FINE_CELLS
    sentinel_pass = active_quartile != 1 or sentinel > 0.0
    passed = phase_pass and midpoint_pass and cell_pass and sentinel_pass
    return {
        "quartile": active_quartile,
        "pooled": float(np.mean(cells, dtype=np.float64)),
        "phase_marginals": phase.tolist(),
        "midpoint_marginals": midpoint.tolist(),
        "fine_cells": cells.tolist(),
        "positive_fine_cell_count": positive_count,
        "minimum_positive_fine_cell_count": MINIMUM_POSITIVE_FINE_CELLS,
        "phase_marginals_strictly_positive": int(phase_pass),
        "midpoint_marginals_strictly_positive": int(midpoint_pass),
        "positive_cell_count_passed": int(cell_pass),
        "phase4_midpoint7": sentinel,
        "q1_sentinel_passed": int(sentinel_pass),
        "passed": int(passed),
    }


@dataclass(frozen=True)
class DirectionEffectMaxTResult:
    """One-sided simultaneous inference over the fixed 72-member family."""

    path_ids: np.ndarray
    family_names: tuple[str, ...]
    point_estimates: np.ndarray
    standard_errors: np.ndarray
    lower_bounds: np.ndarray
    analytic_constant_mask: np.ndarray
    maxima: np.ndarray
    critical_value: float
    confidence: float
    replicates: int
    seed: int
    namespace: int

    def __post_init__(self) -> None:
        paths = np.asarray(self.path_ids)
        point = np.asarray(self.point_estimates)
        error = np.asarray(self.standard_errors)
        lower = np.asarray(self.lower_bounds)
        constants = np.asarray(self.analytic_constant_mask)
        maxima = np.asarray(self.maxima)
        if (
            paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or paths.size < 2
            or np.unique(paths).size != paths.size
            or tuple(self.family_names) != FAMILY_NAMES
            or point.dtype != np.dtype(np.float64)
            or point.shape != (FAMILY_SIZE,)
            or error.dtype != np.dtype(np.float64)
            or error.shape != point.shape
            or lower.dtype != np.dtype(np.float64)
            or lower.shape != point.shape
            or constants.dtype != np.dtype(np.bool_)
            or constants.shape != point.shape
            or maxima.dtype != np.dtype(np.float64)
            or maxima.shape != (int(self.replicates),)
            or not np.isfinite(point).all()
            or not np.isfinite(error).all()
            or not np.isfinite(lower).all()
            or not np.isfinite(maxima).all()
            or np.any(error < 0.0)
            or not math.isfinite(float(self.critical_value))
            or float(self.critical_value) < 0.0
        ):
            raise DirectionalInferenceError("direction/effect max-T result is malformed")
        object.__setattr__(self, "path_ids", _readonly(paths, dtype=np.dtype(np.int64)))
        object.__setattr__(self, "point_estimates", _readonly(point, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "standard_errors", _readonly(error, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "lower_bounds", _readonly(lower, dtype=np.dtype(np.float64)))
        object.__setattr__(self, "analytic_constant_mask", _readonly(constants, dtype=np.dtype(np.bool_)))
        object.__setattr__(self, "maxima", _readonly(maxima, dtype=np.dtype(np.float64)))

    def lower_bound(self, stream: StreamIdentity, statistic: str) -> float:
        name = stream.family_name(statistic)
        return float(self.lower_bounds[self.family_names.index(name)])

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA + "-max-t",
            "schema_version": 1,
            "method": "centered_whole_path_one_sided_studentized_max_t",
            "confidence": float(self.confidence),
            "replicates": int(self.replicates),
            "seed": int(self.seed),
            "namespace": int(self.namespace),
            "quantile_method": "higher",
            "studentization": 1,
            "standard_error_floor_used": 0,
            "zero_standard_error_handling": "finite_constant_analytic_lower_bound",
            "bootstrap_unit": "whole_training_rank_path",
            "family_names": list(self.family_names),
            "family_names_sha256": FAMILY_NAMES_SHA256,
            "family_size": FAMILY_SIZE,
            "path_ids": self.path_ids.tolist(),
            "point_estimates": dict(zip(self.family_names, self.point_estimates.tolist(), strict=True)),
            "standard_errors": dict(zip(self.family_names, self.standard_errors.tolist(), strict=True)),
            "lower_bounds": dict(zip(self.family_names, self.lower_bounds.tolist(), strict=True)),
            "analytic_constant_members": [
                name
                for name, active in zip(self.family_names, self.analytic_constant_mask, strict=True)
                if bool(active)
            ],
            "critical_value": float(self.critical_value),
            "all_values_finite": 1,
        }


def _canonical_path_table(path_values: Any, path_ids: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(path_values)
    raw_paths = np.asarray(path_ids)
    if raw_paths.ndim != 1 or raw_paths.dtype.kind not in "iu":
        raise DirectionalInferenceError("max-T path IDs must be an integer vector")
    paths = np.asarray(raw_paths, dtype=np.int64)
    if paths.size < 2 or np.unique(paths).size != paths.size:
        raise DirectionalInferenceError("max-T paths must be unique")
    if (
        values.dtype != np.dtype(np.float64)
        or values.shape != (paths.size, FAMILY_SIZE)
        or not np.isfinite(values).all()
    ):
        raise DirectionalInferenceError("max-T values must be finite binary64 [path,72]")
    order = np.argsort(paths, kind="stable")
    return np.ascontiguousarray(paths[order]), np.ascontiguousarray(values[order])


def one_sided_direction_effect_max_t(
    path_values: Any,
    *,
    path_ids: Any,
    family_names: Sequence[str] = FAMILY_NAMES,
    confidence: float = DEFAULT_CONFIDENCE,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    namespace: int = DEFAULT_BOOTSTRAP_NAMESPACE,
    chunk_size: int = DEFAULT_BOOTSTRAP_CHUNK_SIZE,
) -> DirectionEffectMaxTResult:
    """Run deterministic whole-path max-T with analytic constant members."""

    if tuple(family_names) != FAMILY_NAMES:
        raise DirectionalInferenceError("direction/effect family order changed")
    if (
        not 0.0 < float(confidence) < 1.0
        or not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates <= 0
        or not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(namespace, int)
        or isinstance(namespace, bool)
    ):
        raise DirectionalInferenceError("max-T configuration is invalid")
    paths, values = _canonical_path_table(path_values, path_ids)
    path_count = int(paths.size)
    point = np.mean(values, axis=0, dtype=np.float64)
    error = np.std(values, axis=0, ddof=1, dtype=np.float64) / math.sqrt(path_count)
    constants = error == 0.0
    stochastic = ~constants
    maxima = np.zeros(replicates, dtype=np.float64)
    generator = np.random.Generator(
        np.random.Philox([int(seed), int(namespace)])
    )
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        indices = generator.integers(
            0,
            path_count,
            size=(stop - start, path_count),
            dtype=np.int64,
        )
        sampled = values[indices]
        draw_mean = np.mean(sampled, axis=1, dtype=np.float64)
        if np.any(stochastic):
            draw_error = np.std(sampled[:, :, stochastic], axis=1, ddof=1, dtype=np.float64) / math.sqrt(path_count)
            if not np.isfinite(draw_error).all() or np.any(draw_error <= 0.0):
                raise DirectionalInferenceError(
                    "bootstrap produced degenerate/nonfinite studentization"
                )
            centered = (draw_mean[:, stochastic] - point[None, stochastic]) / draw_error
            maxima[start:stop] = np.maximum(0.0, np.max(centered, axis=1))
    if not np.isfinite(maxima).all():
        raise DirectionalInferenceError("max-T bootstrap is nonfinite")
    critical = float(np.quantile(maxima, float(confidence), method="higher"))
    lower = point - critical * error
    lower[constants] = point[constants]
    return DirectionEffectMaxTResult(
        path_ids=paths,
        family_names=FAMILY_NAMES,
        point_estimates=np.ascontiguousarray(point),
        standard_errors=np.ascontiguousarray(error),
        lower_bounds=np.ascontiguousarray(lower),
        analytic_constant_mask=np.ascontiguousarray(constants),
        maxima=np.ascontiguousarray(maxima),
        critical_value=critical,
        confidence=float(confidence),
        replicates=int(replicates),
        seed=int(seed),
        namespace=int(namespace),
    )


def build_stream_path_table(
    stream_records: Sequence[Mapping[str, Any]],
    *,
    path_ids: Any,
) -> np.ndarray:
    """Build canonical ``[path,72]`` direction/effect evidence from 36 rows.

    Each row supplies ``direction_path_values`` and ``effect_path_values``.
    A stream without a gain nominee must supply exact-zero effect values, so it
    remains a fixed failing member of the family.
    """

    raw_paths = np.asarray(path_ids)
    if raw_paths.ndim != 1 or raw_paths.dtype.kind not in "iu":
        raise DirectionalInferenceError("stream path IDs must be integer")
    paths = np.asarray(raw_paths, dtype=np.int64)
    if paths.size < 2 or np.unique(paths).size != paths.size:
        raise DirectionalInferenceError("stream path IDs must be unique")
    by_stream: dict[StreamIdentity, Mapping[str, Any]] = {}
    for record in stream_records:
        stream = _record_stream(record)
        if stream in by_stream:
            raise DirectionalInferenceError("rank stream is duplicated")
        by_stream[stream] = record
    if tuple(by_stream) != STREAM_IDENTITIES:
        raise DirectionalInferenceError("rank stream order changed")
    columns: list[np.ndarray] = []
    for stream in STREAM_IDENTITIES:
        record = by_stream[stream]
        direction = np.asarray(record.get("direction_path_values"))
        effect = np.asarray(record.get("effect_path_values"))
        if (
            direction.dtype != np.dtype(np.float64)
            or direction.shape != paths.shape
            or effect.dtype != np.dtype(np.float64)
            or effect.shape != paths.shape
            or not np.isfinite(direction).all()
            or not np.isfinite(effect).all()
        ):
            raise DirectionalInferenceError("rank stream path values are malformed")
        if not bool(record.get("nominee_present", 0)) and np.any(effect != 0.0):
            raise DirectionalInferenceError("a no-nominee stream has a nonzero effect entry")
        columns.extend((direction, effect))
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


def adjudicate_seed_streams(
    stream_records: Sequence[Mapping[str, Any]],
    inference: DirectionEffectMaxTResult,
) -> dict[str, Any]:
    """Apply the complete seed and two-of-three component stability rules."""

    if not isinstance(inference, DirectionEffectMaxTResult):
        raise DirectionalInferenceError("stream adjudication requires max-T evidence")
    by_stream: dict[StreamIdentity, Mapping[str, Any]] = {}
    for record in stream_records:
        stream = _record_stream(record)
        if stream in by_stream:
            raise DirectionalInferenceError("adjudication stream is duplicated")
        by_stream[stream] = record
    if tuple(by_stream) != STREAM_IDENTITIES:
        raise DirectionalInferenceError("adjudication stream order changed")

    seed_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for stream in STREAM_IDENTITIES:
        record = by_stream[stream]
        nominee_present = bool(record.get("nominee_present", 0))
        c_gain = _finite_float(record.get("C_gain", 0.0), "C_gain")
        c_rank = _finite_float(record.get("C_rank", 0.0), "C_rank")
        direction_cells = np.asarray(record.get("direction_cells"), dtype=np.float64)
        effect_cells = np.asarray(record.get("effect_cells"), dtype=np.float64)
        direction_screen = local_compatibility_screen(direction_cells, quartile=stream.quartile)
        effect_screen = local_compatibility_screen(effect_cells, quartile=stream.quartile)
        direction_lower = inference.lower_bound(stream, "direction")
        effect_lower = inference.lower_bound(stream, "effect")
        effect_point = _finite_float(record.get("effect_point"), "effect_point")
        direction_passed = bool(
            nominee_present
            and c_gain > 0.0
            and c_rank > 0.0
            and direction_lower > 0.0
            and direction_screen["passed"]
        )
        effect_passed = bool(
            nominee_present
            and effect_point > 0.0
            and effect_lower > 0.0
            and effect_screen["passed"]
        )
        seed_rows.append(
            {
                **stream.to_record(),
                "nominee_present": int(nominee_present),
                "C_gain": c_gain,
                "C_rank": c_rank,
                "direction_lower_bound": direction_lower,
                "effect_point": effect_point,
                "effect_lower_bound": effect_lower,
                "direction_local_screen": direction_screen,
                "effect_local_screen": effect_screen,
                "direction_passed": int(direction_passed),
                "effect_passed": int(effect_passed),
                "historical_design_evidence": 1,
                "authorizing": 0,
            }
        )

    for quartile in range(QUARTILE_COUNT):
        for component in COMPONENTS:
            rows = [
                row
                for row in seed_rows
                if row["quartile"] == quartile and row["component"] == component
            ]
            direction_seeds = [row["seed"] for row in rows if row["direction_passed"]]
            effect_seeds = [row["seed"] for row in rows if row["effect_passed"]]
            component_rows.append(
                {
                    "quartile": quartile,
                    "component": component,
                    "direction_passing_seed_count": len(direction_seeds),
                    "direction_passing_seeds": direction_seeds,
                    "effect_passing_seed_count": len(effect_seeds),
                    "effect_passing_seeds": effect_seeds,
                    "stable_direction": int(len(direction_seeds) >= 2),
                    "stable_effect": int(len(effect_seeds) >= 2),
                    "historical_design_evidence": 1,
                    "authorizing": 0,
                }
            )
    return {
        "schema": SCHEMA + "-stream-adjudication",
        "schema_version": 1,
        "seed_rows": seed_rows,
        "component_rows": component_rows,
        "family_names_sha256": FAMILY_NAMES_SHA256,
        "historical_design_evidence": 1,
        "authorizing": 0,
    }


def path_stability_summary(path_values: Any, *, simultaneous_lower_bound: Any) -> dict[str, Any]:
    """Summarize whole-path sign stability without changing the inferential unit."""

    values = np.asarray(path_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise DirectionalInferenceError("path stability requires a finite vector")
    lower = _finite_float(simultaneous_lower_bound, "simultaneous_lower_bound")
    point = float(np.mean(values, dtype=np.float64))
    standard_deviation = float(np.std(values, ddof=1, dtype=np.float64))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    positive_count = int(np.count_nonzero(values > 0.0))
    positive_fraction = positive_count / int(values.size)
    entropy = 0.0
    for probability in (positive_fraction, 1.0 - positive_fraction):
        if probability > 0.0:
            entropy -= probability * math.log2(probability)
    return {
        "path_count": int(values.size),
        "mean": point,
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(int(values.size)),
        "median": median,
        "median_absolute_deviation": mad,
        "positive_path_count": positive_count,
        "nonpositive_path_count": int(values.size) - positive_count,
        "positive_path_fraction": positive_fraction,
        "sign_entropy_bits": entropy,
        "simultaneous_lower_bound": lower,
        "path_unstable": int(point > 0.0 and lower <= 0.0),
    }


def forecast_required_paths(
    path_values: Any,
    *,
    critical_value: Any,
    local_point_screen_passed: bool,
    maximum_paths: int = MAXIMUM_POWER_FORECAST_PATHS,
) -> dict[str, Any]:
    """Return the frozen advisory historical-design path-count forecast."""

    values = np.asarray(path_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise DirectionalInferenceError("power forecast requires finite whole-path values")
    critical = _finite_float(critical_value, "critical_value")
    if critical < 0.0:
        raise DirectionalInferenceError("critical value cannot be negative")
    if (
        isinstance(maximum_paths, bool)
        or not isinstance(maximum_paths, (int, np.integer))
        or int(maximum_paths) <= 0
    ):
        raise DirectionalInferenceError("power-forecast path cap is invalid")
    point = float(np.mean(values, dtype=np.float64))
    standard_deviation = float(np.std(values, ddof=1, dtype=np.float64))
    if not local_point_screen_passed or point <= 0.0:
        return {
            "path_count": int(values.size),
            "path_mean": point,
            "path_standard_deviation": standard_deviation,
            "critical_value": critical,
            "n_required": "not_finite",
            "n_required_rounded": "not_finite",
            "above_384_cap": 0,
            "reason": "nonpositive_or_locally_incompatible_effect",
            "planning_forecast_only": 1,
            "authorizing": 0,
        }
    raw = (critical * standard_deviation / point) ** 2
    if not math.isfinite(raw):
        rounded: int | str = "not_finite"
    else:
        rounded = 32 * int(math.ceil(math.ceil(raw) / 32.0))
    return {
        "path_count": int(values.size),
        "path_mean": point,
        "path_standard_deviation": standard_deviation,
        "critical_value": critical,
        "n_required": int(math.ceil(raw)) if math.isfinite(raw) else "not_finite",
        "n_required_rounded": rounded,
        "above_384_cap": int(isinstance(rounded, int) and rounded > int(maximum_paths)),
        "maximum_path_cap": int(maximum_paths),
        "reason": "finite_historical_design_forecast" if isinstance(rounded, int) else "nonfinite_power_forecast",
        "planning_forecast_only": 1,
        "authorizing": 0,
    }


def trajectory_rotation_diagnostics(
    *,
    updates: Any,
    prediction_vectors: Any,
    pooled_cross_terms: Any,
    gain_cell_profile: Any | None = None,
    rank_cell_profile: Any | None = None,
    gain_nominee_rank_cross_term: Any | None = None,
) -> dict[str, Any]:
    """Detect only the four preregistered optimization-rotation events."""

    update_array = np.asarray(updates)
    predictions = np.asarray(prediction_vectors, dtype=np.float64)
    cross = np.asarray(pooled_cross_terms, dtype=np.float64)
    if (
        update_array.ndim != 1
        or update_array.dtype.kind not in "iu"
        or update_array.size < 2
        or np.any(np.diff(update_array.astype(np.int64)) <= 0)
        or predictions.ndim != 2
        or predictions.shape[0] != update_array.size
        or cross.shape != (update_array.size,)
        or not np.isfinite(predictions).all()
        or not np.isfinite(cross).all()
    ):
        raise DirectionalInferenceError("optimization trajectory is malformed")
    cosines: list[float | None] = []
    for left, right in zip(predictions[:-1], predictions[1:], strict=True):
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            cosines.append(None)
        else:
            cosine = float(np.dot(left, right) / (left_norm * right_norm))
            cosines.append(min(1.0, max(-1.0, cosine)))
    nonzero_signs = np.sign(cross[cross != 0.0])
    sign_change_count = int(np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1]))
    profile_correlation: float | None = None
    if gain_cell_profile is not None or rank_cell_profile is not None:
        if gain_cell_profile is None or rank_cell_profile is None:
            raise DirectionalInferenceError("both gain and rank profiles are required")
        gain_profile = np.asarray(gain_cell_profile, dtype=np.float64)
        rank_profile = np.asarray(rank_cell_profile, dtype=np.float64)
        if (
            gain_profile.shape != (PHASE_COUNT, MIDPOINT_COUNT)
            or rank_profile.shape != gain_profile.shape
            or not np.isfinite(gain_profile).all()
            or not np.isfinite(rank_profile).all()
        ):
            raise DirectionalInferenceError("gain/rank directional profiles are malformed")
        gain_flat = gain_profile.ravel(order="C")
        rank_flat = rank_profile.ravel(order="C")
        if float(np.std(gain_flat)) > 0.0 and float(np.std(rank_flat)) > 0.0:
            profile_correlation = float(np.corrcoef(gain_flat, rank_flat)[0, 1])
    nominee_rank_nonpositive = False
    nominee_rank_value: float | None = None
    if gain_nominee_rank_cross_term is not None:
        nominee_rank_value = _finite_float(
            gain_nominee_rank_cross_term, "gain_nominee_rank_cross_term"
        )
        nominee_rank_nonpositive = nominee_rank_value <= 0.0
    negative_adjacent = any(value is not None and value < 0.0 for value in cosines)
    nonpositive_profile = profile_correlation is not None and profile_correlation <= 0.0
    events = {
        "negative_adjacent_prediction_cosine": int(negative_adjacent),
        "pooled_cross_term_changes_sign_at_least_twice": int(sign_change_count >= 2),
        "gain_nominee_rank_cross_term_nonpositive": int(nominee_rank_nonpositive),
        "gain_rank_cell_profile_correlation_nonpositive": int(nonpositive_profile),
    }
    return {
        "updates": update_array.astype(np.int64).tolist(),
        "adjacent_prediction_cosines": cosines,
        "pooled_cross_term_sign_change_count": sign_change_count,
        "gain_nominee_rank_cross_term": nominee_rank_value,
        "gain_rank_cell_profile_correlation": profile_correlation,
        **events,
        "rotation_event_count": sum(events.values()),
        "optimization_time_rotation": int(any(events.values())),
        "checkpoint_nominated_on_rank": 0,
        "authorizing": 0,
    }


# Concise aliases used by callers and tests.
directional_max_t = one_sided_direction_effect_max_t
nominate_gain_directions = select_direction_nominees
directional_screen = local_compatibility_screen
power_forecast = forecast_required_paths
direction_family_names = direction_effect_family_names


__all__ = [
    "COMPONENTS",
    "DEFAULT_BOOTSTRAP_CHUNK_SIZE",
    "DEFAULT_BOOTSTRAP_NAMESPACE",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_REPLICATES",
    "DirectionalInferenceError",
    "DirectionEffectMaxTResult",
    "FAMILY_NAMES",
    "FAMILY_NAMES_SHA256",
    "FAMILY_SIZE",
    "FINE_CELL_COUNT",
    "MAXIMUM_POWER_FORECAST_PATHS",
    "MIDPOINT_COUNT",
    "PHASE_COUNT",
    "PositiveRayMoment",
    "QUARTILE_COUNT",
    "SCHEMA",
    "SEED_COUNT",
    "STATISTICS",
    "STREAM_COUNT",
    "STREAM_IDENTITIES",
    "StreamIdentity",
    "adjudicate_seed_streams",
    "build_stream_path_table",
    "canonical_stream_identities",
    "direction_effect_family_names",
    "direction_family_names",
    "directional_max_t",
    "directional_screen",
    "forecast_required_paths",
    "local_compatibility_screen",
    "nominate_gain_directions",
    "one_sided_direction_effect_max_t",
    "path_stability_summary",
    "positive_ray_moment",
    "power_forecast",
    "quadratic_improvement",
    "select_direction_nominees",
    "trajectory_rotation_diagnostics",
]
