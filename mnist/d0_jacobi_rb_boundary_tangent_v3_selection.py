"""Search-aware whole-path inference for the zero-baseline v3 experiment.

The v2 confirmation statistic is intentionally name-locked.  This additive
module keeps that API untouched while giving validation and confirmation one
shared numeric implementation.  Validation jointly covers the frozen
``120 x 228`` family without materializing a bootstrap tensor indexed by
replicate, path, and family member.

Bootstrap draws are stored as path multiplicities.  Count shards are
prospective design artifacts; maxima shards are restartable derived artifacts.
No standard-error floor or negative-improvement truncation is used.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mnist.d0_jacobi_artifacts import (
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    MIDPOINT_COUNT,
    SELECTED_OUTER_STEPS,
)
from mnist.d0_jacobi_rb_boundary_tangent_confirmation import (
    aggregate_confirmation_improvements,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    COMBINED_VS_ZERO_FAMILY_SIZE,
    CONFIRMATION_FAMILY_SIZE,
    PHASE_COUNT,
    TIME_QUARTILES,
)


SCHEMA = "d0-jacobi-rb-boundary-tangent-v3-selection-v1"
BOOTSTRAP_COUNTS_SCHEMA = SCHEMA + "-bootstrap-count-shard"
BOOTSTRAP_MAXIMA_SCHEMA = SCHEMA + "-bootstrap-maxima-shard"
NUMERIC_MAX_T_SCHEMA = SCHEMA + "-numeric-max-t"

V3_MODEL_SEEDS = (261_312, 261_313, 261_314)
V3_NONZERO_UPDATES = tuple(range(100, 4_001, 100))
V3_CANDIDATE_COUNT = len(V3_MODEL_SEEDS) * len(V3_NONZERO_UPDATES)
V3_COMPONENT_COUNT = CONFIRMATION_FAMILY_SIZE
V3_SEARCH_FAMILY_SIZE = V3_CANDIDATE_COUNT * V3_COMPONENT_COUNT
V3_VALIDATION_PATH_COUNT = 32

DEFAULT_CONFIDENCE = 0.995
DEFAULT_REPLICATES = 50_000
DEFAULT_BOOTSTRAP_SHARD_SIZE = 1_000
DEFAULT_CANDIDATE_BLOCK_SIZE = 20
DEFAULT_COMPONENT_BLOCK_SIZE = 57
MAXIMUM_WORKING_BYTES = 64 * 1024 * 1024
DEFAULT_SELECTION_SEED = 261_320
DEFAULT_CONFIRMATION_SEED = 261_322
SELECTION_NAMESPACE = 0x42545633
CONFIRMATION_NAMESPACE = 0x42544333
PHILOX_CONSTRUCTOR = (
    "np.random.Generator(np.random.Philox([int(seed), int(namespace), "
    "int(shard_index)]))"
)


class V3SelectionError(ValueError):
    """The frozen v3 aggregation or inference contract was violated."""

    def __init__(self, message: str, *, failure_code: str = "validation_inference_invalid"):
        super().__init__(message)
        self.failure_code = str(failure_code)


def v3_family_names() -> tuple[str, ...]:
    """Return the frozen all-versus-zero 228-member family."""

    fine = tuple(
        f"model_vs_zero.q{quartile}.phase{phase}.midpoint{midpoint}"
        for quartile in range(TIME_QUARTILES)
        for phase in range(PHASE_COUNT)
        for midpoint in range(MIDPOINT_COUNT)
    )
    pooled = tuple(
        f"model_vs_zero.q{quartile}.pooled"
        for quartile in range(TIME_QUARTILES)
    )
    names = fine + pooled
    if (
        len(fine) != COMBINED_VS_ZERO_FAMILY_SIZE
        or len(names) != V3_COMPONENT_COUNT
        or len(set(names)) != V3_COMPONENT_COUNT
    ):
        raise AssertionError("the frozen v3 family is malformed")
    return names


V3_FAMILY_NAMES = v3_family_names()
V3_FINE_FAMILY_NAMES = V3_FAMILY_NAMES[:COMBINED_VS_ZERO_FAMILY_SIZE]
V3_POOLED_FAMILY_NAMES = V3_FAMILY_NAMES[COMBINED_VS_ZERO_FAMILY_SIZE:]
V3_FAMILY_NAMES_SHA256 = config_fingerprint(list(V3_FAMILY_NAMES))


def v3_search_family_names() -> tuple[str, ...]:
    """Return the frozen candidate-major 27,360-member search order."""

    names = tuple(
        f"seed{seed}.update{update:04d}.{component}"
        for seed in V3_MODEL_SEEDS
        for update in V3_NONZERO_UPDATES
        for component in V3_FAMILY_NAMES
    )
    if len(names) != V3_SEARCH_FAMILY_SIZE or len(set(names)) != len(names):
        raise AssertionError("the frozen v3 search family is malformed")
    return names


V3_SEARCH_FAMILY_NAMES = v3_search_family_names()
V3_SEARCH_FAMILY_NAMES_SHA256 = config_fingerprint(list(V3_SEARCH_FAMILY_NAMES))


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(config_fingerprint(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_path_ids(value: Any, *, minimum_count: int = 8) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise V3SelectionError("path IDs must be an integer vector")
    paths = np.asarray(source, dtype=np.int64)
    if (
        paths.size < int(minimum_count)
        or np.any(paths < 0)
        or np.any(paths >= 1 << 20)
        or np.unique(paths).size != paths.size
    ):
        raise V3SelectionError("path IDs must be unique valid 20-bit IDs")
    return np.ascontiguousarray(paths)


@dataclass(frozen=True)
class ZeroBaselineRiskTableV3:
    """Canonical path table for one v3 candidate or confirmation audit."""

    path_ids: np.ndarray
    path_values: np.ndarray
    cell_counts: np.ndarray
    sample_key_sha256: str
    row_count: int

    def __post_init__(self) -> None:
        paths = np.asarray(self.path_ids)
        values = np.asarray(self.path_values)
        counts = np.asarray(self.cell_counts)
        if (
            paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or paths.size < 8
            or np.any(paths < 0)
            or np.any(paths >= 1 << 20)
            or np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths, kind="stable"))
            or values.dtype != np.dtype(np.float64)
            or values.shape != (paths.size, V3_COMPONENT_COUNT)
            or not np.isfinite(values).all()
            or counts.dtype != np.dtype(np.int64)
            or counts.shape != values.shape
            or np.any(counts <= 0)
            or not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count <= 0
            or len(self.sample_key_sha256) != 64
        ):
            raise V3SelectionError("zero-baseline path-risk table is malformed")
        object.__setattr__(self, "path_ids", _readonly(paths))
        object.__setattr__(self, "path_values", _readonly(values))
        object.__setattr__(self, "cell_counts", _readonly(counts))

    def to_record(self) -> dict[str, Any]:
        point = np.mean(self.path_values, axis=0, dtype=np.float64)
        return {
            "schema": SCHEMA + "-path-risk-table",
            "schema_version": 1,
            "family_names": list(V3_FAMILY_NAMES),
            "family_names_sha256": V3_FAMILY_NAMES_SHA256,
            "search_family_names_sha256": V3_SEARCH_FAMILY_NAMES_SHA256,
            "family_size": V3_COMPONENT_COUNT,
            "path_ids": self.path_ids.tolist(),
            "path_count": int(self.path_ids.size),
            "row_count": int(self.row_count),
            "sample_key_sha256": self.sample_key_sha256,
            "minimum_cell_count": int(np.min(self.cell_counts)),
            "maximum_cell_count": int(np.max(self.cell_counts)),
            "point_estimates": {
                name: float(value)
                for name, value in zip(V3_FAMILY_NAMES, point, strict=True)
            },
            "baseline_contrast_present": 0,
            "negative_values_truncated": 0,
            "target_transformed": 0,
            "bootstrap_unit": "whole_path",
        }


def aggregate_zero_baseline_improvements(
    *,
    sample_keys: Any,
    row_path_ids: Any,
    outer_steps: Any,
    phases: Any,
    midpoint_indices: Any,
    model_vs_zero_improvements: Any,
    expected_path_ids: Any,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    pooled_model_vs_zero_improvements: Any | None = None,
) -> ZeroBaselineRiskTableV3:
    """Aggregate one identical raw-improvement source into fine and pooled cells.

    The v2 aggregator already enforces complete path/time/phase/midpoint
    identity.  Passing the exact same source into both of its numeric slots
    avoids a second identity implementation.  The returned names and semantics
    are v3-only and contain no baseline contrast.
    """

    fine_source = np.asarray(model_vs_zero_improvements)
    pooled_source = (
        fine_source
        if pooled_model_vs_zero_improvements is None
        else np.asarray(pooled_model_vs_zero_improvements)
    )
    if (
        fine_source.dtype != np.dtype(np.float64)
        or fine_source.ndim != 1
        or not np.isfinite(fine_source).all()
        or pooled_source.dtype != np.dtype(np.float64)
        or pooled_source.shape != fine_source.shape
        or not np.array_equal(fine_source, pooled_source)
    ):
        raise V3SelectionError(
            "fine and pooled v3 risks must use the same binary64 source vector",
            failure_code="v3_risk_source_mismatch",
        )
    replay = aggregate_confirmation_improvements(
        sample_keys=sample_keys,
        row_path_ids=row_path_ids,
        outer_steps=outer_steps,
        phases=phases,
        midpoint_indices=midpoint_indices,
        combined_vs_zero_improvements=fine_source,
        combined_vs_baseline_improvements=pooled_source,
        expected_path_ids=expected_path_ids,
        selected_outer_steps=selected_outer_steps,
    )
    for quartile in range(TIME_QUARTILES):
        left = quartile * PHASE_COUNT * MIDPOINT_COUNT
        right = left + PHASE_COUNT * MIDPOINT_COUNT
        numerators = np.sum(
            replay.path_values[:, left:right] * replay.cell_counts[:, left:right],
            axis=1,
            dtype=np.float64,
        )
        denominators = np.sum(
            replay.cell_counts[:, left:right], axis=1, dtype=np.int64
        )
        reconstructed = numerators / denominators
        pooled = replay.path_values[:, COMBINED_VS_ZERO_FAMILY_SIZE + quartile]
        scale = np.maximum(1.0, np.maximum(np.abs(reconstructed), np.abs(pooled)))
        if np.any(np.abs(reconstructed - pooled) > 32.0 * np.finfo(np.float64).eps * scale):
            raise V3SelectionError(
                "pooled quartile is inconsistent with its fine-cell aggregate",
                failure_code="v3_pooled_aggregation_mismatch",
            )
    return ZeroBaselineRiskTableV3(
        path_ids=np.array(replay.path_ids, copy=True),
        path_values=np.array(replay.path_values, copy=True),
        cell_counts=np.array(replay.cell_counts, copy=True),
        sample_key_sha256=replay.sample_key_sha256,
        row_count=int(fine_source.size),
    )


def aggregate_zero_baseline_risks(
    *,
    sample_keys: Any,
    row_path_ids: Any,
    outer_steps: Any,
    phases: Any,
    midpoint_indices: Any,
    targets: Any,
    predictions: Any,
    expected_path_ids: Any,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
) -> ZeroBaselineRiskTableV3:
    """Aggregate direct raw-MSE improvement ``R(0)-R(model)``."""

    target = np.asarray(targets)
    prediction = np.asarray(predictions)
    if (
        target.dtype != np.dtype(np.float64)
        or target.ndim != 2
        or prediction.dtype != np.dtype(np.float64)
        or prediction.shape != target.shape
        or not np.isfinite(target).all()
        or not np.isfinite(prediction).all()
    ):
        raise V3SelectionError("targets and predictions must be equal-shape binary64 tables")
    improvement = np.mean(
        target * target - (target - prediction) ** 2,
        axis=1,
        dtype=np.float64,
    )
    if not np.isfinite(improvement).all():
        raise V3SelectionError("rowwise v3 risk improvement is nonfinite")
    return aggregate_zero_baseline_improvements(
        sample_keys=sample_keys,
        row_path_ids=row_path_ids,
        outer_steps=outer_steps,
        phases=phases,
        midpoint_indices=midpoint_indices,
        model_vs_zero_improvements=np.ascontiguousarray(improvement),
        expected_path_ids=expected_path_ids,
        selected_outer_steps=selected_outer_steps,
    )


@dataclass(frozen=True)
class CandidateValidationTableV3:
    """Canonical ``[path,120,228]`` validation evidence table."""

    seeds: np.ndarray
    updates: np.ndarray
    path_ids: np.ndarray
    path_values: np.ndarray

    def __post_init__(self) -> None:
        seeds = np.asarray(self.seeds)
        updates = np.asarray(self.updates)
        paths = np.asarray(self.path_ids)
        values = np.asarray(self.path_values)
        if (
            seeds.dtype != np.dtype(np.int64)
            or seeds.shape != (V3_CANDIDATE_COUNT,)
            or updates.dtype != np.dtype(np.int64)
            or updates.shape != seeds.shape
            or paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or paths.size != V3_VALIDATION_PATH_COUNT
            or np.any(paths < 0)
            or np.any(paths >= 1 << 20)
            or np.unique(paths).size != paths.size
            or values.dtype != np.dtype(np.float64)
            or values.shape != (paths.size, V3_CANDIDATE_COUNT, V3_COMPONENT_COUNT)
            or not np.isfinite(values).all()
        ):
            raise V3SelectionError("candidate validation table is malformed")
        pairs = tuple(zip(seeds.tolist(), updates.tolist(), strict=True))
        expected_pairs = tuple(
            (seed, update)
            for seed in V3_MODEL_SEEDS
            for update in V3_NONZERO_UPDATES
        )
        if len(set(pairs)) != V3_CANDIDATE_COUNT or set(pairs) != set(expected_pairs):
            raise V3SelectionError("candidate validation grid changed")
        candidate_order = np.lexsort((updates, seeds))
        path_order = np.argsort(paths, kind="stable")
        object.__setattr__(self, "seeds", _readonly(seeds[candidate_order]))
        object.__setattr__(self, "updates", _readonly(updates[candidate_order]))
        object.__setattr__(self, "path_ids", _readonly(paths[path_order]))
        object.__setattr__(
            self,
            "path_values",
            _readonly(values[path_order][:, candidate_order, :]),
        )

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    @property
    def candidate_count(self) -> int:
        return V3_CANDIDATE_COUNT

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(
            {
                "seeds": self.seeds.tolist(),
                "updates": self.updates.tolist(),
                "path_ids": self.path_ids.tolist(),
                "path_values_sha256": _array_sha256(self.path_values),
                "family_names_sha256": V3_FAMILY_NAMES_SHA256,
            }
        )

    def candidate_index(self, seed: int, update: int) -> int:
        matches = np.flatnonzero(
            (self.seeds == int(seed)) & (self.updates == int(update))
        )
        if matches.size != 1:
            raise V3SelectionError("candidate is absent or duplicated")
        return int(matches[0])

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA + "-candidate-table",
            "schema_version": 1,
            "shape": list(self.path_values.shape),
            "path_ids": self.path_ids.tolist(),
            "seeds": self.seeds.tolist(),
            "updates": self.updates.tolist(),
            "candidate_count": self.candidate_count,
            "component_count": V3_COMPONENT_COUNT,
            "search_family_size": V3_SEARCH_FAMILY_SIZE,
            "family_names": list(V3_FAMILY_NAMES),
            "family_names_sha256": V3_FAMILY_NAMES_SHA256,
            "fingerprint": self.fingerprint,
        }


def build_candidate_validation_table_v3(
    *,
    seeds: Any,
    updates: Any,
    path_ids: Any,
    path_values: Any,
    forbidden_path_ids: Any | None = None,
) -> CandidateValidationTableV3:
    """Build the canonical v3 table and enforce the confirmation firewall."""

    raw_seeds = np.asarray(seeds)
    raw_updates = np.asarray(updates)
    raw_paths = np.asarray(path_ids)
    if raw_seeds.dtype.kind not in "iu" or raw_updates.dtype.kind not in "iu":
        raise V3SelectionError("candidate seeds and updates must be integer vectors")
    if raw_paths.dtype.kind not in "iu":
        raise V3SelectionError("candidate path IDs must be integers")
    canonical_paths = np.asarray(raw_paths, dtype=np.int64)
    if forbidden_path_ids is not None:
        forbidden = np.asarray(forbidden_path_ids)
        if forbidden.ndim != 1 or forbidden.dtype.kind not in "iu":
            raise V3SelectionError("forbidden path IDs are malformed")
        if np.intersect1d(canonical_paths, np.asarray(forbidden, dtype=np.int64)).size:
            raise V3SelectionError(
                "confirmation path IDs entered validation evaluation",
                failure_code="confirmation_path_firewall_violated",
            )
    return CandidateValidationTableV3(
        seeds=np.asarray(raw_seeds, dtype=np.int64),
        updates=np.asarray(raw_updates, dtype=np.int64),
        path_ids=canonical_paths,
        path_values=np.asarray(path_values),
    )


def bootstrap_environment_record() -> dict[str, Any]:
    """Return the frozen CPU/BLAS and NumPy binding for count shards."""

    try:
        numpy_config = np.__config__.show(mode="dicts")
    except TypeError:  # pragma: no cover - supported production NumPy has mode="dicts"
        numpy_config = {"legacy_show_only": 1}
    environment = {
        name: os.environ.get(name)
        for name in (
            "PYTHONHASHSEED",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    cpu_blas = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "numpy_configuration": numpy_config,
        "thread_environment": environment,
    }
    return {
        "numpy_version": np.__version__,
        "philox_constructor": PHILOX_CONSTRUCTOR,
        "byte_order": sys.byteorder,
        "cpu_blas_environment_sha256": config_fingerprint(cpu_blas),
        "cpu_blas_environment": cpu_blas,
    }


def v3_bootstrap_plan(
    *,
    seed: int,
    namespace: int,
    path_count: int,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_BOOTSTRAP_SHARD_SIZE,
    candidate_block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE,
    component_block_size: int = DEFAULT_COMPONENT_BLOCK_SIZE,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete prospective numeric/search design binding."""

    if (
        not isinstance(path_count, int)
        or isinstance(path_count, bool)
        or path_count < 8
        or path_count > np.iinfo(np.uint8).max
        or not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates <= 0
        or not isinstance(shard_size, int)
        or isinstance(shard_size, bool)
        or shard_size <= 0
        or replicates % shard_size != 0
        or not isinstance(candidate_block_size, int)
        or isinstance(candidate_block_size, bool)
        or candidate_block_size <= 0
        or not isinstance(component_block_size, int)
        or isinstance(component_block_size, bool)
        or component_block_size <= 0
    ):
        raise V3SelectionError("bootstrap plan is invalid")
    binding = dict(bootstrap_environment_record() if environment is None else environment)
    body = {
        "schema": SCHEMA + "-bootstrap-plan",
        "schema_version": 1,
        "seed": int(seed),
        "namespace": int(namespace),
        "path_count": int(path_count),
        "replicates": int(replicates),
        "shard_size": int(shard_size),
        "shard_count": int(replicates // shard_size),
        "candidate_block_size": int(candidate_block_size),
        "component_block_size": int(component_block_size),
        "working_family_per_block": int(
            candidate_block_size * component_block_size
        ),
        "confidence": DEFAULT_CONFIDENCE,
        "quantile_method": "higher",
        "negative_values_truncated": 0,
        "standard_error_floor_used": 0,
        "philox_constructor": PHILOX_CONSTRUCTOR,
        "environment": binding,
        "environment_sha256": config_fingerprint(binding),
        "component_family_names_sha256": V3_FAMILY_NAMES_SHA256,
        "search_family_names_sha256": V3_SEARCH_FAMILY_NAMES_SHA256,
        "search_flattening_order": "candidate_major_then_component",
    }
    return {**body, "semantic_sha256": config_fingerprint(body)}


def generate_bootstrap_count_shard(
    *,
    seed: int,
    namespace: int,
    shard_index: int,
    path_count: int,
    shard_size: int = DEFAULT_BOOTSTRAP_SHARD_SIZE,
) -> np.ndarray:
    """Generate one stateless Philox whole-path multiplicity shard."""

    integers = (seed, namespace, shard_index, path_count, shard_size)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in integers
    ) or path_count < 2 or path_count > np.iinfo(np.uint8).max or shard_size <= 0:
        raise V3SelectionError("bootstrap count-shard configuration is invalid")
    generator = np.random.Generator(
        np.random.Philox([int(seed), int(namespace), int(shard_index)])
    )
    indices = generator.integers(
        0,
        int(path_count),
        size=(int(shard_size), int(path_count)),
        dtype=np.int64,
    )
    counts = np.zeros((int(shard_size), int(path_count)), dtype=np.uint8)
    row_indices = np.broadcast_to(
        np.arange(int(shard_size), dtype=np.int64)[:, None], indices.shape
    )
    np.add.at(counts, (row_indices, indices), 1)
    if np.any(np.sum(counts, axis=1, dtype=np.int64) != int(path_count)):
        raise V3SelectionError("bootstrap multiplicity rows do not sum to path count")
    return np.ascontiguousarray(counts)


def _validate_count_shard(
    counts: Any,
    *,
    path_count: int,
    shard_size: int | None = None,
) -> np.ndarray:
    source = np.asarray(counts)
    if (
        source.dtype != np.dtype(np.uint8)
        or source.ndim != 2
        or source.shape[1] != int(path_count)
        or (shard_size is not None and source.shape[0] != int(shard_size))
        or np.any(np.sum(source, axis=1, dtype=np.int64) != int(path_count))
    ):
        raise V3SelectionError("bootstrap count shard is malformed")
    return np.ascontiguousarray(source)


@dataclass(frozen=True)
class NumericMaxTResult:
    """Numeric result shared by validation search and final confirmation."""

    path_ids: np.ndarray
    point_estimates: np.ndarray
    standard_errors: np.ndarray
    lower_bounds: np.ndarray
    maxima: np.ndarray
    critical_value: float
    confidence: float

    def __post_init__(self) -> None:
        paths = np.asarray(self.path_ids)
        point = np.asarray(self.point_estimates)
        error = np.asarray(self.standard_errors)
        lower = np.asarray(self.lower_bounds)
        maxima = np.asarray(self.maxima)
        if (
            paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or np.unique(paths).size != paths.size
            or point.dtype != np.dtype(np.float64)
            or point.ndim != 2
            or point.shape[1] != V3_COMPONENT_COUNT
            or error.dtype != np.dtype(np.float64)
            or error.shape != point.shape
            or lower.dtype != np.dtype(np.float64)
            or lower.shape != point.shape
            or maxima.dtype != np.dtype(np.float64)
            or maxima.ndim != 1
            or maxima.size <= 0
            or not np.isfinite(point).all()
            or not np.isfinite(error).all()
            or np.any(error <= 0.0)
            or not np.isfinite(lower).all()
            or not np.isfinite(maxima).all()
            or not math.isfinite(float(self.critical_value))
            or not 0.5 < float(self.confidence) < 1.0
        ):
            raise V3SelectionError("numeric max-T result is malformed")
        object.__setattr__(self, "path_ids", _readonly(paths))
        object.__setattr__(self, "point_estimates", _readonly(point))
        object.__setattr__(self, "standard_errors", _readonly(error))
        object.__setattr__(self, "lower_bounds", _readonly(lower))
        object.__setattr__(self, "maxima", _readonly(maxima))

    @property
    def candidate_count(self) -> int:
        return int(self.point_estimates.shape[0])

    @property
    def passed(self) -> bool:
        return bool(np.all(self.lower_bounds > 0.0))

    def to_record(
        self,
        *,
        seeds: Sequence[int] | None = None,
        updates: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": NUMERIC_MAX_T_SCHEMA,
            "schema_version": 1,
            "method": "centered_whole_path_one_sided_studentized_max_t",
            "bootstrap_unit": "whole_path_jointly_across_candidates_and_components",
            "quantile_method": "higher",
            "family_names": list(V3_FAMILY_NAMES),
            "family_names_sha256": V3_FAMILY_NAMES_SHA256,
            "component_count": V3_COMPONENT_COUNT,
            "candidate_count": self.candidate_count,
            "search_family_size": self.candidate_count * V3_COMPONENT_COUNT,
            "path_ids": self.path_ids.tolist(),
            "path_count": int(self.path_ids.size),
            "confidence": float(self.confidence),
            "replicates": int(self.maxima.size),
            "critical_value": float(self.critical_value),
            "minimum_lower_bound": float(np.min(self.lower_bounds)),
            "all_lower_bounds_strictly_positive": int(self.passed),
            "negative_values_truncated": 0,
            "standard_error_floor_used": 0,
            "point_estimates_sha256": _array_sha256(self.point_estimates),
            "standard_errors_sha256": _array_sha256(self.standard_errors),
            "lower_bounds_sha256": _array_sha256(self.lower_bounds),
            "maxima_sha256": _array_sha256(self.maxima),
        }
        if seeds is not None or updates is not None:
            if seeds is None or updates is None:
                raise V3SelectionError("candidate identities are incomplete")
            seed_values = tuple(int(value) for value in seeds)
            update_values = tuple(int(value) for value in updates)
            if len(seed_values) != self.candidate_count or len(update_values) != len(seed_values):
                raise V3SelectionError("candidate identities have the wrong length")
            record["seeds"] = list(seed_values)
            record["updates"] = list(update_values)
        return record


def _canonical_numeric_values(
    values: Any,
    path_ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values)
    paths = _canonical_path_ids(path_ids)
    if source.dtype != np.dtype(np.float64):
        raise V3SelectionError("max-T path values must be binary64")
    if source.ndim == 2:
        source = source[:, None, :]
    if (
        source.ndim != 3
        or source.shape[0] != paths.size
        or source.shape[1] <= 0
        or source.shape[2] != V3_COMPONENT_COUNT
        or not np.isfinite(source).all()
    ):
        raise V3SelectionError("max-T values must be finite [path,candidate,228]")
    order = np.argsort(paths, kind="stable")
    return np.ascontiguousarray(paths[order]), np.ascontiguousarray(source[order])


def _observed_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path_count = int(values.shape[0])
    point = np.mean(values, axis=0, dtype=np.float64)
    standard_error = np.std(values, axis=0, ddof=1, dtype=np.float64) / math.sqrt(
        path_count
    )
    if (
        not np.isfinite(point).all()
        or not np.isfinite(standard_error).all()
        or np.any(standard_error <= 0.0)
    ):
        raise V3SelectionError(
            "max-T family has degenerate/nonfinite observed studentization",
            failure_code="max_t_studentization_invalid",
        )
    return np.ascontiguousarray(point), np.ascontiguousarray(standard_error)


def compute_bootstrap_maxima_shard(
    values: Any,
    counts: Any,
    *,
    path_ids: Any,
    candidate_block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE,
    component_block_size: int = DEFAULT_COMPONENT_BLOCK_SIZE,
) -> np.ndarray:
    """Compute one shared-count maxima shard with bounded working memory."""

    paths, table = _canonical_numeric_values(values, path_ids)
    del paths
    path_count, candidate_count, component_count = table.shape
    count_table = _validate_count_shard(counts, path_count=path_count)
    if (
        not isinstance(candidate_block_size, int)
        or isinstance(candidate_block_size, bool)
        or candidate_block_size <= 0
        or not isinstance(component_block_size, int)
        or isinstance(component_block_size, bool)
        or component_block_size <= 0
    ):
        raise V3SelectionError("max-T block sizes are invalid")
    point, _ = _observed_statistics(table)
    draw_count = int(count_table.shape[0])
    maxima = np.full(draw_count, -np.inf, dtype=np.float64)
    count_float = np.asarray(count_table, dtype=np.float64)
    largest_family_block = min(candidate_count, candidate_block_size) * min(
        component_count, component_block_size
    )
    estimated_working_bytes = (
        count_float.nbytes
        + 2 * path_count * largest_family_block * np.dtype(np.float64).itemsize
        + 3 * draw_count * largest_family_block * np.dtype(np.float64).itemsize
        + maxima.nbytes
    )
    if estimated_working_bytes >= MAXIMUM_WORKING_BYTES:
        raise V3SelectionError(
            "max-T block plan exceeds the frozen working-memory target",
            failure_code="max_t_working_memory_invalid",
        )
    for candidate_left in range(0, candidate_count, candidate_block_size):
        candidate_right = min(candidate_count, candidate_left + candidate_block_size)
        for component_left in range(0, component_count, component_block_size):
            component_right = min(
                component_count, component_left + component_block_size
            )
            block = np.ascontiguousarray(
                table[
                    :,
                    candidate_left:candidate_right,
                    component_left:component_right,
                ].reshape(path_count, -1)
            )
            observed = point[
                candidate_left:candidate_right,
                component_left:component_right,
            ].reshape(-1)
            draw_mean = (count_float @ block) / path_count
            draw_second = (count_float @ (block * block)) / path_count
            draw_second -= draw_mean * draw_mean
            draw_second *= path_count
            draw_second /= path_count - 1
            draw_second /= path_count
            np.sqrt(draw_second, out=draw_second)
            if not np.isfinite(draw_second).all() or np.any(draw_second <= 0.0):
                raise V3SelectionError(
                    "bootstrap produced degenerate/nonfinite studentization",
                    failure_code="max_t_bootstrap_studentization_invalid",
                )
            draw_mean -= observed[None, :]
            draw_mean /= draw_second
            maxima = np.maximum(maxima, np.max(draw_mean, axis=1))
    if not np.isfinite(maxima).all():
        raise V3SelectionError("bootstrap maxima are nonfinite")
    return np.ascontiguousarray(maxima)


def numeric_v3_max_t(
    values: Any,
    *,
    path_ids: Any,
    count_shards: Sequence[Any],
    confidence: float = DEFAULT_CONFIDENCE,
    candidate_block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE,
    component_block_size: int = DEFAULT_COMPONENT_BLOCK_SIZE,
) -> NumericMaxTResult:
    """Run the name-agnostic numeric 228-component max-T core."""

    if not 0.5 < float(confidence) < 1.0 or not count_shards:
        raise V3SelectionError("numeric max-T configuration is invalid")
    paths, table = _canonical_numeric_values(values, path_ids)
    point, standard_error = _observed_statistics(table)
    maxima_parts = [
        compute_bootstrap_maxima_shard(
            table,
            counts,
            path_ids=paths,
            candidate_block_size=candidate_block_size,
            component_block_size=component_block_size,
        )
        for counts in count_shards
    ]
    maxima = np.ascontiguousarray(np.concatenate(maxima_parts))
    critical = float(np.quantile(maxima, float(confidence), method="higher"))
    if not math.isfinite(critical):
        raise V3SelectionError("max-T critical value is nonfinite")
    lower = point - critical * standard_error
    return NumericMaxTResult(
        path_ids=paths,
        point_estimates=point,
        standard_errors=standard_error,
        lower_bounds=np.ascontiguousarray(lower),
        maxima=maxima,
        critical_value=critical,
        confidence=float(confidence),
    )


def search_aware_validation_max_t(
    table: CandidateValidationTableV3,
    *,
    count_shards: Sequence[Any],
    confidence: float = DEFAULT_CONFIDENCE,
    candidate_block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE,
    component_block_size: int = DEFAULT_COMPONENT_BLOCK_SIZE,
) -> tuple[NumericMaxTResult, dict[str, Any]]:
    """Run an in-memory search-aware validation audit and rank its nominee."""

    if not isinstance(table, CandidateValidationTableV3):
        raise V3SelectionError("validation search requires CandidateValidationTableV3")
    result = numeric_v3_max_t(
        table.path_values,
        path_ids=table.path_ids,
        count_shards=count_shards,
        confidence=confidence,
        candidate_block_size=candidate_block_size,
        component_block_size=component_block_size,
    )
    selection = rank_validation_nominee(table, result)
    selection.update(
        {
            "critical_value": float(result.critical_value),
            "confidence": float(result.confidence),
            "replicates": int(result.maxima.size),
            "family_names_sha256": V3_FAMILY_NAMES_SHA256,
            "search_family_names_sha256": V3_SEARCH_FAMILY_NAMES_SHA256,
            "candidate_table_fingerprint": table.fingerprint,
            "maxima_sha256": _array_sha256(result.maxima),
            "lower_bounds_sha256": _array_sha256(result.lower_bounds),
        }
    )
    return result, selection


def one_sided_v3_confirmation_max_t(
    path_values: Any,
    *,
    path_ids: Any,
    count_shards: Sequence[Any],
    confidence: float = DEFAULT_CONFIDENCE,
    candidate_block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE,
    component_block_size: int = DEFAULT_COMPONENT_BLOCK_SIZE,
) -> dict[str, Any]:
    """Apply the shared numeric core to one fresh confirmation candidate."""

    result = numeric_v3_max_t(
        path_values,
        path_ids=path_ids,
        count_shards=count_shards,
        confidence=confidence,
        candidate_block_size=candidate_block_size,
        component_block_size=component_block_size,
    )
    return v3_confirmation_max_t_record(result)


def v3_confirmation_max_t_record(result: NumericMaxTResult) -> dict[str, Any]:
    """Attach frozen v3 names to a one-candidate numeric result."""

    if not isinstance(result, NumericMaxTResult):
        raise V3SelectionError("confirmation record requires NumericMaxTResult")
    if result.candidate_count != 1:
        raise V3SelectionError("confirmation max-T requires exactly one candidate")
    point = result.point_estimates[0]
    error = result.standard_errors[0]
    lower = result.lower_bounds[0]
    record = result.to_record()
    record.update(
        {
            "point_estimates": {
                name: float(value)
                for name, value in zip(V3_FAMILY_NAMES, point, strict=True)
            },
            "standard_errors": {
                name: float(value)
                for name, value in zip(V3_FAMILY_NAMES, error, strict=True)
            },
            "lower_bounds": {
                name: float(value)
                for name, value in zip(V3_FAMILY_NAMES, lower, strict=True)
            },
            "passed": int(np.all(lower > 0.0)),
        }
    )
    return record


def rank_validation_nominee(
    table: CandidateValidationTableV3,
    result: NumericMaxTResult,
) -> dict[str, Any]:
    """Rank eligible candidates by maximin lower bound, update, then seed."""

    if not isinstance(table, CandidateValidationTableV3) or not isinstance(
        result, NumericMaxTResult
    ):
        raise V3SelectionError("nominee ranking requires canonical evidence")
    if (
        result.candidate_count != table.candidate_count
        or not np.array_equal(result.path_ids, table.path_ids)
    ):
        raise V3SelectionError("nominee ranking evidence does not match candidate table")
    minimum = np.min(result.lower_bounds, axis=1)
    eligible = np.all(result.lower_bounds > 0.0, axis=1)
    rows = [
        {
            "seed": int(table.seeds[index]),
            "update": int(table.updates[index]),
            "minimum_lower_bound": float(minimum[index]),
            "all_228_lower_bounds_strictly_positive": int(eligible[index]),
            "positive_lower_bound_count": int(
                np.count_nonzero(result.lower_bounds[index] > 0.0)
            ),
        }
        for index in range(table.candidate_count)
    ]
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        return {
            "schema": SCHEMA + "-nominee-ranking",
            "schema_version": 1,
            "decision": "no_validation_candidate",
            "candidate_count": table.candidate_count,
            "eligible_candidate_count": 0,
            "selected_seed": None,
            "selected_update": 0,
            "logical_update_zero_selected": 1,
            "confirmation_authorized": 0,
            "candidate_rows": rows,
        }
    selected = min(
        eligible_indices.tolist(),
        key=lambda index: (
            -float(minimum[index]),
            int(table.updates[index]),
            int(table.seeds[index]),
        ),
    )
    return {
        "schema": SCHEMA + "-nominee-ranking",
        "schema_version": 1,
        "decision": "zero_baseline_v3_validation_nominee_sealed",
        "candidate_count": table.candidate_count,
        "eligible_candidate_count": int(eligible_indices.size),
        "selected_seed": int(table.seeds[selected]),
        "selected_update": int(table.updates[selected]),
        "selected_minimum_lower_bound": float(minimum[selected]),
        "logical_update_zero_selected": 0,
        "confirmation_authorized": 1,
        "ranking_rule": [
            "largest_minimum_lower_bound",
            "earlier_update",
            "lower_seed",
        ],
        "candidate_rows": rows,
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                **{
                    str(name): np.ascontiguousarray(value)
                    for name, value in arrays.items()
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_npz_array(path: Path, name: str) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != (name,):
                raise V3SelectionError(f"{path.name} has an unexpected NPZ schema")
            return np.array(archive[name], copy=True)
    except (OSError, ValueError, KeyError) as exc:
        raise V3SelectionError(f"cannot load committed shard {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V3SelectionError(f"cannot load committed metadata {path}") from exc
    if not isinstance(value, dict):
        raise V3SelectionError(f"metadata is not an object: {path}")
    return value


def _semantic_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _validate_semantic_record(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "semantic_sha256"}
    if value.get("semantic_sha256") != config_fingerprint(body):
        raise V3SelectionError("shard metadata semantic hash changed")


def shard_artifact_paths(directory: str | Path, shard_index: int) -> tuple[Path, Path]:
    if not isinstance(shard_index, int) or isinstance(shard_index, bool) or shard_index < 0:
        raise V3SelectionError("shard index is invalid")
    root = Path(directory)
    stem = f"shard-{int(shard_index):05d}"
    return root / f"{stem}.npz", root / f"{stem}.metadata.json"


def _count_metadata(
    *,
    path: Path,
    counts: np.ndarray,
    seed: int,
    namespace: int,
    shard_index: int,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return _semantic_record(
        {
            "schema": BOOTSTRAP_COUNTS_SCHEMA,
            "schema_version": 1,
            "seed": int(seed),
            "namespace": int(namespace),
            "shard_index": int(shard_index),
            "path_count": int(counts.shape[1]),
            "shard_size": int(counts.shape[0]),
            "dtype": counts.dtype.str,
            "row_sum": int(counts.shape[1]),
            "counts_sha256": _array_sha256(counts),
            "artifact_sha256": file_fingerprint(path),
            "artifact_size": int(path.stat().st_size),
            "environment": dict(environment),
            "environment_sha256": config_fingerprint(dict(environment)),
            "philox_constructor": PHILOX_CONSTRUCTOR,
        }
    )


def prepare_bootstrap_count_shards(
    directory: str | Path,
    *,
    seed: int,
    namespace: int,
    path_count: int,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_BOOTSTRAP_SHARD_SIZE,
    allow_repair: bool = True,
    environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Commit or verify prospective count shards before outcome labels open."""

    if (
        not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates <= 0
        or not isinstance(shard_size, int)
        or isinstance(shard_size, bool)
        or shard_size <= 0
        or replicates % shard_size != 0
    ):
        raise V3SelectionError("replicate and shard counts are invalid")
    root = Path(directory)
    if not root.exists():
        if not allow_repair:
            raise V3SelectionError("required committed count-shard directory is missing")
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise V3SelectionError("count-shard path is not a directory")
    binding = dict(bootstrap_environment_record() if environment is None else environment)
    records: list[dict[str, Any]] = []
    for shard_index in range(replicates // shard_size):
        data_path, metadata_path = shard_artifact_paths(root, shard_index)
        if metadata_path.exists():
            try:
                record = _load_json(metadata_path)
                _validate_semantic_record(record)
                counts = _load_npz_array(data_path, "counts")
                _validate_count_shard(
                    counts, path_count=path_count, shard_size=shard_size
                )
                expected = {
                    "schema": BOOTSTRAP_COUNTS_SCHEMA,
                    "seed": int(seed),
                    "namespace": int(namespace),
                    "shard_index": shard_index,
                    "path_count": int(path_count),
                    "shard_size": int(shard_size),
                    "counts_sha256": _array_sha256(counts),
                    "artifact_sha256": file_fingerprint(data_path),
                    "environment_sha256": config_fingerprint(binding),
                }
                if any(record.get(key) != value for key, value in expected.items()):
                    raise V3SelectionError("committed count-shard binding changed")
                records.append(record)
                continue
            except (OSError, V3SelectionError):
                if not allow_repair:
                    raise
        elif not allow_repair and (data_path.exists() or not metadata_path.exists()):
            raise V3SelectionError("required committed count shard is missing")
        counts = generate_bootstrap_count_shard(
            seed=seed,
            namespace=namespace,
            shard_index=shard_index,
            path_count=path_count,
            shard_size=shard_size,
        )
        _atomic_npz(data_path, counts=counts)
        record = _count_metadata(
            path=data_path,
            counts=counts,
            seed=seed,
            namespace=namespace,
            shard_index=shard_index,
            environment=binding,
        )
        atomic_write_json(metadata_path, record)
        records.append(record)
    return records


def load_bootstrap_count_shards(
    directory: str | Path,
    *,
    seed: int,
    namespace: int,
    path_count: int,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_BOOTSTRAP_SHARD_SIZE,
    environment: Mapping[str, Any] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Load count shards without any regeneration authority."""

    records = prepare_bootstrap_count_shards(
        directory,
        seed=seed,
        namespace=namespace,
        path_count=path_count,
        replicates=replicates,
        shard_size=shard_size,
        allow_repair=False,
        environment=environment,
    )
    arrays = [
        _load_npz_array(shard_artifact_paths(directory, index)[0], "counts")
        for index in range(len(records))
    ]
    return arrays, records


def _maxima_metadata(
    *,
    path: Path,
    maxima: np.ndarray,
    shard_index: int,
    count_record: Mapping[str, Any],
    values_sha256: str,
    candidate_block_size: int,
    component_block_size: int,
) -> dict[str, Any]:
    return _semantic_record(
        {
            "schema": BOOTSTRAP_MAXIMA_SCHEMA,
            "schema_version": 1,
            "shard_index": int(shard_index),
            "replicate_count": int(maxima.size),
            "dtype": maxima.dtype.str,
            "maxima_sha256": _array_sha256(maxima),
            "artifact_sha256": file_fingerprint(path),
            "artifact_size": int(path.stat().st_size),
            "count_metadata_semantic_sha256": count_record["semantic_sha256"],
            "count_artifact_sha256": count_record["artifact_sha256"],
            "values_sha256": values_sha256,
            "candidate_block_size": int(candidate_block_size),
            "component_block_size": int(component_block_size),
        }
    )


def restartable_numeric_v3_max_t(
    values: Any,
    *,
    path_ids: Any,
    count_directory: str | Path,
    maxima_directory: str | Path,
    seed: int,
    namespace: int,
    confidence: float = DEFAULT_CONFIDENCE,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_BOOTSTRAP_SHARD_SIZE,
    candidate_block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE,
    component_block_size: int = DEFAULT_COMPONENT_BLOCK_SIZE,
    environment: Mapping[str, Any] | None = None,
) -> tuple[NumericMaxTResult, list[dict[str, Any]], list[dict[str, Any]]]:
    """Resume maxima shards from immutable committed count shards."""

    paths, table = _canonical_numeric_values(values, path_ids)
    count_arrays, count_records = load_bootstrap_count_shards(
        count_directory,
        seed=seed,
        namespace=namespace,
        path_count=int(paths.size),
        replicates=replicates,
        shard_size=shard_size,
        environment=environment,
    )
    maxima_root = Path(maxima_directory)
    maxima_root.mkdir(parents=True, exist_ok=True)
    values_sha256 = config_fingerprint(
        {
            "path_ids": paths.tolist(),
            "values_sha256": _array_sha256(table),
            "family_names_sha256": V3_FAMILY_NAMES_SHA256,
        }
    )
    maxima_arrays: list[np.ndarray] = []
    maxima_records: list[dict[str, Any]] = []
    for shard_index, (counts, count_record) in enumerate(
        zip(count_arrays, count_records, strict=True)
    ):
        data_path, metadata_path = shard_artifact_paths(maxima_root, shard_index)
        if metadata_path.exists():
            record = _load_json(metadata_path)
            _validate_semantic_record(record)
            maxima = _load_npz_array(data_path, "maxima")
            expected = {
                "schema": BOOTSTRAP_MAXIMA_SCHEMA,
                "shard_index": shard_index,
                "replicate_count": int(counts.shape[0]),
                "maxima_sha256": _array_sha256(maxima),
                "artifact_sha256": file_fingerprint(data_path),
                "count_metadata_semantic_sha256": count_record["semantic_sha256"],
                "count_artifact_sha256": count_record["artifact_sha256"],
                "values_sha256": values_sha256,
                "candidate_block_size": int(candidate_block_size),
                "component_block_size": int(component_block_size),
            }
            if (
                maxima.dtype != np.dtype(np.float64)
                or maxima.shape != (counts.shape[0],)
                or not np.isfinite(maxima).all()
                or any(record.get(key) != value for key, value in expected.items())
            ):
                raise V3SelectionError("committed maxima shard changed")
        else:
            maxima = compute_bootstrap_maxima_shard(
                table,
                counts,
                path_ids=paths,
                candidate_block_size=candidate_block_size,
                component_block_size=component_block_size,
            )
            _atomic_npz(data_path, maxima=maxima)
            record = _maxima_metadata(
                path=data_path,
                maxima=maxima,
                shard_index=shard_index,
                count_record=count_record,
                values_sha256=values_sha256,
                candidate_block_size=candidate_block_size,
                component_block_size=component_block_size,
            )
            atomic_write_json(metadata_path, record)
        maxima_arrays.append(np.ascontiguousarray(maxima))
        maxima_records.append(record)
    point, standard_error = _observed_statistics(table)
    maxima = np.ascontiguousarray(np.concatenate(maxima_arrays))
    critical = float(np.quantile(maxima, float(confidence), method="higher"))
    result = NumericMaxTResult(
        path_ids=paths,
        point_estimates=point,
        standard_errors=standard_error,
        lower_bounds=np.ascontiguousarray(point - critical * standard_error),
        maxima=maxima,
        critical_value=critical,
        confidence=float(confidence),
    )
    return result, count_records, maxima_records


def restartable_validation_search_max_t(
    table: CandidateValidationTableV3,
    *,
    count_directory: str | Path,
    maxima_directory: str | Path,
    seed: int = DEFAULT_SELECTION_SEED,
    namespace: int = SELECTION_NAMESPACE,
    confidence: float = DEFAULT_CONFIDENCE,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_BOOTSTRAP_SHARD_SIZE,
    candidate_block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE,
    component_block_size: int = DEFAULT_COMPONENT_BLOCK_SIZE,
    environment: Mapping[str, Any] | None = None,
) -> tuple[NumericMaxTResult, dict[str, Any]]:
    """Run or resume the full 27,360-member validation search."""

    if not isinstance(table, CandidateValidationTableV3):
        raise V3SelectionError("validation search requires CandidateValidationTableV3")
    result, count_records, maxima_records = restartable_numeric_v3_max_t(
        table.path_values,
        path_ids=table.path_ids,
        count_directory=count_directory,
        maxima_directory=maxima_directory,
        seed=seed,
        namespace=namespace,
        confidence=confidence,
        replicates=replicates,
        shard_size=shard_size,
        candidate_block_size=candidate_block_size,
        component_block_size=component_block_size,
        environment=environment,
    )
    selection = rank_validation_nominee(table, result)
    selection.update(
        {
            "critical_value": float(result.critical_value),
            "confidence": float(result.confidence),
            "replicates": int(result.maxima.size),
            "seed": int(seed),
            "namespace": int(namespace),
            "family_names_sha256": V3_FAMILY_NAMES_SHA256,
            "search_family_names_sha256": V3_SEARCH_FAMILY_NAMES_SHA256,
            "candidate_table_fingerprint": table.fingerprint,
            "count_metadata_semantic_sha256": [
                record["semantic_sha256"] for record in count_records
            ],
            "maxima_metadata_semantic_sha256": [
                record["semantic_sha256"] for record in maxima_records
            ],
            "maxima_sha256": _array_sha256(result.maxima),
            "lower_bounds_sha256": _array_sha256(result.lower_bounds),
        }
    )
    return result, selection


__all__ = [
    "BOOTSTRAP_COUNTS_SCHEMA",
    "BOOTSTRAP_MAXIMA_SCHEMA",
    "CONFIRMATION_NAMESPACE",
    "CandidateValidationTableV3",
    "DEFAULT_BOOTSTRAP_SHARD_SIZE",
    "DEFAULT_CANDIDATE_BLOCK_SIZE",
    "DEFAULT_COMPONENT_BLOCK_SIZE",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_CONFIRMATION_SEED",
    "DEFAULT_REPLICATES",
    "DEFAULT_SELECTION_SEED",
    "NUMERIC_MAX_T_SCHEMA",
    "MAXIMUM_WORKING_BYTES",
    "NumericMaxTResult",
    "PHILOX_CONSTRUCTOR",
    "SCHEMA",
    "SELECTION_NAMESPACE",
    "V3_CANDIDATE_COUNT",
    "V3_COMPONENT_COUNT",
    "V3_FAMILY_NAMES",
    "V3_FAMILY_NAMES_SHA256",
    "V3_FINE_FAMILY_NAMES",
    "V3_MODEL_SEEDS",
    "V3_NONZERO_UPDATES",
    "V3_POOLED_FAMILY_NAMES",
    "V3_SEARCH_FAMILY_SIZE",
    "V3_SEARCH_FAMILY_NAMES",
    "V3_SEARCH_FAMILY_NAMES_SHA256",
    "V3_VALIDATION_PATH_COUNT",
    "V3SelectionError",
    "ZeroBaselineRiskTableV3",
    "aggregate_zero_baseline_improvements",
    "aggregate_zero_baseline_risks",
    "bootstrap_environment_record",
    "build_candidate_validation_table_v3",
    "compute_bootstrap_maxima_shard",
    "generate_bootstrap_count_shard",
    "load_bootstrap_count_shards",
    "numeric_v3_max_t",
    "one_sided_v3_confirmation_max_t",
    "prepare_bootstrap_count_shards",
    "rank_validation_nominee",
    "restartable_numeric_v3_max_t",
    "restartable_validation_search_max_t",
    "search_aware_validation_max_t",
    "shard_artifact_paths",
    "v3_bootstrap_plan",
    "v3_confirmation_max_t_record",
    "v3_family_names",
    "v3_search_family_names",
]
