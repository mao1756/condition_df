"""Sealed whole-path inference for the quartile-specialist experiment.

The physical checkpoint and the q2/q3 scalar gains are selected on disjoint
training-only roles.  Consequently each fresh audit contains one inferential
candidate and exactly six pooled contrasts.  This module owns that frozen
family, the non-inferential 224-cell directional screen, and a new uint16
whole-path bootstrap suitable for the preregistered 384-path audits.

Count shards are prospective artifacts.  They are generated from stateless
Philox namespaces before labels open and can subsequently only be loaded.
Maxima shards are derived artifacts and may be reconstructed from sealed
counts and committed path reductions.  No standard-error floor or negative
improvement truncation is used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
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
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import PHASE_COUNT, TIME_QUARTILES
from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE


SCHEMA = "d0-jacobi-rb-boundary-tangent-quartile-selection-v1"
PATH_TABLE_SCHEMA = SCHEMA + "-path-table"
LOCAL_SCREEN_SCHEMA = SCHEMA + "-local-screen"
BOOTSTRAP_PLAN_SCHEMA = SCHEMA + "-bootstrap-plan"
BOOTSTRAP_COUNTS_SCHEMA = SCHEMA + "-bootstrap-count-shard"
BOOTSTRAP_MAXIMA_SCHEMA = SCHEMA + "-bootstrap-maxima-shard"
MAX_T_SCHEMA = SCHEMA + "-max-t"

PRODUCTION_PATH_COUNT = 384
SELECTION_PATH_START = 0xF5000
SELECTION_PATH_STOP = 0xF5180
CONFIRMATION_PATH_START = 0xF7000
CONFIRMATION_PATH_STOP = 0xF7180
PRIMARY_FAMILY_SIZE = 6
LOCAL_FAMILY_SIZE = TIME_QUARTILES * PHASE_COUNT * MIDPOINT_COUNT
LOCAL_CELLS_PER_QUARTILE = PHASE_COUNT * MIDPOINT_COUNT
MINIMUM_POSITIVE_LOCAL_CELLS = 51
Q1_SENTINEL_PHASE = 4
Q1_SENTINEL_MIDPOINT = 7

DEFAULT_CONFIDENCE = 0.995
DEFAULT_REPLICATES = 50_000
DEFAULT_SHARD_SIZE = 1_000
DEFAULT_SELECTION_SEED = 261_350
DEFAULT_CONFIRMATION_SEED = 261_351
SELECTION_NAMESPACE = 0x51545331
CONFIRMATION_NAMESPACE = 0x51544331
PHILOX_CONSTRUCTOR = (
    "np.random.Generator(np.random.Philox([int(seed), int(namespace), "
    "int(shard_index)]))"
)


class QuartileSelectionError(ValueError):
    """The frozen quartile audit or bootstrap contract was violated."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str = "quartile_specialist_selection_inference_invalid",
    ) -> None:
        super().__init__(message)
        self.failure_code = str(failure_code)


def primary_family_names() -> tuple[str, ...]:
    """Return the exact six-member authorizing family in canonical order."""

    names = tuple(
        [f"specialist_vs_zero.q{quartile}.pooled" for quartile in range(4)]
        + ["shrunken_vs_raw.q2.pooled", "shrunken_vs_raw.q3.pooled"]
    )
    if len(names) != PRIMARY_FAMILY_SIZE or len(set(names)) != len(names):
        raise AssertionError("the frozen primary family is malformed")
    return names


def local_family_names() -> tuple[str, ...]:
    """Return the 224 descriptive screen cells in q/phase/midpoint order."""

    names = tuple(
        f"specialist_vs_zero.q{quartile}.phase{phase}.midpoint{midpoint}"
        for quartile in range(TIME_QUARTILES)
        for phase in range(PHASE_COUNT)
        for midpoint in range(MIDPOINT_COUNT)
    )
    if len(names) != LOCAL_FAMILY_SIZE or len(set(names)) != len(names):
        raise AssertionError("the frozen local family is malformed")
    return names


PRIMARY_FAMILY_NAMES = primary_family_names()
PRIMARY_FAMILY_NAMES_SHA256 = config_fingerprint(list(PRIMARY_FAMILY_NAMES))
LOCAL_FAMILY_NAMES = local_family_names()
LOCAL_FAMILY_NAMES_SHA256 = config_fingerprint(list(LOCAL_FAMILY_NAMES))
# Explicit semantic aliases used by manifests and downstream gate code.
SIX_FAMILY_NAMES = PRIMARY_FAMILY_NAMES
SIX_FAMILY_NAMES_SHA256 = PRIMARY_FAMILY_NAMES_SHA256
LOCAL_SCREEN_FAMILY_NAMES = LOCAL_FAMILY_NAMES
LOCAL_SCREEN_FAMILY_NAMES_SHA256 = LOCAL_FAMILY_NAMES_SHA256


def _readonly(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(config_fingerprint(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _semantic_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _validate_semantic_record(value: Mapping[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "semantic_sha256"}
    if value.get("semantic_sha256") != config_fingerprint(body):
        raise QuartileSelectionError("shard metadata semantic hash changed")


def _canonical_path_ids(
    value: Any,
    *,
    expected_count: int | None = None,
) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise QuartileSelectionError("path IDs must be an integer vector")
    paths = np.asarray(source, dtype=np.int64)
    if (
        paths.size < 8
        or np.any(paths < 0)
        or np.any(paths >= 1 << 20)
        or np.unique(paths).size != paths.size
        or (expected_count is not None and paths.size != int(expected_count))
    ):
        raise QuartileSelectionError("path IDs violate the frozen audit plan")
    return np.ascontiguousarray(np.sort(paths, kind="stable"))


def _canonical_selected_steps(value: Sequence[int]) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value)
    if (
        not steps
        or tuple(sorted(steps)) != steps
        or len(set(steps)) != len(steps)
        or any(step < 0 or step >= 512 for step in steps)
    ):
        raise QuartileSelectionError("selected outer-step plan is malformed")
    counts = np.bincount(
        np.asarray([step // 128 for step in steps], dtype=np.int64), minlength=4
    )
    if counts.shape != (4,) or np.any(counts <= 0) or np.unique(counts).size != 1:
        raise QuartileSelectionError(
            "selected outer steps must populate every quartile equally"
        )
    return steps


def expected_audit_sample_key_sha256(
    path_ids: Any,
    *,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
) -> str:
    """Return the deterministic hash of one complete audit Cartesian grid."""

    paths = _canonical_path_ids(path_ids)
    steps = _canonical_selected_steps(selected_outer_steps)
    count = paths.size * len(steps) * PHASE_COUNT * MIDPOINT_COUNT
    keys = np.fromiter(
        (
            midpoint_sample_key(int(path), int(step), phase, midpoint)
            for path in paths
            for step in steps
            for phase in range(PHASE_COUNT)
            for midpoint in range(MIDPOINT_COUNT)
        ),
        dtype=np.int64,
        count=count,
    )
    return hashlib.sha256(
        np.ascontiguousarray(np.sort(keys, kind="stable")).tobytes(order="C")
    ).hexdigest()


def _integer_vector(value: Any, name: str, rows: int | None = None) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise QuartileSelectionError(f"{name} must be an integer vector")
    result = np.asarray(source, dtype=np.int64)
    if rows is not None and result.shape != (rows,):
        raise QuartileSelectionError(f"{name} has the wrong row count")
    return np.ascontiguousarray(result)


def _float_vector(value: Any, name: str, rows: int) -> np.ndarray:
    source = np.asarray(value)
    if (
        source.dtype != np.dtype(np.float64)
        or source.shape != (rows,)
        or not np.isfinite(source).all()
    ):
        raise QuartileSelectionError(f"{name} must be a finite binary64 vector")
    return np.ascontiguousarray(source)


def _float_edge_table(value: Any, name: str, rows: int) -> np.ndarray:
    source = np.asarray(value)
    if (
        source.dtype != np.dtype(np.float64)
        or source.shape != (rows, EDGES_PER_PHASE)
        or not np.isfinite(source).all()
    ):
        raise QuartileSelectionError(
            f"{name} must be finite binary64 [{rows},{EDGES_PER_PHASE}]"
        )
    return np.ascontiguousarray(source)


@dataclass(frozen=True)
class QuartileAuditPathTable:
    """Canonical whole-path primary and local reductions for one audit."""

    path_ids: np.ndarray
    primary_values: np.ndarray
    local_values: np.ndarray
    primary_counts: np.ndarray
    local_counts: np.ndarray
    selected_outer_steps: np.ndarray
    sample_key_sha256: str
    row_count: int

    def __post_init__(self) -> None:
        paths = np.asarray(self.path_ids)
        primary = np.asarray(self.primary_values)
        local = np.asarray(self.local_values)
        primary_counts = np.asarray(self.primary_counts)
        local_counts = np.asarray(self.local_counts)
        selected_steps = np.asarray(self.selected_outer_steps)
        if (
            paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or paths.size < 8
            or np.any(paths < 0)
            or np.any(paths >= 1 << 20)
            or np.unique(paths).size != paths.size
            or not np.array_equal(paths, np.sort(paths, kind="stable"))
            or primary.dtype != np.dtype(np.float64)
            or primary.shape != (paths.size, PRIMARY_FAMILY_SIZE)
            or local.dtype != np.dtype(np.float64)
            or local.shape != (
                paths.size,
                TIME_QUARTILES,
                LOCAL_CELLS_PER_QUARTILE,
            )
            or not np.isfinite(primary).all()
            or not np.isfinite(local).all()
            or primary_counts.dtype != np.dtype(np.int64)
            or primary_counts.shape != primary.shape
            or local_counts.dtype != np.dtype(np.int64)
            or local_counts.shape != local.shape
            or np.any(primary_counts <= 0)
            or np.any(local_counts <= 0)
            or selected_steps.dtype != np.dtype(np.int64)
            or selected_steps.ndim != 1
            or not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count <= 0
            or not isinstance(self.sample_key_sha256, str)
            or len(self.sample_key_sha256) != 64
        ):
            raise QuartileSelectionError("quartile audit path table is malformed")
        canonical_steps = _canonical_selected_steps(selected_steps.tolist())
        if not np.array_equal(
            selected_steps, np.asarray(canonical_steps, dtype=np.int64)
        ):
            raise QuartileSelectionError("path-table selected-step order changed")
        step_counts = np.bincount(
            np.asarray([step // 128 for step in canonical_steps], dtype=np.int64),
            minlength=4,
        )
        expected_local_counts = np.broadcast_to(
            step_counts[None, :, None], local_counts.shape
        )
        expected_primary_counts = np.broadcast_to(
            np.asarray(
                [
                    step_counts[0] * 56,
                    step_counts[1] * 56,
                    step_counts[2] * 56,
                    step_counts[3] * 56,
                    step_counts[2] * 56,
                    step_counts[3] * 56,
                ],
                dtype=np.int64,
            )[None, :],
            primary_counts.shape,
        )
        if (
            not np.array_equal(local_counts, expected_local_counts)
            or not np.array_equal(primary_counts, expected_primary_counts)
            or self.row_count != paths.size * len(canonical_steps) * 56
        ):
            raise QuartileSelectionError("path-table Cartesian counts changed")
        reconstructed = np.sum(
            local * local_counts, axis=2, dtype=np.float64
        ) / np.sum(local_counts, axis=2, dtype=np.int64)
        scale = np.maximum(
            1.0, np.maximum(np.abs(reconstructed), np.abs(primary[:, :4]))
        )
        if np.any(
            np.abs(reconstructed - primary[:, :4])
            > 32.0 * np.finfo(np.float64).eps * scale
        ):
            raise QuartileSelectionError(
                "primary pooled values disagree with local path reductions"
            )
        object.__setattr__(self, "path_ids", _readonly(paths))
        object.__setattr__(self, "primary_values", _readonly(primary))
        object.__setattr__(self, "local_values", _readonly(local))
        object.__setattr__(self, "primary_counts", _readonly(primary_counts))
        object.__setattr__(self, "local_counts", _readonly(local_counts))
        object.__setattr__(self, "selected_outer_steps", _readonly(selected_steps))

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(
            {
                "path_ids": self.path_ids.tolist(),
                "primary_values_sha256": _array_sha256(self.primary_values),
                "local_values_sha256": _array_sha256(self.local_values),
                "primary_counts_sha256": _array_sha256(self.primary_counts),
                "local_counts_sha256": _array_sha256(self.local_counts),
                "sample_key_sha256": self.sample_key_sha256,
                "selected_outer_steps": self.selected_outer_steps.tolist(),
                "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
                "local_family_names_sha256": LOCAL_FAMILY_NAMES_SHA256,
            }
        )

    def to_record(self) -> dict[str, Any]:
        primary_point = np.mean(self.primary_values, axis=0, dtype=np.float64)
        local_point = np.mean(self.local_values, axis=0, dtype=np.float64).reshape(-1)
        body = {
            "schema": PATH_TABLE_SCHEMA,
            "schema_version": 1,
            "path_ids": self.path_ids.tolist(),
            "path_count": self.path_count,
            "production_path_count": PRODUCTION_PATH_COUNT,
            "production_path_count_match": int(
                self.path_count == PRODUCTION_PATH_COUNT
            ),
            "row_count": int(self.row_count),
            "sample_key_sha256": self.sample_key_sha256,
            "selected_outer_steps": self.selected_outer_steps.tolist(),
            "selected_outer_steps_sha256": config_fingerprint(
                self.selected_outer_steps.tolist()
            ),
            "primary_family_names": list(PRIMARY_FAMILY_NAMES),
            "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
            "local_family_names_sha256": LOCAL_FAMILY_NAMES_SHA256,
            "primary_values_sha256": _array_sha256(self.primary_values),
            "local_values_sha256": _array_sha256(self.local_values),
            "primary_point_estimates": {
                name: float(value)
                for name, value in zip(
                    PRIMARY_FAMILY_NAMES, primary_point, strict=True
                )
            },
            "local_point_estimates_sha256": _array_sha256(local_point),
            "minimum_primary_count": int(np.min(self.primary_counts)),
            "maximum_primary_count": int(np.max(self.primary_counts)),
            "minimum_local_count": int(np.min(self.local_counts)),
            "maximum_local_count": int(np.max(self.local_counts)),
            "bootstrap_unit": "whole_path",
            "negative_values_truncated": 0,
            "raw_labels_persisted": 0,
            "raw_predictions_persisted": 0,
            "target_transformed": 0,
            "fingerprint": self.fingerprint,
        }
        return _semantic_record(body)


def aggregate_quartile_audit_improvements(
    *,
    sample_keys: Any,
    row_path_ids: Any,
    outer_steps: Any,
    phases: Any,
    midpoint_indices: Any,
    specialist_vs_zero_improvements: Any,
    shrunken_vs_raw_improvements: Any,
    expected_path_ids: Any,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    expected_path_count: int | None = PRODUCTION_PATH_COUNT,
) -> QuartileAuditPathTable:
    """Reduce rowwise direct-MSE contrasts to the two sealed path tables.

    Both input improvement vectors are edge-averaged raw-MSE differences.
    The shrinkage source is consumed only for q2/q3, but it is required to be
    finite everywhere so a streaming caller cannot silently pass a partial
    or role-dependent array.
    """

    keys = _integer_vector(sample_keys, "sample_keys")
    rows = int(keys.size)
    paths = _integer_vector(row_path_ids, "row_path_ids", rows)
    steps = _integer_vector(outer_steps, "outer_steps", rows)
    phase = _integer_vector(phases, "phases", rows)
    midpoint = _integer_vector(midpoint_indices, "midpoint_indices", rows)
    zero_improvement = _float_vector(
        specialist_vs_zero_improvements, "specialist_vs_zero_improvements", rows
    )
    shrink_improvement = _float_vector(
        shrunken_vs_raw_improvements, "shrunken_vs_raw_improvements", rows
    )
    expected_paths = _canonical_path_ids(
        expected_path_ids, expected_count=expected_path_count
    )
    expected_steps = _canonical_selected_steps(selected_outer_steps)
    if np.unique(keys).size != rows:
        raise QuartileSelectionError("audit sample keys are not unique")
    if set(np.unique(paths).tolist()) != set(expected_paths.tolist()):
        raise QuartileSelectionError("audit path set changed")
    if set(np.unique(steps).tolist()) != set(expected_steps):
        raise QuartileSelectionError("audit selected outer-step set changed")
    if np.any((phase < 0) | (phase >= PHASE_COUNT)):
        raise QuartileSelectionError("audit phase is invalid")
    if np.any((midpoint < 0) | (midpoint >= MIDPOINT_COUNT)):
        raise QuartileSelectionError("audit midpoint is invalid")
    expected_rows = (
        expected_paths.size * len(expected_steps) * PHASE_COUNT * MIDPOINT_COUNT
    )
    if rows != expected_rows:
        raise QuartileSelectionError(
            f"audit row count {rows} != frozen count {expected_rows}"
        )
    expected_keys = np.fromiter(
        (
            midpoint_sample_key(int(path), int(step), int(item_phase), int(item_mid))
            for path, step, item_phase, item_mid in zip(
                paths, steps, phase, midpoint, strict=True
            )
        ),
        dtype=np.int64,
        count=rows,
    )
    if not np.array_equal(keys, expected_keys):
        raise QuartileSelectionError(
            "sample keys do not match path/time/phase/midpoint identity"
        )
    identity = np.stack((paths, steps, phase, midpoint), axis=1)
    if np.unique(identity, axis=0).shape[0] != rows:
        raise QuartileSelectionError("audit row identity repeats")

    # Accumulation is identity-canonical, so cohort completion order and
    # interruption/resume boundaries cannot perturb a binary64 path sum.
    order = np.lexsort((midpoint, phase, steps, paths))
    paths = np.ascontiguousarray(paths[order])
    steps = np.ascontiguousarray(steps[order])
    phase = np.ascontiguousarray(phase[order])
    midpoint = np.ascontiguousarray(midpoint[order])
    zero_improvement = np.ascontiguousarray(zero_improvement[order])
    shrink_improvement = np.ascontiguousarray(shrink_improvement[order])

    path_index = np.searchsorted(expected_paths, paths)
    if np.any(path_index >= expected_paths.size) or not np.array_equal(
        expected_paths[path_index], paths
    ):
        raise QuartileSelectionError("audit path indexing failed")
    quartile = steps // 128
    local_index = phase * MIDPOINT_COUNT + midpoint

    local_sums = np.zeros(
        (expected_paths.size, TIME_QUARTILES, LOCAL_CELLS_PER_QUARTILE),
        dtype=np.float64,
    )
    local_counts = np.zeros_like(local_sums, dtype=np.int64)
    np.add.at(local_sums, (path_index, quartile, local_index), zero_improvement)
    np.add.at(local_counts, (path_index, quartile, local_index), 1)

    primary_sums = np.zeros(
        (expected_paths.size, PRIMARY_FAMILY_SIZE), dtype=np.float64
    )
    primary_counts = np.zeros_like(primary_sums, dtype=np.int64)
    np.add.at(primary_sums, (path_index, quartile), zero_improvement)
    np.add.at(primary_counts, (path_index, quartile), 1)
    for q_index, primary_index in ((2, 4), (3, 5)):
        selected = quartile == q_index
        np.add.at(
            primary_sums,
            (path_index[selected], primary_index),
            shrink_improvement[selected],
        )
        np.add.at(primary_counts, (path_index[selected], primary_index), 1)
    if np.any(local_counts <= 0) or np.any(primary_counts <= 0):
        raise QuartileSelectionError("an audit reduction cell is empty")
    local_values = local_sums / local_counts
    primary_values = primary_sums / primary_counts

    # The first four pooled contrasts must be the exactly weighted local
    # aggregate; this catches accidental pooling over edges or rows twice.
    reconstructed = np.sum(
        local_values * local_counts, axis=2, dtype=np.float64
    ) / np.sum(local_counts, axis=2, dtype=np.int64)
    scale = np.maximum(1.0, np.maximum(np.abs(reconstructed), np.abs(primary_values[:, :4])))
    if np.any(
        np.abs(reconstructed - primary_values[:, :4])
        > 32.0 * np.finfo(np.float64).eps * scale
    ):
        raise QuartileSelectionError("pooled and local audit reductions disagree")
    key_hash = hashlib.sha256(
        np.ascontiguousarray(np.sort(keys, kind="stable")).tobytes(order="C")
    ).hexdigest()
    return QuartileAuditPathTable(
        path_ids=expected_paths,
        primary_values=np.ascontiguousarray(primary_values),
        local_values=np.ascontiguousarray(local_values),
        primary_counts=np.ascontiguousarray(primary_counts),
        local_counts=np.ascontiguousarray(local_counts),
        selected_outer_steps=np.asarray(expected_steps, dtype=np.int64),
        sample_key_sha256=key_hash,
        row_count=rows,
    )


def aggregate_quartile_audit_risks(
    *,
    sample_keys: Any,
    row_path_ids: Any,
    outer_steps: Any,
    phases: Any,
    midpoint_indices: Any,
    targets: Any,
    raw_predictions: Any,
    gains: Any,
    expected_path_ids: Any,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    expected_path_count: int | None = PRODUCTION_PATH_COUNT,
) -> QuartileAuditPathTable:
    """Compute the exact six direct-MSE contrasts and reduce them by path."""

    keys = _integer_vector(sample_keys, "sample_keys")
    rows = int(keys.size)
    steps = _integer_vector(outer_steps, "outer_steps", rows)
    target = _float_edge_table(targets, "targets", rows)
    raw = _float_edge_table(raw_predictions, "raw_predictions", rows)
    gain_source = np.asarray(gains)
    if (
        gain_source.dtype != np.dtype(np.float64)
        or gain_source.shape != (TIME_QUARTILES,)
        or not np.isfinite(gain_source).all()
        or gain_source[0] != 1.0
        or gain_source[1] != 1.0
        or not 0.0 < float(gain_source[2]) < 1.0
        or not 0.0 < float(gain_source[3]) < 1.0
    ):
        raise QuartileSelectionError("sealed quartile gains are malformed")
    quartile = steps // 128
    if np.any((quartile < 0) | (quartile >= TIME_QUARTILES)):
        raise QuartileSelectionError("outer step reconstructs an invalid quartile")
    final = raw * gain_source[quartile, None]
    zero_improvement = np.mean(
        target * target - (target - final) ** 2, axis=1, dtype=np.float64
    )
    shrink_improvement = np.mean(
        (target - raw) ** 2 - (target - final) ** 2,
        axis=1,
        dtype=np.float64,
    )
    if not np.isfinite(zero_improvement).all() or not np.isfinite(
        shrink_improvement
    ).all():
        raise QuartileSelectionError("direct audit risk contrast is nonfinite")
    return aggregate_quartile_audit_improvements(
        sample_keys=keys,
        row_path_ids=row_path_ids,
        outer_steps=steps,
        phases=phases,
        midpoint_indices=midpoint_indices,
        specialist_vs_zero_improvements=np.ascontiguousarray(zero_improvement),
        shrunken_vs_raw_improvements=np.ascontiguousarray(shrink_improvement),
        expected_path_ids=expected_path_ids,
        selected_outer_steps=selected_outer_steps,
        expected_path_count=expected_path_count,
    )


@dataclass(frozen=True)
class LocalCompatibilityScreen:
    """Fixed directional screen; this record makes no confidence claim."""

    point_estimates: np.ndarray
    phase_marginals: np.ndarray
    midpoint_marginals: np.ndarray
    positive_cell_counts: np.ndarray
    quartile_passed: np.ndarray

    def __post_init__(self) -> None:
        point = np.asarray(self.point_estimates)
        phase = np.asarray(self.phase_marginals)
        midpoint = np.asarray(self.midpoint_marginals)
        counts = np.asarray(self.positive_cell_counts)
        passed = np.asarray(self.quartile_passed)
        if (
            point.dtype != np.dtype(np.float64)
            or point.shape != (4, 7, 8)
            or phase.dtype != np.dtype(np.float64)
            or phase.shape != (4, 7)
            or midpoint.dtype != np.dtype(np.float64)
            or midpoint.shape != (4, 8)
            or not np.isfinite(point).all()
            or not np.isfinite(phase).all()
            or not np.isfinite(midpoint).all()
            or counts.dtype != np.dtype(np.int64)
            or counts.shape != (4,)
            or np.any((counts < 0) | (counts > 56))
            or passed.dtype != np.dtype(np.bool_)
            or passed.shape != (4,)
        ):
            raise QuartileSelectionError("local compatibility screen is malformed")
        expected_phase = np.mean(point, axis=2, dtype=np.float64)
        expected_midpoint = np.mean(point, axis=1, dtype=np.float64)
        expected_counts = np.count_nonzero(point > 0.0, axis=(1, 2)).astype(
            np.int64
        )
        expected_passed = (
            np.all(expected_phase > 0.0, axis=1)
            & np.all(expected_midpoint > 0.0, axis=1)
            & (expected_counts >= MINIMUM_POSITIVE_LOCAL_CELLS)
        )
        expected_passed[1] = bool(
            expected_passed[1]
            and point[1, Q1_SENTINEL_PHASE, Q1_SENTINEL_MIDPOINT] > 0.0
        )
        if (
            not np.array_equal(phase, expected_phase)
            or not np.array_equal(midpoint, expected_midpoint)
            or not np.array_equal(counts, expected_counts)
            or not np.array_equal(passed, expected_passed)
        ):
            raise QuartileSelectionError("local compatibility arithmetic changed")
        object.__setattr__(self, "point_estimates", _readonly(point))
        object.__setattr__(self, "phase_marginals", _readonly(phase))
        object.__setattr__(self, "midpoint_marginals", _readonly(midpoint))
        object.__setattr__(self, "positive_cell_counts", _readonly(counts))
        object.__setattr__(self, "quartile_passed", _readonly(passed))

    @property
    def passed(self) -> bool:
        return bool(np.all(self.quartile_passed))

    def to_record(self) -> dict[str, Any]:
        rows = []
        for quartile in range(4):
            rows.append(
                {
                    "quartile": quartile,
                    "all_phase_marginals_positive": int(
                        np.all(self.phase_marginals[quartile] > 0.0)
                    ),
                    "all_midpoint_marginals_positive": int(
                        np.all(self.midpoint_marginals[quartile] > 0.0)
                    ),
                    "positive_cell_count": int(
                        self.positive_cell_counts[quartile]
                    ),
                    "minimum_positive_cell_requirement": MINIMUM_POSITIVE_LOCAL_CELLS,
                    "sentinel_required": int(quartile == 1),
                    "sentinel_point_estimate": (
                        float(
                            self.point_estimates[
                                1, Q1_SENTINEL_PHASE, Q1_SENTINEL_MIDPOINT
                            ]
                        )
                        if quartile == 1
                        else None
                    ),
                    "passed": int(self.quartile_passed[quartile]),
                }
            )
        body = {
            "schema": LOCAL_SCREEN_SCHEMA,
            "schema_version": 1,
            "local_family_names_sha256": LOCAL_FAMILY_NAMES_SHA256,
            "local_family_size": LOCAL_FAMILY_SIZE,
            "minimum_positive_cells_per_quartile": MINIMUM_POSITIVE_LOCAL_CELLS,
            "q1_sentinel": "q1.phase4.midpoint7",
            "point_estimates_sha256": _array_sha256(self.point_estimates),
            "phase_marginals_sha256": _array_sha256(self.phase_marginals),
            "midpoint_marginals_sha256": _array_sha256(self.midpoint_marginals),
            "quartiles": rows,
            "passed": int(self.passed),
            "inferential_claim_made": 0,
            "used_to_modify_sealed_system": 0,
        }
        return _semantic_record(body)


def evaluate_local_compatibility_screen(
    local_path_values: Any,
) -> LocalCompatibilityScreen:
    """Evaluate the preregistered point-estimate-only local restrictions."""

    source = np.asarray(local_path_values)
    if (
        source.dtype != np.dtype(np.float64)
        or source.ndim != 3
        or source.shape[0] < 8
        or source.shape[1:] != (4, 56)
        or not np.isfinite(source).all()
    ):
        raise QuartileSelectionError("local path values must be finite [path,4,56]")
    point = np.mean(source, axis=0, dtype=np.float64).reshape(4, 7, 8)
    phase = np.mean(point, axis=2, dtype=np.float64)
    midpoint = np.mean(point, axis=1, dtype=np.float64)
    positive = np.count_nonzero(point > 0.0, axis=(1, 2)).astype(np.int64)
    passed = (
        np.all(phase > 0.0, axis=1)
        & np.all(midpoint > 0.0, axis=1)
        & (positive >= MINIMUM_POSITIVE_LOCAL_CELLS)
    )
    passed[1] = bool(
        passed[1] and point[1, Q1_SENTINEL_PHASE, Q1_SENTINEL_MIDPOINT] > 0.0
    )
    return LocalCompatibilityScreen(
        point_estimates=np.ascontiguousarray(point),
        phase_marginals=np.ascontiguousarray(phase),
        midpoint_marginals=np.ascontiguousarray(midpoint),
        positive_cell_counts=np.ascontiguousarray(positive),
        quartile_passed=np.ascontiguousarray(passed),
    )


def bootstrap_plan(
    *,
    seed: int,
    namespace: int,
    path_count: int = PRODUCTION_PATH_COUNT,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    """Return the complete frozen six-family bootstrap plan."""

    _validate_bootstrap_configuration(
        seed=seed,
        namespace=namespace,
        path_count=path_count,
        replicates=replicates,
        shard_size=shard_size,
    )
    body = {
        "schema": BOOTSTRAP_PLAN_SCHEMA,
        "schema_version": 1,
        "seed": int(seed),
        "namespace": int(namespace),
        "path_count": int(path_count),
        "production_path_count": PRODUCTION_PATH_COUNT,
        "production_path_count_match": int(path_count == PRODUCTION_PATH_COUNT),
        "replicates": int(replicates),
        "shard_size": int(shard_size),
        "shard_count": int(replicates // shard_size),
        "count_dtype": np.dtype(np.uint16).str,
        "confidence": DEFAULT_CONFIDENCE,
        "quantile_method": "higher",
        "method": "centered_whole_path_one_sided_studentized_max_t",
        "standard_error_floor_used": 0,
        "negative_values_truncated": 0,
        "philox_constructor": PHILOX_CONSTRUCTOR,
        "primary_family_names": list(PRIMARY_FAMILY_NAMES),
        "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
    }
    return _semantic_record(body)


def _validate_bootstrap_configuration(
    *,
    seed: int,
    namespace: int,
    path_count: int,
    replicates: int,
    shard_size: int,
) -> None:
    integers = (seed, namespace, path_count, replicates, shard_size)
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in integers
    ) or not 2 <= path_count <= np.iinfo(np.uint16).max:
        raise QuartileSelectionError("bootstrap configuration is invalid")
    if replicates <= 0 or shard_size <= 0 or replicates % shard_size != 0:
        raise QuartileSelectionError("bootstrap shard plan is invalid")


def generate_bootstrap_count_shard(
    *,
    seed: int,
    namespace: int,
    shard_index: int,
    path_count: int = PRODUCTION_PATH_COUNT,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> np.ndarray:
    """Generate one stateless uint16 Philox whole-path count shard."""

    _validate_bootstrap_configuration(
        seed=seed,
        namespace=namespace,
        path_count=path_count,
        replicates=shard_size,
        shard_size=shard_size,
    )
    if not isinstance(shard_index, int) or isinstance(shard_index, bool) or shard_index < 0:
        raise QuartileSelectionError("bootstrap shard index is invalid")
    generator = np.random.Generator(
        np.random.Philox([int(seed), int(namespace), int(shard_index)])
    )
    indices = generator.integers(
        0,
        int(path_count),
        size=(int(shard_size), int(path_count)),
        dtype=np.int64,
    )
    counts = np.zeros((int(shard_size), int(path_count)), dtype=np.uint16)
    rows = np.broadcast_to(
        np.arange(int(shard_size), dtype=np.int64)[:, None], indices.shape
    )
    np.add.at(counts, (rows, indices), 1)
    if np.any(np.sum(counts, axis=1, dtype=np.int64) != int(path_count)):
        raise QuartileSelectionError("bootstrap count rows do not sum to path count")
    return np.ascontiguousarray(counts)


def _validate_count_shard(
    value: Any,
    *,
    path_count: int,
    shard_size: int | None = None,
) -> np.ndarray:
    counts = np.asarray(value)
    if (
        counts.dtype != np.dtype(np.uint16)
        or counts.ndim != 2
        or counts.shape[1] != int(path_count)
        or (shard_size is not None and counts.shape[0] != int(shard_size))
        or np.any(np.sum(counts, axis=1, dtype=np.int64) != int(path_count))
    ):
        raise QuartileSelectionError("bootstrap count shard is malformed")
    return np.ascontiguousarray(counts)


def shard_artifact_paths(directory: str | Path, shard_index: int) -> tuple[Path, Path]:
    if not isinstance(shard_index, int) or isinstance(shard_index, bool) or shard_index < 0:
        raise QuartileSelectionError("shard index is invalid")
    stem = f"shard-{shard_index:05d}"
    root = Path(directory)
    return root / f"{stem}.npz", root / f"{stem}.metadata.json"


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
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QuartileSelectionError(f"cannot load shard metadata {path}") from exc
    if not isinstance(value, dict):
        raise QuartileSelectionError(f"shard metadata is not an object: {path}")
    return value


def _load_npz_array(path: Path, name: str) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != (name,):
                raise QuartileSelectionError(f"{path.name} has an invalid NPZ schema")
            return np.array(archive[name], copy=True)
    except (OSError, ValueError, KeyError) as exc:
        raise QuartileSelectionError(f"cannot load shard artifact {path}") from exc


def _count_metadata(
    *,
    path: Path,
    counts: np.ndarray,
    seed: int,
    namespace: int,
    shard_index: int,
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
            "artifact_path": path.resolve().as_posix(),
            "philox_constructor": PHILOX_CONSTRUCTOR,
            "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
        }
    )


def prepare_bootstrap_count_shards(
    directory: str | Path,
    *,
    seed: int,
    namespace: int,
    path_count: int = PRODUCTION_PATH_COUNT,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_SHARD_SIZE,
    allow_repair: bool = True,
) -> list[dict[str, Any]]:
    """Commit or verify prospective count shards before audit labels open."""

    _validate_bootstrap_configuration(
        seed=seed,
        namespace=namespace,
        path_count=path_count,
        replicates=replicates,
        shard_size=shard_size,
    )
    root = Path(directory)
    if not root.exists():
        if not allow_repair:
            raise QuartileSelectionError("sealed count-shard directory is missing")
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise QuartileSelectionError("count-shard path is not a directory")
    records: list[dict[str, Any]] = []
    for shard_index in range(replicates // shard_size):
        data_path, metadata_path = shard_artifact_paths(root, shard_index)
        if metadata_path.exists():
            try:
                record = _load_json(metadata_path)
                _validate_semantic_record(record)
                counts = _validate_count_shard(
                    _load_npz_array(data_path, "counts"),
                    path_count=path_count,
                    shard_size=shard_size,
                )
                expected_counts = generate_bootstrap_count_shard(
                    seed=seed,
                    namespace=namespace,
                    shard_index=shard_index,
                    path_count=path_count,
                    shard_size=shard_size,
                )
                expected = {
                    "schema": BOOTSTRAP_COUNTS_SCHEMA,
                    "seed": int(seed),
                    "namespace": int(namespace),
                    "shard_index": int(shard_index),
                    "path_count": int(path_count),
                    "shard_size": int(shard_size),
                    "counts_sha256": _array_sha256(counts),
                    "artifact_sha256": file_fingerprint(data_path),
                    "artifact_path": data_path.resolve().as_posix(),
                    "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
                }
                if any(record.get(key) != item for key, item in expected.items()) or not np.array_equal(
                    counts, expected_counts
                ):
                    raise QuartileSelectionError("committed count-shard binding changed")
                records.append(record)
                continue
            except (OSError, QuartileSelectionError):
                if not allow_repair:
                    raise
        elif not allow_repair:
            raise QuartileSelectionError("required sealed count shard is missing")
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
        )
        atomic_write_json(metadata_path, record)
        records.append(record)
    return records


def load_bootstrap_count_shards(
    directory: str | Path,
    *,
    seed: int,
    namespace: int,
    path_count: int = PRODUCTION_PATH_COUNT,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Load prospective count shards without regeneration authority."""

    records = prepare_bootstrap_count_shards(
        directory,
        seed=seed,
        namespace=namespace,
        path_count=path_count,
        replicates=replicates,
        shard_size=shard_size,
        allow_repair=False,
    )
    arrays = [
        _load_npz_array(shard_artifact_paths(directory, index)[0], "counts")
        for index in range(len(records))
    ]
    return arrays, records


def count_shard_index_record(
    records: Sequence[Mapping[str, Any]],
    *,
    role: str,
    authorizing: bool | None = None,
) -> dict[str, Any]:
    """Return the prospective seal/index binding all count shards."""

    role_value = str(role)
    if role_value not in {"selection", "confirmation"} or not records:
        raise QuartileSelectionError("count-shard index role or records are invalid")
    normalized = [dict(record) for record in records]
    for record in normalized:
        _validate_semantic_record(record)
    expected_seed = (
        DEFAULT_SELECTION_SEED if role_value == "selection" else DEFAULT_CONFIRMATION_SEED
    )
    expected_namespace = (
        SELECTION_NAMESPACE if role_value == "selection" else CONFIRMATION_NAMESPACE
    )
    if any(
        row.get("schema") != BOOTSTRAP_COUNTS_SCHEMA
        or row.get("primary_family_names_sha256") != PRIMARY_FAMILY_NAMES_SHA256
        or row.get("seed") != expected_seed
        or row.get("namespace") != expected_namespace
        or row.get("shard_index") != index
        or row.get("path_count") != normalized[0].get("path_count")
        or row.get("dtype") != np.dtype(np.uint16).str
        for index, row in enumerate(normalized)
    ):
        raise QuartileSelectionError("count-shard index binding is inconsistent")
    for row in normalized:
        artifact_path = Path(str(row.get("artifact_path", "")))
        if (
            not artifact_path.is_file()
            or file_fingerprint(artifact_path) != row.get("artifact_sha256")
        ):
            raise QuartileSelectionError("count-shard index artifact changed")
        counts = _validate_count_shard(
            _load_npz_array(artifact_path, "counts"),
            path_count=int(row["path_count"]),
            shard_size=int(row["shard_size"]),
        )
        if _array_sha256(counts) != row.get("counts_sha256"):
            raise QuartileSelectionError("count-shard index payload changed")
        expected_counts = generate_bootstrap_count_shard(
            seed=int(row["seed"]),
            namespace=int(row["namespace"]),
            shard_index=int(row["shard_index"]),
            path_count=int(row["path_count"]),
            shard_size=int(row["shard_size"]),
        )
        if not np.array_equal(counts, expected_counts):
            raise QuartileSelectionError("count-shard index is not the frozen Philox draw")
    production_plan = bool(
        len(normalized) == DEFAULT_REPLICATES // DEFAULT_SHARD_SIZE
        and int(normalized[0]["path_count"]) == PRODUCTION_PATH_COUNT
        and all(int(row["shard_size"]) == DEFAULT_SHARD_SIZE for row in normalized)
    )
    authorizing_value = production_plan if authorizing is None else authorizing
    if not isinstance(authorizing_value, bool):
        raise QuartileSelectionError("count-shard index authorizing flag is invalid")
    if authorizing_value and not production_plan:
        raise QuartileSelectionError("nonproduction count plan cannot be authorizing")
    body = {
        "schema": SCHEMA + "-bootstrap-count-index",
        "schema_version": 1,
        "role": role_value,
        "seed": expected_seed,
        "namespace": expected_namespace,
        "path_count": int(normalized[0]["path_count"]),
        "production_path_count": PRODUCTION_PATH_COUNT,
        "production_path_count_match": int(
            int(normalized[0]["path_count"]) == PRODUCTION_PATH_COUNT
        ),
        "shard_count": len(normalized),
        "replicate_count": int(sum(int(row["shard_size"]) for row in normalized)),
        "production_replicate_count": DEFAULT_REPLICATES,
        "production_replicate_count_match": int(
            sum(int(row["shard_size"]) for row in normalized)
            == DEFAULT_REPLICATES
        ),
        "count_dtype": np.dtype(np.uint16).str,
        "metadata_semantic_sha256": [row["semantic_sha256"] for row in normalized],
        "artifact_sha256": [row["artifact_sha256"] for row in normalized],
        "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
        "sealed_before_physical_labels": int(authorizing_value),
        "authorizing_count_plan": int(authorizing_value),
    }
    return _semantic_record(body)


def _canonical_primary_values(
    values: Any,
    path_ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values)
    paths_source = np.asarray(path_ids)
    if paths_source.ndim != 1 or paths_source.dtype.kind not in "iu":
        raise QuartileSelectionError("max-T path IDs must be integer")
    paths = _canonical_path_ids(paths_source)
    if (
        source.dtype != np.dtype(np.float64)
        or source.shape != (paths_source.size, PRIMARY_FAMILY_SIZE)
        or not np.isfinite(source).all()
    ):
        raise QuartileSelectionError("max-T values must be finite [path,6] binary64")
    order = np.argsort(np.asarray(paths_source, dtype=np.int64), kind="stable")
    return paths, np.ascontiguousarray(source[order])


def _observed_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point = np.mean(values, axis=0, dtype=np.float64)
    standard_error = np.std(values, axis=0, ddof=1, dtype=np.float64) / math.sqrt(
        values.shape[0]
    )
    if (
        not np.isfinite(point).all()
        or not np.isfinite(standard_error).all()
        or np.any(standard_error <= 0.0)
    ):
        raise QuartileSelectionError(
            "primary family has degenerate/nonfinite observed studentization",
            failure_code="quartile_max_t_studentization_invalid",
        )
    return np.ascontiguousarray(point), np.ascontiguousarray(standard_error)


def compute_bootstrap_maxima_shard(
    values: Any,
    counts: Any,
    *,
    path_ids: Any,
) -> np.ndarray:
    """Compute one centered six-family studentized maxima shard."""

    paths, table = _canonical_primary_values(values, path_ids)
    del paths
    path_count = int(table.shape[0])
    count_table = _validate_count_shard(counts, path_count=path_count)
    point, _ = _observed_statistics(table)
    count_float = np.asarray(count_table, dtype=np.float64)
    draw_mean = (count_float @ table) / path_count
    draw_second = (count_float @ (table * table)) / path_count
    draw_variance = path_count * (draw_second - draw_mean * draw_mean) / (
        path_count - 1
    )
    draw_error = np.sqrt(draw_variance / path_count)
    if not np.isfinite(draw_error).all() or np.any(draw_error <= 0.0):
        raise QuartileSelectionError(
            "bootstrap produced degenerate/nonfinite studentization",
            failure_code="quartile_max_t_bootstrap_studentization_invalid",
        )
    centered = (draw_mean - point[None, :]) / draw_error
    maxima = np.max(centered, axis=1)
    if not np.isfinite(maxima).all():
        raise QuartileSelectionError("bootstrap maxima are nonfinite")
    return np.ascontiguousarray(maxima, dtype=np.float64)


@dataclass(frozen=True)
class QuartileMaxTResult:
    """One-system, six-component simultaneous inference result."""

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
            or paths.size < 8
            or np.unique(paths).size != paths.size
            or point.dtype != np.dtype(np.float64)
            or point.shape != (PRIMARY_FAMILY_SIZE,)
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
            raise QuartileSelectionError("quartile max-T result is malformed")
        expected_critical = float(
            np.quantile(maxima, float(self.confidence), method="higher")
        )
        expected_lower = point - expected_critical * error
        if self.critical_value != expected_critical or not np.array_equal(
            lower, expected_lower
        ):
            raise QuartileSelectionError("quartile max-T arithmetic changed")
        object.__setattr__(self, "path_ids", _readonly(paths))
        object.__setattr__(self, "point_estimates", _readonly(point))
        object.__setattr__(self, "standard_errors", _readonly(error))
        object.__setattr__(self, "lower_bounds", _readonly(lower))
        object.__setattr__(self, "maxima", _readonly(maxima))

    @property
    def passed(self) -> bool:
        return bool(np.all(self.lower_bounds > 0.0))

    def to_record(self) -> dict[str, Any]:
        body = {
            "schema": MAX_T_SCHEMA,
            "schema_version": 1,
            "method": "centered_whole_path_one_sided_studentized_max_t",
            "bootstrap_unit": "whole_path_shared_across_six_components",
            "primary_family_names": list(PRIMARY_FAMILY_NAMES),
            "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
            "family_size": PRIMARY_FAMILY_SIZE,
            "path_ids": self.path_ids.tolist(),
            "path_count": int(self.path_ids.size),
            "confidence": float(self.confidence),
            "replicates": int(self.maxima.size),
            "critical_value": float(self.critical_value),
            "quantile_method": "higher",
            "standard_error_floor_used": 0,
            "negative_values_truncated": 0,
            "point_estimates": {
                name: float(value)
                for name, value in zip(
                    PRIMARY_FAMILY_NAMES, self.point_estimates, strict=True
                )
            },
            "standard_errors": {
                name: float(value)
                for name, value in zip(
                    PRIMARY_FAMILY_NAMES, self.standard_errors, strict=True
                )
            },
            "lower_bounds": {
                name: float(value)
                for name, value in zip(
                    PRIMARY_FAMILY_NAMES, self.lower_bounds, strict=True
                )
            },
            "critical_value_source": (
                f"{self.maxima.size}_stateless_philox_maxima"
            ),
            "point_estimates_sha256": _array_sha256(self.point_estimates),
            "standard_errors_sha256": _array_sha256(self.standard_errors),
            "lower_bounds_sha256": _array_sha256(self.lower_bounds),
            "maxima_sha256": _array_sha256(self.maxima),
            "all_six_lower_bounds_strictly_positive": int(self.passed),
            "passed": int(self.passed),
        }
        return _semantic_record(body)


def quartile_max_t(
    values: Any,
    *,
    path_ids: Any,
    count_shards: Sequence[Any],
    confidence: float = DEFAULT_CONFIDENCE,
) -> QuartileMaxTResult:
    """Run the fixed-family in-memory numeric core."""

    if not 0.5 < float(confidence) < 1.0 or not count_shards:
        raise QuartileSelectionError("max-T configuration is invalid")
    paths, table = _canonical_primary_values(values, path_ids)
    point, standard_error = _observed_statistics(table)
    maxima = np.ascontiguousarray(
        np.concatenate(
            [
                compute_bootstrap_maxima_shard(
                    table, counts, path_ids=paths
                )
                for counts in count_shards
            ]
        )
    )
    critical = float(np.quantile(maxima, float(confidence), method="higher"))
    if not math.isfinite(critical):
        raise QuartileSelectionError("max-T critical value is nonfinite")
    return QuartileMaxTResult(
        path_ids=paths,
        point_estimates=point,
        standard_errors=standard_error,
        lower_bounds=np.ascontiguousarray(point - critical * standard_error),
        maxima=maxima,
        critical_value=critical,
        confidence=float(confidence),
    )


def _maxima_metadata(
    *,
    path: Path,
    maxima: np.ndarray,
    shard_index: int,
    count_record: Mapping[str, Any],
    evidence_sha256: str,
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
            "artifact_path": path.resolve().as_posix(),
            "count_metadata_semantic_sha256": count_record["semantic_sha256"],
            "count_artifact_sha256": count_record["artifact_sha256"],
            "evidence_sha256": evidence_sha256,
            "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
        }
    )


def restartable_quartile_max_t(
    values: Any,
    *,
    path_ids: Any,
    count_directory: str | Path,
    maxima_directory: str | Path,
    seed: int,
    namespace: int,
    confidence: float = DEFAULT_CONFIDENCE,
    replicates: int = DEFAULT_REPLICATES,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> tuple[QuartileMaxTResult, list[dict[str, Any]], list[dict[str, Any]]]:
    """Resume derived maxima from sealed prospective uint16 count shards."""

    paths, table = _canonical_primary_values(values, path_ids)
    count_arrays, count_records = load_bootstrap_count_shards(
        count_directory,
        seed=seed,
        namespace=namespace,
        path_count=int(paths.size),
        replicates=replicates,
        shard_size=shard_size,
    )
    evidence_sha256 = config_fingerprint(
        {
            "path_ids": paths.tolist(),
            "primary_values_sha256": _array_sha256(table),
            "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
        }
    )
    root = Path(maxima_directory)
    root.mkdir(parents=True, exist_ok=True)
    maxima_arrays: list[np.ndarray] = []
    maxima_records: list[dict[str, Any]] = []
    for shard_index, (counts, count_record) in enumerate(
        zip(count_arrays, count_records, strict=True)
    ):
        data_path, metadata_path = shard_artifact_paths(root, shard_index)
        if metadata_path.exists():
            record = _load_json(metadata_path)
            _validate_semantic_record(record)
            immutable_binding = {
                "schema": BOOTSTRAP_MAXIMA_SCHEMA,
                "shard_index": shard_index,
                "replicate_count": int(counts.shape[0]),
                "count_metadata_semantic_sha256": count_record["semantic_sha256"],
                "count_artifact_sha256": count_record["artifact_sha256"],
                "evidence_sha256": evidence_sha256,
                "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
            }
            if any(
                record.get(key) != item for key, item in immutable_binding.items()
            ):
                raise QuartileSelectionError("committed maxima binding changed")
            try:
                maxima = _load_npz_array(data_path, "maxima")
                payload_valid = (
                    maxima.dtype == np.dtype(np.float64)
                    and maxima.shape == (counts.shape[0],)
                    and np.isfinite(maxima).all()
                    and record.get("maxima_sha256") == _array_sha256(maxima)
                    and record.get("artifact_sha256") == file_fingerprint(data_path)
                )
            except (OSError, QuartileSelectionError):
                payload_valid = False
            if not payload_valid:
                # Maxima are nonauthorizing derived data.  Their immutable
                # evidence/count binding was verified above, so deterministic
                # reconstruction is both safe and required for exact resume.
                maxima = compute_bootstrap_maxima_shard(
                    table, counts, path_ids=paths
                )
                _atomic_npz(data_path, maxima=maxima)
                record = _maxima_metadata(
                    path=data_path,
                    maxima=maxima,
                    shard_index=shard_index,
                    count_record=count_record,
                    evidence_sha256=evidence_sha256,
                )
                atomic_write_json(metadata_path, record)
        else:
            maxima = compute_bootstrap_maxima_shard(
                table, counts, path_ids=paths
            )
            _atomic_npz(data_path, maxima=maxima)
            record = _maxima_metadata(
                path=data_path,
                maxima=maxima,
                shard_index=shard_index,
                count_record=count_record,
                evidence_sha256=evidence_sha256,
            )
            atomic_write_json(metadata_path, record)
        maxima_arrays.append(np.ascontiguousarray(maxima))
        maxima_records.append(record)
    point, standard_error = _observed_statistics(table)
    maxima = np.ascontiguousarray(np.concatenate(maxima_arrays))
    critical = float(np.quantile(maxima, float(confidence), method="higher"))
    result = QuartileMaxTResult(
        path_ids=paths,
        point_estimates=point,
        standard_errors=standard_error,
        lower_bounds=np.ascontiguousarray(point - critical * standard_error),
        maxima=maxima,
        critical_value=critical,
        confidence=float(confidence),
    )
    return result, count_records, maxima_records


def _audit_record(
    *,
    role: str,
    result: QuartileMaxTResult,
    local_screen: LocalCompatibilityScreen,
    path_table: QuartileAuditPathTable | None = None,
    count_records: Sequence[Mapping[str, Any]] | None = None,
    maxima_records: Sequence[Mapping[str, Any]] | None = None,
    authorizing: bool = True,
) -> dict[str, Any]:
    role_value = str(role)
    if role_value not in {"selection", "confirmation"}:
        raise QuartileSelectionError("audit role is invalid")
    if not isinstance(result, QuartileMaxTResult) or not isinstance(
        local_screen, LocalCompatibilityScreen
    ):
        raise QuartileSelectionError("audit record requires canonical evidence")
    if not isinstance(authorizing, bool):
        raise QuartileSelectionError("authorizing flag must be boolean")
    if path_table is not None:
        if not isinstance(path_table, QuartileAuditPathTable) or not np.array_equal(
            result.path_ids, path_table.path_ids
        ):
            raise QuartileSelectionError("max-T and local evidence paths differ")
        expected_screen = evaluate_local_compatibility_screen(path_table.local_values)
        if (
            not np.array_equal(expected_screen.point_estimates, local_screen.point_estimates)
            or not np.array_equal(expected_screen.quartile_passed, local_screen.quartile_passed)
        ):
            raise QuartileSelectionError("local-screen record changed")
        expected_point, expected_error = _observed_statistics(
            path_table.primary_values
        )
        if not np.array_equal(expected_point, result.point_estimates) or not np.array_equal(
            expected_error, result.standard_errors
        ):
            raise QuartileSelectionError("max-T result does not match path reductions")
    count_hashes = []
    count_rows = [dict(record) for record in count_records or ()]
    maxima_rows = [dict(record) for record in maxima_records or ()]
    if len(count_rows) != len(maxima_rows):
        raise QuartileSelectionError("count/maxima shard families have different lengths")
    expected_seed = (
        DEFAULT_SELECTION_SEED if role_value == "selection" else DEFAULT_CONFIRMATION_SEED
    )
    expected_namespace = (
        SELECTION_NAMESPACE if role_value == "selection" else CONFIRMATION_NAMESPACE
    )
    expected_paths = np.arange(
        SELECTION_PATH_START if role_value == "selection" else CONFIRMATION_PATH_START,
        SELECTION_PATH_STOP if role_value == "selection" else CONFIRMATION_PATH_STOP,
        dtype=np.int64,
    )
    if authorizing and (
        path_table is None
        or not np.array_equal(result.path_ids, expected_paths)
        or not np.array_equal(
            path_table.selected_outer_steps,
            np.asarray(SELECTED_OUTER_STEPS, dtype=np.int64),
        )
        or path_table.sample_key_sha256
        != expected_audit_sample_key_sha256(
            expected_paths, selected_outer_steps=SELECTED_OUTER_STEPS
        )
        or result.maxima.size != DEFAULT_REPLICATES
        or result.confidence != DEFAULT_CONFIDENCE
        or len(count_rows) != DEFAULT_REPLICATES // DEFAULT_SHARD_SIZE
        or len(maxima_rows) != DEFAULT_REPLICATES // DEFAULT_SHARD_SIZE
    ):
        raise QuartileSelectionError(
            "authorizing audit does not satisfy the frozen 384-path/50000-draw plan"
        )
    committed_count_arrays: list[np.ndarray] = []
    for index, record in enumerate(count_rows):
        _validate_semantic_record(record)
        if authorizing and (
            record.get("schema") != BOOTSTRAP_COUNTS_SCHEMA
            or record.get("seed") != expected_seed
            or record.get("namespace") != expected_namespace
            or record.get("shard_index") != index
            or record.get("path_count") != PRODUCTION_PATH_COUNT
            or record.get("shard_size") != DEFAULT_SHARD_SIZE
            or record.get("dtype") != np.dtype(np.uint16).str
            or record.get("primary_family_names_sha256")
            != PRIMARY_FAMILY_NAMES_SHA256
            or not isinstance(record.get("counts_sha256"), str)
            or len(str(record.get("counts_sha256"))) != 64
            or not isinstance(record.get("artifact_sha256"), str)
            or len(str(record.get("artifact_sha256"))) != 64
        ):
            raise QuartileSelectionError("authorizing count-shard binding changed")
        if authorizing:
            artifact_path = Path(str(record.get("artifact_path", "")))
            if (
                not artifact_path.is_file()
                or file_fingerprint(artifact_path) != record["artifact_sha256"]
            ):
                raise QuartileSelectionError("authorizing count artifact changed")
            committed_counts = _validate_count_shard(
                _load_npz_array(artifact_path, "counts"),
                path_count=PRODUCTION_PATH_COUNT,
                shard_size=DEFAULT_SHARD_SIZE,
            )
            if _array_sha256(committed_counts) != record["counts_sha256"]:
                raise QuartileSelectionError("authorizing count payload changed")
            expected_counts = generate_bootstrap_count_shard(
                seed=expected_seed,
                namespace=expected_namespace,
                shard_index=index,
                path_count=PRODUCTION_PATH_COUNT,
                shard_size=DEFAULT_SHARD_SIZE,
            )
            if not np.array_equal(committed_counts, expected_counts):
                raise QuartileSelectionError(
                    "authorizing count payload is not the frozen Philox draw"
                )
            committed_count_arrays.append(committed_counts)
        count_hashes.append(str(record["semantic_sha256"]))
    maxima_hashes = []
    expected_evidence_sha256 = (
        config_fingerprint(
            {
                "path_ids": result.path_ids.tolist(),
                "primary_values_sha256": _array_sha256(path_table.primary_values),
                "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
            }
        )
        if path_table is not None
        else None
    )
    for index, record in enumerate(maxima_rows):
        _validate_semantic_record(record)
        if authorizing and (
            record.get("schema") != BOOTSTRAP_MAXIMA_SCHEMA
            or record.get("shard_index") != index
            or record.get("replicate_count") != DEFAULT_SHARD_SIZE
            or record.get("count_metadata_semantic_sha256") != count_hashes[index]
            or record.get("count_artifact_sha256")
            != count_rows[index].get("artifact_sha256")
            or record.get("evidence_sha256") != expected_evidence_sha256
            or record.get("primary_family_names_sha256")
            != PRIMARY_FAMILY_NAMES_SHA256
            or not isinstance(record.get("maxima_sha256"), str)
            or len(str(record.get("maxima_sha256"))) != 64
            or not isinstance(record.get("artifact_sha256"), str)
            or len(str(record.get("artifact_sha256"))) != 64
        ):
            raise QuartileSelectionError("authorizing maxima-shard binding changed")
        if authorizing:
            artifact_path = Path(str(record.get("artifact_path", "")))
            if (
                not artifact_path.is_file()
                or file_fingerprint(artifact_path) != record["artifact_sha256"]
            ):
                raise QuartileSelectionError("authorizing maxima artifact changed")
            committed_maxima = _load_npz_array(artifact_path, "maxima")
            if (
                committed_maxima.dtype != np.dtype(np.float64)
                or committed_maxima.shape != (DEFAULT_SHARD_SIZE,)
                or not np.isfinite(committed_maxima).all()
                or _array_sha256(committed_maxima) != record["maxima_sha256"]
                or _array_sha256(
                    result.maxima[
                        index * DEFAULT_SHARD_SIZE : (index + 1) * DEFAULT_SHARD_SIZE
                    ]
                )
                != record["maxima_sha256"]
            ):
                raise QuartileSelectionError("authorizing maxima payload changed")
            recomputed_maxima = compute_bootstrap_maxima_shard(
                path_table.primary_values,
                committed_count_arrays[index],
                path_ids=path_table.path_ids,
            )
            if not np.array_equal(committed_maxima, recomputed_maxima):
                raise QuartileSelectionError(
                    "authorizing maxima do not match count/path evidence"
                )
        maxima_hashes.append(str(record["semantic_sha256"]))
    passed = bool(authorizing and result.passed and local_screen.passed)
    if not authorizing:
        decision = f"{role_value}_test_only_nonauthorizing"
    elif role_value == "selection":
        decision = (
            "quartile_specialist_selection_passed"
            if passed
            else "no_fresh_quartile_specialist_system"
        )
    else:
        decision = (
            "exact_rb_quartile_specialist_time_local_signal_confirmed"
            if passed
            else "quartile_specialist_time_local_signal_not_confirmed"
        )
    body = {
        "schema": SCHEMA + f"-{role_value}-record",
        "schema_version": 1,
        "role": role_value,
        "decision": decision,
        "path_count": int(result.path_ids.size),
        "primary_family_names": list(PRIMARY_FAMILY_NAMES),
        "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
        "local_family_names_sha256": LOCAL_FAMILY_NAMES_SHA256,
        "max_t_semantic_sha256": result.to_record()["semantic_sha256"],
        "local_screen_semantic_sha256": local_screen.to_record()["semantic_sha256"],
        "count_metadata_semantic_sha256": count_hashes,
        "maxima_metadata_semantic_sha256": maxima_hashes,
        "all_six_lower_bounds_strictly_positive": int(result.passed),
        "all_local_screens_passed": int(local_screen.passed),
        "passed": int(passed),
        "confirmation_authorized": int(role_value == "selection" and passed),
        "reverse_controller_control_planning_authorized": int(
            role_value == "confirmation" and passed
        ),
        "controller_execution_authorized": 0,
        "sampling_authorized": 0,
        "reconstruction_authorized": 0,
        "confirmation_reuse_authorized": 0,
        "negative_values_truncated": 0,
        "local_screen_inferential_claim_made": 0,
        "authorizing_evaluation": int(authorizing),
    }
    if path_table is not None:
        body["path_table_fingerprint"] = path_table.fingerprint
    return _semantic_record(body)


def selection_record(
    result: QuartileMaxTResult,
    local_screen: LocalCompatibilityScreen,
    *,
    path_table: QuartileAuditPathTable | None = None,
    count_records: Sequence[Mapping[str, Any]] | None = None,
    maxima_records: Sequence[Mapping[str, Any]] | None = None,
    authorizing: bool = True,
) -> dict[str, Any]:
    """Build the one-system selection decision record."""

    return _audit_record(
        role="selection",
        result=result,
        local_screen=local_screen,
        path_table=path_table,
        count_records=count_records,
        maxima_records=maxima_records,
        authorizing=authorizing,
    )


def confirmation_record(
    result: QuartileMaxTResult,
    local_screen: LocalCompatibilityScreen,
    *,
    path_table: QuartileAuditPathTable | None = None,
    count_records: Sequence[Mapping[str, Any]] | None = None,
    maxima_records: Sequence[Mapping[str, Any]] | None = None,
    authorizing: bool = True,
) -> dict[str, Any]:
    """Build the final untouched-audit decision record."""

    return _audit_record(
        role="confirmation",
        result=result,
        local_screen=local_screen,
        path_table=path_table,
        count_records=count_records,
        maxima_records=maxima_records,
        authorizing=authorizing,
    )


# Descriptive aliases keep call sites explicit without duplicating behavior.
quartile_selection_record = selection_record
quartile_confirmation_record = confirmation_record
restartable_six_family_max_t = restartable_quartile_max_t


__all__ = [
    "BOOTSTRAP_COUNTS_SCHEMA",
    "BOOTSTRAP_MAXIMA_SCHEMA",
    "BOOTSTRAP_PLAN_SCHEMA",
    "CONFIRMATION_NAMESPACE",
    "CONFIRMATION_PATH_START",
    "CONFIRMATION_PATH_STOP",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_CONFIRMATION_SEED",
    "DEFAULT_REPLICATES",
    "DEFAULT_SELECTION_SEED",
    "DEFAULT_SHARD_SIZE",
    "LOCAL_CELLS_PER_QUARTILE",
    "LOCAL_FAMILY_NAMES",
    "LOCAL_FAMILY_NAMES_SHA256",
    "LOCAL_SCREEN_FAMILY_NAMES",
    "LOCAL_SCREEN_FAMILY_NAMES_SHA256",
    "LOCAL_FAMILY_SIZE",
    "LocalCompatibilityScreen",
    "MAX_T_SCHEMA",
    "MINIMUM_POSITIVE_LOCAL_CELLS",
    "PHILOX_CONSTRUCTOR",
    "PRIMARY_FAMILY_NAMES",
    "PRIMARY_FAMILY_NAMES_SHA256",
    "PRIMARY_FAMILY_SIZE",
    "PRODUCTION_PATH_COUNT",
    "Q1_SENTINEL_MIDPOINT",
    "Q1_SENTINEL_PHASE",
    "QuartileAuditPathTable",
    "QuartileMaxTResult",
    "QuartileSelectionError",
    "SCHEMA",
    "SELECTION_NAMESPACE",
    "SELECTION_PATH_START",
    "SELECTION_PATH_STOP",
    "SIX_FAMILY_NAMES",
    "SIX_FAMILY_NAMES_SHA256",
    "aggregate_quartile_audit_improvements",
    "aggregate_quartile_audit_risks",
    "bootstrap_plan",
    "compute_bootstrap_maxima_shard",
    "confirmation_record",
    "count_shard_index_record",
    "evaluate_local_compatibility_screen",
    "expected_audit_sample_key_sha256",
    "generate_bootstrap_count_shard",
    "load_bootstrap_count_shards",
    "local_family_names",
    "prepare_bootstrap_count_shards",
    "primary_family_names",
    "quartile_confirmation_record",
    "quartile_max_t",
    "quartile_selection_record",
    "restartable_quartile_max_t",
    "restartable_six_family_max_t",
    "selection_record",
    "shard_artifact_paths",
]
