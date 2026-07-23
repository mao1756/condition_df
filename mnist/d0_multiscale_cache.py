from __future__ import annotations

"""Core cache primitives for multiscale Experiment 12 learnability probes.

This module deliberately does not relax the strict elementary Direct-Doob
contract in :mod:`mnist.experiment12_d0`.  A multiscale cache represents a
separate, finite-step regression experiment.  All candidate block lengths use
the same forward paths and later-state anchors, while each block stores both
the exact realized reverse transfer and the sum of the implemented positive
reverse-reference drifts evaluated at the actual intermediate later states.

No model training or reverse sampling lives here.  The module owns only:

* deterministic anchor and whole-path partition plans;
* the versioned in-memory and on-disk cache contracts;
* exact block-target arithmetic and train-only scale inference; and
* fail-closed structural, replay, and shard-integrity validation.
"""

import json
import math
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_one_image_gate import (
    array_fingerprint,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    flux_divergence_torch,
    free_drift_flux_torch,
    harmonic_mobility_channels,
    masked_reference_free_step_torch,
    natural_horizon,
    project_edge_flux_torch,
)
from mnist.experiment12_d0 import (
    Experiment12D0Config,
    _direct_reverse_free_block_baseline_from_batch,
    _lambda_mixed_data_for_paths,
    make_rate_schedule,
)


MULTISCALE_CACHE_SCHEMA = "experiment12-d0-multiscale-cache-shard"
MULTISCALE_CACHE_SCHEMA_VERSION = 1
MULTISCALE_INDEX_SCHEMA = "experiment12-d0-multiscale-cache-index"
MULTISCALE_INDEX_SCHEMA_VERSION = 1
MULTISCALE_TARGET_CONTRACT = "trajectory-summed-direct-doob-block-residual-v1"
DEFAULT_TAU_FRACTION_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class D0MultiscaleCompatibilityError(ValueError):
    """Raised when multiscale content cannot satisfy the exact cache contract."""


def _as_numpy(value: np.ndarray | Tensor, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return np.ascontiguousarray(result)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def _json_restore(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_restore(item) for item in value]
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    return value


def _as_long_tensor(value: Sequence[int] | np.ndarray | Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=torch.long, device="cpu").contiguous()


def _path_ids_array(value: Sequence[int] | np.ndarray | Tensor) -> np.ndarray:
    result = _as_numpy(value, dtype=np.dtype(np.int64)).reshape(-1)
    if result.size == 0:
        raise ValueError("path IDs must not be empty")
    if np.unique(result).size != result.size:
        raise ValueError("path IDs must be unique")
    return result


def _validate_tau_edges(value: Sequence[float] | np.ndarray) -> np.ndarray:
    edges = np.asarray(value, dtype=np.float64).reshape(-1)
    if edges.size < 2 or not np.isfinite(edges).all():
        raise ValueError("tau-fraction edges must contain at least two finite values")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("tau-fraction edges must be strictly increasing")
    if not math.isclose(float(edges[0]), 0.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("tau-fraction edges must start at 0")
    if not math.isclose(float(edges[-1]), 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("tau-fraction edges must end at 1")
    return np.ascontiguousarray(edges)


def _stratum_end_bounds(
    *,
    total_substeps: int,
    minimum_end_substep: int,
    tau_lo: float,
    tau_hi: float,
    last_bin: bool,
) -> tuple[int, int]:
    """Return inclusive completed-substep bounds for one reverse-tau stratum."""

    # tau/T = 1 - end/total.  Every bin is [lo, hi), except the last [lo, hi].
    # The lower end bound is strict in tau_hi for non-last bins.
    raw_min = int(math.floor(float(total_substeps) * (1.0 - float(tau_hi)))) + 1
    if last_bin:
        raw_min = int(math.ceil(float(total_substeps) * (1.0 - float(tau_hi))))
    raw_max = int(math.floor(float(total_substeps) * (1.0 - float(tau_lo))))
    lo = max(int(minimum_end_substep), raw_min, 1)
    hi = min(int(total_substeps), raw_max)
    return lo, hi


@dataclass(frozen=True)
class D0StratifiedAnchorPlan:
    """Common later-state anchors for all candidate block strides.

    ``end_substeps[p, a]`` is the number of completed forward elementary
    substeps at the later state.  Thus a stride ``r`` starts at ``end-r`` and
    uses forward interval indices ``end-r, ..., end-1``.
    """

    path_ids: np.ndarray
    end_substeps: np.ndarray
    stratum_indices: np.ndarray
    tau_fraction_edges: np.ndarray
    bin_counts: np.ndarray
    total_substeps: int
    max_stride: int
    seed: int
    fingerprint: str

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    @property
    def anchors_per_path(self) -> int:
        return int(self.end_substeps.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "experiment12-d0-stratified-anchor-plan",
            "schema_version": 1,
            "path_ids": self.path_ids.astype(np.int64).tolist(),
            "end_substeps": self.end_substeps.astype(np.int64).tolist(),
            "stratum_indices": self.stratum_indices.astype(np.int64).tolist(),
            "tau_fraction_edges": self.tau_fraction_edges.astype(np.float64).tolist(),
            "bin_counts": self.bin_counts.astype(np.int64).tolist(),
            "total_substeps": int(self.total_substeps),
            "max_stride": int(self.max_stride),
            "seed": int(self.seed),
            "fingerprint": str(self.fingerprint),
        }


def _anchor_plan_semantic(
    *,
    path_ids: np.ndarray,
    end_substeps: np.ndarray,
    stratum_indices: np.ndarray,
    tau_fraction_edges: np.ndarray,
    bin_counts: np.ndarray,
    total_substeps: int,
    max_stride: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "path_ids_sha256": array_fingerprint(path_ids),
        "end_substeps_sha256": array_fingerprint(end_substeps),
        "stratum_indices_sha256": array_fingerprint(stratum_indices),
        "tau_fraction_edges": tau_fraction_edges.tolist(),
        "bin_counts": bin_counts.tolist(),
        "total_substeps": int(total_substeps),
        "max_stride": int(max_stride),
        "seed": int(seed),
    }


def make_stratified_anchor_plan(
    *,
    path_ids: Sequence[int] | np.ndarray | Tensor | None = None,
    num_paths: int | None = None,
    anchors_per_path: int,
    total_substeps: int,
    max_stride: int,
    seed: int,
    tau_fraction_edges: Sequence[float] = DEFAULT_TAU_FRACTION_EDGES,
    bin_counts: Sequence[int] | np.ndarray | None = None,
) -> D0StratifiedAnchorPlan:
    """Create a deterministic, nearly equal-count tau-stratified anchor plan.

    Every path receives either ``floor(A/B)`` or ``ceil(A/B)`` anchors in each
    of the ``B`` strata.  Anchor endpoints are unique within a path/stratum and
    are sampled uniformly from the exact integer substeps in that stratum.
    """

    if path_ids is None:
        if num_paths is None or int(num_paths) <= 0:
            raise ValueError("num_paths must be positive when path_ids is omitted")
        paths = np.arange(int(num_paths), dtype=np.int64)
    else:
        paths = _path_ids_array(path_ids)
        if num_paths is not None and int(num_paths) != int(paths.size):
            raise ValueError("num_paths does not match path_ids")
    anchors = int(anchors_per_path)
    total = int(total_substeps)
    maximum = int(max_stride)
    if anchors <= 0:
        raise ValueError("anchors_per_path must be positive")
    if total <= 0 or maximum <= 0 or maximum > total:
        raise ValueError("max_stride must be in [1, total_substeps]")
    edges = _validate_tau_edges(tau_fraction_edges)
    bin_count = int(edges.size - 1)
    if bin_counts is None:
        counts = np.bincount(
            np.arange(anchors, dtype=np.int64) % bin_count, minlength=bin_count
        ).astype(np.int64)
    else:
        counts = np.asarray(bin_counts, dtype=np.int64).reshape(-1)
        if counts.shape != (bin_count,):
            raise ValueError("bin_counts must provide one count per tau stratum")
        if np.any(counts <= 0):
            raise ValueError("every prescribed tau-stratum count must be positive")
        if int(counts.sum()) != anchors:
            raise ValueError("bin_counts must sum to anchors_per_path")
    bounds: list[tuple[int, int]] = []
    for index in range(bin_count):
        bounds.append(
            _stratum_end_bounds(
                total_substeps=total,
                minimum_end_substep=maximum,
                tau_lo=float(edges[index]),
                tau_hi=float(edges[index + 1]),
                last_bin=index == bin_count - 1,
            )
        )
    unavailable = [index for index, (lo, hi) in enumerate(bounds) if lo > hi]
    if unavailable:
        raise ValueError(
            "max_stride leaves tau strata without valid anchors: "
            + ", ".join(str(value) for value in unavailable)
        )

    rng = np.random.default_rng(int(seed))
    ends = np.empty((paths.size, anchors), dtype=np.int64)
    strata = np.empty((paths.size, anchors), dtype=np.int64)
    base_bins = np.repeat(np.arange(bin_count, dtype=np.int64), counts)
    for path_axis in range(paths.size):
        assigned = rng.permutation(base_bins)
        path_ends = np.empty(anchors, dtype=np.int64)
        for bin_index in range(bin_count):
            positions = np.flatnonzero(assigned == bin_index)
            if positions.size == 0:
                continue
            lo, hi = bounds[bin_index]
            width = hi - lo + 1
            if positions.size > width:
                raise ValueError(
                    f"tau stratum {bin_index} has only {width} integer anchors for "
                    f"{positions.size} requested positions"
                )
            draws = rng.choice(width, size=int(positions.size), replace=False).astype(np.int64) + lo
            path_ends[positions] = draws
        ends[path_axis] = path_ends
        strata[path_axis] = assigned

    semantic = _anchor_plan_semantic(
        path_ids=paths,
        end_substeps=ends,
        stratum_indices=strata,
        tau_fraction_edges=edges,
        bin_counts=counts,
        total_substeps=total,
        max_stride=maximum,
        seed=int(seed),
    )
    return D0StratifiedAnchorPlan(
        path_ids=np.ascontiguousarray(paths),
        end_substeps=np.ascontiguousarray(ends),
        stratum_indices=np.ascontiguousarray(strata),
        tau_fraction_edges=edges,
        bin_counts=np.ascontiguousarray(counts),
        total_substeps=total,
        max_stride=maximum,
        seed=int(seed),
        fingerprint=config_fingerprint(semantic),
    )


def validate_anchor_plan(plan: D0StratifiedAnchorPlan) -> None:
    paths = _path_ids_array(plan.path_ids)
    ends = np.asarray(plan.end_substeps, dtype=np.int64)
    strata = np.asarray(plan.stratum_indices, dtype=np.int64)
    edges = _validate_tau_edges(plan.tau_fraction_edges)
    counts = np.asarray(plan.bin_counts, dtype=np.int64).reshape(-1)
    if ends.ndim != 2 or ends.shape[0] != paths.size or ends.shape != strata.shape:
        raise D0MultiscaleCompatibilityError("anchor plan arrays have incompatible shapes")
    if ends.shape[1] <= 0:
        raise D0MultiscaleCompatibilityError("anchor plan has no anchors")
    if counts.shape != (edges.size - 1,) or np.any(counts <= 0) or int(counts.sum()) != ends.shape[1]:
        raise D0MultiscaleCompatibilityError("anchor plan bin counts are incompatible")
    if np.any(ends < int(plan.max_stride)) or np.any(ends > int(plan.total_substeps)):
        raise D0MultiscaleCompatibilityError("anchor endpoints are outside the admissible range")
    if np.any(strata < 0) or np.any(strata >= edges.size - 1):
        raise D0MultiscaleCompatibilityError("anchor stratum index is outside tau edges")
    # Validate against the same inclusive integer bounds used by the planner.
    # Reconstructing tau/T in floating point can move an exact boundary such as
    # 1 - 32/40 = 0.2 just below its recorded stratum.
    for index in range(edges.size - 1):
        lo, hi = _stratum_end_bounds(
            total_substeps=int(plan.total_substeps),
            minimum_end_substep=int(plan.max_stride),
            tau_lo=float(edges[index]),
            tau_hi=float(edges[index + 1]),
            last_bin=index == edges.size - 2,
        )
        selected = ends[strata == index]
        if selected.size and (np.any(selected < lo) or np.any(selected > hi)):
            raise D0MultiscaleCompatibilityError(
                "anchor endpoints do not match recorded tau strata"
            )
    observed_counts = np.stack(
        [np.bincount(row, minlength=edges.size - 1) for row in strata], axis=0
    )
    if not np.all(observed_counts == counts[None, :]):
        raise D0MultiscaleCompatibilityError("anchor plan does not satisfy prescribed per-path bin counts")
    semantic = _anchor_plan_semantic(
        path_ids=paths,
        end_substeps=ends,
        stratum_indices=strata,
        tau_fraction_edges=edges,
        bin_counts=counts,
        total_substeps=int(plan.total_substeps),
        max_stride=int(plan.max_stride),
        seed=int(plan.seed),
    )
    if str(plan.fingerprint) != config_fingerprint(semantic):
        raise D0MultiscaleCompatibilityError("anchor plan fingerprint mismatch")


def slice_anchor_plan(
    plan: D0StratifiedAnchorPlan,
    path_ids: Sequence[int] | np.ndarray | Tensor,
) -> D0StratifiedAnchorPlan:
    """Return a shard-local view while retaining the global plan fingerprint."""

    validate_anchor_plan(plan)
    wanted = _path_ids_array(path_ids)
    lookup = {int(value): index for index, value in enumerate(plan.path_ids.tolist())}
    missing = [int(value) for value in wanted if int(value) not in lookup]
    if missing:
        raise KeyError("anchor plan has no path IDs: " + ", ".join(map(str, missing)))
    axes = np.asarray([lookup[int(value)] for value in wanted], dtype=np.int64)
    # A sliced plan is a distinct content object; caches separately retain the
    # full/global plan fingerprint to bind all shards to one anchor experiment.
    semantic = _anchor_plan_semantic(
        path_ids=wanted,
        end_substeps=plan.end_substeps[axes],
        stratum_indices=plan.stratum_indices[axes],
        tau_fraction_edges=plan.tau_fraction_edges,
        bin_counts=plan.bin_counts,
        total_substeps=int(plan.total_substeps),
        max_stride=int(plan.max_stride),
        seed=int(plan.seed),
    )
    return D0StratifiedAnchorPlan(
        path_ids=wanted.copy(),
        end_substeps=np.ascontiguousarray(plan.end_substeps[axes]),
        stratum_indices=np.ascontiguousarray(plan.stratum_indices[axes]),
        tau_fraction_edges=plan.tau_fraction_edges.copy(),
        bin_counts=plan.bin_counts.copy(),
        total_substeps=int(plan.total_substeps),
        max_stride=int(plan.max_stride),
        seed=int(plan.seed),
        fingerprint=config_fingerprint(semantic),
    )


@dataclass(frozen=True)
class D0ThreeWayPathSplit:
    train_path_ids: np.ndarray
    validation_path_ids: np.ndarray
    confirmation_path_ids: np.ndarray
    seed: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "experiment12-d0-three-way-path-split",
            "schema_version": 1,
            "seed": int(self.seed),
            "train_path_ids": self.train_path_ids.astype(np.int64).tolist(),
            "validation_path_ids": self.validation_path_ids.astype(np.int64).tolist(),
            "confirmation_path_ids": self.confirmation_path_ids.astype(np.int64).tolist(),
            "fingerprint": str(self.fingerprint),
        }


def deterministic_three_way_path_split(
    path_ids: Sequence[int] | np.ndarray | Tensor,
    *,
    seed: int,
    train_paths: int = 40,
    validation_paths: int = 12,
    confirmation_paths: int = 12,
) -> D0ThreeWayPathSplit:
    """Deterministically partition complete forward paths, defaulting to 40/12/12."""

    paths = _path_ids_array(path_ids)
    counts = (int(train_paths), int(validation_paths), int(confirmation_paths))
    if any(value <= 0 for value in counts):
        raise ValueError("all three path split counts must be positive")
    if sum(counts) != int(paths.size):
        raise ValueError(
            f"path split counts {counts} do not cover the {paths.size} available paths"
        )
    permutation = np.random.default_rng(int(seed)).permutation(paths)
    train_end = counts[0]
    validation_end = train_end + counts[1]
    train = np.sort(permutation[:train_end]).astype(np.int64)
    validation = np.sort(permutation[train_end:validation_end]).astype(np.int64)
    confirmation = np.sort(permutation[validation_end:]).astype(np.int64)
    semantic = {
        "seed": int(seed),
        "source_path_ids_sha256": array_fingerprint(paths),
        "train_path_ids": train.tolist(),
        "validation_path_ids": validation.tolist(),
        "confirmation_path_ids": confirmation.tolist(),
    }
    return D0ThreeWayPathSplit(
        train_path_ids=train,
        validation_path_ids=validation,
        confirmation_path_ids=confirmation,
        seed=int(seed),
        fingerprint=config_fingerprint(semantic),
    )


def validate_three_way_path_split(
    split: D0ThreeWayPathSplit,
    expected_path_ids: Sequence[int] | np.ndarray | Tensor,
) -> None:
    expected = _path_ids_array(expected_path_ids)
    groups = [
        _path_ids_array(split.train_path_ids),
        _path_ids_array(split.validation_path_ids),
        _path_ids_array(split.confirmation_path_ids),
    ]
    combined = np.concatenate(groups)
    if np.unique(combined).size != combined.size:
        raise D0MultiscaleCompatibilityError("three-way path split contains overlap")
    if not np.array_equal(np.sort(combined), np.sort(expected)):
        raise D0MultiscaleCompatibilityError("three-way path split does not cover expected paths")
    semantic = {
        "seed": int(split.seed),
        "source_path_ids_sha256": array_fingerprint(expected),
        "train_path_ids": groups[0].tolist(),
        "validation_path_ids": groups[1].tolist(),
        "confirmation_path_ids": groups[2].tolist(),
    }
    if str(split.fingerprint) != config_fingerprint(semantic):
        raise D0MultiscaleCompatibilityError("three-way path split fingerprint mismatch")


@dataclass(frozen=True)
class D0MultiscaleCache:
    """Structured cache whose path and anchor axes are shared across strides."""

    strides: Tensor
    path_ids: Tensor
    later_states: Tensor
    tau: Tensor
    labels: Tensor
    end_substeps: Tensor
    anchor_strata: Tensor
    tau_fraction_edges: np.ndarray
    start_images: Tensor
    earlier_states: Tensor
    reverse_transfers: Tensor
    reference_transfers: Tensor
    innovations: Tensor
    masks: Tensor
    terminal_states: np.ndarray
    source_indices: np.ndarray
    requested_labels: np.ndarray
    rate_schedule: np.ndarray
    horizon: float
    dt_sub: float
    sample_steps: int
    reference_substeps: int
    lambda_mix: float
    anchor_plan_fingerprint: str
    target_contract: str = MULTISCALE_TARGET_CONTRACT
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def path_count(self) -> int:
        return int(self.path_ids.numel())

    @property
    def stride_count(self) -> int:
        return int(self.strides.numel())

    @property
    def anchors_per_path(self) -> int:
        return int(self.end_substeps.shape[1])

    @property
    def grid_size(self) -> int:
        return int(self.reverse_transfers.shape[-1])

    @property
    def total_substeps(self) -> int:
        return int(self.sample_steps * self.reference_substeps)

    def stride_axis(self, stride: int) -> int:
        matches = torch.nonzero(self.strides == int(stride), as_tuple=True)[0]
        if int(matches.numel()) != 1:
            raise KeyError(f"cache does not contain stride {int(stride)}")
        return int(matches.item())

    def start_substeps(self, stride: int) -> Tensor:
        return self.end_substeps - int(stride)


_CACHE_TENSOR_FIELDS = (
    "strides",
    "path_ids",
    "later_states",
    "tau",
    "labels",
    "end_substeps",
    "anchor_strata",
    "start_images",
    "earlier_states",
    "reverse_transfers",
    "reference_transfers",
    "innovations",
    "masks",
)
_CACHE_ARRAY_FIELDS = (
    "tau_fraction_edges",
    "terminal_states",
    "source_indices",
    "requested_labels",
    "rate_schedule",
)
_CACHE_SCALAR_FIELDS = (
    "horizon",
    "dt_sub",
    "sample_steps",
    "reference_substeps",
    "lambda_mix",
    "anchor_plan_fingerprint",
    "target_contract",
    "diagnostics",
)


def validate_multiscale_cache(cache: D0MultiscaleCache) -> None:
    strides = _as_numpy(cache.strides, dtype=np.dtype(np.int64)).reshape(-1)
    paths = _as_numpy(cache.path_ids, dtype=np.dtype(np.int64)).reshape(-1)
    if strides.size == 0 or np.any(strides <= 0) or np.unique(strides).size != strides.size:
        raise D0MultiscaleCompatibilityError("cache strides must be unique and positive")
    if not np.array_equal(strides, np.sort(strides)):
        raise D0MultiscaleCompatibilityError("cache strides must be sorted")
    if paths.size == 0 or np.unique(paths).size != paths.size:
        raise D0MultiscaleCompatibilityError("cache path IDs must be non-empty and unique")
    r_count = int(strides.size)
    p_count = int(paths.size)
    if cache.later_states.ndim != 3:
        raise D0MultiscaleCompatibilityError("later_states must have shape (paths, anchors, pixels)")
    if int(cache.later_states.shape[0]) != p_count:
        raise D0MultiscaleCompatibilityError("later_states path axis differs from path_ids")
    anchors = int(cache.later_states.shape[1])
    pixels = int(cache.later_states.shape[2])
    n = int(round(math.sqrt(float(pixels))))
    if anchors <= 0 or n * n != pixels:
        raise D0MultiscaleCompatibilityError("cache pixel axis is not a non-empty square grid")
    if cache.tau.shape != (p_count, anchors):
        raise D0MultiscaleCompatibilityError("tau shape does not match path/anchor axes")
    if cache.end_substeps.shape != (p_count, anchors) or cache.anchor_strata.shape != (p_count, anchors):
        raise D0MultiscaleCompatibilityError("anchor arrays do not match path/anchor axes")
    if cache.labels.shape != (p_count,) or cache.start_images.shape != (p_count, pixels):
        raise D0MultiscaleCompatibilityError("path-level tensor shapes are incompatible")
    state_shape = (r_count, p_count, anchors, pixels)
    edge_shape = (r_count, p_count, anchors, 2, n, n)
    if cache.earlier_states.shape != state_shape:
        raise D0MultiscaleCompatibilityError("earlier_states shape is incompatible")
    for name in ("reverse_transfers", "reference_transfers", "innovations", "masks"):
        if getattr(cache, name).shape != edge_shape:
            raise D0MultiscaleCompatibilityError(f"{name} shape is incompatible")
    if cache.masks.dtype != torch.bool:
        raise D0MultiscaleCompatibilityError("masks must be boolean")
    for name in (
        "strides",
        "path_ids",
        "later_states",
        "tau",
        "labels",
        "end_substeps",
        "anchor_strata",
        "start_images",
        "earlier_states",
        "reverse_transfers",
        "reference_transfers",
        "innovations",
        "masks",
    ):
        if getattr(cache, name).device.type != "cpu":
            raise D0MultiscaleCompatibilityError(f"cache field {name} must reside on the CPU")
    for name in (
        "later_states",
        "tau",
        "start_images",
        "earlier_states",
        "reverse_transfers",
        "reference_transfers",
        "innovations",
    ):
        if not bool(torch.isfinite(getattr(cache, name)).all()):
            raise D0MultiscaleCompatibilityError(f"cache field {name} contains non-finite values")
    total = int(cache.sample_steps) * int(cache.reference_substeps)
    if int(cache.sample_steps) <= 0 or int(cache.reference_substeps) <= 0 or total <= 0:
        raise D0MultiscaleCompatibilityError("sample_steps and reference_substeps must be positive")
    if any(total % int(stride) != 0 for stride in strides):
        raise D0MultiscaleCompatibilityError("every stride must divide the total elementary substeps")
    ends = _as_numpy(cache.end_substeps, dtype=np.dtype(np.int64))
    if np.any(ends < int(strides.max())) or np.any(ends > total):
        raise D0MultiscaleCompatibilityError("cache anchor endpoints are outside the common stride range")
    if not math.isfinite(float(cache.horizon)) or float(cache.horizon) <= 0.0:
        raise D0MultiscaleCompatibilityError("cache horizon must be finite and positive")
    expected_dt = float(cache.horizon) / float(total)
    if not math.isclose(float(cache.dt_sub), expected_dt, rel_tol=1e-12, abs_tol=1e-15):
        raise D0MultiscaleCompatibilityError("cache dt_sub is inconsistent with its temporal grid")
    expected_tau = float(cache.horizon) - torch.as_tensor(
        ends, dtype=torch.float64
    ) * float(cache.dt_sub)
    tau_error = float((cache.tau.double() - expected_tau).abs().max())
    if tau_error > max(1e-7 * float(cache.horizon), 1e-12):
        raise D0MultiscaleCompatibilityError("cache tau values are inconsistent with anchor endpoints")
    edges = _validate_tau_edges(cache.tau_fraction_edges)
    strata = _as_numpy(cache.anchor_strata, dtype=np.dtype(np.int64))
    if np.any(strata < 0) or np.any(strata >= edges.size - 1):
        raise D0MultiscaleCompatibilityError("cache anchor strata are outside tau edges")
    for index in range(edges.size - 1):
        lo, hi = _stratum_end_bounds(
            total_substeps=total,
            minimum_end_substep=int(strides.max()),
            tau_lo=float(edges[index]),
            tau_hi=float(edges[index + 1]),
            last_bin=index == edges.size - 2,
        )
        selected = ends[strata == index]
        if selected.size and (np.any(selected < lo) or np.any(selected > hi)):
            raise D0MultiscaleCompatibilityError(
                "cache anchor strata are inconsistent with tau"
            )
    for name in ("terminal_states", "source_indices", "requested_labels"):
        if int(np.asarray(getattr(cache, name)).shape[0]) != p_count:
            raise D0MultiscaleCompatibilityError(f"path-level array {name} has the wrong length")
    terminal = np.asarray(cache.terminal_states)
    if terminal.ndim not in {2, 3} or int(np.prod(terminal.shape[1:])) != pixels:
        raise D0MultiscaleCompatibilityError("terminal_states grid shape is incompatible")
    if np.asarray(cache.rate_schedule).shape != (int(cache.sample_steps),):
        raise D0MultiscaleCompatibilityError("rate_schedule length differs from sample_steps")
    if not np.isfinite(cache.rate_schedule).all() or np.any(np.asarray(cache.rate_schedule) < 0.0):
        raise D0MultiscaleCompatibilityError("rate_schedule must be finite and non-negative")
    if not str(cache.anchor_plan_fingerprint):
        raise D0MultiscaleCompatibilityError("cache has no anchor-plan fingerprint")
    if str(cache.target_contract) != MULTISCALE_TARGET_CONTRACT:
        raise D0MultiscaleCompatibilityError("cache target contract is unsupported")


def exact_reverse_reference_step_transfer(
    later_states: Tensor,
    global_substeps: Tensor | Sequence[int] | np.ndarray,
    *,
    rate_schedule: Tensor | Sequence[float] | np.ndarray,
    reference_substeps: int,
    dt_sub: float,
    dynamics_config: DirectFluxMNISTConfig,
) -> Tensor:
    """Evaluate ``+b_ref(S[q+1]) dt`` at each actual intermediate later state."""

    if later_states.ndim != 2:
        raise ValueError("later_states must have shape (batch, pixels)")
    q = torch.as_tensor(global_substeps, dtype=torch.long, device=later_states.device).reshape(-1)
    if int(q.numel()) != int(later_states.shape[0]):
        raise ValueError("global_substeps must align with later_states")
    ref = int(reference_substeps)
    if ref <= 0 or not math.isfinite(float(dt_sub)) or float(dt_sub) <= 0.0:
        raise ValueError("reference_substeps and dt_sub must be positive")
    rates = torch.as_tensor(rate_schedule, dtype=later_states.dtype, device=later_states.device).reshape(-1)
    if rates.numel() == 0 or not bool(torch.isfinite(rates).all()) or bool((rates < 0.0).any()):
        raise ValueError("rate_schedule must be finite, non-negative, and non-empty")
    outer = torch.div(q, ref, rounding_mode="floor")
    if bool((q < 0).any()) or bool((outer >= rates.numel()).any()):
        raise IndexError("global substep is outside the rate schedule")
    rate = rates.index_select(0, outer).view(-1, 1, 1, 1)
    return rate * free_drift_flux_torch(later_states, dynamics_config) * float(dt_sub)


def aggregate_aligned_block_quantities(
    reverse_step_transfers: Tensor,
    reference_step_transfers: Tensor,
    raw_innovations: Tensor,
    valid_masks: Tensor,
    strides: Sequence[int] | np.ndarray | Tensor,
) -> dict[str, Tensor]:
    """Aggregate aligned blocks ending at the final supplied elementary step.

    Step tensors have leading time axis ``Q``.  For each requested ``r`` this
    helper sums the last ``r`` physical transfers, normalizes innovations by
    ``sqrt(r)``, and ANDs masks.  It is useful both to implement a cache builder
    and to assert multiscale arithmetic on controlled fixtures.
    """

    if reverse_step_transfers.shape != reference_step_transfers.shape or reverse_step_transfers.shape != raw_innovations.shape:
        raise ValueError("step transfer and innovation tensors must have identical shapes")
    if valid_masks.shape != reverse_step_transfers.shape or valid_masks.dtype != torch.bool:
        raise ValueError("valid_masks must be boolean and match the transfer shape")
    if reverse_step_transfers.ndim < 2:
        raise ValueError("step tensors require a leading time axis")
    stride_values = _as_long_tensor(strides).reshape(-1)
    if stride_values.numel() == 0 or bool((stride_values <= 0).any()):
        raise ValueError("strides must be positive and non-empty")
    q_count = int(reverse_step_transfers.shape[0])
    if int(stride_values.max()) > q_count:
        raise ValueError("a requested stride exceeds the supplied step history")
    reverse_parts: list[Tensor] = []
    reference_parts: list[Tensor] = []
    innovation_parts: list[Tensor] = []
    mask_parts: list[Tensor] = []
    for stride in stride_values.tolist():
        reverse_parts.append(reverse_step_transfers[-int(stride) :].sum(dim=0))
        reference_parts.append(reference_step_transfers[-int(stride) :].sum(dim=0))
        innovation_parts.append(raw_innovations[-int(stride) :].sum(dim=0) / math.sqrt(float(stride)))
        mask_parts.append(valid_masks[-int(stride) :].all(dim=0))
    return {
        "reverse_transfers": torch.stack(reverse_parts, dim=0),
        "reference_transfers": torch.stack(reference_parts, dim=0),
        "innovations": torch.stack(innovation_parts, dim=0),
        "masks": torch.stack(mask_parts, dim=0),
    }


def derive_projected_block_residual(
    reverse_transfers: Tensor,
    reference_transfers: Tensor,
    *,
    grid_size: int,
) -> Tensor:
    """Return ``Proj(reverse_total - exact_reference_total)``."""

    if reverse_transfers.shape != reference_transfers.shape:
        raise ValueError("reverse and reference transfers must have identical shapes")
    if reverse_transfers.shape[-3:] != (2, int(grid_size), int(grid_size)):
        raise ValueError("transfer tensors do not match grid_size")
    flat = (reverse_transfers - reference_transfers).reshape(-1, 2, int(grid_size), int(grid_size))
    projected = project_edge_flux_torch(flat, grid_size=int(grid_size))
    return projected.reshape(reverse_transfers.shape)


def _path_axes(cache: D0MultiscaleCache, path_ids: Sequence[int] | np.ndarray | Tensor | None) -> np.ndarray:
    cache_paths = _as_numpy(cache.path_ids, dtype=np.dtype(np.int64)).reshape(-1)
    if path_ids is None:
        return np.arange(cache_paths.size, dtype=np.int64)
    wanted = _path_ids_array(path_ids)
    lookup = {int(value): index for index, value in enumerate(cache_paths.tolist())}
    missing = [int(value) for value in wanted if int(value) not in lookup]
    if missing:
        raise KeyError("cache has no path IDs: " + ", ".join(map(str, missing)))
    return np.asarray([lookup[int(value)] for value in wanted], dtype=np.int64)


@torch.no_grad()
def block_residual_targets(
    cache: D0MultiscaleCache,
    dynamics_config: DirectFluxMNISTConfig,
    *,
    stride: int,
    path_ids: Sequence[int] | np.ndarray | Tensor | None = None,
    scale: float | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> Tensor:
    """Return path-major, anchor-minor projected block targets on the CPU."""

    validate_multiscale_cache(cache)
    if int(dynamics_config.grid_size) != int(cache.grid_size):
        raise ValueError("dynamics grid does not match multiscale cache")
    stride_axis = cache.stride_axis(int(stride))
    axes = _path_axes(cache, path_ids)
    reverse = cache.reverse_transfers[stride_axis].index_select(0, torch.as_tensor(axes, dtype=torch.long))
    reference = cache.reference_transfers[stride_axis].index_select(0, torch.as_tensor(axes, dtype=torch.long))
    reverse = reverse.reshape(-1, 2, cache.grid_size, cache.grid_size)
    reference = reference.reshape_as(reverse)
    device_obj = torch.device(device)
    pieces: list[Tensor] = []
    chunk = max(1, int(batch_size))
    for start in range(0, int(reverse.shape[0]), chunk):
        stop = min(int(reverse.shape[0]), start + chunk)
        target = derive_projected_block_residual(
            reverse[start:stop].to(device_obj),
            reference[start:stop].to(device_obj),
            grid_size=cache.grid_size,
        )
        if scale is not None:
            if not math.isfinite(float(scale)) or float(scale) <= 0.0:
                raise ValueError("target scale must be finite and positive")
            target = target / float(scale)
        pieces.append(target.detach().cpu())
    return torch.cat(pieces, dim=0)


def infer_training_block_scales(
    cache: D0MultiscaleCache,
    dynamics_config: DirectFluxMNISTConfig,
    train_path_ids: Sequence[int] | np.ndarray | Tensor,
    *,
    floor: float = 1e-6,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> dict[int, float]:
    """Infer one global RMS per stride, exclusively from whole training paths."""

    if not math.isfinite(float(floor)) or float(floor) <= 0.0:
        raise ValueError("scale floor must be finite and positive")
    # Resolve once so missing/duplicate path IDs fail before any partial result.
    _path_axes(cache, train_path_ids)
    result: dict[int, float] = {}
    for stride in cache.strides.tolist():
        target = block_residual_targets(
            cache,
            dynamics_config,
            stride=int(stride),
            path_ids=train_path_ids,
            device=device,
            batch_size=batch_size,
        )
        finite = target[torch.isfinite(target)]
        if finite.numel() == 0:
            raise ValueError(f"stride {int(stride)} training targets contain no finite values")
        value = math.sqrt(float(finite.double().square().mean().item()))
        value = max(value, float(floor))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"stride {int(stride)} target scale is invalid")
        result[int(stride)] = float(value)
    return result


def build_multiscale_cache_shard(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    anchor_plan: D0StratifiedAnchorPlan,
    strides: Sequence[int] | np.ndarray | Tensor,
    device: torch.device | str,
    seed: int,
    global_anchor_plan_fingerprint: str | None = None,
    verify_slow_sums: bool = False,
    show_progress: bool = True,
) -> D0MultiscaleCache:
    """Build one restartable path shard with exact prefix-difference blocks.

    Prefix sums make the cost essentially independent of the number of
    candidate strides.  At every elementary forward substep this builder adds
    the exact applied reverse transfer, the positive reference drift evaluated
    at that substep's actual later state, the raw innovation, and an integer
    invalid-edge count.  Sparse snapshots at ``anchor-r`` and ``anchor`` then
    produce every requested block by subtraction.

    The function intentionally bypasses ``_validate_direct_doob_config``:
    multiscale targets use their own explicitly versioned finite-step contract
    and cannot be loaded as strict elementary caches.
    """

    validate_anchor_plan(anchor_plan)
    stride_values = np.asarray(_as_numpy(_as_long_tensor(strides)), dtype=np.int64).reshape(-1)
    if stride_values.size == 0 or np.any(stride_values <= 0):
        raise ValueError("strides must be positive and non-empty")
    stride_values = np.sort(np.unique(stride_values))
    sample_steps = int(d0_config.sample_steps)
    reference_substeps = int(d0_config.reference_substeps)
    total_substeps = sample_steps * reference_substeps
    if total_substeps != int(anchor_plan.total_substeps):
        raise ValueError("anchor plan total_substeps differs from D0 configuration")
    if int(stride_values.max()) != int(anchor_plan.max_stride):
        raise ValueError("anchor plan max_stride must equal the largest requested stride")
    if any(total_substeps % int(stride) != 0 for stride in stride_values):
        raise ValueError("every multiscale stride must divide total_substeps")
    if str(d0_config.cache_build_mode).strip().lower().replace("_", "-") not in {
        "substep",
        "exact-substep",
        "exact",
    }:
        raise ValueError("multiscale cache construction requires exact substep mode")
    if int(dynamics_config.grid_size) <= 0:
        raise ValueError("dynamics grid_size must be positive")
    if not math.isfinite(float(d0_config.lambda_mix)):
        raise ValueError("lambda_mix must be finite")

    device_obj = torch.device(device)
    n = int(dynamics_config.grid_size)
    p_count = int(anchor_plan.path_count)
    anchors = int(anchor_plan.anchors_per_path)
    r_count = int(stride_values.size)
    horizon = float(natural_horizon(dynamics_config))
    dt_outer = horizon / float(sample_steps)
    dt_sub = horizon / float(total_substeps)
    rate_schedule = make_rate_schedule(
        sample_steps,
        tau_eff=float(d0_config.tau_eff),
        horizon=horizon,
        time_change_mode=str(d0_config.time_change_mode),
        ramp=str(d0_config.rate_ramp),
        ramp_ratio=float(d0_config.rate_ramp_ratio),
        rate_min=d0_config.reference_rate_min,
        rate_max=d0_config.reference_rate_max,
    )

    numpy_rng = np.random.default_rng(int(seed))
    initial_np, labels_np, source_idx_np = _lambda_mixed_data_for_paths(
        dataset_images,
        dataset_labels,
        count=p_count,
        lambda_mix=float(d0_config.lambda_mix),
        grid_size=n,
        rng=numpy_rng,
        single_image_overfit=bool(d0_config.single_image_overfit),
        single_image_index=int(d0_config.single_image_index),
        single_image_label=d0_config.single_image_label,
    )

    ends = torch.as_tensor(anchor_plan.end_substeps, dtype=torch.long, device=device_obj)
    stride_t = torch.as_tensor(stride_values, dtype=torch.long, device=device_obj)

    # Capture boundaries are scientific metadata, not dynamic device state.
    # Precompute their outer-step/local-substep locations once on the host so
    # the rollout never has to ask the GPU whether a boundary matches the
    # current elementary q.  The old per-q ``bool(matching.any())`` path
    # introduced several device synchronizations per elementary substep.
    starts_np = np.asarray(anchor_plan.end_substeps, dtype=np.int64)[None, :, :] - (
        stride_values[:, None, None]
    )
    ends_np = np.asarray(anchor_plan.end_substeps, dtype=np.int64)
    if np.any(starts_np < 0):
        raise RuntimeError("anchor plan produced a negative block start")

    def make_capture_index(
        boundaries: np.ndarray,
        *,
        include_stride_axis: bool,
    ) -> dict[str, Any]:
        values = np.asarray(boundaries, dtype=np.int64)
        if np.any(values < 0) or np.any(values > total_substeps):
            raise RuntimeError("prefix capture boundary is outside the rollout")
        axes = np.indices(values.shape, dtype=np.int64)
        if include_stride_axis:
            stride_axes = axes[0].reshape(-1)
            path_axes = axes[1].reshape(-1)
            anchor_axes = axes[2].reshape(-1)
        else:
            stride_axes = np.empty(values.size, dtype=np.int64)
            path_axes = axes[0].reshape(-1)
            anchor_axes = axes[1].reshape(-1)
        times = values.reshape(-1)
        zero = times == 0
        positive = ~zero
        positive_times = times[positive]
        outer_axes = (positive_times - 1) // reference_substeps
        local_axes = (positive_times - 1) % reference_substeps
        order = np.argsort(outer_axes, kind="stable")
        outer_axes = outer_axes[order]
        counts = np.bincount(outer_axes, minlength=sample_steps)
        offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        )

        def device_indices(values_np: np.ndarray) -> Tensor:
            return torch.as_tensor(
                np.ascontiguousarray(values_np),
                dtype=torch.long,
                device=device_obj,
            )

        result: dict[str, Any] = {
            "offsets": offsets,
            "local": device_indices(local_axes[order]),
            "path": device_indices(path_axes[positive][order]),
            "anchor": device_indices(anchor_axes[positive][order]),
            "zero_path": device_indices(path_axes[zero]),
            "zero_anchor": device_indices(anchor_axes[zero]),
        }
        if include_stride_axis:
            result["stride"] = device_indices(stride_axes[positive][order])
            result["zero_stride"] = device_indices(stride_axes[zero])
        return result

    start_captures = make_capture_index(starts_np, include_stride_axis=True)
    end_captures = make_capture_index(ends_np, include_stride_axis=False)

    # The optional slow verifier deliberately does not use the global prefix
    # subtraction.  Precompute each interval's intersections with outer steps,
    # then sum those explicit tensor slices during the rollout.  This preserves
    # an independent orientation/indexing check without per-q device predicates.
    slow_segments: list[list[tuple[int, int, int, int, int]]] | None = None
    if bool(verify_slow_sums):
        slow_segments = [[] for _ in range(sample_steps)]
        for stride_axis in range(r_count):
            for path_axis in range(p_count):
                for anchor_axis in range(anchors):
                    start_q = int(starts_np[stride_axis, path_axis, anchor_axis])
                    end_q = int(ends_np[path_axis, anchor_axis])
                    first_outer = start_q // reference_substeps
                    last_outer = (end_q - 1) // reference_substeps
                    for outer_axis in range(first_outer, last_outer + 1):
                        outer_start = outer_axis * reference_substeps
                        local_start = max(start_q - outer_start, 0)
                        local_end = min(end_q - outer_start, reference_substeps)
                        if local_start < local_end:
                            slow_segments[outer_axis].append(
                                (
                                    stride_axis,
                                    path_axis,
                                    anchor_axis,
                                    local_start,
                                    local_end,
                                )
                            )

    edge_shape = (2, n, n)
    later_states = torch.empty((p_count, anchors, n * n), dtype=torch.float32, device=device_obj)
    earlier_states = torch.empty((r_count, p_count, anchors, n * n), dtype=torch.float32, device=device_obj)
    # Prefix differences can subtract two long trajectory sums to recover a
    # very small local transfer.  Accumulate and snapshot in float64 so the
    # exact r=1 target is not lost to float32 cancellation; cache payloads are
    # cast back to their versioned float32 storage dtype below.
    prefix_reverse_start = torch.empty((r_count, p_count, anchors, *edge_shape), dtype=torch.float64, device=device_obj)
    prefix_reference_start = torch.empty_like(prefix_reverse_start)
    prefix_innovation_start = torch.empty_like(prefix_reverse_start)
    prefix_invalid_start = torch.empty((r_count, p_count, anchors, *edge_shape), dtype=torch.int32, device=device_obj)
    prefix_reverse_end = torch.empty((p_count, anchors, *edge_shape), dtype=torch.float64, device=device_obj)
    prefix_reference_end = torch.empty_like(prefix_reverse_end)
    prefix_innovation_end = torch.empty_like(prefix_reverse_end)
    prefix_invalid_end = torch.empty((p_count, anchors, *edge_shape), dtype=torch.int32, device=device_obj)
    start_filled = torch.zeros((r_count, p_count, anchors), dtype=torch.bool, device=device_obj)
    end_filled = torch.zeros((p_count, anchors), dtype=torch.bool, device=device_obj)
    slow_reverse = (
        torch.zeros_like(prefix_reverse_start) if bool(verify_slow_sums) else None
    )
    slow_reference = (
        torch.zeros_like(prefix_reference_start) if bool(verify_slow_sums) else None
    )
    slow_innovation = (
        torch.zeros_like(prefix_innovation_start) if bool(verify_slow_sums) else None
    )
    slow_invalid = (
        torch.zeros_like(prefix_invalid_start) if bool(verify_slow_sums) else None
    )

    diagnostic_names = (
        "limited_edges",
        "proposed_edges",
        "mobility_weight_sum",
        "limited_mobility_weight_sum",
        "noise_energy_sum",
        "limited_noise_energy_sum",
        "floor_correction_l1",
        "renorm_correction_l1",
        "floor_touched_pixels",
        "floor_proposed_pixels",
        "nonfinite_edges",
    )
    diagnostic_zero = torch.zeros((), dtype=torch.float64, device=device_obj)
    device_diagnostic_totals = {
        name: diagnostic_zero.clone() for name in diagnostic_names
    }

    cuda_devices: list[int] = []
    if device_obj.type == "cuda":
        cuda_devices = [
            int(device_obj.index)
            if device_obj.index is not None
            else int(torch.cuda.current_device())
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if device_obj.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        states = torch.as_tensor(initial_np, dtype=torch.float32, device=device_obj)
        cumulative_reverse = torch.zeros((p_count, *edge_shape), dtype=torch.float64, device=device_obj)
        cumulative_reference = torch.zeros_like(cumulative_reverse)
        cumulative_innovation = torch.zeros_like(cumulative_reverse)
        cumulative_invalid = torch.zeros((p_count, *edge_shape), dtype=torch.int32, device=device_obj)

        zero_stride = start_captures["zero_stride"]
        zero_path = start_captures["zero_path"]
        zero_anchor = start_captures["zero_anchor"]
        if int(zero_path.numel()) > 0:
            earlier_states[zero_stride, zero_path, zero_anchor] = states.index_select(
                0, zero_path
            )
            prefix_reverse_start[zero_stride, zero_path, zero_anchor] = (
                cumulative_reverse.index_select(0, zero_path)
            )
            prefix_reference_start[zero_stride, zero_path, zero_anchor] = (
                cumulative_reference.index_select(0, zero_path)
            )
            prefix_innovation_start[zero_stride, zero_path, zero_anchor] = (
                cumulative_innovation.index_select(0, zero_path)
            )
            prefix_invalid_start[zero_stride, zero_path, zero_anchor] = (
                cumulative_invalid.index_select(0, zero_path)
            )
            start_filled[zero_stride, zero_path, zero_anchor] = True

        zero_end_path = end_captures["zero_path"]
        zero_end_anchor = end_captures["zero_anchor"]
        if int(zero_end_path.numel()) > 0:
            later_states[zero_end_path, zero_end_anchor] = states.index_select(
                0, zero_end_path
            )
            prefix_reverse_end[zero_end_path, zero_end_anchor] = (
                cumulative_reverse.index_select(0, zero_end_path)
            )
            prefix_reference_end[zero_end_path, zero_end_anchor] = (
                cumulative_reference.index_select(0, zero_end_path)
            )
            prefix_innovation_end[zero_end_path, zero_end_anchor] = (
                cumulative_innovation.index_select(0, zero_end_path)
            )
            prefix_invalid_end[zero_end_path, zero_end_anchor] = (
                cumulative_invalid.index_select(0, zero_end_path)
            )
            end_filled[zero_end_path, zero_end_anchor] = True

        iterator: Any = range(sample_steps)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, total=sample_steps, desc="D0 multiscale shard")
            except Exception:  # pragma: no cover - progress is optional
                pass
        for outer_k in iterator:
            rate = float(rate_schedule[int(outer_k)])
            result = masked_reference_free_step_torch(
                states,
                dt_outer,
                dynamics_config,
                free_weight=rate,
                noise_weight=math.sqrt(max(rate, 0.0)),
                substeps=reference_substeps,
                stiffness_fraction=float(dynamics_config.limiter_fraction),
                return_innovations=True,
                return_substep_states=True,
                return_realized_transfers=True,
                collect_diagnostics=True,
                diagnostics_device=True,
            )
            if (
                result.raw_innovations is None
                or result.valid_edge_mask is None
                or result.substep_states is None
                or result.realized_edge_transfers is None
                or result.device_diagnostics is None
            ):
                raise RuntimeError("reference integrator omitted exact multiscale cache tensors")
            raw = result.raw_innovations
            valid = result.valid_edge_mask
            sub_states = result.substep_states
            realized = result.realized_edge_transfers
            reference_steps = (
                rate
                * free_drift_flux_torch(
                    sub_states.reshape(reference_substeps * p_count, n * n),
                    dynamics_config,
                ).reshape(reference_substeps, p_count, *edge_shape)
                * dt_sub
            )
            reverse_prefix = torch.cumsum(-realized, dim=0, dtype=torch.float64)
            reference_prefix = torch.cumsum(
                reference_steps, dim=0, dtype=torch.float64
            )
            innovation_prefix = torch.cumsum(raw, dim=0, dtype=torch.float64)
            invalid_steps = (~valid).to(torch.int32)
            invalid_prefix = torch.cumsum(invalid_steps, dim=0, dtype=torch.int32)

            start_begin = int(start_captures["offsets"][int(outer_k)])
            start_end = int(start_captures["offsets"][int(outer_k) + 1])
            if start_begin < start_end:
                capture_slice = slice(start_begin, start_end)
                stride_axes = start_captures["stride"][capture_slice]
                path_axes = start_captures["path"][capture_slice]
                anchor_axes = start_captures["anchor"][capture_slice]
                local_axes = start_captures["local"][capture_slice]
                earlier_states[stride_axes, path_axes, anchor_axes] = sub_states[
                    local_axes, path_axes
                ]
                prefix_reverse_start[stride_axes, path_axes, anchor_axes] = (
                    cumulative_reverse.index_select(0, path_axes)
                    + reverse_prefix[local_axes, path_axes]
                )
                prefix_reference_start[stride_axes, path_axes, anchor_axes] = (
                    cumulative_reference.index_select(0, path_axes)
                    + reference_prefix[local_axes, path_axes]
                )
                prefix_innovation_start[stride_axes, path_axes, anchor_axes] = (
                    cumulative_innovation.index_select(0, path_axes)
                    + innovation_prefix[local_axes, path_axes]
                )
                prefix_invalid_start[stride_axes, path_axes, anchor_axes] = (
                    cumulative_invalid.index_select(0, path_axes)
                    + invalid_prefix[local_axes, path_axes]
                )
                start_filled[stride_axes, path_axes, anchor_axes] = True

            end_begin = int(end_captures["offsets"][int(outer_k)])
            end_end = int(end_captures["offsets"][int(outer_k) + 1])
            if end_begin < end_end:
                capture_slice = slice(end_begin, end_end)
                path_axes = end_captures["path"][capture_slice]
                anchor_axes = end_captures["anchor"][capture_slice]
                local_axes = end_captures["local"][capture_slice]
                later_states[path_axes, anchor_axes] = sub_states[local_axes, path_axes]
                prefix_reverse_end[path_axes, anchor_axes] = (
                    cumulative_reverse.index_select(0, path_axes)
                    + reverse_prefix[local_axes, path_axes]
                )
                prefix_reference_end[path_axes, anchor_axes] = (
                    cumulative_reference.index_select(0, path_axes)
                    + reference_prefix[local_axes, path_axes]
                )
                prefix_innovation_end[path_axes, anchor_axes] = (
                    cumulative_innovation.index_select(0, path_axes)
                    + innovation_prefix[local_axes, path_axes]
                )
                prefix_invalid_end[path_axes, anchor_axes] = (
                    cumulative_invalid.index_select(0, path_axes)
                    + invalid_prefix[local_axes, path_axes]
                )
                end_filled[path_axes, anchor_axes] = True

            if slow_segments is not None:
                assert slow_reverse is not None
                assert slow_reference is not None
                assert slow_innovation is not None
                assert slow_invalid is not None
                for (
                    stride_axis,
                    path_axis,
                    anchor_axis,
                    local_start,
                    local_end,
                ) in slow_segments[int(outer_k)]:
                    slow_reverse[stride_axis, path_axis, anchor_axis].sub_(
                        realized[local_start:local_end, path_axis].sum(
                            dim=0, dtype=torch.float64
                        )
                    )
                    slow_reference[stride_axis, path_axis, anchor_axis].add_(
                        reference_steps[local_start:local_end, path_axis].sum(
                            dim=0, dtype=torch.float64
                        )
                    )
                    slow_innovation[stride_axis, path_axis, anchor_axis].add_(
                        raw[local_start:local_end, path_axis].sum(
                            dim=0, dtype=torch.float64
                        )
                    )
                    slow_invalid[stride_axis, path_axis, anchor_axis].add_(
                        invalid_steps[local_start:local_end, path_axis].sum(
                            dim=0, dtype=torch.int32
                        )
                    )

            cumulative_reverse = cumulative_reverse + reverse_prefix[-1]
            cumulative_reference = cumulative_reference + reference_prefix[-1]
            cumulative_innovation = cumulative_innovation + innovation_prefix[-1]
            cumulative_invalid = cumulative_invalid + invalid_prefix[-1]
            states = result.states
            for name in diagnostic_names:
                if name not in {"limited_edges", "proposed_edges"}:
                    device_diagnostic_totals[name].add_(result.device_diagnostics[name])
            # Preserve the cache mask-tensor denominator used by the original
            # builder (including the fixed boundary slots).  The integrator's
            # own proposed-edge counter covers physical edge classes only.
            device_diagnostic_totals["limited_edges"].add_(
                (~valid).count_nonzero().to(torch.float64)
            )
            device_diagnostic_totals["proposed_edges"].add_(float(valid.numel()))

    diagnostic_values = torch.stack(
        [device_diagnostic_totals[name] for name in diagnostic_names]
    ).detach().cpu().tolist()
    diagnostic_totals = dict(zip(diagnostic_names, diagnostic_values, strict=True))
    total_masked_edges = int(round(diagnostic_totals["limited_edges"]))
    total_proposed_edges = int(round(diagnostic_totals["proposed_edges"]))
    total_mobility_weight = float(diagnostic_totals["mobility_weight_sum"])
    total_limited_mobility_weight = float(
        diagnostic_totals["limited_mobility_weight_sum"]
    )
    total_noise_energy = float(diagnostic_totals["noise_energy_sum"])
    total_limited_noise_energy = float(
        diagnostic_totals["limited_noise_energy_sum"]
    )
    total_floor_correction_l1 = float(diagnostic_totals["floor_correction_l1"])
    total_renorm_correction_l1 = float(diagnostic_totals["renorm_correction_l1"])
    total_floor_touched_pixels = int(
        round(diagnostic_totals["floor_touched_pixels"])
    )
    total_floor_proposed_pixels = int(
        round(diagnostic_totals["floor_proposed_pixels"])
    )
    total_nonfinite_edges = int(round(diagnostic_totals["nonfinite_edges"]))

    if not bool(start_filled.all()) or not bool(end_filled.all()):
        raise RuntimeError("multiscale prefix snapshots did not fill every requested boundary")
    reverse_transfers = prefix_reverse_end.unsqueeze(0) - prefix_reverse_start
    reference_transfers = prefix_reference_end.unsqueeze(0) - prefix_reference_start
    innovations = (
        prefix_innovation_end.unsqueeze(0) - prefix_innovation_start
    ) / torch.sqrt(stride_t.to(torch.float64)).view(-1, 1, 1, 1, 1, 1)
    invalid_counts = prefix_invalid_end.unsqueeze(0) - prefix_invalid_start
    if bool((invalid_counts < 0).any()):
        raise RuntimeError("multiscale prefix invalid-edge counts are not monotone")
    masks = invalid_counts == 0
    slow_metrics: dict[str, Any] = {"slow_sum_verified": 0}
    if slow_reverse is not None:
        assert slow_reference is not None
        assert slow_innovation is not None
        assert slow_invalid is not None
        normalized_slow_innovation = slow_innovation / torch.sqrt(
            stride_t.to(torch.float64)
        ).view(-1, 1, 1, 1, 1, 1)
        prefix_target = derive_projected_block_residual(
            reverse_transfers,
            reference_transfers,
            grid_size=n,
        )
        explicit_target = derive_projected_block_residual(
            slow_reverse,
            slow_reference,
            grid_size=n,
        )
        explicit_replay = later_states.unsqueeze(0).to(torch.float64) + (
            flux_divergence_torch(
                slow_reverse.reshape(-1, 2, n, n)
            ).reshape(r_count, p_count, anchors, n * n)
        )
        explicit_replay_error = explicit_replay - earlier_states.to(torch.float64)
        slow_metrics = {
            "slow_sum_verified": 1,
            "slow_reverse_max_abs_error": float(
                (slow_reverse - reverse_transfers).abs().max().detach().cpu()
            ),
            "slow_reference_max_abs_error": float(
                (slow_reference - reference_transfers).abs().max().detach().cpu()
            ),
            "slow_innovation_max_abs_error": float(
                (normalized_slow_innovation - innovations).abs().max().detach().cpu()
            ),
            "slow_target_max_abs_error": float(
                (explicit_target - prefix_target).abs().max().detach().cpu()
            ),
            "slow_replay_l1_max": float(
                explicit_replay_error.abs().sum(dim=-1).max().detach().cpu()
            ),
            "slow_mask_mismatch_count": int(
                ((slow_invalid == 0) != (invalid_counts == 0))
                .count_nonzero()
                .detach()
                .cpu()
            ),
        }
    later_flat = later_states.reshape(p_count * anchors, n * n)
    mobility_valid = harmonic_mobility_channels(later_flat, dynamics_config) > float(
        d0_config.theta_mask_min
    )
    masks = masks & mobility_valid.reshape(1, p_count, anchors, 2, n, n)
    tau = horizon - ends.to(torch.float64) * dt_sub
    raw_fraction = 0.0 if total_proposed_edges == 0 else total_masked_edges / total_proposed_edges
    mobility_fraction = (
        float("nan")
        if total_mobility_weight <= 0.0
        else total_limited_mobility_weight / total_mobility_weight
    )
    noise_fraction = (
        float("nan")
        if total_noise_energy <= 0.0
        else total_limited_noise_energy / total_noise_energy
    )
    diagnostics = {
        "raw_limited_fraction": float(raw_fraction),
        "mobility_weighted_limited_fraction": float(mobility_fraction),
        "noise_energy_weighted_limited_fraction": float(noise_fraction),
        # Retain additive numerators and denominators.  Cache shards are an
        # implementation detail; production gates must aggregate these exact
        # totals rather than average shard-level percentages.
        "masked_edges": int(total_masked_edges),
        "proposed_edges": int(total_proposed_edges),
        "mobility_weight_sum": float(total_mobility_weight),
        "limited_mobility_weight_sum": float(total_limited_mobility_weight),
        "noise_energy_sum": float(total_noise_energy),
        "limited_noise_energy_sum": float(total_limited_noise_energy),
        "floor_correction_l1": float(total_floor_correction_l1),
        "renorm_correction_l1": float(total_renorm_correction_l1),
        "floor_touched_pixels": int(total_floor_touched_pixels),
        "floor_proposed_pixels": int(total_floor_proposed_pixels),
        "floor_touched_fraction": (
            0.0
            if total_floor_proposed_pixels == 0
            else float(total_floor_touched_pixels / total_floor_proposed_pixels)
        ),
        "nonfinite_edges": int(total_nonfinite_edges),
        "path_substep_count": int(p_count * total_substeps),
        "builder_seed": int(seed),
        "prefix_aggregation": 1,
        "prefix_scan_mode": "outer-cumsum-float64",
        "diagnostic_accumulation": "device-float64",
        **slow_metrics,
    }
    cache = D0MultiscaleCache(
        strides=torch.as_tensor(stride_values, dtype=torch.long),
        path_ids=torch.as_tensor(anchor_plan.path_ids, dtype=torch.long),
        later_states=later_states.detach().cpu().float(),
        tau=tau.detach().cpu().float(),
        labels=torch.as_tensor(labels_np, dtype=torch.long),
        end_substeps=torch.as_tensor(anchor_plan.end_substeps, dtype=torch.long),
        anchor_strata=torch.as_tensor(anchor_plan.stratum_indices, dtype=torch.long),
        tau_fraction_edges=anchor_plan.tau_fraction_edges.copy(),
        start_images=torch.as_tensor(initial_np, dtype=torch.float32),
        earlier_states=earlier_states.detach().cpu().float(),
        reverse_transfers=reverse_transfers.detach().cpu().float(),
        reference_transfers=reference_transfers.detach().cpu().float(),
        innovations=innovations.detach().cpu().float(),
        masks=masks.detach().cpu().bool(),
        terminal_states=states.detach().cpu().numpy().reshape(p_count, n, n).astype(np.float32),
        source_indices=np.asarray(source_idx_np, dtype=np.int64),
        requested_labels=np.asarray(labels_np, dtype=np.int64),
        rate_schedule=np.asarray(rate_schedule, dtype=np.float64),
        horizon=float(horizon),
        dt_sub=float(dt_sub),
        sample_steps=int(sample_steps),
        reference_substeps=int(reference_substeps),
        lambda_mix=float(d0_config.lambda_mix),
        anchor_plan_fingerprint=str(
            global_anchor_plan_fingerprint or anchor_plan.fingerprint
        ),
        target_contract=MULTISCALE_TARGET_CONTRACT,
        diagnostics=diagnostics,
    )
    validate_multiscale_cache(cache)
    return cache


def multiscale_cache_fingerprint(cache: D0MultiscaleCache) -> str:
    validate_multiscale_cache(cache)
    records: list[dict[str, Any]] = []
    for name in _CACHE_TENSOR_FIELDS:
        records.append({"field": name, "sha256": array_fingerprint(getattr(cache, name))})
    for name in _CACHE_ARRAY_FIELDS:
        records.append({"field": name, "sha256": array_fingerprint(np.asarray(getattr(cache, name)))})
    for name in _CACHE_SCALAR_FIELDS:
        records.append(
            {
                "field": name,
                "value_sha256": config_fingerprint(getattr(cache, name)),
            }
        )
    return config_fingerprint(records)


def _cache_manifest(cache: D0MultiscaleCache, metadata: Mapping[str, Any]) -> dict[str, Any]:
    scalars = _json_safe({name: getattr(cache, name) for name in _CACHE_SCALAR_FIELDS})
    return {
        "schema": MULTISCALE_CACHE_SCHEMA,
        "schema_version": MULTISCALE_CACHE_SCHEMA_VERSION,
        "cache_fingerprint": multiscale_cache_fingerprint(cache),
        "cache_scalars": scalars,
        "metadata": _json_safe(dict(metadata)),
    }


def _atomic_save_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class D0MultiscaleShardRecord:
    shard_id: int
    filename: str
    path_ids: tuple[int, ...]
    cache_fingerprint: str
    file_sha256: str
    file_size: int
    anchor_plan_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": int(self.shard_id),
            "filename": str(self.filename),
            "path_ids": [int(value) for value in self.path_ids],
            "cache_fingerprint": str(self.cache_fingerprint),
            "file_sha256": str(self.file_sha256),
            "file_size": int(self.file_size),
            "anchor_plan_fingerprint": str(self.anchor_plan_fingerprint),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "D0MultiscaleShardRecord":
        return cls(
            shard_id=int(value["shard_id"]),
            filename=str(value["filename"]),
            path_ids=tuple(int(item) for item in value["path_ids"]),
            cache_fingerprint=str(value["cache_fingerprint"]),
            file_sha256=str(value["file_sha256"]),
            file_size=int(value["file_size"]),
            anchor_plan_fingerprint=str(value["anchor_plan_fingerprint"]),
        )


def save_multiscale_cache_shard(
    path: str | Path,
    cache: D0MultiscaleCache,
    *,
    shard_id: int,
    metadata: Mapping[str, Any] | None = None,
) -> D0MultiscaleShardRecord:
    """Atomically save one complete path shard and return its integrity record."""

    target = Path(path)
    validate_multiscale_cache(cache)
    manifest = _cache_manifest(cache, dict(metadata or {}))
    payload: dict[str, np.ndarray] = {}
    for name in _CACHE_TENSOR_FIELDS:
        payload[name] = _as_numpy(getattr(cache, name))
    for name in _CACHE_ARRAY_FIELDS:
        payload[name] = _as_numpy(getattr(cache, name))
    payload["__manifest_json__"] = np.asarray(
        json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":"))
    )
    _atomic_save_npz(target, payload)
    return D0MultiscaleShardRecord(
        shard_id=int(shard_id),
        filename=target.name,
        path_ids=tuple(int(value) for value in cache.path_ids.tolist()),
        cache_fingerprint=str(manifest["cache_fingerprint"]),
        file_sha256=file_fingerprint(target),
        file_size=int(target.stat().st_size),
        anchor_plan_fingerprint=str(cache.anchor_plan_fingerprint),
    )


def _load_cache_manifest(archive: Any) -> dict[str, Any]:
    if "__manifest_json__" not in archive.files:
        raise D0MultiscaleCompatibilityError("multiscale cache shard has no manifest")
    manifest = json.loads(str(np.asarray(archive["__manifest_json__"]).item()))
    if (
        manifest.get("schema") != MULTISCALE_CACHE_SCHEMA
        or int(manifest.get("schema_version", -1)) != MULTISCALE_CACHE_SCHEMA_VERSION
    ):
        raise D0MultiscaleCompatibilityError("multiscale cache shard schema is incompatible")
    return manifest


def load_multiscale_cache_shard(
    path: str | Path,
    *,
    expected_record: D0MultiscaleShardRecord | None = None,
    verify_hashes: bool = True,
) -> D0MultiscaleCache:
    """Load and content-verify one exact multiscale path shard."""

    source = Path(path)
    if expected_record is not None:
        if source.name != expected_record.filename:
            raise D0MultiscaleCompatibilityError("multiscale shard filename differs from index")
        if verify_hashes and file_fingerprint(source) != expected_record.file_sha256:
            raise D0MultiscaleCompatibilityError("multiscale shard file hash mismatch")
        if int(source.stat().st_size) != int(expected_record.file_size):
            raise D0MultiscaleCompatibilityError("multiscale shard file size mismatch")
    with np.load(source, allow_pickle=False) as archive:
        manifest = _load_cache_manifest(archive)
        required = set(_CACHE_TENSOR_FIELDS + _CACHE_ARRAY_FIELDS)
        missing = sorted(required.difference(archive.files))
        if missing:
            raise D0MultiscaleCompatibilityError(
                "multiscale shard is missing arrays: " + ", ".join(missing)
            )
        scalars = dict(_json_restore(manifest.get("cache_scalars", {})))
        missing_scalars = sorted(set(_CACHE_SCALAR_FIELDS).difference(scalars))
        if missing_scalars:
            raise D0MultiscaleCompatibilityError(
                "multiscale shard is missing scalar metadata: " + ", ".join(missing_scalars)
            )
        tensors = {
            name: torch.from_numpy(np.array(archive[name], copy=True))
            for name in _CACHE_TENSOR_FIELDS
        }
        arrays = {name: np.array(archive[name], copy=True) for name in _CACHE_ARRAY_FIELDS}
    cache = D0MultiscaleCache(**tensors, **arrays, **scalars)
    validate_multiscale_cache(cache)
    actual = multiscale_cache_fingerprint(cache)
    if verify_hashes and actual != str(manifest.get("cache_fingerprint", "")):
        raise D0MultiscaleCompatibilityError("multiscale shard content hash mismatch")
    if expected_record is not None:
        if actual != expected_record.cache_fingerprint:
            raise D0MultiscaleCompatibilityError("multiscale shard cache hash differs from index")
        if tuple(int(value) for value in cache.path_ids.tolist()) != expected_record.path_ids:
            raise D0MultiscaleCompatibilityError("multiscale shard path IDs differ from index")
        if str(cache.anchor_plan_fingerprint) != expected_record.anchor_plan_fingerprint:
            raise D0MultiscaleCompatibilityError("multiscale shard anchor plan differs from index")
    return cache


def slice_multiscale_cache_paths(
    cache: D0MultiscaleCache,
    path_ids: Sequence[int] | np.ndarray | Tensor,
) -> D0MultiscaleCache:
    """Slice complete paths for deterministic atomic sharding."""

    validate_multiscale_cache(cache)
    axes = _path_axes(cache, path_ids)
    idx = torch.as_tensor(axes, dtype=torch.long)
    return replace(
        cache,
        path_ids=cache.path_ids.index_select(0, idx).clone(),
        later_states=cache.later_states.index_select(0, idx).clone(),
        tau=cache.tau.index_select(0, idx).clone(),
        labels=cache.labels.index_select(0, idx).clone(),
        end_substeps=cache.end_substeps.index_select(0, idx).clone(),
        anchor_strata=cache.anchor_strata.index_select(0, idx).clone(),
        start_images=cache.start_images.index_select(0, idx).clone(),
        earlier_states=cache.earlier_states.index_select(1, idx).clone(),
        reverse_transfers=cache.reverse_transfers.index_select(1, idx).clone(),
        reference_transfers=cache.reference_transfers.index_select(1, idx).clone(),
        innovations=cache.innovations.index_select(1, idx).clone(),
        masks=cache.masks.index_select(1, idx).clone(),
        terminal_states=np.ascontiguousarray(np.asarray(cache.terminal_states)[axes]),
        source_indices=np.ascontiguousarray(np.asarray(cache.source_indices)[axes]),
        requested_labels=np.ascontiguousarray(np.asarray(cache.requested_labels)[axes]),
    )


@dataclass(frozen=True)
class D0MultiscaleCacheIndex:
    expected_path_ids: tuple[int, ...]
    records: tuple[D0MultiscaleShardRecord, ...]
    scientific_fingerprint: str
    anchor_plan_fingerprint: str
    metadata: Mapping[str, Any]
    fingerprint: str

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema": MULTISCALE_INDEX_SCHEMA,
            "schema_version": MULTISCALE_INDEX_SCHEMA_VERSION,
            "expected_path_ids": [int(value) for value in self.expected_path_ids],
            "records": [record.to_dict() for record in self.records],
            "scientific_fingerprint": str(self.scientific_fingerprint),
            "anchor_plan_fingerprint": str(self.anchor_plan_fingerprint),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "fingerprint": str(self.fingerprint)}


def _validate_shard_records(
    records: Sequence[D0MultiscaleShardRecord],
    expected_path_ids: Sequence[int],
    anchor_plan_fingerprint: str,
) -> None:
    if not records:
        raise D0MultiscaleCompatibilityError("multiscale cache index contains no shards")
    shard_ids = [int(record.shard_id) for record in records]
    filenames = [str(record.filename) for record in records]
    if len(set(shard_ids)) != len(shard_ids):
        raise D0MultiscaleCompatibilityError("multiscale cache index repeats a shard ID")
    if len(set(filenames)) != len(filenames):
        raise D0MultiscaleCompatibilityError("multiscale cache index repeats a filename")
    for filename in filenames:
        candidate = Path(filename)
        if candidate.name != filename or candidate.is_absolute() or filename in {"", ".", ".."}:
            raise D0MultiscaleCompatibilityError("multiscale shard filename must be a local basename")
    combined = [int(path_id) for record in records for path_id in record.path_ids]
    if len(set(combined)) != len(combined):
        raise D0MultiscaleCompatibilityError("multiscale shards overlap in path IDs")
    if sorted(combined) != sorted(int(value) for value in expected_path_ids):
        raise D0MultiscaleCompatibilityError("multiscale shards do not cover expected path IDs")
    if any(record.anchor_plan_fingerprint != str(anchor_plan_fingerprint) for record in records):
        raise D0MultiscaleCompatibilityError("multiscale shards do not share one anchor plan")
    for record in records:
        if len(record.cache_fingerprint) != 64 or len(record.file_sha256) != 64:
            raise D0MultiscaleCompatibilityError("multiscale shard record has an invalid hash")
        if int(record.file_size) <= 0:
            raise D0MultiscaleCompatibilityError("multiscale shard record has an invalid file size")


def make_multiscale_cache_index(
    records: Sequence[D0MultiscaleShardRecord],
    *,
    expected_path_ids: Sequence[int] | np.ndarray | Tensor,
    scientific_fingerprint: str,
    anchor_plan_fingerprint: str,
    metadata: Mapping[str, Any] | None = None,
) -> D0MultiscaleCacheIndex:
    expected = tuple(int(value) for value in _path_ids_array(expected_path_ids).tolist())
    ordered = tuple(sorted(records, key=lambda record: int(record.shard_id)))
    _validate_shard_records(ordered, expected, str(anchor_plan_fingerprint))
    provisional = D0MultiscaleCacheIndex(
        expected_path_ids=expected,
        records=ordered,
        scientific_fingerprint=str(scientific_fingerprint),
        anchor_plan_fingerprint=str(anchor_plan_fingerprint),
        metadata=dict(metadata or {}),
        fingerprint="",
    )
    return replace(provisional, fingerprint=config_fingerprint(provisional.semantic_dict()))


def save_multiscale_cache_index(path: str | Path, index: D0MultiscaleCacheIndex) -> None:
    _validate_shard_records(index.records, index.expected_path_ids, index.anchor_plan_fingerprint)
    if str(index.fingerprint) != config_fingerprint(index.semantic_dict()):
        raise D0MultiscaleCompatibilityError("multiscale cache index fingerprint mismatch")
    atomic_write_json(path, index.to_dict())


def load_multiscale_cache_index(
    path: str | Path,
    *,
    verify_shards: bool = True,
) -> D0MultiscaleCacheIndex:
    index_path = Path(path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != MULTISCALE_INDEX_SCHEMA
        or int(payload.get("schema_version", -1)) != MULTISCALE_INDEX_SCHEMA_VERSION
    ):
        raise D0MultiscaleCompatibilityError("multiscale cache index schema is incompatible")
    index = D0MultiscaleCacheIndex(
        expected_path_ids=tuple(int(value) for value in payload["expected_path_ids"]),
        records=tuple(D0MultiscaleShardRecord.from_dict(value) for value in payload["records"]),
        scientific_fingerprint=str(payload["scientific_fingerprint"]),
        anchor_plan_fingerprint=str(payload["anchor_plan_fingerprint"]),
        metadata=dict(payload.get("metadata", {})),
        fingerprint=str(payload.get("fingerprint", "")),
    )
    _validate_shard_records(index.records, index.expected_path_ids, index.anchor_plan_fingerprint)
    if index.fingerprint != config_fingerprint(index.semantic_dict()):
        raise D0MultiscaleCompatibilityError("multiscale cache index fingerprint mismatch")
    if verify_shards:
        for record in index.records:
            load_multiscale_cache_shard(
                index_path.parent / record.filename,
                expected_record=record,
                verify_hashes=True,
            )
    return index


def load_multiscale_cache_shards(
    index_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> tuple[D0MultiscaleCacheIndex, list[D0MultiscaleCache]]:
    index = load_multiscale_cache_index(index_path, verify_shards=False)
    root = Path(index_path).parent
    shards = [
        load_multiscale_cache_shard(
            root / record.filename,
            expected_record=record,
            verify_hashes=verify_hashes,
        )
        for record in index.records
    ]
    return index, shards


@torch.no_grad()
def block_arithmetic_metrics(
    cache: D0MultiscaleCache,
    dynamics_config: DirectFluxMNISTConfig,
) -> dict[str, Any]:
    """Measure exact state replay, projection, tau, and elementary-baseline identities."""

    validate_multiscale_cache(cache)
    if int(dynamics_config.grid_size) != cache.grid_size:
        raise ValueError("dynamics grid does not match multiscale cache")
    later = cache.later_states
    rows: list[dict[str, Any]] = []
    all_finite = True
    for stride_axis, stride in enumerate(cache.strides.tolist()):
        reverse = cache.reverse_transfers[stride_axis]
        reference = cache.reference_transfers[stride_axis]
        replay = later + flux_divergence_torch(
            reverse.reshape(-1, 2, cache.grid_size, cache.grid_size)
        ).reshape_as(later)
        error = replay - cache.earlier_states[stride_axis]
        residual = reverse - reference
        projected = derive_projected_block_residual(
            reverse, reference, grid_size=cache.grid_size
        )
        projected_twice = derive_projected_block_residual(
            projected, torch.zeros_like(projected), grid_size=cache.grid_size
        )
        divergence_error = flux_divergence_torch(
            projected.reshape(-1, 2, cache.grid_size, cache.grid_size)
        ) - flux_divergence_torch(
            residual.reshape(-1, 2, cache.grid_size, cache.grid_size)
        )
        finite = bool(
            torch.isfinite(error).all()
            and torch.isfinite(residual).all()
            and torch.isfinite(projected).all()
        )
        all_finite = all_finite and finite
        rows.append(
            {
                "stride": int(stride),
                "replay_l1_mean": float(error.abs().sum(dim=-1).mean()),
                "replay_l1_max": float(error.abs().sum(dim=-1).max()),
                "replay_max_abs": float(error.abs().max()),
                "projection_idempotence_max_abs": float((projected_twice - projected).abs().max()),
                "projection_divergence_max_abs": float(divergence_error.abs().max()),
                "target_finite": int(finite),
                "target_rms_unscaled": float(torch.sqrt(projected.double().square().mean())),
            }
        )
    r1_reference_error = float("nan")
    r1_direct_target_error = float("nan")
    if bool((cache.strides == 1).any()):
        axis = cache.stride_axis(1)
        states_flat = cache.later_states.reshape(-1, cache.grid_size * cache.grid_size)
        q = (cache.end_substeps - 1).reshape(-1)
        expected = exact_reverse_reference_step_transfer(
            states_flat,
            q,
            rate_schedule=cache.rate_schedule,
            reference_substeps=cache.reference_substeps,
            dt_sub=cache.dt_sub,
            dynamics_config=dynamics_config,
        ).reshape_as(cache.reference_transfers[axis])
        r1_reference_error = float((expected - cache.reference_transfers[axis]).abs().max())
        row_count = int(states_flat.shape[0])
        direct_baseline = _direct_reverse_free_block_baseline_from_batch(
            {
                "states": states_flat,
                "starts": q,
                "stride_substeps": torch.ones(row_count, dtype=torch.long),
                "reference_substeps": torch.full(
                    (row_count,), int(cache.reference_substeps), dtype=torch.long
                ),
                "dt_sub": torch.full(
                    (row_count,), float(cache.dt_sub), dtype=states_flat.dtype
                ),
                "rate_schedule": torch.as_tensor(
                    cache.rate_schedule, dtype=states_flat.dtype
                ),
            },
            dynamics_config,
        ).reshape_as(cache.reference_transfers[axis])
        existing_direct_target = project_edge_flux_torch(
            (cache.reverse_transfers[axis] - direct_baseline).reshape(
                -1, 2, cache.grid_size, cache.grid_size
            ),
            grid_size=cache.grid_size,
        ).reshape_as(cache.reverse_transfers[axis])
        multiscale_target = derive_projected_block_residual(
            cache.reverse_transfers[axis],
            cache.reference_transfers[axis],
            grid_size=cache.grid_size,
        )
        r1_direct_target_error = float(
            (existing_direct_target - multiscale_target).abs().max()
        )
    simplex_values = [
        cache.later_states.sum(dim=-1),
        cache.earlier_states.sum(dim=-1),
        cache.start_images.sum(dim=-1),
        torch.as_tensor(np.asarray(cache.terminal_states).reshape(cache.path_count, -1)).sum(dim=-1),
    ]
    simplex_error = max(float((value - 1.0).abs().max()) for value in simplex_values)
    expected_tau = float(cache.horizon) - cache.end_substeps.double() * float(cache.dt_sub)
    return {
        "target_contract": str(cache.target_contract),
        "path_count": int(cache.path_count),
        "anchors_per_path": int(cache.anchors_per_path),
        "stride_count": int(cache.stride_count),
        "all_finite": int(all_finite),
        "max_simplex_mass_error": float(simplex_error),
        "tau_max_abs_error": float((cache.tau.double() - expected_tau).abs().max()),
        "r1_reference_max_abs_error": float(r1_reference_error),
        "r1_existing_direct_target_max_abs_error": float(r1_direct_target_error),
        "by_stride": rows,
    }


def evaluate_multiscale_cache_preflight(
    cache: D0MultiscaleCache,
    dynamics_config: DirectFluxMNISTConfig,
    *,
    train_path_ids: Sequence[int] | np.ndarray | Tensor,
    scale_floor: float = 1e-6,
    max_replay_l1: float = 1e-6,
    max_tau_error: float = 1e-7,
    max_r1_reference_error: float = 1e-8,
    max_simplex_mass_error: float = 2e-6,
    max_raw_intervention: float = 0.005,
    max_weighted_intervention: float = 0.0005,
    max_floor_correction_l1_per_path_substep: float = 1e-8,
    max_renorm_correction_l1_per_path_substep: float = 1e-6,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Return named, fail-closed preflight checks without making a model claim."""

    metrics = block_arithmetic_metrics(cache, dynamics_config)
    scales = infer_training_block_scales(
        cache,
        dynamics_config,
        train_path_ids,
        floor=float(scale_floor),
        device=device,
    )
    max_replay = max(float(row["replay_l1_max"]) for row in metrics["by_stride"])
    r1_error = float(metrics["r1_reference_max_abs_error"])
    r1_target_error = float(metrics["r1_existing_direct_target_max_abs_error"])
    diagnostics = dict(cache.diagnostics)

    def diagnostic_float(name: str) -> float:
        try:
            return float(diagnostics[name])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    raw_intervention = diagnostic_float("raw_limited_fraction")
    mobility_intervention = diagnostic_float("mobility_weighted_limited_fraction")
    noise_intervention = diagnostic_float("noise_energy_weighted_limited_fraction")
    floor_correction = diagnostic_float("floor_correction_l1")
    renorm_correction = diagnostic_float("renorm_correction_l1")
    path_substeps = diagnostic_float("path_substep_count")
    floor_per = (
        floor_correction / path_substeps
        if math.isfinite(floor_correction) and math.isfinite(path_substeps) and path_substeps > 0.0
        else float("nan")
    )
    renorm_per = (
        renorm_correction / path_substeps
        if math.isfinite(renorm_correction) and math.isfinite(path_substeps) and path_substeps > 0.0
        else float("nan")
    )
    nonfinite_edges = diagnostic_float("nonfinite_edges")
    floor_touches = diagnostic_float("floor_touched_pixels")
    checks = {
        "finite_targets": {
            "value": int(metrics["all_finite"]),
            "operator": "==",
            "threshold": 1,
            "passed": int(metrics["all_finite"]) == 1,
        },
        "block_state_replay": {
            "value": max_replay,
            "operator": "<=",
            "threshold": float(max_replay_l1),
            "passed": math.isfinite(max_replay) and 0.0 <= max_replay <= float(max_replay_l1),
        },
        "tau_identity": {
            "value": float(metrics["tau_max_abs_error"]),
            "operator": "<=",
            "threshold": float(max_tau_error),
            "passed": math.isfinite(float(metrics["tau_max_abs_error"]))
            and 0.0 <= float(metrics["tau_max_abs_error"]) <= float(max_tau_error),
        },
        "r1_reference_identity": {
            "value": r1_error,
            "operator": "<=",
            "threshold": float(max_r1_reference_error),
            "passed": math.isfinite(r1_error)
            and 0.0 <= r1_error <= float(max_r1_reference_error),
        },
        "r1_existing_direct_target_identity": {
            "value": r1_target_error,
            "operator": "<=",
            "threshold": float(max_r1_reference_error),
            "passed": math.isfinite(r1_target_error)
            and 0.0 <= r1_target_error <= float(max_r1_reference_error),
        },
        "simplex_health": {
            "value": float(metrics["max_simplex_mass_error"]),
            "operator": "<=",
            "threshold": float(max_simplex_mass_error),
            "passed": math.isfinite(float(metrics["max_simplex_mass_error"]))
            and 0.0 <= float(metrics["max_simplex_mass_error"]) <= float(max_simplex_mass_error),
        },
        "finite_positive_training_scales": {
            "value": {str(key): float(value) for key, value in scales.items()},
            "operator": "all finite > 0",
            "threshold": 0.0,
            "passed": bool(scales)
            and all(math.isfinite(float(value)) and float(value) > 0.0 for value in scales.values()),
        },
        "raw_intervention": {
            "value": raw_intervention,
            "operator": "<=",
            "threshold": float(max_raw_intervention),
            "passed": math.isfinite(raw_intervention)
            and 0.0 <= raw_intervention <= float(max_raw_intervention),
        },
        "mobility_weighted_intervention": {
            "value": mobility_intervention,
            "operator": "<=",
            "threshold": float(max_weighted_intervention),
            "passed": math.isfinite(mobility_intervention)
            and 0.0 <= mobility_intervention <= float(max_weighted_intervention),
        },
        "noise_weighted_intervention": {
            "value": noise_intervention,
            "operator": "<=",
            "threshold": float(max_weighted_intervention),
            "passed": math.isfinite(noise_intervention)
            and 0.0 <= noise_intervention <= float(max_weighted_intervention),
        },
        "nonfinite_edges": {
            "value": nonfinite_edges,
            "operator": "==",
            "threshold": 0,
            "passed": math.isfinite(nonfinite_edges) and nonfinite_edges == 0.0,
        },
        "floor_touches": {
            "value": floor_touches,
            "operator": "==",
            "threshold": 0,
            "passed": math.isfinite(floor_touches) and floor_touches == 0.0,
        },
        "floor_correction_per_path_substep": {
            "value": floor_per,
            "operator": "<=",
            "threshold": float(max_floor_correction_l1_per_path_substep),
            "passed": math.isfinite(floor_per)
            and 0.0 <= floor_per <= float(max_floor_correction_l1_per_path_substep),
        },
        "renorm_correction_per_path_substep": {
            "value": renorm_per,
            "operator": "<=",
            "threshold": float(max_renorm_correction_l1_per_path_substep),
            "passed": math.isfinite(renorm_per)
            and 0.0 <= renorm_per <= float(max_renorm_correction_l1_per_path_substep),
        },
    }
    return {
        "schema": "experiment12-d0-multiscale-cache-preflight",
        "schema_version": 1,
        "claim_scope": "cache aggregation and arithmetic only; no learned-model evidence",
        "passed": int(all(bool(check["passed"]) for check in checks.values())),
        "checks": checks,
        "metrics": metrics,
        "training_target_scales": {str(key): float(value) for key, value in scales.items()},
    }


__all__ = [
    "DEFAULT_TAU_FRACTION_EDGES",
    "D0MultiscaleCache",
    "D0MultiscaleCacheIndex",
    "D0MultiscaleCompatibilityError",
    "D0MultiscaleShardRecord",
    "D0StratifiedAnchorPlan",
    "D0ThreeWayPathSplit",
    "MULTISCALE_CACHE_SCHEMA",
    "MULTISCALE_CACHE_SCHEMA_VERSION",
    "MULTISCALE_INDEX_SCHEMA",
    "MULTISCALE_INDEX_SCHEMA_VERSION",
    "MULTISCALE_TARGET_CONTRACT",
    "aggregate_aligned_block_quantities",
    "block_arithmetic_metrics",
    "block_residual_targets",
    "build_multiscale_cache_shard",
    "derive_projected_block_residual",
    "deterministic_three_way_path_split",
    "evaluate_multiscale_cache_preflight",
    "exact_reverse_reference_step_transfer",
    "infer_training_block_scales",
    "load_multiscale_cache_index",
    "load_multiscale_cache_shard",
    "load_multiscale_cache_shards",
    "make_multiscale_cache_index",
    "make_stratified_anchor_plan",
    "multiscale_cache_fingerprint",
    "save_multiscale_cache_index",
    "save_multiscale_cache_shard",
    "slice_anchor_plan",
    "slice_multiscale_cache_paths",
    "validate_anchor_plan",
    "validate_multiscale_cache",
    "validate_three_way_path_split",
]
