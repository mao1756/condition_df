from __future__ import annotations

"""Versioned state-only caches for the Experiment 12 implicit-score probe.

The score experiment needs samples from positive-time forward marginals, not
the noisy pathwise block labels stored by :mod:`mnist.d0_multiscale_cache`.
This module therefore owns a deliberately separate artifact contract.  It can
materialize the later states belonging to the parent train/selection roles and
can generate wholly fresh audit paths with the same masked reference
integrator.  Parent audit paths are never legal inputs to this cache.

The module contains no model or reverse sampler code.
"""

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_multiscale_cache import (
    DEFAULT_TAU_FRACTION_EDGES,
    D0MultiscaleCache,
    D0MultiscaleCompatibilityError,
    D0StratifiedAnchorPlan,
    D0ThreeWayPathSplit,
    load_multiscale_cache_index,
    load_multiscale_cache_shard,
    make_stratified_anchor_plan,
    slice_anchor_plan,
    validate_three_way_path_split,
)
from mnist.d0_one_image_gate import (
    array_fingerprint,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    edge_alpha_value,
    masked_reference_free_step_torch,
    natural_horizon,
)
from mnist.experiment12_d0 import (
    Experiment12D0Config,
    _lambda_mixed_data_for_paths,
    make_rate_schedule,
)


SCORE_STATE_CACHE_SCHEMA = "experiment12-d0-score-state-cache-shard"
SCORE_STATE_CACHE_SCHEMA_VERSION = 1
SCORE_STATE_INDEX_SCHEMA = "experiment12-d0-score-state-cache-index"
SCORE_STATE_INDEX_SCHEMA_VERSION = 1
SCORE_STATE_CONTRACT = "positive-time-reference-later-states-v1"
SCORE_STATE_REFERENCE_INTEGRATOR = "masked_reference_free_step_torch"
SCORE_STATE_REFERENCE_INTEGRATOR_VERSION = 1

DEFAULT_SCORE_ANCHORS_PER_PATH = 32
DEFAULT_SCORE_ANCHOR_BIN_COUNTS: tuple[int, ...] = (4, 4, 4, 4, 16)
DEFAULT_SCORE_MINIMUM_FORWARD_SUBSTEP = 1024
DEFAULT_SCORE_SHARD_PATHS = 8

PARENT_ORIGIN = "parent-multiscale"
FRESH_ORIGIN = "fresh-reference"
PARENT_SCORE_ROLES = frozenset({"train", "selection"})
FRESH_SCORE_ROLES = frozenset({"audit", "preflight"})
ALL_SCORE_ROLES = PARENT_SCORE_ROLES | FRESH_SCORE_ROLES

FROZEN_SCORE_KERNEL: Mapping[str, Any] = {
    "grid_size": 28,
    "sample_steps": 512,
    "reference_substeps": 256,
    "tau_eff": 5e-5,
    "edge_alpha_mode": "alpha_eff",
    "edge_alpha_value": 1.0,
    "mass_floor": 1e-7,
    "limiter_fraction": 1.0,
    "lambda_mix": 0.35,
    "integrator": SCORE_STATE_REFERENCE_INTEGRATOR,
}


class D0ScoreStateCompatibilityError(ValueError):
    """Raised when an artifact cannot satisfy the score-state contract."""


def _as_numpy(value: np.ndarray | Tensor, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _path_ids(value: Sequence[int] | np.ndarray | Tensor) -> np.ndarray:
    result = _as_numpy(value, dtype=np.dtype(np.int64)).reshape(-1)
    if result.size == 0:
        raise ValueError("path IDs must not be empty")
    if np.unique(result).size != result.size:
        raise ValueError("path IDs must be unique")
    return result


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


def derive_score_state_shard_seed(base_seed: int, shard_id: int, *, scope: str) -> int:
    """Derive a stable local RNG seed without depending on build order."""

    payload = f"d0-score-state-v1:{str(scope)}:{int(base_seed)}:{int(shard_id)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "little") % (
        2**31 - 1
    )


def validate_frozen_score_kernel(
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
) -> None:
    """Fail closed unless a fresh rollout uses the manuscript-gate kernel."""

    actual: dict[str, Any] = {
        "grid_size": int(dynamics_config.grid_size),
        "sample_steps": int(d0_config.sample_steps),
        "reference_substeps": int(d0_config.reference_substeps),
        "tau_eff": float(d0_config.tau_eff),
        "edge_alpha_mode": str(dynamics_config.edge_alpha_mode),
        "edge_alpha_value": float(edge_alpha_value(dynamics_config)),
        "mass_floor": float(dynamics_config.mass_floor),
        "limiter_fraction": float(dynamics_config.limiter_fraction),
        "lambda_mix": float(d0_config.lambda_mix),
        "integrator": SCORE_STATE_REFERENCE_INTEGRATOR,
    }
    mismatches: list[str] = []
    for key, expected in FROZEN_SCORE_KERNEL.items():
        value = actual[key]
        if isinstance(expected, float):
            if not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-15):
                mismatches.append(f"{key}={value!r}, expected {expected!r}")
        elif value != expected:
            mismatches.append(f"{key}={value!r}, expected {expected!r}")
    mode = str(d0_config.cache_build_mode).strip().lower().replace("_", "-")
    if mode not in {"substep", "exact-substep", "exact"}:
        mismatches.append("cache_build_mode must be exact substep")
    if not bool(d0_config.single_image_overfit) or int(d0_config.single_image_label or -1) != 3:
        mismatches.append("fresh score paths must use the frozen one-image label-3 source")
    if mismatches:
        raise D0ScoreStateCompatibilityError("frozen score kernel mismatch: " + "; ".join(mismatches))


def _kernel_metadata(
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
) -> dict[str, Any]:
    return {
        "grid_size": int(dynamics_config.grid_size),
        "alpha_eff": float(dynamics_config.alpha_eff),
        "edge_alpha_mode": str(dynamics_config.edge_alpha_mode),
        "edge_alpha_value": float(edge_alpha_value(dynamics_config)),
        "mass_floor": float(dynamics_config.mass_floor),
        "limiter_fraction": float(dynamics_config.limiter_fraction),
        "sample_steps": int(d0_config.sample_steps),
        "reference_substeps": int(d0_config.reference_substeps),
        "tau_eff": float(d0_config.tau_eff),
        "lambda_mix": float(d0_config.lambda_mix),
        "integrator": SCORE_STATE_REFERENCE_INTEGRATOR,
        "integrator_version": SCORE_STATE_REFERENCE_INTEGRATOR_VERSION,
    }


def _schedule_metadata(
    *,
    rate_schedule: np.ndarray,
    horizon: float,
    dt_sub: float,
    d0_config: Experiment12D0Config,
) -> dict[str, Any]:
    return {
        "sample_steps": int(d0_config.sample_steps),
        "reference_substeps": int(d0_config.reference_substeps),
        "total_substeps": int(d0_config.sample_steps) * int(d0_config.reference_substeps),
        "horizon": float(horizon),
        "dt_sub": float(dt_sub),
        "tau_eff": float(d0_config.tau_eff),
        "time_change_mode": str(d0_config.time_change_mode),
        "rate_ramp": str(d0_config.rate_ramp),
        "rate_ramp_ratio": float(d0_config.rate_ramp_ratio),
        "reference_rate_min": d0_config.reference_rate_min,
        "reference_rate_max": d0_config.reference_rate_max,
        "rate_schedule_sha256": array_fingerprint(np.asarray(rate_schedule, dtype=np.float64)),
    }


@dataclass(frozen=True)
class D0ScoreStateCache:
    """Path-major positive-time state samples and immutable provenance."""

    path_ids: Tensor
    states: Tensor
    tau: Tensor
    labels: Tensor
    end_substeps: Tensor
    anchor_strata: Tensor
    tau_fraction_edges: np.ndarray
    roles: np.ndarray
    origins: np.ndarray
    origin_path_ids: np.ndarray
    terminal_states: np.ndarray
    source_indices: np.ndarray
    requested_labels: np.ndarray
    rate_schedule: np.ndarray
    horizon: float
    dt_sub: float
    sample_steps: int
    reference_substeps: int
    lambda_mix: float
    minimum_forward_substep: int
    anchor_plan_fingerprint: str
    scientific_fingerprint: str
    kernel_metadata: Mapping[str, Any]
    schedule_metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    state_contract: str = SCORE_STATE_CONTRACT

    @property
    def path_count(self) -> int:
        return int(self.path_ids.numel())

    @property
    def anchors_per_path(self) -> int:
        return int(self.states.shape[1])

    @property
    def grid_size(self) -> int:
        return int(round(math.sqrt(float(self.states.shape[-1]))))

    @property
    def role(self) -> str:
        return str(self.roles.reshape(-1)[0])

    @property
    def origin(self) -> str:
        return str(self.origins.reshape(-1)[0])


_CACHE_TENSOR_FIELDS = (
    "path_ids",
    "states",
    "tau",
    "labels",
    "end_substeps",
    "anchor_strata",
)
_CACHE_ARRAY_FIELDS = (
    "tau_fraction_edges",
    "roles",
    "origins",
    "origin_path_ids",
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
    "minimum_forward_substep",
    "anchor_plan_fingerprint",
    "scientific_fingerprint",
    "kernel_metadata",
    "schedule_metadata",
    "provenance",
    "diagnostics",
    "state_contract",
)


def _stratum_bounds(
    *, total_substeps: int, minimum: int, tau_lo: float, tau_hi: float, last: bool
) -> tuple[int, int]:
    raw_min = int(math.floor(float(total_substeps) * (1.0 - float(tau_hi)))) + 1
    if last:
        raw_min = int(math.ceil(float(total_substeps) * (1.0 - float(tau_hi))))
    raw_max = int(math.floor(float(total_substeps) * (1.0 - float(tau_lo))))
    return max(int(minimum), raw_min, 1), min(int(total_substeps), raw_max)


def validate_score_state_cache(cache: D0ScoreStateCache) -> None:
    """Validate shapes, temporal semantics, simplex health, and role isolation."""

    paths = _as_numpy(cache.path_ids, dtype=np.dtype(np.int64)).reshape(-1)
    if paths.size == 0 or np.unique(paths).size != paths.size:
        raise D0ScoreStateCompatibilityError("score cache path IDs must be non-empty and unique")
    p_count = int(paths.size)
    if cache.states.ndim != 3 or int(cache.states.shape[0]) != p_count:
        raise D0ScoreStateCompatibilityError("states must have shape (paths, anchors, pixels)")
    anchors = int(cache.states.shape[1])
    pixels = int(cache.states.shape[2])
    n = int(round(math.sqrt(float(pixels))))
    if anchors <= 0 or n * n != pixels:
        raise D0ScoreStateCompatibilityError("score cache has no anchors or a non-square grid")
    if cache.tau.shape != (p_count, anchors):
        raise D0ScoreStateCompatibilityError("tau does not match the path/anchor axes")
    if cache.end_substeps.shape != (p_count, anchors) or cache.anchor_strata.shape != (
        p_count,
        anchors,
    ):
        raise D0ScoreStateCompatibilityError("anchor arrays do not match states")
    if cache.labels.shape != (p_count,):
        raise D0ScoreStateCompatibilityError("labels must be path-level")
    for name in _CACHE_TENSOR_FIELDS:
        if getattr(cache, name).device.type != "cpu":
            raise D0ScoreStateCompatibilityError(f"cache field {name} must reside on CPU")
    for name in ("states", "tau"):
        if not bool(torch.isfinite(getattr(cache, name)).all()):
            raise D0ScoreStateCompatibilityError(f"cache field {name} contains non-finite values")
    if not bool((cache.states > 0.0).all()):
        raise D0ScoreStateCompatibilityError("positive-time score states must be strictly positive")
    simplex_error = float((cache.states.double().sum(dim=-1) - 1.0).abs().max())
    if simplex_error > 2e-6:
        raise D0ScoreStateCompatibilityError("score states exceed the simplex mass tolerance")

    total = int(cache.sample_steps) * int(cache.reference_substeps)
    minimum = int(cache.minimum_forward_substep)
    if int(cache.sample_steps) <= 0 or int(cache.reference_substeps) <= 0 or minimum <= 0:
        raise D0ScoreStateCompatibilityError("temporal grid and minimum substep must be positive")
    ends = _as_numpy(cache.end_substeps, dtype=np.dtype(np.int64))
    if np.any(ends < minimum) or np.any(ends > total):
        raise D0ScoreStateCompatibilityError("anchor endpoints violate the positive-time range")
    if not math.isfinite(float(cache.horizon)) or float(cache.horizon) <= 0.0:
        raise D0ScoreStateCompatibilityError("horizon must be finite and positive")
    expected_dt = float(cache.horizon) / float(total)
    if not math.isclose(float(cache.dt_sub), expected_dt, rel_tol=1e-12, abs_tol=1e-15):
        raise D0ScoreStateCompatibilityError("dt_sub is inconsistent with the temporal grid")
    expected_tau = float(cache.horizon) - torch.as_tensor(ends, dtype=torch.float64) * float(
        cache.dt_sub
    )
    if float((cache.tau.double() - expected_tau).abs().max()) > max(
        1e-7 * float(cache.horizon), 1e-12
    ):
        raise D0ScoreStateCompatibilityError("tau is inconsistent with anchor endpoints")
    edges = np.asarray(cache.tau_fraction_edges, dtype=np.float64).reshape(-1)
    if (
        edges.size < 2
        or not np.isfinite(edges).all()
        or not np.all(np.diff(edges) > 0.0)
        or not math.isclose(float(edges[0]), 0.0, abs_tol=1e-15)
        or not math.isclose(float(edges[-1]), 1.0, abs_tol=1e-15)
    ):
        raise D0ScoreStateCompatibilityError("tau-fraction edges are invalid")
    strata = _as_numpy(cache.anchor_strata, dtype=np.dtype(np.int64))
    if np.any(strata < 0) or np.any(strata >= edges.size - 1):
        raise D0ScoreStateCompatibilityError("anchor stratum is outside the tau bins")
    for axis in range(edges.size - 1):
        lo, hi = _stratum_bounds(
            total_substeps=total,
            minimum=minimum,
            tau_lo=float(edges[axis]),
            tau_hi=float(edges[axis + 1]),
            last=axis == edges.size - 2,
        )
        selected = ends[strata == axis]
        if selected.size and (np.any(selected < lo) or np.any(selected > hi)):
            raise D0ScoreStateCompatibilityError("anchor strata are inconsistent with tau")

    for name in ("roles", "origins", "origin_path_ids", "source_indices", "requested_labels"):
        if np.asarray(getattr(cache, name)).reshape(-1).size != p_count:
            raise D0ScoreStateCompatibilityError(f"path-level array {name} has the wrong length")
    roles = np.asarray(cache.roles).astype(str).reshape(-1)
    origins = np.asarray(cache.origins).astype(str).reshape(-1)
    if np.unique(roles).size != 1 or roles[0] not in ALL_SCORE_ROLES:
        raise D0ScoreStateCompatibilityError("a score-state shard must contain one supported role")
    if np.unique(origins).size != 1 or origins[0] not in {PARENT_ORIGIN, FRESH_ORIGIN}:
        raise D0ScoreStateCompatibilityError("a score-state shard must contain one supported origin")
    if origins[0] == PARENT_ORIGIN and roles[0] not in PARENT_SCORE_ROLES:
        raise D0ScoreStateCompatibilityError("parent audit paths are forbidden in score caches")
    if origins[0] == FRESH_ORIGIN and roles[0] not in FRESH_SCORE_ROLES:
        raise D0ScoreStateCompatibilityError("fresh paths may only serve audit/preflight roles")

    terminal = np.asarray(cache.terminal_states)
    if terminal.shape[0] != p_count or int(np.prod(terminal.shape[1:])) != pixels:
        raise D0ScoreStateCompatibilityError("terminal states are incompatible with the path axis")
    if not np.isfinite(terminal).all() or np.any(terminal <= 0.0):
        raise D0ScoreStateCompatibilityError("terminal states must be finite and strictly positive")
    terminal_error = float(np.max(np.abs(terminal.reshape(p_count, -1).sum(axis=1) - 1.0)))
    if terminal_error > 2e-6:
        raise D0ScoreStateCompatibilityError("terminal states exceed the simplex mass tolerance")
    schedule = np.asarray(cache.rate_schedule, dtype=np.float64).reshape(-1)
    if schedule.shape != (int(cache.sample_steps),) or not np.isfinite(schedule).all() or np.any(
        schedule < 0.0
    ):
        raise D0ScoreStateCompatibilityError("rate schedule is invalid")

    required_kernel = {
        "grid_size",
        "mass_floor",
        "limiter_fraction",
        "edge_alpha_mode",
        "edge_alpha_value",
        "integrator",
    }
    if not required_kernel.issubset(cache.kernel_metadata):
        raise D0ScoreStateCompatibilityError("kernel metadata is incomplete")
    if int(cache.kernel_metadata["grid_size"]) != n:
        raise D0ScoreStateCompatibilityError("kernel grid size differs from cached states")
    if str(cache.kernel_metadata["integrator"]) != SCORE_STATE_REFERENCE_INTEGRATOR:
        raise D0ScoreStateCompatibilityError("score cache uses an unsupported reference integrator")
    required_schedule = {
        "sample_steps",
        "reference_substeps",
        "horizon",
        "dt_sub",
        "rate_schedule_sha256",
    }
    if not required_schedule.issubset(cache.schedule_metadata):
        raise D0ScoreStateCompatibilityError("schedule metadata is incomplete")
    if (
        int(cache.schedule_metadata["sample_steps"]) != int(cache.sample_steps)
        or int(cache.schedule_metadata["reference_substeps"]) != int(cache.reference_substeps)
        or not math.isclose(
            float(cache.schedule_metadata["horizon"]), float(cache.horizon), rel_tol=1e-12, abs_tol=1e-15
        )
        or not math.isclose(
            float(cache.schedule_metadata["dt_sub"]), float(cache.dt_sub), rel_tol=1e-12, abs_tol=1e-15
        )
        or str(cache.schedule_metadata["rate_schedule_sha256"]) != array_fingerprint(schedule)
    ):
        raise D0ScoreStateCompatibilityError("schedule metadata differs from cache arrays")
    if not str(cache.anchor_plan_fingerprint) or not str(cache.scientific_fingerprint):
        raise D0ScoreStateCompatibilityError("score cache fingerprints must not be empty")
    if str(cache.state_contract) != SCORE_STATE_CONTRACT:
        raise D0ScoreStateCompatibilityError("score cache state contract is unsupported")


def score_state_cache_fingerprint(cache: D0ScoreStateCache) -> str:
    validate_score_state_cache(cache)
    records: list[dict[str, str]] = []
    for name in _CACHE_TENSOR_FIELDS:
        records.append({"field": name, "sha256": array_fingerprint(getattr(cache, name))})
    for name in _CACHE_ARRAY_FIELDS:
        records.append({"field": name, "sha256": array_fingerprint(np.asarray(getattr(cache, name)))})
    for name in _CACHE_SCALAR_FIELDS:
        records.append({"field": name, "value_sha256": config_fingerprint(getattr(cache, name))})
    return config_fingerprint(records)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _atomic_save_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Write deterministic compressed NPZ bytes and atomically replace ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name in sorted(payload):
                    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(info, _npy_bytes(np.asarray(payload[name])))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _cache_manifest(cache: D0ScoreStateCache, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCORE_STATE_CACHE_SCHEMA,
        "schema_version": SCORE_STATE_CACHE_SCHEMA_VERSION,
        "cache_fingerprint": score_state_cache_fingerprint(cache),
        "cache_scalars": _json_safe({name: getattr(cache, name) for name in _CACHE_SCALAR_FIELDS}),
        "metadata": _json_safe(dict(metadata)),
    }


@dataclass(frozen=True)
class D0ScoreStateShardRecord:
    shard_id: int
    filename: str
    path_ids: tuple[int, ...]
    role: str
    origin: str
    cache_fingerprint: str
    file_sha256: str
    file_size: int
    anchor_plan_fingerprint: str
    scientific_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": int(self.shard_id),
            "filename": str(self.filename),
            "path_ids": [int(value) for value in self.path_ids],
            "role": str(self.role),
            "origin": str(self.origin),
            "cache_fingerprint": str(self.cache_fingerprint),
            "file_sha256": str(self.file_sha256),
            "file_size": int(self.file_size),
            "anchor_plan_fingerprint": str(self.anchor_plan_fingerprint),
            "scientific_fingerprint": str(self.scientific_fingerprint),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "D0ScoreStateShardRecord":
        return cls(
            shard_id=int(value["shard_id"]),
            filename=str(value["filename"]),
            path_ids=tuple(int(item) for item in value["path_ids"]),
            role=str(value["role"]),
            origin=str(value["origin"]),
            cache_fingerprint=str(value["cache_fingerprint"]),
            file_sha256=str(value["file_sha256"]),
            file_size=int(value["file_size"]),
            anchor_plan_fingerprint=str(value["anchor_plan_fingerprint"]),
            scientific_fingerprint=str(value["scientific_fingerprint"]),
        )


def save_score_state_cache_shard(
    path: str | Path,
    cache: D0ScoreStateCache,
    *,
    shard_id: int,
    metadata: Mapping[str, Any] | None = None,
) -> D0ScoreStateShardRecord:
    target = Path(path)
    validate_score_state_cache(cache)
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
    return D0ScoreStateShardRecord(
        shard_id=int(shard_id),
        filename=target.name,
        path_ids=tuple(int(value) for value in cache.path_ids.tolist()),
        role=cache.role,
        origin=cache.origin,
        cache_fingerprint=str(manifest["cache_fingerprint"]),
        file_sha256=file_fingerprint(target),
        file_size=int(target.stat().st_size),
        anchor_plan_fingerprint=str(cache.anchor_plan_fingerprint),
        scientific_fingerprint=str(cache.scientific_fingerprint),
    )


def load_score_state_cache_shard(
    path: str | Path,
    *,
    expected_record: D0ScoreStateShardRecord | None = None,
    verify_hashes: bool = True,
) -> D0ScoreStateCache:
    source = Path(path)
    if expected_record is not None:
        if source.name != expected_record.filename:
            raise D0ScoreStateCompatibilityError("score shard filename differs from index")
        if verify_hashes and file_fingerprint(source) != expected_record.file_sha256:
            raise D0ScoreStateCompatibilityError("score shard file hash mismatch")
        if int(source.stat().st_size) != int(expected_record.file_size):
            raise D0ScoreStateCompatibilityError("score shard file size mismatch")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if "__manifest_json__" not in archive.files:
                raise D0ScoreStateCompatibilityError("score shard has no manifest")
            manifest = json.loads(str(np.asarray(archive["__manifest_json__"]).item()))
            if (
                manifest.get("schema") != SCORE_STATE_CACHE_SCHEMA
                or int(manifest.get("schema_version", -1)) != SCORE_STATE_CACHE_SCHEMA_VERSION
            ):
                raise D0ScoreStateCompatibilityError("score shard schema is incompatible")
            required = set(_CACHE_TENSOR_FIELDS + _CACHE_ARRAY_FIELDS)
            missing = sorted(required.difference(archive.files))
            if missing:
                raise D0ScoreStateCompatibilityError(
                    "score shard is missing arrays: " + ", ".join(missing)
                )
            scalars = dict(_json_restore(manifest.get("cache_scalars", {})))
            missing_scalars = sorted(set(_CACHE_SCALAR_FIELDS).difference(scalars))
            if missing_scalars:
                raise D0ScoreStateCompatibilityError(
                    "score shard is missing scalar metadata: " + ", ".join(missing_scalars)
                )
            tensors = {
                name: torch.from_numpy(np.array(archive[name], copy=True))
                for name in _CACHE_TENSOR_FIELDS
            }
            arrays = {name: np.array(archive[name], copy=True) for name in _CACHE_ARRAY_FIELDS}
    except D0ScoreStateCompatibilityError:
        raise
    except Exception as exc:
        raise D0ScoreStateCompatibilityError(f"cannot read score shard: {exc}") from exc
    cache = D0ScoreStateCache(**tensors, **arrays, **scalars)
    validate_score_state_cache(cache)
    actual = score_state_cache_fingerprint(cache)
    if verify_hashes and actual != str(manifest.get("cache_fingerprint", "")):
        raise D0ScoreStateCompatibilityError("score shard content hash mismatch")
    if expected_record is not None:
        checks = {
            "cache fingerprint": actual == expected_record.cache_fingerprint,
            "path IDs": tuple(int(value) for value in cache.path_ids.tolist()) == expected_record.path_ids,
            "role": cache.role == expected_record.role,
            "origin": cache.origin == expected_record.origin,
            "anchor plan": cache.anchor_plan_fingerprint == expected_record.anchor_plan_fingerprint,
            "scientific fingerprint": cache.scientific_fingerprint
            == expected_record.scientific_fingerprint,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise D0ScoreStateCompatibilityError(
                "score shard differs from index: " + ", ".join(failed)
            )
    return cache


def _record_for_existing(path: Path, cache: D0ScoreStateCache, *, shard_id: int) -> D0ScoreStateShardRecord:
    return D0ScoreStateShardRecord(
        shard_id=int(shard_id),
        filename=path.name,
        path_ids=tuple(int(value) for value in cache.path_ids.tolist()),
        role=cache.role,
        origin=cache.origin,
        cache_fingerprint=score_state_cache_fingerprint(cache),
        file_sha256=file_fingerprint(path),
        file_size=int(path.stat().st_size),
        anchor_plan_fingerprint=cache.anchor_plan_fingerprint,
        scientific_fingerprint=cache.scientific_fingerprint,
    )


def verified_score_state_shard_or_none(
    path: str | Path,
    *,
    expected_record: D0ScoreStateShardRecord | None = None,
    expected_path_ids: Sequence[int] | None = None,
    expected_role: str | None = None,
    expected_origin: str | None = None,
    expected_anchor_plan_fingerprint: str | None = None,
    expected_scientific_fingerprint: str | None = None,
) -> D0ScoreStateCache | None:
    """Return a fully verified reusable shard, or ``None`` for damaged bytes.

    A readable but semantically different shard is not corruption and raises a
    compatibility error so a changed experiment cannot silently overwrite it.
    """

    source = Path(path)
    if not source.exists():
        return None
    try:
        cache = load_score_state_cache_shard(
            source, expected_record=expected_record, verify_hashes=True
        )
    except (D0ScoreStateCompatibilityError, OSError, ValueError):
        # When an index record points at this filename, any integrity failure is
        # recoverable corruption.  Without a record, try an internal load once
        # to distinguish damaged bytes from a valid artifact for another run.
        if expected_record is not None:
            return None
        try:
            cache = load_score_state_cache_shard(source, verify_hashes=True)
        except (D0ScoreStateCompatibilityError, OSError, ValueError):
            return None
    semantic_checks = {
        "path IDs": expected_path_ids is None
        or tuple(int(value) for value in cache.path_ids.tolist())
        == tuple(int(value) for value in expected_path_ids),
        "role": expected_role is None or cache.role == str(expected_role),
        "origin": expected_origin is None or cache.origin == str(expected_origin),
        "anchor plan": expected_anchor_plan_fingerprint is None
        or cache.anchor_plan_fingerprint == str(expected_anchor_plan_fingerprint),
        "scientific fingerprint": expected_scientific_fingerprint is None
        or cache.scientific_fingerprint == str(expected_scientific_fingerprint),
    }
    failed = [name for name, passed in semantic_checks.items() if not passed]
    if failed:
        raise D0ScoreStateCompatibilityError(
            "existing score shard is from a different experiment: " + ", ".join(failed)
        )
    return cache


def recover_score_state_shard(
    path: str | Path,
    cache: D0ScoreStateCache,
    *,
    shard_id: int,
    expected_record: D0ScoreStateShardRecord | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[D0ScoreStateShardRecord, bool]:
    """Reuse a verified shard or atomically rebuild missing/corrupt content.

    Returns ``(record, rebuilt)``.  A valid artifact with different scientific
    semantics raises instead of being overwritten.
    """

    target = Path(path)
    validate_score_state_cache(cache)
    reusable = verified_score_state_shard_or_none(
        target,
        expected_record=expected_record,
        expected_path_ids=cache.path_ids.tolist(),
        expected_role=cache.role,
        expected_origin=cache.origin,
        expected_anchor_plan_fingerprint=cache.anchor_plan_fingerprint,
        expected_scientific_fingerprint=cache.scientific_fingerprint,
    )
    if reusable is not None:
        if score_state_cache_fingerprint(reusable) != score_state_cache_fingerprint(cache):
            raise D0ScoreStateCompatibilityError("existing score shard content differs from rebuild")
        return _record_for_existing(target, reusable, shard_id=int(shard_id)), False
    return (
        save_score_state_cache_shard(
            target, cache, shard_id=int(shard_id), metadata=dict(metadata or {})
        ),
        True,
    )


@dataclass(frozen=True)
class D0ScoreStateCacheIndex:
    expected_path_ids: tuple[int, ...]
    records: tuple[D0ScoreStateShardRecord, ...]
    scientific_fingerprint: str
    metadata: Mapping[str, Any]
    fingerprint: str

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema": SCORE_STATE_INDEX_SCHEMA,
            "schema_version": SCORE_STATE_INDEX_SCHEMA_VERSION,
            "expected_path_ids": [int(value) for value in self.expected_path_ids],
            "records": [record.to_dict() for record in self.records],
            "scientific_fingerprint": str(self.scientific_fingerprint),
            "metadata": _json_safe(dict(self.metadata)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "fingerprint": str(self.fingerprint)}


def _validate_records(
    records: Sequence[D0ScoreStateShardRecord],
    expected_path_ids: Sequence[int],
    scientific_fingerprint: str,
) -> None:
    if not records:
        raise D0ScoreStateCompatibilityError("score cache index contains no shards")
    ids = [int(record.shard_id) for record in records]
    names = [str(record.filename) for record in records]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise D0ScoreStateCompatibilityError("score cache index repeats a shard ID or filename")
    for name in names:
        candidate = Path(name)
        if candidate.name != name or candidate.is_absolute() or name in {"", ".", ".."}:
            raise D0ScoreStateCompatibilityError("score shard filename must be a local basename")
    combined = [int(path_id) for record in records for path_id in record.path_ids]
    if len(set(combined)) != len(combined):
        raise D0ScoreStateCompatibilityError("score cache shards overlap in path IDs")
    if sorted(combined) != sorted(int(value) for value in expected_path_ids):
        raise D0ScoreStateCompatibilityError("score cache shards do not cover expected path IDs")
    for record in records:
        if record.scientific_fingerprint != str(scientific_fingerprint):
            raise D0ScoreStateCompatibilityError("score shard scientific fingerprint differs from index")
        if len(record.cache_fingerprint) != 64 or len(record.file_sha256) != 64 or record.file_size <= 0:
            raise D0ScoreStateCompatibilityError("score shard integrity record is invalid")
        if record.origin == PARENT_ORIGIN and record.role not in PARENT_SCORE_ROLES:
            raise D0ScoreStateCompatibilityError("parent audit role leaked into score cache index")
        if record.origin == FRESH_ORIGIN and record.role not in FRESH_SCORE_ROLES:
            raise D0ScoreStateCompatibilityError("fresh score shard has an invalid role")


def make_score_state_cache_index(
    records: Sequence[D0ScoreStateShardRecord],
    *,
    expected_path_ids: Sequence[int] | np.ndarray | Tensor,
    scientific_fingerprint: str,
    metadata: Mapping[str, Any] | None = None,
) -> D0ScoreStateCacheIndex:
    expected = tuple(int(value) for value in _path_ids(expected_path_ids).tolist())
    ordered = tuple(sorted(records, key=lambda item: int(item.shard_id)))
    _validate_records(ordered, expected, str(scientific_fingerprint))
    provisional = D0ScoreStateCacheIndex(
        expected_path_ids=expected,
        records=ordered,
        scientific_fingerprint=str(scientific_fingerprint),
        metadata=dict(metadata or {}),
        fingerprint="",
    )
    return replace(provisional, fingerprint=config_fingerprint(provisional.semantic_dict()))


def save_score_state_cache_index(path: str | Path, index: D0ScoreStateCacheIndex) -> None:
    _validate_records(index.records, index.expected_path_ids, index.scientific_fingerprint)
    if index.fingerprint != config_fingerprint(index.semantic_dict()):
        raise D0ScoreStateCompatibilityError("score cache index fingerprint mismatch")
    atomic_write_json(path, index.to_dict())


def load_score_state_cache_index(
    path: str | Path, *, verify_shards: bool = True
) -> D0ScoreStateCacheIndex:
    index_path = Path(path)
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise D0ScoreStateCompatibilityError(f"cannot read score cache index: {exc}") from exc
    if (
        payload.get("schema") != SCORE_STATE_INDEX_SCHEMA
        or int(payload.get("schema_version", -1)) != SCORE_STATE_INDEX_SCHEMA_VERSION
    ):
        raise D0ScoreStateCompatibilityError("score cache index schema is incompatible")
    index = D0ScoreStateCacheIndex(
        expected_path_ids=tuple(int(value) for value in payload["expected_path_ids"]),
        records=tuple(D0ScoreStateShardRecord.from_dict(item) for item in payload["records"]),
        scientific_fingerprint=str(payload["scientific_fingerprint"]),
        metadata=dict(_json_restore(payload.get("metadata", {}))),
        fingerprint=str(payload.get("fingerprint", "")),
    )
    _validate_records(index.records, index.expected_path_ids, index.scientific_fingerprint)
    if index.fingerprint != config_fingerprint(index.semantic_dict()):
        raise D0ScoreStateCompatibilityError("score cache index fingerprint mismatch")
    if verify_shards:
        for record in index.records:
            load_score_state_cache_shard(
                index_path.parent / record.filename,
                expected_record=record,
                verify_hashes=True,
            )
    return index


def load_score_state_cache_shards(
    index_path: str | Path, *, verify_hashes: bool = True
) -> tuple[D0ScoreStateCacheIndex, list[D0ScoreStateCache]]:
    index = load_score_state_cache_index(index_path, verify_shards=False)
    root = Path(index_path).parent
    caches = [
        load_score_state_cache_shard(
            root / record.filename, expected_record=record, verify_hashes=verify_hashes
        )
        for record in index.records
    ]
    return index, caches


def slice_score_state_cache_paths(
    cache: D0ScoreStateCache, path_ids: Sequence[int] | np.ndarray | Tensor
) -> D0ScoreStateCache:
    validate_score_state_cache(cache)
    wanted = _path_ids(path_ids)
    lookup = {int(value): axis for axis, value in enumerate(cache.path_ids.tolist())}
    missing = [int(value) for value in wanted if int(value) not in lookup]
    if missing:
        raise KeyError("score cache has no path IDs: " + ", ".join(map(str, missing)))
    axes = np.asarray([lookup[int(value)] for value in wanted], dtype=np.int64)
    idx = torch.as_tensor(axes, dtype=torch.long)
    sliced = replace(
        cache,
        path_ids=cache.path_ids.index_select(0, idx).clone(),
        states=cache.states.index_select(0, idx).clone(),
        tau=cache.tau.index_select(0, idx).clone(),
        labels=cache.labels.index_select(0, idx).clone(),
        end_substeps=cache.end_substeps.index_select(0, idx).clone(),
        anchor_strata=cache.anchor_strata.index_select(0, idx).clone(),
        roles=np.ascontiguousarray(cache.roles[axes]),
        origins=np.ascontiguousarray(cache.origins[axes]),
        origin_path_ids=np.ascontiguousarray(cache.origin_path_ids[axes]),
        terminal_states=np.ascontiguousarray(cache.terminal_states[axes]),
        source_indices=np.ascontiguousarray(cache.source_indices[axes]),
        requested_labels=np.ascontiguousarray(cache.requested_labels[axes]),
        diagnostics={
            **dict(cache.diagnostics),
            "state_finite_fraction": 1.0,
            "state_min": float(cache.states.index_select(0, idx).min()),
            "max_simplex_mass_error": float(
                (cache.states.index_select(0, idx).double().sum(dim=-1) - 1.0).abs().max()
            ),
        },
    )
    validate_score_state_cache(sliced)
    return sliced


def merge_score_state_caches(caches: Sequence[D0ScoreStateCache]) -> D0ScoreStateCache:
    if not caches:
        raise ValueError("at least one score cache is required")
    for cache in caches:
        validate_score_state_cache(cache)
    first = caches[0]
    for cache in caches[1:]:
        compatible = (
            cache.role == first.role
            and cache.origin == first.origin
            and cache.anchor_plan_fingerprint == first.anchor_plan_fingerprint
            and cache.scientific_fingerprint == first.scientific_fingerprint
            and config_fingerprint(cache.kernel_metadata) == config_fingerprint(first.kernel_metadata)
            and config_fingerprint(cache.schedule_metadata) == config_fingerprint(first.schedule_metadata)
            and array_fingerprint(cache.rate_schedule) == array_fingerprint(first.rate_schedule)
            and cache.states.shape[1:] == first.states.shape[1:]
        )
        if not compatible:
            raise D0ScoreStateCompatibilityError("score cache shards cannot be merged")
    combined_ids = [int(value) for cache in caches for value in cache.path_ids.tolist()]
    if len(set(combined_ids)) != len(combined_ids):
        raise D0ScoreStateCompatibilityError("score cache merge repeats path IDs")
    source_diags: list[Any] = []
    source_provenance: list[dict[str, Any]] = []
    for cache in caches:
        source_diags.extend(list(cache.diagnostics.get("source_shards", [])))
        source_provenance.append(dict(cache.provenance))
    common_provenance = {
        key: value
        for key, value in dict(first.provenance).items()
        if key not in {"parent_source_shard_id", "parent_source_shard_sha256"}
    }
    merged_states = torch.cat([cache.states for cache in caches], dim=0)
    additive_names = (
        "limited_edges", "proposed_edges", "mobility_weight_sum",
        "limited_mobility_weight_sum", "noise_energy_sum",
        "limited_noise_energy_sum", "floor_correction_l1",
        "renorm_correction_l1", "floor_touched_pixels",
        "floor_proposed_pixels", "nonfinite_edges", "path_substep_count",
    )
    totals = {name: 0.0 for name in additive_names}
    for cache in caches:
        diagnostics = dict(cache.diagnostics)
        for name in additive_names:
            totals[name] += float(diagnostics.get(name, 0.0))
    totals["raw_limited_fraction"] = (
        0.0 if totals["proposed_edges"] <= 0.0
        else totals["limited_edges"] / totals["proposed_edges"]
    )
    totals["mobility_weighted_limited_fraction"] = (
        0.0 if totals["mobility_weight_sum"] <= 0.0
        else totals["limited_mobility_weight_sum"] / totals["mobility_weight_sum"]
    )
    totals["noise_energy_weighted_limited_fraction"] = (
        0.0 if totals["noise_energy_sum"] <= 0.0
        else totals["limited_noise_energy_sum"] / totals["noise_energy_sum"]
    )
    merged = replace(
        first,
        path_ids=torch.cat([cache.path_ids for cache in caches], dim=0),
        states=merged_states,
        tau=torch.cat([cache.tau for cache in caches], dim=0),
        labels=torch.cat([cache.labels for cache in caches], dim=0),
        end_substeps=torch.cat([cache.end_substeps for cache in caches], dim=0),
        anchor_strata=torch.cat([cache.anchor_strata for cache in caches], dim=0),
        roles=np.concatenate([cache.roles for cache in caches]),
        origins=np.concatenate([cache.origins for cache in caches]),
        origin_path_ids=np.concatenate([cache.origin_path_ids for cache in caches]),
        terminal_states=np.concatenate([cache.terminal_states for cache in caches], axis=0),
        source_indices=np.concatenate([cache.source_indices for cache in caches]),
        requested_labels=np.concatenate([cache.requested_labels for cache in caches]),
        provenance={
            **common_provenance,
            "source_provenance": source_provenance,
        },
        diagnostics={
            **totals,
            "state_finite_fraction": 1.0,
            "state_min": float(merged_states.min()),
            "max_simplex_mass_error": float(
                (merged_states.double().sum(dim=-1) - 1.0).abs().max()
            ),
            "source_shards": source_diags,
        },
    )
    validate_score_state_cache(merged)
    return merged


def make_score_state_anchor_plan(
    *,
    path_ids: Sequence[int] | np.ndarray | Tensor,
    total_substeps: int,
    seed: int,
    anchors_per_path: int = DEFAULT_SCORE_ANCHORS_PER_PATH,
    bin_counts: Sequence[int] = DEFAULT_SCORE_ANCHOR_BIN_COUNTS,
    minimum_forward_substep: int = DEFAULT_SCORE_MINIMUM_FORWARD_SUBSTEP,
    tau_fraction_edges: Sequence[float] = DEFAULT_TAU_FRACTION_EDGES,
) -> D0StratifiedAnchorPlan:
    """Make the 32-anchor positive-time plan used by score-state shards."""

    return make_stratified_anchor_plan(
        path_ids=path_ids,
        anchors_per_path=int(anchors_per_path),
        total_substeps=int(total_substeps),
        max_stride=int(minimum_forward_substep),
        seed=int(seed),
        tau_fraction_edges=tau_fraction_edges,
        bin_counts=bin_counts,
    )


def _state_diagnostics(states: Tensor) -> dict[str, Any]:
    return {
        "state_finite_fraction": float(torch.isfinite(states).double().mean()),
        "state_min": float(states.min()),
        "max_simplex_mass_error": float((states.double().sum(dim=-1) - 1.0).abs().max()),
    }


def _parent_manifest_kernel(index_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = index_path.parent.parent / "run_manifest.json"
    if not manifest_path.exists():
        return {}, {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    scientific = dict(payload.get("scientific_config", {}))
    kernel = dict(scientific.get("kernel", {}))
    provenance = {
        "parent_run_manifest": str(manifest_path.resolve()),
        "parent_run_manifest_sha256": file_fingerprint(manifest_path),
        "parent_run_schema": payload.get("schema"),
        "parent_run_schema_version": payload.get("schema_version"),
        "parent_cache_semantic_fingerprint": payload.get("cache_semantic_fingerprint"),
    }
    return kernel, provenance


def _schedule_from_parent(
    cache: D0MultiscaleCache, kernel: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "sample_steps": int(cache.sample_steps),
        "reference_substeps": int(cache.reference_substeps),
        "total_substeps": int(cache.sample_steps) * int(cache.reference_substeps),
        "horizon": float(cache.horizon),
        "dt_sub": float(cache.dt_sub),
        "tau_eff": kernel.get("tau_eff"),
        "rate_schedule_sha256": array_fingerprint(np.asarray(cache.rate_schedule, dtype=np.float64)),
    }


def _normalize_parent_kernel(
    cache: D0MultiscaleCache, supplied: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(supplied)
    result.setdefault("grid_size", int(cache.grid_size))
    result.setdefault("sample_steps", int(cache.sample_steps))
    result.setdefault("reference_substeps", int(cache.reference_substeps))
    result.setdefault("lambda_mix", float(cache.lambda_mix))
    result.setdefault("integrator", SCORE_STATE_REFERENCE_INTEGRATOR)
    result.setdefault("integrator_version", SCORE_STATE_REFERENCE_INTEGRATOR_VERSION)
    return result


def _validate_frozen_kernel_metadata(metadata: Mapping[str, Any]) -> None:
    missing = sorted(set(FROZEN_SCORE_KERNEL).difference(metadata))
    if missing:
        raise D0ScoreStateCompatibilityError(
            "parent kernel metadata is incomplete: " + ", ".join(missing)
        )
    mismatches: list[str] = []
    for key, expected in FROZEN_SCORE_KERNEL.items():
        actual = metadata[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15):
                mismatches.append(key)
        elif actual != expected:
            mismatches.append(key)
    if mismatches:
        raise D0ScoreStateCompatibilityError(
            "parent cache kernel differs from the frozen score kernel: " + ", ".join(mismatches)
        )


def _load_parent_role_ids(
    path_split_path: str | Path, expected_path_ids: Sequence[int]
) -> tuple[dict[str, np.ndarray], str]:
    split_path = Path(path_split_path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "experiment12-d0-three-way-path-split"
        or int(payload.get("schema_version", -1)) != 1
    ):
        raise D0ScoreStateCompatibilityError("parent path split schema is incompatible")
    split = D0ThreeWayPathSplit(
        train_path_ids=np.asarray(payload["train_path_ids"], dtype=np.int64),
        validation_path_ids=np.asarray(payload["validation_path_ids"], dtype=np.int64),
        confirmation_path_ids=np.asarray(payload["confirmation_path_ids"], dtype=np.int64),
        seed=int(payload["seed"]),
        fingerprint=str(payload["fingerprint"]),
    )
    try:
        validate_three_way_path_split(split, expected_path_ids)
    except (D0MultiscaleCompatibilityError, ValueError) as exc:
        raise D0ScoreStateCompatibilityError(f"parent path split is invalid: {exc}") from exc
    selection_alias = np.asarray(payload.get("selection_path_ids", split.validation_path_ids), dtype=np.int64)
    audit_alias = np.asarray(payload.get("audit_path_ids", split.confirmation_path_ids), dtype=np.int64)
    if not np.array_equal(selection_alias, split.validation_path_ids):
        raise D0ScoreStateCompatibilityError("parent selection alias differs from validation role")
    if not np.array_equal(audit_alias, split.confirmation_path_ids):
        raise D0ScoreStateCompatibilityError("parent audit alias differs from confirmation role")
    return {
        "train": split.train_path_ids.copy(),
        "selection": split.validation_path_ids.copy(),
        "audit": split.confirmation_path_ids.copy(),
    }, file_fingerprint(split_path)


def score_state_cache_from_multiscale(
    cache: D0MultiscaleCache,
    path_ids: Sequence[int] | np.ndarray | Tensor,
    *,
    role: str,
    scientific_fingerprint: str,
    kernel_metadata: Mapping[str, Any],
    schedule_metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> D0ScoreStateCache:
    """Materialize parent later states for an explicitly non-audit role."""

    if str(role) not in PARENT_SCORE_ROLES:
        raise D0ScoreStateCompatibilityError("parent score materialization forbids audit roles")
    wanted = _path_ids(path_ids)
    lookup = {int(value): axis for axis, value in enumerate(cache.path_ids.tolist())}
    missing = [int(value) for value in wanted if int(value) not in lookup]
    if missing:
        raise KeyError("parent cache has no path IDs: " + ", ".join(map(str, missing)))
    axes = np.asarray([lookup[int(value)] for value in wanted], dtype=np.int64)
    idx = torch.as_tensor(axes, dtype=torch.long)
    states = cache.later_states.index_select(0, idx).clone().float().cpu()
    kernel = _normalize_parent_kernel(cache, kernel_metadata)
    schedule = dict(schedule_metadata or _schedule_from_parent(cache, kernel))
    source_record = {
        "cache_fingerprint": config_fingerprint(
            {
                "path_ids": array_fingerprint(cache.path_ids),
                "later_states": array_fingerprint(cache.later_states),
                "anchor_plan_fingerprint": cache.anchor_plan_fingerprint,
            }
        ),
        "path_ids": [int(value) for value in wanted.tolist()],
        "diagnostics": _json_safe(dict(cache.diagnostics)),
    }
    result = D0ScoreStateCache(
        path_ids=torch.as_tensor(wanted, dtype=torch.long),
        states=states,
        tau=cache.tau.index_select(0, idx).clone().float().cpu(),
        labels=cache.labels.index_select(0, idx).clone().long().cpu(),
        end_substeps=cache.end_substeps.index_select(0, idx).clone().long().cpu(),
        anchor_strata=cache.anchor_strata.index_select(0, idx).clone().long().cpu(),
        tau_fraction_edges=np.ascontiguousarray(cache.tau_fraction_edges, dtype=np.float64),
        roles=np.full(wanted.size, str(role), dtype=f"<U{max(1, len(str(role)))}"),
        origins=np.full(wanted.size, PARENT_ORIGIN, dtype=f"<U{len(PARENT_ORIGIN)}"),
        origin_path_ids=wanted.copy(),
        terminal_states=np.ascontiguousarray(np.asarray(cache.terminal_states)[axes], dtype=np.float32),
        source_indices=np.ascontiguousarray(np.asarray(cache.source_indices)[axes], dtype=np.int64),
        requested_labels=np.ascontiguousarray(np.asarray(cache.requested_labels)[axes], dtype=np.int64),
        rate_schedule=np.ascontiguousarray(cache.rate_schedule, dtype=np.float64),
        horizon=float(cache.horizon),
        dt_sub=float(cache.dt_sub),
        sample_steps=int(cache.sample_steps),
        reference_substeps=int(cache.reference_substeps),
        lambda_mix=float(cache.lambda_mix),
        minimum_forward_substep=int(cache.strides.max().item()),
        anchor_plan_fingerprint=str(cache.anchor_plan_fingerprint),
        scientific_fingerprint=str(scientific_fingerprint),
        kernel_metadata=kernel,
        schedule_metadata=schedule,
        provenance={**dict(provenance or {}), "origin": PARENT_ORIGIN, "parent_role": str(role)},
        diagnostics={**_state_diagnostics(states), "source_shards": [source_record]},
    )
    validate_score_state_cache(result)
    return result


def _aggregate_rollout_diagnostics(
    caches: Sequence[D0MultiscaleCache | Mapping[str, Any]],
) -> dict[str, Any]:
    additive = (
        "masked_edges",
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
        "path_substep_count",
    )
    diagnostics = [
        item.diagnostics if isinstance(item, D0MultiscaleCache) else item
        for item in caches
    ]
    totals = {
        name: sum(float(values.get(name, 0.0)) for values in diagnostics)
        for name in additive
    }
    totals.update(
        {
            "raw_limited_fraction": 0.0
            if totals["proposed_edges"] <= 0.0
            else totals["masked_edges"] / totals["proposed_edges"],
            "mobility_weighted_limited_fraction": float("nan")
            if totals["mobility_weight_sum"] <= 0.0
            else totals["limited_mobility_weight_sum"] / totals["mobility_weight_sum"],
            "noise_energy_weighted_limited_fraction": float("nan")
            if totals["noise_energy_sum"] <= 0.0
            else totals["limited_noise_energy_sum"] / totals["noise_energy_sum"],
        }
    )
    return totals


def materialize_parent_score_state_shards(
    parent_index_path: str | Path,
    path_split_path: str | Path,
    output_dir: str | Path,
    *,
    scientific_fingerprint: str,
    roles: Sequence[str] = ("train", "selection"),
    shard_paths: int = DEFAULT_SCORE_SHARD_PATHS,
    resume: bool = True,
    metadata: Mapping[str, Any] | None = None,
    kernel_metadata: Mapping[str, Any] | None = None,
    schedule_metadata: Mapping[str, Any] | None = None,
    enforce_frozen_kernel: bool = True,
) -> D0ScoreStateCacheIndex:
    """Verify parent shards and atomically materialize train/selection states."""

    index_path = Path(parent_index_path)
    parent_index = load_multiscale_cache_index(index_path, verify_shards=False)
    normalized_roles = tuple(str(role) for role in roles)
    if not normalized_roles or len(set(normalized_roles)) != len(normalized_roles):
        raise ValueError("roles must be non-empty and unique")
    if any(role not in PARENT_SCORE_ROLES for role in normalized_roles):
        raise D0ScoreStateCompatibilityError("parent audit roles cannot be materialized")
    role_ids, split_sha = _load_parent_role_ids(path_split_path, parent_index.expected_path_ids)
    audit_ids = set(int(value) for value in role_ids["audit"])
    selected_ids = {int(value) for role in normalized_roles for value in role_ids[role]}
    if selected_ids.intersection(audit_ids):
        raise D0ScoreStateCompatibilityError("parent audit path leaked into requested roles")
    shard_size = int(shard_paths)
    if shard_size <= 0:
        raise ValueError("shard_paths must be positive")

    manifest_kernel, manifest_provenance = _parent_manifest_kernel(index_path)
    supplied_kernel = dict(manifest_kernel)
    supplied_kernel.update(dict(kernel_metadata or {}))
    verified_parent_diagnostics: list[Mapping[str, Any]] = []
    role_pieces: dict[str, list[D0ScoreStateCache]] = {role: [] for role in normalized_roles}
    common_kernel: dict[str, Any] | None = None
    common_schedule: dict[str, Any] | None = None
    parent_provenance = {
        **manifest_provenance,
        "parent_cache_index": str(index_path.resolve()),
        "parent_cache_index_sha256": file_fingerprint(index_path),
        "parent_cache_index_fingerprint": parent_index.fingerprint,
        "parent_scientific_fingerprint": parent_index.scientific_fingerprint,
        "parent_path_split": str(Path(path_split_path).resolve()),
        "parent_path_split_sha256": split_sha,
    }
    for record in parent_index.records:
        parent_cache = load_multiscale_cache_shard(
            index_path.parent / record.filename,
            expected_record=record,
            verify_hashes=True,
        )
        # Retain additive evidence, not the large multiscale transfer tensors.
        # Parent shards are verified one at a time so materializing state-only
        # rows remains practical on a laptop.
        verified_parent_diagnostics.append(dict(parent_cache.diagnostics))
        current_kernel = _normalize_parent_kernel(parent_cache, supplied_kernel)
        current_schedule = dict(schedule_metadata or _schedule_from_parent(parent_cache, current_kernel))
        if common_kernel is None:
            common_kernel = current_kernel
            common_schedule = current_schedule
            if enforce_frozen_kernel:
                _validate_frozen_kernel_metadata(common_kernel)
        elif (
            config_fingerprint(current_kernel) != config_fingerprint(common_kernel)
            or config_fingerprint(current_schedule) != config_fingerprint(common_schedule)
        ):
            raise D0ScoreStateCompatibilityError("parent cache shards disagree on kernel/schedule")
        available = set(int(value) for value in parent_cache.path_ids.tolist())
        for role in normalized_roles:
            wanted = [int(value) for value in role_ids[role] if int(value) in available]
            if wanted:
                role_pieces[role].append(
                    score_state_cache_from_multiscale(
                        parent_cache,
                        wanted,
                        role=role,
                        scientific_fingerprint=str(scientific_fingerprint),
                        kernel_metadata=current_kernel,
                        schedule_metadata=current_schedule,
                        provenance={
                            **parent_provenance,
                            "parent_source_shard_id": int(record.shard_id),
                            "parent_source_shard_sha256": record.file_sha256,
                        },
                    )
                )
    if any(not role_pieces[role] for role in normalized_roles):
        missing = [role for role in normalized_roles if not role_pieces[role]]
        raise D0ScoreStateCompatibilityError("parent cache has no requested role paths: " + ", ".join(missing))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_index_path = output_root / "cache_index.json"
    previous: D0ScoreStateCacheIndex | None = None
    if bool(resume) and output_index_path.exists():
        previous = load_score_state_cache_index(output_index_path, verify_shards=False)
        if previous.scientific_fingerprint != str(scientific_fingerprint):
            raise D0ScoreStateCompatibilityError("resume score cache scientific fingerprint differs")
    previous_records = {} if previous is None else {record.filename: record for record in previous.records}

    records: list[D0ScoreStateShardRecord] = []
    expected: list[int] = []
    shard_id = 0
    for role in normalized_roles:
        merged = merge_score_state_caches(role_pieces[role])
        ordered = np.sort(_as_numpy(merged.path_ids, dtype=np.dtype(np.int64)))
        if not np.array_equal(ordered, np.sort(role_ids[role])):
            raise D0ScoreStateCompatibilityError(f"materialized parent {role} paths are incomplete")
        for offset in range(0, int(ordered.size), shard_size):
            ids = ordered[offset : offset + shard_size]
            shard = slice_score_state_cache_paths(merged, ids)
            filename = f"parent-{role}-shard-{shard_id:05d}.npz"
            target = output_root / filename
            record, _rebuilt = recover_score_state_shard(
                target,
                shard,
                shard_id=shard_id,
                expected_record=previous_records.get(filename),
                metadata={"role": role, "origin": PARENT_ORIGIN, "shard_paths": shard_size},
            )
            records.append(record)
            expected.extend(int(value) for value in ids)
            shard_id += 1
    index = make_score_state_cache_index(
        records,
        expected_path_ids=expected,
        scientific_fingerprint=str(scientific_fingerprint),
        metadata={
            **dict(metadata or {}),
            **parent_provenance,
            "roles": list(normalized_roles),
            "shard_paths": shard_size,
            "parent_audit_paths_excluded": 1,
            "parent_rollout_diagnostics": _aggregate_rollout_diagnostics(
                verified_parent_diagnostics
            ),
            "kernel_metadata": common_kernel,
            "schedule_metadata": common_schedule,
        },
    )
    save_score_state_cache_index(output_index_path, index)
    return index


def build_fresh_score_state_cache_shard(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    anchor_plan: D0StratifiedAnchorPlan,
    path_ids: Sequence[int] | np.ndarray | Tensor,
    role: str,
    device: torch.device | str,
    seed: int,
    scientific_fingerprint: str,
    global_anchor_plan_fingerprint: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    enforce_frozen_kernel: bool = True,
    show_progress: bool = True,
) -> D0ScoreStateCache:
    """Roll out fresh audit/preflight paths and capture only later states."""

    if str(role) not in FRESH_SCORE_ROLES:
        raise D0ScoreStateCompatibilityError("fresh score paths may only be audit/preflight")
    if enforce_frozen_kernel:
        validate_frozen_score_kernel(dynamics_config, d0_config)
    wanted = _path_ids(path_ids)
    if not np.array_equal(wanted, np.asarray(anchor_plan.path_ids, dtype=np.int64)):
        raise ValueError("path_ids must match the supplied anchor plan order")
    sample_steps = int(d0_config.sample_steps)
    reference_substeps = int(d0_config.reference_substeps)
    total_substeps = sample_steps * reference_substeps
    if total_substeps != int(anchor_plan.total_substeps):
        raise ValueError("anchor plan total_substeps differs from D0 configuration")
    minimum = int(anchor_plan.max_stride)
    if np.any(np.asarray(anchor_plan.end_substeps) < minimum):
        raise ValueError("anchor plan violates its minimum forward substep")
    if str(d0_config.cache_build_mode).strip().lower().replace("_", "-") not in {
        "substep",
        "exact-substep",
        "exact",
    }:
        raise ValueError("fresh score cache requires exact substep mode")

    device_obj = torch.device(device)
    n = int(dynamics_config.grid_size)
    p_count = int(wanted.size)
    anchors = int(anchor_plan.anchors_per_path)
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

    ends_np = np.asarray(anchor_plan.end_substeps, dtype=np.int64)
    axes = np.indices(ends_np.shape, dtype=np.int64)
    flat_times = ends_np.reshape(-1)
    positive_outer = (flat_times - 1) // reference_substeps
    local = (flat_times - 1) % reference_substeps
    order = np.argsort(positive_outer, kind="stable")
    counts = np.bincount(positive_outer, minlength=sample_steps)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    capture_local = torch.as_tensor(local[order], dtype=torch.long, device=device_obj)
    capture_path = torch.as_tensor(axes[0].reshape(-1)[order], dtype=torch.long, device=device_obj)
    capture_anchor = torch.as_tensor(axes[1].reshape(-1)[order], dtype=torch.long, device=device_obj)
    captured = torch.empty((p_count, anchors, n * n), dtype=torch.float32, device=device_obj)
    filled = torch.zeros((p_count, anchors), dtype=torch.bool, device=device_obj)

    additive_names = (
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
    totals = {name: torch.zeros((), dtype=torch.float64, device=device_obj) for name in additive_names}
    max_simplex = torch.zeros((), dtype=torch.float64, device=device_obj)
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
        iterator: Any = range(sample_steps)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, total=sample_steps, desc=f"D0 score {role} shard")
            except Exception:  # pragma: no cover - progress is optional
                pass
        for outer in iterator:
            rate = float(rate_schedule[int(outer)])
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
                collect_diagnostics=True,
                diagnostics_device=True,
            )
            if (
                result.substep_states is None
                or result.valid_edge_mask is None
                or result.device_diagnostics is None
            ):
                raise RuntimeError("reference integrator omitted score-cache tensors")
            begin, end = int(offsets[int(outer)]), int(offsets[int(outer) + 1])
            if begin < end:
                section = slice(begin, end)
                p_axes = capture_path[section]
                a_axes = capture_anchor[section]
                l_axes = capture_local[section]
                captured[p_axes, a_axes] = result.substep_states[l_axes, p_axes]
                filled[p_axes, a_axes] = True
            for name in additive_names:
                if name not in {"limited_edges", "proposed_edges"}:
                    totals[name].add_(result.device_diagnostics[name])
            valid = result.valid_edge_mask
            totals["limited_edges"].add_((~valid).count_nonzero().to(torch.float64))
            totals["proposed_edges"].add_(float(valid.numel()))
            max_simplex = torch.maximum(
                max_simplex, result.device_diagnostics["max_simplex_mass_error"].to(torch.float64)
            )
            states = result.states
    if not bool(filled.all()):
        raise RuntimeError("fresh score rollout did not capture every anchor")
    names = list(additive_names) + ["max_simplex_mass_error"]
    values = torch.stack([totals[name] for name in additive_names] + [max_simplex]).detach().cpu().tolist()
    diag = dict(zip(names, values, strict=True))
    raw = 0.0 if diag["proposed_edges"] <= 0.0 else diag["limited_edges"] / diag["proposed_edges"]
    mobility = (
        float("nan")
        if diag["mobility_weight_sum"] <= 0.0
        else diag["limited_mobility_weight_sum"] / diag["mobility_weight_sum"]
    )
    noise = (
        float("nan")
        if diag["noise_energy_sum"] <= 0.0
        else diag["limited_noise_energy_sum"] / diag["noise_energy_sum"]
    )
    captured_cpu = captured.detach().cpu().float()
    state_diagnostics = _state_diagnostics(captured_cpu)
    rollout_simplex_error = float(diag.pop("max_simplex_mass_error"))
    diagnostics = {
        **diag,
        **state_diagnostics,
        "rollout_max_simplex_mass_error": rollout_simplex_error,
        "max_simplex_mass_error": max(
            rollout_simplex_error,
            float(state_diagnostics["max_simplex_mass_error"]),
        ),
        "raw_limited_fraction": float(raw),
        "mobility_weighted_limited_fraction": float(mobility),
        "noise_energy_weighted_limited_fraction": float(noise),
        "path_substep_count": int(p_count * total_substeps),
        "builder_seed": int(seed),
        "diagnostic_accumulation": "device-float64",
        "integrator": SCORE_STATE_REFERENCE_INTEGRATOR,
    }
    kernel = _kernel_metadata(dynamics_config, d0_config)
    schedule = _schedule_metadata(
        rate_schedule=np.asarray(rate_schedule, dtype=np.float64),
        horizon=horizon,
        dt_sub=dt_sub,
        d0_config=d0_config,
    )
    result_cache = D0ScoreStateCache(
        path_ids=torch.as_tensor(wanted, dtype=torch.long),
        states=captured_cpu,
        tau=(
            horizon
            - torch.as_tensor(anchor_plan.end_substeps, dtype=torch.float64) * dt_sub
        ).float(),
        labels=torch.as_tensor(labels_np, dtype=torch.long),
        end_substeps=torch.as_tensor(anchor_plan.end_substeps, dtype=torch.long),
        anchor_strata=torch.as_tensor(anchor_plan.stratum_indices, dtype=torch.long),
        tau_fraction_edges=np.ascontiguousarray(anchor_plan.tau_fraction_edges, dtype=np.float64),
        roles=np.full(p_count, str(role), dtype=f"<U{max(1, len(str(role)))}"),
        origins=np.full(p_count, FRESH_ORIGIN, dtype=f"<U{len(FRESH_ORIGIN)}"),
        origin_path_ids=wanted.copy(),
        terminal_states=states.detach().cpu().numpy().reshape(p_count, n, n).astype(np.float32),
        source_indices=np.asarray(source_idx_np, dtype=np.int64),
        requested_labels=np.asarray(labels_np, dtype=np.int64),
        rate_schedule=np.asarray(rate_schedule, dtype=np.float64),
        horizon=float(horizon),
        dt_sub=float(dt_sub),
        sample_steps=sample_steps,
        reference_substeps=reference_substeps,
        lambda_mix=float(d0_config.lambda_mix),
        minimum_forward_substep=minimum,
        anchor_plan_fingerprint=str(global_anchor_plan_fingerprint or anchor_plan.fingerprint),
        scientific_fingerprint=str(scientific_fingerprint),
        kernel_metadata=kernel,
        schedule_metadata=schedule,
        provenance={
            **dict(provenance or {}),
            "origin": FRESH_ORIGIN,
            "role": str(role),
            "builder_seed": int(seed),
            "local_anchor_plan_fingerprint": anchor_plan.fingerprint,
        },
        diagnostics=diagnostics,
    )
    validate_score_state_cache(result_cache)
    return result_cache


def build_fresh_score_state_shards(
    *,
    dataset_images: np.ndarray,
    dataset_labels: np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    output_dir: str | Path,
    path_ids: Sequence[int] | np.ndarray | Tensor,
    role: str = "audit",
    device: torch.device | str,
    seed: int,
    anchor_seed: int,
    scientific_fingerprint: str,
    anchors_per_path: int = DEFAULT_SCORE_ANCHORS_PER_PATH,
    bin_counts: Sequence[int] = DEFAULT_SCORE_ANCHOR_BIN_COUNTS,
    minimum_forward_substep: int = DEFAULT_SCORE_MINIMUM_FORWARD_SUBSTEP,
    shard_paths: int = DEFAULT_SCORE_SHARD_PATHS,
    resume: bool = True,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    enforce_frozen_kernel: bool = True,
    show_progress: bool = True,
) -> D0ScoreStateCacheIndex:
    """Build restartable fresh audit shards with shard-local deterministic RNG."""

    wanted = _path_ids(path_ids)
    if str(role) not in FRESH_SCORE_ROLES:
        raise D0ScoreStateCompatibilityError("fresh cache role must be audit or preflight")
    shard_size = int(shard_paths)
    if shard_size <= 0:
        raise ValueError("shard_paths must be positive")
    total_substeps = int(d0_config.sample_steps) * int(d0_config.reference_substeps)
    plan = make_score_state_anchor_plan(
        path_ids=wanted,
        total_substeps=total_substeps,
        seed=int(anchor_seed),
        anchors_per_path=int(anchors_per_path),
        bin_counts=bin_counts,
        minimum_forward_substep=int(minimum_forward_substep),
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    anchor_path = output_root / "anchor_plan.json"
    index_path = output_root / "cache_index.json"
    previous: D0ScoreStateCacheIndex | None = None
    if bool(resume) and index_path.exists():
        previous = load_score_state_cache_index(index_path, verify_shards=False)
        if previous.scientific_fingerprint != str(scientific_fingerprint):
            raise D0ScoreStateCompatibilityError("resume score cache scientific fingerprint differs")
        prior_plan = str(previous.metadata.get("anchor_plan_fingerprint", ""))
        if prior_plan != plan.fingerprint:
            raise D0ScoreStateCompatibilityError("resume score cache anchor plan differs")
    if bool(resume) and anchor_path.exists():
        try:
            existing_anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise D0ScoreStateCompatibilityError(
                f"cannot read resume score anchor plan: {exc}"
            ) from exc
        if str(existing_anchor.get("fingerprint", "")) != plan.fingerprint:
            raise D0ScoreStateCompatibilityError("resume score anchor artifact differs")
    else:
        atomic_write_json(anchor_path, plan.to_dict())
    previous_records = {} if previous is None else {record.filename: record for record in previous.records}
    records: list[D0ScoreStateShardRecord] = []
    shard_count = int(math.ceil(wanted.size / shard_size))
    for shard_id in range(shard_count):
        shard_ids = wanted[shard_id * shard_size : (shard_id + 1) * shard_size]
        filename = f"fresh-{role}-shard-{shard_id:05d}.npz"
        target = output_root / filename
        shard_seed = derive_score_state_shard_seed(int(seed), shard_id, scope=f"fresh-{role}")
        existing = verified_score_state_shard_or_none(
            target,
            expected_record=previous_records.get(filename),
            expected_path_ids=shard_ids,
            expected_role=str(role),
            expected_origin=FRESH_ORIGIN,
            expected_anchor_plan_fingerprint=plan.fingerprint,
            expected_scientific_fingerprint=str(scientific_fingerprint),
        )
        if existing is not None:
            if int(existing.provenance.get("builder_seed", -1)) != int(shard_seed):
                raise D0ScoreStateCompatibilityError("resume score shard RNG seed differs")
            records.append(_record_for_existing(target, existing, shard_id=shard_id))
            continue
        local_plan = slice_anchor_plan(plan, shard_ids)
        shard = build_fresh_score_state_cache_shard(
            dataset_images=dataset_images,
            dataset_labels=dataset_labels,
            dynamics_config=dynamics_config,
            d0_config=d0_config,
            anchor_plan=local_plan,
            path_ids=shard_ids,
            role=str(role),
            device=device,
            seed=int(shard_seed),
            scientific_fingerprint=str(scientific_fingerprint),
            global_anchor_plan_fingerprint=plan.fingerprint,
            provenance={
                **dict(provenance or {}),
                "base_seed": int(seed),
                "anchor_seed": int(anchor_seed),
                "shard_id": int(shard_id),
            },
            enforce_frozen_kernel=bool(enforce_frozen_kernel),
            show_progress=bool(show_progress),
        )
        records.append(
            save_score_state_cache_shard(
                target,
                shard,
                shard_id=shard_id,
                metadata={
                    "role": str(role),
                    "origin": FRESH_ORIGIN,
                    "shard_seed": int(shard_seed),
                    "shard_paths": shard_size,
                },
            )
        )
    index = make_score_state_cache_index(
        records,
        expected_path_ids=wanted,
        scientific_fingerprint=str(scientific_fingerprint),
        metadata={
            **dict(metadata or {}),
            "role": str(role),
            "origin": FRESH_ORIGIN,
            "base_seed": int(seed),
            "anchor_seed": int(anchor_seed),
            "anchor_plan_fingerprint": plan.fingerprint,
            "anchors_per_path": int(anchors_per_path),
            "anchor_bin_counts": [int(value) for value in bin_counts],
            "minimum_forward_substep": int(minimum_forward_substep),
            "shard_paths": shard_size,
            "shard_seed_derivation": "sha256(d0-score-state-v1:scope:base-seed:shard-id)",
        },
    )
    save_score_state_cache_index(index_path, index)
    return index


__all__ = [
    "ALL_SCORE_ROLES",
    "DEFAULT_SCORE_ANCHORS_PER_PATH",
    "DEFAULT_SCORE_ANCHOR_BIN_COUNTS",
    "DEFAULT_SCORE_MINIMUM_FORWARD_SUBSTEP",
    "DEFAULT_SCORE_SHARD_PATHS",
    "D0ScoreStateCache",
    "D0ScoreStateCacheIndex",
    "D0ScoreStateCompatibilityError",
    "D0ScoreStateShardRecord",
    "FRESH_ORIGIN",
    "FROZEN_SCORE_KERNEL",
    "PARENT_ORIGIN",
    "SCORE_STATE_CACHE_SCHEMA",
    "SCORE_STATE_CACHE_SCHEMA_VERSION",
    "SCORE_STATE_CONTRACT",
    "SCORE_STATE_INDEX_SCHEMA",
    "SCORE_STATE_INDEX_SCHEMA_VERSION",
    "build_fresh_score_state_cache_shard",
    "build_fresh_score_state_shards",
    "derive_score_state_shard_seed",
    "load_score_state_cache_index",
    "load_score_state_cache_shard",
    "load_score_state_cache_shards",
    "make_score_state_anchor_plan",
    "make_score_state_cache_index",
    "materialize_parent_score_state_shards",
    "merge_score_state_caches",
    "recover_score_state_shard",
    "save_score_state_cache_index",
    "save_score_state_cache_shard",
    "score_state_cache_fingerprint",
    "score_state_cache_from_multiscale",
    "slice_score_state_cache_paths",
    "validate_frozen_score_kernel",
    "validate_score_state_cache",
    "verified_score_state_shard_or_none",
]
