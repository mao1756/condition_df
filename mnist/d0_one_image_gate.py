from __future__ import annotations

"""Production helpers for the Experiment 12 one-image direct-Doob gate.

The orchestration CLI intentionally lives elsewhere.  This module contains the
artifact, split, validation, checkpoint, and fail-closed gate primitives shared
by that CLI and focused regression tests.  It does not change the behavior of
the legacy :mod:`mnist.experiment12_d0` entry point.
"""

import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, project_edge_flux_torch, temporary_ema_weights
from mnist.experiment12_d0 import (
    D0TrainingCache,
    Experiment12D0Config,
    _direct_reverse_free_block_baseline_from_batch,
)


CACHE_SCHEMA = "experiment12-d0-one-image-cache"
CACHE_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = "experiment12-d0-one-image-training-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
TIME_BINS: tuple[tuple[float, float], ...] = tuple((i / 5.0, (i + 1) / 5.0) for i in range(5))


class ArtifactCompatibilityError(ValueError):
    """Raised when an artifact cannot be used by the strict production run."""


class LegacyArtifactError(ArtifactCompatibilityError):
    """Raised when a legacy artifact is requested for strict resume/gating."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _from_jsonable(value: Any) -> Any:
    """Undo the explicit non-finite encoding used by :func:`_jsonable`."""

    if isinstance(value, Mapping):
        return {str(key): _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes suitable for semantic fingerprints."""

    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def config_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def array_fingerprint(value: np.ndarray | Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy() if isinstance(value, Tensor) else np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def file_fingerprint(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(paths: Sequence[str | Path]) -> str:
    """Fingerprint source contents while retaining stable relative labels."""

    records = []
    for path in sorted(
        (Path(path_like) for path_like in paths),
        key=lambda candidate: candidate.as_posix(),
    ):
        records.append({"path": path.as_posix(), "sha256": file_fingerprint(path)})
    return config_fingerprint(records)


def exact_torch_backend_record(
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Capture backend state that can alter an exact resumed trajectory."""

    device_obj = None if device is None else torch.device(device)
    record: dict[str, Any] = {
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_runtime_version": torch.backends.cudnn.version(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    ):
        if hasattr(torch.backends.cuda.matmul, name):
            record["cuda_matmul_" + name] = bool(
                getattr(torch.backends.cuda.matmul, name)
            )
    if device_obj is not None and device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the production gate but is unavailable")
        index = (
            int(device_obj.index)
            if device_obj.index is not None
            else int(torch.cuda.current_device())
        )
        properties = torch.cuda.get_device_properties(index)
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            driver_versions = [
                line.strip()
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "exact CUDA resume could not fingerprint the NVIDIA driver with nvidia-smi"
            ) from exc
        if not driver_versions:
            raise RuntimeError("nvidia-smi returned no NVIDIA driver version")
        record.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": str(properties.name),
                "cuda_device_uuid": str(getattr(properties, "uuid", "")),
                "cuda_compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "cuda_total_memory": int(properties.total_memory),
                "nvidia_driver_versions": driver_versions,
            }
        )
    return record


def configure_exact_torch_backend(
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Enforce deterministic PyTorch backend settings for exact resume."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "exact CUDA resume requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    ):
        if hasattr(torch.backends.cuda.matmul, name):
            setattr(torch.backends.cuda.matmul, name, False)
    torch.set_float32_matmul_precision("highest")
    # The project already disables MKLDNN for its small CPU U-Nets; making that
    # state explicit keeps CPU restart arithmetic fingerprintable as well.
    torch.backends.mkldnn.enabled = False
    return exact_torch_backend_record(device)


def assert_fingerprints_match(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str = "artifact",
    exact_keys: bool = False,
) -> None:
    """Fail closed when any expected semantic fingerprint differs.

    ``expected`` is normally the current run manifest, so subset matching lets
    callers introduce new non-semantic metadata without invalidating an older
    artifact of the same schema.  ``exact_keys`` is available for exact resume
    contracts that freeze the complete fingerprint mapping.
    """

    actual_dict = dict(actual)
    expected_dict = dict(expected)
    missing = sorted(set(expected_dict).difference(actual_dict))
    changed = sorted(key for key in expected_dict if key in actual_dict and actual_dict[key] != expected_dict[key])
    extra = sorted(set(actual_dict).difference(expected_dict)) if exact_keys else []
    if missing or changed or extra:
        pieces = []
        if missing:
            pieces.append("missing=" + ",".join(missing))
        if changed:
            pieces.append("changed=" + ",".join(changed))
        if extra:
            pieces.append("extra=" + ",".join(extra))
        raise ArtifactCompatibilityError(f"{context} fingerprint mismatch ({'; '.join(pieces)})")


def _atomic_replace(path: Path, writer: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(_jsonable(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace(Path(path), write)


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows_list = [dict(row) for row in rows]

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            if rows_list:
                fieldnames: list[str] = []
                for row in rows_list:
                    for key in row:
                        if key not in fieldnames:
                            fieldnames.append(key)
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows({key: _jsonable(value) for key, value in row.items()} for row in rows_list)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace(Path(path), write)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace(Path(path), write)


def atomic_copy_file(source: str | Path, destination: str | Path) -> None:
    """Atomically copy an artifact while preserving its exact byte identity."""

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    def write(temporary: Path) -> None:
        with source_path.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())

    _atomic_replace(Path(destination), write)


@dataclass(frozen=True)
class D0PathSplit:
    train_path_ids: np.ndarray
    validation_path_ids: np.ndarray
    train_slice_indices: np.ndarray
    validation_slice_indices: np.ndarray
    seed: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "train_path_ids": self.train_path_ids.astype(np.int64).tolist(),
            "validation_path_ids": self.validation_path_ids.astype(np.int64).tolist(),
            "train_slice_indices": self.train_slice_indices.astype(np.int64).tolist(),
            "validation_slice_indices": self.validation_slice_indices.astype(np.int64).tolist(),
            "fingerprint": str(self.fingerprint),
        }


def slice_indices_for_paths(cache: D0TrainingCache, path_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    wanted = np.asarray(path_ids, dtype=np.int64).reshape(-1)
    if wanted.size == 0:
        return np.empty(0, dtype=np.int64)
    paths = cache.path_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.flatnonzero(np.isin(paths, wanted)).astype(np.int64)


def deterministic_path_split(
    cache: D0TrainingCache,
    *,
    validation_paths: int,
    seed: int,
) -> D0PathSplit:
    """Split whole forward paths with no slice-level leakage."""

    all_paths = np.unique(cache.path_indices.detach().cpu().numpy().astype(np.int64, copy=False))
    terminal_count = int(np.asarray(cache.terminal_states).shape[0])
    expected_paths = np.arange(terminal_count, dtype=np.int64)
    if all_paths.size != terminal_count or not np.array_equal(all_paths, expected_paths):
        raise ValueError("cache path_indices must cover every terminal path exactly from 0 to P-1")
    count = int(validation_paths)
    if count <= 0 or count >= all_paths.size:
        raise ValueError("validation_paths must be positive and smaller than the cache path count")
    permutation = np.random.default_rng(int(seed)).permutation(all_paths)
    validation = np.sort(permutation[:count]).astype(np.int64)
    train = np.sort(permutation[count:]).astype(np.int64)
    train_slices = slice_indices_for_paths(cache, train)
    validation_slices = slice_indices_for_paths(cache, validation)
    if np.intersect1d(train_slices, validation_slices).size:
        raise RuntimeError("internal whole-path split error: slice leakage")
    semantic = {
        "seed": int(seed),
        "train_path_ids": train.tolist(),
        "validation_path_ids": validation.tolist(),
        "path_indices_sha256": array_fingerprint(cache.path_indices),
    }
    return D0PathSplit(
        train_path_ids=train,
        validation_path_ids=validation,
        train_slice_indices=train_slices,
        validation_slice_indices=validation_slices,
        seed=int(seed),
        fingerprint=config_fingerprint(semantic),
    )


def _cache_batch(cache: D0TrainingCache, indices: Tensor, device: torch.device) -> dict[str, Tensor]:
    count = int(indices.numel())
    return {
        "states": cache.states.index_select(0, indices).to(device),
        "tau": cache.tau.index_select(0, indices).to(device),
        "labels": cache.labels.index_select(0, indices).to(device),
        "starts": cache.starts.index_select(0, indices).to(device),
        "physical_transfers": cache.physical_transfers.index_select(0, indices).to(device),
        "stride_substeps": torch.full((count,), int(cache.stride_substeps), dtype=torch.long, device=device),
        "reference_substeps": torch.full((count,), int(cache.reference_substeps), dtype=torch.long, device=device),
        "dt_sub": torch.full((count,), float(cache.dt_sub), dtype=torch.float32, device=device),
        "rate_schedule": torch.as_tensor(cache.rate_schedule, dtype=torch.float32, device=device),
    }


@torch.no_grad()
def direct_doob_targets(
    cache: D0TrainingCache,
    dynamics_config: DirectFluxMNISTConfig,
    slice_indices: Sequence[int] | np.ndarray,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> Tensor:
    """Return unscaled projected direct-Doob residual targets on the CPU."""

    device_obj = torch.device(device)
    indices_np = np.asarray(slice_indices, dtype=np.int64).reshape(-1)
    if indices_np.size == 0:
        return torch.empty((0, 2, int(dynamics_config.grid_size), int(dynamics_config.grid_size)))
    if int(cache.stride_substeps) != 1:
        raise ValueError("production direct-Doob targets require cache stride_substeps=1")
    pieces: list[Tensor] = []
    for offset in range(0, indices_np.size, max(1, int(batch_size))):
        idx = torch.as_tensor(indices_np[offset : offset + max(1, int(batch_size))], dtype=torch.long)
        batch = _cache_batch(cache, idx, device_obj)
        baseline = _direct_reverse_free_block_baseline_from_batch(batch, dynamics_config)
        target = project_edge_flux_torch(
            batch["physical_transfers"].to(dtype=baseline.dtype) - baseline,
            grid_size=int(dynamics_config.grid_size),
        )
        pieces.append(target.detach().cpu())
    return torch.cat(pieces, dim=0)


def infer_training_target_scale(
    cache: D0TrainingCache,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    train_slice_indices: Sequence[int] | np.ndarray,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> float:
    """Infer global target RMS strictly from training paths."""

    indices = np.asarray(train_slice_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("cannot infer target scale from an empty training split")
    if float(d0_config.physical_target_scale) > 0.0:
        return float(d0_config.physical_target_scale)
    targets = direct_doob_targets(
        cache,
        dynamics_config,
        indices,
        device=device,
        batch_size=batch_size,
    ).float()
    finite = targets[torch.isfinite(targets)]
    if finite.numel() == 0:
        raise ValueError("training direct-Doob targets contain no finite values")
    scale = float(torch.sqrt(finite.double().square().mean()).item())
    scale = max(scale, float(d0_config.physical_target_scale_floor))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("training target scale must be finite and positive")
    return scale


def freeze_training_target_scale(
    cache: D0TrainingCache,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    split: D0PathSplit,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> Experiment12D0Config:
    scale = infer_training_target_scale(
        cache,
        dynamics_config,
        d0_config,
        split.train_slice_indices,
        device=device,
        batch_size=batch_size,
    )
    cache.physical_target_scale = float(scale)
    return replace(d0_config, physical_target_scale=float(scale))


_CACHE_TENSOR_FIELDS = (
    "states",
    "tau",
    "labels",
    "innovations",
    "masks",
    "starts",
    "path_indices",
    "start_images",
    "earlier_states",
    "physical_transfers",
)
_CACHE_ARRAY_FIELDS = ("terminal_states", "source_indices", "requested_labels", "rate_schedule")
_CACHE_SCALAR_FIELDS = (
    "physical_target_scale",
    "horizon",
    "dt_sub",
    "stride_substeps",
    "sample_steps",
    "reference_substeps",
    "lambda_mix",
    "raw_limited_fraction",
    "mobility_weighted_limited_fraction",
    "noise_energy_weighted_limited_fraction",
    "valid_innovation_fraction",
    "valid_innovation_mobility_fraction",
    "valid_innovation_noise_energy_fraction",
    "floor_correction_l1",
    "renorm_correction_l1",
    "teacher_mode",
    "cache_build_mode",
    "requested_stride_substeps",
    "floor_touched_pixels",
    "floor_proposed_pixels",
    "floor_touched_fraction",
)


def fingerprint_d0_cache(cache: D0TrainingCache) -> str:
    digest = hashlib.sha256()
    for name in _CACHE_TENSOR_FIELDS:
        digest.update(name.encode("utf-8"))
        digest.update(array_fingerprint(getattr(cache, name)).encode("ascii"))
    for name in _CACHE_ARRAY_FIELDS:
        digest.update(name.encode("utf-8"))
        digest.update(array_fingerprint(np.asarray(getattr(cache, name))).encode("ascii"))
    for name in _CACHE_SCALAR_FIELDS:
        digest.update(name.encode("utf-8"))
        digest.update(canonical_json_bytes(getattr(cache, name)))
    for name in ("trajectory_window_states", "trajectory_window_valid", "trajectory_window_depths"):
        value = getattr(cache, name)
        digest.update(name.encode("utf-8"))
        digest.update(b"none" if value is None else array_fingerprint(value).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class D0CacheArtifact:
    cache: D0TrainingCache
    metadata: dict[str, Any]
    fingerprints: dict[str, Any]
    cache_fingerprint: str
    schema_version: int = CACHE_SCHEMA_VERSION


def _cache_payload(cache: D0TrainingCache, metadata: Mapping[str, Any], fingerprints: Mapping[str, Any]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for name in _CACHE_TENSOR_FIELDS:
        value = getattr(cache, name).detach().cpu().contiguous().numpy()
        payload[name] = value
    for name in _CACHE_ARRAY_FIELDS:
        payload[name] = np.ascontiguousarray(getattr(cache, name))
    scalar_metadata = {name: getattr(cache, name) for name in _CACHE_SCALAR_FIELDS}
    manifest = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "metadata": _jsonable(metadata),
        "fingerprints": _jsonable(fingerprints),
        "cache_scalars": _jsonable(scalar_metadata),
        "cache_fingerprint": fingerprint_d0_cache(cache),
    }
    payload["__manifest_json__"] = np.asarray(json.dumps(manifest, sort_keys=True))
    for name in ("trajectory_window_states", "trajectory_window_valid", "trajectory_window_depths"):
        value = getattr(cache, name)
        if value is not None:
            payload[name] = value.detach().cpu().contiguous().numpy()
    return payload


def save_cache_bundle(
    path: str | Path,
    cache: D0TrainingCache,
    *,
    metadata: Mapping[str, Any] | None = None,
    fingerprints: Mapping[str, Any] | None = None,
) -> D0CacheArtifact:
    """Atomically persist every field needed for exact cache reuse."""

    metadata_dict = dict(metadata or {})
    fingerprint_dict = dict(fingerprints or {})
    payload = _cache_payload(cache, metadata_dict, fingerprint_dict)

    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace(Path(path), write)
    return D0CacheArtifact(
        cache=cache,
        metadata=metadata_dict,
        fingerprints=fingerprint_dict,
        cache_fingerprint=fingerprint_d0_cache(cache),
    )


def _manifest_from_npz(archive: Any) -> dict[str, Any]:
    if "__manifest_json__" not in archive.files:
        raise LegacyArtifactError("legacy D0 cache has no production schema and is report-only")
    raw = archive["__manifest_json__"]
    manifest = json.loads(str(raw.item()))
    if manifest.get("schema") != CACHE_SCHEMA or int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ArtifactCompatibilityError(
            f"unsupported D0 cache schema {manifest.get('schema')!r} version {manifest.get('schema_version')!r}"
        )
    return manifest


def _validate_loaded_cache(cache: D0TrainingCache) -> None:
    size = int(cache.states.shape[0])
    for name in ("tau", "labels", "innovations", "masks", "starts", "path_indices", "start_images", "earlier_states", "physical_transfers"):
        if int(getattr(cache, name).shape[0]) != size:
            raise ArtifactCompatibilityError(f"cache field {name} has inconsistent slice count")
    if cache.states.ndim != 2 or cache.earlier_states.shape != cache.states.shape or cache.start_images.shape != cache.states.shape:
        raise ArtifactCompatibilityError("cache state tensors have incompatible shapes")
    if cache.innovations.shape != cache.physical_transfers.shape or cache.masks.shape != cache.innovations.shape:
        raise ArtifactCompatibilityError("cache edge tensors have incompatible shapes")
    if int(cache.terminal_states.shape[0]) != int(cache.source_indices.shape[0]) or int(cache.source_indices.shape[0]) != int(cache.requested_labels.shape[0]):
        raise ArtifactCompatibilityError("cache path-level arrays have inconsistent lengths")
    if not np.isfinite(float(cache.physical_target_scale)) or float(cache.physical_target_scale) <= 0.0:
        raise ArtifactCompatibilityError("cache physical_target_scale must be finite and positive")


def load_cache_bundle(
    path: str | Path,
    *,
    expected_fingerprints: Mapping[str, Any] | None = None,
    verify_content: bool = True,
    exact_fingerprints: bool = True,
) -> D0CacheArtifact:
    with np.load(Path(path), allow_pickle=False) as archive:
        manifest = _manifest_from_npz(archive)
        missing = sorted(set(_CACHE_TENSOR_FIELDS + _CACHE_ARRAY_FIELDS).difference(archive.files))
        if missing:
            raise ArtifactCompatibilityError("production cache is missing fields: " + ", ".join(missing))
        scalars = dict(_from_jsonable(manifest.get("cache_scalars", {})))
        missing_scalars = sorted(set(_CACHE_SCALAR_FIELDS).difference(scalars))
        if missing_scalars:
            raise ArtifactCompatibilityError("production cache is missing scalar metadata: " + ", ".join(missing_scalars))
        tensor_values = {name: torch.from_numpy(np.array(archive[name], copy=True)) for name in _CACHE_TENSOR_FIELDS}
        array_values = {name: np.array(archive[name], copy=True) for name in _CACHE_ARRAY_FIELDS}
        optional = {
            name: torch.from_numpy(np.array(archive[name], copy=True)) if name in archive.files else None
            for name in ("trajectory_window_states", "trajectory_window_valid", "trajectory_window_depths")
        }
    cache = D0TrainingCache(
        **tensor_values,
        **array_values,
        **scalars,
        **optional,
    )
    _validate_loaded_cache(cache)
    fingerprints = dict(manifest.get("fingerprints", {}))
    if expected_fingerprints is not None:
        assert_fingerprints_match(
            fingerprints,
            expected_fingerprints,
            context="D0 cache",
            exact_keys=bool(exact_fingerprints),
        )
    expected_cache_fingerprint = str(manifest.get("cache_fingerprint", ""))
    actual_cache_fingerprint = fingerprint_d0_cache(cache)
    if verify_content and actual_cache_fingerprint != expected_cache_fingerprint:
        raise ArtifactCompatibilityError("D0 cache content fingerprint mismatch")
    return D0CacheArtifact(
        cache=cache,
        metadata=dict(manifest.get("metadata", {})),
        fingerprints=fingerprints,
        cache_fingerprint=actual_cache_fingerprint,
        schema_version=int(manifest["schema_version"]),
    )


def capture_rng_state(numpy_rng: np.random.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_global": copy.deepcopy(np.random.get_state()),
        "numpy_generator": copy.deepcopy(numpy_rng.bit_generator.state) if numpy_rng is not None else None,
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any], numpy_rng: np.random.Generator | None = None) -> None:
    required = {"python_random", "numpy_global", "torch_cpu", "torch_cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise ArtifactCompatibilityError("checkpoint RNG state is incomplete: " + ", ".join(missing))
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_global"])
    if numpy_rng is not None:
        generator_state = state.get("numpy_generator")
        if generator_state is None:
            raise ArtifactCompatibilityError("checkpoint has no NumPy Generator state")
        numpy_rng.bit_generator.state = copy.deepcopy(generator_state)
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu"))
    cuda_states = list(state.get("torch_cuda", []))
    if cuda_states:
        if not torch.cuda.is_available():
            raise ArtifactCompatibilityError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ArtifactCompatibilityError("checkpoint CUDA RNG device count differs from the current runtime")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in cuda_states]
        )


def build_training_checkpoint(
    *,
    model: nn.Module,
    ema_state: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    history: Sequence[Mapping[str, Any]],
    best_validation: Mapping[str, Any] | None,
    fingerprints: Mapping[str, Any],
    numpy_rng: np.random.Generator | None,
    dynamics_config: DirectFluxMNISTConfig | Mapping[str, Any] | None = None,
    d0_config: Experiment12D0Config | Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(step) < 0:
        raise ValueError("training checkpoint step must be non-negative")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": int(step),
        # ``state_dict`` tensors share storage with the live module/optimizer.
        # Deep-copy here so the returned payload remains an immutable snapshot
        # if the caller also uses it for best-checkpoint selection.
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "ema_state_dict": {name: value.detach().clone() for name, value in ema_state.items()},
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scaler_state_dict": copy.deepcopy(scaler.state_dict()) if scaler is not None else None,
        "history": copy.deepcopy([dict(row) for row in history]),
        "best_validation": None if best_validation is None else copy.deepcopy(dict(best_validation)),
        "fingerprints": copy.deepcopy(dict(fingerprints)),
        "rng_state": capture_rng_state(numpy_rng),
        "dynamics_config": _jsonable(dynamics_config),
        "d0_config": _jsonable(d0_config),
        "extra": copy.deepcopy(dict(extra or {})),
    }


def save_training_checkpoint(path: str | Path, **kwargs: Any) -> dict[str, Any]:
    payload = build_training_checkpoint(**kwargs)
    atomic_torch_save(path, payload)
    return payload


def _torch_load(path: str | Path, map_location: str | torch.device | None) -> Any:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        return torch.load(Path(path), map_location=map_location)


def load_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
    expected_fingerprints: Mapping[str, Any] | None = None,
    allow_legacy_report_only: bool = False,
    exact_fingerprints: bool = True,
) -> dict[str, Any]:
    payload = _torch_load(path, map_location)
    if not isinstance(payload, dict):
        raise ArtifactCompatibilityError("training checkpoint payload must be a mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA or int(payload.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        if not allow_legacy_report_only:
            raise LegacyArtifactError("legacy checkpoint cannot be used for strict resume or a required gate")
        warnings.warn(
            "legacy checkpoint loaded for report-only evaluation; it cannot satisfy a required gate or exact resume",
            RuntimeWarning,
            stacklevel=2,
        )
        result = dict(payload)
        result["_legacy_report_only"] = True
        return result
    required = {
        "step",
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "history",
        "best_validation",
        "fingerprints",
        "rng_state",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ArtifactCompatibilityError("training checkpoint is incomplete: " + ", ".join(missing))
    if expected_fingerprints is not None:
        assert_fingerprints_match(
            payload["fingerprints"],
            expected_fingerprints,
            context="training checkpoint",
            exact_keys=bool(exact_fingerprints),
        )
    return payload


def restore_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    numpy_rng: np.random.Generator | None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    if payload.get("_legacy_report_only") or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise LegacyArtifactError("legacy checkpoint cannot be restored for exact training resume")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None:
        scaler_state = payload.get("scaler_state_dict")
        if scaler_state is None:
            raise ArtifactCompatibilityError("checkpoint has no scaler state")
        scaler.load_state_dict(scaler_state)
    if restore_rng:
        restore_rng_state(payload["rng_state"], numpy_rng)
    return {
        "step": int(payload["step"]),
        "ema_state": {name: value.detach().clone() for name, value in payload["ema_state_dict"].items()},
        "history": [dict(row) for row in payload["history"]],
        "best_validation": copy.deepcopy(payload.get("best_validation")),
        "extra": copy.deepcopy(payload.get("extra", {})),
    }


def _metric_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("validation target/prediction arrays must have shape (slices, features)")
    finite = np.isfinite(target) & np.isfinite(prediction)
    finite_fraction = float(finite.mean()) if finite.size else float("nan")
    target_finite_fraction = float(np.isfinite(target).mean()) if target.size else float("nan")
    result: dict[str, Any] = {
        "slice_count": int(target.shape[0]),
        "feature_count": int(target.shape[1]) if target.ndim == 2 else 0,
        "finite_fraction": finite_fraction,
        "target_finite_fraction": target_finite_fraction,
    }
    if target.size == 0 or not bool(finite.all()):
        result.update(
            {
                "primary_mse": float("nan"),
                "zero_baseline_mse": float("nan"),
                "prediction_gain": float("nan"),
                "target_rms": float("nan"),
                "prediction_rms": float("nan"),
                "residual_rms": float("nan"),
                "target_prediction_covariance": float("nan"),
                "residual_covariance_trace": float("nan"),
                "residual_covariance_trace_per_feature": float("nan"),
            }
        )
        return result
    target64 = target.astype(np.float64, copy=False)
    prediction64 = prediction.astype(np.float64, copy=False)
    residual = prediction64 - target64
    per_slice_mse = np.mean(np.square(residual), axis=1)
    per_slice_zero = np.mean(np.square(target64), axis=1)
    primary_mse = float(np.mean(per_slice_mse))
    zero_mse = float(np.mean(per_slice_zero))
    target_centered = target64 - float(target64.mean())
    prediction_centered = prediction64 - float(prediction64.mean())
    covariance = float(np.mean(target_centered * prediction_centered))
    trace = float(np.var(residual, axis=0, ddof=0).sum())
    result.update(
        {
            "primary_mse": primary_mse,
            "zero_baseline_mse": zero_mse,
            "prediction_gain": 1.0 - primary_mse / zero_mse if zero_mse > 0.0 else float("nan"),
            "target_rms": float(math.sqrt(np.mean(np.square(target64)))),
            "prediction_rms": float(math.sqrt(np.mean(np.square(prediction64)))),
            "residual_rms": float(math.sqrt(primary_mse)),
            "target_prediction_covariance": covariance,
            "residual_covariance_trace": trace,
            "residual_covariance_trace_per_feature": trace / float(target.shape[1]),
        }
    )
    return result


@torch.no_grad()
def compute_validation_metrics(
    model: nn.Module,
    cache: D0TrainingCache,
    slice_indices: Sequence[int] | np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    *,
    device: torch.device | str,
    batch_size: int = 128,
    step: int = 0,
    weights: str = "raw",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the strict primary target overall and in five fixed tau/T bins."""

    indices_np = np.asarray(slice_indices, dtype=np.int64).reshape(-1)
    if indices_np.size == 0:
        raise ValueError("validation split is empty")
    if float(d0_config.physical_target_scale) <= 0.0 or not math.isfinite(float(d0_config.physical_target_scale)):
        raise ValueError("validation requires a frozen finite positive physical_target_scale")
    device_obj = torch.device(device)
    was_training = bool(model.training)
    model.eval()
    target_pieces: list[np.ndarray] = []
    prediction_pieces: list[np.ndarray] = []
    tau_pieces: list[np.ndarray] = []
    try:
        for offset in range(0, indices_np.size, max(1, int(batch_size))):
            idx_np = indices_np[offset : offset + max(1, int(batch_size))]
            idx = torch.as_tensor(idx_np, dtype=torch.long)
            batch = _cache_batch(cache, idx, device_obj)
            baseline = _direct_reverse_free_block_baseline_from_batch(batch, dynamics_config)
            target = project_edge_flux_torch(
                batch["physical_transfers"].to(dtype=baseline.dtype) - baseline,
                grid_size=int(dynamics_config.grid_size),
            ) / float(d0_config.physical_target_scale)
            prediction = project_edge_flux_torch(
                model(batch["tau"], batch["states"], batch["labels"], None),
                grid_size=int(dynamics_config.grid_size),
            )
            target_pieces.append(target.detach().float().cpu().reshape(target.shape[0], -1).numpy())
            prediction_pieces.append(prediction.detach().float().cpu().reshape(prediction.shape[0], -1).numpy())
            tau_pieces.append(cache.tau.index_select(0, idx).detach().float().cpu().numpy())
    finally:
        model.train(was_training)
    targets = np.concatenate(target_pieces, axis=0)
    predictions = np.concatenate(prediction_pieces, axis=0)
    tau = np.concatenate(tau_pieces, axis=0)
    common = {"step": int(step), "weights": str(weights)}
    overall = {**common, "tau_fraction_lo": 0.0, "tau_fraction_hi": 1.0, "bin": "overall", **_metric_summary(targets, predictions)}
    horizon = max(float(cache.horizon), 1e-30)
    fraction = np.clip(tau.astype(np.float64) / horizon, 0.0, 1.0)
    rows: list[dict[str, Any]] = []
    for bin_index, (lo, hi) in enumerate(TIME_BINS):
        selected = (fraction >= lo) & (fraction <= hi if bin_index == len(TIME_BINS) - 1 else fraction < hi)
        if bool(selected.any()):
            metrics = _metric_summary(targets[selected], predictions[selected])
        else:
            metrics = _metric_summary(np.empty((0, targets.shape[1])), np.empty((0, targets.shape[1])))
        rows.append(
            {
                **common,
                "bin": f"tau_bin{bin_index}",
                "bin_index": int(bin_index),
                "tau_fraction_lo": float(lo),
                "tau_fraction_hi": float(hi),
                **metrics,
            }
        )
    return overall, rows


def evaluate_raw_and_ema_validation(
    model: nn.Module,
    ema_state: Mapping[str, Tensor],
    cache: D0TrainingCache,
    slice_indices: Sequence[int] | np.ndarray,
    dynamics_config: DirectFluxMNISTConfig,
    d0_config: Experiment12D0Config,
    *,
    device: torch.device | str,
    batch_size: int = 128,
    step: int = 0,
) -> dict[str, Any]:
    raw_overall, raw_bins = compute_validation_metrics(
        model,
        cache,
        slice_indices,
        dynamics_config,
        d0_config,
        device=device,
        batch_size=batch_size,
        step=step,
        weights="raw",
    )
    with temporary_ema_weights(model, dict(ema_state)):
        ema_overall, ema_bins = compute_validation_metrics(
            model,
            cache,
            slice_indices,
            dynamics_config,
            d0_config,
            device=device,
            batch_size=batch_size,
            step=step,
            weights="ema",
        )
    return {
        "step": int(step),
        "raw": raw_overall,
        "ema": ema_overall,
        "time_bins": [*raw_bins, *ema_bins],
    }


def select_best_ema_checkpoint(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mse_key: str = "primary_mse",
    step_key: str = "step",
) -> dict[str, Any] | None:
    """Select minimum finite EMA MSE, resolving exact ties to earliest step."""

    eligible = []
    for candidate in candidates:
        value = float(candidate.get(mse_key, float("nan")))
        step = int(candidate.get(step_key, 0))
        if math.isfinite(value) and value >= 0.0:
            eligible.append((value, step, dict(candidate)))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1]))
    return eligible[0][2]


@dataclass(frozen=True)
class OneImageGateThresholds:
    cache_preflight_path_count: int = 8
    terminal_target_abs_corr_max: float = 0.10
    max_simplex_mass_error: float = 2e-6
    floor_correction_l1_per_path_substep: float = 1e-8
    renorm_correction_l1_per_path_substep: float = 1e-6
    raw_intervention_fraction: float = 0.005
    weighted_intervention_fraction: float = 0.0005
    reconstruction_mean_corr: float = 0.90
    reconstruction_mean_l1: float = 0.20
    reconstruction_good_corr: float = 0.85
    reconstruction_good_fraction: float = 0.80
    paired_corr_improvement: float = 0.20
    relative_l1_reduction: float = 0.25
    reconstruction_sample_count: int = 16

    def __post_init__(self) -> None:
        if int(self.cache_preflight_path_count) <= 0:
            raise ValueError("cache_preflight_path_count must be positive")
        if int(self.reconstruction_sample_count) <= 0:
            raise ValueError("reconstruction_sample_count must be positive")
        nonnegative = {
            "max_simplex_mass_error": self.max_simplex_mass_error,
            "floor_correction_l1_per_path_substep": self.floor_correction_l1_per_path_substep,
            "renorm_correction_l1_per_path_substep": self.renorm_correction_l1_per_path_substep,
            "reconstruction_mean_l1": self.reconstruction_mean_l1,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        unit_interval = {
            "terminal_target_abs_corr_max": self.terminal_target_abs_corr_max,
            "raw_intervention_fraction": self.raw_intervention_fraction,
            "weighted_intervention_fraction": self.weighted_intervention_fraction,
            "reconstruction_good_fraction": self.reconstruction_good_fraction,
            "relative_l1_reduction": self.relative_l1_reduction,
        }
        for name, value in unit_interval.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name, value in {
            "reconstruction_mean_corr": self.reconstruction_mean_corr,
            "reconstruction_good_corr": self.reconstruction_good_corr,
        }.items():
            if not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
        if not math.isfinite(float(self.paired_corr_improvement)) or not -2.0 <= float(self.paired_corr_improvement) <= 2.0:
            raise ValueError("paired_corr_improvement must be finite and in [-2, 2]")


def _lookup(metrics: Mapping[str, Any], *keys: str, default: Any = float("nan")) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return default


def _check(name: str, value: Any, operator: str, threshold: Any, passed: bool) -> tuple[str, dict[str, Any]]:
    return name, {
        "passed": int(bool(passed)),
        "value": _jsonable(value),
        "operator": operator,
        "threshold": _jsonable(threshold),
    }


def _finish_gate(name: str, checks: Sequence[tuple[str, dict[str, Any]]], claim_scope: str) -> dict[str, Any]:
    subchecks = dict(checks)
    passed = bool(subchecks) and all(bool(int(check["passed"])) for check in subchecks.values())
    return {
        "gate": name,
        "passed": int(passed),
        f"{name}_pass": int(passed),
        "subchecks": subchecks,
        "claim_scope": claim_scope,
    }


def evaluate_cache_gate(
    metrics: Mapping[str, Any],
    thresholds: OneImageGateThresholds = OneImageGateThresholds(),
) -> dict[str, Any]:
    build_mode = str(_lookup(metrics, "cache_build_mode", default=""))
    path_count = int(_lookup(metrics, "cache_paths", "cache_path_count", default=-1))
    stride = int(_lookup(metrics, "cache_stride_substeps", "stride_substeps", default=-1))
    scale = float(_lookup(metrics, "physical_target_scale", "cache_physical_target_scale"))
    finite_fraction = float(_lookup(metrics, "target_finite_fraction", "cache_target_finite_fraction"))
    direct_l1 = float(_lookup(metrics, "oracle_direct_l1", "direct_oracle_replay_l1"))
    reference_l1 = float(_lookup(metrics, "oracle_positive_free_only_l1", "positive_reference_replay_l1"))
    terminal_corr = float(_lookup(metrics, "terminal_target_abs_corr_mean", "mean_abs_terminal_target_correlation"))
    nonfinite = int(_lookup(metrics, "nonfinite_edges", default=-1))
    floor_touches = int(_lookup(metrics, "floor_touched_pixels", "cache_floor_touched_pixels", default=-1))
    simplex = float(_lookup(metrics, "max_simplex_mass_error"))
    floor_correction = float(_lookup(metrics, "floor_correction_l1_per_path_substep", "cache_floor_correction_l1_per_path_substep"))
    renorm_correction = float(_lookup(metrics, "renorm_correction_l1_per_path_substep", "cache_renorm_correction_l1_per_path_substep"))
    raw = float(_lookup(metrics, "raw_limited_fraction", "cache_raw_limited_fraction"))
    mobility = float(_lookup(metrics, "mobility_weighted_limited_fraction", "cache_mobility_weighted_limited_fraction"))
    noise = float(_lookup(metrics, "noise_energy_weighted_limited_fraction", "cache_noise_energy_weighted_limited_fraction"))
    checks = [
        _check(
            "preflight_path_count",
            path_count,
            "==",
            thresholds.cache_preflight_path_count,
            path_count == thresholds.cache_preflight_path_count,
        ),
        _check("exact_substep_cache", build_mode in {"substep", "exact-substep", "exact"}, "in", ["substep"], build_mode in {"substep", "exact-substep", "exact"}),
        _check("stride_one", stride, "==", 1, stride == 1),
        _check("finite_positive_target_scale", scale, ">", 0.0, math.isfinite(scale) and scale > 0.0),
        _check("target_finite_fraction", finite_fraction, "==", 1.0, math.isfinite(finite_fraction) and finite_fraction == 1.0),
        _check("direct_oracle_beats_reference", {"direct": direct_l1, "reference": reference_l1}, "0 <= direct < reference", None, math.isfinite(direct_l1) and math.isfinite(reference_l1) and 0.0 <= direct_l1 < reference_l1),
        _check("terminal_target_abs_corr", terminal_corr, "in", [0.0, thresholds.terminal_target_abs_corr_max], math.isfinite(terminal_corr) and 0.0 <= terminal_corr <= thresholds.terminal_target_abs_corr_max),
        _check("nonfinite_edges", nonfinite, "==", 0, nonfinite == 0),
        _check("floor_touches", floor_touches, "==", 0, floor_touches == 0),
        _check("simplex_error", simplex, "in", [0.0, thresholds.max_simplex_mass_error], math.isfinite(simplex) and 0.0 <= simplex <= thresholds.max_simplex_mass_error),
        _check("floor_correction", floor_correction, "in", [0.0, thresholds.floor_correction_l1_per_path_substep], math.isfinite(floor_correction) and 0.0 <= floor_correction <= thresholds.floor_correction_l1_per_path_substep),
        _check("renorm_correction", renorm_correction, "in", [0.0, thresholds.renorm_correction_l1_per_path_substep], math.isfinite(renorm_correction) and 0.0 <= renorm_correction <= thresholds.renorm_correction_l1_per_path_substep),
        _check("raw_intervention", raw, "in", [0.0, thresholds.raw_intervention_fraction], math.isfinite(raw) and 0.0 <= raw <= thresholds.raw_intervention_fraction),
        _check("mobility_weighted_intervention", mobility, "in", [0.0, thresholds.weighted_intervention_fraction], math.isfinite(mobility) and 0.0 <= mobility <= thresholds.weighted_intervention_fraction),
        _check("noise_weighted_intervention", noise, "in", [0.0, thresholds.weighted_intervention_fraction], math.isfinite(noise) and 0.0 <= noise <= thresholds.weighted_intervention_fraction),
    ]
    return _finish_gate("cache", checks, "exact one-image direct-Doob cache correctness")


def evaluate_optimization_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    overall_gain = float(_lookup(metrics, "selected_ema_validation_gain", "prediction_gain"))
    data_end_gain = float(_lookup(metrics, "selected_ema_data_end_gain", "data_end_prediction_gain"))
    data_end_count = int(_lookup(metrics, "selected_ema_data_end_count", "data_end_slice_count", default=0))
    checks = [
        _check("overall_ema_gain", overall_gain, "in", [0.0, 1.0], math.isfinite(overall_gain) and 0.0 < overall_gain <= 1.0),
        _check("data_end_bin_populated", data_end_count, ">", 0, data_end_count > 0),
        _check("data_end_ema_gain", data_end_gain, "in", [0.0, 1.0], data_end_count > 0 and math.isfinite(data_end_gain) and 0.0 < data_end_gain <= 1.0),
    ]
    return _finish_gate("optimization", checks, "held-out forward-path direct-residual prediction")


def _arm_numerical_checks(
    arm_name: str,
    metrics: Mapping[str, Any],
    thresholds: OneImageGateThresholds,
) -> list[tuple[str, dict[str, Any]]]:
    nonfinite = int(_lookup(metrics, "nonfinite_edges", default=-1))
    floor_touches = int(_lookup(metrics, "floor_touched_pixels", default=-1))
    simplex = float(_lookup(metrics, "max_simplex_mass_error"))
    floor_correction = float(_lookup(metrics, "floor_correction_l1_per_path_substep"))
    renorm_correction = float(_lookup(metrics, "renorm_correction_l1_per_path_substep"))
    raw = float(_lookup(metrics, "raw_limited_fraction", "limiter_fraction"))
    mobility = float(_lookup(metrics, "mobility_weighted_limited_fraction", "mobility_weighted_limiter_fraction"))
    noise = float(_lookup(metrics, "noise_energy_weighted_limited_fraction", "noise_energy_weighted_limiter_fraction"))
    prefix = f"{arm_name}_"
    return [
        _check(prefix + "nonfinite_edges", nonfinite, "==", 0, nonfinite == 0),
        _check(prefix + "floor_touches", floor_touches, "==", 0, floor_touches == 0),
        _check(prefix + "simplex_error", simplex, "in", [0.0, thresholds.max_simplex_mass_error], math.isfinite(simplex) and 0.0 <= simplex <= thresholds.max_simplex_mass_error),
        _check(prefix + "floor_correction", floor_correction, "in", [0.0, thresholds.floor_correction_l1_per_path_substep], math.isfinite(floor_correction) and 0.0 <= floor_correction <= thresholds.floor_correction_l1_per_path_substep),
        _check(prefix + "renorm_correction", renorm_correction, "in", [0.0, thresholds.renorm_correction_l1_per_path_substep], math.isfinite(renorm_correction) and 0.0 <= renorm_correction <= thresholds.renorm_correction_l1_per_path_substep),
        _check(prefix + "raw_intervention", raw, "in", [0.0, thresholds.raw_intervention_fraction], math.isfinite(raw) and 0.0 <= raw <= thresholds.raw_intervention_fraction),
        _check(prefix + "mobility_weighted_intervention", mobility, "in", [0.0, thresholds.weighted_intervention_fraction], math.isfinite(mobility) and 0.0 <= mobility <= thresholds.weighted_intervention_fraction),
        _check(prefix + "noise_weighted_intervention", noise, "in", [0.0, thresholds.weighted_intervention_fraction], math.isfinite(noise) and 0.0 <= noise <= thresholds.weighted_intervention_fraction),
    ]


def evaluate_reconstruction_gate(
    metrics: Mapping[str, Any],
    thresholds: OneImageGateThresholds = OneImageGateThresholds(),
) -> dict[str, Any]:
    mean_corr = float(_lookup(metrics, "strength_1_mean_corr", "mean_corr_strength_1"))
    mean_l1 = float(_lookup(metrics, "strength_1_mean_l1", "mean_l1_strength_1"))
    good_fraction = float(_lookup(metrics, "strength_1_good_corr_fraction", "corr_at_least_threshold_fraction"))
    corr_improvement = float(_lookup(metrics, "paired_mean_corr_improvement", "mean_paired_corr_delta"))
    relative_l1 = float(_lookup(metrics, "relative_l1_reduction"))
    complete = int(_lookup(metrics, "complete", default=0))
    sample_count = int(
        _lookup(metrics, "sample_count", "paired_sample_count", "num_samples", default=-1)
    )
    checks = [
        _check("sampling_complete", complete, "==", 1, complete == 1),
        _check(
            "paired_sample_count",
            sample_count,
            "==",
            thresholds.reconstruction_sample_count,
            sample_count == thresholds.reconstruction_sample_count,
        ),
        _check("strength_1_mean_corr", mean_corr, "in", [thresholds.reconstruction_mean_corr, 1.0], math.isfinite(mean_corr) and thresholds.reconstruction_mean_corr <= mean_corr <= 1.0),
        _check("strength_1_mean_l1", mean_l1, "in", [0.0, thresholds.reconstruction_mean_l1], math.isfinite(mean_l1) and 0.0 <= mean_l1 <= thresholds.reconstruction_mean_l1),
        _check(
            "strength_1_good_corr_fraction",
            {"fraction": good_fraction, "corr_cutoff": thresholds.reconstruction_good_corr},
            ">=",
            thresholds.reconstruction_good_fraction,
            math.isfinite(good_fraction) and thresholds.reconstruction_good_fraction <= good_fraction <= 1.0,
        ),
        _check("paired_mean_corr_improvement", corr_improvement, "in", [thresholds.paired_corr_improvement, 2.0], math.isfinite(corr_improvement) and thresholds.paired_corr_improvement <= corr_improvement <= 2.0),
        _check("relative_l1_reduction", relative_l1, "in", [thresholds.relative_l1_reduction, 1.0], math.isfinite(relative_l1) and thresholds.relative_l1_reduction <= relative_l1 <= 1.0),
    ]
    for arm_name in ("strength_0", "strength_1"):
        arm_metrics = metrics.get(arm_name, {})
        if not isinstance(arm_metrics, Mapping):
            arm_metrics = {}
        checks.extend(_arm_numerical_checks(arm_name, arm_metrics, thresholds))
    result = _finish_gate("reconstruction", checks, "paired stochastic one-image reconstruction for the frozen fixed-grid kernel")
    result["learned_noise_ratio_gated"] = 0
    return result


def evaluate_overfit_gates(
    *,
    cache_metrics: Mapping[str, Any] | None,
    optimization_metrics: Mapping[str, Any] | None,
    reconstruction_metrics: Mapping[str, Any] | None,
    require_gate: str = "none",
    thresholds: OneImageGateThresholds = OneImageGateThresholds(),
) -> dict[str, Any]:
    required = str(require_gate).strip().lower()
    if required not in {"none", "cache", "optimization", "reconstruction"}:
        raise ValueError("require_gate must be none, cache, optimization, or reconstruction")
    cache_gate = evaluate_cache_gate(cache_metrics or {}, thresholds)
    optimization_gate = evaluate_optimization_gate(optimization_metrics or {})
    reconstruction_gate = evaluate_reconstruction_gate(reconstruction_metrics or {}, thresholds)
    cumulative = {
        "cache": bool(cache_gate["passed"]),
        "optimization": bool(cache_gate["passed"] and optimization_gate["passed"]),
        "reconstruction": bool(cache_gate["passed"] and optimization_gate["passed"] and reconstruction_gate["passed"]),
    }
    required_pass = True if required == "none" else cumulative[required]
    return {
        "schema": "experiment12-d0-one-image-gate",
        "schema_version": 1,
        "required_gate": required,
        "required_gate_pass": int(required_pass),
        "cache": cache_gate,
        "optimization": optimization_gate,
        "reconstruction": reconstruction_gate,
        "cumulative_pass": {key: int(value) for key, value in cumulative.items()},
        "thresholds": asdict(thresholds),
        "claim_scope": "one-image reproduction for one frozen fixed-grid temporal kernel",
        "excluded_claims": [
            "spatial Dirichlet-Ferguson convergence",
            "held-out digit generalization",
            "full-data sample quality",
        ],
    }


def terminal_target_abs_correlation(cache: D0TrainingCache, path_ids: Sequence[int] | np.ndarray | None = None) -> dict[str, float | int]:
    """Mean/max absolute Pearson correlation between terminals and mixed starts."""

    if path_ids is None:
        selected = np.arange(int(cache.terminal_states.shape[0]), dtype=np.int64)
    else:
        selected = np.asarray(path_ids, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        return {"terminal_target_path_count": 0, "terminal_target_abs_corr_mean": float("nan"), "terminal_target_abs_corr_max": float("nan")}
    path_index = cache.path_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    correlations: list[float] = []
    for path_id in selected.tolist():
        rows = np.flatnonzero(path_index == int(path_id))
        if rows.size == 0:
            raise ValueError(f"path {path_id} has no cache slices")
        target = cache.start_images[int(rows[0])].detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        terminal = np.asarray(cache.terminal_states[int(path_id)], dtype=np.float64).reshape(-1)
        target_centered = target - target.mean()
        terminal_centered = terminal - terminal.mean()
        denom = float(np.linalg.norm(target_centered) * np.linalg.norm(terminal_centered))
        corr = float(np.dot(target_centered, terminal_centered) / denom) if denom > 0.0 else 0.0
        correlations.append(abs(corr))
    return {
        "terminal_target_path_count": int(selected.size),
        "terminal_target_abs_corr_mean": float(np.mean(correlations)),
        "terminal_target_abs_corr_max": float(np.max(correlations)),
    }
