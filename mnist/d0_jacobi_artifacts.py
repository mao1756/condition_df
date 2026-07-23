"""Small artifact/runtime helpers for the controls-only Jacobi workflow.

This module intentionally has no dependency on the Experiment 12 trainers or
reverse samplers.  Keeping these generic helpers here lets import-isolation
tests distinguish the exact-kernel feasibility study from physical training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


class ArtifactCompatibilityError(ValueError):
    """Raised when immutable evidence cannot satisfy a strict binding."""


def _jsonable(value: Any) -> Any:
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
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def config_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
    records = [
        {"path": path.as_posix(), "sha256": file_fingerprint(path)}
        for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix())
    ]
    return config_fingerprint(records)


def _atomic_replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
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
    normalized = [dict(row) for row in rows]

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            if normalized:
                fieldnames: list[str] = []
                for row in normalized:
                    for key in row:
                        if key not in fieldnames:
                            fieldnames.append(key)
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    {key: _jsonable(item) for key, item in row.items()}
                    for row in normalized
                )
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace(Path(path), write)


def configure_exact_torch_backend(device: torch.device | str | None = None) -> dict[str, Any]:
    device_obj = None if device is None else torch.device(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("exact CUDA run requires a deterministic CUBLAS workspace")
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
    torch.backends.mkldnn.enabled = False
    record: dict[str, Any] = {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }
    if device_obj is not None and device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = device_obj.index if device_obj.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("could not fingerprint the NVIDIA driver") from exc
        versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not versions:
            raise RuntimeError("nvidia-smi returned no driver version")
        record.update({
            "cuda_device_index": int(index),
            "cuda_device_name": str(properties.name),
            "cuda_device_uuid": str(getattr(properties, "uuid", "")),
            "cuda_compute_capability": [int(properties.major), int(properties.minor)],
            "cuda_total_memory": int(properties.total_memory),
            "nvidia_driver_versions": versions,
        })
    return record
