#!/usr/bin/env python3
"""Verify an extracted directional-continuation bundle byte for byte."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


SCHEMA = "jacobi-rb-quartile-directional-runpod-bundle"
SCHEMA_VERSION = 1
MANIFEST_NAME = "RUNPOD_BUNDLE_MANIFEST.json"
CHECKSUMS_NAME = "SHA256SUMS"
CHUNK_BYTES = 8 * 1024 * 1024
EXPECTED_RUNTIME = {
    "python": "3.14.4",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "torch_cuda": "12.8",
    "numpy": "2.4.4",
    "python-flint": "0.9.0",
    "compute_capability": (12, 0),
}


class VerificationError(RuntimeError):
    """The extracted handoff or selected runtime changed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid bundle manifest: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError("bundle manifest must be a JSON object")
    return value


def _parse_checksums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise VerificationError(f"could not read {path}") from exc
    for line in lines:
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise VerificationError("malformed SHA256SUMS row") from exc
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in rows
        ):
            raise VerificationError("invalid or duplicate SHA256SUMS row")
        rows[relative] = digest
    return rows


def _excluded(path: Path, manifest: dict[str, Any]) -> bool:
    excluded_dirs = set(str(value) for value in manifest["excluded_directory_names"])
    suffixes = tuple(str(value) for value in manifest["excluded_suffixes"])
    return any(part in excluded_dirs for part in path.parts) or str(path).endswith(
        suffixes
    )


def _selected_actual(root: Path, manifest: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for relative in manifest["source_files"]:
        path = root / str(relative)
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
    for relative in (
        *manifest["source_directories"],
        *manifest["run_directories"],
    ):
        directory = root / str(relative)
        if not directory.is_dir():
            raise VerificationError(f"selected directory is missing: {relative}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise VerificationError(f"bundle contains a symlink: {path}")
            if path.is_file():
                candidate = path.relative_to(root)
                if not _excluded(candidate, manifest):
                    result.add(candidate.as_posix())
    return result


def _artifact_count(record: dict[str, Any]) -> int:
    for key in ("artifact_count", "record_count", "registered_path_count"):
        if key in record:
            return int(record[key])
    artifacts = record.get("artifacts")
    if isinstance(artifacts, list):
        return len(artifacts)
    raise VerificationError("artifact count is missing")


def _verify_run_commitments(root: Path, manifest: dict[str, Any]) -> None:
    for commitment in manifest["run_commitments"]:
        run_root = root / str(commitment["relative_path"])
        registry_path = run_root / "artifact_registry.json"
        manifest_path = run_root / "run_manifest.json"
        status_path = run_root / "run_status.json"
        registry = _load_json(registry_path)
        run_manifest = _load_json(manifest_path)
        status = _load_json(status_path)
        observed = {
            "artifact_count": _artifact_count(registry),
            "registry_semantic_sha256": str(registry.get("semantic_sha256", "")),
            "registry_file_sha256": _sha256_file(registry_path),
            "manifest_file_sha256": _sha256_file(manifest_path),
            "status_file_sha256": _sha256_file(status_path),
            "source_fingerprint": str(run_manifest.get("source_fingerprint", "")),
            "scientific_config_sha256": str(
                run_manifest.get("scientific_config_sha256", "")
            ),
            "decision": str(status.get("decision", "")),
            "state": str(status.get("state", "")),
        }
        expected = {key: commitment[key] for key in observed}
        if observed != expected:
            raise VerificationError(
                f"immutable run commitment changed: {commitment['basename']}"
            )


def verify_files(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    checksums_path = root / CHECKSUMS_NAME
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != SCHEMA
        or int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
        or manifest.get("compression") != "ZIP_STORED"
        or int(manifest.get("zip64", 0)) != 1
    ):
        raise VerificationError("bundle schema/profile changed")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != int(
        manifest.get("file_count", -1)
    ):
        raise VerificationError("bundle file table changed")
    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise VerificationError("bundle file row is malformed")
        relative = str(record.get("path", ""))
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or relative in expected
        ):
            raise VerificationError("unsafe or duplicate bundle path")
        expected[relative] = record
    if _selected_actual(root, manifest) != set(expected):
        raise VerificationError("bundle selected path set changed")
    checksum_rows = _parse_checksums(checksums_path)
    expected_checksum_paths = set(expected) | {MANIFEST_NAME}
    if set(checksum_rows) != expected_checksum_paths:
        raise VerificationError("SHA256SUMS path set changed")
    payload_bytes = 0
    for index, relative in enumerate(sorted(expected), start=1):
        path = root / relative
        record = expected[relative]
        if not path.is_file() or path.is_symlink():
            raise VerificationError(f"bundle payload is missing: {relative}")
        size = path.stat().st_size
        digest = _sha256_file(path)
        if size != int(record["size"]) or digest != str(record["sha256"]):
            raise VerificationError(f"bundle payload changed: {relative}")
        if checksum_rows[relative] != digest:
            raise VerificationError(f"SHA256SUMS mismatch: {relative}")
        payload_bytes += size
        if index % 250 == 0 or index == len(expected):
            print(f"RunPod verify {index}/{len(expected)} files", flush=True)
    manifest_digest = _sha256_file(manifest_path)
    if checksum_rows[MANIFEST_NAME] != manifest_digest:
        raise VerificationError("bundle manifest checksum changed")
    if payload_bytes != int(manifest.get("payload_bytes", -1)):
        raise VerificationError("bundle payload byte count changed")
    _verify_run_commitments(root, manifest)
    return {
        "schema": SCHEMA + "-verification",
        "passed": 1,
        "root": str(root),
        "file_count": len(expected),
        "payload_bytes": payload_bytes,
        "manifest_sha256": manifest_digest,
        "run_commitments_verified": len(manifest["run_commitments"]),
    }


def verify_runtime() -> dict[str, Any]:
    import numpy as np
    import torch

    python_version = ".".join(str(value) for value in sys.version_info[:3])
    if python_version != EXPECTED_RUNTIME["python"]:
        raise VerificationError(f"Python version changed: {python_version}")
    if torch.__version__ != EXPECTED_RUNTIME["torch"]:
        raise VerificationError(f"Torch version changed: {torch.__version__}")
    torchvision_version = importlib.metadata.version("torchvision")
    if torchvision_version != EXPECTED_RUNTIME["torchvision"]:
        raise VerificationError(
            f"Torchvision version changed: {torchvision_version}"
        )
    if torch.version.cuda != EXPECTED_RUNTIME["torch_cuda"]:
        raise VerificationError(f"Torch CUDA runtime changed: {torch.version.cuda}")
    if np.__version__ != EXPECTED_RUNTIME["numpy"]:
        raise VerificationError(f"NumPy version changed: {np.__version__}")
    flint_version = importlib.metadata.version("python-flint")
    if flint_version != EXPECTED_RUNTIME["python-flint"]:
        raise VerificationError(f"python-flint version changed: {flint_version}")
    if not torch.cuda.is_available():
        raise VerificationError("CUDA is unavailable")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if capability != EXPECTED_RUNTIME["compute_capability"]:
        raise VerificationError(f"compute capability must be 12.0, got {capability}")
    if torch.are_deterministic_algorithms_enabled():
        raise VerificationError("deterministic-algorithm mode differs from predecessor")
    if bool(torch.backends.cuda.matmul.allow_tf32):
        raise VerificationError("CUDA matmul TF32 must remain disabled")
    if not bool(torch.backends.cudnn.allow_tf32):
        raise VerificationError("cuDNN TF32 predecessor default changed")
    if bool(torch.backends.cudnn.deterministic):
        raise VerificationError("cuDNN deterministic mode differs from predecessor")
    if bool(torch.backends.cudnn.benchmark):
        raise VerificationError("cuDNN benchmark mode differs from predecessor")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        raise VerificationError("CUBLAS_WORKSPACE_CONFIG must remain unset")
    properties = torch.cuda.get_device_properties(0)
    # NVIDIA's marketed 8 GB predecessor reports about 7.96 GiB through
    # PyTorch, so use the corresponding decimal-byte floor.
    if int(properties.total_memory) < 8_000_000_000:
        raise VerificationError("selected GPU has less than 8 GB memory")
    return {
        "schema": SCHEMA + "-runtime-verification",
        "passed": 1,
        "python": python_version,
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "python-flint": flint_version,
        "device_name": properties.name,
        "compute_capability": list(capability),
        "total_memory": int(properties.total_memory),
        "deterministic_algorithms": 0,
        "cuda_matmul_allow_tf32": 0,
        "cudnn_allow_tf32": 1,
        "cudnn_deterministic": 0,
        "cudnn_benchmark": 0,
        "cublas_workspace_config": None,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Extracted condition_df directory",
    )
    parser.add_argument("--check-runtime", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = {"bundle": verify_files(args.root)}
        if args.check_runtime:
            result["runtime"] = verify_runtime()
    except VerificationError as exc:
        print(f"RunPod verification error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
