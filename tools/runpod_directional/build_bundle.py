#!/usr/bin/env python3
"""Build the byte-preserving RunPod directional-continuation handoff.

The archive deliberately contains only the source closure and the three run
trees needed by the portable continuation.  It does not copy the complete
historical ``runs`` directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable
from zipfile import ZIP_STORED, ZipFile, ZipInfo


SCHEMA = "jacobi-rb-quartile-directional-runpod-bundle"
SCHEMA_VERSION = 1
ARCHIVE_ROOT = "condition_df"
CHUNK_BYTES = 8 * 1024 * 1024

SOURCE_DIRS = (
    "mnist",
    "tools/runpod_directional",
)
SOURCE_FILES = (
    "__init__.py",
    "AGENT.md",
    "pyproject.toml",
    "requirements-jacobi-certification.txt",
    "requirements-runpod-directional.txt",
    "docs/experiment12_d0_patch_plan.md",
    "docs/jacobi_rb_quartile_directional_adjudication.md",
    "docs/runpod_directional_continuation.md",
)
RUN_DIRS = (
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist/"
    "20260807-132351_production-exact-quartile-specialist",
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication/"
    "20260807-005609_production-v3-time-local-adjudication",
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication/"
    "20260808-203454_production-read-only-quartile-directional-adjudication-bootstrap-fix",
)
EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".test",
    ".tmp",
    ".venv",
    ".venv-runpod",
    "__pycache__",
}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".tmp")
MANIFEST_NAME = "RUNPOD_BUNDLE_MANIFEST.json"
CHECKSUMS_NAME = "SHA256SUMS"


class BundleError(RuntimeError):
    """The selected handoff is missing, mutable, or internally inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"invalid JSON commitment: {path}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"JSON commitment is not an object: {path}")
    return value


def _artifact_count(record: dict[str, Any]) -> int:
    for key in ("artifact_count", "record_count", "registered_path_count"):
        if key in record:
            return int(record[key])
    artifacts = record.get("artifacts")
    if isinstance(artifacts, list):
        return len(artifacts)
    raise BundleError("artifact registry does not expose its record count")


def _run_commitment(repo_root: Path, relative: str) -> dict[str, Any]:
    root = repo_root / relative
    registry_path = root / "artifact_registry.json"
    manifest_path = root / "run_manifest.json"
    status_path = root / "run_status.json"
    registry = _json(registry_path)
    manifest = _json(manifest_path)
    status = _json(status_path)
    return {
        "relative_path": relative,
        "basename": root.name,
        "artifact_count": _artifact_count(registry),
        "registry_semantic_sha256": str(registry.get("semantic_sha256", "")),
        "registry_file_sha256": _sha256_file(registry_path),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "status_file_sha256": _sha256_file(status_path),
        "source_fingerprint": str(manifest.get("source_fingerprint", "")),
        "scientific_config_sha256": str(
            manifest.get("scientific_config_sha256", "")
        ),
        "decision": str(status.get("decision", "")),
        "state": str(status.get("state", "")),
    }


def _git_record(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return ""
        return result.stdout.strip()

    status = run("status", "--short")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": int(bool(status)),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _excluded(relative: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in relative.parts) or str(
        relative
    ).endswith(EXCLUDED_SUFFIXES)


def selected_files(repo_root: Path) -> tuple[tuple[str, Path], ...]:
    """Return the fixed, sorted payload without following links."""

    selected: dict[str, Path] = {}
    for name in SOURCE_FILES:
        path = repo_root / name
        if not path.is_file():
            raise BundleError(f"required source/document is missing: {name}")
        selected[Path(name).as_posix()] = path
    for name in (*SOURCE_DIRS, *RUN_DIRS):
        root = repo_root / name
        if not root.is_dir():
            raise BundleError(f"required bundle directory is missing: {name}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise BundleError(f"bundle payload may not contain symlinks: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root)
            if _excluded(relative):
                continue
            selected[relative.as_posix()] = path
    return tuple((name, selected[name]) for name in sorted(selected))


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(f"{ARCHIVE_ROOT}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    mode = 0o755 if name.endswith(".sh") else 0o644
    info.external_attr = (0o100000 | mode) << 16
    info.create_system = 3
    return info


def _write_payload(
    archive: ZipFile, relative: str, source: Path
) -> dict[str, Any]:
    before = source.stat()
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as reader, archive.open(
        _zip_info(relative), "w", force_zip64=True
    ) as writer:
        while block := reader.read(CHUNK_BYTES):
            writer.write(block)
            digest.update(block)
            byte_count += len(block)
    after = source.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or byte_count != before.st_size
    ):
        raise BundleError(f"payload changed while it was archived: {source}")
    return {
        "path": relative,
        "size": byte_count,
        "sha256": digest.hexdigest(),
    }


def _write_bytes(archive: ZipFile, relative: str, payload: bytes) -> None:
    with archive.open(_zip_info(relative), "w", force_zip64=True) as writer:
        writer.write(payload)


def build_bundle(repo_root: Path, output: Path, *, force: bool = False) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    if output.exists() and not force:
        raise BundleError(f"output already exists (use --force): {output}")
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if sidecar.exists() and not force:
        raise BundleError(f"checksum sidecar already exists (use --force): {sidecar}")
    files = selected_files(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    records: list[dict[str, Any]] = []
    try:
        with ZipFile(partial, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            for index, (relative, source) in enumerate(files, start=1):
                records.append(_write_payload(archive, relative, source))
                if index % 250 == 0 or index == len(files):
                    print(
                        f"RunPod bundle {index}/{len(files)} files "
                        f"({sum(row['size'] for row in records) / 1024**3:.3f} GiB)",
                        flush=True,
                    )
            metadata = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "archive_root": ARCHIVE_ROOT,
                "compression": "ZIP_STORED",
                "zip64": 1,
                "file_count": len(records),
                "payload_bytes": sum(int(row["size"]) for row in records),
                "files": records,
                "source_directories": list(SOURCE_DIRS),
                "source_files": list(SOURCE_FILES),
                "run_directories": list(RUN_DIRS),
                "excluded_directory_names": sorted(EXCLUDED_DIR_NAMES),
                "excluded_suffixes": list(EXCLUDED_SUFFIXES),
                "git": _git_record(repo_root),
                "run_commitments": [
                    _run_commitment(repo_root, relative) for relative in RUN_DIRS
                ],
            }
            manifest_bytes = (
                json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            ).encode("utf-8")
            _write_bytes(archive, MANIFEST_NAME, manifest_bytes)
            checksum_rows = [
                f"{row['sha256']}  {row['path']}" for row in records
            ]
            checksum_rows.append(
                f"{hashlib.sha256(manifest_bytes).hexdigest()}  {MANIFEST_NAME}"
            )
            checksums_bytes = ("\n".join(checksum_rows) + "\n").encode("utf-8")
            _write_bytes(archive, CHECKSUMS_NAME, checksums_bytes)
        if output.exists():
            output.unlink()
        partial.replace(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    archive_sha256 = _sha256_file(output)
    sidecar.write_text(
        f"{archive_sha256}  {output.name}\n", encoding="utf-8", newline="\n"
    )
    result = {
        "archive": str(output),
        "archive_sha256": archive_sha256,
        "checksum_sidecar": str(sidecar),
        "file_count": len(records),
        "payload_bytes": sum(int(row["size"]) for row in records),
        "archive_bytes": output.stat().st_size,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "handoff/jacobi_directional_runpod_20260808.zip",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        files = selected_files(args.repo_root.resolve())
        if args.list_only:
            total = sum(path.stat().st_size for _, path in files)
            print(json.dumps({"file_count": len(files), "payload_bytes": total}))
            return 0
        build_bundle(args.repo_root, args.output, force=bool(args.force))
    except BundleError as exc:
        print(f"RunPod bundle error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
