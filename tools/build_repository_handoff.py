#!/usr/bin/env python3
"""Build and verify a source-complete repository handoff.

The handoff captures the current working tree, including untracked files, while
excluding local environments, datasets, experiment outputs, prior handoffs, Git
internals, and transient caches.  It does not depend on the repository being
clean or on every source file being tracked by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo


SCHEMA = "condition-df-repository-handoff"
SCHEMA_VERSION = 1
ARCHIVE_ROOT = "condition_df"
MANIFEST_NAME = "REPOSITORY_HANDOFF_MANIFEST.json"
CHECKSUMS_NAME = "SHA256SUMS.txt"
CHUNK_BYTES = 8 * 1024 * 1024

# These are workspace-level generated data, not source.  A nested directory
# with one of these names remains eligible; only the repository-root directory
# is excluded.
EXCLUDED_TOP_LEVEL_DIRS = {
    ".git",
    ".tmp",
    ".venv",
    ".vscode",
    "artifacts",
    "data",
    "handoff",
    "logs",
    "mnist_data",
    "outputs",
    "runpod_runtime",
    "runs",
    "test-output",
    "tuning_runs",
}
EXCLUDED_TOP_LEVEL_PREFIXES = (".venv-",)
EXCLUDED_TRANSIENT_DIR_NAMES = {
    ".cache",
    ".git",
    ".hypothesis",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".test",
    ".tmp",
    ".tox",
    "__pycache__",
}
EXCLUDED_TRANSIENT_DIR_PREFIXES = (".pytest-", ".test-", ".tmp-")
EXCLUDED_FILE_SUFFIXES = (
    ".aux",
    ".bak",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".partial",
    ".pyc",
    ".pyo",
    ".swo",
    ".swp",
    ".synctex.gz",
    ".temp",
    ".tmp",
    ".toc",
    "~",
)
EXCLUDED_FILENAMES = {".DS_Store", "Thumbs.db"}
EXCLUDED_TOP_LEVEL_FILES = {
    "docs.zip",
    "mnist.202606180244.zip",
    "mnist_cp_samples.png",
}


class HandoffError(RuntimeError):
    """The requested repository handoff is unsafe or internally inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _sha256_member(archive: ZipFile, info: ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _run_git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _git_record(repo_root: Path) -> dict[str, Any]:
    status = _run_git(repo_root, "status", "--short")
    return {
        "head": _run_git(repo_root, "rev-parse", "HEAD"),
        "branch": _run_git(repo_root, "branch", "--show-current"),
        "dirty": int(bool(status)),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _policy_record() -> dict[str, list[str]]:
    return {
        "top_level_directories": sorted(EXCLUDED_TOP_LEVEL_DIRS),
        "top_level_prefixes": list(EXCLUDED_TOP_LEVEL_PREFIXES),
        "transient_directory_names": sorted(EXCLUDED_TRANSIENT_DIR_NAMES),
        "transient_directory_prefixes": list(EXCLUDED_TRANSIENT_DIR_PREFIXES),
        "file_suffixes": list(EXCLUDED_FILE_SUFFIXES),
        "filenames": sorted(EXCLUDED_FILENAMES),
        "top_level_files": sorted(EXCLUDED_TOP_LEVEL_FILES),
    }


def _excluded_directory(relative: Path) -> bool:
    lowered = relative.name.lower()
    if lowered in {name.lower() for name in EXCLUDED_TRANSIENT_DIR_NAMES}:
        return True
    if any(
        lowered.startswith(prefix.lower()) for prefix in EXCLUDED_TRANSIENT_DIR_PREFIXES
    ):
        return True
    if len(relative.parts) != 1:
        return False
    if lowered in {name.lower() for name in EXCLUDED_TOP_LEVEL_DIRS}:
        return True
    return any(lowered.startswith(prefix.lower()) for prefix in EXCLUDED_TOP_LEVEL_PREFIXES)


def _excluded_file(relative: Path) -> bool:
    if relative.name in EXCLUDED_FILENAMES:
        return True
    if len(relative.parts) == 1 and relative.name in EXCLUDED_TOP_LEVEL_FILES:
        return True
    return relative.name.lower().endswith(EXCLUDED_FILE_SUFFIXES)


def _relative_if_within(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction(path))


def _walk_files(repo_root: Path) -> Iterator[Path]:
    for directory, names, filenames in os.walk(repo_root, topdown=True, followlinks=False):
        current = Path(directory)
        kept: list[str] = []
        for name in sorted(names):
            candidate = current / name
            relative = candidate.relative_to(repo_root)
            if _excluded_directory(relative):
                continue
            if _is_link_or_junction(candidate):
                raise HandoffError(f"included directory may not be a symlink: {relative}")
            kept.append(name)
        names[:] = kept
        for name in sorted(filenames):
            candidate = current / name
            relative = candidate.relative_to(repo_root)
            if _excluded_file(relative):
                continue
            if _is_link_or_junction(candidate):
                raise HandoffError(f"included file may not be a symlink: {relative}")
            if candidate.is_file():
                yield candidate


def selected_files(
    repo_root: Path,
    *,
    additionally_excluded: Iterable[Path] = (),
) -> tuple[tuple[str, Path], ...]:
    """Return the sorted working-tree payload selected by the fixed policy."""

    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise HandoffError(f"repository root is not a directory: {repo_root}")
    excluded = {path.resolve() for path in additionally_excluded}
    selected: list[tuple[str, Path]] = []
    reserved = {MANIFEST_NAME.casefold(), CHECKSUMS_NAME.casefold()}
    for path in _walk_files(repo_root):
        if path.resolve() in excluded:
            continue
        relative = path.relative_to(repo_root).as_posix()
        if relative.casefold() in reserved:
            raise HandoffError(f"source tree uses reserved handoff path: {relative}")
        selected.append((relative, path))
    selected.sort(key=lambda row: row[0])
    return tuple(selected)


def _zip_info(relative: str) -> ZipInfo:
    info = ZipInfo(f"{ARCHIVE_ROOT}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    mode = 0o755 if relative.endswith(".sh") else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.create_system = 3
    return info


def _write_payload(archive: ZipFile, relative: str, source: Path) -> dict[str, Any]:
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
        raise HandoffError(f"source changed while it was archived: {source}")
    return {"path": relative, "size": byte_count, "sha256": digest.hexdigest()}


def _write_generated(archive: ZipFile, relative: str, payload: bytes) -> None:
    with archive.open(_zip_info(relative), "w", force_zip64=True) as writer:
        writer.write(payload)


def _payload_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = [
        [str(row["path"]), int(row["size"]), str(row["sha256"])] for row in records
    ]
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dynamic_exclusions(repo_root: Path, output: Path) -> tuple[Path, ...]:
    sidecar = output.with_suffix(output.suffix + ".sha256")
    partial = output.with_suffix(output.suffix + ".partial")
    candidates = (output, sidecar, partial)
    return tuple(path for path in candidates if _relative_if_within(path, repo_root) is not None)


def build_handoff(
    repo_root: Path,
    output: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() and not force:
        raise HandoffError(f"output already exists (use --force): {output}")
    if sidecar.exists() and not force:
        raise HandoffError(f"checksum sidecar already exists (use --force): {sidecar}")

    files = selected_files(
        repo_root,
        additionally_excluded=_dynamic_exclusions(repo_root, output),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()

    records: list[dict[str, Any]] = []
    try:
        with ZipFile(
            partial,
            "w",
            compression=ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for index, (relative, source) in enumerate(files, start=1):
                records.append(_write_payload(archive, relative, source))
                if index % 250 == 0 or index == len(files):
                    mib = sum(int(row["size"]) for row in records) / 1024**2
                    print(f"Repository handoff {index}/{len(files)} files ({mib:.1f} MiB)")

            manifest = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "archive_root": ARCHIVE_ROOT,
                "compression": "ZIP_DEFLATED",
                "file_count": len(records),
                "payload_bytes": sum(int(row["size"]) for row in records),
                "payload_fingerprint": _payload_fingerprint(records),
                "scope": (
                    "current working tree including untracked files; local environments, "
                    "datasets, experimental artifacts, prior handoffs, Git internals, and "
                    "transient caches excluded; source/planning snapshot, not a complete "
                    "runtime-environment reproduction"
                ),
                "exclusion_policy": _policy_record(),
                "git": _git_record(repo_root),
                "files": records,
            }
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            _write_generated(archive, MANIFEST_NAME, manifest_bytes)
            checksum_rows = [f"{row['sha256']}  {row['path']}" for row in records]
            checksum_rows.append(
                f"{hashlib.sha256(manifest_bytes).hexdigest()}  {MANIFEST_NAME}"
            )
            _write_generated(
                archive,
                CHECKSUMS_NAME,
                ("\n".join(checksum_rows) + "\n").encode("utf-8"),
            )
        partial.replace(output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    archive_sha256 = _sha256_file(output)
    sidecar.write_text(
        f"{archive_sha256}  {output.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    result = {
        "archive": str(output),
        "archive_sha256": archive_sha256,
        "checksum_sidecar": str(sidecar),
        "file_count": len(records),
        "payload_bytes": manifest["payload_bytes"],
        "archive_bytes": output.stat().st_size,
        "payload_fingerprint": manifest["payload_fingerprint"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _safe_member_relative(name: str) -> str:
    if "\\" in name or name.startswith("/"):
        raise HandoffError(f"unsafe archive member path: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise HandoffError(f"unsafe archive member path: {name}")
    if path.as_posix() != name:
        raise HandoffError(f"non-canonical archive member path: {name}")
    if not path.parts or path.parts[0] != ARCHIVE_ROOT or len(path.parts) < 2:
        raise HandoffError(f"archive member is outside {ARCHIVE_ROOT}/: {name}")
    relative = PurePosixPath(*path.parts[1:]).as_posix()
    if ":" in relative:
        raise HandoffError(f"unsafe archive member path: {name}")
    return relative


def _load_json_bytes(payload: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"invalid {description}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"{description} must be a JSON object")
    return value


def _parse_checksums(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise HandoffError("invalid checksum file encoding") from exc
    rows: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative in rows
        ):
            raise HandoffError("invalid or duplicate checksum row")
        rows[relative] = digest
    return rows


def verify_handoff(archive_path: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise HandoffError(f"archive is missing: {archive_path}")
    try:
        with ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if not infos:
                raise HandoffError("archive is empty")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) != len(
                {name.casefold() for name in names}
            ):
                raise HandoffError("archive contains duplicate member names")
            relative_infos: dict[str, ZipInfo] = {}
            for info in infos:
                if info.is_dir():
                    raise HandoffError(f"archive contains an unexpected directory entry: {info.filename}")
                if info.compress_type != ZIP_DEFLATED:
                    raise HandoffError(f"archive member uses unexpected compression: {info.filename}")
                file_type = (info.external_attr >> 16) & 0o170000
                if file_type == stat.S_IFLNK:
                    raise HandoffError(f"archive contains a symlink: {info.filename}")
                relative = _safe_member_relative(info.filename)
                relative_infos[relative] = info

            required = {MANIFEST_NAME, CHECKSUMS_NAME}
            if not required <= set(relative_infos):
                raise HandoffError("archive manifest or checksum file is missing")
            manifest_bytes = archive.read(relative_infos[MANIFEST_NAME])
            manifest = _load_json_bytes(manifest_bytes, "handoff manifest")
            if (
                manifest.get("schema") != SCHEMA
                or int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
                or manifest.get("archive_root") != ARCHIVE_ROOT
                or manifest.get("compression") != "ZIP_DEFLATED"
            ):
                raise HandoffError("unsupported handoff manifest")

            raw_records = manifest.get("files")
            if not isinstance(raw_records, list):
                raise HandoffError("manifest files must be a list")
            records: list[dict[str, Any]] = []
            for value in raw_records:
                if not isinstance(value, dict):
                    raise HandoffError("manifest file record must be an object")
                try:
                    record = {
                        "path": str(value["path"]),
                        "size": int(value["size"]),
                        "sha256": str(value["sha256"]),
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    raise HandoffError("invalid manifest file record") from exc
                if (
                    _safe_member_relative(f"{ARCHIVE_ROOT}/{record['path']}")
                    != record["path"]
                ):
                    raise HandoffError("invalid manifest file path")
                records.append(record)
            record_paths = [row["path"] for row in records]
            if record_paths != sorted(record_paths) or len(record_paths) != len(set(record_paths)):
                raise HandoffError("manifest file paths are unsorted or duplicated")
            actual_payload = set(relative_infos) - required
            if actual_payload != set(record_paths):
                raise HandoffError("archive payload does not match the manifest")
            if int(manifest.get("file_count", -1)) != len(records):
                raise HandoffError("manifest file count is inconsistent")
            if int(manifest.get("payload_bytes", -1)) != sum(row["size"] for row in records):
                raise HandoffError("manifest payload size is inconsistent")
            if manifest.get("payload_fingerprint") != _payload_fingerprint(records):
                raise HandoffError("manifest payload fingerprint is inconsistent")

            checksums = _parse_checksums(archive.read(relative_infos[CHECKSUMS_NAME]))
            expected_checksum_paths = set(record_paths) | {MANIFEST_NAME}
            if set(checksums) != expected_checksum_paths:
                raise HandoffError("checksum inventory does not match the manifest")
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if checksums[MANIFEST_NAME] != manifest_sha256:
                raise HandoffError("manifest checksum changed")
            for row in records:
                info = relative_infos[row["path"]]
                if info.file_size != row["size"]:
                    raise HandoffError(f"payload size changed: {row['path']}")
                digest = _sha256_member(archive, info)
                if digest != row["sha256"] or digest != checksums[row["path"]]:
                    raise HandoffError(f"payload checksum changed: {row['path']}")
    except BadZipFile as exc:
        raise HandoffError(f"invalid ZIP archive: {archive_path}") from exc

    archive_sha256 = _sha256_file(archive_path)
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if sidecar.is_file():
        try:
            line = sidecar.read_text(encoding="utf-8").strip()
        except UnicodeError as exc:
            raise HandoffError("invalid checksum sidecar encoding") from exc
        digest, separator, filename = line.partition("  ")
        if separator != "  " or filename != archive_path.name or digest != archive_sha256:
            raise HandoffError("archive checksum sidecar does not match")
    result = {
        "archive": str(archive_path),
        "archive_sha256": archive_sha256,
        "file_count": len(records),
        "payload_bytes": sum(row["size"] for row in records),
        "payload_fingerprint": manifest["payload_fingerprint"],
        "verified": 1,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def list_handoff(repo_root: Path, output: Path) -> dict[str, Any]:
    files = selected_files(
        repo_root,
        additionally_excluded=_dynamic_exclusions(repo_root.resolve(), output.resolve()),
    )
    result = {
        "file_count": len(files),
        "payload_bytes": sum(path.stat().st_size for _, path in files),
        "exclusion_policy": _policy_record(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_root = script.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "handoff/repository-handoff.zip",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-only", action="store_true")
    modes.add_argument("--verify", type=Path, metavar="ARCHIVE")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify is not None:
            verify_handoff(args.verify)
        elif args.list_only:
            list_handoff(args.repo_root, args.output)
        else:
            build_handoff(args.repo_root, args.output, force=bool(args.force))
    except (HandoffError, OSError) as exc:
        print(f"Repository handoff error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
