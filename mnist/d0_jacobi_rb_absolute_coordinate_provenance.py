"""Fail-closed provenance for read-only absolute-coordinate adjudication.

The two historical inputs have deliberately different containers:

* the terminal directional result is a portable ZIP and is never extracted;
* the physical coarse witness is an immutable run directory.

Both verifiers bind content rather than the caller's filesystem root.  ZIP
members and registry rows are treated as hostile paths, every payload is read
through :mod:`zipfile` (therefore checking its CRC), and every registered
byte is checked against its frozen size and SHA-256 before any scientific
field is interpreted.
"""

from __future__ import annotations

import hashlib
import json
import stat
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)


SCHEMA = "experiment12-d0-jacobi-rb-absolute-coordinate-provenance"
SCHEMA_VERSION = 1


class AbsoluteCoordinateProvenanceError(ArtifactCompatibilityError):
    """An immutable absolute-coordinate parent failed a strict binding."""


@dataclass(frozen=True)
class PortableResultSpec:
    basename: str
    archive_sha256: str
    archive_size: int
    root_name: str
    entry_count: int
    file_count: int
    directory_paths: tuple[str, ...]
    total_uncompressed_bytes: int
    registry_schema: str
    registry_artifact_count: int
    registry_semantic_sha256: str
    registry_file_sha256: str
    manifest_schema: str
    manifest_semantic_sha256: str
    config_schema: str
    config_semantic_sha256: str
    decision_schema: str
    status_schema: str
    terminal_decision: str


@dataclass(frozen=True)
class CoarsePanelSpec:
    panel: str
    relative_path: str
    file_size: int
    file_sha256: str
    array_sha256: str
    panel_fingerprint: str
    seal_file_sha256: str
    path_ids: tuple[int, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class CoarseWitnessSpec:
    basename: str
    registry_schema: str
    registry_record_count: int
    registry_semantic_sha256: str
    registry_file_sha256: str
    registry_file_size: int
    status_file_sha256: str
    run_schema: str
    config_schema: str
    config_semantic_sha256: str
    source_fingerprint: str
    path_plan_sha256: str
    statistic_plan_sha256: str
    decision_schema: str
    terminal_decision: str
    gate_schema: str
    panels: tuple[CoarsePanelSpec, ...]
    joint_seal_file_sha256: str
    expected_file_count: int
    expected_total_bytes: int


PORTABLE_RESULT_BASENAME = (
    "20260808-135158_production-runpod-quartile-directional-continuation.zip"
)
PORTABLE_RESULT_ROOT_NAME = PORTABLE_RESULT_BASENAME.removesuffix(".zip")
PORTABLE_RESULT_ARCHIVE_SHA256 = (
    "0f9914b79011a1182bac8fd9645e7ac0e222618d5be92047c03268e8b9ab3f7d"
)
PORTABLE_RESULT_REGISTRY_SEMANTIC_SHA256 = (
    "cf206b49a094ede6196fd794f945c8ecf616e3caf48ef12b32c31afc8cafea64"
)
PORTABLE_RESULT_REGISTRY_FILE_SHA256 = (
    "3fe04ff3d4a8a5231f6588b8383610e8d283492774f0e7bbe2146824550f50b6"
)
PORTABLE_RESULT_CONFIG_SHA256 = (
    "00f2464129b4c4dcfbd727aed97173abcc59e0e29697b77bafa76d1d28c0d39e"
)
PORTABLE_RESULT_MANIFEST_SHA256 = (
    "95a9b984e209d949db35811b559ac30e5893d2ed3a2fce105b20969b479b73b0"
)
PORTABLE_RESULT_DECISION = "representation_cancellation_nonidentifying_stop"

PORTABLE_RESULT_SPEC = PortableResultSpec(
    basename=PORTABLE_RESULT_BASENAME,
    archive_sha256=PORTABLE_RESULT_ARCHIVE_SHA256,
    archive_size=305_135_140,
    root_name=PORTABLE_RESULT_ROOT_NAME,
    entry_count=2_047,
    file_count=2_042,
    directory_paths=(
        "",
        "directional_shards",
        "directional_shards/gain_calibration",
        "directional_shards/physical_fit",
        "directional_shards/training_rank",
    ),
    total_uncompressed_bytes=389_824_231,
    registry_schema=(
        "experiment12-d0-jacobi-rb-quartile-directional-adjudication-"
        "artifact-registry"
    ),
    registry_artifact_count=2_041,
    registry_semantic_sha256=PORTABLE_RESULT_REGISTRY_SEMANTIC_SHA256,
    registry_file_sha256=PORTABLE_RESULT_REGISTRY_FILE_SHA256,
    manifest_schema=(
        "experiment12-d0-jacobi-rb-quartile-directional-portable-"
        "continuation-manifest"
    ),
    manifest_semantic_sha256=PORTABLE_RESULT_MANIFEST_SHA256,
    config_schema=(
        "experiment12-d0-jacobi-rb-quartile-directional-portable-"
        "continuation-scientific-config"
    ),
    config_semantic_sha256=PORTABLE_RESULT_CONFIG_SHA256,
    decision_schema=(
        "d0-jacobi-rb-quartile-directional-adjudication-gate-v1-decision"
    ),
    status_schema=(
        "experiment12-d0-jacobi-rb-quartile-directional-adjudication-status"
    ),
    terminal_decision=PORTABLE_RESULT_DECISION,
)


COARSE_WITNESS_BASENAME = (
    "20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix"
)
COARSE_WITNESS_REGISTRY_SEMANTIC_SHA256 = (
    "ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3"
)
COARSE_WITNESS_REGISTRY_FILE_SHA256 = (
    "866984822ef14dbb13f0644a0f23697f6fb42ecec40b07341249d74043319747"
)
COARSE_WITNESS_CONFIG_SHA256 = (
    "b2e28989ef6da6fa2d233b14ee475c04e10326079cf03750f1f427494de90f14"
)
COARSE_WITNESS_SOURCE_FINGERPRINT = (
    "31f1f15008c2db864e282c5d3fa047986a9b576b92c480d50a18d55138e9eafb"
)
COARSE_WITNESS_DECISION = "exact_physical_coarse_signal_detected"
COARSE_WITNESS_PATH_PLAN_SHA256 = (
    "76f44f7c83f5f294ebda5d55f610c0942e7f16f1ab2ea2940ae70a5f2b059a65"
)
COARSE_WITNESS_STATISTIC_PLAN_SHA256 = (
    "91cee2ce9eb5a1688dcfc72ba2a27c02c53a73b88e29807cc556aff75c403ec0"
)

COARSE_PANEL_A = CoarsePanelSpec(
    panel="a",
    relative_path="panels/a/cell_means.npz",
    file_size=5_376_348,
    file_sha256="70d374526df5c02e5c6ab7f9b17205de373b22c694480bb27bf5684b4a579852",
    array_sha256="1fe04953fd50ea3cb0ac163efed216ec5ebbafc58f48ce0de3f77d090c29fe08",
    panel_fingerprint="31b4d463e32c3c207ea9b2739ff64341ba0d16661ecb28c95b74a127e3011a11",
    seal_file_sha256="8c034d0978e36549ee01ef3e281731f17024142697a9d1350ea35c44243be594",
    path_ids=tuple(range(0xE5000, 0xE5040)),
    shape=(64, 4, 7, 392),
)
COARSE_PANEL_B = CoarsePanelSpec(
    panel="b",
    relative_path="panels/b/cell_means.npz",
    file_size=5_376_191,
    file_sha256="d64688f026cc510d586fb6b20e2303fdbe407a99b1a161b4654dc5dd04face81",
    array_sha256="2d949662c098783aa663672528f107a9f73f503529440aca4313cf770cad737e",
    panel_fingerprint="4569ac4baff93c47f2921906bc90746ed7ba11fca94a763372158b68d416505b",
    seal_file_sha256="21f1e1725fbf600171e80dd69b006b6cd4f83cd8a0dc7215590a3a38508e8980",
    path_ids=tuple(range(0xE5100, 0xE5140)),
    shape=(64, 4, 7, 392),
)

COARSE_WITNESS_SPEC = CoarseWitnessSpec(
    basename=COARSE_WITNESS_BASENAME,
    registry_schema=(
        "experiment12-d0-jacobi-rb-physical-coarse-signal-witness-"
        "artifact-registry"
    ),
    registry_record_count=2_616,
    registry_semantic_sha256=COARSE_WITNESS_REGISTRY_SEMANTIC_SHA256,
    registry_file_sha256=COARSE_WITNESS_REGISTRY_FILE_SHA256,
    registry_file_size=445_965,
    status_file_sha256=(
        "ae982dd57034ee54226dc6f84fea9dea48d351773f5e911bc584b2f58600624c"
    ),
    run_schema="experiment12-d0-jacobi-rb-physical-coarse-signal-witness",
    config_schema=(
        "experiment12-d0-jacobi-rb-physical-coarse-signal-witness-"
        "scientific-config"
    ),
    config_semantic_sha256=COARSE_WITNESS_CONFIG_SHA256,
    source_fingerprint=COARSE_WITNESS_SOURCE_FINGERPRINT,
    path_plan_sha256=COARSE_WITNESS_PATH_PLAN_SHA256,
    statistic_plan_sha256=COARSE_WITNESS_STATISTIC_PLAN_SHA256,
    decision_schema="d0-jacobi-rb-physical-coarse-signal-decision-v1",
    terminal_decision=COARSE_WITNESS_DECISION,
    gate_schema="d0-jacobi-rb-physical-coarse-signal-gate-v1",
    panels=(COARSE_PANEL_A, COARSE_PANEL_B),
    joint_seal_file_sha256=(
        "1355f8b6c0b62b9631bf97f9a2aa0b18fba3fc993edf68d75b3cbb8ba03e3113"
    ),
    expected_file_count=2_618,
    expected_total_bytes=362_400_593,
)


_ZERO_SCOPE_FIELDS = (
    "cache_generation_authorized",
    "confirmation_authorized",
    "controller_execution_authorized",
    "controller_planning_authorized",
    "controller_trajectories_executed",
    "fresh_calibration_authorized",
    "fresh_coordinate_learner_plan_authorized",
    "fresh_fit_authorized",
    "fresh_learner_plan_authorized",
    "fresh_rank_authorized",
    "fresh_selection_authorized",
    "full_dataset_training_authorized",
    "historical_design_evidence_authorizing",
    "new_checkpoints_created",
    "new_learner_training_authorized",
    "new_learner_training_performed",
    "new_path_generation_authorized",
    "new_transitions_generated",
    "optimizer_updates_performed",
    "parent_confirmation_opened",
    "parent_files_modified",
    "parent_selection_opened",
    "physical_training_authorized",
    "physical_training_performed",
    "production_refinement_authorized",
    "production_refinement_performed",
    "reconstruction_authorized",
    "reconstruction_claim_authorized",
    "reconstructions_created",
    "reverse_sampling_authorized",
    "reverse_sampling_performed",
    "samples_created",
    "sampling_authorized",
    "sampling_performed",
)

_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class CoarseWitnessPanels:
    """Verified, immutable host arrays from the two historical panels."""

    panel_a_path_ids: np.ndarray
    panel_a: np.ndarray
    panel_b_path_ids: np.ndarray
    panel_b: np.ndarray


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AbsoluteCoordinateProvenanceError(message)


def _hashed(record: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(record)
    body.pop("semantic_sha256", None)
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _assert_semantic(record: Mapping[str, Any], description: str) -> None:
    body = dict(record)
    observed = body.pop("semantic_sha256", None)
    _require(
        observed == config_fingerprint(body),
        f"{description} semantic hash changed",
    )


def _strict_json_bytes(payload: bytes, description: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON value: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AbsoluteCoordinateProvenanceError(
            f"invalid {description} JSON"
        ) from exc
    _require(isinstance(value, dict), f"{description} must be a JSON object")
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {description}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AbsoluteCoordinateProvenanceError(
            f"could not read {description}"
        ) from exc
    return _strict_json_bytes(payload, description)


def _scope_is_closed(record: Mapping[str, Any], description: str) -> None:
    for field in _ZERO_SCOPE_FIELDS:
        if field in record:
            _require(
                type(record[field]) is int and record[field] == 0,
                f"{description} unexpectedly changed {field}",
            )


def _portable_parts(raw: Any, description: str) -> tuple[str, ...]:
    _require(isinstance(raw, str) and raw != "", f"{description} path is invalid")
    _require(raw == unicodedata.normalize("NFC", raw), f"{description} path is not NFC")
    _require(
        not raw.startswith(("/", "\\")) and "\\" not in raw and "\x00" not in raw,
        f"unsafe {description} path: {raw!r}",
    )
    parts = raw.split("/")
    _require(
        all(part not in {"", ".", ".."} for part in parts),
        f"unsafe {description} path: {raw!r}",
    )
    for part in parts:
        _require(
            not part.endswith((" ", "."))
            and ":" not in part
            and all(ord(character) >= 32 for character in part),
            f"non-portable {description} path: {raw!r}",
        )
        stem = part.split(".", 1)[0].casefold()
        _require(
            stem not in _WINDOWS_RESERVED_NAMES,
            f"reserved {description} path: {raw!r}",
        )
    return tuple(parts)


def _portable_relative(raw: Any, description: str) -> str:
    return "/".join(_portable_parts(raw, description))


def _case_key(relative: str) -> str:
    return unicodedata.normalize("NFC", relative).casefold()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_zip_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, description: str
) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    try:
        with archive.open(info, "r") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
                chunks.append(chunk)
    except (OSError, EOFError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
        raise AbsoluteCoordinateProvenanceError(
            f"{description} CRC or payload is invalid"
        ) from exc
    _require(size == info.file_size, f"{description} uncompressed size changed")
    return b"".join(chunks), digest.hexdigest()


def _validate_zip_inventory(
    archive: zipfile.ZipFile, spec: PortableResultSpec
) -> tuple[dict[str, zipfile.ZipInfo], set[str]]:
    infos = archive.infolist()
    _require(len(infos) == spec.entry_count, "portable archive entry count changed")
    files: dict[str, zipfile.ZipInfo] = {}
    directories: set[str] = set()
    exact_seen: set[str] = set()
    case_seen: set[str] = set()
    total = 0
    for info in infos:
        raw = info.filename
        is_directory = info.is_dir()
        logical_raw = raw[:-1] if is_directory and raw.endswith("/") else raw
        parts = _portable_parts(logical_raw, "archive member")
        _require(parts[0] == spec.root_name, "portable archive root changed")
        relative = "/".join(parts[1:])
        _require(relative != "" or is_directory, "archive root is not a directory")
        logical = "/".join(parts)
        _require(logical not in exact_seen, "portable archive member is duplicated")
        exact_seen.add(logical)
        folded = _case_key(logical)
        _require(folded not in case_seen, "portable archive has a case collision")
        case_seen.add(folded)
        _require((info.flag_bits & 0x1) == 0, "encrypted archive member is forbidden")
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if is_directory:
            _require(
                file_type in (0, stat.S_IFDIR)
                and info.file_size == 0
                and info.compress_size == 0
                and info.CRC == 0,
                "portable archive directory entry is invalid",
            )
            directories.add(relative)
        else:
            _require(
                file_type in (0, stat.S_IFREG),
                "portable archive contains a non-regular member",
            )
            _require(relative not in files, "portable archive file is duplicated")
            files[relative] = info
            total += int(info.file_size)
    _require(len(files) == spec.file_count, "portable archive file count changed")
    _require(
        directories == set(spec.directory_paths),
        "portable archive directory inventory changed",
    )
    _require(
        total == spec.total_uncompressed_bytes,
        "portable archive uncompressed byte count changed",
    )
    return files, directories


def _validated_artifact_rows(
    rows: Any,
    *,
    expected_count: int,
    description: str,
) -> list[dict[str, Any]]:
    _require(isinstance(rows, list), f"{description} rows are invalid")
    _require(len(rows) == expected_count, f"{description} row count changed")
    normalized: list[dict[str, Any]] = []
    exact: set[str] = set()
    folded: set[str] = set()
    for row in rows:
        _require(isinstance(row, Mapping), f"{description} row is malformed")
        relative = _portable_relative(row.get("path"), f"{description} row")
        _require(relative not in exact, f"{description} path is duplicated")
        exact.add(relative)
        case_key = _case_key(relative)
        _require(case_key not in folded, f"{description} has a case collision")
        folded.add(case_key)
        size = row.get("size")
        sha256 = row.get("sha256")
        _require(
            type(size) is int and size >= 0,
            f"{description} size is invalid: {relative}",
        )
        _require(_valid_sha256(sha256), f"{description} SHA-256 is invalid: {relative}")
        normalized.append({"path": relative, "size": size, "sha256": sha256})
    _require(
        [row["path"] for row in normalized]
        == sorted(row["path"] for row in normalized),
        f"{description} row order changed",
    )
    return normalized


def _verify_portable_terminal(
    payloads: Mapping[str, bytes], spec: PortableResultSpec
) -> dict[str, Any]:
    manifest = _strict_json_bytes(payloads["run_manifest.json"], "portable manifest")
    config = _strict_json_bytes(payloads["scientific_config.json"], "portable config")
    status = _strict_json_bytes(payloads["run_status.json"], "portable status")
    decision = _strict_json_bytes(
        payloads["quartile_directional_adjudication_decision.json"],
        "portable decision",
    )
    _assert_semantic(manifest, "portable manifest")
    _assert_semantic(config, "portable config")
    _require(
        manifest.get("schema") == spec.manifest_schema
        and manifest.get("schema_version") == 1
        and manifest.get("semantic_sha256") == spec.manifest_semantic_sha256,
        "portable manifest binding changed",
    )
    _require(
        config.get("schema") == spec.config_schema
        and config.get("schema_version") == 1
        and config.get("semantic_sha256") == spec.config_semantic_sha256,
        "portable scientific config binding changed",
    )
    _require(
        manifest.get("scientific_config_sha256") == spec.config_semantic_sha256,
        "portable manifest/config binding changed",
    )
    _require(
        status.get("schema") == spec.status_schema
        and status.get("schema_version") == 1
        and status.get("state") == "valid_scientific_stop"
        and status.get("stage") == "report"
        and status.get("decision") == spec.terminal_decision
        and status.get("failure_domain") == "scientific_gate"
        and status.get("failure_code") == spec.terminal_decision
        and status.get("scientific_evidence_complete") == 1,
        "portable terminal status changed",
    )
    _require(
        decision.get("schema") == spec.decision_schema
        and decision.get("schema_version") == 1
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == spec.terminal_decision
        and decision.get("terminal") == 1
        and decision.get("scientific_evidence_complete") == 1
        and decision.get("invalid_evidence") == 0
        and decision.get("valid_scientific_stop") == 1
        and decision.get("unique_representation_identified") == 0,
        "portable terminal decision changed",
    )
    for description, record in (
        ("portable manifest", manifest),
        ("portable config", config),
        ("portable status", status),
        ("portable decision", decision),
    ):
        _scope_is_closed(record, description)
    return {
        "state": "valid_scientific_stop",
        "stage": "report",
        "decision": spec.terminal_decision,
        "scientific_evidence_complete": 1,
        "valid_scientific_stop": 1,
        "unique_representation_identified": 0,
    }


def verify_portable_result_archive(zip_path: str | Path) -> dict[str, Any]:
    """Verify the exact portable directional ZIP without extracting it."""

    spec = PORTABLE_RESULT_SPEC
    source = Path(zip_path)
    _require(source.is_file(), f"portable result archive does not exist: {source}")
    _require(not source.is_symlink(), "portable result archive may not be a symlink")
    _require(source.name == spec.basename, "wrong portable result archive basename")
    try:
        with zipfile.ZipFile(source, "r") as archive:
            files, directories = _validate_zip_inventory(archive, spec)
            registry_info = files.get("artifact_registry.json")
            _require(registry_info is not None, "portable artifact registry is missing")
            registry_bytes, registry_file_sha = _read_zip_member(
                archive, registry_info, "portable artifact registry"
            )
            _require(
                registry_file_sha == spec.registry_file_sha256,
                "portable registry file hash changed",
            )
            registry = _strict_json_bytes(
                registry_bytes, "portable artifact registry"
            )
            _assert_semantic(registry, "portable artifact registry")
            _require(
                registry.get("schema") == spec.registry_schema
                and registry.get("schema_version") == 1
                and registry.get("artifact_count") == spec.registry_artifact_count
                and registry.get("semantic_sha256")
                == spec.registry_semantic_sha256,
                "portable registry binding changed",
            )
            rows = _validated_artifact_rows(
                registry.get("artifacts"),
                expected_count=spec.registry_artifact_count,
                description="portable registry",
            )
            expected_files = {row["path"] for row in rows} | {
                "artifact_registry.json"
            }
            _require(
                set(files) == expected_files,
                "portable registered file inventory changed",
            )
            required_json = {
                "run_manifest.json",
                "scientific_config.json",
                "run_status.json",
                "quartile_directional_adjudication_decision.json",
            }
            payloads: dict[str, bytes] = {}
            for row in rows:
                relative = row["path"]
                info = files[relative]
                _require(
                    info.file_size == row["size"],
                    f"portable artifact size changed: {relative}",
                )
                payload, observed_sha = _read_zip_member(
                    archive, info, f"portable artifact {relative}"
                )
                _require(
                    observed_sha == row["sha256"],
                    f"portable artifact hash changed: {relative}",
                )
                if relative in required_json:
                    payloads[relative] = payload
            _require(
                set(payloads) == required_json,
                "portable terminal artifacts are missing",
            )
            terminal = _verify_portable_terminal(payloads, spec)
            _scope_is_closed(registry, "portable registry")
    except AbsoluteCoordinateProvenanceError:
        raise
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
        raise AbsoluteCoordinateProvenanceError(
            "portable result ZIP is unreadable or corrupt"
        ) from exc

    observed_size = source.stat().st_size
    observed_archive_sha = file_fingerprint(source)
    _require(observed_size == spec.archive_size, "portable archive size changed")
    _require(
        observed_archive_sha == spec.archive_sha256,
        "portable archive file hash changed",
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-portable-directional-parent",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "archive_basename": spec.basename,
            "archive_file_sha256": observed_archive_sha,
            "archive_file_size": observed_size,
            "entry_count": spec.entry_count,
            "file_count": spec.file_count,
            "directory_count": len(directories),
            "total_uncompressed_bytes": spec.total_uncompressed_bytes,
            "safe_relative_paths": 1,
            "unique_paths": 1,
            "casefold_unique_paths": 1,
            "all_archive_crcs_verified": 1,
            "all_registered_artifact_hashes_verified": 1,
            "registry": {
                "schema": spec.registry_schema,
                "artifact_count": spec.registry_artifact_count,
                "semantic_sha256": spec.registry_semantic_sha256,
                "file_sha256": spec.registry_file_sha256,
            },
            "manifest_semantic_sha256": spec.manifest_semantic_sha256,
            "scientific_config_sha256": spec.config_semantic_sha256,
            "terminal": terminal,
            "root_independent_identity": 1,
        }
    )


def _tree_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    folded: set[str] = set()
    for item in root.rglob("*"):
        _require(not item.is_symlink(), f"coarse witness contains a symlink: {item.name}")
        relative = item.relative_to(root).as_posix()
        _portable_relative(relative, "coarse witness")
        key = _case_key(relative)
        _require(key not in folded, "coarse witness tree has a case collision")
        folded.add(key)
        if item.is_file():
            files[relative] = item
        else:
            _require(item.is_dir(), "coarse witness contains a non-regular entry")
    return files


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _panel_fingerprint(
    panel_spec: CoarsePanelSpec, path_ids: np.ndarray, cell_means: np.ndarray
) -> str:
    panel_digest = hashlib.sha256()
    panel_digest.update(str(cell_means.dtype).encode("ascii"))
    panel_digest.update(
        str(tuple(int(item) for item in cell_means.shape)).encode("ascii")
    )
    panel_digest.update(memoryview(np.ascontiguousarray(cell_means)).cast("B"))
    return config_fingerprint(
        {
            "schema": "d0-jacobi-rb-physical-coarse-signal-v1-panel-v1",
            "role": f"panel-{panel_spec.panel}",
            "path_ids": path_ids.tolist(),
            "shape": list(cell_means.shape),
            "dtype": cell_means.dtype.str,
            "observations_per_cell": 8,
            # The historical panel class predates the persistence seal and
            # used ``str(dtype)`` rather than ``dtype.str`` in this one hash.
            "cell_means_sha256": panel_digest.hexdigest(),
        }
    )


def _load_panel_arrays(
    root: Path, panel_spec: CoarsePanelSpec
) -> tuple[np.ndarray, np.ndarray]:
    path = root.joinpath(*panel_spec.relative_path.split("/"))
    _require(path.is_file() and not path.is_symlink(), f"panel {panel_spec.panel} is missing")
    _require(path.stat().st_size == panel_spec.file_size, f"panel {panel_spec.panel} size changed")
    _require(file_fingerprint(path) == panel_spec.file_sha256, f"panel {panel_spec.panel} hash changed")
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                len(archive.files) == 2 and set(archive.files) == {"path_ids", "cell_means"},
                f"panel {panel_spec.panel} NPZ schema changed",
            )
            path_ids = np.array(archive["path_ids"], copy=True)
            cell_means = np.array(archive["cell_means"], copy=True)
    except AbsoluteCoordinateProvenanceError:
        raise
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise AbsoluteCoordinateProvenanceError(
            f"panel {panel_spec.panel} NPZ is invalid"
        ) from exc
    _require(
        path_ids.dtype == np.dtype(np.int64)
        and path_ids.shape == (len(panel_spec.path_ids),)
        and path_ids.tolist() == list(panel_spec.path_ids),
        f"panel {panel_spec.panel} path IDs changed",
    )
    _require(
        cell_means.dtype == np.dtype(np.float64)
        and cell_means.shape == panel_spec.shape
        and bool(np.isfinite(cell_means).all()),
        f"panel {panel_spec.panel} cell means changed",
    )
    _require(
        _array_sha256(cell_means) == panel_spec.array_sha256,
        f"panel {panel_spec.panel} array hash changed",
    )
    _require(
        _panel_fingerprint(panel_spec, path_ids, cell_means)
        == panel_spec.panel_fingerprint,
        f"panel {panel_spec.panel} fingerprint changed",
    )
    path_ids.setflags(write=False)
    cell_means.setflags(write=False)
    return path_ids, cell_means


def _verify_panel_seal(
    root: Path,
    panel_spec: CoarsePanelSpec,
    path_ids: np.ndarray,
    cell_means: np.ndarray,
    witness_spec: CoarseWitnessSpec,
) -> dict[str, Any]:
    seal_path = root / f"panel_{panel_spec.panel}_seal.json"
    _require(
        file_fingerprint(seal_path) == panel_spec.seal_file_sha256,
        f"panel {panel_spec.panel} seal file hash changed",
    )
    seal = _load_json(seal_path, f"panel {panel_spec.panel} seal")
    _require(
        seal.get("schema") == f"{witness_spec.run_schema}-panel-seal"
        and seal.get("schema_version") == 1
        and seal.get("panel") == panel_spec.panel
        and seal.get("path_ids") == path_ids.tolist()
        and seal.get("cell_means_file") == panel_spec.relative_path
        and seal.get("cell_means_file_sha256") == panel_spec.file_sha256
        and seal.get("cell_means_array_sha256") == _array_sha256(cell_means)
        and seal.get("panel_fingerprint") == panel_spec.panel_fingerprint
        and seal.get("path_plan_sha256") == witness_spec.path_plan_sha256
        and seal.get("statistic_plan_sha256") == witness_spec.statistic_plan_sha256
        and seal.get("analysis_opened") == 0,
        f"panel {panel_spec.panel} seal changed",
    )
    bindings = {
        "execution_metrics_file_sha256": f"panels/{panel_spec.panel}/metrics.json",
        "persistence_audit_file_sha256": (
            f"panel_{panel_spec.panel}_cell_mean_persistence_audit.json"
        ),
        "resource_summary_file_sha256": f"panel_{panel_spec.panel}_resource_summary.json",
    }
    for field, relative in bindings.items():
        target = root.joinpath(*relative.split("/"))
        _require(
            target.is_file() and file_fingerprint(target) == seal.get(field),
            f"panel {panel_spec.panel} seal binding changed: {relative}",
        )
    _scope_is_closed(seal, f"panel {panel_spec.panel} seal")
    return seal


def _verified_tree_rows(
    rows: list[dict[str, Any]],
    root: Path,
    witness_spec: CoarseWitnessSpec,
) -> list[dict[str, Any]]:
    result = list(rows)
    for relative, expected_sha in (
        ("artifact_registry.json", witness_spec.registry_file_sha256),
        ("run_status.json", witness_spec.status_file_sha256),
    ):
        path = root / relative
        result.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": expected_sha,
            }
        )
    return sorted(result, key=lambda row: row["path"])


def verify_coarse_witness_run(run_dir: str | Path) -> dict[str, Any]:
    """Verify the exact coarse-witness registry, seals, and panel arrays."""

    spec = COARSE_WITNESS_SPEC
    supplied = Path(run_dir)
    _require(supplied.is_dir(), f"coarse witness run does not exist: {supplied}")
    _require(not supplied.is_symlink(), "coarse witness root may not be a symlink")
    root = supplied.resolve()
    _require(root.name == spec.basename, "wrong coarse witness run basename")
    files = _tree_files(root)
    _require(len(files) == spec.expected_file_count, "coarse witness file count changed")
    _require(
        sum(path.stat().st_size for path in files.values()) == spec.expected_total_bytes,
        "coarse witness total byte count changed",
    )

    registry_path = root / "artifact_registry.json"
    _require(registry_path.stat().st_size == spec.registry_file_size, "coarse witness registry size changed")
    _require(file_fingerprint(registry_path) == spec.registry_file_sha256, "coarse witness registry file hash changed")
    registry = _load_json(registry_path, "coarse witness registry")
    _require(
        registry.get("schema") == spec.registry_schema
        and registry.get("schema_version") == 1
        and registry.get("record_count") == spec.registry_record_count,
        "coarse witness registry schema or count changed",
    )
    rows = _validated_artifact_rows(
        registry.get("records"),
        expected_count=spec.registry_record_count,
        description="coarse witness registry",
    )
    _require(
        registry.get("registry_sha256") == config_fingerprint(rows)
        and registry.get("registry_sha256") == spec.registry_semantic_sha256,
        "coarse witness registry semantic hash changed",
    )
    _require(
        set(files) == {row["path"] for row in rows} | {"artifact_registry.json", "run_status.json"},
        "coarse witness registered file inventory changed",
    )
    for row in rows:
        path = files[row["path"]]
        _require(path.stat().st_size == row["size"], f"coarse witness artifact size changed: {row['path']}")
        _require(file_fingerprint(path) == row["sha256"], f"coarse witness artifact hash changed: {row['path']}")
    _require(
        file_fingerprint(root / "run_status.json") == spec.status_file_sha256,
        "coarse witness status file hash changed",
    )

    manifest = _load_json(root / "run_manifest.json", "coarse witness manifest")
    config = _load_json(root / "scientific_config.json", "coarse witness config")
    status = _load_json(root / "run_status.json", "coarse witness status")
    decision = _load_json(
        root / "physical_coarse_signal_decision.json", "coarse witness decision"
    )
    _assert_semantic(config, "coarse witness config")
    _require(
        manifest.get("schema") == spec.run_schema
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == spec.source_fingerprint
        and manifest.get("scientific_config_sha256") == spec.config_semantic_sha256
        and manifest.get("path_plan_sha256") == spec.path_plan_sha256
        and manifest.get("statistic_plan_sha256") == spec.statistic_plan_sha256,
        "coarse witness manifest binding changed",
    )
    _require(
        config.get("schema") == spec.config_schema
        and config.get("schema_version") == 1
        and config.get("semantic_sha256") == spec.config_semantic_sha256
        and config.get("path_plan_sha256") == spec.path_plan_sha256
        and config.get("statistic_plan_sha256") == spec.statistic_plan_sha256,
        "coarse witness scientific config changed",
    )
    _require(
        status.get("schema") == f"{spec.run_schema}-status"
        and status.get("schema_version") == 1
        and status.get("state") == "completed"
        and status.get("stage") == "analyze"
        and status.get("decision") == spec.terminal_decision
        and status.get("artifact_registry_record_count") == spec.registry_record_count
        and status.get("artifact_registry_sha256") == spec.registry_semantic_sha256
        and status.get("artifact_registry_file_sha256") == spec.registry_file_sha256
        and status.get("artifact_registry_file_size") == spec.registry_file_size,
        "coarse witness terminal status changed",
    )
    _require(
        decision.get("schema") == spec.decision_schema
        and decision.get("schema_version") == 1
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == spec.terminal_decision
        and decision.get("scientific_outcome") == spec.terminal_decision
        and decision.get("full_state_conditional_mean_zero_proven") == 0,
        "coarse witness terminal decision changed",
    )
    for description, record in (
        ("coarse witness registry", registry),
        ("coarse witness manifest", manifest),
        ("coarse witness config", config),
        ("coarse witness status", status),
        ("coarse witness decision", decision),
    ):
        _scope_is_closed(record, description)

    gate_table: dict[str, dict[str, Any]] = {}
    for filename, gate_name in (
        ("coarse_signal_preflight_gate.json", "preflight"),
        ("coarse_signal_panel_a_gate.json", "panel-a"),
        ("coarse_signal_panel_b_gate.json", "panel-b"),
        ("coarse_signal_witness_gate.json", "witness"),
    ):
        gate = _load_json(root / filename, f"coarse witness {gate_name} gate")
        _require(
            gate.get("schema") == spec.gate_schema
            and gate.get("schema_version") == 1
            and gate.get("evaluation_status") == "evaluated"
            and gate.get("gate") == gate_name
            and gate.get("passed") == 1,
            f"coarse witness {gate_name} gate changed",
        )
        _scope_is_closed(gate, f"coarse witness {gate_name} gate")
        gate_table[gate_name] = {"passed": 1, "schema": spec.gate_schema}

    seals: dict[str, dict[str, Any]] = {}
    panel_table: dict[str, dict[str, Any]] = {}
    for panel_spec in spec.panels:
        path_ids, means = _load_panel_arrays(root, panel_spec)
        seal = _verify_panel_seal(root, panel_spec, path_ids, means, spec)
        seals[panel_spec.panel] = seal
        panel_table[panel_spec.panel] = {
            "path_count": len(panel_spec.path_ids),
            "path_id_first": panel_spec.path_ids[0],
            "path_id_last": panel_spec.path_ids[-1],
            "shape": list(panel_spec.shape),
            "dtype": means.dtype.str,
            "file_sha256": panel_spec.file_sha256,
            "array_sha256": panel_spec.array_sha256,
            "panel_fingerprint": panel_spec.panel_fingerprint,
            "seal_file_sha256": panel_spec.seal_file_sha256,
        }

    joint_path = root / "joint_analysis_seal.json"
    _require(
        file_fingerprint(joint_path) == spec.joint_seal_file_sha256,
        "coarse witness joint seal file hash changed",
    )
    joint = _load_json(joint_path, "coarse witness joint seal")
    _require(
        joint.get("schema") == f"{spec.run_schema}-joint-analysis-seal"
        and joint.get("schema_version") == 1
        and joint.get("panel_a_seal_sha256") == config_fingerprint(seals["a"])
        and joint.get("panel_b_seal_sha256") == config_fingerprint(seals["b"])
        and joint.get("panel_a_file_sha256") == spec.panels[0].file_sha256
        and joint.get("panel_b_file_sha256") == spec.panels[1].file_sha256
        and joint.get("statistic_plan_sha256") == spec.statistic_plan_sha256
        and joint.get("analysis_definition_frozen_before_open") == 1
        and joint.get("analysis_open_count") == 0,
        "coarse witness joint analysis seal changed",
    )
    _scope_is_closed(joint, "coarse witness joint seal")

    tree_rows = _verified_tree_rows(rows, root, spec)
    return _hashed(
        {
            "schema": f"{SCHEMA}-coarse-witness-parent",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "parent_basename": spec.basename,
            "registry": {
                "schema": spec.registry_schema,
                "record_count": spec.registry_record_count,
                "semantic_sha256": spec.registry_semantic_sha256,
                "file_sha256": spec.registry_file_sha256,
            },
            "source_fingerprint": spec.source_fingerprint,
            "scientific_config_sha256": spec.config_semantic_sha256,
            "path_plan_sha256": spec.path_plan_sha256,
            "statistic_plan_sha256": spec.statistic_plan_sha256,
            "terminal": {
                "state": "completed",
                "stage": "analyze",
                "decision": spec.terminal_decision,
            },
            "gates": gate_table,
            "panels": panel_table,
            "joint_seal_file_sha256": spec.joint_seal_file_sha256,
            "all_registered_artifact_hashes_verified": 1,
            "all_panel_hashes_verified": 1,
            "file_count": len(tree_rows),
            "total_bytes": sum(int(row["size"]) for row in tree_rows),
            "tree_sha256": config_fingerprint(tree_rows),
            "root_independent_identity": 1,
            "parent_files_modified": 0,
        }
    )


def load_verified_coarse_witness_panels(
    run_dir: str | Path,
) -> CoarseWitnessPanels:
    """Verify the parent, then return read-only copies of both panel arrays."""

    verify_coarse_witness_run(run_dir)
    root = Path(run_dir).resolve()
    loaded = {
        panel.panel: _load_panel_arrays(root, panel)
        for panel in COARSE_WITNESS_SPEC.panels
    }
    return CoarseWitnessPanels(
        panel_a_path_ids=loaded["a"][0],
        panel_a=loaded["a"][1],
        panel_b_path_ids=loaded["b"][0],
        panel_b=loaded["b"][1],
    )


def snapshot_portable_result_archive(zip_path: str | Path) -> dict[str, Any]:
    """Return a root-independent byte snapshot of the portable ZIP."""

    path = Path(zip_path)
    _require(path.is_file() and not path.is_symlink(), "portable archive is missing")
    _require(path.name == PORTABLE_RESULT_SPEC.basename, "wrong portable archive basename")
    return _hashed(
        {
            "schema": f"{SCHEMA}-portable-archive-snapshot",
            "schema_version": SCHEMA_VERSION,
            "archive_basename": path.name,
            "size": int(path.stat().st_size),
            "sha256": file_fingerprint(path),
        }
    )


def _snapshot_rows(root: Path) -> list[dict[str, Any]]:
    files = _tree_files(root)
    return [
        {
            "path": relative,
            "size": int(path.stat().st_size),
            "sha256": file_fingerprint(path),
        }
        for relative, path in sorted(files.items())
    ]


def snapshot_coarse_witness_run(run_dir: str | Path) -> dict[str, Any]:
    """Hash every witness file by relative path without recording its root."""

    root = Path(run_dir)
    _require(root.is_dir() and not root.is_symlink(), "coarse witness run is missing")
    root = root.resolve()
    _require(root.name == COARSE_WITNESS_SPEC.basename, "wrong coarse witness run basename")
    rows = _snapshot_rows(root)
    return _hashed(
        {
            "schema": f"{SCHEMA}-coarse-witness-tree-snapshot",
            "schema_version": SCHEMA_VERSION,
            "parent_basename": root.name,
            "file_count": len(rows),
            "total_bytes": sum(int(row["size"]) for row in rows),
            "files": rows,
            "tree_sha256": config_fingerprint(rows),
        }
    )


def _validate_snapshot(snapshot: Mapping[str, Any], schema: str) -> None:
    _assert_semantic(snapshot, "parent snapshot")
    _require(
        snapshot.get("schema") == schema and snapshot.get("schema_version") == 1,
        "parent snapshot schema changed",
    )


def compare_portable_result_snapshots(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    schema = f"{SCHEMA}-portable-archive-snapshot"
    _validate_snapshot(before, schema)
    _validate_snapshot(after, schema)
    _require(dict(before) == dict(after), "immutable portable archive snapshot changed")
    return _hashed(
        {
            "schema": f"{SCHEMA}-portable-archive-comparison",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "archive_sha256": before["sha256"],
            "parent_files_modified": 0,
        }
    )


def compare_coarse_witness_snapshots(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    schema = f"{SCHEMA}-coarse-witness-tree-snapshot"
    _validate_snapshot(before, schema)
    _validate_snapshot(after, schema)
    _require(dict(before) == dict(after), "immutable coarse witness snapshot changed")
    return _hashed(
        {
            "schema": f"{SCHEMA}-coarse-witness-tree-comparison",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "file_count": before["file_count"],
            "tree_sha256": before["tree_sha256"],
            "parent_files_modified": 0,
        }
    )


def snapshot_absolute_coordinate_parents(
    *, portable_zip_path: str | Path, coarse_witness_run_dir: str | Path
) -> dict[str, Any]:
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-snapshots",
            "schema_version": SCHEMA_VERSION,
            "portable_directional": snapshot_portable_result_archive(portable_zip_path),
            "coarse_witness": snapshot_coarse_witness_run(coarse_witness_run_dir),
        }
    )


def verify_absolute_coordinate_parent_immutability(
    *,
    portable_zip_path: str | Path,
    coarse_witness_run_dir: str | Path,
    snapshots: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_semantic(snapshots, "absolute-coordinate parent snapshots")
    _require(
        snapshots.get("schema") == f"{SCHEMA}-parent-snapshots"
        and snapshots.get("schema_version") == 1,
        "absolute-coordinate parent snapshot contract changed",
    )
    portable = snapshots.get("portable_directional")
    coarse = snapshots.get("coarse_witness")
    _require(
        isinstance(portable, Mapping) and isinstance(coarse, Mapping),
        "absolute-coordinate parent snapshots are missing",
    )
    portable_after = snapshot_portable_result_archive(portable_zip_path)
    coarse_after = snapshot_coarse_witness_run(coarse_witness_run_dir)
    portable_comparison = compare_portable_result_snapshots(portable, portable_after)
    coarse_comparison = compare_coarse_witness_snapshots(coarse, coarse_after)
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-immutability",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "portable_directional": portable_comparison,
            "coarse_witness": coarse_comparison,
            "parent_files_modified": 0,
        }
    )


def verify_absolute_coordinate_parents(
    *,
    portable_zip_path: str | Path,
    coarse_witness_run_dir: str | Path,
    snapshots: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return complete root-independent bindings for both immutable parents."""

    portable = verify_portable_result_archive(portable_zip_path)
    coarse = verify_coarse_witness_run(coarse_witness_run_dir)
    immutability = None
    if snapshots is not None:
        immutability = verify_absolute_coordinate_parent_immutability(
            portable_zip_path=portable_zip_path,
            coarse_witness_run_dir=coarse_witness_run_dir,
            snapshots=snapshots,
        )
    return _hashed(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "provenance_valid": 1,
            "portable_directional_parent_valid": 1,
            "coarse_witness_parent_valid": 1,
            "parents": {
                "portable_directional": portable,
                "coarse_witness": coarse,
            },
            "parent_immutability": immutability,
            "parent_files_modified": 0,
            **{field: 0 for field in _ZERO_SCOPE_FIELDS},
        }
    )


__all__ = [
    "AbsoluteCoordinateProvenanceError",
    "COARSE_PANEL_A",
    "COARSE_PANEL_B",
    "COARSE_WITNESS_BASENAME",
    "COARSE_WITNESS_CONFIG_SHA256",
    "COARSE_WITNESS_DECISION",
    "COARSE_WITNESS_REGISTRY_FILE_SHA256",
    "COARSE_WITNESS_REGISTRY_SEMANTIC_SHA256",
    "COARSE_WITNESS_SOURCE_FINGERPRINT",
    "COARSE_WITNESS_SPEC",
    "CoarsePanelSpec",
    "CoarseWitnessPanels",
    "CoarseWitnessSpec",
    "PORTABLE_RESULT_ARCHIVE_SHA256",
    "PORTABLE_RESULT_BASENAME",
    "PORTABLE_RESULT_CONFIG_SHA256",
    "PORTABLE_RESULT_DECISION",
    "PORTABLE_RESULT_MANIFEST_SHA256",
    "PORTABLE_RESULT_REGISTRY_FILE_SHA256",
    "PORTABLE_RESULT_REGISTRY_SEMANTIC_SHA256",
    "PORTABLE_RESULT_SPEC",
    "PortableResultSpec",
    "compare_coarse_witness_snapshots",
    "compare_portable_result_snapshots",
    "load_verified_coarse_witness_panels",
    "snapshot_absolute_coordinate_parents",
    "snapshot_coarse_witness_run",
    "snapshot_portable_result_archive",
    "verify_absolute_coordinate_parent_immutability",
    "verify_absolute_coordinate_parents",
    "verify_coarse_witness_run",
    "verify_portable_result_archive",
]
