"""Provenance and allocation contracts for the frequency-one learner.

This module is deliberately additive.  It verifies the immutable diagnostic
and protocol parents, delegates their transitive checks to the existing
verifiers, freezes the fresh path/seed/cohort plans, and provides the source
closure and resume checks used by the workflow.  It never opens historical
training or validation arrays and never creates a transition.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
import mnist.d0_jacobi_rb_absolute_coordinate_provenance as _absolute
import mnist.d0_jacobi_rb_boundary_tangent_v3_time_local_provenance as _time_local
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    v3_transitive_source_paths,
    verify_v3_source_image_binding,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-frequency1-coordinate-provenance"
SCHEMA_VERSION = 1
PROVENANCE_VERSION = "d0-jacobi-rb-boundary-tangent-frequency1-coordinate-provenance-v1"

# Direct scientific-design parent.
ABSOLUTE_PARENT_BASENAME = (
    "20260810-211949_production-read-only-absolute-coordinate-adjudication"
)
ABSOLUTE_PARENT_SCHEMA = (
    "experiment12-d0-jacobi-rb-absolute-coordinate-adjudication"
)
ABSOLUTE_PARENT_DECISION = "absolute_coordinate_representation_hypothesis_supported"
ABSOLUTE_PARENT_REGISTRY_COUNT = 40
ABSOLUTE_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "77f7245cd52b8d4e210f75ceb17f2bf3c0b923bbbabde26173e9409a0d6d9218"
)
ABSOLUTE_PARENT_REGISTRY_FILE_SHA256 = (
    "d05804f468c6485c234a6f6f66d55e3c3075e85a9172efbd1af2ab5654a84158"
)
ABSOLUTE_PARENT_SOURCE_FINGERPRINT = (
    "2a9ba3b2b5078665aaadcab036185f3b5ae8798e1672bc372ff08d32714a707f"
)
ABSOLUTE_PARENT_CONFIG_SHA256 = (
    "8c8eb7e9ee7bf6251dbfc478a7bf2c409b4cbfb7a8f3e55523a68d56040345dc"
)
ABSOLUTE_PARENT_DECISION_FILE_SHA256 = (
    "8f219f69f3057f176db89396ac8c75cb0dc998bb68e19462e112fed7cd997d63"
)
ABSOLUTE_PARENT_COORDINATE_LATTICE_SHA256 = (
    "d85791569d8236f36c424b16b6b2832e405420f5664af109bc4369528e423ba6"
)
ABSOLUTE_PARENT_COORDINATE_BASIS_SHA256 = (
    "5fcaf84ca50d523fd750c5b1fe3f30e3464f0a8ef0883590dded812912b7f9c6"
)
ABSOLUTE_PARENT_HELDOUT_CSV_SHA256 = (
    "4ebd62c0a558b64f1bc8f35cbe4b77357b764628c69193c8e8d2574dd3af61ad"
)
ABSOLUTE_PARENT_PANEL_A_DIRECTION_SHA256 = (
    "bb2cba87f32b67d19da023b2676059223ae38b4f37458b1dfa1944a3f7b8a7c4"
)
ABSOLUTE_PARENT_MAX_T_CRITICAL_VALUE = 2.8891755766437894
ABSOLUTE_PARENT_MAX_RECONSTRUCTION_ERROR = 1.214306433183765e-17
ABSOLUTE_PARENT_COARSE_POINT_ESTIMATE = 0.000648424870102139
ABSOLUTE_PARENT_COARSE_LOWER_BOUND = 0.0005095880255077374
ABSOLUTE_PARENT_FREQUENCY1_POINT_ENERGIES = (
    0.0007060147771907059,
    0.00034895730851380234,
    0.00018907354036646135,
    0.00008189432657901031,
)
ABSOLUTE_PARENT_FREQUENCY1_LOWER_BOUNDS = (
    0.0006381522766084302,
    0.00028943218513485374,
    0.0001397448905623646,
    0.00005043911793781375,
)

# Re-export the authoritative direct-parent bindings rather than restating
# them.  The absolute-coordinate verifier owns these constants.
COARSE_WITNESS_BASENAME = _absolute.COARSE_WITNESS_BASENAME
COARSE_WITNESS_REGISTRY_SEMANTIC_SHA256 = (
    _absolute.COARSE_WITNESS_REGISTRY_SEMANTIC_SHA256
)
COARSE_WITNESS_REGISTRY_FILE_SHA256 = _absolute.COARSE_WITNESS_REGISTRY_FILE_SHA256
COARSE_WITNESS_SOURCE_FINGERPRINT = _absolute.COARSE_WITNESS_SOURCE_FINGERPRINT
COARSE_WITNESS_CONFIG_SHA256 = _absolute.COARSE_WITNESS_CONFIG_SHA256
COARSE_WITNESS_TREE_SHA256 = (
    "39eb12f8565900e1688e9daf8fb9f14861c91ea2b799b3e373b86be846f686cf"
)

PORTABLE_DIRECTIONAL_BASENAME = _absolute.PORTABLE_RESULT_BASENAME
PORTABLE_DIRECTIONAL_ARCHIVE_SHA256 = _absolute.PORTABLE_RESULT_ARCHIVE_SHA256
PORTABLE_DIRECTIONAL_REGISTRY_SEMANTIC_SHA256 = (
    _absolute.PORTABLE_RESULT_REGISTRY_SEMANTIC_SHA256
)
PORTABLE_DIRECTIONAL_REGISTRY_FILE_SHA256 = (
    _absolute.PORTABLE_RESULT_REGISTRY_FILE_SHA256
)
PORTABLE_DIRECTIONAL_CONFIG_SHA256 = _absolute.PORTABLE_RESULT_CONFIG_SHA256
PORTABLE_DIRECTIONAL_DECISION = _absolute.PORTABLE_RESULT_DECISION

# Prospective protocol parent.  These are imported from its authoritative
# read-only adjudication verifier.
MEMORY_V3_PARENT_BASENAME = _time_local.MEMORY_PARENT_BASENAME
MEMORY_V3_PARENT_SCHEMA = _time_local.MEMORY_PARENT_SCHEMA
MEMORY_V3_PARENT_DECISION = _time_local.MEMORY_PARENT_DECISION
MEMORY_V3_PARENT_REGISTRY_COUNT = _time_local.MEMORY_PARENT_REGISTRY_COUNT
MEMORY_V3_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    _time_local.MEMORY_PARENT_REGISTRY_SEMANTIC_SHA256
)
MEMORY_V3_PARENT_REGISTRY_FILE_SHA256 = _time_local.MEMORY_PARENT_REGISTRY_FILE_SHA256
MEMORY_V3_PARENT_SOURCE_FINGERPRINT = _time_local.MEMORY_PARENT_SOURCE_FINGERPRINT
MEMORY_V3_PARENT_CONFIG_SHA256 = _time_local.MEMORY_PARENT_CONFIG_SHA256

# The directional archive verifier closes this negative-design ancestor.
QUARTILE_SPECIALIST_PARENT_BASENAME = (
    "20260807-132351_production-exact-quartile-specialist"
)
QUARTILE_SPECIALIST_PARENT_DECISION = "no_training_only_quartile_system"
QUARTILE_SPECIALIST_PARENT_REGISTRY_COUNT = 4_120
QUARTILE_SPECIALIST_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "e5f6b3ee257b3d4f86ec3ac54f4223540cf76caa24489d22e9c138a54e19c7bb"
)
QUARTILE_SPECIALIST_PARENT_REGISTRY_FILE_SHA256 = (
    "e24c7db28081dbceb8f0acf690d779f835379c82a89a2b263976c3e0b631f798"
)
QUARTILE_SPECIALIST_PARENT_SOURCE_FINGERPRINT = (
    "61a8c2fd6a317c05b9eed73e208d40b0cc6c01d6fdd227ae33d71d0be7c6027d"
)
QUARTILE_SPECIALIST_PARENT_CONFIG_SHA256 = (
    "05263b7b01c2586e9a771bd71fe28fbb74d8e5d9da873ce4746019c5167c08c1"
)

PATH_ID_BITS = 20
PATH_ID_LIMIT = 1 << PATH_ID_BITS
PATH_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-frequency1-coordinate-path-ids-v1"
PATH_ROLE_RANGES: dict[str, tuple[int, int]] = {
    "preflight_seam": (0xF8000, 0xF8008),
    "training": (0xF8100, 0xF8140),
    "validation": (0xF8200, 0xF8220),
    "confirmation": (0xF9000, 0xF9040),
}
# Known realized/reserved history called out by the reviewed plan.  External
# scans are still required; this interval is the independent static guard.
HISTORICAL_FORBIDDEN_RANGES: dict[str, tuple[int, int]] = {
    "known_e_and_f_history": (0xE0000, 0xF7180),
}

COHORT_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-frequency1-coordinate-cohorts-v1"
TRAIN_VALIDATION_COHORT_SIZES = (10,) * 9 + (6,)
CONFIRMATION_COHORT_SIZES = (10,) * 6 + (4,)

ROOT_SEED = 261_371
PHYSICAL_MODEL_SEEDS = (261_372, 261_373, 261_374)
SELECTION_BOOTSTRAP_SEED = 261_380
SELECTION_BOOTSTRAP_NAMESPACE = 0x46435631
FORBIDDEN_SCHEDULER_SEED = 261_381
CONFIRMATION_BOOTSTRAP_SEED = 261_382
CONFIRMATION_BOOTSTRAP_NAMESPACE = 0x46434331
SYNTHETIC_COORDINATE_TEACHER_SEED = 261_383
EXACT_MODEL_NULL_SEED = 261_384
INITIALIZATION_BASIS_CONTROL_SEED = 261_385
RESERVED_FUTURE_CONTROL_SEED = 261_386
ROLE_OPEN_ORDER = ("training", "validation", "confirmation")


class FrequencyOneCoordinateProvenanceError(ArtifactCompatibilityError):
    """An immutable parent, allocation, source, or resume binding changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrequencyOneCoordinateProvenanceError(message)


def _semantic_body(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.pop("semantic_sha256", None)
    return result


def _hashed(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _assert_semantic(record: Mapping[str, Any], description: str) -> None:
    _require(
        record.get("semantic_sha256") == config_fingerprint(_semantic_body(record)),
        f"{description} semantic hash changed",
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrequencyOneCoordinateProvenanceError(
            f"invalid {description}: {path}"
        ) from exc
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


def _safe_registered_path(root: Path, value: Any) -> tuple[str, Path]:
    _require(isinstance(value, str) and bool(value), "registry path is invalid")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe registry path: {value!r}",
    )
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise FrequencyOneCoordinateProvenanceError(
            f"registered path escapes immutable parent: {value!r}"
        ) from exc
    return relative.as_posix(), target


def _snapshot_tree(root: Path, *, role: str) -> dict[str, Any]:
    root = root.resolve()
    _require(root.is_dir(), f"{role} parent does not exist: {root}")
    rows = []
    files: list[Path] = []
    for item in root.rglob("*"):
        resolved = item.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise FrequencyOneCoordinateProvenanceError(
                f"{role} parent entry escapes immutable tree: "
                f"{item.relative_to(root).as_posix()}"
            ) from exc
        if item.is_file():
            files.append(item)
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_fingerprint(path),
            }
        )
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-tree-snapshot",
            "schema_version": SCHEMA_VERSION,
            "role": role,
            "basename": root.name,
            "file_count": len(rows),
            "total_bytes": sum(int(row["size"]) for row in rows),
            "tree_sha256": config_fingerprint(rows),
            "files": rows,
        }
    )


def _snapshot_archive(path: Path, *, role: str) -> dict[str, Any]:
    _require(path.is_file(), f"{role} archive does not exist: {path}")
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-archive-snapshot",
            "schema_version": SCHEMA_VERSION,
            "role": role,
            "basename": path.name,
            "size": path.stat().st_size,
            "sha256": file_fingerprint(path),
        }
    )


def _snapshot_selected_files(
    root: Path, *, role: str, relatives: Sequence[str]
) -> dict[str, Any]:
    _require(root.is_dir(), f"{role} parent does not exist: {root}")
    rows = []
    for relative in sorted(relatives):
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise FrequencyOneCoordinateProvenanceError(
                f"{role} snapshot path escapes parent: {relative}"
            ) from exc
        _require(target.is_file(), f"missing {role} snapshot file: {relative}")
        rows.append(
            {
                "path": PurePosixPath(relative).as_posix(),
                "size": target.stat().st_size,
                "sha256": file_fingerprint(target),
            }
        )
    return _hashed(
        {
            "schema": f"{SCHEMA}-selected-parent-files-snapshot",
            "schema_version": SCHEMA_VERSION,
            "role": role,
            "basename": root.name,
            "file_count": len(rows),
            "tree_sha256": config_fingerprint(rows),
            "files": rows,
        }
    )


def snapshot_frequency1_coordinate_parents(
    *,
    absolute_coordinate_run_dir: str | Path,
    memory_v3_run_dir: str | Path,
    coarse_witness_run_dir: str | Path,
    portable_directional_archive: str | Path,
) -> dict[str, Any]:
    """Return complete content-addressed snapshots of all explicit parents."""

    memory_root = Path(memory_v3_run_dir).resolve()
    cache_binding = _load_json(
        memory_root / "immutable_cache_binding.json", "memory-v3 cache binding"
    )
    external_cache_parent = Path(str(cache_binding.get("parent_run_dir", ""))).resolve()
    source_image_parent = _resolve_source_image_parent(memory_root)
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-snapshots",
            "schema_version": SCHEMA_VERSION,
            "absolute_coordinate": _snapshot_tree(
                Path(absolute_coordinate_run_dir).resolve(),
                role="absolute_coordinate",
            ),
            "memory_v3": _snapshot_tree(
                Path(memory_v3_run_dir).resolve(), role="memory_v3"
            ),
            "coarse_witness": _absolute.snapshot_coarse_witness_run(
                coarse_witness_run_dir
            ),
            "portable_directional": _absolute.snapshot_portable_result_archive(
                portable_directional_archive
            ),
            # The memory verifier reads every registered cache-parent artifact.
            "memory_v3_external_cache_parent": _snapshot_tree(
                external_cache_parent, role="memory_v3_external_cache_parent"
            ),
            # The source verifier reads exactly these two immutable payloads.
            "source_image_parent": _snapshot_selected_files(
                source_image_parent,
                role="source_image_parent",
                relatives=("source_image.json", "source_image.npz"),
            ),
        }
    )


def verify_frequency1_coordinate_parent_immutability(
    *,
    absolute_coordinate_run_dir: str | Path,
    memory_v3_run_dir: str | Path,
    coarse_witness_run_dir: str | Path,
    portable_directional_archive: str | Path,
    snapshots: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_semantic(snapshots, "frequency-one parent snapshots")
    expected = snapshot_frequency1_coordinate_parents(
        absolute_coordinate_run_dir=absolute_coordinate_run_dir,
        memory_v3_run_dir=memory_v3_run_dir,
        coarse_witness_run_dir=coarse_witness_run_dir,
        portable_directional_archive=portable_directional_archive,
    )
    _require(dict(snapshots) == expected, "immutable parent snapshot changed")
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-immutability",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "parent_files_modified": 0,
            "snapshot_sha256": expected["semantic_sha256"],
        }
    )


def compare_frequency1_coordinate_parent_snapshots(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two root-independent explicit-parent snapshots."""

    _assert_semantic(before, "frequency-one parent snapshot before")
    _assert_semantic(after, "frequency-one parent snapshot after")
    _require(dict(before) == dict(after), "immutable parent snapshot changed")
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-snapshot-comparison",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "snapshot_sha256": before["semantic_sha256"],
            "parent_files_modified": 0,
        }
    )


def _verify_absolute_registry(root: Path) -> dict[str, Any]:
    registry_path = root / "artifact_registry.json"
    _require(
        registry_path.is_file()
        and file_fingerprint(registry_path) == ABSOLUTE_PARENT_REGISTRY_FILE_SHA256,
        "absolute-coordinate registry file hash changed",
    )
    registry = _load_json(registry_path, "absolute-coordinate registry")
    artifacts = registry.get("artifacts")
    _require(
        registry.get("schema") == f"{ABSOLUTE_PARENT_SCHEMA}-artifact-registry"
        and registry.get("schema_version") == 1
        and isinstance(artifacts, list)
        and registry.get("artifact_count") == ABSOLUTE_PARENT_REGISTRY_COUNT
        and len(artifacts) == ABSOLUTE_PARENT_REGISTRY_COUNT
        and registry.get("semantic_sha256")
        == ABSOLUTE_PARENT_REGISTRY_SEMANTIC_SHA256
        and config_fingerprint(_semantic_body(registry))
        == ABSOLUTE_PARENT_REGISTRY_SEMANTIC_SHA256,
        "absolute-coordinate registry semantics changed",
    )
    seen: set[str] = set()
    for raw in artifacts:
        _require(isinstance(raw, Mapping), "absolute-coordinate registry row malformed")
        relative, target = _safe_registered_path(root, raw.get("path"))
        _require(relative not in seen, "absolute-coordinate registry path duplicated")
        _require(
            target.is_file()
            and raw.get("size") == target.stat().st_size
            and raw.get("sha256") == file_fingerprint(target),
            f"absolute-coordinate artifact changed: {relative}",
        )
        seen.add(relative)
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    _require(
        actual == seen | {"artifact_registry.json"},
        "absolute-coordinate terminal file set changed",
    )
    return registry


def verify_absolute_coordinate_design_parent(run_dir: str | Path) -> dict[str, Any]:
    """Verify the terminal representation-hypothesis run byte-for-byte."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"absolute-coordinate parent does not exist: {root}")
    _require(root.name == ABSOLUTE_PARENT_BASENAME, "wrong absolute-coordinate basename")
    _verify_absolute_registry(root)

    manifest = _load_json(root / "run_manifest.json", "absolute-coordinate manifest")
    config = _load_json(root / "scientific_config.json", "absolute-coordinate config")
    status = _load_json(root / "run_status.json", "absolute-coordinate status")
    decision_path = root / "absolute_coordinate_adjudication_decision.json"
    decision = _load_json(decision_path, "absolute-coordinate decision")
    _assert_semantic(config, "absolute-coordinate config")
    _require(
        manifest.get("schema") == f"{ABSOLUTE_PARENT_SCHEMA}-manifest"
        and manifest.get("source_fingerprint") == ABSOLUTE_PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256") == ABSOLUTE_PARENT_CONFIG_SHA256
        and config.get("semantic_sha256") == ABSOLUTE_PARENT_CONFIG_SHA256,
        "absolute-coordinate source/config binding changed",
    )
    _require(
        status.get("state") == "complete"
        and status.get("stage") == "report"
        and status.get("decision") == ABSOLUTE_PARENT_DECISION
        and int(status.get("scientific_evidence_complete", 0)) == 1
        and decision.get("decision") == ABSOLUTE_PARENT_DECISION
        and int(decision.get("scientific_evidence_complete", 0)) == 1
        and int(decision.get("fresh_coordinate_learner_plan_drafting_recommended", 0))
        == 1
        and file_fingerprint(decision_path) == ABSOLUTE_PARENT_DECISION_FILE_SHA256,
        "absolute-coordinate terminal decision changed",
    )
    for record, description in ((status, "status"), (decision, "decision")):
        for field in (
            "physical_training_performed",
            "confirmation_performed",
            "controller_trajectories_executed",
            "reconstructions_created",
            "sampling_performed",
            "reverse_sampling_performed",
            "parent_files_modified",
        ):
            _require(int(record.get(field, -1)) == 0, f"absolute parent {description} records {field}")

    lattice = _load_json(root / "coordinate_lattice.json", "coordinate lattice")
    _assert_semantic(lattice, "coordinate lattice")
    _require(
        lattice.get("semantic_sha256") == ABSOLUTE_PARENT_COORDINATE_LATTICE_SHA256
        and lattice.get("basis_sha256") == ABSOLUTE_PARENT_COORDINATE_BASIS_SHA256
        and lattice.get("frequency1_rank") == 4
        and lattice.get("phase_count") == 7
        and lattice.get("edges_per_phase") == 392,
        "absolute-coordinate lattice binding changed",
    )
    heldout_path = root / "frequency1_heldout_inference.csv"
    _require(
        file_fingerprint(heldout_path) == ABSOLUTE_PARENT_HELDOUT_CSV_SHA256,
        "held-out frequency-one table hash changed",
    )
    with heldout_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == 4, "held-out frequency-one table shape changed")
    point = tuple(float(row["signed_cross_panel_energy"]) for row in rows)
    lower = tuple(float(row["signed_cross_simultaneous_lower_bound"]) for row in rows)
    _require(
        tuple(row["quartile"] for row in rows) == ("q0", "q1", "q2", "q3")
        and point == ABSOLUTE_PARENT_FREQUENCY1_POINT_ENERGIES
        and lower == ABSOLUTE_PARENT_FREQUENCY1_LOWER_BOUNDS
        and all(int(row["simultaneously_positive"]) == 1 for row in rows),
        "held-out frequency-one evidence changed",
    )
    _require(
        file_fingerprint(root / "panel_a_frequency1_directions.npz")
        == ABSOLUTE_PARENT_PANEL_A_DIRECTION_SHA256,
        "panel-A frequency-one directions changed",
    )
    inference = _load_json(root / "panel_b_linear_inference.json", "panel-B inference")
    decomposition = _load_json(root / "coordinate_decomposition.json", "coordinate decomposition")
    coarse = _load_json(root / "coarse_witness_replay.json", "coarse witness replay")
    _require(
        inference.get("inference", {}).get("critical_value")
        == ABSOLUTE_PARENT_MAX_T_CRITICAL_VALUE
        and tuple(inference.get("scaled_signed_cross_lower_bounds", ()))
        == ABSOLUTE_PARENT_FREQUENCY1_LOWER_BOUNDS
        and decomposition.get("maximum_reconstruction_error")
        == ABSOLUTE_PARENT_MAX_RECONSTRUCTION_ERROR
        and coarse.get("point_estimate") == ABSOLUTE_PARENT_COARSE_POINT_ESTIMATE
        and coarse.get("bootstrap_lower_bound") == ABSOLUTE_PARENT_COARSE_LOWER_BOUND
        and float(coarse.get("bootstrap_lower_bound", 0.0)) > 0.0,
        "absolute-coordinate design evidence changed",
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-absolute-coordinate-parent",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "run_dir": str(root),
            "registry_count": ABSOLUTE_PARENT_REGISTRY_COUNT,
            "registry_semantic_sha256": ABSOLUTE_PARENT_REGISTRY_SEMANTIC_SHA256,
            "source_fingerprint": ABSOLUTE_PARENT_SOURCE_FINGERPRINT,
            "scientific_config_sha256": ABSOLUTE_PARENT_CONFIG_SHA256,
            "decision": ABSOLUTE_PARENT_DECISION,
            "frequency1_point_energies": list(point),
            "frequency1_lower_bounds": list(lower),
            "historical_design_evidence_authorizing": 0,
            "all_registered_artifact_hashes_verified": 1,
        }
    )


def _verify_memory_v3_parent(
    run_dir: str | Path, *, verify_external_cache: bool
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    _require(root.is_dir() and root.name == MEMORY_V3_PARENT_BASENAME, "wrong memory-v3 parent")
    before = _time_local.snapshot_parent_run(root)
    rows = _time_local._snapshot_rows(root)  # authoritative verifier input
    try:
        result = _time_local._verify_memory_parent(
            root, rows, verify_external_cache=bool(verify_external_cache)
        )
    except ArtifactCompatibilityError as exc:
        raise FrequencyOneCoordinateProvenanceError(str(exc)) from exc
    after = _time_local.snapshot_parent_run(root)
    _require(before == after, "memory-v3 parent changed during verification")
    return result


def _resolve_source_image_parent(memory_v3_run_dir: str | Path) -> Path:
    memory = Path(memory_v3_run_dir).resolve()
    binding = _load_json(memory / "immutable_cache_binding.json", "immutable cache binding")
    parent = Path(str(binding.get("parent_run_dir", ""))).resolve()
    provenance = _load_json(parent / "parent_provenance.json", "v3 parent provenance")
    coarse = provenance.get("parents", {}).get("coarse_residual", {})
    root = Path(str(coarse.get("run_dir", ""))).resolve()
    _require(root.is_dir(), "coarse-residual source-image parent is unavailable")
    return root


def verify_frequency1_source_image_binding(
    memory_v3_run_dir: str | Path,
) -> dict[str, Any]:
    """Reconstruct the exact source tensor through the protocol parent chain."""

    root = _resolve_source_image_parent(memory_v3_run_dir)
    try:
        return verify_v3_source_image_binding(root)
    except ArtifactCompatibilityError as exc:
        raise FrequencyOneCoordinateProvenanceError(str(exc)) from exc


def verify_frequency1_coordinate_parents(
    *,
    absolute_coordinate_run_dir: str | Path,
    memory_v3_run_dir: str | Path,
    coarse_witness_run_dir: str | Path,
    portable_directional_archive: str | Path,
    snapshots: Mapping[str, Any] | None = None,
    verify_external_cache: bool = True,
) -> dict[str, Any]:
    """Verify the complete design/protocol chain and immutable snapshots."""

    current_snapshots = snapshot_frequency1_coordinate_parents(
        absolute_coordinate_run_dir=absolute_coordinate_run_dir,
        memory_v3_run_dir=memory_v3_run_dir,
        coarse_witness_run_dir=coarse_witness_run_dir,
        portable_directional_archive=portable_directional_archive,
    )
    expected_snapshots = current_snapshots if snapshots is None else snapshots
    absolute_design = verify_absolute_coordinate_design_parent(
        absolute_coordinate_run_dir
    )
    direct = _absolute.verify_absolute_coordinate_parents(
        portable_zip_path=portable_directional_archive,
        coarse_witness_run_dir=coarse_witness_run_dir,
        snapshots=_absolute.snapshot_absolute_coordinate_parents(
            portable_zip_path=portable_directional_archive,
            coarse_witness_run_dir=coarse_witness_run_dir,
        ),
    )
    memory = _verify_memory_v3_parent(
        memory_v3_run_dir, verify_external_cache=verify_external_cache
    )
    source_image = verify_frequency1_source_image_binding(memory_v3_run_dir)
    allocation_scan = scan_historical_path_seed_claims(
        run_dirs=(
            absolute_coordinate_run_dir,
            memory_v3_run_dir,
            coarse_witness_run_dir,
        ),
        archives=(portable_directional_archive,),
    )
    path_validation = validate_path_id_plan(
        build_path_id_plan(),
        claimed_ids={"verified_parent_scan": allocation_scan["path_ids"]},
    )
    seed_validation = validate_seed_plan(
        build_seed_plan(),
        historical_seeds=allocation_scan["seeds"],
        historical_namespaces=allocation_scan["namespaces"],
    )
    immutability = verify_frequency1_coordinate_parent_immutability(
        absolute_coordinate_run_dir=absolute_coordinate_run_dir,
        memory_v3_run_dir=memory_v3_run_dir,
        coarse_witness_run_dir=coarse_witness_run_dir,
        portable_directional_archive=portable_directional_archive,
        snapshots=expected_snapshots,
    )
    return _hashed(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "provenance_valid": 1,
            "parents": {
                "absolute_coordinate_design": absolute_design,
                "memory_v3_protocol": memory,
                "coarse_witness": direct["parents"]["coarse_witness"],
                "portable_directional": direct["parents"]["portable_directional"],
            },
            "source_image_binding": source_image,
            "historical_allocation_scan": allocation_scan,
            "path_plan_validation": path_validation,
            "seed_plan_validation": seed_validation,
            "parent_snapshots": current_snapshots,
            "parent_immutability": immutability,
            "all_parent_registries_verified": 1,
            "all_registered_parent_artifact_hashes_verified": 1,
            "quartile_specialist_transitive_binding_verified": 1,
            "time_local_and_memory_v3_verifiers_composed": 1,
            "historical_checkpoints_reused": 0,
            "historical_validation_or_confirmation_labels_opened": 0,
            "parent_files_modified": 0,
        }
    )


def build_path_id_plan() -> dict[str, Any]:
    roles = {
        role: list(range(start, stop))
        for role, (start, stop) in PATH_ROLE_RANGES.items()
    }
    return _hashed(
        {
            "schema": f"{SCHEMA}-path-id-plan",
            "schema_version": SCHEMA_VERSION,
            "path_id_plan_version": PATH_PLAN_VERSION,
            "canonical_path_id_bits": PATH_ID_BITS,
            "roles": roles,
            "role_slots": {
                role: {
                    "start": start,
                    "stop_exclusive": stop,
                    "path_count": stop - start,
                    "opening_rule": (
                        "preflight_only"
                        if role == "preflight_seam"
                        else "cache_stage"
                        if role in {"training", "validation"}
                        else "sealed_nonzero_nominee_only"
                    ),
                }
                for role, (start, stop) in PATH_ROLE_RANGES.items()
            },
            "historical_forbidden_ranges": {
                name: {"start": start, "stop_exclusive": stop}
                for name, (start, stop) in HISTORICAL_FORBIDDEN_RANGES.items()
            },
            "automatic_relocation_authorized": 0,
        }
    )


def _claim_values(value: Any) -> set[int]:
    if isinstance(value, Mapping):
        if "start" in value and "stop_exclusive" in value:
            value = range(int(value["start"]), int(value["stop_exclusive"]))
        elif "path_ids" in value:
            value = value["path_ids"]
    elif (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        value = range(value[0], value[1])
    try:
        raw_values = tuple(value)
    except TypeError as exc:
        raise FrequencyOneCoordinateProvenanceError("path claim is not iterable") from exc
    result: set[int] = set()
    for raw in raw_values:
        _require(
            isinstance(raw, int) and not isinstance(raw, bool),
            "path claim is not an integer",
        )
        _require(0 <= raw < PATH_ID_LIMIT, "path claim is outside 20-bit bounds")
        _require(raw not in result, "path claim contains a duplicate ID")
        result.add(raw)
    return result


def validate_path_id_plan(
    plan: Mapping[str, Any],
    *,
    claimed_ids: Mapping[str, Any] | Iterable[int] | None = None,
) -> dict[str, Any]:
    expected = build_path_id_plan()
    _require(dict(plan) == expected, "frequency-one path plan changed")
    active_by_role = {
        role: set(range(start, stop))
        for role, (start, stop) in PATH_ROLE_RANGES.items()
    }
    active = set().union(*active_by_role.values())
    roles = tuple(active_by_role)
    for index, role in enumerate(roles):
        start, stop = PATH_ROLE_RANGES[role]
        _require(0 <= start < stop <= PATH_ID_LIMIT, f"{role} path range invalid")
        for other in roles[index + 1 :]:
            _require(active_by_role[role].isdisjoint(active_by_role[other]), "path roles overlap")
    historical = {
        value
        for start, stop in HISTORICAL_FORBIDDEN_RANGES.values()
        for value in range(start, stop)
    }
    _require(active.isdisjoint(historical), "new path IDs collide with frozen history")
    claims: Mapping[str, Any]
    if claimed_ids is None:
        claims = {}
    elif isinstance(claimed_ids, Mapping):
        claims = claimed_ids
    else:
        claims = {"external": claimed_ids}
    collisions = []
    for source, raw in claims.items():
        collisions.extend(
            {"source": str(source), "path_id": value}
            for value in sorted(active & _claim_values(raw))
        )
    _require(not collisions, "frequency-one path claim collision")
    return _hashed(
        {
            "schema": f"{SCHEMA}-path-id-plan-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "path_id_plan_sha256": expected["semantic_sha256"],
            "active_path_count": len(active),
            "role_disjointness_pass": 1,
            "historical_disjointness_pass": 1,
            "twenty_bit_bounds_pass": 1,
            "collision_count": 0,
        }
    )


def _collect_numeric_claims(
    value: Any,
    *,
    key: str = "",
    path_ids: set[int],
    seeds: set[int],
    namespaces: set[int],
) -> None:
    """Collect only explicitly named path/seed/namespace JSON claims."""

    lowered = key.casefold()
    if isinstance(value, Mapping):
        if "path_id_first" in value and "path_id_last" in value:
            first, last = value["path_id_first"], value["path_id_last"]
            if (
                isinstance(first, int)
                and not isinstance(first, bool)
                and isinstance(last, int)
                and not isinstance(last, bool)
                and 0 <= first <= last < PATH_ID_LIMIT
            ):
                path_ids.update(range(first, last + 1))
        for child_key, child in value.items():
            child_claim_key = str(child_key)
            if lowered in {"roles", "path_roles"} and isinstance(child, list):
                child_claim_key = "path_ids"
            elif (
                lowered in {"role_slots", "path_ranges", "historical_forbidden_roles"}
                and isinstance(child, Mapping)
                and "start" in child
                and "stop_exclusive" in child
            ):
                start, stop = child["start"], child["stop_exclusive"]
                if (
                    isinstance(start, int)
                    and not isinstance(start, bool)
                    and isinstance(stop, int)
                    and not isinstance(stop, bool)
                    and 0 <= start <= stop <= PATH_ID_LIMIT
                ):
                    path_ids.update(range(start, stop))
            _collect_numeric_claims(
                child,
                key=child_claim_key,
                path_ids=path_ids,
                seeds=seeds,
                namespaces=namespaces,
            )
        return
    if isinstance(value, list):
        if lowered.endswith("path_ids"):
            path_ids.update(
                int(item)
                for item in value
                if isinstance(item, int) and not isinstance(item, bool)
            )
        elif "seed" in lowered:
            seeds.update(
                int(item)
                for item in value
                if isinstance(item, int) and not isinstance(item, bool)
            )
        elif "namespace" in lowered:
            namespaces.update(
                int(item)
                for item in value
                if isinstance(item, int) and not isinstance(item, bool)
            )
        else:
            for child in value:
                _collect_numeric_claims(
                    child,
                    key=key,
                    path_ids=path_ids,
                    seeds=seeds,
                    namespaces=namespaces,
                )
        return
    if not isinstance(value, int) or isinstance(value, bool):
        return
    if lowered == "path_id" or lowered.endswith("_path_id"):
        path_ids.add(int(value))
    elif "seed" in lowered and not lowered.endswith("sha256"):
        seeds.add(int(value))
    elif "namespace" in lowered and not lowered.endswith("sha256"):
        namespaces.add(int(value))


def _claim_json_name(name: str) -> bool:
    base = PurePosixPath(name).name.casefold()
    return base in {
        "scientific_config.json",
        "path_id_plan.json",
        "path_plan.json",
        "seed_plan.json",
        "confirmation_plan.json",
        "parent_provenance.json",
    } or "opening" in base or "panel" in base and "seal" in base


def scan_historical_path_seed_claims(
    *,
    run_dirs: Iterable[str | Path],
    archives: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Recursively scan the verified parents' allocation-bearing JSON files."""

    path_ids: set[int] = set()
    seeds: set[int] = set()
    namespaces: set[int] = set()
    scanned: list[str] = []
    for raw_root in run_dirs:
        root = Path(raw_root).resolve()
        _require(root.is_dir(), f"historical scan root does not exist: {root}")
        for path in sorted(root.rglob("*.json"), key=lambda item: item.as_posix()):
            if not _claim_json_name(path.name):
                continue
            record = _load_json(path, f"historical allocation record {path}")
            _collect_numeric_claims(
                record,
                path_ids=path_ids,
                seeds=seeds,
                namespaces=namespaces,
            )
            scanned.append(f"{root.name}/{path.relative_to(root).as_posix()}")
    for raw_archive in archives:
        archive_path = Path(raw_archive).resolve()
        _require(archive_path.is_file(), f"historical scan archive missing: {archive_path}")
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                for info in sorted(archive.infolist(), key=lambda row: row.filename):
                    if info.is_dir() or not _claim_json_name(info.filename):
                        continue
                    payload = archive.read(info)
                    record = json.loads(payload.decode("utf-8"))
                    _require(isinstance(record, Mapping), "historical archive claim is not an object")
                    _collect_numeric_claims(
                        record,
                        path_ids=path_ids,
                        seeds=seeds,
                        namespaces=namespaces,
                    )
                    scanned.append(f"{archive_path.name}!/{info.filename}")
        except (OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise FrequencyOneCoordinateProvenanceError(
                f"cannot scan historical allocation archive: {archive_path}"
            ) from exc
    return _hashed(
        {
            "schema": f"{SCHEMA}-historical-allocation-scan",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "scanned_records": scanned,
            "scanned_record_count": len(scanned),
            "path_ids": sorted(path_ids),
            "seeds": sorted(seeds),
            "namespaces": sorted(namespaces),
        }
    )


def _partition(values: Sequence[int], sizes: Sequence[int]) -> list[list[int]]:
    _require(sum(sizes) == len(values), "cohort sizes do not partition path IDs")
    result: list[list[int]] = []
    cursor = 0
    for size in sizes:
        result.append(list(values[cursor : cursor + size]))
        cursor += size
    return result


def build_cohort_plan(path_plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    plan = build_path_id_plan() if path_plan is None else dict(path_plan)
    validate_path_id_plan(plan)
    train_validation = plan["roles"]["training"] + plan["roles"]["validation"]
    tv_cohorts = _partition(train_validation, TRAIN_VALIDATION_COHORT_SIZES)
    confirmation = _partition(
        plan["roles"]["confirmation"], CONFIRMATION_COHORT_SIZES
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-cohort-plan",
            "schema_version": SCHEMA_VERSION,
            "cohort_plan_version": COHORT_PLAN_VERSION,
            "path_id_plan_sha256": plan["semantic_sha256"],
            "maximum_exact_generation_cohort_size": 10,
            "preflight": [plan["roles"]["preflight_seam"]],
            "train_validation": tv_cohorts,
            "confirmation": confirmation,
            "train_validation_sizes": list(TRAIN_VALIDATION_COHORT_SIZES),
            "confirmation_sizes": list(CONFIRMATION_COHORT_SIZES),
            "cross_role_cohort_indices": [6],
            "split_by_role_before_persistence": 1,
        }
    )


def validate_cohort_plan(
    cohort_plan: Mapping[str, Any], *, path_plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    plan = build_path_id_plan() if path_plan is None else dict(path_plan)
    expected = build_cohort_plan(plan)
    _require(dict(cohort_plan) == expected, "frequency-one cohort plan changed")
    tv = [value for cohort in expected["train_validation"] for value in cohort]
    confirm = [value for cohort in expected["confirmation"] for value in cohort]
    _require(
        tv == plan["roles"]["training"] + plan["roles"]["validation"]
        and confirm == plan["roles"]["confirmation"]
        and all(1 <= len(row) <= 10 for row in expected["train_validation"])
        and all(1 <= len(row) <= 10 for row in expected["confirmation"]),
        "frequency-one cohort ordering changed",
    )
    crossing = expected["train_validation"][6]
    _require(
        crossing[:4] == plan["roles"]["training"][-4:]
        and crossing[4:] == plan["roles"]["validation"][:6],
        "mixed train/validation cohort changed",
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-cohort-plan-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "cohort_plan_sha256": expected["semantic_sha256"],
            "maximum_cohort_size": 10,
            "mixed_cohort_index": 6,
            "split_before_persistence": 1,
        }
    )


def build_seed_plan() -> dict[str, Any]:
    return _hashed(
        {
            "schema": f"{SCHEMA}-seed-plan",
            "schema_version": SCHEMA_VERSION,
            "root_physical_path_seed": ROOT_SEED,
            "physical_model_seeds": list(PHYSICAL_MODEL_SEEDS),
            "selection_bootstrap": {
                "seed": SELECTION_BOOTSTRAP_SEED,
                "namespace": SELECTION_BOOTSTRAP_NAMESPACE,
            },
            "forbidden_scheduler_seed": FORBIDDEN_SCHEDULER_SEED,
            "confirmation_bootstrap": {
                "seed": CONFIRMATION_BOOTSTRAP_SEED,
                "namespace": CONFIRMATION_BOOTSTRAP_NAMESPACE,
            },
            "synthetic_coordinate_teacher_seed": SYNTHETIC_COORDINATE_TEACHER_SEED,
            "exact_model_null_seed": EXACT_MODEL_NULL_SEED,
            "initialization_basis_control_seed": INITIALIZATION_BASIS_CONTROL_SEED,
            "reserved_future_control_seed": RESERVED_FUTURE_CONTROL_SEED,
            "automatic_renumbering_authorized": 0,
        }
    )


def validate_seed_plan(
    plan: Mapping[str, Any], *, historical_seeds: Iterable[int] | None = None,
    historical_namespaces: Iterable[int] | None = None,
) -> dict[str, Any]:
    expected = build_seed_plan()
    _require(dict(plan) == expected, "frequency-one seed plan changed")
    seeds = (
        ROOT_SEED,
        *PHYSICAL_MODEL_SEEDS,
        SELECTION_BOOTSTRAP_SEED,
        FORBIDDEN_SCHEDULER_SEED,
        CONFIRMATION_BOOTSTRAP_SEED,
        SYNTHETIC_COORDINATE_TEACHER_SEED,
        EXACT_MODEL_NULL_SEED,
        INITIALIZATION_BASIS_CONTROL_SEED,
        RESERVED_FUTURE_CONTROL_SEED,
    )
    _require(len(seeds) == len(set(seeds)), "frequency-one seeds collide")
    _require(
        SELECTION_BOOTSTRAP_NAMESPACE != CONFIRMATION_BOOTSTRAP_NAMESPACE,
        "frequency-one bootstrap namespaces collide",
    )
    historical = set()
    for raw in historical_seeds or ():
        _require(isinstance(raw, int) and not isinstance(raw, bool), "historical seed malformed")
        historical.add(raw)
    _require(set(seeds).isdisjoint(historical), "frequency-one seed collides with history")
    old_namespaces = set()
    for raw in historical_namespaces or ():
        _require(
            isinstance(raw, int) and not isinstance(raw, bool),
            "historical namespace malformed",
        )
        old_namespaces.add(raw)
    _require(
        {SELECTION_BOOTSTRAP_NAMESPACE, CONFIRMATION_BOOTSTRAP_NAMESPACE}.isdisjoint(
            old_namespaces
        ),
        "frequency-one bootstrap namespace collides with history",
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-seed-plan-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "seed_plan_sha256": expected["semantic_sha256"],
            "unique_seed_count": len(seeds),
            "bootstrap_namespace_disjointness": 1,
            "historical_collision_count": 0,
            "historical_namespace_collision_count": 0,
        }
    )


def validate_role_open_order(
    opened_roles: Iterable[str], *, prerequisite_flags: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    roles = tuple(str(value) for value in opened_roles)
    _require(len(roles) == len(set(roles)), "a role was opened more than once")
    _require(all(role in ROLE_OPEN_ORDER for role in roles), "unknown role-open event")
    indices = tuple(ROLE_OPEN_ORDER.index(role) for role in roles)
    _require(indices == tuple(range(len(indices))), "role-open sequence skipped or reordered")
    prerequisites = {
        "training": "prelabel_controls_passed",
        "validation": "physical_training_complete_and_selection_plan_sealed",
        "confirmation": "nonzero_validation_nominee_sealed",
    }
    if prerequisite_flags is not None:
        for role in roles:
            _require(
                int(prerequisite_flags.get(prerequisites[role], 0)) == 1,
                f"missing role prerequisite: {prerequisites[role]}",
            )
    return _hashed(
        {
            "schema": f"{SCHEMA}-role-open-order-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "opened_roles": list(roles),
            "next_role": ROLE_OPEN_ORDER[len(roles)] if len(roles) < 3 else None,
            "confirmation_opened": int("confirmation" in roles),
        }
    )


def frequency1_coordinate_source_paths(
    entry_points: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    if entry_points is None:
        package = Path(__file__).resolve().parent
        entry_points = tuple(
            package / name
            for name in (
                "d0_jacobi_rb_boundary_tangent_frequency1_coordinate.py",
                "d0_jacobi_rb_boundary_tangent_frequency1_coordinate_provenance.py",
                "d0_jacobi_rb_boundary_tangent_frequency1_coordinate_gate.py",
                "d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability.py",
                "diag_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability.py",
            )
        )
    try:
        return v3_transitive_source_paths(entry_points)
    except ArtifactCompatibilityError as exc:
        raise FrequencyOneCoordinateProvenanceError(str(exc)) from exc


def frequency1_coordinate_source_fingerprint(
    entry_points: Iterable[str | Path] | None = None,
) -> str:
    return source_fingerprint(frequency1_coordinate_source_paths(entry_points))


def scientific_config_fingerprint(record: Mapping[str, Any]) -> str:
    return config_fingerprint(_semantic_body(record))


def validate_semantic_config(
    record: Mapping[str, Any], *, expected_schema: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    _assert_semantic(record, "frequency-one scientific config")
    if expected_schema is not None:
        _require(record.get("schema") == expected_schema, "scientific config schema changed")
    if expected_sha256 is not None:
        _require(record.get("semantic_sha256") == expected_sha256, "scientific config hash changed")
    return _hashed(
        {
            "schema": f"{SCHEMA}-scientific-config-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "scientific_config_sha256": record["semantic_sha256"],
        }
    )


def verify_resume_compatibility(
    run_dir: str | Path,
    *,
    expected_bindings: Mapping[str, Any],
    artifact_bindings: Mapping[str, str | Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Verify immutable manifest/artifact bindings before any resume write."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"resume run does not exist: {root}")
    manifest = _load_json(root / "run_manifest.json", "resume manifest")
    _require(
        all(manifest.get(key) == value for key, value in expected_bindings.items()),
        "resume manifest compatibility changed",
    )
    verified: dict[str, Any] = {}
    for relative, expected in (artifact_bindings or {}).items():
        normalized, path = _safe_registered_path(root, relative)
        _require(
            normalized == PurePosixPath(relative).as_posix(),
            f"resume artifact path is not canonical: {relative}",
        )
        if isinstance(expected, Mapping):
            if "file_sha256" in expected:
                _require(file_fingerprint(path) == expected["file_sha256"], f"resume file changed: {relative}")
            record = _load_json(path, f"resume artifact {relative}")
            if "semantic_sha256" in expected:
                _assert_semantic(record, f"resume artifact {relative}")
                _require(record.get("semantic_sha256") == expected["semantic_sha256"], f"resume semantic artifact changed: {relative}")
        else:
            record = _load_json(path, f"resume artifact {relative}")
            _assert_semantic(record, f"resume artifact {relative}")
            _require(record.get("semantic_sha256") == expected, f"resume artifact changed: {relative}")
        verified[normalized] = expected
    return _hashed(
        {
            "schema": f"{SCHEMA}-resume-compatibility",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "run_dir": str(root),
            "expected_bindings": dict(expected_bindings),
            "verified_artifacts": verified,
        }
    )


# Workflow-friendly aliases.
build_frequency1_path_plan = build_path_id_plan
validate_frequency1_path_plan = validate_path_id_plan
build_frequency1_cohort_plan = build_cohort_plan
validate_frequency1_cohort_plan = validate_cohort_plan
build_frequency1_seed_plan = build_seed_plan
validate_frequency1_seed_plan = validate_seed_plan
verify_parent_runs = verify_frequency1_coordinate_parents
verify_parent_immutability = verify_frequency1_coordinate_parent_immutability


__all__ = [
    "ABSOLUTE_PARENT_BASENAME",
    "ABSOLUTE_PARENT_CONFIG_SHA256",
    "ABSOLUTE_PARENT_DECISION",
    "ABSOLUTE_PARENT_REGISTRY_COUNT",
    "ABSOLUTE_PARENT_REGISTRY_FILE_SHA256",
    "ABSOLUTE_PARENT_REGISTRY_SEMANTIC_SHA256",
    "ABSOLUTE_PARENT_SOURCE_FINGERPRINT",
    "CONFIRMATION_BOOTSTRAP_NAMESPACE",
    "CONFIRMATION_BOOTSTRAP_SEED",
    "CONFIRMATION_COHORT_SIZES",
    "EXACT_MODEL_NULL_SEED",
    "FORBIDDEN_SCHEDULER_SEED",
    "FrequencyOneCoordinateProvenanceError",
    "INITIALIZATION_BASIS_CONTROL_SEED",
    "MEMORY_V3_PARENT_BASENAME",
    "PATH_ID_BITS",
    "PATH_ID_LIMIT",
    "PATH_ROLE_RANGES",
    "PHYSICAL_MODEL_SEEDS",
    "PORTABLE_DIRECTIONAL_BASENAME",
    "PROVENANCE_VERSION",
    "RESERVED_FUTURE_CONTROL_SEED",
    "ROLE_OPEN_ORDER",
    "ROOT_SEED",
    "SCHEMA",
    "SELECTION_BOOTSTRAP_NAMESPACE",
    "SELECTION_BOOTSTRAP_SEED",
    "SYNTHETIC_COORDINATE_TEACHER_SEED",
    "TRAIN_VALIDATION_COHORT_SIZES",
    "build_cohort_plan",
    "build_frequency1_cohort_plan",
    "build_frequency1_path_plan",
    "build_frequency1_seed_plan",
    "build_path_id_plan",
    "build_seed_plan",
    "compare_frequency1_coordinate_parent_snapshots",
    "frequency1_coordinate_source_fingerprint",
    "frequency1_coordinate_source_paths",
    "scientific_config_fingerprint",
    "scan_historical_path_seed_claims",
    "snapshot_frequency1_coordinate_parents",
    "validate_cohort_plan",
    "validate_frequency1_cohort_plan",
    "validate_frequency1_path_plan",
    "validate_frequency1_seed_plan",
    "validate_path_id_plan",
    "validate_role_open_order",
    "validate_seed_plan",
    "validate_semantic_config",
    "verify_absolute_coordinate_design_parent",
    "verify_frequency1_coordinate_parent_immutability",
    "verify_frequency1_coordinate_parents",
    "verify_frequency1_source_image_binding",
    "verify_parent_immutability",
    "verify_parent_runs",
    "verify_resume_compatibility",
]
