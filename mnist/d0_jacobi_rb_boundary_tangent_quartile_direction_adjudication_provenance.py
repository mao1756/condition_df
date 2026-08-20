"""Read-only provenance for quartile direction adjudication.

This module binds the completed quartile-specialist run and exposes the only
permitted way for the adjudication to read its two historical evidence roles.
It deliberately contains no cache-generation, role-opening, training,
selection, confirmation, controller, reconstruction, or sampling code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint as _source_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import (
    load_eager_role_inputs as _load_eager_role_inputs,
    load_eager_role_labels as _load_eager_role_labels,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_provenance import (
    PHYSICAL_MODEL_SEEDS,
    build_path_id_plan,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    v3_transitive_source_paths,
)
from mnist.d0_jacobi_rb_learnability import state_dict_sha256


SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-quartile-"
    "direction-adjudication-provenance"
)
SCHEMA_VERSION = 1

PARENT_BASENAME = "20260807-132351_production-exact-quartile-specialist"
PARENT_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-quartile-specialist"
)
PARENT_TERMINAL_DECISION = "no_training_only_quartile_system"
PARENT_ARTIFACT_COUNT = 4_120
PARENT_TREE_FILE_COUNT = 4_124
PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "e5f6b3ee257b3d4f86ec3ac54f4223540cf76caa24489d22e9c138a54e19c7bb"
)
PARENT_REGISTRY_FILE_SHA256 = (
    "e24c7db28081dbceb8f0acf690d779f835379c82a89a2b263976c3e0b631f798"
)
PARENT_SOURCE_FINGERPRINT = (
    "61a8c2fd6a317c05b9eed73e208d40b0cc6c01d6fdd227ae33d71d0be7c6027d"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "05263b7b01c2586e9a771bd71fe28fbb74d8e5d9da873ce4746019c5167c08c1"
)
GAIN_TABLE_FILE_SHA256 = (
    "48ec1f17be4869f9a816c0338e8b23cddbdf44dd7000ca60fe317fd787925815"
)
TRAINING_RANK_PATH_TABLES_FILE_SHA256 = (
    "93f5c4ea39bc658cc5f46b7d31930a0c8c02b2c7ccb106cf314229f0eec32d9b"
)
CHECKPOINT_INDEX_FILE_SHA256 = (
    "6446cec12529f5634870c43eb349c3a43b9e1b64f0850c04c680b13c1c749d2b"
)
CHECKPOINT_INDEX_SEMANTIC_SHA256 = (
    "c4112fa6c971bac1ca3b0da471c8915a955a1ee760529cd948530963e38e77c7"
)
GAIN_CALIBRATION_SEAL_FILE_SHA256 = (
    "a165b1d3c601625ebd058cf67ee564ede751ec5356496c6fdb0c7c8e4094e189"
)
RANK_LABEL_OPEN_FILE_SHA256 = (
    "9eac05c28339202fafbcf5abdf00e4040679f9087ad929729dbd085736e6e1b6"
)

# Exact commitments used by the stricter structural checks below.
GAIN_LABEL_OPEN_FILE_SHA256 = (
    "a51e9dff10cab8df564afb107d42c86409ed087c80b10916f613e014dd74c213"
)
FIT_LABEL_OPEN_FILE_SHA256 = (
    "87a094df4d8ed79b95bdf8e1acb727772b68cc9fae3c8fbc5f0300443178abec"
)
GAIN_ROLE_OPEN_SEMANTIC_SHA256 = (
    "61080a841b8d5c31bb755d6a92ae6f461617b9a857b4e7e2b06d57027349d2cb"
)
RANK_ROLE_OPEN_SEMANTIC_SHA256 = (
    "84d2413ba5529fbdcc9c6ee65e04178b07a06d9b31440bcd1a6d4d1a21510732"
)
GAIN_CACHE_BINDING_FILE_SHA256 = (
    "e8a6492e1d32646d5c6aaeb169497de8d5b3c295897686b69fdbaec273b4409e"
)
RANK_CACHE_BINDING_FILE_SHA256 = (
    "f4032586b181f9a2a8c935e36cf145a26483ec45e2934300aa97516e1aff88b1"
)
GAIN_CACHE_BINDING_SEMANTIC_SHA256 = (
    "dd92addc270df65ff6dfc32e364e434bc3017fad3d71cb5c4ef318828a99e399"
)
RANK_CACHE_BINDING_SEMANTIC_SHA256 = (
    "0c993c7c695ba52def1198bdc8206d866c38beb2455adee3f86e60150396300b"
)
GAIN_ROLE_INDEX_FILE_SHA256 = (
    "592d88a544717e3f185c8589a25545b2668e804d476d63f55c26dba9559a5e0c"
)
RANK_ROLE_INDEX_FILE_SHA256 = (
    "ed895e728650847719010e92bab24244451c7922292f8824d1a423266943fa99"
)
GAIN_ROLE_INDEX_SEMANTIC_SHA256 = (
    "14a2d18444667a56b4a1d29a03782d5d9bcfc5b760762ed739835e8c232bd9ac"
)
RANK_ROLE_INDEX_SEMANTIC_SHA256 = (
    "b44743c2bf9446f5d116928c2caa082f7adcbf967521c11f4d9248c584f4bb84"
)

ROLE_ORDER = (
    "physical_fit",
    "gain_calibration",
    "training_rank",
    "fresh_selection",
    "untouched_confirmation",
)
PERMITTED_HISTORICAL_ROLES = ("gain_calibration", "training_rank")
ROLE_ROW_COUNT = 57_344
ROLE_PATH_COUNT = 32
SELECTED_OUTER_STEPS = tuple(range(15, 512, 16))
CHECKPOINT_UPDATES = tuple(range(0, 4_001, 100))
CHECKPOINT_COUNT = 492
NONZERO_CHECKPOINT_COUNT = 480

_ROLE_OPEN_FILES = {
    "physical_fit": "fit_label_open.json",
    "gain_calibration": "gain_label_open.json",
    "training_rank": "rank_label_open.json",
    "fresh_selection": "selection_open.json",
    "untouched_confirmation": "confirmation_open.json",
}
_ROLE_OPEN_FILE_HASHES = {
    "physical_fit": FIT_LABEL_OPEN_FILE_SHA256,
    "gain_calibration": GAIN_LABEL_OPEN_FILE_SHA256,
    "training_rank": RANK_LABEL_OPEN_FILE_SHA256,
}
_ROLE_OPEN_SEMANTIC_HASHES = {
    "gain_calibration": GAIN_ROLE_OPEN_SEMANTIC_SHA256,
    "training_rank": RANK_ROLE_OPEN_SEMANTIC_SHA256,
}
_CACHE_SPECS = {
    "gain_calibration": {
        "binding_file_sha256": GAIN_CACHE_BINDING_FILE_SHA256,
        "binding_semantic_sha256": GAIN_CACHE_BINDING_SEMANTIC_SHA256,
        "index_file_sha256": GAIN_ROLE_INDEX_FILE_SHA256,
        "index_semantic_sha256": GAIN_ROLE_INDEX_SEMANTIC_SHA256,
    },
    "training_rank": {
        "binding_file_sha256": RANK_CACHE_BINDING_FILE_SHA256,
        "binding_semantic_sha256": RANK_CACHE_BINDING_SEMANTIC_SHA256,
        "index_file_sha256": RANK_ROLE_INDEX_FILE_SHA256,
        "index_semantic_sha256": RANK_ROLE_INDEX_SEMANTIC_SHA256,
    },
}
_TERMINAL_EXCLUDED_HASHES = {
    "artifact_registry.json": PARENT_REGISTRY_FILE_SHA256,
    "quartile_specialist_decision.json": (
        "d28fba7efd7663de975957df65920ce0187ea68ee5c8da0ac4c977fc0644ed49"
    ),
    "run_status.json": (
        "7437ace86a3e3d773fcd257f03a3354ea6dcabdb37951cc1c96f7e395f078fac"
    ),
    "workflow_gate.json": (
        "dc6b3b445a8f09c7959b4a5b79b72a92441add5155f1b7f1da85fa82bb8a46cc"
    ),
}
_CRITICAL_REGISTERED_HASHES = {
    "gain_table.npz": GAIN_TABLE_FILE_SHA256,
    "training_rank_path_tables.npz": TRAINING_RANK_PATH_TABLES_FILE_SHA256,
    "training_checkpoint_index.json": CHECKPOINT_INDEX_FILE_SHA256,
    "gain_calibration_seal.json": GAIN_CALIBRATION_SEAL_FILE_SHA256,
    "rank_label_open.json": RANK_LABEL_OPEN_FILE_SHA256,
    "gain_label_open.json": GAIN_LABEL_OPEN_FILE_SHA256,
    "fit_label_open.json": FIT_LABEL_OPEN_FILE_SHA256,
    "gain_calibration_cache_binding.json": GAIN_CACHE_BINDING_FILE_SHA256,
    "training_rank_cache_binding.json": RANK_CACHE_BINDING_FILE_SHA256,
}
_FORBIDDEN_EVIDENCE_PATHS = frozenset(
    {
        "selection_open.json",
        "confirmation_open.json",
        "selected_experts.json",
        "selected_system.json",
        "selected_system_seal.json",
        "selection_primary_path_values.npz",
        "selection_local_path_values.npz",
        "selection_path_table.json",
        "selection_local_screen.json",
        "selection_max_t.json",
        "selection_record.json",
        "selection_evidence_index.json",
        "selection_metrics.json",
        "selection_gate.json",
        "selection_artifact_seal.json",
        "confirmation_primary_path_values.npz",
        "confirmation_local_path_values.npz",
        "confirmation_path_table.json",
        "confirmation_local_screen.json",
        "confirmation_max_t.json",
        "confirmation_record.json",
        "confirmation_evidence_index.json",
        "confirmation_metrics.json",
        "confirmation_gate.json",
        "confirmation_artifact_seal.json",
    }
)
_IDENTITY_FIELDS = (
    "sample_key",
    "path_id",
    "outer_step",
    "phase",
    "midpoint_index",
    "midpoint_fraction",
)
_ZERO_SCOPE_FIELDS = (
    "full_dataset_training_performed",
    "selection_paths_opened",
    "confirmation_paths_opened",
    "controller_execution_performed",
    "reverse_controller_trajectory_performed",
    "reconstruction_performed",
    "full_reverse_path_performed",
    "sampling_performed",
    "reverse_sampling_performed",
    "image_sampling_performed",
)
_NONAUTHORIZING_CONFIG_FIELDS = (
    "authorizing",
    "new_role_count",
    "new_path_count",
    "new_seed_count",
    "cache_generation_authorized",
    "training_authorized",
    "selection_authorized",
    "confirmation_authorized",
    "controller_execution_authorized",
    "sampling_authorized",
)


class DirectionAdjudicationProvenanceError(ArtifactCompatibilityError):
    """The immutable parent or a read-only evidence contract changed."""


# Long-form compatibility name used by adjacent provenance modules.
QuartileDirectionAdjudicationProvenanceError = DirectionAdjudicationProvenanceError


@dataclass(frozen=True)
class AlreadyOpenRole:
    """One historically opened role, loaded into immutable host arrays."""

    role: str
    inputs: Mapping[str, np.ndarray]
    labels: Mapping[str, np.ndarray]
    input_index: Mapping[str, Any]
    label_index: Mapping[str, Any]
    binding: Mapping[str, Any]
    role_open: Mapping[str, Any]
    row_identity_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DirectionAdjudicationProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DirectionAdjudicationProvenanceError(
            f"invalid {description}: {path}"
        ) from exc
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


def _safe_relative(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "artifact path is invalid")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe artifact path: {value!r}",
    )
    return relative.as_posix()


def _semantic_body(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.pop("semantic_sha256", None)
    return result


def _assert_semantic(record: Mapping[str, Any], description: str) -> None:
    _require(
        record.get("semantic_sha256") == config_fingerprint(_semantic_body(record)),
        f"{description} semantic hash changed",
    )


def _hashed(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _parent_root(run_dir: str | Path) -> Path:
    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"parent run does not exist: {root}")
    _require(root.name == PARENT_BASENAME, "wrong quartile-specialist parent basename")
    return root


def _snapshot_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        _require(not path.is_symlink(), f"parent contains a symbolic link: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    return rows


def snapshot_parent_run(run_dir: str | Path) -> dict[str, Any]:
    """Hash every parent file by relative path without writing the parent."""

    root = _parent_root(run_dir)
    rows = _snapshot_rows(root)
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-tree-snapshot",
            "schema_version": SCHEMA_VERSION,
            "run_dir": str(root),
            "parent_basename": root.name,
            "file_count": len(rows),
            "total_bytes": sum(int(row["size"]) for row in rows),
            "files": rows,
            "tree_sha256": config_fingerprint(rows),
        }
    )


def _validate_snapshot(snapshot: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    _assert_semantic(snapshot, "parent tree snapshot")
    rows = snapshot.get("files")
    _require(
        snapshot.get("schema") == f"{SCHEMA}-parent-tree-snapshot"
        and int(snapshot.get("schema_version", -1)) == SCHEMA_VERSION
        and snapshot.get("run_dir") == str(root)
        and snapshot.get("parent_basename") == PARENT_BASENAME
        and isinstance(rows, list)
        and int(snapshot.get("file_count", -1)) == len(rows)
        and snapshot.get("tree_sha256") == config_fingerprint(rows),
        "parent tree snapshot contract changed",
    )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        _require(isinstance(row, Mapping), "parent snapshot row is malformed")
        relative = _safe_relative(row.get("path"))
        _require(relative not in seen, "parent snapshot path is duplicated")
        seen.add(relative)
        normalized.append(
            {
                "path": relative,
                "size": int(row.get("size", -1)),
                "sha256": str(row.get("sha256", "")),
            }
        )
    _require(
        int(snapshot.get("total_bytes", -1))
        == sum(int(row["size"]) for row in normalized),
        "parent snapshot byte count changed",
    )
    return normalized


def compare_parent_snapshots(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Require byte-for-byte equality of two complete parent snapshots."""

    before_root = Path(str(before.get("run_dir", ""))).resolve()
    before_rows = _validate_snapshot(before, before_root)
    after_rows = _validate_snapshot(after, before_root)
    _require(before_rows == after_rows, "immutable parent tree snapshot changed")
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-tree-comparison",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "run_dir": str(before_root),
            "file_count": len(before_rows),
            "tree_sha256": config_fingerprint(before_rows),
            "parent_files_modified": 0,
        }
    )


def verify_parent_immutability_snapshot(
    run_dir: str | Path, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    current = snapshot_parent_run(run_dir)
    compare_parent_snapshots(snapshot, current)
    return current


def _snapshot_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["path"]): dict(row) for row in rows}


def _ensure_selection_confirmation_absent(root: Path) -> None:
    for relative in sorted(_FORBIDDEN_EVIDENCE_PATHS):
        _require(
            not (root / relative).exists(),
            f"selection or confirmation evidence appeared: {relative}",
        )
    for relative in (
        "role_caches/fresh_selection",
        "role_caches/untouched_confirmation",
        "bootstrap_maxima/selection",
        "bootstrap_maxima/confirmation",
    ):
        _require(
            not (root / relative).exists(),
            f"selection or confirmation evidence appeared: {relative}",
        )


def _verify_registry(
    root: Path, rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    observed = _snapshot_map(rows)
    _require(
        observed.get("artifact_registry.json", {}).get("sha256")
        == PARENT_REGISTRY_FILE_SHA256,
        "parent registry file hash changed",
    )
    _require(len(observed) == PARENT_TREE_FILE_COUNT, "parent terminal file count changed")
    registry = _load_json(root / "artifact_registry.json", "parent registry")
    _assert_semantic(registry, "parent registry")
    artifacts = registry.get("artifacts")
    _require(
        registry.get("schema") == f"{PARENT_RUN_SCHEMA}-artifact-registry"
        and int(registry.get("schema_version", -1)) == 1
        and int(registry.get("artifact_count", -1)) == PARENT_ARTIFACT_COUNT
        and registry.get("semantic_sha256") == PARENT_REGISTRY_SEMANTIC_SHA256
        and isinstance(artifacts, list)
        and len(artifacts) == PARENT_ARTIFACT_COUNT,
        "parent registry binding changed",
    )
    records: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        _require(isinstance(item, Mapping), "parent registry row is malformed")
        relative = _safe_relative(item.get("path"))
        _require(relative not in records, "parent registry path is duplicated")
        expected = {
            "path": relative,
            "size": int(item.get("size", -1)),
            "sha256": str(item.get("sha256", "")),
        }
        _require(
            observed.get(relative) == expected,
            f"parent registered artifact changed: {relative}",
        )
        records[relative] = dict(item)
    _require(
        set(observed) == set(records) | set(_TERMINAL_EXCLUDED_HASHES),
        "parent terminal file set changed",
    )
    for relative, expected in _TERMINAL_EXCLUDED_HASHES.items():
        _require(
            observed.get(relative, {}).get("sha256") == expected,
            f"parent terminal artifact changed: {relative}",
        )
    for relative, expected in _CRITICAL_REGISTERED_HASHES.items():
        _require(
            records.get(relative, {}).get("sha256") == expected,
            f"parent critical artifact changed: {relative}",
        )
    for field in _ZERO_SCOPE_FIELDS:
        _require(int(registry.get(field, -1)) == 0, f"parent registry records {field}")
    _require(
        int(registry.get("fit_labels_opened", -1)) == 1
        and int(registry.get("physical_training_performed", -1)) == 1,
        "parent completed-evidence flags changed",
    )
    return registry, records


def _verify_terminal_contract(root: Path) -> None:
    manifest = _load_json(root / "run_manifest.json", "parent manifest")
    _assert_semantic(manifest, "parent manifest")
    config = _load_json(root / "scientific_config.json", "parent scientific config")
    _assert_semantic(config, "parent scientific config")
    decision = _load_json(
        root / "quartile_specialist_decision.json", "parent terminal decision"
    )
    status = _load_json(root / "run_status.json", "parent terminal status")
    workflow = _load_json(root / "workflow_gate.json", "parent workflow gate")
    negative = _load_json(root / "no_training_only_system.json", "parent negative seal")
    _assert_semantic(negative, "parent negative seal")
    _require(
        manifest.get("schema") == f"{PARENT_RUN_SCHEMA}-manifest"
        and manifest.get("run_schema") == PARENT_RUN_SCHEMA
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent manifest binding changed",
    )
    _require(
        config.get("schema") == f"{PARENT_RUN_SCHEMA}-scientific-config"
        and config.get("semantic_sha256") == PARENT_SCIENTIFIC_CONFIG_SHA256
        and config.get("target") == "exact_binary64_jacobi_rao_blackwell_raw_label"
        and config.get("target_formula") == "y(1-y)*d_y log k_u(y|x)"
        and int(config.get("grid_size", -1)) == 28
        and float(config.get("alpha", float("nan"))) == 1.0
        and int(config.get("outer_steps", -1)) == 512
        and float(config.get("tau_eff", float("nan"))) == 5e-5,
        "parent scientific configuration changed",
    )
    _require(
        decision.get("decision") == PARENT_TERMINAL_DECISION
        and decision.get("evaluation_status") == "evaluated"
        and int(decision.get("terminal", 0)) == 1
        and int(decision.get("valid_scientific_negative", 0)) == 1
        and int(decision.get("scientific_evidence_complete", 0)) == 1,
        "parent terminal negative changed",
    )
    _require(
        status.get("schema") == f"{PARENT_RUN_SCHEMA}-status"
        and status.get("state") == "gate_failed"
        and status.get("stage") == "calibrate"
        and status.get("decision") == PARENT_TERMINAL_DECISION
        and int(status.get("selection_paths_opened", -1)) == 0
        and int(status.get("confirmation_paths_opened", -1)) == 0,
        "parent terminal status changed",
    )
    _require(
        workflow.get("decision") == decision
        and workflow.get("require_gate") == "calibrate"
        and int(workflow.get("required_gate_pass", -1)) == 0,
        "parent terminal workflow binding changed",
    )
    _require(
        negative.get("schema") == f"{PARENT_RUN_SCHEMA}-no-training-only-system"
        and int(negative.get("selection_paths_opened", -1)) == 0
        and [
            int(row.get("eligible_candidate_count", -1))
            for row in negative.get("per_quartile_diagnostics", ())
            if int(row.get("quartile", -1)) in {1, 2, 3}
        ]
        == [0, 0, 0],
        "parent negative seal changed",
    )


def _verify_stage_seal(
    root: Path, name: str, observed: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    seal = _load_json(root / name, f"parent stage seal {name}")
    _assert_semantic(seal, f"parent stage seal {name}")
    artifacts = seal.get("artifacts")
    _require(
        seal.get("schema") == f"{PARENT_RUN_SCHEMA}-stage-seal"
        and seal.get("seal_name") == name
        and isinstance(artifacts, list),
        f"parent stage seal changed: {name}",
    )
    seen: set[str] = set()
    for item in artifacts:
        _require(isinstance(item, Mapping), f"malformed stage seal row: {name}")
        relative = _safe_relative(item.get("path"))
        _require(relative not in seen, f"duplicated stage seal path: {name}")
        seen.add(relative)
        _require(
            observed.get(relative, {}).get("sha256") == item.get("sha256"),
            f"sealed parent artifact changed: {relative}",
        )
        semantic = item.get("semantic_sha256")
        if semantic is not None:
            record = _load_json(root / relative, f"sealed record {relative}")
            _assert_semantic(record, f"sealed record {relative}")
            _require(
                record.get("semantic_sha256") == semantic,
                f"sealed record semantic hash changed: {relative}",
            )
    return seal


def _verify_stage_history(
    root: Path, observed: Mapping[str, Mapping[str, Any]]
) -> None:
    for name in (
        "preflight_artifact_seal.json",
        "cache_artifact_seal.json",
        "controls_artifact_seal.json",
        "train_artifact_seal.json",
        "gain_calibration_seal.json",
        "calibrate_artifact_seal.json",
    ):
        _verify_stage_seal(root, name, observed)
    for stage, expected in (
        ("preflight", 1),
        ("cache", 1),
        ("controls", 1),
        ("train", 1),
        ("calibrate", 0),
    ):
        gate = _load_json(root / f"{stage}_gate.json", f"parent {stage} gate")
        _require(
            gate.get("schema")
            == f"d0-jacobi-rb-boundary-tangent-quartile-gate-v1-{stage}"
            and gate.get("evaluation_status") == "evaluated"
            and int(gate.get("passed", -1)) == expected
            and int(gate.get("stage_execution_valid", 0)) == 1
            and int(gate.get("inference_valid", 0)) == 1,
            f"parent {stage} gate changed",
        )
    calibrate = _load_json(root / "calibrate_gate.json", "parent calibrate gate")
    _require(
        int(calibrate.get("valid_scientific_negative", 0)) == 1,
        "parent calibrate gate is not a valid scientific negative",
    )


def _verify_parent_immutability_records(root: Path) -> None:
    before = _load_json(
        root / "parent_immutability_before.json", "parent before-immutability record"
    )
    after = _load_json(
        root / "parent_immutability_after.json", "parent after-immutability record"
    )
    _assert_semantic(before, "parent before-immutability record")
    _assert_semantic(after, "parent after-immutability record")
    _require(
        before.get("schema") == f"{PARENT_RUN_SCHEMA}-parent-immutability"
        and after.get("schema") == before.get("schema")
        and before.get("phase") == "before"
        and after.get("phase") == "after"
        and int(before.get("parents_mutated", -1)) == 0
        and int(after.get("parents_mutated", -1)) == 0,
        "existing parent immutability record changed",
    )
    before_body = dict(before)
    after_body = dict(after)
    for value in (before_body, after_body):
        value.pop("semantic_sha256", None)
        value.pop("phase", None)
    _require(
        before_body == after_body,
        "existing parent before/after immutability records disagree",
    )
    provenance = _load_json(root / "parent_provenance.json", "parent provenance")
    _assert_semantic(provenance, "parent provenance")
    _require(
        before.get("parent_provenance_semantic_sha256")
        == provenance.get("semantic_sha256"),
        "parent provenance/immutability binding changed",
    )


def _verify_role_open_history(root: Path) -> dict[str, dict[str, Any]]:
    plan = _load_json(root / "path_id_plan.json", "parent path-id plan")
    _assert_semantic(plan, "parent path-id plan")
    expected_plan = build_path_id_plan()
    _require(plan == expected_plan, "parent path-id plan changed")
    records: dict[str, dict[str, Any]] = {}
    opened: list[str] = []
    for role in ROLE_ORDER:
        path = root / _ROLE_OPEN_FILES[role]
        if not path.is_file():
            continue
        opened.append(role)
        record = _load_json(path, f"parent {role} role-open record")
        _assert_semantic(record, f"parent {role} role-open record")
        _require(
            record.get("schema") == f"{PARENT_RUN_SCHEMA}-role-open"
            and record.get("role") == role
            and record.get("path_ids") == plan["roles"][role]
            and int(record.get("replacement_role_authorized", -1)) == 0,
            f"parent role-open contract changed: {role}",
        )
        if role in _ROLE_OPEN_FILE_HASHES:
            _require(
                file_fingerprint(path) == _ROLE_OPEN_FILE_HASHES[role],
                f"parent role-open file hash changed: {role}",
            )
        if role in _ROLE_OPEN_SEMANTIC_HASHES:
            _require(
                record.get("semantic_sha256") == _ROLE_OPEN_SEMANTIC_HASHES[role],
                f"parent role-open semantic hash changed: {role}",
            )
        prerequisites = record.get("prerequisite_file_sha256")
        _require(isinstance(prerequisites, Mapping), "role-open prerequisites changed")
        for relative, expected in prerequisites.items():
            safe = _safe_relative(relative)
            artifact = root / safe
            _require(
                artifact.is_file() and file_fingerprint(artifact) == expected,
                f"parent role-open prerequisite changed: {safe}",
            )
        records[role] = record
    _require(
        tuple(opened) == ROLE_ORDER[:3],
        "parent role-open history or selection/confirmation absence changed",
    )
    firewall = _load_json(root / "role_firewall.json", "parent role firewall")
    _assert_semantic(firewall, "parent role firewall")
    _require(
        firewall.get("role_open_order") == list(ROLE_ORDER)
        and int(firewall.get("selection_and_confirmation_raw_cache_authorized", -1))
        == 0
        and int(firewall.get("cross_role_cohort_authorized", -1)) == 0
        and int(firewall.get("historical_design_label_reuse_authorized", -1)) == 0,
        "parent role firewall changed",
    )
    return records


def _artifact_path(role_root: Path, reference: Mapping[str, Any]) -> Path:
    relative = _safe_relative(reference.get("path"))
    path = role_root / PurePosixPath(relative)
    _require(path.is_file(), f"role-cache artifact is missing: {relative}")
    _require(
        int(path.stat().st_size) == int(reference.get("size", -1))
        and file_fingerprint(path) == reference.get("sha256"),
        f"role-cache artifact changed: {relative}",
    )
    return path


def _read_identity_blocks(
    role_root: Path, index: Mapping[str, Any], artifact_field: str
) -> dict[str, np.ndarray]:
    blocks: dict[str, list[np.ndarray]] = {name: [] for name in _IDENTITY_FIELDS}
    entries = index.get("entries")
    _require(isinstance(entries, list), "role-cache index entries changed")
    for entry in entries:
        _require(isinstance(entry, Mapping), "role-cache index entry is malformed")
        reference = entry.get(artifact_field)
        if reference is None:
            continue
        _require(isinstance(reference, Mapping), "role-cache artifact reference changed")
        path = _artifact_path(role_root, reference)
        try:
            with np.load(path, allow_pickle=False) as archive:
                _require(
                    all(name in archive.files for name in _IDENTITY_FIELDS),
                    "role-cache identity field is missing",
                )
                for name in _IDENTITY_FIELDS:
                    blocks[name].append(np.ascontiguousarray(archive[name]).reshape(-1))
        except (OSError, ValueError) as exc:
            raise DirectionAdjudicationProvenanceError(
                f"invalid role-cache NPZ: {path}"
            ) from exc
    _require(all(blocks.values()), "role-cache identity artifacts are absent")
    result = {
        name: np.concatenate(values, axis=0) for name, values in blocks.items()
    }
    order = np.argsort(np.asarray(result["sample_key"], dtype=np.int64), kind="stable")
    return {name: np.ascontiguousarray(value[order]) for name, value in result.items()}


def _identity_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in _IDENTITY_FIELDS:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _validate_joined_identities(
    inputs: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    expected_paths: tuple[int, ...],
) -> str:
    for name in _IDENTITY_FIELDS:
        _require(name in inputs and name in labels, f"role identity field missing: {name}")
        _require(
            np.array_equal(inputs[name], labels[name]),
            f"role input/label row identity changed: {name}",
        )
    sample_keys = np.asarray(inputs["sample_key"], dtype=np.int64)
    path_ids = np.asarray(inputs["path_id"], dtype=np.int64)
    outer_steps = np.asarray(inputs["outer_step"], dtype=np.int64)
    phases = np.asarray(inputs["phase"], dtype=np.int64)
    midpoints = np.asarray(inputs["midpoint_index"], dtype=np.int64)
    _require(len(sample_keys) == ROLE_ROW_COUNT, "role row count changed")
    _require(
        len(np.unique(sample_keys)) == ROLE_ROW_COUNT,
        "role sample-key identity changed",
    )
    _require(
        tuple(int(value) for value in np.unique(path_ids)) == expected_paths,
        "role path identities changed",
    )
    _require(
        tuple(int(value) for value in np.unique(outer_steps)) == SELECTED_OUTER_STEPS,
        "role outer-step identities changed",
    )
    _require(
        tuple(int(value) for value in np.unique(phases)) == tuple(range(7))
        and tuple(int(value) for value in np.unique(midpoints)) == tuple(range(8)),
        "role phase/midpoint identities changed",
    )
    cells = np.stack((path_ids, outer_steps, phases, midpoints), axis=1)
    _require(
        len(np.unique(cells, axis=0)) == ROLE_ROW_COUNT,
        "role path/step/cell identity changed",
    )
    return _identity_sha256(inputs)


def _verify_cache_binding(
    root: Path,
    role: str,
    *,
    verify_rows: bool,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    _require(role in PERMITTED_HISTORICAL_ROLES, f"forbidden historical role: {role}")
    spec = _CACHE_SPECS[role]
    binding_path = root / f"{role}_cache_binding.json"
    _require(
        file_fingerprint(binding_path) == spec["binding_file_sha256"],
        f"{role} cache-binding file hash changed",
    )
    binding = _load_json(binding_path, f"parent {role} cache binding")
    _assert_semantic(binding, f"parent {role} cache binding")
    role_root = root / "role_caches" / role
    index_path = role_root / "eager_cache" / "train_index.json"
    _require(
        file_fingerprint(index_path) == spec["index_file_sha256"],
        f"{role} role-index file hash changed",
    )
    index = _load_json(index_path, f"parent {role} role index")
    _assert_semantic(index, f"parent {role} role index")
    expected_paths = tuple(build_path_id_plan()["roles"][role])
    _require(
        binding.get("schema") == f"{PARENT_RUN_SCHEMA}-role-cache-binding"
        and binding.get("role") == role
        and Path(str(binding.get("cache_root", ""))).resolve() == role_root.resolve()
        and binding.get("semantic_sha256") == spec["binding_semantic_sha256"]
        and tuple(binding.get("path_ids", ())) == expected_paths
        and int(binding.get("input_row_count", -1)) == ROLE_ROW_COUNT
        and binding.get("role_index_semantic_sha256")
        == spec["index_semantic_sha256"]
        and int(binding.get("physical_labels_opened", -1)) == 0,
        f"{role} cache binding changed",
    )
    _require(
        index.get("semantic_sha256") == spec["index_semantic_sha256"]
        and index.get("role") == "train"
        and tuple(index.get("path_ids", ())) == expected_paths
        and int(index.get("path_count", -1)) == ROLE_PATH_COUNT
        and int(index.get("input_row_count", -1)) == ROLE_ROW_COUNT
        and int(index.get("label_row_count", -1)) == ROLE_ROW_COUNT
        and tuple(index.get("selected_outer_steps", ())) == SELECTED_OUTER_STEPS
        and int(index.get("branch_input_label_separated", -1)) == 1
        and int(index.get("cross_role_artifact_commit", -1)) == 0,
        f"{role} role index changed",
    )
    metrics = _load_json(
        role_root / "eager_cache" / "train_validation_metrics.json",
        f"parent {role} cache metrics",
    )
    _assert_semantic(metrics, f"parent {role} cache metrics")
    _require(
        metrics.get("semantic_sha256") == binding.get("metrics_semantic_sha256"),
        f"{role} cache metrics binding changed",
    )
    row_sha: str | None = None
    if verify_rows:
        input_ids = _read_identity_blocks(role_root, index, "branch_inputs")
        label_ids = _read_identity_blocks(role_root, index, "branch_labels")
        row_sha = _validate_joined_identities(
            input_ids, label_ids, expected_paths=expected_paths
        )
    return binding, index, row_sha


def _verify_checkpoint_grid(root: Path, *, verify_states: bool) -> dict[str, Any]:
    index_path = root / "training_checkpoint_index.json"
    _require(
        file_fingerprint(index_path) == CHECKPOINT_INDEX_FILE_SHA256,
        "training checkpoint index file hash changed",
    )
    index = _load_json(index_path, "training checkpoint index")
    _assert_semantic(index, "training checkpoint index")
    rows = index.get("checkpoints")
    tasks = index.get("tasks")
    _require(
        index.get("schema") == f"{PARENT_RUN_SCHEMA}-training-checkpoint-index"
        and index.get("semantic_sha256") == CHECKPOINT_INDEX_SEMANTIC_SHA256
        and int(index.get("checkpoint_count", -1)) == CHECKPOINT_COUNT
        and int(index.get("task_count", -1)) == 12
        and int(index.get("all_boundary_checkpoints_exactly_resumable", -1)) == 1
        and isinstance(rows, list)
        and len(rows) == CHECKPOINT_COUNT
        and isinstance(tasks, list)
        and len(tasks) == 12,
        "training checkpoint index changed",
    )
    expected_rows: list[tuple[str, str, int, int, int]] = []
    for quartile, seeds in PHYSICAL_MODEL_SEEDS.items():
        for seed in seeds:
            for update in CHECKPOINT_UPDATES:
                key = f"q{quartile}.seed{seed}.update{update:04d}"
                relative = f"checkpoints/q{quartile}/seed-{seed}/update-{update:04d}.pt"
                expected_rows.append((key, relative, quartile, seed, update))
    _require(len(expected_rows) == CHECKPOINT_COUNT, "checkpoint grid constant changed")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item, expected in zip(rows, expected_rows, strict=True):
        _require(isinstance(item, Mapping), "checkpoint-index row is malformed")
        key, relative, _, _, _ = expected
        _require(
            item.get("candidate_key") == key
            and item.get("checkpoint_path") == relative,
            "checkpoint identity ordering changed",
        )
        path = root / relative
        expected_file_sha = item.get("checkpoint_file_sha256")
        expected_state_sha = item.get("model_state_sha256")
        _require(
            path.is_file()
            and isinstance(expected_file_sha, str)
            and file_fingerprint(path) == expected_file_sha
            and isinstance(expected_state_sha, str),
            f"checkpoint payload/hash changed: {key}",
        )
        indexed[key] = item
    for task, (quartile, seeds) in zip(
        tasks,
        (
            (quartile, seed)
            for quartile, values in PHYSICAL_MODEL_SEEDS.items()
            for seed in values
        ),
        strict=True,
    ):
        _require(isinstance(task, Mapping), "checkpoint task row is malformed")
        seed = int(seeds)
        task_path = root / str(task.get("task_path", ""))
        _require(
            int(task.get("quartile", -1)) == quartile
            and int(task.get("seed", -1)) == seed
            and task_path.is_file()
            and file_fingerprint(task_path) == task.get("task_sha256"),
            "checkpoint task artifact changed",
        )
        task_record = _load_json(task_path, "checkpoint task record")
        _assert_semantic(task_record, "checkpoint task record")
        task_checkpoints = task_record.get("checkpoints")
        _require(
            int(task_record.get("complete", 0)) == 1
            and int(task_record.get("checkpoint_count", -1)) == len(CHECKPOINT_UPDATES)
            and isinstance(task_checkpoints, list)
            and len(task_checkpoints) == len(CHECKPOINT_UPDATES),
            "checkpoint task table changed",
        )
        for row in task_checkpoints:
            update = int(row.get("update", -1))
            key = f"q{quartile}.seed{seed}.update{update:04d}"
            parent_row = indexed.get(key)
            _require(
                parent_row is not None
                and row.get("checkpoint_path") == parent_row.get("checkpoint_path")
                and row.get("checkpoint_file_sha256")
                == parent_row.get("checkpoint_file_sha256")
                and row.get("state_sha256") == parent_row.get("model_state_sha256"),
                "checkpoint index/task state binding changed",
            )
        for path_field, hash_field in (
            ("history_path", "history_sha256"),
            ("progress_path", "progress_sha256"),
        ):
            artifact = root / str(task.get(path_field, ""))
            _require(
                artifact.is_file() and file_fingerprint(artifact) == task.get(hash_field),
                f"checkpoint task {path_field} changed",
            )
    if verify_states:
        import torch

        for key, row in indexed.items():
            path = root / str(row["checkpoint_path"])
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                state = payload["state_dict"]
            except (OSError, KeyError, TypeError, RuntimeError) as exc:
                raise DirectionAdjudicationProvenanceError(
                    f"checkpoint payload cannot be loaded: {key}"
                ) from exc
            expected_state = row["model_state_sha256"]
            _require(
                payload.get("state_sha256") == expected_state
                and state_dict_sha256(state) == expected_state,
                f"checkpoint state hash changed: {key}",
            )
    return index


def verify_parent(
    parent_run_dir: str | Path,
    *,
    snapshot: Mapping[str, Any] | None = None,
    verify_registry: bool = True,
    verify_checkpoint_states: bool = True,
    verify_cache_rows: bool = True,
) -> dict[str, Any]:
    """Verify the exact terminal parent and every registered artifact."""

    root = _parent_root(parent_run_dir)
    _require(bool(verify_registry), "full parent registry verification is required")
    _ensure_selection_confirmation_absent(root)
    if snapshot is None:
        snapshot = snapshot_parent_run(root)
    rows = _validate_snapshot(snapshot, root)
    registry, records = _verify_registry(root, rows)
    observed = _snapshot_map(rows)
    _verify_terminal_contract(root)
    _verify_stage_history(root, observed)
    _verify_parent_immutability_records(root)
    role_records = _verify_role_open_history(root)
    cache_results: dict[str, Any] = {}
    for role in PERMITTED_HISTORICAL_ROLES:
        binding, index, identity_sha = _verify_cache_binding(
            root, role, verify_rows=verify_cache_rows
        )
        cache_results[role] = {
            "binding_semantic_sha256": binding["semantic_sha256"],
            "role_index_semantic_sha256": index["semantic_sha256"],
            "role_open_semantic_sha256": role_records[role]["semantic_sha256"],
            "row_identity_sha256": identity_sha,
            "row_identities_verified": int(verify_cache_rows),
        }
    checkpoint_index = _verify_checkpoint_grid(
        root, verify_states=verify_checkpoint_states
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-verification",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "parent_run_dir": str(root),
            "parent_basename": root.name,
            "decision": PARENT_TERMINAL_DECISION,
            "valid_scientific_negative": 1,
            "artifact_count": int(registry["artifact_count"]),
            "registry_semantic_sha256": registry["semantic_sha256"],
            "registry_file_sha256": records.get(
                "artifact_registry.json", {}
            ).get("sha256", PARENT_REGISTRY_FILE_SHA256),
            "all_registered_artifact_hashes_verified": 1,
            "all_checkpoint_hashes_verified": 1,
            "all_checkpoint_state_hashes_verified": int(verify_checkpoint_states),
            "checkpoint_count": int(checkpoint_index["checkpoint_count"]),
            "nonzero_checkpoint_count": NONZERO_CHECKPOINT_COUNT,
            "cache_bindings_valid": 1,
            "cache_row_identities_verified": int(verify_cache_rows),
            "role_cache_bindings": cache_results,
            "role_open_history_valid": 1,
            "opened_roles": list(ROLE_ORDER[:3]),
            "selection_confirmation_absent": 1,
            "selection_paths_opened": 0,
            "confirmation_paths_opened": 0,
            "existing_parent_immutability_records_valid": 1,
            "parent_tree_sha256": snapshot["tree_sha256"],
            "historical_design_evidence_authorizing": 0,
            "parent_files_modified": 0,
        }
    )


def _tree_metadata(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in sorted(
            (item for item in root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    )


def _freeze_arrays(arrays: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        _require(array.flags.c_contiguous, f"role array is not C-contiguous: {name}")
        array.setflags(write=False)
        frozen[str(name)] = array
    return MappingProxyType(frozen)


def load_already_open_role(
    parent_run_dir: str | Path, role: str
) -> AlreadyOpenRole:
    """Load one sealed role without creating or replacing a role-open record.

    Only ``gain_calibration`` and ``training_rank`` are accepted.  Every
    returned array is C-contiguous and non-writeable.  The complete parent
    path/size/mtime inventory is compared before and after the read as a
    cheap immediate mutation guard; the workflow's full SHA-256 snapshots
    provide the terminal byte-for-byte check.
    """

    root = _parent_root(parent_run_dir)
    _require(role in PERMITTED_HISTORICAL_ROLES, f"forbidden historical role: {role}")
    _ensure_selection_confirmation_absent(root)
    _require(
        (root / _ROLE_OPEN_FILES[role]).is_file(),
        f"historical role was not already opened: {role}",
    )
    before = _tree_metadata(root)
    role_records = _verify_role_open_history(root)
    binding, expected_index, _ = _verify_cache_binding(
        root, role, verify_rows=False
    )
    try:
        inputs, input_index = _load_eager_role_inputs(
            root / "role_caches" / role, "train"
        )
        labels, label_index = _load_eager_role_labels(
            root / "role_caches" / role, "train"
        )
    except Exception as exc:
        if isinstance(exc, DirectionAdjudicationProvenanceError):
            raise
        raise DirectionAdjudicationProvenanceError(
            f"could not load already-open role: {role}"
        ) from exc
    _require(
        input_index == expected_index and label_index == expected_index,
        f"{role} loaded role index changed",
    )
    expected_paths = tuple(build_path_id_plan()["roles"][role])
    row_sha = _validate_joined_identities(
        inputs, labels, expected_paths=expected_paths
    )
    frozen_inputs = _freeze_arrays(inputs)
    frozen_labels = _freeze_arrays(labels)
    after = _tree_metadata(root)
    _require(before == after, "strict role loader observed a parent mutation")
    return AlreadyOpenRole(
        role=role,
        inputs=frozen_inputs,
        labels=frozen_labels,
        input_index=MappingProxyType(dict(input_index)),
        label_index=MappingProxyType(dict(label_index)),
        binding=MappingProxyType(dict(binding)),
        role_open=MappingProxyType(dict(role_records[role])),
        row_identity_sha256=row_sha,
    )


def source_paths(
    entry_points: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Return the transitive local source closure for child manifest binding."""

    if entry_points is None:
        package = Path(__file__).resolve().parent
        entry_points = tuple(
            package / name
            for name in (
                "d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication.py",
                "d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_provenance.py",
                "d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_gate.py",
                "diag_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication.py",
            )
        )
    paths = tuple(Path(path).resolve() for path in entry_points)
    _require(paths and all(path.is_file() for path in paths), "source entry point missing")
    return v3_transitive_source_paths(paths)


def source_fingerprint(
    entry_points: Iterable[str | Path] | None = None,
) -> str:
    return _source_fingerprint(source_paths(entry_points))


def scientific_config_fingerprint(record: Mapping[str, Any]) -> str:
    return config_fingerprint(_semantic_body(record))


def validate_semantic_config(
    record: Mapping[str, Any],
    *,
    expected_schema: str | None = None,
    expected_sha256: str | None = None,
    require_non_authorizing: bool = True,
) -> dict[str, Any]:
    _assert_semantic(record, "direction-adjudication scientific config")
    if expected_schema is not None:
        _require(record.get("schema") == expected_schema, "scientific config schema changed")
    if expected_sha256 is not None:
        _require(
            record.get("semantic_sha256") == expected_sha256,
            "scientific config fingerprint changed",
        )
    if require_non_authorizing:
        for field in _NONAUTHORIZING_CONFIG_FIELDS:
            _require(int(record.get(field, -1)) == 0, f"scientific config authorizes {field}")
        _require(
            int(record.get("historical_design_evidence_only", -1)) == 1,
            "scientific config does not mark historical design evidence",
        )
    return _hashed(
        {
            "schema": f"{SCHEMA}-scientific-config-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "scientific_config_sha256": record["semantic_sha256"],
            "authorizing": 0,
        }
    )


def verify_resume_compatibility(
    run_dir: str | Path,
    *,
    expected_bindings: Mapping[str, Any],
    artifact_bindings: Mapping[str, str | Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Verify immutable child bindings before any resume write."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"resume run does not exist: {root}")
    manifest = _load_json(root / "run_manifest.json", "resume manifest")
    if "semantic_sha256" in manifest:
        _assert_semantic(manifest, "resume manifest")
    _require(
        all(manifest.get(key) == value for key, value in expected_bindings.items()),
        "resume manifest compatibility changed",
    )
    verified: dict[str, Any] = {}
    for raw_relative, expectation in (artifact_bindings or {}).items():
        relative = _safe_relative(raw_relative)
        path = root / PurePosixPath(relative)
        _require(path.is_file(), f"resume artifact is missing: {relative}")
        if isinstance(expectation, Mapping):
            expected_file = expectation.get("file_sha256")
            expected_semantic = expectation.get("semantic_sha256")
        else:
            expected_file = None if path.suffix == ".json" else expectation
            expected_semantic = expectation if path.suffix == ".json" else None
        if expected_file is not None:
            _require(
                file_fingerprint(path) == expected_file,
                f"resume artifact file hash changed: {relative}",
            )
        if expected_semantic is not None:
            record = _load_json(path, f"resume artifact {relative}")
            _assert_semantic(record, f"resume artifact {relative}")
            _require(
                record.get("semantic_sha256") == expected_semantic,
                f"resume artifact semantic hash changed: {relative}",
            )
        verified[relative] = dict(expectation) if isinstance(expectation, Mapping) else expectation
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


# Descriptive aliases used by workflow and tests.
verify_quartile_specialist_parent = verify_parent
verify_quartile_direction_adjudication_parent = verify_parent
direction_adjudication_source_paths = source_paths
direction_adjudication_source_fingerprint = source_fingerprint
quartile_direction_adjudication_source_paths = source_paths
quartile_direction_adjudication_source_fingerprint = source_fingerprint
verify_direction_adjudication_resume_compatibility = verify_resume_compatibility

# Explicit parent-prefixed aliases make call sites self-documenting.
QUARTILE_SPECIALIST_PARENT_BASENAME = PARENT_BASENAME
QUARTILE_SPECIALIST_PARENT_REGISTRY_COUNT = PARENT_ARTIFACT_COUNT
QUARTILE_SPECIALIST_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    PARENT_REGISTRY_SEMANTIC_SHA256
)
QUARTILE_SPECIALIST_PARENT_REGISTRY_FILE_SHA256 = PARENT_REGISTRY_FILE_SHA256
QUARTILE_SPECIALIST_PARENT_SOURCE_FINGERPRINT = PARENT_SOURCE_FINGERPRINT
QUARTILE_SPECIALIST_PARENT_CONFIG_SHA256 = PARENT_SCIENTIFIC_CONFIG_SHA256
PARENT_REGISTRY_COUNT = PARENT_ARTIFACT_COUNT
PARENT_CONFIG_SHA256 = PARENT_SCIENTIFIC_CONFIG_SHA256
GAIN_TABLE_SHA256 = GAIN_TABLE_FILE_SHA256
TRAINING_RANK_PATH_TABLES_SHA256 = TRAINING_RANK_PATH_TABLES_FILE_SHA256
TRAINING_CHECKPOINT_INDEX_SHA256 = CHECKPOINT_INDEX_FILE_SHA256
GAIN_CALIBRATION_SEAL_SHA256 = GAIN_CALIBRATION_SEAL_FILE_SHA256
RANK_LABEL_OPEN_SHA256 = RANK_LABEL_OPEN_FILE_SHA256


__all__ = [
    "AlreadyOpenRole",
    "CHECKPOINT_COUNT",
    "CHECKPOINT_INDEX_FILE_SHA256",
    "CHECKPOINT_INDEX_SEMANTIC_SHA256",
    "DirectionAdjudicationProvenanceError",
    "GAIN_CALIBRATION_SEAL_FILE_SHA256",
    "GAIN_CALIBRATION_SEAL_SHA256",
    "GAIN_TABLE_FILE_SHA256",
    "GAIN_TABLE_SHA256",
    "NONZERO_CHECKPOINT_COUNT",
    "PARENT_ARTIFACT_COUNT",
    "PARENT_BASENAME",
    "PARENT_CONFIG_SHA256",
    "PARENT_REGISTRY_COUNT",
    "PARENT_REGISTRY_FILE_SHA256",
    "PARENT_REGISTRY_SEMANTIC_SHA256",
    "PARENT_RUN_SCHEMA",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_FINGERPRINT",
    "PARENT_TERMINAL_DECISION",
    "PERMITTED_HISTORICAL_ROLES",
    "QUARTILE_SPECIALIST_PARENT_BASENAME",
    "QUARTILE_SPECIALIST_PARENT_CONFIG_SHA256",
    "QUARTILE_SPECIALIST_PARENT_REGISTRY_COUNT",
    "QUARTILE_SPECIALIST_PARENT_REGISTRY_FILE_SHA256",
    "QUARTILE_SPECIALIST_PARENT_REGISTRY_SEMANTIC_SHA256",
    "QUARTILE_SPECIALIST_PARENT_SOURCE_FINGERPRINT",
    "RANK_LABEL_OPEN_FILE_SHA256",
    "RANK_LABEL_OPEN_SHA256",
    "QuartileDirectionAdjudicationProvenanceError",
    "ROLE_ROW_COUNT",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TRAINING_RANK_PATH_TABLES_FILE_SHA256",
    "TRAINING_CHECKPOINT_INDEX_SHA256",
    "TRAINING_RANK_PATH_TABLES_SHA256",
    "compare_parent_snapshots",
    "direction_adjudication_source_fingerprint",
    "direction_adjudication_source_paths",
    "load_already_open_role",
    "quartile_direction_adjudication_source_fingerprint",
    "quartile_direction_adjudication_source_paths",
    "scientific_config_fingerprint",
    "snapshot_parent_run",
    "source_fingerprint",
    "source_paths",
    "validate_semantic_config",
    "verify_direction_adjudication_resume_compatibility",
    "verify_parent",
    "verify_parent_immutability_snapshot",
    "verify_quartile_specialist_parent",
    "verify_quartile_direction_adjudication_parent",
    "verify_resume_compatibility",
]
