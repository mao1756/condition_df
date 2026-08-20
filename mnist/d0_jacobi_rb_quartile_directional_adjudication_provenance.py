"""Immutable provenance for the q1--q3 directional representation audit.

The workflow is deliberately read-only.  This module composes the complete
quartile-specialist verifier with the verifier for its authoritative
time-local design parent, and exposes strict readers for the three historical
roles that were already opened by the specialist run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import json

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
from mnist import (
    d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_provenance
    as _prior,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_provenance import (
    TIME_LOCAL_PARENT_BASENAME,
    TIME_LOCAL_PARENT_CONFIG_SHA256,
    TIME_LOCAL_PARENT_DECISION,
    TIME_LOCAL_PARENT_REGISTRY_COUNT,
    TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256,
    TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256,
    TIME_LOCAL_PARENT_SOURCE_FINGERPRINT,
    build_path_id_plan,
    verify_quartile_specialist_parents,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    v3_transitive_source_paths,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local_provenance import (
    snapshot_parent_run as snapshot_time_local_parent_run,
    verify_parent_immutability_snapshot as verify_time_local_immutability_snapshot,
)


SCHEMA = "experiment12-d0-jacobi-rb-quartile-directional-adjudication-provenance"
SCHEMA_VERSION = 1

SPECIALIST_PARENT_BASENAME = _prior.PARENT_BASENAME
SPECIALIST_PARENT_DECISION = _prior.PARENT_TERMINAL_DECISION
SPECIALIST_PARENT_REGISTRY_COUNT = _prior.PARENT_ARTIFACT_COUNT
SPECIALIST_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    _prior.PARENT_REGISTRY_SEMANTIC_SHA256
)
SPECIALIST_PARENT_REGISTRY_FILE_SHA256 = _prior.PARENT_REGISTRY_FILE_SHA256
SPECIALIST_PARENT_SOURCE_FINGERPRINT = _prior.PARENT_SOURCE_FINGERPRINT
SPECIALIST_PARENT_CONFIG_SHA256 = _prior.PARENT_SCIENTIFIC_CONFIG_SHA256

# The physical-fit cache has twice as many rows as either adjudication role.
PHYSICAL_FIT_ROW_COUNT = 114_688
PHYSICAL_FIT_PATH_COUNT = 64
PHYSICAL_FIT_CACHE_BINDING_FILE_SHA256 = (
    "bb299e98009d4e5000162f8dd826416b41fe168cc84b6c87c6c612919c375c5d"
)
PHYSICAL_FIT_CACHE_BINDING_SEMANTIC_SHA256 = (
    "36083b096ad3e9d624c556e916ae846ca23891f04e6d1501ed434ef4e2523b6b"
)
PHYSICAL_FIT_ROLE_INDEX_FILE_SHA256 = (
    "81bc267c715894a211059e111002cfb8133e40667cbae2e62fcb62b6cf47f57d"
)
PHYSICAL_FIT_ROLE_INDEX_SEMANTIC_SHA256 = (
    "ab015c6905278e71b2b8831b8c1d481c7d23995abe63d501f9f00f809b94cab0"
)
PHYSICAL_FIT_ROLE_OPEN_FILE_SHA256 = _prior.FIT_LABEL_OPEN_FILE_SHA256
PHYSICAL_FIT_ROLE_OPEN_SEMANTIC_SHA256 = (
    "03bb6f65cb4f53eadf221bc226d2a7e3528c0e92345eb74f50051b3f0dbc8fa2"
)

PERMITTED_HISTORICAL_ROLES = (
    "physical_fit",
    "gain_calibration",
    "training_rank",
)
ROLE_ORDER = _prior.ROLE_ORDER

_IDENTITY_FIELDS = (
    "sample_key",
    "path_id",
    "outer_step",
    "phase",
    "midpoint_index",
    "midpoint_fraction",
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


class QuartileDirectionalProvenanceError(ArtifactCompatibilityError):
    """An immutable parent, role firewall, or child binding changed."""


@dataclass(frozen=True)
class AlreadyOpenInputs:
    """One historically opened role loaded without deserializing its labels."""

    role: str
    inputs: Mapping[str, np.ndarray]
    input_index: Mapping[str, Any]
    binding: Mapping[str, Any]
    role_open: Mapping[str, Any]
    row_identity_sha256: str


@dataclass(frozen=True)
class AlreadyOpenRole:
    """One historically opened role with immutable input and label arrays."""

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
        raise QuartileDirectionalProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuartileDirectionalProvenanceError(
            f"invalid {description}: {path}"
        ) from exc
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


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


def _same_path(value: Any, expected: Path, description: str) -> None:
    _require(
        isinstance(value, str) and Path(value).resolve() == expected.resolve(),
        f"{description} path binding changed",
    )


def snapshot_parent_runs(
    *, specialist_run_dir: str | Path, time_local_run_dir: str | Path
) -> dict[str, Any]:
    """Take complete content-addressed snapshots of both explicit parents."""

    specialist = _prior.snapshot_parent_run(specialist_run_dir)
    time_local = snapshot_time_local_parent_run(time_local_run_dir)
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-snapshots",
            "schema_version": SCHEMA_VERSION,
            "specialist": specialist,
            "time_local": time_local,
        }
    )


def snapshot_parent_run(run_dir: str | Path) -> dict[str, Any]:
    """Snapshot either explicit parent, dispatching by immutable basename."""

    root = Path(run_dir).resolve()
    if root.name == SPECIALIST_PARENT_BASENAME:
        return _prior.snapshot_parent_run(root)
    if root.name == TIME_LOCAL_PARENT_BASENAME:
        return snapshot_time_local_parent_run(root)
    raise QuartileDirectionalProvenanceError(
        f"unknown directional-adjudication parent basename: {root.name}"
    )


def compare_parent_snapshots(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two snapshots produced for the same explicit parent."""

    before_root = Path(str(before.get("run_dir", ""))).resolve()
    after_root = Path(str(after.get("run_dir", ""))).resolve()
    _require(before_root == after_root, "parent snapshot roots changed")
    if before_root.name == SPECIALIST_PARENT_BASENAME:
        try:
            return _prior.compare_parent_snapshots(before, after)
        except ArtifactCompatibilityError as exc:
            raise QuartileDirectionalProvenanceError(str(exc)) from exc
    if before_root.name == TIME_LOCAL_PARENT_BASENAME:
        _require(dict(before) == dict(after), "immutable parent tree snapshot changed")
        return _hashed(
            {
                "schema": f"{SCHEMA}-parent-tree-comparison",
                "schema_version": SCHEMA_VERSION,
                "evaluation_status": "evaluated",
                "passed": 1,
                "run_dir": str(before_root),
                "tree_sha256": before["tree_sha256"],
                "parent_files_modified": 0,
            }
        )
    raise QuartileDirectionalProvenanceError("unknown parent snapshot basename")


def verify_parent_immutability(
    *,
    specialist_run_dir: str | Path,
    time_local_run_dir: str | Path,
    snapshots: Mapping[str, Any],
) -> dict[str, Any]:
    """Require both explicit parents to remain byte-for-byte unchanged."""

    _assert_semantic(snapshots, "parent snapshots")
    specialist_snapshot = snapshots.get("specialist")
    time_local_snapshot = snapshots.get("time_local")
    _require(
        isinstance(specialist_snapshot, Mapping)
        and isinstance(time_local_snapshot, Mapping),
        "parent snapshot table changed",
    )
    try:
        specialist_after = _prior.verify_parent_immutability_snapshot(
            specialist_run_dir, specialist_snapshot
        )
        time_local_after = verify_time_local_immutability_snapshot(
            time_local_run_dir, time_local_snapshot
        )
    except ArtifactCompatibilityError as exc:
        raise QuartileDirectionalProvenanceError(str(exc)) from exc
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-immutability",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "specialist_tree_sha256": specialist_after["tree_sha256"],
            "time_local_tree_sha256": time_local_after["tree_sha256"],
            "parents_mutated": 0,
        }
    )


def _time_local_transitive_paths(root: Path) -> dict[str, Path]:
    config = _load_json(root / "scientific_config.json", "time-local config")
    return {
        "memory_v3_run_dir": Path(str(config.get("parent_memory_v3_run_dir", ""))).resolve(),
        "coarse_witness_run_dir": Path(
            str(config.get("parent_coarse_witness_run_dir", ""))
        ).resolve(),
        "bayes_power_run_dir": Path(
            str(config.get("parent_bayes_power_run_dir", ""))
        ).resolve(),
    }


def verify_parents(
    specialist_run_dir: str | Path,
    time_local_run_dir: str | Path,
    *,
    snapshots: Mapping[str, Any] | None = None,
    specialist_snapshot: Mapping[str, Any] | None = None,
    time_local_snapshot: Mapping[str, Any] | None = None,
    verify_checkpoint_states: bool = True,
    verify_cache_rows: bool = True,
    verify_external_cache: bool = True,
) -> dict[str, Any]:
    """Verify the two explicit parents and their complete transitive chain."""

    specialist_root = Path(specialist_run_dir).resolve()
    time_local_root = Path(time_local_run_dir).resolve()
    _require(
        specialist_root.name == SPECIALIST_PARENT_BASENAME,
        "wrong quartile-specialist parent basename",
    )
    _require(
        time_local_root.name == TIME_LOCAL_PARENT_BASENAME,
        "wrong time-local parent basename",
    )
    if snapshots is not None:
        _require(
            specialist_snapshot is None and time_local_snapshot is None,
            "duplicate parent snapshot bindings",
        )
    else:
        specialist_snapshot = specialist_snapshot or snapshot_parent_run(
            specialist_root
        )
        time_local_snapshot = time_local_snapshot or snapshot_parent_run(
            time_local_root
        )
        snapshots = _hashed(
            {
                "schema": f"{SCHEMA}-parent-snapshots",
                "schema_version": SCHEMA_VERSION,
                "specialist": specialist_snapshot,
                "time_local": time_local_snapshot,
            }
        )
    _assert_semantic(snapshots, "parent snapshots")
    specialist_snapshot = snapshots.get("specialist")
    _require(isinstance(specialist_snapshot, Mapping), "specialist snapshot missing")

    try:
        specialist = _prior.verify_parent(
            specialist_root,
            snapshot=specialist_snapshot,
            verify_checkpoint_states=verify_checkpoint_states,
            verify_cache_rows=verify_cache_rows,
        )
        if verify_cache_rows:
            _verify_physical_fit_rows(specialist_root)
        paths = _time_local_transitive_paths(time_local_root)
        time_local = verify_quartile_specialist_parents(
            time_local_run_dir=time_local_root,
            verify_external_cache=verify_external_cache,
            **paths,
        )
    except ArtifactCompatibilityError as exc:
        raise QuartileDirectionalProvenanceError(str(exc)) from exc

    parent_record = _load_json(
        specialist_root / "parent_provenance.json", "specialist parent provenance"
    )
    _assert_semantic(parent_record, "specialist parent provenance")
    authoritative = parent_record.get("authoritative_parent")
    _require(isinstance(authoritative, Mapping), "authoritative parent binding missing")
    _same_path(authoritative.get("run_dir"), time_local_root, "authoritative parent")
    _require(
        authoritative.get("basename") == TIME_LOCAL_PARENT_BASENAME
        and authoritative.get("decision") == TIME_LOCAL_PARENT_DECISION
        and authoritative.get("registry_count") == TIME_LOCAL_PARENT_REGISTRY_COUNT
        and authoritative.get("registry_semantic_sha256")
        == TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256
        and authoritative.get("registry_file_sha256")
        == TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256
        and authoritative.get("source_fingerprint")
        == TIME_LOCAL_PARENT_SOURCE_FINGERPRINT
        and authoritative.get("scientific_config_sha256")
        == TIME_LOCAL_PARENT_CONFIG_SHA256,
        "specialist-to-time-local binding changed",
    )
    immutability = verify_parent_immutability(
        specialist_run_dir=specialist_root,
        time_local_run_dir=time_local_root,
        snapshots=snapshots,
    )
    return _hashed(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "specialist_parent": specialist,
            "time_local_parent": time_local["authoritative_parent"],
            "transitive_parents": time_local["transitive_parents"],
            "transitive_provenance": time_local["transitive_provenance"],
            "parent_immutability": immutability,
            "all_explicit_parent_registries_verified": 1,
            "all_transitive_parent_registries_verified": 1,
            "all_registered_artifact_hashes_verified": 1,
            "all_checkpoint_hashes_verified": 1,
            "all_checkpoint_state_hashes_verified": int(verify_checkpoint_states),
            "all_role_cache_payloads_verified": int(verify_cache_rows),
            "checkpoint_payloads_valid": 1,
            "role_cache_payloads_valid": int(verify_cache_rows),
            "cache_bindings_valid": 1,
            "historical_design_evidence_authorizing": 0,
            "selection_paths_opened": 0,
            "confirmation_paths_opened": 0,
            "parents_mutated": 0,
        }
    )


def _validate_input_identities(
    inputs: Mapping[str, np.ndarray], *, role: str, expected_paths: tuple[int, ...]
) -> str:
    for name in _IDENTITY_FIELDS:
        _require(name in inputs, f"role input identity field missing: {name}")
    expected_rows = PHYSICAL_FIT_ROW_COUNT if role == "physical_fit" else _prior.ROLE_ROW_COUNT
    sample_keys = np.asarray(inputs["sample_key"], dtype=np.int64)
    path_ids = np.asarray(inputs["path_id"], dtype=np.int64)
    outer_steps = np.asarray(inputs["outer_step"], dtype=np.int64)
    phases = np.asarray(inputs["phase"], dtype=np.int64)
    midpoints = np.asarray(inputs["midpoint_index"], dtype=np.int64)
    _require(len(sample_keys) == expected_rows, "role row count changed")
    _require(len(np.unique(sample_keys)) == expected_rows, "role sample-key identity changed")
    _require(
        tuple(int(value) for value in np.unique(path_ids)) == expected_paths,
        "role path identities changed",
    )
    _require(
        tuple(int(value) for value in np.unique(outer_steps))
        == _prior.SELECTED_OUTER_STEPS,
        "role outer-step identities changed",
    )
    _require(
        tuple(int(value) for value in np.unique(phases)) == tuple(range(7))
        and tuple(int(value) for value in np.unique(midpoints)) == tuple(range(8)),
        "role phase/midpoint identities changed",
    )
    cells = np.stack((path_ids, outer_steps, phases, midpoints), axis=1)
    _require(
        len(np.unique(cells, axis=0)) == expected_rows,
        "role path/step/cell identity changed",
    )
    return _prior._identity_sha256(inputs)


def _physical_fit_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binding_path = root / "physical_fit_cache_binding.json"
    index_path = root / "role_caches/physical_fit/eager_cache/train_index.json"
    _require(
        file_fingerprint(binding_path) == PHYSICAL_FIT_CACHE_BINDING_FILE_SHA256,
        "physical_fit cache-binding file hash changed",
    )
    _require(
        file_fingerprint(index_path) == PHYSICAL_FIT_ROLE_INDEX_FILE_SHA256,
        "physical_fit role-index file hash changed",
    )
    binding = _load_json(binding_path, "physical_fit cache binding")
    index = _load_json(index_path, "physical_fit role index")
    _assert_semantic(binding, "physical_fit cache binding")
    _assert_semantic(index, "physical_fit role index")
    expected_paths = tuple(build_path_id_plan()["roles"]["physical_fit"])
    _require(
        binding.get("semantic_sha256")
        == PHYSICAL_FIT_CACHE_BINDING_SEMANTIC_SHA256
        and binding.get("role") == "physical_fit"
        and tuple(binding.get("path_ids", ())) == expected_paths
        and int(binding.get("input_row_count", -1)) == PHYSICAL_FIT_ROW_COUNT
        and int(binding.get("physical_labels_opened", -1)) == 0
        and binding.get("role_index_semantic_sha256")
        == PHYSICAL_FIT_ROLE_INDEX_SEMANTIC_SHA256,
        "physical_fit cache binding changed",
    )
    _require(
        index.get("semantic_sha256") == PHYSICAL_FIT_ROLE_INDEX_SEMANTIC_SHA256
        and index.get("role") == "train"
        and tuple(index.get("path_ids", ())) == expected_paths
        and int(index.get("path_count", -1)) == PHYSICAL_FIT_PATH_COUNT
        and int(index.get("input_row_count", -1)) == PHYSICAL_FIT_ROW_COUNT
        and int(index.get("label_row_count", -1)) == PHYSICAL_FIT_ROW_COUNT
        and tuple(index.get("selected_outer_steps", ()))
        == _prior.SELECTED_OUTER_STEPS
        and int(index.get("branch_input_label_separated", -1)) == 1
        and int(index.get("cross_role_artifact_commit", -1)) == 0,
        "physical_fit role index changed",
    )
    return binding, index


def _verify_physical_fit_rows(root: Path) -> str:
    """Verify the physical-fit input/label payload identities shard by shard."""

    _, index = _physical_fit_binding(root)
    role_root = root / "role_caches/physical_fit"
    inputs = _prior._read_identity_blocks(role_root, index, "branch_inputs")
    labels = _prior._read_identity_blocks(role_root, index, "branch_labels")
    expected_paths = tuple(build_path_id_plan()["roles"]["physical_fit"])
    row_sha = _validate_input_identities(
        inputs, role="physical_fit", expected_paths=expected_paths
    )
    for name in _IDENTITY_FIELDS:
        _require(
            name in labels and np.array_equal(inputs[name], labels[name]),
            f"physical_fit input/label row identity changed: {name}",
        )
    return row_sha


def _role_contract(
    root: Path, role: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[int, ...]]:
    _require(role in PERMITTED_HISTORICAL_ROLES, f"forbidden historical role: {role}")
    try:
        _prior._ensure_selection_confirmation_absent(root)
        role_records = _prior._verify_role_open_history(root)
    except ArtifactCompatibilityError as exc:
        raise QuartileDirectionalProvenanceError(str(exc)) from exc
    required_roles = ROLE_ORDER[: ROLE_ORDER.index(role) + 1]
    _require(
        all(name in role_records for name in required_roles),
        f"historical role order changed before {role}",
    )
    if role == "physical_fit":
        binding, index = _physical_fit_binding(root)
    else:
        try:
            binding, index, _ = _prior._verify_cache_binding(
                root, role, verify_rows=False
            )
        except ArtifactCompatibilityError as exc:
            raise QuartileDirectionalProvenanceError(str(exc)) from exc
    expected_paths = tuple(build_path_id_plan()["roles"][role])
    return binding, index, role_records[role], expected_paths


def _freeze_arrays(arrays: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        _require(array.flags.c_contiguous, f"role array is not C-contiguous: {name}")
        array.setflags(write=False)
        frozen[str(name)] = array
    return MappingProxyType(frozen)


def load_already_open_inputs(
    parent_run_dir: str | Path, role: str
) -> AlreadyOpenInputs:
    """Load only permitted inputs; no physical-label file is deserialized."""

    root = Path(parent_run_dir).resolve()
    _require(root.name == SPECIALIST_PARENT_BASENAME, "wrong specialist parent basename")
    before = _prior._tree_metadata(root)
    binding, expected_index, role_open, expected_paths = _role_contract(root, role)
    role_root = root / "role_caches" / role
    try:
        inputs, input_index = _load_eager_role_inputs(role_root, "train")
    except Exception as exc:
        raise QuartileDirectionalProvenanceError(
            f"could not load already-open role inputs: {role}"
        ) from exc
    _require(input_index == expected_index, f"{role} loaded input index changed")
    row_sha = _validate_input_identities(
        inputs, role=role, expected_paths=expected_paths
    )
    result = AlreadyOpenInputs(
        role=role,
        inputs=_freeze_arrays(inputs),
        input_index=MappingProxyType(dict(input_index)),
        binding=MappingProxyType(dict(binding)),
        role_open=MappingProxyType(dict(role_open)),
        row_identity_sha256=row_sha,
    )
    _require(before == _prior._tree_metadata(root), "input loader observed a parent mutation")
    return result


def load_already_open_role(
    parent_run_dir: str | Path, role: str
) -> AlreadyOpenRole:
    """Load one already-open historical role as immutable host arrays."""

    root = Path(parent_run_dir).resolve()
    _require(root.name == SPECIALIST_PARENT_BASENAME, "wrong specialist parent basename")
    before = _prior._tree_metadata(root)
    binding, expected_index, role_open, expected_paths = _role_contract(root, role)
    role_root = root / "role_caches" / role
    try:
        inputs, input_index = _load_eager_role_inputs(role_root, "train")
        labels, label_index = _load_eager_role_labels(role_root, "train")
    except Exception as exc:
        raise QuartileDirectionalProvenanceError(
            f"could not load already-open role: {role}"
        ) from exc
    _require(
        input_index == expected_index and label_index == expected_index,
        f"{role} loaded role index changed",
    )
    row_sha = _prior._validate_joined_identities(
        inputs,
        labels,
        expected_paths=expected_paths,
    ) if role != "physical_fit" else _validate_input_identities(
        inputs, role=role, expected_paths=expected_paths
    )
    if role == "physical_fit":
        for name in _IDENTITY_FIELDS:
            _require(
                name in labels and np.array_equal(inputs[name], labels[name]),
                f"role input/label row identity changed: {name}",
            )
    result = AlreadyOpenRole(
        role=role,
        inputs=_freeze_arrays(inputs),
        labels=_freeze_arrays(labels),
        input_index=MappingProxyType(dict(input_index)),
        label_index=MappingProxyType(dict(label_index)),
        binding=MappingProxyType(dict(binding)),
        role_open=MappingProxyType(dict(role_open)),
        row_identity_sha256=row_sha,
    )
    _require(before == _prior._tree_metadata(root), "strict role loader observed a parent mutation")
    return result


def source_paths(
    entry_points: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Return the transitive local source closure for child manifest binding."""

    if entry_points is None:
        package = Path(__file__).resolve().parent
        entry_points = tuple(
            package / name
            for name in (
                "d0_jacobi_rb_quartile_directional_adjudication.py",
                "d0_jacobi_rb_quartile_directional_adjudication_inference.py",
                "d0_jacobi_rb_quartile_directional_adjudication_provenance.py",
                "d0_jacobi_rb_quartile_directional_adjudication_gate.py",
                "diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication.py",
            )
        )
    paths = tuple(Path(path).resolve() for path in entry_points)
    _require(paths and all(path.is_file() for path in paths), "source entry point missing")
    return v3_transitive_source_paths(paths)


def source_fingerprint(entry_points: Iterable[str | Path] | None = None) -> str:
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
    _assert_semantic(record, "quartile-directional scientific config")
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
    """Delegate the fail-closed child resume contract to the prior workflow."""

    try:
        return _prior.verify_resume_compatibility(
            run_dir,
            expected_bindings=expected_bindings,
            artifact_bindings=artifact_bindings,
        )
    except ArtifactCompatibilityError as exc:
        raise QuartileDirectionalProvenanceError(str(exc)) from exc


# Descriptive aliases used by the workflow and tests.
verify_quartile_directional_parents = verify_parents
verify_parent_runs = verify_parents
quartile_directional_source_paths = source_paths
quartile_directional_source_fingerprint = source_fingerprint
verify_quartile_directional_resume_compatibility = verify_resume_compatibility


__all__ = [
    "AlreadyOpenInputs",
    "AlreadyOpenRole",
    "PERMITTED_HISTORICAL_ROLES",
    "PHYSICAL_FIT_CACHE_BINDING_FILE_SHA256",
    "PHYSICAL_FIT_CACHE_BINDING_SEMANTIC_SHA256",
    "PHYSICAL_FIT_PATH_COUNT",
    "PHYSICAL_FIT_ROLE_INDEX_FILE_SHA256",
    "PHYSICAL_FIT_ROLE_INDEX_SEMANTIC_SHA256",
    "PHYSICAL_FIT_ROW_COUNT",
    "QuartileDirectionalProvenanceError",
    "ROLE_ORDER",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SPECIALIST_PARENT_BASENAME",
    "SPECIALIST_PARENT_CONFIG_SHA256",
    "SPECIALIST_PARENT_DECISION",
    "SPECIALIST_PARENT_REGISTRY_COUNT",
    "SPECIALIST_PARENT_REGISTRY_FILE_SHA256",
    "SPECIALIST_PARENT_REGISTRY_SEMANTIC_SHA256",
    "SPECIALIST_PARENT_SOURCE_FINGERPRINT",
    "TIME_LOCAL_PARENT_BASENAME",
    "TIME_LOCAL_PARENT_CONFIG_SHA256",
    "TIME_LOCAL_PARENT_DECISION",
    "TIME_LOCAL_PARENT_REGISTRY_COUNT",
    "TIME_LOCAL_PARENT_REGISTRY_FILE_SHA256",
    "TIME_LOCAL_PARENT_REGISTRY_SEMANTIC_SHA256",
    "TIME_LOCAL_PARENT_SOURCE_FINGERPRINT",
    "load_already_open_inputs",
    "load_already_open_role",
    "compare_parent_snapshots",
    "quartile_directional_source_fingerprint",
    "quartile_directional_source_paths",
    "scientific_config_fingerprint",
    "snapshot_parent_runs",
    "snapshot_parent_run",
    "source_fingerprint",
    "source_paths",
    "validate_semantic_config",
    "verify_parent_immutability",
    "verify_parent_runs",
    "verify_parents",
    "verify_quartile_directional_parents",
    "verify_quartile_directional_resume_compatibility",
    "verify_resume_compatibility",
]
