"""Content-addressed relocation checks for the directional adjudication.

This module does not make a historical run portable by weakening its path
checks.  Instead, it verifies one exact ``ready_for_fittrace`` run and the two
immutable parent trees at new filesystem roots.  The returned identity omits
all operational roots, so a later continuation workflow can bind the same
bytes on Windows or Linux without modifying any historical artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterator, Mapping
import json

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_gate import (
    safety_record,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_provenance import (
    AlreadyOpenRole,
    PERMITTED_HISTORICAL_ROLES,
    _freeze_arrays,
    _prior as _legacy_provenance,
    load_already_open_role as _load_already_open_role,
    snapshot_parent_run,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import (
    load_eager_role_inputs,
    load_eager_role_labels,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_provenance import (
    build_path_id_plan,
)


SCHEMA = "experiment12-d0-jacobi-rb-quartile-directional-portable"
SCHEMA_VERSION = 1

PREDECESSOR_BASENAME = (
    "20260808-203454_production-read-only-quartile-directional-"
    "adjudication-bootstrap-fix"
)
PREDECESSOR_ARTIFACT_COUNT = 26
PREDECESSOR_REGISTRY_SEMANTIC_SHA256 = (
    "15971dcd90ce6fe17d90fbcd005bfdc5cf855939e6b7b6f884714be202bb1b37"
)
PREDECESSOR_REGISTRY_FILE_SHA256 = (
    "a4776544a85aef4160783a4c77d5b0262c8a56fe105d1e464a3706777a1b0e8f"
)
PREDECESSOR_SOURCE_FINGERPRINT = (
    "6d40bfb7424d43f15e4bd54c8c6984a3db4b774cb34670335ce1e6cfd1070d0c"
)
PREDECESSOR_CONFIG_SHA256 = (
    "48fb83c1a3869386d2e2106e9e21a0062bed21d39cdf483893a3aa09c669b4a3"
)
PREDECESSOR_DECISION = "ready_for_fittrace"
LEGACY_SOURCE_COUNT = 37

SPECIALIST_PARENT_BASENAME = (
    "20260807-132351_production-exact-quartile-specialist"
)
TIME_LOCAL_PARENT_BASENAME = (
    "20260807-005609_production-v3-time-local-adjudication"
)

REPORT_NAMES = {
    "predecessor": "portable_predecessor_binding.json",
    "legacy_sources": "portable_legacy_source_closure.json",
    "parents": "portable_parent_tree_binding.json",
    "identity": "portable_relocation_identity.json",
}

_STAGE_SEALS = (
    "preflight_artifact_seal.json",
    "replay_artifact_seal.json",
    "controls_artifact_seal.json",
)
_PASSED_GATES = (
    "preflight_gate.json",
    "historical_replay_gate.json",
    "controls_gate.json",
)
_FORBIDDEN_LATER_PATHS = (
    "fit_label_open.json",
    "gain_label_open.json",
    "rank_label_open.json",
    "directional_shards",
    "fit_direction_moments.npz",
    "fit_trajectory_stability.csv",
    "fittrace_metrics.json",
    "fittrace_gate.json",
    "gain_direction_moments.npz",
    "nomination_seal.json",
    "nominate_metrics.json",
    "nominate_gate.json",
    "rank_direction_moments.npz",
    "rank_path_moments.npz",
    "adjudicate_metrics.json",
    "adjudicate_gate.json",
    "REPORT.md",
)


class PortableContinuationError(ArtifactCompatibilityError):
    """The copied predecessor, source closure, or parent tree changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PortableContinuationError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortableContinuationError(f"invalid {description}: {path}") from exc
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


def _semantic_body(record: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(record)
    body.pop("semantic_sha256", None)
    return body


def _assert_semantic(record: Mapping[str, Any], description: str) -> None:
    _require(
        record.get("semantic_sha256") == config_fingerprint(_semantic_body(record)),
        f"{description} semantic hash changed",
    )


def _hashed(record: Mapping[str, Any]) -> dict[str, Any]:
    body = _semantic_body(record)
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _safe_relative(value: Any) -> str:
    _require(isinstance(value, str) and value != "", "artifact path is malformed")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and path.as_posix() == value,
        f"artifact path is unsafe: {value}",
    )
    return value


def _regular_files(root: Path) -> dict[str, Path]:
    rows: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        _require(not path.is_symlink(), f"portable evidence contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        _require(relative not in rows, f"duplicate portable evidence path: {relative}")
        rows[relative] = path
    return rows


def _verify_stage_seal(root: Path, name: str) -> dict[str, Any]:
    seal = _load_json(root / name, name)
    _assert_semantic(seal, name)
    artifacts = seal.get("artifacts")
    _require(isinstance(artifacts, list), f"{name} artifact table changed")
    seen: set[str] = set()
    for item in artifacts:
        _require(isinstance(item, Mapping), f"{name} has a malformed artifact row")
        relative = _safe_relative(item.get("path"))
        _require(relative not in seen, f"{name} repeats an artifact path")
        seen.add(relative)
        path = root / PurePosixPath(relative)
        _require(
            path.is_file()
            and int(path.stat().st_size) == int(item.get("size", -1))
            and file_fingerprint(path) == item.get("sha256"),
            f"sealed predecessor artifact changed: {relative}",
        )
    return seal


def verify_ready_predecessor(run_dir: str | Path) -> dict[str, Any]:
    """Verify the exact, unopened Windows predecessor after a bytewise copy."""

    root = Path(run_dir).resolve()
    _require(root.name == PREDECESSOR_BASENAME, "wrong portable predecessor basename")
    registry_path = root / "artifact_registry.json"
    _require(
        file_fingerprint(registry_path) == PREDECESSOR_REGISTRY_FILE_SHA256,
        "portable predecessor registry file changed",
    )
    registry = _load_json(registry_path, "portable predecessor registry")
    _assert_semantic(registry, "portable predecessor registry")
    artifacts = registry.get("artifacts")
    _require(
        int(registry.get("artifact_count", -1)) == PREDECESSOR_ARTIFACT_COUNT
        and registry.get("semantic_sha256")
        == PREDECESSOR_REGISTRY_SEMANTIC_SHA256
        and isinstance(artifacts, list)
        and len(artifacts) == PREDECESSOR_ARTIFACT_COUNT,
        "portable predecessor registry binding changed",
    )

    files = _regular_files(root)
    expected_inventory = {"artifact_registry.json"}
    seen: set[str] = set()
    for item in artifacts:
        _require(isinstance(item, Mapping), "predecessor registry row is malformed")
        relative = _safe_relative(item.get("path"))
        _require(relative not in seen, "predecessor registry path is duplicated")
        seen.add(relative)
        expected_inventory.add(relative)
        path = files.get(relative)
        _require(
            path is not None
            and int(path.stat().st_size) == int(item.get("size", -1))
            and file_fingerprint(path) == item.get("sha256"),
            f"registered predecessor artifact changed: {relative}",
        )
    _require(
        set(files) == expected_inventory,
        "portable predecessor file inventory changed",
    )

    manifest = _load_json(root / "run_manifest.json", "predecessor manifest")
    config = _load_json(root / "scientific_config.json", "predecessor config")
    _assert_semantic(manifest, "predecessor manifest")
    _assert_semantic(config, "predecessor config")
    _require(
        manifest.get("source_fingerprint") == PREDECESSOR_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256") == PREDECESSOR_CONFIG_SHA256
        and config.get("semantic_sha256") == PREDECESSOR_CONFIG_SHA256
        and config.get("source_fingerprint") == PREDECESSOR_SOURCE_FINGERPRINT,
        "predecessor source or scientific configuration changed",
    )
    status = _load_json(root / "run_status.json", "predecessor status")
    decision = _load_json(
        root / "quartile_directional_adjudication_decision.json",
        "predecessor decision",
    )
    workflow = _load_json(root / "workflow_gate.json", "predecessor workflow")
    _require(
        status.get("decision") == PREDECESSOR_DECISION
        and status.get("stage") == "controls"
        and status.get("state") == "running"
        and decision.get("decision") == PREDECESSOR_DECISION
        and workflow.get("decision", {}).get("decision") == PREDECESSOR_DECISION
        and workflow.get("require_gate") == "controls"
        and int(workflow.get("required_gate_pass", 0)) == 1,
        "predecessor is not the exact ready-for-fittrace state",
    )
    for name in _PASSED_GATES:
        gate = _load_json(root / name, name)
        _require(
            gate.get("evaluation_status") == "evaluated"
            and int(gate.get("passed", 0)) == 1
            and int(gate.get("stage_execution_valid", 0)) == 1,
            f"predecessor gate did not pass: {name}",
        )
    seal_hashes = {
        name: file_fingerprint(root / name)
        for name in _STAGE_SEALS
        if _verify_stage_seal(root, name)
    }
    _require(
        all(not (root / relative).exists() for relative in _FORBIDDEN_LATER_PATHS),
        "predecessor opened fit, gain, rank, or later-stage evidence",
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-predecessor-binding",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "basename": PREDECESSOR_BASENAME,
            "artifact_count": PREDECESSOR_ARTIFACT_COUNT,
            "registry_semantic_sha256": PREDECESSOR_REGISTRY_SEMANTIC_SHA256,
            "registry_file_sha256": PREDECESSOR_REGISTRY_FILE_SHA256,
            "source_fingerprint": PREDECESSOR_SOURCE_FINGERPRINT,
            "scientific_config_sha256": PREDECESSOR_CONFIG_SHA256,
            "decision": PREDECESSOR_DECISION,
            "stage_seal_file_sha256": seal_hashes,
            "later_stage_evidence_opened": 0,
            **safety_record(),
        }
    )


def _repo_relative_source(value: Any) -> str:
    _require(isinstance(value, str), "legacy source path is malformed")
    normalized = value.replace("\\", "/")
    marker = "/mnist/"
    position = normalized.rfind(marker)
    _require(position >= 0, f"legacy source path is outside mnist: {value}")
    return _safe_relative(normalized[position + 1 :])


def verify_legacy_source_closure(
    predecessor_run_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the saved Windows closure against repo-relative relocated files."""

    predecessor = Path(predecessor_run_dir).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    closure = _load_json(predecessor / "source_closure.json", "legacy source closure")
    _assert_semantic(closure, "legacy source closure")
    sources = closure.get("sources")
    _require(
        int(closure.get("source_count", -1)) == LEGACY_SOURCE_COUNT
        and closure.get("source_fingerprint") == PREDECESSOR_SOURCE_FINGERPRINT
        and isinstance(sources, list)
        and len(sources) == LEGACY_SOURCE_COUNT,
        "legacy source closure binding changed",
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sources:
        _require(isinstance(item, Mapping), "legacy source row is malformed")
        relative = _repo_relative_source(item.get("path"))
        _require(relative not in seen, "legacy source path is duplicated")
        seen.add(relative)
        path = root / PurePosixPath(relative)
        _require(
            path.is_file()
            and not path.is_symlink()
            and int(path.stat().st_size) == int(item.get("size", -1))
            and file_fingerprint(path) == item.get("sha256"),
            f"relocated legacy source changed: {relative}",
        )
        rows.append(
            {
                "path": relative,
                "size": int(item["size"]),
                "sha256": str(item["sha256"]),
            }
        )
    content_fingerprint = config_fingerprint(rows)
    return _hashed(
        {
            "schema": f"{SCHEMA}-legacy-source-closure",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "source_count": LEGACY_SOURCE_COUNT,
            "legacy_source_fingerprint": PREDECESSOR_SOURCE_FINGERPRINT,
            "content_fingerprint": content_fingerprint,
            "sources": rows,
            **safety_record(),
        }
    )


def _normalized_snapshot(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.pop("run_dir", None)
    result.pop("semantic_sha256", None)
    return result


def verify_relocated_parent_snapshots(
    predecessor_run_dir: str | Path,
    *,
    specialist_run_dir: str | Path,
    time_local_run_dir: str | Path,
) -> dict[str, Any]:
    """Match relocated parents while ignoring only their operational roots."""

    predecessor = Path(predecessor_run_dir).resolve()
    specialist_root = Path(specialist_run_dir).resolve()
    time_local_root = Path(time_local_run_dir).resolve()
    _require(
        specialist_root.name == SPECIALIST_PARENT_BASENAME,
        "wrong relocated specialist basename",
    )
    _require(
        time_local_root.name == TIME_LOCAL_PARENT_BASENAME,
        "wrong relocated time-local basename",
    )
    saved = _load_json(
        predecessor / "parent_immutability_before.json",
        "predecessor parent snapshot",
    )
    _assert_semantic(saved, "predecessor parent snapshot")
    expected_specialist = saved.get("quartile_specialist")
    expected_time_local = saved.get("time_local")
    _require(
        isinstance(expected_specialist, Mapping)
        and isinstance(expected_time_local, Mapping),
        "predecessor parent snapshots are missing",
    )
    try:
        observed_specialist = snapshot_parent_run(specialist_root)
        observed_time_local = snapshot_parent_run(time_local_root)
    except ArtifactCompatibilityError as exc:
        raise PortableContinuationError(str(exc)) from exc
    expected_specialist_body = _normalized_snapshot(expected_specialist)
    expected_time_local_body = _normalized_snapshot(expected_time_local)
    observed_specialist_body = _normalized_snapshot(observed_specialist)
    observed_time_local_body = _normalized_snapshot(observed_time_local)
    _require(
        observed_specialist_body == expected_specialist_body,
        "relocated specialist tree changed",
    )
    _require(
        observed_time_local_body == expected_time_local_body,
        "relocated time-local tree changed",
    )
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-tree-binding",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "quartile_specialist": {
                "basename": SPECIALIST_PARENT_BASENAME,
                "content_sha256": config_fingerprint(expected_specialist_body),
                "tree_sha256": expected_specialist_body["tree_sha256"],
                "file_count": expected_specialist_body["file_count"],
                "total_bytes": expected_specialist_body["total_bytes"],
            },
            "time_local": {
                "basename": TIME_LOCAL_PARENT_BASENAME,
                "content_sha256": config_fingerprint(expected_time_local_body),
                "tree_sha256": expected_time_local_body["tree_sha256"],
                "file_count": expected_time_local_body["file_count"],
                "total_bytes": expected_time_local_body["total_bytes"],
            },
            "ignored_fields": ["run_dir", "semantic_sha256"],
            **safety_record(),
        }
    )


def verify_portable_continuation(
    predecessor_run_dir: str | Path,
    *,
    specialist_run_dir: str | Path,
    time_local_run_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return one root-independent identity for all continuation inputs."""

    predecessor = verify_ready_predecessor(predecessor_run_dir)
    sources = verify_legacy_source_closure(predecessor_run_dir, repo_root=repo_root)
    parents = verify_relocated_parent_snapshots(
        predecessor_run_dir,
        specialist_run_dir=specialist_run_dir,
        time_local_run_dir=time_local_run_dir,
    )
    identity_body = {
        "predecessor_semantic_sha256": predecessor["semantic_sha256"],
        "legacy_sources_semantic_sha256": sources["semantic_sha256"],
        "parent_trees_semantic_sha256": parents["semantic_sha256"],
    }
    relocation_identity = config_fingerprint(identity_body)
    return _hashed(
        {
            "schema": f"{SCHEMA}-relocation-identity",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            **identity_body,
            "relocation_identity_sha256": relocation_identity,
            "root_paths_authorizing": 0,
            "ignored_historical_field_count": 2,
            **safety_record(),
        }
    )


def _portable_cache_binding(
    root: Path, role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one immutable cache while relocating only its lookup root."""

    _require(
        role in {"gain_calibration", "training_rank"},
        f"role does not use the portable cache binding: {role}",
    )
    spec = _legacy_provenance._CACHE_SPECS[role]
    binding_path = root / f"{role}_cache_binding.json"
    _require(
        file_fingerprint(binding_path) == spec["binding_file_sha256"],
        f"{role} cache-binding file hash changed",
    )
    binding = _load_json(binding_path, f"{role} cache binding")
    _assert_semantic(binding, f"{role} cache binding")
    role_root = root / "role_caches" / role
    index_path = role_root / "eager_cache" / "train_index.json"
    _require(
        file_fingerprint(index_path) == spec["index_file_sha256"],
        f"{role} role-index file hash changed",
    )
    index = _load_json(index_path, f"{role} role index")
    _assert_semantic(index, f"{role} role index")
    expected_paths = tuple(build_path_id_plan()["roles"][role])
    historical_cache_root = str(binding.get("cache_root", ""))
    _require(
        binding.get("schema")
        == f"{_legacy_provenance.PARENT_RUN_SCHEMA}-role-cache-binding"
        and binding.get("role") == role
        and historical_cache_root != ""
        and binding.get("semantic_sha256") == spec["binding_semantic_sha256"]
        and tuple(binding.get("path_ids", ())) == expected_paths
        and int(binding.get("input_row_count", -1))
        == _legacy_provenance.ROLE_ROW_COUNT
        and binding.get("role_index_semantic_sha256")
        == spec["index_semantic_sha256"]
        and int(binding.get("physical_labels_opened", -1)) == 0,
        f"{role} cache binding changed",
    )
    _require(
        index.get("semantic_sha256") == spec["index_semantic_sha256"]
        and index.get("role") == "train"
        and tuple(index.get("path_ids", ())) == expected_paths
        and int(index.get("path_count", -1))
        == _legacy_provenance.ROLE_PATH_COUNT
        and int(index.get("input_row_count", -1))
        == _legacy_provenance.ROLE_ROW_COUNT
        and int(index.get("label_row_count", -1))
        == _legacy_provenance.ROLE_ROW_COUNT
        and tuple(index.get("selected_outer_steps", ()))
        == _legacy_provenance.SELECTED_OUTER_STEPS
        and int(index.get("branch_input_label_separated", -1)) == 1
        and int(index.get("cross_role_artifact_commit", -1)) == 0,
        f"{role} role index changed",
    )
    metrics = _load_json(
        role_root / "eager_cache" / "train_validation_metrics.json",
        f"{role} cache metrics",
    )
    _assert_semantic(metrics, f"{role} cache metrics")
    _require(
        metrics.get("semantic_sha256") == binding.get("metrics_semantic_sha256"),
        f"{role} cache metrics binding changed",
    )
    # The exact binding file hash above makes this historical string immutable.
    # It is retained as proof metadata; only payload lookup uses ``role_root``.
    return binding, index


def portable_load_already_open_role(
    parent_run_dir: str | Path, role: str
) -> AlreadyOpenRole:
    """Load a historical role after content-addressed cross-platform relocation."""

    root = Path(parent_run_dir).resolve()
    _require(root.name == SPECIALIST_PARENT_BASENAME, "wrong specialist parent basename")
    _require(role in PERMITTED_HISTORICAL_ROLES, f"forbidden historical role: {role}")
    if role == "physical_fit":
        # This role already has a path-independent binding validator.
        return _load_already_open_role(root, role)

    before = _legacy_provenance._tree_metadata(root)
    try:
        _legacy_provenance._ensure_selection_confirmation_absent(root)
        role_records = _legacy_provenance._verify_role_open_history(root)
    except ArtifactCompatibilityError as exc:
        raise PortableContinuationError(str(exc)) from exc
    required_roles = _legacy_provenance.ROLE_ORDER[
        : _legacy_provenance.ROLE_ORDER.index(role) + 1
    ]
    _require(
        all(name in role_records for name in required_roles),
        f"historical role order changed before {role}",
    )
    binding, expected_index = _portable_cache_binding(root, role)
    role_root = root / "role_caches" / role
    try:
        inputs, input_index = load_eager_role_inputs(role_root, "train")
        labels, label_index = load_eager_role_labels(role_root, "train")
    except Exception as exc:
        raise PortableContinuationError(
            f"could not load relocated historical role: {role}"
        ) from exc
    _require(
        input_index == expected_index and label_index == expected_index,
        f"{role} loaded role index changed",
    )
    expected_paths = tuple(build_path_id_plan()["roles"][role])
    try:
        row_identity_sha256 = _legacy_provenance._validate_joined_identities(
            inputs,
            labels,
            expected_paths=expected_paths,
        )
    except ArtifactCompatibilityError as exc:
        raise PortableContinuationError(str(exc)) from exc
    result = AlreadyOpenRole(
        role=role,
        inputs=_freeze_arrays(inputs),
        labels=_freeze_arrays(labels),
        input_index=MappingProxyType(dict(input_index)),
        label_index=MappingProxyType(dict(label_index)),
        binding=MappingProxyType(dict(binding)),
        role_open=MappingProxyType(dict(role_records[role])),
        row_identity_sha256=row_identity_sha256,
    )
    _require(
        before == _legacy_provenance._tree_metadata(root),
        "portable role loader observed a parent mutation",
    )
    return result


@contextmanager
def portable_role_loading(
    specialist_run_dir: str | Path,
) -> Iterator[None]:
    """Route delegated workflow role loads through the relocated parent root."""

    from mnist import (
        diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication
        as workflow,
    )

    expected_root = Path(specialist_run_dir).resolve()
    _require(
        expected_root.name == SPECIALIST_PARENT_BASENAME,
        "wrong portable specialist basename",
    )
    original = workflow.load_already_open_role

    def relocated_loader(parent_run_dir: str | Path, role: str) -> AlreadyOpenRole:
        supplied = Path(parent_run_dir).resolve()
        _require(supplied == expected_root, "delegated role loader changed parent root")
        return portable_load_already_open_role(expected_root, role)

    workflow.load_already_open_role = relocated_loader
    try:
        yield
    finally:
        workflow.load_already_open_role = original


__all__ = [
    "LEGACY_SOURCE_COUNT",
    "PREDECESSOR_ARTIFACT_COUNT",
    "PREDECESSOR_BASENAME",
    "PREDECESSOR_CONFIG_SHA256",
    "PREDECESSOR_DECISION",
    "PREDECESSOR_REGISTRY_FILE_SHA256",
    "PREDECESSOR_REGISTRY_SEMANTIC_SHA256",
    "PREDECESSOR_SOURCE_FINGERPRINT",
    "PortableContinuationError",
    "REPORT_NAMES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SPECIALIST_PARENT_BASENAME",
    "TIME_LOCAL_PARENT_BASENAME",
    "portable_load_already_open_role",
    "portable_role_loading",
    "verify_legacy_source_closure",
    "verify_portable_continuation",
    "verify_ready_predecessor",
    "verify_relocated_parent_snapshots",
]
