"""Immutable provenance for the boundary-tangent v3 time-local adjudication.

The adjudication is deliberately read-only.  It may inspect the completed
validation search and historical coarse/Bayes controls, but it must not open
the reserved confirmation namespace or modify any parent.  This module binds
the three terminal runs byte-for-byte and returns a compact commitment for the
new workflow.

Unlike several historical provenance helpers, this verifier does not compare
the parents' recorded source fingerprints with the *current* checkout.  The
run registries and manifests are the immutable historical commitments; later
additive patches are allowed to change the live source closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory_provenance import (
    verify_immutable_cache_binding,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-v3-time-local-provenance"
SCHEMA_VERSION = 1

MEMORY_PARENT_BASENAME = (
    "20260806-181326_production-zero-baseline-v3-memory-safe"
)
MEMORY_PARENT_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-v3-memory-confirmation"
)
MEMORY_PARENT_REGISTRY_COUNT = 620
MEMORY_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "a69ca33be9c3281eb54e9285f3d292d9e8c9cd0775781ed652af0a1adda85626"
)
MEMORY_PARENT_REGISTRY_FILE_SHA256 = (
    "a3a697a7a92f2ad6f0cc666d95ebb92b11dfd8c6837005d457356bd326b79076"
)
MEMORY_PARENT_SOURCE_FINGERPRINT = (
    "5cde21b7ed36a806f2e872cf8fb3f7ac859d9317ab87aad1e728b95d544a2cee"
)
MEMORY_PARENT_CONFIG_SHA256 = (
    "bbb3b79cc7afc6311b6e6413cde0e7f93c074f8fe885bd4960140bc39e44f61e"
)
MEMORY_PARENT_DECISION = "no_validation_candidate"

WITNESS_PARENT_BASENAME = (
    "20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix"
)
WITNESS_PARENT_SCHEMA = "experiment12-d0-jacobi-rb-physical-coarse-signal-witness"
WITNESS_REGISTRY_COUNT = 2_616
WITNESS_REGISTRY_SEMANTIC_SHA256 = (
    "ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3"
)
WITNESS_REGISTRY_FILE_SHA256 = (
    "866984822ef14dbb13f0644a0f23697f6fb42ecec40b07341249d74043319747"
)
WITNESS_SOURCE_FINGERPRINT = (
    "31f1f15008c2db864e282c5d3fa047986a9b576b92c480d50a18d55138e9eafb"
)
WITNESS_CONFIG_SHA256 = (
    "b2e28989ef6da6fa2d233b14ee475c04e10326079cf03750f1f427494de90f14"
)
WITNESS_DECISION = "exact_physical_coarse_signal_detected"

BAYES_PARENT_BASENAME = "20260730-012459_production-noisy-jacobi-bayes-power"
BAYES_PARENT_SCHEMA = "experiment12-d0-jacobi-rb-bayes-power-calibration"
BAYES_REGISTRY_COUNT = 74
BAYES_REGISTRY_SEMANTIC_SHA256 = (
    "01b5d772299611e9e17b886658b7eba80a7ab50805241e94d2e9a8ba36562e79"
)
BAYES_REGISTRY_FILE_SHA256 = (
    "4caa9597f1ce7e6e6180ea11bffe55138f10582791b60e5d529e38d9e3b13bec"
)
BAYES_SOURCE_FINGERPRINT = (
    "bbd522fb4ce2219e6759d5e0c78b8fc1baa8c4f39c8fe356f902f676ec1e7462"
)
BAYES_CONFIG_SHA256 = (
    "05cdd8b9b2b03920ef51d099f4b29589b66297fe6664ff6b052aa5f59d08d1ac"
)
BAYES_DECISION = "noisy_bayes_detection_pipeline_calibrated"

PHYSICAL_ANCESTOR_BASENAME = (
    "20260729-015817_production-exact-k512-rb-one-image-learnability"
)
PHYSICAL_ANCESTOR_REGISTRY_COUNT = 544
PHYSICAL_ANCESTOR_REGISTRY_SEMANTIC_SHA256 = (
    "5e0b46328b6783614bdb7d394587b32e63d2d33b76f0279abdab6ecdf7d4e18a"
)
PHYSICAL_ANCESTOR_REGISTRY_FILE_SHA256 = (
    "26370722f9f7ce5a6675bc3b626710373b407f4a4134a3425c852af8b17259a5"
)
PHYSICAL_ANCESTOR_SOURCE_FINGERPRINT = (
    "f651d7322384275f269de3442f8e7a03cf062994b6bd894db735541d9f2a699d"
)
PHYSICAL_ANCESTOR_CONFIG_SHA256 = (
    "58ccdfc5df2c4b30c28da5a143aa2570e390b007e7825b89d164762d7d23b01c"
)

CONFIRMATION_PATH_START = 0xF2000
CONFIRMATION_PATH_STOP = 0xF2040
EXPECTED_CRITICAL_VALUE = 7.1588810358178305
EXPECTED_CANDIDATE_TABLE_SHAPE = (32, 120, 228)
EXPECTED_SELECTION_SEAL_SEMANTIC_SHA256 = (
    "88da8779be2885356f1a210b116f27e467088880f05309121333d8438b588e71"
)


class TimeLocalProvenanceError(ArtifactCompatibilityError):
    """A terminal parent or a sealed-evidence commitment changed."""


@dataclass(frozen=True)
class _LegacyParentSpec:
    role: str
    basename: str
    run_schema: str
    registry_schema: str
    registry_count: int
    registry_semantic_sha256: str
    registry_file_sha256: str
    source_fingerprint: str
    config_schema: str
    config_sha256: str
    status_schema: str
    status_file_sha256: str
    terminal_state: str
    terminal_stage: str
    decision_path: str
    decision_schema: str
    decision: str
    gates: tuple[tuple[str, str, int], ...]


_WITNESS_SPEC = _LegacyParentSpec(
    role="physical_coarse_signal_witness",
    basename=WITNESS_PARENT_BASENAME,
    run_schema=WITNESS_PARENT_SCHEMA,
    registry_schema=f"{WITNESS_PARENT_SCHEMA}-artifact-registry",
    registry_count=WITNESS_REGISTRY_COUNT,
    registry_semantic_sha256=WITNESS_REGISTRY_SEMANTIC_SHA256,
    registry_file_sha256=WITNESS_REGISTRY_FILE_SHA256,
    source_fingerprint=WITNESS_SOURCE_FINGERPRINT,
    config_schema=f"{WITNESS_PARENT_SCHEMA}-scientific-config",
    config_sha256=WITNESS_CONFIG_SHA256,
    status_schema=f"{WITNESS_PARENT_SCHEMA}-status",
    status_file_sha256=(
        "ae982dd57034ee54226dc6f84fea9dea48d351773f5e911bc584b2f58600624c"
    ),
    terminal_state="completed",
    terminal_stage="analyze",
    decision_path="physical_coarse_signal_decision.json",
    decision_schema="d0-jacobi-rb-physical-coarse-signal-decision-v1",
    decision=WITNESS_DECISION,
    gates=(
        (
            "coarse_signal_preflight_gate.json",
            "d0-jacobi-rb-physical-coarse-signal-gate-v1",
            1,
        ),
        (
            "coarse_signal_panel_a_gate.json",
            "d0-jacobi-rb-physical-coarse-signal-gate-v1",
            1,
        ),
        (
            "coarse_signal_panel_b_gate.json",
            "d0-jacobi-rb-physical-coarse-signal-gate-v1",
            1,
        ),
        (
            "coarse_signal_witness_gate.json",
            "d0-jacobi-rb-physical-coarse-signal-gate-v1",
            1,
        ),
    ),
)

_BAYES_SPEC = _LegacyParentSpec(
    role="bayes_power_calibration",
    basename=BAYES_PARENT_BASENAME,
    run_schema=BAYES_PARENT_SCHEMA,
    registry_schema=f"{BAYES_PARENT_SCHEMA}-artifact-registry",
    registry_count=BAYES_REGISTRY_COUNT,
    registry_semantic_sha256=BAYES_REGISTRY_SEMANTIC_SHA256,
    registry_file_sha256=BAYES_REGISTRY_FILE_SHA256,
    source_fingerprint=BAYES_SOURCE_FINGERPRINT,
    config_schema=f"{BAYES_PARENT_SCHEMA}-scientific-config",
    config_sha256=BAYES_CONFIG_SHA256,
    status_schema=f"{BAYES_PARENT_SCHEMA}-status",
    status_file_sha256=(
        "cc5eaf5479968196aec1607f9fb41ab35a5b95b05483ed386cf5058019513459"
    ),
    terminal_state="completed",
    terminal_stage="confirm",
    decision_path="bayes_power_decision.json",
    decision_schema=f"{BAYES_PARENT_SCHEMA}-gate-decision",
    decision=BAYES_DECISION,
    gates=(
        ("preflight_gate.json", f"{BAYES_PARENT_SCHEMA}-gate", 1),
        ("cache_gate.json", f"{BAYES_PARENT_SCHEMA}-gate", 1),
        ("train_gate.json", f"{BAYES_PARENT_SCHEMA}-gate", 1),
        ("controls_gate.json", f"{BAYES_PARENT_SCHEMA}-gate", 1),
    ),
)

_MEMORY_EXCLUDED_HASHES = {
    "artifact_registry.json": MEMORY_PARENT_REGISTRY_FILE_SHA256,
    "boundary_tangent_v3_memory_decision.json": (
        "c4e2c8ef7016ff60bc734e999554509b77c9c4d9ffdb489e8616df9f7549f02d"
    ),
    "run_status.json": (
        "5da7dcb1338e6cf64a8b3fcff621cb959dbd103bae8675318f67dcad6f2ac6fe"
    ),
    "workflow_gate.json": (
        "eff4c0121a6a7efe5cf26e9a19c86d998acf34f75108f320866b886eb5797978"
    ),
}

_MEMORY_CRITICAL_HASHES = {
    "run_manifest.json": (
        "00233c477a6f8242d879b5097c9d25d9db3799a767939168dd48ca08c564f91f"
    ),
    "scientific_config.json": (
        "748773366120b995eb68c217577fd90e225d8f2c068b7834c50bfbb780a2b003"
    ),
    "preflight_gate.json": (
        "f2710cbca74911ca68c6cece0e924995cac2d1434c14923210c3a018594a5b73"
    ),
    "train_gate.json": (
        "abd1699345c4c3dbb0c1d43288e77ddb0f09b5a1a785aeb3b0fbbdb8d5ab436b"
    ),
    "select_gate.json": (
        "ed26c62355257c3d0a170402bf67961023712178b306af2a3fde752cd538e359"
    ),
    "selection_artifact_seal.json": (
        "fafd85604ba0d36cc0fece315e195cf4adf8bed917efecb12332019913cd8f35"
    ),
    "validation_candidate_path_tables.npz": (
        "fa0e2a9a3ca0d574c3798d3599ec228e5a8c29b190af3f91b8d307dcb64043d1"
    ),
    "validation_search_max_t.npz": (
        "54046e099d3d9ff7b021a39d9f116b04985a0957e2d0c10865dfc615f2c255f1"
    ),
    "validation_search_max_t.json": (
        "02922d8fca4a3396969805a157f2b905a77e21e8637c1ff06b730151f189ec5d"
    ),
    "validation_selection.json": (
        "8c5dfb2e5083b45efd6711210c51e9871c0aab54b2777aa2051aa757df7420fb"
    ),
    "validation_candidate_index.json": (
        "42753377014cd702371520ad84436a9b0f1410d5e6b2689506d07494ef7ac3a1"
    ),
    "validation_search_plan.json": (
        "cdae7cc5ca23d274210b5e5de097da4afe5a363d47d482c2b81cc8e497151b1e"
    ),
    "no_validation_candidate.json": (
        "8fb914022c9350d5b8f781711e8916b8a9391c820080f960ccebf5f305e3dd37"
    ),
    "path_id_plan.json": (
        "05c6194ea2aad156ecc46f64d1c858ce24b364548dc1be305335635d9fd3f6c4"
    ),
    "immutable_cache_binding.json": (
        "150e59d5f240a34c79b89466bcff39fbec9d65a771ba8d8ee9e841579159f061"
    ),
}

_SELECTION_SEAL_REQUIRED = frozenset(
    {
        "validation_search_plan.json",
        "validation_label_open.json",
        "update_zero_validation_control.json",
        "validation_candidate_path_tables.npz",
        "validation_candidate_index.json",
        "validation_search_max_t.npz",
        "validation_search_max_t.json",
        "validation_candidate_summary.csv",
        "validation_selection.json",
        "select_metrics.json",
        "select_gate.json",
        "no_validation_candidate.json",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TimeLocalProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TimeLocalProvenanceError(f"invalid {description}: {path}") from exc
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


def _snapshot_rows(root: Path) -> list[dict[str, Any]]:
    _require(root.is_dir(), f"parent run does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for item in sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    ):
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size": item.stat().st_size,
                "sha256": file_fingerprint(item),
            }
        )
    return rows


def snapshot_parent_run(run_dir: str | Path) -> dict[str, Any]:
    """Return a content-addressed snapshot of a terminal parent directory."""

    root = Path(run_dir).resolve()
    rows = _snapshot_rows(root)
    return _hashed(
        {
            "schema": f"{SCHEMA}-parent-tree-snapshot",
            "schema_version": SCHEMA_VERSION,
            "run_dir": str(root),
            "file_count": len(rows),
            "total_bytes": sum(int(row["size"]) for row in rows),
            "tree_sha256": config_fingerprint(rows),
        }
    )


def verify_parent_immutability_snapshot(
    run_dir: str | Path, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute and compare a snapshot without changing the parent."""

    _assert_semantic(snapshot, "parent tree snapshot")
    current = snapshot_parent_run(run_dir)
    _require(
        dict(current) == dict(snapshot),
        "immutable parent tree snapshot changed",
    )
    return current


def _snapshot_map(snapshot_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["path"]): dict(row) for row in snapshot_rows}


def _verify_memory_registry(
    root: Path, snapshot_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    observed = _snapshot_map(snapshot_rows)
    _require(root.name == MEMORY_PARENT_BASENAME, "wrong memory-v3 parent basename")
    registry = _load_json(root / "artifact_registry.json", "memory parent registry")
    raw = registry.get("artifacts")
    _require(
        registry.get("schema") == f"{MEMORY_PARENT_SCHEMA}-artifact-registry"
        and registry.get("schema_version") == 1
        and registry.get("artifact_count") == MEMORY_PARENT_REGISTRY_COUNT
        and isinstance(raw, list)
        and len(raw) == MEMORY_PARENT_REGISTRY_COUNT
        and registry.get("semantic_sha256")
        == MEMORY_PARENT_REGISTRY_SEMANTIC_SHA256
        and config_fingerprint({"artifacts": raw})
        == MEMORY_PARENT_REGISTRY_SEMANTIC_SHA256,
        "memory parent registry binding changed",
    )
    records: dict[str, dict[str, Any]] = {}
    for item in raw:
        _require(isinstance(item, Mapping), "memory registry row is malformed")
        relative = _safe_relative(item.get("path"))
        _require(relative not in records, "memory registry path is duplicated")
        _require(
            observed.get(relative)
            == {
                "path": relative,
                "size": item.get("size"),
                "sha256": item.get("sha256"),
            },
            f"memory parent artifact changed: {relative}",
        )
        records[relative] = dict(item)
    _require(
        set(observed) == set(records) | set(_MEMORY_EXCLUDED_HASHES),
        "memory parent terminal file set changed",
    )
    for relative, expected in _MEMORY_EXCLUDED_HASHES.items():
        _require(
            observed.get(relative, {}).get("sha256") == expected,
            f"memory parent terminal artifact changed: {relative}",
        )
    for relative, expected in _MEMORY_CRITICAL_HASHES.items():
        _require(
            observed.get(relative, {}).get("sha256") == expected,
            f"memory parent critical artifact changed: {relative}",
        )
    return registry, records


def _verify_legacy_registry(
    root: Path,
    spec: _LegacyParentSpec,
    snapshot_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    observed = _snapshot_map(snapshot_rows)
    _require(root.name == spec.basename, f"wrong {spec.role} parent basename")
    _require(
        observed.get("artifact_registry.json", {}).get("sha256")
        == spec.registry_file_sha256,
        f"{spec.role} registry file hash changed",
    )
    _require(
        observed.get("run_status.json", {}).get("sha256")
        == spec.status_file_sha256,
        f"{spec.role} terminal status file hash changed",
    )
    registry = _load_json(root / "artifact_registry.json", f"{spec.role} registry")
    raw = registry.get("records")
    _require(
        registry.get("schema") == spec.registry_schema
        and registry.get("schema_version") == 1
        and registry.get("record_count") == spec.registry_count
        and isinstance(raw, list)
        and len(raw) == spec.registry_count
        and (
            registry.get("registry_sha256", registry.get("semantic_sha256"))
            == spec.registry_semantic_sha256
        )
        and (
            "registry_sha256" not in registry
            or config_fingerprint(raw) == spec.registry_semantic_sha256
        ),
        f"{spec.role} registry binding changed",
    )
    records: dict[str, dict[str, Any]] = {}
    for item in raw:
        _require(isinstance(item, Mapping), f"{spec.role} registry row is malformed")
        relative = _safe_relative(item.get("path"))
        _require(relative not in records, f"{spec.role} registry path is duplicated")
        _require(
            observed.get(relative)
            == {
                "path": relative,
                "size": item.get("size"),
                "sha256": item.get("sha256"),
            },
            f"{spec.role} registered artifact changed: {relative}",
        )
        records[relative] = dict(item)
    _require(
        set(observed) == set(records) | {"artifact_registry.json", "run_status.json"},
        f"{spec.role} terminal file set changed",
    )
    return registry, records


def _verify_zero_scope(record: Mapping[str, Any], description: str) -> None:
    for name in (
        "sampling_performed",
        "reverse_sampling_performed",
        "reconstruction_performed",
        "full_reverse_path_performed",
        "controller_control_trajectory_performed",
        "image_sampling_performed",
    ):
        _require(int(record.get(name, 0)) == 0, f"{description} records {name}")


def _verify_legacy_parent(
    root: Path,
    spec: _LegacyParentSpec,
    snapshot_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _, records = _verify_legacy_registry(root, spec, snapshot_rows)
    manifest = _load_json(root / "run_manifest.json", f"{spec.role} manifest")
    config = _load_json(root / "scientific_config.json", f"{spec.role} config")
    status = _load_json(root / "run_status.json", f"{spec.role} status")
    decision = _load_json(root / spec.decision_path, f"{spec.role} decision")
    _require(
        manifest.get("schema") == spec.run_schema
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == spec.source_fingerprint
        and manifest.get("scientific_config_sha256") == spec.config_sha256,
        f"{spec.role} manifest binding changed",
    )
    _require(
        config.get("schema") == spec.config_schema
        and config.get("schema_version") == 1
        and config.get("semantic_sha256") == spec.config_sha256,
        f"{spec.role} scientific configuration changed",
    )
    _require(
        status.get("schema") == spec.status_schema
        and status.get("schema_version") == 1
        and status.get("state") == spec.terminal_state
        and status.get("stage") == spec.terminal_stage
        and status.get("decision") == spec.decision
        and status.get("artifact_registry_record_count") == spec.registry_count
        and status.get("artifact_registry_sha256")
        == spec.registry_semantic_sha256
        and status.get("artifact_registry_file_sha256")
        == spec.registry_file_sha256,
        f"{spec.role} terminal status changed",
    )
    _require(
        decision.get("schema") == spec.decision_schema
        and decision.get("schema_version") == 1
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == spec.decision,
        f"{spec.role} terminal decision changed",
    )
    _verify_zero_scope(status, f"{spec.role} status")
    _verify_zero_scope(decision, f"{spec.role} decision")
    gate_rows: dict[str, Any] = {}
    for relative, schema, passed in spec.gates:
        gate = _load_json(root / relative, f"{spec.role} {relative}")
        _require(
            gate.get("schema") == schema
            and gate.get("schema_version") == 1
            and gate.get("evaluation_status") == "evaluated"
            and int(gate.get("passed", -1)) == passed,
            f"{spec.role} gate changed: {relative}",
        )
        _verify_zero_scope(gate, f"{spec.role} {relative}")
        gate_rows[relative] = {"schema": schema, "passed": passed}
    return {
        "role": spec.role,
        "run_dir": str(root),
        "basename": root.name,
        "registry": {
            "record_count": spec.registry_count,
            "semantic_sha256": spec.registry_semantic_sha256,
            "file_sha256": spec.registry_file_sha256,
            "all_artifact_hashes_verified": 1,
        },
        "source_fingerprint": spec.source_fingerprint,
        "scientific_config_sha256": spec.config_sha256,
        "decision": spec.decision,
        "gates": gate_rows,
        "registered_paths": sorted(records),
        "verified": 1,
    }


def _verify_selection_seal(
    root: Path, records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    seal = _load_json(root / "selection_artifact_seal.json", "selection seal")
    _assert_semantic(seal, "selection seal")
    rows = seal.get("artifacts")
    _require(
        seal.get("schema") == f"{MEMORY_PARENT_SCHEMA}-stage-seal"
        and seal.get("schema_version") == 1
        and seal.get("semantic_sha256")
        == EXPECTED_SELECTION_SEAL_SEMANTIC_SHA256
        and isinstance(rows, list)
        and len(rows) == 17,
        "selection artifact seal binding changed",
    )
    sealed: set[str] = set()
    for raw in rows:
        _require(isinstance(raw, Mapping), "selection seal row is malformed")
        relative = _safe_relative(raw.get("path"))
        _require(relative not in sealed, "selection seal path is duplicated")
        expected = records.get(relative)
        _require(
            expected is not None
            and raw.get("size") == expected.get("size")
            and raw.get("sha256") == expected.get("sha256"),
            f"selection-sealed artifact changed: {relative}",
        )
        sealed.add(relative)
    _require(
        _SELECTION_SEAL_REQUIRED <= sealed,
        "selection seal is missing authorizing evidence",
    )
    return {
        "artifact_count": len(sealed),
        "semantic_sha256": EXPECTED_SELECTION_SEAL_SEMANTIC_SHA256,
        "required_artifacts_present": 1,
        "verified": 1,
    }


def _verify_checkpoint_grid(
    root: Path, records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    grid = _load_json(root / "candidate_grid.json", "candidate checkpoint grid")
    _assert_semantic(grid, "candidate checkpoint grid")
    checkpoints = grid.get("checkpoints")
    expected_pairs = {
        (seed, update)
        for seed in (261312, 261313, 261314)
        for update in range(0, 4_001, 100)
    }
    _require(
        grid.get("checkpoint_count") == 123
        and grid.get("nonzero_candidate_count") == 120
        and isinstance(checkpoints, list)
        and len(checkpoints) == 123,
        "candidate checkpoint grid shape changed",
    )
    observed_pairs: set[tuple[int, int]] = set()
    for raw in checkpoints:
        _require(isinstance(raw, Mapping), "candidate checkpoint row is malformed")
        seed = raw.get("seed")
        update = raw.get("update")
        _require(
            isinstance(seed, int)
            and not isinstance(seed, bool)
            and isinstance(update, int)
            and not isinstance(update, bool),
            "candidate checkpoint identity is malformed",
        )
        pair = (seed, update)
        _require(pair not in observed_pairs, "candidate checkpoint is duplicated")
        observed_pairs.add(pair)
        relative = _safe_relative(raw.get("checkpoint_path"))
        record = records.get(relative)
        _require(
            record is not None
            and raw.get("checkpoint_file_sha256") == record.get("sha256")
            and (root / relative).is_file(),
            f"candidate checkpoint hash binding changed: {relative}",
        )
    _require(observed_pairs == expected_pairs, "candidate checkpoint identities changed")
    return {
        "checkpoint_count": 123,
        "nonzero_candidate_count": 120,
        "model_seeds": [261312, 261313, 261314],
        "all_checkpoint_hashes_verified": 1,
    }


def _verify_unopened_confirmation(
    root: Path, records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    path_plan = _load_json(root / "path_id_plan.json", "memory parent path plan")
    _assert_semantic(path_plan, "memory parent path plan")
    slots = path_plan.get("role_slots")
    roles = path_plan.get("roles")
    _require(
        isinstance(slots, Mapping)
        and isinstance(roles, Mapping)
        and isinstance(slots.get("confirmation"), Mapping),
        "confirmation path slot is missing",
    )
    slot = dict(slots["confirmation"])
    expected_ids = list(range(CONFIRMATION_PATH_START, CONFIRMATION_PATH_STOP))
    _require(
        slot.get("start") == CONFIRMATION_PATH_START
        and slot.get("stop_exclusive") == CONFIRMATION_PATH_STOP
        and slot.get("path_count") == 64
        and int(slot.get("opened", 1)) == 0
        and list(roles.get("confirmation", ())) == expected_ids
        and int(path_plan.get("confirmation_reserved_unopened", 0)) == 1,
        "confirmation namespace reservation changed",
    )
    selection = _load_json(root / "validation_selection.json", "sealed selection")
    no_candidate = _load_json(
        root / "no_validation_candidate.json", "no-candidate terminal record"
    )
    _assert_semantic(selection, "sealed selection")
    _assert_semantic(no_candidate, "no-candidate terminal record")
    _require(
        selection.get("decision") == MEMORY_PARENT_DECISION
        and selection.get("eligible_candidate_count") == 0
        and int(selection.get("logical_update_zero_selected", 0)) == 1
        and int(selection.get("confirmation_namespace_opened", 1)) == 0
        and int(selection.get("confirmation_paths_created", 1)) == 0
        and int(selection.get("confirmation_authorized", 1)) == 0
        and no_candidate.get("decision") == MEMORY_PARENT_DECISION
        and int(no_candidate.get("confirmation_forbidden", 0)) == 1
        and int(no_candidate.get("confirmation_namespace_opened", 1)) == 0,
        "sealed no-candidate confirmation firewall changed",
    )
    forbidden_paths = sorted(
        relative
        for relative in records
        if relative.startswith("confirmation/")
        or relative.startswith("confirmation_")
        or relative.startswith("confirmation-")
    )
    _require(not forbidden_paths, "confirmation evidence exists in immutable parent")
    return {
        "path_start": CONFIRMATION_PATH_START,
        "path_stop_exclusive": CONFIRMATION_PATH_STOP,
        "path_count": 64,
        "confirmation_namespace_opened": 0,
        "confirmation_paths_created": 0,
        "confirmation_evidence_paths": [],
        "verified": 1,
    }


def _verify_memory_parent(
    root: Path,
    snapshot_rows: list[dict[str, Any]],
    *,
    verify_external_cache: bool,
) -> dict[str, Any]:
    registry, records = _verify_memory_registry(root, snapshot_rows)
    manifest = _load_json(root / "run_manifest.json", "memory parent manifest")
    config = _load_json(root / "scientific_config.json", "memory parent config")
    status = _load_json(root / "run_status.json", "memory parent status")
    decision = _load_json(
        root / "boundary_tangent_v3_memory_decision.json",
        "memory parent decision",
    )
    _assert_semantic(config, "memory parent config")
    _require(
        manifest.get("schema") == f"{MEMORY_PARENT_SCHEMA}-manifest"
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == MEMORY_PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256") == MEMORY_PARENT_CONFIG_SHA256
        and config.get("schema") == f"{MEMORY_PARENT_SCHEMA}-scientific-config"
        and config.get("semantic_sha256") == MEMORY_PARENT_CONFIG_SHA256,
        "memory parent source/config binding changed",
    )
    _require(
        status.get("schema") == f"{MEMORY_PARENT_SCHEMA}-status"
        and status.get("state") == "gate_failed"
        and status.get("stage") == "terminal"
        and status.get("decision") == MEMORY_PARENT_DECISION
        and int(status.get("scientific_evidence_complete", 0)) == 1
        and int(status.get("physical_training_performed", 0)) == 1
        and int(status.get("validation_selection_performed", 0)) == 1
        and int(status.get("confirmation_performed", 1)) == 0,
        "memory parent terminal status changed",
    )
    _require(
        decision.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-v3-memory-gate-decision"
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == MEMORY_PARENT_DECISION
        and int(decision.get("physical_training_performed", 0)) == 1
        and int(decision.get("validation_selection_performed", 0)) == 1
        and int(decision.get("confirmation_authorized", 1)) == 0
        and int(decision.get("confirmation_performed", 1)) == 0,
        "memory parent terminal decision changed",
    )
    _verify_zero_scope(status, "memory parent status")
    _verify_zero_scope(decision, "memory parent decision")
    for relative, expected_pass in (
        ("preflight_gate.json", 1),
        ("train_gate.json", 1),
        ("select_gate.json", 0),
    ):
        gate = _load_json(root / relative, f"memory parent {relative}")
        _require(
            gate.get("evaluation_status") == "evaluated"
            and int(gate.get("passed", -1)) == expected_pass
            and int(gate.get("stage_execution_valid", 0)) == 1
            and int(gate.get("numerically_valid", 0)) == 1
            and int(gate.get("resource_valid", 0)) == 1,
            f"memory parent gate changed: {relative}",
        )
    index = _load_json(
        root / "validation_candidate_index.json", "validation candidate index"
    )
    search = _load_json(root / "validation_search_max_t.json", "validation max-T")
    _assert_semantic(index, "validation candidate index")
    _assert_semantic(search, "validation max-T")
    _require(
        tuple(index.get("shape", ())) == EXPECTED_CANDIDATE_TABLE_SHAPE
        and index.get("candidate_count") == 120
        and index.get("component_count") == 228
        and index.get("search_family_size") == 27_360
        and search.get("candidate_count") == 120
        and search.get("component_count") == 228
        and search.get("search_family_size") == 27_360
        and search.get("replicates") == 50_000
        and search.get("confidence") == 0.995
        and math.isclose(
            float(search.get("critical_value", float("nan"))),
            EXPECTED_CRITICAL_VALUE,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "sealed validation-search dimensions changed",
    )
    selection_seal = _verify_selection_seal(root, records)
    checkpoint_grid = _verify_checkpoint_grid(root, records)
    confirmation = _verify_unopened_confirmation(root, records)
    cache_binding = _load_json(
        root / "immutable_cache_binding.json", "immutable cache binding"
    )
    _assert_semantic(cache_binding, "immutable cache binding")
    _require(
        cache_binding.get("semantic_sha256")
        == manifest.get("immutable_cache_binding_sha256")
        and int(cache_binding.get("cache_is_read_only", 0)) == 1
        and int(cache_binding.get("cache_copied", 1)) == 0
        and int(cache_binding.get("cache_linked", 1)) == 0
        and int(cache_binding.get("confirmation_namespace_opened", 1)) == 0,
        "immutable external-cache binding changed",
    )
    if verify_external_cache:
        try:
            verify_immutable_cache_binding(cache_binding)
        except ArtifactCompatibilityError as exc:
            raise TimeLocalProvenanceError(
                "immutable external cache failed verification"
            ) from exc
    return {
        "role": "memory_safe_v3_selection",
        "run_dir": str(root),
        "basename": root.name,
        "registry": {
            "artifact_count": MEMORY_PARENT_REGISTRY_COUNT,
            "semantic_sha256": MEMORY_PARENT_REGISTRY_SEMANTIC_SHA256,
            "file_sha256": MEMORY_PARENT_REGISTRY_FILE_SHA256,
            "all_artifact_hashes_verified": 1,
        },
        "source_fingerprint": MEMORY_PARENT_SOURCE_FINGERPRINT,
        "scientific_config_sha256": MEMORY_PARENT_CONFIG_SHA256,
        "decision": MEMORY_PARENT_DECISION,
        "selection_seal": selection_seal,
        "checkpoint_grid": checkpoint_grid,
        "confirmation_firewall": confirmation,
        "immutable_cache_binding_sha256": cache_binding["semantic_sha256"],
        "external_cache_verified": int(bool(verify_external_cache)),
        "registered_path_count": len(records),
        "verified": 1,
    }


def _verify_witness_specific(root: Path) -> dict[str, Any]:
    analysis = _load_json(root / "physical_coarse_signal_analysis.json", "witness analysis")
    classification = analysis.get("classification")
    _require(isinstance(classification, Mapping), "witness classification is missing")
    bootstrap = analysis.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), "witness bootstrap is missing")
    point = float(
        classification.get("point_estimate", bootstrap.get("point_estimate", float("nan")))
    )
    bootstrap_lower = float(
        classification.get("bootstrap_lower_bound", float("nan"))
    )
    welch_lower = float(classification.get("welch_lower_bound", float("nan")))
    _require(
        all(math.isfinite(value) for value in (point, bootstrap_lower, welch_lower))
        and point > 0.0
        and bootstrap_lower > 0.0
        and welch_lower > 0.0
        and int(analysis.get("lower_bound_on_full_allowed_input_conditional_mean_energy", 0))
        == 1
        and int(analysis.get("conditional_mean_identically_zero_proven", 1)) == 0,
        "coarse witness no longer establishes positive signal",
    )
    return {
        "point_estimate": point,
        "bootstrap_lower_bound": bootstrap_lower,
        "welch_lower_bound": welch_lower,
        "positive_overall_signal_verified": 1,
    }


def _verify_transitive_provenance(
    witness_root: Path,
    bayes_root: Path,
    bayes_binding: Mapping[str, Any],
) -> dict[str, Any]:
    witness_provenance = _load_json(
        witness_root / "parent_provenance.json", "witness transitive provenance"
    )
    parents = witness_provenance.get("parents")
    _require(
        witness_provenance.get("evaluation_status") == "evaluated"
        and int(witness_provenance.get("passed", 0)) == 1
        and isinstance(parents, Mapping),
        "witness transitive provenance changed",
    )
    bayes = parents.get("bayes_power_calibration")
    physical = parents.get("physical_one_image")
    _require(
        isinstance(bayes, Mapping) and isinstance(physical, Mapping),
        "witness transitive parent map is incomplete",
    )
    bayes_registry = bayes.get("registry")
    physical_registry = physical.get("registry")
    _require(
        isinstance(bayes_registry, Mapping)
        and bayes.get("basename") == bayes_root.name == BAYES_PARENT_BASENAME
        and bayes.get("source_fingerprint") == BAYES_SOURCE_FINGERPRINT
        and bayes.get("scientific_config_sha256") == BAYES_CONFIG_SHA256
        and bayes_registry.get("record_count") == BAYES_REGISTRY_COUNT
        and bayes_registry.get("sha256") == BAYES_REGISTRY_SEMANTIC_SHA256
        and bayes_registry.get("file_sha256") == BAYES_REGISTRY_FILE_SHA256
        and int(bayes.get("verified", 0)) == 1
        and bayes_binding.get("verified") == 1,
        "witness-to-Bayes transitive binding changed",
    )
    _require(
        isinstance(physical_registry, Mapping)
        and physical.get("basename") == PHYSICAL_ANCESTOR_BASENAME
        and physical.get("source_fingerprint")
        == PHYSICAL_ANCESTOR_SOURCE_FINGERPRINT
        and physical.get("scientific_config_sha256")
        == PHYSICAL_ANCESTOR_CONFIG_SHA256
        and physical_registry.get("record_count")
        == PHYSICAL_ANCESTOR_REGISTRY_COUNT
        and physical_registry.get("sha256")
        == PHYSICAL_ANCESTOR_REGISTRY_SEMANTIC_SHA256
        and physical_registry.get("file_sha256")
        == PHYSICAL_ANCESTOR_REGISTRY_FILE_SHA256
        and int(physical.get("verified", 0)) == 1,
        "witness physical-ancestor binding changed",
    )
    bayes_provenance = _load_json(
        bayes_root / "parent_provenance.json", "Bayes transitive provenance"
    )
    _require(
        bayes_provenance.get("evaluation_status") == "evaluated"
        and int(bayes_provenance.get("passed", 0)) == 1
        and bayes_provenance.get("parent_run_basename")
        == PHYSICAL_ANCESTOR_BASENAME
        and bayes_provenance.get("parent_registry_record_count")
        == PHYSICAL_ANCESTOR_REGISTRY_COUNT
        and bayes_provenance.get("parent_registry_semantic_sha256")
        == PHYSICAL_ANCESTOR_REGISTRY_SEMANTIC_SHA256
        and bayes_provenance.get("parent_registry_file_sha256")
        == PHYSICAL_ANCESTOR_REGISTRY_FILE_SHA256
        and bayes_provenance.get("parent_source_fingerprint")
        == PHYSICAL_ANCESTOR_SOURCE_FINGERPRINT
        and bayes_provenance.get("parent_scientific_config_sha256")
        == PHYSICAL_ANCESTOR_CONFIG_SHA256
        and int(bayes_provenance.get("parent_mutated", 1)) == 0,
        "Bayes-to-physical transitive binding changed",
    )
    return {
        "witness_to_bayes_verified": 1,
        "witness_to_physical_verified": 1,
        "bayes_to_physical_verified": 1,
        "shared_physical_ancestor_verified": 1,
    }


def verify_time_local_adjudication_parents(
    *,
    memory_v3_run_dir: str | Path,
    coarse_witness_run_dir: str | Path,
    bayes_power_run_dir: str | Path,
    verify_external_cache: bool = True,
) -> dict[str, Any]:
    """Verify every immutable input to the time-local adjudication.

    The before/after snapshots make the read-only contract observable.  The
    external cache is fully reverified by default; tests may disable that
    expensive transitive read while still exercising direct-parent binding.
    """

    roots = {
        "memory_safe_v3_selection": Path(memory_v3_run_dir).resolve(),
        "physical_coarse_signal_witness": Path(coarse_witness_run_dir).resolve(),
        "bayes_power_calibration": Path(bayes_power_run_dir).resolve(),
    }
    expected_basenames = {
        "memory_safe_v3_selection": MEMORY_PARENT_BASENAME,
        "physical_coarse_signal_witness": WITNESS_PARENT_BASENAME,
        "bayes_power_calibration": BAYES_PARENT_BASENAME,
    }
    for role, root in roots.items():
        _require(root.is_dir(), f"{role} parent does not exist: {root}")
        _require(root.name == expected_basenames[role], f"wrong {role} basename")

    before_rows = {role: _snapshot_rows(root) for role, root in roots.items()}
    before = {
        role: snapshot_parent_run(root)
        for role, root in roots.items()
    }
    memory = _verify_memory_parent(
        roots["memory_safe_v3_selection"],
        before_rows["memory_safe_v3_selection"],
        verify_external_cache=bool(verify_external_cache),
    )
    witness = _verify_legacy_parent(
        roots["physical_coarse_signal_witness"],
        _WITNESS_SPEC,
        before_rows["physical_coarse_signal_witness"],
    )
    witness["signal"] = _verify_witness_specific(
        roots["physical_coarse_signal_witness"]
    )
    bayes = _verify_legacy_parent(
        roots["bayes_power_calibration"],
        _BAYES_SPEC,
        before_rows["bayes_power_calibration"],
    )
    transitive = _verify_transitive_provenance(
        roots["physical_coarse_signal_witness"],
        roots["bayes_power_calibration"],
        bayes,
    )
    after = {
        role: snapshot_parent_run(root)
        for role, root in roots.items()
    }
    _require(before == after, "an immutable parent changed during verification")
    for binding in (witness, bayes):
        binding.pop("registered_paths", None)
    return _hashed(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "parents": {
                "memory_safe_v3_selection": memory,
                "physical_coarse_signal_witness": witness,
                "bayes_power_calibration": bayes,
            },
            "transitive_provenance": transitive,
            "parent_immutability": {
                role: {
                    "before_sha256": before[role]["tree_sha256"],
                    "after_sha256": after[role]["tree_sha256"],
                    "file_count": before[role]["file_count"],
                    "total_bytes": before[role]["total_bytes"],
                    "unchanged": 1,
                }
                for role in roots
            },
            "all_registry_hashes_verified": 1,
            "all_registered_artifact_hashes_verified": 1,
            "all_checkpoint_hashes_verified": 1,
            "selection_seal_verified": 1,
            "cache_binding_verified": int(bool(verify_external_cache)),
            "confirmation_namespace_opened": 0,
            "confirmation_evidence_accessed": 0,
            "parents_mutated": 0,
            "new_transition_generation_performed": 0,
            "physical_training_performed": 0,
            "validation_selection_performed": 0,
            "confirmation_performed": 0,
            "controller_control_trajectory_performed": 0,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        }
    )


def verify_time_local_parents(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for :func:`verify_time_local_adjudication_parents`."""

    return verify_time_local_adjudication_parents(**kwargs)


def verify_parent_runs(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for workflow callers."""

    return verify_time_local_adjudication_parents(**kwargs)


__all__ = [
    "BAYES_CONFIG_SHA256",
    "BAYES_PARENT_BASENAME",
    "BAYES_REGISTRY_COUNT",
    "BAYES_REGISTRY_FILE_SHA256",
    "BAYES_REGISTRY_SEMANTIC_SHA256",
    "CONFIRMATION_PATH_START",
    "CONFIRMATION_PATH_STOP",
    "EXPECTED_CANDIDATE_TABLE_SHAPE",
    "EXPECTED_CRITICAL_VALUE",
    "MEMORY_PARENT_BASENAME",
    "MEMORY_PARENT_CONFIG_SHA256",
    "MEMORY_PARENT_DECISION",
    "MEMORY_PARENT_REGISTRY_COUNT",
    "MEMORY_PARENT_REGISTRY_FILE_SHA256",
    "MEMORY_PARENT_REGISTRY_SEMANTIC_SHA256",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TimeLocalProvenanceError",
    "WITNESS_CONFIG_SHA256",
    "WITNESS_DECISION",
    "WITNESS_PARENT_BASENAME",
    "WITNESS_REGISTRY_COUNT",
    "WITNESS_REGISTRY_FILE_SHA256",
    "WITNESS_REGISTRY_SEMANTIC_SHA256",
    "snapshot_parent_run",
    "verify_parent_immutability_snapshot",
    "verify_parent_runs",
    "verify_time_local_adjudication_parents",
    "verify_time_local_parents",
]
