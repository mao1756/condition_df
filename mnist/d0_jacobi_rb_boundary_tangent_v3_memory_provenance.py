"""Immutable cache provenance for the boundary-tangent v3 memory recovery.

The parent run completed a valid production cache and then failed before any
control or physical training because a full-cache model call exceeded device
memory.  This module verifies that terminal run byte-for-byte and exposes a
small, hash-bound record that a child workflow can use without copying or
mutating the roughly one-gigabyte cache.

No NPZ payload is deserialized here.  Registry verification hashes artifact
bytes, which is permitted before the physical-label opening seals used by the
trainer.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-v3-memory-provenance"
SCHEMA_VERSION = 1
CACHE_BINDING_VERSION = "d0-jacobi-rb-boundary-tangent-v3-external-cache-v1"

PARENT_RUN_BASENAME = (
    "20260805-224211_production-zero-baseline-v3-certificate-semantics-fix"
)
PARENT_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-zero-baseline-v3"
)
PARENT_REGISTRY_COUNT = 2_085
PARENT_REGISTRY_FILE_SHA256 = (
    "6d50edf754a49105f70294b9a6bacd948b2155e9d1f4f614da83f7ae7005da91"
)
PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "4e018e1913e54ad8cf0dab79d027c609db030a2b13d35a7fbc69e184022ac723"
)
PARENT_SOURCE_FINGERPRINT = (
    "860780ae957ea853d3d1254e20ab8d4db68339d9ecacdb48b730825cd07b47f9"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "c6ab05acd0a7f122646b4a5365fb85368b5a8243e062fbc4a57348793e256eb1"
)

PARENT_MANIFEST_FILE_SHA256 = (
    "659f806bbd78ce47b03630366b6dccdaf00938b329b18ce4642c81c78b5d8668"
)
PARENT_CONFIG_FILE_SHA256 = (
    "a441558f7860261aa92e35cb9d97ac1ec11018a08798be4321be323d9875d011"
)
PARENT_PREFLIGHT_GATE_FILE_SHA256 = (
    "5ba1d0110be63c6be3e210c4bee3d9592a72e2979f0c19fd6250fd5181a9cd34"
)
PARENT_CACHE_GATE_FILE_SHA256 = (
    "f875606e5b7208ee1b0a924d7388338d95a51e97a30682f767a0c328adb978c5"
)
PARENT_CACHE_METRICS_FILE_SHA256 = (
    "342c909a9e95df22f639b51af22e1c7f19d0a4a89aed727e0c261d34e431e53d"
)
PARENT_CACHE_SEAL_FILE_SHA256 = (
    "da7f6b98489a54526b5f4a36b8baa81b935d63557075b54172c768b62aa463a0"
)
PARENT_CACHE_SEAL_SEMANTIC_SHA256 = (
    "a6b8425ed755340ca2f8934874aca4fb2df7b86578e594c56ad8effc7aa3eef7"
)
PARENT_TRAIN_SEAL_FILE_SHA256 = (
    "1620ac7a5f48f68ca0879a4bdb8a85640bbcdd27528a949a459a9a46897b8d15"
)
PARENT_TRAIN_SEAL_SEMANTIC_SHA256 = (
    "eb6572bf34d12a522791da1798b65c64701fb16a4e6e4aa2fc95645d9f459b60"
)
PARENT_TRAIN_FAILURE_FILE_SHA256 = (
    "638e5865b0200c0dde89f47e80124a9297ff9f08d184d3ba5f5adfbabd6fa987"
)
PARENT_TRAIN_GATE_FILE_SHA256 = (
    "e484a3e13190a0b2263054fa8fd32dc4993434c5100a4e49a39cb4e94d5de1d1"
)
PARENT_PATH_PLAN_FILE_SHA256 = (
    "05c6194ea2aad156ecc46f64d1c858ce24b364548dc1be305335635d9fd3f6c4"
)
PARENT_COHORT_PLAN_FILE_SHA256 = (
    "816b992ba65633dbd26ac833777c356bdf39eed1750a31ff55f49dc18302060d"
)

PARENT_TRAIN_BINDING_FILE_SHA256 = (
    "2b19c0dd7dd03ff7aaf4177c8495f0150840dea06ddbf73eaaf20d8ff7e1b237"
)
PARENT_VALIDATION_BINDING_FILE_SHA256 = (
    "4ff2d05a53f00f9dd5abc1eb7111b61b995a3228e09c226b55fffa2839e0fac9"
)
PARENT_TRAIN_INDEX_FILE_SHA256 = (
    "346b9d18c34b4d18907c679ae84ee133c667cefdec8371e28ac344c992271d4a"
)
PARENT_VALIDATION_INDEX_FILE_SHA256 = (
    "c8d21e32846b643d60ed85ea48ff6284f57f2d68b58f50c576941076e6052502"
)

PARENT_STATUS_FILE_SHA256 = (
    "69d94594cba6fbc77504a07f91fb3a75ef363761eb7d2e48241ce8e39fb696d4"
)
PARENT_WORKFLOW_FILE_SHA256 = (
    "e6218c0d86821c7ec743b36b090c20dc3d908ed1605eaeec23bf9122317f9bb4"
)
PARENT_DECISION_FILE_SHA256 = (
    "180b21c1bb13fc3cdccdde5a732e84dcb492d1e6aaebf5d03cf2de5e7255ee46"
)

PARENT_HISTORICAL_DECISION = "training_controls_failed"
PARENT_HISTORICAL_FAILURE_CODE = "boundary_tangent_v3_train_execution_failed"
PARENT_HISTORICAL_FAILURE_DOMAIN = "workflow_execution"
PARENT_READJUDICATED_DECISION = "prelabel_control_memory_schedule_invalid"
PARENT_READJUDICATED_FAILURE_DOMAIN = "implementation_contract"

TRAIN_PATH_COUNT = 64
VALIDATION_PATH_COUNT = 32
TRAIN_ROW_COUNT = 114_688
VALIDATION_ROW_COUNT = 57_344
TRAIN_TRANSITION_COUNT = 134_873_088
VALIDATION_TRANSITION_COUNT = 67_436_544
CACHE_ELAPSED_SECONDS = 54_453.820496799715
FROZEN_CONFIRMATION_PROJECTION_SECONDS = 38_507.91786241904
PROJECTED_CACHE_PLUS_CONFIRMATION_SECONDS = 92_961.73835921875
TOTAL_PERSISTED_CACHE_BYTES = 1_091_008_340

# 114,688 rows x 32 channels x 28 x 28 x binary32.
FAILED_FORWARD_ACTIVATION_BYTES = 11_509_170_176
FAILED_FORWARD_ACTIVATION_GIB = 10.71875

_EXCLUDED_FILE_HASHES = {
    "artifact_registry.json": PARENT_REGISTRY_FILE_SHA256,
    "boundary_tangent_v3_decision.json": PARENT_DECISION_FILE_SHA256,
    "run_status.json": PARENT_STATUS_FILE_SHA256,
    "workflow_gate.json": PARENT_WORKFLOW_FILE_SHA256,
}

_CRITICAL_REGISTERED_HASHES = {
    "run_manifest.json": PARENT_MANIFEST_FILE_SHA256,
    "scientific_config.json": PARENT_CONFIG_FILE_SHA256,
    "preflight_gate.json": PARENT_PREFLIGHT_GATE_FILE_SHA256,
    "cache_gate.json": PARENT_CACHE_GATE_FILE_SHA256,
    "cache_metrics.json": PARENT_CACHE_METRICS_FILE_SHA256,
    "cache_artifact_seal.json": PARENT_CACHE_SEAL_FILE_SHA256,
    "train_artifact_seal.json": PARENT_TRAIN_SEAL_FILE_SHA256,
    "train_execution_failure.json": PARENT_TRAIN_FAILURE_FILE_SHA256,
    "train_metrics.json": PARENT_TRAIN_FAILURE_FILE_SHA256,
    "train_gate.json": PARENT_TRAIN_GATE_FILE_SHA256,
    "path_id_plan.json": PARENT_PATH_PLAN_FILE_SHA256,
    "cohort_plan.json": PARENT_COHORT_PLAN_FILE_SHA256,
    "cache/train_index.json": PARENT_TRAIN_BINDING_FILE_SHA256,
    "cache/validation_index.json": PARENT_VALIDATION_BINDING_FILE_SHA256,
    "eager_cache/train_index.json": PARENT_TRAIN_INDEX_FILE_SHA256,
    "eager_cache/validation_index.json": PARENT_VALIDATION_INDEX_FILE_SHA256,
}

_DOWNSTREAM_MARKERS = (
    "checkpoints",
    "controls",
    "selection",
    "confirmation",
    "training_label_open.json",
    "validation_label_open.json",
    "physical_training_started.json",
    "training_target_scale.json",
    "candidate_grid.json",
    "checkpoint_selection.json",
)


class BoundaryTangentV3MemoryProvenanceError(ArtifactCompatibilityError):
    """The immutable failed run or its external-cache commitment changed."""


MemoryRecoveryProvenanceError = BoundaryTangentV3MemoryProvenanceError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryTangentV3MemoryProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundaryTangentV3MemoryProvenanceError(
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


def _safe_path(root: Path, value: Any) -> tuple[str, Path]:
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
        raise BoundaryTangentV3MemoryProvenanceError(
            f"registry path escapes immutable parent: {value!r}"
        ) from exc
    return relative.as_posix(), target


def _verify_registry(root: Path) -> dict[str, Any]:
    registry_path = root / "artifact_registry.json"
    _require(
        registry_path.is_file()
        and file_fingerprint(registry_path) == PARENT_REGISTRY_FILE_SHA256,
        "immutable parent registry file hash changed",
    )
    registry = _load_json(registry_path, "immutable parent registry")
    artifacts = registry.get("artifacts")
    _require(
        registry.get("schema") == PARENT_RUN_SCHEMA + "-artifact-registry"
        and registry.get("schema_version") == 1
        and isinstance(artifacts, list)
        and registry.get("artifact_count") == PARENT_REGISTRY_COUNT
        and len(artifacts) == PARENT_REGISTRY_COUNT
        and registry.get("semantic_sha256") == PARENT_REGISTRY_SEMANTIC_SHA256
        and config_fingerprint({"artifacts": artifacts})
        == PARENT_REGISTRY_SEMANTIC_SHA256,
        "immutable parent registry header or semantics changed",
    )
    for name in (
        "physical_training_performed",
        "validation_selection_performed",
        "confirmation_performed",
        "sampling_performed",
        "reverse_sampling_performed",
        "reconstruction_performed",
    ):
        _require(int(registry.get(name, -1)) == 0, f"parent registry records {name}")
    _require(
        int(registry.get("production_cache_generation_performed", 0)) == 1,
        "parent registry does not record the completed cache",
    )

    registered: set[str] = set()
    for raw in artifacts:
        _require(isinstance(raw, Mapping), "immutable registry row is malformed")
        relative, target = _safe_path(root, raw.get("path"))
        _require(relative not in registered, "immutable registry path is duplicated")
        _require(
            target.is_file()
            and raw.get("sha256") == file_fingerprint(target)
            and raw.get("size") == target.stat().st_size,
            f"immutable parent artifact changed: {relative}",
        )
        registered.add(relative)

    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    _require(
        actual == registered | set(_EXCLUDED_FILE_HASHES),
        "immutable parent terminal file set changed",
    )
    for relative, expected in _EXCLUDED_FILE_HASHES.items():
        _require(
            file_fingerprint(root / relative) == expected,
            f"immutable parent terminal file changed: {relative}",
        )
    return registry


def _verify_seal(root: Path, relative: str, expected_semantic: str) -> dict[str, Any]:
    seal = _load_json(root / relative, relative)
    _assert_semantic(seal, relative)
    _require(
        seal.get("schema") == PARENT_RUN_SCHEMA + "-stage-seal"
        and seal.get("schema_version") == 1
        and seal.get("semantic_sha256") == expected_semantic,
        f"{relative} binding changed",
    )
    rows = seal.get("artifacts")
    _require(isinstance(rows, list) and bool(rows), f"{relative} is empty")
    seen: set[str] = set()
    for raw in rows:
        _require(isinstance(raw, Mapping), f"{relative} row is malformed")
        path_name, target = _safe_path(root, raw.get("path"))
        _require(path_name not in seen, f"{relative} contains a duplicate")
        _require(
            target.is_file()
            and raw.get("sha256") == file_fingerprint(target)
            and raw.get("size") == target.stat().st_size,
            f"{relative} artifact changed: {path_name}",
        )
        seen.add(path_name)
    return seal


def _verify_cache_index(
    root: Path,
    role: str,
    *,
    binding_hash: str,
    source_hash: str,
    path_count: int,
    row_count: int,
    transition_count: int,
) -> dict[str, Any]:
    binding_path = root / "cache" / f"{role}_index.json"
    source_path = root / "eager_cache" / f"{role}_index.json"
    _require(
        file_fingerprint(binding_path) == binding_hash,
        f"{role} cache binding hash changed",
    )
    _require(
        file_fingerprint(source_path) == source_hash,
        f"{role} cache index hash changed",
    )
    binding = _load_json(binding_path, f"{role} cache binding")
    source = _load_json(source_path, f"{role} cache index")
    _assert_semantic(binding, f"{role} cache binding")
    _assert_semantic(source, f"{role} cache index")
    _require(
        binding.get("role") == role
        and binding.get("source_path") == f"eager_cache/{role}_index.json"
        and binding.get("source_sha256") == source_hash,
        f"{role} cache indirection changed",
    )
    _require(
        source.get("role") == role
        and source.get("path_count") == path_count
        and source.get("input_row_count") == row_count
        and source.get("label_row_count") == row_count
        and source.get("transition_count") == transition_count
        and int(source.get("branch_input_label_separated", 0)) == 1,
        f"{role} cache scientific shape changed",
    )
    return source


def verify_failed_v3_train_parent(run_dir: str | Path) -> dict[str, Any]:
    """Verify the exact immutable OOM run and return its corrected adjudication."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"immutable failed v3 run does not exist: {root}")
    _require(root.name == PARENT_RUN_BASENAME, "wrong failed v3 run basename")
    _verify_registry(root)

    for relative, expected in _CRITICAL_REGISTERED_HASHES.items():
        _require(
            file_fingerprint(root / relative) == expected,
            f"immutable parent critical artifact changed: {relative}",
        )

    manifest = _load_json(root / "run_manifest.json", "parent manifest")
    config = _load_json(root / "scientific_config.json", "parent config")
    _assert_semantic(config, "parent scientific config")
    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA + "-manifest"
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256
        and config.get("semantic_sha256") == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "immutable parent source or scientific configuration changed",
    )

    preflight = _load_json(root / "preflight_gate.json", "parent preflight gate")
    cache_gate = _load_json(root / "cache_gate.json", "parent cache gate")
    cache_metrics = _load_json(root / "cache_metrics.json", "parent cache metrics")
    _require(
        preflight.get("evaluation_status") == "evaluated"
        and int(preflight.get("passed", 0)) == 1
        and int(preflight.get("stage_execution_valid", 0)) == 1,
        "parent preflight did not pass",
    )
    _require(
        cache_gate.get("evaluation_status") == "evaluated"
        and int(cache_gate.get("passed", 0)) == 1
        and int(cache_gate.get("stage_execution_valid", 0)) == 1
        and int(cache_gate.get("numerically_valid", 0)) == 1
        and int(cache_gate.get("resource_valid", 0)) == 1,
        "parent cache gate did not pass",
    )
    expected_cache_values = {
        "train_path_count": TRAIN_PATH_COUNT,
        "validation_path_count": VALIDATION_PATH_COUNT,
        "train_row_count": TRAIN_ROW_COUNT,
        "validation_row_count": VALIDATION_ROW_COUNT,
        "train_transition_count": TRAIN_TRANSITION_COUNT,
        "validation_transition_count": VALIDATION_TRANSITION_COUNT,
        "cache_elapsed_seconds": CACHE_ELAPSED_SECONDS,
        "frozen_confirmation_projection_seconds": (
            FROZEN_CONFIRMATION_PROJECTION_SECONDS
        ),
        "projected_cache_plus_confirmation_seconds": (
            PROJECTED_CACHE_PLUS_CONFIRMATION_SECONDS
        ),
        "total_persisted_cache_bytes": TOTAL_PERSISTED_CACHE_BYTES,
        "certificate_fraction": 1.0,
        "forbidden_event_count": 0,
    }
    _require(
        all(cache_metrics.get(name) == value for name, value in expected_cache_values.items())
        and int(cache_metrics.get("cache_complete", 0)) == 1
        and int(cache_metrics.get("confirmation_namespace_unopened", 0)) == 1,
        "parent cache metrics changed",
    )

    _verify_seal(
        root, "cache_artifact_seal.json", PARENT_CACHE_SEAL_SEMANTIC_SHA256
    )
    _verify_seal(
        root, "train_artifact_seal.json", PARENT_TRAIN_SEAL_SEMANTIC_SHA256
    )
    train_index = _verify_cache_index(
        root,
        "train",
        binding_hash=PARENT_TRAIN_BINDING_FILE_SHA256,
        source_hash=PARENT_TRAIN_INDEX_FILE_SHA256,
        path_count=TRAIN_PATH_COUNT,
        row_count=TRAIN_ROW_COUNT,
        transition_count=TRAIN_TRANSITION_COUNT,
    )
    validation_index = _verify_cache_index(
        root,
        "validation",
        binding_hash=PARENT_VALIDATION_BINDING_FILE_SHA256,
        source_hash=PARENT_VALIDATION_INDEX_FILE_SHA256,
        path_count=VALIDATION_PATH_COUNT,
        row_count=VALIDATION_ROW_COUNT,
        transition_count=VALIDATION_TRANSITION_COUNT,
    )

    failure = _load_json(
        root / "train_execution_failure.json", "parent train failure"
    )
    train_gate = _load_json(root / "train_gate.json", "parent train gate")
    status = _load_json(root / "run_status.json", "parent status")
    decision = _load_json(
        root / "boundary_tangent_v3_decision.json", "parent decision"
    )
    message = str(failure.get("message", ""))
    _require(
        failure.get("evaluation_status") == "execution_failed"
        and failure.get("failure_code") == PARENT_HISTORICAL_FAILURE_CODE
        and failure.get("failure_domain") == PARENT_HISTORICAL_FAILURE_DOMAIN
        and int(failure.get("stage_execution_valid", 1)) == 0
        and int(failure.get("scientific_evidence_complete", 1)) == 0
        and "CUDA out of memory" in message
        and "10.72 GiB" in message
        and "7.96 GiB" in message,
        "parent train failure is not the frozen deterministic OOM",
    )
    _require(
        train_gate.get("evaluation_status") == "execution_failed"
        and train_gate.get("failure_code") == PARENT_HISTORICAL_FAILURE_CODE
        and int(train_gate.get("physical_training_performed", 1)) == 0
        and status.get("state") == "execution_failed"
        and status.get("stage") == "train"
        and status.get("failure_code") == PARENT_HISTORICAL_FAILURE_CODE
        and decision.get("decision") == PARENT_HISTORICAL_DECISION,
        "parent terminal OOM records changed",
    )

    for marker in _DOWNSTREAM_MARKERS:
        _require(
            not (root / marker).exists(),
            f"immutable parent unexpectedly opened downstream evidence: {marker}",
        )

    return _hashed(
        {
            "schema": f"{SCHEMA}-failed-parent-adjudication",
            "schema_version": SCHEMA_VERSION,
            "passed": 1,
            "parent_run_dir": str(root),
            "immutable_registry": {
                "artifact_count": PARENT_REGISTRY_COUNT,
                "file_sha256": PARENT_REGISTRY_FILE_SHA256,
                "semantic_sha256": PARENT_REGISTRY_SEMANTIC_SHA256,
                "complete_file_set_verified": 1,
            },
            "source_fingerprint": PARENT_SOURCE_FINGERPRINT,
            "scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
            "historical_decision": PARENT_HISTORICAL_DECISION,
            "historical_failure_code": PARENT_HISTORICAL_FAILURE_CODE,
            "historical_failure_domain": PARENT_HISTORICAL_FAILURE_DOMAIN,
            "decision": PARENT_READJUDICATED_DECISION,
            "readjudicated_decision": PARENT_READJUDICATED_DECISION,
            "failure_domain": PARENT_READJUDICATED_FAILURE_DOMAIN,
            "stage_execution_valid": 0,
            "numerically_valid": 1,
            "resource_valid": 0,
            "scientific_evidence_complete": 0,
            "oom_basis": {
                "row_count": TRAIN_ROW_COUNT,
                "first_activation_channels": 32,
                "grid_size": 28,
                "element_bytes": 4,
                "attempted_activation_bytes": FAILED_FORWARD_ACTIVATION_BYTES,
                "attempted_activation_gib": FAILED_FORWARD_ACTIVATION_GIB,
                "arithmetic_verified": int(
                    TRAIN_ROW_COUNT * 32 * 28 * 28 * 4
                    == FAILED_FORWARD_ACTIVATION_BYTES
                ),
            },
            "cache_gate_passed": 1,
            "cache_seal_verified": 1,
            "train_index_semantic_sha256": train_index["semantic_sha256"],
            "validation_index_semantic_sha256": validation_index["semantic_sha256"],
            "control_evidence_opened": 0,
            "physical_training_performed": 0,
            "validation_selection_performed": 0,
            "confirmation_namespace_opened": 0,
            "confirmation_performed": 0,
            "cache_reuse_requires_fresh_child_preflight": 1,
            "controller_control_trajectory_performed": 0,
            "full_reverse_path_performed": 0,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        }
    )


def corrected_parent_adjudication(run_dir: str | Path) -> dict[str, Any]:
    """Compatibility alias with a descriptive public name."""

    return verify_failed_v3_train_parent(run_dir)


def build_immutable_cache_binding(run_dir: str | Path) -> dict[str, Any]:
    """Return the child workflow's small read-only external-cache commitment."""

    adjudication = verify_failed_v3_train_parent(run_dir)
    root = Path(run_dir).resolve()
    return _hashed(
        {
            "schema": f"{SCHEMA}-immutable-cache-binding",
            "schema_version": SCHEMA_VERSION,
            "binding_version": CACHE_BINDING_VERSION,
            "parent_run_dir": str(root),
            "parent_registry_count": PARENT_REGISTRY_COUNT,
            "parent_registry_file_sha256": PARENT_REGISTRY_FILE_SHA256,
            "parent_registry_semantic_sha256": PARENT_REGISTRY_SEMANTIC_SHA256,
            "parent_adjudication_sha256": adjudication["semantic_sha256"],
            "preflight_gate": {
                "path": "preflight_gate.json",
                "file_sha256": PARENT_PREFLIGHT_GATE_FILE_SHA256,
                "passed": 1,
            },
            "cache_gate": {
                "path": "cache_gate.json",
                "file_sha256": PARENT_CACHE_GATE_FILE_SHA256,
                "passed": 1,
            },
            "cache_metrics": {
                "path": "cache_metrics.json",
                "file_sha256": PARENT_CACHE_METRICS_FILE_SHA256,
            },
            "cache_artifact_seal": {
                "path": "cache_artifact_seal.json",
                "file_sha256": PARENT_CACHE_SEAL_FILE_SHA256,
                "semantic_sha256": PARENT_CACHE_SEAL_SEMANTIC_SHA256,
            },
            "path_plan": {
                "path": "path_id_plan.json",
                "file_sha256": PARENT_PATH_PLAN_FILE_SHA256,
            },
            "cohort_plan": {
                "path": "cohort_plan.json",
                "file_sha256": PARENT_COHORT_PLAN_FILE_SHA256,
            },
            "roles": {
                "train": {
                    "binding_path": "cache/train_index.json",
                    "binding_file_sha256": PARENT_TRAIN_BINDING_FILE_SHA256,
                    "source_path": "eager_cache/train_index.json",
                    "source_file_sha256": PARENT_TRAIN_INDEX_FILE_SHA256,
                    "path_count": TRAIN_PATH_COUNT,
                    "row_count": TRAIN_ROW_COUNT,
                    "transition_count": TRAIN_TRANSITION_COUNT,
                },
                "validation": {
                    "binding_path": "cache/validation_index.json",
                    "binding_file_sha256": PARENT_VALIDATION_BINDING_FILE_SHA256,
                    "source_path": "eager_cache/validation_index.json",
                    "source_file_sha256": PARENT_VALIDATION_INDEX_FILE_SHA256,
                    "path_count": VALIDATION_PATH_COUNT,
                    "row_count": VALIDATION_ROW_COUNT,
                    "transition_count": VALIDATION_TRANSITION_COUNT,
                },
            },
            "cache_elapsed_seconds": CACHE_ELAPSED_SECONDS,
            "frozen_confirmation_projection_seconds": (
                FROZEN_CONFIRMATION_PROJECTION_SECONDS
            ),
            "projected_cache_plus_confirmation_seconds": (
                PROJECTED_CACHE_PLUS_CONFIRMATION_SECONDS
            ),
            "total_persisted_cache_bytes": TOTAL_PERSISTED_CACHE_BYTES,
            "cache_is_read_only": 1,
            "cache_copied": 0,
            "cache_linked": 0,
            "physical_labels_deserialized_during_binding": 0,
            "confirmation_namespace_opened": 0,
        }
    )


def verify_immutable_cache_binding(
    record: Mapping[str, Any],
    *,
    expected_parent_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Reverify a saved binding and its immutable parent before a child stage."""

    _require(isinstance(record, Mapping), "immutable cache binding is malformed")
    binding = dict(record)
    _assert_semantic(binding, "immutable cache binding")
    parent = Path(str(binding.get("parent_run_dir", ""))).resolve()
    if expected_parent_run_dir is not None:
        _require(
            parent == Path(expected_parent_run_dir).resolve(),
            "immutable cache parent path changed",
        )
    expected = build_immutable_cache_binding(parent)
    _require(binding == expected, "immutable cache binding changed")
    return expected


def verify_immutable_cache_binding_file(
    path: str | Path,
    *,
    expected_parent_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    binding = _load_json(Path(path), "immutable cache binding")
    return verify_immutable_cache_binding(
        binding, expected_parent_run_dir=expected_parent_run_dir
    )


__all__ = [
    "BoundaryTangentV3MemoryProvenanceError",
    "CACHE_BINDING_VERSION",
    "CACHE_ELAPSED_SECONDS",
    "FAILED_FORWARD_ACTIVATION_BYTES",
    "FAILED_FORWARD_ACTIVATION_GIB",
    "FROZEN_CONFIRMATION_PROJECTION_SECONDS",
    "MemoryRecoveryProvenanceError",
    "PARENT_CACHE_SEAL_FILE_SHA256",
    "PARENT_READJUDICATED_DECISION",
    "PARENT_REGISTRY_COUNT",
    "PARENT_REGISTRY_FILE_SHA256",
    "PARENT_REGISTRY_SEMANTIC_SHA256",
    "PARENT_RUN_BASENAME",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_FINGERPRINT",
    "PROJECTED_CACHE_PLUS_CONFIRMATION_SECONDS",
    "SCHEMA",
    "SCHEMA_VERSION",
    "build_immutable_cache_binding",
    "corrected_parent_adjudication",
    "verify_failed_v3_train_parent",
    "verify_immutable_cache_binding",
    "verify_immutable_cache_binding_file",
]
