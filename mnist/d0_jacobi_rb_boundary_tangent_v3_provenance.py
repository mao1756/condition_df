"""Immutable evidence and namespace binding for zero-baseline v3.

This module is intentionally read-only with respect to all parent runs.  It
verifies the complete eager-v2 and adjudication registries, allocates the fresh
v3 namespaces inside the historical production reservation, and checks every
resume binding before callers mutate a v3 run directory.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-zero-baseline-v3-provenance"
SCHEMA_VERSION = 1
PATH_PLAN_SCHEMA = f"{SCHEMA}-path-id-plan"
PATH_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-zero-baseline-v3-path-ids-v1"
COHORT_PLAN_SCHEMA = f"{SCHEMA}-cohort-plan"
COHORT_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-zero-baseline-v3-cohorts-v1"

PATH_ID_BITS = 20
PATH_ID_LIMIT = 1 << PATH_ID_BITS
PRODUCTION_RESERVATION = (0xF0000, 0x100000)
PATH_ROLE_RANGES: dict[str, tuple[int, int]] = {
    "preflight_seam": (0xF0000, 0xF0008),
    "train": (0xF1000, 0xF1040),
    "validation": (0xF1100, 0xF1120),
    "confirmation": (0xF2000, 0xF2040),
}
BURNED_V2_CONFIRMATION_RANGE = (0xED000, 0xED040)
ROOT_SEED = 261_311
FORBIDDEN_SCHEDULER_BENCHMARK_SEED = 261_321

V2_RUN_BASENAME = "20260803-113404_production-eager-boundary-tangent-time-local"
V2_RUN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-eager-confirmation-v2"
V2_DECISION = "selection_false_discovery"
V2_SOURCE_FINGERPRINT = (
    "dfe9c3357c1d1ba614cccfdcaca84b3c3bf2d0967d6a3a3b15e5a0421d04243e"
)
V2_SCIENTIFIC_CONFIG_SHA256 = (
    "fadc1eb31ad0fb1ccb900f41f1eb8523c67c6ae39e09c783698aa5a20634cdec"
)
V2_REGISTRY_COUNT = 3_457
V2_REGISTRY_FILE_SHA256 = (
    "c996bdce5935667d247b6ce24c5e88f008c6038ec42ae25b9ea74b8b64a9a0d4"
)
V2_REGISTRY_SEMANTIC_SHA256 = (
    "36bf43c0a108549954617a78625d4fd65820141c950ba84330133de1f8648580"
)
V2_PATH_PLAN_SHA256 = (
    "5596c2ec0d3341a9f97fe9573f46eef48faa021915795c787e32d40611beb6be"
)
V2_SELECTION_SHA256 = (
    "af8f4e7481c89a3669951fd8ef70f7fdcb5617cf78b18c93f458c34b5a6a52b9"
)
V2_CONFIRMATION_SEAL_SHA256 = (
    "9f4c1ea8c3dfb6863bbbed79aee8246ee977704ba78ec376d94c4fcdfc4ab689"
)

ADJUDICATION_RUN_BASENAME = (
    "20260805-125856_production-sealed-false-discovery-adjudication"
)
ADJUDICATION_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-false-discovery-adjudication-v1"
)
ADJUDICATION_DECISION = "zero_baseline_v3_learnability_ready"
ADJUDICATION_BASELINE_CLASSIFICATION = "sealed_baseline_harm_confirmed"
ADJUDICATION_CANDIDATE_CLASSIFICATION = "selected_update_below_resolution"
ADJUDICATION_SOURCE_FINGERPRINT = (
    "a5bfbfdd84292744aef1317fb28cfd56930d126764df72dbac3c3745eaa75968"
)
ADJUDICATION_SCIENTIFIC_CONFIG_SHA256 = (
    "f7b1ffc044bdb507a6563cf2dd836b31a5aff1090d66730cb672b1287700bd9c"
)
ADJUDICATION_REGISTRY_COUNT = 272
ADJUDICATION_REGISTRY_FILE_SHA256 = (
    "e2b6f95ddaded8001bcd2cbaf24075a4bfe8423fac75787b906a8465bc4cf12c"
)
ADJUDICATION_REGISTRY_SEMANTIC_SHA256 = (
    "3aac15ae494ffc82ada769509dfaa8ef080444315fdccb045f7bae40fad5896a"
)
ADJUDICATION_DECISION_FILE_SHA256 = (
    "595630bb87e2bef77f946fb07071a6765267df88ae2950be45ac03fcd80a762e"
)
ADJUDICATION_DECISION_GATE_FILE_SHA256 = (
    "2b9c61944826b9ce4a05f0327f0c4a782feec06fe306bc1ff7f224323a5c0a5b"
)
ADJUDICATION_DECISION_SEAL_FILE_SHA256 = (
    "f5f70b1cf15dce272cd39891f148ede22ad08280c726622f8e309874c32f9e9a"
)
ADJUDICATION_DECISION_SEAL_SHA256 = (
    "7f7568a2def76c0a874ef4ead1a9ea5c2333e49e9cfaf563a2be754ba8ad57ce"
)

EAGER_PIPELINE_BASENAME = "20260803-034008_production-eager-prefix-complete-pipeline"
EAGER_PIPELINE_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-eager-pipeline-confirmation-v1"
)
EAGER_PIPELINE_REGISTRY_COUNT = 615
EAGER_PIPELINE_REGISTRY_FILE_SHA256 = (
    "25f936a6839d67789fb19846fd9beed68bdbeb7c86cf7cbd6c894d3bbf6576dd"
)
EAGER_PIPELINE_REGISTRY_SEMANTIC_SHA256 = (
    "b85907645f1b11be581f1247268729478fb7b4ff49444181663ac90467792eb7"
)
COARSE_PARENT_BASENAME = (
    "20260731-140333_production-exact-k512-coarse-residual-one-image"
)
COARSE_PARENT_RUN_SCHEMA = "experiment12-d0-jacobi-rb-coarse-residual-learnability"
COARSE_PARENT_REGISTRY_COUNT = 3_471
COARSE_PARENT_REGISTRY_FILE_SHA256 = (
    "45408753658b575d2bab52e3a1e991d97c79082923b5698c4057e1893e0d930a"
)
COARSE_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "308c6452158c1198fd6bb0b7996eeb97a708913baa36b91531fd8e5e3af2c291"
)

SOURCE_IMAGE_JSON_SHA256 = (
    "e4f6918a6bd9b01f36ebdebdcf262242dfa714e908af199bde47cb9e025591eb"
)
SOURCE_IMAGE_NPZ_SHA256 = (
    "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
)
SOURCE_IMAGE_NPZ_SIZE = 13_064
SOURCE_IMAGE_SHA256 = (
    "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
)
MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
SOURCE_LABEL = 3
SOURCE_CLASS_INDEX = 0
SOURCE_DATASET_INDEX = 7
SOURCE_LAMBDA_MIX = 0.35
SOURCE_STATE_SIZE = 784

# Exact immutable zero-baseline-v3 preflight whose scientific outputs passed
# but whose legacy seam comparator incorrectly required proof-effort metadata
# to be bit-identical between adaptive- and eager-prefix execution.  These
# bindings are intentionally separate from the live v3 source fingerprint:
# the comparator repair is additive and therefore changes the live closure.
FAILED_V3_PREFLIGHT_RUN_BASENAME = (
    "20260805-170727_production-zero-baseline-v3-learnability"
)
FAILED_V3_PREFLIGHT_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-zero-baseline-v3"
)
FAILED_V3_PREFLIGHT_REGISTRY_COUNT = 19
FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256 = (
    "94fcd4443fe2e45ff148fab9954b372bd3d3d7cf5c5efddb6dce132c8018692d"
)
FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256 = (
    "ed60b3d4130883b39e940ccb3d78f8110ceec8c979e111c21d8b43dfa21ccd3b"
)
FAILED_V3_PREFLIGHT_SOURCE_FINGERPRINT = (
    "ed5e7177d2e81ec3a18b4b877eeea3121176d9f68b38bac1a6f10063e2e83e72"
)
FAILED_V3_PREFLIGHT_SCIENTIFIC_CONFIG_SHA256 = (
    "83a07b9c8660cbd4c41d8fd42534947005e61d7de900d45ad8d94dcadb8dc31b"
)
FAILED_V3_PREFLIGHT_ARTIFACT_SEAL_FILE_SHA256 = (
    "774fceef768e465f6323af4f9f839e4b8fac974062c68fa43b1fdb6211174535"
)
FAILED_V3_PREFLIGHT_ARTIFACT_SEAL_SEMANTIC_SHA256 = (
    "e192630b2433ae24b0e6fee191e9829e4f40c81c2bd722a225de75183e6daa9f"
)
FAILED_V3_PREFLIGHT_SEAM_FILE_SHA256 = (
    "cf8576d07a30f8dfc18192a0d6c1c77de296e03b5f5c924ea9f8cff3abd1d0e0"
)
FAILED_V3_PREFLIGHT_GATE_FILE_SHA256 = (
    "a2cf13263117c5c2fd5447e06012f9a9a571ee450018fdd6e2a11c0afce7042a"
)
FAILED_V3_PREFLIGHT_METRICS_FILE_SHA256 = (
    "75f53961dfb5b7b2096329cfa208d97f4c2c66b7e1b773629fb3f85fdbd9e686"
)
FAILED_V3_PREFLIGHT_MANIFEST_FILE_SHA256 = (
    "4374d212cb11648f25f9128feb84a2349e400a6044a4f2ef6b0e695d6ffac5ac"
)
FAILED_V3_PREFLIGHT_CONFIG_FILE_SHA256 = (
    "8915f9b831d3c8f4bcde4dc269f8f8e8b2e4ccd12a2bf7a2b756a6f7a2ce6091"
)
FAILED_V3_PREFLIGHT_STATUS_FILE_SHA256 = (
    "5151272aae1d1417250680ac090f4613f2deaceffad5a6c52613de501645e51b"
)
FAILED_V3_PREFLIGHT_WORKFLOW_FILE_SHA256 = (
    "50584913ab9a8e7d7b78bcb1a07ac4731f6d767674cb79b855f445e52dfeda5f"
)
FAILED_V3_PREFLIGHT_DECISION_FILE_SHA256 = (
    "2b3005a72304a2914391f35bc8c0c21ab2c80d806a40e1854107b5efef6a64fd"
)
FAILED_V3_PREFLIGHT_HISTORICAL_DECISION = "exact_cache_invalid"
FAILED_V3_PREFLIGHT_READJUDICATED_DECISION = (
    "certificate_semantics_comparator_invalid"
)
FAILED_V3_PREFLIGHT_READJUDICATED_FAILURE_DOMAIN = "implementation_contract"

_V2_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "boundary_tangent_eager_decision.json",
}
_ADJUDICATION_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "false_discovery_decision.json",
}
_EAGER_PIPELINE_EXCLUDED = {
    "artifact_registry.json",
    "eager_pipeline_decision.json",
    "run_status.json",
    "workflow_gate.json",
}
_COARSE_PARENT_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
}
_FAILED_V3_PREFLIGHT_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "boundary_tangent_v3_decision.json",
}
_FORBIDDEN_BASELINE_NAMES = {
    "tangent_baseline.npz",
    "tangent_baseline.json",
    "baseline_numerator.npy",
    "baseline_denominator.npy",
    "baseline_count.npy",
}


class BoundaryTangentV3ProvenanceError(ArtifactCompatibilityError):
    """An immutable parent, path allocation, or resume binding changed."""


V3ProvenanceError = BoundaryTangentV3ProvenanceError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryTangentV3ProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoundaryTangentV3ProvenanceError(
            f"invalid {description}: {path}"
        ) from exc
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


def _hashed_record(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _safe_registry_path(root: Path, value: Any) -> tuple[str, Path]:
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
        raise BoundaryTangentV3ProvenanceError(
            f"registry path escapes parent: {value!r}"
        ) from exc
    return relative.as_posix(), target


def _verify_complete_registry(
    root: Path,
    *,
    description: str,
    schema: str,
    expected_count: int,
    expected_file_sha256: str,
    expected_semantic_sha256: str,
    excluded: set[str],
    semantic_scope: str,
) -> dict[str, Any]:
    path = root / "artifact_registry.json"
    _require(
        path.is_file() and file_fingerprint(path) == expected_file_sha256,
        f"{description} registry file hash changed",
    )
    registry = _load_json(path, f"{description} registry")
    artifacts = registry.get("artifacts")
    _require(
        registry.get("schema") == f"{schema}-artifact-registry"
        and registry.get("schema_version") == 1
        and isinstance(artifacts, list)
        and registry.get("artifact_count") == expected_count
        and len(artifacts) == expected_count
        and registry.get("semantic_sha256") == expected_semantic_sha256,
        f"{description} registry header changed",
    )
    if semantic_scope == "artifacts":
        _require(
            config_fingerprint({"artifacts": artifacts})
            == expected_semantic_sha256,
            f"{description} registry semantics changed",
        )
    elif semantic_scope == "artifacts_and_registry_semantics":
        semantics = registry.get("registry_semantics")
        _require(
            isinstance(semantics, Mapping)
            and config_fingerprint(
                {
                    "artifacts": artifacts,
                    "registry_semantics": dict(semantics),
                }
            )
            == expected_semantic_sha256,
            f"{description} registry semantics changed",
        )
    elif semantic_scope == "record":
        _assert_semantic(registry, f"{description} registry")
    else:
        raise BoundaryTangentV3ProvenanceError("unknown registry semantic scope")

    registered: set[str] = set()
    for raw in artifacts:
        _require(isinstance(raw, Mapping), f"{description} registry row is malformed")
        relative, target = _safe_registry_path(root, raw.get("path"))
        _require(
            relative not in excluded and relative not in registered,
            f"{description} registry path is duplicated or excluded",
        )
        _require(
            target.is_file()
            and raw.get("sha256") == file_fingerprint(target)
            and raw.get("size") == target.stat().st_size,
            f"{description} artifact changed: {relative}",
        )
        registered.add(relative)
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.relative_to(root).as_posix() not in excluded
        and not item.name.endswith(".tmp")
        and ".tmp." not in item.name
    }
    _require(actual == registered, f"{description} terminal file set changed")
    return registry


def _verify_registry_header(
    root: Path,
    *,
    description: str,
    expected_count: int,
    expected_file_sha256: str,
    expected_semantic_sha256: str,
) -> None:
    path = root / "artifact_registry.json"
    _require(
        path.is_file() and file_fingerprint(path) == expected_file_sha256,
        f"{description} registry file changed",
    )
    record = _load_json(path, f"{description} registry")
    _require(
        record.get("artifact_count") == expected_count
        and record.get("semantic_sha256") == expected_semantic_sha256,
        f"{description} registry binding changed",
    )


def verify_and_re_adjudicate_failed_v3_preflight(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Verify and re-adjudicate the exact immutable v3 comparator failure.

    The historical seam compared ``batch_certificate_sha256`` values.  That
    digest intentionally includes proof-effort metadata (mode and prefix-bit
    distributions), which may differ between adaptive- and eager-prefix
    execution even when the transition, target, and certificate-code arrays
    are identical.  The historical ``base_targets_equal`` field compares the
    multipath ``batch_output_sha256`` digest, whose payload is the later
    fraction, Rao--Blackwell target, and certificate code for every
    transition.  Consequently the saved evidence supports an implementation
    contract re-adjudication, but it does not authorize cache generation: a
    fresh preflight under the corrected comparator remains mandatory.

    This function performs no writes and deliberately verifies the complete
    terminal file set, including the four terminal files excluded from the
    registry's artifact rows.
    """

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"failed v3 preflight does not exist: {root}")
    _require(
        root.name == FAILED_V3_PREFLIGHT_RUN_BASENAME,
        "wrong failed v3 preflight basename",
    )
    registry = _verify_complete_registry(
        root,
        description="failed v3 preflight",
        schema=FAILED_V3_PREFLIGHT_RUN_SCHEMA,
        expected_count=FAILED_V3_PREFLIGHT_REGISTRY_COUNT,
        expected_file_sha256=FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256,
        expected_semantic_sha256=FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256,
        excluded=_FAILED_V3_PREFLIGHT_EXCLUDED,
        semantic_scope="artifacts",
    )

    # The registry format excludes mutable terminal summaries by design.  For
    # this historical binding those summaries are frozen too, and no orphan,
    # temporary, or downstream artifact is allowed beside the exact 23 files.
    terminal_hashes = {
        "run_status.json": FAILED_V3_PREFLIGHT_STATUS_FILE_SHA256,
        "workflow_gate.json": FAILED_V3_PREFLIGHT_WORKFLOW_FILE_SHA256,
        "boundary_tangent_v3_decision.json": (
            FAILED_V3_PREFLIGHT_DECISION_FILE_SHA256
        ),
    }
    for relative, expected_sha256 in terminal_hashes.items():
        path = root / relative
        _require(
            path.is_file() and file_fingerprint(path) == expected_sha256,
            f"failed v3 preflight terminal file changed: {relative}",
        )
    registered = {
        str(row["path"])
        for row in registry["artifacts"]
        if isinstance(row, Mapping)
    }
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    _require(
        actual == registered | _FAILED_V3_PREFLIGHT_EXCLUDED,
        "failed v3 preflight exact terminal file set changed",
    )

    exact_registered_hashes = {
        "run_manifest.json": FAILED_V3_PREFLIGHT_MANIFEST_FILE_SHA256,
        "scientific_config.json": FAILED_V3_PREFLIGHT_CONFIG_FILE_SHA256,
        "preflight_artifact_seal.json": (
            FAILED_V3_PREFLIGHT_ARTIFACT_SEAL_FILE_SHA256
        ),
        "preflight_scheduler_seam.json": FAILED_V3_PREFLIGHT_SEAM_FILE_SHA256,
        "preflight_gate.json": FAILED_V3_PREFLIGHT_GATE_FILE_SHA256,
        "preflight_metrics.json": FAILED_V3_PREFLIGHT_METRICS_FILE_SHA256,
    }
    for relative, expected_sha256 in exact_registered_hashes.items():
        path = root / relative
        _require(
            path.is_file() and file_fingerprint(path) == expected_sha256,
            f"failed v3 preflight exact artifact changed: {relative}",
        )

    manifest = _load_json(root / "run_manifest.json", "failed v3 manifest")
    config = _load_json(root / "scientific_config.json", "failed v3 config")
    source_closure = _load_json(
        root / "source_closure.json", "failed v3 source closure"
    )
    seal = _load_json(
        root / "preflight_artifact_seal.json", "failed v3 preflight seal"
    )
    seam = _load_json(
        root / "preflight_scheduler_seam.json", "failed v3 scheduler seam"
    )
    metrics = _load_json(
        root / "preflight_metrics.json", "failed v3 preflight metrics"
    )
    gate = _load_json(root / "preflight_gate.json", "failed v3 preflight gate")
    workflow = _load_json(root / "workflow_gate.json", "failed v3 workflow")
    decision = _load_json(
        root / "boundary_tangent_v3_decision.json", "failed v3 decision"
    )
    status = _load_json(root / "run_status.json", "failed v3 status")
    path_plan = _load_json(root / "path_id_plan.json", "failed v3 path plan")

    _require(
        manifest.get("schema") == FAILED_V3_PREFLIGHT_RUN_SCHEMA + "-manifest"
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint")
        == FAILED_V3_PREFLIGHT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == FAILED_V3_PREFLIGHT_SCIENTIFIC_CONFIG_SHA256,
        "failed v3 manifest binding changed",
    )
    _assert_semantic(config, "failed v3 scientific config")
    _assert_semantic(source_closure, "failed v3 source closure")
    _assert_semantic(seal, "failed v3 preflight seal")
    _require(
        config.get("schema")
        == FAILED_V3_PREFLIGHT_RUN_SCHEMA + "-scientific-config"
        and config.get("semantic_sha256")
        == FAILED_V3_PREFLIGHT_SCIENTIFIC_CONFIG_SHA256
        and source_closure.get("source_fingerprint")
        == FAILED_V3_PREFLIGHT_SOURCE_FINGERPRINT
        and seal.get("semantic_sha256")
        == FAILED_V3_PREFLIGHT_ARTIFACT_SEAL_SEMANTIC_SHA256,
        "failed v3 semantic binding changed",
    )
    _require(
        status.get("schema") == FAILED_V3_PREFLIGHT_RUN_SCHEMA + "-status"
        and status.get("state") == "gate_failed"
        and status.get("stage") == "terminal"
        and status.get("decision") == FAILED_V3_PREFLIGHT_HISTORICAL_DECISION,
        "failed v3 historical status changed",
    )
    _require(
        decision.get("schema")
        == FAILED_V3_PREFLIGHT_RUN_SCHEMA + "-gate-decision"
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == FAILED_V3_PREFLIGHT_HISTORICAL_DECISION,
        "failed v3 historical decision changed",
    )

    expected_seam = {
        "base_states_equal": 1,
        # This is the batch-output digest: later fractions + RB targets +
        # certificate codes, despite the legacy field's narrower name.
        "base_targets_equal": 1,
        "base_certificates_equal": 0,
        "midpoint_states_equal": 1,
        "midpoint_targets_equal": 1,
        "midpoint_certificates_equal": 1,
        "certificate_fraction": 1.0,
        "forbidden_event_count": 0,
        "passed": 0,
    }
    _require(
        seam.get("schema")
        == FAILED_V3_PREFLIGHT_RUN_SCHEMA + "-scheduler-seam"
        and all(seam.get(name) == value for name, value in expected_seam.items())
        and seam.get("maximum_mass_error") == 4.440892098500626e-16
        and seam.get("transitions_per_second") == 4661.284278044443,
        "failed v3 scheduler seam evidence changed",
    )
    _require(
        metrics.get("schema")
        == FAILED_V3_PREFLIGHT_RUN_SCHEMA + "-preflight-metrics"
        and metrics.get("scheduler_seam_valid") == 0
        and metrics.get("certificate_fraction") == 1.0
        and metrics.get("forbidden_event_count") == 0
        and metrics.get("inherited_resource_projection_valid") == 1
        and metrics.get("transitions_per_second") == 3604.471434567184
        and metrics.get("peak_memory_fraction") == 0.008759760158424649,
        "failed v3 preflight metrics changed",
    )
    checks = gate.get("checks")
    _require(isinstance(checks, Mapping), "failed v3 gate checks are malformed")
    failed_checks = {
        str(name)
        for name, raw in checks.items()
        if not isinstance(raw, Mapping) or int(raw.get("passed", 0)) != 1
    }
    _require(
        gate.get("schema")
        == FAILED_V3_PREFLIGHT_RUN_SCHEMA + "-gate-preflight-gate"
        and gate.get("evaluation_status") == "evaluated"
        and gate.get("passed") == 0
        and gate.get("stage_execution_valid") == 1
        and gate.get("numerically_valid") == 1
        and gate.get("resource_valid") == 1
        and failed_checks == {"scheduler_seam_valid"},
        "failed v3 preflight is not the exact comparator-only failure",
    )

    components = workflow.get("components")
    _require(
        workflow.get("schema")
        == FAILED_V3_PREFLIGHT_RUN_SCHEMA + "-gate-workflow"
        and workflow.get("required_gate") == "preflight"
        and workflow.get("required_gate_pass") == 0
        and isinstance(components, Mapping)
        and dict(components.get("preflight", {})).get("evaluation_status")
        == "evaluated"
        and all(
            dict(components.get(name, {})).get("evaluation_status")
            == "not_evaluated"
            for name in ("cache", "train", "select", "confirm")
        ),
        "failed v3 preflight unexpectedly opened a downstream gate",
    )
    role_slots = path_plan.get("role_slots")
    _assert_semantic(path_plan, "failed v3 path plan")
    _require(
        isinstance(role_slots, Mapping)
        and all(
            isinstance(raw, Mapping) and int(raw.get("opened", -1)) == 0
            for raw in role_slots.values()
        )
        and path_plan.get("confirmation_reserved_unopened") == 1,
        "failed v3 preflight opened a production path role",
    )

    forbidden_downstream_paths = (
        "cache",
        "eager_cache",
        "checkpoints",
        "selection",
        "confirmation",
        "validation",
        "cache_metrics.json",
        "cache_gate.json",
        "training_label_open.json",
        "validation_label_open.json",
        "physical_training_started.json",
        "train_gate.json",
        "select_gate.json",
        "checkpoint_selection.json",
        "confirmation_namespace_open.json",
        "confirmation_metrics.json",
        "confirm_gate.json",
    )
    _require(
        not any((root / relative).exists() for relative in forbidden_downstream_paths),
        "failed v3 preflight contains downstream evidence",
    )
    no_work_fields = (
        "production_cache_generation_performed",
        "physical_training_performed",
        "validation_selection_performed",
        "confirmation_performed",
        "controller_control_trajectory_performed",
        "full_dataset_training_performed",
        "full_reverse_path_performed",
        "image_sampling_performed",
        "sampling_performed",
        "reverse_sampling_performed",
        "reconstruction_performed",
    )
    for description, record in (
        ("registry", registry),
        ("status", status),
        ("decision", decision),
        ("workflow", workflow),
        ("preflight gate", gate),
        ("preflight metrics", metrics),
        ("preflight seal", seal),
    ):
        _require(
            all(int(record.get(name, 0)) == 0 for name in no_work_fields),
            f"failed v3 {description} records downstream work",
        )

    return _hashed_record(
        {
            "schema": f"{SCHEMA}-failed-preflight-readjudication",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "failed_run_dir": str(root),
            "failed_run_basename": FAILED_V3_PREFLIGHT_RUN_BASENAME,
            "immutable_registry": {
                "artifact_count": FAILED_V3_PREFLIGHT_REGISTRY_COUNT,
                "file_sha256": FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256,
                "semantic_sha256": (
                    FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256
                ),
                "complete_file_set_verified": 1,
            },
            "historical_source_fingerprint": (
                FAILED_V3_PREFLIGHT_SOURCE_FINGERPRINT
            ),
            "historical_scientific_config_sha256": (
                FAILED_V3_PREFLIGHT_SCIENTIFIC_CONFIG_SHA256
            ),
            "historical_decision": FAILED_V3_PREFLIGHT_HISTORICAL_DECISION,
            "historical_failure_domain": "execution",
            "readjudicated_decision": (
                FAILED_V3_PREFLIGHT_READJUDICATED_DECISION
            ),
            "readjudicated_failure_domain": (
                FAILED_V3_PREFLIGHT_READJUDICATED_FAILURE_DOMAIN
            ),
            "decision": FAILED_V3_PREFLIGHT_READJUDICATED_DECISION,
            "failure_domain": FAILED_V3_PREFLIGHT_READJUDICATED_FAILURE_DOMAIN,
            "readjudication_basis": {
                "base_final_states_bit_identical": 1,
                "base_output_digest_bit_identical": 1,
                "base_output_digest_includes_later_target_and_certificate_codes": 1,
                "midpoint_states_bit_identical": 1,
                "midpoint_targets_bit_identical": 1,
                "midpoint_certificate_codes_bit_identical": 1,
                "both_arms_fully_certified": 1,
                "proof_effort_metadata_digest_bit_identical": 0,
                "proof_effort_metadata_equality_required": 0,
                "only_historical_failed_check": "scheduler_seam_valid",
            },
            "stage_execution_valid": 1,
            "numerically_valid": 1,
            "resource_valid": 1,
            "scientific_evidence_complete": 1,
            "production_path_roles_opened": 0,
            "downstream_scientific_evidence_opened": 0,
            "failed_run_resume_authorized": 0,
            "fresh_preflight_required": 1,
            "fresh_preflight_authorized": 1,
            "cache_generation_authorized": 0,
            "physical_training_authorized": 0,
            "validation_selection_authorized": 0,
            "confirmation_authorized": 0,
            "controller_control_planning_authorized": 0,
            "parent_artifacts_mutated": 0,
            **{name: 0 for name in no_work_fields},
        }
    )


# Historical modules used the compact spelling.  Keep a public alias while
# making the readable ``re_adjudicate`` spelling canonical for new callers.
verify_and_readjudicate_failed_v3_preflight = (
    verify_and_re_adjudicate_failed_v3_preflight
)


def build_v3_path_plan() -> dict[str, Any]:
    """Build the frozen active allocation within the broad reservation."""

    roles = {
        role: list(range(start, stop))
        for role, (start, stop) in PATH_ROLE_RANGES.items()
    }
    return _hashed_record(
        {
            "schema": PATH_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "path_id_plan_version": PATH_PLAN_VERSION,
            "root_seed": ROOT_SEED,
            "canonical_path_id_bits": PATH_ID_BITS,
            "allocator_reservation": {
                "start": PRODUCTION_RESERVATION[0],
                "stop_exclusive": PRODUCTION_RESERVATION[1],
                "claim_kind": "enclosing_allocator_reservation",
            },
            "roles": roles,
            "role_slots": {
                role: {
                    "start": start,
                    "stop_exclusive": stop,
                    "start_hex": f"0x{start:05X}",
                    "stop_exclusive_hex": f"0x{stop:05X}",
                    "path_count": stop - start,
                    "opened": 0,
                }
                for role, (start, stop) in PATH_ROLE_RANGES.items()
            },
            "confirmation_reserved_unopened": 1,
            "burned_v2_confirmation_path_ids": list(
                range(*BURNED_V2_CONFIRMATION_RANGE)
            ),
            "burned_v2_confirmation_reuse_authorized": 0,
            "silent_remapping_performed": 0,
            "collision_free": 1,
        }
    )


def _claim_values(value: Any) -> set[int]:
    if isinstance(value, Mapping):
        if "start" in value and "stop_exclusive" in value:
            start = value["start"]
            stop = value["stop_exclusive"]
            _require(
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(stop, int)
                and not isinstance(stop, bool),
                "path claim interval is malformed",
            )
            return set(range(start, stop))
        if "path_ids" in value:
            value = value["path_ids"]
    try:
        raw_values = tuple(value)
    except TypeError as exc:
        raise BoundaryTangentV3ProvenanceError("path claim is not iterable") from exc
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


def validate_v3_path_plan(
    plan: Mapping[str, Any],
    *,
    claimed_ids: Mapping[str, Any] | Iterable[int] | None = None,
) -> dict[str, Any]:
    """Reject drift and every overlap except the exact allocator reservation."""

    expected = build_v3_path_plan()
    _require(dict(plan) == expected, "v3 path plan changed")
    active_by_role = {
        role: set(range(start, stop))
        for role, (start, stop) in PATH_ROLE_RANGES.items()
    }
    active = set().union(*active_by_role.values())
    for role, values in active_by_role.items():
        start, stop = PATH_ROLE_RANGES[role]
        _require(
            PRODUCTION_RESERVATION[0] <= start < stop <= PRODUCTION_RESERVATION[1],
            f"{role} is outside the production reservation",
        )
        _require(len(values) == stop - start, f"{role} path slot is malformed")
    roles = tuple(active_by_role)
    for index, role in enumerate(roles):
        for other in roles[index + 1 :]:
            _require(
                active_by_role[role].isdisjoint(active_by_role[other]),
                f"v3 path roles overlap: {role} and {other}",
            )
    burned = set(range(*BURNED_V2_CONFIRMATION_RANGE))
    _require(active.isdisjoint(burned), "v3 paths reuse burned v2 confirmation IDs")

    if claimed_ids is None:
        claims: Mapping[str, Any] = {}
    elif isinstance(claimed_ids, Mapping):
        claims = claimed_ids
    else:
        claims = {"external": claimed_ids}
    allocator = set(range(*PRODUCTION_RESERVATION))
    collisions: list[dict[str, Any]] = []
    allowed_allocator_claims: list[str] = []
    for source, raw in claims.items():
        values = _claim_values(raw)
        normalized = str(source).lower()
        is_allocator = "reserv" in normalized and values == allocator
        if is_allocator:
            allowed_allocator_claims.append(str(source))
            continue
        for value in sorted(active & values):
            collisions.append({"source": str(source), "path_id": value})
    _require(not collisions, "v3 path IDs collide with an active/source/run claim")
    return _hashed_record(
        {
            "schema": f"{PATH_PLAN_SCHEMA}-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "path_id_plan_sha256": expected["semantic_sha256"],
            "active_path_count": len(active),
            "role_disjointness_pass": 1,
            "twenty_bit_bounds_pass": 1,
            "burned_v2_confirmation_reuse": 0,
            "collision_count": 0,
            "allowed_allocator_claims": sorted(allowed_allocator_claims),
            "silent_remapping_performed": 0,
        }
    )


def _partition(
    path_ids: tuple[int, ...],
    sizes: tuple[int, ...],
    roles: Mapping[int, str],
    kind: str,
) -> list[dict[str, Any]]:
    _require(sum(sizes) == len(path_ids), f"{kind} cohort sizes do not partition IDs")
    result: list[dict[str, Any]] = []
    cursor = 0
    for index, size in enumerate(sizes):
        values = path_ids[cursor : cursor + size]
        cursor += size
        result.append(
            {
                "index": index,
                "kind": kind,
                "size": size,
                "path_ids": list(values),
                "path_roles": [roles[value] for value in values],
            }
        )
    return result


def build_v3_cohort_plan(
    path_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build stable P10/P6 and P10/P4 execution partitions."""

    plan = build_v3_path_plan() if path_plan is None else dict(path_plan)
    validate_v3_path_plan(plan)
    train = tuple(plan["roles"]["train"])
    validation = tuple(plan["roles"]["validation"])
    confirmation = tuple(plan["roles"]["confirmation"])
    role_by_id = {value: "train" for value in train}
    role_by_id.update({value: "validation" for value in validation})
    role_by_id.update({value: "confirmation" for value in confirmation})
    train_validation_sizes = (10,) * 9 + (6,)
    confirmation_sizes = (10,) * 6 + (4,)
    return _hashed_record(
        {
            "schema": COHORT_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "cohort_plan_version": COHORT_PLAN_VERSION,
            "path_id_plan_sha256": plan["semantic_sha256"],
            "maximum_cohort_size": 10,
            "train_validation_sizes": list(train_validation_sizes),
            "confirmation_sizes": list(confirmation_sizes),
            "train_validation": _partition(
                train + validation,
                train_validation_sizes,
                role_by_id,
                "train_validation",
            ),
            "confirmation": _partition(
                confirmation,
                confirmation_sizes,
                role_by_id,
                "confirmation",
            ),
            "mixed_train_validation_cohort_index": 6,
            "mixed_role_artifact_commit_authorized": 0,
            "split_by_role_before_commit": 1,
            "confirmation_reserved_unopened": 1,
        }
    )


def validate_v3_cohort_plan(
    cohort_plan: Mapping[str, Any],
    *,
    path_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_path = build_v3_path_plan() if path_plan is None else dict(path_plan)
    expected = build_v3_cohort_plan(expected_path)
    _require(dict(cohort_plan) == expected, "v3 cohort plan changed")
    for kind in ("train_validation", "confirmation"):
        flattened: list[int] = []
        for index, record in enumerate(expected[kind]):
            _require(record["index"] == index, "cohort indices are not canonical")
            _require(
                record["size"] == len(record["path_ids"]) <= 10,
                "cohort size exceeds exact scheduler limit",
            )
            _require(
                len(record["path_roles"]) == record["size"],
                "cohort role alignment changed",
            )
            flattened.extend(record["path_ids"])
        _require(len(flattened) == len(set(flattened)), "cohort IDs are duplicated")
    mixed = expected["train_validation"][6]
    _require(
        mixed["path_roles"] == ["train"] * 4 + ["validation"] * 6,
        "the frozen cross-role cohort changed",
    )
    return _hashed_record(
        {
            "schema": f"{COHORT_PLAN_SCHEMA}-validation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "cohort_plan_sha256": expected["semantic_sha256"],
            "canonical_indices": 1,
            "unique_twenty_bit_ids": 1,
            "maximum_cohort_size": 10,
            "mixed_cohort_split_before_commit": 1,
        }
    )


def v3_source_paths(
    paths: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    values = (Path(__file__),) if paths is None else tuple(Path(item) for item in paths)
    resolved = tuple(
        sorted({item.resolve() for item in values}, key=lambda item: item.as_posix())
    )
    _require(bool(resolved), "v3 source closure is empty")
    _require(all(item.is_file() for item in resolved), "v3 source closure is incomplete")
    return resolved


def v3_source_fingerprint(paths: Iterable[str | Path] | None = None) -> str:
    return source_fingerprint(v3_source_paths(paths))


def v3_transitive_source_paths(
    entry_points: Iterable[str | Path],
) -> tuple[Path, ...]:
    """Return the complete statically imported local ``mnist`` closure.

    The package initializer is part of the executable import contract.  Every
    local module reached by either ``import mnist.foo`` or
    ``from mnist import foo`` is followed recursively.  Missing local modules
    fail closed instead of silently shrinking the source fingerprint.
    """

    package_root = Path(__file__).resolve().parent
    initializer = package_root / "__init__.py"
    queue = list(v3_source_paths(entry_points))
    if initializer.is_file():
        queue.append(initializer.resolve())
    resolved: set[Path] = set()
    while queue:
        path = queue.pop(0).resolve()
        if path in resolved:
            continue
        _require(path.is_file(), f"v3 source dependency is missing: {path}")
        try:
            path.relative_to(package_root)
        except ValueError as exc:
            raise BoundaryTangentV3ProvenanceError(
                f"v3 source dependency is outside the mnist package: {path}"
            ) from exc
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise BoundaryTangentV3ProvenanceError(
                f"cannot parse v3 source dependency: {path}"
            ) from exc
        resolved.add(path)
        module_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names.add(node.module)
                if node.module == "mnist":
                    module_names.update(
                        f"mnist.{alias.name}" for alias in node.names
                    )
        for module_name in sorted(module_names):
            if not module_name.startswith("mnist."):
                continue
            candidate = package_root.joinpath(*module_name.split(".")[1:]).with_suffix(
                ".py"
            )
            if candidate.is_file() and candidate.resolve() not in resolved:
                queue.append(candidate.resolve())
    return v3_source_paths(resolved)


def source_measure_sha256(value: np.ndarray) -> str:
    """Reproduce the frozen source-image semantic hash convention."""

    measured = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
    digest = hashlib.sha256()
    digest.update(str(measured.shape).encode("ascii"))
    digest.update(measured.tobytes(order="C"))
    return digest.hexdigest()


def verify_v3_source_image_binding(parent_coarse_residual_run_dir: str | Path) -> dict[str, Any]:
    """Measure the immutable source archive and both represented simplex states."""

    root = Path(parent_coarse_residual_run_dir).resolve()
    _require(
        root.is_dir() and root.name == COARSE_PARENT_BASENAME,
        "wrong coarse-residual source-image parent",
    )
    metadata_path = root / "source_image.json"
    archive_path = root / "source_image.npz"
    _require(
        metadata_path.is_file()
        and file_fingerprint(metadata_path) == SOURCE_IMAGE_JSON_SHA256,
        "source-image metadata file hash changed",
    )
    _require(
        archive_path.is_file()
        and archive_path.stat().st_size == SOURCE_IMAGE_NPZ_SIZE
        and file_fingerprint(archive_path) == SOURCE_IMAGE_NPZ_SHA256,
        "source-image archive binding changed",
    )
    metadata = _load_json(metadata_path, "coarse-residual source-image metadata")
    expected_metadata = {
        "label": SOURCE_LABEL,
        "class_index": SOURCE_CLASS_INDEX,
        "dataset_index": SOURCE_DATASET_INDEX,
        "lambda_mix": SOURCE_LAMBDA_MIX,
        "image_sha256": SOURCE_IMAGE_SHA256,
        "mixed_target_sha256": MIXED_TARGET_SHA256,
        "npz_sha256": SOURCE_IMAGE_NPZ_SHA256,
        "npz_size": SOURCE_IMAGE_NPZ_SIZE,
    }
    _require(
        all(metadata.get(name) == expected for name, expected in expected_metadata.items()),
        "source-image metadata binding changed",
    )
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == {"image", "mixed_target"},
                "source-image archive schema changed",
            )
            image = np.array(archive["image"], copy=True)
            mixed = np.array(archive["mixed_target"], copy=True)
    except (OSError, ValueError, KeyError) as exc:
        raise BoundaryTangentV3ProvenanceError(
            "cannot load the frozen source-image archive"
        ) from exc
    for name, value in (("image", image), ("mixed target", mixed)):
        _require(
            value.dtype == np.float64
            and value.shape == (SOURCE_STATE_SIZE,)
            and bool(np.isfinite(value).all())
            and bool(np.all(value >= 0.0))
            and math.isclose(
                float(np.sum(value)), 1.0, rel_tol=0.0, abs_tol=2.0e-12
            ),
            f"frozen source {name} is invalid",
        )
    measured_image_sha256 = source_measure_sha256(image)
    measured_mixed_target_sha256 = source_measure_sha256(mixed)
    _require(
        measured_image_sha256 == SOURCE_IMAGE_SHA256
        and measured_mixed_target_sha256 == MIXED_TARGET_SHA256,
        "source-image reconstructed tensor hash changed",
    )
    return _hashed_record(
        {
            "schema": f"{SCHEMA}-source-image-binding",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "parent_run_dir": str(root),
            "source_image_json_sha256": file_fingerprint(metadata_path),
            "source_image_npz_sha256": file_fingerprint(archive_path),
            "source_image_npz_size": int(archive_path.stat().st_size),
            "measured_image_sha256": measured_image_sha256,
            "measured_mixed_target_sha256": measured_mixed_target_sha256,
            "image_dtype": str(image.dtype),
            "mixed_target_dtype": str(mixed.dtype),
            "image_shape": list(image.shape),
            "mixed_target_shape": list(mixed.shape),
            "image_simplex_mass": float(np.sum(image)),
            "mixed_target_simplex_mass": float(np.sum(mixed)),
            "lambda_mix": SOURCE_LAMBDA_MIX,
        }
    )


def validate_no_v3_baseline_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Reject every fitted-baseline artifact or checkpoint key in a v3 run."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"v3 run does not exist: {root}")
    violations: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        relative = path.relative_to(root).as_posix()
        if name in _FORBIDDEN_BASELINE_NAMES or any(
            token in name
            for token in (
                "baseline_numerator",
                "baseline_denominator",
                "baseline_count",
            )
        ):
            violations.append(relative)
        if path.suffix.lower() == ".json":
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if '"_q_values"' in text:
                violations.append(f"{relative}:_q_values")
        if path.suffix.lower() in {".pt", ".pth"}:
            try:
                if zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path, mode="r") as checkpoint:
                        pickle_members = tuple(
                            name
                            for name in checkpoint.namelist()
                            if PurePosixPath(name).name == "data.pkl"
                        )
                        _require(
                            bool(pickle_members),
                            f"checkpoint has no data.pkl payload: {relative}",
                        )
                        has_forbidden_key = any(
                            b"_q_values" in checkpoint.read(name)
                            for name in pickle_members
                        )
                else:
                    has_forbidden_key = False
                    overlap = b""
                    with path.open("rb") as handle:
                        while chunk := handle.read(1024 * 1024):
                            joined = overlap + chunk
                            if b"_q_values" in joined:
                                has_forbidden_key = True
                                break
                            overlap = joined[-len(b"_q_values") + 1 :]
                if has_forbidden_key:
                    violations.append(f"{relative}:_q_values")
            except (OSError, zipfile.BadZipFile, KeyError, RuntimeError) as exc:
                raise BoundaryTangentV3ProvenanceError(
                    f"cannot inspect v3 checkpoint for fitted baseline: {relative}"
                ) from exc
    _require(not violations, "v3 run contains fitted-baseline evidence")
    return _hashed_record(
        {
            "schema": f"{SCHEMA}-baseline-artifact-scan",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "forbidden_baseline_artifact_count": 0,
        }
    )


def _verify_v2_parent(root: Path) -> dict[str, Any]:
    _require(root.is_dir(), f"v2 parent does not exist: {root}")
    _require(root.name == V2_RUN_BASENAME, "wrong v2 parent basename")
    _verify_complete_registry(
        root,
        description="eager-v2 parent",
        schema=V2_RUN_SCHEMA,
        expected_count=V2_REGISTRY_COUNT,
        expected_file_sha256=V2_REGISTRY_FILE_SHA256,
        expected_semantic_sha256=V2_REGISTRY_SEMANTIC_SHA256,
        excluded=_V2_EXCLUDED,
        semantic_scope="artifacts",
    )
    manifest = _load_json(root / "run_manifest.json", "v2 manifest")
    config = _load_json(root / "scientific_config.json", "v2 config")
    path_plan = _load_json(root / "path_id_plan.json", "v2 path plan")
    selection = _load_json(root / "checkpoint_selection.json", "v2 selection")
    seal = _load_json(root / "confirmation_seal.json", "v2 confirmation seal")
    status = _load_json(root / "run_status.json", "v2 status")
    decision = _load_json(
        root / "boundary_tangent_eager_decision.json", "v2 decision"
    )
    _require(
        manifest.get("source_fingerprint") == V2_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == V2_SCIENTIFIC_CONFIG_SHA256,
        "v2 manifest binding changed",
    )
    _assert_semantic(config, "v2 scientific config")
    _assert_semantic(path_plan, "v2 path plan")
    _assert_semantic(selection, "v2 selection")
    _assert_semantic(seal, "v2 confirmation seal")
    _require(
        config.get("semantic_sha256") == V2_SCIENTIFIC_CONFIG_SHA256
        and path_plan.get("semantic_sha256") == V2_PATH_PLAN_SHA256
        and selection.get("semantic_sha256") == V2_SELECTION_SHA256
        and seal.get("semantic_sha256") == V2_CONFIRMATION_SEAL_SHA256,
        "v2 semantic binding changed",
    )
    _require(
        status.get("decision") == V2_DECISION
        and status.get("state") == "gate_failed"
        and int(status.get("scientific_evidence_complete", 0)) == 1
        and decision.get("decision") == V2_DECISION
        and int(decision.get("controller_control_planning_authorized", 0)) == 0,
        "v2 terminal decision changed",
    )
    return {
        "run_dir": str(root),
        "basename": root.name,
        "decision": V2_DECISION,
        "source_fingerprint": V2_SOURCE_FINGERPRINT,
        "scientific_config_sha256": V2_SCIENTIFIC_CONFIG_SHA256,
        "path_plan_sha256": V2_PATH_PLAN_SHA256,
        "selection_sha256": V2_SELECTION_SHA256,
        "confirmation_seal_sha256": V2_CONFIRMATION_SEAL_SHA256,
        "registry": {
            "artifact_count": V2_REGISTRY_COUNT,
            "file_sha256": V2_REGISTRY_FILE_SHA256,
            "semantic_sha256": V2_REGISTRY_SEMANTIC_SHA256,
        },
        "old_confirmation_paths_burned": 1,
        "verified": 1,
    }


def _verify_adjudication(root: Path, v2_root: Path) -> dict[str, Any]:
    _require(root.is_dir(), f"adjudication does not exist: {root}")
    _require(root.name == ADJUDICATION_RUN_BASENAME, "wrong adjudication basename")
    _verify_complete_registry(
        root,
        description="completed adjudication",
        schema=ADJUDICATION_RUN_SCHEMA,
        expected_count=ADJUDICATION_REGISTRY_COUNT,
        expected_file_sha256=ADJUDICATION_REGISTRY_FILE_SHA256,
        expected_semantic_sha256=ADJUDICATION_REGISTRY_SEMANTIC_SHA256,
        excluded=_ADJUDICATION_EXCLUDED,
        semantic_scope="record",
    )
    exact_files = {
        "false_discovery_decision.json": ADJUDICATION_DECISION_FILE_SHA256,
        "decision_gate.json": ADJUDICATION_DECISION_GATE_FILE_SHA256,
        "decision_artifact_seal.json": ADJUDICATION_DECISION_SEAL_FILE_SHA256,
    }
    for relative, digest in exact_files.items():
        path = root / relative
        _require(
            path.is_file() and file_fingerprint(path) == digest,
            f"adjudication file changed: {relative}",
        )
    manifest = _load_json(root / "run_manifest.json", "adjudication manifest")
    config = _load_json(root / "scientific_config.json", "adjudication config")
    decision = _load_json(
        root / "false_discovery_decision.json", "adjudication decision"
    )
    gate = _load_json(root / "decision_gate.json", "adjudication decision gate")
    seal = _load_json(
        root / "decision_artifact_seal.json", "adjudication decision seal"
    )
    status = _load_json(root / "run_status.json", "adjudication status")
    _assert_semantic(config, "adjudication scientific config")
    _assert_semantic(seal, "adjudication decision seal")
    _require(
        manifest.get("source_fingerprint") == ADJUDICATION_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == ADJUDICATION_SCIENTIFIC_CONFIG_SHA256
        and Path(str(manifest.get("parent_run_dir", ""))).resolve() == v2_root,
        "adjudication manifest binding changed",
    )
    _require(
        config.get("semantic_sha256") == ADJUDICATION_SCIENTIFIC_CONFIG_SHA256
        and seal.get("semantic_sha256") == ADJUDICATION_DECISION_SEAL_SHA256,
        "adjudication semantic binding changed",
    )
    required_decision = {
        "decision": ADJUDICATION_DECISION,
        "baseline_classification": ADJUDICATION_BASELINE_CLASSIFICATION,
        "candidate_classification": ADJUDICATION_CANDIDATE_CLASSIFICATION,
        "old_confirmation_paths_burned": 1,
        "old_confirmation_reuse_authorized": 0,
        "zero_baseline_v3_design_planning_authorized": 1,
        "fresh_v3_learnability_design_planning_authorized": 1,
        "cache_generation_authorized": 0,
        "physical_training_authorized": 0,
        "confirmation_authorized": 0,
        "controller_control_planning_authorized": 0,
    }
    _require(
        all(decision.get(name) == value for name, value in required_decision.items()),
        "adjudication authority changed",
    )
    _require(
        gate.get("passed") == 1
        and gate.get("terminal_decision") == ADJUDICATION_DECISION
        and status.get("decision") == ADJUDICATION_DECISION
        and status.get("state") == "complete",
        "adjudication terminal gate changed",
    )
    return {
        "run_dir": str(root),
        "basename": root.name,
        "decision": ADJUDICATION_DECISION,
        "baseline_classification": ADJUDICATION_BASELINE_CLASSIFICATION,
        "candidate_classification": ADJUDICATION_CANDIDATE_CLASSIFICATION,
        "source_fingerprint": ADJUDICATION_SOURCE_FINGERPRINT,
        "scientific_config_sha256": ADJUDICATION_SCIENTIFIC_CONFIG_SHA256,
        "decision_artifact_seal_sha256": ADJUDICATION_DECISION_SEAL_SHA256,
        "registry": {
            "artifact_count": ADJUDICATION_REGISTRY_COUNT,
            "file_sha256": ADJUDICATION_REGISTRY_FILE_SHA256,
            "semantic_sha256": ADJUDICATION_REGISTRY_SEMANTIC_SHA256,
        },
        **{name: required_decision[name] for name in required_decision if name != "decision"},
        "verified": 1,
    }


def verify_v3_parent_evidence(
    *,
    parent_v2_run_dir: str | Path,
    adjudication_run_dir: str | Path,
    parent_eager_pipeline_run_dir: str | Path,
    parent_coarse_residual_run_dir: str | Path,
) -> dict[str, Any]:
    """Verify exact v2/adjudication registries and their transitive bindings."""

    v2_root = Path(parent_v2_run_dir).resolve()
    adjudication_root = Path(adjudication_run_dir).resolve()
    eager_root = Path(parent_eager_pipeline_run_dir).resolve()
    coarse_root = Path(parent_coarse_residual_run_dir).resolve()
    _require(
        eager_root.is_dir() and eager_root.name == EAGER_PIPELINE_BASENAME,
        "wrong eager-pipeline parent",
    )
    _require(
        coarse_root.is_dir() and coarse_root.name == COARSE_PARENT_BASENAME,
        "wrong coarse-residual parent",
    )
    v2 = _verify_v2_parent(v2_root)
    adjudication = _verify_adjudication(adjudication_root, v2_root)
    parent_record = _load_json(v2_root / "parent_provenance.json", "v2 provenance")
    _assert_semantic(parent_record, "v2 provenance")
    parents = parent_record.get("parents")
    _require(isinstance(parents, Mapping), "v2 transitive parents changed")
    eager = parents.get("successful_eager_pipeline")
    coarse = parents.get("successful_coarse_residual")
    _require(isinstance(eager, Mapping) and isinstance(coarse, Mapping), "v2 ancestry changed")
    _require(
        Path(str(eager.get("run_dir", ""))).resolve() == eager_root
        and eager.get("registry", {}).get("artifact_count")
        == EAGER_PIPELINE_REGISTRY_COUNT
        and eager.get("registry", {}).get("file_sha256")
        == EAGER_PIPELINE_REGISTRY_FILE_SHA256
        and eager.get("registry", {}).get("semantic_sha256")
        == EAGER_PIPELINE_REGISTRY_SEMANTIC_SHA256,
        "eager-pipeline transitive binding changed",
    )
    _require(
        Path(str(coarse.get("run_dir", ""))).resolve() == coarse_root
        and coarse.get("registry_count") == COARSE_PARENT_REGISTRY_COUNT
        and coarse.get("registry_file_sha256") == COARSE_PARENT_REGISTRY_FILE_SHA256
        and coarse.get("registry_semantic_sha256")
        == COARSE_PARENT_REGISTRY_SEMANTIC_SHA256,
        "coarse-residual transitive binding changed",
    )
    _verify_complete_registry(
        eager_root,
        description="eager-pipeline parent",
        schema=EAGER_PIPELINE_RUN_SCHEMA,
        expected_count=EAGER_PIPELINE_REGISTRY_COUNT,
        expected_file_sha256=EAGER_PIPELINE_REGISTRY_FILE_SHA256,
        expected_semantic_sha256=EAGER_PIPELINE_REGISTRY_SEMANTIC_SHA256,
        excluded=_EAGER_PIPELINE_EXCLUDED,
        semantic_scope="artifacts_and_registry_semantics",
    )
    _verify_complete_registry(
        coarse_root,
        description="coarse-residual parent",
        schema=COARSE_PARENT_RUN_SCHEMA,
        expected_count=COARSE_PARENT_REGISTRY_COUNT,
        expected_file_sha256=COARSE_PARENT_REGISTRY_FILE_SHA256,
        expected_semantic_sha256=COARSE_PARENT_REGISTRY_SEMANTIC_SHA256,
        excluded=_COARSE_PARENT_EXCLUDED,
        semantic_scope="artifacts",
    )
    source_image_binding = verify_v3_source_image_binding(coarse_root)
    return _hashed_record(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "parents": {
                "eager_v2": v2,
                "completed_adjudication": adjudication,
                "eager_pipeline": {
                    "run_dir": str(eager_root),
                    "registry_file_sha256": EAGER_PIPELINE_REGISTRY_FILE_SHA256,
                    "registry_semantic_sha256": EAGER_PIPELINE_REGISTRY_SEMANTIC_SHA256,
                    "registry_count": EAGER_PIPELINE_REGISTRY_COUNT,
                    "verified": 1,
                },
                "coarse_residual": {
                    "run_dir": str(coarse_root),
                    "registry_file_sha256": COARSE_PARENT_REGISTRY_FILE_SHA256,
                    "registry_semantic_sha256": COARSE_PARENT_REGISTRY_SEMANTIC_SHA256,
                    "registry_count": COARSE_PARENT_REGISTRY_COUNT,
                    "verified": 1,
                },
            },
            "source_image_binding": source_image_binding,
            "old_confirmation_paths_burned": 1,
            "old_confirmation_reuse_authorized": 0,
            "fresh_v3_learnability_workflow_authorized": 1,
            "cache_generation_authorized": 0,
            "physical_training_authorized": 0,
            "confirmation_authorized": 0,
            "controller_control_planning_authorized": 0,
            "parent_artifacts_mutated": 0,
        }
    )


def build_v3_adjudication_authorization(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_semantic(provenance, "v3 parent provenance")
    parents = provenance.get("parents")
    _require(
        provenance.get("passed") == 1 and isinstance(parents, Mapping),
        "v3 parent provenance did not pass",
    )
    adjudication = parents.get("completed_adjudication")
    _require(
        isinstance(adjudication, Mapping)
        and adjudication.get("decision") == ADJUDICATION_DECISION
        and adjudication.get("zero_baseline_v3_design_planning_authorized") == 1
        and adjudication.get("fresh_v3_learnability_design_planning_authorized") == 1,
        "adjudication does not authorize design of this fresh workflow",
    )
    return _hashed_record(
        {
            "schema": f"{SCHEMA}-adjudication-authorization",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "adjudication_decision": ADJUDICATION_DECISION,
            "fresh_v3_workflow_is_separately_reviewed": 1,
            "old_confirmation_paths_burned": 1,
            "old_confirmation_reuse_authorized": 0,
            "cache_generation_authorized_by_adjudication": 0,
            "physical_training_authorized_by_adjudication": 0,
            "confirmation_authorized_by_adjudication": 0,
            "controller_control_planning_authorized_by_adjudication": 0,
        }
    )


def verify_v3_resume_compatibility(
    run_dir: str | Path,
    *,
    source_fingerprint_value: str,
    scientific_config_sha256: str,
    parent_provenance_sha256: str,
    adjudication_provenance_sha256: str,
    adjudication_authorization_sha256: str,
    path_plan_sha256: str,
    cohort_plan_sha256: str,
    zero_baseline_contract_sha256: str,
    target_and_input_contract_sha256: str,
    certificate_semantics_contract_sha256: str | None = None,
    failed_v3_preflight_adjudication_sha256: str | None = None,
    certificate_semantics_comparator_version: str | None = None,
) -> dict[str, Any]:
    """Verify all immutable v3 bindings before a resumed run is mutated."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"resume run does not exist: {root}")
    expected = {
        "source_fingerprint": source_fingerprint_value,
        "scientific_config_sha256": scientific_config_sha256,
        "parent_provenance_sha256": parent_provenance_sha256,
        "adjudication_provenance_sha256": adjudication_provenance_sha256,
        "adjudication_authorization_sha256": adjudication_authorization_sha256,
        "path_plan_sha256": path_plan_sha256,
        "cohort_plan_sha256": cohort_plan_sha256,
        "zero_baseline_contract_sha256": zero_baseline_contract_sha256,
        "target_and_input_contract_sha256": target_and_input_contract_sha256,
    }
    repaired_bindings = (
        certificate_semantics_contract_sha256,
        failed_v3_preflight_adjudication_sha256,
        certificate_semantics_comparator_version,
    )
    _require(
        all(value is None for value in repaired_bindings)
        or all(isinstance(value, str) and bool(value) for value in repaired_bindings),
        "resume certificate-semantics bindings are incomplete",
    )
    if all(value is not None for value in repaired_bindings):
        expected.update(
            {
                "certificate_semantics_contract_sha256": (
                    certificate_semantics_contract_sha256
                ),
                "failed_v3_preflight_adjudication_sha256": (
                    failed_v3_preflight_adjudication_sha256
                ),
                "certificate_semantics_comparator_version": (
                    certificate_semantics_comparator_version
                ),
            }
        )
    manifest = _load_json(root / "run_manifest.json", "resume manifest")
    config = _load_json(root / "scientific_config.json", "resume config")
    _assert_semantic(config, "resume scientific config")
    _require(
        config.get("semantic_sha256") == scientific_config_sha256
        and config.get("root_seed") == ROOT_SEED
        and config.get("forbidden_scheduler_benchmark_seed")
        == FORBIDDEN_SCHEDULER_BENCHMARK_SEED,
        "resume scientific configuration changed",
    )
    if certificate_semantics_comparator_version is not None:
        _require(
            config.get("certificate_semantics_comparator_version")
            == certificate_semantics_comparator_version
            and config.get("certificate_semantics_contract_sha256")
            == certificate_semantics_contract_sha256
            and config.get("failed_v3_preflight_adjudication_sha256")
            == failed_v3_preflight_adjudication_sha256
            and config.get("failed_v3_preflight_registry_count")
            == FAILED_V3_PREFLIGHT_REGISTRY_COUNT
            and config.get("failed_v3_preflight_registry_semantic_sha256")
            == FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256
            and config.get("failed_v3_preflight_registry_file_sha256")
            == FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256,
            "resume certificate-semantics scientific binding changed",
        )
    _require(
        all(manifest.get(name) == value for name, value in expected.items()),
        "resume manifest compatibility changed",
    )
    records = {
        "parent_provenance": "parent_provenance.json",
        "adjudication_provenance": "adjudication_provenance.json",
        "adjudication_authorization": "adjudication_authorization.json",
        "path_plan": "path_id_plan.json",
        "cohort_plan": "cohort_plan.json",
        "zero_baseline_contract": "zero_baseline_contract.json",
        "target_and_input_contract": "target_and_input_contract.json",
    }
    if certificate_semantics_contract_sha256 is not None:
        records.update(
            {
                "certificate_semantics_contract": (
                    "certificate_semantics_contract.json"
                ),
                "failed_v3_preflight_adjudication": (
                    "failed_v3_preflight_adjudication.json"
                ),
            }
        )
    hashes = {name: expected[f"{name}_sha256"] for name in records}
    loaded: dict[str, dict[str, Any]] = {}
    for name, relative in records.items():
        record = _load_json(root / relative, f"resume {name.replace('_', ' ')}")
        _assert_semantic(record, f"resume {name.replace('_', ' ')}")
        _require(
            record.get("semantic_sha256") == hashes[name],
            f"resume {name.replace('_', ' ')} changed",
        )
        loaded[name] = record
    validate_v3_path_plan(loaded["path_plan"])
    validate_v3_cohort_plan(loaded["cohort_plan"], path_plan=loaded["path_plan"])
    validate_no_v3_baseline_artifacts(root)
    return _hashed_record(
        {
            "schema": f"{SCHEMA}-resume-compatibility",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "run_dir": str(root),
            **expected,
        }
    )


build_boundary_tangent_v3_path_plan = build_v3_path_plan
validate_boundary_tangent_v3_path_plan = validate_v3_path_plan
build_boundary_tangent_v3_cohort_plan = build_v3_cohort_plan
validate_boundary_tangent_v3_cohort_plan = validate_v3_cohort_plan
boundary_tangent_v3_source_paths = v3_source_paths
boundary_tangent_v3_source_fingerprint = v3_source_fingerprint
verify_boundary_tangent_v3_parents = verify_v3_parent_evidence
verify_boundary_tangent_v3_resume_compatibility = verify_v3_resume_compatibility


__all__ = [
    "ADJUDICATION_DECISION",
    "ADJUDICATION_REGISTRY_COUNT",
    "ADJUDICATION_REGISTRY_FILE_SHA256",
    "ADJUDICATION_REGISTRY_SEMANTIC_SHA256",
    "BURNED_V2_CONFIRMATION_RANGE",
    "BoundaryTangentV3ProvenanceError",
    "COHORT_PLAN_SCHEMA",
    "COHORT_PLAN_VERSION",
    "FAILED_V3_PREFLIGHT_ARTIFACT_SEAL_FILE_SHA256",
    "FAILED_V3_PREFLIGHT_ARTIFACT_SEAL_SEMANTIC_SHA256",
    "FAILED_V3_PREFLIGHT_GATE_FILE_SHA256",
    "FAILED_V3_PREFLIGHT_HISTORICAL_DECISION",
    "FAILED_V3_PREFLIGHT_READJUDICATED_DECISION",
    "FAILED_V3_PREFLIGHT_READJUDICATED_FAILURE_DOMAIN",
    "FAILED_V3_PREFLIGHT_REGISTRY_COUNT",
    "FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256",
    "FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256",
    "FAILED_V3_PREFLIGHT_RUN_BASENAME",
    "FAILED_V3_PREFLIGHT_RUN_SCHEMA",
    "FAILED_V3_PREFLIGHT_SCIENTIFIC_CONFIG_SHA256",
    "FAILED_V3_PREFLIGHT_SEAM_FILE_SHA256",
    "FAILED_V3_PREFLIGHT_SOURCE_FINGERPRINT",
    "PATH_ID_BITS",
    "PATH_ID_LIMIT",
    "PATH_PLAN_SCHEMA",
    "PATH_PLAN_VERSION",
    "PATH_ROLE_RANGES",
    "PRODUCTION_RESERVATION",
    "ROOT_SEED",
    "MIXED_TARGET_SHA256",
    "SOURCE_CLASS_INDEX",
    "SOURCE_DATASET_INDEX",
    "SOURCE_IMAGE_JSON_SHA256",
    "SOURCE_IMAGE_NPZ_SHA256",
    "SOURCE_IMAGE_NPZ_SIZE",
    "SOURCE_IMAGE_SHA256",
    "SOURCE_LAMBDA_MIX",
    "V2_DECISION",
    "V2_REGISTRY_COUNT",
    "V2_REGISTRY_FILE_SHA256",
    "V2_REGISTRY_SEMANTIC_SHA256",
    "V3ProvenanceError",
    "boundary_tangent_v3_source_fingerprint",
    "boundary_tangent_v3_source_paths",
    "build_boundary_tangent_v3_cohort_plan",
    "build_boundary_tangent_v3_path_plan",
    "build_v3_adjudication_authorization",
    "build_v3_cohort_plan",
    "build_v3_path_plan",
    "v3_source_fingerprint",
    "v3_source_paths",
    "v3_transitive_source_paths",
    "source_measure_sha256",
    "validate_boundary_tangent_v3_cohort_plan",
    "validate_boundary_tangent_v3_path_plan",
    "validate_no_v3_baseline_artifacts",
    "validate_v3_cohort_plan",
    "validate_v3_path_plan",
    "verify_boundary_tangent_v3_parents",
    "verify_boundary_tangent_v3_resume_compatibility",
    "verify_and_re_adjudicate_failed_v3_preflight",
    "verify_and_readjudicate_failed_v3_preflight",
    "verify_v3_parent_evidence",
    "verify_v3_resume_compatibility",
    "verify_v3_source_image_binding",
]
