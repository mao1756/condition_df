"""Immutable provenance for recovering the completed Haar panel-A shards.

The bound parent is a terminal, failed controls-only run.  Its numerical work
completed, but the panel aggregator looked for ``record.schedule`` even though
the scheduler's canonical schema stores it at ``record.identity.schedule``.
This module verifies the complete parent and its predecessor, validates every
committed restart chain on CPU, and exposes the defect without modifying either
run directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    canonical_json_bytes,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_haar_gate import (
    evaluate_haar_coupling,
    evaluate_haar_preflight,
    evaluate_haar_workflow,
)
from mnist.d0_jacobi_rb_haar_provenance import (
    verify_right_endpoint_coupling_parent,
)
from mnist.d0_jacobi_rb_haar_scheduler import (
    ANTITHETIC_PROFILE_NAME,
    HAAR_SCHEDULER_VERSION,
    NESTED_PROFILE_NAME,
    HaarSchedulerError,
    HaarShardIdentity,
    NestedHaarSchedule,
    PairwiseHaarAntitheticSchedule,
    expected_haar_shard_input_sha256,
    initialize_nested_branch_states,
    load_committed_haar_shard,
)


PARENT_RUN_BASENAME = (
    "20260725-212650_production-certified-haar-strang-power-adapter-fix-v2"
)
PARENT_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-hierarchical-coupling-confirmation"
)
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 197
PARENT_REGISTRY_SHA256 = (
    "4bf1dab4c0905533fe0df885521fb3309ed6344e13f1fd67faad7fa9ae11abfe"
)
PARENT_SOURCE_COUNT = 35
PARENT_SOURCE_FINGERPRINT = (
    "300bcdab17d9cac5605311bf0b513a5c476e88011662fb1e51ac69ca4f431c39"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "cb26ea614d20f7695b02fa063aafca41d0a229ff34a26e9bbb0610bdda1352cf"
)
PARENT_PATH_ID_PLAN_SHA256 = (
    "7250522feb92331abb1b54733ffde00ae083c1fddce033390870cc3d9b63e4b4"
)
PARENT_PROFILE_PLAN_SHA256 = (
    "fff841b8e07b6328ae6b3c9e5b3e62820b97e00a972f30f1e1da04062c0aa828"
)
PARENT_MODEL_INPUT_CONTRACT_SHA256 = (
    "e40c640938204366883af214faf5a4b8deb50aa909f4535b3ed5048f88c9e4f0"
)
PARENT_DECISION = "hierarchical_scheduler_invalid"
PARENT_FAILURE_CODE = "hierarchical_panel_diagnostics_invalid"
PARENT_FAILURE_MESSAGE = "panel shard schedule binding is missing"
PARENT_RE_ADJUDICATION = "panel_schedule_binding_invalid"
PARENT_ROOT_SEED = 261_181

PARENT_NESTED_MAIN_SHARD_COUNT = 16
PARENT_NESTED_REFERENCE_SHARD_COUNT = 64
PARENT_NESTED_SHARD_COUNT = 80
PARENT_TRANSITION_COUNT = 120_823_808
PARENT_FALLBACK_COUNT = 38
PARENT_MAXIMUM_MASS_ERROR = 1.3322676295501878e-15
PARENT_CONSERVATIVE_RATE = 4202.42901955144
PARENT_COUPLING_PEAK_MEMORY_FRACTION = 0.006818991116410677

SCHEDULE_BINDING_VERSION = "d0-jacobi-rb-haar-shard-schedule-binding-v1"

_FORBIDDEN_COUNTS = (
    "uncertified_count",
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "projection_count",
    "renormalization_count",
    "nonfinite_count",
)
_NO_WORK_KEYS = {
    "physical_training_performed",
    "production_refinement_performed",
    "sampling_performed",
    "reverse_sampling_performed",
}
_REQUIRED_FILES = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "scientific_config": "scientific_config.json",
    "preflight_gate": "haar_preflight_gate.json",
    "preflight_metrics": "haar_preflight_metrics.json",
    "coupling_gate": "haar_coupling_gate.json",
    "coupling_metrics": "haar_coupling_metrics.json",
    "pilot_gate": "haar_pilot_gate.json",
    "pilot_failure": "pilot_failure.json",
    "decision": "haar_coupling_decision.json",
    "workflow": "haar_workflow_gate.json",
    "transitive_provenance": "parent_provenance.json",
    "corrected_parent": "corrected_parent_adjudication.json",
    "path_id_plan": "haar_path_id_plan.json",
    "profile_plan": "haar_profile_plan.json",
    "sealed_panels": "sealed_panel_registry.json",
    "model_input_contract": "future_model_input_contract.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read Haar recovery parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"parent artifact is not an object: {path}")
    return dict(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int, np.bool_, np.integer)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int, np.bool_, np.integer)) and int(value) == 0


def _integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _metrics(record: Mapping[str, Any], description: str) -> dict[str, Any]:
    value = record.get("metrics")
    _require(isinstance(value, Mapping), f"{description} metrics are invalid")
    return dict(value)


def _no_work_tree(value: Any, description: str) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name in _NO_WORK_KEYS:
                _require(_zero(item), f"{description} records forbidden work: {name}")
            _no_work_tree(item, description)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _no_work_tree(item, description)


def _normalized_schedule(value: Mapping[str, Any]) -> dict[str, Any]:
    profile = value.get("profile_name")
    try:
        if profile == NESTED_PROFILE_NAME:
            levels = value.get("levels")
            if (
                not isinstance(levels, Sequence)
                or isinstance(levels, (str, bytes, bytearray))
                or not all(_integer(item) for item in levels)
            ):
                raise ValueError("nested levels are invalid")
            schedule = NestedHaarSchedule(
                pool=value.get("pool"),
                role=value.get("role"),
                levels=tuple(int(item) for item in levels),
            )
            _require(
                _integer(value.get("coarsest_steps"))
                and _integer(value.get("finest_steps"))
                and _one(value.get("single_arm")),
                "nested shard schedule fields are malformed",
            )
        elif profile == ANTITHETIC_PROFILE_NAME:
            signs = value.get("fine_arms")
            if (
                not isinstance(signs, Sequence)
                or isinstance(signs, (str, bytes, bytearray))
                or not all(_integer(item) for item in signs)
            ):
                raise ValueError("antithetic fine-arm signs are invalid")
            schedule = PairwiseHaarAntitheticSchedule(
                coarse_steps=value.get("coarse_steps"),
                fine_steps=value.get("fine_steps"),
                role=value.get("role"),
            )
            _require(
                _one(value.get("pair_local_tree"))
                and list(int(item) for item in signs) == [-1, 1],
                "antithetic shard schedule fields are malformed",
            )
        else:
            raise ValueError("unknown profile")
    except (TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            "committed Haar shard has a malformed schedule"
        ) from exc
    normalized = schedule.to_record()
    _require(
        canonical_json_bytes(value) == canonical_json_bytes(normalized),
        "committed Haar shard schedule is not canonical",
    )
    return normalized


def canonical_shard_schedule(
    record: Mapping[str, Any],
    *,
    expected_profile_name: str | None = None,
    expected_pool: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Return the version-1 canonical schedule from a committed shard record.

    ``identity.schedule`` is authoritative.  A legacy or projected top-level
    schedule is accepted only when it is canonically identical.
    """

    if not isinstance(record, Mapping):
        raise ArtifactCompatibilityError("committed Haar shard is not an object")
    identity = record.get("identity")
    _require(isinstance(identity, Mapping), "committed Haar shard identity is missing")
    raw = identity.get("schedule")
    _require(
        isinstance(raw, Mapping),
        "committed Haar shard identity schedule is missing",
    )
    schedule = _normalized_schedule(raw)
    if "schedule" in record:
        top_level = record.get("schedule")
        _require(
            isinstance(top_level, Mapping)
            and canonical_json_bytes(top_level) == canonical_json_bytes(schedule),
            "top-level and identity Haar shard schedules conflict",
        )
    if expected_profile_name is not None:
        _require(
            schedule.get("profile_name") == expected_profile_name,
            "committed Haar shard profile is incompatible",
        )
    if expected_pool is not None:
        _require(
            schedule.get("pool") == expected_pool,
            "committed Haar shard pool is incompatible",
        )
    if expected_role is not None:
        _require(
            schedule.get("role") == expected_role,
            "committed Haar shard role is incompatible",
        )
    return schedule


def canonical_shard_schedule_binding(
    record: Mapping[str, Any],
    *,
    expected_profile_name: str | None = None,
    expected_pool: str | None = None,
    expected_role: str | None = None,
) -> dict[str, Any]:
    """Return an auditable binding record around :func:`canonical_shard_schedule`."""

    schedule = canonical_shard_schedule(
        record,
        expected_profile_name=expected_profile_name,
        expected_pool=expected_pool,
        expected_role=expected_role,
    )
    return {
        "schema": SCHEDULE_BINDING_VERSION,
        "schema_version": 1,
        "binding_source": "identity.schedule",
        "top_level_schedule_present": int("schedule" in record),
        "schedule": schedule,
    }


def _verify_registry(
    root: Path,
    registry_path: Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    registry = _load(registry_path)
    digest = file_fingerprint(registry_path)
    _require(digest == PARENT_REGISTRY_SHA256, "parent registry SHA-256 changed")
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and registry.get("schema_version") == 1,
        "parent registry schema changed",
    )
    raw_records = registry.get("records")
    _require(isinstance(raw_records, Mapping), "parent registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == PARENT_REGISTRY_RECORD_COUNT,
        f"parent registry must contain {PARENT_REGISTRY_RECORD_COUNT} records",
    )
    excluded = set(
        registry.get("terminal_files_excluded_to_avoid_self_reference", ())
    )
    _require(
        excluded == {"artifact_registry.json", "run_status.json"},
        "parent registry exclusions changed",
    )
    actual = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file()
        and artifact.relative_to(root).as_posix() not in excluded
    }
    _require(actual == set(records), "parent registry file set changed")
    for relative, raw in records.items():
        path = Path(str(relative))
        _require(
            not path.is_absolute() and ".." not in path.parts,
            f"invalid parent registry path: {relative}",
        )
        _require(isinstance(raw, Mapping), f"invalid registry row: {relative}")
        artifact = root / path
        _require(
            artifact.is_file()
            and raw.get("sha256") == file_fingerprint(artifact)
            and raw.get("size") == artifact.stat().st_size,
            f"registered parent artifact changed: {relative}",
        )
    _require(
        status.get("artifact_registry_sha256") == digest
        and status.get("artifact_registry_record_count")
        == PARENT_REGISTRY_RECORD_COUNT
        and status.get("artifact_registry_size") == registry_path.stat().st_size,
        "parent status does not bind its terminal registry",
    )
    return registry


def _verify_sources_and_config(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[Path]:
    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and manifest.get("schema_version") == 1
        and manifest.get("source_count") == PARENT_SOURCE_COUNT
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent manifest/source/config binding changed",
    )
    raw_sources = manifest.get("source_paths")
    _require(
        isinstance(raw_sources, list)
        and len(raw_sources) == PARENT_SOURCE_COUNT
        and all(isinstance(item, str) for item in raw_sources),
        "parent immutable source set changed",
    )
    sources = [Path(item) for item in raw_sources]
    _require(
        len({path.resolve() for path in sources}) == PARENT_SOURCE_COUNT
        and all(path.is_file() for path in sources)
        and source_fingerprint(sources) == PARENT_SOURCE_FINGERPRINT,
        "one of the thirty-five immutable parent sources changed",
    )
    _require(
        config_fingerprint(config) == PARENT_SCIENTIFIC_CONFIG_SHA256
        and config.get("schema") == PARENT_RUN_SCHEMA + "-scientific-config"
        and config.get("schema_version") == 1
        and config.get("root_seed") == PARENT_ROOT_SEED
        and config.get("grid_size") == 28
        and config.get("alpha") == 1.0
        and config.get("tau_eff") == 5.0e-5
        and config.get("sample_steps") == [128, 256, 512, 1024, 2048]
        and config.get("panel_cluster_count") == 8
        and config.get("profile_order")
        == [NESTED_PROFILE_NAME, ANTITHETIC_PROFILE_NAME]
        and config.get("path_id_plan_sha256") == PARENT_PATH_ID_PLAN_SHA256
        and config.get("profile_plan_sha256") == PARENT_PROFILE_PLAN_SHA256
        and config.get("model_input_contract_sha256")
        == PARENT_MODEL_INPUT_CONTRACT_SHA256
        and _zero(config.get("test_only_reduced_workload")),
        "parent scientific configuration changed",
    )
    return sources


def _without_self_hash(record: Mapping[str, Any], name: str) -> dict[str, Any]:
    result = dict(record)
    result.pop(name, None)
    return result


def _verify_frozen_plans(
    path_plan: Mapping[str, Any],
    profile_plan: Mapping[str, Any],
    sealed: Mapping[str, Any],
    model_contract: Mapping[str, Any],
) -> None:
    _require(
        path_plan.get("schema") == PARENT_RUN_SCHEMA + "-path-id-plan"
        and path_plan.get("schema_version") == 1
        and path_plan.get("version") == "d0-jacobi-rb-haar-path-id-v1"
        and path_plan.get("rng_plan_version")
        == "d0-jacobi-rb-certified-haar-rng-v1"
        and path_plan.get("panel_cluster_count") == 8
        and path_plan.get("all_path_id_count") == 112
        and path_plan.get("path_id_plan_sha256") == PARENT_PATH_ID_PLAN_SHA256
        and config_fingerprint(
            _without_self_hash(path_plan, "path_id_plan_sha256")
        )
        == PARENT_PATH_ID_PLAN_SHA256,
        "parent path-ID plan changed",
    )
    checks = path_plan.get("checks")
    _require(
        isinstance(checks, Mapping)
        and bool(checks)
        and all(_one(value) for value in checks.values()),
        "parent path-ID plan checks changed",
    )
    _require(
        profile_plan.get("schema") == PARENT_RUN_SCHEMA + "-profile-plan"
        and profile_plan.get("schema_version") == 1
        and profile_plan.get("profile_order")
        == [NESTED_PROFILE_NAME, ANTITHETIC_PROFILE_NAME]
        and profile_plan.get("profile_plan_sha256")
        == PARENT_PROFILE_PLAN_SHA256
        and config_fingerprint(
            _without_self_hash(profile_plan, "profile_plan_sha256")
        )
        == PARENT_PROFILE_PLAN_SHA256,
        "parent profile plan changed",
    )
    _require(
        sealed.get("schema") == PARENT_RUN_SCHEMA + "-sealed-panel-registry"
        and sealed.get("schema_version") == 1
        and sealed.get("path_id_plan_sha256") == PARENT_PATH_ID_PLAN_SHA256
        and sealed.get("profile_plan_sha256") == PARENT_PROFILE_PLAN_SHA256
        and sealed.get("model_input_contract_sha256")
        == PARENT_MODEL_INPUT_CONTRACT_SHA256
        and _one(sealed.get("panels_frozen_before_device_execution"))
        and _one(sealed.get("panels_disjoint"))
        and _zero(sealed.get("panel_regeneration_permitted"))
        and sealed.get("panel_b_evaluation_limit") == 1,
        "parent sealed-panel registry changed",
    )
    _require(
        model_contract.get("schema")
        == PARENT_RUN_SCHEMA + "-future-model-input-contract"
        and model_contract.get("schema_version") == 1
        and model_contract.get("model_input_contract_sha256")
        == PARENT_MODEL_INPUT_CONTRACT_SHA256
        and config_fingerprint(
            _without_self_hash(
                model_contract, "model_input_contract_sha256"
            )
        )
        == PARENT_MODEL_INPUT_CONTRACT_SHA256
        and _one(model_contract.get("later_state_only_contract_pass")),
        "parent future-model input contract changed",
    )


def _verify_local_outcome(
    *,
    status: Mapping[str, Any],
    preflight_gate: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    coupling_gate: Mapping[str, Any],
    coupling_record: Mapping[str, Any],
    failure: Mapping[str, Any],
    pilot_failure: Mapping[str, Any],
    decision: Mapping[str, Any],
    workflow: Mapping[str, Any],
    transitive: Mapping[str, Any],
) -> None:
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA
        and status.get("schema_version") == 1
        and status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and status.get("required_gate") == "pilot"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == PARENT_DECISION
        and status.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and status.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent terminal status changed",
    )
    recomputed_preflight = evaluate_haar_preflight(
        _metrics(preflight_record, "parent preflight")
    )
    _require(
        preflight_gate == recomputed_preflight
        and _one(preflight_gate.get("passed"))
        and isinstance(preflight_gate.get("subchecks"), Mapping)
        and len(preflight_gate["subchecks"]) == 61,
        "parent Haar preflight no longer recomputes to 61/61 pass",
    )
    recomputed_coupling = evaluate_haar_coupling(
        _metrics(coupling_record, "parent coupling")
    )
    _require(
        coupling_gate == recomputed_coupling
        and _one(coupling_gate.get("passed"))
        and _one(coupling_gate.get("numerically_valid"))
        and _one(coupling_gate.get("resource_valid"))
        and isinstance(coupling_gate.get("subchecks"), Mapping)
        and len(coupling_gate["subchecks"]) == 42
        and math.isclose(
            float(coupling_gate["subchecks"]["peak_memory_fraction"]["value"]),
            PARENT_COUPLING_PEAK_MEMORY_FRACTION,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "parent Haar coupling no longer recomputes to 42/42 pass",
    )
    _require(
        failure == pilot_failure
        and failure.get("schema") == PARENT_RUN_SCHEMA + "-stage-failure"
        and failure.get("schema_version") == 1
        and failure.get("evaluation_status") == "execution_failed"
        and failure.get("stage") == "pilot"
        and failure.get("failure_code") == PARENT_FAILURE_CODE
        and failure.get("failure_domain") == "hierarchical_scheduler"
        and failure.get("error_type") == "HaarPowerError"
        and failure.get("error") == PARENT_FAILURE_MESSAGE
        and _zero(failure.get("passed"))
        and _zero(failure.get("stage_execution_valid"))
        and _zero(failure.get("scientific_evidence_complete")),
        "parent pilot is not the exact schedule-binding execution failure",
    )
    expected_workflow = evaluate_haar_workflow(
        provenance=transitive,
        preflight_gate=preflight_gate,
        coupling_gate=coupling_gate,
        pilot_gate=failure,
        require_gate="pilot",
    )
    _require(
        workflow == expected_workflow
        and decision == expected_workflow.get("decision")
        and decision.get("decision") == PARENT_DECISION
        and _zero(decision.get("production_refinement_patch_authorized"))
        and _zero(
            decision.get(
                "one_image_phase_conditioned_training_patch_authorized"
            )
        )
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "parent workflow no longer binds its failed pilot outcome",
    )


def _load_initial_state(
    transitive: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    parent = Path(str(transitive.get("parent_run_dir", ""))).resolve()
    _require(
        parent.is_dir()
        and parent.name == transitive.get("parent_run_basename"),
        "transitive parent run directory changed",
    )
    metadata_path = parent / "source_image.json"
    archive_path = parent / "source_image.npz"
    metadata = _load(metadata_path)
    _require(
        archive_path.is_file()
        and metadata.get("source_npz_sha256") == file_fingerprint(archive_path),
        "transitive parent source-image hash changed",
    )
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == {"image", "mixed_target"},
                "transitive parent source-image payload changed",
            )
            state = np.asarray(archive["mixed_target"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            "cannot load transitive parent source-image payload"
        ) from exc
    _require(
        state.shape == (784,)
        and np.isfinite(state).all()
        and np.all(state >= 0.0)
        and abs(float(state.sum()) - 1.0) <= 1.0e-12,
        "transitive parent mixed target is invalid",
    )
    return np.array(state, copy=True, order="C"), metadata


def _expected_pool_paths(
    directory: Path,
    identities: Sequence[HaarShardIdentity],
) -> set[str]:
    return {
        (directory / f"{identity.fingerprint}{suffix}").as_posix()
        for identity in identities
        for suffix in (".json", ".npz")
    }


def _verify_nested_pool(
    *,
    root: Path,
    pool: str,
    path_ids: tuple[int, ...],
    mixed_state: np.ndarray,
) -> dict[str, Any]:
    schedule = NestedHaarSchedule(pool=pool, role="nested_a")
    identity_list = [
        HaarShardIdentity(
            schedule=schedule,
            path_ids=path_ids,
            coarsest_start_step=first,
            root_seed=PARENT_ROOT_SEED,
            panel_namespace=f"haar-power:a:{pool}",
        )
        for first in range(0, schedule.coarsest_steps, 8)
    ]
    relative_root = (
        Path("haar_power_shards")
        / NESTED_PROFILE_NAME
        / "a"
        / pool
    )
    directory = root / relative_root
    actual = {
        path.relative_to(root).as_posix()
        for path in directory.glob("*")
        if path.is_file()
    }
    expected = _expected_pool_paths(relative_root, identity_list)
    _require(actual == expected, f"parent {pool} shard set changed")

    initial = torch.as_tensor(
        np.repeat(mixed_state.reshape(1, -1), len(path_ids), axis=0),
        dtype=torch.float64,
        device="cpu",
    ).contiguous()
    states = initialize_nested_branch_states(initial, schedule)
    accumulators: dict[int, Any] = {
        int(level): None for level in schedule.levels or ()
    }
    profile = JacobiRBCudaProfile()
    bindings: list[dict[str, Any]] = []
    checkpoints: dict[int, set[int]] = {
        int(level): set() for level in schedule.levels or ()
    }
    execution_rows: list[dict[str, Any]] = []

    for identity in identity_list:
        named_states = {
            f"k{level}": states[int(level)] for level in schedule.levels or ()
        }
        named_accumulators = {
            f"k{level}": accumulators[int(level)]
            for level in schedule.levels or ()
        }
        expected_input = expected_haar_shard_input_sha256(
            identity,
            named_states,
            named_accumulators,
            profile,
        )
        try:
            resumed = load_committed_haar_shard(
                directory,
                expected_identity=identity,
                device="cpu",
            )
        except (HaarSchedulerError, OSError, ValueError) as exc:
            raise ArtifactCompatibilityError(
                f"parent {pool} shard failed exact archive/output verification"
            ) from exc
        metadata = dict(resumed.metadata)
        binding = canonical_shard_schedule_binding(
            metadata,
            expected_profile_name=NESTED_PROFILE_NAME,
            expected_pool=pool,
            expected_role="nested_a",
        )
        _require(
            "schedule" not in metadata,
            "parent shard unexpectedly has a top-level schedule",
        )
        _require(
            canonical_json_bytes(metadata.get("identity"))
            == canonical_json_bytes(identity.to_record())
            and metadata.get("identity_sha256") == identity.fingerprint,
            f"parent {pool} shard identity changed",
        )
        _require(
            metadata.get("input_sha256") == expected_input,
            f"parent {pool} shard input chain changed",
        )
        archive = metadata.get("state_archive")
        _require(
            isinstance(archive, Mapping)
            and archive.get("path") == f"{identity.fingerprint}.npz",
            f"parent {pool} shard archive binding changed",
        )
        branch_records = metadata.get("branches")
        _require(
            isinstance(branch_records, Mapping),
            f"parent {pool} shard branch records are invalid",
        )
        for level in schedule.levels or ():
            name = f"k{level}"
            branch = branch_records.get(name)
            _require(
                isinstance(branch, Mapping),
                f"parent {pool} shard branch {name} is missing",
            )
            ratio = int(level) // schedule.coarsest_steps
            expected_completed = list(
                range(
                    identity.coarsest_start_step * ratio + 8,
                    (identity.coarsest_start_step + 8) * ratio + 1,
                    8,
                )
            )
            _require(
                branch.get("branch") == name
                and branch.get("sample_steps") == int(level)
                and branch.get("completed_steps") == expected_completed
                and branch.get("state_shape") == [len(path_ids), 784]
                and branch.get("raw_observable_shape")
                == [len(expected_completed), len(path_ids), 10]
                and branch.get("dynkin_observable_shape")
                == [len(expected_completed), len(path_ids), 10],
                f"parent {pool} shard branch/checkpoint layout changed",
            )
            raw = resumed.raw_observables[name]
            dynkin = resumed.dynkin_observables[name]
            state = resumed.states[name]
            accumulator = resumed.accumulators[name]
            _require(
                np.isfinite(raw).all()
                and np.isfinite(dynkin).all()
                and bool(torch.isfinite(state).all())
                and bool(torch.isfinite(accumulator.center).all())
                and bool(torch.isfinite(accumulator.compensation).all())
                and bool(torch.isfinite(accumulator.error_radius).all()),
                f"parent {pool} shard contains a nonfinite committed array",
            )
            checkpoints[int(level)].update(expected_completed)
        diagnostics = metadata.get("diagnostics")
        timing = metadata.get("timing")
        _require(
            isinstance(diagnostics, Mapping)
            and isinstance(timing, Mapping)
            and diagnostics.get("profile_name") == NESTED_PROFILE_NAME
            and diagnostics.get("scheduler_version") == HAAR_SCHEDULER_VERSION
            and float(diagnostics.get("certificate_fraction", -1.0)) == 1.0
            and _one(diagnostics.get("state_updates_device_resident_pass"))
            and all(_zero(diagnostics.get(name)) for name in _FORBIDDEN_COUNTS)
            and _one(timing.get("metadata_control_plane_write_excluded")),
            f"parent {pool} shard diagnostics changed",
        )
        execution_rows.append(
            {
                "transition_count": int(diagnostics["transition_count"]),
                "fallback_count": int(diagnostics["fallback_count"]),
                "fallback_elapsed_seconds": float(
                    diagnostics["fallback_elapsed_seconds"]
                ),
                "elapsed_seconds": float(
                    timing["complete_pipeline_including_state_shard_io_seconds"]
                ),
                "mass_error": float(diagnostics["mass_error"]),
                **{
                    name: int(diagnostics[name]) for name in _FORBIDDEN_COUNTS
                },
            }
        )
        bindings.append(
            {
                **binding,
                "path": (
                    relative_root / f"{identity.fingerprint}.json"
                ).as_posix(),
                "identity_sha256": identity.fingerprint,
                "input_sha256": metadata["input_sha256"],
                "output_sha256": metadata["output_sha256"],
                "coarsest_start_step": identity.coarsest_start_step,
            }
        )
        states = {
            int(level): resumed.states[f"k{level}"]
            for level in schedule.levels or ()
        }
        accumulators = {
            int(level): resumed.accumulators[f"k{level}"]
            for level in schedule.levels or ()
        }

    for level in schedule.levels or ():
        _require(
            checkpoints[int(level)] == set(range(8, int(level) + 1, 8)),
            f"parent {pool} K={level} checkpoint chain is incomplete",
        )
    return {
        "pool": pool,
        "shard_count": len(identity_list),
        "path_ids": list(path_ids),
        "coarsest_starts": [
            identity.coarsest_start_step for identity in identity_list
        ],
        "schedule": schedule.to_record(),
        "schedule_bindings": bindings,
        "execution_rows": execution_rows,
        "shard_hash_pass": 1,
        "shard_input_chain_pass": 1,
        "shard_output_chain_pass": 1,
        "checkpoint_chain_pass": 1,
        "path_id_binding_pass": 1,
        "top_level_schedule_absent_pass": 1,
    }


def _nested_plan_ids(
    path_plan: Mapping[str, Any],
    pool: str,
) -> tuple[int, ...]:
    try:
        raw = path_plan["profiles"][NESTED_PROFILE_NAME]["a"]["roles"][pool][
            "root_path_ids"
        ]
    except (KeyError, TypeError) as exc:
        raise ArtifactCompatibilityError(
            f"parent nested panel-A {pool} path IDs are missing"
        ) from exc
    _require(
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes, bytearray))
        and len(raw) == 8
        and all(_integer(value) for value in raw),
        f"parent nested panel-A {pool} path IDs are invalid",
    )
    result = tuple(int(value) for value in raw)
    _require(
        len(set(result)) == len(result),
        f"parent nested panel-A {pool} path IDs contain duplicates",
    )
    return result


def _verify_power_shards(
    root: Path,
    path_plan: Mapping[str, Any],
    transitive: Mapping[str, Any],
) -> dict[str, Any]:
    mixed_state, source_metadata = _load_initial_state(transitive)
    main_ids = _nested_plan_ids(path_plan, "main")
    reference_ids = _nested_plan_ids(path_plan, "reference")
    _require(
        set(main_ids).isdisjoint(reference_ids),
        "parent nested main/reference path IDs overlap",
    )
    main = _verify_nested_pool(
        root=root,
        pool="main",
        path_ids=main_ids,
        mixed_state=mixed_state,
    )
    reference = _verify_nested_pool(
        root=root,
        pool="reference",
        path_ids=reference_ids,
        mixed_state=mixed_state,
    )
    all_rows = main.pop("execution_rows") + reference.pop("execution_rows")
    transition_count = sum(row["transition_count"] for row in all_rows)
    fallback_count = sum(row["fallback_count"] for row in all_rows)
    fallback_seconds = sum(row["fallback_elapsed_seconds"] for row in all_rows)
    elapsed_seconds = sum(row["elapsed_seconds"] for row in all_rows)
    maximum_mass_error = max(row["mass_error"] for row in all_rows)
    forbidden = {
        name: sum(row[name] for row in all_rows) for name in _FORBIDDEN_COUNTS
    }
    rates = []
    for record in (main, reference):
        rows = (
            all_rows[: PARENT_NESTED_MAIN_SHARD_COUNT]
            if record["pool"] == "main"
            else all_rows[PARENT_NESTED_MAIN_SHARD_COUNT :]
        )
        rates.append(
            sum(row["transition_count"] for row in rows)
            / sum(row["elapsed_seconds"] for row in rows)
        )
    conservative_rate = min(rates)
    _require(
        main["shard_count"] == PARENT_NESTED_MAIN_SHARD_COUNT
        and reference["shard_count"] == PARENT_NESTED_REFERENCE_SHARD_COUNT
        and transition_count == PARENT_TRANSITION_COUNT
        and fallback_count == PARENT_FALLBACK_COUNT
        and maximum_mass_error == PARENT_MAXIMUM_MASS_ERROR
        and all(value == 0 for value in forbidden.values())
        and math.isclose(
            conservative_rate,
            PARENT_CONSERVATIVE_RATE,
            rel_tol=1.0e-14,
            abs_tol=1.0e-12,
        ),
        "parent nested panel-A execution totals changed",
    )

    power_root = root / "haar_power_shards"
    power_files = {
        path.relative_to(root).as_posix()
        for path in power_root.rglob("*")
        if path.is_file()
    }
    _require(
        not any(
            path.startswith(
                f"haar_power_shards/{ANTITHETIC_PROFILE_NAME}/"
            )
            for path in power_files
        ),
        "parent unexpectedly contains antithetic power shards",
    )
    _require(
        not any("/b/" in path for path in power_files),
        "parent unexpectedly contains panel-B power shards",
    )
    return {
        "profile": NESTED_PROFILE_NAME,
        "panel": "a",
        "main": main,
        "reference": reference,
        "total_shard_count": main["shard_count"] + reference["shard_count"],
        "source_npz_sha256": source_metadata["source_npz_sha256"],
        "execution": {
            "transition_count": transition_count,
            "certified_count": transition_count,
            "certificate_fraction": 1.0,
            "fallback_count": fallback_count,
            "fallback_fraction": fallback_count / transition_count,
            "fallback_elapsed_seconds": fallback_seconds,
            "fallback_cost_fraction": fallback_seconds / elapsed_seconds,
            "elapsed_seconds": elapsed_seconds,
            "conservative_rate": conservative_rate,
            "mass_error": maximum_mass_error,
            **forbidden,
        },
        "schedule_binding_pass": 1,
        "top_level_schedule_absent_pass": 1,
        "shard_hash_pass": 1,
        "shard_chain_pass": 1,
        "checkpoint_chain_pass": 1,
        "path_id_binding_pass": 1,
        "antithetic_power_shards_absent_pass": 1,
        "panel_b_absent_pass": 1,
    }


def _verify_absent_work(records: set[str]) -> None:
    forbidden_files = {
        "sealed_profile_selection.json",
        "selected_haar_design.json",
        "haar_pilot_metrics.json",
        f"{NESTED_PROFILE_NAME}_panel_b.json",
        f"{NESTED_PROFILE_NAME}_panel_b_evidence.json",
        f"{ANTITHETIC_PROFILE_NAME}_panel_a.json",
        f"{ANTITHETIC_PROFILE_NAME}_panel_a_evidence.json",
        f"{ANTITHETIC_PROFILE_NAME}_panel_b.json",
        f"{ANTITHETIC_PROFILE_NAME}_panel_b_evidence.json",
    }
    _require(
        records.isdisjoint(forbidden_files)
        and not any(
            path.startswith("haar_power_shards/") and "/b/" in path
            for path in records
        )
        and not any(
            path.startswith(
                f"haar_power_shards/{ANTITHETIC_PROFILE_NAME}/"
            )
            for path in records
        )
        and not any("refinement_observables" in path for path in records)
        and not any("production_refinement" in path for path in records),
        "parent contains antithetic, panel-B, selection, or refinement work",
    )


def verify_haar_power_recovery_parent(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Verify the exact immutable failed Haar parent and its 80 shard chains."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"Haar recovery parent does not exist: {root}")
    _require(
        root.name == PARENT_RUN_BASENAME,
        f"parent basename must be {PARENT_RUN_BASENAME}",
    )
    paths = {name: root / relative for name, relative in _REQUIRED_FILES.items()}
    for name, path in paths.items():
        _require(path.is_file(), f"Haar recovery parent lacks {name}: {path.name}")

    status = _load(paths["status"])
    registry = _verify_registry(root, paths["registry"], status)
    records = set(dict(registry["records"]))
    _verify_absent_work(records)
    manifest = _load(paths["manifest"])
    scientific_config = _load(paths["scientific_config"])
    sources = _verify_sources_and_config(manifest, scientific_config)
    path_plan = _load(paths["path_id_plan"])
    profile_plan = _load(paths["profile_plan"])
    sealed = _load(paths["sealed_panels"])
    model_contract = _load(paths["model_input_contract"])
    _verify_frozen_plans(path_plan, profile_plan, sealed, model_contract)

    preflight_gate = _load(paths["preflight_gate"])
    preflight_record = _load(paths["preflight_metrics"])
    coupling_gate = _load(paths["coupling_gate"])
    coupling_record = _load(paths["coupling_metrics"])
    failure = _load(paths["pilot_gate"])
    pilot_failure = _load(paths["pilot_failure"])
    decision = _load(paths["decision"])
    workflow = _load(paths["workflow"])
    transitive = _load(paths["transitive_provenance"])
    _verify_local_outcome(
        status=status,
        preflight_gate=preflight_gate,
        preflight_record=preflight_record,
        coupling_gate=coupling_gate,
        coupling_record=coupling_record,
        failure=failure,
        pilot_failure=pilot_failure,
        decision=decision,
        workflow=workflow,
        transitive=transitive,
    )
    recomputed_transitive = verify_right_endpoint_coupling_parent(
        transitive.get("parent_run_dir", "")
    )
    _require(
        transitive == recomputed_transitive,
        "parent transitive provenance no longer recomputes",
    )

    for relative in sorted(records):
        if relative.endswith(".json"):
            _no_work_tree(_load(root / relative), f"parent artifact {relative}")
    _no_work_tree(status, "parent status")
    _no_work_tree(registry, "parent registry")

    shard_audit = _verify_power_shards(root, path_plan, transitive)
    bindings = (
        list(shard_audit["main"]["schedule_bindings"])
        + list(shard_audit["reference"]["schedule_bindings"])
    )
    _require(
        len(bindings) == PARENT_NESTED_SHARD_COUNT
        and all(
            binding.get("binding_source") == "identity.schedule"
            and _zero(binding.get("top_level_schedule_present"))
            for binding in bindings
        ),
        "parent canonical schedule-binding table is incomplete",
    )
    execution = dict(shard_audit["execution"])
    return {
        "schema": "d0-jacobi-rb-haar-power-recovery-parent-provenance",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": paths["registry"].stat().st_size,
        "parent_artifact_record_count": PARENT_REGISTRY_RECORD_COUNT,
        "parent_registry_hash_pass": 1,
        "parent_all_artifact_hashes_pass": 1,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_source_count": len(sources),
        "parent_sources_immutable_pass": 1,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent_scientific_config_pass": 1,
        "parent_root_seed": PARENT_ROOT_SEED,
        "parent_path_id_plan_sha256": PARENT_PATH_ID_PLAN_SHA256,
        "parent_profile_plan_sha256": PARENT_PROFILE_PLAN_SHA256,
        "parent_model_input_contract_sha256": (
            PARENT_MODEL_INPUT_CONTRACT_SHA256
        ),
        "parent_path_id_plan_pass": 1,
        "parent_frozen_plans_pass": 1,
        "parent_preflight_pass": 1,
        "parent_preflight_subcheck_count": 61,
        "parent_coupling_pass": 1,
        "parent_coupling_subcheck_count": 42,
        "parent_pilot_execution_failed_pass": 1,
        "parent_decision": PARENT_DECISION,
        "parent_failure_code": PARENT_FAILURE_CODE,
        "parent_failure_message": PARENT_FAILURE_MESSAGE,
        "parent_re_adjudication": PARENT_RE_ADJUDICATION,
        "parent_transitive_provenance_pass": 1,
        "inherited_scientific_parent_run_dir": transitive["parent_run_dir"],
        "parent_nested_main_shard_count": PARENT_NESTED_MAIN_SHARD_COUNT,
        "parent_nested_reference_shard_count": (
            PARENT_NESTED_REFERENCE_SHARD_COUNT
        ),
        "parent_nested_shard_count": PARENT_NESTED_SHARD_COUNT,
        "parent_schedule_binding_pass": 1,
        "parent_identity_schedule_pass": 1,
        "parent_top_level_schedule_absent_pass": 1,
        "parent_shard_hash_pass": 1,
        "parent_shard_chain_pass": 1,
        "parent_checkpoint_chain_pass": 1,
        "parent_no_antithetic_power_shards_pass": 1,
        "parent_panel_b_absent_pass": 1,
        "parent_selection_absent_pass": 1,
        "parent_no_work_pass": 1,
        "parent_transition_count": execution["transition_count"],
        "parent_certificate_fraction": execution["certificate_fraction"],
        "parent_fallback_count": execution["fallback_count"],
        "parent_fallback_fraction": execution["fallback_fraction"],
        "parent_fallback_cost_fraction": execution["fallback_cost_fraction"],
        "parent_mass_error": execution["mass_error"],
        "parent_conservative_rate": execution["conservative_rate"],
        "parent_coupling_peak_memory_fraction": (
            PARENT_COUPLING_PEAK_MEMORY_FRACTION
        ),
        "parent_shard_audit": shard_audit,
        "canonical_schedule_bindings": bindings,
        "parent_mutated": 0,
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "physical_training_performed": 0,
        "production_refinement_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


__all__ = [
    "PARENT_CONSERVATIVE_RATE",
    "PARENT_COUPLING_PEAK_MEMORY_FRACTION",
    "PARENT_DECISION",
    "PARENT_FAILURE_CODE",
    "PARENT_FAILURE_MESSAGE",
    "PARENT_FALLBACK_COUNT",
    "PARENT_MAXIMUM_MASS_ERROR",
    "PARENT_MODEL_INPUT_CONTRACT_SHA256",
    "PARENT_NESTED_MAIN_SHARD_COUNT",
    "PARENT_NESTED_REFERENCE_SHARD_COUNT",
    "PARENT_NESTED_SHARD_COUNT",
    "PARENT_PATH_ID_PLAN_SHA256",
    "PARENT_PROFILE_PLAN_SHA256",
    "PARENT_RE_ADJUDICATION",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_ROOT_SEED",
    "PARENT_RUN_BASENAME",
    "PARENT_RUN_SCHEMA",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_COUNT",
    "PARENT_SOURCE_FINGERPRINT",
    "PARENT_TRANSITION_COUNT",
    "SCHEDULE_BINDING_VERSION",
    "canonical_shard_schedule",
    "canonical_shard_schedule_binding",
    "verify_haar_power_recovery_parent",
]
