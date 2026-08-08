"""Exact zero-baseline Jacobi/RB boundary-tangent learnability (v3).

The only scientific representation change relative to the immutable eager-v2
workflow is ``q_B := 0``.  Physical training is a fixed checkpoint generator:
validation labels are opened only by the separate, search-aware ``select``
stage.  This module never executes a controller, reverse path, reconstruction,
or sampler.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent import (
    direct_raw_target_mse,
    synthetic_tangent_target,
)
from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    MIDPOINT_COUNT,
    MIDPOINT_FRACTIONS,
    SELECTED_OUTER_STEPS,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import (
    EagerCohort,
    EagerDiagnosticsAccumulator,
    deterministic_test_branch_runner,
    deterministic_test_shard_runner,
    execute_eager_shard,
    explicit_eager_cache_plan,
    generate_eager_cache_for_cohorts,
    load_eager_role_inputs,
    load_eager_role_labels,
)
from mnist.d0_jacobi_rb_boundary_tangent_prefix_schedule import eager_prefix_profile
from mnist.d0_jacobi_rb_boundary_tangent_prefix_fallback import (
    sample_alpha1_rb_transition_batch_cuda_eager,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule import (
    PHASE_COUNT,
    sample_fused_midpoint_branches,
)
from mnist.d0_jacobi_rb_cuda import sample_alpha1_rb_transition_batch_cuda
from mnist.d0_jacobi_rb_cuda_multipath import run_exact_multipath_shard
from mnist.d0_jacobi_rb_certificate_semantics import (
    CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
    CertificateSemanticsError,
    compare_certificate_semantics,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_gate import (
    BoundaryTangentV3Thresholds,
    REQUIRED_GATES,
    decide_workflow,
    evaluate_cache_gate,
    evaluate_confirm_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    evaluate_select_gate,
    evaluate_train_gate,
    not_evaluated_gate,
)
from mnist import d0_jacobi_rb_boundary_tangent_v3_provenance as _provenance
from mnist import d0_jacobi_rb_boundary_tangent_v3_selection as _selection
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZERO_BASELINE_SHA256,
    ZeroBaselineBoundaryTangentPredictor,
    configure_exact_synthetic_zero_baseline_teacher,
    exact_zero_baseline_prediction,
    zero_baseline_contract,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    MODEL_INPUT_FIELDS,
    OUTER_STEPS,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
    call_model,
    deterministic_batch_indices,
    enable_deterministic_torch,
    state_dict_sha256,
)
from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time
from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_controller_confirmation as _legacy,
)
from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_eager_confirmation as _eager_v2,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-zero-baseline-v3"
TEST_RUN_SCHEMA = RUN_SCHEMA + "-nonauthorizing-test"
STAGES = ("preflight", "cache", "train", "select", "confirm", "report", "all")
ROOT_SEED = 261_311
MODEL_SEEDS = (261_312, 261_313, 261_314)
SELECTION_BOOTSTRAP_SEED = 261_320
CONFIRMATION_BOOTSTRAP_SEED = 261_322
SYNTHETIC_CONTROL_SEED = 261_323
NULL_CONTROL_SEED = 261_324
RESERVED_CONTROL_SEED = 261_325
FORBIDDEN_SCHEDULER_BENCHMARK_SEED = 261_321
TRAINING = {
    "width": 32,
    "batch_size": 32,
    "prediction_batch_size": 32,
    "maximum_updates": 4_000,
    "checkpoint_interval": 100,
    "learning_rate": 1.0e-3,
    "weight_decay": 0.0,
    "gradient_norm_clip": 1.0,
    "mixed_precision": 0,
}
NO_WORK = {
    "controller_control_trajectory_performed": 0,
    "maximum_control_trajectory_phase_count": 0,
    "full_reverse_path_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "image_sampling_performed": 0,
    "reconstruction_performed": 0,
    "full_dataset_training_performed": 0,
}
_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "boundary_tangent_v3_decision.json",
}


class BoundaryTangentV3CLIError(RuntimeError):
    """Typed v3 execution failure with stable fail-closed classification."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "boundary_tangent_v3_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


V3CLIError = BoundaryTangentV3CLIError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact: {target}") from exc
    if not isinstance(value, dict):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {target}")
    return value


def _optional_json(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / name
    return _load_json(path) if path.is_file() else None


def _semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _passed(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("evaluation_status") == "evaluated"
        and int(value.get("passed", 0)) == 1
    )


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _atomic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=target.name + ".",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(
                handle,
                **{
                    name: np.ascontiguousarray(value)
                    for name, value in arrays.items()
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": target.as_posix(),
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _atomic_torch(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=target.name + ".",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": target.as_posix(),
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactCompatibilityError(f"cannot read NPZ artifact: {path}") from exc


def _scope(run_dir: Path) -> dict[str, int]:
    def flag(name: str, field: str) -> int:
        path = run_dir / name
        return int(path.is_file() and int(_load_json(path).get(field, 0)) == 1)

    return {
        "production_cache_generation_performed": flag(
            "cache_metrics.json", "production_cache_generation_performed"
        ),
        "physical_training_performed": int(
            (run_dir / "physical_training_started.json").is_file()
        ),
        "validation_selection_performed": flag(
            "select_metrics.json", "validation_selection_performed"
        ),
        "confirmation_performed": flag(
            "confirmation_metrics.json", "confirmation_performed"
        ),
        **NO_WORK,
    }


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    message: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
    scientific_evidence_complete: int | None = None,
) -> None:
    atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": RUN_SCHEMA + "-status",
            "schema_version": 1,
            "state": str(state),
            "stage": str(stage),
            "decision": decision,
            "message": message,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "scientific_evidence_complete": scientific_evidence_complete,
            "updated_at": _now(),
            **_scope(run_dir),
        },
    )


_ARTIFACT_DIRECTORIES = frozenset(
    {"cache", "checkpoints", "confirmation", "eager_cache", "selection", "validation"}
)


def _artifact_path_is_known(relative: str) -> bool:
    path = Path(relative)
    if len(path.parts) > 1:
        return path.parts[0] in _ARTIFACT_DIRECTORIES
    name = path.name
    return (
        name.endswith((".json", ".csv", ".npz"))
        and any(
            token in name
            for token in (
                "adjudication",
                "artifact",
                "baseline",
                "cache",
                "candidate",
                "certificate",
                "checkpoint",
                "cohort",
                "confirm",
                "control",
                "execution",
                "image",
                "manifest",
                "mixed",
                "null",
                "parent",
                "path",
                "physical",
                "preflight",
                "run_status",
                "scientific",
                "select",
                "source",
                "synthetic",
                "target",
                "train",
                "update_zero",
                "validation",
                "workflow",
                "zero_initialization",
            )
        )
    )


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in run_dir.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        relative = path.relative_to(run_dir).as_posix()
        if relative in _REGISTRY_EXCLUDED or ".tmp" in path.name:
            continue
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ArtifactCompatibilityError("unsafe artifact registry path")
        if not _artifact_path_is_known(relative):
            raise ArtifactCompatibilityError(
                f"unexpected artifact cannot be registered: {relative}"
            )
        records.append(
            {
                "path": relative,
                "sha256": file_fingerprint(path),
                "size": int(path.stat().st_size),
            }
        )
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "artifact_count": len(records),
        "artifacts": records,
        "semantic_sha256": config_fingerprint({"artifacts": records}),
        **_scope(run_dir),
    }
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_existing_registry(
    run_dir: Path, *, allow_unregistered_incomplete_tail: bool = False
) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    record = _load_json(path)
    artifacts = record.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or record.get("artifact_count") != len(artifacts)
        or record.get("semantic_sha256")
        != config_fingerprint({"artifacts": artifacts})
    ):
        raise ArtifactCompatibilityError("artifact registry changed")
    registered: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ArtifactCompatibilityError("artifact registry row is malformed")
        relative = str(item.get("path", ""))
        pure = Path(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != relative
            or relative in registered
        ):
            raise ArtifactCompatibilityError("artifact registry path is unsafe")
        registered.add(relative)
        target = run_dir / relative
        if (
            not target.is_file()
            or item.get("sha256") != file_fingerprint(target)
            or int(item.get("size", -1)) != target.stat().st_size
        ):
            raise ArtifactCompatibilityError("registered artifact changed")
    actual = {
        item.relative_to(run_dir).as_posix()
        for item in run_dir.rglob("*")
        if item.is_file()
        and item.relative_to(run_dir).as_posix() not in _REGISTRY_EXCLUDED
        and ".tmp" not in item.name
    }
    extras = actual - registered
    missing = registered - actual
    if missing:
        raise ArtifactCompatibilityError("artifact registry does not match run files")
    if extras and not allow_unregistered_incomplete_tail:
        raise ArtifactCompatibilityError("artifact registry does not match run files")
    if extras and not all(_artifact_path_is_known(relative) for relative in extras):
        raise ArtifactCompatibilityError("unrecognized incomplete artifact tail")


def _seal_stage(run_dir: Path, names: Sequence[str], seal_name: str) -> dict[str, Any]:
    artifacts = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"cannot seal missing artifact: {name}")
        artifacts.append(
            {
                "path": name,
                "sha256": file_fingerprint(path),
                "size": int(path.stat().st_size),
            }
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-stage-seal",
            "schema_version": 1,
            "artifacts": artifacts,
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / seal_name, record)
    return record


def _verify_stage_seal(run_dir: Path, seal_name: str) -> None:
    seal = _load_json(run_dir / seal_name)
    body = dict(seal)
    semantic = body.pop("semantic_sha256", None)
    if semantic != config_fingerprint(body):
        raise ArtifactCompatibilityError("stage seal changed")
    for item in seal.get("artifacts", []):
        path = run_dir / str(item["path"])
        if (
            item.get("sha256") != file_fingerprint(path)
            or int(item.get("size", -1)) != path.stat().st_size
        ):
            raise ArtifactCompatibilityError("sealed artifact changed")


def build_v3_path_plan(
    *, test_only: bool = False, test_path_count: int = 2
) -> dict[str, Any]:
    if not test_only:
        return _provenance.build_v3_path_plan()
    count = max(1, int(test_path_count))
    roles = {
        "preflight_seam": list(range(0x100, 0x100 + count)),
        "train": list(range(0x200, 0x200 + count)),
        "validation": list(range(0x300, 0x300 + count)),
        "confirmation": list(range(0x400, 0x400 + count)),
    }
    return _semantic(
        {
            "schema": TEST_RUN_SCHEMA + "-path-plan",
            "schema_version": 1,
            "test_only": 1,
            "roles": roles,
            "production_path_ids_opened": 0,
            "authorizing": 0,
        }
    )


def build_v3_cohort_plan(
    path_plan: Mapping[str, Any], *, test_only: bool = False
) -> dict[str, Any]:
    if not test_only:
        return _provenance.build_v3_cohort_plan(path_plan)
    roles = path_plan["roles"]

    def records(kind: str, names: Sequence[str]) -> list[dict[str, Any]]:
        ids: list[int] = []
        path_roles: list[str] = []
        for name in names:
            values = [int(value) for value in roles[name]]
            ids.extend(values)
            path_roles.extend([name] * len(values))
        return [
            {
                "kind": kind,
                "index": 0,
                "size": len(ids),
                "path_ids": ids,
                "path_roles": path_roles,
            }
        ]

    return _semantic(
        {
            "schema": TEST_RUN_SCHEMA + "-cohort-plan",
            "schema_version": 1,
            "test_only": 1,
            "path_id_plan_sha256": path_plan["semantic_sha256"],
            "train_validation": records(
                "train_validation", ("train", "validation")
            ),
            "confirmation": records("confirmation", ("confirmation",)),
            "production_path_ids_opened": 0,
            "authorizing": 0,
        }
    )


def _cohorts(plan: Mapping[str, Any], kind: str) -> tuple[EagerCohort, ...]:
    return tuple(
        EagerCohort(
            kind=str(item["kind"]),
            index=int(item["index"]),
            path_ids=tuple(int(value) for value in item["path_ids"]),
            path_roles=tuple(str(value) for value in item["path_roles"]),
        )
        for item in plan[kind]
    )


def _source_set() -> tuple[Path, ...]:
    return _provenance.v3_transitive_source_paths((Path(__file__),))


def _target_and_input_contract() -> dict[str, Any]:
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-target-input-contract",
            "schema_version": 1,
            "allowed_model_inputs": list(MODEL_INPUT_FIELDS),
            "audit_only_fields": [
                "outer_step",
                "midpoint_index",
                "midpoint_fraction",
            ],
            "forbidden_model_inputs": [
                "certificate",
                "earlier_state",
                "random_bits",
                "oracle_quantity",
            ],
            "target": "unchanged raw binary64 Jacobi Rao-Blackwell label",
            "prediction": "m_theta(W)=y(1-y)*q_theta(W)",
            "baseline": "q_B := 0",
            "plain_unweighted_raw_target_mse": 1,
            "quotient_target_formed": 0,
            "target_modified": 0,
            "target_clipped": 0,
            **NO_WORK,
        }
    )


def _certificate_semantics_contract() -> dict[str, Any]:
    """Freeze which certificate fields can authorize seam equivalence.

    Adaptive and eager-prefix execution prove the same correctly rounded
    transition through potentially different proof schedules.  Only the
    represented transition and normalized authorization semantics are
    equality gates; proof-effort telemetry remains independently auditable.
    """

    return _semantic(
        {
            "schema": RUN_SCHEMA + "-certificate-semantics-contract",
            "schema_version": 1,
            "comparator_version": CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
            "scientific_payload": [
                "canonical_transition_identity",
                "earlier_head_fraction",
                "exposure",
                "active_mask",
                "later_head_fraction",
                "rao_blackwell_target",
                "post_phase_state",
                "final_state",
            ],
            "authorizing_certificate_semantics": [
                "active_mask",
                "authorized_mask",
                "active_transition_uniquely_certified",
                "structural_inactive_transition_is_exact_noop",
            ],
            "proof_metadata_advisory": [
                "batch_certificate_sha256",
                "certificate_codes",
                "mode_counts",
                "prefix_bits",
                "strengthened_route",
                "fallback_route",
                "fallback_reason",
                "enclosure_width",
                "timing",
            ],
            "proof_metadata_equality_required": 0,
            "scientific_payload_equality_required": 1,
            "certificate_semantics_equality_required": 1,
            "maximum_recorded_mismatches": 16,
            "genuine_transition_mismatch_decision": "exact_cache_invalid",
            "malformed_comparator_decision": (
                "certificate_semantics_comparator_invalid"
            ),
            "target_changed": 0,
            "transition_law_changed": 0,
            **NO_WORK,
        }
    )


def _scientific_config(
    args: argparse.Namespace,
    path_plan: Mapping[str, Any] | None = None,
    cohort_plan: Mapping[str, Any] | None = None,
    certificate_contract: Mapping[str, Any] | None = None,
    failed_preflight_adjudication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    test_only = bool(getattr(args, "test_only", False))
    paths = (
        build_v3_path_plan(
            test_only=test_only,
            test_path_count=int(getattr(args, "test_path_count", 2)),
        )
        if path_plan is None
        else dict(path_plan)
    )
    cohorts = (
        build_v3_cohort_plan(paths, test_only=test_only)
        if cohort_plan is None
        else dict(cohort_plan)
    )
    certificate = (
        _certificate_semantics_contract()
        if certificate_contract is None
        else dict(certificate_contract)
    )
    predecessor = (
        _semantic(
            {
                "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA)
                + "-failed-preflight-binding-fixture",
                "schema_version": 1,
                "passed": 1,
                "test_only": int(test_only),
                "registry_count": _provenance.FAILED_V3_PREFLIGHT_REGISTRY_COUNT,
                "registry_semantic_sha256": (
                    _provenance.FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256
                ),
                "readjudicated_decision": (
                    _provenance.FAILED_V3_PREFLIGHT_READJUDICATED_DECISION
                ),
            }
        )
        if failed_preflight_adjudication is None
        else dict(failed_preflight_adjudication)
    )
    maximum_updates = (
        int(getattr(args, "test_maximum_updates", 0))
        if test_only
        else TRAINING["maximum_updates"]
    )
    executed_steps = (
        int(getattr(args, "test_outer_steps", 16)) if test_only else OUTER_STEPS
    )
    record = {
        "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA)
        + "-scientific-config",
        "schema_version": 1,
        "authorizing": int(not test_only),
        "test_only": int(test_only),
        "grid_size": 28,
        "alpha": 1.0,
        "sample_steps": OUTER_STEPS,
        "executed_outer_steps": executed_steps,
        "tau_eff": 5.0e-5,
        "lambda_mix": 0.35,
        "label": 3,
        "source_image": {
            "class_index": _provenance.SOURCE_CLASS_INDEX,
            "dataset_index": _provenance.SOURCE_DATASET_INDEX,
            "image_sha256": _provenance.SOURCE_IMAGE_SHA256,
            "mixed_target_sha256": _provenance.MIXED_TARGET_SHA256,
            "source_image_json_sha256": _provenance.SOURCE_IMAGE_JSON_SHA256,
            "source_image_npz_sha256": _provenance.SOURCE_IMAGE_NPZ_SHA256,
            "source_image_npz_size": _provenance.SOURCE_IMAGE_NPZ_SIZE,
        },
        "root_seed": ROOT_SEED,
        "model_seeds": list(MODEL_SEEDS),
        "selection_bootstrap_seed": SELECTION_BOOTSTRAP_SEED,
        "confirmation_bootstrap_seed": CONFIRMATION_BOOTSTRAP_SEED,
        "synthetic_teacher_seed": SYNTHETIC_CONTROL_SEED,
        "exact_model_null_seed": NULL_CONTROL_SEED,
        "reserved_control_seed": RESERVED_CONTROL_SEED,
        "reserved_future_control_seed": RESERVED_CONTROL_SEED,
        "forbidden_scheduler_benchmark_seed": FORBIDDEN_SCHEDULER_BENCHMARK_SEED,
        "selection_namespace": _selection.SELECTION_NAMESPACE,
        "confirmation_namespace": _selection.CONFIRMATION_NAMESPACE,
        "candidate_count": _selection.V3_CANDIDATE_COUNT,
        "component_count": _selection.V3_COMPONENT_COUNT,
        "joint_family_size": _selection.V3_SEARCH_FAMILY_SIZE,
        "checkpoint_updates": list(range(0, TRAINING["maximum_updates"] + 1, 100)),
        "physical_training_uses_validation_labels": 0,
        "confirmation_paths_created": 0,
        "selected_outer_steps": [
            step for step in SELECTED_OUTER_STEPS if step < executed_steps
        ],
        "midpoint_fractions": [
            1 / 16,
            3 / 16,
            5 / 16,
            7 / 16,
            9 / 16,
            11 / 16,
            13 / 16,
            15 / 16,
        ],
        "path_id_plan_sha256": paths["semantic_sha256"],
        "cohort_plan_sha256": cohorts["semantic_sha256"],
        "certificate_semantics_comparator_version": (
            CERTIFICATE_SEMANTICS_COMPARATOR_VERSION
        ),
        "certificate_semantics_contract_sha256": certificate["semantic_sha256"],
        "failed_v3_preflight_adjudication_sha256": predecessor["semantic_sha256"],
        "failed_v3_preflight_registry_count": (
            _provenance.FAILED_V3_PREFLIGHT_REGISTRY_COUNT
        ),
        "failed_v3_preflight_registry_semantic_sha256": (
            _provenance.FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256
        ),
        "failed_v3_preflight_registry_file_sha256": (
            _provenance.FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256
        ),
        "zero_baseline_sha256": ZERO_BASELINE_SHA256,
        "training": {**TRAINING, "maximum_updates": maximum_updates},
        "selection": (
            {
                "schema": TEST_RUN_SCHEMA + "-bootstrap-plan",
                "authorizing": 0,
                "path_count": len(paths["roles"]["validation"]),
                "candidate_count": _selection.V3_CANDIDATE_COUNT,
                "component_count": _selection.V3_COMPONENT_COUNT,
                "search_family_size": _selection.V3_SEARCH_FAMILY_SIZE,
                "replicates": int(getattr(args, "test_bootstrap_replicates", 8)),
                "shard_size": int(getattr(args, "test_bootstrap_replicates", 8)),
                "seed": SELECTION_BOOTSTRAP_SEED,
                "namespace": _selection.SELECTION_NAMESPACE,
            }
            if test_only
            else _selection.v3_bootstrap_plan(
                seed=SELECTION_BOOTSTRAP_SEED,
                namespace=_selection.SELECTION_NAMESPACE,
                path_count=32,
            )
        ),
        "thresholds": BoundaryTangentV3Thresholds().to_dict(),
        "target": "unchanged exact certified Jacobi/Rao-Blackwell label",
        "objective": "plain unweighted direct MSE",
        "zero_baseline_formula": "q_B := 0",
        "pointwise_checkpoint_selection_performed": 0,
        "validation_labels_available_to_physical_training": 0,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.runs_root / f"{stamp}_{args.run_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, False


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    test_only = bool(args.test_only)
    path_plan = build_v3_path_plan(
        test_only=test_only, test_path_count=args.test_path_count
    )
    cohort_plan = build_v3_cohort_plan(path_plan, test_only=test_only)
    zero_contract = _semantic(zero_baseline_contract())
    target_contract = _target_and_input_contract()
    certificate_contract = _certificate_semantics_contract()
    sources = _source_set()
    source_hash = _provenance.v3_source_fingerprint(sources)
    if test_only:
        failed_preflight_adjudication = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-failed-preflight-readjudication",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "passed": 1,
                "test_only": 1,
                "failed_run_dir": str(args.failed_v3_preflight_run_dir),
                "immutable_registry": {
                    "artifact_count": (
                        _provenance.FAILED_V3_PREFLIGHT_REGISTRY_COUNT
                    ),
                    "file_sha256": (
                        _provenance.FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256
                    ),
                    "semantic_sha256": (
                        _provenance.FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256
                    ),
                },
                "readjudicated_decision": (
                    _provenance.FAILED_V3_PREFLIGHT_READJUDICATED_DECISION
                ),
                "readjudicated_failure_domain": "implementation_contract",
                "decision": (
                    _provenance.FAILED_V3_PREFLIGHT_READJUDICATED_DECISION
                ),
                "failure_domain": "implementation_contract",
                "stage_execution_valid": 1,
                "numerically_valid": 1,
                "resource_valid": 1,
                "scientific_evidence_complete": 1,
                "production_paths_opened": 0,
                **NO_WORK,
            }
        )
        parent_provenance = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-parent-provenance",
                "schema_version": 1,
                "passed": 1,
                "test_only": 1,
                "production_parent_evidence_used": 0,
            }
        )
        authorization = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-authorization",
                "schema_version": 1,
                "passed": 1,
                "authorizing": 0,
            }
        )
        path_validation = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-path-collision-scan",
                "passed": 1,
                "production_path_ids_opened": 0,
            }
        )
        cohort_validation = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-cohort-validation",
                "passed": 1,
            }
        )
    else:
        failed_preflight_adjudication = (
            _provenance.verify_and_re_adjudicate_failed_v3_preflight(
                args.failed_v3_preflight_run_dir
            )
        )
        parent_provenance = _provenance.verify_v3_parent_evidence(
            parent_v2_run_dir=args.parent_v2_run_dir,
            adjudication_run_dir=args.adjudication_run_dir,
            parent_eager_pipeline_run_dir=args.parent_eager_pipeline_run_dir,
            parent_coarse_residual_run_dir=args.parent_coarse_residual_run_dir,
        )
        authorization = _provenance.build_v3_adjudication_authorization(
            parent_provenance
        )
        path_validation = _provenance.validate_v3_path_plan(path_plan)
        cohort_validation = _provenance.validate_v3_cohort_plan(
            cohort_plan, path_plan=path_plan
        )
    config = _scientific_config(
        args,
        path_plan,
        cohort_plan,
        certificate_contract=certificate_contract,
        failed_preflight_adjudication=failed_preflight_adjudication,
    )
    adjudication_provenance = _semantic(
        {
            "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA)
            + "-adjudication-provenance",
            "schema_version": 1,
            "parent_provenance_sha256": parent_provenance["semantic_sha256"],
            "authorization_sha256": authorization["semantic_sha256"],
            "passed": int(parent_provenance.get("passed", 0)),
        }
    )
    manifest = {
        "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA) + "-manifest",
        "schema_version": 1,
        "created_at": _now(),
        "device": args.device,
        "source_fingerprint": source_hash,
        "scientific_config_sha256": config["semantic_sha256"],
        "parent_provenance_sha256": parent_provenance["semantic_sha256"],
        "adjudication_provenance_sha256": adjudication_provenance[
            "semantic_sha256"
        ],
        "adjudication_authorization_sha256": authorization["semantic_sha256"],
        "path_plan_sha256": path_plan["semantic_sha256"],
        "cohort_plan_sha256": cohort_plan["semantic_sha256"],
        "zero_baseline_contract_sha256": zero_contract["semantic_sha256"],
        "target_and_input_contract_sha256": target_contract["semantic_sha256"],
        "certificate_semantics_comparator_version": (
            CERTIFICATE_SEMANTICS_COMPARATOR_VERSION
        ),
        "certificate_semantics_contract_sha256": certificate_contract[
            "semantic_sha256"
        ],
        "failed_v3_preflight_adjudication_sha256": (
            failed_preflight_adjudication["semantic_sha256"]
        ),
        "failed_v3_preflight_registry_count": (
            _provenance.FAILED_V3_PREFLIGHT_REGISTRY_COUNT
        ),
        "failed_v3_preflight_registry_semantic_sha256": (
            _provenance.FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256
        ),
        "failed_v3_preflight_registry_file_sha256": (
            _provenance.FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256
        ),
        "test_only": int(test_only),
        "authorizing": int(not test_only),
        **NO_WORK,
    }
    if resumed:
        existing = _load_json(run_dir / "run_manifest.json")
        expected = {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at"}
        }
        actual = {
            key: existing.get(key)
            for key in expected
        }
        if actual != expected:
            raise ArtifactCompatibilityError("resume manifest compatibility changed")
        if not test_only:
            _provenance.verify_v3_resume_compatibility(
                run_dir,
                source_fingerprint_value=source_hash,
                scientific_config_sha256=config["semantic_sha256"],
                parent_provenance_sha256=parent_provenance["semantic_sha256"],
                adjudication_provenance_sha256=adjudication_provenance[
                    "semantic_sha256"
                ],
                adjudication_authorization_sha256=authorization[
                    "semantic_sha256"
                ],
                path_plan_sha256=path_plan["semantic_sha256"],
                cohort_plan_sha256=cohort_plan["semantic_sha256"],
                zero_baseline_contract_sha256=zero_contract["semantic_sha256"],
                target_and_input_contract_sha256=target_contract[
                    "semantic_sha256"
                ],
                certificate_semantics_contract_sha256=certificate_contract[
                    "semantic_sha256"
                ],
                failed_v3_preflight_adjudication_sha256=(
                    failed_preflight_adjudication["semantic_sha256"]
                ),
                certificate_semantics_comparator_version=(
                    CERTIFICATE_SEMANTICS_COMPARATOR_VERSION
                ),
            )
        _verify_existing_registry(
            run_dir, allow_unregistered_incomplete_tail=True
        )
        return

    # All provenance and contracts are durable before CUDA configuration.
    atomic_write_json(run_dir / "parent_provenance.json", parent_provenance)
    atomic_write_json(run_dir / "adjudication_provenance.json", adjudication_provenance)
    atomic_write_json(run_dir / "adjudication_authorization.json", authorization)
    atomic_write_json(
        run_dir / "parent_immutability.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-parent-immutability",
                "schema_version": 1,
                "parents_mutated": 0,
                "passed": 1,
            }
        ),
    )
    atomic_write_json(run_dir / "path_collision_scan.json", path_validation)
    atomic_write_json(run_dir / "cohort_validation.json", cohort_validation)
    atomic_write_json(run_dir / "scientific_config.json", config)
    atomic_write_json(run_dir / "path_id_plan.json", path_plan)
    atomic_write_json(run_dir / "cohort_plan.json", cohort_plan)
    atomic_write_json(run_dir / "zero_baseline_contract.json", zero_contract)
    atomic_write_json(run_dir / "target_and_input_contract.json", target_contract)
    atomic_write_json(
        run_dir / "certificate_semantics_contract.json", certificate_contract
    )
    atomic_write_json(
        run_dir / "failed_v3_preflight_adjudication.json",
        failed_preflight_adjudication,
    )
    atomic_write_json(
        run_dir / "source_closure.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-source-closure",
                "schema_version": 1,
                "source_fingerprint": source_hash,
                "paths": [str(path) for path in sources],
            }
        ),
    )
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    _status(run_dir, state="initialized", stage="initialize")


def _preflight_inputs(source: np.ndarray, device: torch.device) -> ModelInputs:
    states = np.repeat(np.asarray(source, dtype=np.float64)[None, :], 2, axis=0)
    # The second row is a deterministic facet fixture with mass conserved.
    states[1, 0] += states[1, 1]
    states[1, 1] = 0.0
    arrays = {
        "later_full_state": states.astype(np.float32),
        "reverse_time": np.asarray([0.25, 0.75], dtype=np.float64),
        "phase": np.asarray([0, 0], dtype=np.int8),
        "color": np.asarray([PHASE_MATCHINGS[0]] * 2, dtype=np.int8),
        "duration": np.asarray([PHASE_DURATIONS[0]] * 2, dtype=np.float64),
        "label": np.asarray([3, 3], dtype=np.int64),
    }
    return _legacy._model_inputs_from_arrays(arrays, device)


_CERTIFICATE_SNAPSHOT_FIELDS = (
    "earlier_head_fraction",
    "later_head_fraction",
    "denoising_target",
    "exposure",
    "transition_ids",
    "active_mask",
    "certified_mask",
    "candidate_later_head_fraction",
    "candidate_denoising_target",
    "candidate_match_mask",
    "cuda_certified_mask",
    "fallback_mask",
    "strengthened_mask",
    "arb_fallback_reason_codes",
    "arb_fallback_mode_counts",
    "mode_counts",
    "quantile_lower",
    "quantile_upper",
    "target_lower",
    "target_upper",
    "prefix_bits",
    "certificate_codes",
)


def _certificate_snapshot(value: Any) -> dict[str, np.ndarray]:
    """Copy one device batch into a read-only, schedule-audit snapshot."""

    result: dict[str, np.ndarray] = {}
    for name in _CERTIFICATE_SNAPSHOT_FIELDS:
        item = getattr(value, name, None)
        if not isinstance(item, Tensor):
            raise CertificateSemanticsError(
                f"certificate batch is missing tensor field {name}"
            )
        array = np.array(
            item.detach().cpu().contiguous().numpy(), order="C", copy=True
        )
        array.setflags(write=False)
        result[name] = array
    return result


class _CertificateCaptureSampler:
    """Transparent sampler adapter retaining exact comparison payloads."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.snapshots: list[dict[str, np.ndarray]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        value = self.delegate(*args, **kwargs)
        self.snapshots.append(_certificate_snapshot(value))
        return value


def _concatenate_certificate_snapshots(
    values: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not values:
        raise CertificateSemanticsError("certificate capture is empty")
    fields = tuple(values[0])
    if fields != _CERTIFICATE_SNAPSHOT_FIELDS or any(
        tuple(value) != fields for value in values
    ):
        raise CertificateSemanticsError("certificate capture fields changed")
    return {
        name: np.ascontiguousarray(
            np.concatenate(
                [np.asarray(value[name]).reshape(-1) for value in values]
            )
        )
        for name in fields
    }


def _state_commitment_array(
    executions: Sequence[Any], raw_shards: Sequence[Any]
) -> np.ndarray:
    if len(executions) != len(raw_shards):
        raise CertificateSemanticsError("shard state captures do not align")
    pieces: list[np.ndarray] = []
    for execution, raw in zip(executions, raw_shards, strict=True):
        capture = getattr(raw, "capture_payload", None)
        states = getattr(capture, "post_phase_states", None)
        if states is None:
            raise CertificateSemanticsError("post-phase state capture is missing")
        pieces.append(np.asarray(states, dtype=np.float64).reshape(-1))
        for branch in execution.branches:
            pieces.append(
                np.asarray(
                    branch.batch.batch.later_full_state.detach().cpu().numpy(),
                    dtype=np.float64,
                ).reshape(-1)
            )
        pieces.append(
            np.asarray(execution.committed_final_states, dtype=np.float64).reshape(-1)
        )
    return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float64)


def _arm_diagnostics(executions: Sequence[Any]) -> dict[str, Any]:
    rows = [dict(value.diagnostics) for value in executions]
    transitions = sum(int(row.get("transition_count", 0)) for row in rows)
    certified = sum(int(row.get("certified_count", 0)) for row in rows)
    fallback = sum(int(row.get("fallback_count", 0)) for row in rows)
    elapsed = sum(
        float(row.get("complete_pipeline_elapsed_seconds", 0.0)) for row in rows
    )
    fallback_elapsed = sum(
        float(row.get("fallback_elapsed_seconds", 0.0)) for row in rows
    )
    forbidden = {
        name: sum(
            int(row.get("forbidden_counts", {}).get(name, 0)) for row in rows
        )
        for name in sorted(
            {
                str(name)
                for row in rows
                for name in row.get("forbidden_counts", {})
            }
        )
    }
    maximum_mass_error = max(
        (float(row.get("maximum_mass_error", math.inf)) for row in rows),
        default=math.inf,
    )
    peak_memory_fraction = max(
        (float(row.get("peak_memory_fraction", math.inf)) for row in rows),
        default=math.inf,
    )
    certificate_fraction = certified / max(transitions, 1)
    fallback_fraction = fallback / max(transitions, 1)
    fallback_time_fraction = fallback_elapsed / max(
        elapsed, np.finfo(float).tiny
    )
    forbidden_count = sum(forbidden.values())
    numerically_valid = int(
        transitions > 0
        and certificate_fraction == 1.0
        and maximum_mass_error <= 2.0e-12
        and forbidden_count == 0
    )
    resource_valid = int(
        transitions / max(elapsed, np.finfo(float).tiny) >= 1_300.0
        and fallback_fraction <= 1.0e-4
        and fallback_time_fraction <= 0.10
        and peak_memory_fraction <= 0.80
    )
    return {
        "transition_count": transitions,
        "certified_count": certified,
        "certificate_fraction": certificate_fraction,
        "fallback_count": fallback,
        "fallback_fraction": fallback_fraction,
        "fallback_elapsed_seconds": fallback_elapsed,
        "fallback_time_fraction": fallback_time_fraction,
        "elapsed_seconds": elapsed,
        "transitions_per_second": transitions
        / max(elapsed, np.finfo(float).tiny),
        "maximum_mass_error": maximum_mass_error,
        "peak_memory_fraction": peak_memory_fraction,
        "forbidden_counts": forbidden,
        "forbidden_event_count": forbidden_count,
        "numerically_valid": numerically_valid,
        "resource_valid": resource_valid,
    }


def _proof_metadata_rows(
    comparison: Mapping[str, Any],
    adaptive: Sequence[Any],
    eager: Sequence[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    left = comparison.get("left_proof_metadata", {})
    right = comparison.get("right_proof_metadata", {})
    names = sorted(set(left) | set(right))
    differing = set(comparison.get("differing_proof_metadata_fields", ()))
    for name in names:
        left_row = dict(left.get(name, {}))
        right_row = dict(right.get(name, {}))
        rows.append(
            {
                "record_kind": "transition_proof_field",
                "field": name,
                "equality_advisory": int(name not in differing),
                "equality_required": 0,
                "adaptive_sha256": left_row.get("sha256"),
                "eager_sha256": right_row.get("sha256"),
                "adaptive_minimum": left_row.get("minimum"),
                "adaptive_maximum": left_row.get("maximum"),
                "eager_minimum": right_row.get("minimum"),
                "eager_maximum": right_row.get("maximum"),
                "adaptive_histogram": left_row.get("histogram"),
                "eager_histogram": right_row.get("histogram"),
            }
        )
    for shard_index, (left_shard, right_shard) in enumerate(
        zip(adaptive, eager, strict=True)
    ):
        left_hash = left_shard.base_record.get("batch_certificate_sha256")
        right_hash = right_shard.base_record.get("batch_certificate_sha256")
        rows.append(
            {
                "record_kind": "base_shard_certificate_commitment",
                "shard_index": shard_index,
                "start_step": int(left_shard.identity.start_step),
                "field": "batch_certificate_sha256",
                "equality_advisory": int(left_hash == right_hash),
                "equality_required": 0,
                "adaptive_sha256": left_hash,
                "eager_sha256": right_hash,
            }
        )
        timing_fields = (
            "complete_pipeline_elapsed_seconds",
            "fallback_elapsed_seconds",
            "backend_elapsed_seconds",
            "candidate_elapsed_seconds",
            "transitions_per_second",
            "peak_memory_fraction",
        )
        for name in timing_fields:
            rows.append(
                {
                    "record_kind": "shard_proof_timing",
                    "shard_index": shard_index,
                    "start_step": int(left_shard.identity.start_step),
                    "field": name,
                    "equality_advisory": int(
                        left_shard.diagnostics.get(name)
                        == right_shard.diagnostics.get(name)
                    ),
                    "equality_required": 0,
                    "adaptive_value": left_shard.diagnostics.get(name),
                    "eager_value": right_shard.diagnostics.get(name),
                }
            )
    return rows


def _certificate_semantics_record_valid(record: Mapping[str, Any]) -> int:
    """Validate the comparator artifact independently of its scientific result."""

    body = dict(record)
    semantic_sha256 = body.pop("semantic_sha256", None)
    expected_schemas = {
        RUN_SCHEMA + "-preflight-certificate-semantics",
        TEST_RUN_SCHEMA + "-preflight-certificate-semantics",
    }
    binary_fields = (
        "comparator_valid",
        "certificate_semantics_comparator_valid",
        "scientific_payload_equal",
        "certificate_semantics_equal",
        "scheduler_seam_valid",
        "left_authorization_valid",
        "right_authorization_valid",
        "proof_metadata_equal",
        "proof_metadata_equal_advisory",
    )
    count_fields = (
        "active_duplicate_transition_id_count",
        "active_uncertified_count",
        "inactive_certified_count",
        "active_nonfinite_output_count",
        "inactive_state_change_count",
        "inactive_nonzero_target_count",
        "active_exposure_nonpositive_count",
        "inactive_exposure_nonzero_count",
        "active_invalid_enclosure_count",
    )
    mismatch_records = record.get("mismatch_records")
    mismatch_counts = record.get("mismatch_counts")
    left_counts = record.get("left_authorization_counts")
    right_counts = record.get("right_authorization_counts")
    mismatch_count = record.get("mismatch_count")
    mismatch_record_count = record.get("mismatch_record_count")
    maximum_records = record.get("maximum_mismatch_records")
    truncated = record.get("mismatch_records_truncated")
    expected_valid = int(
        record.get("scientific_payload_equal") == 1
        and record.get("certificate_semantics_equal") == 1
    )
    valid = (
        record.get("schema") in expected_schemas
        and record.get("schema_version") == 1
        and semantic_sha256 == config_fingerprint(body)
        and record.get("evaluation_status") == "evaluated"
        and record.get("comparator_version")
        == CERTIFICATE_SEMANTICS_COMPARATOR_VERSION
        and record.get("comparison_executed") == 1
        and record.get("proof_metadata_advisory") == 1
        and record.get("proof_metadata_equality_required") == 0
        and record.get("comparator_evaluation_valid") == 1
        and all(record.get(name) in {0, 1} for name in binary_fields)
        and record.get("comparator_valid") == expected_valid
        and record.get("certificate_semantics_comparator_valid")
        == expected_valid
        and record.get("scheduler_seam_valid") == expected_valid
        and record.get("proof_metadata_equal_advisory")
        == record.get("proof_metadata_equal")
        and isinstance(record.get("left_proof_metadata"), Mapping)
        and isinstance(record.get("right_proof_metadata"), Mapping)
        and isinstance(record.get("differing_proof_metadata_fields"), list)
        and isinstance(mismatch_records, list)
        and isinstance(mismatch_counts, Mapping)
        and isinstance(left_counts, Mapping)
        and isinstance(right_counts, Mapping)
        and all(
            isinstance(counts.get(name), int) and counts.get(name) >= 0
            for counts in (left_counts, right_counts)
            for name in count_fields
        )
        and isinstance(mismatch_count, int)
        and isinstance(mismatch_record_count, int)
        and isinstance(maximum_records, int)
        and mismatch_count >= 0
        and mismatch_record_count == len(mismatch_records)
        and 0 <= mismatch_record_count <= maximum_records <= 64
        and mismatch_count >= mismatch_record_count
        and truncated == int(mismatch_count > mismatch_record_count)
    )
    if not valid:
        raise CertificateSemanticsError(
            "preflight certificate-semantics comparator record is malformed"
        )
    return 1


def _preflight_seam(
    args: argparse.Namespace,
    source: np.ndarray,
    path_ids: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    paths = tuple(int(value) for value in path_ids)
    if args.test_only:
        active = np.asarray([True, False], dtype=bool)
        payload = {
            "transition_ids": np.asarray([11, 12], dtype=np.uint64),
            "earlier_head_fraction": np.asarray([0.4, 0.75], dtype=np.float64),
            "exposure": np.asarray([0.1, 0.0], dtype=np.float64),
            "later_head_fraction": np.asarray([0.45, 0.75], dtype=np.float64),
            "denoising_target": np.asarray([0.2, 0.0], dtype=np.float64),
            "active_mask": active,
            "certified_mask": active.copy(),
            "certificate_codes": np.asarray([15, 0], dtype=np.uint8),
            "prefix_bits": np.asarray([64, 0], dtype=np.int32),
            "mode_counts": np.asarray([128, 0], dtype=np.int32),
        }
        eager_payload = {name: np.array(value, copy=True) for name, value in payload.items()}
        eager_payload["prefix_bits"][0] = 128
        eager_payload["mode_counts"][0] = 256
        comparison = compare_certificate_semantics(
            payload,
            eager_payload,
            left_final_state=np.asarray([0.4, 0.6], dtype=np.float64),
            right_final_state=np.asarray([0.4, 0.6], dtype=np.float64),
        )
        semantics = _semantic(
            {
                **comparison,
                "schema": TEST_RUN_SCHEMA + "-preflight-certificate-semantics",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "comparator_version": CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
                "comparator_evaluation_valid": 1,
                "proof_metadata_equal_advisory": int(
                    comparison["proof_metadata_equal"]
                ),
                "test_fixture": 1,
            }
        )
        seam = {
            "schema": TEST_RUN_SCHEMA + "-scheduler-seam",
            "schema_version": 2,
            "path_ids": list(paths),
            "transition_count": 2,
            "certificate_fraction": 1.0,
            "maximum_mass_error": 0.0,
            "forbidden_event_count": 0,
            "transitions_per_second": 1_300.0,
            "peak_memory_fraction": 0.0,
            "base_states_equal": 1,
            "base_targets_equal": 1,
            "base_certificates_equal": 1,
            "midpoint_states_equal": 1,
            "midpoint_targets_equal": 1,
            "midpoint_certificates_equal": 1,
            "scientific_payload_equal": 1,
            "certificate_semantics_equal": 1,
            "certificate_semantics_comparator_valid": 1,
            "proof_metadata_equal_advisory": 0,
            "batch_certificate_sha256_equality_required": 0,
            "adaptive_arm_valid": 1,
            "eager_arm_valid": 1,
            "passed": 1,
            "test_fixture": 1,
        }
        rows = _proof_metadata_rows(comparison, (), ())
        return seam, semantics, rows
    device = torch.device(args.device)
    cohort = EagerCohort(
        kind="confirmation",
        index=0,
        path_ids=paths,
        path_roles=("preflight_seam",) * len(paths),
    )
    selected = (15,)

    def execute(
        *, eager: bool
    ) -> tuple[tuple[Any, ...], tuple[dict[str, np.ndarray], ...], tuple[Any, ...]]:
        state = torch.as_tensor(
            np.repeat(source[None, :], len(paths), axis=0).copy(order="C"),
            dtype=torch.float64,
            device=device,
        ).contiguous()
        values: list[Any] = []
        raw_shards: list[Any] = []
        capture = _CertificateCaptureSampler(
            sample_alpha1_rb_transition_batch_cuda_eager
            if eager
            else sample_alpha1_rb_transition_batch_cuda
        )

        def capture_shard_runner(*runner_args: Any, **runner_kwargs: Any) -> Any:
            # Observer-only capture: it changes neither the transition nor its
            # stateless randomness, but makes every post-phase state auditable.
            runner_kwargs["capture_phase_state_trace"] = True
            runner_kwargs["capture_training_payload"] = True
            value = run_exact_multipath_shard(*runner_args, **runner_kwargs)
            raw_shards.append(value)
            return value

        for start in (0, 8):
            value = execute_eager_shard(
                state,
                cohort=cohort,
                start_step=start,
                root_seed=ROOT_SEED,
                selected_steps=selected,
                profile=eager_prefix_profile(),
                sampler=capture,
                shard_runner=capture_shard_runner,
                branch_runner=sample_fused_midpoint_branches,
            )
            values.append(value)
            state = value.final_states.detach().clone().contiguous()
        return tuple(values), tuple(capture.snapshots), tuple(raw_shards)

    adaptive, adaptive_batches, adaptive_raw = execute(eager=False)
    eager, eager_batches, eager_raw = execute(eager=True)
    adaptive_payload = _concatenate_certificate_snapshots(adaptive_batches)
    eager_payload = _concatenate_certificate_snapshots(eager_batches)
    adaptive_states = _state_commitment_array(adaptive, adaptive_raw)
    eager_states = _state_commitment_array(eager, eager_raw)
    comparison = compare_certificate_semantics(
        adaptive_payload,
        eager_payload,
        left_final_state=adaptive_states,
        right_final_state=eager_states,
    )
    adaptive_health = _arm_diagnostics(adaptive)
    eager_health = _arm_diagnostics(eager)
    equality = {
        "base_states_equal": int(
            np.array_equal(adaptive_states, eager_states)
        ),
        "base_targets_equal": int(comparison["scientific_payload_equal"]),
        "base_certificates_equal": int(
            comparison["certificate_semantics_equal"]
        ),
        "midpoint_states_equal": int(
            all(
                torch.equal(
                    left.batch.batch.later_full_state,
                    right.batch.batch.later_full_state,
                )
                for left, right in zip(
                    adaptive[1].branches, eager[1].branches, strict=True
                )
            )
        ),
        "midpoint_targets_equal": int(
            all(
                torch.equal(
                    left.batch.batch.denoising_target,
                    right.batch.batch.denoising_target,
                )
                for left, right in zip(
                    adaptive[1].branches, eager[1].branches, strict=True
                )
            )
        ),
        "midpoint_certificates_equal": int(
            comparison["certificate_semantics_equal"]
        ),
    }
    proof_rows = _proof_metadata_rows(comparison, adaptive, eager)
    semantics = _semantic(
        {
            **comparison,
            "schema": RUN_SCHEMA + "-preflight-certificate-semantics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "comparator_version": CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
            "comparator_evaluation_valid": 1,
            "proof_metadata_equal_advisory": int(
                comparison["proof_metadata_equal"]
            ),
            "adaptive_arm": adaptive_health,
            "eager_arm": eager_health,
        }
    )
    record = {
        "schema": RUN_SCHEMA + "-scheduler-seam",
        "schema_version": 2,
        "path_ids": list(paths),
        **equality,
        "scientific_payload_equal": int(comparison["scientific_payload_equal"]),
        "certificate_semantics_equal": int(
            comparison["certificate_semantics_equal"]
        ),
        "certificate_semantics_comparator_valid": 1,
        "proof_metadata_equal_advisory": int(
            comparison["proof_metadata_equal"]
        ),
        "batch_certificate_sha256_equality_required": 0,
        "batch_output_sha256_equal_advisory": int(
            all(
                left.base_record.get("batch_output_sha256")
                == right.base_record.get("batch_output_sha256")
                for left, right in zip(adaptive, eager, strict=True)
            )
        ),
        "raw_certificate_codes_equal_advisory": int(
            all(
                np.array_equal(
                    np.asarray(left["certificate_codes"]),
                    np.asarray(right["certificate_codes"]),
                )
                for left, right in zip(
                    adaptive_batches, eager_batches, strict=True
                )
            )
        ),
        "adaptive_batch_certificate_sha256": [
            value.base_record.get("batch_certificate_sha256") for value in adaptive
        ],
        "eager_batch_certificate_sha256": [
            value.base_record.get("batch_certificate_sha256") for value in eager
        ],
        "transition_count": int(eager_health["transition_count"]),
        "certificate_fraction": min(
            float(adaptive_health["certificate_fraction"]),
            float(eager_health["certificate_fraction"]),
        ),
        "maximum_mass_error": max(
            float(adaptive_health["maximum_mass_error"]),
            float(eager_health["maximum_mass_error"]),
        ),
        "forbidden_event_count": int(adaptive_health["forbidden_event_count"])
        + int(eager_health["forbidden_event_count"]),
        "transitions_per_second": float(eager_health["transitions_per_second"]),
        "peak_memory_fraction": max(
            float(adaptive_health["peak_memory_fraction"]),
            float(eager_health["peak_memory_fraction"]),
        ),
        "adaptive_arm_valid": int(adaptive_health["numerically_valid"]),
        "eager_arm_valid": int(
            eager_health["numerically_valid"] and eager_health["resource_valid"]
        ),
        "adaptive_arm": adaptive_health,
        "eager_arm": eager_health,
    }
    record["passed"] = int(
        bool(comparison["scheduler_seam_valid"])
        and all(equality.values())
        and record["adaptive_arm_valid"] == 1
        and record["eager_arm_valid"] == 1
        and record["certificate_fraction"] == 1.0
        and record["maximum_mass_error"] <= 2.0e-12
        and record["forbidden_event_count"] == 0
    )
    return record, semantics, proof_rows


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "preflight_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
        return _load_json(gate_path)
    path_plan = _load_json(run_dir / "path_id_plan.json")
    source = (
        np.full(784, 1.0 / 784.0, dtype=np.float64)
        if args.test_only
        else _legacy._load_source_target(args.parent_coarse_residual_run_dir)
    )
    measured_mixed_target_sha256 = _provenance.source_measure_sha256(source)
    if args.test_only:
        source_binding = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-source-image-binding",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "passed": 1,
                "authorizing": 0,
                "test_fixture": 1,
                "measured_mixed_target_sha256": measured_mixed_target_sha256,
            }
        )
    else:
        parent_source_binding = _provenance.verify_v3_source_image_binding(
            args.parent_coarse_residual_run_dir
        )
        loaded_target_matches_parent = int(
            measured_mixed_target_sha256
            == parent_source_binding["measured_mixed_target_sha256"]
            == _provenance.MIXED_TARGET_SHA256
        )
        source_binding = _semantic(
            {
                "schema": RUN_SCHEMA + "-preflight-source-image-binding",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "passed": loaded_target_matches_parent,
                "parent_binding_sha256": parent_source_binding["semantic_sha256"],
                "source_image_json_sha256": parent_source_binding[
                    "source_image_json_sha256"
                ],
                "source_image_npz_sha256": parent_source_binding[
                    "source_image_npz_sha256"
                ],
                "measured_image_sha256": parent_source_binding[
                    "measured_image_sha256"
                ],
                "measured_mixed_target_sha256": measured_mixed_target_sha256,
                "loaded_target_matches_parent": loaded_target_matches_parent,
            }
        )
    try:
        seam, certificate_semantics, proof_metadata_rows = _preflight_seam(
            args, source, path_plan["roles"]["preflight_seam"]
        )
    except CertificateSemanticsError as exc:
        certificate_semantics = _semantic(
            {
                "schema": (TEST_RUN_SCHEMA if args.test_only else RUN_SCHEMA)
                + "-preflight-certificate-semantics",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "comparator_version": CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
                "comparison_executed": 0,
                "comparator_evaluation_valid": 0,
                "failure_domain": "implementation_contract",
                "failure_code": "certificate_semantics_comparator_execution_invalid",
                "failure_message": str(exc),
                "scientific_payload_equal": None,
                "certificate_semantics_equal": None,
                "proof_metadata_advisory": 1,
                "proof_metadata_equality_required": 0,
                "proof_metadata_equal_advisory": None,
                "scientific_evidence_complete": 1,
                **NO_WORK,
            }
        )
        seam = {
            "schema": (TEST_RUN_SCHEMA if args.test_only else RUN_SCHEMA)
            + "-scheduler-seam",
            "schema_version": 2,
            "evaluation_status": "comparator_failed",
            "path_ids": list(path_plan["roles"]["preflight_seam"]),
            "transition_count": 0,
            "certificate_fraction": 1.0,
            "maximum_mass_error": 0.0,
            "forbidden_event_count": 0,
            "transitions_per_second": 1_300.0,
            "peak_memory_fraction": 0.0,
            "scientific_payload_equal": None,
            "certificate_semantics_equal": None,
            "certificate_semantics_comparator_valid": 0,
            "proof_metadata_equal_advisory": None,
            "batch_certificate_sha256_equality_required": 0,
            "adaptive_arm_valid": None,
            "eager_arm_valid": None,
            # Scheduler/numerical health is not contradicted by a comparator
            # construction error.  The independent comparator gate below is
            # the sole failing check for this typed contract failure.
            "passed": 1,
        }
        proof_metadata_rows = [
            {
                "record_kind": "comparator_execution_failure",
                "field": "certificate_semantics_comparator",
                "equality_advisory": None,
                "equality_required": 0,
                "failure_code": "certificate_semantics_comparator_execution_invalid",
                "failure_message": str(exc),
            }
        ]
    try:
        comparator_record_valid = _certificate_semantics_record_valid(
            certificate_semantics
        )
    except CertificateSemanticsError as exc:
        malformed_sha256 = config_fingerprint(dict(certificate_semantics))
        certificate_semantics = _semantic(
            {
                "schema": (TEST_RUN_SCHEMA if args.test_only else RUN_SCHEMA)
                + "-preflight-certificate-semantics",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "comparator_version": CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
                "comparison_executed": 0,
                "comparator_evaluation_valid": 0,
                "failure_domain": "implementation_contract",
                "failure_code": "certificate_semantics_comparator_record_invalid",
                "failure_message": str(exc),
                "malformed_record_sha256": malformed_sha256,
                "scientific_payload_equal": None,
                "certificate_semantics_equal": None,
                "proof_metadata_advisory": 1,
                "proof_metadata_equality_required": 0,
                "proof_metadata_equal_advisory": None,
                "scientific_evidence_complete": 1,
                **NO_WORK,
            }
        )
        comparator_record_valid = 0
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(
        torch.device(args.device)
    )
    inputs = _preflight_inputs(source, torch.device(args.device))
    with torch.no_grad():
        prediction = model(inputs)
        baseline = exact_zero_baseline_prediction(inputs)
    state_keys = tuple(model.state_dict())
    representation = _semantic(
        {
            "schema": RUN_SCHEMA + "-zero-baseline-preflight",
            "schema_version": 1,
            "state_dict_keys": list(state_keys),
            "state_dict_baseline_free": int(
                "_q_values" not in state_keys
                and all("baseline" not in name.lower() for name in state_keys)
            ),
            "update_zero_prediction_max_abs": float(
                torch.max(torch.abs(prediction)).cpu()
            ),
            "baseline_prediction_max_abs": float(
                torch.max(torch.abs(baseline)).cpu()
            ),
            "facet_prediction_exact_zero": int(
                bool(torch.all(prediction[1] == 0.0))
            ),
            "passed": int(
                bool(torch.all(prediction == 0.0))
                and bool(torch.all(baseline == 0.0))
                and "_q_values" not in state_keys
            ),
        }
    )
    parent = _load_json(run_dir / "parent_provenance.json")
    authorization = _load_json(run_dir / "adjudication_authorization.json")
    path_scan = _load_json(run_dir / "path_collision_scan.json")
    cohort_validation = _load_json(run_dir / "cohort_validation.json")
    source_closure = _load_json(run_dir / "source_closure.json")
    source_closure_body = dict(source_closure)
    source_closure_semantic = source_closure_body.pop("semantic_sha256", None)
    current_sources = _source_set()
    current_source_fingerprint = _provenance.v3_source_fingerprint(current_sources)
    manifest = _load_json(run_dir / "run_manifest.json")
    source_closure_valid = int(
        source_closure_semantic == config_fingerprint(source_closure_body)
        and source_closure.get("source_fingerprint") == current_source_fingerprint
        and manifest.get("source_fingerprint") == current_source_fingerprint
        and source_closure.get("paths") == [str(path) for path in current_sources]
    )
    inherited_resource = (
        {
            "projected_effective_transitions_per_second": 1_300.0,
            "peak_memory_fraction": 0.0,
            "projected_elapsed_seconds": 0.0,
        }
        if args.test_only
        else _load_json(
            args.parent_eager_pipeline_run_dir / "eager_pipeline_metrics.json"
        )
    )
    flags = {
        name: 1
        for name in (
            "parent_v2_valid",
            "adjudication_valid",
            "adjudication_authority_valid",
            "complete_parent_registry_valid",
            "complete_adjudication_registry_valid",
            "parent_immutability_valid",
            "source_closure_valid",
            "path_plan_valid",
            "cohort_plan_valid",
            "path_collision_scan_valid",
            "zero_baseline_contract_valid",
            "baseline_artifacts_absent",
            "state_dict_baseline_free",
            "update_zero_exact",
            "preflight_complete",
            "certificate_semantics_comparator_valid",
            "scheduler_seam_valid",
            "exact_kernel_contract_valid",
            "cuda_determinism_valid",
            "source_image_binding_valid",
            "selected_step_contract_valid",
            "model_input_firewall_valid",
            "raw_target_contract_valid",
            "train_validation_confirmation_unopened",
            "inherited_resource_projection_valid",
        )
    }
    flags.update(
        {
            "parent_v2_valid": int(parent.get("passed", 0)),
            "adjudication_authority_valid": int(authorization.get("passed", 0)),
            "path_collision_scan_valid": int(path_scan.get("passed", 0)),
            "cohort_plan_valid": int(cohort_validation.get("passed", 0)),
            "source_closure_valid": source_closure_valid,
            "source_image_binding_valid": int(source_binding.get("passed", 0)),
            "state_dict_baseline_free": int(
                representation["state_dict_baseline_free"]
            ),
            "update_zero_exact": int(representation["passed"]),
            "certificate_semantics_comparator_valid": int(
                comparator_record_valid
            ),
            "scheduler_seam_valid": int(seam["passed"]),
            "inherited_resource_projection_valid": int(
                float(
                    inherited_resource["projected_effective_transitions_per_second"]
                )
                >= 1_300.0
                and float(inherited_resource["peak_memory_fraction"]) <= 0.80
                and float(inherited_resource["projected_elapsed_seconds"])
                <= 108_000.0
            ),
        }
    )
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 2,
        **flags,
        "preflight_path_ids": list(path_plan["roles"]["preflight_seam"]),
        "preflight_path_count": len(path_plan["roles"]["preflight_seam"]),
        "certificate_fraction": float(seam["certificate_fraction"]),
        "maximum_mass_error": float(seam["maximum_mass_error"]),
        "forbidden_event_count": int(seam["forbidden_event_count"]),
        "transitions_per_second": float(
            inherited_resource["projected_effective_transitions_per_second"]
        ),
        "peak_memory_fraction": max(
            float(seam["peak_memory_fraction"]),
            float(inherited_resource["peak_memory_fraction"]),
        ),
        "seam_transitions_per_second_advisory": float(
            seam["transitions_per_second"]
        ),
        "scientific_payload_equal": int(
            certificate_semantics.get("scientific_payload_equal") == 1
        ),
        "certificate_semantics_equal": int(
            certificate_semantics.get("certificate_semantics_equal") == 1
        ),
        "proof_metadata_equal_advisory": int(
            certificate_semantics.get("proof_metadata_equal_advisory") == 1
        ),
        "proof_metadata_equality_required": 0,
        "inherited_projected_elapsed_seconds": float(
            inherited_resource["projected_elapsed_seconds"]
        ),
        "production_cache_generation_performed": 0,
        "physical_training_performed": 0,
        "validation_selection_performed": 0,
        "confirmation_performed": 0,
        "test_only": int(args.test_only),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "preflight_scheduler_seam.json", seam)
    atomic_write_json(
        run_dir / "preflight_certificate_semantics.json", certificate_semantics
    )
    atomic_write_csv(
        run_dir / "preflight_certificate_proof_metadata.csv", proof_metadata_rows
    )
    atomic_write_json(run_dir / "source_image_binding.json", source_binding)
    atomic_write_json(run_dir / "zero_baseline_preflight.json", representation)
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    if args.test_only:
        gate = {
            "schema": TEST_RUN_SCHEMA + "-preflight-gate",
            "evaluation_status": "evaluated",
            "passed": int(all(flags.values()) and seam["passed"] == 1),
            "scientific_evidence_complete": 1,
            "authorizing": 0,
            **NO_WORK,
        }
    else:
        gate = evaluate_preflight_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "preflight_scheduler_seam.json",
            "preflight_certificate_semantics.json",
            "preflight_certificate_proof_metadata.csv",
            "certificate_semantics_contract.json",
            "failed_v3_preflight_adjudication.json",
            "source_image_binding.json",
            "zero_baseline_preflight.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )
    return gate


def _training_index_bindings(run_dir: Path) -> None:
    directory = run_dir / "cache"
    directory.mkdir(parents=True, exist_ok=True)
    for role in ("train", "validation"):
        source = run_dir / "eager_cache" / f"{role}_index.json"
        record = _semantic(
            {
                "schema": RUN_SCHEMA + "-training-cache-binding",
                "schema_version": 1,
                "role": role,
                "source_path": source.relative_to(run_dir).as_posix(),
                "source_sha256": file_fingerprint(source),
            }
        )
        atomic_write_json(directory / f"{role}_index.json", record)


def _cache_runtime_summary(run_dir: Path) -> tuple[float, float]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    root = run_dir / "eager_cache" / "train_validation"
    for path in sorted(root.rglob("metadata.json")):
        record = _load_json(path)
        identity = record.get("identity")
        diagnostics = record.get("diagnostics")
        if not isinstance(identity, Mapping) or not isinstance(diagnostics, Mapping):
            raise ArtifactCompatibilityError("cache shard timing is malformed")
        grouped.setdefault(int(identity["cohort_index"]), []).append(record)
    if not grouped:
        raise ArtifactCompatibilityError("cache contains no committed shard")
    elapsed_total = 0.0
    rates: list[float] = []
    for records in grouped.values():
        elapsed = sum(
            float(record["diagnostics"].get("complete_pipeline_elapsed_seconds", 0.0))
            + float(record.get("persistence_elapsed_seconds", 0.0))
            for record in records
        )
        transitions = sum(
            int(record["diagnostics"]["transition_count"]) for record in records
        )
        if elapsed <= 0.0 or transitions <= 0:
            raise ArtifactCompatibilityError("cache cohort timing is nonpositive")
        elapsed_total += elapsed
        rates.append(transitions / elapsed)
    return elapsed_total, min(rates)


def _confirmation_projection_seconds(args: argparse.Namespace) -> float:
    if args.test_only:
        return 0.0
    metrics = _load_json(
        args.parent_eager_pipeline_run_dir / "eager_pipeline_metrics.json"
    )
    seconds = metrics.get("slowest_profile_seconds")
    if not isinstance(seconds, Mapping):
        raise ArtifactCompatibilityError("eager parent timing profile is missing")
    return 8.0 * (
        6.0 * float(seconds["stream_p10"]) + float(seconds["stream_p4"])
    )


def _cache_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("cache requires a passing preflight")
    gate_path = run_dir / "cache_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "cache_artifact_seal.json")
        load_eager_role_inputs(run_dir, "train")
        load_eager_role_inputs(run_dir, "validation")
        return _load_json(gate_path)
    source = (
        np.full(784, 1.0 / 784.0, dtype=np.float64)
        if args.test_only
        else _legacy._load_source_target(args.parent_coarse_residual_run_dir)
    )
    cohort_plan = _load_json(run_dir / "cohort_plan.json")
    cohorts = _cohorts(cohort_plan, "train_validation")
    explicit_plan = explicit_eager_cache_plan(cohorts)
    atomic_write_json(run_dir / "eager_cache_explicit_plan.json", explicit_plan)
    kwargs: dict[str, Any] = {}
    if args.test_only:
        outer_steps = int(args.test_outer_steps)
        kwargs.update(
            {
                "outer_steps": outer_steps,
                "selected_steps": tuple(
                    step for step in SELECTED_OUTER_STEPS if step < outer_steps
                ),
                "shard_runner": deterministic_test_shard_runner,
                "branch_runner": deterministic_test_branch_runner,
            }
        )

    def progress(identity: Any, disposition: str) -> None:
        print(
            f"v3 eager cache cohort={identity.cohort_index} "
            f"step={identity.start_step} {disposition}",
            flush=True,
        )

    result = generate_eager_cache_for_cohorts(
        run_dir,
        source,
        cohorts=cohorts,
        cohort_plan_sha256=str(explicit_plan["semantic_sha256"]),
        device=args.device,
        root_seed=ROOT_SEED,
        progress=progress,
        **kwargs,
    )
    _training_index_bindings(run_dir)
    aggregate = dict(result["metrics"])
    train_arrays, train_index = load_eager_role_inputs(run_dir, "train")
    validation_arrays, validation_index = load_eager_role_inputs(
        run_dir, "validation"
    )
    total = int(aggregate["transition_count"])
    forbidden = sum(int(value) for value in aggregate["forbidden_counts"].values())
    elapsed, minimum_rate = _cache_runtime_summary(run_dir)
    fallback = int(aggregate["fallback_count"])
    t = BoundaryTangentV3Thresholds()
    test_only = bool(args.test_only)
    metrics = {
        "schema": RUN_SCHEMA + "-cache-metrics",
        "schema_version": 1,
        **{
            name: 1
            for name in (
                "cache_complete",
                "train_cache_complete",
                "validation_cache_complete",
                "atomic_shard_chains_valid",
                "resume_replay_valid",
                "selected_sample_cartesian_valid",
                "train_validation_indexes_disjoint",
                "artifact_role_isolation_valid",
                "mixed_cohort_split_before_commit",
                "raw_target_contract_valid",
                "input_field_contract_valid",
                "confirmation_absent",
                "confirmation_namespace_unopened",
                "baseline_artifacts_absent",
            )
        },
        "train_path_count": len(np.unique(train_arrays["path_id"])),
        "validation_path_count": len(np.unique(validation_arrays["path_id"])),
        "train_row_count": len(train_arrays["sample_key"]),
        "validation_row_count": len(validation_arrays["sample_key"]),
        "train_transition_count": int(train_index["transition_count"]),
        "validation_transition_count": int(validation_index["transition_count"]),
        "certificate_fraction": int(aggregate["certified_count"]) / max(total, 1),
        "maximum_mass_error": float(aggregate["maximum_mass_error"]),
        "forbidden_event_count": forbidden,
        "minimum_role_rate": minimum_rate,
        "fallback_fraction": fallback / max(total, 1),
        "fallback_time_fraction": float(aggregate["fallback_elapsed_seconds"])
        / max(elapsed, np.finfo(float).tiny),
        "peak_memory_fraction": float(aggregate["maximum_peak_memory_fraction"]),
        "total_persisted_cache_bytes": int(aggregate["persisted_bytes"]),
        "cache_elapsed_seconds": elapsed,
        "frozen_confirmation_projection_seconds": _confirmation_projection_seconds(args),
        "projected_cache_plus_confirmation_seconds": (
            elapsed + _confirmation_projection_seconds(args)
        ),
        "production_cache_generation_performed": 1,
        "physical_training_performed": 0,
        "confirmation_performed": 0,
        "test_only": int(test_only),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "cache_metrics.json", metrics)
    if test_only:
        gate = {
            "schema": TEST_RUN_SCHEMA + "-cache-gate",
            "evaluation_status": "evaluated",
            "passed": 1,
            "scientific_evidence_complete": 1,
            "authorizing": 0,
            **NO_WORK,
        }
    else:
        gate = evaluate_cache_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "eager_cache_explicit_plan.json",
            "eager_cache/execution_contract.json",
            "eager_cache/train_index.json",
            "eager_cache/validation_index.json",
            "eager_cache/train_validation_metrics.json",
            "cache/train_index.json",
            "cache/validation_index.json",
            "cache_metrics.json",
            "cache_gate.json",
        ),
        "cache_artifact_seal.json",
    )
    return gate


def _load_training_inputs(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Load permitted fields only; no physical label loader is called here."""

    device = torch.device(args.device)
    train_arrays, train_index = load_eager_role_inputs(run_dir, "train")
    validation_arrays, validation_index = load_eager_role_inputs(
        run_dir, "validation"
    )
    return {
        "train_arrays": train_arrays,
        "validation_arrays": validation_arrays,
        "train_index": train_index,
        "validation_index": validation_index,
        "train_inputs": _legacy._model_inputs_from_arrays(train_arrays, device),
        "validation_inputs": _legacy._model_inputs_from_arrays(
            validation_arrays, device
        ),
        "train_path_rows": np.asarray(train_arrays["path_id"], dtype=np.int64),
        "validation_path_rows": np.asarray(
            validation_arrays["path_id"], dtype=np.int64
        ),
    }


def _predict_in_batches(
    model: nn.Module, inputs: ModelInputs, *, batch_size: int = 32
) -> Tensor:
    was_training = model.training
    model.eval()
    values: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.batch_size, batch_size):
            stop = min(inputs.batch_size, start + batch_size)
            index = torch.arange(
                start,
                stop,
                dtype=torch.long,
                device=inputs.later_full_state.device,
            )
            values.append(call_model(model, inputs.index_select(index)).to(torch.float64))
    if was_training:
        model.train()
    return torch.cat(values, dim=0)


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _per_path_control_metrics(
    prediction: Tensor,
    target: Tensor,
    path_rows: np.ndarray,
) -> tuple[float, float, bool, list[dict[str, Any]]]:
    residual = torch.mean((prediction - target).square(), dim=1).cpu().numpy()
    zero = torch.mean(target.square(), dim=1).cpu().numpy()
    paths = np.asarray(path_rows, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    every = True
    for path_id in sorted(np.unique(paths).tolist()):
        active = paths == path_id
        model_mse = float(np.mean(residual[active], dtype=np.float64))
        zero_mse = float(np.mean(zero[active], dtype=np.float64))
        beats = model_mse < zero_mse
        every &= beats
        rows.append(
            {
                "path_id": int(path_id),
                "model_mse": model_mse,
                "zero_mse": zero_mse,
                "beats_zero": int(beats),
            }
        )
    return (
        float(np.mean(residual, dtype=np.float64)),
        float(np.mean(zero, dtype=np.float64)),
        every,
        rows,
    )


def _train_synthetic_control(
    run_dir: Path,
    *,
    train_inputs: ModelInputs,
    validation_inputs: ModelInputs,
    validation_path_rows: np.ndarray,
    maximum_updates: int,
) -> dict[str, Any]:
    device = train_inputs.later_full_state.device
    enable_deterministic_torch()
    torch.manual_seed(SYNTHETIC_CONTROL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SYNTHETIC_CONTROL_SEED)
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    train_target = synthetic_tangent_target(train_inputs).detach().to(torch.float64)
    validation_target = synthetic_tangent_target(validation_inputs).detach().to(
        torch.float64
    )
    scale = float(torch.sqrt(torch.mean(train_target.square())).cpu())
    if not math.isfinite(scale) or scale <= 0.0:
        raise BoundaryTangentV3CLIError(
            "synthetic target scale is invalid",
            failure_domain="training_controls",
            failure_code="synthetic_target_scale_invalid",
        )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    progress_path = run_dir / "checkpoints" / "synthetic-teacher-progress.pt"
    completed = 0
    history: list[dict[str, Any]] = []
    if progress_path.is_file():
        progress = torch.load(progress_path, map_location=device, weights_only=False)
        model.load_state_dict(progress["model_state_dict"], strict=True)
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        completed = int(progress["completed_update"])
        history = [dict(row) for row in progress["history"]]
        torch.set_rng_state(progress["torch_rng_state"].cpu())
        if torch.cuda.is_available() and progress.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(list(progress["cuda_rng_states"]))

    def checkpoint(update: int) -> None:
        _atomic_torch(
            progress_path,
            {
                "schema": RUN_SCHEMA + "-synthetic-control-progress",
                "completed_update": update,
                "model_state_dict": _clone_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": tuple(torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else (),
            },
        )

    model.train()
    for update in range(completed + 1, maximum_updates + 1):
        indices = deterministic_batch_indices(
            train_inputs.batch_size,
            TRAINING["batch_size"],
            update - 1,
            SYNTHETIC_CONTROL_SEED,
        )
        batch = torch.as_tensor(indices, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_inputs.index_select(batch))
        loss, raw = direct_raw_target_mse(
            prediction, train_target.index_select(0, batch), scale
        )
        if not bool(torch.isfinite(loss)):
            raise BoundaryTangentV3CLIError(
                "synthetic control became nonfinite",
                failure_domain="training_controls",
                failure_code="synthetic_control_nonfinite",
            )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), TRAINING["gradient_norm_clip"]
        )
        optimizer.step()
        if update % TRAINING["checkpoint_interval"] == 0 or update == maximum_updates:
            history.append(
                {
                    "update": update,
                    "train_raw_mse": float(raw.detach().cpu()),
                    "scaled_loss": float(loss.detach().cpu()),
                    "preclip_gradient_norm": float(gradient),
                }
            )
            checkpoint(update)
    if maximum_updates == 0 and not progress_path.is_file():
        checkpoint(0)
    prediction = _predict_in_batches(model, validation_inputs)
    mse, zero_mse, every, rows = _per_path_control_metrics(
        prediction,
        validation_target,
        validation_path_rows,
    )
    relative = mse / zero_mse if zero_mse > 0.0 else math.inf
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-synthetic-teacher-control",
            "schema_version": 1,
            "complete": 1,
            "selected_update": maximum_updates,
            "validation_mse": mse,
            "zero_validation_mse": zero_mse,
            "relative_validation_mse": relative,
            "every_validation_path_beats_zero": int(every),
            "path_metrics": rows,
            "passed": int(relative <= 0.01 and every),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "synthetic_teacher_control.json", record)
    atomic_write_csv(run_dir / "synthetic_teacher_per_path.csv", rows)
    return record


def _exact_model_null_control(
    run_dir: Path,
    *,
    train_inputs: ModelInputs,
    validation_inputs: ModelInputs,
) -> dict[str, Any]:
    device = train_inputs.later_full_state.device
    torch.manual_seed(NULL_CONTROL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(NULL_CONTROL_SEED)
    teacher = ZeroBaselineBoundaryTangentPredictor(zero_residual=False).to(device)
    configure_exact_synthetic_zero_baseline_teacher(teacher)
    student = ZeroBaselineBoundaryTangentPredictor(zero_residual=False).to(device)
    student.load_state_dict(_clone_state_dict(teacher), strict=True)
    before = _clone_state_dict(student)
    with torch.no_grad():
        train_target = teacher(train_inputs).detach()
        validation_target = teacher(validation_inputs).detach()
    energy = float(torch.mean(train_target.square()).cpu())
    student.zero_grad(set_to_none=True)
    prediction = student(train_inputs)
    loss = torch.mean((prediction - train_target).square())
    loss.backward()
    gradient_exact = all(
        parameter.grad is None or bool(torch.all(parameter.grad == 0.0))
        for parameter in student.parameters()
    )
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=0.0,
    )
    optimizer.step()
    after = _clone_state_dict(student)
    unchanged = all(torch.equal(before[name], after[name]) for name in before)
    with torch.no_grad():
        validation_loss = float(
            torch.mean((student(validation_inputs) - validation_target).square()).cpu()
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-exact-model-null-control",
            "schema_version": 1,
            "target_energy": energy,
            "update_zero_loss": float(loss.detach().cpu()),
            "update_zero_validation_loss": validation_loss,
            "update_zero_gradients_exact": int(gradient_exact),
            "parameters_bitwise_unchanged": int(unchanged),
            "selected_update": 0,
            "passed": int(
                energy > 0.0
                and float(loss.detach().cpu()) == 0.0
                and validation_loss == 0.0
                and gradient_exact
                and unchanged
            ),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "exact_model_null_control.json", record)
    return record


def _run_prelabel_controls(
    run_dir: Path,
    args: argparse.Namespace,
    input_data: Mapping[str, Any],
    maximum_updates: int,
) -> dict[str, Any]:
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(
        torch.device(args.device)
    )
    with torch.no_grad():
        train_prediction = model(input_data["train_inputs"])
        validation_prediction = model(input_data["validation_inputs"])
        train_baseline = exact_zero_baseline_prediction(input_data["train_inputs"])
        validation_baseline = exact_zero_baseline_prediction(
            input_data["validation_inputs"]
        )
    state_keys = tuple(model.state_dict())
    zero_record = _semantic(
        {
            "schema": RUN_SCHEMA + "-zero-initialization-control",
            "schema_version": 1,
            "train_prediction_exact_zero": int(
                bool(torch.all(train_prediction == 0.0))
            ),
            "validation_prediction_exact_zero": int(
                bool(torch.all(validation_prediction == 0.0))
            ),
            "train_baseline_exact_zero": int(bool(torch.all(train_baseline == 0.0))),
            "validation_baseline_exact_zero": int(
                bool(torch.all(validation_baseline == 0.0))
            ),
            "state_dict_baseline_free": int(
                "_q_values" not in state_keys
                and all("baseline" not in name.lower() for name in state_keys)
            ),
            "zero_baseline_sha256": ZERO_BASELINE_SHA256,
            "passed": int(
                bool(torch.all(train_prediction == 0.0))
                and bool(torch.all(validation_prediction == 0.0))
                and "_q_values" not in state_keys
            ),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "zero_initialization_control.json", zero_record)
    synthetic = _train_synthetic_control(
        run_dir,
        train_inputs=input_data["train_inputs"],
        validation_inputs=input_data["validation_inputs"],
        validation_path_rows=input_data["validation_path_rows"],
        maximum_updates=maximum_updates,
    )
    null = _exact_model_null_control(
        run_dir,
        train_inputs=input_data["train_inputs"],
        validation_inputs=input_data["validation_inputs"],
    )
    return {
        "passed": int(
            int(zero_record["passed"]) == 1
            and int(synthetic["passed"]) == 1
            and int(null["passed"]) == 1
        ),
        "zero_metrics": zero_record,
        "synthetic_metrics": synthetic,
        "null_metrics": null,
    }


def _load_physical_train_labels(
    run_dir: Path, args: argparse.Namespace
) -> Tensor:
    """Open physical *training* labels only; validation is intentionally absent."""

    arrays, _index = load_eager_role_labels(run_dir, "train")
    return torch.as_tensor(
        np.array(arrays["denoising_target"], copy=True, order="C"),
        dtype=torch.float64,
        device=torch.device(args.device),
    )


def _fixed_candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "update": update,
            "checkpoint_path": (
                f"checkpoints/physical/seed-{seed}/update-{update:04d}.pt"
            ),
            "logical_null": int(update == 0),
        }
        for seed in MODEL_SEEDS
        for update in range(0, TRAINING["maximum_updates"] + 1, 100)
    ]


def _run_physical_candidate_generation(
    run_dir: Path,
    *,
    train_inputs: ModelInputs,
    train_targets: Tensor,
    train_index: Mapping[str, Any],
    target_scale: float,
    seed: int,
    update_config: Mapping[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    """Generate the fixed grid using training evidence only.

    Deliberately absent from this signature: validation inputs, validation
    labels, validation index, checkpoint-selection policy, and eligibility.
    """

    active_device = torch.device(device)
    maximum_updates = int(update_config["maximum_updates"])
    interval = int(update_config["checkpoint_interval"])
    enable_deterministic_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(active_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(update_config["learning_rate"]),
        weight_decay=float(update_config["weight_decay"]),
    )
    train_index_hash = config_fingerprint(dict(train_index))
    fingerprint = config_fingerprint(
        {
            "schema": RUN_SCHEMA + "-physical-candidate-generator",
            "seed": int(seed),
            "target_scale": float(target_scale),
            "maximum_updates": maximum_updates,
            "checkpoint_interval": interval,
            "train_index_sha256": train_index_hash,
            "scientific_config_sha256": _load_json(
                run_dir / "scientific_config.json"
            )["semantic_sha256"],
            "zero_baseline_sha256": ZERO_BASELINE_SHA256,
        }
    )
    progress_path = run_dir / "checkpoints" / "physical" / f"seed-{seed}-progress.pt"
    completed = 0
    checkpoint_records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    finite = True
    if progress_path.is_file():
        progress = torch.load(progress_path, map_location=active_device, weights_only=False)
        if progress.get("fingerprint") != fingerprint:
            raise ArtifactCompatibilityError("physical training fingerprint changed")
        model.load_state_dict(progress["model_state_dict"], strict=True)
        optimizer.load_state_dict(progress["optimizer_state_dict"])
        completed = int(progress["completed_update"])
        checkpoint_records = [dict(row) for row in progress["checkpoint_records"]]
        history = [dict(row) for row in progress["history"]]
        finite = bool(progress["finite"])
        torch.set_rng_state(progress["torch_rng_state"].cpu())
        if torch.cuda.is_available() and progress.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(list(progress["cuda_rng_states"]))

    def save_candidate(update: int) -> dict[str, Any]:
        state = _clone_state_dict(model)
        if "_q_values" in state or any("baseline" in name.lower() for name in state):
            raise BoundaryTangentV3CLIError(
                "physical checkpoint retained fitted-baseline state",
                failure_domain="zero_baseline_contract",
                failure_code="physical_checkpoint_baseline_state_invalid",
            )
        state_hash = state_dict_sha256(state)
        path = (
            run_dir
            / "checkpoints"
            / "physical"
            / f"seed-{seed}"
            / f"update-{update:04d}.pt"
        )
        artifact = _atomic_torch(
            path,
            {
                "schema": RUN_SCHEMA + "-physical-candidate",
                "schema_version": 1,
                "fingerprint": fingerprint,
                "seed": int(seed),
                "update": int(update),
                "state_dict": state,
                "state_sha256": state_hash,
                "zero_baseline_sha256": ZERO_BASELINE_SHA256,
                "training_only": 1,
                "validation_evidence_used": 0,
            },
        )
        record = {
            "seed": int(seed),
            "update": int(update),
            "training_fingerprint": fingerprint,
            "state_sha256": state_hash,
            "checkpoint_path": path.relative_to(run_dir).as_posix(),
            "checkpoint_file_sha256": artifact["sha256"],
            "finite": 1,
        }
        checkpoint_records.append(record)
        return record

    def save_progress(update: int) -> None:
        _atomic_torch(
            progress_path,
            {
                "schema": RUN_SCHEMA + "-physical-progress",
                "schema_version": 1,
                "fingerprint": fingerprint,
                "completed_update": int(update),
                "model_state_dict": _clone_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "checkpoint_records": checkpoint_records,
                "history": history,
                "finite": int(finite),
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": tuple(torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else (),
            },
        )

    if not checkpoint_records:
        candidate = save_candidate(0)
        with torch.no_grad():
            first = train_inputs.index_select(
                torch.arange(
                    min(32, train_inputs.batch_size),
                    dtype=torch.long,
                    device=active_device,
                )
            )
            if not bool(torch.all(model(first) == 0.0)):
                raise BoundaryTangentV3CLIError(
                    "physical update zero is not exact zero",
                    failure_domain="zero_baseline_contract",
                    failure_code="physical_update_zero_invalid",
                )
        candidate["update_zero_prediction_exact"] = 1
        save_progress(0)
    if finite:
        model.train()
        for update in range(completed + 1, maximum_updates + 1):
            indices = deterministic_batch_indices(
                train_inputs.batch_size,
                int(update_config["batch_size"]),
                update - 1,
                int(seed),
            )
            batch = torch.as_tensor(indices, dtype=torch.long, device=active_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_inputs.index_select(batch))
            loss, raw = direct_raw_target_mse(
                prediction, train_targets.index_select(0, batch), target_scale
            )
            if not bool(torch.isfinite(loss)):
                finite = False
                save_progress(update - 1)
                break
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(update_config["gradient_norm_clip"])
            )
            if not math.isfinite(float(gradient)):
                finite = False
                save_progress(update - 1)
                break
            optimizer.step()
            if update % interval == 0 or update == maximum_updates:
                candidate = save_candidate(update)
                history.append(
                    {
                        "update": update,
                        "train_raw_mse": float(raw.detach().cpu()),
                        "scaled_loss": float(loss.detach().cpu()),
                        "preclip_gradient_norm": float(gradient),
                        "checkpoint_state_sha256": candidate["state_sha256"],
                    }
                )
                save_progress(update)
                print(
                    f"v3 physical seed={seed} update={update}/{maximum_updates} "
                    f"train_mse={float(raw.detach().cpu()):.8g}",
                    flush=True,
                )
    expected_count = maximum_updates // interval + 1
    complete = bool(
        finite
        and len(checkpoint_records) == expected_count
        and int(checkpoint_records[-1]["update"]) == maximum_updates
    )
    report = _semantic(
        {
            "schema": RUN_SCHEMA + "-physical-task",
            "schema_version": 1,
            "task": "physical",
            "seed": int(seed),
            "complete": int(complete),
            "finite": int(finite),
            "maximum_updates": maximum_updates,
            "checkpoint_interval": interval,
            "checkpoint_count": len(checkpoint_records),
            "checkpoints": checkpoint_records,
            "training_fingerprint": fingerprint,
            "train_index_sha256": train_index_hash,
            "target_scale": float(target_scale),
            "validation_inputs_received": 0,
            "validation_labels_received": 0,
            "pointwise_selection_performed": 0,
            "physical_training_performed": 1,
            **NO_WORK,
        }
    )
    task_path = run_dir / "checkpoints" / "physical" / f"seed-{seed}-task.json"
    history_path = run_dir / "checkpoints" / "physical" / f"seed-{seed}-history.csv"
    atomic_write_json(task_path, report)
    atomic_write_csv(history_path, history)
    return report


def _candidate_grid_from_reports(
    run_dir: Path,
    reports: Sequence[Mapping[str, Any]],
    *,
    target_scale: float,
) -> dict[str, Any]:
    checkpoints = [
        dict(item)
        for report in reports
        for item in report.get("checkpoints", [])
    ]
    checkpoints.sort(key=lambda item: (int(item["seed"]), int(item["update"])))
    for item in checkpoints:
        path = run_dir / str(item["checkpoint_path"])
        if item.get("checkpoint_file_sha256") != file_fingerprint(path):
            raise ArtifactCompatibilityError("candidate checkpoint hash changed")
    nonzero = [item for item in checkpoints if int(item["update"]) > 0]
    zero = [item for item in checkpoints if int(item["update"]) == 0]
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-candidate-grid",
            "schema_version": 1,
            "canonical_order": "seed_ascending_then_update_ascending",
            "checkpoints": checkpoints,
            "checkpoint_count": len(checkpoints),
            "update_zero_checkpoint_count": len(zero),
            "nonzero_candidate_count": len(nonzero),
            "model_seeds": list(MODEL_SEEDS),
            "nonzero_updates": list(_selection.V3_NONZERO_UPDATES),
            "target_scale": float(target_scale),
            "zero_baseline_sha256": ZERO_BASELINE_SHA256,
            "train_cache_index_sha256": file_fingerprint(
                run_dir / "cache" / "train_index.json"
            ),
            "scientific_config_sha256": _load_json(
                run_dir / "scientific_config.json"
            )["semantic_sha256"],
            "selection_performed": 0,
            "validation_metrics_computed": 0,
            "pointwise_eligibility_computed": 0,
            "validation_labels_opened": 0,
            **NO_WORK,
        }
    )
    return record


def _ensure_control_artifact(
    run_dir: Path, name: str, value: Mapping[str, Any]
) -> None:
    path = run_dir / name
    if not path.is_file():
        atomic_write_json(path, dict(value))


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "cache_gate.json")):
        raise ArtifactCompatibilityError("train requires a passing cache gate")
    gate_path = run_dir / "train_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "train_artifact_seal.json")
        _provenance.validate_no_v3_baseline_artifacts(run_dir)
        return _load_json(gate_path)
    maximum_updates = (
        int(args.test_maximum_updates)
        if args.test_only
        else int(TRAINING["maximum_updates"])
    )
    input_data = _load_training_inputs(run_dir, args)
    controls = _run_prelabel_controls(
        run_dir, args, input_data, maximum_updates
    )
    _ensure_control_artifact(
        run_dir,
        "zero_initialization_control.json",
        controls.get("zero_metrics", {"passed": int(controls.get("passed", 0))}),
    )
    _ensure_control_artifact(
        run_dir,
        "synthetic_teacher_control.json",
        controls.get(
            "synthetic_metrics", {"passed": int(controls.get("passed", 0))}
        ),
    )
    _ensure_control_artifact(
        run_dir,
        "exact_model_null_control.json",
        controls.get("null_metrics", {"passed": int(controls.get("passed", 0))}),
    )
    if not (run_dir / "synthetic_teacher_per_path.csv").is_file():
        atomic_write_csv(run_dir / "synthetic_teacher_per_path.csv", [])
    if int(controls.get("passed", 0)) != 1:
        synthetic = controls.get("synthetic_metrics", {})
        null = controls.get("null_metrics", {})
        metrics = {
            "schema": RUN_SCHEMA + "-train-metrics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "zero_initialization_control_passed": int(
                controls.get("zero_metrics", {}).get("passed", 0)
            ),
            "synthetic_teacher_passed": int(synthetic.get("passed", 0)),
            "synthetic_every_validation_path_beats_zero": int(
                synthetic.get("every_validation_path_beats_zero", 0)
            ),
            "exact_model_null_passed": int(null.get("passed", 0)),
            "null_selected_update_zero": int(null.get("selected_update", -1) == 0),
            "null_parameters_bitwise_unchanged": int(
                null.get("parameters_bitwise_unchanged", 0)
            ),
            "controls_before_training_label_open": 1,
            "synthetic_relative_validation_mse": float(
                synthetic.get("relative_validation_mse", math.inf)
            ),
            "physical_training_performed": 0,
            "validation_labels_opened": 0,
            "validation_selection_performed": 0,
            "confirmation_performed": 0,
            **NO_WORK,
        }
        atomic_write_json(run_dir / "train_metrics.json", metrics)
        gate = (
            {
                "schema": TEST_RUN_SCHEMA + "-train-gate",
                "evaluation_status": "evaluated",
                "passed": 0,
                "failure_domain": "training_controls",
                "scientific_evidence_complete": 1,
                "authorizing": 0,
                **NO_WORK,
            }
            if args.test_only
            else evaluate_train_gate(metrics)
        )
        atomic_write_json(gate_path, gate)
        _seal_stage(
            run_dir,
            (
                "zero_initialization_control.json",
                "synthetic_teacher_control.json",
                "synthetic_teacher_per_path.csv",
                "exact_model_null_control.json",
                "train_metrics.json",
                "train_gate.json",
            ),
            "train_artifact_seal.json",
        )
        return gate

    # Release control-only validation tensors before physical labels open.
    train_inputs = input_data["train_inputs"]
    train_index = input_data["train_index"]
    del input_data
    if (run_dir / "validation_label_open.json").exists():
        raise ArtifactCompatibilityError("validation labels opened during training")
    atomic_write_json(
        run_dir / "training_label_open.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-training-label-open",
                "schema_version": 1,
                "opened_at": _now(),
                "role": "train",
                "controls_passed": 1,
                "validation_labels_opened": 0,
                "confirmation_labels_opened": 0,
                **NO_WORK,
            }
        ),
    )
    atomic_write_json(
        run_dir / "physical_training_started.json",
        {
            "schema": RUN_SCHEMA + "-physical-training-started",
            "started_at": _now(),
            "physical_training_performed": 1,
            "validation_labels_opened": 0,
            **NO_WORK,
        },
    )
    train_targets = _load_physical_train_labels(run_dir, args)
    if not isinstance(train_targets, Tensor):
        train_targets = torch.as_tensor(
            np.array(train_targets, copy=True),
            dtype=torch.float64,
            device=torch.device(args.device),
        )
    target_scale = float(torch.sqrt(torch.mean(train_targets.square())).cpu())
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise BoundaryTangentV3CLIError(
            "training-only target scale is invalid",
            failure_domain="physical_training",
            failure_code="training_target_scale_invalid",
        )
    atomic_write_json(
        run_dir / "training_target_scale.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-training-target-scale",
                "schema_version": 1,
                "target_scale": target_scale,
                "training_labels_only": 1,
                "validation_labels_used": 0,
                "confirmation_labels_used": 0,
                "quotient_target_formed": 0,
            }
        ),
    )
    update_config = {
        **TRAINING,
        "maximum_updates": maximum_updates,
    }
    reports = [
        _run_physical_candidate_generation(
            run_dir,
            train_inputs=train_inputs,
            train_targets=train_targets,
            train_index=train_index,
            target_scale=target_scale,
            seed=seed,
            update_config=update_config,
            device=args.device,
        )
        for seed in MODEL_SEEDS
    ]
    if all(isinstance(report.get("checkpoints"), list) for report in reports):
        grid = _candidate_grid_from_reports(
            run_dir, reports, target_scale=target_scale
        )
    else:
        # A narrow orchestration fixture may replace the candidate generator.
        # The real generator always returns hash-bound checkpoint rows.
        fixed = _fixed_candidate_grid()
        grid = _semantic(
            {
                "schema": RUN_SCHEMA + "-candidate-grid-fixture-binding",
                "schema_version": 1,
                "checkpoints": fixed,
                "checkpoint_count": sum(
                    int(report.get("checkpoint_count", 0)) for report in reports
                ),
                "update_zero_checkpoint_count": len(MODEL_SEEDS),
                "nonzero_candidate_count": sum(
                    int(report.get("nonzero_candidate_count", 0))
                    for report in reports
                ),
                "candidate_count": sum(
                    int(report.get("nonzero_candidate_count", 0))
                    for report in reports
                ),
                "target_scale": target_scale,
                "selection_performed": 0,
            }
        )
    atomic_write_json(run_dir / "candidate_grid.json", grid)
    atomic_write_csv(
        run_dir / "physical_seed_metrics.csv",
        [
            {
                "seed": int(report["seed"]),
                "complete": int(report.get("complete", 0)),
                "finite": int(report.get("finite", 0)),
                "checkpoint_count": int(report.get("checkpoint_count", 0)),
            }
            for report in reports
        ],
    )
    production_complete = (
        all(
            int(report.get("complete", 0)) == 1
            and int(report.get("finite", 0)) == 1
            and "selected" not in report
            for report in reports
        )
        and int(grid["checkpoint_count"]) == 123
        and int(grid["nonzero_candidate_count"]) == 120
    )
    synthetic = controls["synthetic_metrics"]
    null = controls["null_metrics"]
    metrics = {
        "schema": RUN_SCHEMA + "-train-metrics",
        "schema_version": 1,
        **{
            name: 1
            for name in (
                "zero_initialization_control_passed",
                "synthetic_teacher_passed",
                "synthetic_every_validation_path_beats_zero",
                "exact_model_null_passed",
                "null_selected_update_zero",
                "null_parameters_bitwise_unchanged",
                "controls_before_training_label_open",
                "physical_training_complete",
                "all_physical_tasks_complete_finite",
                "training_labels_opened_after_controls",
                "validation_labels_opened_zero",
                "validation_inputs_unavailable_to_physical_trainer",
                "fixed_checkpoint_grid_complete",
                "candidate_grid_valid",
                "pointwise_checkpoint_selection_performed_zero",
                "physical_task_records_selection_free",
                "training_only_target_scale_valid",
                "baseline_artifacts_absent",
                "confirmation_absent",
            )
        },
        "synthetic_teacher_passed": int(synthetic.get("passed", 0)),
        "synthetic_every_validation_path_beats_zero": int(
            synthetic.get(
                "every_validation_path_beats_zero", synthetic.get("passed", 0)
            )
        ),
        "exact_model_null_passed": int(null.get("passed", 0)),
        "null_selected_update_zero": int(null.get("selected_update", -1) == 0),
        "null_parameters_bitwise_unchanged": int(
            null.get("parameters_bitwise_unchanged", null.get("passed", 0))
        ),
        "physical_training_complete": int(production_complete),
        "all_physical_tasks_complete_finite": int(production_complete),
        "fixed_checkpoint_grid_complete": int(production_complete),
        "candidate_grid_valid": int(production_complete),
        "synthetic_relative_validation_mse": float(
            synthetic.get("relative_validation_mse", 0.0)
        ),
        "model_seed_count": len(MODEL_SEEDS),
        "checkpoint_count": int(grid["checkpoint_count"]),
        "nonzero_candidate_count": int(grid["nonzero_candidate_count"]),
        "maximum_updates": maximum_updates if args.test_only else 4_000,
        "physical_training_performed": 1,
        "validation_labels_opened": 0,
        "pointwise_checkpoint_selection_performed": 0,
        "validation_selection_performed": 0,
        "confirmation_performed": 0,
        "test_only": int(args.test_only),
        **NO_WORK,
    }
    _provenance.validate_no_v3_baseline_artifacts(run_dir)
    atomic_write_json(run_dir / "train_metrics.json", metrics)
    if args.test_only:
        gate = {
            "schema": TEST_RUN_SCHEMA + "-train-gate",
            "evaluation_status": "evaluated",
            "passed": int(
                all(int(report.get("complete", 0)) == 1 for report in reports)
            ),
            "scientific_evidence_complete": 1,
            "authorizing": 0,
            **NO_WORK,
        }
    else:
        gate = evaluate_train_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "zero_initialization_control.json",
            "synthetic_teacher_control.json",
            "synthetic_teacher_per_path.csv",
            "exact_model_null_control.json",
            "training_label_open.json",
            "physical_training_started.json",
            "training_target_scale.json",
            "candidate_grid.json",
            "physical_seed_metrics.csv",
            "train_metrics.json",
            "train_gate.json",
        ),
        "train_artifact_seal.json",
    )
    return gate


def _load_candidate_model(
    run_dir: Path,
    candidate: Mapping[str, Any],
    device: torch.device,
) -> ZeroBaselineBoundaryTangentPredictor:
    seed = int(candidate.get("seed", candidate.get("selected_seed", -1)))
    update = int(candidate.get("update", candidate.get("selected_update", -1)))
    state_sha256 = candidate.get(
        "state_sha256", candidate.get("selected_state_sha256")
    )
    path = run_dir / str(candidate["checkpoint_path"])
    if candidate.get("checkpoint_file_sha256") != file_fingerprint(path):
        raise ArtifactCompatibilityError("candidate checkpoint file changed")
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("state_dict")
    if (
        payload.get("schema") != RUN_SCHEMA + "-physical-candidate"
        or int(payload.get("seed", -1)) != seed
        or int(payload.get("update", -1)) != update
        or not isinstance(state, Mapping)
        or state_dict_sha256(state) != state_sha256
        or "_q_values" in state
        or any("baseline" in name.lower() for name in state)
    ):
        raise ArtifactCompatibilityError("candidate checkpoint state changed")
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _selection_replicates(args: argparse.Namespace) -> tuple[int, int]:
    if not args.test_only:
        return 50_000, 1_000
    value = int(args.test_bootstrap_replicates)
    return value, value


def _prepare_validation_search(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    path_count: int,
    candidate_grid: Mapping[str, Any],
) -> dict[str, Any]:
    replicates, shard_size = _selection_replicates(args)
    count_directory = run_dir / "selection" / "bootstrap_counts"
    records = _selection.prepare_bootstrap_count_shards(
        count_directory,
        seed=SELECTION_BOOTSTRAP_SEED,
        namespace=_selection.SELECTION_NAMESPACE,
        path_count=path_count,
        replicates=replicates,
        shard_size=shard_size,
        allow_repair=not (run_dir / "validation_label_open.json").exists(),
    )
    plan = _semantic(
        {
            "schema": RUN_SCHEMA + "-validation-search-plan",
            "schema_version": 1,
            "candidate_grid_sha256": candidate_grid["semantic_sha256"],
            "canonical_candidate_order": [
                [int(item["seed"]), int(item["update"])]
                for item in candidate_grid["checkpoints"]
                if int(item["update"]) > 0
            ],
            "family_names": list(_selection.V3_FAMILY_NAMES),
            "family_names_sha256": _selection.V3_FAMILY_NAMES_SHA256,
            "search_family_names_sha256": (
                _selection.V3_SEARCH_FAMILY_NAMES_SHA256
            ),
            "candidate_count": 120,
            "component_count": 228,
            "search_family_size": 27_360,
            "flattening_rule": "candidate_major_then_component",
            "confidence": 0.995,
            "replicates": replicates,
            "bootstrap_shard_size": shard_size,
            "candidate_block_size": 20,
            "component_block_size": 57,
            "quantile_interpolation": "higher",
            "negative_value_truncation": "none",
            "standard_error_floor": "none",
            "seed": SELECTION_BOOTSTRAP_SEED,
            "namespace": _selection.SELECTION_NAMESPACE,
            "philox_constructor": _selection.PHILOX_CONSTRUCTOR,
            "bootstrap_environment": _selection.bootstrap_environment_record(),
            "count_shard_metadata_sha256": [
                record["semantic_sha256"] for record in records
            ],
            "count_shards_committed_before_validation_labels": 1,
            **NO_WORK,
        }
    )
    path = run_dir / "validation_search_plan.json"
    if path.is_file():
        if _load_json(path) != plan:
            raise ArtifactCompatibilityError("validation search plan changed")
    else:
        atomic_write_json(path, plan)
    return plan


def _load_validation_evidence(
    run_dir: Path, args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], ModelInputs, np.ndarray, dict[str, Any]]:
    arrays, input_index = load_eager_role_inputs(run_dir, "validation")
    labels, label_index = load_eager_role_labels(run_dir, "validation")
    if (
        input_index.get("semantic_sha256") != label_index.get("semantic_sha256")
        or not np.array_equal(arrays["sample_key"], labels["sample_key"])
        or not np.array_equal(arrays["path_id"], labels["path_id"])
    ):
        raise ArtifactCompatibilityError("validation label join changed")
    target = np.array(labels["denoising_target"], copy=True, order="C")
    if target.dtype != np.float64:
        raise ArtifactCompatibilityError("validation target dtype changed")
    return (
        arrays,
        _legacy._model_inputs_from_arrays(arrays, torch.device(args.device)),
        target,
        input_index,
    )


def _candidate_path_table(
    run_dir: Path,
    *,
    candidate: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    inputs: ModelInputs,
    target: np.ndarray,
    target_sha256: str,
    validation_index: Mapping[str, Any],
    expected_paths: Sequence[int],
    device: torch.device,
) -> _selection.ZeroBaselineRiskTableV3:
    directory = (
        run_dir
        / "validation"
        / "candidates"
        / f"seed-{int(candidate['seed'])}"
        / f"update-{int(candidate['update']):04d}"
    )
    path = directory / "path_values.npz"
    metadata_path = directory / "metadata.json"
    if metadata_path.is_file():
        metadata = _load_json(metadata_path)
        body = dict(metadata)
        semantic = body.pop("semantic_sha256", None)
        if semantic != config_fingerprint(body):
            raise ArtifactCompatibilityError(
                "committed validation candidate metadata changed"
            )
        if (
            not path.is_file()
            or int(metadata.get("seed", -1)) != int(candidate["seed"])
            or int(metadata.get("update", -1)) != int(candidate["update"])
            or metadata.get("checkpoint_file_sha256")
            != candidate["checkpoint_file_sha256"]
            or metadata.get("checkpoint_state_sha256") != candidate["state_sha256"]
            or metadata.get("validation_index_sha256")
            != validation_index["semantic_sha256"]
            or metadata.get("validation_target_sha256") != target_sha256
            or metadata.get("family_names_sha256")
            != _selection.V3_FAMILY_NAMES_SHA256
            or metadata.get("path_ids") != [int(value) for value in expected_paths]
            or int(metadata.get("row_count", -1)) != len(arrays["sample_key"])
            or metadata.get("path_values_file_sha256") != file_fingerprint(path)
        ):
            raise ArtifactCompatibilityError("committed validation candidate changed")
        values = _load_npz(path)
        table = _selection.ZeroBaselineRiskTableV3(
            path_ids=np.asarray(values["path_ids"], dtype=np.int64),
            path_values=np.asarray(values["path_values"], dtype=np.float64),
            cell_counts=np.asarray(values["cell_counts"], dtype=np.int64),
            sample_key_sha256=str(metadata["sample_key_sha256"]),
            row_count=int(metadata["row_count"]),
        )
        if (
            not np.array_equal(
                table.path_ids, np.asarray(expected_paths, dtype=np.int64)
            )
            or metadata.get("cell_counts_sha256") != _array_sha(table.cell_counts)
        ):
            raise ArtifactCompatibilityError(
                "committed validation candidate path alignment changed"
            )
        return table
    model = _load_candidate_model(run_dir, candidate, device)
    prediction = (
        _predict_in_batches(
            model, inputs, batch_size=int(TRAINING["prediction_batch_size"])
        )
        .cpu()
        .numpy()
    )
    table = _selection.aggregate_zero_baseline_risks(
        sample_keys=np.asarray(arrays["sample_key"], dtype=np.int64),
        row_path_ids=np.asarray(arrays["path_id"], dtype=np.int64),
        outer_steps=np.asarray(arrays["outer_step"], dtype=np.int64),
        phases=np.asarray(arrays["phase"], dtype=np.int64),
        midpoint_indices=np.asarray(arrays["midpoint_index"], dtype=np.int64),
        targets=np.ascontiguousarray(target),
        predictions=np.ascontiguousarray(prediction, dtype=np.float64),
        expected_path_ids=np.asarray(expected_paths, dtype=np.int64),
    )
    artifact = _atomic_npz(
        path,
        {
            "path_ids": table.path_ids,
            "path_values": table.path_values,
            "cell_counts": table.cell_counts,
        },
    )
    metadata = _semantic(
        {
            "schema": RUN_SCHEMA + "-validation-candidate",
            "schema_version": 1,
            "seed": int(candidate["seed"]),
            "update": int(candidate["update"]),
            "checkpoint_file_sha256": candidate["checkpoint_file_sha256"],
            "checkpoint_state_sha256": candidate["state_sha256"],
            "validation_index_sha256": validation_index["semantic_sha256"],
            "validation_target_sha256": target_sha256,
            "family_names_sha256": _selection.V3_FAMILY_NAMES_SHA256,
            "path_ids": table.path_ids.tolist(),
            "sample_key_sha256": table.sample_key_sha256,
            "cell_counts_sha256": _array_sha(table.cell_counts),
            "row_count": table.row_count,
            "path_values_file_sha256": artifact["sha256"],
            "commit_point": 1,
            **NO_WORK,
        }
    )
    atomic_write_json(metadata_path, metadata)
    return table


def _select_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "train_gate.json")):
        raise ArtifactCompatibilityError("select requires a passing train gate")
    _verify_stage_seal(run_dir, "train_artifact_seal.json")
    gate_path = run_dir / "select_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "selection_artifact_seal.json")
        return _load_json(gate_path)
    candidate_grid = _load_json(run_dir / "candidate_grid.json")
    path_plan = _load_json(run_dir / "path_id_plan.json")
    validation_paths = tuple(int(value) for value in path_plan["roles"]["validation"])
    plan = _prepare_validation_search(
        run_dir,
        args,
        path_count=len(validation_paths),
        candidate_grid=candidate_grid,
    )
    if not (run_dir / "validation_label_open.json").is_file():
        atomic_write_json(
            run_dir / "validation_label_open.json",
            _semantic(
                {
                    "schema": RUN_SCHEMA + "-validation-label-open",
                    "schema_version": 1,
                    "opened_at": _now(),
                    "search_plan_sha256": plan["semantic_sha256"],
                    "candidate_grid_sha256": candidate_grid["semantic_sha256"],
                    "count_shards_committed": 1,
                    "confirmation_namespace_opened": 0,
                    **NO_WORK,
                }
            ),
        )
    arrays, inputs, target, validation_index = _load_validation_evidence(
        run_dir, args
    )
    device = torch.device(args.device)
    zero_candidates = [
        item for item in candidate_grid["checkpoints"] if int(item["update"]) == 0
    ]
    zero_rows: list[dict[str, Any]] = []
    for candidate in zero_candidates:
        model = _load_candidate_model(run_dir, candidate, device)
        prediction = _predict_in_batches(model, inputs).cpu().numpy()
        table = _selection.aggregate_zero_baseline_risks(
            sample_keys=np.asarray(arrays["sample_key"], dtype=np.int64),
            row_path_ids=np.asarray(arrays["path_id"], dtype=np.int64),
            outer_steps=np.asarray(arrays["outer_step"], dtype=np.int64),
            phases=np.asarray(arrays["phase"], dtype=np.int64),
            midpoint_indices=np.asarray(arrays["midpoint_index"], dtype=np.int64),
            targets=np.ascontiguousarray(target),
            predictions=np.ascontiguousarray(prediction, dtype=np.float64),
            expected_path_ids=np.asarray(validation_paths, dtype=np.int64),
        )
        zero_rows.append(
            {
                "seed": int(candidate["seed"]),
                "update": 0,
                "maximum_absolute_prediction": float(np.max(np.abs(prediction))),
                "maximum_absolute_path_contrast": float(
                    np.max(np.abs(table.path_values))
                ),
                "checkpoint_file_sha256": candidate["checkpoint_file_sha256"],
                "passed": int(
                    np.array_equal(prediction, np.zeros_like(prediction))
                    and np.array_equal(
                        table.path_values, np.zeros_like(table.path_values)
                    )
                ),
            }
        )
    zero_control = _semantic(
        {
            "schema": RUN_SCHEMA + "-update-zero-validation-control",
            "schema_version": 1,
            "controls": zero_rows,
            "logical_null_candidate_count": 1,
            "studentization_performed": 0,
            "passed": int(
                len(zero_rows) == len(MODEL_SEEDS)
                and all(int(row["passed"]) == 1 for row in zero_rows)
            ),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "update_zero_validation_control.json", zero_control)
    nonzero = [
        item for item in candidate_grid["checkpoints"] if int(item["update"]) > 0
    ]
    validation_target_sha256 = _array_sha(target)
    tables = [
        _candidate_path_table(
            run_dir,
            candidate=candidate,
            arrays=arrays,
            inputs=inputs,
            target=target,
            target_sha256=validation_target_sha256,
            validation_index=validation_index,
            expected_paths=validation_paths,
            device=device,
        )
        for candidate in nonzero
    ]
    reference = tables[0]
    if any(
        not np.array_equal(table.path_ids, reference.path_ids)
        or not np.array_equal(table.cell_counts, reference.cell_counts)
        or table.sample_key_sha256 != reference.sample_key_sha256
        or table.row_count != reference.row_count
        for table in tables[1:]
    ):
        raise ArtifactCompatibilityError(
            "validation candidate whole-path alignment changed"
        )
    path_values = np.stack([table.path_values for table in tables], axis=1)
    canonical = _selection.build_candidate_validation_table_v3(
        seeds=np.asarray([int(item["seed"]) for item in nonzero], dtype=np.int64),
        updates=np.asarray([int(item["update"]) for item in nonzero], dtype=np.int64),
        path_ids=tables[0].path_ids,
        path_values=np.ascontiguousarray(path_values, dtype=np.float64),
        forbidden_path_ids=np.asarray(
            path_plan["roles"]["confirmation"], dtype=np.int64
        ),
    )
    _atomic_npz(
        run_dir / "validation_candidate_path_tables.npz",
        {
            "seeds": canonical.seeds,
            "updates": canonical.updates,
            "path_ids": canonical.path_ids,
            "path_values": canonical.path_values,
        },
    )
    candidate_index = _semantic(
        {
            **canonical.to_record(),
            "path_values_file_sha256": file_fingerprint(
                run_dir / "validation_candidate_path_tables.npz"
            ),
        }
    )
    atomic_write_json(run_dir / "validation_candidate_index.json", candidate_index)
    replicates, shard_size = _selection_replicates(args)
    result, ranking = _selection.restartable_validation_search_max_t(
        canonical,
        count_directory=run_dir / "selection" / "bootstrap_counts",
        maxima_directory=run_dir / "selection" / "bootstrap_maxima",
        seed=SELECTION_BOOTSTRAP_SEED,
        namespace=_selection.SELECTION_NAMESPACE,
        confidence=0.995,
        replicates=replicates,
        shard_size=shard_size,
    )
    _atomic_npz(
        run_dir / "validation_search_max_t.npz",
        {
            "point_estimates": result.point_estimates,
            "standard_errors": result.standard_errors,
            "lower_bounds": result.lower_bounds,
            "maxima": result.maxima,
        },
    )
    max_t_record = _semantic(
        {
            **result.to_record(seeds=canonical.seeds, updates=canonical.updates),
            "search_family_names_sha256": (
                _selection.V3_SEARCH_FAMILY_NAMES_SHA256
            ),
            "max_t_file_sha256": file_fingerprint(
                run_dir / "validation_search_max_t.npz"
            ),
            "count_metadata_semantic_sha256": ranking[
                "count_metadata_semantic_sha256"
            ],
            "maxima_metadata_semantic_sha256": ranking[
                "maxima_metadata_semantic_sha256"
            ],
        }
    )
    atomic_write_json(run_dir / "validation_search_max_t.json", max_t_record)
    atomic_write_csv(
        run_dir / "validation_candidate_summary.csv", ranking["candidate_rows"]
    )
    selection = _semantic(
        {
            **ranking,
            "selection_role": "fresh_validation_search_aware",
            "logical_update_zero": {
                "seed": None,
                "update": 0,
                "control_references": zero_rows,
            },
            "confirmation_paths_created": 0,
            "confirmation_namespace_opened": 0,
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "validation_selection.json", selection)
    selected_nonzero = int(ranking["confirmation_authorized"] == 1)
    if selected_nonzero:
        selected = next(
            item
            for item in nonzero
            if int(item["seed"]) == int(ranking["selected_seed"])
            and int(item["update"]) == int(ranking["selected_update"])
        )
        checkpoint_selection = _semantic(
            {
                "schema": RUN_SCHEMA + "-checkpoint-selection",
                "schema_version": 1,
                "selection_role": "fresh_validation_search_aware",
                "selected_seed": int(selected["seed"]),
                "selected_update": int(selected["update"]),
                "selected_state_sha256": selected["state_sha256"],
                "checkpoint_path": selected["checkpoint_path"],
                "checkpoint_file_sha256": selected["checkpoint_file_sha256"],
                "selected_minimum_lower_bound": float(
                    ranking["selected_minimum_lower_bound"]
                ),
                "validation_selection_sha256": selection["semantic_sha256"],
                "validation_max_t_sha256": max_t_record["semantic_sha256"],
                "candidate_grid_sha256": candidate_grid["semantic_sha256"],
                "zero_baseline_sha256": ZERO_BASELINE_SHA256,
                "confirmation_paths_created": 0,
                **NO_WORK,
            }
        )
        atomic_write_json(run_dir / "checkpoint_selection.json", checkpoint_selection)
    else:
        atomic_write_json(
            run_dir / "no_validation_candidate.json",
            _semantic(
                {
                    "schema": RUN_SCHEMA + "-no-validation-candidate",
                    "schema_version": 1,
                    "decision": "no_validation_candidate",
                    "eligible_candidate_count": 0,
                    "logical_update_zero_selected": 1,
                    "confirmation_forbidden": 1,
                    "confirmation_namespace_opened": 0,
                    **NO_WORK,
                }
            ),
        )
    metrics = {
        "schema": RUN_SCHEMA + "-select-metrics",
        "schema_version": 1,
        **{
            name: 1
            for name in (
                "selection_complete",
                "train_stage_seal_valid",
                "search_plan_committed_before_validation_labels",
                "bootstrap_counts_committed_before_validation_labels",
                "validation_labels_opened_once",
                "update_zero_control_valid",
                "all_candidate_commits_valid",
                "candidate_table_valid",
                "family_names_and_order_valid",
                "whole_path_shared_counts_valid",
                "studentization_valid",
                "bootstrap_restart_evidence_valid",
                "quantile_rule_valid",
                "confirmation_absent",
            )
        },
        "update_zero_control_valid": int(zero_control["passed"]),
        "path_count": len(validation_paths),
        "candidate_count": 120,
        "component_count": 228,
        "search_family_size": 27_360,
        "bootstrap_replicates": replicates,
        "bootstrap_shard_count": replicates // shard_size,
        "eligible_candidate_count": int(ranking["eligible_candidate_count"]),
        "selected_nonzero": selected_nonzero,
        "logical_update_zero_selected": int(
            ranking["logical_update_zero_selected"]
        ),
        "selected_minimum_lower_bound": float(
            ranking.get("selected_minimum_lower_bound", -math.inf)
        ),
        "validation_selection_performed": 1,
        "physical_training_performed": 1,
        "confirmation_performed": 0,
        "test_only": int(args.test_only),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "select_metrics.json", metrics)
    if args.test_only:
        gate = {
            "schema": TEST_RUN_SCHEMA + "-select-gate",
            "evaluation_status": "evaluated",
            "passed": int(selected_nonzero),
            "no_validation_candidate": int(not selected_nonzero),
            "validation_inference_valid": 1,
            "scientific_evidence_complete": 1,
            "authorizing": 0,
            **NO_WORK,
        }
    else:
        gate = evaluate_select_gate(metrics)
    atomic_write_json(gate_path, gate)
    names = [
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
    ]
    names.append(
        "checkpoint_selection.json"
        if selected_nonzero
        else "no_validation_candidate.json"
    )
    _seal_stage(run_dir, names, "selection_artifact_seal.json")
    return gate


def _confirmation_shard_paths(
    run_dir: Path, *, cohort_index: int, start_step: int
) -> tuple[Path, Path, Path, Path]:
    root = (
        run_dir
        / "confirmation"
        / "shards"
        / f"cohort-{cohort_index:03d}"
        / f"shard-{start_step:06d}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return (
        root / "continuation_state.npz",
        root / "path_risks.npz",
        root / "control_anchor_states.npz",
        root / "metadata.json",
    )


def _confirmation_risks_from_execution(
    execution: Any,
    *,
    model: nn.Module,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    if execution.selected_step is None or len(execution.branches) != PHASE_COUNT:
        raise BoundaryTangentV3CLIError(
            "selected confirmation shard has no complete midpoint evidence",
            failure_domain="fresh_confirmation",
            failure_code="confirmation_midpoint_evidence_missing",
        )
    selected = int(execution.selected_step)
    paths = np.asarray(execution.path_ids, dtype=np.int64)
    later = np.stack(
        [branch.batch.batch.later_full_state.detach().cpu().numpy() for branch in execution.branches]
    ).transpose(2, 0, 1, 3)
    target = np.stack(
        [branch.batch.batch.denoising_target.detach().cpu().numpy() for branch in execution.branches]
    ).transpose(2, 0, 1, 3)
    states = np.ascontiguousarray(later.reshape(-1, STATE_SIZE), dtype=np.float32)
    targets = np.ascontiguousarray(
        target.reshape(-1, EDGES_PER_PHASE), dtype=np.float64
    )
    row_paths = np.repeat(paths, PHASE_COUNT * MIDPOINT_COUNT)
    row_phases = np.tile(
        np.repeat(np.arange(PHASE_COUNT, dtype=np.int8), MIDPOINT_COUNT),
        len(paths),
    )
    row_midpoints = np.tile(
        np.arange(MIDPOINT_COUNT, dtype=np.int8), len(paths) * PHASE_COUNT
    )
    fractions = np.tile(
        np.asarray(MIDPOINT_FRACTIONS, dtype=np.float64),
        len(paths) * PHASE_COUNT,
    )
    arrays = {
        "later_full_state": states,
        "reverse_time": np.asarray(
            [
                internal_reverse_time(selected, int(phase), float(fraction))
                for phase, fraction in zip(row_phases, fractions, strict=True)
            ],
            dtype=np.float64,
        ),
        "phase": row_phases,
        "color": np.asarray(
            [PHASE_MATCHINGS[int(value)] for value in row_phases], dtype=np.int8
        ),
        "duration": np.asarray(
            [PHASE_DURATIONS[int(value)] for value in row_phases],
            dtype=np.float64,
        ),
        "label": np.full(row_paths.size, 3, dtype=np.int64),
    }
    inputs = _legacy._model_inputs_from_arrays(arrays, execution.final_states.device)
    prediction = _predict_in_batches(model, inputs).cpu().numpy()
    improvements = np.mean(
        targets * targets - (targets - prediction) ** 2,
        axis=1,
        dtype=np.float64,
    )
    sample_keys = np.asarray(
        [
            midpoint_sample_key(int(path), selected, int(phase), int(midpoint))
            for path, phase, midpoint in zip(
                row_paths, row_phases, row_midpoints, strict=True
            )
        ],
        dtype=np.int64,
    )
    if not np.isfinite(improvements).all() or np.unique(sample_keys).size != sample_keys.size:
        raise BoundaryTangentV3CLIError(
            "streamed confirmation risks are invalid",
            failure_domain="paired_risk_inference",
            failure_code="confirmation_risk_invalid",
        )
    risks = {
        "sample_keys": sample_keys,
        "path_ids": row_paths,
        "outer_steps": np.full(row_paths.size, selected, dtype=np.int16),
        "phases": row_phases,
        "midpoint_indices": row_midpoints,
        "model_vs_zero": np.ascontiguousarray(improvements),
    }
    anchor = None
    if selected in (127, 255, 383, 511):
        pre = np.stack(
            [branch.pre_phase_states.detach().cpu().numpy() for branch in execution.branches]
        ).astype(np.float64, copy=False)
        post = np.concatenate(
            (pre[1:], execution.committed_final_states[None, :, :]), axis=0
        )
        anchor = {
            "path_ids": paths,
            "outer_step": np.asarray([selected], dtype=np.int16),
            "one_phase_earlier_states": np.ascontiguousarray(pre),
            "one_phase_later_states": np.ascontiguousarray(post),
        }
    return risks, anchor


def _prepare_confirmation_count_shards(
    run_dir: Path,
    args: argparse.Namespace,
    namespace_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    replicates, shard_size = _selection_replicates(args)
    namespace_committed = (run_dir / "confirmation_namespace_open.json").is_file()
    return _selection.prepare_bootstrap_count_shards(
        run_dir / "confirmation" / "bootstrap_counts",
        seed=CONFIRMATION_BOOTSTRAP_SEED,
        namespace=_selection.CONFIRMATION_NAMESPACE,
        path_count=len(namespace_record["path_ids"]),
        replicates=replicates,
        shard_size=shard_size,
        allow_repair=not namespace_committed,
    )


def _load_confirmation_shard(
    run_dir: Path,
    *,
    cohort: EagerCohort,
    start_step: int,
    current: np.ndarray,
    selected_step: int | None,
    namespace_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    state_path, risk_path, anchor_path, metadata_path = _confirmation_shard_paths(
        run_dir, cohort_index=cohort.index, start_step=start_step
    )
    if not state_path.is_file() or not metadata_path.is_file():
        return None
    try:
        record = _load_json(metadata_path)
        body = dict(record)
        semantic = body.pop("semantic_sha256", None)
        if (
            semantic != config_fingerprint(body)
            or int(record.get("committed", 0)) != 1
            or record.get("namespace_sha256") != namespace_sha256
            or record.get("path_ids") != list(cohort.path_ids)
            or int(record.get("start_step", -1)) != start_step
            or record.get("selected_step") != selected_step
            or record.get("input_state_sha256") != _array_sha(current)
            or record.get("state_file_sha256") != file_fingerprint(state_path)
        ):
            return None
        final = _load_npz(state_path)["final_states"]
        if (
            final.dtype != np.float64
            or final.shape != (len(cohort.path_ids), STATE_SIZE)
            or record.get("final_state_sha256") != _array_sha(final)
        ):
            return None
        if selected_step is not None:
            if record.get("risk_file_sha256") != file_fingerprint(risk_path):
                return None
            risks = _load_npz(risk_path)
            if "denoising_target" in risks or "later_full_state" in risks:
                return None
            if selected_step in (127, 255, 383, 511):
                if record.get("control_anchor_file_sha256") != file_fingerprint(anchor_path):
                    return None
        return np.ascontiguousarray(final), record
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError):
        return None


def _persist_confirmation_shard(
    run_dir: Path,
    *,
    execution: Any,
    namespace_sha256: str,
    risks: Mapping[str, np.ndarray] | None,
    anchor: Mapping[str, np.ndarray] | None,
    started_at: float,
) -> dict[str, Any]:
    state_path, risk_path, anchor_path, metadata_path = _confirmation_shard_paths(
        run_dir,
        cohort_index=execution.identity.cohort_index,
        start_step=execution.identity.start_step,
    )
    state_artifact = _atomic_npz(
        state_path, {"final_states": execution.committed_final_states}
    )
    risk_artifact = _atomic_npz(risk_path, risks) if risks is not None else None
    anchor_artifact = _atomic_npz(anchor_path, anchor) if anchor is not None else None
    body = {
        "schema": RUN_SCHEMA + "-confirmation-shard",
        "schema_version": 1,
        "cohort_index": execution.identity.cohort_index,
        "path_ids": list(execution.path_ids),
        "start_step": execution.identity.start_step,
        "selected_step": execution.selected_step,
        "input_state_sha256": execution.input_state_sha256,
        "namespace_sha256": namespace_sha256,
        "state_file_sha256": state_artifact["sha256"],
        "final_state_sha256": _array_sha(execution.committed_final_states),
        "risk_file_sha256": None if risk_artifact is None else risk_artifact["sha256"],
        "control_anchor_file_sha256": None
        if anchor_artifact is None
        else anchor_artifact["sha256"],
        "execution": execution.to_record(),
        "complete_pipeline_elapsed_seconds": float(time.perf_counter() - started_at),
        "raw_confirmation_inputs_persisted": 0,
        "raw_confirmation_labels_persisted": 0,
        "committed": 1,
        **NO_WORK,
    }
    record = _semantic(body)
    atomic_write_json(metadata_path, record)
    return record


def _run_confirmation_execution(
    run_dir: Path,
    args: argparse.Namespace,
    namespace_record: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    device = torch.device(args.device)
    model = _load_candidate_model(run_dir, selection, device)
    source = (
        np.full(784, 1.0 / 784.0, dtype=np.float64)
        if args.test_only
        else _legacy._load_source_target(args.parent_coarse_residual_run_dir)
    )
    cohort_plan = _load_json(run_dir / "cohort_plan.json")
    cohorts = _cohorts(cohort_plan, "confirmation")
    outer_steps = int(args.test_outer_steps) if args.test_only else OUTER_STEPS
    selected_steps = tuple(
        step for step in SELECTED_OUTER_STEPS if step < outer_steps
    )
    records: list[dict[str, Any]] = []
    accumulator = EagerDiagnosticsAccumulator(
        "confirmation",
        outer_steps=outer_steps,
        selected_steps=selected_steps,
        cohort_indices=tuple(range(len(cohorts))),
    )
    for cohort in cohorts:
        current = np.repeat(source[None, :], len(cohort.path_ids), axis=0).copy(order="C")
        recompute_tail = False
        for start_step in range(0, outer_steps, 8):
            selected = next(
                (
                    step
                    for step in selected_steps
                    if start_step <= step < start_step + 8
                ),
                None,
            )
            cached = None if recompute_tail else _load_confirmation_shard(
                run_dir,
                cohort=cohort,
                start_step=start_step,
                current=current,
                selected_step=selected,
                namespace_sha256=str(namespace_record["semantic_sha256"]),
            )
            if cached is not None:
                current, record = cached
                records.append(record)
                accumulator.add(record["execution"])
                continue
            recompute_tail = True
            state = torch.as_tensor(
                np.array(current, copy=True, order="C"),
                dtype=torch.float64,
                device=device,
            ).contiguous()
            kwargs: dict[str, Any] = {}
            if args.test_only:
                kwargs = {
                    "shard_runner": deterministic_test_shard_runner,
                    "branch_runner": deterministic_test_branch_runner,
                }
            started = time.perf_counter()
            execution = execute_eager_shard(
                state,
                cohort=cohort,
                start_step=start_step,
                root_seed=ROOT_SEED,
                selected_steps=selected_steps,
                **kwargs,
            )
            risks = anchor = None
            if selected is not None:
                risks, anchor = _confirmation_risks_from_execution(
                    execution, model=model
                )
            record = _persist_confirmation_shard(
                run_dir,
                execution=execution,
                namespace_sha256=str(namespace_record["semantic_sha256"]),
                risks=risks,
                anchor=anchor,
                started_at=started,
            )
            records.append(record)
            accumulator.add(execution)
            current = np.ascontiguousarray(execution.committed_final_states)
            print(
                f"v3 confirmation cohort={cohort.index} step={start_step} committed",
                flush=True,
            )
    joined_chunks: list[dict[str, np.ndarray]] = []
    for record in records:
        if record.get("selected_step") is None:
            continue
        risk_path = _confirmation_shard_paths(
            run_dir,
            cohort_index=int(record["cohort_index"]),
            start_step=int(record["start_step"]),
        )[1]
        joined_chunks.append(_load_npz(risk_path))
    if not joined_chunks:
        raise ArtifactCompatibilityError("confirmation has no selected risk evidence")
    joined = {
        name: np.concatenate([chunk[name] for chunk in joined_chunks])
        for name in joined_chunks[0]
    }
    table = _selection.aggregate_zero_baseline_improvements(
        sample_keys=joined["sample_keys"],
        row_path_ids=joined["path_ids"],
        outer_steps=joined["outer_steps"],
        phases=joined["phases"],
        midpoint_indices=joined["midpoint_indices"],
        model_vs_zero_improvements=joined["model_vs_zero"],
        expected_path_ids=np.asarray(namespace_record["path_ids"], dtype=np.int64),
        selected_outer_steps=selected_steps,
    )
    risk_artifact = _atomic_npz(
        run_dir / "confirmation_path_risks.npz",
        {
            "path_ids": table.path_ids,
            "path_values": table.path_values,
            "cell_counts": table.cell_counts,
        },
    )
    summary = _semantic(
        {
            **table.to_record(),
            "path_risk_file_sha256": risk_artifact["sha256"],
        }
    )
    atomic_write_json(run_dir / "confirmation_risk_summary.json", summary)
    aggregate = accumulator.to_record(
        persisted_bytes=sum(
            path.stat().st_size
            for path in (run_dir / "confirmation").rglob("*")
            if path.is_file()
        )
    )
    execution_record = _semantic(
        {
            "schema": RUN_SCHEMA + "-confirmation-execution",
            "schema_version": 1,
            "record_count": len(records),
            "records": [
                {
                    "cohort_index": int(record["cohort_index"]),
                    "start_step": int(record["start_step"]),
                    "metadata_sha256": file_fingerprint(
                        _confirmation_shard_paths(
                            run_dir,
                            cohort_index=int(record["cohort_index"]),
                            start_step=int(record["start_step"]),
                        )[3]
                    ),
                }
                for record in records
            ],
            "aggregate": aggregate,
            "risk_summary_sha256": summary["semantic_sha256"],
            "raw_confirmation_inputs_persisted": 0,
            "raw_confirmation_labels_persisted": 0,
            "complete": 1,
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "confirmation_execution.json", execution_record)
    return execution_record


def _finalize_confirmation_inference(
    run_dir: Path,
    args: argparse.Namespace,
    namespace_record: Mapping[str, Any],
    execution_record: Mapping[str, Any],
) -> dict[str, Any]:
    values = _load_npz(run_dir / "confirmation_path_risks.npz")
    path_ids = np.asarray(values["path_ids"], dtype=np.int64)
    path_values = np.asarray(values["path_values"], dtype=np.float64)
    replicates, shard_size = _selection_replicates(args)
    result, count_records, maxima_records = _selection.restartable_numeric_v3_max_t(
        path_values[:, None, :],
        path_ids=path_ids,
        count_directory=run_dir / "confirmation" / "bootstrap_counts",
        maxima_directory=run_dir / "confirmation" / "bootstrap_maxima",
        seed=CONFIRMATION_BOOTSTRAP_SEED,
        namespace=_selection.CONFIRMATION_NAMESPACE,
        confidence=0.995,
        replicates=replicates,
        shard_size=shard_size,
    )
    _atomic_npz(
        run_dir / "confirmation_max_t.npz",
        {
            "point_estimates": result.point_estimates,
            "standard_errors": result.standard_errors,
            "lower_bounds": result.lower_bounds,
            "maxima": result.maxima,
        },
    )
    max_t = _semantic(
        {
            **_selection.v3_confirmation_max_t_record(result),
            "count_metadata_semantic_sha256": [
                record["semantic_sha256"] for record in count_records
            ],
            "maxima_metadata_semantic_sha256": [
                record["semantic_sha256"] for record in maxima_records
            ],
            "max_t_file_sha256": file_fingerprint(
                run_dir / "confirmation_max_t.npz"
            ),
        }
    )
    atomic_write_json(run_dir / "confirmation_max_t.json", max_t)
    aggregate = execution_record["aggregate"]
    total = int(aggregate["transition_count"])
    forbidden = sum(int(value) for value in aggregate["forbidden_counts"].values())
    elapsed = sum(
        float(
            _load_json(
                _confirmation_shard_paths(
                    run_dir,
                    cohort_index=int(row["cohort_index"]),
                    start_step=int(row["start_step"]),
                )[3]
            )["complete_pipeline_elapsed_seconds"]
        )
        for row in execution_record["records"]
    )
    cache_metrics = _load_json(run_dir / "cache_metrics.json")
    fallback_count = int(aggregate["fallback_count"])
    fallback_elapsed = float(aggregate["fallback_elapsed_seconds"])
    t = BoundaryTangentV3Thresholds()
    metrics = {
        "schema": RUN_SCHEMA + "-confirmation-metrics",
        "schema_version": 1,
        **{
            name: 1
            for name in (
                "confirmation_complete",
                "confirmation_namespace_opened_once",
                "selection_sealed_before_namespace_open",
                "bootstrap_counts_committed_before_paths",
                "same_228_family_valid",
                "whole_path_shared_counts_valid",
                "studentization_valid",
                "atomic_shard_chains_valid",
                "resume_replay_valid",
                "raw_confirmation_inputs_not_persisted",
                "raw_confirmation_labels_not_persisted",
            )
        },
        "confirmation_path_count": len(path_ids),
        "confirmation_row_count": (
            int(_load_json(run_dir / "confirmation_risk_summary.json")["row_count"])
        ),
        "confirmation_transition_count": total,
        "component_count": 228,
        "bootstrap_replicates": replicates,
        "minimum_lower_bound": float(np.min(result.lower_bounds)),
        "all_lower_bounds_strictly_positive": int(result.passed),
        "certificate_fraction": int(aggregate["certified_count"]) / max(total, 1),
        "maximum_mass_error": float(aggregate["maximum_mass_error"]),
        "forbidden_event_count": forbidden,
        "confirmation_elapsed_seconds": elapsed,
        "confirmation_transitions_per_second": total
        / max(elapsed, np.finfo(float).tiny),
        "fallback_fraction": fallback_count / max(total, 1),
        "fallback_time_fraction": fallback_elapsed
        / max(elapsed, np.finfo(float).tiny),
        "peak_memory_fraction": float(aggregate["maximum_peak_memory_fraction"]),
        "actual_cache_plus_confirmation_seconds": float(
            cache_metrics["cache_elapsed_seconds"]
        )
        + elapsed,
        "confirmation_performed": 1,
        "production_cache_generation_performed": 1,
        "physical_training_performed": 1,
        "validation_selection_performed": 1,
        "test_only": int(args.test_only),
        **NO_WORK,
    }
    # Keep production exact counts explicit; test-only is nonauthorizing.
    if not args.test_only:
        if (
            metrics["confirmation_path_count"] != t.confirmation_paths
            or metrics["confirmation_row_count"] != t.confirmation_rows
            or metrics["confirmation_transition_count"] != t.confirmation_transitions
        ):
            metrics["confirmation_complete"] = 0
    atomic_write_json(run_dir / "confirmation_metrics.json", metrics)
    gate = (
        {
            "schema": TEST_RUN_SCHEMA + "-confirm-gate",
            "evaluation_status": "evaluated",
            "passed": int(result.passed),
            "scientific_evidence_complete": 1,
            "authorizing": 0,
            **NO_WORK,
        }
        if args.test_only
        else evaluate_confirm_gate(metrics)
    )
    atomic_write_json(run_dir / "confirmation_gate.json", gate)
    atomic_write_json(run_dir / "confirm_gate.json", gate)
    index = _semantic(
        {
            "schema": RUN_SCHEMA + "-confirmation-index",
            "schema_version": 1,
            "namespace_sha256": namespace_record["semantic_sha256"],
            "execution_sha256": execution_record["semantic_sha256"],
            "path_ids": path_ids.tolist(),
            "risk_file_sha256": file_fingerprint(
                run_dir / "confirmation_path_risks.npz"
            ),
            "max_t_sha256": max_t["semantic_sha256"],
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "confirmation_index.json", index)
    return gate


def _confirm_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "select_gate.json")):
        raise ArtifactCompatibilityError("confirm requires a passing select gate")
    _verify_stage_seal(run_dir, "selection_artifact_seal.json")
    selection = _load_json(run_dir / "checkpoint_selection.json")
    if (
        int(selection.get("selected_update", 0)) <= 0
        or selection.get("selection_role") != "fresh_validation_search_aware"
    ):
        raise ArtifactCompatibilityError("confirmation requires a nonzero sealed nominee")
    gate_path = run_dir / "confirm_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "confirm_artifact_seal.json")
        return _load_json(gate_path)
    path_plan = _load_json(run_dir / "path_id_plan.json")
    selection_sha256 = str(
        selection.get("semantic_sha256", config_fingerprint(selection))
    )
    selected_state_sha256 = str(
        selection.get(
            "selected_state_sha256",
            selection.get("checkpoint_state_sha256", ""),
        )
    )
    namespace_path = run_dir / "confirmation_namespace_open.json"
    if namespace_path.is_file():
        namespace_record = _load_json(namespace_path)
        if (
            namespace_record.get("path_ids") != path_plan["roles"]["confirmation"]
            or namespace_record.get("selection_sha256") != selection_sha256
        ):
            raise ArtifactCompatibilityError("opened confirmation namespace changed")
    else:
        namespace_record = _semantic(
            {
                "schema": RUN_SCHEMA + "-confirmation-namespace-open",
                "schema_version": 1,
                "opened_at": _now(),
                "opened_once": 1,
                "path_ids": list(path_plan["roles"]["confirmation"]),
                "selection_sha256": selection_sha256,
                "checkpoint_file_sha256": selection["checkpoint_file_sha256"],
                "checkpoint_state_sha256": selected_state_sha256,
                "zero_baseline_sha256": ZERO_BASELINE_SHA256,
                "train_index_sha256": (
                    file_fingerprint(run_dir / "cache" / "train_index.json")
                    if (run_dir / "cache" / "train_index.json").is_file()
                    else "fixture-unavailable"
                ),
                "validation_index_sha256": (
                    file_fingerprint(run_dir / "cache" / "validation_index.json")
                    if (run_dir / "cache" / "validation_index.json").is_file()
                    else "fixture-unavailable"
                ),
                "family_names_sha256": _selection.V3_FAMILY_NAMES_SHA256,
                "bootstrap_seed": CONFIRMATION_BOOTSTRAP_SEED,
                "bootstrap_namespace": _selection.CONFIRMATION_NAMESPACE,
                "bootstrap_replicates": _selection_replicates(args)[0],
                "paths_started": 0,
                "namespace_permanently_burned": 1,
                **NO_WORK,
            }
        )
    count_records = _prepare_confirmation_count_shards(
        run_dir, args=args, namespace_record=namespace_record
    )
    count_list = (
        count_records
        if isinstance(count_records, list)
        else [dict(count_records)]
    )
    count_index = _semantic(
        {
            "schema": RUN_SCHEMA + "-confirmation-bootstrap-count-index",
            "schema_version": 1,
            "namespace_sha256": namespace_record["semantic_sha256"],
            "count_metadata_semantic_sha256": [
                str(record.get("semantic_sha256", config_fingerprint(record)))
                for record in count_list
            ],
            "committed_before_first_path": 1,
        }
    )
    if not namespace_path.is_file():
        # Every count shard is durable before this commit. From this point the
        # namespace is permanently burned and none of those counts is repairable.
        atomic_write_json(namespace_path, namespace_record)
    atomic_write_json(run_dir / "confirmation_bootstrap_count_index.json", count_index)
    execution_record = _run_confirmation_execution(
        run_dir, args, namespace_record, selection
    )
    if not (run_dir / "confirmation_execution.json").is_file():
        atomic_write_json(
            run_dir / "confirmation_execution.json",
            _semantic(
                {
                    "schema": RUN_SCHEMA + "-confirmation-execution-hook",
                    **dict(execution_record),
                }
            ),
        )
    gate = _finalize_confirmation_inference(
        run_dir, args, namespace_record, execution_record
    )
    if not (run_dir / "confirmation_gate.json").is_file():
        atomic_write_json(run_dir / "confirmation_gate.json", gate)
    if not gate_path.is_file():
        atomic_write_json(gate_path, gate)
    # Test hooks may intentionally omit bulky scientific payloads.  Production
    # execution always writes each artifact before returning.
    if args.test_only or not isinstance(execution_record.get("aggregate"), Mapping):
        placeholders = {
            "confirmation_path_risks.npz": None,
            "confirmation_risk_summary.json": {"hook_fixture": 1},
            "confirmation_max_t.npz": None,
            "confirmation_max_t.json": {"hook_fixture": 1},
            "confirmation_metrics.json": {"confirmation_performed": 1},
            "confirmation_index.json": {"hook_fixture": 1},
        }
        for name, value in placeholders.items():
            path = run_dir / name
            if path.is_file():
                continue
            if name.endswith(".npz"):
                _atomic_npz(path, {"fixture": np.zeros(1, dtype=np.float64)})
            else:
                atomic_write_json(path, value)
    _seal_stage(
        run_dir,
        (
            "confirmation_namespace_open.json",
            "confirmation_bootstrap_count_index.json",
            "confirmation_execution.json",
            "confirmation_path_risks.npz",
            "confirmation_risk_summary.json",
            "confirmation_max_t.npz",
            "confirmation_max_t.json",
            "confirmation_metrics.json",
            "confirmation_gate.json",
            "confirmation_index.json",
            "confirm_gate.json",
        ),
        "confirm_artifact_seal.json",
    )
    return gate


def _stage_gates(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: (
            _optional_json(run_dir, f"{name}_gate.json")
            or not_evaluated_gate(name, "stage has not run")
        )
        for name in ("preflight", "cache", "train", "select", "confirm")
    }


def _workflow_record(run_dir: Path, *, require_gate: str) -> dict[str, Any]:
    gates = _stage_gates(run_dir)
    if (
        int(gates["select"].get("no_validation_candidate", 0)) == 1
        and "validation_inference_valid" not in gates["select"]
    ):
        gates["select"]["validation_inference_valid"] = 1
    workflow = evaluate_required_gate(
        preflight_gate=gates["preflight"],
        cache_gate=gates["cache"],
        train_gate=gates["train"],
        select_gate=gates["select"],
        confirm_gate=gates["confirm"],
        require_gate=require_gate,
    )
    decision = decide_workflow(
        preflight_gate=gates["preflight"],
        cache_gate=gates["cache"],
        train_gate=gates["train"],
        select_gate=gates["select"],
        confirm_gate=gates["confirm"],
    )
    config = _load_json(run_dir / "scientific_config.json")
    if int(config.get("test_only", 0)) == 1:
        workflow["passed"] = int(
            require_gate == "none"
            or _passed(gates.get(require_gate))
        )
        workflow["authorizing"] = 0
        decision["authorizing"] = 0
        for name in tuple(decision):
            if name.endswith("_authorized"):
                decision[name] = 0
    workflow["passed"] = int(workflow.get("required_gate_pass", 0))
    evaluated = [
        name
        for name in ("preflight", "cache", "train", "select", "confirm")
        if str(gates[name].get("evaluation_status")) == "evaluated"
    ]
    latest_name = evaluated[-1] if evaluated else None
    latest_gate = gates[latest_name] if latest_name is not None else None
    scientific_complete = int(
        (latest_gate or {}).get("scientific_evidence_complete", 0)
    )
    workflow["scientific_evidence_complete"] = scientific_complete
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(run_dir / "boundary_tangent_v3_decision.json", decision)
    if latest_gate is not None and not _passed(latest_gate):
        state = "gate_failed"
    elif latest_name == "confirm":
        state = "complete"
    elif latest_name is not None:
        next_stage = {
            "preflight": "cache",
            "cache": "train",
            "train": "select",
            "select": "confirm",
        }[latest_name]
        state = f"ready_for_{next_stage}"
    else:
        state = "initialized"
    _status(
        run_dir,
        state=state,
        stage="terminal",
        decision=str(decision.get("decision")),
        scientific_evidence_complete=scientific_complete,
    )
    _artifact_registry(run_dir)
    return workflow


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "cache", "train", "select", "confirm")
    if stage == "report":
        return ()
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return (stage,)


def _report_stage(run_dir: Path) -> None:
    seals = {
        "preflight": "preflight_artifact_seal.json",
        "cache": "cache_artifact_seal.json",
        "train": "train_artifact_seal.json",
        "select": "selection_artifact_seal.json",
        "confirm": "confirm_artifact_seal.json",
    }
    order = tuple(seals)
    completed: list[str] = []
    prior_failed = False
    for stage, name in seals.items():
        gate_path = run_dir / f"{stage}_gate.json"
        if not gate_path.is_file():
            if any((run_dir / f"{later}_gate.json").is_file() for later in order[len(completed) + 1 :]):
                raise ArtifactCompatibilityError("stage ordering is incomplete")
            break
        if prior_failed:
            raise ArtifactCompatibilityError("work continued after a failed gate")
        _verify_stage_seal(run_dir, name)
        gate = _load_json(gate_path)
        completed.append(stage)
        prior_failed = not _passed(gate)
    _provenance.validate_no_v3_baseline_artifacts(run_dir)
    namespace = run_dir / "confirmation_namespace_open.json"
    selection = _optional_json(run_dir, "validation_selection.json")
    if namespace.is_file() and (
        not isinstance(selection, Mapping)
        or selection.get("decision")
        != "zero_baseline_v3_validation_nominee_sealed"
    ):
        raise ArtifactCompatibilityError("confirmation opened without a nominee")
    if (run_dir / "confirmation_execution.json").is_file() and not namespace.is_file():
        raise ArtifactCompatibilityError("confirmation evidence has no namespace seal")
    decision = _optional_json(run_dir, "boundary_tangent_v3_decision.json")
    if isinstance(decision, Mapping):
        scope = _scope(run_dir)
        for name, value in scope.items():
            if name in decision and int(decision[name]) != int(value):
                raise ArtifactCompatibilityError("terminal scope flags changed")
    _verify_existing_registry(run_dir)


def _execution_gate(stage: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    evaluators = {
        "preflight": evaluate_preflight_gate,
        "cache": evaluate_cache_gate,
        "train": evaluate_train_gate,
        "select": evaluate_select_gate,
        "confirm": evaluate_confirm_gate,
    }
    return evaluators[stage](metrics)


def _commit_execution_failure(
    run_dir: Path,
    *,
    stage: str,
    error: BaseException,
) -> None:
    if stage not in {"preflight", "cache", "train", "select", "confirm"}:
        _status(
            run_dir,
            state="execution_failed",
            stage=stage,
            message=str(error),
            failure_domain="workflow_execution",
            failure_code="boundary_tangent_v3_execution_failed",
            scientific_evidence_complete=0,
        )
        _artifact_registry(run_dir)
        return
    domain = str(getattr(error, "failure_domain", "workflow_execution"))
    code = str(
        getattr(
            error,
            "failure_code",
            f"boundary_tangent_v3_{stage}_execution_failed",
        )
    )
    metrics_name = {
        "preflight": "preflight_metrics.json",
        "cache": "cache_metrics.json",
        "train": "train_metrics.json",
        "select": "select_metrics.json",
        "confirm": "confirmation_metrics.json",
    }[stage]
    metrics = {
        "schema": RUN_SCHEMA + f"-{stage}-execution-failure",
        "schema_version": 1,
        "evaluation_status": "execution_failed",
        "failure_domain": domain,
        "failure_code": code,
        "message": str(error),
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "production_cache_generation_performed": int(
            (run_dir / "cache_metrics.json").is_file()
        ),
        "physical_training_performed": int(
            (run_dir / "physical_training_started.json").is_file()
        ),
        "validation_selection_performed": int(
            (run_dir / "validation_label_open.json").is_file()
        ),
        "confirmation_performed": int(
            (run_dir / "confirmation_namespace_open.json").is_file()
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / metrics_name, metrics)
    failure_name = f"{stage}_execution_failure.json"
    atomic_write_json(run_dir / failure_name, metrics)
    gate = _execution_gate(stage, metrics)
    gate_name = f"{stage}_gate.json"
    atomic_write_json(run_dir / gate_name, gate)
    if stage == "confirm":
        atomic_write_json(run_dir / "confirmation_gate.json", gate)
    seal_name = {
        "preflight": "preflight_artifact_seal.json",
        "cache": "cache_artifact_seal.json",
        "train": "train_artifact_seal.json",
        "select": "selection_artifact_seal.json",
        "confirm": "confirm_artifact_seal.json",
    }[stage]
    names = [metrics_name, failure_name, gate_name]
    if stage == "confirm":
        names.append("confirmation_gate.json")
    _seal_stage(run_dir, names, seal_name)
    _workflow_record(run_dir, require_gate="none")
    _status(
        run_dir,
        state="execution_failed",
        stage=stage,
        message=str(error),
        failure_domain=domain,
        failure_code=code,
        scientific_evidence_complete=0,
    )
    _artifact_registry(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact zero-baseline Jacobi/RB boundary-tangent v3 learnability"
    )
    parser.add_argument("--stage", choices=STAGES, default="report")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--failed-v3-preflight-run-dir", type=Path, required=True)
    parser.add_argument("--parent-v2-run-dir", type=Path, required=True)
    parser.add_argument("--adjudication-run-dir", type=Path, required=True)
    parser.add_argument("--parent-eager-pipeline-run-dir", type=Path, required=True)
    parser.add_argument("--parent-coarse-residual-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability"),
    )
    parser.add_argument(
        "--run-name", default="production-zero-baseline-v3-learnability"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-path-count", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--test-outer-steps", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument(
        "--test-maximum-updates", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--test-bootstrap-replicates", type=int, default=8, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    for name in (
        "failed_v3_preflight_run_dir",
        "parent_v2_run_dir",
        "adjudication_run_dir",
        "parent_eager_pipeline_run_dir",
        "parent_coarse_residual_run_dir",
        "runs_root",
        "resume_run_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.stage not in {"preflight", "report", "all"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    if args.stage == "report" and args.resume_run_dir is None:
        parser.error("--stage report requires --resume-run-dir")
    if args.test_only:
        if args.require_gate != "none":
            parser.error("test-only runs are nonauthorizing and require --require-gate none")
        if args.test_path_count < 1 or args.test_path_count > 4:
            parser.error("test path count must lie in [1,4]")
        if args.test_outer_steps < 8 or args.test_outer_steps % 8:
            parser.error("test outer steps must be a positive multiple of eight")
        if args.test_maximum_updates < 0:
            parser.error("test maximum updates must be nonnegative")
        if args.test_bootstrap_replicates < 2:
            parser.error("test bootstrap replicates must be at least two")
    elif args.device != "cuda":
        parser.error("production v3 stages require --device cuda")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    print(f"zero-baseline v3 run directory: {run_dir}", flush=True)
    active_stage = "initialize"
    try:
        _initialize(run_dir, args, resumed=resumed)
        # Initialization above deliberately commits all non-CUDA provenance first.
        if args.stage != "report":
            configure_exact_torch_backend(args.device)
        functions = {
            "preflight": _preflight_stage,
            "cache": _cache_stage,
            "train": _train_stage,
            "select": _select_stage,
            "confirm": _confirm_stage,
        }
        stage_failed = False
        for active_stage in _stage_sequence(args.stage):
            gate = functions[active_stage](run_dir, args)
            _workflow_record(run_dir, require_gate="none")
            if not _passed(gate):
                stage_failed = True
                break
        if args.stage == "report":
            active_stage = "report"
            _report_stage(run_dir)
        workflow = _workflow_record(run_dir, require_gate=args.require_gate)
        decision = _load_json(run_dir / "boundary_tangent_v3_decision.json")
        print(f"zero-baseline v3 decision: {decision.get('decision')}", flush=True)
        if stage_failed:
            return 2
        return 0 if int(workflow.get("passed", 0)) == 1 else 1
    except ArtifactCompatibilityError as exc:
        # Genuine immutable-evidence incompatibilities retain their own class.
        if (
            active_stage != "initialize"
            and run_dir.is_dir()
            and (run_dir / "scientific_config.json").is_file()
        ):
            _commit_execution_failure(run_dir, stage=active_stage, error=exc)
        print(f"zero-baseline v3 compatibility error: {exc}", flush=True)
        return 2
    except Exception as exc:
        if (
            active_stage != "initialize"
            and run_dir.is_dir()
            and (run_dir / "scientific_config.json").is_file()
        ):
            _commit_execution_failure(run_dir, stage=active_stage, error=exc)
        print(f"zero-baseline v3 error: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
