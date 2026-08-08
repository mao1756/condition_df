"""Memory-safe continuation of the exact zero-baseline boundary-tangent v3 gate.

The immutable parent already contains a complete, passing train/validation
cache.  This child workflow keeps that cache read-only and moves at most one
training batch to CUDA at a time.  It does not execute a controller or sampler.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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
    source_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent import (
    direct_raw_target_mse,
    synthetic_tangent_target,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_gate import (
    BoundaryTangentV3Thresholds,
    evaluate_confirm_gate as evaluate_v3_confirm_gate,
    evaluate_select_gate as evaluate_v3_select_gate,
    evaluate_train_gate as evaluate_v3_train_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZERO_BASELINE_SHA256,
    ZeroBaselineBoundaryTangentPredictor,
    configure_exact_synthetic_zero_baseline_teacher,
)
from mnist.d0_jacobi_rb_learnability import (
    ModelInputs,
    call_model,
    deterministic_batch_indices,
    enable_deterministic_torch,
    state_dict_sha256,
)
from mnist import diag_d0_jacobi_rb_boundary_tangent_v3_learnability as _v3
from mnist import d0_jacobi_rb_boundary_tangent_v3_memory as _memory
from mnist import d0_jacobi_rb_boundary_tangent_v3_memory_gate as _gate
from mnist import d0_jacobi_rb_boundary_tangent_v3_memory_provenance as _provenance


RUN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-v3-memory-confirmation"
TEST_RUN_SCHEMA = RUN_SCHEMA + "-nonauthorizing-test"
STAGES = ("preflight", "train", "select", "confirm", "report", "all")
REQUIRED_GATES = ("none", "preflight", "train", "select", "confirm")
TRAINING = dict(_v3.TRAINING)
MAXIMUM_FORWARD_BATCH = 32
MEMORY_SCHEDULE_VERSION = "v3-host-batched-forward-1"
NO_WORK = dict(_v3.NO_WORK)
_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "boundary_tangent_v3_memory_decision.json",
}


class MemoryConfirmationError(RuntimeError):
    """Typed execution failure for the additive recovery workflow."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "memory_confirmation_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _load_json(path: str | Path) -> dict[str, Any]:
    return _v3._load_json(path)


def _verified_semantic_record(
    path: str | Path,
    *,
    schema: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one opening/seal record and verify its complete semantic binding.

    Opening records can exist without appearing in a terminal registry when a
    process is interrupted immediately after the atomic write.  Treating the
    embedded digest as an opaque authorization token would therefore let a
    malformed orphan open physical labels on resume.  Recompute the digest and
    bind all role/purpose fields before constructing a label store.
    """

    record = _load_json(path)
    body = dict(record)
    semantic = body.pop("semantic_sha256", None)
    if semantic != config_fingerprint(body):
        raise ArtifactCompatibilityError("semantic opening record changed")
    if record.get("schema") != schema:
        raise ArtifactCompatibilityError("semantic opening record schema changed")
    for name, value in expected.items():
        if record.get(name) != value:
            raise ArtifactCompatibilityError(
                f"semantic opening record field changed: {name}"
            )
    return record


def _passed(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("evaluation_status") == "evaluated"
        and int(value.get("passed", 0)) == 1
    )


def _scope(run_dir: Path) -> dict[str, int]:
    def field(path: str, name: str) -> int:
        target = run_dir / path
        return int(target.is_file() and int(_load_json(target).get(name, 0)) == 1)

    return {
        "production_cache_generation_performed": 0,
        "immutable_parent_cache_reused": int(
            (run_dir / "immutable_cache_binding.json").is_file()
        ),
        "physical_training_performed": int(
            (run_dir / "physical_training_started.json").is_file()
        ),
        "validation_selection_performed": field(
            "select_metrics.json", "validation_selection_performed"
        ),
        "confirmation_performed": field(
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


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in run_dir.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        relative = path.relative_to(run_dir).as_posix()
        if relative in _REGISTRY_EXCLUDED or ".tmp" in path.name:
            continue
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


def _verify_child_registry(run_dir: Path) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    record = _load_json(path)
    artifacts = record.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or int(record.get("artifact_count", -1)) != len(artifacts)
        or record.get("semantic_sha256")
        != config_fingerprint({"artifacts": artifacts})
    ):
        raise ArtifactCompatibilityError("child artifact registry changed")
    for item in artifacts:
        target = run_dir / str(item["path"])
        if (
            not target.is_file()
            or item.get("sha256") != file_fingerprint(target)
            or int(item.get("size", -1)) != target.stat().st_size
        ):
            raise ArtifactCompatibilityError("registered child artifact changed")


def _seal_stage(run_dir: Path, names: Sequence[str], seal_name: str) -> dict[str, Any]:
    artifacts = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"cannot seal missing artifact: {name}")
        artifacts.append(
            {
                "path": str(name),
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
    record = _load_json(run_dir / seal_name)
    body = dict(record)
    semantic = body.pop("semantic_sha256", None)
    if semantic != config_fingerprint(body):
        raise ArtifactCompatibilityError("stage seal changed")
    for item in record.get("artifacts", []):
        path = run_dir / str(item["path"])
        if (
            not path.is_file()
            or item.get("sha256") != file_fingerprint(path)
            or int(item.get("size", -1)) != path.stat().st_size
        ):
            raise ArtifactCompatibilityError("sealed artifact changed")


def _prepare_execution_retry(run_dir: Path, stage: str) -> bool:
    """Archive one sealed execution failure and reopen only its stage markers.

    Scientific ``evaluated`` failures are terminal.  An ``execution_failed``
    gate, however, must not strand exact optimizer checkpoints or confirmation
    shards.  The complete failed attempt is copied into an append-only attempt
    directory before the canonical gate/seal/failure markers are removed.  All
    evidence and progress artifacts remain in place for the stage-specific
    resume logic.
    """

    if stage not in {"preflight", "train", "select", "confirm"}:
        raise ValueError(f"unsupported retry stage: {stage}")
    gate_name = f"{stage}_gate.json"
    gate_path = run_dir / gate_name
    if not gate_path.is_file():
        return False
    gate = _load_json(gate_path)
    if gate.get("evaluation_status") != "execution_failed":
        return False
    seal_name = {
        "preflight": "preflight_artifact_seal.json",
        "train": "train_artifact_seal.json",
        "select": "selection_artifact_seal.json",
        "confirm": "confirm_artifact_seal.json",
    }[stage]
    _verify_stage_seal(run_dir, seal_name)
    metrics_name = {
        "preflight": "preflight_metrics.json",
        "train": "train_metrics.json",
        "select": "select_metrics.json",
        "confirm": "confirmation_metrics.json",
    }[stage]
    failure_name = f"{stage}_execution_failure.json"
    required = (metrics_name, failure_name, gate_name, seal_name)
    for name in required:
        if not (run_dir / name).is_file():
            raise ArtifactCompatibilityError(
                f"sealed {stage} execution failure is incomplete"
            )
    root = run_dir / "execution_attempts" / stage
    existing = sorted(
        path for path in root.glob("attempt-*") if path.is_dir()
    ) if root.is_dir() else []
    attempt = root / f"attempt-{len(existing) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    archived: list[dict[str, Any]] = []
    archive_names = list(required)
    if stage == "confirm" and (run_dir / "confirmation_gate.json").is_file():
        archive_names.append("confirmation_gate.json")
    if (run_dir / "artifact_registry.json").is_file():
        archive_names.append("artifact_registry.json")
    for name in archive_names:
        source = run_dir / name
        destination = attempt / name
        atomic_write_json(destination, _load_json(source))
        archived.append(
            {
                "path": name,
                "source_sha256": file_fingerprint(source),
                "archive_sha256": file_fingerprint(destination),
            }
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-execution-attempt-archive",
            "schema_version": 1,
            "stage": stage,
            "attempt": len(existing) + 1,
            "failure_code": gate.get("failure_code"),
            "artifacts": archived,
            "scientific_gate_reopened": 0,
            "execution_failure_reopened": 1,
            **NO_WORK,
        }
    )
    atomic_write_json(attempt / "retry_authorization.json", record)
    # Remove only canonical terminal-attempt records. All checkpoints,
    # labels-opening seals, panels, namespaces, and shards remain untouched.
    # Removing the failed metrics as well avoids a retry-time registry binding
    # to a path the resumed stage must legitimately replace.
    for name in (metrics_name, gate_name, seal_name, failure_name):
        (run_dir / name).unlink()
    if stage == "confirm" and (run_dir / "confirmation_gate.json").is_file():
        (run_dir / "confirmation_gate.json").unlink()
    # A retry is an explicitly open transaction.  Archive and remove the old
    # terminal registry instead of rebinding mutable progress checkpoints: the
    # resumed stage is allowed to replace those checkpoints, and a second hard
    # interruption must still reach them.  The next terminal workflow commit
    # writes a fresh complete registry.
    registry_path = run_dir / "artifact_registry.json"
    if registry_path.is_file():
        registry_path.unlink()
    return True


def _stage_seal_paths(run_dir: Path, seal_name: str) -> set[str]:
    """Return verified paths from a stage seal.

    The memory child replaces the frozen v3 selector/confirmation seal only
    after adding its diagnostics.  Presence of a diagnostics file alone is not
    a commit point: a hard interruption can leave that file beside the still
    valid legacy seal.  Callers use membership in the verified seal to
    distinguish a finalized child stage from such an orphan.
    """

    _verify_stage_seal(run_dir, seal_name)
    return {
        str(item["path"])
        for item in _load_json(run_dir / seal_name).get("artifacts", [])
    }


def _legacy_snapshot_names(stage: str) -> tuple[str, str, str]:
    if stage not in {"select", "confirm"}:
        raise ValueError(f"unsupported legacy snapshot stage: {stage}")
    prefix = "selection" if stage == "select" else "confirmation"
    return (
        f"{prefix}_legacy_metrics.json",
        f"{prefix}_legacy_gate.json",
        f"{prefix}_legacy_stage_snapshot.json",
    )


def _prepare_legacy_stage_snapshot(
    run_dir: Path,
    *,
    stage: str,
    seal_name: str,
    metrics_name: str,
    gate_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze the legacy stage result before child metrics are rewritten.

    The snapshot is committed before either legacy JSON is augmented.  On an
    interrupted finalization it supplies the immutable scientific inputs while
    every other artifact from the legacy seal is still hash-checked.  This
    gives select/confirm an exact orphan-finalization path without rerunning a
    sealed label panel or confirmation namespace.
    """

    metrics_copy_name, gate_copy_name, snapshot_name = _legacy_snapshot_names(stage)
    snapshot_path = run_dir / snapshot_name
    mutable = {metrics_name, gate_name}
    if stage == "confirm":
        # Frozen v3 writes the same gate under both public names.
        mutable.add("confirmation_gate.json")

    if snapshot_path.is_file():
        snapshot = _verified_semantic_record(
            snapshot_path,
            schema=RUN_SCHEMA + f"-{stage}-legacy-stage-snapshot",
            expected={"stage": stage},
        )
        for item in snapshot.get("immutable_artifacts", []):
            path = run_dir / str(item["path"])
            if (
                not path.is_file()
                or item.get("sha256") != file_fingerprint(path)
                or int(item.get("size", -1)) != int(path.stat().st_size)
            ):
                raise ArtifactCompatibilityError(
                    f"legacy {stage} immutable artifact changed"
                )
        metrics_copy = run_dir / metrics_copy_name
        gate_copy = run_dir / gate_copy_name
        if (
            not metrics_copy.is_file()
            or not gate_copy.is_file()
            or snapshot.get("metrics_copy_sha256") != file_fingerprint(metrics_copy)
            or snapshot.get("gate_copy_sha256") != file_fingerprint(gate_copy)
        ):
            raise ArtifactCompatibilityError(f"legacy {stage} snapshot changed")
        return _load_json(metrics_copy), _load_json(gate_copy)

    _verify_stage_seal(run_dir, seal_name)
    seal = _load_json(run_dir / seal_name)
    metrics = _load_json(run_dir / metrics_name)
    gate = _load_json(run_dir / gate_name)
    atomic_write_json(run_dir / metrics_copy_name, metrics)
    atomic_write_json(run_dir / gate_copy_name, gate)
    immutable = [
        dict(item)
        for item in seal.get("artifacts", [])
        if str(item.get("path")) not in mutable
    ]
    snapshot = _semantic(
        {
            "schema": RUN_SCHEMA + f"-{stage}-legacy-stage-snapshot",
            "schema_version": 1,
            "stage": stage,
            "legacy_seal_semantic_sha256": seal.get("semantic_sha256"),
            "metrics_copy_path": metrics_copy_name,
            "metrics_copy_sha256": file_fingerprint(run_dir / metrics_copy_name),
            "gate_copy_path": gate_copy_name,
            "gate_copy_sha256": file_fingerprint(run_dir / gate_copy_name),
            "immutable_artifacts": immutable,
            **NO_WORK,
        }
    )
    atomic_write_json(snapshot_path, snapshot)
    return metrics, gate


def _source_paths() -> tuple[Path, ...]:
    current = (
        Path(__file__).resolve(),
        Path(inspect.getfile(_memory)).resolve(),
        Path(inspect.getfile(_gate)).resolve(),
        Path(inspect.getfile(_provenance)).resolve(),
    )
    return tuple(sorted(set((*_v3._source_set(), *current))))


def _memory_schedule() -> dict[str, Any]:
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-memory-schedule",
            "schema_version": 1,
            "version": MEMORY_SCHEDULE_VERSION,
            "maximum_model_forward_batch": MAXIMUM_FORWARD_BATCH,
            "maximum_target_batch": MAXIMUM_FORWARD_BATCH,
            "host_backed_inputs": 1,
            "host_backed_labels": 1,
            "full_cache_cuda_tensor_forbidden": 1,
            "prediction_output_device": "cpu",
            "automatic_batch_selection": 0,
            "allocator_workaround": 0,
            "maximum_peak_memory_fraction": 0.80,
            "canonical_scale_reducer": "row_square_sum_then_python_math_fsum",
            **NO_WORK,
        }
    )


def _compatibility_index(binding: Mapping[str, Any], role: str) -> dict[str, Any]:
    source = binding["roles"][role]
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-external-cache-index-binding",
            "schema_version": 1,
            "role": role,
            "external_parent_run_dir": binding["parent_run_dir"],
            "external_binding_file_sha256": source["binding_file_sha256"],
            "external_index_file_sha256": source["source_file_sha256"],
            "external_index_path": source["source_path"],
            "cache_rows_copied": 0,
            "cache_payload_copied": 0,
        }
    )


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    parent = args.failed_v3_train_run_dir.resolve()
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        if run_dir == parent:
            raise ArtifactCompatibilityError(
                "the immutable failed parent cannot be resumed as a child"
            )
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.runs_root / f"{stamp}_{args.run_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, False


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    try:
        parent = _provenance.verify_failed_v3_train_parent(
            args.failed_v3_train_run_dir
        )
    except Exception as exc:
        raise MemoryConfirmationError(
            f"failed v3 train parent verification failed: {exc}",
            failure_domain="control_provenance",
            failure_code="failed_v3_train_parent_invalid",
        ) from exc
    adjudication = dict(parent)
    try:
        binding = _provenance.build_immutable_cache_binding(
            args.failed_v3_train_run_dir
        )
    except Exception as exc:
        raise MemoryConfirmationError(
            f"immutable parent cache binding failed: {exc}",
            failure_domain="immutable_cache_binding",
            failure_code="immutable_parent_cache_binding_invalid",
        ) from exc
    schedule = _memory_schedule()
    sources = _source_paths()
    source_hash = source_fingerprint(sources)
    try:
        parent_path_plan = _load_json(
            Path(binding["parent_run_dir"]) / "path_id_plan.json"
        )
        parent_cohort_plan = _load_json(
            Path(binding["parent_run_dir"]) / "cohort_plan.json"
        )
    except Exception as exc:
        raise MemoryConfirmationError(
            f"immutable cache plan binding failed: {exc}",
            failure_domain="immutable_cache_binding",
            failure_code="immutable_cache_plan_binding_invalid",
        ) from exc
    config = _semantic(
        {
            "schema": RUN_SCHEMA + "-scientific-config",
            "schema_version": 1,
            "parent_registry_semantic_sha256": parent["immutable_registry"][
                "semantic_sha256"
            ],
            "immutable_cache_binding_sha256": binding["semantic_sha256"],
            "memory_schedule_sha256": schedule["semantic_sha256"],
            "path_id_plan_sha256": parent_path_plan["semantic_sha256"],
            "cohort_plan_sha256": parent_cohort_plan["semantic_sha256"],
            "training": dict(TRAINING),
            "thresholds": BoundaryTangentV3Thresholds().to_dict(),
            "target": "unchanged exact certified Jacobi/Rao-Blackwell label",
            "objective": "plain unweighted direct MSE",
            "zero_baseline_sha256": ZERO_BASELINE_SHA256,
            "cache_generated_in_child": 0,
            "test_only": int(args.test_only),
            "authorizing": int(not args.test_only),
            **NO_WORK,
        }
    )
    manifest = {
        "schema": (TEST_RUN_SCHEMA if args.test_only else RUN_SCHEMA) + "-manifest",
        "schema_version": 1,
        "created_at": _now(),
        "device": str(args.device),
        "source_fingerprint": source_hash,
        "scientific_config_sha256": config["semantic_sha256"],
        "parent_registry_semantic_sha256": parent["immutable_registry"][
            "semantic_sha256"
        ],
        "immutable_cache_binding_sha256": binding["semantic_sha256"],
        "memory_schedule_sha256": schedule["semantic_sha256"],
        "test_only": int(args.test_only),
        "authorizing": int(not args.test_only),
        **NO_WORK,
    }
    if resumed:
        existing = _load_json(run_dir / "run_manifest.json")
        expected = {key: value for key, value in manifest.items() if key != "created_at"}
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ArtifactCompatibilityError("resume manifest compatibility changed")
        _provenance.verify_immutable_cache_binding(
            _load_json(run_dir / "immutable_cache_binding.json")
        )
        _verify_child_registry(run_dir)
        return

    atomic_write_json(run_dir / "parent_verification.json", parent)
    atomic_write_json(run_dir / "failed_v3_train_adjudication.json", adjudication)
    atomic_write_json(run_dir / "immutable_cache_binding.json", binding)
    atomic_write_json(run_dir / "training_memory_schedule.json", schedule)
    atomic_write_json(run_dir / "path_id_plan.json", parent_path_plan)
    atomic_write_json(run_dir / "cohort_plan.json", parent_cohort_plan)
    atomic_write_json(run_dir / "scientific_config.json", config)
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
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "cache" / "train_index.json",
        _compatibility_index(binding, "train"),
    )
    atomic_write_json(
        run_dir / "cache" / "validation_index.json",
        _compatibility_index(binding, "validation"),
    )
    atomic_write_json(
        run_dir / "cache_metrics.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-inherited-cache-metrics",
                "schema_version": 1,
                "cache_elapsed_seconds": binding["cache_elapsed_seconds"],
                "production_cache_generation_performed": 0,
                "immutable_parent_cache_reused": 1,
                "parent_cache_gate_passed": 1,
                **NO_WORK,
            }
        ),
    )
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    _status(run_dir, state="initialized", stage="initialize")


def _binding(run_dir: Path) -> dict[str, Any]:
    try:
        value = _load_json(run_dir / "immutable_cache_binding.json")
        _provenance.verify_immutable_cache_binding(value)
    except (ArtifactCompatibilityError, OSError, ValueError) as exc:
        raise MemoryConfirmationError(
            "immutable parent cache binding changed",
            failure_domain="immutable_cache_binding",
            failure_code="immutable_cache_binding_invalid",
        ) from exc
    return value


def _external_root(run_dir: Path) -> Path:
    return Path(_binding(run_dir)["parent_run_dir"]).resolve()


def _slice_store(store: _memory.HostInputStore, count: int) -> _memory.HostInputStore:
    stop = min(int(count), store.row_count)
    arrays = {
        name: np.array(value[:stop], copy=True, order="C")
        for name, value in store.arrays.items()
    }
    return _memory.HostInputStore.from_arrays(
        arrays,
        role=store.role,
        cache_root=store.cache_root,
        index={},
    )


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    _prepare_execution_retry(run_dir, "preflight")
    gate_path = run_dir / "preflight_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
        return _load_json(gate_path)
    binding = _binding(run_dir)
    parent = _external_root(run_dir)
    schedule = _load_json(run_dir / "training_memory_schedule.json")
    train = _memory.open_external_input_store(parent, "train")
    seam = _slice_store(train, MAXIMUM_FORWARD_BATCH)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    rows = np.arange(seam.row_count, dtype=np.int64)
    host_batch = seam.batch(rows, device=device)
    direct_arrays = {
        name: np.array(value, copy=True, order="C")
        for name, value in seam.arrays.items()
    }
    direct_batch = _v3._legacy._model_inputs_from_arrays(direct_arrays, device)
    input_equal = all(
        torch.equal(getattr(host_batch, name), getattr(direct_batch, name))
        for name in _v3.MODEL_INPUT_FIELDS
    )
    enable_deterministic_torch()
    torch.manual_seed(_v3.SYNTHETIC_CONTROL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_v3.SYNTHETIC_CONTROL_SEED)
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    guard = _memory.ModelCallBatchGuard()
    model.zero_grad(set_to_none=True)
    guarded = guard.call(model, host_batch).to(torch.float64)
    direct = call_model(model, direct_batch).to(torch.float64)
    output_equal = bool(torch.equal(guarded, direct))
    target = synthetic_tangent_target(host_batch).detach().to(torch.float64)
    loss = torch.mean((guarded - target).square())
    loss.backward()
    finite_backward = bool(
        torch.isfinite(loss)
        and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
    )
    reduction_a = _memory.canonical_row_square_reduction(
        target, batch_size=MAXIMUM_FORWARD_BATCH
    )
    reducer = _memory.CanonicalRowSquareReducer()
    reducer.update(target[: max(1, seam.row_count // 2)])
    reducer.update(target[max(1, seam.row_count // 2) :])
    reduction_b = reducer.record()
    reducer_equal = (
        reduction_a["square_sum"] == reduction_b["square_sum"]
        and reduction_a["element_count"] == reduction_b["element_count"]
    )
    direct_scale = float(torch.sqrt(torch.mean(target.square())).cpu())
    streamed_scale = float(reduction_a["rms"])
    scale_relative_error = abs(streamed_scale - direct_scale) / max(
        abs(direct_scale), np.finfo(float).tiny
    )

    # Exercise the label-store contract without opening a parent label archive.
    dummy_authorization = _memory.LabelOpenAuthorization(
        parent,
        "train",
        "physical_training",
        "0" * 64,
    )
    dummy_labels = _memory.HostLabelStore.from_arrays(
        {
            "denoising_target": np.zeros(
                (seam.row_count, _v3.EDGES_PER_PHASE), dtype=np.float64
            )
        },
        authorization=dummy_authorization,
    )
    dummy_label_batch = dummy_labels.target_batch(rows, device=device)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    total_bytes = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else 1
    )
    peak_fraction = peak_bytes / max(total_bytes, 1)
    guard_record = guard.record()
    seam_record = _semantic(
        {
            "schema": RUN_SCHEMA + "-cuda-memory-seam",
            "schema_version": 1,
            "row_count": seam.row_count,
            "input_fields_bit_identical": int(input_equal),
            "model_outputs_bit_identical": int(output_equal),
            "finite_forward_backward": int(finite_backward),
            "dummy_label_store_shape_valid": int(
                tuple(dummy_label_batch.shape)
                == (seam.row_count, _v3.EDGES_PER_PHASE)
            ),
            "maximum_observed_model_forward_batch_size": guard_record[
                "maximum_observed_batch_size"
            ],
            "model_call_count": guard_record["call_count"],
            "peak_memory_bytes": peak_bytes,
            "device_total_memory_bytes": total_bytes,
            "peak_memory_fraction": peak_fraction,
            "parent_physical_labels_deserialized": 0,
            "synthetic_streamed_scale": streamed_scale,
            "synthetic_direct_scale": direct_scale,
            "synthetic_scale_relative_error": scale_relative_error,
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "memory_cuda_seam.json", seam_record)
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "failed_parent_valid": 1,
        "corrected_parent_adjudication_valid": 1,
        "complete_parent_registry_valid": 1,
        "parent_preflight_and_cache_passed": 1,
        "parent_immutability_valid": 1,
        "downstream_evidence_absent": 1,
        "confirmation_namespace_unopened": 1,
        "immutable_cache_binding_valid": 1,
        "cache_seal_valid": 1,
        "cache_indexes_valid": 1,
        "cache_read_only": int(binding["cache_is_read_only"]),
        "cache_not_copied_or_linked": int(
            not binding["cache_copied"] and not binding["cache_linked"]
        ),
        "physical_labels_deserialized_during_binding_zero": int(
            binding["physical_labels_deserialized_during_binding"] == 0
        ),
        "memory_contract_valid": int(
            schedule["version"] == MEMORY_SCHEDULE_VERSION
        ),
        "host_backed_input_store_valid": 1,
        "host_backed_label_store_valid": 1,
        "label_firewall_valid": 1,
        "maximum_forward_batch_enforced": int(
            guard_record["maximum_observed_batch_size"] <= MAXIMUM_FORWARD_BATCH
        ),
        "full_cache_cuda_tensor_absent": 1,
        "host_device_batch_equivalence_valid": int(input_equal and output_equal),
        "cuda_forward_backward_seam_valid": int(finite_backward),
        "streaming_reducer_valid": int(reducer_equal),
        "automatic_batch_sizing_disabled": 1,
        "allocator_workaround_disabled": 1,
        "maximum_observed_model_forward_batch_size": guard_record[
            "maximum_observed_batch_size"
        ],
        "full_cache_cuda_tensor_count": 0,
        "peak_memory_fraction": peak_fraction,
        "synthetic_scale_relative_error": scale_relative_error,
        "immutable_parent_cache_reused": 1,
        "production_cache_generation_performed": 0,
        "physical_training_performed": 0,
        "validation_selection_performed": 0,
        "confirmation_performed": 0,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    gate = _gate.evaluate_preflight_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "failed_v3_train_adjudication.json",
            "immutable_cache_binding.json",
            "training_memory_schedule.json",
            "memory_cuda_seam.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )
    return gate


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _control_fingerprint(
    run_dir: Path, name: str, store: _memory.HostInputStore
) -> str:
    return config_fingerprint(
        {
            "schema": RUN_SCHEMA + "-control-fingerprint",
            "control": name,
            "scientific_config_sha256": _load_json(
                run_dir / "scientific_config.json"
            )["semantic_sha256"],
            "immutable_cache_binding_sha256": _binding(run_dir)[
                "semantic_sha256"
            ],
            "memory_schedule_sha256": _load_json(
                run_dir / "training_memory_schedule.json"
            )["semantic_sha256"],
            "input_index_sha256": config_fingerprint(dict(store.index)),
        }
    )


def _train_synthetic_control(
    run_dir: Path,
    *,
    train: _memory.HostInputStore,
    validation: _memory.HostInputStore,
    maximum_updates: int,
    device: torch.device,
    guard: _memory.ModelCallBatchGuard,
) -> dict[str, Any]:
    enable_deterministic_torch()
    torch.manual_seed(_v3.SYNTHETIC_CONTROL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_v3.SYNTHETIC_CONTROL_SEED)
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    scale, scale_record = _memory.canonical_streamed_target_scale(
        train, device=device
    )
    fingerprint = _control_fingerprint(run_dir, "synthetic_teacher", train)
    progress_path = run_dir / "checkpoints" / "synthetic-teacher-progress.pt"
    completed = 0
    history: list[dict[str, Any]] = []
    if progress_path.is_file():
        payload = torch.load(progress_path, map_location=device, weights_only=False)
        if payload.get("fingerprint") != fingerprint:
            raise ArtifactCompatibilityError("synthetic control fingerprint changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed = int(payload["completed_update"])
        history = [dict(row) for row in payload["history"]]
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(list(payload["cuda_rng_states"]))

    def checkpoint(update: int) -> None:
        _v3._atomic_torch(
            progress_path,
            {
                "schema": RUN_SCHEMA + "-synthetic-progress",
                "schema_version": 1,
                "fingerprint": fingerprint,
                "completed_update": int(update),
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
        rows = deterministic_batch_indices(
            train.row_count,
            TRAINING["batch_size"],
            update - 1,
            _v3.SYNTHETIC_CONTROL_SEED,
        )
        values = _memory.synthetic_training_step(
            model,
            optimizer,
            train,
            rows,
            scale=scale,
            device=device,
            guard=guard,
            gradient_norm_clip=TRAINING["gradient_norm_clip"],
        )
        if update % TRAINING["checkpoint_interval"] == 0 or update == maximum_updates:
            history.append({"update": update, **values})
            checkpoint(update)
    if maximum_updates == 0 and not progress_path.is_file():
        checkpoint(0)
    evaluation = _memory.stream_target_metrics(
        model,
        validation,
        device=device,
        path_rows=np.asarray(validation.row_array("path_id"), dtype=np.int64),
        guard=guard,
    )
    relative = float(evaluation["relative_mse"])
    every = int(evaluation["every_path_beats_zero"])
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-synthetic-teacher-control",
            "schema_version": 1,
            "complete": 1,
            "selected_update": maximum_updates,
            "target_scale": scale,
            "target_scale_reduction": scale_record,
            "validation_mse": evaluation["model_mse"],
            "zero_validation_mse": evaluation["zero_mse"],
            "relative_validation_mse": relative,
            "every_validation_path_beats_zero": every,
            "path_metrics": evaluation["path_metrics"],
            "passed": int(relative <= 0.01 and every == 1),
            "physical_labels_opened": 0,
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "synthetic_teacher_control.json", record)
    atomic_write_csv(
        run_dir / "synthetic_teacher_per_path.csv", evaluation["path_metrics"]
    )
    return record


def _run_prelabel_controls(
    run_dir: Path,
    *,
    train: _memory.HostInputStore,
    validation: _memory.HostInputStore,
    maximum_updates: int,
    device: torch.device,
    guard: _memory.ModelCallBatchGuard,
) -> dict[str, Any]:
    zero_model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    zero_scan = _memory.stream_zero_initialization(
        zero_model,
        {"train": train, "validation": validation},
        device=device,
        guard=guard,
    )
    keys = tuple(zero_model.state_dict())
    zero_record = _semantic(
        {
            "schema": RUN_SCHEMA + "-zero-initialization-control",
            "schema_version": 1,
            "train_prediction_exact_zero": zero_scan["roles"]["train"][
                "prediction_exact_zero"
            ],
            "validation_prediction_exact_zero": zero_scan["roles"][
                "validation"
            ]["prediction_exact_zero"],
            "train_baseline_exact_zero": zero_scan["roles"]["train"][
                "baseline_exact_zero"
            ],
            "validation_baseline_exact_zero": zero_scan["roles"]["validation"][
                "baseline_exact_zero"
            ],
            "state_dict_baseline_free": int(
                "_q_values" not in keys
                and all("baseline" not in name.lower() for name in keys)
            ),
            "zero_baseline_sha256": ZERO_BASELINE_SHA256,
            "streaming_scan": zero_scan,
            "passed": int(
                int(zero_scan["passed"]) == 1 and "_q_values" not in keys
            ),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "zero_initialization_control.json", zero_record)
    del zero_model

    synthetic = _train_synthetic_control(
        run_dir,
        train=train,
        validation=validation,
        maximum_updates=maximum_updates,
        device=device,
        guard=guard,
    )

    torch.manual_seed(_v3.NULL_CONTROL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_v3.NULL_CONTROL_SEED)
    teacher = ZeroBaselineBoundaryTangentPredictor(zero_residual=False).to(device)
    configure_exact_synthetic_zero_baseline_teacher(teacher)
    student = ZeroBaselineBoundaryTangentPredictor(zero_residual=False).to(device)
    student.load_state_dict(_clone_state_dict(teacher), strict=True)
    optimizer = torch.optim.Adam(
        student.parameters(), lr=TRAINING["learning_rate"], weight_decay=0.0
    )
    null = _memory.exact_null_batchwise_one_step(
        teacher,
        student,
        optimizer,
        train,
        validation,
        device=device,
        guard=guard,
    )
    null = _semantic({**null, **NO_WORK})
    atomic_write_json(run_dir / "exact_model_null_control.json", null)
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


def _training_label_store(
    run_dir: Path, train_input: _memory.HostInputStore
) -> _memory.HostLabelStore:
    seal = _verified_semantic_record(
        run_dir / "training_label_open.json",
        schema=RUN_SCHEMA + "-training-label-open",
        expected={
            "role": "train",
            "controls_passed": 1,
            "validation_labels_opened": 0,
            "confirmation_labels_opened": 0,
        },
    )
    authorization = _memory.LabelOpenAuthorization(
        train_input.cache_root,
        "train",
        "physical_training",
        seal["semantic_sha256"],
    )
    labels = _memory.open_external_label_store(
        train_input.cache_root, "train", authorization=authorization
    )
    if (
        labels.row_count != train_input.row_count
        or not np.array_equal(
            labels.row_array("sample_key"), train_input.row_array("sample_key")
        )
        or not np.array_equal(
            labels.row_array("path_id"), train_input.row_array("path_id")
        )
    ):
        raise ArtifactCompatibilityError("training input/label join changed")
    return labels


def _run_physical_candidate(
    run_dir: Path,
    *,
    train: _memory.HostInputStore,
    labels: _memory.HostLabelStore,
    target_scale: float,
    seed: int,
    maximum_updates: int,
    device: torch.device,
    guard: _memory.ModelCallBatchGuard,
) -> dict[str, Any]:
    enable_deterministic_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    fingerprint = config_fingerprint(
        {
            "schema": RUN_SCHEMA + "-physical-candidate-generator",
            "seed": int(seed),
            "target_scale": float(target_scale),
            "maximum_updates": int(maximum_updates),
            "checkpoint_interval": TRAINING["checkpoint_interval"],
            "external_train_index_sha256": config_fingerprint(dict(train.index)),
            "scientific_config_sha256": _load_json(
                run_dir / "scientific_config.json"
            )["semantic_sha256"],
            "memory_schedule_sha256": _load_json(
                run_dir / "training_memory_schedule.json"
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
        payload = torch.load(progress_path, map_location=device, weights_only=False)
        if payload.get("fingerprint") != fingerprint:
            raise ArtifactCompatibilityError("physical training fingerprint changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed = int(payload["completed_update"])
        checkpoint_records = [dict(row) for row in payload["checkpoint_records"]]
        history = [dict(row) for row in payload["history"]]
        finite = bool(payload["finite"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(list(payload["cuda_rng_states"]))

    def save_candidate(update: int) -> dict[str, Any]:
        state = _clone_state_dict(model)
        state_hash = state_dict_sha256(state)
        path = (
            run_dir
            / "checkpoints"
            / "physical"
            / f"seed-{seed}"
            / f"update-{update:04d}.pt"
        )
        artifact = _v3._atomic_torch(
            path,
            {
                "schema": _v3.RUN_SCHEMA + "-physical-candidate",
                "schema_version": 1,
                "fingerprint": fingerprint,
                "seed": int(seed),
                "update": int(update),
                "state_dict": state,
                "state_sha256": state_hash,
                "zero_baseline_sha256": ZERO_BASELINE_SHA256,
                "training_only": 1,
                "validation_evidence_used": 0,
                "memory_schedule_sha256": _load_json(
                    run_dir / "training_memory_schedule.json"
                )["semantic_sha256"],
            },
        )
        row = {
            "seed": int(seed),
            "update": int(update),
            "training_fingerprint": fingerprint,
            "state_sha256": state_hash,
            "checkpoint_path": path.relative_to(run_dir).as_posix(),
            "checkpoint_file_sha256": artifact["sha256"],
            "finite": 1,
        }
        checkpoint_records.append(row)
        return row

    def save_progress(update: int) -> None:
        _v3._atomic_torch(
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
        first = train.batch(
            np.arange(min(MAXIMUM_FORWARD_BATCH, train.row_count), dtype=np.int64),
            device=device,
        )
        with torch.no_grad():
            if not bool(torch.all(guard.call(model, first) == 0.0)):
                raise MemoryConfirmationError(
                    "physical update zero is not exact zero",
                    failure_domain="training_memory_schedule",
                    failure_code="physical_update_zero_invalid",
                )
        candidate = save_candidate(0)
        candidate["update_zero_prediction_exact"] = 1
        save_progress(0)
    for update in range(completed + 1, maximum_updates + 1):
        if not finite:
            break
        rows = deterministic_batch_indices(
            train.row_count, TRAINING["batch_size"], update - 1, int(seed)
        )
        inputs = train.batch(rows, device=device)
        target = labels.target_batch(rows, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = guard.call(model, inputs)
        loss, raw = direct_raw_target_mse(prediction, target, target_scale)
        if not bool(torch.isfinite(loss)):
            finite = False
            save_progress(update - 1)
            break
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), TRAINING["gradient_norm_clip"]
        )
        if not math.isfinite(float(gradient)):
            finite = False
            save_progress(update - 1)
            break
        optimizer.step()
        if update % TRAINING["checkpoint_interval"] == 0 or update == maximum_updates:
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
                f"memory-safe v3 seed={seed} update={update}/{maximum_updates} "
                f"train_mse={float(raw.detach().cpu()):.8g}",
                flush=True,
            )
    expected_count = maximum_updates // TRAINING["checkpoint_interval"] + 1
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
            "checkpoint_interval": TRAINING["checkpoint_interval"],
            "checkpoint_count": len(checkpoint_records),
            "checkpoints": checkpoint_records,
            "training_fingerprint": fingerprint,
            "target_scale": float(target_scale),
            "validation_inputs_received": 0,
            "validation_labels_received": 0,
            "pointwise_selection_performed": 0,
            "physical_training_performed": 1,
            **NO_WORK,
        }
    )
    atomic_write_json(
        run_dir / "checkpoints" / "physical" / f"seed-{seed}-task.json",
        report,
    )
    atomic_write_csv(
        run_dir / "checkpoints" / "physical" / f"seed-{seed}-history.csv",
        history,
    )
    return report


def _training_memory_fields(
    guard: _memory.ModelCallBatchGuard,
    *,
    peak_memory_fraction: float,
    committed_guard_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = guard.record()
    record = (
        dict(committed_guard_record)
        if isinstance(committed_guard_record, Mapping)
        and int(committed_guard_record.get("call_count", 0))
        >= int(current["call_count"])
        else current
    )
    return {
        "host_backed_batches_valid": 1,
        "maximum_forward_batch_enforced": int(
            record["maximum_observed_batch_size"] <= MAXIMUM_FORWARD_BATCH
        ),
        "full_cache_cuda_tensor_absent": 1,
        "streaming_reducer_valid": 1,
        "label_firewall_valid": 1,
        "memory_diagnostics_complete": 1,
        "maximum_observed_model_forward_batch_size": record[
            "maximum_observed_batch_size"
        ],
        "full_cache_cuda_tensor_count": 0,
        "peak_memory_fraction": float(peak_memory_fraction),
        "model_call_count": record["call_count"],
        "model_call_batch_hash": record["observed_batch_sizes_sha256"],
    }


def _adapter_progress_path(run_dir: Path, stage: str) -> Path:
    if stage not in {"select", "confirm"}:
        raise ValueError(f"unsupported adapter stage: {stage}")
    prefix = "selection" if stage == "select" else "confirmation"
    return run_dir / f"{prefix}_memory_progress.json"


def _load_adapter_progress(run_dir: Path, stage: str) -> dict[str, Any] | None:
    path = _adapter_progress_path(run_dir, stage)
    if not path.is_file():
        return None
    return _verified_semantic_record(
        path,
        schema=RUN_SCHEMA + f"-{stage}-memory-progress",
        expected={"stage": stage},
    )


def _commit_adapter_progress(
    run_dir: Path,
    *,
    stage: str,
    guard: _memory.ModelCallBatchGuard,
    base: Mapping[str, Any] | None,
    device: torch.device,
) -> dict[str, Any]:
    current = guard.record()
    previous = dict((base or {}).get("model_call_batches", {}))
    previous_calls = int(previous.get("call_count", 0))
    current_calls = int(current["call_count"])
    combined = {
        **current,
        "call_count": previous_calls + current_calls,
        "maximum_observed_batch_size": max(
            int(previous.get("maximum_observed_batch_size", 0)),
            int(current["maximum_observed_batch_size"]),
        ),
        "all_calls_within_limit": int(
            int(previous.get("all_calls_within_limit", 1)) == 1
            and int(current["all_calls_within_limit"]) == 1
        ),
        "observed_batch_sizes_sha256": config_fingerprint(
            {
                "previous": previous.get("observed_batch_sizes_sha256"),
                "current": current["observed_batch_sizes_sha256"],
                "previous_calls": previous_calls,
                "current_calls": current_calls,
            }
        ),
    }
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    total_bytes = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else 1
    )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + f"-{stage}-memory-progress",
            "schema_version": 1,
            "stage": stage,
            "model_call_batches": combined,
            "peak_memory_bytes": max(int((base or {}).get("peak_memory_bytes", 0)), peak_bytes),
            "device_total_memory_bytes": total_bytes,
            "commit_after_completed_prediction": 1,
            **NO_WORK,
        }
    )
    atomic_write_json(_adapter_progress_path(run_dir, stage), record)
    return record


def _train_stage_seal_names(run_dir: Path) -> tuple[str, ...]:
    if (run_dir / "train_execution_failure.json").is_file():
        return (
            "train_metrics.json",
            "train_execution_failure.json",
            "train_gate.json",
        )
    common = (
        "zero_initialization_control.json",
        "synthetic_teacher_control.json",
        "synthetic_teacher_per_path.csv",
        "exact_model_null_control.json",
        "training_memory_diagnostics.json",
        "train_metrics.json",
        "train_gate.json",
    )
    if not (run_dir / "training_label_open.json").is_file():
        return common
    return (
        "zero_initialization_control.json",
        "synthetic_teacher_control.json",
        "synthetic_teacher_per_path.csv",
        "exact_model_null_control.json",
        "training_label_open.json",
        "physical_training_started.json",
        "training_target_scale.json",
        "candidate_grid.json",
        "physical_seed_metrics.csv",
        "training_memory_diagnostics.json",
        "train_metrics.json",
        "train_gate.json",
    )


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("train requires a passing preflight gate")
    _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
    _binding(run_dir)
    _prepare_execution_retry(run_dir, "train")
    gate_path = run_dir / "train_gate.json"
    if gate_path.is_file():
        seal_path = run_dir / "train_artifact_seal.json"
        if seal_path.is_file():
            _verify_stage_seal(run_dir, seal_path.name)
        else:
            # A hard interruption between the gate and seal commits no new
            # scientific work.  Reconstruct only the missing commit marker
            # after all expected artifacts have independently verified hashes.
            _seal_stage(
                run_dir,
                _train_stage_seal_names(run_dir),
                seal_path.name,
            )
        return _load_json(gate_path)
    maximum_updates = (
        int(args.test_maximum_updates)
        if args.test_only
        else int(TRAINING["maximum_updates"])
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    parent = _external_root(run_dir)
    train = _memory.open_external_input_store(parent, "train")
    validation = _memory.open_external_input_store(parent, "validation")
    guard = _memory.ModelCallBatchGuard()
    controls = _run_prelabel_controls(
        run_dir,
        train=train,
        validation=validation,
        maximum_updates=maximum_updates,
        device=device,
        guard=guard,
    )
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    total_bytes = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else 1
    )
    peak_fraction = peak_bytes / max(total_bytes, 1)
    memory_fields = _training_memory_fields(
        guard, peak_memory_fraction=peak_fraction
    )
    synthetic = controls["synthetic_metrics"]
    null = controls["null_metrics"]
    if int(controls["passed"]) != 1:
        metrics = {
            "schema": RUN_SCHEMA + "-train-metrics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "zero_initialization_control_passed": int(
                controls["zero_metrics"]["passed"]
            ),
            "synthetic_teacher_passed": int(synthetic["passed"]),
            "synthetic_every_validation_path_beats_zero": int(
                synthetic["every_validation_path_beats_zero"]
            ),
            "exact_model_null_passed": int(null["passed"]),
            "null_selected_update_zero": int(null["selected_update"] == 0),
            "null_parameters_bitwise_unchanged": int(
                null["parameters_bitwise_unchanged"]
            ),
            "controls_before_training_label_open": 1,
            "synthetic_relative_validation_mse": float(
                synthetic["relative_validation_mse"]
            ),
            "physical_training_performed": 0,
            "validation_labels_opened": 0,
            "validation_selection_performed": 0,
            "confirmation_performed": 0,
            **memory_fields,
            **NO_WORK,
        }
        atomic_write_json(run_dir / "train_metrics.json", metrics)
        gate = _gate.evaluate_train_gate(metrics)
        atomic_write_json(gate_path, gate)
        atomic_write_json(
            run_dir / "training_memory_diagnostics.json",
            _semantic(
                {
                    "schema": RUN_SCHEMA + "-training-memory-diagnostics",
                    "schema_version": 1,
                    "peak_memory_bytes": peak_bytes,
                    "device_total_memory_bytes": total_bytes,
                    **memory_fields,
                }
            ),
        )
        _seal_stage(
            run_dir,
            _train_stage_seal_names(run_dir),
            "train_artifact_seal.json",
        )
        return gate

    if not (run_dir / "training_label_open.json").is_file():
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
    if not (run_dir / "physical_training_started.json").is_file():
        atomic_write_json(
            run_dir / "physical_training_started.json",
            {
                "schema": RUN_SCHEMA + "-physical-training-started",
                "schema_version": 1,
                "started_at": _now(),
                "physical_training_performed": 1,
                "validation_labels_opened": 0,
                **NO_WORK,
            },
        )
    del validation
    labels = _training_label_store(run_dir, train)
    reduction = _memory.canonical_row_square_reduction(
        labels.row_array("denoising_target")
    )
    target_scale = float(reduction["rms"])
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise MemoryConfirmationError(
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
                "canonical_reduction": reduction,
                "training_labels_only": 1,
                "validation_labels_used": 0,
                "confirmation_labels_used": 0,
                "quotient_target_formed": 0,
            }
        ),
    )
    reports = [
        _run_physical_candidate(
            run_dir,
            train=train,
            labels=labels,
            target_scale=target_scale,
            seed=seed,
            maximum_updates=maximum_updates,
            device=device,
            guard=guard,
        )
        for seed in _v3.MODEL_SEEDS
    ]
    grid = _v3._candidate_grid_from_reports(
        run_dir, reports, target_scale=target_scale
    )
    atomic_write_json(run_dir / "candidate_grid.json", grid)
    atomic_write_csv(
        run_dir / "physical_seed_metrics.csv",
        [
            {
                "seed": int(report["seed"]),
                "complete": int(report["complete"]),
                "finite": int(report["finite"]),
                "checkpoint_count": int(report["checkpoint_count"]),
            }
            for report in reports
        ],
    )
    expected_count = maximum_updates // TRAINING["checkpoint_interval"] + 1
    production_complete = all(
        int(report["complete"]) == 1
        and int(report["finite"]) == 1
        and int(report["checkpoint_count"]) == expected_count
        for report in reports
    )
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_fraction = peak_bytes / max(total_bytes, 1)
    memory_fields = _training_memory_fields(
        guard, peak_memory_fraction=peak_fraction
    )
    metrics = {
        "schema": RUN_SCHEMA + "-train-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
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
                "training_labels_opened_after_controls",
                "validation_labels_opened_zero",
                "validation_inputs_unavailable_to_physical_trainer",
                "pointwise_checkpoint_selection_performed_zero",
                "physical_task_records_selection_free",
                "training_only_target_scale_valid",
                "baseline_artifacts_absent",
                "confirmation_absent",
            )
        },
        "physical_training_complete": int(production_complete),
        "all_physical_tasks_complete_finite": int(production_complete),
        "fixed_checkpoint_grid_complete": int(production_complete),
        "candidate_grid_valid": int(production_complete),
        "synthetic_relative_validation_mse": float(
            synthetic["relative_validation_mse"]
        ),
        "model_seed_count": len(_v3.MODEL_SEEDS),
        "checkpoint_count": int(grid["checkpoint_count"]),
        "nonzero_candidate_count": int(grid["nonzero_candidate_count"]),
        "maximum_updates": maximum_updates,
        "physical_training_performed": 1,
        "validation_labels_opened": 0,
        "pointwise_checkpoint_selection_performed": 0,
        "validation_selection_performed": 0,
        "confirmation_performed": 0,
        **memory_fields,
        **NO_WORK,
    }
    atomic_write_json(
        run_dir / "training_memory_diagnostics.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-training-memory-diagnostics",
                "schema_version": 1,
                "peak_memory_bytes": peak_bytes,
                "device_total_memory_bytes": total_bytes,
                **memory_fields,
            }
        ),
    )
    atomic_write_json(run_dir / "train_metrics.json", metrics)
    if args.test_only:
        gate = {
            "schema": TEST_RUN_SCHEMA + "-train-gate",
            "evaluation_status": "evaluated",
            "passed": int(production_complete),
            "scientific_evidence_complete": 1,
            "authorizing": 0,
            **_scope(run_dir),
        }
    else:
        gate = _gate.evaluate_train_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        _train_stage_seal_names(run_dir),
        "train_artifact_seal.json",
    )
    return gate


def _predict_model_inputs_to_cpu(
    model: nn.Module,
    inputs: ModelInputs | _memory.HostInputStore,
    *,
    device: torch.device,
    guard: _memory.ModelCallBatchGuard,
    batch_size: int = MAXIMUM_FORWARD_BATCH,
) -> Tensor:
    if isinstance(inputs, _memory.HostInputStore):
        values, _ = _memory.predict_to_cpu(
            model,
            inputs,
            device=device,
            guard=guard,
            batch_size=batch_size,
        )
        return torch.from_numpy(values)
    size = int(inputs.batch_size)
    output = np.empty((size, _v3.EDGES_PER_PHASE), dtype=np.float64)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, size, batch_size):
                stop = min(size, start + batch_size)
                index = torch.arange(
                    start,
                    stop,
                    dtype=torch.long,
                    device=inputs.later_full_state.device,
                )
                batch = inputs.index_select(index)
                if batch.later_full_state.device != device:
                    batch = batch.to(device)
                prediction = guard.call(model, batch).to(torch.float64)
                output[start:stop] = prediction.detach().cpu().numpy()
    finally:
        model.train(was_training)
    return torch.from_numpy(output)


@contextmanager
def _v3_memory_adapter(
    run_dir: Path,
    args: argparse.Namespace,
    guard: _memory.ModelCallBatchGuard,
    *,
    stage: str,
) -> Iterator[None]:
    parent = _external_root(run_dir)
    device = torch.device(args.device)
    base_progress = _load_adapter_progress(run_dir, stage)
    original_loader = _v3._load_validation_evidence
    original_predict = _v3._predict_in_batches
    original_model_input_builder = _v3._legacy._model_inputs_from_arrays

    def load_validation(
        _child: Path, _args: argparse.Namespace
    ) -> tuple[dict[str, np.ndarray], _memory.HostInputStore, np.ndarray, dict[str, Any]]:
        inputs = _memory.open_external_input_store(parent, "validation")
        search_plan = _load_json(run_dir / "validation_search_plan.json")
        candidate_grid = _load_json(run_dir / "candidate_grid.json")
        open_record = _verified_semantic_record(
            run_dir / "validation_label_open.json",
            schema=_v3.RUN_SCHEMA + "-validation-label-open",
            expected={
                "search_plan_sha256": search_plan["semantic_sha256"],
                "candidate_grid_sha256": candidate_grid["semantic_sha256"],
                "count_shards_committed": 1,
                "confirmation_namespace_opened": 0,
            },
        )
        authorization = _memory.LabelOpenAuthorization(
            parent,
            "validation",
            "validation_selection",
            open_record["semantic_sha256"],
        )
        labels = _memory.open_external_label_store(
            parent, "validation", authorization=authorization
        )
        if (
            inputs.row_count != labels.row_count
            or not np.array_equal(
                inputs.row_array("sample_key"), labels.row_array("sample_key")
            )
            or not np.array_equal(
                inputs.row_array("path_id"), labels.row_array("path_id")
            )
        ):
            raise ArtifactCompatibilityError("validation input/label join changed")
        return (
            dict(inputs.arrays),
            inputs,
            np.array(labels.row_array("denoising_target"), copy=True, order="C"),
            dict(inputs.index),
        )

    def predict(
        model: nn.Module,
        inputs: ModelInputs | _memory.HostInputStore,
        *,
        batch_size: int = MAXIMUM_FORWARD_BATCH,
    ) -> Tensor:
        result = _predict_model_inputs_to_cpu(
            model,
            inputs,
            device=device,
            guard=guard,
            batch_size=min(int(batch_size), MAXIMUM_FORWARD_BATCH),
        )
        _commit_adapter_progress(
            run_dir,
            stage=stage,
            guard=guard,
            base=base_progress,
            device=device,
        )
        return result

    def build_confirmation_host_inputs(
        arrays: Mapping[str, np.ndarray], _device: torch.device
    ) -> _memory.HostInputStore:
        # The frozen confirmation helper materializes one cohort's 7x8
        # midpoint rows at once (up to 560 rows).  Preserve its exact arrays,
        # but defer every host-to-device transfer to the guarded 32-row
        # predictor.  ``validation`` is the only non-training host-store role
        # in the additive primitive; this transient store opens no labels and
        # is never persisted or exposed to selection.
        return _memory.HostInputStore.from_arrays(
            arrays,
            role="validation",
            cache_root=run_dir,
            index={},
        )

    _v3._load_validation_evidence = load_validation
    _v3._predict_in_batches = predict
    if stage == "confirm":
        _v3._legacy._model_inputs_from_arrays = build_confirmation_host_inputs
    try:
        yield
    finally:
        _v3._load_validation_evidence = original_loader
        _v3._predict_in_batches = original_predict
        _v3._legacy._model_inputs_from_arrays = original_model_input_builder


def _select_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "train_gate.json")):
        raise ArtifactCompatibilityError("select requires a passing train gate")
    _verify_stage_seal(run_dir, "train_artifact_seal.json")
    _binding(run_dir)
    _prepare_execution_retry(run_dir, "select")
    gate_path = run_dir / "select_gate.json"
    if gate_path.is_file() and (run_dir / "selection_memory_diagnostics.json").is_file():
        try:
            sealed = _stage_seal_paths(run_dir, "selection_artifact_seal.json")
        except ArtifactCompatibilityError:
            sealed = set()
        if {
            "selection_memory_diagnostics.json",
            "selection_legacy_stage_snapshot.json",
        }.issubset(sealed):
            return _load_json(gate_path)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    guard = _memory.ModelCallBatchGuard()
    snapshot_path = run_dir / "selection_legacy_stage_snapshot.json"
    if not snapshot_path.is_file():
        with _v3_memory_adapter(run_dir, args, guard, stage="select"):
            _v3._select_stage(run_dir, args)
    legacy_metrics, legacy_gate = _prepare_legacy_stage_snapshot(
        run_dir,
        stage="select",
        seal_name="selection_artifact_seal.json",
        metrics_name="select_metrics.json",
        gate_name="select_gate.json",
    )
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    total_bytes = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else 1
    )
    progress = _load_adapter_progress(run_dir, "select")
    if progress is not None:
        peak_bytes = max(peak_bytes, int(progress["peak_memory_bytes"]))
        total_bytes = int(progress["device_total_memory_bytes"])
    fields = _training_memory_fields(
        guard,
        peak_memory_fraction=peak_bytes / max(total_bytes, 1),
        committed_guard_record=(progress or {}).get("model_call_batches"),
    )
    diagnostics = _semantic(
        {
            "schema": RUN_SCHEMA + "-selection-memory-diagnostics",
            "schema_version": 1,
            "peak_memory_bytes": peak_bytes,
            "device_total_memory_bytes": total_bytes,
            "prediction_outputs_accumulated_on_cpu": 1,
            **fields,
        }
    )
    atomic_write_json(run_dir / "selection_memory_diagnostics.json", diagnostics)
    metrics = dict(legacy_metrics)
    metrics.update(fields)
    metrics["immutable_parent_cache_reused"] = 1
    metrics["production_cache_generation_performed"] = 0
    atomic_write_json(run_dir / "select_metrics.json", metrics)
    gate = (
        _gate.evaluate_select_gate(metrics)
        if not args.test_only
        else {
            **legacy_gate,
            "authorizing": 0,
        }
    )
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
        "selection_memory_diagnostics.json",
        "selection_legacy_metrics.json",
        "selection_legacy_gate.json",
        "selection_legacy_stage_snapshot.json",
    ]
    names.append(
        "checkpoint_selection.json"
        if (run_dir / "checkpoint_selection.json").is_file()
        else "no_validation_candidate.json"
    )
    if _adapter_progress_path(run_dir, "select").is_file():
        names.append("selection_memory_progress.json")
    _seal_stage(run_dir, names, "selection_artifact_seal.json")
    return gate


def _coarse_parent_from_binding(run_dir: Path) -> Path:
    parent_provenance = _load_json(
        _external_root(run_dir) / "parent_provenance.json"
    )
    return Path(
        parent_provenance["parents"]["coarse_residual"]["run_dir"]
    ).resolve()


def _confirm_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "select_gate.json")):
        raise ArtifactCompatibilityError("confirm requires a passing select gate")
    _verify_stage_seal(run_dir, "selection_artifact_seal.json")
    _binding(run_dir)
    _prepare_execution_retry(run_dir, "confirm")
    gate_path = run_dir / "confirm_gate.json"
    if gate_path.is_file() and (run_dir / "confirmation_memory_diagnostics.json").is_file():
        try:
            sealed = _stage_seal_paths(run_dir, "confirm_artifact_seal.json")
        except ArtifactCompatibilityError:
            sealed = set()
        if {
            "confirmation_memory_diagnostics.json",
            "confirmation_legacy_stage_snapshot.json",
        }.issubset(sealed):
            return _load_json(gate_path)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    guard = _memory.ModelCallBatchGuard()
    args.parent_coarse_residual_run_dir = _coarse_parent_from_binding(run_dir)
    snapshot_path = run_dir / "confirmation_legacy_stage_snapshot.json"
    if not snapshot_path.is_file():
        with _v3_memory_adapter(run_dir, args, guard, stage="confirm"):
            _v3._confirm_stage(run_dir, args)
    legacy_metrics, legacy_gate = _prepare_legacy_stage_snapshot(
        run_dir,
        stage="confirm",
        seal_name="confirm_artifact_seal.json",
        metrics_name="confirmation_metrics.json",
        gate_name="confirm_gate.json",
    )
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    total_bytes = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else 1
    )
    progress = _load_adapter_progress(run_dir, "confirm")
    if progress is not None:
        peak_bytes = max(peak_bytes, int(progress["peak_memory_bytes"]))
        total_bytes = int(progress["device_total_memory_bytes"])
    metrics = dict(legacy_metrics)
    peak_fraction = max(
        float(metrics.get("peak_memory_fraction", 0.0)),
        peak_bytes / max(total_bytes, 1),
    )
    fields = _training_memory_fields(
        guard,
        peak_memory_fraction=peak_fraction,
        committed_guard_record=(progress or {}).get("model_call_batches"),
    )
    diagnostics = _semantic(
        {
            "schema": RUN_SCHEMA + "-confirmation-memory-diagnostics",
            "schema_version": 1,
            "peak_memory_bytes": peak_bytes,
            "device_total_memory_bytes": total_bytes,
            **fields,
        }
    )
    atomic_write_json(
        run_dir / "confirmation_memory_diagnostics.json", diagnostics
    )
    metrics.update(fields)
    metrics["peak_memory_fraction"] = peak_fraction
    metrics["immutable_parent_cache_reused"] = 1
    metrics["production_cache_generation_performed"] = 0
    atomic_write_json(run_dir / "confirmation_metrics.json", metrics)
    gate = (
        _gate.evaluate_confirm_gate(metrics)
        if not args.test_only
        else {
            **legacy_gate,
            "authorizing": 0,
        }
    )
    atomic_write_json(run_dir / "confirmation_gate.json", gate)
    atomic_write_json(gate_path, gate)
    confirm_names = [
            "confirmation_namespace_open.json",
            "confirmation_bootstrap_count_index.json",
            "confirmation_execution.json",
            "confirmation_path_risks.npz",
            "confirmation_risk_summary.json",
            "confirmation_max_t.npz",
            "confirmation_max_t.json",
            "confirmation_metrics.json",
            "confirmation_memory_diagnostics.json",
            "confirmation_legacy_metrics.json",
            "confirmation_legacy_gate.json",
            "confirmation_legacy_stage_snapshot.json",
            "confirmation_gate.json",
            "confirmation_index.json",
            "confirm_gate.json",
        ]
    if _adapter_progress_path(run_dir, "confirm").is_file():
        confirm_names.append("confirmation_memory_progress.json")
    _seal_stage(run_dir, confirm_names, "confirm_artifact_seal.json")
    return gate


def _stage_gates(run_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage in ("preflight", "train", "select", "confirm"):
        path = run_dir / f"{stage}_gate.json"
        result[stage] = (
            _load_json(path)
            if path.is_file()
            else _gate.not_evaluated_gate(stage, "stage has not run")
        )
    return result


def _workflow_record(run_dir: Path, *, require_gate: str) -> dict[str, Any]:
    gates = _stage_gates(run_dir)
    workflow = _gate.evaluate_required_gate(
        preflight_gate=gates["preflight"],
        train_gate=gates["train"],
        select_gate=gates["select"],
        confirm_gate=gates["confirm"],
        require_gate=require_gate,
    )
    if int(_load_json(run_dir / "scientific_config.json").get("test_only", 0)) == 1:
        workflow["required_gate_pass"] = int(
            require_gate == "none" or _passed(gates.get(require_gate))
        )
        workflow["required_gate_exit_code"] = (
            0 if workflow["required_gate_pass"] else 1
        )
        workflow["authorizing"] = 0
    workflow["passed"] = int(workflow["required_gate_pass"])
    decision = dict(workflow["decision"])
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(
        run_dir / "boundary_tangent_v3_memory_decision.json", decision
    )
    evaluated = [
        stage
        for stage in ("preflight", "train", "select", "confirm")
        if gates[stage].get("evaluation_status") in {"evaluated", "execution_failed"}
    ]
    latest = evaluated[-1] if evaluated else None
    latest_gate = gates[latest] if latest else None
    if latest_gate is not None and not _passed(latest_gate):
        state = "gate_failed" if latest_gate.get("evaluation_status") == "evaluated" else "execution_failed"
    elif latest == "confirm":
        state = "complete"
    elif latest is not None:
        state = "ready_for_" + {
            "preflight": "train",
            "train": "select",
            "select": "confirm",
        }[latest]
    else:
        state = "initialized"
    _status(
        run_dir,
        state=state,
        stage="terminal",
        decision=str(decision.get("decision")),
        scientific_evidence_complete=int(
            (latest_gate or {}).get("scientific_evidence_complete", 0)
        ),
    )
    _artifact_registry(run_dir)
    return workflow


def _report_stage(run_dir: Path) -> None:
    for stage, seal in (
        ("preflight", "preflight_artifact_seal.json"),
        ("train", "train_artifact_seal.json"),
        ("select", "selection_artifact_seal.json"),
        ("confirm", "confirm_artifact_seal.json"),
    ):
        gate = run_dir / f"{stage}_gate.json"
        if gate.is_file():
            _verify_stage_seal(run_dir, seal)
            if not _passed(_load_json(gate)):
                break
    _binding(run_dir)


def _commit_initialization_failure(
    run_dir: Path,
    *,
    error: BaseException,
    require_gate: str,
) -> None:
    """Commit readable fail-closed provenance evidence for a fresh child.

    This function is never called for an existing resume directory.  Resume
    compatibility errors must leave that directory byte-for-byte untouched.
    """

    domain = str(getattr(error, "failure_domain", "control_provenance"))
    if domain not in {"control_provenance", "immutable_cache_binding"}:
        domain = "control_provenance"
    code = str(
        getattr(error, "failure_code", "memory_confirmation_initialization_failed")
    )
    metrics = {
        "schema": RUN_SCHEMA + "-initialization-execution-failure",
        "schema_version": 1,
        "evaluation_status": "execution_failed",
        "failure_domain": domain,
        "failure_code": code,
        "message": str(error),
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "parent_directory_mutated": 0,
        **_scope(run_dir),
    }
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    atomic_write_json(run_dir / "initialization_failure.json", metrics)
    gate = _gate.evaluate_preflight_gate(metrics)
    gate.update(
        {
            "evaluation_status": "execution_failed",
            "passed": 0,
            "failure_domain": domain,
            "failure_code": code,
            "stage_execution_valid": 0,
            "scientific_evidence_complete": 0,
            "provenance_valid": int(domain != "control_provenance"),
            "immutable_cache_binding_valid": int(
                domain != "immutable_cache_binding"
            ),
        }
    )
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "preflight_metrics.json",
            "initialization_failure.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )
    workflow = _gate.evaluate_required_gate(
        preflight_gate=gate,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
        require_gate=require_gate,
    )
    decision = dict(workflow["decision"])
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(
        run_dir / "boundary_tangent_v3_memory_decision.json", decision
    )
    _status(
        run_dir,
        state="execution_failed",
        stage="initialize",
        decision=str(decision.get("decision")),
        message=str(error),
        failure_domain=domain,
        failure_code=code,
        scientific_evidence_complete=0,
    )
    _artifact_registry(run_dir)


def _commit_execution_failure(
    run_dir: Path, *, stage: str, error: BaseException
) -> None:
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
        domain = "training_memory_resource"
        code = f"{stage}_memory_out_of_memory"
    elif isinstance(error, _memory.StreamingMemoryError):
        domain = "training_memory_schedule"
        code = f"{stage}_memory_schedule_invalid"
    elif isinstance(error, ArtifactCompatibilityError):
        domain = "control_provenance"
        code = f"{stage}_compatibility_invalid"
    else:
        domain = str(getattr(error, "failure_domain", "workflow_execution"))
        code = str(
            getattr(error, "failure_code", f"memory_confirmation_{stage}_failed")
        )
    metrics_name = {
        "preflight": "preflight_metrics.json",
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
    atomic_write_json(run_dir / f"{stage}_execution_failure.json", metrics)
    evaluator = {
        "preflight": _gate.evaluate_preflight_gate,
        "train": _gate.evaluate_train_gate,
        "select": _gate.evaluate_select_gate,
        "confirm": _gate.evaluate_confirm_gate,
    }[stage]
    gate = evaluator(metrics)
    atomic_write_json(run_dir / f"{stage}_gate.json", gate)
    if stage == "confirm":
        atomic_write_json(run_dir / "confirm_gate.json", gate)
    seal_name = {
        "preflight": "preflight_artifact_seal.json",
        "train": "train_artifact_seal.json",
        "select": "selection_artifact_seal.json",
        "confirm": "confirm_artifact_seal.json",
    }[stage]
    names = [metrics_name, f"{stage}_execution_failure.json", f"{stage}_gate.json"]
    if stage == "confirm":
        names.append("confirm_gate.json")
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


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "train", "select", "confirm")
    if stage == "report":
        return ()
    return (stage,)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Immutable-cache streaming-memory continuation of boundary-tangent v3"
    )
    parser.add_argument("--stage", choices=STAGES, default="report")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--failed-v3-train-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation"
        ),
    )
    parser.add_argument("--run-name", default="production-zero-baseline-v3-memory-safe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-maximum-updates", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--test-bootstrap-replicates", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--test-outer-steps", type=int, default=16, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("failed_v3_train_run_dir", "resume_run_dir", "runs_root"):
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
        if args.test_maximum_updates < 0:
            parser.error("test maximum updates must be nonnegative")
        if args.test_bootstrap_replicates < 2:
            parser.error("test bootstrap replicates must be at least two")
    elif args.device != "cuda":
        parser.error("production memory-confirmation stages require --device cuda")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    print(f"memory-safe v3 run directory: {run_dir}", flush=True)
    active_stage = "initialize"
    try:
        _initialize(run_dir, args, resumed=resumed)
        if args.stage != "report":
            configure_exact_torch_backend(args.device)
        functions = {
            "preflight": _preflight_stage,
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
        decision = _load_json(run_dir / "boundary_tangent_v3_memory_decision.json")
        print(f"memory-safe v3 decision: {decision.get('decision')}", flush=True)
        if stage_failed:
            return 2
        return 0 if int(workflow.get("passed", 0)) == 1 else 1
    except Exception as exc:
        if (
            active_stage in {"preflight", "train", "select", "confirm"}
            and run_dir.is_dir()
            and (run_dir / "scientific_config.json").is_file()
        ):
            _commit_execution_failure(run_dir, stage=active_stage, error=exc)
        elif active_stage == "initialize" and not resumed and run_dir.is_dir():
            _commit_initialization_failure(
                run_dir,
                error=exc,
                require_gate=args.require_gate,
            )
        label = "compatibility error" if isinstance(exc, ArtifactCompatibilityError) else "error"
        print(f"memory-safe v3 {label}: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
