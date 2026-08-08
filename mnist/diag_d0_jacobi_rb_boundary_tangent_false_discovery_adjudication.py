"""Read-only adjudication of the sealed boundary-tangent false discovery.

The workflow verifies the immutable eager-v2 parent, reconstructs the omitted
baseline-versus-zero confirmation evidence, and replays the complete fixed
validation checkpoint search.  It creates no transition, path, optimizer
update, confirmation label, controller trajectory, reconstruction, or sample.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

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
    BoundaryTangentPredictor,
    load_tangent_baseline,
)
from mnist.d0_jacobi_rb_boundary_tangent_confirmation import (
    CONFIRMATION_FAMILY_NAMES,
    aggregate_confirmation_improvements,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import (
    load_eager_role_inputs,
    load_eager_role_labels,
)
from mnist.d0_jacobi_rb_boundary_tangent_false_discovery import (
    BASELINE_FAMILY_NAMES,
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    HISTORICAL_NONZERO_UPDATES,
    HISTORICAL_SEEDS,
    HISTORICAL_SELECTED_SEED,
    HISTORICAL_SELECTED_UPDATE,
    aggregate_validated_three_contrasts,
    build_candidate_validation_table,
    candidate_directional_screens,
    classify_candidate_audit,
    classify_sealed_baseline,
    replay_historical_selection,
    require_exact_confirmation_replay,
    search_aware_candidate_max_t,
    two_sided_baseline_max_abs_t,
    validate_three_contrast_rows,
)
from mnist.d0_jacobi_rb_boundary_tangent_false_discovery_gate import (
    CANDIDATE_REPLAY_FLAGS,
    PREFLIGHT_FLAGS,
    REQUIRED_GATES,
    BASELINE_REPLAY_FLAGS,
    decide_workflow,
    evaluate_adjudication_gate,
    evaluate_baseline_gate,
    evaluate_candidate_gate,
    evaluate_decision_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_cache import SELECTED_OUTER_STEPS
from mnist import diag_d0_jacobi_rb_boundary_tangent_controller_confirmation as _trainer
from mnist import diag_d0_jacobi_rb_boundary_tangent_eager_confirmation as _parent


RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-"
    "false-discovery-adjudication-v1"
)
STAGES = ("preflight", "adjudicate", "decision", "report", "all")
PARENT_SOURCE_FINGERPRINT = (
    "dfe9c3357c1d1ba614cccfdcaca84b3c3bf2d0967d6a3a3b15e5a0421d04243e"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "fadc1eb31ad0fb1ccb900f41f1eb8523c67c6ae39e09c783698aa5a20634cdec"
)
PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "36bf43c0a108549954617a78625d4fd65820141c950ba84330133de1f8648580"
)
PARENT_REGISTRY_ARTIFACT_COUNT = 3_457
PARENT_REGISTRY_FILE_SHA256 = (
    "c996bdce5935667d247b6ce24c5e88f008c6038ec42ae25b9ea74b8b64a9a0d4"
)
VALIDATION_REPLAY_TOLERANCE = 1.0e-12
THREE_CONTRAST_TOLERANCE = 5.0e-15
BOOTSTRAP_SEED = DEFAULT_BOOTSTRAP_SEED
BASELINE_BOOTSTRAP_NAMESPACE = 1
CANDIDATE_BOOTSTRAP_NAMESPACE = 2
MAXIMUM_ADJUDICATION_SECONDS = 12.0 * 60.0 * 60.0
MAXIMUM_PEAK_MEMORY_FRACTION = 0.80
MAXIMUM_PERSISTED_BYTES = 512 * 1024**2
PREDICTION_BATCH_SIZE = 32

NO_WORK = {
    "new_exact_transitions": 0,
    "new_path_ids": 0,
    "optimizer_updates": 0,
    "confirmation_label_writes": 0,
    "parent_mutations": 0,
    "production_cache_generation_performed": 0,
    "physical_training_performed": 0,
    "new_confirmation_performed": 0,
    "controller_control_trajectory_performed": 0,
    "full_reverse_path_performed": 0,
    "reconstruction_performed": 0,
    "reverse_sampling_performed": 0,
    "sampling_performed": 0,
}

_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "false_discovery_decision.json",
}


class FalseDiscoveryAdjudicationError(RuntimeError):
    """Typed fail-closed workflow error."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "implementation_or_replay",
        failure_code: str = "false_discovery_adjudication_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


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


def _passed(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("evaluation_status") == "evaluated"
        and int(value.get("passed", 0)) == 1
    )


def _semantic_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _verify_semantic(record: Mapping[str, Any], message: str) -> None:
    body = dict(record)
    semantic = body.pop("semantic_sha256", None)
    if semantic != config_fingerprint(body):
        raise ArtifactCompatibilityError(message)


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> dict[str, Any]:
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
                **{name: np.ascontiguousarray(value) for name, value in arrays.items()},
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


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactCompatibilityError(f"cannot load NPZ artifact: {path}") from exc


def _scope() -> dict[str, int]:
    return dict(NO_WORK)


def _resource_limits_passed(
    *, elapsed_seconds: float, peak_memory_fraction: float, persisted_bytes: int
) -> bool:
    return bool(
        math.isfinite(float(elapsed_seconds))
        and 0.0 <= float(elapsed_seconds) <= MAXIMUM_ADJUDICATION_SECONDS
        and math.isfinite(float(peak_memory_fraction))
        and 0.0 <= float(peak_memory_fraction) <= MAXIMUM_PEAK_MEMORY_FRACTION
        and isinstance(persisted_bytes, int)
        and not isinstance(persisted_bytes, bool)
        and 0 <= persisted_bytes <= MAXIMUM_PERSISTED_BYTES
    )


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    message: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
    scientific_evidence_complete: int = 0,
) -> None:
    atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": RUN_SCHEMA + "-status",
            "schema_version": 1,
            "state": state,
            "stage": stage,
            "decision": decision,
            "message": message,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "scientific_evidence_complete": int(scientific_evidence_complete),
            "updated_at": _now(),
            **_scope(),
        },
    )


def _seal_stage(run_dir: Path, names: Sequence[str], seal_name: str) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"stage artifact is missing: {name}")
        artifacts.append(
            {
                "path": name,
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    record = _semantic_record(
        {
            "schema": RUN_SCHEMA + "-stage-seal",
            "schema_version": 1,
            "artifacts": artifacts,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / seal_name, record)
    return record


def _verify_stage_seal(run_dir: Path, seal_name: str) -> None:
    seal = _load_json(run_dir / seal_name)
    _verify_semantic(seal, f"{seal_name} semantic hash changed")
    for item in seal.get("artifacts", []):
        path = run_dir / str(item["path"])
        if (
            not path.is_file()
            or int(item.get("size", -1)) != path.stat().st_size
            or item.get("sha256") != file_fingerprint(path)
        ):
            raise ArtifactCompatibilityError(f"sealed artifact changed: {path}")


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED:
            continue
        artifacts.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    record = _semantic_record(
        {
            "schema": RUN_SCHEMA + "-artifact-registry",
            "schema_version": 1,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_existing_registry(run_dir: Path) -> None:
    registry = _load_json(run_dir / "artifact_registry.json")
    _verify_semantic(registry, "child artifact registry semantic hash changed")
    expected = {
        str(item["path"]): (int(item["size"]), str(item["sha256"]))
        for item in registry.get("artifacts", [])
    }
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in _REGISTRY_EXCLUDED
    }
    if set(expected) != actual:
        raise ArtifactCompatibilityError("child artifact registry file set changed")
    for name, (size, digest) in expected.items():
        path = run_dir / name
        if path.stat().st_size != size or file_fingerprint(path) != digest:
            raise ArtifactCompatibilityError(f"child artifact changed: {name}")


def _verify_registered_prefix(run_dir: Path) -> None:
    """Verify the last committed registry while allowing new restart commits.

    A hard process termination can leave complete per-candidate commits newer
    than the last terminal registry.  Files already named by that registry are
    immutable; additive candidate commits are validated by their own paired
    NPZ/JSON records before reuse.
    """

    registry_path = run_dir / "artifact_registry.json"
    if not registry_path.is_file():
        return
    registry = _load_json(registry_path)
    _verify_semantic(registry, "child artifact registry semantic hash changed")
    for item in registry.get("artifacts", []):
        path = run_dir / str(item.get("path", ""))
        if (
            not path.is_file()
            or int(item.get("size", -1)) != path.stat().st_size
            or item.get("sha256") != file_fingerprint(path)
        ):
            raise ArtifactCompatibilityError(
                f"previously registered child artifact changed: {path}"
            )


def _source_set() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    values = set(_parent._source_set())
    values.update(
        {
            Path(__file__).resolve(),
            (directory / "d0_jacobi_rb_boundary_tangent_false_discovery.py").resolve(),
            (directory / "d0_jacobi_rb_boundary_tangent_false_discovery_gate.py").resolve(),
        }
    )
    if not all(path.is_file() for path in values):
        raise ArtifactCompatibilityError("false-discovery source closure is incomplete")
    return tuple(sorted(values, key=lambda path: path.as_posix()))


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "parent_run_dir": str(args.parent_run_dir),
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent_registry_semantic_sha256": PARENT_REGISTRY_SEMANTIC_SHA256,
        "parent_registry_artifact_count": PARENT_REGISTRY_ARTIFACT_COUNT,
        "historical_seeds": list(HISTORICAL_SEEDS),
        "nonzero_updates": list(HISTORICAL_NONZERO_UPDATES),
        "historical_selected_seed": HISTORICAL_SELECTED_SEED,
        "historical_selected_update": HISTORICAL_SELECTED_UPDATE,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "baseline_bootstrap_namespace": BASELINE_BOOTSTRAP_NAMESPACE,
        "candidate_bootstrap_namespace": CANDIDATE_BOOTSTRAP_NAMESPACE,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "simultaneous_confidence": DEFAULT_BOOTSTRAP_CONFIDENCE,
        "quantile_interpolation": "higher",
        "three_contrast_identity_tolerance": THREE_CONTRAST_TOLERANCE,
        "validation_record_replay_tolerance": VALIDATION_REPLAY_TOLERANCE,
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
        "maximum_adjudication_seconds": MAXIMUM_ADJUDICATION_SECONDS,
        "maximum_peak_memory_fraction": MAXIMUM_PEAK_MEMORY_FRACTION,
        "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
        "test_only": int(args.test_only),
        **_scope(),
    }
    return _semantic_record(record)


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
    config = _scientific_config(args)
    sources = _source_set()
    source_hash = source_fingerprint(sources)
    if resumed:
        existing = _load_json(run_dir / "scientific_config.json")
        manifest = _load_json(run_dir / "run_manifest.json")
        if (
            existing != config
            or manifest.get("source_fingerprint") != source_hash
            or manifest.get("scientific_config_sha256")
            != config["semantic_sha256"]
            or Path(str(manifest.get("parent_run_dir", ""))).resolve()
            != args.parent_run_dir
        ):
            raise ArtifactCompatibilityError("resume scientific configuration changed")
        _verify_registered_prefix(run_dir)
        return
    atomic_write_json(run_dir / "scientific_config.json", config)
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema": RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "parent_run_dir": str(args.parent_run_dir),
            "parent_registry_file_sha256": file_fingerprint(
                args.parent_run_dir / "artifact_registry.json"
            ),
            "scientific_config_sha256": config["semantic_sha256"],
            "source_fingerprint": source_hash,
            "source_paths": [str(path) for path in sources],
            "device": args.device,
            **_scope(),
        },
    )


def _verify_artifact_descriptor(root: Path, record: Mapping[str, Any]) -> Path:
    path = root / str(record.get("path", ""))
    if (
        not path.is_file()
        or int(record.get("size", -1)) != path.stat().st_size
        or record.get("sha256") != file_fingerprint(path)
    ):
        raise ArtifactCompatibilityError(f"indexed artifact changed: {path}")
    return path


def _candidate_records(parent: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_records: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    expected_updates = (0,) + HISTORICAL_NONZERO_UPDATES
    for seed in HISTORICAL_SEEDS:
        task_path = parent / "checkpoints" / "physical" / f"seed-{seed}-task.json"
        task = _load_json(task_path)
        candidates = [dict(value) for value in task.get("candidates", [])]
        if (
            task.get("task") != "physical"
            or int(task.get("seed", -1)) != seed
            or int(task.get("complete", 0)) != 1
            or int(task.get("finite", 0)) != 1
            or int(task.get("candidate_count", -1)) != len(expected_updates)
            or tuple(int(value.get("update", -1)) for value in candidates)
            != expected_updates
        ):
            raise ArtifactCompatibilityError("physical candidate task grid changed")
        for candidate in candidates:
            update = int(candidate.get("update", -1))
            expected = (
                parent
                / "checkpoints"
                / "physical"
                / f"seed-{seed}"
                / f"update-{update:04d}.pt"
            )
            if (
                candidate.get("checkpoint_path")
                != expected.relative_to(parent).as_posix()
                or candidate.get("checkpoint_file_sha256")
                != file_fingerprint(expected)
            ):
                raise ArtifactCompatibilityError("physical checkpoint hash changed")
        tasks.append(task)
        all_records.extend(candidates)
    return all_records, tasks


def _validate_index_artifacts(parent: Path, role: str) -> dict[str, Any]:
    index = _load_json(parent / "eager_cache" / f"{role}_index.json")
    _verify_semantic(index, f"{role} cache index semantic hash changed")
    for entry in index.get("entries", []):
        for name in ("metadata", "continuation_state", "branch_inputs", "branch_labels"):
            artifact = entry.get(name)
            if artifact is not None:
                _verify_artifact_descriptor(parent, artifact)
    return index


def _confirmation_index_audit(parent: Path) -> tuple[dict[str, Any], int]:
    index = _load_json(parent / "confirmation_index.json")
    _verify_semantic(index, "confirmation index semantic hash changed")
    risk_count = 0
    for entry in index.get("shards", []):
        cohort = int(entry["cohort_index"])
        start = int(entry["start_step"])
        directory = (
            parent
            / "confirmation"
            / "shards"
            / f"cohort-{cohort:03d}"
            / f"shard-{start:06d}"
        )
        metadata_path = directory / "metadata.json"
        if entry.get("metadata_sha256") != file_fingerprint(metadata_path):
            raise ArtifactCompatibilityError("confirmation shard metadata changed")
        metadata = _load_json(metadata_path)
        _verify_semantic(metadata, "confirmation shard semantic hash changed")
        if int(metadata.get("committed", 0)) != 1:
            raise ArtifactCompatibilityError("confirmation shard is uncommitted")
        state = directory / "continuation_state.npz"
        if (
            metadata.get("state_file_sha256") != file_fingerprint(state)
            or int(metadata.get("state_file_size", -1)) != state.stat().st_size
        ):
            raise ArtifactCompatibilityError("confirmation continuation state changed")
        risk_sha = metadata.get("risk_file_sha256")
        if risk_sha is not None:
            risk = directory / "path_risks.npz"
            if (
                risk_sha != file_fingerprint(risk)
                or int(metadata.get("risk_file_size", -1)) != risk.stat().st_size
            ):
                raise ArtifactCompatibilityError("confirmation risk shard changed")
            risk_count += 1
        if int(metadata.get("raw_confirmation_inputs_persisted", 0)) != 0 or int(
            metadata.get("raw_confirmation_labels_persisted", 0)
        ) != 0:
            raise ArtifactCompatibilityError("raw confirmation evidence was persisted")
    if index.get("risk_sha256") != file_fingerprint(
        parent / "confirmation_path_risks.npz"
    ) or index.get("max_t_sha256") != file_fingerprint(
        parent / "confirmation_max_t.json"
    ):
        raise ArtifactCompatibilityError("confirmation aggregate binding changed")
    return index, risk_count


def _preflight_metrics(parent: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _parent._verify_existing_registry(parent)
    registry = _load_json(parent / "artifact_registry.json")
    status = _load_json(parent / "run_status.json")
    manifest = _load_json(parent / "run_manifest.json")
    config = _load_json(parent / "scientific_config.json")
    if (
        status.get("decision") != "selection_false_discovery"
        or int(status.get("scientific_evidence_complete", 0)) != 1
        or manifest.get("source_fingerprint") != PARENT_SOURCE_FINGERPRINT
        or manifest.get("scientific_config_sha256")
        != PARENT_SCIENTIFIC_CONFIG_SHA256
        or config.get("semantic_sha256") != PARENT_SCIENTIFIC_CONFIG_SHA256
        or int(registry.get("artifact_count", -1))
        != PARENT_REGISTRY_ARTIFACT_COUNT
        or registry.get("semantic_sha256") != PARENT_REGISTRY_SEMANTIC_SHA256
        or file_fingerprint(parent / "artifact_registry.json")
        != PARENT_REGISTRY_FILE_SHA256
    ):
        raise ArtifactCompatibilityError("authoritative v2 parent identity changed")
    if source_fingerprint(_parent._source_set()) != PARENT_SOURCE_FINGERPRINT:
        raise ArtifactCompatibilityError("executed v2 source closure changed")
    _parent._verify_stage_seal(parent, "train_artifact_seal.json")
    _parent._verify_stage_seal(parent, "confirm_artifact_seal.json")
    selection = _parent._verify_training_selection(parent)
    confirmation_seal = _load_json(parent / "confirmation_seal.json")
    _verify_semantic(confirmation_seal, "confirmation seal semantic hash changed")
    records, tasks = _candidate_records(parent)
    validation = _validate_index_artifacts(parent, "validation")
    confirmation_index, risk_count = _confirmation_index_audit(parent)
    path_plan = _load_json(parent / "path_id_plan.json")
    _verify_semantic(path_plan, "parent path plan semantic hash changed")
    roles = {name: tuple(int(value) for value in values) for name, values in path_plan["roles"].items()}
    train, valid, confirm = map(set, (roles["train"], roles["validation"], roles["confirmation"]))
    if train & valid or train & confirm or valid & confirm:
        raise ArtifactCompatibilityError("parent path namespaces overlap")
    if tuple(int(value) for value in validation.get("path_ids", [])) != roles["validation"]:
        raise ArtifactCompatibilityError("validation path role changed")
    if tuple(int(value) for value in confirmation_index.get("path_ids", [])) != roles["confirmation"]:
        raise ArtifactCompatibilityError("confirmation path role changed")
    registry_paths = [str(item["path"]) for item in registry.get("artifacts", [])]
    raw_confirmation = [
        name
        for name in registry_paths
        if name.startswith("confirmation/")
        and ("label" in Path(name).name.lower() or "target" in Path(name).name.lower())
    ]
    if raw_confirmation:
        raise ArtifactCompatibilityError("raw confirmation targets are present")
    nonzero = [value for value in records if int(value["update"]) > 0]
    updates = sorted({int(value["update"]) for value in nonzero})
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        **{name: 1 for name in PREFLIGHT_FLAGS},
        "nonzero_candidate_count": len(nonzero),
        "model_seed_count": len(tasks),
        "checkpoints_per_seed": len(HISTORICAL_NONZERO_UPDATES),
        "first_nonzero_update": min(updates),
        "update_stride": updates[1] - updates[0],
        "last_nonzero_update": max(updates),
        "validation_path_count": len(roles["validation"]),
        "confirmation_path_count": len(roles["confirmation"]),
        "confirmation_metadata_count": len(confirmation_index["shards"]),
        "confirmation_risk_shard_count": risk_count,
        "validation_index_entry_count": len(validation["entries"]),
        "parent_terminal_decision": str(status["decision"]),
        "parent_source_fingerprint": str(manifest["source_fingerprint"]),
        "parent_scientific_config_sha256": str(config["semantic_sha256"]),
        "parent_registry_artifact_count": int(registry["artifact_count"]),
        "parent_registry_semantic_sha256": registry["semantic_sha256"],
        "parent_registry_file_sha256": file_fingerprint(parent / "artifact_registry.json"),
        "historical_selected_seed": int(selection["selected_seed"]),
        "historical_selected_update": int(selection["selected_update"]),
        **_scope(),
    }
    provenance = _semantic_record(
        {
            "schema": RUN_SCHEMA + "-parent-provenance",
            "schema_version": 1,
            "parent_run_dir": str(parent),
            "parent_source_fingerprint": manifest["source_fingerprint"],
            "parent_scientific_config_sha256": config["semantic_sha256"],
            "parent_registry_semantic_sha256": registry["semantic_sha256"],
            "parent_registry_file_sha256": file_fingerprint(parent / "artifact_registry.json"),
            "parent_registry_artifact_count": int(registry["artifact_count"]),
            "selection_semantic_sha256": selection["semantic_sha256"],
            "confirmation_seal_semantic_sha256": confirmation_seal["semantic_sha256"],
            "path_plan_semantic_sha256": path_plan["semantic_sha256"],
            "historical_v2_decision": status["decision"],
            "historical_v2_decision_remains_terminal": 1,
            **_scope(),
        }
    )
    return metrics, {
        "provenance": provenance,
        "records": records,
        "tasks": tasks,
        "roles": roles,
        "selection": selection,
    }


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "preflight_gate.json"
    if gate_path.is_file():
        gate = _load_json(gate_path)
        seal_path = run_dir / "preflight_artifact_seal.json"
        if seal_path.is_file():
            _verify_stage_seal(run_dir, seal_path.name)
        else:
            metrics = _load_json(run_dir / "preflight_metrics.json")
            if gate != evaluate_preflight_gate(metrics):
                raise ArtifactCompatibilityError(
                    "orphan preflight gate does not replay"
                )
            for name in ("parent_provenance.json", "candidate_plan.json", "role_firewall.json"):
                _verify_semantic(
                    _load_json(run_dir / name), f"orphan preflight artifact changed: {name}"
                )
            _seal_stage(
                run_dir,
                (
                    "parent_provenance.json",
                    "candidate_plan.json",
                    "role_firewall.json",
                    "preflight_metrics.json",
                    "preflight_gate.json",
                ),
                seal_path.name,
            )
        return gate
    try:
        metrics, evidence = _preflight_metrics(args.parent_run_dir)
        atomic_write_json(run_dir / "parent_provenance.json", evidence["provenance"])
        candidate_plan = _semantic_record(
            {
                "schema": RUN_SCHEMA + "-candidate-plan",
                "schema_version": 1,
                "seeds": list(HISTORICAL_SEEDS),
                "updates": list(HISTORICAL_NONZERO_UPDATES),
                "candidate_count": 120,
                "historical_selected_seed": HISTORICAL_SELECTED_SEED,
                "historical_selected_update": HISTORICAL_SELECTED_UPDATE,
                "checkpoint_records": [
                    {
                        "seed": int(value["seed"]),
                        "update": int(value["update"]),
                        "path": str(value["checkpoint_path"]),
                        "sha256": str(value["checkpoint_file_sha256"]),
                        "state_sha256": str(value["state_sha256"]),
                    }
                    for value in evidence["records"]
                    if int(value["update"]) > 0
                ],
                **_scope(),
            }
        )
        atomic_write_json(run_dir / "candidate_plan.json", candidate_plan)
        firewall = _semantic_record(
            {
                "schema": RUN_SCHEMA + "-role-firewall",
                "schema_version": 1,
                "train_path_ids": list(evidence["roles"]["train"]),
                "validation_path_ids": list(evidence["roles"]["validation"]),
                "confirmation_path_ids": list(evidence["roles"]["confirmation"]),
                "confirmation_paths_denied_to_replay": 1,
                "old_confirmation_paths_burned": 1,
                "new_path_ids": 0,
                **_scope(),
            }
        )
        atomic_write_json(run_dir / "role_firewall.json", firewall)
    except Exception as exc:
        metrics = {
            "schema": RUN_SCHEMA + "-preflight-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            "failure_domain": "forensic_evidence",
            "failure_code": str(getattr(exc, "failure_code", "forensic_evidence_invalid")),
            "error_type": type(exc).__name__,
            "error": str(exc),
            **_scope(),
        }
        atomic_write_json(run_dir / "parent_provenance.json", metrics)
        atomic_write_json(run_dir / "candidate_plan.json", metrics)
        atomic_write_json(run_dir / "role_firewall.json", metrics)
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    gate = evaluate_preflight_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "parent_provenance.json",
            "candidate_plan.json",
            "role_firewall.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )
    return gate


def _read_confirmation_rows(parent: Path) -> dict[str, np.ndarray]:
    index = _load_json(parent / "confirmation_index.json")
    blocks: list[dict[str, np.ndarray]] = []
    for entry in index["shards"]:
        cohort = int(entry["cohort_index"])
        start = int(entry["start_step"])
        directory = (
            parent
            / "confirmation"
            / "shards"
            / f"cohort-{cohort:03d}"
            / f"shard-{start:06d}"
        )
        metadata = _load_json(directory / "metadata.json")
        risk_sha = metadata.get("risk_file_sha256")
        if risk_sha is None:
            continue
        risk_path = directory / "path_risks.npz"
        if risk_sha != file_fingerprint(risk_path):
            raise ArtifactCompatibilityError("confirmation risk shard hash changed")
        block = _load_npz(risk_path)
        if set(block) != {
            "sample_keys",
            "path_ids",
            "outer_steps",
            "phases",
            "midpoint_indices",
            "combined_vs_zero",
            "combined_vs_baseline",
            "baseline_vs_zero",
        }:
            raise ArtifactCompatibilityError("confirmation risk shard fields changed")
        blocks.append(block)
    if len(blocks) != 224:
        raise ArtifactCompatibilityError("confirmation risk shard count changed")
    return {
        name: np.ascontiguousarray(np.concatenate([block[name] for block in blocks]))
        for name in blocks[0]
    }


def _baseline_adjudication(
    run_dir: Path, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = args.parent_run_dir
    confirmation_index = _load_json(parent / "confirmation_index.json")
    rows = _read_confirmation_rows(parent)
    validated = validate_three_contrast_rows(
        sample_keys=rows["sample_keys"],
        row_path_ids=rows["path_ids"],
        outer_steps=rows["outer_steps"],
        phases=rows["phases"],
        midpoint_indices=rows["midpoint_indices"],
        combined_vs_zero=rows["combined_vs_zero"],
        combined_vs_baseline=rows["combined_vs_baseline"],
        baseline_vs_zero=rows["baseline_vs_zero"],
        expected_path_ids=confirmation_index["path_ids"],
        selected_outer_steps=SELECTED_OUTER_STEPS,
    )
    tables = aggregate_validated_three_contrasts(validated)
    parent_table = _load_npz(parent / "confirmation_path_risks.npz")
    require_exact_confirmation_replay(
        tables.confirmation,
        parent_path_ids=parent_table["path_ids"],
        parent_path_values=parent_table["path_values"],
        parent_cell_counts=parent_table["cell_counts"],
    )
    parent_max_t = _load_json(parent / "confirmation_max_t.json")
    replay_point = tables.confirmation.to_record()["point_estimates"]
    if replay_point != parent_max_t.get("point_estimates"):
        raise FalseDiscoveryAdjudicationError(
            "parent confirmation point estimates do not replay exactly",
            failure_code="parent_confirmation_point_replay_invalid",
        )
    baseline_max_t = two_sided_baseline_max_abs_t(
        tables.baseline,
        confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
        replicates=int(args.bootstrap_replicates),
        seed=BOOTSTRAP_SEED,
        namespace=BASELINE_BOOTSTRAP_NAMESPACE,
    )
    classification = classify_sealed_baseline(baseline_max_t)
    baseline_record = tables.baseline.to_record()
    baseline_artifact = _atomic_npz(
        run_dir / "sealed_baseline_path_table.npz",
        path_ids=tables.baseline.path_ids,
        path_values=tables.baseline.path_values,
        cell_counts=tables.baseline.cell_counts,
    )
    atomic_write_json(run_dir / "sealed_baseline_path_table.json", baseline_record)
    atomic_write_json(run_dir / "sealed_baseline_max_abs_t.json", baseline_max_t)
    atomic_write_json(
        run_dir / "confirmation_reaggregation_replay.json",
        {
            "schema": RUN_SCHEMA + "-confirmation-reaggregation-replay",
            "schema_version": 1,
            "parent_228_replay_exact": 1,
            "parent_point_estimates_replay_exact": 1,
            "direct_derived_total_exact": 1,
            "maximum_three_contrast_identity_error": tables.maximum_identity_error,
            "sample_key_sha256": validated.sample_key_sha256,
            "row_count": validated.row_count,
            "baseline_path_table_artifact": baseline_artifact,
            **_scope(),
        },
    )
    standard_errors = np.asarray(list(baseline_max_t["standard_errors"].values()))
    metrics = {
        "schema": RUN_SCHEMA + "-baseline-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        **{name: 1 for name in BASELINE_REPLAY_FLAGS},
        "confirmation_path_count": tables.baseline.path_count,
        "parent_family_size": 228,
        "baseline_family_size": 229,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "simultaneous_confidence": DEFAULT_BOOTSTRAP_CONFIDENCE,
        "maximum_three_contrast_identity_error": tables.maximum_identity_error,
        "all_simultaneous_lower_bounds_positive": int(
            classification == "sealed_baseline_advantage_confirmed"
        ),
        "overall_and_four_quartile_upper_bounds_negative": int(
            classification == "sealed_baseline_harm_confirmed"
        ),
        "minimum_standard_error": float(np.min(standard_errors)),
        "maximum_standard_error": float(np.max(standard_errors)),
        "controller_planning_authorized": 0,
        **_scope(),
    }
    gate = evaluate_baseline_gate(metrics)
    atomic_write_json(run_dir / "sealed_baseline_metrics.json", metrics)
    atomic_write_json(run_dir / "sealed_baseline_gate.json", gate)
    return gate, metrics


def _candidate_output_paths(run_dir: Path, seed: int, update: int) -> tuple[Path, Path]:
    directory = run_dir / "candidate_replay" / f"seed-{seed}"
    return directory / f"update-{update:04d}.npz", directory / f"update-{update:04d}.json"


def _load_candidate_output(
    run_dir: Path,
    candidate: Mapping[str, Any],
    *,
    expected_paths: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    seed, update = int(candidate["seed"]), int(candidate["update"])
    npz_path, json_path = _candidate_output_paths(run_dir, seed, update)
    if not npz_path.is_file() and not json_path.is_file():
        return None
    if not npz_path.is_file() or not json_path.is_file():
        # NPZ is committed before its binding JSON.  Either orphan is a
        # child-owned, uncommitted tail and is safe to recompute exactly.
        npz_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        return None
    record = _load_json(json_path)
    _verify_semantic(record, "candidate replay semantic hash changed")
    if (
        int(record.get("seed", -1)) != seed
        or int(record.get("update", -1)) != update
        or record.get("checkpoint_file_sha256")
        != candidate.get("checkpoint_file_sha256")
        or record.get("checkpoint_state_sha256") != candidate.get("state_sha256")
        or record.get("path_table_sha256") != file_fingerprint(npz_path)
    ):
        raise ArtifactCompatibilityError("candidate replay binding changed")
    arrays = _load_npz(npz_path)
    if set(arrays) != {"path_ids", "path_values"}:
        raise ArtifactCompatibilityError("candidate replay fields changed")
    paths = np.asarray(expected_paths, dtype=np.int64)
    values = arrays["path_values"]
    if (
        arrays["path_ids"].dtype != np.dtype(np.int64)
        or not np.array_equal(arrays["path_ids"], np.sort(paths, kind="stable"))
        or values.dtype != np.dtype(np.float64)
        or values.shape != (paths.size, len(CONFIRMATION_FAMILY_NAMES))
        or not np.isfinite(values).all()
        or int(record.get("path_count", -1)) != paths.size
        or int(record.get("family_size", -1)) != len(CONFIRMATION_FAMILY_NAMES)
    ):
        raise ArtifactCompatibilityError("candidate replay table contract changed")
    return arrays["path_values"], record


def _evaluate_candidate(
    run_dir: Path,
    parent: Path,
    candidate: Mapping[str, Any],
    *,
    baseline: Any,
    model_inputs: Any,
    target: torch.Tensor,
    baseline_prediction: torch.Tensor,
    arrays: Mapping[str, np.ndarray],
    expected_paths: Sequence[int],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    cached = _load_candidate_output(
        run_dir, candidate, expected_paths=expected_paths
    )
    if cached is not None:
        return cached
    model = _trainer._load_candidate_model(parent, candidate, baseline, device)
    prediction = _trainer._predict_in_batches(
        model, model_inputs, batch_size=PREDICTION_BATCH_SIZE
    )
    squared = (target - prediction).square()
    baseline_squared = (target - baseline_prediction).square()
    zero_squared = target.square()
    zero_improvement = torch.mean(zero_squared - squared, dim=1).detach().cpu().numpy()
    baseline_improvement = (
        torch.mean(baseline_squared - squared, dim=1).detach().cpu().numpy()
    )
    table = aggregate_confirmation_improvements(
        sample_keys=arrays["sample_key"],
        row_path_ids=arrays["path_id"],
        outer_steps=arrays["outer_step"],
        phases=arrays["phase"],
        midpoint_indices=arrays["midpoint_index"],
        combined_vs_zero_improvements=np.ascontiguousarray(zero_improvement),
        combined_vs_baseline_improvements=np.ascontiguousarray(baseline_improvement),
        expected_path_ids=expected_paths,
        selected_outer_steps=SELECTED_OUTER_STEPS,
    )
    high = torch.as_tensor(
        np.asarray(arrays["outer_step"]) < 128,
        dtype=torch.bool,
        device=device,
    )
    replay = {
        "validation_mse": float(torch.mean(squared).cpu()),
        "validation_high_reverse_time_mse": float(torch.mean(squared[high]).cpu()),
        "baseline_validation_mse": float(torch.mean(baseline_squared).cpu()),
        "baseline_high_reverse_time_mse": float(torch.mean(baseline_squared[high]).cpu()),
        "zero_validation_mse": float(torch.mean(zero_squared).cpu()),
        "zero_high_reverse_time_mse": float(torch.mean(zero_squared[high]).cpu()),
    }
    replay.update(
        {
            "combined_vs_baseline": replay["baseline_validation_mse"]
            - replay["validation_mse"],
            "combined_vs_baseline_high_reverse_time": replay[
                "baseline_high_reverse_time_mse"
            ]
            - replay["validation_high_reverse_time_mse"],
            "combined_vs_zero": replay["zero_validation_mse"]
            - replay["validation_mse"],
            "combined_vs_zero_high_reverse_time": replay[
                "zero_high_reverse_time_mse"
            ]
            - replay["validation_high_reverse_time_mse"],
        }
    )
    errors = {
        name: abs(float(replay[name]) - float(candidate[name]))
        for name in replay
    }
    maximum_error = max(errors.values(), default=0.0)
    if maximum_error > VALIDATION_REPLAY_TOLERANCE:
        raise FalseDiscoveryAdjudicationError(
            "stored validation candidate metrics do not replay",
            failure_code="candidate_metric_replay_invalid",
        )
    if int(candidate["update"]) == 0:
        exact_error = float(torch.max(torch.abs(prediction - baseline_prediction)).cpu())
        if exact_error != 0.0:
            raise FalseDiscoveryAdjudicationError(
                "update zero does not reproduce the frozen baseline",
                failure_code="update_zero_replay_invalid",
            )
    npz_path, json_path = _candidate_output_paths(
        run_dir, int(candidate["seed"]), int(candidate["update"])
    )
    artifact = _atomic_npz(
        npz_path,
        path_ids=table.path_ids,
        path_values=table.path_values,
    )
    record = _semantic_record(
        {
            "schema": RUN_SCHEMA + "-candidate-replay",
            "schema_version": 1,
            "seed": int(candidate["seed"]),
            "update": int(candidate["update"]),
            "checkpoint_path": str(candidate["checkpoint_path"]),
            "checkpoint_file_sha256": str(candidate["checkpoint_file_sha256"]),
            "checkpoint_state_sha256": str(candidate["state_sha256"]),
            "path_table_path": npz_path.relative_to(run_dir).as_posix(),
            "path_table_sha256": artifact["sha256"],
            "path_table_size": artifact["size"],
            "path_count": int(table.path_ids.size),
            "family_size": int(table.path_values.shape[1]),
            "sample_key_sha256": table.sample_key_sha256,
            "replayed_metrics": replay,
            "metric_errors": errors,
            "maximum_candidate_record_replay_error": maximum_error,
            "permitted_later_state_inputs_only": 1,
            "direct_float64_mse": 1,
            **_scope(),
        }
    )
    atomic_write_json(json_path, record)
    del model, prediction, squared
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return table.path_values, record


def _candidate_adjudication(
    run_dir: Path, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = args.parent_run_dir
    records, _ = _candidate_records(parent)
    selection = _parent._verify_training_selection(parent)
    path_plan = _load_json(parent / "path_id_plan.json")
    validation_paths = tuple(int(value) for value in path_plan["roles"]["validation"])
    confirmation_paths = tuple(int(value) for value in path_plan["roles"]["confirmation"])
    input_arrays, input_index = load_eager_role_inputs(parent, "validation")
    label_arrays, label_index = load_eager_role_labels(parent, "validation")
    identity_names = (
        "sample_key",
        "path_id",
        "outer_step",
        "phase",
        "midpoint_index",
        "midpoint_fraction",
    )
    if input_index.get("semantic_sha256") != label_index.get("semantic_sha256") or any(
        not np.array_equal(input_arrays[name], label_arrays[name])
        for name in identity_names
    ):
        raise ArtifactCompatibilityError("validation input/label join changed")
    if set(np.unique(input_arrays["path_id"]).tolist()) != set(validation_paths) or set(
        validation_paths
    ) & set(confirmation_paths):
        raise ArtifactCompatibilityError("validation/confirmation path firewall changed")
    device = torch.device(args.device)
    model_inputs = _trainer._model_inputs_from_arrays(input_arrays, device)
    target = torch.as_tensor(
        np.array(label_arrays["denoising_target"], copy=True, order="C"),
        dtype=torch.float64,
        device=device,
    )
    baseline = load_tangent_baseline(
        parent / "tangent_baseline.npz",
        expected_sha256=str(selection["baseline_file_sha256"]),
    )
    baseline_model = BoundaryTangentPredictor(
        baseline, zero_residual=True
    ).to(device)
    with torch.no_grad():
        baseline_prediction = baseline_model.baseline_prediction(model_inputs).detach()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    path_tables: list[np.ndarray] = []
    seeds: list[int] = []
    updates: list[int] = []
    replay_records: list[dict[str, Any]] = []
    maximum_error = 0.0
    ordered = sorted(records, key=lambda value: (int(value["seed"]), int(value["update"])))
    for position, candidate in enumerate(ordered, 1):
        values, replay = _evaluate_candidate(
            run_dir,
            parent,
            candidate,
            baseline=baseline,
            model_inputs=model_inputs,
            target=target,
            baseline_prediction=baseline_prediction,
            arrays=input_arrays,
            expected_paths=validation_paths,
            device=device,
        )
        replay_records.append(replay)
        maximum_error = max(
            maximum_error,
            float(replay["maximum_candidate_record_replay_error"]),
        )
        if int(candidate["update"]) > 0:
            seeds.append(int(candidate["seed"]))
            updates.append(int(candidate["update"]))
            path_tables.append(np.ascontiguousarray(values))
        elapsed = time.perf_counter() - start_time
        eta = elapsed / position * (len(ordered) - position) if position else 0.0
        print(
            f"candidate replay {position}/{len(ordered)} "
            f"seed={candidate['seed']} update={candidate['update']} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    historical = replay_historical_selection(records)
    if (
        int(historical["historical_selection_reproduced"]) != 1
        or int(historical["selected_seed"]) != int(selection["selected_seed"])
        or int(historical["selected_update"]) != int(selection["selected_update"])
        or historical["selected_state_sha256"]
        != selection["selected_state_sha256"]
    ):
        raise FalseDiscoveryAdjudicationError(
            "historical validation selection does not replay",
            failure_code="historical_selection_replay_invalid",
        )
    table = build_candidate_validation_table(
        seeds=np.asarray(seeds, dtype=np.int64),
        updates=np.asarray(updates, dtype=np.int64),
        path_ids=np.asarray(validation_paths, dtype=np.int64),
        path_values=np.stack(path_tables),
        forbidden_path_ids=np.asarray(confirmation_paths, dtype=np.int64),
    )
    reversed_table = build_candidate_validation_table(
        seeds=np.asarray(seeds[::-1], dtype=np.int64),
        updates=np.asarray(updates[::-1], dtype=np.int64),
        path_ids=np.asarray(validation_paths[::-1], dtype=np.int64),
        path_values=np.stack(path_tables[::-1])[:, ::-1, :],
        forbidden_path_ids=np.asarray(confirmation_paths, dtype=np.int64),
    )
    if not (
        np.array_equal(table.seeds, reversed_table.seeds)
        and np.array_equal(table.updates, reversed_table.updates)
        and np.array_equal(table.path_ids, reversed_table.path_ids)
        and np.array_equal(table.path_values, reversed_table.path_values)
    ):
        raise FalseDiscoveryAdjudicationError(
            "candidate/path ordering changes the replay table",
            failure_code="candidate_order_invariance_invalid",
        )
    search = search_aware_candidate_max_t(
        table,
        confidence=DEFAULT_BOOTSTRAP_CONFIDENCE,
        replicates=int(args.bootstrap_replicates),
        seed=BOOTSTRAP_SEED,
        namespace=CANDIDATE_BOOTSTRAP_NAMESPACE,
    )
    classification = classify_candidate_audit(search)
    screens = candidate_directional_screens(table)
    selected_row = next(
        row
        for row in search["candidate_rows"]
        if int(row["seed"]) == HISTORICAL_SELECTED_SEED
        and int(row["update"]) == HISTORICAL_SELECTED_UPDATE
    )
    combined_artifact = _atomic_npz(
        run_dir / "validation_candidate_path_tables.npz",
        seeds=table.seeds,
        updates=table.updates,
        path_ids=table.path_ids,
        path_values=table.path_values,
    )
    registry_rows: list[dict[str, Any]] = []
    for replay in replay_records:
        seed, update = int(replay["seed"]), int(replay["update"])
        npz_path, json_path = _candidate_output_paths(run_dir, seed, update)
        registry_rows.append(
            {
                "seed": seed,
                "update": update,
                "record_path": json_path.relative_to(run_dir).as_posix(),
                "record_sha256": file_fingerprint(json_path),
                "table_path": npz_path.relative_to(run_dir).as_posix(),
                "table_sha256": file_fingerprint(npz_path),
            }
        )
    replay_registry = _semantic_record(
        {
            "schema": RUN_SCHEMA + "-candidate-replay-registry",
            "schema_version": 1,
            "entry_count": len(registry_rows),
            "entries": registry_rows,
            "combined_table_artifact": combined_artifact,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "candidate_replay_registry.json", replay_registry)
    atomic_write_json(run_dir / "historical_selection_replay.json", historical)
    atomic_write_json(run_dir / "search_aware_candidate_max_t.json", search)
    atomic_write_csv(run_dir / "candidate_directional_screens.csv", screens)
    elapsed = time.perf_counter() - start_time
    peak = 0.0
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        total_bytes = torch.cuda.get_device_properties(device).total_memory
        peak = peak_bytes / max(total_bytes, 1)
    persisted = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    resource_pass = _resource_limits_passed(
        elapsed_seconds=elapsed,
        peak_memory_fraction=peak,
        persisted_bytes=persisted,
    )
    standard_errors = np.asarray(list(search["standard_errors"].values()))
    metrics = {
        "schema": RUN_SCHEMA + "-candidate-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        **{name: 1 for name in CANDIDATE_REPLAY_FLAGS},
        "candidate_count": table.candidate_count,
        "validation_path_count": table.path_count,
        "residual_search_family_size": 480,
        "candidate_direction_family_size": 228,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "simultaneous_confidence": DEFAULT_BOOTSTRAP_CONFIDENCE,
        "historical_selected_seed": int(historical["selected_seed"]),
        "historical_selected_update": int(historical["selected_update"]),
        "maximum_candidate_record_replay_error": maximum_error,
        "selected_update_all_four_lower_bounds_positive": int(
            selected_row["selection_resolved_residual_signal"]
        ),
        "residual_resolved_candidate_count": int(
            search["selection_resolved_candidate_count"]
        ),
        "direction_compatible_candidate_count": int(
            search["directionally_compatible_candidate_count"]
        ),
        "qualifying_candidate_count": int(search["fully_qualified_candidate_count"]),
        "minimum_standard_error": float(np.min(standard_errors)),
        "maximum_standard_error": float(np.max(standard_errors)),
        "candidate_classification": classification,
        "elapsed_seconds": elapsed,
        "peak_memory_fraction": peak,
        "persisted_bytes": persisted,
        "maximum_elapsed_seconds": MAXIMUM_ADJUDICATION_SECONDS,
        "maximum_peak_memory_fraction": MAXIMUM_PEAK_MEMORY_FRACTION,
        "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
        "resource_limits_passed": int(resource_pass),
        **_scope(),
    }
    if not resource_pass:
        metrics["evaluation_status"] = "execution_failed"
        metrics["failure_domain"] = "resource_gate"
        metrics["failure_code"] = "forensic_adjudication_resource_limit_exceeded"
    gate = evaluate_candidate_gate(metrics)
    atomic_write_json(run_dir / "candidate_replay_metrics.json", metrics)
    atomic_write_json(run_dir / "candidate_replay_gate.json", gate)
    del baseline_model, baseline_prediction, target, model_inputs
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return gate, metrics


def _adjudicate_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "adjudication_gate.json"
    if gate_path.is_file():
        gate = _load_json(gate_path)
        seal_path = run_dir / "adjudication_artifact_seal.json"
        if seal_path.is_file():
            _verify_stage_seal(run_dir, seal_path.name)
        else:
            replay = evaluate_adjudication_gate(
                baseline_gate=_load_json(run_dir / "sealed_baseline_gate.json"),
                candidate_gate=_load_json(run_dir / "candidate_replay_gate.json"),
            )
            if gate != replay:
                raise ArtifactCompatibilityError(
                    "orphan adjudication gate does not replay"
                )
            _seal_stage(
                run_dir,
                (
                    "confirmation_reaggregation_replay.json",
                    "sealed_baseline_path_table.json",
                    "sealed_baseline_max_abs_t.json",
                    "sealed_baseline_metrics.json",
                    "sealed_baseline_gate.json",
                    "candidate_replay_registry.json",
                    "historical_selection_replay.json",
                    "search_aware_candidate_max_t.json",
                    "candidate_directional_screens.csv",
                    "candidate_replay_metrics.json",
                    "candidate_replay_gate.json",
                    "parent_immutability.json",
                    "adjudication_gate.json",
                ),
                seal_path.name,
            )
        return gate
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("adjudication requires a passing preflight")
    try:
        baseline_gate, _ = _baseline_adjudication(run_dir, args)
    except Exception as exc:
        metrics = {
            "evaluation_status": "execution_failed",
            "failure_domain": "sealed_baseline_evidence",
            "failure_code": str(getattr(exc, "failure_code", "sealed_baseline_evidence_invalid")),
            "error": str(exc),
            **_scope(),
        }
        atomic_write_json(run_dir / "sealed_baseline_metrics.json", metrics)
        baseline_gate = evaluate_baseline_gate(metrics)
        atomic_write_json(run_dir / "sealed_baseline_gate.json", baseline_gate)
        atomic_write_json(run_dir / "sealed_baseline_path_table.json", metrics)
        atomic_write_json(run_dir / "sealed_baseline_max_abs_t.json", metrics)
        atomic_write_json(run_dir / "confirmation_reaggregation_replay.json", metrics)
    if _passed(baseline_gate):
        try:
            candidate_gate, _ = _candidate_adjudication(run_dir, args)
        except Exception as exc:
            metrics = {
                "evaluation_status": "execution_failed",
                "failure_domain": "implementation_or_replay",
                "failure_code": str(getattr(exc, "failure_code", "implementation_or_replay_defect")),
                "error": str(exc),
                **_scope(),
            }
            atomic_write_json(run_dir / "candidate_replay_metrics.json", metrics)
            candidate_gate = evaluate_candidate_gate(metrics)
            atomic_write_json(run_dir / "candidate_replay_gate.json", candidate_gate)
            atomic_write_json(run_dir / "candidate_replay_registry.json", metrics)
            atomic_write_json(run_dir / "historical_selection_replay.json", metrics)
            atomic_write_json(run_dir / "search_aware_candidate_max_t.json", metrics)
            (run_dir / "candidate_directional_screens.csv").write_text(
                "seed,update,evaluation_status\n", encoding="utf-8"
            )
    else:
        candidate_gate = not_evaluated_gate(
            "candidate", "sealed baseline evidence is invalid"
        )
        atomic_write_json(run_dir / "candidate_replay_gate.json", candidate_gate)
        atomic_write_json(run_dir / "candidate_replay_metrics.json", candidate_gate)
        atomic_write_json(run_dir / "candidate_replay_registry.json", candidate_gate)
        atomic_write_json(run_dir / "historical_selection_replay.json", candidate_gate)
        atomic_write_json(run_dir / "search_aware_candidate_max_t.json", candidate_gate)
        (run_dir / "candidate_directional_screens.csv").write_text(
            "seed,update,evaluation_status\n", encoding="utf-8"
        )
    adjudication = evaluate_adjudication_gate(
        baseline_gate=baseline_gate,
        candidate_gate=candidate_gate,
    )
    atomic_write_json(gate_path, adjudication)
    _parent._verify_existing_registry(args.parent_run_dir)
    immutability = _semantic_record(
        {
            "schema": RUN_SCHEMA + "-parent-immutability",
            "schema_version": 1,
            "parent_registry_file_sha256": file_fingerprint(
                args.parent_run_dir / "artifact_registry.json"
            ),
            "parent_registry_semantic_sha256": _load_json(
                args.parent_run_dir / "artifact_registry.json"
            )["semantic_sha256"],
            "parent_files_immutable": 1,
            "parent_mutations": 0,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "parent_immutability.json", immutability)
    _seal_stage(
        run_dir,
        (
            "confirmation_reaggregation_replay.json",
            "sealed_baseline_path_table.json",
            "sealed_baseline_max_abs_t.json",
            "sealed_baseline_metrics.json",
            "sealed_baseline_gate.json",
            "candidate_replay_registry.json",
            "historical_selection_replay.json",
            "search_aware_candidate_max_t.json",
            "candidate_directional_screens.csv",
            "candidate_replay_metrics.json",
            "candidate_replay_gate.json",
            "parent_immutability.json",
            "adjudication_gate.json",
        ),
        "adjudication_artifact_seal.json",
    )
    return adjudication


def _decision_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    preflight = _optional_json(run_dir, "preflight_gate.json") or not_evaluated_gate(
        "preflight", "not run"
    )
    if _passed(preflight) and not (run_dir / "adjudication_gate.json").is_file():
        raise ArtifactCompatibilityError(
            "decision requires committed adjudication after a passing preflight"
        )
    gate_path = run_dir / "decision_gate.json"
    if gate_path.is_file():
        gate = _load_json(gate_path)
        seal_path = run_dir / "decision_artifact_seal.json"
        if seal_path.is_file():
            _verify_stage_seal(run_dir, seal_path.name)
        else:
            decision = _load_json(run_dir / "false_discovery_decision.json")
            if gate != evaluate_decision_gate(decision):
                raise ArtifactCompatibilityError("orphan decision gate does not replay")
            _seal_stage(
                run_dir,
                ("false_discovery_decision.json", "decision_gate.json"),
                seal_path.name,
            )
        return gate
    baseline = _optional_json(run_dir, "sealed_baseline_gate.json") or not_evaluated_gate(
        "baseline", "not run"
    )
    candidate = _optional_json(run_dir, "candidate_replay_gate.json") or not_evaluated_gate(
        "candidate", "not run"
    )
    adjudication = _optional_json(run_dir, "adjudication_gate.json")
    decision = decide_workflow(
        preflight_gate=preflight,
        baseline_gate=baseline,
        candidate_gate=candidate,
        adjudication_gate=adjudication,
    )
    gate = evaluate_decision_gate(decision)
    atomic_write_json(run_dir / "false_discovery_decision.json", decision)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        ("false_discovery_decision.json", "decision_gate.json"),
        "decision_artifact_seal.json",
    )
    return gate


def _workflow_record(run_dir: Path, *, require_gate: str) -> dict[str, Any]:
    preflight = _optional_json(run_dir, "preflight_gate.json")
    baseline = _optional_json(run_dir, "sealed_baseline_gate.json")
    candidate = _optional_json(run_dir, "candidate_replay_gate.json")
    adjudication = _optional_json(run_dir, "adjudication_gate.json")
    decision_gate = _optional_json(run_dir, "decision_gate.json")
    workflow = evaluate_required_gate(
        preflight_gate=preflight,
        baseline_gate=baseline,
        candidate_gate=candidate,
        adjudication_gate=adjudication,
        decision_gate=decision_gate,
        require_gate=require_gate,
    )
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    if not (run_dir / "false_discovery_decision.json").is_file():
        atomic_write_json(run_dir / "false_discovery_decision.json", workflow["decision"])
    return workflow


def _commit_all_stage_failure_decision(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    active_stage: str,
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit a closed fail-closed decision after an unexpected all-stage error."""

    if not (run_dir / "preflight_gate.json").is_file():
        metrics = {
            "schema": RUN_SCHEMA + "-preflight-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            **dict(failure),
            **_scope(),
        }
        atomic_write_json(run_dir / "preflight_metrics.json", metrics)
        atomic_write_json(
            run_dir / "preflight_gate.json", evaluate_preflight_gate(metrics)
        )
    preflight = _load_json(run_dir / "preflight_gate.json")
    if _passed(preflight) and not (run_dir / "sealed_baseline_gate.json").is_file():
        metrics = {
            "schema": RUN_SCHEMA + "-baseline-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            **dict(failure),
            **_scope(),
        }
        atomic_write_json(run_dir / "sealed_baseline_metrics.json", metrics)
        atomic_write_json(
            run_dir / "sealed_baseline_gate.json", evaluate_baseline_gate(metrics)
        )
    if _passed(preflight) and not (run_dir / "candidate_replay_gate.json").is_file():
        metrics = {
            "schema": RUN_SCHEMA + "-candidate-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            **dict(failure),
            **_scope(),
        }
        atomic_write_json(run_dir / "candidate_replay_metrics.json", metrics)
        atomic_write_json(
            run_dir / "candidate_replay_gate.json", evaluate_candidate_gate(metrics)
        )
    if _passed(preflight) and not (run_dir / "adjudication_gate.json").is_file():
        atomic_write_json(
            run_dir / "adjudication_gate.json",
            evaluate_adjudication_gate(
                baseline_gate=_optional_json(run_dir, "sealed_baseline_gate.json"),
                candidate_gate=_optional_json(run_dir, "candidate_replay_gate.json"),
            ),
        )
    if not (run_dir / "decision_gate.json").is_file():
        _decision_stage(run_dir, args)
    workflow = _workflow_record(run_dir, require_gate=args.require_gate)
    atomic_write_json(
        run_dir / "all_stage_failure_decision_commit.json",
        {
            "schema": RUN_SCHEMA + "-all-stage-failure-decision-commit",
            "schema_version": 1,
            "active_stage": active_stage,
            "terminal_decision": workflow["decision"]["decision"],
            "decision_committed_after_failure": 1,
            **dict(failure),
            **_scope(),
        },
    )
    return workflow


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "adjudicate", "decision")
    if stage == "report":
        return ()
    if stage in {"preflight", "adjudicate", "decision"}:
        return (stage,)
    raise ValueError(f"unknown stage: {stage}")


def _verify_report(run_dir: Path) -> None:
    for gate, seal in (
        ("preflight_gate.json", "preflight_artifact_seal.json"),
        ("adjudication_gate.json", "adjudication_artifact_seal.json"),
        ("decision_gate.json", "decision_artifact_seal.json"),
    ):
        if (run_dir / gate).is_file():
            _verify_stage_seal(run_dir, seal)
    if (run_dir / "artifact_registry.json").is_file():
        _verify_existing_registry(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication"
        ),
    )
    parser.add_argument("--run-name", default="production-sealed-false-discovery-adjudication")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in ("parent_run_dir", "resume_run_dir", "runs_root"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.resume_run_dir is None and args.stage in {"adjudicate", "decision", "report"}:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    expected_gate = {
        "preflight": "preflight",
        "adjudicate": "adjudicate",
        "decision": "decision",
        "all": "decision",
        "report": "none",
    }[args.stage]
    if not args.test_only:
        if args.bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES:
            parser.error("production adjudication freezes 50000 bootstrap replicates")
        if args.stage in {"adjudicate", "all"} and args.device != "cuda":
            parser.error("production checkpoint replay requires --device cuda")
        if args.require_gate not in {"none", expected_gate}:
            parser.error(f"--stage {args.stage} cannot require only {args.require_gate}")
    else:
        if args.require_gate != "none":
            parser.error("test-only runs are nonauthorizing")
        if not 8 <= args.bootstrap_replicates <= DEFAULT_BOOTSTRAP_REPLICATES:
            parser.error("test bootstrap replicates must be in [8,50000]")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    resumed = False
    initialized = False
    active_stage = "initialize"
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"false-discovery adjudication run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        _status(run_dir, state="running", stage=args.stage)
        if args.stage == "report":
            _verify_report(run_dir)
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            _status(run_dir, state="running", stage=stage)
            if stage == "preflight":
                _preflight_stage(run_dir, args)
            elif stage == "adjudicate":
                if not _passed(_optional_json(run_dir, "preflight_gate.json")):
                    if args.stage == "all":
                        continue
                    raise ArtifactCompatibilityError(
                        "adjudication requires a passing preflight gate"
                    )
                configure_exact_torch_backend(args.device)
                _adjudicate_stage(run_dir, args)
            elif stage == "decision":
                _decision_stage(run_dir, args)
        workflow = _workflow_record(run_dir, require_gate=args.require_gate)
        decision = str(workflow["decision"]["decision"])
        required_pass = int(workflow.get("required_gate_pass", 0)) == 1
        state = (
            "test_only_complete"
            if args.test_only and required_pass
            else ("complete" if required_pass else "gate_failed")
        )
        _status(
            run_dir,
            state=state,
            stage=args.stage,
            decision=decision,
            failure_domain=None if required_pass else "scientific_gate",
            failure_code=None if required_pass else f"{args.require_gate}_gate_failed",
            scientific_evidence_complete=int(
                workflow["decision"].get("scientific_evidence_complete", 0)
            ),
        )
        _artifact_registry(run_dir)
        print(f"false-discovery adjudication decision: {decision}", flush=True)
        return 0 if required_pass else 2
    except KeyboardInterrupt:
        if run_dir is not None and initialized:
            _status(
                run_dir,
                state="interrupted",
                stage=active_stage,
                message="interrupted; resume from the same run directory",
                failure_domain="interruption",
                failure_code="keyboard_interrupt",
                scientific_evidence_complete=0,
            )
            _artifact_registry(run_dir)
        raise
    except Exception as exc:
        if run_dir is not None and initialized:
            failure = {
                "schema": RUN_SCHEMA + "-execution-failure",
                "schema_version": 1,
                "evaluation_status": "execution_failed",
                "stage": active_stage,
                "failure_domain": str(getattr(exc, "failure_domain", "workflow_execution")),
                "failure_code": str(getattr(exc, "failure_code", "adjudication_execution_failed")),
                "error_type": type(exc).__name__,
                "error": str(exc),
                **_scope(),
            }
            atomic_write_json(run_dir / f"{active_stage}_execution_failure.json", failure)
            failure_workflow: dict[str, Any] | None = None
            if args.stage == "all":
                try:
                    failure_workflow = _commit_all_stage_failure_decision(
                        run_dir,
                        args,
                        active_stage=active_stage,
                        failure=failure,
                    )
                except Exception as decision_exc:
                    atomic_write_json(
                        run_dir / "all_stage_failure_decision_commit_error.json",
                        {
                            "schema": RUN_SCHEMA
                            + "-all-stage-failure-decision-commit-error",
                            "schema_version": 1,
                            "evaluation_status": "execution_failed",
                            "error_type": type(decision_exc).__name__,
                            "error": str(decision_exc),
                            **_scope(),
                        },
                    )
            _status(
                run_dir,
                state="execution_failed",
                stage=active_stage,
                decision=(
                    str(failure_workflow["decision"]["decision"])
                    if failure_workflow is not None
                    else None
                ),
                message=str(exc),
                failure_domain=failure["failure_domain"],
                failure_code=failure["failure_code"],
                scientific_evidence_complete=(
                    int(
                        failure_workflow["decision"].get(
                            "scientific_evidence_complete", 0
                        )
                    )
                    if failure_workflow is not None
                    else 0
                ),
            )
            _artifact_registry(run_dir)
        print(f"false-discovery adjudication error: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
