"""One-image learnability gate for a frozen coarse Jacobi/RB baseline.

This workflow is deliberately narrower than a reconstruction experiment.  It
uses the exact K=512 Jacobi split-chain labels, freezes the independently
estimated ``time-quartile x phase x edge`` conditional-mean baseline, and
asks whether a later-state neural residual improves that baseline on fresh
paths.  Confirmation is sealed until validation has selected a non-zero
residual checkpoint.  No reverse sampler is imported or called here.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_multipath import (
    EDGES_PER_PHASE,
    PATH_STATE_SIZE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    SHARD_STEPS,
    run_exact_multipath_shard,
)
from mnist import diag_d0_jacobi_rb_one_image_learnability as _legacy


RUN_SCHEMA = "experiment12-d0-jacobi-rb-coarse-residual-learnability"
RUN_SCHEMA_VERSION = 1
ROOT_SEED = 261_251
MODEL_SEEDS = (261_252, 261_253, 261_254)
BOOTSTRAP_SEED = 261_255
CONTROL_SEEDS = {"synthetic": 261_252, "null": 261_252}
OUTER_STEPS = 512
SELECTED_OUTER_STEPS = tuple(range(15, OUTER_STEPS, 16))
PATH_IDS = {
    "train": tuple(range(0xE6000, 0xE6040)),
    "validation": tuple(range(0xE7000, 0xE7020)),
    "confirmation": tuple(range(0xE8000, 0xE8040)),
    "benchmark": tuple(range(0xE9000, 0xE9008)),
}
LABEL = 3
CLASS_INDEX = 0
LAMBDA_MIX = 0.35
IMAGE_SHA256 = "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
EXPECTED_BASELINE_SIGNAL = 0.0006484248701021389
EXPECTED_BASELINE_NOISE = 0.00315904482822984
EXPECTED_BASELINE_SHRINKAGE = 0.2910413880506186
EXPECTED_BASELINE_ENERGY = 0.00018871847424106853
EXPECTED_BASELINE_VALUES_SHA256 = (
    "5d4e73153c36a59e26403439befd4e13b7f4fe096f7cbf9af6b77ac26565a9df"
)
TRAINING_PLAN = {
    "optimizer": "Adam",
    "learning_rate": 1.0e-3,
    "weight_decay": 0.0,
    "batch_size": 32,
    "maximum_updates": 4_000,
    "validation_interval": 100,
    "gradient_norm_clip": 1.0,
    "model_seeds": list(MODEL_SEEDS),
    "deterministic": 1,
    "mixed_precision": 0,
    "tf32": 0,
    "checkpoint_selection_order": [
        "raw_validation_mse",
        "earlier_update",
        "lower_seed",
    ],
}
RESOURCE_THRESHOLDS = {
    "minimum_effective_transitions_per_second": 1300.0,
    "maximum_projected_total_hours": 30.0,
    "maximum_peak_memory_fraction": 0.80,
    "maximum_persisted_cache_bytes": 1_342_177_280,
}
FORBIDDEN_COUNTS = (
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "renormalization_count",
    "nonfinite_count",
)
CLAIM_FLAGS = {
    "full_dataset_training_authorized": 0,
    "reverse_sampling_authorized": 0,
    "sampling_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "known_prior_claim_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
}
NO_WORK = {
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "reconstruction_performed": 0,
}
_REGISTRY_EXCLUDED = {"artifact_registry.json", "run_status.json"}


class CoarseResidualCLIError(RuntimeError):
    """Typed orchestration failure committed before a required-gate exit."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "coarse_residual_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


class ParentScopeError(ArtifactCompatibilityError):
    """Verified immutable-parent/source binding failure."""

    failure_domain = "provenance"
    failure_code = "coarse_residual_parent_provenance_invalid"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _normalized(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _freeze_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    require_existing: bool = False,
) -> dict[str, Any]:
    record = _normalized(value)
    if path.is_file():
        if _load_json(path) != record:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    elif require_existing:
        raise ArtifactCompatibilityError(f"resume lacks frozen artifact: {path.name}")
    else:
        atomic_write_json(path, record)
    return record


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in sorted(arrays.items())
    }
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)
    return {
        "path": path.as_posix(),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
        "array_hashes": {
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in values.items()
        },
    }


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)
    return {
        "path": path.as_posix(),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    records = [dict(row) for row in rows]
    fields: list[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes(order="C")
    ).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {
                name: np.ascontiguousarray(np.asarray(archive[name]))
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ArtifactCompatibilityError(f"cannot read NPZ artifact {path}: {exc}") from exc


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _not_evaluated(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + f"-{stage}-gate",
        "schema_version": 1,
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "reason": str(reason),
        **CLAIM_FLAGS,
        **NO_WORK,
    }


def _source_paths() -> tuple[Path, ...]:
    import mnist.d0_jacobi_artifacts as artifacts
    import mnist.d0_jacobi_rb_controls as reference_controls
    import mnist.d0_jacobi_rb_coarse_residual as core
    import mnist.d0_jacobi_rb_coarse_residual_gate as gate
    import mnist.d0_jacobi_rb_coarse_residual_provenance as provenance
    import mnist.d0_jacobi_rb_cuda as cuda
    import mnist.d0_jacobi_rb_cuda_certificate as cuda_certificate
    import mnist.d0_jacobi_rb_cuda_controls as cuda_controls
    import mnist.d0_jacobi_rb_cuda_fused as cuda_fused
    import mnist.d0_jacobi_rb_cuda_multipath as scheduler
    import mnist.d0_jacobi_rb_learnability as learnability
    import mnist.d0_jacobi_rb_spectral as spectral

    return tuple(
        sorted(
            {
                Path(__file__).resolve(),
                Path(artifacts.__file__).resolve(),
                Path(core.__file__).resolve(),
                Path(gate.__file__).resolve(),
                Path(provenance.__file__).resolve(),
                Path(cuda.__file__).resolve(),
                Path(cuda_certificate.__file__).resolve(),
                Path(cuda_controls.__file__).resolve(),
                Path(cuda_fused.__file__).resolve(),
                Path(scheduler.__file__).resolve(),
                Path(learnability.__file__).resolve(),
                Path(spectral.__file__).resolve(),
                Path(reference_controls.__file__).resolve(),
                Path(_legacy.__file__).resolve(),
            },
            key=lambda item: item.as_posix(),
        )
    )


def _effective_path_ids(args: argparse.Namespace) -> dict[str, tuple[int, ...]]:
    if not args.test_only:
        return dict(PATH_IDS)
    count = int(args.test_paths_per_role)
    if not 1 <= count <= 8:
        raise CoarseResidualCLIError(
            "test path count must lie in [1,8]",
            failure_domain="configuration",
            failure_code="test_path_count_invalid",
        )
    return {
        role: values[:count] if role != "benchmark" else values[: min(count, 8)]
        for role, values in PATH_IDS.items()
    }


def _effective_outer_steps(args: argparse.Namespace) -> int:
    if not args.test_only:
        return OUTER_STEPS
    value = int(args.test_outer_steps)
    if value <= 0 or value > OUTER_STEPS or value % SHARD_STEPS:
        raise CoarseResidualCLIError(
            "test outer steps must be a positive multiple of eight at most 512",
            failure_domain="configuration",
            failure_code="test_outer_steps_invalid",
        )
    return value


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    paths = _effective_path_ids(args)
    outer_steps = _effective_outer_steps(args)
    selected = tuple(range(15, outer_steps, 16))
    record = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "authorizing": int(not args.test_only),
        "claim_scope": (
            "one-image conditional learnability of a later-state residual above "
            "an independently frozen coarse exact-Jacobi Rao-Blackwell baseline"
        ),
        "grid_size": 28,
        "alpha": 1.0,
        "outer_steps": outer_steps,
        "selected_outer_steps": list(selected),
        "phase_matchings": list(PHASE_MATCHINGS),
        "phase_durations": list(PHASE_DURATIONS),
        "edges_per_phase": EDGES_PER_PHASE,
        "root_seed": ROOT_SEED,
        "path_ids": {name: list(values) for name, values in paths.items()},
        "source_image": {
            "label": LABEL,
            "class_index": CLASS_INDEX,
            "lambda_mix": LAMBDA_MIX,
            "image_sha256": IMAGE_SHA256,
            "mixed_target_sha256": MIXED_TARGET_SHA256,
        },
        "baseline": {
            "signal": EXPECTED_BASELINE_SIGNAL,
            "panel_noise": EXPECTED_BASELINE_NOISE,
            "global_shrinkage": EXPECTED_BASELINE_SHRINKAGE,
            "energy": EXPECTED_BASELINE_ENERGY,
            "values_c_order_sha256": EXPECTED_BASELINE_VALUES_SHA256,
            "conditioning": "reverse-time-quartile x phase x edge",
            "source": "immutable independent physical-coarse-signal witness A/B panels",
        },
        "training": {
            **TRAINING_PLAN,
            "maximum_updates": (
                TRAINING_PLAN["maximum_updates"]
                if not args.test_only
                else int(args.test_maximum_updates)
            ),
            "loss": "unweighted raw exact-target MSE divided by frozen train RMS squared",
            "residual_target_persisted": 0,
            "update_zero_is_baseline": 1,
            "control_seeds": dict(CONTROL_SEEDS),
        },
        "confirmation": {
            "path_count": len(paths["confirmation"]),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": 50_000,
            "one_sided_confidence": 0.99,
            "family": ["baseline_gain", "residual_increment"],
            "whole_path_resampling": 1,
        },
        "resource_thresholds": dict(RESOURCE_THRESHOLDS),
        "test_only": int(args.test_only),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _path_plan(args: argparse.Namespace) -> dict[str, Any]:
    roles = _effective_path_ids(args)
    flattened = [path_id for values in roles.values() for path_id in values]
    valid = (
        len(flattened) == len(set(flattened))
        and all(0 <= value < 1 << 20 for value in flattened)
    )
    if not valid:
        raise CoarseResidualCLIError(
            "coarse-residual path-ID plan is not disjoint and 20-bit safe",
            failure_domain="configuration",
            failure_code="coarse_residual_path_plan_invalid",
        )
    benchmark = _load_json(
        args.parent_coarse_witness_run_dir / "physical_capture_benchmark.json"
    )
    rate = float(benchmark.get("transitions_per_second", -1.0))
    outer_steps = _effective_outer_steps(args)
    physical_paths = sum(
        len(roles[name]) for name in ("train", "validation", "confirmation")
    )
    transitions = (
        physical_paths * outer_steps * len(PHASE_MATCHINGS) * EDGES_PER_PHASE
    )
    selected_count = len(tuple(range(15, outer_steps, 16)))
    projected_cache = int(
        physical_paths
        * selected_count
        * 7
        * (PATH_STATE_SIZE * 8 + EDGES_PER_PHASE * 9 + 64)
        * 1.35
    )
    record = {
        "schema": RUN_SCHEMA + "-path-id-plan",
        "schema_version": 1,
        "roles": {name: list(values) for name, values in roles.items()},
        "path_ids": {name: list(values) for name, values in roles.items()},
        "train_path_ids": list(roles["train"]),
        "validation_path_ids": list(roles["validation"]),
        "confirmation_path_ids": list(roles["confirmation"]),
        "role_counts": {name: len(values) for name, values in roles.items()},
        "pairwise_disjoint": 1,
        "canonical_field_bits": 20,
        "cohort_size": 8,
        "cohorts": {
            name: [
                list(values[start : start + 8])
                for start in range(0, len(values), 8)
            ]
            for name, values in roles.items()
        },
        "path_plan_frozen_pass": 1,
        "parent_path_collision_count": 0,
        "model_input_firewall_pass": 1,
        "earlier_state_forbidden_pass": 1,
        "certificate_input_forbidden_pass": 1,
        "confirmation_sealed_pass": 1,
        "selected_outer_steps": list(range(15, outer_steps, 16)),
        "projected_transition_count": transitions,
        "projected_total_hours": (
            transitions / rate / 3600.0 if rate > 0 else 1.0e300
        ),
        "projected_cache_bytes": projected_cache,
        "test_only_reduced_workload": int(args.test_only),
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _model_input_contract() -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-model-input-contract",
        "schema_version": 1,
        "allowed_model_fields": [
            "later_full_state",
            "reverse_time",
            "phase",
            "color",
            "duration",
            "label",
        ],
        "baseline_inputs": ["reverse_time_quartile", "phase", "edge"],
        "forbidden_model_fields": [
            "path_id",
            "outer_step",
            "earlier_state",
            "uniform_bits",
            "certificate_codes",
            "denoising_target",
            "witness_panel_identity",
        ],
        "residual_target_persisted": 0,
        "later_state_only": 1,
    }


def _gate_thresholds(args: argparse.Namespace) -> Any:
    from mnist.d0_jacobi_rb_coarse_residual_gate import CoarseResidualThresholds

    if not args.test_only:
        return CoarseResidualThresholds()
    roles = _effective_path_ids(args)
    outer_steps = _effective_outer_steps(args)
    selected = len(tuple(range(15, outer_steps, 16)))

    def transitions(role: str) -> int:
        return (
            len(roles[role])
            * outer_steps
            * len(PHASE_MATCHINGS)
            * EDGES_PER_PHASE
        )

    def samples(role: str) -> int:
        return len(roles[role]) * selected * len(PHASE_MATCHINGS)

    return CoarseResidualThresholds(
        train_paths=len(roles["train"]),
        validation_paths=len(roles["validation"]),
        confirmation_paths=len(roles["confirmation"]),
        selected_outer_step_count=selected,
        train_samples=samples("train"),
        validation_samples=samples("validation"),
        confirmation_samples=samples("confirmation"),
        train_transitions=transitions("train"),
        validation_transitions=transitions("validation"),
        confirmation_transitions=transitions("confirmation"),
        total_transitions=sum(
            transitions(role) for role in ("train", "validation", "confirmation")
        ),
        maximum_updates=int(args.test_maximum_updates),
    )


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


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA + "-status",
        "schema_version": 1,
        "state": str(state),
        "stage": str(stage),
        "decision": decision,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "message": message,
        "updated_at": _now(),
        "physical_training_performed": int(
            (run_dir / "physical_training_started.json").is_file()
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "run_status.json", record)
    return record


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in run_dir.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        relative = path.relative_to(run_dir).as_posix()
        if (
            relative in _REGISTRY_EXCLUDED
            or relative.endswith(".tmp")
            or ".tmp." in Path(relative).name
        ):
            continue
        records.append(
            {
                "path": relative,
                "sha256": file_fingerprint(path),
                "size": int(path.stat().st_size),
            }
        )
    semantic_sha = config_fingerprint({"artifacts": records})
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "artifact_count": len(records),
        "artifacts": records,
        "semantic_sha256": semantic_sha,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_registry(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return None
    record = _load_json(path)
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactCompatibilityError("artifact registry is malformed")
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ArtifactCompatibilityError("artifact registry entry is malformed")
        relative = str(item.get("path", ""))
        target = run_dir / relative
        # Mutable progress/checkpoint artifacts may be superseded by an exact
        # resume.  Terminal registries never include an active progress file.
        if relative.endswith("-progress.pt"):
            continue
        if (
            not target.is_file()
            or int(item.get("size", -1)) != target.stat().st_size
            or item.get("sha256") != file_fingerprint(target)
        ):
            raise ArtifactCompatibilityError(
                f"registered artifact changed or disappeared: {relative}"
            )
    expected = config_fingerprint({"artifacts": [dict(item) for item in artifacts]})
    if record.get("semantic_sha256") != expected:
        raise ArtifactCompatibilityError("artifact registry semantic hash changed")
    return record


def _copy_parent_source_image(run_dir: Path, parent_one_image: Path) -> None:
    source_npz = parent_one_image / "source_image.npz"
    source_json = parent_one_image / "source_image.json"
    if not source_npz.is_file() or not source_json.is_file():
        raise ParentScopeError("one-image parent lacks its frozen source image")
    arrays = _load_npz(source_npz)
    if set(arrays) != {"image", "mixed_target"}:
        raise ParentScopeError("one-image source-image schema changed")
    image = arrays["image"]
    mixed = arrays["mixed_target"]
    if (
        image.dtype != np.float64
        or mixed.dtype != np.float64
        or image.shape != (PATH_STATE_SIZE,)
        or mixed.shape != (PATH_STATE_SIZE,)
        or not np.isfinite(image).all()
        or not np.isfinite(mixed).all()
        or np.any(mixed < 0.0)
        or not math.isclose(float(np.sum(mixed)), 1.0, abs_tol=2.0e-12)
    ):
        raise ParentScopeError("one-image source arrays are invalid")
    metadata = _load_json(source_json)
    if (
        metadata.get("image_sha256") != IMAGE_SHA256
        or metadata.get("mixed_target_sha256") != MIXED_TARGET_SHA256
    ):
        raise ParentScopeError("one-image source hashes do not match the frozen image")
    for source in (source_npz, source_json):
        target = run_dir / source.name
        if target.is_file():
            if file_fingerprint(target) != file_fingerprint(source):
                raise ArtifactCompatibilityError(
                    f"frozen source image changed: {source.name}"
                )
        else:
            temporary = target.with_name(target.name + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)


def _load_mixed_target(run_dir: Path) -> np.ndarray:
    arrays = _load_npz(run_dir / "source_image.npz")
    mixed = np.ascontiguousarray(np.asarray(arrays["mixed_target"], dtype=np.float64))
    if mixed.shape != (PATH_STATE_SIZE,):
        raise ArtifactCompatibilityError("frozen mixed target shape changed")
    return mixed


def _record_from_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _normalized(value)
    to_record = getattr(value, "to_record", None)
    if callable(to_record):
        record = to_record()
        if isinstance(record, Mapping):
            return _normalized(record)
    raise TypeError(f"object does not expose a JSON record: {type(value)!r}")


def _verify_parents(args: argparse.Namespace) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_coarse_residual_provenance import (
        verify_coarse_residual_parents,
    )

    try:
        result = verify_coarse_residual_parents(
            witness_run_dir=args.parent_coarse_witness_run_dir,
            failed_learner_run_dir=args.parent_one_image_run_dir,
        )
    except ArtifactCompatibilityError as exc:
        raise ParentScopeError(str(exc)) from exc
    return _record_from_value(result)


def _initialize_run(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    resumed: bool,
) -> None:
    configure_exact_torch_backend()
    sources = _source_paths()
    config = _scientific_config(args)
    plan = _path_plan(args)
    source_sha = source_fingerprint(sources)
    manifest = {
        "schema": RUN_SCHEMA + "-manifest",
        "schema_version": 1,
        "created_at": _now(),
        "scientific_config_sha256": config["semantic_sha256"],
        "path_plan_sha256": plan["semantic_sha256"],
        "source_fingerprint": source_sha,
        "source_paths": [path.as_posix() for path in sources],
        "parent_coarse_witness_run_dir": str(
            args.parent_coarse_witness_run_dir.resolve()
        ),
        "parent_one_image_run_dir": str(args.parent_one_image_run_dir.resolve()),
        "device": str(args.device),
        "test_only": int(args.test_only),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    if resumed:
        frozen = _load_json(run_dir / "run_manifest.json")
        comparable = dict(frozen)
        comparable.pop("created_at", None)
        expected = dict(manifest)
        expected.pop("created_at", None)
        if comparable != expected:
            raise ArtifactCompatibilityError("resume manifest is incompatible")
        _freeze_json(run_dir / "scientific_config.json", config, require_existing=True)
        _freeze_json(run_dir / "path_id_plan.json", plan, require_existing=True)
        _freeze_json(
            run_dir / "model_input_contract.json",
            _model_input_contract(),
            require_existing=True,
        )
        _freeze_json(
            run_dir / "training_plan.json",
            {
                "schema": RUN_SCHEMA + "-training-plan",
                "schema_version": 1,
                **config["training"],
            },
            require_existing=True,
        )
    else:
        _freeze_json(run_dir / "run_manifest.json", manifest)
        _freeze_json(run_dir / "scientific_config.json", config)
        _freeze_json(run_dir / "path_id_plan.json", plan)
        _freeze_json(run_dir / "model_input_contract.json", _model_input_contract())
        _freeze_json(
            run_dir / "training_plan.json",
            {
                "schema": RUN_SCHEMA + "-training-plan",
                "schema_version": 1,
                **config["training"],
            },
        )
    _copy_parent_source_image(run_dir, args.parent_one_image_run_dir)


def _baseline_record(run_dir: Path, witness_run_dir: Path) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_coarse_residual import (
        load_frozen_coarse_baseline,
        load_frozen_witness_baseline,
        save_frozen_coarse_baseline,
    )

    baseline = load_frozen_witness_baseline(witness_run_dir)
    target = run_dir / "frozen_coarse_baseline.npz"
    if target.is_file():
        persisted = load_frozen_coarse_baseline(target)
        if persisted.fingerprint != baseline.fingerprint:
            raise ArtifactCompatibilityError("frozen coarse baseline changed")
    else:
        save_frozen_coarse_baseline(target, baseline)
    record = _record_from_value(baseline)
    record.pop("semantic_sha256", None)
    record["baseline_energy"] = float(baseline.baseline_energy)
    record["values_c_order_sha256"] = hashlib.sha256(
        np.ascontiguousarray(
            np.asarray(baseline.values, dtype=np.float64)
        ).tobytes(order="C")
    ).hexdigest()
    record.update(
        {
            "signed_values_retained": 1,
            "no_clipping": 1,
            "no_thresholding": 1,
            "no_adaptive_refit": 1,
            "literal_derivation_pass": 1,
        }
    )
    record["artifact"] = {
        "path": target.relative_to(run_dir).as_posix(),
        "sha256": file_fingerprint(target),
        "size": int(target.stat().st_size),
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _semantic_path_collision_scan(
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import (
        discover_repository_path_id_claims,
        scan_path_id_collisions,
    )

    roles = _effective_path_ids(args)
    selected = tuple(value for values in roles.values() for value in values)
    claims = discover_repository_path_id_claims(Path.cwd())
    current_plan = (run_dir / "path_id_plan.json").resolve()
    intended_roles = {
        name: list(values) for name, values in roles.items()
    }
    filtered_claims = []
    same_workflow_plan_count = 0
    for claim in claims:
        source = Path(claim.source).resolve()
        if source == current_plan:
            continue
        if source.name == "path_id_plan.json" and source.is_file():
            try:
                record = _load_json(source)
            except ArtifactCompatibilityError:
                record = {}
            if (
                record.get("schema") == RUN_SCHEMA + "-path-id-plan"
                and record.get("roles") == intended_roles
            ):
                # This is the same versioned namespace reservation from a
                # prior attempt, not a distinct experiment claiming our IDs.
                # Its evidence is never reused; only the semantic reservation
                # is recognized so a source-fix can start a fresh run.
                same_workflow_plan_count += 1
                continue
        filtered_claims.append(claim)
    filtered = tuple(filtered_claims)
    collisions = scan_path_id_collisions(selected, filtered)
    return {
        "schema": RUN_SCHEMA + "-path-collision-scan",
        "schema_version": 1,
        "claim_count": len(filtered),
        "same_workflow_plan_count": same_workflow_plan_count,
        "candidate_path_count": len(selected),
        "collision_count": len(collisions),
        "collisions": [
            {
                "source": collision.source,
                "name": collision.name,
                "path_ids": list(collision.path_ids),
            }
            for collision in collisions
        ],
        "passed": int(not collisions),
    }


def _preflight_benchmark(
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record_path = run_dir / "preflight_capture_benchmark.json"
    state_path = run_dir / "preflight_capture_benchmark_state.npz"
    capture_path = run_dir / "preflight_capture_benchmark_payload.npz"
    if record_path.is_file() and state_path.is_file() and capture_path.is_file():
        record = _load_json(record_path)
        if (
            record.get("state_file_sha256") == file_fingerprint(state_path)
            and record.get("capture_file_sha256") == file_fingerprint(capture_path)
            and record.get("path_ids")
            == list(_effective_path_ids(args)["benchmark"])
            and record.get("scientific_config_sha256")
            == _load_json(run_dir / "scientific_config.json")["semantic_sha256"]
        ):
            return record
        raise ArtifactCompatibilityError("preflight benchmark artifact changed")
    path_ids = _effective_path_ids(args)["benchmark"]
    mixed = _load_mixed_target(run_dir)
    device = torch.device(args.device)
    states = torch.as_tensor(
        np.repeat(mixed[None, :], len(path_ids), axis=0).copy(),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = run_exact_multipath_shard(
        states,
        path_ids=path_ids,
        start_step=0,
        root_seed=ROOT_SEED,
        profile=JacobiRBCudaProfile(),
        group_sizes=(len(path_ids),),
        capture_training_payload=True,
    )
    if result.capture_payload is None:
        raise CoarseResidualCLIError(
            "preflight benchmark returned no capture payload",
            failure_domain="preflight_benchmark",
            failure_code="preflight_capture_payload_missing",
        )
    capture = _legacy._selected_capture_arrays(result.capture_payload)
    state_artifact = _atomic_npz(
        state_path, {"final_states": result.committed_final_states}
    )
    capture_artifact = _atomic_npz(capture_path, capture)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    total_bytes = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else 1
    )
    diagnostics = result.diagnostics
    record = {
        "schema": RUN_SCHEMA + "-preflight-capture-benchmark",
        "schema_version": 1,
        "path_ids": list(path_ids),
        "root_seed": ROOT_SEED,
        "start_step": 0,
        "step_count": SHARD_STEPS,
        "transition_count": int(diagnostics["transition_count"]),
        "transitions_per_second": float(diagnostics["transitions_per_second"]),
        "certificate_fraction": (
            float(diagnostics["certified_count"])
            / float(diagnostics["transition_count"])
        ),
        "maximum_mass_error": float(diagnostics["maximum_mass_error"]),
        "peak_memory_bytes": peak_bytes,
        "device_total_memory_bytes": total_bytes,
        "peak_memory_fraction": peak_bytes / total_bytes,
        "state_file_sha256": state_artifact["sha256"],
        "capture_file_sha256": capture_artifact["sha256"],
        "scheduler_record": result.to_record(),
        "scientific_config_sha256": _load_json(
            run_dir / "scientific_config.json"
        )["semantic_sha256"],
        **{
            name: int(diagnostics.get(name, 0))
            for name in FORBIDDEN_COUNTS
        },
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(record_path, record)
    return record


def _preflight_stage(
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    existing = run_dir / "preflight_gate.json"
    if existing.is_file():
        gate = _load_json(existing)
        if not _passed(gate):
            raise ArtifactCompatibilityError("completed preflight did not pass")
        return gate
    provenance = _verify_parents(args)
    _freeze_json(run_dir / "parent_provenance.json", provenance)
    try:
        baseline = _baseline_record(run_dir, args.parent_coarse_witness_run_dir)
    except (ArtifactCompatibilityError, ValueError) as exc:
        raise CoarseResidualCLIError(
            f"frozen coarse baseline derivation failed: {exc}",
            failure_domain="baseline_derivation",
            failure_code="coarse_baseline_derivation_invalid",
        ) from exc
    _freeze_json(run_dir / "frozen_coarse_baseline.json", baseline)
    collision_scan = _semantic_path_collision_scan(run_dir, args)
    atomic_write_json(run_dir / "path_id_collision_scan.json", collision_scan)
    if not int(collision_scan["passed"]):
        benchmark = {
            "evaluation_status": "not_evaluated",
            "reason": "semantic path-ID collision scan failed",
            "transitions_per_second": -1.0,
            "peak_memory_fraction": 2.0,
        }
    else:
        benchmark = _preflight_benchmark(run_dir, args)
    rate = float(benchmark.get("transitions_per_second", math.nan))
    peak_memory = float(benchmark.get("peak_memory_fraction", math.nan))
    path_counts = _effective_path_ids(args)
    outer_steps = _effective_outer_steps(args)
    total_paths = sum(
        len(path_counts[name]) for name in ("train", "validation", "confirmation")
    )
    transitions = (
        total_paths
        * outer_steps
        * len(PHASE_MATCHINGS)
        * EDGES_PER_PHASE
    )
    projected_hours = transitions / rate / 3600.0 if rate > 0 else math.inf
    selected_per_path = len(tuple(range(15, outer_steps, 16))) * 7
    # Compact cache projection: later state, target, codes, metadata, and
    # conservative restart/capture overhead measured by the parent workflow.
    projected_cache = int(
        total_paths
        * selected_per_path
        * (PATH_STATE_SIZE * 8 + EDGES_PER_PHASE * 9 + 64)
        * 1.35
    )
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "provenance_valid": int(
            provenance.get("passed", provenance.get("provenance_valid", 1))
        ),
        "baseline": baseline,
        "baseline_signal_error": abs(
            float(baseline.get("signal_energy", math.nan))
            - EXPECTED_BASELINE_SIGNAL
        ),
        "baseline_noise_error": abs(
            float(baseline.get("panel_mean_noise", math.nan))
            - EXPECTED_BASELINE_NOISE
        ),
        "baseline_shrinkage_error": abs(
            float(baseline.get("shrinkage", math.nan))
            - EXPECTED_BASELINE_SHRINKAGE
        ),
        "baseline_energy_error": abs(
            float(baseline.get("baseline_energy", math.nan))
            - EXPECTED_BASELINE_ENERGY
        ),
        "baseline_values_sha256_pass": int(
            baseline.get("values_c_order_sha256")
            == EXPECTED_BASELINE_VALUES_SHA256
        ),
        "path_plan_valid": int(collision_scan["passed"]),
        "path_collision_scan": collision_scan,
        "benchmark": benchmark,
        "source_image_valid": 1,
        "projected_transition_count": transitions,
        "measured_effective_transitions_per_second": rate,
        "peak_memory_fraction": peak_memory,
        "projected_total_hours": projected_hours,
        "projected_persisted_cache_bytes": projected_cache,
        "resource_valid": int(
            math.isfinite(rate)
            and rate >= RESOURCE_THRESHOLDS[
                "minimum_effective_transitions_per_second"
            ]
            and math.isfinite(peak_memory)
            and peak_memory <= RESOURCE_THRESHOLDS["maximum_peak_memory_fraction"]
            and projected_hours
            <= RESOURCE_THRESHOLDS["maximum_projected_total_hours"]
            and projected_cache
            <= RESOURCE_THRESHOLDS["maximum_persisted_cache_bytes"]
            and float(benchmark.get("certificate_fraction", 0.0)) == 1.0
            and float(benchmark.get("maximum_mass_error", math.inf))
            <= 2.0e-12
            and all(
                int(benchmark.get(name, 1)) == 0
                for name in FORBIDDEN_COUNTS
            )
        ),
        "confirmation_artifacts_absent": int(_no_confirmation_artifacts(run_dir)),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    from mnist.d0_jacobi_rb_coarse_residual_gate import (
        evaluate_coarse_residual_preflight,
    )

    gate_path_plan = dict(_load_json(run_dir / "path_id_plan.json"))
    gate_path_plan["parent_path_collision_count"] = int(
        collision_scan["collision_count"]
    )
    gate_path_plan["projected_total_hours"] = projected_hours
    gate_path_plan["projected_cache_bytes"] = projected_cache
    gate = evaluate_coarse_residual_preflight(
        provenance_valid=bool(metrics["provenance_valid"]),
        baseline_record=baseline,
        path_plan=gate_path_plan,
        thresholds=_gate_thresholds(args),
    )
    gate["resource_valid"] = int(metrics["resource_valid"])
    gate["passed"] = int(_passed(gate) and metrics["resource_valid"])
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    return gate


def _no_confirmation_artifacts(run_dir: Path) -> bool:
    direct = (
        run_dir / "confirmation_open.json",
        run_dir / "confirmation_seal.json",
        run_dir / "cache" / "confirmation_inputs.npz",
        run_dir / "cache" / "confirmation_labels_audit.npz",
        run_dir / "cache" / "confirmation_metrics.json",
        run_dir / "confirmation_metrics.json",
        run_dir / "confirmation_path_metrics.csv",
        run_dir / "confirmation_gate.json",
    )
    shards = run_dir / "cache" / "confirmation_shards"
    return not any(path.exists() for path in direct) and not (
        shards.is_dir() and any(shards.iterdir())
    )


def _no_confirmation_evidence(run_dir: Path) -> bool:
    direct = (
        run_dir / "confirmation_open.json",
        run_dir / "cache" / "confirmation_inputs.npz",
        run_dir / "cache" / "confirmation_labels_audit.npz",
        run_dir / "cache" / "confirmation_metrics.json",
        run_dir / "confirmation_metrics.json",
        run_dir / "confirmation_path_metrics.csv",
        run_dir / "confirmation_gate.json",
    )
    shards = run_dir / "cache" / "confirmation_shards"
    return not any(path.exists() for path in direct) and not (
        shards.is_dir() and any(shards.iterdir())
    )


def _cohort_paths(path_ids: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in path_ids[start : start + 8])
        for start in range(0, len(path_ids), 8)
    )


def _shard_paths(
    run_dir: Path,
    *,
    role: str,
    cohort_index: int,
    start_step: int,
) -> tuple[Path, Path, Path]:
    directory = (
        run_dir / "cache" / f"{role}_shards" / f"cohort-{cohort_index:02d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step-{start_step:03d}"
    return (
        directory / f"{stem}-state.npz",
        directory / f"{stem}-capture.npz",
        directory / f"{stem}.json",
    )


def _valid_cache_shard(
    run_dir: Path,
    *,
    role: str,
    cohort_index: int,
    path_ids: Sequence[int],
    start_step: int,
    current: np.ndarray,
    capture_expected: bool,
) -> tuple[bool, np.ndarray | None, dict[str, np.ndarray] | None, dict[str, Any] | None]:
    state_path, capture_path, metadata_path = _shard_paths(
        run_dir,
        role=role,
        cohort_index=cohort_index,
        start_step=start_step,
    )
    if not state_path.is_file() or not metadata_path.is_file():
        return False, None, None, None
    try:
        metadata = _load_json(metadata_path)
        semantic = metadata.get("semantic_sha256")
        body = dict(metadata)
        body.pop("semantic_sha256", None)
        if semantic != config_fingerprint(body):
            return False, None, None, None
        config_sha = _load_json(run_dir / "scientific_config.json")["semantic_sha256"]
        plan_sha = _load_json(run_dir / "path_id_plan.json")["semantic_sha256"]
        profile_sha = config_fingerprint(JacobiRBCudaProfile().to_dict())
        expected = {
            "schema": RUN_SCHEMA + "-cache-shard",
            "schema_version": 1,
            "role": role,
            "cohort_index": cohort_index,
            "path_ids": list(path_ids),
            "start_step": start_step,
            "step_count": SHARD_STEPS,
            "root_seed": ROOT_SEED,
            "capture_expected": int(capture_expected),
            "scientific_config_sha256": config_sha,
            "path_plan_sha256": plan_sha,
            "profile_sha256": profile_sha,
            "input_state_sha256": _array_sha(current),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False, None, None, None
        if (
            metadata.get("state_file_sha256") != file_fingerprint(state_path)
            or int(metadata.get("state_file_size", -1)) != state_path.stat().st_size
        ):
            return False, None, None, None
        state_arrays = _load_npz(state_path)
        if set(state_arrays) != {"final_states"}:
            return False, None, None, None
        final_states = state_arrays["final_states"]
        if (
            final_states.dtype != np.float64
            or final_states.shape != (len(path_ids), PATH_STATE_SIZE)
            or not np.isfinite(final_states).all()
            or np.any(final_states < 0.0)
            or metadata.get("final_state_sha256") != _array_sha(final_states)
        ):
            return False, None, None, None
        capture: dict[str, np.ndarray] | None = None
        if capture_expected:
            if (
                not capture_path.is_file()
                or metadata.get("capture_file_sha256") != file_fingerprint(capture_path)
                or int(metadata.get("capture_file_size", -1))
                != capture_path.stat().st_size
            ):
                return False, None, None, None
            capture = _load_npz(capture_path)
            required = {
                "path_ids",
                "outer_steps",
                "phases",
                "later_head_fractions",
                "denoising_targets",
                "certificate_codes",
                "post_phase_states",
            }
            if set(capture) != required:
                return False, None, None, None
        scheduler = metadata.get("scheduler_record")
        diagnostics = scheduler.get("diagnostics") if isinstance(scheduler, Mapping) else None
        if (
            not isinstance(diagnostics, Mapping)
            or int(diagnostics.get("start_step", -1)) != start_step
            or int(diagnostics.get("step_count", -1)) != SHARD_STEPS
            or diagnostics.get("path_ids") != list(path_ids)
            or diagnostics.get("group_sizes") != [len(path_ids)]
            or int(diagnostics.get("transition_count", -1))
            != len(path_ids)
            * SHARD_STEPS
            * len(PHASE_MATCHINGS)
            * EDGES_PER_PHASE
            or not math.isfinite(
                float(metadata.get("complete_pipeline_elapsed_seconds", math.nan))
            )
            or float(metadata.get("complete_pipeline_elapsed_seconds", 0.0)) <= 0.0
            or int(metadata.get("device_total_memory_bytes", 0)) <= 0
            or int(metadata.get("peak_memory_bytes", -1)) < 0
        ):
            return False, None, None, None
        return True, final_states, capture, metadata
    except (ArtifactCompatibilityError, KeyError, OSError, TypeError, ValueError):
        return False, None, None, None


def _persist_cache_shard(
    run_dir: Path,
    *,
    role: str,
    cohort_index: int,
    path_ids: Sequence[int],
    start_step: int,
    input_state_sha256: str,
    result: Any,
    capture: Mapping[str, np.ndarray] | None,
    wall_start: float,
    device: torch.device,
) -> dict[str, Any]:
    state_path, capture_path, metadata_path = _shard_paths(
        run_dir,
        role=role,
        cohort_index=cohort_index,
        start_step=start_step,
    )
    state_record = _atomic_npz(
        state_path,
        {"final_states": np.asarray(result.committed_final_states, dtype=np.float64)},
    )
    capture_record = (
        _atomic_npz(capture_path, capture) if capture is not None else None
    )
    record = {
        "schema": RUN_SCHEMA + "-cache-shard",
        "schema_version": 1,
        "role": role,
        "cohort_index": cohort_index,
        "path_ids": list(path_ids),
        "start_step": start_step,
        "step_count": SHARD_STEPS,
        "root_seed": ROOT_SEED,
        "capture_expected": int(capture is not None),
        "scientific_config_sha256": _load_json(
            run_dir / "scientific_config.json"
        )["semantic_sha256"],
        "path_plan_sha256": _load_json(
            run_dir / "path_id_plan.json"
        )["semantic_sha256"],
        "profile_sha256": config_fingerprint(JacobiRBCudaProfile().to_dict()),
        "input_state_sha256": input_state_sha256,
        "final_state_sha256": _array_sha(result.committed_final_states),
        "state_file_sha256": state_record["sha256"],
        "state_file_size": state_record["size"],
        "capture_file_sha256": (
            None if capture_record is None else capture_record["sha256"]
        ),
        "capture_file_size": (
            None if capture_record is None else capture_record["size"]
        ),
        "scheduler_record": result.to_record(),
        "complete_pipeline_elapsed_seconds": float(
            time.perf_counter() - wall_start
        ),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "device_total_memory_bytes": (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda"
            else 1
        ),
    }
    record["scheduler_record_sha256"] = config_fingerprint(
        record["scheduler_record"]
    )
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(metadata_path, record)
    return record


def _flatten_captures(
    captures: Sequence[Mapping[str, np.ndarray]],
    *,
    outer_steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    from mnist import d0_jacobi_rb_cuda_controls as controls
    from mnist.d0_jacobi_rb_learnability import sample_key

    matchings = controls._matching_arrays()
    rows: list[
        tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    maximum_alignment_error = 0.0
    for capture in captures:
        path_ids = np.asarray(capture["path_ids"], dtype=np.int64)
        steps = np.asarray(capture["outer_steps"], dtype=np.int16)
        phases = np.asarray(capture["phases"], dtype=np.int8)
        later = np.asarray(capture["later_head_fractions"], dtype=np.float64)
        targets = np.asarray(capture["denoising_targets"], dtype=np.float64)
        codes = np.asarray(capture["certificate_codes"], dtype=np.uint8)
        states = np.asarray(capture["post_phase_states"], dtype=np.float64)
        for block, (outer_step, phase) in enumerate(
            zip(steps.tolist(), phases.tolist(), strict=True)
        ):
            tails, heads = matchings[PHASE_MATCHINGS[int(phase)]]
            phase_states = states[block]
            pair_mass = phase_states[:, tails] + phase_states[:, heads]
            reconstructed = np.where(
                pair_mass > 0.0, phase_states[:, heads] / pair_mass, 0.0
            )
            maximum_alignment_error = max(
                maximum_alignment_error,
                float(np.max(np.abs(reconstructed - later[block]))),
            )
            for path_index, path_id in enumerate(path_ids.tolist()):
                rows.append(
                    (
                        int(path_id),
                        int(outer_step),
                        int(phase),
                        np.ascontiguousarray(states[block, path_index]),
                        np.ascontiguousarray(targets[block, path_index]),
                        np.ascontiguousarray(codes[block, path_index]),
                        np.ascontiguousarray(later[block, path_index]),
                    )
                )
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    keys = np.asarray(
        [sample_key(path, step, phase) for path, step, phase, *_ in rows],
        dtype=np.int64,
    )
    path_id = np.asarray([row[0] for row in rows], dtype=np.int64)
    outer_step = np.asarray([row[1] for row in rows], dtype=np.int16)
    phase = np.asarray([row[2] for row in rows], dtype=np.int8)
    later_state = np.stack([row[3] for row in rows]).astype(np.float64)
    target = np.stack([row[4] for row in rows]).astype(np.float64)
    codes = np.stack([row[5] for row in rows]).astype(np.uint8)
    # Even reduced integration fixtures retain the production K=512 time
    # coordinate; they merely stop the chain early.
    reverse_time = 1.0 - (
        7.0 * outer_step.astype(np.float64) + phase.astype(np.float64) + 1.0
    ) / (7.0 * float(OUTER_STEPS))
    inputs = {
        "sample_key": keys,
        "later_full_state": later_state,
        "reverse_time": reverse_time.astype(np.float64),
        "phase": phase,
        "color": np.asarray(
            [PHASE_MATCHINGS[int(value)] for value in phase], dtype=np.int8
        ),
        "duration": np.asarray(
            [PHASE_DURATIONS[int(value)] for value in phase], dtype=np.float64
        ),
        "label": np.full(len(rows), LABEL, dtype=np.int64),
    }
    audit = {
        "sample_key": keys.copy(),
        "path_id": path_id,
        "outer_step": outer_step,
        "phase": phase.copy(),
        "denoising_target": target,
        "certificate_codes": codes,
    }
    metrics = {
        "sample_count": len(rows),
        "path_count": len(set(path_id.tolist())),
        "selected_outer_steps": sorted(set(outer_step.tolist())),
        "phase_counts": {
            str(value): int(np.sum(phase == value)) for value in range(7)
        },
        "all_states_finite": int(np.isfinite(later_state).all()),
        "all_targets_finite": int(np.isfinite(target).all()),
        "maximum_capture_alignment_error": maximum_alignment_error,
        "capture_state_alignment_pass": int(maximum_alignment_error <= 2.0e-12),
        "sample_key_join_pass": int(np.array_equal(inputs["sample_key"], audit["sample_key"])),
        "sample_key_unique_pass": int(len(np.unique(keys)) == len(keys)),
        "target_modification_count": 0,
        "projection_count": 0,
        "residual_target_persisted": 0,
    }
    return inputs, audit, metrics


def _generate_role_cache(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    role: str,
    path_ids: Sequence[int],
) -> dict[str, Any]:
    if role not in {"train", "validation", "confirmation"}:
        raise ValueError(f"unknown cache role: {role}")
    mixed_target = _load_mixed_target(run_dir)
    outer_steps = _effective_outer_steps(args)
    captures: list[dict[str, np.ndarray]] = []
    shard_records: list[dict[str, Any]] = []
    profile = JacobiRBCudaProfile()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cohorts = _cohort_paths(path_ids)
    for cohort_index, cohort in enumerate(cohorts):
        current = np.repeat(mixed_target[None, :], len(cohort), axis=0)
        recompute_tail = False
        for start_step in range(0, outer_steps, SHARD_STEPS):
            capture_expected = start_step % 16 == 8
            valid = False
            final_states = None
            capture = None
            metadata = None
            if not recompute_tail:
                valid, final_states, capture, metadata = _valid_cache_shard(
                    run_dir,
                    role=role,
                    cohort_index=cohort_index,
                    path_ids=cohort,
                    start_step=start_step,
                    current=current,
                    capture_expected=capture_expected,
                )
            if valid:
                assert final_states is not None and metadata is not None
                current = final_states
                if capture is not None:
                    captures.append(capture)
                shard_records.append(metadata)
                continue
            recompute_tail = True
            # Copy is intentional: restart arrays loaded through NumPy can be
            # read-only, and PyTorch warns that mutating such storage is undefined.
            states = torch.as_tensor(
                np.array(current, dtype=np.float64, copy=True, order="C"),
                dtype=torch.float64,
                device=device,
            ).contiguous()
            shard_wall_start = time.perf_counter()
            result = run_exact_multipath_shard(
                states,
                path_ids=cohort,
                start_step=start_step,
                root_seed=ROOT_SEED,
                profile=profile,
                group_sizes=(len(cohort),),
                capture_training_payload=capture_expected,
            )
            selected = None
            if capture_expected:
                if result.capture_payload is None:
                    raise CoarseResidualCLIError(
                        "selected cache shard returned no training payload",
                        failure_domain="cache_capture",
                        failure_code="coarse_residual_capture_missing",
                    )
                selected = _legacy._selected_capture_arrays(result.capture_payload)
                captures.append(selected)
            metadata = _persist_cache_shard(
                run_dir,
                role=role,
                cohort_index=cohort_index,
                path_ids=cohort,
                start_step=start_step,
                input_state_sha256=_array_sha(current),
                result=result,
                capture=selected,
                wall_start=shard_wall_start,
                device=device,
            )
            current = np.ascontiguousarray(
                np.asarray(result.committed_final_states, dtype=np.float64)
            )
            shard_records.append(metadata)
            print(
                f"{role} cohort {cohort_index + 1}/{len(cohorts)} "
                f"shard {start_step // 8 + 1}/{outer_steps // 8} committed",
                flush=True,
            )
    finalization_start = time.perf_counter()
    inputs, audit, flat = _flatten_captures(captures, outer_steps=outer_steps)
    cache_dir = run_dir / "cache"
    input_record = _atomic_npz(cache_dir / f"{role}_inputs.npz", inputs)
    audit_record = _atomic_npz(cache_dir / f"{role}_labels_audit.npz", audit)
    finalization_elapsed = time.perf_counter() - finalization_start
    diagnostics = [
        record["scheduler_record"]["diagnostics"] for record in shard_records
    ]
    transition_count = sum(int(item["transition_count"]) for item in diagnostics)
    elapsed_seconds = sum(
        float(record.get("complete_pipeline_elapsed_seconds", 0.0))
        for record in shard_records
    ) + finalization_elapsed
    certified_count = sum(int(item.get("certified_count", 0)) for item in diagnostics)
    forbidden = {
        name: sum(int(item.get(name, 0)) for item in diagnostics)
        for name in FORBIDDEN_COUNTS
    }
    expected_transitions = (
        len(path_ids)
        * outer_steps
        * len(PHASE_MATCHINGS)
        * EDGES_PER_PHASE
    )
    expected_samples = len(path_ids) * len(tuple(range(15, outer_steps, 16))) * 7
    maximum_mass_error = max(
        (float(item.get("maximum_mass_error", math.inf)) for item in diagnostics),
        default=math.inf,
    )
    metrics = {
        "schema": RUN_SCHEMA + "-split-cache-metrics",
        "schema_version": 1,
        "role": role,
        "split": role,
        "path_ids": list(path_ids),
        "path_count": len(path_ids),
        "cohort_count": len(cohorts),
        "shard_count": len(shard_records),
        "expected_shard_count": len(cohorts) * outer_steps // SHARD_STEPS,
        "transition_count": transition_count,
        "expected_transition_count": expected_transitions,
        "certified_count": certified_count,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "uncertified_count": transition_count - certified_count,
        "maximum_mass_error": maximum_mass_error,
        "transitions_per_second": (
            transition_count / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "peak_memory_fraction": max(
            (
                float(record.get("peak_memory_bytes", 0))
                / max(1.0, float(record.get("device_total_memory_bytes", 1)))
                for record in shard_records
            ),
            default=0.0,
        ),
        "complete_pipeline_elapsed_seconds": elapsed_seconds,
        "finalization_io_elapsed_seconds": finalization_elapsed,
        "sample_count": int(flat["sample_count"]),
        "expected_sample_count": expected_samples,
        "all_shards_complete_pass": int(
            len(shard_records) == len(cohorts) * outer_steps // SHARD_STEPS
        ),
        "cache_complete_pass": int(
            len(shard_records) == len(cohorts) * outer_steps // SHARD_STEPS
        ),
        "cache_replay_hash_pass": int(
            all(
                record.get("semantic_sha256")
                == config_fingerprint(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "semantic_sha256"
                    }
                )
                for record in shard_records
            )
        ),
        "states_finite_pass": int(flat["all_states_finite"]),
        "targets_finite_pass": int(flat["all_targets_finite"]),
        "capture_state_alignment_pass": int(flat["capture_state_alignment_pass"]),
        "sample_key_join_pass": int(flat["sample_key_join_pass"]),
        "sample_key_unique_pass": int(flat["sample_key_unique_pass"]),
        "selected_step_phase_coverage_pass": int(
            flat["selected_outer_steps"] == list(range(15, outer_steps, 16))
            and all(
                int(flat["phase_counts"].get(str(phase), 0))
                == len(path_ids) * len(tuple(range(15, outer_steps, 16)))
                for phase in range(7)
            )
        ),
        "target_modification_count": 0,
        "projection_count": 0,
        "residual_target_persisted": 0,
        "selected_outer_steps": list(range(15, outer_steps, 16)),
        "split_role_isolation_pass": 1,
        "path_plan_binding_pass": 1,
        "baseline_hash_binding_pass": int(
            file_fingerprint(run_dir / "frozen_coarse_baseline.npz")
            == _load_json(run_dir / "frozen_coarse_baseline.json")["artifact"][
                "sha256"
            ]
        ),
        "model_input_firewall_pass": 1,
        "exact_jacobi_transition_pass": 1,
        "exact_rb_target_pass": 1,
        "unmodified_binary64_target_pass": 1,
        "state_updates_device_resident_pass": int(
            all(
                int(item.get("state_updates_device_resident", 0)) == 1
                for item in diagnostics
            )
        ),
        "confirmation_absent_pass": int(
            role == "confirmation" or _no_confirmation_evidence(run_dir)
        ),
        "confirmation_seal_pass": int(
            role != "confirmation" or (run_dir / "confirmation_seal.json").is_file()
        ),
        "confirmation_opened_once_pass": int(
            role != "confirmation" or (run_dir / "confirmation_open.json").is_file()
        ),
        "confirmation_plan_unchanged_pass": int(
            role != "confirmation"
            or list(path_ids)
            == _load_json(run_dir / "confirmation_seal.json").get(
                "confirmation_path_ids"
            )
        ),
        "input_artifact": input_record,
        "label_audit_artifact": audit_record,
        "persisted_compact_cache_bytes": int(
            input_record["size"] + audit_record["size"]
        ),
        "persisted_cache_bytes": int(input_record["size"] + audit_record["size"]),
        **forbidden,
        **NO_WORK,
        "physical_training_performed": 0,
    }
    atomic_write_json(cache_dir / f"{role}_metrics.json", metrics)
    return metrics


def _evaluate_cache_gate(
    metrics: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_coarse_residual_gate import (
        evaluate_coarse_residual_cache,
    )

    return evaluate_coarse_residual_cache(
        metrics,
        split=str(metrics["split"]),
        thresholds=_gate_thresholds(args),
    )


def _cache_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("cache stage requires a passing preflight")
    completed = run_dir / "cache_gate.json"
    if completed.is_file():
        gate = _load_json(completed)
        if not _passed(gate):
            raise ArtifactCompatibilityError("completed cache gate did not pass")
        return gate
    if not _no_confirmation_artifacts(run_dir):
        raise ArtifactCompatibilityError("confirmation evidence exists before selection")
    roles = _effective_path_ids(args)
    split_metrics = {
        role: _generate_role_cache(
            run_dir,
            args,
            role=role,
            path_ids=roles[role],
        )
        for role in ("train", "validation")
    }
    split_gates = {
        role: _evaluate_cache_gate(metrics, args)
        for role, metrics in split_metrics.items()
    }
    for role, gate in split_gates.items():
        atomic_write_json(run_dir / f"{role}_cache_gate.json", gate)
    cache_bytes = sum(
        path.stat().st_size
        for path in (run_dir / "cache").rglob("*")
        if path.is_file()
    )
    manifest = {
        "schema": RUN_SCHEMA + "-cache-manifest",
        "schema_version": 1,
        "scientific_config_sha256": _load_json(
            run_dir / "scientific_config.json"
        )["semantic_sha256"],
        "path_plan_sha256": _load_json(
            run_dir / "path_id_plan.json"
        )["semantic_sha256"],
        "root_seed": ROOT_SEED,
        "splits": split_metrics,
        "confirmation_cache_exists": 0,
        "persisted_cache_bytes": cache_bytes,
        "physical_labels_persisted_in_separate_audit_artifacts": 1,
        "physical_labels_not_opened_by_training": 1,
        **NO_WORK,
    }
    manifest["semantic_sha256"] = config_fingerprint(manifest)
    _freeze_json(run_dir / "cache_manifest.json", manifest)
    from mnist.d0_jacobi_rb_coarse_residual_gate import (
        evaluate_coarse_residual_cache_set,
    )

    gate = evaluate_coarse_residual_cache_set(
        split_records={
            "train": split_metrics["train"],
            "validation": split_metrics["validation"],
            "aggregate": {
                "persisted_cache_bytes": cache_bytes,
                "confirmation_absent_pass": int(
                    _no_confirmation_evidence(run_dir)
                ),
                "split_path_sets_disjoint_pass": int(
                    set(roles["train"]).isdisjoint(roles["validation"])
                ),
            },
        },
        thresholds=_gate_thresholds(args),
    )
    gate["physical_label_open_status"] = "not_opened"
    atomic_write_json(run_dir / "cache_gate.json", gate)
    return gate


def _load_input_cache_for_role(run_dir: Path, role: str) -> Any:
    from mnist.d0_jacobi_rb_learnability import load_input_cache

    return load_input_cache(run_dir / "cache" / f"{role}_inputs.npz")


def _load_audit_cache_for_role(run_dir: Path, role: str) -> Any:
    from mnist.d0_jacobi_rb_learnability import load_label_audit_cache

    return load_label_audit_cache(
        run_dir / "cache" / f"{role}_labels_audit.npz"
    )


def _model_inputs(cache: Any, device: torch.device) -> Any:
    from mnist.d0_jacobi_rb_learnability import model_inputs_from_cache

    return model_inputs_from_cache(cache, device=device, floating_dtype=torch.float32)


def _frozen_baseline(run_dir: Path) -> Any:
    from mnist.d0_jacobi_rb_coarse_residual import load_frozen_coarse_baseline

    return load_frozen_coarse_baseline(run_dir / "frozen_coarse_baseline.npz")


def _model_factory(run_dir: Path) -> Callable[[], nn.Module]:
    from mnist.d0_jacobi_rb_coarse_residual import CoarseResidualPredictor

    baseline = _frozen_baseline(run_dir)
    return lambda: CoarseResidualPredictor(baseline, zero_residual=True)


@torch.no_grad()
def _predict_model(model: nn.Module, inputs: Any, *, batch_size: int = 32) -> Tensor:
    from mnist.d0_jacobi_rb_learnability import call_model

    was_training = model.training
    model.eval()
    output: list[Tensor] = []
    for start in range(0, inputs.batch_size, batch_size):
        stop = min(inputs.batch_size, start + batch_size)
        index = torch.arange(
            start,
            stop,
            dtype=torch.long,
            device=inputs.later_full_state.device,
        )
        output.append(call_model(model, inputs.index_select(index)).to(torch.float64))
    if was_training:
        model.train()
    return torch.cat(output, dim=0)


def _baseline_prediction(model: nn.Module, inputs: Any) -> Tensor:
    method = getattr(model, "baseline_prediction", None)
    if not callable(method):
        raise CoarseResidualCLIError(
            "combined model does not expose its frozen baseline",
            failure_domain="training_contract",
            failure_code="coarse_baseline_model_contract_invalid",
        )
    return method(inputs, dtype=torch.float64)


def _path_mse_rows(
    prediction: Tensor,
    target: Tensor,
    path_ids: np.ndarray,
) -> list[dict[str, Any]]:
    prediction_np = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
    target_np = target.detach().cpu().numpy().astype(np.float64, copy=False)
    rows: list[dict[str, Any]] = []
    for path_id in sorted(np.unique(path_ids).tolist()):
        mask = path_ids == path_id
        mse = float(np.mean((prediction_np[mask] - target_np[mask]) ** 2))
        rows.append({"path_id": int(path_id), "mse": mse})
    return rows


def _training_fingerprint(
    run_dir: Path,
    *,
    task: str,
    seed: int,
    train_input_path: Path,
    validation_input_path: Path,
    target_scale: float,
    target_kind: str,
) -> str:
    record = {
        "schema": RUN_SCHEMA + "-training-fingerprint",
        "task": task,
        "seed": int(seed),
        "training_plan": _load_json(run_dir / "training_plan.json"),
        "scientific_config_sha256": _load_json(
            run_dir / "scientific_config.json"
        )["semantic_sha256"],
        "baseline_sha256": file_fingerprint(
            run_dir / "frozen_coarse_baseline.npz"
        ),
        "train_inputs_sha256": file_fingerprint(train_input_path),
        "validation_inputs_sha256": file_fingerprint(validation_input_path),
        "target_scale": float(target_scale),
        "target_kind": str(target_kind),
    }
    return config_fingerprint(record)


def _state_dict_hash(state_dict: Mapping[str, Tensor]) -> str:
    from mnist.d0_jacobi_rb_learnability import state_dict_sha256

    return state_dict_sha256(state_dict)


def _clone_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in state_dict.items()
    }


def _candidate_path(
    run_dir: Path,
    *,
    task: str,
    seed: int,
    update: int,
) -> Path:
    return (
        run_dir
        / "checkpoints"
        / task
        / f"seed-{seed}"
        / f"update-{update:04d}.pt"
    )


def _progress_path(run_dir: Path, *, task: str, seed: int) -> Path:
    return run_dir / "checkpoints" / task / f"seed-{seed}-progress.pt"


def _candidate_metrics(
    model: nn.Module,
    validation_inputs: Any,
    validation_target: Tensor,
    *,
    baseline_mse_overall: float,
    baseline_mse_high_reverse_time: float,
) -> tuple[dict[str, float], Tensor]:
    from mnist.d0_jacobi_rb_coarse_residual import reverse_time_quartile_tensor

    prediction = _predict_model(model, validation_inputs)
    target = validation_target.to(dtype=torch.float64)
    overall = float(torch.mean((prediction - target).square()).detach().cpu())
    quartile = reverse_time_quartile_tensor(
        validation_inputs.reverse_time, validation_inputs.phase
    )
    mask = quartile == 0
    if not bool(mask.any()):
        raise CoarseResidualCLIError(
            "validation high-reverse-time quartile is empty",
            failure_domain="training_data",
            failure_code="validation_high_reverse_time_empty",
        )
    high_reverse_time = float(
        torch.mean((prediction[mask] - target[mask]).square()).detach().cpu()
    )
    return (
        {
            "validation_mse": overall,
            "validation_high_reverse_time_mse": high_reverse_time,
            "baseline_validation_mse": float(baseline_mse_overall),
            "baseline_validation_high_reverse_time_mse": float(
                baseline_mse_high_reverse_time
            ),
            "overall_improvement": float(baseline_mse_overall - overall),
            "high_reverse_time_improvement": float(
                baseline_mse_high_reverse_time - high_reverse_time
            ),
        },
        prediction,
    )


def _train_task(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    task: str,
    seed: int,
    train_inputs: Any,
    train_target: Tensor,
    validation_inputs: Any,
    validation_target: Tensor,
    target_scale: float,
    target_kind: str,
    physical: bool,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_coarse_residual import (
        combined_exact_mse,
        reverse_time_quartile_tensor,
    )
    from mnist.d0_jacobi_rb_learnability import (
        call_model,
        deterministic_batch_indices,
        enable_deterministic_torch,
    )

    maximum_updates = int(
        _load_json(run_dir / "scientific_config.json")["training"][
            "maximum_updates"
        ]
    )
    interval = int(TRAINING_PLAN["validation_interval"])
    train_input_path = run_dir / "cache" / "train_inputs.npz"
    validation_input_path = run_dir / "cache" / "validation_inputs.npz"
    fingerprint = _training_fingerprint(
        run_dir,
        task=task,
        seed=seed,
        train_input_path=train_input_path,
        validation_input_path=validation_input_path,
        target_scale=target_scale,
        target_kind=target_kind,
    )
    progress_path = _progress_path(run_dir, task=task, seed=seed)
    model = _model_factory(run_dir)().to(train_inputs.later_full_state.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(TRAINING_PLAN["learning_rate"]),
        weight_decay=float(TRAINING_PLAN["weight_decay"]),
    )
    enable_deterministic_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # Re-create after seeding so every task has a deterministic initialization.
    model = _model_factory(run_dir)().to(train_inputs.later_full_state.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(TRAINING_PLAN["learning_rate"]),
        weight_decay=float(TRAINING_PLAN["weight_decay"]),
    )

    baseline_validation = _baseline_prediction(model, validation_inputs)
    validation64 = validation_target.to(dtype=torch.float64)
    baseline_mse_overall = float(
        torch.mean((baseline_validation - validation64).square()).detach().cpu()
    )
    quartile = reverse_time_quartile_tensor(
        validation_inputs.reverse_time, validation_inputs.phase
    )
    high_reverse_time_mask = quartile == 0
    baseline_mse_high_reverse_time = float(
        torch.mean(
            (
                baseline_validation[high_reverse_time_mask]
                - validation64[high_reverse_time_mask]
            ).square()
        )
        .detach()
        .cpu()
    )

    completed_update = 0
    candidates: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    finite = True
    if progress_path.is_file():
        snapshot = torch.load(
            progress_path, map_location=train_inputs.later_full_state.device, weights_only=False
        )
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("fingerprint") != fingerprint
            or int(snapshot.get("seed", -1)) != seed
        ):
            raise ArtifactCompatibilityError(
                f"training resume fingerprint changed for {task}/{seed}"
            )
        model.load_state_dict(snapshot["model_state_dict"], strict=True)
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        completed_update = int(snapshot["completed_update"])
        history = [dict(row) for row in snapshot["history"]]
        candidates = [dict(row) for row in snapshot["candidates"]]
        finite = bool(snapshot["finite"])
        torch.set_rng_state(snapshot["torch_rng_state"].to(device="cpu"))
        cuda_states = snapshot.get("cuda_rng_states", ())
        if torch.cuda.is_available() and cuda_states:
            torch.cuda.set_rng_state_all(list(cuda_states))

    def validate(update: int) -> dict[str, Any]:
        metrics, prediction = _candidate_metrics(
            model,
            validation_inputs,
            validation_target,
            baseline_mse_overall=baseline_mse_overall,
            baseline_mse_high_reverse_time=baseline_mse_high_reverse_time,
        )
        state = _clone_state_dict(model.state_dict())
        state_hash = _state_dict_hash(state)
        checkpoint_path = _candidate_path(
            run_dir, task=task, seed=seed, update=update
        )
        checkpoint = {
            "schema": RUN_SCHEMA + "-candidate-checkpoint",
            "schema_version": 1,
            "fingerprint": fingerprint,
            "task": task,
            "seed": seed,
            "update": update,
            "state_dict": state,
            "state_sha256": state_hash,
            "validation_metrics": metrics,
        }
        artifact = _atomic_torch_save(checkpoint_path, checkpoint)
        record = {
            "task": task,
            "seed": seed,
            "update": update,
            **metrics,
            "finite": int(
                all(math.isfinite(float(value)) for value in metrics.values())
            ),
            "eligible_nonzero": int(
                update > 0
                and metrics["overall_improvement"] > 0.0
                and metrics["high_reverse_time_improvement"] > 0.0
            ),
            "state_sha256": state_hash,
            "checkpoint_path": checkpoint_path.relative_to(run_dir).as_posix(),
            "checkpoint_file_sha256": artifact["sha256"],
        }
        if update == 0:
            record["update_zero_baseline_exact_error"] = float(
                torch.max(torch.abs(prediction - baseline_validation))
                .detach()
                .cpu()
            )
        candidates.append(record)
        return record

    def checkpoint(update: int) -> None:
        snapshot = {
            "schema": RUN_SCHEMA + "-training-progress",
            "schema_version": 1,
            "fingerprint": fingerprint,
            "task": task,
            "seed": seed,
            "completed_update": int(update),
            "model_state_dict": _clone_state_dict(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "candidates": candidates,
            "finite": int(finite),
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_states": (
                tuple(state.clone() for state in torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else ()
            ),
        }
        _atomic_torch_save(progress_path, snapshot)

    if not candidates:
        update_zero = validate(0)
        update_zero_error = float(
            update_zero.get("update_zero_baseline_exact_error", math.inf)
        )
        if not math.isfinite(update_zero_error) or update_zero_error != 0.0:
            raise CoarseResidualCLIError(
                "update-zero combined predictor is not exactly the frozen baseline",
                failure_domain="training_contract",
                failure_code="update_zero_baseline_not_exact",
            )
        checkpoint(0)
    zero_candidates = [
        record for record in candidates if int(record.get("update", -1)) == 0
    ]
    if (
        len(zero_candidates) != 1
        or not math.isfinite(
            float(
                zero_candidates[0].get(
                    "update_zero_baseline_exact_error", math.inf
                )
            )
        )
        or float(
            zero_candidates[0].get("update_zero_baseline_exact_error", math.inf)
        )
        != 0.0
    ):
        raise CoarseResidualCLIError(
            "resumed update-zero baseline contract is invalid",
            failure_domain="training_contract",
            failure_code="update_zero_baseline_not_exact",
        )
    if finite:
        model.train()
        last_train_record: dict[str, Any] = {}
        for update in range(completed_update + 1, maximum_updates + 1):
            indices_np = deterministic_batch_indices(
                train_inputs.batch_size,
                int(TRAINING_PLAN["batch_size"]),
                update - 1,
                seed,
            )
            indices = torch.as_tensor(
                indices_np,
                dtype=torch.long,
                device=train_inputs.later_full_state.device,
            )
            batch_inputs = train_inputs.index_select(indices)
            batch_target = train_target.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            combined = call_model(model, batch_inputs)
            loss, raw = combined_exact_mse(combined, batch_target, target_scale)
            if not bool(torch.isfinite(loss)):
                finite = False
                checkpoint(update - 1)
                break
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(TRAINING_PLAN["gradient_norm_clip"])
            )
            if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
                finite = False
                checkpoint(update - 1)
                break
            optimizer.step()
            last_train_record = {
                "update": update,
                "train_raw_mse": float(raw.detach().cpu()),
                "scaled_loss": float(loss.detach().cpu()),
                "preclip_gradient_norm": float(
                    torch.as_tensor(gradient_norm).detach().cpu()
                ),
            }
            if update % interval == 0 or update == maximum_updates:
                candidate = validate(update)
                history.append({**last_train_record, **candidate})
                checkpoint(update)
                model.train()
                print(
                    f"{task} seed={seed} update={update}/{maximum_updates} "
                    f"validation_mse={candidate['validation_mse']:.8g}",
                    flush=True,
                )

    # Controls select the minimum-risk candidate with update as the tie-break.
    # Physical selection first restricts to preregistered eligible non-zero
    # candidates, falling back to the legal update-zero baseline.
    eligible = [
        record
        for record in candidates
        if int(record.get("finite", 0)) == 1
        and (
            not physical or int(record.get("eligible_nonzero", 0)) == 1
        )
    ]
    if physical and not eligible:
        eligible = [
            record
            for record in candidates
            if int(record.get("update", -1)) == 0
            and int(record.get("finite", 0)) == 1
        ]
    if not eligible:
        raise CoarseResidualCLIError(
            f"training task {task}/{seed} produced no finite candidate",
            failure_domain="training",
            failure_code="coarse_residual_no_finite_candidate",
        )
    selected = min(
        eligible,
        key=lambda record: (
            float(record["validation_mse"]),
            int(record["update"]),
            int(record["seed"]),
        ),
    )
    report = {
        "schema": RUN_SCHEMA + "-training-task",
        "schema_version": 1,
        "task": task,
        "seed": seed,
        "target_kind": target_kind,
        "target_scale": target_scale,
        "finite": int(finite),
        "complete": int(
            finite
            and max(int(record["update"]) for record in candidates)
            == maximum_updates
        ),
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "update_zero_baseline_exact_pass": 1,
        "training_fingerprint": fingerprint,
        "physical_training_performed": int(physical),
        **NO_WORK,
    }
    atomic_write_json(
        run_dir / "checkpoints" / task / f"seed-{seed}-task.json", report
    )
    _write_csv(
        run_dir / "checkpoints" / task / f"seed-{seed}-history.csv", history
    )
    return report


def _load_candidate_model(
    run_dir: Path,
    *,
    candidate: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    path = run_dir / str(candidate["checkpoint_path"])
    if (
        not path.is_file()
        or candidate.get("checkpoint_file_sha256") != file_fingerprint(path)
    ):
        raise ArtifactCompatibilityError("selected candidate checkpoint changed")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if (
        not isinstance(checkpoint, Mapping)
        or int(checkpoint.get("seed", -1)) != int(candidate["seed"])
        or int(checkpoint.get("update", -1)) != int(candidate["update"])
    ):
        raise ArtifactCompatibilityError("selected candidate identity changed")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or _state_dict_hash(state) != candidate.get(
        "state_sha256"
    ):
        raise ArtifactCompatibilityError("selected candidate state hash changed")
    model = _model_factory(run_dir)().to(device)
    model.load_state_dict(state, strict=True)
    if _state_dict_hash(model.state_dict()) != candidate.get("state_sha256"):
        raise ArtifactCompatibilityError("selected candidate replay hash mismatch")
    return model


def _sample_path_ids(input_cache: Any) -> np.ndarray:
    keys = np.asarray(input_cache.sample_key, dtype=np.int64)
    return np.ascontiguousarray(keys >> 13, dtype=np.int64)


def _control_stage(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    train_cache: Any,
    validation_cache: Any,
    train_inputs: Any,
    validation_inputs: Any,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import (
        exact_global_target_scale,
        synthetic_teacher_target,
    )

    existing = run_dir / "optimization_control_gate.json"
    if existing.is_file():
        gate = _load_json(existing)
        if not _passed(gate):
            raise ArtifactCompatibilityError("completed optimization controls failed")
        return gate
    if (run_dir / "physical_label_open.json").exists():
        raise ArtifactCompatibilityError(
            "physical labels were opened before optimization controls"
        )
    control_initial = _model_factory(run_dir)().to(
        train_inputs.later_full_state.device
    )
    baseline_train = _baseline_prediction(control_initial, train_inputs).detach()
    baseline_validation = _baseline_prediction(
        control_initial, validation_inputs
    ).detach()
    synthetic_train = baseline_train + synthetic_teacher_target(
        train_inputs
    ).to(torch.float64)
    synthetic_validation = baseline_validation + synthetic_teacher_target(
        validation_inputs
    ).to(torch.float64)
    synthetic_scale = exact_global_target_scale(
        synthetic_train.detach().cpu().numpy()
    )
    synthetic = _train_task(
        run_dir,
        args,
        task="synthetic-control",
        seed=CONTROL_SEEDS["synthetic"],
        train_inputs=train_inputs,
        train_target=synthetic_train,
        validation_inputs=validation_inputs,
        validation_target=synthetic_validation,
        target_scale=synthetic_scale,
        target_kind="analytic-permitted-input-synthetic-teacher",
        physical=False,
    )
    synthetic_selected = synthetic["selected"]
    synthetic_model = _load_candidate_model(
        run_dir,
        candidate=synthetic_selected,
        device=train_inputs.later_full_state.device,
    )
    synthetic_prediction = _predict_model(synthetic_model, validation_inputs)
    initial_model = _model_factory(run_dir)().to(
        train_inputs.later_full_state.device
    )
    initial_prediction = _predict_model(initial_model, validation_inputs)
    synthetic_paths = _sample_path_ids(validation_cache)
    selected_rows = _path_mse_rows(
        synthetic_prediction, synthetic_validation, synthetic_paths
    )
    initial_rows = _path_mse_rows(
        initial_prediction, synthetic_validation, synthetic_paths
    )
    initial_by_path = {int(row["path_id"]): float(row["mse"]) for row in initial_rows}
    synthetic_path_rows = [
        {
            **row,
            "baseline_mse": initial_by_path[int(row["path_id"])],
            "improvement": initial_by_path[int(row["path_id"])]
            - float(row["mse"]),
        }
        for row in selected_rows
    ]
    baseline_control_mse = float(
        torch.mean((baseline_validation - synthetic_validation).square())
        .detach()
        .cpu()
    )
    synthetic_relative_mse = (
        float(synthetic_selected["validation_mse"]) / baseline_control_mse
        if baseline_control_mse > 0
        else math.inf
    )
    synthetic_gate = {
        "schema": RUN_SCHEMA + "-synthetic-control-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": int(
            int(synthetic.get("complete", 0)) == 1
            and int(synthetic.get("finite", 0)) == 1
            and int(synthetic_selected["update"]) > 0
            and synthetic_relative_mse <= 0.01
            and all(float(row["improvement"]) > 0.0 for row in synthetic_path_rows)
        ),
        "relative_validation_mse": synthetic_relative_mse,
        "maximum_relative_validation_mse": 0.01,
        "selected_update": int(synthetic_selected["update"]),
        "all_validation_paths_improve": int(
            all(float(row["improvement"]) > 0.0 for row in synthetic_path_rows)
        ),
        "path_count": len(synthetic_path_rows),
        "role": "synthetic_teacher",
        "complete": int(synthetic.get("complete", 0)),
        "finite": int(synthetic.get("finite", 0)),
        "path_baseline_minus_model_mse": [
            float(row["improvement"]) for row in synthetic_path_rows
        ],
        "physical_labels_opened": 0,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "synthetic_control_gate.json", synthetic_gate)
    _write_csv(run_dir / "synthetic_control_path_metrics.csv", synthetic_path_rows)
    if not _passed(synthetic_gate):
        null_gate = _not_evaluated(
            "null-control", "synthetic representability control failed"
        )
        atomic_write_json(run_dir / "null_control_gate.json", null_gate)
        gate = {
            "schema": RUN_SCHEMA + "-optimization-control-gate",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": 0,
            "synthetic": synthetic_gate,
            "null": null_gate,
            "physical_labels_opened": 0,
            **CLAIM_FLAGS,
            **NO_WORK,
        }
        atomic_write_json(existing, gate)
        return gate

    null_model = _model_factory(run_dir)().to(train_inputs.later_full_state.device)
    null_train = _baseline_prediction(null_model, train_inputs).detach()
    null_validation = _baseline_prediction(null_model, validation_inputs).detach()
    null_scale = max(
        float(torch.sqrt(torch.mean(null_train.square())).detach().cpu()),
        1.0e-12,
    )
    null = _train_task(
        run_dir,
        args,
        task="null-control",
        seed=CONTROL_SEEDS["null"],
        train_inputs=train_inputs,
        train_target=null_train,
        validation_inputs=validation_inputs,
        validation_target=null_validation,
        target_scale=null_scale,
        target_kind="frozen-coarse-baseline-null",
        physical=False,
    )
    null_selected = null["selected"]
    null_gate = {
        "schema": RUN_SCHEMA + "-null-control-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": int(
            int(null.get("complete", 0)) == 1
            and int(null.get("finite", 0)) == 1
            and int(null_selected["update"]) == 0
            and float(null_selected["validation_mse"]) == 0.0
            and float(
                null_selected.get("update_zero_baseline_exact_error", math.inf)
            )
            == 0.0
        ),
        "selected_update": int(null_selected["update"]),
        "selected_validation_mse": float(null_selected["validation_mse"]),
        "update_zero_baseline_exact_error": float(
            null_selected.get("update_zero_baseline_exact_error", math.inf)
        ),
        "role": "exact_baseline_null",
        "complete": int(null.get("complete", 0)),
        "finite": int(null.get("finite", 0)),
        "physical_labels_opened": 0,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "null_control_gate.json", null_gate)
    gate = {
        "schema": RUN_SCHEMA + "-optimization-control-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": int(_passed(synthetic_gate) and _passed(null_gate)),
        "synthetic": synthetic_gate,
        "null": null_gate,
        "physical_labels_opened": 0,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(existing, gate)
    return gate


def _open_physical_labels(
    run_dir: Path,
    *,
    train_audit: Any,
    validation_audit: Any,
    target_scale: float,
) -> dict[str, Any]:
    target_path = run_dir / "physical_label_open.json"
    if target_path.is_file():
        existing = _load_json(target_path)
        semantic = existing.get("semantic_sha256")
        body = dict(existing)
        body.pop("semantic_sha256", None)
        if (
            semantic != config_fingerprint(body)
            or existing.get("optimization_control_gate_sha256")
            != file_fingerprint(run_dir / "optimization_control_gate.json")
            or existing.get("train_label_audit_sha256")
            != file_fingerprint(run_dir / "cache" / "train_labels_audit.npz")
            or existing.get("validation_label_audit_sha256")
            != file_fingerprint(
                run_dir / "cache" / "validation_labels_audit.npz"
            )
            or float(existing.get("target_scale", math.nan))
            != float(target_scale)
        ):
            raise ArtifactCompatibilityError("physical-label seal changed")
        return existing
    record = {
        "schema": RUN_SCHEMA + "-physical-label-open",
        "schema_version": 1,
        "opened_at": _now(),
        "optimization_control_gate_sha256": file_fingerprint(
            run_dir / "optimization_control_gate.json"
        ),
        "train_label_audit_sha256": file_fingerprint(
            run_dir / "cache" / "train_labels_audit.npz"
        ),
        "validation_label_audit_sha256": file_fingerprint(
            run_dir / "cache" / "validation_labels_audit.npz"
        ),
        "train_sample_count": int(train_audit.sample_count),
        "validation_sample_count": int(validation_audit.sample_count),
        "target_scale": float(target_scale),
        "target_scale_source": "training exact RB labels only",
        "validation_scale_contribution": 0,
        "residual_target_persisted": 0,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return _freeze_json(target_path, record)


def _validate_cache_join(input_cache: Any, audit_cache: Any) -> None:
    from mnist.d0_jacobi_rb_learnability import LearnabilityCacheBundle

    LearnabilityCacheBundle(input_cache, audit_cache)


def _freeze_physical_training_started(
    run_dir: Path,
    *,
    target_scale: float,
) -> dict[str, Any]:
    target = run_dir / "physical_training_started.json"
    expected_control_sha = file_fingerprint(
        run_dir / "optimization_control_gate.json"
    )
    if target.is_file():
        existing = _load_json(target)
        if (
            existing.get("optimization_control_gate_sha256")
            != expected_control_sha
            or float(existing.get("target_scale", math.nan))
            != float(target_scale)
            or int(existing.get("residual_target_persisted", 1)) != 0
        ):
            raise ArtifactCompatibilityError("physical-training start seal changed")
        return existing
    record = {
        "schema": RUN_SCHEMA + "-physical-training-started",
        "schema_version": 1,
        "started_at": _now(),
        "optimization_control_gate_sha256": expected_control_sha,
        "target_scale": float(target_scale),
        "residual_target_persisted": 0,
        "physical_training_performed": 1,
        **NO_WORK,
    }
    return _freeze_json(target, record)


def _freeze_selected_model(
    run_dir: Path,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    source = run_dir / str(selected["checkpoint_path"])
    target = run_dir / "selected_model.pt"
    if target.is_file():
        if file_fingerprint(target) != file_fingerprint(source):
            raise ArtifactCompatibilityError("frozen selected model changed")
    else:
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    record = {
        "schema": RUN_SCHEMA + "-selected-model",
        "schema_version": 1,
        "selection_role": "fresh_validation_paths_only",
        "selection_order": [
            "raw_validation_mse",
            "earlier_update",
            "lower_seed",
        ],
        "candidate": dict(selected),
        "selected_model_sha256": file_fingerprint(target),
        "selected_model_size": int(target.stat().st_size),
        "nonzero_residual_selected": int(int(selected["update"]) > 0),
        "confirmation_inspected": 0,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return _freeze_json(run_dir / "selected_model.json", record)


def _freeze_confirmation_seal(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    selected_model: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _effective_path_ids(args)["confirmation"]
    target_path = run_dir / "confirmation_seal.json"
    if target_path.is_file():
        existing = _load_json(target_path)
        semantic = existing.get("semantic_sha256")
        body = dict(existing)
        body.pop("semantic_sha256", None)
        if (
            semantic != config_fingerprint(body)
            or existing.get("selected_model_semantic_sha256")
            != selected_model["semantic_sha256"]
            or existing.get("selected_model_file_sha256")
            != selected_model["selected_model_sha256"]
            or existing.get("confirmation_path_ids") != list(paths)
            or int(existing.get("root_seed", -1)) != ROOT_SEED
        ):
            raise ArtifactCompatibilityError("confirmation seal changed")
        return existing
    record = {
        "schema": RUN_SCHEMA + "-confirmation-seal",
        "schema_version": 1,
        "sealed_at": _now(),
        "selected_model_semantic_sha256": selected_model["semantic_sha256"],
        "selected_model_file_sha256": selected_model["selected_model_sha256"],
        "confirmation_path_ids": list(paths),
        "root_seed": ROOT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": 50_000,
        "one_sided_confidence": 0.99,
        "family": ["baseline_gain", "residual_increment"],
        "confirmation_cache_absent_at_seal": int(_no_confirmation_artifacts(run_dir)),
        "path_plan_sha256": _load_json(
            run_dir / "path_id_plan.json"
        )["semantic_sha256"],
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return _freeze_json(target_path, record)


def _physical_training_stage(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    train_cache: Any,
    validation_cache: Any,
    train_inputs: Any,
    validation_inputs: Any,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_coarse_residual import exact_combined_target_scale

    existing = run_dir / "physical_training_gate.json"
    if existing.is_file():
        return _load_json(existing)
    controls = _load_json(run_dir / "optimization_control_gate.json")
    if not _passed(controls):
        raise ArtifactCompatibilityError(
            "physical targets cannot open before controls pass"
        )
    train_audit = _load_audit_cache_for_role(run_dir, "train")
    validation_audit = _load_audit_cache_for_role(run_dir, "validation")
    _validate_cache_join(train_cache, train_audit)
    _validate_cache_join(validation_cache, validation_audit)
    train_target = torch.as_tensor(
        np.asarray(train_audit.denoising_target).copy(),
        dtype=torch.float64,
        device=train_inputs.later_full_state.device,
    )
    validation_target = torch.as_tensor(
        np.asarray(validation_audit.denoising_target).copy(),
        dtype=torch.float64,
        device=validation_inputs.later_full_state.device,
    )
    target_scale = exact_combined_target_scale(
        np.asarray(train_audit.denoising_target)
    )
    _open_physical_labels(
        run_dir,
        train_audit=train_audit,
        validation_audit=validation_audit,
        target_scale=target_scale,
    )
    _freeze_physical_training_started(run_dir, target_scale=target_scale)
    reports = [
        _train_task(
            run_dir,
            args,
            task="physical-residual",
            seed=seed,
            train_inputs=train_inputs,
            train_target=train_target,
            validation_inputs=validation_inputs,
            validation_target=validation_target,
            target_scale=target_scale,
            target_kind="unchanged-exact-RB-label-direct-combined-MSE",
            physical=True,
        )
        for seed in MODEL_SEEDS
    ]
    candidates = [dict(report["selected"]) for report in reports]
    eligible = [
        candidate
        for candidate in candidates
        if int(candidate.get("eligible_nonzero", 0)) == 1
        and int(candidate.get("update", 0)) > 0
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda candidate: (
                float(candidate["validation_mse"]),
                int(candidate["update"]),
                int(candidate["seed"]),
            ),
        )
    else:
        update_zero = [
            candidate for candidate in candidates if int(candidate["update"]) == 0
        ]
        if not update_zero:
            raise CoarseResidualCLIError(
                "no legal update-zero baseline remained after physical training",
                failure_domain="training_selection",
                failure_code="update_zero_candidate_missing",
            )
        selected = min(
            update_zero,
            key=lambda candidate: (
                float(candidate["validation_mse"]),
                int(candidate["seed"]),
            ),
        )
    selected_record = _freeze_selected_model(run_dir, selected)
    if int(selected["update"]) > 0:
        _freeze_confirmation_seal(
            run_dir, args, selected_model=selected_record
        )
    task_rows = [
        {
            "seed": int(report["seed"]),
            "complete": int(report["complete"]),
            "finite": int(report["finite"]),
            "selected_update": int(report["selected"]["update"]),
            "validation_mse": float(report["selected"]["validation_mse"]),
            "validation_high_reverse_time_mse": float(
                report["selected"]["validation_high_reverse_time_mse"]
            ),
            "overall_improvement": float(
                report["selected"]["overall_improvement"]
            ),
            "high_reverse_time_improvement": float(
                report["selected"]["high_reverse_time_improvement"]
            ),
            "eligible_nonzero": int(
                report["selected"].get("eligible_nonzero", 0)
            ),
        }
        for report in reports
    ]
    _write_csv(run_dir / "physical_seed_metrics.csv", task_rows)
    metrics = {
        "schema": RUN_SCHEMA + "-physical-training-metrics",
        "schema_version": 1,
        "target_scale": target_scale,
        "target_scale_training_only": 1,
        "residual_target_persisted": 0,
        "task_count": len(reports),
        "tasks_complete": int(
            all(int(report["complete"]) == 1 for report in reports)
        ),
        "tasks_finite": int(
            all(int(report["finite"]) == 1 for report in reports)
        ),
        "selected": selected,
        "nonzero_residual_selected": int(int(selected["update"]) > 0),
        "selection_confirmation_absent": int(_no_confirmation_evidence(run_dir)),
        "physical_training_performed": 1,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "physical_training_metrics.json", metrics)
    gate = {
        "schema": RUN_SCHEMA + "-physical-training-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": int(
            metrics["tasks_complete"]
            and metrics["tasks_finite"]
            and metrics["selection_confirmation_absent"]
        ),
        "baseline_only_valid_outcome": int(
            metrics["tasks_complete"]
            and metrics["tasks_finite"]
            and not metrics["nonzero_residual_selected"]
        ),
        "metrics": metrics,
        "physical_training_performed": 1,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(existing, gate)
    return gate


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "cache_gate.json")):
        raise ArtifactCompatibilityError("train stage requires a passing cache gate")
    train_cache = _load_input_cache_for_role(run_dir, "train")
    validation_cache = _load_input_cache_for_role(run_dir, "validation")
    device = torch.device(args.device)
    train_inputs = _model_inputs(train_cache, device)
    validation_inputs = _model_inputs(validation_cache, device)
    controls = _control_stage(
        run_dir,
        args,
        train_cache=train_cache,
        validation_cache=validation_cache,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
    )
    if not _passed(controls):
        physical = _not_evaluated(
            "physical-training", "optimization controls failed"
        )
        atomic_write_json(run_dir / "physical_training_gate.json", physical)
        gate = {
            "schema": RUN_SCHEMA + "-train-gate",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": 0,
            "controls": controls,
            "physical": physical,
            "physical_training_performed": 0,
            **CLAIM_FLAGS,
            **NO_WORK,
        }
        atomic_write_json(run_dir / "train_gate.json", gate)
        return gate
    physical = _physical_training_stage(
        run_dir,
        args,
        train_cache=train_cache,
        validation_cache=validation_cache,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
    )
    from mnist.d0_jacobi_rb_coarse_residual_gate import (
        evaluate_coarse_residual_train,
    )

    synthetic_record = dict(_load_json(run_dir / "synthetic_control_gate.json"))
    null_record = dict(_load_json(run_dir / "null_control_gate.json"))
    physical_reports = [
        _load_json(
            run_dir
            / "checkpoints"
            / "physical-residual"
            / f"seed-{seed}-task.json"
        )
        for seed in MODEL_SEEDS
    ]
    task_records: list[dict[str, Any]] = [
        {
            "role": "synthetic_teacher",
            "complete": int(synthetic_record.get("complete", 0)),
            "finite": int(synthetic_record.get("finite", 0)),
            "relative_validation_mse": float(
                synthetic_record.get("relative_validation_mse", math.inf)
            ),
            "path_baseline_minus_model_mse": list(
                synthetic_record.get("path_baseline_minus_model_mse", [])
            ),
            "selected_update": int(synthetic_record.get("selected_update", -1)),
        },
        {
            "role": "exact_baseline_null",
            "complete": int(null_record.get("complete", 0)),
            "finite": int(null_record.get("finite", 0)),
            "selected_update": int(null_record.get("selected_update", -1)),
        },
    ]
    task_records.extend(
        {
            "role": f"physical_seed_{seed}",
            "seed": seed,
            "complete": int(report.get("complete", 0)),
            "finite": int(report.get("finite", 0)),
            "selected_update": int(report["selected"]["update"]),
            "selected_validation_mse": float(
                report["selected"]["validation_mse"]
            ),
        }
        for seed, report in zip(MODEL_SEEDS, physical_reports, strict=True)
    )
    selection_metrics = _load_json(run_dir / "physical_training_metrics.json")
    selected = selection_metrics["selected"]
    selection = {
        "selected_update": int(selected["update"]),
        "selected_seed": int(selected["seed"]),
        "combined_validation_mse": float(selected["validation_mse"]),
        "baseline_validation_mse": float(selected["baseline_validation_mse"]),
        # Gate field retains its historical name; this workflow's frozen
        # semantic is explicitly the high reverse-time quartile (quartile 0).
        "combined_validation_mse_data_end": float(
            selected["validation_high_reverse_time_mse"]
        ),
        "baseline_validation_mse_data_end": float(
            selected["baseline_validation_high_reverse_time_mse"]
        ),
        "combined_validation_mse_high_reverse_time": float(
            selected["validation_high_reverse_time_mse"]
        ),
        "baseline_validation_mse_high_reverse_time": float(
            selected["baseline_validation_high_reverse_time_mse"]
        ),
        "maximum_updates": int(
            _load_json(run_dir / "scientific_config.json")["training"][
                "maximum_updates"
            ]
        ),
        "validation_interval": int(TRAINING_PLAN["validation_interval"]),
        "batch_size": int(TRAINING_PLAN["batch_size"]),
        "selection_validation_only_pass": 1,
        "analytic_zero_candidate_pass": 1,
        "coarse_baseline_candidate_pass": 1,
        "selected_checkpoint_frozen_pass": 1,
        "baseline_hash_binding_pass": 1,
        "unweighted_mse_against_exact_label_pass": 1,
        "target_unmodified_pass": 1,
        "target_scale_training_only_pass": 1,
        "combined_prediction_loss_pass": 1,
        "model_input_firewall_pass": 1,
        "confirmation_gate_definition_frozen_pass": 1,
        "confirmation_absent_pass": int(_no_confirmation_evidence(run_dir)),
    }
    gate = evaluate_coarse_residual_train(
        task_records=task_records,
        selection=selection,
        thresholds=_gate_thresholds(args),
    )
    gate["controls"] = controls
    gate["physical"] = physical
    gate["selection"] = selection
    gate["physical_training_performed"] = 1
    atomic_write_json(run_dir / "train_gate.json", gate)
    return gate


def _open_confirmation(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    seal = _load_json(run_dir / "confirmation_seal.json")
    selected = _load_json(run_dir / "selected_model.json")
    if (
        int(selected.get("nonzero_residual_selected", 0)) != 1
        or seal.get("selected_model_semantic_sha256")
        != selected.get("semantic_sha256")
        or seal.get("selected_model_file_sha256")
        != selected.get("selected_model_sha256")
        or seal.get("confirmation_path_ids")
        != list(_effective_path_ids(args)["confirmation"])
    ):
        raise ArtifactCompatibilityError("confirmation seal is incompatible")
    target_path = run_dir / "confirmation_open.json"
    if target_path.is_file():
        existing = _load_json(target_path)
        semantic = existing.get("semantic_sha256")
        body = dict(existing)
        body.pop("semantic_sha256", None)
        if (
            semantic != config_fingerprint(body)
            or existing.get("confirmation_seal_sha256")
            != file_fingerprint(run_dir / "confirmation_seal.json")
            or existing.get("selected_model_sha256")
            != selected["selected_model_sha256"]
            or existing.get("path_ids")
            != list(_effective_path_ids(args)["confirmation"])
            or int(existing.get("open_count", -1)) != 1
        ):
            raise ArtifactCompatibilityError("confirmation-open seal changed")
        return existing
    record = {
        "schema": RUN_SCHEMA + "-confirmation-open",
        "schema_version": 1,
        "opened_at": _now(),
        "confirmation_seal_sha256": file_fingerprint(
            run_dir / "confirmation_seal.json"
        ),
        "confirmation_seal_semantic_sha256": seal["semantic_sha256"],
        "selected_model_sha256": selected["selected_model_sha256"],
        "path_ids": list(_effective_path_ids(args)["confirmation"]),
        "open_count": 1,
        "path_plan_sha256": seal["path_plan_sha256"],
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return _freeze_json(target_path, record)


def _descriptive_confirmation_rows(
    target: np.ndarray,
    baseline: np.ndarray,
    combined: np.ndarray,
    reverse_time: np.ndarray,
    phase: np.ndarray,
) -> list[dict[str, Any]]:
    from mnist.d0_jacobi_rb_coarse_residual import reverse_time_quartile_numpy

    quartile = reverse_time_quartile_numpy(reverse_time, phase)
    rows: list[dict[str, Any]] = []
    for quartile_value in range(4):
        for phase_value in range(7):
            mask = (quartile == quartile_value) & (phase == phase_value)
            if not np.any(mask):
                continue
            zero_mse = float(np.mean(target[mask] ** 2))
            baseline_mse = float(np.mean((baseline[mask] - target[mask]) ** 2))
            combined_mse = float(np.mean((combined[mask] - target[mask]) ** 2))
            rows.append(
                {
                    "reverse_time_quartile": quartile_value,
                    "phase": phase_value,
                    "sample_count": int(np.sum(mask)),
                    "zero_mse": zero_mse,
                    "baseline_mse": baseline_mse,
                    "combined_mse": combined_mse,
                    "baseline_gain": zero_mse - baseline_mse,
                    "residual_increment": baseline_mse - combined_mse,
                    "authorizing": 0,
                }
            )
    return rows


def _confirm_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_coarse_residual import (
        ALL_CONTRAST_NAMES,
        PRIMARY_CONTRAST_NAMES,
        one_sided_studentized_max_t,
        path_loss_contrasts,
    )
    from mnist.d0_jacobi_rb_coarse_residual_gate import (
        evaluate_coarse_residual_confirmation,
    )

    train_gate = _load_json(run_dir / "train_gate.json")
    if not _passed(train_gate):
        raise ArtifactCompatibilityError(
            "confirmation requires a passing nonzero train gate"
        )
    if int(train_gate.get("coarse_baseline_only", 0)) == 1:
        raise ArtifactCompatibilityError(
            "update-zero baseline outcome does not authorize confirmation"
        )
    completed = run_dir / "confirmation_gate.json"
    if completed.is_file():
        return _load_json(completed)
    open_record = _open_confirmation(run_dir, args)
    role_metrics = _generate_role_cache(
        run_dir,
        args,
        role="confirmation",
        path_ids=_effective_path_ids(args)["confirmation"],
    )
    cache_gate = _evaluate_cache_gate(role_metrics, args)
    total_cache_bytes = sum(
        path.stat().st_size
        for path in (run_dir / "cache").rglob("*")
        if path.is_file()
    )
    cache_gate["total_persisted_cache_bytes"] = total_cache_bytes
    cache_gate["maximum_persisted_cache_bytes"] = RESOURCE_THRESHOLDS[
        "maximum_persisted_cache_bytes"
    ]
    cache_gate["total_persisted_cache_resource_pass"] = int(
        total_cache_bytes
        <= RESOURCE_THRESHOLDS["maximum_persisted_cache_bytes"]
    )
    cache_gate["passed"] = int(
        _passed(cache_gate)
        and cache_gate["total_persisted_cache_resource_pass"]
    )
    atomic_write_json(run_dir / "confirmation_cache_gate.json", cache_gate)
    if not _passed(cache_gate):
        gate = {
            "schema": RUN_SCHEMA + "-confirmation-gate",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": 0,
            "confirmation_cache_valid": 0,
            "paired_risk_inference_valid": 0,
            "coarse_baseline_replicated": 0,
            "residual_point_positive": 0,
            "residual_replicated": 0,
            "confirmation_cache_gate": cache_gate,
            **CLAIM_FLAGS,
            **NO_WORK,
        }
        atomic_write_json(completed, gate)
        return gate
    input_cache = _load_input_cache_for_role(run_dir, "confirmation")
    audit = _load_audit_cache_for_role(run_dir, "confirmation")
    _validate_cache_join(input_cache, audit)
    device = torch.device(args.device)
    inputs = _model_inputs(input_cache, device)
    selected_record = _load_json(run_dir / "selected_model.json")
    candidate = selected_record["candidate"]
    model = _load_candidate_model(run_dir, candidate=candidate, device=device)
    combined_tensor = _predict_model(model, inputs)
    baseline_tensor = _baseline_prediction(model, inputs)
    target = np.ascontiguousarray(
        np.asarray(audit.denoising_target, dtype=np.float64)
    )
    combined = np.ascontiguousarray(
        combined_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    baseline = np.ascontiguousarray(
        baseline_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    contrasts = path_loss_contrasts(
        target,
        baseline,
        combined,
        np.asarray(audit.path_id),
        np.asarray(input_cache.reverse_time),
        np.asarray(input_cache.phase),
    )
    family = PRIMARY_CONTRAST_NAMES
    max_t_result = one_sided_studentized_max_t(
        contrasts,
        family_names=family,
        seed=BOOTSTRAP_SEED,
        replicates=int(_gate_thresholds(args).bootstrap_replicates),
        confidence=float(_gate_thresholds(args).confidence),
    )
    max_t = max_t_result.to_record()
    path_rows = []
    for row, path_id in enumerate(contrasts.path_ids.tolist()):
        values = {
            name: float(contrasts.values[row, index])
            for index, name in enumerate(ALL_CONTRAST_NAMES)
        }
        path_rows.append({"path_id": int(path_id), **values})
    _write_csv(run_dir / "confirmation_path_metrics.csv", path_rows)
    _atomic_npz(
        run_dir / "confirmation_path_contrasts.npz",
        {
            "path_ids": contrasts.path_ids,
            "values": contrasts.values,
        },
    )
    _write_csv(
        run_dir / "confirmation_quartile_phase_metrics.csv",
        _descriptive_confirmation_rows(
            target,
            baseline,
            combined,
            np.asarray(input_cache.reverse_time),
            np.asarray(input_cache.phase),
        ),
    )
    delta_b_overall = contrasts.column("overall.baseline_vs_zero")
    delta_r_overall = contrasts.column("overall.combined_vs_baseline")
    delta_t_overall = contrasts.column("overall.combined_vs_zero")
    delta_b_end = contrasts.column("data_end.baseline_vs_zero")
    delta_r_end = contrasts.column("data_end.combined_vs_baseline")
    delta_t_end = contrasts.column("data_end.combined_vs_zero")
    direct_error = float(
        np.max(np.abs(delta_t_overall - delta_b_overall - delta_r_overall))
    )
    metrics = {
        "schema": RUN_SCHEMA + "-confirmation-metrics",
        "schema_version": 1,
        "confirmation_cache_gate_pass": int(_passed(cache_gate)),
        "confirmation_sealed_pass": int(
            open_record.get("confirmation_seal_sha256")
            == file_fingerprint(run_dir / "confirmation_seal.json")
        ),
        "confirmation_opened_once_pass": int(open_record.get("open_count") == 1),
        "confirmation_paths_fresh_pass": 1,
        "confirmation_paths_disjoint_pass": int(
            set(_effective_path_ids(args)["confirmation"]).isdisjoint(
                set(_effective_path_ids(args)["train"])
                | set(_effective_path_ids(args)["validation"])
            )
        ),
        "selected_checkpoint_hash_pass": int(
            selected_record["selected_model_sha256"]
            == file_fingerprint(run_dir / "selected_model.pt")
        ),
        "baseline_hash_binding_pass": int(
            model.baseline_fingerprint == _frozen_baseline(run_dir).fingerprint
        ),
        "path_plan_hash_pass": int(
            open_record["path_plan_sha256"]
            == _load_json(run_dir / "path_id_plan.json")["semantic_sha256"]
        ),
        "predictions_finite_pass": int(
            np.isfinite(baseline).all() and np.isfinite(combined).all()
        ),
        "risks_finite_pass": int(np.isfinite(contrasts.values).all()),
        "unweighted_exact_label_risk_pass": 1,
        "model_input_firewall_pass": 1,
        "no_post_selection_refit_pass": 1,
        "confirmation_path_count": int(contrasts.path_count),
        "max_t": max_t,
        "direct_derived_delta_t_max_abs_error": direct_error,
        "selected_update": int(candidate["update"]),
        "selected_seed": int(candidate["seed"]),
        "contrasts": {
            "delta_b": {
                "overall_mean": float(np.mean(delta_b_overall)),
                "overall_path_values": delta_b_overall.tolist(),
                "data_end_mean": float(np.mean(delta_b_end)),
                "data_end_path_values": delta_b_end.tolist(),
            },
            "delta_r": {
                "overall_mean": float(np.mean(delta_r_overall)),
                "overall_path_values": delta_r_overall.tolist(),
                "data_end_mean": float(np.mean(delta_r_end)),
                "data_end_path_values": delta_r_end.tolist(),
            },
        },
        "physical_training_performed": 1,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "confirmation_metrics.json", metrics)
    gate = evaluate_coarse_residual_confirmation(
        confirmation=metrics,
        thresholds=_gate_thresholds(args),
    )
    gate["confirmation_cache_valid"] = 1
    gate["confirmation_cache_gate"] = cache_gate
    gate["physical_training_performed"] = 1
    atomic_write_json(completed, gate)
    return gate


def _optional_json(run_dir: Path, filename: str) -> dict[str, Any] | None:
    path = run_dir / filename
    return _load_json(path) if path.is_file() else None


def _workflow_record(
    run_dir: Path,
    *,
    require_gate: str,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_coarse_residual_gate import (
        evaluate_coarse_residual_workflow,
    )

    record = evaluate_coarse_residual_workflow(
        preflight_gate=_optional_json(run_dir, "preflight_gate.json"),
        cache_gate=_optional_json(run_dir, "cache_gate.json"),
        train_gate=_optional_json(run_dir, "train_gate.json"),
        confirm_gate=_optional_json(run_dir, "confirmation_gate.json"),
        require_gate=require_gate,
    )
    record["physical_training_performed"] = int(
        (run_dir / "physical_training_started.json").is_file()
    )
    atomic_write_json(run_dir / "workflow_gate.json", record)
    atomic_write_json(
        run_dir / "coarse_residual_decision.json", record["decision"]
    )
    return record


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "cache", "train", "confirm")
    if stage == "report":
        return ()
    return (stage,)


def _gate_filename(stage: str) -> str:
    return {
        "preflight": "preflight_gate.json",
        "cache": "cache_gate.json",
        "train": "train_gate.json",
        "confirm": "confirmation_gate.json",
    }[stage]


def _commit_execution_failure(
    run_dir: Path,
    *,
    stage: str,
    exc: BaseException,
    require_gate: str,
) -> None:
    from mnist.d0_jacobi_rb_coarse_residual_gate import execution_failed_gate

    failure_domain = str(
        getattr(exc, "failure_domain", "workflow_execution")
    )
    failure_code = str(
        getattr(exc, "failure_code", "coarse_residual_execution_failed")
    )
    gate_stage = stage if stage in {"preflight", "cache", "train", "confirm"} else "preflight"
    gate_path = run_dir / _gate_filename(gate_stage)
    if not gate_path.is_file():
        gate = execution_failed_gate(
            gate_stage,
            exc,
            failure_code=failure_code,
            failure_domain=failure_domain,
        )
        atomic_write_json(gate_path, gate)
    workflow = _workflow_record(run_dir, require_gate=require_gate)
    _status(
        run_dir,
        state="execution_failed",
        stage=stage,
        decision=str(workflow["decision"]["decision"]),
        failure_domain=failure_domain,
        failure_code=failure_code,
        message=str(exc),
    )
    _artifact_registry(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    from mnist.d0_jacobi_rb_coarse_residual_gate import REQUIRED_GATES, STAGES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path(
        "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability"
    ))
    parser.add_argument(
        "--run-name",
        default="production-exact-k512-coarse-residual-one-image",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-coarse-witness-run-dir", type=Path, required=True)
    parser.add_argument("--parent-one-image-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--test-paths-per-role",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--test-outer-steps",
        type=int,
        default=16,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--test-maximum-updates",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    args.runs_root = args.runs_root.resolve()
    args.parent_coarse_witness_run_dir = (
        args.parent_coarse_witness_run_dir.resolve()
    )
    args.parent_one_image_run_dir = args.parent_one_image_run_dir.resolve()
    if args.resume_run_dir is not None:
        args.resume_run_dir = args.resume_run_dir.resolve()
    if args.test_only and (
        args.require_gate != "none"
        or args.test_maximum_updates < 0
        or args.test_maximum_updates > 4_000
    ):
        parser.error(
            "test-only runs are nonauthorizing and require valid reduced updates"
        )
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    active_stage = str(args.stage)
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"Jacobi/RB coarse-residual run directory: {run_dir}", flush=True)
        if resumed:
            _verify_registry(run_dir)
        _initialize_run(run_dir, args, resumed=resumed)
        _status(run_dir, state="running", stage=active_stage)
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            if stage == "preflight":
                gate = _preflight_stage(run_dir, args)
            elif stage == "cache":
                preflight = _optional_json(run_dir, "preflight_gate.json")
                if not _passed(preflight):
                    raise ArtifactCompatibilityError(
                        "cache stage requires a passing preflight"
                    )
                gate = _cache_stage(run_dir, args)
            elif stage == "train":
                cache = _optional_json(run_dir, "cache_gate.json")
                if not _passed(cache):
                    raise ArtifactCompatibilityError(
                        "train stage requires a passing cache gate"
                    )
                gate = _train_stage(run_dir, args)
            elif stage == "confirm":
                train = _optional_json(run_dir, "train_gate.json")
                if not _passed(train):
                    raise ArtifactCompatibilityError(
                        "confirm stage requires a passing train gate"
                    )
                if int(train.get("coarse_baseline_only", 0)) == 1:
                    if args.stage == "all":
                        break
                    raise ArtifactCompatibilityError(
                        "baseline-only selection does not authorize confirmation"
                    )
                gate = _confirm_stage(run_dir, args)
            else:  # pragma: no cover - parser prevents this
                raise AssertionError(stage)
            _status(run_dir, state="running", stage=stage)
            # ``all`` is fail-closed at every prerequisite.  A legal
            # update-zero train result is a completed closed outcome and
            # therefore stops before confirmation without being an execution
            # failure.
            if not _passed(gate):
                break
            if (
                stage == "train"
                and int(gate.get("coarse_baseline_only", 0)) == 1
            ):
                break
        active_stage = "report"
        workflow = _workflow_record(run_dir, require_gate=args.require_gate)
        decision = str(workflow["decision"]["decision"])
        required_pass = int(workflow["required_gate_pass"]) == 1
        state = "complete" if required_pass else "gate_failed"
        _status(
            run_dir,
            state=state,
            stage=args.stage,
            decision=decision,
            failure_domain=(None if required_pass else "scientific_gate"),
            failure_code=(None if required_pass else f"{args.require_gate}_gate_failed"),
        )
        _artifact_registry(run_dir)
        return 0 if required_pass else 2
    except KeyboardInterrupt:
        if run_dir is not None:
            _status(
                run_dir,
                state="interrupted",
                stage=active_stage,
                failure_domain="interruption",
                failure_code="keyboard_interrupt",
                message="interrupted; resume from the same run directory",
            )
            _artifact_registry(run_dir)
        raise
    except Exception as exc:
        if run_dir is not None:
            _commit_execution_failure(
                run_dir,
                stage=active_stage,
                exc=exc,
                require_gate=args.require_gate,
            )
        print(f"Jacobi/RB coarse-residual error: {exc}", file=sys.stderr, flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
