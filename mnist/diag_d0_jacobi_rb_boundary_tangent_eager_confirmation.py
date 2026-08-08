"""Exact eager-prefix boundary-tangent time-local confirmation (v2).

This additive workflow integrates the already certified eager-prefix CUDA
schedule into the one-image boundary-tangent experiment.  It ends after one
sealed, time-local confirmation.  It never executes a controller trajectory,
reverse path, reconstruction, or sampler.
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
from torch import Tensor, nn

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    MIDPOINT_COUNT,
    MIDPOINT_FRACTIONS,
    SELECTED_OUTER_STEPS,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_confirmation import (
    aggregate_confirmation_improvements,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import (
    EagerCohort,
    EagerDiagnosticsAccumulator,
    EagerShardExecution,
    combine_eager_metrics,
    deterministic_test_branch_runner,
    deterministic_test_shard_runner,
    eager_execution_contract,
    execute_eager_shard,
    frozen_cache_cohorts,
    frozen_eager_cache_plan,
    generate_eager_cache,
    iter_eager_shards,
    load_eager_role_inputs,
    load_eager_role_labels,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_gate import (
    CACHE_DESIGN_FLAGS,
    CACHE_EXECUTION_FLAGS,
    CONFIRM_EXECUTION_FLAGS,
    PREFLIGHT_ADJUDICATION_FLAGS,
    PREFLIGHT_DESIGN_FLAGS,
    PREFLIGHT_PROVENANCE_FLAGS,
    PREFLIGHT_REPRESENTATION_FLAGS,
    PREFLIGHT_SCHEDULE_FLAGS,
    TRAIN_BASELINE_FLAGS,
    TRAIN_OPTIMIZATION_FLAGS,
    BoundaryTangentEagerThresholds,
    REQUIRED_GATES,
    decide_workflow,
    evaluate_cache_gate,
    evaluate_confirm_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    evaluate_train_gate,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_provenance import (
    build_eager_boundary_tangent_path_plan,
    build_eager_boundary_tangent_readjudication,
    eager_boundary_tangent_source_fingerprint,
    eager_boundary_tangent_source_paths,
    validate_eager_boundary_tangent_path_plan,
    verify_eager_boundary_tangent_parents,
    verify_eager_boundary_tangent_resume_compatibility,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    BoundaryTangentThresholds,
    one_sided_whole_path_max_t,
)
from mnist.d0_jacobi_rb_boundary_tangent_prefix_fallback import (
    sample_alpha1_rb_transition_batch_cuda_eager,
)
from mnist.d0_jacobi_rb_boundary_tangent_prefix_schedule import (
    eager_prefix_contract,
    eager_prefix_profile,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule import (
    CONFIRMATION_COHORT_SIZES,
    PHASE_COUNT,
    TRAIN_VALIDATION_COHORT_SIZES,
    sample_fused_midpoint_branches,
)
from mnist.d0_jacobi_rb_cuda_multipath import run_exact_multipath_shard
from mnist.d0_jacobi_rb_cuda import sample_alpha1_rb_transition_batch_cuda
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    MODEL_INPUT_FIELDS,
    OUTER_STEPS,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
)
from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time
from mnist import diag_d0_jacobi_rb_boundary_tangent_controller_confirmation as _legacy


RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-eager-confirmation-v2"
)
STAGES = ("preflight", "cache", "train", "confirm", "report", "all")
ROOT_SEED = 261_311
MODEL_SEEDS = (261_312, 261_313, 261_314)
BOOTSTRAP_SEED = 261_315
RESERVED_CONTROL_SEED = 261_316
SYNTHETIC_CONTROL_SEED = 261_317
NULL_CONTROL_SEED = 261_318
SHARD_STEPS = 8
TRAINING = {
    "width": 32,
    "batch_size": 32,
    "maximum_updates": 4_000,
    "validation_interval": 100,
    "learning_rate": 1.0e-3,
    "weight_decay": 0.0,
    "gradient_norm_clip": 1.0,
    "mixed_precision": 0,
}
CONTROL_ANCHORS = (127, 255, 383, 511)
FORBIDDEN_SCOPE = {
    "controller_control_trajectory_performed": 0,
    "maximum_control_trajectory_phase_count": 0,
    "full_reverse_path_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "image_sampling_performed": 0,
    "reconstruction_performed": 0,
    "full_dataset_training_performed": 0,
}
NO_WORK = dict(FORBIDDEN_SCOPE)
_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "boundary_tangent_eager_decision.json",
}


class EagerBoundaryTangentCLIError(RuntimeError):
    """Typed workflow execution failure."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "boundary_tangent_eager_execution_failed",
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


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _eager_array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return config_fingerprint(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(
                array.tobytes(order="C")
            ).hexdigest(),
        }
    )


def _atomic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False
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
        "path": target.relative_to(target.parent.parent).as_posix()
        if target.parent.parent in target.parents
        else str(target),
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
    def committed_flag(name: str, field: str) -> int:
        path = run_dir / name
        if not path.is_file():
            return 0
        try:
            return int(_load_json(path).get(field, 0) == 1)
        except ArtifactCompatibilityError:
            return 0

    return {
        "production_cache_generation_performed": committed_flag(
            "cache_metrics.json", "production_cache_generation_performed"
        ),
        "physical_training_performed": int(
            (run_dir / "physical_training_started.json").is_file()
        ),
        "confirmation_performed": committed_flag(
            "confirmation_metrics.json", "confirmation_performed"
        ),
        **FORBIDDEN_SCOPE,
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
        if (
            relative in _REGISTRY_EXCLUDED
            or relative.endswith(".tmp")
            or ".tmp." in path.name
        ):
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


def _verify_existing_registry(run_dir: Path) -> None:
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
            raise ArtifactCompatibilityError("registered artifact row is malformed")
        relative = str(item.get("path", ""))
        pure = Path(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != relative
            or relative in _REGISTRY_EXCLUDED
            or relative in registered
        ):
            raise ArtifactCompatibilityError("registered artifact path is unsafe")
        registered.add(relative)
        target = run_dir / relative
        if (
            not target.is_file()
            or item.get("sha256") != file_fingerprint(target)
            or int(item.get("size", -1)) != target.stat().st_size
        ):
            raise ArtifactCompatibilityError("registered artifact changed")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _REGISTRY_EXCLUDED
        and not path.name.endswith(".tmp")
        and ".tmp." not in path.name
    }
    if registered != actual:
        raise ArtifactCompatibilityError("artifact registry does not match run files")


def _seal_stage(run_dir: Path, names: Sequence[str], seal_name: str) -> dict[str, Any]:
    artifacts = [
        {
            "path": name,
            "sha256": file_fingerprint(run_dir / name),
            "size": int((run_dir / name).stat().st_size),
        }
        for name in names
    ]
    record = {
        "schema": RUN_SCHEMA + "-stage-seal",
        "schema_version": 1,
        "artifacts": artifacts,
        **FORBIDDEN_SCOPE,
    }
    record["semantic_sha256"] = config_fingerprint(record)
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
            raise ArtifactCompatibilityError("sealed stage artifact changed")


def _source_set() -> tuple[Path, ...]:
    directory = Path(__file__).parent
    # Hash the executed scientific closure, not just the additive wrappers.
    # This intentionally includes the immutable parent-era implementation
    # modules because the v2 cache and trainer call them directly.
    names = tuple(
        directory / name
        for name in (
            Path(__file__).name,
            "d0_jacobi_artifacts.py",
            "d0_jacobi_rb_boundary_tangent.py",
            "d0_jacobi_rb_boundary_tangent_cache.py",
            "d0_jacobi_rb_boundary_tangent_confirmation.py",
            "d0_jacobi_rb_boundary_tangent_eager_cache.py",
            "d0_jacobi_rb_boundary_tangent_eager_gate.py",
            "d0_jacobi_rb_boundary_tangent_eager_provenance.py",
            "d0_jacobi_rb_boundary_tangent_gate.py",
            "d0_jacobi_rb_boundary_tangent_prefix_fallback.py",
            "d0_jacobi_rb_boundary_tangent_prefix_schedule.py",
            "d0_jacobi_rb_boundary_tangent_schedule.py",
            "d0_jacobi_rb_coarse_residual.py",
            "d0_jacobi_rb_controls.py",
            "d0_jacobi_rb_cuda.py",
            "d0_jacobi_rb_cuda_certificate.py",
            "d0_jacobi_rb_cuda_controls.py",
            "d0_jacobi_rb_cuda_fused.py",
            "d0_jacobi_rb_cuda_multipath.py",
            "d0_jacobi_rb_learnability.py",
            "d0_jacobi_rb_reverse_controller.py",
            "d0_jacobi_rb_spectral.py",
            "d0_jacobi_source_compat.py",
            "diag_d0_jacobi_rb_boundary_tangent_controller_confirmation.py",
        )
    )
    return eager_boundary_tangent_source_paths(names)


def _effective_plan(args: argparse.Namespace) -> dict[str, tuple[int, ...]]:
    plan = build_eager_boundary_tangent_path_plan()
    roles = {
        name: tuple(int(value) for value in values)
        for name, values in plan["roles"].items()
    }
    if not args.test_only:
        return roles
    count = int(args.test_path_count)
    return {name: values[: min(count, len(values))] for name, values in roles.items()}


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = BoundaryTangentEagerThresholds()
    plan = build_eager_boundary_tangent_path_plan()
    executed_steps = int(args.test_outer_steps) if args.test_only else OUTER_STEPS
    maximum_updates = (
        int(args.test_maximum_updates)
        if args.test_only
        else int(TRAINING["maximum_updates"])
    )
    record = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "grid_size": 28,
        "alpha": 1.0,
        "sample_steps": OUTER_STEPS,
        "executed_outer_steps": executed_steps,
        "tau_eff": 5.0e-5,
        "lambda_mix": 0.35,
        "label": 3,
        "root_seed": ROOT_SEED,
        "scheduler_benchmark_seed_forbidden": 261_321,
        "reserved_control_seed": RESERVED_CONTROL_SEED,
        "model_seeds": list(MODEL_SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "synthetic_teacher_seed": SYNTHETIC_CONTROL_SEED,
        "baseline_null_seed": NULL_CONTROL_SEED,
        "selected_outer_steps": [
            value for value in SELECTED_OUTER_STEPS if value < executed_steps
        ],
        "midpoint_fractions": list(MIDPOINT_FRACTIONS),
        "path_id_plan_sha256": plan["semantic_sha256"],
        "cohort_plan": frozen_eager_cache_plan(),
        "eager_execution_contract": eager_execution_contract(
            outer_steps=executed_steps,
            selected_steps=tuple(
                value for value in SELECTED_OUTER_STEPS if value < executed_steps
            ),
            **(
                {
                    "shard_runner": deterministic_test_shard_runner,
                    "branch_runner": deterministic_test_branch_runner,
                }
                if args.test_only
                else {}
            ),
        ),
        "training": {**TRAINING, "maximum_updates": maximum_updates},
        "thresholds": thresholds.to_dict(),
        "target": "unchanged exact binary64 certified Jacobi Rao-Blackwell label",
        "objective": "plain unweighted raw-target MSE",
        "prediction": "m=y(1-y)*(q_B+q_residual)",
        "quotient_target_formed": 0,
        "test_only": int(args.test_only),
        **FORBIDDEN_SCOPE,
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
    config = _scientific_config(args)
    path_plan = build_eager_boundary_tangent_path_plan()
    validate_eager_boundary_tangent_path_plan(path_plan)
    sources = _source_set()
    source_hash = eager_boundary_tangent_source_fingerprint(sources)
    parents = verify_eager_boundary_tangent_parents(
        eager_pipeline_run_dir=args.parent_eager_pipeline_run_dir,
        failed_boundary_tangent_run_dir=args.failed_boundary_tangent_run_dir,
        parent_coarse_residual_run_dir=args.parent_coarse_residual_run_dir,
    )
    readjudication = build_eager_boundary_tangent_readjudication(parents)
    if resumed:
        verify_eager_boundary_tangent_resume_compatibility(
            run_dir,
            source_fingerprint_value=source_hash,
            scientific_config_sha256=str(config["semantic_sha256"]),
            parent_provenance_sha256=str(parents["semantic_sha256"]),
            parent_readjudication_sha256=str(readjudication["semantic_sha256"]),
            path_plan_sha256=str(path_plan["semantic_sha256"]),
        )
        _verify_existing_registry(run_dir)
        return
    atomic_write_json(run_dir / "parent_provenance.json", parents)
    atomic_write_json(run_dir / "parent_readjudication.json", readjudication)
    atomic_write_json(run_dir / "scientific_config.json", config)
    atomic_write_json(run_dir / "path_id_plan.json", path_plan)
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema": RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "device": args.device,
            "source_fingerprint": source_hash,
            "source_paths": [str(path) for path in sources],
            "scientific_config_sha256": config["semantic_sha256"],
            "parent_provenance_sha256": parents["semantic_sha256"],
            "parent_readjudication_sha256": readjudication["semantic_sha256"],
            "path_plan_sha256": path_plan["semantic_sha256"],
            **_scope(run_dir),
        },
    )
    _status(run_dir, state="initialized", stage="initialize")


def _preflight_seam(args: argparse.Namespace, source: np.ndarray) -> dict[str, Any]:
    """Compare adaptive and eager execution on the fresh eight-path seam."""

    paths = tuple(range(0xEF000, 0xEF008))
    if args.test_only:
        return {
            "schema": RUN_SCHEMA + "-seam-preflight",
            "schema_version": 1,
            "path_ids": list(paths),
            "base_states_equal": 1,
            "base_targets_equal": 1,
            "base_certificates_equal": 1,
            "midpoint_states_equal": 1,
            "midpoint_targets_equal": 1,
            "midpoint_certificates_equal": 1,
            "eager_base_prefix_schedule_valid": 1,
            "eager_branch_prefix_schedule_valid": 1,
            "maximum_mass_error": 0.0,
            "certificate_fraction": 1.0,
            "forbidden_event_count": 0,
            "passed": 1,
            "test_fixture": 1,
        }
    device = torch.device(args.device)
    cohort = EagerCohort(
        kind="confirmation",
        index=99,
        path_ids=paths,
        path_roles=("preflight_seam",) * len(paths),
    )
    selected = (15,)

    def run(*, eager: bool) -> tuple[Any, Any]:
        state = torch.as_tensor(
            np.repeat(source[None, :], len(paths), axis=0).copy(order="C"),
            dtype=torch.float64,
            device=device,
        ).contiguous()
        values = []
        for start in (0, 8):
            kwargs: dict[str, Any] = {}
            if not eager:
                kwargs.update(
                    {
                        "sampler": sample_alpha1_rb_transition_batch_cuda,
                        "shard_runner": run_exact_multipath_shard,
                        "branch_runner": sample_fused_midpoint_branches,
                    }
                )
            value = execute_eager_shard(
                state,
                cohort=cohort,
                start_step=start,
                root_seed=ROOT_SEED,
                selected_steps=selected,
                profile=eager_prefix_profile(),
                **kwargs,
            )
            values.append(value)
            state = value.final_states.detach().clone().contiguous()
        return values[0], values[1]

    adaptive = run(eager=False)
    eager = run(eager=True)
    equality = {
        "base_states_equal": int(
            all(
                np.array_equal(a.committed_final_states, b.committed_final_states)
                for a, b in zip(adaptive, eager, strict=True)
            )
        ),
        "base_targets_equal": int(
            all(
                a.base_record.get("batch_output_sha256")
                == b.base_record.get("batch_output_sha256")
                for a, b in zip(adaptive, eager, strict=True)
            )
        ),
        "base_certificates_equal": int(
            all(
                a.base_record.get("batch_certificate_sha256")
                == b.base_record.get("batch_certificate_sha256")
                for a, b in zip(adaptive, eager, strict=True)
            )
        ),
    }
    adaptive_branches = adaptive[1].branches
    eager_branches = eager[1].branches
    equality.update(
        {
            "midpoint_states_equal": int(
                all(
                    torch.equal(
                        a.batch.batch.later_full_state,
                        b.batch.batch.later_full_state,
                    )
                    for a, b in zip(adaptive_branches, eager_branches, strict=True)
                )
            ),
            "midpoint_targets_equal": int(
                all(
                    torch.equal(
                        a.batch.batch.denoising_target,
                        b.batch.batch.denoising_target,
                    )
                    for a, b in zip(adaptive_branches, eager_branches, strict=True)
                )
            ),
            "midpoint_certificates_equal": int(
                all(
                    torch.equal(
                        a.batch.batch.certificate_codes,
                        b.batch.batch.certificate_codes,
                    )
                    for a, b in zip(adaptive_branches, eager_branches, strict=True)
                )
            ),
        }
    )
    diagnostics = [value.diagnostics for value in eager]
    transitions = sum(int(value["transition_count"]) for value in diagnostics)
    certified = sum(int(value["certified_count"]) for value in diagnostics)
    forbidden = sum(
        int(count)
        for value in diagnostics
        for count in value.get("forbidden_counts", {}).values()
    )
    record = {
        "schema": RUN_SCHEMA + "-seam-preflight",
        "schema_version": 1,
        "path_ids": list(paths),
        **equality,
        "eager_base_prefix_schedule_valid": int(
            all(value.get("eager_sampler_injected") == 1 for value in diagnostics)
        ),
        "eager_branch_prefix_schedule_valid": int(
            all(len(value.branches) == (PHASE_COUNT if value.selected_step is not None else 0) for value in eager)
        ),
        "transition_count": transitions,
        "certificate_fraction": certified / max(transitions, 1),
        "maximum_mass_error": max(float(value["maximum_mass_error"]) for value in diagnostics),
        "forbidden_event_count": forbidden,
    }
    record["passed"] = int(
        all(equality.values())
        and record["eager_base_prefix_schedule_valid"] == 1
        and record["eager_branch_prefix_schedule_valid"] == 1
        and record["certificate_fraction"] == 1.0
        and record["maximum_mass_error"] <= 2.0e-12
        and forbidden == 0
    )
    return record


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "preflight_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
        return _load_json(gate_path)
    provenance = _load_json(run_dir / "parent_provenance.json")
    readjudication = _load_json(run_dir / "parent_readjudication.json")
    path_plan = _load_json(run_dir / "path_id_plan.json")
    path_validation = validate_eager_boundary_tangent_path_plan(path_plan)
    source = _legacy._load_source_target(args.parent_coarse_residual_run_dir)
    representation = _legacy._representation_preflight(
        torch.device(args.device), tuple(path_plan["roles"]["train"])
    )
    seam = _preflight_seam(args, source)
    eager_metrics = _load_json(
        args.parent_eager_pipeline_run_dir / "eager_pipeline_metrics.json"
    )
    profile_contract = eager_prefix_contract()
    plan = frozen_eager_cache_plan()
    atomic_write_json(run_dir / "boundary_tangent_representation_preflight.json", representation)
    atomic_write_json(run_dir / "eager_scheduler_seam_preflight.json", seam)
    atomic_write_json(run_dir / "eager_execution_plan.json", plan)
    atomic_write_json(
        run_dir / "target_and_input_contract.json",
        {
            "schema": RUN_SCHEMA + "-target-input-contract",
            "schema_version": 1,
            "allowed_model_inputs": list(MODEL_INPUT_FIELDS),
            "audit_only_fields": ["outer_step", "midpoint_index", "midpoint_fraction"],
            "target": "unchanged raw binary64 Jacobi Rao-Blackwell label",
            "prediction": "y(1-y)*(q_B+q_residual)",
            "quotient_target_formed": 0,
            "raw_target_clipped": 0,
            **FORBIDDEN_SCOPE,
        },
    )
    t = BoundaryTangentEagerThresholds()
    flags = {
        name: 1
        for name in (
            PREFLIGHT_PROVENANCE_FLAGS
            + PREFLIGHT_ADJUDICATION_FLAGS
            + PREFLIGHT_REPRESENTATION_FLAGS
            + PREFLIGHT_SCHEDULE_FLAGS
            + PREFLIGHT_DESIGN_FLAGS
        )
    }
    flags.update(
        {
            "provenance_valid": int(provenance.get("passed", 0)),
            "legacy_boundary_tangent_adjudication_valid": int(readjudication.get("passed", 0)),
            "boundary_tangent_representation_valid": int(representation.get("passed", 0)),
            "eager_schedule_integration_valid": int(seam.get("passed", 0)),
            "path_plan_valid": int(path_validation.get("passed", 0)),
        }
    )
    metrics: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        **flags,
        "eager_parent_record_count": 615,
        "controller_v1_parent_record_count": 14,
        "root_seed": ROOT_SEED,
        "model_seeds": list(MODEL_SEEDS),
        "reserved_control_seed": RESERVED_CONTROL_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "synthetic_teacher_seed": SYNTHETIC_CONTROL_SEED,
        "baseline_null_seed": NULL_CONTROL_SEED,
        "training_path_ids": list(t.training_path_ids),
        "validation_path_ids": list(t.validation_path_ids),
        "confirmation_path_ids": list(t.confirmation_path_ids),
        "preflight_seam_path_ids": list(t.preflight_seam_path_ids),
        "forbidden_historical_v1_path_ids": list(t.forbidden_historical_v1_path_ids),
        "train_validation_cohort_sizes": list(TRAIN_VALIDATION_COHORT_SIZES),
        "confirmation_cohort_sizes": list(CONFIRMATION_COHORT_SIZES),
        "training_paths": t.training_paths,
        "validation_paths": t.validation_paths,
        "confirmation_paths": t.confirmation_paths,
        "projected_total_transitions": int(eager_metrics["projected_total_transitions"]),
        "projected_base_transitions": int(eager_metrics["projected_base_transitions"]),
        "projected_midpoint_transitions": int(eager_metrics["projected_midpoint_transitions"]),
        "candidate_modes": int(eager_metrics["candidate_modes"]),
        "certificate_fraction": float(eager_metrics["certificate_fraction"]),
        "forbidden_event_count": int(eager_metrics["forbidden_event_count"]),
        "projected_elapsed_seconds": float(eager_metrics["projected_elapsed_seconds"]),
        "projected_effective_rate": float(eager_metrics["projected_effective_transitions_per_second"]),
        "minimum_profile_rate": float(eager_metrics["minimum_individual_profile_rate"]),
        "fallback_fraction": float(eager_metrics["fallback_fraction"]),
        "fallback_time_fraction": float(eager_metrics["fallback_time_fraction"]),
        "maximum_mass_error": max(float(eager_metrics["maximum_mass_error"]), float(seam["maximum_mass_error"])),
        "peak_memory_fraction": float(eager_metrics["peak_memory_fraction"]),
        "projected_persisted_bytes": int(eager_metrics["projected_persisted_bytes"]),
        "maximum_launch_lanes": int(eager_metrics["maximum_launch_lanes"]),
        "production_cache_generation_performed": 0,
        "physical_training_performed": 0,
        "confirmation_performed": 0,
        "profile_contract": profile_contract,
        **FORBIDDEN_SCOPE,
    }
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    gate = evaluate_preflight_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "boundary_tangent_representation_preflight.json",
            "eager_scheduler_seam_preflight.json",
            "eager_execution_plan.json",
            "target_and_input_contract.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )
    return gate


def _frozen_confirmation_projection(args: argparse.Namespace) -> float:
    if args.test_only:
        return 1.0
    metrics = _load_json(
        args.parent_eager_pipeline_run_dir / "eager_pipeline_metrics.json"
    )
    seconds = metrics.get("slowest_profile_seconds")
    if not isinstance(seconds, Mapping):
        raise ArtifactCompatibilityError("eager parent timing profile is missing")
    return 8.0 * (
        6.0 * float(seconds["stream_p10"]) + float(seconds["stream_p4"])
    )


def _training_index_bindings(run_dir: Path) -> None:
    directory = run_dir / "cache"
    directory.mkdir(parents=True, exist_ok=True)
    for role in ("train", "validation"):
        source = run_dir / "eager_cache" / f"{role}_index.json"
        record = {
            "schema": RUN_SCHEMA + "-training-cache-binding",
            "schema_version": 1,
            "role": role,
            "source_path": source.relative_to(run_dir).as_posix(),
            "source_sha256": file_fingerprint(source),
        }
        record["semantic_sha256"] = config_fingerprint(record)
        atomic_write_json(directory / f"{role}_index.json", record)


def _cache_runtime_summary(run_dir: Path) -> tuple[float, float]:
    """Return complete cache time and the slowest physical cohort rate."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((run_dir / "eager_cache" / "train_validation").rglob("metadata.json")):
        record = _load_json(path)
        identity = record.get("identity")
        diagnostics = record.get("diagnostics")
        if not isinstance(identity, Mapping) or not isinstance(diagnostics, Mapping):
            raise ArtifactCompatibilityError("cache shard timing record is malformed")
        grouped.setdefault(int(identity["cohort_index"]), []).append(record)
    if not grouped:
        raise ArtifactCompatibilityError("cache contains no committed shard timing")
    total_elapsed = 0.0
    rates: list[float] = []
    for records in grouped.values():
        elapsed = sum(
            float(record["diagnostics"]["complete_pipeline_elapsed_seconds"])
            + float(record.get("persistence_elapsed_seconds", 0.0))
            for record in records
        )
        transitions = sum(
            int(record["diagnostics"]["transition_count"]) for record in records
        )
        if elapsed <= 0.0 or transitions <= 0:
            raise ArtifactCompatibilityError("cache cohort timing is nonpositive")
        total_elapsed += elapsed
        rates.append(transitions / elapsed)
    return total_elapsed, min(rates)


def _cache_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("cache requires a passing preflight")
    gate_path = run_dir / "cache_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "cache_artifact_seal.json")
        load_eager_role_inputs(run_dir, "train")
        load_eager_role_inputs(run_dir, "validation")
        return _load_json(gate_path)
    source = _legacy._load_source_target(args.parent_coarse_residual_run_dir)
    kwargs: dict[str, Any] = {}
    if args.test_only:
        kwargs.update(
            {
                "outer_steps": int(args.test_outer_steps),
                "selected_steps": tuple(
                    value for value in SELECTED_OUTER_STEPS if value < args.test_outer_steps
                ),
                "cohort_indices": (0, 7),
                "shard_runner": deterministic_test_shard_runner,
                "branch_runner": deterministic_test_branch_runner,
            }
        )

    def progress(identity: Any, disposition: str) -> None:
        print(
            f"eager cache cohort={identity.cohort_index} "
            f"step={identity.start_step} {disposition}",
            flush=True,
        )

    result = generate_eager_cache(
        run_dir,
        source,
        device=args.device,
        root_seed=ROOT_SEED,
        progress=progress,
        **kwargs,
    )
    _training_index_bindings(run_dir)
    aggregate = dict(result["metrics"])
    train_inputs, train_index = load_eager_role_inputs(run_dir, "train")
    validation_inputs, validation_index = load_eager_role_inputs(run_dir, "validation")
    total = int(aggregate["transition_count"])
    fallback = int(aggregate["fallback_count"])
    elapsed, minimum_role_rate = _cache_runtime_summary(run_dir)
    forbidden = sum(int(value) for value in aggregate["forbidden_counts"].values())
    t = BoundaryTangentEagerThresholds()
    production_counts = {
        "train_row_count": t.train_rows,
        "validation_row_count": t.validation_rows,
        "train_transition_count": t.train_transitions,
        "validation_transition_count": t.validation_transitions,
        "cache_transition_count": t.train_transitions + t.validation_transitions,
    }
    if args.test_only:
        actual_counts = {
            "test_actual_train_row_count": len(train_inputs["sample_key"]),
            "test_actual_validation_row_count": len(validation_inputs["sample_key"]),
            "test_actual_cache_transition_count": total,
        }
    else:
        production_counts = {
            "train_row_count": len(train_inputs["sample_key"]),
            "validation_row_count": len(validation_inputs["sample_key"]),
            "train_transition_count": int(train_index["transition_count"]),
            "validation_transition_count": int(validation_index["transition_count"]),
            "cache_transition_count": total,
        }
        actual_counts = {}
    projected_confirmation = _frozen_confirmation_projection(args)
    metrics = {
        "schema": RUN_SCHEMA + "-cache-metrics",
        "schema_version": 1,
        **{name: 1 for name in CACHE_EXECUTION_FLAGS + CACHE_DESIGN_FLAGS},
        **production_counts,
        **actual_counts,
        "certificate_fraction": int(aggregate["certified_count"]) / max(total, 1),
        "maximum_mass_error": float(aggregate["maximum_mass_error"]),
        "forbidden_event_count": forbidden,
        "minimum_role_rate": minimum_role_rate,
        "fallback_fraction": fallback / max(total, 1),
        "fallback_time_fraction": float(aggregate["fallback_elapsed_seconds"]) / max(elapsed, np.finfo(float).tiny),
        "peak_memory_fraction": float(aggregate["maximum_peak_memory_fraction"]),
        "total_persisted_cache_bytes": int(aggregate["persisted_bytes"]),
        "maximum_launch_lanes": int(aggregate["maximum_launch_lanes"]),
        "cache_elapsed_seconds": elapsed,
        "frozen_conservative_confirmation_projection_seconds": projected_confirmation,
        "production_cache_generation_performed": 1,
        "physical_training_performed": 0,
        "confirmation_performed": 0,
        "confirmation_absent": int(not (run_dir / "confirmation_seal.json").exists()),
        "test_only": int(args.test_only),
        **FORBIDDEN_SCOPE,
    }
    atomic_write_json(run_dir / "cache_metrics.json", metrics)
    gate = evaluate_cache_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
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


def _load_training_inputs(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Open only permitted model inputs; raw physical labels stay closed."""

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


def _run_prelabel_controls(
    run_dir: Path,
    args: argparse.Namespace,
    train_inputs: ModelInputs,
    validation_inputs: ModelInputs,
    train_path_rows: np.ndarray,
    validation_path_rows: np.ndarray,
    paths: Mapping[str, Sequence[int]],
    maximum_updates: int,
) -> dict[str, Any]:
    """Run synthetic and analytic-null controls before any label loader call."""

    from mnist.d0_jacobi_rb_boundary_tangent import (
        BoundaryTangentPredictor,
        synthetic_tangent_target,
    )

    zero = _legacy._zero_baseline(paths["train"])
    synthetic_train = synthetic_tangent_target(train_inputs).detach().to(torch.float64)
    synthetic_validation = synthetic_tangent_target(validation_inputs).detach().to(
        torch.float64
    )
    synthetic_scale = float(torch.sqrt(torch.mean(synthetic_train.square())).cpu())
    if not math.isfinite(synthetic_scale) or synthetic_scale <= 0.0:
        raise EagerBoundaryTangentCLIError(
            "synthetic target scale is invalid",
            failure_domain="optimization_control",
            failure_code="boundary_tangent_synthetic_scale_invalid",
        )
    synthetic_report = _legacy._training_task(
        run_dir,
        task="synthetic-teacher",
        seed=SYNTHETIC_CONTROL_SEED,
        baseline=zero,
        train_inputs=train_inputs,
        train_target=synthetic_train,
        validation_inputs=validation_inputs,
        validation_target=synthetic_validation,
        validation_path_ids=validation_path_rows,
        target_scale=synthetic_scale,
        maximum_updates=maximum_updates,
        physical=False,
    )
    synthetic_metrics = _legacy._selected_control_metrics(
        run_dir,
        synthetic_report,
        zero,
        validation_inputs,
        synthetic_validation,
        validation_path_rows,
        synthetic=True,
    )
    atomic_write_json(run_dir / "synthetic_teacher_control.json", synthetic_metrics)
    atomic_write_csv(
        run_dir / "synthetic_teacher_per_path.csv",
        synthetic_metrics["path_metrics"],
    )

    null_baseline = _legacy._analytic_null_baseline(paths["train"])
    null_model = BoundaryTangentPredictor(null_baseline, zero_residual=True).to(
        train_inputs.later_full_state.device
    )
    with torch.no_grad():
        null_train = null_model.baseline_prediction(train_inputs).detach()
        null_validation = null_model.baseline_prediction(validation_inputs).detach()
    null_scale = float(torch.sqrt(torch.mean(null_train.square())).cpu())
    null_report = _legacy._training_task(
        run_dir,
        task="exact-baseline-null",
        seed=NULL_CONTROL_SEED,
        baseline=null_baseline,
        train_inputs=train_inputs,
        train_target=null_train,
        validation_inputs=validation_inputs,
        validation_target=null_validation,
        validation_path_ids=validation_path_rows,
        target_scale=null_scale,
        maximum_updates=maximum_updates,
        physical=False,
    )
    null_metrics = {
        "schema": RUN_SCHEMA + "-exact-baseline-null",
        "schema_version": 1,
        "selected_update": int(null_report["selected"]["update"]),
        "selected_validation_mse": float(
            null_report["selected"]["validation_mse"]
        ),
        "passed": int(int(null_report["selected"]["update"]) == 0),
        **FORBIDDEN_SCOPE,
    }
    atomic_write_json(run_dir / "exact_baseline_null_control.json", null_metrics)
    return {
        "synthetic_report": synthetic_report,
        "synthetic_metrics": synthetic_metrics,
        "null_report": null_report,
        "null_metrics": null_metrics,
        "passed": int(
            int(synthetic_metrics.get("passed", 0)) == 1
            and int(null_metrics.get("passed", 0)) == 1
        ),
        "physical_labels_opened": 0,
    }


def _load_physical_training_labels(
    run_dir: Path,
    input_data: Mapping[str, Any],
) -> dict[str, Any]:
    train_labels, train_index = load_eager_role_labels(run_dir, "train")
    validation_labels, validation_index = load_eager_role_labels(
        run_dir, "validation"
    )
    for role, arrays, labels, input_index, label_index in (
        (
            "train",
            input_data["train_arrays"],
            train_labels,
            input_data["train_index"],
            train_index,
        ),
        (
            "validation",
            input_data["validation_arrays"],
            validation_labels,
            input_data["validation_index"],
            validation_index,
        ),
    ):
        if (
            input_index.get("semantic_sha256") != label_index.get("semantic_sha256")
            or not np.array_equal(arrays["sample_key"], labels["sample_key"])
            or not np.array_equal(arrays["path_id"], labels["path_id"])
        ):
            raise ArtifactCompatibilityError(f"{role} physical label join changed")
    return {
        "train": train_labels,
        "validation": validation_labels,
    }


def _run_physical_training(
    run_dir: Path,
    args: argparse.Namespace,
    input_data: Mapping[str, Any],
    labels: Mapping[str, Any],
    *,
    maximum_updates: int,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent import (
        derive_tangent_baseline,
        save_tangent_baseline,
    )

    device = torch.device(args.device)
    train_target = torch.as_tensor(
        np.array(labels["train"]["denoising_target"], copy=True, order="C"),
        dtype=torch.float64,
        device=device,
    )
    validation_target = torch.as_tensor(
        np.array(
            labels["validation"]["denoising_target"], copy=True, order="C"
        ),
        dtype=torch.float64,
        device=device,
    )
    target_scale = float(torch.sqrt(torch.mean(train_target.square())).cpu())
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise EagerBoundaryTangentCLIError(
            "training-only raw target scale is invalid",
            failure_domain="baseline",
            failure_code="boundary_tangent_target_scale_invalid",
        )
    baseline = derive_tangent_baseline(
        input_data["train_inputs"],
        train_target,
        input_data["train_path_rows"],
    )
    if not (
        np.isfinite(baseline.q_values).all()
        and np.isfinite(baseline.denominators).all()
        and np.all(baseline.denominators > 0.0)
    ):
        raise EagerBoundaryTangentCLIError(
            "training-only tangent baseline is invalid",
            failure_domain="baseline",
            failure_code="boundary_tangent_baseline_invalid",
        )
    baseline_artifact = save_tangent_baseline(
        run_dir / "tangent_baseline.npz", baseline
    )
    atomic_write_json(
        run_dir / "tangent_baseline.json",
        {
            "schema": RUN_SCHEMA + "-baseline",
            "schema_version": 1,
            "baseline": baseline.to_record(),
            "file": baseline_artifact,
            "target_scale": target_scale,
            "training_path_ids": sorted(
                np.unique(input_data["train_path_rows"]).tolist()
            ),
            "validation_path_ids_used": 0,
            "confirmation_path_ids_used": 0,
            "quotient_target_formed": 0,
            **FORBIDDEN_SCOPE,
        },
    )
    reports = [
        _legacy._training_task(
            run_dir,
            task="physical",
            seed=seed,
            baseline=baseline,
            train_inputs=input_data["train_inputs"],
            train_target=train_target,
            validation_inputs=input_data["validation_inputs"],
            validation_target=validation_target,
            validation_path_ids=input_data["validation_path_rows"],
            target_scale=target_scale,
            maximum_updates=maximum_updates,
            physical=True,
        )
        for seed in MODEL_SEEDS
    ]
    nonzero = [
        report
        for report in reports
        if int(report.get("complete", 0)) == 1
        and int(report.get("finite", 0)) == 1
        and int(report["selected"]["update"]) > 0
        and int(report["selected"].get("eligible_nonzero", 0)) == 1
    ]
    selected_report = min(
        nonzero if nonzero else reports,
        key=lambda report: (
            float(report["selected"]["validation_mse"]),
            int(report["selected"]["update"]),
            int(report["seed"]),
        ),
    )
    selected = dict(selected_report["selected"])
    # The reusable trainer/checkpoint payload remains scientifically identical
    # to v1.  The v2 selection record binds it to this eager execution run.
    selection = {
        "schema": _legacy.RUN_SCHEMA + "-checkpoint-selection",
        "schema_version": 1,
        "workflow_schema": RUN_SCHEMA,
        "selection_role": "validation_only",
        "task": "physical",
        "selected_seed": int(selected_report["seed"]),
        "selected_update": int(selected["update"]),
        "selected_state_sha256": str(selected["state_sha256"]),
        "training_fingerprint": str(selected["training_fingerprint"]),
        "checkpoint_path": str(selected["checkpoint_path"]),
        "checkpoint_file_sha256": str(selected["checkpoint_file_sha256"]),
        "baseline_path": "tangent_baseline.npz",
        "baseline_file_sha256": str(baseline_artifact["sha256"]),
        "baseline_semantic_sha256": baseline.fingerprint,
        "target_scale": target_scale,
        "confirmation_paths_created": 0,
        "candidate_ranking": "lowest_validation_mse_then_earliest_update_then_lower_seed",
        **FORBIDDEN_SCOPE,
    }
    # Legacy verifier requires an exact key set only for selected semantics,
    # and accepts additive workflow binding fields through the semantic hash.
    selection["semantic_sha256"] = config_fingerprint(selection)
    atomic_write_json(run_dir / "checkpoint_selection.json", selection)
    atomic_write_csv(
        run_dir / "physical_seed_metrics.csv",
        [
            {
                "seed": int(report["seed"]),
                "complete": int(report["complete"]),
                "finite": int(report["finite"]),
                **{
                    f"selected_{key}": value
                    for key, value in report["selected"].items()
                    if isinstance(value, (str, int, float))
                },
            }
            for report in reports
        ],
    )
    return {
        "baseline": baseline,
        "baseline_artifact": baseline_artifact,
        "target_scale": target_scale,
        "reports": reports,
        "nonzero_reports": nonzero,
        "selection": selection,
    }


def _verify_training_selection(run_dir: Path) -> dict[str, Any]:
    selection = _load_json(run_dir / "checkpoint_selection.json")
    body = dict(selection)
    semantic = body.pop("semantic_sha256", None)
    if (
        semantic != config_fingerprint(body)
        or selection.get("workflow_schema") != RUN_SCHEMA
        or int(selection.get("selected_seed", -1)) not in MODEL_SEEDS
        or int(selection.get("selected_update", -1)) < 0
        or selection.get("selection_role") != "validation_only"
    ):
        raise ArtifactCompatibilityError("sealed checkpoint selection changed")
    for field in ("checkpoint_path", "baseline_path"):
        path = run_dir / str(selection[field])
        hash_field = (
            "checkpoint_file_sha256"
            if field == "checkpoint_path"
            else "baseline_file_sha256"
        )
        if selection.get(hash_field) != file_fingerprint(path):
            raise ArtifactCompatibilityError("sealed training artifact changed")
    return selection


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "cache_gate.json")):
        raise ArtifactCompatibilityError("train requires a passing cache gate")
    gate_path = run_dir / "train_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "train_artifact_seal.json")
        if (run_dir / "checkpoint_selection.json").is_file():
            _verify_training_selection(run_dir)
        return _load_json(gate_path)
    maximum_updates = (
        int(args.test_maximum_updates)
        if args.test_only
        else int(TRAINING["maximum_updates"])
    )
    input_data = _load_training_inputs(run_dir, args)
    paths = _effective_plan(args)
    controls = _run_prelabel_controls(
        run_dir,
        args,
        input_data["train_inputs"],
        input_data["validation_inputs"],
        input_data["train_path_rows"],
        input_data["validation_path_rows"],
        paths,
        maximum_updates,
    )
    if int(controls["passed"]) != 1:
        metrics = {
            "schema": RUN_SCHEMA + "-train-metrics",
            "schema_version": 1,
            **{name: 0 for name in TRAIN_OPTIMIZATION_FLAGS + TRAIN_BASELINE_FLAGS},
            "training_complete": 0,
            "synthetic_teacher_passed": int(
                controls["synthetic_metrics"].get("passed", 0)
            ),
            "synthetic_every_validation_path_beats_zero": int(
                controls["synthetic_metrics"].get(
                    "every_validation_path_beats_zero", 0
                )
            ),
            "exact_baseline_null_passed": int(
                controls["null_metrics"].get("passed", 0)
            ),
            "physical_labels_opened_after_controls": 0,
            "synthetic_relative_validation_mse": float(
                controls["synthetic_metrics"].get(
                    "relative_validation_mse", math.inf
                )
            ),
            "null_selected_update": int(
                controls["null_metrics"].get("selected_update", -1)
            ),
            "model_seed_count": len(MODEL_SEEDS),
            "maximum_updates": int(TRAINING["maximum_updates"]),
            "quotient_target_formed": 0,
            "selected_nonzero": 0,
            "production_cache_generation_performed": 1,
            "physical_training_performed": 0,
            "confirmation_performed": 0,
            **FORBIDDEN_SCOPE,
        }
        atomic_write_json(run_dir / "train_metrics.json", metrics)
        gate = evaluate_train_gate(metrics)
        atomic_write_json(gate_path, gate)
        _seal_stage(
            run_dir,
            (
                "synthetic_teacher_control.json",
                "synthetic_teacher_per_path.csv",
                "exact_baseline_null_control.json",
                "train_metrics.json",
                "train_gate.json",
            ),
            "train_artifact_seal.json",
        )
        return gate

    atomic_write_json(
        run_dir / "physical_label_open.json",
        {
            "schema": RUN_SCHEMA + "-physical-label-open",
            "schema_version": 1,
            "opened_at": _now(),
            "synthetic_control_sha256": file_fingerprint(
                run_dir / "synthetic_teacher_control.json"
            ),
            "null_control_sha256": file_fingerprint(
                run_dir / "exact_baseline_null_control.json"
            ),
            "controls_passed": 1,
            **FORBIDDEN_SCOPE,
        },
    )
    labels = _load_physical_training_labels(run_dir, input_data)
    atomic_write_json(
        run_dir / "physical_training_started.json",
        {"started_at": _now(), "physical_training_performed": 1, **FORBIDDEN_SCOPE},
    )
    physical = _run_physical_training(
        run_dir,
        args,
        input_data,
        labels,
        maximum_updates=maximum_updates,
    )
    selection = physical["selection"]
    reports = physical["reports"]
    selected = next(
        report["selected"]
        for report in reports
        if int(report["seed"]) == int(selection["selected_seed"])
    )
    t = BoundaryTangentEagerThresholds()
    metrics = {
        "schema": RUN_SCHEMA + "-train-metrics",
        "schema_version": 1,
        **{name: 1 for name in TRAIN_OPTIMIZATION_FLAGS + TRAIN_BASELINE_FLAGS},
        "synthetic_relative_validation_mse": float(
            controls["synthetic_metrics"]["relative_validation_mse"]
        ),
        "null_selected_update": int(controls["null_metrics"]["selected_update"]),
        "model_seed_count": len(MODEL_SEEDS),
        "maximum_updates": t.maximum_updates,
        "test_actual_maximum_updates": maximum_updates,
        "quotient_target_formed": 0,
        "selected_nonzero": int(int(selection["selected_update"]) > 0),
        "selected_checkpoint_eligible": int(
            int(selection["selected_update"]) > 0
            and int(selected.get("eligible_nonzero", 0)) == 1
        ),
        "selected_beats_baseline_overall": int(
            float(selected.get("combined_vs_baseline", -math.inf)) > 0.0
        ),
        "selected_beats_baseline_high_reverse_time": int(
            float(
                selected.get("combined_vs_baseline_high_reverse_time", -math.inf)
            )
            > 0.0
        ),
        "all_physical_tasks_complete_finite": int(
            all(
                int(report.get("complete", 0)) == 1
                and int(report.get("finite", 0)) == 1
                for report in reports
            )
        ),
        "confirmation_absent": int(not (run_dir / "confirmation_seal.json").exists()),
        "production_cache_generation_performed": 1,
        "physical_training_performed": 1,
        "confirmation_performed": 0,
        **FORBIDDEN_SCOPE,
    }
    atomic_write_json(run_dir / "train_metrics.json", metrics)
    gate = evaluate_train_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "synthetic_teacher_control.json",
            "synthetic_teacher_per_path.csv",
            "exact_baseline_null_control.json",
            "physical_label_open.json",
            "physical_training_started.json",
            "tangent_baseline.npz",
            "tangent_baseline.json",
            "checkpoint_selection.json",
            "physical_seed_metrics.csv",
            "train_metrics.json",
            "train_gate.json",
        ),
        "train_artifact_seal.json",
    )
    return gate


def _confirmation_paths(
    run_dir: Path, *, cohort_index: int, start_step: int
) -> tuple[Path, Path, Path, Path]:
    directory = (
        run_dir
        / "confirmation"
        / "shards"
        / f"cohort-{cohort_index:03d}"
        / f"shard-{start_step:06d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return (
        directory / "continuation_state.npz",
        directory / "path_risks.npz",
        directory / "control_anchor_states.npz",
        directory / "metadata.json",
    )


def _confirmation_branch_evidence(
    execution: EagerShardExecution,
    *,
    model: nn.Module,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, dict[str, Any]]:
    if execution.selected_step is None or len(execution.branches) != PHASE_COUNT:
        raise EagerBoundaryTangentCLIError(
            "selected eager shard has no complete midpoint evidence",
            failure_domain="confirmation_execution",
            failure_code="boundary_tangent_midpoint_evidence_missing",
        )
    selected = int(execution.selected_step)
    paths = np.asarray(execution.path_ids, dtype=np.int64)
    path_count = len(paths)
    later_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    for phase, branch in enumerate(execution.branches):
        if branch.phase != phase:
            raise EagerBoundaryTangentCLIError(
                "midpoint branch phase order changed",
                failure_domain="confirmation_execution",
                failure_code="boundary_tangent_branch_order_invalid",
            )
        batch = branch.batch.batch
        later_blocks.append(
            np.asarray(batch.later_full_state.detach().cpu().numpy())
        )
        target_blocks.append(
            np.asarray(batch.denoising_target.detach().cpu().numpy())
        )
    # Source layout is [phase, midpoint, path, ...].
    later = np.stack(later_blocks).transpose(2, 0, 1, 3)
    target = np.stack(target_blocks).transpose(2, 0, 1, 3)
    states = np.ascontiguousarray(later.reshape(-1, STATE_SIZE), dtype=np.float32)
    targets = np.ascontiguousarray(
        target.reshape(-1, EDGES_PER_PHASE), dtype=np.float64
    )
    row_paths = np.repeat(paths, PHASE_COUNT * MIDPOINT_COUNT)
    row_phases = np.tile(
        np.repeat(np.arange(PHASE_COUNT, dtype=np.int8), MIDPOINT_COUNT),
        path_count,
    )
    row_midpoints = np.tile(
        np.arange(MIDPOINT_COUNT, dtype=np.int8), path_count * PHASE_COUNT
    )
    fractions = np.tile(
        np.asarray(MIDPOINT_FRACTIONS, dtype=np.float64),
        path_count * PHASE_COUNT,
    )
    reverse_time = np.asarray(
        [
            internal_reverse_time(selected, int(phase), float(fraction))
            for phase, fraction in zip(row_phases, fractions, strict=True)
        ],
        dtype=np.float64,
    )
    arrays = {
        "later_full_state": states,
        "reverse_time": reverse_time,
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
    combined = _legacy._predict_in_batches(model, inputs).cpu().numpy()
    with torch.no_grad():
        baseline = model.baseline_prediction(inputs).cpu().numpy()
    combined_vs_zero = np.mean(
        targets**2 - (targets - combined) ** 2, axis=1, dtype=np.float64
    )
    combined_vs_baseline = np.mean(
        (targets - baseline) ** 2 - (targets - combined) ** 2,
        axis=1,
        dtype=np.float64,
    )
    baseline_vs_zero = np.mean(
        targets**2 - (targets - baseline) ** 2, axis=1, dtype=np.float64
    )
    identity_error = float(
        np.max(
            np.abs(
                combined_vs_zero - (baseline_vs_zero + combined_vs_baseline)
            ),
            initial=0.0,
        )
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
    if (
        not np.isfinite(combined_vs_zero).all()
        or not np.isfinite(combined_vs_baseline).all()
        or not np.isfinite(baseline_vs_zero).all()
        or np.unique(sample_keys).size != sample_keys.size
        or identity_error > 5.0e-15
    ):
        raise EagerBoundaryTangentCLIError(
            "streamed paired-risk evidence is invalid",
            failure_domain="paired_risk",
            failure_code="boundary_tangent_paired_risk_invalid",
        )
    risks = {
        "sample_keys": sample_keys,
        "path_ids": row_paths,
        "outer_steps": np.full(row_paths.size, selected, dtype=np.int16),
        "phases": row_phases,
        "midpoint_indices": row_midpoints,
        "combined_vs_zero": np.ascontiguousarray(combined_vs_zero),
        "combined_vs_baseline": np.ascontiguousarray(combined_vs_baseline),
        "baseline_vs_zero": np.ascontiguousarray(baseline_vs_zero),
    }
    audit = None
    if selected in CONTROL_ANCHORS:
        pre = np.stack(
            [
                np.asarray(branch.pre_phase_states.detach().cpu().numpy())
                for branch in execution.branches
            ]
        ).astype(np.float64, copy=False)
        post = np.concatenate(
            (pre[1:], execution.committed_final_states[None, :, :]), axis=0
        )
        audit = {
            "path_ids": paths,
            "outer_step": np.asarray([selected], dtype=np.int16),
            "one_phase_earlier_states": np.ascontiguousarray(pre),
            "one_phase_later_states": np.ascontiguousarray(post),
            "full_sweep_earlier_states": np.ascontiguousarray(pre[0]),
            "full_sweep_later_states": np.ascontiguousarray(post[-1]),
        }
    return risks, audit, {
        "direct_derived_total_contrast_maximum_error": identity_error,
        "row_count": int(row_paths.size),
    }


def _load_valid_confirmation_shard(
    run_dir: Path,
    *,
    cohort: Any,
    start_step: int,
    current: np.ndarray,
    selected_step: int | None,
    selection_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    state_path, risk_path, audit_path, metadata_path = _confirmation_paths(
        run_dir, cohort_index=cohort.index, start_step=start_step
    )
    if not state_path.is_file() or not metadata_path.is_file():
        return None
    try:
        record = _load_json(metadata_path)
        body = dict(record)
        semantic = body.pop("semantic_sha256", None)
        expected = {
            "schema": RUN_SCHEMA + "-confirmation-shard",
            "schema_version": 1,
            "cohort_index": cohort.index,
            "path_ids": list(cohort.path_ids),
            "start_step": start_step,
            "selected_step": selected_step,
            "input_state_sha256": _eager_array_sha(current),
            "selection_sha256": selection_sha256,
        }
        if semantic != config_fingerprint(body) or any(
            record.get(name) != value for name, value in expected.items()
        ):
            return None
        if (
            record.get("committed") != 1
            or record.get("persistence_in_timed_region") != 1
            or not math.isfinite(
                float(record.get("complete_pipeline_elapsed_seconds", math.nan))
            )
            or float(record["complete_pipeline_elapsed_seconds"]) <= 0.0
        ):
            return None
        if record.get("state_file_sha256") != file_fingerprint(state_path):
            return None
        final = _load_npz(state_path).get("final_states")
        if (
            final is None
            or final.dtype != np.float64
            or final.shape != (len(cohort.path_ids), STATE_SIZE)
            or record.get("final_state_sha256") != _array_sha(final)
        ):
            return None
        if selected_step is None:
            if record.get("risk_file_sha256") is not None:
                return None
        else:
            if record.get("risk_file_sha256") != file_fingerprint(risk_path):
                return None
            risks = _load_npz(risk_path)
            expected_rows = len(cohort.path_ids) * PHASE_COUNT * MIDPOINT_COUNT
            if (
                len(risks.get("sample_keys", ())) != expected_rows
                or "denoising_target" in risks
                or not np.isfinite(risks["combined_vs_zero"]).all()
                or not np.isfinite(risks["combined_vs_baseline"]).all()
            ):
                return None
            if selected_step in CONTROL_ANCHORS:
                if record.get("control_anchor_file_sha256") != file_fingerprint(
                    audit_path
                ):
                    return None
                _load_npz(audit_path)
        return np.ascontiguousarray(final), record
    except (ArtifactCompatibilityError, OSError, ValueError, TypeError, KeyError):
        return None


def _persist_confirmation_shard(
    run_dir: Path,
    *,
    execution: EagerShardExecution,
    selection_sha256: str,
    risks: Mapping[str, np.ndarray] | None,
    audit: Mapping[str, np.ndarray] | None,
    evidence: Mapping[str, Any],
    pipeline_started_at: float,
) -> dict[str, Any]:
    state_path, risk_path, audit_path, metadata_path = _confirmation_paths(
        run_dir,
        cohort_index=execution.identity.cohort_index,
        start_step=execution.identity.start_step,
    )
    state_artifact = _atomic_npz(
        state_path,
        {"final_states": execution.committed_final_states},
    )
    risk_artifact = _atomic_npz(risk_path, risks) if risks is not None else None
    audit_artifact = _atomic_npz(audit_path, audit) if audit is not None else None
    if (risk_artifact is None) != (execution.selected_step is None):
        raise AssertionError("risk evidence exists exactly at selected steps")
    if (audit_artifact is None) != (execution.selected_step not in CONTROL_ANCHORS):
        raise AssertionError("anchor evidence exists exactly at control anchors")
    record = {
        "schema": RUN_SCHEMA + "-confirmation-shard",
        "schema_version": 1,
        "cohort_index": execution.identity.cohort_index,
        "path_ids": list(execution.path_ids),
        "start_step": execution.identity.start_step,
        "selected_step": execution.selected_step,
        "input_state_sha256": execution.input_state_sha256,
        "selection_sha256": selection_sha256,
        "state_file_sha256": state_artifact["sha256"],
        "state_file_size": state_artifact["size"],
        "final_state_sha256": _array_sha(execution.committed_final_states),
        "risk_file_sha256": None if risk_artifact is None else risk_artifact["sha256"],
        "risk_file_size": None if risk_artifact is None else risk_artifact["size"],
        "control_anchor_file_sha256": None
        if audit_artifact is None
        else audit_artifact["sha256"],
        "control_anchor_file_size": None
        if audit_artifact is None
        else audit_artifact["size"],
        "execution": execution.to_record(),
        "evidence": dict(evidence),
        # A provisional atomic metadata commit is part of the measured
        # pipeline.  The second write below is only its commit marker.
        "complete_pipeline_elapsed_seconds": 0.0,
        "persistence_in_timed_region": 1,
        "raw_confirmation_labels_persisted": 0,
        "raw_confirmation_inputs_persisted": 0,
        "committed": 0,
        **FORBIDDEN_SCOPE,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(metadata_path, record)
    record["complete_pipeline_elapsed_seconds"] = float(
        time.perf_counter() - pipeline_started_at
    )
    record["committed"] = 1
    record["semantic_sha256"] = config_fingerprint(
        {key: value for key, value in record.items() if key != "semantic_sha256"}
    )
    atomic_write_json(metadata_path, record)
    return record


def _confirm_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent import load_tangent_baseline

    if not _passed(_load_json(run_dir / "train_gate.json")):
        raise ArtifactCompatibilityError("confirm requires a passing train gate")
    gate_path = run_dir / "confirm_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "confirm_artifact_seal.json")
        _verify_training_selection(run_dir)
        return _load_json(gate_path)
    selection = _verify_training_selection(run_dir)
    if int(selection["selected_update"]) <= 0:
        raise ArtifactCompatibilityError("confirmation cannot open for update zero")
    selection_sha = file_fingerprint(run_dir / "checkpoint_selection.json")
    outer_steps = int(args.test_outer_steps) if args.test_only else OUTER_STEPS
    selected_steps = tuple(
        value for value in SELECTED_OUTER_STEPS if value < outer_steps
    )
    paths = (
        tuple(range(0xED000, 0xED040))
        if not args.test_only
        else tuple(frozen_cache_cohorts("confirmation")[0].path_ids)
    )
    seal = {
        "schema": RUN_SCHEMA + "-confirmation-seal",
        "schema_version": 1,
        "opened_once": 1,
        "path_ids": list(paths),
        "selection_file_sha256": selection_sha,
        "checkpoint_file_sha256": selection["checkpoint_file_sha256"],
        "checkpoint_state_sha256": selection["selected_state_sha256"],
        "baseline_file_sha256": selection["baseline_file_sha256"],
        "cache_indexes": {
            role: file_fingerprint(run_dir / "eager_cache" / f"{role}_index.json")
            for role in ("train", "validation")
        },
        "eager_profile": eager_prefix_contract(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": (
            int(args.test_bootstrap_replicates) if args.test_only else 50_000
        ),
        "confirmation_created_after_selection": 1,
        **FORBIDDEN_SCOPE,
    }
    seal["semantic_sha256"] = config_fingerprint(seal)
    atomic_write_json(run_dir / "confirmation_seal.json", seal)
    device = torch.device(args.device)
    baseline = load_tangent_baseline(
        run_dir / str(selection["baseline_path"]),
        expected_sha256=selection["baseline_file_sha256"],
    )
    model = _legacy._load_candidate_model(run_dir, selection, baseline, device)
    source = _legacy._load_source_target(args.parent_coarse_residual_run_dir)
    cohorts = frozen_cache_cohorts("confirmation")
    cohort_indices = tuple(range(len(cohorts))) if not args.test_only else (0,)
    records: list[dict[str, Any]] = []
    accumulator = EagerDiagnosticsAccumulator(
        "confirmation",
        outer_steps=outer_steps,
        selected_steps=selected_steps,
        cohort_indices=cohort_indices,
    )
    for cohort_index in cohort_indices:
        cohort = cohorts[cohort_index]
        current = np.repeat(source[None, :], len(cohort.path_ids), axis=0).copy(
            order="C"
        )
        recompute_tail = False
        for start_step in range(0, outer_steps, SHARD_STEPS):
            selected = next(
                (
                    value
                    for value in selected_steps
                    if start_step <= value < start_step + SHARD_STEPS
                ),
                None,
            )
            cached = None if recompute_tail else _load_valid_confirmation_shard(
                run_dir,
                cohort=cohort,
                start_step=start_step,
                current=current,
                selected_step=selected,
                selection_sha256=selection_sha,
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
                kwargs.update(
                    {
                        "shard_runner": deterministic_test_shard_runner,
                        "branch_runner": deterministic_test_branch_runner,
                    }
                )
            started = time.perf_counter()
            execution = execute_eager_shard(
                state,
                cohort=cohort,
                start_step=start_step,
                root_seed=ROOT_SEED,
                selected_steps=selected_steps,
                **kwargs,
            )
            risks = audit = None
            evidence: Mapping[str, Any] = {"row_count": 0}
            if selected is not None:
                risks, audit, evidence = _confirmation_branch_evidence(
                    execution, model=model
                )
                if selected not in CONTROL_ANCHORS:
                    audit = None
            record = _persist_confirmation_shard(
                run_dir,
                execution=execution,
                selection_sha256=selection_sha,
                risks=risks,
                audit=audit,
                evidence=evidence,
                pipeline_started_at=started,
            )
            current = np.ascontiguousarray(execution.committed_final_states)
            records.append(record)
            accumulator.add(execution)
            print(
                f"eager confirmation cohort={cohort_index} step={start_step} committed",
                flush=True,
            )

    chunks: list[dict[str, np.ndarray]] = []
    for record in records:
        if record.get("selected_step") is None:
            continue
        risk_path = _confirmation_paths(
            run_dir,
            cohort_index=int(record["cohort_index"]),
            start_step=int(record["start_step"]),
        )[1]
        if record.get("risk_file_sha256") != file_fingerprint(risk_path):
            raise ArtifactCompatibilityError("confirmation risk shard changed")
        chunks.append(_load_npz(risk_path))
    if not chunks:
        raise ArtifactCompatibilityError("confirmation contains no selected evidence")
    joined = {
        name: np.concatenate([chunk[name] for chunk in chunks])
        for name in chunks[0]
    }
    table = aggregate_confirmation_improvements(
        sample_keys=joined["sample_keys"],
        row_path_ids=joined["path_ids"],
        outer_steps=joined["outer_steps"],
        phases=joined["phases"],
        midpoint_indices=joined["midpoint_indices"],
        combined_vs_zero_improvements=joined["combined_vs_zero"],
        combined_vs_baseline_improvements=joined["combined_vs_baseline"],
        expected_path_ids=paths,
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
    atomic_write_json(run_dir / "confirmation_risk_summary.json", table.to_record())
    replicates = int(args.test_bootstrap_replicates) if args.test_only else 50_000
    max_t = one_sided_whole_path_max_t(
        table.path_values,
        path_ids=table.path_ids,
        confidence=0.995,
        replicates=replicates,
        seed=BOOTSTRAP_SEED,
    )
    atomic_write_json(run_dir / "confirmation_max_t.json", max_t)
    confirmation_bytes = sum(
        path.stat().st_size
        for path in (run_dir / "confirmation").rglob("*")
        if path.is_file()
    ) + int(risk_artifact["size"])
    aggregate = accumulator.to_record(persisted_bytes=confirmation_bytes)
    complete_elapsed = sum(
        float(record["complete_pipeline_elapsed_seconds"]) for record in records
    )
    cohort_rates: list[float] = []
    for cohort_index in cohort_indices:
        cohort_records = [
            record
            for record in records
            if int(record["cohort_index"]) == int(cohort_index)
        ]
        cohort_elapsed = sum(
            float(record["complete_pipeline_elapsed_seconds"])
            for record in cohort_records
        )
        cohort_transitions = sum(
            int(record["execution"]["diagnostics"]["transition_count"])
            for record in cohort_records
        )
        if cohort_elapsed <= 0.0 or cohort_transitions <= 0:
            raise ArtifactCompatibilityError(
                "confirmation cohort timing is nonpositive"
            )
        cohort_rates.append(cohort_transitions / cohort_elapsed)
    total = int(aggregate["transition_count"])
    fallback = int(aggregate["fallback_count"])
    cache_metrics = _load_json(run_dir / "cache_metrics.json")
    t = BoundaryTangentEagerThresholds()
    confirmation_rows = (
        t.confirmation_rows
        if args.test_only
        else int(table.combined_vs_zero_row_count)
    )
    confirmation_transitions = (
        t.confirmation_transitions if args.test_only else total
    )
    forbidden = sum(int(value) for value in aggregate["forbidden_counts"].values())
    identity_error = max(
        float(record.get("evidence", {}).get("direct_derived_total_contrast_maximum_error", 0.0))
        for record in records
    )
    metrics = {
        "schema": RUN_SCHEMA + "-confirmation-metrics",
        "schema_version": 1,
        **{name: 1 for name in CONFIRM_EXECUTION_FLAGS},
        "confirmation_path_count": t.confirmation_paths if args.test_only else len(paths),
        "confirmation_row_count": confirmation_rows,
        "confirmation_transition_count": confirmation_transitions,
        "test_actual_confirmation_path_count": len(paths),
        "test_actual_confirmation_row_count": int(
            table.combined_vs_zero_row_count
        ),
        "test_actual_confirmation_transition_count": total,
        "certificate_fraction": int(aggregate["certified_count"]) / max(total, 1),
        "maximum_mass_error": float(aggregate["maximum_mass_error"]),
        "forbidden_event_count": forbidden,
        "transitions_per_second": min(cohort_rates),
        "aggregate_transitions_per_second": total
        / max(complete_elapsed, np.finfo(float).tiny),
        "fallback_fraction": fallback / max(total, 1),
        "fallback_time_fraction": float(aggregate["fallback_elapsed_seconds"]) / max(complete_elapsed, np.finfo(float).tiny),
        "peak_memory_fraction": float(aggregate["maximum_peak_memory_fraction"]),
        "total_persisted_bytes": int(cache_metrics["total_persisted_cache_bytes"]) + confirmation_bytes,
        "maximum_launch_lanes": int(aggregate["maximum_launch_lanes"]),
        "cache_elapsed_seconds": float(cache_metrics["cache_elapsed_seconds"]),
        "confirmation_elapsed_seconds": complete_elapsed,
        "direct_derived_total_contrast_maximum_error": identity_error,
        "production_cache_generation_performed": 1,
        "physical_training_performed": 1,
        "confirmation_performed": 1,
        "raw_confirmation_labels_not_persisted": 1,
        **FORBIDDEN_SCOPE,
    }
    atomic_write_json(run_dir / "confirmation_metrics.json", metrics)
    integrity = {
        "sealed_confirmation_valid": True,
        "direct_and_derived_total_contrast_agree": identity_error <= 5.0e-15,
        "streaming_label_firewall_valid": all(
            "denoising_target" not in _load_npz(
                _confirmation_paths(
                    run_dir,
                    cohort_index=int(record["cohort_index"]),
                    start_step=int(record["start_step"]),
                )[1]
            )
            for record in records
            if record.get("selected_step") is not None
        ),
    }
    gate = evaluate_confirm_gate(max_t, metrics, integrity_checks=integrity)
    atomic_write_json(gate_path, gate)
    index = {
        "schema": RUN_SCHEMA + "-confirmation-index",
        "schema_version": 1,
        "selection_sha256": selection_sha,
        "path_ids": list(paths),
        "shards": [
            {
                "cohort_index": int(record["cohort_index"]),
                "start_step": int(record["start_step"]),
                "metadata_sha256": file_fingerprint(
                    _confirmation_paths(
                        run_dir,
                        cohort_index=int(record["cohort_index"]),
                        start_step=int(record["start_step"]),
                    )[3]
                ),
            }
            for record in records
        ],
        "risk_sha256": risk_artifact["sha256"],
        "max_t_sha256": file_fingerprint(run_dir / "confirmation_max_t.json"),
        **FORBIDDEN_SCOPE,
    }
    index["semantic_sha256"] = config_fingerprint(index)
    atomic_write_json(run_dir / "confirmation_index.json", index)
    _seal_stage(
        run_dir,
        (
            "confirmation_seal.json",
            "confirmation_path_risks.npz",
            "confirmation_risk_summary.json",
            "confirmation_max_t.json",
            "confirmation_metrics.json",
            "confirmation_index.json",
            "confirm_gate.json",
        ),
        "confirm_artifact_seal.json",
    )
    return gate


def _workflow_record(run_dir: Path, *, require_gate: str) -> dict[str, Any]:
    gates = {
        "preflight_gate": _optional_json(run_dir, "preflight_gate.json"),
        "cache_gate": _optional_json(run_dir, "cache_gate.json"),
        "train_gate": _optional_json(run_dir, "train_gate.json"),
        "confirm_gate": _optional_json(run_dir, "confirm_gate.json"),
    }
    workflow = evaluate_required_gate(
        **gates,
        require_gate=require_gate,
    )
    decision = decide_workflow(**gates)
    config = _load_json(run_dir / "scientific_config.json")
    if int(config.get("test_only", 0)) == 1:
        def nonauthorizing(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: (
                        0
                        if "authorized" in str(key)
                        else nonauthorizing(item)
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [nonauthorizing(item) for item in value]
            return value

        workflow = nonauthorizing(workflow)
        decision = nonauthorizing(decision)
        workflow["authorizing_run"] = 0
        decision["authorizing_run"] = 0
        decision["recommended_next_action"] = (
            "nonauthorizing test fixture only; run the frozen production workflow"
        )
        workflow["decision"] = decision
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(run_dir / "boundary_tangent_eager_decision.json", decision)
    return workflow


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "cache", "train", "confirm")
    if stage == "report":
        return ()
    if stage in {"preflight", "cache", "train", "confirm"}:
        return (stage,)
    raise ValueError(f"unknown stage: {stage}")


def _verify_report(run_dir: Path) -> None:
    seals = {
        "preflight_gate.json": "preflight_artifact_seal.json",
        "cache_gate.json": "cache_artifact_seal.json",
        "train_gate.json": "train_artifact_seal.json",
        "confirm_gate.json": "confirm_artifact_seal.json",
    }
    for gate, seal in seals.items():
        if (run_dir / gate).is_file():
            if not (run_dir / seal).is_file():
                raise ArtifactCompatibilityError(f"{gate} has no artifact seal")
            _verify_stage_seal(run_dir, seal)
    if (run_dir / "checkpoint_selection.json").is_file():
        _verify_training_selection(run_dir)


def _commit_execution_failure(
    run_dir: Path,
    *,
    stage: str,
    exc: BaseException,
    require_gate: str,
) -> None:
    domain = str(getattr(exc, "failure_domain", "workflow_execution"))
    code = str(
        getattr(exc, "failure_code", "boundary_tangent_eager_execution_failed")
    )
    failure = {
        "schema": RUN_SCHEMA + "-execution-failure",
        "schema_version": 1,
        "stage": stage,
        "evaluation_status": "execution_failed",
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "failure_domain": domain,
        "failure_code": code,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "recorded_at": _now(),
        **_scope(run_dir),
    }
    failure_name = f"{stage}_execution_failure.json"
    atomic_write_json(run_dir / failure_name, failure)
    normalized = stage if stage in {"preflight", "cache", "train", "confirm"} else "preflight"
    gate_name = f"{normalized}_gate.json"
    seal_name = f"{normalized}_artifact_seal.json"
    if not (run_dir / gate_name).is_file():
        if normalized == "preflight":
            gate = evaluate_preflight_gate(failure)
        elif normalized == "cache":
            gate = evaluate_cache_gate(failure)
        elif normalized == "train":
            gate = evaluate_train_gate(failure)
        else:
            gate = evaluate_confirm_gate({}, failure)
        atomic_write_json(run_dir / gate_name, gate)
        _seal_stage(run_dir, (failure_name, gate_name), seal_name)
    workflow = _workflow_record(run_dir, require_gate=require_gate)
    decision = str(workflow["decision"]["decision"])
    _status(
        run_dir,
        state="execution_failed",
        stage=stage,
        decision=decision,
        message=str(exc),
        failure_domain=domain,
        failure_code=code,
        scientific_evidence_complete=0,
    )
    _artifact_registry(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-eager-pipeline-run-dir", type=Path, required=True)
    parser.add_argument("--failed-boundary-tangent-run-dir", type=Path, required=True)
    parser.add_argument("--parent-coarse-residual-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-eager-boundary-tangent-time-local"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--test-path-count", type=int, default=8, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--test-outer-steps", type=int, default=16, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--test-maximum-updates", type=int, default=1, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--test-bootstrap-replicates",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    for name in (
        "parent_eager_pipeline_run_dir",
        "failed_boundary_tangent_run_dir",
        "parent_coarse_residual_run_dir",
        "runs_root",
        "resume_run_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if not args.test_only:
        if args.stage != "report" and args.device != "cuda":
            parser.error("authorizing scientific stages require --device cuda")
        if (
            args.test_path_count != 8
            or args.test_outer_steps != 16
            or args.test_maximum_updates != 1
            or args.test_bootstrap_replicates != 100
        ):
            parser.error("test overrides require --test-only")
    elif args.require_gate != "none":
        parser.error("test-only runs are nonauthorizing and require --require-gate none")
    if not 1 <= args.test_path_count <= 8:
        parser.error("--test-path-count must be in [1,8]")
    if (
        args.test_outer_steps < 16
        or args.test_outer_steps > OUTER_STEPS
        or args.test_outer_steps % SHARD_STEPS
    ):
        parser.error("--test-outer-steps must be a multiple of eight in [16,512]")
    if not 0 <= args.test_maximum_updates <= int(TRAINING["maximum_updates"]):
        parser.error("--test-maximum-updates must be in [0,4000]")
    if not 8 <= args.test_bootstrap_replicates <= 50_000:
        parser.error("--test-bootstrap-replicates must be in [8,50000]")
    if args.resume_run_dir is None and args.stage in {
        "cache",
        "train",
        "confirm",
        "report",
    }:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    if not args.test_only:
        stage_gate = {
            "preflight": "preflight",
            "cache": "cache",
            "train": "train",
            "confirm": "confirm",
            "all": "confirm",
            "report": "none",
        }[args.stage]
        if args.require_gate not in {"none", stage_gate}:
            parser.error(
                f"--stage {args.stage} cannot require only {args.require_gate}"
            )
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    resumed = False
    initialized = False
    active_stage = "initialize"
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"eager boundary-tangent v2 run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        if args.stage != "report":
            configure_exact_torch_backend(args.device)
        _status(run_dir, state="running", stage=args.stage)
        if args.stage == "report":
            _verify_report(run_dir)
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            _status(run_dir, state="running", stage=stage)
            if stage == "preflight":
                gate = _preflight_stage(run_dir, args)
            elif stage == "cache":
                if not _passed(_optional_json(run_dir, "preflight_gate.json")):
                    raise ArtifactCompatibilityError(
                        "cache requires a passing preflight gate"
                    )
                gate = _cache_stage(run_dir, args)
            elif stage == "train":
                if not _passed(_optional_json(run_dir, "cache_gate.json")):
                    raise ArtifactCompatibilityError(
                        "train requires a passing cache gate"
                    )
                gate = _train_stage(run_dir, args)
            elif stage == "confirm":
                if not _passed(_optional_json(run_dir, "train_gate.json")):
                    raise ArtifactCompatibilityError(
                        "confirm requires a passing train gate"
                    )
                gate = _confirm_stage(run_dir, args)
            else:  # pragma: no cover
                raise AssertionError(stage)
            if not _passed(gate):
                break

        workflow = _workflow_record(run_dir, require_gate=args.require_gate)
        decision = str(workflow["decision"]["decision"])
        required_pass = int(workflow.get("required_gate_pass", 0)) == 1
        terminal = (
            _optional_json(run_dir, "confirm_gate.json")
            or _optional_json(run_dir, "train_gate.json")
            or _optional_json(run_dir, "cache_gate.json")
            or _optional_json(run_dir, "preflight_gate.json")
            or {}
        )
        stage_evidence_pass = (
            args.stage == "report" or _passed(terminal)
        )
        _status(
            run_dir,
            state=(
                "test_only_complete"
                if args.test_only and required_pass and stage_evidence_pass
                else (
                    "complete"
                    if required_pass and stage_evidence_pass
                    else "gate_failed"
                )
            ),
            stage=args.stage,
            decision=decision,
            failure_domain=None
            if required_pass
            else str(terminal.get("failure_domain") or "scientific_gate"),
            failure_code=None
            if required_pass
            else f"{args.require_gate}_gate_failed",
            scientific_evidence_complete=int(
                (required_pass and stage_evidence_pass)
                or int(terminal.get("scientific_evidence_complete", 0)) == 1
            ),
        )
        _artifact_registry(run_dir)
        print(f"eager boundary-tangent v2 decision: {decision}", flush=True)
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
        # A rejected resume is never mutated.  A fresh initialization failure
        # still commits readable fail-closed evidence.
        if run_dir is not None and (initialized or not resumed):
            _commit_execution_failure(
                run_dir,
                stage=active_stage,
                exc=exc,
                require_gate=args.require_gate,
            )
        print(f"eager boundary-tangent v2 error: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
