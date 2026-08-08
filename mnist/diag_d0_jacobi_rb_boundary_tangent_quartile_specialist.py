"""Exact Jacobi/RB quartile-specialist one-image workflow.

The workflow is deliberately additive and fail closed.  It trains four
independent zero-baseline boundary-tangent experts against the unchanged raw
Rao--Blackwell label, calibrates only q2/q3 on a disjoint training role, and
opens fresh selection/confirmation paths only after one system is sealed.

Nothing in this module executes a reverse controller or a sampler.
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
from typing import Any, Iterable, Mapping, Sequence

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
from mnist.d0_jacobi_rb_boundary_tangent import synthetic_tangent_target
from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    MIDPOINT_COUNT,
    MIDPOINT_FRACTIONS,
    SELECTED_OUTER_STEPS,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import (
    EagerCohort,
    deterministic_test_branch_runner,
    deterministic_test_shard_runner,
    execute_eager_shard,
    explicit_eager_cache_plan,
    generate_eager_cache_for_cohorts,
    iter_eager_shards_for_cohorts,
    load_eager_role_inputs,
    load_eager_role_labels,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_gate import (
    CACHE_FLAGS,
    CALIBRATE_FLAGS,
    CONFIRM_FLAGS,
    CONTROLS_FLAGS,
    PREFLIGHT_FLAGS,
    REQUIRED_GATES,
    SELECT_FLAGS,
    TRAIN_FLAGS,
    decide_workflow,
    decision_exit_code,
    evaluate_cache_gate,
    evaluate_calibrate_gate,
    evaluate_confirm_gate,
    evaluate_controls_gate,
    evaluate_preflight_gate,
    evaluate_required_gate,
    evaluate_select_gate,
    evaluate_train_gate,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_provenance import (
    CONFIRMATION_BOOTSTRAP_NAMESPACE,
    CONFIRMATION_BOOTSTRAP_SEED,
    EXACT_NULL_CONTROL_ROOT_SEED,
    PHYSICAL_MODEL_SEEDS,
    ROLE_OPEN_ORDER,
    ROOT_SEED,
    SELECTION_BOOTSTRAP_NAMESPACE,
    SELECTION_BOOTSTRAP_SEED,
    SYNTHETIC_CONTROL_SEEDS,
    build_cohort_plan,
    build_path_id_plan,
    build_role_firewall,
    build_seed_plan,
    quartile_source_fingerprint,
    quartile_source_paths,
    validate_cohort_plan,
    validate_path_id_plan,
    validate_role_open_order,
    validate_seed_plan,
    validate_semantic_config,
    verify_quartile_specialist_parents,
    verify_resume_compatibility,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_selection import (
    CONFIRMATION_NAMESPACE,
    DEFAULT_CONFIDENCE,
    DEFAULT_REPLICATES,
    DEFAULT_SHARD_SIZE,
    LOCAL_FAMILY_NAMES_SHA256,
    PRIMARY_FAMILY_NAMES,
    PRIMARY_FAMILY_NAMES_SHA256,
    PRODUCTION_PATH_COUNT,
    QuartileAuditPathTable,
    SELECTION_NAMESPACE,
    aggregate_quartile_audit_improvements,
    confirmation_record,
    count_shard_index_record,
    evaluate_local_compatibility_screen,
    prepare_bootstrap_count_shards,
    restartable_quartile_max_t,
    selection_record,
    shard_artifact_paths,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_specialist import (
    CANDIDATE_GRID_SHA256,
    CHECKPOINT_UPDATES,
    MODEL_SEEDS_BY_QUARTILE,
    NONZERO_CANDIDATE_IDENTITIES,
    CandidateIdentity,
    HashBinding,
    NoEligibleQuartileCandidateError,
    QuartileSpecialistBoundaryTangentPredictor,
    SelectedExpert,
    SelectedSystem,
    build_training_rank_record,
    candidate_grid_record,
    exact_quartile_target_scale,
    fixed_unit_gain_record,
    gain_record_from_moments,
    reconstruct_forward_outer_quartile,
    scaled_raw_target_mse,
    select_training_rank_system,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import (
    HostInputStore,
    ModelCallBatchGuard,
    open_external_input_store,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    MIXED_TARGET_SHA256,
    SOURCE_IMAGE_NPZ_SHA256,
    source_measure_sha256,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
    configure_exact_synthetic_zero_baseline_teacher,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    OUTER_STEPS,
    PHASE_COUNT,
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


RUN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-quartile-specialist"
STAGES = (
    "preflight",
    "cache",
    "controls",
    "train",
    "calibrate",
    "select",
    "confirm",
    "report",
    "all",
)
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
SHARD_STEPS = 8
MAXIMUM_PEAK_MEMORY_FRACTION = 0.80
MINIMUM_TRANSITIONS_PER_SECOND = 1_300.0
MAXIMUM_PERSISTED_BYTES = 3 * 1024**3
MAXIMUM_PROJECTED_HOURS = 160.0
PRODUCTION_SELECTED_ROWS_PER_PATH = 32 * PHASE_COUNT * MIDPOINT_COUNT
PRODUCTION_TRANSITIONS_PER_PATH = (
    OUTER_STEPS * PHASE_COUNT * EDGES_PER_PHASE
    + len(SELECTED_OUTER_STEPS) * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
)

NO_WORK = {
    "controller_execution_performed": 0,
    "reverse_controller_trajectory_performed": 0,
    "full_reverse_path_performed": 0,
    "reconstruction_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "image_sampling_performed": 0,
    "full_dataset_training_performed": 0,
}
_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "quartile_specialist_decision.json",
}


class QuartileSpecialistWorkflowError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "quartile_specialist_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


class _RawQuartilePredictionAdapter(nn.Module):
    """Expose the sealed system's pre-gain output through the model firewall."""

    def __init__(self, model: QuartileSpecialistBoundaryTangentPredictor) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: ModelInputs) -> Tensor:
        return self.model.raw_prediction(inputs)


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
    result = dict(value)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _verify_semantic(value: Mapping[str, Any], description: str) -> None:
    body = dict(value)
    observed = body.pop("semantic_sha256", None)
    if observed != config_fingerprint(body):
        raise ArtifactCompatibilityError(f"{description} semantic hash changed")


def _passed(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("evaluation_status") == "evaluated"
        and int(value.get("passed", 0)) == 1
    )


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
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
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": target.as_posix(),
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _atomic_torch(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": target.as_posix(),
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone().contiguous()
        for name, tensor in model.state_dict().items()
    }


def _physical_training_committed(run_dir: Path) -> bool:
    for path in (run_dir / "checkpoints").glob("q*/seed-*/update-*.pt"):
        try:
            if int(path.stem.split("-")[-1]) > 0:
                return True
        except ValueError:
            continue
    return False


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED:
            continue
        relative = path.relative_to(run_dir).as_posix()
        artifacts.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-artifact-registry",
            "schema_version": 1,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "fit_labels_opened": int((run_dir / "fit_label_open.json").is_file()),
            "physical_training_performed": int(
                _physical_training_committed(run_dir)
            ),
            "selection_paths_opened": int((run_dir / "selection_open.json").is_file()),
            "confirmation_paths_opened": int(
                (run_dir / "confirmation_open.json").is_file()
            ),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _seal_stage(run_dir: Path, names: Iterable[str], seal_name: str) -> dict[str, Any]:
    bindings = []
    for name in sorted(set(str(value) for value in names)):
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"stage artifact is absent: {name}")
        semantic_sha = None
        if path.suffix == ".json":
            value = _load_json(path)
            semantic_sha = value.get("semantic_sha256")
        bindings.append(
            {
                "path": name,
                "sha256": file_fingerprint(path),
                "semantic_sha256": semantic_sha,
            }
        )
    seal = _semantic(
        {
            "schema": RUN_SCHEMA + "-stage-seal",
            "schema_version": 1,
            "seal_name": seal_name,
            "artifacts": bindings,
        }
    )
    atomic_write_json(run_dir / seal_name, seal)
    return seal


def _verify_stage_seal(run_dir: Path, seal_name: str) -> dict[str, Any]:
    seal = _load_json(run_dir / seal_name)
    _verify_semantic(seal, seal_name)
    for artifact in seal.get("artifacts", []):
        path = run_dir / str(artifact["path"])
        if not path.is_file() or file_fingerprint(path) != artifact.get("sha256"):
            raise ArtifactCompatibilityError(f"sealed artifact changed: {path}")
        if artifact.get("semantic_sha256") is not None:
            value = _load_json(path)
            if value.get("semantic_sha256") != artifact["semantic_sha256"]:
                raise ArtifactCompatibilityError(f"sealed semantic record changed: {path}")
    return seal


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    message: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
) -> None:
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-status",
        "schema_version": 1,
        "state": state,
        "stage": stage,
        "updated_at": _now(),
        "decision": decision,
        "message": message,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "fit_labels_opened": int((run_dir / "fit_label_open.json").is_file()),
        "physical_training_performed": int(_physical_training_committed(run_dir)),
        "selection_paths_opened": int((run_dir / "selection_open.json").is_file()),
        "confirmation_paths_opened": int(
            (run_dir / "confirmation_open.json").is_file()
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "run_status.json", record)


def _source_target(parent_coarse_witness_run_dir: str | Path) -> np.ndarray:
    parent = Path(parent_coarse_witness_run_dir).resolve()
    candidates = (
        parent / "source_image.npz",
        parent.parents[2] / "runs"
        / "experiment12_d0_jacobi_rb_coarse_residual_learnability"
        / "20260731-140333_production-exact-k512-coarse-residual-one-image"
        / "source_image.npz",
    )
    for path in candidates:
        if path.is_file():
            if file_fingerprint(path) != SOURCE_IMAGE_NPZ_SHA256:
                raise ArtifactCompatibilityError(
                    "immutable source_image.npz fingerprint changed"
                )
            with np.load(path, allow_pickle=False) as archive:
                target = np.array(archive["mixed_target"], dtype=np.float64, copy=True)
            if (
                target.shape == (STATE_SIZE,)
                and np.isfinite(target).all()
                and source_measure_sha256(target) == MIXED_TARGET_SHA256
            ):
                return np.ascontiguousarray(target)
    raise ArtifactCompatibilityError("cannot locate the immutable mixed source target")


def _effective_counts(args: argparse.Namespace) -> dict[str, int]:
    if not args.test_only:
        return {
            "preflight_seam": 8,
            "physical_fit": 64,
            "gain_calibration": 32,
            "training_rank": 32,
            "fresh_selection": 384,
            "untouched_confirmation": 384,
        }
    small = int(args.test_path_count)
    return {
        "preflight_seam": min(8, small),
        "physical_fit": small,
        "gain_calibration": small,
        "training_rank": small,
        "fresh_selection": max(8, small),
        "untouched_confirmation": max(8, small),
    }


def _effective_paths(args: argparse.Namespace, role: str) -> tuple[int, ...]:
    plan = build_path_id_plan()
    count = _effective_counts(args)[role]
    return tuple(int(value) for value in plan["roles"][role][:count])


def _effective_outer_steps(args: argparse.Namespace) -> int:
    return int(args.test_outer_steps) if args.test_only else OUTER_STEPS


def _effective_selected_steps(args: argparse.Namespace) -> tuple[int, ...]:
    outer = _effective_outer_steps(args)
    selected = tuple(value for value in SELECTED_OUTER_STEPS if value < outer)
    if not selected and args.test_only:
        selected = (outer - 1,)
    return selected


def _effective_updates(args: argparse.Namespace) -> int:
    return int(args.test_maximum_updates) if args.test_only else 4_000


def _effective_bootstrap(args: argparse.Namespace) -> tuple[int, int]:
    if not args.test_only:
        return DEFAULT_REPLICATES, DEFAULT_SHARD_SIZE
    replicates = int(args.test_bootstrap_replicates)
    shard = int(args.test_bootstrap_shard_size)
    if replicates % shard:
        raise ArtifactCompatibilityError("test bootstrap replicates must divide by shard size")
    return replicates, shard


def _verify_prospective_bootstrap_counts(
    run_dir: Path, args: argparse.Namespace
) -> None:
    """Verify both precommitted count families without regenerating a shard."""

    expected_path_count = _effective_counts(args)["fresh_selection"]
    expected_replicates, expected_shard_size = _effective_bootstrap(args)
    for role in ("selection", "confirmation"):
        index = _load_json(run_dir / f"{role}_bootstrap_count_index.json")
        _verify_semantic(index, f"{role}_bootstrap_count_index.json")
        shard_count = expected_replicates // expected_shard_size
        if (
            int(index.get("path_count", -1)) != expected_path_count
            or int(index.get("replicate_count", -1)) != expected_replicates
            or int(index.get("shard_count", -1)) != shard_count
        ):
            raise ArtifactCompatibilityError("prospective bootstrap plan changed")
        records = []
        directory = run_dir / "bootstrap_counts" / role
        for shard_index in range(shard_count):
            _data_path, metadata_path = shard_artifact_paths(directory, shard_index)
            record = _load_json(metadata_path)
            _verify_semantic(record, metadata_path.name)
            records.append(record)
        rebuilt = count_shard_index_record(records, role=role)
        if rebuilt != index:
            raise ArtifactCompatibilityError("prospective bootstrap evidence changed")


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    path_plan = build_path_id_plan()
    cohort_plan = build_cohort_plan(path_plan)
    seed_plan = build_seed_plan()
    body = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "grid_size": 28,
        "alpha": 1.0,
        "outer_steps": 512,
        "tau_eff": 5.0e-5,
        "label": 3,
        "class_index": 0,
        "dataset_index": 7,
        "lambda_mix": 0.35,
        "image_sha256": (
            "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
        ),
        "target": "exact_binary64_jacobi_rao_blackwell_raw_label",
        "target_formula": "y(1-y)*d_y log k_u(y|x)",
        "predictor_formula": "y(1-y)*q_theta(W)",
        "expert_count": 4,
        "expert_width": 32,
        "training": dict(TRAINING),
        "quartile_target_scales": "one_training_only_rms_per_quartile",
        "q0_q1_gain": 1.0,
        "q2_q3_gain": "training_only_C_over_P_open_unit_interval",
        "candidate_grid_sha256": CANDIDATE_GRID_SHA256,
        "path_id_plan_sha256": path_plan["semantic_sha256"],
        "cohort_plan_sha256": cohort_plan["semantic_sha256"],
        "seed_plan_sha256": seed_plan["semantic_sha256"],
        "primary_family_names": list(PRIMARY_FAMILY_NAMES),
        "primary_family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
        "local_family_names_sha256": LOCAL_FAMILY_NAMES_SHA256,
        "selection_path_count": 384,
        "confirmation_path_count": 384,
        "bootstrap_replicates": 50_000,
        "bootstrap_confidence": 0.995,
        "bootstrap_quantile_method": "higher",
        "minimum_exact_transitions_per_second": MINIMUM_TRANSITIONS_PER_SECOND,
        "maximum_projected_exact_capture_hours": MAXIMUM_PROJECTED_HOURS,
        "maximum_persisted_artifact_bytes": MAXIMUM_PERSISTED_BYTES,
        "maximum_peak_memory_fraction": MAXIMUM_PEAK_MEMORY_FRACTION,
        "controller_execution_authorized": 0,
        "sampling_authorized": 0,
        "test_only": int(args.test_only),
        "authorizing": int(not args.test_only),
    }
    return _semantic(body)


def _manifest(args: argparse.Namespace, run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "run_schema": RUN_SCHEMA,
            "run_name": args.run_name,
            "run_dir": str(run_dir),
            "created_at": _now(),
            "device": str(args.device),
            "source_fingerprint": quartile_source_fingerprint(),
            "scientific_config_sha256": config["semantic_sha256"],
            "parent_time_local_run_dir": str(Path(args.parent_time_local_run_dir).resolve()),
            "parent_memory_v3_run_dir": str(Path(args.parent_memory_v3_run_dir).resolve()),
            "parent_coarse_witness_run_dir": str(
                Path(args.parent_coarse_witness_run_dir).resolve()
            ),
            "parent_bayes_power_run_dir": str(Path(args.parent_bayes_power_run_dir).resolve()),
            "test_only": int(args.test_only),
            "authorizing": int(not args.test_only),
            **NO_WORK,
        }
    )


def _resume_bindings(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_schema": RUN_SCHEMA,
        "device": str(args.device),
        "source_fingerprint": quartile_source_fingerprint(),
        "scientific_config_sha256": config["semantic_sha256"],
        "parent_time_local_run_dir": str(Path(args.parent_time_local_run_dir).resolve()),
        "parent_memory_v3_run_dir": str(Path(args.parent_memory_v3_run_dir).resolve()),
        "parent_coarse_witness_run_dir": str(
            Path(args.parent_coarse_witness_run_dir).resolve()
        ),
        "parent_bayes_power_run_dir": str(Path(args.parent_bayes_power_run_dir).resolve()),
        "test_only": int(args.test_only),
    }


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    if args.stage not in {"preflight", "all"}:
        raise ArtifactCompatibilityError("a new run may begin only with preflight")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (Path(args.runs_root) / f"{timestamp}_{args.run_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, False


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    config = _scientific_config(args)
    if resumed:
        verify_resume_compatibility(
            run_dir,
            expected_bindings=_resume_bindings(args, config),
            artifact_bindings={
                "scientific_config.json": config["semantic_sha256"],
                "path_id_plan.json": build_path_id_plan()["semantic_sha256"],
                "cohort_plan.json": build_cohort_plan()["semantic_sha256"],
                "seed_plan.json": build_seed_plan()["semantic_sha256"],
                "role_firewall.json": build_role_firewall()["semantic_sha256"],
            },
        )
        return
    manifest = _manifest(args, run_dir, config)
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "scientific_config.json", config)
    atomic_write_json(run_dir / "path_id_plan.json", build_path_id_plan())
    atomic_write_json(run_dir / "cohort_plan.json", build_cohort_plan())
    atomic_write_json(run_dir / "seed_plan.json", build_seed_plan())
    atomic_write_json(run_dir / "role_firewall.json", build_role_firewall())
    atomic_write_json(run_dir / "candidate_grid_plan.json", _semantic(candidate_grid_record()))
    atomic_write_json(
        run_dir / "gain_calibration_plan.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-gain-plan",
                "formula": "lambda=C/P",
                "roles": ["gain_calibration"],
                "quartiles": [2, 3],
                "clipping_or_projection": 0,
                "eligible_interval": "0<lambda<1",
            }
        ),
    )
    atomic_write_json(
        run_dir / "training_rank_plan.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-rank-plan",
                "role": "training_rank",
                "minimum_positive_fine_cells": 51,
                "q1_sentinel": "phase4.midpoint7",
                "ranking": "largest_pooled_then_earlier_update_then_lower_seed",
                "cartesian_product_search": 0,
            }
        ),
    )
    for role, seed, namespace in (
        ("selection", SELECTION_BOOTSTRAP_SEED, SELECTION_BOOTSTRAP_NAMESPACE),
        ("confirmation", CONFIRMATION_BOOTSTRAP_SEED, CONFIRMATION_BOOTSTRAP_NAMESPACE),
    ):
        plan = _semantic(
            {
                "schema": RUN_SCHEMA + f"-{role}-inference-plan",
                "path_count": 384,
                "seed": seed,
                "namespace": namespace,
                "replicates": 50_000,
                "confidence": 0.995,
                "family_names": list(PRIMARY_FAMILY_NAMES),
                "family_names_sha256": PRIMARY_FAMILY_NAMES_SHA256,
                "local_family_names_sha256": LOCAL_FAMILY_NAMES_SHA256,
                "raw_cache_persisted": 0,
            }
        )
        atomic_write_json(run_dir / f"{role}_inference_plan.json", plan)
        if role == "confirmation":
            atomic_write_json(run_dir / "confirmation_plan.json", plan)
    _status(run_dir, state="initialized", stage="preflight")


def _cohorts(paths: Sequence[int], *, kind: str, role: str) -> tuple[EagerCohort, ...]:
    values = tuple(int(value) for value in paths)
    return tuple(
        EagerCohort(
            kind=kind,
            index=index,
            path_ids=values[start : start + 10],
            path_roles=(role,) * len(values[start : start + 10]),
        )
        for index, start in enumerate(range(0, len(values), 10))
    )


def _cache_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "outer_steps": _effective_outer_steps(args),
        "selected_steps": _effective_selected_steps(args),
    }
    if args.test_only:
        result.update(
            {
                "shard_runner": deterministic_test_shard_runner,
                "branch_runner": deterministic_test_branch_runner,
            }
        )
    return result


def _generate_role_cache(
    root: Path,
    source: np.ndarray,
    paths: Sequence[int],
    *,
    args: argparse.Namespace,
    progress_name: str,
) -> dict[str, Any]:
    cohorts = _cohorts(paths, kind="train_validation", role="train")
    plan = explicit_eager_cache_plan(cohorts)

    def progress(identity: Any, disposition: str) -> None:
        print(
            f"quartile {progress_name} cohort={identity.cohort_index} "
            f"step={identity.start_step} {disposition}",
            flush=True,
        )

    result = generate_eager_cache_for_cohorts(
        root,
        source,
        cohorts=cohorts,
        cohort_plan_sha256=plan["semantic_sha256"],
        device=args.device,
        root_seed=ROOT_SEED,
        progress=progress,
        **_cache_generation_kwargs(args),
    )
    return {"result": result, "cohort_plan": plan}


def _metric_fraction(metrics: Mapping[str, Any], numerator: str, denominator: str) -> float:
    return int(metrics.get(numerator, 0)) / max(int(metrics.get(denominator, 0)), 1)


def _exact_execution_health(metrics: Mapping[str, Any]) -> dict[str, Any]:
    transitions = int(metrics.get("transition_count", 0))
    certified = int(metrics.get("certified_count", 0))
    elapsed = float(metrics.get("complete_pipeline_elapsed_seconds", 0.0))
    fallback_count = int(metrics.get("fallback_count", 0))
    fallback_elapsed = float(metrics.get("fallback_elapsed_seconds", 0.0))
    peak = float(
        metrics.get(
            "maximum_peak_memory_fraction",
            metrics.get("peak_memory_fraction", 0.0),
        )
    )
    rate = transitions / max(elapsed, np.finfo(float).tiny)
    fallback_fraction = fallback_count / max(transitions, 1)
    fallback_time_fraction = fallback_elapsed / max(elapsed, np.finfo(float).tiny)
    numerical = bool(
        transitions > 0
        and certified == transitions
        and int(metrics.get("forbidden_event_count", 0)) == 0
        and float(metrics.get("maximum_mass_error", math.inf)) <= 2.0e-12
    )
    resource = bool(
        rate >= MINIMUM_TRANSITIONS_PER_SECOND
        and fallback_fraction <= 1.0e-4
        and fallback_time_fraction <= 0.10
        and peak <= MAXIMUM_PEAK_MEMORY_FRACTION
    )
    return {
        "numerically_valid": int(numerical),
        "resource_valid": int(resource),
        "transitions_per_second": rate,
        "fallback_fraction": fallback_fraction,
        "fallback_time_fraction": fallback_time_fraction,
        "peak_memory_fraction": peak,
    }


def _parent_immutability_record(parent: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-parent-immutability",
            "schema_version": 1,
            "phase": str(phase),
            "parent_provenance_semantic_sha256": parent.get("semantic_sha256"),
            "authoritative_parent": parent.get("authoritative_parent"),
            "transitive_parents": parent.get("transitive_parents"),
            "parents_mutated": 0,
        }
    )


def _verify_frozen_parents(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Reverify every immutable parent before a resumed stage mutates evidence."""

    try:
        frozen = _load_json(run_dir / "parent_provenance.json")
        _verify_semantic(frozen, "parent_provenance.json")
        current = verify_quartile_specialist_parents(
            time_local_run_dir=args.parent_time_local_run_dir,
            memory_v3_run_dir=args.parent_memory_v3_run_dir,
            coarse_witness_run_dir=args.parent_coarse_witness_run_dir,
            bayes_power_run_dir=args.parent_bayes_power_run_dir,
        )
        if current != frozen:
            raise ArtifactCompatibilityError("immutable parent commitment changed")
        before = _load_json(run_dir / "parent_immutability_before.json")
        _verify_semantic(before, "parent_immutability_before.json")
        expected_before = _parent_immutability_record(frozen, phase="before")
        if before != expected_before:
            raise ArtifactCompatibilityError("parent immutability baseline changed")
        return current
    except ArtifactCompatibilityError as exc:
        raise QuartileSpecialistWorkflowError(
            str(exc),
            failure_domain="provenance",
            failure_code="quartile_specialist_parent_provenance_changed",
        ) from exc


def _write_parent_immutability_after(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    current = _verify_frozen_parents(run_dir, args)
    record = _parent_immutability_record(current, phase="after")
    atomic_write_json(run_dir / "parent_immutability_after.json", record)
    return record


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "preflight_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
        _verify_frozen_parents(run_dir, args)
        return _load_json(gate_path)
    parent = verify_quartile_specialist_parents(
        time_local_run_dir=args.parent_time_local_run_dir,
        memory_v3_run_dir=args.parent_memory_v3_run_dir,
        coarse_witness_run_dir=args.parent_coarse_witness_run_dir,
        bayes_power_run_dir=args.parent_bayes_power_run_dir,
    )
    atomic_write_json(run_dir / "parent_provenance.json", parent)
    atomic_write_json(
        run_dir / "parent_immutability_before.json",
        _parent_immutability_record(parent, phase="before"),
    )
    atomic_write_json(
        run_dir / "source_closure.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-source-closure",
                "source_fingerprint": quartile_source_fingerprint(),
                "paths": [str(path) for path in quartile_source_paths()],
            }
        ),
    )
    path_validation = validate_path_id_plan(build_path_id_plan())
    cohort_validation = validate_cohort_plan(build_cohort_plan())
    seed_validation = validate_seed_plan(build_seed_plan())
    config_validation = validate_semantic_config(
        _load_json(run_dir / "scientific_config.json"),
        expected_schema=RUN_SCHEMA + "-scientific-config",
    )
    role_validation = validate_role_open_order(())
    atomic_write_json(run_dir / "path_plan_validation.json", path_validation)
    atomic_write_json(run_dir / "cohort_plan_validation.json", cohort_validation)
    atomic_write_json(run_dir / "seed_plan_validation.json", seed_validation)
    atomic_write_json(run_dir / "scientific_contract_validation.json", config_validation)
    atomic_write_json(run_dir / "role_firewall_validation.json", role_validation)

    replicates, shard_size = _effective_bootstrap(args)
    audit_count = _effective_counts(args)["fresh_selection"]
    count_indexes = []
    for role, seed, namespace in (
        ("selection", SELECTION_BOOTSTRAP_SEED, SELECTION_NAMESPACE),
        ("confirmation", CONFIRMATION_BOOTSTRAP_SEED, CONFIRMATION_NAMESPACE),
    ):
        directory = run_dir / "bootstrap_counts" / role
        records = prepare_bootstrap_count_shards(
            directory,
            seed=seed,
            namespace=namespace,
            path_count=audit_count,
            replicates=replicates,
            shard_size=shard_size,
        )
        index = count_shard_index_record(records, role=role)
        atomic_write_json(run_dir / f"{role}_bootstrap_count_index.json", index)
        count_indexes.append(index)

    source = _source_target(args.parent_coarse_witness_run_dir)
    seam_root = run_dir / "preflight_seam_cache"
    seam = _generate_role_cache(
        seam_root,
        source,
        _effective_paths(args, "preflight_seam"),
        args=args,
        progress_name="preflight-seam",
    )
    seam_metrics = seam["result"]["metrics"]
    seam_inputs, seam_index = load_eager_role_inputs(seam_root, "train")
    atomic_write_json(
        run_dir / "preflight_seam_binding.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-preflight-seam-binding",
                "schema_version": 1,
                "cache_root": str(seam_root),
                "path_ids": list(_effective_paths(args, "preflight_seam")),
                "input_row_count": int(len(seam_inputs["sample_key"])),
                "role_index_semantic_sha256": seam_index["semantic_sha256"],
                "metrics_semantic_sha256": seam_metrics["semantic_sha256"],
                "physical_labels_opened": 0,
            }
        ),
    )
    seam_health = _exact_execution_health(seam_metrics)
    transitions = int(seam_metrics["transition_count"])
    elapsed = float(seam_metrics["complete_pipeline_elapsed_seconds"])
    rate = transitions / max(elapsed, np.finfo(float).tiny)
    projected_seconds = (
        1_905_082_368 / rate if not args.test_only else elapsed
    )
    bootstrap_bytes = sum(
        path.stat().st_size for path in (run_dir / "bootstrap_counts").rglob("*.npz")
    )
    projected_bytes = 1_460_000_000 + bootstrap_bytes
    resource = _semantic(
        {
            "schema": RUN_SCHEMA + "-resource-projection",
            "measured_seam_transition_count": transitions,
            "measured_seam_elapsed_seconds": elapsed,
            "measured_transitions_per_second": rate,
            "projected_total_transition_count": 1_905_082_368,
            "projected_exact_capture_seconds": projected_seconds,
            "projected_exact_capture_hours": projected_seconds / 3600.0,
            "maximum_projected_hours": MAXIMUM_PROJECTED_HOURS,
            "projected_persisted_bytes": projected_bytes,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "maximum_peak_memory_fraction": seam_health["peak_memory_fraction"],
            "fallback_fraction": seam_health["fallback_fraction"],
            "fallback_time_fraction": seam_health["fallback_time_fraction"],
            "test_only": int(args.test_only),
        }
    )
    atomic_write_json(run_dir / "resource_projection.json", resource)
    preflight_metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-preflight-metrics",
            "evaluation_status": "evaluated",
            **{name: 1 for name in PREFLIGHT_FLAGS},
            "parent_provenance_valid": int(parent.get("passed", 0)),
            "scientific_contract_valid": int(config_validation.get("passed", 0)),
            "path_plan_valid": int(path_validation.get("passed", 0)),
            "seed_plan_valid": int(seed_validation.get("passed", 0)),
            "cohort_plan_valid": int(cohort_validation.get("passed", 0)),
            "role_firewall_valid": int(role_validation.get("passed", 0)),
            "bootstrap_count_plans_sealed": int(
                len(count_indexes) == 2
                and (
                    args.test_only
                    or all(
                        int(value.get("sealed_before_physical_labels", 0)) == 1
                        for value in count_indexes
                    )
                )
            ),
            "exact_backend_seam_valid": int(
                _metric_fraction(
                    seam_metrics, "certified_count", "transition_count"
                )
                == 1.0
                and int(seam_metrics.get("forbidden_event_count", 0)) == 0
                and (
                    args.test_only
                    or float(seam_metrics.get("maximum_mass_error", math.inf))
                    <= 2.0e-12
                )
            ),
            "resource_projection_valid": int(
                args.test_only
                or (
                    projected_seconds <= MAXIMUM_PROJECTED_HOURS * 3600.0
                    and projected_bytes <= MAXIMUM_PERSISTED_BYTES
                    and int(seam_health["resource_valid"]) == 1
                )
            ),
            "physical_labels_opened": 0,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "preflight_metrics.json", preflight_metrics)
    gate = evaluate_preflight_gate(preflight_metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "run_manifest.json",
            "scientific_config.json",
            "path_id_plan.json",
            "cohort_plan.json",
            "seed_plan.json",
            "role_firewall.json",
            "candidate_grid_plan.json",
            "gain_calibration_plan.json",
            "training_rank_plan.json",
            "selection_inference_plan.json",
            "confirmation_inference_plan.json",
            "confirmation_plan.json",
            "parent_provenance.json",
            "parent_immutability_before.json",
            "source_closure.json",
            "path_plan_validation.json",
            "cohort_plan_validation.json",
            "seed_plan_validation.json",
            "scientific_contract_validation.json",
            "role_firewall_validation.json",
            "selection_bootstrap_count_index.json",
            "confirmation_bootstrap_count_index.json",
            "preflight_seam_binding.json",
            "resource_projection.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )
    return gate


def _cache_root(run_dir: Path, role: str) -> Path:
    return run_dir / "role_caches" / role


def _cache_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("cache requires a passing preflight")
    _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
    gate_path = run_dir / "cache_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "cache_artifact_seal.json")
        for role in ("physical_fit", "gain_calibration", "training_rank"):
            load_eager_role_inputs(_cache_root(run_dir, role), "train")
        return _load_json(gate_path)
    source = _source_target(args.parent_coarse_witness_run_dir)
    role_results: dict[str, dict[str, Any]] = {}
    indexes: dict[str, dict[str, Any]] = {}
    total_persisted = 0
    for role in ("physical_fit", "gain_calibration", "training_rank"):
        result = _generate_role_cache(
            _cache_root(run_dir, role),
            source,
            _effective_paths(args, role),
            args=args,
            progress_name=role,
        )
        arrays, index = load_eager_role_inputs(_cache_root(run_dir, role), "train")
        role_results[role] = dict(result["result"]["metrics"])
        indexes[role] = dict(index)
        total_persisted += int(result["result"]["metrics"].get("persisted_bytes", 0))
        atomic_write_json(
            run_dir / f"{role}_cache_binding.json",
            _semantic(
                {
                    "schema": RUN_SCHEMA + "-role-cache-binding",
                    "role": role,
                    "cache_root": str(_cache_root(run_dir, role)),
                    "path_ids": list(_effective_paths(args, role)),
                    "input_row_count": int(len(arrays["sample_key"])),
                    "role_index_semantic_sha256": index["semantic_sha256"],
                    "metrics_semantic_sha256": result["result"]["metrics"][
                        "semantic_sha256"
                    ],
                    "physical_labels_opened": 0,
                }
            ),
        )
    cache_bindings = []
    for role in ("physical_fit", "gain_calibration", "training_rank"):
        binding_path = run_dir / f"{role}_cache_binding.json"
        binding = _load_json(binding_path)
        _verify_semantic(binding, binding_path.name)
        cache_bindings.append(
            {
                "role": role,
                "binding_path": binding_path.name,
                "binding_file_sha256": file_fingerprint(binding_path),
                "binding_semantic_sha256": binding["semantic_sha256"],
                "role_index_semantic_sha256": indexes[role]["semantic_sha256"],
            }
        )
    atomic_write_json(
        run_dir / "immutable_cache_binding.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-immutable-cache-binding",
                "schema_version": 1,
                "roles": cache_bindings,
                "path_id_plan_sha256": build_path_id_plan()["semantic_sha256"],
                "physical_labels_opened": 0,
            }
        ),
    )
    expected_rows = {
        role: len(_effective_paths(args, role))
        * len(_effective_selected_steps(args))
        * PHASE_COUNT
        * MIDPOINT_COUNT
        for role in role_results
    }
    expected_transitions = {
        role: len(_effective_paths(args, role))
        * (
            _effective_outer_steps(args) * PHASE_COUNT * EDGES_PER_PHASE
            + len(_effective_selected_steps(args))
            * PHASE_COUNT
            * MIDPOINT_COUNT
            * EDGES_PER_PHASE
        )
        for role in role_results
    }
    cache_valid = {}
    cache_health: dict[str, dict[str, Any]] = {}
    for role, metrics in role_results.items():
        arrays, _ = load_eager_role_inputs(_cache_root(run_dir, role), "train")
        health = _exact_execution_health(metrics)
        cache_health[role] = health
        cache_valid[role] = int(
            len(arrays["sample_key"]) == expected_rows[role]
            and int(metrics["transition_count"]) == expected_transitions[role]
            and int(metrics["certified_count"]) == expected_transitions[role]
            and int(metrics.get("forbidden_event_count", 0)) == 0
            and float(metrics.get("maximum_mass_error", 0.0)) <= 2.0e-12
            and (args.test_only or int(health["resource_valid"]) == 1)
        )
    persistence_valid = bool(args.test_only or total_persisted <= MAXIMUM_PERSISTED_BYTES)
    cache_metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-cache-metrics",
            "evaluation_status": "evaluated",
            **{name: 1 for name in CACHE_FLAGS},
            "fit_cache_valid": cache_valid["physical_fit"],
            "gain_cache_valid": cache_valid["gain_calibration"],
            "rank_cache_valid": cache_valid["training_rank"],
            "path_row_transition_counts_valid": int(all(cache_valid.values())),
            "cache_hashes_valid": int(persistence_valid),
            "cache_role_isolation_valid": 1,
            "all_labels_unopened": int(
                not any(run_dir.glob("*_label_open.json"))
            ),
            "selection_confirmation_evidence_absent": int(
                not (run_dir / "selection_open.json").exists()
                and not (run_dir / "confirmation_open.json").exists()
            ),
            "role_row_counts": expected_rows,
            "role_transition_counts": expected_transitions,
            "total_transition_count": sum(expected_transitions.values()),
            "total_persisted_bytes": total_persisted,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "role_execution_health": cache_health,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "cache_metrics.json", cache_metrics)
    gate = evaluate_cache_gate(cache_metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "immutable_cache_binding.json",
            "physical_fit_cache_binding.json",
            "gain_calibration_cache_binding.json",
            "training_rank_cache_binding.json",
            "cache_metrics.json",
            "cache_gate.json",
        ),
        "cache_artifact_seal.json",
    )
    return gate


def _store_quartiles(store: HostInputStore) -> np.ndarray:
    reverse_time = torch.as_tensor(
        np.array(store.row_array("reverse_time"), dtype=np.float64, copy=True)
    )
    phase = torch.as_tensor(
        np.array(store.row_array("phase"), dtype=np.int64, copy=True)
    )
    from mnist.d0_jacobi_rb_reverse_controller import fractional_coordinate

    values = fractional_coordinate(reverse_time, phase).forward_outer_quartile.numpy()
    return np.ascontiguousarray(values, dtype=np.int64)


def _quartile_rows(store: HostInputStore, quartile: int) -> np.ndarray:
    return np.flatnonzero(_store_quartiles(store) == int(quartile)).astype(np.int64)


def _predict_rows(
    model: nn.Module,
    store: HostInputStore,
    rows: np.ndarray,
    *,
    device: torch.device,
    guard: ModelCallBatchGuard,
) -> np.ndarray:
    output = np.empty((len(rows), EDGES_PER_PHASE), dtype=np.float64)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), 32):
                active = rows[start : start + 32]
                prediction = guard.call(model, store.batch(active, device=device))
                output[start : start + len(active)] = (
                    prediction.detach().to(torch.float64).cpu().numpy()
                )
    finally:
        model.train(was_training)
    if not np.isfinite(output).all():
        raise QuartileSpecialistWorkflowError(
            "streamed prediction is nonfinite",
            failure_domain="optimization_control",
            failure_code="quartile_prediction_nonfinite",
        )
    return output


def _synthetic_targets(
    store: HostInputStore,
    rows: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    result = np.empty((len(rows), EDGES_PER_PHASE), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            active = rows[start : start + 32]
            result[start : start + len(active)] = (
                synthetic_tangent_target(store.batch(active, device=device))
                .detach()
                .to(torch.float64)
                .cpu()
                .numpy()
            )
    return result


def _train_synthetic_quartile(
    run_dir: Path,
    *,
    quartile: int,
    seed: int,
    train: HostInputStore,
    validation: HostInputStore,
    maximum_updates: int,
    device: torch.device,
    guard: ModelCallBatchGuard,
) -> dict[str, Any]:
    train_rows = _quartile_rows(train, quartile)
    validation_rows = _quartile_rows(validation, quartile)
    train_target = _synthetic_targets(train, train_rows, device=device)
    validation_target = _synthetic_targets(validation, validation_rows, device=device)
    squared = math.fsum(float(value) ** 2 for value in train_target.flat)
    scale = math.sqrt(squared / train_target.size)
    enable_deterministic_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    for update in range(maximum_updates):
        local = deterministic_batch_indices(len(train_rows), 32, update, int(seed))
        rows = train_rows[local]
        inputs = train.batch(rows, device=device)
        target = torch.as_tensor(
            np.array(train_target[local], copy=True, order="C"),
            dtype=torch.float64,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = guard.call(model, inputs)
        loss, _ = scaled_raw_target_mse(prediction, target, scale)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not math.isfinite(float(gradient)):
            raise QuartileSpecialistWorkflowError(
                "synthetic control optimization became nonfinite",
                failure_domain="optimization_control",
                failure_code="quartile_synthetic_nonfinite",
            )
        optimizer.step()
    prediction = _predict_rows(
        model, validation, validation_rows, device=device, guard=guard
    )
    residual = np.mean((validation_target - prediction) ** 2, axis=1)
    zero = np.mean(validation_target**2, axis=1)
    relative = float(np.mean(residual) / np.mean(zero))
    path_ids = np.asarray(validation.row_array("path_id"), dtype=np.int64)[
        validation_rows
    ]
    per_path = [
        float(np.mean(zero[path_ids == path]) - np.mean(residual[path_ids == path]))
        for path in np.unique(path_ids)
    ]
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-synthetic-quartile-control",
            "quartile": quartile,
            "seed": seed,
            "relative_validation_mse": relative,
            "every_validation_path_beats_zero": int(all(value > 0.0 for value in per_path)),
            "validation_path_count": len(per_path),
            "passed": int(relative <= 0.01 and all(value > 0.0 for value in per_path)),
            "maximum_updates": maximum_updates,
            "physical_labels_opened": 0,
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / f"synthetic_teacher_q{quartile}.json", record)
    return record


def _exact_null_control(device: torch.device, quartile: int) -> dict[str, Any]:
    # The null is deterministic and uses the one frozen null seed for every
    # quartile; 261357 remains reserved exactly as recorded in the seed plan.
    seed = EXACT_NULL_CONTROL_ROOT_SEED
    torch.manual_seed(seed)
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    before = _clone_state_dict(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    state = torch.full((4, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float32, device=device)
    phase = torch.arange(4, device=device, dtype=torch.long) % PHASE_COUNT
    step = quartile * 128 + 63
    inputs = ModelInputs(
        later_full_state=state,
        reverse_time=torch.as_tensor(
            [internal_reverse_time(step, int(value), 0.5) for value in phase.tolist()],
            dtype=torch.float64,
            device=device,
        ),
        phase=phase,
        color=torch.as_tensor(
            [PHASE_MATCHINGS[int(value)] for value in phase.tolist()],
            dtype=torch.long,
            device=device,
        ),
        duration=torch.as_tensor(
            [PHASE_DURATIONS[int(value)] for value in phase.tolist()],
            dtype=torch.float32,
            device=device,
        ),
        label=torch.full((4,), 3, dtype=torch.long, device=device),
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = torch.mean(prediction.square())
    loss.backward()
    exact_gradients = all(
        parameter.grad is not None and bool(torch.all(parameter.grad == 0.0))
        for parameter in model.parameters()
    )
    optimizer.step()
    after = _clone_state_dict(model)
    unchanged = all(torch.equal(before[name], after[name]) for name in before)
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-exact-null-control",
            "quartile": quartile,
            "seed": seed,
            "loss": float(loss.detach().cpu()),
            "gradients_exact_zero": int(exact_gradients),
            "parameters_bitwise_unchanged": int(unchanged),
            "selected_update": 0,
            "passed": int(float(loss) == 0.0 and exact_gradients and unchanged),
            "physical_labels_opened": 0,
            **NO_WORK,
        }
    )


def _controls_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "cache_gate.json")):
        raise ArtifactCompatibilityError("controls require a passing cache gate")
    _verify_stage_seal(run_dir, "cache_artifact_seal.json")
    gate_path = run_dir / "controls_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "controls_artifact_seal.json")
        return _load_json(gate_path)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fit = open_external_input_store(_cache_root(run_dir, "physical_fit"), "train")
    gain = open_external_input_store(_cache_root(run_dir, "gain_calibration"), "train")
    rank = open_external_input_store(_cache_root(run_dir, "training_rank"), "train")
    guard = ModelCallBatchGuard()

    zero_valid = True
    zero_hashes = []
    for quartile, seeds in enumerate(MODEL_SEEDS_BY_QUARTILE):
        rows_by_store = [
            (_quartile_rows(store, quartile), store) for store in (fit, gain, rank)
        ]
        for seed in seeds:
            torch.manual_seed(seed)
            model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
            zero_hashes.append(state_dict_sha256(_clone_state_dict(model)))
            for rows, store in rows_by_store:
                prediction = _predict_rows(model, store, rows, device=device, guard=guard)
                zero_valid = zero_valid and bool(np.all(prediction == 0.0))

    mixed_indices = np.concatenate(
        [_quartile_rows(fit, quartile)[:2] for quartile in range(4)]
    )
    mixed = fit.batch(mixed_indices, device=device)
    experts = [ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device) for _ in range(4)]
    dispatch_rows = [0, 0, 0, 0]
    with torch.no_grad():
        direct = torch.empty((len(mixed_indices), EDGES_PER_PHASE), dtype=torch.float64, device=device)
        quartiles = reconstruct_forward_outer_quartile(mixed)
        for quartile, expert in enumerate(experts):
            rows = torch.nonzero(quartiles == quartile, as_tuple=False).flatten()
            direct.index_copy_(0, rows, expert(mixed.index_select(rows)))
        hooks = [
            expert.register_forward_pre_hook(
                lambda _module, arguments, quartile=quartile: dispatch_rows.__setitem__(
                    quartile,
                    dispatch_rows[quartile] + int(arguments[0].batch_size),
                )
            )
            for quartile, expert in enumerate(experts)
        ]
        try:
            composite = QuartileSpecialistBoundaryTangentPredictor(experts).to(device)(mixed)
        finally:
            for hook in hooks:
                hook.remove()
    expected_dispatch_rows = [int(torch.sum(quartiles == value)) for value in range(4)]
    dispatch_valid = bool(
        torch.equal(direct, composite)
        and dispatch_rows == expected_dispatch_rows
        and sum(dispatch_rows) == len(mixed_indices)
    )

    maximum_updates = _effective_updates(args)
    synthetic = [
        _train_synthetic_quartile(
            run_dir,
            quartile=quartile,
            seed=SYNTHETIC_CONTROL_SEEDS[quartile],
            train=fit,
            validation=gain,
            maximum_updates=maximum_updates,
            device=device,
            guard=guard,
        )
        for quartile in range(4)
    ]
    nulls = [_exact_null_control(device, quartile) for quartile in range(4)]
    atomic_write_json(
        run_dir / "exact_model_null_control.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-exact-null-controls",
                "quartiles": nulls,
                "passed": int(all(int(value["passed"]) == 1 for value in nulls)),
                "physical_labels_opened": 0,
                **NO_WORK,
            }
        ),
    )
    gain_control = gain_record_from_moments(
        CandidateIdentity(2, MODEL_SEEDS_BY_QUARTILE[2][0], 100),
        cross_term=2.0,
        prediction_energy=4.0,
        sample_count=1,
    )
    gain_valid = bool(gain_control.eligible and gain_control.gain == 0.5)
    firewall_valid = False
    try:
        QuartileSpecialistBoundaryTangentPredictor(experts)({"outer_step": 0})  # type: ignore[arg-type]
    except Exception:
        firewall_valid = True
    peak = (
        torch.cuda.max_memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else 0.0
    )
    preflight_metrics = _load_json(run_dir / "preflight_metrics.json")
    _verify_semantic(preflight_metrics, "preflight_metrics.json")
    seam_binding = _load_json(run_dir / "preflight_seam_binding.json")
    _verify_semantic(seam_binding, "preflight_seam_binding.json")
    seam_inputs, seam_index = load_eager_role_inputs(
        run_dir / "preflight_seam_cache", "train"
    )
    source_backend_seam_valid = int(
        int(preflight_metrics.get("exact_backend_seam_valid", 0)) == 1
        and seam_binding.get("role_index_semantic_sha256")
        == seam_index.get("semantic_sha256")
        and int(seam_binding.get("input_row_count", -1))
        == len(seam_inputs["sample_key"])
        and _source_target(args.parent_coarse_witness_run_dir).shape == (STATE_SIZE,)
    )
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-controls-metrics",
            "evaluation_status": "evaluated",
            **{name: 1 for name in CONTROLS_FLAGS},
            "source_backend_seam_valid": source_backend_seam_valid,
            "zero_initialization_valid": int(zero_valid and len(zero_hashes) == 12),
            "quartile_dispatch_valid": int(dispatch_valid),
            "synthetic_teacher_valid": int(all(int(value["passed"]) == 1 for value in synthetic)),
            "gain_algebra_valid": int(gain_valid),
            "exact_model_null_valid": int(all(int(value["passed"]) == 1 for value in nulls)),
            "input_firewall_valid": int(firewall_valid),
            "resource_guard_valid": int(
                guard.record()["maximum_observed_batch_size"] <= 32
                and peak <= MAXIMUM_PEAK_MEMORY_FRACTION
            ),
            "physical_labels_opened_zero": int(
                not any(run_dir.glob("*_label_open.json"))
            ),
            "maximum_forward_batch": guard.record()["maximum_observed_batch_size"],
            "peak_memory_fraction": peak,
            "physical_labels_opened": 0,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "controls_metrics.json", metrics)
    atomic_write_json(
        run_dir / "zero_initialization_control.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-zero-initialization-control",
                "state_hashes": zero_hashes,
                "passed": int(zero_valid),
                "physical_labels_opened": 0,
            }
        ),
    )
    atomic_write_json(
        run_dir / "quartile_dispatch_control.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-dispatch-control",
                "rows_by_expert": dispatch_rows,
                "expected_rows_by_expert": expected_dispatch_rows,
                "each_row_dispatched_once": int(
                    dispatch_rows == expected_dispatch_rows
                    and sum(dispatch_rows) == len(mixed_indices)
                ),
                "passed": int(dispatch_valid),
                "physical_labels_opened": 0,
            }
        ),
    )
    atomic_write_json(run_dir / "gain_algebra_control.json", _semantic(gain_control.to_record()))
    gate = evaluate_controls_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "zero_initialization_control.json",
            "quartile_dispatch_control.json",
            "gain_algebra_control.json",
            "exact_model_null_control.json",
            *(f"synthetic_teacher_q{quartile}.json" for quartile in range(4)),
            "controls_metrics.json",
            "controls_gate.json",
        ),
        "controls_artifact_seal.json",
    )
    return gate


_OPEN_FILES = {
    "physical_fit": "fit_label_open.json",
    "gain_calibration": "gain_label_open.json",
    "training_rank": "rank_label_open.json",
    "fresh_selection": "selection_open.json",
    "untouched_confirmation": "confirmation_open.json",
}


def _opened_roles(run_dir: Path) -> tuple[str, ...]:
    return tuple(role for role in ROLE_OPEN_ORDER if (run_dir / _OPEN_FILES[role]).is_file())


def _open_role(run_dir: Path, role: str, *, prerequisites: Mapping[str, str]) -> dict[str, Any]:
    if role not in ROLE_OPEN_ORDER:
        raise ArtifactCompatibilityError(f"unknown evidence role: {role}")
    path = run_dir / _OPEN_FILES[role]
    if path.is_file():
        record = _load_json(path)
        _verify_semantic(record, path.name)
        expected_paths = list(build_path_id_plan()["roles"][role])
        if (
            record.get("schema") != RUN_SCHEMA + "-role-open"
            or record.get("role") != role
            or record.get("path_ids") != expected_paths
            or record.get("prerequisite_file_sha256") != dict(prerequisites)
            or int(record.get("replacement_role_authorized", -1)) != 0
        ):
            raise ArtifactCompatibilityError(
                f"existing role-open contract changed: {path.name}"
            )
        for name, sha256 in prerequisites.items():
            artifact = run_dir / name
            if not artifact.is_file() or file_fingerprint(artifact) != sha256:
                raise ArtifactCompatibilityError(
                    f"existing role-open prerequisite changed: {name}"
                )
        validate_role_open_order(_opened_roles(run_dir))
        return record
    expected_prefix = ROLE_OPEN_ORDER[: ROLE_OPEN_ORDER.index(role)]
    if _opened_roles(run_dir) != expected_prefix:
        raise ArtifactCompatibilityError("role-open prerequisite order changed")
    for name, sha256 in prerequisites.items():
        artifact = run_dir / name
        if not artifact.is_file() or file_fingerprint(artifact) != sha256:
            raise ArtifactCompatibilityError(f"role-open prerequisite changed: {name}")
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-role-open",
            "role": role,
            "opened_at": _now(),
            "path_ids": list(build_path_id_plan()["roles"][role]),
            "prerequisite_file_sha256": dict(prerequisites),
            "replacement_role_authorized": 0,
        }
    )
    atomic_write_json(path, record)
    validate_role_open_order(_opened_roles(run_dir))
    return record


def _load_logical_labels(
    run_dir: Path,
    role: str,
    *,
    prerequisites: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _open_role(run_dir, role, prerequisites=prerequisites)
    arrays, index = load_eager_role_labels(_cache_root(run_dir, role), "train")
    inputs, input_index = load_eager_role_inputs(_cache_root(run_dir, role), "train")
    if (
        input_index.get("semantic_sha256") != index.get("semantic_sha256")
        or not np.array_equal(inputs["sample_key"], arrays["sample_key"])
        or not np.array_equal(inputs["path_id"], arrays["path_id"])
    ):
        raise ArtifactCompatibilityError(f"{role} input/label join changed")
    return arrays, index


def _training_fingerprint(
    run_dir: Path,
    *,
    candidate: CandidateIdentity,
    target_scale: float,
    maximum_updates: int,
) -> str:
    return config_fingerprint(
        {
            "schema": RUN_SCHEMA + "-training-fingerprint",
            "candidate_quartile": candidate.quartile,
            "candidate_seed": candidate.seed,
            "target_scale": target_scale,
            "maximum_updates": maximum_updates,
            "training": TRAINING,
            "scientific_config_sha256": _load_json(run_dir / "scientific_config.json")[
                "semantic_sha256"
            ],
            "fit_cache_binding_sha256": _load_json(
                run_dir / "physical_fit_cache_binding.json"
            )["semantic_sha256"],
        }
    )


def _save_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    quartile: int,
    seed: int,
    update: int,
    fingerprint: str,
) -> dict[str, Any]:
    state = _clone_state_dict(model)
    state_hash = state_dict_sha256(state)
    path = (
        run_dir
        / "checkpoints"
        / f"q{quartile}"
        / f"seed-{seed}"
        / f"update-{update:04d}.pt"
    )
    artifact = _atomic_torch(
        path,
        {
            "schema": RUN_SCHEMA + "-checkpoint",
            "quartile": quartile,
            "seed": seed,
            "update": update,
            "training_fingerprint": fingerprint,
            "state_dict": state,
            "state_sha256": state_hash,
            "optimizer_state_dict": optimizer.state_dict(),
            "batch_cursor": update,
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_states": tuple(torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else (),
            "raw_target_unchanged": 1,
            "training_only": 1,
        },
    )
    return {
        "quartile": quartile,
        "seed": seed,
        "update": update,
        "training_fingerprint": fingerprint,
        "state_sha256": state_hash,
        "checkpoint_path": path.relative_to(run_dir).as_posix(),
        "checkpoint_file_sha256": artifact["sha256"],
    }


def _train_physical_trajectory(
    run_dir: Path,
    *,
    quartile: int,
    seed: int,
    store: HostInputStore,
    target: np.ndarray,
    target_scale: float,
    maximum_updates: int,
    device: torch.device,
    guard: ModelCallBatchGuard,
) -> dict[str, Any]:
    rows = _quartile_rows(store, quartile)
    fingerprint = _training_fingerprint(
        run_dir,
        candidate=CandidateIdentity(quartile, seed, 0),
        target_scale=target_scale,
        maximum_updates=maximum_updates,
    )
    progress_path = run_dir / "checkpoints" / f"q{quartile}" / f"seed-{seed}-progress.pt"
    enable_deterministic_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-3, weight_decay=0.0
    )
    completed = 0
    checkpoints: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    finite = True
    if progress_path.is_file():
        payload = torch.load(progress_path, map_location=device, weights_only=False)
        if payload.get("training_fingerprint") != fingerprint:
            raise ArtifactCompatibilityError("physical training fingerprint changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed = int(payload["completed_update"])
        checkpoints = [dict(value) for value in payload["checkpoints"]]
        history = [dict(value) for value in payload["history"]]
        finite = bool(payload["finite"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(list(payload["cuda_rng_states"]))

    def save_progress(update: int) -> None:
        _atomic_torch(
            progress_path,
            {
                "schema": RUN_SCHEMA + "-training-progress",
                "training_fingerprint": fingerprint,
                "completed_update": update,
                "model_state_dict": _clone_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "checkpoints": checkpoints,
                "history": history,
                "finite": int(finite),
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": tuple(torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else (),
            },
        )

    if not checkpoints:
        checkpoint = _save_checkpoint(
            run_dir,
            model,
            optimizer,
            quartile=quartile,
            seed=seed,
            update=0,
            fingerprint=fingerprint,
        )
        checkpoints.append(checkpoint)
        save_progress(0)
    checkpoint_updates = set(CHECKPOINT_UPDATES)
    if maximum_updates not in checkpoint_updates:
        checkpoint_updates.add(maximum_updates)
    for update in range(completed + 1, maximum_updates + 1):
        local = deterministic_batch_indices(len(rows), 32, update - 1, seed)
        active = rows[local]
        inputs = store.batch(active, device=device)
        batch_target = torch.as_tensor(
            np.array(target[active], copy=True, order="C"),
            dtype=torch.float64,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = guard.call(model, inputs)
        loss, raw_mse = scaled_raw_target_mse(prediction, batch_target, target_scale)
        if not bool(torch.isfinite(loss)):
            finite = False
            save_progress(update - 1)
            break
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gradient)):
            finite = False
            save_progress(update - 1)
            break
        optimizer.step()
        if update in checkpoint_updates:
            checkpoint = _save_checkpoint(
                run_dir,
                model,
                optimizer,
                quartile=quartile,
                seed=seed,
                update=update,
                fingerprint=fingerprint,
            )
            checkpoints.append(checkpoint)
            history.append(
                {
                    "quartile": quartile,
                    "seed": seed,
                    "update": update,
                    "raw_mse": float(raw_mse.detach().cpu()),
                    "scaled_loss": float(loss.detach().cpu()),
                    "preclip_gradient_norm": float(gradient),
                    "state_sha256": checkpoint["state_sha256"],
                }
            )
            save_progress(update)
            print(
                f"quartile q{quartile} seed={seed} update={update}/{maximum_updates} "
                f"mse={float(raw_mse.detach().cpu()):.8g}",
                flush=True,
            )
    expected_updates = {0, maximum_updates}
    expected_updates.update(
        update for update in CHECKPOINT_UPDATES if update <= maximum_updates
    )
    expected = len(expected_updates)
    complete = bool(
        finite
        and len(checkpoints) == expected
        and checkpoints[-1]["update"] == maximum_updates
    )
    report = _semantic(
        {
            "schema": RUN_SCHEMA + "-training-trajectory",
            "quartile": quartile,
            "seed": seed,
            "complete": int(complete),
            "finite": int(finite),
            "checkpoint_count": len(checkpoints),
            "checkpoints": checkpoints,
            "target_scale": target_scale,
            "maximum_updates": maximum_updates,
            "training_fingerprint": fingerprint,
            "gain_labels_opened": int((run_dir / "gain_label_open.json").exists()),
            "rank_labels_opened": int((run_dir / "rank_label_open.json").exists()),
            "selection_paths_opened": int((run_dir / "selection_open.json").exists()),
            "confirmation_paths_opened": int((run_dir / "confirmation_open.json").exists()),
            "physical_training_performed": 1,
            **NO_WORK,
        }
    )
    atomic_write_json(
        run_dir / "checkpoints" / f"q{quartile}" / f"seed-{seed}-task.json",
        report,
    )
    atomic_write_csv(
        run_dir / "checkpoints" / f"q{quartile}" / f"seed-{seed}-history.csv",
        history,
    )
    return report


def _write_training_checkpoint_index(
    run_dir: Path, reports: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Bind every candidate-bearing task and exact boundary checkpoint."""

    tasks: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for report in sorted(
        (dict(value) for value in reports),
        key=lambda value: (int(value["quartile"]), int(value["seed"])),
    ):
        quartile = int(report["quartile"])
        seed = int(report["seed"])
        task_path = (
            run_dir / "checkpoints" / f"q{quartile}" / f"seed-{seed}-task.json"
        )
        history_path = (
            run_dir / "checkpoints" / f"q{quartile}" / f"seed-{seed}-history.csv"
        )
        progress_path = (
            run_dir / "checkpoints" / f"q{quartile}" / f"seed-{seed}-progress.pt"
        )
        for path in (task_path, history_path, progress_path):
            if not path.is_file():
                raise ArtifactCompatibilityError(
                    f"training task artifact is absent: {path}"
                )
        task_record = _load_json(task_path)
        _verify_semantic(task_record, task_path.name)
        task_checkpoints = [dict(value) for value in task_record["checkpoints"]]
        if task_checkpoints != [dict(value) for value in report["checkpoints"]]:
            raise ArtifactCompatibilityError("training task checkpoint table changed")
        tasks.append(
            {
                "quartile": quartile,
                "seed": seed,
                "task_path": task_path.relative_to(run_dir).as_posix(),
                "task_sha256": file_fingerprint(task_path),
                "history_path": history_path.relative_to(run_dir).as_posix(),
                "history_sha256": file_fingerprint(history_path),
                "progress_path": progress_path.relative_to(run_dir).as_posix(),
                "progress_sha256": file_fingerprint(progress_path),
            }
        )
        for row in task_checkpoints:
            checkpoint_path = run_dir / str(row["checkpoint_path"])
            measured = file_fingerprint(checkpoint_path)
            if measured != row["checkpoint_file_sha256"]:
                raise ArtifactCompatibilityError("training checkpoint changed")
            checkpoints.append(
                {
                    "candidate_key": CandidateIdentity(
                        quartile, seed, int(row["update"])
                    ).key,
                    "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_file_sha256": measured,
                    "model_state_sha256": row["state_sha256"],
                }
            )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-training-checkpoint-index",
            "schema_version": 1,
            "task_count": len(tasks),
            "checkpoint_count": len(checkpoints),
            "tasks": tasks,
            "checkpoints": checkpoints,
            "all_boundary_checkpoints_exactly_resumable": 1,
        }
    )
    atomic_write_json(run_dir / "training_checkpoint_index.json", record)
    return record


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "controls_gate.json")):
        raise ArtifactCompatibilityError("train requires passing prelabel controls")
    _verify_stage_seal(run_dir, "controls_artifact_seal.json")
    gate_path = run_dir / "train_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "train_artifact_seal.json")
        return _load_json(gate_path)
    controls_hash = file_fingerprint(run_dir / "controls_artifact_seal.json")
    labels, _ = _load_logical_labels(
        run_dir,
        "physical_fit",
        prerequisites={"controls_artifact_seal.json": controls_hash},
    )
    target = np.array(labels["denoising_target"], dtype=np.float64, copy=True, order="C")
    store = open_external_input_store(_cache_root(run_dir, "physical_fit"), "train")
    quartiles = _store_quartiles(store)
    scales = [
        exact_quartile_target_scale(target, quartiles, quartile)
        for quartile in range(4)
    ]
    for quartile, scale in enumerate(scales):
        atomic_write_json(
            run_dir / f"q{quartile}_target_scale.json",
            _semantic(
                {
                    "schema": RUN_SCHEMA + "-quartile-target-scale",
                    "quartile": quartile,
                    "target_scale": scale,
                    "role": "physical_fit",
                    "training_only": 1,
                    "raw_target_unchanged": 1,
                }
            ),
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    guard = ModelCallBatchGuard()
    maximum_updates = _effective_updates(args)
    reports = []
    for quartile, seeds in enumerate(MODEL_SEEDS_BY_QUARTILE):
        for seed in seeds:
            reports.append(
                _train_physical_trajectory(
                    run_dir,
                    quartile=quartile,
                    seed=seed,
                    store=store,
                    target=target,
                    target_scale=scales[quartile],
                    maximum_updates=maximum_updates,
                    device=device,
                    guard=guard,
                )
            )
    checkpoint_count = sum(int(value["checkpoint_count"]) for value in reports)
    expected_per_trajectory = len(
        {0, maximum_updates}
        | {update for update in CHECKPOINT_UPDATES if update <= maximum_updates}
    )
    expected_count = 12 * expected_per_trajectory
    checkpoint_index = _write_training_checkpoint_index(run_dir, reports)
    peak = (
        torch.cuda.max_memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else 0.0
    )
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-train-metrics",
            "evaluation_status": "evaluated",
            **{name: 1 for name in TRAIN_FLAGS},
            "fit_labels_only": int(_opened_roles(run_dir) == ("physical_fit",)),
            "target_scales_training_only": int(all(value > 0.0 for value in scales)),
            "twelve_trajectories_complete": int(
                len(reports) == 12 and all(int(value["complete"]) == 1 for value in reports)
            ),
            "four_hundred_ninety_two_checkpoints_complete": int(
                checkpoint_count == expected_count
            ),
            "finite_outputs": int(all(int(value["finite"]) == 1 for value in reports)),
            "batch_limit_valid": int(guard.record()["maximum_observed_batch_size"] <= 32),
            "memory_limit_valid": int(peak <= MAXIMUM_PEAK_MEMORY_FRACTION),
            "downstream_labels_unopened": int(
                not (run_dir / "gain_label_open.json").exists()
                and not (run_dir / "rank_label_open.json").exists()
                and not (run_dir / "selection_open.json").exists()
                and not (run_dir / "confirmation_open.json").exists()
            ),
            "checkpoint_count": checkpoint_count,
            "checkpoint_index_sha256": checkpoint_index["semantic_sha256"],
            "target_scales": scales,
            "peak_memory_fraction": peak,
            "physical_training_performed": 1,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "train_metrics.json", metrics)
    atomic_write_csv(
        run_dir / "training_trajectory_summary.csv",
        [
            {
                "quartile": value["quartile"],
                "seed": value["seed"],
                "complete": value["complete"],
                "finite": value["finite"],
                "checkpoint_count": value["checkpoint_count"],
                "target_scale": value["target_scale"],
            }
            for value in reports
        ],
    )
    gate = evaluate_train_gate(metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (
            "fit_label_open.json",
            *(f"q{quartile}_target_scale.json" for quartile in range(4)),
            "training_trajectory_summary.csv",
            "training_checkpoint_index.json",
            "train_metrics.json",
            "train_gate.json",
        ),
        "train_artifact_seal.json",
    )
    return gate


def _candidate_checkpoint(run_dir: Path, candidate: CandidateIdentity) -> dict[str, Any]:
    _verify_stage_seal(run_dir, "train_artifact_seal.json")
    index = _load_json(run_dir / "training_checkpoint_index.json")
    _verify_semantic(index, "training_checkpoint_index.json")
    task_path = (
        run_dir
        / "checkpoints"
        / f"q{candidate.quartile}"
        / f"seed-{candidate.seed}-task.json"
    )
    task_entries = [
        row
        for row in index.get("tasks", [])
        if int(row.get("quartile", -1)) == candidate.quartile
        and int(row.get("seed", -1)) == candidate.seed
    ]
    if (
        len(task_entries) != 1
        or task_entries[0].get("task_path") != task_path.relative_to(run_dir).as_posix()
        or task_entries[0].get("task_sha256") != file_fingerprint(task_path)
    ):
        raise ArtifactCompatibilityError("training task index changed")
    task = _load_json(task_path)
    _verify_semantic(task, task_path.name)
    matches = [
        dict(value)
        for value in task.get("checkpoints", [])
        if int(value.get("update", -1)) == candidate.update
    ]
    if len(matches) != 1:
        raise ArtifactCompatibilityError(f"candidate checkpoint is absent: {candidate.key}")
    row = matches[0]
    path = run_dir / row["checkpoint_path"]
    index_matches = [
        value
        for value in index.get("checkpoints", [])
        if value.get("candidate_key") == candidate.key
    ]
    if (
        len(index_matches) != 1
        or index_matches[0].get("checkpoint_path") != row["checkpoint_path"]
        or index_matches[0].get("checkpoint_file_sha256")
        != row["checkpoint_file_sha256"]
        or file_fingerprint(path) != row["checkpoint_file_sha256"]
    ):
        raise ArtifactCompatibilityError(f"candidate checkpoint changed: {candidate.key}")
    return row


def _load_expert(
    run_dir: Path, candidate: CandidateIdentity, device: torch.device
) -> ZeroBaselineBoundaryTangentPredictor:
    row = _candidate_checkpoint(run_dir, candidate)
    payload = torch.load(
        run_dir / row["checkpoint_path"], map_location=device, weights_only=False
    )
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if state_dict_sha256(_clone_state_dict(model)) != row["state_sha256"]:
        raise ArtifactCompatibilityError(f"loaded state changed: {candidate.key}")
    model.eval()
    return model


def _available_nonzero_candidates(run_dir: Path) -> tuple[CandidateIdentity, ...]:
    manifest = _load_json(run_dir / "run_manifest.json")
    test_only = int(manifest.get("test_only", 0)) == 1
    maximum_update = 4_000
    if test_only:
        task_paths = sorted((run_dir / "checkpoints").glob("q*/seed-*-task.json"))
        if not task_paths:
            raise ArtifactCompatibilityError("test training tasks are absent")
        maxima = {
            int(_load_json(path).get("maximum_updates", -1)) for path in task_paths
        }
        if len(maxima) != 1:
            raise ArtifactCompatibilityError("test training update grid changed")
        maximum_update = maxima.pop()
    candidates = tuple(
        value for value in NONZERO_CANDIDATE_IDENTITIES if value.update <= maximum_update
    )
    missing: list[str] = []
    for candidate in candidates:
        path = (
            run_dir
            / "checkpoints"
            / f"q{candidate.quartile}"
            / f"seed-{candidate.seed}"
            / f"update-{candidate.update:04d}.pt"
        )
        if not path.is_file():
            missing.append(candidate.key)
    if missing:
        raise ArtifactCompatibilityError(
            "physical candidate grid is incomplete: " + ",".join(missing[:8])
        )
    return candidates


def _verify_complete_training_checkpoint_index(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Verify every one of the frozen 492 boundary checkpoints before labels open."""

    _verify_stage_seal(run_dir, "train_artifact_seal.json")
    index = _load_json(run_dir / "training_checkpoint_index.json")
    _verify_semantic(index, "training_checkpoint_index.json")
    maximum_update = _effective_updates(args)
    expected_updates = (0,) + tuple(
        value for value in CHECKPOINT_UPDATES if value <= maximum_update
    )
    expected = {
        CandidateIdentity(quartile, seed, update).key
        for quartile, seeds in enumerate(MODEL_SEEDS_BY_QUARTILE)
        for seed in seeds
        for update in expected_updates
    }
    rows = index.get("checkpoints")
    tasks = index.get("tasks")
    if (
        not isinstance(rows, list)
        or not isinstance(tasks, list)
        or int(index.get("checkpoint_count", -1)) != len(expected)
        or int(index.get("task_count", -1)) != 12
        or len(rows) != len(expected)
        or len(tasks) != 12
    ):
        raise ArtifactCompatibilityError("training checkpoint index is incomplete")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArtifactCompatibilityError("training checkpoint row is malformed")
        key = str(row.get("candidate_key", ""))
        path = run_dir / str(row.get("checkpoint_path", ""))
        expected_sha = row.get("checkpoint_file_sha256")
        if (
            key in observed
            or key not in expected
            or not path.is_file()
            or not isinstance(expected_sha, str)
            or file_fingerprint(path) != expected_sha
        ):
            raise ArtifactCompatibilityError("training checkpoint payload changed")
        observed.add(key)
    if observed != expected:
        raise ArtifactCompatibilityError("training checkpoint identity grid changed")
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ArtifactCompatibilityError("training task row is malformed")
        for path_field, hash_field in (
            ("task_path", "task_sha256"),
            ("history_path", "history_sha256"),
            ("progress_path", "progress_sha256"),
        ):
            path = run_dir / str(task.get(path_field, ""))
            if (
                not path.is_file()
                or file_fingerprint(path) != task.get(hash_field)
            ):
                raise ArtifactCompatibilityError("training task artifact changed")
    return index


def _candidate_gain_moments(
    model: nn.Module,
    store: HostInputStore,
    target: np.ndarray,
    rows: np.ndarray,
    *,
    device: torch.device,
    guard: ModelCallBatchGuard,
) -> tuple[float, float, int]:
    cross_parts: list[float] = []
    energy_parts: list[float] = []
    count = 0
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            active = rows[start : start + 32]
            prediction = (
                guard.call(model, store.batch(active, device=device))
                .detach()
                .to(torch.float64)
                .cpu()
                .numpy()
            )
            truth = target[active]
            cross_parts.extend(
                float(left) * float(right)
                for left, right in zip(truth.flat, prediction.flat, strict=True)
            )
            energy_parts.extend(float(value) ** 2 for value in prediction.flat)
            count += prediction.size
    return (
        math.fsum(cross_parts) / count,
        math.fsum(energy_parts) / count,
        count,
    )


def _rank_candidate(
    candidate: CandidateIdentity,
    model: nn.Module,
    gain_record: Any,
    store: HostInputStore,
    target: np.ndarray,
    *,
    device: torch.device,
    guard: ModelCallBatchGuard,
) -> tuple[Any, dict[str, np.ndarray]]:
    rows = _quartile_rows(store, candidate.quartile)
    # An ineligible q2/q3 calibration cannot rank.  We still compute a
    # diagnostic table with unit gain so the closed failure has readable
    # evidence; ``build_training_rank_record`` keeps it ineligible.
    gain = 1.0 if gain_record.gain is None else float(gain_record.gain)
    phase_values = np.asarray(store.row_array("phase"), dtype=np.int64)[rows]
    midpoint_values = np.asarray(store.row_array("midpoint_index"), dtype=np.int64)[rows]
    path_values = np.asarray(store.row_array("path_id"), dtype=np.int64)[rows]
    pooled_sum = 0.0
    pooled_count = 0
    cell_sum = np.zeros((PHASE_COUNT, MIDPOINT_COUNT), dtype=np.float64)
    cell_count = np.zeros((PHASE_COUNT, MIDPOINT_COUNT), dtype=np.int64)
    improvement_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            active = rows[start : start + 32]
            prediction = (
                guard.call(model, store.batch(active, device=device))
                .detach()
                .to(torch.float64)
                .cpu()
                .numpy()
            )
            truth = target[active]
            final = gain * prediction
            improvement = np.mean(
                truth * truth - (truth - final) ** 2, axis=1, dtype=np.float64
            )
            improvement_blocks.append(np.ascontiguousarray(improvement))
            pooled_sum += math.fsum(float(value) for value in improvement)
            pooled_count += len(improvement)
            local_phase = phase_values[start : start + len(active)]
            local_midpoint = midpoint_values[start : start + len(active)]
            np.add.at(cell_sum, (local_phase, local_midpoint), improvement)
            np.add.at(cell_count, (local_phase, local_midpoint), 1)
    if pooled_count == 0 or np.any(cell_count == 0):
        raise QuartileSpecialistWorkflowError(
            "training-rank cell is empty",
            failure_domain="training_rank",
            failure_code="quartile_rank_cell_empty",
        )
    cells = cell_sum / cell_count
    phase = np.sum(cell_sum, axis=1) / np.sum(cell_count, axis=1)
    midpoint = np.sum(cell_sum, axis=0) / np.sum(cell_count, axis=0)
    record = build_training_rank_record(
        candidate,
        gain_record,
        pooled_improvement=pooled_sum / pooled_count,
        phase_improvements=phase.tolist(),
        midpoint_improvements=midpoint.tolist(),
        fine_cell_improvements=cells.tolist(),
    )
    row_improvement = np.ascontiguousarray(np.concatenate(improvement_blocks))
    unique_paths = np.unique(path_values)
    path_pooled = np.empty(unique_paths.size, dtype=np.float64)
    path_cells = np.empty(
        (unique_paths.size, PHASE_COUNT, MIDPOINT_COUNT), dtype=np.float64
    )
    path_cell_counts = np.zeros_like(path_cells, dtype=np.int64)
    for path_index, path_id in enumerate(unique_paths):
        selected = path_values == path_id
        path_pooled[path_index] = math.fsum(
            float(value) for value in row_improvement[selected]
        ) / int(np.count_nonzero(selected))
        sums = np.zeros((PHASE_COUNT, MIDPOINT_COUNT), dtype=np.float64)
        np.add.at(
            sums,
            (phase_values[selected], midpoint_values[selected]),
            row_improvement[selected],
        )
        np.add.at(
            path_cell_counts[path_index],
            (phase_values[selected], midpoint_values[selected]),
            1,
        )
        if np.any(path_cell_counts[path_index] == 0):
            raise QuartileSpecialistWorkflowError(
                "training-rank path cell is empty",
                failure_domain="training_rank",
                failure_code="quartile_rank_path_cell_empty",
            )
        path_cells[path_index] = sums / path_cell_counts[path_index]
    return record, {
        "path_ids": np.ascontiguousarray(unique_paths, dtype=np.int64),
        "pooled_improvement": np.ascontiguousarray(path_pooled),
        "fine_cell_improvement": np.ascontiguousarray(path_cells),
        "fine_cell_count": np.ascontiguousarray(path_cell_counts),
    }


def _gain_records(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    store: HostInputStore,
    target: np.ndarray,
    device: torch.device,
    guard: ModelCallBatchGuard,
) -> dict[str, Any]:
    candidate_records: dict[str, Any] = {}
    rows_by_q = {quartile: _quartile_rows(store, quartile) for quartile in range(4)}
    for candidate in _available_nonzero_candidates(run_dir):
        if candidate.quartile in (0, 1):
            record = fixed_unit_gain_record(candidate)
        else:
            path = run_dir / "gain_records" / f"{candidate.key}.json"
            if path.is_file():
                saved = _load_json(path)
                _verify_semantic(saved, path.name)
                record = gain_record_from_moments(
                    candidate,
                    cross_term=float(saved["cross_term"]),
                    prediction_energy=float(saved["prediction_energy"]),
                    sample_count=int(saved["sample_count"]),
                )
                if record.to_record() != saved["gain_record"]:
                    raise ArtifactCompatibilityError("saved gain record changed")
            else:
                model = _load_expert(run_dir, candidate, device)
                cross, energy, count = _candidate_gain_moments(
                    model,
                    store,
                    target,
                    rows_by_q[candidate.quartile],
                    device=device,
                    guard=guard,
                )
                record = gain_record_from_moments(
                    candidate,
                    cross_term=cross,
                    prediction_energy=energy,
                    sample_count=count,
                )
                atomic_write_json(
                    path,
                    _semantic(
                        {
                            "schema": RUN_SCHEMA + "-candidate-gain",
                            "candidate_key": candidate.key,
                            "cross_term": cross,
                            "prediction_energy": energy,
                            "sample_count": count,
                            "gain_record": record.to_record(),
                            "gain_role_only": 1,
                            "rank_labels_opened": 0,
                        }
                    ),
                )
        candidate_records[candidate.key] = record
    return candidate_records


def _load_frozen_gain_records(run_dir: Path) -> dict[str, Any]:
    """Load a completed gain table without regenerating any frozen evidence."""

    _verify_stage_seal(run_dir, "gain_calibration_seal.json")
    candidates = _available_nonzero_candidates(run_dir)
    records: dict[str, Any] = {}
    for candidate in candidates:
        if candidate.quartile in (0, 1):
            records[candidate.key] = fixed_unit_gain_record(candidate)
            continue
        path = run_dir / "gain_records" / f"{candidate.key}.json"
        saved = _load_json(path)
        _verify_semantic(saved, path.name)
        record = gain_record_from_moments(
            candidate,
            cross_term=float(saved["cross_term"]),
            prediction_energy=float(saved["prediction_energy"]),
            sample_count=int(saved["sample_count"]),
        )
        if (
            saved.get("candidate_key") != candidate.key
            or saved.get("gain_record") != record.to_record()
            or int(saved.get("gain_role_only", 0)) != 1
            or int(saved.get("rank_labels_opened", -1)) != 0
        ):
            raise ArtifactCompatibilityError("frozen candidate gain record changed")
        records[candidate.key] = record

    table_path = run_dir / "gain_table.npz"
    with np.load(table_path, allow_pickle=False) as archive:
        table = {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    expected_identity = {
        "quartile": np.asarray([value.quartile for value in candidates], dtype=np.int8),
        "seed": np.asarray([value.seed for value in candidates], dtype=np.int64),
        "update": np.asarray([value.update for value in candidates], dtype=np.int16),
    }
    for name, expected in expected_identity.items():
        if name not in table or not np.array_equal(table[name], expected):
            raise ArtifactCompatibilityError("frozen gain-table identity grid changed")
    expected_numeric = {
        "cross_term": np.asarray(
            [
                np.nan if records[value.key].cross_term is None else records[value.key].cross_term
                for value in candidates
            ],
            dtype=np.float64,
        ),
        "prediction_energy": np.asarray(
            [
                np.nan
                if records[value.key].prediction_energy is None
                else records[value.key].prediction_energy
                for value in candidates
            ],
            dtype=np.float64,
        ),
        "gain": np.asarray(
            [
                np.nan if records[value.key].gain is None else records[value.key].gain
                for value in candidates
            ],
            dtype=np.float64,
        ),
        "eligible": np.asarray(
            [records[value.key].eligible for value in candidates], dtype=np.uint8
        ),
    }
    for name, expected in expected_numeric.items():
        if name not in table or not np.array_equal(table[name], expected, equal_nan=True):
            raise ArtifactCompatibilityError("frozen gain-table values changed")
    index = _load_json(run_dir / "gain_table.json")
    _verify_semantic(index, "gain_table.json")
    if (
        int(index.get("candidate_count", -1)) != len(candidates)
        or index.get("file_sha256") != file_fingerprint(table_path)
    ):
        raise ArtifactCompatibilityError("frozen gain-table index changed")
    return records


def _write_gain_table(run_dir: Path, records: Mapping[str, Any]) -> dict[str, Any]:
    candidates = [candidate for candidate in _available_nonzero_candidates(run_dir)]
    reasons = sorted({records[candidate.key].reason_code for candidate in candidates})
    reason_code = {reason: index for index, reason in enumerate(reasons)}
    artifact = _atomic_npz(
        run_dir / "gain_table.npz",
        quartile=np.asarray([value.quartile for value in candidates], dtype=np.int8),
        seed=np.asarray([value.seed for value in candidates], dtype=np.int64),
        update=np.asarray([value.update for value in candidates], dtype=np.int16),
        cross_term=np.asarray(
            [
                np.nan if records[value.key].cross_term is None else records[value.key].cross_term
                for value in candidates
            ],
            dtype=np.float64,
        ),
        prediction_energy=np.asarray(
            [
                np.nan
                if records[value.key].prediction_energy is None
                else records[value.key].prediction_energy
                for value in candidates
            ],
            dtype=np.float64,
        ),
        gain=np.asarray(
            [
                np.nan if records[value.key].gain is None else records[value.key].gain
                for value in candidates
            ],
            dtype=np.float64,
        ),
        eligible=np.asarray([records[value.key].eligible for value in candidates], dtype=np.uint8),
        reason_code=np.asarray(
            [reason_code[records[value.key].reason_code] for value in candidates],
            dtype=np.int16,
        ),
    )
    rows = [
        {
            "candidate_key": candidate.key,
            **records[candidate.key].to_record(),
        }
        for candidate in candidates
    ]
    atomic_write_csv(run_dir / "gain_candidate_summary.csv", rows)
    index = _semantic(
        {
            "schema": RUN_SCHEMA + "-gain-table-index",
            "candidate_count": len(candidates),
            "file_sha256": artifact["sha256"],
            "reason_dictionary": reason_code,
            "q0_q1_fixed_unit": 1,
            "q2_q3_no_clipping": 1,
        }
    )
    atomic_write_json(run_dir / "gain_table.json", index)
    return index


def _selected_system(
    run_dir: Path,
    selected: Sequence[Any],
) -> SelectedSystem:
    scales = [
        float(_load_json(run_dir / f"q{quartile}_target_scale.json")["target_scale"])
        for quartile in range(4)
    ]
    experts = []
    for record in selected:
        checkpoint = _candidate_checkpoint(run_dir, record.candidate)
        experts.append(
            SelectedExpert(
                candidate=record.candidate,
                checkpoint_path=checkpoint["checkpoint_path"],
                checkpoint_sha256=checkpoint["checkpoint_file_sha256"],
                model_state_sha256=checkpoint["state_sha256"],
                target_scale=scales[record.candidate.quartile],
                gain=float(record.gain_record.gain),
            )
        )
    bindings = tuple(
        HashBinding(name, file_fingerprint(run_dir / name))
        for name in sorted(
            ("fit_label_open.json", "gain_label_open.json", "rank_label_open.json")
        )
    )
    return SelectedSystem(
        experts=tuple(experts),
        candidate_grid_sha256=CANDIDATE_GRID_SHA256,
        gain_table_sha256=file_fingerprint(run_dir / "gain_table.npz"),
        rank_table_sha256=file_fingerprint(run_dir / "training_rank_path_tables.npz"),
        role_open_bindings=bindings,
    )


def _commit_calibrate_result(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    gains: Mapping[str, Any],
    gain_seal: Mapping[str, Any],
    per_quartile: Sequence[Mapping[str, Any]],
    selected_complete: bool,
    selected_candidate_count: int,
    rank_candidate_count: int,
    seal_names: Sequence[str],
) -> dict[str, Any]:
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-calibrate-metrics",
            "evaluation_status": "evaluated",
            **{name: int(selected_complete) for name in CALIBRATE_FLAGS},
            "gain_label_open_order_valid": 1,
            "gain_table_valid": int(
                len(gains) == len(_available_nonzero_candidates(run_dir))
            ),
            "gain_calibration_seal_valid": int(
                gain_seal.get("semantic_sha256") is not None
            ),
            "rank_label_open_order_valid": 1,
            "rank_rule_valid": 1,
            "selected_system_complete": int(selected_complete),
            "selected_system_sealed": int(selected_complete),
            "valid_scientific_negative": int(not selected_complete),
            "stage_execution_valid": 1,
            "inference_valid": 1,
            "scientific_negative_reason": (
                None
                if selected_complete
                else "one_or_more_quartiles_have_no_training_only_candidate"
            ),
            "per_quartile_diagnostics": [dict(value) for value in per_quartile],
            "candidate_count": int(rank_candidate_count),
            "selected_candidate_count": int(selected_candidate_count),
            "gain_labels_opened_before_rank": 1,
            "selection_paths_opened": 0,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "calibrate_metrics.json", metrics)
    gate = evaluate_calibrate_gate(metrics)
    atomic_write_json(run_dir / "calibrate_gate.json", gate)
    _seal_stage(
        run_dir,
        (*seal_names, "calibrate_metrics.json", "calibrate_gate.json"),
        "calibrate_artifact_seal.json",
    )
    return gate


def _finalize_orphaned_calibration(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    gains: Mapping[str, Any],
    gain_seal: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Finalize a sealed rank decision without regenerating frozen evidence."""

    selected_seal_path = run_dir / "selected_system_seal.json"
    negative_path = run_dir / "no_training_only_system.json"
    if not selected_seal_path.is_file() and not negative_path.is_file():
        return None
    if selected_seal_path.is_file() and negative_path.is_file():
        raise ArtifactCompatibilityError("conflicting orphaned calibration outcomes")
    rank_path = run_dir / "training_rank_path_tables.npz"
    summary_path = run_dir / "training_rank_candidate_summary.csv"
    if not rank_path.is_file() or not summary_path.is_file():
        raise ArtifactCompatibilityError("orphaned rank evidence is incomplete")
    with np.load(rank_path, allow_pickle=False) as archive:
        quartiles = np.array(archive["candidate_quartile"], copy=True)
        seeds = np.array(archive["candidate_seed"], copy=True)
        updates = np.array(archive["candidate_update"], copy=True)
        pooled = np.array(archive["pooled_improvement"], copy=True)
    rank_candidate_count = int(len(quartiles))
    base_names = [
        "gain_label_open.json",
        "gain_table.npz",
        "gain_table.json",
        "gain_candidate_summary.csv",
        "gain_calibration_seal.json",
        "rank_label_open.json",
        "training_rank_path_tables.npz",
        "training_rank_candidate_summary.csv",
    ]
    if selected_seal_path.is_file():
        _verify_stage_seal(run_dir, "selected_system_seal.json")
        system = SelectedSystem.from_record(_load_json(run_dir / "selected_system.json"))
        per_quartile = []
        for expert in system.experts:
            matches = np.flatnonzero(
                (quartiles == expert.candidate.quartile)
                & (seeds == expert.candidate.seed)
                & (updates == expert.candidate.update)
            )
            if matches.size != 1:
                raise ArtifactCompatibilityError("orphaned selected candidate changed")
            per_quartile.append(
                {
                    "quartile": expert.candidate.quartile,
                    "selected": 1,
                    "candidate_key": expert.candidate.key,
                    "pooled_improvement": float(pooled[int(matches[0])]),
                    "gain": expert.gain,
                }
            )
        base_names.extend(
            ["selected_experts.json", "selected_system.json", "selected_system_seal.json"]
        )
        return _commit_calibrate_result(
            run_dir,
            args,
            gains=gains,
            gain_seal=gain_seal,
            per_quartile=per_quartile,
            selected_complete=True,
            selected_candidate_count=4,
            rank_candidate_count=rank_candidate_count,
            seal_names=base_names,
        )
    negative = _load_json(negative_path)
    _verify_semantic(negative, negative_path.name)
    diagnostics = negative.get("per_quartile_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 4:
        raise ArtifactCompatibilityError("orphaned negative diagnostics changed")
    base_names.append("no_training_only_system.json")
    return _commit_calibrate_result(
        run_dir,
        args,
        gains=gains,
        gain_seal=gain_seal,
        per_quartile=diagnostics,
        selected_complete=False,
        selected_candidate_count=0,
        rank_candidate_count=rank_candidate_count,
        seal_names=base_names,
    )


def _calibrate_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "train_gate.json")):
        raise ArtifactCompatibilityError("calibration requires passing physical training")
    _verify_stage_seal(run_dir, "train_artifact_seal.json")
    _verify_complete_training_checkpoint_index(run_dir, args)
    gate_path = run_dir / "calibrate_gate.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, "calibrate_artifact_seal.json")
        return _load_json(gate_path)
    train_seal_sha = file_fingerprint(run_dir / "train_artifact_seal.json")
    device = torch.device(args.device)
    guard = ModelCallBatchGuard()
    gain_seal_path = run_dir / "gain_calibration_seal.json"
    if gain_seal_path.is_file():
        _open_role(
            run_dir,
            "gain_calibration",
            prerequisites={"train_artifact_seal.json": train_seal_sha},
        )
        gains = _load_frozen_gain_records(run_dir)
        gain_seal = _load_json(gain_seal_path)
    else:
        if (run_dir / "rank_label_open.json").is_file():
            raise ArtifactCompatibilityError(
                "rank labels opened without an immutable gain seal"
            )
        gain_labels, _ = _load_logical_labels(
            run_dir,
            "gain_calibration",
            prerequisites={"train_artifact_seal.json": train_seal_sha},
        )
        gain_target = np.array(
            gain_labels["denoising_target"], dtype=np.float64, copy=True, order="C"
        )
        gain_store = open_external_input_store(
            _cache_root(run_dir, "gain_calibration"), "train"
        )
        gains = _gain_records(
            run_dir,
            args,
            store=gain_store,
            target=gain_target,
            device=device,
            guard=guard,
        )
        _write_gain_table(run_dir, gains)
        gain_record_paths = tuple(
            f"gain_records/{candidate.key}.json"
            for candidate in _available_nonzero_candidates(run_dir)
            if candidate.quartile in (2, 3)
        )
        gain_seal = _seal_stage(
            run_dir,
            (
                "gain_label_open.json",
                "gain_table.npz",
                "gain_table.json",
                "gain_candidate_summary.csv",
                *gain_record_paths,
            ),
            "gain_calibration_seal.json",
        )
        gains = _load_frozen_gain_records(run_dir)

    if (run_dir / "rank_label_open.json").is_file():
        _open_role(
            run_dir,
            "training_rank",
            prerequisites={
                "gain_calibration_seal.json": file_fingerprint(
                    run_dir / "gain_calibration_seal.json"
                )
            },
        )
        orphaned = _finalize_orphaned_calibration(
            run_dir, args, gains=gains, gain_seal=gain_seal
        )
        if orphaned is not None:
            return orphaned

    rank_labels, _ = _load_logical_labels(
        run_dir,
        "training_rank",
        prerequisites={"gain_calibration_seal.json": file_fingerprint(run_dir / "gain_calibration_seal.json")},
    )
    rank_target = np.array(
        rank_labels["denoising_target"], dtype=np.float64, copy=True, order="C"
    )
    rank_store = open_external_input_store(_cache_root(run_dir, "training_rank"), "train")
    rank_records = []
    rank_path_records: list[dict[str, np.ndarray]] = []
    for candidate in _available_nonzero_candidates(run_dir):
        model = _load_expert(run_dir, candidate, device)
        rank_record, path_record = _rank_candidate(
            candidate,
            model,
            gains[candidate.key],
            rank_store,
            rank_target,
            device=device,
            guard=guard,
        )
        rank_records.append(rank_record)
        rank_path_records.append(path_record)
    rank_rows = [record.to_record() for record in rank_records]
    atomic_write_csv(run_dir / "training_rank_candidate_summary.csv", rank_rows)
    pooled = np.asarray([record.pooled_improvement for record in rank_records], dtype=np.float64)
    cells = np.asarray([record.fine_cell_improvements for record in rank_records], dtype=np.float64)
    eligible = np.asarray([record.eligible for record in rank_records], dtype=np.uint8)
    rank_paths = np.unique(
        np.asarray(rank_store.row_array("path_id"), dtype=np.int64)
    )
    if any(
        not np.array_equal(value["path_ids"], rank_paths)
        for value in rank_path_records
    ):
        raise ArtifactCompatibilityError("training-rank candidate paths changed")
    path_pooled = (
        np.stack([value["pooled_improvement"] for value in rank_path_records])
        if rank_path_records
        else np.empty((0, rank_paths.size), dtype=np.float64)
    )
    path_cells = (
        np.stack([value["fine_cell_improvement"] for value in rank_path_records])
        if rank_path_records
        else np.empty(
            (0, rank_paths.size, PHASE_COUNT, MIDPOINT_COUNT), dtype=np.float64
        )
    )
    path_cell_counts = (
        np.stack([value["fine_cell_count"] for value in rank_path_records])
        if rank_path_records
        else np.empty(
            (0, rank_paths.size, PHASE_COUNT, MIDPOINT_COUNT), dtype=np.int64
        )
    )
    _atomic_npz(
        run_dir / "training_rank_path_tables.npz",
        candidate_quartile=np.asarray(
            [record.candidate.quartile for record in rank_records], dtype=np.int8
        ),
        candidate_seed=np.asarray(
            [record.candidate.seed for record in rank_records], dtype=np.int64
        ),
        candidate_update=np.asarray(
            [record.candidate.update for record in rank_records], dtype=np.int16
        ),
        path_ids=np.ascontiguousarray(rank_paths, dtype=np.int64),
        per_path_pooled_improvement=np.ascontiguousarray(path_pooled),
        per_path_fine_cell_improvement=np.ascontiguousarray(path_cells),
        per_path_fine_cell_count=np.ascontiguousarray(path_cell_counts),
        pooled_improvement=pooled,
        fine_cell_improvement=cells,
        eligible=eligible,
    )
    per_quartile = []
    selected_records: tuple[Any, ...] = ()
    try:
        selected_records = select_training_rank_system(rank_records)
        for record in selected_records:
            per_quartile.append(
                {
                    "quartile": record.candidate.quartile,
                    "selected": 1,
                    "candidate_key": record.candidate.key,
                    "pooled_improvement": record.pooled_improvement,
                    "gain": record.gain_record.gain,
                }
            )
    except NoEligibleQuartileCandidateError:
        for quartile in range(4):
            eligible_q = [
                record for record in rank_records if record.candidate.quartile == quartile and record.eligible
            ]
            reasons = sorted(
                {
                    record.reason_code
                    for record in rank_records
                    if record.candidate.quartile == quartile
                }
            )
            per_quartile.append(
                {
                    "quartile": quartile,
                    "selected": int(bool(eligible_q)),
                    "eligible_candidate_count": len(eligible_q),
                    "reason_codes": reasons,
                }
            )
    selected_complete = len(selected_records) == 4
    seal_names = [
        "gain_label_open.json",
        "gain_table.npz",
        "gain_table.json",
        "gain_candidate_summary.csv",
        "gain_calibration_seal.json",
        "rank_label_open.json",
        "training_rank_path_tables.npz",
        "training_rank_candidate_summary.csv",
    ]
    if selected_complete:
        system = _selected_system(run_dir, selected_records)
        atomic_write_json(
            run_dir / "selected_experts.json",
            _semantic(
                {
                    "schema": RUN_SCHEMA + "-selected-experts",
                    "experts": [value.to_record() for value in system.experts],
                }
            ),
        )
        atomic_write_json(run_dir / "selected_system.json", system.to_record())
        selected_seal = _seal_stage(
            run_dir,
            (
                "selected_experts.json",
                "selected_system.json",
                "gain_calibration_seal.json",
                "training_rank_path_tables.npz",
                *(value.checkpoint_path for value in system.experts),
            ),
            "selected_system_seal.json",
        )
        seal_names.extend(
            ["selected_experts.json", "selected_system.json", "selected_system_seal.json"]
        )
    else:
        atomic_write_json(
            run_dir / "no_training_only_system.json",
            _semantic(
                {
                    "schema": RUN_SCHEMA + "-no-training-only-system",
                    "per_quartile_diagnostics": per_quartile,
                    "selection_paths_opened": 0,
                }
            ),
        )
        seal_names.append("no_training_only_system.json")
    return _commit_calibrate_result(
        run_dir,
        args,
        gains=gains,
        gain_seal=gain_seal,
        per_quartile=per_quartile,
        selected_complete=selected_complete,
        selected_candidate_count=len(selected_records),
        rank_candidate_count=len(rank_records),
        seal_names=seal_names,
    )


def _load_selected_system(run_dir: Path, device: torch.device) -> tuple[SelectedSystem, QuartileSpecialistBoundaryTangentPredictor]:
    _verify_stage_seal(run_dir, "calibrate_artifact_seal.json")
    _verify_stage_seal(run_dir, "selected_system_seal.json")
    system = SelectedSystem.from_record(_load_json(run_dir / "selected_system.json"))
    if (
        system.candidate_grid_sha256 != CANDIDATE_GRID_SHA256
        or system.gain_table_sha256 != file_fingerprint(run_dir / "gain_table.npz")
        or system.rank_table_sha256
        != file_fingerprint(run_dir / "training_rank_path_tables.npz")
        or tuple(value.candidate.quartile for value in system.experts) != (0, 1, 2, 3)
    ):
        raise ArtifactCompatibilityError("selected-system scientific binding changed")
    expected_role_bindings = tuple(
        HashBinding(name, file_fingerprint(run_dir / name))
        for name in sorted(
            ("fit_label_open.json", "gain_label_open.json", "rank_label_open.json")
        )
    )
    if system.role_open_bindings != expected_role_bindings:
        raise ArtifactCompatibilityError("selected-system role binding changed")
    experts = []
    for value in system.experts:
        checkpoint = _candidate_checkpoint(run_dir, value.candidate)
        if (
            value.checkpoint_path != checkpoint["checkpoint_path"]
            or value.checkpoint_sha256 != checkpoint["checkpoint_file_sha256"]
            or value.model_state_sha256 != checkpoint["state_sha256"]
            or value.target_scale
            != float(
                _load_json(
                    run_dir / f"q{value.candidate.quartile}_target_scale.json"
                )["target_scale"]
            )
        ):
            raise ArtifactCompatibilityError("selected expert binding changed")
        experts.append(_load_expert(run_dir, value.candidate, device))
    model = QuartileSpecialistBoundaryTangentPredictor(
        experts,
        gains=tuple(value.gain for value in system.experts),
        gains_sealed=True,
    ).to(device)
    model.eval()
    return system, model


def _branch_batch(value: Any) -> Any:
    return value.batch if hasattr(value, "batch") and hasattr(value.batch, "later_full_state") else value


def _branch_reductions(
    execution: Any,
    model: QuartileSpecialistBoundaryTangentPredictor,
    raw_model: nn.Module,
    *,
    device: torch.device,
    guard: ModelCallBatchGuard,
) -> dict[str, np.ndarray]:
    if execution.selected_step is None or len(execution.branches) != PHASE_COUNT:
        raise QuartileSpecialistWorkflowError(
            "selected audit shard omitted midpoint branches",
            failure_domain="audit_execution",
            failure_code="quartile_audit_branches_missing",
        )
    paths = tuple(int(value) for value in execution.path_ids)
    step = int(execution.selected_step)
    quartile = step // 128
    gains = model.gains
    blocks: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sample_key",
            "path_id",
            "outer_step",
            "phase",
            "midpoint_index",
            "specialist_vs_zero",
            "shrunken_vs_raw",
        )
    }
    for phase, branch in enumerate(execution.branches):
        if int(branch.phase) != phase:
            raise QuartileSpecialistWorkflowError(
                "audit branch phase order changed",
                failure_domain="audit_execution",
                failure_code="quartile_audit_phase_order_invalid",
            )
        batch = _branch_batch(branch.batch)
        later = batch.later_full_state.permute(1, 0, 2).contiguous()
        target = batch.denoising_target.permute(1, 0, 2).contiguous()
        row_count = len(paths) * MIDPOINT_COUNT
        raw_parts = []
        with torch.no_grad():
            for start in range(0, row_count, 32):
                stop = min(row_count, start + 32)
                rows = stop - start
                flat_later = later.reshape(row_count, STATE_SIZE)[start:stop]
                reverse_time = []
                phase_values = []
                for linear in range(start, stop):
                    midpoint = linear % MIDPOINT_COUNT
                    reverse_time.append(
                        internal_reverse_time(step, phase, MIDPOINT_FRACTIONS[midpoint])
                    )
                    phase_values.append(phase)
                inputs = ModelInputs(
                    later_full_state=flat_later.to(device=device, dtype=torch.float32),
                    reverse_time=torch.as_tensor(reverse_time, dtype=torch.float64, device=device),
                    phase=torch.as_tensor(phase_values, dtype=torch.long, device=device),
                    color=torch.full(
                        (rows,), PHASE_MATCHINGS[phase], dtype=torch.long, device=device
                    ),
                    duration=torch.full(
                        (rows,), PHASE_DURATIONS[phase], dtype=torch.float32, device=device
                    ),
                    label=torch.full((rows,), 3, dtype=torch.long, device=device),
                )
                raw_parts.append(
                    guard.call(raw_model, inputs)
                    .detach()
                    .to(torch.float64)
                    .cpu()
                    .numpy()
                )
        raw = np.ascontiguousarray(np.concatenate(raw_parts, axis=0))
        truth = target.reshape(row_count, EDGES_PER_PHASE).detach().cpu().numpy()
        final = raw * gains[quartile]
        zero_improvement = np.mean(
            truth * truth - (truth - final) ** 2, axis=1, dtype=np.float64
        )
        shrink_improvement = np.mean(
            (truth - raw) ** 2 - (truth - final) ** 2, axis=1, dtype=np.float64
        )
        path_grid = np.repeat(np.asarray(paths, dtype=np.int64), MIDPOINT_COUNT)
        midpoint_grid = np.tile(np.arange(MIDPOINT_COUNT, dtype=np.int64), len(paths))
        keys = np.fromiter(
            (
                midpoint_sample_key(int(path), step, phase, int(midpoint))
                for path, midpoint in zip(path_grid, midpoint_grid, strict=True)
            ),
            dtype=np.int64,
            count=row_count,
        )
        blocks["sample_key"].append(keys)
        blocks["path_id"].append(path_grid)
        blocks["outer_step"].append(np.full(row_count, step, dtype=np.int64))
        blocks["phase"].append(np.full(row_count, phase, dtype=np.int64))
        blocks["midpoint_index"].append(midpoint_grid)
        blocks["specialist_vs_zero"].append(np.ascontiguousarray(zero_improvement))
        blocks["shrunken_vs_raw"].append(np.ascontiguousarray(shrink_improvement))
    return {
        name: np.ascontiguousarray(np.concatenate(values))
        for name, values in blocks.items()
    }


def _audit_cohort_paths(run_dir: Path, role: str, index: int) -> tuple[Path, Path]:
    root = run_dir / f"{role}_cohorts"
    return root / f"cohort-{index:03d}.npz", root / f"cohort-{index:03d}.json"


def _load_audit_cohort(
    run_dir: Path,
    role: str,
    cohort: EagerCohort,
    *,
    selected_system_sha256: str,
    selected_outer_steps: Sequence[int],
    source_sha256: str,
    cohort_plan_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    data_path, metadata_path = _audit_cohort_paths(run_dir, role, cohort.index)
    if not metadata_path.exists():
        if data_path.exists():
            data_path.unlink()
        return None
    metadata = _load_json(metadata_path)
    _verify_semantic(metadata, metadata_path.name)
    selected_steps = tuple(int(value) for value in selected_outer_steps)
    expected_sample_key_sha256 = _canonical_audit_sample_key_sha256(
        cohort.path_ids, selected_steps
    )
    expected_row_count = (
        len(cohort.path_ids)
        * len(selected_steps)
        * PHASE_COUNT
        * MIDPOINT_COUNT
    )
    if (
        metadata.get("schema") != RUN_SCHEMA + "-audit-cohort"
        or metadata.get("role") != role
        or metadata.get("cohort_index") != cohort.index
        or metadata.get("cohort_kind") != cohort.kind
        or metadata.get("path_ids") != list(cohort.path_ids)
        or metadata.get("path_roles") != list(cohort.path_roles)
        or metadata.get("selected_system_sha256") != selected_system_sha256
        or metadata.get("selected_outer_steps") != list(selected_steps)
        or metadata.get("sample_key_sha256") != expected_sample_key_sha256
        or metadata.get("source_row_count") != expected_row_count
        or metadata.get("source_sha256") != source_sha256
        or metadata.get("cohort_plan_sha256") != cohort_plan_sha256
        or not data_path.is_file()
        or metadata.get("artifact_sha256") != file_fingerprint(data_path)
    ):
        raise ArtifactCompatibilityError("committed audit cohort changed")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    if config_fingerprint(
        {name: hashlib.sha256(value.tobytes(order="C")).hexdigest() for name, value in arrays.items()}
    ) != metadata.get("arrays_sha256"):
        raise ArtifactCompatibilityError("committed audit cohort arrays changed")
    expected_arrays = {
        "path_ids",
        "primary_values",
        "primary_counts",
        "local_values",
        "local_counts",
        "selected_outer_steps",
    }
    if set(arrays) != expected_arrays:
        raise ArtifactCompatibilityError("committed audit cohort schema changed")
    # Reconstruct the typed path table so shapes, Cartesian counts, selected
    # steps, and the canonical sample-grid commitment are checked on every
    # resume rather than trusted from self-hashed metadata.
    QuartileAuditPathTable(
        path_ids=arrays["path_ids"],
        primary_values=arrays["primary_values"],
        local_values=arrays["local_values"],
        primary_counts=arrays["primary_counts"],
        local_counts=arrays["local_counts"],
        selected_outer_steps=arrays["selected_outer_steps"],
        sample_key_sha256=expected_sample_key_sha256,
        row_count=expected_row_count,
    )
    return arrays, metadata


def _canonical_audit_sample_key_sha256(
    path_ids: Sequence[int], selected_steps: Sequence[int]
) -> str:
    keys = np.fromiter(
        (
            midpoint_sample_key(path_id, step, phase, midpoint)
            for path_id in sorted(int(value) for value in path_ids)
            for step in selected_steps
            for phase in range(PHASE_COUNT)
            for midpoint in range(MIDPOINT_COUNT)
        ),
        dtype=np.int64,
        count=(
            len(path_ids)
            * len(selected_steps)
            * PHASE_COUNT
            * MIDPOINT_COUNT
        ),
    )
    return hashlib.sha256(
        np.ascontiguousarray(np.sort(keys, kind="stable")).tobytes(order="C")
    ).hexdigest()


def _stream_audit_cohort(
    run_dir: Path,
    role: str,
    cohort: EagerCohort,
    all_cohorts: Sequence[EagerCohort],
    *,
    source: np.ndarray,
    model: QuartileSpecialistBoundaryTangentPredictor,
    raw_model: nn.Module,
    system_sha256: str,
    args: argparse.Namespace,
    device: torch.device,
    guard: ModelCallBatchGuard,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    plan = explicit_eager_cache_plan(all_cohorts)
    selected_steps = _effective_selected_steps(args)
    source_sha = source_measure_sha256(source)
    cached = _load_audit_cohort(
        run_dir,
        role,
        cohort,
        selected_system_sha256=system_sha256,
        selected_outer_steps=selected_steps,
        source_sha256=source_sha,
        cohort_plan_sha256=plan["semantic_sha256"],
    )
    if cached is not None:
        return cached
    kwargs = _cache_generation_kwargs(args)
    executions = iter_eager_shards_for_cohorts(
        source,
        cohorts=all_cohorts,
        cohort_plan_sha256=plan["semantic_sha256"],
        device=args.device,
        root_seed=ROOT_SEED,
        cohort_indices=(cohort.index,),
        **kwargs,
    )
    blocks: list[dict[str, np.ndarray]] = []
    transition_count = 0
    certified_count = 0
    fallback_count = 0
    fallback_elapsed = 0.0
    elapsed = 0.0
    maximum_mass_error = 0.0
    maximum_peak = 0.0
    forbidden_count = 0
    for execution in executions:
        diagnostics = execution.diagnostics
        transition_count += int(diagnostics["transition_count"])
        certified_count += int(diagnostics["certified_count"])
        fallback_count += int(diagnostics.get("fallback_count", 0))
        fallback_elapsed += float(diagnostics.get("fallback_elapsed_seconds", 0.0))
        elapsed += float(diagnostics.get("complete_pipeline_elapsed_seconds", 0.0))
        maximum_mass_error = max(maximum_mass_error, float(diagnostics["maximum_mass_error"]))
        maximum_peak = max(
            maximum_peak, float(diagnostics.get("maximum_peak_memory_fraction", 0.0))
        )
        forbidden_count += int(diagnostics.get("forbidden_event_count", 0))
        if execution.selected_step is not None:
            blocks.append(
                _branch_reductions(
                    execution,
                    model,
                    raw_model,
                    device=device,
                    guard=guard,
                )
            )
    if not blocks:
        raise QuartileSpecialistWorkflowError(
            "audit cohort produced no selected rows",
            failure_domain="audit_execution",
            failure_code="quartile_audit_rows_absent",
        )
    row_arrays = {
        name: np.ascontiguousarray(np.concatenate([block[name] for block in blocks]))
        for name in blocks[0]
    }
    path_table = aggregate_quartile_audit_improvements(
        sample_keys=row_arrays["sample_key"],
        row_path_ids=row_arrays["path_id"],
        outer_steps=row_arrays["outer_step"],
        phases=row_arrays["phase"],
        midpoint_indices=row_arrays["midpoint_index"],
        specialist_vs_zero_improvements=row_arrays["specialist_vs_zero"],
        shrunken_vs_raw_improvements=row_arrays["shrunken_vs_raw"],
        expected_path_ids=cohort.path_ids,
        selected_outer_steps=selected_steps,
        expected_path_count=None,
    )
    expected_sample_key_sha256 = _canonical_audit_sample_key_sha256(
        cohort.path_ids, selected_steps
    )
    if path_table.sample_key_sha256 != expected_sample_key_sha256:
        raise ArtifactCompatibilityError("audit cohort sample-key grid changed")
    arrays = {
        "path_ids": np.ascontiguousarray(path_table.path_ids),
        "primary_values": np.ascontiguousarray(path_table.primary_values),
        "primary_counts": np.ascontiguousarray(path_table.primary_counts),
        "local_values": np.ascontiguousarray(path_table.local_values),
        "local_counts": np.ascontiguousarray(path_table.local_counts),
        "selected_outer_steps": np.ascontiguousarray(
            path_table.selected_outer_steps
        ),
    }
    data_path, metadata_path = _audit_cohort_paths(run_dir, role, cohort.index)
    artifact = _atomic_npz(data_path, **arrays)
    array_hash = config_fingerprint(
        {name: hashlib.sha256(value.tobytes(order="C")).hexdigest() for name, value in arrays.items()}
    )
    metadata = _semantic(
        {
            "schema": RUN_SCHEMA + "-audit-cohort",
            "role": role,
            "cohort_index": cohort.index,
            "cohort_kind": cohort.kind,
            "path_ids": list(cohort.path_ids),
            "path_roles": list(cohort.path_roles),
            "selected_system_sha256": system_sha256,
            "selected_outer_steps": list(selected_steps),
            "source_sha256": source_sha,
            "cohort_plan_sha256": plan["semantic_sha256"],
            "artifact_path": data_path.relative_to(run_dir).as_posix(),
            "artifact_sha256": artifact["sha256"],
            "arrays_sha256": array_hash,
            "path_count": len(arrays["path_ids"]),
            "reduced_row_count": len(arrays["path_ids"]),
            "source_row_count": path_table.row_count,
            "sample_key_sha256": path_table.sample_key_sha256,
            "transition_count": transition_count,
            "certified_count": certified_count,
            "fallback_count": fallback_count,
            "fallback_elapsed_seconds": fallback_elapsed,
            "complete_pipeline_elapsed_seconds": elapsed,
            "maximum_mass_error": maximum_mass_error,
            "maximum_peak_memory_fraction": maximum_peak,
            "forbidden_event_count": forbidden_count,
            "per_path_risk_summaries_only": 1,
            "row_level_risks_persisted": 0,
            "raw_states_persisted": 0,
            "raw_labels_persisted": 0,
            "raw_predictions_persisted": 0,
        }
    )
    atomic_write_json(metadata_path, metadata)
    return arrays, metadata


def _save_path_table(run_dir: Path, role: str, table: Any) -> None:
    _atomic_npz(
        run_dir / f"{role}_primary_path_values.npz",
        path_ids=table.path_ids,
        primary_values=table.primary_values,
        primary_counts=table.primary_counts,
    )
    _atomic_npz(
        run_dir / f"{role}_local_path_values.npz",
        path_ids=table.path_ids,
        local_values=table.local_values,
        local_counts=table.local_counts,
    )
    atomic_write_json(run_dir / f"{role}_path_table.json", table.to_record())


def _audit_evidence_paths(run_dir: Path, role: str) -> tuple[Path, ...]:
    roots = (
        run_dir / f"{role}_cohorts",
        run_dir / "bootstrap_counts" / role,
        run_dir / "bootstrap_maxima" / role,
    )
    if any(not root.is_dir() for root in roots):
        raise ArtifactCompatibilityError("audit evidence directory is absent")
    return tuple(
        sorted(
            path
            for root in roots
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def _audit_evidence_index(
    run_dir: Path,
    role: str,
    *,
    cohorts: Sequence[EagerCohort],
    selected_system_sha256: str,
    selected_outer_steps: Sequence[int],
    source_sha256: str,
    cohort_plan_sha256: str,
    replicates: int,
    shard_size: int,
) -> dict[str, Any]:
    paths = _audit_evidence_paths(run_dir, role)
    rows = []
    for path in paths:
        semantic_sha256 = None
        if path.suffix == ".json":
            value = _load_json(path)
            _verify_semantic(value, path.name)
            semantic_sha256 = value["semantic_sha256"]
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
                "semantic_sha256": semantic_sha256,
            }
        )
    expected_shards = int(replicates) // int(shard_size)
    expected_file_count = 2 * len(cohorts) + 4 * expected_shards
    if len(rows) != expected_file_count:
        raise ArtifactCompatibilityError("audit evidence file set is incomplete")
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-audit-evidence-index",
            "schema_version": 1,
            "role": role,
            "selected_system_sha256": selected_system_sha256,
            "selected_system_seal_sha256": file_fingerprint(
                run_dir / "selected_system_seal.json"
            ),
            "selected_outer_steps": [int(value) for value in selected_outer_steps],
            "source_sha256": source_sha256,
            "cohort_plan_sha256": cohort_plan_sha256,
            "cohort_count": len(cohorts),
            "bootstrap_shard_count": expected_shards,
            "file_count": len(rows),
            "files": rows,
            "row_level_risks_persisted": 0,
            "raw_states_persisted": 0,
            "raw_labels_persisted": 0,
            "raw_predictions_persisted": 0,
        }
    )
    atomic_write_json(run_dir / f"{role}_evidence_index.json", record)
    return record


def _verify_audit_evidence_index(
    run_dir: Path,
    role: str,
    *,
    cohorts: Sequence[EagerCohort],
    selected_system_sha256: str,
    selected_outer_steps: Sequence[int],
    source_sha256: str,
    cohort_plan_sha256: str,
    replicates: int,
    shard_size: int,
) -> dict[str, Any]:
    record = _load_json(run_dir / f"{role}_evidence_index.json")
    _verify_semantic(record, f"{role}_evidence_index.json")
    expected_shards = int(replicates) // int(shard_size)
    if (
        record.get("schema") != RUN_SCHEMA + "-audit-evidence-index"
        or record.get("role") != role
        or record.get("selected_system_sha256") != selected_system_sha256
        or record.get("selected_system_seal_sha256")
        != file_fingerprint(run_dir / "selected_system_seal.json")
        or record.get("selected_outer_steps")
        != [int(value) for value in selected_outer_steps]
        or record.get("source_sha256") != source_sha256
        or record.get("cohort_plan_sha256") != cohort_plan_sha256
        or record.get("cohort_count") != len(cohorts)
        or record.get("bootstrap_shard_count") != expected_shards
        or record.get("file_count") != 2 * len(cohorts) + 4 * expected_shards
        or any(
            record.get(name) != 0
            for name in (
                "row_level_risks_persisted",
                "raw_states_persisted",
                "raw_labels_persisted",
                "raw_predictions_persisted",
            )
        )
    ):
        raise ArtifactCompatibilityError("audit evidence index binding changed")
    expected_rows = record.get("files")
    if not isinstance(expected_rows, list):
        raise ArtifactCompatibilityError("audit evidence index files are malformed")
    actual_paths = _audit_evidence_paths(run_dir, role)
    actual_relative = [path.relative_to(run_dir).as_posix() for path in actual_paths]
    expected_relative = [str(value.get("path")) for value in expected_rows]
    if actual_relative != expected_relative:
        raise ArtifactCompatibilityError("audit evidence file set changed")
    for path, row in zip(actual_paths, expected_rows, strict=True):
        if (
            row.get("size") != path.stat().st_size
            or row.get("sha256") != file_fingerprint(path)
        ):
            raise ArtifactCompatibilityError(f"audit evidence changed: {path}")
        if path.suffix == ".json":
            value = _load_json(path)
            _verify_semantic(value, path.name)
            if row.get("semantic_sha256") != value.get("semantic_sha256"):
                raise ArtifactCompatibilityError(
                    f"audit evidence semantic record changed: {path}"
                )
        elif row.get("semantic_sha256") is not None:
            raise ArtifactCompatibilityError("binary audit evidence has semantic hash")
    return record


def _audit_stage(run_dir: Path, args: argparse.Namespace, role: str) -> dict[str, Any]:
    if role not in {"selection", "confirmation"}:
        raise ValueError("audit role must be selection or confirmation")
    prerequisite_gate = (
        "calibrate_gate.json" if role == "selection" else "selection_gate.json"
    )
    if not _passed(_load_json(run_dir / prerequisite_gate)):
        raise ArtifactCompatibilityError(f"{role} requires a passing prior gate")
    prior_seal = (
        "calibrate_artifact_seal.json"
        if role == "selection"
        else "selection_artifact_seal.json"
    )
    _verify_stage_seal(run_dir, prior_seal)
    gate_path = run_dir / f"{role}_gate.json"
    seal_name = f"{role}_artifact_seal.json"
    if gate_path.is_file():
        _verify_stage_seal(run_dir, seal_name)
        device = torch.device(args.device)
        system, completed_model = _load_selected_system(run_dir, device)
        del completed_model
        source = _source_target(args.parent_coarse_witness_run_dir)
        logical_role = (
            "fresh_selection" if role == "selection" else "untouched_confirmation"
        )
        paths = _effective_paths(args, logical_role)
        cohorts = _cohorts(paths, kind="confirmation", role="confirmation")
        selected_steps = _effective_selected_steps(args)
        audit_plan = explicit_eager_cache_plan(cohorts)
        replicates, shard_size = _effective_bootstrap(args)
        _verify_audit_evidence_index(
            run_dir,
            role,
            cohorts=cohorts,
            selected_system_sha256=system.semantic_sha256,
            selected_outer_steps=selected_steps,
            source_sha256=source_measure_sha256(source),
            cohort_plan_sha256=audit_plan["semantic_sha256"],
            replicates=replicates,
            shard_size=shard_size,
        )
        for cohort in cohorts:
            _verify_stage_seal(run_dir, "selected_system_seal.json")
            committed = _load_audit_cohort(
                run_dir,
                role,
                cohort,
                selected_system_sha256=system.semantic_sha256,
                selected_outer_steps=selected_steps,
                source_sha256=source_measure_sha256(source),
                cohort_plan_sha256=audit_plan["semantic_sha256"],
            )
            if committed is None:
                raise ArtifactCompatibilityError(
                    "completed audit cohort is absent"
                )
        _verify_stage_seal(run_dir, "selected_system_seal.json")
        _verify_audit_evidence_index(
            run_dir,
            role,
            cohorts=cohorts,
            selected_system_sha256=system.semantic_sha256,
            selected_outer_steps=selected_steps,
            source_sha256=source_measure_sha256(source),
            cohort_plan_sha256=audit_plan["semantic_sha256"],
            replicates=replicates,
            shard_size=shard_size,
        )
        _verify_stage_seal(run_dir, seal_name)
        return _load_json(gate_path)
    logical_role = "fresh_selection" if role == "selection" else "untouched_confirmation"
    # Verify the immutable system and source before irreversibly opening a
    # fresh evidence role.
    device = torch.device(args.device)
    system, model = _load_selected_system(run_dir, device)
    source = _source_target(args.parent_coarse_witness_run_dir)
    _open_role(
        run_dir,
        logical_role,
        prerequisites={prior_seal: file_fingerprint(run_dir / prior_seal)},
    )
    raw_model = _RawQuartilePredictionAdapter(model).to(device)
    raw_model.eval()
    system_sha = system.semantic_sha256
    paths = _effective_paths(args, logical_role)
    cohorts = _cohorts(paths, kind="confirmation", role="confirmation")
    guard = ModelCallBatchGuard()
    arrays_list = []
    metadata_list = []
    for cohort in cohorts:
        _verify_stage_seal(run_dir, "selected_system_seal.json")
        arrays, metadata = _stream_audit_cohort(
            run_dir,
            role,
            cohort,
            cohorts,
            source=source,
            model=model,
            raw_model=raw_model,
            system_sha256=system_sha,
            args=args,
            device=device,
            guard=guard,
        )
        arrays_list.append(arrays)
        metadata_list.append(metadata)
        print(
            f"quartile {role} cohort={cohort.index + 1}/{len(cohorts)} committed",
            flush=True,
        )
    _verify_stage_seal(run_dir, "selected_system_seal.json")
    selected_steps = _effective_selected_steps(args)
    table = QuartileAuditPathTable(
        path_ids=np.ascontiguousarray(
            np.concatenate([value["path_ids"] for value in arrays_list])
        ),
        primary_values=np.ascontiguousarray(
            np.concatenate([value["primary_values"] for value in arrays_list])
        ),
        local_values=np.ascontiguousarray(
            np.concatenate([value["local_values"] for value in arrays_list])
        ),
        primary_counts=np.ascontiguousarray(
            np.concatenate([value["primary_counts"] for value in arrays_list])
        ),
        local_counts=np.ascontiguousarray(
            np.concatenate([value["local_counts"] for value in arrays_list])
        ),
        selected_outer_steps=np.asarray(selected_steps, dtype=np.int64),
        sample_key_sha256=_canonical_audit_sample_key_sha256(paths, selected_steps),
        row_count=(
            len(paths) * len(selected_steps) * PHASE_COUNT * MIDPOINT_COUNT
        ),
    )
    if not np.array_equal(table.path_ids, np.asarray(paths, dtype=np.int64)):
        raise ArtifactCompatibilityError("audit path order changed")
    _save_path_table(run_dir, role, table)
    screen = evaluate_local_compatibility_screen(table.local_values)
    atomic_write_json(run_dir / f"{role}_local_screen.json", screen.to_record())
    seed = SELECTION_BOOTSTRAP_SEED if role == "selection" else CONFIRMATION_BOOTSTRAP_SEED
    namespace = SELECTION_NAMESPACE if role == "selection" else CONFIRMATION_NAMESPACE
    replicates, shard_size = _effective_bootstrap(args)
    result, count_records, maxima_records = restartable_quartile_max_t(
        table.primary_values,
        path_ids=table.path_ids,
        count_directory=run_dir / "bootstrap_counts" / role,
        maxima_directory=run_dir / "bootstrap_maxima" / role,
        seed=seed,
        namespace=namespace,
        confidence=DEFAULT_CONFIDENCE,
        replicates=replicates,
        shard_size=shard_size,
    )
    atomic_write_json(run_dir / f"{role}_max_t.json", result.to_record())
    authorizing = not args.test_only
    record = (
        selection_record(
            result,
            screen,
            path_table=table,
            count_records=count_records,
            maxima_records=maxima_records,
            authorizing=authorizing,
        )
        if role == "selection"
        else confirmation_record(
            result,
            screen,
            path_table=table,
            count_records=count_records,
            maxima_records=maxima_records,
            authorizing=authorizing,
        )
    )
    atomic_write_json(run_dir / f"{role}_record.json", record)
    audit_plan = explicit_eager_cache_plan(cohorts)
    _audit_evidence_index(
        run_dir,
        role,
        cohorts=cohorts,
        selected_system_sha256=system_sha,
        selected_outer_steps=selected_steps,
        source_sha256=source_measure_sha256(source),
        cohort_plan_sha256=audit_plan["semantic_sha256"],
        replicates=replicates,
        shard_size=shard_size,
    )
    transition_count = sum(int(value["transition_count"]) for value in metadata_list)
    certified_count = sum(int(value["certified_count"]) for value in metadata_list)
    elapsed = sum(
        float(value["complete_pipeline_elapsed_seconds"]) for value in metadata_list
    )
    fallback_count = sum(int(value["fallback_count"]) for value in metadata_list)
    fallback_elapsed = sum(
        float(value["fallback_elapsed_seconds"]) for value in metadata_list
    )
    minimum_cohort_rate = min(
        int(value["transition_count"])
        / max(
            float(value["complete_pipeline_elapsed_seconds"]),
            np.finfo(float).tiny,
        )
        for value in metadata_list
    )
    fallback_fraction = fallback_count / max(transition_count, 1)
    fallback_time_fraction = fallback_elapsed / max(
        elapsed, np.finfo(float).tiny
    )
    peak_memory_fraction = max(
        float(value["maximum_peak_memory_fraction"]) for value in metadata_list
    )
    numerically_valid = bool(
        certified_count == transition_count
        and all(int(value["forbidden_event_count"]) == 0 for value in metadata_list)
        and all(float(value["maximum_mass_error"]) <= 2.0e-12 for value in metadata_list)
        and (
            args.test_only
            or (
                minimum_cohort_rate >= MINIMUM_TRANSITIONS_PER_SECOND
                and fallback_fraction <= 1.0e-4
                and fallback_time_fraction <= 0.10
                and peak_memory_fraction <= MAXIMUM_PEAK_MEMORY_FRACTION
            )
        )
    )
    if role == "selection":
        flags = SELECT_FLAGS
        flag_values = {
            "selected_system_seal_valid": 1,
            "fresh_selection_paths_valid": int(table.path_count == len(paths)),
            "one_system_only": 1,
            "six_family_inference_valid": int(numerically_valid),
            "all_six_lower_bounds_positive": int(result.passed and authorizing),
            "all_local_screens_pass": int(screen.passed and authorizing),
            "raw_selection_cache_absent": 1,
        }
        evaluator = evaluate_select_gate
    else:
        flags = CONFIRM_FLAGS
        flag_values = {
            "selected_system_unchanged": 1,
            "untouched_confirmation_paths_valid": int(table.path_count == len(paths)),
            "confirmation_open_once": 1,
            "six_family_inference_valid": int(numerically_valid),
            "all_six_lower_bounds_positive": int(result.passed and authorizing),
            "all_local_screens_pass": int(screen.passed and authorizing),
            "no_fitting_or_mutation": 1,
            "raw_confirmation_cache_absent": 1,
        }
        evaluator = evaluate_confirm_gate
    passed = bool(record["passed"])
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + f"-{role}-metrics",
            "evaluation_status": "evaluated",
            **{name: flag_values.get(name, 0) for name in flags},
            "valid_scientific_negative": int(not passed and authorizing),
            "stage_execution_valid": 1,
            "inference_valid": int(numerically_valid),
            "scientific_negative_reason": (
                None
                if passed
                else "six_family_or_local_compatibility_gate_not_satisfied"
            ),
            "path_count": table.path_count,
            "transition_count": transition_count,
            "certified_count": certified_count,
            "complete_pipeline_elapsed_seconds": elapsed,
            "minimum_cohort_transitions_per_second": minimum_cohort_rate,
            "fallback_fraction": fallback_fraction,
            "fallback_time_fraction": fallback_time_fraction,
            "peak_memory_fraction": peak_memory_fraction,
            "critical_value": result.critical_value,
            "lower_bounds": result.to_record()["lower_bounds"],
            "local_quartiles_passed": screen.quartile_passed.astype(int).tolist(),
            "selected_system_sha256": system_sha,
            "raw_cache_persisted": 0,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / f"{role}_metrics.json", metrics)
    gate = evaluator(metrics)
    atomic_write_json(gate_path, gate)
    _verify_stage_seal(run_dir, "selected_system_seal.json")
    _verify_audit_evidence_index(
        run_dir,
        role,
        cohorts=cohorts,
        selected_system_sha256=system_sha,
        selected_outer_steps=selected_steps,
        source_sha256=source_measure_sha256(source),
        cohort_plan_sha256=audit_plan["semantic_sha256"],
        replicates=replicates,
        shard_size=shard_size,
    )
    _seal_stage(
        run_dir,
        (
            _OPEN_FILES[logical_role],
            "selected_system_seal.json",
            f"{role}_primary_path_values.npz",
            f"{role}_local_path_values.npz",
            f"{role}_path_table.json",
            f"{role}_local_screen.json",
            f"{role}_max_t.json",
            f"{role}_record.json",
            f"{role}_evidence_index.json",
            f"{role}_metrics.json",
            f"{role}_gate.json",
        ),
        seal_name,
    )
    return gate


_STAGE_GATE_FILES = {
    "preflight": "preflight_gate.json",
    "cache": "cache_gate.json",
    "controls": "controls_gate.json",
    "train": "train_gate.json",
    "calibrate": "calibrate_gate.json",
    "select": "selection_gate.json",
    "confirm": "confirmation_gate.json",
}

_STAGE_EVALUATORS = {
    "preflight": evaluate_preflight_gate,
    "cache": evaluate_cache_gate,
    "controls": evaluate_controls_gate,
    "train": evaluate_train_gate,
    "calibrate": evaluate_calibrate_gate,
    "select": evaluate_select_gate,
    "confirm": evaluate_confirm_gate,
}


def _gate_bundle(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        stage: (
            _load_json(run_dir / filename)
            if (run_dir / filename).is_file()
            else not_evaluated_gate(stage)
        )
        for stage, filename in _STAGE_GATE_FILES.items()
    }


def _workflow_evidence(run_dir: Path) -> dict[str, Any]:
    for name in (
        "confirmation_metrics.json",
        "selection_metrics.json",
        "calibrate_metrics.json",
    ):
        value = _optional_json(run_dir, name)
        if isinstance(value, dict) and "per_quartile_diagnostics" in value:
            return {"per_quartile_diagnostics": value["per_quartile_diagnostics"]}
    return {}


def _commit_workflow(run_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    gates = _gate_bundle(run_dir)
    decision = decide_workflow(
        preflight_gate=gates["preflight"],
        cache_gate=gates["cache"],
        controls_gate=gates["controls"],
        train_gate=gates["train"],
        calibrate_gate=gates["calibrate"],
        select_gate=gates["select"],
        confirm_gate=gates["confirm"],
        evidence=_workflow_evidence(run_dir),
    )
    workflow = evaluate_required_gate(
        preflight_gate=gates["preflight"],
        cache_gate=gates["cache"],
        controls_gate=gates["controls"],
        train_gate=gates["train"],
        calibrate_gate=gates["calibrate"],
        select_gate=gates["select"],
        confirm_gate=gates["confirm"],
        decision=decision,
        require_gate=args.require_gate,
    )
    atomic_write_json(run_dir / "quartile_specialist_decision.json", decision)
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    decision_name = str(decision["decision"])
    if decision_name.startswith("ready_for_"):
        state = "ready"
    elif decision_name == "exact_rb_quartile_specialist_time_local_signal_confirmed":
        state = "completed"
    elif int(decision.get("valid_scientific_negative", 0)) == 1:
        state = "gate_failed"
    else:
        state = "failed"
    _status(
        run_dir,
        state=state,
        stage=args.stage,
        decision=decision_name,
        message=str(decision.get("next_action", "")),
    )
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    return decision, workflow


def _report_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gates = _gate_bundle(run_dir)
    for stage, filename in _STAGE_GATE_FILES.items():
        gate = gates[stage]
        if gate.get("evaluation_status") == "evaluated" and (run_dir / filename).is_file():
            seal_name = {
                "preflight": "preflight_artifact_seal.json",
                "cache": "cache_artifact_seal.json",
                "controls": "controls_artifact_seal.json",
                "train": "train_artifact_seal.json",
                "calibrate": "calibrate_artifact_seal.json",
                "select": "selection_artifact_seal.json",
                "confirm": "confirmation_artifact_seal.json",
            }[stage]
            if (run_dir / seal_name).is_file():
                _verify_stage_seal(run_dir, seal_name)
    _write_parent_immutability_after(run_dir, args)
    rows = []
    for stage in _STAGE_GATE_FILES:
        gate = gates[stage]
        rows.append(
            {
                "stage": stage,
                "evaluation_status": gate.get("evaluation_status", "not_evaluated"),
                "passed": int(gate.get("passed", 0)),
                "valid_scientific_negative": int(
                    gate.get("valid_scientific_negative", 0)
                ),
                "failure_domain": gate.get("failure_domain"),
                "failure_code": gate.get("failure_code"),
            }
        )
    atomic_write_csv(run_dir / "quartile_specialist_gate_summary.csv", rows)
    report = _semantic(
        {
            "schema": RUN_SCHEMA + "-report",
            "schema_version": 1,
            "generated_at": _now(),
            "stages_evaluated": [
                row["stage"]
                for row in rows
                if row["evaluation_status"] == "evaluated"
            ],
            "opened_roles": list(_opened_roles(run_dir)),
            "physical_training_performed": int(
                (run_dir / "fit_label_open.json").is_file()
            ),
            "selection_paths_opened": int(
                (run_dir / "selection_open.json").is_file()
            ),
            "confirmation_paths_opened": int(
                (run_dir / "confirmation_open.json").is_file()
            ),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "quartile_specialist_report.json", report)
    return report


def _record_execution_failure(
    run_dir: Path,
    args: argparse.Namespace,
    stage: str,
    exc: BaseException,
) -> None:
    if stage not in _STAGE_GATE_FILES:
        _status(
            run_dir,
            state="failed",
            stage=stage,
            message=str(exc),
            failure_domain="workflow_execution",
            failure_code="quartile_specialist_report_execution_failed",
        )
        atomic_write_json(run_dir / "artifact_registry.json", _artifact_registry(run_dir))
        return
    if isinstance(exc, QuartileSpecialistWorkflowError):
        domain = exc.failure_domain
        code = exc.failure_code
    elif isinstance(exc, torch.cuda.OutOfMemoryError):
        domain = "resource_gate"
        code = "quartile_specialist_cuda_memory_resource_infeasible"
    elif isinstance(exc, ArtifactCompatibilityError):
        domain = "provenance" if stage == "preflight" else "artifact_integrity"
        code = f"quartile_specialist_{stage}_compatibility_invalid"
    else:
        domain = "workflow_execution"
        code = f"quartile_specialist_{stage}_execution_failed"
    flags = {
        "preflight": PREFLIGHT_FLAGS,
        "cache": CACHE_FLAGS,
        "controls": CONTROLS_FLAGS,
        "train": TRAIN_FLAGS,
        "calibrate": CALIBRATE_FLAGS,
        "select": SELECT_FLAGS,
        "confirm": CONFIRM_FLAGS,
    }[stage]
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + f"-{stage}-failure-metrics",
            "evaluation_status": "execution_failed",
            **{name: 0 for name in flags},
            "valid_scientific_negative": 0,
            "stage_execution_valid": 0,
            "inference_valid": 0,
            "scientific_evidence_complete": 0,
            "failure_domain": domain,
            "failure_code": code,
            "error": str(exc),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / f"{stage}_failure.json", metrics)
    gate = _STAGE_EVALUATORS[stage](metrics)
    atomic_write_json(run_dir / _STAGE_GATE_FILES[stage], gate)
    _status(
        run_dir,
        state="failed",
        stage=stage,
        message=str(exc),
        failure_domain=domain,
        failure_code=code,
    )
    # Commit readable evidence before the invocation exits nonzero.
    try:
        _commit_workflow(run_dir, args)
    finally:
        atomic_write_json(run_dir / "artifact_registry.json", _artifact_registry(run_dir))


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return (
            "preflight",
            "cache",
            "controls",
            "train",
            "calibrate",
            "select",
            "confirm",
        )
    if stage == "report":
        return ()
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return (stage,)


def _execute_stage(run_dir: Path, args: argparse.Namespace, stage: str) -> dict[str, Any]:
    if stage == "preflight":
        return _preflight_stage(run_dir, args)
    _verify_frozen_parents(run_dir, args)
    ordered = ("preflight", "cache", "controls", "train", "calibrate", "select", "confirm")
    seal_names = {
        "preflight": "preflight_artifact_seal.json",
        "cache": "cache_artifact_seal.json",
        "controls": "controls_artifact_seal.json",
        "train": "train_artifact_seal.json",
        "calibrate": "calibrate_artifact_seal.json",
        "select": "selection_artifact_seal.json",
        "confirm": "confirmation_artifact_seal.json",
    }
    for predecessor in ordered[: ordered.index(stage)]:
        gate = _load_json(run_dir / _STAGE_GATE_FILES[predecessor])
        if not _passed(gate):
            raise ArtifactCompatibilityError(
                f"{stage} requires passing {predecessor} evidence"
            )
        _verify_stage_seal(run_dir, seal_names[predecessor])
    _verify_prospective_bootstrap_counts(run_dir, args)
    if stage == "cache":
        return _cache_stage(run_dir, args)
    if stage == "controls":
        return _controls_stage(run_dir, args)
    if stage == "train":
        return _train_stage(run_dir, args)
    if stage == "calibrate":
        return _calibrate_stage(run_dir, args)
    if stage == "select":
        return _audit_stage(run_dir, args, "selection")
    if stage == "confirm":
        return _audit_stage(run_dir, args, "confirmation")
    raise ValueError(f"unknown executable stage: {stage}")


def _run(args: argparse.Namespace) -> int:
    configure_exact_torch_backend()
    run_dir, resumed = _make_run_dir(args)
    print(f"quartile-specialist run directory: {run_dir}", flush=True)
    initialized = False
    try:
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        if args.stage == "report":
            _report_stage(run_dir, args)
        else:
            for stage in _stage_sequence(args.stage):
                _status(run_dir, state="running", stage=stage)
                gate = _execute_stage(run_dir, args, stage)
                if not _passed(gate):
                    break
        decision, workflow = _commit_workflow(run_dir, args)
        if args.stage != "report" and not str(decision["decision"]).startswith("ready_for_"):
            _write_parent_immutability_after(run_dir, args)
            decision, workflow = _commit_workflow(run_dir, args)
        if args.stage == "all":
            _report_stage(run_dir, args)
            decision, workflow = _commit_workflow(run_dir, args)
        print(
            f"quartile-specialist decision: {decision['decision']}", flush=True
        )
        if args.stage == "report" and int(workflow["required_gate_pass"]) == 1:
            return 0
        if int(workflow["required_gate_pass"]) == 0:
            return decision_exit_code(decision)
        return decision_exit_code(decision)
    except Exception as exc:
        if resumed and not initialized:
            # Resume compatibility is a read-only admission check.  Never
            # mutate an incompatible historical run while rejecting it.
            print(f"quartile-specialist compatibility error: {exc}", flush=True)
            return 1
        stage = args.stage
        if stage == "all":
            gates = _gate_bundle(run_dir)
            stage = next(
                (
                    name
                    for name in (
                        "preflight",
                        "cache",
                        "controls",
                        "train",
                        "calibrate",
                        "select",
                        "confirm",
                    )
                    if not _passed(gates[name])
                ),
                "report",
            )
        _record_execution_failure(run_dir, args, stage, exc)
        print(f"quartile-specialist error: {exc}", flush=True)
        return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact Jacobi/RB quartile-specialist learnability gate"
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-time-local-run-dir", type=Path, required=True)
    parser.add_argument("--parent-memory-v3-run-dir", type=Path, required=True)
    parser.add_argument("--parent-coarse-witness-run-dir", type=Path, required=True)
    parser.add_argument("--parent-bayes-power-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist"),
    )
    parser.add_argument(
        "--run-name",
        default="production-exact-rb-quartile-specialist-time-local",
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
    parser.add_argument(
        "--test-bootstrap-shard-size", type=int, default=4, help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    for name in (
        "parent_time_local_run_dir",
        "parent_memory_v3_run_dir",
        "parent_coarse_witness_run_dir",
        "parent_bayes_power_run_dir",
        "runs_root",
        "resume_run_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.stage not in {"preflight", "all"} and args.resume_run_dir is None:
        parser.error("post-preflight stages require --resume-run-dir")
    if args.test_only and args.require_gate != "none":
        parser.error("test-only runs are nonauthorizing and require --require-gate none")
    if not 1 <= int(args.test_path_count) <= 8:
        parser.error("--test-path-count must lie in [1,8]")
    if not 1 <= int(args.test_outer_steps) <= OUTER_STEPS:
        parser.error(f"--test-outer-steps must lie in [1,{OUTER_STEPS}]")
    if not 0 <= int(args.test_maximum_updates) <= TRAINING["maximum_updates"]:
        parser.error("--test-maximum-updates is out of range")
    if args.test_only and int(args.test_maximum_updates) not in (0,) + CHECKPOINT_UPDATES:
        parser.error("--test-maximum-updates must lie on the frozen checkpoint grid")
    if int(args.test_bootstrap_replicates) <= 0 or int(args.test_bootstrap_shard_size) <= 0:
        parser.error("test bootstrap dimensions must be positive")
    if int(args.test_bootstrap_replicates) % int(args.test_bootstrap_shard_size):
        parser.error("test bootstrap replicates must divide by shard size")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
