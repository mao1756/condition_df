"""Exact fused-lane feasibility gate for the boundary-tangent workflow.

This additive workflow changes CUDA execution packing only.  It benchmarks
the unchanged certified K=512 Jacobi transition, unchanged raw
Rao--Blackwell label, and unchanged width-32 predictor.  It never creates a
scientific cache, trains a model, runs a controller trajectory, reconstructs,
or samples.
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
)
from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    FORBIDDEN_DIAGNOSTICS,
    flatten_midpoint_batches,
    sample_midpoint_branches,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule import (
    MAXIMUM_LAUNCH_LANES,
    PILOT_PROFILE_NAMES,
    PILOT_REPEAT_COUNT,
    PROFILE_CACHE_P10,
    PROFILE_CACHE_P6,
    PROFILE_PATH_COUNTS,
    PROFILE_PATH_IDS,
    PROFILE_STREAM_P10,
    PROFILE_STREAM_P4,
    PROJECTED_BASE_TRANSITIONS,
    PROJECTED_MIDPOINT_TRANSITIONS,
    PROJECTED_TOTAL_TRANSITIONS,
    ROOT_SEED,
    SCHEDULE_VERSION,
    WINDOW_START_STEPS,
    PilotRepeatRecord,
    build_fused_launch_plan,
    expected_profile_transition_counts,
    frozen_production_cohort_plan,
    frozen_repeat_order,
    project_frozen_schedule,
    sample_fused_midpoint_branches,
    split_co_scheduled_payload_by_role,
    validate_repeat_records,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule_provenance import (
    build_schedule_cohort_plan,
    build_schedule_path_plan,
    build_schedule_timing_plan,
    schedule_source_fingerprint,
    schedule_source_paths,
    validate_schedule_cohort_plan,
    validate_schedule_path_plan,
    validate_schedule_timing_plan,
    verify_and_readjudicate_boundary_tangent_schedule_parents,
    verify_schedule_resume_compatibility,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_multipath import run_exact_multipath_shard
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_MATCHINGS,
    PHASE_COUNT,
    STATE_SIZE,
    JacobiRBPhasePredictor,
    ModelInputs,
    call_model,
    enable_deterministic_torch,
    matching_indices,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-schedule-feasibility-v1"
STAGES = ("preflight", "pilot", "report", "all")
REQUIRED_GATES = ("none", "preflight", "pilot")
WINDOW_BRANCH_OFFSET = 15
SHARD_STEPS = 8
NO_WORK = {
    "physical_training_performed": 0,
    "controller_control_trajectory_performed": 0,
    "full_reverse_path_performed": 0,
    "image_sampling_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "reconstruction_performed": 0,
}
NO_AUTHORIZATION = {
    "sampling_authorized": 0,
    "reverse_sampling_authorized": 0,
    "reconstruction_authorized": 0,
}
_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    # These are current-view aliases. Immutable stage-specific copies are
    # registered and verified instead.
    "workflow_gate.json",
    "schedule_decision.json",
}


class BoundaryTangentScheduleCLIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "schedule_execution",
        failure_code: str = "boundary_tangent_schedule_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactCompatibilityError(f"cannot read NPZ artifact: {path}") from exc


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
                **{
                    str(name): np.ascontiguousarray(np.asarray(value))
                    for name, value in sorted(arrays.items())
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(target),
        "sha256": file_fingerprint(target),
        "size": int(target.stat().st_size),
    }


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _arrays_sha(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _freeze_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _load_json(path) != record:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    else:
        atomic_write_json(path, record)
    return record


def _passed(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("evaluation_status") == "evaluated"
        and int(value.get("passed", 0)) == 1
    )


def _not_evaluated(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + f"-{stage}-gate",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": "not_evaluated",
        "reason": str(reason),
        "passed": 0,
        **NO_WORK,
        **NO_AUTHORIZATION,
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
            **NO_WORK,
            **NO_AUTHORIZATION,
        },
    )


def _registry_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(
        (item for item in run_dir.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        relative = path.relative_to(run_dir).as_posix()
        if relative in _REGISTRY_EXCLUDED or relative.endswith(".tmp") or ".tmp." in path.name:
            continue
        artifacts.append(
            {"path": relative, "sha256": file_fingerprint(path), "size": path.stat().st_size}
        )
    return artifacts


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    artifacts = _registry_artifacts(run_dir)
    semantics = {
        "snapshot_kind": "terminal-exact-with-restartable-pilot-extras",
        "excluded_paths": sorted(_REGISTRY_EXCLUDED),
        "restartable_extras_must_match_frozen_pilot_layout": 1,
    }
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "registry_semantics": semantics,
        "semantic_sha256": config_fingerprint(
            {"artifacts": artifacts, "registry_semantics": semantics}
        ),
        **NO_WORK,
        **NO_AUTHORIZATION,
    }
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "root_seed": ROOT_SEED,
        "grid_size": 28,
        "alpha": 1.0,
        "sample_steps": 512,
        "tau_eff": 5.0e-5,
        "training_paths": 64,
        "validation_paths": 32,
        "confirmation_paths": 64,
        "window_start_outer_steps": list(WINDOW_START_STEPS),
        "window_outer_steps": 16,
        "repeat_count": PILOT_REPEAT_COUNT,
        "maximum_launch_lanes": MAXIMUM_LAUNCH_LANES,
        "maximum_projected_exact_cache_hours": 30.0,
        "minimum_effective_projected_rate": PROJECTED_TOTAL_TRANSITIONS / 108_000.0,
        "projected_base_transitions": PROJECTED_BASE_TRANSITIONS,
        "projected_midpoint_transitions": PROJECTED_MIDPOINT_TRANSITIONS,
        "projected_total_transitions": PROJECTED_TOTAL_TRANSITIONS,
        "target": "unchanged exact certified Jacobi Rao-Blackwell label",
        "model": "unchanged width-32 JacobiRBPhasePredictor",
        "workflow_claim": "execution-schedule feasibility only",
        "test_only": int(args.test_only),
        **NO_WORK,
        **NO_AUTHORIZATION,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        root = args.resume_run_dir.resolve()
        if not root.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {root}")
        return root, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (args.runs_root / f"{stamp}_{args.run_name}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    return root, False


def _source_set() -> tuple[Path, ...]:
    sibling = Path(__file__).resolve().parent
    return schedule_source_paths(
        (
            Path(__file__),
            sibling / "d0_jacobi_artifacts.py",
            sibling / "d0_jacobi_source_compat.py",
            sibling / "d0_jacobi_rb_boundary_tangent_cache.py",
            sibling / "d0_jacobi_rb_boundary_tangent_schedule.py",
            sibling / "d0_jacobi_rb_boundary_tangent_schedule_gate.py",
            sibling / "d0_jacobi_rb_boundary_tangent_schedule_provenance.py",
            sibling / "d0_jacobi_rb_coarse_residual.py",
            sibling / "d0_jacobi_rb_controls.py",
            sibling / "d0_jacobi_rb_cuda.py",
            sibling / "d0_jacobi_rb_cuda_certificate.py",
            sibling / "d0_jacobi_rb_cuda_controls.py",
            sibling / "d0_jacobi_rb_cuda_fused.py",
            sibling / "d0_jacobi_rb_cuda_multipath.py",
            sibling / "d0_jacobi_rb_learnability.py",
            sibling / "d0_jacobi_rb_reverse_controller.py",
            sibling / "d0_jacobi_rb_spectral.py",
        )
    )


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    config = _scientific_config(args)
    paths = build_schedule_path_plan()
    cohorts = build_schedule_cohort_plan()
    timing = build_schedule_timing_plan()
    validate_schedule_path_plan(paths)
    validate_schedule_cohort_plan(cohorts)
    validate_schedule_timing_plan(timing)
    source_paths = _source_set()
    source_hash = schedule_source_fingerprint(source_paths)
    parents = verify_and_readjudicate_boundary_tangent_schedule_parents(
        failed_boundary_tangent_run_dir=args.failed_boundary_tangent_run_dir,
        parent_coarse_residual_run_dir=args.parent_coarse_residual_run_dir,
    )
    parent_hash = str(parents["semantic_sha256"])
    if resumed:
        verify_schedule_resume_compatibility(
            run_dir,
            source_fingerprint_value=source_hash,
            scientific_config_sha256=str(config["semantic_sha256"]),
            parent_provenance_sha256=parent_hash,
            path_plan_sha256=str(paths["semantic_sha256"]),
            cohort_plan_sha256=str(cohorts["semantic_sha256"]),
            timing_plan_sha256=str(timing["semantic_sha256"]),
        )
        return
    _freeze_json(run_dir / "parent_provenance.json", parents)
    _freeze_json(
        run_dir / "failed_boundary_tangent_readjudication.json",
        {
            "schema": RUN_SCHEMA + "-failed-parent-readjudication",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": 1,
            "historical_decision": parents["historical_decision"],
            "readjudicated_decision": parents["readjudicated_decision"],
            "historical_failure_domain": parents["historical_failure_domain"],
            "readjudicated_failure_domain": parents["readjudicated_failure_domain"],
            "scientific_evidence_complete": 1,
            "parent_artifacts_mutated": 0,
            **NO_WORK,
            **NO_AUTHORIZATION,
        },
    )
    _freeze_json(run_dir / "scientific_config.json", config)
    _freeze_json(run_dir / "path_id_plan.json", paths)
    _freeze_json(run_dir / "cohort_plan.json", cohorts)
    _freeze_json(run_dir / "timing_plan.json", timing)
    _freeze_json(
        run_dir / "run_manifest.json",
        {
            "schema": RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "device": args.device,
            "source_fingerprint": source_hash,
            "source_paths": [str(path) for path in source_paths],
            "scientific_config_sha256": config["semantic_sha256"],
            "parent_provenance_sha256": parent_hash,
            "path_plan_sha256": paths["semantic_sha256"],
            "cohort_plan_sha256": cohorts["semantic_sha256"],
            "timing_plan_sha256": timing["semantic_sha256"],
            **NO_WORK,
            **NO_AUTHORIZATION,
        },
    )
    _status(run_dir, state="initialized", stage="initialize")


def _semantic_path_collision_scan(run_dir: Path) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import discover_repository_path_id_claims

    plan = _load_json(run_dir / "path_id_plan.json")
    active = {
        int(value)
        for values in plan["roles"].values()
        for value in values
    }
    ignored_sources = {
        Path(__file__).resolve(),
        Path(__file__).with_name("d0_jacobi_rb_boundary_tangent_schedule.py").resolve(),
        Path(__file__).with_name(
            "d0_jacobi_rb_boundary_tangent_schedule_provenance.py"
        ).resolve(),
        (run_dir / "path_id_plan.json").resolve(),
    }
    schedule_parent = _load_json(run_dir / "parent_provenance.json")
    failed_boundary_root = Path(str(schedule_parent["failed_run_dir"])).resolve()
    boundary_parents = _load_json(failed_boundary_root / "parent_provenance.json")
    affine_root = Path(
        str(
            boundary_parents["parents"]["failed_affine_reverse_controller"][
                "run_dir"
            ]
        )
    ).resolve()
    oracle_gate = _load_json(affine_root / "oracle_gate.json")
    oracle_unrealized = bool(
        oracle_gate.get("evaluation_status") == "not_evaluated"
        and int(oracle_gate.get("passed", 0)) == 0
        and not any(
            item.is_file() and item.name != "oracle_gate.json" and item.name.startswith("oracle")
            for item in affine_root.rglob("oracle*")
        )
    )
    oracle_sources = {
        (affine_root / "path_id_plan.json").resolve(),
        Path(__file__).with_name("d0_jacobi_rb_reverse_controller.py").resolve(),
        Path(__file__).with_name("diag_d0_jacobi_rb_reverse_controller.py").resolve(),
    }
    consumed: list[dict[str, Any]] = []
    ignored_stop_metadata: list[dict[str, Any]] = []
    ignored_unrealized_test_only: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    for claim in discover_repository_path_id_claims(Path.cwd()):
        source = Path(claim.source).resolve()
        if source in ignored_sources:
            continue
        if source.name == "path_id_plan.json" and str(claim.name).endswith(
            ".stop_exclusive"
        ):
            # The generic claim scanner pairs any numeric ``*_start`` with a
            # sibling ``*_stop``.  Here ``stop_exclusive`` is metadata inside
            # a role-slot object, not the beginning of a new path interval.
            ignored_stop_metadata.append(
                {
                    "source": str(source),
                    "name": str(claim.name),
                    "start": int(claim.start),
                    "stop_exclusive": int(claim.stop),
                }
            )
            continue
        overlap = sorted(active.intersection(range(int(claim.start), int(claim.stop))))
        if not overlap:
            continue
        if (
            oracle_unrealized
            and source in oracle_sources
            and 0xEE000 <= int(claim.start) < int(claim.stop) <= 0xEE020
            and (
                str(claim.name).startswith("roles.oracle")
                or str(claim.name) == "ORACLE_PATH_IDS"
            )
        ):
            consumed.append(
                {
                    "source": str(source),
                    "name": str(claim.name),
                    "path_ids": overlap,
                    "reservation_start": 0xEE000,
                    "reservation_stop_exclusive": 0xEE020,
                }
            )
            continue
        if source.name == "path_id_plan.json" and source.is_file():
            try:
                prior = _load_json(source)
            except ArtifactCompatibilityError:
                prior = {}
            prior_run = source.parent
            config_path = prior_run / "scientific_config.json"
            workflow_path = prior_run / "workflow_gate.json"
            test_only_unrealized = False
            if config_path.is_file() and workflow_path.is_file():
                try:
                    prior_config = _load_json(config_path)
                    prior_workflow = _load_json(workflow_path)
                    prior_decision = prior_workflow.get("decision", {})
                    test_only_unrealized = bool(
                        int(prior_config.get("test_only", 0)) == 1
                        and int(prior_workflow.get("test_only", 0)) == 1
                        and isinstance(prior_decision, Mapping)
                        and prior_decision.get("decision") == "test_only_complete"
                        and int(
                            prior_decision.get("schedule_integration_authorized", -1)
                        )
                        == 0
                        and not (prior_run / "pilot").exists()
                    )
                except (ArtifactCompatibilityError, TypeError, ValueError):
                    test_only_unrealized = False
            if (
                prior.get("schema") == plan.get("schema")
                and prior.get("roles") == plan.get("roles")
                and not (prior_run / "pilot_gate.json").is_file()
            ):
                continue
            if (
                prior.get("schema") == plan.get("schema")
                and prior.get("roles") == plan.get("roles")
                and test_only_unrealized
            ):
                ignored_unrealized_test_only.append(
                    {
                        "source": str(source),
                        "name": str(claim.name),
                        "path_ids": overlap,
                    }
                )
                continue
        collisions.append(
            {
                "source": str(source),
                "name": str(claim.name),
                "path_ids": overlap,
            }
        )
    record = {
        "schema": RUN_SCHEMA + "-path-collision-scan",
        "schema_version": 1,
        "candidate_path_count": len(active),
        "collision_count": len(collisions),
        "collisions": collisions,
        "consumed_unrealized_oracle_reservation": int(bool(consumed)),
        "consumed_unrealized_oracle_claims": consumed,
        "unrealized_oracle_gate_verified": int(oracle_unrealized),
        "ignored_stop_exclusive_metadata_claims": ignored_stop_metadata,
        "ignored_unrealized_test_only_claims": ignored_unrealized_test_only,
        "passed": int(not collisions),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "path_collision_scan.json", record)
    return record


def _parent_state_files(parent: Path, window_start: int) -> tuple[Path, Path]:
    stem = f"step-{int(window_start):03d}"
    root = parent / "cache" / "validation_shards"
    return (
        root / "cohort-00" / f"{stem}-state.npz",
        root / "cohort-01" / f"{stem}-state.npz",
    )


def _load_benchmark_initial_states(
    parent: Path, *, window_start: int, path_count: int
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 1 <= int(path_count) <= 10:
        raise ValueError("benchmark path count must lie in [1,10]")
    blocks: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for state_path in _parent_state_files(parent, window_start):
        metadata_path = state_path.with_name(state_path.name.replace("-state.npz", ".json"))
        metadata = _load_json(metadata_path)
        if metadata.get("state_file_sha256") != file_fingerprint(state_path):
            raise ArtifactCompatibilityError("parent benchmark state hash changed")
        arrays = _load_npz(state_path)
        states = np.asarray(arrays.get("final_states"))
        if (
            states.dtype != np.float64
            or states.ndim != 2
            or states.shape[1] != STATE_SIZE
            or not np.isfinite(states).all()
            or np.any(states < 0.0)
        ):
            raise ArtifactCompatibilityError("parent benchmark state array is invalid")
        blocks.append(np.ascontiguousarray(states))
        sources.append(
            {
                "path": str(state_path.resolve()),
                "sha256": file_fingerprint(state_path),
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": file_fingerprint(metadata_path),
                "row_count": int(states.shape[0]),
            }
        )
    combined = np.concatenate(blocks, axis=0)[:path_count].copy(order="C")
    if combined.shape != (path_count, STATE_SIZE):
        raise ArtifactCompatibilityError("parent benchmark state rows are incomplete")
    return combined, {
        "window_start": int(window_start),
        "path_count": int(path_count),
        "sources": sources,
        "selected_state_sha256": _array_sha(combined),
    }


def _initial_state_plan(args: argparse.Namespace) -> dict[str, Any]:
    windows = []
    for start in WINDOW_START_STEPS:
        states, evidence = _load_benchmark_initial_states(
            args.parent_coarse_residual_run_dir,
            window_start=start,
            path_count=10,
        )
        evidence["maximum_path_count"] = 10
        evidence["mass_error"] = float(np.max(np.abs(states.sum(axis=1) - 1.0)))
        windows.append(evidence)
    record = {
        "schema": RUN_SCHEMA + "-initial-state-plan",
        "schema_version": 1,
        "source_role": "successful-coarse-parent-validation-shards",
        "windows": windows,
        "passed": int(all(item["mass_error"] <= 2.0e-12 for item in windows)),
        **NO_WORK,
    }
    return record


def _test_sampler(
    head_fraction: torch.Tensor,
    exposure: torch.Tensor,
    *,
    rng_key: Any,
    transition_ids: torch.Tensor,
    profile: JacobiRBCudaProfile,
) -> Any:
    del rng_key, transition_ids, profile
    from types import SimpleNamespace

    later = torch.clamp(
        head_fraction + 0.01 * torch.tanh(exposure) * (0.5 - head_fraction),
        0.0,
        1.0,
    )
    count = int(later.numel())
    return SimpleNamespace(
        later_head_fraction=later,
        denoising_target=later * (1.0 - later) * (0.25 - later),
        certificate_codes=torch.full_like(later, 0b1111, dtype=torch.uint8),
        certified_mask=torch.ones_like(later, dtype=torch.bool),
        active_mask=torch.ones_like(later, dtype=torch.bool),
        strengthened_mask=torch.zeros_like(later, dtype=torch.bool),
        fallback_mask=torch.zeros_like(later, dtype=torch.bool),
        mode_counts=torch.full_like(later, 16, dtype=torch.int32),
        prefix_bits=torch.full_like(later, 64, dtype=torch.int32),
        arb_fallback_reason_codes=torch.zeros_like(later, dtype=torch.uint8),
        diagnostics={
            "active_count": count,
            "certified_count": count,
            "fallback_count": 0,
            "strengthened_count": 0,
            "maximum_mode_count": 16,
            "maximum_prefix_bits": 64,
            "maximum_cuda_launch_lanes": count,
            "fused_authorizer_launch_count": 1,
            "arb_fallback_elapsed_seconds": 0.0,
            "fused_authorizer_elapsed_seconds": 0.0,
            "candidate_elapsed_seconds": 0.0,
            **{name: 0 for name in FORBIDDEN_DIAGNOSTICS},
        },
    )


def _sampler_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {"sampler": _test_sampler} if args.test_only else {}


def _preflight_equivalence(
    args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    path_plan = build_schedule_path_plan()
    path_ids = tuple(int(value) for value in path_plan["roles"]["cuda_warmup"])
    initial, source = _load_benchmark_initial_states(
        args.parent_coarse_residual_run_dir, window_start=0, path_count=len(path_ids)
    )
    states = torch.as_tensor(
        np.array(initial, copy=True, order="C"), dtype=torch.float64, device=device
    ).contiguous()
    profile = JacobiRBCudaProfile()
    kwargs = _sampler_kwargs(args)
    packed = run_exact_multipath_shard(
        states,
        path_ids=path_ids,
        start_step=0,
        root_seed=ROOT_SEED,
        profile=profile,
        group_sizes=(len(path_ids),),
        capture_training_payload=True,
        **kwargs,
    )
    singleton = run_exact_multipath_shard(
        states,
        path_ids=path_ids,
        start_step=0,
        root_seed=ROOT_SEED,
        profile=profile,
        group_sizes=(1,) * len(path_ids),
        capture_training_payload=True,
        **kwargs,
    )
    permutation = tuple(reversed(range(len(path_ids))))
    permuted = run_exact_multipath_shard(
        states.index_select(
            0, torch.as_tensor(permutation, dtype=torch.long, device=device)
        ).contiguous(),
        path_ids=tuple(path_ids[index] for index in permutation),
        start_step=0,
        root_seed=ROOT_SEED,
        profile=profile,
        group_sizes=(len(path_ids),),
        capture_training_payload=True,
        **kwargs,
    )
    packed_capture = packed.capture_payload
    singleton_capture = singleton.capture_payload
    base_equal = bool(
        packed.batch_output_sha256 == singleton.batch_output_sha256
        and packed.batch_final_state_sha256 == singleton.batch_final_state_sha256
        and packed_capture is not None
        and singleton_capture is not None
        and np.array_equal(
            packed_capture.denoising_targets, singleton_capture.denoising_targets
        )
        and np.array_equal(
            packed_capture.certificate_codes, singleton_capture.certificate_codes
        )
        and np.array_equal(
            packed_capture.post_phase_states, singleton_capture.post_phase_states
        )
    )
    permutation_equal = bool(
        packed.batch_output_sha256 == permuted.batch_output_sha256
        and packed.batch_final_state_sha256 == permuted.batch_final_state_sha256
        and packed.batch_certificate_sha256 == permuted.batch_certificate_sha256
    )

    fused_records = []
    legacy_hash = hashlib.sha256()
    fused_hash = hashlib.sha256()
    branch_equal = True
    max_lanes = 0
    for phase in range(PHASE_COUNT):
        fused = sample_fused_midpoint_branches(
            states,
            path_ids=path_ids,
            outer_step=15,
            phase=phase,
            root_seed=ROOT_SEED,
            profile=profile,
            **kwargs,
        )
        legacy = sample_midpoint_branches(
            states,
            path_ids=path_ids,
            outer_step=15,
            phase=phase,
            root_seed=ROOT_SEED,
            profile=profile,
            **kwargs,
        )
        equal = bool(
            torch.equal(fused.batch.later_full_state, legacy.later_full_state)
            and torch.equal(fused.batch.denoising_target, legacy.denoising_target)
            and torch.equal(fused.batch.certificate_codes, legacy.certificate_codes)
        )
        branch_equal = branch_equal and equal
        legacy_hash.update(legacy.output_sha256().encode("ascii"))
        fused_hash.update(fused.output_sha256().encode("ascii"))
        max_lanes = max(max_lanes, fused.launch_plan.maximum_chunk_lanes)
        fused_records.append(fused.to_record())

    production = frozen_production_cohort_plan()
    roles = production["path_roles"]
    firewall_paths = tuple(range(0xEC13C, 0xEC140)) + tuple(range(0xEC200, 0xEC206))
    firewall_payload = {
        "path_id": np.asarray(firewall_paths, dtype=np.int64),
        "value": np.arange(len(firewall_paths), dtype=np.float64)[:, None],
    }
    split_payload = split_co_scheduled_payload_by_role(
        firewall_paths, firewall_payload
    )
    cross_role_valid = bool(
        production["cross_role_artifact_commit"] == 0
        and sum(value == "train" for value in roles.values()) == 64
        and sum(value == "validation" for value in roles.values()) == 32
        and sum(value == "confirmation" for value in roles.values()) == 64
        and set(split_payload) == {"train", "validation"}
        and np.array_equal(
            split_payload["train"]["path_id"], np.asarray(firewall_paths[:4])
        )
        and np.array_equal(
            split_payload["validation"]["path_id"], np.asarray(firewall_paths[4:])
        )
    )
    return {
        "schema": RUN_SCHEMA + "-schedule-equivalence-preflight",
        "schema_version": 1,
        "parent_initial_state": source,
        "base_packed_output_sha256": packed.batch_output_sha256,
        "base_singleton_output_sha256": singleton.batch_output_sha256,
        "base_packed_final_state_sha256": packed.batch_final_state_sha256,
        "base_singleton_final_state_sha256": singleton.batch_final_state_sha256,
        "base_equivalence_valid": int(base_equal),
        "path_permutation_invariance_valid": int(permutation_equal),
        "legacy_branch_sha256": legacy_hash.hexdigest(),
        "fused_branch_sha256": fused_hash.hexdigest(),
        "fused_branch_equivalence_valid": int(
            branch_equal and legacy_hash.digest() == fused_hash.digest()
        ),
        "maximum_observed_launch_lanes": int(max_lanes),
        "launch_lane_cap_valid": int(max_lanes <= MAXIMUM_LAUNCH_LANES),
        "cross_role_isolation_valid": int(cross_role_valid),
        "fused_phase_records": fused_records,
        "passed": int(
            base_equal
            and permutation_equal
            and branch_equal
            and legacy_hash.digest() == fused_hash.digest()
            and max_lanes <= MAXIMUM_LAUNCH_LANES
            and cross_role_valid
        ),
        **NO_WORK,
    }


def _sealed_stage(
    run_dir: Path, *, gate_name: str, seal_name: str
) -> dict[str, Any] | None:
    gate_path = run_dir / gate_name
    seal_path = run_dir / seal_name
    if not gate_path.is_file():
        if seal_path.is_file():
            raise ArtifactCompatibilityError(
                f"orphan {seal_name} exists without {gate_name}"
            )
        return None
    if not seal_path.is_file():
        return None
    seal = _load_json(seal_path)
    body = dict(seal)
    semantic = body.pop("semantic_sha256", None)
    if semantic != config_fingerprint(body) or not isinstance(seal.get("artifacts"), list):
        raise ArtifactCompatibilityError(f"{seal_name} changed")
    for item in seal["artifacts"]:
        path = run_dir / str(item["path"])
        if not path.is_file() or item.get("sha256") != file_fingerprint(path):
            raise ArtifactCompatibilityError(f"sealed artifact changed: {item.get('path')}")
        if path.name == "pilot_shard_registry.json":
            registry = _load_json(path)
            for child in registry.get("artifacts", []):
                child_path = run_dir / str(child["path"])
                if (
                    not child_path.is_file()
                    or child.get("sha256") != file_fingerprint(child_path)
                    or int(child.get("size", -1)) != child_path.stat().st_size
                ):
                    raise ArtifactCompatibilityError(
                        f"pilot shard registry changed: {child.get('path')}"
                    )
    return _load_json(gate_path)


def _seal_stage(run_dir: Path, names: Sequence[str], seal_name: str) -> None:
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-artifact-seal",
        "schema_version": 1,
        "artifacts": [
            {"path": name, "sha256": file_fingerprint(run_dir / name)} for name in names
        ],
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(run_dir / seal_name, record)


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _sealed_stage(
        run_dir, gate_name="preflight_gate.json", seal_name="preflight_artifact_seal.json"
    )
    if existing is not None:
        return existing
    if not args.test_only:
        configure_exact_torch_backend(args.device)
        if args.device != "cuda" or not torch.cuda.is_available():
            raise BoundaryTangentScheduleCLIError(
                "authorizing preflight requires CUDA",
                failure_domain="resource_gate",
                failure_code="boundary_tangent_schedule_cuda_unavailable",
            )
    provenance = _load_json(run_dir / "parent_provenance.json")
    readjudication = _load_json(run_dir / "failed_boundary_tangent_readjudication.json")
    path_validation = validate_schedule_path_plan(_load_json(run_dir / "path_id_plan.json"))
    cohort_validation = validate_schedule_cohort_plan(_load_json(run_dir / "cohort_plan.json"))
    timing_validation = validate_schedule_timing_plan(_load_json(run_dir / "timing_plan.json"))
    collision = _semantic_path_collision_scan(run_dir)
    initial_plan = _initial_state_plan(args)
    atomic_write_json(run_dir / "benchmark_initial_state_plan.json", initial_plan)
    launch_plans = {
        name: build_fused_launch_plan(PROFILE_PATH_COUNTS[name]).to_record()
        for name in PILOT_PROFILE_NAMES
    }
    launch_record = {
        "schema": RUN_SCHEMA + "-launch-packing-plan",
        "schema_version": 1,
        "plans": launch_plans,
        "passed": int(
            all(
                int(item["maximum_observed_chunk_lanes"]) <= MAXIMUM_LAUNCH_LANES
                for item in launch_plans.values()
            )
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "launch_packing_plan.json", launch_record)
    equivalence = _preflight_equivalence(args, torch.device(args.device))
    atomic_write_json(run_dir / "schedule_equivalence_preflight.json", equivalence)
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "provenance_valid": int(provenance.get("passed", 0)),
        "readjudication_valid": int(readjudication.get("passed", 0)),
        "parent_sources_immutable": int(provenance.get("passed", 0)),
        "path_plan_valid": int(path_validation.get("passed", 0)),
        "cohort_plan_valid": int(cohort_validation.get("passed", 0)),
        "timing_plan_valid": int(timing_validation.get("passed", 0)),
        "path_collision_free": int(collision.get("passed", 0)),
        "initial_states_valid": int(initial_plan.get("passed", 0)),
        "launch_plan_valid": int(launch_record.get("passed", 0)),
        "base_equivalence_valid": int(equivalence["base_equivalence_valid"]),
        "fused_branch_equivalence_valid": int(
            equivalence["fused_branch_equivalence_valid"]
        ),
        "cross_role_isolation_valid": int(equivalence["cross_role_isolation_valid"]),
        "no_work_valid": 1,
        "canonical_id_uniqueness_valid": int(path_validation.get("passed", 0)),
        "canonical_id_order_invariance_valid": int(
            equivalence["base_equivalence_valid"]
        ),
        "p10_singleton_equivalence_valid": int(
            equivalence["base_equivalence_valid"]
        ),
        "path_permutation_invariance_valid": int(
            equivalence["path_permutation_invariance_valid"]
        ),
        "chunk_invariance_valid": int(
            equivalence["fused_branch_equivalence_valid"]
        ),
        "repeat_rotation_valid": int(
            tuple(frozen_repeat_order(index) for index in range(PILOT_REPEAT_COUNT))
            == tuple(
                tuple(value)
                for value in _load_json(run_dir / "timing_plan.json")[
                    "repeat_profile_orders"
                ]
            )
        ),
        "atomic_commit_plan_valid": 1,
        "failed_parent_record_count": int(
            provenance["failed_registry"]["artifact_count"]
        ),
        "root_seed": ROOT_SEED,
        "cache_group_sizes": _load_json(run_dir / "cohort_plan.json")["production"][
            "train_validation"
        ]["group_sizes"],
        "stream_group_sizes": _load_json(run_dir / "cohort_plan.json")["production"][
            "confirmation"
        ]["group_sizes"],
        "timing_window_starts": list(WINDOW_START_STEPS),
        "timing_branch_steps": [start + WINDOW_BRANCH_OFFSET for start in WINDOW_START_STEPS],
        "timing_window_outer_steps": 16,
        "pilot_repeats": PILOT_REPEAT_COUNT,
        "restart_outer_steps": SHARD_STEPS,
        "maximum_launch_lanes": int(equivalence["maximum_observed_launch_lanes"]),
        "profile_transition_counts": {
            name: expected_profile_transition_counts(name)[2]
            for name in PILOT_PROFILE_NAMES
        },
        "base_transition_count": PROJECTED_BASE_TRANSITIONS,
        "midpoint_transition_count": PROJECTED_MIDPOINT_TRANSITIONS,
        "projected_transition_count": PROJECTED_TOTAL_TRANSITIONS,
        "maximum_projected_exact_cache_hours": 30.0,
        "transition_count_algebra": int(
            PROJECTED_BASE_TRANSITIONS + PROJECTED_MIDPOINT_TRANSITIONS
            == PROJECTED_TOTAL_TRANSITIONS
        ),
        "maximum_observed_launch_lanes": int(
            equivalence["maximum_observed_launch_lanes"]
        ),
        "scientific_target_changed": 0,
        "production_cache_generated": 0,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "schedule_preflight_metrics.json", metrics)
    from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import evaluate_schedule_preflight

    gate = evaluate_schedule_preflight(metrics)
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    names = (
        "parent_provenance.json",
        "failed_boundary_tangent_readjudication.json",
        "path_id_plan.json",
        "cohort_plan.json",
        "timing_plan.json",
        "path_collision_scan.json",
        "benchmark_initial_state_plan.json",
        "launch_packing_plan.json",
        "schedule_equivalence_preflight.json",
        "schedule_preflight_metrics.json",
        "preflight_gate.json",
    )
    _seal_stage(run_dir, names, "preflight_artifact_seal.json")
    return gate


def _model_inputs(arrays: Mapping[str, np.ndarray], device: torch.device) -> ModelInputs:
    return ModelInputs(
        later_full_state=torch.as_tensor(
            np.array(arrays["later_full_state"], copy=True, order="C"),
            dtype=torch.float32,
            device=device,
        ).contiguous(),
        reverse_time=torch.as_tensor(
            np.array(arrays["reverse_time"], copy=True),
            dtype=torch.float64,
            device=device,
        ),
        phase=torch.as_tensor(
            np.array(arrays["phase"], copy=True), dtype=torch.long, device=device
        ),
        color=torch.as_tensor(
            np.array(arrays["color"], copy=True), dtype=torch.long, device=device
        ),
        duration=torch.as_tensor(
            np.array(arrays["duration"], copy=True),
            dtype=torch.float32,
            device=device,
        ),
        label=torch.as_tensor(
            np.array(arrays["label"], copy=True), dtype=torch.long, device=device
        ),
    )


def _capture_pre_phase_states(result: Any, branch_step: int) -> np.ndarray:
    capture = result.capture_payload
    if capture is None:
        raise BoundaryTangentScheduleCLIError(
            "branch shard omitted its phase-state capture",
            failure_code="boundary_tangent_schedule_capture_missing",
        )
    local = int(branch_step) - int(capture.start_step)
    if local != 7:
        raise BoundaryTangentScheduleCLIError(
            "frozen branch step is not the last local shard step",
            failure_code="boundary_tangent_schedule_branch_offset_invalid",
        )
    trace = np.asarray(capture.post_phase_states, dtype=np.float64)
    pre = np.stack(
        [
            trace[local * PHASE_COUNT + phase - 1]
            if phase
            else trace[local * PHASE_COUNT - 1]
            for phase in range(PHASE_COUNT)
        ]
    )
    return np.ascontiguousarray(pre)


def _shard_paths(
    root: Path, *, window_start: int, shard_start: int
) -> tuple[Path, Path, Path]:
    directory = root / f"window-{window_start:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step-{shard_start:03d}"
    return (
        directory / f"{stem}-state.npz",
        directory / f"{stem}-capture.npz",
        directory / f"{stem}.json",
    )


def _valid_base_shard(
    root: Path,
    *,
    profile_name: str,
    repeat_index: int,
    path_ids: Sequence[int],
    window_start: int,
    shard_start: int,
    input_states: np.ndarray,
    capture_expected: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]] | None:
    state_path, capture_path, metadata_path = _shard_paths(
        root, window_start=window_start, shard_start=shard_start
    )
    if not state_path.is_file() or not metadata_path.is_file():
        return None
    try:
        record = _load_json(metadata_path)
        body = dict(record)
        semantic = body.pop("semantic_sha256", None)
        expected = {
            "schema": RUN_SCHEMA + "-pilot-base-shard",
            "schema_version": 1,
            "profile_name": profile_name,
            "repeat_index": int(repeat_index),
            "path_ids": list(path_ids),
            "window_start": int(window_start),
            "shard_start": int(shard_start),
            "capture_expected": int(capture_expected),
            "input_state_sha256": _array_sha(input_states),
        }
        if semantic != config_fingerprint(body) or any(
            record.get(name) != value for name, value in expected.items()
        ):
            return None
        if record.get("state_file_sha256") != file_fingerprint(state_path):
            return None
        marker_path = metadata_path.with_suffix(".commit.json")
        if (
            not marker_path.is_file()
            or record.get("commit_marker_sha256") != file_fingerprint(marker_path)
        ):
            return None
        final = np.asarray(_load_npz(state_path).get("final_states"))
        if (
            final.dtype != np.float64
            or final.shape != input_states.shape
            or record.get("final_state_sha256") != _array_sha(final)
        ):
            return None
        pre = None
        if capture_expected:
            if (
                not capture_path.is_file()
                or record.get("capture_file_sha256") != file_fingerprint(capture_path)
            ):
                return None
            pre = np.asarray(_load_npz(capture_path).get("pre_phase_states"))
            if pre.dtype != np.float64 or pre.shape != (
                PHASE_COUNT,
                len(path_ids),
                STATE_SIZE,
            ):
                return None
        return np.ascontiguousarray(final), None if pre is None else np.ascontiguousarray(pre), record
    except (ArtifactCompatibilityError, OSError, ValueError, TypeError, KeyError):
        return None


def _run_base_shard(
    root: Path,
    args: argparse.Namespace,
    *,
    profile_name: str,
    repeat_index: int,
    path_ids: Sequence[int],
    window_start: int,
    shard_start: int,
    input_states: np.ndarray,
    capture_expected: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    cached = _valid_base_shard(
        root,
        profile_name=profile_name,
        repeat_index=repeat_index,
        path_ids=path_ids,
        window_start=window_start,
        shard_start=shard_start,
        input_states=input_states,
        capture_expected=capture_expected,
    )
    if cached is not None:
        return cached
    started = time.perf_counter()
    states = torch.as_tensor(
        np.array(input_states, copy=True, order="C"),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    result = run_exact_multipath_shard(
        states,
        path_ids=path_ids,
        start_step=shard_start,
        root_seed=ROOT_SEED,
        profile=JacobiRBCudaProfile(),
        group_sizes=(len(path_ids),),
        capture_training_payload=capture_expected,
        **_sampler_kwargs(args),
    )
    final = np.ascontiguousarray(result.committed_final_states, dtype=np.float64)
    pre = (
        _capture_pre_phase_states(result, window_start + WINDOW_BRANCH_OFFSET)
        if capture_expected
        else None
    )
    state_path, capture_path, metadata_path = _shard_paths(
        root, window_start=window_start, shard_start=shard_start
    )
    state_artifact = _atomic_npz(state_path, {"final_states": final})
    capture_artifact = (
        _atomic_npz(capture_path, {"pre_phase_states": pre}) if pre is not None else None
    )
    marker_path = metadata_path.with_suffix(".commit.json")
    atomic_write_json(
        marker_path,
        {
            "schema": RUN_SCHEMA + "-pilot-base-shard-commit",
            "profile_name": profile_name,
            "repeat_index": int(repeat_index),
            "window_start": int(window_start),
            "shard_start": int(shard_start),
            "state_file_sha256": state_artifact["sha256"],
            "capture_file_sha256": None if capture_artifact is None else capture_artifact["sha256"],
        },
    )
    elapsed = time.perf_counter() - started
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-pilot-base-shard",
        "schema_version": 1,
        "profile_name": profile_name,
        "repeat_index": int(repeat_index),
        "path_ids": list(path_ids),
        "window_start": int(window_start),
        "shard_start": int(shard_start),
        "capture_expected": int(capture_expected),
        "input_state_sha256": _array_sha(input_states),
        "final_state_sha256": _array_sha(final),
        "state_file_sha256": state_artifact["sha256"],
        "state_file_size": state_artifact["size"],
        "capture_file_sha256": None if capture_artifact is None else capture_artifact["sha256"],
        "capture_file_size": None if capture_artifact is None else capture_artifact["size"],
        "commit_marker_sha256": file_fingerprint(marker_path),
        "scheduler_record": result.to_record(),
        "complete_pipeline_elapsed_seconds": elapsed,
        "committed": 1,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(metadata_path, record)
    return final, pre, record


def _branch_commit_paths(root: Path, window_start: int) -> tuple[Path, Path, Path]:
    directory = root / f"window-{window_start:03d}"
    return directory / "branch-data.npz", directory / "branch-audit.npz", directory / "branch.json"


def _branch_conservation_path(root: Path, window_start: int) -> Path:
    return root / f"window-{window_start:03d}" / "branch-conservation.npz"


def _branch_conservation_arrays(
    batches: Sequence[Any], pre_phase_states: np.ndarray
) -> dict[str, np.ndarray]:
    """Return exact per-midpoint/path mass errors before float32 conversion."""

    before = np.ascontiguousarray(np.asarray(pre_phase_states, dtype=np.float64))
    if before.shape != (PHASE_COUNT, len(batches[0].batch.path_ids), STATE_SIZE):
        raise BoundaryTangentScheduleCLIError(
            "branch conservation input shape changed",
            failure_code="boundary_tangent_schedule_branch_conservation_invalid",
        )
    tails_all, heads_all = matching_indices(device="cpu")
    tails_np = tails_all.numpy()
    heads_np = heads_all.numpy()
    global_errors: list[np.ndarray] = []
    pair_errors: list[np.ndarray] = []
    for phase, item in enumerate(batches):
        later = np.ascontiguousarray(
            item.batch.later_full_state.detach().cpu().numpy(), dtype=np.float64
        )
        if later.shape != (8, before.shape[1], STATE_SIZE):
            raise BoundaryTangentScheduleCLIError(
                "branch conservation output shape changed",
                failure_code="boundary_tangent_schedule_branch_conservation_invalid",
            )
        reference = before[phase]
        global_errors.append(
            np.abs(later.sum(axis=2) - reference.sum(axis=1)[None, :])
        )
        matching = int(PHASE_MATCHINGS[phase])
        tails = tails_np[matching]
        heads = heads_np[matching]
        reference_pairs = reference[:, tails] + reference[:, heads]
        later_pairs = later[:, :, tails] + later[:, :, heads]
        pair_errors.append(
            np.max(np.abs(later_pairs - reference_pairs[None, :, :]), axis=2)
        )
    return {
        "global_simplex_error": np.ascontiguousarray(
            np.stack(global_errors), dtype=np.float64
        ),
        "maximum_matched_pair_error": np.ascontiguousarray(
            np.stack(pair_errors), dtype=np.float64
        ),
    }


def _maximum_branch_mass_error(values: Mapping[str, np.ndarray]) -> float:
    required = {"global_simplex_error", "maximum_matched_pair_error"}
    if set(values) != required:
        raise ArtifactCompatibilityError("branch conservation fields changed")
    maxima: list[float] = []
    for name in sorted(required):
        value = np.asarray(values[name])
        if value.dtype != np.float64 or value.ndim != 3 or not np.isfinite(value).all():
            raise ArtifactCompatibilityError("branch conservation diagnostics are malformed")
        if np.any(value < 0.0):
            raise ArtifactCompatibilityError("branch conservation error is negative")
        maxima.append(float(np.max(value, initial=0.0)))
    return max(maxima, default=0.0)


def _valid_branch_commit(
    root: Path,
    *,
    profile_name: str,
    repeat_index: int,
    window_start: int,
    path_ids: Sequence[int] | None = None,
    pre_phase_states: np.ndarray | None = None,
) -> dict[str, Any] | None:
    data_path, audit_path, metadata_path = _branch_commit_paths(root, window_start)
    conservation_path = _branch_conservation_path(root, window_start)
    if not all(
        path.is_file()
        for path in (data_path, audit_path, conservation_path, metadata_path)
    ):
        return None
    try:
        record = _load_json(metadata_path)
        body = dict(record)
        semantic = body.pop("semantic_sha256", None)
        stream_mode = profile_name in {PROFILE_STREAM_P10, PROFILE_STREAM_P4}
        expected_mode = "stream" if stream_mode else "cache"
        if (
            semantic != config_fingerprint(body)
            or record.get("profile_name") != profile_name
            or int(record.get("repeat_index", -1)) != repeat_index
            or int(record.get("window_start", -1)) != window_start
            or (
                path_ids is not None
                and record.get("path_ids") != [int(value) for value in path_ids]
            )
            or (
                pre_phase_states is not None
                and record.get("pre_phase_state_sha256")
                != _array_sha(pre_phase_states)
            )
            or record.get("data_file_sha256") != file_fingerprint(data_path)
            or int(record.get("data_file_size", -1)) != data_path.stat().st_size
            or record.get("audit_file_sha256") != file_fingerprint(audit_path)
            or int(record.get("audit_file_size", -1)) != audit_path.stat().st_size
            or record.get("conservation_file_sha256")
            != file_fingerprint(conservation_path)
            or int(record.get("conservation_file_size", -1))
            != conservation_path.stat().st_size
            or record.get("mode") != expected_mode
            or int(record.get("raw_float64_label_conversion", 0)) != 1
            or int(record.get("permitted_float32_input_conversion", 0)) != 1
            or int(record.get("target_transformed", -1)) != 0
            or int(record.get("width32_forward_performed", -1)) != int(stream_mode)
            or record.get("risk_accumulation_device")
            != ("cuda" if stream_mode else "not_applicable")
            or int(record.get("risk_host_transfer_count", -1)) != int(stream_mode)
        ):
            return None
        data_arrays = _load_npz(data_path)
        audit_arrays = _load_npz(audit_path)
        conservation = _load_npz(conservation_path)
        if (
            record.get("data_semantic_sha256") != _arrays_sha(data_arrays)
            or record.get("audit_semantic_sha256") != _arrays_sha(audit_arrays)
            or record.get("conservation_semantic_sha256")
            != _arrays_sha(conservation)
            or float(record.get("maximum_mass_error", math.inf))
            != _maximum_branch_mass_error(conservation)
        ):
            return None
        marker_path = metadata_path.with_suffix(".commit.json")
        if (
            not marker_path.is_file()
            or record.get("commit_marker_sha256") != file_fingerprint(marker_path)
        ):
            return None
        return record
    except (ArtifactCompatibilityError, ValueError, TypeError, OSError):
        return None


def _run_branch_commit(
    root: Path,
    args: argparse.Namespace,
    *,
    profile_name: str,
    repeat_index: int,
    path_ids: Sequence[int],
    window_start: int,
    pre_phase_states: np.ndarray,
    device: torch.device,
    model: JacobiRBPhasePredictor,
) -> dict[str, Any]:
    cached = _valid_branch_commit(
        root,
        profile_name=profile_name,
        repeat_index=repeat_index,
        window_start=window_start,
        path_ids=path_ids,
        pre_phase_states=pre_phase_states,
    )
    if cached is not None:
        return cached
    started = time.perf_counter()
    batches = []
    for phase in range(PHASE_COUNT):
        state = torch.as_tensor(
            np.array(pre_phase_states[phase], copy=True, order="C"),
            dtype=torch.float64,
            device=device,
        ).contiguous()
        batches.append(
            sample_fused_midpoint_branches(
                state,
                path_ids=path_ids,
                outer_step=window_start + WINDOW_BRANCH_OFFSET,
                phase=phase,
                root_seed=ROOT_SEED,
                profile=JacobiRBCudaProfile(),
                **_sampler_kwargs(args),
            )
        )
    permitted, audit = flatten_midpoint_batches([item.batch for item in batches])
    conservation = _branch_conservation_arrays(batches, pre_phase_states)
    maximum_mass_error = _maximum_branch_mass_error(conservation)
    permitted["later_full_state"] = np.ascontiguousarray(
        permitted["later_full_state"], dtype=np.float32
    )
    data_path, audit_path, metadata_path = _branch_commit_paths(root, window_start)
    conservation_path = _branch_conservation_path(root, window_start)
    risk_accumulation_device = "not_applicable"
    risk_host_transfer_count = 0
    if profile_name in {PROFILE_CACHE_P10, PROFILE_CACHE_P6}:
        data_arrays = permitted
        audit_arrays = audit
    else:
        inputs = _model_inputs(permitted, device)
        prediction_blocks = []
        with torch.no_grad():
            for start in range(0, inputs.batch_size, 32):
                indices = torch.arange(
                    start,
                    min(start + 32, inputs.batch_size),
                    dtype=torch.long,
                    device=device,
                )
                prediction_blocks.append(call_model(model, inputs.index_select(indices)))
        prediction = torch.cat(prediction_blocks).to(dtype=torch.float64)
        target = torch.as_tensor(
            np.array(audit["denoising_target"], copy=True),
            dtype=torch.float64,
            device=device,
        )
        row_risk = torch.mean((prediction - target).square(), dim=1)
        risk_accumulation_device = str(row_risk.device.type)
        path_values = torch.as_tensor(
            np.array(audit["path_id"], copy=True), dtype=torch.long, device=device
        )
        per_path_tensor = torch.stack(
            [torch.mean(row_risk[path_values == int(path_id)]) for path_id in path_ids]
        )
        per_path = np.ascontiguousarray(
            per_path_tensor.detach().cpu().numpy(), dtype=np.float64
        )
        risk_host_transfer_count = 1
        data_arrays = {
            "sample_key": permitted["sample_key"],
            "later_full_state": permitted["later_full_state"],
        }
        audit_arrays = {
            "path_ids": np.asarray(path_ids, dtype=np.int64),
            "per_path_risk": per_path,
            "prediction_sha256_bytes": np.frombuffer(
                hashlib.sha256(
                    np.ascontiguousarray(prediction.detach().cpu().numpy()).tobytes(order="C")
                ).digest(),
                dtype=np.uint8,
            ),
        }
    data_artifact = _atomic_npz(data_path, data_arrays)
    audit_artifact = _atomic_npz(audit_path, audit_arrays)
    conservation_artifact = _atomic_npz(conservation_path, conservation)
    data_semantic_sha256 = _arrays_sha(data_arrays)
    audit_semantic_sha256 = _arrays_sha(audit_arrays)
    conservation_semantic_sha256 = _arrays_sha(conservation)
    transition_count = sum(item.batch.transition_count for item in batches)
    certified_count = sum(item.batch.certified_count for item in batches)
    fallback_count = sum(int(item.batch.fallback_mask.sum().item()) for item in batches)
    fallback_elapsed = sum(float(item.batch.fallback_elapsed_seconds) for item in batches)
    forbidden = {
        name: sum(int(item.batch.forbidden_counts[name]) for item in batches)
        for name in FORBIDDEN_DIAGNOSTICS
    }
    output_digest = hashlib.sha256()
    for item in batches:
        output_digest.update(item.output_sha256().encode("ascii"))
    output_digest.update(data_semantic_sha256.encode("ascii"))
    output_digest.update(audit_semantic_sha256.encode("ascii"))
    output_digest.update(conservation_semantic_sha256.encode("ascii"))
    marker_path = metadata_path.with_suffix(".commit.json")
    atomic_write_json(
        marker_path,
        {
            "schema": RUN_SCHEMA + "-pilot-branch-commit-marker",
            "profile_name": profile_name,
            "repeat_index": int(repeat_index),
            "window_start": int(window_start),
            "data_file_sha256": data_artifact["sha256"],
            "audit_file_sha256": audit_artifact["sha256"],
            "conservation_file_sha256": conservation_artifact["sha256"],
            "output_sha256": output_digest.hexdigest(),
        },
    )
    measured_elapsed = time.perf_counter() - started
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-pilot-branch-commit",
        "schema_version": 1,
        "profile_name": profile_name,
        "repeat_index": int(repeat_index),
        "window_start": int(window_start),
        "path_ids": list(path_ids),
        "pre_phase_state_sha256": _array_sha(pre_phase_states),
        "mode": "cache" if profile_name.startswith("cache_") else "stream",
        "transition_count": transition_count,
        "certified_count": certified_count,
        "fallback_count": fallback_count,
        "fallback_elapsed_seconds": fallback_elapsed,
        "forbidden_counts": forbidden,
        "maximum_launch_lanes": max(
            item.launch_plan.maximum_chunk_lanes for item in batches
        ),
        "output_sha256": output_digest.hexdigest(),
        "data_file_sha256": data_artifact["sha256"],
        "data_file_size": data_artifact["size"],
        "data_semantic_sha256": data_semantic_sha256,
        "audit_file_sha256": audit_artifact["sha256"],
        "audit_file_size": audit_artifact["size"],
        "audit_semantic_sha256": audit_semantic_sha256,
        "conservation_file_sha256": conservation_artifact["sha256"],
        "conservation_file_size": conservation_artifact["size"],
        "conservation_semantic_sha256": conservation_semantic_sha256,
        "maximum_mass_error": maximum_mass_error,
        "commit_marker_sha256": file_fingerprint(marker_path),
        "complete_pipeline_elapsed_seconds": measured_elapsed,
        "raw_float64_label_conversion": 1,
        "permitted_float32_input_conversion": 1,
        "width32_forward_performed": int(profile_name.startswith("stream_")),
        "risk_accumulation_device": risk_accumulation_device,
        "risk_host_transfer_count": risk_host_transfer_count,
        "target_transformed": 0,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(metadata_path, record)
    return record


def _profile_repeat_root(run_dir: Path, profile_name: str, repeat_index: int) -> Path:
    return run_dir / "pilot" / profile_name / f"repeat-{repeat_index:02d}"


def _record_from_json(value: Mapping[str, Any]) -> PilotRepeatRecord:
    return PilotRepeatRecord(
        profile_name=str(value["profile_name"]),
        repeat_index=int(value["repeat_index"]),
        execution_order_index=int(value["execution_order_index"]),
        elapsed_seconds=float(value["elapsed_seconds"]),
        base_transition_count=int(value["base_transition_count"]),
        midpoint_transition_count=int(value["midpoint_transition_count"]),
        certified_count=int(value["certified_count"]),
        fallback_count=int(value["fallback_count"]),
        fallback_elapsed_seconds=float(value["fallback_elapsed_seconds"]),
        maximum_mass_error=float(value["maximum_mass_error"]),
        peak_memory_fraction=float(value["peak_memory_fraction"]),
        committed_bytes=int(value["committed_bytes"]),
        maximum_launch_lanes=int(value["maximum_launch_lanes"]),
        output_sha256=str(value["output_sha256"]),
        final_state_sha256=str(value["final_state_sha256"]),
        forbidden_counts={
            name: int(value["forbidden_counts"][name]) for name in FORBIDDEN_DIAGNOSTICS
        },
    )


def _load_completed_repeat(path: Path) -> PilotRepeatRecord | None:
    if not path.is_file():
        return None
    try:
        value = _load_json(path)
        body = dict(value)
        semantic = body.pop("semantic_sha256", None)
        if semantic != config_fingerprint(body):
            return None
        return _record_from_json(value)
    except (ArtifactCompatibilityError, ValueError, TypeError, KeyError):
        return None


def _repeat_chain_valid(
    root: Path,
    args: argparse.Namespace,
    *,
    profile_name: str,
    repeat_index: int,
    path_ids: Sequence[int],
) -> bool:
    return _reconstruct_repeat_from_children(
        root,
        args,
        profile_name=profile_name,
        repeat_index=repeat_index,
        execution_order_index=frozen_repeat_order(repeat_index).index(profile_name),
        path_ids=path_ids,
        peak_memory_fraction=0.0,
    ) is not None


def _repeat_certificate_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.json"), key=lambda item: item.as_posix()):
        if path.name == "repeat.json":
            continue
        record = _load_json(path)
        scheduler = record.get("scheduler_record")
        if isinstance(scheduler, Mapping):
            digest.update(str(scheduler["batch_certificate_sha256"]).encode("ascii"))
        if record.get("schema") == RUN_SCHEMA + "-pilot-branch-commit":
            digest.update(str(record["output_sha256"]).encode("ascii"))
    return digest.hexdigest()


def _window_files_size(root: Path) -> int:
    return sum(item.stat().st_size for item in _repeat_child_paths(root))


def _repeat_child_paths(root: Path) -> tuple[Path, ...]:
    expected: list[Path] = []
    for window_start in WINDOW_START_STEPS:
        directory = root / f"window-{window_start:03d}"
        for shard_start, capture_expected in (
            (window_start, False),
            (window_start + SHARD_STEPS, True),
        ):
            stem = f"step-{shard_start:03d}"
            expected.extend(
                (
                    directory / f"{stem}-state.npz",
                    directory / f"{stem}.commit.json",
                    directory / f"{stem}.json",
                )
            )
            if capture_expected:
                expected.append(directory / f"{stem}-capture.npz")
        expected.extend(
            (
                directory / "branch-data.npz",
                directory / "branch-audit.npz",
                directory / "branch-conservation.npz",
                directory / "branch.commit.json",
                directory / "branch.json",
            )
        )
    return tuple(expected)


def _repeat_child_file_set_valid(root: Path) -> bool:
    expected = {path.resolve() for path in _repeat_child_paths(root)}
    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "repeat.json"
        and not path.name.endswith(".tmp")
        and ".tmp." not in path.name
    }
    return actual == expected and all(path.is_file() for path in expected)


def _reconstruct_repeat_from_children(
    root: Path,
    args: argparse.Namespace,
    *,
    profile_name: str,
    repeat_index: int,
    execution_order_index: int,
    path_ids: Sequence[int],
    peak_memory_fraction: float,
) -> PilotRepeatRecord | None:
    if not _repeat_child_file_set_valid(root):
        return None
    elapsed = 0.0
    base_count = 0
    midpoint_count = 0
    certified = 0
    fallback = 0
    fallback_seconds = 0.0
    maximum_mass = 0.0
    maximum_lanes = 0
    forbidden = {name: 0 for name in FORBIDDEN_DIAGNOSTICS}
    output_digest = hashlib.sha256()
    final_digest = hashlib.sha256()
    for window_start in WINDOW_START_STEPS:
        current, _ = _load_benchmark_initial_states(
            args.parent_coarse_residual_run_dir,
            window_start=window_start,
            path_count=len(path_ids),
        )
        first = _valid_base_shard(
            root,
            profile_name=profile_name,
            repeat_index=repeat_index,
            path_ids=path_ids,
            window_start=window_start,
            shard_start=window_start,
            input_states=current,
            capture_expected=False,
        )
        if first is None:
            return None
        second = _valid_base_shard(
            root,
            profile_name=profile_name,
            repeat_index=repeat_index,
            path_ids=path_ids,
            window_start=window_start,
            shard_start=window_start + SHARD_STEPS,
            input_states=first[0],
            capture_expected=True,
        )
        if second is None or second[1] is None:
            return None
        branch = _valid_branch_commit(
            root,
            profile_name=profile_name,
            repeat_index=repeat_index,
            window_start=window_start,
            path_ids=path_ids,
            pre_phase_states=second[1],
        )
        if branch is None:
            return None
        for base in (first[2], second[2]):
            diagnostics = base["scheduler_record"]["diagnostics"]
            elapsed += float(base["complete_pipeline_elapsed_seconds"])
            base_count += int(diagnostics["transition_count"])
            certified += int(diagnostics["certified_count"])
            fallback += int(diagnostics.get("fallback_count", 0))
            fallback_seconds += float(diagnostics.get("fallback_elapsed_seconds", 0.0))
            maximum_mass = max(
                maximum_mass, float(diagnostics["maximum_mass_error"])
            )
            maximum_lanes = max(
                maximum_lanes,
                int(diagnostics.get("maximum_cuda_launch_lanes", 0)),
            )
            for name in FORBIDDEN_DIAGNOSTICS:
                if name == "uncertified_count":
                    forbidden[name] += int(diagnostics["transition_count"]) - int(
                        diagnostics["certified_count"]
                    )
                else:
                    forbidden[name] += int(diagnostics.get(name, 0))
            output_digest.update(
                str(base["scheduler_record"]["batch_output_sha256"]).encode("ascii")
            )
        elapsed += float(branch["complete_pipeline_elapsed_seconds"])
        midpoint_count += int(branch["transition_count"])
        certified += int(branch["certified_count"])
        fallback += int(branch["fallback_count"])
        fallback_seconds += float(branch["fallback_elapsed_seconds"])
        maximum_mass = max(maximum_mass, float(branch["maximum_mass_error"]))
        maximum_lanes = max(maximum_lanes, int(branch["maximum_launch_lanes"]))
        for name in FORBIDDEN_DIAGNOSTICS:
            forbidden[name] += int(branch["forbidden_counts"][name])
        output_digest.update(str(branch["output_sha256"]).encode("ascii"))
        final_digest.update(_array_sha(second[0]).encode("ascii"))
    try:
        return PilotRepeatRecord(
            profile_name=profile_name,
            repeat_index=repeat_index,
            execution_order_index=execution_order_index,
            elapsed_seconds=elapsed,
            base_transition_count=base_count,
            midpoint_transition_count=midpoint_count,
            certified_count=certified,
            fallback_count=fallback,
            fallback_elapsed_seconds=fallback_seconds,
            maximum_mass_error=maximum_mass,
            peak_memory_fraction=peak_memory_fraction,
            committed_bytes=_window_files_size(root),
            maximum_launch_lanes=maximum_lanes,
            output_sha256=output_digest.hexdigest(),
            final_state_sha256=final_digest.hexdigest(),
            forbidden_counts=forbidden,
        )
    except (ValueError, TypeError, KeyError):
        return None


def _run_profile_repeat(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    profile_name: str,
    repeat_index: int,
    execution_order_index: int,
    device: torch.device,
    model: JacobiRBPhasePredictor,
) -> PilotRepeatRecord:
    root = _profile_repeat_root(run_dir, profile_name, repeat_index)
    root.mkdir(parents=True, exist_ok=True)
    repeat_path = root / "repeat.json"
    completed = _load_completed_repeat(repeat_path)
    path_ids = tuple(int(value) for value in PROFILE_PATH_IDS[profile_name])
    if completed is not None:
        reconstructed = _reconstruct_repeat_from_children(
            root,
            args,
            profile_name=profile_name,
            repeat_index=repeat_index,
            execution_order_index=execution_order_index,
            path_ids=path_ids,
            peak_memory_fraction=completed.peak_memory_fraction,
        )
        if reconstructed is not None and reconstructed.to_record() == completed.to_record():
            return completed
    elapsed = 0.0
    base_count = 0
    midpoint_count = 0
    certified = 0
    fallback = 0
    fallback_seconds = 0.0
    maximum_mass = 0.0
    maximum_lanes = 0
    forbidden = {name: 0 for name in FORBIDDEN_DIAGNOSTICS}
    output_digest = hashlib.sha256()
    final_digest = hashlib.sha256()
    for window_start in WINDOW_START_STEPS:
        current, _ = _load_benchmark_initial_states(
            args.parent_coarse_residual_run_dir,
            window_start=window_start,
            path_count=len(path_ids),
        )
        first_final, _, first_record = _run_base_shard(
            root,
            args,
            profile_name=profile_name,
            repeat_index=repeat_index,
            path_ids=path_ids,
            window_start=window_start,
            shard_start=window_start,
            input_states=current,
            capture_expected=False,
            device=device,
        )
        second_final, pre_states, second_record = _run_base_shard(
            root,
            args,
            profile_name=profile_name,
            repeat_index=repeat_index,
            path_ids=path_ids,
            window_start=window_start,
            shard_start=window_start + SHARD_STEPS,
            input_states=first_final,
            capture_expected=True,
            device=device,
        )
        if pre_states is None:
            raise BoundaryTangentScheduleCLIError("branch pre-phase states are absent")
        branch = _run_branch_commit(
            root,
            args,
            profile_name=profile_name,
            repeat_index=repeat_index,
            path_ids=path_ids,
            window_start=window_start,
            pre_phase_states=pre_states,
            device=device,
            model=model,
        )
        for base in (first_record, second_record):
            diagnostics = base["scheduler_record"]["diagnostics"]
            elapsed += float(base["complete_pipeline_elapsed_seconds"])
            base_count += int(diagnostics["transition_count"])
            certified += int(diagnostics["certified_count"])
            fallback += int(diagnostics.get("fallback_count", 0))
            fallback_seconds += float(diagnostics.get("fallback_elapsed_seconds", 0.0))
            maximum_mass = max(maximum_mass, float(diagnostics["maximum_mass_error"]))
            maximum_lanes = max(
                maximum_lanes, int(diagnostics.get("maximum_cuda_launch_lanes", 0))
            )
            for name in FORBIDDEN_DIAGNOSTICS:
                if name == "uncertified_count":
                    forbidden[name] += int(diagnostics["transition_count"]) - int(
                        diagnostics["certified_count"]
                    )
                else:
                    forbidden[name] += int(diagnostics.get(name, 0))
            output_digest.update(
                str(base["scheduler_record"]["batch_output_sha256"]).encode("ascii")
            )
        elapsed += float(branch["complete_pipeline_elapsed_seconds"])
        midpoint_count += int(branch["transition_count"])
        certified += int(branch["certified_count"])
        fallback += int(branch["fallback_count"])
        fallback_seconds += float(branch["fallback_elapsed_seconds"])
        maximum_mass = max(maximum_mass, float(branch["maximum_mass_error"]))
        maximum_lanes = max(maximum_lanes, int(branch["maximum_launch_lanes"]))
        for name in FORBIDDEN_DIAGNOSTICS:
            forbidden[name] += int(branch["forbidden_counts"][name])
        output_digest.update(str(branch["output_sha256"]).encode("ascii"))
        final_digest.update(_array_sha(second_final).encode("ascii"))
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
        total_memory = int(torch.cuda.get_device_properties(device).total_memory)
        memory_fraction = peak / total_memory
    else:
        memory_fraction = 0.0
    expected_base, expected_midpoint, _ = expected_profile_transition_counts(profile_name)
    if base_count != expected_base or midpoint_count != expected_midpoint:
        raise BoundaryTangentScheduleCLIError(
            "complete profile transition count changed",
            failure_code="boundary_tangent_schedule_transition_count_invalid",
        )
    record = PilotRepeatRecord(
        profile_name=profile_name,
        repeat_index=repeat_index,
        execution_order_index=execution_order_index,
        elapsed_seconds=elapsed,
        base_transition_count=base_count,
        midpoint_transition_count=midpoint_count,
        certified_count=certified,
        fallback_count=fallback,
        fallback_elapsed_seconds=fallback_seconds,
        maximum_mass_error=maximum_mass,
        peak_memory_fraction=memory_fraction,
        committed_bytes=_window_files_size(root),
        maximum_launch_lanes=maximum_lanes,
        output_sha256=output_digest.hexdigest(),
        final_state_sha256=final_digest.hexdigest(),
        forbidden_counts=forbidden,
    )
    reconstructed = _reconstruct_repeat_from_children(
        root,
        args,
        profile_name=profile_name,
        repeat_index=repeat_index,
        execution_order_index=execution_order_index,
        path_ids=path_ids,
        peak_memory_fraction=memory_fraction,
    )
    if reconstructed is None or reconstructed.to_record() != record.to_record():
        raise BoundaryTangentScheduleCLIError(
            "completed repeat does not equal its committed child-shard chain",
            failure_code="boundary_tangent_schedule_repeat_aggregate_invalid",
        )
    payload = record.to_record()
    payload["semantic_sha256"] = config_fingerprint(payload)
    atomic_write_json(repeat_path, payload)
    return record


def _test_pilot_records() -> list[PilotRepeatRecord]:
    records = []
    for repeat in range(PILOT_REPEAT_COUNT):
        order = frozen_repeat_order(repeat)
        for order_index, profile_name in enumerate(order):
            base, midpoint, total = expected_profile_transition_counts(profile_name)
            digest = hashlib.sha256(profile_name.encode("ascii")).hexdigest()
            records.append(
                PilotRepeatRecord(
                    profile_name=profile_name,
                    repeat_index=repeat,
                    execution_order_index=order_index,
                    elapsed_seconds=total / 4_000.0,
                    base_transition_count=base,
                    midpoint_transition_count=midpoint,
                    certified_count=total,
                    fallback_count=0,
                    fallback_elapsed_seconds=0.0,
                    maximum_mass_error=0.0,
                    peak_memory_fraction=0.0,
                    committed_bytes=1,
                    maximum_launch_lanes=build_fused_launch_plan(
                        PROFILE_PATH_COUNTS[profile_name]
                    ).maximum_chunk_lanes,
                    output_sha256=digest,
                    final_state_sha256=digest,
                    forbidden_counts={name: 0 for name in FORBIDDEN_DIAGNOSTICS},
                )
            )
    return records


def _pilot_child_evidence(
    run_dir: Path, args: argparse.Namespace, records: Sequence[PilotRepeatRecord]
) -> dict[str, Any]:
    chain_valid = True
    repeat_reconstruction_valid = True
    base_shard_count = 0
    branch_records: list[dict[str, Any]] = []
    for value in records:
        root = _profile_repeat_root(run_dir, value.profile_name, value.repeat_index)
        reconstructed = _reconstruct_repeat_from_children(
            root,
            args,
            profile_name=value.profile_name,
            repeat_index=value.repeat_index,
            execution_order_index=value.execution_order_index,
            path_ids=PROFILE_PATH_IDS[value.profile_name],
            peak_memory_fraction=value.peak_memory_fraction,
        )
        exact = reconstructed is not None and reconstructed.to_record() == value.to_record()
        chain_valid = chain_valid and reconstructed is not None
        repeat_reconstruction_valid = repeat_reconstruction_valid and exact
        for window_start in WINDOW_START_STEPS:
            for shard_start in (window_start, window_start + SHARD_STEPS):
                if (root / f"window-{window_start:03d}" / f"step-{shard_start:03d}.json").is_file():
                    base_shard_count += 1
            branch_path = root / f"window-{window_start:03d}" / "branch.json"
            if branch_path.is_file():
                branch_records.append(_load_json(branch_path))
    cache = [
        value for value in branch_records if value.get("mode") == "cache"
    ]
    stream = [
        value for value in branch_records if value.get("mode") == "stream"
    ]
    expected_branch_count = len(records) * len(WINDOW_START_STEPS)
    return {
        "chain_valid": int(chain_valid),
        "repeat_reconstruction_valid": int(repeat_reconstruction_valid),
        "completed_base_shard_count": base_shard_count,
        "branch_record_count": len(branch_records),
        "permitted_input_conversion_valid": int(
            len(branch_records) == expected_branch_count
            and all(int(value.get("permitted_float32_input_conversion", 0)) == 1 for value in branch_records)
        ),
        "raw_label_conversion_valid": int(
            len(branch_records) == expected_branch_count
            and all(int(value.get("raw_float64_label_conversion", 0)) == 1 for value in branch_records)
        ),
        "cache_commit_valid": int(
            len(cache) == expected_branch_count // 2
            and all(int(value.get("width32_forward_performed", -1)) == 0 for value in cache)
        ),
        "predictor_forward_valid": int(
            len(stream) == expected_branch_count // 2
            and all(int(value.get("width32_forward_performed", 0)) == 1 for value in stream)
        ),
        "gpu_risk_accumulation_valid": int(
            len(stream) == expected_branch_count // 2
            and all(
                value.get("risk_accumulation_device") == "cuda"
                and int(value.get("risk_host_transfer_count", -1)) == 1
                for value in stream
            )
        ),
        "stream_commit_valid": int(len(stream) == expected_branch_count // 2),
        "cross_role_isolation_valid": int(
            len(cache) == expected_branch_count // 2
            and len(stream) == expected_branch_count // 2
            and int(
                _load_json(run_dir / "schedule_equivalence_preflight.json").get(
                    "cross_role_isolation_valid", 0
                )
            )
            == 1
        ),
        "maximum_observed_launch_lanes": max(
            (int(value.maximum_launch_lanes) for value in records), default=0
        ),
    }


def _run_untimed_warmup(
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    model: JacobiRBPhasePredictor,
) -> None:
    path_ids = tuple(build_schedule_path_plan()["roles"]["cuda_warmup"])
    initial, source = _load_benchmark_initial_states(
        args.parent_coarse_residual_run_dir, window_start=0, path_count=len(path_ids)
    )
    states = torch.as_tensor(
        np.array(initial, copy=True, order="C"), dtype=torch.float64, device=device
    ).contiguous()
    base = run_exact_multipath_shard(
        states,
        path_ids=path_ids,
        start_step=0,
        root_seed=ROOT_SEED,
        profile=JacobiRBCudaProfile(),
        group_sizes=(len(path_ids),),
        capture_training_payload=True,
    )
    pre = _capture_pre_phase_states(base, 7)[0]
    branch = sample_fused_midpoint_branches(
        torch.as_tensor(pre, dtype=torch.float64, device=device).contiguous(),
        path_ids=path_ids,
        outer_step=7,
        phase=0,
        root_seed=ROOT_SEED,
        profile=JacobiRBCudaProfile(),
    )
    permitted, _ = flatten_midpoint_batches((branch.batch,))
    permitted["later_full_state"] = np.ascontiguousarray(
        permitted["later_full_state"], dtype=np.float32
    )
    with torch.no_grad():
        prediction = call_model(model, _model_inputs(permitted, device))
    atomic_write_json(
        run_dir / "cuda_model_warmup.json",
        {
            "schema": RUN_SCHEMA + "-cuda-model-warmup",
            "schema_version": 1,
            "timing_role": "untimed",
            "path_ids": list(path_ids),
            "source": source,
            "base_output_sha256": base.batch_output_sha256,
            "branch_output_sha256": branch.output_sha256(),
            "prediction_sha256": _array_sha(prediction.detach().cpu().numpy()),
            "passed": 1,
            **NO_WORK,
        },
    )


def _execute_pilot_panel(
    run_dir: Path, args: argparse.Namespace
) -> list[PilotRepeatRecord]:
    if args.test_only:
        return _test_pilot_records()
    device = torch.device(args.device)
    enable_deterministic_torch()
    torch.manual_seed(ROOT_SEED)
    torch.cuda.manual_seed_all(ROOT_SEED)
    model = JacobiRBPhasePredictor(width=32).to(device).eval()
    _run_untimed_warmup(run_dir, args, device, model)
    records: list[PilotRepeatRecord] = []
    for repeat in range(PILOT_REPEAT_COUNT):
        order = frozen_repeat_order(repeat)
        for order_index, profile_name in enumerate(order):
            torch.cuda.reset_peak_memory_stats(device)
            record = _run_profile_repeat(
                run_dir,
                args,
                profile_name=profile_name,
                repeat_index=repeat,
                execution_order_index=order_index,
                device=device,
                model=model,
            )
            records.append(record)
            print(
                f"schedule pilot repeat {repeat + 1}/3 profile {profile_name} "
                f"rate={record.transitions_per_second:.1f}/s",
                flush=True,
            )
    return records


def _pilot_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _sealed_stage(
        run_dir, gate_name="pilot_gate.json", seal_name="pilot_artifact_seal.json"
    )
    if existing is not None:
        return existing
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("pilot requires a passing preflight gate")
    if not args.test_only:
        configure_exact_torch_backend(args.device)
    records = _execute_pilot_panel(run_dir, args)
    grouped = validate_repeat_records(records)
    projection = project_frozen_schedule(records)
    child_evidence = (
        {
            "chain_valid": 1,
            "repeat_reconstruction_valid": 1,
            "completed_base_shard_count": 96,
            "branch_record_count": 48,
            "permitted_input_conversion_valid": 1,
            "raw_label_conversion_valid": 1,
            "cache_commit_valid": 1,
            "predictor_forward_valid": 1,
            "gpu_risk_accumulation_valid": 1,
            "stream_commit_valid": 1,
            "cross_role_isolation_valid": 1,
            "maximum_observed_launch_lanes": max(
                value.maximum_launch_lanes for value in records
            ),
        }
        if args.test_only
        else _pilot_child_evidence(run_dir, args, records)
    )
    repeat_rows = [record.to_record() for record in records]
    certificate_hashes = {
        name: [
            _repeat_certificate_sha(
                _profile_repeat_root(run_dir, name, repeat_index)
            )
            for repeat_index in range(PILOT_REPEAT_COUNT)
        ]
        for name in PILOT_PROFILE_NAMES
    }
    certificate_hashes_identical = all(
        len(set(values)) == 1 for values in certificate_hashes.values()
    )
    output_hashes_identical = all(
        len({value.output_sha256 for value in grouped[name]}) == 1
        for name in PILOT_PROFILE_NAMES
    )
    final_state_hashes_identical = all(
        len({value.final_state_sha256 for value in grouped[name]}) == 1
        for name in PILOT_PROFILE_NAMES
    )
    repeat_hashes_identical = bool(
        output_hashes_identical
        and final_state_hashes_identical
        and certificate_hashes_identical
    )
    atomic_write_csv(
        run_dir / "pilot_repeat_metrics.csv",
        [
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            }
            for row in repeat_rows
        ],
    )
    panel_record = {
        "schema": RUN_SCHEMA + "-pilot-panel",
        "schema_version": 1,
        "repeat_records": repeat_rows,
        "profile_output_hashes": {
            name: grouped[name][0].output_sha256 for name in PILOT_PROFILE_NAMES
        },
        "profile_final_state_hashes": {
            name: grouped[name][0].final_state_sha256 for name in PILOT_PROFILE_NAMES
        },
        "profile_certificate_hashes": certificate_hashes,
        "repeat_hashes_identical": int(repeat_hashes_identical),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "pilot_repeat_registry.json", panel_record)
    projection_record = projection.to_record()
    atomic_write_json(run_dir / "schedule_projection.json", projection_record)
    total_transitions = sum(value.transition_count for value in records)
    total_certified = sum(value.certified_count for value in records)
    total_fallback = sum(value.fallback_count for value in records)
    total_elapsed = sum(value.elapsed_seconds for value in records)
    total_fallback_elapsed = sum(value.fallback_elapsed_seconds for value in records)
    forbidden_counts = {
        name: sum(int(value.forbidden_counts[name]) for value in records)
        for name in FORBIDDEN_DIAGNOSTICS
    }
    collision_record = _load_json(run_dir / "path_collision_scan.json")
    metrics = {
        "schema": RUN_SCHEMA + "-pilot-metrics",
        "schema_version": 1,
        "profile_elapsed_seconds": {
            name: [value.elapsed_seconds for value in grouped[name]]
            for name in PILOT_PROFILE_NAMES
        },
        "profile_transition_counts": {
            name: grouped[name][0].transition_count for name in PILOT_PROFILE_NAMES
        },
        "projected_elapsed_seconds": projection.projected_seconds,
        "projected_effective_transitions_per_second": projection.projected_effective_rate,
        "all_profiles_complete": int(len(records) == 12),
        "repeat_hashes_identical": int(repeat_hashes_identical),
        "output_hashes_identical": int(output_hashes_identical),
        "final_state_hashes_identical": int(final_state_hashes_identical),
        "certificate_hashes_identical": int(certificate_hashes_identical),
        "atomic_shard_chains_valid": int(child_evidence["chain_valid"]),
        "resume_replay_valid": int(child_evidence["chain_valid"]),
        "completed_repeat_skipping_valid": int(
            child_evidence["repeat_reconstruction_valid"]
        ),
        "permitted_input_conversion_valid": int(
            child_evidence["permitted_input_conversion_valid"]
        ),
        "raw_label_conversion_valid": int(
            child_evidence["raw_label_conversion_valid"]
        ),
        "cache_commit_valid": int(child_evidence["cache_commit_valid"]),
        "predictor_forward_valid": int(
            child_evidence["predictor_forward_valid"]
        ),
        "gpu_risk_accumulation_valid": int(
            child_evidence["gpu_risk_accumulation_valid"]
        ),
        "stream_commit_valid": int(child_evidence["stream_commit_valid"]),
        "cross_role_isolation_valid": int(
            child_evidence["cross_role_isolation_valid"]
        ),
        "slowest_repeat_selection_valid": 1,
        "repeat_averaging_not_used": 1,
        "posthoc_allowance_not_used": 1,
        "uncertified_count": total_transitions - total_certified,
        "cap_count": forbidden_counts["resource_cap_count"],
        "invalid_density_count": forbidden_counts["invalid_density_count"],
        "approximation_count": forbidden_counts["approximation_count"],
        "correction_count": forbidden_counts["correction_count"],
        "floor_count": forbidden_counts["floor_count"],
        "limiter_count": forbidden_counts["limiter_count"],
        "projection_count": 0,
        "renormalization_count": forbidden_counts["renormalization_count"],
        "nonfinite_count": forbidden_counts["nonfinite_count"],
        "boundary_rejection_count": 0,
        "transition_id_collision_count": int(collision_record["collision_count"]),
        "repeat_hash_mismatch_count": int(not certificate_hashes_identical),
        "certificate_fraction": total_certified / total_transitions,
        "maximum_mass_error": projection.maximum_mass_error,
        "fallback_fraction": total_fallback / total_transitions,
        "fallback_time_fraction": total_fallback_elapsed / total_elapsed,
        "peak_memory_fraction": projection.maximum_peak_memory_fraction,
        "projected_persisted_bytes": projection.projected_persistence_bytes,
        "projected_transition_count": PROJECTED_TOTAL_TRANSITIONS,
        "base_transition_count": PROJECTED_BASE_TRANSITIONS,
        "midpoint_transition_count": PROJECTED_MIDPOINT_TRANSITIONS,
        "restart_outer_steps": SHARD_STEPS,
        "pilot_repeats": PILOT_REPEAT_COUNT,
        "completed_shard_count": int(child_evidence["completed_base_shard_count"]),
        "pilot_total_executed_transition_count": total_transitions,
        "maximum_observed_launch_lanes": int(
            child_evidence["maximum_observed_launch_lanes"]
        ),
        "production_cache_generated": 0,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "pilot_metrics.json", metrics)
    from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import evaluate_schedule_pilot

    gate = evaluate_schedule_pilot(metrics)
    atomic_write_json(run_dir / "pilot_gate.json", gate)
    shard_artifacts = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": file_fingerprint(path),
            "size": path.stat().st_size,
        }
        for path in sorted((run_dir / "pilot").rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    ] if (run_dir / "pilot").is_dir() else []
    atomic_write_json(
        run_dir / "pilot_shard_registry.json",
        {
            "schema": RUN_SCHEMA + "-pilot-shard-registry",
            "schema_version": 1,
            "artifact_count": len(shard_artifacts),
            "artifacts": shard_artifacts,
            "semantic_sha256": config_fingerprint({"artifacts": shard_artifacts}),
            **NO_WORK,
        },
    )
    names = (
        "pilot_repeat_metrics.csv",
        "pilot_repeat_registry.json",
        "schedule_projection.json",
        "pilot_metrics.json",
        "pilot_gate.json",
        "pilot_shard_registry.json",
    )
    _seal_stage(run_dir, names, "pilot_artifact_seal.json")
    return gate


def _optional_json(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / name
    return _load_json(path) if path.is_file() else None


def _workflow_record(run_dir: Path, *, require_gate: str) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import (
        evaluate_schedule_workflow,
    )

    scientific_config = _optional_json(run_dir, "scientific_config.json") or {}

    def build(required: str) -> dict[str, Any]:
        value = evaluate_schedule_workflow(
            provenance=_optional_json(run_dir, "parent_provenance.json") or False,
            preflight_gate=_optional_json(run_dir, "preflight_gate.json"),
            pilot_gate=_optional_json(run_dir, "pilot_gate.json"),
            require_gate=required,
        )
        if int(scientific_config.get("test_only", 0)) == 1:
            value = dict(value)
            value["test_only"] = 1
            value["authorizing"] = 0
            value["decision"] = {
                "schema": RUN_SCHEMA + "-test-only-decision",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "decision": "test_only_complete",
                "recommended_next_action": "run a fresh production CUDA workflow",
                "schedule_integration_authorized": 0,
                **NO_AUTHORIZATION,
                **NO_WORK,
            }
        return value

    workflow = build(require_gate)
    stage_name = (
        "pilot"
        if (run_dir / "pilot_gate.json").is_file()
        else ("preflight" if (run_dir / "preflight_gate.json").is_file() else "initialize")
    )
    stage_workflow = run_dir / f"workflow_{stage_name}.json"
    stage_decision = run_dir / f"schedule_decision_{stage_name}.json"
    stage_value = workflow
    if stage_workflow.is_file():
        recorded_required = str(_load_json(stage_workflow).get("required_gate"))
        stage_value = build(recorded_required)
    _freeze_json(stage_workflow, stage_value)
    _freeze_json(stage_decision, stage_value["decision"])
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(run_dir / "schedule_decision.json", workflow["decision"])
    return workflow


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "pilot")
    if stage == "report":
        return ()
    if stage in {"preflight", "pilot"}:
        return (stage,)
    raise ValueError(f"unknown stage: {stage}")


def _verify_report(run_dir: Path) -> None:
    if (run_dir / "preflight_gate.json").is_file():
        _sealed_stage(
            run_dir,
            gate_name="preflight_gate.json",
            seal_name="preflight_artifact_seal.json",
        )
    if (run_dir / "pilot_gate.json").is_file():
        _sealed_stage(
            run_dir,
            gate_name="pilot_gate.json",
            seal_name="pilot_artifact_seal.json",
        )


def _verify_existing_registry(run_dir: Path) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    registry = _load_json(path)
    artifacts = registry.get("artifacts")
    semantics = registry.get("registry_semantics")
    expected_semantics = {
        "snapshot_kind": "terminal-exact-with-restartable-pilot-extras",
        "excluded_paths": sorted(_REGISTRY_EXCLUDED),
        "restartable_extras_must_match_frozen_pilot_layout": 1,
    }
    if (
        registry.get("schema") != RUN_SCHEMA + "-artifact-registry"
        or int(registry.get("schema_version", -1)) != 1
        or not isinstance(artifacts, list)
        or int(registry.get("artifact_count", -1)) != len(artifacts)
        or semantics != expected_semantics
        or registry.get("semantic_sha256")
        != config_fingerprint(
            {"artifacts": artifacts, "registry_semantics": expected_semantics}
        )
        or any(int(registry.get(name, -1)) != 0 for name in NO_WORK)
        or any(int(registry.get(name, -1)) != 0 for name in NO_AUTHORIZATION)
    ):
        raise ArtifactCompatibilityError("terminal artifact registry changed")
    recorded: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ArtifactCompatibilityError("terminal artifact registry entry changed")
        relative = str(item["path"])
        if relative in recorded:
            raise ArtifactCompatibilityError("terminal artifact registry has duplicate paths")
        recorded[relative] = item
        target = run_dir / relative
        if (
            not target.is_file()
            or item.get("sha256") != file_fingerprint(target)
            or int(item.get("size", -1)) != target.stat().st_size
        ):
            raise ArtifactCompatibilityError(f"terminal artifact changed: {relative}")
    current = {str(item["path"]) for item in _registry_artifacts(run_dir)}
    recorded_paths = set(recorded)
    missing = sorted(recorded_paths - current)
    added = sorted(current - recorded_paths)
    allowed = _allowed_restartable_extra_paths(run_dir)
    unexpected = sorted(set(added) - allowed)
    if missing or unexpected:
        raise ArtifactCompatibilityError(
            "terminal artifact file set or content changed: "
            f"missing={missing}, unexpected_added={unexpected}"
        )


def _allowed_restartable_extra_paths(run_dir: Path) -> set[str]:
    """Enumerate files that a hard-killed pilot may add after a preflight snapshot."""

    allowed = {
        "cuda_model_warmup.json",
        "pilot_repeat_metrics.csv",
        "pilot_repeat_registry.json",
        "schedule_projection.json",
        "pilot_metrics.json",
        "pilot_gate.json",
        "pilot_shard_registry.json",
        "pilot_artifact_seal.json",
        "workflow_pilot.json",
        "schedule_decision_pilot.json",
    }
    for profile_name in PILOT_PROFILE_NAMES:
        for repeat_index in range(PILOT_REPEAT_COUNT):
            root = _profile_repeat_root(run_dir, profile_name, repeat_index)
            allowed.add((root / "repeat.json").relative_to(run_dir).as_posix())
            allowed.update(
                path.relative_to(run_dir).as_posix()
                for path in _repeat_child_paths(root)
            )
    return allowed


def _commit_execution_failure(
    run_dir: Path,
    *,
    stage: str,
    exc: Exception,
    require_gate: str,
) -> None:
    failure_domain = str(getattr(exc, "failure_domain", "schedule_execution"))
    failure_code = str(
        getattr(exc, "failure_code", "boundary_tangent_schedule_execution_failed")
    )
    failure = {
        "schema": RUN_SCHEMA + "-execution-failure",
        "schema_version": 1,
        "stage": stage,
        "evaluation_status": "execution_failed",
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "recorded_at": _now(),
        **NO_WORK,
    }
    atomic_write_json(run_dir / f"{stage}_execution_failure.json", failure)
    gate_stage = stage if stage in {"preflight", "pilot"} else "preflight"
    gate_path = run_dir / f"{gate_stage}_gate.json"
    if not gate_path.is_file():
        from mnist.d0_jacobi_rb_boundary_tangent_schedule_gate import (
            evaluate_schedule_pilot,
            evaluate_schedule_preflight,
        )

        evaluator = (
            evaluate_schedule_preflight
            if gate_stage == "preflight"
            else evaluate_schedule_pilot
        )
        atomic_write_json(gate_path, evaluator(failure))
    workflow = _workflow_record(run_dir, require_gate=require_gate)
    _status(
        run_dir,
        state="execution_failed",
        stage=stage,
        decision=str(workflow["decision"]["decision"]),
        message=str(exc),
        failure_domain=failure_domain,
        failure_code=failure_code,
        scientific_evidence_complete=0,
    )
    _artifact_registry(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--failed-boundary-tangent-run-dir", type=Path, required=True)
    parser.add_argument("--parent-coarse-residual-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_schedule_feasibility"
        ),
    )
    parser.add_argument("--run-name", default="production-fused-boundary-tangent-schedule")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.failed_boundary_tangent_run_dir = args.failed_boundary_tangent_run_dir.resolve()
    args.parent_coarse_residual_run_dir = args.parent_coarse_residual_run_dir.resolve()
    args.runs_root = args.runs_root.resolve()
    if args.resume_run_dir is not None:
        args.resume_run_dir = args.resume_run_dir.resolve()
    if not args.test_only:
        if args.stage != "report" and args.device != "cuda":
            parser.error("authorizing schedule stages require --device cuda")
    elif args.require_gate != "none":
        parser.error("test-only runs are nonauthorizing and require --require-gate none")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    active_stage = "initialize"
    resumed = False
    initialized = False
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"boundary-tangent schedule run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        if resumed:
            _verify_existing_registry(run_dir)
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
            else:
                if not _passed(_optional_json(run_dir, "preflight_gate.json")):
                    raise ArtifactCompatibilityError(
                        "pilot requires a passing preflight gate"
                    )
                gate = _pilot_stage(run_dir, args)
            gate_path = run_dir / f"{stage}_gate.json"
            if not gate_path.is_file():
                atomic_write_json(gate_path, gate)
            if not _passed(gate):
                break
        workflow = _workflow_record(run_dir, require_gate=args.require_gate)
        decision = str(workflow["decision"]["decision"])
        required_pass = int(workflow["required_gate_pass"]) == 1
        terminal_gate = (
            _optional_json(run_dir, "pilot_gate.json")
            or _optional_json(run_dir, "preflight_gate.json")
            or {}
        )
        resource_failure = bool(
            not required_pass
            and terminal_gate.get("failure_domain") == "resource_gate"
            and int(terminal_gate.get("scientific_evidence_complete", 0)) == 1
        )
        _status(
            run_dir,
            state=(
                "test_only_complete"
                if args.test_only and required_pass
                else ("complete" if required_pass else "gate_failed")
            ),
            stage=args.stage,
            decision=decision,
            failure_domain=None if required_pass else (
                "resource_gate" if resource_failure else str(
                    terminal_gate.get("failure_domain") or "schedule_gate"
                )
            ),
            failure_code=None if required_pass else f"{args.require_gate}_gate_failed",
            scientific_evidence_complete=(
                1 if required_pass or resource_failure else 0
            ),
        )
        _artifact_registry(run_dir)
        print(f"boundary-tangent schedule decision: {decision}", flush=True)
        return 0 if required_pass else 2
    except KeyboardInterrupt:
        if run_dir is not None and (initialized or not resumed):
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
        committed_exc: Exception = exc
        if (
            isinstance(exc, ArtifactCompatibilityError)
            and active_stage == "initialize"
            and not resumed
        ):
            committed_exc = BoundaryTangentScheduleCLIError(
                str(exc),
                failure_domain="provenance",
                failure_code="control_provenance_invalid",
            )
        if run_dir is not None and (initialized or not resumed):
            _commit_execution_failure(
                run_dir,
                stage=active_stage,
                exc=committed_exc,
                require_gate=args.require_gate,
            )
        import sys

        print(f"boundary-tangent schedule error: {exc}", file=sys.stderr, flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
