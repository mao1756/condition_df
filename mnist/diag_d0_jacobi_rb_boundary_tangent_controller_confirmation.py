"""Boundary-tangent exact-RB learnability and short controller controls.

This additive workflow keeps the certified K=512 Jacobi transition and raw
Rao--Blackwell target unchanged.  It retrains only the controller coordinate:
``m = y(1-y) q``.  It never runs a full reverse path or emits an image.
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
from typing import Any, Callable, Mapping, Sequence

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
    BOUNDARY_TANGENT_CACHE_VERSION,
    MIDPOINT_COUNT,
    MIDPOINT_FRACTIONS,
    SELECTED_OUTER_STEPS,
    flatten_midpoint_batches,
    sample_midpoint_branches,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    BoundaryTangentThresholds,
    claim_scope_flags,
    decide_boundary_tangent_workflow,
    evaluate_confirmation_gate,
    evaluate_controller_gate,
    one_sided_whole_path_max_t,
)
from mnist.d0_jacobi_rb_boundary_tangent_provenance import (
    boundary_tangent_source_fingerprint,
    boundary_tangent_source_paths,
    build_boundary_tangent_path_plan,
    build_failed_controller_readjudication,
    validate_boundary_tangent_path_plan,
    verify_boundary_tangent_parents,
    verify_resume_compatibility,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_multipath import run_exact_multipath_shard
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    MODEL_INPUT_FIELDS,
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


RUN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-controller-v1"
ROOT_SEED = 261_311
MODEL_SEEDS = (261_312, 261_313, 261_314)
BOOTSTRAP_SEED = 261_315
CONTROL_SEED = 261_316
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
CONTROL_MICROSTEPS = (2, 4, 8)
CONFIRMATION_INDEX_ARTIFACTS = (
    "confirmation_seal.json",
    "confirmation_path_risks.npz",
    "confirmation_risk_summary.json",
    "confirmation_max_t.json",
    "confirmation_cache_metrics.json",
)
CONTROL_INDEX_ARTIFACTS = (
    "controller_control_started.json",
    "controller_trajectory_max_t.json",
    "controller_trajectory_path_values.npz",
    "controller_trajectory_raw_metrics.csv",
    "controller_numerical_health.json",
)
FORBIDDEN_COUNTS = (
    "uncertified_count",
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "renormalization_count",
    "nonfinite_count",
)
NO_WORK = {
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "reconstruction_performed": 0,
    "full_reverse_path_performed": 0,
    "image_sampling_performed": 0,
}


class BoundaryTangentCLIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "execution",
        failure_code: str = "boundary_tangent_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return value


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
        "path": str(target),
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _atomic_torch(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(target),
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactCompatibilityError(f"cannot read NPZ artifact: {path}") from exc


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _passed(value: Mapping[str, Any] | None) -> bool:
    return isinstance(value, Mapping) and int(value.get("passed", 0)) == 1


def _not_evaluated(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + f"-{stage}-gate",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": "not_evaluated",
        "reason": str(reason),
        "passed": 0,
        **claim_scope_flags(),
    }


def _gate(
    stage: str,
    checks: Mapping[str, Any],
    *,
    physical_training_performed: bool = False,
    controller_control_trajectory_performed: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    normalized = {str(name): int(bool(value)) for name, value in checks.items()}
    return {
        "schema": RUN_SCHEMA + f"-{stage}-gate",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": "evaluated",
        "checks": normalized,
        "passed": int(bool(normalized) and all(normalized.values())),
        **extra,
        **claim_scope_flags(
            physical_training_performed=physical_training_performed,
            controller_control_trajectory_performed=controller_control_trajectory_performed,
        ),
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
) -> None:
    controller_control, maximum_control_phases = _controller_scope(run_dir)
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
            "updated_at": _now(),
            "physical_training_performed": int(
                (run_dir / "physical_training_started.json").is_file()
            ),
            "controller_control_trajectory_performed": int(controller_control),
            "maximum_control_trajectory_phase_count": maximum_control_phases,
            **NO_WORK,
        },
    )


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = []
    for path in sorted(
        (item for item in run_dir.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        relative = path.relative_to(run_dir).as_posix()
        if relative in excluded or relative.endswith(".tmp") or ".tmp." in path.name:
            continue
        records.append(
            {"path": relative, "sha256": file_fingerprint(path), "size": path.stat().st_size}
        )
    controller_control, maximum_control_phases = _controller_scope(run_dir)
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "artifact_count": len(records),
        "artifacts": records,
        "semantic_sha256": config_fingerprint({"artifacts": records}),
        **claim_scope_flags(
            physical_training_performed=(run_dir / "physical_training_started.json").is_file(),
            controller_control_trajectory_performed=controller_control,
        ),
        "maximum_control_trajectory_phase_count": maximum_control_phases,
    }
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _effective_paths(args: argparse.Namespace) -> dict[str, tuple[int, ...]]:
    plan = build_boundary_tangent_path_plan()
    roles = plan["roles"]
    values = {
        "preflight": tuple(int(item) for item in roles["preflight_benchmark"]),
        "train": tuple(int(item) for item in roles["train"]),
        "validation": tuple(int(item) for item in roles["validation"]),
        "confirmation": tuple(int(item) for item in roles["confirmation"]),
    }
    if not args.test_only:
        return values
    count = int(args.test_path_count)
    return {name: paths[: min(count, len(paths))] for name, paths in values.items()}


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    paths = _effective_paths(args)
    test_steps = int(args.test_outer_steps) if args.test_only else OUTER_STEPS
    updates = int(args.test_maximum_updates) if args.test_only else TRAINING["maximum_updates"]
    record = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "grid_size": 28,
        "alpha": 1.0,
        "sample_steps": OUTER_STEPS,
        "executed_outer_steps": test_steps,
        "tau_eff": 5.0e-5,
        "lambda_mix": 0.35,
        "label": 3,
        "root_seed": ROOT_SEED,
        "model_seeds": list(MODEL_SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "control_seed": CONTROL_SEED,
        "selected_outer_steps": list(
            step for step in SELECTED_OUTER_STEPS if step < test_steps
        ),
        "midpoint_fractions": list(MIDPOINT_FRACTIONS),
        "path_ids": {name: list(value) for name, value in paths.items()},
        "training": {**TRAINING, "maximum_updates": updates},
        "thresholds": BoundaryTangentThresholds(
            confirmation_paths=max(8, len(paths["confirmation"])),
            controller_paths=max(8, len(paths["confirmation"])),
            bootstrap_replicates=(
                int(args.test_bootstrap_replicates)
                if args.test_only
                else 50_000
            ),
        ).to_dict(),
        "target": "unchanged exact certified Jacobi Rao-Blackwell label",
        "optimizer_objective": "plain unweighted raw-target MSE",
        "prediction_coordinate": "m=y(1-y)*q",
        "controller_flow": "exact frozen-q logistic flow",
        "quotient_target_persisted": 0,
        "test_only": int(args.test_only),
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
    config = _scientific_config(args)
    path_plan = build_boundary_tangent_path_plan()
    validate_boundary_tangent_path_plan(path_plan)
    source_paths = boundary_tangent_source_paths(
        (
            Path(__file__),
            Path(__file__).with_name("d0_jacobi_rb_boundary_tangent.py"),
            Path(__file__).with_name("d0_jacobi_rb_boundary_tangent_cache.py"),
            Path(__file__).with_name("d0_jacobi_rb_boundary_tangent_gate.py"),
            Path(__file__).with_name("d0_jacobi_rb_boundary_tangent_provenance.py"),
            Path(__file__).with_name("d0_jacobi_rb_boundary_tangent_confirmation.py"),
        )
    )
    source_hash = boundary_tangent_source_fingerprint(source_paths)
    if resumed:
        parents = verify_boundary_tangent_parents(
            coarse_residual_run_dir=args.parent_coarse_residual_run_dir,
            failed_controller_run_dir=args.failed_controller_run_dir,
        )
        parent_hash = str(parents["semantic_sha256"])
        verify_resume_compatibility(
            run_dir,
            source_fingerprint=source_hash,
            scientific_config_sha256=config["semantic_sha256"],
            path_plan_sha256=path_plan["semantic_sha256"],
            parent_provenance_sha256=parent_hash,
        )
        if (
            _load_json(run_dir / "parent_provenance.json").get("semantic_sha256")
            != parent_hash
            or _load_json(run_dir / "run_manifest.json").get(
                "parent_provenance_sha256"
            )
            != parent_hash
        ):
            raise ArtifactCompatibilityError("resume parent provenance changed")
        return
    parents = verify_boundary_tangent_parents(
        coarse_residual_run_dir=args.parent_coarse_residual_run_dir,
        failed_controller_run_dir=args.failed_controller_run_dir,
    )
    atomic_write_json(run_dir / "parent_provenance.json", parents)
    atomic_write_json(
        run_dir / "failed_controller_readjudication.json",
        build_failed_controller_readjudication(parents),
    )
    atomic_write_json(run_dir / "scientific_config.json", config)
    atomic_write_json(run_dir / "path_id_plan.json", path_plan)
    manifest = {
        "schema": RUN_SCHEMA + "-manifest",
        "schema_version": 1,
        "created_at": _now(),
        "device": args.device,
        "source_fingerprint": source_hash,
        "source_paths": [str(path) for path in source_paths],
        "scientific_config_sha256": config["semantic_sha256"],
        "path_plan_sha256": path_plan["semantic_sha256"],
        "parent_provenance_sha256": str(parents["semantic_sha256"]),
        **claim_scope_flags(),
    }
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    _status(run_dir, state="initialized", stage="initialize")


def _load_source_target(parent: Path) -> np.ndarray:
    arrays = _load_npz(parent / "source_image.npz")
    target = np.asarray(arrays.get("mixed_target"))
    if (
        target.dtype != np.float64
        or target.shape != (STATE_SIZE,)
        or not np.isfinite(target).all()
        or np.any(target < 0.0)
        or not math.isclose(float(target.sum()), 1.0, rel_tol=0.0, abs_tol=2.0e-12)
    ):
        raise ArtifactCompatibilityError("parent mixed target is incompatible")
    return np.ascontiguousarray(target)


def _zero_baseline(path_ids: Sequence[int]) -> Any:
    from mnist.d0_jacobi_rb_boundary_tangent import TangentBaseline, TANGENT_BASELINE_SHAPE

    shape = TANGENT_BASELINE_SHAPE
    zeros = np.zeros(shape, dtype=np.float64)
    ones = np.ones(shape, dtype=np.float64)
    return TangentBaseline(
        q_values=zeros,
        numerators=zeros.copy(),
        denominators=ones,
        counts=np.ones(shape, dtype=np.int64),
        training_path_ids=np.asarray(sorted(path_ids), dtype=np.int64),
        training_inputs_sha256="0" * 64,
        training_targets_sha256="0" * 64,
        training_row_path_ids_sha256="0" * 64,
    )


def _analytic_null_baseline(path_ids: Sequence[int]) -> Any:
    """A deterministic nonzero baseline used only by the optimizer null."""

    from mnist.d0_jacobi_rb_boundary_tangent import TangentBaseline, TANGENT_BASELINE_SHAPE

    values = np.full(TANGENT_BASELINE_SHAPE, 0.25, dtype=np.float64)
    ones = np.ones(TANGENT_BASELINE_SHAPE, dtype=np.float64)
    identity = hashlib.sha256(b"boundary-tangent-analytic-null-q=0.25-v1").hexdigest()
    return TangentBaseline(
        q_values=values,
        numerators=values.copy(),
        denominators=ones,
        counts=np.ones(TANGENT_BASELINE_SHAPE, dtype=np.int64),
        training_path_ids=np.asarray(sorted(path_ids), dtype=np.int64),
        training_inputs_sha256=identity,
        training_targets_sha256=identity,
        training_row_path_ids_sha256=identity,
    )


def _model_inputs_from_arrays(
    arrays: Mapping[str, np.ndarray], device: torch.device
) -> ModelInputs:
    return ModelInputs(
        later_full_state=torch.as_tensor(
            np.array(arrays["later_full_state"], copy=True, order="C"),
            dtype=torch.float32,
            device=device,
        ),
        reverse_time=torch.as_tensor(
            np.array(arrays["reverse_time"], copy=True),
            dtype=torch.float64,
            device=device,
        ),
        phase=torch.as_tensor(np.array(arrays["phase"], copy=True), dtype=torch.long, device=device),
        color=torch.as_tensor(np.array(arrays["color"], copy=True), dtype=torch.long, device=device),
        duration=torch.as_tensor(
            np.array(arrays["duration"], copy=True), dtype=torch.float32, device=device
        ),
        label=torch.as_tensor(np.array(arrays["label"], copy=True), dtype=torch.long, device=device),
    )


def _representation_preflight(device: torch.device, path_ids: Sequence[int]) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent import (
        BoundaryTangentPredictor,
        direct_raw_target_mse,
        edge_pair_geometry,
        frozen_score_logistic_fraction,
        frozen_score_logistic_flow,
    )
    from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time

    baseline = _zero_baseline(path_ids)
    model = BoundaryTangentPredictor(baseline, zero_residual=True).to(device)
    state = torch.full((4, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float32, device=device)
    # Matching 0 contains the 0->1 edge.  Exercise both exact facets while
    # keeping all other cells finite and nonnegative.
    state[1, 0], state[1, 1] = 2.0 / STATE_SIZE, 0.0
    state[2, 0], state[2, 1] = 0.0, 2.0 / STATE_SIZE
    state[3, 0], state[3, 1] = 0.0, 0.0
    phase = 0
    inputs = ModelInputs(
        later_full_state=state,
        reverse_time=torch.full(
            (4,), internal_reverse_time(127, phase, 0.75), dtype=torch.float64, device=device
        ),
        phase=torch.full((4,), phase, dtype=torch.long, device=device),
        color=torch.full((4,), PHASE_MATCHINGS[phase], dtype=torch.long, device=device),
        duration=torch.full((4,), PHASE_DURATIONS[phase], dtype=torch.float32, device=device),
        label=torch.full((4,), 3, dtype=torch.long, device=device),
    )
    with torch.no_grad():
        coefficient = model.score_prediction(inputs)
        prediction = model(inputs)
        baseline_prediction = model.baseline_prediction(inputs)
    geometry = edge_pair_geometry(inputs)
    target = torch.zeros_like(prediction, dtype=torch.float64)
    scaled, raw = direct_raw_target_mse(prediction, target, 1.0)
    y = torch.tensor(
        [0.0, 1.0e-8, 1.0e-6, 1.0e-4, 0.25, 0.75, 1.0 - 1.0e-8, 1.0],
        dtype=torch.float64,
        device=device,
    )
    q = torch.full_like(y, 0.75)
    flowed = frozen_score_logistic_fraction(y, q, 0.01)
    twice = frozen_score_logistic_fraction(
        frozen_score_logistic_fraction(y, q, 0.004), q, 0.006
    )
    semigroup_error = float(
        torch.max(torch.abs(flowed - twice) / torch.clamp(torch.abs(flowed), min=1.0e-15)).item()
    )
    eps = torch.tensor([1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4], dtype=torch.float64, device=device)
    tangent = eps * (1.0 - eps) * 0.75
    slope = float(
        torch.polyfit(torch.log(eps), torch.log(torch.abs(tangent)), 1)[0].item()
    ) if hasattr(torch, "polyfit") else float(
        np.polyfit(
            np.log(eps.detach().cpu().numpy()),
            np.log(tangent.detach().cpu().numpy()),
            1,
        )[0]
    )
    state64 = state.to(dtype=torch.float64)
    q_edges = torch.full(
        (state64.shape[0], EDGES_PER_PHASE),
        0.75,
        dtype=torch.float64,
        device=device,
    )
    flowed_state = frozen_score_logistic_flow(state64, PHASE_MATCHINGS[phase], q_edges, 0.01)
    from mnist.d0_jacobi_rb_learnability import matching_indices
    all_tails, all_heads = matching_indices(device=device)
    tails, heads = all_tails[PHASE_MATCHINGS[phase]], all_heads[PHASE_MATCHINGS[phase]]
    pair_before = state64[:, tails] + state64[:, heads]
    pair_after = flowed_state[:, tails] + flowed_state[:, heads]
    full_flow_pair_error = float(torch.max(torch.abs(pair_after - pair_before)).item())
    full_flow_simplex_error = float(
        torch.max(torch.abs(flowed_state.sum(dim=1) - state64.sum(dim=1))).item()
    )
    interior_orientation = bool(
        torch.all(flowed_state[0, heads] > state64[0, heads])
    )
    expected_raw = torch.mean(prediction.to(torch.float64).square())
    checks = {
        "coefficient_finite": bool(torch.isfinite(coefficient).all()),
        "prediction_finite": bool(torch.isfinite(prediction).all()),
        "update_zero_exact_baseline": bool(torch.equal(prediction, baseline_prediction)),
        "facet_zero": bool(
            prediction[1, 0].item() == 0.0
            and prediction[2, 0].item() == 0.0
            and prediction[3, 0].item() == 0.0
        ),
        "zero_mass_zero": bool(
            geometry.mobility[3, 0].item() == 0.0 and prediction[3, 0].item() == 0.0
        ),
        "direct_raw_mse_finite": bool(torch.isfinite(scaled) and torch.isfinite(raw)),
        "direct_raw_mse_algebra": bool(torch.equal(raw, expected_raw) and torch.equal(scaled, raw)),
        "mobility_score_algebra": bool(
            torch.equal(prediction, geometry.mobility * coefficient)
        ),
        "logistic_facets_fixed": bool(flowed[0].item() == 0.0 and flowed[-1].item() == 1.0),
        "logistic_interior": bool(torch.all((flowed[1:-1] > 0.0) & (flowed[1:-1] < 1.0))),
        "logistic_semigroup": semigroup_error <= 2.0e-6,
        "facet_ray_slope": slope >= 0.9,
        "full_flow_orientation": interior_orientation,
        "full_flow_pair_conservation": full_flow_pair_error <= 2.0e-12,
        "full_flow_simplex_conservation": full_flow_simplex_error <= 2.0e-12,
    }
    return {
        "schema": RUN_SCHEMA + "-representation-preflight",
        "schema_version": 1,
        "checks": {name: int(value) for name, value in checks.items()},
        "facet_ray_log_log_slope": slope,
        "logistic_semigroup_relative_error": semigroup_error,
        "full_flow_maximum_pair_mass_error": full_flow_pair_error,
        "full_flow_maximum_simplex_mass_error": full_flow_simplex_error,
        "maximum_abs_update_zero_prediction": float(torch.max(torch.abs(prediction)).item()),
        "quotient_target_formed": 0,
        "passed": int(all(checks.values())),
    }


def _historical_boundary_diagnosis(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Reproduce the old M2/M4/M8 rejection; never reinterpret its weights."""

    from mnist import diag_d0_jacobi_rb_coarse_residual_learnability as parent_cli
    from mnist import diag_d0_jacobi_rb_reverse_controller as old_cli
    from mnist import d0_jacobi_rb_reverse_controller as old_core
    from mnist.d0_jacobi_rb_learnability import selected_reverse_time

    if args.test_only:
        return {
            "schema": RUN_SCHEMA + "-legacy-boundary-diagnosis",
            "schema_version": 1,
            "evaluation_status": "test_fixture",
            "rejections": {"M2": 1, "M4": 1, "M8": 1},
            "historical_first_failure": {
                "outer_step": 127,
                "phase": 0,
                "path_row": 0,
                "edge": 88,
                "candidate_fraction": 1.0018200816811438,
            },
            "passed": 1,
            **NO_WORK,
        }
    controller = old_core.load_frozen_controller(
        args.parent_coarse_residual_run_dir, device=device
    )
    validation = parent_cli._load_input_cache_for_role(  # noqa: SLF001
        args.parent_coarse_residual_run_dir, "validation"
    )
    phase = 0
    target_time = selected_reverse_time(127, phase)
    indices = np.flatnonzero(
        (validation.phase == phase) & (validation.reverse_time == target_time)
    )[:8]
    if indices.size != 8:
        raise ArtifactCompatibilityError("parent validation boundary panel is missing")
    state = torch.as_tensor(
        np.array(validation.later_full_state[indices], copy=True, order="C"),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    profile = JacobiRBCudaProfile()
    rejected: dict[str, int] = {}
    messages: dict[str, str] = {}
    for microsteps in CONTROL_MICROSTEPS:
        reference = old_cli._CertifiedReference(  # noqa: SLF001
            root_seed=ROOT_SEED,
            profile=profile,
            stream_role=f"boundary-tangent-legacy-M{microsteps}",
        )
        try:
            old_core.controlled_reverse_phase(
                state,
                127,
                phase,
                microsteps,
                old_core.NAMESPACE_VERSION,
                controller=controller,
                reference_transition=reference,
                path_ids=old_core.PREFLIGHT_PATH_IDS,
                label=3,
            )
        except old_core.ControllerBoundaryStepRejected as exc:
            rejected[f"M{microsteps}"] = 1
            messages[f"M{microsteps}"] = str(exc)
        else:
            rejected[f"M{microsteps}"] = 0
    return {
        "schema": RUN_SCHEMA + "-legacy-boundary-diagnosis",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "outer_step": 127,
        "phase": 0,
        "rejections": rejected,
        "messages": messages,
        "historical_first_failure": {
            "path_row": 0,
            "path_id": 0xEA000,
            "edge": 88,
            "tail_cell": 176,
            "head_cell": 177,
            "pair_mass": 0.0033212956866858576,
            "post_reference_fraction": 0.9995690075155423,
            "prediction": 0.06510134784541541,
            "delta_u": 0.017288998155204367,
            "requested_increment": 0.0022510741656014094,
            "available_headroom": 0.0004309924844576596,
            "candidate_fraction": 1.0018200816811438,
        },
        "old_checkpoint_modified": 0,
        "passed": int(all(rejected.values())),
        **NO_WORK,
    }


def _branch_resource_preflight(
    args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    paths = _effective_paths(args)["preflight"]
    if args.test_only:
        rate = 3000.0
        transitions = len(paths) * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
        return {
            "schema": RUN_SCHEMA + "-resource-projection",
            "schema_version": 1,
            "benchmark_transition_count": transitions,
            "transitions_per_second": rate,
            "certificate_fraction": 1.0,
            "fallback_fraction": 0.0,
            "projected_exact_cache_hours": 1.0,
            "peak_memory_fraction": 0.0,
            "projected_maximum_split_bytes": 1,
            "passed": 1,
        }
    from mnist import diag_d0_jacobi_rb_coarse_residual_learnability as parent_cli
    from mnist.d0_jacobi_rb_learnability import selected_reverse_time

    validation = parent_cli._load_input_cache_for_role(  # noqa: SLF001
        args.parent_coarse_residual_run_dir, "validation"
    )
    indices = np.flatnonzero(
        (validation.phase == 0)
        & (validation.reverse_time == selected_reverse_time(127, 0))
    )[: len(paths)]
    state_np = np.array(validation.later_full_state[indices], copy=True, order="C")
    states = torch.as_tensor(state_np, dtype=torch.float64, device=device).contiguous()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batches = []
    started = time.perf_counter()
    for phase in range(PHASE_COUNT):
        batches.append(
            sample_midpoint_branches(
                states,
                path_ids=paths,
                outer_step=127,
                phase=phase,
                root_seed=ROOT_SEED,
                profile=JacobiRBCudaProfile(),
            )
        )
    elapsed = time.perf_counter() - started
    transitions = sum(batch.transition_count for batch in batches)
    certified = sum(batch.certified_count for batch in batches)
    fallback = sum(int(batch.fallback_mask.sum().item()) for batch in batches)
    fallback_seconds = sum(float(batch.fallback_elapsed_seconds) for batch in batches)
    rate = transitions / max(elapsed, np.finfo(float).tiny)
    role_counts = _effective_paths(args)
    per_path = OUTER_STEPS * PHASE_COUNT * EDGES_PER_PHASE + len(SELECTED_OUTER_STEPS) * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
    projected = sum(len(role_counts[name]) * per_path for name in ("train", "validation", "confirmation"))
    projected_hours = projected / max(rate, np.finfo(float).tiny) / 3600.0
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    total = int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 1
    # A production split stores branch states/targets/codes once in compressed
    # shards.  This conservative raw-byte projection is checked per split.
    maximum_paths = max(len(role_counts[name]) for name in ("train", "validation", "confirmation"))
    rows = maximum_paths * len(SELECTED_OUTER_STEPS) * PHASE_COUNT * MIDPOINT_COUNT
    # Per selected train/validation row: the exact model input rounded once to
    # its actual float32 network dtype, one binary64 target, and one
    # certificate byte per edge. Mode/prefix/fallback data are aggregated into
    # shard metadata rather than duplicated per transition. Confirmation is
    # evaluated streaming and does not add a third persistent branch cache.
    projected_train_bytes = rows * (
        STATE_SIZE * 4 + EDGES_PER_PHASE * (8 + 1)
    )
    validation_rows = len(role_counts["validation"]) * len(SELECTED_OUTER_STEPS) * PHASE_COUNT * MIDPOINT_COUNT
    projected_bytes = projected_train_bytes + validation_rows * (
        STATE_SIZE * 4 + EDGES_PER_PHASE * (8 + 1)
    )
    checks = {
        "certificate_fraction": certified == transitions,
        "fallback_fraction": fallback / max(transitions, 1) <= 1.0e-4,
        "throughput": rate >= 1300.0,
        "fallback_time_fraction": fallback_seconds / max(elapsed, np.finfo(float).tiny) <= 0.10,
        "projected_runtime": projected_hours <= 30.0,
        "peak_memory": peak / total <= 0.80,
        "persisted_split": projected_bytes <= 5 * 1024**3 // 4,
    }
    return {
        "schema": RUN_SCHEMA + "-resource-projection",
        "schema_version": 1,
        "benchmark_transition_count": transitions,
        "benchmark_elapsed_seconds": elapsed,
        "transitions_per_second": rate,
        "certificate_fraction": certified / max(transitions, 1),
        "fallback_fraction": fallback / max(transitions, 1),
        "fallback_time_fraction": fallback_seconds / max(elapsed, np.finfo(float).tiny),
        "projected_transition_count": projected,
        "projected_exact_cache_hours": projected_hours,
        "peak_memory_fraction": peak / total,
        "projected_train_plus_validation_cache_bytes": projected_bytes,
        "projected_maximum_split_bytes": projected_train_bytes,
        "checks": {name: int(value) for name, value in checks.items()},
        "passed": int(all(checks.values())),
    }


def _semantic_path_collision_scan(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Reject overlap with distinct, already versioned path namespaces."""

    from mnist.d0_jacobi_rb_learnability import (
        discover_repository_path_id_claims,
        scan_path_id_collisions,
    )

    roles = _effective_paths(args)
    proposed = tuple(value for values in roles.values() for value in values)
    current_path = (run_dir / "path_id_plan.json").resolve()
    current = _load_json(current_path)
    filtered = []
    same_workflow = 0
    consumed_parent_reservations: list[str] = []
    failed_parent_plan = (args.failed_controller_run_dir / "path_id_plan.json").resolve()
    for claim in discover_repository_path_id_claims(Path.cwd()):
        source = Path(claim.source).resolve()
        if source == current_path:
            continue
        if source == failed_parent_plan and claim.name in {
            "reserved_roles.fresh_selection",
            "reserved_roles.fresh_confirmation",
        }:
            # The failed controller preregistered these child slots without
            # realizing a path in them.  This workflow consumes those exact
            # reservations; it does not ignore any realized parent evidence.
            consumed_parent_reservations.append(claim.name)
            continue
        if source.name == "path_id_plan.json" and source.is_file():
            try:
                other = _load_json(source)
            except ArtifactCompatibilityError:
                other = {}
            prior_run = source.parent
            stochastic_evidence = (
                (prior_run / "cache_gate.json").is_file()
                or (prior_run / "physical_label_open.json").is_file()
                or (
                    any((prior_run / "cache").rglob("*-branch-labels-audit.npz"))
                    if (prior_run / "cache").is_dir()
                    else False
                )
            )
            if (
                other.get("schema") == current.get("schema")
                and other.get("roles") == current.get("roles")
                and not stochastic_evidence
            ):
                same_workflow += 1
                continue
        filtered.append(claim)
    collisions = scan_path_id_collisions(proposed, tuple(filtered))
    record = {
        "schema": RUN_SCHEMA + "-path-collision-scan",
        "schema_version": 1,
        "claim_count": len(filtered),
        "same_workflow_pre_evidence_retry_count": same_workflow,
        "consumed_parent_reservations": sorted(set(consumed_parent_reservations)),
        "consumed_parent_reservation_count": len(set(consumed_parent_reservations)),
        "candidate_path_count": len(proposed),
        "collision_count": len(collisions),
        "collisions": [
            {
                "source": item.source,
                "name": item.name,
                "path_ids": list(item.path_ids),
            }
            for item in collisions
        ],
        "passed": int(not collisions),
    }
    atomic_write_json(run_dir / "path_collision_scan.json", record)
    return record


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = run_dir / "preflight_gate.json"
    sealed_names = (
        "parent_provenance.json",
        "failed_controller_readjudication.json",
        "path_id_plan.json",
        "path_collision_scan.json",
        "boundary_tangent_representation_preflight.json",
        "legacy_boundary_diagnosis.json",
        "resource_projection.json",
        "boundary_tangent_target_contract.json",
        "preflight_gate.json",
    )
    if existing.is_file():
        seal = _load_json(run_dir / "preflight_artifact_seal.json")
        seal_body = dict(seal)
        semantic = seal_body.pop("semantic_sha256", None)
        artifacts = seal.get("artifacts")
        if (
            semantic != config_fingerprint(seal_body)
            or not isinstance(artifacts, list)
            or tuple(item.get("path") for item in artifacts) != sealed_names
        ):
            raise ArtifactCompatibilityError("preflight artifact seal changed")
        for item in seal.get("artifacts", []):
            path = run_dir / str(item["path"])
            if item.get("sha256") != file_fingerprint(path):
                raise ArtifactCompatibilityError("sealed preflight artifact changed")
        gate = _load_json(existing)
        if not _passed(gate):
            raise ArtifactCompatibilityError("completed preflight gate did not pass")
        return gate
    if not args.test_only:
        configure_exact_torch_backend(args.device)
        if args.device != "cuda" or not torch.cuda.is_available():
            raise BoundaryTangentCLIError(
                "authorizing preflight requires CUDA",
                failure_domain="resource",
                failure_code="boundary_tangent_cuda_unavailable",
            )
    device = torch.device(args.device)
    provenance = _load_json(run_dir / "parent_provenance.json")
    path_plan = _load_json(run_dir / "path_id_plan.json")
    path_validation = validate_boundary_tangent_path_plan(path_plan)
    collision_scan = _semantic_path_collision_scan(run_dir, args)
    readjudication = _load_json(run_dir / "failed_controller_readjudication.json")
    representation = _representation_preflight(device, _effective_paths(args)["train"])
    historical = _historical_boundary_diagnosis(args, device)
    resource = _branch_resource_preflight(args, device)
    atomic_write_json(run_dir / "boundary_tangent_representation_preflight.json", representation)
    atomic_write_json(run_dir / "legacy_boundary_diagnosis.json", historical)
    atomic_write_json(run_dir / "resource_projection.json", resource)
    contract = {
        "schema": RUN_SCHEMA + "-target-contract",
        "allowed_model_input_fields": list(MODEL_INPUT_FIELDS),
        "target": "unchanged exact binary64 Rao-Blackwell label",
        "objective": "plain unweighted raw-target MSE",
        "prediction": "m=y(1-y)*q",
        "quotient_target_forbidden": 1,
        "quotient_target_persisted": 0,
        "clip_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "projection_count": 0,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "boundary_tangent_target_contract.json", contract)
    checks = {
        "parent_provenance": int(provenance.get("passed", 0)) == 1,
        "failed_controller_readjudication": int(readjudication.get("passed", 0)) == 1,
        "path_plan": int(path_validation.get("passed", 0)) == 1,
        "path_collision_scan": int(collision_scan.get("passed", 0)) == 1,
        "representation": int(representation.get("passed", 0)) == 1,
        "legacy_boundary_diagnosis": int(historical.get("passed", 0)) == 1,
        "resource_projection": int(resource.get("passed", 0)) == 1,
        "target_contract": contract["quotient_target_persisted"] == 0,
    }
    gate = _gate(
        "preflight",
        checks,
        numerically_valid=int(
            checks["representation"] and checks["legacy_boundary_diagnosis"]
        ),
        resource_valid=int(checks["resource_projection"]),
        provenance_valid=int(checks["parent_provenance"]),
        failed_controller_adjudication_valid=int(checks["failed_controller_readjudication"]),
        boundary_tangent_representation_valid=int(checks["representation"]),
    )
    atomic_write_json(existing, gate)
    seal = {
        "schema": RUN_SCHEMA + "-preflight-artifact-seal",
        "schema_version": 1,
        "artifacts": [
            {"path": name, "sha256": file_fingerprint(run_dir / name)}
            for name in sealed_names
        ],
        **NO_WORK,
    }
    seal["semantic_sha256"] = config_fingerprint(seal)
    atomic_write_json(
        run_dir / "preflight_artifact_seal.json",
        seal,
    )
    return gate


def _test_sampler(
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    rng_key: Any,
    transition_ids: Tensor,
    profile: JacobiRBCudaProfile,
) -> Any:
    """Deterministic CPU test double with the exact production result schema."""

    del rng_key, transition_ids, profile
    from types import SimpleNamespace

    later = torch.clamp(
        head_fraction + 0.01 * torch.tanh(exposure) * (0.5 - head_fraction),
        0.0,
        1.0,
    )
    target = later * (1.0 - later) * (0.25 - later)
    count = int(later.numel())
    return SimpleNamespace(
        later_head_fraction=later,
        denoising_target=target,
        certificate_codes=torch.full_like(later, 0b1111, dtype=torch.uint8),
        certified_mask=torch.ones_like(later, dtype=torch.bool),
        active_mask=torch.ones_like(later, dtype=torch.bool),
        strengthened_mask=torch.zeros_like(later, dtype=torch.bool),
        fallback_mask=torch.zeros_like(later, dtype=torch.bool),
        mode_counts=torch.full_like(later, 16, dtype=torch.int32),
        prefix_bits=torch.full_like(later, 64, dtype=torch.int32),
        arb_fallback_reason_codes=torch.zeros_like(later, dtype=torch.uint8),
        diagnostics={
            "maximum_cuda_launch_lanes": count,
            "fused_authorizer_launch_count": 1,
            "arb_fallback_elapsed_seconds": 0.0,
            "fused_authorizer_elapsed_seconds": 1.0e-6,
            "candidate_elapsed_seconds": 1.0e-6,
            **{name: 0 for name in FORBIDDEN_COUNTS},
        },
    )


def _cache_paths(
    run_dir: Path, *, role: str, cohort_index: int, start_step: int
) -> tuple[Path, Path, Path, Path]:
    directory = run_dir / "cache" / f"{role}_shards" / f"cohort-{cohort_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step-{start_step:03d}"
    return (
        directory / f"{stem}-state.npz",
        directory / f"{stem}-branch-inputs.npz",
        directory / f"{stem}-branch-labels-audit.npz",
        directory / f"{stem}.json",
    )


def _cohorts(paths: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(item) for item in paths[start : start + 8]) for start in range(0, len(paths), 8))


def _selected_step_in_shard(start_step: int, outer_steps: int) -> int | None:
    values = [
        step
        for step in SELECTED_OUTER_STEPS
        if start_step <= step < start_step + SHARD_STEPS and step < outer_steps
    ]
    if len(values) > 1:
        raise AssertionError("an eight-step shard contains multiple selected rows")
    return values[0] if values else None


def _valid_cache_shard(
    run_dir: Path,
    *,
    role: str,
    cohort_index: int,
    path_ids: Sequence[int],
    start_step: int,
    current: np.ndarray,
    selected_step: int | None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    state_path, branch_input_path, branch_label_path, metadata_path = _cache_paths(
        run_dir, role=role, cohort_index=cohort_index, start_step=start_step
    )
    if not state_path.is_file() or not metadata_path.is_file():
        return None
    try:
        record = _load_json(metadata_path)
        body = dict(record)
        semantic = body.pop("semantic_sha256", None)
        expected = {
            "schema": RUN_SCHEMA + "-cache-shard",
            "schema_version": 1,
            "role": role,
            "cohort_index": cohort_index,
            "path_ids": list(path_ids),
            "start_step": start_step,
            "selected_step": selected_step,
            "input_state_sha256": _array_sha(current),
            "scientific_config_sha256": _load_json(run_dir / "scientific_config.json")["semantic_sha256"],
            "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")["semantic_sha256"],
        }
        if semantic != config_fingerprint(body) or any(record.get(key) != value for key, value in expected.items()):
            return None
        if record.get("state_file_sha256") != file_fingerprint(state_path):
            return None
        state_arrays = _load_npz(state_path)
        final = np.asarray(state_arrays.get("final_states"))
        if (
            final.dtype != np.float64
            or final.shape != (len(path_ids), STATE_SIZE)
            or record.get("final_state_sha256") != _array_sha(final)
            or not np.isfinite(final).all()
            or np.any(final < 0.0)
        ):
            return None
        if selected_step is not None:
            if (
                not branch_input_path.is_file()
                or not branch_label_path.is_file()
                or record.get("branch_input_file_sha256") != file_fingerprint(branch_input_path)
                or record.get("branch_label_file_sha256") != file_fingerprint(branch_label_path)
            ):
                return None
            input_arrays = _load_npz(branch_input_path)
            label_arrays = _load_npz(branch_label_path)
            required_inputs = {
                "path_ids", "outer_step", "phases", "midpoint_fractions",
                "later_full_state",
            }
            required_labels = {
                "path_ids", "outer_step", "phases", "midpoint_fractions",
                "denoising_target", "certificate_codes",
            }
            if set(input_arrays) != required_inputs or set(label_arrays) != required_labels:
                return None
            for name in ("path_ids", "outer_step", "phases", "midpoint_fractions"):
                if not np.array_equal(input_arrays[name], label_arrays[name]):
                    return None
            expected_branch_shape = (
                PHASE_COUNT,
                MIDPOINT_COUNT,
                len(path_ids),
            )
            expected_identity = {
                "path_ids": np.asarray(path_ids, dtype=np.int64),
                "outer_step": np.asarray([selected_step], dtype=np.int16),
                "phases": np.arange(PHASE_COUNT, dtype=np.int8),
                "midpoint_fractions": np.asarray(MIDPOINT_FRACTIONS, dtype=np.float64),
            }
            if any(
                not np.array_equal(input_arrays[name], value)
                for name, value in expected_identity.items()
            ):
                return None
            later = np.asarray(input_arrays["later_full_state"])
            target = np.asarray(label_arrays["denoising_target"])
            codes = np.asarray(label_arrays["certificate_codes"])
            if (
                later.dtype != np.float32
                or later.shape != expected_branch_shape + (STATE_SIZE,)
                or target.dtype != np.float64
                or target.shape != expected_branch_shape + (EDGES_PER_PHASE,)
                or codes.dtype != np.uint8
                or codes.shape != target.shape
                or not np.isfinite(later).all()
                or not np.isfinite(target).all()
                or np.any(later < 0.0)
                or not np.all((codes & np.uint8(0b1111)) == np.uint8(0b1111))
            ):
                return None
            diagnostics = record.get("branch_diagnostics")
            if (
                not isinstance(diagnostics, Mapping)
                or int(diagnostics.get("transition_count", -1))
                != len(path_ids) * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
                or int(diagnostics.get("certified_count", -1))
                != int(diagnostics.get("transition_count", -2))
                or not isinstance(diagnostics.get("forbidden_counts"), Mapping)
                or any(
                    int(diagnostics["forbidden_counts"].get(name, -1)) != 0
                    for name in FORBIDDEN_COUNTS
                    if name != "uncertified_count"
                )
            ):
                return None
        return np.ascontiguousarray(final), record
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError):
        return None


def _persist_cache_shard(
    run_dir: Path,
    *,
    role: str,
    cohort_index: int,
    path_ids: Sequence[int],
    start_step: int,
    selected_step: int | None,
    input_state_sha256: str,
    result: Any,
    branch_inputs: Mapping[str, np.ndarray] | None,
    branch_labels: Mapping[str, np.ndarray] | None,
    branch_diagnostics: Mapping[str, Any],
    started_at: float,
    device: torch.device,
) -> dict[str, Any]:
    state_path, branch_input_path, branch_label_path, metadata_path = _cache_paths(
        run_dir, role=role, cohort_index=cohort_index, start_step=start_step
    )
    final = np.ascontiguousarray(result.committed_final_states, dtype=np.float64)
    state_artifact = _atomic_npz(state_path, {"final_states": final})
    input_artifact = _atomic_npz(branch_input_path, branch_inputs) if branch_inputs is not None else None
    label_artifact = _atomic_npz(branch_label_path, branch_labels) if branch_labels is not None else None
    if (input_artifact is None) != (label_artifact is None):
        raise AssertionError("branch input/label artifacts must be committed together")
    record = {
        "schema": RUN_SCHEMA + "-cache-shard",
        "schema_version": 1,
        "role": role,
        "cohort_index": cohort_index,
        "path_ids": list(path_ids),
        "start_step": start_step,
        "selected_step": selected_step,
        "input_state_sha256": input_state_sha256,
        "final_state_sha256": _array_sha(final),
        "state_file_sha256": state_artifact["sha256"],
        "state_file_size": state_artifact["size"],
        "branch_input_file_sha256": None if input_artifact is None else input_artifact["sha256"],
        "branch_input_file_size": None if input_artifact is None else input_artifact["size"],
        "branch_label_file_sha256": None if label_artifact is None else label_artifact["sha256"],
        "branch_label_file_size": None if label_artifact is None else label_artifact["size"],
        "scientific_config_sha256": _load_json(run_dir / "scientific_config.json")["semantic_sha256"],
        "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")["semantic_sha256"],
        "scheduler_record": result.to_record(),
        "branch_diagnostics": dict(branch_diagnostics),
        "complete_pipeline_elapsed_seconds": float(time.perf_counter() - started_at),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 1,
        "committed": 1,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(metadata_path, record)
    return record


def _branch_arrays(
    result: Any,
    selected_step: int,
    path_ids: Sequence[int],
    args: argparse.Namespace,
    role: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    capture = result.capture_payload
    if capture is None:
        raise BoundaryTangentCLIError(
            "selected shard returned no phase-state trace",
            failure_domain="cache_capture",
            failure_code="boundary_tangent_phase_trace_missing",
        )
    capture_paths = tuple(int(value) for value in getattr(capture, "path_ids", path_ids))
    if capture_paths != tuple(int(value) for value in path_ids):
        raise BoundaryTangentCLIError(
            "captured phase-state path order changed",
            failure_domain="cache_capture",
            failure_code="boundary_tangent_capture_path_order_invalid",
        )
    # The selected step is local step seven in every frozen selected shard.
    local = selected_step - int(capture.start_step)
    if local != 7:
        raise AssertionError("selected K=512 row is not the seventh local shard step")
    post_trace = np.asarray(capture.post_phase_states, dtype=np.float64)
    pre_states = np.stack([post_trace[local * PHASE_COUNT + phase - 1] if phase else post_trace[local * PHASE_COUNT - 1] for phase in range(PHASE_COUNT)])
    post_states = np.stack([post_trace[local * PHASE_COUNT + phase] for phase in range(PHASE_COUNT)])
    device = result.final_states.device
    branch_batches = []
    sampler = _test_sampler if args.test_only else None
    for phase in range(PHASE_COUNT):
        state = torch.as_tensor(
            np.array(pre_states[phase], copy=True, order="C"), dtype=torch.float64, device=device
        ).contiguous()
        kwargs: dict[str, Any] = {}
        if sampler is not None:
            kwargs["sampler"] = sampler
        branch_batches.append(
            sample_midpoint_branches(
                state,
                path_ids=path_ids,
                outer_step=selected_step,
                phase=phase,
                root_seed=ROOT_SEED,
                profile=JacobiRBCudaProfile(),
                **kwargs,
            )
        )
    identity = {
        "path_ids": np.asarray(path_ids, dtype=np.int64),
        "outer_step": np.asarray([selected_step], dtype=np.int16),
        "phases": np.arange(PHASE_COUNT, dtype=np.int8),
        "midpoint_fractions": np.asarray(MIDPOINT_FRACTIONS, dtype=np.float64),
    }
    input_arrays = {
        **identity,
        "later_full_state": np.ascontiguousarray(
            np.stack([batch.later_full_state.detach().cpu().numpy() for batch in branch_batches]),
            dtype=np.float32,
        ),
    }
    label_arrays = {
        **identity,
        "denoising_target": np.ascontiguousarray(
            np.stack([batch.denoising_target.detach().cpu().numpy() for batch in branch_batches])
        ),
        "certificate_codes": np.ascontiguousarray(
            np.stack([batch.certificate_codes.detach().cpu().numpy() for batch in branch_batches])
        ),
    }
    mode_counts = np.stack([batch.mode_counts.detach().cpu().numpy() for batch in branch_batches])
    prefix_bits = np.stack([batch.prefix_bits.detach().cpu().numpy() for batch in branch_batches])
    fallback_count = sum(int(batch.fallback_mask.sum().item()) for batch in branch_batches)
    strengthened_count = sum(int(batch.strengthened_mask.sum().item()) for batch in branch_batches)
    transitions = sum(batch.transition_count for batch in branch_batches)
    diagnostics = {
        "transition_count": transitions,
        "certified_count": sum(batch.certified_count for batch in branch_batches),
        "fallback_count": fallback_count,
        "strengthened_count": strengthened_count,
        "fallback_elapsed_seconds": sum(
            float(batch.fallback_elapsed_seconds) for batch in branch_batches
        ),
        "backend_elapsed_seconds": sum(
            float(batch.backend_elapsed_seconds) for batch in branch_batches
        ),
        "maximum_mode_count": int(mode_counts.max(initial=0)),
        "maximum_prefix_bits": int(prefix_bits.max(initial=0)),
        "mode_count_histogram": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(mode_counts, return_counts=True), strict=True)
        },
        "prefix_bit_histogram": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(prefix_bits, return_counts=True), strict=True)
        },
        "forbidden_counts": {
            name: sum(int(batch.forbidden_counts[name]) for batch in branch_batches)
            for name in FORBIDDEN_COUNTS
            if name != "uncertified_count"
        },
    }
    return input_arrays, label_arrays, diagnostics


def _generate_role_cache(
    run_dir: Path, args: argparse.Namespace, *, role: str, path_ids: Sequence[int]
) -> dict[str, Any]:
    if role not in {"train", "validation", "confirmation"}:
        raise ValueError("unknown cache role")
    device = torch.device(args.device)
    outer_steps = int(args.test_outer_steps) if args.test_only else OUTER_STEPS
    if outer_steps % SHARD_STEPS:
        raise ValueError("executed outer steps must be divisible by eight")
    source = _load_source_target(args.parent_coarse_residual_run_dir)
    records: list[dict[str, Any]] = []
    recomputed = 0
    profile = JacobiRBCudaProfile()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for cohort_index, cohort in enumerate(_cohorts(path_ids)):
        current = np.repeat(source[None, :], len(cohort), axis=0).copy(order="C")
        recompute_tail = False
        for start_step in range(0, outer_steps, SHARD_STEPS):
            selected = _selected_step_in_shard(start_step, outer_steps)
            cached = None if recompute_tail else _valid_cache_shard(
                run_dir,
                role=role,
                cohort_index=cohort_index,
                path_ids=cohort,
                start_step=start_step,
                current=current,
                selected_step=selected,
            )
            if cached is not None:
                current, record = cached
                records.append(record)
                continue
            recompute_tail = True
            recomputed += 1
            states = torch.as_tensor(
                np.array(current, copy=True, order="C"), dtype=torch.float64, device=device
            ).contiguous()
            sampler = _test_sampler if args.test_only else None
            kwargs: dict[str, Any] = {}
            if sampler is not None:
                kwargs["sampler"] = sampler
            started = time.perf_counter()
            result = run_exact_multipath_shard(
                states,
                path_ids=cohort,
                start_step=start_step,
                root_seed=ROOT_SEED,
                profile=profile,
                group_sizes=(len(cohort),),
                capture_training_payload=selected is not None,
                **kwargs,
            )
            branch_inputs = None
            branch_labels = None
            branch_diagnostics = {"transition_count": 0, "certified_count": 0, "fallback_count": 0}
            if selected is not None:
                branch_inputs, branch_labels, branch_diagnostics = _branch_arrays(
                    result, selected, cohort, args, role
                )
            record = _persist_cache_shard(
                run_dir,
                role=role,
                cohort_index=cohort_index,
                path_ids=cohort,
                start_step=start_step,
                selected_step=selected,
                input_state_sha256=_array_sha(current),
                result=result,
                branch_inputs=branch_inputs,
                branch_labels=branch_labels,
                branch_diagnostics=branch_diagnostics,
                started_at=started,
                device=device,
            )
            current = np.ascontiguousarray(result.committed_final_states, dtype=np.float64)
            records.append(record)
            print(
                f"{role} cohort {cohort_index + 1}/{len(_cohorts(path_ids))} "
                f"shard {start_step // 8 + 1}/{outer_steps // 8} committed",
                flush=True,
            )
    diagnostics = [record["scheduler_record"]["diagnostics"] for record in records]
    main_transitions = sum(int(item["transition_count"]) for item in diagnostics)
    main_certified = sum(int(item.get("certified_count", 0)) for item in diagnostics)
    main_fallback = sum(int(item.get("fallback_count", 0)) for item in diagnostics)
    main_fallback_seconds = sum(
        float(item.get("fallback_elapsed_seconds", 0.0)) for item in diagnostics
    )
    branch_transitions = sum(int(record.get("branch_diagnostics", {}).get("transition_count", 0)) for record in records)
    branch_certified = sum(int(record.get("branch_diagnostics", {}).get("certified_count", 0)) for record in records)
    branch_fallback = sum(
        int(record.get("branch_diagnostics", {}).get("fallback_count", 0))
        for record in records
    )
    branch_fallback_seconds = sum(
        float(record.get("branch_diagnostics", {}).get("fallback_elapsed_seconds", 0.0))
        for record in records
    )
    transition_count = main_transitions + branch_transitions
    certified_count = main_certified + branch_certified
    elapsed = sum(float(record["complete_pipeline_elapsed_seconds"]) for record in records)
    expected_rows = len(path_ids) * len([step for step in SELECTED_OUTER_STEPS if step < outer_steps]) * PHASE_COUNT * MIDPOINT_COUNT
    committed_rows = sum(
        len(record["path_ids"]) * PHASE_COUNT * MIDPOINT_COUNT
        for record in records
        if record.get("selected_step") is not None
    )
    persisted = sum(
        int(record.get("state_file_size", 0))
        + int(record.get("branch_input_file_size") or 0)
        + int(record.get("branch_label_file_size") or 0)
        + int(
            _cache_paths(
                run_dir,
                role=role,
                cohort_index=int(record["cohort_index"]),
                start_step=int(record["start_step"]),
            )[3].stat().st_size
        )
        for record in records
    )
    maximum_mass_error = max(float(item.get("maximum_mass_error", math.inf)) for item in diagnostics)
    forbidden = {}
    for name in FORBIDDEN_COUNTS:
        main_value = sum(int(item.get(name, 0)) for item in diagnostics)
        if name == "uncertified_count":
            branch_value = branch_transitions - branch_certified
        else:
            branch_value = sum(
                int(record.get("branch_diagnostics", {})
                    .get("forbidden_counts", {})
                    .get(name, 0))
                for record in records
            )
        forbidden[name] = main_value + branch_value
    metrics = {
        "schema": RUN_SCHEMA + "-role-cache-metrics",
        "schema_version": 1,
        "role": role,
        "path_ids": list(path_ids),
        "path_count": len(path_ids),
        "shard_count": len(records),
        "recomputed_shard_count": recomputed,
        "selected_row_count": committed_rows,
        "expected_selected_row_count": expected_rows,
        "transition_count": transition_count,
        "certified_count": certified_count,
        "certificate_fraction": certified_count / max(transition_count, 1),
        "fallback_count": main_fallback + branch_fallback,
        "fallback_fraction": (main_fallback + branch_fallback) / max(transition_count, 1),
        "fallback_time_fraction": (main_fallback_seconds + branch_fallback_seconds)
        / max(elapsed, np.finfo(float).tiny),
        "maximum_mass_error": maximum_mass_error,
        "transitions_per_second": transition_count / max(elapsed, np.finfo(float).tiny),
        "peak_memory_fraction": max(
            float(record.get("peak_memory_bytes", 0)) / max(1, int(record.get("device_total_memory_bytes", 1)))
            for record in records
        ),
        "persisted_cache_bytes": persisted,
        "confirmation_created_after_selection": int(role != "confirmation" or (run_dir / "confirmation_seal.json").is_file()),
        "input_and_labels_separate": 1,
        "quotient_target_persisted": 0,
        **forbidden,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "cache" / f"{role}_metrics.json", metrics)
    index = {
        "schema": RUN_SCHEMA + "-role-cache-index",
        "schema_version": 1,
        "role": role,
        "path_ids": list(path_ids),
        "shards": [
            {
                "cohort_index": int(record["cohort_index"]),
                "start_step": int(record["start_step"]),
                "selected_step": record["selected_step"],
                "metadata_sha256": file_fingerprint(
                    _cache_paths(
                        run_dir,
                        role=role,
                        cohort_index=int(record["cohort_index"]),
                        start_step=int(record["start_step"]),
                    )[3]
                ),
            }
            for record in records
        ],
        "metrics_sha256": file_fingerprint(run_dir / "cache" / f"{role}_metrics.json"),
    }
    index["semantic_sha256"] = config_fingerprint(index)
    atomic_write_json(run_dir / "cache" / f"{role}_index.json", index)
    return metrics


def _cache_metric_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "rows": int(metrics.get("selected_row_count", -1)) == int(metrics.get("expected_selected_row_count", -2)),
        "certification": float(metrics.get("certificate_fraction", 0.0)) == 1.0,
        "fallback_fraction": float(metrics.get("fallback_fraction", math.inf)) <= 1.0e-4,
        "fallback_time": float(metrics.get("fallback_time_fraction", math.inf)) <= 0.10,
        "mass": float(metrics.get("maximum_mass_error", math.inf)) <= 2.0e-12,
        "throughput": float(metrics.get("transitions_per_second", 0.0)) >= 1300.0,
        "memory": float(metrics.get("peak_memory_fraction", math.inf)) <= 0.80,
        "persisted": int(metrics.get("persisted_cache_bytes", 1 << 62)) <= 5 * 1024**3 // 4,
        "separation": int(metrics.get("input_and_labels_separate", 0)) == 1,
        "raw_target": int(metrics.get("quotient_target_persisted", 1)) == 0,
        "forbidden": all(int(metrics.get(name, 0)) == 0 for name in FORBIDDEN_COUNTS),
    }


def _cache_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("cache stage requires a passing preflight")
    existing = run_dir / "cache_gate.json"
    if existing.is_file():
        for role in ("train", "validation"):
            _verify_role_cache_integrity(run_dir, role)
        gate = _load_json(existing)
        if not _passed(gate):
            raise ArtifactCompatibilityError("completed cache gate did not pass")
        return gate
    paths = _effective_paths(args)
    role_metrics = {
        role: _generate_role_cache(run_dir, args, role=role, path_ids=paths[role])
        for role in ("train", "validation")
    }
    checks: dict[str, bool] = {"confirmation_absent": not (run_dir / "confirmation_seal.json").exists()}
    for role, metrics in role_metrics.items():
        for name, passed in _cache_metric_checks(metrics).items():
            checks[f"{role}.{name}"] = passed
    total_persisted = sum(
        int(metrics.get("persisted_cache_bytes", 0)) for metrics in role_metrics.values()
    )
    checks["total_persisted_cache"] = total_persisted <= 5 * 1024**3 // 4
    gate = _gate(
        "cache",
        checks,
        role_metrics=role_metrics,
        numerically_valid=int(all(
            value for name, value in checks.items()
            if not any(token in name for token in ("throughput", "memory", "persisted", "fallback"))
        )),
        resource_valid=int(all(
            value for name, value in checks.items()
            if any(token in name for token in ("throughput", "memory", "persisted", "fallback"))
        )),
        total_persisted_cache_bytes=total_persisted,
    )
    atomic_write_json(existing, gate)
    return gate


def _verify_role_cache_integrity(run_dir: Path, role: str) -> None:
    config = _load_json(run_dir / "scientific_config.json")
    plan = _load_json(run_dir / "path_id_plan.json")
    config_sha = str(config["semantic_sha256"])
    plan_sha = str(plan["semantic_sha256"])
    expected_paths = tuple(int(value) for value in config["path_ids"][role])
    outer_steps = int(config["executed_outer_steps"])
    index_path = run_dir / "cache" / f"{role}_index.json"
    index = _load_json(index_path)
    semantic = index.get("semantic_sha256")
    body = dict(index)
    body.pop("semantic_sha256", None)
    if semantic != config_fingerprint(body):
        raise ArtifactCompatibilityError(f"{role} cache index semantic hash changed")
    metrics_path = run_dir / "cache" / f"{role}_metrics.json"
    if index.get("metrics_sha256") != file_fingerprint(metrics_path):
        raise ArtifactCompatibilityError(f"{role} cache metrics changed")
    if tuple(int(value) for value in index.get("path_ids", ())) != expected_paths:
        raise ArtifactCompatibilityError(f"{role} cache path role changed")
    items = list(index.get("shards", []))
    expected_count = len(_cohorts(expected_paths)) * (outer_steps // SHARD_STEPS)
    if len(items) != expected_count:
        raise ArtifactCompatibilityError(f"{role} cache shard count changed")
    previous_by_cohort: dict[int, tuple[int, str]] = {}
    for item in sorted(items, key=lambda value: (int(value["cohort_index"]), int(value["start_step"]))):
        cohort = int(item["cohort_index"])
        start = int(item["start_step"])
        state_path, input_path, label_path, metadata_path = _cache_paths(
            run_dir, role=role, cohort_index=cohort, start_step=start
        )
        if item.get("metadata_sha256") != file_fingerprint(metadata_path):
            raise ArtifactCompatibilityError(f"{role} shard metadata changed")
        metadata = _load_json(metadata_path)
        semantic = metadata.get("semantic_sha256")
        metadata_body = dict(metadata)
        metadata_body.pop("semantic_sha256", None)
        if semantic != config_fingerprint(metadata_body):
            raise ArtifactCompatibilityError(f"{role} shard semantic hash changed")
        expected_cohort = _cohorts(expected_paths)[cohort]
        if (
            metadata.get("scientific_config_sha256") != config_sha
            or metadata.get("path_plan_sha256") != plan_sha
            or metadata.get("role") != role
            or int(metadata.get("cohort_index", -1)) != cohort
            or tuple(int(value) for value in metadata.get("path_ids", ())) != expected_cohort
            or int(metadata.get("start_step", -1)) != start
            or metadata.get("selected_step") != item.get("selected_step")
            or metadata.get("selected_step") != _selected_step_in_shard(start, outer_steps)
            or start % SHARD_STEPS != 0
            or not 0 <= start < outer_steps
        ):
            raise ArtifactCompatibilityError(f"{role} shard binding changed")
        previous = previous_by_cohort.get(cohort)
        if previous is not None:
            previous_start, previous_final = previous
            if start != previous_start + SHARD_STEPS or metadata.get("input_state_sha256") != previous_final:
                raise ArtifactCompatibilityError(f"{role} shard chain changed")
        elif (run_dir / "parent_provenance.json").is_file():
            parent = _load_json(run_dir / "parent_provenance.json")["parents"][
                "successful_coarse_residual"
            ]["run_dir"]
            initial = np.repeat(
                _load_source_target(Path(parent))[None, :],
                len(expected_cohort),
                axis=0,
            ).copy(order="C")
            if metadata.get("input_state_sha256") != _array_sha(initial):
                raise ArtifactCompatibilityError(f"{role} initial state changed")
        if metadata.get("state_file_sha256") != file_fingerprint(state_path):
            raise ArtifactCompatibilityError(f"{role} shard state changed")
        state = _load_npz(state_path).get("final_states")
        if (
            state is None
            or state.dtype != np.float64
            or state.shape != (len(expected_cohort), STATE_SIZE)
            or not np.isfinite(state).all()
            or np.any(state < 0.0)
            or metadata.get("final_state_sha256") != _array_sha(state)
        ):
            raise ArtifactCompatibilityError(f"{role} shard final state changed")
        previous_by_cohort[cohort] = (start, str(metadata["final_state_sha256"]))
        if metadata.get("selected_step") is not None:
            if (
                metadata.get("branch_input_file_sha256") != file_fingerprint(input_path)
                or metadata.get("branch_label_file_sha256") != file_fingerprint(label_path)
            ):
                raise ArtifactCompatibilityError(f"{role} branch cache changed")


def _load_role_cache_arrays(
    run_dir: Path, role: str, *, open_labels: bool
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Load a split in canonical path/step/phase/midpoint order."""

    _verify_role_cache_integrity(run_dir, role)
    index = _load_json(run_dir / "cache" / f"{role}_index.json")
    input_chunks: list[dict[str, np.ndarray]] = []
    label_chunks: list[dict[str, np.ndarray]] = []
    for item in index.get("shards", []):
        if item.get("selected_step") is None:
            continue
        cohort = int(item["cohort_index"])
        start = int(item["start_step"])
        _, input_path, label_path, metadata_path = _cache_paths(
            run_dir, role=role, cohort_index=cohort, start_step=start
        )
        if item.get("metadata_sha256") != file_fingerprint(metadata_path):
            raise ArtifactCompatibilityError(f"{role} cache-index shard changed")
        input_chunks.append(_load_npz(input_path))
        if open_labels:
            label_chunks.append(_load_npz(label_path))
    if not input_chunks:
        raise ArtifactCompatibilityError(f"{role} cache contains no selected branches")

    paths: list[np.ndarray] = []
    steps: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    midpoints: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for chunk_index, inputs in enumerate(input_chunks):
        path_ids = np.asarray(inputs["path_ids"], dtype=np.int64)
        step = int(np.asarray(inputs["outer_step"]).reshape(-1)[0])
        phase_values = np.asarray(inputs["phases"], dtype=np.int8)
        fraction_values = np.asarray(inputs["midpoint_fractions"], dtype=np.float64)
        later = np.asarray(inputs["later_full_state"])
        expected = (PHASE_COUNT, MIDPOINT_COUNT, len(path_ids), STATE_SIZE)
        if later.shape != expected or later.dtype != np.dtype(np.float32):
            raise ArtifactCompatibilityError(f"{role} branch input shape changed")
        # File order is [phase,midpoint,path,...]; canonical learning order is
        # path,step,phase,midpoint.
        states.append(np.transpose(later, (2, 0, 1, 3)).reshape(-1, STATE_SIZE))
        rows = len(path_ids) * PHASE_COUNT * MIDPOINT_COUNT
        paths.append(np.repeat(path_ids, PHASE_COUNT * MIDPOINT_COUNT))
        steps.append(np.full(rows, step, dtype=np.int16))
        phases.append(np.tile(np.repeat(phase_values, MIDPOINT_COUNT), len(path_ids)))
        midpoints.append(np.tile(np.arange(MIDPOINT_COUNT, dtype=np.int8), len(path_ids) * PHASE_COUNT))
        fractions.append(np.tile(fraction_values, len(path_ids) * PHASE_COUNT))
        if open_labels:
            labels = label_chunks[chunk_index]
            for identity in ("path_ids", "outer_step", "phases", "midpoint_fractions"):
                if not np.array_equal(inputs[identity], labels[identity]):
                    raise ArtifactCompatibilityError(f"{role} input/label join changed")
            target = np.asarray(labels["denoising_target"], dtype=np.float64)
            certificate = np.asarray(labels["certificate_codes"], dtype=np.uint8)
            if target.shape != (PHASE_COUNT, MIDPOINT_COUNT, len(path_ids), EDGES_PER_PHASE):
                raise ArtifactCompatibilityError(f"{role} target shape changed")
            targets.append(np.transpose(target, (2, 0, 1, 3)).reshape(-1, EDGES_PER_PHASE))
            codes.append(np.transpose(certificate, (2, 0, 1, 3)).reshape(-1, EDGES_PER_PHASE))
    path_id = np.concatenate(paths)
    outer_step = np.concatenate(steps)
    phase = np.concatenate(phases)
    midpoint_index = np.concatenate(midpoints)
    midpoint_fraction = np.concatenate(fractions)
    later_state = np.concatenate(states)
    order = np.lexsort((midpoint_index, phase, outer_step, path_id))
    path_id = np.ascontiguousarray(path_id[order])
    outer_step = np.ascontiguousarray(outer_step[order])
    phase = np.ascontiguousarray(phase[order])
    midpoint_index = np.ascontiguousarray(midpoint_index[order])
    midpoint_fraction = np.ascontiguousarray(midpoint_fraction[order])
    later_state = np.ascontiguousarray(later_state[order])
    from mnist.d0_jacobi_rb_boundary_tangent_cache import midpoint_sample_key
    from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time

    sample_key = np.asarray(
        [
            midpoint_sample_key(int(path), int(step), int(occurrence), int(midpoint))
            for path, step, occurrence, midpoint in zip(
                path_id, outer_step, phase, midpoint_index, strict=True
            )
        ],
        dtype=np.int64,
    )
    reverse_time = np.asarray(
        [
            internal_reverse_time(int(step), int(occurrence), float(fraction))
            for step, occurrence, fraction in zip(
                outer_step, phase, midpoint_fraction, strict=True
            )
        ],
        dtype=np.float64,
    )
    inputs = {
        "sample_key": sample_key,
        # Join coordinates remain audit metadata.  `_model_inputs_from_arrays`
        # deliberately exposes only the six permitted model fields below.
        "path_id": path_id,
        "outer_step": outer_step,
        "midpoint_index": midpoint_index,
        "midpoint_fraction": midpoint_fraction,
        "later_full_state": later_state,
        "reverse_time": reverse_time,
        "phase": phase,
        "color": np.asarray([PHASE_MATCHINGS[int(value)] for value in phase], dtype=np.int8),
        "duration": np.asarray([PHASE_DURATIONS[int(value)] for value in phase], dtype=np.float64),
        "label": np.full(len(path_id), 3, dtype=np.int64),
    }
    audit = None
    if open_labels:
        target = np.concatenate(targets)[order]
        certificate = np.concatenate(codes)[order]
        audit = {
            "sample_key": sample_key.copy(),
            "path_id": path_id,
            "outer_step": outer_step,
            "phase": phase.copy(),
            "midpoint_index": midpoint_index,
            "midpoint_fraction": midpoint_fraction,
            "denoising_target": np.ascontiguousarray(target),
            "certificate_codes": np.ascontiguousarray(certificate),
        }
        if not np.array_equal(inputs["sample_key"], audit["sample_key"]):
            raise AssertionError("input/label sample-key join failed")
    return inputs, audit


def _predict_in_batches(
    model: nn.Module, inputs: ModelInputs, *, batch_size: int = 32
) -> Tensor:
    was_training = model.training
    model.eval()
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.batch_size, batch_size):
            stop = min(inputs.batch_size, start + batch_size)
            index = torch.arange(start, stop, dtype=torch.long, device=inputs.later_full_state.device)
            chunks.append(call_model(model, inputs.index_select(index)).to(torch.float64))
    if was_training:
        model.train()
    return torch.cat(chunks)


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _high_reverse_time_mask(inputs: ModelInputs) -> Tensor:
    """Return the frozen high reverse-time quartile (forward quartile zero)."""

    from mnist.d0_jacobi_rb_reverse_controller import fractional_coordinate

    coordinate = fractional_coordinate(inputs.reverse_time, inputs.phase)
    return coordinate.forward_outer_quartile == 0


def _training_task(
    run_dir: Path,
    *,
    task: str,
    seed: int,
    baseline: Any,
    train_inputs: ModelInputs,
    train_target: Tensor,
    validation_inputs: ModelInputs,
    validation_target: Tensor,
    validation_path_ids: np.ndarray,
    target_scale: float,
    maximum_updates: int,
    physical: bool,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent import (
        BoundaryTangentPredictor,
        direct_raw_target_mse,
    )
    device = train_inputs.later_full_state.device
    enable_deterministic_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = BoundaryTangentPredictor(baseline, zero_residual=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    fingerprint = config_fingerprint(
        {
            "schema": RUN_SCHEMA + "-training-fingerprint",
            "task": task,
            "seed": seed,
            "baseline": baseline.fingerprint,
            "target_scale": float(target_scale),
            "maximum_updates": int(maximum_updates),
            "scientific_config": _load_json(run_dir / "scientific_config.json")["semantic_sha256"],
            "train_cache_index_sha256": file_fingerprint(run_dir / "cache" / "train_index.json"),
            "validation_cache_index_sha256": file_fingerprint(
                run_dir / "cache" / "validation_index.json"
            ),
        }
    )
    progress_path = run_dir / "checkpoints" / task / f"seed-{seed}-progress.pt"
    completed = 0
    candidates: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    finite = True
    if progress_path.is_file():
        snapshot = torch.load(progress_path, map_location=device, weights_only=False)
        if snapshot.get("fingerprint") != fingerprint:
            raise ArtifactCompatibilityError(f"{task} training fingerprint changed")
        model.load_state_dict(snapshot["model_state_dict"], strict=True)
        optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        completed = int(snapshot["completed_update"])
        candidates = [dict(value) for value in snapshot["candidates"]]
        history = [dict(value) for value in snapshot["history"]]
        finite = bool(snapshot["finite"])
        torch.set_rng_state(snapshot["torch_rng_state"].cpu())
        if torch.cuda.is_available() and snapshot.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(list(snapshot["cuda_rng_states"]))

    baseline_validation = model.baseline_prediction(validation_inputs).detach()
    zero_validation = torch.zeros_like(baseline_validation)
    # ``internal_reverse_time`` decreases as the forward outer-step index
    # increases.  The high reverse-time quartile is therefore forward
    # quartile zero, matching the already-frozen coarse-residual selection
    # semantics.  ``coordinate.reverse_start`` denotes the opposite end of
    # this clock and must not be used for this gate.
    high_reverse_time = _high_reverse_time_mask(validation_inputs)

    def raw_mse(prediction: Tensor, mask: Tensor | None = None) -> float:
        active_prediction = prediction if mask is None else prediction[mask]
        active_target = validation_target if mask is None else validation_target[mask]
        return float(torch.mean((active_prediction - active_target).square()).detach().cpu())

    baseline_overall = raw_mse(baseline_validation)
    baseline_high_reverse_time = raw_mse(
        baseline_validation, high_reverse_time
    )
    zero_overall = raw_mse(zero_validation)
    zero_high_reverse_time = raw_mse(zero_validation, high_reverse_time)

    def validate(update: int) -> dict[str, Any]:
        prediction = _predict_in_batches(model, validation_inputs)
        overall = raw_mse(prediction)
        high = raw_mse(prediction, high_reverse_time)
        state = _clone_state_dict(model)
        state_hash = state_dict_sha256(state)
        path = run_dir / "checkpoints" / task / f"seed-{seed}" / f"update-{update:04d}.pt"
        artifact = _atomic_torch(
            path,
            {
                "schema": RUN_SCHEMA + "-candidate",
                "fingerprint": fingerprint,
                "task": task,
                "seed": seed,
                "update": update,
                "state_dict": state,
                "state_sha256": state_hash,
            },
        )
        record = {
            "task": task,
            "seed": seed,
            "update": update,
            "training_fingerprint": fingerprint,
            "validation_mse": overall,
            "validation_high_reverse_time_mse": high,
            "baseline_validation_mse": baseline_overall,
            "baseline_high_reverse_time_mse": baseline_high_reverse_time,
            "zero_validation_mse": zero_overall,
            "zero_high_reverse_time_mse": zero_high_reverse_time,
            "combined_vs_baseline": baseline_overall - overall,
            "combined_vs_baseline_high_reverse_time": (
                baseline_high_reverse_time - high
            ),
            "combined_vs_zero": zero_overall - overall,
            "combined_vs_zero_high_reverse_time": zero_high_reverse_time - high,
            "finite": int(math.isfinite(overall) and math.isfinite(high)),
            "eligible_nonzero": int(
                update > 0
                and overall < baseline_overall
                and high < baseline_high_reverse_time
            ),
            "state_sha256": state_hash,
            "checkpoint_path": path.relative_to(run_dir).as_posix(),
            "checkpoint_file_sha256": artifact["sha256"],
        }
        if update == 0:
            record["update_zero_exact_baseline_error"] = float(
                torch.max(torch.abs(prediction - baseline_validation)).item()
            )
        candidates.append(record)
        return record

    def checkpoint(update: int) -> None:
        _atomic_torch(
            progress_path,
            {
                "schema": RUN_SCHEMA + "-training-progress",
                "fingerprint": fingerprint,
                "completed_update": update,
                "model_state_dict": _clone_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "candidates": candidates,
                "history": history,
                "finite": int(finite),
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
            },
        )

    if not candidates:
        zero = validate(0)
        if float(zero.get("update_zero_exact_baseline_error", math.inf)) != 0.0:
            raise BoundaryTangentCLIError(
                "update zero is not the exact tangent baseline",
                failure_domain="training_contract",
                failure_code="boundary_tangent_update_zero_invalid",
            )
        checkpoint(0)
    interval = min(int(TRAINING["validation_interval"]), max(1, maximum_updates))
    if finite:
        model.train()
        for update in range(completed + 1, maximum_updates + 1):
            batch_np = deterministic_batch_indices(
                train_inputs.batch_size, TRAINING["batch_size"], update - 1, seed
            )
            batch = torch.as_tensor(batch_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = call_model(model, train_inputs.index_select(batch))
            loss, raw = direct_raw_target_mse(
                prediction, train_target.index_select(0, batch), target_scale
            )
            if not bool(torch.isfinite(loss)):
                finite = False
                checkpoint(update - 1)
                break
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), TRAINING["gradient_norm_clip"]
            )
            if not math.isfinite(float(gradient)):
                finite = False
                checkpoint(update - 1)
                break
            optimizer.step()
            if update % interval == 0 or update == maximum_updates:
                candidate = validate(update)
                history.append(
                    {
                        "update": update,
                        "train_raw_mse": float(raw.detach().cpu()),
                        "scaled_loss": float(loss.detach().cpu()),
                        "preclip_gradient_norm": float(gradient),
                        **candidate,
                    }
                )
                checkpoint(update)
                model.train()
                print(
                    f"{task} seed={seed} update={update}/{maximum_updates} "
                    f"validation_mse={candidate['validation_mse']:.8g}",
                    flush=True,
                )
    eligible = [
        value
        for value in candidates
        if int(value["finite"]) == 1
        and (not physical or int(value["eligible_nonzero"]) == 1)
    ]
    if physical and not eligible:
        eligible = [value for value in candidates if int(value["update"]) == 0]
    if not eligible:
        raise BoundaryTangentCLIError(
            f"{task} produced no finite candidate",
            failure_domain="training",
            failure_code="boundary_tangent_no_finite_candidate",
        )
    selected = min(
        eligible,
        key=lambda value: (float(value["validation_mse"]), int(value["update"]), int(value["seed"])),
    )
    report = {
        "schema": RUN_SCHEMA + "-training-task",
        "schema_version": 1,
        "task": task,
        "seed": seed,
        "finite": int(finite),
        "complete": int(finite and max(int(value["update"]) for value in candidates) == maximum_updates),
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "training_fingerprint": fingerprint,
        "physical_training_performed": int(physical),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "checkpoints" / task / f"seed-{seed}-task.json", report)
    atomic_write_csv(run_dir / "checkpoints" / task / f"seed-{seed}-history.csv", history)
    return report


def _load_candidate_model(
    run_dir: Path,
    candidate: Mapping[str, Any],
    baseline: Any,
    device: torch.device,
) -> nn.Module:
    """Load one hash-bound candidate without opening any other checkpoint."""

    from mnist.d0_jacobi_rb_boundary_tangent import BoundaryTangentPredictor

    task = str(candidate.get("task", ""))
    seed = int(candidate.get("seed", candidate.get("selected_seed", -1)))
    update = int(candidate.get("update", candidate.get("selected_update", -1)))
    fingerprint = str(candidate.get("training_fingerprint", ""))
    expected_state_sha = str(
        candidate.get("state_sha256", candidate.get("selected_state_sha256", ""))
    )
    if (
        task not in {"synthetic-teacher", "exact-baseline-null", "physical"}
        or seed < 0
        or update < 0
        or len(fingerprint) != 64
        or len(expected_state_sha) != 64
    ):
        raise ArtifactCompatibilityError("selected checkpoint identity changed")
    path = run_dir / str(candidate["checkpoint_path"])
    if candidate.get("checkpoint_file_sha256") != file_fingerprint(path):
        raise ArtifactCompatibilityError("selected checkpoint file changed")
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("state_dict")
    if (
        payload.get("schema") != RUN_SCHEMA + "-candidate"
        or payload.get("fingerprint") != fingerprint
        or payload.get("task") != task
        or int(payload.get("seed", -1)) != seed
        or int(payload.get("update", -1)) != update
        or not isinstance(state, Mapping)
        or state_dict_sha256(state) != expected_state_sha
    ):
        raise ArtifactCompatibilityError("selected checkpoint state changed")
    frozen_q = state.get("_q_values")
    if (
        not isinstance(frozen_q, Tensor)
        or frozen_q.dtype != torch.float64
        or not np.array_equal(
            frozen_q.detach().cpu().numpy(), np.asarray(baseline.q_values)
        )
    ):
        raise ArtifactCompatibilityError(
            "selected checkpoint is not bound to the sealed tangent baseline"
        )
    model = BoundaryTangentPredictor(baseline, zero_residual=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _selected_control_metrics(
    run_dir: Path,
    report: Mapping[str, Any],
    baseline: Any,
    validation_inputs: ModelInputs,
    validation_target: Tensor,
    validation_path_ids: np.ndarray,
    *,
    synthetic: bool,
) -> dict[str, Any]:
    candidate = dict(report["selected"])
    device = validation_inputs.later_full_state.device
    model = _load_candidate_model(run_dir, candidate, baseline, device)
    prediction = _predict_in_batches(model, validation_inputs)
    squared = torch.mean((prediction - validation_target).square(), dim=1).cpu().numpy()
    zero_squared = torch.mean(validation_target.square(), dim=1).cpu().numpy()
    paths = np.asarray(validation_path_ids, dtype=np.int64)
    path_rows: list[dict[str, Any]] = []
    every_path = True
    for path_id in sorted(np.unique(paths).tolist()):
        active = paths == path_id
        model_mse = float(np.mean(squared[active], dtype=np.float64))
        zero_mse = float(np.mean(zero_squared[active], dtype=np.float64))
        beats = model_mse < zero_mse
        every_path &= beats
        path_rows.append(
            {
                "path_id": int(path_id),
                "model_mse": model_mse,
                "zero_mse": zero_mse,
                "beats_zero": int(beats),
            }
        )
    overall = float(np.mean(squared, dtype=np.float64))
    zero = float(np.mean(zero_squared, dtype=np.float64))
    relative = overall / zero if zero > 0.0 else math.inf
    return {
        "selected_update": int(candidate["update"]),
        "selected_state_sha256": str(candidate["state_sha256"]),
        "validation_mse": overall,
        "zero_validation_mse": zero,
        "relative_validation_mse": relative,
        "every_validation_path_beats_zero": int(every_path),
        "passed": int(
            math.isfinite(relative)
            and (relative <= 0.01 if synthetic else int(candidate["update"]) == 0)
            and (every_path if synthetic else True)
        ),
        "path_metrics": path_rows,
    }


def _verify_training_selection(run_dir: Path) -> dict[str, Any]:
    selection = _load_json(run_dir / "checkpoint_selection.json")
    body = dict(selection)
    semantic = body.pop("semantic_sha256", None)
    seed = int(selection.get("selected_seed", -1))
    update = int(selection.get("selected_update", -1))
    expected_checkpoint = (
        f"checkpoints/physical/seed-{seed}/update-{update:04d}.pt"
    )
    if (
        semantic != config_fingerprint(body)
        or selection.get("schema") != RUN_SCHEMA + "-checkpoint-selection"
        or int(selection.get("schema_version", -1)) != 1
        or selection.get("selection_role") != "validation_only"
        or selection.get("task") != "physical"
        or seed not in MODEL_SEEDS
        or update < 0
        or selection.get("checkpoint_path") != expected_checkpoint
        or selection.get("baseline_path") != "tangent_baseline.npz"
        or selection.get("candidate_ranking")
        != "lowest_validation_mse_then_earliest_update_then_lower_seed"
        or int(selection.get("confirmation_paths_created", -1)) != 0
        or len(str(selection.get("training_fingerprint", ""))) != 64
        or len(str(selection.get("selected_state_sha256", ""))) != 64
    ):
        raise ArtifactCompatibilityError("sealed checkpoint selection changed")
    baseline_path = run_dir / str(selection["baseline_path"])
    checkpoint_path = run_dir / str(selection["checkpoint_path"])
    if selection.get("baseline_file_sha256") != file_fingerprint(baseline_path):
        raise ArtifactCompatibilityError("sealed tangent baseline changed")
    if selection.get("checkpoint_file_sha256") != file_fingerprint(checkpoint_path):
        raise ArtifactCompatibilityError("sealed physical checkpoint changed")
    from mnist.d0_jacobi_rb_boundary_tangent import load_tangent_baseline

    baseline = load_tangent_baseline(
        baseline_path, expected_sha256=selection["baseline_file_sha256"]
    )
    if baseline.fingerprint != selection.get("baseline_semantic_sha256"):
        raise ArtifactCompatibilityError("sealed tangent baseline semantic changed")
    # Verify payload identity and the persistent q-grid binding without moving
    # the model to the production device.
    _load_candidate_model(run_dir, selection, baseline, torch.device("cpu"))
    return selection


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent import (
        derive_tangent_baseline,
        save_tangent_baseline,
    )

    if not _passed(_load_json(run_dir / "cache_gate.json")):
        raise ArtifactCompatibilityError("train stage requires a passing cache gate")
    gate_path = run_dir / "train_gate.json"
    if gate_path.is_file():
        selection = _verify_training_selection(run_dir)
        gate = _load_json(gate_path)
        if int(gate.get("passed", 0)) == 1 and int(selection.get("selected_update", 0)) <= 0:
            raise ArtifactCompatibilityError("passing train gate selected update zero")
        return gate

    device = torch.device(args.device)
    maximum_updates = (
        int(args.test_maximum_updates) if args.test_only else int(TRAINING["maximum_updates"])
    )
    paths = _effective_paths(args)

    # Controls see only model inputs.  The physical label archives remain
    # unopened until both controls have passed and that event is committed.
    train_arrays, _ = _load_role_cache_arrays(run_dir, "train", open_labels=False)
    validation_arrays, _ = _load_role_cache_arrays(run_dir, "validation", open_labels=False)
    train_inputs = _model_inputs_from_arrays(train_arrays, device)
    validation_inputs = _model_inputs_from_arrays(validation_arrays, device)
    train_path_rows = np.asarray(train_arrays["path_id"], dtype=np.int64)
    validation_path_rows = np.asarray(validation_arrays["path_id"], dtype=np.int64)
    zero = _zero_baseline(paths["train"])

    from mnist.d0_jacobi_rb_boundary_tangent import synthetic_tangent_target

    synthetic_train = synthetic_tangent_target(train_inputs).detach().to(torch.float64)
    synthetic_validation = synthetic_tangent_target(validation_inputs).detach().to(torch.float64)
    synthetic_scale = float(torch.sqrt(torch.mean(synthetic_train.square())).cpu())
    if not math.isfinite(synthetic_scale) or synthetic_scale <= 0.0:
        raise BoundaryTangentCLIError(
            "synthetic target scale is invalid",
            failure_domain="optimization_control",
            failure_code="boundary_tangent_synthetic_scale_invalid",
        )
    synthetic_report = _training_task(
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
    synthetic_metrics = _selected_control_metrics(
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
        run_dir / "synthetic_teacher_per_path.csv", synthetic_metrics["path_metrics"]
    )

    from mnist.d0_jacobi_rb_boundary_tangent import BoundaryTangentPredictor

    null_baseline = _analytic_null_baseline(paths["train"])
    null_model = BoundaryTangentPredictor(null_baseline, zero_residual=True).to(device)
    with torch.no_grad():
        null_train = null_model.baseline_prediction(train_inputs).detach()
        null_validation = null_model.baseline_prediction(validation_inputs).detach()
    null_scale = float(torch.sqrt(torch.mean(null_train.square())).cpu())
    null_report = _training_task(
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
    null_passed = int(null_report["selected"]["update"]) == 0
    null_metrics = {
        "schema": RUN_SCHEMA + "-exact-baseline-null",
        "schema_version": 1,
        "selected_update": int(null_report["selected"]["update"]),
        "selected_validation_mse": float(null_report["selected"]["validation_mse"]),
        "passed": int(null_passed),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "exact_baseline_null_control.json", null_metrics)

    controls_passed = bool(synthetic_metrics["passed"] and null_metrics["passed"])
    if not controls_passed:
        gate = _gate(
            "train",
            {
                "synthetic_teacher": bool(synthetic_metrics["passed"]),
                "exact_baseline_null": bool(null_metrics["passed"]),
                "physical_labels_unopened": not (run_dir / "physical_label_open.json").exists(),
            },
            boundary_tangent_baseline_valid=0,
            optimization_pipeline_valid=0,
            boundary_tangent_baseline_only=0,
            selected_nonzero_seed_count=0,
        )
        atomic_write_json(gate_path, gate)
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
            **NO_WORK,
        },
    )
    atomic_write_json(
        run_dir / "physical_training_started.json",
        {"started_at": _now(), "physical_training_performed": 1, **NO_WORK},
    )

    # Only now are the separate raw-label artifacts opened.
    train_arrays_open, train_audit = _load_role_cache_arrays(run_dir, "train", open_labels=True)
    validation_arrays_open, validation_audit = _load_role_cache_arrays(
        run_dir, "validation", open_labels=True
    )
    if train_audit is None or validation_audit is None:
        raise AssertionError("physical labels were not opened")
    if not np.array_equal(train_arrays_open["sample_key"], train_arrays["sample_key"]):
        raise ArtifactCompatibilityError("train inputs changed when labels opened")
    if not np.array_equal(validation_arrays_open["sample_key"], validation_arrays["sample_key"]):
        raise ArtifactCompatibilityError("validation inputs changed when labels opened")
    train_target = torch.as_tensor(
        np.array(train_audit["denoising_target"], copy=True, order="C"),
        dtype=torch.float64,
        device=device,
    )
    validation_target = torch.as_tensor(
        np.array(validation_audit["denoising_target"], copy=True, order="C"),
        dtype=torch.float64,
        device=device,
    )
    target_scale = float(torch.sqrt(torch.mean(train_target.square())).cpu())
    if not math.isfinite(target_scale) or target_scale <= 0.0:
        raise BoundaryTangentCLIError(
            "training-only raw target scale is invalid",
            failure_domain="baseline",
            failure_code="boundary_tangent_target_scale_invalid",
        )
    baseline = derive_tangent_baseline(train_inputs, train_target, train_path_rows)
    baseline_artifact = save_tangent_baseline(run_dir / "tangent_baseline.npz", baseline)
    baseline_record = {
        "schema": RUN_SCHEMA + "-baseline",
        "schema_version": 1,
        "baseline": baseline.to_record(),
        "file": baseline_artifact,
        "target_scale": target_scale,
        "training_path_ids": sorted(np.unique(train_path_rows).tolist()),
        "validation_path_ids_used": 0,
        "confirmation_path_ids_used": 0,
        "quotient_target_formed": 0,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "tangent_baseline.json", baseline_record)

    reports = [
        _training_task(
            run_dir,
            task="physical",
            seed=seed,
            baseline=baseline,
            train_inputs=train_inputs,
            train_target=train_target,
            validation_inputs=validation_inputs,
            validation_target=validation_target,
            validation_path_ids=validation_path_rows,
            target_scale=target_scale,
            maximum_updates=maximum_updates,
            physical=True,
        )
        for seed in MODEL_SEEDS
    ]
    nonzero_reports = [
        report
        for report in reports
        if int(report.get("complete", 0)) == 1
        and int(report.get("finite", 0)) == 1
        and int(report["selected"]["update"]) > 0
        and int(report["selected"].get("eligible_nonzero", 0)) == 1
    ]
    selected_report = min(
        nonzero_reports if nonzero_reports else reports,
        key=lambda report: (
            float(report["selected"]["validation_mse"]),
            int(report["selected"]["update"]),
            int(report["seed"]),
        ),
    )
    selected = dict(selected_report["selected"])
    selection = {
        "schema": RUN_SCHEMA + "-checkpoint-selection",
        "schema_version": 1,
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
        **NO_WORK,
    }
    selection["semantic_sha256"] = config_fingerprint(selection)
    atomic_write_json(run_dir / "checkpoint_selection.json", selection)
    atomic_write_csv(
        run_dir / "physical_seed_metrics.csv",
        [
            {
                "seed": int(report["seed"]),
                "complete": int(report["complete"]),
                "finite": int(report["finite"]),
                **{f"selected_{key}": value for key, value in report["selected"].items()
                   if isinstance(value, (str, int, float))},
            }
            for report in reports
        ],
    )
    baseline_only = int(selection["selected_update"]) == 0
    gate = _gate(
        "train",
        {
            "baseline_valid": bool(np.isfinite(baseline.q_values).all()),
            "baseline_training_only": set(baseline.training_path_ids.tolist()) == set(paths["train"]),
            "synthetic_teacher": bool(synthetic_metrics["passed"]),
            "exact_baseline_null": bool(null_metrics["passed"]),
            "all_physical_tasks_complete_finite": all(
                int(report.get("complete", 0)) == 1 and int(report.get("finite", 0)) == 1
                for report in reports
            ),
            "selected_nonzero": int(selection["selected_update"]) > 0,
            "validation_only_selection": selection["selection_role"] == "validation_only",
            "confirmation_absent": not (run_dir / "confirmation_seal.json").exists(),
        },
        boundary_tangent_baseline_valid=1,
        optimization_pipeline_valid=int(
            bool(synthetic_metrics["passed"])
            and bool(null_metrics["passed"])
            and all(int(report.get("complete", 0)) == 1 for report in reports)
        ),
        boundary_tangent_baseline_only=int(baseline_only),
        selected_nonzero_seed_count=len(nonzero_reports),
        checkpoint_selection=selection,
        synthetic_control=synthetic_metrics,
        null_control=null_metrics,
        physical_training_performed=True,
    )
    atomic_write_json(gate_path, gate)
    return gate


def _confirmation_shard_paths(
    run_dir: Path, *, cohort_index: int, start_step: int
) -> tuple[Path, Path, Path, Path]:
    directory = run_dir / "confirmation" / "shards" / f"cohort-{cohort_index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step-{start_step:03d}"
    return (
        directory / f"{stem}-state.npz",
        directory / f"{stem}-risk.npz",
        directory / f"{stem}-control-audit.npz",
        directory / f"{stem}.json",
    )


def _confirmation_audit_mass_errors(
    one_phase_earlier: np.ndarray,
    one_phase_later: np.ndarray,
    eight_phase_earlier: np.ndarray,
    eight_phase_later: np.ndarray,
) -> tuple[float, float]:
    """Return exact pair/simplex diagnostics for sealed control-audit states."""

    from mnist.d0_jacobi_rb_learnability import matching_indices

    tails_all, heads_all = matching_indices(device="cpu")
    pair_error = 0.0
    for occurrence in range(PHASE_COUNT):
        color = int(PHASE_MATCHINGS[occurrence])
        tails = tails_all[color].cpu().numpy()
        heads = heads_all[color].cpu().numpy()
        before_pair = (
            one_phase_earlier[occurrence, :, tails]
            + one_phase_earlier[occurrence, :, heads]
        )
        after_pair = (
            one_phase_later[occurrence, :, tails]
            + one_phase_later[occurrence, :, heads]
        )
        pair_error = max(
            pair_error,
            float(np.max(np.abs(after_pair - before_pair), initial=0.0)),
        )
    simplex_error = max(
        float(
            np.max(
                np.abs(one_phase_earlier.sum(axis=-1) - 1.0), initial=0.0
            )
        ),
        float(
            np.max(np.abs(one_phase_later.sum(axis=-1) - 1.0), initial=0.0)
        ),
        float(
            np.max(
                np.abs(eight_phase_earlier.sum(axis=-1) - 1.0), initial=0.0
            )
        ),
        float(
            np.max(
                np.abs(eight_phase_later.sum(axis=-1) - 1.0), initial=0.0
            )
        ),
    )
    return pair_error, simplex_error


def _confirmation_branch_evidence(
    result: Any,
    *,
    selected_step: int,
    path_ids: Sequence[int],
    args: argparse.Namespace,
    model: nn.Module,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate a selected branch in memory and retain only risk/audit evidence."""

    from mnist.d0_jacobi_rb_boundary_tangent_cache import midpoint_sample_key
    from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time

    inputs_raw, labels_raw, diagnostics = _branch_arrays(
        result, selected_step, path_ids, args, "confirmation"
    )
    paths = np.asarray(path_ids, dtype=np.int64)
    later = np.asarray(inputs_raw["later_full_state"], dtype=np.float32)
    target = np.asarray(labels_raw["denoising_target"], dtype=np.float64)
    states = np.transpose(later, (2, 0, 1, 3)).reshape(-1, STATE_SIZE)
    targets = np.transpose(target, (2, 0, 1, 3)).reshape(-1, EDGES_PER_PHASE)
    row_paths = np.repeat(paths, PHASE_COUNT * MIDPOINT_COUNT)
    row_steps = np.full(row_paths.size, selected_step, dtype=np.int16)
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
    reverse_time = np.asarray(
        [
            internal_reverse_time(int(selected_step), int(phase), float(fraction))
            for phase, fraction in zip(row_phases, fractions, strict=True)
        ],
        dtype=np.float64,
    )
    input_arrays = {
        "later_full_state": np.ascontiguousarray(states),
        "reverse_time": reverse_time,
        "phase": row_phases,
        "color": np.asarray(
            [PHASE_MATCHINGS[int(value)] for value in row_phases], dtype=np.int8
        ),
        "duration": np.asarray(
            [PHASE_DURATIONS[int(value)] for value in row_phases], dtype=np.float64
        ),
        "label": np.full(row_paths.size, 3, dtype=np.int64),
    }
    model_inputs = _model_inputs_from_arrays(
        input_arrays, result.final_states.device
    )
    combined = _predict_in_batches(model, model_inputs).cpu().numpy()
    with torch.no_grad():
        baseline = model.baseline_prediction(model_inputs).cpu().numpy()
    zero_improvement = np.mean(
        targets * targets - (targets - combined) ** 2,
        axis=1,
        dtype=np.float64,
    )
    baseline_improvement = np.mean(
        (targets - baseline) ** 2 - (targets - combined) ** 2,
        axis=1,
        dtype=np.float64,
    )
    sample_keys = np.asarray(
        [
            midpoint_sample_key(int(path), selected_step, int(phase), int(midpoint))
            for path, phase, midpoint in zip(
                row_paths, row_phases, row_midpoints, strict=True
            )
        ],
        dtype=np.int64,
    )
    risk = {
        "sample_keys": sample_keys,
        "path_ids": row_paths,
        "outer_steps": row_steps,
        "phases": row_phases,
        "midpoint_indices": row_midpoints,
        "combined_vs_zero": np.ascontiguousarray(zero_improvement),
        "combined_vs_baseline": np.ascontiguousarray(baseline_improvement),
    }
    capture = result.capture_payload
    post_trace = np.asarray(capture.post_phase_states, dtype=np.float64)
    local = selected_step - int(capture.start_step)
    pre = np.stack(
        [
            post_trace[local * PHASE_COUNT + phase - 1]
            if phase
            else post_trace[local * PHASE_COUNT - 1]
            for phase in range(PHASE_COUNT)
        ]
    )
    post = np.stack(
        [post_trace[local * PHASE_COUNT + phase] for phase in range(PHASE_COUNT)]
    )
    audit = {
        "path_ids": paths,
        "outer_step": np.asarray([selected_step], dtype=np.int16),
        "one_phase_earlier_states": np.ascontiguousarray(pre),
        "one_phase_later_states": np.ascontiguousarray(post),
        "eight_phase_earlier_states": np.ascontiguousarray(
            post_trace[(local - 1) * PHASE_COUNT + 5]
        ),
        "eight_phase_later_states": np.ascontiguousarray(post[-1]),
    }
    audit_pair_error, audit_simplex_error = _confirmation_audit_mass_errors(
        audit["one_phase_earlier_states"],
        audit["one_phase_later_states"],
        audit["eight_phase_earlier_states"],
        audit["eight_phase_later_states"],
    )
    diagnostics = {
        **dict(diagnostics),
        "control_audit_maximum_pair_mass_error": audit_pair_error,
        "control_audit_maximum_simplex_mass_error": audit_simplex_error,
    }
    if (
        not np.isfinite(zero_improvement).all()
        or not np.isfinite(baseline_improvement).all()
        or np.unique(sample_keys).size != sample_keys.size
        or audit_pair_error > 2.0e-12
        or audit_simplex_error > 2.0e-12
    ):
        raise BoundaryTangentCLIError(
            "confirmation risk evidence is invalid",
            failure_domain="confirmation",
            failure_code="boundary_tangent_confirmation_risk_invalid",
        )
    return risk, audit, diagnostics


def _valid_confirmation_shard(
    run_dir: Path,
    *,
    cohort_index: int,
    path_ids: Sequence[int],
    start_step: int,
    current: np.ndarray,
    selected_step: int | None,
    selection_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    state_path, risk_path, audit_path, metadata_path = _confirmation_shard_paths(
        run_dir, cohort_index=cohort_index, start_step=start_step
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
            "cohort_index": cohort_index,
            "path_ids": list(path_ids),
            "start_step": start_step,
            "selected_step": selected_step,
            "input_state_sha256": _array_sha(current),
            "selection_sha256": selection_sha256,
            "scientific_config_sha256": _load_json(run_dir / "scientific_config.json")[
                "semantic_sha256"
            ],
            "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")[
                "semantic_sha256"
            ],
        }
        if semantic != config_fingerprint(body) or any(
            record.get(name) != value for name, value in expected.items()
        ):
            return None
        if record.get("state_file_sha256") != file_fingerprint(state_path):
            return None
        final = _load_npz(state_path).get("final_states")
        if (
            final is None
            or final.dtype != np.float64
            or final.shape != (len(path_ids), STATE_SIZE)
            or not np.isfinite(final).all()
            or np.any(final < 0.0)
            or record.get("final_state_sha256") != _array_sha(final)
        ):
            return None
        if selected_step is not None:
            if (
                record.get("risk_file_sha256") != file_fingerprint(risk_path)
            ):
                return None
            risk = _load_npz(risk_path)
            audit = _load_npz(audit_path) if selected_step in CONTROL_ANCHORS else None
            if selected_step in CONTROL_ANCHORS and (
                record.get("control_audit_file_sha256") != file_fingerprint(audit_path)
            ):
                return None
            if selected_step not in CONTROL_ANCHORS and record.get("control_audit_file_sha256") is not None:
                return None
            rows = len(path_ids) * PHASE_COUNT * MIDPOINT_COUNT
            if (
                set(risk)
                != {
                    "sample_keys",
                    "path_ids",
                    "outer_steps",
                    "phases",
                    "midpoint_indices",
                    "combined_vs_zero",
                    "combined_vs_baseline",
                }
                or any(np.asarray(risk[name]).shape != (rows,) for name in risk)
                or np.asarray(risk["combined_vs_zero"]).dtype != np.float64
                or np.asarray(risk["combined_vs_baseline"]).dtype != np.float64
                or not np.isfinite(risk["combined_vs_zero"]).all()
                or not np.isfinite(risk["combined_vs_baseline"]).all()
                or (audit is not None and set(audit) != {
                    "path_ids",
                    "outer_step",
                    "one_phase_earlier_states",
                    "one_phase_later_states",
                    "eight_phase_earlier_states",
                    "eight_phase_later_states",
                })
            ):
                return None
            from mnist.d0_jacobi_rb_boundary_tangent_cache import midpoint_sample_key

            expected_identity = np.asarray(
                [
                    (path, selected_step, phase, midpoint)
                    for path in path_ids
                    for phase in range(PHASE_COUNT)
                    for midpoint in range(MIDPOINT_COUNT)
                ],
                dtype=np.int64,
            )
            if (
                not np.array_equal(risk["path_ids"], expected_identity[:, 0])
                or not np.array_equal(risk["outer_steps"], expected_identity[:, 1])
                or not np.array_equal(risk["phases"], expected_identity[:, 2])
                or not np.array_equal(risk["midpoint_indices"], expected_identity[:, 3])
                or not np.array_equal(
                    risk["sample_keys"],
                    np.asarray(
                        [midpoint_sample_key(*row) for row in expected_identity],
                        dtype=np.int64,
                    ),
                )
            ):
                return None
            branch = record.get("branch_diagnostics")
            if (
                not isinstance(branch, Mapping)
                or int(branch.get("transition_count", -1))
                != len(path_ids) * PHASE_COUNT * MIDPOINT_COUNT * EDGES_PER_PHASE
                or int(branch.get("certified_count", -1))
                != int(branch.get("transition_count", -2))
                or any(
                    int(branch.get("forbidden_counts", {}).get(name, -1)) != 0
                    for name in FORBIDDEN_COUNTS
                    if name != "uncertified_count"
                )
            ):
                return None
            if audit is not None:
                one_earlier = np.asarray(audit["one_phase_earlier_states"])
                one_later = np.asarray(audit["one_phase_later_states"])
                eight_earlier = np.asarray(audit["eight_phase_earlier_states"])
                eight_later = np.asarray(audit["eight_phase_later_states"])
                state_arrays = (one_earlier, one_later, eight_earlier, eight_later)
                audit_paths = np.asarray(audit["path_ids"])
                audit_step = np.asarray(audit["outer_step"])
                if (
                    audit_paths.dtype != np.dtype(np.int64)
                    or audit_step.dtype != np.dtype(np.int16)
                    or not np.array_equal(audit_paths, np.asarray(path_ids, dtype=np.int64))
                    or not np.array_equal(audit_step, np.asarray([selected_step], dtype=np.int16))
                    or one_earlier.shape != (PHASE_COUNT, len(path_ids), STATE_SIZE)
                    or one_later.shape != one_earlier.shape
                    or eight_earlier.shape != (len(path_ids), STATE_SIZE)
                    or eight_later.shape != eight_earlier.shape
                    or any(value.dtype != np.float64 for value in state_arrays)
                    or any(not np.isfinite(value).all() or np.any(value < 0.0) for value in state_arrays)
                    or any(
                        float(np.max(np.abs(value.sum(axis=-1) - 1.0), initial=0.0)) > 2.0e-12
                        for value in state_arrays
                    )
                ):
                    return None
                audit_pair_error, audit_simplex_error = (
                    _confirmation_audit_mass_errors(
                        one_earlier, one_later, eight_earlier, eight_later
                    )
                )
                if (
                    audit_pair_error > 2.0e-12
                    or audit_simplex_error > 2.0e-12
                    or float(
                        branch.get(
                            "control_audit_maximum_pair_mass_error", math.inf
                        )
                    )
                    != audit_pair_error
                    or float(
                        branch.get(
                            "control_audit_maximum_simplex_mass_error", math.inf
                        )
                    )
                    != audit_simplex_error
                ):
                    return None
        return np.ascontiguousarray(final), record
    except (ArtifactCompatibilityError, OSError, ValueError, TypeError, KeyError):
        return None


def _persist_confirmation_shard(
    run_dir: Path,
    *,
    cohort_index: int,
    path_ids: Sequence[int],
    start_step: int,
    selected_step: int | None,
    input_state_sha256: str,
    selection_sha256: str,
    result: Any,
    risk: Mapping[str, np.ndarray] | None,
    audit: Mapping[str, np.ndarray] | None,
    branch_diagnostics: Mapping[str, Any],
    started_at: float,
    device: torch.device,
) -> dict[str, Any]:
    state_path, risk_path, audit_path, metadata_path = _confirmation_shard_paths(
        run_dir, cohort_index=cohort_index, start_step=start_step
    )
    final = np.ascontiguousarray(result.committed_final_states, dtype=np.float64)
    state_artifact = _atomic_npz(state_path, {"final_states": final})
    risk_artifact = _atomic_npz(risk_path, risk) if risk is not None else None
    audit_artifact = _atomic_npz(audit_path, audit) if audit is not None else None
    if (risk_artifact is None) != (selected_step is None):
        raise AssertionError("confirmation risk must exist exactly at selected steps")
    if (audit_artifact is None) != (selected_step not in CONTROL_ANCHORS):
        raise AssertionError("control audit must exist exactly at controller anchors")
    record = {
        "schema": RUN_SCHEMA + "-confirmation-shard",
        "schema_version": 1,
        "cohort_index": cohort_index,
        "path_ids": list(path_ids),
        "start_step": start_step,
        "selected_step": selected_step,
        "input_state_sha256": input_state_sha256,
        "selection_sha256": selection_sha256,
        "scientific_config_sha256": _load_json(run_dir / "scientific_config.json")[
            "semantic_sha256"
        ],
        "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")[
            "semantic_sha256"
        ],
        "state_file_sha256": state_artifact["sha256"],
        "state_file_size": state_artifact["size"],
        "final_state_sha256": _array_sha(final),
        "risk_file_sha256": None if risk_artifact is None else risk_artifact["sha256"],
        "risk_file_size": None if risk_artifact is None else risk_artifact["size"],
        "control_audit_file_sha256": None if audit_artifact is None else audit_artifact["sha256"],
        "control_audit_file_size": None if audit_artifact is None else audit_artifact["size"],
        "scheduler_record": result.to_record(),
        "branch_diagnostics": dict(branch_diagnostics),
        "complete_pipeline_elapsed_seconds": float(time.perf_counter() - started_at),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 1,
        "raw_branch_inputs_persisted": 0,
        "raw_branch_labels_persisted": 0,
        "committed": 1,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(metadata_path, record)
    return record


def _confirm_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent import load_tangent_baseline
    from mnist.d0_jacobi_rb_boundary_tangent_confirmation import (
        aggregate_confirmation_improvements,
    )

    if not _passed(_load_json(run_dir / "train_gate.json")):
        raise ArtifactCompatibilityError("confirm stage requires a passing train gate")
    gate_path = run_dir / "confirm_gate.json"
    if gate_path.is_file():
        _verify_training_selection(run_dir)
        _verify_confirmation_integrity(run_dir, args)
        return _load_json(gate_path)
    selection = _verify_training_selection(run_dir)
    if int(selection["selected_update"]) <= 0:
        raise ArtifactCompatibilityError("confirmation cannot open for update zero")
    paths = _effective_paths(args)["confirmation"]
    if len(paths) < 8:
        raise BoundaryTangentCLIError(
            "confirmation requires at least eight whole paths",
            failure_domain="confirmation_design",
            failure_code="boundary_tangent_confirmation_paths_invalid",
        )
    selection_sha = file_fingerprint(run_dir / "checkpoint_selection.json")
    seal_path = run_dir / "confirmation_seal.json"
    seal = {
        "schema": RUN_SCHEMA + "-confirmation-seal",
        "schema_version": 1,
        "path_ids": list(paths),
        "selection_file_sha256": selection_sha,
        "checkpoint_file_sha256": selection["checkpoint_file_sha256"],
        "checkpoint_state_sha256": selection["selected_state_sha256"],
        "baseline_file_sha256": selection["baseline_file_sha256"],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": (
            int(args.test_bootstrap_replicates) if args.test_only else 50_000
        ),
        "confirmation_created_after_selection": 1,
        "opened_once": 1,
        **NO_WORK,
    }
    seal["semantic_sha256"] = config_fingerprint(seal)
    if seal_path.is_file():
        if _load_json(seal_path) != seal:
            raise ArtifactCompatibilityError("confirmation seal changed")
    else:
        atomic_write_json(seal_path, seal)

    device = torch.device(args.device)
    baseline = load_tangent_baseline(
        run_dir / selection["baseline_path"],
        expected_sha256=selection["baseline_file_sha256"],
    )
    model = _load_candidate_model(run_dir, selection, baseline, device)
    outer_steps = int(args.test_outer_steps) if args.test_only else OUTER_STEPS
    source = _load_source_target(args.parent_coarse_residual_run_dir)
    profile = JacobiRBCudaProfile()
    records: list[dict[str, Any]] = []
    recompute = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for cohort_index, cohort in enumerate(_cohorts(paths)):
        current = np.repeat(source[None, :], len(cohort), axis=0).copy(order="C")
        recompute_tail = False
        for start_step in range(0, outer_steps, SHARD_STEPS):
            selected_step = _selected_step_in_shard(start_step, outer_steps)
            cached = None if recompute_tail else _valid_confirmation_shard(
                run_dir,
                cohort_index=cohort_index,
                path_ids=cohort,
                start_step=start_step,
                current=current,
                selected_step=selected_step,
                selection_sha256=selection_sha,
            )
            if cached is not None:
                current, record = cached
                records.append(record)
                continue
            recompute_tail = True
            recompute += 1
            state = torch.as_tensor(
                np.array(current, copy=True, order="C"),
                dtype=torch.float64,
                device=device,
            ).contiguous()
            kwargs: dict[str, Any] = {}
            if args.test_only:
                kwargs["sampler"] = _test_sampler
            started = time.perf_counter()
            result = run_exact_multipath_shard(
                state,
                path_ids=cohort,
                start_step=start_step,
                root_seed=ROOT_SEED,
                profile=profile,
                group_sizes=(len(cohort),),
                capture_training_payload=selected_step is not None,
                **kwargs,
            )
            risk = audit = None
            branch_diagnostics: Mapping[str, Any] = {
                "transition_count": 0,
                "certified_count": 0,
                "fallback_count": 0,
                "forbidden_counts": {name: 0 for name in FORBIDDEN_COUNTS if name != "uncertified_count"},
            }
            if selected_step is not None:
                risk, audit, branch_diagnostics = _confirmation_branch_evidence(
                    result,
                    selected_step=selected_step,
                    path_ids=cohort,
                    args=args,
                    model=model,
                )
                if selected_step not in CONTROL_ANCHORS:
                    audit = None
            record = _persist_confirmation_shard(
                run_dir,
                cohort_index=cohort_index,
                path_ids=cohort,
                start_step=start_step,
                selected_step=selected_step,
                input_state_sha256=_array_sha(current),
                selection_sha256=selection_sha,
                result=result,
                risk=risk,
                audit=audit,
                branch_diagnostics=branch_diagnostics,
                started_at=started,
                device=device,
            )
            current = np.ascontiguousarray(result.committed_final_states, dtype=np.float64)
            records.append(record)
            print(
                f"confirmation cohort {cohort_index + 1}/{len(_cohorts(paths))} "
                f"shard {start_step // SHARD_STEPS + 1}/{outer_steps // SHARD_STEPS} committed",
                flush=True,
            )

    risk_chunks = []
    for record in records:
        if record.get("selected_step") is None:
            continue
        risk_path = _confirmation_shard_paths(
            run_dir,
            cohort_index=int(record["cohort_index"]),
            start_step=int(record["start_step"]),
        )[1]
        if record.get("risk_file_sha256") != file_fingerprint(risk_path):
            raise ArtifactCompatibilityError("confirmation risk shard changed")
        risk_chunks.append(_load_npz(risk_path))
    joined = {
        name: np.concatenate([chunk[name] for chunk in risk_chunks])
        for name in risk_chunks[0]
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
        selected_outer_steps=tuple(
            step for step in SELECTED_OUTER_STEPS if step < outer_steps
        ),
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
    thresholds = BoundaryTangentThresholds(
        confirmation_paths=len(paths),
        controller_paths=len(paths),
        bootstrap_replicates=(
            int(args.test_bootstrap_replicates) if args.test_only else 50_000
        ),
    )
    max_t = one_sided_whole_path_max_t(
        table.path_values,
        path_ids=table.path_ids,
        confidence=thresholds.simultaneous_confidence,
        replicates=thresholds.bootstrap_replicates,
        seed=thresholds.bootstrap_seed,
    )
    atomic_write_json(run_dir / "confirmation_max_t.json", max_t)
    diagnostics = [record["scheduler_record"]["diagnostics"] for record in records]
    main_count = sum(int(item.get("transition_count", 0)) for item in diagnostics)
    main_certified = sum(int(item.get("certified_count", 0)) for item in diagnostics)
    branch_count = sum(int(record["branch_diagnostics"].get("transition_count", 0)) for record in records)
    branch_certified = sum(int(record["branch_diagnostics"].get("certified_count", 0)) for record in records)
    total_count = main_count + branch_count
    total_certified = main_certified + branch_certified
    elapsed = sum(float(record["complete_pipeline_elapsed_seconds"]) for record in records)
    persisted_confirmation = sum(
        int(record.get("state_file_size", 0))
        + int(record.get("risk_file_size") or 0)
        + int(record.get("control_audit_file_size") or 0)
        + _confirmation_shard_paths(
            run_dir,
            cohort_index=int(record["cohort_index"]),
            start_step=int(record["start_step"]),
        )[3].stat().st_size
        for record in records
    )
    prior_persisted = int(_load_json(run_dir / "cache_gate.json").get("total_persisted_cache_bytes", 0))
    total_persisted = prior_persisted + persisted_confirmation
    forbidden = {}
    for name in FORBIDDEN_COUNTS:
        main = sum(int(item.get(name, 0)) for item in diagnostics)
        branch = (
            branch_count - branch_certified
            if name == "uncertified_count"
            else sum(
                int(record["branch_diagnostics"].get("forbidden_counts", {}).get(name, 0))
                for record in records
            )
        )
        forbidden[name] = main + branch
    fallback_count = sum(int(item.get("fallback_count", 0)) for item in diagnostics) + sum(
        int(record["branch_diagnostics"].get("fallback_count", 0)) for record in records
    )
    fallback_seconds = sum(float(item.get("fallback_elapsed_seconds", 0.0)) for item in diagnostics) + sum(
        float(record["branch_diagnostics"].get("fallback_elapsed_seconds", 0.0)) for record in records
    )
    metrics = {
        "schema": RUN_SCHEMA + "-confirmation-cache-metrics",
        "schema_version": 1,
        "path_count": len(paths),
        "selected_row_count": int(joined["sample_keys"].size),
        "expected_selected_row_count": len(paths)
        * len([step for step in SELECTED_OUTER_STEPS if step < outer_steps])
        * PHASE_COUNT
        * MIDPOINT_COUNT,
        "transition_count": total_count,
        "certificate_fraction": total_certified / max(total_count, 1),
        "maximum_mass_error": max(float(item.get("maximum_mass_error", math.inf)) for item in diagnostics),
        "transitions_per_second": total_count / max(elapsed, np.finfo(float).tiny),
        "fallback_fraction": fallback_count / max(total_count, 1),
        "fallback_time_fraction": fallback_seconds / max(elapsed, np.finfo(float).tiny),
        "peak_memory_fraction": max(
            float(record.get("peak_memory_bytes", 0))
            / max(1, int(record.get("device_total_memory_bytes", 1)))
            for record in records
        ),
        "confirmation_persisted_bytes": persisted_confirmation,
        "total_persisted_bytes": total_persisted,
        "recomputed_shard_count": recompute,
        "risk_artifact": risk_artifact,
        "forbidden_counts": forbidden,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "confirmation_cache_metrics.json", metrics)
    integrity = {
        "complete_cartesian_rows": metrics["selected_row_count"] == metrics["expected_selected_row_count"],
        "certificate_fraction": metrics["certificate_fraction"] == 1.0,
        "mass_conservation": metrics["maximum_mass_error"] <= 2.0e-12,
        "forbidden_events": all(value == 0 for value in forbidden.values()),
        "throughput": metrics["transitions_per_second"] >= 1300.0,
        "fallback_fraction": metrics["fallback_fraction"] <= 1.0e-4,
        "fallback_time_fraction": metrics["fallback_time_fraction"] <= 0.10,
        "memory": metrics["peak_memory_fraction"] <= 0.80,
        "persisted_size": total_persisted <= 5 * 1024**3 // 4,
        "selection_sealed_before_paths": int(seal["confirmation_created_after_selection"]) == 1,
        "raw_confirmation_labels_not_persisted": all(
            int(record["raw_branch_labels_persisted"]) == 0 for record in records
        ),
    }
    gate = evaluate_confirmation_gate(
        max_t,
        integrity_checks=integrity,
        thresholds=thresholds,
    )
    index = {
        "schema": RUN_SCHEMA + "-confirmation-index",
        "schema_version": 1,
        "path_ids": list(paths),
        "selection_sha256": selection_sha,
        "scientific_config_sha256": _load_json(run_dir / "scientific_config.json")[
            "semantic_sha256"
        ],
        "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")[
            "semantic_sha256"
        ],
        "source_fingerprint": _load_json(run_dir / "run_manifest.json")[
            "source_fingerprint"
        ],
        "shards": [
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
        "artifacts": {
            name: file_fingerprint(run_dir / name)
            for name in CONFIRMATION_INDEX_ARTIFACTS
        },
        **NO_WORK,
    }
    index["semantic_sha256"] = config_fingerprint(index)
    atomic_write_json(run_dir / "confirmation_index.json", index)
    gate = {
        **gate,
        "confirmation_index_sha256": file_fingerprint(
            run_dir / "confirmation_index.json"
        ),
    }
    gate["semantic_sha256"] = config_fingerprint(gate)
    # The gate is the last commit.  An interruption before this write leaves
    # only resumable shards/summaries, never an apparently complete stage.
    atomic_write_json(gate_path, gate)
    return gate


def _verify_confirmation_integrity(run_dir: Path, args: argparse.Namespace) -> None:
    index = _load_json(run_dir / "confirmation_index.json")
    body = dict(index)
    semantic = body.pop("semantic_sha256", None)
    if semantic != config_fingerprint(body):
        raise ArtifactCompatibilityError("confirmation index changed")
    paths = tuple(sorted(_effective_paths(args)["confirmation"]))
    selection_sha = file_fingerprint(run_dir / "checkpoint_selection.json")
    config_sha = _load_json(run_dir / "scientific_config.json")["semantic_sha256"]
    plan_sha = _load_json(run_dir / "path_id_plan.json")["semantic_sha256"]
    source_sha = _load_json(run_dir / "run_manifest.json")["source_fingerprint"]
    if (
        tuple(index.get("path_ids", ())) != paths
        or index.get("selection_sha256") != selection_sha
        or index.get("scientific_config_sha256") != config_sha
        or index.get("path_plan_sha256") != plan_sha
        or index.get("source_fingerprint") != source_sha
        or set(index.get("artifacts", {})) != set(CONFIRMATION_INDEX_ARTIFACTS)
    ):
        raise ArtifactCompatibilityError("confirmation index binding changed")
    for name, digest in index.get("artifacts", {}).items():
        if digest != file_fingerprint(run_dir / name):
            raise ArtifactCompatibilityError("confirmation summary artifact changed")
    gate = _load_json(run_dir / "confirm_gate.json")
    gate_body = dict(gate)
    gate_semantic = gate_body.pop("semantic_sha256", None)
    if (
        gate_semantic != config_fingerprint(gate_body)
        or gate.get("confirmation_index_sha256")
        != file_fingerprint(run_dir / "confirmation_index.json")
    ):
        raise ArtifactCompatibilityError("confirmation gate binding changed")
    shard_items = index.get("shards")
    if not isinstance(shard_items, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"cohort_index", "start_step", "metadata_sha256"}
        for item in shard_items
    ):
        raise ArtifactCompatibilityError("confirmation shard index is malformed")
    metadata_hashes = {
        (int(item["cohort_index"]), int(item["start_step"])): item["metadata_sha256"]
        for item in shard_items
    }
    outer_steps = int(args.test_outer_steps) if args.test_only else OUTER_STEPS
    expected_count = len(_cohorts(paths)) * (outer_steps // SHARD_STEPS)
    if len(shard_items) != expected_count or len(metadata_hashes) != expected_count:
        raise ArtifactCompatibilityError("confirmation shard index changed")
    source = _load_source_target(args.parent_coarse_residual_run_dir)
    for cohort_index, cohort in enumerate(_cohorts(paths)):
        current = np.repeat(source[None, :], len(cohort), axis=0).copy(order="C")
        for start_step in range(0, outer_steps, SHARD_STEPS):
            metadata_path = _confirmation_shard_paths(
                run_dir, cohort_index=cohort_index, start_step=start_step
            )[3]
            if metadata_hashes.get((cohort_index, start_step)) != file_fingerprint(metadata_path):
                raise ArtifactCompatibilityError("confirmation shard metadata changed")
            valid = _valid_confirmation_shard(
                run_dir,
                cohort_index=cohort_index,
                path_ids=cohort,
                start_step=start_step,
                current=current,
                selected_step=_selected_step_in_shard(start_step, outer_steps),
                selection_sha256=selection_sha,
            )
            if valid is None:
                raise ArtifactCompatibilityError("confirmation shard chain changed")
            current, _ = valid


def _load_confirmation_control_audit(
    run_dir: Path,
    *,
    anchor: int,
    phase: int | None,
    expected_path_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for metadata_path in sorted((run_dir / "confirmation" / "shards").rglob("step-*.json")):
        record = _load_json(metadata_path)
        if int(record.get("selected_step", -1)) != int(anchor):
            continue
        audit_path = metadata_path.with_name(
            metadata_path.stem + "-control-audit.npz"
        )
        if record.get("control_audit_file_sha256") != file_fingerprint(audit_path):
            raise ArtifactCompatibilityError("confirmation control audit changed")
        audit = _load_npz(audit_path)
        paths = np.asarray(audit["path_ids"], dtype=np.int64)
        if phase is None:
            earlier = np.asarray(audit["eight_phase_earlier_states"], dtype=np.float64)
            later = np.asarray(audit["eight_phase_later_states"], dtype=np.float64)
        else:
            earlier = np.asarray(
                audit["one_phase_earlier_states"][int(phase)], dtype=np.float64
            )
            later = np.asarray(
                audit["one_phase_later_states"][int(phase)], dtype=np.float64
            )
        rows.append((paths, earlier, later))
    if not rows:
        raise ArtifactCompatibilityError("confirmation control audit is missing")
    paths = np.concatenate([item[0] for item in rows])
    earlier = np.concatenate([item[1] for item in rows])
    later = np.concatenate([item[2] for item in rows])
    order = np.argsort(paths, kind="stable")
    paths = paths[order]
    expected = np.asarray(sorted(expected_path_ids), dtype=np.int64)
    if (
        not np.array_equal(paths, expected)
        or earlier.shape != (len(expected), STATE_SIZE)
        or later.shape != earlier.shape
        or not np.isfinite(earlier).all()
        or not np.isfinite(later).all()
        or np.any(earlier < 0.0)
        or np.any(later < 0.0)
    ):
        raise ArtifactCompatibilityError("confirmation control audit design changed")
    earlier = np.ascontiguousarray(earlier[order])
    later = np.ascontiguousarray(later[order])
    simplex_error = max(
        float(np.max(np.abs(earlier.sum(axis=-1) - 1.0), initial=0.0)),
        float(np.max(np.abs(later.sum(axis=-1) - 1.0), initial=0.0)),
    )
    if phase is not None:
        from mnist.d0_jacobi_rb_learnability import matching_indices

        tails_all, heads_all = matching_indices(device="cpu")
        color = int(PHASE_MATCHINGS[int(phase)])
        tails = tails_all[color].cpu().numpy()
        heads = heads_all[color].cpu().numpy()
        pair_error = float(
            np.max(
                np.abs(
                    (earlier[:, tails] + earlier[:, heads])
                    - (later[:, tails] + later[:, heads])
                ),
                initial=0.0,
            )
        )
    else:
        pair_error = 0.0
    if simplex_error > 2.0e-12 or pair_error > 2.0e-12:
        raise ArtifactCompatibilityError(
            "confirmation control audit violates exact mass conservation"
        )
    return earlier, later


def _valid_control_phase_checkpoint(
    state_path: Path,
    record_path: Path,
    *,
    binding: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not state_path.is_file() or not record_path.is_file():
        return None
    try:
        record = _load_json(record_path)
        body = dict(record)
        semantic = body.pop("semantic_sha256", None)
        if semantic != config_fingerprint(body) or any(
            record.get(name) != value for name, value in binding.items()
        ):
            return None
        if (
            int(record.get("committed", 0)) != 1
            or record.get("state_file_sha256") != file_fingerprint(state_path)
            or int(record.get("state_file_size", -1)) != state_path.stat().st_size
        ):
            return None
        state = _load_npz(state_path).get("state")
        expected_paths = tuple(int(value) for value in binding.get("path_ids", ()))
        if (
            state is None
            or state.dtype != np.float64
            or state.shape != (len(expected_paths), STATE_SIZE)
            or not np.isfinite(state).all()
            or np.any(state < 0.0)
            or float(
                np.max(np.abs(state.sum(axis=-1) - 1.0), initial=0.0)
            )
            > 2.0e-12
            or record.get("state_array_sha256") != _array_sha(state)
        ):
            return None
        diagnostics = record.get("reference_diagnostics")
        microsteps = int(binding.get("microsteps", -1))
        expected_transitions = (
            len(expected_paths) * EDGES_PER_PHASE * 2 * microsteps
            if microsteps in CONTROL_MICROSTEPS
            else -1
        )
        controller_forbidden = record.get("controller_forbidden_counts")
        if (
            not isinstance(diagnostics, Mapping)
            or int(diagnostics.get("transition_count", -1))
            != expected_transitions
            or int(diagnostics.get("certified_count", -1))
            != expected_transitions
            or int(diagnostics.get("fallback_count", -1)) < 0
            or not math.isfinite(float(diagnostics.get("fallback_seconds", math.nan)))
            or not math.isfinite(float(diagnostics.get("elapsed_seconds", math.nan)))
            or int(diagnostics.get("maximum_transition_count_per_call", 1 << 30))
            > 4096
            or not isinstance(diagnostics.get("forbidden_counts"), Mapping)
            or any(
                int(diagnostics["forbidden_counts"].get(name, -1)) != 0
                for name in FORBIDDEN_COUNTS
                if name != "uncertified_count"
            )
            or not isinstance(controller_forbidden, Mapping)
            or set(controller_forbidden)
            != {
                "clip_count",
                "floor_count",
                "limiter_count",
                "projection_count",
                "renormalization_count",
            }
            or any(int(value) != 0 for value in controller_forbidden.values())
            or int(record.get("states_finite", 0)) != 1
            or int(record.get("states_nonnegative", 0)) != 1
            or int(record.get("boundary_rejection_count", -1)) != 0
            or float(record.get("maximum_pair_mass_error", math.inf)) > 2.0e-12
            or float(record.get("maximum_simplex_mass_error", math.inf)) > 2.0e-12
        ):
            return None
        return np.ascontiguousarray(state), record
    except (ArtifactCompatibilityError, OSError, ValueError, TypeError, KeyError):
        return None


class _TangentCertifiedReference:
    """Small exact-reference adapter with workflow-local RNG provenance."""

    def __init__(self, *, profile: JacobiRBCudaProfile, stream_role: str) -> None:
        self.profile = profile
        self.stream_role = str(stream_role)
        self.transition_count = 0
        self.certified_count = 0
        self.fallback_count = 0
        self.fallback_seconds = 0.0
        self.elapsed_seconds = 0.0
        self.maximum_transition_count_per_call = 0
        self.forbidden = {
            name: 0 for name in FORBIDDEN_COUNTS if name != "uncertified_count"
        }

    def __call__(
        self,
        *,
        head_fraction: Tensor,
        exposure: Tensor,
        transition_ids: Tensor,
        role: str,
    ) -> Any:
        from mnist.d0_jacobi_rb_cuda import sample_alpha1_rb_transition_batch_cuda

        count = int(head_fraction.numel())
        self.maximum_transition_count_per_call = max(
            self.maximum_transition_count_per_call, count
        )
        if count > 4096:
            raise BoundaryTangentCLIError(
                "controller reference launch exceeds 4096 transitions",
                failure_domain="resource",
                failure_code="boundary_tangent_reference_launch_too_large",
            )
        started = time.perf_counter()
        result = sample_alpha1_rb_transition_batch_cuda(
            head_fraction.contiguous(),
            exposure.contiguous(),
            rng_key=(
                CONTROL_SEED,
                "d0-jacobi-rb-boundary-tangent-controller-reference-v1",
                self.stream_role,
                str(role),
            ),
            transition_ids=transition_ids.contiguous(),
            profile=self.profile,
        )
        certified = int(result.certified_mask.sum().detach().cpu().item())
        if certified != count:
            raise BoundaryTangentCLIError(
                "controller reference transition was uncertified",
                failure_domain="controller_numerics",
                failure_code="boundary_tangent_reference_uncertified",
            )
        diagnostics = result.diagnostics if isinstance(result.diagnostics, Mapping) else {}
        self.transition_count += count
        self.certified_count += certified
        self.fallback_count += int(result.fallback_mask.sum().detach().cpu())
        self.fallback_seconds += float(diagnostics.get("arb_fallback_elapsed_seconds", 0.0))
        for name in self.forbidden:
            self.forbidden[name] += int(diagnostics.get(name, 0))
        self.elapsed_seconds += time.perf_counter() - started
        return result

    def record(self) -> dict[str, Any]:
        return {
            "transition_count": self.transition_count,
            "certified_count": self.certified_count,
            "fallback_count": self.fallback_count,
            "fallback_seconds": self.fallback_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "maximum_transition_count_per_call": self.maximum_transition_count_per_call,
            "forbidden_counts": dict(self.forbidden),
        }


def _run_tangent_control_trajectory(
    run_dir: Path,
    *,
    stem: str,
    later_state: np.ndarray,
    sequence: Sequence[tuple[int, int]],
    microsteps: int,
    controller: nn.Module,
    device: torch.device,
    path_ids: Sequence[int],
    profile: JacobiRBCudaProfile,
    stream_role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from mnist.d0_jacobi_rb_boundary_tangent import controlled_reverse_phase_tangent
    from mnist.d0_jacobi_rb_reverse_controller import NAMESPACE_VERSION

    if len(sequence) > 8:
        raise BoundaryTangentCLIError(
            "controller trajectory exceeds eight phase occurrences",
            failure_domain="controller_contract",
            failure_code="boundary_tangent_full_reverse_path_forbidden",
        )
    state = np.ascontiguousarray(later_state, dtype=np.float64)
    phase_records: list[dict[str, Any]] = []
    directory = run_dir / "control" / "phase_checkpoints"
    for occurrence, (outer_step, phase) in enumerate(sequence):
        state_path = directory / f"{stem}-M{microsteps}-occurrence-{occurrence:02d}.npz"
        record_path = directory / f"{stem}-M{microsteps}-occurrence-{occurrence:02d}.json"
        binding = {
            "schema": RUN_SCHEMA + "-control-phase-checkpoint",
            "schema_version": 1,
            "stem": stem,
            "microsteps": int(microsteps),
            "occurrence": occurrence,
            "outer_step": int(outer_step),
            "phase": int(phase),
            "phase_count": 1,
            "input_state_sha256": _array_sha(state),
            "path_ids": list(path_ids),
            "stream_role": stream_role,
            "selection_sha256": file_fingerprint(run_dir / "checkpoint_selection.json"),
            "controller_control_start_sha256": file_fingerprint(
                run_dir / "controller_control_started.json"
            ),
            "confirmation_index_sha256": file_fingerprint(
                run_dir / "confirmation_index.json"
            ),
            "selected_state_sha256": _load_json(run_dir / "checkpoint_selection.json")[
                "selected_state_sha256"
            ],
            "baseline_file_sha256": _load_json(run_dir / "checkpoint_selection.json")[
                "baseline_file_sha256"
            ],
            "scientific_config_sha256": _load_json(run_dir / "scientific_config.json")[
                "semantic_sha256"
            ],
            "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")[
                "semantic_sha256"
            ],
            "source_fingerprint": _load_json(run_dir / "run_manifest.json")[
                "source_fingerprint"
            ],
            "profile_sha256": config_fingerprint(profile.to_dict()),
        }
        cached = _valid_control_phase_checkpoint(
            state_path, record_path, binding=binding
        )
        if cached is not None:
            state, record = cached
            phase_records.append(record)
            continue
        tensor = torch.as_tensor(
            np.array(state, copy=True, order="C"), dtype=torch.float64, device=device
        ).contiguous()
        reference = _TangentCertifiedReference(
            profile=profile, stream_role=stream_role
        )
        started = time.perf_counter()
        pieces: list[Tensor] = []
        maximum_pair = 0.0
        maximum_simplex = 0.0
        for first in range(0, len(path_ids), 8):
            last = min(first + 8, len(path_ids))
            result = controlled_reverse_phase_tangent(
                tensor[first:last],
                outer_step,
                phase,
                microsteps,
                NAMESPACE_VERSION,
                controller=controller,
                reference_transition=reference,
                path_ids=path_ids[first:last],
                label=3,
            )
            pieces.append(result.state)
            maximum_pair = max(maximum_pair, result.maximum_pair_mass_error)
            maximum_simplex = max(maximum_simplex, result.maximum_simplex_mass_error)
        tensor = torch.cat(pieces, dim=0).contiguous()
        state = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float64)
        artifact = _atomic_npz(state_path, {"state": state})
        record = {
            **binding,
            "state_file_sha256": artifact["sha256"],
            "state_file_size": artifact["size"],
            "state_array_sha256": _array_sha(state),
            "reference_diagnostics": reference.record(),
            "maximum_pair_mass_error": maximum_pair,
            "maximum_simplex_mass_error": maximum_simplex,
            "states_finite": int(np.isfinite(state).all()),
            "states_nonnegative": int(np.all(state >= 0.0)),
            "boundary_rejection_count": 0,
            "controller_forbidden_counts": {
                "clip_count": 0,
                "floor_count": 0,
                "limiter_count": 0,
                "projection_count": 0,
                "renormalization_count": 0,
            },
            "complete_pipeline_elapsed_seconds": float(time.perf_counter() - started),
            "committed": 1,
            **NO_WORK,
        }
        record["semantic_sha256"] = config_fingerprint(record)
        atomic_write_json(record_path, record)
        phase_records.append(record)
    reference_names = tuple(phase_records[0]["reference_diagnostics"]["forbidden_counts"])
    aggregate = {
        "schema": RUN_SCHEMA + "-control-trajectory",
        "schema_version": 1,
        "stem": stem,
        "microsteps": microsteps,
        "sequence": [[int(step), int(phase)] for step, phase in sequence],
        "phase_count": len(sequence),
        "full_reverse_path": 0,
        "path_ids": list(path_ids),
        "final_state_sha256": _array_sha(state),
        "reference_diagnostics": {
            "transition_count": sum(int(item["reference_diagnostics"]["transition_count"]) for item in phase_records),
            "certified_count": sum(int(item["reference_diagnostics"]["certified_count"]) for item in phase_records),
            "fallback_count": sum(int(item["reference_diagnostics"]["fallback_count"]) for item in phase_records),
            "fallback_seconds": sum(float(item["reference_diagnostics"]["fallback_seconds"]) for item in phase_records),
            "elapsed_seconds": sum(float(item["reference_diagnostics"]["elapsed_seconds"]) for item in phase_records),
            "forbidden_counts": {
                name: sum(int(item["reference_diagnostics"]["forbidden_counts"][name]) for item in phase_records)
                for name in reference_names
            },
        },
        "maximum_pair_mass_error": max(float(item["maximum_pair_mass_error"]) for item in phase_records),
        "maximum_simplex_mass_error": max(float(item["maximum_simplex_mass_error"]) for item in phase_records),
        "states_finite": int(all(int(item["states_finite"]) == 1 for item in phase_records)),
        "states_nonnegative": int(all(int(item["states_nonnegative"]) == 1 for item in phase_records)),
        "boundary_rejection_count": 0,
        "controller_forbidden_counts": {
            name: sum(int(item["controller_forbidden_counts"][name]) for item in phase_records)
            for name in phase_records[0]["controller_forbidden_counts"]
        },
        "complete_pipeline_elapsed_seconds": sum(float(item["complete_pipeline_elapsed_seconds"]) for item in phase_records),
        **NO_WORK,
    }
    return state, aggregate


def _control_phase_index_entries(
    run_dir: Path,
    *,
    path_ids: Sequence[int],
    profile: JacobiRBCudaProfile,
) -> list[dict[str, Any]]:
    """Verify every phase-checkpoint chain and return its canonical index."""

    paths = tuple(int(value) for value in path_ids)
    descriptors: list[
        tuple[str, int, tuple[tuple[int, int], ...], np.ndarray]
    ] = []
    for anchor in CONTROL_ANCHORS:
        for phase in range(PHASE_COUNT):
            _, later = _load_confirmation_control_audit(
                run_dir,
                anchor=anchor,
                phase=phase,
                expected_path_ids=paths,
            )
            for microsteps in CONTROL_MICROSTEPS:
                descriptors.append(
                    (
                        f"one-phase-anchor-{anchor:04d}-phase-{phase}",
                        microsteps,
                        ((anchor, phase),),
                        later,
                    )
                )
        _, later = _load_confirmation_control_audit(
            run_dir,
            anchor=anchor,
            phase=None,
            expected_path_ids=paths,
        )
        sequence = tuple(
            [(anchor, phase) for phase in range(PHASE_COUNT - 1, -1, -1)]
            + [(anchor - 1, PHASE_COUNT - 1)]
        )
        for microsteps in CONTROL_MICROSTEPS:
            descriptors.append(
                (
                    f"eight-phase-anchor-{anchor:04d}",
                    microsteps,
                    sequence,
                    later,
                )
            )

    directory = run_dir / "control" / "phase_checkpoints"
    entries: list[dict[str, Any]] = []
    expected_state_paths: set[Path] = set()
    expected_record_paths: set[Path] = set()
    shared_binding = {
        "selection_sha256": file_fingerprint(
            run_dir / "checkpoint_selection.json"
        ),
        "controller_control_start_sha256": file_fingerprint(
            run_dir / "controller_control_started.json"
        ),
        "confirmation_index_sha256": file_fingerprint(
            run_dir / "confirmation_index.json"
        ),
        "selected_state_sha256": _load_json(
            run_dir / "checkpoint_selection.json"
        )["selected_state_sha256"],
        "baseline_file_sha256": _load_json(
            run_dir / "checkpoint_selection.json"
        )["baseline_file_sha256"],
        "scientific_config_sha256": _load_json(
            run_dir / "scientific_config.json"
        )["semantic_sha256"],
        "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")[
            "semantic_sha256"
        ],
        "source_fingerprint": _load_json(run_dir / "run_manifest.json")[
            "source_fingerprint"
        ],
        "profile_sha256": config_fingerprint(profile.to_dict()),
    }
    for stem, microsteps, sequence, initial_state in descriptors:
        state = np.ascontiguousarray(initial_state, dtype=np.float64)
        stream_role = (
            f"tangent-eight-phase-anchor-{sequence[0][0]}-M{microsteps}"
            if len(sequence) == 8
            else f"tangent-one-phase-anchor-{sequence[0][0]}-"
            f"phase-{sequence[0][1]}-M{microsteps}"
        )
        for occurrence, (outer_step, phase) in enumerate(sequence):
            state_path = directory / (
                f"{stem}-M{microsteps}-occurrence-{occurrence:02d}.npz"
            )
            record_path = directory / (
                f"{stem}-M{microsteps}-occurrence-{occurrence:02d}.json"
            )
            expected_state_paths.add(state_path.resolve())
            expected_record_paths.add(record_path.resolve())
            binding = {
                "schema": RUN_SCHEMA + "-control-phase-checkpoint",
                "schema_version": 1,
                "stem": stem,
                "microsteps": int(microsteps),
                "occurrence": occurrence,
                "outer_step": int(outer_step),
                "phase": int(phase),
                "phase_count": 1,
                "input_state_sha256": _array_sha(state),
                "path_ids": list(paths),
                "stream_role": stream_role,
                **shared_binding,
            }
            valid = _valid_control_phase_checkpoint(
                state_path, record_path, binding=binding
            )
            if valid is None:
                raise ArtifactCompatibilityError(
                    "controller phase-checkpoint chain changed"
                )
            state, record = valid
            entries.append(
                {
                    "stem": stem,
                    "microsteps": int(microsteps),
                    "occurrence": occurrence,
                    "state_path": state_path.relative_to(run_dir).as_posix(),
                    "state_sha256": file_fingerprint(state_path),
                    "record_path": record_path.relative_to(run_dir).as_posix(),
                    "record_sha256": file_fingerprint(record_path),
                    "final_state_sha256": record["state_array_sha256"],
                }
            )
    actual_states = {
        path.resolve() for path in directory.glob("*.npz") if path.is_file()
    }
    actual_records = {
        path.resolve() for path in directory.glob("*.json") if path.is_file()
    }
    if actual_states != expected_state_paths or actual_records != expected_record_paths:
        raise ArtifactCompatibilityError(
            "controller phase-checkpoint file set changed"
        )
    return entries


def _control_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent import load_tangent_baseline
    from mnist.d0_jacobi_rb_boundary_tangent_confirmation import (
        normalized_controller_trajectory_max_t,
    )
    from mnist.d0_jacobi_rb_reverse_controller import (
        NAMESPACE_VERSION,
        paired_observables,
    )

    if not _passed(_load_json(run_dir / "confirm_gate.json")):
        raise ArtifactCompatibilityError("control stage requires a passing confirmation gate")
    _verify_confirmation_integrity(run_dir, args)
    gate_path = run_dir / "control_gate.json"
    if gate_path.is_file():
        _verify_control_integrity(run_dir, args)
        return _load_json(gate_path)
    selection = _verify_training_selection(run_dir)
    paths = tuple(sorted(_effective_paths(args)["confirmation"]))
    if len(paths) < 8:
        raise ArtifactCompatibilityError("control requires at least eight confirmation paths")
    profile = JacobiRBCudaProfile()
    profile_sha = config_fingerprint(profile.to_dict())
    start_record = {
        "schema": RUN_SCHEMA + "-controller-control-start",
        "schema_version": 1,
        "selection_sha256": file_fingerprint(run_dir / "checkpoint_selection.json"),
        "confirmation_gate_sha256": file_fingerprint(run_dir / "confirm_gate.json"),
        "confirmation_index_sha256": file_fingerprint(
            run_dir / "confirmation_index.json"
        ),
        "checkpoint_file_sha256": selection["checkpoint_file_sha256"],
        "checkpoint_state_sha256": selection["selected_state_sha256"],
        "baseline_file_sha256": selection["baseline_file_sha256"],
        "scientific_config_sha256": _load_json(run_dir / "scientific_config.json")[
            "semantic_sha256"
        ],
        "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")[
            "semantic_sha256"
        ],
        "source_fingerprint": _load_json(run_dir / "run_manifest.json")[
            "source_fingerprint"
        ],
        "cuda_profile_sha256": profile_sha,
        "transition_namespace": NAMESPACE_VERSION,
        "root_seed": CONTROL_SEED,
        "path_ids": list(paths),
        "anchors": list(CONTROL_ANCHORS),
        "microsteps": list(CONTROL_MICROSTEPS),
        "maximum_phase_count": 8,
        **NO_WORK,
    }
    start_record["semantic_sha256"] = config_fingerprint(start_record)
    start_path = run_dir / "controller_control_started.json"
    if start_path.is_file():
        if _load_json(start_path) != start_record:
            raise ArtifactCompatibilityError("controller-control start seal changed")
    else:
        atomic_write_json(start_path, start_record)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline = load_tangent_baseline(
        run_dir / selection["baseline_path"],
        expected_sha256=selection["baseline_file_sha256"],
    )
    controller = _load_candidate_model(run_dir, selection, baseline, device)
    numerators: list[np.ndarray] = []
    forward_changes: list[np.ndarray] = []
    names: list[str] = []
    records: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    structural_pass = True
    for anchor in CONTROL_ANCHORS:
        for phase in range(PHASE_COUNT):
            earlier, later = _load_confirmation_control_audit(
                run_dir, anchor=anchor, phase=phase, expected_path_ids=paths
            )
            forward = paired_observables(earlier, later, phase=phase)
            outputs: dict[int, np.ndarray] = {}
            phase_records: dict[int, dict[str, Any]] = {}
            for microsteps in CONTROL_MICROSTEPS:
                outputs[microsteps], phase_records[microsteps] = _run_tangent_control_trajectory(
                    run_dir,
                    stem=f"one-phase-anchor-{anchor:04d}-phase-{phase}",
                    later_state=later,
                    sequence=((anchor, phase),),
                    microsteps=microsteps,
                    controller=controller,
                    device=device,
                    path_ids=paths,
                    profile=profile,
                    stream_role=f"tangent-one-phase-anchor-{anchor}-phase-{phase}-M{microsteps}",
                )
                records.append(phase_records[microsteps])
            reverse8 = paired_observables(earlier, outputs[8], phase=phase)
            refine = paired_observables(outputs[4], outputs[8], phase=phase)
            structural = np.asarray(forward.structural_invariant, dtype=bool)
            structural_pass &= bool(
                np.all(forward.difference[:, structural] == 0.0)
                and np.all(reverse8.difference[:, structural] == 0.0)
            )
            for observable, name in enumerate(forward.names):
                if structural[observable]:
                    continue
                prefix = f"one_phase.anchor{anchor}.phase{phase}.{name}"
                numerators.extend(
                    (reverse8.difference[:, observable], refine.difference[:, observable])
                )
                forward_changes.extend(
                    (forward.difference[:, observable], forward.difference[:, observable])
                )
                names.extend((prefix + ".bias", prefix + ".M8_vs_M4"))
                raw_rows.append(
                    {
                        "scope": "one_phase",
                        "anchor": anchor,
                        "phase": phase,
                        "observable": name,
                        "M8_bias_mean": float(np.mean(reverse8.difference[:, observable])),
                        "M8_vs_M4_mean": float(np.mean(refine.difference[:, observable])),
                    }
                )
        earlier, later = _load_confirmation_control_audit(
            run_dir, anchor=anchor, phase=None, expected_path_ids=paths
        )
        sequence = tuple(
            [(anchor, phase) for phase in range(PHASE_COUNT - 1, -1, -1)]
            + [(anchor - 1, PHASE_COUNT - 1)]
        )
        forward = paired_observables(
            earlier, later, phase=PHASE_COUNT - 1, structural_phase_invariants=False
        )
        outputs = {}
        phase_records = {}
        for microsteps in CONTROL_MICROSTEPS:
            outputs[microsteps], phase_records[microsteps] = _run_tangent_control_trajectory(
                run_dir,
                stem=f"eight-phase-anchor-{anchor:04d}",
                later_state=later,
                sequence=sequence,
                microsteps=microsteps,
                controller=controller,
                device=device,
                path_ids=paths,
                profile=profile,
                stream_role=f"tangent-eight-phase-anchor-{anchor}-M{microsteps}",
            )
            records.append(phase_records[microsteps])
        reverse8 = paired_observables(
            earlier, outputs[8], phase=PHASE_COUNT - 1, structural_phase_invariants=False
        )
        refine = paired_observables(
            outputs[4], outputs[8], phase=PHASE_COUNT - 1, structural_phase_invariants=False
        )
        for observable, name in enumerate(forward.names):
            prefix = f"eight_phase.anchor{anchor}.{name}"
            numerators.extend(
                (reverse8.difference[:, observable], refine.difference[:, observable])
            )
            forward_changes.extend(
                (forward.difference[:, observable], forward.difference[:, observable])
            )
            names.extend((prefix + ".bias", prefix + ".M8_vs_M4"))
            raw_rows.append(
                {
                    "scope": "eight_phase",
                    "anchor": anchor,
                    "phase": "eight_occurrences",
                    "observable": name,
                    "M8_bias_mean": float(np.mean(reverse8.difference[:, observable])),
                    "M8_vs_M4_mean": float(np.mean(refine.difference[:, observable])),
                }
            )
    numerator_table = np.ascontiguousarray(np.stack(numerators, axis=1), dtype=np.float64)
    forward_table = np.ascontiguousarray(np.stack(forward_changes, axis=1), dtype=np.float64)
    if numerator_table.shape != (len(paths), 784) or len(names) != 784:
        raise AssertionError("controller family is not the frozen path-by-784 table")
    replicates = int(args.test_bootstrap_replicates) if args.test_only else 50_000
    trajectory = normalized_controller_trajectory_max_t(
        numerators=numerator_table,
        forward_changes=forward_table,
        path_ids=np.asarray(paths, dtype=np.int64),
        names=names,
        confidence=0.995,
        replicates=replicates,
        seed=CONTROL_SEED,
    )
    atomic_write_json(run_dir / "controller_trajectory_max_t.json", trajectory)
    _atomic_npz(
        run_dir / "controller_trajectory_path_values.npz",
        {
            "path_ids": np.asarray(paths, dtype=np.int64),
            "numerators": numerator_table,
            "forward_changes": forward_table,
        },
    )
    atomic_write_csv(run_dir / "controller_trajectory_raw_metrics.csv", raw_rows)
    total_transitions = sum(int(item["reference_diagnostics"]["transition_count"]) for item in records)
    expected_transitions = (
        len(paths)
        * EDGES_PER_PHASE
        * 2
        * sum(CONTROL_MICROSTEPS)
        * (len(CONTROL_ANCHORS) * PHASE_COUNT + len(CONTROL_ANCHORS) * 8)
    )
    certified = sum(int(item["reference_diagnostics"]["certified_count"]) for item in records)
    fallback = sum(int(item["reference_diagnostics"]["fallback_count"]) for item in records)
    fallback_seconds = sum(float(item["reference_diagnostics"]["fallback_seconds"]) for item in records)
    reference_seconds = sum(float(item["reference_diagnostics"]["elapsed_seconds"]) for item in records)
    wall_seconds = sum(float(item["complete_pipeline_elapsed_seconds"]) for item in records)
    reference_forbidden_names = tuple(records[0]["reference_diagnostics"]["forbidden_counts"])
    forbidden = {
        name: sum(int(item["reference_diagnostics"]["forbidden_counts"][name]) for item in records)
        for name in reference_forbidden_names
    }
    controller_forbidden_names = tuple(records[0]["controller_forbidden_counts"])
    controller_forbidden = {
        name: sum(int(item["controller_forbidden_counts"][name]) for item in records)
        for name in controller_forbidden_names
    }
    persisted = sum(
        path.stat().st_size
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_registry.json", "run_status.json"}
        and not path.name.endswith(".tmp")
        and ".tmp." not in path.name
    )
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    total_memory = int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 1
    health = {
        "maximum_pair_mass_error": max(float(item["maximum_pair_mass_error"]) for item in records),
        "maximum_simplex_mass_error": max(float(item["maximum_simplex_mass_error"]) for item in records),
        "certificate_fraction": certified / max(total_transitions, 1),
        "states_finite": int(all(int(item["states_finite"]) == 1 for item in records)),
        "states_nonnegative": int(all(int(item["states_nonnegative"]) == 1 for item in records)),
        "boundary_rejection_count": sum(int(item["boundary_rejection_count"]) for item in records),
        "forbidden_counts": forbidden,
        "controller_forbidden_counts": controller_forbidden,
        "fallback_fraction": fallback / max(total_transitions, 1),
        "fallback_time_fraction": fallback_seconds / max(reference_seconds, np.finfo(float).tiny),
        "transitions_per_second": total_transitions / max(wall_seconds, np.finfo(float).tiny),
        "peak_device_memory_fraction": peak / total_memory,
        "total_persisted_bytes": persisted,
        "maximum_phase_count": max(int(item["phase_count"]) for item in records),
        "maximum_transition_count_per_call": max(
            int(item["reference_diagnostics"]["maximum_transition_count_per_call"])
            for item in records
        ),
        "transition_count": total_transitions,
        "expected_transition_count": expected_transitions,
    }
    atomic_write_json(run_dir / "controller_numerical_health.json", health)
    thresholds = BoundaryTangentThresholds(
        confirmation_paths=len(paths),
        controller_paths=len(paths),
        bootstrap_replicates=replicates,
    )
    gate = evaluate_controller_gate(
        trajectory,
        health,
        integrity_checks={
            "maximum_eight_phases": health["maximum_phase_count"] <= 8,
            "exact_transition_count": total_transitions == expected_transitions,
            "maximum_reference_launch": health["maximum_transition_count_per_call"] <= 4096,
            "structural_invariants": structural_pass,
            "selection_unchanged": start_record["selection_sha256"]
            == file_fingerprint(run_dir / "checkpoint_selection.json"),
            "confirmation_unchanged": start_record["confirmation_gate_sha256"]
            == file_fingerprint(run_dir / "confirm_gate.json"),
            "no_full_reverse_path": all(int(item["full_reverse_path"]) == 0 for item in records),
        },
        thresholds=thresholds,
    )
    phase_entries = _control_phase_index_entries(
        run_dir, path_ids=paths, profile=profile
    )
    index = {
        "schema": RUN_SCHEMA + "-controller-control-index",
        "schema_version": 1,
        "controller_control_start_sha256": file_fingerprint(start_path),
        "selection_sha256": start_record["selection_sha256"],
        "confirmation_gate_sha256": start_record["confirmation_gate_sha256"],
        "confirmation_index_sha256": start_record["confirmation_index_sha256"],
        "scientific_config_sha256": start_record["scientific_config_sha256"],
        "path_plan_sha256": start_record["path_plan_sha256"],
        "source_fingerprint": start_record["source_fingerprint"],
        "cuda_profile_sha256": profile_sha,
        "path_ids": list(paths),
        "phase_checkpoints": phase_entries,
        "artifacts": {
            name: file_fingerprint(run_dir / name)
            for name in CONTROL_INDEX_ARTIFACTS
        },
        **NO_WORK,
    }
    index["semantic_sha256"] = config_fingerprint(index)
    index_path = run_dir / "controller_control_index.json"
    atomic_write_json(index_path, index)
    gate = {
        **gate,
        "controller_control_index_sha256": file_fingerprint(index_path),
    }
    gate["semantic_sha256"] = config_fingerprint(gate)
    # Commit the required gate only after every checkpoint and its exact index.
    atomic_write_json(gate_path, gate)
    return gate


def _verify_control_integrity(run_dir: Path, args: argparse.Namespace) -> None:
    """Verify the terminal control commit and every resumable phase chain."""

    _verify_confirmation_integrity(run_dir, args)
    selection = _verify_training_selection(run_dir)
    paths = tuple(sorted(_effective_paths(args)["confirmation"]))
    profile = JacobiRBCudaProfile()
    profile_sha = config_fingerprint(profile.to_dict())
    start_path = run_dir / "controller_control_started.json"
    start = _load_json(start_path)
    start_body = dict(start)
    start_semantic = start_body.pop("semantic_sha256", None)
    from mnist.d0_jacobi_rb_reverse_controller import NAMESPACE_VERSION

    expected_start = {
        "schema": RUN_SCHEMA + "-controller-control-start",
        "schema_version": 1,
        "selection_sha256": file_fingerprint(
            run_dir / "checkpoint_selection.json"
        ),
        "confirmation_gate_sha256": file_fingerprint(
            run_dir / "confirm_gate.json"
        ),
        "confirmation_index_sha256": file_fingerprint(
            run_dir / "confirmation_index.json"
        ),
        "checkpoint_file_sha256": selection["checkpoint_file_sha256"],
        "checkpoint_state_sha256": selection["selected_state_sha256"],
        "baseline_file_sha256": selection["baseline_file_sha256"],
        "scientific_config_sha256": _load_json(
            run_dir / "scientific_config.json"
        )["semantic_sha256"],
        "path_plan_sha256": _load_json(run_dir / "path_id_plan.json")[
            "semantic_sha256"
        ],
        "source_fingerprint": _load_json(run_dir / "run_manifest.json")[
            "source_fingerprint"
        ],
        "cuda_profile_sha256": profile_sha,
        "transition_namespace": NAMESPACE_VERSION,
        "root_seed": CONTROL_SEED,
        "path_ids": list(paths),
        "anchors": list(CONTROL_ANCHORS),
        "microsteps": list(CONTROL_MICROSTEPS),
        "maximum_phase_count": 8,
        **NO_WORK,
    }
    if start_semantic != config_fingerprint(start_body) or start_body != expected_start:
        raise ArtifactCompatibilityError("controller-control start seal changed")

    index_path = run_dir / "controller_control_index.json"
    index = _load_json(index_path)
    index_body = dict(index)
    index_semantic = index_body.pop("semantic_sha256", None)
    if (
        index_semantic != config_fingerprint(index_body)
        or index.get("controller_control_start_sha256")
        != file_fingerprint(start_path)
        or index.get("selection_sha256") != expected_start["selection_sha256"]
        or index.get("confirmation_gate_sha256")
        != expected_start["confirmation_gate_sha256"]
        or index.get("confirmation_index_sha256")
        != expected_start["confirmation_index_sha256"]
        or index.get("scientific_config_sha256")
        != expected_start["scientific_config_sha256"]
        or index.get("path_plan_sha256") != expected_start["path_plan_sha256"]
        or index.get("source_fingerprint") != expected_start["source_fingerprint"]
        or index.get("cuda_profile_sha256") != profile_sha
        or tuple(index.get("path_ids", ())) != paths
        or set(index.get("artifacts", {})) != set(CONTROL_INDEX_ARTIFACTS)
    ):
        raise ArtifactCompatibilityError("controller-control index binding changed")
    for name, digest in index["artifacts"].items():
        if digest != file_fingerprint(run_dir / name):
            raise ArtifactCompatibilityError("controller-control artifact changed")
    expected_entries = _control_phase_index_entries(
        run_dir, path_ids=paths, profile=profile
    )
    if index.get("phase_checkpoints") != expected_entries:
        raise ArtifactCompatibilityError("controller phase-checkpoint index changed")

    values = _load_npz(run_dir / "controller_trajectory_path_values.npz")
    if (
        set(values) != {"path_ids", "numerators", "forward_changes"}
        or not np.array_equal(
            values["path_ids"], np.asarray(paths, dtype=np.int64)
        )
        or np.asarray(values["numerators"]).dtype != np.dtype(np.float64)
        or np.asarray(values["forward_changes"]).dtype != np.dtype(np.float64)
        or np.asarray(values["numerators"]).shape != (len(paths), 784)
        or np.asarray(values["forward_changes"]).shape != (len(paths), 784)
        or not np.isfinite(values["numerators"]).all()
        or not np.isfinite(values["forward_changes"]).all()
    ):
        raise ArtifactCompatibilityError("controller trajectory evidence changed")
    gate = _load_json(run_dir / "control_gate.json")
    gate_body = dict(gate)
    gate_semantic = gate_body.pop("semantic_sha256", None)
    if (
        gate_semantic != config_fingerprint(gate_body)
        or gate.get("controller_control_index_sha256")
        != file_fingerprint(index_path)
    ):
        raise ArtifactCompatibilityError("controller-control gate binding changed")


STAGES = ("preflight", "cache", "train", "confirm", "control", "report", "all")
REQUIRED_GATES = ("none", "preflight", "cache", "train", "confirm", "control")
_GATE_FILENAMES = {
    "preflight": "preflight_gate.json",
    "cache": "cache_gate.json",
    "train": "train_gate.json",
    "confirm": "confirm_gate.json",
    "control": "control_gate.json",
}


def _gate_schema(stage: str) -> str:
    if stage in {"confirm", "control"}:
        return f"d0-jacobi-rb-boundary-tangent-gate-v1-{stage}-gate"
    return RUN_SCHEMA + f"-{stage}-gate"


def _optional_json(run_dir: Path, filename: str) -> dict[str, Any] | None:
    path = run_dir / filename
    return _load_json(path) if path.is_file() else None


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "cache", "train", "confirm", "control")
    if stage == "report":
        return ()
    if stage not in _GATE_FILENAMES:
        raise ValueError(f"unknown boundary-tangent stage: {stage}")
    return (stage,)


def _gate_passed_for_requirement(value: Mapping[str, Any] | None) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("evaluation_status") == "evaluated"
        and int(value.get("passed", 0)) == 1
    )


def _controller_scope(run_dir: Path) -> tuple[bool, int]:
    completed = (run_dir / "controller_control_index.json").is_file()
    health_path = run_dir / "controller_numerical_health.json"
    maximum_phases = 0
    if completed and health_path.is_file():
        try:
            candidate = int(_load_json(health_path).get("maximum_phase_count", 0))
        except (ArtifactCompatibilityError, TypeError, ValueError):
            candidate = 0
        maximum_phases = candidate if 0 <= candidate <= 8 else 0
    return completed, maximum_phases


def _workflow_record(run_dir: Path, *, require_gate: str) -> dict[str, Any]:
    gates = {
        stage: _optional_json(run_dir, filename)
        for stage, filename in _GATE_FILENAMES.items()
    }
    decision = decide_boundary_tangent_workflow(
        preflight_gate=gates["preflight"],
        cache_gate=gates["cache"],
        train_gate=gates["train"],
        confirm_gate=gates["confirm"],
        control_gate=gates["control"],
    )
    if require_gate not in REQUIRED_GATES:
        raise ValueError(f"unknown required gate: {require_gate}")
    required_pass = require_gate == "none" or _gate_passed_for_requirement(
        gates[require_gate]
    )
    controlled = (
        decision.get("decision")
        == "exact_rb_boundary_tangent_controller_controlled"
    )
    physical_training = (run_dir / "physical_training_started.json").is_file()
    controller_control, maximum_control_phases = _controller_scope(run_dir)
    decision = {
        **decision,
        "physical_training_performed": int(physical_training),
        "controller_control_trajectory_performed": int(controller_control),
        "maximum_control_trajectory_phase_count": maximum_control_phases,
    }
    workflow = {
        "schema": RUN_SCHEMA + "-workflow-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "gate": "workflow",
        "required_gate": require_gate,
        "required_gate_pass": int(required_pass),
        "stage_gates": {
            stage: {
                "evaluation_status": (
                    "not_evaluated"
                    if gate is None
                    else str(gate.get("evaluation_status", "invalid"))
                ),
                "passed": 0 if gate is None else int(gate.get("passed", 0)),
                "artifact": _GATE_FILENAMES[stage],
            }
            for stage, gate in gates.items()
        },
        "decision": decision,
        "maximum_control_trajectory_phase_count": maximum_control_phases,
        **claim_scope_flags(
            controlled=controlled,
            physical_training_performed=physical_training,
            controller_control_trajectory_performed=controller_control,
        ),
    }
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(run_dir / "controller_decision.json", decision)
    return workflow


def _verify_report(run_dir: Path, args: argparse.Namespace) -> None:
    """Revalidate committed evidence without opening an unexecuted stage."""

    gates = {
        stage: _optional_json(run_dir, filename)
        for stage, filename in _GATE_FILENAMES.items()
    }
    for stage, gate in gates.items():
        if gate is None:
            continue
        if (
            gate.get("schema") != _gate_schema(stage)
            or gate.get("schema_version") != 1
            or gate.get("gate") != stage
            or gate.get("evaluation_status")
            not in {"evaluated", "execution_failed", "not_evaluated"}
            or int(gate.get("passed", 0)) not in {0, 1}
        ):
            raise ArtifactCompatibilityError(f"{stage} gate envelope changed")

    # Passing stages have complete hash-sealed evidence and must be fully
    # revalidated.  Failed/incomplete stages remain readable but are never
    # reopened by report-only execution.
    if _gate_passed_for_requirement(gates["preflight"]):
        _preflight_stage(run_dir, args)
    if _gate_passed_for_requirement(gates["cache"]):
        _cache_stage(run_dir, args)
    if gates["train"] is not None and (run_dir / "checkpoint_selection.json").is_file():
        _verify_training_selection(run_dir)
    if (
        gates["confirm"] is not None
        and gates["confirm"].get("evaluation_status") == "evaluated"
        and (run_dir / "confirmation_index.json").is_file()
    ):
        _verify_confirmation_integrity(run_dir, args)
    if (
        gates["control"] is not None
        and gates["control"].get("evaluation_status") == "evaluated"
        and (run_dir / "controller_control_index.json").is_file()
    ):
        _verify_control_integrity(run_dir, args)


def _execution_failed_gate(stage: str, exc: BaseException) -> dict[str, Any]:
    failure_domain = str(getattr(exc, "failure_domain", "workflow_execution"))
    failure_code = str(
        getattr(exc, "failure_code", "boundary_tangent_execution_failed")
    )
    record: dict[str, Any] = {
        "schema": _gate_schema(stage),
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": "execution_failed",
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "passed": 0,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **claim_scope_flags(),
    }
    if stage == "preflight":
        provenance_failure = failure_domain == "provenance" or isinstance(
            exc, ArtifactCompatibilityError
        )
        record.update(
            {
                "provenance_valid": int(not provenance_failure),
                "failed_controller_adjudication_valid": int(
                    failure_domain not in {"adjudication", "parent_adjudication"}
                ),
                "boundary_tangent_representation_valid": int(
                    failure_domain not in {"representation", "operator"}
                ),
                "numerically_valid": 0,
                "resource_valid": 0,
            }
        )
    elif stage == "train":
        record.update(
            {
                "optimization_pipeline_valid": 0,
                "boundary_tangent_baseline_valid": int(
                    failure_domain != "baseline"
                ),
                "boundary_tangent_baseline_only": 0,
            }
        )
    elif stage == "confirm":
        record.update({"paired_risk_inference_valid": 0})
    elif stage == "control":
        record.update(
            {
                "controller_family_valid": 0,
                "numerically_valid": 0,
                "resource_valid": 0,
                "weak_law_controlled": 0,
                "microstep_refinement_controlled": 0,
            }
        )
    return record


def _commit_execution_failure(
    run_dir: Path,
    *,
    stage: str,
    exc: BaseException,
    require_gate: str,
) -> None:
    failure_domain = str(getattr(exc, "failure_domain", "workflow_execution"))
    failure_code = str(
        getattr(exc, "failure_code", "boundary_tangent_execution_failed")
    )
    gate_stage = stage if stage in _GATE_FILENAMES else "preflight"
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
        **claim_scope_flags(
            physical_training_performed=(
                run_dir / "physical_training_started.json"
            ).is_file(),
            controller_control_trajectory_performed=(
                run_dir / "controller_control_index.json"
            ).is_file(),
        ),
    }
    atomic_write_json(run_dir / f"{stage}_execution_failure.json", failure)
    gate_path = run_dir / _GATE_FILENAMES[gate_stage]
    if not gate_path.is_file():
        atomic_write_json(gate_path, _execution_failed_gate(gate_stage, exc))
    workflow = _workflow_record(run_dir, require_gate=require_gate)
    _status(
        run_dir,
        state="execution_failed",
        stage=stage,
        decision=str(workflow["decision"]["decision"]),
        message=str(exc),
        failure_domain=failure_domain,
        failure_code=failure_code,
    )
    _artifact_registry(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_controller_confirmation"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-boundary-tangent-rb-controller"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--require-gate", choices=REQUIRED_GATES, default="none"
    )
    parser.add_argument(
        "--parent-coarse-residual-run-dir", type=Path, required=True
    )
    parser.add_argument("--failed-controller-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
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
    args.runs_root = args.runs_root.resolve()
    args.parent_coarse_residual_run_dir = args.parent_coarse_residual_run_dir.resolve()
    args.failed_controller_run_dir = args.failed_controller_run_dir.resolve()
    if args.resume_run_dir is not None:
        args.resume_run_dir = args.resume_run_dir.resolve()

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
        or args.test_outer_steps % SHARD_STEPS != 0
    ):
        parser.error("--test-outer-steps must be a multiple of 8 in [16,512]")
    if not 0 <= args.test_maximum_updates <= int(TRAINING["maximum_updates"]):
        parser.error("--test-maximum-updates must be in [0,4000]")
    if not 8 <= args.test_bootstrap_replicates <= 50_000:
        parser.error("--test-bootstrap-replicates must be in [8,50000]")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    active_stage = "initialize"
    resumed = False
    initialized = False
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"boundary-tangent controller run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        active_stage = str(args.stage)
        if args.stage != "report":
            configure_exact_torch_backend(args.device)
        _status(run_dir, state="running", stage=active_stage)
        if args.stage == "report":
            _verify_report(run_dir, args)
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            _status(run_dir, state="running", stage=stage)
            if stage == "preflight":
                gate = _preflight_stage(run_dir, args)
            elif stage == "cache":
                if not _gate_passed_for_requirement(
                    _optional_json(run_dir, "preflight_gate.json")
                ):
                    raise ArtifactCompatibilityError(
                        "cache stage requires a passing preflight gate"
                    )
                gate = _cache_stage(run_dir, args)
            elif stage == "train":
                if not _gate_passed_for_requirement(
                    _optional_json(run_dir, "cache_gate.json")
                ):
                    raise ArtifactCompatibilityError(
                        "train stage requires a passing cache gate"
                    )
                gate = _train_stage(run_dir, args)
            elif stage == "confirm":
                if not _gate_passed_for_requirement(
                    _optional_json(run_dir, "train_gate.json")
                ):
                    raise ArtifactCompatibilityError(
                        "confirm stage requires a passing train gate"
                    )
                gate = _confirm_stage(run_dir, args)
            elif stage == "control":
                if not _gate_passed_for_requirement(
                    _optional_json(run_dir, "confirm_gate.json")
                ):
                    raise ArtifactCompatibilityError(
                        "control stage requires a passing confirm gate"
                    )
                gate = _control_stage(run_dir, args)
            else:  # pragma: no cover - parser and _stage_sequence prevent this
                raise AssertionError(stage)
            if not _gate_passed_for_requirement(gate):
                break

        active_stage = "report"
        workflow = _workflow_record(run_dir, require_gate=args.require_gate)
        decision = str(workflow["decision"]["decision"])
        required_pass = int(workflow["required_gate_pass"]) == 1
        _status(
            run_dir,
            state="complete" if required_pass else "gate_failed",
            stage=str(args.stage),
            decision=decision,
            failure_domain=None if required_pass else "scientific_gate",
            failure_code=(
                None if required_pass else f"{args.require_gate}_gate_failed"
            ),
        )
        _artifact_registry(run_dir)
        print(f"boundary-tangent decision: {decision}", flush=True)
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
            )
            _artifact_registry(run_dir)
        raise
    except Exception as exc:
        # Resume compatibility is verified before any mutation.  A rejected
        # resume must leave the existing directory byte-for-byte untouched.
        if run_dir is not None and (initialized or not resumed):
            _commit_execution_failure(
                run_dir,
                stage=active_stage,
                exc=exc,
                require_gate=args.require_gate,
            )
        import sys

        print(f"boundary-tangent controller error: {exc}", file=sys.stderr, flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
