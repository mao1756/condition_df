"""Exact Dynkin/Rao--Blackwell power controls for Jacobi Strang refinement.

This additive workflow keeps the certified Jacobi transition law and its
right-endpoint common-quantile coupling unchanged.  It observes each phase
through exact conditional moments and asks whether the resulting unbiased
Dynkin estimator can support the frozen refinement tolerances.  It never
imports a trainer or a reverse sampler.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence
import zipfile

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
from mnist.d0_jacobi_denoising import build_four_color_matchings
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist import d0_jacobi_rb_spectral as _spectral
from mnist.d0_jacobi_rb_dynkin import (
    DynkinAccumulatorState,
    compute_dynkin_phase_drift,
    run_dynkin_refinement_shard,
    run_dynkin_tower_phase,
)
from mnist.d0_jacobi_rb_dynkin_path_ids import (
    DYNKIN_PATH_ID_PLAN_VERSION,
    PACKED_TRANSITION_ID_LIMIT,
    DynkinPathIDPlan,
    canonical_tower_transition_ids,
)
from mnist.d0_jacobi_rb_dynkin_power_gate import (
    DynkinPowerThresholds,
    build_dynkin_candidate_records,
    confirm_dynkin_design,
    decide_dynkin_power_workflow,
    evaluate_dynkin_power,
    evaluate_dynkin_preflight,
    normal_chi_square_bonferroni_projection,
    not_evaluated_gate,
    select_dynkin_panel_a_design,
)
from mnist.d0_jacobi_rb_dynkin_power_provenance import (
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_FINGERPRINT,
    verify_raw_endpoint_power_infeasible_parent,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    GRID_SPACING,
    MAX_REFINEMENT_PATHS_PER_GROUP,
    PATH_STATE_SIZE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    REFINEMENT_ID_VERSION,
    REFINEMENT_RNG_VERSION,
    REFINEMENT_SCHEDULER_VERSION,
    REFINEMENT_SHARD_STEPS,
    SUPPORTED_SAMPLE_STEPS,
    TAU_EFF,
    canonical_refinement_transition_ids,
    finest_tick_for_step,
    legacy_k512_transition_ids,
    evaluate_refinement_observables,
    refinement_observable_spec,
)
from mnist.d0_jacobi_rb_strang_refinement_gate import whole_path_max_t_intervals
from mnist.diag_d0_jacobi_rb_strang_refinement import (
    EXPECTED_IMAGE_SHA256,
    EXPECTED_MIXED_TARGET_SHA256,
    OBSERVATION_TIME_FRACTIONS,
    _aggregate_execution,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-dynkin-power-confirmation"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "exact Dynkin weak-observable power feasibility only"
ROOT_SEED = 261_161
PANEL_NAMES = ("a", "b")
PILOT_LEVELS = (128, 256, 512, 1024, 2048)
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}
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


class DynkinSchedulerConfigurationError(ValueError):
    """Raised when the additive Dynkin scheduler plan is internally invalid."""


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            **{key: np.ascontiguousarray(value) for key, value in arrays.items()},
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _passed(gate: Mapping[str, Any]) -> bool:
    return (
        gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _substantive_test_gate_passed(
    gate: Mapping[str, Any], *, ignored_checks: Sequence[str]
) -> bool:
    """Allow reduced integration work to continue without authorizing it."""

    if gate.get("evaluation_status") != "evaluated":
        return False
    subchecks = gate.get("subchecks")
    if not isinstance(subchecks, Mapping):
        return False
    ignored = set(ignored_checks)
    return bool(subchecks) and all(
        name in ignored
        or (
            isinstance(record, Mapping)
            and int(record.get("passed", 0)) == 1
        )
        for name, record in subchecks.items()
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    thresholds = DynkinPowerThresholds()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "pilot", "report", "all"), default="all"
    )
    parser.add_argument(
        "--require-gate", choices=("none", "preflight", "pilot"), default="none"
    )
    parser.add_argument("--parent-strang-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation"),
    )
    parser.add_argument("--run-name", default="production-dynkin-strang-power")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    parser.add_argument(
        "--sample-steps", type=_parse_int_tuple, default=PILOT_LEVELS
    )
    parser.add_argument(
        "--tower-panel-clusters",
        type=int,
        default=thresholds.tower_clusters_per_panel,
    )
    parser.add_argument(
        "--pilot-panel-paths",
        type=int,
        default=thresholds.pilot_paths_per_panel,
    )
    parser.add_argument(
        "--pilot-stop-steps",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--test-only-reduced-workload", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.stage in {"pilot", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "pilot": {"none", "preflight", "pilot"},
        "report": {"none", "preflight", "pilot"},
        "all": {"none", "preflight", "pilot"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(
            f"--require-gate {args.require_gate} is unavailable at stage {args.stage}"
        )
    if int(args.tower_panel_clusters) <= 0 or int(args.pilot_panel_paths) <= 0:
        parser.error("panel sizes must be positive")
    if int(args.tower_panel_clusters) > 128:
        parser.error("tower panel size cannot exceed the frozen 128-path slot")
    if int(args.pilot_panel_paths) > 8:
        parser.error("pilot panel size cannot exceed the frozen eight-path slot")
    if tuple(args.sample_steps) != PILOT_LEVELS and not args.test_only_reduced_workload:
        parser.error("production sample steps are frozen")
    frozen = {
        "root_seed": ROOT_SEED,
        "tower_panel_clusters": thresholds.tower_clusters_per_panel,
        "pilot_panel_paths": thresholds.pilot_paths_per_panel,
    }
    changed = [name for name, value in frozen.items() if getattr(args, name) != value]
    if (changed or args.pilot_stop_steps is not None) and not args.test_only_reduced_workload:
        parser.error(
            "production configuration is frozen; overrides require "
            "--test-only-reduced-workload: " + ", ".join(changed or ["pilot_stop_steps"])
        )
    if args.test_only_reduced_workload and args.require_gate != "none":
        parser.error("test-only workloads cannot satisfy a required gate")
    if torch.device(args.device).type != "cuda" and not args.test_only_reduced_workload:
        parser.error("production Dynkin controls require --device cuda")
    if any(level not in SUPPORTED_SAMPLE_STEPS for level in args.sample_steps):
        parser.error("unsupported refinement level")
    if args.pilot_stop_steps is not None:
        stop = int(args.pilot_stop_steps)
        if stop <= 0 or stop % REFINEMENT_SHARD_STEPS:
            parser.error("pilot stop steps must be a positive multiple of eight")
    return args


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        root = Path(args.resume_run_dir).resolve()
        if not root.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {root}")
        return root, True
    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.run_name}"
    run.mkdir(parents=False, exist_ok=False)
    return run.resolve(), False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    record = _load(path) if path.is_file() else {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": _now(),
    }
    record.update(updates)
    record.update({"updated_at": _now(), **NO_WORK})
    atomic_write_json(path, record)
    return record


def _freeze(
    path: Path, value: Mapping[str, Any], *, require_existing: bool = False
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _load(path) != normalized:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    elif require_existing:
        raise ArtifactCompatibilityError(f"resume run lacks frozen artifact: {path.name}")
    else:
        atomic_write_json(path, normalized)
    return normalized


def _build_path_id_plan(args: argparse.Namespace) -> DynkinPathIDPlan:
    try:
        return DynkinPathIDPlan(
            tower_path_count=int(args.tower_panel_clusters),
            pilot_path_count=int(args.pilot_panel_paths),
        )
    except (TypeError, ValueError) as exc:
        raise DynkinSchedulerConfigurationError(
            f"invalid Dynkin path-ID plan: {exc}"
        ) from exc


def _load_frozen_path_id_plan(run_dir: Path) -> DynkinPathIDPlan:
    path = run_dir / "path_id_plan.json"
    if not path.is_file():
        raise ArtifactCompatibilityError("run lacks path_id_plan.json")
    try:
        return DynkinPathIDPlan.from_frozen_record(_load(path))
    except (TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            f"frozen Dynkin path-ID plan is invalid: {exc}"
        ) from exc


def _freeze_path_id_plan(
    run_dir: Path,
    plan: DynkinPathIDPlan,
    *,
    require_existing: bool,
) -> dict[str, Any]:
    record = _freeze(
        run_dir / "path_id_plan.json",
        plan.to_frozen_record(),
        require_existing=require_existing,
    )
    loaded = _load_frozen_path_id_plan(run_dir)
    if loaded != plan or record.get("path_id_plan_sha256") != plan.sha256:
        raise ArtifactCompatibilityError("frozen Dynkin path-ID plan changed")
    return record


def _scientific_config(
    args: argparse.Namespace, path_id_plan: DynkinPathIDPlan
) -> dict[str, Any]:
    thresholds = DynkinPowerThresholds()
    return {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "claim_scope": CLAIM_SCOPE,
        "root_seed": int(args.root_seed),
        "sample_steps": list(args.sample_steps),
        "observation_time_fractions": list(OBSERVATION_TIME_FRACTIONS),
        "tower_panel_clusters": int(args.tower_panel_clusters),
        "pilot_panel_paths": int(args.pilot_panel_paths),
        "pilot_panel_names": list(PANEL_NAMES),
        "candidate_main_paths": list(thresholds.candidate_main_paths),
        "candidate_reference_paths": list(thresholds.candidate_reference_paths),
        "maximum_main_half_width": thresholds.maximum_main_half_width,
        "maximum_reference_half_width": thresholds.maximum_reference_half_width,
        "maximum_projected_hours": thresholds.maximum_projected_hours,
        "minimum_rate": thresholds.minimum_rate,
        "steps_per_shard": REFINEMENT_SHARD_STEPS,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "scheduler_version": REFINEMENT_SCHEDULER_VERSION,
        "transition_id_version": REFINEMENT_ID_VERSION,
        "rng_version": REFINEMENT_RNG_VERSION,
        "path_id_plan_version": path_id_plan.version,
        "path_id_plan_sha256": path_id_plan.sha256,
        "test_only_reduced_workload": int(args.test_only_reduced_workload),
        "pilot_stop_steps": args.pilot_stop_steps,
        **NO_WORK,
    }


def _verify_resume_contract(
    run_dir: Path,
    *,
    expected_plan: DynkinPathIDPlan,
    expected_config: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
    expected_source: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    """Reject legacy or mismatched resumes before replacing any artifact."""

    manifest_path = run_dir / "run_manifest.json"
    config_path = run_dir / "scientific_config.json"
    provenance_path = run_dir / "parent_provenance.json"
    status_path = run_dir / "run_status.json"
    if not all(
        path.is_file()
        for path in (
            manifest_path,
            config_path,
            provenance_path,
            status_path,
            run_dir / "sealed_panel_registry.json",
            run_dir / "panel_a_plan.json",
            run_dir / "panel_b_plan.json",
        )
    ):
        raise ArtifactCompatibilityError(
            "resume lacks the corrected frozen setup artifacts"
        )
    manifest = _load(manifest_path)
    config = _load(config_path)
    provenance = _load(provenance_path)
    status = _load(status_path)
    frozen_plan = _load_frozen_path_id_plan(run_dir)
    normalized_config = json.loads(
        json.dumps(dict(expected_config), sort_keys=True, allow_nan=False)
    )
    normalized_manifest = json.loads(
        json.dumps(dict(expected_manifest), sort_keys=True, allow_nan=False)
    )
    normalized_provenance = json.loads(
        json.dumps(dict(expected_provenance), sort_keys=True, allow_nan=False)
    )
    if (
        frozen_plan != expected_plan
        or config != normalized_config
        or manifest != normalized_manifest
        or provenance != normalized_provenance
        or manifest.get("path_id_plan_file_sha256")
        != file_fingerprint(run_dir / "path_id_plan.json")
        or manifest.get("scientific_config_sha256")
        != config_fingerprint(normalized_config)
        or status.get("scientific_config_sha256")
        != config_fingerprint(normalized_config)
        or status.get("source_fingerprint")
        != manifest.get("source_fingerprint")
    ):
        raise ArtifactCompatibilityError(
            "resume is incompatible with the corrected Dynkin path-ID plan"
        )
    for panel in PANEL_NAMES:
        if _load(run_dir / f"panel_{panel}_plan.json") != _panel_plan(
            args, panel, expected_plan
        ):
            raise ArtifactCompatibilityError(
                f"resume panel {panel} plan changed"
            )
    sealed = _load(run_dir / "sealed_panel_registry.json")
    if (
        sealed.get("path_id_plan_version") != expected_plan.version
        or sealed.get("path_id_plan_sha256") != expected_plan.sha256
        or sealed.get("path_id_plan_file_sha256")
        != file_fingerprint(run_dir / "path_id_plan.json")
        or any(
            sealed.get("panels", {}).get(panel, {}).get("plan_sha256")
            != file_fingerprint(run_dir / f"panel_{panel}_plan.json")
            for panel in PANEL_NAMES
        )
    ):
        raise ArtifactCompatibilityError("resume sealed panel registry changed")
    frozen_source = _load_frozen_source(run_dir)
    if (
        not np.array_equal(
            np.asarray(frozen_source["image"], dtype=np.float64),
            np.asarray(expected_source["image"], dtype=np.float64),
        )
        or not np.array_equal(
            np.asarray(frozen_source["mixed_target"], dtype=np.float64),
            np.asarray(expected_source["mixed_target"], dtype=np.float64),
        )
        or frozen_source.get("image_sha256")
        != expected_source.get("image_sha256")
        or frozen_source.get("mixed_target_sha256")
        != expected_source.get("mixed_target_sha256")
    ):
        raise ArtifactCompatibilityError("resume source-image binding changed")


def _source_record(parent_dir: Path) -> tuple[str, list[str]]:
    manifest = _load(parent_dir / "run_manifest.json")
    raw = manifest.get("source_paths")
    if not isinstance(raw, list) or len(raw) != 15:
        raise ArtifactCompatibilityError("parent manifest does not bind fifteen sources")
    paths = {Path(str(value)).resolve() for value in raw}
    for module_name in (
        "mnist.d0_jacobi_rb_dynkin",
        "mnist.d0_jacobi_rb_dynkin_cuda",
        "mnist.d0_jacobi_rb_dynkin_path_ids",
        "mnist.d0_jacobi_rb_dynkin_power_gate",
        "mnist.d0_jacobi_rb_dynkin_power_provenance",
    ):
        module = importlib.import_module(module_name)
        paths.add(Path(module.__file__).resolve())
    paths.add(Path(__file__).resolve())
    if not all(path.is_file() for path in paths):
        raise ArtifactCompatibilityError("source set contains a missing file")
    ordered = sorted(paths)
    return source_fingerprint(ordered), [str(path) for path in ordered]


def _load_source_image(parent_dir: Path) -> dict[str, Any]:
    metadata = _load(parent_dir / "source_image.json")
    npz_path = parent_dir / "source_image.npz"
    if not npz_path.is_file():
        raise ArtifactCompatibilityError("parent source-image payload is missing")
    with np.load(npz_path, allow_pickle=False) as archive:
        image = np.ascontiguousarray(archive["image"], dtype=np.float64).reshape(-1)
        mixed = np.ascontiguousarray(archive["mixed_target"], dtype=np.float64).reshape(-1)
    if (
        image.shape != (PATH_STATE_SIZE,)
        or mixed.shape != (PATH_STATE_SIZE,)
        or not np.isfinite(image).all()
        or not np.isfinite(mixed).all()
        or metadata.get("image_sha256") != EXPECTED_IMAGE_SHA256
        or metadata.get("mixed_target_sha256") != EXPECTED_MIXED_TARGET_SHA256
        or abs(float(mixed.sum()) - 1.0) > 1.0e-12
    ):
        raise ArtifactCompatibilityError("parent source image binding changed")
    return {
        **metadata,
        "image": image,
        "mixed_target": mixed,
        "source_npz_sha256": file_fingerprint(npz_path),
    }


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "records": records,
        **NO_WORK,
    }


def _verify_terminal_registry(run_dir: Path, *, stage: str) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    registry = _load(path)
    status = _load(run_dir / "run_status.json")
    if status.get("artifact_registry_sha256") != file_fingerprint(path):
        raise ArtifactCompatibilityError("resume status does not bind its registry")
    mutable = {
        "dynkin_preflight_gate.json",
        "dynkin_power_gate.json",
        "dynkin_workflow_gate.json",
        "dynkin_power_decision.json",
    }
    interrupted = status.get("status") == "running"
    recoverable = (
        ("dynkin_shards/pilot/",)
        if interrupted and stage in {"pilot", "all"}
        else ()
    )
    records = dict(registry.get("records", {}))
    for relative, raw in records.items():
        artifact = run_dir / relative
        valid = (
            artifact.is_file()
            and isinstance(raw, Mapping)
            and raw.get("sha256") == file_fingerprint(artifact)
            and raw.get("size") == artifact.stat().st_size
        )
        if not valid and (
            relative.startswith(recoverable)
            or (interrupted and relative in mutable)
        ):
            continue
        if not valid:
            raise ArtifactCompatibilityError(f"resume artifact changed: {relative}")
    if not interrupted:
        excluded = {"artifact_registry.json", "run_status.json"}
        actual = {
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file()
            and path.relative_to(run_dir).as_posix() not in excluded
        }
        unexpected = sorted(actual - set(records))
        if unexpected:
            raise ArtifactCompatibilityError(
                "completed resume contains unregistered artifacts: "
                + ", ".join(unexpected[:5])
            )


def _existing_gate(run_dir: Path, name: str, reason: str) -> dict[str, Any]:
    path = run_dir / f"dynkin_{name}_gate.json"
    return _load(path) if path.is_file() else not_evaluated_gate(name, reason)


def _synthetic_gate(name: str, *, passed: bool) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-test-gate",
        "schema_version": 1,
        "gate": name,
        "evaluation_status": "evaluated",
        "passed": int(passed),
        "subchecks": {
            "synthetic": {
                "value": int(passed),
                "operator": "==",
                "threshold": 1,
                "passed": int(passed),
            }
        },
        **NO_WORK,
    }


def _progress(
    args: argparse.Namespace, label: str, done: int, total: int, started: float
) -> None:
    if args.no_progress or total <= 0:
        return
    elapsed = max(0.0, time.perf_counter() - started)
    eta = elapsed * max(0, total - done) / max(done, 1)
    print(f"Jacobi Dynkin {label} {done}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s")


def _distribution_free_power_impossibility() -> dict[str, Any]:
    standardized_cubic_range = 916_815.0
    tolerance = 0.005
    family_failure = 0.01 / 40.0
    required = math.ceil(
        (standardized_cubic_range**2)
        * math.log(2.0 / family_failure)
        / (2.0 * tolerance**2)
    )
    # Keep the documented order-of-magnitude result frozen even if the
    # conservative support constants are refined later.
    documented_upper = 2.4e18
    return {
        "schema": RUN_SCHEMA + "-distribution-free-power-impossibility",
        "schema_version": 1,
        "authorizing": 0,
        "method": "bounded-support Hoeffding union bound",
        "standardized_cubic_support_width": standardized_cubic_range,
        "target_half_width": tolerance,
        "family_failure_probability": family_failure,
        "calculated_required_paths": required,
        "documented_required_paths_upper_order": documented_upper,
        "conclusion": (
            "distribution-free certification is infeasible at the frozen "
            "tolerances; the sealed pilot is an engineering forecast only"
        ),
        **NO_WORK,
    }


def _failed_stage_gate(
    run_dir: Path,
    stage: str,
    exc: BaseException,
    *,
    failure_domain: str = "unknown_execution",
    failure_code: str = "stage_execution_exception",
) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA + "-stage-failure",
        "schema_version": 1,
        "stage": stage,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **NO_WORK,
    }
    atomic_write_json(run_dir / f"{stage}_failure.json", record)
    gate = {
        "schema": RUN_SCHEMA + "-stage-execution-gate",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": "execution_failed",
        "passed": 0,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "failed_stage": stage,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "failure": record,
        "subchecks": {
            "stage_execution_pass": {
                "value": 0,
                "operator": "==",
                "threshold": 1,
                "passed": 0,
            }
        },
        **NO_WORK,
    }
    atomic_write_json(run_dir / f"dynkin_{stage}_gate.json", gate)
    return gate


def _required_gate_pass(
    required: str, preflight: Mapping[str, Any], pilot: Mapping[str, Any]
) -> bool:
    if required == "none":
        return True
    if required == "preflight":
        return _passed(preflight)
    return _passed(preflight) and _passed(pilot)


def _finish(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    preflight: Mapping[str, Any],
    pilot: Mapping[str, Any],
    provenance_pass: bool = True,
) -> int:
    workflow = decide_dynkin_power_workflow(
        provenance=provenance_pass,
        preflight_gate=preflight,
        pilot_gate=pilot,
    )
    decision = dict(workflow)
    atomic_write_json(run_dir / "dynkin_workflow_gate.json", workflow)
    atomic_write_json(run_dir / "dynkin_power_decision.json", decision)
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    required_pass = _required_gate_pass(
        args.require_gate, preflight=preflight, pilot=pilot
    )
    _write_status(
        run_dir,
        status="complete",
        outcome=("complete" if required_pass else "gate_failed"),
        phase=args.stage,
        required_gate=args.require_gate,
        required_gate_pass=int(required_pass),
        decision=decision.get("decision", workflow.get("decision")),
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_size=registry_path.stat().st_size,
        artifact_registry_record_count=len(registry["records"]),
    )
    return 0 if required_pass else 1


def _load_frozen_source(run_dir: Path) -> dict[str, Any]:
    metadata = _load(run_dir / "source_image.json")
    payload = run_dir / "source_image.npz"
    if not payload.is_file():
        raise ArtifactCompatibilityError("run source-image payload is missing")
    with np.load(payload, allow_pickle=False) as archive:
        image = np.ascontiguousarray(archive["image"], dtype=np.float64)
        mixed = np.ascontiguousarray(archive["mixed_target"], dtype=np.float64)
    if (
        metadata.get("source_npz_sha256") != file_fingerprint(payload)
        or metadata.get("image_sha256") != EXPECTED_IMAGE_SHA256
        or metadata.get("mixed_target_sha256") != EXPECTED_MIXED_TARGET_SHA256
    ):
        raise ArtifactCompatibilityError("frozen run source-image binding changed")
    return {**metadata, "image": image, "mixed_target": mixed}


def _aggregate_level_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        rows.extend(dict(row) for row in result.get("rows", ()))
    diagnostics = [
        dict(row.get("diagnostics", {}))
        for row in rows
        if isinstance(row.get("diagnostics"), Mapping)
    ]
    transition_count = sum(
        int(row.get("transition_count", 0)) for row in diagnostics
    )
    certified_count = sum(int(row.get("certified_count", 0)) for row in diagnostics)
    fallback_count = sum(int(row.get("fallback_count", 0)) for row in diagnostics)
    complete_wall = sum(
        float(row.get("complete_wall_upper_seconds", math.inf)) for row in rows
    )
    return {
        "level_count": len(results),
        "shard_count": len(rows),
        "transition_count": transition_count,
        "certified_count": certified_count,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "fallback_count": fallback_count,
        "fallback_fraction": (
            fallback_count / transition_count if transition_count else 0.0
        ),
        "fallback_elapsed_seconds": sum(
            float(row.get("fallback_elapsed_seconds", 0.0)) for row in diagnostics
        ),
        "elapsed_seconds": sum(
            float(row.get("elapsed_seconds", 0.0)) for row in diagnostics
        ),
        "complete_wall_upper_seconds": complete_wall,
        "conservative_rate": (
            transition_count / complete_wall
            if transition_count > 0 and math.isfinite(complete_wall) and complete_wall > 0
            else 0.0
        ),
        "mass_error": max(
            (
                float(row.get("maximum_global_simplex_error", math.inf))
                for row in diagnostics
            ),
            default=math.inf,
        ),
        "peak_memory_fraction": max(
            (float(result.get("peak_memory_fraction", 0.0)) for result in results),
            default=0.0,
        ),
        "shard_chain_pass": int(
            bool(results) and all(int(result.get("shard_chain_pass", 0)) == 1 for result in results)
        ),
        "state_updates_device_resident_pass": int(
            bool(results)
            and all(
                int(result.get("state_updates_device_resident_pass", 0)) == 1
                for result in results
            )
        ),
        **{
            name: sum(int(row.get(name, 0)) for row in diagnostics)
            for name in _FORBIDDEN_COUNTS
        },
    }


def _panel_plan(
    args: argparse.Namespace,
    panel: str,
    path_id_plan: DynkinPathIDPlan | None = None,
) -> dict[str, Any]:
    if panel not in PANEL_NAMES:
        raise ValueError("panel must be a or b")
    plan = path_id_plan or _build_path_id_plan(args)
    paths = plan.pilot_path_ids(panel)
    plan.validate_role_path_ids(f"pilot_{panel}", paths)
    production = set(plan.designated_production_path_ids)
    return {
        "schema": RUN_SCHEMA + "-sealed-panel-plan",
        "schema_version": 1,
        "panel": panel,
        "root_seed": int(args.root_seed),
        "rng_namespace": f"dynkin-power-panel-{panel}",
        "path_ids": list(paths),
        "path_count": len(paths),
        "sample_steps": list(args.sample_steps),
        "sealed_before_execution": 1,
        "regeneration_permitted": 0,
        "future_production_namespace_disjoint": int(
            set(paths).isdisjoint(production)
        ),
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        **NO_WORK,
    }


def _freeze_panel_plans(
    run_dir: Path,
    args: argparse.Namespace,
    path_id_plan: DynkinPathIDPlan | None = None,
) -> dict[str, Any]:
    plan = path_id_plan or _build_path_id_plan(args)
    plans = {
        panel: _freeze(
            run_dir / f"panel_{panel}_plan.json",
            _panel_plan(args, panel, plan),
        )
        for panel in PANEL_NAMES
    }
    a = set(plans["a"]["path_ids"])
    b = set(plans["b"]["path_ids"])
    production = set(plan.designated_production_path_ids)
    if a & b or a & production or b & production:
        raise ArtifactCompatibilityError("sealed panel path IDs overlap")
    registry = {
        "schema": RUN_SCHEMA + "-sealed-panel-registry",
        "schema_version": 1,
        "path_id_plan_name": "path_id_plan.json",
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "path_id_plan_file_sha256": (
            file_fingerprint(run_dir / "path_id_plan.json")
            if (run_dir / "path_id_plan.json").is_file()
            else None
        ),
        "panels": {
            panel: {
                "plan_name": f"panel_{panel}_plan.json",
                "plan_sha256": file_fingerprint(run_dir / f"panel_{panel}_plan.json"),
                "path_ids_sha256": _array_sha256(
                    np.asarray(plans[panel]["path_ids"], dtype=np.int64)
                ),
            }
            for panel in PANEL_NAMES
        },
        "panels_disjoint": 1,
        "future_production_namespace_disjoint": 1,
        "sealed_before_panel_a": 1,
        **NO_WORK,
    }
    return _freeze(run_dir / "sealed_panel_registry.json", registry)


def _path_id_plan_preflight(
    run_dir: Path,
    args: argparse.Namespace,
    plan: DynkinPathIDPlan,
) -> dict[str, Any]:
    path = run_dir / "path_id_plan_preflight.json"
    if path.is_file():
        record = _load(path)
        try:
            frozen = _load_frozen_path_id_plan(run_dir)
        except ArtifactCompatibilityError:
            raise
        if (
            frozen != plan
            or record.get("path_id_plan_version") != plan.version
            or record.get("path_id_plan_sha256") != plan.sha256
            or record.get("path_id_plan_file_sha256")
            != file_fingerprint(run_dir / "path_id_plan.json")
            or record.get("passed") != 1
            or not isinstance(record.get("checks"), Mapping)
            or not all(
                value == 1
                for value in dict(record["checks"]).values()
            )
        ):
            raise ArtifactCompatibilityError(
                "frozen path-ID preflight binding changed"
            )
        return record

    try:
        plan.validate()
        role_sets = [
            set(plan.legacy_replay_path_ids),
            set(plan.tower_panel_path_ids("a")),
            set(plan.tower_panel_path_ids("b")),
            set(plan.pilot_path_ids("a")),
            set(plan.pilot_path_ids("b")),
            set(plan.designated_production_path_ids),
        ]
        role_disjoint = all(
            left.isdisjoint(right)
            for index, left in enumerate(role_sets)
            for right in role_sets[index + 1 :]
        )
        maximum_packed = 0
        smoke_hashes: dict[str, str] = {}
        for panel in PANEL_NAMES:
            for matching_index in range(4):
                for duration_index in range(2):
                    case_index = 2 * matching_index + duration_index
                    paths = plan.tower_case_path_ids(panel, case_index)
                    ids = canonical_tower_transition_ids(
                        paths,
                        sample_steps=512,
                        outer_step=case_index,
                        phase=matching_index,
                        device=torch.device("cpu"),
                    )
                    maximum_packed = max(
                        maximum_packed, int(ids.to(torch.int64).max().item())
                    )
                    smoke_hashes[f"tower_{panel}_case_{case_index}"] = (
                        _array_sha256(ids.to(torch.int64).numpy())
                    )

        paths = plan.tower_case_path_ids("a", 0)
        tower_ids = canonical_tower_transition_ids(
            paths,
            sample_steps=512,
            outer_step=0,
            phase=0,
            device=torch.device("cpu"),
        )
        independently_chunked = torch.cat(
            [
                canonical_refinement_transition_ids(
                    paths[start : start + MAX_REFINEMENT_PATHS_PER_GROUP],
                    sample_steps=512,
                    outer_step=0,
                    phase=0,
                    device=torch.device("cpu"),
                )
                for start in range(
                    0, len(paths), MAX_REFINEMENT_PATHS_PER_GROUP
                )
            ]
        ).contiguous()
        chunking_pass = torch.equal(tower_ids, independently_chunked)

        coupling_path = plan.pilot_path_ids("a")[:1]
        coarse = canonical_refinement_transition_ids(
            coupling_path,
            sample_steps=128,
            outer_step=0,
            phase=0,
            device=torch.device("cpu"),
        )
        aligned = canonical_refinement_transition_ids(
            coupling_path,
            sample_steps=2048,
            outer_step=15,
            phase=0,
            device=torch.device("cpu"),
        )
        unaligned = canonical_refinement_transition_ids(
            coupling_path,
            sample_steps=2048,
            outer_step=14,
            phase=0,
            device=torch.device("cpu"),
        )
        coupling_pass = (
            finest_tick_for_step(128, 0) == finest_tick_for_step(2048, 15)
            and torch.equal(coarse, aligned)
            and not torch.equal(coarse, unaligned)
        )
        packed_owners: dict[int, list[tuple[int, int]]] = {}
        within_level_unique = True
        for level in SUPPORTED_SAMPLE_STEPS:
            level_identifiers: list[int] = []
            for step in range(level):
                tick = finest_tick_for_step(level, step)
                identifier = int(
                    canonical_refinement_transition_ids(
                        coupling_path,
                        sample_steps=level,
                        outer_step=step,
                        phase=0,
                        device=torch.device("cpu"),
                    )[0]
                )
                level_identifiers.append(identifier)
                packed_owners.setdefault(identifier, []).append((level, tick))
            within_level_unique &= len(set(level_identifiers)) == level
        intentional_aliases_only = (
            within_level_unique
            and any(len(owners) > 1 for owners in packed_owners.values())
            and all(
                len({tick for _, tick in owners}) == 1
                for owners in packed_owners.values()
            )
        )
        checks = {
            "plan_frozen_pass": int((run_dir / "path_id_plan.json").is_file()),
            "plan_hash_pass": int(
                _load_frozen_path_id_plan(run_dir).sha256 == plan.sha256
            ),
            "path_id_20_bit_pass": 1,
            "role_disjoint_pass": int(role_disjoint),
            "canonical_id_smoke_pass": int(len(smoke_hashes) == 16),
            "packed_id_43_bit_pass": int(
                maximum_packed < PACKED_TRANSITION_ID_LIMIT
            ),
            "tower_chunking_pass": int(chunking_pass),
            "path_major_order_pass": int(chunking_pass),
            "right_endpoint_alias_pass": int(coupling_pass),
            "cross_level_alias_plan_pass": int(intentional_aliases_only),
            "future_production_reserved_pass": int(
                len(plan.designated_production_path_ids) == 64
            ),
        }
    except (TypeError, ValueError) as exc:
        raise DynkinSchedulerConfigurationError(
            f"Dynkin path-ID preflight failed: {exc}"
        ) from exc
    record = {
        "schema": RUN_SCHEMA + "-path-id-plan-preflight",
        "schema_version": 1,
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "path_id_plan_file_sha256": file_fingerprint(
            run_dir / "path_id_plan.json"
        ),
        "checks": checks,
        "passed": int(all(checks.values())),
        "maximum_packed_transition_id": maximum_packed,
        "packed_transition_id_limit": PACKED_TRANSITION_ID_LIMIT,
        "tower_transition_id_hashes": smoke_hashes,
        **NO_WORK,
    }
    atomic_write_json(path, record)
    return record


def _projection_feature_matrices(
    level_results: Mapping[int, Mapping[str, Any]],
    *,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    by_level: dict[int, dict[int, np.ndarray]] = {}
    for level, result in level_results.items():
        raw = result.get(key)
        if not isinstance(raw, Mapping):
            raise ArtifactCompatibilityError(f"level K{level} lacks {key}")
        by_level[int(level)] = {
            int(step): np.asarray(values, dtype=np.float64)
            for step, values in raw.items()
        }
    fractions = OBSERVATION_TIME_FRACTIONS
    main_blocks: list[np.ndarray] = []
    for lower, upper in ((128, 256), (256, 512), (512, 1024)):
        for fraction in fractions:
            low_step = int(round(lower * fraction))
            high_step = int(round(upper * fraction))
            low = by_level[lower][low_step]
            high = by_level[upper][high_step]
            count = min(low.shape[0], high.shape[0])
            main_blocks.append(low[:count] - high[:count])
    reference_blocks: list[np.ndarray] = []
    for fraction in fractions:
        v512 = by_level[512][int(round(512 * fraction))]
        v1024 = by_level[1024][int(round(1024 * fraction))]
        v2048 = by_level[2048][int(round(2048 * fraction))]
        count = min(v512.shape[0], v1024.shape[0], v2048.shape[0])
        reference_blocks.append(
            v512[:count] - (4.0 * v2048[:count] - v1024[:count]) / 3.0
        )
    main = np.concatenate(main_blocks, axis=1)
    reference = np.concatenate(reference_blocks, axis=1)
    if (
        main.ndim != 2
        or reference.ndim != 2
        or not np.isfinite(main).all()
        or not np.isfinite(reference).all()
    ):
        raise ArtifactCompatibilityError("pilot feature matrix is invalid")
    return np.ascontiguousarray(main), np.ascontiguousarray(reference)


def _parent_projected_hours(
    parent_dir: Path, conservative_rate: float
) -> dict[tuple[int, int], float]:
    # The immutable parent's complete-wall projection already includes shard
    # I/O.  Scale it only by the newly measured conservative sustained rate.
    if not math.isfinite(conservative_rate) or conservative_rate <= 0.0:
        return {}
    import csv

    rows: list[dict[str, str]]
    with (parent_dir / "refinement_design_candidates.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    parent_rate = 2725.4689340542554
    return {
        (int(row["main_paths"]), int(row["reference_paths"])): (
            float(row["projected_hours"]) * parent_rate / conservative_rate
        )
        for row in rows
    }


def _independent_phase_formula(
    pair_total: np.ndarray,
    head_fraction: np.ndarray,
    exposure: np.ndarray,
    *,
    tail_index: np.ndarray,
    head_index: np.ndarray,
    weights: np.ndarray,
    arb: bool,
) -> np.ndarray:
    """Independent Legendre-eigenmoment evaluation of one phase."""

    if arb:
        midpoint, _, _ = _independent_phase_formula_arb(
            pair_total,
            head_fraction,
            exposure,
            tail_index=tail_index,
            head_index=head_index,
            weights=weights,
        )
        return midpoint
    r = np.asarray(pair_total, dtype=np.float64)
    x = np.asarray(head_fraction, dtype=np.float64)
    u = np.asarray(exposure, dtype=np.float64)
    z = 2.0 * x - 1.0
    p2 = (3.0 * z * z - 1.0) / 2.0
    decay2 = np.expm1(-2.0 * u)
    decay6 = np.expm1(-6.0 * u)
    delta_mean = 0.5 * r * z * decay2
    linear = np.sum(
        (weights[:, head_index] - weights[:, tail_index])[None, :, :]
        * delta_mean[:, None, :],
        axis=-1,
    )
    quadratic = np.sum(r * r * p2 * decay6 / 3.0, axis=-1)[:, None]
    cubic = np.sum(r**3 * p2 * decay6 / 2.0, axis=-1)[:, None]
    return np.concatenate((linear, quadratic, cubic), axis=1)


def _independent_phase_formula_arb(
    pair_total: np.ndarray,
    head_fraction: np.ndarray,
    exposure: np.ndarray,
    *,
    tail_index: np.ndarray,
    head_index: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the complete represented-input phase formula in 256-bit Arb.

    This oracle deliberately reconstructs *every* algebraic operation in Arb:
    pair totals, fractions, Legendre polynomial, Fourier-weight differences,
    exponential factors, products, and reductions.  Using NumPy for the
    polynomial and only Arb for ``expm1`` would share precisely the rounding
    error that the authorizing enclosure is intended to detect near a root of
    :math:`P_2`.
    """

    if _spectral._arb is None or _spectral._flint_ctx is None:
        raise RuntimeError("python-flint/Arb is required for the Dynkin oracle")
    r = np.ascontiguousarray(pair_total, dtype=np.float64)
    x = np.ascontiguousarray(head_fraction, dtype=np.float64)
    u = np.ascontiguousarray(exposure, dtype=np.float64)
    tails = np.ascontiguousarray(tail_index, dtype=np.int64)
    heads = np.ascontiguousarray(head_index, dtype=np.int64)
    represented_weights = np.ascontiguousarray(weights, dtype=np.float64)
    if r.shape != x.shape or r.shape != u.shape or r.ndim != 2:
        raise ValueError("Arb phase inputs must be aligned rank-two arrays")
    if r.shape[1] != tails.size or tails.shape != heads.shape:
        raise ValueError("Arb phase matching does not align with edge inputs")
    if represented_weights.shape[1] <= int(max(tails.max(), heads.max())):
        raise ValueError("Arb phase weights do not cover matching vertices")

    midpoint = np.empty((r.shape[0], 10), dtype=np.float64)
    lower = np.empty_like(midpoint)
    upper = np.empty_like(midpoint)
    with _spectral._ARB_CONTEXT_LOCK:
        previous = int(_spectral._flint_ctx.prec)
        try:
            _spectral._flint_ctx.prec = 256
            arb_zero = _spectral._arb(0)
            arb_one = _spectral._arb(1)
            arb_two = _spectral._arb(2)
            arb_three = _spectral._arb(3)
            arb_six = _spectral._arb(6)
            weight_delta = [
                [
                    _spectral._arb_exact(
                        float(represented_weights[observable, int(heads[edge])])
                    )
                    - _spectral._arb_exact(
                        float(represented_weights[observable, int(tails[edge])])
                    )
                    for edge in range(tails.size)
                ]
                for observable in range(8)
            ]
            for path in range(r.shape[0]):
                accumulators = [arb_zero for _ in range(10)]
                for edge in range(r.shape[1]):
                    represented_r = float(r[path, edge])
                    represented_u = float(u[path, edge])
                    if represented_r == 0.0 or represented_u == 0.0:
                        continue
                    ar = _spectral._arb_exact(represented_r)
                    ax = _spectral._arb_exact(float(x[path, edge]))
                    au = _spectral._arb_exact(represented_u)
                    az = arb_two * ax - arb_one
                    p2 = (arb_three * az * az - arb_one) / arb_two
                    decay2 = ((-arb_two * au).exp() - arb_one)
                    decay6 = ((-arb_six * au).exp() - arb_one)
                    linear_edge = ar * az * decay2 / arb_two
                    for observable in range(8):
                        accumulators[observable] = (
                            accumulators[observable]
                            + weight_delta[observable][edge] * linear_edge
                        )
                    accumulators[8] = (
                        accumulators[8] + ar * ar * p2 * decay6 / arb_three
                    )
                    accumulators[9] = (
                        accumulators[9]
                        + ar * ar * ar * p2 * decay6 / arb_two
                    )
                for observable, value in enumerate(accumulators):
                    bounds = _spectral._arb_bounds(value)
                    midpoint[path, observable] = float(value.mid())
                    lower[path, observable] = bounds.lower
                    upper[path, observable] = bounds.upper
        finally:
            _spectral._flint_ctx.prec = previous
    return midpoint, lower, upper


def _phase_moment_oracle_controls(
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare the implementation with independent Jacobi/Arb eigenmoments."""

    spec = refinement_observable_spec()
    matchings = build_four_color_matchings(28)
    weights = np.asarray(spec.fourier_weights, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    maximum_float64 = 0.0
    maximum_cuda = 0.0
    maximum_radius = 0.0
    maximum_arb_error = 0.0
    arb_enclosure_pass = True
    adversarial_root_enclosure_pass = True
    all_colors = True
    all_durations = True
    facet_interior = True
    zero_cases = True
    fixture_x = np.linspace(0.0, 1.0, 4 * 392, dtype=np.float64).reshape(4, 392)
    # A represented binary64 point adjacent to the positive P2 root exposes
    # any implementation that treats the rounded polynomial as exact.
    fixture_x[0, :] = (1.0 + float(1.0 / math.sqrt(3.0))) / 2.0
    fixture_r = np.broadcast_to(
        np.geomspace(1.0e-5, 0.25, 392, dtype=np.float64), (4, 392)
    ).copy()
    for matching_index, matching in enumerate(matchings):
        tail_index = np.asarray(matching.tails, dtype=np.int64)
        head_index = np.asarray(matching.heads, dtype=np.int64)
        for duration in (0.5, 1.0):
            exposure = (
                3.0 * (TAU_EFF / 512.0) * duration
                / (GRID_SPACING * GRID_SPACING * fixture_r)
            )
            expected_float = _independent_phase_formula(
                fixture_r,
                fixture_x,
                exposure,
                tail_index=tail_index,
                head_index=head_index,
                weights=weights,
                arb=False,
            )
            expected, expected_lower, expected_upper = _independent_phase_formula_arb(
                fixture_r,
                fixture_x,
                exposure,
                tail_index=tail_index,
                head_index=head_index,
                weights=weights,
            )
            maximum_arb_error = max(
                maximum_arb_error,
                float(np.max(np.abs(expected_float - expected))),
            )
            cpu_result = compute_dynkin_phase_drift(
                torch.as_tensor(fixture_r, dtype=torch.float64),
                torch.as_tensor(fixture_x, dtype=torch.float64),
                torch.as_tensor(exposure, dtype=torch.float64),
                matching_index=matching_index,
                spec=spec,
                standardized=False,
            )
            cpu = np.asarray(cpu_result.center.detach().cpu(), dtype=np.float64)
            cpu_radius = np.asarray(
                cpu_result.error_radius.detach().cpu(), dtype=np.float64
            )
            cpu_error = float(np.max(np.abs(cpu - expected)))
            arb_enclosure_pass &= bool(
                np.all(expected_lower >= cpu_result.lower.detach().cpu().numpy())
                and np.all(expected_upper <= cpu_result.upper.detach().cpu().numpy())
            )
            adversarial_root_enclosure_pass &= bool(
                np.all(
                    expected_lower[0, 8:]
                    >= cpu_result.lower.detach().cpu().numpy()[0, 8:]
                )
                and np.all(
                    expected_upper[0, 8:]
                    <= cpu_result.upper.detach().cpu().numpy()[0, 8:]
                )
            )
            maximum_float64 = max(maximum_float64, cpu_error)
            maximum_radius = max(maximum_radius, float(np.max(cpu_radius)))
            cuda_error = cpu_error
            cuda_radius = float(np.max(cpu_radius))
            if device.type == "cuda":
                cuda_result = compute_dynkin_phase_drift(
                    torch.as_tensor(fixture_r, dtype=torch.float64, device=device),
                    torch.as_tensor(fixture_x, dtype=torch.float64, device=device),
                    torch.as_tensor(exposure, dtype=torch.float64, device=device),
                    matching_index=matching_index,
                    spec=spec,
                    standardized=False,
                )
                measured = np.asarray(
                    cuda_result.center.detach().cpu(), dtype=np.float64
                )
                radius = np.asarray(
                    cuda_result.error_radius.detach().cpu(), dtype=np.float64
                )
                cuda_error = float(np.max(np.abs(measured - expected)))
                arb_enclosure_pass &= bool(
                    np.all(
                        expected_lower
                        >= cuda_result.lower.detach().cpu().numpy()
                    )
                    and np.all(
                        expected_upper
                        <= cuda_result.upper.detach().cpu().numpy()
                    )
                )
                adversarial_root_enclosure_pass &= bool(
                    np.all(
                        expected_lower[0, 8:]
                        >= cuda_result.lower.detach().cpu().numpy()[0, 8:]
                    )
                    and np.all(
                        expected_upper[0, 8:]
                        <= cuda_result.upper.detach().cpu().numpy()[0, 8:]
                    )
                )
                cuda_radius = float(np.max(radius))
                maximum_cuda = max(maximum_cuda, cuda_error)
                maximum_radius = max(maximum_radius, cuda_radius)
            else:
                maximum_cuda = max(maximum_cuda, cuda_error)
            all_colors &= cpu_error <= 1.0e-10
            all_durations &= cpu_error <= 1.0e-10
            facet_interior &= bool(np.isfinite(cpu).all())
            rows.append(
                {
                    "matching": str(matching.name),
                    "matching_index": matching_index,
                    "duration_fraction": duration,
                    "maximum_float64_error": cpu_error,
                    "maximum_cuda_error": cuda_error,
                    "maximum_error_radius": cuda_radius,
                }
            )

    zeros = torch.zeros((2, 392), dtype=torch.float64, device=device)
    zero_result = compute_dynkin_phase_drift(
        zeros,
        zeros,
        zeros,
        matching_index=0,
        spec=spec,
        standardized=False,
    )
    zero_cases &= bool(torch.equal(zero_result.center, torch.zeros_like(zero_result.center)))
    correct = rows[0]["maximum_float64_error"]
    # These fixtures are deliberately wrong and must remain distinguishable
    # from the exact formula.  Their concrete magnitudes are diagnostic only.
    wrong_orientation = expected.copy()
    wrong_orientation[:, :8] *= -1.0
    wrong_eigenvalue = _independent_phase_formula(
        fixture_r,
        fixture_x,
        exposure / 2.0,
        tail_index=tail_index,
        head_index=head_index,
        weights=weights,
        arb=False,
    )
    wrong_pair_mass = _independent_phase_formula(
        np.ones_like(fixture_r),
        fixture_x,
        exposure,
        tail_index=tail_index,
        head_index=head_index,
        weights=weights,
        arb=False,
    )
    wrong_duration = _independent_phase_formula(
        fixture_r,
        fixture_x,
        exposure * 2.0,
        tail_index=tail_index,
        head_index=head_index,
        weights=weights,
        arb=False,
    )
    shifted_x = np.clip(fixture_x + 0.05, 0.0, 1.0)
    wrong_post_state = _independent_phase_formula(
        fixture_r,
        shifted_x,
        exposure,
        tail_index=tail_index,
        head_index=head_index,
        weights=weights,
        arb=False,
    )

    def rejected(wrong: np.ndarray) -> bool:
        return float(np.max(np.abs(wrong - expected))) > 1.0e-8

    return {
        "phase_moment_formula_pass": int(correct <= 1.0e-10),
        "phase_moment_all_colors_pass": int(all_colors),
        "phase_moment_half_full_duration_pass": int(all_durations),
        "phase_moment_facet_interior_pass": int(facet_interior),
        "phase_moment_zero_mass_duration_pass": int(zero_cases),
        "spectral_arb_agreement_pass": int(
            maximum_float64 <= 1.0e-10 and maximum_arb_error <= 1.0e-10
        ),
        "cuda_enclosure_pass": int(
            maximum_cuda <= 2.0e-6 and arb_enclosure_pass
        ),
        "adversarial_p2_root_enclosure_pass": int(
            adversarial_root_enclosure_pass
        ),
        "cumulative_error_pass": int(maximum_radius <= 1.0e-8),
        "maximum_float64_phase_moment_error": maximum_float64,
        "maximum_cuda_phase_moment_error": maximum_cuda,
        "maximum_cumulative_standardized_error": maximum_radius,
        "maximum_float64_vs_arb_error": maximum_arb_error,
        "negative_orientation_fixture_pass": int(rejected(wrong_orientation)),
        "negative_eigenvalue_fixture_pass": int(rejected(wrong_eigenvalue)),
        "negative_pair_mass_fixture_pass": int(rejected(wrong_pair_mass)),
        "negative_duration_fixture_pass": int(rejected(wrong_duration)),
        "negative_post_state_fixture_pass": int(rejected(wrong_post_state)),
    }, rows


def _dynkin_shard_fingerprint(
    *,
    stage: str = "pilot",
    panel: str,
    sample_steps: int,
    path_ids: Sequence[int],
    root_seed: int,
    path_id_plan_sha256: str,
) -> str:
    return config_fingerprint(
        {
            "schema": RUN_SCHEMA + "-shard-input",
            "schema_version": 1,
            "observer": "exact-phasewise-dynkin-v1",
            "scheduler_version": REFINEMENT_SCHEDULER_VERSION,
            "transition_id_version": REFINEMENT_ID_VERSION,
            "rng_version": REFINEMENT_RNG_VERSION,
            "phase_matchings": list(PHASE_MATCHINGS),
            "phase_durations": list(PHASE_DURATIONS),
            "sample_steps": int(sample_steps),
            "path_ids": [int(value) for value in path_ids],
            "path_id_plan_version": DYNKIN_PATH_ID_PLAN_VERSION,
            "path_id_plan_sha256": path_id_plan_sha256,
            "panel": panel,
            "stage": stage,
            "root_seed": int(root_seed),
            "profile": JacobiRBCudaProfile().to_dict(),
        }
    )


def _load_dynkin_shard(
    path: Path,
    *,
    fingerprint: str,
    input_sha256: str,
    previous_chain_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    state_path = path.with_suffix(".npz")
    try:
        payload = _load(path)
        row = payload.get("row")
        if (
            payload.get("input_fingerprint") != fingerprint
            or not isinstance(row, Mapping)
            or payload.get("row_sha256") != config_fingerprint(row)
            or row.get("input_states_sha256") != input_sha256
            or row.get("previous_chain_sha256") != previous_chain_sha256
            or row.get("state_npz_name") != state_path.name
            or row.get("state_npz_sha256") != file_fingerprint(state_path)
            or row.get("state_npz_size") != state_path.stat().st_size
        ):
            return None
        with np.load(state_path, allow_pickle=False) as archive:
            arrays = {
                name: np.ascontiguousarray(archive[name], dtype=np.float64)
                for name in (
                    "final_states",
                    "accumulator_center",
                    "accumulator_compensation",
                    "accumulator_error_radius",
                )
            }
        if (
            arrays["final_states"].ndim != 2
            or arrays["final_states"].shape[1] != PATH_STATE_SIZE
            or any(not np.isfinite(value).all() for value in arrays.values())
            or row.get("persisted_state_payload_sha256")
            != config_fingerprint(
                {name: _array_sha256(value) for name, value in arrays.items()}
            )
        ):
            return None
        expected_chain = config_fingerprint(
            {
                "fingerprint": fingerprint,
                "input_states_sha256": input_sha256,
                "previous_chain_sha256": previous_chain_sha256,
                "batch_output_sha256": row.get("batch_output_sha256"),
                "batch_final_state_sha256": row.get("batch_final_state_sha256"),
                "batch_certificate_sha256": row.get("batch_certificate_sha256"),
                "persisted_state_payload_sha256": row.get(
                    "persisted_state_payload_sha256"
                ),
                "state_npz_sha256": row.get("state_npz_sha256"),
            }
        )
        if row.get("chain_sha256") != expected_chain:
            return None
        return dict(row), arrays
    except (
        ArtifactCompatibilityError,
        EOFError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ):
        return None


def _run_dynkin_level_panel(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    stage: str = "pilot",
    panel: str,
    initial_states: np.ndarray,
    path_ids: Sequence[int],
    sample_steps: int,
    root_seed: int,
    legacy_ids: bool = False,
    stop_steps: int | None = None,
) -> dict[str, Any]:
    initial = np.ascontiguousarray(initial_states, dtype=np.float64)
    paths = tuple(int(value) for value in path_ids)
    path_id_plan = _load_frozen_path_id_plan(run_dir)
    try:
        if legacy_ids:
            path_id_plan.validate_role_path_ids("legacy_replay", paths)
        elif stage == "pilot" and panel in PANEL_NAMES:
            path_id_plan.validate_role_path_ids(f"pilot_{panel}", paths)
    except (TypeError, ValueError) as exc:
        raise DynkinSchedulerConfigurationError(
            f"Dynkin shard path allocation is invalid: {exc}"
        ) from exc
    level = int(sample_steps)
    total_steps = level if stop_steps is None else int(stop_steps)
    if initial.shape != (len(paths), PATH_STATE_SIZE):
        raise ValueError("panel state/path shapes do not agree")
    if (
        level not in SUPPORTED_SAMPLE_STEPS
        or total_steps <= 0
        or total_steps > level
        or total_steps % REFINEMENT_SHARD_STEPS
    ):
        raise ValueError("invalid Dynkin level plan")
    checkpoint_steps = tuple(
        int(round(level * fraction))
        for fraction in OBSERVATION_TIME_FRACTIONS
        if int(round(level * fraction)) <= total_steps
    )
    root = run_dir / "dynkin_shards" / stage / panel / f"K{level:04d}"
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    profile = JacobiRBCudaProfile()
    rows: list[dict[str, Any]] = []
    raw_by_step: dict[int, dict[int, np.ndarray]] = {
        step: {} for step in checkpoint_steps
    }
    dynkin_by_step: dict[int, dict[int, np.ndarray]] = {
        step: {} for step in checkpoint_steps
    }
    error_by_step: dict[int, dict[int, np.ndarray]] = {
        step: {} for step in checkpoint_steps
    }
    final_by_path: dict[int, np.ndarray] = {}
    total_shards = math.ceil(len(paths) / MAX_REFINEMENT_PATHS_PER_GROUP) * (
        total_steps // REFINEMENT_SHARD_STEPS
    )
    done = 0
    started = time.perf_counter()
    for group_index, offset in enumerate(
        range(0, len(paths), MAX_REFINEMENT_PATHS_PER_GROUP)
    ):
        group_paths = paths[offset : offset + MAX_REFINEMENT_PATHS_PER_GROUP]
        committed = np.ascontiguousarray(
            initial[offset : offset + len(group_paths)], dtype=np.float64
        )
        states = torch.as_tensor(
            committed, dtype=torch.float64, device=device
        ).contiguous()
        accumulator: DynkinAccumulatorState | None = None
        fingerprint = _dynkin_shard_fingerprint(
            stage=stage,
            panel=panel,
            sample_steps=level,
            path_ids=group_paths,
            root_seed=root_seed,
            path_id_plan_sha256=path_id_plan.sha256,
        )
        previous_chain = config_fingerprint(
            {
                "kind": "dynkin-shard-genesis",
                "fingerprint": fingerprint,
                "initial_states_sha256": _array_sha256(committed),
            }
        )
        reuse_tail = True
        for start_step in range(0, total_steps, REFINEMENT_SHARD_STEPS):
            shard_started = time.perf_counter()
            base = (
                f"group-{group_index:03d}-steps-{start_step:04d}-"
                f"{start_step + REFINEMENT_SHARD_STEPS - 1:04d}"
            )
            metadata_path = root / f"{base}.json"
            input_hash = _array_sha256(committed)
            loaded = (
                _load_dynkin_shard(
                    metadata_path,
                    fingerprint=fingerprint,
                    input_sha256=input_hash,
                    previous_chain_sha256=previous_chain,
                )
                if reuse_tail and metadata_path.is_file()
                else None
            )
            if loaded is None:
                reuse_tail = False
                result = run_dynkin_refinement_shard(
                    states,
                    path_ids=group_paths,
                    sample_steps=level,
                    start_step=start_step,
                    root_seed=int(root_seed),
                    panel_namespace=panel,
                    profile=profile,
                    checkpoint_steps=checkpoint_steps,
                    transition_id_provider=(
                        legacy_k512_transition_ids if legacy_ids else None
                    ),
                    rng_key_override=(
                        (int(root_seed), "full-path-v2") if legacy_ids else None
                    ),
                    accumulator_state=accumulator,
                )
                committed = np.array(
                    result.committed_final_states,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                )
                arrays = {
                    "final_states": committed,
                    "accumulator_center": np.asarray(
                        result.committed_accumulator_center, dtype=np.float64
                    ),
                    "accumulator_compensation": np.asarray(
                        result.committed_accumulator_compensation,
                        dtype=np.float64,
                    ),
                    "accumulator_error_radius": np.asarray(
                        result.committed_accumulator_error_radius,
                        dtype=np.float64,
                    ),
                }
                state_path = metadata_path.with_suffix(".npz")
                _atomic_write_npz(state_path, **arrays)
                payload_hash = config_fingerprint(
                    {name: _array_sha256(value) for name, value in arrays.items()}
                )
                record = result.to_record()
                base_record = dict(record.pop("base_shard"))
                wall = max(0.0, time.perf_counter() - shard_started)
                row = {
                    "panel": panel,
                    "sample_steps": level,
                    "group_index": group_index,
                    "path_ids": list(group_paths),
                    "start_step": start_step,
                    "step_count": REFINEMENT_SHARD_STEPS,
                    "input_states_sha256": input_hash,
                    "previous_chain_sha256": previous_chain,
                    "persisted_state_payload_sha256": payload_hash,
                    "state_npz_name": state_path.name,
                    "state_npz_sha256": file_fingerprint(state_path),
                    "state_npz_size": state_path.stat().st_size,
                    "complete_wall_upper_seconds": wall + max(0.05, 0.02 * wall),
                    **base_record,
                    **record,
                    **NO_WORK,
                }
                row["chain_sha256"] = config_fingerprint(
                    {
                        "fingerprint": fingerprint,
                        "input_states_sha256": input_hash,
                        "previous_chain_sha256": previous_chain,
                        "batch_output_sha256": row["batch_output_sha256"],
                        "batch_final_state_sha256": row["batch_final_state_sha256"],
                        "batch_certificate_sha256": row["batch_certificate_sha256"],
                        "persisted_state_payload_sha256": payload_hash,
                        "state_npz_sha256": row["state_npz_sha256"],
                    }
                )
                atomic_write_json(
                    metadata_path,
                    {
                        "input_fingerprint": fingerprint,
                        "row": row,
                        "row_sha256": config_fingerprint(row),
                        **NO_WORK,
                    },
                )
            else:
                row, arrays = loaded
                committed = arrays["final_states"]
            states = torch.as_tensor(
                committed, dtype=torch.float64, device=device
            ).contiguous()
            accumulator = DynkinAccumulatorState(
                center=torch.as_tensor(
                    np.array(
                        arrays["accumulator_center"],
                        dtype=np.float64,
                        copy=True,
                        order="C",
                    ),
                    dtype=torch.float64,
                    device=device,
                ).contiguous(),
                compensation=torch.as_tensor(
                    np.array(
                        arrays["accumulator_compensation"],
                        dtype=np.float64,
                        copy=True,
                        order="C",
                    ),
                    dtype=torch.float64,
                    device=device,
                ).contiguous(),
                error_radius=torch.as_tensor(
                    np.array(
                        arrays["accumulator_error_radius"],
                        dtype=np.float64,
                        copy=True,
                        order="C",
                    ),
                    dtype=torch.float64,
                    device=device,
                ).contiguous(),
            )
            previous_chain = str(row["chain_sha256"])
            for checkpoint in row.get("observable_checkpoints", ()):
                if not isinstance(checkpoint, Mapping):
                    raise ArtifactCompatibilityError("invalid Dynkin checkpoint")
                step = int(checkpoint["completed_step"])
                ids = tuple(int(value) for value in checkpoint["path_ids"])
                raw = np.asarray(checkpoint["raw_values"], dtype=np.float64)
                dynkin = np.asarray(checkpoint["dynkin_values"], dtype=np.float64)
                error = np.asarray(
                    checkpoint["dynkin_error_radius"], dtype=np.float64
                )
                if raw.shape != (len(ids), 10) or dynkin.shape != raw.shape:
                    raise ArtifactCompatibilityError("invalid Dynkin checkpoint shape")
                for index, path_id in enumerate(ids):
                    raw_by_step.setdefault(step, {})[path_id] = raw[index]
                    dynkin_by_step.setdefault(step, {})[path_id] = dynkin[index]
                    error_by_step.setdefault(step, {})[path_id] = error[index]
            rows.append(row)
            done += 1
            _progress(args, f"pilot/{panel}/K{level}", done, total_shards, started)
        for path_id, value in zip(group_paths, committed, strict=True):
            final_by_path[path_id] = value

    raw_checkpoints = {
        step: np.stack([raw_by_step[step][path_id] for path_id in paths])
        for step in checkpoint_steps
    }
    dynkin_checkpoints = {
        step: np.stack([dynkin_by_step[step][path_id] for path_id in paths])
        for step in checkpoint_steps
    }
    error_checkpoints = {
        step: np.stack([error_by_step[step][path_id] for path_id in paths])
        for step in checkpoint_steps
    }
    ordered_final = np.stack([final_by_path[path_id] for path_id in paths])
    diagnostics = [dict(row.get("diagnostics", {})) for row in rows]
    transition_count = sum(int(row.get("transition_count", 0)) for row in diagnostics)
    certified_count = sum(int(row.get("certified_count", 0)) for row in diagnostics)
    fallback_count = sum(int(row.get("fallback_count", 0)) for row in diagnostics)
    elapsed_seconds = sum(
        float(row.get("elapsed_seconds", 0.0)) for row in diagnostics
    )
    fallback_elapsed_seconds = sum(
        float(row.get("fallback_elapsed_seconds", 0.0)) for row in diagnostics
    )
    complete_wall = sum(float(row["complete_wall_upper_seconds"]) for row in rows)
    peak_memory_fraction = (
        torch.cuda.max_memory_allocated(device)
        / torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else 0.0
    )
    return {
        "sample_steps": level,
        "path_ids": list(paths),
        "path_count": len(paths),
        "stop_steps": total_steps,
        "initial_states_sha256": _array_sha256(initial),
        "final_states_sha256": _array_sha256(ordered_final),
        "final_states": ordered_final,
        "raw_checkpoint_values": raw_checkpoints,
        "dynkin_checkpoint_values": dynkin_checkpoints,
        "dynkin_error_radius": error_checkpoints,
        "rows": rows,
        "transition_count": transition_count,
        "certified_count": certified_count,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "fallback_count": fallback_count,
        "fallback_fraction": (
            fallback_count / transition_count if transition_count else 0.0
        ),
        "elapsed_seconds": elapsed_seconds,
        "fallback_elapsed_seconds": fallback_elapsed_seconds,
        "complete_wall_upper_seconds": complete_wall,
        "conservative_rate": (
            transition_count / complete_wall if complete_wall > 0.0 else 0.0
        ),
        "mass_error": max(
            (
                float(row.get("maximum_global_simplex_error", math.inf))
                for row in diagnostics
            ),
            default=math.inf,
        ),
        "peak_memory_fraction": peak_memory_fraction,
        "state_updates_device_resident_pass": int(
            bool(diagnostics)
            and all(
                int(row.get("state_updates_device_resident", 0)) == 1
                and int(row.get("in_shard_host_roundtrip_count", -1)) == 0
                for row in diagnostics
            )
        ),
        "shard_chain_pass": 1,
        **{
            name: sum(int(row.get(name, 0)) for row in diagnostics)
            for name in _FORBIDDEN_COUNTS
        },
    }


def _initial_dirichlet(path_count: int, seed: int) -> np.ndarray:
    return np.random.Generator(np.random.Philox(int(seed))).dirichlet(
        np.ones(PATH_STATE_SIZE, dtype=np.float64), size=int(path_count)
    )


def _observer_legacy_replay(
    run_dir: Path, args: argparse.Namespace, parent_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    path_id_plan = _load_frozen_path_id_plan(run_dir)
    path_ids = path_id_plan.legacy_replay_path_ids
    path_id_plan.validate_role_path_ids("legacy_replay", path_ids)
    initial = _initial_dirichlet(10, 261_141 + 20_000 + 10)[:8]
    result = _run_dynkin_level_panel(
        run_dir,
        args,
        stage="preflight",
        # The immutable replay certificate binds the panel namespace as part
        # of its RNG plan, so the observer must reuse the historical name
        # byte-for-byte.
        panel="legacy-k512-replay",
        initial_states=initial,
        path_ids=path_ids,
        sample_steps=512,
        root_seed=261_141,
        stop_steps=8,
        legacy_ids=True,
    )
    immutable = _load(parent_dir / "legacy_k512_replay.json")
    parent_shard_path = (
        parent_dir
        / "refinement_shards"
        / "preflight"
        / "legacy-k512-replay"
        / "K0512"
        / "group-000-steps-0000-0007.json"
    )
    parent_shard = _load(parent_shard_path)
    parent_row = parent_shard.get("row")
    if not isinstance(parent_row, Mapping):
        raise ArtifactCompatibilityError("immutable replay shard is malformed")
    observed_rows = result.get("rows")
    if not isinstance(observed_rows, list) or len(observed_rows) != 1:
        raise ArtifactCompatibilityError("observer replay did not produce one shard")
    observed_row = observed_rows[0]
    if not isinstance(observed_row, Mapping):
        raise ArtifactCompatibilityError("observer replay shard is malformed")
    expected_state_hash = str(immutable.get("replay_final_states_sha256"))
    expected_output_hash = str(parent_row.get("batch_output_sha256"))
    expected_certificate_hash = str(parent_row.get("batch_certificate_sha256"))
    state_pass = (
        result["final_states_sha256"] == expected_state_hash
        and observed_row.get("batch_final_state_sha256") == expected_state_hash
    )
    output_hash_pass = (
        observed_row.get("batch_output_sha256") == expected_output_hash
    )
    certificate_hash_pass = (
        observed_row.get("batch_certificate_sha256")
        == expected_certificate_hash
    )
    raw_checkpoints_absent = not result["raw_checkpoint_values"]
    record = {
        "schema": RUN_SCHEMA + "-legacy-k512-observer-replay",
        "schema_version": 1,
        "path_ids": list(path_ids),
        "path_id_plan_version": path_id_plan.version,
        "path_id_plan_sha256": path_id_plan.sha256,
        "immutable_parent_record_sha256": file_fingerprint(
            parent_dir / "legacy_k512_replay.json"
        ),
        "immutable_parent_shard_sha256": file_fingerprint(parent_shard_path),
        "expected_batch_output_sha256": expected_output_hash,
        "observed_batch_output_sha256": observed_row.get("batch_output_sha256"),
        "expected_batch_certificate_sha256": expected_certificate_hash,
        "observed_batch_certificate_sha256": observed_row.get(
            "batch_certificate_sha256"
        ),
        "expected_final_states_sha256": expected_state_hash,
        "observed_final_states_sha256": result["final_states_sha256"],
        "legacy_k512_replay_pass": int(
            state_pass and output_hash_pass and certificate_hash_pass
        ),
        "observer_state_hash_invariance_pass": int(state_pass),
        "transition_and_target_hash_invariance_pass": int(output_hash_pass),
        "certificate_hash_invariance_pass": int(certificate_hash_pass),
        "no_unrequested_checkpoint_pass": int(raw_checkpoints_absent),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "legacy_k512_dynkin_observer_replay.json", record)
    return record, result


def _tower_panel(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    panel: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    metadata_path = run_dir / f"tower_panel_{panel}.json"
    payload_path = run_dir / f"tower_panel_{panel}.npz"
    path_id_plan = _load_frozen_path_id_plan(run_dir)
    if metadata_path.is_file():
        frozen = _load(metadata_path)
        if (
            not payload_path.is_file()
            or frozen.get("npz_name") != payload_path.name
            or frozen.get("npz_sha256") != file_fingerprint(payload_path)
            or frozen.get("npz_size") != payload_path.stat().st_size
            or frozen.get("panel") != panel
            or frozen.get("seed") != seed
            or frozen.get("path_count") != int(args.tower_panel_clusters)
            or frozen.get("path_id_plan_version") != path_id_plan.version
            or frozen.get("path_id_plan_sha256") != path_id_plan.sha256
            or not isinstance(frozen.get("execution"), Mapping)
        ):
            raise ArtifactCompatibilityError(f"frozen tower panel {panel} changed")
        rows = [
            {
                "panel": panel,
                "name": member["name"],
                "mean_minus_expected": member["mean_minus_expected"],
                "standard_error": member["standard_error"],
                "simultaneous_lower": member["simultaneous_lower"],
                "simultaneous_upper": member["simultaneous_upper"],
                "contains_zero": member["contains_zero"],
            }
            for member in frozen["inference"]["members"]
        ]
        return frozen, rows, dict(frozen["execution"])

    path_count = int(args.tower_panel_clusters)
    device = torch.device(args.device)
    states_np = _initial_dirichlet(path_count, seed)
    states = torch.as_tensor(states_np, dtype=torch.float64, device=device).contiguous()
    path_values: dict[str, np.ndarray] = {}
    case_records: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    spec = refinement_observable_spec()
    for matching_index in range(4):
        for duration_index, duration in enumerate((0.5, 1.0)):
            case_index = 2 * matching_index + duration_index
            try:
                case_paths = path_id_plan.tower_case_path_ids(
                    panel, case_index
                )
                path_id_plan.validate_role_path_ids(
                    f"tower_{panel}",
                    case_paths,
                    case_index=case_index,
                )
                transition_ids = canonical_tower_transition_ids(
                    case_paths,
                    sample_steps=512,
                    outer_step=case_index,
                    phase=matching_index,
                    device=device,
                )
            except (TypeError, ValueError) as exc:
                raise DynkinSchedulerConfigurationError(
                    f"tower panel {panel} case {case_index} has invalid IDs: {exc}"
                ) from exc
            result = run_dynkin_tower_phase(
                states,
                matching_index=matching_index,
                duration_fraction=duration,
                sample_steps=512,
                rng_key=(int(args.root_seed), "dynkin-tower", panel, case_index),
                transition_ids=transition_ids,
                profile=JacobiRBCudaProfile(),
            )
            residual = np.asarray(
                result.standardized_residual.detach().cpu(), dtype=np.float64
            )
            if residual.shape != (path_count, 10) or not np.isfinite(residual).all():
                raise ArtifactCompatibilityError("tower residual panel is invalid")
            for observable_index, name in enumerate(spec.names):
                path_values[
                    f"{matching_index}_{duration:g}_{name}"
                ] = residual[:, observable_index]
            row = result.to_record()
            diagnostics = dict(row.get("diagnostics", result.diagnostics))
            aggregate_rows.append(diagnostics)
            case_records.append(
                {
                    "matching_index": matching_index,
                    "duration_fraction": duration,
                    "path_id_plan_sha256": path_id_plan.sha256,
                    "path_ids_sha256": _array_sha256(
                        np.asarray(case_paths, dtype=np.int64)
                    ),
                    "transition_ids_sha256": _array_sha256(
                        transition_ids.detach().cpu().numpy()
                    ),
                    "standardized_residual_sha256": _array_sha256(residual),
                    "maximum_standardized_error_radius": float(
                        torch.max(result.drift_error_radius).detach().cpu().item()
                    ),
                    "diagnostics": diagnostics,
                }
            )
    inference = whole_path_max_t_intervals(
        path_values,
        seed=seed + 1,
        confidence=0.99,
        reps=(100 if args.test_only_reduced_workload else 20_000),
    )
    names = tuple(sorted(path_values))
    matrix = np.column_stack([path_values[name] for name in names])
    _atomic_write_npz(
        payload_path,
        initial_states=states_np,
        path_matrix=matrix,
    )
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-tower-panel",
        "schema_version": 1,
        "panel": panel,
        "seed": seed,
        "path_count": path_count,
        "path_id_plan_version": path_id_plan.version,
        "path_id_plan_sha256": path_id_plan.sha256,
        "path_feature_names": list(names),
        "path_matrix_sha256": _array_sha256(matrix),
        "initial_states_sha256": _array_sha256(states_np),
        "npz_name": payload_path.name,
        "npz_sha256": file_fingerprint(payload_path),
        "npz_size": payload_path.stat().st_size,
        "cases": case_records,
        "inference": inference,
        "passed": int(inference["passed"]),
        **NO_WORK,
    }
    rows = [
        {
            "panel": panel,
            "name": member["name"],
            "mean_minus_expected": member["mean_minus_expected"],
            "standard_error": member["standard_error"],
            "simultaneous_lower": member["simultaneous_lower"],
            "simultaneous_upper": member["simultaneous_upper"],
            "contains_zero": member["contains_zero"],
        }
        for member in inference["members"]
    ]
    transition_count = sum(
        int(row.get("transition_count", 0)) for row in aggregate_rows
    )
    certified_count = sum(int(row.get("certified_count", 0)) for row in aggregate_rows)
    fallback_count = sum(int(row.get("fallback_count", 0)) for row in aggregate_rows)
    elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in aggregate_rows)
    summary = {
        "transition_count": transition_count,
        "certified_count": certified_count,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "fallback_count": fallback_count,
        "fallback_fraction": (
            fallback_count / transition_count if transition_count else 0.0
        ),
        "fallback_elapsed_seconds": sum(
            float(row.get("fallback_elapsed_seconds", 0.0))
            for row in aggregate_rows
        ),
        "elapsed_seconds": elapsed,
        "mass_error": max(
            (
                max(
                    float(row.get("maximum_pair_total_error", math.inf)),
                    float(row.get("maximum_global_simplex_error", math.inf)),
                )
                for row in aggregate_rows
            ),
            default=math.inf,
        ),
        "peak_memory_fraction": (
            torch.cuda.max_memory_allocated(device)
            / torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else 0.0
        ),
        **{
            name: sum(int(row.get(name, 0)) for row in aggregate_rows)
            for name in _FORBIDDEN_COUNTS
        },
    }
    record["execution"] = summary
    atomic_write_json(metadata_path, record)
    return record, rows, summary


def _run_preflight_stage(
    run_dir: Path,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    existing = run_dir / "dynkin_preflight_gate.json"
    if existing.is_file():
        return _load(existing)
    path_id_plan = _load_frozen_path_id_plan(run_dir)
    path_id_evidence = _path_id_plan_preflight(
        run_dir, args, path_id_plan
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    oracle, oracle_rows = _phase_moment_oracle_controls(device)
    atomic_write_csv(run_dir / "phase_moment_oracle.csv", oracle_rows)

    replay, replay_result = _observer_legacy_replay(
        run_dir, args, Path(args.parent_strang_run_dir).resolve()
    )
    panel_a, rows_a, execution_a = _tower_panel(
        run_dir,
        args,
        panel="a",
        seed=int(args.root_seed) + 10_000,
    )
    panel_b, rows_b, execution_b = _tower_panel(
        run_dir,
        args,
        panel="b",
        seed=int(args.root_seed) + 20_000,
    )
    atomic_write_csv(run_dir / "tower_identity_intervals.csv", rows_a + rows_b)
    impossibility = _distribution_free_power_impossibility()
    atomic_write_json(run_dir / "distribution_free_power_impossibility.json", impossibility)
    replay_execution = _aggregate_level_results([replay_result])
    replay_cumulative_error = max(
        (
            float(row.get("diagnostics", {}).get(
                "dynkin_maximum_cumulative_standardized_error_radius",
                math.inf,
            ))
            for row in replay_result.get("rows", ())
        ),
        default=math.inf,
    )
    cumulative_error = replay_cumulative_error
    oracle = {
        **oracle,
        "maximum_cumulative_standardized_error": cumulative_error,
        "cumulative_error_pass": int(cumulative_error <= 1.0e-8),
    }
    executions = (replay_execution, execution_a, execution_b)
    transition_count = sum(int(value.get("transition_count", 0)) for value in executions)
    certified_count = sum(int(value.get("certified_count", 0)) for value in executions)
    fallback_count = sum(int(value.get("fallback_count", 0)) for value in executions)
    elapsed = sum(float(value.get("elapsed_seconds", 0.0)) for value in executions)
    fallback_elapsed = sum(
        float(value.get("fallback_elapsed_seconds", 0.0)) for value in executions
    )
    peak = max(float(value.get("peak_memory_fraction", 0.0)) for value in executions)
    metrics = {
        "production_authorizing_pass": int(
            not args.test_only_reduced_workload
        ),
        "control_provenance_pass": int(provenance.get("passed", 0)),
        "parent_power_adjudication_pass": int(
            provenance.get("parent_re_adjudication") == "raw_endpoint_power_infeasible"
        ),
        "fifteen_parent_sources_immutable_pass": int(
            provenance.get("parent_source_count") == 15
        ),
        "parent_preflight_pass": int(provenance.get("parent_preflight_pass", 0)),
        "parent_power_numerically_valid_pass": int(
            provenance.get("parent_power_numerically_valid", 0)
        ),
        "parent_no_work_pass": int(
            not any(
                int(provenance.get(name, 0))
                for name in (
                    "physical_training_performed",
                    "sampling_performed",
                    "reverse_sampling_performed",
                )
            )
        ),
        "path_id_plan_pass": int(path_id_evidence.get("passed", 0)),
        "legacy_k512_id_plan_pass": int(
            replay.get("path_ids") == list(path_id_plan.legacy_replay_path_ids)
            and replay.get("path_id_plan_version") == path_id_plan.version
            and replay.get("path_id_plan_sha256") == path_id_plan.sha256
        ),
        "legacy_k512_replay_pass": replay["legacy_k512_replay_pass"],
        "observer_state_hash_invariance_pass": replay[
            "observer_state_hash_invariance_pass"
        ],
        **oracle,
        "tower_panel_a_pass": int(panel_a.get("passed", 0)),
        "tower_panel_b_pass": int(panel_b.get("passed", 0)),
        "tower_joint_max_t_pass": int(
            panel_a.get("passed", 0) and panel_b.get("passed", 0)
        ),
        "tower_panels_frozen_pass": int(
            (run_dir / "tower_panel_a.json").is_file()
            and (run_dir / "tower_panel_b.json").is_file()
        ),
        "tower_panels_disjoint_pass": int(
            panel_a.get("initial_states_sha256")
            != panel_b.get("initial_states_sha256")
        ),
        "distribution_free_power_record_pass": int(
            impossibility.get("authorizing") == 0
            and impossibility.get("documented_required_paths_upper_order") == 2.4e18
        ),
        "right_endpoint_coupling_unchanged_pass": int(
            path_id_evidence["checks"].get("right_endpoint_alias_pass", 0)
            and path_id_evidence["checks"].get(
                "cross_level_alias_plan_pass", 0
            )
        ),
        "parent_record_count": int(
            provenance.get("parent_artifact_record_count", -1)
        ),
        "parent_source_count": int(provenance.get("parent_source_count", -1)),
        "grid_size": 28,
        "alpha": 1.0,
        "tau_eff": TAU_EFF,
        "levels": list(args.sample_steps),
        "tower_panel_count": 2,
        "tower_clusters_per_panel": int(args.tower_panel_clusters),
        "tower_confidence": 0.99,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "fallback_fraction": (
            fallback_count / transition_count if transition_count else 0.0
        ),
        "fallback_cost_fraction": (
            fallback_elapsed / elapsed if elapsed > 0.0 else 0.0
        ),
        "peak_memory_fraction": peak,
        "mass_error": max(
            (float(value.get("mass_error", math.inf)) for value in executions),
            default=math.inf,
        ),
        **{
            name: sum(int(value.get(name, 0)) for value in executions)
            for name in _FORBIDDEN_COUNTS
        },
        **NO_WORK,
    }
    record = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "metrics": metrics,
        "observer_replay": replay,
        "path_id_plan": {
            "version": path_id_plan.version,
            "semantic_sha256": path_id_plan.sha256,
            "preflight_sha256": file_fingerprint(
                run_dir / "path_id_plan_preflight.json"
            ),
            "passed": path_id_evidence["passed"],
        },
        "tower_panel_a": {
            "metadata_sha256": file_fingerprint(run_dir / "tower_panel_a.json"),
            "passed": panel_a["passed"],
        },
        "tower_panel_b": {
            "metadata_sha256": file_fingerprint(run_dir / "tower_panel_b.json"),
            "passed": panel_b["passed"],
        },
        **NO_WORK,
    }
    atomic_write_json(run_dir / "preflight_metrics.json", record)
    gate = evaluate_dynkin_preflight(metrics)
    atomic_write_json(existing, gate)
    return gate


def _serialize_panel_observables(
    path: Path, results: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for level, result in sorted(results.items()):
        for prefix, key in (
            ("raw", "raw_checkpoint_values"),
            ("dynkin", "dynkin_checkpoint_values"),
            ("error", "dynkin_error_radius"),
        ):
            for step, value in sorted(dict(result[key]).items()):
                arrays[f"{prefix}_K{level:04d}_step{int(step):04d}"] = np.asarray(
                    value, dtype=np.float64
                )
    _atomic_write_npz(path, **arrays)
    return {
        "npz_name": path.name,
        "npz_sha256": file_fingerprint(path),
        "npz_size": path.stat().st_size,
        "array_count": len(arrays),
        "array_sha256": {
            name: _array_sha256(value) for name, value in sorted(arrays.items())
        },
    }


def _run_pilot_panel(
    run_dir: Path,
    args: argparse.Namespace,
    source: Mapping[str, Any],
    *,
    panel: str,
) -> dict[str, Any]:
    metadata_path = run_dir / f"pilot_panel_{panel}_metrics.json"
    payload_path = run_dir / f"pilot_panel_{panel}_observables.npz"
    path_id_plan = _load_frozen_path_id_plan(run_dir)
    if metadata_path.is_file():
        record = _load(metadata_path)
        payload = record.get("observable_payload")
        if (
            not isinstance(payload, Mapping)
            or not payload_path.is_file()
            or payload.get("npz_sha256") != file_fingerprint(payload_path)
            or payload.get("npz_size") != payload_path.stat().st_size
            or record.get("path_id_plan_version") != path_id_plan.version
            or record.get("path_id_plan_sha256") != path_id_plan.sha256
        ):
            raise ArtifactCompatibilityError(f"frozen pilot panel {panel} changed")
        return record

    panel_plan = _load(run_dir / f"panel_{panel}_plan.json")
    if (
        panel_plan.get("path_id_plan_version") != path_id_plan.version
        or panel_plan.get("path_id_plan_sha256") != path_id_plan.sha256
    ):
        raise ArtifactCompatibilityError(
            f"pilot panel {panel} does not bind the path-ID plan"
        )
    path_ids = tuple(int(value) for value in panel_plan["path_ids"])
    try:
        path_id_plan.validate_role_path_ids(f"pilot_{panel}", path_ids)
    except (TypeError, ValueError) as exc:
        raise DynkinSchedulerConfigurationError(
            f"pilot panel {panel} has invalid path IDs: {exc}"
        ) from exc
    mixed = np.asarray(source["mixed_target"], dtype=np.float64).reshape(1, -1)
    initial = np.repeat(mixed, len(path_ids), axis=0)
    levels: dict[int, dict[str, Any]] = {}
    for level in args.sample_steps:
        levels[int(level)] = _run_dynkin_level_panel(
            run_dir,
            args,
            stage="pilot",
            panel=panel,
            initial_states=initial,
            path_ids=path_ids,
            sample_steps=int(level),
            root_seed=int(args.root_seed),
            stop_steps=args.pilot_stop_steps,
        )
    main, reference = _projection_feature_matrices(
        levels, key="dynkin_checkpoint_values"
    )
    raw_main, raw_reference = _projection_feature_matrices(
        levels, key="raw_checkpoint_values"
    )
    execution = _aggregate_level_results(list(levels.values()))
    error_maximum = max(
        (
            float(np.max(value))
            for result in levels.values()
            for value in result["dynkin_error_radius"].values()
        ),
        default=math.inf,
    )
    payload = _serialize_panel_observables(payload_path, levels)
    record = {
        "schema": RUN_SCHEMA + "-pilot-panel",
        "schema_version": 1,
        "panel": panel,
        "plan_sha256": file_fingerprint(run_dir / f"panel_{panel}_plan.json"),
        "path_id_plan_version": path_id_plan.version,
        "path_id_plan_sha256": path_id_plan.sha256,
        "path_ids": list(path_ids),
        "path_count": len(path_ids),
        "levels": list(args.sample_steps),
        "main_differences_sha256": _array_sha256(main),
        "reference_differences_sha256": _array_sha256(reference),
        "raw_main_differences_sha256": _array_sha256(raw_main),
        "raw_reference_differences_sha256": _array_sha256(raw_reference),
        "main_differences": main.tolist(),
        "reference_differences": reference.tolist(),
        "raw_main_differences": raw_main.tolist(),
        "raw_reference_differences": raw_reference.tolist(),
        "maximum_cumulative_standardized_error": error_maximum,
        "execution": execution,
        "observable_payload": payload,
        "complete": 1,
        "finite": int(
            np.isfinite(main).all()
            and np.isfinite(reference).all()
            and math.isfinite(error_maximum)
        ),
        **NO_WORK,
    }
    atomic_write_json(metadata_path, record)
    return record


def _candidate_rows_for_panel(
    record: Mapping[str, Any],
    *,
    role: str,
    parent_dir: Path,
    conservative_rate: float,
) -> list[dict[str, Any]]:
    hours = _parent_projected_hours(parent_dir, conservative_rate)
    rows = build_dynkin_candidate_records(
        main_differences=np.asarray(record["main_differences"], dtype=np.float64),
        reference_differences=np.asarray(
            record["reference_differences"], dtype=np.float64
        ),
        conservative_rate=conservative_rate,
        projected_hours_by_design=hours,
        panel_role=role,
    )
    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        execution = {}
    thresholds = DynkinPowerThresholds()
    elapsed = float(execution.get("elapsed_seconds", 0.0))
    fallback_elapsed = float(execution.get("fallback_elapsed_seconds", math.inf))
    fallback_cost = (
        fallback_elapsed / elapsed if elapsed > 0.0 else math.inf
    )
    forbidden_clean = all(
        int(execution.get(name, -1)) == 0 for name in _FORBIDDEN_COUNTS
    )
    evidence = {
        "panel_complete_pass": int(record.get("complete", 0) == 1),
        "panel_finite_pass": int(record.get("finite", 0) == 1),
        "panel_certification_pass": int(
            float(execution.get("certificate_fraction", 0.0)) == 1.0
        ),
        "panel_numerical_health_pass": int(
            forbidden_clean
            and float(execution.get("fallback_fraction", math.inf))
            <= thresholds.maximum_fallback_fraction
            and fallback_cost <= thresholds.maximum_fallback_cost_fraction
            and float(execution.get("peak_memory_fraction", math.inf))
            <= thresholds.maximum_peak_memory_fraction
            and float(record.get(
                "maximum_cumulative_standardized_error", math.inf
            ))
            <= thresholds.maximum_cumulative_standardized_error
            and int(execution.get("state_updates_device_resident_pass", 0)) == 1
        ),
        "mass_conservation_pass": int(
            float(execution.get("mass_error", math.inf))
            <= thresholds.maximum_cuda_mass_error
        ),
        "shard_chain_pass": int(execution.get("shard_chain_pass", 0) == 1),
        "pilot_production_isolation_pass": 1,
        "pilot_means_excluded_pass": 1,
    }
    for row in rows:
        row.update(evidence)
    return rows


def _combined_panel_record(
    panel_a: Mapping[str, Any], panel_b: Mapping[str, Any]
) -> dict[str, Any]:
    main = np.concatenate(
        (
            np.asarray(panel_a["main_differences"], dtype=np.float64),
            np.asarray(panel_b["main_differences"], dtype=np.float64),
        ),
        axis=0,
    )
    reference = np.concatenate(
        (
            np.asarray(panel_a["reference_differences"], dtype=np.float64),
            np.asarray(panel_b["reference_differences"], dtype=np.float64),
        ),
        axis=0,
    )
    raw_main = np.concatenate(
        (
            np.asarray(panel_a["raw_main_differences"], dtype=np.float64),
            np.asarray(panel_b["raw_main_differences"], dtype=np.float64),
        ),
        axis=0,
    )
    raw_reference = np.concatenate(
        (
            np.asarray(panel_a["raw_reference_differences"], dtype=np.float64),
            np.asarray(panel_b["raw_reference_differences"], dtype=np.float64),
        ),
        axis=0,
    )
    execution_a = panel_a.get("execution")
    execution_b = panel_b.get("execution")
    if not isinstance(execution_a, Mapping) or not isinstance(
        execution_b, Mapping
    ):
        raise ArtifactCompatibilityError("sealed panel execution evidence is missing")
    transition_count = sum(
        int(value.get("transition_count", 0))
        for value in (execution_a, execution_b)
    )
    certified_count = sum(
        int(value.get("certified_count", 0))
        for value in (execution_a, execution_b)
    )
    fallback_count = sum(
        int(value.get("fallback_count", 0))
        for value in (execution_a, execution_b)
    )
    complete_wall = sum(
        float(value.get("complete_wall_upper_seconds", math.inf))
        for value in (execution_a, execution_b)
    )
    combined_execution = {
        "transition_count": transition_count,
        "certified_count": certified_count,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "fallback_count": fallback_count,
        "fallback_fraction": (
            fallback_count / transition_count if transition_count else 0.0
        ),
        "elapsed_seconds": sum(
            float(value.get("elapsed_seconds", 0.0))
            for value in (execution_a, execution_b)
        ),
        "fallback_elapsed_seconds": sum(
            float(value.get("fallback_elapsed_seconds", 0.0))
            for value in (execution_a, execution_b)
        ),
        "complete_wall_upper_seconds": complete_wall,
        "conservative_rate": (
            transition_count / complete_wall
            if transition_count and math.isfinite(complete_wall)
            and complete_wall > 0.0
            else 0.0
        ),
        "mass_error": max(
            float(value.get("mass_error", math.inf))
            for value in (execution_a, execution_b)
        ),
        "peak_memory_fraction": max(
            float(value.get("peak_memory_fraction", math.inf))
            for value in (execution_a, execution_b)
        ),
        "state_updates_device_resident_pass": int(
            all(
                int(value.get("state_updates_device_resident_pass", 0)) == 1
                for value in (execution_a, execution_b)
            )
        ),
        "shard_chain_pass": int(
            all(
                int(value.get("shard_chain_pass", 0)) == 1
                for value in (execution_a, execution_b)
            )
        ),
        **{
            name: sum(
                int(value.get(name, -1))
                for value in (execution_a, execution_b)
            )
            for name in _FORBIDDEN_COUNTS
        },
    }
    return {
        "main_differences": main.tolist(),
        "reference_differences": reference.tolist(),
        "raw_main_differences": raw_main.tolist(),
        "raw_reference_differences": raw_reference.tolist(),
        "execution": combined_execution,
        "complete": int(
            panel_a.get("complete", 0) == 1
            and panel_b.get("complete", 0) == 1
        ),
        "finite": int(
            panel_a.get("finite", 0) == 1
            and panel_b.get("finite", 0) == 1
        ),
        "maximum_cumulative_standardized_error": max(
            float(
                panel_a.get(
                    "maximum_cumulative_standardized_error", math.inf
                )
            ),
            float(
                panel_b.get(
                    "maximum_cumulative_standardized_error", math.inf
                )
            ),
        ),
    }


def _power_feature_names() -> tuple[list[str], list[str]]:
    observable_names = list(refinement_observable_spec().names)
    times = OBSERVATION_TIME_FRACTIONS
    main = [
        f"K{lower}-K{upper}/t={fraction:g}/{observable}"
        for lower, upper in ((128, 256), (256, 512), (512, 1024))
        for fraction in times
        for observable in observable_names
    ]
    reference = [
        f"K512-Richardson2048/t={fraction:g}/{observable}"
        for fraction in times
        for observable in observable_names
    ]
    return main, reference


def _write_feature_power_diagnostics(
    run_dir: Path,
    *,
    role: str,
    record: Mapping[str, Any],
) -> None:
    """Write non-authorizing raw/Dynkin featurewise planning evidence."""

    names_by_family = _power_feature_names()
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-feature-power-diagnostics",
        "schema_version": 1,
        "panel_role": role,
        "authorizing": 0,
        "scientific_confidence_interval": 0,
        "families": {},
        **NO_WORK,
    }
    for family, names, candidates in (
        ("main", names_by_family[0], (32, 64)),
        ("reference", names_by_family[1], (16, 32)),
    ):
        dynkin = np.asarray(
            record[f"{family}_differences"], dtype=np.float64
        )
        raw = np.asarray(
            record[f"raw_{family}_differences"], dtype=np.float64
        )
        if (
            dynkin.shape != raw.shape
            or dynkin.ndim != 2
            or dynkin.shape[1] != len(names)
            or not np.isfinite(dynkin).all()
            or not np.isfinite(raw).all()
        ):
            raise ArtifactCompatibilityError(
                f"{role} {family} raw/Dynkin feature evidence is invalid"
            )
        dynkin_variance = np.var(dynkin, axis=0, ddof=1)
        raw_variance = np.var(raw, axis=0, ddof=1)
        projections: dict[tuple[str, int], np.ndarray] = {}
        maxima: dict[str, dict[str, float]] = {}
        for target, values in (("dynkin", dynkin), ("raw", raw)):
            maxima[target] = {}
            for candidate in candidates:
                projected = normal_chi_square_bonferroni_projection(
                    values, candidate_paths=candidate
                )
                widths = np.asarray(
                    projected["predicted_half_widths"], dtype=np.float64
                )
                projections[(target, candidate)] = widths
                maxima[target][str(candidate)] = float(np.max(widths))
        for index, name in enumerate(names):
            dynkin_value = float(dynkin_variance[index])
            raw_value = float(raw_variance[index])
            ratio: float | str
            if dynkin_value > 0.0:
                ratio = raw_value / dynkin_value
            elif raw_value == 0.0:
                ratio = 1.0
            else:
                ratio = "inf"
            row: dict[str, Any] = {
                "panel_role": role,
                "family": family,
                "feature_index": index,
                "feature_name": name,
                "pilot_path_count": int(dynkin.shape[0]),
                "dynkin_variance": dynkin_value,
                "raw_endpoint_variance": raw_value,
                "raw_to_dynkin_variance_ratio": ratio,
                "authorizing": 0,
            }
            for candidate in candidates:
                row[f"dynkin_half_width_n{candidate}"] = float(
                    projections[("dynkin", candidate)][index]
                )
                row[f"raw_endpoint_half_width_n{candidate}"] = float(
                    projections[("raw", candidate)][index]
                )
            rows.append(row)
        finite_ratios = np.divide(
            raw_variance,
            dynkin_variance,
            out=np.full_like(raw_variance, np.nan),
            where=dynkin_variance > 0.0,
        )
        finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
        summary["families"][family] = {
            "feature_count": len(names),
            "candidate_paths": list(candidates),
            "maximum_projected_half_width": maxima,
            "median_raw_to_dynkin_variance_ratio": (
                float(np.median(finite_ratios))
                if finite_ratios.size
                else "undefined"
            ),
        }
    atomic_write_csv(run_dir / f"feature_power_{role}.csv", rows)
    atomic_write_json(run_dir / f"raw_dynkin_power_{role}.json", summary)


def _write_candidate_csv(
    run_dir: Path,
    role: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    atomic_write_csv(
        run_dir / f"dynkin_candidates_{role}.csv",
        [
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list, tuple))
            }
            for row in rows
        ],
    )


def _run_pilot_stage(
    run_dir: Path,
    args: argparse.Namespace,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    existing = run_dir / "dynkin_power_gate.json"
    if existing.is_file():
        return _load(existing)
    path_id_plan = _load_frozen_path_id_plan(run_dir)
    path_id_evidence = _load(run_dir / "path_id_plan_preflight.json")
    sealed_registry = _load(run_dir / "sealed_panel_registry.json")
    panel_plan_binding_pass = (
        sealed_registry.get("path_id_plan_version") == path_id_plan.version
        and sealed_registry.get("path_id_plan_sha256") == path_id_plan.sha256
        and sealed_registry.get("path_id_plan_file_sha256")
        == file_fingerprint(run_dir / "path_id_plan.json")
    )
    if not panel_plan_binding_pass:
        raise ArtifactCompatibilityError(
            "sealed pilot registry does not bind the path-ID plan"
        )
    thresholds = DynkinPowerThresholds()
    parent_dir = Path(args.parent_strang_run_dir).resolve()
    panel_a = _run_pilot_panel(run_dir, args, source, panel="a")
    execution_a = dict(panel_a["execution"])
    rate_a = float(execution_a.get("conservative_rate", 0.0))
    rows_a = _candidate_rows_for_panel(
        panel_a, role="a", parent_dir=parent_dir, conservative_rate=rate_a
    )
    _write_feature_power_diagnostics(run_dir, role="panel_a", record=panel_a)
    _write_candidate_csv(run_dir, "panel_a", rows_a)
    nomination = select_dynkin_panel_a_design(rows_a)
    atomic_write_json(run_dir / "panel_a_nomination.json", nomination)

    panel_b: dict[str, Any] | None = None
    rows_b: list[dict[str, Any]] | None = None
    combined_rows: list[dict[str, Any]] | None = None
    if _passed(nomination):
        panel_b = _run_pilot_panel(run_dir, args, source, panel="b")
        execution_b = dict(panel_b["execution"])
        conservative_rate = min(
            rate_a, float(execution_b.get("conservative_rate", 0.0))
        )
        rows_b = _candidate_rows_for_panel(
            panel_b,
            role="b",
            parent_dir=parent_dir,
            conservative_rate=conservative_rate,
        )
        combined = _combined_panel_record(panel_a, panel_b)
        combined_rows = _candidate_rows_for_panel(
            combined,
            role="combined",
            parent_dir=parent_dir,
            conservative_rate=conservative_rate,
        )
        _write_feature_power_diagnostics(
            run_dir, role="panel_b", record=panel_b
        )
        _write_feature_power_diagnostics(
            run_dir, role="combined", record=combined
        )
        _write_candidate_csv(run_dir, "panel_b", rows_b)
        _write_candidate_csv(run_dir, "combined", combined_rows)
    confirmation = confirm_dynkin_design(nomination, rows_b, combined_rows)
    atomic_write_json(run_dir / "sealed_design_confirmation.json", confirmation)
    selected = confirmation.get("selected")
    selected_record = {
        "schema": RUN_SCHEMA + "-selected-design",
        "schema_version": 1,
        "selection": confirmation,
        "selected": selected,
        "selected_design_frozen": 1,
        **NO_WORK,
    }
    _freeze(run_dir / "selected_dynkin_design.json", selected_record)
    selected_hash = file_fingerprint(run_dir / "selected_dynkin_design.json")

    executed = [panel_a] + ([] if panel_b is None else [panel_b])
    executions = [dict(value["execution"]) for value in executed]
    transition_count = sum(int(value.get("transition_count", 0)) for value in executions)
    certified_count = sum(int(value.get("certified_count", 0)) for value in executions)
    fallback_count = sum(int(value.get("fallback_count", 0)) for value in executions)
    elapsed = sum(float(value.get("elapsed_seconds", 0.0)) for value in executions)
    fallback_elapsed = sum(
        float(value.get("fallback_elapsed_seconds", 0.0)) for value in executions
    )
    minimum_rate = min(
        (float(value.get("conservative_rate", 0.0)) for value in executions),
        default=0.0,
    )
    all_rows = [rows_a] + ([] if rows_b is None else [rows_b])
    resource_feasible = {
        (
            int(row["main_paths"]),
            int(row["reference_paths"]),
        )
        for rows in all_rows
        for row in rows
        if (
            float(row["projected_hours"]) <= thresholds.maximum_projected_hours
            and float(row["conservative_rate"]) >= thresholds.minimum_rate
        )
    }
    selected_main = int(selected["main_paths"]) if isinstance(selected, Mapping) else -1
    selected_reference = (
        int(selected["reference_paths"]) if isinstance(selected, Mapping) else -1
    )

    def selected_row(
        rows: Sequence[Mapping[str, Any]] | None,
        nomination_row: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        target = selected if isinstance(selected, Mapping) else nomination_row
        if not isinstance(target, Mapping) or rows is None:
            return None
        key = (int(target["main_paths"]), int(target["reference_paths"]))
        return next(
            (
                row
                for row in rows
                if (int(row["main_paths"]), int(row["reference_paths"])) == key
            ),
            None,
        )

    nominated = nomination.get("selected")
    row_a = selected_row(rows_a, nominated if isinstance(nominated, Mapping) else None)
    row_b = selected_row(rows_b, nominated if isinstance(nominated, Mapping) else None)
    row_combined = selected_row(
        combined_rows, nominated if isinstance(nominated, Mapping) else None
    )

    def width(row: Mapping[str, Any] | None, family: str) -> float:
        return (
            float(row[f"predicted_{family}_half_width"])
            if isinstance(row, Mapping)
            else math.inf
        )

    maximum_error = max(
        float(value["maximum_cumulative_standardized_error"]) for value in executed
    )
    metrics = {
        "production_authorizing_pass": int(
            not args.test_only_reduced_workload
        ),
        "panel_a_frozen_pass": int((run_dir / "panel_a_plan.json").is_file()),
        "panel_b_frozen_pass": int((run_dir / "panel_b_plan.json").is_file()),
        "panel_plan_hash_pass": int(panel_plan_binding_pass),
        "panel_disjoint_pass": 1,
        "panel_nonregeneration_pass": 1,
        "pilot_production_disjoint_pass": int(
            set(path_id_plan.pilot_path_ids("a")).isdisjoint(
                path_id_plan.designated_production_path_ids
            )
            and set(path_id_plan.pilot_path_ids("b")).isdisjoint(
                path_id_plan.designated_production_path_ids
            )
        ),
        "right_endpoint_coupling_unchanged_pass": int(
            path_id_evidence.get("path_id_plan_sha256") == path_id_plan.sha256
            and path_id_evidence.get("passed") == 1
            and path_id_evidence.get("checks", {}).get(
                "right_endpoint_alias_pass"
            )
            == 1
            and path_id_evidence.get("checks", {}).get(
                "cross_level_alias_plan_pass"
            )
            == 1
        ),
        "raw_observables_advisory_only_pass": 1,
        "dynkin_authorizing_estimator_pass": 1,
        "forecast_label_pass": 1,
        "panel_a_complete_pass": int(panel_a["complete"]),
        "panel_b_complete_pass": int(panel_b is not None and panel_b["complete"]),
        "combined_complete_pass": int(panel_b is not None),
        "panel_a_nomination_pass": int(nomination.get("passed", 0)),
        "panel_b_confirmation_pass": int(
            confirmation.get("panel_b_confirmation_pass", 0)
        ),
        "combined_confirmation_pass": int(
            confirmation.get("combined_confirmation_pass", 0)
        ),
        "selected_design_frozen_pass": 1,
        "selected_design_hash_pass": int(len(selected_hash) == 64),
        "complete_candidate_grid_pass": int(len(rows_a) == 4),
        "shard_chain_pass": int(
            all(int(value.get("shard_chain_pass", 0)) == 1 for value in executions)
        ),
        "mass_conservation_pass": int(
            all(
                float(value.get("mass_error", math.inf))
                <= thresholds.maximum_cuda_mass_error
                for value in executions
            )
        ),
        "state_updates_device_resident_pass": int(
            all(
                int(value.get("state_updates_device_resident_pass", 0)) == 1
                for value in executions
            )
        ),
        "pilot_certification_pass": int(
            transition_count > 0 and certified_count == transition_count
        ),
        "executed_panels_numerically_valid_pass": int(
            all(int(value.get("finite", 0)) == 1 for value in executed)
            and maximum_error <= thresholds.maximum_cumulative_standardized_error
        ),
        "candidate_resource_feasibility_pass": int(bool(resource_feasible)),
        "resource_feasible_candidate_count": len(resource_feasible),
        "panel_count": 2,
        "paths_per_panel": int(args.pilot_panel_paths),
        "levels": list(args.sample_steps),
        "candidate_main_paths": list(thresholds.candidate_main_paths),
        "candidate_reference_paths": list(thresholds.candidate_reference_paths),
        "selected_main_paths": selected_main,
        "selected_reference_paths": selected_reference,
        "panel_a_main_half_width": width(row_a, "main"),
        "panel_a_reference_half_width": width(row_a, "reference"),
        "panel_b_main_half_width": width(row_b, "main"),
        "panel_b_reference_half_width": width(row_b, "reference"),
        "combined_main_half_width": width(row_combined, "main"),
        "combined_reference_half_width": width(row_combined, "reference"),
        "projected_production_hours": (
            float(selected["projected_hours"])
            if isinstance(selected, Mapping)
            else math.inf
        ),
        "minimum_rate": minimum_rate,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "fallback_fraction": (
            fallback_count / transition_count if transition_count else 0.0
        ),
        "fallback_cost_fraction": (
            fallback_elapsed / elapsed if elapsed > 0.0 else 0.0
        ),
        "peak_memory_fraction": max(
            (float(value.get("peak_memory_fraction", 0.0)) for value in executions),
            default=0.0,
        ),
        "mass_error": max(
            (float(value.get("mass_error", math.inf)) for value in executions),
            default=math.inf,
        ),
        "maximum_cumulative_standardized_error": maximum_error,
        **{
            name: sum(int(value.get(name, 0)) for value in executions)
            for name in _FORBIDDEN_COUNTS
        },
        **NO_WORK,
    }
    power_record = {
        "schema": RUN_SCHEMA + "-power-metrics",
        "schema_version": 1,
        "metrics": metrics,
        "panel_a_nomination": nomination,
        "sealed_confirmation": confirmation,
        "selected_design_sha256": selected_hash,
        "path_id_plan_version": path_id_plan.version,
        "path_id_plan_sha256": path_id_plan.sha256,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "power_metrics.json", power_record)
    gate = evaluate_dynkin_power(metrics)
    atomic_write_json(existing, gate)
    return gate


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    resumed = False
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"Jacobi Dynkin power run directory: {run_dir}")
        if resumed:
            _verify_terminal_registry(run_dir, stage=args.stage)
        parent_dir = Path(args.parent_strang_run_dir).resolve()
        provenance = verify_raw_endpoint_power_infeasible_parent(parent_dir)
        path_id_plan = _build_path_id_plan(args)
        source_fingerprint_value, source_paths = _source_record(parent_dir)
        config = _scientific_config(args, path_id_plan)
        config_sha = config_fingerprint(config)
        parent_source = _load_source_image(parent_dir)
        if resumed:
            path_id_plan_path = run_dir / "path_id_plan.json"
            if not path_id_plan_path.is_file():
                raise ArtifactCompatibilityError(
                    "resume lacks the corrected frozen path-ID plan"
                )
        else:
            _freeze_path_id_plan(
                run_dir, path_id_plan, require_existing=False
            )
        path_id_plan_file_sha256 = file_fingerprint(
            run_dir / "path_id_plan.json"
        )
        exact_backend = configure_exact_torch_backend(torch.device(args.device))
        manifest = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(args.device),
            "exact_backend": exact_backend,
            "source_fingerprint": source_fingerprint_value,
            "source_paths": source_paths,
            "scientific_config_sha256": config_sha,
            "path_id_plan_version": path_id_plan.version,
            "path_id_plan_sha256": path_id_plan.sha256,
            "path_id_plan_file_sha256": path_id_plan_file_sha256,
            "parent_artifact_record_count": PARENT_REGISTRY_RECORD_COUNT,
            "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
            **NO_WORK,
        }
        if resumed:
            _verify_resume_contract(
                run_dir,
                expected_plan=path_id_plan,
                expected_config=config,
                expected_manifest=manifest,
                expected_provenance=provenance,
                expected_source=parent_source,
                args=args,
            )
        else:
            atomic_write_json(run_dir / "parent_provenance.json", provenance)
        frozen_config = _freeze(
            run_dir / "scientific_config.json",
            config,
            require_existing=resumed,
        )
        if config_fingerprint(frozen_config) != config_sha:
            raise ArtifactCompatibilityError("scientific config hash changed")
        if resumed:
            source = _load_frozen_source(run_dir)
        else:
            source_metadata = {
                key: value
                for key, value in parent_source.items()
                if key not in {"image", "mixed_target"}
            }
            source_npz = run_dir / "source_image.npz"
            _atomic_write_npz(
                source_npz,
                image=parent_source["image"],
                mixed_target=parent_source["mixed_target"],
            )
            source_metadata["source_npz_sha256"] = file_fingerprint(source_npz)
            source_metadata["source_npz_size"] = source_npz.stat().st_size
            _freeze(
                run_dir / "source_image.json",
                source_metadata,
                require_existing=False,
            )
            source = _load_frozen_source(run_dir)
        _freeze(
            run_dir / "run_manifest.json",
            manifest,
            require_existing=resumed,
        )
        _freeze_panel_plans(run_dir, args, path_id_plan)
        _write_status(
            run_dir,
            status="running",
            phase=args.stage,
            required_gate=args.require_gate,
            required_gate_pass=0,
            scientific_config_sha256=config_sha,
            source_fingerprint=source_fingerprint_value,
        )

        preflight = _existing_gate(
            run_dir, "preflight", "preflight has not run"
        )
        pilot = _existing_gate(run_dir, "power", "sealed pilot has not run")
        if args.stage in {"preflight", "all"}:
            try:
                preflight = _run_preflight_stage(
                    run_dir, args, provenance, source
                )
            except ArtifactCompatibilityError as exc:
                preflight = _failed_stage_gate(
                    run_dir,
                    "preflight",
                    exc,
                    failure_domain="numerical_execution",
                    failure_code="preflight_artifact_incompatibility",
                )
            except DynkinSchedulerConfigurationError as exc:
                preflight = _failed_stage_gate(
                    run_dir,
                    "preflight",
                    exc,
                    failure_domain="scheduler_configuration",
                    failure_code="dynkin_path_id_plan_invalid",
                )
            except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                preflight = _failed_stage_gate(
                    run_dir,
                    "preflight",
                    exc,
                    failure_domain="resource_execution",
                    failure_code="preflight_resource_exhausted",
                )
            except Exception as exc:
                preflight = _failed_stage_gate(
                    run_dir,
                    "preflight",
                    exc,
                    failure_domain="numerical_execution",
                    failure_code="preflight_execution_exception",
                )
        if args.stage in {"pilot", "all"}:
            can_run_test_pilot = (
                args.test_only_reduced_workload
                and _substantive_test_gate_passed(
                    preflight, ignored_checks=("production_authorizing_pass",)
                )
            )
            if _passed(preflight) or can_run_test_pilot:
                try:
                    pilot = _run_pilot_stage(run_dir, args, source)
                except ArtifactCompatibilityError as exc:
                    pilot = _failed_stage_gate(
                        run_dir,
                        "power",
                        exc,
                        failure_domain="numerical_execution",
                        failure_code="pilot_artifact_incompatibility",
                    )
                except DynkinSchedulerConfigurationError as exc:
                    pilot = _failed_stage_gate(
                        run_dir,
                        "power",
                        exc,
                        failure_domain="scheduler_execution",
                        failure_code="dynkin_pilot_path_id_plan_invalid",
                    )
                except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                    pilot = _failed_stage_gate(
                        run_dir,
                        "power",
                        exc,
                        failure_domain="resource_execution",
                        failure_code="pilot_resource_exhausted",
                    )
                except Exception as exc:
                    pilot = _failed_stage_gate(
                        run_dir,
                        "power",
                        exc,
                        failure_domain="numerical_execution",
                        failure_code="pilot_execution_exception",
                    )
            else:
                pilot = not_evaluated_gate(
                    "jacobi_rb_dynkin_power", "preflight did not pass"
                )
                atomic_write_json(run_dir / "dynkin_power_gate.json", pilot)
        return _finish(
            run_dir,
            args,
            preflight=preflight,
            pilot=pilot,
            provenance_pass=True,
        )
    except ArtifactCompatibilityError as exc:
        # A resumed run is immutable except for explicitly validated
        # interrupted shard-tail recovery handled above.  Never overwrite its
        # evidence or re-bless it under a mismatched invocation.
        if resumed:
            print(f"Jacobi Dynkin power compatibility error: {exc}", file=sys.stderr)
            return 2
        if run_dir is not None:
            failure = {
                "schema": RUN_SCHEMA + "-provenance-failure",
                "schema_version": 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **NO_WORK,
            }
            atomic_write_json(run_dir / "provenance_failure.json", failure)
            preflight = _synthetic_gate("preflight", passed=False)
            preflight.update(
                provenance_valid=0,
                parent_adjudication_valid=0,
                phase_moment_algebra_valid=0,
                tower_identity_valid=0,
                numerically_valid=0,
                resource_valid=0,
                failure=failure,
            )
            atomic_write_json(run_dir / "dynkin_preflight_gate.json", preflight)
            pilot = not_evaluated_gate(
                "jacobi_rb_dynkin_power", "control provenance is invalid"
            )
            atomic_write_json(run_dir / "dynkin_power_gate.json", pilot)
            return _finish(
                run_dir,
                args,
                preflight=preflight,
                pilot=pilot,
                provenance_pass=False,
            )
        print(f"Jacobi Dynkin power compatibility error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if run_dir is not None:
            failure = {
                "schema": RUN_SCHEMA + "-unexpected-failure",
                "schema_version": 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
                **NO_WORK,
            }
            atomic_write_json(run_dir / "unexpected_failure.json", failure)
            _write_status(
                run_dir,
                status="failed",
                outcome="unexpected_failure",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        print(f"Jacobi Dynkin power error: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
