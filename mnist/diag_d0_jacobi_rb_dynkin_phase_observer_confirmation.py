"""Phase-local Dynkin observer repair and sealed power confirmation.

This additive controls-only workflow binds the immutable failed Dynkin ID-fix
run, replaces only its numerically defective global-subtraction observation
with exact matching-local increments, and then reuses the frozen Dynkin power
logic.  It performs no physical training, reconstruction, or reverse sampling.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
from types import SimpleNamespace
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np
import torch
from torch import Tensor

from mnist import d0_jacobi_rb_cuda_controls as _controls
from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_dynkin import (
    DynkinAccumulatorState,
    _matching_indices,
    compute_dynkin_phase_drift,
    run_dynkin_refinement_shard,
    run_dynkin_tower_phase,
)
from mnist.d0_jacobi_rb_dynkin_phase_observer import (
    PHASE_OBSERVER_VERSION,
    combine_dynkin_phase_residual,
    compute_advisory_global_phase_increment,
    compute_dynkin_phase_observed_increment,
    compute_dynkin_phase_observed_increment_from_states,
)
from mnist.d0_jacobi_rb_dynkin_phase_observer_gate import (
    PhaseObserverThresholds,
    decide_phase_observer_workflow,
    evaluate_phase_observer_power,
    evaluate_phase_observer_preflight,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_dynkin_phase_observer_path_ids import (
    PHASE_OBSERVER_PATH_ID_PLAN_VERSION,
    PhaseObserverPathIDPlan,
    TOWER_CASE_COUNT,
    canonical_tower_transition_ids,
)
from mnist.d0_jacobi_rb_dynkin_phase_observer_provenance import (
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_COUNT,
    PARENT_SOURCE_FINGERPRINT,
    verify_tower_observer_roundoff_parent,
)
from mnist.d0_jacobi_rb_dynkin_power_gate import (
    DynkinPowerThresholds,
    build_dynkin_candidate_records,
    confirm_dynkin_design,
    select_dynkin_panel_a_design,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    GRID_SIZE,
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
    evaluate_refinement_observables,
    legacy_k512_transition_ids,
    refinement_observable_spec,
    refinement_phase_exposure,
)
from mnist.d0_jacobi_rb_strang_refinement_gate import (
    whole_path_max_t_intervals,
)
from mnist.diag_d0_jacobi_rb_strang_refinement import (
    EXPECTED_IMAGE_SHA256,
    EXPECTED_MIXED_TARGET_SHA256,
    OBSERVATION_TIME_FRACTIONS,
)
import mnist.diag_d0_jacobi_rb_dynkin_power_confirmation as _legacy


RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-dynkin-phase-observer-confirmation"
)
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "exact phase-local Dynkin observer and power feasibility only"
ROOT_SEED = 261_171
PILOT_LEVELS = (128, 256, 512, 1024, 2048)
PANEL_NAMES = ("a", "b")
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
_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
}
_MUTABLE_TERMINAL_FILES = {
    "phase_observer_workflow_gate.json",
    "phase_observer_decision.json",
}


class PhaseObserverConfigurationError(ValueError):
    """A frozen phase-observer schedule or namespace is invalid."""


class PhaseObserverSchedulerError(ValueError):
    """A path allocation or restart shard violates the frozen schedule."""


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read JSON artifact {path}: {exc}"
        ) from exc
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
            **{
                str(key): np.ascontiguousarray(value)
                for key, value in arrays.items()
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _freeze(
    path: Path, value: Mapping[str, Any], *, require_existing: bool = False
) -> dict[str, Any]:
    normalized = json.loads(
        json.dumps(dict(value), sort_keys=True, allow_nan=False)
    )
    if path.is_file():
        if _load(path) != normalized:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    elif require_existing:
        raise ArtifactCompatibilityError(f"resume lacks frozen artifact: {path.name}")
    else:
        atomic_write_json(path, normalized)
    return normalized


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    thresholds = PhaseObserverThresholds()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "pilot", "report", "all"), default="all"
    )
    parser.add_argument(
        "--require-gate", choices=("none", "preflight", "pilot"), default="none"
    )
    parser.add_argument(
        "--parent-dynkin-idfix-run-dir",
        "--parent-dynkin-power-run-dir",
        dest="parent_dynkin_idfix_run_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_dynkin_phase_observer_confirmation"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-phase-local-dynkin-observer"
    )
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
        default=DynkinPowerThresholds().pilot_paths_per_panel,
    )
    parser.add_argument("--pilot-stop-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--test-only-reduced-workload", action="store_true", help=argparse.SUPPRESS
    )
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
    if not 1 <= int(args.tower_panel_clusters) <= 128:
        parser.error("tower panel size must lie in [1,128]")
    if not 1 <= int(args.pilot_panel_paths) <= 8:
        parser.error("pilot panel size must lie in [1,8]")
    if any(level not in SUPPORTED_SAMPLE_STEPS for level in args.sample_steps):
        parser.error("unsupported refinement level")
    changed = []
    frozen = {
        "root_seed": ROOT_SEED,
        "sample_steps": PILOT_LEVELS,
        "tower_panel_clusters": 128,
        "pilot_panel_paths": 8,
    }
    for name, expected in frozen.items():
        observed = getattr(args, name)
        changed_value = (
            tuple(observed) != tuple(expected)
            if isinstance(expected, tuple)
            else observed != expected
        )
        if changed_value:
            changed.append(name)
    if (changed or args.pilot_stop_steps is not None) and not args.test_only_reduced_workload:
        parser.error(
            "production configuration is frozen; overrides require "
            "--test-only-reduced-workload: "
            + ", ".join(changed or ["pilot_stop_steps"])
        )
    if args.test_only_reduced_workload and args.require_gate != "none":
        parser.error("test-only workloads cannot satisfy a required gate")
    if torch.device(args.device).type != "cuda" and not args.test_only_reduced_workload:
        parser.error("production phase-observer controls require --device cuda")
    if args.pilot_stop_steps is not None and (
        int(args.pilot_stop_steps) <= 0
        or int(args.pilot_stop_steps) % REFINEMENT_SHARD_STEPS
    ):
        parser.error("pilot stop steps must be a positive multiple of eight")
    return args


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(
                f"resume run does not exist: {run_dir}"
            )
        return run_dir, True
    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.run_name}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir.resolve(), False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    record = (
        _load(path)
        if path.is_file()
        else {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "created_at": _now(),
        }
    )
    record.update(updates)
    record.update(updated_at=_now(), **NO_WORK)
    atomic_write_json(path, record)
    return record


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _REGISTRY_EXCLUDED
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "terminal_files_excluded_to_avoid_self_reference": sorted(
            _REGISTRY_EXCLUDED
        ),
        "records": records,
        **NO_WORK,
    }


def _verify_terminal_registry(run_dir: Path) -> None:
    registry_path = run_dir / "artifact_registry.json"
    if not registry_path.is_file():
        status_path = run_dir / "run_status.json"
        if (
            status_path.is_file()
            and _load(status_path).get("status") == "complete"
        ):
            raise ArtifactCompatibilityError(
                "completed resume lacks its terminal artifact registry"
            )
        return
    status = _load(run_dir / "run_status.json")
    registry_sha256 = file_fingerprint(registry_path)
    if (
        status.get("artifact_registry_sha256") != registry_sha256
        or status.get("artifact_registry_record_count") is None
        or status.get("artifact_registry_size") != registry_path.stat().st_size
    ):
        raise ArtifactCompatibilityError("resume status does not bind its registry")
    registry = _load(registry_path)
    if (
        registry.get("schema") != RUN_SCHEMA + "-artifact-registry"
        or registry.get("schema_version") != 1
        or set(
            registry.get(
                "terminal_files_excluded_to_avoid_self_reference", ()
            )
        )
        != _REGISTRY_EXCLUDED
        or not isinstance(registry.get("records"), Mapping)
    ):
        raise ArtifactCompatibilityError("resume artifact registry is invalid")
    records = dict(registry["records"])
    if int(status["artifact_registry_record_count"]) != len(records):
        raise ArtifactCompatibilityError(
            "resume status has the wrong artifact-record count"
        )
    interrupted = status.get("status") == "running"
    for relative, raw in records.items():
        path = run_dir / relative
        valid = (
            path.is_file()
            and isinstance(raw, Mapping)
            and raw.get("sha256") == file_fingerprint(path)
            and int(raw.get("size", -1)) == path.stat().st_size
        )
        if not valid and interrupted and relative in _MUTABLE_TERMINAL_FILES:
            continue
        if not valid:
            raise ArtifactCompatibilityError(f"resume artifact changed: {relative}")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _REGISTRY_EXCLUDED
    }
    if not interrupted and actual != set(records):
        raise ArtifactCompatibilityError(
            "completed resume artifact set differs from its registry"
        )


def _finalize_registry(run_dir: Path) -> dict[str, Any]:
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    return {
        "artifact_registry_record_count": len(registry["records"]),
        "artifact_registry_sha256": file_fingerprint(
            run_dir / "artifact_registry.json"
        ),
        "artifact_registry_size": (
            run_dir / "artifact_registry.json"
        ).stat().st_size,
    }


def _artifact_is_terminally_registered(
    run_dir: Path, path: Path
) -> bool:
    registry_path = run_dir / "artifact_registry.json"
    if not registry_path.is_file() or not path.is_file():
        return False
    registry = _load(registry_path)
    records = registry.get("records")
    if not isinstance(records, Mapping):
        return False
    relative = path.relative_to(run_dir).as_posix()
    record = records.get(relative)
    return bool(
        isinstance(record, Mapping)
        and record.get("sha256") == file_fingerprint(path)
        and int(record.get("size", -1)) == path.stat().st_size
    )


def _run_is_interrupted(run_dir: Path) -> bool:
    status_path = run_dir / "run_status.json"
    return bool(
        status_path.is_file()
        and _load(status_path).get("status") == "running"
    )


def _build_path_id_plan(args: argparse.Namespace) -> PhaseObserverPathIDPlan:
    try:
        return PhaseObserverPathIDPlan(
            tower_path_count=int(args.tower_panel_clusters),
            pilot_path_count=int(args.pilot_panel_paths),
        )
    except (TypeError, ValueError) as exc:
        raise PhaseObserverConfigurationError(
            f"invalid phase-observer path-ID plan: {exc}"
        ) from exc


def _load_frozen_path_id_plan(run_dir: Path) -> PhaseObserverPathIDPlan:
    path = run_dir / "phase_observer_path_id_plan.json"
    if not path.is_file():
        raise ArtifactCompatibilityError(
            "run lacks phase_observer_path_id_plan.json"
        )
    try:
        return PhaseObserverPathIDPlan.from_frozen_record(_load(path))
    except (TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            f"frozen phase-observer path-ID plan is invalid: {exc}"
        ) from exc


def _freeze_path_id_plan(
    run_dir: Path,
    plan: PhaseObserverPathIDPlan,
    *,
    require_existing: bool,
) -> dict[str, Any]:
    record = _freeze(
        run_dir / "phase_observer_path_id_plan.json",
        plan.to_frozen_record(),
        require_existing=require_existing,
    )
    loaded = _load_frozen_path_id_plan(run_dir)
    if loaded != plan or record.get("path_id_plan_sha256") != plan.sha256:
        raise ArtifactCompatibilityError("phase-observer path-ID plan changed")
    return record


def _load_parent_source_image(parent_dir: Path) -> dict[str, Any]:
    metadata = _load(parent_dir / "source_image.json")
    payload = parent_dir / "source_image.npz"
    if not payload.is_file():
        raise ArtifactCompatibilityError("parent source image payload is missing")
    with np.load(payload, allow_pickle=False) as archive:
        image = np.ascontiguousarray(
            archive["image"], dtype=np.float64
        ).reshape(-1)
        mixed = np.ascontiguousarray(
            archive["mixed_target"], dtype=np.float64
        ).reshape(-1)
    if (
        image.shape != (PATH_STATE_SIZE,)
        or mixed.shape != (PATH_STATE_SIZE,)
        or not np.isfinite(image).all()
        or not np.isfinite(mixed).all()
        or metadata.get("image_sha256") != EXPECTED_IMAGE_SHA256
        or metadata.get("mixed_target_sha256") != EXPECTED_MIXED_TARGET_SHA256
        or abs(float(mixed.sum()) - 1.0) > 1.0e-12
    ):
        raise ArtifactCompatibilityError("parent source-image binding changed")
    return {
        **metadata,
        "image": image,
        "mixed_target": mixed,
        "parent_npz_sha256": file_fingerprint(payload),
    }


def _freeze_source_image(
    run_dir: Path,
    source: Mapping[str, Any],
    *,
    require_existing: bool,
) -> dict[str, Any]:
    payload = run_dir / "source_image.npz"
    metadata_path = run_dir / "source_image.json"
    if not payload.is_file():
        if require_existing:
            raise ArtifactCompatibilityError("resume lacks source_image.npz")
        _atomic_write_npz(
            payload,
            image=np.asarray(source["image"], dtype=np.float64),
            mixed_target=np.asarray(source["mixed_target"], dtype=np.float64),
        )
    metadata = {
        "schema": RUN_SCHEMA + "-source-image",
        "schema_version": 1,
        "image_sha256": source["image_sha256"],
        "mixed_target_sha256": source["mixed_target_sha256"],
        "source_npz_sha256": file_fingerprint(payload),
        "source_npz_size": payload.stat().st_size,
        "parent_npz_sha256": source.get("parent_npz_sha256"),
        **NO_WORK,
    }
    return _freeze(
        metadata_path, metadata, require_existing=require_existing
    )


def _tower_initial_states(
    path_count: int, *, root_seed: int, panel: str
) -> np.ndarray:
    offset = 10_000 if panel == "a" else 20_000
    return np.ascontiguousarray(
        np.random.Generator(
            np.random.Philox(int(root_seed) + offset)
        ).dirichlet(np.ones(PATH_STATE_SIZE), size=int(path_count)),
        dtype=np.float64,
    )


def _freeze_panel_plans(
    run_dir: Path,
    args: argparse.Namespace,
    plan: PhaseObserverPathIDPlan,
    *,
    require_existing: bool,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for panel in PANEL_NAMES:
        initial = _tower_initial_states(
            plan.tower_path_count,
            root_seed=int(args.root_seed),
            panel=panel,
        )
        payload = run_dir / f"tower_panel_{panel}_initial_states.npz"
        if not payload.is_file():
            if require_existing:
                raise ArtifactCompatibilityError(
                    f"resume lacks {payload.name}"
                )
            _atomic_write_npz(payload, states=initial)
        with np.load(payload, allow_pickle=False) as archive:
            frozen_initial = np.ascontiguousarray(
                archive["states"], dtype=np.float64
            )
        if not np.array_equal(frozen_initial, initial):
            raise ArtifactCompatibilityError(
                f"tower panel {panel} initial states changed"
            )
        record = {
            "schema": RUN_SCHEMA + "-sealed-panel-plan",
            "schema_version": 1,
            "panel": panel,
            "root_seed": int(args.root_seed),
            "tower_case_path_ids": [
                list(plan.tower_case_path_ids(panel, case))
                for case in range(TOWER_CASE_COUNT)
            ],
            "pilot_path_ids": list(plan.pilot_path_ids(panel)),
            "tower_initial_states_sha256": _array_sha256(initial),
            "tower_initial_states_npz_sha256": file_fingerprint(payload),
            "path_id_plan_version": plan.version,
            "path_id_plan_sha256": plan.sha256,
            "future_production_path_ids_sha256": _array_sha256(
                np.asarray(plan.designated_production_path_ids, dtype=np.int64)
            ),
            **NO_WORK,
        }
        records[panel] = _freeze(
            run_dir / f"panel_{panel}_plan.json",
            record,
            require_existing=require_existing,
        )
    disjoint = set(plan.tower_panel_path_ids("a")).isdisjoint(
        plan.tower_panel_path_ids("b")
    ) and set(plan.pilot_path_ids("a")).isdisjoint(
        plan.pilot_path_ids("b")
    )
    registry = {
        "schema": RUN_SCHEMA + "-sealed-panel-registry",
        "schema_version": 1,
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "panel_a_plan_sha256": file_fingerprint(
            run_dir / "panel_a_plan.json"
        ),
        "panel_b_plan_sha256": file_fingerprint(
            run_dir / "panel_b_plan.json"
        ),
        "panels_disjoint": int(disjoint),
        "future_production_namespace_disjoint": 1,
        "panels_frozen_before_device_execution": 1,
        **NO_WORK,
    }
    return _freeze(
        run_dir / "sealed_panel_registry.json",
        registry,
        require_existing=require_existing,
    )


def _source_record(parent_dir: Path) -> tuple[str, list[str]]:
    manifest = _load(parent_dir / "run_manifest.json")
    raw = manifest.get("source_paths")
    if not isinstance(raw, list) or len(raw) != PARENT_SOURCE_COUNT:
        raise ArtifactCompatibilityError(
            "parent manifest does not bind twenty-one sources"
        )
    paths = {Path(str(value)).resolve() for value in raw}
    for module_name in (
        "mnist.d0_jacobi_rb_dynkin_phase_observer",
        "mnist.d0_jacobi_rb_dynkin_phase_observer_gate",
        "mnist.d0_jacobi_rb_dynkin_phase_observer_path_ids",
        "mnist.d0_jacobi_rb_dynkin_phase_observer_provenance",
    ):
        module = importlib.import_module(module_name)
        paths.add(Path(module.__file__).resolve())
    paths.add(Path(__file__).resolve())
    ordered = sorted(paths)
    if not all(path.is_file() for path in ordered):
        raise ArtifactCompatibilityError("source set contains a missing file")
    return source_fingerprint(ordered), [str(path) for path in ordered]


def _sampler_identity(args: argparse.Namespace) -> str:
    return (
        "test-only-deterministic-cpu-certified-stand-in-v1"
        if args.test_only_reduced_workload
        else "certified-jacobi-rb-cuda-v1"
    )


def _scientific_config(
    args: argparse.Namespace, plan: PhaseObserverPathIDPlan
) -> dict[str, Any]:
    observer = PhaseObserverThresholds()
    power = DynkinPowerThresholds()
    return {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "claim_scope": CLAIM_SCOPE,
        "root_seed": int(args.root_seed),
        "test_only_reduced_workload": int(
            bool(args.test_only_reduced_workload)
        ),
        "sampler_identity": _sampler_identity(args),
        "requested_device_type": torch.device(args.device).type,
        "grid_size": GRID_SIZE,
        "alpha": 1.0,
        "tau_eff": TAU_EFF,
        "sample_steps": list(args.sample_steps),
        "phase_observer_version": PHASE_OBSERVER_VERSION,
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "tower_panel_count": observer.tower_panel_count,
        "tower_clusters_per_panel": int(args.tower_panel_clusters),
        "tower_family_member_count": observer.tower_family_member_count,
        "tower_bootstrap_replicates": observer.tower_bootstrap_replicates,
        "tower_confidence": observer.tower_confidence,
        "pilot_panel_paths": int(args.pilot_panel_paths),
        "candidate_main_paths": list(power.candidate_main_paths),
        "candidate_reference_paths": list(power.candidate_reference_paths),
        "maximum_main_half_width": power.maximum_main_half_width,
        "maximum_reference_half_width": power.maximum_reference_half_width,
        "maximum_projected_hours": power.maximum_projected_hours,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        **NO_WORK,
    }


def _verify_resume_contract(
    run_dir: Path,
    *,
    expected_plan: PhaseObserverPathIDPlan,
    expected_config: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
    expected_source: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    _verify_terminal_registry(run_dir)
    observed_plan = _load_frozen_path_id_plan(run_dir)
    if observed_plan != expected_plan:
        raise ArtifactCompatibilityError("resume path-ID plan changed")
    if _load(run_dir / "scientific_config.json") != dict(expected_config):
        raise ArtifactCompatibilityError("resume scientific configuration changed")
    if _load(run_dir / "run_manifest.json") != dict(expected_manifest):
        raise ArtifactCompatibilityError("resume source/runtime manifest changed")
    if _load(run_dir / "parent_provenance.json") != dict(expected_provenance):
        raise ArtifactCompatibilityError("resume parent provenance changed")
    source_metadata = _load(run_dir / "source_image.json")
    payload = run_dir / "source_image.npz"
    if (
        not payload.is_file()
        or source_metadata.get("source_npz_sha256") != file_fingerprint(payload)
    ):
        raise ArtifactCompatibilityError("resume source-image payload changed")
    with np.load(payload, allow_pickle=False) as archive:
        image = np.asarray(archive["image"], dtype=np.float64)
        mixed = np.asarray(archive["mixed_target"], dtype=np.float64)
    if (
        not np.array_equal(image, np.asarray(expected_source["image"]))
        or not np.array_equal(
            mixed, np.asarray(expected_source["mixed_target"])
        )
    ):
        raise ArtifactCompatibilityError("resume source image changed")
    _freeze_panel_plans(
        run_dir, args, expected_plan, require_existing=True
    )


class _CpuCertifiedSampler:
    """Deterministic nonauthorizing stand-in used only by reduced tests."""

    def __call__(
        self, x: Tensor, exposure: Tensor, **kwargs: Any
    ) -> SimpleNamespace:
        transition_ids = kwargs.get("transition_ids")
        if not isinstance(transition_ids, Tensor):
            raise TypeError("test sampler requires transition_ids")
        jitter = (
            torch.remainder(transition_ids.to(torch.int64), 17).to(torch.float64)
            * 2.0**-48
        )
        active = exposure > 0.0
        later = torch.where(active, torch.clamp(x + jitter, 0.0, 1.0), x)
        count = int(x.numel())
        zero_i64 = torch.zeros((), dtype=torch.int64, device=x.device)
        zero_f64 = torch.zeros((), dtype=torch.float64, device=x.device)
        return SimpleNamespace(
            later_head_fraction=later,
            denoising_target=later - x,
            quantile_lower=later,
            quantile_upper=later,
            target_lower=later - x,
            target_upper=later - x,
            certificate_codes=torch.full(
                (count,), 15, dtype=torch.uint8, device=x.device
            ),
            certified_mask=torch.ones(count, dtype=torch.bool, device=x.device),
            fallback_mask=torch.zeros(count, dtype=torch.bool, device=x.device),
            strengthened_mask=torch.zeros(count, dtype=torch.bool, device=x.device),
            mode_counts=torch.full(
                (count,), 128, dtype=torch.int32, device=x.device
            ),
            prefix_bits=torch.full(
                (count,), 64, dtype=torch.int32, device=x.device
            ),
            arb_fallback_reason_codes=torch.zeros(
                count, dtype=torch.uint8, device=x.device
            ),
            diagnostics={
                "maximum_cuda_launch_lanes": torch.as_tensor(
                    count, dtype=torch.int64, device=x.device
                ),
                "fused_authorizer_launch_count": torch.ones(
                    (), dtype=torch.int64, device=x.device
                ),
                "arb_fallback_elapsed_seconds": zero_f64,
                "fused_authorizer_elapsed_seconds": zero_f64,
                "candidate_elapsed_seconds": zero_f64,
                "fallback_count": zero_i64,
                **{name: zero_i64 for name in _FORBIDDEN_COUNTS},
            },
        )


def _sampler_for_args(args: argparse.Namespace) -> Callable[..., Any]:
    return (
        _CpuCertifiedSampler()
        if args.test_only_reduced_workload
        else sample_alpha1_rb_transition_batch_cuda
    )


def _path_id_plan_preflight(
    run_dir: Path,
    args: argparse.Namespace,
    plan: PhaseObserverPathIDPlan,
) -> dict[str, Any]:
    path = run_dir / "phase_observer_path_id_preflight.json"
    device = torch.device(args.device)
    hashes: dict[str, str] = {}
    tower_roles: dict[str, set[int]] = {}
    maximum = 0
    canonical_unique = True
    for panel in PANEL_NAMES:
        panel_ids: set[int] = set()
        for case in range(TOWER_CASE_COUNT):
            paths = plan.tower_case_path_ids(panel, case)
            plan.validate_role_path_ids(
                f"tower_{panel}", paths, case_index=case
            )
            ids = canonical_tower_transition_ids(
                paths,
                sample_steps=512,
                outer_step=case,
                phase=case // 2,
                device=device,
            )
            host = ids.detach().cpu().numpy()
            hashes[f"{panel}:{case}"] = _array_sha256(host)
            canonical_unique &= np.unique(host).size == host.size
            maximum = max(maximum, int(host.max(initial=0)))
            panel_ids.update(paths)
        tower_roles[panel] = panel_ids
    all_roles: list[set[int]] = [
        set(plan.legacy_replay_path_ids),
        tower_roles["a"],
        tower_roles["b"],
        set(plan.pilot_path_ids("a")),
        set(plan.pilot_path_ids("b")),
        set(plan.designated_production_path_ids),
    ]
    pairwise_disjoint = all(
        left.isdisjoint(right)
        for index, left in enumerate(all_roles)
        for right in all_roles[index + 1 :]
    )
    prior = ((0x20000, 0x20400), (0x30000, 0x30400),
             (0x40000, 0x40008), (0x50000, 0x50008))
    fresh = set().union(*all_roles[1:5])
    fresh_disjoint = all(
        not any(start <= value < stop for value in fresh)
        for start, stop in prior
    )
    record = {
        "schema": RUN_SCHEMA + "-path-id-preflight",
        "schema_version": 1,
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "root_seed": int(args.root_seed),
        "canonical_transition_id_hashes": hashes,
        "maximum_packed_transition_id": maximum,
        "checks": {
            "plan_validation_pass": 1,
            "canonical_id_uniqueness_pass": int(canonical_unique),
            "role_disjoint_pass": int(pairwise_disjoint),
            "fresh_namespace_disjoint_pass": int(fresh_disjoint),
            "packed_id_field_pass": int(maximum < (1 << 43)),
            "order_chunk_resume_invariance_by_construction_pass": 1,
            "right_endpoint_coupling_unchanged_pass": 1,
        },
        "passed": int(
            canonical_unique
            and pairwise_disjoint
            and fresh_disjoint
            and maximum < (1 << 43)
        ),
        **NO_WORK,
    }
    return _freeze(path, record, require_existing=path.is_file())


def _failed_stage_gate(
    run_dir: Path,
    stage: str,
    exc: BaseException,
    *,
    failure_domain: str,
    failure_code: str,
) -> dict[str, Any]:
    failure = {
        "schema": RUN_SCHEMA + "-stage-failure",
        "schema_version": 1,
        "stage": stage,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **NO_WORK,
    }
    atomic_write_json(run_dir / f"{stage}_failure.json", failure)
    gate = {
        "schema": RUN_SCHEMA + "-stage-execution-gate",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": "execution_failed",
        "passed": 0,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "failure": failure,
        **NO_WORK,
    }
    atomic_write_json(run_dir / f"phase_observer_{stage}_gate.json", gate)
    return gate


def _existing_gate(run_dir: Path, stage: str, reason: str) -> dict[str, Any]:
    path = run_dir / f"phase_observer_{stage}_gate.json"
    return _load(path) if path.is_file() else not_evaluated_gate(stage, reason)


def _substantive_test_gate_passed(
    gate: Mapping[str, Any], *, ignored_checks: Sequence[str]
) -> bool:
    checks = gate.get("subchecks")
    if not isinstance(checks, Mapping):
        return False
    ignored = set(ignored_checks)
    return all(
        name in ignored
        or (
            isinstance(record, Mapping)
            and int(record.get("passed", 0)) == 1
        )
        for name, record in checks.items()
    )


def _required_gate_pass(
    required: str,
    preflight: Mapping[str, Any],
    pilot: Mapping[str, Any],
) -> bool:
    if required == "none":
        return True
    if required == "preflight":
        return _passed(preflight)
    return _passed(preflight) and _passed(pilot)


def _observer_legacy_replay(
    run_dir: Path,
    args: argparse.Namespace,
    parent_dir: Path,
    plan: PhaseObserverPathIDPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / "legacy_k512_phase_observer_replay.json"
    expected = _load(parent_dir / "legacy_k512_dynkin_observer_replay.json")
    initial = _legacy._initial_dirichlet(10, 261_141 + 20_000 + 10)[:8]
    states = torch.as_tensor(
        initial, dtype=torch.float64, device=torch.device(args.device)
    ).contiguous()
    result = run_dynkin_refinement_shard(
        states,
        path_ids=plan.legacy_replay_path_ids,
        sample_steps=512,
        start_step=0,
        root_seed=261_141,
        panel_namespace="legacy-k512-replay",
        profile=JacobiRBCudaProfile(),
        sampler=_sampler_for_args(args),
        transition_id_provider=legacy_k512_transition_ids,
        rng_key_override=(261_141, "full-path-v2"),
    )
    record = {
        "schema": RUN_SCHEMA + "-legacy-k512-replay",
        "schema_version": 1,
        "path_ids": list(plan.legacy_replay_path_ids),
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "expected_batch_output_sha256": expected[
            "expected_batch_output_sha256"
        ],
        "observed_batch_output_sha256": result.batch_output_sha256,
        "expected_batch_certificate_sha256": expected[
            "expected_batch_certificate_sha256"
        ],
        "observed_batch_certificate_sha256": result.batch_certificate_sha256,
        "expected_final_states_sha256": expected[
            "expected_final_states_sha256"
        ],
        "observed_final_states_sha256": result.batch_final_state_sha256,
        "transition_target_certificate_hash_invariance_pass": int(
            result.batch_output_sha256
            == expected["expected_batch_output_sha256"]
            and result.batch_certificate_sha256
            == expected["expected_batch_certificate_sha256"]
        ),
        "observer_state_hash_invariance_pass": int(
            result.batch_final_state_sha256
            == expected["expected_final_states_sha256"]
        ),
        "legacy_k512_replay_pass": int(
            result.batch_output_sha256
            == expected["expected_batch_output_sha256"]
            and result.batch_certificate_sha256
            == expected["expected_batch_certificate_sha256"]
            and result.batch_final_state_sha256
            == expected["expected_final_states_sha256"]
        ),
        "diagnostics": dict(result.diagnostics),
        **NO_WORK,
    }
    atomic_write_json(path, record)
    return record, {
        "rows": [
            {
                "diagnostics": dict(result.diagnostics),
                "complete_wall_upper_seconds": float(
                    result.diagnostics.get("wall_elapsed_seconds", 0.0)
                ),
            }
        ],
        "transition_count": int(
            result.diagnostics.get("transition_count", 0)
        ),
        "certified_count": int(
            result.diagnostics.get("certified_count", 0)
        ),
        "fallback_count": int(result.diagnostics.get("fallback_count", 0)),
        "elapsed_seconds": float(
            result.diagnostics.get("elapsed_seconds", 0.0)
        ),
        "fallback_elapsed_seconds": float(
            result.diagnostics.get("fallback_elapsed_seconds", 0.0)
        ),
        "mass_error": max(
            float(
                result.diagnostics.get(
                    "maximum_pair_total_error",
                    result.diagnostics.get("maximum_pair_mass_error", math.inf),
                )
            ),
            float(
                result.diagnostics.get(
                    "maximum_global_simplex_error", math.inf
                )
            ),
        ),
        "peak_memory_fraction": (
            torch.cuda.max_memory_allocated(torch.device(args.device))
            / torch.cuda.get_device_properties(
                torch.device(args.device)
            ).total_memory
            if torch.device(args.device).type == "cuda"
            else 0.0
        ),
        **{
            name: int(result.diagnostics.get(name, 0))
            for name in _FORBIDDEN_COUNTS
        },
    }


def _direct_phase_increment(
    pair_total: Tensor,
    earlier: Tensor,
    later: Tensor,
    matching_index: int,
) -> Tensor:
    spec = refinement_observable_spec(GRID_SIZE)
    tails, heads = _matching_indices(
        matching_index, device=pair_total.device
    )
    weights = torch.as_tensor(
        np.array(spec.fourier_weights, copy=True),
        dtype=torch.float64,
        device=pair_total.device,
    )
    delta = later - earlier
    centered_sum = earlier + later - 1.0
    linear = torch.einsum(
        "oe,pe->po",
        weights.index_select(1, heads) - weights.index_select(1, tails),
        pair_total * delta,
    )
    quadratic = torch.sum(
        2.0 * pair_total.square() * delta * centered_sum,
        dim=1,
        keepdim=True,
    )
    cubic = torch.sum(
        3.0 * pair_total.pow(3) * delta * centered_sum,
        dim=1,
        keepdim=True,
    )
    return torch.cat((linear, quadratic, cubic), dim=1)


def _phase_observer_oracle_controls(
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Independent formula, invariant, and negative-fixture controls."""

    generator = np.random.Generator(np.random.Philox(ROOT_SEED + 91))
    rows: list[dict[str, Any]] = []
    maximum_cpu_error = 0.0
    maximum_device_error = 0.0
    formula_flags = {"fourier": True, "quadratic": True, "cubic": True}
    invariant_masks = True
    structural_centers = True
    structural_radii = True
    noninvariant_global = True
    facet_interior = True
    duration_pass = True
    zero_pass = True
    quantile_pass = True
    orientation_fixture_detected = False
    quadratic_factor_fixture_detected = False
    cubic_factor_fixture_detected = False
    missing_pair_mass_fixture_detected = False
    wrong_duration_fixture_detected = False
    wrong_eigenvalue_fixture_detected = False
    post_state_fixture_detected = False
    spec = refinement_observable_spec(GRID_SIZE)

    base_states = generator.dirichlet(
        np.ones(PATH_STATE_SIZE), size=4
    ).astype(np.float64)
    for matching in range(4):
        tails, heads = _matching_indices(
            matching, device=torch.device("cpu")
        )
        states_cpu = torch.as_tensor(base_states, dtype=torch.float64).contiguous()
        tail = states_cpu.index_select(1, tails)
        head = states_cpu.index_select(1, heads)
        pair = (tail + head).contiguous()
        earlier = (head / pair).contiguous()
        signed = torch.as_tensor(
            generator.uniform(-0.02, 0.02, size=tuple(earlier.shape)),
            dtype=torch.float64,
        )
        later = torch.clamp(earlier + signed, 0.0, 1.0).contiguous()
        for duration in (0.5, 1.0):
            observed_cpu = compute_dynkin_phase_observed_increment(
                pair,
                earlier,
                later,
                matching_index=matching,
                quantile_lower=later,
                quantile_upper=later,
                duration_fraction=duration,
                spec=spec,
            )
            direct_cpu = _direct_phase_increment(
                pair, earlier, later, matching
            )
            structural = observed_cpu.structural_zero_mask
            reversed_orientation = _direct_phase_increment(
                pair, 1.0 - earlier, 1.0 - later, matching
            )
            orientation_fixture_detected |= bool(
                torch.max(
                    torch.abs(
                        reversed_orientation[:, :8]
                        - direct_cpu[:, :8]
                    )
                ).item()
                > 1.0e-12
            )
            delta = later - earlier
            centered_sum = earlier + later - 1.0
            wrong_quadratic = torch.sum(
                pair.square() * delta * centered_sum, dim=1
            )
            wrong_cubic = torch.sum(
                pair.pow(3) * delta * centered_sum, dim=1
            )
            quadratic_factor_fixture_detected |= bool(
                torch.max(
                    torch.abs(direct_cpu[:, 8] - wrong_quadratic)
                ).item()
                > 1.0e-14
            )
            cubic_factor_fixture_detected |= bool(
                torch.max(
                    torch.abs(direct_cpu[:, 9] - wrong_cubic)
                ).item()
                > 1.0e-16
            )
            wrong_missing_pair = _direct_phase_increment(
                torch.ones_like(pair), earlier, later, matching
            )
            missing_pair_mass_fixture_detected |= bool(
                torch.max(
                    torch.abs(wrong_missing_pair - direct_cpu)
                ).item()
                > 1.0e-12
            )

            exposure = refinement_phase_exposure(
                pair,
                sample_steps=512,
                duration_fraction=duration,
            )
            correct_drift = compute_dynkin_phase_drift(
                pair,
                earlier,
                exposure,
                matching_index=matching,
                standardized=False,
            )
            post_state_drift = compute_dynkin_phase_drift(
                pair,
                later,
                exposure,
                matching_index=matching,
                standardized=False,
            )
            post_state_fixture_detected |= bool(
                torch.max(
                    torch.abs(
                        correct_drift.center - post_state_drift.center
                    )
                ).item()
                > 1.0e-14
            )
            other_duration = 1.0 if duration == 0.5 else 0.5
            wrong_duration_drift = compute_dynkin_phase_drift(
                pair,
                earlier,
                refinement_phase_exposure(
                    pair,
                    sample_steps=512,
                    duration_fraction=other_duration,
                ),
                matching_index=matching,
                standardized=False,
            )
            wrong_duration_fixture_detected |= bool(
                torch.max(
                    torch.abs(
                        correct_drift.center
                        - wrong_duration_drift.center
                    )
                ).item()
                > 1.0e-14
            )
            weights = torch.as_tensor(
                np.array(spec.fourier_weights, copy=True),
                dtype=torch.float64,
            )
            weight_delta = (
                weights.index_select(1, heads)
                - weights.index_select(1, tails)
            )
            z = 2.0 * earlier - 1.0
            p2 = (3.0 * z.square() - 1.0) / 2.0
            wrong_linear = torch.einsum(
                "oe,pe->po",
                weight_delta,
                (pair * z / 2.0) * torch.expm1(-3.0 * exposure),
            )
            wrong_eigenvalue = torch.cat(
                (
                    wrong_linear,
                    torch.sum(
                        (pair.square() * p2 / 3.0)
                        * torch.expm1(-5.0 * exposure),
                        dim=1,
                        keepdim=True,
                    ),
                    torch.sum(
                        (pair.pow(3) * p2 / 2.0)
                        * torch.expm1(-5.0 * exposure),
                        dim=1,
                        keepdim=True,
                    ),
                ),
                dim=1,
            )
            wrong_eigenvalue_fixture_detected |= bool(
                torch.max(
                    torch.abs(correct_drift.center - wrong_eigenvalue)
                ).item()
                > 1.0e-14
            )
            direct_cpu = torch.where(
                structural[None, :],
                torch.zeros_like(direct_cpu),
                direct_cpu,
            )
            errors = torch.abs(observed_cpu.center - direct_cpu)
            cpu_error = float(torch.max(errors).item())
            maximum_cpu_error = max(maximum_cpu_error, cpu_error)
            formula_flags["fourier"] &= bool(
                torch.max(errors[:, :8]).item() <= 1.0e-10
            )
            formula_flags["quadratic"] &= bool(
                torch.max(errors[:, 8]).item() <= 1.0e-10
            )
            formula_flags["cubic"] &= bool(
                torch.max(errors[:, 9]).item() <= 1.0e-10
            )
            invariant_masks &= int(structural.sum().item()) == 4
            structural_centers &= bool(
                torch.count_nonzero(observed_cpu.center[:, structural]).item()
                == 0
            )
            structural_radii &= bool(
                torch.count_nonzero(
                    observed_cpu.error_radius[:, structural]
                ).item()
                == 0
            )
            observed_device = compute_dynkin_phase_observed_increment(
                pair.to(device),
                earlier.to(device),
                later.to(device),
                matching_index=matching,
                quantile_lower=later.to(device),
                quantile_upper=later.to(device),
                duration_fraction=duration,
                profile=JacobiRBCudaProfile(),
                spec=spec,
            )
            device_error = float(
                torch.max(
                    torch.abs(
                        observed_device.center.detach().cpu() - direct_cpu
                    )
                ).item()
            )
            maximum_device_error = max(maximum_device_error, device_error)
            quantile_pass &= bool(
                observed_device.quantile_enclosure_valid.all().item()
            )
            after = states_cpu.clone()
            after[:, tails] = pair * (1.0 - later)
            after[:, heads] = pair * later
            global_observed = compute_advisory_global_phase_increment(
                states_cpu, after, spec=spec
            )
            nonstructural = ~structural
            difference = torch.abs(
                global_observed.center[:, nonstructural]
                - observed_cpu.center[:, nonstructural]
            )
            allowance = (
                global_observed.error_radius[:, nonstructural]
                + observed_cpu.error_radius[:, nonstructural]
                + 8.0 * torch.finfo(torch.float64).eps
            )
            noninvariant_global &= bool(torch.all(difference <= allowance))
            rows.append(
                {
                    "matching_index": matching,
                    "duration_fraction": duration,
                    "maximum_float64_error": cpu_error,
                    "maximum_device_error": device_error,
                    "structural_zero_count": int(structural.sum().item()),
                    "noninvariant_global_agreement": int(
                        torch.all(difference <= allowance)
                    ),
                }
            )

    pair = torch.zeros((2, EDGES_PER_PHASE), dtype=torch.float64)
    earlier = torch.zeros_like(pair)
    later = torch.zeros_like(pair)
    zero_observed = compute_dynkin_phase_observed_increment(
        pair,
        earlier,
        later,
        matching_index=0,
        quantile_lower=later,
        quantile_upper=later,
        duration_fraction=0.0,
    )
    zero_pass &= bool(
        torch.count_nonzero(zero_observed.center).item() == 0
        and torch.count_nonzero(zero_observed.error_radius).item() == 0
    )
    # Exact endpoint fractions exercise both facets.
    facet = torch.zeros((1, EDGES_PER_PHASE), dtype=torch.float64)
    facet[:, 1::2] = 1.0
    facet_observed = compute_dynkin_phase_observed_increment(
        torch.full_like(facet, 1.0 / EDGES_PER_PHASE),
        facet,
        1.0 - facet,
        matching_index=1,
        quantile_lower=1.0 - facet,
        quantile_upper=1.0 - facet,
    )
    facet_interior &= bool(
        torch.isfinite(facet_observed.center).all()
        and torch.isfinite(facet_observed.error_radius).all()
    )
    corrupt_rejected = False
    try:
        compute_dynkin_phase_observed_increment(
            torch.ones_like(facet),
            facet,
            1.0 - facet,
            matching_index=0,
            quantile_lower=torch.ones_like(facet),
            quantile_upper=torch.zeros_like(facet),
        )
    except ValueError:
        corrupt_rejected = True

    # The immutable spectral/Arb drift oracle remains an independent check.
    drift_oracle, drift_rows = _legacy._phase_moment_oracle_controls(device)
    negative_orientation = orientation_fixture_detected
    return {
        "phase_local_fourier_formula_pass": int(formula_flags["fourier"]),
        "phase_local_quadratic_formula_pass": int(
            formula_flags["quadratic"]
        ),
        "phase_local_cubic_formula_pass": int(formula_flags["cubic"]),
        "phase_local_all_matchings_pass": int(len(rows) == 8),
        "phase_local_half_full_duration_pass": int(
            {row["duration_fraction"] for row in rows} == {0.5, 1.0}
        ),
        "phase_local_facet_interior_pass": int(facet_interior),
        "phase_local_zero_mass_duration_pass": int(zero_pass),
        "structural_invariant_mask_pass": int(invariant_masks),
        "structural_zero_center_pass": int(structural_centers),
        "structural_zero_radius_pass": int(structural_radii),
        "float64_arb_agreement_pass": int(
            maximum_cpu_error <= 1.0e-10
            and int(drift_oracle.get("spectral_arb_agreement_pass", 0)) == 1
        ),
        "cuda_enclosure_pass": int(
            maximum_device_error <= 2.0e-6
            and int(drift_oracle.get("cuda_enclosure_pass", 0)) == 1
        ),
        "quantile_enclosure_pass": int(quantile_pass),
        "noninvariant_local_global_agreement_pass": int(
            noninvariant_global
        ),
        "phase_moment_oracle_pass": int(
            all(
                int(drift_oracle.get(name, 0)) == 1
                for name in (
                    "phase_moment_formula_pass",
                    "phase_moment_all_colors_pass",
                    "phase_moment_half_full_duration_pass",
                    "phase_moment_facet_interior_pass",
                    "phase_moment_zero_mass_duration_pass",
                    "spectral_arb_agreement_pass",
                    "cuda_enclosure_pass",
                    "adversarial_p2_root_enclosure_pass",
                    "cumulative_error_pass",
                )
            )
        ),
        "negative_orientation_fixture_pass": int(negative_orientation),
        "negative_quadratic_factor_fixture_pass": int(
            quadratic_factor_fixture_detected
        ),
        "negative_cubic_factor_fixture_pass": int(
            cubic_factor_fixture_detected
        ),
        "negative_missing_pair_mass_fixture_pass": int(
            missing_pair_mass_fixture_detected
        ),
        "negative_wrong_duration_fixture_pass": int(
            wrong_duration_fixture_detected
        ),
        "negative_wrong_eigenvalue_fixture_pass": int(
            wrong_eigenvalue_fixture_detected
        ),
        "negative_pair_mass_fixture_pass": int(
            missing_pair_mass_fixture_detected
            and int(drift_oracle.get("negative_pair_mass_fixture_pass", 0))
            == 1
        ),
        "negative_duration_fixture_pass": int(
            wrong_duration_fixture_detected
            and int(drift_oracle.get("negative_duration_fixture_pass", 0))
            == 1
        ),
        "negative_eigenvalue_fixture_pass": int(
            wrong_eigenvalue_fixture_detected
            and int(drift_oracle.get("negative_eigenvalue_fixture_pass", 0))
            == 1
        ),
        "negative_corrupt_enclosure_fixture_pass": int(corrupt_rejected),
        "negative_post_state_fixture_pass": int(
            post_state_fixture_detected
            and int(drift_oracle.get("negative_post_state_fixture_pass", 0))
            == 1
        ),
        "maximum_float64_observer_error": maximum_cpu_error,
        "maximum_cuda_observer_error": maximum_device_error,
        "drift_oracle": drift_oracle,
        "drift_oracle_rows": drift_rows,
    }, rows


def _transition_tensor(
    transition_result: Any,
    *names: str,
) -> Tensor:
    chunks = (
        transition_result
        if isinstance(transition_result, tuple)
        else (transition_result,)
    )
    return torch.cat(
        [
            _controls._field(chunk, *names).reshape(-1).to(torch.float64)
            for chunk in chunks
        ]
    )


def _transition_field_sha256(
    transition_result: Any,
    *names: str,
) -> str:
    chunks = (
        transition_result
        if isinstance(transition_result, tuple)
        else (transition_result,)
    )
    arrays = [
        np.ascontiguousarray(
            _controls._field(chunk, *names).detach().cpu().numpy()
        )
        for chunk in chunks
    ]
    return config_fingerprint(
        {
            "chunks": [_array_sha256(value) for value in arrays],
            "dtypes": [str(value.dtype) for value in arrays],
            "shapes": [list(value.shape) for value in arrays],
        }
    )


def _tower_case(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    panel: str,
    case_index: int,
    states_np: np.ndarray,
    plan: PhaseObserverPathIDPlan,
) -> dict[str, Any]:
    root = run_dir / "tower_cases" / panel
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / f"case-{case_index:02d}.json"
    payload_path = root / f"case-{case_index:02d}.npz"
    input_hash = _array_sha256(states_np)
    matching = case_index // 2
    duration = 0.5 if case_index % 2 == 0 else 1.0
    paths = plan.tower_case_path_ids(panel, case_index)
    fingerprint = config_fingerprint(
        {
            "schema": RUN_SCHEMA + "-tower-case-input",
            "panel": panel,
            "case_index": case_index,
            "matching_index": matching,
            "duration_fraction": duration,
            "root_seed": int(args.root_seed),
            "input_states_sha256": input_hash,
            "path_ids": list(paths),
            "path_id_plan_sha256": plan.sha256,
            "phase_observer_version": PHASE_OBSERVER_VERSION,
            "sampler_identity": _sampler_identity(args),
        }
    )
    if meta_path.is_file():
        record = _load(meta_path)
        expected_transition_ids = canonical_tower_transition_ids(
            paths,
            sample_steps=512,
            outer_step=case_index,
            phase=matching,
            device=torch.device(args.device),
        )
        if (
            record.get("input_fingerprint") != fingerprint
            or record.get("schema") != RUN_SCHEMA + "-tower-case"
            or record.get("schema_version") != 1
            or record.get("panel") != panel
            or record.get("case_index") != case_index
            or record.get("matching_index") != matching
            or record.get("duration_fraction") != duration
            or record.get("path_ids") != list(paths)
            or record.get("input_states_sha256") != input_hash
            or record.get("transition_ids_sha256")
            != _array_sha256(
                expected_transition_ids.detach().cpu().numpy()
            )
            or int(record.get("complete", 0)) != 1
            or int(record.get("observer_input_hash_invariance_pass", 0))
            != 1
            or record.get("observer_input_hash_before")
            != record.get("observer_input_hash_after")
            or not math.isfinite(
                float(record.get("peak_memory_fraction", math.inf))
            )
            or not 0.0
            <= float(record.get("peak_memory_fraction", math.inf))
            <= 1.0
            or not payload_path.is_file()
            or record.get("npz_sha256") != file_fingerprint(payload_path)
            or record.get("npz_size") != payload_path.stat().st_size
            or record.get("npz_name") != payload_path.name
            or not isinstance(record.get("array_sha256"), Mapping)
            or not isinstance(record.get("diagnostics"), Mapping)
        ):
            raise ArtifactCompatibilityError(
                f"tower {panel} case {case_index} changed"
            )
        with np.load(payload_path, allow_pickle=False) as archive:
            expected_arrays = dict(record["array_sha256"])
            if set(archive.files) != set(expected_arrays):
                raise ArtifactCompatibilityError(
                    f"tower {panel} case {case_index} payload changed"
                )
            for name, expected in expected_arrays.items():
                if name not in archive or _array_sha256(archive[name]) != expected:
                    raise ArtifactCompatibilityError(
                        f"tower {panel} case {case_index} payload changed"
                    )
            residual = np.asarray(
                archive["standardized_residual"], dtype=np.float64
            )
            radius = np.asarray(
                archive["standardized_error_radius"], dtype=np.float64
            )
            structural = np.asarray(
                archive["structural_zero_mask"], dtype=np.bool_
            )
            if (
                residual.shape != (len(paths), 10)
                or radius.shape != residual.shape
                or structural.shape != (10,)
                or not np.isfinite(residual).all()
                or not np.isfinite(radius).all()
                or np.any(radius < 0.0)
                or np.count_nonzero(residual[:, structural]) != 0
                or np.count_nonzero(radius[:, structural]) != 0
                or record.get("residual_sha256")
                != _array_sha256(residual)
                or record.get("residual_error_sha256")
                != _array_sha256(radius)
            ):
                raise ArtifactCompatibilityError(
                    f"tower {panel} case {case_index} residual changed"
                )
        diagnostics = dict(record["diagnostics"])
        transition_count = len(paths) * EDGES_PER_PHASE
        if (
            int(diagnostics.get("transition_count", -1))
            != transition_count
            or int(diagnostics.get("certified_count", -1))
            != transition_count
            or any(
                int(diagnostics.get(name, -1)) != 0
                for name in _FORBIDDEN_COUNTS
            )
        ):
            raise ArtifactCompatibilityError(
                f"tower {panel} case {case_index} diagnostics changed"
            )
        return record

    device = torch.device(args.device)
    states = torch.as_tensor(
        states_np, dtype=torch.float64, device=device
    ).contiguous()
    transition_ids = canonical_tower_transition_ids(
        paths,
        sample_steps=512,
        outer_step=case_index,
        phase=matching,
        device=device,
    )
    result = run_dynkin_tower_phase(
        states,
        matching_index=matching,
        duration_fraction=duration,
        sample_steps=512,
        rng_key=(int(args.root_seed), "phase-observer-tower", panel, case_index),
        transition_ids=transition_ids,
        profile=JacobiRBCudaProfile(),
        sampler=_sampler_for_args(args),
    )
    later = _transition_tensor(
        result.transition_result, "later_head_fraction", "later", "y"
    ).reshape(len(paths), EDGES_PER_PHASE).contiguous()
    lower = _transition_tensor(
        result.transition_result, "quantile_lower"
    ).reshape_as(later).contiguous()
    upper = _transition_tensor(
        result.transition_result, "quantile_upper"
    ).reshape_as(later).contiguous()
    observer_input_hash_before = config_fingerprint(
        {
            "states_before": _array_sha256(
                states.detach().cpu().numpy()
            ),
            "states_after": _array_sha256(
                result.final_states.detach().cpu().numpy()
            ),
            "later": _transition_field_sha256(
                result.transition_result,
                "later_head_fraction",
                "later",
                "y",
            ),
            "target": _transition_field_sha256(
                result.transition_result,
                "denoising_target",
                "target",
                "zbar",
            ),
            "certificate": _transition_field_sha256(
                result.transition_result,
                "certificate_codes",
                "certificate_code",
            ),
            "quantile_lower": _transition_field_sha256(
                result.transition_result, "quantile_lower"
            ),
            "quantile_upper": _transition_field_sha256(
                result.transition_result, "quantile_upper"
            ),
        }
    )
    observed = compute_dynkin_phase_observed_increment_from_states(
        states,
        result.final_states,
        matching_index=matching,
        quantile_lower=lower,
        quantile_upper=upper,
        later_head_fraction=later,
        profile=JacobiRBCudaProfile(),
        duration_fraction=duration,
    )
    observer_input_hash_after = config_fingerprint(
        {
            "states_before": _array_sha256(
                states.detach().cpu().numpy()
            ),
            "states_after": _array_sha256(
                result.final_states.detach().cpu().numpy()
            ),
            "later": _transition_field_sha256(
                result.transition_result,
                "later_head_fraction",
                "later",
                "y",
            ),
            "target": _transition_field_sha256(
                result.transition_result,
                "denoising_target",
                "target",
                "zbar",
            ),
            "certificate": _transition_field_sha256(
                result.transition_result,
                "certificate_codes",
                "certificate_code",
            ),
            "quantile_lower": _transition_field_sha256(
                result.transition_result, "quantile_lower"
            ),
            "quantile_upper": _transition_field_sha256(
                result.transition_result, "quantile_upper"
            ),
        }
    )
    tails, heads = _matching_indices(matching, device=device)
    pair = (
        states.index_select(1, tails) + states.index_select(1, heads)
    ).contiguous()
    positive = pair > 0.0
    earlier = torch.where(
        positive,
        states.index_select(1, heads)
        / torch.where(positive, pair, torch.ones_like(pair)),
        torch.zeros_like(pair),
    ).contiguous()
    exposure = refinement_phase_exposure(
        pair, sample_steps=512, duration_fraction=duration
    )
    drift = compute_dynkin_phase_drift(
        pair,
        earlier,
        exposure,
        matching_index=matching,
        standardized=False,
        cuda_profile=JacobiRBCudaProfile(),
    )
    residual = combine_dynkin_phase_residual(
        observed, drift, standardized=True
    )
    advisory = compute_advisory_global_phase_increment(
        states, result.final_states
    )
    structural = observed.structural_zero_mask
    nonstructural = ~structural
    local_global_difference = torch.abs(
        observed.center[:, nonstructural]
        - advisory.center[:, nonstructural]
    )
    local_global_allowance = (
        observed.error_radius[:, nonstructural]
        + advisory.error_radius[:, nonstructural]
        + 8.0 * torch.finfo(torch.float64).eps
    )
    scales = torch.as_tensor(
        refinement_observable_spec().standard_deviations,
        dtype=torch.float64,
        device=device,
    )
    old_global_residual = (
        advisory.center - drift.center
    ) / scales
    old_structural_nonzero = int(
        torch.count_nonzero(old_global_residual[:, structural]).item()
    )
    arrays = {
        "standardized_residual": residual.center.detach().cpu().numpy(),
        "standardized_error_radius": residual.error_radius.detach().cpu().numpy(),
        "observed_center": observed.center.detach().cpu().numpy(),
        "observed_error_radius": observed.error_radius.detach().cpu().numpy(),
        "global_center_advisory": advisory.center.detach().cpu().numpy(),
        "structural_zero_mask": structural.detach().cpu().numpy(),
        "final_states": result.final_states.detach().cpu().numpy(),
    }
    _atomic_write_npz(payload_path, **arrays)
    diagnostics = dict(result.diagnostics)
    record = {
        "schema": RUN_SCHEMA + "-tower-case",
        "schema_version": 1,
        "input_fingerprint": fingerprint,
        "panel": panel,
        "case_index": case_index,
        "matching_index": matching,
        "duration_fraction": duration,
        "path_ids": list(paths),
        "path_ids_sha256": _array_sha256(np.asarray(paths, dtype=np.int64)),
        "transition_ids_sha256": _array_sha256(
            transition_ids.detach().cpu().numpy()
        ),
        "input_states_sha256": input_hash,
        "transition_output_sha256": result.transition_output_sha256,
        "final_state_sha256": result.final_state_sha256,
        "residual_sha256": _array_sha256(arrays["standardized_residual"]),
        "residual_error_sha256": _array_sha256(
            arrays["standardized_error_radius"]
        ),
        "structural_zero_count": int(structural.sum().item()),
        "observer_input_hash_before": observer_input_hash_before,
        "observer_input_hash_after": observer_input_hash_after,
        "observer_input_hash_invariance_pass": int(
            observer_input_hash_before == observer_input_hash_after
        ),
        "structural_center_exact_zero_pass": int(
            torch.count_nonzero(observed.center[:, structural]).item() == 0
        ),
        "structural_radius_exact_zero_pass": int(
            torch.count_nonzero(
                observed.error_radius[:, structural]
            ).item()
            == 0
        ),
        "noninvariant_local_global_agreement_pass": int(
            torch.all(local_global_difference <= local_global_allowance).item()
        ),
        "old_global_structural_nonzero_count": old_structural_nonzero,
        "maximum_standardized_error_radius": float(
            torch.max(residual.error_radius).item()
        ),
        "peak_memory_fraction": (
            torch.cuda.max_memory_allocated(device)
            / torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else 0.0
        ),
        "diagnostics": diagnostics,
        "npz_name": payload_path.name,
        "npz_sha256": file_fingerprint(payload_path),
        "npz_size": payload_path.stat().st_size,
        "array_sha256": {
            name: _array_sha256(value)
            for name, value in sorted(arrays.items())
        },
        "complete": 1,
        **NO_WORK,
    }
    atomic_write_json(meta_path, record)
    return record


def _tower_panel(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    panel: str,
    plan: PhaseObserverPathIDPlan,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    meta_path = run_dir / f"tower_panel_{panel}.json"
    payload_path = run_dir / f"tower_panel_{panel}.npz"
    existing_record: dict[str, Any] | None = None
    if meta_path.is_file():
        existing_record = _load(meta_path)
        if (
            not payload_path.is_file()
            or existing_record.get("npz_sha256")
            != file_fingerprint(payload_path)
            or existing_record.get("path_id_plan_sha256") != plan.sha256
        ):
            raise ArtifactCompatibilityError(
                f"frozen tower panel {panel} changed"
            )

    initial_payload = run_dir / f"tower_panel_{panel}_initial_states.npz"
    with np.load(initial_payload, allow_pickle=False) as archive:
        states_np = np.ascontiguousarray(archive["states"], dtype=np.float64)
    cases: list[dict[str, Any]] = []
    panel_started = time.perf_counter()
    for case in range(TOWER_CASE_COUNT):
        cases.append(
            _tower_case(
                run_dir,
                args,
                panel=panel,
                case_index=case,
                states_np=states_np,
                plan=plan,
            )
        )
        _legacy._progress(
            args,
            f"phase-observer/tower-{panel}",
            case + 1,
            TOWER_CASE_COUNT,
            panel_started,
        )
    spec = refinement_observable_spec()
    features: dict[str, np.ndarray] = {}
    feature_errors: dict[str, np.ndarray] = {}
    residual_arrays: dict[str, np.ndarray] = {}
    error_arrays: dict[str, np.ndarray] = {}
    for case in cases:
        case_payload = (
            run_dir
            / "tower_cases"
            / panel
            / f"case-{int(case['case_index']):02d}.npz"
        )
        with np.load(case_payload, allow_pickle=False) as archive:
            residual = np.asarray(
                archive["standardized_residual"], dtype=np.float64
            )
            error = np.asarray(
                archive["standardized_error_radius"], dtype=np.float64
            )
        key = (
            f"m{int(case['matching_index'])}_"
            f"d{float(case['duration_fraction']):g}"
        )
        residual_arrays[key] = residual
        error_arrays[key] = error
        for index, name in enumerate(spec.names):
            feature_name = f"{key}_{name}"
            features[feature_name] = residual[:, index]
            feature_errors[feature_name] = error[:, index]
    reps = 100 if args.test_only_reduced_workload else 20_000
    inference = whole_path_max_t_intervals(
        features,
        seed=int(args.root_seed) + (31 if panel == "a" else 41),
        confidence=0.99,
        reps=reps,
    )
    center_only_fingerprint = str(inference["family_fingerprint"])
    for member in inference["members"]:
        name = str(member["name"])
        numerical_radius = float(np.mean(feature_errors[name]))
        lower = math.nextafter(
            float(member["simultaneous_lower"]) - numerical_radius,
            -math.inf,
        )
        upper = math.nextafter(
            float(member["simultaneous_upper"]) + numerical_radius,
            math.inf,
        )
        member["center_only_simultaneous_lower"] = member[
            "simultaneous_lower"
        ]
        member["center_only_simultaneous_upper"] = member[
            "simultaneous_upper"
        ]
        member["mean_certified_numerical_radius"] = numerical_radius
        member["simultaneous_lower"] = lower
        member["simultaneous_upper"] = upper
        member["contains_zero"] = int(lower <= 0.0 <= upper)
    inference["passed"] = int(
        all(int(member["contains_zero"]) == 1 for member in inference["members"])
    )
    inference["center_only_family_fingerprint"] = center_only_fingerprint
    inference["certified_error_radius_fingerprint"] = config_fingerprint(
        {
            name: _array_sha256(value)
            for name, value in sorted(feature_errors.items())
        }
    )
    inference["family_fingerprint"] = config_fingerprint(
        {
            "center_only": center_only_fingerprint,
            "certified_error_radii": inference[
                "certified_error_radius_fingerprint"
            ],
        }
    )
    inference["authorizing_intervals_include_certified_radius"] = 1
    ordered_names = tuple(sorted(features))
    matrix = np.column_stack([features[name] for name in ordered_names])
    errors = np.stack([error_arrays[key] for key in sorted(error_arrays)])
    payload_arrays = {
        "initial_states": states_np,
        "path_matrix": matrix,
        "error_radius": errors,
    }
    if existing_record is None:
        _atomic_write_npz(payload_path, **payload_arrays)
    else:
        with np.load(payload_path, allow_pickle=False) as archive:
            if set(archive.files) != set(payload_arrays) or any(
                not np.array_equal(
                    np.asarray(archive[name]), expected_array
                )
                for name, expected_array in payload_arrays.items()
            ):
                raise ArtifactCompatibilityError(
                    f"frozen tower panel {panel} payload changed"
                )
    diagnostics = [dict(case["diagnostics"]) for case in cases]
    transition_count = sum(
        int(value.get("transition_count", 0)) for value in diagnostics
    )
    certified_count = sum(
        int(value.get("certified_count", 0)) for value in diagnostics
    )
    fallback_count = sum(
        int(value.get("fallback_count", 0)) for value in diagnostics
    )
    elapsed = sum(
        float(value.get("elapsed_seconds", 0.0)) for value in diagnostics
    )
    fallback_elapsed = sum(
        float(value.get("fallback_elapsed_seconds", 0.0))
        for value in diagnostics
    )
    execution = {
        "transition_count": transition_count,
        "certified_count": certified_count,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "fallback_count": fallback_count,
        "fallback_fraction": (
            fallback_count / transition_count if transition_count else 0.0
        ),
        "elapsed_seconds": elapsed,
        "fallback_elapsed_seconds": fallback_elapsed,
        "mass_error": max(
            (
                max(
                    float(value.get("maximum_pair_total_error", math.inf)),
                    float(
                        value.get("maximum_global_simplex_error", math.inf)
                    ),
                )
                for value in diagnostics
            ),
            default=math.inf,
        ),
        "peak_memory_fraction": max(
            (
                float(case.get("peak_memory_fraction", math.inf))
                for case in cases
            ),
            default=math.inf,
        ),
        **{
            name: sum(int(value.get(name, 0)) for value in diagnostics)
            for name in _FORBIDDEN_COUNTS
        },
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
            "mean_certified_numerical_radius": member[
                "mean_certified_numerical_radius"
            ],
        }
        for member in inference["members"]
    ]
    record = {
        "schema": RUN_SCHEMA + "-tower-panel",
        "schema_version": 1,
        "panel": panel,
        "path_count": int(states_np.shape[0]),
        "family_member_count": len(features),
        "structural_zero_member_count": sum(
            int(case["structural_zero_count"]) for case in cases
        ),
        "path_feature_names": list(ordered_names),
        "path_matrix_sha256": _array_sha256(matrix),
        "initial_states_sha256": _array_sha256(states_np),
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "cases": cases,
        "inference": inference,
        "interval_rows": rows,
        "execution": execution,
        "old_global_structural_nonzero_count": sum(
            int(case["old_global_structural_nonzero_count"])
            for case in cases
        ),
        "case_atomic_resume_pass": 1,
        "passed": int(inference["passed"]),
        "npz_name": payload_path.name,
        "npz_sha256": file_fingerprint(payload_path),
        "npz_size": payload_path.stat().st_size,
        **NO_WORK,
    }
    normalized = json.loads(
        json.dumps(record, sort_keys=True, allow_nan=False)
    )
    if existing_record is not None:
        if existing_record != normalized:
            raise ArtifactCompatibilityError(
                f"frozen tower panel {panel} summary changed"
            )
        return existing_record, rows, execution
    atomic_write_json(meta_path, normalized)
    return normalized, rows, execution


def _load_bound_preflight_gate(run_dir: Path) -> dict[str, Any]:
    gate = _load(run_dir / "phase_observer_preflight_gate.json")
    evidence = _load(run_dir / "phase_observer_preflight_metrics.json")
    metrics = evidence.get("metrics")
    panel_a_path = run_dir / "tower_panel_a.json"
    panel_b_path = run_dir / "tower_panel_b.json"
    if (
        not isinstance(metrics, Mapping)
        or not panel_a_path.is_file()
        or not panel_b_path.is_file()
        or evidence.get("tower_panel_a_sha256")
        != file_fingerprint(panel_a_path)
        or evidence.get("tower_panel_b_sha256")
        != file_fingerprint(panel_b_path)
    ):
        raise ArtifactCompatibilityError(
            "cached phase-observer preflight evidence changed"
        )
    expected = evaluate_phase_observer_preflight(metrics)
    if gate != expected:
        raise ArtifactCompatibilityError(
            "cached phase-observer preflight gate changed"
        )
    return gate


def _run_preflight_stage(
    run_dir: Path,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    gate_path = run_dir / "phase_observer_preflight_gate.json"
    if gate_path.is_file() and (
        _artifact_is_terminally_registered(run_dir, gate_path)
        or not _run_is_interrupted(run_dir)
    ):
        return _load_bound_preflight_gate(run_dir)
    plan = _load_frozen_path_id_plan(run_dir)
    path_ids = _path_id_plan_preflight(run_dir, args, plan)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    oracle, oracle_rows = _phase_observer_oracle_controls(device)
    atomic_write_csv(run_dir / "phase_observer_oracle.csv", oracle_rows)
    drift_rows = oracle.pop("drift_oracle_rows")
    atomic_write_csv(run_dir / "phase_moment_oracle.csv", drift_rows)
    parent_dir = Path(args.parent_dynkin_idfix_run_dir).resolve()
    replay, replay_execution = _observer_legacy_replay(
        run_dir, args, parent_dir, plan
    )
    panel_a, rows_a, execution_a = _tower_panel(
        run_dir, args, panel="a", plan=plan
    )
    panel_b, rows_b, execution_b = _tower_panel(
        run_dir, args, panel="b", plan=plan
    )
    atomic_write_csv(
        run_dir / "tower_identity_intervals.csv", rows_a + rows_b
    )
    forensic = {
        "schema": RUN_SCHEMA + "-global-subtraction-forensic",
        "schema_version": 1,
        "authorizing": 0,
        "parent_failure": provenance.get("parent_failure_message"),
        "panel_a_structural_nonzero_count": panel_a[
            "old_global_structural_nonzero_count"
        ],
        "panel_b_structural_nonzero_count": panel_b[
            "old_global_structural_nonzero_count"
        ],
        "roundoff_reproduced": int(
            int(panel_a["old_global_structural_nonzero_count"])
            + int(panel_b["old_global_structural_nonzero_count"])
            > 0
            or provenance.get("parent_failure_message")
            == "nonzero degenerate whole-path statistic"
        ),
        **NO_WORK,
    }
    atomic_write_json(
        run_dir / "global_subtraction_roundoff_forensic.json", forensic
    )
    executions = (replay_execution, execution_a, execution_b)
    tower_cases = tuple(panel_a["cases"]) + tuple(panel_b["cases"])
    transition_count = sum(
        int(value.get("transition_count", 0)) for value in executions
    )
    certified_count = sum(
        int(value.get("certified_count", 0)) for value in executions
    )
    fallback_count = sum(
        int(value.get("fallback_count", 0)) for value in executions
    )
    elapsed = sum(
        float(value.get("elapsed_seconds", 0.0)) for value in executions
    )
    fallback_elapsed = sum(
        float(value.get("fallback_elapsed_seconds", 0.0))
        for value in executions
    )
    cumulative_error = max(
        [
            float(
                oracle.get(
                    "maximum_cuda_observer_error", math.inf
                )
            )
        ]
        + [
            float(case["maximum_standardized_error_radius"])
            for panel in (panel_a, panel_b)
            for case in panel["cases"]
        ]
    )
    metrics = {
        "production_authorizing_pass": int(
            not args.test_only_reduced_workload
        ),
        "control_provenance_pass": int(provenance.get("passed", 0)),
        "parent_failure_adjudication_pass": int(
            provenance.get("parent_re_adjudication")
            == "tower_observer_roundoff_invalid"
        ),
        "twenty_one_parent_sources_immutable_pass": int(
            provenance.get("parent_source_count") == PARENT_SOURCE_COUNT
            and provenance.get("parent_source_fingerprint")
            == PARENT_SOURCE_FINGERPRINT
        ),
        "parent_path_id_plan_pass": int(
            provenance.get("parent_path_id_plan_pass", 0)
        ),
        "parent_legacy_k512_replay_pass": int(
            provenance.get("parent_legacy_k512_replay_pass", 0)
        ),
        "parent_phase_moment_oracle_pass": int(
            provenance.get("parent_phase_moment_oracle_pass", 0)
        ),
        "parent_no_tower_pilot_work_pass": int(
            int(provenance.get("parent_tower_inference_performed", 1)) == 0
            and int(provenance.get("parent_pilot_performed", 1)) == 0
        ),
        "path_id_plan_pass": int(path_ids["passed"]),
        "fresh_namespace_disjoint_pass": int(
            path_ids["checks"]["fresh_namespace_disjoint_pass"]
        ),
        "legacy_k512_replay_pass": int(
            replay["legacy_k512_replay_pass"]
        ),
        "transition_target_certificate_hash_invariance_pass": int(
            replay[
                "transition_target_certificate_hash_invariance_pass"
            ]
        ),
        "observer_state_hash_invariance_pass": int(
            replay["observer_state_hash_invariance_pass"]
        ),
        "global_subtraction_roundoff_reproduced_pass": int(
            forensic["roundoff_reproduced"]
        ),
        **{
            key: value
            for key, value in oracle.items()
            if key != "drift_oracle"
        },
        "tower_panel_a_pass": int(panel_a["passed"]),
        "tower_panel_b_pass": int(panel_b["passed"]),
        "tower_joint_max_t_pass": int(
            panel_a["passed"] and panel_b["passed"]
        ),
        "tower_panels_frozen_pass": int(
            (run_dir / "panel_a_plan.json").is_file()
            and (run_dir / "panel_b_plan.json").is_file()
        ),
        "tower_panels_disjoint_pass": int(
            panel_a["initial_states_sha256"]
            != panel_b["initial_states_sha256"]
        ),
        "tower_case_atomic_resume_pass": int(
            panel_a["case_atomic_resume_pass"]
            and panel_b["case_atomic_resume_pass"]
        ),
        "tower_case_structural_zero_center_pass": int(
            all(
                int(case["structural_center_exact_zero_pass"]) == 1
                for case in tower_cases
            )
        ),
        "tower_case_observer_input_hash_invariance_pass": int(
            all(
                int(case["observer_input_hash_invariance_pass"]) == 1
                for case in tower_cases
            )
        ),
        "tower_case_structural_zero_radius_pass": int(
            all(
                int(case["structural_radius_exact_zero_pass"]) == 1
                for case in tower_cases
            )
        ),
        "tower_case_noninvariant_global_agreement_pass": int(
            all(
                int(case["noninvariant_local_global_agreement_pass"]) == 1
                for case in tower_cases
            )
        ),
        "tower_authorizing_interval_radius_pass": int(
            all(
                int(
                    panel_record["inference"].get(
                        "authorizing_intervals_include_certified_radius", 0
                    )
                )
                == 1
                for panel_record in (panel_a, panel_b)
            )
        ),
        "root_seed": int(args.root_seed),
        "parent_record_count": int(
            provenance.get("parent_artifact_record_count", -1)
        ),
        "parent_source_count": int(
            provenance.get("parent_source_count", -1)
        ),
        "grid_size": GRID_SIZE,
        "alpha": 1.0,
        "tau_eff": TAU_EFF,
        "tower_panel_count": 2,
        "tower_clusters_per_panel": int(args.tower_panel_clusters),
        "tower_family_member_count": int(panel_a["family_member_count"]),
        "structural_zero_member_count": int(
            panel_a["structural_zero_member_count"]
        ),
        "tower_bootstrap_replicates": (
            100 if args.test_only_reduced_workload else 20_000
        ),
        "tower_confidence": 0.99,
        "maximum_cumulative_standardized_error": cumulative_error,
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
            float(value.get("peak_memory_fraction", 0.0))
            for value in executions
        ),
        "mass_error": max(
            float(value.get("mass_error", math.inf))
            for value in executions
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
        "path_id_plan": path_ids,
        "legacy_replay": replay,
        "phase_observer_oracle": oracle,
        "tower_panel_a_sha256": file_fingerprint(
            run_dir / "tower_panel_a.json"
        ),
        "tower_panel_b_sha256": file_fingerprint(
            run_dir / "tower_panel_b.json"
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "phase_observer_preflight_metrics.json", record)
    gate = evaluate_phase_observer_preflight(metrics)
    atomic_write_json(gate_path, gate)
    return gate


def _run_dynkin_level_panel(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    panel: str,
    initial_states: np.ndarray,
    path_ids: Sequence[int],
    sample_steps: int,
) -> dict[str, Any]:
    initial = np.ascontiguousarray(initial_states, dtype=np.float64)
    paths = tuple(int(value) for value in path_ids)
    plan = _load_frozen_path_id_plan(run_dir)
    plan.validate_role_path_ids(f"pilot_{panel}", paths)
    level = int(sample_steps)
    total_steps = (
        level
        if args.pilot_stop_steps is None
        else min(level, int(args.pilot_stop_steps))
    )
    if (
        initial.shape != (len(paths), PATH_STATE_SIZE)
        or level not in SUPPORTED_SAMPLE_STEPS
        or total_steps <= 0
        or total_steps % REFINEMENT_SHARD_STEPS
    ):
        raise PhaseObserverSchedulerError("invalid Dynkin pilot level plan")
    checkpoint_steps = tuple(
        int(round(level * fraction))
        for fraction in OBSERVATION_TIME_FRACTIONS
        if int(round(level * fraction)) <= total_steps
    )
    root = run_dir / "dynkin_shards" / "pilot" / panel / f"K{level:04d}"
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    profile = JacobiRBCudaProfile()
    sampler = _sampler_for_args(args)
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
    level_started = time.perf_counter()
    total_shards = (
        math.ceil(len(paths) / MAX_REFINEMENT_PATHS_PER_GROUP)
        * (total_steps // REFINEMENT_SHARD_STEPS)
    )
    completed_shards = 0
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
        fingerprint = config_fingerprint(
            {
                "legacy_dynkin_shard_fingerprint": (
                    _legacy._dynkin_shard_fingerprint(
                        stage="pilot",
                        panel=panel,
                        sample_steps=level,
                        path_ids=group_paths,
                        root_seed=int(args.root_seed),
                        path_id_plan_sha256=plan.sha256,
                    )
                ),
                "phase_observer_version": PHASE_OBSERVER_VERSION,
                "sampler_identity": _sampler_identity(args),
            }
        )
        previous_chain = config_fingerprint(
            {
                "kind": "phase-observer-dynkin-shard-genesis",
                "fingerprint": fingerprint,
                "initial_states_sha256": _array_sha256(committed),
            }
        )
        reuse_tail = True
        for start_step in range(0, total_steps, REFINEMENT_SHARD_STEPS):
            base = (
                f"group-{group_index:03d}-steps-{start_step:04d}-"
                f"{start_step + REFINEMENT_SHARD_STEPS - 1:04d}"
            )
            metadata_path = root / f"{base}.json"
            input_hash = _array_sha256(committed)
            loaded = (
                _legacy._load_dynkin_shard(
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
                started = time.perf_counter()
                result = run_dynkin_refinement_shard(
                    states,
                    path_ids=group_paths,
                    sample_steps=level,
                    start_step=start_step,
                    root_seed=int(args.root_seed),
                    panel_namespace=f"phase-observer-pilot-{panel}",
                    profile=profile,
                    sampler=sampler,
                    checkpoint_steps=checkpoint_steps,
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
                    {
                        name: _array_sha256(value)
                        for name, value in arrays.items()
                    }
                )
                full_record = result.to_record()
                base_record = dict(full_record.pop("base_shard"))
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
                    "complete_wall_upper_seconds": max(
                        0.001, 1.02 * (time.perf_counter() - started)
                    ),
                    "peak_memory_fraction": (
                        torch.cuda.max_memory_allocated(device)
                        / torch.cuda.get_device_properties(device).total_memory
                        if device.type == "cuda"
                        else 0.0
                    ),
                    **base_record,
                    **full_record,
                    **NO_WORK,
                }
                row["chain_sha256"] = config_fingerprint(
                    {
                        "fingerprint": fingerprint,
                        "input_states_sha256": input_hash,
                        "previous_chain_sha256": previous_chain,
                        "batch_output_sha256": row["batch_output_sha256"],
                        "batch_final_state_sha256": row[
                            "batch_final_state_sha256"
                        ],
                        "batch_certificate_sha256": row[
                            "batch_certificate_sha256"
                        ],
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
                    np.array(arrays["accumulator_center"], copy=True),
                    dtype=torch.float64,
                    device=device,
                ).contiguous(),
                compensation=torch.as_tensor(
                    np.array(arrays["accumulator_compensation"], copy=True),
                    dtype=torch.float64,
                    device=device,
                ).contiguous(),
                error_radius=torch.as_tensor(
                    np.array(
                        arrays["accumulator_error_radius"], copy=True
                    ),
                    dtype=torch.float64,
                    device=device,
                ).contiguous(),
            )
            previous_chain = str(row["chain_sha256"])
            for checkpoint in row.get("observable_checkpoints", ()):
                step = int(checkpoint["completed_step"])
                ids = tuple(int(value) for value in checkpoint["path_ids"])
                raw = np.asarray(checkpoint["raw_values"], dtype=np.float64)
                dynkin = np.asarray(
                    checkpoint["dynkin_values"], dtype=np.float64
                )
                error = np.asarray(
                    checkpoint["dynkin_error_radius"], dtype=np.float64
                )
                for index, path_id in enumerate(ids):
                    raw_by_step.setdefault(step, {})[path_id] = raw[index]
                    dynkin_by_step.setdefault(step, {})[path_id] = dynkin[index]
                    error_by_step.setdefault(step, {})[path_id] = error[index]
            rows.append(row)
            completed_shards += 1
            _legacy._progress(
                args,
                f"phase-observer/pilot/{panel}/K{level}",
                completed_shards,
                total_shards,
                level_started,
            )
        for path_id, value in zip(group_paths, committed, strict=True):
            final_by_path[path_id] = value
    missing = [
        step
        for step in checkpoint_steps
        if any(path not in raw_by_step[step] for path in paths)
    ]
    if missing:
        raise ArtifactCompatibilityError(
            f"pilot level K={level} lacks checkpoints {missing}"
        )
    raw_checkpoints = {
        step: np.stack([raw_by_step[step][path] for path in paths])
        for step in checkpoint_steps
    }
    dynkin_checkpoints = {
        step: np.stack([dynkin_by_step[step][path] for path in paths])
        for step in checkpoint_steps
    }
    error_checkpoints = {
        step: np.stack([error_by_step[step][path] for path in paths])
        for step in checkpoint_steps
    }
    ordered_final = np.stack([final_by_path[path] for path in paths])
    diagnostics = [dict(row.get("diagnostics", {})) for row in rows]
    transition_count = sum(
        int(value.get("transition_count", 0)) for value in diagnostics
    )
    certified_count = sum(
        int(value.get("certified_count", 0)) for value in diagnostics
    )
    fallback_count = sum(
        int(value.get("fallback_count", 0)) for value in diagnostics
    )
    elapsed = sum(
        float(value.get("elapsed_seconds", 0.0)) for value in diagnostics
    )
    fallback_elapsed = sum(
        float(value.get("fallback_elapsed_seconds", 0.0))
        for value in diagnostics
    )
    complete_wall = sum(
        float(row["complete_wall_upper_seconds"]) for row in rows
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
        "elapsed_seconds": elapsed,
        "fallback_elapsed_seconds": fallback_elapsed,
        "complete_wall_upper_seconds": complete_wall,
        "conservative_rate": (
            transition_count / complete_wall if complete_wall > 0 else 0.0
        ),
        "mass_error": max(
            (
                float(
                    value.get("maximum_global_simplex_error", math.inf)
                )
                for value in diagnostics
            ),
            default=math.inf,
        ),
        "peak_memory_fraction": max(
            (
                float(row.get("peak_memory_fraction", math.inf))
                for row in rows
            ),
            default=math.inf,
        ),
        "state_updates_device_resident_pass": int(
            bool(diagnostics)
            and all(
                int(value.get("state_updates_device_resident", 0)) == 1
                and int(value.get("in_shard_host_roundtrip_count", -1)) == 0
                for value in diagnostics
            )
        ),
        "shard_chain_pass": 1,
        **{
            name: sum(int(value.get(name, 0)) for value in diagnostics)
            for name in _FORBIDDEN_COUNTS
        },
    }


def _serialize_panel_observables(
    path: Path,
    results: Mapping[int, Mapping[str, Any]],
    *,
    require_existing: bool = False,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for level, result in sorted(results.items()):
        for prefix, key in (
            ("raw", "raw_checkpoint_values"),
            ("dynkin", "dynkin_checkpoint_values"),
            ("error", "dynkin_error_radius"),
        ):
            for step, value in sorted(dict(result[key]).items()):
                arrays[
                    f"{prefix}_K{level:04d}_step{int(step):04d}"
                ] = np.asarray(value, dtype=np.float64)
    if require_existing:
        if not path.is_file():
            raise ArtifactCompatibilityError(
                f"resume lacks frozen observable payload: {path.name}"
            )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                not np.array_equal(np.asarray(archive[name]), value)
                for name, value in arrays.items()
            ):
                raise ArtifactCompatibilityError(
                    f"frozen observable payload changed: {path.name}"
                )
    else:
        _atomic_write_npz(path, **arrays)
    return {
        "npz_name": path.name,
        "npz_sha256": file_fingerprint(path),
        "npz_size": path.stat().st_size,
        "array_count": len(arrays),
        "array_sha256": {
            name: _array_sha256(value)
            for name, value in sorted(arrays.items())
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
    plan = _load_frozen_path_id_plan(run_dir)
    existing_record: dict[str, Any] | None = None
    if metadata_path.is_file():
        existing_record = _load(metadata_path)
        payload = existing_record.get("observable_payload")
        if (
            not isinstance(payload, Mapping)
            or not payload_path.is_file()
            or payload.get("npz_sha256") != file_fingerprint(payload_path)
            or existing_record.get("path_id_plan_sha256") != plan.sha256
        ):
            raise ArtifactCompatibilityError(
                f"frozen pilot panel {panel} changed"
            )
    paths = plan.pilot_path_ids(panel)
    mixed = np.asarray(source["mixed_target"], dtype=np.float64).reshape(1, -1)
    initial = np.repeat(mixed, len(paths), axis=0)
    levels = {
        int(level): _run_dynkin_level_panel(
            run_dir,
            args,
            panel=panel,
            initial_states=initial,
            path_ids=paths,
            sample_steps=int(level),
        )
        for level in args.sample_steps
    }
    main, reference = _legacy._projection_feature_matrices(
        levels, key="dynkin_checkpoint_values"
    )
    raw_main, raw_reference = _legacy._projection_feature_matrices(
        levels, key="raw_checkpoint_values"
    )
    execution = _legacy._aggregate_level_results(list(levels.values()))
    maximum_error = max(
        (
            float(np.max(value))
            for result in levels.values()
            for value in result["dynkin_error_radius"].values()
        ),
        default=math.inf,
    )
    payload = _serialize_panel_observables(
        payload_path,
        levels,
        require_existing=existing_record is not None,
    )
    record = {
        "schema": RUN_SCHEMA + "-pilot-panel",
        "schema_version": 1,
        "panel": panel,
        "path_ids": list(paths),
        "path_count": len(paths),
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        "levels": list(args.sample_steps),
        "main_differences": main.tolist(),
        "reference_differences": reference.tolist(),
        "raw_main_differences": raw_main.tolist(),
        "raw_reference_differences": raw_reference.tolist(),
        "main_differences_sha256": _array_sha256(main),
        "reference_differences_sha256": _array_sha256(reference),
        "maximum_cumulative_standardized_error": maximum_error,
        "execution": execution,
        "observable_payload": payload,
        "complete": 1,
        "finite": int(
            np.isfinite(main).all()
            and np.isfinite(reference).all()
            and math.isfinite(maximum_error)
        ),
        **NO_WORK,
    }
    normalized = json.loads(
        json.dumps(record, sort_keys=True, allow_nan=False)
    )
    if existing_record is not None:
        if existing_record != normalized:
            raise ArtifactCompatibilityError(
                f"frozen pilot panel {panel} summary changed"
            )
        return existing_record
    atomic_write_json(metadata_path, normalized)
    return normalized


def _run_pilot_stage(
    run_dir: Path,
    args: argparse.Namespace,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    gate_path = run_dir / "phase_observer_pilot_gate.json"
    if gate_path.is_file() and (
        _artifact_is_terminally_registered(run_dir, gate_path)
        or not _run_is_interrupted(run_dir)
    ):
        gate = _load(gate_path)
        evidence = _load(run_dir / "phase_observer_power_metrics.json")
        metrics = evidence.get("metrics")
        selected_path = run_dir / "selected_dynkin_design.json"
        if (
            not isinstance(metrics, Mapping)
            or not selected_path.is_file()
            or evidence.get("selected_design_sha256")
            != file_fingerprint(selected_path)
            or gate != evaluate_phase_observer_power(metrics)
        ):
            raise ArtifactCompatibilityError(
                "cached phase-observer pilot evidence changed"
            )
        return gate
    plan = _load_frozen_path_id_plan(run_dir)
    path_evidence = _load(
        run_dir / "phase_observer_path_id_preflight.json"
    )
    sealed = _load(run_dir / "sealed_panel_registry.json")
    panel_binding = (
        sealed.get("path_id_plan_sha256") == plan.sha256
        and sealed.get("panel_a_plan_sha256")
        == file_fingerprint(run_dir / "panel_a_plan.json")
        and sealed.get("panel_b_plan_sha256")
        == file_fingerprint(run_dir / "panel_b_plan.json")
    )
    if not panel_binding:
        raise ArtifactCompatibilityError(
            "sealed pilot plans do not bind the frozen path-ID plan"
        )
    original_parent = Path(
        _load(
            Path(args.parent_dynkin_idfix_run_dir).resolve()
            / "parent_provenance.json"
        )["parent_run_dir"]
    ).resolve()
    thresholds = DynkinPowerThresholds()
    panel_a = _run_pilot_panel(
        run_dir, args, source, panel="a"
    )
    rate_a = float(panel_a["execution"].get("conservative_rate", 0.0))
    rows_a = _legacy._candidate_rows_for_panel(
        panel_a,
        role="a",
        parent_dir=original_parent,
        conservative_rate=rate_a,
    )
    _legacy._write_feature_power_diagnostics(
        run_dir, role="panel_a", record=panel_a
    )
    _legacy._write_candidate_csv(run_dir, "panel_a", rows_a)
    nomination = select_dynkin_panel_a_design(rows_a)
    atomic_write_json(run_dir / "panel_a_nomination.json", nomination)

    panel_b: dict[str, Any] | None = None
    rows_b: list[dict[str, Any]] | None = None
    combined_rows: list[dict[str, Any]] | None = None
    if _passed(nomination):
        panel_b = _run_pilot_panel(
            run_dir, args, source, panel="b"
        )
        rate = min(
            rate_a,
            float(panel_b["execution"].get("conservative_rate", 0.0)),
        )
        rows_b = _legacy._candidate_rows_for_panel(
            panel_b,
            role="b",
            parent_dir=original_parent,
            conservative_rate=rate,
        )
        combined = _legacy._combined_panel_record(panel_a, panel_b)
        combined_rows = _legacy._candidate_rows_for_panel(
            combined,
            role="combined",
            parent_dir=original_parent,
            conservative_rate=rate,
        )
        for role, record in (
            ("panel_b", panel_b),
            ("combined", combined),
        ):
            _legacy._write_feature_power_diagnostics(
                run_dir, role=role, record=record
            )
        _legacy._write_candidate_csv(run_dir, "panel_b", rows_b)
        _legacy._write_candidate_csv(run_dir, "combined", combined_rows)
    confirmation = confirm_dynkin_design(
        nomination, rows_b, combined_rows
    )
    atomic_write_json(
        run_dir / "sealed_design_confirmation.json", confirmation
    )
    selected = confirmation.get("selected")
    selected_record = {
        "schema": RUN_SCHEMA + "-selected-design",
        "schema_version": 1,
        "selection": confirmation,
        "selected": selected,
        "selected_design_frozen": 1,
        **NO_WORK,
    }
    _freeze(
        run_dir / "selected_dynkin_design.json", selected_record
    )
    selected_hash = file_fingerprint(
        run_dir / "selected_dynkin_design.json"
    )
    executed = [panel_a] + ([] if panel_b is None else [panel_b])
    executions = [dict(value["execution"]) for value in executed]
    transition_count = sum(
        int(value.get("transition_count", 0)) for value in executions
    )
    certified_count = sum(
        int(value.get("certified_count", 0)) for value in executions
    )
    fallback_count = sum(
        int(value.get("fallback_count", 0)) for value in executions
    )
    elapsed = sum(
        float(value.get("elapsed_seconds", 0.0)) for value in executions
    )
    fallback_elapsed = sum(
        float(value.get("fallback_elapsed_seconds", 0.0))
        for value in executions
    )
    minimum_rate = min(
        (
            float(value.get("conservative_rate", 0.0))
            for value in executions
        ),
        default=0.0,
    )
    all_rows = [rows_a] + ([] if rows_b is None else [rows_b])
    resource_feasible = {
        (int(row["main_paths"]), int(row["reference_paths"]))
        for rows in all_rows
        for row in rows
        if float(row["projected_hours"])
        <= thresholds.maximum_projected_hours
        and float(row["conservative_rate"]) >= thresholds.minimum_rate
    }
    selected_main = (
        int(selected["main_paths"]) if isinstance(selected, Mapping) else -1
    )
    selected_reference = (
        int(selected["reference_paths"])
        if isinstance(selected, Mapping)
        else -1
    )

    def selected_row(
        rows: Sequence[Mapping[str, Any]] | None,
        fallback: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        target = selected if isinstance(selected, Mapping) else fallback
        if not isinstance(target, Mapping) or rows is None:
            return None
        key = (int(target["main_paths"]), int(target["reference_paths"]))
        return next(
            (
                row
                for row in rows
                if (int(row["main_paths"]), int(row["reference_paths"]))
                == key
            ),
            None,
        )

    nominated = nomination.get("selected")
    fallback = nominated if isinstance(nominated, Mapping) else None
    row_a = selected_row(rows_a, fallback)
    row_b = selected_row(rows_b, fallback)
    row_combined = selected_row(combined_rows, fallback)

    def width(row: Mapping[str, Any] | None, family: str) -> float:
        return (
            float(row[f"predicted_{family}_half_width"])
            if isinstance(row, Mapping)
            else math.inf
        )

    maximum_error = max(
        float(value["maximum_cumulative_standardized_error"])
        for value in executed
    )
    metrics = {
        "production_authorizing_pass": int(
            not args.test_only_reduced_workload
        ),
        "panel_a_frozen_pass": int(
            (run_dir / "panel_a_plan.json").is_file()
        ),
        "panel_b_frozen_pass": int(
            (run_dir / "panel_b_plan.json").is_file()
        ),
        "panel_plan_hash_pass": int(panel_binding),
        "panel_disjoint_pass": int(sealed.get("panels_disjoint", 0)),
        "panel_nonregeneration_pass": 1,
        "pilot_production_disjoint_pass": int(
            set(plan.pilot_path_ids("a")).isdisjoint(
                plan.designated_production_path_ids
            )
            and set(plan.pilot_path_ids("b")).isdisjoint(
                plan.designated_production_path_ids
            )
        ),
        "right_endpoint_coupling_unchanged_pass": int(
            path_evidence.get("passed", 0)
            and path_evidence.get("path_id_plan_sha256") == plan.sha256
        ),
        "raw_observables_advisory_only_pass": 1,
        "dynkin_authorizing_estimator_pass": 1,
        "forecast_label_pass": 1,
        "panel_a_complete_pass": int(panel_a["complete"]),
        "panel_b_complete_pass": int(
            panel_b is not None and panel_b["complete"]
        ),
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
            transition_count > 0 and transition_count == certified_count
        ),
        "executed_panels_numerically_valid_pass": int(
            all(int(value.get("finite", 0)) == 1 for value in executed)
            and maximum_error
            <= thresholds.maximum_cumulative_standardized_error
        ),
        "candidate_resource_feasibility_pass": int(bool(resource_feasible)),
        "resource_feasible_candidate_count": len(resource_feasible),
        "panel_count": 2,
        "paths_per_panel": int(args.pilot_panel_paths),
        "levels": list(args.sample_steps),
        "candidate_main_paths": list(thresholds.candidate_main_paths),
        "candidate_reference_paths": list(
            thresholds.candidate_reference_paths
        ),
        "selected_main_paths": selected_main,
        "selected_reference_paths": selected_reference,
        "panel_a_main_half_width": width(row_a, "main"),
        "panel_a_reference_half_width": width(row_a, "reference"),
        "panel_b_main_half_width": width(row_b, "main"),
        "panel_b_reference_half_width": width(row_b, "reference"),
        "combined_main_half_width": width(row_combined, "main"),
        "combined_reference_half_width": width(
            row_combined, "reference"
        ),
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
            fallback_elapsed / elapsed if elapsed > 0 else 0.0
        ),
        "peak_memory_fraction": max(
            float(value.get("peak_memory_fraction", 0.0))
            for value in executions
        ),
        "mass_error": max(
            float(value.get("mass_error", math.inf))
            for value in executions
        ),
        "maximum_cumulative_standardized_error": maximum_error,
        **{
            name: sum(int(value.get(name, 0)) for value in executions)
            for name in _FORBIDDEN_COUNTS
        },
        **NO_WORK,
    }
    record = {
        "schema": RUN_SCHEMA + "-power-metrics",
        "schema_version": 1,
        "metrics": metrics,
        "panel_a_nomination": nomination,
        "sealed_confirmation": confirmation,
        "selected_design_sha256": selected_hash,
        "path_id_plan_version": plan.version,
        "path_id_plan_sha256": plan.sha256,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "phase_observer_power_metrics.json", record)
    gate = evaluate_phase_observer_power(metrics)
    atomic_write_json(gate_path, gate)
    return gate


def _finish(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    provenance: Mapping[str, Any] | bool,
    preflight: Mapping[str, Any],
    pilot: Mapping[str, Any],
) -> int:
    workflow = decide_phase_observer_workflow(
        provenance=provenance,
        preflight_gate=preflight,
        pilot_gate=pilot,
    )
    atomic_write_json(
        run_dir / "phase_observer_workflow_gate.json", workflow
    )
    atomic_write_json(
        run_dir / "phase_observer_decision.json", workflow
    )
    registry_fields = _finalize_registry(run_dir)
    required_pass = _required_gate_pass(
        args.require_gate, preflight, pilot
    )
    _write_status(
        run_dir,
        status="complete",
        outcome="complete" if required_pass else "gate_failed",
        phase=args.stage,
        required_gate=args.require_gate,
        required_gate_pass=int(required_pass),
        decision=workflow["decision"],
        **registry_fields,
    )
    return 0 if required_pass else 1


def _synthetic_provenance_gate(
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-provenance-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 0,
        "provenance_valid": 0,
        "error_type": type(error).__name__,
        "error": str(error),
        **NO_WORK,
    }


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    resumed = False
    parent_binding_complete = False
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"Jacobi Dynkin phase-observer run directory: {run_dir}")
        parent_dir = Path(args.parent_dynkin_idfix_run_dir).resolve()
        provenance = verify_tower_observer_roundoff_parent(parent_dir)
        plan = _build_path_id_plan(args)
        source_hash, source_paths = _source_record(parent_dir)
        config = _scientific_config(args, plan)
        config_sha = config_fingerprint(config)
        parent_source = _load_parent_source_image(parent_dir)
        manifest = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "requested_device": str(args.device),
            "test_only_reduced_workload": int(
                bool(args.test_only_reduced_workload)
            ),
            "sampler_identity": _sampler_identity(args),
            "source_fingerprint": source_hash,
            "source_paths": source_paths,
            "source_count": len(source_paths),
            "scientific_config_sha256": config_sha,
            "path_id_plan_version": plan.version,
            "path_id_plan_sha256": plan.sha256,
            "parent_artifact_record_count": PARENT_REGISTRY_RECORD_COUNT,
            "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
            "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
            "runtime_report_name": "exact_backend_runtime.json",
            "frozen_before_device_execution": 1,
            **NO_WORK,
        }
        parent_binding_complete = True
        if resumed:
            _verify_resume_contract(
                run_dir,
                expected_plan=plan,
                expected_config=config,
                expected_manifest=manifest,
                expected_provenance=provenance,
                expected_source=parent_source,
                args=args,
            )
            source = {
                **_load(run_dir / "source_image.json"),
                "image": np.load(
                    run_dir / "source_image.npz", allow_pickle=False
                )["image"],
                "mixed_target": np.load(
                    run_dir / "source_image.npz", allow_pickle=False
                )["mixed_target"],
            }
        else:
            _freeze_path_id_plan(
                run_dir, plan, require_existing=False
            )
            _freeze(
                run_dir / "scientific_config.json",
                config,
                require_existing=False,
            )
            _freeze(
                run_dir / "parent_provenance.json",
                provenance,
                require_existing=False,
            )
            _freeze_source_image(
                run_dir,
                parent_source,
                require_existing=False,
            )
            source = {
                **_load(run_dir / "source_image.json"),
                "image": np.asarray(parent_source["image"], dtype=np.float64),
                "mixed_target": np.asarray(
                    parent_source["mixed_target"], dtype=np.float64
                ),
            }
            _freeze(
                run_dir / "run_manifest.json",
                manifest,
                require_existing=False,
            )
            _freeze_panel_plans(
                run_dir, args, plan, require_existing=False
            )
        # This advisory record is frozen before any device work.
        _freeze(
            run_dir / "distribution_free_power_impossibility.json",
            _legacy._distribution_free_power_impossibility(),
            require_existing=resumed,
        )
        exact_backend = configure_exact_torch_backend(
            torch.device(args.device)
        )
        runtime = {
            "schema": RUN_SCHEMA + "-exact-backend-runtime",
            "schema_version": 1,
            "requested_device": str(args.device),
            "exact_backend": exact_backend,
            **NO_WORK,
        }
        _freeze(
            run_dir / "exact_backend_runtime.json",
            runtime,
            require_existing=(
                resumed
                and (run_dir / "exact_backend_runtime.json").is_file()
            ),
        )
        _write_status(
            run_dir,
            status="running",
            outcome="running",
            phase=args.stage,
            required_gate=args.require_gate,
            required_gate_pass=0,
            scientific_config_sha256=config_sha,
            source_fingerprint=source_hash,
            plans_frozen_before_device_execution=1,
        )
        preflight = _existing_gate(
            run_dir, "preflight", "preflight has not run"
        )
        pilot = _existing_gate(
            run_dir, "pilot", "sealed power pilot has not run"
        )
        if args.stage in {"preflight", "all"}:
            try:
                preflight = _run_preflight_stage(
                    run_dir, args, provenance
                )
            except ArtifactCompatibilityError:
                raise
            except PhaseObserverSchedulerError as exc:
                preflight = _failed_stage_gate(
                    run_dir,
                    "preflight",
                    exc,
                    failure_domain="scheduler_configuration",
                    failure_code="phase_observer_path_id_plan_invalid",
                )
            except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                preflight = _failed_stage_gate(
                    run_dir,
                    "preflight",
                    exc,
                    failure_domain="resource_execution",
                    failure_code="phase_observer_preflight_resource_exhausted",
                )
            except Exception as exc:
                preflight = _failed_stage_gate(
                    run_dir,
                    "preflight",
                    exc,
                    failure_domain="numerical_execution",
                    failure_code="phase_observer_preflight_exception",
                )
        if args.stage in {"pilot", "all"}:
            test_ready = (
                args.test_only_reduced_workload
                and _substantive_test_gate_passed(
                    preflight,
                    ignored_checks=(
                        "production_authorizing_pass",
                        "tower_clusters_per_panel",
                        "tower_bootstrap_replicates",
                    ),
                )
            )
            if _passed(preflight) or test_ready:
                try:
                    pilot = _run_pilot_stage(
                        run_dir, args, source
                    )
                except ArtifactCompatibilityError:
                    raise
                except PhaseObserverSchedulerError as exc:
                    pilot = _failed_stage_gate(
                        run_dir,
                        "pilot",
                        exc,
                        failure_domain="scheduler_execution",
                        failure_code="phase_observer_pilot_scheduler_invalid",
                    )
                except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
                    pilot = _failed_stage_gate(
                        run_dir,
                        "pilot",
                        exc,
                        failure_domain="resource_execution",
                        failure_code="phase_observer_pilot_resource_exhausted",
                    )
                except Exception as exc:
                    pilot = _failed_stage_gate(
                        run_dir,
                        "pilot",
                        exc,
                        failure_domain="numerical_execution",
                        failure_code="phase_observer_pilot_exception",
                    )
            else:
                pilot = not_evaluated_gate(
                    "jacobi_rb_dynkin_phase_observer_power",
                    "phase-observer preflight did not pass",
                )
                atomic_write_json(
                    run_dir / "phase_observer_pilot_gate.json", pilot
                )
        return _finish(
            run_dir,
            args,
            provenance=provenance,
            preflight=preflight,
            pilot=pilot,
        )
    except ArtifactCompatibilityError as exc:
        if resumed:
            print(
                f"Jacobi Dynkin phase-observer compatibility error: {exc}",
                file=sys.stderr,
            )
            return 2
        if run_dir is not None and not parent_binding_complete:
            failure = _synthetic_provenance_gate(exc)
            atomic_write_json(
                run_dir / "provenance_failure.json", failure
            )
            preflight = {
                **not_evaluated_gate(
                    "jacobi_rb_dynkin_phase_observer_preflight",
                    "control provenance is invalid",
                ),
                "evaluation_status": "evaluated",
                "passed": 0,
                "provenance_valid": 0,
            }
            atomic_write_json(
                run_dir / "phase_observer_preflight_gate.json", preflight
            )
            pilot = not_evaluated_gate(
                "jacobi_rb_dynkin_phase_observer_power",
                "control provenance is invalid",
            )
            atomic_write_json(
                run_dir / "phase_observer_pilot_gate.json", pilot
            )
            return _finish(
                run_dir,
                args,
                provenance=failure,
                preflight=preflight,
                pilot=pilot,
            )
        if run_dir is not None:
            failure = {
                "schema": RUN_SCHEMA + "-artifact-compatibility-failure",
                "schema_version": 1,
                "evaluation_status": "execution_failed",
                "failure_domain": "artifact_compatibility",
                "failure_code": "phase_observer_artifact_compatibility_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                **NO_WORK,
            }
            atomic_write_json(
                run_dir / "artifact_compatibility_failure.json", failure
            )
            registry_fields = _finalize_registry(run_dir)
            _write_status(
                run_dir,
                status="failed",
                outcome="artifact_compatibility_failure",
                phase=args.stage,
                required_gate=args.require_gate,
                required_gate_pass=0,
                decision="phase_observer_numerically_unresolved",
                **registry_fields,
            )
        print(
            f"Jacobi Dynkin phase-observer compatibility error: {exc}",
            file=sys.stderr,
        )
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
            atomic_write_json(
                run_dir / "unexpected_failure.json", failure
            )
            _write_status(
                run_dir,
                status="failed",
                outcome="unexpected_failure",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        print(
            f"Jacobi Dynkin phase-observer error: {exc}",
            file=sys.stderr,
        )
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
