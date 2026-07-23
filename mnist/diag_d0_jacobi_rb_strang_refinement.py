"""Exact state-dependent Jacobi Strang-refinement controls.

This workflow verifies the temporal split used by the fixed-grid Eulerian
reference.  It consumes the immutable, passing certified multi-path Jacobi
kernel and target run; it does not import a neural trainer or reverse sampler.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import platform
from pathlib import Path
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
from mnist.d0_jacobi_rb_controls import (
    sample_alpha1_rb_transition_batch,
)
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    _philox_uniform_midpoint,
)
from mnist.d0_jacobi_rb_spectral import JacobiRBSpectralProfile
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    FINEST_SAMPLE_STEPS,
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
    evaluate_refinement_observables,
    legacy_k512_transition_ids,
    refinement_observable_spec,
    refinement_phase_exposure,
    run_refinement_shard,
)
from mnist.d0_jacobi_rb_strang_refinement_gate import (
    StrangRefinementThresholds,
    decide_strang_refinement_workflow,
    evaluate_refinement_power,
    evaluate_strang_preflight,
    evaluate_strang_refinement,
    not_evaluated_gate,
    select_refinement_design,
    whole_path_max_t_intervals,
    whole_path_refinement_bootstrap,
)
from mnist.d0_jacobi_rb_strang_refinement_provenance import (
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    verify_exact_jacobi_rb_multipath_parent,
)
from mnist.eulerian_flux_mnist import load_mnist_measure_dataset


RUN_SCHEMA = "experiment12-d0-jacobi-rb-strang-refinement"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "exact state-dependent fixed-grid Jacobi Strang refinement only"
ROOT_SEED = 261_151
DATASET_SEED = 260_718
EXPECTED_IMAGE_SHA256 = (
    "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
)
EXPECTED_MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
OBSERVATION_TIME_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
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


def _measure_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32).reshape(-1))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _passed(gate: Mapping[str, Any]) -> bool:
    return (
        gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
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
    t = StrangRefinementThresholds()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "power", "refinement", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-gate", choices=("none", "preflight", "power", "refinement"),
        default="none",
    )
    parser.add_argument("--parent-multipath-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_strang_refinement"),
    )
    parser.add_argument("--run-name", default="production-state-dependent-strang-refinement")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    parser.add_argument(
        "--sample-steps", type=_parse_int_tuple,
        default=tuple((*t.levels, t.reference_level)),
    )
    parser.add_argument("--steps-per-shard", type=int, default=t.restart_steps_per_shard)
    parser.add_argument(
        "--stationarity-panel-paths", type=int, default=t.preflight_paths_per_panel
    )
    parser.add_argument(
        "--stationarity-transitions-per-path",
        type=int,
        default=t.preflight_transitions_per_path,
    )
    parser.add_argument("--pilot-main-paths", type=int, default=t.pilot_main_paths)
    parser.add_argument(
        "--pilot-reference-paths", type=int, default=t.pilot_reference_paths
    )
    parser.add_argument("--bootstrap-reps", type=int, default=t.bootstrap_replicates)
    parser.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--examples-per-class", type=int, default=1000)
    parser.add_argument("--label", type=int, default=3)
    parser.add_argument("--class-index", type=int, default=0)
    parser.add_argument("--lambda-mix", type=float, default=0.35)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--test-only-reduced-workload", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.stage in {"power", "refinement", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "power": {"none", "preflight", "power"},
        "refinement": {"none", "preflight", "power", "refinement"},
        "report": {"none", "preflight", "power", "refinement"},
        "all": {"none", "preflight", "power", "refinement"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(
            f"--require-gate {args.require_gate} is unavailable at stage {args.stage}"
        )
    positive = (
        "steps_per_shard", "stationarity_panel_paths",
        "stationarity_transitions_per_path", "pilot_main_paths",
        "pilot_reference_paths", "bootstrap_reps", "examples_per_class",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    frozen = {
        "root_seed": ROOT_SEED,
        "sample_steps": tuple((*t.levels, t.reference_level)),
        "steps_per_shard": t.restart_steps_per_shard,
        "stationarity_panel_paths": t.preflight_paths_per_panel,
        "stationarity_transitions_per_path": t.preflight_transitions_per_path,
        "pilot_main_paths": t.pilot_main_paths,
        "pilot_reference_paths": t.pilot_reference_paths,
        "bootstrap_reps": t.bootstrap_replicates,
        "examples_per_class": 1000,
        "label": 3,
        "class_index": 0,
        "lambda_mix": 0.35,
    }
    changed = [name for name, value in frozen.items() if getattr(args, name) != value]
    if changed and not args.test_only_reduced_workload:
        parser.error(
            "production configuration is frozen; overrides require "
            "--test-only-reduced-workload: " + ", ".join(changed)
        )
    if args.test_only_reduced_workload and args.require_gate != "none":
        parser.error("test-only workloads cannot satisfy a required gate")
    if torch.device(args.device).type != "cuda" and not args.test_only_reduced_workload:
        parser.error("production refinement requires --device cuda")
    if int(args.steps_per_shard) != REFINEMENT_SHARD_STEPS:
        parser.error("the exact scheduler requires eight-step shards")
    if (
        int(args.pilot_reference_paths) > int(args.pilot_main_paths)
        or int(args.stationarity_transitions_per_path) < 2
    ):
        parser.error("invalid path-panel configuration")
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


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    t = StrangRefinementThresholds()
    return {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "claim_scope": CLAIM_SCOPE,
        "root_seed": int(args.root_seed),
        "dataset_seed": DATASET_SEED,
        "grid_size": t.grid_size,
        "alpha": t.alpha,
        "tau_eff": t.tau_eff,
        "sample_steps": list(args.sample_steps),
        "observation_time_fractions": list(t.observation_time_fractions),
        "steps_per_shard": int(args.steps_per_shard),
        "stationarity_panel_paths": int(args.stationarity_panel_paths),
        "stationarity_transitions_per_path": int(
            args.stationarity_transitions_per_path
        ),
        "pilot_main_paths": int(args.pilot_main_paths),
        "pilot_reference_paths": int(args.pilot_reference_paths),
        "candidate_main_paths": list(t.candidate_main_paths),
        "candidate_reference_paths": list(t.candidate_reference_paths),
        "bootstrap_reps": int(args.bootstrap_reps),
        "label": int(args.label),
        "class_index": int(args.class_index),
        "lambda_mix": float(args.lambda_mix),
        "expected_image_sha256": EXPECTED_IMAGE_SHA256,
        "expected_mixed_target_sha256": EXPECTED_MIXED_TARGET_SHA256,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "scheduler_version": REFINEMENT_SCHEDULER_VERSION,
        "transition_id_version": REFINEMENT_ID_VERSION,
        "rng_version": REFINEMENT_RNG_VERSION,
        "test_only_reduced_workload": int(args.test_only_reduced_workload),
        **NO_WORK,
    }


def _source_image(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_mnist_measure_dataset(
        args.data_root,
        max_train=args.max_train,
        examples_per_class=int(args.examples_per_class),
        download=bool(args.download),
        seed=DATASET_SEED,
    )
    images = np.asarray(dataset.train_images, dtype=np.float64)
    labels = np.asarray(dataset.train_labels, dtype=np.int64)
    candidates = np.flatnonzero(labels == int(args.label))
    if candidates.size == 0:
        raise ArtifactCompatibilityError(f"dataset has no label-{args.label} image")
    if not 0 <= int(args.class_index) < int(candidates.size):
        raise ArtifactCompatibilityError("source class index lies outside the dataset")
    dataset_index = int(candidates[int(args.class_index)])
    raw = np.asarray(images[dataset_index], dtype=np.float64).reshape(-1)
    if raw.size != PATH_STATE_SIZE or not np.isfinite(raw).all():
        raise ArtifactCompatibilityError("selected source image is invalid")
    image = np.maximum(raw, 0.0)
    image /= max(float(image.sum()), 1.0e-300)
    mixed = (1.0 - float(args.lambda_mix)) * image
    mixed += float(args.lambda_mix) / float(image.size)
    mixed /= float(mixed.sum())
    record = {
        "dataset_index": dataset_index,
        "class_index": int(args.class_index),
        "label": int(args.label),
        "lambda_mix": float(args.lambda_mix),
        "image": np.ascontiguousarray(image, dtype=np.float64),
        "mixed_target": np.ascontiguousarray(mixed, dtype=np.float64),
        "image_sha256": _measure_digest(image),
        "mixed_target_sha256": _measure_digest(mixed),
    }
    if not args.test_only_reduced_workload and (
        record["image_sha256"] != EXPECTED_IMAGE_SHA256
        or record["mixed_target_sha256"] != EXPECTED_MIXED_TARGET_SHA256
    ):
        raise ArtifactCompatibilityError(
            "source image SHA-256 does not match the frozen first label-3 image"
        )
    return record


def _source_record(parent_dir: Path) -> tuple[str, list[str]]:
    manifest = _load(parent_dir / "run_manifest.json")
    raw = manifest.get("source_paths")
    if not isinstance(raw, list) or len(raw) != 11:
        raise ArtifactCompatibilityError("parent manifest does not bind eleven sources")
    paths = {Path(str(value)).resolve() for value in raw}
    for module_name in (
        "mnist.d0_jacobi_rb_strang_refinement",
        "mnist.d0_jacobi_rb_strang_refinement_gate",
        "mnist.d0_jacobi_rb_strang_refinement_provenance",
    ):
        module = sys.modules[module_name]
        paths.add(Path(module.__file__).resolve())
    paths.add(Path(__file__).resolve())
    ordered = sorted(paths)
    return source_fingerprint(ordered), [str(path) for path in ordered]


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


def _recoverable_prefixes(stage: str) -> tuple[str, ...]:
    if stage == "power":
        return ("refinement_shards/power/",)
    if stage == "refinement":
        return ("refinement_shards/refinement/",)
    if stage == "all":
        return ("refinement_shards/power/", "refinement_shards/refinement/")
    return ()


def _verify_terminal_registry(run_dir: Path, *, stage: str) -> None:
    registry_path = run_dir / "artifact_registry.json"
    if not registry_path.is_file():
        return
    registry = _load(registry_path)
    status = _load(run_dir / "run_status.json")
    if status.get("artifact_registry_sha256") != file_fingerprint(registry_path):
        raise ArtifactCompatibilityError("resume status does not bind its registry")
    recoverable = _recoverable_prefixes(stage)
    mutable = {
        "strang_preflight_gate.json",
        "strang_power_gate.json",
        "strang_refinement_gate.json",
        "strang_workflow_gate.json",
        "strang_refinement_decision.json",
    }
    interrupted = status.get("status") == "running"
    records = dict(registry.get("records", {}))
    for relative, raw in records.items():
        path = run_dir / relative
        valid = (
            path.is_file()
            and isinstance(raw, Mapping)
            and raw.get("sha256") == file_fingerprint(path)
            and raw.get("size") == path.stat().st_size
        )
        if not valid and (
            relative.startswith(recoverable)
            or (interrupted and relative in mutable)
        ):
            continue
        if not valid:
            raise ArtifactCompatibilityError(f"resume artifact changed: {relative}")
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", ()))
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.relative_to(run_dir).as_posix() not in exclusions
    }
    unexpected = actual - set(records)
    invalid = {
        relative
        for relative in unexpected
        if not interrupted
        or (
            relative.startswith("refinement_shards/")
            and not relative.startswith(recoverable)
        )
    }
    if invalid:
        raise ArtifactCompatibilityError(
            "unregistered resume artifacts: " + ", ".join(sorted(invalid))
        )


def _existing_gate(run_dir: Path, name: str, reason: str) -> dict[str, Any]:
    path = run_dir / f"strang_{name}_gate.json"
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
    print(
        f"Jacobi Strang {label} {done}/{total} "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
    )


def _initial_dirichlet(path_count: int, seed: int) -> np.ndarray:
    return np.random.Generator(np.random.Philox(int(seed))).dirichlet(
        np.ones(PATH_STATE_SIZE, dtype=np.float64), size=int(path_count)
    )


def _shard_fingerprint(
    *,
    stage: str,
    panel: str,
    sample_steps: int,
    path_ids: Sequence[int],
    root_seed: int,
) -> str:
    return config_fingerprint(
        {
            "schema": RUN_SCHEMA + "-shard-input",
            "schema_version": 1,
            "scheduler_version": REFINEMENT_SCHEDULER_VERSION,
            "transition_id_version": REFINEMENT_ID_VERSION,
            "rng_version": REFINEMENT_RNG_VERSION,
            "phase_matchings": list(PHASE_MATCHINGS),
            "phase_durations": list(PHASE_DURATIONS),
            "tau_eff": TAU_EFF,
            "grid_spacing": GRID_SPACING,
            "cuda_profile": JacobiRBCudaProfile().to_dict(),
            "stage": stage,
            "panel": panel,
            "sample_steps": int(sample_steps),
            "path_ids": list(path_ids),
            "root_seed": int(root_seed),
            "step_count": REFINEMENT_SHARD_STEPS,
        }
    )


def _load_refinement_shard(
    path: Path,
    *,
    fingerprint: str,
    input_sha256: str,
    previous_chain_sha256: str,
) -> tuple[dict[str, Any], np.ndarray] | None:
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
            or not math.isfinite(
                float(row.get("complete_wall_upper_seconds", math.inf))
            )
            or float(row.get("complete_wall_upper_seconds", 0.0)) <= 0.0
        ):
            return None
        with np.load(state_path, allow_pickle=False) as archive:
            final = np.ascontiguousarray(archive["final_states"], dtype=np.float64)
        if (
            row.get("persisted_final_states_sha256") != _array_sha256(final)
            or final.ndim != 2
            or final.shape[1] != PATH_STATE_SIZE
            or not np.isfinite(final).all()
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
                "state_npz_sha256": row.get("state_npz_sha256"),
                "state_npz_size": row.get("state_npz_size"),
            }
        )
        if row.get("chain_sha256") != expected_chain:
            return None
        return dict(row), final
    except (
        ArtifactCompatibilityError,
        EOFError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ):
        return None


def _run_level_panel(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    stage: str,
    panel: str,
    initial_states: np.ndarray,
    path_ids: Sequence[int],
    sample_steps: int,
    root_seed: int,
    stop_steps: int | None = None,
    legacy_ids: bool = False,
    checkpoint_steps_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run and atomically persist one coupled level, grouped by eight paths."""

    initial = np.ascontiguousarray(initial_states, dtype=np.float64)
    paths = tuple(int(value) for value in path_ids)
    if initial.shape != (len(paths), PATH_STATE_SIZE):
        raise ValueError("panel states/path IDs have incompatible shapes")
    level = int(sample_steps)
    total_steps = level if stop_steps is None else int(stop_steps)
    if (
        level not in SUPPORTED_SAMPLE_STEPS
        or total_steps <= 0
        or total_steps > level
        or total_steps % REFINEMENT_SHARD_STEPS
    ):
        raise ValueError("invalid refinement level/stop-step plan")
    device = torch.device(args.device)
    profile = JacobiRBCudaProfile()
    checkpoint_steps = (
        tuple(
            int(round(level * fraction))
            for fraction in OBSERVATION_TIME_FRACTIONS
            if int(round(level * fraction)) <= total_steps
        )
        if checkpoint_steps_override is None
        else tuple(
            sorted(
                {
                    int(step)
                    for step in checkpoint_steps_override
                    if 0 < int(step) <= total_steps
                }
            )
        )
    )
    root = (
        run_dir
        / "refinement_shards"
        / stage
        / panel
        / f"K{level:04d}"
    )
    root.mkdir(parents=True, exist_ok=True)
    total_shards = math.ceil(len(paths) / MAX_REFINEMENT_PATHS_PER_GROUP) * (
        total_steps // REFINEMENT_SHARD_STEPS
    )
    done = 0
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    checkpoint_by_step: dict[int, dict[int, np.ndarray]] = {
        step: {} for step in checkpoint_steps
    }
    final_by_path: dict[int, np.ndarray] = {}
    for group_index, offset in enumerate(
        range(0, len(paths), MAX_REFINEMENT_PATHS_PER_GROUP)
    ):
        group_paths = paths[offset : offset + MAX_REFINEMENT_PATHS_PER_GROUP]
        group_initial = initial[offset : offset + len(group_paths)]
        states = torch.as_tensor(
            group_initial, dtype=torch.float64, device=device
        ).contiguous()
        committed = np.ascontiguousarray(group_initial)
        fingerprint = _shard_fingerprint(
            stage=stage,
            panel=panel,
            sample_steps=level,
            path_ids=group_paths,
            root_seed=root_seed,
        )
        previous_chain = config_fingerprint(
            {
                "kind": "strang-refinement-genesis",
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
            path = root / f"{base}.json"
            input_hash = _array_sha256(committed)
            loaded = (
                _load_refinement_shard(
                    path,
                    fingerprint=fingerprint,
                    input_sha256=input_hash,
                    previous_chain_sha256=previous_chain,
                )
                if reuse_tail and path.is_file()
                else None
            )
            if loaded is None:
                reuse_tail = False
                result = run_refinement_shard(
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
                )
                committed = np.array(
                    result.committed_final_states,
                    dtype=np.float64,
                    copy=True,
                    order="C",
                )
                state_path = path.with_suffix(".npz")
                _atomic_write_npz(state_path, final_states=committed)
                wall_before_metadata = max(
                    0.0, time.perf_counter() - shard_started
                )
                metadata_allowance = max(
                    0.05, 0.02 * wall_before_metadata
                )
                row = {
                    "stage": stage,
                    "panel": panel,
                    "sample_steps": level,
                    "group_index": group_index,
                    "path_ids": list(group_paths),
                    "start_step": start_step,
                    "step_count": REFINEMENT_SHARD_STEPS,
                    "input_states_sha256": input_hash,
                    "previous_chain_sha256": previous_chain,
                    "persisted_final_states_sha256": _array_sha256(committed),
                    "state_npz_name": state_path.name,
                    "state_npz_sha256": file_fingerprint(state_path),
                    "state_npz_size": state_path.stat().st_size,
                    "wall_before_metadata_seconds": wall_before_metadata,
                    "metadata_commit_timing_allowance_seconds": (
                        metadata_allowance
                    ),
                    "complete_wall_upper_seconds": (
                        wall_before_metadata + metadata_allowance
                    ),
                    **result.to_record(),
                    **NO_WORK,
                }
                row["chain_sha256"] = config_fingerprint(
                    {
                        "fingerprint": fingerprint,
                        "input_states_sha256": input_hash,
                        "previous_chain_sha256": previous_chain,
                        "batch_output_sha256": row["batch_output_sha256"],
                        "batch_final_state_sha256": row["batch_final_state_sha256"],
                        "batch_certificate_sha256": row[
                            "batch_certificate_sha256"
                        ],
                        "state_npz_sha256": row["state_npz_sha256"],
                        "state_npz_size": row["state_npz_size"],
                    }
                )
                atomic_write_json(
                    path,
                    {
                        "input_fingerprint": fingerprint,
                        "row": row,
                        "row_sha256": config_fingerprint(row),
                        **NO_WORK,
                    },
                )
            else:
                row, committed = loaded
            states = torch.as_tensor(
                committed, dtype=torch.float64, device=device
            ).contiguous()
            previous_chain = str(row["chain_sha256"])
            for checkpoint in row.get("observable_checkpoints", ()):
                if not isinstance(checkpoint, Mapping):
                    raise ArtifactCompatibilityError("invalid observable checkpoint")
                step = int(checkpoint["completed_step"])
                values = np.asarray(checkpoint["values"], dtype=np.float64)
                ids = tuple(int(value) for value in checkpoint["path_ids"])
                if values.shape != (len(ids), 10):
                    raise ArtifactCompatibilityError("invalid checkpoint shape")
                for path_id, value in zip(ids, values, strict=True):
                    checkpoint_by_step.setdefault(step, {})[path_id] = value
            rows.append(row)
            done += 1
            _progress(args, f"{stage}/{panel}/K{level}", done, total_shards, started)
        for path_id, value in zip(group_paths, committed, strict=True):
            final_by_path[path_id] = value

    ordered_final = np.stack([final_by_path[path_id] for path_id in paths])
    checkpoint_values = {
        step: np.stack([checkpoint_by_step[step][path_id] for path_id in paths])
        for step in checkpoint_steps
    }
    diagnostics = [dict(row["diagnostics"]) for row in rows]
    shard_chain_pass = True
    for row in rows:
        base = (
            f"group-{int(row['group_index']):03d}-"
            f"steps-{int(row['start_step']):04d}-"
            f"{int(row['start_step']) + REFINEMENT_SHARD_STEPS - 1:04d}"
        )
        metadata_path = root / f"{base}.json"
        row_paths = tuple(int(value) for value in row["path_ids"])
        shard_chain_pass &= (
            _load_refinement_shard(
                metadata_path,
                fingerprint=_shard_fingerprint(
                    stage=stage,
                    panel=panel,
                    sample_steps=level,
                    path_ids=row_paths,
                    root_seed=root_seed,
                ),
                input_sha256=str(row["input_states_sha256"]),
                previous_chain_sha256=str(row["previous_chain_sha256"]),
            )
            is not None
        )
    transition_count = sum(int(row.get("transition_count", 0)) for row in diagnostics)
    certified_count = sum(int(row.get("certified_count", 0)) for row in diagnostics)
    fallback_count = sum(int(row.get("fallback_count", 0)) for row in diagnostics)
    elapsed = sum(float(row.get("elapsed_seconds", 0.0)) for row in diagnostics)
    wall_elapsed = max(0.0, time.perf_counter() - started)
    complete_wall_upper = sum(
        float(row.get("complete_wall_upper_seconds", math.inf)) for row in rows
    )
    result = {
        "sample_steps": level,
        "path_ids": list(paths),
        "path_count": len(paths),
        "stop_steps": total_steps,
        "initial_states_sha256": _array_sha256(initial),
        "final_states_sha256": _array_sha256(ordered_final),
        "final_states": ordered_final,
        "checkpoint_values": checkpoint_values,
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
        "fallback_elapsed_seconds": sum(
            float(row.get("fallback_elapsed_seconds", 0.0)) for row in diagnostics
        ),
        "elapsed_seconds": elapsed,
        "wall_elapsed_seconds": wall_elapsed,
        "complete_wall_upper_seconds": complete_wall_upper,
        "transitions_per_second": transition_count / elapsed if elapsed > 0 else 0.0,
        "wall_transitions_per_second": (
            transition_count / wall_elapsed if wall_elapsed > 0 else 0.0
        ),
        "mass_error": max(
            [float(row.get("maximum_global_simplex_error", math.inf)) for row in diagnostics],
            default=math.inf,
        ),
        "maximum_launch_lanes": max(
            [int(row.get("maximum_cuda_launch_lanes", 0)) for row in diagnostics],
            default=0,
        ),
        "state_updates_device_resident_pass": int(
            bool(diagnostics)
            and all(
                int(row.get("state_updates_device_resident", 0)) == 1
                and int(row.get("in_shard_host_roundtrip_count", -1)) == 0
                for row in diagnostics
            )
        ),
        "shard_chain_pass": int(shard_chain_pass),
        **{
            name: sum(int(row.get(name, 0)) for row in diagnostics)
            for name in _FORBIDDEN_COUNTS
        },
    }
    return result


def _local_generator_fixture(state: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    """Compare exact Jacobi eigenmoments with the local Eulerian generator."""

    grid_size = 28
    count = grid_size * grid_size
    h = 1.0 / grid_size
    values = np.asarray(state, dtype=np.float64).reshape(-1)
    values = np.maximum(values, 0.0)
    values /= values.sum()
    spec = refinement_observable_spec(grid_size)
    weights = np.asarray(spec.fourier_weights, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    for matching in build_four_color_matchings(grid_size):
        tail = values[matching.tails]
        head = values[matching.heads]
        pair_total = tail + head
        if np.any(pair_total <= 0.0):
            return math.inf, []
        x = head / pair_total
        maximum_exposure = 5.0e-5
        physical_step = (
            maximum_exposure * h * h * float(np.min(pair_total)) / 3.0
        )

        def exact_rates(step: float) -> np.ndarray:
            exposure = 3.0 * step / (h * h * pair_total)
            delta_m1 = (x - 0.5) * np.expm1(-2.0 * exposure)
            p2 = 6.0 * x * x - 6.0 * x + 1.0
            delta_p2 = p2 * np.expm1(-6.0 * exposure)
            delta_m2 = (delta_p2 + 6.0 * delta_m1) / 6.0
            linear = np.sum(
                pair_total[None, :]
                * (weights[:, matching.heads] - weights[:, matching.tails])
                * delta_m1[None, :],
                axis=1,
            )
            quadratic = np.sum(
                pair_total**2 * (-2.0 * delta_m1 + 2.0 * delta_m2)
            )
            cubic = np.sum(
                pair_total**3 * (-3.0 * delta_m1 + 3.0 * delta_m2)
            )
            return np.concatenate((linear, (quadratic, cubic))) / step

        measured = 2.0 * exact_rates(0.5 * physical_step) - exact_rates(
            physical_step
        )
        difference = head - tail
        directional = np.column_stack(
            (
                (weights[:, matching.heads] - weights[:, matching.tails]).T,
                2.0 * difference,
                3.0 * (head * head - tail * tail),
            )
        )
        directional_second = np.column_stack(
            (
                np.zeros((pair_total.size, 8), dtype=np.float64),
                np.full_like(pair_total, 4.0),
                6.0 * pair_total,
            )
        )
        analytic = 3.0 / (h * h) * np.sum(
            (tail * head / pair_total)[:, None] * directional_second
            + ((tail - head) / pair_total)[:, None] * directional,
            axis=0,
        )
        for index, name in enumerate(spec.names):
            error = abs(float(measured[index] - analytic[index])) / max(
                1.0, abs(float(analytic[index]))
            )
            errors.append(error)
            rows.append(
                {
                    "matching": matching.name,
                    "observable": name,
                    "analytic_generator": float(analytic[index]),
                    "richardson_semigroup_generator": float(measured[index]),
                    "relative_error": error,
                    "maximum_phase_exposure": maximum_exposure,
                }
            )
    return max(errors, default=math.inf), rows


def _legendre(degree: int, value: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(int(degree) + 1, dtype=np.float64)
    coefficients[-1] = 1.0
    return np.polynomial.legendre.legval(value, coefficients)


def _powered_stationarity_panel(
    *,
    path_count: int,
    transitions_per_path: int,
    root_seed: int,
    reps: int,
    path_id_start: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    count = int(path_count) * int(transitions_per_path)
    rng = np.random.Generator(np.random.Philox(int(root_seed)))
    earlier = rng.random(count)
    exposure = np.full(count, 0.5, dtype=np.float64)
    result = sample_alpha1_rb_transition_batch(
        earlier,
        exposure,
        rng_key=(int(root_seed), "powered-stationarity"),
        profile=JacobiRBSpectralProfile(require_correct_rounding=True),
    )
    later = np.asarray(result.later_head_fraction, dtype=np.float64)
    zx = 2.0 * earlier - 1.0
    zy = 2.0 * later - 1.0
    path_values: dict[str, np.ndarray] = {}
    for degree in range(1, 9):
        path_values[f"stationarity_degree_{degree}"] = _legendre(
            degree, zy
        ).reshape(path_count, transitions_per_path).mean(axis=1)
    reverse = _legendre(1, zx) * _legendre(2, zy)
    reverse -= _legendre(2, zx) * _legendre(1, zy)
    path_values["reversibility_witness"] = reverse.reshape(
        path_count, transitions_per_path
    ).mean(axis=1)
    feature_names = tuple(sorted(path_values))
    path_matrix = np.column_stack([path_values[name] for name in feature_names])
    path_ids = np.arange(
        int(path_id_start),
        int(path_id_start) + int(path_count),
        dtype=np.int64,
    )
    inference = whole_path_max_t_intervals(
        path_values,
        seed=int(root_seed) + 1,
        confidence=0.99,
        reps=max(100, int(reps)),
    )
    replay = sample_alpha1_rb_transition_batch(
        earlier,
        exposure,
        rng_key=(int(root_seed), "powered-stationarity"),
        profile=JacobiRBSpectralProfile(require_correct_rounding=True),
    )
    replay_pass = (
        np.array_equal(result.later_head_fraction, replay.later_head_fraction)
        and np.array_equal(result.denoising_target, replay.denoising_target)
        and np.array_equal(result.certificate_codes, replay.certificate_codes)
    )
    rows = [
        {
            "name": member["name"],
            "mean_minus_expected": member["mean_minus_expected"],
            "standard_error": member["standard_error"],
            "simultaneous_lower": member["simultaneous_lower"],
            "simultaneous_upper": member["simultaneous_upper"],
            "contains_zero": member["contains_zero"],
        }
        for member in inference["members"]
    ]
    record = {
        "passed": int(inference["passed"] and replay_pass),
        "replay_pass": int(replay_pass),
        "backend": "certified-spectral-cpu-alpha1",
        "root_seed": int(root_seed),
        "rng_namespace": "powered-stationarity",
        "sample_count": count,
        "path_count": path_count,
        "transitions_per_path": transitions_per_path,
        "path_id_start": int(path_id_start),
        "path_ids_sha256": _array_sha256(path_ids),
        "earlier_fraction_sha256": _array_sha256(
            earlier.reshape(path_count, transitions_per_path)
        ),
        "inference": inference,
        "later_fraction_sha256": _array_sha256(
            later.reshape(path_count, transitions_per_path)
        ),
        "denoising_target_sha256": _array_sha256(
            np.asarray(result.denoising_target).reshape(
                path_count, transitions_per_path
            )
        ),
        "certificate_codes_sha256": _array_sha256(
            np.asarray(result.certificate_codes).reshape(
                path_count, transitions_per_path
            )
        ),
        "path_matrix_sha256": _array_sha256(path_matrix),
        "path_feature_names": list(feature_names),
        **NO_WORK,
    }
    payload = {
        "path_ids": path_ids,
        "earlier_fraction": earlier.reshape(path_count, transitions_per_path),
        "later_fraction": later.reshape(path_count, transitions_per_path),
        "denoising_target": np.asarray(result.denoising_target).reshape(
            path_count, transitions_per_path
        ),
        "certificate_codes": np.asarray(result.certificate_codes).reshape(
            path_count, transitions_per_path
        ),
        "path_matrix": path_matrix,
    }
    return record, rows, payload


def _load_frozen_powered_stationarity_panel(
    metadata_path: Path,
    npz_path: Path,
    *,
    root_seed: int,
    path_id_start: int,
    path_count: int,
    transitions_per_path: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read an immutable powered panel without regenerating inspected paths."""

    record = _load(metadata_path)
    expected_plan = config_fingerprint(
        {
            "root_seed": int(root_seed),
            "rng_namespace": "powered-stationarity",
            "path_id_start": int(path_id_start),
            "path_count": int(path_count),
            "transitions_per_path": int(transitions_per_path),
        }
    )
    if (
        record.get("root_seed") != int(root_seed)
        or record.get("path_id_start") != int(path_id_start)
        or record.get("path_count") != int(path_count)
        or record.get("transitions_per_path") != int(transitions_per_path)
        or record.get("panel_plan_sha256") != expected_plan
        or record.get("npz_name") != npz_path.name
        or record.get("npz_sha256") != file_fingerprint(npz_path)
        or record.get("npz_size") != npz_path.stat().st_size
    ):
        raise ArtifactCompatibilityError(
            f"frozen powered stationarity panel changed: {metadata_path.name}"
        )
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    expected_shapes = {
        "path_ids": (path_count,),
        "earlier_fraction": (path_count, transitions_per_path),
        "later_fraction": (path_count, transitions_per_path),
        "denoising_target": (path_count, transitions_per_path),
        "certificate_codes": (path_count, transitions_per_path),
        "path_matrix": (path_count, 9),
    }
    if set(arrays) != set(expected_shapes) or any(
        arrays[name].shape != shape for name, shape in expected_shapes.items()
    ):
        raise ArtifactCompatibilityError("frozen stationarity panel payload is invalid")
    checks = {
        "path_ids_sha256": _array_sha256(arrays["path_ids"]),
        "earlier_fraction_sha256": _array_sha256(arrays["earlier_fraction"]),
        "later_fraction_sha256": _array_sha256(arrays["later_fraction"]),
        "denoising_target_sha256": _array_sha256(arrays["denoising_target"]),
        "certificate_codes_sha256": _array_sha256(arrays["certificate_codes"]),
        "path_matrix_sha256": _array_sha256(arrays["path_matrix"]),
    }
    if any(record.get(name) != digest for name, digest in checks.items()):
        raise ArtifactCompatibilityError("frozen stationarity panel array hash changed")
    expected_ids = np.arange(
        path_id_start, path_id_start + path_count, dtype=np.int64
    )
    if not np.array_equal(arrays["path_ids"], expected_ids):
        raise ArtifactCompatibilityError("frozen stationarity path IDs changed")
    inference = record.get("inference")
    if not isinstance(inference, Mapping) or not isinstance(
        inference.get("members"), list
    ):
        raise ArtifactCompatibilityError("frozen stationarity inference is invalid")
    rows = [
        {
            "name": member["name"],
            "mean_minus_expected": member["mean_minus_expected"],
            "standard_error": member["standard_error"],
            "simultaneous_lower": member["simultaneous_lower"],
            "simultaneous_upper": member["simultaneous_upper"],
            "contains_zero": member["contains_zero"],
        }
        for member in inference["members"]
    ]
    return record, rows


def _id_preflight(device: torch.device) -> dict[str, Any]:
    path_ids = (17, 29)
    uniqueness = True
    aliasing = True
    order = True
    resume = True
    tick_sets: dict[int, set[int]] = {}
    for level in SUPPORTED_SAMPLE_STEPS:
        stride = FINEST_SAMPLE_STEPS // level
        ticks = {(step + 1) * stride - 1 for step in range(level)}
        tick_sets[level] = ticks
        uniqueness &= len(ticks) == level
        uniqueness &= min(ticks) >= 0 and max(ticks) < FINEST_SAMPLE_STEPS
        # Exercise endpoints and an interior step on-device.  Exhaustive
        # uniqueness follows from the disjoint frozen bit fields and the
        # injective right-endpoint tick map above.
        for step in sorted({0, level // 2, level - 1}):
            for phase in range(len(PHASE_MATCHINGS)):
                ids = canonical_refinement_transition_ids(
                    path_ids,
                    sample_steps=level,
                    outer_step=step,
                    phase=phase,
                    device=device,
                ).reshape(len(path_ids), EDGES_PER_PHASE)
                uniqueness &= int(torch.unique(ids).numel()) == int(ids.numel())
                reversed_ids = canonical_refinement_transition_ids(
                    tuple(reversed(path_ids)),
                    sample_steps=level,
                    outer_step=step,
                    phase=phase,
                    device=device,
                ).reshape(len(path_ids), EDGES_PER_PHASE).flip(0)
                order &= torch.equal(ids, reversed_ids)
                resume &= torch.equal(
                    ids,
                    canonical_refinement_transition_ids(
                        path_ids,
                        sample_steps=level,
                        outer_step=step,
                        phase=phase,
                        device=device,
                    ).reshape(len(path_ids), EDGES_PER_PHASE),
                )
    for coarse in SUPPORTED_SAMPLE_STEPS[:-1]:
        ratio = FINEST_SAMPLE_STEPS // coarse
        # Right endpoints are the only intentional aliases.
        aliasing &= tick_sets[coarse] <= tick_sets[FINEST_SAMPLE_STEPS]
        aliasing &= len(tick_sets[coarse]) == coarse
        aliasing &= ratio >= 2
    # Philox uses the transition ID as a counter coordinate.  Reusing an
    # aligned counter therefore reuses exactly one quantile, while replacing
    # the counter by a disjoint independently namespaced value leaves the
    # one-event uniform marginal unchanged.  Exercise both constructions on
    # a frozen 4,096-word panel in addition to the analytic injectivity proof.
    from scipy import stats

    proof_ids = canonical_refinement_transition_ids(
        tuple(range(8)),
        sample_steps=FINEST_SAMPLE_STEPS,
        outer_step=FINEST_SAMPLE_STEPS - 1,
        phase=len(PHASE_MATCHINGS) - 1,
        device=device,
    ).detach().cpu().numpy().astype(np.uint64, copy=False)
    # The full call has 3,136 lanes; add a disjoint phase to reach at least
    # 4,096 deterministic counters without changing the construction.
    extra_ids = canonical_refinement_transition_ids(
        tuple(range(8)),
        sample_steps=FINEST_SAMPLE_STEPS,
        outer_step=FINEST_SAMPLE_STEPS - 2,
        phase=len(PHASE_MATCHINGS) - 1,
        device=device,
    ).detach().cpu().numpy().astype(np.uint64, copy=False)
    law_ids = np.concatenate((proof_ids, extra_ids))[:4096]
    independent_ids = law_ids ^ np.uint64(1 << 62)
    key = (ROOT_SEED, REFINEMENT_RNG_VERSION, "id-marginal-proof")
    coupled_uniform = np.asarray(
        [_philox_uniform_midpoint(key, int(value)) for value in law_ids],
        dtype=np.float64,
    )
    independent_uniform = np.asarray(
        [_philox_uniform_midpoint(key, int(value)) for value in independent_ids],
        dtype=np.float64,
    )
    sample_count = int(law_ids.size)
    one_sample_limit = math.sqrt(
        math.log(2.0 / 0.01) / (2.0 * sample_count)
    )
    two_sample_limit = math.sqrt(math.log(4.0 / 0.01) / sample_count)
    coupled_ks = float(stats.kstest(coupled_uniform, "uniform").statistic)
    independent_ks = float(
        stats.kstest(independent_uniform, "uniform").statistic
    )
    paired_ks = float(
        stats.ks_2samp(
            coupled_uniform, independent_uniform, alternative="two-sided"
        ).statistic
    )
    aligned_ids = canonical_refinement_transition_ids(
        (17,),
        sample_steps=512,
        outer_step=0,
        phase=0,
        device=device,
    )
    aligned_tick = canonical_refinement_transition_ids(
        (17,),
        sample_steps=2048,
        outer_step=3,
        phase=0,
        device=device,
    )
    aligned_ids_equal = torch.equal(aligned_ids, aligned_tick)
    aligned_words_equal = bool(
        aligned_ids_equal
        and all(
            _philox_uniform_midpoint(key, int(left))
            == _philox_uniform_midpoint(key, int(right))
            for left, right in zip(
                aligned_ids[:64].cpu().tolist(),
                aligned_tick[:64].cpu().tolist(),
                strict=True,
            )
        )
    )
    marginal = bool(
        sample_count == 4096
        and aligned_words_equal
        and np.all((coupled_uniform > 0.0) & (coupled_uniform < 1.0))
        and np.all((independent_uniform > 0.0) & (independent_uniform < 1.0))
        and coupled_ks <= one_sample_limit
        and independent_ks <= one_sample_limit
        and paired_ks <= two_sample_limit
    )
    return {
        "nested_id_uniqueness_pass": int(uniqueness),
        "nested_id_aliasing_exact_pass": int(aliasing),
        "nested_id_marginal_law_pass": int(marginal),
        "nested_id_order_invariance_pass": int(order),
        "nested_id_resume_invariance_pass": int(resume),
        "nested_id_aligned_philox_word_pass": int(aligned_words_equal),
        "marginal_law_sample_count": sample_count,
        "coupled_uniform_ks": coupled_ks,
        "independent_uniform_ks": independent_ks,
        "coupled_independent_ks": paired_ks,
        "one_sample_ks_99_limit": one_sample_limit,
        "two_sample_ks_99_limit": two_sample_limit,
        "coupled_uniform_sha256": _array_sha256(coupled_uniform),
        "independent_uniform_sha256": _array_sha256(independent_uniform),
    }


def _legacy_k512_replay(
    run_dir: Path, args: argparse.Namespace
) -> tuple[bool, dict[str, Any]]:
    parent_root = Path(args.parent_multipath_run_dir)
    parent_path = (
        parent_root
        / "multipath_shards"
        / "kernel"
        / "b10-repeat-00-steps-000-007.json"
    )
    parent = _load(parent_path)["row"]
    initial_all = _initial_dirichlet(10, 261_141 + 20_000 + 10)
    path_ids = tuple(20_000 + index for index in range(8))
    result = _run_level_panel(
        run_dir,
        args,
        stage="preflight",
        panel="legacy-k512-replay",
        initial_states=initial_all[:8],
        path_ids=path_ids,
        sample_steps=512,
        root_seed=261_141,
        stop_steps=8,
        legacy_ids=True,
    )
    replay_records = {
        int(record["path_id"]): record
        for row in result["rows"]
        for record in row["path_records"]
    }
    parent_records = {
        int(record["path_id"]): record
        for record in parent["path_records"]
        if int(record["path_id"]) in path_ids
    }
    passed = (
        set(replay_records) == set(parent_records)
        and all(
            replay_records[path_id]["output_sha256"]
            == parent_records[path_id]["output_sha256"]
            and replay_records[path_id]["final_state_sha256"]
            == parent_records[path_id]["final_state_sha256"]
            for path_id in path_ids
        )
    )
    record = {
        "passed": int(passed),
        "path_ids": list(path_ids),
        "parent_shard_sha256": file_fingerprint(parent_path),
        "replay_final_states_sha256": result["final_states_sha256"],
        **NO_WORK,
    }
    atomic_write_json(run_dir / "legacy_k512_replay.json", record)
    return passed, result


def _run_preflight_stage(
    run_dir: Path,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    id_metrics = _id_preflight(device)
    atomic_write_json(
        run_dir / "refinement_id_plan.json",
        {
            "schema": RUN_SCHEMA + "-id-plan",
            "schema_version": 1,
            "finest_sample_steps": FINEST_SAMPLE_STEPS,
            "transition_id_version": REFINEMENT_ID_VERSION,
            "rng_version": REFINEMENT_RNG_VERSION,
            "right_endpoint_tick_formula": "(k+1)*(2048/K)-1",
            "metrics": id_metrics,
            **NO_WORK,
        },
    )
    pair = torch.tensor([0.2, 0.4], dtype=torch.float64, device=device)
    exposure = refinement_phase_exposure(
        pair, sample_steps=512, duration_fraction=0.5
    )
    expected = 3.0 * (TAU_EFF / 512.0) * 0.5 / (
        GRID_SPACING * GRID_SPACING * pair
    )
    variable_exposure_pass = torch.equal(exposure, expected)
    phase_pass = (
        tuple(PHASE_MATCHINGS) == (0, 1, 2, 3, 2, 1, 0)
        and tuple(PHASE_DURATIONS) == (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
    )
    legacy_pass, legacy_result = _legacy_k512_replay(run_dir, args)

    interior_index = np.arange(PATH_STATE_SIZE, dtype=np.float64)
    interior = 1.0 + 0.22 * np.cos(
        2.0 * math.pi * interior_index / PATH_STATE_SIZE
    )
    interior += 0.13 * np.sin(6.0 * math.pi * interior_index / PATH_STATE_SIZE)
    interior /= interior.sum()
    mixed_error, mixed_rows = _local_generator_fixture(
        np.asarray(source["mixed_target"], dtype=np.float64)
    )
    interior_error, interior_rows = _local_generator_fixture(interior)
    atomic_write_csv(
        run_dir / "local_generator_controls.csv",
        [
            {"fixture": "mixed_digit", **row} for row in mixed_rows
        ]
        + [{"fixture": "smooth_interior", **row} for row in interior_rows],
    )

    support_initial = _initial_dirichlet(8, int(args.root_seed) + 1_000)
    support: dict[int, dict[str, Any]] = {}
    for level in (1024, 2048):
        support[level] = _run_level_panel(
            run_dir,
            args,
            stage="preflight",
            panel=f"support-k{level}",
            initial_states=support_initial,
            path_ids=tuple(100_000 + index for index in range(8)),
            sample_steps=level,
            root_seed=int(args.root_seed),
            stop_steps=8,
        )

    stationarity: list[dict[str, Any]] = []
    stationarity_rows: list[dict[str, Any]] = []
    stationarity_npz_paths: list[Path] = []
    for index, name in enumerate(("a", "b")):
        panel_seed = int(args.root_seed) + 2_000 + 100 * index
        path_id_start = 700_000 + 100_000 * index
        metadata_path = run_dir / f"stationarity_panel_{name}.json"
        npz_path = run_dir / f"powered_stationarity_panel_{name}.npz"
        if metadata_path.exists() or npz_path.exists():
            if not metadata_path.is_file() or not npz_path.is_file():
                raise ArtifactCompatibilityError(
                    f"incomplete frozen stationarity panel {name}"
                )
            panel, rows = _load_frozen_powered_stationarity_panel(
                metadata_path,
                npz_path,
                root_seed=panel_seed,
                path_id_start=path_id_start,
                path_count=int(args.stationarity_panel_paths),
                transitions_per_path=int(args.stationarity_transitions_per_path),
            )
        else:
            panel, rows, payload = _powered_stationarity_panel(
                path_count=int(args.stationarity_panel_paths),
                transitions_per_path=int(args.stationarity_transitions_per_path),
                root_seed=panel_seed,
                reps=int(args.bootstrap_reps),
                path_id_start=path_id_start,
            )
            _atomic_write_npz(npz_path, **payload)
            panel = {
                **panel,
                "npz_name": npz_path.name,
                "npz_sha256": file_fingerprint(npz_path),
                "npz_size": npz_path.stat().st_size,
                "panel_plan_sha256": config_fingerprint(
                    {
                        "root_seed": panel["root_seed"],
                        "rng_namespace": panel["rng_namespace"],
                        "path_id_start": panel["path_id_start"],
                        "path_count": panel["path_count"],
                        "transitions_per_path": panel[
                            "transitions_per_path"
                        ],
                    }
                ),
            }
            _freeze(metadata_path, panel)
        stationarity.append(panel)
        stationarity_npz_paths.append(npz_path)
        stationarity_rows.extend(
            {"panel": name, **row} for row in rows
        )
    atomic_write_csv(
        run_dir / "powered_stationarity_intervals.csv", stationarity_rows
    )

    results = [legacy_result, *support.values()]
    transitions = sum(int(result["transition_count"]) for result in results)
    certified = sum(int(result["certified_count"]) for result in results)
    fallbacks = sum(int(result["fallback_count"]) for result in results)
    fallback_seconds = sum(
        float(result["fallback_elapsed_seconds"]) for result in results
    )
    elapsed = sum(float(result["elapsed_seconds"]) for result in results)
    peak_fraction = (
        torch.cuda.max_memory_allocated(device)
        / torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else 0.0
    )
    metrics = {
        "control_provenance_pass": int(provenance.get("passed", 0)),
        "parent_record_count": int(
            provenance.get("parent_artifact_record_count", -1)
        ),
        "parent_kernel_gate_pass": int(provenance.get("parent_kernel_pass", 0)),
        "parent_target_gate_pass": int(provenance.get("parent_target_pass", 0)),
        "parent_strang_authorized_pass": int(
            provenance.get("state_dependent_strang_refinement_authorized", 0)
        ),
        "eleven_parent_sources_immutable_pass": int(
            provenance.get("parent_source_count", 0) == 11
        ),
        "parent_scientific_config_pass": int(
            bool(provenance.get("parent_scientific_config_sha256"))
        ),
        "parent_no_work_pass": int(
            not provenance.get("physical_training_authorized", 0)
            and not provenance.get("sampling_authorized", 0)
        ),
        "grid_size": 28,
        "alpha": 1.0,
        "tau_eff": TAU_EFF,
        "levels": list(SUPPORTED_SAMPLE_STEPS),
        "variable_k_exposure_pass": int(variable_exposure_pass),
        "palindromic_phase_order_pass": int(phase_pass),
        **id_metrics,
        "legacy_k512_replay_pass": int(legacy_pass),
        "local_generator_mixed_digit_pass": int(mixed_error <= 1.0e-8),
        "local_generator_interior_fixture_pass": int(interior_error <= 1.0e-8),
        "local_generator_max_error": max(mixed_error, interior_error),
        "k1024_support_certificate_pass": int(
            support[1024]["certificate_fraction"] == 1.0
        ),
        "k2048_support_certificate_pass": int(
            support[2048]["certificate_fraction"] == 1.0
        ),
        "stationarity_panel_a_pass": int(stationarity[0]["passed"]),
        "stationarity_panel_b_pass": int(stationarity[1]["passed"]),
        "stationarity_joint_max_t_pass": int(
            all(panel["passed"] for panel in stationarity)
        ),
        "stationarity_panels_immutable_pass": int(
            all(
                path.is_file()
                and panel.get("npz_sha256") == file_fingerprint(path)
                and panel.get("npz_size") == path.stat().st_size
                for panel, path in zip(
                    stationarity, stationarity_npz_paths, strict=True
                )
            )
        ),
        "stationarity_panel_disjoint_pass": int(
            set(
                range(
                    int(stationarity[0]["path_id_start"]),
                    int(stationarity[0]["path_id_start"])
                    + int(stationarity[0]["path_count"]),
                )
            ).isdisjoint(
                range(
                    int(stationarity[1]["path_id_start"]),
                    int(stationarity[1]["path_id_start"])
                    + int(stationarity[1]["path_count"]),
                )
            )
            and stationarity[0]["root_seed"] != stationarity[1]["root_seed"]
            and stationarity[0]["panel_plan_sha256"]
            != stationarity[1]["panel_plan_sha256"]
            and
            stationarity[0]["later_fraction_sha256"]
            != stationarity[1]["later_fraction_sha256"]
        ),
        "stationarity_panel_count": 2,
        "stationarity_paths_per_panel": int(args.stationarity_panel_paths),
        "stationarity_transitions_per_path": int(
            args.stationarity_transitions_per_path
        ),
        "minimum_support_rate": min(
            support[1024]["transitions_per_second"],
            support[2048]["transitions_per_second"],
        ),
        "certificate_fraction": certified / transitions if transitions else 0.0,
        "fallback_fraction": fallbacks / transitions if transitions else 0.0,
        "fallback_cost_fraction": fallback_seconds / elapsed if elapsed else 0.0,
        "peak_memory_fraction": float(peak_fraction),
        **{
            name: sum(int(result.get(name, 0)) for result in results)
            for name in _FORBIDDEN_COUNTS
        },
        **NO_WORK,
    }
    gate = evaluate_strang_preflight(metrics)
    atomic_write_json(
        run_dir / "preflight_metrics.json",
        {"schema": RUN_SCHEMA + "-preflight-metrics", "metrics": metrics, **NO_WORK},
    )
    atomic_write_json(run_dir / "strang_preflight_gate.json", gate)
    return gate


def _checkpoint_tensor(result: Mapping[str, Any]) -> np.ndarray:
    level = int(result["sample_steps"])
    blocks = []
    for fraction in OBSERVATION_TIME_FRACTIONS:
        step = int(round(level * fraction))
        values = result["checkpoint_values"].get(step)
        if values is None:
            raise ArtifactCompatibilityError(
                f"K={level} lacks the frozen t/T={fraction} checkpoint"
            )
        blocks.append(np.asarray(values, dtype=np.float64))
    return np.stack(blocks, axis=1)


def _aggregate_execution(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    transitions = sum(int(result.get("transition_count", 0)) for result in results)
    certified = sum(int(result.get("certified_count", 0)) for result in results)
    fallback = sum(int(result.get("fallback_count", 0)) for result in results)
    elapsed = sum(float(result.get("elapsed_seconds", 0.0)) for result in results)
    wall_elapsed = sum(
        float(result.get("wall_elapsed_seconds", math.inf)) for result in results
    )
    complete_wall_upper = sum(
        float(result.get("complete_wall_upper_seconds", math.inf))
        for result in results
    )
    return {
        "transition_count": transitions,
        "certified_count": certified,
        "certificate_fraction": certified / transitions if transitions else 0.0,
        "fallback_count": fallback,
        "fallback_fraction": fallback / transitions if transitions else 0.0,
        "elapsed_seconds": elapsed,
        "wall_elapsed_seconds": wall_elapsed,
        "complete_wall_upper_seconds": complete_wall_upper,
        "mass_error": max(
            (float(result.get("mass_error", math.inf)) for result in results),
            default=math.inf,
        ),
        "state_updates_device_resident_pass": int(
            bool(results)
            and all(
                int(result.get("state_updates_device_resident_pass", 0)) == 1
                for result in results
            )
        ),
        "shard_chain_pass": int(
            bool(results)
            and all(int(result.get("shard_chain_pass", 0)) == 1 for result in results)
        ),
        **{
            name: sum(int(result.get(name, 0)) for result in results)
            for name in _FORBIDDEN_COUNTS
        },
    }


def _run_power_stage(
    run_dir: Path,
    args: argparse.Namespace,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    t = StrangRefinementThresholds()
    main_paths = int(args.pilot_main_paths)
    reference_paths = int(args.pilot_reference_paths)
    initial = np.repeat(
        np.asarray(source["mixed_target"], dtype=np.float64)[None, :],
        main_paths,
        axis=0,
    )
    path_ids = tuple(300_000 + index for index in range(main_paths))
    results: dict[int, dict[str, Any]] = {}
    for level in (512, 1024):
        results[level] = _run_level_panel(
            run_dir,
            args,
            stage="power",
            panel="variance-only-pilot",
            initial_states=initial,
            path_ids=path_ids,
            sample_steps=level,
            root_seed=int(args.root_seed),
        )
    results[2048] = _run_level_panel(
        run_dir,
        args,
        stage="power",
        panel="variance-only-pilot",
        initial_states=initial[:reference_paths],
        path_ids=path_ids[:reference_paths],
        sample_steps=2048,
        root_seed=int(args.root_seed),
    )
    values = {level: _checkpoint_tensor(result) for level, result in results.items()}
    _atomic_write_npz(
        run_dir / "pilot_observables.npz",
        k512=values[512],
        k1024=values[1024],
        k2048=values[2048],
        main_path_ids=np.asarray(path_ids, dtype=np.int64),
        reference_path_ids=np.asarray(path_ids[:reference_paths], dtype=np.int64),
    )
    feature_count = int(values[512].shape[1] * values[512].shape[2])
    main_delta = (
        values[512].reshape(main_paths, -1)
        - values[1024].reshape(main_paths, -1)
    )
    ref_high = (
        4.0 * values[2048].reshape(reference_paths, -1)
        - values[1024][:reference_paths].reshape(reference_paths, -1)
    ) / 3.0
    reference_delta = (
        values[512][:reference_paths].reshape(reference_paths, -1) - ref_high
    )
    from scipy import stats

    def sd_upper(sample: np.ndarray) -> np.ndarray:
        count = sample.shape[0]
        if count < 2:
            return np.full(sample.shape[1], math.inf)
        variance = np.var(sample, axis=0, ddof=1)
        # One-sided 99% familywise upper envelope for all frozen features.
        # The chi-square construction is the predeclared normal-variance
        # approximation; Bonferroni allocation prevents forty marginal
        # statements from masquerading as one simultaneous bound.
        lower_chi = float(
            stats.chi2.ppf(0.01 / float(feature_count), count - 1)
        )
        if not lower_chi > 0.0:
            return np.full(sample.shape[1], math.inf)
        return np.sqrt((count - 1) * variance / lower_chi)

    main_sd = sd_upper(main_delta)
    reference_sd = sd_upper(reference_delta)
    critical = math.sqrt(2.0 * math.log(2.0 * feature_count / 0.01))
    complete_panel_rates = np.asarray(
        [
            float(result["transition_count"])
            / max(float(result["complete_wall_upper_seconds"]), 1.0e-300)
            for result in results.values()
        ],
        dtype=np.float64,
    )
    if (
        complete_panel_rates.size != len(results)
        or not np.isfinite(complete_panel_rates).all()
    ):
        conservative_rate = 0.0
    else:
        # Each panel rate includes exact CUDA execution plus atomic shard I/O.
        # Use the slowest complete panel directly; no optimistic mean-rate
        # extrapolation enters the authorizing 48-hour projection.
        conservative_rate = max(0.0, float(np.min(complete_panel_rates)))
    pilot_id_set = set(path_ids)
    preflight_id_set = (
        set(range(20_000, 20_008))
        | set(range(100_000, 100_008))
        | set(
            range(
                700_000,
                700_000 + int(args.stationarity_panel_paths),
            )
        )
        | set(
            range(
                800_000,
                800_000 + int(args.stationarity_panel_paths),
            )
        )
    )
    production_id_set = set(
        range(400_000, 400_000 + max(t.candidate_main_paths))
    )
    pilot_preflight_disjoint = pilot_id_set.isdisjoint(preflight_id_set)
    pilot_production_disjoint = pilot_id_set.isdisjoint(production_id_set)
    variance_inputs_valid = bool(
        np.isfinite(main_sd).all()
        and np.isfinite(reference_sd).all()
        and conservative_rate > 0.0
    )
    candidates = []
    for candidate_main in t.candidate_main_paths:
        for candidate_reference in t.candidate_reference_paths:
            if candidate_reference > candidate_main:
                continue
            target_path_steps = (
                candidate_main * (128 + 256 + 512 + 1024)
                + candidate_reference * 2048
                + 2
                * int(args.stationarity_panel_paths)
                * len(SUPPORTED_SAMPLE_STEPS)
                * REFINEMENT_SHARD_STEPS
            )
            candidates.append(
                {
                    "main_paths": candidate_main,
                    "reference_paths": candidate_reference,
                    "predicted_main_half_width": float(
                        critical * np.max(main_sd) / math.sqrt(candidate_main)
                    ),
                    "predicted_reference_half_width": float(
                        critical
                        * np.max(reference_sd)
                        / math.sqrt(candidate_reference)
                    ),
                    "projected_hours": float(
                        target_path_steps
                        * len(PHASE_MATCHINGS)
                        * EDGES_PER_PHASE
                        / conservative_rate
                        / 3600.0
                        if conservative_rate > 0.0
                        else math.inf
                    ),
                    "variance_upper_confidence": 0.99,
                    "variance_bound": "normal-chi-square-bonferroni",
                    "variance_family_size": feature_count,
                    "timing_lower_confidence": 0.99,
                    "conservative_rate": conservative_rate,
                    "timing_bound": "minimum-complete-panel-rate-including-io",
                    "variance_only_pass": int(variance_inputs_valid),
                    "pilot_production_isolation_pass": int(
                        pilot_production_disjoint
                    ),
                    "pilot_means_excluded_pass": 1,
                }
            )
    selection = select_refinement_design(candidates)
    pilot_means_excluded = all(
        not any(
            token in row
            for token in (
                "pilot_mean",
                "point_estimate",
                "observed_difference",
            )
        )
        for row in candidates
    )
    _freeze(run_dir / "selected_refinement_design.json", selection)
    selection_sha256 = file_fingerprint(
        run_dir / "selected_refinement_design.json"
    )
    atomic_write_csv(run_dir / "refinement_design_candidates.csv", candidates)
    execution = _aggregate_execution(list(results.values()))
    selected = selection.get("selected")
    selected_record = dict(selected) if isinstance(selected, Mapping) else {}
    metrics = {
        "pilot_complete_pass": int(len(results) == 3),
        "pilot_finite_pass": int(
            all(np.isfinite(value).all() for value in values.values())
        ),
        "pilot_paths_disjoint_from_preflight_pass": int(
            pilot_preflight_disjoint
        ),
        "pilot_paths_disjoint_from_production_pass": int(
            pilot_production_disjoint
        ),
        "pilot_means_excluded_pass": int(pilot_means_excluded),
        "variance_only_selection_pass": int(selection.get("passed", 0)),
        "complete_candidate_grid_pass": int(
            len(candidates)
            == len(
                [
                    (main, reference)
                    for main in t.candidate_main_paths
                    for reference in t.candidate_reference_paths
                    if reference <= main
                ]
            )
        ),
        "selected_design_frozen_pass": int(
            (run_dir / "selected_refinement_design.json").is_file()
        ),
        "selected_design_hash_pass": int(
            len(selection_sha256) == 64
        ),
        "selected_design_sha256": selection_sha256,
        "pilot_certification_pass": int(
            execution["certificate_fraction"] == 1.0
        ),
        "pilot_main_paths": main_paths,
        "pilot_reference_paths": reference_paths,
        "candidate_main_paths": list(t.candidate_main_paths),
        "candidate_reference_paths": list(t.candidate_reference_paths),
        "selected_main_paths": selected_record.get("main_paths", -1),
        "selected_reference_paths": selected_record.get("reference_paths", -1),
        "predicted_main_half_width": selected_record.get(
            "predicted_main_half_width", math.inf
        ),
        "predicted_reference_half_width": selected_record.get(
            "predicted_reference_half_width", math.inf
        ),
        "projected_production_hours": selected_record.get(
            "projected_hours", math.inf
        ),
        "certificate_fraction": execution["certificate_fraction"],
        **{name: execution[name] for name in _FORBIDDEN_COUNTS},
        **NO_WORK,
    }
    gate = evaluate_refinement_power(metrics)
    atomic_write_json(
        run_dir / "power_metrics.json",
        {
            "schema": RUN_SCHEMA + "-power-metrics",
            "metrics": metrics,
            "selection": selection,
            "pilot_execution": execution,
            **NO_WORK,
        },
    )
    atomic_write_json(run_dir / "strang_power_gate.json", gate)
    return gate


def _stationarity_sweep_panel(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    panel: str,
    path_count: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    initial = _initial_dirichlet(path_count, seed)
    initial_observables = np.asarray(
        evaluate_refinement_observables(initial), dtype=np.float64
    )
    panel_offset = 500_000 if panel == "a" else 600_000
    path_ids = tuple(panel_offset + index for index in range(path_count))
    path_values: dict[str, np.ndarray] = {}
    for feature in range(initial_observables.shape[1]):
        path_values[
            f"initial_stationarity_exact_moment_f{feature:02d}"
        ] = initial_observables[:, feature]
    results: list[dict[str, Any]] = []
    saved_values: dict[str, np.ndarray] = {"initial": initial_observables}
    for level in SUPPORTED_SAMPLE_STEPS:
        result = _run_level_panel(
            run_dir,
            args,
            stage="refinement",
            panel=f"stationarity-{panel}",
            initial_states=initial,
            path_ids=path_ids,
            sample_steps=level,
            root_seed=int(args.root_seed) + 20_000 + seed,
            stop_steps=8,
            checkpoint_steps_override=(1, 8),
        )
        results.append(result)
        one = np.asarray(result["checkpoint_values"][1], dtype=np.float64)
        eight = np.asarray(result["checkpoint_values"][8], dtype=np.float64)
        saved_values[f"k{level}_one"] = one
        saved_values[f"k{level}_eight"] = eight
        for feature in range(one.shape[1]):
            path_values[
                f"k{level}_stationarity_exact_moment_f{feature:02d}"
            ] = one[:, feature]
            path_values[f"k{level}_stationarity_f{feature:02d}"] = (
                one[:, feature] - initial_observables[:, feature]
            )
        for pair_index, (left, right) in enumerate(
            ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        ):
            path_values[f"k{level}_balance_w{pair_index:02d}"] = (
                initial_observables[:, left] * one[:, right]
                - one[:, left] * initial_observables[:, right]
            )
        if level == 512:
            for feature in range(eight.shape[1]):
                path_values[f"k512_eight_sweep_f{feature:02d}"] = (
                    eight[:, feature] - initial_observables[:, feature]
                )
    inference = whole_path_max_t_intervals(
        path_values,
        seed=int(seed) + 71,
        confidence=0.99,
        reps=int(args.bootstrap_reps),
    )
    payload = {
        "path_ids": np.asarray(path_ids, dtype=np.int64),
        **saved_values,
    }
    npz_path = run_dir / f"stationarity_sweep_panel_{panel}.npz"
    metadata_path = run_dir / f"stationarity_sweep_panel_{panel}.json"
    existing = _load(metadata_path) if metadata_path.is_file() else None
    if npz_path.exists() or existing is not None:
        if not npz_path.is_file() or existing is None:
            raise ArtifactCompatibilityError(
                f"incomplete frozen full-sweep stationarity panel {panel}"
            )
        if (
            existing.get("npz_sha256") != file_fingerprint(npz_path)
            or existing.get("npz_size") != npz_path.stat().st_size
        ):
            raise ArtifactCompatibilityError(
                f"frozen full-sweep stationarity panel {panel} changed"
            )
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(payload) or any(
                not np.array_equal(
                    np.ascontiguousarray(archive[name]), payload[name]
                )
                for name in payload
            ):
                raise ArtifactCompatibilityError(
                    f"full-sweep stationarity arrays changed for panel {panel}"
                )
    else:
        _atomic_write_npz(npz_path, **payload)
    rows = [
        {"panel": panel, **dict(member)} for member in inference["members"]
    ]
    record = {
        "panel": panel,
        "path_count": path_count,
        "inference": inference,
        "passed": int(inference["passed"]),
        "path_ids_sha256": _array_sha256(np.asarray(path_ids, dtype=np.int64)),
        "npz_name": npz_path.name,
        "npz_sha256": file_fingerprint(npz_path),
        "npz_size": npz_path.stat().st_size,
        "panel_plan_sha256": config_fingerprint(
            {
                "panel": panel,
                "seed": int(seed),
                "path_count": int(path_count),
                "path_ids_sha256": _array_sha256(
                    np.asarray(path_ids, dtype=np.int64)
                ),
                "sample_steps": list(SUPPORTED_SAMPLE_STEPS),
                "one_sweep_levels": list(SUPPORTED_SAMPLE_STEPS),
                "eight_sweep_level": 512,
            }
        ),
        **NO_WORK,
    }
    _freeze(metadata_path, record)
    return record, rows, results


def _bootstrap_family_pass(
    result: Mapping[str, Any], thresholds: StrangRefinementThresholds
) -> bool:
    if int(result.get("valid", 0)) != 1:
        return False
    rows = result.get("feature_metrics")
    if not isinstance(rows, list) or not rows:
        return False
    return (
        all(
            float(row["observed_weak_order"])
            >= thresholds.minimum_observed_weak_order
            and float(row["weak_order_interval_lower"])
            >= thresholds.minimum_weak_order_interval_lower
            and int(row["weak_order_interval_contains_two"]) == 1
            for row in rows
        )
        and float(result["simultaneous_512_1024_upper_bound"])
        <= thresholds.maximum_512_1024_discrepancy
        and float(result["simultaneous_512_reference_upper_bound"])
        <= thresholds.maximum_512_reference_error
        and float(result["simultaneous_reference_stability_upper_bound"])
        <= thresholds.maximum_reference_instability
    )


def _write_refinement_plot(
    run_dir: Path, values: Mapping[int, np.ndarray]
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    levels = sorted(values)
    means = np.asarray(
        [np.mean(values[level], axis=0).reshape(-1) for level in levels]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(levels, means[:, -2], marker="o", label="quadratic")
    axes[0].plot(levels, means[:, -1], marker="o", label="cubic")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("standardized final-time mean")
    axes[0].legend()
    differences = np.max(np.abs(np.diff(means, axis=0)), axis=1)
    axes[1].plot(levels[:-1], differences, marker="o")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("coarser K")
    axes[1].set_ylabel("max successive mean difference")
    figure.tight_layout()
    target = run_dir / "strang_refinement.png"
    temporary = target.with_name(target.name + ".tmp.png")
    figure.savefig(temporary, dpi=150, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(target)


def _run_refinement_stage(
    run_dir: Path,
    args: argparse.Namespace,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    t = StrangRefinementThresholds()
    selection_path = run_dir / "selected_refinement_design.json"
    selection = _load(selection_path)
    power_record = _load(run_dir / "power_metrics.json")
    power_metrics = power_record.get("metrics")
    if not isinstance(power_metrics, Mapping):
        raise ArtifactCompatibilityError("power metrics are invalid")
    selected_binding_pass = (
        power_metrics.get("selected_design_sha256")
        == file_fingerprint(selection_path)
    )
    selected_raw = selection.get("selected")
    if not isinstance(selected_raw, Mapping) or int(selection.get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("power stage did not freeze an eligible design")
    selected = dict(selected_raw)
    main_paths = int(selected["main_paths"])
    reference_paths = int(selected["reference_paths"])
    if main_paths not in t.candidate_main_paths or reference_paths not in t.candidate_reference_paths:
        raise ArtifactCompatibilityError("selected refinement design is outside the frozen grid")
    initial = np.repeat(
        np.asarray(source["mixed_target"], dtype=np.float64)[None, :],
        main_paths,
        axis=0,
    )
    path_ids = tuple(400_000 + index for index in range(main_paths))
    level_results: dict[int, dict[str, Any]] = {}
    for level in (128, 256, 512, 1024):
        level_results[level] = _run_level_panel(
            run_dir,
            args,
            stage="refinement",
            panel="mixed-digit-production",
            initial_states=initial,
            path_ids=path_ids,
            sample_steps=level,
            root_seed=int(args.root_seed),
        )
    level_results[2048] = _run_level_panel(
        run_dir,
        args,
        stage="refinement",
        panel="mixed-digit-production",
        initial_states=initial[:reference_paths],
        path_ids=path_ids[:reference_paths],
        sample_steps=2048,
        root_seed=int(args.root_seed),
    )
    values = {
        level: _checkpoint_tensor(result)
        for level, result in level_results.items()
    }
    _atomic_write_npz(
        run_dir / "production_refinement_observables.npz",
        **{f"k{level}": value for level, value in values.items()},
        main_path_ids=np.asarray(path_ids, dtype=np.int64),
        reference_path_ids=np.asarray(path_ids[:reference_paths], dtype=np.int64),
    )

    family_indices = {
        "linear": [
            time_index * 10 + feature
            for time_index in range(4)
            for feature in range(8)
        ],
        "quadratic": [time_index * 10 + 8 for time_index in range(4)],
        "cubic": [time_index * 10 + 9 for time_index in range(4)],
        "pooled": list(range(40)),
    }
    bootstraps: dict[str, dict[str, Any]] = {}
    bootstrap_rows: list[dict[str, Any]] = []
    for family_index, (family, indices) in enumerate(family_indices.items()):
        family_values = {
            level: value.reshape(value.shape[0], -1)[:, indices]
            for level, value in values.items()
        }
        bootstrap = whole_path_refinement_bootstrap(
            family_values,
            seed=int(args.root_seed) + 30_000 + family_index,
            reps=int(args.bootstrap_reps),
            confidence=0.99,
            order_confidence=0.90,
        )
        bootstraps[family] = bootstrap
        bootstrap_rows.extend(
            {"family": family, **dict(row)}
            for row in bootstrap.get("feature_metrics", ())
        )
    atomic_write_json(
        run_dir / "refinement_bootstrap.json",
        {
            "schema": RUN_SCHEMA + "-bootstrap",
            "families": bootstraps,
            **NO_WORK,
        },
    )
    atomic_write_csv(run_dir / "refinement_bootstrap_features.csv", bootstrap_rows)

    stationarity_records: list[dict[str, Any]] = []
    stationarity_rows: list[dict[str, Any]] = []
    stationarity_results: list[dict[str, Any]] = []
    stationarity_paths = int(args.stationarity_panel_paths)
    for index, name in enumerate(("a", "b")):
        record, rows, execution = _stationarity_sweep_panel(
            run_dir,
            args,
            panel=name,
            path_count=stationarity_paths,
            seed=int(args.root_seed) + 40_000 + 1_000 * index,
        )
        stationarity_records.append(record)
        stationarity_rows.extend(rows)
        stationarity_results.extend(execution)
    atomic_write_csv(
        run_dir / "strang_stationarity_detailed_balance.csv",
        stationarity_rows,
    )

    main_execution = _aggregate_execution(list(level_results.values()))
    stationarity_execution = _aggregate_execution(stationarity_results)
    all_execution = _aggregate_execution(
        [*level_results.values(), *stationarity_results]
    )
    valid_bootstraps = [
        value
        for value in bootstraps.values()
        if int(value.get("valid", 0)) == 1
    ]
    feature_rows = [
        dict(row)
        for value in valid_bootstraps
        for row in value.get("feature_metrics", ())
    ]
    order_values = [
        float(row["observed_weak_order"]) for row in feature_rows
    ]
    order_lowers = [
        float(row["weak_order_interval_lower"]) for row in feature_rows
    ]
    order_coverage = [
        int(row["weak_order_interval_contains_two"]) for row in feature_rows
    ]
    family_passes = {
        family: _bootstrap_family_pass(value, t)
        for family, value in bootstraps.items()
    }
    inference_members = [
        member
        for record in stationarity_records
        for member in record["inference"]["members"]
    ]
    stationarity_only = [
        member for member in inference_members if "stationarity" in member["name"]
    ]
    eight_sweep = [
        member for member in inference_members if "eight_sweep" in member["name"]
    ]
    balance = [
        member for member in inference_members if "balance" in member["name"]
    ]
    all_stationary = lambda rows: bool(rows) and all(
        int(row["contains_zero"]) == 1 for row in rows
    )
    projected_hours = float(selected.get("projected_hours", math.inf))
    actual_hours = all_execution["complete_wall_upper_seconds"] / 3600.0
    spec = refinement_observable_spec()
    moments_valid = (
        spec.names[-2:] == ("quadratic_mass", "cubic_mass")
        and all(
            math.isfinite(moment.mean)
            and math.isfinite(moment.variance)
            and moment.variance > 0.0
            for moment in spec.moments
        )
    )
    pilot_ids = set(
        range(300_000, 300_000 + int(args.pilot_main_paths))
    )
    stationarity_ids = set(
        range(500_000, 500_000 + stationarity_paths)
    ) | set(range(600_000, 600_000 + stationarity_paths))
    production_isolated = set(path_ids).isdisjoint(
        pilot_ids | stationarity_ids
    )
    metrics = {
        "selected_design_binding_pass": int(
            selected_binding_pass and selection.get("selection_status") == "selected"
        ),
        "production_pilot_isolation_pass": int(production_isolated),
        "production_complete_pass": int(len(level_results) == 5),
        "production_finite_pass": int(
            all(np.isfinite(value).all() for value in values.values())
        ),
        "all_levels_complete_pass": int(
            tuple(sorted(level_results)) == SUPPORTED_SAMPLE_STEPS
        ),
        "reference_level_subset_pass": int(
            level_results[2048]["path_ids"]
            == level_results[1024]["path_ids"][:reference_paths]
        ),
        "observation_plan_pass": int(
            all(value.shape[1:] == (4, 10) for value in values.values())
        ),
        "observable_family_plan_pass": int(
            tuple(spec.families)
            == ("linear",) * 8 + ("quadratic", "cubic")
        ),
        "dirichlet_normalization_pass": int(moments_valid),
        "paired_whole_path_bootstrap_pass": int(
            len(valid_bootstraps) == len(bootstraps)
        ),
        **{
            f"{family}_family_pass": int(passed)
            for family, passed in family_passes.items()
        },
        "stationarity_panel_a_pass": int(stationarity_records[0]["passed"]),
        "stationarity_panel_b_pass": int(stationarity_records[1]["passed"]),
        "stationarity_all_levels_pass": int(
            all_stationary(stationarity_only)
        ),
        "stationarity_eight_sweep_k512_pass": int(all_stationary(eight_sweep)),
        "detailed_balance_max_t_pass": int(all_stationary(balance)),
        "mass_conservation_pass": int(
            all_execution["mass_error"] <= t.maximum_cuda_mass_error
        ),
        "shard_chain_pass": all_execution["shard_chain_pass"],
        "state_updates_device_resident_pass": all_execution[
            "state_updates_device_resident_pass"
        ],
        "image_sha256": source["image_sha256"],
        "levels": list(SUPPORTED_SAMPLE_STEPS),
        "observation_time_fractions": list(OBSERVATION_TIME_FRACTIONS),
        "bootstrap_replicates": int(args.bootstrap_reps),
        "bootstrap_confidence": 0.99,
        "minimum_observed_weak_order": min(order_values, default=-math.inf),
        "minimum_weak_order_interval_lower": min(
            order_lowers, default=-math.inf
        ),
        "weak_order_two_coverage_fraction": (
            sum(order_coverage) / len(order_coverage)
            if order_coverage
            else 0.0
        ),
        "maximum_512_1024_upper_bound": max(
            (
                float(value.get("simultaneous_512_1024_upper_bound", math.inf))
                for value in bootstraps.values()
            ),
            default=math.inf,
        ),
        "maximum_512_reference_upper_bound": max(
            (
                float(value.get("simultaneous_512_reference_upper_bound", math.inf))
                for value in bootstraps.values()
            ),
            default=math.inf,
        ),
        "maximum_reference_stability_upper_bound": max(
            (
                float(
                    value.get(
                        "simultaneous_reference_stability_upper_bound", math.inf
                    )
                )
                for value in bootstraps.values()
            ),
            default=math.inf,
        ),
        "mass_error": all_execution["mass_error"],
        "certificate_fraction": all_execution["certificate_fraction"],
        "projected_or_actual_hours": max(projected_hours, actual_hours),
        **{name: all_execution[name] for name in _FORBIDDEN_COUNTS},
        **NO_WORK,
    }
    _write_refinement_plot(run_dir, values)
    gate = evaluate_strang_refinement(metrics)
    atomic_write_json(
        run_dir / "refinement_metrics.json",
        {
            "schema": RUN_SCHEMA + "-refinement-metrics",
            "metrics": metrics,
            "selected_design": selected,
            "main_execution": main_execution,
            "stationarity_execution": stationarity_execution,
            **NO_WORK,
        },
    )
    atomic_write_json(run_dir / "strang_refinement_gate.json", gate)
    return gate


def _failed_stage_gate(
    run_dir: Path, name: str, exc: Exception
) -> dict[str, Any]:
    atomic_write_json(
        run_dir / f"{name}_failure.json",
        {
            "schema": RUN_SCHEMA + "-stage-failure",
            "schema_version": 1,
            "stage": name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            **NO_WORK,
        },
    )
    gate = {
        "schema": RUN_SCHEMA + "-gate",
        "schema_version": 1,
        "gate": f"strang_{name}",
        "claim_scope": "stage failed before complete evidence",
        "evaluation_status": "evaluated",
        "passed": 0,
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
    atomic_write_json(run_dir / f"strang_{name}_gate.json", gate)
    return gate


def _required_gate_pass(
    required: str, gates: Mapping[str, Mapping[str, Any]]
) -> bool:
    if required == "none":
        return True
    order = ("preflight", "power", "refinement")
    required_index = order.index(required)
    return all(_passed(gates[name]) for name in order[: required_index + 1])


def _finish(
    run_dir: Path,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
    gates: Mapping[str, Mapping[str, Any]],
) -> int:
    decision = decide_strang_refinement_workflow(
        provenance=provenance,
        preflight_gate=gates["preflight"],
        power_gate=gates["power"],
        refinement_gate=gates["refinement"],
    )
    required_pass = int(_required_gate_pass(args.require_gate, gates))
    workflow = {
        "schema": RUN_SCHEMA + "-workflow-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "components": {name: dict(gate) for name, gate in gates.items()},
        "required_gate": args.require_gate,
        "required_gate_pass": required_pass,
        "decision": decision,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "strang_workflow_gate.json", workflow)
    atomic_write_json(run_dir / "strang_refinement_decision.json", decision)
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    _write_status(
        run_dir,
        status="complete",
        outcome="complete" if required_pass else "gate_failed",
        phase=args.stage,
        required_gate=args.require_gate,
        required_gate_pass=required_pass,
        decision=decision["decision"],
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_record_count=len(registry["records"]),
        artifact_registry_size=registry_path.stat().st_size,
    )
    return 0 if required_pass else 2


def _load_frozen_source(run_dir: Path) -> dict[str, Any]:
    metadata = _load(run_dir / "source_image.json")
    npz_path = run_dir / "source_image.npz"
    if (
        metadata.get("npz_sha256") != file_fingerprint(npz_path)
        or metadata.get("npz_size") != npz_path.stat().st_size
    ):
        raise ArtifactCompatibilityError("frozen source-image payload changed")
    with np.load(npz_path, allow_pickle=False) as archive:
        image = np.ascontiguousarray(archive["image"], dtype=np.float64)
        mixed = np.ascontiguousarray(archive["mixed_target"], dtype=np.float64)
    if (
        _measure_digest(image) != metadata.get("image_sha256")
        or _measure_digest(mixed) != metadata.get("mixed_target_sha256")
    ):
        raise ArtifactCompatibilityError("frozen source-image digest changed")
    return {**metadata, "image": image, "mixed_target": mixed}


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    print(f"Jacobi Strang refinement run directory: {run_dir}")
    if resumed:
        _verify_terminal_registry(run_dir, stage=args.stage)
    else:
        args._active_run_dir = run_dir

    provenance = verify_exact_jacobi_rb_multipath_parent(
        args.parent_multipath_run_dir
    )
    config = _scientific_config(args)
    config_sha = config_fingerprint(config)
    source_hash, source_paths = _source_record(
        Path(args.parent_multipath_run_dir)
    )
    device = torch.device(args.device)
    backend = configure_exact_torch_backend(device)
    _freeze(
        run_dir / "scientific_config.json", config, require_existing=resumed
    )
    _freeze(
        run_dir / "parent_provenance.json",
        provenance,
        require_existing=resumed,
    )
    _freeze(
        run_dir / "run_manifest.json",
        {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "scientific_config_sha256": config_sha,
            "source_fingerprint": source_hash,
            "source_paths": source_paths,
            "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
            "parent_artifact_record_count": PARENT_REGISTRY_RECORD_COUNT,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "exact_backend": backend,
            **NO_WORK,
        },
        require_existing=resumed,
    )
    if resumed:
        source = (
            {}
            if args.stage == "report"
            else _load_frozen_source(run_dir)
        )
    else:
        source = _source_image(args)
        _atomic_write_npz(
            run_dir / "source_image.npz",
            image=source["image"],
            mixed_target=source["mixed_target"],
        )
        npz_path = run_dir / "source_image.npz"
        _freeze(
            run_dir / "source_image.json",
            {
                key: value
                for key, value in source.items()
                if key not in {"image", "mixed_target"}
            }
            | {
                "npz_sha256": file_fingerprint(npz_path),
                "npz_size": npz_path.stat().st_size,
                **NO_WORK,
            },
        )
    args._active_run_dir = run_dir
    _write_status(
        run_dir,
        status="running",
        phase=args.stage,
        required_gate=args.require_gate,
        scientific_config_sha256=config_sha,
    )
    gates = {
        "preflight": _existing_gate(
            run_dir, "preflight", "preflight not run"
        ),
        "power": _existing_gate(run_dir, "power", "power pilot not run"),
        "refinement": _existing_gate(
            run_dir, "refinement", "production refinement not run"
        ),
    }
    if args.stage in {"preflight", "all"}:
        try:
            gates["preflight"] = _run_preflight_stage(
                run_dir, args, provenance, source
            )
        except (RuntimeError, ValueError) as exc:
            gates["preflight"] = _failed_stage_gate(
                run_dir, "preflight", exc
            )
    if args.stage in {"power", "all"}:
        if not _passed(gates["preflight"]):
            gates["power"] = not_evaluated_gate(
                "power", "preflight gate failed"
            )
            atomic_write_json(run_dir / "strang_power_gate.json", gates["power"])
        else:
            try:
                gates["power"] = _run_power_stage(
                    run_dir, args, source
                )
            except (RuntimeError, ValueError) as exc:
                gates["power"] = _failed_stage_gate(run_dir, "power", exc)
    if args.stage in {"refinement", "all"}:
        if not _passed(gates["power"]):
            gates["refinement"] = not_evaluated_gate(
                "refinement", "power gate failed"
            )
            atomic_write_json(
                run_dir / "strang_refinement_gate.json",
                gates["refinement"],
            )
        else:
            try:
                gates["refinement"] = _run_refinement_stage(
                    run_dir, args, source
                )
            except (RuntimeError, ValueError) as exc:
                gates["refinement"] = _failed_stage_gate(
                    run_dir, "refinement", exc
                )
    return _finish(run_dir, args, provenance, gates)


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = parse_args(argv)
        return _run(args)
    except (ArtifactCompatibilityError, RuntimeError, ValueError) as exc:
        active = getattr(args, "_active_run_dir", None) if args is not None else None
        if active is not None and Path(active).is_dir():
            run_dir = Path(active).resolve()
            atomic_write_json(
                run_dir / "unexpected_failure.json",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **NO_WORK,
                },
            )
            registry = _artifact_registry(run_dir)
            atomic_write_json(run_dir / "artifact_registry.json", registry)
            registry_path = run_dir / "artifact_registry.json"
            _write_status(
                run_dir,
                status="complete",
                outcome="error",
                phase=args.stage,
                required_gate=args.require_gate,
                required_gate_pass=0,
                artifact_registry_sha256=file_fingerprint(registry_path),
                artifact_registry_record_count=len(registry["records"]),
                artifact_registry_size=registry_path.stat().st_size,
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
