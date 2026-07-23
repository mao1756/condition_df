"""Exact multi-path scheduling confirmation for the certified Jacobi RB API.

This workflow changes execution scheduling only.  Independent paths are
packed into ten-path and four-path CUDA calls while every path retains the
same seven serial phases, canonical random IDs, exact transition law, and
Rao--Blackwell target.  It performs no training or reverse sampling.
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
from mnist.d0_jacobi_rb_controls import (
    target_identity_controls,
    transition_law_controls,
)
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    _runtime_report as _cuda_runtime_report,
)
from mnist.d0_jacobi_rb_cuda_controls import (
    run_cuda_target_identity_controls,
    target_metrics_from_certificate_rows,
)
from mnist.d0_jacobi_rb_spectral import JacobiRBSpectralProfile
from mnist.d0_jacobi_rb_cuda_multipath import (
    EDGES_PER_PHASE,
    FROZEN_PROJECTION_GROUP_SIZES,
    FROZEN_PROJECTION_PATH_COUNT,
    FROZEN_VALIDATION_GROUP_SIZES,
    MULTIPATH_SCHEDULER_VERSION,
    SHARD_STEPS,
    canonical_same_phase_transition_ids,
    run_exact_multipath_shard,
)
from mnist.d0_jacobi_rb_cuda_multipath_gate import (
    JacobiRBMultipathThresholds,
    decide_multipath_workflow,
    evaluate_multipath_kernel,
    evaluate_multipath_pilot,
    evaluate_multipath_preflight,
    evaluate_multipath_target,
    evaluate_multipath_workflow,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_cuda_multipath_provenance import (
    PARENT_REGISTRY_SHA256,
    PARENT_RUN_BASENAME,
    verify_and_readjudicate_jacobi_rb_cuda_multipath_parent,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-cuda-multipath-confirmation"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "exact certified Jacobi RB multi-path execution scheduling only"
ROOT_SEED = 261_141
PARENT_ROOT_SEED = 261_131
PILOT_OUTER_STEPS = 64
PILOT_REPEATS = 3
FULL_OUTER_STEPS = 512
FULL_REPEATS = 3
TRANSITIONS_PER_PATH_STEP = 7 * EDGES_PER_PHASE
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


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
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write_state_npz(path: Path, final_states: np.ndarray) -> None:
    """Commit one packed scheduler boundary without another device transfer."""

    array = np.ascontiguousarray(np.asarray(final_states, dtype=np.float64))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, final_states=array)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_save_figure(path: Path, figure: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.png")
    figure.savefig(temporary, dpi=150, bbox_inches="tight")
    temporary.replace(path)


def _passed(gate: Mapping[str, Any]) -> bool:
    return (
        gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _progress(
    args: argparse.Namespace, label: str, done: int, total: int, started: float
) -> None:
    if args.no_progress or total <= 0:
        return
    elapsed = max(0.0, time.perf_counter() - started)
    eta = elapsed * max(0, total - done) / max(1, done)
    print(
        f"Jacobi RB multi-path {label} {done}/{total} "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "pilot", "kernel", "target", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-gate", choices=("none", "preflight", "pilot", "kernel", "target"),
        default="none",
    )
    parser.add_argument("--parent-cuda-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation"),
    )
    parser.add_argument("--run-name", default="production-exact-jacobi-rb-multipath")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    parser.add_argument("--pilot-outer-steps", type=int, default=PILOT_OUTER_STEPS)
    parser.add_argument("--pilot-repeats-per-group", type=int, default=PILOT_REPEATS)
    parser.add_argument("--full-outer-steps", type=int, default=FULL_OUTER_STEPS)
    parser.add_argument("--full-repeats-per-group", type=int, default=FULL_REPEATS)
    parser.add_argument("--steps-per-shard", type=int, default=SHARD_STEPS)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--test-only-reduced-workload", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.stage in {"pilot", "kernel", "target", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "pilot": {"none", "preflight", "pilot"},
        "kernel": {"none", "preflight", "pilot", "kernel"},
        "target": {"none", "preflight", "pilot", "kernel", "target"},
        "report": {"none", "preflight", "pilot", "kernel", "target"},
        "all": {"none", "preflight", "pilot", "kernel", "target"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(f"--require-gate {args.require_gate} is unavailable at stage {args.stage}")
    for name in (
        "pilot_outer_steps", "pilot_repeats_per_group", "full_outer_steps",
        "full_repeats_per_group", "steps_per_shard",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.pilot_outer_steps) % SHARD_STEPS or int(args.full_outer_steps) % SHARD_STEPS:
        parser.error("outer-step counts must be divisible by eight")
    frozen = {
        "root_seed": ROOT_SEED,
        "pilot_outer_steps": PILOT_OUTER_STEPS,
        "pilot_repeats_per_group": PILOT_REPEATS,
        "full_outer_steps": FULL_OUTER_STEPS,
        "full_repeats_per_group": FULL_REPEATS,
        "steps_per_shard": SHARD_STEPS,
    }
    changed = [name for name, value in frozen.items() if getattr(args, name) != value]
    if changed and not args.test_only_reduced_workload:
        parser.error(
            "production workload is frozen; overrides require "
            "--test-only-reduced-workload: " + ", ".join(changed)
        )
    if args.test_only_reduced_workload and args.require_gate != "none":
        parser.error("test-only workloads cannot satisfy a required gate")
    if torch.device(args.device).type != "cuda" and not args.test_only_reduced_workload:
        parser.error("production confirmation requires --device cuda")
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
        "schema": RUN_SCHEMA, "schema_version": RUN_SCHEMA_VERSION,
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
        raise ArtifactCompatibilityError(
            f"resume run lacks frozen artifact: {path.name}"
        )
    else:
        atomic_write_json(path, normalized)
    return normalized


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "claim_scope": CLAIM_SCOPE,
        "scheduler_version": MULTIPATH_SCHEDULER_VERSION,
        "root_seed": int(args.root_seed),
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "projection_path_count": FROZEN_PROJECTION_PATH_COUNT,
        "projection_group_sizes": list(FROZEN_PROJECTION_GROUP_SIZES),
        "validation_group_sizes": list(FROZEN_VALIDATION_GROUP_SIZES),
        "pilot_outer_steps": int(args.pilot_outer_steps),
        "pilot_repeats_per_group": int(args.pilot_repeats_per_group),
        "full_outer_steps": int(args.full_outer_steps),
        "full_repeats_per_group": int(args.full_repeats_per_group),
        "steps_per_shard": int(args.steps_per_shard),
        "test_only_reduced_workload": int(args.test_only_reduced_workload),
        **NO_WORK,
    }


def _source_record(parent_dir: Path) -> tuple[str, list[str]]:
    manifest = _load(parent_dir / "run_manifest.json")
    raw = manifest.get("source_paths")
    if not isinstance(raw, list) or len(raw) != 7:
        raise ArtifactCompatibilityError("parent manifest does not bind seven sources")
    paths = {Path(str(value)).resolve() for value in raw}
    for module_name in (
        "mnist.d0_jacobi_rb_cuda_multipath",
        "mnist.d0_jacobi_rb_cuda_multipath_gate",
        "mnist.d0_jacobi_rb_cuda_multipath_provenance",
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
            "sha256": file_fingerprint(path), "size": int(path.stat().st_size)
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry", "schema_version": 1,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "records": records, **NO_WORK,
    }


def _recoverable_shard_prefixes(stage: str) -> tuple[str, ...]:
    """Return only the shard families that the requested stage will replay.

    A later stage must never silently adopt corruption in already-authorizing
    evidence.  Pilot and kernel shards are recoverable only when that family
    is actually being rerun (or when ``all`` will rerun both families).
    """

    if stage == "pilot":
        return ("multipath_shards/pilot/",)
    if stage == "kernel":
        return ("multipath_shards/kernel/",)
    if stage == "all":
        return ("multipath_shards/pilot/", "multipath_shards/kernel/")
    return ()


def _verify_terminal_registry(run_dir: Path, *, stage: str) -> None:
    registry_path = run_dir / "artifact_registry.json"
    if not registry_path.is_file():
        return
    registry = _load(registry_path)
    status = _load(run_dir / "run_status.json")
    if status.get("artifact_registry_sha256") != file_fingerprint(registry_path):
        raise ArtifactCompatibilityError("resume status does not bind its registry")
    recoverable = _recoverable_shard_prefixes(stage)
    mutable = {
        "multipath_preflight_gate.json", "multipath_pilot_gate.json",
        "multipath_kernel_gate.json", "multipath_target_gate.json",
        "multipath_workflow_gate.json", "multipath_decision.json",
    }
    interrupted = status.get("status") == "running"
    records = dict(registry.get("records", {}))
    for relative, raw in records.items():
        path = run_dir / relative
        valid = (
            path.is_file() and isinstance(raw, Mapping)
            and raw.get("sha256") == file_fingerprint(path)
            and raw.get("size") == path.stat().st_size
        )
        if not valid and (
            relative.startswith(recoverable) or (interrupted and relative in mutable)
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
    invalid_unexpected = {
        relative
        for relative in unexpected
        if not interrupted
        or (
            relative.startswith("multipath_shards/")
            and not relative.startswith(recoverable)
        )
    }
    if invalid_unexpected:
        raise ArtifactCompatibilityError(
            "unregistered resume artifacts: "
            + ", ".join(sorted(invalid_unexpected))
        )


def _existing_gate(run_dir: Path, kind: str, reason: str) -> dict[str, Any]:
    path = run_dir / f"multipath_{kind}_gate.json"
    return _load(path) if path.is_file() else not_evaluated_gate(kind, reason)


def _save_gate(
    run_dir: Path, kind: str, metrics: Mapping[str, Any], gate: Mapping[str, Any]
) -> None:
    atomic_write_json(run_dir / f"{kind}_metrics.json", {
        "schema": RUN_SCHEMA + f"-{kind}-metrics", "schema_version": 1,
        "metrics": dict(metrics), **NO_WORK,
    })
    atomic_write_json(run_dir / f"multipath_{kind}_gate.json", dict(gate))


def _initial_states(path_count: int, seed: int) -> np.ndarray:
    return np.random.Generator(np.random.Philox(int(seed))).dirichlet(
        np.ones(28 * 28, dtype=np.float64), size=int(path_count)
    )


def _records_by_id(result: Any) -> dict[int, Any]:
    return {int(record.path_id): record for record in result.path_records}


def _result_counts(results: Sequence[Any]) -> dict[str, Any]:
    names = (
        "uncertified_count", "fallback_count", "resource_cap_count",
        "invalid_density_count", "approximation_count", "correction_count",
        "floor_count", "limiter_count", "renormalization_count", "nonfinite_count",
    )
    return {
        name: sum(int(result.diagnostics.get(name, 0)) for result in results)
        for name in names
    }


def _preflight(
    run_dir: Path, args: argparse.Namespace, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    device = torch.device(args.device)
    profile = JacobiRBCudaProfile()
    states_np = _initial_states(10, PARENT_ROOT_SEED + 700)
    # The immutable parent generated exactly one draw from this stream.  Its
    # first state is therefore states_np[0].
    states = torch.as_tensor(states_np, dtype=torch.float64, device=device).contiguous()
    path_ids = tuple(range(10))
    grouped = run_exact_multipath_shard(
        states.clone(), path_ids=path_ids, start_step=0,
        root_seed=PARENT_ROOT_SEED, profile=profile, group_sizes=(10,),
        capture_phase_state_trace=True,
    )
    serial = run_exact_multipath_shard(
        states.clone(), path_ids=path_ids, start_step=0,
        root_seed=PARENT_ROOT_SEED, profile=profile, group_sizes=(1,) * 10,
        capture_phase_state_trace=True,
    )
    split = run_exact_multipath_shard(
        states.clone(), path_ids=path_ids, start_step=0,
        root_seed=PARENT_ROOT_SEED, profile=profile, group_sizes=(4, 6),
        capture_phase_state_trace=True,
    )
    b4 = run_exact_multipath_shard(
        states[:4].clone(), path_ids=path_ids[:4], start_step=0,
        root_seed=PARENT_ROOT_SEED, profile=profile, group_sizes=(4,),
        capture_phase_state_trace=True,
    )
    permutation = tuple(reversed(range(10)))
    permuted = run_exact_multipath_shard(
        states[list(permutation)].contiguous(),
        path_ids=tuple(path_ids[index] for index in permutation), start_step=0,
        root_seed=PARENT_ROOT_SEED, profile=profile, group_sizes=(10,),
        capture_phase_state_trace=True,
    )
    grouped_records, serial_records, split_records = (
        _records_by_id(grouped), _records_by_id(serial), _records_by_id(split)
    )
    b4_records = _records_by_id(b4)
    permuted_records = _records_by_id(permuted)
    parity = all(
        grouped_records[path_id].output_sha256 == serial_records[path_id].output_sha256
        and grouped_records[path_id].final_state_sha256
        == serial_records[path_id].final_state_sha256
        and grouped_records[path_id].certificate_sha256
        == serial_records[path_id].certificate_sha256
        for path_id in path_ids
    ) and torch.equal(grouped.final_states, serial.final_states)
    split_parity = all(
        grouped_records[path_id].output_sha256 == split_records[path_id].output_sha256
        and grouped_records[path_id].final_state_sha256
        == split_records[path_id].final_state_sha256
        for path_id in path_ids
    ) and torch.equal(grouped.final_states, split.final_states)
    b4_parity = all(
        grouped_records[path_id].output_sha256 == b4_records[path_id].output_sha256
        and grouped_records[path_id].final_state_sha256
        == b4_records[path_id].final_state_sha256
        and grouped_records[path_id].certificate_sha256
        == b4_records[path_id].certificate_sha256
        for path_id in path_ids[:4]
    ) and torch.equal(grouped.final_states[:4], b4.final_states)
    permutation_parity = all(
        grouped_records[path_id].output_sha256
        == permuted_records[path_id].output_sha256
        and grouped_records[path_id].final_state_sha256
        == permuted_records[path_id].final_state_sha256
        and grouped_records[path_id].certificate_sha256
        == permuted_records[path_id].certificate_sha256
        for path_id in path_ids
    ) and (
        grouped.batch_output_sha256 == permuted.batch_output_sha256
        and grouped.batch_final_state_sha256 == permuted.batch_final_state_sha256
    )
    full_phase_trace_parity = (
        grouped.phase_state_records == serial.phase_state_records
        == split.phase_state_records == permuted.phase_state_records
        and len(grouped.phase_state_records) == SHARD_STEPS * 7
    )
    b4_phase_trace_parity = (
        len(b4.phase_state_records) == len(grouped.phase_state_records)
        and all(
            dict(b4_record.path_state_sha256_by_id)
            == {
                path_id: digest
                for path_id, digest in grouped_record.path_state_sha256_by_id
                if path_id in path_ids[:4]
            }
            for grouped_record, b4_record in zip(
                grouped.phase_state_records, b4.phase_state_records, strict=True
            )
        )
    )
    phase_trace_parity = full_phase_trace_parity and b4_phase_trace_parity

    parent_first = _load(
        Path(args.parent_cuda_run_dir)
        / "cuda_benchmark_shards/full-path/repeat-00-steps-000-007.json"
    )["row"]
    parent_replay = (
        grouped_records[0].output_sha256 == parent_first.get("output_sha256")
        and grouped_records[0].final_state_sha256 == parent_first.get("final_state_sha256")
    )
    ids = canonical_same_phase_transition_ids(
        tuple(range(FROZEN_PROJECTION_PATH_COUNT)), outer_step=0, phase=0,
        device=device,
    )
    reverse_ids = canonical_same_phase_transition_ids(
        tuple(reversed(range(FROZEN_PROJECTION_PATH_COUNT))),
        outer_step=0, phase=0, device=device,
    ).reshape(FROZEN_PROJECTION_PATH_COUNT, EDGES_PER_PHASE).flip(0).reshape(-1)
    id_unique = int(torch.unique(ids).numel()) == int(ids.numel())
    id_order = torch.equal(ids, reverse_ids)
    results = (grouped, serial, split, b4, permuted)
    counts = _result_counts(results)
    path_hash_mismatches = sum(
        int(grouped_records[path_id].output_sha256 != serial_records[path_id].output_sha256)
        for path_id in path_ids
    )
    state_hash_mismatches = sum(
        int(grouped_records[path_id].final_state_sha256 != serial_records[path_id].final_state_sha256)
        for path_id in path_ids
    )
    parent_manifest = _load(Path(args.parent_cuda_run_dir) / "run_manifest.json")
    parent_runtime = _load(
        Path(args.parent_cuda_run_dir) / "fused_cuda_runtime_report.json"
    )
    current_runtime = _cuda_runtime_report(device, probe_authorizer=True)
    atomic_write_json(run_dir / "multipath_cuda_runtime_report.json", {
        "schema": RUN_SCHEMA + "-cuda-runtime", "schema_version": 1,
        "parent": parent_runtime, "current": current_runtime, **NO_WORK,
    })
    runtime = configure_exact_torch_backend(device)
    runtime_match = (
        torch.__version__ == parent_manifest.get("torch")
        and str(device) == parent_manifest.get("device")
        and runtime.get("cuda_device_uuid")
        == dict(parent_manifest.get("exact_backend", {})).get("cuda_device_uuid")
        and runtime.get("cuda_compute_capability")
        == dict(parent_manifest.get("exact_backend", {})).get("cuda_compute_capability")
    )
    restart_probe_path = run_dir / "multipath_preflight_restart_probe.npz"
    _atomic_write_state_npz(restart_probe_path, grouped.committed_final_states)
    restart_row = {
        "predecessor_batch_output_sha256": grouped.batch_output_sha256,
        "predecessor_batch_final_state_sha256": grouped.batch_final_state_sha256,
        "predecessor_batch_certificate_sha256": grouped.batch_certificate_sha256,
        "state_npz_name": restart_probe_path.name,
        "state_npz_sha256": file_fingerprint(restart_probe_path),
        "state_npz_size": restart_probe_path.stat().st_size,
        "final_states_sha256": _array_sha256(grouped.committed_final_states),
    }
    restart_row["predecessor_chain_sha256"] = config_fingerprint(restart_row)
    restart_metadata_path = run_dir / "multipath_preflight_restart_probe.json"
    atomic_write_json(restart_metadata_path, {
        "schema": RUN_SCHEMA + "-restart-probe", "schema_version": 1,
        "row": restart_row, "row_sha256": config_fingerprint(restart_row),
        **NO_WORK,
    })
    saved_restart = _load(restart_metadata_path)
    saved_row = dict(saved_restart.get("row", {}))
    with np.load(restart_probe_path, allow_pickle=False) as archive:
        restart_probe = np.ascontiguousarray(archive["final_states"])
    expected_predecessor_chain = config_fingerprint({
        name: value for name, value in saved_row.items()
        if name != "predecessor_chain_sha256"
    })
    restart_metadata_pass = (
        saved_restart.get("row_sha256") == config_fingerprint(saved_row)
        and saved_row.get("state_npz_name") == restart_probe_path.name
        and saved_row.get("state_npz_sha256")
        == file_fingerprint(restart_probe_path)
        and saved_row.get("state_npz_size") == restart_probe_path.stat().st_size
        and saved_row.get("predecessor_chain_sha256")
        == expected_predecessor_chain
        and saved_row.get("predecessor_batch_output_sha256")
        == grouped.batch_output_sha256
        and saved_row.get("predecessor_batch_final_state_sha256")
        == grouped.batch_final_state_sha256
        and saved_row.get("predecessor_batch_certificate_sha256")
        == grouped.batch_certificate_sha256
    )
    restart_state_pass = (
        np.array_equal(restart_probe, grouped.committed_final_states)
        and _array_sha256(restart_probe)
        == _array_sha256(grouped.committed_final_states)
        == saved_row.get("final_states_sha256")
    )
    continued = run_exact_multipath_shard(
        grouped.final_states.clone(), path_ids=path_ids, start_step=8,
        root_seed=PARENT_ROOT_SEED, profile=profile, group_sizes=(10,),
        capture_phase_state_trace=True,
    )
    resumed = run_exact_multipath_shard(
        torch.as_tensor(
            restart_probe, dtype=torch.float64, device=device
        ).contiguous(),
        path_ids=path_ids, start_step=8, root_seed=PARENT_ROOT_SEED,
        profile=profile, group_sizes=(10,), capture_phase_state_trace=True,
    )
    continued_records = _records_by_id(continued)
    resumed_records = _records_by_id(resumed)
    restart_execution_pass = (
        restart_metadata_pass and restart_state_pass
        and continued.batch_output_sha256 == resumed.batch_output_sha256
        and continued.batch_final_state_sha256 == resumed.batch_final_state_sha256
        and continued.batch_certificate_sha256 == resumed.batch_certificate_sha256
        and continued.phase_state_records == resumed.phase_state_records
        and torch.equal(continued.final_states, resumed.final_states)
        and all(
            continued_records[path_id] == resumed_records[path_id]
            for path_id in path_ids
        )
    )
    results = (*results, continued, resumed)
    counts = _result_counts(results)
    atomic_write_json(
        run_dir / "multipath_preflight_restart_execution.json",
        {
            "schema": RUN_SCHEMA + "-restart-execution",
            "schema_version": 1,
            "metadata_validation_pass": int(restart_metadata_pass),
            "persisted_state_validation_pass": int(restart_state_pass),
            "continued": continued.to_record(),
            "resumed": resumed.to_record(),
            "passed": int(restart_execution_pass),
            **NO_WORK,
        },
    )
    metrics = {
        "control_provenance_pass": int(provenance.get("passed", 0)),
        "parent_record_count": provenance.get("parent_artifact_record_count"),
        "parent_certificate_pass": int(provenance.get("parent_certificate_valid", 0)),
        "parent_kernel_numerically_valid_pass": int(
            provenance.get("parent_kernel_numerically_valid", 0)
        ),
        "parent_single_path_resource_failure_pass": int(
            provenance.get("parent_kernel_resource_valid") == 0
        ),
        "parent_target_not_evaluated_pass": int(
            provenance.get("parent_target_evaluation_status") == "not_evaluated"
        ),
        "seven_parent_sources_immutable_pass": int(provenance.get("parent_mutated") == 0),
        "frozen_runtime_match_pass": int(runtime_match),
        "cuda_backend_replay_pass": int(
            sum(counts[name] for name in counts) == 0
            and all(
                int(result.diagnostics.get("certified_count", 0))
                == int(result.diagnostics.get("transition_count", -1))
                for result in results
            )
        ),
        "parent_cuda_source_hash_pass": int(
            current_runtime.get("source_sha256")
            == parent_runtime.get("source_sha256")
            and current_runtime.get("kernel_sha256")
            == parent_runtime.get("kernel_sha256")
        ),
        "parent_cubin_hash_pass": int(
            current_runtime.get("cubin_sha256")
            == parent_runtime.get("cubin_sha256")
            and current_runtime.get("binary_sha256")
            == parent_runtime.get("binary_sha256")
        ),
        "parent_compile_options_hash_pass": int(
            current_runtime.get("compile_options_sha256")
            == parent_runtime.get("compile_options_sha256")
        ),
        "canonical_id_uniqueness_pass": int(id_unique),
        "canonical_id_group_order_invariance_pass": int(id_order),
        "path_zero_parent_replay_pass": int(parent_replay),
        "serial_batch_parity_pass": int(parity),
        "phase_order_pass": int(
            all(int(result.diagnostics.get("phase_count", 0)) == 56 for result in results)
            and all(
                [(record.outer_step, record.phase) for record in result.phase_state_records]
                == [
                    (start_step + local_step, phase)
                    for local_step in range(SHARD_STEPS)
                    for phase in range(7)
                ]
                for result, start_step in (
                    (grouped, 0), (serial, 0), (split, 0), (b4, 0),
                    (permuted, 0), (continued, 8), (resumed, 8),
                )
            )
        ),
        "phase_by_phase_equivalence_pass": int(phase_trace_parity),
        "group_order_invariance_pass": int(split_parity),
        "fresh_b4_parity_pass": int(b4_parity),
        "path_permutation_invariance_pass": int(permutation_parity),
        "canonical_full_id_field_proof_pass": int(
            FROZEN_PROJECTION_PATH_COUNT < (1 << 20)
            and FULL_OUTER_STEPS < (1 << 10)
            and 7 < (1 << 3)
            and EDGES_PER_PHASE < (1 << 10)
        ),
        "canonical_full_id_plan_count": (
            FROZEN_PROJECTION_PATH_COUNT
            * FULL_OUTER_STEPS * 7 * EDGES_PER_PHASE
        ),
        "no_cross_path_write_pass": int(
            parity and split_parity and b4_parity and permutation_parity
        ),
        "resume_replay_pass": int(parity and restart_execution_pass),
        "state_updates_device_resident_pass": int(
            all(int(result.diagnostics.get("state_updates_device_resident", 0)) == 1 for result in results)
        ),
        "evolving_state_host_roundtrip_pass": int(
            all(
                int(result.diagnostics.get("evolving_state_host_roundtrip_count", 1))
                == 0
                for result in results
            )
        ),
        "path_count": FROZEN_PROJECTION_PATH_COUNT,
        "projection_group_sizes": list(FROZEN_PROJECTION_GROUP_SIZES),
        "validation_group_sizes": list(FROZEN_VALIDATION_GROUP_SIZES),
        "restart_steps_per_shard": SHARD_STEPS,
        "maximum_cuda_launch_lanes": max(
            int(result.diagnostics.get("maximum_cuda_launch_lanes", 0)) for result in results
        ),
        "mass_error": max(
            float(result.diagnostics.get("maximum_mass_error", np.inf)) for result in results
        ),
        "transition_id_collision_count": int(ids.numel() - torch.unique(ids).numel()),
        "path_hash_mismatch_count": path_hash_mismatches,
        "state_hash_mismatch_count": state_hash_mismatches,
        **counts,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "multipath_equivalence_report.json", {
        "schema": RUN_SCHEMA + "-equivalence", "schema_version": 1,
        "grouped": grouped.to_record(), "serial": serial.to_record(),
        "split": split.to_record(), "b4": b4.to_record(),
        "permuted": permuted.to_record(), "metrics": metrics, **NO_WORK,
    })
    atomic_write_csv(
        run_dir / "multipath_equivalence_paths.csv",
        [
            {
                "path_id": path_id,
                "grouped_output_sha256": grouped_records[path_id].output_sha256,
                "serial_output_sha256": serial_records[path_id].output_sha256,
                "split_output_sha256": split_records[path_id].output_sha256,
                "output_match": int(
                    grouped_records[path_id].output_sha256
                    == serial_records[path_id].output_sha256
                    == split_records[path_id].output_sha256
                ),
                "final_state_match": int(
                    grouped_records[path_id].final_state_sha256
                    == serial_records[path_id].final_state_sha256
                    == split_records[path_id].final_state_sha256
                ),
            }
            for path_id in path_ids
        ],
    )
    gate = evaluate_multipath_preflight(metrics)
    _save_gate(run_dir, "preflight", metrics, gate)
    return gate


def _shard_fingerprint(
    *, family: str, group_size: int, repeat: int, args: argparse.Namespace
) -> str:
    return config_fingerprint({
        "schema": RUN_SCHEMA + "-shard-input", "schema_version": 1,
        "family": family, "group_size": group_size, "repeat": repeat,
        "root_seed": int(args.root_seed), "steps_per_shard": SHARD_STEPS,
        "scheduler_version": MULTIPATH_SCHEDULER_VERSION,
        "profile": repr(JacobiRBCudaProfile()),
    })


def _load_shard(
    path: Path, fingerprint: str, input_hash: str, previous_chain: str
) -> tuple[dict[str, Any], np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        payload = _load(path)
        row = payload.get("row")
        if not isinstance(row, Mapping):
            return None
        row = dict(row)
        state_path = path.with_suffix(".npz")
        if (
            payload.get("input_fingerprint") != fingerprint
            or payload.get("row_sha256") != config_fingerprint(row)
            or row.get("shard_input_fingerprint") != fingerprint
            or row.get("input_states_sha256") != input_hash
            or row.get("previous_shard_sha256") != previous_chain
            or row.get("state_npz_name") != state_path.name
            or not state_path.is_file()
            or row.get("state_npz_sha256") != file_fingerprint(state_path)
            or row.get("state_npz_size") != state_path.stat().st_size
            or int(row.get("commit_reuses_packed_host_snapshot", 0)) != 1
            or int(row.get("state_npz_and_metadata_commit_included", 0)) != 1
            or int(row.get("conservative_timing_bound_pass", 0)) != 1
            or not np.isfinite(float(row.get("wall_elapsed_seconds", np.nan)))
            or float(row.get("wall_elapsed_seconds", 0.0)) <= 0.0
            or not np.isfinite(float(row.get("transitions_per_second", np.nan)))
            or float(row.get("transitions_per_second", 0.0)) <= 0.0
            or not 1 <= int(row.get("timing_finalization_attempt", 0)) <= 8
            or not np.isfinite(float(row.get("timing_allowance_seconds", np.nan)))
            or float(row.get("timing_allowance_seconds", 0.0)) < 0.1
        ):
            return None
        expected_rate = int(row.get("diagnostics", {}).get("transition_count", 0)) / float(
            row["wall_elapsed_seconds"]
        )
        if not math.isclose(
            float(row["transitions_per_second"]), expected_rate,
            rel_tol=1.0e-15, abs_tol=0.0,
        ):
            return None
        with np.load(state_path, allow_pickle=False) as archive:
            if set(archive.files) != {"final_states"}:
                return None
            final = np.ascontiguousarray(
                np.asarray(archive["final_states"], dtype=np.float64)
            )
        if (
            final.shape != (int(row.get("group_size", -1)), 28 * 28)
            or _array_sha256(final) != row.get("persisted_final_states_sha256")
        ):
            return None
        expected_chain = config_fingerprint({
            "input_states_sha256": input_hash,
            "previous_shard_sha256": previous_chain,
            "batch_output_sha256": row.get("batch_output_sha256"),
            "batch_final_state_sha256": row.get("batch_final_state_sha256"),
            "batch_certificate_sha256": row.get("batch_certificate_sha256"),
            "state_npz_sha256": row.get("state_npz_sha256"),
            "state_npz_size": row.get("state_npz_size"),
        })
        if row.get("chain_sha256") != expected_chain:
            return None
        return row, final
    except (
        ArtifactCompatibilityError, EOFError, OSError, TypeError, ValueError,
        zipfile.BadZipFile,
    ):
        return None


def _validate_completed_shard_family(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    family: str,
    outer_steps: int,
    repeats: int,
) -> None:
    """Verify every committed NPZ/JSON link in a prerequisite family.

    This is deliberately read-only.  A later stage may consume a passing gate
    only after the complete persisted chain behind that gate has replayed its
    fingerprints, predecessor links, state hashes, and frozen path plan.
    """

    root = run_dir / "multipath_shards" / family
    if not root.is_dir():
        raise ArtifactCompatibilityError(
            f"passing {family} gate lacks its persisted shard family"
        )
    expected_files: set[str] = set()
    for group_size in FROZEN_VALIDATION_GROUP_SIZES:
        namespace = 10_000 if family == "pilot" else 20_000
        group_offset = 0 if group_size == 10 else 100
        expected_path_ids = {
            namespace + group_offset + index for index in range(group_size)
        }
        initial = _initial_states(
            group_size, int(args.root_seed) + namespace + group_size
        )
        for repeat in range(int(repeats)):
            fingerprint = _shard_fingerprint(
                family=family,
                group_size=group_size,
                repeat=repeat,
                args=args,
            )
            input_hash = _array_sha256(initial)
            previous_chain = config_fingerprint(
                {
                    "kind": "multipath-genesis",
                    "fingerprint": fingerprint,
                    "initial_states_sha256": input_hash,
                }
            )
            for start_step in range(0, int(outer_steps), SHARD_STEPS):
                name = (
                    f"b{group_size:02d}-repeat-{repeat:02d}-"
                    f"steps-{start_step:03d}-{start_step + SHARD_STEPS - 1:03d}"
                )
                path = root / f"{name}.json"
                expected_files.update({f"{name}.json", f"{name}.npz"})
                loaded = _load_shard(
                    path, fingerprint, input_hash, previous_chain
                )
                if loaded is None:
                    raise ArtifactCompatibilityError(
                        f"completed {family} shard chain is invalid at {path.name}"
                    )
                row, final = loaded
                diagnostics = row.get("diagnostics")
                path_records = row.get("path_records")
                actual_path_ids = (
                    {
                        int(record["path_id"])
                        for record in path_records
                        if isinstance(record, Mapping) and "path_id" in record
                    }
                    if isinstance(path_records, list)
                    else set()
                )
                expected_transitions = (
                    group_size * SHARD_STEPS * TRANSITIONS_PER_PATH_STEP
                )
                if (
                    row.get("family") != family
                    or int(row.get("group_size", -1)) != group_size
                    or int(row.get("repeat", -1)) != repeat
                    or int(row.get("start_step", -1)) != start_step
                    or final.shape != (group_size, 28 * 28)
                    or not isinstance(diagnostics, Mapping)
                    or int(diagnostics.get("start_step", -1)) != start_step
                    or int(diagnostics.get("step_count", -1)) != SHARD_STEPS
                    or int(diagnostics.get("transition_count", -1))
                    != expected_transitions
                    or diagnostics.get("group_sizes") != [group_size]
                    or int(diagnostics.get("maximum_backend_call_size", -1))
                    != group_size * EDGES_PER_PHASE
                    or not isinstance(path_records, list)
                    or len(path_records) != group_size
                    or actual_path_ids != expected_path_ids
                ):
                    raise ArtifactCompatibilityError(
                        f"completed {family} shard semantics changed at {path.name}"
                    )
                input_hash = str(row["persisted_final_states_sha256"])
                previous_chain = str(row["chain_sha256"])
    actual_files = {
        path.name for path in root.iterdir() if path.is_file()
    }
    if actual_files != expected_files:
        raise ArtifactCompatibilityError(
            f"completed {family} shard file set changed"
        )


def _validate_prerequisite_shard_families(
    run_dir: Path, args: argparse.Namespace
) -> None:
    """Fail closed on corrupt prior families before mutating a resumed run."""

    recoverable = set(_recoverable_shard_prefixes(args.stage))
    plans = {
        "pilot": (
            int(args.pilot_outer_steps),
            int(args.pilot_repeats_per_group),
        ),
        "kernel": (
            int(args.full_outer_steps),
            int(args.full_repeats_per_group),
        ),
    }
    for family, (outer_steps, repeats) in plans.items():
        if f"multipath_shards/{family}/" in recoverable:
            continue
        gate = _existing_gate(run_dir, family, f"{family} not run")
        if _passed(gate):
            _validate_completed_shard_family(
                run_dir,
                args,
                family=family,
                outer_steps=outer_steps,
                repeats=repeats,
            )


def _run_performance_family(
    run_dir: Path, args: argparse.Namespace, *, family: str,
    outer_steps: int, repeats: int,
) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    profile = JacobiRBCudaProfile()
    root = run_dir / "multipath_shards" / family
    root.mkdir(parents=True, exist_ok=True)
    total = len(FROZEN_VALIDATION_GROUP_SIZES) * int(repeats) * (int(outer_steps) // SHARD_STEPS)
    done = 0
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for group_size in FROZEN_VALIDATION_GROUP_SIZES:
        namespace = 10_000 if family == "pilot" else 20_000
        group_offset = 0 if group_size == 10 else 100
        path_ids = tuple(
            namespace + group_offset + index for index in range(group_size)
        )
        initial = _initial_states(
            group_size, int(args.root_seed) + namespace + group_size
        )
        for repeat in range(int(repeats)):
            states = torch.as_tensor(
                initial, dtype=torch.float64, device=device
            ).contiguous()
            committed_input = np.ascontiguousarray(initial)
            fingerprint = _shard_fingerprint(
                family=family, group_size=group_size, repeat=repeat, args=args
            )
            previous_chain = config_fingerprint({
                "kind": "multipath-genesis", "fingerprint": fingerprint,
                "initial_states_sha256": _array_sha256(initial),
            })
            reuse_tail = True
            for start_step in range(0, int(outer_steps), SHARD_STEPS):
                input_hash = _array_sha256(committed_input)
                path = root / (
                    f"b{group_size:02d}-repeat-{repeat:02d}-"
                    f"steps-{start_step:03d}-{start_step + SHARD_STEPS - 1:03d}.json"
                )
                loaded = (
                    _load_shard(path, fingerprint, input_hash, previous_chain)
                    if reuse_tail else None
                )
                if loaded is None:
                    reuse_tail = False
                    wall_started = time.perf_counter()
                    result = run_exact_multipath_shard(
                        states, path_ids=path_ids, start_step=start_step,
                        root_seed=int(args.root_seed), profile=profile,
                        group_sizes=(group_size,),
                    )
                    states = result.final_states
                    final_np = result.committed_final_states
                    state_path = path.with_suffix(".npz")
                    _atomic_write_state_npz(state_path, final_np)
                    row = {
                        "family": family, "group_size": group_size,
                        "repeat": repeat, "start_step": start_step,
                        **result.to_record(),
                        "input_states_sha256": input_hash,
                        "shard_input_fingerprint": fingerprint,
                        "persisted_final_states_sha256": _array_sha256(final_np),
                        "previous_shard_sha256": previous_chain,
                        "state_npz_name": state_path.name,
                        "state_npz_sha256": file_fingerprint(state_path),
                        "state_npz_size": state_path.stat().st_size,
                        "commit_reuses_packed_host_snapshot": 1,
                        **NO_WORK,
                    }
                    row["chain_sha256"] = config_fingerprint({
                        "input_states_sha256": input_hash,
                        "previous_shard_sha256": previous_chain,
                        "batch_output_sha256": row["batch_output_sha256"],
                        "batch_final_state_sha256": row["batch_final_state_sha256"],
                        "batch_certificate_sha256": row["batch_certificate_sha256"],
                        "state_npz_sha256": row["state_npz_sha256"],
                        "state_npz_size": row["state_npz_size"],
                    })
                    payload = {
                        "schema": RUN_SCHEMA + "-performance-shard", "schema_version": 1,
                        "input_fingerprint": fingerprint, "row": row,
                        "row_sha256": config_fingerprint(row), **NO_WORK,
                    }
                    # Precommit an explicitly non-final row to measure this
                    # metadata size and filesystem.  The sole shard JSON is
                    # then replaced with a conservative timing certificate.
                    precommit_started = time.perf_counter()
                    atomic_write_json(path, payload)
                    precommit_json_duration = (
                        time.perf_counter() - precommit_started
                    )
                    allowance = max(0.1, 10.0 * precommit_json_duration)
                    for timing_attempt in range(1, 9):
                        before_commit = time.perf_counter()
                        conservative_wall = (
                            before_commit - wall_started + allowance
                        )
                        finalized = {
                            **row,
                            "wall_elapsed_seconds": conservative_wall,
                            "transitions_per_second": (
                                int(row["diagnostics"]["transition_count"])
                                / conservative_wall
                            ),
                            "state_npz_and_metadata_commit_included": 1,
                            "conservative_timing_bound_pass": 1,
                            "timing_finalization_attempt": timing_attempt,
                            "timing_allowance_seconds": allowance,
                        }
                        atomic_write_json(path, {
                            **payload, "row": finalized,
                            "row_sha256": config_fingerprint(finalized),
                        })
                        actual_elapsed = time.perf_counter() - wall_started
                        if actual_elapsed <= conservative_wall:
                            row = finalized
                            break
                        # Fail closed while increasing the allowance; an
                        # interrupted retry cannot be mistaken for final.
                        atomic_write_json(path, payload)
                        allowance = max(allowance * 2.0, 4.0 * actual_elapsed)
                    else:
                        raise RuntimeError(
                            "could not conservatively bound final shard metadata commit"
                        )
                else:
                    row, final_np = loaded
                    states = torch.as_tensor(
                        final_np, dtype=torch.float64, device=device
                    ).contiguous()
                committed_input = final_np
                previous_chain = str(row["chain_sha256"])
                summary = dict(row)
                rows.append(summary)
                done += 1
                if done == total or done % 8 == 0:
                    _progress(args, family, done, total, started)
    return rows


def _performance_metrics(
    rows: Sequence[Mapping[str, Any]], *, outer_steps: int, repeats: int,
    pilot: bool, parent_dir: Path,
) -> dict[str, Any]:
    expected: dict[int, int] = {
        size: size * int(outer_steps) * TRANSITIONS_PER_PATH_STEP
        for size in FROZEN_VALIDATION_GROUP_SIZES
    }
    per_repeat_seconds: dict[tuple[int, int], float] = {}
    per_repeat_transitions: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (int(row["group_size"]), int(row["repeat"]))
        per_repeat_seconds[key] = per_repeat_seconds.get(key, 0.0) + float(
            row.get("wall_elapsed_seconds", row["diagnostics"].get("elapsed_seconds", 0.0))
        )
        per_repeat_transitions[key] = per_repeat_transitions.get(key, 0) + int(
            row["diagnostics"]["transition_count"]
        )
    complete = all(
        per_repeat_transitions.get((size, repeat), 0) == expected[size]
        for size in FROZEN_VALIDATION_GROUP_SIZES for repeat in range(int(repeats))
    )
    rates: dict[int, list[float]] = {10: [], 4: []}
    walls: dict[int, list[float]] = {10: [], 4: []}
    for size in FROZEN_VALIDATION_GROUP_SIZES:
        for repeat in range(int(repeats)):
            seconds = per_repeat_seconds.get((size, repeat), 0.0)
            walls[size].append(seconds)
            rates[size].append(expected[size] / seconds if seconds > 0 else 0.0)
    scale = FULL_OUTER_STEPS / int(outer_steps) if pilot else 1.0
    projected_seconds = scale * (6.0 * max(walls[10]) + max(walls[4]))
    projected_count = 89_915_392
    diagnostics = [dict(row["diagnostics"]) for row in rows]
    transition_count = sum(per_repeat_transitions.values())
    certified_count = sum(
        int(value.get("certified_count", 0)) for value in diagnostics
    )
    count_names = (
        "uncertified_count", "fallback_count", "resource_cap_count",
        "invalid_density_count", "approximation_count", "correction_count",
        "floor_count", "limiter_count", "renormalization_count", "nonfinite_count",
    )
    counts = {
        name: sum(int(value.get(name, 0)) for value in diagnostics)
        for name in count_names
    }
    fallback_seconds = sum(float(value.get("fallback_elapsed_seconds", 0.0)) for value in diagnostics)
    wall_seconds = sum(per_repeat_seconds.values())
    output_replay = True
    final_replay = True
    certificate_replay = True
    chain_pass = True
    for size in FROZEN_VALIDATION_GROUP_SIZES:
        reference = sorted([
            row for row in rows
            if int(row["group_size"]) == size and int(row["repeat"]) == 0
        ], key=lambda row: int(row["start_step"]))
        for repeat in range(1, int(repeats)):
            comparison = sorted([
                row for row in rows
                if int(row["group_size"]) == size and int(row["repeat"]) == repeat
            ], key=lambda row: int(row["start_step"]))
            output_replay &= [r["batch_output_sha256"] for r in reference] == [r["batch_output_sha256"] for r in comparison]
            final_replay &= [r["batch_final_state_sha256"] for r in reference] == [r["batch_final_state_sha256"] for r in comparison]
            certificate_replay &= [r["batch_certificate_sha256"] for r in reference] == [r["batch_certificate_sha256"] for r in comparison]
    chain_pass = bool(rows)
    for size in FROZEN_VALIDATION_GROUP_SIZES:
        for repeat in range(int(repeats)):
            sequence = sorted(
                (
                    row for row in rows
                    if int(row["group_size"]) == size
                    and int(row["repeat"]) == repeat
                ),
                key=lambda row: int(row["start_step"]),
            )
            expected_starts = list(range(0, int(outer_steps), SHARD_STEPS))
            chain_pass &= [int(row["start_step"]) for row in sequence] == expected_starts
            if not sequence:
                continue
            first = sequence[0]
            genesis = config_fingerprint({
                "kind": "multipath-genesis",
                "fingerprint": first.get("shard_input_fingerprint"),
                "initial_states_sha256": first.get("input_states_sha256"),
            })
            chain_pass &= first.get("previous_shard_sha256") == genesis
            for index, row in enumerate(sequence):
                recomputed = config_fingerprint({
                    "input_states_sha256": row.get("input_states_sha256"),
                    "previous_shard_sha256": row.get("previous_shard_sha256"),
                    "batch_output_sha256": row.get("batch_output_sha256"),
                    "batch_final_state_sha256": row.get("batch_final_state_sha256"),
                    "batch_certificate_sha256": row.get("batch_certificate_sha256"),
                    "state_npz_sha256": row.get("state_npz_sha256"),
                    "state_npz_size": row.get("state_npz_size"),
                })
                chain_pass &= row.get("chain_sha256") == recomputed
                if index:
                    predecessor = sequence[index - 1]
                    chain_pass &= (
                        row.get("previous_shard_sha256")
                        == predecessor.get("chain_sha256")
                        and row.get("input_states_sha256")
                        == predecessor.get("persisted_final_states_sha256")
                    )
    parent_kernel = _load(parent_dir / "kernel_metrics.json")["metrics"]
    ids_by_group: dict[int, set[int]] = {}
    group_ids_stable = True
    for size in FROZEN_VALIDATION_GROUP_SIZES:
        selected = [row for row in rows if int(row["group_size"]) == size]
        reference_ids = (
            {int(value["path_id"]) for value in selected[0].get("path_records", ())}
            if selected else set()
        )
        ids_by_group[size] = reference_ids
        group_ids_stable &= bool(selected) and all(
            {int(value["path_id"]) for value in row.get("path_records", ())}
            == reference_ids
            for row in selected
        )
    metrics = {
        "all_groups_completed_pass": int(complete),
        "all_certificates_pass": int(
            complete
            and counts["uncertified_count"] == 0
            and certified_count == transition_count
        ),
        "output_hash_replay_pass": int(output_replay),
        "final_state_hash_replay_pass": int(final_replay),
        "certificate_hash_replay_pass": int(certificate_replay),
        "restart_shard_chain_pass": int(chain_pass),
        "state_updates_device_resident_pass": int(
            bool(rows) and all(int(value.get("state_updates_device_resident", 0)) == 1 for value in diagnostics)
        ),
        "evolving_state_host_roundtrip_pass": int(
            bool(rows)
            and all(
                int(value.get("evolving_state_host_roundtrip_count", 1)) == 0
                for value in diagnostics
            )
        ),
        "path_isolation_pass": int(
            bool(rows) and all(
                len(row.get("path_records", ())) == int(row["group_size"])
                and len({int(value["path_id"]) for value in row.get("path_records", ())})
                == int(row["group_size"])
                for row in rows
            )
        ),
        "group_path_id_disjoint_pass": int(
            group_ids_stable
            and len(ids_by_group[10]) == 10
            and len(ids_by_group[4]) == 4
            and ids_by_group[10].isdisjoint(ids_by_group[4])
        ),
        "group_schedule_pass": int(
            bool(rows) and all(row["diagnostics"].get("group_sizes") == [int(row["group_size"])] for row in rows)
        ),
        "commit_reuses_packed_host_snapshot_pass": int(
            bool(rows)
            and all(int(row.get("commit_reuses_packed_host_snapshot", 0)) == 1 for row in rows)
            and all(int(row.get("state_npz_and_metadata_commit_included", 0)) == 1 for row in rows)
            and all(int(row.get("conservative_timing_bound_pass", 0)) == 1 for row in rows)
        ),
        "group_sizes": list(FROZEN_VALIDATION_GROUP_SIZES),
        "outer_steps": int(outer_steps),
        "repeats_per_group": int(repeats),
        "restart_steps_per_shard": SHARD_STEPS,
        "maximum_cuda_launch_lanes": max(
            [int(value.get("maximum_cuda_launch_lanes", 0)) for value in diagnostics] or [0]
        ),
        "mass_error": max(
            [float(value.get("maximum_mass_error", np.inf)) for value in diagnostics] or [np.inf]
        ),
        "cuda_kernel_max_error": float(parent_kernel.get("cuda_kernel_max_error", np.inf)),
        "fallback_fraction": counts["fallback_count"] / max(1, sum(per_repeat_transitions.values())),
        "fallback_cost_fraction": fallback_seconds / wall_seconds if wall_seconds > 0 else float(counts["fallback_count"] > 0),
        "peak_memory_fraction": (
            float(torch.cuda.max_memory_allocated())
            / float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
            if torch.cuda.is_available() else 0.0
        ),
        "b10_slowest_transitions_per_second": min(rates[10]) if complete else 0.0,
        "b4_slowest_transitions_per_second": min(rates[4]) if complete else 0.0,
        "b10_repeat_wall_seconds": walls[10],
        "b4_repeat_wall_seconds": walls[4],
        "projected_cache_seconds": projected_seconds,
        "projected_cache_hours": projected_seconds / 3600.0,
        "projected_effective_transitions_per_second": projected_count / max(projected_seconds, np.finfo(np.float64).tiny),
        "projected_transition_count": projected_count,
        "certificate_fraction": certified_count / max(1, transition_count),
        "completed_shard_count": len(rows),
        "replay_bit_mismatch_count": int(not (output_replay and final_replay and certificate_replay)),
        **counts,
        **NO_WORK,
    }
    if not pilot:
        metrics.update({
            **{
                name: int(parent_kernel.get(name, 0))
                for name in (
                    "production_support_pass",
                    "cdf_endpoint_certificate_pass",
                    "cdf_monotonicity_pass",
                    "normalization_pass",
                    "semigroup_pass",
                    "detailed_balance_pass",
                    "law_control_pass",
                    "precision_doubling_hash_pass",
                )
            },
            "cuda_pair_mass_error": float(
                parent_kernel.get("cuda_pair_mass_error", np.inf)
            ),
            "cuda_simplex_error": float(
                parent_kernel.get("cuda_simplex_error", np.inf)
            ),
            "b10_transitions_per_repeat": expected[10],
            "b4_transitions_per_repeat": expected[4],
            "total_full_benchmark_transitions": sum(per_repeat_transitions.values()),
        })
    return metrics


def _plot_performance(
    run_dir: Path, family: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for group_size, color in ((10, "tab:blue"), (4, "tab:orange")):
        selected = [row for row in rows if int(row["group_size"]) == group_size]
        rates = [float(row.get("transitions_per_second", 0.0)) for row in selected]
        axes[0].plot(
            np.arange(len(rates)), rates, marker=".", linewidth=0.8,
            label=f"B={group_size}", color=color,
        )
        repeat_walls: dict[int, float] = {}
        for row in selected:
            repeat = int(row["repeat"])
            repeat_walls[repeat] = repeat_walls.get(repeat, 0.0) + float(
                row.get("wall_elapsed_seconds", 0.0)
            )
        axes[1].plot(
            sorted(repeat_walls),
            [repeat_walls[index] for index in sorted(repeat_walls)],
            marker="o", label=f"B={group_size}", color=color,
        )
    axes[0].axhline(1_300.0, color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("restart shard")
    axes[0].set_ylabel("transitions / second")
    axes[0].set_title("Exact shard throughput")
    axes[0].legend()
    axes[1].set_xlabel("repeat")
    axes[1].set_ylabel("wall seconds")
    axes[1].set_title("Complete group repeat")
    axes[1].legend()
    figure.tight_layout()
    _atomic_save_figure(
        run_dir / f"multipath_{family}_throughput.png", figure
    )
    plt.close(figure)


def _performance_stage(
    run_dir: Path, args: argparse.Namespace, *, pilot: bool
) -> dict[str, Any]:
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    family = "pilot" if pilot else "kernel"
    outer_steps = int(args.pilot_outer_steps if pilot else args.full_outer_steps)
    repeats = int(args.pilot_repeats_per_group if pilot else args.full_repeats_per_group)
    rows = _run_performance_family(
        run_dir, args, family=family, outer_steps=outer_steps, repeats=repeats
    )
    metrics = _performance_metrics(
        rows, outer_steps=outer_steps, repeats=repeats, pilot=pilot,
        parent_dir=Path(args.parent_cuda_run_dir),
    )
    atomic_write_json(run_dir / f"multipath_{family}_benchmark.json", {
        "schema": RUN_SCHEMA + f"-{family}-benchmark", "schema_version": 1,
        "metrics": metrics, "rows": rows, **NO_WORK,
    })
    atomic_write_csv(
        run_dir / f"multipath_{family}_shards.csv",
        [
            {
                "family": row["family"], "group_size": row["group_size"],
                "repeat": row["repeat"], "start_step": row["start_step"],
                "transition_count": row["diagnostics"]["transition_count"],
                "wall_elapsed_seconds": row.get("wall_elapsed_seconds"),
                "transitions_per_second": row.get("transitions_per_second"),
                "fallback_count": row["diagnostics"].get("fallback_count"),
                "maximum_cuda_launch_lanes": row["diagnostics"].get("maximum_cuda_launch_lanes"),
                "batch_output_sha256": row["batch_output_sha256"],
                "batch_final_state_sha256": row["batch_final_state_sha256"],
                "chain_sha256": row["chain_sha256"],
            }
            for row in rows
        ],
    )
    _plot_performance(run_dir, family, rows)
    gate = evaluate_multipath_pilot(metrics) if pilot else evaluate_multipath_kernel(metrics)
    _save_gate(run_dir, family, metrics, gate)
    return gate


def _target(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    panel = _load(Path(args.parent_cuda_run_dir) / "jacobi_rb_cuda_certificate_panel.json")
    raw = panel.get("rows")
    if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
        raise ArtifactCompatibilityError("parent certificate panel is invalid")
    rows = [dict(row) for row in raw]
    metrics = target_metrics_from_certificate_rows(rows)
    parent = [row for row in rows if row.get("panel") == "parent_replay"]
    cuda_rows, cuda_metrics = run_cuda_target_identity_controls(
        device=torch.device(args.device), profile=JacobiRBCudaProfile(),
        count=512, root_seed=int(args.root_seed) + 100,
    )
    spectral = JacobiRBSpectralProfile(require_correct_rounding=True)
    identity = target_identity_controls(
        count=512, root_seed=int(args.root_seed) + 200, profile=spectral
    )
    law = transition_law_controls(
        count=512, root_seed=int(args.root_seed) + 300, profile=spectral
    )
    target_mismatch = sum(
        1 - int(row.get("parent_target_bit_match", 0)) for row in parent
    )
    negative_pass = all(
        int(identity.metrics.get(name, 0)) == 1
        for name in (
            "orientation_negative_fixture_pass", "h_scaling_negative_fixture_pass",
            "invariant_beta_score_negative_fixture_pass", "pair_mass_negative_fixture_pass",
        )
    )
    preflight_metrics = _load(run_dir / "preflight_metrics.json")["metrics"]
    metrics.update(cuda_metrics)
    metrics.update({
        "rao_blackwell_identity_pass": int(
            bool(parent) and target_mismatch == 0
            and int(cuda_metrics.get("rao_blackwell_identity_pass", 0)) == 1
        ),
        "target_rounding_certificate_pass": int(metrics.get("target_unique_rounding_pass", 0)),
        "cuda_target_relative_error": (
            float(cuda_metrics.get("cuda_target_relative_error", np.inf))
            if target_mismatch == 0 else float("inf")
        ),
        "target_uncertified_count": int(
            metrics["target_count"] - metrics["target_certified_count"]
            + int(cuda_metrics.get("target_uncertified_count", 0))
        ),
        "target_replay_bit_mismatch_count": target_mismatch,
        "law_control_pass": int(
            law.metrics.get("cdf_statistics_pass", 0)
            and law.metrics.get("sample_eigenmoments_pass", 0)
            and law.metrics.get("stationarity_simultaneous_pass", 0)
            and law.metrics.get("reversibility_simultaneous_pass", 0)
        ),
        "all_four_colors_pass": int(cuda_metrics.get("all_four_colors_pass", 0)),
        "half_full_duration_pass": int(cuda_metrics.get("half_full_duration_pass", 0)),
        "negative_fixtures_pass": int(negative_pass),
        "serial_multipath_target_parity_pass": int(preflight_metrics.get("serial_batch_parity_pass", 0)),
        "target_path_isolation_pass": int(preflight_metrics.get("no_cross_path_write_pass", 0)),
        **NO_WORK,
    })
    atomic_write_json(run_dir / "multipath_target_controls.json", {
        "schema": RUN_SCHEMA + "-target-controls", "schema_version": 1,
        "identity_metrics": identity.metrics, "law_metrics": law.metrics,
        "cuda_identity_metrics": cuda_metrics, "identity_rows": identity.rows,
        "law_rows": law.rows, "cuda_identity_rows": cuda_rows, **NO_WORK,
    })
    atomic_write_csv(run_dir / "multipath_target_identity.csv", cuda_rows)
    gate = evaluate_multipath_target(metrics)
    _save_gate(run_dir, "target", metrics, gate)
    return gate


def _failed_stage_gate(run_dir: Path, kind: str, exc: Exception) -> dict[str, Any]:
    atomic_write_json(run_dir / f"{kind}_failure.json", {
        "schema": RUN_SCHEMA + "-stage-failure", "schema_version": 1,
        "stage": kind, "error_type": type(exc).__name__, "error": str(exc), **NO_WORK,
    })
    gate = {
        "schema": "experiment12-d0-jacobi-rb-cuda-multipath-gate",
        "schema_version": 1, "gate": f"multipath_{kind}",
        "claim_scope": "stage failed before complete evidence",
        "evaluation_status": "evaluated", "passed": 0,
        "numerically_valid": 0, "resource_valid": 0,
        "subchecks": {
            "stage_execution_pass": {
                "value": 0, "operator": "==", "threshold": 1, "passed": 0
            }
        }, **NO_WORK,
    }
    atomic_write_json(run_dir / f"multipath_{kind}_gate.json", gate)
    return gate


def _finish(
    run_dir: Path, args: argparse.Namespace, provenance: Mapping[str, Any],
    gates: Mapping[str, Mapping[str, Any]],
) -> int:
    workflow = evaluate_multipath_workflow(
        provenance=provenance, preflight_gate=gates["preflight"],
        pilot_gate=gates["pilot"], kernel_gate=gates["kernel"],
        target_gate=gates["target"], require_gate=args.require_gate,
    )
    decision = decide_multipath_workflow(
        provenance=provenance, preflight_gate=gates["preflight"],
        pilot_gate=gates["pilot"], kernel_gate=gates["kernel"],
        target_gate=gates["target"],
    )
    atomic_write_json(run_dir / "multipath_workflow_gate.json", workflow)
    atomic_write_json(run_dir / "multipath_decision.json", decision)
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    required_pass = int(workflow["required_gate_pass"])
    _write_status(
        run_dir, status="complete", outcome="complete" if required_pass else "gate_failed",
        phase=args.stage, required_gate=args.require_gate,
        required_gate_pass=required_pass, decision=decision["decision"],
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_record_count=len(registry["records"]),
        artifact_registry_size=registry_path.stat().st_size,
    )
    return 0 if required_pass else 2


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    print(f"Jacobi RB multi-path run directory: {run_dir}")
    if resumed:
        _verify_terminal_registry(run_dir, stage=args.stage)
    else:
        # A new run may record fail-closed provenance/runtime evidence.  An
        # existing run is not writable until all resume bindings below pass.
        args._active_run_dir = run_dir
    provenance = verify_and_readjudicate_jacobi_rb_cuda_multipath_parent(
        args.parent_cuda_run_dir
    )
    config = _scientific_config(args)
    fingerprint = config_fingerprint(config)
    source_hash, sources = _source_record(Path(args.parent_cuda_run_dir))
    device = torch.device(args.device)
    backend = configure_exact_torch_backend(device)
    _freeze(
        run_dir / "scientific_config.json", config, require_existing=resumed
    )
    _freeze(
        run_dir / "parent_provenance.json", provenance, require_existing=resumed
    )
    _freeze(run_dir / "run_manifest.json", {
        "schema": RUN_SCHEMA, "schema_version": 1, "claim_scope": CLAIM_SCOPE,
        "scientific_config_sha256": fingerprint, "source_fingerprint": source_hash,
        "source_paths": sources, "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "python": platform.python_version(), "torch": torch.__version__,
        "device": str(device), "exact_backend": backend, **NO_WORK,
    }, require_existing=resumed)
    if resumed:
        _validate_prerequisite_shard_families(run_dir, args)
    # Only now may a resumed run be mutated or terminally resealed.
    args._active_run_dir = run_dir
    _write_status(
        run_dir, status="running", phase=args.stage, required_gate=args.require_gate,
        scientific_config_sha256=fingerprint,
    )
    gates = {
        "preflight": _existing_gate(run_dir, "preflight", "preflight not run"),
        "pilot": _existing_gate(run_dir, "pilot", "pilot not run"),
        "kernel": _existing_gate(run_dir, "kernel", "kernel not run"),
        "target": _existing_gate(run_dir, "target", "target not run"),
    }
    if args.stage in {"preflight", "all"}:
        try:
            gates["preflight"] = _preflight(run_dir, args, provenance)
        except (RuntimeError, ValueError) as exc:
            gates["preflight"] = _failed_stage_gate(run_dir, "preflight", exc)
    if args.stage in {"pilot", "all"}:
        if not _passed(gates["preflight"]):
            gates["pilot"] = not_evaluated_gate("pilot", "preflight gate failed")
            atomic_write_json(run_dir / "multipath_pilot_gate.json", gates["pilot"])
        else:
            try:
                gates["pilot"] = _performance_stage(run_dir, args, pilot=True)
            except (RuntimeError, ValueError) as exc:
                gates["pilot"] = _failed_stage_gate(run_dir, "pilot", exc)
    if args.stage in {"kernel", "all"}:
        if not _passed(gates["pilot"]):
            gates["kernel"] = not_evaluated_gate("kernel", "pilot gate failed")
            atomic_write_json(run_dir / "multipath_kernel_gate.json", gates["kernel"])
        else:
            try:
                gates["kernel"] = _performance_stage(run_dir, args, pilot=False)
            except (RuntimeError, ValueError) as exc:
                gates["kernel"] = _failed_stage_gate(run_dir, "kernel", exc)
    if args.stage in {"target", "all"}:
        if not _passed(gates["kernel"]):
            gates["target"] = not_evaluated_gate("target", "kernel gate failed")
            atomic_write_json(run_dir / "multipath_target_gate.json", gates["target"])
        else:
            try:
                gates["target"] = _target(run_dir, args)
            except (RuntimeError, ValueError) as exc:
                gates["target"] = _failed_stage_gate(run_dir, "target", exc)
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
            atomic_write_json(run_dir / "unexpected_failure.json", {
                "error_type": type(exc).__name__, "error": str(exc), **NO_WORK,
            })
            registry = _artifact_registry(run_dir)
            atomic_write_json(run_dir / "artifact_registry.json", registry)
            registry_path = run_dir / "artifact_registry.json"
            _write_status(
                run_dir, status="complete", outcome="error", phase=args.stage,
                required_gate=args.require_gate, required_gate_pass=0,
                artifact_registry_sha256=file_fingerprint(registry_path),
                artifact_registry_record_count=len(registry["records"]),
                artifact_registry_size=registry_path.stat().st_size,
            )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
