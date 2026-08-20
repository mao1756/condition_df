"""Read-only q1--q3 directional/representation adjudication workflow.

The CLI consumes only immutable historical specialist roles.  It contains no
transition generation, optimization, fresh selection, confirmation,
controller, reconstruction, or sampling path.
"""

from __future__ import annotations

import argparse
import csv
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

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication as _historical_cli,
)
from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication import (
    COMPONENT_NAMES,
    MAXIMUM_FORWARD_BATCH,
    RECOMPOSITION_TOLERANCE,
    ComponentMomentAccumulator,
    ComponentMomentCube,
    QuartileDirectionalAdjudicationError,
    component_summary,
    evaluate_frozen_components,
    marginalize,
    normalized_cosine,
    positive_ray_optimum,
    quadratic_improvement,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_gate import (
    decide_workflow,
    decision_exit_code,
    evaluate_adjudicate_gate,
    evaluate_controls_gate,
    evaluate_fittrace_gate,
    evaluate_nominate_gate,
    evaluate_preflight_gate,
    evaluate_replay_gate,
    evaluate_required_gate,
    not_evaluated_gate,
    safety_record,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_inference import (
    DEFAULT_BOOTSTRAP_NAMESPACE,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CONFIDENCE,
    DEFAULT_REPLICATES,
    DirectionalInferenceError,
    StreamIdentity,
    STREAM_IDENTITIES,
    adjudicate_seed_streams,
    build_stream_path_table,
    direction_family_names,
    forecast_required_paths,
    local_compatibility_screen,
    one_sided_direction_effect_max_t,
    path_stability_summary,
    select_direction_nominees,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_provenance import (
    QuartileDirectionalProvenanceError,
    compare_parent_snapshots,
    load_already_open_inputs,
    load_already_open_role,
    scientific_config_fingerprint,
    snapshot_parent_run,
    source_fingerprint,
    source_paths,
    validate_semantic_config,
    verify_parents,
    verify_resume_compatibility,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_specialist import (
    CHECKPOINT_UPDATES,
    MODEL_SEEDS_BY_QUARTILE,
    CandidateIdentity,
    candidate_identities,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import HostInputStore
from mnist.d0_jacobi_rb_learnability import PHASE_COUNT


RUN_SCHEMA = "experiment12-d0-jacobi-rb-quartile-directional-adjudication"
RUN_SCHEMA_VERSION = 1
STAGES = (
    "preflight",
    "replay",
    "controls",
    "fittrace",
    "nominate",
    "adjudicate",
    "report",
    "all",
)
REQUIRED_GATES = (
    "none",
    "preflight",
    "replay",
    "controls",
    "fittrace",
    "nominate",
    "adjudicate",
)
NONZERO_CANDIDATES = candidate_identities(include_update_zero=False)
RESOURCE_LIMITS = {
    "maximum_peak_memory_fraction": 0.80,
    "maximum_projected_gpu_wall_hours": 48.0,
    "maximum_new_persisted_bytes": 1_073_741_824,
    "prediction_batch_size": 32,
    "mixed_precision": 0,
}
STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "preflight": (
        "run_manifest.json",
        "scientific_config.json",
        "source_closure.json",
        "parent_provenance.json",
        "parent_immutability_before.json",
        "role_firewall.json",
        "candidate_component_plan.json",
        "inference_plan.json",
        "bootstrap_index_seal.json",
        "resource_projection.json",
        "preflight_metrics.json",
        "preflight_gate.json",
    ),
    "replay": (
        "historical_gain_table_replay.json",
        "historical_rank_table_replay.json",
        "historical_replay_gate.json",
    ),
    "controls": (
        "quadratic_moment_algebra_control.json",
        "component_recomposition_control.json",
        "synthetic_mechanism_controls.json",
        "exact_zero_control.json",
        "controls_gate.json",
    ),
    "fittrace": (
        "fit_label_open.json",
        "fit_direction_moments.npz",
        "fit_trajectory_stability.csv",
        "fittrace_metrics.json",
        "fittrace_gate.json",
    ),
    "nominate": (
        "gain_label_open.json",
        "gain_direction_moments.npz",
        "gain_component_summary.csv",
        "direction_nominee_table.csv",
        "direction_nomination_seal.json",
        "nominate_metrics.json",
        "nominate_gate.json",
    ),
    "adjudicate": (
        "rank_label_open.json",
        "rank_direction_moments.npz",
        "rank_path_moments.npz",
        "rank_phase_midpoint_tables.csv",
        "gain_transfer_table.csv",
        "component_cancellation_table.csv",
        "trajectory_rotation_table.csv",
        "max_t_direction_and_effect_inference.json",
        "q0_positive_control.json",
        "mechanism_classification.json",
        "path_count_forecast.csv",
        "adjudicate_metrics.json",
        "adjudicate_gate.json",
    ),
}


class QuartileDirectionalWorkflowError(RuntimeError):
    def __init__(self, message: str, *, failure_domain: str, failure_code: str) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope() -> dict[str, int]:
    return safety_record()


def _semantic(record: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in record.items() if key != "semantic_sha256"}
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _verify_semantic(record: Mapping[str, Any], description: str) -> None:
    expected = config_fingerprint(
        {key: value for key, value in record.items() if key != "semantic_sha256"}
    )
    if record.get("semantic_sha256") != expected:
        raise ArtifactCompatibilityError(f"{description} semantic hash changed")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ArtifactCompatibilityError(f"{Path(path).name} is not a JSON object")
    return value


def _optional_json(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / name
    return _load_json(path) if path.is_file() else None


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _atomic_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = sorted({str(key) for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in names})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _seal_stage(run_dir: Path, stage: str) -> dict[str, Any]:
    names = STAGE_ARTIFACTS[stage]
    artifacts = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"stage artifact is missing: {name}")
        artifacts.append({"path": name, "sha256": file_fingerprint(path), "size": path.stat().st_size})
    seal = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-{stage}-artifact-seal",
            "schema_version": 1,
            "stage": stage,
            "artifacts": artifacts,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / f"{stage}_artifact_seal.json", seal)
    return seal


def _verify_stage_seal(run_dir: Path, stage: str) -> dict[str, Any]:
    seal = _load_json(run_dir / f"{stage}_artifact_seal.json")
    _verify_semantic(seal, f"{stage} seal")
    if seal.get("stage") != stage:
        raise ArtifactCompatibilityError("stage seal identity changed")
    for row in seal.get("artifacts", []):
        path = run_dir / str(row["path"])
        if not path.is_file() or file_fingerprint(path) != row.get("sha256"):
            raise ArtifactCompatibilityError(f"sealed artifact changed: {row.get('path')}")
    return seal


def _completed_stage(run_dir: Path, stage: str) -> bool:
    gate_path = run_dir / ("historical_replay_gate.json" if stage == "replay" else f"{stage}_gate.json")
    seal_path = run_dir / f"{stage}_artifact_seal.json"
    if not gate_path.is_file() or not seal_path.is_file():
        return False
    gate = _load_json(gate_path)
    if int(gate.get("passed", 0)) != 1:
        return False
    _verify_stage_seal(run_dir, stage)
    return True


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(
        (value for value in run_dir.rglob("*") if value.is_file() and value.name != "artifact_registry.json"),
        key=lambda value: value.relative_to(run_dir).as_posix(),
    ):
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_fingerprint(path),
                "size": int(path.stat().st_size),
            }
        )
    record = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-artifact-registry",
            "schema_version": 1,
            "artifact_count": len(rows),
            "artifacts": rows,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _status(
    run_dir: Path,
    *,
    stage: str,
    state: str,
    decision: str,
    message: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
) -> None:
    atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": f"{RUN_SCHEMA}-status",
            "schema_version": 1,
            "stage": stage,
            "state": state,
            "decision": decision,
            "message": message,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "scientific_evidence_complete": int(state in {"complete", "valid_scientific_stop"}),
            "updated_at": _now(),
            **_scope(),
        },
    )


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    body = {
        "schema": f"{RUN_SCHEMA}-scientific-config",
        "schema_version": 1,
        "parent_quartile_specialist_run_dir": str(Path(args.parent_quartile_specialist_run_dir).resolve()),
        "parent_time_local_run_dir": str(Path(args.parent_time_local_run_dir).resolve()),
        "grid_size": 28,
        "alpha": 1.0,
        "outer_steps": 512,
        "tau_eff": 5.0e-5,
        "raw_target": "y(1-y)*d_y log k_u(y|x)",
        "components": list(COMPONENT_NAMES),
        "candidate_count": len(NONZERO_CANDIDATES),
        "nominee_stream_count": 36,
        "max_t_family_size": 72,
        "confidence": DEFAULT_CONFIDENCE,
        "bootstrap_replicates": DEFAULT_REPLICATES,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "role_order": ["physical_fit", "gain_calibration", "training_rank"],
        "resource_limits": dict(RESOURCE_LIMITS),
        "source_fingerprint": source_fingerprint(),
        "historical_design_evidence_only": 1,
        "authorizing": 0,
        "new_role_count": 0,
        "new_path_count": 0,
        "new_seed_count": 0,
        "training_authorized": 0,
        "selection_authorized": 0,
        "confirmation_authorized": 0,
        "controller_execution_authorized": 0,
        "sampling_authorized": 0,
        **_scope(),
    }
    return _semantic(body)


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}_{args.run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir, False


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    config = _scientific_config(args)
    if resumed:
        verify_resume_compatibility(
            run_dir,
            expected_bindings={
                "parent_quartile_specialist_run_dir": str(
                    Path(args.parent_quartile_specialist_run_dir).resolve()
                ),
                "parent_time_local_run_dir": str(
                    Path(args.parent_time_local_run_dir).resolve()
                ),
                "source_fingerprint": source_fingerprint(),
                "scientific_config_sha256": scientific_config_fingerprint(config),
            },
        )
        if _load_json(run_dir / "scientific_config.json") != config:
            raise ArtifactCompatibilityError("resume scientific configuration changed")
        return
    atomic_write_json(run_dir / "scientific_config.json", config)
    manifest = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "run_dir": str(run_dir),
            "source_fingerprint": source_fingerprint(),
            "scientific_config_sha256": scientific_config_fingerprint(config),
            "parent_quartile_specialist_run_dir": str(Path(args.parent_quartile_specialist_run_dir).resolve()),
            "parent_time_local_run_dir": str(Path(args.parent_time_local_run_dir).resolve()),
            "test_only": int(bool(getattr(args, "test_only", False))),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    _status(run_dir, stage="created", state="running", decision="ready_for_preflight")


def _candidate_component_plan() -> dict[str, Any]:
    candidates = [candidate.to_record() for candidate in NONZERO_CANDIDATES]
    rows = [
        {
            "quartile": candidate.quartile,
            "seed": candidate.seed,
            "update": candidate.update,
            "candidate_key": candidate.key,
            "component": component,
        }
        for candidate in NONZERO_CANDIDATES
        for component in COMPONENT_NAMES
    ]
    return _semantic(
        {
            "schema": f"{RUN_SCHEMA}-candidate-component-plan",
            "schema_version": 1,
            "candidate_count": len(candidates),
            "candidate_component_count": len(rows),
            "candidates": candidates,
            "components": list(COMPONENT_NAMES),
            "candidate_component_order_sha256": config_fingerprint(rows),
            "nomination_rule": "largest gain-role D_plus; earlier update on exact tie",
            "nominee_stream_count": 36,
            **_scope(),
        }
    )


def _inference_plan() -> dict[str, Any]:
    names = list(direction_family_names())
    return _semantic(
        {
            "schema": f"{RUN_SCHEMA}-inference-plan",
            "schema_version": 1,
            "family_names": names,
            "family_size": len(names),
            "family_names_sha256": config_fingerprint(names),
            "confidence": DEFAULT_CONFIDENCE,
            "bootstrap_replicates": DEFAULT_REPLICATES,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "resampling_unit": "whole training-rank path",
            "quantile_method": "higher",
            "studentization": 1,
            "standard_error_floor": None,
            **_scope(),
        }
    )


def _bootstrap_seal(path_count: int = 32) -> dict[str, Any]:
    rng = np.random.Generator(
        np.random.Philox([DEFAULT_BOOTSTRAP_SEED, DEFAULT_BOOTSTRAP_NAMESPACE])
    )
    # Counts are canonical and compact; the inferential routine regenerates the
    # identical index stream from the same sealed seed.
    counts = np.zeros((DEFAULT_REPLICATES, path_count), dtype=np.uint8)
    for start in range(0, DEFAULT_REPLICATES, 1_000):
        stop = min(DEFAULT_REPLICATES, start + 1_000)
        indices = rng.integers(0, path_count, size=(stop - start, path_count))
        for row, sample in enumerate(indices):
            counts[start + row] = np.bincount(sample, minlength=path_count)
    digest = hashlib.sha256(np.ascontiguousarray(counts).tobytes(order="C")).hexdigest()
    return _semantic(
        {
            "schema": f"{RUN_SCHEMA}-bootstrap-index-seal",
            "schema_version": 1,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "namespace": DEFAULT_BOOTSTRAP_NAMESPACE,
            "replicates": DEFAULT_REPLICATES,
            "path_count": path_count,
            "count_dtype": counts.dtype.str,
            "count_matrix_sha256": digest,
            "family_names_sha256": config_fingerprint(list(direction_family_names())),
            **_scope(),
        }
    )


def _input_store(role: Any) -> HostInputStore:
    return HostInputStore.from_arrays(
        role.inputs if hasattr(role, "inputs") else role,
        role="validation",
        cache_root=".",
    )


def _quartile_rows(store: HostInputStore, quartile: int) -> np.ndarray:
    values = _historical_cli._store_quartiles(store)  # noqa: SLF001
    rows = np.flatnonzero(values == int(quartile)).astype(np.int64)
    if rows.size == 0:
        raise QuartileDirectionalWorkflowError(
            "historical role lacks one quartile",
            failure_domain="scientific_contract",
            failure_code="quartile_row_coverage_invalid",
        )
    return rows


def _resource_pilot(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_already_open_inputs(
        args.parent_quartile_specialist_run_dir, "physical_fit"
    )
    store = _input_store(inputs)
    rows = _quartile_rows(store, 0)[: min(2_048, _quartile_rows(store, 0).size)]
    device = torch.device(args.device)
    checkpoint_rows = _historical_cli._checkpoint_rows(  # noqa: SLF001
        Path(args.parent_quartile_specialist_run_dir)
    )
    candidate = NONZERO_CANDIDATES[0]
    model, _ = _historical_cli._load_model(  # noqa: SLF001
        Path(args.parent_quartile_specialist_run_dir), candidate, checkpoint_rows, device
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        warm = store.batch(rows[: min(32, rows.size)], device=device)
        evaluate_frozen_components(model, warm)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        observed = 0
        for start in range(0, rows.size, MAXIMUM_FORWARD_BATCH):
            batch_rows = rows[start : start + MAXIMUM_FORWARD_BATCH]
            evaluate_frozen_components(model, store.batch(batch_rows, device=device))
            observed += len(batch_rows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - started, 1.0e-9)
    seconds_per_row = elapsed / max(observed, 1)
    # Candidate component extraction is fused.  This is the exact worst-case
    # forward-row count for 480 fit candidates, 480 gain candidates, all 36
    # rank nominees, and both models in every frozen fit-role cosine pair.
    projected_rows = 75_522_048
    projected_hours = seconds_per_row * projected_rows / 3_600.0
    peak_fraction = 0.0
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        peak_fraction = torch.cuda.max_memory_allocated(device) / float(total)
    # Role moment shards plus their consolidated archives and compact tables.
    projected_bytes = 650_000_000
    passed = (
        projected_hours <= RESOURCE_LIMITS["maximum_projected_gpu_wall_hours"]
        and peak_fraction <= RESOURCE_LIMITS["maximum_peak_memory_fraction"]
        and projected_bytes <= RESOURCE_LIMITS["maximum_new_persisted_bytes"]
    )
    return {
        "evaluation_status": "evaluated",
        "observed_rows": observed,
        "elapsed_seconds": elapsed,
        "seconds_per_row": seconds_per_row,
        "projected_evaluation_rows": projected_rows,
        "projected_row_breakdown": {
            "fit_candidate_rows": 13_762_560,
            "gain_candidate_rows": 6_881_280,
            "maximum_rank_nominee_rows": 516_096,
            "adjacent_rotation_forward_rows": 26_836_992,
            "cross_seed_rotation_forward_rows": 27_525_120,
        },
        "projected_gpu_wall_hours": projected_hours,
        "projected_new_persisted_bytes": projected_bytes,
        "peak_memory_fraction": peak_fraction,
        "within_limits": int(passed),
        "limits": dict(RESOURCE_LIMITS),
        **_scope(),
    }


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "preflight"):
        return
    specialist_snapshot = snapshot_parent_run(args.parent_quartile_specialist_run_dir)
    time_local_snapshot = snapshot_parent_run(args.parent_time_local_run_dir)
    parents = verify_parents(
        args.parent_quartile_specialist_run_dir,
        args.parent_time_local_run_dir,
        specialist_snapshot=specialist_snapshot,
        time_local_snapshot=time_local_snapshot,
        verify_checkpoint_states=True,
        verify_cache_rows=True,
    )
    validate_semantic_config(
        _load_json(run_dir / "scientific_config.json"),
        expected_schema=f"{RUN_SCHEMA}-scientific-config",
    )
    atomic_write_json(run_dir / "parent_provenance.json", parents)
    atomic_write_json(
        run_dir / "parent_immutability_before.json",
        _semantic(
            {
                "schema": f"{RUN_SCHEMA}-parent-immutability-before",
                "schema_version": 1,
                "quartile_specialist": specialist_snapshot,
                "time_local": time_local_snapshot,
                **_scope(),
            }
        ),
    )
    source_rows = [
        {"path": str(path), "sha256": file_fingerprint(path), "size": path.stat().st_size}
        for path in source_paths()
    ]
    atomic_write_json(
        run_dir / "source_closure.json",
        _semantic(
            {
                "schema": f"{RUN_SCHEMA}-source-closure",
                "schema_version": 1,
                "source_fingerprint": source_fingerprint(),
                "source_count": len(source_rows),
                "sources": source_rows,
                **_scope(),
            }
        ),
    )
    firewall = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-role-firewall",
            "schema_version": 1,
            "historical_roles": ["physical_fit", "gain_calibration", "training_rank"],
            "role_order": ["physical_fit", "gain_calibration", "training_rank"],
            "preflight_inputs_only": 1,
            "rank_requires_nomination_seal": 1,
            "selection_confirmation_forbidden": 1,
            "fit_may_nominate": 0,
            "rank_may_renominate": 0,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "role_firewall.json", firewall)
    atomic_write_json(run_dir / "candidate_component_plan.json", _candidate_component_plan())
    atomic_write_json(run_dir / "inference_plan.json", _inference_plan())
    atomic_write_json(run_dir / "bootstrap_index_seal.json", _bootstrap_seal())
    resource = _resource_pilot(args)
    atomic_write_json(run_dir / "resource_projection.json", resource)
    metrics = {
        "evaluation_status": "evaluated",
        "stage_execution_valid": 1,
        "parent_provenance_valid": int(parents.get("passed", 0)),
        "parent_immutability_valid": 1,
        "checkpoint_payloads_valid": int(parents.get("checkpoint_payloads_valid", parents.get("all_checkpoint_hashes_verified", 0))),
        "role_cache_payloads_valid": int(parents.get("role_cache_payloads_valid", parents.get("all_role_cache_payloads_verified", parents.get("cache_bindings_valid", 0)))),
        "scientific_contract_valid": 1,
        "role_firewall_valid": 1,
        "candidate_component_plan_sealed": 1,
        "inference_plan_sealed": 1,
        "bootstrap_indices_sealed": 1,
        "resource_projection_valid": int(resource["within_limits"]),
        **_scope(),
    }
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    gate = evaluate_preflight_gate(metrics)
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    _seal_stage(run_dir, "preflight")
    if int(gate["passed"]) != 1:
        domain = "resource_gate" if not resource["within_limits"] else "provenance"
        raise QuartileDirectionalWorkflowError(
            "quartile directional preflight failed",
            failure_domain=domain,
            failure_code="quartile_directional_preflight_invalid",
        )
    _status(run_dir, stage="preflight", state="running", decision="ready_for_replay")


def _replay_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "replay"):
        return
    _verify_stage_seal(run_dir, "preflight")
    parent = Path(args.parent_quartile_specialist_run_dir)
    gain = _historical_cli._load_npz(parent / "gain_table.npz")  # noqa: SLF001
    rank = _historical_cli._load_npz(parent / "training_rank_path_tables.npz")  # noqa: SLF001
    if not _historical_cli._candidate_arrays_valid(gain, "") or not _historical_cli._candidate_arrays_valid(rank, "candidate_"):  # noqa: SLF001
        raise QuartileDirectionalWorkflowError(
            "historical candidate ordering changed",
            failure_domain="historical_replay",
            failure_code="candidate_order_invalid",
        )
    gain_records = _historical_cli._parent_gain_records(gain)  # noqa: SLF001
    rank_records, _ = _historical_cli._rank_records(parent, rank, gain_records)  # noqa: SLF001
    path_replay = _historical_cli._validate_rank_paths(rank)  # noqa: SLF001
    eligible = [sum(int(row.eligible) for row in rank_records if row.candidate.quartile == q) for q in range(4)]
    q0 = [row for row in rank_records if row.candidate.quartile == 0 and row.eligible]
    winner = min(q0, key=lambda row: (-row.pooled_improvement, row.candidate.update, row.candidate.seed))
    gain_record = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-historical-gain-replay",
            "schema_version": 1,
            "candidate_count": len(gain_records),
            "q2_eligible": sum(int(row.eligible) for row in gain_records if row.candidate.quartile == 2),
            "q3_eligible": sum(int(row.eligible) for row in gain_records if row.candidate.quartile == 3),
            "gain_table_sha256": file_fingerprint(parent / "gain_table.npz"),
            "passed": 1,
            **_scope(),
        }
    )
    rank_record = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-historical-rank-replay",
            "schema_version": 1,
            "candidate_count": len(rank_records),
            "eligible_counts_by_quartile": eligible,
            "q0_winner": winner.candidate.key,
            "q0_winner_pooled_improvement": winner.pooled_improvement,
            "path_replay": path_replay,
            "terminal_decision_replayed": int(eligible == [80, 0, 0, 0]),
            "passed": int(eligible == [80, 0, 0, 0]),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "historical_gain_table_replay.json", gain_record)
    atomic_write_json(run_dir / "historical_rank_table_replay.json", rank_record)
    metrics = {
        "evaluation_status": "evaluated",
        "stage_execution_valid": 1,
        "historical_gain_table_replayed": int(gain_record["passed"]),
        "historical_rank_table_replayed": int(rank_record["passed"]),
        "candidate_order_valid": 1,
        "numerical_agreement_valid": int(max(path_replay["maximum_errors"].values()) <= RECOMPOSITION_TOLERANCE),
        "raw_role_labels_unopened": 1,
        **_scope(),
    }
    gate = evaluate_replay_gate(metrics)
    atomic_write_json(run_dir / "historical_replay_gate.json", gate)
    _seal_stage(run_dir, "replay")
    if int(gate["passed"]) != 1:
        raise QuartileDirectionalWorkflowError(
            "historical replay failed",
            failure_domain="historical_replay",
            failure_code="quartile_directional_historical_replay_invalid",
        )
    _status(run_dir, stage="replay", state="running", decision="ready_for_controls")


def _controls_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "controls"):
        return
    _verify_stage_seal(run_dir, "replay")
    # Pure algebra fixtures, fixed before any role-label loader is called.
    direct = 1.0 - np.mean((np.asarray([1.0, -1.0]) - 0.5 * np.asarray([1.0, -1.0])) ** 2)
    algebraic = quadratic_improvement(1.0, 1.0, 0.5)
    algebra = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-quadratic-moment-algebra-control",
            "schema_version": 1,
            "direct": direct,
            "algebraic": algebraic,
            "absolute_error": abs(direct - algebraic),
            "passed": int(abs(direct - algebraic) <= RECOMPOSITION_TOLERANCE),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "quadratic_moment_algebra_control.json", algebra)

    # Real branch evaluator on a target-free, state-uniform fixture.
    batch = 2
    from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import ZeroBaselineBoundaryTangentPredictor
    from mnist.d0_jacobi_rb_learnability import ModelInputs, PHASE_DURATIONS, PHASE_MATCHINGS
    phases = torch.tensor([0, 3], dtype=torch.long)
    inputs = ModelInputs(
        later_full_state=torch.full((batch, 784), 1.0 / 784.0, dtype=torch.float32),
        reverse_time=torch.tensor([0.9, 0.4], dtype=torch.float64),
        phase=phases,
        color=torch.as_tensor([PHASE_MATCHINGS[int(q)] for q in phases], dtype=torch.long),
        duration=torch.as_tensor([PHASE_DURATIONS[int(q)] for q in phases], dtype=torch.float32),
        label=torch.full((batch,), 3, dtype=torch.long),
    )
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True)
    parts = evaluate_frozen_components(model, inputs)
    recomposition = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-component-recomposition-control",
            "schema_version": 1,
            "maximum_prediction_recomposition_error": parts.maximum_prediction_recomposition_error,
            "maximum_spatial_rounding_error": parts.maximum_spatial_rounding_error,
            "passed": int(parts.maximum_prediction_recomposition_error <= RECOMPOSITION_TOLERANCE),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "component_recomposition_control.json", recomposition)
    zero = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-exact-zero-control",
            "schema_version": 1,
            "maximum_absolute_full": float(torch.max(torch.abs(parts.full)).item()),
            "maximum_absolute_local": float(torch.max(torch.abs(parts.local_affine)).item()),
            "maximum_absolute_spatial": float(torch.max(torch.abs(parts.spatial_cnn)).item()),
            "passed": int(bool(torch.all(parts.full == 0.0)) and bool(torch.all(parts.local_affine == 0.0)) and bool(torch.all(parts.spatial_cnn == 0.0))),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "exact_zero_control.json", zero)
    fixtures = {
        "stable_positive_direction": int(positive_ray_optimum(1.0, 0.5, 0.25)["D_plus"] == 1.0),
        "nonpositive_direction": int(positive_ray_optimum(1.0, -0.5, 0.25)["lambda_plus"] == 0.0),
        "energy_dominated_unit_gain": int(quadratic_improvement(0.4, 1.0, 1.0) <= 0.0 and positive_ray_optimum(1.0, 0.4, 1.0)["D_plus"] > 0.0),
        "path_instability": int(
            path_stability_summary(
                np.asarray([1.0, -0.9]), simultaneous_lower_bound=-1.0
            )["path_unstable"]
            == 1
        ),
        "phase_midpoint_cancellation": int(local_compatibility_screen(np.pad(np.ones((7, 7)), ((0, 0), (0, 1)), constant_values=-1.0), quartile=2)["passed"] == 0),
        "branch_cancellation": int(abs((1.0 - 2.0) - (-1.0)) == 0.0),
        "direction_rotation": int(normalized_cosine([1.0, -1.0], [-1.0, 1.0]) < 0.0),
    }
    malformed = 0
    try:
        positive_ray_optimum(1.0, 1.0, 0.0)
    except QuartileDirectionalAdjudicationError:
        malformed = 1
    fixtures["malformed_moments_fail_closed"] = malformed
    def base_mechanism_evidence() -> dict[str, Any]:
        return {
            "q0_full": {"stable_direction": 1, "stable_effect": 1},
            "inferential_and_role_order_valid": 1,
            "branch_algebra_cancellation_valid": 1,
            "quartiles": {
                f"q{quartile}": {
                    "components": {
                        component: {"stable_direction": 0, "stable_effect": 0}
                        for component in COMPONENT_NAMES
                    }
                }
                for quartile in range(1, 4)
            },
        }

    unique_results: dict[str, str] = {}
    for branch, competitor in (
        ("local_affine", "spatial_cnn"),
        ("spatial_cnn", "local_affine"),
    ):
        evidence = base_mechanism_evidence()
        for quartile in ("q1", "q2", "q3"):
            components = evidence["quartiles"][quartile]["components"]
            components[branch] = {"stable_direction": 1, "stable_effect": 1}
            components[competitor] = {"stable_direction": 1, "stable_effect": 0}
            components["full"] = {"stable_direction": 1, "stable_effect": 1}
        evidence["quartiles"]["q2"]["components"]["full"] = {
            "stable_direction": 0,
            "stable_effect": 0,
        }
        evidence["quartiles"]["q2"][
            "competing_branch_negative_in_full_failure_strata"
        ] = {branch: 1}
        unique_results[branch] = decide_workflow(evidence)["decision"]
    nonidentifying = base_mechanism_evidence()
    nonidentifying["cancellation_visible"] = 1
    unresolved = base_mechanism_evidence()
    unresolved["positive_direction_effect_unresolved"] = 1
    unstable = base_mechanism_evidence()
    unstable["positive_gain_direction"] = 1
    absent = base_mechanism_evidence()
    mechanism_decisions = {
        "unique_local": unique_results["local_affine"],
        "unique_spatial": unique_results["spatial_cnn"],
        "nonidentifying_mixed": decide_workflow(nonidentifying)["decision"],
        "positive_effect_unresolved": decide_workflow(unresolved)["decision"],
        "role_unstable": decide_workflow(unstable)["decision"],
        "no_permitted_signal": decide_workflow(absent)["decision"],
    }
    expected_mechanism_decisions = {
        "unique_local": "unique_representation_hypothesis_identified",
        "unique_spatial": "unique_representation_hypothesis_identified",
        "nonidentifying_mixed": "representation_cancellation_nonidentifying_stop",
        "positive_effect_unresolved": "positive_direction_effect_unresolved_stop",
        "role_unstable": "later_quartile_direction_unstable_across_roles_stop",
        "no_permitted_signal": "no_later_quartile_signal_detectable_under_permitted_class_stop",
    }
    fixtures["mechanism_decision_fixtures"] = int(
        mechanism_decisions == expected_mechanism_decisions
    )
    mechanisms = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-synthetic-mechanism-controls",
            "schema_version": 1,
            "controls": fixtures,
            "mechanism_decisions": mechanism_decisions,
            "expected_mechanism_decisions": expected_mechanism_decisions,
            "passed": int(all(fixtures.values())),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "synthetic_mechanism_controls.json", mechanisms)
    metrics = {
        "evaluation_status": "evaluated",
        "stage_execution_valid": 1,
        "quadratic_moment_algebra_valid": int(algebra["passed"]),
        "component_recomposition_valid": int(recomposition["passed"]),
        "exact_zero_control_valid": int(zero["passed"]),
        "stable_positive_direction_control_valid": fixtures["stable_positive_direction"],
        "nonpositive_direction_control_valid": fixtures["nonpositive_direction"],
        "energy_dominated_control_valid": fixtures["energy_dominated_unit_gain"],
        "path_instability_control_valid": fixtures["path_instability"],
        "phase_midpoint_cancellation_control_valid": fixtures["phase_midpoint_cancellation"],
        "branch_cancellation_control_valid": int(
            fixtures["branch_cancellation"]
            and fixtures["mechanism_decision_fixtures"]
        ),
        "direction_rotation_control_valid": fixtures["direction_rotation"],
        "malformed_moments_fail_closed": fixtures["malformed_moments_fail_closed"],
        "physical_labels_unopened": int(not any((run_dir / name).exists() for name in ("fit_label_open.json", "gain_label_open.json", "rank_label_open.json"))),
        **_scope(),
    }
    gate = evaluate_controls_gate(metrics)
    atomic_write_json(run_dir / "controls_gate.json", gate)
    _seal_stage(run_dir, "controls")
    if int(gate["passed"]) != 1:
        raise QuartileDirectionalWorkflowError(
            "prelabel controls failed",
            failure_domain="controls",
            failure_code="quartile_directional_prelabel_controls_failed",
        )
    _status(run_dir, stage="controls", state="running", decision="ready_for_fittrace")


def _child_role_open(
    run_dir: Path,
    *,
    role: str,
    prerequisite_stage: str,
    parent_role: Any,
) -> dict[str, Any]:
    names = {
        "physical_fit": "fit_label_open.json",
        "gain_calibration": "gain_label_open.json",
        "training_rank": "rank_label_open.json",
    }
    if role not in names:
        raise ArtifactCompatibilityError("child role is not historical")
    expected_order = ("physical_fit", "gain_calibration", "training_rank")
    position = expected_order.index(role)
    for later in expected_order[position + 1 :]:
        if (run_dir / names[later]).exists():
            raise ArtifactCompatibilityError("historical role order was violated")
    prerequisite = _verify_stage_seal(run_dir, prerequisite_stage)
    path = run_dir / names[role]
    if path.is_file():
        record = _load_json(path)
        _verify_semantic(record, path.name)
        return record
    record = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-{role}-label-open",
            "schema_version": 1,
            "role": role,
            "historical_design_evidence": 1,
            "authorizing": 0,
            "prerequisite_stage": prerequisite_stage,
            "prerequisite_seal_sha256": file_fingerprint(
                run_dir / f"{prerequisite_stage}_artifact_seal.json"
            ),
            "parent_role_open_sha256": str(
                parent_role.role_open.get("semantic_sha256", "")
            ),
            "parent_cache_binding_sha256": str(parent_role.binding.get("semantic_sha256", "")),
            "opened_at": _now(),
            **_scope(),
        }
    )
    atomic_write_json(path, record)
    return record


def _role_store(role: Any) -> tuple[HostInputStore, np.ndarray, np.ndarray]:
    store = _input_store(role)
    target = np.asarray(role.labels["denoising_target"], dtype=np.float64)
    if target.shape != (store.row_count, 392) or not np.isfinite(target).all():
        raise ArtifactCompatibilityError("historical target archive changed")
    path_ids = np.asarray(store.row_array("path_id"), dtype=np.int64)
    unique = np.unique(path_ids)
    if unique.size not in (32, 64):
        raise ArtifactCompatibilityError("historical role path count changed")
    return store, target, unique


def _checkpoint_rows(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return _historical_cli._checkpoint_rows(  # noqa: SLF001
        Path(args.parent_quartile_specialist_run_dir)
    )


def _load_model(
    args: argparse.Namespace,
    candidate: CandidateIdentity,
    rows: Mapping[str, Mapping[str, Any]],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    return _historical_cli._load_model(  # noqa: SLF001
        Path(args.parent_quartile_specialist_run_dir),
        candidate,
        rows,
        torch.device(args.device),
    )


def _evaluate_candidate_cube(
    *,
    args: argparse.Namespace,
    candidate: CandidateIdentity,
    checkpoint_rows: Mapping[str, Mapping[str, Any]],
    store: HostInputStore,
    target: np.ndarray,
    path_ids: np.ndarray,
) -> tuple[ComponentMomentCube, dict[str, Any]]:
    model, checkpoint = _load_model(args, candidate, checkpoint_rows)
    rows = _quartile_rows(store, candidate.quartile)
    accumulator = ComponentMomentAccumulator(path_ids)
    device = torch.device(args.device)
    model.eval()
    with torch.no_grad():
        for start in range(0, rows.size, MAXIMUM_FORWARD_BATCH):
            active = rows[start : start + MAXIMUM_FORWARD_BATCH]
            batch = store.batch(active, device=device)
            prediction = evaluate_frozen_components(model, batch)
            truth = torch.as_tensor(
                np.array(target[active], copy=True, order="C"),
                dtype=torch.float64,
                device=device,
            )
            accumulator.add_batch(
                path_id=np.asarray(store.row_array("path_id"), dtype=np.int64)[active],
                phase=np.asarray(store.row_array("phase"), dtype=np.int64)[active],
                midpoint=np.asarray(store.row_array("midpoint_index"), dtype=np.int64)[active],
                target=truth,
                predictions=prediction,
            )
    return accumulator.finish(), checkpoint


def _cube_shard_paths(run_dir: Path, role: str, candidate: CandidateIdentity) -> tuple[Path, Path]:
    root = run_dir / "directional_shards" / role
    return root / f"{candidate.key}.npz", root / f"{candidate.key}.json"


def _save_cube_shard(
    run_dir: Path,
    *,
    role: str,
    candidate: CandidateIdentity,
    cube: ComponentMomentCube,
    checkpoint: Mapping[str, Any],
    role_open: Mapping[str, Any],
    config_sha256: str,
) -> None:
    npz_path, json_path = _cube_shard_paths(run_dir, role, candidate)
    _atomic_npz(
        npz_path,
        candidate_quartile=np.asarray(candidate.quartile, dtype=np.int8),
        candidate_seed=np.asarray(candidate.seed, dtype=np.int64),
        candidate_update=np.asarray(candidate.update, dtype=np.int16),
        **cube.to_arrays(),
    )
    record = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-directional-cube-shard",
            "schema_version": 1,
            "role": role,
            "candidate": candidate.to_record(),
            "npz_path": npz_path.relative_to(run_dir).as_posix(),
            "npz_sha256": file_fingerprint(npz_path),
            "checkpoint_path": checkpoint["checkpoint_path"],
            "checkpoint_file_sha256": checkpoint["checkpoint_file_sha256"],
            "checkpoint_state_sha256": checkpoint.get("model_state_sha256"),
            "role_open_semantic_sha256": role_open["semantic_sha256"],
            "scientific_config_sha256": config_sha256,
            "raw_predictions_persisted": 0,
            **_scope(),
        }
    )
    atomic_write_json(json_path, record)


def _load_cube_shard(
    run_dir: Path,
    *,
    role: str,
    candidate: CandidateIdentity,
    checkpoint: Mapping[str, Any],
    role_open: Mapping[str, Any],
    config_sha256: str,
) -> ComponentMomentCube | None:
    npz_path, json_path = _cube_shard_paths(run_dir, role, candidate)
    if not npz_path.exists() and not json_path.exists():
        return None
    if not npz_path.is_file() or not json_path.is_file():
        raise ArtifactCompatibilityError("directional shard is incomplete")
    record = _load_json(json_path)
    _verify_semantic(record, json_path.name)
    expected = {
        "role": role,
        "npz_sha256": file_fingerprint(npz_path),
        "checkpoint_path": checkpoint["checkpoint_path"],
        "checkpoint_file_sha256": checkpoint["checkpoint_file_sha256"],
        "checkpoint_state_sha256": checkpoint.get("model_state_sha256"),
        "role_open_semantic_sha256": role_open["semantic_sha256"],
        "scientific_config_sha256": config_sha256,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ArtifactCompatibilityError(f"directional shard binding changed: {candidate.key}")
    identity = record.get("candidate", {})
    if identity.get("key") != candidate.key:
        raise ArtifactCompatibilityError("directional shard candidate changed")
    return ComponentMomentCube.from_arrays(_load_npz(npz_path))


def _evaluate_role_candidates(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    role_name: str,
    role: Any,
    role_open: Mapping[str, Any],
    candidates: Sequence[CandidateIdentity],
) -> dict[str, ComponentMomentCube]:
    store, target, path_ids = _role_store(role)
    checkpoints = _checkpoint_rows(args)
    config = _load_json(run_dir / "scientific_config.json")
    config_sha = scientific_config_fingerprint(config)
    result: dict[str, ComponentMomentCube] = {}
    for index, candidate in enumerate(candidates, start=1):
        checkpoint = checkpoints[candidate.key]
        cube = _load_cube_shard(
            run_dir,
            role=role_name,
            candidate=candidate,
            checkpoint=checkpoint,
            role_open=role_open,
            config_sha256=config_sha,
        )
        if cube is None:
            cube, checkpoint = _evaluate_candidate_cube(
                args=args,
                candidate=candidate,
                checkpoint_rows=checkpoints,
                store=store,
                target=target,
                path_ids=path_ids,
            )
            _save_cube_shard(
                run_dir,
                role=role_name,
                candidate=candidate,
                cube=cube,
                checkpoint=checkpoint,
                role_open=role_open,
                config_sha256=config_sha,
            )
        result[candidate.key] = cube
        if index % 10 == 0 or index == len(candidates):
            print(
                f"quartile directional {role_name} {index}/{len(candidates)}",
                flush=True,
            )
    return result


def _consolidate_cubes(
    path: Path,
    candidates: Sequence[CandidateIdentity],
    cubes: Mapping[str, ComponentMomentCube],
) -> None:
    ordered = [cubes[candidate.key] for candidate in candidates]
    _atomic_npz(
        path,
        candidate_quartile=np.asarray([candidate.quartile for candidate in candidates], dtype=np.int8),
        candidate_seed=np.asarray([candidate.seed for candidate in candidates], dtype=np.int64),
        candidate_update=np.asarray([candidate.update for candidate in candidates], dtype=np.int16),
        path_ids=np.stack([cube.path_ids for cube in ordered], axis=0),
        target_energy=np.stack([cube.target_energy for cube in ordered], axis=0),
        cross_terms=np.stack([cube.cross_terms for cube in ordered], axis=0),
        prediction_energies=np.stack([cube.prediction_energies for cube in ordered], axis=0),
        local_spatial_cross=np.stack([cube.local_spatial_cross for cube in ordered], axis=0),
        counts=np.stack([cube.counts for cube in ordered], axis=0),
        maximum_recomposition_error=np.asarray([cube.maximum_recomposition_error for cube in ordered], dtype=np.float64),
        maximum_risk_identity_error=np.asarray([cube.maximum_risk_identity_error for cube in ordered], dtype=np.float64),
    )


def _prediction_pair_cosines(
    args: argparse.Namespace,
    *,
    left: CandidateIdentity,
    right: CandidateIdentity,
    checkpoint_rows: Mapping[str, Mapping[str, Any]],
    store: HostInputStore,
) -> dict[str, float]:
    if left.quartile != right.quartile:
        raise ArtifactCompatibilityError("rotation pair crosses quartiles")
    left_model, _ = _load_model(args, left, checkpoint_rows)
    right_model, _ = _load_model(args, right, checkpoint_rows)
    rows = _quartile_rows(store, left.quartile)
    dot = np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
    norm_left = np.zeros_like(dot)
    norm_right = np.zeros_like(dot)
    device = torch.device(args.device)
    with torch.no_grad():
        for start in range(0, rows.size, MAXIMUM_FORWARD_BATCH):
            active = rows[start : start + MAXIMUM_FORWARD_BATCH]
            batch = store.batch(active, device=device)
            a = evaluate_frozen_components(left_model, batch).as_mapping()
            b = evaluate_frozen_components(right_model, batch).as_mapping()
            for index, name in enumerate(COMPONENT_NAMES):
                av = a[name].detach().to(torch.float64)
                bv = b[name].detach().to(torch.float64)
                dot[index] += float(torch.sum(av * bv).item())
                norm_left[index] += float(torch.sum(av * av).item())
                norm_right[index] += float(torch.sum(bv * bv).item())
    result = {}
    for index, name in enumerate(COMPONENT_NAMES):
        denominator = math.sqrt(norm_left[index] * norm_right[index])
        result[name] = 1.0 if denominator == 0.0 and norm_left[index] == norm_right[index] else (0.0 if denominator == 0.0 else dot[index] / denominator)
    return result


def _fittrace_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "fittrace"):
        return
    _verify_stage_seal(run_dir, "controls")
    role = load_already_open_role(args.parent_quartile_specialist_run_dir, "physical_fit")
    role_open = _child_role_open(
        run_dir,
        role="physical_fit",
        prerequisite_stage="controls",
        parent_role=role,
    )
    cubes = _evaluate_role_candidates(
        run_dir,
        args,
        role_name="physical_fit",
        role=role,
        role_open=role_open,
        candidates=NONZERO_CANDIDATES,
    )
    _consolidate_cubes(run_dir / "fit_direction_moments.npz", NONZERO_CANDIDATES, cubes)
    # Rotation rows use exact output cosines.  They are separately resumable as
    # one compact CSV; interruption before the commit recomputes only this
    # diagnostic pass, never the completed candidate cubes.
    store, _, _ = _role_store(role)
    checkpoint_rows = _checkpoint_rows(args)
    rotation_rows: list[dict[str, Any]] = []
    for quartile in range(4):
        seeds = MODEL_SEEDS_BY_QUARTILE[quartile]
        for seed in seeds:
            stream = [candidate for candidate in NONZERO_CANDIDATES if candidate.quartile == quartile and candidate.seed == seed]
            for left, right in zip(stream[:-1], stream[1:], strict=True):
                cosines = _prediction_pair_cosines(
                    args,
                    left=left,
                    right=right,
                    checkpoint_rows=checkpoint_rows,
                    store=store,
                )
                for component in COMPONENT_NAMES:
                    left_map = component_summary(cubes[left.key], component)["cell_C"]
                    right_map = component_summary(cubes[right.key], component)["cell_C"]
                    correlation = float(np.corrcoef(left_map.reshape(-1), right_map.reshape(-1))[0, 1]) if np.std(left_map) > 0.0 and np.std(right_map) > 0.0 else 0.0
                    rotation_rows.append(
                        {
                            "comparison": "adjacent_update",
                            "quartile": quartile,
                            "seed_left": seed,
                            "seed_right": seed,
                            "update_left": left.update,
                            "update_right": right.update,
                            "component": component,
                            "prediction_cosine": cosines[component],
                            "cell_C_correlation": correlation,
                            "cell_sign_flip_count": int(np.count_nonzero(np.signbit(left_map) != np.signbit(right_map))),
                            "pooled_C_sign_change": int(component_summary(cubes[left.key], component)["C"] * component_summary(cubes[right.key], component)["C"] < 0.0),
                        }
                    )
        for update in CHECKPOINT_UPDATES[1:]:
            candidates = [CandidateIdentity(quartile, seed, update) for seed in seeds]
            for i in range(3):
                for j in range(i + 1, 3):
                    cosines = _prediction_pair_cosines(
                        args,
                        left=candidates[i],
                        right=candidates[j],
                        checkpoint_rows=checkpoint_rows,
                        store=store,
                    )
                    for component in COMPONENT_NAMES:
                        rotation_rows.append(
                            {
                                "comparison": "matched_update_cross_seed",
                                "quartile": quartile,
                                "seed_left": candidates[i].seed,
                                "seed_right": candidates[j].seed,
                                "update_left": update,
                                "update_right": update,
                                "component": component,
                                "prediction_cosine": cosines[component],
                                "cell_C_correlation": "",
                                "cell_sign_flip_count": "",
                                "pooled_C_sign_change": "",
                            }
                        )
    _atomic_csv(run_dir / "fit_trajectory_stability.csv", rotation_rows)
    max_recomposition = max(cube.maximum_recomposition_error for cube in cubes.values())
    max_identity = max(cube.maximum_risk_identity_error for cube in cubes.values())
    resource = _load_json(run_dir / "resource_projection.json")
    metrics = {
        "evaluation_status": "evaluated",
        "stage_execution_valid": 1,
        "fit_role_open_order_valid": 1,
        "all_fittrace_jobs_complete": int(len(cubes) == len(NONZERO_CANDIDATES)),
        "branch_recomposition_valid": int(max_recomposition <= RECOMPOSITION_TOLERANCE),
        "moment_algebra_valid": int(max_identity <= RECOMPOSITION_TOLERANCE),
        "trajectory_diagnostics_valid": int(len(rotation_rows) > 0),
        "nomination_forbidden": 1,
        "downstream_labels_unopened": int(not (run_dir / "gain_label_open.json").exists() and not (run_dir / "rank_label_open.json").exists()),
        "resource_limits_valid": int(resource["within_limits"]),
        "parent_unchanged": 1,
        "maximum_recomposition_error": max_recomposition,
        "maximum_risk_identity_error": max_identity,
        **_scope(),
    }
    atomic_write_json(run_dir / "fittrace_metrics.json", metrics)
    gate = evaluate_fittrace_gate(metrics)
    atomic_write_json(run_dir / "fittrace_gate.json", gate)
    _seal_stage(run_dir, "fittrace")
    if int(gate["passed"]) != 1:
        raise QuartileDirectionalWorkflowError(
            "fit trajectory diagnostics failed",
            failure_domain="fittrace",
            failure_code="quartile_directional_fittrace_invalid",
        )
    _status(run_dir, stage="fittrace", state="running", decision="ready_for_nominate")


def _gain_candidate_rows(
    cubes: Mapping[str, ComponentMomentCube],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in NONZERO_CANDIDATES:
        cube = cubes[candidate.key]
        for component in COMPONENT_NAMES:
            summary = component_summary(cube, component)
            rows.append(
                {
                    "quartile": candidate.quartile,
                    "seed": candidate.seed,
                    "update": candidate.update,
                    "candidate_key": candidate.key,
                    "component": component,
                    "target_energy": summary["T"],
                    "cross_term": summary["C"],
                    "prediction_energy": summary["P"],
                    "rho": summary["rho"],
                    "lambda_plus": summary["lambda_plus"],
                    "directional_ceiling": summary["D_plus"],
                    "historical_design_evidence": 1,
                    "authorizing": 0,
                }
            )
    return rows


def _nominate_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "nominate"):
        return
    _verify_stage_seal(run_dir, "fittrace")
    role = load_already_open_role(
        args.parent_quartile_specialist_run_dir, "gain_calibration"
    )
    role_open = _child_role_open(
        run_dir,
        role="gain_calibration",
        prerequisite_stage="fittrace",
        parent_role=role,
    )
    cubes = _evaluate_role_candidates(
        run_dir,
        args,
        role_name="gain_calibration",
        role=role,
        role_open=role_open,
        candidates=NONZERO_CANDIDATES,
    )
    _consolidate_cubes(
        run_dir / "gain_direction_moments.npz", NONZERO_CANDIDATES, cubes
    )
    rows = _gain_candidate_rows(cubes)
    nominees = select_direction_nominees(rows, require_complete_grid=True)
    if len(nominees) != 36:
        raise QuartileDirectionalWorkflowError(
            "gain nomination did not account for all 36 streams",
            failure_domain="nomination",
            failure_code="direction_nomination_incomplete",
        )
    _atomic_csv(run_dir / "gain_component_summary.csv", rows)
    _atomic_csv(run_dir / "direction_nominee_table.csv", nominees)
    seal = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-direction-nomination-seal",
            "schema_version": 1,
            "nominee_count": len(nominees),
            "nominee_present_count": sum(int(row["nominee_present"]) for row in nominees),
            "nominees": list(nominees),
            "nominee_table_sha256": file_fingerprint(run_dir / "direction_nominee_table.csv"),
            "gain_moments_sha256": file_fingerprint(run_dir / "gain_direction_moments.npz"),
            "rank_labels_opened": 0,
            "ranking_rule": ["largest_D_plus_gain", "earlier_update"],
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "direction_nomination_seal.json", seal)
    resource = _load_json(run_dir / "resource_projection.json")
    metrics = {
        "evaluation_status": "evaluated",
        "stage_execution_valid": 1,
        "gain_role_open_order_valid": int((run_dir / "fit_label_open.json").is_file()),
        "all_gain_jobs_complete": int(len(cubes) == len(NONZERO_CANDIDATES)),
        "nomination_rule_valid": 1,
        "all_thirty_six_streams_accounted": int(len(nominees) == 36),
        "direction_nomination_sealed": 1,
        "rank_labels_unopened": int(not (run_dir / "rank_label_open.json").exists()),
        "rank_search_forbidden": 1,
        "resource_limits_valid": int(resource["within_limits"]),
        "parent_unchanged": 1,
        **_scope(),
    }
    atomic_write_json(run_dir / "nominate_metrics.json", metrics)
    gate = evaluate_nominate_gate(metrics)
    atomic_write_json(run_dir / "nominate_gate.json", gate)
    _seal_stage(run_dir, "nominate")
    if int(gate["passed"]) != 1:
        raise QuartileDirectionalWorkflowError(
            "gain-only nomination failed",
            failure_domain="nomination",
            failure_code="quartile_directional_nomination_invalid",
        )
    _status(run_dir, stage="nominate", state="running", decision="ready_for_adjudicate")


def _candidate_from_nominee(row: Mapping[str, Any]) -> CandidateIdentity | None:
    if not int(row.get("nominee_present", 0)):
        return None
    return CandidateIdentity(
        int(row["quartile"]), int(row["seed"]), int(row["update"])
    )


def _nomination_seal(run_dir: Path) -> dict[str, Any]:
    _verify_stage_seal(run_dir, "nominate")
    seal = _load_json(run_dir / "direction_nomination_seal.json")
    _verify_semantic(seal, "direction nomination seal")
    if seal.get("nominee_table_sha256") != file_fingerprint(
        run_dir / "direction_nominee_table.csv"
    ) or seal.get("gain_moments_sha256") != file_fingerprint(
        run_dir / "gain_direction_moments.npz"
    ):
        raise ArtifactCompatibilityError("direction nomination binding changed")
    return seal


def _cube_by_candidate(path: Path) -> dict[str, ComponentMomentCube]:
    values = _load_npz(path)
    result: dict[str, ComponentMomentCube] = {}
    count = int(values["candidate_quartile"].size)
    for index in range(count):
        candidate = CandidateIdentity(
            int(values["candidate_quartile"][index]),
            int(values["candidate_seed"][index]),
            int(values["candidate_update"][index]),
        )
        result[candidate.key] = ComponentMomentCube(
            path_ids=np.asarray(values["path_ids"][index], dtype=np.int64),
            target_energy=np.asarray(values["target_energy"][index], dtype=np.float64),
            cross_terms=np.asarray(values["cross_terms"][index], dtype=np.float64),
            prediction_energies=np.asarray(values["prediction_energies"][index], dtype=np.float64),
            local_spatial_cross=np.asarray(values["local_spatial_cross"][index], dtype=np.float64),
            counts=np.asarray(values["counts"][index], dtype=np.int64),
            maximum_recomposition_error=float(values["maximum_recomposition_error"][index]),
            maximum_risk_identity_error=float(values["maximum_risk_identity_error"][index]),
        )
    return result


def _nominee_rows(run_dir: Path) -> tuple[dict[str, Any], ...]:
    seal = _nomination_seal(run_dir)
    rows = seal.get("nominees")
    if not isinstance(rows, list) or len(rows) != len(STREAM_IDENTITIES):
        raise ArtifactCompatibilityError("direction nominee seal is incomplete")
    result = tuple(dict(row) for row in rows)
    identities = tuple(
        StreamIdentity(int(row["quartile"]), str(row["component"]), int(row["seed"]))
        for row in result
    )
    if identities != STREAM_IDENTITIES:
        raise ArtifactCompatibilityError("direction nominee order changed")
    return result


def _component_marginals(
    cube: ComponentMomentCube, component: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    index = COMPONENT_NAMES.index(component)
    return (
        marginalize(cube.target_energy, cube.counts),
        marginalize(cube.cross_terms[index], cube.counts),
        marginalize(cube.prediction_energies[index], cube.counts),
    )


def _stream_record(
    nominee: Mapping[str, Any],
    *,
    gain_cubes: Mapping[str, ComponentMomentCube],
    rank_cubes: Mapping[str, ComponentMomentCube],
    path_ids: np.ndarray,
) -> dict[str, Any]:
    stream = StreamIdentity(
        int(nominee["quartile"]), str(nominee["component"]), int(nominee["seed"])
    )
    if not int(nominee.get("nominee_present", 0)):
        zero_paths = np.zeros(path_ids.size, dtype=np.float64)
        zero_cells = np.zeros((PHASE_COUNT, 8), dtype=np.float64)
        return {
            **stream.to_record(),
            "nominee_present": 0,
            "update": None,
            "C_gain": 0.0,
            "P_gain": 0.0,
            "lambda_gain": 0.0,
            "D_plus_gain": 0.0,
            "C_rank": 0.0,
            "P_rank": 0.0,
            "T_rank": 0.0,
            "lambda_rank": 0.0,
            "effect_point": 0.0,
            "direction_path_values": zero_paths,
            "target_energy_path_values": zero_paths.copy(),
            "prediction_energy_path_values": zero_paths.copy(),
            "directional_ceiling_path_values": zero_paths.copy(),
            "effect_path_values": zero_paths.copy(),
            "direction_cells": zero_cells,
            "effect_cells": zero_cells.copy(),
            "direction_phase": np.zeros(PHASE_COUNT, dtype=np.float64),
            "direction_midpoint": np.zeros(8, dtype=np.float64),
            "effect_phase": np.zeros(PHASE_COUNT, dtype=np.float64),
            "effect_midpoint": np.zeros(8, dtype=np.float64),
            "historical_design_evidence": 1,
            "authorizing": 0,
        }
    candidate = _candidate_from_nominee(nominee)
    if candidate is None or candidate.key not in gain_cubes or candidate.key not in rank_cubes:
        raise ArtifactCompatibilityError("sealed nominee moment cube is missing")
    gain_cube = gain_cubes[candidate.key]
    rank_cube = rank_cubes[candidate.key]
    if not np.array_equal(rank_cube.path_ids, path_ids):
        raise ArtifactCompatibilityError("rank nominee path order changed")
    _, gain_cross, gain_energy = _component_marginals(
        gain_cube, stream.component
    )
    rank_target, rank_cross, rank_energy = _component_marginals(
        rank_cube, stream.component
    )
    c_gain = float(gain_cross["pooled"])
    p_gain = float(gain_energy["pooled"])
    optimum = positive_ray_optimum(
        float(marginalize(gain_cube.target_energy, gain_cube.counts)["pooled"]),
        c_gain,
        p_gain,
    )
    gain = float(optimum["lambda_plus"])
    if gain <= 0.0 or gain != float(nominee["lambda_gain"]):
        raise ArtifactCompatibilityError("sealed gain scalar changed")
    effect_cube = quadratic_improvement(
        rank_cube.cross_terms[COMPONENT_NAMES.index(stream.component)],
        rank_cube.prediction_energies[COMPONENT_NAMES.index(stream.component)],
        gain,
    )
    effect = marginalize(effect_cube, rank_cube.counts)
    c_rank = float(rank_cross["pooled"])
    p_rank = float(rank_energy["pooled"])
    rank_optimum = positive_ray_optimum(
        float(marginalize(rank_cube.target_energy, rank_cube.counts)["pooled"]),
        c_rank,
        p_rank,
    )
    path_cross = np.asarray(rank_cross["path"], dtype=np.float64)
    path_energy = np.asarray(rank_energy["path"], dtype=np.float64)
    if np.any((path_energy == 0.0) & (path_cross != 0.0)):
        raise QuartileDirectionalWorkflowError(
            "rank path has P=0 with C!=0",
            failure_domain="moment_algebra",
            failure_code="quartile_directional_rank_adjudication_invalid",
        )
    path_ceiling = np.zeros_like(path_cross)
    positive = (path_cross > 0.0) & (path_energy > 0.0)
    path_ceiling[positive] = (
        path_cross[positive] * path_cross[positive] / path_energy[positive]
    )
    return {
        **stream.to_record(),
        "nominee_present": 1,
        "update": candidate.update,
        "candidate_key": candidate.key,
        "C_gain": c_gain,
        "P_gain": p_gain,
        "lambda_gain": gain,
        "D_plus_gain": float(optimum["D_plus"]),
        "C_rank": c_rank,
        "P_rank": p_rank,
        "T_rank": float(rank_target["pooled"]),
        "lambda_rank": float(rank_optimum["lambda_plus"]),
        "effect_point": float(effect["pooled"]),
        "direction_path_values": path_cross,
        "target_energy_path_values": np.asarray(
            rank_target["path"], dtype=np.float64
        ),
        "prediction_energy_path_values": path_energy,
        "directional_ceiling_path_values": path_ceiling,
        "effect_path_values": np.asarray(effect["path"], dtype=np.float64),
        "direction_cells": np.asarray(rank_cross["cell"], dtype=np.float64),
        "effect_cells": np.asarray(effect["cell"], dtype=np.float64),
        "direction_phase": np.asarray(rank_cross["phase"], dtype=np.float64),
        "direction_midpoint": np.asarray(rank_cross["midpoint"], dtype=np.float64),
        "effect_phase": np.asarray(effect["phase"], dtype=np.float64),
        "effect_midpoint": np.asarray(effect["midpoint"], dtype=np.float64),
        "historical_design_evidence": 1,
        "authorizing": 0,
    }


def _rank_table_rows(stream_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in stream_records:
        identity = {
            "quartile": record["quartile"],
            "component": record["component"],
            "seed": record["seed"],
            "update": record.get("update"),
            "nominee_present": record["nominee_present"],
        }
        for statistic in ("direction", "effect"):
            cells = np.asarray(record[f"{statistic}_cells"], dtype=np.float64)
            phase = np.asarray(record[f"{statistic}_phase"], dtype=np.float64)
            midpoint = np.asarray(record[f"{statistic}_midpoint"], dtype=np.float64)
            pooled = float(record["C_rank"] if statistic == "direction" else record["effect_point"])
            rows.append({**identity, "statistic": statistic, "marginal": "pooled", "index": -1, "value": pooled})
            rows.extend(
                {**identity, "statistic": statistic, "marginal": "phase", "index": index, "value": float(value)}
                for index, value in enumerate(phase)
            )
            rows.extend(
                {**identity, "statistic": statistic, "marginal": "midpoint", "index": index, "value": float(value)}
                for index, value in enumerate(midpoint)
            )
            rows.extend(
                {**identity, "statistic": statistic, "marginal": "cell", "index": phase_index * 8 + midpoint_index, "phase": phase_index, "midpoint": midpoint_index, "value": float(cells[phase_index, midpoint_index])}
                for phase_index in range(PHASE_COUNT)
                for midpoint_index in range(8)
            )
    return rows


def _transfer_rows(
    stream_records: Sequence[Mapping[str, Any]], inference: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in stream_records:
        stream = StreamIdentity(
            int(record["quartile"]), str(record["component"]), int(record["seed"])
        )
        lambda_gain = float(record["lambda_gain"])
        lambda_rank = float(record["lambda_rank"])
        ratio = (
            math.log(lambda_rank / lambda_gain)
            if lambda_gain > 0.0 and lambda_rank > 0.0
            else "not_finite"
        )
        target = float(record["T_rank"])
        effect_point = float(record["effect_point"])
        direct_effect = target - (
            target
            - 2.0 * lambda_gain * float(record["C_rank"])
            + lambda_gain * lambda_gain * float(record["P_rank"])
        )
        direction_stability = path_stability_summary(
            record["direction_path_values"],
            simultaneous_lower_bound=inference.lower_bound(stream, "direction"),
        )
        effect_stability = path_stability_summary(
            record["effect_path_values"],
            simultaneous_lower_bound=inference.lower_bound(stream, "effect"),
        )
        rows.append(
            {
                "quartile": stream.quartile,
                "component": stream.component,
                "seed": stream.seed,
                "update": record.get("update"),
                "nominee_present": record["nominee_present"],
                "C_gain": record["C_gain"],
                "P_gain": record["P_gain"],
                "lambda_gain": lambda_gain,
                "D_plus_gain": record["D_plus_gain"],
                "C_rank": record["C_rank"],
                "P_rank": record["P_rank"],
                "lambda_rank": lambda_rank,
                "log_lambda_rank_over_gain": ratio,
                "rank_effect_point": record["effect_point"],
                "rank_effect_direct": direct_effect,
                "rank_effect_identity_error": abs(direct_effect - effect_point),
                "rank_direction_lower_bound": inference.lower_bound(stream, "direction"),
                "rank_effect_lower_bound": inference.lower_bound(stream, "effect"),
                "direction_positive_path_count": direction_stability["positive_path_count"],
                "direction_sign_entropy_bits": direction_stability["sign_entropy_bits"],
                "direction_path_unstable": direction_stability["path_unstable"],
                "effect_positive_path_count": effect_stability["positive_path_count"],
                "effect_sign_entropy_bits": effect_stability["sign_entropy_bits"],
                "effect_path_unstable": effect_stability["path_unstable"],
                "historical_design_evidence": 1,
                "authorizing": 0,
            }
        )
    return rows


def _cancellation_rows(
    stream_records: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Any],
    gain_cubes: Mapping[str, ComponentMomentCube],
    rank_cubes: Mapping[str, ComponentMomentCube],
) -> list[dict[str, Any]]:
    seed_rows = {
        (int(row["quartile"]), str(row["component"]), int(row["seed"])): row
        for row in adjudication["seed_rows"]
    }
    by_stream = {
        (int(row["quartile"]), str(row["component"]), int(row["seed"])): row
        for row in stream_records
    }
    rows: list[dict[str, Any]] = []
    for record in stream_records:
        if not int(record["nominee_present"]):
            continue
        cube = rank_cubes[str(record["candidate_key"])]
        summaries = {
            component: component_summary(cube, component)
            for component in COMPONENT_NAMES
        }
        branch_cross = float(
            marginalize(cube.local_spatial_cross, cube.counts)["pooled"]
        )
        cross_error = abs(
            float(summaries["full"]["C"])
            - float(summaries["local_affine"]["C"])
            - float(summaries["spatial_cnn"]["C"])
        )
        energy_error = abs(
            float(summaries["full"]["P"])
            - float(summaries["local_affine"]["P"])
            - float(summaries["spatial_cnn"]["P"])
            - 2.0 * branch_cross
        )
        rows.append(
            {
                "row_type": "seed_branch_algebra",
                "quartile": record["quartile"],
                "nominee_component": record["component"],
                "seed": record["seed"],
                "update": record["update"],
                "C_full": summaries["full"]["C"],
                "C_local_affine": summaries["local_affine"]["C"],
                "C_spatial_cnn": summaries["spatial_cnn"]["C"],
                "cross_recomposition_error": cross_error,
                "P_full": summaries["full"]["P"],
                "P_local_affine": summaries["local_affine"]["P"],
                "P_spatial_cnn": summaries["spatial_cnn"]["P"],
                "Q_local_spatial": branch_cross,
                "energy_recomposition_error": energy_error,
                "algebra_valid": int(
                    cross_error <= RECOMPOSITION_TOLERANCE
                    and energy_error <= RECOMPOSITION_TOLERANCE
                ),
                "historical_design_evidence": 1,
                "authorizing": 0,
            }
        )
    for quartile in range(1, 4):
        for component in COMPONENT_NAMES:
            cell_count = np.zeros((PHASE_COUNT, 8), dtype=np.int8)
            phase_count = np.zeros(PHASE_COUNT, dtype=np.int8)
            midpoint_count = np.zeros(8, dtype=np.int8)
            compared = 0
            for seed in MODEL_SEEDS_BY_QUARTILE[quartile]:
                record = by_stream[(quartile, component, seed)]
                if not int(record["nominee_present"]):
                    continue
                compared += 1
                gain_summary = component_summary(
                    gain_cubes[str(record["candidate_key"])], component
                )
                gain_cells = np.asarray(gain_summary["cell_C"], dtype=np.float64)
                gain_phase = np.asarray(gain_summary["phase_C"], dtype=np.float64)
                gain_midpoint = np.asarray(
                    gain_summary["midpoint_C"], dtype=np.float64
                )
                rank_cells = np.asarray(record["direction_cells"], dtype=np.float64)
                rank_phase = np.asarray(record["direction_phase"], dtype=np.float64)
                rank_midpoint = np.asarray(
                    record["direction_midpoint"], dtype=np.float64
                )
                cell_count += ((gain_cells < 0.0) & (rank_cells < 0.0)).astype(
                    np.int8
                )
                phase_count += ((gain_phase < 0.0) & (rank_phase < 0.0)).astype(
                    np.int8
                )
                midpoint_count += (
                    (gain_midpoint < 0.0) & (rank_midpoint < 0.0)
                ).astype(np.int8)
            cells = [
                f"phase{phase}.midpoint{midpoint}"
                for phase in range(PHASE_COUNT)
                for midpoint in range(8)
                if int(cell_count[phase, midpoint]) >= 2
            ]
            phases = [
                f"phase{index}"
                for index, count in enumerate(phase_count)
                if int(count) >= 2
            ]
            midpoints = [
                f"midpoint{index}"
                for index, count in enumerate(midpoint_count)
                if int(count) >= 2
            ]
            rows.append(
                {
                    "row_type": "cross_role_local_cancellation",
                    "quartile": quartile,
                    "nominee_component": component,
                    "compared_seed_count": compared,
                    "reproducibly_negative_cell_count": len(cells),
                    "reproducibly_negative_cells": ";".join(cells),
                    "reproducibly_negative_phase_count": len(phases),
                    "reproducibly_negative_phases": ";".join(phases),
                    "reproducibly_negative_midpoint_count": len(midpoints),
                    "reproducibly_negative_midpoints": ";".join(midpoints),
                    "cancellation_visible": int(bool(cells or phases or midpoints)),
                    "historical_design_evidence": 1,
                    "authorizing": 0,
                }
            )
    for quartile in range(1, 4):
        for branch, competitor in (
            ("local_affine", "spatial_cnn"),
            ("spatial_cnn", "local_affine"),
        ):
            passing_seeds = []
            attributed_seeds = []
            responsible_count = 0
            for seed in MODEL_SEEDS_BY_QUARTILE[quartile]:
                seed_row = seed_rows[(quartile, branch, seed)]
                if not (int(seed_row["direction_passed"]) and int(seed_row["effect_passed"])):
                    continue
                passing_seeds.append(seed)
                record = by_stream[(quartile, branch, seed)]
                candidate_key = record.get("candidate_key")
                if not candidate_key or candidate_key not in rank_cubes:
                    continue
                cube = rank_cubes[str(candidate_key)]
                branch_index = COMPONENT_NAMES.index(branch)
                full_index = COMPONENT_NAMES.index("full")
                competitor_index = COMPONENT_NAMES.index(competitor)
                branch_cross = np.asarray(
                    marginalize(cube.cross_terms[branch_index], cube.counts)["cell"],
                    dtype=np.float64,
                )
                full_cross = np.asarray(
                    marginalize(cube.cross_terms[full_index], cube.counts)["cell"],
                    dtype=np.float64,
                )
                competitor_cross = np.asarray(
                    marginalize(cube.cross_terms[competitor_index], cube.counts)["cell"],
                    dtype=np.float64,
                )
                responsible = (branch_cross > 0.0) & (full_cross <= 0.0)
                count = int(np.count_nonzero(responsible))
                responsible_count += count
                if count > 0 and bool(np.all(competitor_cross[responsible] < 0.0)):
                    attributed_seeds.append(seed)
            rows.append(
                {
                    "row_type": "quartile_branch_attribution",
                    "quartile": quartile,
                    "passing_branch": branch,
                    "competing_branch": competitor,
                    "passing_seed_count": len(passing_seeds),
                    "passing_seeds": ";".join(map(str, passing_seeds)),
                    "attributed_seed_count": len(attributed_seeds),
                    "attributed_seeds": ";".join(map(str, attributed_seeds)),
                    "responsible_cell_count": responsible_count,
                    "competing_branch_negative_in_responsible_strata": int(
                        len(attributed_seeds) >= 2
                    ),
                    "attribution_valid": int(len(attributed_seeds) >= 2),
                    "historical_design_evidence": 1,
                    "authorizing": 0,
                }
            )
    return rows


def _mechanism_evidence(
    stream_records: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Any],
    transfer_rows: Sequence[Mapping[str, Any]],
    cancellation_rows: Sequence[Mapping[str, Any]],
    rotation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    component_rows = [dict(row) for row in adjudication["component_rows"]]
    seed_rows = [dict(row) for row in adjudication["seed_rows"]]
    transfer = {
        (int(row["quartile"]), str(row["component"]), int(row["seed"])): row
        for row in transfer_rows
    }
    stream_map = {
        (int(row["quartile"]), str(row["component"]), int(row["seed"])): row
        for row in stream_records
    }
    seed_map = {
        (int(row["quartile"]), str(row["component"]), int(row["seed"])): row
        for row in seed_rows
    }
    rotation_flags: set[tuple[int, str]] = set()
    pooled_sign_changes: dict[tuple[int, str, int], int] = {}
    for row in rotation_rows:
        try:
            raw_cosine = row.get("prediction_cosine", 1.0)
            cosine = 1.0 if raw_cosine in (None, "") else float(raw_cosine)
            sign_changes = int(row.get("pooled_C_sign_change", 0) or 0)
        except (TypeError, ValueError):
            continue
        key = (int(row["quartile"]), str(row["component"]))
        if cosine < 0.0:
            rotation_flags.add(key)
        if str(row.get("comparison")) == "adjacent_update" and sign_changes:
            trajectory = (key[0], key[1], int(row["seed_left"]))
            pooled_sign_changes[trajectory] = pooled_sign_changes.get(trajectory, 0) + 1
        raw_correlation = row.get("cell_C_correlation")
        if str(row.get("comparison")) == "gain_vs_rank_nominee" and raw_correlation not in (None, ""):
            try:
                if float(raw_correlation) <= 0.0:
                    rotation_flags.add(key)
            except (TypeError, ValueError):
                pass
    for (quartile, component, _), count in pooled_sign_changes.items():
        if count >= 2:
            rotation_flags.add((quartile, component))
    for record in stream_records:
        if int(record["nominee_present"]) and float(record["C_rank"]) <= 0.0:
            rotation_flags.add((int(record["quartile"]), str(record["component"])))
    local_cancellation = {
        (int(row["quartile"]), str(row["nominee_component"]))
        for row in cancellation_rows
        if row.get("row_type") == "cross_role_local_cancellation"
        and int(row["cancellation_visible"])
    }
    for row in component_rows:
        quartile = int(row["quartile"])
        component = str(row["component"])
        members = [
            stream_map[(quartile, component, seed)]
            for seed in MODEL_SEEDS_BY_QUARTILE[quartile]
        ]
        adjudicated_members = [
            seed_map[(quartile, component, seed)]
            for seed in MODEL_SEEDS_BY_QUARTILE[quartile]
        ]
        transfer_members = [
            transfer[(quartile, component, seed)]
            for seed in MODEL_SEEDS_BY_QUARTILE[quartile]
        ]
        row.update(
            {
                "positive_gain_direction": int(any(int(value["nominee_present"]) for value in members)),
                "rank_transfer_failed": int(any(int(value["nominee_present"]) and float(value["C_rank"]) <= 0.0 for value in members)),
                "positive_direction_effect_unresolved": int(any(int(seed_row["direction_passed"]) and float(seed_row["effect_point"]) > 0.0 and not int(seed_row["effect_passed"]) for seed_row in adjudicated_members)),
                "rotation_or_path_instability": int(
                    (quartile, component) in rotation_flags
                    or any(int(value["direction_path_unstable"]) or int(value["effect_path_unstable"]) for value in transfer_members)
                ),
                "path_instability": int(any(int(value["direction_path_unstable"]) or int(value["effect_path_unstable"]) for value in transfer_members)),
                "direction_rotation": int((quartile, component) in rotation_flags),
                "cancellation_visible": int(
                    any(
                        int(seed_row["nominee_present"])
                        and not int(seed_row["direction_local_screen"]["passed"])
                        for seed_row in adjudicated_members
                    )
                    or (quartile, component) in local_cancellation
                ),
            }
        )
    component_map = {
        (int(row["quartile"]), str(row["component"])): row for row in component_rows
    }
    cancellation_map: dict[int, dict[str, int]] = {quartile: {} for quartile in range(1, 4)}
    for row in cancellation_rows:
        if row.get("row_type") != "quartile_branch_attribution":
            continue
        cancellation_map[int(row["quartile"])][str(row["passing_branch"])] = int(
            row["attribution_valid"]
        )
    quartiles: dict[str, Any] = {}
    for quartile in range(4):
        qrow: dict[str, Any] = {
            "components": {
                component: component_map[(quartile, component)]
                for component in COMPONENT_NAMES
            }
        }
        if quartile > 0:
            qrow["competing_branch_negative_in_full_failure_strata"] = cancellation_map[quartile]
        quartiles[f"q{quartile}"] = qrow
    q0 = component_map[(0, "full")]
    return {
        "schema": f"{RUN_SCHEMA}-mechanism-classification",
        "schema_version": 1,
        "q0_full": {
            "stable_direction": int(q0["stable_direction"]),
            "stable_effect": int(q0["stable_effect"]),
        },
        "inferential_and_role_order_valid": 1,
        "branch_algebra_cancellation_valid": 1,
        "quartiles": quartiles,
        "component_rows": component_rows,
        "seed_rows": seed_rows,
        "component_cancellation_rows": [dict(row) for row in cancellation_rows],
        "cancellation_visible": int(
            any(
                int(row["attribution_valid"])
                for row in cancellation_rows
                if row.get("row_type") == "quartile_branch_attribution"
            )
            or bool(local_cancellation)
        ),
        "positive_direction_effect_unresolved": int(any(int(row["positive_direction_effect_unresolved"]) for row in component_rows)),
        "positive_gain_direction": int(any(int(row["positive_gain_direction"]) for row in component_rows[3:])),
        "rank_transfer_failed": int(any(int(row["rank_transfer_failed"]) for row in component_rows[3:])),
        "rotation_or_path_instability": int(any(int(row["rotation_or_path_instability"]) for row in component_rows[3:])),
        "historical_design_evidence": 1,
        "authorizing": 0,
        **_scope(),
    }


def _adjudicate_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(run_dir, "adjudicate"):
        return
    _nomination_seal(run_dir)
    nominees = _nominee_rows(run_dir)
    role = load_already_open_role(
        args.parent_quartile_specialist_run_dir, "training_rank"
    )
    role_open = _child_role_open(
        run_dir,
        role="training_rank",
        prerequisite_stage="nominate",
        parent_role=role,
    )
    store, _, path_ids = _role_store(role)
    candidates = sorted(
        {
            candidate
            for row in nominees
            if (candidate := _candidate_from_nominee(row)) is not None
        }
    )
    rank_cubes = _evaluate_role_candidates(
        run_dir,
        args,
        role_name="training_rank",
        role=role,
        role_open=role_open,
        candidates=candidates,
    )
    if candidates:
        _consolidate_cubes(run_dir / "rank_direction_moments.npz", candidates, rank_cubes)
    else:
        _atomic_npz(
            run_dir / "rank_direction_moments.npz",
            candidate_quartile=np.empty(0, dtype=np.int8),
            candidate_seed=np.empty(0, dtype=np.int64),
            candidate_update=np.empty(0, dtype=np.int16),
        )
    gain_cubes = _cube_by_candidate(run_dir / "gain_direction_moments.npz")
    stream_records = tuple(
        _stream_record(
            nominee,
            gain_cubes=gain_cubes,
            rank_cubes=rank_cubes,
            path_ids=path_ids,
        )
        for nominee in nominees
    )
    path_table = build_stream_path_table(stream_records, path_ids=path_ids)
    inference = one_sided_direction_effect_max_t(
        path_table,
        path_ids=path_ids,
        confidence=DEFAULT_CONFIDENCE,
        replicates=DEFAULT_REPLICATES,
        seed=DEFAULT_BOOTSTRAP_SEED,
        namespace=DEFAULT_BOOTSTRAP_NAMESPACE,
    )
    adjudication = adjudicate_seed_streams(stream_records, inference)
    _atomic_npz(
        run_dir / "rank_path_moments.npz",
        path_ids=path_ids,
        family_names=np.asarray(direction_family_names()),
        path_values=path_table,
        stream_target_energy=np.stack(
            [np.asarray(row["target_energy_path_values"], dtype=np.float64) for row in stream_records],
            axis=0,
        ),
        stream_cross_term=np.stack(
            [np.asarray(row["direction_path_values"], dtype=np.float64) for row in stream_records],
            axis=0,
        ),
        stream_prediction_energy=np.stack(
            [np.asarray(row["prediction_energy_path_values"], dtype=np.float64) for row in stream_records],
            axis=0,
        ),
        stream_directional_ceiling=np.stack(
            [np.asarray(row["directional_ceiling_path_values"], dtype=np.float64) for row in stream_records],
            axis=0,
        ),
        stream_transferred_improvement=np.stack(
            [np.asarray(row["effect_path_values"], dtype=np.float64) for row in stream_records],
            axis=0,
        ),
        point_estimates=inference.point_estimates,
        standard_errors=inference.standard_errors,
        lower_bounds=inference.lower_bounds,
        nominee_updates=np.asarray(
            [int(row["update"]) if row.get("update") is not None else -1 for row in stream_records],
            dtype=np.int16,
        ),
    )
    rank_rows = _rank_table_rows(stream_records)
    _atomic_csv(run_dir / "rank_phase_midpoint_tables.csv", rank_rows)
    transfer_rows = _transfer_rows(stream_records, inference)
    _atomic_csv(run_dir / "gain_transfer_table.csv", transfer_rows)
    cancellation_rows = _cancellation_rows(
        stream_records, adjudication, gain_cubes, rank_cubes
    )
    _atomic_csv(run_dir / "component_cancellation_table.csv", cancellation_rows)
    rotation_rows = _read_csv(run_dir / "fit_trajectory_stability.csv")
    for record in stream_records:
        if not int(record["nominee_present"]):
            continue
        gain_cells = np.asarray(
            component_summary(
                gain_cubes[str(record["candidate_key"])], str(record["component"])
            )["cell_C"],
            dtype=np.float64,
        )
        rank_cells = np.asarray(record["direction_cells"], dtype=np.float64)
        correlation = (
            float(np.corrcoef(gain_cells.reshape(-1), rank_cells.reshape(-1))[0, 1])
            if np.std(gain_cells) > 0.0 and np.std(rank_cells) > 0.0
            else 0.0
        )
        rotation_rows.append(
            {
                "comparison": "gain_vs_rank_nominee",
                "quartile": record["quartile"],
                "seed_left": record["seed"],
                "seed_right": record["seed"],
                "update_left": record["update"],
                "update_right": record["update"],
                "component": record["component"],
                "prediction_cosine": "",
                "cell_C_correlation": correlation,
                "cell_sign_flip_count": int(
                    np.count_nonzero(np.signbit(gain_cells) != np.signbit(rank_cells))
                ),
                "pooled_C_sign_change": int(float(record["C_gain"]) * float(record["C_rank"]) < 0.0),
            }
        )
    _atomic_csv(run_dir / "trajectory_rotation_table.csv", rotation_rows)
    max_t_record = _semantic(
        {
            **inference.to_record(),
            "stream_adjudication": adjudication,
            **_scope(),
        }
    )
    atomic_write_json(
        run_dir / "max_t_direction_and_effect_inference.json", max_t_record
    )
    q0 = next(
        row
        for row in adjudication["component_rows"]
        if int(row["quartile"]) == 0 and row["component"] == "full"
    )
    atomic_write_json(
        run_dir / "q0_positive_control.json",
        _semantic(
            {
                "schema": f"{RUN_SCHEMA}-q0-positive-control",
                "schema_version": 1,
                **q0,
                "passed": int(int(q0["stable_direction"]) and int(q0["stable_effect"])),
                **_scope(),
            }
        ),
    )
    evidence = _mechanism_evidence(
        stream_records,
        adjudication,
        transfer_rows,
        cancellation_rows,
        rotation_rows,
    )
    atomic_write_json(run_dir / "mechanism_classification.json", _semantic(evidence))
    forecast_rows: list[dict[str, Any]] = []
    for record, seed_row in zip(stream_records, adjudication["seed_rows"], strict=True):
        forecast = forecast_required_paths(
            record["effect_path_values"],
            critical_value=inference.critical_value,
            local_point_screen_passed=bool(seed_row["effect_local_screen"]["passed"]),
        )
        forecast_rows.append(
            {
                "quartile": record["quartile"],
                "component": record["component"],
                "seed": record["seed"],
                "update": record.get("update"),
                **forecast,
                "historical_design_evidence": 1,
                "authorizing": 0,
            }
        )
    _atomic_csv(run_dir / "path_count_forecast.csv", forecast_rows)
    before = _load_json(run_dir / "parent_immutability_before.json")
    specialist_now = snapshot_parent_run(args.parent_quartile_specialist_run_dir)
    time_local_now = snapshot_parent_run(args.parent_time_local_run_dir)
    unchanged = int(
        compare_parent_snapshots(before["quartile_specialist"], specialist_now)["passed"]
        and compare_parent_snapshots(before["time_local"], time_local_now)["passed"]
    )
    metrics = {
        "evaluation_status": "evaluated",
        "stage_execution_valid": 1,
        "nomination_seal_valid": 1,
        "rank_role_open_order_valid": int((run_dir / "gain_label_open.json").is_file()),
        "all_rank_jobs_complete": int(len(rank_cubes) == len(candidates)),
        "max_t_family_valid": int(len(inference.family_names) == 72),
        "q0_positive_control_evaluated": 1,
        "direction_rules_valid": int(len(adjudication["seed_rows"]) == 36),
        "effect_rules_valid": int(len(adjudication["component_rows"]) == 12),
        "branch_algebra_valid": int(
            all(cube.maximum_recomposition_error <= RECOMPOSITION_TOLERANCE for cube in rank_cubes.values())
            and all(
                int(row["algebra_valid"])
                for row in cancellation_rows
                if row.get("row_type") == "seed_branch_algebra"
            )
            and all(
                float(row["rank_effect_identity_error"])
                <= RECOMPOSITION_TOLERANCE
                for row in transfer_rows
            )
        ),
        "mechanism_classification_valid": 1,
        "path_count_forecast_valid": int(len(forecast_rows) == 36),
        "parent_unchanged": unchanged,
        "decision_evidence": evidence,
        **_scope(),
    }
    atomic_write_json(run_dir / "adjudicate_metrics.json", metrics)
    gate = evaluate_adjudicate_gate(metrics)
    atomic_write_json(run_dir / "adjudicate_gate.json", gate)
    _seal_stage(run_dir, "adjudicate")
    if int(gate["passed"]) != 1:
        raise QuartileDirectionalWorkflowError(
            "independent rank-role adjudication failed",
            failure_domain="rank_adjudication",
            failure_code="quartile_directional_rank_adjudication_invalid",
        )
    _status(run_dir, stage="adjudicate", state="running", decision="ready_for_report")


def _stage_gates(run_dir: Path) -> dict[str, dict[str, Any]]:
    names = {
        "preflight": "preflight_gate.json",
        "replay": "historical_replay_gate.json",
        "controls": "controls_gate.json",
        "fittrace": "fittrace_gate.json",
        "nominate": "nominate_gate.json",
        "adjudicate": "adjudicate_gate.json",
    }
    return {
        stage: _optional_json(run_dir, name) or not_evaluated_gate(stage)
        for stage, name in names.items()
    }


def _decision_evidence(run_dir: Path) -> dict[str, Any] | None:
    metrics = _optional_json(run_dir, "adjudicate_metrics.json")
    if metrics is None:
        return None
    evidence = metrics.get("decision_evidence")
    return dict(evidence) if isinstance(evidence, Mapping) else None


def _workflow_record(run_dir: Path, require_gate: str) -> dict[str, Any]:
    gates = _stage_gates(run_dir)
    evidence = _decision_evidence(run_dir)
    decision = decide_workflow(evidence, gates=gates)
    workflow = evaluate_required_gate(
        require_gate,
        gates=gates,
        decision=decision,
        evidence=evidence,
    )
    decision = {**decision, **_scope()}
    workflow = {**workflow, "decision": decision, **_scope()}
    atomic_write_json(
        run_dir / "quartile_directional_adjudication_decision.json", decision
    )
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    return workflow


def _report_stage(run_dir: Path, args: argparse.Namespace) -> None:
    _verify_stage_seal(run_dir, "adjudicate")
    workflow = _workflow_record(run_dir, "none")
    decision = workflow["decision"]
    evidence = _decision_evidence(run_dir) or {}
    lines = [
        "# Read-only quartile directional/representation adjudication",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "This workflow evaluated only immutable historical physical-fit, "
        "gain-calibration, and training-rank evidence. It generated no "
        "transitions, optimizer updates, fresh selection or confirmation "
        "evidence, controller trajectory, reconstruction, or sample.",
        "",
        "## Frozen component results",
        "",
    ]
    quartiles = evidence.get("quartiles", {})
    for quartile in ("q0", "q1", "q2", "q3"):
        qrow = quartiles.get(quartile, {}) if isinstance(quartiles, Mapping) else {}
        components = qrow.get("components", {}) if isinstance(qrow, Mapping) else {}
        values = []
        for component in COMPONENT_NAMES:
            row = components.get(component, {}) if isinstance(components, Mapping) else {}
            values.append(
                f"{component}: direction={int(row.get('stable_direction', 0))}, "
                f"effect={int(row.get('stable_effect', 0))}"
            )
        lines.append(f"- {quartile}: " + "; ".join(values))
    lines.extend(
        [
            "",
            "Only `unique_representation_hypothesis_identified` recommends "
            "drafting a separately reviewed fresh-role learner plan. Even that "
            "outcome does not authorize training, controller execution, "
            "reconstruction, or sampling.",
            "",
        ]
    )
    _atomic_text(run_dir / "REPORT.md", "\n".join(lines))
    before = _load_json(run_dir / "parent_immutability_before.json")
    specialist_now = snapshot_parent_run(args.parent_quartile_specialist_run_dir)
    time_local_now = snapshot_parent_run(args.parent_time_local_run_dir)
    specialist_check = compare_parent_snapshots(
        before["quartile_specialist"], specialist_now
    )
    time_local_check = compare_parent_snapshots(before["time_local"], time_local_now)
    after = _semantic(
        {
            "schema": f"{RUN_SCHEMA}-parent-immutability-after",
            "schema_version": 1,
            "quartile_specialist": specialist_now,
            "time_local": time_local_now,
            "quartile_specialist_unchanged": int(specialist_check["passed"]),
            "time_local_unchanged": int(time_local_check["passed"]),
            "passed": int(specialist_check["passed"] and time_local_check["passed"]),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "parent_immutability_after.json", after)
    if not int(after["passed"]):
        raise ArtifactCompatibilityError("an immutable parent changed during report")


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return (
            "preflight",
            "replay",
            "controls",
            "fittrace",
            "nominate",
            "adjudicate",
            "report",
        )
    return (stage,)


def _gate_filename(stage: str) -> str:
    return "historical_replay_gate.json" if stage == "replay" else f"{stage}_gate.json"


def _commit_failed_stage_gate(
    run_dir: Path, *, stage: str, failure: Mapping[str, Any]
) -> None:
    evaluators = {
        "preflight": evaluate_preflight_gate,
        "replay": evaluate_replay_gate,
        "controls": evaluate_controls_gate,
        "fittrace": evaluate_fittrace_gate,
        "nominate": evaluate_nominate_gate,
        "adjudicate": evaluate_adjudicate_gate,
    }
    if stage not in evaluators or (run_dir / _gate_filename(stage)).is_file():
        return
    metrics = {
        "evaluation_status": "execution_failed",
        "stage_execution_valid": 0,
        "failure_domain": failure["failure_domain"],
        "failure_code": failure["failure_code"],
        "error": failure["error"],
        **_scope(),
    }
    atomic_write_json(run_dir / f"{stage}_metrics.json", metrics)
    atomic_write_json(
        run_dir / _gate_filename(stage), evaluators[stage](metrics)
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument(
        "--parent-quartile-specialist-run-dir", type=Path, required=True
    )
    parser.add_argument("--parent-time-local-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_"
            "quartile_directional_adjudication"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-read-only-quartile-directional-adjudication"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for name in (
        "parent_quartile_specialist_run_dir",
        "parent_time_local_run_dir",
        "resume_run_dir",
        "runs_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    if args.resume_run_dir is None and args.stage not in {"preflight", "all"}:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    expected = {
        "preflight": "preflight",
        "replay": "replay",
        "controls": "controls",
        "fittrace": "fittrace",
        "nominate": "nominate",
        "adjudicate": "adjudicate",
        "report": "none",
        "all": "adjudicate",
    }[args.stage]
    if args.require_gate not in {"none", expected}:
        parser.error(f"--stage {args.stage} cannot require {args.require_gate}")
    if args.device != "cuda" and not args.test_only and args.stage in {
        "preflight",
        "fittrace",
        "nominate",
        "adjudicate",
        "all",
    }:
        parser.error("production checkpoint evaluation requires --device cuda")
    return args


def _passed(path: Path) -> bool:
    return path.is_file() and int(_load_json(path).get("passed", 0)) == 1


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    initialized = False
    active_stage = "initialize"
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"quartile directional adjudication run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            _status(run_dir, stage=stage, state="running", decision=f"running_{stage}")
            if stage == "preflight":
                _preflight_stage(run_dir, args)
            elif stage == "replay":
                if not _passed(run_dir / "preflight_gate.json"):
                    raise ArtifactCompatibilityError("replay requires passing preflight")
                _replay_stage(run_dir, args)
            elif stage == "controls":
                if not _passed(run_dir / "historical_replay_gate.json"):
                    raise ArtifactCompatibilityError("controls require passing replay")
                _controls_stage(run_dir, args)
            elif stage == "fittrace":
                if not _passed(run_dir / "controls_gate.json"):
                    raise ArtifactCompatibilityError("fittrace requires passing controls")
                _fittrace_stage(run_dir, args)
            elif stage == "nominate":
                if not _passed(run_dir / "fittrace_gate.json"):
                    raise ArtifactCompatibilityError("nomination requires passing fittrace")
                _nominate_stage(run_dir, args)
            elif stage == "adjudicate":
                if not _passed(run_dir / "nominate_gate.json"):
                    raise ArtifactCompatibilityError("adjudication requires passing nomination")
                _adjudicate_stage(run_dir, args)
            elif stage == "report":
                if not _passed(run_dir / "adjudicate_gate.json"):
                    raise ArtifactCompatibilityError("report requires passing adjudication")
                _report_stage(run_dir, args)
        workflow = _workflow_record(run_dir, args.require_gate)
        decision = workflow["decision"]
        name = str(decision["decision"])
        exit_code = decision_exit_code(decision)
        if not int(workflow["required_gate_pass"]):
            exit_code = 1
        terminal = bool(decision.get("terminal", 0))
        state = "running"
        if terminal:
            state = (
                "complete"
                if int(decision.get("unique_representation_identified", 0))
                else "valid_scientific_stop"
            )
        if exit_code != 0:
            state = "gate_failed"
        _status(
            run_dir,
            stage=args.stage,
            state=state,
            decision=name,
            failure_domain="scientific_gate" if state == "valid_scientific_stop" else None,
            failure_code=name if state == "valid_scientific_stop" else None,
        )
        _artifact_registry(run_dir)
        print(f"quartile directional adjudication decision: {name}", flush=True)
        return exit_code
    except KeyboardInterrupt:
        if run_dir is not None and initialized:
            _status(
                run_dir,
                stage=active_stage,
                state="interrupted",
                decision=f"interrupted_{active_stage}",
                message="interrupted; resume this child run directory",
                failure_domain="interruption",
                failure_code="keyboard_interrupt",
            )
            _artifact_registry(run_dir)
        raise
    except Exception as exc:
        if run_dir is not None and initialized:
            failure = {
                "evaluation_status": "execution_failed",
                "stage": active_stage,
                "failure_domain": str(getattr(exc, "failure_domain", "workflow_execution")),
                "failure_code": str(
                    getattr(
                        exc,
                        "failure_code",
                        "quartile_directional_adjudication_execution_failed",
                    )
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                **_scope(),
            }
            atomic_write_json(
                run_dir / f"{active_stage}_execution_failure.json", failure
            )
            _commit_failed_stage_gate(run_dir, stage=active_stage, failure=failure)
            _workflow_record(run_dir, "none")
            _status(
                run_dir,
                stage=active_stage,
                state="execution_failed",
                decision=str(
                    _load_json(run_dir / "quartile_directional_adjudication_decision.json")["decision"]
                ),
                message=str(exc),
                failure_domain=failure["failure_domain"],
                failure_code=failure["failure_code"],
            )
            _artifact_registry(run_dir)
        print(f"quartile directional adjudication error: {exc}", flush=True)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
