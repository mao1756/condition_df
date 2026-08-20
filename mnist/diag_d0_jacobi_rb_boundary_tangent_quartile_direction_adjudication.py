"""Read-only directional adjudication of the sealed quartile-specialist run.

This workflow replays the historical gain/rank decision and evaluates the
already-sealed checkpoint grid on the two already-open evidence roles.  It
never generates a transition, opens a new evidence role, performs an optimizer
update, selects a system, or executes a controller/sampler.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

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
from mnist.d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication import (
    CRITICAL_VALUE,
    CandidateRoleDecomposition,
    cancellation_diagnostics,
    classify_mechanism_flags,
    classify_power_only_evidence,
    compare_direction_maps,
    directional_compatibility_screen,
    evaluate_cross_role_directional_stability,
    forecast_required_paths,
    gain_transfer_diagnostics,
    path_stability_diagnostics,
    pooled_cell_map,
    quadratic_improvement,
    scalar_optimum,
    summarize_cell_map,
    summarize_optimization_rotation,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_gate import (
    REQUIRED_GATES,
    decision_exit_code,
    decide_workflow,
    evaluate_adjudicate_gate,
    evaluate_decompose_gate,
    evaluate_preflight_gate,
    evaluate_replay_gate,
    evaluate_required_gate,
    not_evaluated_gate,
    safety_record,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_provenance import (
    compare_parent_snapshots,
    load_already_open_role,
    source_paths as direction_source_paths,
    snapshot_parent_run,
    verify_parent,
    verify_resume_compatibility,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_specialist import (
    MODEL_SEEDS_BY_QUARTILE,
    NONZERO_CANDIDATE_IDENTITIES,
    CandidateIdentity,
    build_training_rank_record,
    fixed_unit_gain_record,
    gain_record_from_moments,
    select_training_rank_candidate,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import (
    HostInputStore,
    ModelCallBatchGuard,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_learnability import state_dict_sha256
from mnist.d0_jacobi_rb_reverse_controller import fractional_coordinate


RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-quartile-direction-adjudication"
)
STAGES = ("preflight", "replay", "decompose", "adjudicate", "report", "all")
ROLE_NAMES = ("gain_calibration", "training_rank")
ROLE_CODES = {"gain_calibration": 0, "training_rank": 1}
MAXIMUM_MODEL_FORWARD_BATCH = 32
MAXIMUM_PEAK_MEMORY_FRACTION = 0.80
IDENTITY_TOLERANCE = 5.0e-15
MAXIMUM_PERSISTED_BYTES = 512 * 1024**2
EXPECTED_CANDIDATE_COUNT = 480
EXPECTED_JOB_COUNT = 960
EXPECTED_CHECKPOINT_COUNT = 492
EXPECTED_PATH_COUNT = 32
PHASE_COUNT = 7
MIDPOINT_COUNT = 8

NO_WORK = {
    "new_transitions_generated": 0,
    "new_physical_labels_opened": 0,
    "optimizer_updates_performed": 0,
    "checkpoints_created_or_modified": 0,
    "fresh_selection_paths_opened": 0,
    "confirmation_paths_opened": 0,
    "controller_or_sampling_work_performed": 0,
    "parent_files_modified": 0,
    "historical_design_evidence_authorizing": 0,
    "cache_generation_authorized": 0,
    "physical_training_authorized": 0,
    "fresh_selection_authorized": 0,
    "confirmation_authorized": 0,
    "confirmation_reuse_authorized": 0,
    "controller_control_planning_authorized": 0,
    "controller_execution_authorized": 0,
    "reconstruction_authorized": 0,
    "sampling_authorized": 0,
}

_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "direction_adjudication_decision.json",
}


class QuartileDirectionWorkflowError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "quartile_direction_adjudication_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = failure_domain
        self.failure_code = failure_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope() -> dict[str, int]:
    return {**safety_record(), **NO_WORK}


def _semantic(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _verify_semantic(record: Mapping[str, Any], description: str) -> None:
    body = dict(record)
    observed = body.pop("semantic_sha256", None)
    if observed != config_fingerprint(body):
        raise ArtifactCompatibilityError(f"{description} semantic hash changed")


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


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactCompatibilityError(f"cannot load NPZ artifact: {path}") from exc


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=target.name + ".",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
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
        "path": target.name,
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _atomic_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=target.name + ".",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    message: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
    scientific_evidence_complete: int = 0,
) -> None:
    atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": RUN_SCHEMA + "-status",
            "schema_version": 1,
            "state": state,
            "stage": stage,
            "decision": decision,
            "message": message,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "scientific_evidence_complete": int(scientific_evidence_complete),
            "updated_at": _now(),
            **_scope(),
        },
    )


def _seal_stage(run_dir: Path, names: Sequence[str], seal_name: str) -> dict[str, Any]:
    rows = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"stage artifact is missing: {name}")
        rows.append(
            {
                "path": name,
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-stage-seal",
            "schema_version": 1,
            "artifacts": rows,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / seal_name, record)
    return record


def _verify_stage_seal(run_dir: Path, seal_name: str) -> None:
    seal = _load_json(run_dir / seal_name)
    _verify_semantic(seal, seal_name)
    for row in seal.get("artifacts", []):
        path = run_dir / str(row["path"])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row["size"])
            or file_fingerprint(path) != row["sha256"]
        ):
            raise ArtifactCompatibilityError(f"sealed artifact changed: {path}")


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED or ".tmp" in path.name:
            continue
        rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-artifact-registry",
            "schema_version": 1,
            "artifact_count": len(rows),
            "artifacts": rows,
            "parent_unchanged": int(
                _optional_json(run_dir, "parent_snapshot_after.json") is not None
            ),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_registered_prefix(run_dir: Path) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    registry = _load_json(path)
    _verify_semantic(registry, "child artifact registry")
    for row in registry.get("artifacts", []):
        target = run_dir / str(row["path"])
        if (
            not target.is_file()
            or int(target.stat().st_size) != int(row["size"])
            or file_fingerprint(target) != row["sha256"]
        ):
            raise ArtifactCompatibilityError(f"registered child artifact changed: {target}")


def _source_set() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    entries = (
        package / "d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication.py",
        package
        / "d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_provenance.py",
        package / "d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication_gate.py",
        Path(__file__).resolve(),
    )
    return direction_source_paths(entries)


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-scientific-config",
            "schema_version": 1,
            "parent_quartile_specialist_run_dir": str(
                args.parent_quartile_specialist_run_dir
            ),
            "target": "exact_binary64_jacobi_rao_blackwell_raw_label",
            "target_formula": "y(1-y)*d_y log k_u(y|x)",
            "grid_size": 28,
            "alpha": 1.0,
            "outer_steps": 512,
            "tau_eff": 5.0e-5,
            "candidate_count": EXPECTED_CANDIDATE_COUNT,
            "role_order": list(ROLE_NAMES),
            "maximum_model_forward_batch_size": MAXIMUM_MODEL_FORWARD_BATCH,
            "maximum_peak_cuda_memory_fraction": MAXIMUM_PEAK_MEMORY_FRACTION,
            "quadratic_identity_tolerance": IDENTITY_TOLERANCE,
            "critical_value": CRITICAL_VALUE,
            "authorizing": 0,
            "historical_design_evidence_only": 1,
            "new_role_count": 0,
            "new_path_count": 0,
            "new_seed_count": 0,
            "training_authorized": 0,
            "selection_authorized": 0,
            **_scope(),
        }
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


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    config = _scientific_config(args)
    sources = _source_set()
    source_hash = source_fingerprint(sources)
    if resumed:
        verify_resume_compatibility(
            run_dir,
            expected_bindings={
                "parent_quartile_specialist_run_dir": str(
                    args.parent_quartile_specialist_run_dir
                ),
                "source_fingerprint": source_hash,
                "scientific_config_sha256": config["semantic_sha256"],
            },
        )
        if _load_json(run_dir / "scientific_config.json") != config:
            raise ArtifactCompatibilityError("resume scientific configuration changed")
        _verify_registered_prefix(run_dir)
        return
    atomic_write_json(run_dir / "scientific_config.json", config)
    source_rows = [
        {
            "path": str(path),
            "size": int(path.stat().st_size),
            "sha256": file_fingerprint(path),
        }
        for path in sources
    ]
    atomic_write_json(
        run_dir / "source_closure.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-source-closure",
                "schema_version": 1,
                "source_count": len(source_rows),
                "source_fingerprint": source_hash,
                "sources": source_rows,
                **_scope(),
            }
        ),
    )
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema": RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "parent_quartile_specialist_run_dir": str(
                args.parent_quartile_specialist_run_dir
            ),
            "source_fingerprint": source_hash,
            "source_paths": [str(path) for path in sources],
            "scientific_config_sha256": config["semantic_sha256"],
            "device": args.device,
            **_scope(),
        },
    )


def _write_plan(run_dir: Path) -> dict[str, Any]:
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-plan",
            "schema_version": 1,
            "candidate_count": EXPECTED_CANDIDATE_COUNT,
            "roles": list(ROLE_NAMES),
            "restartable_job_count": EXPECTED_JOB_COUNT,
            "candidate_order": [value.key for value in NONZERO_CANDIDATE_IDENTITIES],
            "checkpoint_selection_performed": 0,
            "historical_roles_authorizing": 0,
            "fresh_evidence_opened": 0,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "adjudication_plan.json", record)
    return record


def _completed_stage(run_dir: Path, *, gate_name: str, seal_name: str) -> bool:
    gate_path = run_dir / gate_name
    seal_path = run_dir / seal_name
    if not gate_path.is_file() and not seal_path.is_file():
        return False
    if not gate_path.is_file() or not seal_path.is_file():
        raise ArtifactCompatibilityError(f"stage completion is partial: {gate_name}")
    gate = _load_json(gate_path)
    _verify_stage_seal(run_dir, seal_name)
    if gate.get("evaluation_status") == "evaluated":
        return True
    raise ArtifactCompatibilityError(f"failed stage cannot be silently reopened: {gate_name}")


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(
        run_dir,
        gate_name="preflight_gate.json",
        seal_name="preflight_artifact_seal.json",
    ):
        return
    before = snapshot_parent_run(args.parent_quartile_specialist_run_dir)
    atomic_write_json(run_dir / "parent_snapshot_before.json", before)
    provenance = verify_parent(
        args.parent_quartile_specialist_run_dir,
        snapshot=before,
        verify_checkpoint_states=True,
        verify_cache_rows=True,
    )
    atomic_write_json(run_dir / "parent_provenance.json", provenance)
    _write_plan(run_dir)
    firewall = _semantic(
        {
            "schema": RUN_SCHEMA + "-role-firewall",
            "schema_version": 1,
            "permitted_roles": list(ROLE_NAMES),
            "forbidden_roles": ["fresh_selection", "untouched_confirmation"],
            "role_loader": "load_already_open_role",
            "role_open_creation_allowed": 0,
            "parent_write_allowed": 0,
            "selection_confirmation_absent": int(
                provenance.get("selection_confirmation_absent", 0)
            ),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "role_firewall.json", firewall)
    # Five float64 arrays plus one int64 count table, before compression.
    consolidated = EXPECTED_CANDIDATE_COUNT * 2 * 32 * 7 * 8 * 6 * 8
    projected = int(2 * consolidated + 64 * 1024**2)
    resource = _semantic(
        {
            "schema": RUN_SCHEMA + "-resource-projection",
            "schema_version": 1,
            "consolidated_uncompressed_bytes": consolidated,
            "projected_total_persisted_bytes": projected,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "within_limit": int(projected <= MAXIMUM_PERSISTED_BYTES),
            "maximum_forward_batch_size": MAXIMUM_MODEL_FORWARD_BATCH,
            "maximum_peak_memory_fraction": MAXIMUM_PEAK_MEMORY_FRACTION,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "resource_projection.json", resource)
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "parent_provenance_valid": int(provenance.get("passed", 0)),
        "parent_registry_valid": int(
            provenance.get("all_registered_artifact_hashes_verified", 0)
        ),
        "parent_terminal_negative_valid": int(
            provenance.get("decision") == "no_training_only_quartile_system"
            and provenance.get("valid_scientific_negative") == 1
        ),
        "checkpoint_grid_valid": int(
            provenance.get("all_checkpoint_hashes_verified", 0)
            and provenance.get("checkpoint_count") == EXPECTED_CHECKPOINT_COUNT
        ),
        "cache_bindings_valid": int(provenance.get("cache_bindings_valid", 0)),
        "role_open_history_valid": int(provenance.get("role_open_history_valid", 0)),
        "selection_confirmation_absent": int(
            provenance.get("selection_confirmation_absent", 0)
        ),
        "scientific_contract_valid": 1,
        "resource_projection_valid": int(resource["within_limit"]),
        "parent_snapshot_valid": 1,
        **_scope(),
    }
    gate = evaluate_preflight_gate(metrics)
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "parent_provenance.json",
            "parent_snapshot_before.json",
            "role_firewall.json",
            "resource_projection.json",
            "adjudication_plan.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )


def _candidate_arrays_valid(table: Mapping[str, np.ndarray], prefix: str) -> bool:
    candidates = NONZERO_CANDIDATE_IDENTITIES
    expected = {
        "quartile": np.asarray([value.quartile for value in candidates]),
        "seed": np.asarray([value.seed for value in candidates]),
        "update": np.asarray([value.update for value in candidates]),
    }
    names = {
        "quartile": f"{prefix}quartile",
        "seed": f"{prefix}seed",
        "update": f"{prefix}update",
    }
    return all(
        names[key] in table
        and tuple(np.asarray(table[names[key]]).shape) == (EXPECTED_CANDIDATE_COUNT,)
        and np.array_equal(np.asarray(table[names[key]], dtype=np.int64), values)
        for key, values in expected.items()
    )


def _parent_gain_records(gain: Mapping[str, np.ndarray]) -> list[Any]:
    records = []
    reason_codes = {
        "cross_term_nonpositive": 0,
        "eligible": 1,
        "fixed_unit_gain": 2,
    }
    for index, candidate in enumerate(NONZERO_CANDIDATE_IDENTITIES):
        if candidate.quartile in (0, 1):
            record = fixed_unit_gain_record(candidate)
        else:
            record = gain_record_from_moments(
                candidate,
                cross_term=float(gain["cross_term"][index]),
                prediction_energy=float(gain["prediction_energy"][index]),
                sample_count=5_619_712,
            )
        expected_gain = float(gain["gain"][index])
        if not (
            int(record.eligible) == int(gain["eligible"][index])
            and reason_codes[record.reason_code] == int(gain["reason_code"][index])
            and (
                (record.gain is None and np.isnan(expected_gain))
                or (record.gain is not None and float(record.gain) == expected_gain)
            )
        ):
            raise QuartileDirectionWorkflowError(
                f"gain record changed: {candidate.key}",
                failure_domain="sealed_replay",
                failure_code="gain_table_replay_invalid",
            )
        records.append(record)
    return records


def _rank_records(
    parent: Path, rank: Mapping[str, np.ndarray], gains: Sequence[Any]
) -> tuple[list[Any], list[dict[str, str]]]:
    with (parent / "training_rank_candidate_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != EXPECTED_CANDIDATE_COUNT:
        raise QuartileDirectionWorkflowError(
            "rank summary row count changed",
            failure_domain="sealed_replay",
            failure_code="rank_table_replay_invalid",
        )
    records = []
    for index, (candidate, gain_record, csv_row) in enumerate(
        zip(NONZERO_CANDIDATE_IDENTITIES, gains, csv_rows, strict=True)
    ):
        cells = np.asarray(rank["fine_cell_improvement"][index], dtype=np.float64)
        phase = np.mean(cells, axis=1, dtype=np.float64)
        midpoint = np.mean(cells, axis=0, dtype=np.float64)
        record = build_training_rank_record(
            candidate,
            gain_record,
            pooled_improvement=float(rank["pooled_improvement"][index]),
            phase_improvements=phase.tolist(),
            midpoint_improvements=midpoint.tolist(),
            fine_cell_improvements=cells.tolist(),
        )
        parsed_candidate = ast.literal_eval(csv_row["candidate"])
        sealed_phase = np.asarray(
            ast.literal_eval(csv_row["phase_improvements"]), dtype=np.float64
        )
        sealed_midpoint = np.asarray(
            ast.literal_eval(csv_row["midpoint_improvements"]), dtype=np.float64
        )
        sealed_cells = np.asarray(
            ast.literal_eval(csv_row["fine_cell_improvements"]), dtype=np.float64
        )
        if (
            parsed_candidate.get("key") != candidate.key
            or csv_row["reason_code"] != record.reason_code
            or int(csv_row["eligible"]) != int(record.eligible)
            or int(rank["eligible"][index]) != int(record.eligible)
            or abs(float(csv_row["pooled_improvement"]) - record.pooled_improvement)
            > IDENTITY_TOLERANCE
            or sealed_phase.shape != (7,)
            or sealed_midpoint.shape != (8,)
            or sealed_cells.shape != (7, 8)
            or float(np.max(np.abs(sealed_phase - phase))) > IDENTITY_TOLERANCE
            or float(np.max(np.abs(sealed_midpoint - midpoint))) > IDENTITY_TOLERANCE
            or float(np.max(np.abs(sealed_cells - cells))) > IDENTITY_TOLERANCE
        ):
            raise QuartileDirectionWorkflowError(
                f"rank record changed: {candidate.key}",
                failure_domain="sealed_replay",
                failure_code="rank_table_replay_invalid",
            )
        records.append(record)
    return records, csv_rows


def _validate_rank_paths(rank: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path_cells = np.asarray(rank["per_path_fine_cell_improvement"], dtype=np.float64)
    counts = np.asarray(rank["per_path_fine_cell_count"], dtype=np.int64)
    path_pooled = np.asarray(rank["per_path_pooled_improvement"], dtype=np.float64)
    expected_shape = (EXPECTED_CANDIDATE_COUNT, EXPECTED_PATH_COUNT, 7, 8)
    if path_cells.shape != expected_shape or counts.shape != expected_shape:
        raise QuartileDirectionWorkflowError(
            "rank path table shape changed",
            failure_domain="sealed_replay",
            failure_code="rank_path_table_replay_invalid",
        )
    weighted_path = np.sum(path_cells * counts, axis=(2, 3), dtype=np.float64) / np.sum(
        counts, axis=(2, 3), dtype=np.int64
    )
    pooled = np.sum(path_cells * counts, axis=(1, 2, 3), dtype=np.float64) / np.sum(
        counts, axis=(1, 2, 3), dtype=np.int64
    )
    fine = np.sum(path_cells * counts, axis=1, dtype=np.float64) / np.sum(
        counts, axis=1, dtype=np.int64
    )
    errors = {
        "path_pooled": float(np.max(np.abs(weighted_path - path_pooled))),
        "candidate_pooled": float(
            np.max(np.abs(pooled - np.asarray(rank["pooled_improvement"])))
        ),
        "fine_cells": float(
            np.max(np.abs(fine - np.asarray(rank["fine_cell_improvement"])))
        ),
    }
    if max(errors.values()) > IDENTITY_TOLERANCE:
        raise QuartileDirectionWorkflowError(
            "rank path reductions changed",
            failure_domain="sealed_replay",
            failure_code="rank_path_table_replay_invalid",
        )
    return {
        "path_ids": np.asarray(rank["path_ids"], dtype=np.int64).tolist(),
        "maximum_errors": errors,
        "passed": 1,
    }


def _replay_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(
        run_dir,
        gate_name="replay_gate.json",
        seal_name="replay_artifact_seal.json",
    ):
        return
    _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
    verify_parent(
        args.parent_quartile_specialist_run_dir,
        verify_checkpoint_states=False,
        verify_cache_rows=False,
    )
    parent = args.parent_quartile_specialist_run_dir
    gain = _load_npz(parent / "gain_table.npz")
    rank = _load_npz(parent / "training_rank_path_tables.npz")
    if not _candidate_arrays_valid(gain, "") or not _candidate_arrays_valid(
        rank, "candidate_"
    ):
        raise QuartileDirectionWorkflowError(
            "canonical candidate order changed",
            failure_domain="sealed_replay",
            failure_code="candidate_order_replay_invalid",
        )
    gains = _parent_gain_records(gain)
    records, csv_rows = _rank_records(parent, rank, gains)
    path_replay = _validate_rank_paths(rank)
    eligible_counts = [
        sum(record.eligible for record in records if record.candidate.quartile == q)
        for q in range(4)
    ]
    winner = select_training_rank_candidate(records, 0)
    terminal_replayed = bool(
        eligible_counts == [80, 0, 0, 0]
        and winner.candidate.key == "q0.seed261333.update1800"
    )
    gain_record = _semantic(
        {
            "schema": RUN_SCHEMA + "-sealed-gain-replay",
            "schema_version": 1,
            "candidate_count": len(gains),
            "q0_q1_unit_gain_count": sum(
                value.candidate.quartile in (0, 1) and value.gain == 1.0
                for value in gains
            ),
            "q2_eligible_count": sum(
                value.candidate.quartile == 2 and value.eligible for value in gains
            ),
            "q3_eligible_count": sum(
                value.candidate.quartile == 3 and value.eligible for value in gains
            ),
            "gain_table_sha256": file_fingerprint(parent / "gain_table.npz"),
            "replayed": 1,
            **_scope(),
        }
    )
    rank_record = _semantic(
        {
            "schema": RUN_SCHEMA + "-sealed-rank-replay",
            "schema_version": 1,
            "candidate_count": len(records),
            "eligible_counts_by_quartile": eligible_counts,
            "q0_winner": winner.candidate.key,
            "q0_winner_pooled_improvement": winner.pooled_improvement,
            "later_eligible_count": sum(eligible_counts[1:]),
            "terminal_decision_replayed": int(terminal_replayed),
            "rank_table_sha256": file_fingerprint(
                parent / "training_rank_path_tables.npz"
            ),
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "sealed_gain_replay.json", gain_record)
    atomic_write_json(run_dir / "sealed_rank_replay.json", rank_record)
    atomic_write_csv(
        run_dir / "rank_candidate_replay.csv",
        [
            {
                "candidate_index": index,
                "candidate_key": record.candidate.key,
                "quartile": record.candidate.quartile,
                "seed": record.candidate.seed,
                "update": record.candidate.update,
                "pooled_improvement": record.pooled_improvement,
                "positive_fine_cells": record.positive_fine_cells,
                "eligible": int(record.eligible),
                "reason_code": record.reason_code,
                "sealed_reason_code": csv_rows[index]["reason_code"],
            }
            for index, record in enumerate(records)
        ],
    )
    atomic_write_json(
        run_dir / "rank_path_replay.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-rank-path-replay",
                "schema_version": 1,
                **path_replay,
                **_scope(),
            }
        ),
    )
    after = snapshot_parent_run(parent)
    unchanged = compare_parent_snapshots(
        _load_json(run_dir / "parent_snapshot_before.json"), after
    )
    metrics = {
        "schema": RUN_SCHEMA + "-replay-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "gain_table_replayed": 1,
        "rank_table_replayed": 1,
        "rank_path_table_replayed": int(path_replay["passed"]),
        "eligibility_replayed": int(eligible_counts == [80, 0, 0, 0]),
        "terminal_decision_replayed": int(terminal_replayed),
        "no_parent_write": int(unchanged.get("passed", 0)),
        **_scope(),
    }
    gate = evaluate_replay_gate(metrics)
    atomic_write_json(run_dir / "replay_metrics.json", metrics)
    atomic_write_json(run_dir / "replay_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "sealed_gain_replay.json",
            "sealed_rank_replay.json",
            "rank_candidate_replay.csv",
            "rank_path_replay.json",
            "replay_metrics.json",
            "replay_gate.json",
        ),
        "replay_artifact_seal.json",
    )


def _checkpoint_rows(parent: Path) -> dict[str, dict[str, Any]]:
    index = _load_json(parent / "training_checkpoint_index.json")
    _verify_semantic(index, "training checkpoint index")
    rows = index.get("checkpoints", [])
    if int(index.get("checkpoint_count", -1)) != EXPECTED_CHECKPOINT_COUNT or len(
        rows
    ) != EXPECTED_CHECKPOINT_COUNT:
        raise ArtifactCompatibilityError("checkpoint grid changed")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("candidate_key", ""))
        if key in result:
            raise ArtifactCompatibilityError("checkpoint identity duplicated")
        result[key] = dict(row)
    return result


def _load_model(
    parent: Path,
    candidate: CandidateIdentity,
    checkpoint_rows: Mapping[str, Mapping[str, Any]],
    device: torch.device,
) -> tuple[ZeroBaselineBoundaryTangentPredictor, dict[str, Any]]:
    row = dict(checkpoint_rows[candidate.key])
    path = parent / str(row["checkpoint_path"])
    if file_fingerprint(path) != row["checkpoint_file_sha256"]:
        raise ArtifactCompatibilityError(f"checkpoint changed: {candidate.key}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        int(payload.get("quartile", -1)) != candidate.quartile
        or int(payload.get("seed", -1)) != candidate.seed
        or int(payload.get("update", -1)) != candidate.update
        or state_dict_sha256(payload["state_dict"]) != row["model_state_sha256"]
        or payload.get("state_sha256") != row["model_state_sha256"]
    ):
        raise ArtifactCompatibilityError(f"checkpoint payload changed: {candidate.key}")
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, row


def _store_quartiles(store: Any) -> np.ndarray:
    reverse_time = torch.as_tensor(
        np.array(store.row_array("reverse_time"), dtype=np.float64, copy=True)
    )
    phase = torch.as_tensor(
        np.array(store.row_array("phase"), dtype=np.int64, copy=True)
    )
    values = fractional_coordinate(reverse_time, phase).forward_outer_quartile.numpy()
    return np.ascontiguousarray(values, dtype=np.int64)


def _reduce_streamed_candidate(
    *,
    model: torch.nn.Module,
    store: Any,
    target: np.ndarray,
    rows: np.ndarray,
    device: torch.device,
    guard: ModelCallBatchGuard,
    expected_path_ids: np.ndarray,
    parent_gain: float,
    diagnostic_gain: float,
) -> CandidateRoleDecomposition:
    path_all = np.asarray(store.row_array("path_id"), dtype=np.int64)[rows]
    phase_all = np.asarray(store.row_array("phase"), dtype=np.int64)[rows]
    midpoint_all = np.asarray(store.row_array("midpoint_index"), dtype=np.int64)[rows]
    path_lookup = {int(value): index for index, value in enumerate(expected_path_ids)}
    shape = (len(expected_path_ids), PHASE_COUNT, MIDPOINT_COUNT)
    cross_parts: list[list[float]] = [[] for _ in range(int(np.prod(shape)))]
    energy_parts: list[list[float]] = [[] for _ in range(int(np.prod(shape)))]
    raw_parts: list[list[float]] = [[] for _ in range(int(np.prod(shape)))]
    parent_parts: list[list[float]] = [[] for _ in range(int(np.prod(shape)))]
    counts = np.zeros(shape, dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), MAXIMUM_MODEL_FORWARD_BATCH):
            active_rows = rows[start : start + MAXIMUM_MODEL_FORWARD_BATCH]
            prediction = (
                guard.call(model, store.batch(active_rows, device=device))
                .detach()
                .to(torch.float64)
                .cpu()
                .numpy()
            )
            truth = np.ascontiguousarray(target[active_rows], dtype=np.float64)
            prediction = np.ascontiguousarray(prediction, dtype=np.float64)
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                raise QuartileDirectionWorkflowError(
                    "checkpoint prediction is nonfinite or wrong-shaped",
                    failure_domain="decomposition",
                    failure_code="finite_reductions_invalid",
                )
            row_cross = np.mean(truth * prediction, axis=1, dtype=np.float64)
            row_energy = np.mean(prediction * prediction, axis=1, dtype=np.float64)
            row_raw = np.mean(
                truth * truth - (truth - prediction) ** 2,
                axis=1,
                dtype=np.float64,
            )
            scaled = parent_gain * prediction
            row_parent = np.mean(
                truth * truth - (truth - scaled) ** 2,
                axis=1,
                dtype=np.float64,
            )
            for local in range(len(active_rows)):
                path_index = path_lookup[int(path_all[start + local])]
                phase = int(phase_all[start + local])
                midpoint = int(midpoint_all[start + local])
                flat = np.ravel_multi_index((path_index, phase, midpoint), shape)
                cross_parts[flat].append(float(row_cross[local]))
                energy_parts[flat].append(float(row_energy[local]))
                raw_parts[flat].append(float(row_raw[local]))
                parent_parts[flat].append(float(row_parent[local]))
                counts[path_index, phase, midpoint] += 1
    if np.any(counts <= 0):
        raise QuartileDirectionWorkflowError(
            "candidate role has an empty path/phase/midpoint cell",
            failure_domain="decomposition",
            failure_code="fine_cell_coverage_invalid",
        )

    def finish(parts: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.empty(shape, dtype=np.float64)
        for flat, local in enumerate(parts):
            values.flat[flat] = math.fsum(local) / len(local)
        return values

    cross = finish(cross_parts)
    energy = finish(energy_parts)
    raw = finish(raw_parts)
    parent = finish(parent_parts)
    raw_error = float(np.max(np.abs(raw - quadratic_improvement(cross, energy, 1.0))))
    parent_error = float(
        np.max(
            np.abs(
                parent - quadratic_improvement(cross, energy, float(parent_gain))
            )
        )
    )
    return CandidateRoleDecomposition(
        path_ids=np.ascontiguousarray(expected_path_ids, dtype=np.int64),
        cross_term=cross,
        prediction_energy=energy,
        raw_improvement=raw,
        parent_gain_improvement=parent,
        diagnostic_gain_improvement=quadratic_improvement(
            cross, energy, diagnostic_gain
        ),
        fine_cell_row_count=counts,
        parent_gain=float(parent_gain),
        diagnostic_gain=float(diagnostic_gain),
        maximum_raw_identity_error=raw_error,
        maximum_parent_gain_identity_error=parent_error,
    )


def _shard_paths(run_dir: Path, role: str, candidate: CandidateIdentity) -> tuple[Path, Path]:
    root = run_dir / "decomposition_shards" / role
    return root / f"{candidate.key}.npz", root / f"{candidate.key}.json"


def _save_shard(
    run_dir: Path,
    *,
    role: str,
    candidate: CandidateIdentity,
    decomposition: CandidateRoleDecomposition,
    checkpoint_row: Mapping[str, Any],
    role_binding_sha256: str,
    role_open_sha256: str,
    scientific_config_sha256: str,
) -> None:
    npz_path, json_path = _shard_paths(run_dir, role, candidate)
    _atomic_npz(
        npz_path,
        candidate_quartile=np.asarray(candidate.quartile, dtype=np.int8),
        candidate_seed=np.asarray(candidate.seed, dtype=np.int64),
        candidate_update=np.asarray(candidate.update, dtype=np.int16),
        role_code=np.asarray(ROLE_CODES[role], dtype=np.int8),
        parent_gain=np.asarray(decomposition.parent_gain, dtype=np.float64),
        diagnostic_gain=np.asarray(decomposition.diagnostic_gain, dtype=np.float64),
        maximum_raw_identity_error=np.asarray(
            decomposition.maximum_raw_identity_error, dtype=np.float64
        ),
        maximum_parent_gain_identity_error=np.asarray(
            decomposition.maximum_parent_gain_identity_error, dtype=np.float64
        ),
        **decomposition.to_arrays(),
    )
    metadata = _semantic(
        {
            "schema": RUN_SCHEMA + "-decomposition-shard",
            "schema_version": 1,
            "candidate": candidate.to_record(),
            "role": role,
            "npz_path": npz_path.relative_to(run_dir).as_posix(),
            "npz_sha256": file_fingerprint(npz_path),
            "checkpoint_path": checkpoint_row["checkpoint_path"],
            "checkpoint_file_sha256": checkpoint_row["checkpoint_file_sha256"],
            "checkpoint_state_sha256": checkpoint_row["model_state_sha256"],
            "role_cache_binding_sha256": role_binding_sha256,
            "role_open_semantic_sha256": role_open_sha256,
            "scientific_config_sha256": scientific_config_sha256,
            "raw_predictions_persisted": 0,
            **_scope(),
        }
    )
    atomic_write_json(json_path, metadata)


def _load_shard(
    run_dir: Path,
    *,
    role: str,
    candidate: CandidateIdentity,
    checkpoint_row: Mapping[str, Any],
    role_binding_sha256: str,
    role_open_sha256: str,
    scientific_config_sha256: str,
) -> CandidateRoleDecomposition | None:
    npz_path, json_path = _shard_paths(run_dir, role, candidate)
    if not npz_path.exists() and not json_path.exists():
        return None
    if not npz_path.is_file() or not json_path.is_file():
        return None
    metadata = _load_json(json_path)
    _verify_semantic(metadata, json_path.name)
    expected = {
        "role": role,
        "npz_sha256": file_fingerprint(npz_path),
        "checkpoint_path": checkpoint_row["checkpoint_path"],
        "checkpoint_file_sha256": checkpoint_row["checkpoint_file_sha256"],
        "checkpoint_state_sha256": checkpoint_row["model_state_sha256"],
        "role_cache_binding_sha256": role_binding_sha256,
        "role_open_semantic_sha256": role_open_sha256,
        "scientific_config_sha256": scientific_config_sha256,
    }
    if metadata.get("candidate") != candidate.to_record() or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise ArtifactCompatibilityError(f"decomposition shard binding changed: {candidate.key}")
    arrays = _load_npz(npz_path)
    scalar = lambda name: np.asarray(arrays[name]).reshape(-1)[0]
    scalar_identity = (
        int(scalar("candidate_quartile")) == candidate.quartile
        and int(scalar("candidate_seed")) == candidate.seed
        and int(scalar("candidate_update")) == candidate.update
        and int(scalar("role_code")) == ROLE_CODES[role]
    )
    if not scalar_identity:
        raise ArtifactCompatibilityError(f"decomposition shard identity changed: {candidate.key}")
    return CandidateRoleDecomposition(
        path_ids=arrays["path_ids"],
        cross_term=arrays["cross_term"],
        prediction_energy=arrays["prediction_energy"],
        raw_improvement=arrays["raw_improvement"],
        parent_gain_improvement=arrays["parent_gain_improvement"],
        diagnostic_gain_improvement=arrays["diagnostic_gain_improvement"],
        fine_cell_row_count=arrays["fine_cell_row_count"],
        parent_gain=float(scalar("parent_gain")),
        diagnostic_gain=float(scalar("diagnostic_gain")),
        maximum_raw_identity_error=float(scalar("maximum_raw_identity_error")),
        maximum_parent_gain_identity_error=float(
            scalar("maximum_parent_gain_identity_error")
        ),
    )


def _pooled(values: np.ndarray, counts: np.ndarray) -> float:
    return math.fsum(
        float(value) * int(weight)
        for value, weight in zip(values.ravel(order="C"), counts.ravel(order="C"), strict=True)
    ) / int(np.sum(counts, dtype=np.int64))


def _consolidate_shards(
    run_dir: Path,
    *,
    shards: Mapping[tuple[str, str], CandidateRoleDecomposition],
) -> dict[str, np.ndarray]:
    candidates = NONZERO_CANDIDATE_IDENTITIES
    first = shards[(ROLE_NAMES[0], candidates[0].key)]
    arrays: dict[str, np.ndarray] = {
        "candidate_quartile": np.asarray([value.quartile for value in candidates], dtype=np.int8),
        "candidate_seed": np.asarray([value.seed for value in candidates], dtype=np.int64),
        "candidate_update": np.asarray([value.update for value in candidates], dtype=np.int16),
        "role_code": np.asarray([ROLE_CODES[value] for value in ROLE_NAMES], dtype=np.int8),
        "role_path_ids": np.empty((2, EXPECTED_PATH_COUNT), dtype=np.int64),
    }
    for name in (
        "cross_term",
        "prediction_energy",
        "raw_improvement",
        "parent_gain_improvement",
        "diagnostic_gain_improvement",
    ):
        arrays[name] = np.empty((480, 2, 32, 7, 8), dtype=np.float64)
    arrays["fine_cell_row_count"] = np.empty((480, 2, 32, 7, 8), dtype=np.int64)
    for role_index, role in enumerate(ROLE_NAMES):
        arrays["role_path_ids"][role_index] = shards[(role, candidates[0].key)].path_ids
        for candidate_index, candidate in enumerate(candidates):
            shard = shards[(role, candidate.key)]
            if not np.array_equal(shard.path_ids, arrays["role_path_ids"][role_index]):
                raise ArtifactCompatibilityError("candidate role path IDs changed")
            for name in (
                "cross_term",
                "prediction_energy",
                "raw_improvement",
                "parent_gain_improvement",
                "diagnostic_gain_improvement",
                "fine_cell_row_count",
            ):
                arrays[name][candidate_index, role_index] = getattr(shard, name)
    _atomic_npz(run_dir / "gain_rank_quadratic_decomposition.npz", **arrays)
    return arrays


def _summary_rows(arrays: Mapping[str, np.ndarray]) -> tuple[list[dict[str, Any]], ...]:
    candidate_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(NONZERO_CANDIDATE_IDENTITIES):
        role_summary: dict[str, Any] = {}
        for role_index, role in enumerate(ROLE_NAMES):
            counts = arrays["fine_cell_row_count"][index, role_index]
            cross = arrays["cross_term"][index, role_index]
            energy = arrays["prediction_energy"][index, role_index]
            cross_summary = summarize_cell_map(cross, counts)
            energy_summary = summarize_cell_map(energy, counts)
            role_summary[role] = (cross_summary, energy_summary)
            candidate_rows.append(
                {
                    "candidate_index": index,
                    "candidate_key": candidate.key,
                    "quartile": candidate.quartile,
                    "seed": candidate.seed,
                    "update": candidate.update,
                    "role": role,
                    "pooled_cross_term": cross_summary["pooled"],
                    "pooled_prediction_energy": energy_summary["pooled"],
                    "lambda_star": scalar_optimum(
                        cross_summary["pooled"], energy_summary["pooled"]
                    )["lambda_star"],
                    "positive_fine_cells": cross_summary["positive_fine_cell_count"],
                    "authorizing": 0,
                }
            )
            path_ids = arrays["role_path_ids"][role_index]
            for path_index, path_id in enumerate(path_ids):
                path_rows.append(
                    {
                        "candidate_index": index,
                        "candidate_key": candidate.key,
                        "role": role,
                        "path_id": int(path_id),
                        "cross_term": _pooled(cross[path_index], counts[path_index]),
                        "prediction_energy": _pooled(
                            energy[path_index], counts[path_index]
                        ),
                        "raw_improvement": _pooled(
                            arrays["raw_improvement"][index, role_index, path_index],
                            counts[path_index],
                        ),
                        "parent_gain_improvement": _pooled(
                            arrays["parent_gain_improvement"][
                                index, role_index, path_index
                            ],
                            counts[path_index],
                        ),
                        "diagnostic_gain_improvement": _pooled(
                            arrays["diagnostic_gain_improvement"][
                                index, role_index, path_index
                            ],
                            counts[path_index],
                        ),
                        "authorizing": 0,
                    }
                )
            pooled_cross, pooled_counts = pooled_cell_map(cross, counts)
            pooled_energy, _ = pooled_cell_map(energy, counts)
            for phase in range(7):
                for midpoint in range(8):
                    stratified_rows.append(
                        {
                            "candidate_index": index,
                            "candidate_key": candidate.key,
                            "role": role,
                            "phase": phase,
                            "midpoint": midpoint,
                            "cross_term": pooled_cross[phase, midpoint],
                            "prediction_energy": pooled_energy[phase, midpoint],
                            "row_count": int(pooled_counts[phase, midpoint]),
                            "authorizing": 0,
                        }
                    )
        gain_cross, gain_energy = role_summary["gain_calibration"]
        rank_cross, rank_energy = role_summary["training_rank"]
        transfer = gain_transfer_diagnostics(
            gain_cross_term=gain_cross["pooled"],
            gain_prediction_energy=gain_energy["pooled"],
            rank_cross_term=rank_cross["pooled"],
            rank_prediction_energy=rank_energy["pooled"],
            gain_permitted=bool(candidate.quartile in (0, 1) or gain_cross["pooled"] > 0.0),
        )
        transfer_rows.append(
            {
                "candidate_index": index,
                "candidate_key": candidate.key,
                "quartile": candidate.quartile,
                "seed": candidate.seed,
                "update": candidate.update,
                **transfer,
            }
        )
    return candidate_rows, path_rows, stratified_rows, transfer_rows


def _decompose_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(
        run_dir,
        gate_name="decompose_gate.json",
        seal_name="decompose_artifact_seal.json",
    ):
        return
    _verify_stage_seal(run_dir, "replay_artifact_seal.json")
    provenance = verify_parent(
        args.parent_quartile_specialist_run_dir,
        verify_checkpoint_states=False,
        verify_cache_rows=False,
    )
    config = _load_json(run_dir / "scientific_config.json")
    parent = args.parent_quartile_specialist_run_dir
    checkpoint_rows = _checkpoint_rows(parent)
    gain_table = _load_npz(parent / "gain_table.npz")
    parent_gains = np.asarray(gain_table["gain"], dtype=np.float64)
    parent_eligible = np.asarray(gain_table["eligible"], dtype=np.uint8)
    device = torch.device(args.device)
    configure_exact_torch_backend(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    guard = ModelCallBatchGuard(maximum_batch_size=MAXIMUM_MODEL_FORWARD_BATCH)
    shards: dict[tuple[str, str], CandidateRoleDecomposition] = {}
    gain_diagnostic: dict[str, float] = {}
    maximum_gain_replay_error = 0.0
    maximum_rank_replay_error = 0.0
    maximum_identity_error = 0.0
    rank_parent = _load_npz(parent / "training_rank_path_tables.npz")
    completed_jobs = 0
    started_at = time.perf_counter()
    for role in ROLE_NAMES:
        opened = load_already_open_role(parent, role)
        store = HostInputStore.from_arrays(
            opened.inputs,
            role="train",
            cache_root=parent / "role_caches" / role,
            index=opened.input_index,
        )
        labels = opened.labels
        target = np.asarray(labels["denoising_target"], dtype=np.float64)
        quartiles = _store_quartiles(store)
        path_ids = np.unique(np.asarray(store.row_array("path_id"), dtype=np.int64))
        if path_ids.shape != (EXPECTED_PATH_COUNT,):
            raise ArtifactCompatibilityError(f"{role} path set changed")
        role_binding_sha = str(opened.binding["semantic_sha256"])
        role_open_sha = str(opened.role_open["semantic_sha256"])
        for candidate_index, candidate in enumerate(NONZERO_CANDIDATE_IDENTITIES):
            row = checkpoint_rows[candidate.key]
            cached = _load_shard(
                run_dir,
                role=role,
                candidate=candidate,
                checkpoint_row=row,
                role_binding_sha256=role_binding_sha,
                role_open_sha256=role_open_sha,
                scientific_config_sha256=config["semantic_sha256"],
            )
            if cached is None:
                model, row = _load_model(parent, candidate, checkpoint_rows, device)
                rows = np.flatnonzero(quartiles == candidate.quartile).astype(np.int64)
                if role == "gain_calibration":
                    diagnostic_gain = 1.0
                else:
                    diagnostic_gain = gain_diagnostic[candidate.key]
                parent_gain = 1.0
                if candidate.quartile in (2, 3) and parent_eligible[candidate_index]:
                    parent_gain = float(parent_gains[candidate_index])
                cached = _reduce_streamed_candidate(
                    model=model,
                    store=store,
                    target=target,
                    rows=rows,
                    device=device,
                    guard=guard,
                    expected_path_ids=path_ids,
                    parent_gain=parent_gain,
                    diagnostic_gain=diagnostic_gain,
                )
                if role == "gain_calibration":
                    pooled_c = _pooled(cached.cross_term, cached.fine_cell_row_count)
                    pooled_p = _pooled(
                        cached.prediction_energy, cached.fine_cell_row_count
                    )
                    diagnostic = scalar_optimum(pooled_c, pooled_p)["lambda_star"]
                    diagnostic_gain = float(diagnostic if diagnostic is not None else 1.0)
                    cached = CandidateRoleDecomposition(
                        path_ids=cached.path_ids,
                        cross_term=cached.cross_term,
                        prediction_energy=cached.prediction_energy,
                        raw_improvement=cached.raw_improvement,
                        parent_gain_improvement=cached.parent_gain_improvement,
                        diagnostic_gain_improvement=quadratic_improvement(
                            cached.cross_term,
                            cached.prediction_energy,
                            diagnostic_gain,
                        ),
                        fine_cell_row_count=cached.fine_cell_row_count,
                        parent_gain=cached.parent_gain,
                        diagnostic_gain=diagnostic_gain,
                        maximum_raw_identity_error=cached.maximum_raw_identity_error,
                        maximum_parent_gain_identity_error=(
                            cached.maximum_parent_gain_identity_error
                        ),
                    )
                _save_shard(
                    run_dir,
                    role=role,
                    candidate=candidate,
                    decomposition=cached,
                    checkpoint_row=row,
                    role_binding_sha256=role_binding_sha,
                    role_open_sha256=role_open_sha,
                    scientific_config_sha256=config["semantic_sha256"],
                )
                del model
            shards[(role, candidate.key)] = cached
            completed_jobs += 1
            elapsed = time.perf_counter() - started_at
            eta = elapsed * (EXPECTED_JOB_COUNT - completed_jobs) / completed_jobs
            print(
                "quartile-direction decomposition "
                f"{completed_jobs}/{EXPECTED_JOB_COUNT} role={role} "
                f"candidate={candidate.key} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
            maximum_identity_error = max(maximum_identity_error, cached.maximum_identity_error)
            pooled_c = _pooled(cached.cross_term, cached.fine_cell_row_count)
            pooled_p = _pooled(cached.prediction_energy, cached.fine_cell_row_count)
            optimum = scalar_optimum(pooled_c, pooled_p)
            if role == "gain_calibration":
                gain_diagnostic[candidate.key] = float(
                    optimum["lambda_star"] if optimum["lambda_star"] is not None else 1.0
                )
                if candidate.quartile in (2, 3):
                    expected_c = float(gain_table["cross_term"][candidate_index])
                    expected_p = float(gain_table["prediction_energy"][candidate_index])
                    maximum_gain_replay_error = max(
                        maximum_gain_replay_error,
                        abs(pooled_c - expected_c),
                        abs(pooled_p - expected_p),
                    )
            else:
                observed = cached.parent_gain_improvement
                sealed = rank_parent["per_path_fine_cell_improvement"][candidate_index]
                maximum_rank_replay_error = max(
                    maximum_rank_replay_error,
                    float(np.max(np.abs(observed - sealed))),
                )
    arrays = _consolidate_shards(run_dir, shards=shards)
    candidate_rows, path_rows, stratified_rows, transfer_rows = _summary_rows(arrays)
    atomic_write_csv(run_dir / "candidate_direction_summary.csv", candidate_rows)
    atomic_write_csv(run_dir / "path_direction_summary.csv", path_rows)
    atomic_write_csv(run_dir / "stratified_direction_summary.csv", stratified_rows)
    atomic_write_csv(run_dir / "gain_transfer_summary.csv", transfer_rows)
    peak_fraction = 0.0
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        peak_fraction = peak / total
    q0_control = any(
        row["quartile"] == 0
        and row["role"] == "training_rank"
        and row["pooled_cross_term"] > 0.0
        for row in candidate_rows
    )
    controls = _semantic(
        {
            "schema": RUN_SCHEMA + "-decomposition-controls",
            "schema_version": 1,
            "maximum_gain_table_C_P_error": maximum_gain_replay_error,
            "maximum_rank_path_reconstruction_error": maximum_rank_replay_error,
            "maximum_quadratic_identity_error": maximum_identity_error,
            "model_call_guard": guard.record(),
            "peak_cuda_memory_fraction": peak_fraction,
            "raw_predictions_persisted": 0,
            "new_evidence_roles_opened": 0,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "decomposition_controls.json", controls)
    metrics = {
        "schema": RUN_SCHEMA + "-decompose-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "all_960_candidate_role_jobs_complete": int(len(shards) == EXPECTED_JOB_COUNT),
        "finite_reductions": int(
            all(np.isfinite(arrays[name]).all() for name in (
                "cross_term", "prediction_energy", "raw_improvement",
                "parent_gain_improvement", "diagnostic_gain_improvement"
            ))
        ),
        "batch_limit_valid": int(guard.maximum_observed_batch_size <= 32),
        "memory_limit_valid": int(peak_fraction <= MAXIMUM_PEAK_MEMORY_FRACTION),
        "gain_table_C_P_replayed": int(maximum_gain_replay_error <= IDENTITY_TOLERANCE),
        "rank_direct_reconstruction_valid": int(
            maximum_identity_error <= IDENTITY_TOLERANCE
            and maximum_rank_replay_error <= IDENTITY_TOLERANCE
        ),
        "q0_positive_control_valid": int(q0_control),
        "algebra_controls_valid": int(maximum_identity_error <= IDENTITY_TOLERANCE),
        "no_raw_predictions_persisted": 1,
        "no_new_evidence_opened": int(provenance.get("selection_confirmation_absent", 0)),
        **_scope(),
    }
    gate = evaluate_decompose_gate(metrics)
    atomic_write_json(run_dir / "decompose_metrics.json", metrics)
    atomic_write_json(run_dir / "decompose_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "gain_rank_quadratic_decomposition.npz",
            "candidate_direction_summary.csv",
            "path_direction_summary.csv",
            "stratified_direction_summary.csv",
            "gain_transfer_summary.csv",
            "decomposition_controls.json",
            "decompose_metrics.json",
            "decompose_gate.json",
        ),
        "decompose_artifact_seal.json",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "infinity" if value > 0.0 else "-infinity"
    return value


def _adjudicate_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(
        run_dir,
        gate_name="adjudicate_gate.json",
        seal_name="adjudicate_artifact_seal.json",
    ):
        return
    _verify_stage_seal(run_dir, "decompose_artifact_seal.json")
    arrays = _load_npz(run_dir / "gain_rank_quadratic_decomposition.npz")
    cancellation_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    candidate_records: dict[int, list[dict[str, Any]]] = {q: [] for q in range(4)}
    maps: dict[int, dict[str, dict[int, dict[int, np.ndarray]]]] = {
        q: {role: {} for role in ROLE_NAMES} for q in range(4)
    }
    counts_maps: dict[int, dict[str, dict[int, dict[int, np.ndarray]]]] = {
        q: {role: {} for role in ROLE_NAMES} for q in range(4)
    }
    for index, candidate in enumerate(NONZERO_CANDIDATE_IDENTITIES):
        role_screens: dict[str, Mapping[str, Any]] = {}
        role_paths: dict[str, Mapping[str, Any]] = {}
        role_maps: dict[str, np.ndarray] = {}
        role_energies: dict[str, np.ndarray] = {}
        for role_index, role in enumerate(ROLE_NAMES):
            cross = arrays["cross_term"][index, role_index]
            energy = arrays["prediction_energy"][index, role_index]
            counts = arrays["fine_cell_row_count"][index, role_index]
            role_maps[role], _ = pooled_cell_map(cross, counts)
            role_energies[role], _ = pooled_cell_map(energy, counts)
            screen = directional_compatibility_screen(
                cross, quartile=candidate.quartile, counts=counts
            )
            role_screens[role] = screen
            transferred = arrays["diagnostic_gain_improvement"][index, role_index]
            stability = path_stability_diagnostics(
                cross,
                counts=counts,
                transferred_improvements=transferred,
                transferred_counts=counts,
            )
            role_paths[role] = stability
            cancellation = cancellation_diagnostics(cross, counts)
            cancellation_rows.append(
                {
                    "candidate_index": index,
                    "candidate_key": candidate.key,
                    "quartile": candidate.quartile,
                    "seed": candidate.seed,
                    "update": candidate.update,
                    "role": role,
                    **cancellation,
                }
            )
            path_rows.append(
                {
                    "candidate_index": index,
                    "candidate_key": candidate.key,
                    "quartile": candidate.quartile,
                    "seed": candidate.seed,
                    "update": candidate.update,
                    "role": role,
                    "positive_cross_term_path_count": stability[
                        "positive_cross_term_path_count"
                    ],
                    "minimum_leave_one_path_out_cross_term": stability[
                        "minimum_leave_one_path_out_cross_term"
                    ],
                    "path_standard_deviation": stability["path_standard_deviation"],
                    "path_standard_error": stability["path_standard_error"],
                    "positive_transferred_improvement_path_count": stability[
                        "positive_transferred_improvement_path_count"
                    ],
                    "minimum_leave_one_path_out_transferred_improvement": stability[
                        "minimum_leave_one_path_out_transferred_improvement"
                    ],
                    "authorizing": 0,
                }
            )
            maps[candidate.quartile][role].setdefault(candidate.seed, {})[
                candidate.update
            ] = role_maps[role]
            counts_maps[candidate.quartile][role].setdefault(candidate.seed, {})[
                candidate.update
            ] = np.sum(counts, axis=0, dtype=np.int64)
        stability = evaluate_cross_role_directional_stability(
            gain_screen=role_screens["gain_calibration"],
            rank_screen=role_screens["training_rank"],
            gain_path_stability=role_paths["gain_calibration"],
            rank_path_stability=role_paths["training_rank"],
        )
        gain_c = role_screens["gain_calibration"]["pooled"]
        gain_p = summarize_cell_map(
            arrays["prediction_energy"][index, 0],
            arrays["fine_cell_row_count"][index, 0],
        )["pooled"]
        rank_c = role_screens["training_rank"]["pooled"]
        rank_p = summarize_cell_map(
            arrays["prediction_energy"][index, 1],
            arrays["fine_cell_row_count"][index, 1],
        )["pooled"]
        transfer = gain_transfer_diagnostics(
            gain_cross_term=gain_c,
            gain_prediction_energy=gain_p,
            rank_cross_term=rank_c,
            rank_prediction_energy=rank_p,
            gain_permitted=bool(candidate.quartile in (0, 1) or gain_c > 0.0),
        )
        forecast = forecast_required_paths(
            arrays["diagnostic_gain_improvement"][index, 1],
            quartile=candidate.quartile,
            counts=arrays["fine_cell_row_count"][index, 1],
        )
        fixed_design_margin = float(forecast["path_point_estimate"]) - (
            CRITICAL_VALUE
            * float(forecast["path_standard_deviation"])
            / math.sqrt(float(forecast["path_count"]))
        )
        forecast_rows.append(
            {
                "candidate_index": index,
                "candidate_key": candidate.key,
                "quartile": candidate.quartile,
                "seed": candidate.seed,
                "update": candidate.update,
                **forecast,
            }
        )
        rank_point = directional_compatibility_screen(
            arrays["diagnostic_gain_improvement"][index, 1],
            quartile=candidate.quartile,
            counts=arrays["fine_cell_row_count"][index, 1],
        )
        candidate_records[candidate.quartile].append(
            {
                "candidate_index": index,
                "candidate_key": candidate.key,
                "seed": candidate.seed,
                "update": candidate.update,
                **transfer,
                "gain_directional_screen_passed": role_screens[
                    "gain_calibration"
                ]["passed"],
                "rank_directional_screen_passed": role_screens["training_rank"][
                    "passed"
                ],
                "cross_role_directionally_stable": stability["passed"],
                "transferred_rank_point_screen_passed": rank_point["passed"],
                "fixed_design_margin": fixed_design_margin,
                "n_rounded": forecast["n_rounded"],
                "required_path_count": forecast["required_path_count"],
            }
        )
    rotation_rows: list[dict[str, Any]] = []
    rotation_flags: dict[int, bool] = {}
    for quartile in range(4):
        for role in ROLE_NAMES:
            summary = summarize_optimization_rotation(
                maps[quartile][role], counts_by_seed=counts_maps[quartile][role]
            )
            rotation_flags[quartile] = bool(
                rotation_flags.get(quartile, False)
                or summary["optimization_time_rotation"]
            )
            for row in summary["adjacent_update_comparisons"]:
                rotation_rows.append(
                    {"quartile": quartile, "role": role, "comparison": "adjacent", **row}
                )
            for row in summary["same_update_cross_seed_comparisons"]:
                rotation_rows.append(
                    {"quartile": quartile, "role": role, "comparison": "cross_seed", **row}
                )
        for record in candidate_records[quartile]:
            index = int(record["candidate_index"])
            comparison = compare_direction_maps(
                arrays["cross_term"][index, 0],
                arrays["cross_term"][index, 1],
                first_counts=arrays["fine_cell_row_count"][index, 0],
                second_counts=arrays["fine_cell_row_count"][index, 1],
            )
            rotation_rows.append(
                {
                    "quartile": quartile,
                    "role": "gain_vs_rank",
                    "comparison": "cross_role",
                    "candidate_key": record["candidate_key"],
                    "seed": record["seed"],
                    "update": record["update"],
                    **comparison,
                }
            )
    classification_rows: list[dict[str, Any]] = []
    decision_evidence: dict[str, Any] = {"quartiles": {}}
    for quartile in range(1, 4):
        flags = classify_mechanism_flags(
            candidate_records[quartile],
            optimization_time_rotation=rotation_flags[quartile],
        )
        power = classify_power_only_evidence(
            candidate_records[quartile],
            expected_seeds=MODEL_SEEDS_BY_QUARTILE[quartile],
        )
        row = {
            "quartile": quartile,
            **flags,
            **power,
        }
        classification_rows.append(row)
        decision_evidence["quartiles"][str(quartile)] = {
            "cross_role_stable_candidate_count": flags[
                "cross_role_stable_candidate_count"
            ],
            "power_only_evidence": power["power_only_evidence"],
            "mechanism_localized": flags["mechanism_localized"],
            "flags": {name: flags[name] for name in (
                "conditional_direction_absent",
                "direction_present_but_role_unstable",
                "phase_midpoint_cancellation",
                "gain_transfer_failure",
                "optimization_time_rotation",
                "strictly_positive_but_too_small",
            )},
        }
    atomic_write_csv(run_dir / "phase_midpoint_cancellation.csv", cancellation_rows)
    atomic_write_csv(run_dir / "optimization_rotation_summary.csv", rotation_rows)
    atomic_write_csv(run_dir / "path_stability_summary.csv", path_rows)
    atomic_write_csv(run_dir / "quartile_mechanism_classification.csv", classification_rows)
    atomic_write_csv(run_dir / "path_count_forecast.csv", forecast_rows)
    after = snapshot_parent_run(args.parent_quartile_specialist_run_dir)
    parent_check = compare_parent_snapshots(
        _load_json(run_dir / "parent_snapshot_before.json"), after
    )
    atomic_write_json(run_dir / "parent_snapshot_adjudicate.json", after)
    metrics = {
        "schema": RUN_SCHEMA + "-adjudicate-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "all_mechanism_records_complete": int(
            len(classification_rows) == 3
            and all(int(row["candidate_count"]) == 120 for row in classification_rows)
        ),
        "all_thresholds_frozen": 1,
        "path_forecasts_valid": int(len(forecast_rows) == EXPECTED_CANDIDATE_COUNT),
        "classifications_deterministic": 1,
        "parent_unchanged": int(parent_check.get("passed", 0)),
        "decision_evidence": decision_evidence,
        **_scope(),
    }
    gate = evaluate_adjudicate_gate(metrics)
    atomic_write_json(run_dir / "adjudicate_metrics.json", metrics)
    atomic_write_json(run_dir / "adjudicate_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "phase_midpoint_cancellation.csv",
            "optimization_rotation_summary.csv",
            "path_stability_summary.csv",
            "quartile_mechanism_classification.csv",
            "path_count_forecast.csv",
            "parent_snapshot_adjudicate.json",
            "adjudicate_metrics.json",
            "adjudicate_gate.json",
        ),
        "adjudicate_artifact_seal.json",
    )


def _decision_evidence(run_dir: Path) -> Mapping[str, Any] | None:
    metrics = _optional_json(run_dir, "adjudicate_metrics.json")
    return metrics.get("decision_evidence") if metrics else None


def _workflow_record(run_dir: Path, require_gate: str) -> dict[str, Any]:
    preflight = _optional_json(run_dir, "preflight_gate.json") or not_evaluated_gate(
        "preflight"
    )
    replay = _optional_json(run_dir, "replay_gate.json") or not_evaluated_gate("replay")
    decompose = _optional_json(run_dir, "decompose_gate.json") or not_evaluated_gate(
        "decompose"
    )
    adjudicate = _optional_json(run_dir, "adjudicate_gate.json") or not_evaluated_gate(
        "adjudicate"
    )
    decision = decide_workflow(
        preflight_gate=preflight,
        replay_gate=replay,
        decompose_gate=decompose,
        adjudicate_gate=adjudicate,
        evidence=_decision_evidence(run_dir),
    )
    workflow = evaluate_required_gate(
        preflight_gate=preflight,
        replay_gate=replay,
        decompose_gate=decompose,
        adjudicate_gate=adjudicate,
        decision=decision,
        require_gate=require_gate,
    )
    decision = {**decision, **_scope()}
    workflow["decision"] = decision
    workflow.update(_scope())
    atomic_write_json(run_dir / "direction_adjudication_decision.json", decision)
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    return workflow


def _write_report(run_dir: Path) -> None:
    workflow = _workflow_record(run_dir, "none")
    decision = workflow["decision"]
    evidence = _decision_evidence(run_dir) or {}
    lines = [
        "# Quartile direction adjudication",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        "This workflow used only the already-open gain-calibration and "
        "training-rank roles. It generated no transitions, labels, optimizer "
        "updates, selection evidence, confirmation evidence, controller "
        "trajectory, reconstruction, or sample.",
        "",
        "## Later-quartile evidence",
        "",
    ]
    for quartile, row in sorted((evidence.get("quartiles") or {}).items()):
        lines.append(
            f"- q{quartile}: stable candidates={row['cross_role_stable_candidate_count']}; "
            f"power-only={row['power_only_evidence']}; flags={row['flags']}"
        )
    lines.extend(
        [
            "",
            "No candidate named by this report is selected or authorized for a "
            "future audit. All execution-authorization fields remain zero.",
            "",
        ]
    )
    _atomic_text(run_dir / "REPORT.md", "\n".join(lines))


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "replay", "decompose", "adjudicate", "report")
    return (stage,)


def _commit_failed_stage_gate(
    run_dir: Path, *, stage: str, failure: Mapping[str, Any]
) -> None:
    definitions = {
        "preflight": ("preflight_metrics.json", "preflight_gate.json", evaluate_preflight_gate),
        "replay": ("replay_metrics.json", "replay_gate.json", evaluate_replay_gate),
        "decompose": ("decompose_metrics.json", "decompose_gate.json", evaluate_decompose_gate),
        "adjudicate": (
            "adjudicate_metrics.json",
            "adjudicate_gate.json",
            evaluate_adjudicate_gate,
        ),
    }
    if stage not in definitions:
        return
    metrics_name, gate_name, evaluator = definitions[stage]
    if (run_dir / gate_name).is_file():
        return
    metrics = {
        "schema": RUN_SCHEMA + f"-{stage}-failure-metrics",
        "schema_version": 1,
        "evaluation_status": "execution_failed",
        "failure_domain": failure.get("failure_domain"),
        "failure_code": failure.get("failure_code"),
        "error": failure.get("error"),
        **_scope(),
    }
    atomic_write_json(run_dir / metrics_name, metrics)
    atomic_write_json(run_dir / gate_name, evaluator(metrics))


def _verify_report(run_dir: Path) -> None:
    for seal in (
        "preflight_artifact_seal.json",
        "replay_artifact_seal.json",
        "decompose_artifact_seal.json",
        "adjudicate_artifact_seal.json",
    ):
        if (run_dir / seal).is_file():
            _verify_stage_seal(run_dir, seal)
    _write_report(run_dir)
    before = _load_json(run_dir / "parent_snapshot_before.json")
    current = snapshot_parent_run(Path(before["run_dir"]))
    result = compare_parent_snapshots(before, current)
    if not int(result.get("passed", 0)):
        raise ArtifactCompatibilityError("immutable parent changed after report")
    atomic_write_json(run_dir / "parent_snapshot_after.json", current)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-quartile-specialist-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_"
            "quartile_direction_adjudication"
        ),
    )
    parser.add_argument("--run-name", default="production-readonly-q1-q3-direction-adjudication")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    for name in (
        "parent_quartile_specialist_run_dir",
        "resume_run_dir",
        "runs_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.resume_run_dir is None and args.stage in {
        "replay",
        "decompose",
        "adjudicate",
        "report",
    }:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    expected = {
        "preflight": "preflight",
        "replay": "replay",
        "decompose": "decompose",
        "adjudicate": "adjudicate",
        "report": "none",
        "all": "adjudicate",
    }[args.stage]
    if args.require_gate not in {"none", expected}:
        parser.error(f"--stage {args.stage} cannot require only {args.require_gate}")
    if args.stage in {"decompose", "all"} and args.device != "cuda":
        parser.error("production checkpoint decomposition requires --device cuda")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    initialized = False
    active_stage = "initialize"
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"quartile-direction adjudication run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        _status(run_dir, state="running", stage=args.stage)
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            _status(run_dir, state="running", stage=stage)
            if stage == "preflight":
                _preflight_stage(run_dir, args)
            elif stage == "replay":
                if not _passed(_optional_json(run_dir, "preflight_gate.json")):
                    raise ArtifactCompatibilityError("replay requires passing preflight")
                _replay_stage(run_dir, args)
            elif stage == "decompose":
                if not _passed(_optional_json(run_dir, "replay_gate.json")):
                    raise ArtifactCompatibilityError("decompose requires passing replay")
                _decompose_stage(run_dir, args)
            elif stage == "adjudicate":
                if not _passed(_optional_json(run_dir, "decompose_gate.json")):
                    raise ArtifactCompatibilityError("adjudicate requires passing decompose")
                _adjudicate_stage(run_dir, args)
            elif stage == "report":
                _verify_report(run_dir)
        workflow = _workflow_record(run_dir, args.require_gate)
        decision = workflow["decision"]
        name = str(decision["decision"])
        required_pass = int(workflow["required_gate_pass"]) == 1
        terminal_code = decision_exit_code(decision)
        if not required_pass:
            terminal_code = 1
        state = "complete"
        if terminal_code == 1:
            state = "gate_failed"
        elif terminal_code == 2:
            state = "valid_scientific_negative"
        _status(
            run_dir,
            state=state,
            stage=args.stage,
            decision=name,
            failure_domain="scientific_gate" if terminal_code == 2 else None,
            failure_code=name if terminal_code == 2 else None,
            scientific_evidence_complete=int(decision.get("scientific_evidence_complete", 0)),
        )
        _artifact_registry(run_dir)
        print(f"quartile-direction adjudication decision: {name}", flush=True)
        return terminal_code
    except KeyboardInterrupt:
        if run_dir is not None and initialized:
            _status(
                run_dir,
                state="interrupted",
                stage=active_stage,
                message="interrupted; resume the same child run directory",
                failure_domain="interruption",
                failure_code="keyboard_interrupt",
            )
            _artifact_registry(run_dir)
        raise
    except Exception as exc:
        if run_dir is not None and initialized:
            failure = {
                "schema": RUN_SCHEMA + "-execution-failure",
                "schema_version": 1,
                "evaluation_status": "execution_failed",
                "stage": active_stage,
                "failure_domain": str(getattr(exc, "failure_domain", "workflow_execution")),
                "failure_code": str(
                    getattr(
                        exc,
                        "failure_code",
                        "quartile_direction_adjudication_execution_failed",
                    )
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                **_scope(),
            }
            atomic_write_json(run_dir / f"{active_stage}_execution_failure.json", failure)
            _commit_failed_stage_gate(run_dir, stage=active_stage, failure=failure)
            _workflow_record(run_dir, "none")
            _status(
                run_dir,
                state="execution_failed",
                stage=active_stage,
                message=str(exc),
                failure_domain=failure["failure_domain"],
                failure_code=failure["failure_code"],
            )
            _artifact_registry(run_dir)
        print(f"quartile-direction adjudication error: {exc}", flush=True)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
