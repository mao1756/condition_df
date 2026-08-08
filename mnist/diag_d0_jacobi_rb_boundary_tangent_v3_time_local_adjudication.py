"""Read-only adjudication of the sealed zero-baseline v3 validation result.

This workflow verifies immutable parents, replays the original 228-component
selection exactly, and decomposes three frozen q0 nominees into target alignment
and prediction energy.  It creates no paths, transitions, optimizer updates,
confirmation evidence, controller trajectories, reconstructions, or samples.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
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
    source_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import (
    LabelOpenAuthorization,
    ModelCallBatchGuard,
    open_external_input_store,
    open_external_label_store,
    predict_to_cpu,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_learnability import state_dict_sha256
from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local import (
    PRODUCTION_CRITICAL_VALUE,
    PRODUCTION_Q0_NOMINEES,
    Q0_POOLED_COMPONENT,
    TIME_QUARTILES,
    aggregate_quadratic_risk_decomposition,
    advisory_scalar_calibration,
    build_resolution_ladder,
    build_sealed_selection_replay,
    classify_quartile_signal,
    forecast_required_path_count,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local_gate import (
    REQUIRED_GATES,
    decide_workflow,
    evaluate_decomposition_gate,
    evaluate_preflight_gate,
    evaluate_replay_gate,
    evaluate_required_gate,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local_provenance import (
    verify_time_local_adjudication_parents,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_selection import V3_FAMILY_NAMES
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    v3_transitive_source_paths,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-v3-time-local-adjudication"
STAGES = ("preflight", "replay", "decompose", "report", "all")
MAXIMUM_FORWARD_BATCH = 32
MAXIMUM_PEAK_MEMORY_FRACTION = 0.80
IDENTITY_TOLERANCE = 5.0e-15
WITNESS_QUARTILE_ENERGIES = (
    0.0016263861318650955,
    0.0006212765822032824,
    0.00021024758421042043,
    0.00013578918212975752,
)

MECHANISM_RECOMMENDATIONS = {
    "directional_alignment_missing": (
        "train an independent width-32 expert for this quartile"
    ),
    "prediction_energy_dominates": (
        "use training-only time-local scalar shrinkage for this quartile"
    ),
    "positive_but_underpowered": (
        "design a powered fresh selection panel without changing the target"
    ),
    "resolved": "retain the resolved historical diagnostic without reselection",
}

NO_WORK = {
    "new_exact_transitions": 0,
    "new_path_ids": 0,
    "optimizer_updates": 0,
    "physical_training_performed": 0,
    "validation_selection_performed": 0,
    "confirmation_performed": 0,
    "confirmation_evidence_accessed": 0,
    "controller_control_trajectory_performed": 0,
    "reconstruction_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "parent_mutations": 0,
}

_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "workflow_gate.json",
    "time_local_adjudication_decision.json",
}


class TimeLocalWorkflowError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "time_local_adjudication_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = failure_domain
        self.failure_code = failure_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope() -> dict[str, int]:
    return dict(NO_WORK)


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


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _semantic(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _verify_semantic(record: Mapping[str, Any], description: str) -> None:
    body = dict(record)
    observed = body.pop("semantic_sha256", None)
    if observed != config_fingerprint(body):
        raise ArtifactCompatibilityError(f"{description} semantic hash changed")


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> dict[str, Any]:
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
        "path": target.name,
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ArtifactCompatibilityError(f"cannot load NPZ artifact: {path}") from exc


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
    artifacts = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"stage artifact is missing: {name}")
        artifacts.append(
            {"path": name, "size": int(path.stat().st_size), "sha256": file_fingerprint(path)}
        )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-stage-seal",
            "schema_version": 1,
            "artifacts": artifacts,
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
            or path.stat().st_size != int(row["size"])
            or file_fingerprint(path) != row["sha256"]
        ):
            raise ArtifactCompatibilityError(f"sealed artifact changed: {path}")


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(run_dir.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED or ".tmp" in path.name:
            continue
        artifacts.append(
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
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
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
    _verify_semantic(registry, "child registry")
    for row in registry.get("artifacts", []):
        target = run_dir / str(row["path"])
        if (
            not target.is_file()
            or target.stat().st_size != int(row["size"])
            or file_fingerprint(target) != row["sha256"]
        ):
            raise ArtifactCompatibilityError(f"registered child artifact changed: {target}")


def _source_set() -> tuple[Path, ...]:
    return v3_transitive_source_paths((Path(__file__).resolve(),))


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-scientific-config",
            "schema_version": 1,
            "parent_memory_v3_run_dir": str(args.parent_memory_v3_run_dir),
            "parent_coarse_witness_run_dir": str(args.parent_coarse_witness_run_dir),
            "parent_bayes_power_run_dir": str(args.parent_bayes_power_run_dir),
            "frozen_q0_nominees": [
                {
                    "seed": seed,
                    "update": update,
                    "point_estimate": point,
                    "adjusted_lower_bound": lower,
                    "positive_fine_cells": count,
                }
                for seed, update, point, lower, count in PRODUCTION_Q0_NOMINEES
            ],
            "critical_value": PRODUCTION_CRITICAL_VALUE,
            "quadratic_identity_tolerance": IDENTITY_TOLERANCE,
            "maximum_model_forward_batch_size": MAXIMUM_FORWARD_BATCH,
            "maximum_peak_cuda_memory_fraction": MAXIMUM_PEAK_MEMORY_FRACTION,
            "post_hoc_resolution_ladder_authorizing": 0,
            "advisory_scalar_optimum_applied": 0,
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
        if _load_json(run_dir / "scientific_config.json") != config:
            raise ArtifactCompatibilityError("resume scientific configuration changed")
        manifest = _load_json(run_dir / "run_manifest.json")
        if (
            manifest.get("source_fingerprint") != source_hash
            or manifest.get("scientific_config_sha256") != config["semantic_sha256"]
        ):
            raise ArtifactCompatibilityError("resume source or config binding changed")
        _verify_registered_prefix(run_dir)
        return
    atomic_write_json(run_dir / "scientific_config.json", config)
    atomic_write_json(
        run_dir / "run_manifest.json",
        {
            "schema": RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
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
            "created_before_checkpoint_loading": 1,
            "nominees": [
                {
                    "seed": seed,
                    "update": update,
                    "point_estimate": point,
                    "adjusted_lower_bound": lower,
                    "positive_q0_fine_cells": count,
                }
                for seed, update, point, lower, count in PRODUCTION_Q0_NOMINEES
            ],
            "resolution_ladder_authorizing": 0,
            "scalar_calibration_authorizing": 0,
            "historical_validation_can_authorize_confirmation": 0,
            "confirmation_path_start": 0xF2000,
            "confirmation_path_stop_exclusive": 0xF2040,
            **_scope(),
        }
    )
    atomic_write_json(run_dir / "adjudication_plan.json", record)
    return record


def _verify_parents(args: argparse.Namespace) -> dict[str, Any]:
    return verify_time_local_adjudication_parents(
        memory_v3_run_dir=args.parent_memory_v3_run_dir,
        coarse_witness_run_dir=args.parent_coarse_witness_run_dir,
        bayes_power_run_dir=args.parent_bayes_power_run_dir,
        verify_external_cache=True,
    )


def _completed_stage(
    run_dir: Path, *, gate_name: str, seal_name: str
) -> bool:
    gate_path = run_dir / gate_name
    seal_path = run_dir / seal_name
    if not gate_path.is_file() and not seal_path.is_file():
        return False
    if not gate_path.is_file() or not seal_path.is_file():
        return False
    gate = _load_json(gate_path)
    status = gate.get("evaluation_status")
    if status not in {"evaluated", "execution_failed"}:
        return False
    _verify_stage_seal(run_dir, seal_name)
    if status == "evaluated":
        return True

    retry_names = {
        "preflight_gate.json": ("preflight", "preflight_metrics.json"),
        "time_local_replay_gate.json": ("replay", "replay_metrics.json"),
        "time_local_decomposition_gate.json": (
            "decompose",
            "decomposition_metrics.json",
        ),
    }
    stage, metrics_name = retry_names[gate_name]
    failure_name = f"{stage}_execution_failure.json"
    root = run_dir / "execution_attempts" / stage
    attempts = sorted(root.glob("attempt-*")) if root.is_dir() else []
    attempt = root / f"attempt-{len(attempts) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    names = [failure_name, metrics_name, gate_name, seal_name]
    if (run_dir / "artifact_registry.json").is_file():
        names.append("artifact_registry.json")
    archived = []
    for name in names:
        source = run_dir / name
        if not source.is_file():
            raise ArtifactCompatibilityError(
                f"failed {stage} attempt is missing {name}"
            )
        atomic_write_json(attempt / name, _load_json(source))
        archived.append({"path": name, "sha256": file_fingerprint(source)})
    atomic_write_json(
        attempt / "retry_authorization.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-execution-retry",
                "schema_version": 1,
                "stage": stage,
                "attempt": len(attempts) + 1,
                "archived": archived,
                "scientific_gate_reopened": 0,
                "execution_failure_reopened": 1,
                **_scope(),
            }
        ),
    )
    for name in names:
        (run_dir / name).unlink(missing_ok=True)
    return False


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(
        run_dir,
        gate_name="preflight_gate.json",
        seal_name="preflight_artifact_seal.json",
    ):
        return
    provenance = _verify_parents(args)
    atomic_write_json(run_dir / "parent_provenance.json", provenance)
    atomic_write_json(
        run_dir / "parent_immutability_report.json",
        {
            "schema": RUN_SCHEMA + "-parent-immutability",
            "schema_version": 1,
            "parents": provenance["parent_immutability"],
            "confirmation_namespace_opened": 0,
            "parents_mutated": 0,
            **_scope(),
        },
    )
    _write_plan(run_dir)
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "memory_parent_valid": 1,
        "coarse_witness_parent_valid": 1,
        "bayes_power_parent_valid": 1,
        "selection_seal_valid": int(provenance["selection_seal_verified"]),
        "checkpoint_bindings_valid": int(provenance["all_checkpoint_hashes_verified"]),
        "confirmation_namespace_unopened": int(
            provenance["confirmation_namespace_opened"] == 0
        ),
        "parent_immutability_valid": int(provenance["parents_mutated"] == 0),
        **_scope(),
    }
    gate = evaluate_preflight_gate(metrics)
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "parent_provenance.json",
            "parent_immutability_report.json",
            "adjudication_plan.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )


def _load_replay(parent: Path):
    table = _load_npz(parent / "validation_candidate_path_tables.npz")
    max_t = _load_npz(parent / "validation_search_max_t.npz")
    return build_sealed_selection_replay(
        seeds=table["seeds"],
        updates=table["updates"],
        path_ids=table["path_ids"],
        path_values=table["path_values"],
        point_estimates=max_t["point_estimates"],
        standard_errors=max_t["standard_errors"],
        lower_bounds=max_t["lower_bounds"],
        maxima=max_t["maxima"],
        confidence=0.995,
        require_production_fixture=True,
    )


def _witness_energies(root: Path) -> np.ndarray:
    panel_a = _load_npz(root / "panels" / "a" / "cell_means.npz")["cell_means"]
    panel_b = _load_npz(root / "panels" / "b" / "cell_means.npz")["cell_means"]
    mean_a = np.mean(panel_a, axis=0, dtype=np.float64)
    mean_b = np.mean(panel_b, axis=0, dtype=np.float64)
    energies = np.mean(mean_a * mean_b, axis=(1, 2), dtype=np.float64)
    if not np.allclose(
        energies, np.asarray(WITNESS_QUARTILE_ENERGIES), rtol=0.0, atol=5e-15
    ):
        raise TimeLocalWorkflowError(
            "coarse witness quartile energies changed",
            failure_code="coarse_witness_replay_invalid",
        )
    return np.ascontiguousarray(energies)


def _resolution_rows(levels) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in levels:
        for candidate_index in range(level.point_estimates.shape[0]):
            for component_index, name in enumerate(level.names):
                rows.append(
                    {
                        "level": level.level,
                        "candidate_index": candidate_index,
                        "component_index": component_index,
                        "component": name,
                        "point_estimate": level.point_estimates[candidate_index, component_index],
                        "standard_error": level.standard_errors[candidate_index, component_index],
                        "descriptive_lower_bound": level.descriptive_lower_bounds[
                            candidate_index, component_index
                        ],
                        "authorizing": 0,
                    }
                )
    return rows


def _later_adjusted_positive_count(lower_bounds: np.ndarray) -> int:
    values = np.asarray(lower_bounds)
    if values.shape != (120, 228) or values.dtype != np.dtype(np.float64):
        raise TimeLocalWorkflowError(
            "sealed adjusted-bound table has the wrong shape or dtype",
            failure_code="sealed_selection_replay_invalid",
        )
    return int(
        np.count_nonzero(values[:, 56:224] > 0.0)
        + np.count_nonzero(values[:, 225:228] > 0.0)
    )


def _compare_sealed_validation_risks(
    *,
    current_path_ids: np.ndarray,
    current_values: np.ndarray,
    sealed_path_ids: np.ndarray,
    sealed_values: np.ndarray,
) -> dict[str, Any]:
    paths_equal = bool(np.array_equal(current_path_ids, sealed_path_ids))
    shapes_equal = tuple(current_values.shape) == tuple(sealed_values.shape)
    maximum_error: float | None = None
    bitwise_equal = False
    if paths_equal and shapes_equal:
        maximum_error = float(np.max(np.abs(current_values - sealed_values)))
        bitwise_equal = bool(np.array_equal(current_values, sealed_values))
    passed = bool(
        paths_equal
        and shapes_equal
        and maximum_error is not None
        and np.isfinite(maximum_error)
        and maximum_error <= IDENTITY_TOLERANCE
    )
    return {
        "path_ids_equal": int(paths_equal),
        "shapes_equal": int(shapes_equal),
        "bitwise_equal": int(bitwise_equal),
        "maximum_absolute_error": maximum_error,
        "maximum_allowed_error": IDENTITY_TOLERANCE,
        "passed": int(passed),
    }


def _replay_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(
        run_dir,
        gate_name="time_local_replay_gate.json",
        seal_name="replay_artifact_seal.json",
    ):
        return
    _verify_parents(args)
    _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
    replay = _load_replay(args.parent_memory_v3_run_dir)
    replay_record = replay.to_record()
    atomic_write_json(run_dir / "sealed_selection_replay.json", replay_record)
    atomic_write_json(
        run_dir / "partial_discovery_census.json",
        {
            "schema": RUN_SCHEMA + "-partial-discovery",
            "schema_version": 1,
            "positive_simultaneous_candidate_components": replay.positive_component_count,
            "discovering_candidate_count": replay.discovering_candidate_count,
            "discovered_components": list(replay.discovered_component_indices),
            "all_in_q0": int(
                all(index < 56 or index == Q0_POOLED_COMPONENT for index in replay.discovered_component_indices)
            ),
            "later_adjusted_positive_count": int(
                _later_adjusted_positive_count(replay.result.lower_bounds)
            ),
            "all_point_positive_candidate_count": int(
                np.count_nonzero(np.all(replay.result.point_estimates > 0.0, axis=1))
            ),
            **_scope(),
        },
    )
    ladder = build_resolution_ladder(replay.table, critical_value=replay.result.critical_value)
    resolution_rows = _resolution_rows(ladder)
    atomic_write_csv(run_dir / "resolution_ladder.csv", resolution_rows)
    atomic_write_json(
        run_dir / "resolution_ladder.json",
        {
            "schema": RUN_SCHEMA + "-resolution-ladder",
            "schema_version": 1,
            "levels": [
                {"level": level.level, "component_count": len(level.names)} for level in ladder
            ],
            "row_count": len(resolution_rows),
            "post_hoc": 1,
            "authorizing": 0,
            **_scope(),
        },
    )
    energies = _witness_energies(args.parent_coarse_witness_run_dir)
    energy_rows = [
        {"quartile": q, "independent_panel_conditional_mean_energy": value}
        for q, value in enumerate(energies.tolist())
    ]
    atomic_write_csv(run_dir / "coarse_witness_quartile_energy.csv", energy_rows)
    atomic_write_json(
        run_dir / "coarse_witness_quartile_energy.json",
        {
            "schema": RUN_SCHEMA + "-coarse-witness-energy",
            "schema_version": 1,
            "quartile_energies": energies.tolist(),
            "overall_energy": float(np.mean(energies, dtype=np.float64)),
            "finite_positive_overall": int(np.isfinite(energies).all() and np.mean(energies) > 0),
            **_scope(),
        },
    )
    signal_rows = []
    for nominee in replay.nominees:
        for q in range(TIME_QUARTILES):
            point = float(replay.result.point_estimates[nominee.candidate_index, 224 + q])
            signal_rows.append(
                {
                    "seed": nominee.seed,
                    "update": nominee.update,
                    "quartile": q,
                    "point_improvement": point,
                    "witness_energy": energies[q],
                    "signal_capture_ratio_advisory": point / energies[q],
                    "authorizing": 0,
                }
            )
    atomic_write_csv(run_dir / "signal_capture_table.csv", signal_rows)
    metrics = {
        "schema": RUN_SCHEMA + "-replay-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "candidate_table_shape_valid": 1,
        "candidate_grid_valid": 1,
        "critical_value_reproduced": int(replay.result.critical_value == PRODUCTION_CRITICAL_VALUE),
        "zero_eligible_candidates_reproduced": 1,
        "logical_update_zero_reproduced": 1,
        "partial_discovery_census_reproduced": 1,
        "nominee_tuples_reproduced": 1,
        "coarse_witness_reproduced": 1,
        **_scope(),
    }
    gate = evaluate_replay_gate(metrics)
    atomic_write_json(run_dir / "replay_metrics.json", metrics)
    atomic_write_json(run_dir / "time_local_replay_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "sealed_selection_replay.json",
            "partial_discovery_census.json",
            "resolution_ladder.csv",
            "resolution_ladder.json",
            "coarse_witness_quartile_energy.csv",
            "coarse_witness_quartile_energy.json",
            "signal_capture_table.csv",
            "replay_metrics.json",
            "time_local_replay_gate.json",
        ),
        "replay_artifact_seal.json",
    )


def _opening_authorization(memory_parent: Path, cache_root: Path, role: str):
    if role == "train":
        record = _load_json(memory_parent / "training_label_open.json")
        purpose = "physical_training"
    else:
        record = _load_json(memory_parent / "validation_label_open.json")
        purpose = "validation_selection"
    _verify_semantic(record, f"historical {role} label opening")
    return LabelOpenAuthorization(
        cache_root=cache_root,
        role=role,
        purpose=purpose,
        opening_seal_sha256=str(record["semantic_sha256"]),
    )


def _candidate_grid(parent: Path) -> dict[tuple[int, int], dict[str, Any]]:
    grid = _load_json(parent / "candidate_grid.json")
    _verify_semantic(grid, "candidate grid")
    return {
        (int(row["seed"]), int(row["update"])): dict(row)
        for row in grid["checkpoints"]
    }


def _load_nominee_model(
    parent: Path,
    *,
    seed: int,
    update: int,
    grid: Mapping[tuple[int, int], Mapping[str, Any]],
    device: str,
) -> tuple[ZeroBaselineBoundaryTangentPredictor, dict[str, Any]]:
    row = dict(grid[(seed, update)])
    path = parent / str(row["checkpoint_path"])
    if file_fingerprint(path) != row["checkpoint_file_sha256"]:
        raise ArtifactCompatibilityError(f"nominee checkpoint file changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        int(payload.get("seed", -1)) != seed
        or int(payload.get("update", -1)) != update
        or payload.get("state_sha256") != row["state_sha256"]
        or state_dict_sha256(payload["state_dict"]) != row["state_sha256"]
    ):
        raise ArtifactCompatibilityError("nominee checkpoint payload changed")
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, {
        "seed": seed,
        "update": update,
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_file_sha256": row["checkpoint_file_sha256"],
        "state_sha256": row["state_sha256"],
    }


def _decompose_role(
    *,
    role: str,
    memory_parent: Path,
    cache_root: Path,
    grid: Mapping[tuple[int, int], Mapping[str, Any]],
    device: str,
) -> tuple[Any, list[dict[str, Any]], float, int]:
    inputs = open_external_input_store(cache_root, role)
    authorization = _opening_authorization(memory_parent, cache_root, role)
    labels = open_external_label_store(cache_root, role, authorization=authorization)
    if inputs.row_count != labels.row_count:
        raise TimeLocalWorkflowError("input/label row count changed")
    audit_names = ("sample_key", "path_id", "outer_step", "phase", "midpoint_index")
    for name in audit_names:
        if not np.array_equal(inputs.row_array(name), labels.row_array(name)):
            raise TimeLocalWorkflowError(f"input/label audit join changed: {name}")
    expected_paths = np.unique(inputs.row_array("path_id")).astype(np.int64, copy=False)
    decompositions = []
    model_records: list[dict[str, Any]] = []
    maximum_batch = 0
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    for seed, update, *_ in PRODUCTION_Q0_NOMINEES:
        model, model_record = _load_nominee_model(
            memory_parent, seed=seed, update=update, grid=grid, device=device
        )
        guard = ModelCallBatchGuard(maximum_batch_size=MAXIMUM_FORWARD_BATCH)
        predictions, batch_record = predict_to_cpu(
            model,
            inputs,
            device=device,
            guard=guard,
            batch_size=MAXIMUM_FORWARD_BATCH,
            output_dtype=np.float64,
        )
        maximum_batch = max(maximum_batch, int(batch_record["maximum_observed_batch_size"]))
        decomposition = aggregate_quadratic_risk_decomposition(
            sample_keys=inputs.row_array("sample_key"),
            row_path_ids=inputs.row_array("path_id"),
            outer_steps=inputs.row_array("outer_step"),
            phases=inputs.row_array("phase"),
            midpoint_indices=inputs.row_array("midpoint_index"),
            targets=labels.row_array("denoising_target"),
            predictions=predictions,
            expected_path_ids=expected_paths,
            candidate_labels=(f"seed-{seed}-update-{update}",),
            identity_tolerance=IDENTITY_TOLERANCE,
        )
        decompositions.append(decomposition)
        model_record["model_call_batches"] = batch_record
        model_records.append(model_record)
        del predictions, model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    first = decompositions[0]
    arrays = {}
    for name in (
        "cross_terms",
        "prediction_energies",
        "direct_improvements",
        "reconstructed_improvements",
    ):
        arrays[name] = np.ascontiguousarray(
            np.concatenate([getattr(item, name) for item in decompositions], axis=1)
        )
    merged = type(first)(
        path_ids=first.path_ids,
        candidate_labels=tuple(
            f"seed-{seed}-update-{update}" for seed, update, *_ in PRODUCTION_Q0_NOMINEES
        ),
        maximum_identity_error=max(item.maximum_identity_error for item in decompositions),
        **arrays,
    )
    if torch.device(device).type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(torch.device(device)))
        total = int(torch.cuda.get_device_properties(torch.device(device)).total_memory)
        peak_fraction = peak / total
    else:
        peak_fraction = 0.0
    return merged, model_records, peak_fraction, maximum_batch


def _decomposition_rows(role: str, decomposition) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for path_index, path_id in enumerate(decomposition.path_ids.tolist()):
        for candidate, label in enumerate(decomposition.candidate_labels):
            for component, name in enumerate(V3_FAMILY_NAMES):
                path_rows.append(
                    {
                        "role": role,
                        "path_id": path_id,
                        "candidate": label,
                        "component": name,
                        "cross_term": decomposition.cross_terms[path_index, candidate, component],
                        "prediction_energy": decomposition.prediction_energies[path_index, candidate, component],
                        "direct_improvement": decomposition.direct_improvements[path_index, candidate, component],
                        "reconstructed_improvement": decomposition.reconstructed_improvements[
                            path_index, candidate, component
                        ],
                    }
                )

    def append_summary(
        *,
        candidate: int,
        label: str,
        resolution: str,
        names: Sequence[str],
        cross_values: np.ndarray,
        energy_values: np.ndarray,
        direct_values: np.ndarray,
        component_offset: int | None = None,
    ) -> None:
        for index, name in enumerate(names):
            cross = cross_values[:, index]
            energy = energy_values[:, index]
            direct = direct_values[:, index]
            calibration = advisory_scalar_calibration(
                float(np.mean(cross)), float(np.mean(energy))
            )
            summary_rows.append(
                {
                    "role": role,
                    "candidate": label,
                    "resolution": resolution,
                    "component_index": (
                        int(component_offset + index)
                        if component_offset is not None
                        else None
                    ),
                    "component": name,
                    "cross_term": np.mean(cross),
                    "prediction_energy": np.mean(energy),
                    "point_improvement": np.mean(direct),
                    "whole_path_standard_error": np.std(direct, ddof=1)
                    / np.sqrt(direct.size),
                    "scalar_optimum_advisory": calibration["scalar_optimum"],
                    "directional_ceiling_advisory": calibration["directional_ceiling"],
                    "authorizing": int(resolution == "original_228"),
                }
            )

    for candidate, label in enumerate(decomposition.candidate_labels):
        cross = decomposition.cross_terms[:, candidate, :]
        energy = decomposition.prediction_energies[:, candidate, :]
        direct = decomposition.direct_improvements[:, candidate, :]
        append_summary(
            candidate=candidate,
            label=label,
            resolution="original_228",
            names=V3_FAMILY_NAMES,
            cross_values=cross,
            energy_values=energy,
            direct_values=direct,
            component_offset=0,
        )
        fine_cross = cross[:, :224].reshape(-1, 4, 7, 8)
        fine_energy = energy[:, :224].reshape(-1, 4, 7, 8)
        fine_direct = direct[:, :224].reshape(-1, 4, 7, 8)
        derived = (
            (
                "quartile_phase",
                tuple(f"q{q}.phase{phase}" for q in range(4) for phase in range(7)),
                (3,),
            ),
            (
                "quartile_midpoint",
                tuple(f"q{q}.midpoint{mid}" for q in range(4) for mid in range(8)),
                (2,),
            ),
            ("phase", tuple(f"phase{phase}" for phase in range(7)), (1, 3)),
            ("midpoint", tuple(f"midpoint{mid}" for mid in range(8)), (1, 2)),
            ("overall", ("overall",), (1, 2, 3)),
        )
        for resolution, names, axes in derived:
            pooled_cross = np.mean(fine_cross, axis=axes, dtype=np.float64).reshape(
                fine_cross.shape[0], -1
            )
            pooled_energy = np.mean(fine_energy, axis=axes, dtype=np.float64).reshape(
                fine_energy.shape[0], -1
            )
            pooled_direct = np.mean(fine_direct, axis=axes, dtype=np.float64).reshape(
                fine_direct.shape[0], -1
            )
            append_summary(
                candidate=candidate,
                label=label,
                resolution=resolution,
                names=names,
                cross_values=pooled_cross,
                energy_values=pooled_energy,
                direct_values=pooled_direct,
            )
    return path_rows, summary_rows


def _decompose_stage(run_dir: Path, args: argparse.Namespace) -> None:
    if _completed_stage(
        run_dir,
        gate_name="time_local_decomposition_gate.json",
        seal_name="decomposition_artifact_seal.json",
    ):
        return
    _verify_parents(args)
    _verify_stage_seal(run_dir, "preflight_artifact_seal.json")
    _verify_stage_seal(run_dir, "replay_artifact_seal.json")
    plan = _load_json(run_dir / "adjudication_plan.json")
    _verify_semantic(plan, "adjudication plan")
    binding = _load_json(args.parent_memory_v3_run_dir / "immutable_cache_binding.json")
    _verify_semantic(binding, "immutable cache binding")
    cache_root = Path(str(binding["parent_run_dir"])).resolve()
    grid = _candidate_grid(args.parent_memory_v3_run_dir)
    configure_exact_torch_backend(args.device)
    train, train_models, train_peak, train_batch = _decompose_role(
        role="train",
        memory_parent=args.parent_memory_v3_run_dir,
        cache_root=cache_root,
        grid=grid,
        device=args.device,
    )
    validation, validation_models, validation_peak, validation_batch = _decompose_role(
        role="validation",
        memory_parent=args.parent_memory_v3_run_dir,
        cache_root=cache_root,
        grid=grid,
        device=args.device,
    )
    replay = _load_replay(args.parent_memory_v3_run_dir)
    nominee_indices = [nominee.candidate_index for nominee in replay.nominees]
    sealed_validation = np.ascontiguousarray(
        replay.table.path_values[:, nominee_indices, :]
    )
    validation_replay = _compare_sealed_validation_risks(
        current_path_ids=validation.path_ids,
        current_values=validation.direct_improvements,
        sealed_path_ids=replay.table.path_ids,
        sealed_values=sealed_validation,
    )
    validation_risk_replay_valid = bool(validation_replay["passed"])
    atomic_write_json(
        run_dir / "sealed_validation_risk_replay.json",
        {
            "schema": RUN_SCHEMA + "-sealed-validation-risk-replay",
            "schema_version": 1,
            "nominee_candidate_indices": nominee_indices,
            **validation_replay,
            "historical_selection_changed": 0,
            "confirmation_authorized": 0,
            **_scope(),
        },
    )
    _atomic_npz(
        run_dir / "quadratic_decomposition.npz",
        train_path_ids=train.path_ids,
        train_cross_terms=train.cross_terms,
        train_prediction_energies=train.prediction_energies,
        train_direct_improvements=train.direct_improvements,
        train_reconstructed_improvements=train.reconstructed_improvements,
        validation_path_ids=validation.path_ids,
        validation_cross_terms=validation.cross_terms,
        validation_prediction_energies=validation.prediction_energies,
        validation_direct_improvements=validation.direct_improvements,
        validation_reconstructed_improvements=validation.reconstructed_improvements,
    )
    train_path_rows, train_summary = _decomposition_rows("train", train)
    validation_path_rows, validation_summary = _decomposition_rows("validation", validation)
    atomic_write_csv(run_dir / "quadratic_decomposition_per_path.csv", train_path_rows + validation_path_rows)
    atomic_write_csv(run_dir / "quadratic_decomposition_stratified.csv", train_summary + validation_summary)
    lookup = {
        (row["role"], row["candidate"], int(row["component_index"])): row
        for row in train_summary + validation_summary
        if row["component_index"] is not None
    }
    gap_rows = []
    mechanism_rows = []
    forecast_rows = []
    for candidate, label in enumerate(validation.candidate_labels):
        for q in range(4):
            component = 224 + q
            train_row = lookup[("train", label, component)]
            validation_row = lookup[("validation", label, component)]
            gap_rows.append(
                {
                    "candidate": label,
                    "quartile": q,
                    "train_cross_term": train_row["cross_term"],
                    "validation_cross_term": validation_row["cross_term"],
                    "cross_term_gap": train_row["cross_term"]
                    - validation_row["cross_term"],
                    "train_prediction_energy": train_row["prediction_energy"],
                    "validation_prediction_energy": validation_row[
                        "prediction_energy"
                    ],
                    "prediction_energy_gap": train_row["prediction_energy"]
                    - validation_row["prediction_energy"],
                    "train_improvement": train_row["point_improvement"],
                    "validation_improvement": validation_row["point_improvement"],
                    "train_validation_gap": train_row["point_improvement"]
                    - validation_row["point_improvement"],
                    "train_directional_ceiling_advisory": train_row[
                        "directional_ceiling_advisory"
                    ],
                    "validation_directional_ceiling_advisory": validation_row[
                        "directional_ceiling_advisory"
                    ],
                    "directional_ceiling_gap_advisory": (
                        train_row["directional_ceiling_advisory"]
                        - validation_row["directional_ceiling_advisory"]
                        if train_row["directional_ceiling_advisory"] is not None
                        and validation_row["directional_ceiling_advisory"] is not None
                        else None
                    ),
                }
            )
            values = validation.direct_improvements[:, candidate, component]
            required = forecast_required_path_count(
                point_estimate=float(np.mean(values)),
                path_standard_deviation=float(np.std(values, ddof=1)),
                critical_value=replay.result.critical_value,
            )
            forecast_rows.append(
                {
                    "candidate": label,
                    "quartile": q,
                    "point_estimate": np.mean(values),
                    "path_standard_deviation": np.std(values, ddof=1),
                    "required_path_count": required,
                    "infinite_requirement": int(required is None),
                    "authorizing": 0,
                }
            )
    for q in range(4):
        cross = np.mean(validation.cross_terms[:, :, 224 + q], axis=0)
        energy = np.mean(validation.prediction_energies[:, :, 224 + q], axis=0)
        lower = replay.result.lower_bounds[
            [nominee.candidate_index for nominee in replay.nominees], 224 + q
        ]
        mechanism_rows.append(
            {
                "quartile": q,
                "mechanism": classify_quartile_signal(
                    nominee_cross_terms=cross,
                    nominee_prediction_energies=energy,
                    nominee_adjusted_lower_bounds=lower,
                ),
                "median_cross_term": np.median(cross),
                "median_prediction_energy": np.median(energy),
                "median_improvement": np.median(2.0 * cross - energy),
                "original_adjusted_bounds": lower.tolist(),
            }
        )
    atomic_write_csv(run_dir / "train_validation_gap.csv", gap_rows)
    atomic_write_csv(run_dir / "quartile_mechanism_classification.csv", mechanism_rows)
    atomic_write_csv(run_dir / "path_count_forecast.csv", forecast_rows)
    atomic_write_json(
        run_dir / "advisory_scalar_calibration.json",
        {
            "schema": RUN_SCHEMA + "-scalar-calibration",
            "schema_version": 1,
            "records": [
                row
                for row in validation_summary
                if row["component_index"] is not None
                and int(row["component_index"]) >= 224
            ],
            "authorizing": 0,
            "applied_to_checkpoint": 0,
            **_scope(),
        },
    )
    peak = max(train_peak, validation_peak)
    maximum_batch = max(train_batch, validation_batch)
    maximum_identity = max(train.maximum_identity_error, validation.maximum_identity_error)
    metrics = {
        "schema": RUN_SCHEMA + "-decomposition-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "nominee_checkpoint_hashes_valid": 1,
        "input_label_join_valid": 1,
        "sealed_validation_risk_replay_valid": int(
            validation_risk_replay_valid
        ),
        "batch_limit_valid": int(maximum_batch <= MAXIMUM_FORWARD_BATCH),
        "finite_outputs": 1,
        "quadratic_identity_valid": int(maximum_identity <= IDENTITY_TOLERANCE),
        "resource_limit_valid": int(peak <= MAXIMUM_PEAK_MEMORY_FRACTION),
        "confirmation_firewall_valid": 1,
        "maximum_model_forward_batch_size": maximum_batch,
        "maximum_quadratic_identity_error": maximum_identity,
        "peak_cuda_allocation_fraction": peak,
        "train_model_records": train_models,
        "validation_model_records": validation_models,
        "quartile_mechanisms": [row["mechanism"] for row in mechanism_rows],
        **_scope(),
    }
    gate = evaluate_decomposition_gate(metrics)
    atomic_write_json(run_dir / "decomposition_metrics.json", metrics)
    atomic_write_json(run_dir / "time_local_decomposition_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "quadratic_decomposition.npz",
            "sealed_validation_risk_replay.json",
            "quadratic_decomposition_per_path.csv",
            "quadratic_decomposition_stratified.csv",
            "train_validation_gap.csv",
            "quartile_mechanism_classification.csv",
            "path_count_forecast.csv",
            "advisory_scalar_calibration.json",
            "decomposition_metrics.json",
            "time_local_decomposition_gate.json",
        ),
        "decomposition_artifact_seal.json",
    )


def _decision_evidence(run_dir: Path) -> dict[str, Any]:
    replay = _optional_json(run_dir, "sealed_selection_replay.json") or {}
    census = _optional_json(run_dir, "partial_discovery_census.json") or {}
    witness = _optional_json(run_dir, "coarse_witness_quartile_energy.json") or {}
    decompose = _optional_json(run_dir, "decomposition_metrics.json") or {}
    nominees = replay.get("q0_nominees", [])
    mechanisms = list(decompose.get("quartile_mechanisms", []))
    return {
        "q0_nominee_lower_bounds": [row.get("pooled_q0_adjusted_lower_bound") for row in nominees],
        "q0_positive_fine_counts": [row.get("positive_q0_fine_cell_count") for row in nominees],
        "later_adjusted_positive_count": census.get("later_adjusted_positive_count", -1),
        "all_point_positive_candidate_count": census.get("all_point_positive_candidate_count", -1),
        "positive_adjusted_component_count": census.get(
            "positive_simultaneous_candidate_components", -1
        ),
        "coarse_witness_overall_energy": witness.get("overall_energy", 0.0),
        "later_mechanisms": mechanisms[1:] if len(mechanisms) == 4 else (),
        "later_positive_point_count": int(
            sum(1 for value in (mechanisms[1:] if len(mechanisms) == 4 else ()) if value == "positive_but_underpowered")
        ),
    }


def _workflow_record(run_dir: Path, require_gate: str) -> dict[str, Any]:
    preflight = _optional_json(run_dir, "preflight_gate.json")
    replay = _optional_json(run_dir, "time_local_replay_gate.json")
    decomposition = _optional_json(run_dir, "time_local_decomposition_gate.json")
    decision = decide_workflow(
        preflight_gate=preflight,
        replay_gate=replay,
        decomposition_gate=decomposition,
        evidence=_decision_evidence(run_dir),
    )
    workflow = evaluate_required_gate(
        preflight_gate=preflight,
        replay_gate=replay,
        decomposition_gate=decomposition,
        decision=decision,
        require_gate=require_gate,
    )
    decomposition_metrics = _optional_json(run_dir, "decomposition_metrics.json") or {}
    mechanisms = tuple(
        str(value) for value in decomposition_metrics.get("quartile_mechanisms", ())
    )
    decision = {
        **decision,
        "new_training_authorized": 0,
        "fresh_quartile_specialist_planning_authorized": int(
            decision["decision"] == "exact_rb_high_reverse_time_only_signal"
        ),
        "mechanism_conditioned_recommendation_policy": dict(
            MECHANISM_RECOMMENDATIONS
        ),
        "observed_quartile_recommendations": [
            {
                "quartile": quartile,
                "mechanism": mechanism,
                "recommended_next_design": MECHANISM_RECOMMENDATIONS[mechanism],
                "authorizing": 0,
            }
            for quartile, mechanism in enumerate(mechanisms)
            if mechanism in MECHANISM_RECOMMENDATIONS
        ],
        "future_learner_contract": {
            "width_per_quartile": 32,
            "target": "unchanged exact raw Jacobi/Rao-Blackwell label",
            "objective_within_quartile": "plain unweighted MSE",
            "confirmation_authorized": 0,
            "controller_execution_authorized": 0,
            "sampling_authorized": 0,
        },
        **_scope(),
    }
    workflow["decision"] = decision
    workflow.update(_scope())
    atomic_write_json(run_dir / "time_local_adjudication_decision.json", decision)
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    return workflow


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "replay", "decompose")
    if stage == "report":
        return ()
    return (stage,)


def _commit_failed_stage_gate(
    run_dir: Path,
    *,
    stage: str,
    failure: Mapping[str, Any],
) -> None:
    """Commit the named failed gate before producing a terminal decision."""

    definitions = {
        "preflight": (
            "preflight_metrics.json",
            "preflight_gate.json",
            "preflight_artifact_seal.json",
            evaluate_preflight_gate,
        ),
        "replay": (
            "replay_metrics.json",
            "time_local_replay_gate.json",
            "replay_artifact_seal.json",
            evaluate_replay_gate,
        ),
        "decompose": (
            "decomposition_metrics.json",
            "time_local_decomposition_gate.json",
            "decomposition_artifact_seal.json",
            evaluate_decomposition_gate,
        ),
    }
    if stage not in definitions:
        return
    metrics_name, gate_name, seal_name, evaluator = definitions[stage]
    gate_path = run_dir / gate_name
    if gate_path.is_file() and _passed(_load_json(gate_path)):
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
    gate = evaluator(metrics)
    atomic_write_json(run_dir / metrics_name, metrics)
    atomic_write_json(gate_path, gate)
    _seal_stage(
        run_dir,
        (f"{stage}_execution_failure.json", metrics_name, gate_name),
        seal_name,
    )


def _verify_report(run_dir: Path) -> None:
    for gate, seal in (
        ("preflight_gate.json", "preflight_artifact_seal.json"),
        ("time_local_replay_gate.json", "replay_artifact_seal.json"),
        ("time_local_decomposition_gate.json", "decomposition_artifact_seal.json"),
    ):
        if (run_dir / gate).is_file():
            _verify_stage_seal(run_dir, seal)
    _verify_registered_prefix(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-memory-v3-run-dir", type=Path, required=True)
    parser.add_argument("--parent-coarse-witness-run-dir", type=Path, required=True)
    parser.add_argument("--parent-bayes-power-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication"),
    )
    parser.add_argument("--run-name", default="production-v3-time-local-adjudication")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    for name in (
        "parent_memory_v3_run_dir",
        "parent_coarse_witness_run_dir",
        "parent_bayes_power_run_dir",
        "resume_run_dir",
        "runs_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.resume_run_dir is None and args.stage in {"replay", "decompose", "report"}:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    expected = {
        "preflight": "preflight",
        "replay": "replay",
        "decompose": "decompose",
        "report": "none",
        "all": "decompose",
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
        print(f"v3 time-local adjudication run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        initialized = True
        _status(run_dir, state="running", stage=args.stage)
        if args.stage == "report":
            _verify_report(run_dir)
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
                if not _passed(_optional_json(run_dir, "time_local_replay_gate.json")):
                    raise ArtifactCompatibilityError("decomposition requires passing replay")
                _decompose_stage(run_dir, args)
        workflow = _workflow_record(run_dir, args.require_gate)
        decision = str(workflow["decision"]["decision"])
        required_pass = int(workflow["required_gate_pass"]) == 1
        _status(
            run_dir,
            state="complete" if required_pass else "gate_failed",
            stage=args.stage,
            decision=decision,
            failure_domain=None if required_pass else "scientific_gate",
            failure_code=None if required_pass else f"{args.require_gate}_gate_failed",
            scientific_evidence_complete=int(
                workflow["decision"].get("scientific_evidence_complete", 0)
            ),
        )
        _artifact_registry(run_dir)
        print(f"v3 time-local adjudication decision: {decision}", flush=True)
        return 0 if required_pass else 2
    except KeyboardInterrupt:
        if run_dir is not None and initialized:
            _status(
                run_dir,
                state="interrupted",
                stage=active_stage,
                message="interrupted; resume the same run directory",
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
                    getattr(exc, "failure_code", "time_local_adjudication_execution_failed")
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
                **_scope(),
            }
            atomic_write_json(run_dir / f"{active_stage}_execution_failure.json", failure)
            _commit_failed_stage_gate(
                run_dir,
                stage=active_stage,
                failure=failure,
            )
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
        print(f"v3 time-local adjudication error: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
