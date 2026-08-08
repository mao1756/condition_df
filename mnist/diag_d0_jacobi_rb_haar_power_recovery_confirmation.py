"""Recover immutable Haar panel A and run its sealed antithetic fallback.

The workflow is controls-only.  It reads the completed nested-Haar panel from
an immutable failed parent, reconstructs the originally preregistered
no-nomination decision, and only then permits the untouched pairwise-
antithetic A/B panels to execute.  It never trains a model, performs a
production refinement experiment, or samples a reverse process.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    canonical_json_bytes,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_haar_gate import (
    ANTITHETIC_HAAR_PROFILE,
    NESTED_HAAR_PROFILE,
    nominate_haar_power_design,
)
from mnist.d0_jacobi_rb_haar_power import (
    FORBIDDEN_COUNTS,
    combine_certified_haar_power_panels,
    panel_confirmation_record,
)
from mnist.d0_jacobi_rb_haar_power_recovery import (
    HaarPowerRecoveryError,
    replay_nested_panel_a,
    run_recovery_antithetic_panel,
)
from mnist.d0_jacobi_rb_haar_power_recovery_gate import (
    HaarPowerRecoveryThresholds,
    evaluate_antithetic_pilot,
    evaluate_nested_replay,
    evaluate_recovery_preflight,
    evaluate_recovery_workflow,
    execution_failed_gate,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_haar_power_recovery_provenance import (
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_RUN_BASENAME,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_COUNT,
    PARENT_SOURCE_FINGERPRINT,
    verify_haar_power_recovery_parent,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-haar-power-recovery-confirmation"
RUN_SCHEMA_VERSION = 1
ROOT_SEED = 261_181
PANEL_CLUSTERS = 8
NO_WORK = {
    "physical_training_performed": 0,
    "production_refinement_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}
_REGISTRY_EXCLUDED = {"artifact_registry.json", "run_status.json"}
_MUTABLE_TERMINAL_FILES = {
    "haar_power_recovery_workflow_gate.json",
    "haar_power_recovery_decision.json",
}


class HaarPowerRecoveryCLIError(RuntimeError):
    """A recovery stage could not produce complete authorizing evidence."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "recovery_orchestration",
        failure_code: str = "haar_power_recovery_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


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


def _normalized(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _freeze(
    path: Path,
    value: Mapping[str, Any],
    *,
    require_existing: bool = False,
) -> dict[str, Any]:
    record = _normalized(value)
    if path.is_file():
        if _load(path) != record:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    elif require_existing:
        raise ArtifactCompatibilityError(f"resume lacks frozen artifact: {path.name}")
    else:
        atomic_write_json(path, record)
    return record


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    normalized = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in sorted(arrays.items())
    }
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **normalized)
    os.replace(temporary, path)
    return {
        "path": path.name,
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
        "array_hashes": {
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in normalized.items()
        },
    }


def _freeze_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    normalized = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in sorted(arrays.items())
    }
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as archive:
                valid = set(archive.files) == set(normalized) and all(
                    np.array_equal(np.asarray(archive[name]), value)
                    for name, value in normalized.items()
                )
        except (OSError, ValueError):
            valid = False
        if not valid:
            raise ArtifactCompatibilityError(f"frozen NPZ changed: {path.name}")
        return {
            "path": path.name,
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
            "array_hashes": {
                name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                for name, value in normalized.items()
            },
        }
    return _atomic_npz(path, normalized)


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "replay", "pilot", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-gate",
        choices=("none", "preflight", "replay", "pilot"),
        default="none",
    )
    parser.add_argument("--parent-haar-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_haar_power_recovery_confirmation"
        ),
    )
    parser.add_argument("--run-name", default="production-haar-power-recovery")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    parser.add_argument(
        "--panel-clusters",
        type=int,
        default=PANEL_CLUSTERS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--test-only-reduced-workload",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.stage in {"replay", "pilot", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "replay": {"none", "preflight", "replay"},
        "pilot": {"none", "preflight", "replay", "pilot"},
        "report": {"none", "preflight", "replay", "pilot"},
        "all": {"none", "preflight", "replay", "pilot"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(
            f"--require-gate {args.require_gate} is unavailable at stage {args.stage}"
        )
    changed: list[str] = []
    if int(args.root_seed) != ROOT_SEED:
        changed.append("root_seed")
    if int(args.panel_clusters) != PANEL_CLUSTERS:
        changed.append("panel_clusters")
    if changed and not args.test_only_reduced_workload:
        parser.error(
            "production configuration is frozen; overrides require "
            "--test-only-reduced-workload: " + ", ".join(changed)
        )
    if args.test_only_reduced_workload and args.require_gate != "none":
        parser.error("test-only workloads cannot satisfy a required gate")
    if not 1 <= int(args.panel_clusters) <= PANEL_CLUSTERS:
        parser.error("panel clusters must lie in [1,8]")
    if (
        torch.device(args.device).type != "cuda"
        and not args.test_only_reduced_workload
    ):
        parser.error("production antithetic controls require --device cuda")
    return args


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.run_name}"
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
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        if (
            (run_dir / "run_status.json").is_file()
            and _load(run_dir / "run_status.json").get("status") == "complete"
        ):
            raise ArtifactCompatibilityError(
                "completed resume lacks terminal artifact registry"
            )
        return
    status = _load(run_dir / "run_status.json")
    registry = _load(path)
    records = registry.get("records")
    if (
        registry.get("schema") != RUN_SCHEMA + "-artifact-registry"
        or registry.get("schema_version") != 1
        or set(
            registry.get("terminal_files_excluded_to_avoid_self_reference", ())
        )
        != _REGISTRY_EXCLUDED
        or not isinstance(records, Mapping)
        or status.get("artifact_registry_sha256") != file_fingerprint(path)
        or int(status.get("artifact_registry_record_count", -1)) != len(records)
        or int(status.get("artifact_registry_size", -1)) != path.stat().st_size
    ):
        raise ArtifactCompatibilityError("resume terminal registry is invalid")
    interrupted = status.get("status") == "running"
    for relative, raw in records.items():
        artifact = run_dir / str(relative)
        valid = (
            artifact.is_file()
            and isinstance(raw, Mapping)
            and raw.get("sha256") == file_fingerprint(artifact)
            and int(raw.get("size", -1)) == artifact.stat().st_size
        )
        if not valid and interrupted and relative in _MUTABLE_TERMINAL_FILES:
            continue
        if not valid:
            raise ArtifactCompatibilityError(f"resume artifact changed: {relative}")
    actual = {
        artifact.relative_to(run_dir).as_posix()
        for artifact in run_dir.rglob("*")
        if artifact.is_file()
        and artifact.relative_to(run_dir).as_posix() not in _REGISTRY_EXCLUDED
    }
    if not interrupted and actual != set(records):
        raise ArtifactCompatibilityError(
            "completed resume artifact set differs from terminal registry"
        )


def _finalize_registry(run_dir: Path) -> dict[str, Any]:
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    path = run_dir / "artifact_registry.json"
    return {
        "artifact_registry_record_count": len(registry["records"]),
        "artifact_registry_sha256": file_fingerprint(path),
        "artifact_registry_size": int(path.stat().st_size),
    }


def _source_record(parent_dir: Path) -> tuple[str, list[str]]:
    manifest = _load(parent_dir / "run_manifest.json")
    raw = manifest.get("source_paths")
    if (
        not isinstance(raw, list)
        or len(raw) != PARENT_SOURCE_COUNT
        or manifest.get("source_fingerprint") != PARENT_SOURCE_FINGERPRINT
    ):
        raise ArtifactCompatibilityError("parent source manifest changed")
    parent_paths = [Path(str(value)).resolve() for value in raw]
    if (
        not all(path.is_file() for path in parent_paths)
        or source_fingerprint(parent_paths) != PARENT_SOURCE_FINGERPRINT
    ):
        raise ArtifactCompatibilityError("one of the 35 parent sources changed")
    modules = (
        "mnist.d0_jacobi_rb_haar_power_recovery_provenance",
        "mnist.d0_jacobi_rb_haar_power_recovery",
        "mnist.d0_jacobi_rb_haar_power_recovery_gate",
        "mnist.diag_d0_jacobi_rb_haar_power_recovery_confirmation",
    )
    additions = [
        Path(str(importlib.import_module(name).__file__)).resolve()
        for name in modules
    ]
    ordered = sorted({*parent_paths, *additions}, key=lambda value: value.as_posix())
    return source_fingerprint(ordered), [str(path) for path in ordered]


def _scientific_config(
    args: argparse.Namespace,
    *,
    path_plan: Mapping[str, Any],
    profile_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "parent_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_registry_record_count": PARENT_REGISTRY_RECORD_COUNT,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "root_seed": int(args.root_seed),
        "panel_clusters": int(args.panel_clusters),
        "path_id_plan_sha256": path_plan.get("path_id_plan_sha256"),
        "profile_plan_sha256": profile_plan.get("profile_plan_sha256"),
        "nested_parent_replay_only": 1,
        "antithetic_profile": ANTITHETIC_HAAR_PROFILE,
        "test_only_reduced_workload": int(args.test_only_reduced_workload),
        "thresholds": HaarPowerRecoveryThresholds().to_dict(),
        **NO_WORK,
    }


def _existing_gate(run_dir: Path, stage: str, reason: str) -> dict[str, Any]:
    path = run_dir / f"haar_power_recovery_{stage}_gate.json"
    return _load(path) if path.is_file() else not_evaluated_gate(stage, reason)


def _typed_failure(
    error: BaseException,
    *,
    default_domain: str,
    default_code: str,
) -> tuple[str, str]:
    return (
        str(getattr(error, "failure_domain", default_domain)),
        str(getattr(error, "failure_code", default_code)),
    )


def _write_stage_failure(
    run_dir: Path,
    stage: str,
    error: BaseException,
    *,
    default_domain: str,
    default_code: str,
) -> dict[str, Any]:
    domain, code = _typed_failure(
        error,
        default_domain=default_domain,
        default_code=default_code,
    )
    gate = execution_failed_gate(
        stage,
        failure_domain=domain,
        failure_code=code,
        error_type=type(error).__name__,
        error=str(error),
    )
    atomic_write_json(run_dir / f"{stage}_failure.json", gate)
    atomic_write_json(
        run_dir / f"haar_power_recovery_{stage}_gate.json",
        gate,
    )
    return gate


def _finish(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    provenance: Mapping[str, Any] | bool,
    preflight: Mapping[str, Any],
    replay: Mapping[str, Any],
    pilot: Mapping[str, Any],
) -> int:
    workflow = evaluate_recovery_workflow(
        provenance=provenance,
        preflight_gate=preflight,
        replay_gate=replay,
        pilot_gate=pilot,
        require_gate=args.require_gate,
    )
    decision = dict(workflow["decision"])
    atomic_write_json(
        run_dir / "haar_power_recovery_workflow_gate.json", workflow
    )
    atomic_write_json(run_dir / "haar_power_recovery_decision.json", decision)
    registry = _finalize_registry(run_dir)
    passed = bool(workflow["required_gate_pass"])
    _write_status(
        run_dir,
        status="complete",
        outcome="complete" if passed else "gate_failed",
        phase=args.stage,
        required_gate=args.require_gate,
        required_gate_pass=int(passed),
        decision=decision.get("decision"),
        decision_status=decision.get("evaluation_status"),
        **registry,
    )
    return 0 if passed else 1


def _write_candidate_csv(path: Path, candidates: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "main_paths",
        "reference_paths",
        "predicted_main_half_width",
        "predicted_generator_reference_half_width",
        "predicted_reference_stability_half_width",
        "projected_hours",
        "conservative_rate",
        "eligible",
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({name: candidate.get(name) for name in fields})
    os.replace(temporary, path)


def _manifest(
    args: argparse.Namespace,
    *,
    source_hash: str,
    source_paths: Sequence[str],
    config_sha256: str,
    path_plan: Mapping[str, Any],
    profile_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "claim_scope": (
            "immutable nested-panel recovery and sealed antithetic "
            "refinement-power feasibility only"
        ),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_record_count": PARENT_REGISTRY_RECORD_COUNT,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "source_fingerprint": source_hash,
        "source_paths": list(source_paths),
        "source_count": len(source_paths),
        "scientific_config_sha256": config_sha256,
        "path_id_plan_sha256": path_plan.get("path_id_plan_sha256"),
        "profile_plan_sha256": profile_plan.get("profile_plan_sha256"),
        "requested_device": str(args.device),
        "root_seed": int(args.root_seed),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "test_only_reduced_workload": int(args.test_only_reduced_workload),
        "parent_nested_shards_recomputed": 0,
        **NO_WORK,
    }


def _forbidden_count(execution: Mapping[str, Any]) -> int:
    return sum(int(execution.get(name, 0)) for name in FORBIDDEN_COUNTS)


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _candidate_for_design(
    candidates: Sequence[Mapping[str, Any]],
    *,
    main_paths: int = 16,
    reference_paths: int = 16,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in candidates
        if int(row.get("main_paths", -1)) == int(main_paths)
        and int(row.get("reference_paths", -1)) == int(reference_paths)
    ]
    if len(rows) != 1:
        raise ArtifactCompatibilityError(
            "sealed antithetic panel lacks the frozen 16/16 candidate"
        )
    return rows[0]


def _confirmation_pass(record: Mapping[str, Any]) -> bool:
    t = HaarPowerRecoveryThresholds()
    required = (
        "complete_pass",
        "finite_pass",
        "certification_pass",
        "numerical_health_pass",
        "mass_conservation_pass",
        "shard_chain_pass",
        "production_authorizing_pass",
        "raw_endpoint_authorizing_pass",
        "dynkin_advisory_only_pass",
        "independent_pool_variance_pass",
        "richardson_formula_pass",
        "pilot_production_isolation_pass",
    )
    return bool(
        all(int(record.get(name, 0)) == 1 for name in required)
        and _finite_float(record.get("main_half_width"), math.inf)
        <= t.maximum_main_half_width
        and _finite_float(
            record.get("generator_reference_half_width"), math.inf
        )
        <= t.maximum_reference_half_width
        and _finite_float(
            record.get("reference_stability_half_width"), math.inf
        )
        <= t.maximum_reference_half_width
        and _finite_float(record.get("projected_hours"), math.inf)
        <= t.maximum_projected_hours
        and _finite_float(record.get("minimum_rate"), 0.0) >= t.minimum_rate
        and _finite_float(record.get("certificate_fraction"), 0.0) == 1.0
        and _finite_float(record.get("fallback_fraction"), math.inf)
        <= t.maximum_fallback_fraction
        and _finite_float(record.get("fallback_cost_fraction"), math.inf)
        <= t.maximum_fallback_cost_fraction
        and _finite_float(record.get("peak_memory_fraction"), math.inf)
        <= t.maximum_peak_memory_fraction
        and _finite_float(record.get("mass_error"), math.inf)
        <= t.maximum_mass_error
        and _forbidden_count(record) == 0
    )


def _static_parent_artifacts(parent_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "path_plan": _load(parent_dir / "haar_path_id_plan.json"),
        "profile_plan": _load(parent_dir / "haar_profile_plan.json"),
        "sealed_registry": _load(parent_dir / "sealed_panel_registry.json"),
        "model_input_contract": _load(
            parent_dir / "future_model_input_contract.json"
        ),
    }


def _corrected_parent_adjudication(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-corrected-parent-adjudication",
        "schema_version": 1,
        "parent_run_dir": provenance.get("parent_run_dir"),
        "parent_recorded_decision": provenance.get("parent_decision"),
        "parent_failure_code": provenance.get("parent_failure_code"),
        "parent_failure_message": provenance.get("parent_failure_message"),
        "corrected_adjudication": "panel_schedule_binding_invalid",
        "completed_nested_main_shards": int(
            provenance.get("parent_nested_main_shard_count", -1)
        ),
        "completed_nested_reference_shards": int(
            provenance.get("parent_nested_reference_shard_count", -1)
        ),
        "completed_nested_shards": int(
            provenance.get("parent_nested_shard_count", -1)
        ),
        "canonical_schedule_location": "identity.schedule",
        "invalid_legacy_aggregator_location": "record.schedule",
        "nested_transitions_recomputed": 0,
        "parent_mutated": 0,
        **NO_WORK,
    }


def _initialize_or_verify_run(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    resumed: bool,
    parent_dir: Path,
    provenance: Mapping[str, Any],
    source_hash: str,
    source_paths: Sequence[str],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    path_plan = artifacts["path_plan"]
    profile_plan = artifacts["profile_plan"]
    sealed_registry = artifacts["sealed_registry"]
    scientific_config = _scientific_config(
        args,
        path_plan=path_plan,
        profile_plan=profile_plan,
    )
    scientific_config.update(
        {
            "sealed_panel_registry_sha256": file_fingerprint(
                parent_dir / "sealed_panel_registry.json"
            ),
            "model_input_contract_sha256": file_fingerprint(
                parent_dir / "future_model_input_contract.json"
            ),
        }
    )
    config_sha256 = config_fingerprint(scientific_config)
    manifest = _manifest(
        args,
        source_hash=source_hash,
        source_paths=source_paths,
        config_sha256=config_sha256,
        path_plan=path_plan,
        profile_plan=profile_plan,
    )
    manifest.update(
        {
            "sealed_panel_registry_sha256": scientific_config[
                "sealed_panel_registry_sha256"
            ],
            "model_input_contract_sha256": scientific_config[
                "model_input_contract_sha256"
            ],
            "plans_frozen_before_device_execution": 1,
        }
    )
    frozen = {
        "haar_path_id_plan.json": path_plan,
        "haar_profile_plan.json": profile_plan,
        "sealed_panel_registry.json": sealed_registry,
        "future_model_input_contract.json": artifacts[
            "model_input_contract"
        ],
        "scientific_config.json": scientific_config,
        "parent_provenance.json": provenance,
        "corrected_parent_adjudication.json": (
            _corrected_parent_adjudication(provenance)
        ),
        "run_manifest.json": manifest,
    }
    for name, record in frozen.items():
        _freeze(
            run_dir / name,
            record,
            require_existing=resumed,
        )
    runtime_path = run_dir / "exact_backend_runtime.json"
    if args.stage == "report":
        if not runtime_path.is_file():
            raise ArtifactCompatibilityError(
                "report-only recovery lacks exact_backend_runtime.json"
            )
        runtime = _load(runtime_path)
    else:
        runtime = {
            "schema": RUN_SCHEMA + "-exact-backend-runtime",
            "schema_version": 1,
            "requested_device": str(args.device),
            "exact_backend": configure_exact_torch_backend(
                torch.device(args.device)
            ),
            **NO_WORK,
        }
        _freeze(runtime_path, runtime, require_existing=resumed)
    return runtime, config_sha256


def _preflight_metrics(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "control_provenance_pass": int(provenance.get("passed", 0)),
        "parent_registry_verified_pass": int(
            provenance.get("parent_registry_hash_pass", 0)
            and provenance.get("parent_all_artifact_hashes_pass", 0)
        ),
        "parent_sources_immutable_pass": int(
            provenance.get("parent_sources_immutable_pass", 0)
        ),
        "parent_scientific_config_pass": int(
            provenance.get("parent_scientific_config_pass", 0)
        ),
        "parent_preflight_pass": int(
            provenance.get("parent_preflight_pass", 0)
        ),
        "parent_coupling_pass": int(
            provenance.get("parent_coupling_pass", 0)
        ),
        "parent_pilot_execution_failure_pass": int(
            provenance.get("parent_pilot_execution_failed_pass", 0)
        ),
        "parent_failure_code_pass": int(
            provenance.get("parent_failure_code")
            == "hierarchical_panel_diagnostics_invalid"
        ),
        "parent_shard_layout_pass": int(
            provenance.get("parent_nested_main_shard_count") == 16
            and provenance.get("parent_nested_reference_shard_count") == 64
            and provenance.get("parent_nested_shard_count") == 80
        ),
        "parent_shard_hashes_pass": int(
            provenance.get("parent_shard_hash_pass", 0)
        ),
        "parent_shard_chains_pass": int(
            provenance.get("parent_shard_chain_pass", 0)
            and provenance.get("parent_checkpoint_chain_pass", 0)
        ),
        "parent_schedule_location_pass": int(
            provenance.get("parent_identity_schedule_pass", 0)
            and provenance.get("parent_top_level_schedule_absent_pass", 0)
            and provenance.get("parent_schedule_binding_pass", 0)
        ),
        "parent_antithetic_absent_pass": int(
            provenance.get("parent_no_antithetic_power_shards_pass", 0)
        ),
        "parent_panel_b_absent_pass": int(
            provenance.get("parent_panel_b_absent_pass", 0)
        ),
        "parent_selection_absent_pass": int(
            provenance.get("parent_selection_absent_pass", 0)
        ),
        "parent_no_work_pass": int(
            provenance.get("parent_no_work_pass", 0)
        ),
        "transitive_provenance_pass": int(
            provenance.get("parent_transitive_provenance_pass", 0)
        ),
        "path_plan_frozen_pass": int(
            provenance.get("parent_path_id_plan_pass", 0)
            and provenance.get("parent_frozen_plans_pass", 0)
        ),
        "root_seed": provenance.get("parent_root_seed"),
        "parent_registry_record_count": provenance.get(
            "parent_artifact_record_count"
        ),
        "parent_source_count": provenance.get("parent_source_count"),
        "parent_main_shards": provenance.get(
            "parent_nested_main_shard_count"
        ),
        "parent_reference_shards": provenance.get(
            "parent_nested_reference_shard_count"
        ),
    }


def _run_preflight_stage(
    run_dir: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    gate_path = run_dir / "haar_power_recovery_preflight_gate.json"
    metrics_path = run_dir / "haar_power_recovery_preflight_metrics.json"
    if gate_path.is_file() and metrics_path.is_file():
        metrics_record = _load(metrics_path)
        gate = _load(gate_path)
        expected = evaluate_recovery_preflight(metrics_record["metrics"])
        if gate != expected:
            raise ArtifactCompatibilityError(
                "frozen recovery preflight gate changed"
            )
        return gate
    shard_audit = provenance.get("parent_shard_audit")
    schedule_bindings = provenance.get("canonical_schedule_bindings")
    if not isinstance(shard_audit, Mapping) or not isinstance(
        schedule_bindings, list
    ):
        raise ArtifactCompatibilityError(
            "verified parent lacks shard audit or schedule bindings"
        )
    atomic_write_json(
        run_dir / "parent_shard_audit.json",
        {
            "schema": RUN_SCHEMA + "-parent-shard-audit",
            "schema_version": 1,
            "audit": dict(shard_audit),
            **NO_WORK,
        },
    )
    atomic_write_json(
        run_dir / "canonical_parent_schedule_table.json",
        {
            "schema": RUN_SCHEMA + "-canonical-schedule-table",
            "schema_version": 1,
            "binding_source": "identity.schedule",
            "invalid_legacy_source": "record.schedule",
            "binding_count": len(schedule_bindings),
            "bindings": schedule_bindings,
            **NO_WORK,
        },
    )
    metrics = _preflight_metrics(provenance)
    atomic_write_json(
        metrics_path,
        {
            "schema": RUN_SCHEMA + "-preflight-metrics",
            "schema_version": 1,
            "metrics": metrics,
            **NO_WORK,
        },
    )
    gate = evaluate_recovery_preflight(metrics)
    atomic_write_json(gate_path, gate)
    return gate


def _replay_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(record.get("execution", {}))
    candidates = list(record.get("candidates", ()))
    nomination = dict(record.get("nomination", {}))
    shard_audit = dict(record.get("shard_audit", {}))
    numerical_flags = (
        "panel_complete_pass",
        "panel_finite_pass",
        "panel_certification_pass",
        "panel_numerical_health_pass",
        "mass_conservation_pass",
        "shard_chain_pass",
        "pilot_production_isolation_pass",
        "pilot_means_excluded_pass",
        "raw_endpoint_authorizing_pass",
        "dynkin_advisory_only_pass",
    )
    resource_valid = bool(
        candidates
        and all(
            _finite_float(row.get("projected_hours"), math.inf) <= 48.0
            and _finite_float(row.get("conservative_rate"), 0.0) >= 1300.0
            for row in candidates
        )
    )
    return {
        "canonical_schedule_binding_pass": int(
            record.get("schedule_binding_pass", 0)
            and execution.get("canonical_schedule_binding_pass", 0)
        ),
        "parent_read_only_pass": int(
            shard_audit.get("parent_mutated", 1) == 0
        ),
        "no_nested_gpu_recomputation_pass": int(
            shard_audit.get("nested_gpu_recomputation_performed", 1) == 0
        ),
        "observable_replay_pass": int(
            record.get("complete", 0) == 1
            and record.get("finite", 0) == 1
        ),
        "candidate_reconstruction_pass": int(len(candidates) == 4),
        "candidate_numerical_health_pass": int(
            len(candidates) == 4
            and all(
                all(int(row.get(name, 0)) == 1 for name in numerical_flags)
                for row in candidates
            )
        ),
        "candidate_resource_health_pass": int(resource_valid),
        "frozen_no_nominee_pass": int(
            record.get("recovery_decision")
            == "panel_a_no_eligible_design"
            and nomination.get("selected") is None
        ),
        "raw_endpoint_authorizing_pass": int(
            record.get("raw_endpoint_authorizing_pass", 0)
        ),
        "dynkin_advisory_only_pass": int(
            record.get("dynkin_advisory_only_pass", 0)
        ),
        "shard_chain_pass": int(
            shard_audit.get("predecessor_chain_pass", 0)
            and shard_audit.get("archive_hash_pass", 0)
        ),
        "state_updates_device_resident_pass": int(
            execution.get("state_updates_device_resident_pass", 0)
        ),
        "antithetic_path_ids_untouched_pass": 1,
        "shard_count": shard_audit.get("total_shard_count"),
        "transition_count": execution.get("transition_count"),
        "certificate_fraction": execution.get("certificate_fraction"),
        "fallback_count": execution.get("fallback_count"),
        "fallback_fraction": execution.get("fallback_fraction"),
        "fallback_cost_fraction": execution.get("fallback_cost_fraction"),
        "mass_error": execution.get("mass_error"),
        "peak_memory_fraction": execution.get("peak_memory_fraction"),
        "conservative_rate": execution.get("conservative_rate"),
        "candidate_count": len(candidates),
        "eligible_candidate_count": nomination.get(
            "eligible_candidate_count"
        ),
        "selection_status": nomination.get("selection_status"),
        "uncertified_count": execution.get("uncertified_count"),
        "forbidden_event_count": _forbidden_count(execution),
    }


def _run_replay_stage(
    run_dir: Path,
    parent_dir: Path,
) -> dict[str, Any]:
    gate_path = run_dir / "haar_power_recovery_replay_gate.json"
    metrics_path = run_dir / "haar_power_recovery_replay_metrics.json"
    if gate_path.is_file() and metrics_path.is_file():
        metrics_record = _load(metrics_path)
        gate = _load(gate_path)
        expected = evaluate_nested_replay(metrics_record["metrics"])
        if gate != expected:
            raise ArtifactCompatibilityError("frozen nested replay gate changed")
        return gate
    replay = replay_nested_panel_a(parent_dir)
    arrays = replay.pop("observable_arrays")
    payload = _freeze_npz(
        run_dir / "recovered_nested_panel_a_observables.npz",
        arrays,
    )
    schedule_bindings = replay.pop("schedule_bindings")
    shard_audit = replay.pop("shard_audit")
    replay_candidates = replay.pop("candidates")
    nomination = replay.pop("nomination")
    nominated_candidates = nomination.get("candidates")
    candidates = (
        [dict(row) for row in nominated_candidates]
        if isinstance(nominated_candidates, list)
        else replay_candidates
    )
    replay.update(
        {
            "schedule_bindings_artifact": (
                "recovered_nested_schedule_bindings.json"
            ),
            "shard_audit_artifact": "recovered_nested_shard_audit.json",
            "candidate_artifact": "recovered_nested_candidates.json",
            "nomination_artifact": "recovered_nested_nomination.json",
            "observable_payload": payload,
        }
    )
    atomic_write_json(
        run_dir / "recovered_nested_schedule_bindings.json",
        {
            "schema": RUN_SCHEMA + "-recovered-schedule-bindings",
            "schema_version": 1,
            "binding_count": len(schedule_bindings),
            "bindings": schedule_bindings,
            **NO_WORK,
        },
    )
    atomic_write_json(
        run_dir / "recovered_nested_shard_audit.json",
        {
            "schema": RUN_SCHEMA + "-recovered-shard-audit",
            "schema_version": 1,
            **dict(shard_audit),
            **NO_WORK,
        },
    )
    atomic_write_json(
        run_dir / "recovered_nested_candidates.json",
        {
            "schema": RUN_SCHEMA + "-recovered-candidates",
            "schema_version": 1,
            "candidates": candidates,
            **NO_WORK,
        },
    )
    _write_candidate_csv(run_dir / "recovered_nested_candidates.csv", candidates)
    atomic_write_json(
        run_dir / "recovered_nested_nomination.json",
        {
            **dict(nomination),
            "immutable_parent_evidence_replayed": 1,
            **NO_WORK,
        },
    )
    replay_for_metrics = {
        **replay,
        "schedule_bindings": schedule_bindings,
        "shard_audit": shard_audit,
        "candidates": replay_candidates,
        "nomination": nomination,
    }
    atomic_write_json(
        run_dir / "recovered_nested_panel_a.json",
        replay,
    )
    metrics = _replay_metrics(replay_for_metrics)
    atomic_write_json(
        metrics_path,
        {
            "schema": RUN_SCHEMA + "-replay-metrics",
            "schema_version": 1,
            "metrics": metrics,
            "observable_payload_sha256": payload["sha256"],
            **NO_WORK,
        },
    )
    gate = evaluate_nested_replay(metrics)
    atomic_write_json(gate_path, gate)
    return gate


def _panel_metrics(
    *,
    panel_a: Mapping[str, Any],
    nomination: Mapping[str, Any],
    panel_b: Mapping[str, Any] | None,
    panel_b_confirmation: Mapping[str, Any] | None,
    combined: Mapping[str, Any] | None,
    sealed_registry: Mapping[str, Any],
) -> dict[str, Any]:
    selected = nomination.get("selected")
    a_execution = dict(panel_a.get("execution", {}))
    a_candidate = _candidate_for_design(panel_a.get("candidates", ()))
    authorizing = dict(combined or panel_b_confirmation or {})
    authorizing_execution: Mapping[str, Any]
    if combined is not None:
        authorizing_execution = authorizing
    elif panel_b is not None:
        authorizing_execution = dict(panel_b.get("execution", {}))
    else:
        authorizing_execution = a_execution
    panel_records = [panel_a] + ([panel_b] if panel_b is not None else [])
    confirmation_records = (
        [record for record in (panel_b_confirmation, combined) if record is not None]
    )
    numerically_valid = bool(
        panel_records
        and all(
            int(record.get("complete", 0)) == 1
            and int(record.get("finite", 0)) == 1
            and int(record.get("production_authorizing_pass", 0)) == 1
            and _forbidden_count(dict(record.get("execution", {}))) == 0
            for record in panel_records
        )
        and all(
            int(record.get("complete_pass", 0)) == 1
            and int(record.get("finite_pass", 0)) == 1
            and int(record.get("numerical_health_pass", 0)) == 1
            for record in confirmation_records
        )
    )
    if combined is not None:
        width_source = combined
    else:
        width_source = {
            "main_half_width": a_candidate.get("predicted_main_half_width"),
            "generator_reference_half_width": a_candidate.get(
                "predicted_generator_reference_half_width"
            ),
            "reference_stability_half_width": a_candidate.get(
                "predicted_reference_stability_half_width"
            ),
            "projected_hours": a_candidate.get("projected_hours"),
            "minimum_rate": a_candidate.get("conservative_rate"),
        }
    b_pass = (
        _confirmation_pass(panel_b_confirmation)
        if panel_b_confirmation is not None
        else False
    )
    combined_pass = _confirmation_pass(combined) if combined is not None else False
    forbidden = _forbidden_count(authorizing_execution)
    return {
        "plans_frozen_pass": int(
            sealed_registry.get("panels_frozen_before_device_execution") == 1
        ),
        "profile_order_pass": int(
            sealed_registry.get("profile_order")
            == [NESTED_HAAR_PROFILE, ANTITHETIC_HAAR_PROFILE]
        ),
        "panel_nonregeneration_pass": int(
            sealed_registry.get("panel_regeneration_permitted") == 0
        ),
        "no_fallback_after_panel_b_pass": 1,
        "raw_endpoint_authorizing_pass": int(
            all(
                int(record.get("raw_endpoint_authorizing_pass", 0)) == 1
                for record in panel_records
            )
        ),
        "dynkin_advisory_only_pass": int(
            all(
                int(record.get("dynkin_advisory_only_pass", 0)) == 1
                for record in panel_records
            )
        ),
        "independent_pool_variance_pass": int(
            all(
                int(record.get("independent_pool_variance_pass", 0)) == 1
                for record in panel_records
            )
        ),
        "richardson_formula_pass": int(
            all(
                int(record.get("richardson_formula_pass", 0)) == 1
                for record in panel_records
            )
        ),
        "executed_panels_complete_pass": int(
            all(int(record.get("complete", 0)) == 1 for record in panel_records)
        ),
        "executed_panels_numerically_valid_pass": int(numerically_valid),
        "shard_chain_pass": int(
            all(
                int(dict(record.get("execution", {})).get("shard_chain_pass", 0))
                == 1
                for record in panel_records
            )
        ),
        "mass_conservation_pass": int(
            all(
                _finite_float(
                    dict(record.get("execution", {})).get("mass_error"),
                    math.inf,
                )
                <= HaarPowerRecoveryThresholds().maximum_mass_error
                for record in panel_records
            )
        ),
        "pilot_production_isolation_pass": int(
            all(
                int(record.get("pilot_production_isolation_pass", 0)) == 1
                for record in panel_records
            )
        ),
        "production_authorizing_pass": int(
            all(
                int(record.get("production_authorizing_pass", 0)) == 1
                for record in panel_records
            )
        ),
        "antithetic_panel_a_nominated": int(isinstance(selected, Mapping)),
        "antithetic_panel_b_opened": int(panel_b is not None),
        "antithetic_panels_agree": int(b_pass and combined_pass),
        "selected_profile": ANTITHETIC_HAAR_PROFILE,
        "main_paths": 16,
        "reference_paths": 16,
        "combined_main_half_width": width_source.get("main_half_width"),
        "combined_generator_reference_half_width": width_source.get(
            "generator_reference_half_width"
        ),
        "combined_reference_stability_half_width": width_source.get(
            "reference_stability_half_width"
        ),
        "projected_hours": width_source.get("projected_hours"),
        "minimum_rate": width_source.get(
            "minimum_rate",
            authorizing_execution.get("conservative_rate"),
        ),
        "certificate_fraction": authorizing_execution.get(
            "certificate_fraction"
        ),
        "fallback_fraction": authorizing_execution.get("fallback_fraction"),
        "fallback_cost_fraction": authorizing_execution.get(
            "fallback_cost_fraction"
        ),
        "peak_memory_fraction": authorizing_execution.get(
            "peak_memory_fraction"
        ),
        "forbidden_event_count": forbidden,
    }


def _run_pilot_stage(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    parent_dir: Path,
    sealed_registry: Mapping[str, Any],
) -> dict[str, Any]:
    gate_path = run_dir / "haar_power_recovery_pilot_gate.json"
    metrics_path = run_dir / "haar_power_recovery_pilot_metrics.json"
    if gate_path.is_file() and metrics_path.is_file():
        metrics_record = _load(metrics_path)
        gate = _load(gate_path)
        expected = evaluate_antithetic_pilot(metrics_record["metrics"])
        if gate != expected:
            raise ArtifactCompatibilityError(
                "frozen antithetic pilot gate changed"
            )
        return gate

    panel_a = run_recovery_antithetic_panel(
        run_dir=run_dir,
        parent_haar_run_dir=parent_dir,
        panel="a",
        device=args.device,
    )
    nomination = nominate_haar_power_design(
        profile=ANTITHETIC_HAAR_PROFILE,
        panel_role="a",
        candidates=panel_a["candidates"],
    )
    _freeze(
        run_dir / "antithetic_panel_a_nomination.json",
        nomination,
    )
    selected = nomination.get("selected")
    selected_record = {
        "schema": RUN_SCHEMA + "-selected-antithetic-design",
        "schema_version": 1,
        "selection_status": nomination.get("selection_status"),
        "selected": selected,
        "selected_design_frozen_before_panel_b": int(
            isinstance(selected, Mapping)
        ),
        "fallback_after_panel_b_permitted": 0,
        **NO_WORK,
    }
    _freeze(run_dir / "selected_haar_design.json", selected_record)

    panel_b: Mapping[str, Any] | None = None
    panel_b_confirmation: Mapping[str, Any] | None = None
    combined: Mapping[str, Any] | None = None
    b_evidence_path = (
        run_dir / f"{ANTITHETIC_HAAR_PROFILE}_panel_b_evidence.json"
    )
    b_shard_root = (
        run_dir / "haar_power_shards" / ANTITHETIC_HAAR_PROFILE / "b"
    )
    if not isinstance(selected, Mapping):
        if b_evidence_path.exists() or b_shard_root.exists():
            raise ArtifactCompatibilityError(
                "sealed panel B exists without a panel-A nomination"
            )
        selection = {
            "schema": RUN_SCHEMA + "-sealed-antithetic-selection",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "selection_status": "antithetic_panel_a_no_eligible_design",
            "selected": None,
            "selected_profile": None,
            "panel_a_nominated": 0,
            "panel_b_opened": 0,
            "panels_agree": 0,
            "fallback_after_panel_b_permitted": 0,
            **NO_WORK,
        }
    else:
        panel_b = run_recovery_antithetic_panel(
            run_dir=run_dir,
            parent_haar_run_dir=parent_dir,
            panel="b",
            device=args.device,
        )
        panel_b_confirmation = panel_confirmation_record(panel_b, selected)
        _freeze(
            run_dir / "antithetic_panel_b_confirmation.json",
            panel_b_confirmation,
        )
        combined = combine_certified_haar_power_panels(
            run_dir=run_dir,
            profile=ANTITHETIC_HAAR_PROFILE,
            selected=selected,
            panel_a=panel_a,
            panel_b=panel_b,
        )
        _freeze(
            run_dir / "antithetic_combined_confirmation.json",
            combined,
        )
        agree = bool(
            _confirmation_pass(panel_b_confirmation)
            and _confirmation_pass(combined)
        )
        selection = {
            "schema": RUN_SCHEMA + "-sealed-antithetic-selection",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "selection_status": (
                "sealed_design_confirmed"
                if agree
                else "sealed_panel_b_or_combined_disagrees"
            ),
            "selected": dict(selected),
            "selected_profile": ANTITHETIC_HAAR_PROFILE,
            "panel_a_nominated": 1,
            "panel_b_opened": 1,
            "panel_b_confirmation_pass": int(
                _confirmation_pass(panel_b_confirmation)
            ),
            "combined_confirmation_pass": int(_confirmation_pass(combined)),
            "panels_agree": int(agree),
            "fallback_after_panel_b_permitted": 0,
            **NO_WORK,
        }
    _freeze(run_dir / "sealed_antithetic_selection.json", selection)
    metrics = _panel_metrics(
        panel_a=panel_a,
        nomination=nomination,
        panel_b=panel_b,
        panel_b_confirmation=panel_b_confirmation,
        combined=combined,
        sealed_registry=sealed_registry,
    )
    atomic_write_json(
        metrics_path,
        {
            "schema": RUN_SCHEMA + "-pilot-metrics",
            "schema_version": 1,
            "metrics": metrics,
            "selection_sha256": file_fingerprint(
                run_dir / "sealed_antithetic_selection.json"
            ),
            **NO_WORK,
        },
    )
    gate = evaluate_antithetic_pilot(metrics)
    atomic_write_json(gate_path, gate)
    return gate


def _provenance_failure(error: BaseException) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-provenance-failure",
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
    parent_verified = False
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"Jacobi Haar power recovery run directory: {run_dir}")
        if resumed:
            _verify_terminal_registry(run_dir)
        parent_dir = Path(args.parent_haar_run_dir).resolve()
        provenance = verify_haar_power_recovery_parent(parent_dir)
        artifacts = _static_parent_artifacts(parent_dir)
        source_hash, source_paths = _source_record(parent_dir)
        if resumed:
            # All compatibility checks precede the first mutation.
            _initialize_or_verify_run(
                run_dir,
                args,
                resumed=True,
                parent_dir=parent_dir,
                provenance=provenance,
                source_hash=source_hash,
                source_paths=source_paths,
                artifacts=artifacts,
            )
        else:
            _initialize_or_verify_run(
                run_dir,
                args,
                resumed=False,
                parent_dir=parent_dir,
                provenance=provenance,
                source_hash=source_hash,
                source_paths=source_paths,
                artifacts=artifacts,
            )
        parent_verified = True
        config_sha256 = config_fingerprint(
            _load(run_dir / "scientific_config.json")
        )
        _write_status(
            run_dir,
            status="running",
            outcome="running",
            phase=args.stage,
            required_gate=args.require_gate,
            required_gate_pass=0,
            scientific_config_sha256=config_sha256,
            source_fingerprint=source_hash,
            parent_run_dir=str(parent_dir),
            parent_mutated=0,
        )
        preflight = _existing_gate(
            run_dir, "preflight", "recovery preflight has not run"
        )
        replay = _existing_gate(
            run_dir, "replay", "immutable nested replay has not run"
        )
        pilot = _existing_gate(
            run_dir, "pilot", "sealed antithetic pilot has not run"
        )

        if args.stage in {"preflight", "all"}:
            try:
                preflight = _run_preflight_stage(run_dir, provenance)
            except Exception as exc:
                preflight = _write_stage_failure(
                    run_dir,
                    "preflight",
                    exc,
                    default_domain="schedule_binding",
                    default_code="panel_schedule_binding_invalid",
                )

        if args.stage in {"replay", "all"}:
            if _passed(preflight):
                try:
                    replay = _run_replay_stage(run_dir, parent_dir)
                except Exception as exc:
                    replay = _write_stage_failure(
                        run_dir,
                        "replay",
                        exc,
                        default_domain="nested_panel_replay",
                        default_code="nested_panel_replay_invalid",
                    )
            else:
                replay = not_evaluated_gate(
                    "replay", "immutable-parent preflight did not pass"
                )
                atomic_write_json(
                    run_dir / "haar_power_recovery_replay_gate.json",
                    replay,
                )

        if args.stage in {"pilot", "all"}:
            if _passed(replay):
                try:
                    pilot = _run_pilot_stage(
                        run_dir,
                        args,
                        parent_dir=parent_dir,
                        sealed_registry=artifacts["sealed_registry"],
                    )
                except Exception as exc:
                    domain, code = _typed_failure(
                        exc,
                        default_domain="antithetic_scheduler",
                        default_code="antithetic_scheduler_invalid",
                    )
                    if (
                        "comput" in domain.lower()
                        or "resource" in domain.lower()
                        or "comput" in code.lower()
                        or "resource" in code.lower()
                    ):
                        domain = "antithetic_resource"
                        code = "antithetic_coupling_computationally_infeasible"
                    pilot = _write_stage_failure(
                        run_dir,
                        "pilot",
                        exc,
                        default_domain=domain,
                        default_code=code,
                    )
            else:
                pilot = not_evaluated_gate(
                    "pilot", "immutable nested replay did not pass"
                )
                atomic_write_json(
                    run_dir / "haar_power_recovery_pilot_gate.json",
                    pilot,
                )

        return _finish(
            run_dir,
            args,
            provenance=provenance,
            preflight=preflight,
            replay=replay,
            pilot=pilot,
        )
    except ArtifactCompatibilityError as exc:
        if resumed:
            print(f"Jacobi Haar recovery compatibility error: {exc}", file=sys.stderr)
            return 2
        if run_dir is not None and not parent_verified:
            failure = _provenance_failure(exc)
            atomic_write_json(run_dir / "provenance_failure.json", failure)
            preflight = not_evaluated_gate(
                "preflight", "control provenance is invalid"
            )
            replay = not_evaluated_gate(
                "replay", "control provenance is invalid"
            )
            pilot = not_evaluated_gate("pilot", "control provenance is invalid")
            atomic_write_json(
                run_dir / "haar_power_recovery_preflight_gate.json",
                preflight,
            )
            atomic_write_json(
                run_dir / "haar_power_recovery_replay_gate.json",
                replay,
            )
            atomic_write_json(
                run_dir / "haar_power_recovery_pilot_gate.json",
                pilot,
            )
            return _finish(
                run_dir,
                args,
                provenance=failure,
                preflight=preflight,
                replay=replay,
                pilot=pilot,
            )
        if run_dir is not None:
            _write_status(
                run_dir,
                status="complete",
                outcome="compatibility_error",
                phase=args.stage,
                required_gate=args.require_gate,
                required_gate_pass=0,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        print(f"Jacobi Haar recovery compatibility error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive terminalization
        if run_dir is not None:
            atomic_write_json(
                run_dir / "unexpected_failure.json",
                {
                    "schema": RUN_SCHEMA + "-unexpected-failure",
                    "schema_version": 1,
                    "evaluation_status": "execution_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **NO_WORK,
                },
            )
            registry = _finalize_registry(run_dir)
            _write_status(
                run_dir,
                status="complete",
                outcome="execution_failed",
                phase=args.stage,
                required_gate=args.require_gate,
                required_gate_pass=0,
                error_type=type(exc).__name__,
                error=str(exc),
                **registry,
            )
        print(f"Jacobi Haar recovery error: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
