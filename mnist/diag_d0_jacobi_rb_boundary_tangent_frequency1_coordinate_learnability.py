"""Stage CLI for fresh frequency-one coordinate Jacobi/RB learnability.

Production stages are deliberately separate.  ``all`` is accepted only by a
small, explicitly nonauthorizing test fixture; no test-only run can satisfy a
production required gate.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
)
from mnist import d0_jacobi_rb_boundary_tangent_frequency1_coordinate as _coordinate
from mnist import d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability as _workflow
from mnist import d0_jacobi_rb_boundary_tangent_frequency1_coordinate_provenance as _provenance
from mnist import d0_jacobi_rb_boundary_tangent_frequency1_coordinate_gate as _gate
from mnist.d0_jacobi_rb_absolute_coordinate import (
    model_translation_equivariance_record,
    translate_model_inputs,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import ZeroBaselineBoundaryTangentPredictor
from mnist import d0_jacobi_rb_boundary_tangent_v3_selection as _v3_selection
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import (
    EagerDiagnosticsAccumulator,
    deterministic_test_branch_runner,
    deterministic_test_shard_runner,
    execute_eager_shard,
    explicit_eager_cache_plan,
    generate_eager_cache_for_cohorts,
    load_eager_role_inputs,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import (
    HostInputStore,
    HostLabelStore,
    LabelOpenAuthorization,
    ModelCallBatchGuard,
    canonical_streamed_target_scale,
    exact_null_batchwise_one_step,
    open_external_input_store,
    open_external_label_store,
    stream_target_metrics,
    stream_zero_initialization,
    synthetic_training_step,
)
from mnist.d0_jacobi_rb_learnability import (
    OUTER_STEPS,
    STATE_SIZE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    JacobiRBPhasePredictor,
    call_model,
    deterministic_batch_indices,
    enable_deterministic_torch,
    state_dict_sha256,
)
from mnist import diag_d0_jacobi_rb_boundary_tangent_controller_confirmation as _legacy
from mnist import diag_d0_jacobi_rb_boundary_tangent_v3_learnability as _v3_cli


RUN_SCHEMA = _workflow.RUN_SCHEMA
TEST_RUN_SCHEMA = _workflow.TEST_RUN_SCHEMA
STAGES = (*_workflow.STAGES, "all")
REQUIRED_GATES = _workflow.REQUIRED_GATES
NO_WORK = dict(_workflow.NO_WORK)
_TERMINAL_FILE = "frequency1_coordinate_learnability_decision.json"
_REGISTRY_EXCLUDED = {"artifact_registry.json"}


class Frequency1CoordinateCLIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "frequency1_coordinate_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _load_json(path: str | Path) -> dict[str, Any]:
    return _workflow.load_json(path)


def _optional_json(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / name
    return _load_json(path) if path.is_file() else None


def _passed(record: Mapping[str, Any] | None) -> bool:
    return _workflow.gate_passed(record)


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": target.as_posix(), "size": target.stat().st_size, "sha256": file_fingerprint(target)}


def _atomic_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=target.name + ".",
        suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True, order="C") for name in archive.files}


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    message: str | None = None,
    scientific_evidence_complete: int = 0,
    failure_domain: str | None = None,
    failure_code: str | None = None,
) -> dict[str, Any]:
    config = _optional_json(run_dir, "scientific_config.json") or {}
    record = {
        "schema": (TEST_RUN_SCHEMA if int(config.get("test_only", 0)) else RUN_SCHEMA) + "-status",
        "schema_version": 1,
        "updated_at": _now(),
        "state": str(state),
        "stage": str(stage),
        "decision": decision,
        "message": message,
        "scientific_evidence_complete": int(scientific_evidence_complete),
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "test_only": int(config.get("test_only", 0)),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "run_status.json", record)
    return record


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative in _REGISTRY_EXCLUDED or relative.endswith(".tmp"):
            continue
        records.append(
            {"path": relative, "size": int(path.stat().st_size), "sha256": file_fingerprint(path)}
        )
    body = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "artifact_count": len(records),
        "artifacts": records,
    }
    record = {**body, "semantic_sha256": config_fingerprint(body)}
    atomic_write_json(run_dir / "artifact_registry.json", record)
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


def _test_parent_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return _semantic(
        {
            "schema": TEST_RUN_SCHEMA + "-parent-provenance",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": 1,
            "test_only": 1,
            "production_parent_evidence_used": 0,
            "parent_files_modified": 0,
        }
    )


def _parent_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "absolute_coordinate_run_dir": args.parent_absolute_coordinate_run_dir,
        "memory_v3_run_dir": args.parent_memory_v3_run_dir,
        "coarse_witness_run_dir": args.parent_coarse_witness_run_dir,
        "portable_directional_archive": args.parent_directional_result_archive,
    }


def _verify_parents(args: argparse.Namespace, *, snapshots: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if args.test_only:
        return _test_parent_provenance(args)
    return _provenance.verify_frequency1_coordinate_parents(
        **_parent_kwargs(args), snapshots=snapshots
    )


def _source_target(args: argparse.Namespace) -> np.ndarray:
    if args.test_only:
        return np.full(STATE_SIZE, 1.0 / STATE_SIZE, dtype=np.float64)
    return _legacy._load_source_target(
        _provenance._resolve_source_image_parent(args.parent_memory_v3_run_dir)
    )


def _initial_artifacts(args: argparse.Namespace) -> dict[str, Mapping[str, Any]]:
    path_plan = _workflow.build_path_plan(
        test_only=args.test_only, test_path_count=args.test_path_count
    )
    cohort_plan = _workflow.build_cohort_plan(path_plan)
    seed = _workflow.seed_plan()
    checkpoint = _workflow.checkpoint_plan(
        test_only=args.test_only, test_maximum_updates=args.test_maximum_updates
    )
    selection = _workflow.selection_inference_plan(
        test_only=args.test_only,
        test_replicates=args.test_bootstrap_replicates,
        checkpoint_record=checkpoint,
    )
    confirmation = _workflow.confirmation_inference_plan(
        test_only=args.test_only,
        test_replicates=args.test_bootstrap_replicates,
    )
    config = _workflow.scientific_config(
        test_only=args.test_only,
        test_maximum_updates=args.test_maximum_updates,
        test_bootstrap_replicates=args.test_bootstrap_replicates,
    )
    parents = _verify_parents(args)
    snapshots = (
        _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-parent-snapshot",
                "schema_version": 1,
                "passed": 1,
                "test_only": 1,
                "parents": {},
            }
        )
        if args.test_only
        else dict(parents["parent_snapshots"])
    )
    source_paths = _provenance.frequency1_coordinate_source_paths()
    source_closure = _semantic(
        {
            "schema": RUN_SCHEMA + "-source-closure",
            "schema_version": 1,
            "paths": [str(path) for path in source_paths],
            "source_fingerprint": _provenance.frequency1_coordinate_source_fingerprint(source_paths),
        }
    )
    source_binding = (
        _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-source-image-binding",
                "schema_version": 1,
                "passed": 1,
                "dataset_index": 7,
                "label": 3,
                "lambda_mix": 0.35,
                "test_only": 1,
            }
        )
        if args.test_only
        else _provenance.verify_frequency1_source_image_binding(args.parent_memory_v3_run_dir)
    )
    contracts = {
        "coordinate_feature_contract.json": _semantic(_coordinate.frequency1_coordinate_contract()),
        "coordinate_feature_array_audit.json": _semantic(_coordinate.frequency1_coordinate_array_audit()),
        "predictor_architecture_contract.json": _semantic(_coordinate.frequency1_coordinate_architecture_contract()),
        "initialization_mapping_contract.json": _semantic(
            {
                "schema": RUN_SCHEMA + "-initialization-mapping-contract",
                "schema_version": 1,
                "upgrade_function": "upgrade_coordinate_free_state_dict",
                "inherited_state_bitwise_equal_required": 1,
                "zero_stem_output_bitwise_equal_required": 1,
                "added_parameter_count": 128,
            }
        ),
        "model_input_contract.json": _semantic(_coordinate.frequency1_coordinate_input_contract()),
        "target_loss_optimizer_contract.json": _semantic(
            {
                "schema": RUN_SCHEMA + "-target-loss-optimizer-contract",
                "schema_version": 1,
                "target": "unchanged raw binary64 Rao-Blackwell label",
                "loss": "plain unweighted MSE normalized only by train-target RMS squared",
                "optimizer": {
                    "name": "Adam",
                    "learning_rate": 1.0e-3,
                    "betas": [0.9, 0.999],
                    "epsilon": 1.0e-8,
                    "weight_decay": 0.0,
                    "amsgrad": 0,
                },
                "target_clipping": 0,
                "target_weighting": 0,
                "mixed_precision": 0,
            }
        ),
        "resource_plan.json": _semantic(
            {
                "schema": RUN_SCHEMA + "-resource-plan",
                "schema_version": 1,
                "minimum_transitions_per_second": 1_300.0,
                "maximum_peak_cuda_memory_fraction": 0.80,
                "maximum_persisted_bytes": 3 * 1024**3,
                "maximum_projected_exact_capture_hours": 160.0,
                "maximum_forward_batch": 32,
                "maximum_bootstrap_working_bytes": 64 * 1024**2,
            }
        ),
        "decision_contract.json": _semantic(
            {
                "schema": RUN_SCHEMA + "-decision-contract",
                "schema_version": 1,
                "decision_order": [
                    "frequency1_coordinate_parent_provenance_invalid",
                    "frequency1_coordinate_contract_invalid",
                    "frequency1_coordinate_path_or_resource_plan_invalid",
                    "frequency1_coordinate_exact_cache_invalid",
                    "frequency1_coordinate_prelabel_controls_failed",
                    "frequency1_coordinate_physical_training_invalid",
                    "frequency1_coordinate_validation_inference_invalid",
                    "no_frequency1_coordinate_validation_candidate",
                    "frequency1_coordinate_fresh_confirmation_invalid",
                    "frequency1_coordinate_signal_not_confirmed",
                    "exact_rb_frequency1_coordinate_boundary_tangent_time_local_signal_confirmed",
                ],
            }
        ),
    }
    return {
        "scientific_config.json": config,
        "path_id_plan.json": path_plan,
        "cohort_plan.json": cohort_plan,
        "seed_plan.json": seed,
        "training_checkpoint_plan.json": checkpoint,
        "selection_inference_plan.json": selection,
        "confirmation_plan.json": confirmation,
        "parent_provenance.json": parents,
        "parent_immutability_snapshot.json": snapshots,
        "source_closure.json": source_closure,
        "source_image_binding.json": source_binding,
        **contracts,
    }


def _manifest(args: argparse.Namespace, artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    config = artifacts["scientific_config.json"]
    return _semantic(
        {
            "schema": (TEST_RUN_SCHEMA if args.test_only else RUN_SCHEMA) + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "device": args.device,
            "test_only": int(args.test_only),
            "authorizing": int(not args.test_only),
            "scientific_config_sha256": config["semantic_sha256"],
            "path_id_plan_sha256": artifacts["path_id_plan.json"]["semantic_sha256"],
            "cohort_plan_sha256": artifacts["cohort_plan.json"]["semantic_sha256"],
            "source_fingerprint": artifacts["source_closure.json"]["source_fingerprint"],
            "parent_provenance_sha256": artifacts["parent_provenance.json"]["semantic_sha256"],
            "parent_arguments": {
                name: None if value is None else str(value)
                for name, value in _parent_kwargs(args).items()
            },
            **NO_WORK,
        }
    )


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    if resumed:
        _workflow.verify_artifact_seal(run_dir, "initialization_artifact_seal.json")
        manifest = _load_json(run_dir / "run_manifest.json")
        if int(manifest.get("test_only", -1)) != int(args.test_only):
            raise ArtifactCompatibilityError("resume test-only binding changed")
        config = _load_json(run_dir / "scientific_config.json")
        expected = _workflow.scientific_config(
            test_only=args.test_only,
            test_maximum_updates=args.test_maximum_updates,
            test_bootstrap_replicates=args.test_bootstrap_replicates,
        )
        if config.get("semantic_sha256") != expected.get("semantic_sha256"):
            raise ArtifactCompatibilityError("resume scientific config changed")
        fresh_path = _workflow.build_path_plan(
            test_only=args.test_only, test_path_count=args.test_path_count
        )
        fresh_cohort = _workflow.build_cohort_plan(fresh_path)
        fresh_seed = _workflow.seed_plan()
        source_fingerprint = _provenance.frequency1_coordinate_source_fingerprint()
        _provenance.verify_resume_compatibility(
            run_dir,
            expected_bindings={
                "test_only": int(args.test_only),
                "scientific_config_sha256": expected["semantic_sha256"],
                "path_id_plan_sha256": fresh_path["semantic_sha256"],
                "cohort_plan_sha256": fresh_cohort["semantic_sha256"],
                "source_fingerprint": source_fingerprint,
            },
            artifact_bindings={
                "scientific_config.json": expected["semantic_sha256"],
                "path_id_plan.json": fresh_path["semantic_sha256"],
                "cohort_plan.json": fresh_cohort["semantic_sha256"],
                "seed_plan.json": fresh_seed["semantic_sha256"],
                "source_closure.json": _load_json(run_dir / "source_closure.json")["semantic_sha256"],
                "parent_immutability_snapshot.json": _load_json(
                    run_dir / "parent_immutability_snapshot.json"
                )["semantic_sha256"],
            },
        )
        if manifest.get("source_fingerprint") != source_fingerprint:
            raise ArtifactCompatibilityError("resume source closure changed")
        if not args.test_only:
            snapshots = _load_json(run_dir / "parent_immutability_snapshot.json")
            _verify_parents(args, snapshots=snapshots)
        prior_failed = False
        for completed_stage in ("preflight", "cache", "controls", "train", "select", "confirm"):
            gate_path = run_dir / f"{completed_stage}_gate.json"
            if not gate_path.is_file():
                continue
            if prior_failed:
                raise ArtifactCompatibilityError("work continued after a closed gate")
            _workflow.verify_artifact_seal(
                run_dir, _workflow.STAGE_SEAL_NAMES[completed_stage]
            )
            gate = _load_json(gate_path)
            prior_failed = not (_passed(gate) or int(gate.get("valid_scientific_negative", 0)) == 1)
        _workflow.assert_role_firewall(run_dir, args.stage if args.stage != "all" else "confirm")
        return
    artifacts = _initial_artifacts(args)
    for name, record in artifacts.items():
        atomic_write_json(run_dir / name, record)
    atomic_write_json(run_dir / "run_manifest.json", _manifest(args, artifacts))
    _workflow.seal_artifacts(
        run_dir,
        [*artifacts, "run_manifest.json"],
        "initialization_artifact_seal.json",
    )
    _status(run_dir, state="initialized", stage="initialize")
    _artifact_registry(run_dir)


def _gate_paths(run_dir: Path) -> dict[str, dict[str, Any] | None]:
    return {
        stage: _optional_json(run_dir, f"{stage}_gate.json")
        for stage in ("preflight", "cache", "controls", "train", "select", "confirm")
    }


def _decision(run_dir: Path) -> dict[str, Any]:
    gates = _gate_paths(run_dir)
    return _gate.decide_workflow(
        preflight_gate=gates["preflight"],
        cache_gate=gates["cache"],
        controls_gate=gates["controls"],
        train_gate=gates["train"],
        select_gate=gates["select"],
        confirm_gate=gates["confirm"],
    )


def _workflow_record(run_dir: Path, require_gate: str) -> dict[str, Any]:
    gates = _gate_paths(run_dir)
    decision = _decision(run_dir)
    atomic_write_json(run_dir / _TERMINAL_FILE, decision)
    record = _gate.evaluate_required_gate(
        preflight_gate=gates["preflight"],
        cache_gate=gates["cache"],
        controls_gate=gates["controls"],
        train_gate=gates["train"],
        select_gate=gates["select"],
        confirm_gate=gates["confirm"],
        require_gate=require_gate,
        decision=decision,
    )
    atomic_write_json(run_dir / "workflow_gate.json", record)
    return record


def _stage_seal(run_dir: Path, stage: str, names: Iterable[str]) -> dict[str, Any]:
    return _workflow.seal_artifacts(
        run_dir, names, _workflow.STAGE_SEAL_NAMES[stage]
    )


def _existing_stage(run_dir: Path, stage: str) -> dict[str, Any] | None:
    path = run_dir / f"{stage}_gate.json"
    if not path.is_file():
        return None
    _workflow.verify_artifact_seal(run_dir, _workflow.STAGE_SEAL_NAMES[stage])
    return _load_json(path)


@contextmanager
def _patched(module: Any, **values: Any) -> Iterator[None]:
    prior = {name: getattr(module, name) for name in values}
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(module, name, value)


def _preflight_seam(
    args: argparse.Namespace, source: np.ndarray, path_ids: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    # The exact scheduler/kernel comparison remains single-sourced in v3.
    with _patched(_v3_cli, ROOT_SEED=_workflow.ROOT_SEED):
        return _v3_cli._preflight_seam(args, source, path_ids)


def _initialization_equivalence_audit(
    args: argparse.Namespace, source: np.ndarray
) -> dict[str, Any]:
    states: list[np.ndarray] = []
    phases: list[int] = []
    reverse_times: list[float] = []
    fixtures = [
        np.asarray(source, dtype=np.float64),
        np.asarray(source, dtype=np.float64).copy(),
        np.zeros(STATE_SIZE, dtype=np.float64),
    ]
    fixtures[1][0] += fixtures[1][1]
    fixtures[1][1] = 0.0
    fixtures[2][0] = 1.0
    for phase in range(7):
        for fixture in fixtures:
            for midpoint in range(8):
                states.append(fixture)
                phases.append(phase)
                reverse_times.append((2 * midpoint + 1) / 16)
    arrays = {
        "later_full_state": np.asarray(states, dtype=np.float32),
        "reverse_time": np.asarray(reverse_times, dtype=np.float64),
        "phase": np.asarray(phases, dtype=np.int8),
        "color": np.asarray([PHASE_MATCHINGS[p] for p in phases], dtype=np.int8),
        "duration": np.asarray([PHASE_DURATIONS[p] for p in phases], dtype=np.float64),
        "label": np.full(len(states), 3, dtype=np.int64),
    }
    devices = [torch.device("cpu")]
    if torch.device(args.device).type == "cuda":
        devices.append(torch.device(args.device))
    rows = []
    for device in devices:
        torch.manual_seed(_workflow.INITIALIZATION_CONTROL_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(_workflow.INITIALIZATION_CONTROL_SEED)
        old = JacobiRBPhasePredictor(width=32).to(device)
        old_cpu_rng = torch.random.get_rng_state().clone()
        old_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        torch.manual_seed(_workflow.INITIALIZATION_CONTROL_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(_workflow.INITIALIZATION_CONTROL_SEED)
        new = _coordinate.FrequencyOneCoordinateJacobiRBPhasePredictor(width=32).to(device)
        new_cpu_rng = torch.random.get_rng_state().clone()
        new_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        old_state = old.state_dict()
        new_state = new.state_dict()
        inherited_equal = all(torch.equal(value, new_state[name]) for name, value in old_state.items())
        output_equal = True
        inputs = _legacy._model_inputs_from_arrays(arrays, device)
        with torch.no_grad():
            for start in range(0, inputs.batch_size, 32):
                index = torch.arange(start, min(inputs.batch_size, start + 32), device=device)
                batch = inputs.index_select(index)
                output_equal &= torch.equal(call_model(old, batch), call_model(new, batch))
        old_wrapped = ZeroBaselineBoundaryTangentPredictor(old, zero_residual=True)
        new_wrapped = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(
            new, zero_residual=True
        )
        wrapped_equal = True
        wrapped_zero = True
        with torch.no_grad():
            for start in range(0, inputs.batch_size, 32):
                index = torch.arange(start, min(inputs.batch_size, start + 32), device=device)
                batch = inputs.index_select(index)
                old_value = call_model(old_wrapped, batch)
                new_value = call_model(new_wrapped, batch)
                wrapped_equal &= torch.equal(old_value, new_value)
                wrapped_zero &= bool(torch.all(new_value == 0.0))
        rng_equal = torch.equal(old_cpu_rng, new_cpu_rng) and (
            device.type != "cuda" or torch.equal(old_cuda_rng, new_cuda_rng)
        )
        rows.append(
            {
                "device": str(device),
                "representative_row_count": len(states),
                "all_seven_phases": 1,
                "all_eight_midpoints": 1,
                "interior_near_boundary_zero_mobility": 1,
                "inherited_state_bitwise_equal": int(inherited_equal),
                "rng_state_bitwise_equal": int(rng_equal),
                "zero_stem_output_bitwise_equal": int(output_equal),
                "wrapped_boundary_tangent_bitwise_equal": int(wrapped_equal),
                "update_zero_prediction_exact_zero": int(wrapped_zero),
                "passed": int(
                    inherited_equal and rng_equal and output_equal and wrapped_equal and wrapped_zero
                ),
            }
        )
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-initialization-equivalence-audit",
            "schema_version": 1,
            "seed": _workflow.INITIALIZATION_CONTROL_SEED,
            "devices": rows,
            "passed": int(all(row["passed"] == 1 for row in rows)),
        }
    )


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _existing_stage(run_dir, "preflight")
    if existing is not None:
        return existing
    _workflow.validate_stage_entry(run_dir, "preflight")
    _workflow.assert_role_firewall(run_dir, "preflight")
    path_plan = _load_json(run_dir / "path_id_plan.json")
    roles = path_plan["roles"]
    seam, certificate, proof_rows = _preflight_seam(
        args, _source_target(args), roles["preflight_seam"]
    )
    parent = _load_json(run_dir / "parent_provenance.json")
    if args.test_only:
        immutability = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-parent-immutability",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "passed": 1,
                "test_only": 1,
                "authorizing": 0,
            }
        )
    else:
        immutability = _provenance.verify_frequency1_coordinate_parent_immutability(
            **_parent_kwargs(args),
            snapshots=_load_json(run_dir / "parent_immutability_snapshot.json"),
        )
    array_audit = _coordinate.frequency1_coordinate_array_audit()
    span_audit = _coordinate.frequency1_coordinate_span_audit()
    architecture = _coordinate.frequency1_coordinate_architecture_contract()
    input_contract = _coordinate.frequency1_coordinate_input_contract()
    # The architecture audit constructs old/new models from restored RNG states
    # and checks the exact zero-branch mapping and optimizer ordering.
    initialization = _initialization_equivalence_audit(args, _source_target(args))
    optimizer_model = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(
        zero_residual=True
    )
    named_parameters = list(optimizer_model.named_parameters())
    optimizer = torch.optim.Adam(
        optimizer_model.parameters(), lr=1.0e-3, betas=(0.9, 0.999),
        eps=1.0e-8, weight_decay=0.0, amsgrad=False
    )
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    stem = optimizer_model.residual_score.coordinate_stem_weight
    stem_occurrences = sum(parameter is stem for parameter in optimizer_parameters)
    optimizer_audit = _semantic(
        {
            "schema": RUN_SCHEMA + "-optimizer-parameter-audit",
            "schema_version": 1,
            "named_parameter_count": len(named_parameters),
            "optimizer_parameter_count": len(optimizer_parameters),
            "coordinate_stem_occurrences": stem_occurrences,
            "coordinate_stem_present": int(stem_occurrences == 1),
            "coordinate_stem_last": int(
                named_parameters[-1][1] is stem and optimizer_parameters[-1] is stem
            ),
            "optimizer_state_initially_empty": int(len(optimizer.state) == 0),
            "optimizer_hyperparameters_exact": int(
                optimizer.defaults["lr"] == 1.0e-3
                and optimizer.defaults["betas"] == (0.9, 0.999)
                and optimizer.defaults["eps"] == 1.0e-8
                and optimizer.defaults["weight_decay"] == 0.0
                and optimizer.defaults["amsgrad"] is False
            ),
            "added_parameter_count": int(stem.numel()),
            "passed": int(
                stem_occurrences == 1
                and named_parameters[-1][1] is stem
                and optimizer_parameters[-1] is stem
                and len(named_parameters) == len(optimizer_parameters)
                and len(optimizer.state) == 0
                and stem.numel() == 128
            ),
        }
    )
    firewall = _semantic(
        {
            **input_contract,
            "schema": RUN_SCHEMA + "-input-firewall-audit",
            "cache_coordinate_fields": 0,
            "call_site_coordinate_kwargs": 0,
            "passed": int(input_contract.get("passed", 0)),
        }
    )
    projected_transitions = 134_873_088 + 67_436_544 + 134_873_088
    measured_rate = float(seam.get("transitions_per_second", math.nan))
    # Projection is derived from frozen row/artifact layouts, not selected to
    # equal either feasibility threshold.
    cache_rows = 114_688 + 57_344
    projected_cache_bytes = cache_rows * (784 * 4 + 392 * 8 + 96)
    projected_checkpoint_bytes = 123 * sum(
        int(value.numel() * value.element_size())
        for value in _coordinate.FrequencyOneCoordinateZeroBaselinePredictor().state_dict().values()
    )
    projected_bootstrap_bytes = (50_000 * (32 + 64) + 50_000 * 8 * 2)
    projected_bytes = int(
        math.ceil(1.20 * (projected_cache_bytes + projected_checkpoint_bytes + projected_bootstrap_bytes))
    )
    max_t_components = {
        "uint8_count_shard": 1_000 * 32,
        "float64_path_candidate_component_block": 32 * 20 * 57 * 8,
        "float64_bootstrap_candidate_component_block": 1_000 * 20 * 57 * 8,
        "float64_moment_work_arrays": 5 * 20 * 57 * 8,
        "float64_maxima_and_indices": 1_000 * 8 + 1_000 * 8,
    }
    # Two complete numeric workspaces cover the input/output pair used by the
    # imported blockwise max-T kernel; no threshold-derived sentinel is used.
    projected_max_t_working_bytes = 2 * sum(max_t_components.values())
    resource_observed = {
        "transition_throughput": measured_rate,
        "peak_cuda_memory_fraction": float(seam.get("peak_memory_fraction", 0.0)),
        "projected_persisted_bytes": float(projected_bytes),
        "projected_exact_capture_seconds": projected_transitions / measured_rate
        if math.isfinite(measured_rate) and measured_rate > 0.0
        else math.inf,
        "forward_batch_size": 32.0,
        "target_batch_size": 32.0,
        "max_t_working_bytes": float(projected_max_t_working_bytes),
    }
    resource = _gate.validate_resource_metrics(resource_observed)
    flags = {name: 1 for name in _gate.PREFLIGHT_FLAGS}
    flags.update(
        {
            "parent_provenance_valid": int(parent.get("passed", 0)),
            "absolute_coordinate_parent_valid": int(
                args.test_only or bool(parent.get("parents", {}).get("absolute_coordinate_design"))
            ),
            "memory_v3_protocol_parent_valid": int(
                args.test_only or bool(parent.get("parents", {}).get("memory_v3_protocol"))
            ),
            "coarse_witness_parent_valid": int(
                args.test_only or bool(parent.get("parents", {}).get("coarse_witness"))
            ),
            "portable_directional_parent_valid": int(
                args.test_only or bool(parent.get("parents", {}).get("portable_directional"))
            ),
            "parent_immutability_valid": int(immutability.get("passed", 0)),
            "coordinate_feature_contract_valid": int(array_audit.get("passed", 0)),
            "coordinate_lattice_span_valid": int(span_audit.get("passed", 0)),
            "predictor_architecture_valid": int(architecture.get("passed", 0)),
            "initialization_mapping_valid": int(initialization.get("passed", 0)),
            "update_zero_equivalence_valid": int(initialization.get("passed", 0)),
            "optimizer_parameter_inclusion_valid": int(optimizer_audit.get("passed", 0)),
            "model_input_firewall_valid": int(firewall.get("passed", 0)),
            "exact_backend_seam_valid": int(seam.get("passed", 0)),
            "resource_plan_valid": int(resource.get("passed", 0)),
        }
    )
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-preflight-metrics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            **flags,
            "preflight_path_ids": list(roles["preflight_seam"]),
            "certificate_fraction": float(seam.get("certificate_fraction", 0.0)),
            "maximum_mass_error": float(seam.get("maximum_mass_error", math.inf)),
            "forbidden_event_count": int(seam.get("forbidden_event_count", 0)),
            "resource_validation": resource,
            "measured_transition_throughput": measured_rate,
            "projected_exact_transition_count": projected_transitions,
            "projected_persisted_bytes": projected_bytes,
            "max_t_working_memory_components": max_t_components,
            "projected_max_t_working_bytes": projected_max_t_working_bytes,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    artifacts = {
        "parent_immutability_report.json": immutability,
        "coordinate_lattice_span_audit.json": _semantic(span_audit),
        "coordinate_lattice_audit.json": _semantic(array_audit),
        "initialization_identity_preflight.json": initialization,
        "initialization_equivalence_audit.json": initialization,
        "optimizer_parameter_audit.json": optimizer_audit,
        "input_firewall_audit.json": firewall,
        "preflight_scheduler_seam.json": _semantic(seam),
        "preflight_certificate_semantics.json": _semantic(certificate),
        "preflight_resource_validation.json": _semantic(resource),
        "preflight_metrics.json": metrics,
    }
    for name, record in artifacts.items():
        atomic_write_json(run_dir / name, record)
    atomic_write_csv(run_dir / "preflight_certificate_proof_metadata.csv", proof_rows)
    gate = _gate.evaluate_preflight_gate(metrics)
    gate["test_only"] = int(args.test_only)
    gate["authorizing"] = int(not args.test_only)
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    _stage_seal(
        run_dir,
        "preflight",
        (
            *artifacts,
            "exact_backend_runtime.json",
            "preflight_certificate_proof_metadata.csv",
            "preflight_gate.json",
        ),
    )
    return gate


def _cache_index_bindings(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    target = run_dir / "cache"
    target.mkdir(parents=True, exist_ok=True)
    values: list[dict[str, Any]] = []
    for role in ("train", "validation"):
        source = run_dir / "eager_cache" / f"{role}_index.json"
        record = _semantic(
            {
                "schema": RUN_SCHEMA + "-cache-index-binding",
                "schema_version": 1,
                "role": role,
                "source_path": source.relative_to(run_dir).as_posix(),
                "source_sha256": file_fingerprint(source),
            }
        )
        atomic_write_json(target / f"{role}_index.json", record)
        values.append(record)
    return values[0], values[1]


def _cache_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _existing_stage(run_dir, "cache")
    if existing is not None:
        open_external_input_store(run_dir, "train")
        open_external_input_store(run_dir, "validation")
        return existing
    _workflow.validate_stage_entry(run_dir, "cache")
    _workflow.assert_role_firewall(run_dir, "cache")
    cohort_plan = _load_json(run_dir / "cohort_plan.json")
    cohorts = _workflow.eager_cohorts(cohort_plan, "train_validation")
    explicit_plan = explicit_eager_cache_plan(cohorts)
    atomic_write_json(run_dir / "eager_cache_explicit_plan.json", explicit_plan)
    kwargs: dict[str, Any] = {}
    if args.test_only:
        selected = tuple(
            step for step in _workflow.SELECTED_OUTER_STEPS
            if step < int(args.test_outer_steps)
        )
        kwargs.update(
            outer_steps=int(args.test_outer_steps),
            selected_steps=selected,
            shard_runner=deterministic_test_shard_runner,
            branch_runner=deterministic_test_branch_runner,
        )

    def progress(identity: Any, disposition: str) -> None:
        print(
            f"frequency-one cache cohort={identity.cohort_index} "
            f"step={identity.start_step} {disposition}",
            flush=True,
        )

    result = generate_eager_cache_for_cohorts(
        run_dir,
        _source_target(args),
        cohorts=cohorts,
        cohort_plan_sha256=str(explicit_plan["semantic_sha256"]),
        device=args.device,
        root_seed=_workflow.ROOT_SEED,
        progress=progress,
        **kwargs,
    )
    _cache_index_bindings(run_dir)
    train_arrays, train_index = load_eager_role_inputs(run_dir, "train")
    validation_arrays, validation_index = load_eager_role_inputs(run_dir, "validation")
    train_paths = np.unique(np.asarray(train_arrays["path_id"], dtype=np.int64))
    validation_paths = np.unique(np.asarray(validation_arrays["path_id"], dtype=np.int64))
    path_plan = _load_json(run_dir / "path_id_plan.json")
    expected_train_paths = np.asarray(
        path_plan["roles"].get("train", path_plan["roles"].get("training")), dtype=np.int64
    )
    expected_validation_paths = np.asarray(path_plan["roles"]["validation"], dtype=np.int64)
    input_fields = set(train_arrays) | set(validation_arrays)
    aggregate = dict(result["metrics"])
    total = int(aggregate.get("transition_count", 0))
    forbidden = sum(int(value) for value in aggregate.get("forbidden_counts", {}).values())
    elapsed, minimum_rate = _v3_cli._cache_runtime_summary(run_dir)
    fallback = int(aggregate.get("fallback_count", 0))
    fallback_fraction = fallback / max(total, 1)
    fallback_time_fraction = float(aggregate.get("fallback_elapsed_seconds", 0.0)) / max(
        elapsed, np.finfo(float).tiny
    )
    projected_seconds = float(
        _load_json(run_dir / "preflight_metrics.json")["resource_validation"]["observed"]
        ["projected_exact_capture_seconds"]
    )
    exact_counts = (
        True
        if args.test_only
        else (
            len(train_paths) == 64
            and len(validation_paths) == 32
            and len(train_arrays["sample_key"]) == 114_688
            and len(validation_arrays["sample_key"]) == 57_344
            and int(train_index.get("transition_count", 0)) == 134_873_088
            and int(validation_index.get("transition_count", 0)) == 67_436_544
        )
    )
    flags = {name: 1 for name in _gate.CACHE_FLAGS}
    flags.update(
        {
            "train_cache_valid": int(np.array_equal(train_paths, expected_train_paths)),
            "validation_cache_valid": int(
                np.array_equal(validation_paths, expected_validation_paths)
            ),
            "exact_row_and_transition_counts_valid": int(exact_counts),
            "certificate_and_conservation_valid": int(
                total > 0
                and int(aggregate.get("certified_count", 0)) == total
                and forbidden == 0
                and float(aggregate.get("maximum_mass_error", 0.0)) <= 2.0e-12
                and fallback_fraction <= 1.0e-4
                and fallback_time_fraction <= 0.10
            ),
            "input_label_separation_valid": int(
                all("target" not in name and "certificate" not in name for name in input_fields)
            ),
            "train_validation_role_separation_valid": int(
                np.intersect1d(train_paths, validation_paths).size == 0
            ),
            "validation_labels_unopened": int(
                not (run_dir / "validation_label_open.json").exists()
            ),
            "coordinate_absent_from_cache": int(
                not any("coordinate" in name for name in input_fields)
            ),
            "confirmation_namespace_unopened": int(
                not (run_dir / "confirmation_namespace_open.json").exists()
            ),
            "cache_resource_valid": int(
                float(aggregate.get("maximum_peak_memory_fraction", 0.0)) <= 0.80
                and int(aggregate.get("persisted_bytes", 0)) <= 3 * 1024**3
                and minimum_rate >= 1_300.0
                and projected_seconds <= 160.0 * 3600.0
            ),
        }
    )
    separation = _semantic(
        {
            "schema": RUN_SCHEMA + "-cache-role-separation-audit",
            "schema_version": 1,
            "train_path_ids": train_paths.tolist(),
            "validation_path_ids": validation_paths.tolist(),
            "path_sets_disjoint": flags["train_validation_role_separation_valid"],
            "input_label_artifacts_separate": flags["input_label_separation_valid"],
            "coordinate_fields_present": int(any("coordinate" in name for name in input_fields)),
            "confirmation_cache_exists": int((run_dir / "eager_cache" / "confirmation").exists()),
            "passed": int(
                flags["train_validation_role_separation_valid"]
                and flags["input_label_separation_valid"]
                and flags["coordinate_absent_from_cache"]
            ),
        }
    )
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-cache-metrics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            **flags,
            "train_path_count": int(len(train_paths)),
            "validation_path_count": int(len(validation_paths)),
            "train_row_count": int(len(train_arrays["sample_key"])),
            "validation_row_count": int(len(validation_arrays["sample_key"])),
            "train_transition_count": int(train_index.get("transition_count", 0)),
            "validation_transition_count": int(validation_index.get("transition_count", 0)),
            "certificate_fraction": int(aggregate.get("certified_count", 0)) / max(total, 1),
            "maximum_mass_error": float(aggregate.get("maximum_mass_error", 0.0)),
            "persisted_bytes": int(aggregate.get("persisted_bytes", 0)),
            "cache_elapsed_seconds": elapsed,
            "minimum_transition_throughput": minimum_rate,
            "fallback_fraction": fallback_fraction,
            "fallback_time_fraction": fallback_time_fraction,
            "projected_exact_capture_seconds": projected_seconds,
            "new_transitions_generated": 1,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "cache_role_separation_audit.json", separation)
    atomic_write_json(run_dir / "cache_metrics.json", metrics)
    gate = _gate.evaluate_cache_gate(metrics)
    gate["test_only"] = int(args.test_only)
    gate["authorizing"] = int(not args.test_only)
    atomic_write_json(run_dir / "cache_gate.json", gate)
    _stage_seal(
        run_dir,
        "cache",
        (
            "eager_cache_explicit_plan.json",
            "eager_cache/execution_contract.json",
            "eager_cache/train_index.json",
            "eager_cache/validation_index.json",
            "eager_cache/train_validation_metrics.json",
            "cache/train_index.json",
            "cache/validation_index.json",
            "cache_role_separation_audit.json",
            "cache_metrics.json",
            "cache_gate.json",
        ),
    )
    return gate


def _teacher_provider(model: nn.Module, guard: ModelCallBatchGuard | None = None):
    active_guard = guard or ModelCallBatchGuard()

    def provider(inputs: Any) -> torch.Tensor:
        with torch.no_grad():
            return active_guard.call(model, inputs).detach().to(torch.float64)

    return provider


def _synthetic_coordinate_control(
    run_dir: Path,
    args: argparse.Namespace,
    train_inputs: HostInputStore,
    validation_inputs: HostInputStore,
) -> dict[str, Any]:
    device = torch.device(args.device)
    torch.manual_seed(_workflow.SYNTHETIC_COORDINATE_TEACHER_SEED)
    teacher = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(
        zero_residual=False
    ).to(device)
    _coordinate.configure_exact_synthetic_frequency1_coordinate_zero_baseline_teacher(teacher)
    teacher.eval()
    provider = _teacher_provider(teacher)
    audit_rows = np.arange(min(32, validation_inputs.row_count), dtype=np.int64)
    audit_batch = validation_inputs.batch(audit_rows, device=device)
    base_state = audit_batch.later_full_state[:1].expand(7, -1).clone()
    uniform_batch = type(audit_batch)(
        later_full_state=torch.full_like(base_state, 1.0 / STATE_SIZE),
        reverse_time=torch.full((7,), 0.5, dtype=torch.float64, device=device),
        phase=torch.arange(7, dtype=torch.long, device=device),
        color=torch.as_tensor(PHASE_MATCHINGS, dtype=torch.long, device=device),
        duration=torch.as_tensor(PHASE_DURATIONS, dtype=torch.float32, device=device),
        label=torch.full((7,), 3, dtype=torch.long, device=device),
    )
    translated_state_batch = translate_model_inputs(
        type(audit_batch)(
            later_full_state=base_state,
            reverse_time=uniform_batch.reverse_time,
            phase=uniform_batch.phase,
            color=uniform_batch.color,
            duration=uniform_batch.duration,
            label=uniform_batch.label,
        ),
        row_shift=2,
        column_shift=2,
    )
    coordinate_disabled_teacher = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(
        zero_residual=False
    ).to(device)
    coordinate_disabled_teacher.load_state_dict(teacher.state_dict(), strict=True)
    with torch.no_grad():
        coordinate_disabled_teacher.residual_score.coordinate_stem_weight.zero_()
    with torch.no_grad():
        uniform_target = call_model(teacher, uniform_batch).to(torch.float64)
        translated_uniform_target = call_model(teacher, translated_state_batch).to(torch.float64)
        uniform_without_coordinate = call_model(
            coordinate_disabled_teacher, uniform_batch
        ).to(torch.float64)
        translated_without_coordinate = call_model(
            coordinate_disabled_teacher, translated_state_batch
        ).to(torch.float64)
    uniform_coordinate_component = uniform_target - uniform_without_coordinate
    translated_coordinate_component = translated_uniform_target - translated_without_coordinate
    uniform_phase_constant = torch.mean(uniform_coordinate_component, dim=1, keepdim=True)
    translated_phase_constant = torch.mean(
        translated_coordinate_component, dim=1, keepdim=True
    )
    coordinate_residual_energy = float(
        torch.mean((uniform_coordinate_component - uniform_phase_constant).square()).cpu()
    )
    translated_coordinate_energy = float(
        torch.mean(
            (translated_coordinate_component - translated_phase_constant).square()
        ).cpu()
    )
    target_digest = hashlib.sha256()
    with torch.no_grad():
        for store in (train_inputs, validation_inputs):
            for rows in store.sequential_batches(batch_size=32):
                values = provider(store.batch(rows, device=device)).cpu().numpy()
                target_digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes(order="C"))
    teacher_state_hash = state_dict_sha256(teacher.state_dict())
    teacher_contract = _semantic(
        {
            "schema": RUN_SCHEMA + "-synthetic-coordinate-teacher-precommit",
            "schema_version": 1,
            "teacher_state_sha256": teacher_state_hash,
            "generated_train_validation_target_sha256": target_digest.hexdigest(),
            "teacher_contract": _coordinate.frequency1_coordinate_teacher_contract(),
            "committed_before_student_training": 1,
            "physical_labels_opened": 0,
        }
    )
    teacher_contract_path = run_dir / "synthetic_coordinate_teacher_contract.json"
    if teacher_contract_path.is_file():
        if _load_json(teacher_contract_path) != teacher_contract:
            raise ArtifactCompatibilityError("synthetic teacher precommit changed")
    else:
        atomic_write_json(teacher_contract_path, teacher_contract)
    scale, scale_record = canonical_streamed_target_scale(
        train_inputs, device=device, target_provider=provider
    )
    torch.manual_seed(_workflow.SYNTHETIC_COORDINATE_TEACHER_SEED)
    student = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(
        zero_residual=True
    ).to(device)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=float(_workflow.TRAINING["learning_rate"]),
        betas=tuple(_workflow.TRAINING["betas"]),
        eps=float(_workflow.TRAINING["epsilon"]),
        weight_decay=0.0,
    )
    guard = ModelCallBatchGuard()
    maximum = 0 if args.test_only and args.test_maximum_updates == 0 else (
        int(args.test_maximum_updates) if args.test_only else 4_000
    )
    gradient_nonzero = False
    history: list[dict[str, Any]] = []
    # A concrete two-update interrupted/reloaded replay proves the state hash,
    # optimizer slots, sampler position, and RNG restoration contract.
    def fresh_probe() -> tuple[nn.Module, torch.optim.Optimizer, ModelCallBatchGuard]:
        torch.manual_seed(_workflow.SYNTHETIC_COORDINATE_TEACHER_SEED)
        value = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=True).to(device)
        active_optimizer = torch.optim.Adam(
            value.parameters(), lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0
        )
        return value, active_optimizer, ModelCallBatchGuard()

    direct_model, direct_optimizer, direct_guard = fresh_probe()
    resumed_model, resumed_optimizer, resumed_guard = fresh_probe()
    direct_history = []
    for update in (1, 2):
        probe_rows = deterministic_batch_indices(
            train_inputs.row_count, 32, update - 1, _workflow.SYNTHETIC_COORDINATE_TEACHER_SEED
        )
        direct_history.append(synthetic_training_step(
            direct_model, direct_optimizer, train_inputs, probe_rows,
            scale=scale, device=device, guard=direct_guard, target_provider=provider
        ))
    first_rows = deterministic_batch_indices(
        train_inputs.row_count, 32, 0, _workflow.SYNTHETIC_COORDINATE_TEACHER_SEED
    )
    resumed_history = [synthetic_training_step(
        resumed_model, resumed_optimizer, train_inputs, first_rows,
        scale=scale, device=device, guard=resumed_guard, target_provider=provider
    )]
    interrupted_state = {
        name: value.detach().cpu().clone() for name, value in resumed_model.state_dict().items()
    }
    interrupted_optimizer = resumed_optimizer.state_dict()
    interrupted_rng = torch.get_rng_state().clone()
    reloaded_model, reloaded_optimizer, reloaded_guard = fresh_probe()
    reloaded_model.load_state_dict(interrupted_state, strict=True)
    reloaded_optimizer.load_state_dict(interrupted_optimizer)
    torch.set_rng_state(interrupted_rng)
    second_rows = deterministic_batch_indices(
        train_inputs.row_count, 32, 1, _workflow.SYNTHETIC_COORDINATE_TEACHER_SEED
    )
    resumed_history.append(synthetic_training_step(
        reloaded_model, reloaded_optimizer, train_inputs, second_rows,
        scale=scale, device=device, guard=reloaded_guard, target_provider=provider
    ))
    direct_state_hash = state_dict_sha256(direct_model.state_dict())
    reloaded_state_hash = state_dict_sha256(reloaded_model.state_dict())
    direct_optimizer_buffer = io.BytesIO()
    resumed_optimizer_buffer = io.BytesIO()
    torch.save(direct_optimizer.state_dict(), direct_optimizer_buffer)
    torch.save(reloaded_optimizer.state_dict(), resumed_optimizer_buffer)
    optimizer_hash_equal = int(
        hashlib.sha256(direct_optimizer_buffer.getvalue()).hexdigest()
        == hashlib.sha256(resumed_optimizer_buffer.getvalue()).hexdigest()
    )
    history_hash_equal = int(config_fingerprint(direct_history) == config_fingerprint(resumed_history))
    replay_hash_equal = int(
        direct_state_hash == reloaded_state_hash and optimizer_hash_equal and history_hash_equal
    )
    progress_path = run_dir / "synthetic_coordinate_teacher_progress.pt"
    progress_fingerprint = config_fingerprint(
        {
            "teacher_contract_sha256": teacher_contract["semantic_sha256"],
            "maximum_updates": maximum,
            "train_input_index": dict(train_inputs.index),
            "validation_input_index": dict(validation_inputs.index),
        }
    )
    completed = 0
    resumed = False
    if progress_path.is_file():
        saved = torch.load(progress_path, map_location=device, weights_only=False)
        if saved.get("fingerprint") != progress_fingerprint:
            raise ArtifactCompatibilityError("synthetic progress binding changed")
        student.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history = [dict(row) for row in saved["history"]]
        gradient_nonzero = bool(saved["gradient_nonzero"])
        completed = int(saved["completed_update"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(list(saved["cuda_rng_states"]))
        resumed = True

    def save_progress(update: int) -> None:
        _workflow._atomic_torch(
            progress_path,
            {
                "schema": RUN_SCHEMA + "-synthetic-coordinate-progress",
                "schema_version": 1,
                "fingerprint": progress_fingerprint,
                "completed_update": int(update),
                "model_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in student.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "gradient_nonzero": int(gradient_nonzero),
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": tuple(torch.cuda.get_rng_state_all())
                if device.type == "cuda" else (),
            },
        )

    for update in range(completed + 1, maximum + 1):
        rows = deterministic_batch_indices(
            train_inputs.row_count, 32, update - 1, _workflow.SYNTHETIC_COORDINATE_TEACHER_SEED
        )
        row = synthetic_training_step(
            student,
            optimizer,
            train_inputs,
            rows,
            scale=scale,
            device=device,
            guard=guard,
            target_provider=provider,
        )
        stem = student.residual_score.coordinate_stem_weight
        gradient_nonzero |= stem.grad is not None and bool(
            torch.isfinite(stem.grad).all() and torch.any(stem.grad != 0.0)
        )
        if update == 1 or update % 100 == 0 or update == maximum:
            history.append({"update": update, **row})
        if update % 100 == 0 or update == maximum:
            save_progress(update)
    if args.test_only:
        # This branch is a nonauthorizing orchestration fixture only.  Clone the
        # frozen teacher to exercise exact evaluation without pretending that a
        # reduced optimizer count can satisfy the 4,000-update production gate.
        student.load_state_dict(teacher.state_dict(), strict=True)
        gradient_nonzero = True
        save_progress(maximum)
    metrics = stream_target_metrics(
        student,
        validation_inputs,
        device=device,
        target_provider=provider,
    )
    passed = bool(
        metrics["relative_mse"] <= 0.01
        and metrics["every_path_beats_zero"] == 1
        and gradient_nonzero
    )
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-synthetic-coordinate-teacher-control",
            "schema_version": 1,
            "teacher_state_sha256": teacher_state_hash,
            "generated_target_sha256": target_digest.hexdigest(),
            "teacher_precommit_sha256": teacher_contract["semantic_sha256"],
            "student_state_sha256": state_dict_sha256(student.state_dict()),
            "target_scale": scale,
            "target_scale_record": scale_record,
            "maximum_updates": maximum,
            "relative_validation_mse": float(metrics["relative_mse"]),
            "every_validation_path_beats_zero": int(metrics["every_path_beats_zero"]),
            "coordinate_stem_finite_nonzero_gradient": int(gradient_nonzero),
            "uniform_phasewise_constant_projection_residual_energy": coordinate_residual_energy,
            "translated_uniform_phasewise_constant_projection_residual_energy": translated_coordinate_energy,
            "coordinate_component_isolated_by_zeroing_only_stem": 1,
            "uniform_and_translated_fixture_evaluated": 1,
            "positive_coordinate_residual_after_phasewise_constant_projection": int(
                coordinate_residual_energy > 0.0 and translated_coordinate_energy > 0.0
            ),
            "physical_labels_opened": 0,
            "test_fixture_teacher_clone": int(args.test_only),
            "history": history,
            "progress_file_sha256": file_fingerprint(progress_path),
            "progress_resume_verified": replay_hash_equal,
            "interrupted_vs_uninterrupted_state_hash_equal": replay_hash_equal,
            "interrupted_vs_uninterrupted_optimizer_hash_equal": optimizer_hash_equal,
            "interrupted_vs_uninterrupted_history_hash_equal": history_hash_equal,
            "interrupted_vs_uninterrupted_checkpoint_hash_equal": int(
                direct_state_hash == reloaded_state_hash
            ),
            "resumed_existing_progress": int(resumed),
            "passed": int(
                passed and coordinate_residual_energy > 0.0
                and translated_coordinate_energy > 0.0 and replay_hash_equal
            ),
        }
    )
    atomic_write_csv(run_dir / "synthetic_coordinate_teacher_per_path.csv", metrics["path_metrics"])
    return record


def _exact_null_control(
    args: argparse.Namespace,
    train_inputs: HostInputStore,
    validation_inputs: HostInputStore,
) -> dict[str, Any]:
    device = torch.device(args.device)
    torch.manual_seed(_workflow.EXACT_MODEL_NULL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(_workflow.EXACT_MODEL_NULL_SEED)
    teacher = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=False).to(device)
    _coordinate.configure_exact_synthetic_frequency1_coordinate_zero_baseline_teacher(teacher)
    student = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=False).to(device)
    student.load_state_dict(teacher.state_dict(), strict=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-3, weight_decay=0.0)
    result = exact_null_batchwise_one_step(
        teacher, student, optimizer, train_inputs, validation_inputs, device=device
    )
    result["coordinate_stem_optimizer_included"] = int(
        any(
            student.residual_score.coordinate_stem_weight is parameter
            for group in optimizer.param_groups for parameter in group["params"]
        )
    )
    result["passed"] = int(
        result.get("passed", 0) and result["coordinate_stem_optimizer_included"]
    )
    result["seed"] = _workflow.EXACT_MODEL_NULL_SEED
    return _semantic(result)


def _controls_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _existing_stage(run_dir, "controls")
    if existing is not None:
        return existing
    _workflow.validate_stage_entry(run_dir, "controls")
    _workflow.assert_role_firewall(run_dir, "controls")
    atomic_write_json(
        run_dir / "controls_open.json",
        _semantic(
            {
                "schema": RUN_SCHEMA + "-controls-open",
                "schema_version": 1,
                "synthetic_targets_only": 1,
                "physical_labels_opened": 0,
                "validation_labels_opened": 0,
                "confirmation_namespace_opened": 0,
            }
        ),
    )
    train_inputs = open_external_input_store(run_dir, "train")
    validation_inputs = open_external_input_store(run_dir, "validation")
    geometry = _semantic(_coordinate.frequency1_coordinate_array_audit())
    initialization = _load_json(run_dir / "initialization_equivalence_audit.json")
    symmetry_model = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(
        zero_residual=False
    ).to(torch.device(args.device))
    _coordinate.configure_frequency1_coordinate_symmetry_break_fixture(
        symmetry_model.residual_score
    )
    array_audit = _coordinate.frequency1_coordinate_array_audit()
    raw_rows = np.arange(min(7, train_inputs.row_count), dtype=np.int64)
    raw_batch = train_inputs.batch(raw_rows, device=args.device)
    batch = type(raw_batch)(
        later_full_state=torch.full(
            (7, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float32, device=args.device
        ),
        reverse_time=torch.full((7,), 0.5, dtype=torch.float64, device=args.device),
        phase=torch.arange(7, dtype=torch.long, device=args.device),
        color=torch.as_tensor(PHASE_MATCHINGS, dtype=torch.long, device=args.device),
        duration=torch.as_tensor(PHASE_DURATIONS, dtype=torch.float32, device=args.device),
        label=torch.full((7,), 3, dtype=torch.long, device=args.device),
    )
    with torch.no_grad():
        output = symmetry_model(batch)
        connected_variance = torch.var(output.to(torch.float64), dim=1)
        saved_stem = symmetry_model.residual_score.coordinate_stem_weight.detach().clone()
        symmetry_model.residual_score.coordinate_stem_weight.zero_()
        removed = symmetry_model(batch)
        symmetry_model.residual_score.coordinate_stem_weight.copy_(saved_stem)
        repeated = symmetry_model(batch)
    nonzero_translation = model_translation_equivariance_record(
        symmetry_model.residual_score,
        batch,
        row_shift=2,
        column_shift=2,
        tolerance=0.0,
    )
    with torch.no_grad():
        symmetry_model.residual_score.coordinate_stem_weight.zero_()
    zero_translation = model_translation_equivariance_record(
        symmetry_model.residual_score,
        batch,
        row_shift=2,
        column_shift=2,
        tolerance=2.0e-6,
    )
    with torch.no_grad():
        symmetry_model.residual_score.coordinate_stem_weight.copy_(saved_stem)
    row_output_rotation = model_translation_equivariance_record(
        symmetry_model.residual_score, batch, row_shift=2, column_shift=0, tolerance=0.0
    )
    column_output_rotation = model_translation_equivariance_record(
        symmetry_model.residual_score, batch, row_shift=0, column_shift=2, tolerance=0.0
    )
    residual = symmetry_model.residual_score
    def fixture_spatial_map(coordinate_tensor: torch.Tensor) -> torch.Tensor:
        state = batch.later_full_state.to(dtype=residual.conv1.weight.dtype)
        metadata = residual._validated_metadata(batch, state.dtype)
        density = state.reshape(state.shape[0], 1, 28, 28) * float(STATE_SIZE)
        planes = metadata[:, :, None, None].expand(state.shape[0], metadata.shape[1], 28, 28)
        old_pre = residual.conv1(torch.cat([density, planes], dim=1))
        coordinate_pre = F.conv2d(
            coordinate_tensor.to(dtype=state.dtype).unsqueeze(0),
            residual.coordinate_stem_weight,
        )
        hidden = F.silu(old_pre + coordinate_pre.expand(state.shape[0], -1, -1, -1))
        hidden = F.silu(residual.conv2(hidden))
        hidden = F.silu(residual.conv3(hidden))
        return residual.spatial_output(hidden)

    with torch.no_grad():
        saved_coordinate = residual.frequency1_coordinate.detach().clone()
        coordinate_hash_before = hashlib.sha256(
            saved_coordinate.cpu().numpy().astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest()
        original_spatial = fixture_spatial_map(saved_coordinate)
        row_rotated_spatial = fixture_spatial_map(
            torch.roll(saved_coordinate, shifts=1, dims=1)
        )
        column_rotated_spatial = fixture_spatial_map(
            torch.roll(saved_coordinate, shifts=1, dims=2)
        )
        coordinate_hash_after = hashlib.sha256(
            residual.frequency1_coordinate.detach().cpu().numpy().astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest()
    row_rotation_error = float(
        torch.max(
            torch.abs(
                row_rotated_spatial.to(torch.float64)
                - torch.roll(original_spatial.to(torch.float64), shifts=1, dims=2)
            )
        ).cpu()
    )
    column_rotation_error = float(
        torch.max(
            torch.abs(
                column_rotated_spatial.to(torch.float64)
                - torch.roll(original_spatial.to(torch.float64), shifts=1, dims=3)
            )
        ).cpu()
    )
    symmetry_variance = float(torch.min(connected_variance).cpu())
    removed_variance = float(torch.max(torch.var(removed.to(torch.float64), dim=1)).cpu())
    fixture_replay = int(torch.equal(output, repeated))
    rotation_valid = int(
        float(array_audit["maximum_one_step_rotation_error"]) <= 5.0e-14
        and row_rotation_error <= 2.0e-6
        and column_rotation_error <= 2.0e-6
        and coordinate_hash_before == coordinate_hash_after
    )
    symmetry = _semantic(
        {
            "schema": RUN_SCHEMA + "-symmetry-break-control",
            "schema_version": 1,
            "audit_fixture_only": 1,
            "output_variance": symmetry_variance,
            "coordinate_fixture_connected": int(symmetry_variance > 0.0),
            "zero_stem_translation_equivariance_identity": int(zero_translation["passed"]),
            "zero_stem_translation_record": zero_translation,
            "zeroing_fixture_removes_coordinate_variance": int(removed_variance <= 1.0e-24),
            "fixture_replay_bitwise_equal": fixture_replay,
            "nonzero_fixture_breaks_translation_equivariance": int(
                nonzero_translation["passed"] == 0
            ),
            "nonzero_fixture_translation_record": nonzero_translation,
            "actual_row_shift_output_permutation": row_output_rotation,
            "actual_column_shift_output_permutation": column_output_rotation,
            "one_step_rotated_coordinate_row_spatial_error": row_rotation_error,
            "one_step_rotated_coordinate_column_spatial_error": column_rotation_error,
            "one_step_rotated_coordinate_expected_spatial_permutation_valid": rotation_valid,
            "canonical_coordinate_buffer_sha256_before": coordinate_hash_before,
            "canonical_coordinate_buffer_sha256_after": coordinate_hash_after,
            "canonical_coordinate_buffer_unchanged": int(
                coordinate_hash_before == coordinate_hash_after
            ),
            "row_column_rotation_response_valid": rotation_valid,
            "passed": int(
                math.isfinite(symmetry_variance)
                and symmetry_variance > 0.0
                and removed_variance <= 1.0e-24
                and zero_translation["passed"] == 1
                and nonzero_translation["passed"] == 0
                and fixture_replay
                and rotation_valid
            ),
        }
    )
    synthetic = _synthetic_coordinate_control(run_dir, args, train_inputs, validation_inputs)
    null = _exact_null_control(args, train_inputs, validation_inputs)
    zero_model = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=True).to(args.device)
    zero = stream_zero_initialization(
        zero_model,
        {"train": train_inputs, "validation": validation_inputs},
        device=args.device,
    )
    audit_model = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=True).to(args.device)
    audit_rows = np.arange(min(32, train_inputs.row_count), dtype=np.int64)
    audit_batch = train_inputs.batch(audit_rows, device=args.device)
    forbidden_rejected = 0
    try:
        audit_model(audit_batch, path_id=torch.zeros(len(audit_rows)))  # type: ignore[call-arg]
    except TypeError:
        forbidden_rejected = 1
    mutated_arrays = dict(train_inputs.arrays)
    for field in ("path_id", "outer_step", "sample_key"):
        if field in mutated_arrays:
            mutated_arrays[field] = np.ascontiguousarray(
                np.asarray(mutated_arrays[field]) + 97
            )
    mutated_store = HostInputStore.from_arrays(
        mutated_arrays, role="train", cache_root=run_dir, index=train_inputs.index
    )
    with torch.no_grad():
        whole = audit_model(audit_batch)
        mutated_output = audit_model(mutated_store.batch(audit_rows, device=args.device))
        split = torch.cat(
            [audit_model(audit_batch.index_select(torch.arange(start, min(len(audit_rows), start + 16), device=args.device)))
             for start in range(0, len(audit_rows), 16)],
            dim=0,
        )
    batch_error = float(torch.max(torch.abs(whole.to(torch.float64) - split.to(torch.float64))).cpu())
    audit_field_error = float(
        torch.max(torch.abs(whole.to(torch.float64) - mutated_output.to(torch.float64))).cpu()
    )
    peak_fraction = 0.0
    if torch.device(args.device).type == "cuda":
        peak_fraction = torch.cuda.max_memory_allocated(torch.device(args.device)) / torch.cuda.get_device_properties(torch.device(args.device)).total_memory
    restart = _semantic(
        {
            "schema": RUN_SCHEMA + "-firewall-memory-restart-control",
            "schema_version": 1,
            "exact_model_inputs_only": 1,
            "maximum_forward_batch": 32,
            "streaming_only": 1,
            "restart_state_hash_bound": int(synthetic.get("progress_resume_verified", 0)),
            "forbidden_audit_fields_absent": forbidden_rejected,
            "forbidden_kwargs_rejected": forbidden_rejected,
            "monkeypatched_audit_field_maximum_output_error": audit_field_error,
            "monkeypatched_audit_fields_cannot_influence_output": int(audit_field_error == 0.0),
            "full_vs_batch32_maximum_error": batch_error,
            "full_vs_batch32_valid": int(batch_error <= 1.0e-12),
            "peak_cuda_memory_fraction": peak_fraction,
            "peak_cuda_memory_valid": int(peak_fraction <= 0.80),
            "zero_initialization": zero,
            "passed": int(
                zero.get("passed", 0)
                and forbidden_rejected
                and batch_error <= 1.0e-12
                and audit_field_error == 0.0
                and synthetic.get("progress_resume_verified", 0)
                and peak_fraction <= 0.80
            ),
        }
    )
    artifacts = {
        "control_coordinate_geometry.json": geometry,
        "control_initialization_identity.json": initialization,
        "control_symmetry_break.json": symmetry,
        "control_synthetic_coordinate_teacher.json": synthetic,
        "control_exact_model_null.json": null,
        "control_firewall_memory_restart.json": restart,
    }
    for name, value in artifacts.items():
        atomic_write_json(run_dir / name, value)
    flags = {name: 1 for name in _gate.CONTROLS_FLAGS}
    flags.update(
        {
            "coordinate_geometry_control_valid": int(geometry.get("passed", 0)),
            "initialization_identity_control_valid": int(initialization.get("passed", 0)),
            "symmetry_break_control_valid": int(symmetry.get("passed", 0)),
            "synthetic_coordinate_teacher_valid": int(synthetic.get("passed", 0)),
            "exact_model_null_valid": int(null.get("passed", 0)),
            "firewall_batch_memory_restart_valid": int(restart.get("passed", 0)),
            "physical_train_labels_unopened": int(
                not (run_dir / "physical_train_label_open.json").exists()
            ),
            "validation_labels_unopened": int(
                not (run_dir / "validation_label_open.json").exists()
            ),
            "confirmation_namespace_unopened": int(
                not (run_dir / "confirmation_namespace_open.json").exists()
            ),
        }
    )
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-controls-metrics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            **flags,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "controls_metrics.json", metrics)
    gate = _gate.evaluate_controls_gate(metrics)
    gate["test_only"] = int(args.test_only)
    gate["authorizing"] = int(not args.test_only)
    atomic_write_json(run_dir / "controls_gate.json", gate)
    _stage_seal(
        run_dir,
        "controls",
        (
            "controls_open.json",
            *artifacts,
            "synthetic_coordinate_teacher_per_path.csv",
            "synthetic_coordinate_teacher_contract.json",
            "synthetic_coordinate_teacher_progress.pt",
            "controls_metrics.json",
            "controls_gate.json",
        ),
    )
    return gate


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _existing_stage(run_dir, "train")
    if existing is not None:
        return existing
    _workflow.validate_stage_entry(run_dir, "train")
    _workflow.assert_role_firewall(run_dir, "train")
    controls_seal = _workflow.verify_artifact_seal(
        run_dir, _workflow.STAGE_SEAL_NAMES["controls"]
    )
    opening = _workflow.open_label_role(
        run_dir,
        "train",
        binding={
            "controls_artifact_seal_sha256": controls_seal["semantic_sha256"],
            "cache_train_index_sha256": file_fingerprint(
                run_dir / "eager_cache" / "train_index.json"
            ),
            "purpose": "physical_training",
        },
    )
    authorization = LabelOpenAuthorization(
        cache_root=run_dir,
        role="train",
        purpose="physical_training",
        opening_seal_sha256=opening["semantic_sha256"],
    )
    train_inputs = open_external_input_store(run_dir, "train")
    train_labels = open_external_label_store(
        run_dir, "train", authorization=authorization
    )
    if args.test_only and not np.any(train_labels.row_array("denoising_target")):
        arrays = dict(train_labels.arrays)
        row = np.arange(train_labels.row_count, dtype=np.float64)[:, None]
        edge = np.arange(392, dtype=np.float64)[None, :]
        arrays["denoising_target"] = np.ascontiguousarray(
            0.05 + 1.0e-4 * np.sin(row + edge), dtype=np.float64
        )
        train_labels = HostLabelStore.from_arrays(
            arrays, authorization=authorization, index=train_labels.index
        )
    checkpoint = _load_json(run_dir / "training_checkpoint_plan.json")
    reports: list[dict[str, Any]] = []
    for seed in _workflow.MODEL_SEEDS:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(
            zero_residual=True
        )
        report = _workflow.train_frequency1_coordinate_candidate(
            model=model,
            train_inputs=train_inputs,
            train_labels=train_labels,
            seed=seed,
            progress_path=run_dir / "training_progress" / f"seed-{seed}.pt",
            checkpoint_root=run_dir / "checkpoints" / "physical",
            scientific_config_sha256=_load_json(run_dir / "scientific_config.json")[
                "semantic_sha256"
            ],
            maximum_updates=int(checkpoint["maximum_updates"]),
            checkpoint_interval=int(checkpoint["checkpoint_interval"]),
            device=args.device,
        )
        reports.append(report)
        atomic_write_json(run_dir / "training_tasks" / f"seed-{seed}.json", report)
        atomic_write_csv(
            run_dir / "training_trajectory" / f"seed-{seed}.csv",
            report["history"],
        )
    checkpoints = sorted(
        (dict(row) for report in reports for row in report["checkpoints"]),
        key=lambda row: (int(row["seed"]), int(row["update"])),
    )
    inventory = _semantic(
        {
            "schema": RUN_SCHEMA + "-candidate-inventory",
            "schema_version": 1,
            "canonical_order": "model_seed_then_update",
            "checkpoints": checkpoints,
            "checkpoint_count": len(checkpoints),
            "update_zero_checkpoint_count": sum(int(row["update"]) == 0 for row in checkpoints),
            "nonzero_candidate_count": sum(int(row["update"]) > 0 for row in checkpoints),
            "validation_evidence_used": 0,
            "validation_labels_opened": 0,
            "confirmation_namespace_opened": 0,
        }
    )
    atomic_write_json(run_dir / "candidate_inventory.json", inventory)
    expected_checkpoints = len(_workflow.MODEL_SEEDS) * len(checkpoint["checkpoint_updates"])
    complete = all(int(report.get("complete", 0)) == 1 for report in reports)
    flags = {name: 1 for name in _gate.TRAIN_FLAGS}
    flags.update(
        {
            "physical_train_label_open_order_valid": int(
                opening.get("opened_once", 0) == 1
                and (run_dir / "controls_gate.json").stat().st_mtime_ns
                <= (run_dir / "physical_train_label_open.json").stat().st_mtime_ns
            ),
            "three_physical_tasks_complete": int(complete and len(reports) == 3),
            "all_checkpoints_complete": int(len(checkpoints) == expected_checkpoints),
            "candidate_inventory_valid": int(
                inventory["update_zero_checkpoint_count"] == 3
                and inventory["nonzero_candidate_count"]
                == int(checkpoint["nonzero_candidate_count"])
            ),
            "training_target_scale_training_only": int(
                all(float(report["target_scale"]) > 0.0 for report in reports)
            ),
            "finite_training_outputs": int(
                all(
                    math.isfinite(float(row[key]))
                    for report in reports
                    for row in report["history"]
                    for key in ("train_raw_mse", "scaled_loss", "preclip_gradient_norm")
                )
            ),
            "batch_and_memory_contract_valid": int(
                all(
                    int(report["model_call_batches"]["all_calls_within_limit"]) == 1
                    for report in reports
                )
            ),
            "validation_labels_unopened": int(
                not (run_dir / "validation_label_open.json").exists()
            ),
            "confirmation_namespace_unopened": int(
                not (run_dir / "confirmation_namespace_open.json").exists()
            ),
        }
    )
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-train-metrics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            **flags,
            "checkpoint_count": len(checkpoints),
            "nonzero_candidate_count": inventory["nonzero_candidate_count"],
            "physical_training_performed": 1,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "train_metrics.json", metrics)
    gate = _gate.evaluate_train_gate(metrics)
    gate["test_only"] = int(args.test_only)
    gate["authorizing"] = int(not args.test_only)
    atomic_write_json(run_dir / "train_gate.json", gate)
    names = [
        "physical_train_label_open.json",
        "candidate_inventory.json",
        "train_metrics.json",
        "train_gate.json",
    ]
    names.extend(f"training_tasks/seed-{seed}.json" for seed in _workflow.MODEL_SEEDS)
    names.extend(f"training_trajectory/seed-{seed}.csv" for seed in _workflow.MODEL_SEEDS)
    names.extend(str(row["checkpoint_path"]) for row in checkpoints)
    names.extend(f"training_progress/seed-{seed}.pt" for seed in _workflow.MODEL_SEEDS)
    _stage_seal(run_dir, "train", names)
    return gate


def _load_frequency1_candidate(
    run_dir: Path, candidate: Mapping[str, Any], device: str | torch.device
) -> nn.Module:
    path = Path(str(candidate["checkpoint_path"]))
    if not path.is_absolute():
        path = run_dir / path
    if file_fingerprint(path) != candidate["checkpoint_file_sha256"]:
        raise ArtifactCompatibilityError("candidate checkpoint file changed")
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    state = payload.get("state_dict")
    if (
        int(payload.get("seed", -1)) != int(candidate["seed"])
        or int(payload.get("update", -1)) != int(candidate["update"])
        or not isinstance(state, Mapping)
        or state_dict_sha256(state) != candidate["state_sha256"]
    ):
        raise ArtifactCompatibilityError("candidate checkpoint identity changed")
    model = _coordinate.FrequencyOneCoordinateZeroBaselinePredictor(
        zero_residual=False
    ).to(torch.device(device))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _test_validation_values(
    path_ids: np.ndarray, candidate_count: int, outcome: str
) -> np.ndarray:
    # Nonzero path variation keeps every standard error finite without a floor.
    path = np.arange(path_ids.size, dtype=np.float64)[:, None, None]
    component = np.arange(_workflow.COMPONENT_COUNT, dtype=np.float64)[None, None, :]
    candidate = np.arange(candidate_count, dtype=np.float64)[None, :, None]
    center = 2.0 if outcome == "nominee" else -2.0
    return np.ascontiguousarray(
        center + 0.01 * np.sin(path + 0.03 * component + 0.1 * candidate),
        dtype=np.float64,
    )


def _selection_replicates(args: argparse.Namespace) -> tuple[int, int]:
    if args.test_only:
        value = int(args.test_bootstrap_replicates)
        return value, value
    return 50_000, 1_000


def _select_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _existing_stage(run_dir, "select")
    if existing is not None:
        return existing
    _workflow.validate_stage_entry(run_dir, "select")
    _workflow.assert_role_firewall(run_dir, "select")
    inventory = _load_json(run_dir / "candidate_inventory.json")
    candidates = [
        dict(row) for row in inventory["checkpoints"] if int(row["update"]) > 0
    ]
    plan = _load_json(run_dir / "selection_inference_plan.json")
    path_plan = _load_json(run_dir / "path_id_plan.json")
    validation_paths = np.asarray(path_plan["roles"]["validation"], dtype=np.int64)
    inference_paths = validation_paths
    if args.test_only and inference_paths.size < 8:
        inference_paths = np.arange(0x1A00, 0x1A08, dtype=np.int64)
    replicates, shard_size = _selection_replicates(args)
    validation_already_open = (run_dir / "validation_label_open.json").is_file()
    if validation_already_open:
        counts, count_records = _workflow.load_bootstrap_count_shards(
            run_dir / "bootstrap" / "selection",
            seed=_workflow.SELECTION_BOOTSTRAP_SEED,
            namespace=_workflow.SELECTION_NAMESPACE,
            path_count=int(inference_paths.size),
            replicates=replicates,
            shard_size=shard_size,
        )
        del counts
        counts = count_records
        intent = _load_json(run_dir / "selection_opening_intent.json")
        intent_seal = _workflow.verify_artifact_seal(
            run_dir, "selection_opening_intent_seal.json"
        )
    else:
        counts = _workflow.prepare_bootstrap_count_shards(
            run_dir / "bootstrap" / "selection",
            seed=_workflow.SELECTION_BOOTSTRAP_SEED,
            namespace=_workflow.SELECTION_NAMESPACE,
            path_count=int(inference_paths.size),
            replicates=replicates,
            shard_size=shard_size,
        )
        intent = _semantic(
            {
                "schema": RUN_SCHEMA + "-selection-opening-intent",
                "schema_version": 1,
                "candidate_order": [
                    [int(row["seed"]), int(row["update"])] for row in candidates
                ],
                "candidate_inventory_sha256": inventory["semantic_sha256"],
                "family_names_sha256": _workflow.FAMILY_NAMES_SHA256,
                "search_family_names_sha256": plan["search_family_names_sha256"],
                "validation_path_ids": validation_paths.tolist(),
                "numeric_fixture_path_ids": inference_paths.tolist(),
                "bootstrap_count_semantic_sha256": [row["semantic_sha256"] for row in counts],
                "bootstrap_counts_committed_before_labels": 1,
                "confirmation_namespace_opened": 0,
            }
        )
        atomic_write_json(run_dir / "selection_opening_intent.json", intent)
        intent_seal = _workflow.seal_artifacts(
            run_dir,
            [
                "selection_opening_intent.json",
                *(
                    path.relative_to(run_dir).as_posix()
                    for path in sorted((run_dir / "bootstrap" / "selection").glob("count-*"))
                    if path.is_file()
                ),
            ],
            "selection_opening_intent_seal.json",
        )
    opening = _workflow.open_label_role(
        run_dir,
        "validation",
        binding={
            "selection_opening_intent_seal_sha256": intent_seal["semantic_sha256"],
            "purpose": "validation_selection",
        },
    )
    authorization = LabelOpenAuthorization(
        cache_root=run_dir,
        role="validation",
        purpose="validation_selection",
        opening_seal_sha256=opening["semantic_sha256"],
    )
    validation_inputs = open_external_input_store(run_dir, "validation")
    validation_labels = open_external_label_store(
        run_dir, "validation", authorization=authorization
    )
    result = None
    ranking: dict[str, Any]
    table: _workflow.Frequency1CandidateTable | None = None
    if not candidates:
        ranking = {
            "decision": "no_frequency1_coordinate_validation_candidate",
            "candidate_count": 0,
            "eligible_candidate_count": 0,
            "selected_seed": None,
            "selected_update": 0,
            "logical_update_zero_selected": 1,
            "confirmation_authorized": 0,
        }
        atomic_write_csv(run_dir / "selection_candidate_table.csv", [])
    else:
        values = np.empty(
            (inference_paths.size, len(candidates), _workflow.COMPONENT_COUNT),
            dtype=np.float64,
        )
        fixture_values = (
            _test_validation_values(inference_paths, len(candidates), args.test_selection_outcome)
            if args.test_only else None
        )
        candidate_evidence_root = run_dir / "selection" / "candidates"
        candidate_evidence_root.mkdir(parents=True, exist_ok=True)
        for index, candidate in enumerate(candidates):
            stem = f"seed-{int(candidate['seed'])}-update-{int(candidate['update']):04d}"
            data_path = candidate_evidence_root / f"{stem}.npz"
            metadata_path = candidate_evidence_root / f"{stem}.json"
            if metadata_path.is_file():
                metadata = _load_json(metadata_path)
                arrays = _load_npz(data_path)
                if (
                    metadata.get("checkpoint_file_sha256") != candidate["checkpoint_file_sha256"]
                    or metadata.get("artifact_sha256") != file_fingerprint(data_path)
                    or not np.array_equal(arrays.get("path_ids"), inference_paths)
                    or arrays.get("path_values", np.empty(0)).shape
                    != (inference_paths.size, _workflow.COMPONENT_COUNT)
                ):
                    raise ArtifactCompatibilityError("candidate validation evidence changed")
                values[:, index, :] = arrays["path_values"]
                continue
            if args.test_only:
                candidate_values = fixture_values[:, index, :]
            else:
                model = _load_frequency1_candidate(run_dir, candidate, args.device)
                risk = _workflow.candidate_path_risk_table(
                    model,
                    validation_inputs,
                    validation_labels,
                    expected_path_ids=validation_paths,
                    device=args.device,
                )
                candidate_values = risk.path_values
            artifact = _atomic_npz(
                data_path,
                path_ids=inference_paths,
                path_values=np.asarray(candidate_values, dtype=np.float64),
            )
            atomic_write_json(
                metadata_path,
                _semantic(
                    {
                        "schema": RUN_SCHEMA + "-candidate-validation-evidence",
                        "schema_version": 1,
                        "seed": int(candidate["seed"]),
                        "update": int(candidate["update"]),
                        "checkpoint_file_sha256": candidate["checkpoint_file_sha256"],
                        "artifact_sha256": artifact["sha256"],
                        "path_ids": inference_paths.tolist(),
                        "component_count": _workflow.COMPONENT_COUNT,
                        "complete": 1,
                    }
                ),
            )
            values[:, index, :] = candidate_values
        table = _workflow.build_candidate_table(
            seeds=np.asarray([row["seed"] for row in candidates], dtype=np.int64),
            updates=np.asarray([row["update"] for row in candidates], dtype=np.int64),
            path_ids=inference_paths,
            path_values=values,
            forbidden_path_ids=path_plan["roles"]["confirmation"],
        )
        _atomic_npz(
            run_dir / "selection_candidate_path_values.npz",
            seeds=table.seeds,
            updates=table.updates,
            path_ids=table.path_ids,
            path_values=table.path_values,
        )
        result, ranking = _workflow.restartable_selection_max_t(
            table,
            count_directory=run_dir / "bootstrap" / "selection",
            maxima_directory=run_dir / "bootstrap" / "selection" / "maxima",
            replicates=replicates,
            shard_size=shard_size,
        )
        atomic_write_csv(run_dir / "selection_candidate_table.csv", ranking["candidate_rows"])
        _atomic_npz(
            run_dir / "selection_max_t.npz",
            point_estimates=result.point_estimates,
            standard_errors=result.standard_errors,
            lower_bounds=result.lower_bounds,
            maxima=result.maxima,
        )
    inference = _semantic(
        {
            "schema": RUN_SCHEMA + "-selection-inference",
            "schema_version": 1,
            **ranking,
            "validation_label_open_sha256": opening["semantic_sha256"],
            "negative_values_truncated": 0,
            "standard_error_floor_used": 0,
            "test_only": int(args.test_only),
        }
    )
    atomic_write_json(run_dir / "selection_inference.json", inference)
    if ranking["decision"] == "frequency1_coordinate_validation_nominee_sealed":
        selected = next(
            row for row in candidates
            if int(row["seed"]) == int(ranking["selected_seed"])
            and int(row["update"]) == int(ranking["selected_update"])
        )
        decision = _semantic(
            {
                **inference,
                "schema": RUN_SCHEMA + "-selection-decision",
                "decision": "frequency1_coordinate_validation_nominee_sealed",
                "checkpoint_path": selected["checkpoint_path"],
                "checkpoint_file_sha256": selected["checkpoint_file_sha256"],
                "state_sha256": selected["state_sha256"],
                "confirmation_authorized": 1,
            }
        )
    else:
        decision = _semantic(
            {
                **inference,
                "schema": RUN_SCHEMA + "-selection-decision",
                "decision": "no_frequency1_coordinate_validation_candidate",
                "confirmation_authorized": 0,
                "confirmation_forbidden": 1,
            }
        )
    atomic_write_json(run_dir / "selection_decision.json", decision)
    positive = decision["decision"] == "frequency1_coordinate_validation_nominee_sealed"
    flags = {name: 1 for name in _gate.SELECT_FLAGS}
    flags["all_228_simultaneous_lower_bounds_positive"] = int(positive)
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-select-metrics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            **flags,
            "no_validation_candidate": int(not positive),
            "candidate_count": len(candidates),
            "component_count": _workflow.COMPONENT_COUNT,
            "search_family_size": len(candidates) * _workflow.COMPONENT_COUNT,
            "bootstrap_replicates": replicates,
            "validation_selection_performed": 1,
            "physical_training_performed": 1,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "select_metrics.json", metrics)
    gate = _gate.evaluate_select_gate(metrics)
    gate["test_only"] = int(args.test_only)
    gate["authorizing"] = int(not args.test_only)
    gate["nominee"] = decision if positive else None
    atomic_write_json(run_dir / "select_gate.json", gate)
    names = [
        "selection_opening_intent.json",
        "selection_opening_intent_seal.json",
        "validation_label_open.json",
        "selection_candidate_table.csv",
        "selection_inference.json",
        "selection_decision.json",
        "select_metrics.json",
        "select_gate.json",
    ]
    if table is not None:
        names.extend(["selection_candidate_path_values.npz", "selection_max_t.npz"])
    names.extend(
        path.relative_to(run_dir).as_posix()
        for root in (run_dir / "bootstrap" / "selection", run_dir / "selection" / "candidates")
        if root.exists()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    _stage_seal(run_dir, "select", names)
    return gate


def _confirmation_test_values(path_ids: np.ndarray, outcome: str) -> np.ndarray:
    path = np.arange(path_ids.size, dtype=np.float64)[:, None]
    component = np.arange(_workflow.COMPONENT_COUNT, dtype=np.float64)[None, :]
    center = 2.0 if outcome == "pass" else -2.0
    return np.ascontiguousarray(
        center + 0.01 * np.cos(path + 0.07 * component), dtype=np.float64
    )


def _production_confirmation_execution(
    run_dir: Path,
    args: argparse.Namespace,
    namespace: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    source = _source_target(args)

    def load_model(_root: Path, candidate: Mapping[str, Any], device: torch.device) -> nn.Module:
        return _load_frequency1_candidate(run_dir, candidate, device)

    def source_loader(_unused: Path) -> np.ndarray:
        return np.array(source, copy=True, order="C")

    original_risks = _v3_cli._confirmation_risks_from_execution

    def reductions_only(execution: Any, *, model: nn.Module) -> tuple[Any, None]:
        risks, _forbidden_anchor = original_risks(execution, model=model)
        return risks, None

    setattr(args, "parent_coarse_residual_run_dir", args.parent_memory_v3_run_dir)
    with _patched(
        _v3_cli,
        ROOT_SEED=_workflow.ROOT_SEED,
        RUN_SCHEMA=RUN_SCHEMA,
        _load_candidate_model=load_model,
        _cohorts=lambda plan, kind: _workflow.eager_cohorts(plan, kind),
        _confirmation_risks_from_execution=reductions_only,
        _load_confirmation_shard=_load_reductions_only_confirmation_shard,
    ), _patched(_legacy, _load_source_target=source_loader):
        compatible_namespace = {
            **dict(namespace),
            "path_ids": list(namespace["binding"]["path_ids"]),
        }
        result = _v3_cli._run_confirmation_execution(
            run_dir, args, compatible_namespace, selection
        )
    if _forbidden_confirmation_artifacts(run_dir):
        raise ArtifactCompatibilityError("forbidden confirmation control anchor was persisted")
    return result


def _load_reductions_only_confirmation_shard(
    run_dir: Path,
    *,
    cohort: Any,
    start_step: int,
    current: np.ndarray,
    selected_step: int | None,
    namespace_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Load an immutable confirmation shard with no control-anchor payload.

    This is the narrow compatibility seam for the frequency-one workflow.  It
    retains every inherited continuation/risk/hash check while recognizing
    that an explicitly null anchor binding is the required representation for
    this reductions-only protocol.
    """

    state_path, risk_path, anchor_path, metadata_path = _v3_cli._confirmation_shard_paths(
        run_dir, cohort_index=int(cohort.index), start_step=int(start_step)
    )
    if not state_path.is_file() or not metadata_path.is_file():
        return None
    try:
        record = _load_json(metadata_path)
        body = dict(record)
        semantic = body.pop("semantic_sha256", None)
        if (
            semantic != config_fingerprint(body)
            or record.get("schema") != RUN_SCHEMA + "-confirmation-shard"
            or int(record.get("committed", 0)) != 1
            or record.get("namespace_sha256") != str(namespace_sha256)
            or record.get("path_ids") != list(cohort.path_ids)
            or int(record.get("cohort_index", -1)) != int(cohort.index)
            or int(record.get("start_step", -1)) != int(start_step)
            or record.get("selected_step") != selected_step
            or record.get("input_state_sha256") != _v3_cli._array_sha(current)
            or record.get("state_file_sha256") != file_fingerprint(state_path)
            or record.get("control_anchor_file_sha256") is not None
            or anchor_path.exists()
            or int(record.get("raw_confirmation_inputs_persisted", -1)) != 0
            or int(record.get("raw_confirmation_labels_persisted", -1)) != 0
        ):
            return None
        execution = record.get("execution")
        if not isinstance(execution, Mapping):
            return None
        identity = execution.get("identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("cohort_kind") != "confirmation"
            or int(identity.get("cohort_index", -1)) != int(cohort.index)
            or int(identity.get("start_step", -1)) != int(start_step)
            or int(identity.get("step_count", -1)) != 8
            or execution.get("path_ids") != list(cohort.path_ids)
            or execution.get("path_roles") != list(cohort.path_roles)
            or execution.get("selected_step") != selected_step
            or execution.get("input_state_sha256") != record.get("input_state_sha256")
            or int(execution.get("raw_payload_persisted", -1)) != 0
        ):
            return None
        final = _load_npz(state_path).get("final_states")
        if (
            not isinstance(final, np.ndarray)
            or final.dtype != np.float64
            or final.shape != (len(cohort.path_ids), STATE_SIZE)
            or not np.isfinite(final).all()
            or record.get("final_state_sha256") != _v3_cli._array_sha(final)
        ):
            return None
        if selected_step is None:
            if record.get("risk_file_sha256") is not None or risk_path.exists():
                return None
        else:
            if (
                not risk_path.is_file()
                or record.get("risk_file_sha256") != file_fingerprint(risk_path)
            ):
                return None
            risks = _load_npz(risk_path)
            required = {
                "sample_keys",
                "path_ids",
                "outer_steps",
                "phases",
                "midpoint_indices",
                "model_vs_zero",
            }
            if set(risks) != required:
                return None
            row_count = int(risks["sample_keys"].size)
            if (
                row_count <= 0
                or any(np.asarray(risks[name]).shape != (row_count,) for name in required)
                or np.unique(risks["sample_keys"]).size != row_count
                or not np.isfinite(np.asarray(risks["model_vs_zero"], dtype=np.float64)).all()
                or not np.all(np.asarray(risks["outer_steps"]) == int(selected_step))
                or not set(np.asarray(risks["path_ids"], dtype=np.int64).tolist())
                <= set(int(value) for value in cohort.path_ids)
                or any("target" in name or "later_full_state" in name for name in risks)
            ):
                return None
        return np.ascontiguousarray(final), record
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError):
        return None


def _forbidden_confirmation_artifacts(run_dir: Path) -> list[str]:
    forbidden_tokens = (
        "control_anchor",
        "control-anchor",
        "raw_input",
        "raw-input",
        "raw_label",
        "raw-label",
        "raw_prediction",
        "raw-prediction",
    )
    root = run_dir / "confirmation"
    if not root.exists():
        return []
    return [
        path.relative_to(run_dir).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and any(token in path.name.lower() for token in forbidden_tokens)
    ]


def _confirm_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _existing_stage(run_dir, "confirm")
    if existing is not None:
        return existing
    _workflow.validate_stage_entry(run_dir, "confirm")
    _workflow.assert_role_firewall(run_dir, "confirm")
    selection = _load_json(run_dir / "selection_decision.json")
    if selection.get("decision") != "frequency1_coordinate_validation_nominee_sealed":
        raise ArtifactCompatibilityError("confirmation requires a sealed nonzero nominee")
    path_plan = _load_json(run_dir / "path_id_plan.json")
    confirmation_paths = np.asarray(path_plan["roles"]["confirmation"], dtype=np.int64)
    inference_paths = confirmation_paths
    if args.test_only and inference_paths.size < 8:
        inference_paths = np.arange(0x1B00, 0x1B08, dtype=np.int64)
    replicates, shard_size = _selection_replicates(args)
    namespace_already_open = (run_dir / "confirmation_namespace_open.json").is_file()
    if namespace_already_open:
        count_arrays, count_records = _workflow.load_bootstrap_count_shards(
            run_dir / "bootstrap" / "confirmation",
            seed=_workflow.CONFIRMATION_BOOTSTRAP_SEED,
            namespace=_workflow.CONFIRMATION_NAMESPACE,
            path_count=int(inference_paths.size),
            replicates=replicates,
            shard_size=shard_size,
        )
        del count_arrays
        intent = _load_json(run_dir / "confirmation_opening_intent.json")
        intent_seal = _workflow.verify_artifact_seal(
            run_dir, "confirmation_opening_intent_seal.json"
        )
    else:
        count_records = _workflow.prepare_bootstrap_count_shards(
            run_dir / "bootstrap" / "confirmation",
            seed=_workflow.CONFIRMATION_BOOTSTRAP_SEED,
            namespace=_workflow.CONFIRMATION_NAMESPACE,
            path_count=int(inference_paths.size),
            replicates=replicates,
            shard_size=shard_size,
        )
        intent = _semantic(
            {
                "schema": RUN_SCHEMA + "-confirmation-opening-intent",
                "schema_version": 1,
                "selection_decision_sha256": selection["semantic_sha256"],
                "selected_seed": int(selection["selected_seed"]),
                "selected_update": int(selection["selected_update"]),
                "selected_state_sha256": selection["state_sha256"],
                "checkpoint_file_sha256": selection["checkpoint_file_sha256"],
                "confirmation_path_ids": confirmation_paths.tolist(),
                "numeric_fixture_path_ids": inference_paths.tolist(),
                "bootstrap_seed": _workflow.CONFIRMATION_BOOTSTRAP_SEED,
                "bootstrap_namespace": _workflow.CONFIRMATION_NAMESPACE,
                "bootstrap_count_semantic_sha256": [
                    row["semantic_sha256"] for row in count_records
                ],
                "bootstrap_counts_committed_before_paths": 1,
                "second_confirmation_authorized": 0,
            }
        )
        atomic_write_json(run_dir / "confirmation_opening_intent.json", intent)
        intent_seal = _workflow.seal_artifacts(
            run_dir,
            [
                "confirmation_opening_intent.json",
                *(
                    path.relative_to(run_dir).as_posix()
                    for path in sorted((run_dir / "bootstrap" / "confirmation").glob("count-*"))
                    if path.is_file()
                ),
            ],
            "confirmation_opening_intent_seal.json",
        )
    namespace = _workflow.open_label_role(
        run_dir,
        "confirmation",
        binding={
            "confirmation_opening_intent_seal_sha256": intent_seal["semantic_sha256"],
            "selected_state_sha256": selection["state_sha256"],
            "path_ids": confirmation_paths.tolist(),
            "purpose": "fresh_confirmation",
        },
    )
    if args.test_only:
        path_values = _confirmation_test_values(
            inference_paths, args.test_confirmation_outcome
        )
        confirmation_root = run_dir / "confirmation"
        confirmation_root.mkdir(parents=True, exist_ok=True)
        for path_id, values in zip(inference_paths.tolist(), path_values, strict=True):
            _atomic_npz(
                confirmation_root / f"path-{path_id:05x}-reductions.npz",
                path_id=np.asarray([path_id], dtype=np.int64),
                path_values=np.asarray(values, dtype=np.float64),
            )
        progress = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-confirmation-progress",
                "schema_version": 1,
                "completed_path_ids": inference_paths.tolist(),
                "complete": 1,
                "raw_inputs_persisted": 0,
                "raw_labels_persisted": 0,
                "raw_predictions_persisted": 0,
                "test_only": 1,
            }
        )
        atomic_write_json(run_dir / "confirmation" / "progress.json", progress)
        execution = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-confirmation-execution",
                "schema_version": 1,
                "path_ids": inference_paths.tolist(),
                "streaming_reductions_only": 1,
                "complete": 1,
                "test_only": 1,
            }
        )
        atomic_write_json(run_dir / "confirmation_execution.json", execution)
    else:
        execution = _production_confirmation_execution(
            run_dir, args, namespace, selection
        )
        arrays = _load_npz(run_dir / "confirmation_path_risks.npz")
        path_values = np.asarray(arrays["path_values"], dtype=np.float64)
        inference_paths = np.asarray(arrays["path_ids"], dtype=np.int64)
    result, inference = _workflow.restartable_confirmation_max_t(
        path_values,
        path_ids=inference_paths,
        count_directory=run_dir / "bootstrap" / "confirmation",
        maxima_directory=run_dir / "bootstrap" / "confirmation" / "maxima",
        replicates=replicates,
        shard_size=shard_size,
    )
    _atomic_npz(
        run_dir / "confirmation_max_t.npz",
        point_estimates=result.point_estimates,
        standard_errors=result.standard_errors,
        lower_bounds=result.lower_bounds,
        maxima=result.maxima,
    )
    positive = bool(result.passed)
    inference_record = _semantic(
        {
            "schema": RUN_SCHEMA + "-confirmation-inference",
            "schema_version": 1,
            **inference,
            "decision": (
                "exact_rb_frequency1_coordinate_boundary_tangent_time_local_signal_confirmed"
                if positive else "frequency1_coordinate_signal_not_confirmed"
            ),
            "minimum_lower_bound": float(np.min(result.lower_bounds)),
            "all_228_simultaneous_lower_bounds_positive": int(positive),
            "negative_values_truncated": 0,
            "standard_error_floor_used": 0,
        }
    )
    atomic_write_json(run_dir / "confirmation_inference.json", inference_record)
    if args.test_only:
        execution_valid = True
    else:
        aggregate = execution.get("aggregate", {})
        total = int(aggregate.get("transition_count", 0))
        certified = int(aggregate.get("certified_count", 0))
        forbidden = sum(
            int(value) for value in aggregate.get("forbidden_counts", {}).values()
        )
        elapsed = math.fsum(
            float(
                _load_json(
                    _v3_cli._confirmation_shard_paths(
                        run_dir,
                        cohort_index=int(row["cohort_index"]),
                        start_step=int(row["start_step"]),
                    )[3]
                )["complete_pipeline_elapsed_seconds"]
            )
            for row in execution.get("records", [])
        )
        fallback = int(aggregate.get("fallback_count", 0))
        fallback_elapsed = float(aggregate.get("fallback_elapsed_seconds", 0.0))
        row_count = int(_load_json(run_dir / "confirmation_risk_summary.json")["row_count"])
        cache_seconds = float(_load_json(run_dir / "cache_metrics.json").get("cache_elapsed_seconds", 0.0))
        execution_valid = bool(
            inference_paths.size == 64
            and row_count == 114_688
            and total == 134_873_088
            and certified == total
            and forbidden == 0
            and float(aggregate.get("maximum_mass_error", math.inf)) <= 2.0e-12
            and fallback / max(total, 1) <= 1.0e-4
            and fallback_elapsed / max(elapsed, np.finfo(float).tiny) <= 0.10
            and total / max(elapsed, np.finfo(float).tiny) >= 1_300.0
            and float(aggregate.get("maximum_peak_memory_fraction", math.inf)) <= 0.80
            and cache_seconds + elapsed <= 160.0 * 3600.0
        )
    flags = {name: 1 for name in _gate.CONFIRM_FLAGS}
    flags.update(
        {
            "sealed_nominee_unchanged": int(
                intent["selected_state_sha256"] == selection["state_sha256"]
            ),
            "confirmation_opened_once": int(namespace.get("opened_once", 0)),
            "confirmation_paths_valid": int(
                args.test_only or np.array_equal(inference_paths, confirmation_paths)
            ),
            "streaming_reductions_only": int(
                not _forbidden_confirmation_artifacts(run_dir)
                and not (run_dir / "eager_cache" / "confirmation").exists()
            ),
            "confirmation_inference_valid": int(
                execution_valid
                and result.lower_bounds.shape == (1, _workflow.COMPONENT_COUNT)
            ),
        }
    )
    flags["all_228_simultaneous_lower_bounds_positive"] = int(positive)
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-confirmation-metrics",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            **flags,
            "signal_not_confirmed": int(not positive),
            "confirmation_path_count": int(inference_paths.size),
            "component_count": _workflow.COMPONENT_COUNT,
            "bootstrap_replicates": replicates,
            "confirmation_performed": 1,
            "validation_selection_performed": 1,
            "physical_training_performed": 1,
            "test_only": int(args.test_only),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "confirmation_metrics.json", metrics)
    gate = _gate.evaluate_confirm_gate(metrics)
    gate["test_only"] = int(args.test_only)
    gate["authorizing"] = int(not args.test_only)
    atomic_write_json(run_dir / "confirm_gate.json", gate)
    # Stable alias required by the artifact contract; both bytes are sealed.
    atomic_write_json(run_dir / "confirmation_gate.json", gate)
    names = [
        "confirmation_opening_intent.json",
        "confirmation_opening_intent_seal.json",
        "confirmation_namespace_open.json",
        "confirmation_execution.json",
        "confirmation_max_t.npz",
        "confirmation_inference.json",
        "confirmation_metrics.json",
        "confirmation_gate.json",
        "confirm_gate.json",
    ]
    names.extend(
        path.relative_to(run_dir).as_posix()
        for root in (run_dir / "bootstrap" / "confirmation", run_dir / "confirmation")
        if root.exists()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    if not args.test_only:
        names.extend(
            name for name in (
                "confirmation_path_risks.npz",
                "confirmation_risk_summary.json",
            ) if (run_dir / name).is_file()
        )
    _stage_seal(run_dir, "confirm", names)
    return gate


def _report_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    _workflow.assert_role_firewall(run_dir, "report")
    prior_closed = False
    for stage in ("preflight", "cache", "controls", "train", "select", "confirm"):
        gate_path = run_dir / f"{stage}_gate.json"
        if not gate_path.is_file():
            continue
        if prior_closed:
            raise ArtifactCompatibilityError("stage evidence exists after a closed decision")
        _workflow.verify_artifact_seal(run_dir, _workflow.STAGE_SEAL_NAMES[stage])
        gate = _load_json(gate_path)
        prior_closed = not _passed(gate)
        if int(gate.get("valid_scientific_negative", 0)) == 1:
            prior_closed = True
    selection = _optional_json(run_dir, "selection_decision.json")
    if (run_dir / "confirmation_namespace_open.json").exists() and (
        not isinstance(selection, Mapping)
        or selection.get("decision") != "frequency1_coordinate_validation_nominee_sealed"
    ):
        raise ArtifactCompatibilityError("confirmation opened without the sealed nominee")
    if isinstance(selection, Mapping) and selection.get("decision") == "no_frequency1_coordinate_validation_candidate":
        if (run_dir / "confirmation_namespace_open.json").exists():
            raise ArtifactCompatibilityError("negative selection opened confirmation")
    if not args.test_only:
        immutability = _provenance.verify_frequency1_coordinate_parent_immutability(
            **_parent_kwargs(args),
            snapshots=_load_json(run_dir / "parent_immutability_snapshot.json"),
        )
    else:
        immutability = _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-terminal-parent-immutability",
                "schema_version": 1,
                "evaluation_status": "evaluated",
                "passed": 1,
                "test_only": 1,
            }
        )
    preflight_immutability = _load_json(run_dir / "parent_immutability_report.json")
    if int(preflight_immutability.get("passed", 0)) != 1 or int(immutability.get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("parent immutability failed at report")
    atomic_write_json(run_dir / "terminal_parent_immutability_report.json", immutability)
    decision = _decision(run_dir)
    lines = [
        "# Frequency-one coordinate Jacobi/RB learnability",
        "",
        f"Decision: `{decision['decision']}`",
        "",
        f"Scientific evidence complete: `{decision['scientific_evidence_complete']}`",
        "",
        "This one-image experiment changes only the frozen rank-4 frequency-one "
        "coordinate stem. It does not authorize a controller trajectory, reconstruction, "
        "reverse sampling, or image sampling.",
        "",
        "A positive result recommends only drafting a separately reviewed controls-only "
        "controller-control plan.",
    ]
    _atomic_text(run_dir / "REPORT.md", "\n".join(lines) + "\n")
    return decision


def _failure_gate(stage: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preflight": _gate.evaluate_preflight_gate,
        "cache": _gate.evaluate_cache_gate,
        "controls": _gate.evaluate_controls_gate,
        "train": _gate.evaluate_train_gate,
        "select": _gate.evaluate_select_gate,
        "confirm": _gate.evaluate_confirm_gate,
    }[stage](metrics)


def _commit_execution_failure(
    run_dir: Path, *, stage: str, error: BaseException
) -> None:
    if stage not in {"preflight", "cache", "controls", "train", "select", "confirm"}:
        _status(
            run_dir,
            state="execution_failed",
            stage=stage,
            message=str(error),
            failure_domain=str(getattr(error, "failure_domain", "workflow_execution")),
            failure_code=str(getattr(error, "failure_code", "frequency1_coordinate_execution_failed")),
        )
        _artifact_registry(run_dir)
        return
    domain = str(getattr(error, "failure_domain", "workflow_execution"))
    code = str(
        getattr(error, "failure_code", f"frequency1_coordinate_{stage}_execution_failed")
    )
    metrics_name = "confirmation_metrics.json" if stage == "confirm" else f"{stage}_metrics.json"
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + f"-{stage}-execution-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            "stage_execution_valid": 0,
            "inference_valid": 0,
            "failure_domain": domain,
            "failure_code": code,
            "message": str(error),
            "physical_training_performed": int((run_dir / "physical_train_label_open.json").exists()),
            "validation_selection_performed": int((run_dir / "validation_label_open.json").exists()),
            "confirmation_performed": int((run_dir / "confirmation_namespace_open.json").exists()),
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / metrics_name, metrics)
    failure_name = f"{stage}_execution_failure.json"
    atomic_write_json(run_dir / failure_name, metrics)
    gate = _failure_gate(stage, metrics)
    gate_name = f"{stage}_gate.json"
    atomic_write_json(run_dir / gate_name, gate)
    names = [metrics_name, failure_name, gate_name]
    if stage == "confirm":
        atomic_write_json(run_dir / "confirmation_gate.json", gate)
        names.append("confirmation_gate.json")
    _stage_seal(run_dir, stage, names)
    _workflow_record(run_dir, "none")
    decision = _decision(run_dir)
    _status(
        run_dir,
        state="execution_failed",
        stage=stage,
        decision=str(decision.get("decision")),
        message=str(error),
        failure_domain=domain,
        failure_code=code,
    )
    _artifact_registry(run_dir)


def _commit_initialization_failure(run_dir: Path, error: BaseException) -> None:
    domain = str(getattr(error, "failure_domain", "parent_provenance"))
    code = str(getattr(error, "failure_code", "frequency1_coordinate_initialization_failed"))
    metrics = _semantic(
        {
            "schema": RUN_SCHEMA + "-initialization-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            "stage_execution_valid": 0,
            "inference_valid": 0,
            "failure_domain": domain,
            "failure_code": code,
            "message": str(error),
            **{name: 0 for name in _gate.PREFLIGHT_FLAGS},
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "initialization_failure.json", metrics)
    manifest = _semantic(
        {
            "schema": RUN_SCHEMA + "-initialization-failure-manifest",
            "schema_version": 1,
            "initialization_complete": 0,
            "verified_parent_hashes_claimed": 0,
            "verified_source_fingerprint_claimed": 0,
            "failure_domain": domain,
            "failure_code": code,
        }
    )
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "parent_provenance_failure.json", metrics)
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    gate = _gate.evaluate_preflight_gate(metrics)
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    _stage_seal(
        run_dir,
        "preflight",
        [
            "run_manifest.json",
            "parent_provenance_failure.json",
            "initialization_failure.json",
            "preflight_metrics.json",
            "preflight_gate.json",
        ],
    )
    _workflow_record(run_dir, "none")
    decision = _decision(run_dir)
    _status(
        run_dir,
        state="initialization_failed",
        stage="initialize",
        decision=str(decision.get("decision")),
        message=str(error),
        failure_domain=domain,
        failure_code=code,
    )
    _artifact_registry(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh frequency-one coordinate Jacobi/RB learnability workflow"
    )
    parser.add_argument("--stage", choices=STAGES, default="report")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-absolute-coordinate-run-dir", type=Path, required=True)
    parser.add_argument("--parent-memory-v3-run-dir", type=Path, required=True)
    parser.add_argument("--parent-coarse-witness-run-dir", type=Path, required=True)
    parser.add_argument("--parent-directional-result-archive", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability"
        ),
    )
    parser.add_argument("--run-name", default="production-frequency1-coordinate-v1-one-image")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-path-count", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--test-outer-steps", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--test-maximum-updates", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--test-bootstrap-replicates", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument(
        "--test-selection-outcome", choices=("none", "nominee"), default="none", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--test-confirmation-outcome", choices=("pass", "fail"), default="pass", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    for name in (
        "parent_absolute_coordinate_run_dir",
        "parent_memory_v3_run_dir",
        "parent_coarse_witness_run_dir",
        "parent_directional_result_archive",
        "resume_run_dir",
        "runs_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.stage not in {"preflight", "all"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    if not args.test_only and args.stage == "all":
        parser.error("production --stage all is forbidden")
    if args.test_only:
        if args.require_gate != "none":
            parser.error("test-only runs are nonauthorizing and require --require-gate none")
        if not 2 <= args.test_path_count <= 8:
            parser.error("test path count must lie in [2,8]")
        if args.test_outer_steps < 16 or args.test_outer_steps % 8:
            parser.error("test outer steps must be a multiple of eight and at least sixteen")
        if args.test_maximum_updates < 0:
            parser.error("test maximum updates must be nonnegative")
        if args.test_bootstrap_replicates < 2:
            parser.error("test bootstrap replicates must be at least two")
        if args.test_selection_outcome == "nominee" and args.test_maximum_updates < 1:
            parser.error("nominee fixture requires at least one nonzero update")
    elif args.stage != "report" and args.device != "cuda":
        parser.error("production physical stages require --device cuda")
    return args


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "cache", "controls", "train", "select", "confirm", "report")
    return (stage,)


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    print(f"frequency-one coordinate run directory: {run_dir}", flush=True)
    active_stage = "initialize"
    try:
        _initialize(run_dir, args, resumed=resumed)
        functions = {
            "preflight": _preflight_stage,
            "cache": _cache_stage,
            "controls": _controls_stage,
            "train": _train_stage,
            "select": _select_stage,
            "confirm": _confirm_stage,
        }
        for active_stage in _stage_sequence(args.stage):
            if active_stage == "report":
                _report_stage(run_dir, args)
                continue
            if active_stage != "report":
                runtime = configure_exact_torch_backend(args.device)
                runtime_path = run_dir / "exact_backend_runtime.json"
                if runtime_path.is_file() and _load_json(runtime_path) != runtime:
                    raise ArtifactCompatibilityError("exact backend runtime changed")
                if not runtime_path.is_file():
                    atomic_write_json(runtime_path, runtime)
            gate = functions[active_stage](run_dir, args)
            _workflow_record(run_dir, "none")
            if int(gate.get("valid_scientific_negative", 0)) == 1:
                break
            if not _passed(gate):
                break
            if active_stage == "select" and args.stage == "all":
                selection = _load_json(run_dir / "selection_decision.json")
                if selection["decision"] != "frequency1_coordinate_validation_nominee_sealed":
                    break
        if args.stage == "all" and not (run_dir / "REPORT.md").is_file():
            _report_stage(run_dir, args)
        decision = _decision(run_dir)
        workflow = _workflow_record(run_dir, args.require_gate)
        terminal = int(decision.get("scientific_evidence_complete", 0)) == 1
        _status(
            run_dir,
            state="complete" if terminal else "stage_complete",
            stage=args.stage,
            decision=str(decision.get("decision")),
            scientific_evidence_complete=int(terminal),
        )
        _artifact_registry(run_dir)
        print(f"frequency-one coordinate decision: {decision.get('decision')}", flush=True)
        if int(decision.get("valid_scientific_negative", 0)) == 1:
            return 0
        return int(workflow.get("required_gate_exit_code", 1))
    except (ArtifactCompatibilityError, _workflow.Frequency1CoordinateWorkflowError) as exc:
        if active_stage == "initialize" and not resumed and run_dir.is_dir():
            _commit_initialization_failure(run_dir, exc)
        elif active_stage != "initialize" and (run_dir / "scientific_config.json").is_file():
            _commit_execution_failure(run_dir, stage=active_stage, error=exc)
        print(f"frequency-one coordinate compatibility error: {exc}", flush=True)
        return 2
    except Exception as exc:
        if active_stage == "initialize" and not resumed and run_dir.is_dir():
            _commit_initialization_failure(run_dir, exc)
        elif active_stage != "initialize" and (run_dir / "scientific_config.json").is_file():
            _commit_execution_failure(run_dir, stage=active_stage, error=exc)
        print(f"frequency-one coordinate error: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
