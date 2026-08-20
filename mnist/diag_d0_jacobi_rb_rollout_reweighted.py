from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import atomic_write_csv, atomic_write_json, file_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import LabelOpenAuthorization, open_external_input_store, open_external_label_store
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_global_dilated import GLOBAL_DILATED_PARAMETER_COUNT, GlobalDilatedZeroBaselinePredictor, global_dilated_architecture_contract
from mnist.d0_jacobi_rb_learnability import MODEL_INPUT_FIELDS, ModelInputs, TrainingPlan, TrainingResumeSnapshot, discover_repository_path_id_claims, enable_deterministic_torch, evaluate_model_mse, scan_path_id_collisions, semantic_sha256, state_dict_sha256, train_deterministic_regressor
from mnist.d0_jacobi_rb_rollout_reweight import TRAIN_PATH_IDS, VALIDATION_PATH_IDS, _array_sha256 as _reweight_array_sha256, augment_mapping, build_rollout_reweighting, validate_cache_role_path_ids
from mnist.d0_jacobi_rb_tangent_fused import CANDIDATE_REFERENCE_CONTRACT, CandidateApproximateFusedReference, FusedRowSpec, FusedTangentControllerBank, prepare_deferred_reference_rng_seed_map, run_fused_reverse_family
from mnist.d0_jacobi_rb_tangent_rollout import TargetFractionOracleController, atomic_rollout_npz, fixed_rendering_scale, load_verified_source_target, raw_state_metrics, render_background_demixed, render_raw_density, render_source_image, reverse_suffix_sequence, rollout_array_sha256, rollout_file_sha256, save_png
from mnist.diag_d0_jacobi_rb_candidate_complete import CompletedStepCutoffTangentScoreController, _DIRECT_SOURCE_FILES as _CANDIDATE_DIRECT_SOURCE_FILES, _atomic_text, _contact_sheet, _copy_file as _copy, _directory_bytes, _npz, _prior_state, _read_json, _utc_now, _verify_manifest as _verify_stage_e_manifest
from mnist.diag_d0_jacobi_rb_global_dilated_rollout import _committed_numerical_path_ids


VERSION = "d0-jacobi-rb-rollout-reweighted-v1"
RUN_NAME = SPEC = "stage-e-rollout-reweighted-v1"
RESEARCH_MODE = "exploratory"
ROW_ORDER = ("zero", "frozen-cutoff-216", "reweighted-cutoff-216", "source-informed")
RENDER_HORIZONS = (0, 16, 128, 208, 216, 224, 256, 384, 512)
EVAL_PATH_IDS = (0xE9008, 0xE9009, 0xE900A)
EVAL_PRIOR_SEEDS = (261406, 261407, 261408)
TRAINING_SEED, TRANSITION_ROOT_SEED = 261405, 261409
STREAM_ROLE = "global_dilated_rollout_reweighted_eval_v1"
APPROVAL_REFERENCE = "rollout-reweighted-user-authorized-20260815"
ACTIVE_SECONDS_CAP = 7200.0
STORAGE_CAP_BYTES = 500 * 1024**2
CUDA_FRACTION_CAP = 0.80
TARGET_RMS = 2.610130414663935
SHARD_STEPS, SHARD_COUNT, MICROSTEPS = 8, 64, 2
MASS_TOLERANCE = 2.0e-12
PER_SHARD_TRANSITIONS = 351_232
BASELINE_FILE_SHA256 = "5831a950a979726bf7a648d4c276bdc13f032f17ad1bc739c5d73c25d4841d38"
BASELINE_STATE_SHA256 = "1df9888bef6c63db10f41f89a58891321e058e55ed7d8b36622c9cdf9827a218"
SOURCE_JSON_SHA256 = "e4f6918a6bd9b01f36ebdebdcf262242dfa714e908af199bde47cb9e025591eb"
SOURCE_NPZ_SHA256 = "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
SOURCE_ARRAY_SHA256 = "9d1f95ff3487901dabcd9b8261e0241511d2191997cd9daa260e390a3bc26c96"
TARGET_ARRAY_SHA256 = "14513cf5aa1aceda2bcff9befdd685297040ea33a3a773809e2b03d9401d5fc8"
STAGE_E_HASHES = {"config.json": "0c03de75d80c51ce16c441b58af4c40ef55e812ee2d706288ef8e20963c64079", "outcome.json": "e71f5784a94e51fd87cc569f83e369a27a4de66282133709d0e2ffd8e81bda84", "reverse/trajectory_boundaries.npz": "bf70dcde099b38777c5640d287e36b911210d26f1b561ddd6c5bc2e102072af7"}
CACHE_INDEX_AUTHORITIES = {"train": {"sha256": "04fed25023c54d4b3829c49a788f4cb3482c717fa5be5415bfd57f40c74954f2", "semantic_sha256": "6329e7d745b46260e9841603663f2deb8c1b847c95db1d4336294270928c07e3"}, "validation": {"sha256": "07020cf59479b26f475155c5b7cefdc9bfbf677f6ebff1f5f136ba7d9681d635", "semantic_sha256": "919858d8b491f3c370580f3f5c45f49b44004987f5bee55b50368726bc394850"}}
COPY_AUTHORITIES = {"baseline_checkpoint": ("inputs/baseline_checkpoint.pt", BASELINE_FILE_SHA256), "source_json": ("inputs/source/source_image.json", SOURCE_JSON_SHA256), "source_npz": ("inputs/source/source_image.npz", SOURCE_NPZ_SHA256), "stage_e_config.json": ("inputs/stage_e_development/config.json", STAGE_E_HASHES["config.json"]), "stage_e_outcome.json": ("inputs/stage_e_development/outcome.json", STAGE_E_HASHES["outcome.json"]), "stage_e_reverse_trajectory_boundaries.npz": ("inputs/stage_e_development/trajectory_boundaries.npz", STAGE_E_HASHES["reverse/trajectory_boundaries.npz"])}
PROTECTED_CONFIRMATION_PATH_IDS = tuple(range(0xF9000, 0xF9040))
HUMAN_LABELS = tuple(str(value) for value in range(10)) + ("ambiguous", "noise")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIRECT_SOURCE_FILES = tuple(dict.fromkeys((*_CANDIDATE_DIRECT_SOURCE_FILES, "mnist/diag_d0_jacobi_rb_rollout_reweighted.py", "mnist/d0_jacobi_rb_rollout_reweight.py", "mnist/d0_jacobi_rb_boundary_tangent_v3_memory.py", "mnist/d0_jacobi_rb_boundary_tangent_eager_cache.py", "mnist/diag_d0_jacobi_rb_global_dilated_rollout.py")))


class RolloutReweightedRunError(RuntimeError): pass


class ResourcePause(RolloutReweightedRunError): pass


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RolloutReweightedRunError(message)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor); temporary = Path(name)
    try:
        torch.save(value, temporary); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _paths_overlap(left: Path, right: Path) -> bool:
    left, right = left.resolve(), right.resolve(); return left == right or left in right.parents or right in left.parents


def _approval(value: str) -> str:
    approval = str(value).strip()
    _require(bool(approval) and not (approval.startswith("<") and approval.endswith(">")), "a real approval reference is required")
    _require(approval == APPROVAL_REFERENCE, "approval reference differs from the frozen authorization")
    return approval


def _validate_stage_e_capsule(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    _verify_stage_e_manifest(root)
    for relative, digest in STAGE_E_HASHES.items():
        path = root / relative
        _require(path.is_file() and not path.is_symlink() and file_fingerprint(path) == digest, f"Stage E authority changed: {relative}")
    config, outcome = _read_json(root / "config.json"), _read_json(root / "outcome.json")
    row_order = config.get("row_order")
    semantics = (config.get("experiment_name"), config.get("rng_binding", {}).get("canonical_path_id"), outcome.get("run_state"), outcome.get("health_passed"), outcome.get("schedule_identity_passed"), outcome.get("scientific_objective_completed"))
    _require(isinstance(row_order, list) and row_order.count("global-cutoff-216") == 1 and semantics == ("stage-e-prior-cutoff-216-v1", 1_028_865, "complete", 1, 1, 1), "Stage E capsule semantics changed")
    archive = _npz(root / "reverse/trajectory_boundaries.npz")
    steps, states = archive.get("completed_reverse_steps"), archive.get("states")
    valid = steps is not None and states is not None and (steps.dtype, states.dtype, steps.shape, states.shape) == (np.dtype(np.int64), np.dtype(np.float64), (65,), (65, 4, 784))
    valid = valid and set(archive) == {"completed_reverse_steps", "states"} and np.array_equal(steps, np.arange(0, 513, 8, dtype=np.int64)) and np.isfinite(states).all() and not np.any(states < 0.0) and float(np.max(np.abs(np.sum(states, axis=2) - 1.0))) <= MASS_TOLERANCE
    _require(valid, "Stage E trajectory contract changed")
    return {"config": config, "outcome": outcome, "steps": steps, "states": states, "selected_states": states[:, row_order.index("global-cutoff-216"), :]}


def _load_checkpoint(path: Path, device: torch.device | str = "cpu") -> torch.nn.Module:
    _require(file_fingerprint(path) == BASELINE_FILE_SHA256, "baseline checkpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    _require(isinstance(state, Mapping) and payload.get("state_sha256") == state_dict_sha256(state) == BASELINE_STATE_SHA256, "baseline checkpoint state changed")
    architecture = global_dilated_architecture_contract()
    _require((architecture.get("passed"), architecture.get("trainable_parameter_count")) == (1, GLOBAL_DILATED_PARAMETER_COUNT), "global architecture contract changed")
    model = GlobalDilatedZeroBaselinePredictor(zero_residual=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _frozen_config() -> dict[str, Any]:
    return {"schema": VERSION + "-config", "spec": SPEC, "research_mode": RESEARCH_MODE, "decision": "does target-preserving rollout-proximity reweighting materially improve fresh prior-start output", "training": {"seed": TRAINING_SEED, "updates": 4000, "batch_size": 32, "validation_interval": 100, "target_rms": TARGET_RMS, "parent_update": 3100, "parent_actual_suffix": "-v1-one-image", "selection": "minimum finite nonzero validation MSE, earlier update tie"}, "reweighting": {"eligible_midpoint_fraction": "15/16", "retain_every_original_once": 1, "threshold_source": "training medians by (outer_step,phase)", "active_outer_step_minimum": 296}, "evaluation": {"path_ids": list(EVAL_PATH_IDS), "prior_seeds": list(EVAL_PRIOR_SEEDS), "transition_root_seed": TRANSITION_ROOT_SEED, "stream_role": STREAM_ROLE, "row_order": list(ROW_ORDER), "render_horizons": list(RENDER_HORIZONS), "cutoff": 216, "backend": CANDIDATE_REFERENCE_CONTRACT}, "resource_caps": {"active_seconds": ACTIVE_SECONDS_CAP, "storage_bytes": STORAGE_CAP_BYTES, "cuda_fraction": CUDA_FRACTION_CAP}, "confirmation_evidence_opened": 0, "automatic_stage_f_launch": 0}


def _initialize_run(args: argparse.Namespace) -> Path:
    repository, training_parent, baseline, stage_e, runs_root = (Path(value).resolve() for value in (args.repository_root, args.training_parent, args.baseline_run_dir, args.stage_e_run_dir, args.runs_root))
    _require(args.run_name == RUN_NAME and _SAFE_NAME.fullmatch(args.run_name), "run name differs from the frozen name")
    _require(float(args.maximum_active_seconds) == ACTIVE_SECONDS_CAP, "initial active-time cap must be exactly 7200 seconds")
    approval = _approval(args.approval_reference)
    destination = runs_root / args.run_name
    if destination.exists(): return destination
    _require(all(path.is_dir() for path in (repository, training_parent, baseline, stage_e)), "a required input directory is absent")
    _require(not any(_paths_overlap(destination, path) for path in (training_parent, baseline, stage_e)), "output overlaps an input authority")
    collisions = scan_path_id_collisions(EVAL_PATH_IDS, discover_repository_path_id_claims(repository))
    _require(not collisions, f"frozen evaluation IDs collide with repository claims: {collisions}")
    _require(not set(EVAL_PATH_IDS).intersection(_committed_numerical_path_ids(repository)), "frozen evaluation IDs already occur in committed numerical artifacts")
    capsule = _validate_stage_e_capsule(stage_e)
    temporary = runs_root / f".{args.run_name}.initializing"
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        sources = {"baseline_checkpoint": baseline / "inputs/model/update-3100.pt", "source_json": baseline / "inputs/source/source_image.json", "source_npz": baseline / "inputs/source/source_image.npz", "stage_e_config.json": stage_e / "config.json", "stage_e_outcome.json": stage_e / "outcome.json", "stage_e_reverse_trajectory_boundaries.npz": stage_e / "reverse/trajectory_boundaries.npz"}
        copies = {name: {**_copy(sources[name], temporary / path, digest), "path": path} for name, (path, digest) in COPY_AUTHORITIES.items()}
        atomic_write_json(temporary / "config.json", _frozen_config())
        atomic_write_json(temporary / "bindings.json", {"schema": VERSION + "-bindings", "copies": copies, "external_locators": {"training_parent": str(training_parent), "baseline_run_dir": str(baseline), "stage_e_run_dir": str(stage_e)}, "stage_e_selected_row_sha256": rollout_array_sha256(capsule["selected_states"]), "source_files": {relative: file_fingerprint(Path(__file__).resolve().parents[1] / relative) for relative in _DIRECT_SOURCE_FILES}})
        atomic_write_json(temporary / "resource_ledger.json", {"schema": VERSION + "-resource-ledger", "maximum_active_seconds": ACTIVE_SECONDS_CAP, "active_seconds": 0.0, "storage_cap_bytes": STORAGE_CAP_BYTES, "cuda_fraction_cap": CUDA_FRACTION_CAP, "approval_reference": approval, "events": [], "latest_projection": None})
        atomic_write_json(temporary / "status.json", {"schema": VERSION + "-status", "state": "initialized", "resumable": 1, "completed_evaluation_paths": 0, "error": None})
        _atomic_text(temporary / "command.txt", subprocess.list2cmdline([sys.executable, "-B", "-m", "mnist.diag_d0_jacobi_rb_rollout_reweighted", *sys.argv[1:]]) + "\n")
        atomic_rollout_npz(temporary / "inputs/stage_e_development/selected_boundary_states.npz", {"completed_reverse_steps": capsule["steps"], "states": capsule["selected_states"]})
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _charge(run_dir: Path, role: str, elapsed: float, **detail: Any) -> dict[str, Any]:
    ledger = _read_json(run_dir / "resource_ledger.json")
    value = max(0.0, float(elapsed))
    ledger["active_seconds"] = math.fsum((float(ledger["active_seconds"]), value))
    ledger["events"].append({"at": _utc_now(), "role": role, "elapsed_seconds": value, **detail})
    atomic_write_json(run_dir / "resource_ledger.json", ledger)
    return ledger


@contextmanager
def _attempt(run_dir: Path, role: str, **detail: Any) -> Any:
    started = time.perf_counter()
    try:
        yield started
    except BaseException:
        _charge(run_dir, role, time.perf_counter() - started, failed=1, **detail)
        raise
    _charge(run_dir, role, time.perf_counter() - started, failed=0, **detail)


def _resource_check(run_dir: Path, device: torch.device | None = None, *, pending_seconds: float = 0.0) -> None:
    ledger = _read_json(run_dir / "resource_ledger.json")
    pending = max(0.0, float(pending_seconds))
    if float(ledger["active_seconds"]) + pending >= ACTIVE_SECONDS_CAP:
        raise ResourcePause("the approved active-time cap has been reached")
    if _directory_bytes(run_dir) >= STORAGE_CAP_BYTES:
        raise ResourcePause("the storage cap has been reached")
    if device is not None and device.type == "cuda":
        total = int(torch.cuda.get_device_properties(device).total_memory)
        peak = int(torch.cuda.max_memory_allocated(device))
        if total and peak / total >= CUDA_FRACTION_CAP:
            raise ResourcePause("CUDA allocation reached the 80% cap")


def _project(run_dir: Path, stage: str, value: float, **detail: Any) -> None:
    ledger = _read_json(run_dir / "resource_ledger.json")
    ledger["latest_projection"] = {"stage": stage, **detail, "projected_total_seconds": value, "passed": int(value <= ACTIVE_SECONDS_CAP)}
    atomic_write_json(run_dir / "resource_ledger.json", ledger)
    if value > ACTIVE_SECONDS_CAP: raise ResourcePause(f"{stage} projection exceeds the active-time cap")


def _verify_copied_inputs(run_dir: Path) -> None:
    bindings = _read_json(run_dir / "bindings.json")
    copies = bindings.get("copies")
    _require(isinstance(copies, Mapping) and set(copies) == set(COPY_AUTHORITIES), "copied input bindings changed")
    for name, record in copies.items():
        _require(isinstance(record, Mapping), "copied input binding changed")
        relative = Path(str(record.get("path", "")))
        expected_path, expected_sha256 = COPY_AUTHORITIES[name]
        candidate = run_dir / relative
        path = candidate.resolve()
        valid = relative.as_posix() == expected_path and not relative.is_absolute() and path.is_relative_to(run_dir.resolve()) and path.is_file() and not candidate.is_symlink() and path.stat().st_nlink == 1
        _require(valid and (path.stat().st_size, file_fingerprint(path), record.get("sha256")) == (record.get("size"), expected_sha256, expected_sha256), "copied input binding changed")
    selected_path = run_dir / "inputs/stage_e_development/selected_boundary_states.npz"
    archive = _npz(selected_path)
    steps, states = archive.get("completed_reverse_steps"), archive.get("states")
    valid = steps is not None and states is not None and set(archive) == {"completed_reverse_steps", "states"} and (steps.dtype, states.dtype, steps.shape, states.shape) == (np.dtype(np.int64), np.dtype(np.float64), (65,), (65, 784))
    valid = valid and not selected_path.is_symlink() and selected_path.stat().st_nlink == 1 and np.array_equal(steps, np.arange(0, 513, 8, dtype=np.int64)) and np.isfinite(states).all() and not np.any(states < 0.0) and float(np.max(np.abs(np.sum(states, axis=1) - 1.0))) <= MASS_TOLERANCE and rollout_array_sha256(states) == bindings.get("stage_e_selected_row_sha256")
    _require(valid, "copied Stage E selected trajectory changed")
    stage_config = _read_json(run_dir / "inputs/stage_e_development/config.json")
    stage_trajectory = _npz(run_dir / "inputs/stage_e_development/trajectory_boundaries.npz")
    row_order = stage_config.get("row_order")
    valid = isinstance(row_order, list) and row_order.count("global-cutoff-216") == 1 and set(stage_trajectory) == {"completed_reverse_steps", "states"} and np.array_equal(stage_trajectory["completed_reverse_steps"], steps) and stage_trajectory["states"].shape == (65, 4, 784)
    _require(valid and np.array_equal(stage_trajectory["states"][:, row_order.index("global-cutoff-216"), :], states), "copied Stage E selected-row authority changed")


def _validate_ledger(run_dir: Path, *, completed: bool = False) -> dict[str, Any]:
    ledger = _read_json(run_dir / "resource_ledger.json")
    events = ledger.get("events")
    active = ledger.get("active_seconds")
    contract = (ledger.get("schema"), ledger.get("maximum_active_seconds"), ledger.get("storage_cap_bytes"), ledger.get("cuda_fraction_cap"), ledger.get("approval_reference"))
    _require(contract == (VERSION + "-resource-ledger", ACTIVE_SECONDS_CAP, STORAGE_CAP_BYTES, CUDA_FRACTION_CAP, APPROVAL_REFERENCE) and isinstance(active, (int, float)) and not isinstance(active, bool) and math.isfinite(float(active)) and 0.0 <= float(active) <= ACTIVE_SECONDS_CAP and isinstance(events, list), "resource ledger contract changed")
    valid_event = lambda event: isinstance(event, Mapping) and isinstance(event.get("role"), str) and bool(event["role"]) and isinstance(event.get("at"), str) and isinstance(event.get("elapsed_seconds"), (int, float)) and not isinstance(event.get("elapsed_seconds"), bool) and math.isfinite(float(event["elapsed_seconds"])) and float(event["elapsed_seconds"]) >= 0.0 and event.get("failed", 0) in (0, 1)
    _require(all(valid_event(event) for event in events), "resource ledger event changed")
    elapsed = [float(event["elapsed_seconds"]) for event in events]
    projection = ledger.get("latest_projection")
    valid_projection = projection is None or isinstance(projection, Mapping) and isinstance(projection.get("projected_total_seconds"), (int, float)) and not isinstance(projection.get("projected_total_seconds"), bool) and math.isfinite(float(projection["projected_total_seconds"])) and projection.get("passed") in (0, 1)
    _require(valid_projection, "resource projection changed")
    event_sum = math.fsum(elapsed)
    storage = _directory_bytes(run_dir)
    _require(math.isclose(event_sum, float(active), rel_tol=1.0e-12, abs_tol=1.0e-9) and storage <= STORAGE_CAP_BYTES, "resource ledger totals exceeded or changed")
    if completed:
        required = {"training-data-preparation", "training-attempt", "training-diagnostics", "evaluation-setup", "evaluation-aggregation", "terminalization", "terminal-verification"} | {f"{prefix}-path-0x{path_id:X}" for prefix in ("evaluation", "derive") for path_id in EVAL_PATH_IDS}
        successful = {event["role"] for event in events if event.get("failed", 0) == 0}
        _require(required <= successful and isinstance(projection, Mapping) and projection.get("stage") == "evaluation" and projection.get("passed") == 1, "completed resource event history changed")
    return {"schema": VERSION + "-resource-health", "passed": 1, "active_seconds": float(active), "maximum_active_seconds": ACTIVE_SECONDS_CAP, "event_count": len(events), "event_seconds_sum": event_sum, "storage_bytes": storage, "storage_cap_bytes": STORAGE_CAP_BYTES}


def _cache_index_authorities(training_parent: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for role, authority in CACHE_INDEX_AUTHORITIES.items():
        path = training_parent / "eager_cache" / f"{role}_index.json"
        record = _read_json(path)
        _require((file_fingerprint(path), record.get("semantic_sha256"), record.get("role")) == (authority["sha256"], authority["semantic_sha256"], role), f"sealed {role} cache index changed")
        records[role] = record
    return records


def _role_inventory() -> dict[str, Any]:
    return {"schema": VERSION + "-role-inventory", "train_path_ids": list(TRAIN_PATH_IDS), "validation_path_ids": list(VALIDATION_PATH_IDS), "development_path_ids": [1_028_865], "evaluation_path_ids": list(EVAL_PATH_IDS), "protected_confirmation_path_ids_opened": [], "pairwise_role_separation_passed": 1, "train_index_semantic_sha256": CACHE_INDEX_AUTHORITIES["train"]["semantic_sha256"], "validation_index_semantic_sha256": CACHE_INDEX_AUTHORITIES["validation"]["semantic_sha256"], "train_index_file_sha256": CACHE_INDEX_AUTHORITIES["train"]["sha256"], "validation_index_file_sha256": CACHE_INDEX_AUTHORITIES["validation"]["sha256"]}


def select_nonzero_checkpoint(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = []
    for candidate in candidates:
        update = candidate.get("update")
        mse = candidate.get("validation_mse")
        if isinstance(update, int) and not isinstance(update, bool) and update > 0 and isinstance(mse, (int, float)) and not isinstance(mse, bool) and math.isfinite(float(mse)):
            eligible.append(dict(candidate))
    _require(eligible, "no finite nonzero checkpoint is eligible")
    return min(eligible, key=lambda item: (float(item["validation_mse"]), int(item["update"])))


def _input_arrays(store: Any) -> dict[str, np.ndarray]:
    return {name: np.asarray(store.row_array(name)) for name in MODEL_INPUT_FIELDS}


def _model_inputs(values: Mapping[str, np.ndarray], device: torch.device) -> ModelInputs:
    return ModelInputs(later_full_state=torch.as_tensor(values["later_full_state"], dtype=torch.float32, device=device), reverse_time=torch.as_tensor(values["reverse_time"], dtype=torch.float64, device=device), phase=torch.as_tensor(values["phase"], dtype=torch.long, device=device), color=torch.as_tensor(values["color"], dtype=torch.long, device=device), duration=torch.as_tensor(values["duration"], dtype=torch.float32, device=device), label=torch.as_tensor(values["label"], dtype=torch.long, device=device))


def _persist_reweighting(run_dir: Path, result: Any) -> None:
    root = run_dir / "reweighting"
    expected = {"train_duplicate_indices.npz": {"distances": result.train_distances, "duplicate_indices": result.train_duplicate_indices, "augmented_indices": result.train_augmented_indices}, "validation_duplicate_indices.npz": {"distances": result.validation_distances, "duplicate_indices": result.validation_duplicate_indices, "augmented_indices": result.validation_augmented_indices}, "thresholds.npz": {"outer_steps": result.threshold_outer_steps, "thresholds": result.thresholds}}
    for name, arrays in expected.items():
        path = root / name
        if path.exists():
            observed = _npz(path)
            _require(set(observed) == set(arrays) and all(np.array_equal(observed[key], value, equal_nan=True) for key, value in arrays.items()), f"persisted reweighting changed: {name}")
        else:
            atomic_rollout_npz(path, arrays)
    summary_path = root / "summary.json"
    _require(not summary_path.exists() or _read_json(summary_path) == dict(result.record), "persisted reweighting summary changed")
    if not summary_path.exists(): atomic_write_json(summary_path, dict(result.record))


def _verify_reweighting(run_dir: Path, summary: Mapping[str, Any]) -> None:
    root, hashes = run_dir / "reweighting", summary.get("hashes")
    files = {"train_duplicate_indices.npz": (("distances", "train_distances", (summary["train"]["original_row_count"],), np.float64), ("duplicate_indices", "train_duplicate_indices", (summary["train"]["duplicate_row_count"],), np.int64), ("augmented_indices", "train_augmented_indices", (summary["train"]["augmented_row_count"],), np.int64)), "validation_duplicate_indices.npz": (("distances", "validation_distances", (summary["validation"]["original_row_count"],), np.float64), ("duplicate_indices", "validation_duplicate_indices", (summary["validation"]["duplicate_row_count"],), np.int64), ("augmented_indices", "validation_augmented_indices", (summary["validation"]["augmented_row_count"],), np.int64)), "thresholds.npz": (("outer_steps", "threshold_outer_steps", (len(summary["eligibility"]["active_outer_steps"]),), np.int64), ("thresholds", "thresholds", (len(summary["eligibility"]["active_outer_steps"]), 7), np.float64))}
    _require(isinstance(hashes, Mapping) and {path.name for path in root.iterdir() if path.is_file()} == {*files, "summary.json"}, "reweighting artifact inventory changed")
    for filename, fields in files.items():
        archive = _npz(root / filename)
        _require(set(archive) == {name for name, *_ in fields} and all((archive[name].shape, archive[name].dtype) == (shape, np.dtype(dtype)) and _reweight_array_sha256(archive[name]) == hashes.get(hash_name + "_sha256") for name, hash_name, shape, dtype in fields), f"reweighting artifact changed: {filename}")


def _label_authorization(training_parent: Path, role: str) -> LabelOpenAuthorization:
    filename, purpose = ("physical_train_label_open.json", "physical_training") if role == "train" else ("validation_label_open.json", "validation_selection")
    opening = _read_json(training_parent / filename)
    seal = str(opening.get("semantic_sha256", ""))
    _require(opening.get("role") == role and len(seal) == 64, f"{role} label opening authority changed")
    return LabelOpenAuthorization(training_parent, role, purpose, seal)


def _prepare_training_data(run_dir: Path, training_parent: Path, device: torch.device) -> tuple[ModelInputs, torch.Tensor, ModelInputs, torch.Tensor, Any]:
    _verify_copied_inputs(run_dir)
    cache_indexes = _cache_index_authorities(training_parent)
    train_store = open_external_input_store(training_parent, "train")
    validation_store = open_external_input_store(training_parent, "validation")
    _require(dict(train_store.index) == cache_indexes["train"] and dict(validation_store.index) == cache_indexes["validation"], "opened cache index differs from its sealed authority")
    train_ids = validate_cache_role_path_ids("train", tuple(train_store.index.get("path_ids", ())))
    validation_ids = validate_cache_role_path_ids("validation", tuple(validation_store.index.get("path_ids", ())))
    _require(not set(train_ids).intersection(validation_ids) and not set(train_ids + validation_ids).intersection(PROTECTED_CONFIRMATION_PATH_IDS), "cache evidence roles overlap")
    train_arrays, validation_arrays = _input_arrays(train_store), _input_arrays(validation_store)
    stage_e = _npz(run_dir / "inputs/stage_e_development/selected_boundary_states.npz")
    _require(set(stage_e) == {"completed_reverse_steps", "states"} and stage_e["states"].shape == (65, 784), "copied Stage E selected trajectory changed")
    result = build_rollout_reweighting(train_arrays, validation_arrays, stage_e["states"])
    _persist_reweighting(run_dir, result)
    inventory = _role_inventory()
    path = run_dir / "inputs/role_inventory.json"
    _require(not path.exists() or _read_json(path) == inventory, "role inventory changed")
    atomic_write_json(path, inventory)
    train_labels = open_external_label_store(training_parent, "train", authorization=_label_authorization(training_parent, "train"))
    validation_labels = open_external_label_store(training_parent, "validation", authorization=_label_authorization(training_parent, "validation"))
    _require((train_labels.row_count, validation_labels.row_count) == (train_store.row_count, validation_store.row_count), "input/label row counts differ")
    train_augmented = augment_mapping(train_arrays, result.train_augmented_indices)
    validation_augmented = augment_mapping(validation_arrays, result.validation_augmented_indices)
    train_target_np = np.ascontiguousarray(train_labels.row_array("denoising_target")[result.train_augmented_indices])
    validation_target_np = np.ascontiguousarray(validation_labels.row_array("denoising_target")[result.validation_augmented_indices])
    _require(np.array_equal(train_target_np[train_store.row_count:], train_labels.row_array("denoising_target")[result.train_duplicate_indices]), "duplicated train targets changed")
    return _model_inputs(train_augmented, device), torch.as_tensor(train_target_np, dtype=torch.float64, device=device), _model_inputs(validation_augmented, device), torch.as_tensor(validation_target_np, dtype=torch.float64, device=device), result


def _validated_model_state(state: Any, expected_sha256: Any) -> dict[str, torch.Tensor]:
    _require(isinstance(state, Mapping) and all(isinstance(value, torch.Tensor) for value in state.values()), "candidate model state changed")
    copied = dict(state)
    _require(state_dict_sha256(copied) == expected_sha256 and all(not (value.is_floating_point() or value.is_complex()) or bool(torch.isfinite(value).all()) for value in copied.values()), "candidate model state changed")
    try:
        GlobalDilatedZeroBaselinePredictor(zero_residual=False).load_state_dict(copied, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise RolloutReweightedRunError("candidate architecture state changed") from exc
    return copied


def _validation_mse(history: Sequence[Mapping[str, Any]], update: int) -> float:
    rows = [row for row in history if int(row.get("update", -1)) == update and "validation_mse" in row]
    if len(rows) != 1:
        raise RolloutReweightedRunError("training validation history changed")
    value = rows[0]["validation_mse"]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise RolloutReweightedRunError("training validation MSE changed")
    return float(value)


def _history_argmin(history: Sequence[Mapping[str, Any]], maximum_update: int = 4000) -> tuple[float, int]:
    candidates = []
    for row in history:
        update, mse = row.get("update"), row.get("validation_mse")
        if isinstance(update, int) and not isinstance(update, bool) and 0 < update <= maximum_update and update % 100 == 0 and isinstance(mse, (int, float)) and not isinstance(mse, bool) and math.isfinite(float(mse)):
            candidates.append((float(mse), update))
    if not candidates:
        raise RolloutReweightedRunError("training history has no finite nonzero candidate")
    return min(candidates)


def _validated_best_nonzero(payload: Any, history: Sequence[Mapping[str, Any]], maximum_update: int) -> dict[str, Any]:
    expected_keys = {"schema", "update", "validation_mse", "state_sha256", "state_dict"}
    _require(isinstance(payload, Mapping) and set(payload) == expected_keys and payload.get("schema") == VERSION + "-best-nonzero", "best nonzero checkpoint changed")
    update, mse = payload.get("update"), payload.get("validation_mse")
    valid = isinstance(update, int) and not isinstance(update, bool) and 100 <= update <= min(maximum_update, 4000) and update % 100 == 0
    valid = valid and isinstance(mse, (int, float)) and not isinstance(mse, bool) and math.isfinite(float(mse)) and float(mse) == _validation_mse(history, update)
    _require(valid, "best nonzero checkpoint metadata changed")
    state = _validated_model_state(payload.get("state_dict"), payload.get("state_sha256"))
    return {**dict(payload), "state_dict": state}


def _resume_snapshot(path: Path) -> TrainingResumeSnapshot:
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    valid = isinstance(snapshot, TrainingResumeSnapshot) and snapshot.seed == TRAINING_SEED and 0 <= snapshot.completed_update <= 4000 and snapshot.torch_rng_state.device.type == "cpu" and all(value.device.type == "cpu" for value in snapshot.cuda_rng_states)
    _require(valid, "training resume snapshot changed")
    return snapshot


def _candidate_from_resume(snapshot: TrainingResumeSnapshot) -> dict[str, Any] | None:
    update = int(snapshot.completed_update)
    if update == 0:
        return None
    if not snapshot.finite or update < 100 or update > 4000 or update % 100:
        raise RolloutReweightedRunError("training resume cursor changed")
    mse = _validation_mse(snapshot.history, update)
    state = _validated_model_state(snapshot.model_state_dict, state_dict_sha256(snapshot.model_state_dict))
    return {"schema": VERSION + "-best-nonzero", "update": update, "validation_mse": mse, "state_sha256": state_dict_sha256(state), "state_dict": state}


def _reconcile_best_from_resume(snapshot: TrainingResumeSnapshot, best_path: Path) -> None:
    choices: list[dict[str, Any]] = []
    candidate = _candidate_from_resume(snapshot)
    if candidate is not None:
        choices.append(candidate)
    if best_path.exists():
        choices.append(_validated_best_nonzero(torch.load(best_path, map_location="cpu", weights_only=True), snapshot.history, snapshot.completed_update))
    if not choices:
        return
    winner = select_nonzero_checkpoint(choices)
    current = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
    if not isinstance(current, Mapping) or current.get("state_sha256") != winner["state_sha256"] or current.get("update") != winner["update"]:
        _atomic_torch(best_path, winner)


def _history_csv(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = []
            for raw in csv.DictReader(handle):
                row = {key: int(value) if key == "update" else float(value) for key, value in raw.items() if value not in (None, "")}
                rows.append(row)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RolloutReweightedRunError("training history changed") from exc
    return tuple(rows)


def _selected_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = {"schema", "fine_tune_update", "validation_mse", "state_sha256", "state_dict", "parent_file_sha256", "parent_state_sha256", "reweighting_semantic_sha256"}
    update = payload.get("fine_tune_update") if isinstance(payload, Mapping) else None
    mse = payload.get("validation_mse") if isinstance(payload, Mapping) else None
    valid = isinstance(payload, Mapping) and set(payload) == expected_keys and payload.get("schema") == VERSION + "-selected-checkpoint" and isinstance(update, int) and not isinstance(update, bool) and 100 <= update <= 4000 and update % 100 == 0
    valid = valid and isinstance(mse, (int, float)) and not isinstance(mse, bool) and math.isfinite(float(mse)) and (payload.get("parent_file_sha256"), payload.get("parent_state_sha256")) == (BASELINE_FILE_SHA256, BASELINE_STATE_SHA256)
    _require(valid, "selected candidate checkpoint changed")
    state = _validated_model_state(payload.get("state_dict"), payload.get("state_sha256"))
    run_dir = path.parent.parent
    summary = _read_json(run_dir / "reweighting/summary.json")
    body = {key: value for key, value in summary.items() if key != "semantic_sha256"}
    _require(summary.get("semantic_sha256") == semantic_sha256(body) == payload.get("reweighting_semantic_sha256"), "selected checkpoint reweighting binding changed")
    _verify_reweighting(run_dir, summary)
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    _require(_read_json(path.parent / "selection.json") == metadata, "selected checkpoint metadata commit changed")
    resume = _resume_snapshot(path.parent / "resume.pt")
    history = _history_csv(path.parent / "history.csv")
    _require(resume.completed_update == 4000 and resume.finite and history == tuple(dict(row) for row in resume.history) and _validation_mse(history, update) == float(mse), "completed training history changed")
    best = _validated_best_nonzero(torch.load(path.parent / "best_nonzero.pt", map_location="cpu", weights_only=True), resume.history, resume.completed_update)
    _require(_history_argmin(history) == (float(mse), update) == (float(best["validation_mse"]), best["update"]) and best["state_sha256"] == payload.get("state_sha256"), "selected checkpoint differs from the frozen best")
    diagnostics = _read_json(path.parent / "diagnostics.json")
    splits = diagnostics.get("splits")
    header = (diagnostics.get("schema"), diagnostics.get("selected_update"), diagnostics.get("selected_state_sha256"))
    _require(header == (VERSION + "-training-diagnostics", update, payload.get("state_sha256")) and isinstance(splits, Mapping) and set(splits) == {"train", "validation"}, "training diagnostics changed")
    for split, summary_key in (("train", "train"), ("validation", "validation")):
        row = splits[split]
        names = ("parent_weighted_mse", "parent_original_unweighted_mse", "candidate_weighted_mse", "candidate_original_unweighted_mse")
        counts = (row.get("original_row_count"), row.get("weighted_row_count")) if isinstance(row, Mapping) else ()
        finite = isinstance(row, Mapping) and all(isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool) and math.isfinite(float(row[name])) and float(row[name]) >= 0.0 for name in names)
        _require(counts == (summary[summary_key].get("original_row_count"), summary[summary_key].get("augmented_row_count")) and finite, "training diagnostics changed")
    return {**dict(payload), "state_dict": state}


def _candidate_model(run_dir: Path, device: torch.device | str) -> torch.nn.Module:
    payload = _selected_payload(run_dir / "training/selected_checkpoint.pt")
    model = GlobalDilatedZeroBaselinePredictor(zero_residual=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def _training_diagnostics(run_dir: Path, winner: Mapping[str, Any], train_inputs: ModelInputs, train_target: torch.Tensor, validation_inputs: ModelInputs, validation_target: torch.Tensor, reweighting: Any, device: torch.device) -> dict[str, Any]:
    parent = _load_checkpoint(run_dir / "inputs/baseline_checkpoint.pt", device).requires_grad_(False)
    candidate = GlobalDilatedZeroBaselinePredictor(zero_residual=False).to(device)
    candidate.load_state_dict(winner["state_dict"], strict=True)
    candidate.eval().requires_grad_(False)
    split_values = {"train": (train_inputs, train_target, int(reweighting.record["train"]["original_row_count"])), "validation": (validation_inputs, validation_target, int(reweighting.record["validation"]["original_row_count"]))}
    rows: dict[str, Any] = {}
    with torch.no_grad():
        for split, (inputs, target, original_count) in split_values.items():
            indices = torch.arange(original_count, dtype=torch.long, device=device)
            original_inputs, original_target = inputs.index_select(indices), target.index_select(0, indices)
            rows[split] = {"original_row_count": original_count, "weighted_row_count": inputs.batch_size}
            for name, model in (("parent", parent), ("candidate", candidate)):
                weighted, _ = evaluate_model_mse(model, inputs, target, batch_size=32)
                unweighted, _ = evaluate_model_mse(model, original_inputs, original_target, batch_size=32)
                rows[split][name + "_weighted_mse"] = float(weighted)
                rows[split][name + "_original_unweighted_mse"] = float(unweighted)
    return {"schema": VERSION + "-training-diagnostics", "selected_update": int(winner["update"]), "selected_state_sha256": winner["state_sha256"], "splits": rows}


def _train(run_dir: Path, training_parent: Path, device: torch.device) -> dict[str, Any]:
    selected_path = run_dir / "training/selected_checkpoint.pt"
    if selected_path.exists():
        return _selected_payload(selected_path)
    with _attempt(run_dir, "training-data-preparation"):
        train_inputs, train_target, validation_inputs, validation_target, reweighting = _prepare_training_data(run_dir, training_parent, device)
    _resource_check(run_dir, device)
    resume_path = run_dir / "training/resume.pt"
    resume = _resume_snapshot(resume_path) if resume_path.exists() else None
    best_path = run_dir / "training/best_nonzero.pt"
    if resume is not None: _reconcile_best_from_resume(resume, best_path)
    _require(resume is not None or not best_path.exists(), "best checkpoint exists without a resume snapshot")
    base_update = int(resume.completed_update) if resume is not None else 0

    def checkpoint(snapshot: TrainingResumeSnapshot) -> None:
        _atomic_torch(resume_path, snapshot)
        candidate = _candidate_from_resume(snapshot)
        if candidate is not None:
            choices = [candidate]
            if best_path.exists():
                choices.append(_validated_best_nonzero(torch.load(best_path, map_location="cpu", weights_only=True), snapshot.history, snapshot.completed_update))
            winner = select_nonzero_checkpoint(choices)
            current = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
            if not isinstance(current, Mapping) or current.get("update") != winner["update"] or current.get("state_sha256") != winner["state_sha256"]:
                _atomic_torch(best_path, winner)
        pending = max(time.perf_counter() - started, 0.0)
        progress = int(snapshot.completed_update) - base_update
        if progress > 0:
            ledger = _read_json(run_dir / "resource_ledger.json")
            remaining = max(0, 4000 - int(snapshot.completed_update))
            projection = float(ledger["active_seconds"]) + pending + pending / progress * remaining + 3_000.0 + 300.0
            _project(run_dir, "training", projection, completed_update=int(snapshot.completed_update))
        _resource_check(run_dir, device, pending_seconds=pending)

    with _attempt(run_dir, "training-attempt", base_update=base_update) as started:
        result = train_deterministic_regressor(lambda: _load_checkpoint(run_dir / "inputs/baseline_checkpoint.pt", "cpu"), train_inputs, train_target, validation_inputs, validation_target, target_scale=TARGET_RMS, seed=TRAINING_SEED, plan=TrainingPlan(), resume_snapshot=resume, checkpoint_callback=checkpoint)
    _resource_check(run_dir, device)
    _require(result.finite and best_path.exists(), "fine-tuning produced no finite nonzero checkpoint")
    best = _validated_best_nonzero(torch.load(best_path, map_location="cpu", weights_only=True), result.history, 4000)
    winner = select_nonzero_checkpoint([best])
    _require(_history_argmin(result.history) == (float(winner["validation_mse"]), int(winner["update"])), "persisted best is not the validation argmin")
    payload = {"schema": VERSION + "-selected-checkpoint", "fine_tune_update": int(winner["update"]), "validation_mse": float(winner["validation_mse"]), "state_sha256": winner["state_sha256"], "state_dict": winner["state_dict"], "parent_file_sha256": BASELINE_FILE_SHA256, "parent_state_sha256": BASELINE_STATE_SHA256, "reweighting_semantic_sha256": reweighting.record["semantic_sha256"]}
    with _attempt(run_dir, "training-diagnostics"):
        diagnostics = _training_diagnostics(run_dir, winner, train_inputs, train_target, validation_inputs, validation_target, reweighting, device)
    _resource_check(run_dir, device)
    atomic_write_json(run_dir / "training/diagnostics.json", diagnostics)
    atomic_write_csv(run_dir / "training/history.csv", result.history)
    atomic_write_json(run_dir / "training/selection.json", {key: value for key, value in payload.items() if key != "state_dict"})
    _atomic_torch(selected_path, payload)
    return _selected_payload(selected_path)


def _row_specs(path_id: int, candidate_sha256: str) -> tuple[FusedRowSpec, ...]:
    common = {"schedule_kind": "completed_reverse_step_prefix", "cutoff_completed_reverse_steps": 216, "active_predicate": "reverse_time_lt_cutoff_over_512", "active_outer_step_min_inclusive": 296, "inactive_outer_step_max_inclusive": 295}
    return (
        FusedRowSpec("zero", path_id, "zero", "zero", "complete-512"),
        FusedRowSpec("frozen-cutoff-216", path_id, "learned", "global-dilated-cutoff-216", "complete-512", gain=1.0, controller_binding={**common, "checkpoint_state_sha256": BASELINE_STATE_SHA256}),
        FusedRowSpec("reweighted-cutoff-216", path_id, "learned", "global-dilated-rollout-reweighted-cutoff-216", "complete-512", gain=1.0, controller_binding={**common, "checkpoint_state_sha256": candidate_sha256}),
        FusedRowSpec("source-informed", path_id, "oracle", "mixed-target-fraction", "complete-512", controller_binding={"target_sha256": TARGET_ARRAY_SHA256}),
    )


def _controller_binding(specs: Sequence[FusedRowSpec], candidate_sha256: str) -> dict[str, Any]:
    return {"row_table": [row.to_record() for row in specs], "baseline_state_sha256": BASELINE_STATE_SHA256, "candidate_state_sha256": candidate_sha256, "target_sha256": TARGET_ARRAY_SHA256, "dispatch": "stable_one_row_canonical_order", "model_input_contract": "exact_ModelInputs_six_fields"}


def _rng_binding(path_id: int) -> dict[str, Any]:
    return {"root_seed": TRANSITION_ROOT_SEED, "stream_role": STREAM_ROLE, "canonical_path_id": int(path_id), "same_random_bits_across_rows": 1}


def _eval_root(run_dir: Path, path_id: int) -> Path:
    return run_dir / "evaluation" / f"path-0x{path_id:X}"


def _recover_eval_orphan(run_dir: Path, path_id: int) -> Path | None:
    root = _eval_root(run_dir, path_id) / "reverse/fused_families/same-path-four-row/complete-512"
    for index in range(SHARD_COUNT):
        record_path, state_path = root / f"shard-{index:04d}.json", root / f"shard-{index:04d}.npz"
        if record_path.exists(): continue
        if not state_path.exists(): return None
        digest = file_fingerprint(state_path)
        destination = _eval_root(run_dir, path_id) / "recovery" / f"orphan-shard-{index:04d}-{digest}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require(not destination.exists() or file_fingerprint(destination) == digest, "evaluation recovery archive collision")
        os.replace(state_path, destination)
        _require(file_fingerprint(destination) == digest, "evaluation orphan recovery changed bytes")
        return destination
    return None


def _scan_eval(run_dir: Path, path_id: int) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    root = _eval_root(run_dir, path_id) / "reverse/fused_families/same-path-four-row/complete-512"
    candidate_sha256 = _selected_payload(run_dir / "training/selected_checkpoint.pt")["state_sha256"]
    specs = _row_specs(path_id, candidate_sha256)
    sequence = tuple(reverse_suffix_sequence(511))
    controller_hash = semantic_sha256(_controller_binding(specs, candidate_sha256))
    rng_hash = semantic_sha256(_rng_binding(path_id))
    starts = _npz(run_dir / "inputs/evaluation_start_states.npz")
    path_ids = starts.get("path_ids")
    states0 = starts.get("states")
    _require(path_ids is not None and states0 is not None and path_id in path_ids.tolist(), "evaluation start-state authority changed")
    state = np.repeat(states0[path_ids.tolist().index(path_id)][None, :], 4, axis=0)
    records: list[dict[str, Any]] = []
    states = [state]
    for index in range(SHARD_COUNT):
        record_path, state_path = root / f"shard-{index:04d}.json", root / f"shard-{index:04d}.npz"
        if not record_path.exists():
            _require(not state_path.exists(), "uncommitted evaluation shard archive is present")
            break
        _require(state_path.is_file(), "committed evaluation shard lacks its state")
        record = _read_json(record_path)
        archive = _npz(state_path)
        next_state = archive.get("state")
        shard_sequence = sequence[index * 56:(index + 1) * 56]
        _require(set(archive) == {"state"} and next_state is not None and (next_state.shape, next_state.dtype) == ((4, 784), np.dtype(np.float64)), "evaluation shard state changed")
        expected = {"shard_index": index, "committed": 1, "input_state_sha256": rollout_array_sha256(state), "output_state_sha256": rollout_array_sha256(next_state), "state_file_sha256": rollout_file_sha256(state_path), "transition_count": PER_SHARD_TRANSITIONS, "family_name": "same-path-four-row", "segment_name": "complete-512", "row_keys": list(ROW_ORDER), "row_table": [row.to_record() for row in specs], "canonical_path_ids": [path_id] * 4, "controller_binding_sha256": controller_hash, "rng_binding_sha256": rng_hash, "sequence_start": list(shard_sequence[0]), "sequence_end": list(shard_sequence[-1]), "sequence_sha256": semantic_sha256([list(item) for item in shard_sequence]), "microsteps": MICROSTEPS, "label": 3, "variant_in_rng_key": 0, "reference_contract": CANDIDATE_REFERENCE_CONTRACT}
        _require(all(record.get(key) == value for key, value in expected.items()), "evaluation shard chain changed")
        records.append(record)
        states.append(next_state)
        state = next_state
    return records, states


def _cutoff_identity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for shard_index in range(27, len(records)):
        record = records[shard_index]
        for row_index, row_key in ((1, ROW_ORDER[1]), (2, ROW_ORDER[2])):
            controller = record.get("controller_diagnostics", [])[row_index]
            dynamics = record.get("per_row_diagnostics", [])[row_index]
            invalid = {}
            for name, source in (("score_squared_sum", controller), ("score_maximum_absolute", controller), ("unscaled_score_squared_sum", controller), ("unscaled_score_maximum_absolute", controller), ("logistic_shift_squared_sum", dynamics), ("logistic_shift_maximum_absolute", dynamics)):
                value = source.get(name)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) != 0.0:
                    invalid[name] = value
            if invalid: mismatches.append({"shard_index": shard_index, "row_key": row_key, "invalid_or_nonzero": invalid})
    return {"schema": VERSION + "-cutoff-identity", "cutoff_completed_reverse_steps": 216, "post_cutoff_first_shard_index": 27, "checked_shards": list(range(27, len(records))), "mismatches": mismatches, "passed": int(not mismatches)}


def _path_health(records: Sequence[Mapping[str, Any]], states: Sequence[np.ndarray]) -> dict[str, Any]:
    identity = _cutoff_identity(records)
    references = [record.get("diagnostics", {}).get("reference", {}) for record in records]
    forbidden_names = ("resource_cap_count", "invalid_density_count", "clipping_count", "correction_count", "floor_count", "limiter_count", "projection_count", "renormalization_count", "nonfinite_count")
    forbidden = {name: sum(int(record.get("diagnostics", {}).get("forbidden_counts", {}).get(name, 0)) for record in records) for name in forbidden_names}
    controller_forbidden = {name: sum(int(row.get(name, 0)) for record in records for row in record.get("controller_diagnostics", ())) for name in ("clipping_count", "floor_count", "projection_count", "nonfinite_score_count")}
    dynamics_invalid = {name: sum(int(row.get(name, 0)) for record in records for row in record.get("per_row_diagnostics", ())) for name in ("input_invalid", "state_invalid", "mass_invalid", "metadata_invalid", "score_invalid", "logistic_shift_invalid", "reference_fraction_invalid", "reference_invalid_count")}
    reported_mass_error = max((float(record.get("diagnostics", {}).get("maximum_mass_error", math.inf)) for record in records), default=0.0)
    invalid = sum(int(record.get("diagnostics", {}).get("invalid_count", 0)) + int(record.get("diagnostics", {}).get("fallback_count", 0)) for record in records)
    transition_count = sum(int(record.get("transition_count", 0)) for record in records)
    nonfinite = int(sum(np.count_nonzero(~np.isfinite(state)) for state in states))
    negative = int(sum(np.count_nonzero(state < 0.0) for state in states))
    boundary_mass_error = math.inf if nonfinite else max((float(np.max(np.abs(np.sum(state, axis=1) - 1.0))) for state in states), default=0.0)
    maximum_mass_error = max(reported_mass_error, boundary_mass_error)
    rng_valid = all(reference.get("root_seed") == TRANSITION_ROOT_SEED and reference.get("stream_role") == STREAM_ROLE and reference.get("variant_in_rng_key") == 0 for reference in references)
    per_row_transitions_valid = all(all(row.get("transition_count") == 87_808 and row.get("reference_transition_count") == 87_808 for row in record.get("per_row_diagnostics", ())) for record in records)
    peak = max((int(reference.get("peak_cuda_memory_bytes", 0)) for reference in references), default=0)
    total = max((int(reference.get("total_cuda_memory_bytes", 0)) for reference in references), default=0)
    passed = len(records) == SHARD_COUNT and transition_count == PER_SHARD_TRANSITIONS * SHARD_COUNT and maximum_mass_error <= MASS_TOLERANCE and invalid == 0 and not any(forbidden.values()) and not any(controller_forbidden.values()) and not any(dynamics_invalid.values()) and not nonfinite and not negative and identity["passed"] == 1 and rng_valid and per_row_transitions_valid and (not total or peak / total < CUDA_FRACTION_CAP)
    return {"schema": VERSION + "-path-health", "passed": int(passed), "committed_shards": len(records), "completed_reverse_steps": len(records) * SHARD_STEPS, "transition_count": transition_count, "reported_maximum_mass_error": reported_mass_error, "boundary_maximum_mass_error": boundary_mass_error, "maximum_mass_error": maximum_mass_error, "invalid_or_fallback_count": invalid, "forbidden_counts": forbidden, "controller_forbidden_counts": controller_forbidden, "dynamics_invalid_counts": dynamics_invalid, "rng_identity_passed": int(rng_valid), "per_row_transition_counts_passed": int(per_row_transitions_valid), "state_nonfinite_count": nonfinite, "state_negative_count": negative, "maximum_cuda_memory_fraction": peak / total if total else 0.0, "cutoff_identity": identity, "reference_contract": CANDIDATE_REFERENCE_CONTRACT}


def _metric_rows(path_id: int, states: Sequence[np.ndarray], source: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        completed = index * SHARD_STEPS
        zero = raw_state_metrics(state[0], source.mixed_target).to_dict()
        frozen = raw_state_metrics(state[1], source.mixed_target).to_dict()
        for row_index, row_key in enumerate(ROW_ORDER):
            target = raw_state_metrics(state[row_index], source.mixed_target).to_dict()
            source_metric = raw_state_metrics(state[row_index], source.source_image).to_dict()
            error = float(target["squared_l2_error"])
            zero_error = float(zero["squared_l2_error"])
            frozen_error = float(frozen["squared_l2_error"])
            delta = state[row_index] - state[0]
            frozen_delta = state[row_index] - state[1]
            rows.append({"path_id": path_id, "completed_reverse_steps": completed, "row_key": row_key, "target_squared_l2_error": error, "target_centered_correlation": float(target["centered_contrast_correlation"]), "source_squared_l2_error": float(source_metric["squared_l2_error"]), "paired_improvement_over_zero": zero_error - error, "relative_improvement_over_zero": (zero_error - error) / zero_error if zero_error else 0.0, "paired_improvement_over_frozen": frozen_error - error, "relative_improvement_over_frozen": (frozen_error - error) / frozen_error if frozen_error else 0.0, "state_vs_zero_squared_l2": float(np.dot(delta, delta)), "state_vs_frozen_squared_l2": float(np.dot(frozen_delta, frozen_delta)), "simplex_mass_error": abs(float(np.sum(state[row_index])) - 1.0), "state_nonfinite_count": int(np.count_nonzero(~np.isfinite(state[row_index]))), "state_negative_count": int(np.count_nonzero(state[row_index] < 0.0))})
    return rows


def _path_summary(path_id: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for horizon in (208, 216, 512):
        selected = [row for row in rows if int(row["completed_reverse_steps"]) == horizon]
        fields = {"squared_l2_error": "target_squared_l2_error", "centered_correlation": "target_centered_correlation", "paired_improvement_over_zero": "paired_improvement_over_zero", "relative_improvement_over_zero": "relative_improvement_over_zero", "paired_improvement_over_frozen": "paired_improvement_over_frozen", "relative_improvement_over_frozen": "relative_improvement_over_frozen", "state_vs_zero_squared_l2": "state_vs_zero_squared_l2", "state_vs_frozen_squared_l2": "state_vs_frozen_squared_l2"}
        horizons[str(horizon)] = {str(row["row_key"]): {name: float(row[source]) for name, source in fields.items()} for row in selected}
    return {"path_id": path_id, "horizons": horizons}


def _mechanism(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_row = []
    for row_index, row_key in enumerate(ROW_ORDER):
        values = [record["per_row_diagnostics"][row_index] for record in records]
        controller = [record["controller_diagnostics"][row_index] for record in records]
        item: dict[str, Any] = {"row_key": row_key}
        for prefix, rows in (("score", controller), ("logistic_shift", values), ("reference_fraction_displacement", values), ("control_fraction_displacement", values)):
            count = sum(int(row.get(prefix + "_count", 0)) for row in rows)
            squared = math.fsum(float(row.get(prefix + "_squared_sum", 0.0)) for row in rows)
            item[prefix + "_rms"] = math.sqrt(squared / count) if count else 0.0
            item[prefix + "_maximum_absolute"] = max((float(row.get(prefix + "_maximum_absolute", 0.0)) for row in rows), default=0.0)
        reference_rms = float(item["reference_fraction_displacement_rms"])
        control_rms = float(item["control_fraction_displacement_rms"])
        item["control_to_reference_rms_ratio"] = control_rms / reference_rms if reference_rms > 0.0 else (0.0 if control_rms == 0.0 else math.inf)
        per_row.append(item)
    return {"schema": VERSION + "-mechanism", "per_row": per_row}


def _render_path(root: Path, states: Sequence[np.ndarray], source: Any) -> None:
    scale = fixed_rendering_scale(source.source_image, source.mixed_target, float(source.metadata["lambda_mix"]))
    for completed in RENDER_HORIZONS:
        state = states[completed // SHARD_STEPS]
        for kind, renderer in (("raw", render_raw_density), ("demixed", render_background_demixed)):
            cells = []
            for row_index, row_key in enumerate(ROW_ORDER):
                image = renderer(state[row_index], scale)
                cells.append(image)
                save_png(root / f"images/individual/{kind}/{row_key}/step-{completed:03d}.png", image)
            _contact_sheet(root / f"images/contact_sheets/{kind}-step-{completed:03d}.png", cells)
    save_png(root / "images/context/source.png", render_source_image(source.source_image, scale))
    save_png(root / "images/context/mixed-target.png", render_raw_density(source.mixed_target, scale))


def _derive_path(run_dir: Path, path_id: int, source: Any) -> dict[str, Any]:
    records, states = _scan_eval(run_dir, path_id)
    if len(records) != SHARD_COUNT:
        raise RolloutReweightedRunError("cannot derive an incomplete evaluation path")
    root = _eval_root(run_dir, path_id)
    trajectory = np.stack(states).astype(np.float64, copy=False)
    atomic_rollout_npz(root / "trajectory_boundaries.npz", {"completed_reverse_steps": np.arange(0, 513, 8, dtype=np.int64), "states": trajectory})
    rows = _metric_rows(path_id, states, source)
    atomic_write_csv(root / "metrics.csv", rows)
    atomic_write_json(root / "mechanism.json", _mechanism(records))
    health = _path_health(records, states)
    atomic_write_json(root / "health.json", health)
    _render_path(root, states, source)
    return {"summary": _path_summary(path_id, rows), "rows": rows, "health": health}


def _verify_csv(path: Path, expected: Sequence[Mapping[str, Any]]) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            observed = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RolloutReweightedRunError(f"derived CSV changed: {path}") from exc
    serialized = [{key: str(value) for key, value in row.items()} for row in expected]
    _require(observed == serialized, f"derived CSV changed: {path}")


def _verify_images(root: Path, states: Sequence[np.ndarray], source: Any) -> None:
    from PIL import Image
    scale = fixed_rendering_scale(source.source_image, source.mixed_target, float(source.metadata["lambda_mix"]))
    expected: dict[str, np.ndarray] = {}
    for completed in RENDER_HORIZONS:
        state = states[completed // SHARD_STEPS]
        for kind, renderer in (("raw", render_raw_density), ("demixed", render_background_demixed)):
            cells = [renderer(state[index], scale) for index in range(4)]
            expected.update({f"individual/{kind}/{ROW_ORDER[index]}/step-{completed:03d}.png": cell for index, cell in enumerate(cells)})
            expected[f"contact_sheets/{kind}-step-{completed:03d}.png"] = np.concatenate(cells, axis=1)
    expected["context/source.png"] = render_source_image(source.source_image, scale)
    expected["context/mixed-target.png"] = render_raw_density(source.mixed_target, scale)
    image_root = root / "images"
    actual = {path.relative_to(image_root).as_posix() for path in image_root.rglob("*.png")}
    _require(actual == set(expected) and len(actual) == 92, "rendered image inventory changed")
    for relative, pixels in expected.items():
        with Image.open(image_root / relative) as image:
            _require(np.array_equal(np.asarray(image), pixels), f"rendered image pixels changed: {relative}")


def _evaluation_start_states(run_dir: Path) -> np.ndarray:
    path = run_dir / "inputs/evaluation_start_states.npz"
    states = np.stack([_prior_state(seed)[0] for seed in EVAL_PRIOR_SEEDS])
    expected = {"path_ids": np.asarray(EVAL_PATH_IDS, np.int64), "prior_seeds": np.asarray(EVAL_PRIOR_SEEDS, np.int64), "states": states}
    if path.exists():
        observed = _npz(path)
        _require(set(observed) == set(expected) and all(np.array_equal(observed[key], value) for key, value in expected.items()), "evaluation priors changed")
    else:
        atomic_rollout_npz(path, expected)
    return states


def _evaluate(run_dir: Path, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with _attempt(run_dir, "evaluation-setup"):
        candidate = _selected_payload(run_dir / "training/selected_checkpoint.pt")
        baseline_model = _load_checkpoint(run_dir / "inputs/baseline_checkpoint.pt", device)
        candidate_model = _candidate_model(run_dir, device)
        source = load_verified_source_target(run_dir / "inputs/source")
        _require((rollout_array_sha256(source.source_image), rollout_array_sha256(source.mixed_target)) == (SOURCE_ARRAY_SHA256, TARGET_ARRAY_SHA256), "source/target arrays changed")
        starts = _evaluation_start_states(run_dir)
        from mnist.d0_jacobi_rb_cuda_deferred import prepare_alpha1_rb_transition_batch_cuda_candidate
        profile = JacobiRBCudaProfile()
        prepared = prepare_alpha1_rb_transition_batch_cuda_candidate(device=device, profile=profile)
        seeds = prepare_deferred_reference_rng_seed_map(prepared_backend=prepared, root_seed=TRANSITION_ROOT_SEED, stream_role=STREAM_ROLE)
    _resource_check(run_dir, device)
    summaries, all_rows, health_rows = [], [], []
    for slot, path_id in enumerate(EVAL_PATH_IDS):
        _recover_eval_orphan(run_dir, path_id)
        records, _ = _scan_eval(run_dir, path_id)
        identity = _cutoff_identity(records)
        _require(identity["passed"] == 1, "cutoff identity failed before resume")
        specs = _row_specs(path_id, candidate["state_sha256"])
        controllers = {ROW_ORDER[1]: CompletedStepCutoffTangentScoreController(baseline_model, 216), ROW_ORDER[2]: CompletedStepCutoffTangentScoreController(candidate_model, 216), ROW_ORDER[3]: TargetFractionOracleController(source.mixed_target, microsteps=MICROSTEPS).to(device)}
        bank = FusedTangentControllerBank(specs, controllers)
        binding = _controller_binding(specs, candidate["state_sha256"])

        def admit(_plan: Any) -> None:
            committed, _states = _scan_eval(run_dir, path_id)
            _require(_cutoff_identity(committed)["passed"] == 1, "cutoff identity failed before a new shard")
            pending = max(time.perf_counter() - started, 0.0)
            if committed:
                observed = [float(row.get("elapsed_seconds", 0.0)) for row in committed]
                remaining = (3 - slot) * SHARD_COUNT - len(committed)
                ledger = _read_json(run_dir / "resource_ledger.json")
                projection = float(ledger["active_seconds"]) + pending + 1.2 * max(observed) * remaining + 300.0
                _project(run_dir, "evaluation", projection, path_id=path_id, committed_shards=len(committed))
            _resource_check(run_dir, device, pending_seconds=pending)

        with _attempt(run_dir, f"evaluation-path-0x{path_id:X}") as started:
            factory = lambda _index: CandidateApproximateFusedReference(profile=profile, root_seed=TRANSITION_ROOT_SEED, stream_role=STREAM_ROLE, prepared_backend=prepared, prepared_rng_seeds=seeds)
            run_fused_reverse_family(np.repeat(starts[slot][None, :], 4, axis=0), sequence=reverse_suffix_sequence(511), output_dir=_eval_root(run_dir, path_id) / "reverse", family_name="same-path-four-row", segment_name="complete-512", row_specs=specs, controller_bank=bank, reference_factory=factory, controller_binding=binding, rng_binding=_rng_binding(path_id), label=3, microsteps=MICROSTEPS, device=device, before_uncommitted_shard=admit, reference_contract="candidate_approximate")
        _resource_check(run_dir, device)
        with _attempt(run_dir, f"derive-path-0x{path_id:X}"):
            derived = _derive_path(run_dir, path_id, source)
        _resource_check(run_dir, device)
        summaries.append(derived["summary"])
        all_rows.extend(derived["rows"])
        health_rows.append(derived["health"])
        _require(derived["health"].get("passed") == 1, f"Gate C trajectory health failed for path 0x{path_id:X}")
    with _attempt(run_dir, "evaluation-aggregation"):
        aggregate_health = {"schema": VERSION + "-aggregate-health", "passed": int(all(row["passed"] == 1 for row in health_rows)), "paths": health_rows}
        atomic_write_csv(run_dir / "evaluation/aggregate_metrics.csv", all_rows)
        atomic_write_json(run_dir / "evaluation/aggregate_mechanism.json", {"schema": VERSION + "-aggregate-mechanism", "path_summaries": summaries})
    _resource_check(run_dir, device)
    return summaries, all_rows, aggregate_health


def _gap_closure(rows: Mapping[str, Mapping[str, float]]) -> float:
    zero, candidate, oracle = (float(rows[ROW_ORDER[index]]["squared_l2_error"]) for index in (0, 2, 3))
    denominator = zero - oracle
    return (zero - candidate) / denominator if denominator > 0 else -math.inf


def _validate_human_review(value: Mapping[str, Any]) -> tuple[bool, list[Mapping[str, Any]]]:
    status, labels = value.get("status"), value.get("labels")
    _require(value.get("schema") == VERSION + "-human-review" and status in {"not_reviewed", "reviewed"} and value.get("allowed_labels") == list(HUMAN_LABELS) and value.get("automated_recognizability") == 0 and isinstance(labels, list), "human review record is malformed")
    if status == "not_reviewed":
        _require(not labels and value.get("reviewer") is None, "pending human review contains a label or reviewer")
        return False, []
    reviewer = value.get("reviewer")
    _require(isinstance(reviewer, str) and reviewer.strip() and not (reviewer.startswith("<") and reviewer.endswith(">")) and len(labels) == 3 and {row.get("path_id") for row in labels if isinstance(row, Mapping)} == set(EVAL_PATH_IDS) and all(isinstance(row, Mapping) and row.get("label") in HUMAN_LABELS for row in labels), "completed human review is malformed")
    return True, labels


def _outcome(path_summaries: Sequence[Mapping[str, Any]], human_review: Mapping[str, Any], health: Mapping[str, Any]) -> dict[str, Any]:
    _require(len(path_summaries) == 3 and {int(row["path_id"]) for row in path_summaries} == set(EVAL_PATH_IDS), "outcome requires the three frozen evaluation paths")
    endpoint = [row["horizons"]["512"] for row in path_summaries]
    oracle_gate = all(float(rows[ROW_ORDER[3]]["squared_l2_error"]) <= 0.99 * float(rows[ROW_ORDER[0]]["squared_l2_error"]) for rows in endpoint)
    wins = sum(float(rows[ROW_ORDER[2]]["squared_l2_error"]) < float(rows[ROW_ORDER[1]]["squared_l2_error"]) for rows in endpoint)
    mean_candidate = math.fsum(float(rows[ROW_ORDER[2]]["squared_l2_error"]) for rows in endpoint) / 3.0
    mean_baseline = math.fsum(float(rows[ROW_ORDER[1]]["squared_l2_error"]) for rows in endpoint) / 3.0
    endpoint_effects = [{"path_id": int(summary["path_id"]), "candidate_minus_zero_error": float(rows[ROW_ORDER[2]]["squared_l2_error"]) - float(rows[ROW_ORDER[0]]["squared_l2_error"]), "candidate_minus_frozen_error": float(rows[ROW_ORDER[2]]["squared_l2_error"]) - float(rows[ROW_ORDER[1]]["squared_l2_error"]), "relative_improvement_over_frozen": (float(rows[ROW_ORDER[1]]["squared_l2_error"]) - float(rows[ROW_ORDER[2]]["squared_l2_error"])) / float(rows[ROW_ORDER[1]]["squared_l2_error"]) if float(rows[ROW_ORDER[1]]["squared_l2_error"]) else 0.0} for summary, rows in zip(path_summaries, endpoint)]
    terminal_closures = [_gap_closure(rows) for rows in endpoint]
    correlations = [float(rows[ROW_ORDER[2]]["centered_correlation"]) for rows in endpoint]
    reviewed, labels = _validate_human_review(human_review)
    digit3 = sum(row.get("label") == "3" for row in labels)
    healthy = health.get("passed") == 1
    gate_e = healthy and oracle_gate and wins >= 2 and mean_candidate < mean_baseline and float(np.median(terminal_closures)) >= 0.10 and float(np.median(correlations)) >= 0.5 and digit3 >= 2 and reviewed
    boundary_marker = False
    boundary_detail: dict[str, Any] = {}
    retained = float(np.median(terminal_closures))
    for horizon in (208, 216):
        rows_at_horizon = [row["horizons"][str(horizon)] for row in path_summaries]
        boundary_wins = sum(float(rows[ROW_ORDER[2]]["squared_l2_error"]) < float(rows[ROW_ORDER[1]]["squared_l2_error"]) for rows in rows_at_horizon)
        closures = [_gap_closure(rows) for rows in rows_at_horizon]
        median_boundary = float(np.median(closures))
        marker = boundary_wins >= 2 and median_boundary >= 0.10 and (retained < 0.05 or retained < 0.5 * median_boundary)
        boundary_detail[str(horizon)] = {"candidate_wins": boundary_wins, "median_oracle_gap_closure": median_boundary, "marker": int(marker)}
        boundary_marker = boundary_marker or marker
    routes = ((not healthy, "repair", "repair only the localized trajectory, pairing, cutoff, or resource defect and rerun unchanged"), (not oracle_gate, "common_path_repair", "fix the prior, oracle, or composition path before changing the learner"), (not reviewed, "human_review_required", "visually label each candidate endpoint as 0-9, ambiguous, or noise; target-like means label 3"), (gate_e, "stage_f_plan", "plan bounded multi-image/multi-seed Stage F; do not auto-launch"), (boundary_marker, "material_controller_comparison", "run one material state/time-dependent controller comparison; do not sweep gains or cutoffs"), (True, "conventional_ddpm", "stop incremental Jacobi/RB learner work and run the conventional MNIST DDPM baseline"))
    _, route, action = next(row for row in routes if row[0])
    return {"schema": VERSION + "-outcome", "research_mode": "exploratory", "health_passed": int(healthy), "gate_d_passed": int(oracle_gate), "gate_e_passed": int(gate_e), "human_review_completed": int(reviewed), "candidate_endpoint_wins": wins, "mean_candidate_endpoint_error": mean_candidate, "mean_frozen_endpoint_error": mean_baseline, "mean_relative_improvement_over_frozen": math.fsum(float(row["relative_improvement_over_frozen"]) for row in endpoint_effects) / 3.0, "endpoint_effects": endpoint_effects, "median_terminal_oracle_gap_closure": float(np.median(terminal_closures)), "median_candidate_endpoint_correlation": float(np.median(correlations)), "candidate_digit3_reviews": digit3, "material_controller_marker": int(boundary_marker), "boundary_detail": boundary_detail, "route": route, "next_action": action, "confirmatory_claim": 0, "stage_f_machine_eligible": 0, "stage_f_automatically_launched": 0, "automatic_compute_launched": 0, "claim_scope": "one target-specific model, one training seed, three fixed fresh prior paths, approximate candidate law"}


def _human_template() -> dict[str, Any]:
    return {"schema": VERSION + "-human-review", "status": "not_reviewed", "reviewer": None, "labels": [], "allowed_labels": list(HUMAN_LABELS), "automated_recognizability": 0}


def _report_text(outcome: Mapping[str, Any], health: Mapping[str, Any], run_dir: Path | None = None) -> str:
    if isinstance(outcome, (str, Path)):
        supplied_health = run_dir
        run_dir, outcome, health = Path(outcome), health, supplied_health  # type: ignore[assignment]
    _require(isinstance(health, Mapping), "report health record is invalid")
    rich = run_dir is not None and (run_dir / "training/selection.json").is_file()
    selection = _read_json(run_dir / "training/selection.json") if rich else {}
    ledger = _read_json(run_dir / "resource_ledger.json") if rich else {}
    training_diagnostics = _read_json(run_dir / "training/diagnostics.json") if rich else {"splits": {}}
    human = _read_json(run_dir / "evaluation/human_review.json") if rich else {"labels": []}
    summaries = _read_json(run_dir / "evaluation/aggregate_mechanism.json")["path_summaries"] if rich else []
    endpoint_lines = []
    for summary in summaries:
        rows = summary["horizons"]["512"]
        cells = "; ".join(f"{key}: L2={float(rows[key]['squared_l2_error']):.17g}, corr={float(rows[key]['centered_correlation']):.17g}" for key in ROW_ORDER)
        effect = float(rows[ROW_ORDER[1]]["squared_l2_error"]) - float(rows[ROW_ORDER[2]]["squared_l2_error"])
        zero_effect = float(rows[ROW_ORDER[0]]["squared_l2_error"]) - float(rows[ROW_ORDER[2]]["squared_l2_error"])
        relative = effect / float(rows[ROW_ORDER[1]]["squared_l2_error"]) if float(rows[ROW_ORDER[1]]["squared_l2_error"]) else 0.0
        state_l2 = float(rows[ROW_ORDER[2]].get("state_vs_frozen_squared_l2", math.nan))
        endpoint_lines.append(f"- `0x{int(summary['path_id']):X}` — {cells}; improvement over frozen={effect:.17g} (relative {relative:.17g}); improvement over zero={zero_effect:.17g}; candidate-vs-frozen state L2²={state_l2:.17g}")
    invalid = sum(int(path.get("invalid_or_fallback_count", 0)) + sum(int(value) for value in path.get("controller_forbidden_counts", {}).values()) + sum(int(value) for value in path.get("dynamics_invalid_counts", {}).values()) for path in health.get("paths", ()))
    mass = max((float(path.get("maximum_mass_error", 0.0)) for path in health.get("paths", ())), default=0.0)
    peak = max((float(path.get("maximum_cuda_memory_fraction", 0.0)) for path in health.get("paths", ())), default=0.0)
    labels = ", ".join(f"0x{int(row['path_id']):X}={row['label']}" for row in human.get("labels", ()))
    training_lines = "; ".join(f"{split}: parent weighted/original={row.get('parent_weighted_mse')}/{row.get('parent_original_unweighted_mse')}, candidate weighted/original={row.get('candidate_weighted_mse')}/{row.get('candidate_original_unweighted_mse')}" for split, row in training_diagnostics.get("splits", {}).items())
    mechanism_lines = [f"0x{path_id:X}={next(row for row in _read_json(_eval_root(run_dir, path_id) / 'mechanism.json')['per_row'] if row['row_key'] == ROW_ORDER[2]).get('control_to_reference_rms_ratio')}" for path_id in EVAL_PATH_IDS] if rich else []
    lines = ["# Stage E rollout-reweighted pilot", "", "Primary mode: exploratory. Decision: does target-preserving rollout-proximity reweighting materially and recognizably improve fresh intended-prior output over the frozen cutoff-216 learner?", "", f"Selected update/MSE: `{selection.get('fine_tune_update')}` / `{selection.get('validation_mse')}`. Active/cap seconds: `{ledger.get('active_seconds')}` / `{ledger.get('maximum_active_seconds')}`. Health: `{health.get('passed')}`; invalid={invalid}; max mass error={mass}; max CUDA fraction={peak}.", f"Weighted/original-unweighted training diagnostics: {training_lines}. Candidate control/reference RMS ratios: `{', '.join(mechanism_lines)}`.", "", "## Fresh-path endpoints", "", *endpoint_lines, "", f"Wins: `{outcome.get('candidate_endpoint_wins')}/3`; mean candidate/frozen L2: `{outcome.get('mean_candidate_endpoint_error')}` / `{outcome.get('mean_frozen_endpoint_error')}`; relative frozen improvement: `{outcome.get('mean_relative_improvement_over_frozen')}`; median zero-oracle closure/correlation: `{outcome.get('median_terminal_oracle_gap_closure')}` / `{outcome.get('median_candidate_endpoint_correlation')}`; 208/216: `{json.dumps(outcome.get('boundary_detail'), sort_keys=True)}`; labels: `{labels}`.", f"Gate D/E/review: `{outcome.get('gate_d_passed')}` / `{outcome.get('gate_e_passed')}` / `{outcome.get('human_review_completed')}`. Route: `{outcome.get('route')}`. Next: {outcome.get('next_action')}.", "", "Approximate-backend, one-target, one-training-seed exploratory evidence only—not a diverse generator, exact-law/population result, or confirmatory claim. Failures remain saved; Stage F and all automatic launches remain zero.", "", "Load-bearing: `training/selected_checkpoint.pt`, `training/diagnostics.json`, `reweighting/summary.json`, `evaluation/aggregate_metrics.csv`, `evaluation/aggregate_mechanism.json`, `evaluation/human_review.json`, `outcome.json`."]
    return "\n".join(lines) + "\n"


def _refresh_manifest(run_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.relative_to(run_dir).as_posix()):
        if path.is_file() and path != run_dir / "artifact_manifest.json":
            _require(not path.is_symlink() and path.stat().st_nlink == 1, "manifest cannot bind linked artifacts")
            rows.append({"path": path.relative_to(run_dir).as_posix(), "size": path.stat().st_size, "sha256": file_fingerprint(path)})
    manifest = {"schema": VERSION + "-artifact-manifest", "artifact_count": len(rows), "artifact_bytes": sum(row["size"] for row in rows), "artifacts": rows}
    atomic_write_json(run_dir / "artifact_manifest.json", manifest)
    return manifest


def _verify_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "artifact_manifest.json")
    rows = manifest.get("artifacts")
    actual = sorted(path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file() and path != run_dir / "artifact_manifest.json")
    _require(manifest.get("schema") == VERSION + "-artifact-manifest" and isinstance(rows, list) and [row.get("path") for row in rows] == actual, "artifact manifest inventory changed")
    for row in rows:
        path = run_dir / row["path"]
        _require(not path.is_symlink() and path.stat().st_nlink == 1 and (path.stat().st_size, file_fingerprint(path)) == (row.get("size"), row.get("sha256")), f"artifact changed: {row.get('path')}")
    _require((manifest.get("artifact_count"), manifest.get("artifact_bytes")) == (len(rows), sum(int(row["size"]) for row in rows)), "artifact manifest totals changed")
    return manifest


def _verify_source_files(run_dir: Path) -> None:
    sources = _read_json(run_dir / "bindings.json").get("source_files")
    repository = Path(__file__).resolve().parents[1]
    if not isinstance(sources, Mapping) or set(sources) != set(_DIRECT_SOURCE_FILES) or any(not (repository / relative).is_file() or file_fingerprint(repository / relative) != digest for relative, digest in sources.items()):
        raise RolloutReweightedRunError("bound source closure changed")


def _terminal_inputs(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate = _read_json(run_dir / "evaluation/aggregate_mechanism.json")
    summaries = aggregate.get("path_summaries")
    if not isinstance(summaries, list):
        raise RolloutReweightedRunError("aggregate path summaries changed")
    health_rows = [_read_json(_eval_root(run_dir, path_id) / "health.json") for path_id in EVAL_PATH_IDS]
    resource = _validate_ledger(run_dir)
    health = {"schema": VERSION + "-aggregate-health", "passed": int(all(row.get("passed") == 1 for row in health_rows) and resource["passed"] == 1), "paths": health_rows, "resource": resource}
    return summaries, health


def _finalize(run_dir: Path, *, charge: bool = True) -> dict[str, Any]:
    if charge:
        with _attempt(run_dir, "terminalization"):
            _finalize(run_dir, charge=False)
        _resource_check(run_dir)
        return _finalize(run_dir, charge=False)
    human_path = run_dir / "evaluation/human_review.json"
    if not human_path.exists(): atomic_write_json(human_path, _human_template())
    summaries, health = _terminal_inputs(run_dir)
    human = _read_json(human_path)
    outcome = _outcome(summaries, human, health)
    atomic_write_json(run_dir / "outcome.json", outcome)
    _atomic_text(run_dir / "REPORT.md", _report_text(outcome, health, run_dir))
    state = "complete" if human.get("status") == "reviewed" else "human_review_required"
    ledger = _read_json(run_dir / "resource_ledger.json")
    atomic_write_json(run_dir / "status.json", {"schema": VERSION + "-status", "state": state, "resumable": 0, "completed_evaluation_paths": 3, "active_seconds": ledger["active_seconds"], "maximum_active_seconds": ACTIVE_SECONDS_CAP, "error": None})
    _refresh_manifest(run_dir)
    return outcome


def verify_run(run_dir: Path, *, _completed_ledger: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    before = {path.relative_to(run_dir).as_posix(): (path.stat().st_size, file_fingerprint(path)) for path in run_dir.rglob("*") if path.is_file()}
    _require(_read_json(run_dir / "config.json") == _frozen_config(), "frozen configuration changed")
    _verify_source_files(run_dir)
    _verify_copied_inputs(run_dir)
    _load_checkpoint(run_dir / "inputs/baseline_checkpoint.pt")
    _selected_payload(run_dir / "training/selected_checkpoint.pt")
    _require(_read_json(run_dir / "inputs/role_inventory.json") == _role_inventory(), "evidence role inventory changed")
    starts = _npz(run_dir / "inputs/evaluation_start_states.npz")
    expected_states = np.stack([_prior_state(seed)[0] for seed in EVAL_PRIOR_SEEDS])
    _require(set(starts) == {"path_ids", "prior_seeds", "states"} and np.array_equal(starts["path_ids"], EVAL_PATH_IDS) and np.array_equal(starts["prior_seeds"], EVAL_PRIOR_SEEDS) and np.array_equal(starts["states"], expected_states), "evaluation start states changed")
    source = load_verified_source_target(run_dir / "inputs/source")
    summaries, all_rows, health_rows = [], [], []
    for path_id in EVAL_PATH_IDS:
        records, states = _scan_eval(run_dir, path_id)
        health = _path_health(records, states)
        _require(_read_json(_eval_root(run_dir, path_id) / "health.json") == health, "derived path health changed")
        trajectory = _npz(_eval_root(run_dir, path_id) / "trajectory_boundaries.npz")
        _require(np.array_equal(trajectory.get("completed_reverse_steps"), np.arange(0, 513, 8)) and np.array_equal(trajectory.get("states"), np.stack(states)), "derived trajectory changed")
        rows = _metric_rows(path_id, states, source)
        _verify_csv(_eval_root(run_dir, path_id) / "metrics.csv", rows)
        _require(_read_json(_eval_root(run_dir, path_id) / "mechanism.json") == _mechanism(records), "derived mechanism changed")
        _verify_images(_eval_root(run_dir, path_id), states, source)
        summaries.append(_path_summary(path_id, rows))
        all_rows.extend(rows)
        health_rows.append(health)
    aggregate = _read_json(run_dir / "evaluation/aggregate_mechanism.json")
    _require(aggregate == {"schema": VERSION + "-aggregate-mechanism", "path_summaries": summaries}, "aggregate mechanism changed")
    _verify_csv(run_dir / "evaluation/aggregate_metrics.csv", all_rows)
    resource = _validate_ledger(run_dir, completed=_completed_ledger)
    aggregate_health = {"schema": VERSION + "-aggregate-health", "passed": int(all(row["passed"] == 1 for row in health_rows) and resource["passed"] == 1), "paths": health_rows, "resource": resource}
    human = _read_json(run_dir / "evaluation/human_review.json")
    outcome = _outcome(summaries, human, aggregate_health)
    _require(_read_json(run_dir / "outcome.json") == outcome and (run_dir / "REPORT.md").read_text(encoding="utf-8") == _report_text(outcome, aggregate_health, run_dir), "terminal outcome/report changed")
    status = _read_json(run_dir / "status.json")
    expected_state = "complete" if human.get("status") == "reviewed" else "human_review_required"
    _require((status.get("state"), status.get("completed_evaluation_paths"), status.get("resumable"), status.get("active_seconds")) == (expected_state, 3, 0, resource["active_seconds"]), "terminal status changed")
    manifest = _verify_manifest(run_dir)
    after = {path.relative_to(run_dir).as_posix(): (path.stat().st_size, file_fingerprint(path)) for path in run_dir.rglob("*") if path.is_file()}
    _require(before == after, "read-only verification mutated the run")
    return {"passed": 1, "outcome": outcome, "manifest": manifest}


def record_human_review(run_dir: Path, labels_by_path: Mapping[Any, str], reviewer: str = "codex-visual-review") -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    current = verify_run(run_dir)
    _require(current["outcome"].get("human_review_completed") != 1, "human review is already recorded")
    normalized = {}
    for key, value in labels_by_path.items():
        path_id = int(key, 0) if isinstance(key, str) else int(key)
        _require(value in HUMAN_LABELS, "human review label is invalid")
        normalized[path_id] = value
    _require(set(normalized) == set(EVAL_PATH_IDS) and str(reviewer).strip() and not (str(reviewer).startswith("<") and str(reviewer).endswith(">")), "human review must label every frozen path with a real reviewer")
    record = {"schema": VERSION + "-human-review", "status": "reviewed", "reviewer": str(reviewer).strip(), "reviewed_at": _utc_now(), "labels": [{"path_id": path_id, "label": normalized[path_id]} for path_id in EVAL_PATH_IDS], "allowed_labels": list(HUMAN_LABELS), "automated_recognizability": 0}
    atomic_write_json(run_dir / "evaluation/human_review.json", record)
    outcome = _finalize(run_dir)
    verify_run(run_dir)
    return outcome


def _record_failure(run_dir: Path, state: str, exc: BaseException) -> None:
    atomic_write_json(run_dir / "status.json", {"schema": VERSION + "-status", "state": state, "resumable": 1, "error": str(exc)})
    atomic_write_json(run_dir / "failure.json", {"schema": VERSION + "-failure", "kind": "resource_pause" if state == "resource_paused" else type(exc).__name__, "message": str(exc), "at": _utc_now()})
    _refresh_manifest(run_dir)


def _run(run_dir: Path, args: argparse.Namespace) -> int:
    binding = _read_json(run_dir / "bindings.json").get("external_locators", {})
    _require(Path(binding.get("training_parent", "")).resolve() == Path(args.training_parent).resolve(), "training-parent locator changed on resume")
    _verify_source_files(run_dir)
    _verify_copied_inputs(run_dir)
    _validate_ledger(run_dir)
    prior_status = _read_json(run_dir / "status.json")
    successful = {event["role"] for event in _read_json(run_dir / "resource_ledger.json")["events"] if event.get("failed", 0) == 0}
    if prior_status.get("state") in {"complete", "human_review_required"} and {"terminalization", "terminal-verification"} <= successful:
        try: _verify_manifest(run_dir)
        except Exception: pass
        else:
            verify_run(run_dir); return 0
    if (run_dir / "artifact_manifest.json").exists() and prior_status.get("state") in {"resource_paused", "failed"}:
        _verify_manifest(run_dir)
    device = torch.device(args.device)
    _require(device.type == "cuda" and torch.cuda.is_available(), "production run requires CUDA")
    atomic_write_json(run_dir / "status.json", {"schema": VERSION + "-status", "state": "running", "resumable": 1, "completed_evaluation_paths": 0, "error": None})
    try:
        enable_deterministic_torch()
        _train(run_dir, Path(args.training_parent).resolve(), device)
        _evaluate(run_dir, device)
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failure = _read_json(failure_path)
            failure.update({"resolved": 1, "resolved_at": _utc_now(), "resolution": "resumed run reached terminalization"})
            atomic_write_json(failure_path, failure)
        _finalize(run_dir)
        with _attempt(run_dir, "terminal-verification"):
            verify_run(run_dir, _completed_ledger=False)
        _resource_check(run_dir, device)
        _finalize(run_dir, charge=False)
        verify_run(run_dir)
        return 0
    except ResourcePause as exc:
        _record_failure(run_dir, "resource_paused", exc)
        return 2
    except Exception as exc:
        _record_failure(run_dir, "failed", exc)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exploratory rollout-reweighted training and three-path prior evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for flag in ("--repository-root", "--training-parent", "--baseline-run-dir", "--stage-e-run-dir", "--runs-root", "--approval-reference"): run.add_argument(flag, required=True)
    run.add_argument("--run-name", default=RUN_NAME)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--maximum-active-seconds", type=float, default=ACTIVE_SECONDS_CAP)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        verify_run(Path(args.run_dir).resolve())
        return 0
    return _run(_initialize_run(args), args)


if __name__ == "__main__":
    raise SystemExit(main())
