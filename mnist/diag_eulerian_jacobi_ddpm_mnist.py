from __future__ import annotations

"""Exploratory end-to-end runner for the fixed-grid Eulerian Jacobi DDPM-v0."""

import argparse
import ast
import copy
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from mnist import eulerian_jacobi_ddpm as core
from mnist.d0_jacobi_rb_global_dilated import (
    GLOBAL_DILATED_PARAMETER_COUNT,
    GLOBAL_DILATED_VERSION,
    global_dilated_architecture_contract,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    v3_transitive_source_paths,
)
from mnist.mnist_generation_benchmark import (
    MNIST_ARFF_SHA256,
    compute_generation_metrics,
    evaluate_generated_labels,
    load_test_mnist_terminal,
    load_train_validation_mnist,
    score_human_review,
    sha256_file,
    write_blinded_review_bundle,
    write_contact_sheet,
)
from mnist.conditioned_diffusion import SmallMnistCNN, evaluate_image_classifier


VERSION = "eulerian-jacobi-ddpm-mnist-v0"
EMBEDDED_PILOT_DIRECTORY = "objective_pilot"
ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256 = (
    "3d31d42a14fee0ecc72adc1644718a037cc48e649948407da6c272b981840c92"
)
ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256 = (
    "e6cd9e49ca61237d3a10e9ad2fe0ad09f7a33ea22911fdd73fd99f3a07e4c668"
)
ACCEPTED_DDPM_METRICS_SHA256 = (
    "2e2fc75b6398f25a84bdaef0558c2f99c51117c71a009a3a94ed0afe8d27be33"
)
ACCEPTED_DDPM_MANIFEST_SHA256 = (
    "79aa5d9ae1ca6615a46c9d699f947bea4b6a380cc32e86547cc7e49cee612953"
)
PLACEHOLDER = re.compile(
    r"(?:<[^>]+>|fresh[ _-]*approval|approval[ _-]*reference|placeholder|todo|tbd)",
    re.I,
)
DIRECT_SOURCE_FILES = (
    "mnist/diag_eulerian_jacobi_ddpm_mnist.py",
    "mnist/eulerian_jacobi_ddpm.py",
    "mnist/d0_jacobi_rb_global_dilated.py",
    "mnist/mnist_generation_benchmark.py",
    "mnist/conditioned_diffusion.py",
)
POPULATION_FILES = (
    "selected_checkpoint.pt",
    "start_banks.npz",
    "populations.npz",
    "uint8_populations.npz",
    "telemetry.csv",
)
POPULATION_STAGE_NAMES = (
    "null_prior",
    "learned_prior",
    "null_forward_terminal",
    "learned_forward_terminal",
    "null_oracle",
    "oracle",
)


def _path_id_range(start: int, count: int) -> dict[str, Any]:
    values = np.arange(start, start + count, dtype=np.int64)
    return {
        "start": int(start),
        "stop_exclusive": int(start + count),
        "count": int(count),
        "sha256": hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest(),
    }


def _frozen_config() -> dict[str, Any]:
    """Return the complete, JSON-serializable scientific and resource contract."""

    return {
        "schema": VERSION,
        "research_mode": "exploratory",
        "decision": (
            "can one frozen ten-class global model trained on exact Jacobi "
            "Rao--Blackwell labels produce recognizable, class-consistent, "
            "noncollapsed MNIST images from the symmetric Dirichlet prior"
        ),
        "objective_artifact": (
            "all paired null/learned prior endpoints and trajectories, plus "
            "forward-terminal and oracle controls"
        ),
        "candidate_selection": "none; every declared endpoint is retained",
        "claim_scope": (
            "one exploratory model seed, one fixed K=128 split chain and the "
            "prespecified prior/forward-terminal populations"
        ),
        "core_contract": core.frozen_config(),
        "data": {
            "arff_sha256": MNIST_ARFF_SHA256,
            "train": [0, 55_000],
            "validation": [55_000, 60_000],
            "terminal_test": [60_000, 70_000],
            "lambda_mix": 0.35,
        },
        "chain": {
            "kind": "declared finite four-matching palindromic Jacobi split chain",
            "outer_steps": int(core.OUTER_STEPS),
            "reference_outer_steps": int(core.REFERENCE_OUTER_STEPS),
            "phase_count": 7,
            "active_edges_per_phase": 392,
            "controller_microsteps": 2,
            "controller_microsteps_per_phase": 2,
            "reverse_microstep_execution_quantiles": [0.75, 0.25],
            "reverse_time_binding": (
                "1-(7*outer_step+phase+q_mid)/(7*K), evaluated in "
                "backward phase order q_mid=0.75 then 0.25"
            ),
            "model_evaluations_per_path": 1_792,
            "model_evaluations_per_reverse_path": 1_792,
            "production_reverse_claim": (
                "boundary-preserving approximate controller composition, not an "
                "exact discrete reverse kernel"
            ),
        },
        "determinism": {
            "torch_deterministic_algorithms": 1,
            "cudnn_deterministic": 1,
            "cudnn_benchmark": 0,
            "cuda_matmul_tf32": 0,
            "cudnn_tf32": 0,
            "cublas_workspace_config": ":4096:8",
        },
        "records": {
            "train_paths": int(core.TRAIN_PATH_COUNT),
            "train_paths_per_class": int(core.TRAIN_PATH_COUNT // 10),
            "validation_paths": int(core.VALIDATION_PATH_COUNT),
            "validation_paths_per_class": int(core.VALIDATION_PATH_COUNT // 10),
            "records_per_path": 4,
            "labels_per_record": 392,
            "projected_cache_pair_transitions": 1_756_160_000,
            "path_split": "whole path; no path crosses train/validation roles",
            "later_state_dtype": "float32",
            "label_dtype": "float32",
            "transition_compute_dtype": "float64",
        },
        "path_ids": {
            "width_bits": 20,
            "reserved_haar": {"start": 0xB0000, "stop_exclusive": 0xB2000},
            "preflight_kernel": _path_id_range(0xB2000, 0x100),
            "preflight_k128_k512": _path_id_range(0xB2100, 1),
            "pilot_train": _path_id_range(0xB2200, 250),
            "pilot_validation": _path_id_range(0xB2300, 100),
            "pilot_prior": _path_id_range(0xB2500, 20),
            "pilot_forward_terminal": _path_id_range(0xB2520, 20),
            "pilot_oracle": _path_id_range(0xB2540, 10),
            "train": _path_id_range(0xB3000, 4_000),
            "validation": _path_id_range(0xB4000, 1_000),
            "prior_evaluation": _path_id_range(0xB5000, 160),
            "forward_terminal": _path_id_range(0xB5100, 40),
            "oracle_controls": _path_id_range(0xB5200, 10),
        },
        "model": {
            "class": "GlobalDilatedZeroBaselinePredictor",
            "version": GLOBAL_DILATED_VERSION,
            "parameter_count": int(GLOBAL_DILATED_PARAMETER_COUNT),
            "width": 32,
            "classes": 10,
            "source_conditioning": False,
            "architecture_fallback": None,
        },
        "training": {
            "seed": 0xE14A01,
            "optimizer": "Adam",
            "learning_rate": 2e-4,
            "batch_size": 64,
            "updates": 10_000,
            "ema_decay": 0.999,
            "gradient_norm_cap": 1.0,
            "validation_interval": 250,
            "selection": "earliest finite EMA checkpoint with minimum validation normalized MSE",
            "zero_checkpoint_eligible": False,
        },
        "objective_pilot": {
            "all_ten_classes_required": 1,
            "full_cache_launch_requires_pass": 1,
            "train_paths": 250,
            "train_paths_per_class": 25,
            "validation_paths": 100,
            "validation_paths_per_class": 10,
            "training_updates": 750,
            "prior_paths": 20,
            "prior_paths_per_class": 2,
            "forward_terminal_paths": 20,
            "forward_terminal_paths_per_class": 2,
            "oracle_paths": 10,
            "oracle_paths_per_class": 1,
            "projected_cache_pair_transitions": 122_931_200,
            "projected_reverse_paths": 100,
            "projected_reverse_reference_transitions": 140_492_800,
            "projected_forward_sampling_paths": 30,
            "projected_forward_sampling_transitions": 10_536_960,
            "projected_sampling_transition_work": 151_029_760,
            "projected_base_transition_work": 273_960_960,
            "shared_k128_k512_audit_transitions": 8_780_800,
            "projected_transition_work_including_shared_audit": 282_741_760,
            "full_run_admission": (
                "all health and Gate C pass; learned forward-terminal wins at least "
                "12/20 paths with aggregate L1 relative improvement at least 1%; "
                "learned controller RMS is finite and positive; learned prior top-1 "
                "requested-label accuracy is at least 0.20 and exceeds null; learned "
                "requested-class log probability beats null on at least 12/20 paired "
                "starts with positive mean improvement"
            ),
            "minimum_forward_l1_wins": 12,
            "minimum_aggregate_l1_relative_improvement": 0.01,
            "prior_scale_admission_gate_type": "diagnostic threshold",
            "minimum_learned_prior_requested_accuracy": 0.20,
            "learned_prior_accuracy_must_exceed_null": 1,
            "minimum_prior_requested_log_probability_wins": 12,
            "minimum_mean_prior_requested_log_probability_improvement": 0.0,
        },
        "populations": {
            "prior_per_class": 16,
            "prior_total": 160,
            "forward_terminal_per_class": 4,
            "forward_terminal_total": 40,
            "oracle_per_class": 1,
            "oracle_total": 10,
            "anchors": [0, 32, 64, 96, 128],
            "review_per_row_per_class": 2,
            "review_total": 40,
        },
        "rasterization": {
            "rule": "rint(clip(demixed_mass * (25471/255), 0, 1) * 255)",
            "scale": float(core.RASTER_SCALE),
            "lambda_mix": 0.35,
            "save_raw_and_demixed_masses": True,
            "source": "new frozen Eulerian global mass-to-uint8 binding",
        },
        "evaluator": {
            "source": "frozen conventional-DDPM benchmark evaluator",
            "accepted_checkpoint_sha256": ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256,
            "accepted_selection_sha256": ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256,
            "accepted_contextual_metrics_sha256": ACCEPTED_DDPM_METRICS_SHA256,
            "accepted_run_manifest_sha256": ACCEPTED_DDPM_MANIFEST_SHA256,
            "accepted_run_status": "complete",
            "minimum_real_accuracy": 0.97,
            "minimum_real_accuracy_gate_type": "diagnostic threshold",
            "accepted_for_exploratory_use": 1,
            "user_accepted_observed_validation_accuracy": 0.9546,
            "user_accepted_observed_test_accuracy": 0.9476,
            "batch_size": 256,
        },
        "diagnostics": {
            "human_recognizable_count": 15,
            "human_requested_label_count": 12,
            "machine_requested_label_accuracy": 0.70,
            "minimum_unique_learned_endpoints": 150,
            "minimum_diversity_ratio": 0.25,
            "oracle_minimum_improved_paths": 9,
        },
        "seeds": {
            "records_train": 0xE14B01,
            "records_validation": 0xE14B02,
            "prior": 0xE14C01,
            "prior_reverse": 0xE14C02,
            "forward_terminal": 0xE14C03,
            "oracle": 0xE14C04,
            "review": 0xE14D01,
        },
        "resource_defaults": {
            "maximum_active_seconds": 86_400.0,
            "maximum_storage_mib": 2_048.0,
            "maximum_cuda_fraction": 0.75,
            "terminal_reserve_seconds": 900.0,
        },
        "resource_projection": {
            "expected_wall_seconds": 60_000.0,
            "expected_accelerator_seconds": 50_000.0,
            "expected_peak_memory_bytes": 8_589_934_592,
            "expected_persisted_storage_bytes": 1_073_741_824,
            "new_source_test_artifact_complexity": 3,
            "full_cache_pair_transitions": 1_756_160_000,
            "reverse_reference_transitions_per_row": 1_404_928,
            "full_reverse_rows": 420,
            "full_reverse_reference_transitions": 590_069_760,
            "full_forward_sampling_rows": 50,
            "full_forward_sampling_transitions": 17_561_600,
            "full_sampling_transition_work": 607_631_360,
            "full_base_transition_work": 2_363_791_360,
            "k128_k512_audit_transitions": 8_780_800,
            "full_transition_work_including_audit": 2_372_572_160,
            "scientific_decision": (
                "whether the fresh fixed-grid Jacobi denoising model has direct "
                "prior-start image feasibility"
            ),
            "why_smaller_is_insufficient": (
                "local labels and short suffixes cannot test recursive prior-start generation"
            ),
        },
        "production_lifecycle": {
            "pilot_whole_run_restart": 1,
            "full_whole_run_restart": 1,
            "full_cache_shard_paths": None,
            "training_checkpoint_interval_updates": 250,
            "atomic_population_batches_required": 0,
            "bounded_full_stage_resume_required": 0,
            "bounded_full_stage_resume_implemented": 0,
            "production_launch_supported": 1,
            "blocker": None,
        },
        "automatic_launches": 0,
        "proxy_only_patches_since_last_objective_bearing_experiment": 0,
    }


FROZEN_CONFIG = _frozen_config()


class EulerianJacobiDDPMRunError(RuntimeError):
    pass


EulerianDDPMRunError = EulerianJacobiDDPMRunError


class ResourceStop(EulerianJacobiDDPMRunError):
    pass


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise EulerianJacobiDDPMRunError(message)


def _embedded_pilot_dir(
    run_dir: Path, admission: Mapping[str, Any] | None = None
) -> Path:
    """Resolve the sole admitted pilot location inside a portable run tree."""

    run_dir = Path(run_dir).resolve()
    if admission is not None:
        _require(
            admission.get("pilot_directory") == EMBEDDED_PILOT_DIRECTORY,
            "objective pilot must use the embedded relative directory",
        )
    pilot_dir = (run_dir / EMBEDDED_PILOT_DIRECTORY).resolve()
    _require(
        pilot_dir.parent == run_dir,
        "objective pilot must remain inside the run directory",
    )
    return pilot_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _semantic_sha256(value: Any) -> str:
    raw = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _atomic_replace(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    def write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    _atomic_replace(path, write)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    names = (
        list(fieldnames)
        if fieldnames is not None
        else list(dict.fromkeys(key for row in rows for key in row))
    )

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=names)
            output.writeheader()
            output.writerows([_jsonable(row) for row in rows])

    _atomic_replace(path, write)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    def write(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    _atomic_replace(path, write)


def _write_torch(path: Path, payload: Any) -> None:
    _atomic_replace(path, lambda temporary: torch.save(payload, temporary))


def _write_text(path: Path, text: str) -> None:
    _atomic_replace(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _source_hashes(repository_root: Path) -> dict[str, str]:
    entry_points = tuple(repository_root / relative for relative in DIRECT_SOURCE_FILES)
    paths = v3_transitive_source_paths(entry_points)
    return {
        path.relative_to(repository_root).as_posix(): _file_sha256(path)
        for path in paths
    }


def _scan_path_id_collisions(repository_root: Path) -> dict[str, Any]:
    """Fail if legacy MNIST source already claims the fresh 0xB2000+ roles."""

    allocated_start, allocated_stop = 0xB2000, 0xB520A
    haar_path = (repository_root / "mnist/d0_jacobi_rb_haar.py").resolve()
    excluded = {
        (repository_root / "mnist/eulerian_jacobi_ddpm.py").resolve(),
        (repository_root / "mnist/diag_eulerian_jacobi_ddpm_mnist.py").resolve(),
        haar_path,
    }
    collisions: list[dict[str, Any]] = []
    _require(haar_path.is_file(), "legacy Haar path-ID authority is missing")
    haar_tree = ast.parse(haar_path.read_text(encoding="utf-8"), filename=str(haar_path))
    haar_assignments: dict[str, Any] = {}
    for node in haar_tree.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else None
        )
        if (
            isinstance(target, ast.Name)
            and target.id in {"HAAR_ROLE_SLOTS", "HAAR_PRODUCTION_RESERVED"}
        ):
            haar_assignments[target.id] = ast.literal_eval(node.value)
    _require(
        set(haar_assignments) == {"HAAR_ROLE_SLOTS", "HAAR_PRODUCTION_RESERVED"},
        "legacy Haar path-ID authority cannot be parsed",
    )
    haar_ranges = [
        *haar_assignments["HAAR_ROLE_SLOTS"].values(),
        haar_assignments["HAAR_PRODUCTION_RESERVED"],
    ]
    semantic_ranges: list[dict[str, Any]] = []
    for lower, upper in haar_ranges:
        lower, upper = int(lower), int(upper)
        semantic_ranges.append(
            {
                "path": haar_path.relative_to(repository_root).as_posix(),
                "start": lower,
                "stop_exclusive": upper,
            }
        )
        if max(lower, allocated_start) < min(upper, allocated_stop):
            collisions.append(
                {
                    "path": haar_path.relative_to(repository_root).as_posix(),
                    "range": [lower, upper],
                }
            )
    token = re.compile(r"0x[bB][0-9a-fA-F]{4}")
    for path in sorted((repository_root / "mnist").glob("*.py")):
        if path.resolve() in excluded:
            continue
        for match in token.finditer(path.read_text(encoding="utf-8")):
            value = int(match.group(0), 16)
            if allocated_start <= value < allocated_stop:
                collisions.append(
                    {"path": path.relative_to(repository_root).as_posix(), "token": match.group(0)}
                )
    result = {
        "schema": VERSION + "-path-id-collision-scan",
        "allocated_start": allocated_start,
        "allocated_stop_exclusive": allocated_stop,
        "semantic_half_open_ranges": semantic_ranges,
        "collision_count": len(collisions),
        "collisions": collisions,
        "passed": int(not collisions),
    }
    _require(not collisions, f"fresh path-ID range collides with legacy source: {collisions}")
    return result


def _git_revision(repository_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _environment(device: str) -> dict[str, Any]:
    selected = torch.device(device)
    cuda_selected = selected.type == "cuda" and torch.cuda.is_available()
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "os": platform.platform(),
        "device": device,
        "cuda_available": int(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(selected) if cuda_selected else None,
        "gpu_total_memory_bytes": (
            int(torch.cuda.get_device_properties(selected).total_memory)
            if cuda_selected
            else None
        ),
        "gpu_capability": (
            list(torch.cuda.get_device_capability(selected)) if cuda_selected else None
        ),
        "torch_deterministic_algorithms": int(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": int(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": int(torch.backends.cudnn.benchmark),
        "cuda_matmul_tf32": int(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": int(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _status(run_dir: Path, state: str, *, error: str | None = None) -> None:
    _write_json(
        run_dir / "status.json",
        {
            "schema": VERSION + "-status",
            "state": state,
            "whole_run_restart_required": int(
                state in {"failed", "resource_stopped"}
            ),
            "error": error,
            "updated_at": _utc_now(),
        },
    )


def _validate_approval(value: str) -> str:
    cleaned = str(value).strip()
    _require(bool(cleaned) and not PLACEHOLDER.search(cleaned), "a real approval ID is required")
    return cleaned


def _validate_resources(
    maximum_active_seconds: float,
    maximum_storage_mib: float,
    maximum_cuda_fraction: float,
) -> None:
    values = (maximum_active_seconds, maximum_storage_mib, maximum_cuda_fraction)
    _require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in values), "resource caps must be finite numbers")
    _require(maximum_active_seconds > 0.0, "maximum active seconds must be positive")
    _require(maximum_storage_mib > 0.0, "maximum storage MiB must be positive")
    _require(0.0 < maximum_cuda_fraction <= 1.0, "maximum CUDA fraction must be in (0,1]")


def _resource_check(
    run_dir: Path,
    *,
    projected_seconds: float = 0.0,
    preserve_terminal_reserve: bool = True,
) -> None:
    ledger = _read_json(run_dir / "resource_ledger.json")
    config = _read_json(run_dir / "config.json")
    reserve = (
        float(config.get("resource_defaults", {}).get("terminal_reserve_seconds", 0.0))
        if preserve_terminal_reserve and config.get("schema") == VERSION
        else 0.0
    )
    storage = _directory_bytes(run_dir)
    ledger["peak_storage_bytes"] = max(int(ledger["peak_storage_bytes"]), storage)
    environment = _read_json(run_dir / "environment.json")
    device = torch.device(str(environment["device"]))
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = int(torch.cuda.max_memory_allocated(device))
        total = int(torch.cuda.get_device_properties(device).total_memory)
        fraction = allocated / total if total else 0.0
        ledger["peak_cuda_allocated_bytes"] = max(
            int(ledger["peak_cuda_allocated_bytes"]), allocated
        )
        ledger["peak_cuda_fraction"] = max(float(ledger["peak_cuda_fraction"]), fraction)
    _write_json(run_dir / "resource_ledger.json", ledger)
    if (
        float(ledger["active_seconds"])
        + float(projected_seconds)
        + reserve
        >= float(ledger["maximum_active_seconds"])
    ):
        raise ResourceStop("active-time cap or projected workload cap reached")
    if storage >= int(ledger["maximum_storage_bytes"]):
        raise ResourceStop("persisted-storage cap reached")
    if float(ledger["peak_cuda_fraction"]) >= float(ledger["maximum_cuda_fraction"]):
        raise ResourceStop("CUDA allocation cap reached")


def _charge(run_dir: Path, role: str, started: float, *, failed: bool = False) -> None:
    ledger = _read_json(run_dir / "resource_ledger.json")
    seconds = max(0.0, time.perf_counter() - started)
    ledger["active_seconds"] = math.fsum((float(ledger["active_seconds"]), seconds))
    ledger["events"].append(
        {"role": role, "seconds": seconds, "failed": int(failed), "at": _utc_now()}
    )
    _write_json(run_dir / "resource_ledger.json", ledger)


def _run_stage(
    run_dir: Path,
    role: str,
    function: Callable[[], Any],
    *,
    commit: Callable[[Any], None] | None = None,
    preserve_terminal_reserve: bool = True,
) -> Any:
    _resource_check(
        run_dir, preserve_terminal_reserve=preserve_terminal_reserve
    )
    environment = _read_json(run_dir / "environment.json")
    selected = torch.device(str(environment["device"]))
    if selected.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(selected)
    started = time.perf_counter()
    try:
        result = function()
        if selected.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(selected)
        if commit is not None:
            commit(result)
    except BaseException:
        if selected.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(selected)
        _charge(run_dir, role, started, failed=True)
        raise
    _charge(run_dir, role, started)
    _resource_check(
        run_dir, preserve_terminal_reserve=preserve_terminal_reserve
    )
    return result


def _manifest_rows(run_dir: Path) -> list[dict[str, Any]]:
    ignored = {"artifact_manifest.json", "SHA256SUMS.txt"}
    return [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.relative_to(run_dir).as_posix() not in ignored
    ]


def _seal_manifest(run_dir: Path) -> dict[str, Any]:
    _require(all(not path.is_symlink() for path in run_dir.rglob("*")), "linked artifacts are forbidden")
    rows = _manifest_rows(run_dir)
    manifest = {
        "schema": VERSION + "-artifact-manifest",
        "artifact_count": len(rows),
        "artifact_bytes": sum(int(row["size"]) for row in rows),
        "artifacts": rows,
    }
    _write_json(run_dir / "artifact_manifest.json", manifest)
    sums = [f"{row['sha256']}  {row['path']}" for row in rows]
    sums.append(f"{_file_sha256(run_dir / 'artifact_manifest.json')}  artifact_manifest.json")
    _write_text(run_dir / "SHA256SUMS.txt", "\n".join(sums) + "\n")
    return manifest


def _verify_manifest(run_dir: Path, ignored: set[str] | None = None) -> dict[str, Any]:
    ignored = ignored or set()
    manifest = _read_json(run_dir / "artifact_manifest.json")
    rows = manifest.get("artifacts")
    _require(isinstance(rows, list), "artifact manifest rows are invalid")
    expected = [str(row["path"]) for row in rows if str(row["path"]) not in ignored]
    actual = [row["path"] for row in _manifest_rows(run_dir) if row["path"] not in ignored]
    _require(expected == actual, "artifact inventory changed")
    for row in rows:
        relative = str(row["path"])
        if relative in ignored:
            continue
        path = run_dir / relative
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(row["size"])
            and _file_sha256(path) == row["sha256"],
            f"artifact changed: {relative}",
        )
    _require(
        manifest.get("artifact_count") == len(rows)
        and manifest.get("artifact_bytes") == sum(int(row["size"]) for row in rows),
        "artifact manifest totals changed",
    )
    expected_sums = [f"{row['sha256']}  {row['path']}" for row in rows]
    expected_sums.append(
        f"{_file_sha256(run_dir / 'artifact_manifest.json')}  artifact_manifest.json"
    )
    _require(
        (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
        == "\n".join(expected_sums) + "\n",
        "checksum file changed",
    )
    return manifest


def initialize_run(
    repository_root: Path,
    arff: Path,
    ddpm_run_dir: Path,
    run_dir: Path,
    *,
    device: str,
    maximum_active_seconds: float,
    maximum_storage_mib: float,
    maximum_cuda_fraction: float,
    approval_id: str,
    config: Mapping[str, Any] = FROZEN_CONFIG,
) -> Path:
    """Create a fresh production directory; incomplete runs are never resumed."""

    repository_root = Path(repository_root).resolve()
    arff = Path(arff).resolve()
    ddpm_run_dir = Path(ddpm_run_dir).resolve()
    run_dir = Path(run_dir).resolve()
    production_device = torch.device(device)
    _require(production_device.type == "cuda", "production requires a CUDA device; smoke/tests use CPU")
    _require(torch.cuda.is_available(), "CUDA production device is unavailable")
    approval = _validate_approval(approval_id)
    _validate_resources(maximum_active_seconds, maximum_storage_mib, maximum_cuda_fraction)
    _require(repository_root.is_dir(), "repository root is missing")
    _require(arff.is_file(), "authenticated MNIST ARFF is missing")
    _require(ddpm_run_dir.is_dir(), "conventional DDPM benchmark run is missing")
    _require(_file_sha256(arff) == MNIST_ARFF_SHA256, "authenticated MNIST ARFF hash mismatch")
    _require(not run_dir.exists(), "run directory already exists; restart in a new directory")
    evaluator_checkpoint = ddpm_run_dir / "evaluator/selected_checkpoint.pt"
    evaluator_selection = ddpm_run_dir / "evaluator/selection.json"
    ddpm_metrics = ddpm_run_dir / "evaluation/metrics.json"
    ddpm_manifest = ddpm_run_dir / "artifact_manifest.json"
    ddpm_status = ddpm_run_dir / "status.json"
    _require(
        evaluator_checkpoint.is_file()
        and evaluator_selection.is_file()
        and ddpm_metrics.is_file()
        and ddpm_manifest.is_file()
        and ddpm_status.is_file(),
        "accepted DDPM evaluator authority is incomplete",
    )
    _require(
        _file_sha256(evaluator_checkpoint)
        == ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256
        and _file_sha256(evaluator_selection)
        == ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256
        and _file_sha256(ddpm_metrics) == ACCEPTED_DDPM_METRICS_SHA256
        and _file_sha256(ddpm_manifest) == ACCEPTED_DDPM_MANIFEST_SHA256
        and _read_json(ddpm_status).get("state") == "complete",
        "supplied DDPM run is not the frozen user-accepted evaluator authority",
    )
    run_dir.mkdir(parents=True)
    for name in ("review", "images"):
        (run_dir / name).mkdir()
    runtime_config = copy.deepcopy(dict(config))
    runtime_config["execution_authority"] = {
        "approval_id": approval,
        "device": str(production_device),
        "maximum_active_seconds": float(maximum_active_seconds),
        "maximum_storage_mib": float(maximum_storage_mib),
        "maximum_cuda_fraction": float(maximum_cuda_fraction),
        "whole_run_restart_only": 1,
        "full_scale_launch_supported": 1,
    }
    _write_json(run_dir / "config.json", runtime_config)
    bindings = {
        "schema": VERSION + "-source-bindings",
        "repository_root": str(repository_root),
        "git_revision": _git_revision(repository_root),
        "source_files": _source_hashes(repository_root),
        "arff": str(arff),
        "arff_sha256": _file_sha256(arff),
        "ddpm_run_dir": str(ddpm_run_dir),
        "evaluator_checkpoint": str(evaluator_checkpoint),
        "evaluator_checkpoint_sha256": _file_sha256(evaluator_checkpoint),
        "evaluator_selection_sha256": _file_sha256(evaluator_selection),
        "ddpm_metrics_sha256": _file_sha256(ddpm_metrics),
        "ddpm_manifest_sha256": _file_sha256(ddpm_manifest),
        "ddpm_status": "complete",
        "config_sha256": _semantic_sha256(runtime_config),
    }
    _write_json(run_dir / "source_bindings.json", bindings)
    _write_json(run_dir / "path_id_audit.json", _scan_path_id_collisions(repository_root))
    _write_json(run_dir / "environment.json", _environment(device))
    _write_json(
        run_dir / "resource_ledger.json",
        {
            "schema": VERSION + "-resource-ledger",
            "approval_id": approval,
            "maximum_active_seconds": float(maximum_active_seconds),
            "maximum_storage_bytes": int(maximum_storage_mib * 1024**2),
            "maximum_cuda_fraction": float(maximum_cuda_fraction),
            "active_seconds": 0.0,
            "peak_storage_bytes": 0,
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_fraction": 0.0,
            "projected_cache_pair_transitions": int(
                config["records"]["projected_cache_pair_transitions"]
            ),
            "projected_reverse_reference_transitions": int(
                config["resource_projection"]["full_reverse_reference_transitions"]
            ),
            "projected_forward_sampling_transitions": int(
                config["resource_projection"]["full_forward_sampling_transitions"]
            ),
            "projected_sampling_transition_work": int(
                config["resource_projection"]["full_sampling_transition_work"]
            ),
            "projected_base_transition_work": int(
                config["resource_projection"]["full_base_transition_work"]
            ),
            "latest_projection": None,
            "events": [],
        },
    )
    _write_text(
        run_dir / "command.txt",
        subprocess.list2cmdline(
            [sys.executable, "-B", "-m", "mnist.diag_eulerian_jacobi_ddpm_mnist", *sys.argv[1:]]
        )
        + "\n",
    )
    _status(run_dir, "initialized")
    return run_dir


def initialize_smoke_run(run_dir: Path) -> Path:
    """Create a fresh, synthetic CPU-only lifecycle-smoke directory."""

    run_dir = Path(run_dir).resolve()
    _require(not run_dir.exists(), "smoke run directory already exists")
    run_dir.mkdir(parents=True)
    (run_dir / "review").mkdir()
    (run_dir / "images").mkdir()
    config = {
        "schema": VERSION + "-smoke",
        "research_mode": "engineering-control",
        "scientific_claim": None,
        "device": "cpu",
        "grid_size": 4,
        "class_count": 2,
        "outer_steps": 8,
        "population_schema": VERSION + "-populations",
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", _environment("cpu"))
    _write_json(
        run_dir / "resource_ledger.json",
        {
            "schema": VERSION + "-smoke-resource-ledger",
            "approval_id": "not-required-for-synthetic-smoke",
            "maximum_active_seconds": 600.0,
            "maximum_storage_bytes": 64 * 1024**2,
            "maximum_cuda_fraction": 0.01,
            "active_seconds": 0.0,
            "peak_storage_bytes": 0,
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_fraction": 0.0,
            "projected_pair_transitions": 0,
            "latest_projection": None,
            "events": [],
        },
    )
    _write_text(
        run_dir / "command.txt",
        "python -B -m mnist.diag_eulerian_jacobi_ddpm_mnist smoke "
        f"--output-dir {run_dir}\n",
    )
    _status(run_dir, "initialized")
    return run_dir


def _rasterize_rows(masses: np.ndarray) -> np.ndarray:
    values = np.asarray(masses, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 784, "population masses must be [N,784]")
    return np.stack([core.rasterize_unit_masses(row) for row in values])


def _demix_and_rasterize_rows(masses: np.ndarray) -> np.ndarray:
    values = np.asarray(masses, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 784, "mixed masses must be [N,784]")
    return np.stack(
        [core.rasterize_unit_masses(core.demix_unit_masses(row)) for row in values]
    )


def execute_smoke_run(run_dir: Path) -> dict[str, Any]:
    """Execute the core synthetic smoke and emit a minimal population fixture."""

    run_dir = Path(run_dir).resolve()
    _require(_read_json(run_dir / "status.json")["state"] == "initialized", "smoke is not initialized")
    result = _run_stage(run_dir, "synthetic_smoke", lambda: core.tiny_synthetic_smoke(seed=26_140_004))
    _require(int(result.get("passed", 0)) == 1, "core synthetic smoke failed")
    _write_json(run_dir / "smoke_result.json", result)

    path_ids = np.asarray([0xB5000, 0xB5001], dtype=np.int64)
    labels = np.asarray([0, 1], dtype=np.int64)
    sample_ids = np.asarray(["smoke-0", "smoke-1"], dtype=np.str_)
    starts = core.sample_dirichlet_starts(path_ids, root_seed=26_140_004)
    # These small arrays exercise only lifecycle, rendering, and integrity.  The
    # scientific null/learned/oracle execution is asserted by smoke_result.json.
    null = np.ascontiguousarray(starts, dtype=np.float64)
    learned = np.ascontiguousarray(starts[:, ::-1], dtype=np.float64)
    oracle = np.ascontiguousarray(np.roll(starts, 1, axis=1), dtype=np.float64)
    _write_npz(
        run_dir / "start_banks.npz",
        prior_starts=starts,
        requested_labels=labels,
        path_ids=path_ids,
        sample_ids=sample_ids,
    )
    _write_npz(
        run_dir / "populations.npz",
        null_prior=null,
        learned_prior=learned,
        oracle=oracle,
        requested_labels=labels,
        path_ids=path_ids,
        sample_ids=sample_ids,
    )
    _write_npz(
        run_dir / "uint8_populations.npz",
        null_prior=_rasterize_rows(null),
        learned_prior=_rasterize_rows(learned),
        oracle=_rasterize_rows(oracle),
        requested_labels=labels,
        sample_ids=sample_ids,
    )
    _write_csv(
        run_dir / "telemetry.csv",
        [
            {
                "population": row,
                "finite": 1,
                "nonnegative": 1,
                "maximum_mass_error": 0.0,
            }
            for row in ("null", "learned", "oracle")
        ],
    )
    model = core.make_model()
    _write_torch(
        run_dir / "selected_checkpoint.pt",
        {
            "schema": VERSION + "-smoke-checkpoint",
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "scientific_checkpoint": 0,
        },
    )
    _status(run_dir, "populations_written")
    return result


def _population_seal(run_dir: Path) -> dict[str, Any]:
    seal_path = run_dir / "POPULATIONS_SEALED.json"
    _require(seal_path.is_file(), "population seal is required")
    seal = _read_json(seal_path)
    hashes = seal.get("hashes")
    _require(isinstance(hashes, dict), "population seal hashes are invalid")
    for relative, digest in hashes.items():
        path = run_dir / str(relative)
        _require(path.is_file() and _file_sha256(path) == digest, f"sealed population hash changed: {relative}")
    return seal


def _simplex_rows(array: np.ndarray, rows: int, name: str) -> np.ndarray:
    value = np.asarray(array)
    _require(value.shape == (rows, 784), f"{name} must be [{rows},784]")
    _require(
        value.dtype.kind == "f"
        and np.isfinite(value).all()
        and np.all(value >= 0.0)
        and float(np.max(np.abs(np.sum(value, axis=1) - 1.0))) <= 2e-12,
        f"{name} is not a healthy simplex population",
    )
    return value.astype(np.float64, copy=False)


def _validate_population_semantics(run_dir: Path) -> dict[str, Any]:
    """Validate population meaning before bytes gain terminal-opening authority."""

    config = _read_json(run_dir / "config.json")
    with np.load(run_dir / "populations.npz", allow_pickle=False) as raw_archive, np.load(
        run_dir / "uint8_populations.npz", allow_pickle=False
    ) as rendered_archive, np.load(run_dir / "start_banks.npz", allow_pickle=False) as starts_archive:
        raw = {name: np.asarray(raw_archive[name]) for name in raw_archive.files}
        rendered = {name: np.asarray(rendered_archive[name]) for name in rendered_archive.files}
        starts = {name: np.asarray(starts_archive[name]) for name in starts_archive.files}
    if config.get("schema") == VERSION + "-smoke":
        labels = np.asarray(raw.get("requested_labels"), dtype=np.int64)
        _require(np.array_equal(labels, np.asarray([0, 1], dtype=np.int64)), "smoke labels changed")
        for name in ("null_prior", "learned_prior", "oracle"):
            state = _simplex_rows(raw[name], 2, name)
            _require(
                name in rendered and np.array_equal(rendered[name], _rasterize_rows(state)),
                f"smoke raster changed: {name}",
            )
        _require(
            np.array_equal(rendered["requested_labels"], labels)
            and np.array_equal(rendered["sample_ids"], raw["sample_ids"])
            and np.array_equal(starts["requested_labels"], labels)
            and np.array_equal(starts["sample_ids"], raw["sample_ids"]),
            "smoke population identities changed",
        )
        return {"scope": "smoke", "prior_count": 2, "passed": 1}

    scope = str(config.get("run_scope", "full"))
    _require(scope in {"objective_pilot", "full"}, "population run scope changed")
    if scope == "objective_pilot":
        counts = {"prior": 20, "forward": 20, "oracle": 10}
        roles = {
            "prior": "pilot_prior",
            "forward": "pilot_forward_terminal",
            "oracle": "pilot_oracle",
        }
    else:
        counts = {"prior": 160, "forward": 40, "oracle": 10}
        roles = {
            "prior": "prior_evaluation",
            "forward": "forward_terminal",
            "oracle": "oracle_controls",
        }
    rows_by_population = {
        "null_prior": counts["prior"],
        "learned_prior": counts["prior"],
        "null_forward_terminal": counts["forward"],
        "learned_forward_terminal": counts["forward"],
        "forward_targets": counts["forward"],
        "null_oracle": counts["oracle"],
        "oracle": counts["oracle"],
        "oracle_targets": counts["oracle"],
    }
    for name, count in rows_by_population.items():
        state = _simplex_rows(raw[name], count, name)
        demixed_name = name + "_demixed"
        expected_demixed = np.stack([core.demix_unit_masses(row) for row in state])
        _require(
            demixed_name in raw and np.allclose(raw[demixed_name], expected_demixed, rtol=0, atol=2e-18),
            f"saved demixed masses changed: {name}",
        )
        _require(
            name in rendered
            and np.array_equal(rendered[name], _rasterize_rows(expected_demixed)),
            f"fixed population raster changed: {name}",
        )
    for prefix in ("prior", "forward", "oracle"):
        count = counts[prefix]
        labels_name = prefix + "_requested_labels"
        ids_name = prefix + "_path_ids"
        samples_name = prefix + "_sample_ids"
        labels = np.asarray(raw[labels_name], dtype=np.int64)
        per_class = count // 10
        role = FROZEN_CONFIG["path_ids"][roles[prefix]]
        expected_ids = np.arange(role["start"], role["stop_exclusive"], dtype=np.int64)
        _require(
            np.array_equal(np.bincount(labels, minlength=10), np.full(10, per_class))
            and np.array_equal(raw[ids_name], expected_ids)
            and raw[samples_name].shape == (count,)
            and len(set(str(value) for value in raw[samples_name])) == count,
            f"{scope} {prefix} identities changed",
        )
        for name in (labels_name, ids_name, samples_name):
            _require(
                name in rendered and np.array_equal(rendered[name], raw[name]),
                f"rendered identity changed: {name}",
            )
    anchors = np.asarray(raw["prior_completed_steps"], dtype=np.int64)
    expected_anchors = np.asarray([0, 32, 64, 96, 128], dtype=np.int64)
    _require(np.array_equal(anchors, expected_anchors), "prior anchors changed")
    for name in ("null_prior_trajectories", "learned_prior_trajectories"):
        trajectory = np.asarray(raw[name])
        _require(
            trajectory.shape == (counts["prior"], 5, 784)
            and np.isfinite(trajectory).all()
            and np.all(trajectory >= 0.0)
            and float(np.max(np.abs(np.sum(trajectory, axis=2) - 1.0))) <= 2e-12,
            f"prior trajectory changed: {name}",
        )
    _require(
        np.array_equal(starts["prior_path_ids"], raw["prior_path_ids"])
        and np.array_equal(starts["prior_requested_labels"], raw["prior_requested_labels"])
        and np.array_equal(starts["prior_sample_ids"], raw["prior_sample_ids"])
        and starts["prior_starts"].shape == (counts["prior"], 784)
        and np.array_equal(raw["null_prior_trajectories"][:, 0], starts["prior_starts"])
        and np.array_equal(raw["learned_prior_trajectories"][:, 0], starts["prior_starts"]),
        "prior start bank or paired start changed",
    )
    _simplex_rows(starts["prior_starts"], counts["prior"], "prior_starts")
    if config.get("schema") == VERSION:
        _validate_prior_start_authority(run_dir, starts, raw)
        _validate_population_stages(run_dir, raw, starts, counts)
    _require(
        np.array_equal(raw["null_prior_trajectories"][:, -1], raw["null_prior"])
        and np.array_equal(
            raw["learned_prior_trajectories"][:, -1], raw["learned_prior"]
        ),
        "prior trajectory endpoints changed",
    )
    with (run_dir / "telemetry.csv").open("r", encoding="utf-8", newline="") as handle:
        telemetry = list(csv.DictReader(handle))
    expected_telemetry = {
        (population, str(quartile))
        for population in (
            "null-prior",
            "learned-prior",
            "null-forward-terminal",
            "learned-forward-terminal",
            "oracle",
        )
        for quartile in range(4)
    }
    paths_by_population = {
        "null-prior": counts["prior"],
        "learned-prior": counts["prior"],
        "null-forward-terminal": counts["forward"],
        "learned-forward-terminal": counts["forward"],
        "oracle": counts["oracle"],
    }
    strict_telemetry = config.get("schema") == VERSION
    _require(
        len(telemetry) == 20
        and {(str(row.get("population")), str(row.get("time_quarter"))) for row in telemetry}
        == expected_telemetry
        and all(
            row.get("controller_rms") not in {None, ""}
            and math.isfinite(float(row["controller_rms"]))
            and float(row["controller_rms"]) >= 0.0
            and (
                not strict_telemetry
                or (
                    int(row.get("score_count", -1))
                    == paths_by_population[str(row["population"])] * 175_616
                    and int(row.get("finite", 0)) == 1
                    and int(row.get("nonnegative", 0)) == 1
                    and int(row.get("microsteps", 0)) == 2
                    and math.isfinite(float(row.get("maximum_mass_error", "nan")))
                    and float(row["maximum_mass_error"]) <= 2e-12
                    and math.isfinite(
                        float(row.get("maximum_pair_total_error", "nan"))
                    )
                    and float(row["maximum_pair_total_error"]) <= 2e-12
                    and 0 <= int(row.get("exact_facet_count", -1))
                    <= paths_by_population[str(row["population"])] * 784
                )
            )
            for row in telemetry
        ),
        "population telemetry inventory or values changed",
    )
    return {
        "scope": scope,
        "counts": counts,
        "anchors": expected_anchors.tolist(),
        "passed": 1,
    }


def seal_populations(run_dir: Path) -> dict[str, Any]:
    """Seal model and generated populations before test evidence or review opens."""

    run_dir = Path(run_dir).resolve()
    if (run_dir / "POPULATIONS_SEALED.json").is_file():
        return _population_seal(run_dir)
    config = _read_json(run_dir / "config.json")
    population_files = list(POPULATION_FILES)
    if config.get("schema") == VERSION:
        population_files.extend(
            ("prior_start_authority.npz", "prior_start_authority.json")
        )
        population_files.extend(
            f"population_stages/{name}.npz" for name in POPULATION_STAGE_NAMES
        )
    missing = [relative for relative in population_files if not (run_dir / relative).is_file()]
    _require(not missing, f"cannot seal populations; missing {missing}")
    semantics = _validate_population_semantics(run_dir)
    hashes = {relative: _file_sha256(run_dir / relative) for relative in population_files}
    seal = {
        "schema": VERSION + "-population-seal",
        "sealed": 1,
        "sealed_at": _utc_now(),
        "hashes": hashes,
        "semantics": semantics,
    }
    _write_json(run_dir / "POPULATIONS_SEALED.json", seal)
    _status(run_dir, "populations_sealed")
    return seal


def open_terminal_evidence(
    run_dir: Path,
    arff: Path | None = None,
) -> tuple[np.ndarray, np.ndarray] | dict[str, Any]:
    """Open terminal-test rows only after population sealing.

    Synthetic smoke runs write a marker instead of touching MNIST.
    """

    run_dir = Path(run_dir).resolve()
    seal = _population_seal(run_dir)
    config = _read_json(run_dir / "config.json")
    event_path = run_dir / "TERMINAL_EVIDENCE_OPENED.json"
    if config.get("schema") == VERSION + "-smoke":
        event = {
            "schema": VERSION + "-smoke-terminal-evidence",
            "opened_after_population_seal": 1,
            "population_seal_sha256": _file_sha256(run_dir / "POPULATIONS_SEALED.json"),
            "opened_at": _utc_now(),
            "synthetic": 1,
        }
        _write_json(event_path, event)
        return event
    _require(arff is not None, "production terminal evidence requires the ARFF path")
    bindings = _read_json(run_dir / "source_bindings.json")
    scientific = dict(config)
    authority = scientific.pop("execution_authority", None)
    repository = Path(bindings["repository_root"])
    _require(
        _semantic_sha256(scientific) == _semantic_sha256(FROZEN_CONFIG)
        and isinstance(authority, dict)
        and authority.get("whole_run_restart_only") == 1
        and bindings.get("config_sha256") == _semantic_sha256(config)
        and bindings.get("source_files") == _source_hashes(repository),
        "terminal firewall config/source authority changed",
    )
    evaluator_checkpoint = Path(bindings["ddpm_run_dir"]) / "evaluator/selected_checkpoint.pt"
    evaluator_selection = Path(bindings["ddpm_run_dir"]) / "evaluator/selection.json"
    contextual_metrics = Path(bindings["ddpm_run_dir"]) / "evaluation/metrics.json"
    ddpm_manifest = Path(bindings["ddpm_run_dir"]) / "artifact_manifest.json"
    ddpm_status = Path(bindings["ddpm_run_dir"]) / "status.json"
    _require(
        evaluator_checkpoint.is_file()
        and evaluator_selection.is_file()
        and contextual_metrics.is_file()
        and ddpm_manifest.is_file()
        and ddpm_status.is_file()
        and _file_sha256(evaluator_checkpoint) == bindings["evaluator_checkpoint_sha256"]
        and _file_sha256(evaluator_selection) == bindings["evaluator_selection_sha256"]
        and _file_sha256(contextual_metrics) == bindings["ddpm_metrics_sha256"]
        and _file_sha256(ddpm_manifest) == bindings["ddpm_manifest_sha256"]
        and _read_json(ddpm_status).get("state") == bindings["ddpm_status"] == "complete"
        and bindings["evaluator_checkpoint_sha256"]
        == ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256
        and bindings["evaluator_selection_sha256"]
        == ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256
        and bindings["ddpm_metrics_sha256"] == ACCEPTED_DDPM_METRICS_SHA256
        and bindings["ddpm_manifest_sha256"] == ACCEPTED_DDPM_MANIFEST_SHA256,
        "terminal firewall evaluator/context binding changed",
    )
    arff = Path(arff).resolve()
    _require(str(arff) == bindings["arff"] and _file_sha256(arff) == bindings["arff_sha256"], "terminal ARFF binding changed")
    event = {
        "schema": VERSION + "-terminal-open-event",
        "opened_after_population_seal": 1,
        "population_seal_sha256": _file_sha256(run_dir / "POPULATIONS_SEALED.json"),
        "sealed_hashes": seal["hashes"],
        "opened_at": _utc_now(),
        "slice": config["data"]["terminal_test"],
    }
    _write_json(event_path, event)
    return load_test_mnist_terminal(arff)


def _review_population(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = _read_json(run_dir / "config.json")
    with np.load(run_dir / "uint8_populations.npz", allow_pickle=False) as archive:
        learned = np.asarray(archive["learned_prior"])
        null = np.asarray(archive["null_prior"])
        label_name = (
            "requested_labels"
            if config.get("schema") == VERSION + "-smoke"
            else "prior_requested_labels"
        )
        labels = np.asarray(archive[label_name], dtype=np.int64)
    _require(learned.shape == null.shape and learned.shape[0] == labels.size, "review population alignment changed")
    if config.get("schema") == VERSION + "-smoke":
        indices = np.arange(labels.size, dtype=np.int64)
    else:
        per_class = int(config["populations"]["review_per_row_per_class"])
        indices = np.concatenate(
            [np.flatnonzero(labels == digit)[:per_class] for digit in range(10)]
        ).astype(np.int64)
        _require(indices.size == 10 * per_class, "review population is not class complete")
    images = np.concatenate((learned[indices], null[indices]))
    requested = np.concatenate((labels[indices], labels[indices]))
    ids = np.asarray(
        [*(f"learned-prior-{int(index):03d}" for index in indices), *(f"null-prior-{int(index):03d}" for index in indices)],
        dtype=np.str_,
    )
    return images, requested, ids


def create_review_bundle(run_dir: Path) -> dict[str, Any]:
    """Create the fixed blinded learned/null review only after population sealing."""

    run_dir = Path(run_dir).resolve()
    _population_seal(run_dir)
    images, labels, sample_ids = _review_population(run_dir)
    config = _read_json(run_dir / "config.json")
    seed = int(config.get("seeds", {}).get("review", 26_140_005))
    result = write_blinded_review_bundle(
        run_dir / "review", images, labels, sample_ids, seed=seed, columns=8
    )
    key = _read_json(run_dir / "review/review_key.json")
    for entry in key["entries"]:
        source = str(entry["source_sample_id"])
        entry["population"] = "learned-prior" if source.startswith("learned-") else "null-prior"
    _write_json(run_dir / "review/review_key.json", key)
    ready = {
        "schema": VERSION + "-review-ready",
        "created_after_population_seal": 1,
        "sample_count": int(images.shape[0]),
        "learned_count": int(sum(str(value).startswith("learned-") for value in sample_ids)),
        "null_count": int(sum(str(value).startswith("null-") for value in sample_ids)),
        "review_key_sha256": _file_sha256(run_dir / "review/review_key.json"),
    }
    _write_json(run_dir / "review/READY.json", ready)
    return ready | {"template": str(result["template"]), "contact_sheet": str(result["contact_sheet"])}


def _validate_review_bundle(run_dir: Path) -> dict[str, Any]:
    images, labels, sample_ids = _review_population(run_dir)
    config = _read_json(run_dir / "config.json")
    seed = int(config.get("seeds", {}).get("review", 26_140_005))
    order = np.random.default_rng(seed).permutation(len(images))
    key = _read_json(run_dir / "review/review_key.json")
    expected_entries = [
        {
            "review_order": index,
            "sample_id": f"blind-{index:03d}",
            "source_sample_id": str(sample_ids[source]),
            "requested_label": int(labels[source]),
            "population": (
                "learned-prior"
                if str(sample_ids[source]).startswith("learned-")
                else "null-prior"
            ),
        }
        for index, source in enumerate(order)
    ]
    _require(
        key.get("schema") == "mnist-blinded-review-v1"
        and int(key.get("seed", -1)) == seed
        and key.get("entries") == expected_entries,
        "blinded review membership, seed, or population mapping changed",
    )
    samples = sorted((run_dir / "review/samples").glob("sample-*.png"))
    _require(len(samples) == len(images), "blinded review sample inventory changed")
    for index, path in enumerate(samples):
        with Image.open(path) as opened:
            observed = np.asarray(opened.convert("L"), dtype=np.uint8)
        _require(
            np.array_equal(observed, images[order[index]]),
            f"blinded review sample changed: {index}",
        )
    with (run_dir / "review/human_review_template.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    _require(
        len(rows) == len(images)
        and all(
            row == {
                "review_order": str(index),
                "sample_id": f"blind-{index:03d}",
                "assigned_label": "",
                "notes": "",
            }
            for index, row in enumerate(rows)
        ),
        "blinded review template changed",
    )
    ready = _read_json(run_dir / "review/READY.json")
    _require(
        ready.get("created_after_population_seal") == 1
        and int(ready.get("sample_count", -1)) == len(images)
        and ready.get("review_key_sha256")
        == _file_sha256(run_dir / "review/review_key.json"),
        "review-ready binding changed",
    )
    return {"passed": 1, "sample_count": len(images), "seed": seed}


def _smoke_report(run_dir: Path) -> str:
    result = _read_json(run_dir / "smoke_result.json")
    return (
        "# Eulerian Jacobi DDPM lifecycle smoke\n\n"
        "Research mode: engineering control. This synthetic CPU run makes no scientific claim.\n\n"
        f"Core result: `{result}`. Populations were sealed before synthetic terminal "
        "evidence and the blinded-review key were created.\n\n"
        "The artifact tree exercises whole-run restart, fixed rendering, population "
        "sealing, review creation, hashing, and read-only verification.\n"
    )


def finalize_and_verify(run_dir: Path) -> dict[str, Any]:
    """Close a smoke or production machine run and verify its compact tree."""

    run_dir = Path(run_dir).resolve()
    _population_seal(run_dir)
    _require((run_dir / "TERMINAL_EVIDENCE_OPENED.json").is_file(), "terminal evidence was not opened")
    _require((run_dir / "review/READY.json").is_file(), "review bundle is missing")
    config = _read_json(run_dir / "config.json")
    if config.get("schema") == VERSION + "-smoke":
        _write_text(run_dir / "REPORT.md", _smoke_report(run_dir))
        _status(run_dir, "complete")
        _seal_manifest(run_dir)
    else:
        started = time.perf_counter()
        _write_text(run_dir / "REPORT.md", _production_report(run_dir, outcome=None))
        _status(run_dir, "awaiting_human_review")
        _seal_manifest(run_dir)
        _charge(run_dir, "terminal_finalize_awaiting_review", started)
        _resource_check(run_dir, preserve_terminal_reserve=False)
        _write_text(run_dir / "REPORT.md", _production_report(run_dir, outcome=None))
        _seal_manifest(run_dir)
    return verify_run(run_dir)


def verify_run(run_dir: Path) -> dict[str, Any]:
    """Read-only integrity verification for a terminal artifact tree."""

    run_dir = Path(run_dir).resolve()
    before = _directory_digest(run_dir)
    manifest = _verify_manifest(run_dir)
    config = _read_json(run_dir / "config.json")
    status = _read_json(run_dir / "status.json")
    prepopulation_states = {
        "pilot_negative_stop_before_scale",
        "pilot_prior_negative_stop_before_scale",
        "repair_reverse_composition_before_judging_learner",
        "pilot_invalid_health_repair_before_interpretation",
        "resource_stopped",
        "failed",
    }
    if status["state"] in prepopulation_states:
        scientific = dict(config)
        authority = scientific.pop("execution_authority", None)
        _require(
            _semantic_sha256(scientific) == _semantic_sha256(FROZEN_CONFIG)
            and isinstance(authority, dict)
            and authority.get("whole_run_restart_only") == 1,
            "stopped-run config or authority changed",
        )
        bindings = _read_json(run_dir / "source_bindings.json")
        _require(
            bindings["source_files"] == _source_hashes(Path(bindings["repository_root"]))
            and bindings["config_sha256"] == _semantic_sha256(config),
            "stopped-run source/config binding changed",
        )
        if status["state"] == "failed":
            failure = _read_json(run_dir / "failure.json")
            _require(
                isinstance(failure.get("error_type"), str)
                and isinstance(failure.get("message"), str),
                "failure record changed",
            )
            if (run_dir / "POPULATIONS_SEALED.json").is_file():
                _population_seal(run_dir)
                _validate_population_semantics(run_dir)
        elif status["state"] == "resource_stopped":
            stop = _read_json(run_dir / "resource_stop.json")
            if (run_dir / "objective_pilot_admission.json").is_file():
                admission = _read_json(run_dir / "objective_pilot_admission.json")
                pilot_result = validate_objective_pilot(
                    _embedded_pilot_dir(run_dir, admission)
                )
                _require(
                    stop.get("scientific_result") == pilot_result["route"]
                    and admission["tree_digest"] == pilot_result["tree_digest"],
                    "resource-stop pilot result changed",
                )
            else:
                _require(stop.get("scientific_result") is None, "resource-stop claim changed")
        else:
            admission = _read_json(run_dir / "objective_pilot_admission.json")
            pilot_result = validate_objective_pilot(
                _embedded_pilot_dir(run_dir, admission)
            )
            _require(
                pilot_result["route"] == status["state"]
                and pilot_result["full_scale_admitted"] == 0
                and pilot_result["tree_digest"] == admission["tree_digest"],
                "objective-pilot stop route changed",
            )
        _require(before == _directory_digest(run_dir), "verification mutated the stopped run")
        return {
            "passed": 1,
            "route": status["state"],
            "artifact_count": int(manifest["artifact_count"]),
            "tree_digest": before,
        }
    _population_seal(run_dir)
    _validate_population_semantics(run_dir)
    _require((run_dir / "TERMINAL_EVIDENCE_OPENED.json").is_file(), "terminal evidence marker is missing")
    terminal = _read_json(run_dir / "TERMINAL_EVIDENCE_OPENED.json")
    _require(terminal.get("opened_after_population_seal") == 1, "terminal firewall changed")
    ready = _read_json(run_dir / "review/READY.json")
    _require(ready.get("created_after_population_seal") == 1, "review firewall changed")
    _validate_review_bundle(run_dir)
    if config.get("schema") == VERSION + "-smoke":
        _require(status["state"] == "complete", "smoke status changed")
        _require(_read_json(run_dir / "smoke_result.json").get("passed") == 1, "smoke result changed")
    else:
        scientific = dict(config)
        authority = scientific.pop("execution_authority", None)
        _require(_semantic_sha256(scientific) == _semantic_sha256(FROZEN_CONFIG), "frozen config changed")
        _require(
            isinstance(authority, dict)
            and authority.get("whole_run_restart_only") == 1
            and isinstance(authority.get("device"), str)
            and re.fullmatch(r"cuda(?::\d+)?", str(authority["device"])) is not None
            and float(authority.get("maximum_active_seconds", 0.0)) > 0.0
            and float(authority.get("maximum_storage_mib", 0.0)) > 0.0
            and 0.0 < float(authority.get("maximum_cuda_fraction", 0.0)) <= 1.0,
            "execution authority changed",
        )
        _require(status["state"] in {"awaiting_human_review", "complete"}, "production status changed")
        bindings = _read_json(run_dir / "source_bindings.json")
        repository = Path(bindings["repository_root"])
        _require(bindings["source_files"] == _source_hashes(repository), "bound source changed")
        _require(bindings["config_sha256"] == _semantic_sha256(config), "config binding changed")
        _validate_full_decision_semantics(run_dir)
    _require(before == _directory_digest(run_dir), "verification mutated the artifact tree")
    return {
        "passed": 1,
        "artifact_count": int(manifest["artifact_count"]),
        "tree_digest": before,
    }


def _directory_digest(path: Path) -> str:
    return _semantic_sha256(
        [
            (item.relative_to(path).as_posix(), item.stat().st_size, _file_sha256(item))
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
    )


def _preflight_stage(
    run_dir: Path, role: str, function: Callable[[], Any]
) -> tuple[Any, float]:
    if (run_dir / "resource_ledger.json").is_file():
        result = _run_stage(run_dir, role, function)
        return result, _last_stage_seconds(run_dir, role)
    started = time.perf_counter()
    return function(), time.perf_counter() - started


def run_numerical_preflight(run_dir: Path) -> dict[str, Any]:
    """Run the theory/code, certified-kernel, and paired K=128/K=512 admission."""

    run_dir = Path(run_dir).resolve()
    config = _read_json(run_dir / "config.json")
    device = str(_read_json(run_dir / "environment.json")["device"])
    architecture = global_dilated_architecture_contract()
    schedule = core.k128_schedule_contract()
    exposure = core.paired_schedule_exposure_audit(
        np.asarray([0.001, 0.01, 0.1, 0.9], dtype=np.float64)
    )
    reverse_midpoints = [
        core.reverse_midpoint_time(0, 0, index, sample_steps=core.OUTER_STEPS)
        for index in range(2)
    ]
    expected_reverse_midpoints = [
        1.0 - 0.75 / (7 * core.OUTER_STEPS),
        1.0 - 0.25 / (7 * core.OUTER_STEPS),
    ]
    gate_a_checks = {
        "exploratory_mode": int(config["research_mode"] == "exploratory"),
        "global_architecture": int(
            architecture.get("passed") == 1
            and architecture.get("trainable_parameter_count")
            == GLOBAL_DILATED_PARAMETER_COUNT
        ),
        "k128_declared": int(
            schedule.get("outer_steps") == 128
            and schedule.get("reference_outer_steps") == 512
            and schedule.get("scientifically_identical_to_k512") == 0
        ),
        "nominal_exposure_only": int(
            exposure.get("passed") == 1
            and exposure.get("same_nominal_cumulative_exposure") == 1
            and exposure.get("same_finite_split_chain_law") == 0
        ),
        "fixed_model": int(
            config["model"]["class"] == "GlobalDilatedZeroBaselinePredictor"
            and config["model"]["parameter_count"] == 34_974
            and config["model"]["architecture_fallback"] is None
        ),
        "fixed_raster": int(
            config["rasterization"]["scale"] == 25_471 / 255
            and config["rasterization"]["lambda_mix"] == 0.35
        ),
        "canonical_reverse_time_order": int(
            config["chain"]["reverse_microstep_execution_quantiles"]
            == [0.75, 0.25]
            and all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
                for actual, expected in zip(
                    reverse_midpoints, expected_reverse_midpoints, strict=True
                )
            )
        ),
        "path_ids": int(_read_json(run_dir / "path_id_audit.json")["passed"] == 1),
    }
    theory = {
        "schema": VERSION + "-theory-code-identity",
        "checks": gate_a_checks,
        "architecture": architecture,
        "schedule": schedule,
        "reverse_midpoint_times_at_outer0_phase0": reverse_midpoints,
        "exposure_audit": exposure,
        "passed": int(all(gate_a_checks.values())),
    }
    _write_json(run_dir / "theory_code_identity.json", theory)
    _require(theory["passed"] == 1, "Gate A theory/code identity failed")

    kernel, kernel_seconds = _preflight_stage(
        run_dir,
        "fast_vs_certified_kernel_audit",
        lambda: core.fast_vs_certified_audit(
            transition_count=4_096, seed=0xB2000, device=device
        ),
    )
    kernel = dict(kernel) | {"elapsed_seconds": kernel_seconds}
    _write_json(run_dir / "kernel_audit.json", kernel)
    _require(
        kernel.get("passed") == 1
        and (
            not (run_dir / "resource_ledger.json").is_file()
            or str(kernel.get("device")) == device
        ),
        "Gate B fast-versus-certified kernel audit failed or ran on the wrong device",
    )

    paired, paired_seconds = _preflight_stage(
        run_dir,
        "paired_k128_k512_law_oracle_audit",
        lambda: core.paired_k128_k512_oracle_audit(
            path_count=1,
            grid_size=28,
            seed=0xB2100,
            device=device,
            path_ids=_role_path_ids("preflight_k128_k512"),
            progress_callback=_stage_progress_callback(run_dir),
        ),
    )
    paired = dict(paired) | {"elapsed_seconds": paired_seconds}
    _write_json(run_dir / "k128_k512_audit.json", paired)
    _require(
        paired.get("passed") == 1
        and paired.get("admission_capable") == 1
        and paired.get("backend") == "real-28x28-fast-jacobi-and-oracle-controller"
        and paired.get("finite_chain_identity_claimed") == 0
        and paired.get("paired_initial_states") == 1
        and paired.get("aligned_transition_randomness_coupled") == 1
        and paired.get("full_path_common_random_numbers_claimed") == 0
        and paired.get("path_ids") == [0xB2100]
        and paired.get("path_ids_sha256")
        == FROZEN_CONFIG["path_ids"]["preflight_k128_k512"]["sha256"]
        and math.isfinite(float(paired.get("paired_law_discrepancy", float("nan"))))
        and math.isfinite(float(paired.get("paired_oracle_discrepancy", float("nan")))),
        "Gate B paired K=128/K=512 law-oracle audit failed",
    )
    return {"gate_a_passed": 1, "gate_b_passed": 1, "theory": theory, "kernel": kernel, "paired": paired}


def _review_row_metrics(answers: Path, key_path: Path) -> dict[str, Any]:
    key = _read_json(key_path)
    entries = {str(row["sample_id"]): row for row in key["entries"]}
    counts = {
        name: {"sample_count": 0, "recognizable_count": 0, "requested_label_count": 0}
        for name in ("learned-prior", "null-prior")
    }
    with Path(answers).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            entry = entries[str(row["sample_id"]).strip()]
            population = str(entry["population"])
            assignment = str(row["assigned_label"]).strip().lower()
            record = counts[population]
            record["sample_count"] += 1
            record["recognizable_count"] += int(assignment.isdigit())
            record["requested_label_count"] += int(assignment == str(entry["requested_label"]))
    for record in counts.values():
        count = int(record["sample_count"])
        _require(count > 0, "human review row is empty")
        record["recognizability"] = record["recognizable_count"] / count
        record["requested_label_agreement"] = record["requested_label_count"] / count
    return counts


def _outcome(run_dir: Path, human_rows: Mapping[str, Any]) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    metrics = _read_json(run_dir / "metrics.json")
    thresholds = config["diagnostics"]
    learned = metrics["prior"]["learned"]
    null = metrics["prior"]["null"]
    learned_human = human_rows["learned-prior"]
    null_human = human_rows["null-prior"]
    machine_learned = float(learned["classifier"]["requested_label_accuracy"])
    machine_null = float(null["classifier"]["requested_label_accuracy"])
    unique = int(learned["duplicates"]["unique_count"])
    diversity = float(learned["diversity"]["aggregate_median_ratio"])
    gates = metrics["gates"]
    feasibility = int(
        int(gates["gate_a_passed"]) == 1
        and int(gates["gate_b_passed"]) == 1
        and int(gates["gate_c_passed"]) == 1
        and int(learned_human["recognizable_count"])
        >= int(thresholds["human_recognizable_count"])
        and int(learned_human["requested_label_count"])
        >= int(thresholds["human_requested_label_count"])
        and machine_learned >= float(thresholds["machine_requested_label_accuracy"])
        and float(learned_human["requested_label_agreement"])
        > float(null_human["requested_label_agreement"])
        and machine_learned > machine_null
        and unique >= int(thresholds["minimum_unique_learned_endpoints"])
        and diversity >= float(thresholds["minimum_diversity_ratio"])
    )
    forward = metrics["forward_terminal"]
    forward_improves = float(forward["mean_learned_minus_null_l1"] or 0.0) < 0.0
    if int(gates["gate_a_passed"]) != 1 or int(gates["gate_b_passed"]) != 1:
        route = "repair_implementation_or_numerical_backend"
        action = (
            "repair the localized theory/code or numerical admission defect and rerun "
            "the unchanged experiment; do not reinterpret learner evidence"
        )
    elif int(gates["gate_c_passed"]) != 1:
        route = "repair_reverse_composition_before_judging_learner"
        action = (
            "repair the oracle/reference reverse composition and rerun it before "
            "judging or changing the learner"
        )
    elif feasibility:
        route = "freeze_v0_and_plan_one_fresh_seed"
        action = "freeze this exploratory result and plan one fresh model-seed replication; do not auto-launch"
    elif forward_improves:
        route = "stop_v0_review_terminal_law"
        action = "stop v0 and review terminal mixing/exposure as a major strategy decision; do not tune on opened outputs"
    else:
        route = "stop_v0_strategy_review"
        action = "stop this v0 learner and compare one materially different fixed-grid score/controller or stop the hypothesis"
    return {
        "schema": VERSION + "-outcome",
        "research_mode": "exploratory",
        "gate_a_passed": int(gates["gate_a_passed"]),
        "gate_b_passed": int(gates["gate_b_passed"]),
        "gate_c_passed": int(gates["gate_c_passed"]),
        "diagnostic_e_passed": feasibility,
        "learned_human": learned_human,
        "null_human": null_human,
        "learned_machine_accuracy": machine_learned,
        "null_machine_accuracy": machine_null,
        "learned_unique_count": unique,
        "learned_diversity_ratio": diversity,
        "route": route,
        "next_action": action,
        "automatic_launches": 0,
        "claim_scope": config["claim_scope"],
        "negative_scope": (
            "failure applies only to this frozen architecture, target, K=128 "
            "controller, paths, renderer, evaluator, and exploratory diagnostics"
        ),
    }


def _validate_training_selection(run_dir: Path, *, updates: int) -> dict[str, Any]:
    checkpoint = run_dir / "selected_checkpoint.pt"
    selection = _read_json(run_dir / "training_selection.json")
    _require(
        selection.get("checkpoint_sha256") == _file_sha256(checkpoint)
        and int(selection.get("completed_updates", -1)) == updates,
        "training selection binding changed",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    expected = core.make_model().state_dict()
    _require(
        isinstance(state, dict)
        and set(state) == set(expected)
        and all(
            isinstance(state[name], torch.Tensor)
            and state[name].shape == expected[name].shape
            and bool(torch.isfinite(state[name]).all())
            for name in expected
        )
        and int(payload.get("completed_updates", -1)) == updates
        and int(payload.get("selected_update", 0)) > 0
        and int(payload["selected_update"]) == int(selection["selected_update"])
        and float(payload["selected_validation_normalized_mse"])
        == float(selection["selected_validation_normalized_mse"]),
        "selected training checkpoint changed",
    )
    with (run_dir / "training_history.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        history = list(csv.DictReader(handle))
    eligible = [
        (int(row["update"]), float(row["validation_normalized_mse"]))
        for row in history
        if int(row["update"]) > 0
        and int(row["eligible"]) == 1
        and math.isfinite(float(row["validation_normalized_mse"]))
    ]
    _require(bool(eligible), "training history has no eligible checkpoint")
    chosen = min(eligible, key=lambda row: (row[1], row[0]))
    _require(
        chosen
        == (
            int(selection["selected_update"]),
            float(selection["selected_validation_normalized_mse"]),
        ),
        "training selection is not the earliest finite nonzero normalized-MSE argmin",
    )
    return {"passed": 1, "selected_update": chosen[0]}


def _compare_classifier(saved: Mapping[str, Any], replayed: Mapping[str, Any], name: str) -> None:
    _require(
        float(saved["accuracy"]) == float(replayed["accuracy"])
        and float(saved["requested_label_accuracy"])
        == float(replayed["requested_label_accuracy"])
        and saved["per_class"] == _jsonable(replayed["per_class"])
        and np.array_equal(saved["sample_ids"], replayed["sample_ids"])
        and np.array_equal(saved["requested_labels"], replayed["requested_labels"])
        and np.array_equal(saved["predictions"], replayed["predictions"])
        and math.isclose(
            float(saved["loss"]), float(replayed["loss"]), rel_tol=2e-4, abs_tol=2e-5
        )
        and np.allclose(
            saved["logits"], replayed["logits"], rtol=2e-4, atol=2e-5
        ),
        f"saved classifier metrics changed: {name}",
    )


def _validate_full_decision_semantics(run_dir: Path) -> dict[str, Any]:
    _validate_full_pilot_admission(run_dir)
    _validate_training_selection(run_dir, updates=10_000)
    config = _read_json(run_dir / "config.json")
    bindings = _read_json(run_dir / "source_bindings.json")
    _require(
        bindings.get("evaluator_checkpoint_sha256")
        == ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256
        and bindings.get("evaluator_selection_sha256")
        == ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256
        and bindings.get("ddpm_metrics_sha256") == ACCEPTED_DDPM_METRICS_SHA256
        and bindings.get("ddpm_manifest_sha256") == ACCEPTED_DDPM_MANIFEST_SHA256
        and bindings.get("ddpm_status") == "complete",
        "accepted DDPM evaluator authority changed",
    )
    ledger = _read_json(run_dir / "resource_ledger.json")
    projection = _read_json(run_dir / "preflight_projection.json")
    post_pilot = projection.get("post_pilot_full_admission", {})
    receipt_fields = {
        "checked_after_positive_pilot",
        "approval_id",
        "ledger_event_count_at_check",
        "active_seconds_at_check",
        "storage_bytes_at_check",
        "peak_cuda_fraction_at_check",
        "maximum_active_seconds",
        "maximum_storage_bytes",
        "maximum_cuda_fraction",
        "full_projected_seconds",
        "full_projected_storage_bytes",
        "terminal_reserve_seconds",
        "time_cap_passed",
        "storage_cap_passed",
        "cuda_cap_passed",
        "passed",
    }
    _require(
        isinstance(post_pilot, dict) and set(post_pilot) == receipt_fields,
        "post-pilot resource receipt schema changed",
    )
    receipt_event_count = int(post_pilot["ledger_event_count_at_check"])
    receipt_active = float(post_pilot["active_seconds_at_check"])
    receipt_storage = int(post_pilot["storage_bytes_at_check"])
    receipt_peak_cuda = float(post_pilot["peak_cuda_fraction_at_check"])
    receipt_time_pass = int(
        receipt_active
        + float(post_pilot["full_projected_seconds"])
        + float(post_pilot["terminal_reserve_seconds"])
        < float(post_pilot["maximum_active_seconds"])
    )
    receipt_storage_pass = int(
        receipt_storage + int(post_pilot["full_projected_storage_bytes"])
        < int(post_pilot["maximum_storage_bytes"])
    )
    receipt_cuda_pass = int(
        receipt_peak_cuda < float(post_pilot["maximum_cuda_fraction"])
    )
    receipt_prefix_seconds = math.fsum(
        float(row["seconds"])
        for row in ledger["events"][:receipt_event_count]
    )
    execution = config["execution_authority"]
    _require(
        int(ledger["projected_cache_pair_transitions"]) == 1_756_160_000
        and int(ledger["projected_sampling_transition_work"]) == 607_631_360
        and int(ledger["projected_base_transition_work"]) == 2_363_791_360
        and _semantic_sha256(ledger.get("latest_projection"))
        == _semantic_sha256(projection)
        and ledger.get("approval_id") == execution.get("approval_id")
        and post_pilot.get("approval_id") == execution.get("approval_id")
        and float(ledger["maximum_active_seconds"])
        == float(execution["maximum_active_seconds"])
        and int(ledger["maximum_storage_bytes"])
        == int(float(execution["maximum_storage_mib"]) * 1024**2)
        and float(ledger["maximum_cuda_fraction"])
        == float(execution["maximum_cuda_fraction"])
        and int(post_pilot["checked_after_positive_pilot"]) == 1
        and 0 <= receipt_event_count <= len(ledger["events"])
        and math.isclose(receipt_active, receipt_prefix_seconds, rel_tol=0.0, abs_tol=1e-9)
        and 0.0 <= receipt_active <= float(ledger["active_seconds"])
        and 0 <= receipt_storage <= _directory_bytes(run_dir)
        and 0.0 <= receipt_peak_cuda <= float(ledger["peak_cuda_fraction"])
        and float(post_pilot["maximum_active_seconds"])
        == float(ledger["maximum_active_seconds"])
        and int(post_pilot["maximum_storage_bytes"])
        == int(ledger["maximum_storage_bytes"])
        and float(post_pilot["maximum_cuda_fraction"])
        == float(ledger["maximum_cuda_fraction"])
        and float(post_pilot["full_projected_seconds"])
        == float(projection["full"]["projected_seconds"])
        and int(post_pilot["full_projected_storage_bytes"])
        == int(projection["full_projected_storage_bytes"])
        and float(post_pilot["terminal_reserve_seconds"])
        == float(projection["terminal_reserve_seconds"])
        and int(post_pilot["time_cap_passed"]) == receipt_time_pass == 1
        and int(post_pilot["storage_cap_passed"]) == receipt_storage_pass == 1
        and int(post_pilot["cuda_cap_passed"]) == receipt_cuda_pass == 1
        and int(post_pilot["passed"])
        == int(receipt_time_pass and receipt_storage_pass and receipt_cuda_pass)
        == 1,
        "resource workload authority changed",
    )
    theory = _read_json(run_dir / "theory_code_identity.json")
    kernel = _read_json(run_dir / "kernel_audit.json")
    paired = _read_json(run_dir / "k128_k512_audit.json")
    _require(
        theory.get("passed") == 1
        and kernel.get("passed") == 1
        and str(kernel.get("device")) == config["execution_authority"]["device"]
        and paired.get("passed") == 1
        and paired.get("admission_capable") == 1
        and paired.get("aligned_transition_randomness_coupled") == 1
        and paired.get("full_path_common_random_numbers_claimed") == 0
        and paired.get("path_ids") == [0xB2100]
        and paired.get("path_ids_sha256")
        == FROZEN_CONFIG["path_ids"]["preflight_k128_k512"]["sha256"],
        "numerical admission authority changed",
    )
    seal = _population_seal(run_dir)
    terminal = _read_json(run_dir / "TERMINAL_EVIDENCE_OPENED.json")
    _require(
        terminal.get("population_seal_sha256")
        == _file_sha256(run_dir / "POPULATIONS_SEALED.json")
        and terminal.get("sealed_hashes") == seal["hashes"],
        "terminal opening is not bound to the population seal",
    )
    with np.load(run_dir / "populations.npz", allow_pickle=False) as raw, np.load(
        run_dir / "uint8_populations.npz", allow_pickle=False
    ) as rendered:
        raw_rows = {name: np.asarray(raw[name]) for name in raw.files}
        image_rows = {name: np.asarray(rendered[name]) for name in rendered.files}
    saved = _read_json(run_dir / "metrics.json")
    forward = _paired_endpoint_metrics(
        raw_rows["null_forward_terminal"],
        raw_rows["learned_forward_terminal"],
        raw_rows["forward_targets"],
    )
    oracle = _paired_endpoint_metrics(
        raw_rows["null_oracle"], raw_rows["oracle"], raw_rows["oracle_targets"]
    )
    gate_c = int(
        int(oracle["learned_l1_win_count"]) >= 9
        and float(oracle["mean_learned_l1"]) < float(oracle["mean_null_l1"])
    )
    for key in (
        "path_count",
        "null_l1",
        "learned_l1",
        "learned_l1_win_count",
        "mean_null_l1",
        "mean_learned_l1",
        "mean_learned_minus_null_l1",
        "aggregate_l1_relative_improvement",
        "learned_centered_correlation",
    ):
        _require(
            _semantic_sha256(saved["forward_terminal"][key])
            == _semantic_sha256(forward[key])
            and _semantic_sha256(saved["oracle"][key]) == _semantic_sha256(oracle[key]),
            f"raw endpoint metric changed: {key}",
        )
    _require(
        saved["gates"]
        == {"gate_a_passed": 1, "gate_b_passed": 1, "gate_c_passed": gate_c},
        "saved scientific gates changed",
    )
    expected_render = {
        name: {
            "minimum": int(np.min(values)),
            "maximum": int(np.max(values)),
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
        }
        for name, values in image_rows.items()
        if name in saved["fixed_render"]
    }
    _require(saved["fixed_render"] == expected_render, "fixed-render metrics changed")

    arff = Path(bindings["arff"])
    train_u8, _, validation_u8, validation_y = load_train_validation_mnist(arff)
    test_u8, test_y = load_test_mnist_terminal(arff)
    evaluator = _load_bound_evaluator(run_dir, device="cpu")
    for population in ("learned", "null"):
        key = population + "_prior"
        replayed = compute_generation_metrics(
            evaluator,
            image_rows[key],
            raw_rows["prior_requested_labels"],
            raw_rows["prior_sample_ids"],
            real_reference_images=test_u8,
            real_reference_labels=test_y,
            train_images=train_u8,
            test_images=test_u8,
            batch_size=256,
            device="cpu",
        )
        replayed["exact_reference_match_count"]["validation"] = _exact_match_count(
            image_rows[key], validation_u8
        )
        observed = saved["prior"][population]
        _compare_classifier(observed["classifier"], replayed["classifier"], key)
        _require(
            observed["duplicates"] == _jsonable(replayed["duplicates"])
            and observed["diversity"] == _jsonable(replayed["diversity"])
            and observed["exact_reference_match_count"]
            == replayed["exact_reference_match_count"],
            f"saved discrete generation metrics changed: {key}",
        )
    for population in ("null", "learned"):
        replayed = evaluate_generated_labels(
            evaluator,
            image_rows[f"{population}_forward_terminal"],
            raw_rows["forward_requested_labels"],
            raw_rows["forward_sample_ids"],
            device="cpu",
        )
        _compare_classifier(
            saved["forward_terminal"][f"{population}_classifier"],
            replayed,
            f"{population}-forward-terminal",
        )
    validation_eval = evaluate_image_classifier(
        evaluator,
        validation_u8.astype(np.float32)[:, None] / np.float32(255),
        validation_y,
        batch_size=256,
        device="cpu",
    )
    test_eval = evaluate_image_classifier(
        evaluator,
        test_u8.astype(np.float32)[:, None] / np.float32(255),
        test_y,
        batch_size=256,
        device="cpu",
    )
    _require(
        float(saved["evaluator_health"]["validation_accuracy"])
        == float(validation_eval["accuracy"])
        and math.isclose(
            float(saved["evaluator_health"]["validation_loss"]),
            float(validation_eval["loss"]),
            rel_tol=2e-4,
            abs_tol=2e-5,
        )
        and float(saved["evaluator_health"]["test_accuracy"])
        == float(test_eval["accuracy"])
        and math.isclose(
            float(saved["evaluator_health"]["test_loss"]),
            float(test_eval["loss"]),
            rel_tol=2e-4,
            abs_tol=2e-5,
        ),
        "bound evaluator validation metrics changed",
    )
    status = _read_json(run_dir / "status.json")
    if status["state"] == "complete":
        answers = run_dir / "review/human_review_answers.csv"
        _require(answers.is_file(), "completed review answers are missing")
        rows = _review_row_metrics(answers, run_dir / "review/review_key.json")
        human = _read_json(run_dir / "review/human_review.json")
        _require(human.get("by_population") == rows, "saved human-review scores changed")
        _require(
            _semantic_sha256(_read_json(run_dir / "outcome.json"))
            == _semantic_sha256(_outcome(run_dir, rows)),
            "saved outcome routing changed",
        )
    return {"passed": 1, "gate_c_passed": gate_c}


def _production_report(run_dir: Path, outcome: Mapping[str, Any] | None) -> str:
    config = _read_json(run_dir / "config.json")
    ledger = _read_json(run_dir / "resource_ledger.json")
    bindings = _read_json(run_dir / "source_bindings.json")
    selection = _read_json(run_dir / "training_selection.json")
    command = (run_dir / "command.txt").read_text(encoding="utf-8").strip()
    metrics = _read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").is_file() else None
    pilot_admission = _read_json(run_dir / "objective_pilot_admission.json")
    state = "complete" if outcome is not None else "awaiting_human_review"
    lines = [
        "# Eulerian Jacobi DDPM-v0 exploratory run",
        "",
        f"Status: `{state}`. Primary mode: exploratory.",
        "",
        f"Decision: {config['decision']}",
        "",
        f"Source revision: `{bindings['git_revision']}`; bound scientific source files: "
        f"{len(bindings['source_files'])}; config SHA-256: `{bindings['config_sha256']}`.",
        "",
        f"Exact command: `{command}`.",
        "",
        "Selected checkpoint: `selected_checkpoint.pt`, update "
        f"{selection['selected_update']}, validation normalized MSE "
        f"{selection['selected_validation_normalized_mse']}, SHA-256 "
        f"`{selection['checkpoint_sha256']}`.",
        "",
        "The production chain is the declared finite K=128 Jacobi split with a "
        "boundary-preserving approximate reverse controller; exact finite-chain "
        "reverse-kernel identity is not claimed.",
        "",
        f"Frozen model/records/rasterization: `{config['model']}` / `{config['records']}` / `{config['rasterization']}`.",
        "",
        f"Resource ledger: `{ledger}`.",
        "",
        "Full-scale pilot admission, including the paired prior task-signal diagnostic: "
        f"`{pilot_admission}`.",
        "",
        "All null, learned, forward-terminal, and oracle outputs, including failures, are "
        "stored in atomic `population_stages/`, `populations.npz`, and "
        "`uint8_populations.npz`. The selected model "
        "and populations were sealed before terminal-test scoring or review-key creation.",
        "",
        "Evidence roles are frozen in `data_roles.npz` and `config.json`: development "
        "train/validation paths are disjoint from terminal ARFF rows, while prior, "
        "forward-terminal, and oracle path-ID inventories are disjoint.",
    ]
    if metrics is not None:
        lines.extend(["", f"Machine metrics and gates: `{metrics}`."])
    if outcome is None:
        lines.extend(
            [
                "",
                "Human review is pending. Complete a copy of "
                "`review/human_review_template.csv`; no final scientific route is assigned yet.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"Exploratory outcome: `{outcome}`.",
                "",
                f"Required next action: {outcome['next_action']}",
            ]
        )
    lines.extend(
        [
            "",
            "Claim boundary: this run supports only the frozen exploratory scope. A "
            "negative does not establish absence of useful local signal or universal "
            "failure of fixed-grid Eulerian generators.",
            "",
            "Deliberate omissions: no protected confirmatory replication, no exact "
            "discrete reverse-kernel claim, no continuum-limit claim, and no automatic "
            "post-hoc learner tuning. The accepted frozen evaluator is secondary to "
            "the blinded human review and does not meet its original 0.97 diagnostic.",
        ]
    )
    return "\n".join(lines) + "\n"


def _production_handoff(run_dir: Path, outcome: Mapping[str, Any] | None) -> str:
    state = "pending manual review" if outcome is None else str(outcome["route"])
    action = (
        "complete the blinded review and record it"
        if outcome is None
        else str(outcome["next_action"])
    )
    return (
        "# Eulerian Jacobi DDPM-v0: run handoff\n\n"
        "Primary mode: exploratory. Nearest objective-bearing milestone: prior-start "
        "class-conditional MNIST images from the fixed-grid Eulerian chain.\n\n"
        f"Terminal state: **{state}**. Required action: {action}.\n\n"
        "Proxy-only patches since the last objective-bearing experiment: 0.\n\n"
        "Evidence map: `REPORT.md`, `metrics.json`, `populations.npz`, "
        "`uint8_populations.npz`, `telemetry.csv`, `kernel_audit.json`, "
        "`k128_k512_audit.json`, and `preflight_projection.json`.\n\n"
        "The result is exploratory and does not claim an exact discrete reverse "
        "kernel, continuum limit, population confirmation, or broad Eulerian success/failure.\n"
    )


def record_human_review(
    run_dir: Path,
    answers: Path,
    reviewer: str,
    confirm_manual_review: bool,
) -> dict[str, Any]:
    """Record one complete manual review and close the exploratory decision."""

    run_dir = Path(run_dir).resolve()
    answers = Path(answers).resolve()
    status = _read_json(run_dir / "status.json")
    if status["state"] == "complete" and (run_dir / "outcome.json").is_file():
        verify_run(run_dir)
        return _read_json(run_dir / "outcome.json")
    _require(status["state"] == "awaiting_human_review", "run is not awaiting human review")
    ignored: set[str] = set()
    if answers.is_relative_to(run_dir):
        ignored.add(answers.relative_to(run_dir).as_posix())
    _verify_manifest(run_dir, ignored)
    _population_seal(run_dir)
    _validate_population_semantics(run_dir)
    _validate_review_bundle(run_dir)
    (run_dir / "artifact_manifest.json").unlink()
    (run_dir / "SHA256SUMS.txt").unlink()
    started = time.perf_counter()
    try:
        _resource_check(run_dir, preserve_terminal_reserve=False)
        _validate_full_decision_semantics(run_dir)
        human = score_human_review(
            answers,
            run_dir / "review/review_key.json",
            reviewer=reviewer,
            confirm_manual_review=confirm_manual_review,
        )
        rows = _review_row_metrics(answers, run_dir / "review/review_key.json")
        destination = run_dir / "review/human_review_answers.csv"
        if answers != destination:
            _atomic_replace(
                destination, lambda temporary: shutil.copyfile(answers, temporary)
            )
        _write_json(run_dir / "review/human_review.json", human | {"by_population": rows})
        outcome = _outcome(run_dir, rows)
        _write_json(run_dir / "outcome.json", outcome)
        _write_text(run_dir / "REPORT.md", _production_report(run_dir, outcome))
        _write_text(run_dir / "HANDOFF.md", _production_handoff(run_dir, outcome))
        _status(run_dir, "complete")
        _seal_manifest(run_dir)
    except BaseException:
        _charge(run_dir, "terminal_record_human_review", started, failed=True)
        _seal_manifest(run_dir)
        raise
    _charge(run_dir, "terminal_record_human_review", started)
    try:
        _resource_check(run_dir, preserve_terminal_reserve=False)
        _write_text(run_dir / "REPORT.md", _production_report(run_dir, outcome))
    finally:
        _seal_manifest(run_dir)
    verify_run(run_dir)
    return outcome


def _mnist_unit_masses(images: np.ndarray) -> np.ndarray:
    pixels = np.asarray(images)
    _require(
        pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[1:] == (28, 28),
        "MNIST images must be uint8 [N,28,28]",
    )
    flat = pixels.reshape(pixels.shape[0], 784).astype(np.float64)
    totals = flat.sum(axis=1, dtype=np.float64)
    _require(np.all(totals > 0.0), "a selected MNIST image has zero total mass")
    return np.ascontiguousarray(flat / totals[:, None])


def _role_path_ids(role: str) -> np.ndarray:
    row = FROZEN_CONFIG["path_ids"][role]
    return np.arange(row["start"], row["stop_exclusive"], dtype=np.int64)


def _write_prior_start_authority(
    run_dir: Path,
    starts: np.ndarray,
    labels: np.ndarray,
    path_ids: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, Any]:
    """Commit the paired prior population before any sampler can fail."""

    path = run_dir / "prior_start_authority.npz"
    _write_npz(
        path,
        prior_starts=np.asarray(starts, dtype=np.float64),
        prior_requested_labels=np.asarray(labels, dtype=np.int64),
        prior_path_ids=np.asarray(path_ids, dtype=np.int64),
        prior_sample_ids=np.asarray(sample_ids, dtype=np.str_),
    )
    record = {
        "schema": VERSION + "-prior-start-authority",
        "committed_before_sampling": 1,
        "row_count": int(len(path_ids)),
        "npz_sha256": _file_sha256(path),
        "path_ids_sha256": hashlib.sha256(
            np.asarray(path_ids, dtype="<i8").tobytes()
        ).hexdigest(),
        "sample_ids_sha256": _semantic_sha256(np.asarray(sample_ids, dtype=np.str_)),
        "committed_at": _utc_now(),
    }
    _write_json(run_dir / "prior_start_authority.json", record)
    return record


def _validate_prior_start_authority(
    run_dir: Path,
    starts: Mapping[str, np.ndarray],
    raw: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    path = run_dir / "prior_start_authority.npz"
    _require(
        path.is_file() and (run_dir / "prior_start_authority.json").is_file(),
        "prior-start authority is missing",
    )
    record = _read_json(run_dir / "prior_start_authority.json")
    _require(
        record.get("committed_before_sampling") == 1
        and record.get("npz_sha256") == _file_sha256(path),
        "prior-start authority hash changed",
    )
    with np.load(path, allow_pickle=False) as archive:
        authority = {name: np.asarray(archive[name]) for name in archive.files}
    for name in (
        "prior_starts",
        "prior_requested_labels",
        "prior_path_ids",
        "prior_sample_ids",
    ):
        _require(
            name in authority
            and np.array_equal(authority[name], starts[name])
            and (
                name == "prior_starts"
                or np.array_equal(authority[name], raw[name])
            ),
            f"prior-start authority changed: {name}",
        )
    _require(
        int(record.get("row_count", -1)) == len(authority["prior_path_ids"])
        and record.get("path_ids_sha256")
        == hashlib.sha256(authority["prior_path_ids"].astype("<i8").tobytes()).hexdigest()
        and record.get("sample_ids_sha256")
        == _semantic_sha256(authority["prior_sample_ids"]),
        "prior-start authority inventory changed",
    )
    return {"passed": 1, "row_count": int(record["row_count"])}


def _write_population_stage(
    run_dir: Path,
    name: str,
    result: core.SamplingResult,
    labels: np.ndarray,
    path_ids: np.ndarray,
    sample_ids: np.ndarray,
) -> Path:
    """Atomically retain one completed named population before the next stage."""

    _require(name in POPULATION_STAGE_NAMES, f"unknown population stage: {name}")
    anchor_steps = np.asarray(sorted(result.anchors), dtype=np.int64)
    path = Path(run_dir) / "population_stages" / f"{name}.npz"
    _write_npz(
        path,
        final_states=np.asarray(result.final_states, dtype=np.float64),
        starts=np.asarray(result.starts, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        path_ids=np.asarray(path_ids, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        anchor_steps=anchor_steps,
        anchor_states=np.stack(
            [np.asarray(result.anchors[int(step)], dtype=np.float64) for step in anchor_steps],
            axis=1,
        ),
        telemetry_json=np.asarray(
            json.dumps(
                _jsonable(result.telemetry),
                sort_keys=True,
                separators=(",", ":"),
            ),
            dtype=np.str_,
        ),
    )
    return path


def _validate_population_stages(
    run_dir: Path,
    raw: Mapping[str, np.ndarray],
    starts: Mapping[str, np.ndarray],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    identity_prefix = {
        "null_prior": "prior",
        "learned_prior": "prior",
        "null_forward_terminal": "forward",
        "learned_forward_terminal": "forward",
        "null_oracle": "oracle",
        "oracle": "oracle",
    }
    expected_starts = {
        "null_prior": starts["prior_starts"],
        "learned_prior": starts["prior_starts"],
        "null_forward_terminal": starts["forward_terminal_starts"],
        "learned_forward_terminal": starts["forward_terminal_starts"],
        "null_oracle": starts["oracle_starts"],
        "oracle": starts["oracle_starts"],
    }
    expected_counts = {
        "null_prior": counts["prior"],
        "learned_prior": counts["prior"],
        "null_forward_terminal": counts["forward"],
        "learned_forward_terminal": counts["forward"],
        "null_oracle": counts["oracle"],
        "oracle": counts["oracle"],
    }
    expected_fields = {
        "final_states",
        "starts",
        "labels",
        "path_ids",
        "sample_ids",
        "anchor_steps",
        "anchor_states",
        "telemetry_json",
    }
    for name in POPULATION_STAGE_NAMES:
        path = Path(run_dir) / "population_stages" / f"{name}.npz"
        _require(path.is_file(), f"completed population stage is missing: {name}")
        with np.load(path, allow_pickle=False) as archive:
            _require(set(archive.files) == expected_fields, f"population-stage schema changed: {name}")
            stage = {field: np.asarray(archive[field]) for field in archive.files}
        count = expected_counts[name]
        prefix = identity_prefix[name]
        final_states = _simplex_rows(stage["final_states"], count, f"stage {name} final")
        stage_starts = _simplex_rows(stage["starts"], count, f"stage {name} starts")
        _require(
            np.array_equal(final_states, raw[name])
            and np.array_equal(stage_starts, expected_starts[name])
            and np.array_equal(stage["labels"], raw[f"{prefix}_requested_labels"])
            and np.array_equal(stage["path_ids"], raw[f"{prefix}_path_ids"])
            and np.array_equal(stage["sample_ids"], raw[f"{prefix}_sample_ids"]),
            f"completed population stage changed: {name}",
        )
        anchor_steps = np.asarray(stage["anchor_steps"], dtype=np.int64)
        anchor_states = np.asarray(stage["anchor_states"], dtype=np.float64)
        _require(
            np.array_equal(anchor_steps, np.asarray([0, 32, 64, 96, 128], dtype=np.int64))
            and anchor_states.shape == (count, 5, 784)
            and np.array_equal(anchor_states[:, 0], stage_starts)
            and np.array_equal(anchor_states[:, -1], final_states),
            f"completed population-stage anchors changed: {name}",
        )
        if name in {"null_prior", "learned_prior"}:
            _require(
                np.array_equal(anchor_states, raw[f"{name}_trajectories"]),
                f"prior stage trajectory changed: {name}",
            )
        telemetry = json.loads(str(stage["telemetry_json"].item()))
        _require(
            isinstance(telemetry, dict)
            and int(telemetry.get("finite", 0)) == 1
            and int(telemetry.get("nonnegative", 0)) == 1
            and int(telemetry.get("microsteps", 0)) == 2
            and len(telemetry.get("by_time_quarter", [])) == 4,
            f"completed population-stage telemetry changed: {name}",
        )
    return {"passed": 1, "stage_count": len(POPULATION_STAGE_NAMES)}


def _dataset_arrays(prefix: str, dataset: core.ForwardRecordDataset) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_{name}": np.asarray(getattr(dataset, name))
        for name in core.ForwardRecordDataset.__dataclass_fields__
    }


def _dataset_bytes(dataset: core.ForwardRecordDataset) -> int:
    return sum(
        int(np.asarray(getattr(dataset, name)).nbytes)
        for name in core.ForwardRecordDataset.__dataclass_fields__
    )


def _merge_sampling_results(parts: Sequence[core.SamplingResult]) -> core.SamplingResult:
    _require(bool(parts), "at least one sampling cohort is required")
    anchors = sorted(set(parts[0].anchors))
    _require(
        all(sorted(set(part.anchors)) == anchors for part in parts),
        "sampling cohort anchors changed",
    )
    quarterly: list[dict[str, Any]] = []
    for quarter in range(4):
        source = [part.telemetry["by_time_quarter"][quarter] for part in parts]
        _require(
            all(int(row["time_quarter"]) == quarter for row in source),
            "sampling telemetry quarter changed",
        )
        weights = np.asarray([int(row["score_count"]) for row in source], dtype=np.float64)
        _require(bool(np.all(weights > 0.0)), "sampling telemetry count changed")

        def pooled_rms(name: str) -> float:
            values = np.asarray([float(row[name]) for row in source], dtype=np.float64)
            return math.sqrt(float(np.sum(weights * np.square(values))) / float(np.sum(weights)))

        quarterly.append(
            {
                "quarter": quarter,
                "time_quarter": quarter,
                "score_count": int(np.sum(weights)),
                "score_rms": pooled_rms("score_rms"),
                "controller_rms": pooled_rms("controller_rms"),
                "reference_fraction_displacement_rms": pooled_rms(
                    "reference_fraction_displacement_rms"
                ),
                "control_fraction_displacement_rms": pooled_rms(
                    "control_fraction_displacement_rms"
                ),
                "maximum_absolute_q": max(float(row["maximum_absolute_q"]) for row in source),
                "maximum_absolute_logit_increment": max(
                    float(row["maximum_absolute_logit_increment"]) for row in source
                ),
                "maximum_mass_error": max(float(row["maximum_mass_error"]) for row in source),
                "maximum_pair_total_error": max(
                    float(row["maximum_pair_total_error"]) for row in source
                ),
            }
        )
    total_count = sum(int(row["score_count"]) for row in quarterly)
    controller_rms = math.sqrt(
        math.fsum(
            int(row["score_count"]) * float(row["controller_rms"]) ** 2
            for row in quarterly
        )
        / total_count
    )
    return core.SamplingResult(
        starts=np.concatenate([part.starts for part in parts], axis=0),
        final_states=np.concatenate([part.final_states for part in parts], axis=0),
        anchors={
            anchor: np.concatenate([part.anchors[anchor] for part in parts], axis=0)
            for anchor in anchors
        },
        telemetry={
            "controller": str(parts[0].telemetry["controller"]),
            "controller_rms": controller_rms,
            "maximum_absolute_q": max(
                float(part.telemetry["maximum_absolute_q"]) for part in parts
            ),
            "maximum_mass_error": max(
                float(part.telemetry["maximum_mass_error"]) for part in parts
            ),
            "maximum_pair_total_error": max(
                float(part.telemetry["maximum_pair_total_error"]) for part in parts
            ),
            "exact_facet_count": sum(
                int(part.telemetry["exact_facet_count"]) for part in parts
            ),
            "finite": int(all(int(part.telemetry["finite"]) == 1 for part in parts)),
            "nonnegative": int(
                all(int(part.telemetry["nonnegative"]) == 1 for part in parts)
            ),
            "microsteps": int(parts[0].telemetry["microsteps"]),
            "by_time_quarter": quarterly,
        },
    )


def _reverse_cohorts(
    starts: np.ndarray,
    labels: np.ndarray,
    path_ids: np.ndarray,
    *,
    controller: str,
    root_seed: int,
    device: str,
    model: core.EulerianJacobiDDPMModel | None = None,
    oracle_targets: np.ndarray | None = None,
    resource_run_dir: Path | None = None,
) -> core.SamplingResult:
    progress = (
        None if resource_run_dir is None else _stage_progress_callback(resource_run_dir)
    )
    parts: list[core.SamplingResult] = []
    for start in range(0, len(path_ids), 8):
        stop = min(start + 8, len(path_ids))
        parts.append(
            core.reverse_sample(
                starts[start:stop],
                labels[start:stop],
                path_ids[start:stop],
                controller=controller,
                root_seed=int(root_seed),
                model=model,
                oracle_targets=(
                    None if oracle_targets is None else oracle_targets[start:stop]
                ),
                device=device,
                progress_callback=progress,
            )
        )
    return _merge_sampling_results(parts)


def _forward_terminal_pairs(
    initial_states: np.ndarray,
    labels: np.ndarray,
    path_ids: np.ndarray,
    *,
    forward_seed: int,
    reverse_seed: int,
    device: str,
    model: core.EulerianJacobiDDPMModel,
    oracle: bool = False,
    resource_run_dir: Path | None = None,
) -> tuple[core.SamplingResult, core.SamplingResult]:
    progress = (
        None if resource_run_dir is None else _stage_progress_callback(resource_run_dir)
    )
    null_parts: list[core.SamplingResult] = []
    candidate_parts: list[core.SamplingResult] = []
    for start in range(0, len(path_ids), 8):
        stop = min(start + 8, len(path_ids))
        null = core.forward_terminal_sample(
            initial_states[start:stop],
            labels[start:stop],
            path_ids[start:stop],
            forward_seed=int(forward_seed),
            controller="null",
            root_seed=int(reverse_seed),
            device=device,
            progress_callback=progress,
        )
        if oracle:
            candidate = core.oracle_sample(
                null.starts,
                labels[start:stop],
                path_ids[start:stop],
                initial_states[start:stop],
                root_seed=int(reverse_seed),
                device=device,
                progress_callback=progress,
            )
        else:
            candidate = core.reverse_sample(
                null.starts,
                labels[start:stop],
                path_ids[start:stop],
                controller="learned",
                root_seed=int(reverse_seed),
                model=model,
                device=device,
                progress_callback=progress,
            )
        null_parts.append(null)
        candidate_parts.append(candidate)
    return _merge_sampling_results(null_parts), _merge_sampling_results(candidate_parts)


def _telemetry_rows(population: str, result: core.SamplingResult) -> list[dict[str, Any]]:
    return [
        {
            "population": population,
            **dict(row),
            "finite": int(result.telemetry["finite"]),
            "nonnegative": int(result.telemetry["nonnegative"]),
            "microsteps": int(result.telemetry["microsteps"]),
            "exact_facet_count": int(result.telemetry["exact_facet_count"]),
        }
        for row in result.telemetry["by_time_quarter"]
    ]


def _pilot_prior_classifier_evidence(
    evaluator: SmallMnistCNN,
    null_images: np.ndarray,
    learned_images: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    evaluations = {
        "null": evaluate_generated_labels(
            evaluator, null_images, labels, sample_ids, batch_size=256, device="cpu"
        ),
        "learned": evaluate_generated_labels(
            evaluator, learned_images, labels, sample_ids, batch_size=256, device="cpu"
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "requested_labels": np.asarray(labels, dtype=np.int64),
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
    }
    requested_log_probabilities: dict[str, np.ndarray] = {}
    for name, evaluation in evaluations.items():
        logits = np.asarray(evaluation["logits"], dtype=np.float64)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
        log_probabilities = shifted - np.log(
            np.sum(exponentials, axis=1, keepdims=True)
        )
        requested = log_probabilities[np.arange(len(labels)), labels]
        requested_log_probabilities[name] = requested
        arrays[f"{name}_predictions"] = np.asarray(
            evaluation["predictions"], dtype=np.int64
        )
        arrays[f"{name}_logits"] = logits
        arrays[f"{name}_probabilities"] = probabilities
        arrays[f"{name}_requested_log_probabilities"] = requested
    null_accuracy = float(np.mean(arrays["null_predictions"] == labels))
    learned_accuracy = float(np.mean(arrays["learned_predictions"] == labels))
    paired_delta = (
        requested_log_probabilities["learned"]
        - requested_log_probabilities["null"]
    )
    frozen = FROZEN_CONFIG["objective_pilot"]
    wins = int(np.sum(paired_delta > 0.0))
    mean_delta = float(np.mean(paired_delta, dtype=np.float64))
    passed = int(
        learned_accuracy
        >= float(frozen["minimum_learned_prior_requested_accuracy"])
        and learned_accuracy > null_accuracy
        and wins >= int(frozen["minimum_prior_requested_log_probability_wins"])
        and mean_delta
        > float(frozen["minimum_mean_prior_requested_log_probability_improvement"])
    )
    summary = {
        "gate_type": "diagnostic threshold",
        "terminal_test_rows_used": 0,
        "path_count": int(len(labels)),
        "null_requested_label_accuracy": null_accuracy,
        "learned_requested_label_accuracy": learned_accuracy,
        "paired_requested_log_probability_win_count": wins,
        "mean_paired_requested_log_probability_improvement": mean_delta,
        "passed": passed,
    }
    return arrays, summary


def _pilot_metrics(
    null_forward: np.ndarray,
    learned_forward: np.ndarray,
    forward_targets: np.ndarray,
    null_oracle: np.ndarray,
    oracle: np.ndarray,
    oracle_targets: np.ndarray,
    telemetry: Sequence[Mapping[str, Any]],
    prior_task_signal: Mapping[str, Any],
) -> dict[str, Any]:
    null_l1 = np.sum(np.abs(null_forward - forward_targets), axis=1, dtype=np.float64)
    learned_l1 = np.sum(
        np.abs(learned_forward - forward_targets), axis=1, dtype=np.float64
    )
    null_total = float(np.sum(null_l1, dtype=np.float64))
    _require(null_total > 0.0 and math.isfinite(null_total), "pilot null L1 is invalid")
    null_oracle_l1 = np.sum(
        np.abs(null_oracle - oracle_targets), axis=1, dtype=np.float64
    )
    oracle_l1 = np.sum(np.abs(oracle - oracle_targets), axis=1, dtype=np.float64)
    oracle_wins = int(np.sum(oracle_l1 < null_oracle_l1))
    telemetry_health = all(
        int(row["finite"]) == 1
        and int(row["nonnegative"]) == 1
        and int(row["microsteps"]) == 2
        and math.isfinite(float(row["controller_rms"]))
        and math.isfinite(float(row["maximum_mass_error"]))
        and float(row["maximum_mass_error"]) <= 2e-12
        and math.isfinite(float(row["maximum_pair_total_error"]))
        and float(row["maximum_pair_total_error"]) <= 2e-12
        and int(row["exact_facet_count"]) >= 0
        for row in telemetry
    )
    states = (
        null_forward,
        learned_forward,
        forward_targets,
        null_oracle,
        oracle,
        oracle_targets,
    )
    state_health = all(
        np.isfinite(value).all()
        and np.all(value >= 0.0)
        and float(np.max(np.abs(value.sum(axis=1) - 1.0))) <= 2e-12
        for value in states
    )
    learned_quarters = [
        float(row["controller_rms"])
        for row in telemetry
        if str(row["population"]).startswith("learned-")
    ]
    return {
        "gates": {
            "gate_c_passed": int(
                oracle_wins
                >= int(FROZEN_CONFIG["diagnostics"]["oracle_minimum_improved_paths"])
                and float(np.sum(oracle_l1)) < float(np.sum(null_oracle_l1))
            ),
            "health_passed": int(telemetry_health and state_health),
        },
        "forward_terminal": {
            "learned_l1_win_count": int(np.sum(learned_l1 < null_l1)),
            "aggregate_l1_relative_improvement": (
                null_total - float(np.sum(learned_l1, dtype=np.float64))
            )
            / null_total,
        },
        "oracle": {"l1_win_count": oracle_wins},
        "controller": {
            "learned_rms": math.sqrt(float(np.mean(np.square(learned_quarters))))
        },
        "prior_task_signal": dict(prior_task_signal),
    }


def _last_stage_seconds(run_dir: Path, role: str) -> float:
    events = _read_json(run_dir / "resource_ledger.json")["events"]
    _require(bool(events) and events[-1]["role"] == role, f"resource event changed: {role}")
    return float(events[-1]["seconds"])


def _stage_progress_callback(run_dir: Path) -> Callable[[], None]:
    started = time.perf_counter()
    last_check = [started]

    def check() -> None:
        now = time.perf_counter()
        if now - last_check[0] < 1.0:
            return
        last_check[0] = now
        _resource_check(run_dir, projected_seconds=now - started)

    return check


def run_resource_preflight(
    run_dir: Path,
    train_u8: np.ndarray,
    train_y: np.ndarray,
    validation_u8: np.ndarray,
    validation_y: np.ndarray,
    *,
    device: str,
) -> dict[str, Any]:
    """Measure all heavy phases and project the declared pilot plus full workload."""

    train_states = core.mix_unit_masses(_mnist_unit_masses(train_u8[:4]))
    validation_states = core.mix_unit_masses(_mnist_unit_masses(validation_u8[:4]))
    train_ids = np.arange(0xB20C0, 0xB20C4, dtype=np.int64)
    validation_ids = np.arange(0xB20D0, 0xB20D4, dtype=np.int64)
    train_records = _run_stage(
        run_dir,
        "resource_forward_train_16_records",
        lambda: core.build_forward_records(
            train_states,
            np.asarray(train_y[:4], dtype=np.int64),
            train_ids,
            root_seed=0xE14E01,
            device=device,
            progress_callback=_stage_progress_callback(run_dir),
        ),
    )
    train_record_seconds = _last_stage_seconds(run_dir, "resource_forward_train_16_records")
    validation_records = _run_stage(
        run_dir,
        "resource_forward_validation_16_records",
        lambda: core.build_forward_records(
            validation_states,
            np.asarray(validation_y[:4], dtype=np.int64),
            validation_ids,
            root_seed=0xE14E02,
            device=device,
            progress_callback=_stage_progress_callback(run_dir),
        ),
    )
    validation_record_seconds = _last_stage_seconds(
        run_dir, "resource_forward_validation_16_records"
    )
    training = _run_stage(
        run_dir,
        "resource_training_25_updates",
        lambda: core.train_jacobi_ddpm(
            train_records,
            validation_records,
            device=device,
            updates=25,
            validation_interval=25,
            seed=0xE14E03,
        ),
    )
    training_seconds = _last_stage_seconds(run_dir, "resource_training_25_updates")
    model = core.make_model()
    model.load_state_dict(training.selected_state_dict)
    prior_ids = np.arange(0xB20E0, 0xB20E8, dtype=np.int64)
    prior_labels = np.arange(8, dtype=np.int64) % 10
    prior = _run_stage(
        run_dir,
        "resource_prior_8_paths",
        lambda: core.prior_sample(
            prior_ids,
            prior_labels,
            start_seed=0xE14E04,
            controller="learned",
            root_seed=0xE14E05,
            model=model,
            device=device,
            progress_callback=_stage_progress_callback(run_dir),
        ),
    )
    prior_seconds = _last_stage_seconds(run_dir, "resource_prior_8_paths")
    forward_ids = np.arange(0xB20F0, 0xB20F4, dtype=np.int64)
    forward = _run_stage(
        run_dir,
        "resource_forward_terminal_4_paths",
        lambda: core.forward_terminal_sample(
            validation_states,
            np.asarray(validation_y[:4], dtype=np.int64),
            forward_ids,
            forward_seed=0xE14E06,
            controller="learned",
            root_seed=0xE14E07,
            model=model,
            device=device,
            progress_callback=_stage_progress_callback(run_dir),
        ),
    )
    forward_seconds = _last_stage_seconds(run_dir, "resource_forward_terminal_4_paths")
    _require(
        int(prior.telemetry["finite"]) == 1
        and int(prior.telemetry["nonnegative"]) == 1
        and int(forward.telemetry["finite"]) == 1
        and int(forward.telemetry["nonnegative"]) == 1,
        "resource-smoke sampling health failed",
    )

    measured_cache_transitions = 8 * 128 * 7 * 392
    measured_reverse_transitions = 8 * 1_404_928
    cache_seconds_per_transition = (
        train_record_seconds + validation_record_seconds
    ) / measured_cache_transitions
    reverse_seconds_per_transition = prior_seconds / measured_reverse_transitions
    training_seconds_per_update = training_seconds / 25.0
    pilot = FROZEN_CONFIG["objective_pilot"]
    full = FROZEN_CONFIG["resource_projection"]
    pilot_seconds = (
        cache_seconds_per_transition * int(pilot["projected_cache_pair_transitions"])
        + reverse_seconds_per_transition
        * int(pilot["projected_reverse_reference_transitions"])
        + cache_seconds_per_transition
        * int(pilot["projected_forward_sampling_transitions"])
        + training_seconds_per_update * int(pilot["training_updates"])
    )
    full_seconds = (
        cache_seconds_per_transition * int(full["full_cache_pair_transitions"])
        + reverse_seconds_per_transition
        * int(full["full_reverse_reference_transitions"])
        + cache_seconds_per_transition * int(full["full_forward_sampling_transitions"])
        + training_seconds_per_update * int(FROZEN_CONFIG["training"]["updates"])
    )
    reserve = float(FROZEN_CONFIG["resource_defaults"]["terminal_reserve_seconds"])
    projected_seconds = pilot_seconds + full_seconds + reserve
    full_projected_storage = max(
        int(FROZEN_CONFIG["resource_projection"]["expected_persisted_storage_bytes"]),
        int(
            (_dataset_bytes(train_records) + _dataset_bytes(validation_records))
            / 8
            * (core.TRAIN_PATH_COUNT + core.VALIDATION_PATH_COUNT)
            * 1.10
        ),
    )
    pilot_projected_storage = max(
        256 * 1024**2,
        int(
            (_dataset_bytes(train_records) + _dataset_bytes(validation_records))
            / 8
            * 350
            * 1.10
        ),
    )
    projection = {
        "schema": VERSION + "-measured-resource-projection",
        "measured": {
            "forward_record_count": 32,
            "forward_path_count": 8,
            "forward_pair_transitions": measured_cache_transitions,
            "forward_seconds": train_record_seconds + validation_record_seconds,
            "optimizer_updates": 25,
            "optimizer_seconds": training_seconds,
            "prior_reverse_paths": 8,
            "prior_reverse_reference_transitions": measured_reverse_transitions,
            "prior_reverse_seconds": prior_seconds,
            "forward_terminal_paths": 4,
            "forward_terminal_seconds": forward_seconds,
        },
        "rates": {
            "cache_pair_transitions_per_second": 1.0 / cache_seconds_per_transition,
            "reverse_reference_transitions_per_second": 1.0 / reverse_seconds_per_transition,
            "optimizer_updates_per_second": 1.0 / training_seconds_per_update,
        },
        "pilot": {
            "cache_pair_transitions": int(pilot["projected_cache_pair_transitions"]),
            "reverse_reference_transitions": int(
                pilot["projected_reverse_reference_transitions"]
            ),
            "forward_sampling_transitions": int(
                pilot["projected_forward_sampling_transitions"]
            ),
            "sampling_transition_work": int(pilot["projected_sampling_transition_work"]),
            "projected_seconds": pilot_seconds,
            "base_transition_work": 273_960_960,
            "transition_work_including_shared_audit": 282_741_760,
        },
        "full": {
            "cache_pair_transitions": 1_756_160_000,
            "reverse_reference_transitions": 590_069_760,
            "forward_sampling_transitions": 17_561_600,
            "sampling_transition_work": 607_631_360,
            "base_transition_work": 2_363_791_360,
            "shared_k128_k512_audit_transitions": 8_780_800,
            "transition_work_including_shared_audit": 2_372_572_160,
            "projected_seconds": full_seconds,
        },
        "shared_k128_k512_audit_transitions": 8_780_800,
        "pilot_plus_full_transition_work_with_one_shared_audit": 2_646_533_120,
        "projection_mode": "pilot_plus_conditional_full",
        "audit_seconds_already_charged": 1,
        "terminal_reserve_seconds": reserve,
        "pilot_plus_full_projected_seconds": projected_seconds,
        "pilot_projected_storage_bytes": pilot_projected_storage,
        "full_projected_storage_bytes": full_projected_storage,
    }
    ledger = _read_json(run_dir / "resource_ledger.json")
    current_storage = _directory_bytes(run_dir)
    pilot_time_passed = (
        float(ledger["active_seconds"]) + pilot_seconds + reserve
        < float(ledger["maximum_active_seconds"])
    )
    full_time_passed = (
        float(ledger["active_seconds"]) + full_seconds + reserve
        < float(ledger["maximum_active_seconds"])
    )
    pilot_storage_passed = (
        current_storage + pilot_projected_storage < int(ledger["maximum_storage_bytes"])
    )
    full_storage_passed = (
        current_storage + full_projected_storage < int(ledger["maximum_storage_bytes"])
    )
    cuda_passed = float(ledger["peak_cuda_fraction"]) < float(
        ledger["maximum_cuda_fraction"]
    )
    projection["admission"] = {
        "pilot_time_cap_passed": int(pilot_time_passed),
        "pilot_storage_cap_passed": int(pilot_storage_passed),
        "pilot_passed": int(pilot_time_passed and pilot_storage_passed and cuda_passed),
        "full_time_cap_passed": int(full_time_passed),
        "full_storage_cap_passed": int(full_storage_passed),
        "full_passed": int(full_time_passed and full_storage_passed and cuda_passed),
        "cuda_cap_passed": int(cuda_passed),
        "required_scope": "pilot",
        "passed": int(pilot_time_passed and pilot_storage_passed and cuda_passed),
    }
    ledger["latest_projection"] = projection
    _write_json(run_dir / "resource_ledger.json", ledger)
    _write_json(run_dir / "preflight_projection.json", projection)
    if projection["admission"]["passed"] != 1:
        raise ResourceStop("measured required workload exceeds an explicit resource cap")
    return projection


def require_post_pilot_full_resources(run_dir: Path) -> dict[str, Any]:
    """Recheck remaining full-stage authority after the decisive pilot has run."""

    run_dir = Path(run_dir).resolve()
    projection = _read_json(run_dir / "preflight_projection.json")
    ledger = _read_json(run_dir / "resource_ledger.json")
    full_seconds = float(projection["full"]["projected_seconds"])
    reserve = float(projection["terminal_reserve_seconds"])
    storage = int(projection["full_projected_storage_bytes"])
    active_at_check = float(ledger["active_seconds"])
    storage_at_check = _directory_bytes(run_dir)
    peak_cuda_at_check = float(ledger["peak_cuda_fraction"])
    time_passed = (
        active_at_check + full_seconds + reserve
        < float(ledger["maximum_active_seconds"])
    )
    storage_passed = (
        storage_at_check + storage < int(ledger["maximum_storage_bytes"])
    )
    cuda_passed = peak_cuda_at_check < float(ledger["maximum_cuda_fraction"])
    result = {
        "checked_after_positive_pilot": 1,
        "approval_id": str(ledger["approval_id"]),
        "ledger_event_count_at_check": len(ledger["events"]),
        "active_seconds_at_check": active_at_check,
        "storage_bytes_at_check": storage_at_check,
        "peak_cuda_fraction_at_check": peak_cuda_at_check,
        "maximum_active_seconds": float(ledger["maximum_active_seconds"]),
        "maximum_storage_bytes": int(ledger["maximum_storage_bytes"]),
        "maximum_cuda_fraction": float(ledger["maximum_cuda_fraction"]),
        "full_projected_seconds": full_seconds,
        "full_projected_storage_bytes": storage,
        "terminal_reserve_seconds": reserve,
        "time_cap_passed": int(time_passed),
        "storage_cap_passed": int(storage_passed),
        "cuda_cap_passed": int(cuda_passed),
        "passed": int(time_passed and storage_passed and cuda_passed),
    }
    projection["post_pilot_full_admission"] = result
    ledger["latest_projection"] = projection
    _write_json(run_dir / "preflight_projection.json", projection)
    _write_json(run_dir / "resource_ledger.json", ledger)
    if result["passed"] != 1:
        raise ResourceStop(
            "positive pilot completed but remaining full workload exceeds an explicit cap"
        )
    return result


def execute_objective_pilot(
    parent_run_dir: Path,
    pilot_dir: Path,
    train_u8: np.ndarray,
    train_y: np.ndarray,
    validation_u8: np.ndarray,
    validation_y: np.ndarray,
    *,
    device: str,
) -> dict[str, Any]:
    """Execute the prespecified all-ten-class objective pilot without test data."""

    parent_run_dir = Path(parent_run_dir).resolve()
    pilot_dir = Path(pilot_dir).resolve()
    _require(not pilot_dir.exists(), "objective pilot directory already exists")
    pilot_dir.mkdir(parents=True)
    (pilot_dir / "images").mkdir()
    (pilot_dir / "authority").mkdir()
    parent_authorities = (
        "config.json",
        "source_bindings.json",
        "environment.json",
        "theory_code_identity.json",
        "kernel_audit.json",
        "k128_k512_audit.json",
        "preflight_projection.json",
        "path_id_audit.json",
    )
    _require(
        all((parent_run_dir / name).is_file() for name in parent_authorities),
        "objective pilot parent authorities are incomplete",
    )
    for name in parent_authorities:
        destination = pilot_dir / "authority" / name
        _atomic_replace(
            destination,
            lambda temporary, source=parent_run_dir / name: shutil.copyfile(
                source, temporary
            ),
        )
    config = copy.deepcopy(FROZEN_CONFIG)
    config["run_scope"] = "objective_pilot"
    config["device"] = device
    config["parent_source_bindings_sha256"] = _file_sha256(
        parent_run_dir / "source_bindings.json"
    )
    config["parent_config_sha256"] = _file_sha256(parent_run_dir / "config.json")
    _write_json(pilot_dir / "config.json", config)
    _status(pilot_dir, "initialized")

    train_indices = core.balanced_class_indices(
        train_y, per_class=25, start=0, stop=len(train_y)
    )
    validation_indices = core.balanced_class_indices(
        validation_y, per_class=10, start=0, stop=len(validation_y)
    )
    train_labels = np.asarray(train_y[train_indices], dtype=np.int64)
    validation_labels = np.asarray(validation_y[validation_indices], dtype=np.int64)
    _require(
        np.array_equal(np.bincount(train_labels, minlength=10), np.full(10, 25))
        and np.array_equal(
            np.bincount(validation_labels, minlength=10), np.full(10, 10)
        ),
        "objective-pilot data allocation is not exactly balanced",
    )
    train_states = core.mix_unit_masses(_mnist_unit_masses(train_u8[train_indices]))
    validation_states = core.mix_unit_masses(
        _mnist_unit_masses(validation_u8[validation_indices])
    )
    train_path_ids = _role_path_ids("pilot_train")
    validation_path_ids = _role_path_ids("pilot_validation")
    train_records = _run_stage(
        parent_run_dir,
        "pilot_forward_cache_train_250",
        lambda: core.build_forward_records(
            train_states,
            train_labels,
            train_path_ids,
            root_seed=int(FROZEN_CONFIG["seeds"]["records_train"]),
            device=device,
            progress_callback=_stage_progress_callback(parent_run_dir),
        ),
    )
    validation_records = _run_stage(
        parent_run_dir,
        "pilot_forward_cache_validation_100",
        lambda: core.build_forward_records(
            validation_states,
            validation_labels,
            validation_path_ids,
            root_seed=int(FROZEN_CONFIG["seeds"]["records_validation"]),
            device=device,
            progress_callback=_stage_progress_callback(parent_run_dir),
        ),
    )
    _write_npz(
        pilot_dir / "forward_records.npz",
        **_dataset_arrays("train", train_records),
        **_dataset_arrays("validation", validation_records),
    )
    _write_npz(
        pilot_dir / "data_roles.npz",
        train_arff_indices=train_indices,
        train_labels=train_labels,
        train_path_ids=train_path_ids,
        validation_arff_indices=validation_indices + 55_000,
        validation_labels=validation_labels,
        validation_path_ids=validation_path_ids,
    )
    training_started = time.perf_counter()

    def checkpoint(record: Mapping[str, Any]) -> None:
        _write_torch(
            pilot_dir / "training_latest.pt",
            {"schema": VERSION + "-pilot-training-checkpoint", **dict(record)},
        )
        _resource_check(
            parent_run_dir, projected_seconds=time.perf_counter() - training_started
        )

    training = _run_stage(
        parent_run_dir,
        "pilot_training_750_updates",
        lambda: core.train_jacobi_ddpm(
            train_records,
            validation_records,
            device=device,
            updates=750,
            batch_size=64,
            learning_rate=2e-4,
            ema_decay=0.999,
            validation_interval=250,
            seed=int(FROZEN_CONFIG["training"]["seed"]),
            checkpoint_callback=checkpoint,
        ),
    )
    _write_torch(
        pilot_dir / "selected_checkpoint.pt",
        {
            "schema": VERSION + "-pilot-selected-checkpoint",
            "state_dict": training.selected_state_dict,
            "selected_update": training.selected_update,
            "selected_validation_normalized_mse": training.selected_validation_mse,
            "training_target_energy": training.training_target_energy,
            "completed_updates": training.completed_updates,
            "architecture": global_dilated_architecture_contract(),
        },
    )
    _write_csv(pilot_dir / "training_history.csv", list(training.history))
    _write_json(
        pilot_dir / "training_selection.json",
        {
            "selected_update": training.selected_update,
            "selected_validation_normalized_mse": training.selected_validation_mse,
            "training_target_energy": training.training_target_energy,
            "completed_updates": training.completed_updates,
            "checkpoint_sha256": _file_sha256(pilot_dir / "selected_checkpoint.pt"),
        },
    )
    model = core.make_model()
    model.load_state_dict(training.selected_state_dict)

    prior_path_ids = _role_path_ids("pilot_prior")
    prior_labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    prior_sample_ids = np.asarray(
        [f"pilot-prior-{value:05x}" for value in prior_path_ids], dtype=np.str_
    )
    prior_starts = core.sample_dirichlet_starts(
        prior_path_ids, root_seed=int(FROZEN_CONFIG["seeds"]["prior"])
    )
    _write_prior_start_authority(
        pilot_dir, prior_starts, prior_labels, prior_path_ids, prior_sample_ids
    )
    null_prior = _run_stage(
        parent_run_dir,
        "pilot_null_prior_20",
        lambda: _reverse_cohorts(
            prior_starts,
            prior_labels,
            prior_path_ids,
            controller="null",
            root_seed=int(FROZEN_CONFIG["seeds"]["prior_reverse"]),
            device=device,
            resource_run_dir=parent_run_dir,
        ),
        commit=lambda result: _write_population_stage(
            pilot_dir,
            "null_prior",
            result,
            prior_labels,
            prior_path_ids,
            prior_sample_ids,
        ),
    )
    learned_prior = _run_stage(
        parent_run_dir,
        "pilot_learned_prior_20",
        lambda: _reverse_cohorts(
            prior_starts,
            prior_labels,
            prior_path_ids,
            controller="learned",
            root_seed=int(FROZEN_CONFIG["seeds"]["prior_reverse"]),
            model=model,
            device=device,
            resource_run_dir=parent_run_dir,
        ),
        commit=lambda result: _write_population_stage(
            pilot_dir,
            "learned_prior",
            result,
            prior_labels,
            prior_path_ids,
            prior_sample_ids,
        ),
    )

    forward_indices = np.concatenate(
        [np.flatnonzero(validation_labels == digit)[:2] for digit in range(10)]
    ).astype(np.int64)
    forward_targets = validation_states[forward_indices]
    forward_labels = validation_labels[forward_indices]
    forward_path_ids = _role_path_ids("pilot_forward_terminal")
    forward_sample_ids = np.asarray(
        [f"pilot-forward-{value:05x}" for value in forward_path_ids], dtype=np.str_
    )
    null_forward, learned_forward = _run_stage(
        parent_run_dir,
        "pilot_forward_terminal_null_learned_20",
        lambda: _forward_terminal_pairs(
            forward_targets,
            forward_labels,
            forward_path_ids,
            forward_seed=int(FROZEN_CONFIG["seeds"]["forward_terminal"]),
            reverse_seed=int(FROZEN_CONFIG["seeds"]["forward_terminal"]) ^ 0x51,
            device=device,
            model=model,
            resource_run_dir=parent_run_dir,
        ),
        commit=lambda results: (
            _write_population_stage(
                pilot_dir,
                "null_forward_terminal",
                results[0],
                forward_labels,
                forward_path_ids,
                forward_sample_ids,
            ),
            _write_population_stage(
                pilot_dir,
                "learned_forward_terminal",
                results[1],
                forward_labels,
                forward_path_ids,
                forward_sample_ids,
            ),
        ),
    )

    oracle_indices = np.concatenate(
        [np.flatnonzero(validation_labels == digit)[:1] for digit in range(10)]
    ).astype(np.int64)
    oracle_targets = validation_states[oracle_indices]
    oracle_labels = validation_labels[oracle_indices]
    oracle_path_ids = _role_path_ids("pilot_oracle")
    oracle_sample_ids = np.asarray(
        [f"pilot-oracle-{value:05x}" for value in oracle_path_ids], dtype=np.str_
    )
    null_oracle, oracle = _run_stage(
        parent_run_dir,
        "pilot_oracle_null_positive_10",
        lambda: _forward_terminal_pairs(
            oracle_targets,
            oracle_labels,
            oracle_path_ids,
            forward_seed=int(FROZEN_CONFIG["seeds"]["oracle"]),
            reverse_seed=int(FROZEN_CONFIG["seeds"]["oracle"]) ^ 0x51,
            device=device,
            model=model,
            oracle=True,
            resource_run_dir=parent_run_dir,
        ),
        commit=lambda results: (
            _write_population_stage(
                pilot_dir,
                "null_oracle",
                results[0],
                oracle_labels,
                oracle_path_ids,
                oracle_sample_ids,
            ),
            _write_population_stage(
                pilot_dir,
                "oracle",
                results[1],
                oracle_labels,
                oracle_path_ids,
                oracle_sample_ids,
            ),
        ),
    )

    population_rows = {
        "null_prior": null_prior.final_states,
        "learned_prior": learned_prior.final_states,
        "null_forward_terminal": null_forward.final_states,
        "learned_forward_terminal": learned_forward.final_states,
        "forward_targets": forward_targets,
        "null_oracle": null_oracle.final_states,
        "oracle": oracle.final_states,
        "oracle_targets": oracle_targets,
    }
    demixed_rows = {
        name + "_demixed": core.demix_unit_masses(values)
        for name, values in population_rows.items()
    }
    identity_rows = {
        "prior_requested_labels": prior_labels,
        "prior_path_ids": prior_path_ids,
        "prior_sample_ids": prior_sample_ids,
        "forward_requested_labels": forward_labels,
        "forward_path_ids": forward_path_ids,
        "forward_sample_ids": forward_sample_ids,
        "oracle_requested_labels": oracle_labels,
        "oracle_path_ids": oracle_path_ids,
        "oracle_sample_ids": oracle_sample_ids,
    }
    anchors = np.asarray([0, 32, 64, 96, 128], dtype=np.int64)
    null_trajectory = np.stack([null_prior.anchors[int(value)] for value in anchors], axis=1)
    learned_trajectory = np.stack(
        [learned_prior.anchors[int(value)] for value in anchors], axis=1
    )
    _write_npz(
        pilot_dir / "start_banks.npz",
        prior_starts=prior_starts,
        prior_requested_labels=prior_labels,
        prior_path_ids=prior_path_ids,
        prior_sample_ids=prior_sample_ids,
        forward_terminal_starts=null_forward.starts,
        oracle_starts=null_oracle.starts,
    )
    _write_npz(
        pilot_dir / "populations.npz",
        **population_rows,
        **demixed_rows,
        **identity_rows,
        prior_completed_steps=anchors,
        null_prior_trajectories=null_trajectory,
        learned_prior_trajectories=learned_trajectory,
    )
    rendered_rows = {name: _rasterize_rows(demixed_rows[name + "_demixed"]) for name in population_rows}
    _write_npz(
        pilot_dir / "uint8_populations.npz", **rendered_rows, **identity_rows
    )
    telemetry = [
        *_telemetry_rows("null-prior", null_prior),
        *_telemetry_rows("learned-prior", learned_prior),
        *_telemetry_rows("null-forward-terminal", null_forward),
        *_telemetry_rows("learned-forward-terminal", learned_forward),
        *_telemetry_rows("oracle", oracle),
    ]
    _write_csv(pilot_dir / "telemetry.csv", telemetry)
    _prior_classifier_outputs, prior_task_signal = _run_stage(
        parent_run_dir,
        "pilot_prior_cpu_classifier_scoring",
        lambda: _pilot_prior_classifier_evidence(
            _load_bound_evaluator(parent_run_dir, device="cpu"),
            rendered_rows["null_prior"],
            rendered_rows["learned_prior"],
            prior_labels,
            prior_sample_ids,
        ),
        commit=lambda result: (
            _write_npz(
                pilot_dir / "prior_classifier_outputs.npz", **result[0]
            ),
            _write_json(
                pilot_dir / "prior_classifier_metrics.json", result[1]
            ),
        ),
    )
    metrics = _pilot_metrics(
        null_forward.final_states,
        learned_forward.final_states,
        forward_targets,
        null_oracle.final_states,
        oracle.final_states,
        oracle_targets,
        telemetry,
        prior_task_signal,
    )
    _write_json(pilot_dir / "metrics.json", metrics)
    for name, images, labels in (
        ("prior_null", rendered_rows["null_prior"], prior_labels),
        ("prior_learned", rendered_rows["learned_prior"], prior_labels),
        ("forward_null", rendered_rows["null_forward_terminal"], forward_labels),
        ("forward_learned", rendered_rows["learned_forward_terminal"], forward_labels),
        ("oracle", rendered_rows["oracle"], oracle_labels),
    ):
        write_contact_sheet(
            pilot_dir / f"images/{name}.png",
            images,
            columns=10,
            captions=[str(int(label)) for label in labels],
        )
    frozen = FROZEN_CONFIG["objective_pilot"]
    dynamics_admitted = int(
        int(metrics["gates"]["gate_c_passed"]) == 1
        and int(metrics["gates"]["health_passed"]) == 1
        and int(metrics["forward_terminal"]["learned_l1_win_count"])
        >= int(frozen["minimum_forward_l1_wins"])
        and float(metrics["forward_terminal"]["aggregate_l1_relative_improvement"])
        >= float(frozen["minimum_aggregate_l1_relative_improvement"])
        and float(metrics["controller"]["learned_rms"]) > 0.0
    )
    admitted = int(
        dynamics_admitted == 1
        and int(metrics["prior_task_signal"]["passed"]) == 1
    )
    if int(metrics["gates"]["health_passed"]) != 1:
        route = "pilot_invalid_health_repair_before_interpretation"
    elif int(metrics["gates"]["gate_c_passed"]) != 1:
        route = "repair_reverse_composition_before_judging_learner"
    elif dynamics_admitted == 1 and int(metrics["prior_task_signal"]["passed"]) != 1:
        route = "pilot_prior_negative_stop_before_scale"
    elif admitted:
        route = "full_scale_admitted"
    else:
        route = "pilot_negative_stop_before_scale"
    _write_text(
        pilot_dir / "REPORT.md",
        "# Eulerian Jacobi DDPM-v0 objective pilot\n\n"
        "Primary mode: exploratory. This all-ten-class pilot directly sampled prior, "
        "forward-terminal, null, learned, and oracle populations.\n\n"
        f"Diagnostic route: `{route}`. Metrics: `{metrics}`.\n\n"
        "A negative applies only to the frozen K=128 model/controller/pilot and is a "
        "stop-before-scale result, not a universal learner or Eulerian failure. No "
        "terminal-test evidence was opened.\n",
    )
    _status(pilot_dir, route)
    seal_populations(pilot_dir)
    _seal_manifest(pilot_dir)
    return validate_objective_pilot(pilot_dir)


def _load_accepted_evaluator(
    bindings: Mapping[str, Any], *, device: str
) -> SmallMnistCNN:
    checkpoint = Path(bindings["ddpm_run_dir"]) / "evaluator/selected_checkpoint.pt"
    _require(
        checkpoint.is_file()
        and _file_sha256(checkpoint) == bindings["evaluator_checkpoint_sha256"]
        and bindings["evaluator_checkpoint_sha256"]
        == ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256,
        "bound DDPM evaluator checkpoint changed",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict) and "state_dict" in payload, "evaluator payload changed")
    evaluator = SmallMnistCNN()
    evaluator.load_state_dict(payload["state_dict"])
    return evaluator.to(torch.device(device)).eval()


def _load_bound_evaluator(run_dir: Path, *, device: str) -> SmallMnistCNN:
    return _load_accepted_evaluator(
        _read_json(run_dir / "source_bindings.json"), device=device
    )


def _validate_pilot_prior_classifier(pilot_dir: Path) -> dict[str, Any]:
    with np.load(pilot_dir / "uint8_populations.npz", allow_pickle=False) as rendered:
        null_images = np.asarray(rendered["null_prior"], dtype=np.uint8)
        learned_images = np.asarray(rendered["learned_prior"], dtype=np.uint8)
        labels = np.asarray(rendered["prior_requested_labels"], dtype=np.int64)
        sample_ids = np.asarray(rendered["prior_sample_ids"], dtype=np.str_)
    bindings = _read_json(pilot_dir / "authority/source_bindings.json")
    expected_arrays, expected_summary = _pilot_prior_classifier_evidence(
        _load_accepted_evaluator(bindings, device="cpu"),
        null_images,
        learned_images,
        labels,
        sample_ids,
    )
    with np.load(
        pilot_dir / "prior_classifier_outputs.npz", allow_pickle=False
    ) as archive:
        _require(
            set(archive.files) == set(expected_arrays),
            "pilot prior-classifier raw schema changed",
        )
        observed_arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name, expected in expected_arrays.items():
        observed = observed_arrays[name]
        if expected.dtype.kind in {"f", "c"}:
            matches = np.allclose(observed, expected, rtol=2e-6, atol=2e-7)
        else:
            matches = np.array_equal(observed, expected)
        _require(bool(matches), f"pilot prior-classifier raw output changed: {name}")
    _require(
        _semantic_sha256(_read_json(pilot_dir / "prior_classifier_metrics.json"))
        == _semantic_sha256(expected_summary),
        "pilot prior-classifier metrics changed",
    )
    return expected_summary


def _exact_match_count(images: np.ndarray, reference: np.ndarray) -> int:
    keys = {
        row.tobytes()
        for row in np.ascontiguousarray(reference).reshape(len(reference), -1)
    }
    return sum(
        row.tobytes() in keys
        for row in np.ascontiguousarray(images).reshape(len(images), -1)
    )


def _paired_endpoint_metrics(
    null: np.ndarray,
    learned: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    null_l1 = np.sum(np.abs(null - targets), axis=1, dtype=np.float64)
    learned_l1 = np.sum(np.abs(learned - targets), axis=1, dtype=np.float64)
    correlations: list[float | None] = []
    for value, target in zip(learned, targets, strict=True):
        a = value - np.mean(value)
        b = target - np.mean(target)
        denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        correlations.append(float(np.sum(a * b) / denominator) if denominator > 0 else None)
    return {
        "path_count": int(len(null)),
        "null_l1": null_l1,
        "learned_l1": learned_l1,
        "learned_l1_win_count": int(np.sum(learned_l1 < null_l1)),
        "mean_null_l1": float(np.mean(null_l1)),
        "mean_learned_l1": float(np.mean(learned_l1)),
        "mean_learned_minus_null_l1": float(np.mean(learned_l1 - null_l1)),
        "aggregate_l1_relative_improvement": (
            float(np.sum(null_l1)) - float(np.sum(learned_l1))
        )
        / float(np.sum(null_l1)),
        "learned_centered_correlation": correlations,
    }


def execute_full_experiment(
    run_dir: Path,
    arff: Path,
    train_u8: np.ndarray,
    train_y: np.ndarray,
    validation_u8: np.ndarray,
    validation_y: np.ndarray,
    *,
    device: str,
) -> dict[str, Any]:
    """Execute the frozen full whole-run-restart experiment after pilot admission."""

    run_dir = Path(run_dir).resolve()
    train_indices = core.balanced_class_indices(
        train_y, per_class=400, start=0, stop=len(train_y)
    )
    validation_indices = core.balanced_class_indices(
        validation_y, per_class=100, start=0, stop=len(validation_y)
    )
    train_labels = np.asarray(train_y[train_indices], dtype=np.int64)
    validation_labels = np.asarray(validation_y[validation_indices], dtype=np.int64)
    _require(
        np.array_equal(np.bincount(train_labels, minlength=10), np.full(10, 400))
        and np.array_equal(
            np.bincount(validation_labels, minlength=10), np.full(10, 100)
        ),
        "full data allocation is not exactly balanced",
    )
    train_states = core.mix_unit_masses(_mnist_unit_masses(train_u8[train_indices]))
    validation_states = core.mix_unit_masses(
        _mnist_unit_masses(validation_u8[validation_indices])
    )
    train_path_ids = _role_path_ids("train")
    validation_path_ids = _role_path_ids("validation")
    train_records = _run_stage(
        run_dir,
        "full_forward_cache_train_4000",
        lambda: core.build_forward_records(
            train_states,
            train_labels,
            train_path_ids,
            root_seed=int(FROZEN_CONFIG["seeds"]["records_train"]),
            device=device,
            progress_callback=_stage_progress_callback(run_dir),
        ),
    )
    validation_records = _run_stage(
        run_dir,
        "full_forward_cache_validation_1000",
        lambda: core.build_forward_records(
            validation_states,
            validation_labels,
            validation_path_ids,
            root_seed=int(FROZEN_CONFIG["seeds"]["records_validation"]),
            device=device,
            progress_callback=_stage_progress_callback(run_dir),
        ),
    )
    _write_npz(
        run_dir / "forward_records.npz",
        **_dataset_arrays("train", train_records),
        **_dataset_arrays("validation", validation_records),
    )
    _write_npz(
        run_dir / "data_roles.npz",
        train_arff_indices=train_indices,
        train_labels=train_labels,
        train_path_ids=train_path_ids,
        validation_arff_indices=validation_indices + 55_000,
        validation_labels=validation_labels,
        validation_path_ids=validation_path_ids,
    )
    training_started = time.perf_counter()

    def checkpoint(record: Mapping[str, Any]) -> None:
        _write_torch(
            run_dir / "training_latest.pt",
            {"schema": VERSION + "-full-training-checkpoint", **dict(record)},
        )
        _resource_check(run_dir, projected_seconds=time.perf_counter() - training_started)

    training = _run_stage(
        run_dir,
        "full_training_10000_updates",
        lambda: core.train_jacobi_ddpm(
            train_records,
            validation_records,
            device=device,
            updates=10_000,
            batch_size=64,
            learning_rate=2e-4,
            ema_decay=0.999,
            validation_interval=250,
            seed=int(FROZEN_CONFIG["training"]["seed"]),
            checkpoint_callback=checkpoint,
        ),
    )
    _write_torch(
        run_dir / "selected_checkpoint.pt",
        {
            "schema": VERSION + "-full-selected-checkpoint",
            "state_dict": training.selected_state_dict,
            "selected_update": training.selected_update,
            "selected_validation_normalized_mse": training.selected_validation_mse,
            "training_target_energy": training.training_target_energy,
            "completed_updates": training.completed_updates,
            "architecture": global_dilated_architecture_contract(),
        },
    )
    _write_csv(run_dir / "training_history.csv", list(training.history))
    _write_json(
        run_dir / "training_selection.json",
        {
            "selected_update": training.selected_update,
            "selected_validation_normalized_mse": training.selected_validation_mse,
            "training_target_energy": training.training_target_energy,
            "completed_updates": training.completed_updates,
            "checkpoint_sha256": _file_sha256(run_dir / "selected_checkpoint.pt"),
        },
    )
    model = core.make_model()
    model.load_state_dict(training.selected_state_dict)

    prior_path_ids = _role_path_ids("prior_evaluation")
    prior_labels = np.repeat(np.arange(10, dtype=np.int64), 16)
    prior_sample_ids = np.asarray(
        [f"full-prior-{value:05x}" for value in prior_path_ids], dtype=np.str_
    )
    prior_starts = core.sample_dirichlet_starts(
        prior_path_ids, root_seed=int(FROZEN_CONFIG["seeds"]["prior"])
    )
    _write_prior_start_authority(
        run_dir, prior_starts, prior_labels, prior_path_ids, prior_sample_ids
    )
    null_prior = _run_stage(
        run_dir,
        "full_null_prior_160",
        lambda: _reverse_cohorts(
            prior_starts,
            prior_labels,
            prior_path_ids,
            controller="null",
            root_seed=int(FROZEN_CONFIG["seeds"]["prior_reverse"]),
            device=device,
            resource_run_dir=run_dir,
        ),
        commit=lambda result: _write_population_stage(
            run_dir,
            "null_prior",
            result,
            prior_labels,
            prior_path_ids,
            prior_sample_ids,
        ),
    )
    learned_prior = _run_stage(
        run_dir,
        "full_learned_prior_160",
        lambda: _reverse_cohorts(
            prior_starts,
            prior_labels,
            prior_path_ids,
            controller="learned",
            root_seed=int(FROZEN_CONFIG["seeds"]["prior_reverse"]),
            model=model,
            device=device,
            resource_run_dir=run_dir,
        ),
        commit=lambda result: _write_population_stage(
            run_dir,
            "learned_prior",
            result,
            prior_labels,
            prior_path_ids,
            prior_sample_ids,
        ),
    )
    forward_indices = np.concatenate(
        [np.flatnonzero(validation_labels == digit)[:4] for digit in range(10)]
    ).astype(np.int64)
    forward_targets = validation_states[forward_indices]
    forward_labels = validation_labels[forward_indices]
    forward_path_ids = _role_path_ids("forward_terminal")
    forward_sample_ids = np.asarray(
        [f"full-forward-{value:05x}" for value in forward_path_ids], dtype=np.str_
    )
    null_forward, learned_forward = _run_stage(
        run_dir,
        "full_forward_terminal_null_learned_40",
        lambda: _forward_terminal_pairs(
            forward_targets,
            forward_labels,
            forward_path_ids,
            forward_seed=int(FROZEN_CONFIG["seeds"]["forward_terminal"]),
            reverse_seed=int(FROZEN_CONFIG["seeds"]["forward_terminal"]) ^ 0x51,
            device=device,
            model=model,
            resource_run_dir=run_dir,
        ),
        commit=lambda results: (
            _write_population_stage(
                run_dir,
                "null_forward_terminal",
                results[0],
                forward_labels,
                forward_path_ids,
                forward_sample_ids,
            ),
            _write_population_stage(
                run_dir,
                "learned_forward_terminal",
                results[1],
                forward_labels,
                forward_path_ids,
                forward_sample_ids,
            ),
        ),
    )
    oracle_indices = np.concatenate(
        [np.flatnonzero(validation_labels == digit)[:1] for digit in range(10)]
    ).astype(np.int64)
    oracle_targets = validation_states[oracle_indices]
    oracle_labels = validation_labels[oracle_indices]
    oracle_path_ids = _role_path_ids("oracle_controls")
    oracle_sample_ids = np.asarray(
        [f"full-oracle-{value:05x}" for value in oracle_path_ids], dtype=np.str_
    )
    null_oracle, oracle = _run_stage(
        run_dir,
        "full_oracle_null_positive_10",
        lambda: _forward_terminal_pairs(
            oracle_targets,
            oracle_labels,
            oracle_path_ids,
            forward_seed=int(FROZEN_CONFIG["seeds"]["oracle"]),
            reverse_seed=int(FROZEN_CONFIG["seeds"]["oracle"]) ^ 0x51,
            device=device,
            model=model,
            oracle=True,
            resource_run_dir=run_dir,
        ),
        commit=lambda results: (
            _write_population_stage(
                run_dir,
                "null_oracle",
                results[0],
                oracle_labels,
                oracle_path_ids,
                oracle_sample_ids,
            ),
            _write_population_stage(
                run_dir,
                "oracle",
                results[1],
                oracle_labels,
                oracle_path_ids,
                oracle_sample_ids,
            ),
        ),
    )
    population_rows = {
        "null_prior": null_prior.final_states,
        "learned_prior": learned_prior.final_states,
        "null_forward_terminal": null_forward.final_states,
        "learned_forward_terminal": learned_forward.final_states,
        "forward_targets": forward_targets,
        "null_oracle": null_oracle.final_states,
        "oracle": oracle.final_states,
        "oracle_targets": oracle_targets,
    }
    demixed_rows = {
        name + "_demixed": core.demix_unit_masses(values)
        for name, values in population_rows.items()
    }
    identity_rows = {
        "prior_requested_labels": prior_labels,
        "prior_path_ids": prior_path_ids,
        "prior_sample_ids": prior_sample_ids,
        "forward_requested_labels": forward_labels,
        "forward_path_ids": forward_path_ids,
        "forward_sample_ids": forward_sample_ids,
        "oracle_requested_labels": oracle_labels,
        "oracle_path_ids": oracle_path_ids,
        "oracle_sample_ids": oracle_sample_ids,
    }
    anchors = np.asarray([0, 32, 64, 96, 128], dtype=np.int64)
    null_trajectory = np.stack([null_prior.anchors[int(value)] for value in anchors], axis=1)
    learned_trajectory = np.stack(
        [learned_prior.anchors[int(value)] for value in anchors], axis=1
    )
    _write_npz(
        run_dir / "start_banks.npz",
        prior_starts=prior_starts,
        prior_requested_labels=prior_labels,
        prior_path_ids=prior_path_ids,
        prior_sample_ids=prior_sample_ids,
        forward_terminal_starts=null_forward.starts,
        oracle_starts=null_oracle.starts,
    )
    _write_npz(
        run_dir / "populations.npz",
        **population_rows,
        **demixed_rows,
        **identity_rows,
        prior_completed_steps=anchors,
        null_prior_trajectories=null_trajectory,
        learned_prior_trajectories=learned_trajectory,
    )
    rendered_rows = {
        name: _rasterize_rows(demixed_rows[name + "_demixed"])
        for name in population_rows
    }
    _write_npz(run_dir / "uint8_populations.npz", **rendered_rows, **identity_rows)
    telemetry = [
        *_telemetry_rows("null-prior", null_prior),
        *_telemetry_rows("learned-prior", learned_prior),
        *_telemetry_rows("null-forward-terminal", null_forward),
        *_telemetry_rows("learned-forward-terminal", learned_forward),
        *_telemetry_rows("oracle", oracle),
    ]
    _write_csv(run_dir / "telemetry.csv", telemetry)
    for name, images, labels in (
        ("prior_null", rendered_rows["null_prior"], prior_labels),
        ("prior_learned", rendered_rows["learned_prior"], prior_labels),
        ("forward_null", rendered_rows["null_forward_terminal"], forward_labels),
        ("forward_learned", rendered_rows["learned_forward_terminal"], forward_labels),
        ("oracle", rendered_rows["oracle"], oracle_labels),
    ):
        write_contact_sheet(
            run_dir / f"images/{name}.png",
            images,
            columns=10,
            captions=[str(int(label)) for label in labels],
        )
    _status(run_dir, "populations_written")
    seal_populations(run_dir)

    test_u8, test_y = _run_stage(
        run_dir,
        "terminal_open_and_load",
        lambda: open_terminal_evidence(run_dir, arff),
        preserve_terminal_reserve=False,
    )
    # The frozen benchmark evaluator is deliberately replayed on CPU so read-only
    # verification is portable and bit-stable across CUDA hosts.
    evaluator = _load_bound_evaluator(run_dir, device="cpu")
    validation_eval, test_eval = _run_stage(
        run_dir,
        "terminal_evaluator_real_health",
        lambda: (
            evaluate_image_classifier(
                evaluator,
                validation_u8.astype(np.float32)[:, None] / np.float32(255),
                validation_y,
                batch_size=256,
                device="cpu",
            ),
            evaluate_image_classifier(
                evaluator,
                test_u8.astype(np.float32)[:, None] / np.float32(255),
                test_y,
                batch_size=256,
                device="cpu",
            ),
        ),
        preserve_terminal_reserve=False,
    )
    evaluator_health = {
        "validation_accuracy": float(validation_eval["accuracy"]),
        "validation_loss": float(validation_eval["loss"]),
        "test_accuracy": float(test_eval["accuracy"]),
        "test_loss": float(test_eval["loss"]),
        "minimum_real_accuracy": 0.97,
        "qualified_under_original_threshold": int(
            float(validation_eval["accuracy"]) >= 0.97
        ),
        "finite_metrics": int(
            all(
                math.isfinite(value)
                for value in (
                    float(validation_eval["accuracy"]),
                    float(validation_eval["loss"]),
                    float(test_eval["accuracy"]),
                    float(test_eval["loss"]),
                )
            )
        ),
        "accepted_for_exploratory_use": 1,
    }
    _require(evaluator_health["finite_metrics"] == 1, "bound evaluator metrics are nonfinite")
    learned_generation, null_generation = _run_stage(
        run_dir,
        "terminal_prior_generation_metrics",
        lambda: (
            compute_generation_metrics(
                evaluator,
                rendered_rows["learned_prior"],
                prior_labels,
                prior_sample_ids,
                real_reference_images=test_u8,
                real_reference_labels=test_y,
                train_images=train_u8,
                test_images=test_u8,
                batch_size=256,
                device="cpu",
            ),
            compute_generation_metrics(
                evaluator,
                rendered_rows["null_prior"],
                prior_labels,
                prior_sample_ids,
                real_reference_images=test_u8,
                real_reference_labels=test_y,
                train_images=train_u8,
                test_images=test_u8,
                batch_size=256,
                device="cpu",
            ),
        ),
        preserve_terminal_reserve=False,
    )
    learned_generation["exact_reference_match_count"]["validation"] = _exact_match_count(
        rendered_rows["learned_prior"], validation_u8
    )
    null_generation["exact_reference_match_count"]["validation"] = _exact_match_count(
        rendered_rows["null_prior"], validation_u8
    )
    forward_metrics = _paired_endpoint_metrics(
        null_forward.final_states, learned_forward.final_states, forward_targets
    )
    null_forward_classifier, learned_forward_classifier = _run_stage(
        run_dir,
        "terminal_forward_classifier_metrics",
        lambda: (
            evaluate_generated_labels(
                evaluator,
                rendered_rows["null_forward_terminal"],
                forward_labels,
                forward_sample_ids,
                device="cpu",
            ),
            evaluate_generated_labels(
                evaluator,
                rendered_rows["learned_forward_terminal"],
                forward_labels,
                forward_sample_ids,
                device="cpu",
            ),
        ),
        preserve_terminal_reserve=False,
    )
    forward_metrics["null_classifier"] = null_forward_classifier
    forward_metrics["learned_classifier"] = learned_forward_classifier
    oracle_metrics = _paired_endpoint_metrics(
        null_oracle.final_states, oracle.final_states, oracle_targets
    )
    gate_c = int(
        int(oracle_metrics["learned_l1_win_count"])
        >= int(FROZEN_CONFIG["diagnostics"]["oracle_minimum_improved_paths"])
        and float(oracle_metrics["mean_learned_l1"])
        < float(oracle_metrics["mean_null_l1"])
    )
    metrics = {
        "gates": {
            "gate_a_passed": int(_read_json(run_dir / "theory_code_identity.json")["passed"]),
            "gate_b_passed": int(
                _read_json(run_dir / "kernel_audit.json")["passed"]
                and _read_json(run_dir / "k128_k512_audit.json")["passed"]
            ),
            "gate_c_passed": gate_c,
        },
        "evaluator_health": evaluator_health,
        "prior": {"learned": learned_generation, "null": null_generation},
        "forward_terminal": forward_metrics,
        "oracle": oracle_metrics,
        "fixed_render": {
            name: {
                "minimum": int(np.min(values)),
                "maximum": int(np.max(values)),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
            }
            for name, values in rendered_rows.items()
        },
    }
    _write_json(run_dir / "metrics.json", metrics)
    bindings = _read_json(run_dir / "source_bindings.json")
    _write_json(
        run_dir / "contextual_ddpm_metrics.json",
        _read_json(Path(bindings["ddpm_run_dir"]) / "evaluation/metrics.json"),
    )
    _run_stage(
        run_dir,
        "terminal_review_bundle",
        lambda: create_review_bundle(run_dir),
        preserve_terminal_reserve=False,
    )
    return finalize_and_verify(run_dir)


def _validate_real_pilot_authority(
    pilot_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Reject coordinated pilot bytes that are not bound to a real admitted run."""

    scientific = copy.deepcopy(dict(config))
    device = scientific.pop("device", None)
    run_scope = scientific.pop("run_scope", None)
    parent_source_hash = scientific.pop("parent_source_bindings_sha256", None)
    parent_config_hash = scientific.pop("parent_config_sha256", None)
    _require(isinstance(device, str) and bool(device.strip()), "objective pilot device is missing")
    try:
        pilot_device = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise EulerianJacobiDDPMRunError("objective pilot device is invalid") from error
    _require(
        run_scope == "objective_pilot"
        and pilot_device.type == "cuda"
        and _semantic_sha256(scientific) == _semantic_sha256(FROZEN_CONFIG),
        "objective pilot does not contain the full frozen scientific config",
    )
    authority_dir = pilot_dir / "authority"
    names = (
        "config.json",
        "source_bindings.json",
        "environment.json",
        "theory_code_identity.json",
        "kernel_audit.json",
        "k128_k512_audit.json",
        "preflight_projection.json",
        "path_id_audit.json",
    )
    _require(
        authority_dir.is_dir() and all((authority_dir / name).is_file() for name in names),
        "objective pilot authority bundle is missing",
    )
    parent_config_path = authority_dir / "config.json"
    source_path = authority_dir / "source_bindings.json"
    _require(
        _file_sha256(parent_config_path) == parent_config_hash
        and _file_sha256(source_path) == parent_source_hash,
        "objective pilot parent authority hash changed",
    )
    parent_config = _read_json(parent_config_path)
    parent_scientific = dict(parent_config)
    execution = parent_scientific.pop("execution_authority", None)
    _require(
        _semantic_sha256(parent_scientific) == _semantic_sha256(FROZEN_CONFIG)
        and isinstance(execution, dict)
        and execution.get("whole_run_restart_only") == 1
        and isinstance(execution.get("device"), str)
        and re.fullmatch(r"cuda(?::\d+)?", str(execution["device"])) is not None
        and str(execution.get("device")) == str(device)
        and float(execution.get("maximum_active_seconds", 0.0)) > 0.0
        and float(execution.get("maximum_storage_mib", 0.0)) > 0.0
        and 0.0 < float(execution.get("maximum_cuda_fraction", 0.0)) <= 1.0,
        "objective pilot execution authority changed",
    )
    source = _read_json(source_path)
    repository = Path(source["repository_root"])
    arff = Path(source["arff"])
    _require(
        repository.is_dir()
        and arff.is_file()
        and source["source_files"] == _source_hashes(repository)
        and source["arff_sha256"] == MNIST_ARFF_SHA256 == _file_sha256(arff)
        and source["config_sha256"] == _semantic_sha256(parent_config)
        and source["evaluator_checkpoint_sha256"]
        == ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256
        and source["evaluator_selection_sha256"]
        == ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256
        and source["ddpm_metrics_sha256"] == ACCEPTED_DDPM_METRICS_SHA256
        and source["ddpm_manifest_sha256"] == ACCEPTED_DDPM_MANIFEST_SHA256
        and source["ddpm_status"] == "complete",
        "objective pilot source/data binding changed",
    )
    environment = _read_json(authority_dir / "environment.json")
    theory = _read_json(authority_dir / "theory_code_identity.json")
    kernel = _read_json(authority_dir / "kernel_audit.json")
    paired = _read_json(authority_dir / "k128_k512_audit.json")
    projection = _read_json(authority_dir / "preflight_projection.json")
    path_audit = _read_json(authority_dir / "path_id_audit.json")
    _require(
        str(environment.get("device")) == str(device)
        and int(environment.get("cuda_available", 0)) == 1
        and theory.get("passed") == 1
        and kernel.get("passed") == 1
        and paired.get("passed") == 1
        and paired.get("admission_capable") == 1
        and paired.get("backend") == "real-28x28-fast-jacobi-and-oracle-controller"
        and paired.get("aligned_transition_randomness_coupled") == 1
        and paired.get("full_path_common_random_numbers_claimed") == 0
        and paired.get("paired_initial_states") == 1
        and paired.get("path_ids") == [0xB2100]
        and paired.get("path_ids_sha256")
        == FROZEN_CONFIG["path_ids"]["preflight_k128_k512"]["sha256"]
        and projection.get("admission", {}).get("passed") == 1
        and projection.get("full", {}).get("cache_pair_transitions") == 1_756_160_000
        and projection.get("full", {}).get("sampling_transition_work") == 607_631_360
        and projection.get("full", {}).get("base_transition_work") == 2_363_791_360
        and path_audit.get("passed") == 1,
        "objective pilot preflight authority changed",
    )

    selection = _read_json(pilot_dir / "training_selection.json")
    checkpoint_path = pilot_dir / "selected_checkpoint.pt"
    _require(
        selection.get("checkpoint_sha256") == _file_sha256(checkpoint_path)
        and int(selection.get("completed_updates", -1)) == 750,
        "objective pilot checkpoint selection binding changed",
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict), "objective pilot checkpoint payload changed")
    state = payload.get("state_dict")
    expected_state = core.make_model().state_dict()
    _require(
        isinstance(state, dict)
        and set(state) == set(expected_state)
        and all(
            isinstance(state[name], torch.Tensor)
            and state[name].shape == expected_state[name].shape
            and bool(torch.isfinite(state[name]).all())
            for name in expected_state
        )
        and int(payload.get("completed_updates", -1)) == 750
        and int(payload.get("selected_update", 0)) > 0
        and int(payload["selected_update"]) == int(selection["selected_update"])
        and float(payload.get("selected_validation_normalized_mse", float("nan")))
        == float(selection["selected_validation_normalized_mse"])
        and math.isfinite(float(payload.get("training_target_energy", float("nan"))))
        and float(payload["training_target_energy"]) > 0.0,
        "objective pilot selected model changed",
    )
    with (pilot_dir / "training_history.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        history = list(csv.DictReader(handle))
    eligible = [
        (int(row["update"]), float(row["validation_normalized_mse"]))
        for row in history
        if int(row["update"]) > 0
        and int(row["eligible"]) == 1
        and math.isfinite(float(row["validation_normalized_mse"]))
    ]
    _require(bool(eligible), "objective pilot training history has no eligible checkpoint")
    expected_update, expected_mse = min(eligible, key=lambda row: (row[1], row[0]))
    _require(
        expected_update == int(selection["selected_update"])
        and expected_mse == float(selection["selected_validation_normalized_mse"]),
        "objective pilot selected checkpoint is not the prespecified argmin",
    )
    with np.load(pilot_dir / "data_roles.npz", allow_pickle=False) as archive:
        roles = {name: np.asarray(archive[name]) for name in archive.files}
    _require(
        np.array_equal(roles["train_path_ids"], _role_path_ids("pilot_train"))
        and np.array_equal(
            roles["validation_path_ids"], _role_path_ids("pilot_validation")
        )
        and roles["train_arff_indices"].shape == (250,)
        and roles["validation_arff_indices"].shape == (100,)
        and np.array_equal(
            np.bincount(roles["train_labels"].astype(np.int64), minlength=10),
            np.full(10, 25),
        )
        and np.array_equal(
            np.bincount(roles["validation_labels"].astype(np.int64), minlength=10),
            np.full(10, 10),
        ),
        "objective pilot data-role inventory changed",
    )
    train_u8, train_y, validation_u8, validation_y = load_train_validation_mnist(arff)
    train_indices = roles["train_arff_indices"].astype(np.int64)
    validation_indices = roles["validation_arff_indices"].astype(np.int64) - 55_000
    _require(
        np.array_equal(train_y[train_indices], roles["train_labels"])
        and np.array_equal(validation_y[validation_indices], roles["validation_labels"])
        and np.array_equal(
            train_indices,
            core.balanced_class_indices(train_y, per_class=25, start=0, stop=len(train_y)),
        )
        and np.array_equal(
            validation_indices,
            core.balanced_class_indices(
                validation_y, per_class=10, start=0, stop=len(validation_y)
            ),
        ),
        "objective pilot ARFF allocation changed",
    )
    with np.load(pilot_dir / "forward_records.npz", allow_pickle=False) as records:
        _require(
            records["train_later_states"].shape == (1_000, 784)
            and records["train_targets"].shape == (1_000, 392)
            and records["validation_later_states"].shape == (400, 784)
            and records["validation_targets"].shape == (400, 392)
            and np.isfinite(records["train_later_states"]).all()
            and np.isfinite(records["train_targets"]).all()
            and np.isfinite(records["validation_later_states"]).all()
            and np.isfinite(records["validation_targets"]).all(),
            "objective pilot forward-record cache changed",
        )
    return {
        "passed": 1,
        "source_bindings_sha256": parent_source_hash,
        "parent_config_sha256": parent_config_hash,
    }


def validate_objective_pilot(pilot_dir: Path) -> dict[str, Any]:
    """Verify the fixed ten-class pilot and compute its diagnostic scale admission."""

    pilot_dir = Path(pilot_dir).resolve()
    _require(pilot_dir.is_dir(), "objective pilot directory is missing")
    _require(
        (pilot_dir / "artifact_manifest.json").is_file()
        and (pilot_dir / "SHA256SUMS.txt").is_file(),
        "objective pilot integrity seal is missing",
    )
    manifest = _verify_manifest(pilot_dir)
    _population_seal(pilot_dir)
    _validate_population_semantics(pilot_dir)
    config = _read_json(pilot_dir / "config.json")
    pilot = config.get("objective_pilot", config)
    frozen = FROZEN_CONFIG["objective_pilot"]
    for key in (
        "all_ten_classes_required",
        "train_paths",
        "train_paths_per_class",
        "validation_paths",
        "validation_paths_per_class",
        "training_updates",
        "prior_paths",
        "prior_paths_per_class",
        "forward_terminal_paths",
        "forward_terminal_paths_per_class",
        "oracle_paths",
        "oracle_paths_per_class",
    ):
        _require(pilot.get(key) == frozen[key], f"objective pilot config changed: {key}")
    _validate_real_pilot_authority(pilot_dir, config)
    saved_metrics = _read_json(pilot_dir / "metrics.json")
    with np.load(pilot_dir / "populations.npz", allow_pickle=False) as archive:
        required = {
            "null_prior",
            "learned_prior",
            "prior_requested_labels",
            "prior_path_ids",
            "prior_sample_ids",
            "null_forward_terminal",
            "learned_forward_terminal",
            "forward_targets",
            "forward_requested_labels",
            "forward_path_ids",
            "forward_sample_ids",
            "null_oracle",
            "oracle",
            "oracle_targets",
            "oracle_requested_labels",
            "oracle_path_ids",
            "oracle_sample_ids",
        }
        _require(required.issubset(archive.files), "pilot raw population schema is incomplete")
        arrays = {name: np.asarray(archive[name]) for name in required}
    prior_labels = np.asarray(arrays["prior_requested_labels"], dtype=np.int64)
    forward_labels = np.asarray(arrays["forward_requested_labels"], dtype=np.int64)
    oracle_labels = np.asarray(arrays["oracle_requested_labels"], dtype=np.int64)
    _require(
        np.array_equal(np.bincount(prior_labels, minlength=10), np.full(10, 2))
        and np.array_equal(np.bincount(forward_labels, minlength=10), np.full(10, 2))
        and np.array_equal(np.bincount(oracle_labels, minlength=10), np.ones(10, dtype=np.int64)),
        "pilot populations are not exactly class balanced",
    )
    for prefix, role in (
        ("prior", "pilot_prior"),
        ("forward", "pilot_forward_terminal"),
        ("oracle", "pilot_oracle"),
    ):
        ids = np.asarray(arrays[f"{prefix}_path_ids"], dtype=np.int64)
        row = FROZEN_CONFIG["path_ids"][role]
        expected_ids = np.arange(row["start"], row["stop_exclusive"], dtype=np.int64)
        sample_ids = np.asarray(arrays[f"{prefix}_sample_ids"])
        _require(
            np.array_equal(ids, expected_ids)
            and sample_ids.shape == ids.shape
            and len(set(str(value) for value in sample_ids)) == ids.size,
            f"pilot {prefix} path/sample IDs changed",
        )
    _require(
        np.asarray(arrays["null_prior"]).shape
        == np.asarray(arrays["learned_prior"]).shape
        == (20, 784),
        "pilot prior arrays are malformed",
    )
    null_forward = np.asarray(arrays["null_forward_terminal"], dtype=np.float64)
    learned_forward = np.asarray(arrays["learned_forward_terminal"], dtype=np.float64)
    forward_targets = np.asarray(arrays["forward_targets"], dtype=np.float64)
    _require(
        null_forward.shape == learned_forward.shape == forward_targets.shape == (20, 784),
        "pilot forward-terminal arrays are malformed",
    )
    null_l1 = np.sum(np.abs(null_forward - forward_targets), axis=1, dtype=np.float64)
    learned_l1 = np.sum(np.abs(learned_forward - forward_targets), axis=1, dtype=np.float64)
    wins = int(np.sum(learned_l1 < null_l1))
    null_total = float(np.sum(null_l1, dtype=np.float64))
    _require(null_total > 0.0 and math.isfinite(null_total), "pilot null forward L1 is invalid")
    relative = (null_total - float(np.sum(learned_l1, dtype=np.float64))) / null_total
    null_oracle = np.asarray(arrays["null_oracle"], dtype=np.float64)
    oracle = np.asarray(arrays["oracle"], dtype=np.float64)
    oracle_targets = np.asarray(arrays["oracle_targets"], dtype=np.float64)
    _require(
        null_oracle.shape == oracle.shape == oracle_targets.shape == (10, 784),
        "pilot oracle arrays are malformed",
    )
    null_oracle_l1 = np.sum(np.abs(null_oracle - oracle_targets), axis=1, dtype=np.float64)
    oracle_l1 = np.sum(np.abs(oracle - oracle_targets), axis=1, dtype=np.float64)
    oracle_wins = int(np.sum(oracle_l1 < null_oracle_l1))
    gate_c = int(
        oracle_wins >= int(FROZEN_CONFIG["diagnostics"]["oracle_minimum_improved_paths"])
        and float(np.sum(oracle_l1)) < float(np.sum(null_oracle_l1))
    )
    state_names = (
        "null_prior",
        "learned_prior",
        "null_forward_terminal",
        "learned_forward_terminal",
        "null_oracle",
        "oracle",
        "forward_targets",
        "oracle_targets",
    )
    health = int(
        all(
            np.asarray(arrays[name]).ndim == 2
            and np.asarray(arrays[name]).shape[1] == 784
            and np.isfinite(arrays[name]).all()
            and np.all(arrays[name] >= 0.0)
            and float(np.max(np.abs(np.sum(arrays[name], axis=1) - 1.0))) <= 2e-12
            for name in state_names
        )
    )
    with (pilot_dir / "telemetry.csv").open("r", encoding="utf-8", newline="") as handle:
        telemetry = list(csv.DictReader(handle))
    expected_telemetry = {
        (population, str(quartile))
        for population in (
            "null-prior",
            "learned-prior",
            "null-forward-terminal",
            "learned-forward-terminal",
            "oracle",
        )
        for quartile in range(4)
    }
    paths_by_population = {
        "null-prior": 20,
        "learned-prior": 20,
        "null-forward-terminal": 20,
        "learned-forward-terminal": 20,
        "oracle": 10,
    }
    _require(
        len(telemetry) == len(expected_telemetry)
        and {(str(row.get("population")), str(row.get("time_quarter"))) for row in telemetry}
        == expected_telemetry,
        "pilot telemetry inventory changed",
    )
    _require(
        all(
            row.get("controller_rms") not in {None, ""}
            and math.isfinite(float(row["controller_rms"]))
            and float(row["controller_rms"]) >= 0.0
            and int(row.get("score_count", -1))
            == paths_by_population[str(row["population"])] * 175_616
            and int(row.get("finite", 0)) == 1
            and int(row.get("nonnegative", 0)) == 1
            and int(row.get("microsteps", 0)) == 2
            and math.isfinite(float(row.get("maximum_mass_error", "nan")))
            and float(row["maximum_mass_error"]) <= 2e-12
            and math.isfinite(float(row.get("maximum_pair_total_error", "nan")))
            and float(row["maximum_pair_total_error"]) <= 2e-12
            and 0 <= int(row.get("exact_facet_count", -1))
            <= paths_by_population[str(row["population"])] * 784
            for row in telemetry
        ),
        "pilot telemetry contains invalid counts or health values",
    )
    telemetry_health = 1
    learned_rms_values = [
        float(row["controller_rms"])
        for row in telemetry
        if str(row.get("population")).startswith("learned-")
    ]
    controller_rms = (
        math.sqrt(float(np.mean(np.square(learned_rms_values))))
        if learned_rms_values and all(math.isfinite(value) for value in learned_rms_values)
        else float("nan")
    )
    prior_task_signal = _validate_pilot_prior_classifier(pilot_dir)
    recomputed_metrics = {
        "gates": {
            "gate_c_passed": gate_c,
            "health_passed": int(health == 1 and telemetry_health == 1),
        },
        "forward_terminal": {
            "learned_l1_win_count": wins,
            "aggregate_l1_relative_improvement": relative,
        },
        "oracle": {"l1_win_count": oracle_wins},
        "controller": {"learned_rms": controller_rms},
        "prior_task_signal": prior_task_signal,
    }
    _require(
        _semantic_sha256(saved_metrics) == _semantic_sha256(recomputed_metrics),
        "pilot metrics changed or do not match raw evidence",
    )
    dynamics_admitted = int(
        gate_c == 1
        and health == 1
        and wins >= int(frozen["minimum_forward_l1_wins"])
        and math.isfinite(relative)
        and relative >= float(frozen["minimum_aggregate_l1_relative_improvement"])
        and math.isfinite(controller_rms)
        and controller_rms > 0.0
    )
    admitted = int(
        dynamics_admitted == 1 and int(prior_task_signal["passed"]) == 1
    )
    if health != 1:
        route = "pilot_invalid_health_repair_before_interpretation"
    elif gate_c != 1:
        route = "repair_reverse_composition_before_judging_learner"
    elif dynamics_admitted == 1 and int(prior_task_signal["passed"]) != 1:
        route = "pilot_prior_negative_stop_before_scale"
    elif admitted:
        route = "full_scale_admitted"
    else:
        route = "pilot_negative_stop_before_scale"
    for relative_path in (
        "populations.npz",
        "uint8_populations.npz",
        "telemetry.csv",
        "selected_checkpoint.pt",
        "prior_classifier_outputs.npz",
        "prior_classifier_metrics.json",
    ):
        _require((pilot_dir / relative_path).is_file(), f"pilot artifact is missing: {relative_path}")
    with np.load(pilot_dir / "uint8_populations.npz", allow_pickle=False) as rendered:
        image_rows = (
            "null_prior",
            "learned_prior",
            "null_forward_terminal",
            "learned_forward_terminal",
            "forward_targets",
            "null_oracle",
            "oracle",
            "oracle_targets",
        )
        _require(all(name in rendered for name in image_rows), "pilot uint8 rows are incomplete")
        for name in image_rows:
            _require(
                np.array_equal(rendered[name], _demix_and_rasterize_rows(arrays[name])),
                f"pilot fixed rasterization changed: {name}",
            )
        for prefix in ("prior", "forward", "oracle"):
            for suffix in ("requested_labels", "path_ids", "sample_ids"):
                name = f"{prefix}_{suffix}"
                _require(
                    name in rendered and np.array_equal(rendered[name], arrays[name]),
                    f"pilot rendered identity changed: {name}",
                )
    return {
        "schema": VERSION + "-objective-pilot-admission",
        "integrity_passed": 1,
        "gate_c_passed": gate_c,
        "health_passed": health,
        "learned_l1_win_count": wins,
        "aggregate_l1_relative_improvement": relative,
        "learned_controller_rms": controller_rms,
        "prior_task_signal": prior_task_signal,
        "full_scale_admitted": admitted,
        "route": route,
        "artifact_count": int(manifest["artifact_count"]),
        "tree_digest": _directory_digest(pilot_dir),
    }


def require_pilot_for_full_run(
    pilot_dir: Path, parent_run_dir: Path | None = None
) -> dict[str, Any]:
    """Fail closed unless the immutable objective pilot buys the full-scale decision."""

    try:
        result = validate_objective_pilot(pilot_dir)
    except EulerianJacobiDDPMRunError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise EulerianJacobiDDPMRunError(f"objective pilot validation failed: {error}") from error
    _require(
        result["full_scale_admitted"] == 1,
        "objective pilot was negative; stop before full scale and preserve its outputs",
    )
    if parent_run_dir is not None:
        parent_run_dir = Path(parent_run_dir).resolve()
        pilot_dir = Path(pilot_dir).resolve()
        _require(
            pilot_dir == _embedded_pilot_dir(parent_run_dir),
            "full scale requires the objective pilot embedded in its run tree",
        )
        pilot_source = _read_json(pilot_dir / "authority/source_bindings.json")
        parent_source = _read_json(parent_run_dir / "source_bindings.json")
        pilot_environment = _read_json(pilot_dir / "authority/environment.json")
        parent_environment = _read_json(parent_run_dir / "environment.json")
        _require(
            pilot_source["source_files"] == parent_source["source_files"]
            and pilot_source["arff_sha256"] == parent_source["arff_sha256"]
            and pilot_source["evaluator_checkpoint_sha256"]
            == parent_source["evaluator_checkpoint_sha256"]
            and pilot_environment["device"] == parent_environment["device"],
            "objective pilot is not bound to this full-run source/data/device",
        )
    return result


def _write_pilot_admission_authority(
    run_dir: Path,
    pilot_dir: Path,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the compact pilot decision authority into the parent run."""

    run_dir = Path(run_dir).resolve()
    pilot_dir = Path(pilot_dir).resolve()
    _require(
        pilot_dir == _embedded_pilot_dir(run_dir)
        and admission.get("pilot_directory") == EMBEDDED_PILOT_DIRECTORY,
        "pilot admission authority requires the embedded objective pilot",
    )
    sources = {
        "artifact_manifest.json": pilot_dir / "artifact_manifest.json",
        "config.json": pilot_dir / "config.json",
        "metrics.json": pilot_dir / "metrics.json",
        "status.json": pilot_dir / "status.json",
        "source_bindings.json": pilot_dir / "authority/source_bindings.json",
        "environment.json": pilot_dir / "authority/environment.json",
    }
    _require(all(path.is_file() for path in sources.values()), "pilot admission authority is incomplete")
    authority_dir = run_dir / "pilot_admission_authority"
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        destination = authority_dir / name
        _atomic_replace(
            destination,
            lambda temporary, source=source: shutil.copyfile(source, temporary),
        )
        hashes[name] = _file_sha256(destination)
    record = {
        "schema": VERSION + "-pilot-admission-authority",
        "pilot_directory": EMBEDDED_PILOT_DIRECTORY,
        "pilot_tree_digest": str(admission["tree_digest"]),
        "admission": dict(admission),
        "copied_file_sha256": hashes,
    }
    _write_json(authority_dir / "record.json", record)
    return record


def _validate_full_pilot_admission(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    admission = _read_json(run_dir / "objective_pilot_admission.json")
    _require(
        admission.get("integrity_passed") == 1
        and admission.get("gate_c_passed") == 1
        and admission.get("health_passed") == 1
        and admission.get("full_scale_admitted") == 1
        and admission.get("route") == "full_scale_admitted",
        "full run lacks a positive objective-pilot admission",
    )
    pilot_dir = _embedded_pilot_dir(run_dir, admission)
    result = require_pilot_for_full_run(pilot_dir, run_dir)
    expected_admission = dict(result) | {
        "pilot_directory": EMBEDDED_PILOT_DIRECTORY
    }
    _require(
        _semantic_sha256(admission) == _semantic_sha256(expected_admission),
        "full-run objective-pilot admission changed",
    )
    authority_dir = run_dir / "pilot_admission_authority"
    record = _read_json(authority_dir / "record.json")
    sources = {
        "artifact_manifest.json": pilot_dir / "artifact_manifest.json",
        "config.json": pilot_dir / "config.json",
        "metrics.json": pilot_dir / "metrics.json",
        "status.json": pilot_dir / "status.json",
        "source_bindings.json": pilot_dir / "authority/source_bindings.json",
        "environment.json": pilot_dir / "authority/environment.json",
    }
    copied_hashes = record.get("copied_file_sha256")
    _require(
        record.get("pilot_directory") == EMBEDDED_PILOT_DIRECTORY
        and record.get("pilot_tree_digest") == result["tree_digest"]
        and _semantic_sha256(record.get("admission")) == _semantic_sha256(admission)
        and isinstance(copied_hashes, dict)
        and all(
            source.is_file()
            and (authority_dir / name).is_file()
            and _file_sha256(source) == _file_sha256(authority_dir / name)
            == copied_hashes.get(name)
            for name, source in sources.items()
        ),
        "compact objective-pilot authority changed",
    )
    return result


def run_production(args: argparse.Namespace) -> int:
    """Run admitted CUDA preflight, pilot routing, and conditional full experiment."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    repository_root = Path(__file__).resolve().parents[1]
    run_dir = initialize_run(
        repository_root,
        Path(args.arff),
        Path(args.ddpm_run_dir),
        Path(args.run_dir),
        device=args.device,
        maximum_active_seconds=args.max_active_seconds,
        maximum_storage_mib=args.max_storage_mib,
        maximum_cuda_fraction=args.max_cuda_fraction,
        approval_id=args.approval_id,
    )
    try:
        run_numerical_preflight(run_dir)
        train_u8, train_y, validation_u8, validation_y = _run_stage(
            run_dir,
            "load_authenticated_development_data",
            lambda: load_train_validation_mnist(Path(args.arff)),
        )
        run_resource_preflight(
            run_dir,
            train_u8,
            train_y,
            validation_u8,
            validation_y,
            device=args.device,
        )
        pilot_dir = _embedded_pilot_dir(run_dir)
        pilot_admission = execute_objective_pilot(
            run_dir,
            pilot_dir,
            train_u8,
            train_y,
            validation_u8,
            validation_y,
            device=args.device,
        )
        admission_record = dict(pilot_admission) | {
            "pilot_directory": EMBEDDED_PILOT_DIRECTORY
        }
        _write_json(run_dir / "objective_pilot_admission.json", admission_record)
        _write_pilot_admission_authority(run_dir, pilot_dir, admission_record)
        if int(pilot_admission["full_scale_admitted"]) != 1:
            pilot_route = str(pilot_admission["route"])
            if pilot_route == "pilot_negative_stop_before_scale":
                interpretation = (
                    "The verifier-clean pilot was healthy but missed the learned-dynamics "
                    "routing thresholds; stop before scale and conduct the strategy review."
                )
                exit_code = 0
            elif pilot_route == "pilot_prior_negative_stop_before_scale":
                interpretation = (
                    "Forward-terminal dynamics passed, but the paired learned prior row "
                    "missed the prespecified classifier task-signal thresholds; stop before "
                    "scale and conduct the strategy review."
                )
                exit_code = 0
            elif pilot_route == "repair_reverse_composition_before_judging_learner":
                interpretation = (
                    "The oracle interface control failed; repair reverse composition "
                    "before making any learner judgment."
                )
                exit_code = 3
            else:
                interpretation = (
                    "Pilot numerical health failed; the run is invalid for scientific "
                    "interpretation and the localized health defect must be repaired."
                )
                exit_code = 3
            _write_text(
                run_dir / "REPORT.md",
                "# Eulerian Jacobi DDPM-v0 pilot stop\n\n"
                "Primary mode: exploratory. The all-ten-class objective pilot completed, "
                "and the 2.36-billion-transition full stage was not launched.\n\n"
                f"Route: `{pilot_route}`. {interpretation}\n\n"
                f"Pilot admission: `{pilot_admission}`.\n\n"
                "This establishes only the stated pilot route for the frozen K=128 "
                "system. It does not establish universal learner or Eulerian failure. "
                "All null, learned, "
                "oracle, prior, forward-terminal, and image artifacts were retained.\n",
            )
            _write_text(
                run_dir / "HANDOFF.md",
                "# Eulerian Jacobi DDPM-v0 pilot-negative handoff\n\n"
                f"Nearest objective-bearing milestone was executed. Route: `{pilot_route}`. "
                f"{interpretation}\n\n"
                "Proxy-only patches since the last objective-bearing experiment: 0.\n",
            )
            _status(run_dir, pilot_route)
            _seal_manifest(run_dir)
            result = verify_run(run_dir)
            print(json.dumps(result | {"route": pilot_admission["route"]}, sort_keys=True))
            return exit_code
        require_post_pilot_full_resources(run_dir)
        result = execute_full_experiment(
            run_dir,
            Path(args.arff),
            train_u8,
            train_y,
            validation_u8,
            validation_y,
            device=args.device,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except ResourceStop as error:
        completed_pilot = (
            _read_json(run_dir / "objective_pilot_admission.json")
            if (run_dir / "objective_pilot_admission.json").is_file()
            else None
        )
        record = {
            "schema": VERSION + "-resource-stop",
            "message": str(error),
            "scientific_result": (
                None if completed_pilot is None else completed_pilot.get("route")
            ),
            "terminal_test_opened": int((run_dir / "TERMINAL_EVIDENCE_OPENED.json").is_file()),
            "recorded_at": _utc_now(),
        }
        _write_json(run_dir / "resource_stop.json", record)
        _write_text(
            run_dir / "REPORT.md",
            "# Eulerian Jacobi DDPM-v0 resource stop\n\n"
            "The measured workload exceeded an explicit approved cap, so execution "
            "stopped without changing the frozen scientific design. This is not a "
            "scientific negative. See `resource_stop.json` and `resource_ledger.json`.\n",
        )
        _status(run_dir, "resource_stopped", error=str(error))
        _seal_manifest(run_dir)
        result = verify_run(run_dir)
        print(json.dumps(result | {"resource_stop": str(error)}, sort_keys=True))
        return 2
    except Exception as error:
        completed_pilot = (
            _read_json(run_dir / "objective_pilot_admission.json")
            if (run_dir / "objective_pilot_admission.json").is_file()
            else None
        )
        failure = {
            "schema": VERSION + "-failure",
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "completed_pilot_route": (
                None if completed_pilot is None else completed_pilot.get("route")
            ),
            "population_sealed": int((run_dir / "POPULATIONS_SEALED.json").is_file()),
            "terminal_test_opened": int(
                (run_dir / "TERMINAL_EVIDENCE_OPENED.json").is_file()
            ),
            "recorded_at": _utc_now(),
        }
        _write_json(run_dir / "failure.json", failure)
        _write_text(
            run_dir / "REPORT.md",
            "# Eulerian Jacobi DDPM-v0 operational failure\n\n"
            f"Execution stopped with `{failure['error_type']}: {failure['message']}`. "
            "Existing outputs were retained and sealed into this failure tree. "
            "This is not a scientific negative; repair the localized defect and rerun "
            "the frozen whole experiment in a fresh directory.\n",
        )
        _status(run_dir, "failed", error=str(error))
        _seal_manifest(run_dir)
        result = verify_run(run_dir)
        print(json.dumps(result | {"failure": str(error)}, sort_keys=True))
        return 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke", help="run the synthetic CPU lifecycle smoke")
    smoke.add_argument("--output-dir", required=True)

    run = commands.add_parser("run", help="run the CUDA exploratory pilot/full workflow")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--arff", required=True)
    run.add_argument("--ddpm-run-dir", required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--approval-id", required=True)
    run.add_argument("--max-active-seconds", required=True, type=float)
    run.add_argument("--max-storage-mib", required=True, type=float)
    run.add_argument("--max-cuda-fraction", required=True, type=float)

    review = commands.add_parser("record-review", help="record a completed manual review")
    review.add_argument("--run-dir", required=True)
    review.add_argument("--answers", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--confirm-manual-review", action="store_true")

    verify = commands.add_parser("verify", help="verify a sealed run without compute")
    verify.add_argument("--run-dir", required=True)
    return parser


def _run_smoke(output_dir: Path) -> dict[str, Any]:
    run_dir = initialize_smoke_run(output_dir)
    execute_smoke_run(run_dir)
    seal_populations(run_dir)
    open_terminal_evidence(run_dir)
    create_review_bundle(run_dir)
    return finalize_and_verify(run_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(_run_smoke(Path(args.output_dir)), sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_run(Path(args.run_dir)), sort_keys=True))
        return 0
    if args.command == "record-review":
        outcome = record_human_review(
            Path(args.run_dir),
            Path(args.answers),
            args.reviewer,
            args.confirm_manual_review,
        )
        print(json.dumps(outcome, sort_keys=True))
        return 0
    return run_production(args)


if __name__ == "__main__":
    raise SystemExit(main())
