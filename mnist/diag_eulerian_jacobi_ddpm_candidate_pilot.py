from __future__ import annotations

"""Bounded exploratory K=128 candidate-backend Eulerian Jacobi objective pilot.

The production command is deliberately a single governed lifecycle.  It admits the
approximate candidate backend with a fixed numerical bank, runs a complete oracle
control, trains one frozen model, and saves paired task-level populations.  It never
launches the historical full-scale population and never parses or uses terminal
MNIST content rows; whole-file hashing is authority-only.
"""

import argparse
import ast
import copy
import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from fractions import Fraction
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
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from mnist import eulerian_jacobi_ddpm as core
from mnist import eulerian_jacobi_ddpm_candidate as candidate
from mnist import d0_jacobi_rb_cuda as certified_cuda
from mnist.conditioned_diffusion import SmallMnistCNN
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_cuda_deferred import (
    CandidateRBCudaBatch,
    enqueue_alpha1_rb_transition_batch_cuda_candidate,
)
from mnist.d0_jacobi_rb_learnability import matching_indices
from mnist.d0_jacobi_rb_spectral import (
    evaluate_alpha1_rb_torch_fixed_modes,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    canonical_refinement_transition_ids,
    refinement_phase_exposure,
)
from mnist.mnist_generation_benchmark import (
    MNIST_ARFF_SHA256,
    REVIEW_ASSIGNMENTS,
    evaluate_generated_labels,
    score_human_review,
    sha256_file,
    write_contact_sheet,
)


VERSION = "eulerian-jacobi-ddpm-candidate-k128-pilot-v2"
CANDIDATE_TARGET_SEMANTICS = "approximate-candidate Rao--Blackwell target"
AUDIT_RNG_CONTRACT = "philox4x32-10-canonical-transition-v2"
AUDIT_PAIRING_CONTRACT = "stateless-philox-v2-transition-id"
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
PROTECTED_DEFERRED_CUDA_SOURCE_SHA256 = (
    "8e41eacb48896c4e6fecbeb03b035fb8da6172cd17d5ad6b17436a4f9d0ff41a"
)
PLACEHOLDER = re.compile(
    r"(?:<[^>]+>|fresh[ _-]*approval|approval[ _-]*reference|placeholder|todo|tbd)",
    re.I,
)

ADDITIVE_SOURCE_FILES = (
    "mnist/eulerian_jacobi_ddpm_candidate.py",
    "mnist/diag_eulerian_jacobi_ddpm_candidate_pilot.py",
    "tests/test_eulerian_jacobi_ddpm_candidate.py",
    "docs/eulerian_jacobi_ddpm_candidate_pilot.md",
)

# The immutable 27-path inventory copied from fixed-grid-v0-retry1.  Production
# initialization rehashes every entry; it does not trust the historical receipt.
PROTECTED_SOURCE_HASHES: dict[str, str] = {
    "mnist/__init__.py": "1afbf919b879fc8c499db24009ce92e92ee03b198cfb427830a18df37df86ce4",
    "mnist/conditioned_diffusion.py": "96906c6c1cf7fed4de191e56d6861621446e65b6952171cbf6fa556303450892",
    "mnist/d0_jacobi_artifacts.py": "75bac4e947349993f6f7bdc3cf6df31e0861f67d8fbe7688d3761ee7d6325e21",
    "mnist/d0_jacobi_denoising.py": "a434b27daba832ae81de5e677b8c8482d30680e9cbd079f427fb7ceec1a47b39",
    "mnist/d0_jacobi_rb_absolute_coordinate.py": "5fbd05880e584bfe5fcfb5090955b1cf15db8f1dc3eb0395cc2c915d7c5a7183",
    "mnist/d0_jacobi_rb_boundary_tangent.py": "be4ab9ad8007e567bb518b98c04b37e4900669c53607ef6612f951d15d3a17ce",
    "mnist/d0_jacobi_rb_boundary_tangent_frequency1_coordinate.py": "bb67d8f44136e82647b881e8badd4f6b72382432874aff9fb30d65da21edbed4",
    "mnist/d0_jacobi_rb_boundary_tangent_v3_provenance.py": "c591d4047c6b3763247d56e7eedcff97d4c8e7d82fa28d5ec844ae142b59e4f6",
    "mnist/d0_jacobi_rb_boundary_tangent_zero_baseline.py": "5aa6fdfe7f6e23a92ef37fa86deeca317760c1ca90abdffee059bcf06b09235c",
    "mnist/d0_jacobi_rb_coarse_residual.py": "b3157a81cad5dcb257cb5deb09054b515e659023e44a4546edd61d58659826b3",
    "mnist/d0_jacobi_rb_controls.py": "3186c3321a4f48bda6b7a2a28a600812b7686d0b68aa499ca9fe6735bc7a7d17",
    "mnist/d0_jacobi_rb_cuda.py": "94b95db6c93510c97c36b7cd67b2dec3b1f13a62b3077299e6edd6b97f0ba97a",
    "mnist/d0_jacobi_rb_cuda_certificate.py": "f43bd0459a3200bbead706cf7def1cca17e344bbccd7ae4de5cfb26b1eb9aced",
    "mnist/d0_jacobi_rb_cuda_controls.py": "a834445afa5f4003931254a13fbe1e0838904bf9e47726abeaf3faa5955f01ff",
    "mnist/d0_jacobi_rb_cuda_fused.py": "184a3e9e8e476b835e808de4f1b5b7d641d33997448968539ab240a54f91204d",
    "mnist/d0_jacobi_rb_cuda_multipath.py": "5949dc794085cde340b42133a4a2102815ac85dece4e6799b23762de62507f77",
    "mnist/d0_jacobi_rb_global_dilated.py": "2ea368bc0d001803ce8e8c5f9862feefe01aa88ada395f0279636e8ce6e4135a",
    "mnist/d0_jacobi_rb_learnability.py": "081c9dfa7414c3c9fda80b262162eb3ad6c84ddaff905a058896580c1f1d50b2",
    "mnist/d0_jacobi_rb_reverse_controller.py": "adac975b5d64e23f7d0861dce0ac1b497fa054eed6db2a76358e462d94c8ee5f",
    "mnist/d0_jacobi_rb_spectral.py": "f16851db6f9b5f91cec5fc7ab1121461a4b915a63003dba88530b1a8a4f1b635",
    "mnist/d0_jacobi_rb_strang_refinement.py": "9ba9a12032fb9e4babc72568d5494380c5cc06a74c5495c81369b658fd048975",
    "mnist/d0_jacobi_source_compat.py": "f90ac705d105e03ca258f8507fa74e77e9cc2ef3cea2bf8615594cc5dc5c07ed",
    "mnist/d0_jacobi_v3_source_compat.py": "17e9f47c573944a1affcae43ee63bd057fc18fc54ea917fdbdd6fceecc0c6b8c",
    "mnist/diag_eulerian_jacobi_ddpm_mnist.py": "3f5a0f963bc4b2042a10e71f9290478b2ce27c4520913d219c98c03a807a2c9e",
    "mnist/eulerian_jacobi_ddpm.py": "5875373c34fa6fd4620749c5763ce91903728f7f0d2f70c6e7a65a1f8023ab98",
    "mnist/mnist_generation_benchmark.py": "2ebf13e37e03646222b8decde91034f10dff564fdc0d3f7a967af789ef3cbfd6",
    "mnist/weighted_point_cloud.py": "b70db19c8adbaf7cd89818a61a7dc8b167ec83e013911682702161c7e28fca7d",
}

AUDIT_TRAIN_POSITIONS = np.asarray(
    [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 1, 26, 51, 76, 101, 126],
    dtype=np.int64,
)
ORACLE_VALIDATION_POSITIONS = np.asarray(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int64
)
FORWARD_VALIDATION_POSITIONS = np.asarray(
    [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71, 80, 81, 90, 91],
    dtype=np.int64,
)

PATH_RANGES: dict[str, tuple[int, int]] = {
    "numerical_audit": (0xB6000, 0xB6010),
    "resource_smoke": (0xB6100, 0xB6108),
    "training": (0xB6200, 0xB62FA),
    "validation": (0xB6400, 0xB6464),
    "prior": (0xB6500, 0xB6514),
    "forward_terminal": (0xB6520, 0xB6534),
    "oracle": (0xB6540, 0xB654A),
}

SEEDS = {
    "training_model": 0xE14A01,
    "records_train": 0xE14B01,
    "records_validation": 0xE14B02,
    "prior_start": 0xE14C01,
    "prior_reverse": 0xE14C02,
    "forward_terminal_forward": 0xE14C03,
    "forward_terminal_reverse": 0xE14C03 ^ 0x51,
    "oracle_forward": 0xE14C04,
    "oracle_reverse": 0xE14C04 ^ 0x51,
    "review_shuffle": 0xE14D01,
    "candidate_audit": 0xE14E01,
    "resource_smoke": 0xE14E02,
}


def _path_range(start: int, stop: int) -> dict[str, Any]:
    values = np.arange(start, stop, dtype="<i8")
    return {
        "start": int(start),
        "stop_exclusive": int(stop),
        "count": int(stop - start),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def candidate_config() -> dict[str, Any]:
    """Return the complete scientific configuration; no science value is a CLI flag."""

    return {
        "schema": VERSION,
        "research_mode": "exploratory",
        "decision": (
            "after fixed numerical and oracle controls, does the frozen global "
            "candidate-K128 learner improve over null for forward-terminal and prior starts"
        ),
        "objective_artifact": "all learned, null, oracle, target and anchor images",
        "claim_scope": (
            "one exploratory model seed and the approximate-candidate K=128 finite chain; "
            "not an exact reverse score or continuum claim"
        ),
        "candidate_target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "candidate_audit_rng": {
            "rng_contract": AUDIT_RNG_CONTRACT,
            "pairing_contract": AUDIT_PAIRING_CONTRACT,
            "initial_prefix_bits": 64,
            "fast_reference_modes": 568,
            "fast_reference_bisection_steps": 56,
            "certified_reference": "sample_alpha1_rb_transition_batch_cuda",
        },
        "core_contract": core.frozen_config(),
        "chain": {
            "outer_steps": 128,
            "phase_count": 7,
            "controller_microsteps": 2,
            "record_outer_steps": [15, 47, 79, 111],
            "anchors": [0, 32, 64, 96, 128],
            "candidate_modes": 128,
            "candidate_modes_semantics": "adaptive minimum",
            "candidate_adaptive_maximum_modes": 1024,
            "candidate_bisection_steps": 56,
            "threads_per_block": 128,
        },
        "data": {
            "arff_sha256": MNIST_ARFF_SHA256,
            "train_slice": [0, 55_000],
            "validation_slice": [55_000, 60_000],
            "terminal_test_content_rows_parsed": 0,
            "whole_file_sha256_read": 1,
            "lambda_mix": 0.35,
            "raster_scale": float(25_471 / 255),
        },
        "model": {
            "parameter_count": 34_974,
            "architecture_fallback": None,
            "forbidden_inputs": sorted(core.FORBIDDEN_MODEL_INPUT_FIELDS),
        },
        "records": {
            "train_paths": 250,
            "train_paths_per_class": 25,
            "validation_paths": 100,
            "validation_paths_per_class": 10,
            "records_per_path": 4,
            "target_semantics": CANDIDATE_TARGET_SEMANTICS,
            "target_dtype": "float32",
            "transition_dtype": "float64",
        },
        "training": {
            "updates": 750,
            "batch_size": 64,
            "learning_rate": 2e-4,
            "ema_decay": 0.999,
            "gradient_norm_cap": 1.0,
            "validation_interval": 250,
            "selection": "earliest finite EMA checkpoint with minimum validation normalized MSE",
            "update_zero_eligible": 0,
        },
        "populations": {
            "prior_paths": 20,
            "forward_terminal_paths": 20,
            "oracle_paths": 10,
            "maximum_cohort_paths": 8,
            "automatic_full_scale_launches": 0,
        },
        "fixed_positions": {
            "audit_train": AUDIT_TRAIN_POSITIONS.tolist(),
            "oracle_validation": ORACLE_VALIDATION_POSITIONS.tolist(),
            "forward_validation": FORWARD_VALIDATION_POSITIONS.tolist(),
        },
        "path_ids": {name: _path_range(*bounds) for name, bounds in PATH_RANGES.items()},
        "seeds": dict(SEEDS),
        "numerical_gate": {
            "lane_count": 512,
            "mnist_lane_count": 448,
            "analytic_lane_count": 64,
            "maximum_later_error": 2e-10,
            "maximum_target_error": 2e-8,
            "maximum_pair_total_error": 2e-12,
            "gate_type": "execution/integrity",
        },
        "oracle_gate": {
            "minimum_improved_paths": 9,
            "aggregate_improvement_required": 1,
            "gate_type": "execution/integrity",
            "rationale": "required only for valid learner attribution",
        },
        "diagnostic": {
            "forward_minimum_l1_wins": 12,
            "forward_minimum_relative_improvement": 0.01,
            "human_minimum_recognizable_learned": 15,
            "human_minimum_requested_learned": 12,
            "evaluator_minimum_learned_accuracy": 0.20,
            "evaluator_minimum_log_probability_wins": 12,
            "gate_type": "diagnostic threshold",
        },
        "resource_defaults": {
            "maximum_active_seconds": 10_800.0,
            "maximum_storage_mib": 2_048.0,
            "maximum_cuda_fraction": 0.75,
            "terminal_reserve_seconds": 900.0,
            "maximum_quantum_seconds": 60.0,
            "projection_multiplier": 1.25,
        },
        "projected_work": {
            "forward_record_transitions": 122_931_200,
            "reverse_transitions": 140_492_800,
            "forward_evaluation_transitions": 10_536_960,
            "base_candidate_transition_work": 273_960_960,
            "full_population_auto_launch": 0,
        },
        "proxy_only_patches_since_last_objective_bearing_experiment": 2,
    }


FROZEN_CONFIG = candidate_config()


class CandidatePilotError(RuntimeError):
    pass


class IntegrityFailure(CandidatePilotError):
    pass


class CandidateHealthFailure(CandidatePilotError):
    pass


class ResourceProjectionFailed(CandidatePilotError):
    pass


class OracleControlFailed(CandidatePilotError):
    pass


class ResourceStop(CandidatePilotError):
    pass


def _require(condition: Any, message: str, error: type[Exception] = IntegrityFailure) -> None:
    if not bool(condition):
        raise error(message)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    _atomic_replace(
        path,
        lambda target: target.write_text(
            json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        ),
    )


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(result, dict), f"expected JSON object: {path}")
    return result


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    def writer(target: Path) -> None:
        with target.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    _atomic_replace(path, writer)


def _write_torch(path: Path, value: Any) -> None:
    _atomic_replace(path, lambda target: torch.save(value, target))


def _write_text(path: Path, value: str) -> None:
    _atomic_replace(path, lambda target: target.write_text(value, encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    def writer(target: Path) -> None:
        with target.open("w", encoding="utf-8", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=list(fieldnames))
            output.writeheader()
            output.writerows([_jsonable(row) for row in rows])

    _atomic_replace(path, writer)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _tree_snapshot(path: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (
            item.relative_to(path).as_posix(),
            int(item.stat().st_size),
            _file_sha256(item),
        )
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


def _git_record(repository_root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {
        "revision": command("rev-parse", "HEAD"),
        "worktree_status": command("status", "--short", "--untracked-files=all"),
    }


def _environment(device: str) -> dict[str, Any]:
    selected = torch.device(device)
    has_device = selected.type == "cuda" and torch.cuda.is_available()
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": str(selected),
        "cuda_available": int(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(selected) if has_device else None,
        "gpu_total_memory_bytes": (
            int(torch.cuda.get_device_properties(selected).total_memory) if has_device else None
        ),
        "gpu_capability": list(torch.cuda.get_device_capability(selected)) if has_device else None,
        "torch_deterministic_algorithms": int(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": int(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": int(torch.backends.cudnn.benchmark),
        "cuda_matmul_tf32": int(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": int(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _set_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _role_ids(role: str) -> np.ndarray:
    return np.arange(*PATH_RANGES[role], dtype=np.int64)


def path_id_audit(repository_root: Path) -> dict[str, Any]:
    """Validate the new role allocation and reject live historical overlap."""

    roles = {name: _path_range(*bounds) for name, bounds in PATH_RANGES.items()}
    intervals = list(PATH_RANGES.items())
    overlaps = [
        [left, right]
        for index, (left, (a0, a1)) in enumerate(intervals)
        for right, (b0, b1) in intervals[index + 1 :]
        if max(a0, b0) < min(a1, b1)
    ]
    excluded = {
        "eulerian_jacobi_ddpm_candidate.py",
        "diag_eulerian_jacobi_ddpm_candidate_pilot.py",
    }
    collisions: list[dict[str, Any]] = []
    lower = min(start for start, _ in PATH_RANGES.values())
    upper = max(stop for _, stop in PATH_RANGES.values())
    for path in sorted((Path(repository_root) / "mnist").glob("*.py")):
        if path.name in excluded:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        range_constants: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "range"
                and len(node.args) in {1, 2}
                and all(isinstance(argument, ast.Constant) and isinstance(argument.value, int) for argument in node.args)
            ):
                if len(node.args) == 1:
                    start, stop = 0, int(node.args[0].value)
                else:
                    start, stop = (int(argument.value) for argument in node.args)
                range_constants.update(id(argument) for argument in node.args)
                if max(start, lower) < min(stop, upper):
                    collisions.append(
                        {
                            "path": path.relative_to(repository_root).as_posix(),
                            "range": [start, stop],
                        }
                    )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and id(node) not in range_constants
                and lower <= int(node.value) < upper
            ):
                collisions.append(
                    {
                        "path": path.relative_to(repository_root).as_posix(),
                        "integer": int(node.value),
                    }
                )
    passed = not overlaps and not collisions and all(
        0 <= start < stop <= 1 << 20 for start, stop in PATH_RANGES.values()
    )
    result = {
        "schema": VERSION + "-path-ids",
        "roles": roles,
        "pairwise_overlaps": overlaps,
        "historical_collisions": collisions,
        "twenty_bit": int(all(0 <= a < b <= 1 << 20 for a, b in PATH_RANGES.values())),
        "passed": int(passed),
    }
    _require(passed, f"candidate path-ID allocation collides: {overlaps or collisions}")
    return result


@dataclass(frozen=True)
class ResourceBudget:
    max_active_seconds: float
    max_storage_bytes: int
    max_cuda_fraction: float
    reserve_seconds: float = 900.0
    maximum_quantum_seconds: float = 60.0
    projection_multiplier: float = 1.25


RESOURCE_STAGE_REMAINING = {
    "oracle_stage": (273_960_960, 750, 100, 120.0),
    "forward_record_caches_stage": (242_350_080, 750, 90, 120.0),
    "training_stage": (119_418_880, 750, 40, 120.0),
    "objective_sampling_stage": (119_418_880, 0, 30, 120.0),
    "population_seal_stage": (0, 0, 10, 120.0),
    "sealed_evaluation_stage": (0, 0, 5, 120.0),
    "post_evaluator_finalization": (0, 0, 2, 5.0),
    "record_review_terminalization": (0, 0, 2, 5.0),
}

# Before the measured smoke exists, every nontrivial CUDA/reference quantum gets
# a conservative, strictly sub-cap reservation.  The 512-lane audit is one eighth
# of the previously measured 4,096-lane audit, so 55 seconds leaves useful margin
# while preserving the fixed 60-second automatic-stop boundary.
PRE_SMOKE_QUANTUM_SECONDS = 55.0
FAILURE_TERMINALIZATION_SECONDS = 5.0
FAILURE_TERMINALIZATION_BYTES = 1024 * 1024


def _sampling_cohort_predicted_bytes(rows: int, *, include_images: bool) -> int:
    array_bytes = int(rows) * 784 * np.dtype(np.float64).itemsize * 7
    identity_bytes = int(rows) * (2 * np.dtype(np.int64).itemsize + 96)
    image_bytes = int(rows) * 4_096 if include_images else 0
    return array_bytes + identity_bytes + image_bytes + 65_536


def _record_cohort_predicted_bytes(paths: int) -> int:
    records = int(paths) * 4
    per_record = (
        784 * np.dtype(np.float32).itemsize
        + EDGES_PER_PHASE * np.dtype(np.float32).itemsize
        + len(core.ForwardRecordDataset.__dataclass_fields__) * 8
    )
    return records * per_record + 65_536


def _forward_terminal_cohort_predicted_bytes(rows: int) -> int:
    return int(rows) * 784 * np.dtype(np.float64).itemsize * 2 + int(rows) * 128 + 65_536


def _stage_resource_projection(run_dir: Path, kind: str) -> dict[str, Any]:
    _require(kind in RESOURCE_STAGE_REMAINING, f"unknown resource stage: {kind}")
    timings = _read_json(Path(run_dir) / "resource_smoke/timings.json")
    transitions, updates, persistence_units, fixed_seconds = RESOURCE_STAGE_REMAINING[kind]
    transition_rate = float(timings["candidate_transition_seconds"]) / int(
        timings["candidate_transition_count"]
    )
    update_rate = float(timings["training_seconds"]) / int(timings["training_updates"])
    persistence_seconds = float(timings["persistence_seconds"])
    seconds = math.fsum(
        (
            transition_rate * int(transitions),
            update_rate * int(updates),
            persistence_seconds * int(persistence_units),
            float(fixed_seconds),
        )
    )
    bytes_per_record = float(timings["storage_bytes_per_record"])
    byte_units = {
        "oracle_stage": 2_000,
        "forward_record_caches_stage": 1_400,
        "training_stage": 4_000,
        "objective_sampling_stage": 8_000,
        "population_seal_stage": 1_000,
        "sealed_evaluation_stage": 2_000,
        "post_evaluator_finalization": 512,
        "record_review_terminalization": 512,
    }[kind]
    return {
        "kind": kind,
        "remaining_transitions": int(transitions),
        "remaining_training_updates": int(updates),
        "remaining_persistence_units": int(persistence_units),
        "fixed_seconds": float(fixed_seconds),
        "projected_remaining_seconds": float(seconds),
        "predicted_next_bytes": max(1, int(math.ceil(bytes_per_record * byte_units))),
    }


def _admit_major_stage(
    run_dir: Path, governor: "ResourceGovernor", kind: str
) -> dict[str, Any]:
    projection = _stage_resource_projection(run_dir, kind)
    return governor.admit(
        kind,
        predicted_seconds=min(
            governor.budget.maximum_quantum_seconds - 1e-9,
            max(1e-9, float(projection["fixed_seconds"])),
        ),
        predicted_bytes=int(projection["predicted_next_bytes"]),
        projected_remaining_seconds=float(projection["projected_remaining_seconds"]),
        major_stage=True,
    )


def resource_admission(
    budget: ResourceBudget,
    *,
    active_seconds: float,
    projected_remaining_seconds: float = 0.0,
    predicted_next_quantum_seconds: float = 0.0,
    storage_bytes: int = 0,
    predicted_next_bytes: int = 0,
    cuda_fraction: float = 0.0,
    major_stage: bool = False,
    populations_sealed: bool = False,
    terminalization: bool = False,
) -> dict[str, Any]:
    reserve = (
        0.0
        if populations_sealed or terminalization
        else float(budget.reserve_seconds)
    )
    projected_total = (
        float(active_seconds)
        + float(budget.projection_multiplier) * float(projected_remaining_seconds)
        + reserve
    )
    quantum_total = (
        float(active_seconds)
        + float(budget.projection_multiplier) * float(predicted_next_quantum_seconds)
        + reserve
    )
    reasons: list[str] = []
    if major_stage and projected_total >= float(budget.max_active_seconds):
        reasons.append("major_stage_active_projection")
    if not major_stage and float(predicted_next_quantum_seconds) >= float(budget.maximum_quantum_seconds):
        reasons.append("next_quantum_duration")
    if not major_stage and quantum_total >= float(budget.max_active_seconds):
        reasons.append("next_quantum_active_projection")
    if int(storage_bytes) + int(predicted_next_bytes) >= int(budget.max_storage_bytes):
        reasons.append("storage_projection")
    if float(cuda_fraction) > float(budget.max_cuda_fraction):
        reasons.append("cuda_fraction")
    return {
        "passed": int(not reasons),
        "reason": reasons[0] if reasons else None,
        "reasons": reasons,
        "active_seconds": float(active_seconds),
        "projected_remaining_seconds": float(projected_remaining_seconds),
        "predicted_next_quantum_seconds": float(predicted_next_quantum_seconds),
        "reserve_remaining_seconds": reserve,
        "major_stage_inequality_lhs": projected_total,
        "quantum_inequality_lhs": quantum_total,
        "maximum_active_seconds": float(budget.max_active_seconds),
        "storage_bytes": int(storage_bytes),
        "predicted_next_bytes": int(predicted_next_bytes),
        "storage_after_next_bytes": int(storage_bytes) + int(predicted_next_bytes),
        "maximum_storage_bytes": int(budget.max_storage_bytes),
        "cuda_fraction": float(cuda_fraction),
        "maximum_cuda_fraction": float(budget.max_cuda_fraction),
        "major_stage": int(bool(major_stage)),
        "populations_sealed": int(bool(populations_sealed)),
        "terminalization": int(bool(terminalization)),
    }


class EvaluatorFirewall:
    BOUND_ONLY = "BOUND_ONLY"
    POPULATIONS_SEALED = "POPULATIONS_SEALED"
    EVALUATOR_OPENED = "EVALUATOR_OPENED"

    def __init__(self) -> None:
        self.state = self.BOUND_ONLY
        self.seal_sha256: str | None = None

    def bind(self) -> str:
        _require(self.state == self.BOUND_ONLY, "evaluator firewall was already advanced")
        return self.state

    def mark_populations_sealed(self, seal_sha256: str) -> None:
        _require(self.state == self.BOUND_ONLY, "population seal firewall order changed")
        _require(bool(re.fullmatch(r"[0-9a-f]{64}", str(seal_sha256))), "invalid seal hash")
        self.state = self.POPULATIONS_SEALED
        self.seal_sha256 = str(seal_sha256)

    def open(self, seal_sha256: str) -> None:
        _require(
            self.state == self.POPULATIONS_SEALED and self.seal_sha256 == str(seal_sha256),
            "evaluator may open only against the current population seal",
        )
        self.state = self.EVALUATOR_OPENED


class ResourceGovernor:
    """Small run-local ledger; it prices one declared quantum at a time."""

    def __init__(
        self,
        run_dir: Path,
        device: torch.device,
        budget: ResourceBudget,
        *,
        started_at: float | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.device = device
        self.budget = budget
        self.last_boundary = time.monotonic() if started_at is None else float(started_at)

    def _ledger(self) -> dict[str, Any]:
        return _read_json(self.run_dir / "resource_ledger.json")

    def _cuda_fraction(self) -> tuple[int, int, float]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0, 0, 0.0
        allocated = int(torch.cuda.max_memory_allocated(self.device))
        reserved = int(torch.cuda.max_memory_reserved(self.device))
        total = int(torch.cuda.get_device_properties(self.device).total_memory)
        return allocated, reserved, allocated / total if total else 0.0

    def admit(
        self,
        kind: str,
        *,
        predicted_seconds: float,
        predicted_bytes: int = 0,
        projected_remaining_seconds: float = 0.0,
        major_stage: bool = False,
        terminalization: bool = False,
    ) -> dict[str, Any]:
        ledger = self._ledger()
        now = time.monotonic()
        current_elapsed = max(0.0, now - self.last_boundary)
        if current_elapsed:
            overhead_tree_bytes = _directory_bytes(self.run_dir)
            ledger["active_seconds"] = math.fsum(
                (float(ledger["active_seconds"]), current_elapsed)
            )
            ledger["events"].append(
                {
                    "kind": "admission_overhead",
                    "seconds": current_elapsed,
                    "candidate_transitions": 0,
                    "model_evaluations": 0,
                    "tree_bytes": overhead_tree_bytes,
                    "completed": 1,
                    "at": _utc_now(),
                }
            )
            ledger["peak_storage_bytes"] = max(
                int(ledger["peak_storage_bytes"]), overhead_tree_bytes
            )
        self.last_boundary = now
        allocated, reserved, fraction = self._cuda_fraction()
        recent = [
            float(row["seconds"])
            for row in ledger["events"]
            if row.get("kind") == kind and row.get("completed") == 1
        ][-3:]
        declared_seconds = float(predicted_seconds)
        next_seconds = max([declared_seconds, *recent])
        decision = resource_admission(
            self.budget,
            active_seconds=float(ledger["active_seconds"]),
            projected_remaining_seconds=float(projected_remaining_seconds),
            predicted_next_quantum_seconds=next_seconds,
            storage_bytes=_directory_bytes(self.run_dir),
            predicted_next_bytes=int(predicted_bytes),
            cuda_fraction=fraction,
            major_stage=major_stage,
            populations_sealed=(self.run_dir / "POPULATIONS_SEALED.json").is_file(),
            terminalization=bool(terminalization),
        )
        prior_same_kind = [
            row for row in ledger.get("admissions", []) if row.get("kind") == kind
        ]
        ledger["last_admission"] = {
            "kind": kind,
            **decision,
            "declared_predicted_next_quantum_seconds": declared_seconds,
            "recent_completed_same_kind_seconds": recent,
            "event_count_before_admission": len(ledger["events"]),
            "kind_admission_ordinal": len(prior_same_kind) + 1,
            "post_completion_check": 0,
            "at": _utc_now(),
        }
        ledger.setdefault("admissions", []).append(ledger["last_admission"])
        ledger["peak_cuda_allocated_bytes"] = max(
            int(ledger["peak_cuda_allocated_bytes"]), allocated
        )
        ledger["peak_cuda_reserved_bytes"] = max(
            int(ledger["peak_cuda_reserved_bytes"]), reserved
        )
        ledger["peak_cuda_fraction"] = max(
            float(ledger["peak_cuda_fraction"]), fraction
        )
        ledger["peak_storage_bytes"] = max(
            int(ledger["peak_storage_bytes"]), int(decision["storage_bytes"])
        )
        _write_json(self.run_dir / "resource_ledger.json", ledger)
        if not decision["passed"] and not terminalization:
            error = ResourceProjectionFailed if major_stage else ResourceStop
            raise error(f"resource admission failed: {decision['reason']}")
        return decision

    def complete(
        self,
        kind: str,
        *,
        transitions: int = 0,
        model_evaluations: int = 0,
        synchronize: bool = True,
        terminalization: bool = False,
    ) -> dict[str, Any]:
        if synchronize and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        boundary = time.monotonic()
        seconds = max(0.0, boundary - self.last_boundary)
        ledger = self._ledger()
        allocated, reserved, fraction = self._cuda_fraction()
        event = {
            "kind": str(kind),
            "seconds": seconds,
            "candidate_transitions": int(transitions),
            "model_evaluations": int(model_evaluations),
            "tree_bytes": _directory_bytes(self.run_dir),
            "completed": 1,
            "at": _utc_now(),
        }
        ledger["events"].append(event)
        ledger["active_seconds"] = math.fsum((float(ledger["active_seconds"]), seconds))
        ledger["peak_storage_bytes"] = max(int(ledger["peak_storage_bytes"]), event["tree_bytes"])
        ledger["peak_cuda_allocated_bytes"] = max(
            int(ledger["peak_cuda_allocated_bytes"]), allocated
        )
        ledger["peak_cuda_reserved_bytes"] = max(
            int(ledger["peak_cuda_reserved_bytes"]), reserved
        )
        ledger["peak_cuda_fraction"] = max(float(ledger["peak_cuda_fraction"]), fraction)
        actual_cap_breach = bool(
            seconds >= float(self.budget.maximum_quantum_seconds)
            or float(ledger["active_seconds"]) >= float(self.budget.max_active_seconds)
            or int(event["tree_bytes"]) >= int(self.budget.max_storage_bytes)
            or fraction > float(self.budget.max_cuda_fraction)
        )
        if actual_cap_breach:
            post_seconds = (
                seconds
                if seconds >= float(self.budget.maximum_quantum_seconds)
                else 0.0
            )
            post_kind = f"post_complete:{kind}"
            decision = resource_admission(
                self.budget,
                active_seconds=float(ledger["active_seconds"]),
                predicted_next_quantum_seconds=post_seconds,
                storage_bytes=int(event["tree_bytes"]),
                cuda_fraction=fraction,
                populations_sealed=(
                    self.run_dir / "POPULATIONS_SEALED.json"
                ).is_file(),
                terminalization=bool(terminalization),
            )
            _require(
                int(decision["passed"]) == 0,
                "post-completion cap breach was not represented as a failed admission",
            )
            ledger["last_admission"] = {
                "kind": post_kind,
                **decision,
                "declared_predicted_next_quantum_seconds": post_seconds,
                "recent_completed_same_kind_seconds": [],
                "event_count_before_admission": len(ledger["events"]),
                "kind_admission_ordinal": 1,
                "post_completion_check": 1,
                "completed_kind": str(kind),
                "at": _utc_now(),
            }
            ledger.setdefault("admissions", []).append(ledger["last_admission"])
        self.last_boundary = boundary
        _write_json(self.run_dir / "resource_ledger.json", ledger)
        if actual_cap_breach and not terminalization:
            raise ResourceStop(
                f"completed resource quantum exceeded its cap: {ledger['last_admission']['reason']}"
            )
        return event

    def outer_step_callback(self, kind: str, *, paths: int) -> Callable[[Mapping[str, Any]], None]:
        def callback(_: Mapping[str, Any]) -> None:
            self.complete(
                kind,
                transitions=int(paths) * 7 * EDGES_PER_PHASE,
                synchronize=False,
            )
            self.admit(kind, predicted_seconds=0.0)

        return callback


def _read_mnist_arff_prefix(
    lines: Iterable[str], *, stop: int = 60_000
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Parse exactly the authorized ARFF prefix without fetching the next row."""

    _require(int(stop) == 60_000, "candidate pilot ARFF prefix changed")
    images: list[np.ndarray] = []
    labels: list[int] = []
    in_data = False
    row = 0
    last_line_number = 0
    for line_number, line in enumerate(lines, 1):
        last_line_number = int(line_number)
        text = line.strip()
        if not in_data:
            in_data = text.upper() == "@DATA"
            continue
        if not text or text.startswith("%"):
            continue
        fields = text.split(",")
        _require(
            len(fields) == 785,
            f"ARFF row {row} (line {line_number}) must have 785 fields",
        )
        try:
            values = np.asarray(fields, dtype=np.float64)
        except ValueError as error:
            raise IntegrityFailure(f"ARFF row {row} has a nonnumeric field") from error
        _require(
            bool(np.isfinite(values).all())
            and bool(np.all(values == np.rint(values))),
            f"ARFF row {row} must contain finite integers",
        )
        pixels, label = values[:784], int(values[-1])
        _require(
            not bool(np.any((pixels < 0) | (pixels > 255))) and 0 <= label <= 9,
            f"ARFF row {row} has an out-of-range pixel or label",
        )
        images.append(pixels.astype(np.uint8).reshape(28, 28))
        labels.append(label)
        row += 1
        if row == int(stop):
            break
    _require(in_data, "ARFF file has no @DATA marker")
    _require(row == int(stop), f"ARFF has only {row} rows; requested stop={stop}")
    return (
        np.stack(images),
        np.asarray(labels, dtype=np.int64),
        {
            "content_rows_read": int(row),
            "last_content_row_index": int(row - 1),
            "terminal_content_rows_read": 0,
            "last_text_line_number_read": int(last_line_number),
            "full_file_read_purpose": "sha256-only",
        },
    )


def _load_train_validation_mnist_strict(
    arff: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Authenticate all bytes, then parse only development/validation content."""

    path = Path(arff).resolve()
    _require(
        path.is_file() and _file_sha256(path) == MNIST_ARFF_SHA256,
        "MNIST ARFF authority changed",
    )
    with path.open("r", encoding="utf-8") as handle:
        images, labels, access = _read_mnist_arff_prefix(handle)
    access["full_file_sha256"] = MNIST_ARFF_SHA256
    return (
        images[:55_000],
        labels[:55_000],
        images[55_000:],
        labels[55_000:],
        access,
    )


def _mnist_unit_masses(images: np.ndarray) -> np.ndarray:
    pixels = np.asarray(images)
    _require(
        pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[1:] == (28, 28),
        "MNIST images must be uint8 [N,28,28]",
    )
    values = pixels.reshape(len(pixels), 784).astype(np.float64)
    totals = values.sum(axis=1, dtype=np.float64)
    _require(bool(np.all(totals > 0.0)), "selected MNIST image has zero mass")
    return np.ascontiguousarray(values / totals[:, None])


def rasterize_population(masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the sole fixed global demix/raster rule."""

    rows = np.asarray(masses, dtype=np.float64)
    _require(rows.ndim == 2 and rows.shape[1] == 784, "population must be [N,784]")
    _require(
        bool(np.isfinite(rows).all())
        and bool(np.all(rows >= 0.0))
        and float(np.max(np.abs(rows.sum(axis=1) - 1.0))) <= 2e-12,
        "population rows are not finite unit masses",
    )
    demixed = np.maximum((rows - 0.35 / 784.0) / (1.0 - 0.35), 0.0)
    totals = demixed.sum(axis=1, dtype=np.float64)
    _require(bool(np.all(totals > 0.0)), "demixed population has a zero row")
    demixed = np.ascontiguousarray(demixed / totals[:, None], dtype=np.float64)
    rendered = np.rint(
        255.0 * np.clip(demixed * (25_471 / 255), 0.0, 1.0)
    ).astype(np.uint8)
    return demixed, rendered.reshape(-1, 28, 28)


def _save_individual_pngs(directory: Path, images: np.ndarray, sample_ids: Sequence[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _require(len(images) == len(sample_ids), "image and sample-ID counts differ")
    for image, sample_id in zip(images, sample_ids, strict=True):
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
            directory / f"{sample_id}.png", format="PNG"
        )


def prepare_data_roles(
    run_dir: Path,
    train_u8: np.ndarray,
    train_labels_all: np.ndarray,
    validation_u8: np.ndarray,
    validation_labels_all: np.ndarray,
    *,
    arff_access: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    train_positions = core.balanced_class_indices(
        train_labels_all, per_class=25, start=0, stop=len(train_labels_all)
    )
    validation_positions = core.balanced_class_indices(
        validation_labels_all,
        per_class=10,
        start=0,
        stop=len(validation_labels_all),
    )
    train_labels = np.asarray(train_labels_all[train_positions], dtype=np.int64)
    validation_labels = np.asarray(
        validation_labels_all[validation_positions], dtype=np.int64
    )
    train_states = core.mix_unit_masses(_mnist_unit_masses(train_u8[train_positions]))
    validation_states = core.mix_unit_masses(
        _mnist_unit_masses(validation_u8[validation_positions])
    )
    arrays = {
        "train_arff_indices": np.asarray(train_positions, dtype=np.int64),
        "train_labels": train_labels,
        "train_mixed_masses": train_states,
        "train_path_ids": _role_ids("training"),
        "validation_arff_indices": np.asarray(validation_positions + 55_000, dtype=np.int64),
        "validation_labels": validation_labels,
        "validation_mixed_masses": validation_states,
        "validation_path_ids": _role_ids("validation"),
        "audit_train_positions": AUDIT_TRAIN_POSITIONS.copy(),
        "oracle_validation_positions": ORACLE_VALIDATION_POSITIONS.copy(),
        "forward_validation_positions": FORWARD_VALIDATION_POSITIONS.copy(),
    }
    _require(
        np.array_equal(np.bincount(train_labels, minlength=10), np.full(10, 25))
        and np.array_equal(np.bincount(validation_labels, minlength=10), np.full(10, 10)),
        "data roles are not exactly class balanced",
    )
    _require(
        int(np.max(arrays["validation_arff_indices"])) < 60_000,
        "terminal MNIST slice entered data roles",
    )
    _write_npz(run_dir / "data_roles.npz", **arrays)
    _write_json(
        run_dir / "data_roles.json",
        {
            "schema": VERSION + "-data-roles",
            "terminal_test_rows_used": 0,
            "content_rows_read": int(arff_access["content_rows_read"]),
            "last_content_row_index": int(arff_access["last_content_row_index"]),
            "terminal_content_rows_read": int(
                arff_access["terminal_content_rows_read"]
            ),
            "last_text_line_number_read": int(
                arff_access["last_text_line_number_read"]
            ),
            "full_file_read_purpose": str(arff_access["full_file_read_purpose"]),
            "full_file_sha256": str(arff_access["full_file_sha256"]),
            "train_class_counts": np.bincount(train_labels, minlength=10),
            "validation_class_counts": np.bincount(validation_labels, minlength=10),
            "path_roles": {
                "training": _path_range(*PATH_RANGES["training"]),
                "validation": _path_range(*PATH_RANGES["validation"]),
            },
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "sha256": _array_sha256(value),
                }
                for name, value in arrays.items()
            },
        },
    )
    return arrays


def build_candidate_audit_bank(
    train_states: np.ndarray, *, device: str | torch.device = "cpu"
) -> dict[str, np.ndarray]:
    """Construct the frozen 448-MNIST plus 64-analytic transition bank."""

    rows = np.asarray(train_states, dtype=np.float64)
    _require(rows.shape == (250, 784), "audit requires the fixed 250 training states")
    selected = torch.as_tensor(
        rows[AUDIT_TRAIN_POSITIONS], dtype=torch.float64, device=torch.device(device)
    )
    tails_all, heads_all = matching_indices(device=selected.device)
    path_ids = _role_ids("numerical_audit")
    slots = np.asarray([0, 97, 196, 391], dtype=np.int64)
    parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "head_fraction",
            "pair_total",
            "exposure",
            "transition_ids",
            "section",
            "phase",
            "path_id",
            "edge_slot",
            "analytic_head_index",
            "analytic_total_index",
            "analytic_duration_index",
        )
    }
    for phase in range(7):
        color = int(PHASE_MATCHINGS[phase])
        tails = tails_all[color]
        heads = heads_all[color]
        pair = selected[:, tails] + selected[:, heads]
        fraction = selected[:, heads] / pair
        exposure = refinement_phase_exposure(
            pair,
            sample_steps=128,
            duration_fraction=float(PHASE_DURATIONS[phase]),
        )
        ids = torch.cat(
            [
                canonical_refinement_transition_ids(
                    path_ids[start : start + 8],
                    sample_steps=128,
                    outer_step=63,
                    phase=phase,
                    device=selected.device,
                ).reshape(-1, EDGES_PER_PHASE)
                for start in range(0, 16, 8)
            ],
            dim=0,
        )
        parts["head_fraction"].append(fraction[:, slots].detach().cpu().numpy().reshape(-1))
        parts["pair_total"].append(pair[:, slots].detach().cpu().numpy().reshape(-1))
        parts["exposure"].append(exposure[:, slots].detach().cpu().numpy().reshape(-1))
        parts["transition_ids"].append(ids[:, slots].detach().cpu().numpy().reshape(-1))
        parts["section"].append(np.zeros(64, dtype=np.int8))
        parts["phase"].append(np.full(64, phase, dtype=np.int64))
        parts["path_id"].append(np.repeat(path_ids, 4))
        parts["edge_slot"].append(np.tile(slots, 16))
        parts["analytic_head_index"].append(np.full(64, -1, dtype=np.int64))
        parts["analytic_total_index"].append(np.full(64, -1, dtype=np.int64))
        parts["analytic_duration_index"].append(np.full(64, -1, dtype=np.int64))

    head_values = np.asarray(
        [0.0, 2.0**-20, 1e-3, 0.1, 0.5, 0.9, 1.0 - 1e-3, 1.0],
        dtype=np.float64,
    )
    pair_values = np.asarray(
        [0.0, 2.0**-20, 1e-6, 1e-4, 1e-3, 1e-2, 0.25, 1.0],
        dtype=np.float64,
    )
    duration_indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 3], dtype=np.int64)
    analytic_heads = np.repeat(head_values, 8)
    analytic_pairs = np.tile(pair_values, 8)
    analytic_exposure = np.empty(64, dtype=np.float64)
    for index in range(8):
        mask = np.arange(64) % 8 == index
        analytic_exposure[mask] = refinement_phase_exposure(
            torch.as_tensor(analytic_pairs[mask], dtype=torch.float64),
            sample_steps=128,
            duration_fraction=float(PHASE_DURATIONS[int(duration_indices[index])]),
        ).numpy()
    analytic_ids = torch.cat(
        [
            canonical_refinement_transition_ids(
                path_ids[start : start + 8],
                sample_steps=128,
                outer_step=127,
                phase=6,
                device=torch.device(device),
            ).reshape(-1, EDGES_PER_PHASE)
            for start in range(0, 16, 8)
        ],
        dim=0,
    )[:, slots].detach().cpu().numpy().reshape(-1)
    parts["head_fraction"].append(analytic_heads)
    parts["pair_total"].append(analytic_pairs)
    parts["exposure"].append(analytic_exposure)
    parts["transition_ids"].append(analytic_ids)
    parts["section"].append(np.ones(64, dtype=np.int8))
    parts["phase"].append(np.full(64, 6, dtype=np.int64))
    parts["path_id"].append(np.repeat(path_ids, 4))
    parts["edge_slot"].append(np.tile(slots, 16))
    parts["analytic_head_index"].append(np.repeat(np.arange(8), 8))
    parts["analytic_total_index"].append(np.tile(np.arange(8), 8))
    parts["analytic_duration_index"].append(np.tile(duration_indices, 8))
    bank = {
        name: np.ascontiguousarray(np.concatenate(values)) for name, values in parts.items()
    }
    bank["active_mask"] = np.asarray(bank["exposure"] > 0.0)
    bank["structural_noop_mask"] = np.asarray(bank["exposure"] == 0.0)
    _require(
        all(len(value) == 512 for value in bank.values())
        and int(np.sum(bank["section"] == 0)) == 448
        and int(np.sum(bank["section"] == 1)) == 64,
        "candidate audit bank size changed",
    )
    _require(np.unique(bank["transition_ids"]).size == 512, "audit transition IDs repeat")
    return bank


_CANDIDATE_ZERO_DIAGNOSTICS = (
    "invalid_input_count",
    "invalid_output_count",
    "nonfinite_count",
    "negative_bracket_width_count",
    "bracket_order_invalid_count",
    "correction_count",
    "clipping_count",
    "floor_count",
    "limiter_count",
    "projection_count",
    "renormalization_count",
    "resource_cap_count",
    "invalid_density_count",
)

_CANDIDATE_REQUIRED_DIAGNOSTICS = (
    "candidate_modes",
    "candidate_minimum_modes",
    "candidate_maximum_adaptive_modes",
    "candidate_adaptive_above_minimum_count",
    "candidate_bisection_steps",
    "candidate_binary_sha256",
    "candidate_batch_type",
    "authorizer_calls",
    "fallback_calls",
    "audit_rng_contract",
    "audit_pairing_contract",
    "audit_canonical_seed",
    "audit_reference_certified_calls",
    "audit_reference_authorizer_calls",
    "audit_reference_fallback_calls",
    "audit_reference_fallback_lane_count",
    "audit_reference_runtime_contract_pass",
    "sample_count",
    "candidate_kernel_launch_count",
    "maximum_candidate_bracket_width",
    "active_count",
    "structural_noop_count",
    "approximation_count",
    *_CANDIDATE_ZERO_DIAGNOSTICS,
)

_SAMPLING_ZERO_COUNTERS = (
    "candidate_invalid_input_count",
    "candidate_invalid_output_count",
    "candidate_nonfinite_count",
    "candidate_negative_bracket_width_count",
    "candidate_bracket_order_invalid_count",
    "candidate_correction_count",
    "candidate_clipping_count",
    "candidate_floor_count",
    "candidate_limiter_count",
    "candidate_projection_count",
    "candidate_renormalization_count",
    "candidate_invalid_lane_count",
    "candidate_approximation_mismatch_count",
    "state_nonfinite_count",
    "state_negative_count",
    "controller_nonfinite_count",
)

_SAMPLING_REQUIRED_TELEMETRY = {
    "controller",
    "controller_rms",
    "finite",
    "nonnegative",
    "candidate_modes",
    "candidate_bisection_steps",
    "backend",
    "candidate_target_semantics",
    "candidate_binary_sha256",
    "candidate_maximum_bracket_width",
    "maximum_mass_error",
    "maximum_pair_total_error",
    "maximum_absolute_q",
    "exact_facet_count",
    "by_time_quarter",
    "outer_step_seconds",
    *_SAMPLING_ZERO_COUNTERS,
}


def _v2_audit_randomness(
    rng_key: Any,
    transition_ids: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, Any]:
    """Rebuild the candidate kernel's first v2 Philox prefix exactly."""

    ids = np.asarray(transition_ids)
    active = np.asarray(active_mask, dtype=bool)
    _require(
        ids.shape == active.shape and np.issubdtype(ids.dtype, np.integer),
        "candidate audit RNG identities changed",
    )
    ids = np.ascontiguousarray(ids, dtype=np.uint64)
    numerators = np.zeros(ids.shape, dtype=np.uint64)
    bits = np.zeros(ids.shape, dtype=np.int32)
    uniforms = np.zeros(ids.shape, dtype=np.float64)
    canonical_seed = certified_cuda._canonical_seed(rng_key)
    flat_ids = ids.reshape(-1)
    flat_active = active.reshape(-1)
    flat_numerators = numerators.reshape(-1)
    flat_bits = bits.reshape(-1)
    flat_uniforms = uniforms.reshape(-1)
    for index, (transition_id, is_active) in enumerate(
        zip(flat_ids, flat_active, strict=True)
    ):
        if not bool(is_active):
            continue
        word = certified_cuda._philox_u64_from_canonical_seed(
            canonical_seed, int(transition_id), 0
        )
        flat_numerators[index] = np.uint64(word)
        flat_bits[index] = 64
        # This is the exact operation ordering in dyadic_midpoint_candidate.
        flat_uniforms[index] = math.ldexp(float(word >> 32), -32) + math.ldexp(
            float(word & 0xFFFFFFFF) + 0.5, -64
        )
        _require(
            flat_uniforms[index]
            == float(Fraction(2 * int(word) + 1, 1 << 65)),
            "host and candidate CUDA dyadic midpoint rounding differ",
        )
    return {
        "rng_contract": AUDIT_RNG_CONTRACT,
        "pairing_contract": AUDIT_PAIRING_CONTRACT,
        "canonical_seed": int(canonical_seed),
        "initial_prefix_numerators": numerators,
        "initial_prefix_bits": bits,
        "uniform_midpoints": uniforms,
    }


def _candidate_adaptive_mode_counts(
    exposure: np.ndarray, *, minimum_modes: int = 128
) -> np.ndarray:
    """Replay the proposal kernel's exposure-dependent mode floor/ceiling rule."""

    durations = np.asarray(exposure, dtype=np.float64)
    _require(
        bool(np.isfinite(durations).all()) and bool(np.all(durations >= 0.0)),
        "candidate audit exposure changed",
    )
    _require(int(minimum_modes) == 128, "candidate audit mode minimum changed")
    result = np.zeros(durations.shape, dtype=np.int32)
    for index, duration in enumerate(durations.reshape(-1)):
        if float(duration) == 0.0:
            continue
        modes = max(128, min(1024, int(minimum_modes)))
        while modes < 1024:
            decay = math.exp(-float(modes * (modes + 1)) * float(duration))
            ratio = math.exp(-2.0 * float(modes + 1) * float(duration))
            tail = decay / (1.0 - ratio) if ratio < 1.0 else 1.0
            if tail < math.ldexp(1.0, -62):
                break
            modes = min(1024, 2 * modes)
        result.reshape(-1)[index] = modes
    return result


def _fixed_uniform_568_reference(
    head_fraction: torch.Tensor,
    exposure: torch.Tensor,
    uniform_midpoints: torch.Tensor,
    *,
    modes: int = 568,
    bisection_steps: int = 56,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert the 568-mode reference using explicit candidate-v2 uniforms."""

    x, duration, uniform = torch.broadcast_tensors(
        head_fraction, exposure, uniform_midpoints
    )
    _require(
        x.dtype == duration.dtype == uniform.dtype == torch.float64,
        "aligned 568 reference requires float64 tensors",
    )
    _require(
        int(modes) == 568 and int(bisection_steps) == 56,
        "aligned 568 reference profile changed",
    )
    _require(
        bool(torch.isfinite(x).all().item())
        and bool(torch.isfinite(duration).all().item())
        and bool(torch.isfinite(uniform).all().item())
        and bool(((x >= 0.0) & (x <= 1.0)).all().item())
        and bool((duration >= 0.0).all().item()),
        "aligned 568 reference inputs changed",
    )
    active = duration > 0.0
    _require(
        bool(((~active) | ((uniform > 0.0) & (uniform < 1.0))).all().item())
        and bool((uniform[~active] == 0.0).all().item()),
        "aligned 568 reference uniform contract changed",
    )
    lower = torch.zeros_like(x)
    upper = torch.ones_like(x)
    safe_duration = torch.where(active, duration, torch.ones_like(duration))
    for _ in range(int(bisection_steps)):
        midpoint = 0.5 * (lower + upper)
        evaluated = evaluate_alpha1_rb_torch_fixed_modes(
            x, midpoint, safe_duration, modes=int(modes)
        )
        go_left = uniform < evaluated.cdf
        upper = torch.where(active & go_left, midpoint, upper)
        lower = torch.where(active & ~go_left, midpoint, lower)
    later = torch.where(active, 0.5 * (lower + upper), x)
    evaluated = evaluate_alpha1_rb_torch_fixed_modes(
        x, later, safe_duration, modes=int(modes)
    )
    target = torch.where(active, evaluated.denoising_target, torch.zeros_like(x))
    return later, target


def recompute_candidate_audit_metrics(
    bank: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every descriptive discrepancy and the fixed Gate-B decision."""

    x = np.asarray(bank["head_fraction"], dtype=np.float64)
    pair = np.asarray(bank["pair_total"], dtype=np.float64)
    active = np.asarray(bank["active_mask"], dtype=bool)
    noop = np.asarray(bank["structural_noop_mask"], dtype=bool)
    candidate_later = np.asarray(outputs["candidate_later"], dtype=np.float64)
    candidate_target = np.asarray(outputs["candidate_target"], dtype=np.float64)
    candidate_approximation = np.asarray(outputs["candidate_approximation_mask"], dtype=bool)
    candidate_valid = np.asarray(outputs["candidate_valid_mask"], dtype=bool)
    transition_ids = np.asarray(bank["transition_ids"], dtype=np.uint64)
    expected_rng = _v2_audit_randomness(
        (SEEDS["candidate_audit"], "k128-candidate-audit"), transition_ids, active
    )
    rng_numerators = np.asarray(
        outputs["rng_v2_initial_prefix_numerators"], dtype=np.uint64
    )
    rng_bits = np.asarray(outputs["rng_v2_initial_prefix_bits"], dtype=np.int32)
    rng_uniforms = np.asarray(outputs["rng_v2_uniform_midpoints"], dtype=np.float64)
    adaptive_modes = np.asarray(outputs["candidate_adaptive_modes"], dtype=np.int32)
    expected_adaptive_modes = _candidate_adaptive_mode_counts(
        np.asarray(bank["exposure"], dtype=np.float64)
    )
    certified_candidate_later = np.asarray(
        outputs["certified_candidate_later"], dtype=np.float64
    )
    certified_candidate_target = np.asarray(
        outputs["certified_candidate_target"], dtype=np.float64
    )
    certified_mask = np.asarray(outputs["certified_mask"], dtype=bool)
    certified_cuda_mask = np.asarray(outputs["certified_cuda_mask"], dtype=bool)
    certified_fallback_mask = np.asarray(outputs["certified_fallback_mask"], dtype=bool)
    expected_shape = x.shape
    aligned_arrays = (
        rng_numerators,
        rng_bits,
        rng_uniforms,
        adaptive_modes,
        certified_candidate_later,
        certified_candidate_target,
        certified_mask,
        certified_cuda_mask,
        certified_fallback_mask,
    )
    _require(
        all(value.shape == expected_shape for value in aligned_arrays),
        "candidate audit aligned-array shape changed",
    )

    def comparison(prefix: str) -> dict[str, Any]:
        later = np.asarray(outputs[f"{prefix}_later"], dtype=np.float64)
        target = np.asarray(outputs[f"{prefix}_target"], dtype=np.float64)
        later_error = candidate_later - later
        target_error = candidate_target - target
        displacement = later - x
        meaningful = active & (np.abs(target) > 1e-10)
        return {
            "maximum_later_fraction_error": float(np.max(np.abs(later_error))),
            "rms_later_fraction_error": float(np.sqrt(np.mean(np.square(later_error)))),
            "maximum_target_error": float(np.max(np.abs(target_error))),
            "rms_target_error": float(np.sqrt(np.mean(np.square(target_error)))),
            "signed_displacement_contrast_agreement": float(
                np.mean(np.sign(candidate_later[active] - x[active]) == np.sign(displacement[active]))
            ),
            "target_sign_agreement_above_1e_10": (
                float(np.mean(np.sign(candidate_target[meaningful]) == np.sign(target[meaningful])))
                if np.any(meaningful)
                else None
            ),
            "state_displacement_relative_rms": float(
                np.sqrt(np.mean(np.square(later_error)))
                / max(float(np.sqrt(np.mean(np.square(displacement)))), 1e-12)
            ),
            "target_relative_rms": float(
                np.sqrt(np.mean(np.square(target_error)))
                / max(float(np.sqrt(np.mean(np.square(target)))), 1e-12)
            ),
        }

    reconstructed = pair * candidate_later + pair * (1.0 - candidate_later)
    normalized_diagnostics = {name: _jsonable(value) for name, value in diagnostics.items()}
    diagnostics_complete = set(_CANDIDATE_REQUIRED_DIAGNOSTICS).issubset(
        normalized_diagnostics
    )
    zero_diagnostics = diagnostics_complete and all(
        int(normalized_diagnostics[name]) == 0 for name in _CANDIDATE_ZERO_DIAGNOSTICS
    )
    candidate_vs_fast = comparison("fast")
    candidate_vs_certified = comparison("certified")
    rng_arrays_exact = bool(
        np.array_equal(
            rng_numerators, expected_rng["initial_prefix_numerators"]
        )
        and np.array_equal(rng_bits, expected_rng["initial_prefix_bits"])
        and np.array_equal(rng_uniforms, expected_rng["uniform_midpoints"])
    )
    internal_candidate_later_error = float(
        np.max(np.abs(candidate_later - certified_candidate_later))
    )
    internal_candidate_target_error = float(
        np.max(np.abs(candidate_target - certified_candidate_target))
    )
    certified_partition_exact = bool(
        np.array_equal(certified_mask, active)
        and np.array_equal(certified_cuda_mask | certified_fallback_mask, active)
        and not np.any(certified_cuda_mask & certified_fallback_mask)
    )
    safe_exposure = np.where(
        active, np.asarray(bank["exposure"], dtype=np.float64), 1.0
    )
    at_candidate = evaluate_alpha1_rb_torch_fixed_modes(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(candidate_later, dtype=torch.float64),
        torch.as_tensor(safe_exposure, dtype=torch.float64),
        modes=568,
    )
    candidate_cdf = at_candidate.cdf.detach().cpu().numpy()
    candidate_target_568 = at_candidate.denoising_target.detach().cpu().numpy()
    maximum_cdf_residual = float(
        np.max(np.abs(candidate_cdf[active] - rng_uniforms[active]))
    )
    maximum_same_y_target_error = float(
        np.max(np.abs(candidate_target[active] - candidate_target_568[active]))
    )
    structural_ok = bool(
        np.all(candidate_later[noop] == x[noop])
        and np.all(candidate_target[noop] == 0.0)
    )
    passed = bool(
        diagnostics_complete
        and int(normalized_diagnostics["candidate_modes"]) == 128
        and int(normalized_diagnostics["candidate_minimum_modes"]) == 128
        and int(normalized_diagnostics["candidate_maximum_adaptive_modes"])
        == int(np.max(expected_adaptive_modes))
        and int(normalized_diagnostics["candidate_adaptive_above_minimum_count"])
        == int(np.sum(expected_adaptive_modes > 128))
        and int(normalized_diagnostics["candidate_bisection_steps"]) == 56
        and normalized_diagnostics["candidate_batch_type"]
        == "CandidateRBCudaBatch"
        and int(normalized_diagnostics["active_count"]) == int(np.sum(active))
        and int(normalized_diagnostics["structural_noop_count"]) == int(np.sum(noop))
        and int(normalized_diagnostics["approximation_count"]) == int(np.sum(active))
        and int(normalized_diagnostics["sample_count"]) == 512
        and int(normalized_diagnostics["candidate_kernel_launch_count"]) == 1
        and math.isfinite(float(normalized_diagnostics["maximum_candidate_bracket_width"]))
        and float(normalized_diagnostics["maximum_candidate_bracket_width"]) >= 0.0
        and np.array_equal(candidate_approximation, active)
        and bool(np.all(candidate_valid))
        and np.array_equal(adaptive_modes, expected_adaptive_modes)
        and rng_arrays_exact
        and normalized_diagnostics["audit_rng_contract"] == AUDIT_RNG_CONTRACT
        and normalized_diagnostics["audit_pairing_contract"]
        == AUDIT_PAIRING_CONTRACT
        and int(normalized_diagnostics["audit_canonical_seed"])
        == int(expected_rng["canonical_seed"])
        and int(normalized_diagnostics["audit_reference_certified_calls"]) == 1
        and int(normalized_diagnostics["audit_reference_authorizer_calls"]) == 1
        and int(normalized_diagnostics["audit_reference_fallback_calls"])
        == int(bool(np.any(certified_fallback_mask)))
        and int(normalized_diagnostics["audit_reference_fallback_lane_count"])
        == int(np.sum(certified_fallback_mask))
        and int(normalized_diagnostics["audit_reference_runtime_contract_pass"])
        == 1
        and internal_candidate_later_error == 0.0
        and internal_candidate_target_error == 0.0
        and certified_partition_exact
        and zero_diagnostics
        and candidate_vs_fast["maximum_later_fraction_error"] <= 2e-10
        and candidate_vs_certified["maximum_later_fraction_error"] <= 2e-10
        and candidate_vs_fast["maximum_target_error"] <= 2e-8
        and candidate_vs_certified["maximum_target_error"] <= 2e-8
        and float(np.max(np.abs(reconstructed - pair))) <= 2e-12
        and structural_ok
    )
    by_section_phase: list[dict[str, Any]] = []
    for section in (0, 1):
        phases = sorted(set(np.asarray(bank["phase"])[np.asarray(bank["section"]) == section].tolist()))
        for phase in phases:
            mask = (np.asarray(bank["section"]) == section) & (np.asarray(bank["phase"]) == phase)
            by_section_phase.append(
                {
                    "section": "mnist" if section == 0 else "analytic",
                    "phase": int(phase),
                    "lane_count": int(np.sum(mask)),
                    "active_count": int(np.sum(active & mask)),
                    "structural_noop_count": int(np.sum(noop & mask)),
                    "approximation_count": int(np.sum(candidate_approximation & mask)),
                }
            )
    return {
        "schema": VERSION + "-candidate-audit-report",
        "gate_type": "execution/integrity",
        "target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "lane_count": 512,
        "candidate_vs_568": candidate_vs_fast,
        "candidate_vs_certified": candidate_vs_certified,
        "rng_alignment": {
            "rng_contract": AUDIT_RNG_CONTRACT,
            "pairing_contract": AUDIT_PAIRING_CONTRACT,
            "canonical_seed": int(expected_rng["canonical_seed"]),
            "initial_prefix_numerators_sha256": _array_sha256(rng_numerators),
            "initial_prefix_bits_sha256": _array_sha256(rng_bits),
            "uniform_midpoints_sha256": _array_sha256(rng_uniforms),
            "rng_arrays_exact": int(rng_arrays_exact),
            "candidate_internal_later_maximum_error": internal_candidate_later_error,
            "candidate_internal_target_maximum_error": internal_candidate_target_error,
            "certified_partition_exact": int(certified_partition_exact),
            "maximum_568_cdf_residual_at_candidate_later": maximum_cdf_residual,
            "maximum_568_target_error_at_candidate_later": maximum_same_y_target_error,
            "certified_reference_calls": int(
                normalized_diagnostics.get("audit_reference_certified_calls", -1)
            ),
            "authorizer_calls": int(
                normalized_diagnostics.get("audit_reference_authorizer_calls", -1)
            ),
            "arb_fallback_calls": int(
                normalized_diagnostics.get("audit_reference_fallback_calls", -1)
            ),
            "arb_fallback_lane_count": int(
                normalized_diagnostics.get("audit_reference_fallback_lane_count", -1)
            ),
            "runtime_contract_pass": int(
                normalized_diagnostics.get("audit_reference_runtime_contract_pass", -1)
            ),
        },
        "candidate_mode_telemetry": {
            "minimum_modes": 128,
            "maximum_adaptive_modes": int(np.max(adaptive_modes)),
            "adaptive_above_minimum_count": int(np.sum(adaptive_modes > 128)),
            "adaptive_modes_sha256": _array_sha256(adaptive_modes),
        },
        "maximum_reconstructed_pair_total_error": float(np.max(np.abs(reconstructed - pair))),
        "structural_noops_exact": int(structural_ok),
        "diagnostics": normalized_diagnostics,
        "counts_by_section_phase": by_section_phase,
        "passed": int(passed),
    }


def gate_b_passed(report: Mapping[str, Any]) -> bool:
    return bool(int(report.get("passed", 0)) == 1)


def oracle_control_metrics(
    targets: np.ndarray,
    null: np.ndarray,
    oracle: np.ndarray,
    null_telemetry: Mapping[str, Any],
    oracle_telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    target_rows = np.asarray(targets, dtype=np.float64)
    null_rows = np.asarray(null, dtype=np.float64)
    oracle_rows = np.asarray(oracle, dtype=np.float64)
    _require(
        target_rows.shape == null_rows.shape == oracle_rows.shape == (10, 784),
        "oracle control requires ten aligned paths",
    )
    null_l1 = np.sum(np.abs(null_rows - target_rows), axis=1, dtype=np.float64)
    oracle_l1 = np.sum(np.abs(oracle_rows - target_rows), axis=1, dtype=np.float64)

    def healthy(rows: np.ndarray, telemetry: Mapping[str, Any]) -> bool:
        complete = _SAMPLING_REQUIRED_TELEMETRY.issubset(telemetry)
        zero_counts = complete and all(
            int(telemetry[name]) == 0 for name in _SAMPLING_ZERO_COUNTERS
        )
        return bool(
            complete
            and
            np.isfinite(rows).all()
            and np.all(rows >= 0.0)
            and float(np.max(np.abs(rows.sum(axis=1) - 1.0))) <= 2e-12
            and float(telemetry["maximum_mass_error"]) <= 2e-12
            and float(telemetry["maximum_pair_total_error"]) <= 2e-12
            and int(telemetry["finite"]) == 1
            and int(telemetry["nonnegative"]) == 1
            and int(telemetry["candidate_modes"]) == 128
            and int(telemetry["candidate_bisection_steps"]) == 56
            and telemetry["backend"] == candidate.CANDIDATE_BACKEND_NAME
            and telemetry["candidate_target_semantics"] == CANDIDATE_TARGET_SEMANTICS
            and zero_counts
        )

    health = healthy(null_rows, null_telemetry) and healthy(oracle_rows, oracle_telemetry)
    wins = int(np.sum(oracle_l1 < null_l1))
    passed = health and wins >= 9 and float(np.sum(oracle_l1)) < float(np.sum(null_l1))
    return {
        "schema": VERSION + "-oracle-control-metrics",
        "gate_type": "execution/integrity",
        "rationale": "positive control required for learner attribution",
        "path_count": 10,
        "null_final_raw_mass_l1": null_l1,
        "oracle_final_raw_mass_l1": oracle_l1,
        "oracle_improved_path_count": wins,
        "aggregate_null_final_raw_mass_l1": float(np.sum(null_l1)),
        "aggregate_oracle_final_raw_mass_l1": float(np.sum(oracle_l1)),
        "health_passed": int(health),
        "passed": int(passed),
    }


def gate_c_passed(metrics: Mapping[str, Any]) -> bool:
    return bool(int(metrics.get("passed", 0)) == 1)


def select_earliest_checkpoint(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [
        dict(row)
        for row in history
        if int(row.get("update", 0)) in {250, 500, 750}
        and int(row.get("eligible", 0)) == 1
        and math.isfinite(float(row.get("validation_normalized_mse", math.inf)))
    ]
    _require(len(eligible) == 3, "training history lacks the three finite checkpoints")
    selected = min(
        eligible,
        key=lambda row: (float(row["validation_normalized_mse"]), int(row["update"])),
    )
    return {
        "selected_update": int(selected["update"]),
        "selected_validation_normalized_mse": float(selected["validation_normalized_mse"]),
        "selection": FROZEN_CONFIG["training"]["selection"],
        "update_zero_eligible": 0,
    }


def _manifest_rows(run_dir: Path) -> list[dict[str, Any]]:
    ignored = {"artifact_manifest.json", "SHA256SUMS.txt"}
    return [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": _file_sha256(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.relative_to(run_dir).as_posix() not in ignored
    ]


def _tree_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(row["size"])).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _seal_manifest(run_dir: Path) -> dict[str, Any]:
    _require(
        all(not item.is_symlink() for item in run_dir.rglob("*")),
        "linked artifacts are forbidden",
    )
    rows = _manifest_rows(run_dir)
    manifest = {
        "schema": VERSION + "-artifact-manifest",
        "artifact_count": len(rows),
        "artifact_bytes": sum(int(row["size"]) for row in rows),
        "tree_digest": _tree_digest(rows),
        "artifacts": rows,
    }
    _write_json(run_dir / "artifact_manifest.json", manifest)
    sums = [f"{row['sha256']}  {row['path']}" for row in rows]
    sums.append(f"{_file_sha256(run_dir / 'artifact_manifest.json')}  artifact_manifest.json")
    _write_text(run_dir / "SHA256SUMS.txt", "\n".join(sums) + "\n")
    return manifest


def _verify_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "artifact_manifest.json")
    rows = manifest.get("artifacts")
    _require(isinstance(rows, list), "artifact manifest rows are invalid")
    actual = _manifest_rows(run_dir)
    _require([row["path"] for row in rows] == [row["path"] for row in actual], "artifact inventory changed")
    _require(rows == actual, "artifact size or hash changed")
    _require(
        int(manifest.get("artifact_count", -1)) == len(rows)
        and int(manifest.get("artifact_bytes", -1)) == sum(int(row["size"]) for row in rows)
        and manifest.get("tree_digest") == _tree_digest(rows),
        "artifact manifest totals changed",
    )
    expected_sums = [f"{row['sha256']}  {row['path']}" for row in rows]
    expected_sums.append(f"{_file_sha256(run_dir / 'artifact_manifest.json')}  artifact_manifest.json")
    _require(
        (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8")
        == "\n".join(expected_sums) + "\n",
        "checksum list changed",
    )
    return manifest


def _status(run_dir: Path, route: str, *, error: str | None = None) -> None:
    _write_json(
        run_dir / "status.json",
        {
            "schema": VERSION + "-status",
            "route": route,
            "error": error,
            "whole_run_restart_required": int(
                route
                in {
                    "integrity_failed",
                    "candidate_health_failed",
                    "resource_projection_failed",
                    "oracle_control_failed",
                    "resource_stopped",
                }
            ),
            "updated_at": _utc_now(),
        },
    )


def _append_stage(run_dir: Path, stage: str, state: str, **details: Any) -> None:
    ledger = _read_json(run_dir / "stage_ledger.json")
    ledger["events"].append(
        {"stage": stage, "state": state, "at": _utc_now(), **_jsonable(details)}
    )
    _write_json(run_dir / "stage_ledger.json", ledger)


def _validate_approval(value: str) -> str:
    approval = str(value).strip()
    _require(bool(approval) and not PLACEHOLDER.search(approval), "fresh real approval ID is required")
    return approval


def _validate_resource_values(seconds: float, storage_mib: float, cuda_fraction: float) -> None:
    defaults = FROZEN_CONFIG["resource_defaults"]
    _require(
        all(math.isfinite(float(value)) for value in (seconds, storage_mib, cuda_fraction))
        and float(seconds) > 0.0
        and float(storage_mib) > 0.0
        and 0.0 < float(cuda_fraction) <= 1.0,
        "resource caps are invalid",
    )
    _require(
        float(seconds) <= float(defaults["maximum_active_seconds"])
        and float(storage_mib) <= float(defaults["maximum_storage_mib"])
        and float(cuda_fraction) <= float(defaults["maximum_cuda_fraction"]),
        "resource caps exceed the frozen approval envelope",
    )


def _canonical_run_argv(
    run_dir: Path,
    arff: Path,
    ddpm_run_dir: Path,
    *,
    device: str,
    approval_id: str,
    maximum_active_seconds: float,
    maximum_storage_mib: float,
    maximum_cuda_fraction: float,
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "mnist.diag_eulerian_jacobi_ddpm_candidate_pilot",
        "run",
        "--run-dir",
        str(Path(run_dir).resolve()),
        "--arff",
        str(Path(arff).resolve()),
        "--ddpm-run-dir",
        str(Path(ddpm_run_dir).resolve()),
        "--device",
        str(device),
        "--approval-id",
        str(approval_id),
        "--max-active-seconds",
        format(float(maximum_active_seconds), ".15g"),
        "--max-storage-mib",
        format(float(maximum_storage_mib), ".15g"),
        "--max-cuda-fraction",
        format(float(maximum_cuda_fraction), ".15g"),
    ]


def _canonical_command_text(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(value) for value in argv]) + "\n"


def _source_bindings(
    repository_root: Path, arff: Path, ddpm_run_dir: Path
) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in PROTECTED_SOURCE_HASHES.items():
        path = repository_root / relative
        _require(path.is_file(), f"protected source is missing: {relative}")
        observed[relative] = _file_sha256(path)
        _require(observed[relative] == expected, f"protected source drifted: {relative}")
    deferred_path = repository_root / "mnist/d0_jacobi_rb_cuda_deferred.py"
    _require(
        deferred_path.is_file()
        and _file_sha256(deferred_path) == PROTECTED_DEFERRED_CUDA_SOURCE_SHA256,
        "protected candidate CUDA source drifted",
    )
    additive: dict[str, str] = {}
    for relative in ADDITIVE_SOURCE_FILES:
        path = repository_root / relative
        _require(path.is_file(), f"additive source is missing: {relative}")
        additive[relative] = _file_sha256(path)

    evaluator_paths = {
        "checkpoint": ddpm_run_dir / "evaluator/selected_checkpoint.pt",
        "selection": ddpm_run_dir / "evaluator/selection.json",
        "metrics": ddpm_run_dir / "evaluation/metrics.json",
        "manifest": ddpm_run_dir / "artifact_manifest.json",
        "status": ddpm_run_dir / "status.json",
    }
    for name, path in evaluator_paths.items():
        _require(path.is_file(), f"accepted evaluator {name} is missing")
    evaluator_hashes = {
        name: _file_sha256(path)
        for name, path in evaluator_paths.items()
        if name != "status"
    }
    _require(
        evaluator_hashes
        == {
            "checkpoint": ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256,
            "selection": ACCEPTED_DDPM_EVALUATOR_SELECTION_SHA256,
            "metrics": ACCEPTED_DDPM_METRICS_SHA256,
            "manifest": ACCEPTED_DDPM_MANIFEST_SHA256,
        }
        and _read_json(evaluator_paths["status"]).get("state") == "complete",
        "DDPM evaluator authority changed",
    )
    return {
        "schema": VERSION + "-source-bindings",
        "repository_root": str(repository_root),
        "historical_source_inventory": dict(PROTECTED_SOURCE_HASHES),
        "observed_protected_source_hashes": observed,
        "protected_candidate_cuda_source": {
            "path": "mnist/d0_jacobi_rb_cuda_deferred.py",
            "sha256": PROTECTED_DEFERRED_CUDA_SOURCE_SHA256,
        },
        "additive_source_hashes": additive,
        "arff": str(arff),
        "arff_sha256": _file_sha256(arff),
        "ddpm_run_dir": str(ddpm_run_dir),
        "evaluator_files": {name: str(path) for name, path in evaluator_paths.items()},
        "evaluator_hashes": evaluator_hashes,
        "evaluator_weights_loaded_at_binding": 0,
        "git": _git_record(repository_root),
    }


def initialize_run(
    repository_root: Path,
    arff: Path,
    ddpm_run_dir: Path,
    run_dir: Path,
    *,
    device: str,
    approval_id: str,
    maximum_active_seconds: float,
    maximum_storage_mib: float,
    maximum_cuda_fraction: float,
) -> tuple[Path, ResourceGovernor, EvaluatorFirewall]:
    initialize_started = time.monotonic()
    repository_root = Path(repository_root).resolve()
    arff = Path(arff).resolve()
    ddpm_run_dir = Path(ddpm_run_dir).resolve()
    run_dir = Path(run_dir).resolve()
    selected = torch.device(device)
    _require(selected.type == "cuda", "production requires CUDA; CPU is only for tests/fakes")
    _require(torch.cuda.is_available(), "CUDA production device is unavailable")
    approval = _validate_approval(approval_id)
    _validate_resource_values(
        maximum_active_seconds, maximum_storage_mib, maximum_cuda_fraction
    )
    _require(repository_root.is_dir(), "repository root is missing")
    _require(arff.is_file() and _file_sha256(arff) == MNIST_ARFF_SHA256, "MNIST ARFF authority changed")
    _require(ddpm_run_dir.is_dir(), "DDPM evaluator run is missing")
    _require(not run_dir.exists(), "run directory must be fresh and absent")
    bindings = _source_bindings(repository_root, arff, ddpm_run_dir)
    path_audit = path_id_audit(repository_root)
    _set_determinism()
    run_dir.mkdir(parents=True)
    for relative in (
        "candidate_audit",
        "resource_smoke",
        "oracle_control/stages/forward_terminal",
        "oracle_control/stages/null",
        "oracle_control/stages/oracle",
        "oracle_control/images",
        "forward_records/train",
        "forward_records/validation",
        "training",
        "populations/stages/null_prior",
        "populations/stages/learned_prior",
        "populations/stages/forward_terminal_starts",
        "populations/stages/null_forward_terminal",
        "populations/stages/learned_forward_terminal",
        "populations/images",
        "populations/contact_sheets",
        "evaluation",
        "review",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    runtime_config = copy.deepcopy(FROZEN_CONFIG)
    canonical_argv = _canonical_run_argv(
        run_dir,
        arff,
        ddpm_run_dir,
        device=str(selected),
        approval_id=approval,
        maximum_active_seconds=maximum_active_seconds,
        maximum_storage_mib=maximum_storage_mib,
        maximum_cuda_fraction=maximum_cuda_fraction,
    )
    command_text = _canonical_command_text(canonical_argv)
    runtime_config["execution_authority"] = {
        "approval_id": approval,
        "device": str(selected),
        "maximum_active_seconds": float(maximum_active_seconds),
        "maximum_storage_mib": float(maximum_storage_mib),
        "maximum_cuda_fraction": float(maximum_cuda_fraction),
        "terminal_reserve_seconds": 900.0,
        "exact_cli_subcommand": "run",
        "canonical_argv": canonical_argv,
        "command_sha256": hashlib.sha256(command_text.encode("utf-8")).hexdigest(),
        "whole_run_restart_only": 1,
        "automatic_full_scale_launches": 0,
    }
    _write_json(run_dir / "config.json", runtime_config)
    _write_text(run_dir / "command.txt", command_text)
    bindings["config_sha256"] = _semantic_sha256(runtime_config)
    _write_json(run_dir / "source_bindings.json", bindings)
    _write_json(run_dir / "environment.json", _environment(str(selected)))
    _write_json(run_dir / "path_id_audit.json", path_audit)
    _write_json(
        run_dir / "stage_ledger.json",
        {"schema": VERSION + "-stage-ledger", "events": []},
    )
    budget = ResourceBudget(
        max_active_seconds=float(maximum_active_seconds),
        max_storage_bytes=int(float(maximum_storage_mib) * 1024 * 1024),
        max_cuda_fraction=float(maximum_cuda_fraction),
    )
    _write_json(
        run_dir / "resource_ledger.json",
        {
            "schema": VERSION + "-resource-ledger",
            "budget": asdict(budget),
            "active_seconds": 0.0,
            "peak_storage_bytes": _directory_bytes(run_dir),
            "peak_cuda_allocated_bytes": 0,
            "peak_cuda_reserved_bytes": 0,
            "peak_cuda_fraction": 0.0,
            "events": [],
            "admissions": [],
            "last_admission": None,
        },
    )
    _status(run_dir, "initialized")
    _append_stage(run_dir, "initialize_and_bind", "complete")
    governor = ResourceGovernor(
        run_dir, selected, budget, started_at=initialize_started
    )
    governor.complete("initialize_and_bind", synchronize=False)
    return run_dir, governor, EvaluatorFirewall()


def candidate_rng_keys() -> tuple[tuple[Any, ...], ...]:
    keys: list[tuple[Any, ...]] = [(SEEDS["candidate_audit"], "k128-candidate-audit")]
    for name in (
        "resource_smoke",
        "records_train",
        "records_validation",
        "forward_terminal_forward",
        "oracle_forward",
    ):
        keys.append((SEEDS[name], "forward"))
    for name in (
        "resource_smoke",
        "prior_reverse",
        "forward_terminal_reverse",
        "oracle_reverse",
    ):
        root = SEEDS[name]
        keys.extend(
            (root, "reverse", micro, side)
            for micro in (0, 1)
            for side in ("pre", "post")
        )
    return tuple(keys)


def prepare_runtime_stage(
    run_dir: Path, device: torch.device, governor: ResourceGovernor
) -> candidate.CandidateRuntime:
    governor.admit(
        "prepare_candidate_backend",
        predicted_seconds=PRE_SMOKE_QUANTUM_SECONDS,
        predicted_bytes=256 * 1024,
    )
    runtime = candidate.prepare_candidate_runtime(device=device, rng_keys=candidate_rng_keys())
    _require(
        isinstance(runtime, candidate.CandidateRuntime),
        "candidate runtime dispatch changed",
    )
    record = {
        "schema": VERSION + "-candidate-backend",
        "backend": candidate.CANDIDATE_BACKEND_NAME,
        "target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "runtime_type": type(runtime).__name__,
        "candidate_modes": int(runtime.profile.candidate_modes),
        "candidate_modes_semantics": "adaptive minimum",
        "candidate_adaptive_maximum_modes": 1024,
        "candidate_bisection_steps": int(runtime.profile.candidate_bisection_steps),
        "threads_per_block": int(runtime.profile.threads_per_block),
        "candidate_binary_sha256": runtime.candidate_binary_sha256,
        "dispatch": [
            "prepare_alpha1_rb_transition_batch_cuda_candidate",
            "prepare_alpha1_rb_transition_cuda_rng_seed",
            "enqueue_alpha1_rb_transition_batch_cuda_candidate",
        ],
        "authorizer_calls": 0,
        "certified_cuda_calls": 0,
        "arb_fallback_calls": 0,
        "scope": "production_candidate_runtime_excludes_audit_references",
        "prepared_rng_keys_sha256": _semantic_sha256(candidate_rng_keys()),
    }
    _write_json(run_dir / "candidate_backend.json", record)
    _append_stage(run_dir, "prepare_candidate_backend", "complete")
    governor.complete("prepare_candidate_backend")
    return runtime


def run_candidate_audit(
    run_dir: Path,
    train_states: np.ndarray,
    runtime: candidate.CandidateRuntime,
    governor: ResourceGovernor,
) -> dict[str, Any]:
    _require(
        isinstance(runtime, candidate.CandidateRuntime),
        "candidate audit runtime dispatch changed",
    )
    governor.admit(
        "candidate_audit",
        predicted_seconds=PRE_SMOKE_QUANTUM_SECONDS,
        predicted_bytes=8 * 1024 * 1024,
    )
    bank = build_candidate_audit_bank(train_states, device="cpu")
    _write_npz(run_dir / "candidate_audit/bank.npz", **bank)
    key = (SEEDS["candidate_audit"], "k128-candidate-audit")
    device = runtime.device
    head = torch.as_tensor(bank["head_fraction"], dtype=torch.float64, device=device)
    exposure = torch.as_tensor(bank["exposure"], dtype=torch.float64, device=device)
    transition_ids = torch.as_tensor(bank["transition_ids"], dtype=torch.uint64, device=device)
    randomness = _v2_audit_randomness(
        key, bank["transition_ids"], bank["active_mask"]
    )
    _require(
        int(runtime.prepared_seeds[key].seed) == int(randomness["canonical_seed"]),
        "prepared candidate audit seed changed",
    )
    batch = enqueue_alpha1_rb_transition_batch_cuda_candidate(
        head,
        exposure,
        rng_key=key,
        transition_ids=transition_ids,
        prepared=runtime.prepared,
        prepared_rng_seed=runtime.prepared_seeds[key],
    )
    _require(isinstance(batch, CandidateRBCudaBatch), "candidate audit dispatch changed")
    torch.cuda.synchronize(device)
    candidate_diagnostics = {
        name: value.detach().cpu().item() for name, value in batch.diagnostics.items()
    }
    candidate_diagnostics.update(
        {
            "candidate_modes": int(runtime.profile.candidate_modes),
            "candidate_bisection_steps": int(runtime.profile.candidate_bisection_steps),
            "candidate_binary_sha256": runtime.candidate_binary_sha256,
            "candidate_batch_type": type(batch).__name__,
            "authorizer_calls": 0,
            "fallback_calls": 0,
        }
    )
    candidate_later = batch.later_head_fraction.detach().cpu().numpy()
    candidate_target = batch.denoising_target.detach().cpu().numpy()
    candidate_lower = batch.candidate_lower.detach().cpu().numpy()
    candidate_upper = batch.candidate_upper.detach().cpu().numpy()
    candidate_valid = batch.valid_mask.detach().cpu().numpy()
    candidate_approximation = batch.approximation_mask.detach().cpu().numpy()
    adaptive_modes = _candidate_adaptive_mode_counts(bank["exposure"])
    uniform_tensor = torch.as_tensor(
        randomness["uniform_midpoints"], dtype=torch.float64, device=device
    )
    fast_later_tensor, fast_target_tensor = _fixed_uniform_568_reference(
        head, exposure, uniform_tensor
    )
    fast_later = fast_later_tensor.detach().cpu().numpy()
    fast_target = fast_target_tensor.detach().cpu().numpy()

    certified = sample_alpha1_rb_transition_batch_cuda(
        head,
        exposure,
        rng_key=key,
        profile=runtime.profile,
        transition_ids=transition_ids,
    )
    _require(
        certified.runtime_report.get("rng_contract") == AUDIT_RNG_CONTRACT,
        "certified audit RNG contract changed",
    )
    certified_diagnostics = {
        name: value.detach().cpu().item()
        for name, value in certified.diagnostics.items()
    }
    certified_candidate_later = (
        certified.candidate_later_head_fraction.detach().cpu().numpy()
    )
    certified_candidate_target = (
        certified.candidate_denoising_target.detach().cpu().numpy()
    )
    _require(
        np.array_equal(candidate_later, certified_candidate_later)
        and np.array_equal(candidate_target, certified_candidate_target),
        "candidate-only and certified-reference candidate draws changed",
    )
    certified_fallback_mask = certified.fallback_mask.detach().cpu().numpy()
    candidate_diagnostics.update(
        {
            "candidate_minimum_modes": 128,
            "candidate_maximum_adaptive_modes": int(np.max(adaptive_modes)),
            "candidate_adaptive_above_minimum_count": int(
                np.sum(adaptive_modes > 128)
            ),
            "audit_rng_contract": AUDIT_RNG_CONTRACT,
            "audit_pairing_contract": AUDIT_PAIRING_CONTRACT,
            "audit_canonical_seed": int(randomness["canonical_seed"]),
            "audit_reference_certified_calls": 1,
            "audit_reference_authorizer_calls": 1,
            "audit_reference_fallback_calls": int(
                bool(np.any(certified_fallback_mask))
            ),
            "audit_reference_fallback_lane_count": int(
                np.sum(certified_fallback_mask)
            ),
            "audit_reference_runtime_contract_pass": int(
                bool(certified.runtime_report.get("runtime_contract_pass"))
            ),
        }
    )
    outputs = {
        "candidate_later": candidate_later,
        "candidate_target": candidate_target,
        "candidate_lower": candidate_lower,
        "candidate_upper": candidate_upper,
        "candidate_valid_mask": candidate_valid,
        "candidate_approximation_mask": candidate_approximation,
        "fast_later": np.asarray(fast_later, dtype=np.float64),
        "fast_target": np.asarray(fast_target, dtype=np.float64),
        "certified_later": certified.later_head_fraction.detach().cpu().numpy(),
        "certified_target": certified.denoising_target.detach().cpu().numpy(),
        "certified_candidate_later": certified_candidate_later,
        "certified_candidate_target": certified_candidate_target,
        "certified_mask": certified.certified_mask.detach().cpu().numpy(),
        "certified_cuda_mask": certified.cuda_certified_mask.detach().cpu().numpy(),
        "certified_fallback_mask": certified_fallback_mask,
        "certified_quantile_lower": certified.quantile_lower.detach().cpu().numpy(),
        "certified_quantile_upper": certified.quantile_upper.detach().cpu().numpy(),
        "certified_target_lower": certified.target_lower.detach().cpu().numpy(),
        "certified_target_upper": certified.target_upper.detach().cpu().numpy(),
        "certified_certificate_codes": certified.certificate_codes.detach().cpu().numpy(),
        "certified_prefix_bits": certified.prefix_bits.detach().cpu().numpy(),
        "candidate_adaptive_modes": adaptive_modes,
        "rng_v2_initial_prefix_numerators": randomness[
            "initial_prefix_numerators"
        ],
        "rng_v2_initial_prefix_bits": randomness["initial_prefix_bits"],
        "rng_v2_uniform_midpoints": randomness["uniform_midpoints"],
        "transition_ids": np.asarray(bank["transition_ids"]),
        "earlier_head_fraction": np.asarray(bank["head_fraction"]),
        "exposure": np.asarray(bank["exposure"]),
    }
    _write_npz(run_dir / "candidate_audit/outputs.npz", **outputs)
    report = recompute_candidate_audit_metrics(bank, outputs, candidate_diagnostics)
    report["certified_diagnostics"] = certified_diagnostics
    report["certified_runtime"] = _jsonable(certified.runtime_report)
    _write_json(run_dir / "candidate_audit/report.json", report)
    governor.complete("candidate_audit", transitions=512 * 3, synchronize=False)
    _append_stage(run_dir, "candidate_audit", "complete", passed=report["passed"])
    if not gate_b_passed(report):
        raise CandidateHealthFailure("fixed 512-lane candidate numerical health gate failed")
    return report


def _synthetic_records(count: int, seed: int) -> core.ForwardRecordDataset:
    rng = np.random.default_rng(seed)
    states = rng.dirichlet(np.ones(784), size=count).astype(np.float32)
    targets = rng.normal(0.0, 0.05, size=(count, EDGES_PER_PHASE)).astype(np.float32)
    return core.ForwardRecordDataset(
        later_states=states,
        reverse_time=np.linspace(0.01, 0.99, count, dtype=np.float32),
        phase=np.arange(count, dtype=np.int64) % 7,
        color=np.asarray([PHASE_MATCHINGS[index % 7] for index in range(count)], dtype=np.int64),
        duration=np.asarray([PHASE_DURATIONS[index % 7] for index in range(count)], dtype=np.float32),
        labels=np.arange(count, dtype=np.int64) % 10,
        targets=targets,
        path_ids=np.resize(_role_ids("resource_smoke"), count).astype(np.int64),
        outer_steps=np.asarray([FROZEN_CONFIG["chain"]["record_outer_steps"][index % 4] for index in range(count)], dtype=np.int64),
    )


def _record_arrays(dataset: core.ForwardRecordDataset) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(dataset, name))
        for name in core.ForwardRecordDataset.__dataclass_fields__
    }


def _merge_records(parts: Sequence[core.ForwardRecordDataset]) -> core.ForwardRecordDataset:
    _require(bool(parts), "no forward-record cohorts were completed")
    return core.ForwardRecordDataset(
        **{
            name: np.concatenate([np.asarray(getattr(part, name)) for part in parts], axis=0)
            for name in core.ForwardRecordDataset.__dataclass_fields__
        }
    )


def _merge_sampling(parts: Sequence[core.SamplingResult]) -> core.SamplingResult:
    _require(bool(parts), "no sampling cohorts were completed")
    _require(
        all(
            _SAMPLING_REQUIRED_TELEMETRY.issubset(part.telemetry)
            and isinstance(part.telemetry.get("by_time_quarter"), list)
            and len(part.telemetry["by_time_quarter"]) == 4
            for part in parts
        ),
        "sampling cohort telemetry inventory changed",
    )
    anchor_keys = sorted(parts[0].anchors)
    _require(
        all(sorted(part.anchors) == anchor_keys for part in parts),
        "sampling cohort anchor sets differ",
    )
    step_seconds = [
        float(value)
        for part in parts
        for value in part.telemetry.get("outer_step_seconds", [])
    ]
    quarter_rows: list[dict[str, Any]] = []
    for quarter in range(4):
        source = [part.telemetry["by_time_quarter"][quarter] for part in parts]

        def rms(sum_name: str, count_name: str, fallback_name: str) -> float:
            if all(sum_name in row and count_name in row for row in source):
                total = math.fsum(float(row[sum_name]) for row in source)
                count = sum(int(row[count_name]) for row in source)
                return math.sqrt(total / count) if count else 0.0
            weights = np.asarray([int(row.get("score_count", 1)) for row in source], dtype=np.float64)
            values = np.asarray([float(row.get(fallback_name, 0.0)) for row in source])
            return math.sqrt(float(np.sum(weights * np.square(values))) / float(np.sum(weights)))

        score_count = sum(int(row.get("score_count", 0)) for row in source)
        quarter_rows.append(
            {
                "quarter": quarter,
                "time_quarter": quarter,
                "score_count": score_count,
                "score_rms": rms("score_square_sum", "score_count", "score_rms"),
                "controller_rms": rms("score_square_sum", "score_count", "controller_rms"),
                "reference_fraction_displacement_rms": rms(
                    "reference_square_sum", "reference_count", "reference_fraction_displacement_rms"
                ),
                "control_fraction_displacement_rms": rms(
                    "control_square_sum", "control_count", "control_fraction_displacement_rms"
                ),
                "maximum_absolute_q": max(float(row.get("maximum_absolute_q", 0.0)) for row in source),
                "maximum_absolute_logit_increment": max(
                    float(row.get("maximum_absolute_logit_increment", 0.0)) for row in source
                ),
                "maximum_mass_error": max(float(row.get("maximum_mass_error", 0.0)) for row in source),
                "maximum_pair_total_error": max(
                    float(row.get("maximum_pair_total_error", 0.0)) for row in source
                ),
            }
        )
    telemetry: dict[str, Any] = {
        "controller": str(parts[0].telemetry["controller"]),
        "controller_rms": math.sqrt(
            math.fsum(
                int(row["score_count"]) * float(row["controller_rms"]) ** 2
                for row in quarter_rows
            )
            / max(1, sum(int(row["score_count"]) for row in quarter_rows))
        ),
        "maximum_absolute_q": max(float(part.telemetry["maximum_absolute_q"]) for part in parts),
        "maximum_mass_error": max(float(part.telemetry["maximum_mass_error"]) for part in parts),
        "maximum_pair_total_error": max(
            float(part.telemetry["maximum_pair_total_error"]) for part in parts
        ),
        "exact_facet_count": sum(int(part.telemetry["exact_facet_count"]) for part in parts),
        "finite": int(all(int(part.telemetry["finite"]) == 1 for part in parts)),
        "nonnegative": int(all(int(part.telemetry["nonnegative"]) == 1 for part in parts)),
        "microsteps": 2,
        "by_time_quarter": quarter_rows,
        "backend": candidate.CANDIDATE_BACKEND_NAME,
        "candidate_target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "candidate_modes": 128,
        "candidate_bisection_steps": 56,
        "candidate_binary_sha256": str(parts[0].telemetry["candidate_binary_sha256"]),
        "outer_step_seconds": step_seconds,
    }
    counter_names = {
        key
        for part in parts
        for key, value in part.telemetry.items()
        if key.endswith("_count")
        and isinstance(value, (int, np.integer))
    }
    _require(
        all(counter_names.issubset(part.telemetry) for part in parts)
        and set(_SAMPLING_ZERO_COUNTERS).issubset(counter_names),
        "sampling cohort counter inventory changed",
    )
    for name in counter_names:
        telemetry[name] = sum(int(part.telemetry[name]) for part in parts)
    telemetry["candidate_maximum_bracket_width"] = max(
        float(part.telemetry["candidate_maximum_bracket_width"]) for part in parts
    )
    return core.SamplingResult(
        starts=np.concatenate([part.starts for part in parts]),
        final_states=np.concatenate([part.final_states for part in parts]),
        anchors={
            anchor: np.concatenate([part.anchors[anchor] for part in parts])
            for anchor in anchor_keys
        },
        telemetry=telemetry,
    )


def _governed_outer_callback(
    governor: ResourceGovernor,
    kind: str,
    *,
    paths: int,
    predicted_seconds: float,
    predicted_bytes: int = 0,
    persistence_seconds: float = 0.0,
    defer_final_completion: bool = False,
) -> Callable[[Mapping[str, Any]], None]:
    def callback(record: Mapping[str, Any]) -> None:
        direction = str(record["direction"])
        transitions = int(paths) * 7 * EDGES_PER_PHASE * (4 if direction == "reverse" else 1)
        step = int(record["outer_step"])
        final_step = (direction.startswith("forward") and step == 127) or (
            direction == "reverse" and step == 0
        )
        if final_step and defer_final_completion:
            return
        governor.complete(kind, transitions=transitions, synchronize=False)
        continues = (direction.startswith("forward") and step < 127) or (
            direction == "reverse" and step > 0
        )
        if continues:
            next_is_final = (
                direction.startswith("forward") and step == 126
            ) or (direction == "reverse" and step == 1)
            governor.admit(
                kind,
                predicted_seconds=(
                    float(predicted_seconds) + float(persistence_seconds)
                    if next_is_final
                    else float(predicted_seconds)
                ),
                predicted_bytes=int(predicted_bytes),
            )

    return callback


def _save_sampling_npz(
    path: Path,
    result: core.SamplingResult,
    labels: np.ndarray,
    path_ids: np.ndarray,
    sample_ids: np.ndarray,
    **extra: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "starts": np.asarray(result.starts, dtype=np.float64),
        "final_states": np.asarray(result.final_states, dtype=np.float64),
        "requested_labels": np.asarray(labels, dtype=np.int64),
        "path_ids": np.asarray(path_ids, dtype=np.int64),
        "sample_ids": np.asarray(sample_ids, dtype=np.str_),
    }
    for anchor in (0, 32, 64, 96, 128):
        arrays[f"anchors_{anchor:03d}"] = np.asarray(result.anchors[anchor], dtype=np.float64)
    arrays.update({name: np.asarray(value) for name, value in extra.items()})
    _write_npz(path, **arrays)
    _write_json(path.with_suffix(".telemetry.json"), result.telemetry)


def _smoke_projection(timings: Mapping[str, Any]) -> dict[str, Any]:
    transition_seconds = float(timings["candidate_transition_seconds"])
    transition_count = int(timings["candidate_transition_count"])
    training_seconds = float(timings["training_seconds"])
    update_count = int(timings["training_updates"])
    persistence_seconds = float(timings["persistence_seconds"])
    projected_transition = transition_seconds / transition_count * 273_960_960
    projected_training = training_seconds / update_count * 750
    projected_persistence = persistence_seconds * 100.0
    projected_fixed = 120.0
    total = math.fsum(
        (projected_transition, projected_training, projected_persistence, projected_fixed)
    )
    return {
        "forward_record_transitions": 122_931_200,
        "reverse_transitions": 140_492_800,
        "forward_evaluation_transitions": 10_536_960,
        "base_candidate_transition_work": 273_960_960,
        "measured_transition_seconds": transition_seconds,
        "measured_transition_count": transition_count,
        "candidate_transitions_per_second": transition_count / transition_seconds,
        "projected_transition_seconds": projected_transition,
        "projected_training_seconds": projected_training,
        "projected_persistence_seconds": projected_persistence,
        "projected_audit_evaluator_report_seconds": projected_fixed,
        "projected_remaining_seconds": total,
    }


def run_resource_smoke(
    run_dir: Path,
    train_states: np.ndarray,
    train_labels: np.ndarray,
    runtime: candidate.CandidateRuntime,
    governor: ResourceGovernor,
) -> dict[str, Any]:
    path_ids = _role_ids("resource_smoke")
    labels = torch.as_tensor(train_labels[:8], dtype=torch.long, device=runtime.device)
    state = torch.as_tensor(train_states[:8], dtype=torch.float64, device=runtime.device)
    outer_timings: dict[str, list[float]] = {"forward": [], "null_reverse": [], "zero_model_reverse": []}
    transition_count = 0
    for outer_step in range(8):
        governor.admit(
            "smoke_forward_outer",
            predicted_seconds=PRE_SMOKE_QUANTUM_SECONDS,
        )
        started = time.perf_counter()
        health_parts: list[Mapping[str, Any]] = []
        for phase in range(7):
            state, _, health = candidate.candidate_forward_phase(
                state,
                path_ids,
                outer_step=outer_step,
                phase=phase,
                root_seed=SEEDS["resource_smoke"],
                sample_steps=128,
                runtime=runtime,
            )
            health_parts.append(health)
        torch.cuda.synchronize(runtime.device)
        zero_fields = (
            "candidate_invalid_input_count",
            "candidate_invalid_output_count",
            "candidate_nonfinite_count",
            "candidate_negative_bracket_width_count",
            "candidate_bracket_order_invalid_count",
            "candidate_correction_count",
            "candidate_clipping_count",
            "candidate_floor_count",
            "candidate_limiter_count",
            "candidate_projection_count",
            "candidate_renormalization_count",
            "candidate_invalid_lane_count",
            "candidate_approximation_mismatch_count",
            "state_nonfinite_count",
            "state_negative_count",
        )
        _require(
            all(sum(int(row[name].item()) for row in health_parts) == 0 for name in zero_fields)
            and max(float(row["maximum_mass_error"].item()) for row in health_parts) <= 2e-12
            and max(float(row["maximum_pair_total_error"].item()) for row in health_parts) <= 2e-12,
            "resource-smoke forward outer step failed candidate health",
        )
        outer_timings["forward"].append(time.perf_counter() - started)
        governor.complete(
            "smoke_forward_outer",
            transitions=8 * 7 * EDGES_PER_PHASE,
            synchronize=False,
        )
        transition_count += 8 * 7 * EDGES_PER_PHASE
    reverse_start = state.clone()
    for controller, name, model in (
        ("null", "null_reverse", None),
        ("learned", "zero_model_reverse", core.make_model().to(runtime.device).eval()),
    ):
        state = reverse_start.clone()
        for outer_step in range(127, 119, -1):
            governor.admit(
                f"smoke_{name}_outer",
                predicted_seconds=PRE_SMOKE_QUANTUM_SECONDS,
            )
            started = time.perf_counter()
            state, _ = candidate.candidate_reverse_outer_step(
                state,
                labels,
                path_ids,
                outer_step=outer_step,
                controller=controller,  # type: ignore[arg-type]
                root_seed=SEEDS["resource_smoke"],
                runtime=runtime,
                model=model,
            )
            outer_timings[name].append(time.perf_counter() - started)
            governor.complete(
                f"smoke_{name}_outer", transitions=8 * 7 * 4 * EDGES_PER_PHASE,
                model_evaluations=(8 * 7 * 2 if model is not None else 0),
                synchronize=False,
            )
            transition_count += 8 * 7 * 4 * EDGES_PER_PHASE

    synthetic_train = _synthetic_records(1_000, SEEDS["resource_smoke"])
    synthetic_validation = _synthetic_records(400, SEEDS["resource_smoke"] ^ 0x51)
    governor.admit(
        "smoke_training",
        predicted_seconds=PRE_SMOKE_QUANTUM_SECONDS,
    )
    training_started = time.perf_counter()
    core.train_jacobi_ddpm(
        synthetic_train,
        synthetic_validation,
        device=runtime.device,
        updates=25,
        batch_size=64,
        learning_rate=2e-4,
        ema_decay=0.999,
        validation_interval=25,
        seed=SEEDS["training_model"],
    )
    torch.cuda.synchronize(runtime.device)
    training_seconds = time.perf_counter() - training_started
    governor.complete("smoke_training", model_evaluations=25 * 64 + 400)

    smoke_path = run_dir / "resource_smoke/synthetic_records.npz"
    governor.admit(
        "smoke_persistence",
        predicted_seconds=1.0,
        predicted_bytes=_record_cohort_predicted_bytes(350) + 8 * 784 * 8,
    )
    persistence_started = time.perf_counter()
    _write_npz(
        smoke_path,
        **{f"train_{name}": value for name, value in _record_arrays(synthetic_train).items()},
        **{f"validation_{name}": value for name, value in _record_arrays(synthetic_validation).items()},
        smoke_population=state.detach().cpu().numpy(),
    )
    persistence_seconds = time.perf_counter() - persistence_started
    governor.complete("smoke_persistence", synchronize=False)
    ledger = _read_json(run_dir / "resource_ledger.json")
    transition_seconds = math.fsum(
        seconds for values in outer_timings.values() for seconds in values
    )
    timings: dict[str, Any] = {
        "schema": VERSION + "-resource-smoke",
        "outer_step_seconds": outer_timings,
        "outer_step_medians": {
            name: float(np.median(values)) for name, values in outer_timings.items()
        },
        "outer_step_maxima": {name: max(values) for name, values in outer_timings.items()},
        "candidate_transition_seconds": transition_seconds,
        "candidate_transition_count": transition_count,
        "training_seconds": training_seconds,
        "training_updates": 25,
        "persistence_seconds": persistence_seconds,
        "synthetic_record_count": 1_400,
        "storage_bytes_per_record": smoke_path.stat().st_size / 1_400.0,
        "peak_cuda_allocated_bytes": int(ledger["peak_cuda_allocated_bytes"]),
        "peak_cuda_reserved_bytes": int(ledger["peak_cuda_reserved_bytes"]),
        "environment": _read_json(run_dir / "environment.json"),
    }
    timings["projection"] = _smoke_projection(timings)
    _write_json(run_dir / "resource_smoke/timings.json", timings)
    _append_stage(run_dir, "resource_smoke", "complete")
    return timings


def _expected_outer_seconds(run_dir: Path, direction: str) -> float:
    path = run_dir / "resource_smoke/timings.json"
    if not path.is_file():
        return 0.0
    timings = _read_json(path)
    maxima = timings["outer_step_maxima"]
    if direction == "forward":
        return float(maxima["forward"])
    return max(float(maxima["null_reverse"]), float(maxima["zero_model_reverse"]))


def _declared_quantum_floor(
    run_dir: Path, kind: str, kind_admission_ordinal: int
) -> float:
    """Rebuild the minimum declared duration from fixed or measured authority."""

    if kind in {
        "prepare_candidate_backend",
        "candidate_audit",
        "smoke_forward_outer",
        "smoke_null_reverse_outer",
        "smoke_zero_model_reverse_outer",
        "smoke_training",
    }:
        return PRE_SMOKE_QUANTUM_SECONDS
    if kind == "data_roles_write":
        return 30.0
    if kind == "failure_terminalization":
        return FAILURE_TERMINALIZATION_SECONDS
    if kind == "smoke_persistence":
        return 1.0
    if kind in RESOURCE_STAGE_REMAINING:
        fixed_seconds = float(_stage_resource_projection(run_dir, kind)["fixed_seconds"])
        return min(60.0 - 1e-9, max(1e-9, fixed_seconds))

    smoke_path = run_dir / "resource_smoke/timings.json"
    _require(smoke_path.is_file(), f"measured quantum lacks resource smoke: {kind}")
    smoke = _read_json(smoke_path)
    persistence = max(1e-9, float(smoke["persistence_seconds"]))
    forward_kinds = {
        "oracle_forward_outer",
        "records_train_outer",
        "records_validation_outer",
        "objective_forward_outer",
    }
    reverse_kinds = {
        "null_reverse_outer",
        "learned_reverse_outer",
        "oracle_reverse_outer",
    }
    if kind in forward_kinds | reverse_kinds:
        base = _expected_outer_seconds(
            run_dir, "forward" if kind in forward_kinds else "reverse"
        )
        if int(kind_admission_ordinal) % 128 == 0:
            base += persistence * (3.0 if kind in reverse_kinds else 1.0)
        return float(base)
    if kind == "training_250_updates":
        return float(smoke["training_seconds"]) * 10.0
    if kind == "oracle_assembly_write":
        return persistence
    if kind == "training_finalization_write":
        return persistence * 5.0
    if kind == "population_assembly_write":
        return persistence * 30.0
    if kind == "evaluator_batch":
        return 30.0
    raise IntegrityFailure(f"unknown resource admission kind: {kind}")


def _run_reverse_cohorts(
    run_dir: Path,
    stage_directory: Path,
    starts: np.ndarray,
    labels: np.ndarray,
    path_ids: np.ndarray,
    sample_ids: np.ndarray,
    *,
    controller: str,
    root_seed: int,
    runtime: candidate.CandidateRuntime,
    governor: ResourceGovernor,
    model: core.EulerianJacobiDDPMModel | None = None,
    oracle_targets: np.ndarray | None = None,
    image_directory: Path | None = None,
) -> core.SamplingResult:
    parts: list[core.SamplingResult] = []
    predicted = _expected_outer_seconds(run_dir, "reverse")
    persistence = max(
        1e-9,
        3.0
        * float(
            _read_json(run_dir / "resource_smoke/timings.json")[
                "persistence_seconds"
            ]
        ),
    )
    for cohort_index, start in enumerate(range(0, len(path_ids), 8)):
        stop = min(start + 8, len(path_ids))
        quantum_kind = f"{controller}_reverse_outer"
        predicted_bytes = _sampling_cohort_predicted_bytes(
            stop - start, include_images=image_directory is not None
        )
        governor.admit(
            quantum_kind,
            predicted_seconds=predicted,
            predicted_bytes=predicted_bytes,
        )
        result = candidate.reverse_sample_candidate(
            starts[start:stop],
            labels[start:stop],
            path_ids[start:stop],
            controller=controller,  # type: ignore[arg-type]
            root_seed=root_seed,
            runtime=runtime,
            model=model,
            oracle_targets=(None if oracle_targets is None else oracle_targets[start:stop]),
            outer_step_callback=_governed_outer_callback(
                governor,
                quantum_kind,
                paths=stop - start,
                predicted_seconds=predicted,
                predicted_bytes=predicted_bytes,
                persistence_seconds=persistence,
                defer_final_completion=True,
            ),
        )
        cohort_path = stage_directory / f"cohort_{cohort_index:03d}.npz"
        _save_sampling_npz(
            cohort_path,
            result,
            labels[start:stop],
            path_ids[start:stop],
            sample_ids[start:stop],
        )
        if image_directory is not None:
            _, rendered = rasterize_population(result.final_states)
            _save_individual_pngs(image_directory, rendered, sample_ids[start:stop])
        governor.complete(
            quantum_kind,
            transitions=(stop - start) * 7 * 4 * EDGES_PER_PHASE,
            synchronize=False,
        )
        parts.append(result)
    return _merge_sampling(parts)


def run_oracle_control(
    run_dir: Path,
    validation_states: np.ndarray,
    validation_labels: np.ndarray,
    validation_arff_indices: np.ndarray,
    runtime: candidate.CandidateRuntime,
    governor: ResourceGovernor,
) -> tuple[core.SamplingResult, core.SamplingResult, dict[str, Any]]:
    _require(not any((run_dir / "training").iterdir()), "training artifacts exist before oracle control")
    _admit_major_stage(run_dir, governor, "oracle_stage")
    positions = ORACLE_VALIDATION_POSITIONS
    targets = np.asarray(validation_states[positions], dtype=np.float64)
    labels = np.asarray(validation_labels[positions], dtype=np.int64)
    arff_indices = np.asarray(validation_arff_indices[positions], dtype=np.int64)
    path_ids = _role_ids("oracle")
    sample_ids = np.asarray([f"oracle-{value:05x}" for value in path_ids], dtype=np.str_)
    _write_npz(
        run_dir / "oracle_control/authority.npz",
        source_targets=targets,
        requested_labels=labels,
        arff_indices=arff_indices,
        path_ids=path_ids,
        sample_ids=sample_ids,
    )
    forward_parts: list[np.ndarray] = []
    predicted_forward = _expected_outer_seconds(run_dir, "forward")
    forward_persistence = max(
        1e-9,
        float(
            _read_json(run_dir / "resource_smoke/timings.json")[
                "persistence_seconds"
            ]
        ),
    )
    for cohort_index, start in enumerate(range(0, 10, 8)):
        stop = min(start + 8, 10)
        predicted_bytes = _forward_terminal_cohort_predicted_bytes(stop - start)
        governor.admit(
            "oracle_forward_outer",
            predicted_seconds=predicted_forward,
            predicted_bytes=predicted_bytes,
        )
        terminal, telemetry = candidate.forward_terminal_states_candidate(
            targets[start:stop],
            path_ids[start:stop],
            root_seed=SEEDS["oracle_forward"],
            runtime=runtime,
            outer_step_callback=_governed_outer_callback(
                governor,
                "oracle_forward_outer",
                paths=stop - start,
                predicted_seconds=predicted_forward,
                predicted_bytes=predicted_bytes,
                persistence_seconds=forward_persistence,
                defer_final_completion=True,
            ),
        )
        _write_npz(
            run_dir / f"oracle_control/stages/forward_terminal/cohort_{cohort_index:03d}.npz",
            source_targets=targets[start:stop],
            terminal_starts=terminal,
            requested_labels=labels[start:stop],
            arff_indices=arff_indices[start:stop],
            path_ids=path_ids[start:stop],
            sample_ids=sample_ids[start:stop],
        )
        _write_json(
            run_dir / f"oracle_control/stages/forward_terminal/cohort_{cohort_index:03d}.telemetry.json",
            telemetry,
        )
        governor.complete(
            "oracle_forward_outer",
            transitions=(stop - start) * 7 * EDGES_PER_PHASE,
            synchronize=False,
        )
        forward_parts.append(terminal)
    terminal_starts = np.concatenate(forward_parts)
    null = _run_reverse_cohorts(
        run_dir,
        run_dir / "oracle_control/stages/null",
        terminal_starts,
        labels,
        path_ids,
        sample_ids,
        controller="null",
        root_seed=SEEDS["oracle_reverse"],
        runtime=runtime,
        governor=governor,
        image_directory=run_dir / "oracle_control/images/null",
    )
    oracle = _run_reverse_cohorts(
        run_dir,
        run_dir / "oracle_control/stages/oracle",
        terminal_starts,
        labels,
        path_ids,
        sample_ids,
        controller="oracle",
        root_seed=SEEDS["oracle_reverse"],
        runtime=runtime,
        governor=governor,
        oracle_targets=targets,
        image_directory=run_dir / "oracle_control/images/oracle",
    )
    governor.admit(
        "oracle_assembly_write",
        predicted_seconds=max(1e-9, float(_read_json(run_dir / "resource_smoke/timings.json")["persistence_seconds"])),
        predicted_bytes=2 * _sampling_cohort_predicted_bytes(10, include_images=True),
    )
    _save_sampling_npz(
        run_dir / "oracle_control/null.npz",
        null,
        labels,
        path_ids,
        sample_ids,
        source_targets=targets,
        terminal_starts=terminal_starts,
    )
    _save_sampling_npz(
        run_dir / "oracle_control/oracle.npz",
        oracle,
        labels,
        path_ids,
        sample_ids,
        source_targets=targets,
        terminal_starts=terminal_starts,
    )
    metrics = oracle_control_metrics(
        targets, null.final_states, oracle.final_states, null.telemetry, oracle.telemetry
    )
    metrics["anchor_raw_mass_l1"] = {
        str(anchor): {
            "null": np.sum(np.abs(null.anchors[anchor] - targets), axis=1),
            "oracle": np.sum(np.abs(oracle.anchors[anchor] - targets), axis=1),
        }
        for anchor in (0, 32, 64, 96, 128)
    }
    _write_json(run_dir / "oracle_control/metrics.json", metrics)
    _, target_images = rasterize_population(targets)
    _, null_images = rasterize_population(null.final_states)
    _, oracle_images = rasterize_population(oracle.final_states)
    _save_individual_pngs(
        run_dir / "oracle_control/images/targets", target_images, sample_ids
    )
    _save_individual_pngs(run_dir / "oracle_control/images/targets", target_images, sample_ids)
    write_contact_sheet(
        run_dir / "oracle_control/contact_sheet.png",
        np.stack([row for triple in zip(target_images, null_images, oracle_images, strict=True) for row in triple]),
        columns=3,
        captions=[f"{int(label)}:{name}" for label in labels for name in ("target", "null", "oracle")],
    )
    if not gate_c_passed(metrics):
        _append_stage(run_dir, "oracle_control", "failed")
        raise OracleControlFailed("complete ten-class oracle positive control failed")
    complete = {
        "schema": VERSION + "-oracle-complete",
        "passed": 1,
        "authority_sha256": _file_sha256(run_dir / "oracle_control/authority.npz"),
        "null_sha256": _file_sha256(run_dir / "oracle_control/null.npz"),
        "oracle_sha256": _file_sha256(run_dir / "oracle_control/oracle.npz"),
        "metrics_sha256": _file_sha256(run_dir / "oracle_control/metrics.json"),
    }
    _write_json(run_dir / "oracle_control/COMPLETE.json", complete)
    _append_stage(run_dir, "oracle_control", "complete")
    governor.complete("oracle_assembly_write", synchronize=False)
    return null, oracle, metrics


def _sort_records(dataset: core.ForwardRecordDataset) -> core.ForwardRecordDataset:
    order = np.lexsort((dataset.phase, dataset.outer_steps, dataset.path_ids))
    return core.ForwardRecordDataset(
        **{
            name: np.ascontiguousarray(np.asarray(getattr(dataset, name))[order])
            for name in core.ForwardRecordDataset.__dataclass_fields__
        }
    )


def build_forward_record_caches(
    run_dir: Path,
    train_states: np.ndarray,
    train_labels: np.ndarray,
    validation_states: np.ndarray,
    validation_labels: np.ndarray,
    runtime: candidate.CandidateRuntime,
    governor: ResourceGovernor,
) -> tuple[core.ForwardRecordDataset, core.ForwardRecordDataset]:
    _admit_major_stage(run_dir, governor, "forward_record_caches_stage")
    index: dict[str, Any] = {
        "schema": VERSION + "-forward-record-index",
        "target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "transition_dtype": "float64",
        "stored_state_dtype": "float32",
        "stored_target_dtype": "float32",
        "cohorts": {"train": [], "validation": []},
    }
    predicted = _expected_outer_seconds(run_dir, "forward")
    persistence = max(
        1e-9,
        float(
            _read_json(run_dir / "resource_smoke/timings.json")[
                "persistence_seconds"
            ]
        ),
    )
    completed: dict[str, list[core.ForwardRecordDataset]] = {"train": [], "validation": []}
    for role, states, labels, path_ids, seed in (
        ("train", train_states, train_labels, _role_ids("training"), SEEDS["records_train"]),
        (
            "validation",
            validation_states,
            validation_labels,
            _role_ids("validation"),
            SEEDS["records_validation"],
        ),
    ):
        for cohort_index, start in enumerate(range(0, len(path_ids), 8)):
            stop = min(start + 8, len(path_ids))
            predicted_bytes = _record_cohort_predicted_bytes(stop - start)
            governor.admit(
                f"records_{role}_outer",
                predicted_seconds=predicted,
                predicted_bytes=predicted_bytes,
            )
            yielded = list(
                candidate.iter_forward_record_batches_candidate(
                    states[start:stop],
                    labels[start:stop],
                    path_ids[start:stop],
                    root_seed=seed,
                    runtime=runtime,
                    outer_step_callback=_governed_outer_callback(
                        governor,
                        f"records_{role}_outer",
                        paths=stop - start,
                        predicted_seconds=predicted,
                        predicted_bytes=predicted_bytes,
                        persistence_seconds=persistence,
                        defer_final_completion=True,
                    ),
                )
            )
            _require(len(yielded) == 1, "one bounded input cohort produced multiple record cohorts")
            part = _sort_records(yielded[0])
            cohort_path = run_dir / f"forward_records/{role}/cohort_{cohort_index:03d}.npz"
            _write_npz(cohort_path, **_record_arrays(part))
            governor.complete(
                f"records_{role}_outer",
                transitions=(stop - start) * 7 * EDGES_PER_PHASE,
                synchronize=False,
            )
            record = {
                "cohort": cohort_index,
                "path_start": int(path_ids[start]),
                "path_stop_exclusive": int(path_ids[stop - 1]) + 1,
                "path_count": int(stop - start),
                "record_count": len(part),
                "sha256": _file_sha256(cohort_path),
            }
            index["cohorts"][role].append(record)
            completed[role].append(part)
            _write_json(run_dir / "forward_records/index.json", index)
    train = _sort_records(_merge_records(completed["train"]))
    validation = _sort_records(_merge_records(completed["validation"]))
    _require(
        len(train) == 1_000
        and len(validation) == 400
        and np.unique(train.path_ids).size == 250
        and np.unique(validation.path_ids).size == 100
        and set(train.path_ids).isdisjoint(set(validation.path_ids)),
        "assembled forward-record contract changed",
    )
    for dataset in (train, validation):
        unique, counts = np.unique(dataset.path_ids, return_counts=True)
        _require(
            len(unique) > 0
            and np.all(counts == 4)
            and dataset.targets.dtype == np.float32
            and dataset.later_states.dtype == np.float32,
            "forward records are not four approximate-candidate rows per whole path",
        )
    index["assembled"] = {
        "train_records": len(train),
        "validation_records": len(validation),
        "train_path_ids_sha256": _array_sha256(np.asarray(np.unique(train.path_ids), dtype=np.int64)),
        "validation_path_ids_sha256": _array_sha256(
            np.asarray(np.unique(validation.path_ids), dtype=np.int64)
        ),
    }
    _write_json(run_dir / "forward_records/index.json", index)
    _append_stage(run_dir, "forward_record_caches", "complete")
    return train, validation


def train_candidate_model(
    run_dir: Path,
    train: core.ForwardRecordDataset,
    validation: core.ForwardRecordDataset,
    runtime: candidate.CandidateRuntime,
    governor: ResourceGovernor,
) -> tuple[core.EulerianJacobiDDPMModel, core.TrainingResult]:
    _admit_major_stage(run_dir, governor, "training_stage")
    smoke = _read_json(run_dir / "resource_smoke/timings.json")
    predicted_block = float(smoke["training_seconds"]) * 10.0
    checkpoint_bytes = 4 * 1024 * 1024
    governor.admit(
        "training_250_updates",
        predicted_seconds=predicted_block,
        predicted_bytes=checkpoint_bytes,
    )

    def checkpoint(record: Mapping[str, Any]) -> None:
        update = int(record["completed_update"])
        _require(update in {250, 500, 750}, "unexpected training checkpoint update")
        _write_torch(
            run_dir / f"training/checkpoint_{update:04d}.pt",
            {
                "schema": VERSION + "-training-checkpoint",
                "target_semantics": CANDIDATE_TARGET_SEMANTICS,
                **dict(record),
            },
        )
        governor.complete("training_250_updates", model_evaluations=250 * 64 + 400)
        if update < 750:
            governor.admit(
                "training_250_updates",
                predicted_seconds=predicted_block,
                predicted_bytes=checkpoint_bytes,
            )

    training = core.train_jacobi_ddpm(
        train,
        validation,
        device=runtime.device,
        updates=750,
        batch_size=64,
        learning_rate=2e-4,
        ema_decay=0.999,
        validation_interval=250,
        seed=SEEDS["training_model"],
        checkpoint_callback=checkpoint,
    )
    governor.admit(
        "training_finalization_write",
        predicted_seconds=max(1e-9, float(smoke["persistence_seconds"]) * 5.0),
        predicted_bytes=4 * 1024 * 1024,
    )
    history = [dict(row) for row in training.history]
    selection = select_earliest_checkpoint(history)
    _require(
        selection["selected_update"] == int(training.selected_update)
        and math.isclose(
            selection["selected_validation_normalized_mse"],
            float(training.selected_validation_mse),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "training routine violated the frozen earliest-minimum rule",
    )
    selected_path = run_dir / "training/selected_checkpoint.pt"
    _write_torch(
        selected_path,
        {
            "schema": VERSION + "-selected-checkpoint",
            "state_dict": training.selected_state_dict,
            "selected_update": int(training.selected_update),
            "selected_validation_normalized_mse": float(training.selected_validation_mse),
            "training_target_energy": float(training.training_target_energy),
            "completed_updates": int(training.completed_updates),
            "model_parameter_count": 34_974,
        },
    )
    selection.update(
        {
            "checkpoint_sha256": _file_sha256(selected_path),
            "completed_updates": int(training.completed_updates),
            "training_target_energy": float(training.training_target_energy),
        }
    )
    _write_json(run_dir / "training/history.json", {"rows": history})
    _write_json(run_dir / "training/selection.json", selection)
    model = core.make_model()
    _require(sum(parameter.numel() for parameter in model.parameters()) == 34_974, "model shape changed")
    model.load_state_dict(training.selected_state_dict)
    model = model.to(runtime.device).eval()
    _append_stage(run_dir, "training", "complete", selected_update=training.selected_update)
    governor.complete("training_finalization_write", synchronize=False)
    return model, training


def _forward_terminal_starts_cohorts(
    run_dir: Path,
    targets: np.ndarray,
    labels: np.ndarray,
    path_ids: np.ndarray,
    sample_ids: np.ndarray,
    *,
    runtime: candidate.CandidateRuntime,
    governor: ResourceGovernor,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    predicted = _expected_outer_seconds(run_dir, "forward")
    persistence = max(
        1e-9,
        float(
            _read_json(run_dir / "resource_smoke/timings.json")[
                "persistence_seconds"
            ]
        ),
    )
    for cohort_index, start in enumerate(range(0, len(path_ids), 8)):
        stop = min(start + 8, len(path_ids))
        predicted_bytes = _forward_terminal_cohort_predicted_bytes(stop - start)
        governor.admit(
            "objective_forward_outer",
            predicted_seconds=predicted,
            predicted_bytes=predicted_bytes,
        )
        terminal, telemetry = candidate.forward_terminal_states_candidate(
            targets[start:stop],
            path_ids[start:stop],
            root_seed=SEEDS["forward_terminal_forward"],
            runtime=runtime,
            outer_step_callback=_governed_outer_callback(
                governor,
                "objective_forward_outer",
                paths=stop - start,
                predicted_seconds=predicted,
                predicted_bytes=predicted_bytes,
                persistence_seconds=persistence,
                defer_final_completion=True,
            ),
        )
        cohort_path = run_dir / f"populations/stages/forward_terminal_starts/cohort_{cohort_index:03d}.npz"
        _write_npz(
            cohort_path,
            source_targets=targets[start:stop],
            terminal_starts=terminal,
            requested_labels=labels[start:stop],
            path_ids=path_ids[start:stop],
            sample_ids=sample_ids[start:stop],
        )
        _write_json(cohort_path.with_suffix(".telemetry.json"), telemetry)
        governor.complete(
            "objective_forward_outer",
            transitions=(stop - start) * 7 * EDGES_PER_PHASE,
            synchronize=False,
        )
        parts.append(terminal)
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)


def _forward_direct_marker(
    targets: np.ndarray,
    null: core.SamplingResult,
    learned: core.SamplingResult,
) -> dict[str, Any]:
    null_l1 = np.sum(np.abs(null.final_states - targets), axis=1, dtype=np.float64)
    learned_l1 = np.sum(np.abs(learned.final_states - targets), axis=1, dtype=np.float64)
    null_total = float(np.sum(null_l1))
    health = bool(
        _SAMPLING_REQUIRED_TELEMETRY.issubset(null.telemetry)
        and _SAMPLING_REQUIRED_TELEMETRY.issubset(learned.telemetry)
        and int(null.telemetry["finite"]) == 1
        and int(null.telemetry["nonnegative"]) == 1
        and int(learned.telemetry["finite"]) == 1
        and int(learned.telemetry["nonnegative"]) == 1
        and all(
            int(telemetry[name]) == 0
            for telemetry in (null.telemetry, learned.telemetry)
            for name in _SAMPLING_ZERO_COUNTERS
        )
        and max(
            float(null.telemetry["maximum_mass_error"]),
            float(learned.telemetry["maximum_mass_error"]),
            float(null.telemetry["maximum_pair_total_error"]),
            float(learned.telemetry["maximum_pair_total_error"]),
        )
        <= 2e-12
    )
    wins = int(np.sum(learned_l1 < null_l1))
    relative = (null_total - float(np.sum(learned_l1))) / null_total
    controller_rms = float(learned.telemetry["controller_rms"])
    passed = health and wins >= 12 and relative >= 0.01 and math.isfinite(controller_rms) and controller_rms > 0.0
    return {
        "gate_type": "diagnostic threshold",
        "null_final_raw_mass_l1": null_l1,
        "learned_final_raw_mass_l1": learned_l1,
        "learned_l1_win_count": wins,
        "aggregate_relative_l1_improvement": relative,
        "learned_controller_rms": controller_rms,
        "trajectory_health_passed": int(health),
        "passed": int(passed),
    }


def sample_objective_populations(
    run_dir: Path,
    validation_states: np.ndarray,
    validation_labels: np.ndarray,
    model: core.EulerianJacobiDDPMModel,
    runtime: candidate.CandidateRuntime,
    governor: ResourceGovernor,
    oracle_null: core.SamplingResult,
    oracle: core.SamplingResult,
) -> dict[str, Any]:
    if governor is not None:
        _admit_major_stage(run_dir, governor, "objective_sampling_stage")
    prior_path_ids = _role_ids("prior")
    prior_labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    prior_sample_ids = np.asarray([f"prior-{value:05x}" for value in prior_path_ids], dtype=np.str_)
    prior_starts = core.sample_dirichlet_starts(prior_path_ids, root_seed=SEEDS["prior_start"])
    prior_authority_path = run_dir / "populations/prior_start_authority.npz"
    _write_npz(
        prior_authority_path,
        prior_starts=prior_starts,
        requested_labels=prior_labels,
        path_ids=prior_path_ids,
        sample_ids=prior_sample_ids,
    )
    _write_json(
        run_dir / "populations/prior_start_authority.json",
        {
            "schema": VERSION + "-prior-start-authority",
            "committed_before_sampling": 1,
            "label_independent_start_law": "Dirichlet(1,...,1)",
            "npz_sha256": _file_sha256(prior_authority_path),
            "path_ids_sha256": _array_sha256(prior_path_ids),
        },
    )
    null_prior = _run_reverse_cohorts(
        run_dir,
        run_dir / "populations/stages/null_prior",
        prior_starts,
        prior_labels,
        prior_path_ids,
        prior_sample_ids,
        controller="null",
        root_seed=SEEDS["prior_reverse"],
        runtime=runtime,
        governor=governor,
        image_directory=run_dir / "populations/images/prior/null",
    )
    learned_prior = _run_reverse_cohorts(
        run_dir,
        run_dir / "populations/stages/learned_prior",
        prior_starts,
        prior_labels,
        prior_path_ids,
        prior_sample_ids,
        controller="learned",
        root_seed=SEEDS["prior_reverse"],
        runtime=runtime,
        governor=governor,
        model=model,
        image_directory=run_dir / "populations/images/prior/learned",
    )

    forward_positions = FORWARD_VALIDATION_POSITIONS
    forward_targets = np.asarray(validation_states[forward_positions], dtype=np.float64)
    forward_labels = np.asarray(validation_labels[forward_positions], dtype=np.int64)
    forward_path_ids = _role_ids("forward_terminal")
    forward_sample_ids = np.asarray(
        [f"forward-{value:05x}" for value in forward_path_ids], dtype=np.str_
    )
    forward_starts = _forward_terminal_starts_cohorts(
        run_dir,
        forward_targets,
        forward_labels,
        forward_path_ids,
        forward_sample_ids,
        runtime=runtime,
        governor=governor,
    )
    null_forward = _run_reverse_cohorts(
        run_dir,
        run_dir / "populations/stages/null_forward_terminal",
        forward_starts,
        forward_labels,
        forward_path_ids,
        forward_sample_ids,
        controller="null",
        root_seed=SEEDS["forward_terminal_reverse"],
        runtime=runtime,
        governor=governor,
        image_directory=run_dir / "populations/images/forward/null",
    )
    learned_forward = _run_reverse_cohorts(
        run_dir,
        run_dir / "populations/stages/learned_forward_terminal",
        forward_starts,
        forward_labels,
        forward_path_ids,
        forward_sample_ids,
        controller="learned",
        root_seed=SEEDS["forward_terminal_reverse"],
        runtime=runtime,
        governor=governor,
        model=model,
        image_directory=run_dir / "populations/images/forward/learned",
    )
    oracle_authority = np.load(run_dir / "oracle_control/authority.npz", allow_pickle=False)
    oracle_targets = np.asarray(oracle_authority["source_targets"], dtype=np.float64)
    oracle_labels = np.asarray(oracle_authority["requested_labels"], dtype=np.int64)
    oracle_path_ids = np.asarray(oracle_authority["path_ids"], dtype=np.int64)
    oracle_sample_ids = np.asarray(oracle_authority["sample_ids"], dtype=np.str_)
    oracle_authority.close()
    oracle_complete = _read_json(run_dir / "oracle_control/COMPLETE.json")
    _require(
        oracle_complete["null_sha256"]
        == _file_sha256(run_dir / "oracle_control/null.npz")
        and oracle_complete["oracle_sha256"]
        == _file_sha256(run_dir / "oracle_control/oracle.npz"),
        "oracle control was changed instead of reused",
    )

    stages = {
        "null_prior": (null_prior, prior_labels, prior_path_ids, prior_sample_ids),
        "learned_prior": (learned_prior, prior_labels, prior_path_ids, prior_sample_ids),
        "null_forward_terminal": (
            null_forward,
            forward_labels,
            forward_path_ids,
            forward_sample_ids,
        ),
        "learned_forward_terminal": (
            learned_forward,
            forward_labels,
            forward_path_ids,
            forward_sample_ids,
        ),
        "null_oracle": (oracle_null, oracle_labels, oracle_path_ids, oracle_sample_ids),
        "oracle": (oracle, oracle_labels, oracle_path_ids, oracle_sample_ids),
    }
    if governor is not None:
        governor.admit(
            "population_assembly_write",
            predicted_seconds=max(
                1e-9,
                float(_read_json(run_dir / "resource_smoke/timings.json")["persistence_seconds"])
                * 30.0,
            ),
            predicted_bytes=8 * 1024 * 1024,
        )
    for name, (result, labels, path_ids, sample_ids) in stages.items():
        if "forward_terminal" in name:
            extra = {"source_targets": forward_targets, "terminal_starts": forward_starts}
        elif name in {"null_oracle", "oracle"}:
            extra = {"source_targets": oracle_targets, "terminal_starts": result.starts}
        else:
            extra = {}
        _save_sampling_npz(
            run_dir / f"populations/{name}.npz",
            result,
            labels,
            path_ids,
            sample_ids,
            **extra,
        )

    raw = {
        "null_prior": null_prior.final_states,
        "learned_prior": learned_prior.final_states,
        "null_forward_terminal": null_forward.final_states,
        "learned_forward_terminal": learned_forward.final_states,
        "forward_targets": forward_targets,
        "null_oracle": oracle_null.final_states,
        "oracle": oracle.final_states,
        "oracle_targets": oracle_targets,
    }
    demixed: dict[str, np.ndarray] = {}
    rendered: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        demixed[name], rendered[name] = rasterize_population(values)
    identities = {
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
    _write_npz(run_dir / "populations/identities.npz", **identities)
    _write_npz(run_dir / "populations/raw_populations.npz", **raw, **identities)
    _write_npz(run_dir / "populations/demixed_populations.npz", **demixed, **identities)
    _write_npz(run_dir / "populations/uint8_populations.npz", **rendered, **identities)
    _write_json(
        run_dir / "populations/telemetry.json",
        {
            "null_prior": null_prior.telemetry,
            "learned_prior": learned_prior.telemetry,
            "null_forward_terminal": null_forward.telemetry,
            "learned_forward_terminal": learned_forward.telemetry,
            "null_oracle": oracle_null.telemetry,
            "oracle": oracle.telemetry,
        },
    )
    stage_index = {
        "schema": VERSION + "-population-stage-index",
        "oracle_recomputed_after_training": 0,
        "oracle_complete_sha256": _file_sha256(run_dir / "oracle_control/COMPLETE.json"),
        "stages": {
            name: {
                "row_count": int(len(result.final_states)),
                "path_ids_sha256": _array_sha256(path_ids),
                "assembled_sha256": _file_sha256(run_dir / f"populations/{name}.npz"),
                **(
                    {
                        "reused_source_path": f"oracle_control/{'null' if name == 'null_oracle' else 'oracle'}.npz",
                        "reused_source_sha256": _file_sha256(
                            run_dir
                            / f"oracle_control/{'null' if name == 'null_oracle' else 'oracle'}.npz"
                        ),
                    }
                    if name in {"null_oracle", "oracle"}
                    else {}
                ),
            }
            for name, (result, _, path_ids, _) in stages.items()
        },
    }
    _write_json(run_dir / "populations/stage_index.json", stage_index)

    for directory, images, ids in (
        ("forward/targets", rendered["forward_targets"], forward_sample_ids),
        ("oracle/targets", rendered["oracle_targets"], oracle_sample_ids),
        ("oracle/null", rendered["null_oracle"], oracle_sample_ids),
        ("oracle/oracle", rendered["oracle"], oracle_sample_ids),
    ):
        _save_individual_pngs(run_dir / f"populations/images/{directory}", images, ids)
    write_contact_sheet(
        run_dir / "populations/contact_sheets/prior_null_learned.png",
        np.stack(
            [row for pair in zip(rendered["null_prior"], rendered["learned_prior"], strict=True) for row in pair]
        ),
        columns=4,
        captions=[f"{int(label)}:{kind}" for label in prior_labels for kind in ("null", "learned")],
    )
    write_contact_sheet(
        run_dir / "populations/contact_sheets/forward_target_null_learned.png",
        np.stack(
            [
                row
                for triple in zip(
                    rendered["forward_targets"],
                    rendered["null_forward_terminal"],
                    rendered["learned_forward_terminal"],
                    strict=True,
                )
                for row in triple
            ]
        ),
        columns=6,
        captions=[
            f"{int(label)}:{kind}"
            for label in forward_labels
            for kind in ("target", "null", "learned")
        ],
    )
    write_contact_sheet(
        run_dir / "populations/contact_sheets/oracle_target_null_oracle.png",
        np.stack(
            [
                row
                for triple in zip(
                    rendered["oracle_targets"], rendered["null_oracle"], rendered["oracle"], strict=True
                )
                for row in triple
            ]
        ),
        columns=6,
        captions=[
            f"{int(label)}:{kind}" for label in oracle_labels for kind in ("target", "null", "oracle")
        ],
    )
    oracle_anchor_images = np.concatenate(
        [rasterize_population(oracle.anchors[anchor])[1] for anchor in (0, 32, 64, 96, 128)]
    )
    write_contact_sheet(
        run_dir / "populations/contact_sheets/oracle_anchors.png",
        oracle_anchor_images,
        columns=10,
        captions=[f"a{anchor}" for anchor in (0, 32, 64, 96, 128) for _ in range(10)],
    )
    forward_marker = _forward_direct_marker(forward_targets, null_forward, learned_forward)
    _write_json(run_dir / "populations/forward_direct_metrics.json", forward_marker)
    _append_stage(run_dir, "objective_populations", "complete")
    if governor is not None:
        governor.complete("population_assembly_write", synchronize=False)
    return {
        "raw": raw,
        "demixed": demixed,
        "rendered": rendered,
        "identities": identities,
        "forward_marker": forward_marker,
    }


def seal_populations(
    run_dir: Path,
    firewall: EvaluatorFirewall,
    governor: ResourceGovernor | None = None,
) -> dict[str, Any]:
    if governor is not None:
        _admit_major_stage(run_dir, governor, "population_seal_stage")
    _require(
        not any((run_dir / "evaluation").iterdir())
        and not (run_dir / "review/review_key.json").exists(),
        "evaluator or review evidence exists before the population seal",
    )
    roots = [run_dir / name for name in ("training", "populations", "oracle_control")]
    fixed = [
        run_dir / "data_roles.npz",
        run_dir / "data_roles.json",
        run_dir / "config.json",
        run_dir / "source_bindings.json",
        run_dir / "candidate_backend.json",
        run_dir / "candidate_audit/report.json",
    ]
    paths = sorted(
        {path for root in roots for path in root.rglob("*") if path.is_file()} | set(fixed)
    )
    rows = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]
    seal = {
        "schema": VERSION + "-population-seal",
        "sealed_before_evaluator_or_review": 1,
        "artifact_count": len(rows),
        "tree_digest": _tree_digest(rows),
        "artifacts": rows,
        "sealed_at": _utc_now(),
    }
    path = run_dir / "POPULATIONS_SEALED.json"
    _write_json(path, seal)
    seal_sha256 = _file_sha256(path)
    firewall.mark_populations_sealed(seal_sha256)
    _append_stage(run_dir, "population_seal", "complete", seal_sha256=seal_sha256)
    if governor is not None:
        governor.complete("population_seal_write", synchronize=False)
    return {**seal, "seal_sha256": seal_sha256}


def _validate_population_seal(run_dir: Path) -> tuple[dict[str, Any], str]:
    path = run_dir / "POPULATIONS_SEALED.json"
    _require(path.is_file(), "population seal is missing")
    seal = _read_json(path)
    rows = seal.get("artifacts")
    _require(isinstance(rows, list) and len(rows) == int(seal.get("artifact_count", -1)), "population seal inventory changed")
    for row in rows:
        artifact = run_dir / str(row["path"])
        _require(
            artifact.is_file()
            and artifact.stat().st_size == int(row["size"])
            and _file_sha256(artifact) == row["sha256"],
            f"sealed population artifact changed: {row['path']}",
        )
    _require(seal.get("tree_digest") == _tree_digest(rows), "population seal digest changed")
    return seal, _file_sha256(path)


def _load_accepted_evaluator(
    run_dir: Path,
    firewall: EvaluatorFirewall,
    population_seal_sha256: str,
) -> SmallMnistCNN:
    _, observed_seal_sha256 = _validate_population_seal(run_dir)
    _require(
        firewall.state == EvaluatorFirewall.POPULATIONS_SEALED
        and firewall.seal_sha256 == str(population_seal_sha256)
        and observed_seal_sha256 == str(population_seal_sha256),
        "accepted evaluator may load only after the current population seal",
    )
    firewall.open(population_seal_sha256)
    bindings = _read_json(run_dir / "source_bindings.json")
    checkpoint = Path(bindings["evaluator_files"]["checkpoint"])
    _require(
        checkpoint.is_file()
        and _file_sha256(checkpoint) == ACCEPTED_DDPM_EVALUATOR_CHECKPOINT_SHA256,
        "accepted evaluator checkpoint changed",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict) and "state_dict" in payload, "accepted evaluator payload changed")
    evaluator = SmallMnistCNN()
    evaluator.load_state_dict(payload["state_dict"])
    return evaluator.eval()


def _evaluation_record(
    evaluator: SmallMnistCNN,
    images: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    result = evaluate_generated_labels(
        evaluator, images, labels, sample_ids, batch_size=256, device="cpu"
    )
    logits = np.asarray(result["logits"], dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    requested_log_probabilities = log_probabilities[np.arange(len(labels)), labels]
    arrays = {
        "logits": logits,
        "probabilities": probabilities,
        "predictions": np.asarray(result["predictions"], dtype=np.int64),
        "requested_labels": labels,
        "sample_ids": sample_ids,
        "requested_log_probabilities": requested_log_probabilities,
    }
    metrics = {
        "loss": float(result["loss"]),
        "requested_label_accuracy": float(result["requested_label_accuracy"]),
        "per_class": result["per_class"],
    }
    return arrays, metrics


def _write_blind_review(
    run_dir: Path,
    learned_images: np.ndarray,
    null_images: np.ndarray,
    labels: np.ndarray,
    path_ids: np.ndarray,
    sample_ids: np.ndarray,
    seal_sha256: str,
) -> dict[str, Any]:
    images = np.concatenate((learned_images, null_images))
    controllers = np.asarray(["learned"] * 20 + ["null"] * 20, dtype=np.str_)
    requested = np.concatenate((labels, labels))
    paired_paths = np.concatenate((path_ids, path_ids))
    sources = np.asarray(
        [f"learned:{value}" for value in sample_ids] + [f"null:{value}" for value in sample_ids],
        dtype=np.str_,
    )
    order = np.random.default_rng(SEEDS["review_shuffle"]).permutation(40)
    blind_ids = np.asarray([f"blind-{index:03d}" for index in range(40)], dtype=np.str_)
    ordered_images = images[order]
    review_dir = run_dir / "review"
    _save_individual_pngs(review_dir / "samples", ordered_images, blind_ids)
    write_contact_sheet(
        review_dir / "blind_contact_sheet.png",
        ordered_images,
        columns=8,
        captions=blind_ids.tolist(),
    )
    template_rows = [
        {
            "review_order": index,
            "sample_id": str(blind_ids[index]),
            "assigned_label": "",
            "notes": "",
        }
        for index in range(40)
    ]
    _write_csv(
        review_dir / "review_template.csv",
        template_rows,
        ("review_order", "sample_id", "assigned_label", "notes"),
    )
    entries = [
        {
            "review_order": index,
            "sample_id": str(blind_ids[index]),
            "source_sample_id": str(sources[order[index]]),
            "controller": str(controllers[order[index]]),
            "requested_label": int(requested[order[index]]),
            "path_id": int(paired_paths[order[index]]),
        }
        for index in range(40)
    ]
    key = {
        "schema": VERSION + "-blind-review-key",
        "population_seal_sha256": seal_sha256,
        "seed": SEEDS["review_shuffle"],
        "entries": entries,
    }
    _write_json(review_dir / "review_key.json", key)
    ready = {
        "schema": VERSION + "-review-ready",
        "population_seal_sha256": seal_sha256,
        "row_count": 40,
        "learned_count": 20,
        "null_count": 20,
        "template_sha256": _file_sha256(review_dir / "review_template.csv"),
        "key_sha256": _file_sha256(review_dir / "review_key.json"),
        "contact_sheet_sha256": _file_sha256(review_dir / "blind_contact_sheet.png"),
    }
    _write_json(review_dir / "READY.json", ready)
    return ready


def evaluate_sealed_populations(
    run_dir: Path,
    populations: Mapping[str, Any],
    firewall: EvaluatorFirewall,
    governor: ResourceGovernor,
) -> dict[str, Any]:
    _, seal_sha256 = _validate_population_seal(run_dir)
    _admit_major_stage(run_dir, governor, "sealed_evaluation_stage")
    if firewall.state == EvaluatorFirewall.BOUND_ONLY:
        firewall.mark_populations_sealed(seal_sha256)
    governor.admit(
        "evaluator_batch",
        predicted_seconds=30.0,
        predicted_bytes=4 * 1024 * 1024,
    )
    evaluator = _load_accepted_evaluator(run_dir, firewall, seal_sha256)
    _write_json(
        run_dir / "evaluation/OPEN_EVENT.json",
        {
            "schema": VERSION + "-evaluator-open",
            "population_seal_sha256": seal_sha256,
            "opened_after_population_seal": 1,
            "terminal_test_rows_opened": 0,
            "opened_at": _utc_now(),
        },
    )
    rendered = populations["rendered"]
    identities = populations["identities"]
    evaluated: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for name, label_key, id_key in (
        ("null_prior", "prior_requested_labels", "prior_sample_ids"),
        ("learned_prior", "prior_requested_labels", "prior_sample_ids"),
        ("null_forward_terminal", "forward_requested_labels", "forward_sample_ids"),
        ("learned_forward_terminal", "forward_requested_labels", "forward_sample_ids"),
    ):
        evaluated[name] = _evaluation_record(
            evaluator,
            rendered[name],
            np.asarray(identities[label_key], dtype=np.int64),
            np.asarray(identities[id_key], dtype=np.str_),
        )
    output_arrays = {
        f"{name}_{key}": value
        for name, (arrays, _) in evaluated.items()
        for key, value in arrays.items()
    }
    _write_npz(run_dir / "evaluation/outputs.npz", **output_arrays)
    prior_delta = (
        evaluated["learned_prior"][0]["requested_log_probabilities"]
        - evaluated["null_prior"][0]["requested_log_probabilities"]
    )
    null_accuracy = float(evaluated["null_prior"][1]["requested_label_accuracy"])
    learned_accuracy = float(evaluated["learned_prior"][1]["requested_label_accuracy"])
    evaluator_marker = {
        "gate_type": "diagnostic threshold",
        "null_prior_requested_label_accuracy": null_accuracy,
        "learned_prior_requested_label_accuracy": learned_accuracy,
        "paired_requested_log_probability_win_count": int(np.sum(prior_delta > 0.0)),
        "mean_paired_requested_log_probability_improvement": float(np.mean(prior_delta)),
    }
    evaluator_marker["passed"] = int(
        learned_accuracy >= 0.20
        and learned_accuracy > null_accuracy
        and evaluator_marker["paired_requested_log_probability_win_count"] >= 12
        and evaluator_marker["mean_paired_requested_log_probability_improvement"] > 0.0
    )
    metrics = {
        "schema": VERSION + "-evaluation-metrics",
        "terminal_test_rows_used": 0,
        "populations": {name: value[1] for name, value in evaluated.items()},
        "prior_paired_requested_log_probability_effect": prior_delta,
        "evaluator_marker": evaluator_marker,
        "forward_direct_marker": populations["forward_marker"],
    }
    _write_json(run_dir / "evaluation/metrics.json", metrics)
    bindings = _read_json(run_dir / "source_bindings.json")
    _write_json(
        run_dir / "evaluation/evaluator_binding.json",
        {
            "checkpoint_sha256": bindings["evaluator_hashes"]["checkpoint"],
            "selection_sha256": bindings["evaluator_hashes"]["selection"],
            "population_seal_sha256": seal_sha256,
            "device": "cpu",
        },
    )
    _write_blind_review(
        run_dir,
        rendered["learned_prior"],
        rendered["null_prior"],
        np.asarray(identities["prior_requested_labels"], dtype=np.int64),
        np.asarray(identities["prior_path_ids"], dtype=np.int64),
        np.asarray(identities["prior_sample_ids"], dtype=np.str_),
        seal_sha256,
    )
    governor.complete("evaluator_batch", model_evaluations=80)
    outcome = {
        "schema": VERSION + "-outcome-pre-review",
        "route": "review_pending",
        "research_mode": "exploratory",
        "forward_direct_marker": populations["forward_marker"],
        "evaluator_marker": evaluator_marker,
        "human_marker": None,
        "full_scale_auto_launched": 0,
    }
    _write_json(run_dir / "outcome_pre_review.json", outcome)
    _status(run_dir, "review_pending")
    _append_stage(run_dir, "post_seal_evaluation_and_review_bundle", "complete")
    return metrics


def compute_human_review_metrics(
    answers: Path, key: Mapping[str, Any]
) -> dict[str, Any]:
    entries = key.get("entries")
    _require(isinstance(entries, list) and len(entries) == 40, "review key must contain 40 rows")
    by_id = {str(row["sample_id"]): row for row in entries}
    _require(len(by_id) == 40, "review key contains duplicate IDs")
    with Path(answers).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames == ["review_order", "sample_id", "assigned_label", "notes"],
            "review columns changed",
        )
        rows = list(reader)
    _require(len(rows) == 40, "review must answer all 40 rows")
    seen: set[str] = set()
    controllers: dict[str, list[tuple[int, str]]] = {"learned": [], "null": []}
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"]).strip()
        assignment = str(row["assigned_label"]).strip().lower()
        _require(sample_id in by_id and sample_id not in seen, "review ID is unknown or duplicated")
        entry = by_id[sample_id]
        _require(
            int(row["review_order"]) == int(entry["review_order"])
            and assignment in REVIEW_ASSIGNMENTS,
            "review order or answer is invalid",
        )
        controller = str(entry["controller"])
        _require(controller in controllers, "review controller role changed")
        requested = int(entry["requested_label"])
        controllers[controller].append((requested, assignment))
        normalized_rows.append(
            {
                "review_order": int(entry["review_order"]),
                "sample_id": sample_id,
                "source_sample_id": str(entry["source_sample_id"]),
                "controller": controller,
                "requested_label": requested,
                "path_id": int(entry["path_id"]),
                "assigned_label": assignment,
                "notes": str(row["notes"]),
            }
        )
        seen.add(sample_id)
    _require(seen == set(by_id) and all(len(value) == 20 for value in controllers.values()), "review role counts changed")

    def summarize(values: Sequence[tuple[int, str]]) -> dict[str, Any]:
        recognizable = [assignment.isdigit() for _, assignment in values]
        requested = [assignment == str(label) for label, assignment in values]
        return {
            "count": len(values),
            "recognizable_count": int(sum(recognizable)),
            "requested_label_count": int(sum(requested)),
            "recognizability": float(np.mean(recognizable)),
            "requested_label_rate": float(np.mean(requested)),
        }

    learned = summarize(controllers["learned"])
    null = summarize(controllers["null"])
    paired: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in normalized_rows:
        paired.setdefault(int(row["path_id"]), {})[str(row["controller"])] = row
    _require(
        len(paired) == 20
        and all(set(rows) == {"learned", "null"} for rows in paired.values()),
        "review rows do not form 20 learned/null path pairs",
    )
    paired_paths: list[dict[str, Any]] = []
    for path_id, pair in sorted(paired.items()):
        learned_row = pair["learned"]
        null_row = pair["null"]
        _require(
            int(learned_row["requested_label"]) == int(null_row["requested_label"]),
            "paired review labels differ",
        )
        learned_recognizable = int(str(learned_row["assigned_label"]).isdigit())
        null_recognizable = int(str(null_row["assigned_label"]).isdigit())
        learned_requested = int(
            str(learned_row["assigned_label"]) == str(learned_row["requested_label"])
        )
        null_requested = int(
            str(null_row["assigned_label"]) == str(null_row["requested_label"])
        )
        paired_paths.append(
            {
                "path_id": path_id,
                "requested_label": int(learned_row["requested_label"]),
                "learned_sample_id": str(learned_row["sample_id"]),
                "null_sample_id": str(null_row["sample_id"]),
                "learned_recognizable": learned_recognizable,
                "null_recognizable": null_recognizable,
                "learned_requested_label": learned_requested,
                "null_requested_label": null_requested,
                "recognizable_difference": learned_recognizable - null_recognizable,
                "requested_label_difference": learned_requested - null_requested,
            }
        )
    recognizable_differences = [int(row["recognizable_difference"]) for row in paired_paths]
    requested_differences = [int(row["requested_label_difference"]) for row in paired_paths]
    return {
        "schema": VERSION + "-human-review-metrics",
        "rows": sorted(normalized_rows, key=lambda row: int(row["review_order"])),
        "learned": learned,
        "null": null,
        "learned_counts_exceed_null": int(
            learned["recognizable_count"] > null["recognizable_count"]
            and learned["requested_label_count"] > null["requested_label_count"]
        ),
        "paired_paths": paired_paths,
        "aggregate_paired_recognizable_difference": int(sum(recognizable_differences)),
        "aggregate_paired_requested_label_difference": int(sum(requested_differences)),
        "paired_recognizable_win_loss_tie": {
            "wins": int(sum(value > 0 for value in recognizable_differences)),
            "losses": int(sum(value < 0 for value in recognizable_differences)),
            "ties": int(sum(value == 0 for value in recognizable_differences)),
        },
        "paired_requested_label_win_loss_tie": {
            "wins": int(sum(value > 0 for value in requested_differences)),
            "losses": int(sum(value < 0 for value in requested_differences)),
            "ties": int(sum(value == 0 for value in requested_differences)),
        },
    }


def route_outcome(
    forward_marker: Mapping[str, Any],
    human_marker: Mapping[str, Any],
    evaluator_marker: Mapping[str, Any],
) -> str:
    forward = int(forward_marker.get("passed", 0)) == 1
    human = int(human_marker.get("passed", 0)) == 1
    evaluator = int(evaluator_marker.get("passed", 0)) == 1
    if not forward and not human and evaluator:
        return "human_negative_evaluator_positive_audit"
    if forward and human:
        return (
            "approximate_candidate_feasibility_reference_audit_next"
            if evaluator
            else "human_direct_positive_evaluator_disagreement"
        )
    if forward and not human:
        return (
            "evaluator_render_mismatch_task_negative"
            if evaluator
            else "prior_terminal_mismatch_or_on_policy_shift"
        )
    if human and not forward:
        return "suspicious_prior_forward_disagreement"
    return "v0_negative_pivot_experiment10"


def _outcome_action(route: str) -> str:
    return {
        "approximate_candidate_feasibility_reference_audit_next": (
            "freeze this pilot and seek separate approval for one fixed reference audit"
        ),
        "human_direct_positive_evaluator_disagreement": (
            "preserve the task-positive evidence and audit evaluator symmetry"
        ),
        "evaluator_render_mismatch_task_negative": (
            "treat the prior task result as negative and audit the evaluator/renderer mismatch"
        ),
        "prior_terminal_mismatch_or_on_policy_shift": (
            "localize terminal/prior mismatch or on-policy shift before any v0 scale-up"
        ),
        "suspicious_prior_forward_disagreement": (
            "audit pairing, rendering, and review before any scale-up"
        ),
        "human_negative_evaluator_positive_audit": (
            "treat the task result as negative; audit evaluator/render/proxy mismatch only; "
            "do not select samples"
        ),
        "v0_negative_pivot_experiment10": (
            "stop this v0 recipe and run the materially different Experiment-10 pivot or stop"
        ),
    }[route]


def _human_marker_from_metrics(
    run_dir: Path, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    populations = _npz_arrays(run_dir / "populations/uint8_populations.npz")
    learned = np.asarray(populations["learned_prior"], dtype=np.uint8)
    labels = np.asarray(populations["prior_requested_labels"], dtype=np.int64)
    learned_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in learned]
    all_unique = len(set(learned_hashes)) == 20
    within_class_unique = all(
        len({learned_hashes[index] for index in np.flatnonzero(labels == digit)}) == 2
        for digit in range(10)
    )
    learned_metrics = metrics["learned"]
    null_metrics = metrics["null"]
    human_passed = bool(
        int(learned_metrics["recognizable_count"]) >= 15
        and int(learned_metrics["requested_label_count"]) >= 12
        and int(learned_metrics["recognizable_count"])
        > int(null_metrics["recognizable_count"])
        and int(learned_metrics["requested_label_count"])
        > int(null_metrics["requested_label_count"])
        and all_unique
        and within_class_unique
    )
    return {
        "gate_type": "diagnostic threshold",
        "learned_recognizable_count": int(learned_metrics["recognizable_count"]),
        "learned_requested_label_count": int(learned_metrics["requested_label_count"]),
        "null_recognizable_count": int(null_metrics["recognizable_count"]),
        "null_requested_label_count": int(null_metrics["requested_label_count"]),
        "all_learned_endpoint_hashes_distinct": int(all_unique),
        "within_requested_class_endpoints_distinct": int(within_class_unique),
        "passed": int(human_passed),
    }


def _production_report(run_dir: Path, outcome: Mapping[str, Any]) -> str:
    route = str(outcome["route"])
    selected_action = _outcome_action(route)
    config = _read_json(run_dir / "config.json")
    bindings = _read_json(run_dir / "source_bindings.json")
    selection = _read_json(run_dir / "training/selection.json")
    audit = _read_json(run_dir / "candidate_audit/report.json")
    oracle = _read_json(run_dir / "oracle_control/metrics.json")
    evaluation = _read_json(run_dir / "evaluation/metrics.json")
    review = _read_json(run_dir / "review/metrics.json")
    ledger = _read_json(run_dir / "resource_ledger.json")
    smoke = _read_json(run_dir / "resource_smoke/timings.json")
    forward = evaluation["forward_direct_marker"]
    human = outcome["human_marker"]
    evaluator = evaluation["evaluator_marker"]
    verify_command = _canonical_command_text(
        [
            str(Path(sys.executable).resolve()),
            "-B",
            "-m",
            "mnist.diag_eulerian_jacobi_ddpm_candidate_pilot",
            "verify",
            "--run-dir",
            str(run_dir),
        ]
    ).strip()
    run_command = (run_dir / "command.txt").read_text(encoding="utf-8").strip()
    review_command = (run_dir / "review/record_command.txt").read_text(
        encoding="utf-8"
    ).strip()
    path_roles = ", ".join(
        f"{name}=[0x{bounds[0]:x},0x{bounds[1]:x})"
        for name, bounds in PATH_RANGES.items()
    )
    return f"""# K=128 approximate-candidate Eulerian Jacobi objective pilot

## Research mode and decision

Research mode: **exploratory**. Decision: {config['decision']}.
Nearest objective-bearing artifact: {config['objective_artifact']}.

## Source revision and fixed scientific configuration

Git revision: `{bindings['git']['revision']}`. Protected-source inventory: 27 files,
semantic digest `{_semantic_sha256(bindings['historical_source_inventory'])}`.
Candidate backend: `{candidate.CANDIDATE_BACKEND_NAME}`; target semantics:
`{CANDIDATE_TARGET_SEMANTICS}`; adaptive mode minimum / observed audit maximum:
128 / `{audit['candidate_mode_telemetry']['maximum_adaptive_modes']}`; bisections: 56;
outer steps: 128;
model parameters: 34,974; training seed: `{SEEDS['training_model']}`.

Evidence roles and path IDs: {path_roles}. Training/development uses 250 training
and 100 exploratory-validation whole paths. Oracle control uses ten held-out
validation-role paths. Whole-file bytes were read only for SHA-256 authority;
content parsing stopped after row 59,999 and never requested a terminal row.

Selected checkpoint `training/selected_checkpoint.pt`: update `{selection['selected_update']}`, validation normalized
MSE `{selection['selected_validation_normalized_mse']:.9g}`, SHA-256
`{selection['checkpoint_sha256']}` under the frozen earliest-finite-minimum rule.

## Result and metric hierarchy

This run tested one frozen 34,974-parameter all-class learner against paired null
controls on 20 forward-terminal and 20 Dirichlet-prior starts, after a fixed
512-lane numerical check and a complete ten-class oracle control.  Its decision
route is `{route}`.

Primary objective metrics: forward learned-over-null raw-mass L1 wins
`{forward['learned_l1_win_count']}/20`, aggregate relative improvement
`{forward['aggregate_relative_l1_improvement']:.6g}`; human learned recognizable
`{human['learned_recognizable_count']}/20`, requested-label
`{human['learned_requested_label_count']}/20`; paired human recognizable/requested
differences `{review['aggregate_paired_recognizable_difference']}` /
`{review['aggregate_paired_requested_label_difference']}`.

Mechanism diagnostics: learned controller RMS `{forward['learned_controller_rms']:.6g}`;
evaluator learned/null prior accuracy
`{evaluator['learned_prior_requested_label_accuracy']:.6g}` /
`{evaluator['null_prior_requested_label_accuracy']:.6g}`; paired requested-class
log-probability wins `{evaluator['paired_requested_log_probability_win_count']}/20`
and mean effect `{evaluator['mean_paired_requested_log_probability_improvement']:.6g}`.

Health metrics and typed gates: Gate B (**execution/integrity**;
`candidate_audit/report.json`) pass
`{audit['passed']}`, max candidate/certified later error
`{audit['candidate_vs_certified']['maximum_later_fraction_error']:.6g}`, max target
error `{audit['candidate_vs_certified']['maximum_target_error']:.6g}`; Gate C
(**execution/integrity**; `oracle_control/metrics.json`) pass `{oracle['passed']}`, oracle-improved paths
`{oracle['oracle_improved_path_count']}/10`; forward, human, and evaluator markers
from `evaluation/metrics.json` are **diagnostic-threshold** metrics, with passes `{forward['passed']}` / `{human['passed']}` /
`{evaluator['passed']}`.

Resource accounting and effect: active seconds `{ledger['active_seconds']:.3f}` of
`{ledger['budget']['max_active_seconds']:.3f}`, peak storage
`{ledger['peak_storage_bytes']}` bytes of `{ledger['budget']['max_storage_bytes']}`,
peak CUDA allocation fraction `{ledger['peak_cuda_fraction']:.6g}` of
`{ledger['budget']['max_cuda_fraction']}`. Initial conservative remaining-work
projection: `{smoke['projection']['projected_remaining_seconds']:.3f}` seconds.

## Outcome-to-action table

| Observation | Required action |
|---|---|
| Oracle/control invalid | Repair composition before learner attribution |
| Forward positive, prior human negative | Audit prior mismatch/on-policy shift |
| Human negative, evaluator positive | Treat task result as negative; audit evaluator/render/proxy mismatch only; do not select samples |
| Human/direct positive, evaluator negative | Audit evaluator/render disagreement |
| Both objective markers positive | Seek separate approval for one fixed reference audit |
| Both objective markers and evaluator negative | Run the materially different Experiment-10 pivot |

Selected action route: `{route}`. No full-scale run was automatically launched.
Next action: {selected_action}. Any reference audit requires a separate approval.

## Exact commands

Run command:

Canonical authority: `command.txt`.

```text
{run_command}
```

Review command:

```text
{review_command}
```

Read-only verification command:

```text
{verify_command}
```

## Claim boundary and deliberate omissions

This result establishes only the observed behavior of one exploratory K=128
finite chain using **{CANDIDATE_TARGET_SEMANTICS}** values, the frozen seed,
populations, renderer, and review.  It does not establish candidate/reference
equivalence, continuum consistency, an exact Doob transform, or a universal
success/failure of Eulerian generation.

Deliberately omitted: terminal MNIST test rows, a reference/certified full reverse
population, a second training seed, confirmatory inference, and the historical
full-scale population. Therefore this run cannot support terminal-test,
candidate/reference-equivalence, across-seed, confirmatory, or scale-up claims.
"""


def _experiment_note(run_dir: Path, outcome: Mapping[str, Any]) -> str:
    config = _read_json(run_dir / "config.json")
    authority = config["execution_authority"]
    selection = _read_json(run_dir / "training/selection.json")
    ledger = _read_json(run_dir / "resource_ledger.json")
    return f"""# Experiment note

## Mode, objective, and evidence

Mode: exploratory. Proxy-only patches before this objective-bearing run: 2.
Objective: {config['decision']}. Evidence roles and path IDs are frozen in
`config.json`; the selected checkpoint is update `{selection['selected_update']}`
at validation normalized MSE `{selection['selected_validation_normalized_mse']:.9g}`.

## Resources and typed metrics

Resource approval: `{authority['approval_id']}`; active cap
`{authority['maximum_active_seconds']}` seconds; storage cap
`{authority['maximum_storage_mib']}` MiB; CUDA allocation fraction
`{authority['maximum_cuda_fraction']}`. Recorded active seconds:
`{ledger['active_seconds']:.3f}`. The scientific decision purchased was
whether the assembled candidate-K128 system produces direct learned-over-null
task effects; another local proxy could not answer it.

Gate B and Gate C are execution/integrity gates. Forward, human, and evaluator
markers are diagnostic thresholds. Outcome route: `{outcome['route']}`.
Required next action: {_outcome_action(str(outcome['route']))}.
Candidate targets and transitions remain explicitly approximate. Failed images
were retained. No automatic full-scale run or Experiment-10 run occurred.

## Claim boundary and deliberate omissions

No terminal-test content rows, certified/reference full reverse population, second
model seed, confirmatory analysis, or full-scale population were parsed or run.
These omissions prevent terminal-test, certified-full-reverse, across-seed,
confirmatory, and scale-up claims.

## Exact commands

Run: `{(run_dir / 'command.txt').read_text(encoding='utf-8').strip()}`

Review: `{(run_dir / 'review/record_command.txt').read_text(encoding='utf-8').strip()}`

Verify from the repository root:

```powershell
.\\.venv\\Scripts\\python.exe -B -m mnist.diag_eulerian_jacobi_ddpm_candidate_pilot verify --run-dir "{run_dir}"
```
"""


def _rehydrate_resource_governor(
    run_dir: Path, *, started_at: float | None = None
) -> ResourceGovernor:
    config = _read_json(Path(run_dir) / "config.json")
    authority = config["execution_authority"]
    budget_row = _read_json(Path(run_dir) / "resource_ledger.json")["budget"]
    budget = ResourceBudget(**budget_row)
    return ResourceGovernor(
        Path(run_dir),
        torch.device(str(authority["device"])),
        budget,
        started_at=started_at,
    )


def _finish_priced_terminalization(
    run_dir: Path, governor: ResourceGovernor, kind: str
) -> dict[str, Any]:
    _seal_manifest(run_dir)
    verify_run(run_dir)
    governor.complete(kind, synchronize=False)
    if (run_dir / "outcome.json").is_file():
        outcome = _read_json(run_dir / "outcome.json")
        _write_text(run_dir / "REPORT.md", _production_report(run_dir, outcome))
        _write_text(run_dir / "experiment_note.md", _experiment_note(run_dir, outcome))
    _seal_manifest(run_dir)
    return verify_run(run_dir)


def record_review(
    run_dir: Path,
    answers: Path,
    reviewer: str,
    confirm_manual_review: bool,
    *,
    resource_governor: ResourceGovernor | None = None,
) -> dict[str, Any]:
    terminalization_started = time.monotonic()
    run_dir = Path(run_dir).resolve()
    answers = Path(answers).resolve()
    _require(run_dir.is_dir() and answers.is_file(), "run or answer file is missing")
    _require(confirm_manual_review is True and bool(str(reviewer).strip()), "manual review confirmation and reviewer are required")
    _require(_read_json(run_dir / "status.json").get("route") == "review_pending", "run is not awaiting review")
    _, seal_sha256 = _validate_population_seal(run_dir)
    ready = _read_json(run_dir / "review/READY.json")
    _require(ready.get("population_seal_sha256") == seal_sha256, "review bundle is not bound to the population seal")
    key = _read_json(run_dir / "review/review_key.json")
    metrics = compute_human_review_metrics(answers, key)
    benchmark_metrics = score_human_review(
        answers,
        run_dir / "review/review_key.json",
        reviewer=reviewer,
        confirm_manual_review=True,
    )
    metrics["reviewer"] = str(reviewer).strip()
    metrics["answers_source_path"] = str(answers)
    metrics["recorded_at"] = benchmark_metrics["recorded_at"]
    metrics["per_class"] = benchmark_metrics["per_class"]
    human_marker = _human_marker_from_metrics(run_dir, metrics)
    metrics["human_marker"] = human_marker
    evaluation = _read_json(run_dir / "evaluation/metrics.json")
    evaluator_marker = evaluation["evaluator_marker"]
    forward_marker = evaluation["forward_direct_marker"]
    predictions = {}
    with np.load(run_dir / "evaluation/outputs.npz", allow_pickle=False) as archive:
        source_to_prediction: dict[str, int] = {}
        for role in ("learned_prior", "null_prior"):
            ids = np.asarray(archive[f"{role}_sample_ids"], dtype=np.str_)
            values = np.asarray(archive[f"{role}_predictions"], dtype=np.int64)
            prefix = "learned" if role.startswith("learned") else "null"
            source_to_prediction.update(
                {f"{prefix}:{sample_id}": int(value) for sample_id, value in zip(ids, values, strict=True)}
            )
        for row in metrics["rows"]:
            predictions[row["sample_id"]] = source_to_prediction[row["source_sample_id"]]
    metrics["human_machine_disagreement_count"] = int(
        sum(
            str(row["assigned_label"]).isdigit()
            and int(row["assigned_label"]) != predictions[row["sample_id"]]
            for row in metrics["rows"]
        )
    )
    governor = (
        resource_governor
        if resource_governor is not None
        else _rehydrate_resource_governor(
            run_dir, started_at=terminalization_started
        )
    )
    _admit_major_stage(run_dir, governor, "record_review_terminalization")
    review_argv = [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "mnist.diag_eulerian_jacobi_ddpm_candidate_pilot",
        "record-review",
        "--run-dir",
        str(run_dir),
        "--answers",
        str(answers),
        "--reviewer",
        str(reviewer).strip(),
        "--confirm-manual-review",
    ]
    _write_text(
        run_dir / "review/record_command.txt",
        _canonical_command_text(review_argv),
    )
    _atomic_replace(
        run_dir / "review/submitted_answers.csv",
        lambda target: shutil.copyfile(answers, target),
    )
    _write_json(run_dir / "review/metrics.json", metrics)
    outcome = {
        "schema": VERSION + "-outcome",
        "status": "complete",
        "route": route_outcome(forward_marker, human_marker, evaluator_marker),
        "research_mode": "exploratory",
        "forward_direct_marker": forward_marker,
        "human_marker": human_marker,
        "evaluator_marker": evaluator_marker,
        "candidate_target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "full_scale_auto_launched": 0,
    }
    _write_json(run_dir / "outcome.json", outcome)
    _write_text(run_dir / "REPORT.md", _production_report(run_dir, outcome))
    _write_text(run_dir / "experiment_note.md", _experiment_note(run_dir, outcome))
    _status(run_dir, "complete")
    _append_stage(run_dir, "record_human_review", "complete")
    _finish_priced_terminalization(
        run_dir, governor, "record_review_terminalization"
    )
    return outcome


def _npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _verify_source_bindings(run_dir: Path) -> None:
    bindings = _read_json(run_dir / "source_bindings.json")
    config = _read_json(run_dir / "config.json")
    _require(
        bindings.get("config_sha256") == _semantic_sha256(config),
        "source binding/config hash changed",
    )
    repository_root = Path(bindings["repository_root"]).resolve()
    _require(
        repository_root == Path(__file__).resolve().parents[1],
        "bound repository root changed",
    )
    _require(bindings["historical_source_inventory"] == PROTECTED_SOURCE_HASHES, "historical source inventory changed")
    for relative, expected in PROTECTED_SOURCE_HASHES.items():
        _require(_file_sha256(repository_root / relative) == expected, f"protected source changed: {relative}")
    deferred = repository_root / bindings["protected_candidate_cuda_source"]["path"]
    _require(
        _file_sha256(deferred) == bindings["protected_candidate_cuda_source"]["sha256"]
        == PROTECTED_DEFERRED_CUDA_SOURCE_SHA256,
        "candidate CUDA source changed",
    )
    for relative, expected in bindings["additive_source_hashes"].items():
        _require(_file_sha256(repository_root / relative) == expected, f"additive source changed: {relative}")
    _require(
        _file_sha256(Path(bindings["arff"])) == bindings["arff_sha256"] == MNIST_ARFF_SHA256,
        "ARFF authority changed",
    )
    for name, expected in bindings["evaluator_hashes"].items():
        _require(
            _file_sha256(Path(bindings["evaluator_files"][name])) == expected,
            f"evaluator binding changed: {name}",
        )


def _verify_config_and_resources(run_dir: Path, route: str) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    scientific = {name: value for name, value in config.items() if name != "execution_authority"}
    _require(
        _semantic_sha256(scientific) == _semantic_sha256(FROZEN_CONFIG),
        "scientific configuration changed",
    )
    authority = config.get("execution_authority")
    _require(isinstance(authority, dict), "execution authority is missing")
    approval = _validate_approval(str(authority.get("approval_id", "")))
    _validate_resource_values(
        float(authority["maximum_active_seconds"]),
        float(authority["maximum_storage_mib"]),
        float(authority["maximum_cuda_fraction"]),
    )
    _require(
        authority.get("device") == _read_json(run_dir / "environment.json").get("device")
        and authority.get("exact_cli_subcommand") == "run"
        and authority.get("whole_run_restart_only") == 1
        and authority.get("automatic_full_scale_launches") == 0
        and float(authority.get("terminal_reserve_seconds", -1.0)) == 900.0,
        "execution authority changed",
    )
    bindings = _read_json(run_dir / "source_bindings.json")
    expected_argv = _canonical_run_argv(
        run_dir,
        Path(bindings["arff"]),
        Path(bindings["ddpm_run_dir"]),
        device=str(authority["device"]),
        approval_id=approval,
        maximum_active_seconds=float(authority["maximum_active_seconds"]),
        maximum_storage_mib=float(authority["maximum_storage_mib"]),
        maximum_cuda_fraction=float(authority["maximum_cuda_fraction"]),
    )
    command_text = _canonical_command_text(expected_argv)
    _require(
        authority.get("canonical_argv") == expected_argv
        and authority.get("command_sha256")
        == hashlib.sha256(command_text.encode("utf-8")).hexdigest()
        and (run_dir / "command.txt").read_text(encoding="utf-8") == command_text,
        "canonical run command changed",
    )

    ledger = _read_json(run_dir / "resource_ledger.json")
    expected_budget = ResourceBudget(
        max_active_seconds=float(authority["maximum_active_seconds"]),
        max_storage_bytes=int(float(authority["maximum_storage_mib"]) * 1024 * 1024),
        max_cuda_fraction=float(authority["maximum_cuda_fraction"]),
    )
    _require(ledger.get("budget") == asdict(expected_budget), "resource budget authority changed")
    events = ledger.get("events")
    admissions = ledger.get("admissions")
    _require(isinstance(events, list) and isinstance(admissions, list), "resource ledger rows changed")
    event_seconds = [float(row["seconds"]) for row in events]
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in event_seconds)
        and math.isclose(
            float(ledger["active_seconds"]),
            math.fsum(event_seconds),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and int(ledger["peak_storage_bytes"])
        >= max([0, *(int(row["tree_bytes"]) for row in events)]),
        "resource ledger totals changed",
    )
    decision_keys = set(
        resource_admission(
            expected_budget,
            active_seconds=0.0,
        )
    )
    seen_admission_kinds: dict[str, int] = {}
    prior_event_count = 0
    for index, receipt in enumerate(admissions):
        kind = str(receipt["kind"])
        event_count = int(receipt.get("event_count_before_admission", -1))
        ordinal = int(receipt.get("kind_admission_ordinal", -1))
        post_completion = int(receipt.get("post_completion_check", -1))
        terminalization = int(receipt.get("terminalization", -1))
        _require(
            prior_event_count <= event_count <= len(events),
            f"resource admission event boundary changed: {index}",
        )
        prior_event_count = event_count
        prefix = events[:event_count]
        prefix_active = math.fsum(float(row["seconds"]) for row in prefix)
        expected_ordinal = seen_admission_kinds.get(kind, 0) + 1
        seen_admission_kinds[kind] = expected_ordinal
        _require(
            ordinal == expected_ordinal
            and math.isclose(
                float(receipt["active_seconds"]),
                prefix_active,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ),
            f"resource admission sequence changed: {index}",
        )
        declared_seconds = float(
            receipt.get("declared_predicted_next_quantum_seconds", -1.0)
        )
        recent = [
            float(row["seconds"])
            for row in prefix
            if row.get("kind") == kind and row.get("completed") == 1
        ][-3:]
        if post_completion:
            _require(
                post_completion == 1
                and event_count > 0
                and receipt.get("completed_kind") == prefix[-1].get("kind")
                and kind == f"post_complete:{prefix[-1].get('kind')}"
                and receipt.get("recent_completed_same_kind_seconds") == []
                and declared_seconds
                == (
                    float(prefix[-1]["seconds"])
                    if float(prefix[-1]["seconds"])
                    >= float(expected_budget.maximum_quantum_seconds)
                    else 0.0
                ),
                f"post-completion resource receipt changed: {index}",
            )
            recent = []
        else:
            floor = _declared_quantum_floor(run_dir, kind, ordinal)
            _require(
                post_completion == 0
                and receipt.get("recent_completed_same_kind_seconds") == recent
                and math.isfinite(declared_seconds)
                and declared_seconds >= floor,
                f"resource quantum prediction basis changed: {index}",
            )
        _require(
            terminalization in {0, 1}
            and (
                terminalization == 0
                or kind == "failure_terminalization"
                or (
                    post_completion == 1
                    and receipt.get("completed_kind")
                    == "failure_terminalization"
                )
            )
            and (
                kind != "failure_terminalization"
                or (
                    terminalization == 1
                    and int(receipt.get("major_stage", -1)) == 0
                    and float(receipt.get("projected_remaining_seconds", -1.0))
                    == 0.0
                    and float(receipt.get("reserve_remaining_seconds", -1.0))
                    == 0.0
                )
            ),
            f"terminal resource receipt changed: {index}",
        )
        expected_next_seconds = max([declared_seconds, *recent])
        _require(
            float(receipt["predicted_next_quantum_seconds"])
            == expected_next_seconds,
            f"resource quantum prediction changed: {index}",
        )
        replayed = resource_admission(
            expected_budget,
            active_seconds=float(receipt["active_seconds"]),
            projected_remaining_seconds=float(receipt["projected_remaining_seconds"]),
            predicted_next_quantum_seconds=float(receipt["predicted_next_quantum_seconds"]),
            storage_bytes=int(receipt["storage_bytes"]),
            predicted_next_bytes=int(receipt["predicted_next_bytes"]),
            cuda_fraction=float(receipt["cuda_fraction"]),
            major_stage=bool(receipt["major_stage"]),
            populations_sealed=bool(receipt["populations_sealed"]),
            terminalization=bool(terminalization),
        )
        _require(
            all(receipt.get(name) == replayed[name] for name in decision_keys),
            f"resource admission replay changed: {index}",
        )
        if kind in RESOURCE_STAGE_REMAINING:
            projection = _stage_resource_projection(run_dir, kind)
            _require(
                receipt["major_stage"] == 1
                and float(receipt["projected_remaining_seconds"])
                == float(projection["projected_remaining_seconds"])
                and int(receipt["predicted_next_bytes"])
                == int(projection["predicted_next_bytes"])
                and float(receipt["projected_remaining_seconds"]) > 0.0
                and int(receipt["predicted_next_bytes"]) > 0,
                f"major-stage projection changed: {kind}",
            )
    admission_fractions = [float(row["cuda_fraction"]) for row in admissions]
    peak_fraction = float(ledger["peak_cuda_fraction"])
    _require(
        math.isfinite(peak_fraction)
        and peak_fraction >= 0.0
        and peak_fraction >= max([0.0, *admission_fractions]),
        "resource CUDA peak changed",
    )
    if admissions:
        _require(
            ledger.get("last_admission") == admissions[-1],
            "last resource admission changed",
        )
    completed_stages = {
        str(row.get("stage"))
        for row in _read_json(run_dir / "stage_ledger.json").get("events", [])
        if row.get("state") == "complete"
    }
    admission_kinds = [str(row.get("kind")) for row in admissions]
    required_major_receipts = {
        "resource_smoke": "oracle_stage",
        "oracle_control": "forward_record_caches_stage",
        "forward_record_caches": "training_stage",
        "training": "objective_sampling_stage",
        "objective_populations": "population_seal_stage",
        "population_seal": "sealed_evaluation_stage",
        "post_seal_evaluation_and_review_bundle": "post_evaluator_finalization",
        "record_human_review": "record_review_terminalization",
    }
    for stage, kind in required_major_receipts.items():
        if stage in completed_stages:
            _require(kind in admission_kinds, f"major-stage resource receipt is missing: {kind}")

    failure_routes = {
        "integrity_failed",
        "candidate_health_failed",
        "resource_projection_failed",
        "oracle_control_failed",
        "resource_stopped",
    }
    terminal_admissions = [
        row for row in admissions if row.get("kind") == "failure_terminalization"
    ]
    terminal_events = [
        row for row in events if row.get("kind") == "failure_terminalization"
    ]
    if route in failure_routes:
        _require(
            len(terminal_admissions) == len(terminal_events) == 1
            and int(terminal_admissions[0].get("terminalization", 0)) == 1
            and int(terminal_admissions[0].get("predicted_next_bytes", 0))
            == FAILURE_TERMINALIZATION_BYTES
            and float(
                terminal_admissions[0].get(
                    "declared_predicted_next_quantum_seconds", -1.0
                )
            )
            == FAILURE_TERMINALIZATION_SECONDS
            and int(terminal_events[0].get("completed", 0)) == 1,
            "failure terminalization accounting changed",
        )
        manifest = _read_json(run_dir / "artifact_manifest.json")
        _require(
            int(ledger["peak_storage_bytes"])
            >= int(manifest.get("artifact_bytes", -1)),
            "failure terminalization peak storage changed",
        )
    else:
        _require(
            not terminal_admissions and not terminal_events,
            "successful route contains failure terminalization accounting",
        )

    smoke_path = run_dir / "resource_smoke/timings.json"
    if smoke_path.is_file():
        timings = _read_json(smoke_path)
        replayed_projection = _smoke_projection(timings)
        _require(
            _semantic_sha256(timings.get("projection"))
            == _semantic_sha256(replayed_projection)
            and int(timings.get("synthetic_record_count", -1)) == 1_400
            and math.isclose(
                float(timings["storage_bytes_per_record"]),
                (run_dir / "resource_smoke/synthetic_records.npz").stat().st_size / 1_400.0,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            "resource-smoke projection changed",
        )

    if route in {"resource_projection_failed", "resource_stopped"}:
        stop = _read_json(run_dir / "resource_stop.json")
        failed = stop.get("failed_admission")
        _require(
            stop.get("route") == route
            and isinstance(failed, dict)
            and any(failed == row for row in admissions)
            and failed.get("passed") == 0
            and failed.get("reason") in failed.get("reasons", []),
            "resource-stop failed admission changed",
        )
        _require(
            peak_fraction <= expected_budget.max_cuda_fraction
            or "cuda_fraction" in failed["reasons"],
            "resource CUDA peak exceeds the cap without a CUDA-fraction stop",
        )
    elif route in failure_routes:
        _require(
            all(
                int(row["passed"]) == 1
                or int(row.get("terminalization", 0)) == 1
                for row in admissions
            )
            and float(ledger["active_seconds"])
            < expected_budget.max_active_seconds
            and int(ledger["peak_storage_bytes"])
            < expected_budget.max_storage_bytes
            and _directory_bytes(run_dir) < expected_budget.max_storage_bytes
            and peak_fraction <= expected_budget.max_cuda_fraction,
            "failure route resource terminalization changed",
        )
    else:
        _require(
            (not admissions or all(int(row["passed"]) == 1 for row in admissions))
            and float(ledger["active_seconds"]) < expected_budget.max_active_seconds
            and int(ledger["peak_storage_bytes"]) < expected_budget.max_storage_bytes
            and _directory_bytes(run_dir) < expected_budget.max_storage_bytes
            and peak_fraction <= expected_budget.max_cuda_fraction,
            "successful route contains a failed resource admission",
        )
    return ledger


def _verify_data_roles(run_dir: Path) -> None:
    arrays = _npz_arrays(run_dir / "data_roles.npz")
    report = _read_json(run_dir / "data_roles.json")
    expected_shapes = {
        "train_arff_indices": ((250,), np.dtype(np.int64)),
        "train_labels": ((250,), np.dtype(np.int64)),
        "train_mixed_masses": ((250, 784), np.dtype(np.float64)),
        "train_path_ids": ((250,), np.dtype(np.int64)),
        "validation_arff_indices": ((100,), np.dtype(np.int64)),
        "validation_labels": ((100,), np.dtype(np.int64)),
        "validation_mixed_masses": ((100, 784), np.dtype(np.float64)),
        "validation_path_ids": ((100,), np.dtype(np.int64)),
        "audit_train_positions": (AUDIT_TRAIN_POSITIONS.shape, np.dtype(np.int64)),
        "oracle_validation_positions": (
            ORACLE_VALIDATION_POSITIONS.shape,
            np.dtype(np.int64),
        ),
        "forward_validation_positions": (
            FORWARD_VALIDATION_POSITIONS.shape,
            np.dtype(np.int64),
        ),
    }
    _require(
        report.get("schema") == VERSION + "-data-roles"
        and set(arrays) == set(report["arrays"]) == set(expected_shapes),
        "data-role array inventory changed",
    )
    for name, value in arrays.items():
        row = report["arrays"][name]
        _require(
            value.shape == expected_shapes[name][0]
            and value.dtype == expected_shapes[name][1]
            and list(value.shape) == row["shape"]
            and value.dtype.str == row["dtype"]
            and _array_sha256(value) == row["sha256"],
            f"data-role array changed: {name}",
        )
    _require(
        np.array_equal(np.bincount(arrays["train_labels"], minlength=10), np.full(10, 25))
        and np.array_equal(np.bincount(arrays["validation_labels"], minlength=10), np.full(10, 10))
        and np.unique(arrays["train_arff_indices"]).size == 250
        and np.unique(arrays["validation_arff_indices"]).size == 100
        and np.all((arrays["train_arff_indices"] >= 0) & (arrays["train_arff_indices"] < 55_000))
        and np.all(
            (arrays["validation_arff_indices"] >= 55_000)
            & (arrays["validation_arff_indices"] < 60_000)
        )
        and np.array_equal(arrays["train_path_ids"], _role_ids("training"))
        and np.array_equal(arrays["validation_path_ids"], _role_ids("validation"))
        and np.array_equal(arrays["audit_train_positions"], AUDIT_TRAIN_POSITIONS)
        and np.array_equal(
            arrays["oracle_validation_positions"], ORACLE_VALIDATION_POSITIONS
        )
        and np.array_equal(
            arrays["forward_validation_positions"], FORWARD_VALIDATION_POSITIONS
        )
        and all(
            np.isfinite(masses).all()
            and np.all(masses >= 0.35 / 784.0)
            and float(np.max(np.abs(masses.sum(axis=1) - 1.0))) <= 2e-12
            for masses in (
                arrays["train_mixed_masses"],
                arrays["validation_mixed_masses"],
            )
        )
        and report.get("path_roles")
        == {
            "training": _path_range(*PATH_RANGES["training"]),
            "validation": _path_range(*PATH_RANGES["validation"]),
        }
        and report.get("terminal_test_rows_used") == 0
        and report.get("content_rows_read") == 60_000
        and report.get("last_content_row_index") == 59_999
        and report.get("terminal_content_rows_read") == 0
        and isinstance(report.get("last_text_line_number_read"), int)
        and int(report["last_text_line_number_read"]) > 60_000
        and report.get("full_file_read_purpose") == "sha256-only"
        and report.get("full_file_sha256") == MNIST_ARFF_SHA256,
        "data roles are unbalanced or opened terminal rows",
    )


def _verify_candidate_backend(run_dir: Path) -> dict[str, Any]:
    record = _read_json(run_dir / "candidate_backend.json")
    expected = {
        "schema": VERSION + "-candidate-backend",
        "backend": candidate.CANDIDATE_BACKEND_NAME,
        "target_semantics": CANDIDATE_TARGET_SEMANTICS,
        "runtime_type": "CandidateRuntime",
        "candidate_modes": 128,
        "candidate_modes_semantics": "adaptive minimum",
        "candidate_adaptive_maximum_modes": 1024,
        "candidate_bisection_steps": 56,
        "threads_per_block": 128,
        "dispatch": [
            "prepare_alpha1_rb_transition_batch_cuda_candidate",
            "prepare_alpha1_rb_transition_cuda_rng_seed",
            "enqueue_alpha1_rb_transition_batch_cuda_candidate",
        ],
        "authorizer_calls": 0,
        "certified_cuda_calls": 0,
        "arb_fallback_calls": 0,
        "scope": "production_candidate_runtime_excludes_audit_references",
        "prepared_rng_keys_sha256": _semantic_sha256(candidate_rng_keys()),
    }
    for name, value in expected.items():
        _require(record.get(name) == value, f"candidate backend changed: {name}")
    _require(
        bool(re.fullmatch(r"[0-9a-f]{64}", str(record.get("candidate_binary_sha256", "")))),
        "candidate backend binary hash changed",
    )
    return record


def _verify_candidate_audit(run_dir: Path) -> dict[str, Any]:
    bank = _npz_arrays(run_dir / "candidate_audit/bank.npz")
    roles = _npz_arrays(run_dir / "data_roles.npz")
    rebuilt = build_candidate_audit_bank(roles["train_mixed_masses"], device="cpu")
    _require(set(bank) == set(rebuilt), "candidate audit bank inventory changed")
    for name, expected in rebuilt.items():
        _require(
            np.array_equal(bank[name], expected),
            f"candidate audit bank changed: {name}",
        )
    outputs = _npz_arrays(run_dir / "candidate_audit/outputs.npz")
    expected_output_names = {
        "candidate_later",
        "candidate_target",
        "candidate_lower",
        "candidate_upper",
        "candidate_valid_mask",
        "candidate_approximation_mask",
        "fast_later",
        "fast_target",
        "certified_later",
        "certified_target",
        "certified_candidate_later",
        "certified_candidate_target",
        "certified_mask",
        "certified_cuda_mask",
        "certified_fallback_mask",
        "certified_quantile_lower",
        "certified_quantile_upper",
        "certified_target_lower",
        "certified_target_upper",
        "certified_certificate_codes",
        "certified_prefix_bits",
        "candidate_adaptive_modes",
        "rng_v2_initial_prefix_numerators",
        "rng_v2_initial_prefix_bits",
        "rng_v2_uniform_midpoints",
        "transition_ids",
        "earlier_head_fraction",
        "exposure",
    }
    _require(
        set(outputs) == expected_output_names
        and np.array_equal(outputs["transition_ids"], bank["transition_ids"])
        and np.array_equal(outputs["earlier_head_fraction"], bank["head_fraction"])
        and np.array_equal(outputs["exposure"], bank["exposure"]),
        "candidate audit output authority changed",
    )
    expected_rng = _v2_audit_randomness(
        (SEEDS["candidate_audit"], "k128-candidate-audit"),
        bank["transition_ids"],
        bank["active_mask"],
    )
    _require(
        np.array_equal(
            outputs["rng_v2_initial_prefix_numerators"],
            expected_rng["initial_prefix_numerators"],
        )
        and np.array_equal(
            outputs["rng_v2_initial_prefix_bits"],
            expected_rng["initial_prefix_bits"],
        )
        and np.array_equal(
            outputs["rng_v2_uniform_midpoints"],
            expected_rng["uniform_midpoints"],
        )
        and np.array_equal(
            outputs["candidate_adaptive_modes"],
            _candidate_adaptive_mode_counts(bank["exposure"]),
        ),
        "candidate audit v2 RNG witness changed",
    )
    saved = _read_json(run_dir / "candidate_audit/report.json")
    diagnostics = saved.get("diagnostics")
    _require(
        isinstance(diagnostics, dict)
        and set(_CANDIDATE_REQUIRED_DIAGNOSTICS).issubset(diagnostics),
        "candidate audit diagnostics inventory changed",
    )
    replayed = recompute_candidate_audit_metrics(bank, outputs, saved["diagnostics"])
    for key, expected in replayed.items():
        _require(_semantic_sha256(saved.get(key)) == _semantic_sha256(expected), f"candidate audit changed: {key}")
    backend = _verify_candidate_backend(run_dir)
    _require(
        saved.get("rng_alignment", {}).get("rng_arrays_exact") == 1
        and saved.get("rng_alignment", {}).get("rng_contract")
        == AUDIT_RNG_CONTRACT
        and saved.get("rng_alignment", {}).get("pairing_contract")
        == AUDIT_PAIRING_CONTRACT,
        "candidate audit RNG alignment changed",
    )
    certified_diagnostics = saved.get("certified_diagnostics")
    certified_runtime = saved.get("certified_runtime")
    _require(
        isinstance(certified_diagnostics, dict)
        and isinstance(certified_runtime, dict)
        and certified_runtime.get("rng_contract") == AUDIT_RNG_CONTRACT
        and certified_runtime.get("profile")
        == _jsonable(JacobiRBCudaProfile().to_dict())
        and certified_runtime.get("runtime_contract_pass") is True
        and int(certified_diagnostics.get("sample_count", -1)) == 512
        and int(certified_diagnostics.get("active_count", -1))
        == int(np.sum(bank["active_mask"]))
        and int(certified_diagnostics.get("certified_count", -1))
        == int(np.sum(outputs["certified_mask"]))
        and int(certified_diagnostics.get("cuda_authorized_count", -1))
        == int(np.sum(outputs["certified_cuda_mask"]))
        and int(certified_diagnostics.get("fallback_count", -1))
        == int(np.sum(outputs["certified_fallback_mask"])),
        "candidate audit certified reference changed",
    )
    _require(
        diagnostics["candidate_modes"] == backend["candidate_modes"]
        and diagnostics["candidate_bisection_steps"]
        == backend["candidate_bisection_steps"]
        and diagnostics["candidate_binary_sha256"]
        == backend["candidate_binary_sha256"]
        and diagnostics["candidate_batch_type"] == "CandidateRBCudaBatch"
        and backend["runtime_type"] == "CandidateRuntime"
        and diagnostics["authorizer_calls"] == 0
        and diagnostics["fallback_calls"] == 0,
        "candidate audit/backend binding changed",
    )
    return saved


def _verify_oracle_control(run_dir: Path) -> dict[str, Any]:
    authority = _npz_arrays(run_dir / "oracle_control/authority.npz")
    roles = _npz_arrays(run_dir / "data_roles.npz")
    expected_authority = {
        "source_targets": roles["validation_mixed_masses"][ORACLE_VALIDATION_POSITIONS],
        "requested_labels": roles["validation_labels"][ORACLE_VALIDATION_POSITIONS],
        "arff_indices": roles["validation_arff_indices"][ORACLE_VALIDATION_POSITIONS],
        "path_ids": _role_ids("oracle"),
        "sample_ids": np.asarray(
            [f"oracle-{value:05x}" for value in _role_ids("oracle")], dtype=np.str_
        ),
    }
    _require(set(authority) == set(expected_authority), "oracle authority inventory changed")
    for name, expected in expected_authority.items():
        _require(np.array_equal(authority[name], expected), f"oracle authority changed: {name}")
    null = _verify_population_stage(run_dir / "oracle_control/null.npz", 10)
    oracle = _verify_population_stage(run_dir / "oracle_control/oracle.npz", 10)
    for arrays, name in ((null, "null"), (oracle, "oracle")):
        _require(
            np.array_equal(arrays["source_targets"], authority["source_targets"])
            and np.array_equal(arrays["requested_labels"], authority["requested_labels"])
            and np.array_equal(arrays["path_ids"], authority["path_ids"])
            and np.array_equal(arrays["sample_ids"], authority["sample_ids"])
            and np.array_equal(arrays["starts"], arrays["terminal_starts"]),
            f"oracle {name} alignment changed",
        )
        _verify_assembled_from_cohorts(
            run_dir / f"oracle_control/stages/{name}", arrays
        )
    _require(np.array_equal(null["starts"], oracle["starts"]), "oracle paired starts changed")
    forward_paths = sorted((run_dir / "oracle_control/stages/forward_terminal").glob("cohort_*.npz"))
    _require(bool(forward_paths), "oracle forward cohorts are missing")
    forward = [_npz_arrays(path) for path in forward_paths]
    forward_terminal = np.concatenate([row["terminal_starts"] for row in forward])
    for name in ("source_targets", "requested_labels", "arff_indices", "path_ids", "sample_ids"):
        combined = np.concatenate([row[name] for row in forward])
        _require(
            np.array_equal(combined, authority[name]),
            f"oracle forward cohort alignment changed: {name}",
        )
    _require(
        np.array_equal(forward_terminal, null["starts"])
        and all(
            path.with_suffix(".telemetry.json").is_file() for path in forward_paths
        ),
        "oracle terminal starts changed",
    )
    null_telemetry = _read_json(run_dir / "oracle_control/null.telemetry.json")
    oracle_telemetry = _read_json(run_dir / "oracle_control/oracle.telemetry.json")
    replayed = oracle_control_metrics(
        authority["source_targets"],
        null["final_states"],
        oracle["final_states"],
        null_telemetry,
        oracle_telemetry,
    )
    saved = _read_json(run_dir / "oracle_control/metrics.json")
    for key, expected in replayed.items():
        _require(_semantic_sha256(saved.get(key)) == _semantic_sha256(expected), f"oracle metric changed: {key}")
    expected_anchor_l1 = {
        str(anchor): {
            "null": np.sum(
                np.abs(null[f"anchors_{anchor:03d}"] - authority["source_targets"]), axis=1
            ),
            "oracle": np.sum(
                np.abs(oracle[f"anchors_{anchor:03d}"] - authority["source_targets"]), axis=1
            ),
        }
        for anchor in (0, 32, 64, 96, 128)
    }
    _require(
        _semantic_sha256(saved.get("anchor_raw_mass_l1"))
        == _semantic_sha256(expected_anchor_l1),
        "oracle anchor metrics changed",
    )
    for directory, rows in (
        ("targets", authority["source_targets"]),
        ("null", null["final_states"]),
        ("oracle", oracle["final_states"]),
    ):
        rendered = rasterize_population(rows)[1]
        for image, sample_id in zip(rendered, authority["sample_ids"], strict=True):
            path = run_dir / f"oracle_control/images/{directory}/{sample_id}.png"
            _require(
                path.is_file() and np.array_equal(np.asarray(Image.open(path)), image),
                f"oracle image changed: {directory}/{sample_id}",
            )
    target_images = rasterize_population(authority["source_targets"])[1]
    null_images = rasterize_population(null["final_states"])[1]
    oracle_images = rasterize_population(oracle["final_states"])[1]
    _verify_contact_sheet(
        run_dir / "oracle_control/contact_sheet.png",
        np.stack(
            [
                row
                for triple in zip(target_images, null_images, oracle_images, strict=True)
                for row in triple
            ]
        ),
        columns=3,
        captions=[
            f"{int(label)}:{name}"
            for label in authority["requested_labels"]
            for name in ("target", "null", "oracle")
        ],
    )
    if (run_dir / "oracle_control/COMPLETE.json").is_file():
        complete = _read_json(run_dir / "oracle_control/COMPLETE.json")
        _require(
            complete.get("passed") == 1
            and complete["authority_sha256"] == _file_sha256(run_dir / "oracle_control/authority.npz")
            and complete["null_sha256"] == _file_sha256(run_dir / "oracle_control/null.npz")
            and complete["oracle_sha256"] == _file_sha256(run_dir / "oracle_control/oracle.npz")
            and complete["metrics_sha256"] == _file_sha256(run_dir / "oracle_control/metrics.json"),
            "oracle completion seal changed",
        )
    return saved


def _verify_forward_records(run_dir: Path) -> None:
    index = _read_json(run_dir / "forward_records/index.json")
    _require(index.get("target_semantics") == CANDIDATE_TARGET_SEMANTICS, "record target semantics changed")
    seen: dict[str, list[np.ndarray]] = {"train": [], "validation": []}
    for role in seen:
        for row in index["cohorts"][role]:
            path = run_dir / f"forward_records/{role}/cohort_{int(row['cohort']):03d}.npz"
            _require(path.is_file() and _file_sha256(path) == row["sha256"], "record cohort changed")
            arrays = _npz_arrays(path)
            _require(
                arrays["targets"].dtype == np.float32
                and arrays["later_states"].dtype == np.float32,
                "record storage dtype changed",
            )
            seen[role].append(arrays["path_ids"])
    train_ids = np.concatenate(seen["train"])
    validation_ids = np.concatenate(seen["validation"])
    _require(
        len(train_ids) == 1_000
        and len(validation_ids) == 400
        and np.unique(train_ids).size == 250
        and np.unique(validation_ids).size == 100
        and set(train_ids).isdisjoint(set(validation_ids)),
        "record path inventory changed",
    )


def _verify_partial_forward_records(run_dir: Path) -> None:
    index = _read_json(run_dir / "forward_records/index.json")
    _require(index.get("target_semantics") == CANDIDATE_TARGET_SEMANTICS, "record target semantics changed")
    seen_ids: set[int] = set()
    for role in ("train", "validation"):
        for row in index["cohorts"][role]:
            path = run_dir / f"forward_records/{role}/cohort_{int(row['cohort']):03d}.npz"
            _require(path.is_file() and _file_sha256(path) == row["sha256"], "partial record cohort changed")
            arrays = _npz_arrays(path)
            paths = np.asarray(arrays["path_ids"], dtype=np.int64)
            unique, counts = np.unique(paths, return_counts=True)
            _require(
                np.all(counts == 4)
                and all(int(value) not in seen_ids for value in unique)
                and arrays["targets"].dtype == np.float32,
                "partial record cohort semantics changed",
            )
            seen_ids.update(int(value) for value in unique)


def _bound_candidate_binary(path: Path) -> str:
    for parent in (Path(path).parent, *Path(path).parents):
        candidate_path = parent / "candidate_backend.json"
        if candidate_path.is_file():
            return str(_read_json(candidate_path)["candidate_binary_sha256"])
    raise IntegrityFailure(f"candidate backend binding is missing for {path}")


def _verify_forward_cohort_telemetry(path: Path) -> dict[str, Any]:
    telemetry = _read_json(path.with_suffix(".telemetry.json"))
    required = {
        "backend",
        "candidate_target_semantics",
        "candidate_modes",
        "candidate_bisection_steps",
        "candidate_binary_sha256",
        "outer_steps",
        "outer_step_seconds",
        "candidate_maximum_bracket_width",
        "maximum_mass_error",
        "maximum_pair_total_error",
        *_SAMPLING_ZERO_COUNTERS,
    }
    _require(
        required.issubset(telemetry)
        and telemetry["backend"] == candidate.CANDIDATE_BACKEND_NAME
        and telemetry["candidate_target_semantics"] == CANDIDATE_TARGET_SEMANTICS
        and telemetry["candidate_binary_sha256"] == _bound_candidate_binary(path)
        and int(telemetry["candidate_modes"]) == 128
        and int(telemetry["candidate_bisection_steps"]) == 56
        and int(telemetry["outer_steps"]) == 128
        and len(telemetry["outer_step_seconds"]) == 128
        and all(int(telemetry[name]) == 0 for name in _SAMPLING_ZERO_COUNTERS)
        and float(telemetry["maximum_mass_error"]) <= 2e-12
        and float(telemetry["maximum_pair_total_error"]) <= 2e-12,
        f"forward cohort telemetry changed: {path}",
    )
    return telemetry


def _verify_partial_sampling_directory(
    stage_directory: Path,
    *,
    role: str,
    image_directory: Path | None = None,
) -> list[dict[str, np.ndarray]]:
    paths = sorted(stage_directory.glob("cohort_*.npz"))
    cohorts: list[dict[str, np.ndarray]] = []
    seen: set[int] = set()
    lower, upper = PATH_RANGES[role]
    for expected_index, path in enumerate(paths):
        _require(
            path.name == f"cohort_{expected_index:03d}.npz",
            f"partial cohort index changed: {path}",
        )
        raw = _npz_arrays(path)
        rows = len(raw.get("path_ids", ()))
        _require(0 < rows <= 8, f"partial cohort size changed: {path}")
        arrays = _verify_population_stage(path, rows)
        ids = [int(value) for value in arrays["path_ids"]]
        _require(
            all(lower <= value < upper and value not in seen for value in ids),
            f"partial cohort path IDs changed: {path}",
        )
        seen.update(ids)
        if image_directory is not None:
            rendered = rasterize_population(arrays["final_states"])[1]
            for image, sample_id in zip(rendered, arrays["sample_ids"], strict=True):
                image_path = image_directory / f"{sample_id}.png"
                _require(
                    image_path.is_file()
                    and np.array_equal(np.asarray(Image.open(image_path)), image),
                    f"partial cohort image changed: {image_path}",
                )
        cohorts.append(arrays)
    return cohorts


def _verify_partial_oracle_control(run_dir: Path) -> None:
    authority_path = run_dir / "oracle_control/authority.npz"
    if not authority_path.is_file():
        return
    authority = _npz_arrays(authority_path)
    roles = _npz_arrays(run_dir / "data_roles.npz")
    expected = {
        "source_targets": roles["validation_mixed_masses"][ORACLE_VALIDATION_POSITIONS],
        "requested_labels": roles["validation_labels"][ORACLE_VALIDATION_POSITIONS],
        "arff_indices": roles["validation_arff_indices"][ORACLE_VALIDATION_POSITIONS],
        "path_ids": _role_ids("oracle"),
        "sample_ids": np.asarray(
            [f"oracle-{value:05x}" for value in _role_ids("oracle")], dtype=np.str_
        ),
    }
    _require(
        set(authority) == set(expected)
        and all(np.array_equal(authority[name], value) for name, value in expected.items()),
        "partial oracle authority changed",
    )
    terminals: dict[int, np.ndarray] = {}
    forward_paths = sorted(
        (run_dir / "oracle_control/stages/forward_terminal").glob("cohort_*.npz")
    )
    for expected_index, path in enumerate(forward_paths):
        _require(path.name == f"cohort_{expected_index:03d}.npz", "partial oracle forward index changed")
        arrays = _npz_arrays(path)
        rows = len(arrays["path_ids"])
        _require(
            0 < rows <= 8
            and arrays["source_targets"].shape
            == arrays["terminal_starts"].shape
            == (rows, 784),
            "partial oracle forward cohort shape changed",
        )
        positions = np.asarray(arrays["path_ids"], dtype=np.int64) - PATH_RANGES["oracle"][0]
        _require(
            np.all((positions >= 0) & (positions < 10))
            and np.array_equal(arrays["source_targets"], authority["source_targets"][positions])
            and np.array_equal(arrays["requested_labels"], authority["requested_labels"][positions])
            and np.array_equal(arrays["arff_indices"], authority["arff_indices"][positions])
            and np.array_equal(arrays["sample_ids"], authority["sample_ids"][positions]),
            "partial oracle forward alignment changed",
        )
        for path_id, terminal in zip(arrays["path_ids"], arrays["terminal_starts"], strict=True):
            terminals[int(path_id)] = terminal
        _verify_forward_cohort_telemetry(path)
    for controller in ("null", "oracle"):
        cohorts = _verify_partial_sampling_directory(
            run_dir / f"oracle_control/stages/{controller}",
            role="oracle",
            image_directory=run_dir / f"oracle_control/images/{controller}",
        )
        for arrays in cohorts:
            _require(
                all(
                    int(path_id) in terminals
                    and np.array_equal(start, terminals[int(path_id)])
                    for path_id, start in zip(arrays["path_ids"], arrays["starts"], strict=True)
                ),
                f"partial oracle {controller} starts changed",
            )


def _verify_partial_training(run_dir: Path) -> None:
    paths = sorted((run_dir / "training").glob("checkpoint_*.pt"))
    updates = [int(path.stem.split("_")[-1]) for path in paths]
    _require(
        updates == [250, 500, 750][: len(updates)],
        "partial training checkpoint order changed",
    )
    model_keys = set(core.make_model().state_dict())
    for path, update in zip(paths, updates, strict=True):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        history = payload.get("history") if isinstance(payload, dict) else None
        _require(
            isinstance(payload, dict)
            and payload.get("schema") == VERSION + "-training-checkpoint"
            and payload.get("target_semantics") == CANDIDATE_TARGET_SEMANTICS
            and int(payload.get("completed_update", -1)) == update
            and set(payload.get("model_state_dict", {})) == model_keys
            and set(payload.get("ema_state_dict", {})) == model_keys
            and isinstance(payload.get("optimizer_state_dict"), dict)
            and isinstance(history, (list, tuple))
            and int(history[-1]["update"]) == update,
            f"partial training checkpoint changed: {path.name}",
        )


def _verify_partial_objective_populations(run_dir: Path) -> None:
    authority_path = run_dir / "populations/prior_start_authority.npz"
    if authority_path.is_file():
        authority = _npz_arrays(authority_path)
        ids = _role_ids("prior")
        expected = {
            "prior_starts": core.sample_dirichlet_starts(
                ids, root_seed=SEEDS["prior_start"]
            ),
            "requested_labels": np.repeat(np.arange(10, dtype=np.int64), 2),
            "path_ids": ids,
            "sample_ids": np.asarray(
                [f"prior-{value:05x}" for value in ids], dtype=np.str_
            ),
        }
        _require(
            set(authority) == set(expected)
            and all(np.array_equal(authority[name], value) for name, value in expected.items()),
            "partial prior-start authority changed",
        )
        authority_json = _read_json(
            run_dir / "populations/prior_start_authority.json"
        )
        _require(
            authority_json.get("committed_before_sampling") == 1
            and authority_json.get("npz_sha256") == _file_sha256(authority_path)
            and authority_json.get("path_ids_sha256") == _array_sha256(ids),
            "partial prior-start authority seal changed",
        )
        for stage, image_directory in (
            ("null_prior", run_dir / "populations/images/prior/null"),
            ("learned_prior", run_dir / "populations/images/prior/learned"),
        ):
            cohorts = _verify_partial_sampling_directory(
                run_dir / f"populations/stages/{stage}",
                role="prior",
                image_directory=image_directory,
            )
            for arrays in cohorts:
                positions = np.asarray(arrays["path_ids"], dtype=np.int64) - ids[0]
                _require(
                    np.array_equal(arrays["starts"], authority["prior_starts"][positions])
                    and np.array_equal(
                        arrays["requested_labels"], authority["requested_labels"][positions]
                    )
                    and np.array_equal(
                        arrays["sample_ids"], authority["sample_ids"][positions]
                    ),
                    f"partial {stage} authority changed",
                )

    forward_directory = run_dir / "populations/stages/forward_terminal_starts"
    forward_paths = sorted(forward_directory.glob("cohort_*.npz"))
    if forward_paths:
        roles = _npz_arrays(run_dir / "data_roles.npz")
        targets = roles["validation_mixed_masses"][FORWARD_VALIDATION_POSITIONS]
        labels = roles["validation_labels"][FORWARD_VALIDATION_POSITIONS]
        ids = _role_ids("forward_terminal")
        sample_ids = np.asarray(
            [f"forward-{value:05x}" for value in ids], dtype=np.str_
        )
        terminals: dict[int, np.ndarray] = {}
        for expected_index, path in enumerate(forward_paths):
            _require(
                path.name == f"cohort_{expected_index:03d}.npz",
                "partial forward-terminal cohort index changed",
            )
            arrays = _npz_arrays(path)
            rows = len(arrays["path_ids"])
            positions = np.asarray(arrays["path_ids"], dtype=np.int64) - ids[0]
            _require(
                0 < rows <= 8
                and np.all((positions >= 0) & (positions < 20))
                and np.array_equal(arrays["source_targets"], targets[positions])
                and np.array_equal(arrays["requested_labels"], labels[positions])
                and np.array_equal(arrays["sample_ids"], sample_ids[positions]),
                "partial forward-terminal authority changed",
            )
            for path_id, terminal in zip(
                arrays["path_ids"], arrays["terminal_starts"], strict=True
            ):
                terminals[int(path_id)] = terminal
            _verify_forward_cohort_telemetry(path)
        for stage, image_directory in (
            (
                "null_forward_terminal",
                run_dir / "populations/images/forward/null",
            ),
            (
                "learned_forward_terminal",
                run_dir / "populations/images/forward/learned",
            ),
        ):
            cohorts = _verify_partial_sampling_directory(
                run_dir / f"populations/stages/{stage}",
                role="forward_terminal",
                image_directory=image_directory,
            )
            for arrays in cohorts:
                _require(
                    all(
                        int(path_id) in terminals
                        and np.array_equal(start, terminals[int(path_id)])
                        for path_id, start in zip(
                            arrays["path_ids"], arrays["starts"], strict=True
                        )
                    ),
                    f"partial {stage} starts changed",
                )

    for name, role in (
        ("null_prior", "prior"),
        ("learned_prior", "prior"),
        ("null_forward_terminal", "forward_terminal"),
        ("learned_forward_terminal", "forward_terminal"),
    ):
        path = run_dir / f"populations/{name}.npz"
        if path.is_file():
            arrays = _verify_population_stage(path, 20)
            _verify_assembled_from_cohorts(
                run_dir / f"populations/stages/{name}", arrays
            )


def _verify_training(run_dir: Path) -> None:
    _verify_partial_training(run_dir)
    history = _read_json(run_dir / "training/history.json")["rows"]
    replayed = select_earliest_checkpoint(history)
    saved = _read_json(run_dir / "training/selection.json")
    _require(
        saved["selected_update"] == replayed["selected_update"]
        and saved["selected_validation_normalized_mse"]
        == replayed["selected_validation_normalized_mse"]
        and saved["checkpoint_sha256"]
        == _file_sha256(run_dir / "training/selected_checkpoint.pt"),
        "training selection changed",
    )
    for update in (250, 500, 750):
        _require((run_dir / f"training/checkpoint_{update:04d}.pt").is_file(), "training callback checkpoint missing")
    final_checkpoint = torch.load(
        run_dir / "training/checkpoint_0750.pt",
        map_location="cpu",
        weights_only=True,
    )
    selected = torch.load(
        run_dir / "training/selected_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    _require(
        list(final_checkpoint["history"]) == history
        and selected.get("schema") == VERSION + "-selected-checkpoint"
        and int(selected["selected_update"]) == int(saved["selected_update"])
        and int(selected["completed_updates"]) == 750
        and int(selected["model_parameter_count"]) == 34_974
        and set(selected["state_dict"]) == set(final_checkpoint["best_state_dict"])
        and all(
            torch.equal(selected["state_dict"][name], final_checkpoint["best_state_dict"][name])
            for name in selected["state_dict"]
        ),
        "selected training checkpoint changed",
    )


def _verify_population_stage(path: Path, expected_rows: int) -> dict[str, np.ndarray]:
    arrays = _npz_arrays(path)
    required = {
        "starts",
        "anchors_000",
        "anchors_032",
        "anchors_064",
        "anchors_096",
        "anchors_128",
        "final_states",
        "requested_labels",
        "path_ids",
        "sample_ids",
    }
    _require(required.issubset(arrays), f"population stage schema changed: {path.name}")
    _require(
        arrays["starts"].shape == arrays["final_states"].shape == (expected_rows, 784)
        and np.array_equal(arrays["starts"], arrays["anchors_000"])
        and np.array_equal(arrays["final_states"], arrays["anchors_128"]),
        f"population stage endpoints changed: {path.name}",
    )
    _require(
        arrays["requested_labels"].shape
        == arrays["path_ids"].shape
        == arrays["sample_ids"].shape
        == (expected_rows,)
        and np.unique(arrays["path_ids"]).size == expected_rows
        and np.unique(arrays["sample_ids"]).size == expected_rows,
        f"population stage identities changed: {path.name}",
    )
    for name in ("starts", "anchors_000", "anchors_032", "anchors_064", "anchors_096", "anchors_128", "final_states"):
        values = np.asarray(arrays[name], dtype=np.float64)
        _require(
            np.isfinite(values).all()
            and np.all(values >= 0.0)
            and float(np.max(np.abs(values.sum(axis=1) - 1.0))) <= 2e-12,
            f"population stage health changed: {path.name}:{name}",
        )
    telemetry = _read_json(path.with_suffix(".telemetry.json"))
    _require(
        _SAMPLING_REQUIRED_TELEMETRY.issubset(telemetry)
        and int(telemetry["finite"]) == 1
        and int(telemetry["nonnegative"]) == 1
        and int(telemetry["candidate_modes"]) == 128
        and int(telemetry["candidate_bisection_steps"]) == 56
        and telemetry["backend"] == candidate.CANDIDATE_BACKEND_NAME
        and telemetry["candidate_target_semantics"] == CANDIDATE_TARGET_SEMANTICS
        and telemetry["candidate_binary_sha256"] == _bound_candidate_binary(path)
        and all(int(telemetry[name]) == 0 for name in _SAMPLING_ZERO_COUNTERS)
        and float(telemetry["maximum_mass_error"]) <= 2e-12
        and float(telemetry["maximum_pair_total_error"]) <= 2e-12,
        f"population stage telemetry changed: {path.name}",
    )
    return arrays


def _verify_assembled_from_cohorts(
    stage_directory: Path, assembled: Mapping[str, np.ndarray]
) -> None:
    paths = sorted(stage_directory.glob("cohort_*.npz"))
    _require(bool(paths), f"population cohorts are missing: {stage_directory}")
    cohorts = [_verify_population_stage(path, len(_npz_arrays(path)["path_ids"])) for path in paths]
    for name in (
        "starts",
        "anchors_000",
        "anchors_032",
        "anchors_064",
        "anchors_096",
        "anchors_128",
        "final_states",
        "requested_labels",
        "path_ids",
        "sample_ids",
    ):
        _require(
            np.array_equal(np.concatenate([row[name] for row in cohorts]), assembled[name]),
            f"assembled population differs from cohorts: {stage_directory.name}:{name}",
        )


def _verify_populations(run_dir: Path, *, require_seal: bool = True) -> None:
    expected_rows = {
        "null_prior": 20,
        "learned_prior": 20,
        "null_forward_terminal": 20,
        "learned_forward_terminal": 20,
        "null_oracle": 10,
        "oracle": 10,
    }
    stage_arrays = {
        name: _verify_population_stage(
            run_dir / f"populations/{name}.npz", rows
        )
        for name, rows in expected_rows.items()
    }
    roles = _npz_arrays(run_dir / "data_roles.npz")
    prior_ids = _role_ids("prior")
    prior_labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    prior_sample_ids = np.asarray(
        [f"prior-{value:05x}" for value in prior_ids], dtype=np.str_
    )
    prior_authority_path = run_dir / "populations/prior_start_authority.npz"
    prior_authority = _npz_arrays(prior_authority_path)
    expected_prior_authority = {
        "prior_starts": core.sample_dirichlet_starts(
            prior_ids, root_seed=SEEDS["prior_start"]
        ),
        "requested_labels": prior_labels,
        "path_ids": prior_ids,
        "sample_ids": prior_sample_ids,
    }
    prior_authority_record = _read_json(
        run_dir / "populations/prior_start_authority.json"
    )
    _require(
        set(prior_authority) == set(expected_prior_authority)
        and all(
            np.array_equal(prior_authority[name], expected)
            for name, expected in expected_prior_authority.items()
        )
        and prior_authority_record
        == {
            "schema": VERSION + "-prior-start-authority",
            "committed_before_sampling": 1,
            "label_independent_start_law": "Dirichlet(1,...,1)",
            "npz_sha256": _file_sha256(prior_authority_path),
            "path_ids_sha256": _array_sha256(prior_ids),
        },
        "prior-start authority changed",
    )
    for name in ("null_prior", "learned_prior"):
        arrays = stage_arrays[name]
        _require(
            np.array_equal(arrays["starts"], prior_authority["prior_starts"])
            and np.array_equal(arrays["requested_labels"], prior_labels)
            and np.array_equal(arrays["path_ids"], prior_ids)
            and np.array_equal(arrays["sample_ids"], prior_sample_ids),
            f"frozen prior population identity changed: {name}",
        )

    forward_ids = _role_ids("forward_terminal")
    forward_labels = roles["validation_labels"][FORWARD_VALIDATION_POSITIONS]
    forward_targets = roles["validation_mixed_masses"][
        FORWARD_VALIDATION_POSITIONS
    ]
    forward_sample_ids = np.asarray(
        [f"forward-{value:05x}" for value in forward_ids], dtype=np.str_
    )
    forward_paths = sorted(
        (run_dir / "populations/stages/forward_terminal_starts").glob(
            "cohort_*.npz"
        )
    )
    _require(
        [path.name for path in forward_paths]
        == ["cohort_000.npz", "cohort_001.npz", "cohort_002.npz"],
        "forward-terminal start cohort inventory changed",
    )
    forward_start_cohorts = [_npz_arrays(path) for path in forward_paths]
    for path in forward_paths:
        _verify_forward_cohort_telemetry(path)
    forward_terminal_starts = np.concatenate(
        [arrays["terminal_starts"] for arrays in forward_start_cohorts]
    )
    for name, expected in {
        "source_targets": forward_targets,
        "requested_labels": forward_labels,
        "path_ids": forward_ids,
        "sample_ids": forward_sample_ids,
    }.items():
        _require(
            np.array_equal(
                np.concatenate([arrays[name] for arrays in forward_start_cohorts]),
                expected,
            ),
            f"forward-terminal start authority changed: {name}",
        )
    for name in ("null_forward_terminal", "learned_forward_terminal"):
        arrays = stage_arrays[name]
        _require(
            np.array_equal(arrays["starts"], forward_terminal_starts)
            and np.array_equal(arrays["terminal_starts"], forward_terminal_starts)
            and np.array_equal(arrays["source_targets"], forward_targets)
            and np.array_equal(arrays["requested_labels"], forward_labels)
            and np.array_equal(arrays["path_ids"], forward_ids)
            and np.array_equal(arrays["sample_ids"], forward_sample_ids),
            f"frozen forward-terminal population identity changed: {name}",
        )
    for left, right in (
        ("null_prior", "learned_prior"),
        ("null_forward_terminal", "learned_forward_terminal"),
        ("null_oracle", "oracle"),
    ):
        _require(
            all(
                np.array_equal(stage_arrays[left][field], stage_arrays[right][field])
                for field in ("requested_labels", "path_ids", "sample_ids")
            ),
            f"paired population identity changed: {left}/{right}",
        )
    stage_index = _read_json(run_dir / "populations/stage_index.json")
    population_telemetry = _read_json(run_dir / "populations/telemetry.json")
    _require(
        stage_index.get("oracle_recomputed_after_training") == 0
        and set(stage_index.get("stages", {})) == set(expected_rows)
        and stage_index.get("oracle_complete_sha256")
        == _file_sha256(run_dir / "oracle_control/COMPLETE.json"),
        "population stage index changed",
    )
    for name, arrays in stage_arrays.items():
        row = stage_index["stages"][name]
        _require(
            int(row["row_count"]) == expected_rows[name]
            and row["path_ids_sha256"] == _array_sha256(arrays["path_ids"])
            and row["assembled_sha256"]
            == _file_sha256(run_dir / f"populations/{name}.npz"),
            f"population stage index changed: {name}",
        )
        _require(
            _semantic_sha256(population_telemetry.get(name))
            == _semantic_sha256(
                _read_json(run_dir / f"populations/{name}.telemetry.json")
            ),
            f"population telemetry changed: {name}",
        )
        if name in {"null_oracle", "oracle"}:
            source_name = "null" if name == "null_oracle" else "oracle"
            source_path = run_dir / f"oracle_control/{source_name}.npz"
            source = _npz_arrays(source_path)
            _require(
                row.get("reused_source_path") == f"oracle_control/{source_name}.npz"
                and row.get("reused_source_sha256") == _file_sha256(source_path)
                and all(
                    key in source and np.array_equal(value, source[key])
                    for key, value in arrays.items()
                )
                and _read_json(run_dir / f"populations/{name}.telemetry.json")
                == _read_json(run_dir / f"oracle_control/{source_name}.telemetry.json"),
                f"oracle population reuse changed: {name}",
            )
        else:
            _verify_assembled_from_cohorts(
                run_dir / f"populations/stages/{name}", arrays
            )
    _require(
        np.array_equal(stage_arrays["null_prior"]["starts"], stage_arrays["learned_prior"]["starts"])
        and np.array_equal(
            stage_arrays["null_forward_terminal"]["starts"],
            stage_arrays["learned_forward_terminal"]["starts"],
        ),
        "paired population starts changed",
    )
    raw = _npz_arrays(run_dir / "populations/raw_populations.npz")
    demixed = _npz_arrays(run_dir / "populations/demixed_populations.npz")
    rendered = _npz_arrays(run_dir / "populations/uint8_populations.npz")
    identities = _npz_arrays(run_dir / "populations/identities.npz")
    expected_raw = {
        name: arrays["final_states"] for name, arrays in stage_arrays.items()
    }
    expected_raw["forward_targets"] = stage_arrays["null_forward_terminal"][
        "source_targets"
    ]
    expected_raw["oracle_targets"] = _npz_arrays(
        run_dir / "oracle_control/authority.npz"
    )["source_targets"]
    for name, expected in expected_raw.items():
        _require(
            name in raw and np.array_equal(raw[name], expected),
            f"raw population differs from assembled authority: {name}",
        )
    expected_identities = {
        "prior_requested_labels": stage_arrays["null_prior"]["requested_labels"],
        "prior_path_ids": stage_arrays["null_prior"]["path_ids"],
        "prior_sample_ids": stage_arrays["null_prior"]["sample_ids"],
        "forward_requested_labels": stage_arrays["null_forward_terminal"]["requested_labels"],
        "forward_path_ids": stage_arrays["null_forward_terminal"]["path_ids"],
        "forward_sample_ids": stage_arrays["null_forward_terminal"]["sample_ids"],
        "oracle_requested_labels": stage_arrays["oracle"]["requested_labels"],
        "oracle_path_ids": stage_arrays["oracle"]["path_ids"],
        "oracle_sample_ids": stage_arrays["oracle"]["sample_ids"],
    }
    _require(set(identities) == set(expected_identities), "population identity inventory changed")
    for name, expected in expected_identities.items():
        _require(np.array_equal(identities[name], expected), f"population identity changed: {name}")
    for name in (
        "null_prior",
        "learned_prior",
        "null_forward_terminal",
        "learned_forward_terminal",
        "forward_targets",
        "null_oracle",
        "oracle",
        "oracle_targets",
    ):
        expected_demixed, expected_uint8 = rasterize_population(raw[name])
        _require(
            np.array_equal(demixed[name], expected_demixed)
            and np.array_equal(rendered[name], expected_uint8),
            f"population raster replay changed: {name}",
        )
    for name, expected in expected_identities.items():
        _require(
            np.array_equal(raw[name], expected)
            and np.array_equal(demixed[name], expected)
            and np.array_equal(rendered[name], expected),
            f"population authority identity changed: {name}",
        )
    def sampling_result(name: str) -> core.SamplingResult:
        arrays = stage_arrays[name]
        return core.SamplingResult(
            starts=arrays["starts"],
            final_states=arrays["final_states"],
            anchors={
                anchor: arrays[f"anchors_{anchor:03d}"]
                for anchor in (0, 32, 64, 96, 128)
            },
            telemetry=_read_json(run_dir / f"populations/{name}.telemetry.json"),
        )

    replayed_forward_marker = _forward_direct_marker(
        expected_raw["forward_targets"],
        sampling_result("null_forward_terminal"),
        sampling_result("learned_forward_terminal"),
    )
    _require(
        _semantic_sha256(
            _read_json(run_dir / "populations/forward_direct_metrics.json")
        )
        == _semantic_sha256(replayed_forward_marker),
        "forward direct marker changed",
    )
    image_roles = (
        ("prior/null", "null_prior", "prior_sample_ids"),
        ("prior/learned", "learned_prior", "prior_sample_ids"),
        ("forward/null", "null_forward_terminal", "forward_sample_ids"),
        ("forward/learned", "learned_forward_terminal", "forward_sample_ids"),
        ("forward/targets", "forward_targets", "forward_sample_ids"),
        ("oracle/null", "null_oracle", "oracle_sample_ids"),
        ("oracle/oracle", "oracle", "oracle_sample_ids"),
        ("oracle/targets", "oracle_targets", "oracle_sample_ids"),
    )
    for directory, array_name, id_name in image_roles:
        for image, sample_id in zip(rendered[array_name], identities[id_name], strict=True):
            path = run_dir / f"populations/images/{directory}/{sample_id}.png"
            _require(path.is_file() and np.array_equal(np.asarray(Image.open(path)), image), f"PNG changed: {path}")
    _verify_contact_sheet(
        run_dir / "populations/contact_sheets/prior_null_learned.png",
        np.stack(
            [
                row
                for pair in zip(
                    rendered["null_prior"], rendered["learned_prior"], strict=True
                )
                for row in pair
            ]
        ),
        columns=4,
        captions=[
            f"{int(label)}:{kind}"
            for label in identities["prior_requested_labels"]
            for kind in ("null", "learned")
        ],
    )
    _verify_contact_sheet(
        run_dir / "populations/contact_sheets/forward_target_null_learned.png",
        np.stack(
            [
                row
                for triple in zip(
                    rendered["forward_targets"],
                    rendered["null_forward_terminal"],
                    rendered["learned_forward_terminal"],
                    strict=True,
                )
                for row in triple
            ]
        ),
        columns=6,
        captions=[
            f"{int(label)}:{kind}"
            for label in identities["forward_requested_labels"]
            for kind in ("target", "null", "learned")
        ],
    )
    _verify_contact_sheet(
        run_dir / "populations/contact_sheets/oracle_target_null_oracle.png",
        np.stack(
            [
                row
                for triple in zip(
                    rendered["oracle_targets"],
                    rendered["null_oracle"],
                    rendered["oracle"],
                    strict=True,
                )
                for row in triple
            ]
        ),
        columns=6,
        captions=[
            f"{int(label)}:{kind}"
            for label in identities["oracle_requested_labels"]
            for kind in ("target", "null", "oracle")
        ],
    )
    oracle_anchor_images = np.concatenate(
        [
            rasterize_population(stage_arrays["oracle"][f"anchors_{anchor:03d}"])[1]
            for anchor in (0, 32, 64, 96, 128)
        ]
    )
    _verify_contact_sheet(
        run_dir / "populations/contact_sheets/oracle_anchors.png",
        oracle_anchor_images,
        columns=10,
        captions=[f"a{anchor}" for anchor in (0, 32, 64, 96, 128) for _ in range(10)],
    )
    if require_seal:
        _validate_population_seal(run_dir)


def _verify_evaluation(run_dir: Path) -> dict[str, Any]:
    _, seal_sha256 = _validate_population_seal(run_dir)
    event = _read_json(run_dir / "evaluation/OPEN_EVENT.json")
    _require(
        event.get("population_seal_sha256") == seal_sha256
        and event.get("opened_after_population_seal") == 1
        and event.get("terminal_test_rows_opened") == 0,
        "evaluator open event changed",
    )
    outputs = _npz_arrays(run_dir / "evaluation/outputs.npz")
    saved = _read_json(run_dir / "evaluation/metrics.json")
    identities = _npz_arrays(run_dir / "populations/identities.npz")
    role_authorities = {
        "null_prior": ("prior_requested_labels", "prior_sample_ids"),
        "learned_prior": ("prior_requested_labels", "prior_sample_ids"),
        "null_forward_terminal": (
            "forward_requested_labels",
            "forward_sample_ids",
        ),
        "learned_forward_terminal": (
            "forward_requested_labels",
            "forward_sample_ids",
        ),
    }
    expected_output_names = {
        f"{role}_{field}"
        for role in role_authorities
        for field in (
            "logits",
            "probabilities",
            "predictions",
            "requested_labels",
            "sample_ids",
            "requested_log_probabilities",
        )
    }
    _require(
        set(outputs) == expected_output_names,
        "evaluator output inventory changed",
    )
    for role, (label_authority, sample_authority) in role_authorities.items():
        _require(
            np.array_equal(
                outputs[f"{role}_requested_labels"], identities[label_authority]
            )
            and np.array_equal(
                outputs[f"{role}_sample_ids"], identities[sample_authority]
            ),
            f"evaluator identity changed: {role}",
        )
        logits = np.asarray(outputs[f"{role}_logits"], dtype=np.float64)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        predictions = np.argmax(probabilities, axis=1)
        labels = outputs[f"{role}_requested_labels"]
        requested_log_probabilities = log_probabilities[np.arange(len(labels)), labels]
        per_class = {
            str(digit): {
                "count": int(np.sum(labels == digit)),
                "accuracy": (
                    float(np.mean(predictions[labels == digit] == digit))
                    if np.any(labels == digit)
                    else None
                ),
            }
            for digit in range(10)
        }
        _require(
            np.allclose(outputs[f"{role}_probabilities"], probabilities, rtol=1e-12, atol=1e-14)
            and np.array_equal(outputs[f"{role}_predictions"], predictions)
            and np.allclose(
                outputs[f"{role}_requested_log_probabilities"],
                requested_log_probabilities,
                rtol=1e-12,
                atol=1e-14,
            )
            and math.isclose(
                saved["populations"][role]["requested_label_accuracy"],
                float(np.mean(predictions == labels)),
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            f"saved evaluator output changed: {role}",
        )
        _require(
            saved["populations"][role]["per_class"] == per_class
            and math.isclose(
                float(saved["populations"][role]["loss"]),
                float(-np.mean(requested_log_probabilities)),
                rel_tol=1e-6,
                abs_tol=1e-7,
            ),
            f"saved evaluator metrics changed: {role}",
        )
    prior_delta = (
        outputs["learned_prior_requested_log_probabilities"]
        - outputs["null_prior_requested_log_probabilities"]
    )
    null_accuracy = float(saved["populations"]["null_prior"]["requested_label_accuracy"])
    learned_accuracy = float(saved["populations"]["learned_prior"]["requested_label_accuracy"])
    marker = {
        "gate_type": "diagnostic threshold",
        "null_prior_requested_label_accuracy": null_accuracy,
        "learned_prior_requested_label_accuracy": learned_accuracy,
        "paired_requested_log_probability_win_count": int(np.sum(prior_delta > 0.0)),
        "mean_paired_requested_log_probability_improvement": float(np.mean(prior_delta)),
    }
    marker["passed"] = int(
        learned_accuracy >= 0.20
        and learned_accuracy > null_accuracy
        and marker["paired_requested_log_probability_win_count"] >= 12
        and marker["mean_paired_requested_log_probability_improvement"] > 0.0
    )
    _require(
        np.array_equal(saved["prior_paired_requested_log_probability_effect"], prior_delta)
        and _semantic_sha256(saved["evaluator_marker"]) == _semantic_sha256(marker)
        and _semantic_sha256(saved["forward_direct_marker"])
        == _semantic_sha256(_read_json(run_dir / "populations/forward_direct_metrics.json")),
        "evaluator paired effect or marker changed",
    )
    binding = _read_json(run_dir / "evaluation/evaluator_binding.json")
    source_bindings = _read_json(run_dir / "source_bindings.json")
    _require(
        binding
        == {
            "checkpoint_sha256": source_bindings["evaluator_hashes"]["checkpoint"],
            "selection_sha256": source_bindings["evaluator_hashes"]["selection"],
            "population_seal_sha256": seal_sha256,
            "device": "cpu",
        },
        "evaluator binding changed",
    )
    return saved


_STAGE_ORDER = (
    "initialize_and_bind",
    "data_roles",
    "prepare_candidate_backend",
    "candidate_audit",
    "resource_smoke",
    "oracle_control",
    "forward_record_caches",
    "training",
    "objective_populations",
    "population_seal",
    "post_seal_evaluation_and_review_bundle",
    "record_human_review",
)


def _verify_status_stage_order(run_dir: Path, route: str) -> set[str]:
    status = _read_json(run_dir / "status.json")
    _require(
        status.get("schema") == VERSION + "-status" and status.get("route") == route,
        "status route changed",
    )
    events = _read_json(run_dir / "stage_ledger.json").get("events")
    _require(isinstance(events, list), "stage ledger events changed")
    completed_order = [
        str(row["stage"])
        for row in events
        if row.get("state") == "complete" and row.get("stage") in _STAGE_ORDER
    ]
    _require(
        completed_order == list(_STAGE_ORDER[: len(completed_order)])
        and len(completed_order) == len(set(completed_order)),
        "stage ledger completion order changed",
    )
    failed_stage_rows = [row for row in events if row.get("state") == "failed"]
    _require(
        all(
            row.get("stage") in _STAGE_ORDER
            and _STAGE_ORDER.index(str(row["stage"])) == len(completed_order)
            for row in failed_stage_rows
        ),
        "stage ledger failure order changed",
    )
    terminal = [row for row in events if row.get("stage") == "terminal_failure"]
    failure_routes = {
        "integrity_failed",
        "candidate_health_failed",
        "resource_projection_failed",
        "oracle_control_failed",
        "resource_stopped",
    }
    if route in failure_routes:
        _require(
            len(terminal) == 1
            and events[-1] == terminal[0]
            and terminal[0].get("state") == "complete"
            and terminal[0].get("route") == route
            and _read_json(run_dir / "failure.json").get("route") == route,
            "failure status and stage ledger disagree",
        )
    elif route == "review_pending":
        _require(
            completed_order == list(_STAGE_ORDER[:-1])
            and not terminal
            and not (run_dir / "failure.json").exists(),
            "review-pending status and stage ledger disagree",
        )
    else:
        _require(
            route == "complete"
            and completed_order == list(_STAGE_ORDER)
            and not terminal
            and not (run_dir / "failure.json").exists(),
            "complete status and stage ledger disagree",
        )
    return set(completed_order)


def _verify_path_id_authority(run_dir: Path) -> None:
    bindings = _read_json(run_dir / "source_bindings.json")
    saved = _read_json(run_dir / "path_id_audit.json")
    replayed = path_id_audit(Path(bindings["repository_root"]))
    _require(
        _semantic_sha256(saved) == _semantic_sha256(replayed)
        and saved.get("roles")
        == {name: _path_range(*bounds) for name, bounds in PATH_RANGES.items()}
        and saved.get("twenty_bit") == 1
        and saved.get("passed") == 1,
        "path-ID audit changed",
    )


def _verify_contact_sheet(
    path: Path,
    images: np.ndarray,
    *,
    columns: int,
    captions: Sequence[str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="candidate-contact-sheet-") as directory:
        replay = Path(directory) / "replay.png"
        write_contact_sheet(replay, images, columns=columns, captions=captions)
        _require(
            path.is_file() and _file_sha256(path) == _file_sha256(replay),
            f"contact sheet changed: {path}",
        )


def _verify_review_bundle(run_dir: Path) -> None:
    _, seal_sha256 = _validate_population_seal(run_dir)
    ready = _read_json(run_dir / "review/READY.json")
    key = _read_json(run_dir / "review/review_key.json")
    entries = key.get("entries")
    _require(
        ready.get("schema") == VERSION + "-review-ready"
        and ready.get("population_seal_sha256") == seal_sha256
        and ready.get("row_count") == 40
        and ready.get("learned_count") == 20
        and ready.get("null_count") == 20
        and ready.get("template_sha256")
        == _file_sha256(run_dir / "review/review_template.csv")
        and ready.get("key_sha256") == _file_sha256(run_dir / "review/review_key.json")
        and ready.get("contact_sheet_sha256")
        == _file_sha256(run_dir / "review/blind_contact_sheet.png")
        and key.get("schema") == VERSION + "-blind-review-key"
        and key.get("population_seal_sha256") == seal_sha256
        and key.get("seed") == SEEDS["review_shuffle"]
        and isinstance(entries, list)
        and len(entries) == 40,
        "blind review bundle binding changed",
    )
    with (run_dir / "review/review_template.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        template = list(csv.DictReader(handle))
    _require(
        len(template) == 40
        and all(
            int(row["review_order"]) == index
            and row["sample_id"] == f"blind-{index:03d}"
            and row["assigned_label"] == ""
            and row["notes"] == ""
            for index, row in enumerate(template)
        ),
        "blind review template changed",
    )
    populations = _npz_arrays(run_dir / "populations/uint8_populations.npz")
    source_images: dict[str, np.ndarray] = {}
    for prefix, role in (("learned", "learned_prior"), ("null", "null_prior")):
        source_images.update(
            {
                f"{prefix}:{sample_id}": image
                for sample_id, image in zip(
                    populations["prior_sample_ids"], populations[role], strict=True
                )
            }
        )
    ordered: list[np.ndarray] = []
    pair_roles: dict[int, set[str]] = {}
    for index, entry in enumerate(entries):
        blind_id = f"blind-{index:03d}"
        source_id = str(entry["source_sample_id"])
        _require(
            entry.get("review_order") == index
            and entry.get("sample_id") == blind_id
            and source_id in source_images
            and entry.get("controller") in {"learned", "null"},
            "blind review key row changed",
        )
        pair_roles.setdefault(int(entry["path_id"]), set()).add(
            str(entry["controller"])
        )
        image = source_images[source_id]
        image_path = run_dir / f"review/samples/{blind_id}.png"
        _require(
            image_path.is_file()
            and np.array_equal(np.asarray(Image.open(image_path)), image),
            f"blind review image changed: {blind_id}",
        )
        ordered.append(image)
    _require(
        len(pair_roles) == 20
        and all(value == {"learned", "null"} for value in pair_roles.values()),
        "blind review pair roles changed",
    )
    _verify_contact_sheet(
        run_dir / "review/blind_contact_sheet.png",
        np.stack(ordered),
        columns=8,
        captions=[f"blind-{index:03d}" for index in range(40)],
    )


def _verify_reports(run_dir: Path, outcome: Mapping[str, Any]) -> None:
    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    note = (run_dir / "experiment_note.md").read_text(encoding="utf-8")
    selection = _read_json(run_dir / "training/selection.json")
    required_report = (
        "## Research mode and decision",
        "## Source revision and fixed scientific configuration",
        "## Result and metric hierarchy",
        "Primary objective metrics",
        "Mechanism diagnostics",
        "Health metrics and typed gates",
        "execution/integrity",
        "diagnostic-threshold",
        "training/selected_checkpoint.pt",
        "candidate_audit/report.json",
        "oracle_control/metrics.json",
        "evaluation/metrics.json",
        "Resource accounting",
        "command.txt",
        "Next action",
        "## Outcome-to-action table",
        "## Exact commands",
        "## Claim boundary and deliberate omissions",
        str(outcome["route"]),
        str(selection["checkpoint_sha256"]),
        (run_dir / "command.txt").read_text(encoding="utf-8").strip(),
        (run_dir / "review/record_command.txt").read_text(encoding="utf-8").strip(),
    )
    required_note = (
        "## Mode, objective, and evidence",
        "## Resources and typed metrics",
        "## Claim boundary and deliberate omissions",
        "## Exact commands",
        str(outcome["route"]),
    )
    _require(
        all(value in report for value in required_report)
        and all(value in note for value in required_note),
        "report or experiment-note contract changed",
    )


def _verify_failure_reports(run_dir: Path, route: str) -> None:
    failure = _read_json(run_dir / "failure.json")
    objective_available = int(
        (run_dir / "populations/raw_populations.npz").is_file()
        and (run_dir / "evaluation/metrics.json").is_file()
    )
    expected_proposition = {
        "candidate_health_failed": (
            "the fixed 512-lane shared-v2-Philox candidate numerical execution/integrity criterion did not pass"
        ),
        "oracle_control_failed": (
            "the fixed ten-path oracle/null positive-control execution/integrity criterion did not pass"
        ),
        "resource_projection_failed": "a required major-stage resource projection did not fit",
        "resource_stopped": (
            "a priced resource quantum did not fit or exceeded its cap after completion"
        ),
        "integrity_failed": "an execution or artifact-integrity requirement failed",
    }[route]
    expected_boundary = (
        "saved direct and evaluator artifacts may be interpreted only at their exact exploratory scope; "
        "this resource/integrity terminal route does not supply a budget-valid completed human claim"
        if objective_available
        else "no learned-controller or generator scientific conclusion is available from this route"
    )
    terminal_receipts = [
        row
        for row in _read_json(run_dir / "resource_ledger.json").get(
            "admissions", []
        )
        if row.get("kind") == "failure_terminalization"
    ]
    _require(
        failure.get("schema") == VERSION + "-failure"
        and failure.get("route") == route
        and isinstance(failure.get("error_type"), str)
        and isinstance(failure.get("error"), str)
        and isinstance(failure.get("traceback"), str)
        and int(failure.get("scientific_objective_result_available", -1))
        == objective_available
        and failure.get("exact_failure_proposition") == expected_proposition
        and failure.get("claim_boundary") == expected_boundary
        and failure.get("full_scale_auto_launched") == 0
        and len(terminal_receipts) == 1
        and failure.get("failure_terminalization_admitted")
        == terminal_receipts[0].get("passed")
        and float(failure.get("failure_terminalization_reserve_seconds", -1.0))
        == float(terminal_receipts[0]["reserve_remaining_seconds"])
        == 0.0
        and _read_json(run_dir / "status.json").get("error")
        == failure.get("error"),
        "failure authority changed",
    )
    _require(
        (run_dir / "REPORT.md").read_text(encoding="utf-8")
        == _failure_report(run_dir, failure)
        and (run_dir / "experiment_note.md").read_text(encoding="utf-8")
        == _failure_experiment_note(run_dir, failure),
        "failure report or experiment note changed",
    )


def verify_run(run_dir: Path) -> dict[str, Any]:
    """Read-only verification of complete and explicitly partial terminal trees."""

    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), "run directory is missing")
    before = _tree_snapshot(run_dir)
    status = _read_json(run_dir / "status.json")
    route = str(status.get("route"))
    allowed = {
        "integrity_failed",
        "candidate_health_failed",
        "resource_projection_failed",
        "oracle_control_failed",
        "resource_stopped",
        "review_pending",
        "complete",
    }
    _require(route in allowed, f"run route is not terminal: {route}")
    _verify_source_bindings(run_dir)
    _verify_config_and_resources(run_dir, route)
    completed_stages = _verify_status_stage_order(run_dir, route)
    _verify_path_id_authority(run_dir)
    _require(
        not any("terminal_test_open" in path.name.lower() for path in run_dir.rglob("*")),
        "terminal-test access marker exists",
    )

    def has_files(relative: str) -> bool:
        path = run_dir / relative
        return path.is_file() or (path.is_dir() and any(item.is_file() for item in path.rglob("*")))

    later_stage_rules = (
        (
            "candidate_audit",
            ("resource_smoke", "oracle_control", "forward_records", "training", "populations", "evaluation"),
        ),
        (
            "oracle_control",
            ("forward_records", "training", "populations", "evaluation"),
        ),
        ("forward_record_caches", ("training", "populations", "evaluation")),
        ("training", ("populations", "evaluation")),
        ("objective_populations", ("POPULATIONS_SEALED.json", "evaluation")),
        ("population_seal", ("evaluation", "review/READY.json")),
    )
    for prerequisite, later_paths in later_stage_rules:
        if prerequisite not in completed_stages:
            _require(
                not any(has_files(relative) for relative in later_paths),
                f"artifact appears beyond the completed stage before {prerequisite}",
            )

    if (run_dir / "data_roles.npz").is_file():
        _verify_data_roles(run_dir)
    if (run_dir / "candidate_backend.json").is_file():
        _verify_candidate_backend(run_dir)
    audit_report: dict[str, Any] | None = None
    if (run_dir / "candidate_audit/report.json").is_file():
        audit_report = _verify_candidate_audit(run_dir)
    if "resource_smoke" in completed_stages:
        _require(
            audit_report is not None
            and gate_b_passed(audit_report)
            and (run_dir / "resource_smoke/timings.json").is_file(),
            "a later stage exists without a passing candidate audit and smoke",
        )
    if route == "candidate_health_failed":
        _require(audit_report is not None and not gate_b_passed(audit_report), "candidate failure route changed")
        _require(
            not any(
                has_files(relative)
                for relative in (
                    "resource_smoke",
                    "oracle_control",
                    "forward_records",
                    "training",
                    "populations",
                    "evaluation",
                )
            ),
            "later-stage artifact exists after candidate failure",
        )
    oracle_metrics: dict[str, Any] | None = None
    if (run_dir / "oracle_control/metrics.json").is_file():
        oracle_metrics = _verify_oracle_control(run_dir)
        if route == "oracle_control_failed":
            _require(not gate_c_passed(oracle_metrics), "oracle failure route changed")
            _require(
                not any(
                    has_files(relative)
                    for relative in ("forward_records", "training", "populations", "evaluation")
                ),
                "later-stage artifact exists after oracle failure",
            )
    else:
        _verify_partial_oracle_control(run_dir)
    if "forward_record_caches" in completed_stages:
        _require(
            oracle_metrics is not None and gate_c_passed(oracle_metrics),
            "a later stage exists without a passing oracle control",
        )
    if (run_dir / "forward_records/index.json").is_file():
        if "forward_record_caches" in completed_stages:
            _verify_forward_records(run_dir)
        else:
            _verify_partial_forward_records(run_dir)
    if (run_dir / "training/selection.json").is_file():
        _verify_training(run_dir)
    elif has_files("training"):
        _verify_partial_training(run_dir)
    if "objective_populations" in completed_stages:
        _require(
            (run_dir / "populations/raw_populations.npz").is_file(),
            "completed objective population authority is missing",
        )
        _verify_populations(
            run_dir,
            require_seal=(run_dir / "POPULATIONS_SEALED.json").is_file(),
        )
    else:
        _verify_partial_objective_populations(run_dir)

    route_required = {
        "review_pending": (
            "POPULATIONS_SEALED.json",
            "evaluation/OPEN_EVENT.json",
            "evaluation/outputs.npz",
            "evaluation/metrics.json",
            "evaluation/evaluator_binding.json",
            "review/READY.json",
            "review/review_key.json",
            "review/review_template.csv",
            "review/blind_contact_sheet.png",
            "outcome_pre_review.json",
        ),
        "complete": (
            "POPULATIONS_SEALED.json",
            "evaluation/OPEN_EVENT.json",
            "evaluation/outputs.npz",
            "evaluation/metrics.json",
            "evaluation/evaluator_binding.json",
            "review/READY.json",
            "review/review_key.json",
            "review/review_template.csv",
            "review/blind_contact_sheet.png",
            "review/record_command.txt",
            "review/submitted_answers.csv",
            "review/metrics.json",
            "outcome.json",
            "REPORT.md",
            "experiment_note.md",
        ),
    }
    failure_routes = {
        "integrity_failed",
        "candidate_health_failed",
        "resource_projection_failed",
        "oracle_control_failed",
        "resource_stopped",
    }
    if route in failure_routes:
        required_failure = ["failure.json", "REPORT.md", "experiment_note.md"]
        if route in {"resource_projection_failed", "resource_stopped"}:
            required_failure.append("resource_stop.json")
        _require(
            all((run_dir / relative).is_file() for relative in required_failure),
            f"{route} required failure artifacts are missing",
        )
        _verify_failure_reports(run_dir, route)
    if route in route_required:
        _require(
            all((run_dir / relative).is_file() for relative in route_required[route]),
            f"{route} required artifacts are missing",
        )
        _verify_populations(run_dir)
        _verify_review_bundle(run_dir)
    if (run_dir / "evaluation/metrics.json").is_file():
        evaluation = _verify_evaluation(run_dir)
        if route == "review_pending":
            _require(
                not (run_dir / "review/submitted_answers.csv").exists()
                and not (run_dir / "outcome.json").exists()
                and not (run_dir / "REPORT.md").exists()
                and not (run_dir / "experiment_note.md").exists(),
                "review-pending tree contains opened answers",
            )
        if route == "complete":
            replayed = compute_human_review_metrics(
                run_dir / "review/submitted_answers.csv",
                _read_json(run_dir / "review/review_key.json"),
            )
            saved_review = _read_json(run_dir / "review/metrics.json")
            expected_review_command = _canonical_command_text(
                [
                    str(Path(sys.executable).resolve()),
                    "-B",
                    "-m",
                    "mnist.diag_eulerian_jacobi_ddpm_candidate_pilot",
                    "record-review",
                    "--run-dir",
                    str(run_dir),
                    "--answers",
                    str(saved_review["answers_source_path"]),
                    "--reviewer",
                    str(saved_review["reviewer"]),
                    "--confirm-manual-review",
                ]
            )
            _require(
                (run_dir / "review/record_command.txt").read_text(encoding="utf-8")
                == expected_review_command,
                "record-review command changed",
            )
            for key in (
                "learned",
                "null",
                "learned_counts_exceed_null",
                "rows",
                "paired_paths",
                "aggregate_paired_recognizable_difference",
                "aggregate_paired_requested_label_difference",
                "paired_recognizable_win_loss_tie",
                "paired_requested_label_win_loss_tie",
            ):
                _require(
                    _semantic_sha256(saved_review[key]) == _semantic_sha256(replayed[key]),
                    f"human review metric changed: {key}",
                )
            replayed_marker = _human_marker_from_metrics(run_dir, replayed)
            _require(
                _semantic_sha256(saved_review.get("human_marker"))
                == _semantic_sha256(replayed_marker),
                "human review marker changed",
            )
            evaluation_outputs = _npz_arrays(run_dir / "evaluation/outputs.npz")
            source_to_prediction: dict[str, int] = {}
            for role in ("learned_prior", "null_prior"):
                prefix = "learned" if role.startswith("learned") else "null"
                source_to_prediction.update(
                    {
                        f"{prefix}:{sample_id}": int(prediction)
                        for sample_id, prediction in zip(
                            evaluation_outputs[f"{role}_sample_ids"],
                            evaluation_outputs[f"{role}_predictions"],
                            strict=True,
                        )
                    }
                )
            disagreement = int(
                sum(
                    str(row["assigned_label"]).isdigit()
                    and int(row["assigned_label"])
                    != source_to_prediction[str(row["source_sample_id"])]
                    for row in replayed["rows"]
                )
            )
            _require(
                int(saved_review.get("human_machine_disagreement_count", -1))
                == disagreement,
                "human/machine disagreement count changed",
            )
            outcome = _read_json(run_dir / "outcome.json")
            _require(
                _semantic_sha256(outcome.get("human_marker"))
                == _semantic_sha256(replayed_marker)
                and
                outcome["route"]
                == route_outcome(
                    evaluation["forward_direct_marker"],
                    outcome["human_marker"],
                    evaluation["evaluator_marker"],
                ),
                "outcome route changed",
            )
            _verify_reports(run_dir, outcome)
    manifest = _verify_manifest(run_dir)
    after = _tree_snapshot(run_dir)
    _require(before == after, "verification mutated the run tree")
    return {
        "artifact_count": int(manifest["artifact_count"]),
        "passed": 1,
        "route": route,
        "tree_digest": str(manifest["tree_digest"]),
    }


def _failure_route(error: BaseException) -> str:
    if isinstance(error, CandidateHealthFailure):
        return "candidate_health_failed"
    if isinstance(error, ResourceProjectionFailed):
        return "resource_projection_failed"
    if isinstance(error, OracleControlFailed):
        return "oracle_control_failed"
    if isinstance(error, ResourceStop):
        return "resource_stopped"
    return "integrity_failed"


def _failure_action(route: str) -> str:
    return {
        "candidate_health_failed": (
            "repair or reject the candidate numerical integration before judging the learner"
        ),
        "oracle_control_failed": (
            "repair composition, orientation, schedule, or backend integration before learner attribution"
        ),
        "resource_projection_failed": (
            "inspect the retained evidence and redesign a smaller fixed pilot or seek fresh approval"
        ),
        "resource_stopped": (
            "inspect the last durable cohort and redesign a smaller fixed pilot or seek fresh approval"
        ),
        "integrity_failed": "repair the named integrity defect and restart the whole run",
    }[route]


def _failure_report(run_dir: Path, failure: Mapping[str, Any]) -> str:
    config = _read_json(run_dir / "config.json")
    bindings = _read_json(run_dir / "source_bindings.json")
    ledger = _read_json(run_dir / "resource_ledger.json")
    stage_rows = _read_json(run_dir / "stage_ledger.json")["events"]
    completed = [
        str(row["stage"])
        for row in stage_rows
        if row.get("state") == "complete" and row.get("stage") != "terminal_failure"
    ]
    retained_files = [
        relative
        for relative in (
            "data_roles.npz",
            "candidate_backend.json",
            "candidate_audit/bank.npz",
            "candidate_audit/outputs.npz",
            "candidate_audit/report.json",
            "resource_smoke/timings.json",
            "oracle_control/authority.npz",
            "oracle_control/null.npz",
            "oracle_control/oracle.npz",
            "oracle_control/metrics.json",
            "forward_records/index.json",
            "training/history.json",
            "training/selection.json",
            "populations/raw_populations.npz",
            "POPULATIONS_SEALED.json",
            "evaluation/metrics.json",
            "review/READY.json",
            "review/metrics.json",
        )
        if (run_dir / relative).is_file()
    ]
    cohort_count = len(list(run_dir.rglob("cohort_*.npz")))
    png_count = len(list(run_dir.rglob("*.png")))
    run_command = (run_dir / "command.txt").read_text(encoding="utf-8").strip()
    verify_command = _canonical_command_text(
        [
            str(Path(sys.executable).resolve()),
            "-B",
            "-m",
            "mnist.diag_eulerian_jacobi_ddpm_candidate_pilot",
            "verify",
            "--run-dir",
            str(run_dir),
        ]
    ).strip()
    completed_text = ", ".join(completed) if completed else "none"
    retained_text = ", ".join(retained_files) if retained_files else "none"
    objective_available = int(failure["scientific_objective_result_available"])
    route = str(failure["route"])
    if route == "candidate_health_failed":
        audit = _read_json(run_dir / "candidate_audit/report.json")
        fast = audit["candidate_vs_568"]
        exact = audit["candidate_vs_certified"]
        rng = audit["rng_alignment"]
        decisive_evidence = (
            "Gate B used the shared `philox4x32-10-canonical-transition-v2` "
            "transition-ID stream. Candidate-versus-568 maximum later/target errors: "
            f"`{float(fast['maximum_later_fraction_error']):.17g}` / "
            f"`{float(fast['maximum_target_error']):.17g}`; "
            "candidate-versus-certified maximum later/target errors: "
            f"`{float(exact['maximum_later_fraction_error']):.17g}` / "
            f"`{float(exact['maximum_target_error']):.17g}`. Fixed thresholds: "
            f"`{float(config['numerical_gate']['maximum_later_error']):.17g}` / "
            f"`{float(config['numerical_gate']['maximum_target_error']):.17g}`. "
            "The saved RNG witness is `candidate_audit/outputs.npz`; its alignment "
            f"replay flag is `{int(rng['rng_arrays_exact'])}` and the maximum 568-CDF "
            "residual at the candidate state is "
            f"`{float(rng['maximum_568_cdf_residual_at_candidate_later']):.17g}`."
        )
    elif route == "oracle_control_failed":
        oracle = _read_json(run_dir / "oracle_control/metrics.json")
        decisive_evidence = (
            "The oracle/null control retained `oracle_control/metrics.json`. "
            f"Improved paths: `{int(oracle['oracle_improved_path_count'])}` of 10; "
            "aggregate null/oracle L1: "
            f"`{float(oracle['aggregate_null_final_raw_mass_l1']):.17g}` / "
            f"`{float(oracle['aggregate_oracle_final_raw_mass_l1']):.17g}`. "
            "The fixed requirement is at least 9 improved paths and positive "
            "aggregate improvement."
        )
    elif route in {"resource_projection_failed", "resource_stopped"}:
        stop = _read_json(run_dir / "resource_stop.json")
        failed = stop["failed_admission"]
        decisive_evidence = (
            "The exact failed resource inequality is retained in "
            f"`resource_stop.json`: kind `{failed['kind']}`, reason "
            f"`{failed['reason']}`, quantum lhs "
            f"`{float(failed['quantum_inequality_lhs']):.17g}`, and active cap "
            f"`{float(failed['maximum_active_seconds']):.17g}`."
        )
    else:
        decisive_evidence = (
            "The exact integrity exception and traceback are retained in "
            "`failure.json`; no broader numerical or learner result is implied."
        )
    resource_stop_reference = (
        "The failed inequality is bound in `resource_stop.json`; all resource "
        "events and the failure-terminalization receipt are in `resource_ledger.json`."
        if route in {"resource_projection_failed", "resource_stopped"}
        else "All admissions, events, and the failure-terminalization receipt are "
        "bound in `resource_ledger.json`."
    )
    data_access = (
        "The entire ARFF was read only for its SHA-256 authority; content parsing "
        "stopped immediately after row 59,999 and never requested a terminal row."
        if (run_dir / "data_roles.json").is_file()
        else "The ARFF byte hash was bound, but no saved data-role content stage completed."
    )
    return f"""# Partial K=128 approximate-candidate Eulerian Jacobi pilot

## Research mode and decision

Research mode: **exploratory**. Decision question: {config['decision']}.
This terminal route is `{failure['route']}`; no full-scale job was launched.

## Source, configuration, and evidence roles

Git revision: `{bindings['git']['revision']}`. Protected-source inventory digest:
`{_semantic_sha256(bindings['historical_source_inventory'])}`. Scientific-config
digest: `{_semantic_sha256({name: value for name, value in config.items() if name != 'execution_authority'})}`.
The exact path-ID/evidence-role allocation is in `config.json`. {data_access}
Canonical run authority is recorded in `command.txt`.

## Exact scoped failure and retained evidence

Error type: `{failure['error_type']}`. Exact proposition: {failure['exact_failure_proposition']}.
Observed error: `{failure['error']}`.

Decisive route evidence: {decisive_evidence}

Completed stages before terminal sealing: {completed_text}.
Retained authority files: {retained_text}. Durable cohort NPZ count: {cohort_count};
retained PNG count: {png_count}. Scientific objective result available:
`{objective_available}`.

## Resource accounting

Active seconds: `{float(ledger['active_seconds']):.6f}` of
`{float(ledger['budget']['max_active_seconds']):.6f}`. Peak storage:
`{int(ledger['peak_storage_bytes'])}` bytes of
`{int(ledger['budget']['max_storage_bytes'])}`. Peak CUDA fraction:
`{float(ledger['peak_cuda_fraction']):.9g}` of
`{float(ledger['budget']['max_cuda_fraction']):.9g}`. {resource_stop_reference}

## Outcome-to-action and claim boundary

Required next action: {_failure_action(str(failure['route']))}.
Claim boundary: {failure['claim_boundary']}. This route is not a scientific
negative for the learner or for Eulerian/Jacobi generation, and it authorizes no
automatic full-scale run.

## Exact commands

Run command:

```text
{run_command}
```

Read-only verification command:

```text
{verify_command}
```

## Deliberate omissions

No terminal-test rows, certified full reverse population, second training seed,
confirmatory analysis, or full-scale population were opened or run. Any stage not
listed as completed is deliberately incomplete; conclusions that require it are
unavailable.
"""


def _failure_experiment_note(run_dir: Path, failure: Mapping[str, Any]) -> str:
    config = _read_json(run_dir / "config.json")
    ledger = _read_json(run_dir / "resource_ledger.json")
    return f"""# Partial experiment note

## Mode, decision, and route

Mode: exploratory. Decision: {config['decision']}. Terminal route:
`{failure['route']}`. Exact failure proposition:
{failure['exact_failure_proposition']}.

## Evidence and resources

All completed cohorts and failed task images remain in the run tree. Active time:
`{float(ledger['active_seconds']):.6f}` seconds; peak persisted storage:
`{int(ledger['peak_storage_bytes'])}` bytes; peak CUDA fraction:
`{float(ledger['peak_cuda_fraction']):.9g}`. See `REPORT.md`, `failure.json`,
`stage_ledger.json`, and `resource_ledger.json` for the replayable authority.

## Claim boundary and next action

{failure['claim_boundary']}. Required action: {_failure_action(str(failure['route']))}.
No automatic full-scale launch occurred. Terminal-test, certified full-reverse,
across-seed, confirmatory, and scale-up claims remain deliberately omitted.

## Exact run authority

`{(run_dir / 'command.txt').read_text(encoding='utf-8').strip()}`
"""


def _finalize_failure(
    run_dir: Path,
    error: BaseException,
    governor: ResourceGovernor,
) -> dict[str, Any]:
    route = _failure_route(error)
    ledger_before_terminal = _read_json(run_dir / "resource_ledger.json")
    original_failed_admission = (
        copy.deepcopy(ledger_before_terminal.get("last_admission"))
        if route in {"resource_projection_failed", "resource_stopped"}
        else None
    )
    terminal_receipt = governor.admit(
        "failure_terminalization",
        predicted_seconds=FAILURE_TERMINALIZATION_SECONDS,
        predicted_bytes=FAILURE_TERMINALIZATION_BYTES,
        terminalization=True,
    )
    propositions = {
        "candidate_health_failed": (
            "the fixed 512-lane shared-v2-Philox candidate numerical execution/integrity criterion did not pass"
        ),
        "oracle_control_failed": (
            "the fixed ten-path oracle/null positive-control execution/integrity criterion did not pass"
        ),
        "resource_projection_failed": "a required major-stage resource projection did not fit",
        "resource_stopped": (
            "a priced resource quantum did not fit or exceeded its cap after completion"
        ),
        "integrity_failed": "an execution or artifact-integrity requirement failed",
    }
    objective_available = int(
        (run_dir / "populations/raw_populations.npz").is_file()
        and (run_dir / "evaluation/metrics.json").is_file()
    )
    claim_boundary = (
        "saved direct and evaluator artifacts may be interpreted only at their exact exploratory scope; "
        "this resource/integrity terminal route does not supply a budget-valid completed human claim"
        if objective_available
        else "no learned-controller or generator scientific conclusion is available from this route"
    )
    failure = {
        "schema": VERSION + "-failure",
        "route": route,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "scientific_objective_result_available": objective_available,
        "exact_failure_proposition": propositions[route],
        "claim_boundary": claim_boundary,
        "full_scale_auto_launched": 0,
        "failure_terminalization_admitted": int(terminal_receipt["passed"]),
        "failure_terminalization_reserve_seconds": float(
            terminal_receipt["reserve_remaining_seconds"]
        ),
        "at": _utc_now(),
    }
    _write_json(run_dir / "failure.json", failure)
    if route in {"resource_projection_failed", "resource_stopped"}:
        _require(
            isinstance(original_failed_admission, dict)
            and int(original_failed_admission.get("passed", 1)) == 0,
            "resource failure lacks its original failed admission",
        )
        ledger = _read_json(run_dir / "resource_ledger.json")
        _write_json(
            run_dir / "resource_stop.json",
            {
                "schema": VERSION + "-resource-stop",
                "route": route,
                "error": str(error),
                "failed_admission": original_failed_admission,
                "active_seconds": ledger.get("active_seconds"),
                "tree_bytes": _directory_bytes(run_dir),
                "completed_artifacts_retained": 1,
            },
        )
    _status(run_dir, route, error=str(error))
    _append_stage(run_dir, "terminal_failure", "complete", route=route)
    _write_text(run_dir / "REPORT.md", _failure_report(run_dir, failure))
    _write_text(
        run_dir / "experiment_note.md",
        _failure_experiment_note(run_dir, failure),
    )
    # Price through a provisional seal, then bind the final ledger into the
    # regenerated reports and final manifest.  Terminal mode consumes the
    # reserved shutdown budget and never starts further scientific work.
    _seal_manifest(run_dir)
    governor.complete(
        "failure_terminalization", synchronize=False, terminalization=True
    )
    if route in {"resource_projection_failed", "resource_stopped"}:
        ledger = _read_json(run_dir / "resource_ledger.json")
        _write_json(
            run_dir / "resource_stop.json",
            {
                "schema": VERSION + "-resource-stop",
                "route": route,
                "error": str(error),
                "failed_admission": original_failed_admission,
                "active_seconds": ledger.get("active_seconds"),
                "tree_bytes": _directory_bytes(run_dir),
                "completed_artifacts_retained": 1,
            },
        )
    _write_text(run_dir / "REPORT.md", _failure_report(run_dir, failure))
    _write_text(
        run_dir / "experiment_note.md",
        _failure_experiment_note(run_dir, failure),
    )
    _seal_manifest(run_dir)
    return verify_run(run_dir)


def run_production(args: argparse.Namespace) -> int:
    repository_root = Path(__file__).resolve().parents[1]
    run_dir: Path | None = None
    governor: ResourceGovernor | None = None
    try:
        run_dir, governor, firewall = initialize_run(
            repository_root,
            Path(args.arff),
            Path(args.ddpm_run_dir),
            Path(args.run_dir),
            device=str(args.device),
            approval_id=str(args.approval_id),
            maximum_active_seconds=float(args.max_active_seconds),
            maximum_storage_mib=float(args.max_storage_mib),
            maximum_cuda_fraction=float(args.max_cuda_fraction),
        )
        governor.admit(
            "data_roles_write",
            predicted_seconds=30.0,
            predicted_bytes=16 * 1024 * 1024,
        )
        (
            train_u8,
            train_y,
            validation_u8,
            validation_y,
            arff_access,
        ) = _load_train_validation_mnist_strict(Path(args.arff))
        roles = prepare_data_roles(
            run_dir,
            train_u8,
            train_y,
            validation_u8,
            validation_y,
            arff_access=arff_access,
        )
        _append_stage(run_dir, "data_roles", "complete")
        governor.complete("data_roles_write", synchronize=False)
        runtime = prepare_runtime_stage(run_dir, torch.device(args.device), governor)
        run_candidate_audit(run_dir, roles["train_mixed_masses"], runtime, governor)
        run_resource_smoke(
            run_dir,
            roles["train_mixed_masses"],
            roles["train_labels"],
            runtime,
            governor,
        )
        oracle_null, oracle, _ = run_oracle_control(
            run_dir,
            roles["validation_mixed_masses"],
            roles["validation_labels"],
            roles["validation_arff_indices"],
            runtime,
            governor,
        )
        train_records, validation_records = build_forward_record_caches(
            run_dir,
            roles["train_mixed_masses"],
            roles["train_labels"],
            roles["validation_mixed_masses"],
            roles["validation_labels"],
            runtime,
            governor,
        )
        model, _ = train_candidate_model(
            run_dir, train_records, validation_records, runtime, governor
        )
        populations = sample_objective_populations(
            run_dir,
            roles["validation_mixed_masses"],
            roles["validation_labels"],
            model,
            runtime,
            governor,
            oracle_null,
            oracle,
        )
        seal_populations(run_dir, firewall, governor)
        evaluate_sealed_populations(run_dir, populations, firewall, governor)
        _admit_major_stage(run_dir, governor, "post_evaluator_finalization")
        receipt = _finish_priced_terminalization(
            run_dir, governor, "post_evaluator_finalization"
        )
    except BaseException as error:
        if run_dir is None or not run_dir.is_dir():
            raise
        _require(governor is not None, "resource governor is unavailable for failure sealing")
        receipt = _finalize_failure(run_dir, error, governor)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded K=128 approximate-candidate Eulerian Jacobi objective pilot"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the one governed objective pilot")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--arff", required=True)
    run.add_argument("--ddpm-run-dir", required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--approval-id", required=True)
    run.add_argument("--max-active-seconds", type=float, default=10_800.0)
    run.add_argument("--max-storage-mib", type=float, default=2_048.0)
    run.add_argument("--max-cuda-fraction", type=float, default=0.75)

    review = subparsers.add_parser("record-review", help="record the fixed 40-row blind review")
    review.add_argument("--run-dir", required=True)
    review.add_argument("--answers", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--confirm-manual-review", action="store_true")

    verify = subparsers.add_parser("verify", help="verify a run tree without mutation or compute")
    verify.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_production(args)
    if args.command == "record-review":
        run_dir = Path(args.run_dir).resolve()
        review_started = time.monotonic()
        governor: ResourceGovernor | None = None
        try:
            governor = _rehydrate_resource_governor(
                run_dir, started_at=review_started
            )
            outcome = record_review(
                run_dir,
                Path(args.answers),
                str(args.reviewer),
                bool(args.confirm_manual_review),
                resource_governor=governor,
            )
            receipt = verify_run(run_dir)
            result = {**receipt, "outcome_route": outcome["route"]}
        except BaseException as error:
            if not run_dir.is_dir():
                raise
            if governor is None:
                governor = _rehydrate_resource_governor(
                    run_dir, started_at=review_started
                )
            result = _finalize_failure(run_dir, error, governor)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_run(Path(args.run_dir)), sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
