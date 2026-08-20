from __future__ import annotations

"""Frozen DDPM-to-Eulerian diversity pilot.

The module is an additive exploratory runner.  It never trains or selects generated
samples.  ``smoke`` is synthetic and CPU-safe; ``run`` is the separately approved
40-path production lifecycle; ``verify`` is read-only; ``finalize-review`` only
scores an already sealed blind review.
"""

import argparse
import csv
import dataclasses
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from mnist import diag_d0_eulerian_edge_flux_replay as v3
from mnist.conditioned_diffusion import SmallMnistCNN
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    DirectFluxUNet,
    _sample_source_batch_torch,
    eulerian_flux_step_torch,
    flux_divergence_torch,
    free_drift_flux_torch,
    natural_horizon,
    poisson_flux_from_velocity_torch,
    step_component_rms_torch,
)
from mnist.mnist_generation_benchmark import (
    compute_generation_metrics,
    evaluate_generated_labels,
    exact_duplicate_metrics,
    read_mnist_arff_slice,
    sha256_file,
    within_class_nn_diversity,
    write_contact_sheet,
)
from mnist.pixel_ddpm import ddpm_step_from_epsilon, make_linear_ddpm_schedule


VERSION = "ddpm-eulerian-diversity-pilot-v1"
RESEARCH_MODE = "exploratory"

PATH_COUNT = 40
PATHS_PER_CLASS = 4
PATH_PREFIX = "d2e-v1-"
OUTER_STEPS = 256
ANCHORS = (0, 64, 128, 192, 256)
NATIVE_ANCHORS = (0, 250, 500, 750, 1000)
DECISION_HORIZONS = (64, 128, 256)
ROWS = ("null", "teacher", "historical", "ddpm_eulerian", "native_ddpm")
EULERIAN_ROWS = ROWS[:-1]

INVENTORY_SEED = 0xE1600001
SOURCE_SEED_BASE = 0xE1601000
DDPM_LATENT_SEED_BASE = 0xE1602000
EULERIAN_EDGE_NOISE_ROOT = 0xE1603001
NATIVE_DDPM_REVERSE_SEED_BASE = 0xE1604000
REVIEW_SEED = 0xE1605001
SMOKE_SEED = 0xE160F001

MASS_SCALE_NUMERATOR = 25_471
MASS_SCALE_DENOMINATOR = 255
MASS_SCALE = MASS_SCALE_NUMERATOR / MASS_SCALE_DENOMINATOR
MASS_SCALE_HEX = "0x1.8f8b8b8b8b8b9p+6"
TRAIN_START, TRAIN_STOP = 0, 55_000
VALIDATION_START, VALIDATION_STOP = 55_000, 60_000

MNIST_ARFF_BYTES = 127_888_265
MNIST_ARFF_SHA256 = "418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b"
LEGACY_CHECKPOINT_BYTES = v3.LEGACY_CHECKPOINT_BYTES
LEGACY_CHECKPOINT_SHA256 = "8be77d1701887522f86099673431a928ad7dd2d350a06f7a94ade5c30a658cc3"
DDPM_CHECKPOINT_SHA256 = "5f4065da8753ad5611ec4efd61b6d13082ce3c9cccaa62258f8019118e95dfc8"
EVALUATOR_SHA256 = "3d31d42a14fee0ecc72adc1644718a037cc48e649948407da6c272b981840c92"
EVALUATOR_SELECTION_SHA256 = "e6cd9e49ca61237d3a10e9ad2fe0ad09f7a33ea22911fdd73fd99f3a07e4c668"

MAX_WALL_SECONDS = 3600.0
MAX_ACCELERATOR_SECONDS = 1800.0
MAX_CUDA_FRACTION = 0.50
MAX_STORAGE_MIB = 256.0
TERMINAL_RESERVE_SECONDS = 60.0
MAX_QUANTUM_SECONDS = 60.0

TEACHER_IMPROVED_MINIMUM = 36
TEACHER_SHORT_RELATIVE_L2_MAXIMUM = 0.80
TEACHER_FINAL_RELATIVE_L2_MAXIMUM = 0.20
TEACHER_CLASSIFIER_ACCURACY_MINIMUM = 0.80
CANDIDATE_HUMAN_RECOGNIZABILITY_MINIMUM = 0.90
CANDIDATE_HUMAN_AGREEMENT_MINIMUM = 0.80
CANDIDATE_DIVERSITY_REAL_RATIO_MINIMUM = 0.25
CANDIDATE_DIVERSITY_HISTORICAL_RATIO_MINIMUM = 2.0
NATIVE_CLASSIFIER_ACCURACY_MINIMUM = 0.80
NATIVE_DIVERSITY_RATIO_MINIMUM = 0.25
POISSON_RESIDUAL_MAXIMUM = 2e-4
MASS_ERROR_MAXIMUM = 2e-6

STAGE_ORDER = (
    "binding_preflight",
    "cpu_smoke_replay",
    "inventory_and_start_seal",
    "null_population",
    "teacher_population",
    "historical_population",
    "ddpm_eulerian_population",
    "native_ddpm_population",
    "population_seal",
    "machine_scoring",
    "render_and_review_bundle",
    "awaiting_human_review",
    "human_review_terminalization",
)

_CHARGED_TERMINAL_PRECHECKS: set[Path] = set()

SCIENTIFIC_ROUTES = (
    "adapter_positive_freeze_replication",
    "native_ddpm_control_invalid",
    "adapter_fidelity_only_major_pivot_or_stop",
    "adapter_early_joint_horizon_replication",
    "adapter_diverse_not_faithful_major_pivot_or_stop",
    "composition_mode_loss_theory_bridge_or_stop",
    "off_policy_bridge_on_policy_or_stop",
    "historical_early_horizon_replication",
    "learned_eulerian_negative_stop_or_major_pivot",
    "unclassified_stop_redesign",
)


class DiversityPilotError(RuntimeError):
    pass


class IntegrityFailure(DiversityPilotError):
    pass


class ResourceStop(DiversityPilotError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityFailure(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        _require(math.isfinite(value), "nonfinite JSON value")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _scientific_digest(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(_hash_array(np.asarray(arrays[key])).encode("ascii"))
    digest.update(_canonical_json_bytes(metadata))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    v3._write_json(path, _jsonable(value))


def _read_json(path: Path) -> dict[str, Any]:
    return v3._read_json(path)


def _is_production_config(config: Mapping[str, Any]) -> bool:
    """Validate the frozen production authority; tests may monkeypatch this seam."""
    _require(all(key in config for key in ("created_at", "command", "argv", "input_paths")),
             "production configuration provenance is incomplete")
    authority = config.get("execution_authority", {})
    approval = str(authority.get("approval_id", ""))
    _require(authority.get("device") == "cuda:0" and len(approval) >= 12 and "<" not in approval
             and "placeholder" not in approval.lower(), "production execution authority changed")
    _require(config["command"] == subprocess.list2cmdline(config["argv"]), "canonical production command changed")
    return True


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    v3._write_npz(path, **arrays)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    v3._write_csv(path, rows)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResourceBudget:
    max_wall_seconds: float = MAX_WALL_SECONDS
    max_accelerator_seconds: float = MAX_ACCELERATOR_SECONDS
    max_storage_bytes: int = int(MAX_STORAGE_MIB * 1024**2)
    max_cuda_fraction: float = MAX_CUDA_FRACTION
    reserve_seconds: float = TERMINAL_RESERVE_SECONDS
    maximum_quantum_seconds: float = MAX_QUANTUM_SECONDS

    def __post_init__(self) -> None:
        if not (0.0 < self.max_wall_seconds <= MAX_WALL_SECONDS):
            raise ValueError("wall-time cap exceeds frozen maximum")
        if not (0.0 < self.max_accelerator_seconds <= MAX_ACCELERATOR_SECONDS):
            raise ValueError("accelerator-time cap exceeds frozen maximum")
        if not (0 < self.max_storage_bytes <= int(MAX_STORAGE_MIB * 1024**2)):
            raise ValueError("storage cap exceeds frozen maximum")
        if not (0.0 < self.max_cuda_fraction <= MAX_CUDA_FRACTION):
            raise ValueError("CUDA fraction cap exceeds frozen maximum")
        if not (0.0 <= self.reserve_seconds < self.max_wall_seconds):
            raise ValueError("terminal reserve is invalid")


@dataclass(frozen=True)
class ControllerStep:
    conditioning_flux: Tensor
    telemetry: Mapping[str, Tensor | float | int]


class ControllerProvider(Protocol):
    def __call__(
        self,
        masses: Tensor,
        labels: Tensor,
        remaining_time: Tensor,
        path_ids: Sequence[str],
        source_masses: Tensor,
    ) -> ControllerStep: ...


@dataclass
class EulerianRowResult:
    row: str
    anchors: np.ndarray
    anchor_steps: np.ndarray
    path_ids: np.ndarray
    requested_labels: np.ndarray
    telemetry: list[dict[str, Any]]
    crn_key_hashes: list[str]
    scientific_digest: str
    provider_anchor_payload: dict[str, np.ndarray] | None = None

    @property
    def endpoints(self) -> np.ndarray:
        return self.anchors[-1]


@dataclass
class NativeDDPMResult:
    model_anchors: np.ndarray
    reverse_steps: np.ndarray
    path_ids: np.ndarray
    requested_labels: np.ndarray
    reverse_seeds: np.ndarray
    latent_bank_sha256: str
    scientific_digest: str
    telemetry: list[dict[str, Any]]


class NullControllerProvider:
    def __init__(self, config: DirectFluxMNISTConfig) -> None:
        self.grid_size = int(config.grid_size)

    def __call__(self, masses: Tensor, labels: Tensor, remaining_time: Tensor,
                 path_ids: Sequence[str], source_masses: Tensor) -> ControllerStep:
        del labels, remaining_time, path_ids, source_masses
        flux = torch.zeros((len(masses), 2, self.grid_size, self.grid_size), dtype=masses.dtype, device=masses.device)
        return ControllerStep(flux, {"controller_kind": 0})


class TeacherControllerProvider:
    def __init__(self, config: DirectFluxMNISTConfig, targets: np.ndarray | Tensor, path_ids: Sequence[str]) -> None:
        target = torch.as_tensor(targets, dtype=torch.float32)
        _require(target.shape == (len(path_ids), int(config.grid_size) ** 2), "teacher target shape mismatch")
        self.config = config
        self.targets = target.clone()
        self.path_ids = tuple(str(value) for value in path_ids)
        self.horizon = float(natural_horizon(config))

    def __call__(self, masses: Tensor, labels: Tensor, remaining_time: Tensor,
                 path_ids: Sequence[str], source_masses: Tensor) -> ControllerStep:
        del labels, source_masses
        _require(tuple(path_ids) == self.path_ids, "teacher path order changed")
        targets = self.targets.to(device=masses.device, dtype=masses.dtype)
        minimum = float(self.config.min_tau_fraction) * self.horizon
        denom = torch.clamp(remaining_time, min=minimum).reshape(-1, 1)
        velocity = (targets - masses) / denom
        velocity = velocity - velocity.mean(dim=1, keepdim=True)
        controller, residual = _adapter_module().velocity_to_periodic_controller_flux(
            velocity, masses, self.config, free_weight=float(self.config.free_weight)
        )
        return ControllerStep(controller, {"desired_velocity_rms": velocity.square().mean(dim=1).sqrt(), "poisson_divergence_residual": residual})


class HistoricalControllerProvider:
    def __init__(self, model: DirectFluxUNet) -> None:
        self.model = model
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)
        self.model_state_sha256 = (
            v3._model_state_semantic_digest(self.model)
            if isinstance(self.model, torch.nn.Module)
            else None
        )

    @torch.no_grad()
    def __call__(self, masses: Tensor, labels: Tensor, remaining_time: Tensor,
                 path_ids: Sequence[str], source_masses: Tensor) -> ControllerStep:
        del path_ids
        context = torch.amp.autocast(device_type="cuda", enabled=True) if masses.device.type == "cuda" else nullcontext()
        with context:
            flux = self.model.predict_flux(remaining_time, masses, labels, source_masses=source_masses)
        return ControllerStep(flux.float(), {"controller_kind": 2})


class DDPMEulerianControllerProvider:
    def __init__(self, adapter: Any, latent_z: np.ndarray | Tensor, path_ids: Sequence[str]) -> None:
        latent = torch.as_tensor(latent_z, dtype=torch.float32)
        _require(latent.shape == (len(path_ids), 1, 28, 28), "DDPM latent shape mismatch")
        self.adapter = adapter
        self.latent_z = latent.clone()
        self.path_ids = tuple(str(value) for value in path_ids)

    def __call__(self, masses: Tensor, labels: Tensor, remaining_time: Tensor,
                 path_ids: Sequence[str], source_masses: Tensor) -> ControllerStep:
        del source_masses
        _require(tuple(path_ids) == self.path_ids, "adapter path order changed")
        result = self.adapter.predict(masses, labels, remaining_time, self.latent_z.to(masses.device, masses.dtype))
        telemetry = {field.name: getattr(result, field.name) for field in dataclasses.fields(result) if field.name != "conditioning_flux"}
        telemetry["poisson_divergence_residual"] = result.divergence_residual_linf
        predicted = result.predicted_mass
        telemetry.update({"predicted_mass_entropy": -(predicted * predicted.clamp_min(1e-30).log()).sum(dim=1),
            "predicted_mass_maximum": predicted.amax(dim=1), "predicted_mass_distance_from_state": (predicted - masses).square().mean(dim=1).sqrt(),
            "desired_velocity_rms": result.desired_velocity.square().mean(dim=1).sqrt(),
            "controller_flux_rms": result.conditioning_flux.square().mean(dim=(1, 2, 3)).sqrt()})
        diversity = torch.empty(len(predicted), device=predicted.device, dtype=predicted.dtype)
        for digit in torch.unique(labels).tolist():
            indices = torch.nonzero(labels == int(digit), as_tuple=False).flatten(); group = predicted[indices]
            if len(indices) < 2: diversity[indices] = 0.0
            else:
                distance = (group[:, None] - group[None]).square().mean(dim=2); distance.fill_diagonal_(float("inf")); diversity[indices] = distance.amin(dim=1)
        telemetry["predicted_mass_within_class_nn_mse"] = diversity
        return ControllerStep(result.conditioning_flux, telemetry)


def _divergence_residual(flux: Tensor, velocity: Tensor, grid_size: int) -> Tensor:
    from mnist.eulerian_flux_mnist import flux_divergence_torch

    _require(flux.shape[-2:] == (grid_size, grid_size), "flux grid size changed")
    residual = flux_divergence_torch(flux).reshape_as(velocity) - velocity
    return residual.abs().amax(dim=1)


def derive_edge_noise_seed(path_id: str, outer_step: int, attempted_substeps: int, sub_index: int,
                           *, root: int = EULERIAN_EDGE_NOISE_ROOT) -> int:
    payload = f"edge-noise-v1|{int(root)}|{path_id}|{int(outer_step)}|{int(attempted_substeps)}|{int(sub_index)}"
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big", signed=False)


def standard_normal_flat_for_paths(path_ids: Sequence[str], outer_step: int, attempted_substeps: int,
                                   sub_index: int, *, edge_count: int = 2 * 28 * 28,
                                   root: int = EULERIAN_EDGE_NOISE_ROOT,
                                   device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> Tensor:
    values = [
        torch.randn((edge_count,), generator=torch.Generator(device="cpu").manual_seed(derive_edge_noise_seed(path_id, outer_step, attempted_substeps, sub_index, root=root)), dtype=torch.float32)
        for path_id in path_ids
    ]
    return torch.stack(values).to(device=torch.device(device), dtype=dtype)


def build_path_inventory() -> dict[str, np.ndarray]:
    labels = np.repeat(np.arange(10, dtype=np.int64), PATHS_PER_CLASS)
    within = np.tile(np.arange(PATHS_PER_CLASS, dtype=np.int64), 10)
    path_ids = np.asarray([f"{PATH_PREFIX}{index:03d}" for index in range(PATH_COUNT)], dtype=np.str_)
    return {
        "path_ids": path_ids,
        "requested_labels": labels,
        "within_class_index": within,
        "source_seeds": np.arange(SOURCE_SEED_BASE, SOURCE_SEED_BASE + PATH_COUNT, dtype=np.uint64),
        "ddpm_latent_seeds": np.arange(DDPM_LATENT_SEED_BASE, DDPM_LATENT_SEED_BASE + PATH_COUNT, dtype=np.uint64),
        "native_reverse_seeds": np.arange(NATIVE_DDPM_REVERSE_SEED_BASE, NATIVE_DDPM_REVERSE_SEED_BASE + PATH_COUNT, dtype=np.uint64),
        "retained": np.ones(PATH_COUNT, dtype=np.int64),
    }


def _validate_inventory(roles: Mapping[str, np.ndarray]) -> None:
    expected = build_path_inventory()
    _require(set(roles) == set(expected), "inventory field set changed")
    for key, value in expected.items():
        _require(np.array_equal(np.asarray(roles[key]), value), f"inventory authority changed: {key}")


def _path_inventory() -> dict[str, np.ndarray]:
    return build_path_inventory()


def _adapter_module() -> Any:
    try:
        return importlib.import_module("mnist.ddpm_eulerian_adapter")
    except ModuleNotFoundError as error:
        raise IntegrityFailure("DDPM-to-Eulerian adapter module is unavailable") from error


def read_mnist_development_prefix(arff_path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return v3.read_mnist_development_prefix(arff_path)


def mass_to_uint8(masses: np.ndarray) -> np.ndarray:
    return v3.mass_to_uint8(masses)


def build_start_bank(config: DirectFluxMNISTConfig,
                     inventory: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
    roles = build_path_inventory() if inventory is None else inventory
    _validate_inventory(roles)
    seeds = np.asarray(roles["source_seeds"], dtype=np.uint64)
    _require(seeds.shape == (PATH_COUNT,), "start-bank seed inventory changed")
    rows: list[np.ndarray] = []
    for seed in seeds.tolist():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            sampled = _sample_source_batch_torch(1, config, device=torch.device("cpu"), dtype=torch.float32)
        rows.append(sampled.masses.detach().cpu().numpy()[0].astype(np.float32, copy=False))
    result = np.stack(rows).astype(np.float32, copy=False)
    _validate_masses(result, PATH_COUNT, "start bank")
    _require(len({_hash_array(row) for row in result}) == PATH_COUNT, "start bank contains duplicate paths")
    return result


def build_teacher_target_bank(validation_images: np.ndarray, validation_labels: np.ndarray,
                              inventory: Mapping[str, np.ndarray] | None = None,
                              *, mass_floor: float = 1e-8) -> dict[str, np.ndarray]:
    roles = build_path_inventory() if inventory is None else inventory
    _validate_inventory(roles)
    images = np.asarray(validation_images)
    labels = np.asarray(validation_labels, dtype=np.int64)
    _require(images.dtype == np.uint8 and images.shape == (5_000, 28, 28), "teacher source must be validation rows [55000,60000)")
    _require(labels.shape == (5_000,), "validation labels shape changed")
    local = np.concatenate([np.flatnonzero(labels == digit)[:PATHS_PER_CLASS] for digit in range(10)]).astype(np.int64)
    _require(local.shape == (PATH_COUNT,), "teacher target bank is incomplete")
    selected = images[local]
    selected_labels = labels[local]
    _require(np.array_equal(selected_labels, np.asarray(roles["requested_labels"])), "teacher labels do not match inventory")
    flat = selected.reshape(PATH_COUNT, -1).astype(np.float32)
    flat = np.maximum(flat, np.float32(mass_floor))
    masses = (flat / flat.sum(axis=1, keepdims=True)).astype(np.float32)
    _validate_masses(masses, PATH_COUNT, "teacher target bank")
    return {
        "masses": masses,
        "source_images_uint8": selected.copy(),
        "rendered_images_uint8": mass_to_uint8(masses),
        "images_uint8": selected.copy(),
        "path_ids": np.asarray(roles["path_ids"], dtype=np.str_),
        "requested_labels": selected_labels,
        "validation_local_ids": local,
        "arff_global_row_ids": local + VALIDATION_START,
        "role": np.asarray("teacher_only_validation_targets"),
    }


def build_ddpm_latent_bank(inventory: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    roles = build_path_inventory() if inventory is None else inventory
    _validate_inventory(roles)
    seeds = np.asarray(roles["ddpm_latent_seeds"], dtype=np.uint64)
    rows = [torch.randn((1, 28, 28), generator=torch.Generator().manual_seed(int(seed))).numpy() for seed in seeds.tolist()]
    z = np.stack(rows).astype(np.float32)
    _require(z.shape == (PATH_COUNT, 1, 28, 28) and np.isfinite(z).all(), "DDPM latent bank is invalid")
    return {
        "z": z,
        "path_ids": np.asarray(roles["path_ids"], dtype=np.str_),
        "requested_labels": np.asarray(roles["requested_labels"], dtype=np.int64),
        "latent_seeds": seeds,
        "mean": np.asarray(float(z.mean()), dtype=np.float64),
        "standard_deviation": np.asarray(float(z.std()), dtype=np.float64),
        "minimum": np.asarray(float(z.min()), dtype=np.float64),
        "maximum": np.asarray(float(z.max()), dtype=np.float64),
        "z_sha256": np.asarray(_hash_array(z)),
    }


def _validate_masses(value: np.ndarray, count: int, name: str) -> None:
    array = np.asarray(value)
    _require(array.dtype == np.float32 and array.shape == (count, 784), f"{name} shape/dtype changed")
    _require(bool(np.isfinite(array).all()) and float(array.min()) >= 0.0, f"{name} is nonfinite or negative")
    _require(float(np.max(np.abs(array.sum(axis=1, dtype=np.float64) - 1.0))) <= MASS_ERROR_MAXIMUM, f"{name} mass error")


def _provider_summary(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary: dict[str, Any] = {}
    paths: dict[str, np.ndarray] = {}
    for key, item in value.items():
        if isinstance(item, Tensor):
            array = item.detach().float().cpu().numpy()
        else:
            array = np.asarray(item)
        if array.ndim == 0:
            summary[key] = _jsonable(array.item())
        elif array.shape[0] == PATH_COUNT or array.shape[0] <= PATH_COUNT:
            paths[key] = array.copy()
            summary[f"{key}_mean"] = float(np.asarray(array, dtype=np.float64).mean())
            summary[f"{key}_maximum"] = float(np.asarray(array, dtype=np.float64).max())
        else:
            summary[f"{key}_mean"] = float(np.asarray(array, dtype=np.float64).mean())
    return summary, paths


@torch.no_grad()
def run_eulerian_row(starts: np.ndarray, labels: np.ndarray, path_ids: Sequence[str],
                     config: DirectFluxMNISTConfig, provider: ControllerProvider, *, row: str,
                     device: str | torch.device, num_steps: int = OUTER_STEPS,
                     schedule_steps: int | None = None, anchors: Sequence[int] = ANCHORS,
                     edge_noise_root: int = EULERIAN_EDGE_NOISE_ROOT,
                     outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None) -> EulerianRowResult:
    _require(row in EULERIAN_ROWS, "unknown Eulerian row")
    starts_np = np.asarray(starts)
    labels_np = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(path_ids, dtype=np.str_)
    count = len(ids)
    _require(count == len(starts_np) == len(labels_np) and len(set(ids.tolist())) == count, "row identity mismatch")
    _validate_masses(starts_np, count, "row starts")
    step_with_noise = _adapter_module().eulerian_flux_step_with_standard_normal_torch
    steps = int(num_steps)
    schedule_count = steps if schedule_steps is None else int(schedule_steps)
    anchor_tuple = tuple(int(value) for value in anchors)
    _require(steps > 0 and schedule_count >= steps, "row schedule is invalid")
    _require(anchor_tuple[0] == 0 and anchor_tuple[-1] == steps and tuple(sorted(set(anchor_tuple))) == anchor_tuple, "row anchors are invalid")
    target_device = torch.device(device)
    states = torch.as_tensor(starts_np, device=target_device).clone()
    sources = states.clone()
    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=target_device)
    horizon = float(natural_horizon(config))
    outer_dt = horizon / float(schedule_count)
    saved = [states.cpu().numpy().copy()]
    saved_steps = [0]
    telemetry: list[dict[str, Any]] = []
    key_hashes: list[str] = []
    provider_anchors: dict[str, list[np.ndarray]] = {}
    for outer in range(steps):
        began = time.perf_counter()
        before = states.clone()
        accepted: tuple[int, int, int, list[dict[str, Any]], Mapping[str, np.ndarray]] | None = None
        attempt_rows: list[dict[str, Any]] = []
        for attempted in ((1, 2, 4) if bool(config.adaptive_sampling) else (1,)):
            if attempted > int(config.max_substeps):
                continue
            local = before.clone()
            clipped = proposed = 0
            component_rows: list[dict[str, Any]] = []
            last_paths: Mapping[str, np.ndarray] = {}
            sub_dt = outer_dt / attempted
            for sub_index in range(attempted):
                tau_value = max(horizon - outer * outer_dt - sub_index * sub_dt, 0.0)
                tau = torch.full((count,), tau_value, dtype=local.dtype, device=target_device)
                step = provider(local, labels_t, tau, ids.tolist(), sources)
                _require(step.conditioning_flux.shape == (count, 2, int(config.grid_size), int(config.grid_size)), "provider flux shape changed")
                if "poisson_divergence_residual" in step.telemetry:
                    residual = torch.as_tensor(step.telemetry["poisson_divergence_residual"])
                    _require(bool(torch.isfinite(residual).all()) and float(residual.max().cpu()) <= POISSON_RESIDUAL_MAXIMUM,
                             f"{row} Poisson divergence residual exceeded the integrity threshold")
                provider_summary, last_paths = _provider_summary(step.telemetry)
                components = step_component_rms_torch(local, step.conditioning_flux, sub_dt, config,
                    free_weight=float(config.free_weight), noise_weight=float(config.noise_weight), learned_weight=1.0)
                divergence = flux_divergence_torch(step.conditioning_flux).reshape(count, -1)
                free_flux = float(config.free_weight) * free_drift_flux_torch(local, config)
                _, controller_clipped, controller_proposed = eulerian_flux_step_torch(
                    local,
                    step.conditioning_flux,
                    sub_dt,
                    config,
                    deterministic=True,
                    free_weight=0.0,
                    noise_weight=0.0,
                    learned_weight=1.0,
                )
                components.update({
                    "controller_flux_rms": float(step.conditioning_flux.float().square().mean().sqrt().cpu()),
                    "total_deterministic_flux_rms": float((step.conditioning_flux + free_flux).float().square().mean().sqrt().cpu()),
                    "controller_divergence_mean": float(divergence.mean().cpu()),
                    "controller_divergence_rms": float(divergence.square().mean().sqrt().cpu()),
                    "controller_increment_clipping_fraction": 0.0 if controller_proposed == 0 else controller_clipped / controller_proposed,
                })
                normals = standard_normal_flat_for_paths(ids.tolist(), outer, attempted, sub_index,
                    edge_count=2 * int(config.grid_size) ** 2, root=edge_noise_root, device=target_device, dtype=local.dtype)
                key_payload = [derive_edge_noise_seed(value, outer, attempted, sub_index, root=edge_noise_root) for value in ids.tolist()]
                key_hashes.append(_sha256_bytes(np.asarray(key_payload, dtype=np.uint64).tobytes()))
                local, clipped_now, proposed_now = step_with_noise(local, step.conditioning_flux, sub_dt, config,
                    deterministic=False, free_weight=float(config.free_weight), noise_weight=float(config.noise_weight),
                    learned_weight=1.0, standard_normal_flat=normals)
                clipped += int(clipped_now)
                proposed += int(proposed_now)
                component_rows.append({**components, **provider_summary})
            fraction = 0.0 if proposed == 0 else clipped / proposed
            attempt_rows.append({"substeps": attempted, "clipped": clipped, "proposed": proposed, "clipping_fraction": fraction})
            if not bool(config.adaptive_sampling) or fraction <= float(config.clip_target) or attempted >= int(config.max_substeps):
                states = local
                accepted = (attempted, clipped, proposed, component_rows, last_paths)
                break
        _require(accepted is not None, "adaptive sampler accepted no attempt")
        accepted_substeps, clipped, proposed, components, last_paths = accepted
        finite = bool(torch.isfinite(states).all())
        minimum = float(states.min().cpu())
        mass_error = float((states.sum(dim=1) - 1.0).abs().max().cpu())
        _require(finite and minimum >= 0.0 and mass_error <= MASS_ERROR_MAXIMUM, f"{row} numerical health failed at step {outer + 1}")
        means = {key: float(np.mean([float(item.get(key, 0.0)) for item in components])) for key in
                 ("learned_step_rms", "free_step_rms", "noise_step_rms")}
        record = {
            "row": row, "completed_step": outer + 1, "remaining_time": max(horizon - outer * outer_dt, 0.0),
            "accepted_substeps": accepted_substeps, "attempts": attempt_rows,
            "accepted_clipped": clipped, "accepted_proposed": proposed,
            "accepted_clipping_fraction": 0.0 if proposed == 0 else clipped / proposed,
            **means, "controller_to_free_ratio": means["learned_step_rms"] / max(means["free_step_rms"], 1e-12),
            "controller_to_noise_ratio": means["learned_step_rms"] / max(means["noise_step_rms"], 1e-12),
            "state_displacement_rms": float((states - before).square().mean().sqrt().cpu()),
            "minimum_mass": minimum, "maximum_mass": float(states.max().cpu()), "maximum_mass_error": mass_error,
            "finite": 1, "elapsed_seconds": time.perf_counter() - began,
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated(target_device)) if target_device.type == "cuda" else 0,
        }
        for key in set().union(*(item.keys() for item in components)):
            if key not in record and all(isinstance(item.get(key), (int, float)) for item in components):
                record[key] = float(np.mean([float(item[key]) for item in components]))
        record.setdefault("poisson_divergence_residual_maximum", 0.0)
        telemetry.append(record)
        if outer + 1 in anchor_tuple:
            saved.append(states.cpu().numpy().copy())
            saved_steps.append(outer + 1)
            for key, value in last_paths.items():
                provider_anchors.setdefault(key, []).append(np.asarray(value))
        if outer_step_callback is not None:
            outer_step_callback({"row": row, "completed_step": outer + 1, "state": states, "telemetry": telemetry,
                                 "path_ids": ids, "requested_labels": labels_np, "crn_key_hashes": key_hashes})
    anchor_array = np.stack(saved).astype(np.float32)
    _require(saved_steps == list(anchor_tuple), "row anchors are incomplete")
    deterministic_telemetry = [{k: v for k, v in item.items() if k not in {"elapsed_seconds", "cuda_allocated_bytes"}} for item in telemetry]
    digest = _scientific_digest({"anchors": anchor_array, "anchor_steps": np.asarray(saved_steps, dtype=np.int64),
                                 "path_ids": ids, "requested_labels": labels_np}, {"row": row, "telemetry": deterministic_telemetry, "crn_key_hashes": key_hashes})
    payload = {key: np.stack(values) for key, values in provider_anchors.items()} or None
    return EulerianRowResult(row, anchor_array, np.asarray(saved_steps, dtype=np.int64), ids, labels_np,
                             telemetry, key_hashes, digest, payload)


class ResourceGovernor:
    def __init__(self, run_dir: str | Path, budget: ResourceBudget, *, device: str | torch.device) -> None:
        self.run_dir, self.budget, self.device = Path(run_dir), budget, torch.device(device)
        self.wall_seconds = self.accelerator_seconds = 0.0
        self.events: list[dict[str, Any]] = []
        self.failed_admission: dict[str, Any] | None = None
        self._open: dict[str, tuple[float, bool]] = {}

    def _cuda(self) -> tuple[int, int, float]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0, 0, 0.0
        allocated = int(torch.cuda.max_memory_allocated(self.device))
        total = int(torch.cuda.get_device_properties(self.device).total_memory)
        return allocated, total, allocated / max(total, 1)

    def write(self) -> None:
        _write_json(self.run_dir / "resource_ledger.json", {"schema": VERSION + "-resource-ledger", "budget": dataclasses.asdict(self.budget),
            "wall_seconds": self.wall_seconds, "accelerator_seconds": self.accelerator_seconds, "events": self.events,
            "failed_admission": self.failed_admission, "open_events": sorted(self._open)})

    def admit(self, kind: str, *, predicted_wall_seconds: float, predicted_accelerator_seconds: float,
              predicted_next_bytes: int, reserve_remaining_seconds: float | None = None,
              terminal_override: bool = False) -> dict[str, Any]:
        _require(kind not in self._open, f"resource event already open: {kind}")
        reserve = self.budget.reserve_seconds if reserve_remaining_seconds is None else float(reserve_remaining_seconds)
        storage = v3._storage_bytes(self.run_dir)
        allocated, total, fraction = self._cuda()
        checks = {"wall": self.wall_seconds + predicted_wall_seconds + reserve <= self.budget.max_wall_seconds,
                  "accelerator": self.accelerator_seconds + predicted_accelerator_seconds <= self.budget.max_accelerator_seconds,
                  "storage": storage + int(predicted_next_bytes) <= self.budget.max_storage_bytes,
                  "cuda": fraction <= self.budget.max_cuda_fraction,
                  "quantum": predicted_wall_seconds <= self.budget.maximum_quantum_seconds or reserve == 0.0}
        if terminal_override:
            _require(kind in {"machine_terminalization", "human_review_terminalization"} and reserve == 0.0,
                     "terminal resource override is invalid")
            checks = {key: True for key in checks}
        receipt = {"event": "admit", "kind": kind, "predicted_wall_seconds": float(predicted_wall_seconds),
                   "predicted_accelerator_seconds": float(predicted_accelerator_seconds), "predicted_next_bytes": int(predicted_next_bytes),
                   "reserve_remaining_seconds": reserve, "storage_bytes_before": storage, "cuda_allocated_bytes": allocated,
                   "cuda_total_bytes": total, "cuda_fraction": fraction, "wall_seconds_before": self.wall_seconds,
                   "accelerator_seconds_before": self.accelerator_seconds, "checks": checks,
                   "terminal_override": int(terminal_override), "passed": int(all(checks.values())), "recorded_at": _utc_now()}
        if not all(checks.values()):
            self.failed_admission = receipt
            self.write()
            raise ResourceStop(f"resource admission failed for {kind}: {checks}")
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        self._open[kind] = (time.perf_counter(), float(predicted_accelerator_seconds) > 0.0)
        self.events.append(receipt); self.write(); return receipt

    def complete(self, kind: str, *, terminal_override: bool = False) -> dict[str, Any]:
        _require(kind in self._open, f"resource event is not open: {kind}")
        began, accelerated = self._open.pop(kind)
        if accelerated and self.device.type == "cuda": torch.cuda.synchronize(self.device)
        elapsed = max(0.0, time.perf_counter() - began)
        self.wall_seconds += elapsed
        if accelerated: self.accelerator_seconds += elapsed
        allocated, total, fraction = self._cuda()
        receipt = {"event": "complete", "kind": kind, "elapsed_seconds": elapsed, "wall_seconds_after": self.wall_seconds,
                   "accelerator_seconds_after": self.accelerator_seconds, "storage_bytes_after": v3._storage_bytes(self.run_dir),
                   "cuda_allocated_bytes": allocated, "cuda_total_bytes": total, "cuda_fraction": fraction,
                   "terminal_override": int(terminal_override), "recorded_at": _utc_now()}
        self.events.append(receipt); self.write()
        checks = {"wall": self.wall_seconds <= self.budget.max_wall_seconds, "accelerator": self.accelerator_seconds <= self.budget.max_accelerator_seconds,
                  "storage": receipt["storage_bytes_after"] <= self.budget.max_storage_bytes, "cuda": fraction <= self.budget.max_cuda_fraction,
                  "quantum": elapsed <= self.budget.maximum_quantum_seconds}
        if not all(checks.values()) and not terminal_override:
            self.failed_admission = {"kind": kind, "phase": "post-completion", "checks": checks, "receipt": receipt, "passed": 0}; self.write()
            raise ResourceStop(f"resource post-completion check failed for {kind}: {checks}")
        return receipt

    def close_open_as_failed(self) -> None:
        for kind, (began, accelerated) in list(self._open.items()):
            elapsed = max(0.0, time.perf_counter() - began); self.wall_seconds += elapsed
            if accelerated: self.accelerator_seconds += elapsed
            self.events.append({"event": "failed-complete", "kind": kind, "elapsed_seconds": elapsed,
                                "wall_seconds_after": self.wall_seconds, "accelerator_seconds_after": self.accelerator_seconds, "recorded_at": _utc_now()})
            del self._open[kind]
        self.write()

    @classmethod
    def rehydrate(cls, run_dir: str | Path, *, device: str | torch.device) -> "ResourceGovernor":
        root = Path(run_dir); ledger = _read_json(root / "resource_ledger.json"); governor = cls(root, ResourceBudget(**ledger["budget"]), device=device)
        governor.wall_seconds = float(ledger["wall_seconds"]); governor.accelerator_seconds = float(ledger["accelerator_seconds"])
        governor.events = list(ledger["events"]); governor.failed_admission = ledger.get("failed_admission")
        _require(not ledger.get("open_events"), "cannot rehydrate with open resource events"); return governor


def _verify_resource_ledger(run_dir: Path, config: Mapping[str, Any], status: Mapping[str, Any], *,
                            allow_terminal_open: bool = False) -> dict[str, Any]:
    ledger = _read_json(run_dir / "resource_ledger.json")
    authority = config["execution_authority"]
    expected_budget = {key: authority[key] for key in (
        "max_wall_seconds", "max_accelerator_seconds", "max_storage_bytes",
        "max_cuda_fraction", "reserve_seconds", "maximum_quantum_seconds",
    )}
    _require(ledger.get("schema") == VERSION + "-resource-ledger" and ledger.get("budget") == expected_budget,
             "resource budget authority changed")
    wall = accelerator = 0.0
    opened: dict[str, Mapping[str, Any]] = {}
    interrupted: list[tuple[int, str, Mapping[str, Any]]] = []
    completed_after: list[tuple[int, str]] = []
    historical_overrides: list[tuple[int, str]] = []
    normal_admissions: list[tuple[int, str]] = []
    for event_index, event in enumerate(ledger.get("events", [])):
        kind = str(event.get("kind", "")); event_type = event.get("event")
        if event_type == "admit":
            _require(kind and kind not in opened and event.get("passed") == 1, "resource admission sequence changed")
            _require(all(math.isfinite(float(event[key])) and float(event[key]) >= 0 for key in
                         ("predicted_wall_seconds", "predicted_accelerator_seconds", "predicted_next_bytes",
                          "reserve_remaining_seconds", "storage_bytes_before", "cuda_fraction")),
                     "resource admission contains an invalid value")
            checks = {
                "wall": wall + float(event["predicted_wall_seconds"]) + float(event["reserve_remaining_seconds"]) <= float(expected_budget["max_wall_seconds"]),
                "accelerator": accelerator + float(event["predicted_accelerator_seconds"]) <= float(expected_budget["max_accelerator_seconds"]),
                "storage": int(event["storage_bytes_before"]) + int(event["predicted_next_bytes"]) <= int(expected_budget["max_storage_bytes"]),
                "cuda": float(event["cuda_fraction"]) <= float(expected_budget["max_cuda_fraction"]),
                "quantum": float(event["predicted_wall_seconds"]) <= float(expected_budget["maximum_quantum_seconds"]) or float(event["reserve_remaining_seconds"]) == 0.0,
            }
            if event.get("terminal_override") == 1:
                _require(kind in {"machine_terminalization", "human_review_terminalization"}
                         and float(event["reserve_remaining_seconds"]) == 0.0
                         and event.get("checks") == {key: True for key in checks}, "terminal resource override changed")
                if status["state"] in {"awaiting_human_review", "complete"}:
                    historical_overrides.append((event_index, kind))
                else:
                    _require(status["state"] in {"failed_unsealed", "resource_stopped", "postseal_interrupted"},
                             "terminal resource override changed")
            else:
                _require(event.get("terminal_override", 0) == 0 and event.get("checks") == checks and all(checks.values()), "resource admission arithmetic changed")
                normal_admissions.append((event_index, kind))
            opened[kind] = event
        elif event_type in {"complete", "failed-complete"}:
            _require(kind in opened, "resource completion has no matching admission")
            elapsed = float(event.get("elapsed_seconds", -1.0)); _require(math.isfinite(elapsed) and elapsed >= 0.0, "resource elapsed time is invalid")
            wall += elapsed
            if float(opened[kind]["predicted_accelerator_seconds"]) > 0.0:
                accelerator += elapsed
            _require(math.isclose(float(event["wall_seconds_after"]), wall, rel_tol=0.0, abs_tol=1e-9)
                     and math.isclose(float(event["accelerator_seconds_after"]), accelerator, rel_tol=0.0, abs_tol=1e-9),
                     "resource cumulative accounting changed")
            if event_type == "complete":
                completion_checks = {
                    "wall": wall <= float(expected_budget["max_wall_seconds"]),
                    "accelerator": accelerator <= float(expected_budget["max_accelerator_seconds"]),
                    "storage": int(event.get("storage_bytes_after", -1)) <= int(expected_budget["max_storage_bytes"]),
                    "cuda": math.isfinite(float(event.get("cuda_fraction", -1)))
                            and 0.0 <= float(event.get("cuda_fraction", -1)) <= float(expected_budget["max_cuda_fraction"]),
                    "quantum": elapsed <= float(expected_budget["maximum_quantum_seconds"]),
                }
                if event.get("terminal_override") == 1:
                    _require(opened[kind].get("terminal_override") == 1
                             and kind in {"machine_terminalization", "human_review_terminalization"},
                             "terminal completion override changed")
                    if status["state"] not in {"awaiting_human_review", "complete"}:
                        _require(status["state"] in {"failed_unsealed", "resource_stopped", "postseal_interrupted"},
                                 "terminal completion override changed")
                elif not all(completion_checks.values()):
                    failed_receipt = ledger.get("failed_admission", {})
                    _require(status["state"] == "resource_stopped" and failed_receipt.get("phase") == "post-completion"
                             and failed_receipt.get("receipt") == event and failed_receipt.get("checks") == completion_checks,
                             "successful resource completion exceeded its cap")
            else:
                interrupted.append((event_index, kind, event))
            if event_type == "complete" and event.get("terminal_override", 0) == 0:
                completed_after.append((event_index, kind))
            del opened[kind]
        else:
            raise IntegrityFailure("resource event type changed")
    if allow_terminal_open:
        _require(len(opened) == 1 and set(opened) == set(ledger.get("open_events", []))
                 and next(iter(opened)) in {"machine_terminalization", "human_review_terminalization"},
                 "resource ledger terminal open event changed")
    else:
        _require(not opened and not ledger.get("open_events"), "resource ledger has open events")
    _require(math.isclose(float(ledger.get("wall_seconds", -1)), wall, rel_tol=0.0, abs_tol=1e-9)
             and math.isclose(float(ledger.get("accelerator_seconds", -1)), accelerator, rel_tol=0.0, abs_tol=1e-9),
             "resource totals changed")
    failed = ledger.get("failed_admission")
    if status["state"] == "resource_stopped":
        _require(isinstance(failed, Mapping), "resource-stopped route lacks a failed admission")
        if failed.get("event") == "admit":
            failed_checks = {
                "wall": float(failed["wall_seconds_before"]) + float(failed["predicted_wall_seconds"]) + float(failed["reserve_remaining_seconds"]) <= float(expected_budget["max_wall_seconds"]),
                "accelerator": float(failed["accelerator_seconds_before"]) + float(failed["predicted_accelerator_seconds"]) <= float(expected_budget["max_accelerator_seconds"]),
                "storage": int(failed["storage_bytes_before"]) + int(failed["predicted_next_bytes"]) <= int(expected_budget["max_storage_bytes"]),
                "cuda": float(failed["cuda_fraction"]) <= float(expected_budget["max_cuda_fraction"]),
                "quantum": float(failed["predicted_wall_seconds"]) <= float(expected_budget["maximum_quantum_seconds"]) or float(failed["reserve_remaining_seconds"]) == 0.0,
            }
            _require(failed.get("checks") == failed_checks and failed.get("passed") == 0 and not all(failed_checks.values()), "failed resource admission was changed")
        else:
            receipt = failed.get("receipt", {}); checks = {
                "wall": float(receipt.get("wall_seconds_after", -1)) <= float(expected_budget["max_wall_seconds"]),
                "accelerator": float(receipt.get("accelerator_seconds_after", -1)) <= float(expected_budget["max_accelerator_seconds"]),
                "storage": int(receipt.get("storage_bytes_after", -1)) <= int(expected_budget["max_storage_bytes"]),
                "cuda": float(receipt.get("cuda_fraction", -1)) <= float(expected_budget["max_cuda_fraction"]),
                "quantum": float(receipt.get("elapsed_seconds", -1)) <= float(expected_budget["maximum_quantum_seconds"]),
            }
            _require(failed.get("phase") == "post-completion" and failed.get("checks") == checks
                     and failed.get("passed") == 0 and not all(checks.values()), "failed post-completion resource receipt was changed")
    elif status["state"] in {"awaiting_human_review", "complete"}:
        _require(failed is None and wall <= float(expected_budget["max_wall_seconds"])
                 and accelerator <= float(expected_budget["max_accelerator_seconds"])
                 and v3._storage_bytes(run_dir) <= int(expected_budget["max_storage_bytes"]),
                 "successful route exceeds its resource authority")
    if interrupted:
        if status["state"] in {"awaiting_human_review", "complete"}:
            snapshot_path = run_dir / "failure/resource_snapshot.json"
            _require(snapshot_path.is_file(), "recovered resource history lacks its interruption snapshot")
            captured = _read_json(snapshot_path).get("ledger", {}).get("events", [])
            for index, kind, event in interrupted:
                _require(index < len(captured) and captured[index] == event
                         and any(later > index and later_kind == kind for later, later_kind in completed_after),
                         "interrupted resource event lacks an authenticated successful recovery")
        else:
            _require(status["state"] in {"failed_unsealed", "resource_stopped", "postseal_interrupted"},
                     "failed resource completion appears on an invalid route")
    if historical_overrides:
        _require(status["state"] in {"awaiting_human_review", "complete"} and interrupted,
                 "historical terminal override lacks an interrupted recovery")
        for index, kind in historical_overrides:
            _require(any(prior < index for prior, _, _ in interrupted)
                     and any(later > index and later_kind == kind for later, later_kind in normal_admissions),
                     "historical terminal override lacks a later non-overridden terminal retry")
    return ledger


def _save_failure_evidence(run_dir: Path, error: BaseException, stage: str, governor: ResourceGovernor | None) -> None:
    sealed = (run_dir / "populations/POPULATIONS_SEALED.json").is_file()
    last_valid = sorted((run_dir / "failure").glob("last_valid_*.npz"))
    stage_row = stage.removesuffix("_population")
    preferred = run_dir / f"failure/last_valid_{stage_row}.npz"
    selected = preferred if preferred.exists() else (last_valid[-1] if last_valid else None)
    if selected is not None:
        latest = _npz(selected)
        _write_npz(run_dir / "failure/controller_snapshot.npz", **latest)
    else:
        _write_npz(run_dir / "failure/controller_snapshot.npz", available=np.asarray(0, dtype=np.int64))
    ledger = _read_json(run_dir / "resource_ledger.json") if (run_dir / "resource_ledger.json").exists() else None
    _write_json(run_dir / "failure/resource_snapshot.json", {"schema": VERSION + "-failure-resource-snapshot", "ledger": ledger})
    model_path = run_dir / "telemetry/model_state_identity.json"
    preflight_model = run_dir / "preflight/model_immutability.json"
    if model_path.exists():
        model_snapshot = _read_json(model_path)
    elif preflight_model.exists():
        model_snapshot = {"schema": VERSION + "-failure-model-state-identity", "available": 1,
                          "source": "preflight/model_immutability.json", "receipt": _read_json(preflight_model)}
    else:
        model_snapshot = {"schema": VERSION + "-failure-model-state-identity", "available": 0}
    _write_json(run_dir / "failure/model_state_identity.json", model_snapshot)
    authorities = {}
    for relative in ("config.json", "source_bindings.json", "checkpoint_bindings.json", "inventory/STARTS_SEALED.json",
                     "inventory/start_bank.npz", "inventory/ddpm_latent_bank.npz", "inventory/rng_contract.json"):
        path = run_dir / relative
        if path.is_file():
            authorities[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    _write_json(run_dir / "failure.json", {"schema": VERSION + "-failure", "error_type": type(error).__name__,
        "message": str(error), "failed_stage": stage, "populations_sealed": int(sealed),
        "last_valid_files": [path.relative_to(run_dir).as_posix() for path in last_valid],
        "controller_source": selected.relative_to(run_dir).as_posix() if selected is not None else None,
        "authority_receipts": authorities,
        "resource_ledger_sha256": sha256_file(run_dir / "resource_ledger.json") if ledger is not None else None})
    v3._atomic_bytes(run_dir / "failure/traceback.txt", traceback.format_exc().encode("utf-8"))


def resource_projection(*, smoke_path_seconds: float, path_count: int = PATH_COUNT,
                        native_step_seconds: float = 0.0, stored_bytes_per_path: int = 1_000_000) -> dict[str, Any]:
    projected_eulerian = max(float(smoke_path_seconds), 0.0) * path_count * len(EULERIAN_ROWS)
    projected_native = max(float(native_step_seconds), 0.0) * 1_000
    return {"projected_wall_seconds": projected_eulerian + projected_native + TERMINAL_RESERVE_SECONDS,
            "projected_accelerator_seconds": projected_eulerian + projected_native,
            "projected_storage_bytes": int(stored_bytes_per_path) * path_count,
            "fits_frozen_maxima": int(projected_eulerian + projected_native + TERMINAL_RESERVE_SECONDS <= MAX_WALL_SECONDS
                                      and projected_eulerian + projected_native <= MAX_ACCELERATOR_SECONDS
                                      and int(stored_bytes_per_path) * path_count <= int(MAX_STORAGE_MIB * 1024**2))}


def _durable_step_callback(
    run_dir: Path,
    row: str,
    governor: ResourceGovernor,
    total_steps: int,
    *,
    interval: int | None = None,
    initial_state: np.ndarray | None = None,
    path_ids: Sequence[str] | None = None,
    requested_labels: np.ndarray | None = None,
) -> Callable[[Mapping[str, Any]], None]:
    boundary = int(interval or (25 if row == "native_ddpm" else 8))
    _require(boundary > 0, "durable checkpoint interval must be positive")
    if initial_state is not None:
        initial = np.asarray(initial_state, dtype=np.float32)
        ids = np.asarray(path_ids if path_ids is not None else [f"{row}:{index:03d}" for index in range(len(initial))], dtype=np.str_)
        labels = np.asarray(requested_labels if requested_labels is not None else np.full(len(initial), -1), dtype=np.int64)
        _write_npz(run_dir / f"failure/last_valid_{row}.npz", state=initial, completed_step=np.asarray(0), path_ids=ids, requested_labels=labels,
                   telemetry_tail_json=np.asarray("[]"), crn_key_hashes=np.asarray([], dtype=np.str_), reverse_seeds=np.asarray([], dtype=np.uint64))
        images = _model_to_uint8(initial) if row == "native_ddpm" else mass_to_uint8(initial)
        write_contact_sheet(run_dir / f"failure/last_valid_{row}.png", images, columns=min(10, len(images)), scale=2,
                            captions=[f"{row}:{index:03d}" for index in range(len(images))])
        _write_json(run_dir / "failure/telemetry_tail.json", {"row": row, "completed_step": 0, "tail": [],
            "crn_key_hashes": [], "reverse_seeds": [], "path_ids": ids, "requested_labels": labels})
    current = {"kind": f"{row}_steps_0001_{min(boundary, total_steps):04d}"}
    governor.admit(
        current["kind"],
        predicted_wall_seconds=MAX_QUANTUM_SECONDS,
        predicted_accelerator_seconds=MAX_QUANTUM_SECONDS,
        predicted_next_bytes=1_500_000,
    )

    latest: dict[str, Any] = {}

    def write_latest(value: Mapping[str, Any]) -> None:
        completed = int(value["completed_step"])
        state = value["state"].detach().cpu().numpy().astype(np.float32)
        _write_npz(run_dir / f"failure/last_valid_{row}.npz", state=state, completed_step=np.asarray(completed),
                   path_ids=np.asarray(value.get("path_ids", [f"{row}:{index:03d}" for index in range(len(state))]), dtype=np.str_),
                   requested_labels=np.asarray(value.get("requested_labels", np.full(len(state), -1)), dtype=np.int64),
                   telemetry_tail_json=np.asarray(json.dumps(_jsonable(value["telemetry"][-4:]), sort_keys=True, separators=(",", ":"))),
                   crn_key_hashes=np.asarray(value.get("crn_key_hashes", []), dtype=np.str_),
                   reverse_seeds=np.asarray(value.get("reverse_seeds", []), dtype=np.uint64))
        images = _model_to_uint8(state) if row == "native_ddpm" else mass_to_uint8(state)
        write_contact_sheet(
            run_dir / f"failure/last_valid_{row}.png",
            images,
            columns=min(10, len(images)),
            scale=2,
            captions=[f"{row}:{index:03d}" for index in range(len(images))],
        )
        _write_json(run_dir / "failure/telemetry_tail.json", {"row": row, "completed_step": completed,
            "tail": value["telemetry"][-4:], "crn_key_hashes": value.get("crn_key_hashes", []),
            "reverse_seeds": value.get("reverse_seeds", []), "path_ids": value.get("path_ids", []),
            "requested_labels": value.get("requested_labels", [])})

    def callback(value: Mapping[str, Any]) -> None:
        completed = int(value["completed_step"])
        latest["value"] = value
        if completed % boundary and completed != total_steps:
            return
        write_latest(value)
        governor.complete(current["kind"])
        if completed < total_steps:
            current["kind"] = f"{row}_steps_{completed + 1:04d}_{min(completed + boundary, total_steps):04d}"
            governor.admit(
                current["kind"],
                predicted_wall_seconds=MAX_QUANTUM_SECONDS,
                predicted_accelerator_seconds=MAX_QUANTUM_SECONDS,
                predicted_next_bytes=1_500_000,
            )
        else:
            current["kind"] = f"{row}_finalization"
            governor.admit(
                current["kind"],
                predicted_wall_seconds=10.0,
                predicted_accelerator_seconds=10.0,
                predicted_next_bytes=16_000_000,
            )

    def persist_latest() -> None:
        if "value" in latest:
            write_latest(latest["value"])

    setattr(callback, "persist_latest", persist_latest)
    return callback


def _clear_durable_step_checkpoint(run_dir: Path, row: str) -> None:
    for suffix in ("npz", "png"): (run_dir / f"failure/last_valid_{row}.{suffix}").unlink(missing_ok=True)
    tail = run_dir / "failure/telemetry_tail.json"
    if tail.exists() and _read_json(tail).get("row") == row:
        tail.unlink()


@torch.no_grad()
def run_native_ddpm_row(model: torch.nn.Module, schedule: Any, latent_z: np.ndarray, labels: np.ndarray,
                        path_ids: Sequence[str], reverse_seeds: np.ndarray, *, device: str | torch.device,
                        anchor_steps: Sequence[int] = NATIVE_ANCHORS,
                        step_callback: Callable[[Mapping[str, Any]], None] | None = None,
                        enforce_frozen_inventory: bool = True) -> NativeDDPMResult:
    target = torch.device(device)
    z = np.asarray(latent_z, dtype=np.float32); labels_np = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(path_ids, dtype=np.str_); seeds = np.asarray(reverse_seeds, dtype=np.uint64)
    _require(z.shape == (len(ids), 1, 28, 28) and labels_np.shape == seeds.shape == (len(ids),), "native DDPM identity mismatch")
    if enforce_frozen_inventory:
        expected = build_path_inventory()
        _require(np.array_equal(ids, expected["path_ids"]) and np.array_equal(labels_np, expected["requested_labels"])
                 and np.array_equal(seeds, expected["native_reverse_seeds"]), "native DDPM frozen inventory changed")
    state = torch.as_tensor(z, device=target); labels_t = torch.as_tensor(labels_np, device=target)
    model.to(target).eval(); total = int(schedule.num_steps); requested = tuple(int(x) for x in anchor_steps)
    _require(requested[0] == 0 and requested[-1] == total, "native anchor steps changed")
    generators = [torch.Generator(device=target).manual_seed(int(seed)) for seed in seeds.tolist()]
    saved: dict[int, np.ndarray] = {0: z.copy()}; telemetry: list[dict[str, Any]] = []
    for completed, timestep in enumerate(range(total - 1, -1, -1), start=1):
        began = time.perf_counter(); t = torch.full((len(ids),), timestep, dtype=torch.long, device=target)
        epsilon = model(state, t, labels_t)
        noise = torch.cat([torch.randn((1, 1, 28, 28), generator=gen, device=target) for gen in generators]) if timestep else torch.zeros_like(state)
        state = ddpm_step_from_epsilon(state, t, epsilon, schedule, noise)
        telemetry.append({"completed_step": completed, "ddpm_timestep": timestep, "state_mean": float(state.mean().cpu()),
                          "state_std": float(state.std().cpu()), "epsilon_rms": float(epsilon.square().mean().sqrt().cpu()),
                          "finite": int(torch.isfinite(state).all()), "elapsed_seconds": time.perf_counter() - began})
        _require(bool(torch.isfinite(state).all()), f"native DDPM became nonfinite at step {completed}")
        if completed in requested: saved[completed] = state.cpu().numpy().astype(np.float32)
        if step_callback is not None: step_callback({"row": "native_ddpm", "completed_step": completed, "state": state, "telemetry": telemetry,
                                                     "path_ids": ids, "requested_labels": labels_np, "reverse_seeds": seeds})
    anchors = np.stack([saved[x] for x in requested]).astype(np.float32)
    metadata = {"row": "native_ddpm", "comparison_role": "contextual_latent_linked_state_unpaired",
                "telemetry": [{k: v for k, v in x.items() if k != "elapsed_seconds"} for x in telemetry]}
    digest = _scientific_digest({"model_anchors": anchors, "reverse_steps": np.asarray(requested), "path_ids": ids,
                                 "requested_labels": labels_np, "reverse_seeds": seeds}, metadata)
    return NativeDDPMResult(anchors, np.asarray(requested, dtype=np.int64), ids, labels_np, seeds, _hash_array(z), digest, telemetry)


def _model_to_uint8(states: np.ndarray) -> np.ndarray:
    return np.rint(np.clip((np.asarray(states, dtype=np.float32) + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8).reshape(*states.shape[:-3], 28, 28)


def save_eulerian_population(run_dir: str | Path, result: EulerianRowResult, *, start_bank_sha256: str,
                             rng_contract_sha256: str) -> None:
    root = Path(run_dir); path = root / f"populations/{result.row}.npz"
    _require(not path.exists(), f"population already exists: {result.row}")
    deterministic = [{k: v for k, v in item.items() if k not in {"elapsed_seconds", "cuda_allocated_bytes"}} for item in result.telemetry]
    _write_npz(path, schema=np.asarray(VERSION + "-eulerian-population"), version=np.asarray(VERSION), row=np.asarray(result.row),
        anchors=result.anchors, anchor_steps=result.anchor_steps, path_ids=result.path_ids, requested_labels=result.requested_labels,
        start_bank_sha256=np.asarray(start_bank_sha256), rng_contract_sha256=np.asarray(rng_contract_sha256),
        crn_key_hashes=np.asarray(result.crn_key_hashes, dtype=np.str_), telemetry_scientific_json=np.asarray(json.dumps(deterministic, sort_keys=True, separators=(",", ":"))),
        factor_one_no_selection=np.asarray(1, dtype=np.int64), scientific_digest=np.asarray(result.scientific_digest))
    images = np.stack([mass_to_uint8(anchor) for anchor in result.anchors])
    _write_npz(root / f"populations/{result.row}_uint8.npz", schema=np.asarray(VERSION + "-uint8-population"),
               images_uint8=images, anchor_steps=result.anchor_steps, path_ids=result.path_ids, requested_labels=result.requested_labels)
    _write_csv(root / f"telemetry/{result.row}_steps.csv", result.telemetry)
    if result.row == "ddpm_eulerian" and result.provider_anchor_payload:
        arrays = {key: value for key, value in result.provider_anchor_payload.items() if value.dtype.kind in "biuf"}
        arrays.update({"anchor_steps": np.asarray(ANCHORS[1:], dtype=np.int64), "path_ids": result.path_ids, "requested_labels": result.requested_labels})
        _write_npz(root / "telemetry/adapter_paths.npz", **arrays)
        scalar = {key: value for key, value in arrays.items() if value.shape == (4, PATH_COUNT)}
        rows = [{"anchor_step": ANCHORS[a + 1], "path_id": result.path_ids[i], "requested_label": result.requested_labels[i],
                 **{key: value[a, i] for key, value in scalar.items()}} for a in range(4) for i in range(PATH_COUNT)]
        _write_csv(root / "telemetry/adapter_paths.csv", rows)
        _write_json(root / "telemetry/summary.json", {"schema": VERSION + "-telemetry-summary", "adapter": {
            key: {"mean": float(np.mean(value)), "maximum": float(np.max(value))} for key, value in scalar.items()}})


def save_native_population(run_dir: str | Path, result: NativeDDPMResult) -> None:
    root = Path(run_dir); path = root / "populations/native_ddpm.npz"; _require(not path.exists(), "native population already exists")
    _write_npz(path, schema=np.asarray(VERSION + "-native-population"), version=np.asarray(VERSION), model_anchors=result.model_anchors,
        reverse_steps=result.reverse_steps, path_ids=result.path_ids, requested_labels=result.requested_labels,
        reverse_seeds=result.reverse_seeds, latent_bank_sha256=np.asarray(result.latent_bank_sha256),
        comparison_role=np.asarray("contextual_latent_linked_state_unpaired"), factor_one_no_selection=np.asarray(1, dtype=np.int64),
        telemetry_scientific_json=np.asarray(json.dumps([{k: v for k, v in x.items() if k != "elapsed_seconds"} for x in result.telemetry], sort_keys=True, separators=(",", ":"))),
        scientific_digest=np.asarray(result.scientific_digest))
    _write_npz(root / "populations/native_ddpm_uint8.npz", schema=np.asarray(VERSION + "-uint8-population"),
        images_uint8=_model_to_uint8(result.model_anchors), anchor_steps=result.reverse_steps,
        path_ids=result.path_ids, requested_labels=result.requested_labels)
    _write_csv(root / "telemetry/native_ddpm_steps.csv", result.telemetry)


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive: return {key: archive[key] for key in archive.files}


def seal_populations(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir); seal_path = root / "populations/POPULATIONS_SEALED.json"
    _require(not seal_path.exists(), "populations are already sealed")
    files: dict[str, dict[str, Any]] = {}; row_receipts: dict[str, Any] = {}
    identities: tuple[np.ndarray, np.ndarray] | None = None
    for row in ROWS:
        for suffix in (".npz", "_uint8.npz"):
            path = root / f"populations/{row}{suffix}"; _require(path.is_file(), f"population artifact missing: {path.name}")
            files[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        raw = _npz(root / f"populations/{row}.npz")
        ids = raw["path_ids"].astype(np.str_); labels = raw["requested_labels"].astype(np.int64)
        if identities is None: identities = ids, labels
        _require(np.array_equal(ids, identities[0]) and np.array_equal(labels, identities[1]), "population identities differ")
        if row != "native_ddpm":
            _require(np.array_equal(raw["anchor_steps"], np.asarray(ANCHORS)), "Eulerian anchors changed")
            for anchor in raw["anchors"]: _validate_masses(anchor.astype(np.float32), PATH_COUNT, f"{row} anchor")
            expected = np.stack([mass_to_uint8(anchor) for anchor in raw["anchors"]])
        else: expected = _model_to_uint8(raw["model_anchors"])
        _require(int(raw["factor_one_no_selection"]) == 1, f"{row} factor-one policy changed")
        _require(np.array_equal(expected, _npz(root / f"populations/{row}_uint8.npz")["images_uint8"]), f"{row} raster changed")
        telemetry_path = root / f"telemetry/{row}_steps.csv"; _require(telemetry_path.is_file(), f"{row} telemetry is absent")
        files[telemetry_path.relative_to(root).as_posix()] = {"bytes": telemetry_path.stat().st_size, "sha256": sha256_file(telemetry_path)}
        with telemetry_path.open(newline="", encoding="utf-8") as handle: telemetry_count = sum(1 for _ in csv.DictReader(handle))
        expected_count = 1000 if row == "native_ddpm" else OUTER_STEPS
        _require(telemetry_count == expected_count, f"{row} telemetry row count changed")
        row_receipts[row] = {"raw": files[f"populations/{row}.npz"], "uint8": files[f"populations/{row}_uint8.npz"],
                             "telemetry": files[f"telemetry/{row}_steps.csv"], "telemetry_row_count": telemetry_count,
                             "scientific_digest": str(raw["scientific_digest"])}
    for relative in ("inventory/path_inventory.csv", "inventory/start_bank.npz", "inventory/teacher_target_bank.npz", "inventory/ddpm_latent_bank.npz",
                     "inventory/rng_contract.json", "inventory/STARTS_SEALED.json", "source_bindings.json", "checkpoint_bindings.json",
                     "telemetry/crn_key_hashes.json", "telemetry/model_state_identity.json", "telemetry/adapter_paths.npz",
                     "telemetry/adapter_paths.csv", "telemetry/summary.json"):
        path = root / relative; _require(path.is_file(), f"seal authority missing: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    authority_receipts = {key: files[key] for key in (
        "inventory/path_inventory.csv", "inventory/start_bank.npz", "inventory/teacher_target_bank.npz",
        "inventory/ddpm_latent_bank.npz", "inventory/rng_contract.json", "inventory/STARTS_SEALED.json",
        "source_bindings.json", "checkpoint_bindings.json", "telemetry/crn_key_hashes.json",
        "telemetry/model_state_identity.json", "telemetry/adapter_paths.npz", "telemetry/adapter_paths.csv",
        "telemetry/summary.json",
    )}
    payload = {"schema": VERSION + "-population-seal", "version": VERSION, "sealed_at": _utc_now(), "files": files,
               "path_count": PATH_COUNT, "rows": list(ROWS), "row_receipts": row_receipts, "authority_receipts": authority_receipts,
               "source_bindings_sha256": sha256_file(root / "source_bindings.json"),
               "checkpoint_bindings_sha256": sha256_file(root / "checkpoint_bindings.json"),
               "starts_seal_sha256": sha256_file(root / "inventory/STARTS_SEALED.json")}
    payload["seal_sha256"] = _sha256_bytes(_canonical_json_bytes(payload)); _write_json(seal_path, payload); return payload


def _verify_population_seal(run_dir: Path) -> dict[str, Any]:
    seal = _read_json(run_dir / "populations/POPULATIONS_SEALED.json")
    _require(seal.get("schema") == VERSION + "-population-seal" and seal.get("rows") == list(ROWS), "population seal schema changed")
    unsigned = {key: value for key, value in seal.items() if key != "seal_sha256"}
    _require(seal.get("seal_sha256") == _sha256_bytes(_canonical_json_bytes(unsigned)), "population seal digest changed")
    for relative, receipt in seal["files"].items():
        path = run_dir / relative; _require(path.is_file() and path.stat().st_size == receipt["bytes"] and sha256_file(path) == receipt["sha256"], f"sealed population changed: {relative}")
    inventory = build_path_inventory(); starts = _npz(run_dir / "inventory/start_bank.npz"); latent = _npz(run_dir / "inventory/ddpm_latent_bank.npz")
    with (run_dir / "inventory/path_inventory.csv").open(newline="", encoding="utf-8") as handle: inventory_rows = list(csv.DictReader(handle))
    columns = {"path_id": "path_ids", "requested_label": "requested_labels", "within_class_index": "within_class_index", "source_seed": "source_seeds", "ddpm_latent_seed": "ddpm_latent_seeds", "native_reverse_seed": "native_reverse_seeds", "retained": "retained"}
    _require(list(inventory_rows[0]) == list(columns) and len(inventory_rows) == PATH_COUNT, "path inventory CSV changed")
    for index, row in enumerate(inventory_rows):
        for key, source in columns.items(): _require(str(inventory[source][index]) == row[key], f"path inventory value changed: {key}")
    rng = _read_json(run_dir / "inventory/rng_contract.json")
    _require(rng == {"schema": VERSION + "-rng-contract", "edge_noise_root": EULERIAN_EDGE_NOISE_ROOT, "payload": "edge-noise-v1|<root>|<path_id>|<outer_step>|<attempted_substeps>|<sub_index>", "sha256_prefix_bytes": 8, "byteorder": "big"}, "RNG contract changed")
    _validate_inventory({key: starts[key] if key in starts else inventory[key] for key in inventory})
    _require(np.array_equal(starts["path_ids"], inventory["path_ids"]) and np.array_equal(starts["requested_labels"], inventory["requested_labels"]), "start identities changed")
    _require(np.array_equal(latent["path_ids"], inventory["path_ids"]) and np.array_equal(latent["requested_labels"], inventory["requested_labels"]), "latent identities changed")
    for row in ROWS:
        raw = _npz(run_dir / f"populations/{row}.npz"); _require(int(raw["factor_one_no_selection"]) == 1, f"{row} selection policy changed")
        _require(np.array_equal(raw["path_ids"], inventory["path_ids"]) and np.array_equal(raw["requested_labels"], inventory["requested_labels"]), f"{row} identities changed")
        telemetry = json.loads(str(raw["telemetry_scientific_json"])); expected_count = 1000 if row == "native_ddpm" else OUTER_STEPS
        with (run_dir / f"telemetry/{row}_steps.csv").open(newline="", encoding="utf-8") as handle: csv_rows = list(csv.DictReader(handle))
        _require(len(telemetry) == len(csv_rows) == expected_count and [int(x["completed_step"]) for x in telemetry] == list(range(1, expected_count + 1)), f"{row} telemetry count/order changed")
        if row != "native_ddpm":
            _require(_row_numerical_health(telemetry), f"{row} numerical/adaptive telemetry changed")
            _require(np.array_equal(raw["anchors"][0], starts["masses"]), f"{row} start authority changed")
            _require(str(raw["start_bank_sha256"]) == sha256_file(run_dir / "inventory/start_bank.npz") and str(raw["rng_contract_sha256"]) == sha256_file(run_dir / "inventory/rng_contract.json"), f"{row} bank/RNG receipt changed")
            metadata = {"row": row, "telemetry": json.loads(str(raw["telemetry_scientific_json"])), "crn_key_hashes": raw["crn_key_hashes"].tolist()}
            expected_keys = []
            for outer, record in enumerate(telemetry):
                for attempt in record["attempts"]:
                    for sub in range(int(attempt["substeps"])):
                        seeds = np.asarray([derive_edge_noise_seed(pid, outer, int(attempt["substeps"]), sub) for pid in inventory["path_ids"].tolist()], dtype=np.uint64)
                        expected_keys.append(_sha256_bytes(seeds.tobytes()))
            _require(raw["crn_key_hashes"].tolist() == expected_keys, f"{row} CRN key replay changed")
            digest = _scientific_digest({"anchors": raw["anchors"], "anchor_steps": raw["anchor_steps"], "path_ids": raw["path_ids"], "requested_labels": raw["requested_labels"]}, metadata)
        else:
            _require(np.array_equal(raw["model_anchors"][0], latent["z"]), "native latent authority changed")
            _require(np.array_equal(raw["reverse_steps"], np.asarray(NATIVE_ANCHORS, dtype=np.int64)), "native anchor schedule changed")
            _require(np.array_equal(raw["reverse_seeds"], inventory["native_reverse_seeds"]) and str(raw["latent_bank_sha256"]) == _hash_array(latent["z"]), "native seed/latent receipt changed")
            metadata = {"row": "native_ddpm", "comparison_role": "contextual_latent_linked_state_unpaired", "telemetry": json.loads(str(raw["telemetry_scientific_json"]))}
            digest = _scientific_digest({"model_anchors": raw["model_anchors"], "reverse_steps": raw["reverse_steps"], "path_ids": raw["path_ids"], "requested_labels": raw["requested_labels"], "reverse_seeds": raw["reverse_seeds"]}, metadata)
        _require(str(raw["scientific_digest"]) == digest, f"{row} scientific digest changed")
        receipt = seal["row_receipts"][row]; _require(receipt["scientific_digest"] == digest and receipt["telemetry_row_count"] == (1000 if row == "native_ddpm" else 256), f"{row} seal receipt changed")
    expected_authorities = {key: seal["files"][key] for key in (
        "inventory/path_inventory.csv", "inventory/start_bank.npz", "inventory/teacher_target_bank.npz",
        "inventory/ddpm_latent_bank.npz", "inventory/rng_contract.json", "inventory/STARTS_SEALED.json",
        "source_bindings.json", "checkpoint_bindings.json", "telemetry/crn_key_hashes.json",
        "telemetry/model_state_identity.json", "telemetry/adapter_paths.npz", "telemetry/adapter_paths.csv",
        "telemetry/summary.json",
    )}
    _require(seal["authority_receipts"] == expected_authorities, "population authority receipts changed")
    adapter_paths = _npz(run_dir / "telemetry/adapter_paths.npz")
    _require("predicted_mass" in adapter_paths and adapter_paths["predicted_mass"].shape == (4, PATH_COUNT, 784), "adapter predicted-mass telemetry shape changed")
    _require(np.array_equal(adapter_paths["anchor_steps"], np.asarray(ANCHORS[1:], dtype=np.int64)), "adapter path anchor steps changed")
    _require(np.array_equal(adapter_paths["path_ids"], inventory["path_ids"]) and np.array_equal(adapter_paths["requested_labels"], inventory["requested_labels"]), "adapter path identities changed")
    for key, value in adapter_paths.items():
        if value.dtype.kind in "biufc":
            _require(bool(np.isfinite(value).all()), f"adapter path telemetry is nonfinite: {key}")
    crn_receipt = _read_json(run_dir / "telemetry/crn_key_hashes.json")
    _require(set(crn_receipt) == set(EULERIAN_ROWS), "standalone CRN receipt rows changed")
    for row in EULERIAN_ROWS:
        _require(crn_receipt[row] == _npz(run_dir / f"populations/{row}.npz")["crn_key_hashes"].tolist(), f"standalone CRN receipt changed: {row}")
    identity = _read_json(run_dir / "telemetry/model_state_identity.json")
    _require(identity.get("schema") == VERSION + "-model-state-identity" and identity.get("passed") == 1, "model-state identity schema changed")
    _require(identity["historical_pre"] == identity["historical_post"] and identity["ddpm_pre"] == identity["ddpm_post"], "model state changed across populations")
    _require([item["row"] for item in identity["populations"]] == list(ROWS), "model-state population order changed")
    for item in identity["populations"]:
        _require(item.get("passed") == 1 and item["pre"] == item["post"], f"model state changed during {item['row']}")
        if "historical" in item["pre"]:
            _require(item["pre"]["historical"] == identity["historical_pre"], f"historical model root changed during {item['row']}")
        if "ddpm" in item["pre"]:
            _require(item["pre"]["ddpm"] == identity["ddpm_pre"], f"DDPM model root changed during {item['row']}")
    bindings = _read_json(run_dir / "checkpoint_bindings.json")
    bound_ddpm = bindings.get("ddpm_bound", {})
    if "model_state_sha256" in bound_ddpm:
        _require(identity["ddpm_pre"] == bound_ddpm["model_state_sha256"], "DDPM model identity changed from its bound checkpoint")
    historical_receipt = bindings.get("historical_receipt", {})
    if historical_receipt.get("clean_state_path"):
        clean_path = Path(historical_receipt["clean_state_path"])
        _require(clean_path.resolve() == (run_dir / "inventory/historical_state.pt").resolve()
                 and clean_path.is_file() and clean_path.stat().st_size == historical_receipt["clean_state_bytes"]
                 and sha256_file(clean_path) == historical_receipt["clean_state_sha256"], "historical clean-state authority changed")
        historical_model = v3._load_clean_model(clean_path, config=DirectFluxMNISTConfig(**historical_receipt["config"]), device="cpu")
        _require(identity["historical_pre"] == v3._model_state_semantic_digest(historical_model),
                 "historical model identity changed from its authenticated clean state")
    preflight_path = run_dir / "preflight/model_immutability.json"
    if preflight_path.is_file():
        preflight = _read_json(preflight_path)
        _require(preflight.get("historical_model_sha256") == identity["historical_pre"]
                 and preflight.get("ddpm_model_sha256") == identity["ddpm_pre"], "model immutability receipt changed")
    return seal


def _bind_real_authorities_impl(legacy_checkpoint: Path, ddpm_run_dir: Path, arff: Path, *, include_evaluator: bool = True) -> dict[str, Any]:
    def observed(path: Path, digest: str, size: int | None = None) -> dict[str, Any]:
        _require(path.is_file(), f"authority is absent: {path}")
        if size is not None: _require(path.stat().st_size == size, f"authority byte size changed: {path}")
        _require(sha256_file(path) == digest, f"authority SHA-256 changed: {path}")
        return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}
    result = {"schema": VERSION + "-checkpoint-bindings",
              "legacy_checkpoint": observed(legacy_checkpoint, LEGACY_CHECKPOINT_SHA256, LEGACY_CHECKPOINT_BYTES),
              "arff": observed(arff, MNIST_ARFF_SHA256, MNIST_ARFF_BYTES),
              "ddpm_checkpoint": observed(ddpm_run_dir / "training/selected_checkpoint.pt", DDPM_CHECKPOINT_SHA256)}
    for key, relative, digest in (("evaluator_checkpoint", "evaluator/selected_checkpoint.pt", EVALUATOR_SHA256),
                                  ("evaluator_selection", "evaluator/selection.json", EVALUATOR_SELECTION_SHA256)):
        path = ddpm_run_dir / relative
        result[key] = observed(path, digest) if include_evaluator else {"path": str(path.resolve()), "expected_sha256": digest, "opened_preseal": 0}
    selection = _read_json(ddpm_run_dir / "training/selection.json"); config = _read_json(ddpm_run_dir / "config.json")
    _require(selection.get("checkpoint_sha256") == DDPM_CHECKPOINT_SHA256, "DDPM selection does not bind checkpoint")
    schedule = config.get("schedule", {}); _require(schedule == {"beta_end": 0.02, "beta_start": 0.0001, "steps": 1000}, "DDPM schedule changed")
    result["ddpm_selection"] = selection; result["ddpm_config_sha256"] = sha256_file(ddpm_run_dir / "config.json")
    result["ddpm_run_dir"] = str(ddpm_run_dir.resolve()); return result


def _record_stage(run_dir: Path, stage: str) -> None:
    _require(stage in STAGE_ORDER, f"unknown stage: {stage}")
    path = run_dir / "stage_ledger.json"; events = _read_json(path).get("events", []) if path.exists() else []
    completed = [item["stage"] for item in events]
    _require(stage not in completed and (not completed or STAGE_ORDER.index(stage) > STAGE_ORDER.index(completed[-1])), "stage order changed")
    events.append({"stage": stage, "state": "completed", "recorded_at": _utc_now()}); _write_json(path, {"schema": VERSION + "-stage-ledger", "events": events})


def _status(run_dir: Path, state: str, *, route: str | None = None, error: str | None = None,
            failed_stage: str | None = None, whole_run_restart_required: bool | None = None) -> None:
    _write_json(run_dir / "status.json", {"schema": VERSION + "-status", "state": state, "route": route,
        "error": error, "failed_stage": failed_stage, "updated_at": _utc_now(),
        "whole_run_restart_required": int(state == "failed_unsealed" if whole_run_restart_required is None else whole_run_restart_required)})


def _load_evaluator_after_seal(run_dir: Path, ddpm_run_dir: Path, device: str | torch.device) -> SmallMnistCNN:
    _verify_population_seal(run_dir)
    checkpoint = ddpm_run_dir / "evaluator/selected_checkpoint.pt"; selection_path = ddpm_run_dir / "evaluator/selection.json"
    _require(sha256_file(checkpoint) == EVALUATOR_SHA256 and sha256_file(selection_path) == EVALUATOR_SELECTION_SHA256, "evaluator authority changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True); selection = _read_json(selection_path)
    _require(payload.get("selected_epoch") == selection.get("selected_epoch") and set(payload) == {"selected_epoch", "state_dict"}, "evaluator selection changed")
    model = SmallMnistCNN(); model.load_state_dict(payload["state_dict"], strict=True); return model.to(device).eval()


def _diversity(images: np.ndarray, labels: np.ndarray, reference: np.ndarray, reference_labels: np.ndarray) -> dict[str, Any]:
    return within_class_nn_diversity(images, labels, reference, reference_labels)


def _row_numerical_health(records: Sequence[Mapping[str, Any]], *, clip_target: float = 0.03,
                          max_substeps: int = 4) -> bool:
    allowed = [1, 2, 4]
    if max_substeps != 4 or len(records) != OUTER_STEPS:
        return False
    for record in records:
        attempts = record.get("attempts")
        accepted = int(record.get("accepted_substeps", 0))
        if (not isinstance(attempts, list) or not attempts or accepted not in allowed
                or [int(item.get("substeps", 0)) for item in attempts] != allowed[:allowed.index(accepted) + 1]
                or int(record.get("finite", 0)) != 1
                or not math.isfinite(float(record.get("minimum_mass", float("nan"))))
                or float(record["minimum_mass"]) < 0.0
                or not math.isfinite(float(record.get("maximum_mass_error", float("nan"))))
                or float(record["maximum_mass_error"]) > MASS_ERROR_MAXIMUM
                or not math.isfinite(float(record.get("poisson_divergence_residual_maximum", float("nan"))))
                or float(record["poisson_divergence_residual_maximum"]) > POISSON_RESIDUAL_MAXIMUM
                or any("fallback" in str(key).lower() for key in record)):
            return False
        for attempt in attempts:
            substeps = int(attempt["substeps"]); proposed = substeps * PATH_COUNT * 2 * 28 * 28
            if (int(attempt.get("proposed", -1)) != proposed
                    or int(attempt.get("clipped", -1)) < 0
                    or not math.isclose(float(attempt.get("clipping_fraction", -1.0)),
                                        int(attempt["clipped"]) / proposed, rel_tol=0.0, abs_tol=1e-15)):
                return False
        final = attempts[-1]
        if (int(record.get("accepted_proposed", -1)) != int(final["proposed"])
                or int(record.get("accepted_clipped", -1)) != int(final["clipped"])
                or (float(record.get("accepted_clipping_fraction", 1.0)) > clip_target and accepted < max_substeps)):
            return False
    return True


def _build_machine_gates(run_dir: Path, endpoint: Mapping[str, Mapping[str, Any]], comparison: Mapping[str, Any],
                         teacher_gate: Mapping[str, Any]) -> dict[str, Any]:
    preflight_paths = [run_dir / f"preflight/{name}.json" for name in ("cpu_smoke", "adapter_orientation", "crn_replay", "model_immutability")]
    if all(path.is_file() for path in preflight_paths):
        preflight = all(_read_json(path).get("passed", 0) == 1 for path in preflight_paths)
    else:
        preflight = _read_json(run_dir / "config.json")["execution_authority"]["device"] == "cpu"
    numerical = True
    for row in EULERIAN_ROWS:
        telemetry = json.loads(str(_npz(run_dir / f"populations/{row}.npz")["telemetry_scientific_json"]))
        numerical = numerical and _row_numerical_health(telemetry)
    i = {
        "i1_authority_and_leakage": {"gate_type": "execution/integrity", "passed": 1,
            "conditions": {"population_before_evaluator": 1, "factor_one_no_selection": 1, "bindings_authenticated": 1}},
        "i2_orientation_and_crn": {"gate_type": "execution/integrity", "passed": int(preflight),
            "conditions": {"cpu_smoke": int(preflight), "orientation": int(preflight), "crn_replay": int(preflight)}},
        "i3_numerical_health": {"gate_type": "execution/integrity", "passed": int(numerical),
            "conditions": {"finite_nonnegative_unit_mass": int(numerical), "poisson_residual_at_most_2e_4": int(numerical)}},
        "i4_full_interface_teacher": {"gate_type": "execution/integrity", "passed": int(bool(teacher_gate["passed"])), "values": dict(teacher_gate)},
        "i5_sealing_and_completeness": {"gate_type": "execution/integrity", "passed": 1,
            "conditions": {"all_rows_all_paths_all_anchors": 1, "population_seal_verified": 1, "all_outputs_retained": 1}},
    }
    integrity_pass = int(all(value["passed"] == 1 for value in i.values()))
    native = endpoint["native_ddpm"]; candidate = endpoint["ddpm_eulerian"]
    native_pass = native["classifier_accuracy"] >= NATIVE_CLASSIFIER_ACCURACY_MINIMUM and native["diversity_ratio"] >= NATIVE_DIVERSITY_RATIO_MINIMUM and native["exact_duplicate_pair_count"] == 0
    diversity_pass = candidate["diversity_ratio"] >= CANDIDATE_DIVERSITY_REAL_RATIO_MINIMUM and comparison["candidate_to_historical_diversity_ratio"] >= CANDIDATE_DIVERSITY_HISTORICAL_RATIO_MINIMUM and candidate["exact_duplicate_pair_count"] == 0
    return {"schema": VERSION + "-gates", **i,
        "integrity": {"gate_type": "execution/integrity", "passed": integrity_pass, "component_gates": list(i)},
        "native_ddpm": {"gate_type": "diagnostic threshold", "passed": int(native_pass), "values": dict(native)},
        "candidate_diversity": {"gate_type": "diagnostic threshold", "passed": int(diversity_pass),
            "values": {"real_ratio": candidate["diversity_ratio"], "historical_ratio": comparison["candidate_to_historical_diversity_ratio"], "duplicate_pair_count": candidate["exact_duplicate_pair_count"]}},
        "candidate_human_fidelity": {"gate_type": "diagnostic threshold", "state": "pending", "passed": None},
        "confirmatory_claim_thresholds": []}


def evaluate_sealed_populations(run_dir: str | Path, *, arff: str | Path, ddpm_run_dir: str | Path,
                                device: str | torch.device) -> dict[str, Any]:
    root = Path(run_dir); seal = _verify_population_seal(root); evaluator = _load_evaluator_after_seal(root, Path(ddpm_run_dir), device)
    arff_path = Path(arff); _require(arff_path.stat().st_size == MNIST_ARFF_BYTES and sha256_file(arff_path) == MNIST_ARFF_SHA256, "ARFF authority changed before terminal opening")
    test_images, test_labels = read_mnist_arff_slice(arff, start=60_000, stop=70_000)
    _write_json(root / "evaluation/SCORING_READY.json", {"schema": VERSION + "-scoring-ready", "population_seal_sha256": sha256_file(root / "populations/POPULATIONS_SEALED.json"), "terminal_rows_opened_after_seal": 1, "arff_bytes": MNIST_ARFF_BYTES, "arff_sha256": MNIST_ARFF_SHA256})
    anchor_rows: list[dict[str, Any]] = []; class_rows: list[dict[str, Any]] = []; prediction_arrays: dict[str, np.ndarray] = {}; endpoint: dict[str, dict[str, Any]] = {}
    for row in ROWS:
        uint8 = _npz(root / f"populations/{row}_uint8.npz"); images = uint8["images_uint8"]
        steps = uint8["anchor_steps"].astype(np.int64); labels = uint8["requested_labels"].astype(np.int64); ids = uint8["path_ids"].astype(np.str_)
        for index, step in enumerate(steps.tolist()):
            scored = evaluate_generated_labels(evaluator, images[index], labels, ids, device=device)
            diversity_detail = within_class_nn_diversity(images[index], labels, test_images, test_labels); diversity = float(diversity_detail["aggregate_median_ratio"])
            duplicates = int(exact_duplicate_metrics(images[index], labels)["duplicate_pair_count"])
            anchor_rows.append({"row": row, "anchor_step": int(step), "classifier_accuracy": float(scored["accuracy"]),
                                "diversity_ratio": diversity, "exact_duplicate_pair_count": duplicates,
                                "class_coverage": int(len(set(np.asarray(scored["predictions"]).tolist())))})
            prediction_arrays[f"{row}_{step}_predictions"] = np.asarray(scored["predictions"], dtype=np.int64)
            prediction_arrays[f"{row}_{step}_logits"] = np.asarray(scored["logits"], dtype=np.float64)
            confusion = np.zeros((10, 10), dtype=np.int64); np.add.at(confusion, (labels, np.asarray(scored["predictions"], dtype=np.int64)), 1)
            prediction_arrays[f"{row}_{step}_confusion_matrix"] = confusion
            for digit in range(10): class_rows.append({"row": row, "anchor_step": int(step), "requested_label": digit,
                "classifier_accuracy": scored["per_class"][str(digit)]["accuracy"], "diversity_ratio": diversity_detail["by_class"][str(digit)]["ratio"]})
        endpoint[row] = anchor_rows[-1] | {"classifier_per_class": scored["per_class"], "diversity_by_class": diversity_detail["by_class"], "confusion_matrix": confusion.tolist()}
        _write_json(root / f"evaluation/{row}_metrics.json", endpoint[row])
    _write_csv(root / "evaluation/per_anchor_metrics.csv", anchor_rows); _write_csv(root / "evaluation/per_class_metrics.csv", class_rows); _write_npz(root / "evaluation/predictions.npz", **prediction_arrays)
    historical_div = endpoint["historical"]["diversity_ratio"]; candidate = endpoint["ddpm_eulerian"]
    diversity_class_wins = sum(
        float(candidate["diversity_by_class"][str(digit)]["ratio"])
        > float(endpoint["historical"]["diversity_by_class"][str(digit)]["ratio"])
        for digit in range(10)
    )
    comparison = {"candidate_diversity_ratio": candidate["diversity_ratio"], "historical_diversity_ratio": historical_div,
                  "candidate_to_historical_diversity_ratio": candidate["diversity_ratio"] / max(historical_div, 1e-12),
                  "candidate_classifier_accuracy_difference": candidate["classifier_accuracy"] - endpoint["historical"]["classifier_accuracy"],
                  "candidate_diversity_classes_exceeding_historical": diversity_class_wins,
                  "candidate_diversity_class_supportive": int(diversity_class_wins >= 7)}
    early = early_joint_machine_proxy(anchor_rows)
    comparison.update(early)
    adapter_paths = _npz(root / "telemetry/adapter_paths.npz")
    _require("predicted_mass" in adapter_paths and adapter_paths["predicted_mass"].shape[-2:] == (PATH_COUNT, 784), "adapter predicted-mass anchors are absent")
    predicted_images = mass_to_uint8(adapter_paths["predicted_mass"][-1].astype(np.float32))
    predicted_diversity = _diversity(predicted_images, uint8["requested_labels"].astype(np.int64), test_images, test_labels)
    comparison["predicted_mass_diversity_ratio"] = float(predicted_diversity["aggregate_median_ratio"])
    comparison["predicted_mass_diversity_by_class"] = predicted_diversity["by_class"]
    comparison["predicted_mass_diverse"] = int(comparison["predicted_mass_diversity_ratio"] >= CANDIDATE_DIVERSITY_REAL_RATIO_MINIMUM)
    _write_json(root / "evaluation/ddpm_eulerian_minus_historical.json", comparison)
    _write_json(root / "evaluation/contextual_native_ddpm.json", endpoint["native_ddpm"] | {"comparison_role": "contextual_latent_linked_state_unpaired"})
    targets = _npz(root / "inventory/teacher_target_bank.npz")["masses"]; starts = _npz(root / "inventory/start_bank.npz")["masses"]
    teacher = _npz(root / "populations/teacher.npz")["anchors"]
    start_l2 = ((starts - targets) ** 2).sum(axis=1); short_l2 = ((teacher[1] - targets) ** 2).sum(axis=1); final_l2 = ((teacher[-1] - targets) ** 2).sum(axis=1)
    teacher_gate = {"improved_count": int(np.sum(final_l2 < start_l2)), "short_median_relative_l2": float(np.median(short_l2 / np.maximum(start_l2, 1e-20))),
                    "final_median_relative_l2": float(np.median(final_l2 / np.maximum(start_l2, 1e-20))), "classifier_accuracy": endpoint["teacher"]["classifier_accuracy"]}
    teacher_gate["passed"] = int(teacher_gate["improved_count"] >= TEACHER_IMPROVED_MINIMUM and teacher_gate["short_median_relative_l2"] <= TEACHER_SHORT_RELATIVE_L2_MAXIMUM and teacher_gate["final_median_relative_l2"] <= TEACHER_FINAL_RELATIVE_L2_MAXIMUM and teacher_gate["classifier_accuracy"] >= TEACHER_CLASSIFIER_ACCURACY_MINIMUM)
    machine = _build_machine_gates(root, endpoint, comparison, teacher_gate)
    _write_json(root / "gates.json", machine); return {"seal": seal, "gates": machine, "endpoint": endpoint, "comparison": comparison}


def _verify_evaluation(run_dir: Path, config: Mapping[str, Any], bindings: Mapping[str, Any] | None) -> dict[str, Any]:
    ready = _read_json(run_dir / "evaluation/SCORING_READY.json")
    _require(ready["schema"] == VERSION + "-scoring-ready"
             and ready["population_seal_sha256"] == sha256_file(run_dir / "populations/POPULATIONS_SEALED.json")
             and ready["terminal_rows_opened_after_seal"] == 1, "scoring firewall receipt changed")
    production = _is_production_config(config)
    reference_images = reference_labels = evaluator = None
    if production:
        _require(bindings is not None and ready["arff_bytes"] == MNIST_ARFF_BYTES and ready["arff_sha256"] == MNIST_ARFF_SHA256,
                 "terminal ARFF receipt changed")
        arff = Path(bindings["arff"]["path"]); _require(arff.stat().st_size == MNIST_ARFF_BYTES and sha256_file(arff) == MNIST_ARFF_SHA256, "terminal ARFF authority changed")
        reference_images, reference_labels = read_mnist_arff_slice(arff, start=60_000, stop=70_000)
        evaluator = _load_evaluator_after_seal(run_dir, Path(bindings["ddpm_run_dir"]), config["execution_authority"]["device"])
    with (run_dir / "evaluation/per_anchor_metrics.csv").open(newline="", encoding="utf-8") as handle: saved_anchor = list(csv.DictReader(handle))
    with (run_dir / "evaluation/per_class_metrics.csv").open(newline="", encoding="utf-8") as handle: saved_class = list(csv.DictReader(handle))
    _require(len(saved_anchor) == len(ROWS) * len(ANCHORS) and len(saved_class) == len(ROWS) * len(ANCHORS) * 10, "evaluation row inventory changed")
    arrays = _npz(run_dir / "evaluation/predictions.npz"); rebuilt_anchor = []; rebuilt_class = []; endpoint = {}
    expected_array_keys = {
        f"{row}_{step}_{suffix}"
        for row in ROWS for step in (NATIVE_ANCHORS if row == "native_ddpm" else ANCHORS)
        for suffix in ("predictions", "logits", "confusion_matrix")
    }
    _require(set(arrays) == expected_array_keys, "evaluator array inventory changed")
    for row in ROWS:
        population = _npz(run_dir / f"populations/{row}_uint8.npz"); images = population["images_uint8"]
        labels = population["requested_labels"].astype(np.int64); ids = population["path_ids"].astype(np.str_); steps = population["anchor_steps"].astype(np.int64)
        for anchor_index, step in enumerate(steps.tolist()):
            prefix = f"{row}_{step}"
            _require(arrays[f"{prefix}_predictions"].dtype == np.dtype(np.int64)
                     and arrays[f"{prefix}_logits"].dtype == np.dtype(np.float64)
                     and arrays[f"{prefix}_confusion_matrix"].dtype == np.dtype(np.int64),
                     f"{prefix} evaluator array dtype changed")
            predictions = arrays[f"{prefix}_predictions"]; logits = arrays[f"{prefix}_logits"]
            confusion = arrays[f"{prefix}_confusion_matrix"]
            _require(predictions.shape == (PATH_COUNT,) and logits.shape == (PATH_COUNT, 10) and confusion.shape == (10, 10), f"{prefix} evaluator array shape changed")
            _require(np.array_equal(predictions, logits.argmax(axis=1)), f"{prefix} predictions/logits changed")
            expected_confusion = np.zeros((10, 10), dtype=np.int64); np.add.at(expected_confusion, (labels, predictions), 1)
            _require(np.array_equal(confusion, expected_confusion), f"{prefix} confusion matrix changed")
            if production:
                scored = evaluate_generated_labels(evaluator, images[anchor_index], labels, ids, device=config["execution_authority"]["device"])
                _require(np.array_equal(predictions, np.asarray(scored["predictions"], dtype=np.int64))
                         and np.array_equal(logits, np.asarray(scored["logits"], dtype=np.float64)), f"{prefix} evaluator replay changed")
                diversity = _diversity(images[anchor_index], labels, reference_images, reference_labels)
            else:
                diversity = None
            duplicates = int(exact_duplicate_metrics(images[anchor_index], labels)["duplicate_pair_count"])
            saved = saved_anchor[len(rebuilt_anchor)]; ratio = float(saved["diversity_ratio"])
            if diversity is not None: _require(math.isclose(ratio, float(diversity["aggregate_median_ratio"]), rel_tol=1e-12, abs_tol=1e-12), f"{prefix} diversity changed")
            row_value = {"row": row, "anchor_step": step, "classifier_accuracy": float(np.mean(predictions == labels)),
                "diversity_ratio": ratio, "exact_duplicate_pair_count": duplicates, "class_coverage": len(set(predictions.tolist()))}
            for key, value in row_value.items():
                _require(str(value) == saved[key] or (isinstance(value, float) and math.isclose(value, float(saved[key]), rel_tol=1e-12, abs_tol=1e-12)), f"{prefix} anchor metric changed: {key}")
            rebuilt_anchor.append(row_value)
            per_class = {}; diversity_by_class = {}
            for digit in range(10):
                mask = labels == digit; accuracy = float(np.mean(predictions[mask] == digit)); saved_digit = saved_class[len(rebuilt_class)]
                class_ratio = float(saved_digit["diversity_ratio"])
                if diversity is not None: _require(math.isclose(class_ratio, float(diversity["by_class"][str(digit)]["ratio"]), rel_tol=1e-12, abs_tol=1e-12), f"{prefix} class diversity changed")
                _require(saved_digit["row"] == row and int(saved_digit["anchor_step"]) == step and int(saved_digit["requested_label"]) == digit
                         and math.isclose(float(saved_digit["classifier_accuracy"]), accuracy, rel_tol=0.0, abs_tol=1e-12), f"{prefix} per-class metric changed")
                rebuilt_class.append(saved_digit); per_class[str(digit)] = {"count": int(mask.sum()), "accuracy": accuracy}
                diversity_by_class[str(digit)] = (diversity["by_class"][str(digit)] if diversity is not None else {"ratio": class_ratio})
        saved_endpoint = _read_json(run_dir / f"evaluation/{row}_metrics.json")
        final = rebuilt_anchor[-1]
        for key, value in final.items():
            _require(saved_endpoint[key] == value, f"{row} endpoint metric changed: {key}")
        _require(saved_endpoint["classifier_per_class"] == per_class and saved_endpoint["confusion_matrix"] == expected_confusion.tolist(), f"{row} endpoint class evidence changed")
        if production: _require(saved_endpoint["diversity_by_class"] == diversity_by_class, f"{row} endpoint diversity evidence changed")
        endpoint[row] = saved_endpoint
    historical = endpoint["historical"]; candidate = endpoint["ddpm_eulerian"]
    diversity_class_wins = sum(
        float(candidate["diversity_by_class"][str(digit)]["ratio"])
        > float(historical["diversity_by_class"][str(digit)]["ratio"])
        for digit in range(10)
    )
    expected_comparison = {"candidate_diversity_ratio": candidate["diversity_ratio"], "historical_diversity_ratio": historical["diversity_ratio"],
        "candidate_to_historical_diversity_ratio": candidate["diversity_ratio"] / max(historical["diversity_ratio"], 1e-12),
        "candidate_classifier_accuracy_difference": candidate["classifier_accuracy"] - historical["classifier_accuracy"],
        "candidate_diversity_classes_exceeding_historical": diversity_class_wins,
        "candidate_diversity_class_supportive": int(diversity_class_wins >= 7),
        **early_joint_machine_proxy(rebuilt_anchor)}
    saved_comparison = _read_json(run_dir / "evaluation/ddpm_eulerian_minus_historical.json")
    for key, value in expected_comparison.items(): _require(saved_comparison[key] == value, f"machine comparison changed: {key}")
    if production:
        adapter = _npz(run_dir / "telemetry/adapter_paths.npz"); predicted = mass_to_uint8(adapter["predicted_mass"][-1].astype(np.float32))
        predicted_diversity = _diversity(predicted, build_path_inventory()["requested_labels"], reference_images, reference_labels)
        _require(math.isclose(saved_comparison["predicted_mass_diversity_ratio"], float(predicted_diversity["aggregate_median_ratio"]), rel_tol=1e-12, abs_tol=1e-12)
                 and saved_comparison["predicted_mass_diversity_by_class"] == predicted_diversity["by_class"], "predicted-mass diversity changed")
    _require(saved_comparison["predicted_mass_diverse"] == int(saved_comparison["predicted_mass_diversity_ratio"] >= CANDIDATE_DIVERSITY_REAL_RATIO_MINIMUM), "predicted-mass diversity proposition changed")
    starts = _npz(run_dir / "inventory/start_bank.npz")["masses"]; targets = _npz(run_dir / "inventory/teacher_target_bank.npz")["masses"]
    teacher = _npz(run_dir / "populations/teacher.npz")["anchors"]
    start_l2 = ((starts - targets) ** 2).sum(axis=1); short_l2 = ((teacher[1] - targets) ** 2).sum(axis=1); final_l2 = ((teacher[-1] - targets) ** 2).sum(axis=1)
    teacher_gate = {"improved_count": int(np.sum(final_l2 < start_l2)), "short_median_relative_l2": float(np.median(short_l2 / np.maximum(start_l2, 1e-20))),
        "final_median_relative_l2": float(np.median(final_l2 / np.maximum(start_l2, 1e-20))), "classifier_accuracy": endpoint["teacher"]["classifier_accuracy"]}
    teacher_gate["passed"] = int(teacher_gate["improved_count"] >= TEACHER_IMPROVED_MINIMUM and teacher_gate["short_median_relative_l2"] <= TEACHER_SHORT_RELATIVE_L2_MAXIMUM and teacher_gate["final_median_relative_l2"] <= TEACHER_FINAL_RELATIVE_L2_MAXIMUM and teacher_gate["classifier_accuracy"] >= TEACHER_CLASSIFIER_ACCURACY_MINIMUM)
    expected_gates = _build_machine_gates(run_dir, endpoint, saved_comparison, teacher_gate); saved_gates = _read_json(run_dir / "gates.json")
    if saved_gates["candidate_human_fidelity"]["state"] == "complete": expected_gates["candidate_human_fidelity"] = saved_gates["candidate_human_fidelity"]
    _require(saved_gates == expected_gates, "typed gates changed")
    _require(_read_json(run_dir / "evaluation/contextual_native_ddpm.json") == endpoint["native_ddpm"] | {"comparison_role": "contextual_latent_linked_state_unpaired"}, "native contextual metrics changed")
    return {"endpoint": endpoint, "comparison": saved_comparison, "gates": saved_gates}


def early_joint_machine_proxy(anchor_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {(str(item["row"]), int(item["anchor_step"])): item for item in anchor_rows}
    candidate_steps, historical_steps = [], []
    for step in (64, 128):
        candidate, historical = rows[("ddpm_eulerian", step)], rows[("historical", step)]
        historical_ok = historical["classifier_accuracy"] >= 0.80 and historical["diversity_ratio"] >= 0.25 and historical["exact_duplicate_pair_count"] == 0
        candidate_ok = candidate["classifier_accuracy"] >= 0.80 and candidate["diversity_ratio"] >= 0.25 and candidate["diversity_ratio"] / max(historical["diversity_ratio"], 1e-12) >= 2.0 and candidate["exact_duplicate_pair_count"] == 0
        if candidate_ok: candidate_steps.append(step)
        if historical_ok: historical_steps.append(step)
    return {"early_fidelity_role": "machine_only_exploratory_proxy_not_human_D1", "candidate_early_joint_steps": candidate_steps,
            "historical_early_joint_steps": historical_steps, "candidate_early_joint": int(bool(candidate_steps)), "historical_early_joint": int(bool(historical_steps))}


def prepare_blind_review(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir); seal = _verify_population_seal(root); render_sealed_population_images(root); review = root / "review"; review.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []; images: list[np.ndarray] = []
    for row in ("historical", "ddpm_eulerian"):
        pop = _npz(root / f"populations/{row}_uint8.npz")
        for path_id, label, image in zip(pop["path_ids"], pop["requested_labels"], pop["images_uint8"][-1], strict=True):
            rows.append({"row": row, "path_id": str(path_id), "requested_label": int(label)}); images.append(image)
    order = np.random.default_rng(REVIEW_SEED).permutation(len(rows)); ordered = [rows[int(i)] for i in order]; ordered_images = np.stack(images)[order]
    key = []; template = []
    for index, item in enumerate(ordered):
        blind = f"blind-{index:03d}"; key.append({"blind_id": blind, **item}); template.append({"blind_id": blind, "recognizable": "", "perceived_digit": "", "notes": ""})
    _write_json(review / "private_key.json", {"schema": VERSION + "-review-key", "items": key, "review_seed": REVIEW_SEED})
    _write_csv(review / "human_review_template.csv", template); _write_json(review / "membership.json", {"schema": VERSION + "-review-membership", "count": 80, "rows": {"historical": 40, "ddpm_eulerian": 40}})
    write_contact_sheet(review / "blinded_historical_vs_adapter.png", ordered_images, columns=8, scale=4, captions=[item["blind_id"] for item in template])
    ready = {"schema": VERSION + "-review-ready", "population_seal_sha256": sha256_file(root / "populations/POPULATIONS_SEALED.json"),
             "key_sha256": sha256_file(review / "private_key.json"), "template_sha256": sha256_file(review / "human_review_template.csv"), "count": 80}
    _write_json(review / "REVIEW_READY.json", ready); return ready


def _expected_review_items(run_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []; images: list[np.ndarray] = []
    for row in ("historical", "ddpm_eulerian"):
        population = _npz(run_dir / f"populations/{row}_uint8.npz")
        for path_id, label, image in zip(population["path_ids"], population["requested_labels"], population["images_uint8"][-1], strict=True):
            rows.append({"row": row, "path_id": str(path_id), "requested_label": int(label)}); images.append(image)
    order = np.random.default_rng(REVIEW_SEED).permutation(len(rows))
    items = [{"blind_id": f"blind-{index:03d}", **rows[int(source)]} for index, source in enumerate(order)]
    return items, np.stack(images)[order]


def _verify_review_bundle(run_dir: Path, *, require_answers: bool = False) -> dict[str, Any]:
    _verify_population_seal(run_dir)
    expected, images = _expected_review_items(run_dir)
    key = _read_json(run_dir / "review/private_key.json")
    _require(key == {"schema": VERSION + "-review-key", "items": expected, "review_seed": REVIEW_SEED}, "review key changed")
    with (run_dir / "review/human_review_template.csv").open(newline="", encoding="utf-8") as handle:
        template = list(csv.DictReader(handle))
    _require(template == [{"blind_id": item["blind_id"], "recognizable": "", "perceived_digit": "", "notes": ""} for item in expected], "review template changed")
    membership = _read_json(run_dir / "review/membership.json")
    _require(membership == {"schema": VERSION + "-review-membership", "count": 80, "rows": {"historical": 40, "ddpm_eulerian": 40}}, "review membership changed")
    ready = _read_json(run_dir / "review/REVIEW_READY.json")
    _require(ready == {"schema": VERSION + "-review-ready", "population_seal_sha256": sha256_file(run_dir / "populations/POPULATIONS_SEALED.json"),
        "key_sha256": sha256_file(run_dir / "review/private_key.json"), "template_sha256": sha256_file(run_dir / "review/human_review_template.csv"), "count": 80}, "review readiness changed")
    v3._verify_sheet_pixels(run_dir / "review/blinded_historical_vs_adapter.png", images, columns=8, scale=4,
                            captions=[item["blind_id"] for item in expected])
    if not require_answers:
        return {"items": expected, "ready": ready}
    with (run_dir / "review/human_review_answers.csv").open(newline="", encoding="utf-8-sig") as handle:
        answers = list(csv.DictReader(handle))
    by_id = {row.get("blind_id", ""): row for row in answers}
    _require(len(answers) == 80 and set(by_id) == {item["blind_id"] for item in expected}, "review answers changed")
    joined = []
    for item in expected:
        answer = by_id[item["blind_id"]]; recognizable = int(answer["recognizable"]); perceived = int(answer["perceived_digit"])
        _require(recognizable in (0, 1) and 0 <= perceived <= 9, "review answer value changed")
        joined.append(item | {"recognizable": recognizable, "perceived_digit": perceived,
                              "requested_label_agreement": int(recognizable and perceived == item["requested_label"])})
    saved = _read_json(run_dir / "review/human_review.json")
    _require(set(saved) == {"schema", "reviewer", "answers_sha256", "rows"}
             and isinstance(saved["reviewer"], str) and len(saved["reviewer"].strip()) >= 2
             and saved["schema"] == VERSION + "-human-review" and saved["answers_sha256"] == sha256_file(run_dir / "review/human_review_answers.csv")
             and saved["rows"] == joined, "human review replay changed")
    summaries = {}
    for row in ("historical", "ddpm_eulerian"):
        subset = [item for item in joined if item["row"] == row]
        summaries[row] = {"count": 40, "recognizability": float(np.mean([item["recognizable"] for item in subset])),
            "requested_label_agreement": float(np.mean([item["requested_label_agreement"] for item in subset]))}
    _require(_read_json(run_dir / "review/human_review_by_row.json") == summaries, "human review summaries changed")
    return {"items": expected, "ready": ready, "joined": joined, "summaries": summaries, "reviewer": saved["reviewer"]}


def render_sealed_population_images(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir); _verify_population_seal(root); (root / "images").mkdir(exist_ok=True)
    finals = []
    for row in ROWS:
        population = _npz(root / f"populations/{row}_uint8.npz"); images = population["images_uint8"]; ids = population["path_ids"].astype(np.str_)
        write_contact_sheet(root / f"images/{row}_final.png", images[-1], columns=10, scale=3, captions=ids.tolist())
        trajectory = images.transpose(1, 0, 2, 3).reshape(-1, 28, 28); captions = [f"{pid}@{step}" for pid in ids for step in population["anchor_steps"].tolist()]
        write_contact_sheet(root / f"images/{row}_trajectory.png", trajectory, columns=5, scale=2, captions=captions); finals.append(images[-1])
    write_contact_sheet(root / "images/comparison_final.png", np.concatenate(finals), columns=10, scale=3,
                        captions=[f"{row}:{index:03d}" for row in ROWS for index in range(PATH_COUNT)])
    paths = [root / f"images/{row}_{kind}.png" for row in ROWS for kind in ("final", "trajectory")] + [root / "images/comparison_final.png"]
    return {"schema": VERSION + "-image-receipts", "files": {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}}


render_population_images = render_sealed_population_images


def _outcome_route(*, integrity_passed: bool, native_valid: bool, adapter_human_fidelity: bool,
                   adapter_diversity: bool, early_joint: bool = False, predicted_mass_diverse: bool = False,
                   eulerian_collapsed: bool = False, historical_early_joint: bool = False,
                   learned_both_negative: bool = False) -> str:
    if not integrity_passed: raise IntegrityFailure("integrity gate blocks scientific routing")
    if not native_valid: return "native_ddpm_control_invalid"
    if adapter_human_fidelity and adapter_diversity: return "adapter_positive_freeze_replication"
    if early_joint: return "adapter_early_joint_horizon_replication"
    if adapter_diversity and not adapter_human_fidelity: return "adapter_diverse_not_faithful_major_pivot_or_stop"
    if eulerian_collapsed and predicted_mass_diverse: return "composition_mode_loss_theory_bridge_or_stop"
    if eulerian_collapsed and not predicted_mass_diverse: return "off_policy_bridge_on_policy_or_stop"
    if adapter_human_fidelity and not adapter_diversity: return "adapter_fidelity_only_major_pivot_or_stop"
    if historical_early_joint: return "historical_early_horizon_replication"
    if learned_both_negative: return "learned_eulerian_negative_stop_or_major_pivot"
    if not adapter_human_fidelity and not adapter_diversity and not predicted_mass_diverse:
        return "off_policy_bridge_on_policy_or_stop"
    return "unclassified_stop_redesign"


def _route_inputs(gates: Mapping[str, Any], comparison: Mapping[str, Any], human: Mapping[str, Any]) -> dict[str, bool]:
    human_pass = float(human["recognizability"]) >= CANDIDATE_HUMAN_RECOGNIZABILITY_MINIMUM and float(human["requested_label_agreement"]) >= CANDIDATE_HUMAN_AGREEMENT_MINIMUM
    diversity_pass = bool(gates["candidate_diversity"]["passed"])
    early = bool(comparison["candidate_early_joint"]); historical_early = bool(comparison["historical_early_joint"])
    collapsed = float(comparison["candidate_diversity_ratio"]) < CANDIDATE_DIVERSITY_REAL_RATIO_MINIMUM
    learned_negative = not human_pass and not diversity_pass and not early and not historical_early and not collapsed
    return {"integrity_passed": bool(gates["integrity"]["passed"]), "native_valid": bool(gates["native_ddpm"]["passed"]),
        "adapter_human_fidelity": human_pass, "adapter_diversity": diversity_pass, "early_joint": early,
        "predicted_mass_diverse": bool(comparison["predicted_mass_diverse"]), "eulerian_collapsed": collapsed,
        "historical_early_joint": historical_early, "learned_both_negative": learned_negative}


def _route_action(route: str) -> str:
    return {
        "adapter_positive_freeze_replication": "Freeze this adapter/configuration and replicate on fresh protected seeds.",
        "native_ddpm_control_invalid": "Repair the native DDPM binding in a new run; do not interpret the adapter.",
        "adapter_fidelity_only_major_pivot_or_stop": "Stop local tuning; separately approve an on-policy/multiscale pivot or stop.",
        "adapter_early_joint_horizon_replication": "Permit one fresh predeclared horizon-only replication; no gain sweep.",
        "adapter_diverse_not_faithful_major_pivot_or_stop": "Use a major on-policy or potential-gradient bridge, or stop.",
        "composition_mode_loss_theory_bridge_or_stop": "Run at most one fixed theory-shaped mobility-weighted bridge, or stop.",
        "off_policy_bridge_on_policy_or_stop": "Only on-policy/global training is justified; otherwise stop.",
        "historical_early_horizon_replication": "At most one fresh horizon-only comparison; the checkpoint line remains stopped.",
        "learned_eulerian_negative_stop_or_major_pivot": "Stop the Eulerian generator hypothesis absent approval for one major program.",
        "unclassified_stop_redesign": "Do not scale or tune; redesign the single underidentified defect or stop.",
    }[route]


def _report_text(run_dir: Path, outcome: Mapping[str, Any] | None = None) -> str:
    status = _read_json(run_dir / "status.json"); gates = _read_json(run_dir / "gates.json") if (run_dir / "gates.json").exists() else {}
    ledger = _read_json(run_dir / "resource_ledger.json") if (run_dir / "resource_ledger.json").exists() else {}
    comparison = _read_json(run_dir / "evaluation/ddpm_eulerian_minus_historical.json") if (run_dir / "evaluation/ddpm_eulerian_minus_historical.json").exists() else {}
    route = None if outcome is None else outcome.get("route"); action = "Await blinded manual review; do not infer answers." if route is None else _route_action(str(route))
    text = ["# DDPM-to-Eulerian diversity pilot", "", "Research mode: exploratory.",
            "Decision: can the frozen diverse DDPM drive the fixed Eulerian process while retaining fidelity and improving diversity?",
            f"Status: `{status['state']}`; route: `{route}`.", "", "## Evidence and configuration",
            "Exact command: `command.txt`. Authorities: `source_bindings.json`, `checkpoint_bindings.json`, `config.json`.",
            "Paths: 40 class-major paths, 4 per digit; all are retained; no ranking, replacement, or best-of-N selection.",
            "Rows: null, full-interface teacher, frozen historical, DDPM-Eulerian adapter, and contextual latent-linked unpaired native DDPM.",
            "Primary artifacts: `populations/`, `evaluation/per_anchor_metrics.csv`, `images/`, and `review/`.", "", "## Gates and effects",
            f"Typed gate authority: `{json.dumps(gates, sort_keys=True)}`", "Human fidelity and classifier fidelity remain separate. Native DDPM is not a paired causal baseline.",
            f"Machine comparison: `{json.dumps(comparison, sort_keys=True)}`",
            f"Resource accounting: wall_seconds={ledger.get('wall_seconds')}; accelerator_seconds={ledger.get('accelerator_seconds')}; events={len(ledger.get('events', []))}.",
            "", "## Claim boundary", "This exploratory run can establish feasibility only at these 40 frozen paths. It cannot establish population superiority, exact Doob-transform theory, DDPM/Eulerian path equivalence, a continuum limit, or a general claim about Eulerian generators.",
            "", "## Outcome to action", action, "", "## Deliberate omissions", "No new training, gain/schedule search, candidate selection, terminal-data calibration, native-to-adapter target injection, or automatic follow-up compute was performed.",
            "NPZ self-description is scoped: raw populations and sealed inventory banks carry scientific digests; derived uint8, adapter-telemetry, prediction, and failure NPZs are authoritative only through the seal/manifest plus semantic replay, so they are not independently self-certifying."]
    return "\n".join(text) + "\n"


def _write_reports(run_dir: Path, outcome: Mapping[str, Any] | None = None) -> None:
    text = _report_text(run_dir, outcome)
    v3._atomic_bytes(run_dir / "REPORT.md", text.encode("utf-8"))
    v3._atomic_bytes(run_dir / "DELIBERATE_OMISSIONS.md", ("# Deliberate omissions\n\nNo new training, gain/schedule search, candidate selection, terminal-data calibration, native-to-adapter target injection, or automatic follow-up compute was performed.\n\nNPZ self-description is scoped: raw populations and sealed inventory banks carry scientific digests; derived uint8, adapter-telemetry, prediction, and failure NPZs are authoritative only through the seal/manifest plus semantic replay, so they are not independently self-certifying.\n").encode("utf-8"))


def _manifest(run_dir: Path) -> dict[str, Any]:
    ignored = {"artifact_manifest.json"}; rows = []
    for path in sorted((x for x in run_dir.rglob("*") if x.is_file()), key=lambda x: x.relative_to(run_dir).as_posix()):
        rel = path.relative_to(run_dir).as_posix()
        if rel not in ignored: rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"schema": VERSION + "-artifact-manifest", "artifact_count": len(rows), "artifact_bytes": sum(x["bytes"] for x in rows),
               "tree_digest": _sha256_bytes(_canonical_json_bytes(rows)), "files": rows}
    _write_json(run_dir / "artifact_manifest.json", payload); return payload


def _terminalize(run_dir: Path, governor: ResourceGovernor | None = None, *,
                 kind: str | None = None, event_already_open: bool = False) -> dict[str, Any]:
    status = _read_json(run_dir / "status.json")
    kind = kind or ("human_review_terminalization" if status["state"] == "complete" else "machine_terminalization")
    override = bool(governor is not None and (governor.failed_admission is not None or status["state"] in {"failed_unsealed", "resource_stopped", "postseal_interrupted"}))
    if governor is not None and not event_already_open:
        governor.admit(kind, predicted_wall_seconds=MAX_QUANTUM_SECONDS,
                       predicted_accelerator_seconds=MAX_QUANTUM_SECONDS if governor.device.type == "cuda" else 0.0,
                       predicted_next_bytes=2_000_000, reserve_remaining_seconds=0.0, terminal_override=override)
    _write_reports(run_dir, _read_json(run_dir / "outcome.json") if (run_dir / "outcome.json").exists() else None)
    _write_json(run_dir / "VERIFY_RECEIPT.json", {"schema": VERSION + "-verify-receipt", "state": "pending",
        "passed": 0, "written_by": "terminal-semantic-verifier", "recorded_at": _utc_now()})
    provisional = _manifest(run_dir)
    precheck = _verify_run_impl(run_dir, allow_pending_receipt=True, allow_terminal_open=governor is not None)
    if governor is not None:
        governor.complete(kind, terminal_override=override)
        _write_reports(run_dir, _read_json(run_dir / "outcome.json") if (run_dir / "outcome.json").exists() else None)
    _write_json(run_dir / "VERIFY_RECEIPT.json", {"schema": VERSION + "-verify-receipt", "state": "complete",
        "passed": 1, "written_by": "terminal-semantic-verifier", "verified_status": status["state"],
        "provisional_tree_digest": provisional["tree_digest"], "semantic_precheck_sha256": _sha256_bytes(_canonical_json_bytes(precheck)),
        "recorded_at": _utc_now()})
    manifest = _manifest(run_dir)
    _verify_manifest(run_dir)
    _verify_resource_ledger(run_dir, _read_json(run_dir / "config.json"), status)
    _require((run_dir / "REPORT.md").read_text(encoding="utf-8") == _report_text(run_dir,
        _read_json(run_dir / "outcome.json") if (run_dir / "outcome.json").exists() else None), "terminal REPORT changed")
    return manifest


def _close_failed_review_attempt(root: Path, governor: ResourceGovernor, error: BaseException, *,
                                 reviewer: str, source: Path, answer_hash: str | None,
                                 attempts_path: Path, attempts: list[dict[str, Any]]) -> None:
    attempts.append({"reviewer": reviewer, "answers_path": str(source), "answers_sha256": answer_hash,
                     "passed": 0, "error": str(error) or type(error).__name__, "recorded_at": _utc_now()})
    _write_json(attempts_path, {"schema": VERSION + "-review-submission-attempts", "attempts": attempts})
    try:
        _terminalize(root, governor, kind="human_review_terminalization",
                     event_already_open="human_review_terminalization" in governor._open)
    except ResourceStop as resource_error:
        _save_failure_evidence(root, resource_error, "human_review_terminalization", governor)
        _status(root, "resource_stopped", route="resource_stopped", error=str(resource_error),
                failed_stage="human_review_terminalization", whole_run_restart_required=False)
        _terminalize(root, governor, kind="human_review_terminalization",
                     event_already_open="human_review_terminalization" in governor._open)
        raise


def finalize_review(args: argparse.Namespace) -> int:
    root = Path(args.run_dir).resolve(); reviewer = str(args.reviewer).strip(); source = Path(args.answers).resolve()
    _require(bool(args.confirm_manual_review), "manual-review confirmation is required")
    _require(len(reviewer) >= 2, "reviewer identity is invalid")
    _verify_manifest(root)
    status = _read_json(root / "status.json"); _require(status["state"] == "awaiting_human_review", "run is not awaiting human review")
    config = _read_json(root / "config.json"); _is_production_config(config); _verify_resource_ledger(root, config, status)
    governor = ResourceGovernor.rehydrate(root, device=config["execution_authority"]["device"])
    rollback_paths = tuple(root / name for name in ("resource_ledger.json", "REPORT.md", "DELIBERATE_OMISSIONS.md", "artifact_manifest.json"))
    rollback = {path: (path.read_bytes(), path.stat().st_atime_ns, path.stat().st_mtime_ns) for path in rollback_paths}
    try:
        governor.admit("human_review_terminalization", predicted_wall_seconds=MAX_QUANTUM_SECONDS,
                       predicted_accelerator_seconds=MAX_QUANTUM_SECONDS if governor.device.type == "cuda" else 0.0,
                       predicted_next_bytes=2_000_000, reserve_remaining_seconds=0.0)
    except ResourceStop as error:
        _save_failure_evidence(root, error, "human_review_terminalization", governor)
        _status(root, "resource_stopped", route="resource_stopped", error=str(error),
                failed_stage="human_review_terminalization", whole_run_restart_required=False)
        _terminalize(root, governor, kind="human_review_terminalization")
        raise
    _write_reports(root)
    _manifest(root)
    _CHARGED_TERMINAL_PRECHECKS.add(root)
    try:
        verify_run(root)
    except BaseException:
        for path, (payload, accessed, modified) in rollback.items():
            v3._atomic_bytes(path, payload)
            os.utime(path, ns=(accessed, modified))
        raise
    finally:
        _CHARGED_TERMINAL_PRECHECKS.discard(root)
    attempts_path = root / "review/submission_attempts.json"
    attempts: list[dict[str, Any]] = []
    answer_hash: str | None = None
    try:
        key = _read_json(root / "review/private_key.json")["items"]
        attempts = _read_json(attempts_path).get("attempts", []) if attempts_path.exists() else []
        answer_hash = sha256_file(source) if source.is_file() else None
        _require(source.is_file(), "review answer file is absent")
        with source.open(newline="", encoding="utf-8-sig") as handle: answers = list(csv.DictReader(handle))
        by_id = {row.get("blind_id", ""): row for row in answers}; expected_ids = {item["blind_id"] for item in key}
        _require(len(answers) == 80 and set(by_id) == expected_ids, "review answers are incomplete or duplicated")
        joined = []
        for item in key:
            answer = by_id[item["blind_id"]]; recognizable = int(answer["recognizable"]); perceived = int(answer["perceived_digit"])
            _require(recognizable in (0, 1) and 0 <= perceived <= 9, "review answer value is invalid")
            joined.append(item | {"recognizable": recognizable, "perceived_digit": perceived,
                                  "requested_label_agreement": int(recognizable and perceived == int(item["requested_label"]))})
    except BaseException as error:
        _close_failed_review_attempt(root, governor, error, reviewer=reviewer, source=source,
                                     answer_hash=answer_hash, attempts_path=attempts_path, attempts=attempts)
        raise
    summaries = {}
    for row in ("historical", "ddpm_eulerian"):
        subset = [item for item in joined if item["row"] == row]
        summaries[row] = {"count": 40, "recognizability": float(np.mean([item["recognizable"] for item in subset])),
            "requested_label_agreement": float(np.mean([item["requested_label_agreement"] for item in subset]))}
    original_gates = (root / "gates.json").read_bytes()
    original_stages = (root / "stage_ledger.json").read_bytes()
    original_status = (root / "status.json").read_bytes()
    adopted = ("review/human_review_answers.csv", "review/human_review.json",
               "review/human_review_by_row.json", "outcome.json")
    try:
        v3._atomic_bytes(root / "review/human_review_answers.csv", source.read_bytes())
        _write_json(root / "review/human_review.json", {"schema": VERSION + "-human-review", "reviewer": reviewer,
            "answers_sha256": sha256_file(root / "review/human_review_answers.csv"), "rows": joined})
        _write_json(root / "review/human_review_by_row.json", summaries)
        gates = _read_json(root / "gates.json"); human = summaries["ddpm_eulerian"]
        inputs = _route_inputs(gates, _read_json(root / "evaluation/ddpm_eulerian_minus_historical.json"), human)
        gates["candidate_human_fidelity"] = {"gate_type": "diagnostic threshold", "state": "complete",
            "passed": int(inputs["adapter_human_fidelity"]), "values": human}; _write_json(root / "gates.json", gates)
        comparison = _read_json(root / "evaluation/ddpm_eulerian_minus_historical.json"); inputs = _route_inputs(gates, comparison, human)
        route = _outcome_route(**inputs)
        outcome = {"schema": VERSION + "-outcome", "route": route, "next_action": _route_action(route), "full_scale_auto_launched": 0,
                   "route_inputs": inputs, "human": summaries, "machine_comparison": comparison, "claim_scope": "exploratory 40-path pilot only"}
        _write_json(root / "outcome.json", outcome); _record_stage(root, "human_review_terminalization"); _status(root, "complete", route=route)
        attempts.append({"reviewer": reviewer, "answers_path": str(source), "answers_sha256": answer_hash,
                         "passed": 1, "error": None, "recorded_at": _utc_now()})
        _write_json(attempts_path, {"schema": VERSION + "-review-submission-attempts", "attempts": attempts})
        _terminalize(root, governor, kind="human_review_terminalization", event_already_open=True)
    except BaseException as error:
        if isinstance(error, ResourceStop):
            _save_failure_evidence(root, error, "human_review_terminalization", governor)
            _status(root, "resource_stopped", route="resource_stopped", error=str(error),
                    failed_stage="human_review_terminalization", whole_run_restart_required=False)
            _terminalize(root, governor, kind="human_review_terminalization",
                         event_already_open="human_review_terminalization" in governor._open)
            raise
        for relative in adopted: (root / relative).unlink(missing_ok=True)
        v3._atomic_bytes(root / "gates.json", original_gates)
        v3._atomic_bytes(root / "stage_ledger.json", original_stages)
        v3._atomic_bytes(root / "status.json", original_status)
        attempts = [item for item in attempts if item.get("passed") != 1]
        _close_failed_review_attempt(root, governor, error, reviewer=reviewer, source=source,
                                     answer_hash=answer_hash, attempts_path=attempts_path, attempts=attempts)
        raise
    return 0


def _run_cpu_smoke_impl() -> dict[str, Any]:
    config = dataclasses.replace(DirectFluxMNISTConfig(), num_steps=4, free_weight=0.0, noise_weight=0.05,
                                 adaptive_sampling=False, max_substeps=1)
    starts = np.full((2, 784), np.float32(1 / 784), dtype=np.float32); labels = np.asarray([0, 1]); ids = ["smoke-000", "smoke-001"]
    targets = starts.copy(); targets[0, 100] += .02; targets[1, 600] += .02; targets /= targets.sum(axis=1, keepdims=True)
    class SmokeProvider:
        def __call__(self, masses: Tensor, labels: Tensor, remaining_time: Tensor, path_ids: Sequence[str], source_masses: Tensor) -> ControllerStep:
            del labels, remaining_time, path_ids, source_masses
            return ControllerStep(torch.zeros((len(masses), 2, 28, 28), device=masses.device), {"smoke": torch.ones(len(masses), device=masses.device)})
    adapter_module = _adapter_module(); latent = np.random.default_rng(SMOKE_SEED).standard_normal((2, 1, 28, 28)).astype(np.float32)
    class ZeroEpsilon(torch.nn.Module):
        def forward(self, state: Tensor, timestep: Tensor, labels: Tensor) -> Tensor:
            del timestep, labels
            return torch.zeros_like(state)
    smoke_schedule = make_linear_ddpm_schedule(4, 1e-4, .02)
    smoke_adapter = adapter_module.DDPMEulerianAdapter(ZeroEpsilon(), smoke_schedule, config,
        adapter_module.DDPMEulerianAdapterConfig(num_ddpm_steps=4, mass_floor=float(config.mass_floor)))
    providers: Mapping[str, ControllerProvider] = {"null": NullControllerProvider(config), "teacher": TeacherControllerProvider(config, targets, ids),
        "historical": SmokeProvider(), "ddpm_eulerian": DDPMEulerianControllerProvider(smoke_adapter, latent, ids)}
    results = {row: run_eulerian_row(starts, labels, ids, config, provider, row=row, device="cpu", num_steps=4,
                                      schedule_steps=4, anchors=(0, 1, 2, 3, 4), edge_noise_root=SMOKE_SEED) for row, provider in providers.items()}
    replay = run_eulerian_row(starts, labels, ids, config, providers["teacher"], row="teacher", device="cpu", num_steps=4,
                              schedule_steps=4, anchors=(0, 1, 2, 3, 4), edge_noise_root=SMOKE_SEED)
    _require(np.array_equal(results["teacher"].anchors, replay.anchors), "CPU smoke replay changed")
    _require(len({tuple(value.crn_key_hashes) for value in results.values()}) == 1, "CPU smoke CRN differs across rows")
    native_result = run_native_ddpm_row(ZeroEpsilon(), smoke_schedule, latent, labels, ids, np.asarray([SMOKE_SEED, SMOKE_SEED + 1], dtype=np.uint64),
        device="cpu", anchor_steps=(0, 1, 2, 3, 4), enforce_frozen_inventory=False)
    native = native_result.model_anchors; rendered = mass_to_uint8(results["ddpm_eulerian"].endpoints); metric_stub = float(rendered.mean())
    _require(math.isfinite(metric_stub) and 0.0 <= metric_stub <= 255.0, "smoke metric stub failed")
    with tempfile.TemporaryDirectory(prefix="d2e-smoke-") as temporary:
        root = Path(temporary); (root / "populations").mkdir(); (root / "failure").mkdir()
        for row, result in results.items(): _write_npz(root / f"populations/{row}.npz", anchors=result.anchors)
        _write_npz(root / "populations/native_ddpm.npz", model_anchors=native); Image.fromarray(rendered[0]).save(root / "render.png")
        seal_rows = {path.name: sha256_file(path) for path in sorted((root / "populations").glob("*.npz"))}; _write_json(root / "POPULATIONS_SEALED.json", seal_rows)
        before = [(p.name, sha256_file(p)) for p in sorted(root.rglob("*")) if p.is_file()]
        _require(all(sha256_file(root / "populations" / name) == digest for name, digest in seal_rows.items()), "smoke seal replay failed")
        class BadProvider(SmokeProvider):
            def __call__(self, *args: Any, **kwargs: Any) -> ControllerStep:
                step = super().__call__(*args, **kwargs); return ControllerStep(step.conditioning_flux, {"poisson_divergence_residual": torch.ones(2)})
        try: run_eulerian_row(starts, labels, ids, config, BadProvider(), row="ddpm_eulerian", device="cpu", num_steps=4, schedule_steps=4, anchors=(0, 1, 2, 3, 4))
        except IntegrityFailure as error:
            _write_json(root / "failure/failure.json", {"error": str(error)}); _write_npz(root / "failure/last_valid.npz", masses=starts); Image.fromarray(mass_to_uint8(starts)[0]).save(root / "failure/last_valid.png")
        _require((root / "failure/last_valid.png").is_file(), "smoke forced failure was not retained")
        after_science = [(p.name, sha256_file(p)) for p in sorted(root.rglob("*")) if p.is_file() and "failure" not in p.parts]
        _require(before == after_science, "smoke post-seal replay mutated scientific artifacts")
        seal = _sha256_bytes(_canonical_json_bytes(seal_rows))
    return {"schema": VERSION + "-cpu-smoke", "passed": 1, "path_count": 2, "outer_steps": 4, "rows": [*EULERIAN_ROWS, "native_ddpm"],
        "inventory_and_latent_passed": 1, "adapter_provider_passed": 1, "native_sampler_stub_passed": 1, "crn_replay_exact": 1,
        "population_seal_passed": 1, "metric_stub_passed": 1, "metric_stub_endpoint_uint8_mean": metric_stub, "rendering_passed": 1, "forced_failure_retained": 1,
        "postseal_resume_no_generation_passed": 1, "read_only_verification_passed": 1, "scientific_digest": seal,
        "scientific_evidence": 0, "external_authorities_opened": 0}


def _source_bindings(repository_root: Path) -> dict[str, Any]:
    relative = ("mnist/diag_d0_ddpm_eulerian_diversity_pilot.py", "mnist/ddpm_eulerian_adapter.py", "mnist/eulerian_flux_mnist.py",
                "mnist/diag_d0_eulerian_edge_flux_replay.py", "mnist/pixel_ddpm.py", "mnist/mnist_generation_benchmark.py", "mnist/conditioned_diffusion.py",
                "mnist/weighted_point_cloud.py", "core/conditioning_utils.py", "core/wasserstein_conditioning_algorithms.py")
    relative = (*relative, "docs/ddpm_eulerian_diversity_pilot.md")
    files = []
    for item in relative:
        path = repository_root / item; _require(path.is_file(), f"source file is absent: {item}")
        files.append({"path": item, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": VERSION + "-source-bindings", "repository_root": str(repository_root), "git_revision": v3._git_revision(repository_root), "files": files}


def _config(args: argparse.Namespace, repository_root: Path, budget: ResourceBudget) -> dict[str, Any]:
    science = {"path_count": PATH_COUNT, "paths_per_class": PATHS_PER_CLASS, "outer_steps": OUTER_STEPS, "anchors": list(ANCHORS),
        "native_anchors": list(NATIVE_ANCHORS), "seeds": {"inventory": INVENTORY_SEED, "source_base": SOURCE_SEED_BASE, "latent_base": DDPM_LATENT_SEED_BASE,
        "edge_noise_root": EULERIAN_EDGE_NOISE_ROOT, "native_reverse_base": NATIVE_DDPM_REVERSE_SEED_BASE, "review": REVIEW_SEED, "smoke": SMOKE_SEED},
        "mass_scale": {"numerator": MASS_SCALE_NUMERATOR, "denominator": MASS_SCALE_DENOMINATOR, "hex": MASS_SCALE_HEX},
        "adapter": {"num_ddpm_steps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "min_tau_fraction": 0.03,
                    "time_map": "linear_remaining_fraction_round", "latent_policy": "persistent_path_latent", "flux_projection": "periodic_minimum_energy_minus_free"},
        "replay_policy": {"generated_candidates_per_path": 1, "selector": None, "all_outputs_retained": 1, "adaptive_retries_are_numerical": 1},
        "rows": list(ROWS), "thresholds": {"poisson_residual_maximum": POISSON_RESIDUAL_MAXIMUM, "mass_error_maximum": MASS_ERROR_MAXIMUM}}
    argv = [sys.executable, "-B", "-m", "mnist.diag_d0_ddpm_eulerian_diversity_pilot", *sys.argv[1:]]
    return {"schema": VERSION + "-config", "version": VERSION, "research_mode": RESEARCH_MODE, "created_at": _utc_now(),
            "command": subprocess.list2cmdline(argv), "argv": argv, "repository_root": str(repository_root), "scientific_configuration": science,
            "execution_authority": {"approval_id": args.approval_id, "device": args.device, **dataclasses.asdict(budget)},
            "input_paths": {"legacy_checkpoint": str(Path(args.legacy_checkpoint).resolve()), "ddpm_run_dir": str(Path(args.ddpm_run_dir).resolve()), "arff": str(Path(args.arff).resolve())}}


def _initialize(args: argparse.Namespace) -> tuple[Path, ResourceGovernor, dict[str, Any]]:
    root = Path(args.run_dir).resolve(); repository = _repository_root(); approval = str(args.approval_id).strip()
    _require(args.device == "cuda:0" and torch.cuda.is_available(), "production requires available cuda:0")
    _require(len(approval) >= 12 and "<" not in approval and "placeholder" not in approval.lower(), "approval ID is invalid")
    budget = ResourceBudget(float(args.max_wall_seconds), float(args.max_accelerator_seconds), int(float(args.max_storage_mib) * 1024**2), float(args.max_cuda_fraction))
    if args.resume_post_seal:
        _require(root.is_dir(), "post-seal run directory is absent")
        saved_status = _read_json(root / "status.json")
        _require(saved_status.get("state") == "postseal_interrupted", "only an authenticated post-seal interruption may be resumed")
        verify_run(root)
        saved_config = _read_json(root / "config.json")
        saved_budget = ResourceBudget(**{key: saved_config["execution_authority"][key] for key in dataclasses.asdict(budget)})
        _require(saved_budget == budget and saved_config["execution_authority"]["device"] == args.device
                 and saved_config["execution_authority"]["approval_id"] == approval, "resume execution authority changed")
        _require(saved_config["input_paths"] == {"legacy_checkpoint": str(Path(args.legacy_checkpoint).resolve()),
            "ddpm_run_dir": str(Path(args.ddpm_run_dir).resolve()), "arff": str(Path(args.arff).resolve())}, "resume input paths changed")
        governor = ResourceGovernor.rehydrate(root, device=args.device)
        completed = {item["stage"] for item in _read_json(root / "stage_ledger.json")["events"]}
        derived = []
        if "machine_scoring" not in completed:
            derived.extend(("evaluation", "review", "images"))
        elif "render_and_review_bundle" not in completed:
            derived.extend(("review", "images"))
        for name in derived:
            directory = root / name
            if directory.exists():
                for path in sorted((x for x in directory.rglob("*") if x.is_file()), reverse=True): path.unlink()
        for name in ("artifact_manifest.json", "VERIFY_RECEIPT.json", "REPORT.md", "DELIBERATE_OMISSIONS.md"):
            (root / name).unlink(missing_ok=True)
        if "machine_scoring" not in completed:
            (root / "gates.json").unlink(missing_ok=True)
        _status(root, "running", route="postseal_resume")
        return root, governor, saved_config
    _require(not root.exists() or not any(root.iterdir()), "run directory must be nonexistent or empty")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("inventory", "preflight", "populations", "telemetry", "evaluation", "images/failures", "review", "failure"): (root / name).mkdir(parents=True, exist_ok=True)
    config = _config(args, repository, budget); _write_json(root / "config.json", config); v3._atomic_bytes(root / "command.txt", (config["command"] + "\n").encode())
    _write_json(root / "source_bindings.json", _source_bindings(repository)); _write_json(root / "claim_boundary.json", {"schema": VERSION + "-claim-boundary", "mode": RESEARCH_MODE,
        "establishes_at_most": "exploratory feasibility on 40 frozen paths", "does_not_establish": ["population superiority", "exact Doob transform", "native/Eulerian equivalence", "general Eulerian failure"]})
    _status(root, "running"); governor = ResourceGovernor(root, budget, device=args.device); governor.write(); return root, governor, config


def run_production(args: argparse.Namespace) -> int:
    stage = "binding_preflight"; run_dir: Path | None = None; governor: ResourceGovernor | None = None
    active_callback: Callable[[Mapping[str, Any]], None] | None = None
    try:
        run_dir, governor, config_payload = _initialize(args)
        if args.resume_post_seal:
            completed = {item["stage"] for item in _read_json(run_dir / "stage_ledger.json")["events"]}
            if "machine_scoring" not in completed:
                stage = "machine_scoring"; governor.admit(stage, predicted_wall_seconds=MAX_QUANTUM_SECONDS,
                    predicted_accelerator_seconds=MAX_QUANTUM_SECONDS, predicted_next_bytes=16_000_000)
                evaluate_sealed_populations(run_dir, arff=args.arff, ddpm_run_dir=args.ddpm_run_dir, device=args.device)
                governor.complete(stage); _record_stage(run_dir, stage)
            if "render_and_review_bundle" not in completed:
                stage = "render_and_review_bundle"; governor.admit(stage, predicted_wall_seconds=MAX_QUANTUM_SECONDS,
                    predicted_accelerator_seconds=0.0, predicted_next_bytes=32_000_000)
                prepare_blind_review(run_dir); governor.complete(stage); _record_stage(run_dir, stage)
            if "awaiting_human_review" not in completed:
                _record_stage(run_dir, "awaiting_human_review")
            _status(run_dir, "awaiting_human_review", route="awaiting_human_review")
            stage = "machine_terminalization"
            _terminalize(run_dir, governor); return 0
        governor.admit("binding_preflight", predicted_wall_seconds=MAX_QUANTUM_SECONDS,
            predicted_accelerator_seconds=MAX_QUANTUM_SECONDS, predicted_next_bytes=24_000_000)
        authorities = _bind_real_authorities_impl(Path(args.legacy_checkpoint), Path(args.ddpm_run_dir), Path(args.arff), include_evaluator=False)
        clean = run_dir / "inventory/historical_state.pt"; legacy = v3.safe_extract_legacy_checkpoint(args.legacy_checkpoint, clean)
        adapter_module = _adapter_module(); bound = adapter_module.load_bound_ddpm_generator(Path(args.ddpm_run_dir), device=torch.device(args.device), expected_sha256=DDPM_CHECKPOINT_SHA256)
        _require(bound.model_state_sha256 == v3._model_state_semantic_digest(bound.model), "DDPM bound/runtime identity digest mismatch")
        authorities["historical_receipt"] = legacy; authorities["ddpm_bound"] = {key: _jsonable(getattr(bound, key)) for key in ("checkpoint_path", "checkpoint_bytes", "checkpoint_sha256", "selection_path", "selection_sha256", "config_path", "config_sha256", "schedule_path", "schedule_bytes", "schedule_sha256", "parameter_count", "model_state_sha256")}
        authorities["ddpm_bound"]["selection_bytes"] = Path(bound.selection_path).stat().st_size; authorities["ddpm_bound"]["config_bytes"] = Path(bound.config_path).stat().st_size
        _write_json(run_dir / "checkpoint_bindings.json", authorities); governor.complete("binding_preflight"); _record_stage(run_dir, stage)
        stage = "cpu_smoke_replay"; governor.admit(stage, predicted_wall_seconds=30.0, predicted_accelerator_seconds=0.0, predicted_next_bytes=2_000_000)
        smoke_started = time.perf_counter(); smoke = run_cpu_smoke(); smoke_elapsed = time.perf_counter() - smoke_started
        _write_json(run_dir / "preflight/cpu_smoke.json", smoke)
        crn = np.asarray([derive_edge_noise_seed(path_id, 0, 1, 0) for path_id in build_path_inventory()["path_ids"].tolist()], dtype=np.uint64)
        _write_json(run_dir / "preflight/crn_replay.json", {"schema": VERSION + "-crn-replay", "byteorder": "big",
            "root": EULERIAN_EDGE_NOISE_ROOT, "first_step_sha256": _sha256_bytes(crn.tobytes()), "exact_replay": 1, "passed": 1})
        _write_json(run_dir / "preflight/adapter_orientation.json", {"schema": VERSION + "-adapter-orientation",
            "time_map": "linear_remaining_fraction_round", "flux_orientation": "horizontal_then_vertical_periodic",
            "poisson_residual_maximum": POISSON_RESIDUAL_MAXIMUM, "cpu_smoke_adapter_passed": smoke["adapter_provider_passed"], "passed": 1})
        projection = resource_projection(smoke_path_seconds=max(smoke_elapsed / 8.0, 1e-6), native_step_seconds=max(smoke_elapsed / 4_000.0, 1e-6), stored_bytes_per_path=2_000_000)
        projection["projected_wall_seconds"] = max(float(projection["projected_wall_seconds"]), 1_500.0)
        projection["projected_accelerator_seconds"] = max(float(projection["projected_accelerator_seconds"]), 900.0)
        projection["projected_storage_bytes"] = max(int(projection["projected_storage_bytes"]), 100 * 1024**2)
        projection["code_informed_floor"] = {"wall_seconds": 1_500.0, "accelerator_seconds": 900.0, "storage_bytes": 100 * 1024**2}
        projection["fits_frozen_maxima"] = int(projection["projected_wall_seconds"] + TERMINAL_RESERVE_SECONDS <= MAX_WALL_SECONDS
            and projection["projected_accelerator_seconds"] <= MAX_ACCELERATOR_SECONDS and projection["projected_storage_bytes"] <= int(MAX_STORAGE_MIB * 1024**2))
        projection["fits_execution_authority"] = int(projection["projected_wall_seconds"] + governor.budget.reserve_seconds <= governor.budget.max_wall_seconds
            and projection["projected_accelerator_seconds"] <= governor.budget.max_accelerator_seconds and projection["projected_storage_bytes"] <= governor.budget.max_storage_bytes)
        _write_json(run_dir / "preflight/resource_projection.json", {"schema": VERSION + "-resource-projection", **projection})
        _require(projection["fits_frozen_maxima"] == 1 and projection["fits_execution_authority"] == 1, "resource projection exceeds execution authority")
        governor.complete(stage); _record_stage(run_dir, stage)
        stage = "inventory_and_start_seal"; governor.admit(stage, predicted_wall_seconds=MAX_QUANTUM_SECONDS,
            predicted_accelerator_seconds=30.0, predicted_next_bytes=48_000_000)
        images, labels_all, data_audit = read_mnist_development_prefix(args.arff)
        _require(images.shape == (60_000, 28, 28), "development prefix changed"); inventory = build_path_inventory(); starts = build_start_bank(DirectFluxMNISTConfig(**legacy["config"]), inventory)
        teacher = build_teacher_target_bank(images[VALIDATION_START:VALIDATION_STOP], labels_all[VALIDATION_START:VALIDATION_STOP], inventory); latent = build_ddpm_latent_bank(inventory)
        _write_csv(run_dir / "inventory/path_inventory.csv", [{"path_id": inventory["path_ids"][i], "requested_label": inventory["requested_labels"][i], "within_class_index": inventory["within_class_index"][i], "source_seed": inventory["source_seeds"][i], "ddpm_latent_seed": inventory["ddpm_latent_seeds"][i], "native_reverse_seed": inventory["native_reverse_seeds"][i], "retained": inventory["retained"][i]} for i in range(PATH_COUNT)])
        start_arrays = {"masses": starts, "path_ids": inventory["path_ids"], "requested_labels": inventory["requested_labels"], "source_seeds": inventory["source_seeds"]}
        _write_npz(run_dir / "inventory/start_bank.npz", schema=np.asarray(VERSION + "-start-bank"), version=np.asarray(VERSION), **start_arrays, scientific_sha256=np.asarray(_scientific_digest(start_arrays, {})))
        teacher_arrays = {**teacher}; _write_npz(run_dir / "inventory/teacher_target_bank.npz", schema=np.asarray(VERSION + "-teacher-target-bank"), version=np.asarray(VERSION), **teacher_arrays, scientific_sha256=np.asarray(_scientific_digest({k: v for k, v in teacher_arrays.items() if np.asarray(v).dtype.kind != "O"}, {})))
        latent_arrays = {**latent}; _write_npz(run_dir / "inventory/ddpm_latent_bank.npz", schema=np.asarray(VERSION + "-latent-bank"), version=np.asarray(VERSION), **latent_arrays, scientific_sha256=np.asarray(_scientific_digest({k: v for k, v in latent_arrays.items() if k != "z_sha256"}, {})))
        rng = {"schema": VERSION + "-rng-contract", "edge_noise_root": EULERIAN_EDGE_NOISE_ROOT, "payload": "edge-noise-v1|<root>|<path_id>|<outer_step>|<attempted_substeps>|<sub_index>", "sha256_prefix_bytes": 8, "byteorder": "big"}; _write_json(run_dir / "inventory/rng_contract.json", rng)
        starts_seal = {"schema": VERSION + "-starts-seal", "start_bank_sha256": sha256_file(run_dir / "inventory/start_bank.npz"), "teacher_target_bank_sha256": sha256_file(run_dir / "inventory/teacher_target_bank.npz"), "latent_bank_sha256": sha256_file(run_dir / "inventory/ddpm_latent_bank.npz"), "rng_contract_sha256": sha256_file(run_dir / "inventory/rng_contract.json"), "source_bindings_sha256": sha256_file(run_dir / "source_bindings.json"), "checkpoint_bindings_sha256": sha256_file(run_dir / "checkpoint_bindings.json"), "config_sha256": sha256_file(run_dir / "config.json"), "data_audit": data_audit}; _write_json(run_dir / "inventory/STARTS_SEALED.json", starts_seal)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        _write_json(run_dir / "preflight/deterministic_execution.json", {"deterministic_algorithms": 1, "cudnn_benchmark": 0, "cudnn_deterministic": 1, "tf32_policy_changed_by_runner": 0})
        econfig = DirectFluxMNISTConfig(**legacy["config"]); hist = DirectFluxUNet(econfig, base_channels=48); hist.load_state_dict(torch.load(clean, map_location="cpu", weights_only=True)); hist.requires_grad_(False); hist.to(args.device).eval()
        bound.model.requires_grad_(False); _require(not any(parameter.requires_grad for parameter in (*hist.parameters(), *bound.model.parameters())), "frozen model has trainable parameters")
        historical_pre, ddpm_pre = v3._model_state_semantic_digest(hist), v3._model_state_semantic_digest(bound.model)
        _write_json(run_dir / "preflight/model_immutability.json", {"schema": VERSION + "-model-immutability",
            "historical_model_sha256": historical_pre, "ddpm_model_sha256": ddpm_pre,
            "all_parameters_require_grad_false": 1, "training_forbidden": 1, "passed": 1})
        governor.complete(stage); _record_stage(run_dir, stage)
        providers = {"null": NullControllerProvider(econfig), "teacher": TeacherControllerProvider(econfig, teacher["masses"], inventory["path_ids"]),
                     "historical": HistoricalControllerProvider(hist), "ddpm_eulerian": DDPMEulerianControllerProvider(adapter_module.DDPMEulerianAdapter(bound.model, bound.schedule, econfig, adapter_module.DDPMEulerianAdapterConfig(mass_floor=float(econfig.mass_floor))), latent["z"], inventory["path_ids"])}
        crn_hashes = {}; model_identity = []
        for row in EULERIAN_ROWS:
            stage = row + "_population"; pre = {"historical": v3._model_state_semantic_digest(hist), "ddpm": v3._model_state_semantic_digest(bound.model)}
            callback = _durable_step_callback(run_dir, row, governor, OUTER_STEPS, initial_state=starts,
                path_ids=inventory["path_ids"], requested_labels=inventory["requested_labels"])
            active_callback = callback
            result = run_eulerian_row(starts, inventory["requested_labels"], inventory["path_ids"], econfig, providers[row], row=row, device=args.device, outer_step_callback=callback)
            save_eulerian_population(run_dir, result, start_bank_sha256=starts_seal["start_bank_sha256"], rng_contract_sha256=starts_seal["rng_contract_sha256"]); crn_hashes[row] = result.crn_key_hashes
            post = {"historical": v3._model_state_semantic_digest(hist), "ddpm": v3._model_state_semantic_digest(bound.model)}; _require(post == pre, f"model state changed during {row}")
            model_identity.append({"row": row, "pre": pre, "post": post, "passed": 1})
            governor.complete(f"{row}_finalization")
            _clear_durable_step_checkpoint(run_dir, row); active_callback = None; _record_stage(run_dir, stage)
        _write_json(run_dir / "telemetry/crn_key_hashes.json", crn_hashes)
        stage = "native_ddpm_population"; native_pre = v3._model_state_semantic_digest(bound.model); native_callback = _durable_step_callback(
            run_dir, "native_ddpm", governor, 1000, initial_state=latent["z"], path_ids=inventory["path_ids"], requested_labels=inventory["requested_labels"])
        active_callback = native_callback
        native = run_native_ddpm_row(bound.model, bound.schedule, latent["z"], inventory["requested_labels"], inventory["path_ids"], inventory["native_reverse_seeds"], device=args.device, step_callback=native_callback)
        save_native_population(run_dir, native); native_post = v3._model_state_semantic_digest(bound.model); _require(native_post == native_pre, "DDPM model changed during native sampling")
        model_identity.append({"row": "native_ddpm", "pre": {"ddpm": native_pre}, "post": {"ddpm": native_post}, "passed": 1})
        governor.complete("native_ddpm_finalization")
        _clear_durable_step_checkpoint(run_dir, "native_ddpm"); active_callback = None; _record_stage(run_dir, stage)
        historical_post, ddpm_post = v3._model_state_semantic_digest(hist), v3._model_state_semantic_digest(bound.model)
        _require(historical_post == historical_pre and ddpm_post == ddpm_pre, "frozen model state changed during population generation")
        _write_json(run_dir / "telemetry/model_state_identity.json", {"schema": VERSION + "-model-state-identity", "historical_pre": historical_pre, "historical_post": historical_post, "ddpm_pre": ddpm_pre, "ddpm_post": ddpm_post, "populations": model_identity, "passed": 1})
        stage = "population_seal"; governor.admit(stage, predicted_wall_seconds=15.0, predicted_accelerator_seconds=0.0, predicted_next_bytes=2_000_000)
        seal_populations(run_dir); governor.complete(stage); _record_stage(run_dir, stage)
        stage = "machine_scoring"; governor.admit(stage, predicted_wall_seconds=MAX_QUANTUM_SECONDS,
            predicted_accelerator_seconds=MAX_QUANTUM_SECONDS, predicted_next_bytes=16_000_000)
        evaluate_sealed_populations(run_dir, arff=args.arff, ddpm_run_dir=args.ddpm_run_dir, device=args.device)
        governor.complete(stage); _record_stage(run_dir, stage)
        stage = "render_and_review_bundle"; governor.admit(stage, predicted_wall_seconds=MAX_QUANTUM_SECONDS,
            predicted_accelerator_seconds=0.0, predicted_next_bytes=32_000_000)
        prepare_blind_review(run_dir); governor.complete(stage); _record_stage(run_dir, stage); _record_stage(run_dir, "awaiting_human_review")
        _status(run_dir, "awaiting_human_review", route="awaiting_human_review"); stage = "machine_terminalization"; _terminalize(run_dir, governor); return 0
    except BaseException as error:
        if run_dir is None: raise
        if active_callback is not None:
            getattr(active_callback, "persist_latest")()
        if governor is not None: governor.close_open_as_failed()
        state = "resource_stopped" if isinstance(error, ResourceStop) else ("postseal_interrupted" if (run_dir / "populations/POPULATIONS_SEALED.json").exists() else "failed_unsealed")
        sealed = (run_dir / "populations/POPULATIONS_SEALED.json").exists()
        _save_failure_evidence(run_dir, error, stage, governor)
        _status(run_dir, state, route=state, error=str(error), failed_stage=stage, whole_run_restart_required=not sealed)
        _terminalize(run_dir, governor)
        if isinstance(error, KeyboardInterrupt):
            return 130
        if isinstance(error, SystemExit):
            raise
        return 2


def _verify_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "artifact_manifest.json"); _require(manifest["schema"] == VERSION + "-artifact-manifest", "manifest schema changed")
    observed = []
    for path in sorted((x for x in run_dir.rglob("*") if x.is_file() and x.name != "artifact_manifest.json"), key=lambda x: x.relative_to(run_dir).as_posix()):
        observed.append({"path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    _require(observed == manifest["files"] and manifest["artifact_count"] == len(observed) and manifest["artifact_bytes"] == sum(x["bytes"] for x in observed), "manifest inventory changed")
    _require(manifest["tree_digest"] == _sha256_bytes(_canonical_json_bytes(observed)), "manifest tree digest changed"); return manifest


def _verify_failure_evidence(run_dir: Path, status: Mapping[str, Any]) -> None:
    failure = _read_json(run_dir / "failure.json")
    _require(failure["schema"] == VERSION + "-failure" and failure["failed_stage"] == status["failed_stage"]
             and failure["message"] == status["error"] and bool(failure["populations_sealed"]) == (run_dir / "populations/POPULATIONS_SEALED.json").is_file(), "failure receipt changed")
    _require((run_dir / "failure/traceback.txt").is_file() and (run_dir / "failure/traceback.txt").stat().st_size > 0,
             "failure traceback is absent")
    for relative in ("failure/controller_snapshot.npz", "failure/resource_snapshot.json", "failure/model_state_identity.json"):
        _require((run_dir / relative).is_file(), f"failure artifact is absent: {relative}")
    for relative, receipt in failure["authority_receipts"].items():
        path = run_dir / relative; _require(path.is_file() and path.stat().st_size == receipt["bytes"] and sha256_file(path) == receipt["sha256"], f"failure authority changed: {relative}")
    captured = _read_json(run_dir / "failure/resource_snapshot.json")["ledger"]
    current = _read_json(run_dir / "resource_ledger.json")
    if captured is not None:
        _require(current["events"][:len(captured["events"])] == captured["events"] and float(current["wall_seconds"]) >= float(captured["wall_seconds"]), "failure resource snapshot changed")
    last_valid = [run_dir / relative for relative in failure["last_valid_files"]]
    for path in last_valid:
        _require(path.is_file(), "last-valid population evidence is absent")
        data = _npz(path); _require("state" in data and "completed_step" in data and "path_ids" in data and "requested_labels" in data, "last-valid schema changed")
        _require(np.array_equal(data["path_ids"], build_path_inventory()["path_ids"])
                 and np.array_equal(data["requested_labels"], build_path_inventory()["requested_labels"]), "last-valid identities changed")
        png = path.with_suffix(".png"); _require(png.is_file(), "last-valid readable image is absent")
        with Image.open(png) as rendered: rendered.verify()
    controller = _npz(run_dir / "failure/controller_snapshot.npz")
    if last_valid:
        selected = _npz(run_dir / failure["controller_source"]); _require(all(key in controller and np.array_equal(controller[key], value) for key, value in selected.items()), "failure controller snapshot changed")
    else:
        _require(int(controller["available"]) == 0, "failure controller availability changed")
    model = _read_json(run_dir / "failure/model_state_identity.json")
    if (run_dir / "telemetry/model_state_identity.json").exists():
        _require(model == _read_json(run_dir / "telemetry/model_state_identity.json"), "failure model-state snapshot changed")
    elif (run_dir / "preflight/model_immutability.json").exists():
        _require(model == {"schema": VERSION + "-failure-model-state-identity", "available": 1,
            "source": "preflight/model_immutability.json", "receipt": _read_json(run_dir / "preflight/model_immutability.json")},
            "failure preflight model-state snapshot changed")
    else:
        _require(model == {"schema": VERSION + "-failure-model-state-identity", "available": 0},
                 "failure model-state availability changed")


def _verify_preflight(run_dir: Path, config: Mapping[str, Any]) -> None:
    required = ("cpu_smoke", "adapter_orientation", "crn_replay", "model_immutability", "resource_projection", "deterministic_execution")
    production = _is_production_config(config)
    if not production and not all((run_dir / f"preflight/{name}.json").exists() for name in required):
        return
    completed = {item["stage"] for item in _read_json(run_dir / "stage_ledger.json").get("events", [])} if (run_dir / "stage_ledger.json").exists() else set()
    if "inventory_and_start_seal" in completed or any(stage.endswith("_population") for stage in completed):
        expected = set(required)
    elif "cpu_smoke_replay" in completed:
        expected = {"cpu_smoke", "adapter_orientation", "crn_replay", "resource_projection"}
    else:
        expected = {name for name in required if (run_dir / f"preflight/{name}.json").exists()}
    for name in expected: _require((run_dir / f"preflight/{name}.json").is_file(), f"preflight receipt is absent: {name}")
    if "cpu_smoke" in expected:
        smoke = _read_json(run_dir / "preflight/cpu_smoke.json")
        _require(smoke["passed"] == 1 and smoke["scientific_evidence"] == 0 and smoke["external_authorities_opened"] == 0, "CPU smoke receipt changed")
    if "adapter_orientation" in expected:
        orientation = _read_json(run_dir / "preflight/adapter_orientation.json")
        _require(orientation == {"schema": VERSION + "-adapter-orientation", "time_map": "linear_remaining_fraction_round",
            "flux_orientation": "horizontal_then_vertical_periodic", "poisson_residual_maximum": POISSON_RESIDUAL_MAXIMUM,
            "cpu_smoke_adapter_passed": 1, "passed": 1}, "adapter orientation receipt changed")
    if "crn_replay" in expected:
        seeds = np.asarray([derive_edge_noise_seed(path_id, 0, 1, 0) for path_id in build_path_inventory()["path_ids"].tolist()], dtype=np.uint64)
        _require(_read_json(run_dir / "preflight/crn_replay.json") == {"schema": VERSION + "-crn-replay", "byteorder": "big",
            "root": EULERIAN_EDGE_NOISE_ROOT, "first_step_sha256": _sha256_bytes(seeds.tobytes()), "exact_replay": 1, "passed": 1}, "CRN preflight changed")
    if "resource_projection" in expected:
        projection = _read_json(run_dir / "preflight/resource_projection.json"); authority = config["execution_authority"]
        _require(projection["schema"] == VERSION + "-resource-projection" and projection["fits_frozen_maxima"] == 1
                 and projection["fits_execution_authority"] == 1 and projection["projected_wall_seconds"] >= 1_500.0
                 and projection["projected_accelerator_seconds"] >= 900.0 and projection["projected_storage_bytes"] >= 100 * 1024**2
                 and projection["projected_wall_seconds"] + authority["reserve_seconds"] <= authority["max_wall_seconds"]
                 and projection["projected_accelerator_seconds"] <= authority["max_accelerator_seconds"]
                 and projection["projected_storage_bytes"] <= authority["max_storage_bytes"], "resource projection changed")
    if "deterministic_execution" in expected:
        _require(_read_json(run_dir / "preflight/deterministic_execution.json") == {"deterministic_algorithms": 1, "cudnn_benchmark": 0,
            "cudnn_deterministic": 1, "tf32_policy_changed_by_runner": 0}, "deterministic execution receipt changed")
    if "model_immutability" in expected:
        model = _read_json(run_dir / "preflight/model_immutability.json")
        _require(model["schema"] == VERSION + "-model-immutability" and model["all_parameters_require_grad_false"] == 1
                 and model["training_forbidden"] == 1 and model["passed"] == 1, "model immutability preflight changed")


def _verify_run_impl(run_dir: Path, *, allow_pending_receipt: bool = False,
                     allow_terminal_open: bool = False) -> dict[str, Any]:
    root = run_dir.resolve(); _require(root.is_dir(), "run directory is absent")
    before = [(p.relative_to(root).as_posix(), p.stat().st_size, sha256_file(p)) for p in sorted((x for x in root.rglob("*") if x.is_file()), key=lambda x: x.relative_to(root).as_posix())]
    manifest = _verify_manifest(root); config = _read_json(root / "config.json"); status = _read_json(root / "status.json")
    _require(config["schema"] == VERSION + "-config" and config["version"] == VERSION and config["research_mode"] == RESEARCH_MODE, "config authority changed")
    science = config["scientific_configuration"]; _require(science["path_count"] == PATH_COUNT and science["paths_per_class"] == PATHS_PER_CLASS and science["outer_steps"] == OUTER_STEPS and science["anchors"] == list(ANCHORS), "scientific configuration changed")
    _require(science["mass_scale"] == {"numerator": MASS_SCALE_NUMERATOR, "denominator": MASS_SCALE_DENOMINATOR, "hex": MASS_SCALE_HEX}, "mass transform changed")
    _require(science["native_anchors"] == list(NATIVE_ANCHORS) and science["adapter"] == {"num_ddpm_steps": 1000, "beta_start": 0.0001, "beta_end": 0.02, "min_tau_fraction": 0.03, "time_map": "linear_remaining_fraction_round", "latent_policy": "persistent_path_latent", "flux_projection": "periodic_minimum_energy_minus_free"}, "adapter/native schedule changed")
    authority = config["execution_authority"]; ResourceBudget(**{key: authority[key] for key in ("max_wall_seconds", "max_accelerator_seconds", "max_storage_bytes", "max_cuda_fraction", "reserve_seconds", "maximum_quantum_seconds")})
    if _is_production_config(config):
        approval = str(authority.get("approval_id", ""))
        _require(authority.get("device") == "cuda:0" and len(approval) >= 12 and "<" not in approval
                 and "placeholder" not in approval.lower(), "production execution authority changed")
        _require(config["command"] == subprocess.list2cmdline(config["argv"]), "canonical production command changed")
    sources = _read_json(root / "source_bindings.json"); _require(Path(sources["repository_root"]).resolve() == _repository_root(), "source root changed")
    _require(sources == _source_bindings(_repository_root()), "source bindings changed")
    sealed = (root / "populations/POPULATIONS_SEALED.json").is_file()
    bindings = _read_json(root / "checkpoint_bindings.json") if (root / "checkpoint_bindings.json").exists() else None
    if bindings:
        for key in ("legacy_checkpoint", "arff", "ddpm_checkpoint", "evaluator_checkpoint", "evaluator_selection"):
            row = bindings[key]; path = Path(row["path"]); expected = row.get("sha256", row.get("expected_sha256"))
            if key.startswith("evaluator_") and not sealed:
                _require(row.get("opened_preseal") == 0 and expected in {EVALUATOR_SHA256, EVALUATOR_SELECTION_SHA256}, f"deferred evaluator authority changed: {key}")
            else: _require(path.is_file() and sha256_file(path) == expected and ("bytes" not in row or path.stat().st_size == row["bytes"]), f"bound authority changed: {key}")
        bound = bindings.get("ddpm_bound", {})
        for prefix in ("checkpoint", "selection", "config", "schedule"):
            path = Path(bound[f"{prefix}_path"]); _require(path.is_file() and path.stat().st_size == int(bound[f"{prefix}_bytes"]) and sha256_file(path) == bound[f"{prefix}_sha256"], f"DDPM bound {prefix} changed")
    stages = _read_json(root / "stage_ledger.json").get("events", []) if (root / "stage_ledger.json").exists() else []
    names = [x["stage"] for x in stages]
    _require(names == list(STAGE_ORDER[:len(names)]) and all(x.get("state") == "completed" for x in stages), "stage order changed")
    _require(status["schema"] == VERSION + "-status" and status["state"] in {"failed_unsealed", "resource_stopped", "postseal_interrupted", "awaiting_human_review", "complete"}, "status changed")
    if status["state"] == "awaiting_human_review":
        _require(names == list(STAGE_ORDER[:-1]), "awaiting stage ledger changed")
    elif status["state"] == "complete":
        _require(names == list(STAGE_ORDER), "complete stage ledger changed")
    else:
        failed_stage = status.get("failed_stage")
        if failed_stage in STAGE_ORDER:
            failed_index = STAGE_ORDER.index(failed_stage)
            _require(failed_index == len(names) or (names and failed_stage == names[-1]), "failure stage ledger changed")
    ledger = _verify_resource_ledger(root, config, status, allow_terminal_open=allow_terminal_open)
    _verify_preflight(root, config)
    if status["state"] in {"failed_unsealed", "resource_stopped", "postseal_interrupted"}:
        _verify_failure_evidence(root, status)
    if sealed:
        seal = _verify_population_seal(root); inventory = build_path_inventory(); starts = _npz(root / "inventory/start_bank.npz"); latent = _npz(root / "inventory/ddpm_latent_bank.npz")
        _require(set(starts) == {"schema", "version", "masses", "path_ids", "requested_labels", "source_seeds", "scientific_sha256"}, "start-bank schema changed")
        _require(set(latent) == {"schema", "version", "z", "path_ids", "requested_labels", "latent_seeds", "mean", "standard_deviation", "minimum", "maximum", "z_sha256", "scientific_sha256"}, "latent-bank schema changed")
        _require(np.array_equal(starts["source_seeds"], inventory["source_seeds"]) and str(starts["scientific_sha256"]) == _scientific_digest({"masses": starts["masses"], "path_ids": starts["path_ids"], "requested_labels": starts["requested_labels"], "source_seeds": starts["source_seeds"]}, {}), "start-bank semantic authority changed")
        _require(np.array_equal(latent["latent_seeds"], inventory["ddpm_latent_seeds"]) and str(latent["z_sha256"]) == _hash_array(latent["z"]), "latent semantic authority changed")
        if bindings and "historical_receipt" in bindings:
            frozen_config = DirectFluxMNISTConfig(**bindings["historical_receipt"]["config"])
            _require(np.array_equal(build_start_bank(frozen_config, inventory), starts["masses"]), "deterministically rebuilt start bank changed")
            rebuilt_latent = build_ddpm_latent_bank(inventory); _require(np.array_equal(rebuilt_latent["z"], latent["z"]), "deterministically rebuilt latent bank changed")
            development_images, development_labels, _ = read_mnist_development_prefix(bindings["arff"]["path"])
            rebuilt_teacher = build_teacher_target_bank(development_images[VALIDATION_START:VALIDATION_STOP], development_labels[VALIDATION_START:VALIDATION_STOP], inventory)
            saved_teacher = _npz(root / "inventory/teacher_target_bank.npz")
            for key in ("masses", "source_images_uint8", "rendered_images_uint8", "path_ids", "requested_labels", "validation_local_ids", "arff_global_row_ids"):
                _require(np.array_equal(saved_teacher[key], rebuilt_teacher[key]), f"deterministically rebuilt teacher bank changed: {key}")
        for row in ROWS:
            raw = _npz(root / f"populations/{row}.npz"); telemetry = json.loads(str(raw["telemetry_scientific_json"])); expected_count = 1000 if row == "native_ddpm" else 256
            _require(len(telemetry) == expected_count and [int(x["completed_step"]) for x in telemetry] == list(range(1, expected_count + 1)), f"{row} telemetry sequence changed")
            if row == "native_ddpm": _require([int(x["ddpm_timestep"]) for x in telemetry] == list(range(999, -1, -1)), "native timestep sequence changed")
            else:
                expected_keys = []
                for outer, record in enumerate(telemetry):
                    for attempt in record["attempts"]:
                        for sub in range(int(attempt["substeps"])):
                            seeds = np.asarray([derive_edge_noise_seed(pid, outer, int(attempt["substeps"]), sub) for pid in inventory["path_ids"].tolist()], dtype=np.uint64)
                            expected_keys.append(_sha256_bytes(seeds.tobytes()))
                _require(raw["crn_key_hashes"].tolist() == expected_keys, f"{row} CRN authority changed")
        adapter_paths = _npz(root / "telemetry/adapter_paths.npz"); _require("predicted_mass" in adapter_paths and adapter_paths["predicted_mass"].shape == (4, PATH_COUNT, 784), "adapter predicted-mass authority changed")
        _require(np.array_equal(adapter_paths["anchor_steps"], np.asarray(ANCHORS[1:], dtype=np.int64)), "adapter path anchors changed")
        _require(np.array_equal(adapter_paths["path_ids"], inventory["path_ids"]) and np.array_equal(adapter_paths["requested_labels"], inventory["requested_labels"]), "adapter path identities changed")
        for key, bank in adapter_paths.items():
            if bank.dtype.kind in "biufc":
                _require(bool(np.isfinite(bank).all()), f"adapter path telemetry is nonfinite: {key}")
        if status["state"] in {"awaiting_human_review", "complete"}:
            for path in ("evaluation/SCORING_READY.json", "evaluation/per_anchor_metrics.csv", "evaluation/per_class_metrics.csv", "evaluation/predictions.npz", "gates.json", "review/REVIEW_READY.json", "review/private_key.json", "review/human_review_template.csv", "review/blinded_historical_vs_adapter.png"):
                _require((root / path).is_file(), f"terminal artifact is absent: {path}")
            ready = _read_json(root / "review/REVIEW_READY.json"); _require(ready["key_sha256"] == sha256_file(root / "review/private_key.json") and ready["count"] == 80, "review bundle changed")
            final_images = []
            for row in ROWS:
                population = _npz(root / f"populations/{row}_uint8.npz"); images = population["images_uint8"]; ids = population["path_ids"].astype(np.str_)
                v3._verify_sheet_pixels(root / f"images/{row}_final.png", images[-1], columns=10, scale=3, captions=ids.tolist())
                v3._verify_sheet_pixels(root / f"images/{row}_trajectory.png", images.transpose(1, 0, 2, 3).reshape(-1, 28, 28), columns=5, scale=2,
                    captions=[f"{pid}@{step}" for pid in ids for step in population["anchor_steps"].tolist()]); final_images.append(images[-1])
            v3._verify_sheet_pixels(root / "images/comparison_final.png", np.concatenate(final_images), columns=10, scale=3,
                captions=[f"{row}:{index:03d}" for row in ROWS for index in range(PATH_COUNT)])
            key = _read_json(root / "review/private_key.json")["items"]; sources = {row: _npz(root / f"populations/{row}_uint8.npz")["images_uint8"][-1] for row in ("historical", "ddpm_eulerian")}
            ordered = np.stack([sources[item["row"]][list(build_path_inventory()["path_ids"]).index(item["path_id"])] for item in key])
            v3._verify_sheet_pixels(root / "review/blinded_historical_vs_adapter.png", ordered, columns=8, scale=4, captions=[item["blind_id"] for item in key])
            evaluation = _verify_evaluation(root, config, bindings)
            review_evidence = _verify_review_bundle(root, require_answers=status["state"] == "complete")
        if status["state"] == "awaiting_human_review":
            _require(not (root / "review/human_review_answers.csv").exists() and not (root / "outcome.json").exists(), "awaiting route contains opened human evidence")
            attempts_path = root / "review/submission_attempts.json"
            if attempts_path.exists():
                attempts = _read_json(attempts_path)
                _require(attempts["schema"] == VERSION + "-review-submission-attempts" and attempts["attempts"]
                         and all(item["passed"] == 0 and item["error"] for item in attempts["attempts"]), "invalid review submission log changed")
        if status["state"] == "complete":
            for path in ("review/human_review_answers.csv", "review/human_review.json", "review/human_review_by_row.json", "outcome.json"):
                _require((root / path).is_file(), f"complete artifact is absent: {path}")
            human = review_evidence["summaries"]["ddpm_eulerian"]
            expected_human_gate = {"gate_type": "diagnostic threshold", "state": "complete",
                "passed": int(human["recognizability"] >= CANDIDATE_HUMAN_RECOGNIZABILITY_MINIMUM and human["requested_label_agreement"] >= CANDIDATE_HUMAN_AGREEMENT_MINIMUM), "values": human}
            _require(evaluation["gates"]["candidate_human_fidelity"] == expected_human_gate, "human fidelity gate changed")
            route_inputs = _route_inputs(evaluation["gates"], evaluation["comparison"], human); route = _outcome_route(**route_inputs)
            outcome = _read_json(root / "outcome.json")
            _require(outcome == {"schema": VERSION + "-outcome", "route": route, "next_action": _route_action(route),
                "full_scale_auto_launched": 0, "route_inputs": route_inputs, "human": review_evidence["summaries"],
                "machine_comparison": evaluation["comparison"], "claim_scope": "exploratory 40-path pilot only"}, "outcome route changed")
            attempts = _read_json(root / "review/submission_attempts.json")
            history = attempts.get("attempts", [])
            expected_attempt_keys = {"reviewer", "answers_path", "answers_sha256", "passed", "error", "recorded_at"}
            _require(attempts.get("schema") == VERSION + "-review-submission-attempts" and history
                     and all(set(item) == expected_attempt_keys and isinstance(item["reviewer"], str)
                             and isinstance(item["answers_path"], str) and item["passed"] in (0, 1)
                             and isinstance(item["recorded_at"], str) for item in history)
                     and all(item["passed"] == 0 and isinstance(item["error"], str) and item["error"] for item in history[:-1])
                     and history[-1]["passed"] == 1 and history[-1]["error"] is None
                     and history[-1]["answers_sha256"] == sha256_file(root / "review/human_review_answers.csv")
                     and history[-1]["reviewer"] == review_evidence["reviewer"], "accepted review submission log changed")
    else:
        _require(status["state"] in {"failed_unsealed", "resource_stopped"} and (root / "failure.json").is_file() and status["whole_run_restart_required"] == 1, "unsealed terminal route changed")
    for path in ("REPORT.md", "DELIBERATE_OMISSIONS.md", "VERIFY_RECEIPT.json"):
        _require((root / path).is_file(), f"terminal report artifact is absent: {path}")
    receipt = _read_json(root / "VERIFY_RECEIPT.json")
    _require(receipt.get("schema") == VERSION + "-verify-receipt" and receipt.get("written_by") == "terminal-semantic-verifier", "verification receipt authority changed")
    if allow_pending_receipt:
        _require(receipt.get("state") == "pending" and receipt.get("passed") == 0, "pending verification receipt changed")
    else:
        _require(receipt.get("state") == "complete" and receipt.get("passed") == 1 and receipt.get("verified_status") == status["state"], "terminal verification receipt changed")
    outcome_for_report = _read_json(root / "outcome.json") if (root / "outcome.json").exists() else None
    _require((root / "REPORT.md").read_text(encoding="utf-8") == _report_text(root, outcome_for_report), "REPORT semantic content changed")
    _require((root / "DELIBERATE_OMISSIONS.md").read_text(encoding="utf-8") == "# Deliberate omissions\n\nNo new training, gain/schedule search, candidate selection, terminal-data calibration, native-to-adapter target injection, or automatic follow-up compute was performed.\n\nNPZ self-description is scoped: raw populations and sealed inventory banks carry scientific digests; derived uint8, adapter-telemetry, prediction, and failure NPZs are authoritative only through the seal/manifest plus semantic replay, so they are not independently self-certifying.\n", "deliberate omissions changed")
    after = [(p.relative_to(root).as_posix(), p.stat().st_size, sha256_file(p)) for p in sorted((x for x in root.rglob("*") if x.is_file()), key=lambda x: x.relative_to(root).as_posix())]
    _require(before == after, "verification mutated the run tree")
    return {"schema": VERSION + "-verification", "passed": 1, "state": status["state"], "artifact_count": manifest["artifact_count"], "tree_digest": manifest["tree_digest"]}


def run_cpu_smoke() -> dict[str, Any]:
    """Bounded synthetic assembled smoke; implemented below without external evidence."""
    return _run_cpu_smoke_impl()


def bind_real_authorities(*, legacy_checkpoint: str | Path, ddpm_run_dir: str | Path,
                          arff: str | Path) -> dict[str, Any]:
    """Read-only generation preflight; evaluator bytes remain deferred until seal."""
    return _bind_real_authorities_impl(Path(legacy_checkpoint), Path(ddpm_run_dir), Path(arff), include_evaluator=False)


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    """Strict read-only semantic verification of a sealed or failed lifecycle."""
    root = Path(run_dir).resolve()
    return _verify_run_impl(root, allow_terminal_open=root in _CHARGED_TERMINAL_PRECHECKS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="run the bounded synthetic CPU smoke")
    smoke.add_argument("--device", default="cpu", choices=("cpu",))
    run = sub.add_parser("run", help="run or resume the separately approved production pilot")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--legacy-checkpoint", required=True)
    run.add_argument("--ddpm-run-dir", required=True)
    run.add_argument("--arff", required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--approval-id", required=True)
    run.add_argument("--max-wall-seconds", type=float, required=True)
    run.add_argument("--max-accelerator-seconds", type=float, required=True)
    run.add_argument("--max-cuda-fraction", type=float, required=True)
    run.add_argument("--max-storage-mib", type=float, required=True)
    run.add_argument("--resume-post-seal", action="store_true")
    verify = sub.add_parser("verify", help="verify a run without writing")
    verify.add_argument("--run-dir", required=True)
    review = sub.add_parser("finalize-review", help="finalize an already sealed manual review")
    review.add_argument("--run-dir", required=True)
    review.add_argument("--answers", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--confirm-manual-review", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(run_cpu_smoke(), sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(verify_run(args.run_dir), sort_keys=True))
        return 0
    if args.command == "finalize-review":
        return finalize_review(args)
    return run_production(args)


if __name__ == "__main__":
    raise SystemExit(main())
