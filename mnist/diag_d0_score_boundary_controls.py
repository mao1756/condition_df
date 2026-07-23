"""Boundary-admissible controls-only repair for Experiment 12 D0.

The command binds the completed failed implicit-score run, exercises the
closed-simplex model/operator domain, and trains only exact synthetic controls.
It never loads physical score states into a training task and never imports a
reverse sampler.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_dirichlet_score import (
    D0LinearSplinePotential,
    dirichlet_score_objective,
    edge_difference_channels,
    fit_linear_spline_baseline,
    harmonic_mobility_exact,
    physical_flux_from_edge_score,
)
from mnist.d0_score_boundary_controls import (
    BOUNDARY_SMOOTH_MODEL_VERSION,
    BOUNDED_TEACHER_VERSION,
    ORTHOGONAL_HADAMARD_PROBE_VERSION,
    D0BoundarySmoothPotentialUNet,
    bounded_teacher_edge_score,
    orthogonal_hadamard_edge_probes,
    run_boundary_operator_preflight,
    sample_bounded_teacher_mixture,
)
from mnist.d0_score_boundary_control_gate import (
    BoundaryControlThresholds,
    evaluate_boundary_control_gates,
    evaluate_boundary_preflight,
    evaluate_implicit_teacher_seed,
    evaluate_implicit_teacher_study,
    evaluate_null_seed,
    evaluate_null_study,
    evaluate_supervised_teacher,
    select_dual_bank_checkpoint,
)
from mnist.d0_score_optimizer_scale import (
    scaled_backward_and_clip,
    summarize_scaled_gradient_history,
)
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    array_fingerprint,
    atomic_copy_file,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    capture_rng_state,
    configure_exact_torch_backend,
    config_fingerprint,
    file_fingerprint,
    restore_rng_state,
    source_fingerprint,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    init_ema_state,
    natural_horizon,
    temporary_ema_weights,
    update_ema_state,
)


RUN_SCHEMA = "experiment12-d0-score-boundary-controls"
RUN_SCHEMA_VERSION = 1
MODEL_SCHEMA = BOUNDARY_SMOOTH_MODEL_VERSION
MODEL_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = RUN_SCHEMA + "-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
CONTROL_ARRAY_SCHEMA = RUN_SCHEMA + "-synthetic-arrays"
CONTROL_ARRAY_SCHEMA_VERSION = 1
CLAIM_SCOPE = "boundary-admissible exact synthetic implicit-score controls only"
EXPECTED_FAILED_SCORE_SCIENTIFIC_FINGERPRINT = (
    "9a19131cda6b52c63f6b93f934c1bc702b71a44efb9bdee1c4b050f306a36f14"
)

EXPECTED_KERNEL: dict[str, Any] = {
    "grid_size": 28,
    "sample_steps": 512,
    "reference_substeps": 256,
    "tau_eff": 5e-5,
    "edge_alpha_mode": "alpha_eff",
    "alpha_eff": 1.0,
    "mass_floor": 1e-7,
    "limiter_fraction": 1.0,
    "lambda_mix": 0.35,
}

DEFAULTS: dict[str, Any] = {
    **EXPECTED_KERNEL,
    "train_paths": 128,
    "selection_paths": 32,
    "audit_paths": 32,
    "anchors_per_path": 32,
    "anchor_bin_counts": (4, 4, 4, 4, 16),
    "base_channels": 32,
    "batch_size": 64,
    "validation_batch_size": 64,
    "train_steps": 4000,
    "validation_every": 250,
    "checkpoint_every": 250,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "ema_decay": 0.99,
    "grad_clip": 1.0,
    "clip_warmup_steps": 500,
    "training_probes": 4,
    "selection_probes": 16,
    "audit_probes": 64,
    "bootstrap_reps": 10_000,
    "bootstrap_confidence": 0.90,
    "operator_hutchinson_probes": 4096,
    "teacher_seeds": (260771, 260772, 260773),
    "null_seeds": (260771, 260772, 260773),
    "supervised_seed": 260770,
    "teacher_data_seed": 260767,
    "null_data_seed": 260768,
    "calibration_seed": 260769,
    "training_probe_seed": 260774,
    "selection_probe_a_seed": 260775,
    "selection_probe_b_seed": 260776,
    "audit_probe_a_seed": 260777,
    "audit_probe_b_seed": 260778,
    "bootstrap_seed": 260779,
    "batch_index_seed": 260780,
}


def _json_load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _parse_csv_ints(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        result = tuple(int(item) for item in value)
    if not result:
        raise ValueError("at least one integer is required")
    return result


def _semantic_close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _device(value: str | None) -> torch.device:
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _make_dynamics(args: argparse.Namespace) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=int(args.grid_size), alpha=1.0, beta=1.0,
        alpha_eff=float(args.alpha_eff), edge_alpha_mode=str(args.edge_alpha_mode),
        horizon_scale=1.0, num_steps=int(args.sample_steps),
        limiter_fraction=float(args.limiter_fraction), mass_floor=float(args.mass_floor),
        source_lowfreq_size=min(7, int(args.grid_size)), source_blur_sigma=0.0,
        source_uniform_mix=0.15, source_concentration=1.0,
        condition_on_source=False, flux_parameterization="edge",
        ot_lowres_size=min(7, int(args.grid_size)), ot_blur_sigma=0.0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "controls", "report", "all"), default="all")
    parser.add_argument("--failed-score-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/experiment12_d0_score_boundary_controls"))
    parser.add_argument("--run-name", default="production-boundary-admissible-controls")
    parser.add_argument("--require-gate", choices=("none", "preflight", "controls"), default="none")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--grid-size", type=int, default=28)
    parser.add_argument("--sample-steps", type=int, default=512)
    parser.add_argument("--reference-substeps", type=int, default=256)
    parser.add_argument("--tau-eff", type=float, default=5e-5)
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff",), default="alpha_eff")
    parser.add_argument("--alpha-eff", type=float, default=1.0)
    parser.add_argument("--mass-floor", type=float, default=1e-7)
    parser.add_argument("--limiter-fraction", type=float, default=1.0)
    parser.add_argument("--lambda-mix", type=float, default=0.35)

    parser.add_argument("--train-paths", type=int, default=128)
    parser.add_argument("--selection-paths", type=int, default=32)
    parser.add_argument("--audit-paths", type=int, default=32)
    parser.add_argument("--anchors-per-path", type=int, default=32)
    parser.add_argument("--anchor-bin-counts", default="4,4,4,4,16")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=4000)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--clip-warmup-steps", type=int, default=500)
    parser.add_argument("--training-probes", type=int, default=4)
    parser.add_argument("--selection-probes", type=int, default=16)
    parser.add_argument("--audit-probes", type=int, default=64)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.90)

    parser.add_argument("--teacher-seeds", default="260771,260772,260773")
    parser.add_argument("--null-seeds", default="260771,260772,260773")
    parser.add_argument("--supervised-seed", type=int, default=260770)
    parser.add_argument("--teacher-data-seed", type=int, default=260767)
    parser.add_argument("--null-data-seed", type=int, default=260768)
    parser.add_argument("--calibration-seed", type=int, default=260769)
    parser.add_argument("--training-probe-seed", type=int, default=260774)
    parser.add_argument("--selection-probe-a-seed", type=int, default=260775)
    parser.add_argument("--selection-probe-b-seed", type=int, default=260776)
    parser.add_argument("--audit-probe-a-seed", type=int, default=260777)
    parser.add_argument("--audit-probe-b-seed", type=int, default=260778)
    parser.add_argument("--bootstrap-seed", type=int, default=260779)
    parser.add_argument("--batch-index-seed", type=int, default=260780)
    parser.add_argument("--operator-hutchinson-probes", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        args.anchor_bin_counts = _parse_csv_ints(args.anchor_bin_counts)
        args.teacher_seeds = _parse_csv_ints(args.teacher_seeds)
        args.null_seeds = _parse_csv_ints(args.null_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.anchor_bin_counts) != 5 or sum(args.anchor_bin_counts) != int(args.anchors_per_path):
        parser.error("anchor-bin-counts must contain five values summing to anchors-per-path")
    if len(args.teacher_seeds) != 3 or len(set(args.teacher_seeds)) != 3:
        parser.error("teacher-seeds must contain three distinct seeds")
    if len(args.null_seeds) != 3 or len(set(args.null_seeds)) != 3:
        parser.error("null-seeds must contain three distinct seeds")
    positive = (
        "train_paths", "selection_paths", "audit_paths", "anchors_per_path",
        "base_channels", "batch_size", "validation_batch_size", "train_steps",
        "validation_every", "checkpoint_every", "training_probes",
        "selection_probes", "audit_probes", "bootstrap_reps",
        "operator_hutchinson_probes",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < float(args.ema_decay) < 1.0:
        parser.error("ema-decay must be in (0,1)")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        parser.error("bootstrap-confidence must be in (0,1)")
    if args.require_gate != "none":
        mismatches = []
        for key, expected in DEFAULTS.items():
            if not _semantic_close(getattr(args, key), expected):
                mismatches.append(f"{key}={getattr(args, key)!r}, expected {expected!r}")
        if mismatches:
            parser.error("required production gate rejects overrides: " + "; ".join(mismatches))
    return args


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": file_fingerprint(path), "size": int(path.stat().st_size)}


def verify_failed_score_run(path: str | Path) -> dict[str, Any]:
    """Verify the exact failed-run evidence authorized by this repair patch."""

    run_dir = Path(path)
    required = {
        "manifest": run_dir / "run_manifest.json",
        "status": run_dir / "run_status.json",
        "preflight": run_dir / "preflight_gate.json",
        "cache": run_dir / "cache_gate.json",
        "controls": run_dir / "controls_gate.json",
        "operator": run_dir / "operator_preflight.json",
        "cache_index": run_dir / "cache" / "parent" / "cache_index.json",
    }
    for item in required.values():
        if not item.is_file():
            raise FileNotFoundError(item)
    values = {name: _json_load(item) for name, item in required.items()}
    status = values["status"]
    manifest = values["manifest"]
    controls = values["controls"]
    if manifest.get("schema") != "experiment12-d0-dirichlet-score-learnability":
        raise ArtifactCompatibilityError("failed score parent has an incompatible schema")
    parent_scientific_fingerprint = str(manifest.get("scientific_fingerprint", ""))
    if parent_scientific_fingerprint != EXPECTED_FAILED_SCORE_SCIENTIFIC_FINGERPRINT:
        raise ArtifactCompatibilityError(
            "failed score parent is not the frozen production control-failure run"
        )
    if config_fingerprint(dict(manifest.get("scientific_config", {}))) != parent_scientific_fingerprint:
        raise ArtifactCompatibilityError("failed score parent scientific fingerprint is internally inconsistent")

    registry_record = dict(dict(manifest.get("artifacts", {})).get("artifact_registry", {}))
    registry_path = run_dir / "artifact_registry.json"
    if (
        not registry_path.is_file()
        or Path(str(registry_record.get("path", ""))).resolve() != registry_path.resolve()
        or registry_record.get("sha256") != file_fingerprint(registry_path)
        or int(registry_record.get("size", -1)) != int(registry_path.stat().st_size)
    ):
        raise ArtifactCompatibilityError("failed score parent artifact registry is not manifest-bound")
    registry = _json_load(registry_path)
    registry_records = dict(registry.get("records", {}))

    def verify_registry_record(artifact: Path) -> None:
        try:
            relative = artifact.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise ArtifactCompatibilityError(
                f"failed parent artifact escapes its run directory: {artifact}"
            ) from exc
        record = dict(registry_records.get(relative, {}))
        if (
            not record
            or Path(str(record.get("path", ""))).resolve() != artifact.resolve()
            or record.get("sha256") != file_fingerprint(artifact)
            or int(record.get("size", -1)) != int(artifact.stat().st_size)
        ):
            raise ArtifactCompatibilityError(
                f"failed parent artifact disagrees with its registry: {relative}"
            )

    for name, artifact in required.items():
        # The parent registry deliberately excludes its self-referential
        # manifest; the manifest instead binds the registry record above.
        if name != "manifest":
            verify_registry_record(artifact)
    expected_status = {
        "status": "complete", "outcome": "gate_failed",
        "decision": "optimization_pipeline_invalid", "sampling_performed": 0,
    }
    for key, expected in expected_status.items():
        if status.get(key) != expected:
            raise ArtifactCompatibilityError(f"failed score parent status mismatch for {key}")
    if int(values["preflight"].get("passed", 0)) != 1 or int(values["cache"].get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("failed score parent did not pass preflight and cache")
    if int(controls.get("passed", 1)) != 0:
        raise ArtifactCompatibilityError("failed score parent controls unexpectedly passed")
    if int(dict(controls.get("teacher", {})).get("passed", 1)) != 0:
        raise ArtifactCompatibilityError("failed score parent teacher was not the recorded failure")
    if int(dict(controls.get("null", {})).get("passed", 1)) != 0:
        raise ArtifactCompatibilityError("failed score parent null was not the recorded failure")
    scientific = dict(manifest.get("scientific_config", {}))
    kernel = dict(scientific.get("kernel", manifest.get("parent_confirmation", {}).get("kernel", {})))
    if not kernel:
        kernel = dict(dict(manifest.get("parent_confirmation", {})).get("kernel", {}))
    for key, expected in EXPECTED_KERNEL.items():
        actual = kernel.get(key)
        if not _semantic_close(actual, expected):
            raise ArtifactCompatibilityError(f"failed score parent kernel mismatch for {key}: {actual!r}")
    for name, record in dict(controls.get("evidence", {})).items():
        artifact = Path(str(record.get("path", "")))
        if not artifact.is_file() or file_fingerprint(artifact) != record.get("sha256"):
            raise ArtifactCompatibilityError(f"failed control evidence hash mismatch for {name}")
        verify_registry_record(artifact)
    physical_files = list((run_dir / "tasks").glob("seed-*/*")) if (run_dir / "tasks").is_dir() else []
    if physical_files:
        raise ArtifactCompatibilityError("failed parent contains physical task evidence")
    for name in ("audit_path_score_risks.csv", "stein_path_metrics.csv", "score_seed_metrics.csv"):
        artifact = run_dir / name
        if artifact.is_file() and artifact.stat().st_size > 1:
            with artifact.open("r", encoding="utf-8", newline="") as handle:
                if any(True for _ in csv.DictReader(handle)):
                    raise ArtifactCompatibilityError(f"failed parent performed physical evaluation: {name}")
    cache_index = values["cache_index"]
    schedule = dict(dict(cache_index.get("metadata", {})).get("schedule_metadata", {}))
    if not schedule or not _finite_positive(schedule.get("horizon")):
        raise ArtifactCompatibilityError("failed parent cache has no valid time-plan provenance")
    return {
        "passed": 1,
        "run_dir": str(run_dir.resolve()),
        "scientific_fingerprint": parent_scientific_fingerprint,
        "kernel": kernel,
        "schedule_metadata": schedule,
        "artifacts": {
            **{name: _artifact_record(item) for name, item in required.items()},
            "artifact_registry": _artifact_record(registry_path),
        },
        "failed_teacher_gate": dict(controls.get("teacher", {})),
        "failed_null_gate": dict(controls.get("null", {})),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _finite_positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ControlArrays:
    states: Tensor
    tau: Tensor
    tau_fraction: Tensor
    labels: Tensor
    path_ids: np.ndarray
    strata: np.ndarray
    role: str
    law: str
    horizon: float

    def __post_init__(self) -> None:
        rows = int(self.states.shape[0])
        if self.states.ndim != 2 or rows <= 0:
            raise ValueError("control states must be a nonempty matrix")
        if any(value.shape != (rows,) for value in (self.tau, self.tau_fraction, self.labels)):
            raise ValueError("control tensor row counts disagree")
        if any(np.asarray(value).shape != (rows,) for value in (self.path_ids, self.strata)):
            raise ValueError("control path rows disagree")
        if not bool(torch.isfinite(self.states).all() and (self.states > 0).all()):
            raise ValueError("control states must be finite and strictly positive")
        mass_error = float((self.states.sum(1) - 1.0).abs().max())
        if mass_error > 2e-5:
            raise ValueError("control states are not simplex-valued")


def _time_template(bin_counts: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    fractions: list[float] = []
    strata: list[int] = []
    for bin_index, count in enumerate(bin_counts):
        for offset in range(int(count)):
            fractions.append((float(bin_index) + (offset + 0.5) / float(count)) / 5.0)
            strata.append(bin_index)
    return np.asarray(fractions, dtype=np.float32), np.asarray(strata, dtype=np.int64)


def _sample_null(fractions: Tensor, pixels: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    concentration = torch.ones((fractions.numel(), int(pixels)), dtype=torch.float64)
    raw = torch._standard_gamma(concentration, generator=generator)
    return (raw / raw.sum(dim=1, keepdim=True)).float()


def _build_control_arrays(
    *, role: str, law: str, path_count: int, first_path_id: int,
    bin_counts: Sequence[int], horizon: float, grid_size: int, seed: int,
) -> ControlArrays:
    per_path_fraction, per_path_strata = _time_template(bin_counts)
    fractions_np = np.tile(per_path_fraction, int(path_count))
    strata = np.tile(per_path_strata, int(path_count))
    path_ids = np.repeat(
        np.arange(int(first_path_id), int(first_path_id) + int(path_count), dtype=np.int64),
        per_path_fraction.size,
    )
    fractions = torch.from_numpy(fractions_np.copy()).float()
    if law == "bounded_teacher":
        states = sample_bounded_teacher_mixture(
            fractions, int(grid_size), seed=int(seed), device="cpu", dtype=torch.float32
        )
    elif law == "dirichlet_null":
        states = _sample_null(fractions, int(grid_size) ** 2, int(seed))
    else:
        raise ValueError(f"unknown synthetic law {law!r}")
    return ControlArrays(
        states=states.contiguous(), tau=(fractions * float(horizon)).contiguous(),
        tau_fraction=fractions.contiguous(), labels=torch.full_like(fractions, 3, dtype=torch.long),
        path_ids=path_ids, strata=strata, role=str(role), law=str(law), horizon=float(horizon),
    )


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _arrays_identity(arrays: ControlArrays) -> dict[str, Any]:
    return {
        "state_sha256": array_fingerprint(np.asarray(arrays.states, dtype=np.float32)),
        "tau_fraction_sha256": array_fingerprint(np.asarray(arrays.tau_fraction, dtype=np.float32)),
        "path_ids_sha256": array_fingerprint(np.asarray(arrays.path_ids, dtype=np.int64)),
        "strata_sha256": array_fingerprint(np.asarray(arrays.strata, dtype=np.int64)),
        "rows": int(arrays.states.shape[0]), "pixels": int(arrays.states.shape[1]),
        "role": arrays.role, "law": arrays.law, "horizon": float(arrays.horizon),
    }


def _save_arrays(path: Path, arrays: ControlArrays, binding: Mapping[str, Any]) -> dict[str, Any]:
    identity = _arrays_identity(arrays)
    _atomic_save_npz(
        path,
        states=np.asarray(arrays.states, dtype=np.float32),
        tau=np.asarray(arrays.tau, dtype=np.float32),
        tau_fraction=np.asarray(arrays.tau_fraction, dtype=np.float32),
        labels=np.asarray(arrays.labels, dtype=np.int64),
        path_ids=np.asarray(arrays.path_ids, dtype=np.int64),
        strata=np.asarray(arrays.strata, dtype=np.int64),
    )
    record = {
        "schema": CONTROL_ARRAY_SCHEMA, "schema_version": CONTROL_ARRAY_SCHEMA_VERSION,
        "identity": identity, "binding": dict(binding), "path": str(path.resolve()),
        "sha256": file_fingerprint(path), "sampling_performed": 0,
    }
    atomic_write_json(path.with_suffix(".json"), record)
    return record


def _load_arrays(path: Path, binding: Mapping[str, Any]) -> tuple[ControlArrays, dict[str, Any]]:
    sidecar = _json_load(path.with_suffix(".json"))
    if (
        sidecar.get("schema") != CONTROL_ARRAY_SCHEMA
        or int(sidecar.get("schema_version", -1)) != CONTROL_ARRAY_SCHEMA_VERSION
        or dict(sidecar.get("binding", {})) != dict(binding)
        or sidecar.get("sha256") != file_fingerprint(path)
    ):
        raise ArtifactCompatibilityError("synthetic array fingerprint mismatch")
    with np.load(path, allow_pickle=False) as value:
        arrays = ControlArrays(
            states=torch.from_numpy(value["states"].copy()).float(),
            tau=torch.from_numpy(value["tau"].copy()).float(),
            tau_fraction=torch.from_numpy(value["tau_fraction"].copy()).float(),
            labels=torch.from_numpy(value["labels"].copy()).long(),
            path_ids=value["path_ids"].astype(np.int64, copy=True),
            strata=value["strata"].astype(np.int64, copy=True),
            role=str(sidecar["identity"]["role"]), law=str(sidecar["identity"]["law"]),
            horizon=float(sidecar["identity"]["horizon"]),
        )
    if _arrays_identity(arrays) != dict(sidecar.get("identity", {})):
        raise ArtifactCompatibilityError("synthetic array contents do not match sidecar")
    return arrays, sidecar


class _ZeroPotential(nn.Module):
    def forward(self, tau: Tensor | float, states: Tensor, labels: Tensor) -> Tensor:
        del tau, labels
        return (states * 0.0).sum(dim=1)


def _set_seed(seed: int, batch_index_seed: int = 260780) -> np.random.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return np.random.default_rng(int(batch_index_seed))


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _derived_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps([int(seed), *parts], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _probe_bank(
    *, probes: int, batch: int, grid_size: int, seed: int,
    device: torch.device, dtype: torch.dtype,
) -> Tensor:
    generator_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    return orthogonal_hadamard_edge_probes(
        int(probes), int(batch), int(grid_size), device=device, dtype=dtype,
        generator=generator,
    )


def _risk_values(
    model: nn.Module,
    arrays: ControlArrays,
    dynamics: DirectFluxMNISTConfig,
    *,
    device: torch.device,
    batch_size: int,
    probes_per_state: int,
    probe_seed: int,
) -> dict[str, np.ndarray]:
    was_training = bool(model.training)
    model.eval()
    output: dict[str, list[np.ndarray]] = {
        key: [] for key in ("model", "zero", "energy", "trace", "drift")
    }
    try:
        with torch.enable_grad():
            for start in range(0, int(arrays.states.shape[0]), int(batch_size)):
                stop = min(int(arrays.states.shape[0]), start + int(batch_size))
                states = arrays.states[start:stop].to(device)
                tau = arrays.tau[start:stop].to(device)
                labels = arrays.labels[start:stop].to(device)
                probes = _probe_bank(
                    probes=int(probes_per_state), batch=stop - start,
                    grid_size=int(dynamics.grid_size),
                    seed=_derived_seed(int(probe_seed), start, stop),
                    device=device, dtype=states.dtype,
                )
                objective = dirichlet_score_objective(
                    model, tau, states, labels, dynamics, probes, create_graph=False
                )
                values = {
                    "model": objective.per_sample,
                    "zero": torch.zeros_like(objective.per_sample),
                    "energy": objective.energy,
                    "trace": objective.trace,
                    "drift": objective.drift,
                }
                for name, value in values.items():
                    output[name].append(value.detach().double().cpu().numpy())
    finally:
        model.train(was_training)
    return {
        name: np.concatenate(parts).astype(np.float64, copy=False)
        for name, parts in output.items()
    }


def _path_interval(
    state_values: np.ndarray,
    path_ids: np.ndarray,
    *,
    mask: np.ndarray | None,
    reps: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(state_values, dtype=np.float64)
    paths = np.asarray(path_ids, dtype=np.int64)
    selected = np.ones(values.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if values.shape != paths.shape or selected.shape != values.shape or not selected.any():
        raise ValueError("whole-path interval arrays are inconsistent")
    unique = np.unique(paths[selected])
    path_means = np.asarray(
        [values[selected & (paths == path_id)].mean() for path_id in unique],
        dtype=np.float64,
    )
    if not np.isfinite(path_means).all():
        return {
            "path_count": int(unique.size), "point_estimate": None,
            "lower_bound": None, "finite": 0,
        }
    rng = np.random.default_rng(int(seed))
    totals = np.empty(int(reps), dtype=np.float64)
    chunk = 1024
    for start in range(0, int(reps), chunk):
        count = min(chunk, int(reps) - start)
        indices = rng.integers(0, unique.size, size=(count, unique.size))
        totals[start : start + count] = path_means[indices].mean(axis=1)
    return {
        "path_count": int(unique.size),
        "state_count": int(selected.sum()),
        "point_estimate": float(path_means.mean()),
        "lower_bound": float(np.quantile(totals, 1.0 - float(confidence))),
        "confidence": float(confidence), "reps": int(reps), "finite": 1,
        "path_ids": unique.tolist(), "path_values": path_means.tolist(),
    }


def _risk_bank_record(
    components: Mapping[str, np.ndarray],
    arrays: ControlArrays,
    *,
    reps: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    model = np.asarray(components["model"], dtype=np.float64)
    improvement = -model

    def scope(mask: np.ndarray | None, scope_name: str) -> dict[str, Any]:
        selected = np.ones(model.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        interval = _path_interval(
            improvement, arrays.path_ids, mask=selected, reps=int(reps),
            confidence=float(confidence), seed=_derived_seed(bootstrap_seed, scope_name),
        )
        return {
            "state_count": int(selected.sum()),
            "model_score_risk": float(model[selected].mean()),
            "zero_score_risk": 0.0,
            "objective_improvement": float(improvement[selected].mean()),
            "lower_bound": interval["lower_bound"],
            "bootstrap": interval,
            "finite_fraction": float(np.isfinite(model[selected]).mean()),
        }

    return {
        "overall": scope(None, "overall"),
        "data_end": scope(arrays.strata == 4, "data_end"),
    }


def _implicit_selection_record(
    model: nn.Module,
    arrays: ControlArrays,
    dynamics: DirectFluxMNISTConfig,
    *, step: int, args: argparse.Namespace, device: torch.device,
) -> dict[str, Any]:
    banks: dict[str, Any] = {}
    for name, seed in (
        ("a", int(args.selection_probe_a_seed)),
        ("b", int(args.selection_probe_b_seed)),
    ):
        components = _risk_values(
            model, arrays, dynamics, device=device,
            batch_size=int(args.validation_batch_size),
            probes_per_state=int(args.selection_probes), probe_seed=seed,
        )
        banks[name] = _risk_bank_record(
            components, arrays, reps=int(args.bootstrap_reps),
            confidence=float(args.bootstrap_confidence),
            bootstrap_seed=_derived_seed(int(args.bootstrap_seed), "selection", name),
        )
    return {"step": int(step), "finite": int(_banks_finite(banks)), "banks": banks}


def _banks_finite(banks: Mapping[str, Any]) -> bool:
    try:
        return all(
            math.isfinite(float(dict(dict(banks[name])[scope])["model_score_risk"]))
            and math.isfinite(float(dict(dict(banks[name])[scope])["lower_bound"]))
            for name in ("a", "b") for scope in ("overall", "data_end")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _cell_gradients(
    model: nn.Module, arrays: ControlArrays, *, device: torch.device, batch_size: int
) -> np.ndarray:
    was_training = bool(model.training)
    model.eval()
    pieces: list[np.ndarray] = []
    try:
        with torch.enable_grad():
            for start in range(0, int(arrays.states.shape[0]), int(batch_size)):
                stop = min(int(arrays.states.shape[0]), start + int(batch_size))
                states = arrays.states[start:stop].to(device).detach().requires_grad_(True)
                potential = model(
                    arrays.tau[start:stop].to(device), states,
                    arrays.labels[start:stop].to(device),
                )
                gradient = torch.autograd.grad(potential.sum(), states)[0]
                pieces.append(gradient.detach().cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _analytic_per_state_errors(
    model: nn.Module, arrays: ControlArrays, dynamics: DirectFluxMNISTConfig,
    *, device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    gradients = torch.from_numpy(
        _cell_gradients(model, arrays, device=device, batch_size=batch_size)
    ).to(device)
    states = arrays.states.to(device)
    predicted = edge_difference_channels(gradients, int(dynamics.grid_size))
    target = bounded_teacher_edge_score(states, arrays.tau_fraction.to(device))
    theta = harmonic_mobility_exact(states, dynamics)
    error = (theta * (predicted - target).square()).flatten(1).mean(1)
    zero = (theta * target.square()).flatten(1).mean(1)
    return (
        error.detach().double().cpu().numpy(),
        zero.detach().double().cpu().numpy(),
    )


def _supervised_selection_record(
    model: nn.Module, arrays: ControlArrays, dynamics: DirectFluxMNISTConfig,
    *, step: int, args: argparse.Namespace, device: torch.device,
) -> dict[str, Any]:
    error, zero = _analytic_per_state_errors(
        model, arrays, dynamics, device=device, batch_size=int(args.validation_batch_size)
    )
    improvement = zero - error

    def scope(mask: np.ndarray | None, name: str) -> dict[str, Any]:
        selected = np.ones(error.size, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        interval = _path_interval(
            improvement, arrays.path_ids, mask=selected,
            reps=int(args.bootstrap_reps), confidence=float(args.bootstrap_confidence),
            seed=_derived_seed(int(args.bootstrap_seed), "supervised", name),
        )
        return {
            "model_score_risk": float(error[selected].mean()),
            "zero_score_risk": float(zero[selected].mean()),
            "lower_bound": interval["lower_bound"], "bootstrap": interval,
        }

    common = {"overall": scope(None, "overall"), "data_end": scope(arrays.strata == 4, "data_end")}
    return {"step": int(step), "finite": int(np.isfinite(error).all()), "banks": {"a": common, "b": copy.deepcopy(common)}}


def _supervised_loss(
    model: nn.Module, states: Tensor, tau: Tensor, fractions: Tensor, labels: Tensor,
    dynamics: DirectFluxMNISTConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    states_req = states.detach().clone().requires_grad_(True)
    potential = model(tau, states_req, labels)
    gradient = torch.autograd.grad(potential.sum(), states_req, create_graph=True)[0]
    prediction = edge_difference_channels(gradient, int(dynamics.grid_size))
    target = bounded_teacher_edge_score(states_req.detach(), fractions).detach()
    theta = harmonic_mobility_exact(states_req.detach(), dynamics).detach()
    per_state = (theta * (prediction - target).square()).flatten(1).mean(1)
    zero = (theta * target.square()).flatten(1).mean(1).clamp_min(1e-20)
    loss = (per_state / zero.detach()).mean()
    return loss, {"loss": loss, "energy": per_state.mean(), "trace": loss.new_zeros(()), "drift": loss.new_zeros(())}


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> Tensor:
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack(norms))


def _model_boundary_certificate(model: nn.Module) -> dict[str, Any]:
    finite_parameters = all(
        bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters()
    )
    model_version = getattr(model, "model_version", None)
    features = tuple(getattr(model, "state_feature_names", ()))
    passed = bool(
        isinstance(model, D0BoundarySmoothPotentialUNet)
        and model_version == BOUNDARY_SMOOTH_MODEL_VERSION
        and features == ("relative_density", "log1p_relative_density")
        and finite_parameters
    )
    return {
        "model_version": model_version,
        "expected_model_version": BOUNDARY_SMOOTH_MODEL_VERSION,
        "state_feature_names": list(features),
        "raw_log_density_used": 0,
        "finite_parameters": int(finite_parameters),
        "structural_facet_certificate": "smooth-closed-simplex-inputs-plus-preflight-v1",
        "passed": int(passed),
    }


def _calibrate_loss_scale(
    path: Path,
    *, arrays: ControlArrays, dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace, device: torch.device, binding: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_file():
        value = _json_load(path)
        if dict(value.get("binding", {})) != dict(binding) or not _finite_positive(value.get("loss_scale")):
            raise ArtifactCompatibilityError("loss-scale calibration fingerprint mismatch")
        return value
    _set_seed(int(args.calibration_seed))
    model = D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    model.zero_grad(set_to_none=True)
    count = min(256, int(arrays.states.shape[0]))
    chunk_size = min(int(args.batch_size), count)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        states = arrays.states[start:stop].to(device)
        probes = _probe_bank(
            probes=int(args.training_probes), batch=stop - start,
            grid_size=int(dynamics.grid_size),
            seed=_derived_seed(int(args.training_probe_seed), "calibration", start, stop),
            device=device, dtype=states.dtype,
        )
        objective = dirichlet_score_objective(
            model, arrays.tau[start:stop].to(device), states,
            arrays.labels[start:stop].to(device), dynamics, probes, create_graph=True,
        )
        (objective.loss * float(stop - start) / float(count)).backward()
    unscaled_norm = float(_gradient_norm(list(model.parameters())).detach().cpu())
    if not _finite_positive(unscaled_norm):
        raise FloatingPointError("loss-scale calibration produced a zero/nonfinite gradient")
    value = {
        "schema": RUN_SCHEMA + "-loss-scale", "schema_version": 1,
        "binding": dict(binding), "calibration_state_count": int(count),
        "calibration_state_sha256": array_fingerprint(np.asarray(arrays.states[:count], dtype=np.float32)),
        "unscaled_initial_gradient_norm": unscaled_norm,
        "target_initial_gradient_norm": 0.5,
        "loss_scale": 0.5 / unscaled_norm,
        "shared_by_implicit_teacher_and_null": 1,
        "sampling_performed": 0,
    }
    atomic_write_json(path, value)
    return value


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema_state: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    step: int,
    history: Sequence[Mapping[str, Any]],
    validations: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
    batch_rng: np.random.Generator,
    probe_generator: torch.Generator,
    task_kind: str,
) -> None:
    atomic_torch_save(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_schema": MODEL_SCHEMA,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "task_kind": str(task_kind),
            "step": int(step),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "ema_state_dict": {key: value.detach().clone() for key, value in ema_state.items()},
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "history": [dict(value) for value in history],
            "validation_records": copy.deepcopy(list(validations)),
            "checkpoint_selection": copy.deepcopy(dict(selection)),
            "fingerprints": dict(fingerprints),
            "rng_state": capture_rng_state(batch_rng),
            "training_probe_generator_state": probe_generator.get_state().cpu(),
            "scaler_state_dict": None,
            "amp": False,
            "sampling_performed": 0,
        },
    )


def _load_checkpoint(
    path: Path, *, device: torch.device, fingerprints: Mapping[str, Any], task_kind: str
) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        value = torch.load(path, map_location=device)
    if (
        value.get("schema") != CHECKPOINT_SCHEMA
        or int(value.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION
        or value.get("model_schema") != MODEL_SCHEMA
        or int(value.get("model_schema_version", -1)) != MODEL_SCHEMA_VERSION
        or value.get("task_kind") != str(task_kind)
    ):
        raise ArtifactCompatibilityError(
            "legacy/foreign checkpoint is reportable but incompatible with the boundary gate"
        )
    if dict(value.get("fingerprints", {})) != dict(fingerprints):
        raise ArtifactCompatibilityError("boundary-control checkpoint fingerprint mismatch")
    required = {
        "step", "model_state_dict", "ema_state_dict", "optimizer_state_dict",
        "history", "validation_records", "checkpoint_selection", "rng_state",
        "training_probe_generator_state", "scaler_state_dict",
    }
    if not required.issubset(value):
        raise ArtifactCompatibilityError("boundary-control checkpoint is incomplete")
    return value


def _flatten_checkpoint_rows(validations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in validations:
        for bank_name, bank in dict(record.get("banks", {})).items():
            for scope_name, scope in dict(bank).items():
                value = dict(scope)
                value.pop("bootstrap", None)
                rows.append(
                    {
                        "step": int(record["step"]), "bank": bank_name,
                        "scope": scope_name, "selection_eligible": int(record.get("selection_eligible", 0)),
                        **value,
                    }
                )
    return rows


def _flatten_selection_path_rows(
    validations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in validations:
        for bank_name, bank in dict(record.get("banks", {})).items():
            for scope_name, scope in dict(bank).items():
                bootstrap = dict(dict(scope).get("bootstrap", {}))
                path_ids = list(bootstrap.get("path_ids", []))
                path_values = list(bootstrap.get("path_values", []))
                if len(path_ids) != len(path_values):
                    raise ArtifactCompatibilityError(
                        "selection bootstrap path IDs and values disagree"
                    )
                rows.extend(
                    {
                        "step": int(record["step"]),
                        "bank": str(bank_name),
                        "scope": str(scope_name),
                        "path_id": int(path_id),
                        "objective_improvement_vs_step_zero": float(value),
                    }
                    for path_id, value in zip(path_ids, path_values)
                )
    return rows


def _history_diagnostics(
    history: Sequence[Mapping[str, Any]], *, warmup_steps: int, grad_clip: float
) -> dict[str, Any]:
    if history and all("scaled_preclip_gradient_norm" in row for row in history):
        scaled = summarize_scaled_gradient_history(
            history, warmup_steps=warmup_steps, grad_clip=grad_clip
        )
        # Retain the component quantiles expected by legacy reports while
        # making the optimizer-health norm source explicit.
        after = [row for row in history if int(row.get("step", 0)) > int(warmup_steps)]
        if not after:
            after = list(history)

        def component_quantiles(name: str) -> dict[str, float | None]:
            values = np.asarray(
                [
                    float(row[name])
                    for row in after
                    if name in row and math.isfinite(float(row[name]))
                ],
                dtype=np.float64,
            )
            names = ("q00", "q10", "q50", "q90", "q99", "q100")
            if values.size == 0:
                return {key: None for key in names}
            return {
                key: float(value)
                for key, value in zip(
                    names, np.quantile(values, (0.0, 0.1, 0.5, 0.9, 0.99, 1.0))
                )
            }

        scaled["quantiles"].update(
            {
                name: component_quantiles(name)
                for name in ("energy", "trace", "drift", "cancellation_ratio")
            }
        )
        # `grad_norm` remains an artifact compatibility alias only.  Gates use
        # `scaled_preclip_gradient_norm` through the source field above.
        scaled["quantiles"]["grad_norm"] = dict(
            scaled["quantiles"]["scaled_preclip_gradient_norm"]
        )
        scaled["quantiles"]["loss"] = dict(scaled["quantiles"]["scaled_loss"])
        return scaled

    after = [row for row in history if int(row.get("step", 0)) > int(warmup_steps)]
    if not after:
        after = list(history)

    def quantiles(name: str) -> dict[str, float | None]:
        values = np.asarray(
            [float(row[name]) for row in after if name in row and math.isfinite(float(row[name]))],
            dtype=np.float64,
        )
        if values.size == 0:
            return {"q00": None, "q10": None, "q50": None, "q90": None, "q99": None, "q100": None}
        return {
            key: float(value)
            for key, value in zip(
                ("q00", "q10", "q50", "q90", "q99", "q100"),
                np.quantile(values, (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)),
            )
        }

    clip_fraction = (
        float(np.mean([float(row.get("grad_norm", 0.0)) > float(grad_clip) for row in after]))
        if after else 0.0
    )
    return {
        "post_warmup_steps": len(after),
        "post_warmup_clip_fraction": clip_fraction,
        "quantiles": {
            name: quantiles(name)
            for name in ("loss", "unscaled_loss", "energy", "trace", "drift", "grad_norm", "cancellation_ratio")
        },
    }


def _task_fingerprints(
    *, scientific_fingerprint: str, runtime_fingerprint: str,
    source_fingerprint_value: str, arrays: ControlArrays,
    selection_arrays: ControlArrays, audit_arrays: ControlArrays,
    task_kind: str, model_seed: int, loss_scale: float,
    array_registry_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "scientific_fingerprint": str(scientific_fingerprint),
        "runtime_fingerprint": str(runtime_fingerprint),
        "source_fingerprint": str(source_fingerprint_value),
        "array_identities": {
            "train": _arrays_identity(arrays),
            "selection": _arrays_identity(selection_arrays),
            "audit": _arrays_identity(audit_arrays),
        },
        "array_registry_sha256": (
            None if array_registry_sha256 is None else str(array_registry_sha256)
        ),
        "task_kind": str(task_kind), "model_seed": int(model_seed),
        "loss_scale": float(loss_scale),
        "model_schema": MODEL_SCHEMA, "model_schema_version": MODEL_SCHEMA_VERSION,
    }


def _train_task(
    *,
    task_dir: Path,
    task_kind: str,
    train: ControlArrays,
    selection_arrays: ControlArrays,
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    args: argparse.Namespace,
    model_seed: int,
    loss_scale: float,
    fingerprints: Mapping[str, Any],
    show_progress: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    if task_kind not in {"supervised_teacher", "implicit_teacher", "null"}:
        raise ValueError(f"unknown task kind {task_kind!r}")
    task_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoints = task_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints / "latest.json"
    best_path = checkpoints / "best_ema.pt"
    best_pointer_path = checkpoints / "best.json"

    batch_rng = _set_seed(int(model_seed), int(args.batch_index_seed))
    probe_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    probe_generator = torch.Generator(device=probe_device).manual_seed(int(args.training_probe_seed))
    model = D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    ema_state = init_ema_state(model)
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    completed = 0

    if latest_path.is_file():
        latest = _json_load(latest_path)
        filename = str(latest.get("filename", ""))
        checkpoint_path = checkpoints / filename
        if (
            Path(filename).name != filename or not checkpoint_path.is_file()
            or latest.get("sha256") != file_fingerprint(checkpoint_path)
            or dict(latest.get("fingerprints", {})) != dict(fingerprints)
        ):
            raise ArtifactCompatibilityError("boundary-control latest pointer is invalid")
        payload = _load_checkpoint(
            checkpoint_path, device=device, fingerprints=fingerprints, task_kind=task_kind
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        ema_state = {key: value.detach().clone().to(device) for key, value in payload["ema_state_dict"].items()}
        history = [dict(value) for value in payload["history"]]
        validations = [dict(value) for value in payload["validation_records"]]
        selection = dict(payload["checkpoint_selection"])
        completed = int(payload["step"])
        restore_rng_state(payload["rng_state"], batch_rng)
        probe_generator.set_state(torch.as_tensor(payload["training_probe_generator_state"], dtype=torch.uint8, device="cpu"))

    def validate(step: int) -> dict[str, Any]:
        with temporary_ema_weights(model, ema_state):
            if task_kind == "supervised_teacher":
                return _supervised_selection_record(
                    model, selection_arrays, dynamics, step=step, args=args, device=device
                )
            return _implicit_selection_record(
                model, selection_arrays, dynamics, step=step, args=args, device=device
            )

    def publish_checkpoint(step: int) -> None:
        nonlocal selection
        selection = select_dual_bank_checkpoint(validations)
        for record in validations:
            record["selection_eligible"] = int(
                next(
                    row["selection_eligible"]
                    for row in selection["records"] if int(row["step"]) == int(record["step"])
                )
            )
        path = checkpoints / f"step-{int(step):08d}.pt"
        _save_checkpoint(
            path, model=model, ema_state=ema_state, optimizer=optimizer, step=step,
            history=history, validations=validations, selection=selection,
            fingerprints=fingerprints, batch_rng=batch_rng, probe_generator=probe_generator,
            task_kind=task_kind,
        )
        selected_step = int(selection["selected_step"])
        authoritative = checkpoints / f"step-{selected_step:08d}.pt"
        if not authoritative.is_file():
            if selected_step == int(step):
                authoritative = path
            else:
                raise ArtifactCompatibilityError("selected checkpoint is missing")
        if not best_path.is_file() or file_fingerprint(best_path) != file_fingerprint(authoritative):
            atomic_copy_file(authoritative, best_path)
        best_pointer = {
            "schema": CHECKPOINT_SCHEMA + "-best", "schema_version": 1,
            "selected_step": selected_step,
            "authoritative_filename": authoritative.name,
            "authoritative_sha256": file_fingerprint(authoritative),
            "best_ema_filename": best_path.name,
            "best_ema_sha256": file_fingerprint(best_path),
            "fingerprints": dict(fingerprints),
        }
        atomic_write_json(best_pointer_path, best_pointer)
        atomic_write_json(
            latest_path,
            {
                "schema": CHECKPOINT_SCHEMA + "-latest", "schema_version": 1,
                "filename": path.name, "step": int(step), "sha256": file_fingerprint(path),
                "fingerprints": dict(fingerprints),
            },
        )
        atomic_write_json(
            task_dir / "task_status.json",
            {
                "schema": RUN_SCHEMA + "-task-status", "schema_version": 1,
                "status": "running", "task_kind": task_kind, "model_seed": int(model_seed),
                "training_step": int(step), "selected_step": selected_step,
                "fingerprints": dict(fingerprints), "sampling_performed": 0,
            },
        )

    if not validations:
        validations.append(validate(0))
        publish_checkpoint(0)

    started = time.perf_counter()
    for step in range(completed + 1, int(args.train_steps) + 1):
        choices = batch_rng.integers(
            0, int(train.states.shape[0]), size=int(args.batch_size), dtype=np.int64
        )
        ids = torch.as_tensor(choices, dtype=torch.long)
        states = train.states.index_select(0, ids).to(device)
        tau = train.tau.index_select(0, ids).to(device)
        fractions = train.tau_fraction.index_select(0, ids).to(device)
        labels = train.labels.index_select(0, ids).to(device)
        optimizer.zero_grad(set_to_none=True)
        if task_kind == "supervised_teacher":
            unscaled, components = _supervised_loss(
                model, states, tau, fractions, labels, dynamics
            )
        else:
            probes = orthogonal_hadamard_edge_probes(
                int(args.training_probes), int(ids.numel()), int(dynamics.grid_size),
                device=device, dtype=states.dtype, generator=probe_generator,
            )
            objective = dirichlet_score_objective(
                model, tau, states, labels, dynamics, probes, create_graph=True
            )
            unscaled = objective.loss
            components = {
                "loss": unscaled, "energy": objective.energy.mean(),
                "trace": objective.trace.mean(), "drift": objective.drift.mean(),
            }
        if not bool(torch.isfinite(unscaled)):
            raise FloatingPointError(f"non-finite {task_kind} loss at step {step}")
        gradient_diagnostics = scaled_backward_and_clip(
            unscaled,
            model.parameters(),
            loss_scale=float(loss_scale),
            grad_clip=float(args.grad_clip),
        )
        optimizer.step()
        update_ema_state(ema_state, model, float(args.ema_decay))
        energy = float(components["energy"].detach().cpu())
        trace = float(components["trace"].detach().cpu())
        drift = float(components["drift"].detach().cpu())
        unscaled_value = float(unscaled.detach().cpu())
        cancellation_denom = abs(energy) + abs(trace) + abs(drift)
        history.append(
            {
                "step": int(step),
                "loss": float(gradient_diagnostics.scaled_loss),
                "unscaled_loss": unscaled_value,
                "scaled_loss": float(gradient_diagnostics.scaled_loss),
                "loss_scale": float(gradient_diagnostics.loss_scale),
                "raw_gradient_norm": float(gradient_diagnostics.raw_gradient_norm),
                "scaled_preclip_gradient_norm": float(
                    gradient_diagnostics.scaled_preclip_gradient_norm
                ),
                # Compatibility alias for existing CSV readers and plots.
                "grad_norm": float(gradient_diagnostics.scaled_preclip_gradient_norm),
                "energy": energy, "trace": trace,
                "drift": drift, "clipped": int(gradient_diagnostics.clipped),
                "cancellation_ratio": abs(unscaled_value) / max(cancellation_denom, 1e-30),
            }
        )
        validation_due = step % int(args.validation_every) == 0 or step == int(args.train_steps)
        checkpoint_due = step % int(args.checkpoint_every) == 0 or step == int(args.train_steps)
        if validation_due:
            validations.append(validate(step))
        if validation_due or checkpoint_due:
            publish_checkpoint(step)
        if show_progress and (step % 50 == 0 or step == int(args.train_steps)):
            elapsed = time.perf_counter() - started
            done = max(1, step - completed)
            eta = elapsed / done * max(0, int(args.train_steps) - step)
            print(
                f"{task_dir.name}: step {step}/{args.train_steps} "
                f"loss={history[-1]['loss']:.6g} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    if not selection or "selected_step" not in selection:
        raise RuntimeError("boundary-control task did not select a checkpoint")
    selected_step = int(selection["selected_step"])
    authoritative = checkpoints / f"step-{selected_step:08d}.pt"
    if not authoritative.is_file():
        raise RuntimeError("boundary-control selected checkpoint is missing")
    authoritative_hash = file_fingerprint(authoritative)
    if not best_path.is_file() or file_fingerprint(best_path) != authoritative_hash:
        atomic_copy_file(authoritative, best_path)
    expected_pointer = {
        "schema": CHECKPOINT_SCHEMA + "-best", "schema_version": 1,
        "selected_step": selected_step,
        "authoritative_filename": authoritative.name,
        "authoritative_sha256": authoritative_hash,
        "best_ema_filename": best_path.name,
        "best_ema_sha256": file_fingerprint(best_path),
        "fingerprints": dict(fingerprints),
    }
    if not best_pointer_path.is_file() or _json_load(best_pointer_path) != expected_pointer:
        atomic_write_json(best_pointer_path, expected_pointer)
    selected = _load_checkpoint(
        best_path, device=device, fingerprints=fingerprints, task_kind=task_kind
    )
    model.load_state_dict(selected["ema_state_dict"], strict=True)
    model.eval()
    boundary_certificate = _model_boundary_certificate(model)
    diagnostics = _history_diagnostics(
        history, warmup_steps=int(args.clip_warmup_steps), grad_clip=float(args.grad_clip)
    )
    atomic_write_csv(task_dir / "training_history.csv", history)
    atomic_write_csv(task_dir / "checkpoint_metrics.csv", _flatten_checkpoint_rows(validations))
    atomic_write_csv(
        task_dir / "selection_path_risks.csv",
        _flatten_selection_path_rows(validations),
    )
    summary = {
        "complete": 1, "finite": int(all(math.isfinite(float(row["loss"])) for row in history)),
        "task_kind": task_kind, "model_seed": int(model_seed),
        "selected_step": int(selection["selected_step"]),
        "checkpoint_selection": selection,
        "checkpoint_path": str(best_path.resolve()), "checkpoint_sha256": file_fingerprint(best_path),
        "best_pointer_path": str(best_pointer_path.resolve()), "best_pointer_sha256": file_fingerprint(best_pointer_path),
        "post_warmup_clip_fraction": diagnostics["post_warmup_clip_fraction"],
        "optimization_diagnostics": diagnostics,
        "boundary_admissibility_certificate": boundary_certificate,
        "peak_memory_gib": (
            float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
            if device.type == "cuda" else 0.0
        ),
        "fingerprints": dict(fingerprints), "sampling_performed": 0,
    }
    atomic_write_json(task_dir / "training_summary.json", summary)
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "schema": RUN_SCHEMA + "-task-status", "schema_version": 1,
            "status": "training_complete", "task_kind": task_kind,
            "model_seed": int(model_seed), "training_step": int(args.train_steps),
            "selected_step": int(selection["selected_step"]),
            "fingerprints": dict(fingerprints), "sampling_performed": 0,
        },
    )
    return model, summary


def _teacher_metrics(
    *,
    model: nn.Module,
    arrays: ControlArrays,
    dynamics: DirectFluxMNISTConfig,
    summary: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    include_objective_banks: bool,
) -> dict[str, Any]:
    gradients = torch.from_numpy(
        _cell_gradients(model, arrays, device=device, batch_size=int(args.validation_batch_size))
    ).to(device)
    states = arrays.states.to(device)
    prediction = edge_difference_channels(gradients, int(dynamics.grid_size))
    target = bounded_teacher_edge_score(states, arrays.tau_fraction.to(device))
    theta = harmonic_mobility_exact(states, dynamics)

    def scope(mask: np.ndarray) -> dict[str, Any]:
        ids = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=device)
        target_scope = target.index_select(0, ids)
        prediction_scope = prediction.index_select(0, ids)
        weights = theta.index_select(0, ids)
        full_mse = float((weights * (prediction_scope - target_scope).square()).mean().detach().cpu())
        zero_mse = float((weights * target_scope.square()).mean().detach().cpu())
        target_flux = physical_flux_from_edge_score(
            target_scope, states.index_select(0, ids), dynamics
        )
        prediction_flux = physical_flux_from_edge_score(
            prediction_scope, states.index_select(0, ids), dynamics
        )
        flat_target = target_flux.reshape(-1).double()
        flat_prediction = prediction_flux.reshape(-1).double()
        denom = torch.linalg.vector_norm(flat_target) * torch.linalg.vector_norm(flat_prediction)
        cosine = float((flat_target @ flat_prediction / denom.clamp_min(1e-30)).detach().cpu())
        relative = float(
            (
                torch.linalg.vector_norm(flat_prediction - flat_target)
                / torch.linalg.vector_norm(flat_target).clamp_min(1e-30)
            ).detach().cpu()
        )
        return {
            "state_count": int(ids.numel()), "target_mse": zero_mse,
            "model_mse": full_mse, "score_gain": 1.0 - full_mse / max(zero_mse, 1e-30),
            "flux_cosine": cosine, "flux_relative_l2": relative,
        }

    overall = scope(np.ones(len(arrays.path_ids), dtype=bool))
    data_end = scope(arrays.strata == 4)
    bins = [scope(arrays.strata == index) for index in range(5)]
    metrics: dict[str, Any] = {
        "complete": 1, "finite": int(bool(torch.isfinite(prediction).all())),
        "model_seed": int(summary["model_seed"]),
        "selected_step": int(summary["selected_step"]),
        "audit_overall_score_gain": overall["score_gain"],
        "audit_data_end_score_gain": data_end["score_gain"],
        "overall_flux_cosine": overall["flux_cosine"],
        "time_bin_flux_cosines": [value["flux_cosine"] for value in bins],
        "overall_relative_flux_l2": overall["flux_relative_l2"],
        "time_bin_relative_flux_l2": [value["flux_relative_l2"] for value in bins],
        "overall": overall, "data_end": data_end, "time_bins": bins,
        "boundary_admissible": int(
            bool(dict(summary.get("boundary_admissibility_certificate", {})).get("passed", 0))
        ),
        "post_warmup_clip_fraction": float(summary["post_warmup_clip_fraction"]),
        "sampling_performed": 0,
    }
    if include_objective_banks:
        objective_banks: dict[str, Any] = {}
        per_path_rows: list[dict[str, Any]] = []
        for bank_name, probe_seed in (
            ("a", int(args.audit_probe_a_seed)),
            ("b", int(args.audit_probe_b_seed)),
        ):
            components = _risk_values(
                model, arrays, dynamics, device=device,
                batch_size=int(args.validation_batch_size), probes_per_state=int(args.audit_probes),
                probe_seed=probe_seed,
            )
            record = _risk_bank_record(
                components, arrays, reps=int(args.bootstrap_reps),
                confidence=float(args.bootstrap_confidence),
                bootstrap_seed=_derived_seed(int(args.bootstrap_seed), "audit", "teacher", bank_name),
            )
            objective_banks[bank_name] = record
            for scope_name, scope_record in record.items():
                bootstrap = dict(scope_record["bootstrap"])
                for path_id, value in zip(bootstrap["path_ids"], bootstrap["path_values"]):
                    per_path_rows.append(
                        {
                            "model_seed": int(summary["model_seed"]), "bank": bank_name,
                            "scope": scope_name, "path_id": int(path_id),
                            "objective_improvement_vs_zero": float(value),
                        }
                    )
        metrics["audit_objective_banks"] = objective_banks
        metrics["audit_path_rows"] = per_path_rows
    return metrics


def _null_metrics(
    *, model: nn.Module, arrays: ControlArrays, dynamics: DirectFluxMNISTConfig,
    summary: Mapping[str, Any], args: argparse.Namespace, device: torch.device,
) -> dict[str, Any]:
    banks: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    all_finite = True
    for bank_name, probe_seed in (
        ("a", int(args.audit_probe_a_seed)),
        ("b", int(args.audit_probe_b_seed)),
    ):
        components = _risk_values(
            model, arrays, dynamics, device=device,
            batch_size=int(args.validation_batch_size), probes_per_state=int(args.audit_probes),
            probe_seed=probe_seed,
        )
        all_finite = all_finite and all(np.isfinite(value).all() for value in components.values())
        record = _risk_bank_record(
            components, arrays, reps=int(args.bootstrap_reps),
            confidence=float(args.bootstrap_confidence),
            bootstrap_seed=_derived_seed(int(args.bootstrap_seed), "audit", "null", bank_name),
        )
        banks[bank_name] = record
        for scope_name, scope_record in record.items():
            bootstrap = dict(scope_record["bootstrap"])
            for path_id, value in zip(bootstrap["path_ids"], bootstrap["path_values"]):
                rows.append(
                    {
                        "model_seed": int(summary["model_seed"]), "bank": bank_name,
                        "scope": scope_name, "path_id": int(path_id),
                        "objective_improvement_vs_zero": float(value),
                    }
                )
    return {
        "complete": 1, "finite": int(all_finite),
        "model_seed": int(summary["model_seed"]),
        "selected_step": int(summary["selected_step"]),
        "comparator": "analytic_zero", "linear_comparator_role": "advisory_only",
        "audit_objective_banks": banks, "audit_path_rows": rows,
        "boundary_admissible": int(
            bool(dict(summary.get("boundary_admissibility_certificate", {})).get("passed", 0))
        ),
        "post_warmup_clip_fraction": float(summary["post_warmup_clip_fraction"]),
        "sampling_performed": 0,
    }


def _load_completed_task(
    task_dir: Path, *, fingerprints: Mapping[str, Any]
) -> dict[str, Any] | None:
    status_path = task_dir / "task_status.json"
    result_path = task_dir / "task_result.json"
    if not status_path.is_file() or not result_path.is_file():
        return None
    status = _json_load(status_path)
    result = _json_load(result_path)
    if (
        dict(status.get("fingerprints", {})) != dict(fingerprints)
        or dict(result.get("fingerprints", {})) != dict(fingerprints)
        or int(result.get("sampling_performed", -1)) != 0
    ):
        raise ArtifactCompatibilityError("completed boundary-control task is incompatible")
    if status.get("status") != "complete":
        if status.get("status") not in {"running", "training_complete"}:
            raise ArtifactCompatibilityError("boundary-control task status is incompatible")
        # A crash between publishing task_result.json and its terminal status
        # is safe to replay: the exact latest checkpoint is authoritative and
        # the deterministic audit is recomputed below.
        return None
    if status.get("task_result_sha256") != file_fingerprint(result_path):
        raise ArtifactCompatibilityError("completed boundary-control result hash mismatch")
    summary = dict(result.get("training_summary", {}))
    best_path = task_dir / "checkpoints" / "best_ema.pt"
    pointer_path = task_dir / "checkpoints" / "best.json"
    if not best_path.is_file() or not pointer_path.is_file():
        raise FileNotFoundError(best_path if not best_path.is_file() else pointer_path)
    pointer = _json_load(pointer_path)
    authoritative_name = str(pointer.get("authoritative_filename", ""))
    authoritative = task_dir / "checkpoints" / authoritative_name
    if (
        Path(authoritative_name).name != authoritative_name
        or not authoritative.is_file()
        or int(pointer.get("selected_step", -1)) != int(summary.get("selected_step", -2))
        or dict(pointer.get("fingerprints", {})) != dict(fingerprints)
        or pointer.get("authoritative_sha256") != file_fingerprint(authoritative)
        or pointer.get("best_ema_sha256") != file_fingerprint(best_path)
        or summary.get("checkpoint_sha256") != file_fingerprint(best_path)
        or summary.get("best_pointer_sha256") != file_fingerprint(pointer_path)
    ):
        raise ArtifactCompatibilityError("completed boundary-control checkpoint binding mismatch")
    return result


def _complete_task(
    task_dir: Path, *, result: Mapping[str, Any], fingerprints: Mapping[str, Any]
) -> None:
    result_path = task_dir / "task_result.json"
    atomic_write_json(result_path, dict(result))
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "schema": RUN_SCHEMA + "-task-status", "schema_version": 1,
            "status": "complete", "task_kind": result["task_kind"],
            "model_seed": int(result["model_seed"]),
            "selected_step": int(dict(result["metrics"])["selected_step"]),
            "fingerprints": dict(fingerprints),
            "task_result_sha256": file_fingerprint(result_path),
            "sampling_performed": 0,
        },
    )
    (task_dir / "task_failure.json").unlink(missing_ok=True)


def _run_control_task(
    *, task_dir: Path, task_kind: str, train: ControlArrays,
    selection_arrays: ControlArrays, audit: ControlArrays,
    dynamics: DirectFluxMNISTConfig, args: argparse.Namespace, device: torch.device,
    model_seed: int, loss_scale: float, fingerprints: Mapping[str, Any],
    show_progress: bool, thresholds: BoundaryControlThresholds,
) -> dict[str, Any]:
    existing = _load_completed_task(task_dir, fingerprints=fingerprints)
    if existing is not None:
        return existing
    model, summary = _train_task(
        task_dir=task_dir, task_kind=task_kind, train=train,
        selection_arrays=selection_arrays, dynamics=dynamics, device=device,
        args=args, model_seed=int(model_seed), loss_scale=float(loss_scale),
        fingerprints=fingerprints, show_progress=show_progress,
    )
    if task_kind == "null":
        metrics = _null_metrics(
            model=model, arrays=audit, dynamics=dynamics, summary=summary,
            args=args, device=device,
        )
        gate = evaluate_null_seed(metrics, thresholds)
    else:
        metrics = _teacher_metrics(
            model=model, arrays=audit, dynamics=dynamics, summary=summary,
            args=args, device=device, include_objective_banks=task_kind == "implicit_teacher",
        )
        gate = (
            evaluate_supervised_teacher(metrics, thresholds)
            if task_kind == "supervised_teacher"
            else evaluate_implicit_teacher_seed(metrics, thresholds)
        )
    path_rows = list(metrics.pop("audit_path_rows", []))
    if path_rows:
        atomic_write_csv(task_dir / "audit_path_risks.csv", path_rows)
    result = {
        "schema": RUN_SCHEMA + "-task-result", "schema_version": 1,
        "task_kind": task_kind, "model_seed": int(model_seed),
        "metrics": metrics, "gate": gate, "training_summary": summary,
        "fingerprints": dict(fingerprints), "sampling_performed": 0,
    }
    _complete_task(task_dir, result=result, fingerprints=fingerprints)
    return result


def _source_record() -> tuple[str, list[str]]:
    here = Path(__file__).resolve()
    paths = [
        here,
        here.with_name("d0_score_boundary_controls.py"),
        here.with_name("d0_score_boundary_control_gate.py"),
        here.with_name("d0_dirichlet_score.py"),
        here.with_name("d0_one_image_gate.py"),
        here.with_name("eulerian_flux_mnist.py"),
    ]
    existing = [path for path in paths if path.is_file()]
    return source_fingerprint(existing), [str(path) for path in existing]


def _runtime_record(device: torch.device, backend: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": str(device), "device_type": device.type,
        "exact_backend": dict(backend),
    }
    if device.type == "cuda":
        value["device_name"] = torch.cuda.get_device_name(device)
        value["device_count"] = int(torch.cuda.device_count())
    return value


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        path = Path(args.resume_run_dir)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(args.runs_root) / f"{stamp}_{args.run_name}"
    path.mkdir(parents=True, exist_ok=False)
    return path, False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    current = _json_load(path) if path.is_file() else {}
    current.update(updates)
    current.setdefault("schema", RUN_SCHEMA)
    current.setdefault("schema_version", RUN_SCHEMA_VERSION)
    current.setdefault("sampling_performed", 0)
    current["updated_at"] = _now()
    atomic_write_json(path, current)
    return current


def _frozen_scientific_config(
    args: argparse.Namespace,
    *, parent: Mapping[str, Any], thresholds: BoundaryControlThresholds,
) -> dict[str, Any]:
    return {
        "algorithm": RUN_SCHEMA, "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "model": {
            "schema": MODEL_SCHEMA, "schema_version": MODEL_SCHEMA_VERSION,
            "input_channels": ["N*s", "log1p(N*s)", "tau/T", "label", "periodic_coordinates"],
            "raw_log_density_used": 0, "base_channels": int(args.base_channels),
        },
        "kernel": {key: getattr(args, key) for key in EXPECTED_KERNEL},
        "synthetic_data": {
            "train_paths": int(args.train_paths), "selection_paths": int(args.selection_paths),
            "audit_paths": int(args.audit_paths), "anchors_per_path": int(args.anchors_per_path),
            "anchor_bin_counts": list(args.anchor_bin_counts), "teacher_epsilon": 0.5,
            "teacher_and_null_states_independent": 1,
            "teacher_version": BOUNDED_TEACHER_VERSION,
        },
        "optimization": {
            "train_steps": int(args.train_steps), "batch_size": int(args.batch_size),
            "validation_batch_size": int(args.validation_batch_size),
            "validation_every": int(args.validation_every), "checkpoint_every": int(args.checkpoint_every),
            "learning_rate": float(args.learning_rate), "weight_decay": float(args.weight_decay),
            "ema_decay": float(args.ema_decay), "grad_clip": float(args.grad_clip),
            "clip_warmup_steps": int(args.clip_warmup_steps),
            "training_probes": int(args.training_probes),
            "selection_probes_per_bank": int(args.selection_probes),
            "audit_probes_per_bank": int(args.audit_probes),
            "trace_probes": "randomized-orthogonal-hadamard",
            "trace_probe_version": ORTHOGONAL_HADAMARD_PROBE_VERSION,
            "amp": False,
        },
        "seeds": {
            name: (list(getattr(args, name)) if name in {"teacher_seeds", "null_seeds"} else int(getattr(args, name)))
            for name in (
                "teacher_seeds", "null_seeds", "supervised_seed", "teacher_data_seed",
                "null_data_seed", "calibration_seed", "training_probe_seed",
                "selection_probe_a_seed", "selection_probe_b_seed",
                "audit_probe_a_seed", "audit_probe_b_seed", "bootstrap_seed",
                "batch_index_seed",
            )
        },
        "bootstrap": {"reps": int(args.bootstrap_reps), "confidence": float(args.bootstrap_confidence)},
        "preflight": {"operator_hutchinson_probes": int(args.operator_hutchinson_probes)},
        "thresholds": thresholds.to_dict(),
        "parent_failed_run_sha256": dict(parent.get("artifacts", {})).get("status", {}).get("sha256"),
        "parent_scientific_fingerprint": parent.get("scientific_fingerprint"),
        "sampling_performed": 0,
    }


def _write_or_validate_plan(path: Path, value: Mapping[str, Any], *, resumed: bool) -> dict[str, Any]:
    expected = dict(value)
    if path.is_file():
        if _json_load(path) != expected:
            raise ArtifactCompatibilityError(f"frozen plan mismatch for {path.name}")
    else:
        atomic_write_json(path, expected)
    return expected


def _prepare_control_arrays(
    run_dir: Path,
    *, args: argparse.Namespace, parent: Mapping[str, Any],
    scientific_fingerprint: str, resumed: bool,
) -> tuple[dict[str, ControlArrays], dict[str, Any]]:
    horizon = float(dict(parent["schedule_metadata"])["horizon"])
    # The original boundary-control workflow keeps its historical path-ID
    # ranges.  Versioned follow-up workflows may provide fresh bases without
    # changing the shared array/cache implementation.
    teacher_path_base = int(getattr(args, "teacher_path_base", 4_000_000))
    null_path_base = int(getattr(args, "null_path_base", 5_000_000))
    synthetic_seeds = {
        "teacher_train": _derived_seed(int(args.teacher_data_seed), "bounded_teacher", "train"),
        "teacher_selection": _derived_seed(int(args.teacher_data_seed), "bounded_teacher", "selection"),
        "teacher_audit": _derived_seed(int(args.teacher_data_seed), "bounded_teacher", "audit"),
        "null_train": _derived_seed(int(args.null_data_seed), "dirichlet_null", "train"),
        "null_selection": _derived_seed(int(args.null_data_seed), "dirichlet_null", "selection"),
        "null_audit": _derived_seed(int(args.null_data_seed), "dirichlet_null", "audit"),
    }
    fractions, strata = _time_template(args.anchor_bin_counts)
    time_plan = {
        "schema": RUN_SCHEMA + "-synthetic-time-plan", "schema_version": 1,
        "horizon": horizon, "fractions": fractions.tolist(), "strata": strata.tolist(),
        "anchor_bin_counts": list(args.anchor_bin_counts),
        "parent_schedule_metadata": dict(parent["schedule_metadata"]),
        "physical_time_values_reused": 1, "physical_state_values_reused": 0,
        "sampling_performed": 0,
    }
    time_plan["fingerprint"] = config_fingerprint(time_plan)
    _write_or_validate_plan(run_dir / "synthetic_time_plan.json", time_plan, resumed=resumed)
    split_plan = {
        "schema": RUN_SCHEMA + "-synthetic-split-plan", "schema_version": 1,
        "split_counts": [int(args.train_paths), int(args.selection_paths), int(args.audit_paths)],
        "anchors_per_path": int(args.anchors_per_path),
        "teacher_path_ranges": {
            "train": [teacher_path_base, teacher_path_base + int(args.train_paths) - 1],
            "selection": [teacher_path_base + 100_000, teacher_path_base + 100_000 + int(args.selection_paths) - 1],
            "audit": [teacher_path_base + 200_000, teacher_path_base + 200_000 + int(args.audit_paths) - 1],
        },
        "null_path_ranges": {
            "train": [null_path_base, null_path_base + int(args.train_paths) - 1],
            "selection": [null_path_base + 100_000, null_path_base + 100_000 + int(args.selection_paths) - 1],
            "audit": [null_path_base + 200_000, null_path_base + 200_000 + int(args.audit_paths) - 1],
        },
        "whole_path_isolation": 1, "teacher_null_independent": 1,
        "synthetic_seed_derivation": "sha256(master,law,role)-v1",
        "synthetic_seed_assignments": synthetic_seeds,
        "physical_path_ids_reused": 0, "sampling_performed": 0,
    }
    split_plan["fingerprint"] = config_fingerprint(split_plan)
    _write_or_validate_plan(run_dir / "synthetic_split_plan.json", split_plan, resumed=resumed)
    probe_plan = {
        "schema": RUN_SCHEMA + "-probe-plan", "schema_version": 1,
        "assignment": "fixed-batch-order-randomized-orthogonal-hadamard-v1",
        "common_across_model_seeds": 1,
        "training": {"probes_per_state": int(args.training_probes), "seed": int(args.training_probe_seed), "stream_checkpointed": 1},
        "selection_a": {"probes_per_state": int(args.selection_probes), "seed": int(args.selection_probe_a_seed)},
        "selection_b": {"probes_per_state": int(args.selection_probes), "seed": int(args.selection_probe_b_seed)},
        "audit_a": {"probes_per_state": int(args.audit_probes), "seed": int(args.audit_probe_a_seed)},
        "audit_b": {"probes_per_state": int(args.audit_probes), "seed": int(args.audit_probe_b_seed)},
        "edge_order": "right-then-down-periodic", "sampling_performed": 0,
    }
    probe_plan["fingerprint"] = config_fingerprint(probe_plan)
    _write_or_validate_plan(run_dir / "probe_plan.json", probe_plan, resumed=resumed)

    specs = {
        "teacher_train": ("train", "bounded_teacher", int(args.train_paths), teacher_path_base, synthetic_seeds["teacher_train"]),
        "teacher_selection": ("selection", "bounded_teacher", int(args.selection_paths), teacher_path_base + 100_000, synthetic_seeds["teacher_selection"]),
        "teacher_audit": ("audit", "bounded_teacher", int(args.audit_paths), teacher_path_base + 200_000, synthetic_seeds["teacher_audit"]),
        "null_train": ("train", "dirichlet_null", int(args.train_paths), null_path_base, synthetic_seeds["null_train"]),
        "null_selection": ("selection", "dirichlet_null", int(args.selection_paths), null_path_base + 100_000, synthetic_seeds["null_selection"]),
        "null_audit": ("audit", "dirichlet_null", int(args.audit_paths), null_path_base + 200_000, synthetic_seeds["null_audit"]),
    }
    arrays: dict[str, ControlArrays] = {}
    records: dict[str, Any] = {}
    root = run_dir / "synthetic_arrays"
    root.mkdir(parents=True, exist_ok=True)
    for name, (role, law, paths, first_path, seed) in specs.items():
        path = root / f"{name}.npz"
        binding = {
            "scientific_fingerprint": scientific_fingerprint,
            "time_plan_fingerprint": time_plan["fingerprint"],
            "split_plan_fingerprint": split_plan["fingerprint"],
            "name": name, "role": role, "law": law, "seed": seed,
        }
        sidecar_path = path.with_suffix(".json")
        loaded = False
        if path.is_file() and sidecar_path.is_file():
            try:
                arrays[name], records[name] = _load_arrays(path, binding)
                loaded = True
            except (ArtifactCompatibilityError, KeyError, OSError, ValueError, json.JSONDecodeError):
                # A crash can publish the NPZ before its JSON commit.  Recover
                # deterministic bytes only when the surviving sidecar does not
                # bind a different scientific configuration.
                try:
                    stale_sidecar = _json_load(sidecar_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    stale_sidecar = {}
                stale_binding = stale_sidecar.get("binding")
                if stale_binding is not None and dict(stale_binding) != dict(binding):
                    raise ArtifactCompatibilityError(
                        f"synthetic array resume binding mismatch for {name}"
                    )
        if not loaded:
            path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)
            value = _build_control_arrays(
                role=role, law=law, path_count=paths, first_path_id=first_path,
                bin_counts=args.anchor_bin_counts, horizon=horizon,
                grid_size=int(args.grid_size), seed=seed,
            )
            records[name] = _save_arrays(path, value, binding)
            arrays[name] = value
    registry = {
        "schema": RUN_SCHEMA + "-synthetic-array-registry", "schema_version": 1,
        "records": records, "sampling_performed": 0,
    }
    atomic_write_json(run_dir / "synthetic_array_registry.json", registry)
    return arrays, {"time": time_plan, "split": split_plan, "probes": probe_plan, "arrays": registry}


def _run_production_workload_smoke(
    dynamics: DirectFluxMNISTConfig,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Exercise the actual model's training and audit HVP shapes.

    CUDA production runs use the frozen batch/probe counts.  CPU fixtures keep
    the same code path but use two states and at most four probes so the unit
    suite remains quick.
    """

    production_shape = device.type == "cuda" and int(dynamics.grid_size) == 28
    batch = int(getattr(args, "batch_size", 64)) if production_shape else 2
    training_probes = int(getattr(args, "training_probes", 4)) if production_shape else 1
    audit_probes = int(getattr(args, "audit_probes", 64)) if production_shape else 4
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    try:
        generator_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(260781)
        cells = int(dynamics.grid_size) ** 2
        draws = torch._standard_gamma(
            torch.ones((batch, cells), device=device), generator=generator
        )
        states = draws / draws.sum(dim=1, keepdim=True)
        tau = torch.linspace(
            0.1,
            0.9,
            batch,
            device=device,
            dtype=states.dtype,
        ) * float(natural_horizon(dynamics))
        labels = torch.full((batch,), 3, device=device, dtype=torch.long)
        with torch.random.fork_rng(
            devices=[device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        ):
            torch.manual_seed(260782)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(260782)
            model = D0BoundarySmoothPotentialUNet(
                dynamics, base_channels=int(getattr(args, "base_channels", 32))
            ).to(device)
        with torch.no_grad():
            model.out.weight.fill_(1e-3)
            model.out.bias.zero_()
        train_bank = orthogonal_hadamard_edge_probes(
            training_probes,
            batch,
            int(dynamics.grid_size),
            device=device,
            dtype=states.dtype,
            generator=generator,
        )
        train_objective = dirichlet_score_objective(
            model, tau, states, labels, dynamics, train_bank, create_graph=True
        )
        train_objective.loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        model.zero_grad(set_to_none=True)
        audit_bank = orthogonal_hadamard_edge_probes(
            audit_probes,
            batch,
            int(dynamics.grid_size),
            device=device,
            dtype=states.dtype,
            generator=generator,
        )
        audit_objective = dirichlet_score_objective(
            model, tau, states, labels, dynamics, audit_bank, create_graph=False
        )
        finite = bool(
            torch.isfinite(train_objective.loss)
            and torch.isfinite(audit_objective.per_sample).all()
            and gradients_finite
        )
        result = {
            "device": str(device),
            "production_shape": int(production_shape),
            "grid_size": int(dynamics.grid_size),
            "batch_size": batch,
            "training_probes": training_probes,
            "audit_probes": audit_probes,
            "training_loss": float(train_objective.loss.detach().cpu()),
            "audit_loss": float(audit_objective.loss.detach().cpu()),
            "gradients_finite": int(gradients_finite),
            "finite": int(finite),
            "peak_memory_gib": (
                float(torch.cuda.max_memory_allocated(device)) / float(1024**3)
                if device.type == "cuda"
                else 0.0
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "passed": int(finite),
        }
    except Exception as exc:
        result = {
            "device": str(device),
            "production_shape": int(production_shape),
            "grid_size": int(dynamics.grid_size),
            "batch_size": batch,
            "training_probes": training_probes,
            "audit_probes": audit_probes,
            "type": type(exc).__name__,
            "message": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
            "passed": 0,
        }
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def _run_preflight(
    run_dir: Path,
    *, dynamics: DirectFluxMNISTConfig, args: argparse.Namespace,
    device: torch.device, binding: Mapping[str, Any],
    thresholds: BoundaryControlThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_path = run_dir / "boundary_operator_preflight.json"
    gate_path = run_dir / "boundary_preflight_gate.json"
    if artifact_path.is_file() and gate_path.is_file():
        artifact = _json_load(artifact_path)
        gate = _json_load(gate_path)
        if dict(artifact.get("binding", {})) != dict(binding):
            raise ArtifactCompatibilityError("boundary preflight fingerprint mismatch")
        recomputed = evaluate_boundary_preflight(dict(artifact["gate_metrics"]), thresholds)
        if recomputed != gate:
            raise ArtifactCompatibilityError("boundary preflight evidence and gate disagree")
        return artifact, gate
    raw = run_boundary_operator_preflight(
        dynamics, device=device, hutchinson_probes=int(args.operator_hutchinson_probes)
    )
    workload_smoke = _run_production_workload_smoke(
        dynamics, args=args, device=device
    )
    raw["production_workload_smoke"] = workload_smoke
    raw["passed"] = bool(raw.get("passed", False)) and bool(workload_smoke.get("passed", 0))
    facet = dict(raw.get("facet_ray", {}))
    facet_checks = dict(facet.get("checks", {}))
    model_facet = dict(raw.get("model_facet_ray", facet))
    model_facet_checks = dict(model_facet.get("checks", {}))
    legacy = dict(raw.get("legacy_log_barrier", {}))
    legacy_checks = dict(legacy.get("checks", {}))
    smooth_finite = bool(
        dict(model_facet_checks.get("quantities_finite", {})).get(
            "passed",
            dict(facet_checks.get("smooth_quantities_finite", {})).get("passed", False),
        )
    )
    gate_metrics = {
        "potential_finite": int(smooth_finite), "gradient_finite": int(smooth_finite),
        "hvp_finite": int(smooth_finite), "generator_finite": int(smooth_finite),
        "energy_finite": int(smooth_finite),
        "incident_flux_loglog_slope": model_facet.get(
            "incident_flux_loglog_slope",
            dict(facet_checks.get("conormal_log_log_slope", {})).get("value"),
        ),
        "incident_flux_endpoint_ratio": model_facet.get(
            "incident_flux_endpoint_ratio",
            dict(facet_checks.get("conormal_four_decade_decay", {})).get("value"),
        ),
        "legacy_barrier_rejected": int(
            bool(legacy.get("passed", False))
            and bool(dict(legacy_checks.get("conormal_does_not_vanish", {})).get("passed", False))
            and bool(dict(facet_checks.get("legacy_barrier_nonvanishing", {})).get("passed", False))
        ),
        "legacy_coefficient_relative_error": legacy.get("empirical_relative_error"),
        "operator_pass": int(bool(dict(raw.get("operator", {})).get("passed", False))),
        "orthogonal_probe_pass": int(bool(dict(raw.get("orthogonal_probe_preflight", {})).get("passed", False))),
        "aggregate_preflight_pass": int(bool(raw.get("passed", False))),
        "production_workload_smoke_pass": int(bool(workload_smoke.get("passed", 0))),
    }
    artifact = {
        "schema": RUN_SCHEMA + "-boundary-operator-preflight", "schema_version": 1,
        "binding": dict(binding), "raw": raw, "gate_metrics": gate_metrics,
        "sampling_performed": 0,
    }
    atomic_write_json(artifact_path, artifact)
    rows: list[dict[str, Any]] = []
    for row in facet.get("rows", []):
        rows.append({"fixture": "boundary_smooth_analytic_witness", **dict(row)})
    for row in model_facet.get("rows", []):
        rows.append({"fixture": "boundary_smooth_model", **dict(row)})
    for row in legacy.get("boundary_rows", []):
        rows.append({"fixture": "legacy_log_barrier", **dict(row)})
    atomic_write_csv(run_dir / "boundary_ray_metrics.csv", rows)
    gate = evaluate_boundary_preflight(gate_metrics, thresholds)
    atomic_write_json(gate_path, gate)
    return artifact, gate


def _empty_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "gate": str(name), "passed": 0, "subchecks": {}, "reason": str(reason),
        "sampling_performed": 0,
    }


def _probe_banks_agree(
    *, teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
) -> bool:
    for result in [*teacher_results, *null_results]:
        metrics = dict(result.get("metrics", result))
        if "complete" in metrics and not bool(int(metrics.get("complete", 0))):
            continue
        if "finite" in metrics and not bool(int(metrics.get("finite", 0))):
            continue
        banks = dict(metrics.get("audit_objective_banks", {}))
        if set(banks) != {"a", "b"}:
            return False
        for scope in ("overall", "data_end"):
            signs = []
            for name in ("a", "b"):
                value = dict(dict(banks[name]).get(scope, {})).get("lower_bound")
                if value is None or not math.isfinite(float(value)):
                    return False
                signs.append(float(value) > 0.0)
            if signs[0] != signs[1]:
                return False
    return True


def _failed_task_result(task_kind: str, seed: int, exc: BaseException) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "complete": 0, "finite": 0, "model_seed": int(seed), "selected_step": 0,
        "boundary_admissible": 0, "post_warmup_clip_fraction": None,
        "sampling_performed": 0,
    }
    if task_kind == "null":
        metrics.update({"comparator": "analytic_zero", "audit_objective_banks": {}})
        gate = evaluate_null_seed(metrics)
    else:
        metrics.update(
            {
                "audit_overall_score_gain": None, "audit_data_end_score_gain": None,
                "overall_flux_cosine": None, "time_bin_flux_cosines": [],
                "overall_relative_flux_l2": None, "time_bin_relative_flux_l2": [],
            }
        )
        gate = (
            evaluate_supervised_teacher(metrics)
            if task_kind == "supervised_teacher"
            else evaluate_implicit_teacher_seed(metrics)
        )
    return {
        "task_kind": task_kind, "model_seed": int(seed), "metrics": metrics,
        "gate": gate, "failure": {"type": type(exc).__name__, "message": str(exc)},
        "sampling_performed": 0,
    }


def _null_linear_advisory(
    run_dir: Path,
    *, train: ControlArrays, audit: ControlArrays,
    dynamics: DirectFluxMNISTConfig, args: argparse.Namespace, device: torch.device,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit/report the legacy linear comparator without using it in any gate."""

    root = run_dir / "advisory"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "null_linear_spline.pt"
    sidecar_path = root / "null_linear_spline.json"
    model: D0LinearSplinePotential
    if checkpoint_path.is_file():
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover
            payload = torch.load(checkpoint_path, map_location="cpu")
        if (
            payload.get("schema") != RUN_SCHEMA + "-null-linear-advisory"
            or int(payload.get("schema_version", -1)) != 1
            or dict(payload.get("binding", {})) != dict(binding)
        ):
            raise ArtifactCompatibilityError("null linear advisory fingerprint mismatch")
        model = D0LinearSplinePotential(dynamics, payload["coefficients"].float())
        fit_record = dict(payload["fit"])
    else:
        fit = fit_linear_spline_baseline(
            train.states.to(device), train.tau.to(device), dynamics,
            tolerance=1e-10, max_iterations=2000,
        )
        if not bool(fit.converged) or not math.isfinite(float(fit.relative_residual)):
            raise RuntimeError("null advisory linear-spline fit did not converge")
        model = fit.model.to("cpu")
        fit_record = {
            "iterations": int(fit.iterations),
            "relative_residual": float(fit.relative_residual),
            "converged": int(fit.converged),
        }
        atomic_torch_save(
            checkpoint_path,
            {
                "schema": RUN_SCHEMA + "-null-linear-advisory", "schema_version": 1,
                "coefficients": model.coefficients.detach().cpu(), "fit": fit_record,
                "binding": dict(binding), "eligible_for_gate": 0,
                "sampling_performed": 0,
            },
        )
    components = _risk_values(
        model.to(device), audit, dynamics, device=device,
        batch_size=int(args.validation_batch_size),
        probes_per_state=int(args.audit_probes), probe_seed=int(args.audit_probe_a_seed),
    )
    record = _risk_bank_record(
        components, audit, reps=int(args.bootstrap_reps),
        confidence=float(args.bootstrap_confidence),
        bootstrap_seed=_derived_seed(int(args.bootstrap_seed), "null-linear-advisory"),
    )
    value = {
        "schema": RUN_SCHEMA + "-null-linear-advisory-report", "schema_version": 1,
        "binding": dict(binding), "fit": fit_record, "audit": record,
        "primary_null_comparator": "analytic_zero",
        "role": "advisory_only", "eligible_for_checkpoint_selection": 0,
        "eligible_for_gate": 0, "checkpoint_sha256": file_fingerprint(checkpoint_path),
        "sampling_performed": 0,
    }
    atomic_write_json(sidecar_path, value)
    return value


def _run_controls(
    run_dir: Path,
    *, arrays: Mapping[str, ControlArrays], dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace, device: torch.device, thresholds: BoundaryControlThresholds,
    scientific_fingerprint: str, runtime_fingerprint: str,
    source_fingerprint_value: str, preflight_gate: Mapping[str, Any],
    show_progress: bool,
) -> dict[str, Any]:
    tasks_root = run_dir / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    array_registry_sha256 = file_fingerprint(run_dir / "synthetic_array_registry.json")

    supervised_fp = _task_fingerprints(
        scientific_fingerprint=scientific_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint_value=source_fingerprint_value,
        arrays=arrays["teacher_train"],
        selection_arrays=arrays["teacher_selection"],
        audit_arrays=arrays["teacher_audit"],
        array_registry_sha256=array_registry_sha256,
        task_kind="supervised_teacher",
        model_seed=int(args.supervised_seed), loss_scale=1.0,
    )
    try:
        supervised = _run_control_task(
            task_dir=tasks_root / "supervised-teacher", task_kind="supervised_teacher",
            train=arrays["teacher_train"], selection_arrays=arrays["teacher_selection"],
            audit=arrays["teacher_audit"], dynamics=dynamics, args=args, device=device,
            model_seed=int(args.supervised_seed), loss_scale=1.0,
            fingerprints=supervised_fp, show_progress=show_progress, thresholds=thresholds,
        )
    except Exception as exc:
        supervised = _failed_task_result("supervised_teacher", int(args.supervised_seed), exc)
        failures.append(dict(supervised["failure"]))
        atomic_write_json(tasks_root / "supervised-teacher" / "task_failure.json", supervised)
    atomic_write_json(run_dir / "supervised_teacher_control.json", supervised)
    supervised_gate = dict(supervised["gate"])

    if not bool(int(supervised_gate.get("passed", 0))):
        teacher_study = evaluate_implicit_teacher_study([], thresholds)
        null_study = evaluate_null_study([], thresholds)
        atomic_write_json(run_dir / "implicit_teacher_study.json", teacher_study)
        atomic_write_json(run_dir / "null_study.json", null_study)
        atomic_write_json(
            run_dir / "task_failures.json",
            {"failure_count": len(failures), "failures": failures, "skips": [{"stage": "implicit_controls", "reason": "supervised analytic teacher failed"}]},
        )
        report = evaluate_boundary_control_gates(
            provenance_pass=True, boundary_preflight=preflight_gate,
            supervised_teacher=supervised_gate, implicit_teacher_study=teacher_study,
            null_study=null_study, require_gate=str(args.require_gate),
        )
        atomic_write_json(run_dir / "boundary_control_gate.json", report["controls"])
        atomic_write_json(run_dir / "control_repair_decision.json", report["decision"])
        return report

    calibration_binding = {
        "scientific_fingerprint": scientific_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "source_fingerprint": source_fingerprint_value,
        "teacher_train_identity": _arrays_identity(arrays["teacher_train"]),
        "calibration_seed": int(args.calibration_seed),
        "training_probe_seed": int(args.training_probe_seed),
    }
    calibration = _calibrate_loss_scale(
        run_dir / "loss_scale_calibration.json", arrays=arrays["teacher_train"],
        dynamics=dynamics, args=args, device=device, binding=calibration_binding,
    )
    loss_scale = float(calibration["loss_scale"])
    advisory_binding = {
        "scientific_fingerprint": scientific_fingerprint,
        "null_train_identity": _arrays_identity(arrays["null_train"]),
        "null_audit_identity": _arrays_identity(arrays["null_audit"]),
        "probe_seed": int(args.audit_probe_a_seed),
    }
    try:
        _null_linear_advisory(
            run_dir, train=arrays["null_train"], audit=arrays["null_audit"],
            dynamics=dynamics, args=args, device=device, binding=advisory_binding,
        )
    except Exception as exc:
        atomic_write_json(
            run_dir / "advisory" / "null_linear_spline_warning.json",
            {
                "type": type(exc).__name__, "message": str(exc),
                "role": "advisory_only", "eligible_for_gate": 0,
                "sampling_performed": 0,
            },
        )
    teacher_results: list[dict[str, Any]] = []
    null_results: list[dict[str, Any]] = []

    for seed in args.teacher_seeds:
        task_dir = tasks_root / "implicit-teacher" / f"seed-{int(seed)}"
        fingerprints = _task_fingerprints(
            scientific_fingerprint=scientific_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            source_fingerprint_value=source_fingerprint_value,
            arrays=arrays["teacher_train"],
            selection_arrays=arrays["teacher_selection"],
            audit_arrays=arrays["teacher_audit"],
            array_registry_sha256=array_registry_sha256,
            task_kind="implicit_teacher",
            model_seed=int(seed), loss_scale=loss_scale,
        )
        try:
            result = _run_control_task(
                task_dir=task_dir, task_kind="implicit_teacher",
                train=arrays["teacher_train"], selection_arrays=arrays["teacher_selection"],
                audit=arrays["teacher_audit"], dynamics=dynamics, args=args, device=device,
                model_seed=int(seed), loss_scale=loss_scale, fingerprints=fingerprints,
                show_progress=show_progress, thresholds=thresholds,
            )
        except Exception as exc:
            result = _failed_task_result("implicit_teacher", int(seed), exc)
            failure = {"task_kind": "implicit_teacher", "model_seed": int(seed), **dict(result["failure"])}
            failures.append(failure)
            task_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(task_dir / "task_failure.json", result)
        teacher_results.append(result)

    for seed in args.null_seeds:
        task_dir = tasks_root / "null" / f"seed-{int(seed)}"
        fingerprints = _task_fingerprints(
            scientific_fingerprint=scientific_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            source_fingerprint_value=source_fingerprint_value,
            arrays=arrays["null_train"],
            selection_arrays=arrays["null_selection"],
            audit_arrays=arrays["null_audit"],
            array_registry_sha256=array_registry_sha256,
            task_kind="null",
            model_seed=int(seed), loss_scale=loss_scale,
        )
        try:
            result = _run_control_task(
                task_dir=task_dir, task_kind="null", train=arrays["null_train"],
                selection_arrays=arrays["null_selection"], audit=arrays["null_audit"],
                dynamics=dynamics, args=args, device=device, model_seed=int(seed),
                loss_scale=loss_scale, fingerprints=fingerprints,
                show_progress=show_progress, thresholds=thresholds,
            )
        except Exception as exc:
            result = _failed_task_result("null", int(seed), exc)
            failure = {"task_kind": "null", "model_seed": int(seed), **dict(result["failure"])}
            failures.append(failure)
            task_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(task_dir / "task_failure.json", result)
        null_results.append(result)

    teacher_study = evaluate_implicit_teacher_study(
        [dict(value["metrics"]) for value in teacher_results], thresholds
    )
    null_study = evaluate_null_study(
        [dict(value["metrics"]) for value in null_results], thresholds
    )
    teacher_study["task_results"] = teacher_results
    null_study["task_results"] = null_results
    atomic_write_json(run_dir / "implicit_teacher_study.json", teacher_study)
    atomic_write_json(run_dir / "null_study.json", null_study)
    atomic_write_json(
        run_dir / "task_failures.json",
        {"failure_count": len(failures), "failures": failures, "skips": []},
    )
    banks_agree = _probe_banks_agree(
        teacher_results=teacher_results, null_results=null_results
    )
    report = evaluate_boundary_control_gates(
        provenance_pass=True, boundary_preflight=preflight_gate,
        supervised_teacher=supervised_gate, implicit_teacher_study=teacher_study,
        null_study=null_study, require_gate=str(args.require_gate),
        probe_banks_agree=banks_agree,
    )
    report["probe_banks_agree"] = int(banks_agree)
    atomic_write_json(run_dir / "boundary_control_gate.json", report["controls"])
    atomic_write_json(run_dir / "control_repair_decision.json", report["decision"])
    return report


def _write_report_artifacts(run_dir: Path) -> dict[str, Any]:
    history_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    selection_path_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "tasks").glob("**/training_history.csv")):
        task = path.parent.relative_to(run_dir / "tasks").as_posix()
        with path.open("r", encoding="utf-8", newline="") as handle:
            history_rows.extend({"task": task, **row} for row in csv.DictReader(handle))
    for path in sorted((run_dir / "tasks").glob("**/checkpoint_metrics.csv")):
        task = path.parent.relative_to(run_dir / "tasks").as_posix()
        with path.open("r", encoding="utf-8", newline="") as handle:
            checkpoint_rows.extend({"task": task, **row} for row in csv.DictReader(handle))
    for path in sorted((run_dir / "tasks").glob("**/audit_path_risks.csv")):
        task = path.parent.relative_to(run_dir / "tasks").as_posix()
        with path.open("r", encoding="utf-8", newline="") as handle:
            path_rows.extend({"task": task, **row} for row in csv.DictReader(handle))
    for path in sorted((run_dir / "tasks").glob("**/selection_path_risks.csv")):
        task = path.parent.relative_to(run_dir / "tasks").as_posix()
        with path.open("r", encoding="utf-8", newline="") as handle:
            selection_path_rows.extend(
                {"task": task, **row} for row in csv.DictReader(handle)
            )
    atomic_write_csv(run_dir / "training_component_diagnostics.csv", history_rows)
    atomic_write_csv(run_dir / "checkpoint_metrics.csv", checkpoint_rows)
    atomic_write_csv(
        run_dir / "selection_path_objective_risks.csv", selection_path_rows
    )
    atomic_write_csv(run_dir / "audit_path_objective_risks.csv", path_rows)

    result_records: list[dict[str, Any]] = []
    supervised_path = run_dir / "supervised_teacher_control.json"
    if supervised_path.is_file():
        result_records.append(_json_load(supervised_path))
    for study_name in ("implicit_teacher_study.json", "null_study.json"):
        study_path = run_dir / study_name
        if study_path.is_file():
            result_records.extend(
                dict(value) for value in _json_load(study_path).get("task_results", [])
            )

    teacher_rows: list[dict[str, Any]] = []
    time_bin_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for result in result_records:
        metrics = dict(result.get("metrics", {}))
        task_kind = str(result.get("task_kind", ""))
        banks = dict(metrics.get("audit_objective_banks", {}))

        def bank_lcb(bank: str, scope: str) -> Any:
            return dict(dict(banks.get(bank, {})).get(scope, {})).get("lower_bound")

        seed_row = {
            "task_kind": task_kind,
            "model_seed": result.get("model_seed"),
            "selected_step": metrics.get("selected_step"),
            "complete": metrics.get("complete"),
            "finite": metrics.get("finite"),
            "boundary_admissible": metrics.get("boundary_admissible"),
            "post_warmup_clip_fraction": metrics.get("post_warmup_clip_fraction"),
            "overall_score_gain": metrics.get("audit_overall_score_gain"),
            "data_end_score_gain": metrics.get("audit_data_end_score_gain"),
            "overall_flux_cosine": metrics.get("overall_flux_cosine"),
            "overall_relative_flux_l2": metrics.get("overall_relative_flux_l2"),
            "audit_a_overall_lower_bound": bank_lcb("a", "overall"),
            "audit_a_data_end_lower_bound": bank_lcb("a", "data_end"),
            "audit_b_overall_lower_bound": bank_lcb("b", "overall"),
            "audit_b_data_end_lower_bound": bank_lcb("b", "data_end"),
            "passed": int(dict(result.get("gate", {})).get("passed", 0)),
        }
        seed_rows.append(seed_row)
        if task_kind in {"supervised_teacher", "implicit_teacher"}:
            teacher_rows.append(dict(seed_row))
            for bin_index, bin_metrics in enumerate(metrics.get("time_bins", [])):
                time_bin_rows.append(
                    {
                        "task_kind": task_kind,
                        "model_seed": result.get("model_seed"),
                        "selected_step": metrics.get("selected_step"),
                        "time_bin": int(bin_index),
                        **dict(bin_metrics),
                    }
                )
    atomic_write_csv(run_dir / "analytic_teacher_metrics.csv", teacher_rows)
    atomic_write_csv(run_dir / "control_time_bin_metrics.csv", time_bin_rows)
    atomic_write_csv(run_dir / "control_seed_metrics.csv", seed_rows)

    plots: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if history_rows:
            figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=False)
            tasks = sorted({str(row["task"]) for row in history_rows})
            for task in tasks:
                selected = [row for row in history_rows if row["task"] == task]
                steps = np.asarray([int(row["step"]) for row in selected])
                loss = np.asarray([float(row["unscaled_loss"]) for row in selected])
                gradients = np.asarray([float(row["grad_norm"]) for row in selected])
                axes[0].plot(steps, loss, label=task, alpha=0.8)
                axes[1].plot(steps, gradients, label=task, alpha=0.8)
            axes[0].axhline(0.0, color="black", linewidth=0.8)
            axes[0].set_ylabel("unscaled objective")
            axes[0].set_title("Boundary-control learning curves")
            axes[1].axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="clip=1")
            axes[1].set_yscale("log")
            axes[1].set_xlabel("training step")
            axes[1].set_ylabel("gradient norm")
            axes[0].legend(fontsize=6, ncol=2)
            figure.tight_layout()
            output = run_dir / "control_learning_curves.png"
            figure.savefig(output, dpi=160)
            plt.close(figure)
            plots.append(str(output.resolve()))

        boundary_csv = run_dir / "boundary_ray_metrics.csv"
        if boundary_csv.is_file():
            with boundary_csv.open("r", encoding="utf-8", newline="") as handle:
                boundary_rows = list(csv.DictReader(handle))
            figure, axis = plt.subplots(figsize=(6, 4))
            plotted = False
            for fixture in (
                "boundary_smooth_model",
                "boundary_smooth_analytic_witness",
                "legacy_log_barrier",
            ):
                selected = [row for row in boundary_rows if row.get("fixture") == fixture]
                if not selected:
                    continue
                plotted = True
                axis.loglog(
                    [float(row["epsilon"]) for row in selected],
                    [float(row["incident_conormal_max_abs"]) for row in selected],
                    marker="o", label=fixture,
                )
            axis.set_xlabel("facet mass epsilon")
            axis.set_ylabel("max incident conormal flux")
            axis.set_title("Closed-simplex boundary regression")
            if plotted:
                axis.legend()
            figure.tight_layout()
            output = run_dir / "boundary_ray_refinement.png"
            figure.savefig(output, dpi=160)
            plt.close(figure)
            plots.append(str(output.resolve()))
    except Exception as exc:  # plotting is evidence presentation, not a scientific gate
        atomic_write_json(
            run_dir / "plot_warning.json",
            {"type": type(exc).__name__, "message": str(exc), "scientific_gate_affected": 0},
        )
    return {
        "training_rows": len(history_rows), "checkpoint_rows": len(checkpoint_rows),
        "selection_path_rows": len(selection_path_rows),
        "audit_path_rows": len(path_rows), "teacher_metric_rows": len(teacher_rows),
        "time_bin_rows": len(time_bin_rows), "seed_metric_rows": len(seed_rows),
        "plots": plots,
    }


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    terminal_exclusions = {"artifact_registry.json", "run_status.json"}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in terminal_exclusions:
            records[path.relative_to(run_dir).as_posix()] = _artifact_record(path)
    return {
        "schema": RUN_SCHEMA + "-artifact-registry", "schema_version": 1,
        "records": records,
        "terminal_files_excluded_to_avoid_self_reference": sorted(terminal_exclusions),
        "sampling_performed": 0,
    }


def _finish(
    run_dir: Path,
    *, report: Mapping[str, Any], phase: str, stage: str,
    skips: Sequence[Mapping[str, Any]],
    execution_failed: bool = False,
) -> int:
    decision = dict(report.get("decision", {}))
    required_pass = 0 if execution_failed else int(report.get("required_gate_pass", 0))
    final_skips = [dict(value) for value in skips]
    try:
        _write_report_artifacts(run_dir)
    except Exception as exc:
        execution_failed = True
        required_pass = 0
        final_skips.append(
            {
                "stage": "report_artifacts",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(
            run_dir / "report_artifact_failure.json",
            {
                "schema": RUN_SCHEMA + "-report-artifact-failure",
                "schema_version": 1,
                "type": type(exc).__name__,
                "message": str(exc),
                "sampling_performed": 0,
            },
        )
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _json_load(manifest_path)
        manifest["artifacts"] = {
            "boundary_preflight": str((run_dir / "boundary_preflight_gate.json").resolve()) if (run_dir / "boundary_preflight_gate.json").is_file() else None,
            "boundary_control_gate": str((run_dir / "boundary_control_gate.json").resolve()) if (run_dir / "boundary_control_gate.json").is_file() else None,
            "control_repair_decision": str((run_dir / "control_repair_decision.json").resolve()) if (run_dir / "control_repair_decision.json").is_file() else None,
            "artifact_registry": str((run_dir / "artifact_registry.json").resolve()),
        }
        atomic_write_json(manifest_path, manifest)
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_record = _artifact_record(run_dir / "artifact_registry.json")
    _write_status(
        run_dir,
        status="failed" if execution_failed else "complete",
        outcome=(
            "implementation_error"
            if execution_failed
            else ("complete" if required_pass else "gate_failed")
        ),
        phase=str(phase), stage=str(stage), required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=str(decision.get("decision", "controls_not_run")),
        recommended_next_action=decision.get("recommended_next_action"),
        skips=final_skips,
        physical_training_authorized=(
            0 if execution_failed else int(decision.get("physical_training_authorized", 0))
        ),
        physical_training_performed=0, sampling_authorized=0, sampling_performed=0,
        artifact_registry_sha256=registry_record["sha256"],
        artifact_registry_size=registry_record["size"],
    )
    return 2 if execution_failed else (0 if required_pass else 2)


def _pending_preflight_report(
    *, provenance: Mapping[str, Any], preflight_gate: Mapping[str, Any], require_gate: str
) -> dict[str, Any]:
    passed = bool(int(provenance.get("passed", 0))) and bool(int(preflight_gate.get("passed", 0)))
    required_pass = True if require_gate == "none" else (passed if require_gate == "preflight" else False)
    return {
        "schema": RUN_SCHEMA + "-gate-report", "schema_version": 1,
        "required_gate": str(require_gate), "required_gate_pass": int(required_pass),
        "preflight_pass": int(passed),
        "controls": _empty_gate("boundary_controls", "controls stage not run"),
        "decision": {
            "decision": "controls_not_run",
            "recommended_next_action": "resume this run with --stage all",
            "physical_training_authorized": 0, "sampling_authorized": 0,
            "sampling_performed": 0,
        },
        "sampling_performed": 0,
    }


def _verify_terminal_artifact_registry(run_dir: Path) -> dict[str, Any]:
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    if not registry_path.is_file() or not status_path.is_file():
        raise ArtifactCompatibilityError("report requires a finalized artifact registry and status")
    status = _json_load(status_path)
    if (
        status.get("artifact_registry_sha256") != file_fingerprint(registry_path)
        or int(status.get("artifact_registry_size", -1)) != int(registry_path.stat().st_size)
    ):
        raise ArtifactCompatibilityError("terminal status does not bind the artifact registry")
    registry = _json_load(registry_path)
    if registry.get("schema") != RUN_SCHEMA + "-artifact-registry":
        raise ArtifactCompatibilityError("artifact registry schema is incompatible")
    records = dict(registry.get("records", {}))
    excluded = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    expected_excluded = {"artifact_registry.json", "run_status.json"}
    if excluded != expected_excluded:
        raise ArtifactCompatibilityError("artifact registry terminal exclusions are incompatible")
    actual_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    if set(records) != actual_files:
        raise ArtifactCompatibilityError("artifact registry is incomplete or contains stale records")
    for relative, raw_record in records.items():
        artifact = (run_dir / relative).resolve()
        try:
            artifact.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ArtifactCompatibilityError(
                f"registered artifact escapes the run directory: {relative}"
            ) from exc
        record = dict(raw_record)
        if (
            Path(str(record.get("path", ""))).resolve() != artifact
            or not artifact.is_file()
            or record.get("sha256") != file_fingerprint(artifact)
            or int(record.get("size", -1)) != int(artifact.stat().st_size)
        ):
            raise ArtifactCompatibilityError(f"registered artifact hash mismatch: {relative}")
    return registry


def _expected_report_task_fingerprints(
    run_dir: Path, *, task_kind: str, seed: int
) -> dict[str, Any]:
    manifest = _json_load(run_dir / "run_manifest.json")
    scientific = dict(manifest.get("scientific_config", {}))
    configured_seeds = dict(scientific.get("seeds", {}))
    if task_kind == "supervised_teacher":
        expected_seeds = {int(configured_seeds.get("supervised_seed", -1))}
        names = ("teacher_train", "teacher_selection", "teacher_audit")
        loss_scale = 1.0
    elif task_kind == "implicit_teacher":
        expected_seeds = {int(value) for value in configured_seeds.get("teacher_seeds", [])}
        names = ("teacher_train", "teacher_selection", "teacher_audit")
        calibration = _json_load(run_dir / "loss_scale_calibration.json")
        loss_scale = float(calibration["loss_scale"])
    elif task_kind == "null":
        expected_seeds = {int(value) for value in configured_seeds.get("null_seeds", [])}
        names = ("null_train", "null_selection", "null_audit")
        calibration = _json_load(run_dir / "loss_scale_calibration.json")
        loss_scale = float(calibration["loss_scale"])
    else:
        raise ArtifactCompatibilityError(f"unknown report task kind {task_kind!r}")
    if int(seed) not in expected_seeds:
        raise ArtifactCompatibilityError(
            f"report task seed {seed} is not frozen for {task_kind}"
        )
    array_registry_path = run_dir / "synthetic_array_registry.json"
    array_registry = _json_load(array_registry_path)
    array_records = dict(array_registry.get("records", {}))
    identities = {
        role: dict(array_records[name].get("identity", {}))
        for role, name in zip(("train", "selection", "audit"), names)
    }
    return {
        "scientific_fingerprint": str(manifest.get("scientific_fingerprint", "")),
        "runtime_fingerprint": str(manifest.get("runtime_fingerprint", "")),
        "source_fingerprint": str(manifest.get("source_fingerprint", "")),
        "array_identities": identities,
        "array_registry_sha256": file_fingerprint(array_registry_path),
        "task_kind": str(task_kind),
        "model_seed": int(seed),
        "loss_scale": float(loss_scale),
        "model_schema": MODEL_SCHEMA,
        "model_schema_version": MODEL_SCHEMA_VERSION,
    }


def _verify_report_task(
    run_dir: Path,
    *,
    result: Mapping[str, Any],
    thresholds: BoundaryControlThresholds,
) -> dict[str, Any]:
    value = dict(result)
    task_kind = str(value.get("task_kind", ""))
    seed = int(value.get("model_seed", -1))
    if task_kind == "supervised_teacher":
        task_dir = run_dir / "tasks" / "supervised-teacher"
        evaluator = evaluate_supervised_teacher
    elif task_kind == "implicit_teacher":
        task_dir = run_dir / "tasks" / "implicit-teacher" / f"seed-{seed}"
        evaluator = evaluate_implicit_teacher_seed
    elif task_kind == "null":
        task_dir = run_dir / "tasks" / "null" / f"seed-{seed}"
        evaluator = evaluate_null_seed
    else:
        raise ArtifactCompatibilityError(f"unknown report task kind {task_kind!r}")

    expected_fingerprints = _expected_report_task_fingerprints(
        run_dir, task_kind=task_kind, seed=seed
    )
    fingerprints = value.get("fingerprints")
    if isinstance(fingerprints, Mapping):
        if dict(fingerprints) != expected_fingerprints:
            raise ArtifactCompatibilityError(
                f"report task fingerprint is not the frozen run fingerprint for {task_kind} seed {seed}"
            )
        completed = _load_completed_task(task_dir, fingerprints=expected_fingerprints)
        if completed is None or completed != value:
            raise ArtifactCompatibilityError(
                f"report task chain is incomplete or stale for {task_kind} seed {seed}"
            )
    else:
        failure_path = task_dir / "task_failure.json"
        if not failure_path.is_file() or _json_load(failure_path) != value:
            raise ArtifactCompatibilityError(
                f"report failure evidence is incomplete for {task_kind} seed {seed}"
            )

    recomputed_gate = evaluator(dict(value.get("metrics", {})), thresholds)
    if recomputed_gate != dict(value.get("gate", {})):
        raise ArtifactCompatibilityError(
            f"report task gate disagrees with metrics for {task_kind} seed {seed}"
        )
    return value


def _verify_study(
    run_dir: Path,
    *,
    path: Path,
    task_kind: str,
    thresholds: BoundaryControlThresholds,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stored = _json_load(path)
    results = [
        _verify_report_task(run_dir, result=dict(value), thresholds=thresholds)
        for value in list(stored.get("task_results", []))
    ]
    if any(str(value.get("task_kind")) != task_kind for value in results):
        raise ArtifactCompatibilityError(f"{path.name} contains the wrong task kind")
    if task_kind == "implicit_teacher":
        recomputed = evaluate_implicit_teacher_study(
            [dict(value["metrics"]) for value in results], thresholds
        )
    elif task_kind == "null":
        recomputed = evaluate_null_study(
            [dict(value["metrics"]) for value in results], thresholds
        )
    else:  # pragma: no cover - private helper contract
        raise ValueError(task_kind)
    expected = dict(recomputed)
    if "task_results" in stored:
        expected["task_results"] = results
    if expected != stored:
        raise ArtifactCompatibilityError(f"{path.name} disagrees with verified task evidence")
    return stored, results


def _report_from_artifacts(
    run_dir: Path, *, require_gate: str, thresholds: BoundaryControlThresholds
) -> dict[str, Any]:
    _verify_terminal_artifact_registry(run_dir)
    manifest = _json_load(run_dir / "run_manifest.json")
    provenance = _json_load(run_dir / "failed_run_provenance.json")
    preflight_path = run_dir / "boundary_preflight_gate.json"
    preflight = _json_load(preflight_path) if preflight_path.is_file() else _empty_gate(
        "boundary_preflight", "not run"
    )
    preflight_evidence_path = run_dir / "boundary_operator_preflight.json"
    if preflight_path.is_file() and not preflight_evidence_path.is_file():
        raise ArtifactCompatibilityError("report preflight evidence is missing")
    if preflight_evidence_path.is_file():
        evidence = _json_load(preflight_evidence_path)
        expected_binding = {
            "scientific_fingerprint": manifest.get("scientific_fingerprint"),
            "runtime_fingerprint": manifest.get("runtime_fingerprint"),
            "source_fingerprint": manifest.get("source_fingerprint"),
            "failed_run_status_sha256": dict(provenance.get("artifacts", {}))
            .get("status", {})
            .get("sha256"),
        }
        if dict(evidence.get("binding", {})) != expected_binding:
            raise ArtifactCompatibilityError("report preflight binding is stale or foreign")
        if evaluate_boundary_preflight(dict(evidence.get("gate_metrics", {})), thresholds) != preflight:
            raise ArtifactCompatibilityError("report preflight gate disagrees with its evidence")
    if not (run_dir / "supervised_teacher_control.json").is_file():
        return _pending_preflight_report(
            provenance=provenance, preflight_gate=preflight, require_gate=require_gate
        )
    supervised_record = _verify_report_task(
        run_dir,
        result=_json_load(run_dir / "supervised_teacher_control.json"),
        thresholds=thresholds,
    )
    supervised = dict(supervised_record["gate"])
    teacher_path = run_dir / "implicit_teacher_study.json"
    null_path = run_dir / "null_study.json"
    if not teacher_path.is_file() or not null_path.is_file():
        raise ArtifactCompatibilityError("report control-study artifacts are incomplete")
    teacher, teacher_results = _verify_study(
        run_dir, path=teacher_path, task_kind="implicit_teacher", thresholds=thresholds
    )
    null, null_results = _verify_study(
        run_dir, path=null_path, task_kind="null", thresholds=thresholds
    )
    banks_agree = _probe_banks_agree(
        teacher_results=teacher_results, null_results=null_results
    )
    report = evaluate_boundary_control_gates(
        provenance_pass=provenance, boundary_preflight=preflight,
        supervised_teacher=supervised, implicit_teacher_study=teacher,
        null_study=null, require_gate=require_gate, probe_banks_agree=banks_agree,
    )
    report["probe_banks_agree"] = int(banks_agree)
    atomic_write_json(run_dir / "boundary_control_gate.json", report["controls"])
    atomic_write_json(run_dir / "control_repair_decision.json", report["decision"])
    return report


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    print(f"boundary-control run directory: {run_dir.resolve()}", flush=True)
    thresholds = BoundaryControlThresholds(
        bootstrap_confidence=float(args.bootstrap_confidence)
    )
    try:
        device = _device(args.device)
        backend = configure_exact_torch_backend(device)
        source_hash, source_paths = _source_record()
        runtime = _runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
    except Exception as exc:
        if resumed:
            if not bool(args.no_progress):
                print(
                    f"boundary controls resume rejected without mutation during runtime setup: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return 2
        failure = {
            "schema": RUN_SCHEMA + "-failure",
            "schema_version": 1,
            "type": type(exc).__name__,
            "message": str(exc),
            "stage": str(args.stage),
            "phase": "runtime_setup",
            "sampling_performed": 0,
        }
        atomic_write_json(run_dir / "failure.json", failure)
        report = evaluate_boundary_control_gates(
            provenance_pass=False,
            boundary_preflight=False,
            supervised_teacher=False,
            implicit_teacher_study=False,
            null_study=False,
            require_gate=str(args.require_gate),
        )
        atomic_write_json(run_dir / "boundary_control_gate.json", report["controls"])
        atomic_write_json(run_dir / "control_repair_decision.json", report["decision"])
        atomic_write_json(run_dir / "boundary_control_report.json", report)
        _finish(
            run_dir,
            report=report,
            phase="runtime_setup_failure",
            stage=str(args.stage),
            skips=[{"stage": "all", "reason": f"{type(exc).__name__}: {exc}"}],
            execution_failed=True,
        )
        if not bool(args.no_progress):
            print(
                f"boundary controls failed during runtime setup: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        return 2
    mutation_started = False
    try:
        parent = verify_failed_score_run(args.failed_score_run_dir)
        provenance = {
            "schema": RUN_SCHEMA + "-failed-run-provenance", "schema_version": 1,
            **parent, "source_fingerprint": source_hash, "sampling_performed": 0,
        }
        provenance_path = run_dir / "failed_run_provenance.json"
        if provenance_path.is_file():
            if _json_load(provenance_path) != provenance:
                raise ArtifactCompatibilityError("resume failed-run provenance mismatch")
        else:
            if resumed:
                raise ArtifactCompatibilityError("resume is missing failed-run provenance")
            atomic_write_json(provenance_path, provenance)
        advisory_path = run_dir / "legacy_singular_teacher_advisory.json"
        advisory = {
            "schema": RUN_SCHEMA + "-legacy-singular-teacher-advisory",
            "schema_version": 1,
            "teacher_family": "nonuniform_dirichlet_log_density_ratio",
            "role": "advisory_singular_stress_test_only",
            "rerun_in_boundary_control_workflow": 0,
            "eligible_for_control_gate": 0,
            "failed_parent_gate": dict(parent.get("failed_teacher_gate", {})),
            "failed_parent_status_sha256": dict(parent["artifacts"])["status"]["sha256"],
            "sampling_performed": 0,
        }
        if advisory_path.is_file():
            if _json_load(advisory_path) != advisory:
                raise ArtifactCompatibilityError("legacy teacher advisory fingerprint mismatch")
        else:
            if resumed:
                raise ArtifactCompatibilityError("resume is missing legacy teacher advisory provenance")
            atomic_write_json(advisory_path, advisory)

        scientific = _frozen_scientific_config(args, parent=parent, thresholds=thresholds)
        scientific_fingerprint = config_fingerprint(scientific)
        manifest_value = {
            "schema": RUN_SCHEMA, "schema_version": RUN_SCHEMA_VERSION,
            "created_at": _now(), "run_dir": str(run_dir.resolve()),
            "scientific_config": scientific,
            "scientific_fingerprint": scientific_fingerprint,
            "runtime": runtime, "runtime_fingerprint": runtime_fingerprint,
            "source_fingerprint": source_hash, "source_paths": source_paths,
            "failed_run_provenance_sha256": file_fingerprint(provenance_path),
            "claim_scope": CLAIM_SCOPE, "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            existing = _json_load(manifest_path)
            for key in (
                "schema", "schema_version", "scientific_config", "scientific_fingerprint",
                "runtime", "runtime_fingerprint", "source_fingerprint", "source_paths",
                "failed_run_provenance_sha256", "claim_scope",
            ):
                if existing.get(key) != manifest_value.get(key):
                    raise ArtifactCompatibilityError(f"resume manifest mismatch for {key}")
        else:
            if resumed:
                raise ArtifactCompatibilityError("resume is missing its frozen manifest")
            atomic_write_json(manifest_path, manifest_value)

        if resumed and str(args.stage) == "report":
            # Report mode is read/verify-first: reject missing or altered
            # terminal evidence before changing even run_status.json.
            _verify_terminal_artifact_registry(run_dir)

        previous_status = (
            _json_load(run_dir / "run_status.json")
            if (run_dir / "run_status.json").is_file()
            else {}
        )
        _write_status(
            run_dir, status="running", phase="provenance", stage=str(args.stage),
            require_gate=str(args.require_gate),
            attempt_count=int(previous_status.get("attempt_count", 0)) + 1,
            physical_training_performed=0, sampling_performed=0,
        )
        mutation_started = True

        if str(args.stage) == "report":
            report = _report_from_artifacts(
                run_dir, require_gate=str(args.require_gate), thresholds=thresholds
            )
            atomic_write_json(run_dir / "boundary_control_report.json", report)
            return _finish(run_dir, report=report, phase="report", stage="report", skips=[])

        dynamics = _make_dynamics(args)
        parent_horizon = float(dict(parent["schedule_metadata"])["horizon"])
        if not math.isclose(
            parent_horizon, float(natural_horizon(dynamics)), rel_tol=1e-12, abs_tol=1e-18
        ):
            raise ArtifactCompatibilityError("failed parent and repaired model horizons differ")
        preflight_binding = {
            "scientific_fingerprint": scientific_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "source_fingerprint": source_hash,
            "failed_run_status_sha256": dict(parent["artifacts"])["status"]["sha256"],
        }
        _write_status(run_dir, status="running", phase="preflight")
        _, preflight_gate = _run_preflight(
            run_dir, dynamics=dynamics, args=args, device=device,
            binding=preflight_binding, thresholds=thresholds,
        )
        if str(args.stage) == "preflight" or not bool(int(preflight_gate.get("passed", 0))):
            report = _pending_preflight_report(
                provenance=provenance, preflight_gate=preflight_gate,
                require_gate=str(args.require_gate),
            )
            if not bool(int(preflight_gate.get("passed", 0))):
                report = evaluate_boundary_control_gates(
                    provenance_pass=provenance, boundary_preflight=preflight_gate,
                    supervised_teacher=False, implicit_teacher_study=False,
                    null_study=False, require_gate=str(args.require_gate),
                )
                atomic_write_json(run_dir / "boundary_control_gate.json", report["controls"])
                atomic_write_json(run_dir / "control_repair_decision.json", report["decision"])
            atomic_write_json(run_dir / "boundary_control_report.json", report)
            skips = [] if bool(int(preflight_gate.get("passed", 0))) else [{"stage": "controls", "reason": "boundary/operator preflight failed"}]
            return _finish(run_dir, report=report, phase="preflight", stage=str(args.stage), skips=skips)

        _write_status(run_dir, status="running", phase="synthetic_data")
        arrays, plans = _prepare_control_arrays(
            run_dir, args=args, parent=parent,
            scientific_fingerprint=scientific_fingerprint, resumed=resumed,
        )
        atomic_write_json(
            run_dir / "control_plan_registry.json",
            {
                "schema": RUN_SCHEMA + "-control-plan-registry", "schema_version": 1,
                "time_plan_fingerprint": plans["time"]["fingerprint"],
                "split_plan_fingerprint": plans["split"]["fingerprint"],
                "probe_plan_fingerprint": plans["probes"]["fingerprint"],
                "synthetic_array_registry_sha256": file_fingerprint(run_dir / "synthetic_array_registry.json"),
                "sampling_performed": 0,
            },
        )
        _write_status(run_dir, status="running", phase="controls")
        report = _run_controls(
            run_dir, arrays=arrays, dynamics=dynamics, args=args, device=device,
            thresholds=thresholds, scientific_fingerprint=scientific_fingerprint,
            runtime_fingerprint=runtime_fingerprint, source_fingerprint_value=source_hash,
            preflight_gate=preflight_gate, show_progress=not bool(args.no_progress),
        )
        atomic_write_json(run_dir / "boundary_control_report.json", report)
        return _finish(run_dir, report=report, phase="controls", stage=str(args.stage), skips=[])
    except Exception as exc:
        if resumed and not mutation_started:
            if not bool(args.no_progress):
                print(
                    f"boundary controls resume rejected without mutation: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            return 2
        failure = {
            "schema": RUN_SCHEMA + "-failure", "schema_version": 1,
            "type": type(exc).__name__, "message": str(exc),
            "stage": str(args.stage), "sampling_performed": 0,
        }
        atomic_write_json(run_dir / "failure.json", failure)
        provenance = (
            _json_load(run_dir / "failed_run_provenance.json")
            if (run_dir / "failed_run_provenance.json").is_file()
            else {"passed": 0, "reason": str(exc), "sampling_performed": 0}
        )
        preflight = (
            _json_load(run_dir / "boundary_preflight_gate.json")
            if (run_dir / "boundary_preflight_gate.json").is_file()
            else _empty_gate("boundary_preflight", f"{type(exc).__name__}: {exc}")
        )
        report = evaluate_boundary_control_gates(
            provenance_pass=provenance, boundary_preflight=preflight,
            supervised_teacher=False, implicit_teacher_study=False, null_study=False,
            require_gate=str(args.require_gate),
        )
        atomic_write_json(run_dir / "boundary_control_gate.json", report["controls"])
        atomic_write_json(run_dir / "control_repair_decision.json", report["decision"])
        atomic_write_json(run_dir / "boundary_control_report.json", report)
        _finish(
            run_dir, report=report, phase="failure", stage=str(args.stage),
            skips=[{"stage": "remaining", "reason": f"{type(exc).__name__}: {exc}"}],
            execution_failed=True,
        )
        if not bool(args.no_progress):
            print(f"boundary controls failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
