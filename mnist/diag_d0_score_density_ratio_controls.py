"""Boundary-admissible D0 density-ratio classification controls.

This additive, controls-only workflow is the predeclared fallback after the
streamed implicit-score confirmation.  It learns an equal-prior binary logit
between the exact bounded teacher and ``Dirichlet(1)`` and differentiates that
logit only for held-out analytic score/flux diagnostics.  It never trains on
physical score states and never imports or invokes a reverse sampler.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
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
import torch.nn.functional as F

from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
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
from mnist.d0_score_boundary_controls import (
    BOUNDARY_SMOOTH_MODEL_VERSION,
    D0BoundarySmoothPotentialUNet,
    bounded_teacher_cell_score,
    bounded_teacher_density_ratio,
    bounded_teacher_log_relative_potential,
    bounded_teacher_physical_flux,
)
from mnist.d0_dirichlet_score import (
    edge_difference_channels,
    physical_flux_from_edge_score,
)
from mnist.d0_score_density_ratio import (
    DENSITY_RATIO_OBJECTIVE_VERSION,
    DENSITY_RATIO_PANEL_VERSION,
    DENSITY_RATIO_STREAM_VERSION,
    DensityRatioPanel,
    DensityRatioStreamPlan,
    analytic_teacher_metrics,
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
    calibrate_density_ratio_loss_scale,
    classification_loss,
    density_ratio_replay_record,
    evaluate_classification_panel,
    equal_prior_bayes_logit,
    generate_density_ratio_batch,
    load_density_ratio_panel,
    panel_disjointness_record,
    panel_identity,
    save_density_ratio_panel,
    stream_plan_record,
    verify_density_ratio_replay,
)
from mnist.d0_score_control_stability import bootstrap_path_mean_interval
from mnist.d0_score_density_ratio_gate import (
    DensityRatioThresholds,
    evaluate_density_ratio_controls,
    evaluate_density_ratio_pilot,
    evaluate_density_ratio_workflow,
    evaluate_null_seed,
    evaluate_ratio_preflight,
    evaluate_teacher_seed,
    nominate_checkpoint_on_a,
    select_density_ratio_checkpoint,
    select_density_ratio_profile,
)
from mnist.d0_score_density_ratio_provenance import verify_parent_stability_run
from mnist.d0_score_optimizer_scale import (
    scaled_backward_and_clip,
    summarize_scaled_gradient_history,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    init_ema_state,
    natural_horizon,
    temporary_ema_weights,
    update_ema_state,
)


RUN_SCHEMA = "experiment12-d0-score-density-ratio-controls"
RUN_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
CLAIM_SCOPE = "boundary-admissible exact synthetic density-ratio controls only"

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
    "root_seed": 260821,
    "pilot_learning_rates": (3e-4, 1e-4, 3e-5, 1e-5),
    "pilot_steps": 2000,
    "confirm_steps": 4000,
    "pilot_selection_paths": 16,
    "confirm_selection_paths": 32,
    "confirm_audit_paths": 32,
    "base_channels": 32,
    "batch_size": 64,
    "validation_batch_size": 64,
    "weight_decay": 1e-4,
    "ema_decay": 0.99,
    "grad_clip": 1.0,
    "clip_warmup_steps": 500,
    "calibration_state_count": 256,
    "initial_grad_target": 0.10,
    "bootstrap_reps": 10_000,
    "bootstrap_confidence": 0.90,
    "preflight_paths": 128,
    "preflight_confidence": 0.99,
    "confirm_model_seeds": (260831, 260832, 260833),
    "pilot_validation_steps": (
        0, 25, 50, 100, 150, 250, 500, 750, 1000, 1500, 2000
    ),
    "confirm_dense_validation_steps": (0, 25, 50, 100, 150, 250),
    "confirm_validation_every": 250,
}


def _json_load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("expected finite comma-separated numbers")
    return result


def _semantic_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        try:
            values = tuple(actual)
        except TypeError:
            return False
        return len(values) == len(expected) and all(
            _semantic_equal(left, right) for left, right in zip(values, expected)
        )
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "gate": str(name),
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "reason": str(reason),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "pilot", "confirm", "report", "all"),
        default="all",
    )
    parser.add_argument("--parent-stability-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("runs/experiment12_d0_score_density_ratio_controls"),
    )
    parser.add_argument("--run-name", default="production-density-ratio-controls")
    parser.add_argument(
        "--require-gate", choices=("none", "preflight", "pilot", "controls"),
        default="none",
    )
    parser.add_argument("--root-seed", type=int, default=DEFAULTS["root_seed"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument("--grid-size", type=int, default=DEFAULTS["grid_size"])
    parser.add_argument("--sample-steps", type=int, default=DEFAULTS["sample_steps"])
    parser.add_argument(
        "--reference-substeps", type=int, default=DEFAULTS["reference_substeps"]
    )
    parser.add_argument("--tau-eff", type=float, default=DEFAULTS["tau_eff"])
    parser.add_argument(
        "--edge-alpha-mode", choices=("alpha_eff",),
        default=DEFAULTS["edge_alpha_mode"],
    )
    parser.add_argument("--alpha-eff", type=float, default=DEFAULTS["alpha_eff"])
    parser.add_argument("--mass-floor", type=float, default=DEFAULTS["mass_floor"])
    parser.add_argument(
        "--limiter-fraction", type=float, default=DEFAULTS["limiter_fraction"]
    )
    parser.add_argument("--lambda-mix", type=float, default=DEFAULTS["lambda_mix"])

    parser.add_argument(
        "--pilot-learning-rates", type=_parse_csv_floats,
        default=DEFAULTS["pilot_learning_rates"],
    )
    parser.add_argument("--pilot-steps", type=int, default=DEFAULTS["pilot_steps"])
    parser.add_argument("--confirm-steps", type=int, default=DEFAULTS["confirm_steps"])
    parser.add_argument(
        "--pilot-selection-paths", type=int,
        default=DEFAULTS["pilot_selection_paths"],
    )
    parser.add_argument(
        "--confirm-selection-paths", type=int,
        default=DEFAULTS["confirm_selection_paths"],
    )
    parser.add_argument(
        "--confirm-audit-paths", type=int,
        default=DEFAULTS["confirm_audit_paths"],
    )
    parser.add_argument("--base-channels", type=int, default=DEFAULTS["base_channels"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument(
        "--validation-batch-size", type=int,
        default=DEFAULTS["validation_batch_size"],
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--ema-decay", type=float, default=DEFAULTS["ema_decay"])
    parser.add_argument("--grad-clip", type=float, default=DEFAULTS["grad_clip"])
    parser.add_argument(
        "--clip-warmup-steps", type=int, default=DEFAULTS["clip_warmup_steps"]
    )
    parser.add_argument(
        "--calibration-state-count", type=int,
        default=DEFAULTS["calibration_state_count"],
    )
    parser.add_argument(
        "--initial-grad-target", type=float,
        default=DEFAULTS["initial_grad_target"],
    )
    parser.add_argument(
        "--bootstrap-reps", type=int, default=DEFAULTS["bootstrap_reps"]
    )
    parser.add_argument(
        "--bootstrap-confidence", type=float,
        default=DEFAULTS["bootstrap_confidence"],
    )
    parser.add_argument(
        "--preflight-paths", type=int, default=DEFAULTS["preflight_paths"]
    )
    parser.add_argument(
        "--preflight-confidence", type=float,
        default=DEFAULTS["preflight_confidence"],
    )
    parser.add_argument(
        "--confirm-model-seeds", type=_parse_csv_ints,
        default=DEFAULTS["confirm_model_seeds"],
    )
    parser.add_argument(
        "--pilot-validation-steps", type=_parse_csv_ints,
        default=DEFAULTS["pilot_validation_steps"],
    )
    parser.add_argument(
        "--confirm-dense-validation-steps", type=_parse_csv_ints,
        default=DEFAULTS["confirm_dense_validation_steps"],
    )
    parser.add_argument(
        "--confirm-validation-every", type=int,
        default=DEFAULTS["confirm_validation_every"],
    )
    args = parser.parse_args(argv)

    for name in (
        "grid_size", "sample_steps", "reference_substeps", "pilot_steps",
        "confirm_steps", "pilot_selection_paths", "confirm_selection_paths",
        "confirm_audit_paths", "base_channels", "batch_size",
        "validation_batch_size", "calibration_state_count", "bootstrap_reps",
        "preflight_paths",
        "confirm_validation_every",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.batch_size) != 64:
        parser.error("density-ratio stream schema v1 requires --batch-size 64")
    if len(args.confirm_model_seeds) != 3 or len(set(args.confirm_model_seeds)) != 3:
        parser.error("--confirm-model-seeds must contain three distinct seeds")
    if any(value <= 0.0 for value in args.pilot_learning_rates):
        parser.error("pilot learning rates must be positive")
    if tuple(sorted(set(args.pilot_validation_steps))) != tuple(args.pilot_validation_steps):
        parser.error("pilot validation steps must be sorted and unique")
    if args.pilot_validation_steps[0] != 0 or args.pilot_validation_steps[-1] != args.pilot_steps:
        parser.error("pilot validation steps must include zero and pilot-steps")
    if args.confirm_dense_validation_steps[0] != 0:
        parser.error("confirm dense validation steps must begin at zero")
    if not 0.0 < float(args.ema_decay) < 1.0:
        parser.error("--ema-decay must lie in (0,1)")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        parser.error("--bootstrap-confidence must lie in (0,1)")
    if not 0.0 < float(args.preflight_confidence) < 1.0:
        parser.error("--preflight-confidence must lie in (0,1)")
    if int(args.calibration_state_count) % 64:
        parser.error("--calibration-state-count must be divisible by 64")
    if not math.isfinite(float(args.initial_grad_target)) or args.initial_grad_target <= 0.0:
        parser.error("--initial-grad-target must be finite and positive")
    if args.stage in {"confirm", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    if args.require_gate != "none":
        mismatches = [
            f"{key}={getattr(args, key)!r}, expected {expected!r}"
            for key, expected in DEFAULTS.items()
            if hasattr(args, key) and not _semantic_equal(getattr(args, key), expected)
        ]
        if mismatches:
            parser.error("required production gate rejects overrides: " + "; ".join(mismatches))
    return args


def _device(value: str | None) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _make_dynamics(args: argparse.Namespace) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=int(args.grid_size),
        alpha_eff=float(args.alpha_eff),
        edge_alpha_mode=str(args.edge_alpha_mode),
        num_steps=int(args.sample_steps),
        mass_floor=float(args.mass_floor),
        limiter_fraction=float(args.limiter_fraction),
        source_lowfreq_size=4,
        ot_lowres_size=4,
    )


def _runtime_record(device: torch.device, backend: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "backend": dict(backend),
    }


def _source_record() -> tuple[str, list[str]]:
    here = Path(__file__).resolve()
    names = (
        here.name,
        "d0_score_density_ratio.py",
        "d0_score_density_ratio_gate.py",
        "d0_score_density_ratio_provenance.py",
        "d0_score_boundary_controls.py",
        "d0_score_boundary_control_gate.py",
        "d0_score_optimizer_scale.py",
        "d0_score_control_stability.py",
        "d0_score_control_stability_gate.py",
        "d0_score_control_stability_provenance.py",
        "d0_score_control_scale_repair_gate.py",
        "d0_score_control_scale_repair_provenance.py",
        "d0_dirichlet_score.py",
        "d0_one_image_gate.py",
        "eulerian_flux_mnist.py",
    )
    paths = [here.with_name(name) for name in names]
    existing = [path for path in paths if path.is_file()]
    return source_fingerprint(existing), [str(path) for path in existing]


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        path = args.resume_run_dir.resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path, True
    root = args.runs_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(
        char if char.isalnum() or char in "-_" else "-" for char in str(args.run_name)
    ).strip("-")
    path = root / f"{timestamp}_{safe or 'run'}"
    suffix = 1
    while path.exists():
        path = root / f"{timestamp}_{safe or 'run'}-{suffix}"
        suffix += 1
    path.mkdir(parents=False)
    return path, False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    current = _json_load(path) if path.is_file() else {}
    current.update(updates)
    current.setdefault("schema", RUN_SCHEMA)
    current.setdefault("schema_version", RUN_SCHEMA_VERSION)
    current["physical_training_performed"] = 0
    current["sampling_performed"] = 0
    current["updated_at"] = _now()
    atomic_write_json(path, current)
    return current


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path), "size": int(path.stat().st_size)
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "records": records,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _verify_terminal_registry(run_dir: Path) -> dict[str, Any]:
    registry_path, status_path = run_dir / "artifact_registry.json", run_dir / "run_status.json"
    if not registry_path.is_file() or not status_path.is_file():
        raise ArtifactCompatibilityError("terminal run is missing status or artifact registry")
    status, registry = _json_load(status_path), _json_load(registry_path)
    if (
        status.get("artifact_registry_sha256") != file_fingerprint(registry_path)
        or int(status.get("artifact_registry_size", -1)) != registry_path.stat().st_size
        or registry.get("schema") != RUN_SCHEMA + "-artifact-registry"
    ):
        raise ArtifactCompatibilityError("terminal artifact registry binding mismatch")
    excluded = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    records = dict(registry.get("records", {}))
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    if set(records) != actual:
        raise ArtifactCompatibilityError("artifact registry file set mismatch")
    for relative, raw in records.items():
        path, record = run_dir / relative, dict(raw)
        if (
            not path.is_file()
            or record.get("sha256") != file_fingerprint(path)
            or int(record.get("size", -1)) != path.stat().st_size
        ):
            raise ArtifactCompatibilityError(f"artifact registry hash mismatch: {relative}")
    return registry


def _scientific_config(
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    thresholds: DensityRatioThresholds,
) -> dict[str, Any]:
    value = {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "model_schema": BOUNDARY_SMOOTH_MODEL_VERSION,
        "model_schema_version": 1,
        "objective_version": DENSITY_RATIO_OBJECTIVE_VERSION,
        "stream_version": DENSITY_RATIO_STREAM_VERSION,
        "panel_version": DENSITY_RATIO_PANEL_VERSION,
        "kernel": {key: getattr(args, key) for key in EXPECTED_KERNEL},
        "root_seed": int(args.root_seed),
        "stream": {
            "batch_size": int(args.batch_size),
            "examples_per_class": 32,
            "class_bin_counts": [4, 4, 4, 4, 16],
            "batch_bin_counts": [8, 8, 8, 8, 32],
            "class_prior": 0.5,
            "stateless_seed_tuple": [
                "root_seed", "phase", "task", "step", "namespace"
            ],
        },
        "preflight": {
            "paths": int(args.preflight_paths),
            "confidence": float(args.preflight_confidence),
            "bootstrap_reps": int(args.bootstrap_reps),
        },
        "pilot": {
            "learning_rates": list(args.pilot_learning_rates),
            "steps": int(args.pilot_steps),
            "selection_panels": ["a", "b"],
            "selection_paths_per_panel": int(args.pilot_selection_paths),
            "validation_steps": list(args.pilot_validation_steps),
            "audit_paths": 0,
        },
        "confirmation": {
            "steps": int(args.confirm_steps),
            "model_seeds": list(args.confirm_model_seeds),
            "selection_panels": ["a", "b"],
            "selection_paths_per_panel": int(args.confirm_selection_paths),
            "audit_panels": ["c", "d"],
            "audit_paths_per_panel": int(args.confirm_audit_paths),
            "dense_validation_steps": list(args.confirm_dense_validation_steps),
            "validation_every": int(args.confirm_validation_every),
        },
        "optimization": {
            "optimizer": "AdamW",
            "weight_decay": float(args.weight_decay),
            "ema_decay": float(args.ema_decay),
            "grad_clip": float(args.grad_clip),
            "clip_warmup_steps": int(args.clip_warmup_steps),
            "calibration_state_count": int(args.calibration_state_count),
            "initial_grad_target": float(args.initial_grad_target),
            "adaptive_loss_scaling": 0,
        },
        "bootstrap": {
            "reps": int(args.bootstrap_reps),
            "selection_audit_confidence": float(args.bootstrap_confidence),
        },
        "thresholds": thresholds.to_dict(),
        "parent_scientific_fingerprint": parent.get("scientific_fingerprint"),
        "parent_artifact_registry_sha256": parent.get("artifact_registry_sha256"),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _panel_binding(
    *,
    phase: str,
    role: str,
    task: str,
    path_count: int,
    start_step: int,
    stream_plan: DensityRatioStreamPlan,
    scientific_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-panel-binding",
        "schema_version": 1,
        "phase": str(phase),
        "role": str(role),
        "task": str(task),
        "path_count": int(path_count),
        "start_step": int(start_step),
        "stream_plan_fingerprint": stream_plan.fingerprint,
        "scientific_fingerprint": str(scientific_fingerprint),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _prepare_panel(
    run_dir: Path,
    *,
    phase: str,
    role: str,
    task: str,
    path_count: int,
    start_step: int,
    stream_plan: DensityRatioStreamPlan,
    scientific_fingerprint: str,
) -> DensityRatioPanel:
    path = run_dir / "panels" / phase / f"{task}-{role}.pt"
    binding = _panel_binding(
        phase=phase,
        role=role,
        task=task,
        path_count=path_count,
        start_step=start_step,
        stream_plan=stream_plan,
        scientific_fingerprint=scientific_fingerprint,
    )
    sidecar_path = path.with_suffix(".json")
    if path.is_file() or sidecar_path.is_file():
        if not path.is_file() or not sidecar_path.is_file():
            raise ArtifactCompatibilityError("density-ratio panel bundle is incomplete")
        sidecar = _json_load(sidecar_path)
        if (
            dict(sidecar.get("binding", {})) != binding
            or sidecar.get("sha256") != file_fingerprint(path)
            or int(sidecar.get("size", -1)) != path.stat().st_size
        ):
            raise ArtifactCompatibilityError("density-ratio panel binding mismatch")
        try:
            panel = load_density_ratio_panel(
                path,
                binding=binding,
                device="cpu",
                expected_plan_fingerprint=stream_plan.fingerprint,
                expected_role=role,
                expected_task=task,
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactCompatibilityError(
                f"density-ratio panel verification failed: {exc}"
            ) from exc
        if panel_identity(panel) != dict(sidecar.get("identity", {})):
            raise ArtifactCompatibilityError("density-ratio panel identity mismatch")
        return panel

    panel = build_density_ratio_panel(
        stream_plan,
        phase=phase,
        role=role,
        task=task,
        path_count=int(path_count),
        start_step=int(start_step),
        device="cpu",
        dtype=torch.float32,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_density_ratio_panel(path, panel, binding=binding)
    atomic_write_json(
        sidecar_path,
        {
            "schema": RUN_SCHEMA + "-panel-sidecar",
            "schema_version": 1,
            "binding": binding,
            "identity": panel_identity(panel),
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return panel


def _prepare_panel_set(
    run_dir: Path,
    *,
    phase: str,
    task: str,
    roles: Sequence[str],
    path_count: int,
    stream_plan: DensityRatioStreamPlan,
    scientific_fingerprint: str,
    start_offset: int,
) -> dict[str, DensityRatioPanel]:
    panels = {
        role: _prepare_panel(
            run_dir,
            phase=phase,
            role=role,
            task=task,
            path_count=int(path_count),
            start_step=int(start_offset + index * 100_000),
            stream_plan=stream_plan,
            scientific_fingerprint=scientific_fingerprint,
        )
        for index, role in enumerate(roles)
    }
    disjointness = panel_disjointness_record(list(panels.values()))
    if not bool(int(disjointness.get("passed", 0))):
        raise ArtifactCompatibilityError("density-ratio panels are not disjoint")
    return panels


def _set_seed(seed: int) -> np.random.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return np.random.default_rng(int(seed))


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _derived_seed(root: int, *parts: Any) -> int:
    return int.from_bytes(
        __import__("hashlib").sha256(
            json.dumps([int(root), *parts], separators=(",", ":"), sort_keys=True).encode()
        ).digest()[:8],
        "little",
    ) & ((1 << 63) - 1)


def _boundary_certificate(model: nn.Module) -> dict[str, Any]:
    finite = all(bool(torch.isfinite(value.detach()).all()) for value in model.parameters())
    version = getattr(model, "model_version", None)
    features = tuple(getattr(model, "state_feature_names", ()))
    passed = bool(
        isinstance(model, D0BoundarySmoothPotentialUNet)
        and version == BOUNDARY_SMOOTH_MODEL_VERSION
        and features == ("relative_density", "log1p_relative_density")
        and finite
    )
    return {
        "model_version": version,
        "expected_model_version": BOUNDARY_SMOOTH_MODEL_VERSION,
        "state_feature_names": list(features),
        "raw_log_density_used": 0,
        "finite_parameters": int(finite),
        "structural_facet_certificate": "smooth-closed-simplex-inputs-plus-parent-preflight-v1",
        "passed": int(passed),
    }


def _calibrate_loss_scale(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = run_dir / "density_ratio_loss_scale_calibration.json"
    calibration_paths = int(args.calibration_state_count) // 64
    panel = _prepare_panel(
        run_dir,
        phase="calibration",
        role="training-calibration",
        task="bounded_teacher",
        path_count=calibration_paths,
        start_step=7_000_000,
        stream_plan=stream_plan,
        scientific_fingerprint=str(manifest["scientific_fingerprint"]),
    )
    binding = {
        "schema": RUN_SCHEMA + "-loss-scale-binding",
        "schema_version": 1,
        "scientific_fingerprint": manifest["scientific_fingerprint"],
        "runtime_fingerprint": manifest["runtime_fingerprint"],
        "source_fingerprint": manifest["source_fingerprint"],
        "stream_plan_fingerprint": stream_plan.fingerprint,
        "panel_identity": panel_identity(panel),
        "model_seed": _derived_seed(int(args.root_seed), "calibration-model"),
        "target_initial_gradient_norm": float(args.initial_grad_target),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    if path.is_file():
        record = _json_load(path)
        if (
            dict(record.get("binding", {})) != binding
            or not math.isfinite(float(record.get("loss_scale", math.nan)))
            or not 0.0 < float(record["loss_scale"]) <= 1.0
        ):
            raise ArtifactCompatibilityError("density-ratio calibration mismatch")
        return record
    _set_seed(int(binding["model_seed"]))
    model = D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    calibration = calibrate_density_ratio_loss_scale(
        model,
        panel,
        device=device,
        batch_size=int(args.validation_batch_size),
        target_initial_gradient_norm=float(args.initial_grad_target),
        binding=binding,
    )
    record = calibration.to_record()
    record["binding"] = binding
    record["shared_by_teacher_and_null"] = 1
    record["physical_training_performed"] = 0
    record["sampling_performed"] = 0
    atomic_write_json(path, record)
    return record


def _classification_panel_record(
    model: nn.Module,
    panel: DensityRatioPanel,
    *,
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    device: torch.device,
    bootstrap_seed: int,
    include_analytic_teacher: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = evaluate_classification_panel(
        model,
        panel,
        device=device,
        batch_size=int(args.validation_batch_size),
        return_logits=True,
    )
    logits = torch.as_tensor(raw.pop("raw_logits")).reshape(-1).float()
    targets = panel.class_targets.detach().cpu().reshape(-1).float()
    losses = classification_loss(logits, targets, reduction="none").detach().double().numpy()
    improvements = math.log(2.0) - losses
    strata = np.asarray(panel.strata, dtype=np.int64)
    path_ids = np.asarray(panel.path_ids, dtype=np.int64)
    probabilities = torch.sigmoid(logits).double().numpy()
    targets_np = targets.double().numpy()

    def auc(mask: np.ndarray) -> float:
        scores = np.asarray(logits.double().numpy()[mask], dtype=np.float64)
        truth = np.asarray(targets_np[mask], dtype=np.int64)
        positive_count = int(truth.sum())
        negative_count = int(truth.size - positive_count)
        if not positive_count or not negative_count:
            return float("nan")
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        ranks = np.empty(scores.size, dtype=np.float64)
        start = 0
        while start < scores.size:
            stop = start + 1
            while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
            start = stop
        return float(
            (ranks[truth == 1].sum() - positive_count * (positive_count + 1) / 2)
            / (positive_count * negative_count)
        )

    path_rows: list[dict[str, Any]] = []

    def scope(mask: np.ndarray, name: str) -> dict[str, Any]:
        interval = bootstrap_path_mean_interval(
            improvements[mask],
            path_ids[mask],
            reps=int(args.bootstrap_reps),
            confidence=float(args.bootstrap_confidence),
            seed=_derived_seed(int(bootstrap_seed), panel.role, name),
        )
        for path_id, value in zip(
            interval.get("path_ids", []), interval.get("path_values", [])
        ):
            path_rows.append(
                {
                    "panel": panel.role,
                    "scope": name,
                    "path_id": int(path_id),
                    "classification_improvement_vs_zero": float(value),
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
        risk = float(losses[mask].mean())
        return {
            "state_count": int(mask.sum()),
            "bce": risk,
            "model_bce": risk,
            "classification_risk": risk,
            "zero_logit_bce": math.log(2.0),
            "improvement": float(improvements[mask].mean()),
            "objective_improvement": float(improvements[mask].mean()),
            "accuracy": float(
                np.mean((np.asarray(logits.numpy())[mask] >= 0.0) == (targets_np[mask] == 1.0))
            ),
            "auc": auc(mask),
            "brier": float(np.mean((probabilities[mask] - targets_np[mask]) ** 2)),
            "mean_probability": float(probabilities[mask].mean()),
            "mean_logit": float(np.asarray(logits.numpy())[mask].mean()),
            "lower_bound": interval.get("lower_bound"),
            "upper_bound": interval.get("upper_bound"),
            "confidence": float(args.bootstrap_confidence),
            "bootstrap": interval,
        }

    reliability_bins: list[dict[str, Any]] = []
    reliability_index = np.minimum((probabilities * 10.0).astype(np.int64), 9)
    for index in range(10):
        mask = reliability_index == index
        reliability_bins.append(
            {
                "bin": index,
                "lower": index / 10.0,
                "upper": (index + 1) / 10.0,
                "count": int(mask.sum()),
                "mean_probability": None
                if not bool(mask.any())
                else float(probabilities[mask].mean()),
                "positive_fraction": None
                if not bool(mask.any())
                else float(targets_np[mask].mean()),
            }
        )
    reference_logits = logits[targets == 0.0].double()
    log_reference_mean_exp = float(
        (
            torch.logsumexp(reference_logits, dim=0)
            - math.log(max(1, int(reference_logits.numel())))
        )
    )
    logit_quantile_probabilities = torch.tensor(
        [0.0, 0.1, 0.5, 0.9, 0.99, 1.0], dtype=torch.float64
    )
    logit_quantiles = {
        name: float(value)
        for name, value in zip(
            ("q00", "q10", "q50", "q90", "q99", "q100"),
            torch.quantile(
                logits.detach().abs().double(), logit_quantile_probabilities
            ).tolist(),
        )
    }
    record: dict[str, Any] = {
        "evaluation_status": "evaluated",
        "finite": int(np.isfinite(losses).all()),
        "role": panel.role,
        "task": panel.task,
        "panel_fingerprint": panel.fingerprint,
        "path_count": int(panel.path_count),
        "anchors_per_path": 32,
        "confidence": float(args.bootstrap_confidence),
        "overall": scope(np.ones(losses.size, dtype=bool), "overall"),
        "data_end": scope(strata == 4, "data_end"),
        "classification_metrics": raw,
        "classification_advisory": {
            "accuracy": float(np.mean((np.asarray(logits.numpy()) >= 0.0) == (targets_np == 1.0))),
            "auc": auc(np.ones(losses.size, dtype=bool)),
            "brier": float(np.mean((probabilities - targets_np) ** 2)),
            "reliability_bins": reliability_bins,
            "log_reference_mean_exp_logit": log_reference_mean_exp,
            "reference_normalization_error": log_reference_mean_exp,
            "absolute_logit_quantiles": logit_quantiles,
        },
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    if include_analytic_teacher:
        analytic = analytic_teacher_metrics(
            model,
            panel,
            dynamics,
            device=device,
            batch_size=int(args.validation_batch_size),
            evaluate_class_target=1,
            epsilon=0.5,
        )
        analytic_reference = analytic_teacher_metrics(
            model,
            panel,
            dynamics,
            device=device,
            batch_size=int(args.validation_batch_size),
            evaluate_class_target=0,
            epsilon=0.5,
        )
        record.update(
            {
                "audit_overall_score_gain": analytic["audit_overall_score_gain"],
                "overall_score_gain": analytic["audit_overall_score_gain"],
                "audit_data_end_score_gain": analytic["audit_data_end_score_gain"],
                "data_end_score_gain": analytic["audit_data_end_score_gain"],
                "overall_flux_cosine": analytic["overall_flux_cosine"],
                "time_bin_flux_cosines": analytic["time_bin_flux_cosines"],
                "overall_relative_flux_l2": analytic["overall_relative_flux_l2"],
                "time_bin_relative_flux_l2": analytic["time_bin_relative_flux_l2"],
                "analytic": analytic,
                "analytic_reference": analytic_reference,
            }
        )
        record["finite"] = int(
            bool(record["finite"])
            and bool(analytic["finite"])
            and bool(analytic_reference["finite"])
        )
    return record, path_rows


def _task_fingerprints(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    task: str,
    model_seed: int,
    learning_rate: float,
    loss_scale: float,
    stream_plan: DensityRatioStreamPlan,
    selection_panels: Mapping[str, DensityRatioPanel],
    audit_panels: Mapping[str, DensityRatioPanel] | None,
    profile_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-task-fingerprints",
        "schema_version": 1,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "scientific_fingerprint": manifest["scientific_fingerprint"],
        "runtime_fingerprint": manifest["runtime_fingerprint"],
        "source_fingerprint": manifest["source_fingerprint"],
        "phase": str(phase),
        "task": str(task),
        "model_seed": int(model_seed),
        "learning_rate": float(learning_rate),
        "loss_scale": float(loss_scale),
        "stream_plan_fingerprint": stream_plan.fingerprint,
        "selection_panel_identities": {
            name: panel_identity(panel) for name, panel in selection_panels.items()
        },
        "audit_panel_identities": None
        if audit_panels is None
        else {name: panel_identity(panel) for name, panel in audit_panels.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    if profile_binding is not None:
        normalized = json.loads(json.dumps(dict(profile_binding), sort_keys=True))
        value["profile_binding"] = normalized
        value["profile_binding_fingerprint"] = config_fingerprint(normalized)
    return value


def _task_args(
    args: argparse.Namespace, *, phase: str, learning_rate: float
) -> argparse.Namespace:
    values = vars(args).copy()
    values["learning_rate"] = float(learning_rate)
    values["train_steps"] = int(
        args.pilot_steps if phase == "pilot" else args.confirm_steps
    )
    values["validation_steps"] = (
        tuple(args.pilot_validation_steps)
        if phase == "pilot"
        else tuple(
            sorted(
                set(args.confirm_dense_validation_steps)
                | set(
                    range(
                        int(args.confirm_validation_every),
                        int(args.confirm_steps) + 1,
                        int(args.confirm_validation_every),
                    )
                )
                | {int(args.confirm_steps)}
            )
        )
    )
    return argparse.Namespace(**values)


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
    stream_plan: DensityRatioStreamPlan,
    task: str,
    fingerprints: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    atomic_torch_save(
        path,
        {
            "schema": RUN_SCHEMA + "-checkpoint",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_schema": BOUNDARY_SMOOTH_MODEL_VERSION,
            "model_schema_version": 1,
            "task": str(task),
            "step": int(step),
            "stream_cursor": int(step),
            "stream_derivation_version": DENSITY_RATIO_STREAM_VERSION,
            "stream_plan": stream_plan_record(stream_plan),
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "ema_state_dict": {
                key: value.detach().clone() for key, value in ema_state.items()
            },
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "history": [dict(value) for value in history],
            "validation_records": copy.deepcopy(list(validations)),
            "checkpoint_selection": copy.deepcopy(dict(selection)),
            "rng_state": capture_rng_state(rng),
            "fingerprints": dict(fingerprints),
            "scaler_state_dict": None,
            "amp": False,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    task: str,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        value = torch.load(path, map_location=device)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != RUN_SCHEMA + "-checkpoint"
        or int(value.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION
        or value.get("task") != task
        or dict(value.get("fingerprints", {})) != dict(fingerprints)
        or value.get("stream_derivation_version") != DENSITY_RATIO_STREAM_VERSION
    ):
        raise ArtifactCompatibilityError(
            "legacy, foreign, or mismatched density-ratio checkpoint"
        )
    required = {
        "step", "stream_cursor", "model_state_dict", "ema_state_dict",
        "optimizer_state_dict", "history", "validation_records",
        "checkpoint_selection", "rng_state", "stream_plan", "fingerprints",
    }
    if not required.issubset(value):
        raise ArtifactCompatibilityError("density-ratio checkpoint is incomplete")
    return dict(value)


def _history_diagnostics(
    history: Sequence[Mapping[str, Any]], *, args: argparse.Namespace
) -> dict[str, Any]:
    return summarize_scaled_gradient_history(
        history,
        warmup_steps=int(args.clip_warmup_steps),
        grad_clip=float(args.grad_clip),
    )


def _clip_fraction(
    history: Sequence[Mapping[str, Any]], *, start: int, stop: int
) -> float:
    rows = [
        row for row in history if int(start) <= int(row.get("step", -1)) <= int(stop)
    ]
    if not rows:
        return 0.0
    return float(np.mean([bool(int(row.get("clipped", 0))) for row in rows]))


def _load_completed_task(
    task_dir: Path,
    *,
    device: torch.device,
    task: str,
    fingerprints: Mapping[str, Any],
) -> dict[str, Any] | None:
    result_path, status_path = task_dir / "task_result.json", task_dir / "task_status.json"
    if not result_path.is_file() and not status_path.is_file():
        return None
    if not result_path.is_file() or not status_path.is_file():
        return None
    result, status = _json_load(result_path), _json_load(status_path)
    if status.get("status") != "complete":
        return None
    checkpoints = task_dir / "checkpoints"
    latest_path, best_pointer_path = checkpoints / "latest.json", checkpoints / "best.json"
    if not latest_path.is_file() or not best_pointer_path.is_file():
        raise ArtifactCompatibilityError("completed task is missing checkpoint pointers")
    latest, best = _json_load(latest_path), _json_load(best_pointer_path)
    latest_file = checkpoints / str(latest.get("filename", ""))
    selected_file = checkpoints / str(best.get("selected_filename", ""))
    nominee_file = checkpoints / str(best.get("nominee_filename", ""))
    best_copy = checkpoints / str(best.get("best_ema_filename", ""))
    nominee_copy = checkpoints / str(best.get("nominee_ema_filename", ""))
    summary = dict(result.get("training_summary", {}))
    if (
        dict(status.get("fingerprints", {})) != dict(fingerprints)
        or dict(result.get("fingerprints", {})) != dict(fingerprints)
        or dict(best.get("fingerprints", {})) != dict(fingerprints)
        or dict(latest.get("fingerprints", {})) != dict(fingerprints)
        or status.get("task_result_sha256") != file_fingerprint(result_path)
        or not latest_file.is_file()
        or not selected_file.is_file()
        or not nominee_file.is_file()
        or not best_copy.is_file()
        or not nominee_copy.is_file()
        or latest.get("sha256") != file_fingerprint(latest_file)
        or best.get("selected_sha256") != file_fingerprint(selected_file)
        or best.get("nominee_sha256") != file_fingerprint(nominee_file)
        or best.get("best_ema_sha256") != file_fingerprint(best_copy)
        or best.get("nominee_ema_sha256") != file_fingerprint(nominee_copy)
        or summary.get("selected_checkpoint_sha256") != file_fingerprint(best_copy)
        or summary.get("nominee_checkpoint_sha256") != file_fingerprint(nominee_copy)
        or int(status.get("training_step", -1)) != int(summary.get("training_step", -2))
        or int(status.get("selected_step", -1)) != int(summary.get("selected_step", -2))
        or int(status.get("nominee_step", -1)) != int(summary.get("nominee_step", -2))
    ):
        raise ArtifactCompatibilityError("completed density-ratio checkpoint chain mismatch")
    _load_checkpoint(latest_file, device=device, task=task, fingerprints=fingerprints)
    _load_checkpoint(selected_file, device=device, task=task, fingerprints=fingerprints)
    _load_checkpoint(nominee_file, device=device, task=task, fingerprints=fingerprints)
    return result


def _failed_task_result(
    task_dir: Path,
    *,
    task: str,
    model_seed: int,
    fingerprints: Mapping[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    failure = {"type": type(exc).__name__, "message": str(exc)}
    metrics = {
        "evaluation_status": "evaluated",
        "complete": 0,
        "finite": 0,
        "boundary_admissible": 0,
        "model_seed": int(model_seed),
        "selected_step": 0,
        "nominee_step": 0,
        "post_warmup_clip_fraction": 1.0,
        "checkpoints": [],
        "audit_panels": {},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result = {
        "schema": RUN_SCHEMA + "-task-result",
        "schema_version": 1,
        "task": task,
        "model_seed": int(model_seed),
        "metrics": metrics,
        "gate": _not_evaluated_gate("density_ratio_task", str(exc)),
        "failure": failure,
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(task_dir / "task_failure.json", result)
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "schema": RUN_SCHEMA + "-task-status",
            "schema_version": 1,
            "status": "failed",
            "task": task,
            "model_seed": int(model_seed),
            "failure": failure,
            "fingerprints": dict(fingerprints),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return result


def run_density_ratio_task(
    *,
    task_dir: Path,
    task: str,
    selection_panels: Mapping[str, DensityRatioPanel],
    audit_panels: Mapping[str, DensityRatioPanel] | None,
    dynamics: DirectFluxMNISTConfig,
    args: argparse.Namespace,
    device: torch.device,
    model_seed: int,
    learning_rate: float,
    loss_scale: float,
    stream_plan: DensityRatioStreamPlan,
    fingerprints: Mapping[str, Any],
    phase: str,
    thresholds: DensityRatioThresholds,
    show_progress: bool,
    interrupt_after_checkpoint_step: int | None = None,
) -> dict[str, Any]:
    """Train one exactly resumable teacher or null classifier task."""

    if task not in {"bounded_teacher", "dirichlet_null"}:
        raise ValueError("unsupported density-ratio task")
    if set(selection_panels) != {"a", "b"}:
        raise ValueError("selection panels must be exactly a and b")
    if audit_panels is not None and set(audit_panels) != {"c", "d"}:
        raise ValueError("audit panels must be exactly c and d")
    task_dir.mkdir(parents=True, exist_ok=True)
    completed_result = _load_completed_task(
        task_dir, device=device, task=task, fingerprints=fingerprints
    )
    if completed_result is not None:
        return completed_result

    checkpoints = task_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints / "latest.json"
    best_pointer_path = checkpoints / "best.json"
    rng = _set_seed(int(model_seed))
    model = D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(args.weight_decay),
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
            Path(filename).name != filename
            or not checkpoint_path.is_file()
            or latest.get("sha256") != file_fingerprint(checkpoint_path)
            or dict(latest.get("fingerprints", {})) != dict(fingerprints)
        ):
            raise ArtifactCompatibilityError("density-ratio latest pointer is invalid")
        payload = _load_checkpoint(
            checkpoint_path, device=device, task=task, fingerprints=fingerprints
        )
        # Final selection is committed to a distinct checkpoint before the
        # latest pointer is swapped.  If a process died in that narrow window,
        # recover the complete orphan instead of re-evaluating sealed panel B.
        orphan_finalized = checkpoints / f"finalized-step-{int(payload['step']):08d}.pt"
        if checkpoint_path != orphan_finalized and orphan_finalized.is_file():
            recovered = _load_checkpoint(
                orphan_finalized,
                device=device,
                task=task,
                fingerprints=fingerprints,
            )
            recovered_selection = dict(recovered.get("checkpoint_selection", {}))
            recovered_validations = [
                dict(value) for value in recovered.get("validation_records", [])
            ]
            b_records = sum(
                "b" in dict(value.get("panels", {}))
                for value in recovered_validations
            )
            if (
                int(recovered.get("step", -1)) == int(payload["step"])
                and recovered_selection.get("gate")
                == "density_ratio_checkpoint_selection"
                and b_records == 1
            ):
                payload = recovered
                checkpoint_path = orphan_finalized
                atomic_write_json(
                    latest_path,
                    {
                        "schema": RUN_SCHEMA + "-latest",
                        "schema_version": 1,
                        "filename": checkpoint_path.name,
                        "sha256": file_fingerprint(checkpoint_path),
                        "step": int(payload["step"]),
                        "stream_cursor": int(payload["stream_cursor"]),
                        "fingerprints": dict(fingerprints),
                        "recovered_orphan_finalization": 1,
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    },
                )
        if dict(payload["stream_plan"]) != stream_plan_record(stream_plan):
            raise ArtifactCompatibilityError("density-ratio checkpoint stream mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(optimizer, device)
        ema_state = {
            key: value.detach().clone().to(device)
            for key, value in dict(payload["ema_state_dict"]).items()
        }
        history = [dict(value) for value in payload["history"]]
        validations = [dict(value) for value in payload["validation_records"]]
        selection = dict(payload["checkpoint_selection"])
        completed = int(payload["step"])
        if int(payload["stream_cursor"]) != completed:
            raise ArtifactCompatibilityError("checkpoint cursor differs from step")
        restore_rng_state(payload["rng_state"], rng)

    def validate(step: int) -> dict[str, Any]:
        with temporary_ema_weights(model, ema_state):
            record, _ = _classification_panel_record(
                model,
                selection_panels["a"],
                dynamics=dynamics,
                args=args,
                device=device,
                bootstrap_seed=_derived_seed(
                    int(args.root_seed), phase, task, "selection", "a", step
                ),
                include_analytic_teacher=False,
            )
        return {
            "step": int(step),
            "finite": int(bool(int(record["finite"]))),
            "ema": 1,
            # Panel B is deliberately absent until panel A has nominated a
            # single checkpoint after training.  This prevents confirmatory
            # state reuse for checkpoint scanning.
            "panels": {"a": record},
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }

    def publish(step: int) -> None:
        nonlocal selection
        nomination = nominate_checkpoint_on_a(validations, thresholds)
        selection = {
            "gate": "density_ratio_checkpoint_selection_pending_panel_b",
            "evaluation_status": "not_evaluated",
            "passed": 0,
            "selected_step": 0,
            "nominee_step": nomination.get("nominee_step"),
            "nomination": nomination,
            "confirmation": _not_evaluated_gate(
                "density_ratio_panel_b_confirmation",
                "panel B is sealed until the final panel-A nominee is frozen",
            ),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        checkpoint_path = checkpoints / f"step-{int(step):08d}.pt"
        _save_checkpoint(
            checkpoint_path,
            model=model,
            ema_state=ema_state,
            optimizer=optimizer,
            step=int(step),
            history=history,
            validations=validations,
            selection=selection,
            stream_plan=stream_plan,
            task=task,
            fingerprints=fingerprints,
            rng=rng,
        )
        atomic_write_json(
            latest_path,
            {
                "schema": RUN_SCHEMA + "-latest",
                "schema_version": 1,
                "filename": checkpoint_path.name,
                "sha256": file_fingerprint(checkpoint_path),
                "step": int(step),
                "stream_cursor": int(step),
                "fingerprints": dict(fingerprints),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )

    if not validations:
        validations.append(validate(0))
        publish(0)

    validation_steps = set(int(value) for value in args.validation_steps)
    total_steps = int(args.train_steps)
    started = time.perf_counter()
    for step in range(completed + 1, total_steps + 1):
        batch = generate_density_ratio_batch(
            stream_plan,
            phase=phase,
            task=task,
            step=int(step),
            device=device,
            dtype=torch.float32,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.tau, batch.states, batch.labels).reshape(-1)
        unscaled = classification_loss(logits, batch.class_targets)
        if not bool(torch.isfinite(unscaled.detach())):
            raise FloatingPointError(
                f"nonfinite density-ratio loss for {task} at step {step}"
            )
        gradient = scaled_backward_and_clip(
            unscaled,
            model.parameters(),
            loss_scale=float(loss_scale),
            grad_clip=float(args.grad_clip),
        )
        optimizer.step()
        update_ema_state(ema_state, model, float(args.ema_decay))
        history.append(
            {
                "step": int(step),
                "loss": float(gradient.scaled_loss),
                "unscaled_loss": float(gradient.unscaled_loss),
                "scaled_loss": float(gradient.scaled_loss),
                "loss_scale": float(gradient.loss_scale),
                "raw_gradient_norm": float(gradient.raw_gradient_norm),
                "scaled_preclip_gradient_norm": float(
                    gradient.scaled_preclip_gradient_norm
                ),
                "grad_norm": float(gradient.scaled_preclip_gradient_norm),
                "clipped": int(gradient.clipped),
                "stream_batch_fingerprint": batch.fingerprint,
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
        if step in validation_steps:
            validations.append(validate(step))
            publish(step)
            if (
                interrupt_after_checkpoint_step is not None
                and int(step) == int(interrupt_after_checkpoint_step)
            ):
                raise RuntimeError(
                    f"injected interruption after density-ratio checkpoint {step}"
                )
        if show_progress and (step % 50 == 0 or step == total_steps):
            elapsed = time.perf_counter() - started
            done = max(1, step - completed)
            eta = elapsed / done * max(0, total_steps - step)
            print(
                f"{phase}/{task}: step {step}/{total_steps} "
                f"loss={history[-1]['scaled_loss']:.6g} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    final_step = int(history[-1]["step"]) if history else int(completed)
    if not latest_path.is_file() or int(_json_load(latest_path).get("step", -1)) != final_step:
        if not any(int(value.get("step", -1)) == final_step for value in validations):
            validations.append(validate(final_step))
        publish(final_step)

    # Freeze the single panel-A nominee first.  Panel B is then evaluated once
    # and only once on that nominee; it is never scanned across checkpoints.
    nomination = nominate_checkpoint_on_a(validations, thresholds)
    nominee_raw = nomination.get("nominee_step")
    nominee_step = int(nominee_raw) if nominee_raw is not None else 0
    nominee_path = checkpoints / f"step-{nominee_step:08d}.pt"
    if not nominee_path.is_file():
        raise ArtifactCompatibilityError("panel-A nominee checkpoint is missing")
    nominee_rows = [
        value for value in validations if int(value.get("step", -1)) == nominee_step
    ]
    if len(nominee_rows) != 1:
        raise ArtifactCompatibilityError("panel-A nominee validation is ambiguous")
    nominee_validation = nominee_rows[0]
    if "b" not in dict(nominee_validation.get("panels", {})):
        nominee_payload = _load_checkpoint(
            nominee_path, device=device, task=task, fingerprints=fingerprints
        )
        model.load_state_dict(nominee_payload["ema_state_dict"], strict=True)
        model.eval()
        panel_b, _ = _classification_panel_record(
            model,
            selection_panels["b"],
            dynamics=dynamics,
            args=args,
            device=device,
            bootstrap_seed=_derived_seed(
                int(args.root_seed), phase, task, "selection", "b", nominee_step
            ),
            include_analytic_teacher=task == "bounded_teacher",
        )
        nominee_validation.setdefault("panels", {})["b"] = panel_b
        nominee_validation["finite"] = int(
            bool(int(nominee_validation.get("finite", 0)))
            and bool(int(panel_b.get("finite", 0)))
        )
    selection = select_density_ratio_checkpoint(validations, thresholds)
    selected_step = int(selection.get("selected_step", 0))
    selected_path = checkpoints / f"step-{selected_step:08d}.pt"
    if not selected_path.is_file():
        raise ArtifactCompatibilityError("selected checkpoint is missing")

    # Commit the one-time B result and final selection to a distinct checkpoint
    # without changing model/optimizer/RNG state.  The original step checkpoint
    # remains valid until latest.json is atomically swapped, making finalization
    # recoverable across every crash boundary.
    final_step_checkpoint = checkpoints / f"step-{final_step:08d}.pt"
    finalized_checkpoint_path = checkpoints / f"finalized-step-{final_step:08d}.pt"
    final_payload = _load_checkpoint(
        final_step_checkpoint, device=device, task=task, fingerprints=fingerprints
    )
    final_payload["validation_records"] = copy.deepcopy(validations)
    final_payload["checkpoint_selection"] = copy.deepcopy(selection)
    atomic_torch_save(finalized_checkpoint_path, final_payload)
    atomic_write_json(
        latest_path,
        {
            "schema": RUN_SCHEMA + "-latest",
            "schema_version": 1,
            "filename": finalized_checkpoint_path.name,
            "sha256": file_fingerprint(finalized_checkpoint_path),
            "step": final_step,
            "stream_cursor": final_step,
            "fingerprints": dict(fingerprints),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )

    best_copy, nominee_copy = checkpoints / "best_ema.pt", checkpoints / "nominee_ema.pt"
    atomic_copy_file(selected_path, best_copy)
    atomic_copy_file(nominee_path, nominee_copy)
    atomic_write_json(
        best_pointer_path,
        {
            "schema": RUN_SCHEMA + "-best",
            "schema_version": 1,
            "selected_step": selected_step,
            "selected_filename": selected_path.name,
            "selected_sha256": file_fingerprint(selected_path),
            "best_ema_filename": best_copy.name,
            "best_ema_sha256": file_fingerprint(best_copy),
            "nominee_step": nominee_step,
            "nominee_filename": nominee_path.name,
            "nominee_sha256": file_fingerprint(nominee_path),
            "nominee_ema_filename": nominee_copy.name,
            "nominee_ema_sha256": file_fingerprint(nominee_copy),
            "fingerprints": dict(fingerprints),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )

    audit_step = selected_step if task == "bounded_teacher" else nominee_step
    audit_path = checkpoints / f"step-{audit_step:08d}.pt"
    audited_payload = _load_checkpoint(
        audit_path, device=device, task=task, fingerprints=fingerprints
    )
    model.load_state_dict(audited_payload["ema_state_dict"], strict=True)
    model.eval()
    diagnostics = _history_diagnostics(history, args=args)
    boundary_certificate = _boundary_certificate(model)
    summary = {
        "complete": 1,
        "finite": int(
            all(math.isfinite(float(value["scaled_loss"])) for value in history)
        ),
        "task": task,
        "model_seed": int(model_seed),
        "selected_step": selected_step,
        "nominee_step": nominee_step,
        "audit_step": audit_step,
        "training_step": final_step,
        "target_training_steps": total_steps,
        "checkpoint_selection": selection,
        "post_warmup_clip_fraction": diagnostics["post_warmup_clip_fraction"],
        "optimization_diagnostics": diagnostics,
        "boundary_admissibility_certificate": boundary_certificate,
        "selected_checkpoint_path": str(best_copy.resolve()),
        "selected_checkpoint_sha256": file_fingerprint(best_copy),
        "nominee_checkpoint_path": str(nominee_copy.resolve()),
        "nominee_checkpoint_sha256": file_fingerprint(nominee_copy),
        "stream_plan": stream_plan_record(stream_plan),
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    metrics: dict[str, Any] = {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": int(summary["finite"]),
        "model_seed": int(model_seed),
        "selected_step": selected_step,
        "nominee_step": nominee_step,
        "audit_step": audit_step,
        "selection": selection,
        "checkpoints": validations,
        "boundary_admissible": int(boundary_certificate["passed"]),
        "post_warmup_clip_fraction": diagnostics["post_warmup_clip_fraction"],
        "audit_panels": {},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    panel_b_record = dict(dict(nominee_validation.get("panels", {})).get("b", {}))
    if task == "bounded_teacher" and isinstance(panel_b_record.get("analytic"), Mapping):
        metrics["selected_analytic_metrics"] = dict(panel_b_record["analytic"])
        for source, target in (
            ("audit_overall_score_gain", "selection_overall_score_gain"),
            ("audit_data_end_score_gain", "selection_data_end_score_gain"),
            ("overall_flux_cosine", "selection_overall_flux_cosine"),
            ("overall_relative_flux_l2", "selection_overall_relative_flux_l2"),
        ):
            metrics[target] = panel_b_record["analytic"].get(source)
        time_cosines = list(panel_b_record["analytic"].get("time_bin_flux_cosines", []))
        time_relatives = list(panel_b_record["analytic"].get("time_bin_relative_flux_l2", []))
        metrics["selection_data_end_flux_cosine"] = time_cosines[-1] if time_cosines else None
        metrics["selection_data_end_relative_flux_l2"] = time_relatives[-1] if time_relatives else None
    if task == "dirichlet_null":
        metrics["comparator"] = "analytic_zero"

    audit_path_rows: list[dict[str, Any]] = []
    if audit_panels is not None:
        for name in ("c", "d"):
            record, rows = _classification_panel_record(
                model,
                audit_panels[name],
                dynamics=dynamics,
                args=args,
                device=device,
                bootstrap_seed=_derived_seed(
                    int(args.root_seed), phase, task, "audit", name, audit_step
                ),
                include_analytic_teacher=task == "bounded_teacher",
            )
            metrics["audit_panels"][name] = record
            audit_path_rows.extend(rows)

    gate = (
        evaluate_teacher_seed(metrics, thresholds)
        if task == "bounded_teacher" and audit_panels is not None
        else evaluate_null_seed(metrics, thresholds)
        if task == "dirichlet_null" and audit_panels is not None
        else {
            "gate": "density_ratio_pilot_task",
            "evaluation_status": "evaluated",
            "passed": int(bool(metrics["complete"]) and bool(metrics["finite"])),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    )

    atomic_write_csv(task_dir / "training_history.csv", history)
    checkpoint_rows: list[dict[str, Any]] = []
    selection_path_rows: list[dict[str, Any]] = []
    selection_time_rows: list[dict[str, Any]] = []
    selection_reliability_rows: list[dict[str, Any]] = []
    for validation in validations:
        for panel_name, panel_record in dict(validation.get("panels", {})).items():
            checkpoint_rows.append(
                {
                    "step": validation["step"],
                    "panel": panel_name,
                    "bce_overall": dict(panel_record["overall"])["bce"],
                    "improvement_overall": dict(panel_record["overall"])["improvement"],
                    "lower_bound_overall": dict(panel_record["overall"])["lower_bound"],
                    "bce_data_end": dict(panel_record["data_end"])["bce"],
                    "improvement_data_end": dict(panel_record["data_end"])["improvement"],
                    "lower_bound_data_end": dict(panel_record["data_end"])["lower_bound"],
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
            for scope_name in ("overall", "data_end"):
                bootstrap = dict(dict(panel_record[scope_name]).get("bootstrap", {}))
                for path_id, value in zip(
                    bootstrap.get("path_ids", []), bootstrap.get("path_values", [])
                ):
                    selection_path_rows.append(
                        {
                            "step": validation["step"],
                            "panel": panel_name,
                            "scope": scope_name,
                            "path_id": path_id,
                            "classification_improvement_vs_zero": value,
                            "physical_training_performed": 0,
                            "sampling_performed": 0,
                        }
                    )
            classification_bins = list(
                dict(panel_record.get("classification_metrics", {})).get(
                    "time_bins", []
                )
            )
            analytic_bins = list(
                dict(panel_record.get("analytic", {})).get("time_bins", [])
            )
            reference_bins = list(
                dict(panel_record.get("analytic_reference", {})).get(
                    "time_bins", []
                )
            )
            for index, classification_bin in enumerate(classification_bins):
                analytic_bin = (
                    dict(analytic_bins[index]) if index < len(analytic_bins) else {}
                )
                reference_bin = (
                    dict(reference_bins[index])
                    if index < len(reference_bins)
                    else {}
                )
                selection_time_rows.append(
                    {
                        "step": validation["step"],
                        "panel": panel_name,
                        "time_bin": index,
                        **{
                            f"classification_{key}": value
                            for key, value in dict(classification_bin).items()
                            if not isinstance(value, (dict, list))
                        },
                        **{
                            f"teacher_{key}": value
                            for key, value in analytic_bin.items()
                            if not isinstance(value, (dict, list))
                        },
                        **{
                            f"reference_{key}": value
                            for key, value in reference_bin.items()
                            if not isinstance(value, (dict, list))
                        },
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
            for reliability in dict(
                panel_record.get("classification_advisory", {})
            ).get("reliability_bins", []):
                selection_reliability_rows.append(
                    {
                        "step": validation["step"],
                        "panel": panel_name,
                        **dict(reliability),
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
    atomic_write_csv(task_dir / "checkpoint_metrics.csv", checkpoint_rows)
    atomic_write_csv(task_dir / "selection_path_risks.csv", selection_path_rows)
    if selection_time_rows:
        atomic_write_csv(
            task_dir / "selection_time_bin_metrics.csv", selection_time_rows
        )
    if selection_reliability_rows:
        atomic_write_csv(
            task_dir / "selection_reliability_bins.csv",
            selection_reliability_rows,
        )
    if audit_path_rows:
        atomic_write_csv(task_dir / "audit_path_risks.csv", audit_path_rows)
    if audit_panels is not None:
        audit_time_rows: list[dict[str, Any]] = []
        audit_reliability_rows: list[dict[str, Any]] = []
        for panel_name, panel_record in metrics["audit_panels"].items():
            classification_bins = list(
                dict(panel_record.get("classification_metrics", {})).get(
                    "time_bins", []
                )
            )
            analytic_bins = list(
                dict(panel_record.get("analytic", {})).get("time_bins", [])
            )
            reference_bins = list(
                dict(panel_record.get("analytic_reference", {})).get(
                    "time_bins", []
                )
            )
            for index, classification_bin in enumerate(classification_bins):
                analytic_bin = (
                    dict(analytic_bins[index]) if index < len(analytic_bins) else {}
                )
                reference_bin = (
                    dict(reference_bins[index])
                    if index < len(reference_bins)
                    else {}
                )
                audit_time_rows.append(
                    {
                        "panel": panel_name,
                        "time_bin": index,
                        **{
                            f"classification_{key}": value
                            for key, value in dict(classification_bin).items()
                            if not isinstance(value, (dict, list))
                        },
                        **{
                            f"teacher_{key}": value
                            for key, value in analytic_bin.items()
                            if not isinstance(value, (dict, list))
                        },
                        **{
                            f"reference_{key}": value
                            for key, value in reference_bin.items()
                            if not isinstance(value, (dict, list))
                        },
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
            for reliability in dict(
                panel_record.get("classification_advisory", {})
            ).get("reliability_bins", []):
                audit_reliability_rows.append(
                    {
                        "panel": panel_name,
                        **dict(reliability),
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
        if audit_time_rows:
            atomic_write_csv(task_dir / "audit_time_bin_metrics.csv", audit_time_rows)
        if audit_reliability_rows:
            atomic_write_csv(
                task_dir / "audit_reliability_bins.csv", audit_reliability_rows
            )
    atomic_write_json(task_dir / "training_summary.json", summary)
    result = {
        "schema": RUN_SCHEMA + "-task-result",
        "schema_version": 1,
        "task": task,
        "model_seed": int(model_seed),
        "metrics": metrics,
        "gate": gate,
        "training_summary": summary,
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result = json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
    result_path = task_dir / "task_result.json"
    atomic_write_json(result_path, result)
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "schema": RUN_SCHEMA + "-task-status",
            "schema_version": 1,
            "status": "complete",
            "task": task,
            "model_seed": int(model_seed),
            "training_step": final_step,
            "selected_step": selected_step,
            "nominee_step": nominee_step,
            "fingerprints": dict(fingerprints),
            "task_result_sha256": file_fingerprint(result_path),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return result


def _oracle_improvement_record(
    losses: np.ndarray,
    panel: DensityRatioPanel,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    improvements = math.log(2.0) - np.asarray(losses, dtype=np.float64)
    strata = np.asarray(panel.strata, dtype=np.int64)
    paths = np.asarray(panel.path_ids, dtype=np.int64)

    def scope(mask: np.ndarray, name: str) -> dict[str, Any]:
        interval = bootstrap_path_mean_interval(
            improvements[mask],
            paths[mask],
            reps=int(args.bootstrap_reps),
            confidence=float(args.preflight_confidence),
            seed=_derived_seed(seed, name),
        )
        return {
            "bce": float(np.asarray(losses)[mask].mean()),
            "improvement": float(improvements[mask].mean()),
            "lower_bound": interval.get("lower_bound"),
            "upper_bound": interval.get("upper_bound"),
            "confidence": float(args.preflight_confidence),
            "bootstrap": interval,
        }

    return {
        "overall": scope(np.ones(losses.size, dtype=bool), "overall"),
        "data_end": scope(strata == 4, "data_end"),
    }


def _run_preflight(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    thresholds: DensityRatioThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    replay_rows: list[dict[str, Any]] = []
    for phase in ("pilot", "confirm"):
        for task in ("bounded_teacher", "dirichlet_null"):
            for step in (1, 17):
                record = density_ratio_replay_record(
                    stream_plan, phase=phase, task=task, step=step
                )
                replay_rows.append(
                    {
                        "record": record,
                        "verification": verify_density_ratio_replay(
                            stream_plan, record
                        ),
                    }
                )
    replay = {
        "schema": RUN_SCHEMA + "-stream-replay-preflight",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": int(
            all(
                bool(int(value["verification"].get("passed", 0)))
                for value in replay_rows
            )
        ),
        "records": replay_rows,
        "stream_plan_fingerprint": stream_plan.fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(run_dir / "density_ratio_stream_replay_preflight.json", replay)

    panels = {
        task: _prepare_panel(
            run_dir,
            phase="preflight",
            role=f"oracle-{task}",
            task=task,
            path_count=int(args.preflight_paths),
            start_step=1_000_000 + index * 1_000_000,
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
        )
        for index, task in enumerate(("bounded_teacher", "dirichlet_null"))
    }
    isolation = panel_disjointness_record(list(panels.values()))
    teacher = panels["bounded_teacher"]
    null = panels["dirichlet_null"]

    teacher_logits = equal_prior_bayes_logit(
        teacher.states.double(), teacher.tau_fraction.double(),
        task="bounded_teacher", epsilon=0.5,
    )
    teacher_losses = classification_loss(
        teacher_logits,
        teacher.class_targets.double(),
        reduction="none",
    ).detach().cpu().numpy()
    teacher_signal = _oracle_improvement_record(
        teacher_losses,
        teacher,
        args=args,
        seed=_derived_seed(int(args.root_seed), "preflight", "oracle-bce"),
    )

    reference_mask = teacher.class_targets.detach().cpu().numpy() == 0
    reference_ratio = bounded_teacher_density_ratio(
        teacher.states[reference_mask].double(),
        teacher.tau_fraction[reference_mask].double(),
        epsilon=0.5,
    ).detach().cpu().numpy()
    normalization = bootstrap_path_mean_interval(
        reference_ratio,
        np.asarray(teacher.path_ids)[reference_mask],
        reps=int(args.bootstrap_reps),
        confidence=float(args.preflight_confidence),
        seed=_derived_seed(int(args.root_seed), "preflight", "normalization"),
    )

    analytic_batch = generate_density_ratio_batch(
        stream_plan,
        phase="preflight-analytic",
        task="bounded_teacher",
        step=1,
        device="cpu",
        dtype=torch.float64,
    )
    analytic_states = analytic_batch.states.detach().clone().requires_grad_(True)
    analytic_fractions = analytic_batch.tau_fraction
    oracle = equal_prior_bayes_logit(
        analytic_states,
        analytic_fractions,
        task="bounded_teacher",
        epsilon=0.5,
    )
    direct = bounded_teacher_log_relative_potential(
        analytic_states, analytic_fractions, epsilon=0.5
    )
    gradient = torch.autograd.grad(oracle.sum(), analytic_states)[0]
    target_gradient = bounded_teacher_cell_score(
        analytic_states.detach(), analytic_fractions, epsilon=0.5
    )
    predicted_edge = edge_difference_channels(gradient, int(args.grid_size))
    predicted_flux = physical_flux_from_edge_score(
        predicted_edge, analytic_states.detach(), dynamics
    )
    target_flux = bounded_teacher_physical_flux(
        analytic_states.detach(), analytic_fractions, dynamics, epsilon=0.5
    )

    null_logits = equal_prior_bayes_logit(
        null.states.double(), null.tau_fraction.double(), task="dirichlet_null"
    )
    null_loss = classification_loss(
        null_logits, null.class_targets.double()
    )
    null_probe_states = null.states[:2].double().detach().clone()
    null_probe_fractions = null.tau_fraction[:2].double().detach().clone()
    null_jacobian = torch.autograd.functional.jacobian(
        lambda states: equal_prior_bayes_logit(
            states,
            null_probe_fractions,
            task="dirichlet_null",
        ),
        null_probe_states,
        vectorize=True,
    )
    null_cell_score = torch.stack(
        [null_jacobian[index, index] for index in range(null_probe_states.shape[0])]
    )
    null_edge_score = edge_difference_channels(
        null_cell_score, int(args.grid_size)
    )
    null_flux = physical_flux_from_edge_score(
        null_edge_score, null_probe_states, dynamics
    )
    teacher_record = analytic_batch.record()
    null_record = generate_density_ratio_batch(
        stream_plan,
        phase="preflight-analytic",
        task="dirichlet_null",
        step=1,
        device="cpu",
        dtype=torch.float64,
    ).record()
    class_balance = all(
        value["record"]["batch"]["class_counts"] == [32, 32]
        for value in replay_rows
    )
    time_strata = all(
        all(
            counts == [4, 4, 4, 4, 16]
            for counts in value["class_bin_counts"].values()
        )
        for value in (teacher_record, null_record)
    )
    teacher_seeds = dict(teacher_record.get("seeds", {}))
    null_seeds = dict(null_record.get("seeds", {}))
    namespaces = bool(
        teacher_seeds.get("positive-states") != teacher_seeds.get("reference-states")
        and null_seeds.get("null-pool") != null_seeds.get("null-swaps")
    )
    null_exchangeability_certificate = {
        "task": null_record.get("task"),
        "derivation_version": null_record.get("derivation_version"),
        "rows": null_record.get("rows"),
        "class_counts": null_record.get("class_counts"),
        "class_bin_counts": null_record.get("class_bin_counts"),
        "pooled_state_namespace_present": int("null-pool" in null_seeds),
        "independent_swap_namespace_present": int("null-swaps" in null_seeds),
        "pool_and_swap_seeds_distinct": int(
            null_seeds.get("null-pool") != null_seeds.get("null-swaps")
        ),
        "replay_verified": int(replay["passed"]),
    }
    null_exchangeability_certificate["passed"] = int(
        null_exchangeability_certificate["task"] == "dirichlet_null"
        and int(null_exchangeability_certificate["rows"] or -1) == 64
        and null_exchangeability_certificate["class_counts"] == [32, 32]
        and all(
            counts == [4, 4, 4, 4, 16]
            for counts in dict(
                null_exchangeability_certificate["class_bin_counts"] or {}
            ).values()
        )
        and bool(null_exchangeability_certificate["pooled_state_namespace_present"])
        and bool(null_exchangeability_certificate["independent_swap_namespace_present"])
        and bool(null_exchangeability_certificate["pool_and_swap_seeds_distinct"])
        and bool(null_exchangeability_certificate["replay_verified"])
    )
    boundary = _boundary_certificate(
        D0BoundarySmoothPotentialUNet(
            dynamics, base_channels=int(args.base_channels)
        )
    )
    _set_seed(_derived_seed(int(args.root_seed), "preflight", "device-smoke-model"))
    smoke_model = D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    smoke_batch = generate_density_ratio_batch(
        stream_plan,
        phase="preflight-device-smoke",
        task="bounded_teacher",
        step=1,
        device=device,
        dtype=torch.float32,
    )
    smoke_model.zero_grad(set_to_none=True)
    smoke_logits = smoke_model(
        smoke_batch.tau, smoke_batch.states, smoke_batch.labels
    ).reshape(-1)
    smoke_loss = classification_loss(smoke_logits, smoke_batch.class_targets)
    smoke_loss.backward()
    smoke_gradients = [
        parameter.grad.detach()
        for parameter in smoke_model.parameters()
        if parameter.grad is not None
    ]
    smoke_finite = bool(
        torch.isfinite(smoke_logits).all()
        and torch.isfinite(smoke_loss.detach())
        and smoke_gradients
        and all(torch.isfinite(value).all() for value in smoke_gradients)
    )
    smoke_grad_norm = float(
        torch.linalg.vector_norm(
            torch.stack([value.norm(2) for value in smoke_gradients])
        ).detach().cpu()
    ) if smoke_gradients else float("nan")
    device_smoke = {
        "device": str(device),
        "batch_fingerprint": smoke_batch.fingerprint,
        "finite_logits": int(bool(torch.isfinite(smoke_logits).all())),
        "finite_loss": int(bool(torch.isfinite(smoke_loss.detach()))),
        "finite_gradients": int(
            bool(smoke_gradients)
            and all(bool(torch.isfinite(value).all()) for value in smoke_gradients)
        ),
        "loss": float(smoke_loss.detach().cpu()),
        "gradient_norm": smoke_grad_norm,
        "passed": int(smoke_finite and math.isfinite(smoke_grad_norm)),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    # The smoke test intentionally constructs a complete backward graph on the
    # production device.  Release it before allocating the independent
    # calibration model and its optimizer state.
    del smoke_model, smoke_batch, smoke_logits, smoke_loss, smoke_gradients
    if device.type == "cuda":
        torch.cuda.empty_cache()
    transitive = provenance.get("transitive_parent_provenance", {})
    transitive = dict(transitive) if isinstance(transitive, Mapping) else {}
    transitive_artifacts = transitive.get("artifacts", {})
    operator_record = (
        dict(transitive_artifacts.get("operator_preflight", {}))
        if isinstance(transitive_artifacts, Mapping)
        else {}
    )
    operator_pass = bool(int(provenance.get("passed", 0))) and bool(operator_record)
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": int(
            np.isfinite(teacher_losses).all()
            and np.isfinite(reference_ratio).all()
            and bool(torch.isfinite(gradient).all())
            and bool(torch.isfinite(predicted_flux).all())
        ),
        "analytic_logit_max_error": float(
            (oracle - direct).detach().abs().max().cpu()
        ),
        "analytic_score_max_error": float(
            (gradient - target_gradient).detach().abs().max().cpu()
        ),
        "analytic_flux_max_error": float(
            (predicted_flux - target_flux).detach().abs().max().cpu()
        ),
        "teacher_normalization_interval": {
            "lower": normalization.get("lower_bound"),
            "upper": normalization.get("upper_bound"),
            **normalization,
        },
        "teacher_bce_improvement_lower_bounds": teacher_signal,
        "oracle_bootstrap_confidence": float(args.preflight_confidence),
        "null_bce_error": abs(float(null_loss) - math.log(2.0)),
        "null_score_max_abs": float(null_cell_score.detach().abs().max().cpu()),
        "null_flux_max_abs": float(null_flux.detach().abs().max().cpu()),
        "class_balance_pass": int(class_balance),
        "time_strata_pass": int(time_strata),
        "null_exchangeability_pass": int(
            null_exchangeability_certificate["passed"]
        ),
        "null_exchangeability_certificate": null_exchangeability_certificate,
        "independent_class_namespaces": int(namespaces),
        "stream_replay_pass": int(replay["passed"]),
        "panel_isolation_pass": int(isolation.get("passed", 0)),
        "boundary_admissible": int(boundary["passed"]),
        "boundary_certificate": boundary,
        "operator_preflight_pass": int(operator_pass),
        "operator_preflight_source": operator_record,
        "device_smoke_pass": int(device_smoke["passed"]),
        "device_smoke": device_smoke,
        "provenance_pass": int(bool(int(provenance.get("passed", 0)))),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    gate = evaluate_ratio_preflight(metrics, thresholds)
    gate["provenance_pass"] = int(bool(int(provenance.get("passed", 0))))
    gate["passed"] = int(
        bool(int(gate.get("passed", 0)))
        and bool(int(provenance.get("passed", 0)))
    )
    atomic_write_json(run_dir / "density_ratio_preflight.json", metrics)
    atomic_write_json(run_dir / "density_ratio_preflight_gate.json", gate)
    calibration = _calibrate_loss_scale(
        run_dir,
        args=args,
        dynamics=dynamics,
        device=device,
        stream_plan=stream_plan,
        manifest=manifest,
    )
    return gate, calibration


def _freeze_selected_profile(
    path: Path, selected: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(selected), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _json_load(path) != normalized:
            raise ArtifactCompatibilityError(
                "frozen density-ratio profile changed on resume"
            )
    else:
        atomic_write_json(path, normalized)
    return normalized


def _bind_confirmation_profile(
    run_dir: Path, selected: Mapping[str, Any]
) -> dict[str, Any]:
    profile_path = run_dir / "selected_density_ratio_profile.json"
    pilot_path = run_dir / "density_ratio_pilot_gate.json"
    normalized = json.loads(json.dumps(dict(selected), sort_keys=True, allow_nan=False))
    if not profile_path.is_file() or _json_load(profile_path) != normalized:
        raise ArtifactCompatibilityError("confirmation profile is not frozen")
    if not pilot_path.is_file():
        raise ArtifactCompatibilityError("confirmation is missing its pilot gate")
    pilot_selected = dict(_json_load(pilot_path).get("selected_profile", {}))
    if pilot_selected != normalized:
        raise ArtifactCompatibilityError("pilot gate and frozen profile disagree")
    record = {
        "schema": RUN_SCHEMA + "-confirmation-profile-binding",
        "schema_version": 1,
        "selected_profile": normalized,
        "selected_profile_sha256": file_fingerprint(profile_path),
        "pilot_gate_sha256": file_fingerprint(pilot_path),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    path = run_dir / "confirmation_profile_binding.json"
    if path.is_file():
        if _json_load(path) != record:
            raise ArtifactCompatibilityError("confirmation profile binding changed")
    else:
        atomic_write_json(path, record)
    return record


def _panel_registry(
    *,
    phase: str,
    panels: Mapping[str, Mapping[str, DensityRatioPanel]],
) -> dict[str, Any]:
    flat = [panel for values in panels.values() for panel in values.values()]
    return {
        "schema": RUN_SCHEMA + "-panel-registry",
        "schema_version": 1,
        "phase": phase,
        "panels": {
            task: {name: panel_identity(panel) for name, panel in values.items()}
            for task, values in panels.items()
        },
        "disjointness": panel_disjointness_record(flat),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _run_pilot(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    thresholds: DensityRatioThresholds,
    loss_scale: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    panels = {
        task: _prepare_panel_set(
            run_dir,
            phase="pilot",
            task=task,
            roles=("a", "b"),
            path_count=int(args.pilot_selection_paths),
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            start_offset=2_000_000 + index * 1_000_000,
        )
        for index, task in enumerate(("bounded_teacher", "dirichlet_null"))
    }
    registry = _panel_registry(phase="pilot", panels=panels)
    if not bool(int(dict(registry["disjointness"]).get("passed", 0))):
        raise ArtifactCompatibilityError("pilot panels are not mutually disjoint")
    atomic_write_json(run_dir / "pilot_panel_registry.json", registry)

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shared_seed = _derived_seed(int(args.root_seed), "pilot", "shared-model")
    for candidate_index, learning_rate in enumerate(args.pilot_learning_rates):
        results: dict[str, Any] = {}
        for task in ("bounded_teacher", "dirichlet_null"):
            task_dir = run_dir / "pilot" / f"lr-{candidate_index:02d}" / task
            fingerprints = _task_fingerprints(
                manifest=manifest,
                phase="pilot",
                task=task,
                model_seed=int(shared_seed),
                learning_rate=float(learning_rate),
                loss_scale=float(loss_scale),
                stream_plan=stream_plan,
                selection_panels=panels[task],
                audit_panels=None,
            )
            try:
                results[task] = run_density_ratio_task(
                    task_dir=task_dir,
                    task=task,
                    selection_panels=panels[task],
                    audit_panels=None,
                    dynamics=dynamics,
                    args=_task_args(
                        args, phase="pilot", learning_rate=float(learning_rate)
                    ),
                    device=device,
                    model_seed=int(shared_seed),
                    learning_rate=float(learning_rate),
                    loss_scale=float(loss_scale),
                    stream_plan=stream_plan,
                    fingerprints=fingerprints,
                    phase="pilot",
                    thresholds=thresholds,
                    show_progress=not bool(args.no_progress),
                )
            except FloatingPointError as exc:
                results[task] = _failed_task_result(
                    task_dir,
                    task=task,
                    model_seed=int(shared_seed),
                    fingerprints=fingerprints,
                    exc=exc,
                )
                failures.append(
                    {
                        "candidate_index": candidate_index,
                        "learning_rate": float(learning_rate),
                        "task": task,
                        "model_seed": int(shared_seed),
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "learning_rate": float(learning_rate),
                "teacher": results["bounded_teacher"],
                "null": results["dirichlet_null"],
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
    pilot = evaluate_density_ratio_pilot(candidates, thresholds)
    selected = select_density_ratio_profile(candidates, thresholds)
    pilot["selected_profile"] = selected
    atomic_write_json(
        run_dir / "pilot_candidate_registry.json",
        {
            "schema": RUN_SCHEMA + "-pilot-candidate-registry",
            "schema_version": 1,
            "candidates": candidates,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "pilot_task_failures.json",
        {
            "schema": RUN_SCHEMA + "-pilot-task-failures",
            "schema_version": 1,
            "failure_count": len(failures),
            "failures": failures,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(run_dir / "density_ratio_pilot_gate.json", pilot)
    if bool(int(pilot.get("passed", 0))):
        _freeze_selected_profile(
            run_dir / "selected_density_ratio_profile.json", selected
        )
    return pilot, selected


def _run_confirmation(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pilot: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    thresholds: DensityRatioThresholds,
    loss_scale: float,
    selected_profile: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(int(selected_profile.get("selected", selected_profile.get("passed", 0)))):
        raise ArtifactCompatibilityError("confirmation requires an eligible profile")
    profile = selected_profile.get("profile", selected_profile)
    if not isinstance(profile, Mapping):
        raise ArtifactCompatibilityError("selected profile payload is missing")
    learning_rate = float(profile["learning_rate"])
    profile_binding = _bind_confirmation_profile(run_dir, selected_profile)

    panels: dict[str, dict[str, DensityRatioPanel]] = {}
    for task_index, task in enumerate(("bounded_teacher", "dirichlet_null")):
        selection = _prepare_panel_set(
            run_dir,
            phase="confirm",
            task=task,
            roles=("a", "b"),
            path_count=int(args.confirm_selection_paths),
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            start_offset=4_000_000 + task_index * 1_000_000,
        )
        audit = _prepare_panel_set(
            run_dir,
            phase="confirm",
            task=task,
            roles=("c", "d"),
            path_count=int(args.confirm_audit_paths),
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            start_offset=6_000_000 + task_index * 1_000_000,
        )
        panels[task] = {**selection, **audit}
    registry = _panel_registry(phase="confirm", panels=panels)
    if not bool(int(dict(registry["disjointness"]).get("passed", 0))):
        raise ArtifactCompatibilityError("confirmation panels are not mutually disjoint")
    atomic_write_json(run_dir / "confirmation_panel_registry.json", registry)

    teacher_results: list[dict[str, Any]] = []
    null_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in args.confirm_model_seeds:
        for task, output in (
            ("bounded_teacher", teacher_results),
            ("dirichlet_null", null_results),
        ):
            task_dir = run_dir / "confirmation" / task / f"seed-{int(seed)}"
            selection_panels = {name: panels[task][name] for name in ("a", "b")}
            audit_panels = {name: panels[task][name] for name in ("c", "d")}
            fingerprints = _task_fingerprints(
                manifest=manifest,
                phase="confirm",
                task=task,
                model_seed=int(seed),
                learning_rate=learning_rate,
                loss_scale=float(loss_scale),
                stream_plan=stream_plan,
                selection_panels=selection_panels,
                audit_panels=audit_panels,
                profile_binding=profile_binding,
            )
            try:
                result = run_density_ratio_task(
                    task_dir=task_dir,
                    task=task,
                    selection_panels=selection_panels,
                    audit_panels=audit_panels,
                    dynamics=dynamics,
                    args=_task_args(
                        args, phase="confirm", learning_rate=learning_rate
                    ),
                    device=device,
                    model_seed=int(seed),
                    learning_rate=learning_rate,
                    loss_scale=float(loss_scale),
                    stream_plan=stream_plan,
                    fingerprints=fingerprints,
                    phase="confirm",
                    thresholds=thresholds,
                    show_progress=not bool(args.no_progress),
                )
            except FloatingPointError as exc:
                result = _failed_task_result(
                    task_dir,
                    task=task,
                    model_seed=int(seed),
                    fingerprints=fingerprints,
                    exc=exc,
                )
                failures.append(
                    {
                        "task": task,
                        "model_seed": int(seed),
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
            output.append(result)
    controls = evaluate_density_ratio_controls(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        teacher_results=teacher_results,
        null_results=null_results,
        thresholds=thresholds,
    )
    controls.update(
        {
            "teacher_results": teacher_results,
            "null_results": null_results,
            "selected_profile": dict(selected_profile),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    )
    atomic_write_json(
        run_dir / "density_ratio_teacher_confirmation.json",
        {
            "task_results": teacher_results,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "density_ratio_null_confirmation.json",
        {
            "task_results": null_results,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "confirmation_task_failures.json",
        {
            "failure_count": len(failures),
            "failures": failures,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(run_dir / "density_ratio_control_gate.json", controls)
    return controls, teacher_results, null_results


def _workflow_report(
    *,
    provenance: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pilot: Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    require_gate: str,
    thresholds: DensityRatioThresholds,
) -> dict[str, Any]:
    report = evaluate_density_ratio_workflow(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        teacher_results=teacher_results,
        null_results=null_results,
        require_gate=require_gate,
        thresholds=thresholds,
    )
    report.setdefault("schema", RUN_SCHEMA + "-report")
    report.setdefault("schema_version", 1)
    report["physical_training_performed"] = 0
    report["sampling_performed"] = 0
    return report


def _save_report(run_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(
        run_dir / "density_ratio_control_decision.json",
        dict(report.get("decision", {})),
    )
    atomic_write_json(run_dir / "density_ratio_control_report.json", dict(report))


def _atomic_save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format=path.suffix.removeprefix("."),
            dpi=150,
            bbox_inches="tight",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_plot_artifacts(run_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    written: list[str] = []
    for phase, pattern, filename in (
        ("pilot", "pilot/**/training_history.csv", "pilot_learning_curves.png"),
        (
            "confirmation",
            "confirmation/**/training_history.csv",
            "confirmation_learning_curves.png",
        ),
    ):
        paths = sorted(run_dir.glob(pattern))
        if not paths:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                continue
            steps = [int(row["step"]) for row in rows]
            losses = [float(row["scaled_loss"]) for row in rows]
            gradients = [float(row["scaled_preclip_gradient_norm"]) for row in rows]
            label = path.parent.relative_to(run_dir).as_posix()
            axes[0].plot(steps, losses, linewidth=0.9, label=label)
            axes[1].plot(steps, gradients, linewidth=0.9, label=label)
        axes[0].set_title(f"{phase} scaled BCE")
        axes[1].set_title(f"{phase} scaled pre-clip gradient")
        axes[1].set_yscale("log")
        for axis in axes:
            axis.set_xlabel("step")
            axis.grid(alpha=0.25)
        if len(paths) <= 10:
            axes[1].legend(fontsize=6)
        figure.tight_layout()
        output = run_dir / filename
        _atomic_save_figure(figure, output)
        plt.close(figure)
        written.append(output.name)

    pilot_path = run_dir / "density_ratio_pilot_gate.json"
    if pilot_path.is_file():
        pilot = _json_load(pilot_path)
        candidates = [dict(value) for value in pilot.get("candidate_gates", [])]
        if candidates:
            figure, axes = plt.subplots(1, 2, figsize=(9, 4))
            rates = [float(value["learning_rate"]) for value in candidates]
            risks = [
                float(value.get("teacher_mean_ab_bce", math.nan)) for value in candidates
            ]
            clips = [
                float(value.get("maximum_clip_fraction_observed", math.nan))
                for value in candidates
            ]
            axes[0].plot(rates, risks, marker="o")
            axes[1].plot(rates, clips, marker="o")
            axes[1].axhline(0.10, color="red", linestyle="--", linewidth=1)
            for axis in axes:
                axis.set_xscale("log")
                axis.set_xlabel("learning rate")
                axis.grid(alpha=0.25)
            axes[0].set_title("teacher mean panel-A/B BCE")
            axes[1].set_title("maximum clipping fraction")
            figure.tight_layout()
            output = run_dir / "selected_density_ratio_profile.png"
            _atomic_save_figure(figure, output)
            plt.close(figure)
            written.append(output.name)
    return written


def _write_summary_csvs(run_dir: Path) -> list[str]:
    written: list[str] = []
    pilot_path = run_dir / "density_ratio_pilot_gate.json"
    if pilot_path.is_file():
        rows = []
        for index, value in enumerate(
            dict(item) for item in _json_load(pilot_path).get("candidate_gates", [])
        ):
            rows.append(
                {
                    "candidate_index": index,
                    "learning_rate": value.get("learning_rate"),
                    "passed": value.get("passed"),
                    "teacher_mean_ab_bce": value.get("teacher_mean_ab_bce"),
                    "teacher_panel_b_bce": value.get("teacher_panel_b_bce"),
                    "maximum_clip_fraction_observed": value.get(
                        "maximum_clip_fraction_observed"
                    ),
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
        if rows:
            atomic_write_csv(run_dir / "pilot_profile_summary.csv", rows)
            written.append("pilot_profile_summary.csv")

    rows = []
    for filename, task in (
        ("density_ratio_teacher_confirmation.json", "bounded_teacher"),
        ("density_ratio_null_confirmation.json", "dirichlet_null"),
    ):
        path = run_dir / filename
        if not path.is_file():
            continue
        for result in _json_load(path).get("task_results", []):
            metrics = dict(dict(result).get("metrics", {}))
            audit = dict(metrics.get("audit_panels", {}))
            panel_c = dict(audit.get("c", {}))
            panel_d = dict(audit.get("d", {}))
            rows.append(
                {
                    "task": task,
                    "model_seed": result.get("model_seed"),
                    "complete": metrics.get("complete"),
                    "finite": metrics.get("finite"),
                    "selected_step": metrics.get("selected_step"),
                    "nominee_step": metrics.get("nominee_step"),
                    "post_warmup_clip_fraction": metrics.get(
                        "post_warmup_clip_fraction"
                    ),
                    "panel_c_bce": dict(panel_c.get("overall", {})).get("bce"),
                    "panel_c_lower_bound": dict(panel_c.get("overall", {})).get(
                        "lower_bound"
                    ),
                    "panel_d_bce": dict(panel_d.get("overall", {})).get("bce"),
                    "panel_d_lower_bound": dict(panel_d.get("overall", {})).get(
                        "lower_bound"
                    ),
                    "panel_c_score_gain": panel_c.get("overall_score_gain"),
                    "panel_d_score_gain": panel_d.get("overall_score_gain"),
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
    if rows:
        atomic_write_csv(run_dir / "confirmation_seed_metrics.csv", rows)
        written.append("confirmation_seed_metrics.csv")
    return written


def _finish(
    run_dir: Path,
    *,
    report: Mapping[str, Any],
    stage: str,
    phase: str,
    execution_failed: bool = False,
    skips: Sequence[Mapping[str, Any]] = (),
) -> int:
    final_skips = [dict(value) for value in skips]
    try:
        _write_summary_csvs(run_dir)
        _write_plot_artifacts(run_dir)
    except Exception as exc:
        execution_failed = True
        final_skips.append(
            {"stage": "report_artifacts", "reason": f"{type(exc).__name__}: {exc}"}
        )
        atomic_write_json(
            run_dir / "report_artifact_failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )
    required_pass = 0 if execution_failed else int(report.get("required_gate_pass", 0))
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    decision = dict(report.get("decision", {}))
    _write_status(
        run_dir,
        status="failed" if execution_failed else "complete",
        outcome="implementation_error"
        if execution_failed
        else ("complete" if required_pass else "gate_failed"),
        phase=phase,
        stage=stage,
        required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=decision.get("decision", "controls_not_run"),
        recommended_next_action=decision.get("recommended_next_action"),
        physical_training_authorized=int(
            decision.get("physical_training_authorized", 0)
        )
        if not execution_failed
        else 0,
        physical_training_performed=0,
        sampling_authorized=0,
        sampling_performed=0,
        skips=final_skips,
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_size=int(registry_path.stat().st_size),
    )
    return 2 if execution_failed or not required_pass else 0


def _load_report_inputs(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    preflight_path = run_dir / "density_ratio_preflight_gate.json"
    pilot_path = run_dir / "density_ratio_pilot_gate.json"
    teacher_path = run_dir / "density_ratio_teacher_confirmation.json"
    null_path = run_dir / "density_ratio_null_confirmation.json"
    preflight = (
        _json_load(preflight_path)
        if preflight_path.is_file()
        else _not_evaluated_gate("density_ratio_preflight", "preflight was not run")
    )
    pilot = (
        _json_load(pilot_path)
        if pilot_path.is_file()
        else _not_evaluated_gate("density_ratio_pilot", "pilot was not run")
    )
    teacher_results = (
        [dict(value) for value in _json_load(teacher_path).get("task_results", [])]
        if teacher_path.is_file()
        else []
    )
    null_results = (
        [dict(value) for value in _json_load(null_path).get("task_results", [])]
        if null_path.is_file()
        else []
    )
    return preflight, pilot, teacher_results, null_results


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    if not args.no_progress:
        print(f"density-ratio control run directory: {run_dir.resolve()}", flush=True)
    thresholds = DensityRatioThresholds(
        oracle_confidence=float(args.preflight_confidence),
        confirm_confidence=float(args.bootstrap_confidence),
        audit_confidence=float(args.bootstrap_confidence),
        maximum_clip_fraction=0.10,
        pilot_learning_rates=tuple(float(value) for value in args.pilot_learning_rates),
        expected_teacher_seeds=len(args.confirm_model_seeds),
        minimum_passing_teacher_seeds=2,
        expected_null_seeds=len(args.confirm_model_seeds),
        audit_paths_per_panel=int(args.confirm_audit_paths),
        anchors_per_path=32,
    )
    mutation_started = False
    try:
        device = _device(args.device)
        backend = configure_exact_torch_backend(device)
        runtime = _runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
        source_hash, source_paths = _source_record()
        parent = verify_parent_stability_run(args.parent_stability_run_dir)
        provenance = {
            **dict(parent),
            "schema": RUN_SCHEMA + "-parent-provenance",
            "schema_version": 1,
            "verifier_source_fingerprint": source_hash,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        provenance_path = run_dir / "parent_provenance.json"
        if provenance_path.is_file():
            if _json_load(provenance_path) != provenance:
                raise ArtifactCompatibilityError("resume parent provenance mismatch")
        elif resumed:
            raise ArtifactCompatibilityError("resume is missing parent provenance")
        else:
            atomic_write_json(provenance_path, provenance)

        scientific = _scientific_config(args, parent, thresholds)
        scientific_fingerprint = config_fingerprint(scientific)
        manifest = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "created_at": _now(),
            "run_dir": str(run_dir.resolve()),
            "scientific_config": scientific,
            "scientific_fingerprint": scientific_fingerprint,
            "runtime": runtime,
            "runtime_fingerprint": runtime_fingerprint,
            "source_fingerprint": source_hash,
            "source_paths": source_paths,
            "parent_provenance_sha256": file_fingerprint(provenance_path),
            "claim_scope": CLAIM_SCOPE,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            existing = _json_load(manifest_path)
            for key in (
                "schema",
                "schema_version",
                "scientific_config",
                "scientific_fingerprint",
                "runtime",
                "runtime_fingerprint",
                "source_fingerprint",
                "source_paths",
                "parent_provenance_sha256",
                "claim_scope",
            ):
                if existing.get(key) != manifest.get(key):
                    raise ArtifactCompatibilityError(
                        f"resume manifest mismatch for {key}"
                    )
            manifest = existing
        elif resumed:
            raise ArtifactCompatibilityError("resume is missing frozen manifest")
        else:
            atomic_write_json(manifest_path, manifest)

        status_path = run_dir / "run_status.json"
        previous = _json_load(status_path) if status_path.is_file() else {}
        if resumed and str(previous.get("status", "")) in {"complete", "failed"}:
            _verify_terminal_registry(run_dir)
        _write_status(
            run_dir,
            status="running",
            outcome="running",
            phase="provenance",
            stage=str(args.stage),
            require_gate=str(args.require_gate),
            attempt_count=int(previous.get("attempt_count", 0)) + 1,
        )
        mutation_started = True

        if args.stage == "report":
            preflight, pilot, teacher_results, null_results = _load_report_inputs(
                run_dir
            )
            report = _workflow_report(
                provenance=provenance,
                preflight=preflight,
                pilot=pilot,
                teacher_results=teacher_results,
                null_results=null_results,
                require_gate=str(args.require_gate),
                thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(
                run_dir, report=report, stage="report", phase="report"
            )

        dynamics = _make_dynamics(args)
        horizon = float(natural_horizon(dynamics))
        if not math.isclose(
            horizon,
            float(parent.get("horizon", math.nan)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ArtifactCompatibilityError(
                "density-ratio horizon does not match verified parent"
            )
        stream_plan = build_density_ratio_stream_plan(
            root_seed=int(args.root_seed),
            grid_size=int(args.grid_size),
            horizon=horizon,
            label=3,
            bin_counts=(4, 4, 4, 4, 16),
            teacher_epsilon=0.5,
        )
        stream_record = stream_plan_record(stream_plan)
        stream_path = run_dir / "density_ratio_stream_plan.json"
        if stream_path.is_file():
            if _json_load(stream_path) != stream_record:
                raise ArtifactCompatibilityError(
                    "resume density-ratio stream plan mismatch"
                )
        elif resumed:
            raise ArtifactCompatibilityError(
                "resume is missing the frozen density-ratio stream plan"
            )
        else:
            atomic_write_json(stream_path, stream_record)

        _write_status(run_dir, status="running", phase="preflight")
        preflight, calibration = _run_preflight(
            run_dir,
            args=args,
            manifest=manifest,
            provenance=provenance,
            dynamics=dynamics,
            device=device,
            stream_plan=stream_plan,
            thresholds=thresholds,
        )
        loss_scale = float(calibration["loss_scale"])
        pilot = _not_evaluated_gate("density_ratio_pilot", "pilot was not run")
        teacher_results: list[dict[str, Any]] = []
        null_results: list[dict[str, Any]] = []
        if args.stage == "preflight" or not bool(int(preflight.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance,
                preflight=preflight,
                pilot=pilot,
                teacher_results=teacher_results,
                null_results=null_results,
                require_gate=str(args.require_gate),
                thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(
                run_dir,
                report=report,
                stage=str(args.stage),
                phase="preflight",
                skips=(
                    []
                    if bool(int(preflight.get("passed", 0)))
                    else [
                        {
                            "stage": "pilot_and_confirmation",
                            "reason": "density-ratio preflight failed",
                        }
                    ]
                ),
            )

        selected_profile: dict[str, Any]
        if args.stage in {"pilot", "all"}:
            _write_status(run_dir, status="running", phase="pilot")
            pilot, selected_profile = _run_pilot(
                run_dir,
                args=args,
                manifest=manifest,
                dynamics=dynamics,
                device=device,
                stream_plan=stream_plan,
                thresholds=thresholds,
                loss_scale=loss_scale,
            )
            if args.stage == "pilot" or not bool(int(pilot.get("passed", 0))):
                report = _workflow_report(
                    provenance=provenance,
                    preflight=preflight,
                    pilot=pilot,
                    teacher_results=teacher_results,
                    null_results=null_results,
                    require_gate=str(args.require_gate),
                    thresholds=thresholds,
                )
                _save_report(run_dir, report)
                return _finish(
                    run_dir,
                    report=report,
                    stage=str(args.stage),
                    phase="pilot",
                    skips=(
                        []
                        if bool(int(pilot.get("passed", 0)))
                        else [
                            {
                                "stage": "confirmation",
                                "reason": "no density-ratio pilot profile qualified",
                            }
                        ]
                    ),
                )
        else:
            pilot_path = run_dir / "density_ratio_pilot_gate.json"
            profile_path = run_dir / "selected_density_ratio_profile.json"
            if not pilot_path.is_file() or not profile_path.is_file():
                raise ArtifactCompatibilityError(
                    "confirmation requires a completed, frozen pilot"
                )
            pilot = _json_load(pilot_path)
            if not bool(int(pilot.get("passed", 0))):
                raise ArtifactCompatibilityError(
                    "confirmation requires a passing density-ratio pilot"
                )
            selected_profile = _json_load(profile_path)
            if dict(pilot.get("selected_profile", {})) != selected_profile:
                raise ArtifactCompatibilityError(
                    "pilot gate and frozen density-ratio profile disagree"
                )

        _write_status(run_dir, status="running", phase="confirmation")
        _, teacher_results, null_results = _run_confirmation(
            run_dir,
            args=args,
            manifest=manifest,
            provenance=provenance,
            preflight=preflight,
            pilot=pilot,
            dynamics=dynamics,
            device=device,
            stream_plan=stream_plan,
            thresholds=thresholds,
            loss_scale=loss_scale,
            selected_profile=selected_profile,
        )
        report = _workflow_report(
            provenance=provenance,
            preflight=preflight,
            pilot=pilot,
            teacher_results=teacher_results,
            null_results=null_results,
            require_gate=str(args.require_gate),
            thresholds=thresholds,
        )
        _save_report(run_dir, report)
        return _finish(
            run_dir,
            report=report,
            stage=str(args.stage),
            phase="confirmation",
        )
    except Exception as exc:
        if resumed and not mutation_started:
            if not args.no_progress:
                print(
                    "density-ratio control resume rejected without mutation: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            return 2
        atomic_write_json(
            run_dir / "failure.json",
            {
                "schema": RUN_SCHEMA + "-failure",
                "schema_version": 1,
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": str(args.stage),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )
        failed = _not_evaluated_gate("workflow", f"{type(exc).__name__}: {exc}")
        report = _workflow_report(
            provenance=failed,
            preflight=failed,
            pilot=failed,
            teacher_results=[],
            null_results=[],
            require_gate=str(args.require_gate),
            thresholds=thresholds,
        )
        _save_report(run_dir, report)
        _finish(
            run_dir,
            report=report,
            stage=str(args.stage),
            phase="failure",
            execution_failed=True,
            skips=[
                {
                    "stage": "remaining",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            ],
        )
        if not args.no_progress:
            print(
                f"density-ratio control failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
