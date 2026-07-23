"""Paired-mixture stability confirmation for D0 density-ratio controls.

This additive workflow reduces the variance of the already-validated balanced
BCE control objective with a common-gamma mixture coupling and deterministic
gradient accumulation.  It is deliberately controls-only: it never trains on
physical states and never imports or invokes a reverse sampler.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

import mnist.diag_d0_score_density_ratio_controls as base
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
    bounded_teacher_anchor_indices,
)
from mnist.d0_score_density_ratio import (
    DensityRatioPanel,
    DensityRatioStreamPlan,
    build_density_ratio_stream_plan,
    panel_disjointness_record,
    panel_identity,
    stream_plan_record,
)
from mnist.d0_score_density_ratio_paired import (
    PAIRED_MIXTURE_ACCUMULATION_VERSION,
    PAIRED_MIXTURE_OBJECTIVE_VERSION,
    PAIRED_MIXTURE_SCHEMA,
    PAIRED_MIXTURE_STREAM_VERSION,
    PairedMixtureMicrobatch,
    PairedMixtureStreamPlan,
    accumulation_diagnostics,
    backward_accumulated_objective,
    build_paired_mixture_stream_plan,
    generate_accumulated_paired_stream,
    generate_paired_mixture_microbatch,
    paired_mixture_replay_record,
    paired_mixture_stream_plan_record,
    verify_paired_mixture_replay,
    weighted_paired_softplus_components,
    weighted_paired_softplus_loss,
)
from mnist.d0_score_density_ratio_stability_gate import (
    RatioStabilityThresholds,
    evaluate_paired_ratio_preflight,
    evaluate_stability_pilot,
    evaluate_stability_pilot_candidate,
    evaluate_stability_pilot_level,
    evaluate_ratio_stability_controls,
    evaluate_ratio_stability_workflow,
    evaluate_teacher_seed,
    evaluate_null_seed,
    select_stability_profile,
)
from mnist.d0_score_density_ratio_stability_provenance import (
    PARENT_LOSS_SCALE,
    verify_parent_density_ratio_run,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    init_ema_state,
    natural_horizon,
    temporary_ema_weights,
    update_ema_state,
)


RUN_SCHEMA = "experiment12-d0-score-density-ratio-stability-confirmation"
RUN_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
CLAIM_SCOPE = "paired-mixture bounded synthetic density-ratio controls only"

EXPECTED_KERNEL = dict(base.EXPECTED_KERNEL)
DEFAULTS: dict[str, Any] = {
    **EXPECTED_KERNEL,
    "root_seed": 260851,
    "pilot_learning_rates": (3e-5, 1e-5),
    "accumulation_levels": (2, 4, 8),
    "microbatch_clusters": 32,
    "pilot_steps": 2_000,
    "confirm_steps": 4_000,
    "pilot_selection_paths": 16,
    "confirm_selection_paths": 32,
    "confirm_audit_paths": 32,
    "preflight_paths": 128,
    "base_channels": 32,
    "validation_batch_size": 64,
    "weight_decay": 1e-4,
    "ema_decay": 0.99,
    "grad_clip": 1.0,
    "clip_warmup_steps": 500,
    "loss_scale": float(PARENT_LOSS_SCALE),
    "bootstrap_reps": 10_000,
    "bootstrap_confidence": 0.90,
    "preflight_confidence": 0.99,
    "confirm_model_seeds": (260861, 260862, 260863),
    "pilot_validation_steps": (
        0, 25, 50, 100, 150, 250, 500, 750, 1_000, 1_500, 2_000
    ),
    "confirm_dense_validation_steps": (0, 25, 50, 100, 150, 250),
    "confirm_validation_every": 250,
}


def _json_load(path: str | Path) -> dict[str, Any]:
    return base._json_load(path)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


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
    parser.add_argument("--parent-density-ratio-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path(
            "runs/experiment12_d0_score_density_ratio_stability_confirmation"
        ),
    )
    parser.add_argument("--run-name", default="production-paired-density-ratio-controls")
    parser.add_argument(
        "--require-gate", choices=("none", "preflight", "pilot", "controls"),
        default="none",
    )
    parser.add_argument("--root-seed", type=int, default=DEFAULTS["root_seed"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")

    for name in (
        "grid_size", "sample_steps", "reference_substeps", "base_channels",
        "validation_batch_size", "pilot_steps", "confirm_steps",
        "pilot_selection_paths", "confirm_selection_paths",
        "confirm_audit_paths", "preflight_paths", "microbatch_clusters",
        "bootstrap_reps", "confirm_validation_every", "clip_warmup_steps",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=DEFAULTS[name]
        )
    for name in (
        "tau_eff", "alpha_eff", "mass_floor", "limiter_fraction", "lambda_mix",
        "weight_decay", "ema_decay", "grad_clip", "loss_scale",
        "bootstrap_confidence", "preflight_confidence",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=DEFAULTS[name]
        )
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff",), default="alpha_eff")
    parser.add_argument(
        "--pilot-learning-rates", type=base._parse_csv_floats,
        default=DEFAULTS["pilot_learning_rates"],
    )
    parser.add_argument(
        "--accumulation-levels", type=base._parse_csv_ints,
        default=DEFAULTS["accumulation_levels"],
    )
    parser.add_argument(
        "--confirm-model-seeds", type=base._parse_csv_ints,
        default=DEFAULTS["confirm_model_seeds"],
    )
    parser.add_argument(
        "--pilot-validation-steps", type=base._parse_csv_ints,
        default=DEFAULTS["pilot_validation_steps"],
    )
    parser.add_argument(
        "--confirm-dense-validation-steps", type=base._parse_csv_ints,
        default=DEFAULTS["confirm_dense_validation_steps"],
    )
    args = parser.parse_args(argv)

    for name in (
        "grid_size", "sample_steps", "reference_substeps", "base_channels",
        "validation_batch_size", "pilot_steps", "confirm_steps",
        "pilot_selection_paths", "confirm_selection_paths",
        "confirm_audit_paths", "preflight_paths", "microbatch_clusters",
        "bootstrap_reps", "confirm_validation_every",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.microbatch_clusters) != 32:
        parser.error("paired estimator v1 requires --microbatch-clusters 32")
    if tuple(args.accumulation_levels) != tuple(sorted(set(args.accumulation_levels))):
        parser.error("--accumulation-levels must be sorted and unique")
    if any(int(value) <= 0 for value in args.accumulation_levels):
        parser.error("accumulation levels must be positive")
    if any(float(value) <= 0.0 for value in args.pilot_learning_rates):
        parser.error("pilot learning rates must be positive")
    if len(args.confirm_model_seeds) != 3 or len(set(args.confirm_model_seeds)) != 3:
        parser.error("--confirm-model-seeds must contain three distinct seeds")
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
    if not math.isclose(
        float(args.loss_scale), float(PARENT_LOSS_SCALE), rel_tol=0.0, abs_tol=0.0
    ):
        parser.error("--loss-scale is frozen to the verified parent multiplier")
    if args.stage in {"confirm", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    if args.require_gate != "none":
        mismatches = [
            f"{key}={getattr(args, key)!r}, expected {expected!r}"
            for key, expected in DEFAULTS.items()
            if hasattr(args, key) and not base._semantic_equal(getattr(args, key), expected)
        ]
        if mismatches:
            parser.error(
                "required production gate rejects overrides: " + "; ".join(mismatches)
            )
    return args


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        path = args.resume_run_dir.resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path, True
    root = args.runs_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(args.run_name)
    ).strip("-")
    path = root / f"{stamp}_{safe or 'run'}"
    suffix = 1
    while path.exists():
        path = root / f"{stamp}_{safe or 'run'}-{suffix}"
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


def _source_record() -> tuple[str, list[str]]:
    here = Path(__file__).resolve()
    names = (
        here.name,
        "d0_score_density_ratio_paired.py",
        "d0_score_density_ratio_stability_gate.py",
        "d0_score_density_ratio_stability_provenance.py",
        "diag_d0_score_density_ratio_controls.py",
        "d0_score_density_ratio.py",
        "d0_score_density_ratio_gate.py",
        "d0_score_density_ratio_provenance.py",
        "d0_score_boundary_controls.py",
        "d0_score_optimizer_scale.py",
        "d0_one_image_gate.py",
        "eulerian_flux_mnist.py",
    )
    paths = [here.with_name(name) for name in names]
    existing = [path for path in paths if path.is_file()]
    return source_fingerprint(existing), [str(path) for path in existing]


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
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    if not registry_path.is_file() or not status_path.is_file():
        raise ArtifactCompatibilityError("terminal run lacks status or registry")
    registry, status = _json_load(registry_path), _json_load(status_path)
    if (
        registry.get("schema") != RUN_SCHEMA + "-artifact-registry"
        or status.get("artifact_registry_sha256") != file_fingerprint(registry_path)
        or int(status.get("artifact_registry_size", -1)) != registry_path.stat().st_size
    ):
        raise ArtifactCompatibilityError("terminal registry binding mismatch")
    excluded = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    records = dict(registry.get("records", {}))
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    if set(records) != actual:
        raise ArtifactCompatibilityError("terminal registry file set mismatch")
    for relative, raw in records.items():
        path, record = run_dir / relative, dict(raw)
        if (
            not path.is_file()
            or record.get("sha256") != file_fingerprint(path)
            or int(record.get("size", -1)) != path.stat().st_size
        ):
            raise ArtifactCompatibilityError(f"terminal artifact mismatch: {relative}")
    return registry


def _scientific_config(
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    thresholds: RatioStabilityThresholds,
) -> dict[str, Any]:
    value = {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "model_schema": BOUNDARY_SMOOTH_MODEL_VERSION,
        "paired_estimator_schema": PAIRED_MIXTURE_SCHEMA,
        "paired_objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
        "paired_stream_version": PAIRED_MIXTURE_STREAM_VERSION,
        "paired_accumulation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
        "kernel": {key: getattr(args, key) for key in EXPECTED_KERNEL},
        "root_seed": int(args.root_seed),
        "loss_scale": float(args.loss_scale),
        "microbatch": {
            "clusters": int(args.microbatch_clusters),
            "time_bin_counts": [4, 4, 4, 4, 16],
            "teacher_coupling": "common-gamma-stochastic-anchor",
            "null_coupling": "independent-dirichlet-pooled-label-swap",
        },
        "pilot": {
            "learning_rates": list(args.pilot_learning_rates),
            "accumulation_levels": list(args.accumulation_levels),
            "effective_cluster_counts": [
                int(args.microbatch_clusters) * int(value)
                for value in args.accumulation_levels
            ],
            "hierarchical_stop": "first-level-with-eligible-profile",
            "steps": int(args.pilot_steps),
            "selection_paths_per_panel": int(args.pilot_selection_paths),
            "validation_steps": list(args.pilot_validation_steps),
        },
        "confirmation": {
            "steps": int(args.confirm_steps),
            "model_seeds": list(args.confirm_model_seeds),
            "selection_paths_per_panel": int(args.confirm_selection_paths),
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
            "adaptive_loss_scaling": 0,
            "gradient_accumulation": "mean-then-clip-once",
        },
        "preflight": {
            "paths": int(args.preflight_paths),
            "confidence": float(args.preflight_confidence),
            "bootstrap_reps": int(args.bootstrap_reps),
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
    # The panel format and exact laws are deliberately inherited unchanged.
    return base._prepare_panel_set(
        run_dir,
        phase=phase,
        task=task,
        roles=roles,
        path_count=path_count,
        stream_plan=stream_plan,
        scientific_fingerprint=scientific_fingerprint,
        start_offset=start_offset,
    )


def _panel_registry(
    *, phase: str, panels: Mapping[str, Mapping[str, DensityRatioPanel]]
) -> dict[str, Any]:
    flat = [panel for group in panels.values() for panel in group.values()]
    return {
        "schema": RUN_SCHEMA + "-panel-registry",
        "schema_version": 1,
        "phase": str(phase),
        "panels": {
            task: {name: panel_identity(panel) for name, panel in group.items()}
            for task, group in panels.items()
        },
        "disjointness": panel_disjointness_record(flat),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _task_args(
    args: argparse.Namespace,
    *,
    phase: str,
    learning_rate: float,
    accumulation_level: int,
) -> argparse.Namespace:
    values = vars(args).copy()
    values["learning_rate"] = float(learning_rate)
    values["accumulation_level"] = int(accumulation_level)
    values["train_steps"] = int(
        args.pilot_steps if phase.startswith("pilot") else args.confirm_steps
    )
    values["validation_steps"] = (
        tuple(args.pilot_validation_steps)
        if phase.startswith("pilot")
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


def _task_fingerprints(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    task: str,
    model_seed: int,
    learning_rate: float,
    accumulation_level: int,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
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
        "paired_estimator_schema": PAIRED_MIXTURE_SCHEMA,
        "paired_objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
        "paired_stream_version": PAIRED_MIXTURE_STREAM_VERSION,
        "paired_accumulation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
        "phase": str(phase),
        "task": str(task),
        "model_seed": int(model_seed),
        "learning_rate": float(learning_rate),
        "accumulation_level": int(accumulation_level),
        "microbatch_clusters": 32,
        "loss_scale": float(PARENT_LOSS_SCALE),
        "stream_plan_fingerprint": stream_plan.fingerprint,
        "paired_stream_plan_fingerprint": paired_stream_plan.fingerprint,
        "selection_panel_identities": {
            name: panel_identity(panel) for name, panel in selection_panels.items()
        },
        "audit_panel_identities": None if audit_panels is None else {
            name: panel_identity(panel) for name, panel in audit_panels.items()
        },
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    if profile_binding is not None:
        normalized = json.loads(json.dumps(dict(profile_binding), sort_keys=True))
        value["profile_binding"] = normalized
        value["profile_binding_fingerprint"] = config_fingerprint(normalized)
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
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
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
            "accumulation_cursor": 0,
            "paired_estimator_schema": PAIRED_MIXTURE_SCHEMA,
            "paired_objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
            "paired_stream_version": PAIRED_MIXTURE_STREAM_VERSION,
            "paired_accumulation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
            "stream_plan": stream_plan_record(stream_plan),
            "paired_stream_plan": paired_mixture_stream_plan_record(
                paired_stream_plan
            ),
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
    *, device: torch.device, task: str, fingerprints: Mapping[str, Any]
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
        or value.get("paired_estimator_schema") != PAIRED_MIXTURE_SCHEMA
        or value.get("paired_objective_version") != PAIRED_MIXTURE_OBJECTIVE_VERSION
        or value.get("paired_stream_version") != PAIRED_MIXTURE_STREAM_VERSION
        or value.get("paired_accumulation_version")
        != PAIRED_MIXTURE_ACCUMULATION_VERSION
        or int(value.get("accumulation_cursor", -1)) != 0
    ):
        raise ArtifactCompatibilityError("legacy, foreign, or partial paired checkpoint")
    required = {
        "step", "stream_cursor", "model_state_dict", "ema_state_dict",
        "optimizer_state_dict", "history", "validation_records",
        "checkpoint_selection", "rng_state", "stream_plan",
        "paired_stream_plan", "fingerprints",
    }
    if not required.issubset(value):
        raise ArtifactCompatibilityError("paired checkpoint is incomplete")
    return dict(value)


def _load_completed_task(
    task_dir: Path,
    *, device: torch.device, task: str, fingerprints: Mapping[str, Any]
) -> dict[str, Any] | None:
    result_path = task_dir / "task_result.json"
    status_path = task_dir / "task_status.json"
    if not result_path.is_file() and not status_path.is_file():
        return None
    if not result_path.is_file() or not status_path.is_file():
        return None
    result, status = _json_load(result_path), _json_load(status_path)
    if status.get("status") != "complete":
        return None
    checkpoints = task_dir / "checkpoints"
    latest_path, best_path = checkpoints / "latest.json", checkpoints / "best.json"
    if not latest_path.is_file() or not best_path.is_file():
        raise ArtifactCompatibilityError("completed paired task lacks checkpoint pointers")
    latest, best = _json_load(latest_path), _json_load(best_path)
    files = {
        "latest": checkpoints / str(latest.get("filename", "")),
        "selected": checkpoints / str(best.get("selected_filename", "")),
        "nominee": checkpoints / str(best.get("nominee_filename", "")),
        "best_copy": checkpoints / str(best.get("best_ema_filename", "")),
        "nominee_copy": checkpoints / str(best.get("nominee_ema_filename", "")),
    }
    summary = dict(result.get("training_summary", {}))
    sealed_b_path = task_dir / "sealed_panel_b.json"
    if (
        dict(status.get("fingerprints", {})) != dict(fingerprints)
        or dict(result.get("fingerprints", {})) != dict(fingerprints)
        or dict(latest.get("fingerprints", {})) != dict(fingerprints)
        or dict(best.get("fingerprints", {})) != dict(fingerprints)
        or status.get("task_result_sha256") != file_fingerprint(result_path)
        or not sealed_b_path.is_file()
        or status.get("sealed_panel_b_sha256") != file_fingerprint(sealed_b_path)
        or not all(path.is_file() for path in files.values())
        or latest.get("sha256") != file_fingerprint(files["latest"])
        or best.get("selected_sha256") != file_fingerprint(files["selected"])
        or best.get("nominee_sha256") != file_fingerprint(files["nominee"])
        or best.get("best_ema_sha256") != file_fingerprint(files["best_copy"])
        or best.get("nominee_ema_sha256") != file_fingerprint(files["nominee_copy"])
        or summary.get("selected_checkpoint_sha256")
        != file_fingerprint(files["best_copy"])
        or summary.get("nominee_checkpoint_sha256")
        != file_fingerprint(files["nominee_copy"])
        or summary.get("sealed_panel_b_sha256") != file_fingerprint(sealed_b_path)
        or int(status.get("training_step", -1))
        != int(summary.get("training_step", -2))
    ):
        raise ArtifactCompatibilityError("completed paired task hash chain mismatch")
    for name in ("latest", "selected", "nominee"):
        _load_checkpoint(
            files[name], device=device, task=task, fingerprints=fingerprints
        )
    return result


def _clip_fraction(
    history: Sequence[Mapping[str, Any]], *, start: int, stop: int
) -> float:
    rows = [
        value for value in history
        if int(start) <= int(value.get("step", -1)) <= int(stop)
    ]
    return float(np.mean([int(value.get("clipped", 0)) for value in rows])) if rows else 0.0


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"q00": math.nan, "q50": math.nan, "q90": math.nan, "q99": math.nan, "q100": math.nan}
    levels = np.quantile(finite, [0.0, 0.5, 0.9, 0.99, 1.0])
    return {
        name: float(value)
        for name, value in zip(("q00", "q50", "q90", "q99", "q100"), levels)
    }


def _history_diagnostics(
    history: Sequence[Mapping[str, Any]], *, train_steps: int, warmup_steps: int
) -> dict[str, Any]:
    end = int(train_steps)
    diagnostics = {
        "post_warmup_clip_fraction": _clip_fraction(
            history, start=int(warmup_steps) + 1, stop=end
        ),
        "final_500_clip_fraction": _clip_fraction(
            history, start=max(1, end - 499), stop=end
        ),
        "final_200_clip_fraction": _clip_fraction(
            history, start=max(1, end - 199), stop=end
        ),
        "raw_accumulated_gradient_norm_quantiles": _quantiles(
            [float(value.get("raw_accumulated_gradient_norm", math.nan)) for value in history]
        ),
        "scaled_preclip_gradient_norm_quantiles": _quantiles(
            [float(value.get("scaled_preclip_gradient_norm", math.nan)) for value in history]
        ),
        "optimizer_update_norm_quantiles": _quantiles(
            [float(value.get("optimizer_update_norm", math.nan)) for value in history]
        ),
        "ema_update_norm_quantiles": _quantiles(
            [float(value.get("ema_update_norm", math.nan)) for value in history]
        ),
    }
    windows = []
    for start in range(1, end + 1, 250):
        stop = min(end, start + 249)
        windows.append(
            {
                "start_step": start,
                "stop_step": stop,
                "clip_fraction": _clip_fraction(history, start=start, stop=stop),
            }
        )
    diagnostics["clipping_windows"] = windows
    return diagnostics


def _failed_task_result(
    task_dir: Path,
    *, task: str, model_seed: int, fingerprints: Mapping[str, Any], exc: BaseException
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
        "final_500_clip_fraction": 1.0,
        "final_200_clip_fraction": 1.0,
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
        "gate": _not_evaluated_gate("paired_ratio_task", str(exc)),
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


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(values)).detach().cpu())


def _batch_fingerprint(batch: PairedMixtureMicrobatch) -> str:
    fingerprint = getattr(batch, "fingerprint", None)
    if fingerprint is not None:
        return str(fingerprint)
    record = batch.record()
    return config_fingerprint(record)


def _microbatch_objective(
    model: nn.Module,
    batch: PairedMixtureMicrobatch,
    *, task: str,
) -> Tensor:
    objective, _ = _microbatch_objective_and_components(model, batch, task=task)
    return objective


def _microbatch_objective_and_components(
    model: nn.Module,
    batch: PairedMixtureMicrobatch,
    *, task: str,
) -> tuple[Tensor, dict[str, float]]:
    reference_logits = model(
        batch.tau, batch.reference_states, batch.labels
    ).reshape(-1)
    component_logits = model(
        batch.tau, batch.component_states, batch.labels
    ).reshape(-1)
    raw_components = weighted_paired_softplus_components(
        reference_logits,
        component_logits,
        task=task,
        teacher_epsilon=0.5,
    )
    objective = raw_components["total"].mean()
    components = {
        name: float(value.detach().mean().cpu())
        for name, value in raw_components.items()
    }
    return objective, components


def run_paired_density_ratio_task(
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
    accumulation_level: int,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
    fingerprints: Mapping[str, Any],
    phase: str,
    thresholds: RatioStabilityThresholds,
    show_progress: bool,
    interrupt_after_checkpoint_step: int | None = None,
    interrupt_during_accumulation: tuple[int, int] | None = None,
    interrupt_after_sealed_panel_b: bool = False,
) -> dict[str, Any]:
    """Train one paired-estimator task with exact committed-step resume."""

    if task not in {"bounded_teacher", "dirichlet_null"}:
        raise ValueError("unsupported paired density-ratio task")
    if set(selection_panels) != {"a", "b"}:
        raise ValueError("selection panels must be exactly a and b")
    if audit_panels is not None and set(audit_panels) != {"c", "d"}:
        raise ValueError("audit panels must be exactly c and d")
    if int(accumulation_level) not in {2, 4, 8}:
        raise ValueError("accumulation level must be 2, 4, or 8")
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
    rng = base._set_seed(int(model_seed))
    model = D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
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
            raise ArtifactCompatibilityError("paired latest pointer is invalid")
        payload = _load_checkpoint(
            checkpoint_path, device=device, task=task, fingerprints=fingerprints
        )
        orphan = checkpoints / f"finalized-step-{int(payload['step']):08d}.pt"
        if checkpoint_path != orphan and orphan.is_file():
            recovered = _load_checkpoint(
                orphan, device=device, task=task, fingerprints=fingerprints
            )
            b_records = sum(
                "b" in dict(value.get("panels", {}))
                for value in recovered.get("validation_records", [])
            )
            if (
                int(recovered.get("step", -1)) == int(payload["step"])
                and b_records == 1
                and dict(recovered.get("checkpoint_selection", {})).get("gate")
                == "density_ratio_checkpoint_selection"
            ):
                payload = recovered
                checkpoint_path = orphan
                atomic_write_json(
                    latest_path,
                    {
                        "schema": RUN_SCHEMA + "-latest",
                        "schema_version": 1,
                        "filename": orphan.name,
                        "sha256": file_fingerprint(orphan),
                        "step": int(payload["step"]),
                        "stream_cursor": int(payload["stream_cursor"]),
                        "accumulation_cursor": 0,
                        "fingerprints": dict(fingerprints),
                        "recovered_orphan_finalization": 1,
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    },
                )
        if dict(payload["stream_plan"]) != stream_plan_record(stream_plan):
            raise ArtifactCompatibilityError("checkpoint evaluation stream mismatch")
        if dict(payload["paired_stream_plan"]) != paired_mixture_stream_plan_record(
            paired_stream_plan
        ):
            raise ArtifactCompatibilityError("checkpoint paired stream mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        base._optimizer_to_device(optimizer, device)
        ema_state = {
            key: value.detach().clone().to(device)
            for key, value in dict(payload["ema_state_dict"]).items()
        }
        history = [dict(value) for value in payload["history"]]
        validations = [dict(value) for value in payload["validation_records"]]
        selection = dict(payload["checkpoint_selection"])
        completed = int(payload["step"])
        if int(payload["stream_cursor"]) != completed:
            raise ArtifactCompatibilityError("paired checkpoint cursor differs from step")
        restore_rng_state(payload["rng_state"], rng)

    def validate(step: int) -> dict[str, Any]:
        with temporary_ema_weights(model, ema_state):
            record, _ = base._classification_panel_record(
                model,
                selection_panels["a"],
                dynamics=dynamics,
                args=args,
                device=device,
                bootstrap_seed=base._derived_seed(
                    int(args.root_seed), phase, task, "selection", "a", step
                ),
                include_analytic_teacher=False,
            )
        return {
            "step": int(step),
            "finite": int(bool(int(record["finite"]))),
            "ema": 1,
            "panels": {"a": record},
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }

    def publish(step: int) -> None:
        nonlocal selection
        nomination = base.nominate_checkpoint_on_a(
            validations, thresholds.density_ratio
        )
        selection = {
            "gate": "density_ratio_checkpoint_selection_pending_panel_b",
            "evaluation_status": "not_evaluated",
            "passed": 0,
            "selected_step": 0,
            "nominee_step": nomination.get("nominee_step"),
            "nomination": nomination,
            "confirmation": _not_evaluated_gate(
                "density_ratio_panel_b_confirmation", "panel B remains sealed"
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
            step=step,
            history=history,
            validations=validations,
            selection=selection,
            stream_plan=stream_plan,
            paired_stream_plan=paired_stream_plan,
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
                "accumulation_cursor": 0,
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
        model.train()
        optimizer.zero_grad(set_to_none=True)
        microbatch_losses: list[float] = []
        microbatch_fingerprints: list[str] = []
        microbatch_components: list[dict[str, float]] = []
        for microbatch_index in range(int(accumulation_level)):
            batch = generate_paired_mixture_microbatch(
                paired_stream_plan,
                phase=phase,
                task=task,
                optimizer_step=int(step),
                microbatch_index=int(microbatch_index),
                device=device,
                dtype=torch.float32,
            )
            objective, component_record = _microbatch_objective_and_components(
                model, batch, task=task
            )
            if not bool(torch.isfinite(objective.detach())):
                raise FloatingPointError(
                    f"nonfinite paired loss for {task} at step {step}"
                )
            microbatch_losses.append(float(objective.detach().cpu()))
            microbatch_fingerprints.append(_batch_fingerprint(batch))
            microbatch_components.append(component_record)
            (
                objective
                * float(PARENT_LOSS_SCALE)
                / float(accumulation_level)
            ).backward()
            if interrupt_during_accumulation == (step, microbatch_index):
                # No optimizer or EMA update has occurred.  The only durable
                # cursor remains the preceding committed optimizer step.
                raise RuntimeError("injected interruption during accumulation")

        scaled_preclip = _gradient_norm(parameters)
        if not math.isfinite(scaled_preclip):
            raise FloatingPointError(
                f"nonfinite accumulated gradient for {task} at step {step}"
            )
        raw_accumulated = scaled_preclip / float(PARENT_LOSS_SCALE)
        if not math.isfinite(raw_accumulated):
            raise FloatingPointError(
                f"nonfinite raw accumulated gradient for {task} at step {step}"
            )
        clipped = int(scaled_preclip > float(args.grad_clip))
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(args.grad_clip), error_if_nonfinite=True
        )
        if not bool(torch.isfinite(clipped_norm)):
            raise FloatingPointError(
                f"nonfinite clipped gradient for {task} at step {step}"
            )
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.step()
        update_sq = torch.zeros((), device=device)
        for old, parameter in zip(before, parameters):
            update_sq = update_sq + (parameter.detach() - old).square().sum()
        if not bool(
            torch.isfinite(update_sq)
            and all(torch.isfinite(parameter.detach()).all() for parameter in parameters)
        ):
            raise FloatingPointError(
                f"nonfinite optimizer update for {task} at step {step}"
            )
        # EMA is updated from the post-AdamW parameters, so its diagnostic
        # norm must use that same post-step state.
        ema_update_sq = torch.zeros((), device=device)
        for name, parameter in model.named_parameters():
            if name in ema_state:
                delta = (1.0 - float(args.ema_decay)) * (
                    parameter.detach() - ema_state[name]
                )
                ema_update_sq = ema_update_sq + delta.square().sum()
        if not bool(torch.isfinite(ema_update_sq)):
            raise FloatingPointError(
                f"nonfinite EMA update for {task} at step {step}"
            )
        update_ema_state(ema_state, model, float(args.ema_decay))
        if not all(torch.isfinite(value).all() for value in ema_state.values()):
            raise FloatingPointError(
                f"nonfinite EMA state for {task} at step {step}"
            )
        history.append(
            {
                "step": int(step),
                "unscaled_loss": float(np.mean(microbatch_losses)),
                "scaled_loss": float(np.mean(microbatch_losses))
                * float(PARENT_LOSS_SCALE),
                "loss_scale": float(PARENT_LOSS_SCALE),
                "accumulation_level": int(accumulation_level),
                "effective_clusters": 32 * int(accumulation_level),
                "microbatch_losses": microbatch_losses,
                "microbatch_fingerprints": microbatch_fingerprints,
                **{
                    f"component_{name}": float(
                        np.mean([value[name] for value in microbatch_components])
                    )
                    for name in (
                        "base_positive", "mixture_positive",
                        "reference_negative", "total",
                    )
                },
                "raw_accumulated_gradient_norm": raw_accumulated,
                "raw_gradient_norm": raw_accumulated,
                "scaled_preclip_gradient_norm": scaled_preclip,
                "grad_norm": scaled_preclip,
                "clipped": clipped,
                "optimizer_update_norm": float(torch.sqrt(update_sq).detach().cpu()),
                "ema_update_norm": float(torch.sqrt(ema_update_sq).detach().cpu()),
                "optimizer_finite": 1,
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
        if step in validation_steps:
            validations.append(validate(step))
            publish(step)
            if interrupt_after_checkpoint_step == step:
                raise RuntimeError(
                    f"injected interruption after paired checkpoint {step}"
                )
        if show_progress and (step % 50 == 0 or step == total_steps):
            elapsed = time.perf_counter() - started
            done = max(1, step - completed)
            eta = elapsed / done * max(0, total_steps - step)
            print(
                f"{phase}/{task}/acc-{accumulation_level}: "
                f"step {step}/{total_steps} loss={history[-1]['scaled_loss']:.6g} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    final_step = int(history[-1]["step"]) if history else completed
    if not latest_path.is_file() or int(_json_load(latest_path).get("step", -1)) != final_step:
        if not any(int(value.get("step", -1)) == final_step for value in validations):
            validations.append(validate(final_step))
        publish(final_step)

    nomination = base.nominate_checkpoint_on_a(validations, thresholds.density_ratio)
    nominee_step = int(nomination.get("nominee_step") or 0)
    nominee_path = checkpoints / f"step-{nominee_step:08d}.pt"
    if not nominee_path.is_file():
        raise ArtifactCompatibilityError("panel-A nominee checkpoint is missing")
    nominee_rows = [
        value for value in validations if int(value.get("step", -1)) == nominee_step
    ]
    if len(nominee_rows) != 1:
        raise ArtifactCompatibilityError("panel-A nominee is ambiguous")
    nominee_validation = nominee_rows[0]
    sealed_b_path = task_dir / "sealed_panel_b.json"
    sealed_binding = {
        "schema": RUN_SCHEMA + "-sealed-panel-b",
        "schema_version": 1,
        "task": task,
        "phase": phase,
        "nominee_step": nominee_step,
        "nominee_checkpoint_sha256": file_fingerprint(nominee_path),
        "panel_identity": panel_identity(selection_panels["b"]),
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    if sealed_b_path.is_file():
        sealed = _json_load(sealed_b_path)
        for key, expected in sealed_binding.items():
            if sealed.get(key) != expected:
                raise ArtifactCompatibilityError(
                    f"sealed panel-B binding mismatch for {key}"
                )
        panel_b = sealed.get("panel_record")
        if not isinstance(panel_b, Mapping):
            raise ArtifactCompatibilityError("sealed panel-B record is missing")
        if "b" in dict(nominee_validation.get("panels", {})):
            if dict(nominee_validation["panels"]["b"]) != dict(panel_b):
                raise ArtifactCompatibilityError(
                    "finalized checkpoint and sealed panel B disagree"
                )
        else:
            nominee_validation.setdefault("panels", {})["b"] = dict(panel_b)
            nominee_validation["finite"] = int(
                bool(int(nominee_validation.get("finite", 0)))
                and bool(int(dict(panel_b).get("finite", 0)))
            )
    elif "b" not in dict(nominee_validation.get("panels", {})):
        nominee_payload = _load_checkpoint(
            nominee_path, device=device, task=task, fingerprints=fingerprints
        )
        model.load_state_dict(nominee_payload["ema_state_dict"], strict=True)
        model.eval()
        panel_b, _ = base._classification_panel_record(
            model,
            selection_panels["b"],
            dynamics=dynamics,
            args=args,
            device=device,
            bootstrap_seed=base._derived_seed(
                int(args.root_seed), phase, task, "selection", "b", nominee_step
            ),
            include_analytic_teacher=task == "bounded_teacher",
        )
        atomic_write_json(
            sealed_b_path,
            {
                **sealed_binding,
                "panel_record": panel_b,
            },
        )
        if interrupt_after_sealed_panel_b:
            raise RuntimeError("injected interruption after sealed panel B")
        nominee_validation.setdefault("panels", {})["b"] = panel_b
        nominee_validation["finite"] = int(
            bool(int(nominee_validation.get("finite", 0)))
            and bool(int(panel_b.get("finite", 0)))
        )
    else:
        raise ArtifactCompatibilityError(
            "checkpoint contains panel B without its sealed evidence artifact"
        )
    selection = base.select_density_ratio_checkpoint(
        validations, thresholds.density_ratio
    )
    selected_step = int(selection.get("selected_step", 0))
    selected_path = checkpoints / f"step-{selected_step:08d}.pt"
    if not selected_path.is_file():
        raise ArtifactCompatibilityError("selected checkpoint is missing")

    final_original = checkpoints / f"step-{final_step:08d}.pt"
    finalized = checkpoints / f"finalized-step-{final_step:08d}.pt"
    final_payload = _load_checkpoint(
        final_original, device=device, task=task, fingerprints=fingerprints
    )
    final_payload["validation_records"] = copy.deepcopy(validations)
    final_payload["checkpoint_selection"] = copy.deepcopy(selection)
    atomic_torch_save(finalized, final_payload)
    atomic_write_json(
        latest_path,
        {
            "schema": RUN_SCHEMA + "-latest",
            "schema_version": 1,
            "filename": finalized.name,
            "sha256": file_fingerprint(finalized),
            "step": final_step,
            "stream_cursor": final_step,
            "accumulation_cursor": 0,
            "fingerprints": dict(fingerprints),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    best_copy = checkpoints / "best_ema.pt"
    nominee_copy = checkpoints / "nominee_ema.pt"
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
    audited = _load_checkpoint(
        audit_path, device=device, task=task, fingerprints=fingerprints
    )
    model.load_state_dict(audited["ema_state_dict"], strict=True)
    model.eval()
    diagnostics = _history_diagnostics(
        history,
        train_steps=total_steps,
        warmup_steps=int(args.clip_warmup_steps),
    )
    boundary = base._boundary_certificate(model)
    metrics: dict[str, Any] = {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": int(
            all(
                bool(int(value.get("optimizer_finite", 0)))
                and all(
                    math.isfinite(float(value.get(name, math.nan)))
                    for name in (
                        "scaled_loss", "raw_accumulated_gradient_norm",
                        "scaled_preclip_gradient_norm", "optimizer_update_norm",
                        "ema_update_norm",
                    )
                )
                for value in history
            )
        ),
        "model_seed": int(model_seed),
        "selected_step": selected_step,
        "nominee_step": nominee_step,
        "audit_step": audit_step,
        "selection": selection,
        "checkpoints": validations,
        "boundary_admissible": int(boundary["passed"]),
        "optimization_diagnostics": diagnostics,
        "post_warmup_clip_fraction": diagnostics["post_warmup_clip_fraction"],
        "final_500_clip_fraction": diagnostics["final_500_clip_fraction"],
        "final_200_clip_fraction": diagnostics["final_200_clip_fraction"],
        "audit_panels": {},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    panel_b_record = dict(dict(nominee_validation.get("panels", {})).get("b", {}))
    if task == "bounded_teacher" and isinstance(panel_b_record.get("analytic"), Mapping):
        analytic = dict(panel_b_record["analytic"])
        metrics["selected_analytic_metrics"] = analytic
        for source, target in (
            ("audit_overall_score_gain", "selection_overall_score_gain"),
            ("audit_data_end_score_gain", "selection_data_end_score_gain"),
            ("overall_flux_cosine", "selection_overall_flux_cosine"),
            ("overall_relative_flux_l2", "selection_overall_relative_flux_l2"),
        ):
            metrics[target] = analytic.get(source)
        cosines = list(analytic.get("time_bin_flux_cosines", []))
        relatives = list(analytic.get("time_bin_relative_flux_l2", []))
        metrics["selection_data_end_flux_cosine"] = cosines[-1] if cosines else None
        metrics["selection_data_end_relative_flux_l2"] = relatives[-1] if relatives else None
    if task == "dirichlet_null":
        metrics["comparator"] = "analytic_zero"

    audit_rows: list[dict[str, Any]] = []
    if audit_panels is not None:
        for name in ("c", "d"):
            record, rows = base._classification_panel_record(
                model,
                audit_panels[name],
                dynamics=dynamics,
                args=args,
                device=device,
                bootstrap_seed=base._derived_seed(
                    int(args.root_seed), phase, task, "audit", name, audit_step
                ),
                include_analytic_teacher=task == "bounded_teacher",
            )
            metrics["audit_panels"][name] = record
            audit_rows.extend(rows)

    gate = (
        evaluate_teacher_seed(metrics, thresholds)
        if task == "bounded_teacher" and audit_panels is not None
        else evaluate_null_seed(metrics, thresholds)
        if task == "dirichlet_null" and audit_panels is not None
        else {
            "gate": "paired_ratio_pilot_task",
            "evaluation_status": "evaluated",
            "passed": int(bool(metrics["complete"]) and bool(metrics["finite"])),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    )
    summary = {
        "complete": 1,
        "finite": metrics["finite"],
        "task": task,
        "model_seed": int(model_seed),
        "learning_rate": float(learning_rate),
        "accumulation_level": int(accumulation_level),
        "selected_step": selected_step,
        "nominee_step": nominee_step,
        "audit_step": audit_step,
        "training_step": final_step,
        "target_training_steps": total_steps,
        "checkpoint_selection": selection,
        "optimization_diagnostics": diagnostics,
        "boundary_admissibility_certificate": boundary,
        "selected_checkpoint_path": str(best_copy.resolve()),
        "selected_checkpoint_sha256": file_fingerprint(best_copy),
        "nominee_checkpoint_path": str(nominee_copy.resolve()),
        "nominee_checkpoint_sha256": file_fingerprint(nominee_copy),
        "sealed_panel_b_sha256": file_fingerprint(sealed_b_path),
        "stream_plan": stream_plan_record(stream_plan),
        "paired_stream_plan": paired_mixture_stream_plan_record(paired_stream_plan),
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_csv(task_dir / "training_history.csv", history)
    checkpoint_rows = []
    for validation in validations:
        for panel_name, panel_record in dict(validation.get("panels", {})).items():
            checkpoint_rows.append(
                {
                    "step": validation["step"],
                    "panel": panel_name,
                    "bce_overall": dict(panel_record["overall"])["bce"],
                    "lower_bound_overall": dict(panel_record["overall"])["lower_bound"],
                    "bce_data_end": dict(panel_record["data_end"])["bce"],
                    "lower_bound_data_end": dict(panel_record["data_end"])["lower_bound"],
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
    atomic_write_csv(task_dir / "checkpoint_metrics.csv", checkpoint_rows)
    atomic_write_csv(
        task_dir / "clipping_windows.csv", diagnostics["clipping_windows"]
    )
    selection_time_rows: list[dict[str, Any]] = []
    for validation in validations:
        for panel_name, panel_record in dict(validation.get("panels", {})).items():
            classification_bins = list(
                dict(panel_record.get("classification_metrics", {})).get(
                    "time_bins", []
                )
            )
            analytic_bins = list(
                dict(panel_record.get("analytic", {})).get("time_bins", [])
            )
            for index, classification_bin in enumerate(classification_bins):
                analytic_bin = (
                    dict(analytic_bins[index]) if index < len(analytic_bins) else {}
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
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
    if selection_time_rows:
        atomic_write_csv(
            task_dir / "selection_time_bin_metrics.csv", selection_time_rows
        )
    if audit_rows:
        atomic_write_csv(task_dir / "audit_path_risks.csv", audit_rows)
    audit_time_rows: list[dict[str, Any]] = []
    for panel_name, panel_record in metrics["audit_panels"].items():
        classification_bins = list(
            dict(panel_record.get("classification_metrics", {})).get("time_bins", [])
        )
        analytic_bins = list(
            dict(panel_record.get("analytic", {})).get("time_bins", [])
        )
        for index, classification_bin in enumerate(classification_bins):
            analytic_bin = (
                dict(analytic_bins[index]) if index < len(analytic_bins) else {}
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
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
    if audit_time_rows:
        atomic_write_csv(task_dir / "audit_time_bin_metrics.csv", audit_time_rows)
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
            "sealed_panel_b_sha256": file_fingerprint(sealed_b_path),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return result


def _bootstrap_contains_zero(
    values: Sequence[float], *, confidence: float, reps: int, seed: int
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    interval = base.bootstrap_path_mean_interval(
        array,
        np.arange(array.size, dtype=np.int64),
        reps=int(reps),
        confidence=float(confidence),
        seed=int(seed),
    )
    lower = float(interval["lower_bound"])
    upper = float(interval["upper_bound"])
    return {
        **interval,
        "contains_zero": int(lower <= 0.0 <= upper),
        "mean": float(array.mean()) if array.size else math.nan,
    }


def _scaled_loss_derivative(
    reference_logits: Tensor,
    component_logits: Tensor,
    *, task: str,
) -> float:
    scale = torch.ones((), device=reference_logits.device, requires_grad=True)
    loss = weighted_paired_softplus_loss(
        scale * reference_logits,
        scale * component_logits,
        task=task,
        teacher_epsilon=0.5,
        reduction="mean",
    )
    derivative = torch.autograd.grad(loss, scale)[0]
    return float(derivative.detach().cpu())


def _iid_scaled_loss_derivative(logits: Tensor, targets: Tensor) -> float:
    scale = torch.ones((), device=logits.device, requires_grad=True)
    loss = base.classification_loss(scale * logits, targets)
    derivative = torch.autograd.grad(loss, scale)[0]
    return float(derivative.detach().cpu())


def _load_parent_forensic_model(
    parent_run_dir: Path,
    *,
    task: str,
    candidate_index: int,
    dynamics: DirectFluxMNISTConfig,
    base_channels: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    task_dir = parent_run_dir / "pilot" / f"lr-{candidate_index:02d}" / task
    pointer_path = task_dir / "checkpoints" / "best.json"
    if not pointer_path.is_file():
        raise ArtifactCompatibilityError("parent forensic checkpoint pointer is missing")
    pointer = _json_load(pointer_path)
    filename = str(pointer.get("nominee_ema_filename", ""))
    checkpoint_path = task_dir / "checkpoints" / filename
    if (
        Path(filename).name != filename
        or not checkpoint_path.is_file()
        or pointer.get("nominee_ema_sha256") != file_fingerprint(checkpoint_path)
    ):
        raise ArtifactCompatibilityError("parent forensic checkpoint hash mismatch")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, Mapping) or "ema_state_dict" not in payload:
        raise ArtifactCompatibilityError("parent forensic checkpoint payload is invalid")
    model = D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(base_channels)
    ).to(device)
    model.load_state_dict(payload["ema_state_dict"], strict=True)
    model.eval()
    return model, {
        "task": task,
        "candidate_index": int(candidate_index),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_fingerprint(checkpoint_path),
        "nominee_step": pointer.get("nominee_step"),
    }


def _run_variance_forensics(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    simultaneous_loss = True
    simultaneous_gradient = True
    forensic_panels: dict[tuple[int, str], dict[str, list[str]]] = {}
    # The two learning rates that independently exposed teacher signal in the
    # parent are its candidate indices 2 and 3.  Their teacher and paired-null
    # nominees are advisory fixed functions here; no checkpoint is reused for
    # initialization or selection.
    for candidate_index in (2, 3):
        for task in ("bounded_teacher", "dirichlet_null"):
            model, source = _load_parent_forensic_model(
                args.parent_density_ratio_run_dir,
                task=task,
                candidate_index=candidate_index,
                dynamics=dynamics,
                base_channels=int(args.base_channels),
                device=device,
            )
            loss_differences: list[float] = []
            gradient_differences: list[float] = []
            paired_losses: list[float] = []
            iid_losses: list[float] = []
            paired_gradients: list[float] = []
            iid_gradients: list[float] = []
            panel_paired_fingerprints: list[str] = []
            panel_iid_fingerprints: list[str] = []
            for path_index in range(int(args.preflight_paths)):
                paired = generate_paired_mixture_microbatch(
                    paired_stream_plan,
                    phase="preflight-forensic",
                    task=task,
                    optimizer_step=path_index + 1,
                    microbatch_index=0,
                    device=device,
                    dtype=torch.float32,
                )
                iid = base.generate_density_ratio_batch(
                    stream_plan,
                    phase="preflight-forensic-iid",
                    task=task,
                    step=path_index + 1,
                    device=device,
                    dtype=torch.float32,
                )
                paired_fingerprint = _batch_fingerprint(paired)
                iid_fingerprint = str(iid.fingerprint)
                panel_paired_fingerprints.append(paired_fingerprint)
                panel_iid_fingerprints.append(iid_fingerprint)
                with torch.no_grad():
                    reference_logits = model(
                        paired.tau, paired.reference_states, paired.labels
                    ).reshape(-1)
                    component_logits = model(
                        paired.tau, paired.component_states, paired.labels
                    ).reshape(-1)
                    iid_logits = model(iid.tau, iid.states, iid.labels).reshape(-1)
                paired_loss = float(
                    weighted_paired_softplus_loss(
                        reference_logits,
                        component_logits,
                        task=task,
                        teacher_epsilon=0.5,
                        reduction="mean",
                    ).detach().cpu()
                )
                iid_loss = float(
                    base.classification_loss(iid_logits, iid.class_targets)
                    .detach().cpu()
                )
                paired_gradient = _scaled_loss_derivative(
                    reference_logits, component_logits, task=task
                )
                iid_gradient = _iid_scaled_loss_derivative(
                    iid_logits, iid.class_targets
                )
                paired_losses.append(paired_loss)
                iid_losses.append(iid_loss)
                paired_gradients.append(paired_gradient)
                iid_gradients.append(iid_gradient)
                loss_differences.append(paired_loss - iid_loss)
                gradient_differences.append(paired_gradient - iid_gradient)

            loss_interval = _bootstrap_contains_zero(
                loss_differences,
                confidence=float(args.preflight_confidence),
                reps=int(args.bootstrap_reps),
                seed=base._derived_seed(
                    int(args.root_seed), "forensic", candidate_index, task, "loss"
                ),
            )
            gradient_interval = _bootstrap_contains_zero(
                gradient_differences,
                confidence=float(args.preflight_confidence),
                reps=int(args.bootstrap_reps),
                seed=base._derived_seed(
                    int(args.root_seed), "forensic", candidate_index, task, "gradient"
                ),
            )
            simultaneous_loss &= bool(int(loss_interval["contains_zero"]))
            simultaneous_gradient &= bool(int(gradient_interval["contains_zero"]))
            comparisons.append(
                {
                    **source,
                    "loss_difference_interval": loss_interval,
                    "directional_gradient_difference_interval": gradient_interval,
                    "paired_loss_variance": float(np.var(paired_losses, ddof=1)),
                    "iid_loss_variance": float(np.var(iid_losses, ddof=1)),
                    "paired_gradient_variance": float(np.var(paired_gradients, ddof=1)),
                    "iid_gradient_variance": float(np.var(iid_gradients, ddof=1)),
                    "loss_variance_ratio": float(
                        np.var(paired_losses, ddof=1)
                        / max(np.var(iid_losses, ddof=1), 1e-30)
                    ),
                    "gradient_variance_ratio": float(
                        np.var(paired_gradients, ddof=1)
                        / max(np.var(iid_gradients, ddof=1), 1e-30)
                    ),
                    "loss_absolute_q99_ratio": float(
                        np.quantile(np.abs(paired_losses), 0.99)
                        / max(np.quantile(np.abs(iid_losses), 0.99), 1e-30)
                    ),
                    "gradient_absolute_q99_ratio": float(
                        np.quantile(np.abs(paired_gradients), 0.99)
                        / max(np.quantile(np.abs(iid_gradients), 0.99), 1e-30)
                    ),
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
            forensic_panels[(candidate_index, task)] = {
                "paired": panel_paired_fingerprints,
                "iid": panel_iid_fingerprints,
            }
            del model
    within_panel_unique = all(
        len(values["paired"]) == len(set(values["paired"]))
        and len(values["iid"]) == len(set(values["iid"]))
        and set(values["paired"]).isdisjoint(values["iid"])
        for values in forensic_panels.values()
    )
    common_across_checkpoints = all(
        forensic_panels[(2, task)] == forensic_panels[(3, task)]
        for task in ("bounded_teacher", "dirichlet_null")
    )
    teacher_fingerprints = set(
        forensic_panels[(2, "bounded_teacher")]["paired"]
        + forensic_panels[(2, "bounded_teacher")]["iid"]
    )
    null_fingerprints = set(
        forensic_panels[(2, "dirichlet_null")]["paired"]
        + forensic_panels[(2, "dirichlet_null")]["iid"]
    )
    law_namespaces_disjoint = teacher_fingerprints.isdisjoint(null_fingerprints)
    result = {
        "schema": RUN_SCHEMA + "-variance-forensics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "path_count": int(args.preflight_paths),
        "confidence": float(args.preflight_confidence),
        "comparisons": comparisons,
        "simultaneous_loss_interval_contains_zero": int(simultaneous_loss),
        "simultaneous_directional_gradient_intervals_contain_zero": int(
            simultaneous_gradient
        ),
        # Honest names for the actual contract: every individual 99% interval
        # must contain zero.  The simultaneous_* aliases remain only because
        # gate schema v1 uses those historical field names.
        "all_individual_99pct_loss_intervals_contain_zero": int(
            simultaneous_loss
        ),
        "all_individual_99pct_directional_gradient_intervals_contain_zero": int(
            simultaneous_gradient
        ),
        "familywise_99pct_claimed": 0,
        "stream_fingerprint_isolation_pass": int(
            within_panel_unique and law_namespaces_disjoint
        ),
        "common_forensic_panels_across_checkpoints_pass": int(
            common_across_checkpoints
        ),
        "forensic_law_namespaces_disjoint": int(law_namespaces_disjoint),
        "variance_ratios_advisory_only": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(run_dir / "paired_ratio_variance_forensics.json", result)
    return result


def _run_preflight(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
    thresholds: RatioStabilityThresholds,
) -> dict[str, Any]:
    replay_records = []
    replay_pass = True
    for phase in ("pilot", "confirmation"):
        for task in ("bounded_teacher", "dirichlet_null"):
            for step, accumulation_level in ((1, 2), (17, 4), (31, 8)):
                record = paired_mixture_replay_record(
                    paired_stream_plan,
                    phase=phase,
                    task=task,
                    optimizer_step=step,
                    accumulation_level=accumulation_level,
                )
                verification = verify_paired_mixture_replay(
                    paired_stream_plan, record
                )
                replay_pass &= bool(int(verification.get("passed", 0)))
                replay_records.append({"record": record, "verification": verification})
    replay = {
        "schema": RUN_SCHEMA + "-stream-replay",
        "schema_version": 1,
        "passed": int(replay_pass),
        "records": replay_records,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(run_dir / "paired_ratio_stream_replay.json", replay)

    teacher = generate_paired_mixture_microbatch(
        paired_stream_plan,
        phase="preflight-exact",
        task="bounded_teacher",
        optimizer_step=1,
        microbatch_index=0,
        device="cpu",
        dtype=torch.float64,
    )
    null = generate_paired_mixture_microbatch(
        paired_stream_plan,
        phase="preflight-exact",
        task="dirichlet_null",
        optimizer_step=1,
        microbatch_index=0,
        device="cpu",
        dtype=torch.float64,
    )
    reference_logits = torch.linspace(-2.0, 2.0, 32, dtype=torch.float64)
    component_logits = torch.linspace(1.5, -1.5, 32, dtype=torch.float64)
    paired_loss = weighted_paired_softplus_loss(
        reference_logits,
        component_logits,
        task="bounded_teacher",
        teacher_epsilon=0.5,
        reduction="mean",
    )
    manual_loss = (
        0.25 * torch.nn.functional.softplus(-reference_logits)
        + 0.25 * torch.nn.functional.softplus(-component_logits)
        + 0.50 * torch.nn.functional.softplus(reference_logits)
    ).mean()
    loss_algebra_error = float((paired_loss - manual_loss).abs())

    expanded_logits = torch.cat(
        (reference_logits, component_logits, reference_logits), dim=0
    )
    expanded_targets = torch.cat(
        (
            torch.ones_like(reference_logits),
            torch.ones_like(component_logits),
            torch.zeros_like(reference_logits),
        )
    )
    expanded_weights = torch.cat(
        (
            torch.full_like(reference_logits, 0.25 / 32.0),
            torch.full_like(component_logits, 0.25 / 32.0),
            torch.full_like(reference_logits, 0.50 / 32.0),
        )
    )
    expanded_loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            expanded_logits, expanded_targets, reduction="none"
        )
        * expanded_weights
    ).sum()
    expanded_loss_error = float((paired_loss - expanded_loss).abs())
    scalar = torch.ones((), dtype=torch.float64, requires_grad=True)
    direct_directional = torch.autograd.grad(
        weighted_paired_softplus_loss(
            scalar * reference_logits,
            scalar * component_logits,
            task="bounded_teacher",
            teacher_epsilon=0.5,
            reduction="mean",
        ),
        scalar,
    )[0]
    scalar_expanded = torch.ones((), dtype=torch.float64, requires_grad=True)
    expanded_directional = torch.autograd.grad(
        (
            torch.nn.functional.binary_cross_entropy_with_logits(
                scalar_expanded * expanded_logits,
                expanded_targets,
                reduction="none",
            )
            * expanded_weights
        ).sum(),
        scalar_expanded,
    )[0]
    expanded_gradient_error = float(
        (direct_directional - expanded_directional).abs()
    )

    # Exercise the actual paired objective and production level eight.  The
    # tiny deterministic logit model isolates accumulation arithmetic from the
    # U-Net while retaining the exact teacher/null weighted softplus losses.
    accumulation_records: list[dict[str, Any]] = []
    accumulation_errors: list[float] = []
    for accumulation_task in ("bounded_teacher", "dirichlet_null"):
        accumulated_stream = generate_accumulated_paired_stream(
            paired_stream_plan,
            phase="preflight-accumulation",
            task=accumulation_task,
            optimizer_step=7,
            accumulation_level=8,
            device="cpu",
            dtype=torch.float64,
        )

        def objectives_for(parameter: Tensor) -> list[tuple[Tensor, int]]:
            values: list[tuple[Tensor, int]] = []
            for batch in accumulated_stream.canonical_microbatches:
                reference = parameter * (
                    batch.reference_states[:, 0]
                    + 0.125 * batch.tau_fraction
                )
                component = parameter * (
                    batch.component_states[:, 0]
                    + 0.125 * batch.tau_fraction
                )
                values.append(
                    (
                        weighted_paired_softplus_loss(
                            reference,
                            component,
                            task=accumulation_task,
                            teacher_epsilon=0.5,
                            reduction="mean",
                        ),
                        batch.clusters,
                    )
                )
            return values

        concatenated_parameter = torch.tensor(
            0.37, dtype=torch.float64, requires_grad=True
        )
        concatenated_objectives = objectives_for(concatenated_parameter)
        torch.stack([value for value, _ in concatenated_objectives]).mean().backward()
        concatenated_gradient = concatenated_parameter.grad.detach().clone()

        accumulated_parameter = torch.tensor(
            0.37, dtype=torch.float64, requires_grad=True
        )
        diagnostics = backward_accumulated_objective(
            objectives_for(accumulated_parameter),
            (accumulated_parameter,),
            expected_microbatches=8,
            expected_clusters=256,
            loss_scale=1.0,
        )
        error = float(
            (concatenated_gradient - accumulated_parameter.grad).abs()
        )
        accumulation_errors.append(error)
        accumulation_records.append(
            {
                "task": accumulation_task,
                "accumulation_level": 8,
                "effective_clusters": accumulated_stream.effective_clusters,
                "gradient_max_error": error,
                "diagnostics": diagnostics.to_record(),
                "stream_fingerprint": accumulated_stream.fingerprint,
            }
        )
    accumulation_error = max(accumulation_errors)

    teacher_record = teacher.record()
    null_record = null.record()
    records = (teacher_record, null_record)
    strata_pass = all(
        list(record.get("time_bin_counts", []))
        == [4, 4, 4, 4, 16]
        for record in records
    )
    balance_pass = all(int(record.get("clusters", record.get("rows", 0))) == 32 for record in records)
    simplex_pass = all(
        bool(torch.isfinite(states).all())
        and bool((states > 0).all())
        and float((states.sum(1) - 1.0).abs().max()) <= 2e-12
        for batch in (teacher, null)
        for states in (batch.reference_states, batch.component_states)
    )
    anchor_indices = bounded_teacher_anchor_indices(
        int(args.grid_size), device=teacher.reference_states.device
    )[teacher.component_indices]
    reconstructed_gamma = (
        teacher.reference_states * teacher.base_gamma_sums[:, None]
    ).clone()
    reconstructed_gamma[
        torch.arange(teacher.clusters), anchor_indices
    ] += teacher.tilt_increments
    reconstructed_teacher = reconstructed_gamma / (
        teacher.base_gamma_sums + teacher.tilt_increments
    )[:, None]
    reconstruction_error = float(
        (reconstructed_teacher - teacher.component_states).abs().max()
    )
    teacher_seed_keys = set(teacher_record.get("seeds", {}))
    null_seed_keys = set(null_record.get("seeds", {}))
    expected_teacher_seeds = {
        "cluster-permutation", "common-base-gamma",
        "tilt-increment", "mixture-choice",
    }
    expected_null_seeds = {
        "cluster-permutation", "null-pool", "null-swaps",
    }
    teacher_seed_values = set(dict(teacher_record.get("seeds", {})).values())
    null_seed_values = set(dict(null_record.get("seeds", {})).values())
    seed_namespace_pass = (
        teacher_seed_keys == expected_teacher_seeds
        and null_seed_keys == expected_null_seeds
        and len(teacher_seed_values) == len(expected_teacher_seeds)
        and len(null_seed_values) == len(expected_null_seeds)
        and teacher_seed_values.isdisjoint(null_seed_values)
    )
    null_structure_pass = bool(
        int(null_record.get("null_pooled_stateless_swaps", 0)) == 1
        and torch.all(null.component_indices == -1)
        and torch.all((null.swap_bits == 0) | (null.swap_bits == 1))
        and torch.all(null.base_gamma_sums == 0.0)
        and torch.all(null.tilt_increments == 0.0)
        and dict(null_record.get("seeds", {})).get("null-pool")
        != dict(null_record.get("seeds", {})).get("null-swaps")
    )
    common_gamma_pass = bool(
        int(teacher_record.get("common_gamma_teacher", 0)) == 1
        and reconstruction_error <= 2e-12
        and torch.all(teacher.base_gamma_sums > 0.0)
        and torch.all(teacher.tilt_increments > 0.0)
    )
    structural_law_certificate = {
        "schema": RUN_SCHEMA + "-structural-law-certificate",
        "schema_version": 1,
        "simplex_positive_finite_pass": int(simplex_pass),
        "teacher_common_gamma_reconstruction_max_error": reconstruction_error,
        "teacher_common_gamma_pass": int(common_gamma_pass),
        "exact_seed_namespaces_pass": int(seed_namespace_pass),
        "null_pool_swap_structure_pass": int(null_structure_pass),
        "teacher_seed_keys": sorted(teacher_seed_keys),
        "null_seed_keys": sorted(null_seed_keys),
        "dirichlet_marginal_construction_pass": int(
            simplex_pass
            and common_gamma_pass
            and null_structure_pass
            and seed_namespace_pass
        ),
        "stochastic_moment_thresholds_used": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(
        run_dir / "paired_ratio_structural_law_certificate.json",
        structural_law_certificate,
    )

    order_a = generate_accumulated_paired_stream(
        paired_stream_plan,
        phase="preflight-order",
        task="bounded_teacher",
        optimizer_step=3,
        accumulation_level=4,
        microbatch_order=(0, 1, 2, 3),
        device="cpu",
        dtype=torch.float32,
    )
    order_b = generate_accumulated_paired_stream(
        paired_stream_plan,
        phase="preflight-order",
        task="bounded_teacher",
        optimizer_step=3,
        accumulation_level=4,
        microbatch_order=(3, 2, 1, 0),
        device="cpu",
        dtype=torch.float32,
    )
    fingerprints_a = sorted(_batch_fingerprint(value) for value in order_a.microbatches)
    fingerprints_b = sorted(_batch_fingerprint(value) for value in order_b.microbatches)
    order_pass = fingerprints_a == fingerprints_b
    nested_streams = {
        level: generate_accumulated_paired_stream(
            paired_stream_plan,
            phase="pilot",
            task="bounded_teacher",
            optimizer_step=19,
            accumulation_level=level,
            device="cpu",
            dtype=torch.float32,
        )
        for level in (2, 4, 8)
    }
    nested_fingerprints = {
        level: [value.fingerprint for value in stream.canonical_microbatches]
        for level, stream in nested_streams.items()
    }
    nested_pass = (
        nested_fingerprints[4][:2] == nested_fingerprints[2]
        and nested_fingerprints[8][:4] == nested_fingerprints[4]
    )

    forensic = _run_variance_forensics(
        run_dir,
        args=args,
        dynamics=dynamics,
        device=device,
        stream_plan=stream_plan,
        paired_stream_plan=paired_stream_plan,
    )
    boundary = base._boundary_certificate(
        D0BoundarySmoothPotentialUNet(dynamics, base_channels=int(args.base_channels))
    )
    base._set_seed(base._derived_seed(int(args.root_seed), "preflight", "device"))
    smoke_model = D0BoundarySmoothPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    smoke_batch = generate_paired_mixture_microbatch(
        paired_stream_plan,
        phase="preflight-device",
        task="bounded_teacher",
        optimizer_step=1,
        microbatch_index=0,
        device=device,
        dtype=torch.float32,
    )
    smoke_model.zero_grad(set_to_none=True)
    smoke_loss = _microbatch_objective(
        smoke_model, smoke_batch, task="bounded_teacher"
    )
    smoke_loss.backward()
    smoke_gradients = [
        parameter.grad for parameter in smoke_model.parameters()
        if parameter.grad is not None
    ]
    device_pass = bool(
        torch.isfinite(smoke_loss.detach())
        and smoke_gradients
        and all(bool(torch.isfinite(value).all()) for value in smoke_gradients)
    )
    del smoke_model, smoke_batch, smoke_loss, smoke_gradients
    if device.type == "cuda":
        torch.cuda.empty_cache()

    parent_boundary = bool(int(provenance.get("passed", 0)))
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": int(
            all(
                math.isfinite(value)
                for value in (
                    loss_algebra_error,
                    expanded_loss_error,
                    expanded_gradient_error,
                    accumulation_error,
                )
            )
        ),
        "preflight_paths": int(args.preflight_paths),
        "preflight_confidence": float(args.preflight_confidence),
        "loss_algebra_max_error": loss_algebra_error,
        "expanded_loss_max_error": expanded_loss_error,
        "expanded_gradient_max_error": expanded_gradient_error,
        "accumulation_gradient_max_error": accumulation_error,
        "parent_provenance_pass": int(bool(int(provenance.get("passed", 0)))),
        "mixture_coefficients_pass": int(loss_algebra_error <= 1e-12),
        "dirichlet_marginals_pass": structural_law_certificate[
            "dirichlet_marginal_construction_pass"
        ],
        "common_gamma_coupling_pass": int(common_gamma_pass),
        "exact_seed_namespaces_pass": int(seed_namespace_pass),
        "null_pool_swap_structure_pass": int(null_structure_pass),
        "time_strata_pass": int(strata_pass),
        "class_balance_pass": int(balance_pass),
        "stream_replay_pass": int(replay_pass),
        "candidate_order_invariance_pass": int(order_pass and nested_pass),
        "nested_accumulation_prefix_pass": int(nested_pass),
        "nested_accumulation_fingerprints": nested_fingerprints,
        "actual_paired_accumulation_records": accumulation_records,
        "fresh_panel_isolation_pass": int(
            bool(int(forensic["stream_fingerprint_isolation_pass"]))
            and bool(
                int(forensic["common_forensic_panels_across_checkpoints_pass"])
            )
        ),
        "common_forensic_panels_across_checkpoints_pass": forensic[
            "common_forensic_panels_across_checkpoints_pass"
        ],
        "simultaneous_loss_interval_contains_zero": forensic[
            "simultaneous_loss_interval_contains_zero"
        ],
        "simultaneous_directional_gradient_intervals_contain_zero": forensic[
            "simultaneous_directional_gradient_intervals_contain_zero"
        ],
        "boundary_operator_pass": int(bool(boundary["passed"]) and parent_boundary),
        "device_smoke_pass": int(device_pass),
        "parent_loss_scale_reused": int(
            math.isclose(float(args.loss_scale), float(PARENT_LOSS_SCALE), rel_tol=0.0, abs_tol=0.0)
        ),
        "adaptive_loss_scaling": 0,
        "paired_stream_plan": paired_mixture_stream_plan_record(paired_stream_plan),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    gate = evaluate_paired_ratio_preflight(metrics, thresholds)
    atomic_write_json(run_dir / "paired_ratio_preflight.json", metrics)
    atomic_write_json(run_dir / "paired_ratio_preflight_gate.json", gate)
    return gate


def _freeze_selected_profile(
    run_dir: Path, selected: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(selected), sort_keys=True, allow_nan=False))
    path = run_dir / "selected_paired_ratio_profile.json"
    if path.is_file():
        if _json_load(path) != normalized:
            raise ArtifactCompatibilityError("frozen paired profile changed on resume")
    else:
        atomic_write_json(path, normalized)
    return normalized


def _bind_confirmation_profile(
    run_dir: Path, selected: Mapping[str, Any]
) -> dict[str, Any]:
    profile_path = run_dir / "selected_paired_ratio_profile.json"
    pilot_path = run_dir / "stability_pilot_gate.json"
    normalized = json.loads(json.dumps(dict(selected), sort_keys=True, allow_nan=False))
    if not profile_path.is_file() or _json_load(profile_path) != normalized:
        raise ArtifactCompatibilityError("confirmation profile is not frozen")
    if not pilot_path.is_file():
        raise ArtifactCompatibilityError("confirmation lacks the pilot gate")
    pilot = _json_load(pilot_path)
    frozen = dict(dict(pilot.get("selected_profile", {})).get("profile", {}) or {})
    if frozen != normalized:
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


def _run_pilot(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
    thresholds: RatioStabilityThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    level_records: list[dict[str, Any]] = []
    shared_seed = base._derived_seed(int(args.root_seed), "pilot", "shared-model")
    selected: dict[str, Any] = {}
    for level_index, accumulation_level in enumerate(args.accumulation_levels):
        # Selection panels are fresh at every hierarchy level, while training
        # uses one common stateless phase.  Hence level 2 is an exact prefix of
        # level 4, which is an exact prefix of level 8 for every optimizer step.
        panel_phase = f"pilot-selection-accum-{int(accumulation_level)}"
        training_phase = "pilot"
        panels = {
            task: _prepare_panel_set(
                run_dir,
                phase=panel_phase,
                task=task,
                roles=("a", "b"),
                path_count=int(args.pilot_selection_paths),
                stream_plan=stream_plan,
                scientific_fingerprint=str(manifest["scientific_fingerprint"]),
                start_offset=(
                    3_000_000
                    + level_index * 2_000_000
                    + task_index * 1_000_000
                ),
            )
            for task_index, task in enumerate(("bounded_teacher", "dirichlet_null"))
        }
        registry = _panel_registry(phase=panel_phase, panels=panels)
        if not bool(int(dict(registry["disjointness"]).get("passed", 0))):
            raise ArtifactCompatibilityError("pilot panels overlap")
        atomic_write_json(
            run_dir / f"pilot_panel_registry_accum_{int(accumulation_level)}.json",
            registry,
        )
        level_candidates: list[dict[str, Any]] = []
        for rate_index, learning_rate in enumerate(args.pilot_learning_rates):
            results: dict[str, Any] = {}
            for task in ("bounded_teacher", "dirichlet_null"):
                task_dir = (
                    run_dir / "pilot" / f"accum-{int(accumulation_level):02d}"
                    / f"lr-{rate_index:02d}" / task
                )
                fingerprints = _task_fingerprints(
                    manifest=manifest,
                    phase=training_phase,
                    task=task,
                    model_seed=int(shared_seed),
                    learning_rate=float(learning_rate),
                    accumulation_level=int(accumulation_level),
                    stream_plan=stream_plan,
                    paired_stream_plan=paired_stream_plan,
                    selection_panels=panels[task],
                    audit_panels=None,
                )
                try:
                    results[task] = run_paired_density_ratio_task(
                        task_dir=task_dir,
                        task=task,
                        selection_panels=panels[task],
                        audit_panels=None,
                        dynamics=dynamics,
                        args=_task_args(
                            args,
                            phase=training_phase,
                            learning_rate=float(learning_rate),
                            accumulation_level=int(accumulation_level),
                        ),
                        device=device,
                        model_seed=int(shared_seed),
                        learning_rate=float(learning_rate),
                        accumulation_level=int(accumulation_level),
                        stream_plan=stream_plan,
                        paired_stream_plan=paired_stream_plan,
                        fingerprints=fingerprints,
                        phase=training_phase,
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
                            "accumulation_level": int(accumulation_level),
                            "learning_rate": float(learning_rate),
                            "task": task,
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "physical_training_performed": 0,
                            "sampling_performed": 0,
                        }
                    )
            candidate = {
                "evaluation_status": "evaluated",
                "learning_rate": float(learning_rate),
                "accumulation_steps": int(accumulation_level),
                "teacher": results["bounded_teacher"],
                "null": results["dirichlet_null"],
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
            candidate_gate = evaluate_stability_pilot_candidate(candidate, thresholds)
            candidate["gate"] = candidate_gate
            level_candidates.append(candidate)
            candidates.append(candidate)
        level_gate = evaluate_stability_pilot_level(
            level_candidates, int(accumulation_level), thresholds
        )
        level_records.append(level_gate)
        atomic_write_json(
            run_dir / f"stability_pilot_level_accum_{int(accumulation_level)}.json",
            {
                **level_gate,
                "candidates": level_candidates,
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )
        if bool(int(level_gate.get("passed", 0))):
            break

    pilot = evaluate_stability_pilot(candidates, thresholds)
    pilot["candidate_records"] = candidates
    pilot["executed_level_gates"] = level_records
    atomic_write_json(run_dir / "stability_pilot_gate.json", pilot)
    atomic_write_json(
        run_dir / "pilot_task_failures.json",
        {
            "failures": failures,
            "count": len(failures),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    profile_wrapper = dict(pilot.get("selected_profile", {}))
    raw_profile = profile_wrapper.get("profile")
    if bool(int(pilot.get("passed", 0))) and isinstance(raw_profile, Mapping):
        selected = _freeze_selected_profile(run_dir, dict(raw_profile))
    return pilot, selected


def _run_confirmation(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
    thresholds: RatioStabilityThresholds,
    selected_profile: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    binding = _bind_confirmation_profile(run_dir, selected_profile)
    learning_rate = float(selected_profile["learning_rate"])
    accumulation_level = int(selected_profile["accumulation_steps"])
    panels: dict[str, dict[str, DensityRatioPanel]] = {}
    for task_index, task in enumerate(("bounded_teacher", "dirichlet_null")):
        selection = _prepare_panel_set(
            run_dir,
            phase="confirmation",
            task=task,
            roles=("a", "b"),
            path_count=int(args.confirm_selection_paths),
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            start_offset=10_000_000 + task_index * 2_000_000,
        )
        audit = _prepare_panel_set(
            run_dir,
            phase="confirmation",
            task=task,
            roles=("c", "d"),
            path_count=int(args.confirm_audit_paths),
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            start_offset=11_000_000 + task_index * 2_000_000,
        )
        panels[task] = {**selection, **audit}
    registry = _panel_registry(phase="confirmation", panels=panels)
    if not bool(int(dict(registry["disjointness"]).get("passed", 0))):
        raise ArtifactCompatibilityError("confirmation panels overlap")
    atomic_write_json(run_dir / "confirmation_panel_registry.json", registry)
    teacher_results: list[dict[str, Any]] = []
    null_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_seed in args.confirm_model_seeds:
        for task, output in (
            ("bounded_teacher", teacher_results),
            ("dirichlet_null", null_results),
        ):
            task_dir = run_dir / "confirmation" / f"seed-{int(model_seed)}" / task
            fingerprints = _task_fingerprints(
                manifest=manifest,
                phase="confirmation",
                task=task,
                model_seed=int(model_seed),
                learning_rate=learning_rate,
                accumulation_level=accumulation_level,
                stream_plan=stream_plan,
                paired_stream_plan=paired_stream_plan,
                selection_panels={name: panels[task][name] for name in ("a", "b")},
                audit_panels={name: panels[task][name] for name in ("c", "d")},
                profile_binding=binding,
            )
            try:
                result = run_paired_density_ratio_task(
                    task_dir=task_dir,
                    task=task,
                    selection_panels={name: panels[task][name] for name in ("a", "b")},
                    audit_panels={name: panels[task][name] for name in ("c", "d")},
                    dynamics=dynamics,
                    args=_task_args(
                        args,
                        phase="confirmation",
                        learning_rate=learning_rate,
                        accumulation_level=accumulation_level,
                    ),
                    device=device,
                    model_seed=int(model_seed),
                    learning_rate=learning_rate,
                    accumulation_level=accumulation_level,
                    stream_plan=stream_plan,
                    paired_stream_plan=paired_stream_plan,
                    fingerprints=fingerprints,
                    phase="confirmation",
                    thresholds=thresholds,
                    show_progress=not bool(args.no_progress),
                )
            except FloatingPointError as exc:
                result = _failed_task_result(
                    task_dir,
                    task=task,
                    model_seed=int(model_seed),
                    fingerprints=fingerprints,
                    exc=exc,
                )
                failures.append(
                    {
                        "task": task,
                        "model_seed": int(model_seed),
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    }
                )
            output.append(result)
    atomic_write_json(
        run_dir / "paired_ratio_teacher_confirmation.json",
        {
            "task_results": teacher_results,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "paired_ratio_null_confirmation.json",
        {
            "task_results": null_results,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "confirmation_task_failures.json",
        {
            "failures": failures,
            "count": len(failures),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return teacher_results, null_results


def _load_report_inputs(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    preflight_path = run_dir / "paired_ratio_preflight_gate.json"
    pilot_path = run_dir / "stability_pilot_gate.json"
    teacher_path = run_dir / "paired_ratio_teacher_confirmation.json"
    null_path = run_dir / "paired_ratio_null_confirmation.json"
    preflight = (
        _json_load(preflight_path)
        if preflight_path.is_file()
        else _not_evaluated_gate("paired_ratio_preflight", "preflight was not run")
    )
    pilot = (
        _json_load(pilot_path)
        if pilot_path.is_file()
        else _not_evaluated_gate("paired_ratio_stability_pilot", "pilot was not run")
    )
    teacher = (
        [dict(value) for value in _json_load(teacher_path).get("task_results", [])]
        if teacher_path.is_file() else []
    )
    null = (
        [dict(value) for value in _json_load(null_path).get("task_results", [])]
        if null_path.is_file() else []
    )
    return preflight, pilot, teacher, null


def _workflow_report(
    *,
    provenance: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pilot: Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    require_gate: str,
    thresholds: RatioStabilityThresholds,
) -> dict[str, Any]:
    return evaluate_ratio_stability_workflow(
        provenance=provenance,
        preflight=preflight,
        pilot=pilot,
        teacher_results=teacher_results,
        null_results=null_results,
        require_gate=require_gate,
        thresholds=thresholds,
    )


def _mark_interim_stage_success(
    report: Mapping[str, Any], *, stage: str
) -> dict[str, Any]:
    value = copy.deepcopy(dict(report))
    if stage == "preflight":
        decision = "paired_ratio_preflight_passed"
        action = "run the hierarchical paired-mixture optimizer pilot"
    elif stage == "pilot":
        decision = "paired_ratio_pilot_passed"
        action = "run the fresh three-seed paired-mixture confirmation"
    else:
        return value
    value["decision"] = {
        "decision": decision,
        "recommended_next_action": action,
        "interim_stage_success": 1,
        "closed_terminal_scientific_outcome": 0,
        "physical_training_authorized": 0,
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }
    value["interim_stage"] = stage
    return value


def _save_report(run_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / "boundary_control_stability_gate.json", report)
    atomic_write_json(run_dir / "density_ratio_stability_decision.json", report["decision"])


def _write_summary_csvs(run_dir: Path) -> list[str]:
    written: list[str] = []
    pilot_path = run_dir / "stability_pilot_gate.json"
    if pilot_path.is_file():
        rows = []
        for candidate in _json_load(pilot_path).get("candidate_records", []):
            candidate = dict(candidate)
            gate = dict(candidate.get("gate", {}))
            rows.append(
                {
                    "accumulation_steps": candidate.get("accumulation_steps"),
                    "learning_rate": candidate.get("learning_rate"),
                    "passed": gate.get("passed"),
                    "teacher_mean_ab_bce": gate.get("teacher_mean_ab_bce"),
                    "maximum_clip_fraction_observed": gate.get(
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
        ("paired_ratio_teacher_confirmation.json", "bounded_teacher"),
        ("paired_ratio_null_confirmation.json", "dirichlet_null"),
    ):
        path = run_dir / filename
        if not path.is_file():
            continue
        for raw in _json_load(path).get("task_results", []):
            metrics = dict(dict(raw).get("metrics", {}))
            rows.append(
                {
                    "task": task,
                    "model_seed": raw.get("model_seed"),
                    "complete": metrics.get("complete"),
                    "finite": metrics.get("finite"),
                    "selected_step": metrics.get("selected_step"),
                    "nominee_step": metrics.get("nominee_step"),
                    "post_warmup_clip_fraction": metrics.get(
                        "post_warmup_clip_fraction"
                    ),
                    "final_500_clip_fraction": metrics.get(
                        "final_500_clip_fraction"
                    ),
                    "final_200_clip_fraction": metrics.get(
                        "final_200_clip_fraction"
                    ),
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
    if rows:
        atomic_write_csv(run_dir / "confirmation_seed_metrics.csv", rows)
        written.append("confirmation_seed_metrics.csv")
    return written


def _write_plot_artifacts(run_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written: list[str] = []
    histories = sorted(run_dir.glob("pilot/**/training_history.csv"))
    if histories:
        figure, axis = plt.subplots(figsize=(8, 5))
        for path in histories:
            try:
                data = np.genfromtxt(path, delimiter=",", names=True)
                if data.size:
                    axis.plot(
                        np.atleast_1d(data["step"]),
                        np.atleast_1d(data["scaled_loss"]),
                        alpha=0.65,
                        label=path.parent.relative_to(run_dir / "pilot").as_posix(),
                    )
            except Exception:
                continue
        axis.set_xlabel("optimizer update")
        axis.set_ylabel("scaled paired BCE")
        axis.set_title("Hierarchical paired-mixture pilot")
        if len(axis.lines) <= 12:
            axis.legend(fontsize=6)
        figure.tight_layout()
        output = run_dir / "paired_pilot_learning_curves.png"
        base._atomic_save_figure(figure, output)
        plt.close(figure)
        written.append(output.name)
    confirmation_histories = sorted(
        run_dir.glob("confirmation/**/training_history.csv")
    )
    if confirmation_histories:
        figure, (loss_axis, gradient_axis) = plt.subplots(
            2, 1, figsize=(9, 8), sharex=True
        )
        for path in confirmation_histories:
            try:
                data = np.genfromtxt(path, delimiter=",", names=True)
                if not data.size:
                    continue
                step = np.atleast_1d(data["step"])
                label = path.parent.relative_to(
                    run_dir / "confirmation"
                ).as_posix()
                loss_axis.plot(
                    step,
                    np.atleast_1d(data["scaled_loss"]),
                    alpha=0.7,
                    label=label,
                )
                gradients = np.atleast_1d(data["scaled_preclip_gradient_norm"])
                gradient_axis.plot(step, gradients, alpha=0.7, label=label)
                if "clipped" in (data.dtype.names or ()):
                    clipped = np.atleast_1d(data["clipped"]).astype(bool)
                    gradient_axis.scatter(
                        step[clipped], gradients[clipped], s=8, marker="x"
                    )
            except Exception:
                continue
        loss_axis.set_ylabel("scaled paired BCE")
        loss_axis.set_title("Paired-mixture confirmation learning curves")
        gradient_axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        gradient_axis.set_ylabel("scaled pre-clip gradient norm")
        gradient_axis.set_xlabel("optimizer update")
        if len(loss_axis.lines) <= 12:
            loss_axis.legend(fontsize=6)
        figure.tight_layout()
        output = run_dir / "paired_confirmation_learning_optimizer_health.png"
        base._atomic_save_figure(figure, output)
        plt.close(figure)
        written.append(output.name)
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
        outcome=(
            "implementation_error" if execution_failed
            else ("complete" if required_pass else "gate_failed")
        ),
        phase=phase,
        stage=stage,
        required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=decision.get("decision", "controls_not_run"),
        recommended_next_action=decision.get("recommended_next_action"),
        physical_training_authorized=(
            int(decision.get("physical_training_authorized", 0))
            if not execution_failed else 0
        ),
        physical_training_performed=0,
        sampling_authorized=0,
        sampling_performed=0,
        skips=final_skips,
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_size=int(registry_path.stat().st_size),
    )
    return 2 if execution_failed or not required_pass else 0


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    if not args.no_progress:
        print(f"paired density-ratio run directory: {run_dir.resolve()}", flush=True)
    thresholds = RatioStabilityThresholds()
    mutation_started = False
    try:
        device = base._device(args.device)
        backend = configure_exact_torch_backend(device)
        runtime = base._runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
        source_hash, source_paths = _source_record()
        parent = verify_parent_density_ratio_run(args.parent_density_ratio_run_dir)
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
            raise ArtifactCompatibilityError("resume lacks parent provenance")
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
                "schema", "schema_version", "scientific_config",
                "scientific_fingerprint", "runtime", "runtime_fingerprint",
                "source_fingerprint", "source_paths", "parent_provenance_sha256",
                "claim_scope",
            ):
                if existing.get(key) != manifest.get(key):
                    raise ArtifactCompatibilityError(f"resume manifest mismatch for {key}")
            manifest = existing
        elif resumed:
            raise ArtifactCompatibilityError("resume lacks frozen manifest")
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
            preflight, pilot, teacher_results, null_results = _load_report_inputs(run_dir)
            report = _workflow_report(
                provenance=provenance,
                preflight=preflight,
                pilot=pilot,
                teacher_results=teacher_results,
                null_results=null_results,
                require_gate=str(args.require_gate),
                thresholds=thresholds,
            )
            if (
                bool(int(pilot.get("passed", 0)))
                and not teacher_results
                and not null_results
            ):
                report = _mark_interim_stage_success(report, stage="pilot")
            elif (
                bool(int(preflight.get("passed", 0)))
                and str(pilot.get("evaluation_status")) == "not_evaluated"
            ):
                report = _mark_interim_stage_success(report, stage="preflight")
            _save_report(run_dir, report)
            return _finish(run_dir, report=report, stage="report", phase="report")

        dynamics = base._make_dynamics(args)
        horizon = float(natural_horizon(dynamics))
        if not math.isclose(
            horizon, float(parent.get("horizon", math.nan)), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ArtifactCompatibilityError("paired horizon differs from parent")
        stream_plan = build_density_ratio_stream_plan(
            root_seed=int(args.root_seed),
            grid_size=int(args.grid_size),
            horizon=horizon,
            label=3,
            bin_counts=(4, 4, 4, 4, 16),
            teacher_epsilon=0.5,
        )
        paired_stream_plan = build_paired_mixture_stream_plan(
            root_seed=int(args.root_seed),
            grid_size=int(args.grid_size),
            horizon=horizon,
            label=3,
            teacher_epsilon=0.5,
        )
        plans = {
            "evaluation_stream": stream_plan_record(stream_plan),
            "paired_training_stream": paired_mixture_stream_plan_record(
                paired_stream_plan
            ),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        plans_path = run_dir / "paired_ratio_stream_plans.json"
        if plans_path.is_file():
            if _json_load(plans_path) != plans:
                raise ArtifactCompatibilityError("resume stream plans changed")
        elif resumed:
            raise ArtifactCompatibilityError("resume lacks frozen stream plans")
        else:
            atomic_write_json(plans_path, plans)

        _write_status(run_dir, status="running", phase="preflight")
        preflight = _run_preflight(
            run_dir,
            args=args,
            manifest=manifest,
            provenance=provenance,
            dynamics=dynamics,
            device=device,
            stream_plan=stream_plan,
            paired_stream_plan=paired_stream_plan,
            thresholds=thresholds,
        )
        pilot = _not_evaluated_gate("paired_ratio_stability_pilot", "pilot was not run")
        teacher_results: list[dict[str, Any]] = []
        null_results: list[dict[str, Any]] = []
        if args.stage == "preflight" or not bool(int(preflight.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance,
                preflight=preflight,
                pilot=pilot,
                teacher_results=[],
                null_results=[],
                require_gate=str(args.require_gate),
                thresholds=thresholds,
            )
            if args.stage == "preflight" and bool(int(preflight.get("passed", 0))):
                report = _mark_interim_stage_success(report, stage="preflight")
            _save_report(run_dir, report)
            return _finish(
                run_dir,
                report=report,
                stage=str(args.stage),
                phase="preflight",
                skips=([] if bool(int(preflight.get("passed", 0))) else [
                    {"stage": "pilot_and_confirmation", "reason": "paired preflight failed"}
                ]),
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
                paired_stream_plan=paired_stream_plan,
                thresholds=thresholds,
            )
            if args.stage == "pilot" or not bool(int(pilot.get("passed", 0))):
                report = _workflow_report(
                    provenance=provenance,
                    preflight=preflight,
                    pilot=pilot,
                    teacher_results=[],
                    null_results=[],
                    require_gate=str(args.require_gate),
                    thresholds=thresholds,
                )
                if args.stage == "pilot" and bool(int(pilot.get("passed", 0))):
                    report = _mark_interim_stage_success(report, stage="pilot")
                _save_report(run_dir, report)
                return _finish(
                    run_dir,
                    report=report,
                    stage=str(args.stage),
                    phase="pilot",
                    skips=([] if bool(int(pilot.get("passed", 0))) else [
                        {
                            "stage": "confirmation",
                            "reason": "no paired variance-reduction profile qualified",
                        }
                    ]),
                )
        else:
            pilot_path = run_dir / "stability_pilot_gate.json"
            profile_path = run_dir / "selected_paired_ratio_profile.json"
            if not pilot_path.is_file() or not profile_path.is_file():
                raise ArtifactCompatibilityError("confirmation requires frozen pilot")
            pilot = _json_load(pilot_path)
            if not bool(int(pilot.get("passed", 0))):
                raise ArtifactCompatibilityError("confirmation requires passing pilot")
            selected_profile = _json_load(profile_path)
            if dict(dict(pilot.get("selected_profile", {})).get("profile", {}) or {}) != selected_profile:
                raise ArtifactCompatibilityError("pilot/profile binding mismatch")

        _write_status(run_dir, status="running", phase="confirmation")
        teacher_results, null_results = _run_confirmation(
            run_dir,
            args=args,
            manifest=manifest,
            dynamics=dynamics,
            device=device,
            stream_plan=stream_plan,
            paired_stream_plan=paired_stream_plan,
            thresholds=thresholds,
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
                    f"paired density-ratio resume rejected: {type(exc).__name__}: {exc}",
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
            skips=[{"stage": "remaining", "reason": f"{type(exc).__name__}: {exc}"}],
        )
        if not args.no_progress:
            print(
                f"paired density-ratio control failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
