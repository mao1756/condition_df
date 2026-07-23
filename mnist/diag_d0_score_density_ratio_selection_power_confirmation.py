"""Oracle-qualified evidence-power confirmation for D0 density-ratio controls.

This workflow changes only the fixed selection/audit panel sizes.  It binds
the immutable normalized-head run, verifies that exact Bayes logits are
detectable before any optimizer step, and then reuses the normalized-head
model, paired BCE estimator, optimizer, checkpointing, and scientific gates.
It is controls-only and deliberately contains no sampler integration.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

import mnist.diag_d0_score_density_ratio_controls as base
import mnist.diag_d0_score_density_ratio_head_confirmation as head
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_score_density_ratio import (
    DensityRatioPanel,
    DensityRatioStreamPlan,
    build_density_ratio_stream_plan,
    load_density_ratio_panel,
    panel_disjointness_record,
    panel_identity,
    stream_plan_record,
)
from mnist.d0_score_density_ratio_head import (
    COORDINATE_CONJUGATE_ADAMW_VERSION,
    NORMALIZED_HEAD_COORDINATE_VERSION,
    NORMALIZED_HEAD_MODEL_VERSION,
)
from mnist.d0_score_density_ratio_head_gate import HeadCoordinateThresholds
from mnist.d0_score_density_ratio_head_provenance import PARENT_LOSS_SCALE
from mnist.d0_score_density_ratio_paired import (
    PAIRED_MIXTURE_ACCUMULATION_VERSION,
    PAIRED_MIXTURE_OBJECTIVE_VERSION,
    PAIRED_MIXTURE_SCHEMA,
    PAIRED_MIXTURE_STREAM_VERSION,
    PairedMixtureStreamPlan,
    build_paired_mixture_stream_plan,
    paired_mixture_stream_plan_record,
)
from mnist.d0_score_density_ratio_selection_power import (
    evaluate_oracle_panel_feasibility,
    evaluate_oracle_power_calibration,
    reproduce_saved_16_path_oracle_forensic,
)
from mnist.d0_score_density_ratio_selection_power_gate import (
    SelectionPowerThresholds,
    evaluate_oracle_panel_set,
    evaluate_power_pilot,
    evaluate_power_pilot_candidate,
    evaluate_selection_power_preflight,
    evaluate_selection_power_workflow,
)
from mnist.d0_score_density_ratio_selection_power_provenance import (
    verify_parent_normalized_head_run,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


RUN_SCHEMA = "experiment12-d0-score-density-ratio-selection-power-confirmation"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = (
    "oracle-qualified fixed-panel bounded synthetic density-ratio controls only"
)

EXPECTED_KERNEL = dict(base.EXPECTED_KERNEL)
DEFAULTS: dict[str, Any] = {
    **EXPECTED_KERNEL,
    "root_seed": 260931,
    "pilot_learning_rates": (3e-5, 1e-5),
    "accumulation_levels": (8,),
    "microbatch_clusters": 32,
    "pilot_steps": 2_000,
    "confirm_steps": 4_000,
    "oracle_calibration_paths": 256,
    "oracle_half_paths": 128,
    "pilot_selection_paths": 128,
    "confirm_selection_paths": 128,
    "confirm_audit_paths": 128,
    "base_channels": 32,
    "validation_batch_size": 64,
    "weight_decay": 1e-4,
    "ema_decay": 0.99,
    "grad_clip": 1.0,
    "clip_warmup_steps": 500,
    "loss_scale": float(PARENT_LOSS_SCALE),
    "bootstrap_reps": 10_000,
    "bootstrap_confidence": 0.90,
    "oracle_calibration_confidence": 0.99,
    "confirm_model_seeds": (260941, 260942, 260943),
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
    parser.add_argument("--parent-normalized-head-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_score_density_ratio_selection_power_confirmation"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-density-ratio-selection-power"
    )
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
        "oracle_calibration_paths", "oracle_half_paths",
        "pilot_selection_paths", "confirm_selection_paths",
        "confirm_audit_paths", "microbatch_clusters", "bootstrap_reps",
        "confirm_validation_every", "clip_warmup_steps",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=DEFAULTS[name]
        )
    for name in (
        "tau_eff", "alpha_eff", "mass_floor", "limiter_fraction", "lambda_mix",
        "weight_decay", "ema_decay", "grad_clip", "loss_scale",
        "bootstrap_confidence", "oracle_calibration_confidence",
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

    positive_names = (
        "grid_size", "sample_steps", "reference_substeps", "base_channels",
        "validation_batch_size", "pilot_steps", "confirm_steps",
        "oracle_calibration_paths", "oracle_half_paths",
        "pilot_selection_paths", "confirm_selection_paths",
        "confirm_audit_paths", "microbatch_clusters", "bootstrap_reps",
        "confirm_validation_every",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.oracle_calibration_paths) != 2 * int(args.oracle_half_paths):
        parser.error("--oracle-calibration-paths must equal two oracle half-panels")
    if int(args.microbatch_clusters) != 32:
        parser.error("paired estimator v1 requires --microbatch-clusters 32")
    if tuple(args.accumulation_levels) != (8,):
        parser.error("the normalized-head confirmation freezes accumulation at 8")
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
    for name in ("ema_decay", "bootstrap_confidence", "oracle_calibration_confidence"):
        if not 0.0 < float(getattr(args, name)) < 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in (0,1)")
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
            if hasattr(args, key)
            and not base._semantic_equal(getattr(args, key), expected)
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
        "d0_score_density_ratio_selection_power.py",
        "d0_score_density_ratio_selection_power_gate.py",
        "d0_score_density_ratio_selection_power_provenance.py",
        "diag_d0_score_density_ratio_head_confirmation.py",
        "d0_score_density_ratio_head.py",
        "d0_score_density_ratio_head_gate.py",
        "d0_score_density_ratio_head_provenance.py",
        "d0_score_density_ratio_paired.py",
        "d0_score_density_ratio_stability_gate.py",
        "d0_score_density_ratio.py",
        "d0_score_density_ratio_gate.py",
        "d0_score_boundary_controls.py",
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
    registry_path, status_path = (
        run_dir / "artifact_registry.json", run_dir / "run_status.json"
    )
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


def _freeze_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _json_load(path) != normalized:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    else:
        atomic_write_json(path, normalized)
    return normalized


def _scientific_config(
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    thresholds: SelectionPowerThresholds,
) -> dict[str, Any]:
    value = {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "model_schema": NORMALIZED_HEAD_MODEL_VERSION,
        "head_coordinate_version": NORMALIZED_HEAD_COORDINATE_VERSION,
        "optimizer_coordinate_version": COORDINATE_CONJUGATE_ADAMW_VERSION,
        "paired_estimator_schema": PAIRED_MIXTURE_SCHEMA,
        "paired_objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
        "paired_stream_version": PAIRED_MIXTURE_STREAM_VERSION,
        "paired_accumulation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
        "kernel": {key: getattr(args, key) for key in EXPECTED_KERNEL},
        "root_seed": int(args.root_seed),
        "loss_scale": float(args.loss_scale),
        "oracle_power": {
            "saved_parent_paths_per_panel": 16,
            "calibration_paths": int(args.oracle_calibration_paths),
            "predetermined_half_paths": int(args.oracle_half_paths),
            "calibration_confidence": float(args.oracle_calibration_confidence),
            "evaluation_confidence": float(args.bootstrap_confidence),
            "bootstrap_reps": int(args.bootstrap_reps),
            "panel_regeneration_after_inspection": 0,
        },
        "pilot": {
            "learning_rates": list(args.pilot_learning_rates),
            "accumulation_levels": list(args.accumulation_levels),
            "steps": int(args.pilot_steps),
            "paths_per_panel": int(args.pilot_selection_paths),
            "validation_steps": list(args.pilot_validation_steps),
        },
        "confirmation": {
            "steps": int(args.confirm_steps),
            "model_seeds": list(args.confirm_model_seeds),
            "paths_per_selection_panel": int(args.confirm_selection_paths),
            "paths_per_audit_panel": int(args.confirm_audit_paths),
            "dense_validation_steps": list(args.confirm_dense_validation_steps),
            "validation_every": int(args.confirm_validation_every),
        },
        "optimization": {
            "optimizer": "coordinate-conjugate-AdamW",
            "body_weight_decay": float(args.weight_decay),
            "head_lr_factor": int(args.grid_size) ** 2,
            "head_eps_factor": 1.0 / float(int(args.grid_size) ** 2),
            "head_weight_decay_factor": 1.0 / float(int(args.grid_size) ** 2),
            "ema_decay": float(args.ema_decay),
            "grad_clip": float(args.grad_clip),
            "clip_warmup_steps": int(args.clip_warmup_steps),
            "gradient_accumulation": "mean-then-clip-once",
            "adaptive_loss_scaling": 0,
        },
        "microbatch": {
            "clusters": int(args.microbatch_clusters),
            "time_bin_counts": [4, 4, 4, 4, 16],
        },
        "thresholds": thresholds.to_dict(),
        "parent_scientific_fingerprint": parent.get("scientific_fingerprint"),
        "parent_artifact_registry_sha256": parent.get("artifact_registry_sha256"),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _panel_registry(
    *, phase: str, panels: Mapping[str, Mapping[str, DensityRatioPanel]],
    extra_panels: Sequence[DensityRatioPanel] = (),
) -> dict[str, Any]:
    flat = [panel for group in panels.values() for panel in group.values()]
    flat.extend(extra_panels)
    return {
        "schema": RUN_SCHEMA + "-panel-registry",
        "schema_version": 1,
        "phase": str(phase),
        "panels": {
            task: {role: panel_identity(panel) for role, panel in group.items()}
            for task, group in panels.items()
        },
        "extra_panel_identities": [panel_identity(panel) for panel in extra_panels],
        "disjointness": panel_disjointness_record(flat),
        "panel_regeneration_after_inspection": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


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
    return head._prepare_panel_set(
        run_dir,
        phase=phase,
        task=task,
        roles=roles,
        path_count=path_count,
        stream_plan=stream_plan,
        scientific_fingerprint=scientific_fingerprint,
        start_offset=start_offset,
    )


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
    oracle_feasibility_path: Path,
    profile_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = head._task_fingerprints(
        manifest=manifest,
        phase=phase,
        task=task,
        model_seed=model_seed,
        learning_rate=learning_rate,
        accumulation_level=accumulation_level,
        stream_plan=stream_plan,
        paired_stream_plan=paired_stream_plan,
        selection_panels=selection_panels,
        audit_panels=audit_panels,
        profile_binding=profile_binding,
    )
    value.update(
        {
            "selection_power_workflow_schema": RUN_SCHEMA,
            "oracle_feasibility_sha256": file_fingerprint(oracle_feasibility_path),
            "oracle_feasibility_size": int(oracle_feasibility_path.stat().st_size),
        }
    )
    return value


def _oracle_panel_bundle(
    panels: Mapping[str, DensityRatioPanel],
    *,
    calibration_panel: DensityRatioPanel,
    confidence: float,
    reps: int,
    root_seed: int,
    namespace: str,
) -> dict[str, Any]:
    records = {
        role: evaluate_oracle_panel_feasibility(
            panel,
            confidence=float(confidence),
            reps=int(reps),
            seed=base._derived_seed(int(root_seed), namespace, role),
        )
        for role, panel in panels.items()
    }
    flat = list(panels.values())
    calibration_streams = set(calibration_panel.stream_fingerprints)
    raw = {
        "schema": RUN_SCHEMA + "-oracle-panel-feasibility",
        "schema_version": 1,
        "namespace": str(namespace),
        "confidence": float(confidence),
        "panels": records,
        "pairwise_disjoint": int(
            bool(int(panel_disjointness_record(flat).get("passed", 0)))
        ),
        "calibration_overlap_path_count": int(
            sum(
                stream in calibration_streams
                for panel in flat
                for stream in panel.stream_fingerprints
            )
        ),
        "frozen_before_training": 1,
        "optimizer_steps_before_oracle_gate": 0,
        "panel_regeneration_after_inspection": 0,
        "regenerated_after_inspection": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    gate = evaluate_oracle_panel_set(
        raw,
        expected_roles=tuple(panels),
    )
    gate["raw_evidence"] = raw
    return gate


def _run_preflight(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    stream_plan: DensityRatioStreamPlan,
    thresholds: SelectionPowerThresholds,
) -> tuple[dict[str, Any], DensityRatioPanel]:
    parent_dir = Path(args.parent_normalized_head_run_dir).resolve()
    parent_panels = {
        role: load_density_ratio_panel(
            parent_dir
            / "panels"
            / "pilot-selection-accum-8"
            / f"bounded_teacher-{role}.pt",
            device="cpu",
            expected_role=role,
            expected_task="bounded_teacher",
        )
        for role in ("a", "b")
    }
    forensic = reproduce_saved_16_path_oracle_forensic(
        parent_panels["a"],
        parent_panels["b"],
        reps=int(args.bootstrap_reps),
        confidence=float(args.bootstrap_confidence),
        seed=base._derived_seed(int(args.root_seed), "saved-parent-oracle"),
    )
    _freeze_json(run_dir / "saved_16_path_oracle_forensic.json", forensic)

    calibration = _prepare_panel_set(
        run_dir,
        phase="oracle-power-calibration",
        task="bounded_teacher",
        roles=("calibration",),
        path_count=int(args.oracle_calibration_paths),
        stream_plan=stream_plan,
        scientific_fingerprint=str(manifest["scientific_fingerprint"]),
        start_offset=1_000_000,
    )["calibration"]
    registry = _panel_registry(
        phase="oracle-power-calibration",
        panels={"bounded_teacher": {"calibration": calibration}},
    )
    if not bool(int(dict(registry["disjointness"]).get("passed", 0))):
        raise ArtifactCompatibilityError("oracle calibration registry is invalid")
    _freeze_json(run_dir / "oracle_calibration_panel_registry.json", registry)
    calibration_record = evaluate_oracle_power_calibration(
        calibration,
        reps=int(args.bootstrap_reps),
        seed=base._derived_seed(int(args.root_seed), "oracle-calibration"),
        expected_paths=int(args.oracle_calibration_paths),
        expected_half_paths=int(args.oracle_half_paths),
        full_confidence=float(args.oracle_calibration_confidence),
        half_confidence=float(args.bootstrap_confidence),
    )
    _freeze_json(run_dir / "oracle_power_calibration.json", calibration_record)
    gate = evaluate_selection_power_preflight(
        normalized_head_preflight={
            "evaluation_status": "evaluated",
            "passed": int(provenance.get("preflight_pass", 0)),
        },
        saved_forensic=forensic,
        calibration=calibration_record,
        thresholds=thresholds,
    )
    _freeze_json(run_dir / "selection_power_preflight_gate.json", gate)
    return gate, calibration


def _pilot_panels(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    stream_plan: DensityRatioStreamPlan,
    calibration_panel: DensityRatioPanel,
) -> tuple[dict[str, dict[str, DensityRatioPanel]], dict[str, Any]]:
    panels = {
        task: _prepare_panel_set(
            run_dir,
            phase="selection-power-pilot",
            task=task,
            roles=("a", "b"),
            path_count=int(args.pilot_selection_paths),
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            start_offset=3_000_000 + task_index * 1_000_000,
        )
        for task_index, task in enumerate(("bounded_teacher", "dirichlet_null"))
    }
    registry = _panel_registry(
        phase="selection-power-pilot", panels=panels,
        extra_panels=(calibration_panel,),
    )
    if not bool(int(dict(registry["disjointness"]).get("passed", 0))):
        raise ArtifactCompatibilityError("pilot/calibration panels overlap")
    _freeze_json(run_dir / "pilot_panel_registry.json", registry)
    oracle = _oracle_panel_bundle(
        panels["bounded_teacher"],
        calibration_panel=calibration_panel,
        confidence=float(args.bootstrap_confidence),
        reps=int(args.bootstrap_reps),
        root_seed=int(args.root_seed),
        namespace="pilot-actual-panels",
    )
    _freeze_json(run_dir / "pilot_oracle_feasibility.json", oracle)
    return panels, oracle


def _freeze_selected_profile(run_dir: Path, selected: Mapping[str, Any]) -> dict[str, Any]:
    return _freeze_json(run_dir / "selected_selection_power_profile.json", selected)


def _run_pilot(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
    calibration_panel: DensityRatioPanel,
    thresholds: SelectionPowerThresholds,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    panels, oracle = _pilot_panels(
        run_dir,
        args=args,
        manifest=manifest,
        stream_plan=stream_plan,
        calibration_panel=calibration_panel,
    )
    feasibility_path = run_dir / "pilot_oracle_feasibility.json"
    if not bool(int(oracle.get("passed", 0))):
        pilot = _not_evaluated_gate(
            "selection_power_pilot", "actual fixed pilot panels failed oracle power"
        )
        pilot["oracle_feasibility"] = oracle
        _freeze_json(run_dir / "selection_power_pilot_gate.json", pilot)
        multiplicity = {
            "status": "not_evaluated", "reason": "oracle feasibility failed",
            "physical_training_performed": 0, "sampling_performed": 0,
        }
        _freeze_json(run_dir / "pilot_null_multiplicity_analysis.json", multiplicity)
        return pilot, {}, multiplicity

    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shared_seed = base._derived_seed(int(args.root_seed), "power-pilot", "shared-model")
    accumulation_level = 8
    for rate_index, learning_rate in enumerate(args.pilot_learning_rates):
        results: dict[str, Any] = {}
        for task in ("bounded_teacher", "dirichlet_null"):
            task_dir = run_dir / "pilot" / f"lr-{rate_index:02d}" / task
            fingerprints = _task_fingerprints(
                manifest=manifest,
                phase="selection-power-pilot",
                task=task,
                model_seed=int(shared_seed),
                learning_rate=float(learning_rate),
                accumulation_level=accumulation_level,
                stream_plan=stream_plan,
                paired_stream_plan=paired_stream_plan,
                selection_panels=panels[task],
                audit_panels=None,
                oracle_feasibility_path=feasibility_path,
            )
            try:
                results[task] = head.run_paired_density_ratio_task(
                    task_dir=task_dir,
                    task=task,
                    selection_panels=panels[task],
                    audit_panels=None,
                    dynamics=dynamics,
                    args=head._task_args(
                        args,
                        phase="pilot",
                        learning_rate=float(learning_rate),
                        accumulation_level=accumulation_level,
                    ),
                    device=device,
                    model_seed=int(shared_seed),
                    learning_rate=float(learning_rate),
                    accumulation_level=accumulation_level,
                    stream_plan=stream_plan,
                    paired_stream_plan=paired_stream_plan,
                    fingerprints=fingerprints,
                    phase="selection-power-pilot",
                    thresholds=thresholds.head,
                    show_progress=not bool(args.no_progress),
                )
            except FloatingPointError as exc:
                results[task] = head._failed_task_result(
                    task_dir,
                    task=task,
                    model_seed=int(shared_seed),
                    fingerprints=fingerprints,
                    exc=exc,
                )
                failures.append(
                    {
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
            "accumulation_steps": accumulation_level,
            "teacher": results["bounded_teacher"],
            "null": results["dirichlet_null"],
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        candidate["gate"] = evaluate_power_pilot_candidate(candidate, thresholds)
        candidates.append(candidate)

    pilot = evaluate_power_pilot(candidates, panel_power=oracle, thresholds=thresholds)
    pilot["candidate_records"] = candidates
    _freeze_json(run_dir / "selection_power_pilot_gate.json", pilot)
    atomic_write_json(
        run_dir / "pilot_task_failures.json",
        {
            "failures": failures,
            "count": len(failures),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    multiplicity = dict(pilot.get("null_multiplicity_analysis", {}))
    _freeze_json(run_dir / "pilot_null_multiplicity_analysis.json", multiplicity)
    selected: dict[str, Any] = {}
    wrapper = dict(pilot.get("selected_profile", {}))
    raw_profile = wrapper.get("profile")
    if bool(int(pilot.get("passed", 0))) and isinstance(raw_profile, Mapping):
        selected = _freeze_selected_profile(run_dir, dict(raw_profile))
    return pilot, selected, multiplicity


def _confirmation_profile_binding(
    run_dir: Path, selected_profile: Mapping[str, Any]
) -> dict[str, Any]:
    profile_path = run_dir / "selected_selection_power_profile.json"
    pilot_path = run_dir / "selection_power_pilot_gate.json"
    if not profile_path.is_file() or _json_load(profile_path) != dict(selected_profile):
        raise ArtifactCompatibilityError("confirmation profile is not frozen")
    pilot = _json_load(pilot_path)
    frozen = dict(dict(pilot.get("selected_profile", {})).get("profile", {}) or {})
    if frozen != dict(selected_profile):
        raise ArtifactCompatibilityError("pilot/profile binding mismatch")
    return _freeze_json(
        run_dir / "confirmation_profile_binding.json",
        {
            "schema": RUN_SCHEMA + "-confirmation-profile-binding",
            "schema_version": 1,
            "selected_profile": dict(selected_profile),
            "selected_profile_sha256": file_fingerprint(profile_path),
            "pilot_gate_sha256": file_fingerprint(pilot_path),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )


def _run_confirmation(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
    calibration_panel: DensityRatioPanel,
    selected_profile: Mapping[str, Any],
    thresholds: SelectionPowerThresholds,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    binding = _confirmation_profile_binding(run_dir, selected_profile)
    learning_rate_value = selected_profile.get("learning_rate")
    if learning_rate_value is None:
        learning_rate_value = selected_profile["body_learning_rate"]
    learning_rate = float(learning_rate_value)
    accumulation_level = int(selected_profile["accumulation_steps"])
    panels = {
        task: _prepare_panel_set(
            run_dir,
            phase="selection-power-confirmation",
            task=task,
            roles=("a", "b", "c", "d"),
            path_count=int(args.confirm_selection_paths),
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            start_offset=10_000_000 + task_index * 2_000_000,
        )
        for task_index, task in enumerate(("bounded_teacher", "dirichlet_null"))
    }
    if int(args.confirm_selection_paths) != int(args.confirm_audit_paths):
        raise ArtifactCompatibilityError(
            "single frozen A/B/C/D builder requires equal selection/audit path counts"
        )
    registry = _panel_registry(
        phase="selection-power-confirmation", panels=panels,
        extra_panels=(calibration_panel,),
    )
    pilot_registry_path = run_dir / "pilot_panel_registry.json"
    if not pilot_registry_path.is_file():
        raise ArtifactCompatibilityError("confirmation lacks the frozen pilot panel registry")
    pilot_registry = _json_load(pilot_registry_path)
    pilot_streams = {
        str(stream)
        for group in dict(pilot_registry.get("panels", {})).values()
        if isinstance(group, Mapping)
        for identity in dict(group).values()
        if isinstance(identity, Mapping)
        for stream in identity.get("stream_fingerprints", [])
    }
    confirmation_streams = {
        str(stream)
        for group in panels.values()
        for panel in group.values()
        for stream in panel.stream_fingerprints
    }
    registry["pilot_overlap_path_count"] = len(
        pilot_streams.intersection(confirmation_streams)
    )
    registry["pilot_confirmation_isolation_pass"] = int(
        bool(pilot_streams)
        and int(registry["pilot_overlap_path_count"]) == 0
    )
    if not bool(int(dict(registry["disjointness"]).get("passed", 0))):
        raise ArtifactCompatibilityError("confirmation panels overlap")
    if not bool(int(registry["pilot_confirmation_isolation_pass"])):
        raise ArtifactCompatibilityError("confirmation panels overlap the frozen pilot")
    _freeze_json(run_dir / "confirmation_panel_registry.json", registry)
    oracle = _oracle_panel_bundle(
        panels["bounded_teacher"],
        calibration_panel=calibration_panel,
        confidence=float(args.bootstrap_confidence),
        reps=int(args.bootstrap_reps),
        root_seed=int(args.root_seed),
        namespace="confirmation-actual-panels",
    )
    _freeze_json(run_dir / "confirmation_oracle_feasibility.json", oracle)
    if not bool(int(oracle.get("passed", 0))):
        multiplicity = {
            "status": "not_evaluated",
            "reason": "actual fixed confirmation panels failed oracle power",
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        _freeze_json(
            run_dir / "confirmation_null_multiplicity_analysis.json", multiplicity
        )
        return [], [], oracle, multiplicity

    teacher_results: list[dict[str, Any]] = []
    null_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    feasibility_path = run_dir / "confirmation_oracle_feasibility.json"
    for model_seed in args.confirm_model_seeds:
        for task, output in (
            ("bounded_teacher", teacher_results),
            ("dirichlet_null", null_results),
        ):
            task_dir = run_dir / "confirmation" / f"seed-{int(model_seed)}" / task
            fingerprints = _task_fingerprints(
                manifest=manifest,
                phase="selection-power-confirmation",
                task=task,
                model_seed=int(model_seed),
                learning_rate=learning_rate,
                accumulation_level=accumulation_level,
                stream_plan=stream_plan,
                paired_stream_plan=paired_stream_plan,
                selection_panels={name: panels[task][name] for name in ("a", "b")},
                audit_panels={name: panels[task][name] for name in ("c", "d")},
                oracle_feasibility_path=feasibility_path,
                profile_binding=binding,
            )
            try:
                result = head.run_paired_density_ratio_task(
                    task_dir=task_dir,
                    task=task,
                    selection_panels={name: panels[task][name] for name in ("a", "b")},
                    audit_panels={name: panels[task][name] for name in ("c", "d")},
                    dynamics=dynamics,
                    args=head._task_args(
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
                    phase="selection-power-confirmation",
                    thresholds=thresholds.head,
                    show_progress=not bool(args.no_progress),
                )
            except FloatingPointError as exc:
                result = head._failed_task_result(
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
        run_dir / "selection_power_teacher_confirmation.json",
        {"task_results": teacher_results, "physical_training_performed": 0, "sampling_performed": 0},
    )
    atomic_write_json(
        run_dir / "selection_power_null_confirmation.json",
        {"task_results": null_results, "physical_training_performed": 0, "sampling_performed": 0},
    )
    atomic_write_json(
        run_dir / "confirmation_task_failures.json",
        {"failures": failures, "count": len(failures), "physical_training_performed": 0, "sampling_performed": 0},
    )
    multiplicity = {
        "schema": RUN_SCHEMA + "-confirmation-null-multiplicity-analysis",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "authorizing": 0,
        "seed_records": [
            {
                "model_seed": value.get("model_seed"),
                "selected_step": dict(value.get("metrics", value)).get("selected_step"),
                "nominee_step": dict(value.get("metrics", value)).get("nominee_step"),
                "selection": dict(value.get("metrics", value)).get("selection", {}),
            }
            for value in null_results
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    _freeze_json(run_dir / "confirmation_null_multiplicity_analysis.json", multiplicity)
    return teacher_results, null_results, oracle, multiplicity


def _load_report_inputs(
    run_dir: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    def load_gate(name: str, gate: str) -> dict[str, Any]:
        path = run_dir / name
        return _json_load(path) if path.is_file() else _not_evaluated_gate(gate, f"{gate} was not run")

    preflight = load_gate("selection_power_preflight_gate.json", "selection_power_preflight")
    pilot_power = load_gate("pilot_oracle_feasibility.json", "pilot_oracle_panel_power")
    pilot = load_gate("selection_power_pilot_gate.json", "selection_power_pilot")
    confirmation_power = load_gate(
        "confirmation_oracle_feasibility.json", "confirmation_oracle_panel_power"
    )
    teacher_path = run_dir / "selection_power_teacher_confirmation.json"
    null_path = run_dir / "selection_power_null_confirmation.json"
    teacher = [dict(value) for value in _json_load(teacher_path).get("task_results", [])] if teacher_path.is_file() else []
    null = [dict(value) for value in _json_load(null_path).get("task_results", [])] if null_path.is_file() else []
    return preflight, pilot_power, pilot, confirmation_power, teacher, null


def _workflow_report(
    *,
    provenance: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pilot_panel_power: Mapping[str, Any],
    pilot: Mapping[str, Any],
    confirmation_panel_power: Mapping[str, Any],
    teacher_results: Sequence[Mapping[str, Any]],
    null_results: Sequence[Mapping[str, Any]],
    require_gate: str,
    thresholds: SelectionPowerThresholds,
) -> dict[str, Any]:
    return evaluate_selection_power_workflow(
        provenance=provenance,
        preflight=preflight,
        pilot_panel_power=pilot_panel_power,
        pilot=pilot,
        confirmation_panel_power=confirmation_panel_power,
        teacher_results=teacher_results,
        null_results=null_results,
        require_gate=require_gate,
        thresholds=thresholds,
    )


def _mark_interim_stage_success(report: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(report))
    value["required_gate_pass"] = 1
    value["interim_stage"] = str(stage)
    value["decision"] = {
        "decision": f"selection_power_{stage}_passed",
        "recommended_next_action": (
            "run the fixed 128-path normalized-head pilot"
            if stage == "preflight"
            else "run the fresh oracle-qualified three-seed confirmation"
        ),
        "interim_stage_success": 1,
        "closed_terminal_scientific_outcome": 0,
        "physical_training_authorized": 0,
        "physical_training_performed": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }
    return value


def _save_report(run_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / "selection_power_control_gate.json", report)
    atomic_write_json(run_dir / "selection_power_decision.json", dict(report.get("decision", {})))


def _write_oracle_csv(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []

    def add_record(phase: str, panel: str, record: Mapping[str, Any]) -> None:
        for scope in ("overall", "data_end"):
            scope_value = record.get(scope, {})
            if not isinstance(scope_value, Mapping):
                continue
            rows.append(
                {
                    "phase": phase,
                    "panel": panel,
                    "scope": scope,
                    "path_count": scope_value.get("path_count", record.get("path_count")),
                    "point_estimate": scope_value.get(
                        "point_estimate", scope_value.get("improvement")
                    ),
                    "lower_bound": scope_value.get("lower_bound"),
                    "confidence": scope_value.get("confidence", record.get("confidence")),
                    "passed": record.get("passed"),
                }
            )

    for filename, phase in (
        ("saved_16_path_oracle_forensic.json", "saved-parent"),
        ("oracle_power_calibration.json", "calibration"),
        ("pilot_oracle_feasibility.json", "pilot"),
        ("confirmation_oracle_feasibility.json", "confirmation"),
    ):
        path = run_dir / filename
        if not path.is_file():
            continue
        value = _json_load(path)
        if filename == "oracle_power_calibration.json":
            full = value.get("full", {})
            if isinstance(full, Mapping):
                add_record(phase, "full", full)
            for index, half in enumerate(value.get("halves", [])):
                if isinstance(half, Mapping):
                    add_record(phase, f"half-{index + 1}", half)
            continue
        raw = value.get("raw_evidence", value)
        records = raw.get("records", raw.get("panels", {})) if isinstance(raw, Mapping) else {}
        if not isinstance(records, Mapping):
            continue
        for role, record in records.items():
            if not isinstance(record, Mapping):
                continue
            if "evidence" in record and isinstance(record["evidence"], Mapping):
                add_record(phase, str(role), dict(record["evidence"]))
            else:
                add_record(phase, str(role), record)
    if rows:
        atomic_write_csv(run_dir / "oracle_panel_power.csv", rows)


def _write_task_summary_csv(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("task_result.json")):
        value = _json_load(path)
        metrics = dict(value.get("metrics", {}))
        rows.append(
            {
                "task_path": path.parent.relative_to(run_dir).as_posix(),
                "task": value.get("task"),
                "model_seed": value.get("model_seed"),
                "complete": metrics.get("complete"),
                "finite": metrics.get("finite"),
                "boundary_admissible": metrics.get("boundary_admissible"),
                "nominee_step": metrics.get("nominee_step"),
                "selected_step": metrics.get("selected_step"),
                "post_warmup_clip_fraction": metrics.get("post_warmup_clip_fraction"),
                "final_500_clip_fraction": metrics.get("final_500_clip_fraction"),
                "final_200_clip_fraction": metrics.get("final_200_clip_fraction"),
                "task_gate_pass": dict(value.get("gate", {})).get("passed"),
            }
        )
    if rows:
        atomic_write_csv(run_dir / "selection_power_task_summary.csv", rows)


def _write_learning_plot(run_dir: Path) -> None:
    histories = sorted(run_dir.rglob("training_history.csv"))
    if not histories:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    plotted = 0
    for path in histories:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "step" not in rows[0]:
            continue
        loss_key = next(
            (key for key in ("unscaled_loss", "optimizer_loss", "loss") if key in rows[0]),
            None,
        )
        if loss_key is None:
            continue
        points = [
            (float(row["step"]), float(row[loss_key]))
            for row in rows
            if row.get("step") not in {None, ""} and row.get(loss_key) not in {None, ""}
        ]
        if not points:
            continue
        x, y = zip(*points)
        axis.plot(x, y, alpha=0.65, linewidth=0.9, label=str(path.parent.relative_to(run_dir)))
        plotted += 1
    if not plotted:
        plt.close(figure)
        return
    axis.set_xlabel("optimizer update")
    axis.set_ylabel("training objective")
    axis.set_title("Oracle-qualified density-ratio control learning curves")
    if plotted <= 12:
        axis.legend(fontsize=6)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(run_dir / "selection_power_learning_curves.png", dpi=160)
    plt.close(figure)


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
        _write_oracle_csv(run_dir)
        _write_task_summary_csv(run_dir)
        _write_learning_plot(run_dir)
    except Exception as exc:
        execution_failed = True
        final_skips.append({"stage": "report_artifacts", "reason": f"{type(exc).__name__}: {exc}"})
        atomic_write_json(
            run_dir / "report_artifact_failure.json",
            {"type": type(exc).__name__, "message": str(exc), "physical_training_performed": 0, "sampling_performed": 0},
        )
    required_pass = 0 if execution_failed else int(report.get("required_gate_pass", 0))
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    decision = dict(report.get("decision", {}))
    _write_status(
        run_dir,
        status="failed" if execution_failed else "complete",
        outcome="implementation_error" if execution_failed else ("complete" if required_pass else "gate_failed"),
        phase=phase,
        stage=stage,
        required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=decision.get("decision", "controls_not_run"),
        recommended_next_action=decision.get("recommended_next_action"),
        physical_training_authorized=int(decision.get("physical_training_authorized", 0)) if not execution_failed else 0,
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
        print(f"selection-power density-ratio run directory: {run_dir.resolve()}", flush=True)
    thresholds = SelectionPowerThresholds(head=HeadCoordinateThresholds())
    mutation_started = False
    try:
        device = base._device(args.device)
        backend = configure_exact_torch_backend(device)
        runtime = base._runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
        source_hash, source_paths = _source_record()
        parent = verify_parent_normalized_head_run(args.parent_normalized_head_run_dir)
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
        manifest = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "created_at": _now(),
            "run_dir": str(run_dir.resolve()),
            "scientific_config": scientific,
            "scientific_fingerprint": config_fingerprint(scientific),
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
                "schema", "schema_version", "scientific_config", "scientific_fingerprint",
                "runtime", "runtime_fingerprint", "source_fingerprint", "source_paths",
                "parent_provenance_sha256", "claim_scope",
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
            preflight, pilot_power, pilot, confirmation_power, teacher, null = (
                _load_report_inputs(run_dir)
            )
            report = _workflow_report(
                provenance=provenance, preflight=preflight, pilot=pilot,
                pilot_panel_power=pilot_power,
                confirmation_panel_power=confirmation_power,
                teacher_results=teacher, null_results=null,
                require_gate=str(args.require_gate), thresholds=thresholds,
            )
            if bool(int(pilot.get("passed", 0))) and not teacher and not null:
                report = _mark_interim_stage_success(report, stage="pilot")
            elif bool(int(preflight.get("passed", 0))) and str(pilot.get("evaluation_status")) == "not_evaluated":
                report = _mark_interim_stage_success(report, stage="preflight")
            _save_report(run_dir, report)
            return _finish(run_dir, report=report, stage="report", phase="report")

        dynamics = base._make_dynamics(args)
        horizon = float(natural_horizon(dynamics))
        if not math.isclose(horizon, float(parent.get("horizon", math.nan)), rel_tol=1e-12, abs_tol=1e-15):
            raise ArtifactCompatibilityError("selection-power horizon differs from parent")
        stream_plan = build_density_ratio_stream_plan(
            root_seed=int(args.root_seed), grid_size=int(args.grid_size), horizon=horizon,
            label=3, bin_counts=(4, 4, 4, 4, 16), teacher_epsilon=0.5,
        )
        paired_stream_plan = build_paired_mixture_stream_plan(
            root_seed=int(args.root_seed), grid_size=int(args.grid_size), horizon=horizon,
            label=3, teacher_epsilon=0.5,
        )
        plans = {
            "model_schema": NORMALIZED_HEAD_MODEL_VERSION,
            "head_coordinate_version": NORMALIZED_HEAD_COORDINATE_VERSION,
            "optimizer_coordinate_version": COORDINATE_CONJUGATE_ADAMW_VERSION,
            "evaluation_stream": stream_plan_record(stream_plan),
            "paired_training_stream": paired_mixture_stream_plan_record(paired_stream_plan),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        _freeze_json(run_dir / "selection_power_stream_plans.json", plans)

        _write_status(run_dir, status="running", phase="preflight")
        preflight, calibration_panel = _run_preflight(
            run_dir,
            args=args,
            manifest=manifest,
            provenance=provenance,
            stream_plan=stream_plan,
            thresholds=thresholds,
        )
        pilot = _not_evaluated_gate("selection_power_pilot", "pilot was not run")
        pilot_power = _not_evaluated_gate(
            "pilot_oracle_panel_power", "pilot panels were not frozen"
        )
        confirmation_power = _not_evaluated_gate(
            "confirmation_oracle_panel_power", "confirmation panels were not frozen"
        )
        pilot_multi: dict[str, Any] = {}
        if args.stage == "preflight" or not bool(int(preflight.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance, preflight=preflight, pilot=pilot,
                pilot_panel_power=pilot_power,
                confirmation_panel_power=confirmation_power,
                teacher_results=[], null_results=[], require_gate=str(args.require_gate),
                thresholds=thresholds,
            )
            if args.stage == "preflight" and bool(int(preflight.get("passed", 0))):
                report = _mark_interim_stage_success(report, stage="preflight")
            _save_report(run_dir, report)
            return _finish(
                run_dir, report=report, stage=str(args.stage), phase="preflight",
                skips=[] if bool(int(preflight.get("passed", 0))) else [
                    {"stage": "pilot_and_confirmation", "reason": "oracle-power preflight failed"}
                ],
            )

        selected_profile: dict[str, Any]
        if args.stage in {"pilot", "all"}:
            _write_status(run_dir, status="running", phase="pilot-panel-power")
            pilot, selected_profile, pilot_multi = _run_pilot(
                run_dir,
                args=args,
                manifest=manifest,
                dynamics=dynamics,
                device=device,
                stream_plan=stream_plan,
                paired_stream_plan=paired_stream_plan,
                calibration_panel=calibration_panel,
                thresholds=thresholds,
            )
            pilot_power = _json_load(run_dir / "pilot_oracle_feasibility.json")
            if args.stage == "pilot" or not bool(int(pilot.get("passed", 0))):
                report = _workflow_report(
                    provenance=provenance, preflight=preflight, pilot=pilot,
                    pilot_panel_power=pilot_power,
                    confirmation_panel_power=confirmation_power,
                    teacher_results=[], null_results=[],
                    require_gate=str(args.require_gate), thresholds=thresholds,
                )
                if args.stage == "pilot" and bool(int(pilot.get("passed", 0))):
                    report = _mark_interim_stage_success(report, stage="pilot")
                _save_report(run_dir, report)
                return _finish(
                    run_dir, report=report, stage=str(args.stage), phase="pilot",
                    skips=[] if bool(int(pilot.get("passed", 0))) else [
                        {"stage": "confirmation", "reason": "no oracle-qualified pilot profile"}
                    ],
                )
        else:
            pilot_path = run_dir / "selection_power_pilot_gate.json"
            profile_path = run_dir / "selected_selection_power_profile.json"
            if not pilot_path.is_file() or not profile_path.is_file():
                raise ArtifactCompatibilityError("confirmation requires frozen pilot")
            pilot = _json_load(pilot_path)
            if not bool(int(pilot.get("passed", 0))):
                raise ArtifactCompatibilityError("confirmation requires passing pilot")
            selected_profile = _json_load(profile_path)
            if dict(dict(pilot.get("selected_profile", {})).get("profile", {}) or {}) != selected_profile:
                raise ArtifactCompatibilityError("pilot/profile binding mismatch")
            pilot_multi_path = run_dir / "pilot_null_multiplicity_analysis.json"
            pilot_multi = _json_load(pilot_multi_path) if pilot_multi_path.is_file() else {}
            pilot_power_path = run_dir / "pilot_oracle_feasibility.json"
            if not pilot_power_path.is_file():
                raise ArtifactCompatibilityError("confirmation lacks pilot oracle power")
            pilot_power = _json_load(pilot_power_path)

        _write_status(run_dir, status="running", phase="confirmation-panel-power")
        teacher, null, confirmation_oracle, confirm_multi = _run_confirmation(
            run_dir,
            args=args,
            manifest=manifest,
            dynamics=dynamics,
            device=device,
            stream_plan=stream_plan,
            paired_stream_plan=paired_stream_plan,
            calibration_panel=calibration_panel,
            selected_profile=selected_profile,
            thresholds=thresholds,
        )
        report = _workflow_report(
            provenance=provenance, preflight=preflight, pilot=pilot,
            pilot_panel_power=pilot_power,
            confirmation_panel_power=confirmation_oracle,
            teacher_results=teacher, null_results=null,
            require_gate=str(args.require_gate), thresholds=thresholds,
        )
        if not bool(int(confirmation_oracle.get("passed", 0))):
            report["confirmation_oracle_feasibility"] = confirmation_oracle
        _save_report(run_dir, report)
        return _finish(run_dir, report=report, stage=str(args.stage), phase="confirmation")
    except Exception as exc:
        if resumed and not mutation_started:
            if not args.no_progress:
                print(f"selection-power resume rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        atomic_write_json(
            run_dir / "failure.json",
            {
                "schema": RUN_SCHEMA + "-failure", "schema_version": 1,
                "type": type(exc).__name__, "message": str(exc), "stage": str(args.stage),
                "physical_training_performed": 0, "sampling_performed": 0,
            },
        )
        failed = _not_evaluated_gate("workflow", f"{type(exc).__name__}: {exc}")
        report = _workflow_report(
            provenance=failed, preflight=failed, pilot=failed,
            pilot_panel_power=failed,
            confirmation_panel_power=failed,
            teacher_results=[], null_results=[], require_gate=str(args.require_gate),
            thresholds=thresholds,
        )
        _save_report(run_dir, report)
        _finish(
            run_dir, report=report, stage=str(args.stage), phase="failure",
            execution_failed=True,
            skips=[{"stage": "remaining", "reason": f"{type(exc).__name__}: {exc}"}],
        )
        if not args.no_progress:
            print(f"selection-power control failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
