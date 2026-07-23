"""Multiplicity-corrected confirmation for the synthetic D0 ratio controls.

This additive workflow treats the completed selection-power pilot as immutable
discovery evidence, recovers its predeclared normalized-head profile, and runs
an entirely fresh three-seed confirmation.  Panel A is nomination-only.  The
stationary-null authorization is based on one simultaneous whole-path family
over panels B/C/D, never on the selected panel-A fluctuation.

The workflow is controls-only: it does not import a reverse sampler, train on
physical MNIST score states, or authorize sampling.
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
import mnist.diag_d0_score_density_ratio_selection_power_confirmation as power
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    atomic_copy_file,
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
)
from mnist.d0_score_density_ratio_selection_power_gate import (
    SelectionPowerThresholds,
    evaluate_oracle_panel_set,
    evaluate_power_teacher_study,
)
from mnist.d0_score_density_ratio_sealed_null_gate import (
    MAX_T_VERSION,
    SealedNullThresholds,
    evaluate_confirmation_null_family,
    evaluate_max_t_null_family,
    evaluate_parent_pilot_replay,
    evaluate_sealed_null_workflow,
    evaluate_simultaneous_bootstrap_preflight,
    studentized_whole_path_max_t,
)
from mnist.d0_score_density_ratio_multiplicity_provenance import (
    verify_parent_selection_power_run,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


RUN_SCHEMA = "experiment12-d0-score-density-ratio-multiplicity-confirmation"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = (
    "multiplicity-corrected bounded synthetic density-ratio controls only"
)

EXPECTED_KERNEL = dict(base.EXPECTED_KERNEL)
DEFAULTS: dict[str, Any] = {
    **EXPECTED_KERNEL,
    "root_seed": 260961,
    "selected_learning_rate": 3e-5,
    "accumulation_steps": 8,
    "microbatch_clusters": 32,
    "pilot_steps": 2_000,
    "confirm_steps": 4_000,
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
    "simultaneous_bootstrap_reps": 50_000,
    "familywise_confidence": 0.95,
    "confirm_model_seeds": (260971, 260972, 260973),
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
        "--stage",
        choices=("preflight", "replay", "confirm", "report", "all"),
        default="all",
    )
    parser.add_argument("--parent-selection-power-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_score_density_ratio_multiplicity_confirmation"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-multiplicity-aware-density-ratio-controls"
    )
    parser.add_argument(
        "--require-gate",
        choices=("none", "preflight", "replay", "controls"),
        default="none",
    )
    parser.add_argument("--root-seed", type=int, default=DEFAULTS["root_seed"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")

    for name in (
        "grid_size",
        "sample_steps",
        "reference_substeps",
        "base_channels",
        "validation_batch_size",
        "confirm_steps",
        "confirm_selection_paths",
        "confirm_audit_paths",
        "microbatch_clusters",
        "bootstrap_reps",
        "simultaneous_bootstrap_reps",
        "confirm_validation_every",
        "clip_warmup_steps",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=DEFAULTS[name]
        )
    for name in (
        "tau_eff",
        "alpha_eff",
        "mass_floor",
        "limiter_fraction",
        "lambda_mix",
        "weight_decay",
        "ema_decay",
        "grad_clip",
        "loss_scale",
        "bootstrap_confidence",
        "familywise_confidence",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=DEFAULTS[name]
        )
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff",), default="alpha_eff")
    parser.add_argument(
        "--confirm-model-seeds",
        type=base._parse_csv_ints,
        default=DEFAULTS["confirm_model_seeds"],
    )
    parser.add_argument(
        "--confirm-dense-validation-steps",
        type=base._parse_csv_ints,
        default=DEFAULTS["confirm_dense_validation_steps"],
    )
    args = parser.parse_args(argv)

    if args.stage in {"replay", "confirm", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed_required = {
        "preflight": {"none", "preflight"},
        "replay": {"none", "preflight", "replay"},
        "confirm": {"none", "preflight", "replay", "controls"},
        "all": {"none", "preflight", "replay", "controls"},
        "report": {"none", "preflight", "replay", "controls"},
    }
    if str(args.require_gate) not in allowed_required[str(args.stage)]:
        parser.error(f"--require-gate {args.require_gate} is unavailable at stage {args.stage}")
    for name in (
        "grid_size",
        "sample_steps",
        "reference_substeps",
        "base_channels",
        "validation_batch_size",
        "confirm_steps",
        "confirm_selection_paths",
        "confirm_audit_paths",
        "microbatch_clusters",
        "bootstrap_reps",
        "simultaneous_bootstrap_reps",
        "confirm_validation_every",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not (0.0 < float(args.bootstrap_confidence) < 1.0):
        parser.error("--bootstrap-confidence must lie strictly between zero and one")
    if not (0.0 < float(args.familywise_confidence) < 1.0):
        parser.error("--familywise-confidence must lie strictly between zero and one")
    if int(args.confirm_selection_paths) != int(args.confirm_audit_paths):
        parser.error("confirmation selection and audit panel sizes must match")

    # Compatibility attributes consumed by the unchanged task runner.
    args.pilot_steps = int(DEFAULTS["pilot_steps"])
    args.pilot_validation_steps = tuple(DEFAULTS["pilot_validation_steps"])
    args.accumulation_levels = (int(DEFAULTS["accumulation_steps"]),)
    args.pilot_learning_rates = (float(DEFAULTS["selected_learning_rate"]),)

    if args.require_gate != "none":
        frozen = {
            key: DEFAULTS[key]
            for key in (
                *EXPECTED_KERNEL,
                "root_seed",
                "base_channels",
                "validation_batch_size",
                "confirm_steps",
                "confirm_selection_paths",
                "confirm_audit_paths",
                "microbatch_clusters",
                "bootstrap_reps",
                "bootstrap_confidence",
                "simultaneous_bootstrap_reps",
                "familywise_confidence",
                "weight_decay",
                "ema_decay",
                "grad_clip",
                "clip_warmup_steps",
                "loss_scale",
                "confirm_model_seeds",
                "confirm_dense_validation_steps",
                "confirm_validation_every",
            )
        }
        changed = [
            name
            for name, expected in frozen.items()
            if getattr(args, name) != expected
        ]
        if changed:
            parser.error(
                "production required gates reject overrides: " + ", ".join(changed)
            )
    return args


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{stamp}_{args.run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir.resolve(), False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    value = _json_load(path) if path.is_file() else {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": _now(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    value.update(updates)
    value["updated_at"] = _now()
    value["physical_training_performed"] = 0
    value["sampling_performed"] = 0
    atomic_write_json(path, value)
    return value


def _freeze_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _json_load(path) != normalized:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    else:
        atomic_write_json(path, normalized)
    return normalized


def _source_record() -> tuple[str, list[str]]:
    local = [
        Path(__file__).resolve(),
        Path(sys.modules[verify_parent_selection_power_run.__module__].__file__).resolve(),
        Path(sys.modules[studentized_whole_path_max_t.__module__].__file__).resolve(),
    ]
    _, inherited = power._source_record()
    paths: list[Path] = []
    for path in local + [Path(value).resolve() for value in inherited]:
        if path.is_file() and path not in paths:
            paths.append(path)
    return source_fingerprint(paths), [str(path) for path in paths]


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": RUN_SCHEMA_VERSION,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "records": records,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _verify_terminal_registry(run_dir: Path) -> dict[str, Any]:
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    if not registry_path.is_file() or not status_path.is_file():
        raise ArtifactCompatibilityError("terminal run lacks registry or status")
    registry = _json_load(registry_path)
    status = _json_load(status_path)
    if (
        status.get("artifact_registry_sha256") != file_fingerprint(registry_path)
        or int(status.get("artifact_registry_size", -1)) != registry_path.stat().st_size
    ):
        raise ArtifactCompatibilityError("terminal artifact registry binding changed")
    records = registry.get("records", {})
    if not isinstance(records, Mapping):
        raise ArtifactCompatibilityError("terminal artifact registry records are invalid")
    for relative, record in records.items():
        path = run_dir / str(relative)
        if (
            not path.is_file()
            or not isinstance(record, Mapping)
            or record.get("sha256") != file_fingerprint(path)
            or int(record.get("size", -1)) != path.stat().st_size
        ):
            raise ArtifactCompatibilityError(f"terminal artifact mismatch: {relative}")
    return registry


def _scientific_config(
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    thresholds: SealedNullThresholds,
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
        "parent_replay": {
            "selection_rule": "lowest-mean-teacher-a-b-bce-then-clip-then-lr",
            "expected_learning_rates": [3e-5, 1e-5],
            "selected_learning_rate": float(DEFAULTS["selected_learning_rate"]),
            "accumulation_steps": int(DEFAULTS["accumulation_steps"]),
            "parent_all_evidence_is_discovery": 1,
            "optimizer_steps_performed": 0,
        },
        "simultaneous_bootstrap": {
            "version": MAX_T_VERSION,
            "bootstrap_replicates": int(args.simultaneous_bootstrap_reps),
            "one_sided_familywise_confidence": float(args.familywise_confidence),
            "quantile_method": "higher",
            "studentized": 1,
            "path_clustered": 1,
            "confirmation_family": {
                "model_seeds": list(args.confirm_model_seeds),
                "panel_roles": ["b", "c", "d"],
                "scopes": ["overall", "data_end"],
                "expected_member_count": 18,
            },
            "panel_a_authorizing": 0,
        },
        "confirmation": {
            "steps": int(args.confirm_steps),
            "model_seeds": list(args.confirm_model_seeds),
            "paths_per_selection_panel": int(args.confirm_selection_paths),
            "paths_per_audit_panel": int(args.confirm_audit_paths),
            "anchors_per_path": 32,
            "time_bin_counts": [4, 4, 4, 4, 16],
            "dense_validation_steps": list(args.confirm_dense_validation_steps),
            "validation_every": int(args.confirm_validation_every),
            "oracle_before_optimizer": 1,
            "panel_regeneration_after_inspection": 0,
        },
        "optimization": {
            "optimizer": "coordinate-conjugate-AdamW",
            "body_learning_rate": float(DEFAULTS["selected_learning_rate"]),
            "body_weight_decay": float(args.weight_decay),
            "head_lr_factor": int(args.grid_size) ** 2,
            "head_eps_factor": 1.0 / float(int(args.grid_size) ** 2),
            "head_weight_decay_factor": 1.0 / float(int(args.grid_size) ** 2),
            "loss_scale": float(args.loss_scale),
            "ema_decay": float(args.ema_decay),
            "grad_clip": float(args.grad_clip),
            "clip_warmup_steps": int(args.clip_warmup_steps),
            "accumulation_steps": int(DEFAULTS["accumulation_steps"]),
            "microbatch_clusters": int(args.microbatch_clusters),
            "adaptive_loss_scaling": 0,
        },
        "legacy_panel_bootstrap": {
            "replicates": int(args.bootstrap_reps),
            "confidence": float(args.bootstrap_confidence),
        },
        "thresholds": thresholds.to_dict(),
        "parent_scientific_fingerprint": parent.get("scientific_fingerprint"),
        "parent_artifact_registry_sha256": parent.get(
            "artifact_registry_sha256"
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _panel_registry(
    *,
    panels: Mapping[str, Mapping[str, DensityRatioPanel]],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    flat = [panel for group in panels.values() for panel in group.values()]

    def streams(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, Mapping):
            raw = value.get("stream_fingerprints")
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                found.update(str(item) for item in raw)
            for child in value.values():
                found.update(streams(child))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                found.update(streams(child))
        return found

    parent_registry_path = Path(str(parent["run_dir"])) / "pilot_panel_registry.json"
    parent_registry = _json_load(parent_registry_path)
    parent_streams = streams(parent_registry)
    fresh_streams = {
        str(value)
        for panel in flat
        for value in panel.stream_fingerprints
    }
    overlap = sorted(parent_streams.intersection(fresh_streams))
    return {
        "schema": RUN_SCHEMA + "-panel-registry",
        "schema_version": RUN_SCHEMA_VERSION,
        "phase": "multiplicity-confirmation",
        "panels": {
            task: {role: panel_identity(panel) for role, panel in group.items()}
            for task, group in panels.items()
        },
        "disjointness": panel_disjointness_record(flat),
        "parent_panel_registry_sha256": file_fingerprint(parent_registry_path),
        "parent_stream_fingerprint_count": len(parent_streams),
        "fresh_stream_fingerprint_count": len(fresh_streams),
        "parent_overlap_path_count": len(overlap),
        "parent_overlap_stream_fingerprints": overlap,
        "parent_confirmation_isolation_pass": int(bool(parent_streams) and not overlap),
        "panel_regeneration_after_inspection": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _prepare_confirmation_panels(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    stream_plan: DensityRatioStreamPlan,
) -> tuple[dict[str, dict[str, DensityRatioPanel]], dict[str, Any]]:
    panels = {
        task: power._prepare_panel_set(
            run_dir,
            phase="multiplicity-confirmation",
            task=task,
            roles=("a", "b", "c", "d"),
            path_count=int(args.confirm_selection_paths),
            stream_plan=stream_plan,
            scientific_fingerprint=str(manifest["scientific_fingerprint"]),
            start_offset=20_000_000 + task_index * 2_000_000,
        )
        for task_index, task in enumerate(("bounded_teacher", "dirichlet_null"))
    }
    registry = _panel_registry(panels=panels, parent=parent)
    if not bool(int(dict(registry["disjointness"]).get("passed", 0))):
        raise ArtifactCompatibilityError("fresh A/B/C/D panels overlap")
    if not bool(int(registry["parent_confirmation_isolation_pass"])):
        raise ArtifactCompatibilityError("fresh panels overlap the immutable parent")
    _freeze_json(run_dir / "confirmation_panel_registry.json", registry)
    return panels, registry


def _oracle_panel_bundle(
    panels: Mapping[str, DensityRatioPanel],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = {
        role: evaluate_oracle_panel_feasibility(
            panel,
            confidence=float(args.bootstrap_confidence),
            reps=int(args.bootstrap_reps),
            seed=base._derived_seed(
                int(args.root_seed), "multiplicity-confirmation-oracle", role
            ),
        )
        for role, panel in panels.items()
    }
    raw = {
        "schema": RUN_SCHEMA + "-oracle-panel-feasibility",
        "schema_version": RUN_SCHEMA_VERSION,
        "panels": records,
        "pairwise_disjoint": int(
            bool(int(panel_disjointness_record(list(panels.values())).get("passed", 0)))
        ),
        "calibration_overlap_path_count": 0,
        "frozen_before_training": 1,
        "optimizer_steps_before_oracle_gate": 0,
        "panel_regeneration_after_inspection": 0,
        "regenerated_after_inspection": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    gate = evaluate_oracle_panel_set(raw, expected_roles=("a", "b", "c", "d"))
    gate["raw_evidence"] = raw
    return gate


def _task_fingerprints(
    *,
    manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    profile_binding: Mapping[str, Any],
    oracle_path: Path,
    task: str,
    model_seed: int,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
    panels: Mapping[str, DensityRatioPanel],
) -> dict[str, Any]:
    value = head._task_fingerprints(
        manifest=manifest,
        phase="multiplicity-confirmation",
        task=task,
        model_seed=int(model_seed),
        learning_rate=float(DEFAULTS["selected_learning_rate"]),
        accumulation_level=int(DEFAULTS["accumulation_steps"]),
        stream_plan=stream_plan,
        paired_stream_plan=paired_stream_plan,
        selection_panels={name: panels[name] for name in ("a", "b")},
        audit_panels={name: panels[name] for name in ("c", "d")},
        profile_binding=profile_binding,
    )
    value.update(
        {
            "multiplicity_workflow_schema": RUN_SCHEMA,
            "multiplicity_workflow_schema_version": RUN_SCHEMA_VERSION,
            "simultaneous_bootstrap_version": MAX_T_VERSION,
            "simultaneous_family_fingerprint": config_fingerprint(
                dict(manifest["scientific_config"])["simultaneous_bootstrap"]
            ),
            "parent_registry_sha256": parent["artifact_registry_sha256"],
            "parent_registry_size": int(parent["artifact_registry_size"]),
            "oracle_feasibility_sha256": file_fingerprint(oracle_path),
            "oracle_feasibility_size": int(oracle_path.stat().st_size),
        }
    )
    return value


def _bootstrap_member(
    panel_record: Mapping[str, Any],
    *,
    scope: str,
    name: str,
    block: str,
    role: str,
) -> tuple[str, dict[str, Any]]:
    scope_record = panel_record.get(scope)
    if not isinstance(scope_record, Mapping):
        raise ArtifactCompatibilityError(f"{name} lacks scope {scope}")
    bootstrap = scope_record.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ArtifactCompatibilityError(f"{name} lacks whole-path bootstrap data")
    path_ids = bootstrap.get("path_ids")
    path_values = bootstrap.get("path_values")
    if (
        not isinstance(path_ids, Sequence)
        or isinstance(path_ids, (str, bytes))
        or not isinstance(path_values, Sequence)
        or isinstance(path_values, (str, bytes))
    ):
        raise ArtifactCompatibilityError(f"{name} has invalid whole-path data")
    return name, {
        "resampling_block": str(block),
        "panel_role": str(role),
        "scope": str(scope),
        "path_ids": list(path_ids),
        "path_values": [float(value) for value in path_values],
    }


def _parent_candidate_bundle(
    parent: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    parent_dir = Path(str(parent["run_dir"]))
    pilot = _json_load(parent_dir / "selection_power_pilot_gate.json")
    raw_candidates = pilot.get("candidate_records")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raise ArtifactCompatibilityError("parent pilot candidate records are missing")
    candidates: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    discovery_members: dict[str, dict[str, Any]] = {}
    sealed_b_members: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ArtifactCompatibilityError("parent candidate is not a record")
        candidate = copy.deepcopy(dict(raw))
        gate = candidate.get("gate")
        if not isinstance(gate, Mapping):
            raise ArtifactCompatibilityError("parent candidate gate is missing")
        for key in (
            "optimizer_health_pass",
            "teacher_mean_ab_bce",
            "maximum_clip_fraction_observed",
        ):
            candidate[key] = gate.get(key)
        candidates.append(candidate)

        null_result = candidate.get("null")
        if not isinstance(null_result, Mapping):
            raise ArtifactCompatibilityError("parent null task result is missing")
        task_dir = parent_dir / "pilot" / f"lr-{index:02d}" / "dirichlet_null"
        disk_result_path = task_dir / "task_result.json"
        sealed_path = task_dir / "sealed_panel_b.json"
        if _json_load(disk_result_path) != dict(null_result):
            raise ArtifactCompatibilityError("embedded and on-disk parent null task differ")
        sealed = _json_load(sealed_path)
        metrics = dict(null_result.get("metrics", {}))
        nominee_step = int(metrics.get("nominee_step", 0))
        checkpoint_path = task_dir / "checkpoints" / f"step-{nominee_step:08d}.pt"
        validations = [
            dict(value)
            for value in metrics.get("checkpoints", [])
            if isinstance(value, Mapping)
        ]
        nominee_rows = [
            value for value in validations if int(value.get("step", -1)) == nominee_step
        ]
        panel_record = sealed.get("panel_record")
        bound_panel = (
            dict(dict(nominee_rows[0].get("panels", {})).get("b", {}))
            if len(nominee_rows) == 1
            else {}
        )
        binding_pass = (
            nominee_step > 0
            and len(nominee_rows) == 1
            and checkpoint_path.is_file()
            and sealed.get("task") == "dirichlet_null"
            and sealed.get("phase") == "selection-power-pilot"
            and int(sealed.get("nominee_step", -1)) == nominee_step
            and sealed.get("nominee_checkpoint_sha256")
            == file_fingerprint(checkpoint_path)
            and isinstance(panel_record, Mapping)
            and dict(panel_record) == bound_panel
        )
        binding = {
            "evaluation_status": "evaluated",
            "passed": int(binding_pass),
            "candidate_index": index,
            "learning_rate": candidate.get("learning_rate"),
            "nominee_step": nominee_step,
            "task_result_sha256": file_fingerprint(disk_result_path),
            "sealed_panel_b_sha256": file_fingerprint(sealed_path),
            "nominee_checkpoint_sha256": (
                file_fingerprint(checkpoint_path) if checkpoint_path.is_file() else None
            ),
            "panel_fingerprint": (
                dict(panel_record).get("panel_fingerprint")
                if isinstance(panel_record, Mapping)
                else None
            ),
            "sealed_panel_b_evaluation_count": 1,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        bindings.append(binding)

        for validation in validations:
            step = int(validation.get("step", -1))
            if step <= 0:
                continue
            panel_a = dict(dict(validation.get("panels", {})).get("a", {}))
            for scope in ("overall", "data_end"):
                name = f"lr-{index:02d}/step-{step:08d}/a/{scope}"
                member_name, member = _bootstrap_member(
                    panel_a,
                    scope=scope,
                    name=name,
                    block="parent-panel-a",
                    role="a",
                )
                discovery_members[member_name] = member
        if not isinstance(panel_record, Mapping):
            raise ArtifactCompatibilityError("parent sealed B panel record is missing")
        for scope in ("overall", "data_end"):
            name = f"lr-{index:02d}/b/{scope}"
            member_name, member = _bootstrap_member(
                panel_record,
                scope=scope,
                name=name,
                block="parent-panel-b",
                role="b",
            )
            sealed_b_members[member_name] = member
    return candidates, bindings, discovery_members, sealed_b_members


def _advisory_family(record: Mapping[str, Any], *, gate_name: str) -> dict[str, Any]:
    return {
        "gate": str(gate_name),
        "evaluation_status": "evaluated",
        "passed": 1,
        "authorizing": 0,
        "advisory_familywise_excursion": int(
            bool(int(record.get("familywise_false_discovery", 0)))
        ),
        "max_t_record": dict(record),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _run_preflight(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    thresholds: SealedNullThresholds,
) -> dict[str, Any]:
    members = {
        "shared/a": {
            "resampling_block": "shared",
            "panel_role": "b",
            "scope": "overall",
            "path_ids": list(range(8)),
            "path_values": [-0.7, -0.2, -0.4, -0.1, -0.5, -0.3, -0.8, -0.6],
        },
        "shared/end": {
            "resampling_block": "shared",
            "panel_role": "b",
            "scope": "data_end",
            "path_ids": list(range(8)),
            "path_values": [-0.35, -0.1, -0.2, -0.05, -0.25, -0.15, -0.4, -0.3],
        },
        "independent/zero": {
            "resampling_block": "independent",
            "panel_role": "c",
            "scope": "overall",
            "path_ids": list(range(8)),
            "path_values": [0.0] * 8,
        },
    }
    first = studentized_whole_path_max_t(
        members,
        seed=base._derived_seed(int(args.root_seed), "max-t-self-test"),
        confidence=float(args.familywise_confidence),
        reps=int(args.simultaneous_bootstrap_reps),
    )
    reordered = {
        name: {
            **dict(members[name]),
            "path_ids": list(reversed(list(members[name]["path_ids"]))),
            "path_values": list(reversed(list(members[name]["path_values"]))),
        }
        for name in reversed(tuple(members))
    }
    second = studentized_whole_path_max_t(
        reordered,
        seed=base._derived_seed(int(args.root_seed), "max-t-self-test"),
        confidence=float(args.familywise_confidence),
        reps=int(args.simultaneous_bootstrap_reps),
    )
    _, _, _, parent_b_members = _parent_candidate_bundle(parent)
    parent_forensic = studentized_whole_path_max_t(
        parent_b_members,
        seed=base._derived_seed(int(args.root_seed), "preflight-parent-b-family"),
        confidence=float(args.familywise_confidence),
        reps=int(args.simultaneous_bootstrap_reps),
    )
    _freeze_json(run_dir / "parent_b_bootstrap_preflight_forensic.json", parent_forensic)
    first_rows = {row["name"]: row for row in first["members"]}
    reference = first_rows["shared/a"]
    expected_t = float(reference["mean"]) / float(reference["standard_error"])
    metrics = {
        "schema": RUN_SCHEMA + "-bootstrap-preflight-metrics",
        "schema_version": RUN_SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": int(bool(int(first["finite"])) and bool(int(parent_forensic["finite"]))),
        "version": MAX_T_VERSION,
        "deterministic_replay_pass": int(first == studentized_whole_path_max_t(
            members,
            seed=base._derived_seed(int(args.root_seed), "max-t-self-test"),
            confidence=float(args.familywise_confidence),
            reps=int(args.simultaneous_bootstrap_reps),
        )),
        "member_order_invariance_pass": int(first["members"] == second["members"]),
        "path_order_invariance_pass": int(
            first["family_fingerprint"] == second["family_fingerprint"]
            and first["critical_value"] == second["critical_value"]
        ),
        "shared_block_coupling_pass": int(first["resampling_block_count"] == 2),
        "disjoint_block_stream_pass": int(first["resampling_block_count"] == 2),
        "studentization_reference_pass": int(
            math.isclose(float(reference["t_statistic"]), expected_t, rel_tol=1e-12, abs_tol=1e-12)
        ),
        "simultaneous_coverage_fixture_pass": int(
            not bool(int(first["familywise_false_discovery"]))
        ),
        "whole_path_only_pass": int(
            all(int(row["path_count"]) == 8 for row in first["members"])
        ),
        "parent_family_coverage_pass": int(
            int(parent_forensic["family_size"]) == 4
            and all(int(row["path_count"]) == 128 for row in parent_forensic["members"])
        ),
        "self_test_record": first,
        "parent_family_record": parent_forensic,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    _freeze_json(run_dir / "simultaneous_bootstrap_preflight.json", metrics)
    gate = evaluate_simultaneous_bootstrap_preflight(metrics)
    _freeze_json(run_dir / "multiplicity_preflight_gate.json", gate)
    return gate


def _write_family_csv(path: Path, phase: str, record: Mapping[str, Any]) -> None:
    rows = [
        {
            "phase": phase,
            **dict(member),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        for member in record.get("members", [])
        if isinstance(member, Mapping)
    ]
    if rows:
        atomic_write_csv(path, rows)


def _run_replay(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    parent: Mapping[str, Any],
    thresholds: SealedNullThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates, bindings, discovery_members, sealed_b_members = (
        _parent_candidate_bundle(parent)
    )
    discovery_record = studentized_whole_path_max_t(
        discovery_members,
        seed=base._derived_seed(int(args.root_seed), "parent-discovery-a-family"),
        confidence=float(args.familywise_confidence),
        reps=int(args.simultaneous_bootstrap_reps),
    )
    sealed_b_record = studentized_whole_path_max_t(
        sealed_b_members,
        seed=base._derived_seed(int(args.root_seed), "parent-sealed-b-family"),
        confidence=float(args.familywise_confidence),
        reps=int(args.simultaneous_bootstrap_reps),
    )
    discovery_gate = _advisory_family(
        discovery_record, gate_name="parent_discovery_a_max_t_advisory"
    )
    sealed_b_gate = evaluate_max_t_null_family(
        sealed_b_record,
        expected_members=tuple(sorted(sealed_b_members)),
        expected_member_count=4,
        required_confidence=float(args.familywise_confidence),
        required_replicates=int(args.simultaneous_bootstrap_reps),
        gate_name="parent_sealed_b_simultaneous_null_family",
    )
    replay = evaluate_parent_pilot_replay(
        candidates,
        sealed_b_bindings=bindings,
        discovery_family=discovery_gate,
        sealed_b_family=sealed_b_gate,
        thresholds=thresholds,
    )
    replay.update(
        {
            "parent_registry_sha256": parent["artifact_registry_sha256"],
            "parent_registry_size": int(parent["artifact_registry_size"]),
            "selection_rule": "lowest-mean-teacher-a-b-bce-then-clip-then-lr",
            "parent_all_evidence_is_discovery": 1,
            "optimizer_steps_performed": 0,
        }
    )
    _freeze_json(run_dir / "parent_discovery_a_max_t.json", discovery_record)
    _freeze_json(run_dir / "parent_sealed_b_max_t.json", sealed_b_record)
    _freeze_json(run_dir / "parent_sealed_b_bindings.json", {
        "bindings": bindings,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    })
    _freeze_json(run_dir / "multiplicity_replay_gate.json", replay)
    _write_family_csv(
        run_dir / "parent_discovery_a_simultaneous_bounds.csv",
        "parent-discovery-a",
        discovery_record,
    )
    _write_family_csv(
        run_dir / "parent_sealed_b_simultaneous_bounds.csv",
        "parent-sealed-b",
        sealed_b_record,
    )

    wrapper = replay.get("selected_profile")
    raw_profile = dict(wrapper.get("profile", {})) if isinstance(wrapper, Mapping) else {}
    if bool(int(replay.get("passed", 0))):
        if (
            not math.isclose(
                float(raw_profile.get("body_learning_rate", math.nan)),
                float(DEFAULTS["selected_learning_rate"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or int(raw_profile.get("accumulation_steps", -1))
            != int(DEFAULTS["accumulation_steps"])
        ):
            raise ArtifactCompatibilityError("parent replay selected an unexpected profile")
        selected_index = int(raw_profile["candidate_index"])
        selected_binding = bindings[selected_index]
        profile = {
            **raw_profile,
            "schema": RUN_SCHEMA + "-selected-profile",
            "schema_version": RUN_SCHEMA_VERSION,
            "selection_rule": "lowest-mean-teacher-a-b-bce-then-clip-then-lr",
            "parent_all_evidence_is_discovery": 1,
            "parent_registry_sha256": parent["artifact_registry_sha256"],
            "parent_registry_size": int(parent["artifact_registry_size"]),
            "parent_replay_gate_sha256": file_fingerprint(
                run_dir / "multiplicity_replay_gate.json"
            ),
            "selected_parent_evidence": selected_binding,
            "parent_weights_reused": 0,
            "parent_states_reused_for_confirmation": 0,
            "optimizer_steps_performed": 0,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        profile = _freeze_json(run_dir / "selected_multiplicity_profile.json", profile)
    else:
        profile = {}
    return replay, profile


def _profile_binding(
    run_dir: Path,
    *,
    profile: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    profile_path = run_dir / "selected_multiplicity_profile.json"
    replay_path = run_dir / "multiplicity_replay_gate.json"
    if not profile_path.is_file() or _json_load(profile_path) != dict(profile):
        raise ArtifactCompatibilityError("confirmation profile is not frozen")
    replay = _json_load(replay_path)
    selected = dict(dict(replay.get("selected_profile", {})).get("profile", {}) or {})
    for key in ("body_learning_rate", "accumulation_steps", "candidate_index"):
        if selected.get(key) != profile.get(key):
            raise ArtifactCompatibilityError(f"profile/replay binding mismatch for {key}")
    return _freeze_json(
        run_dir / "confirmation_profile_binding.json",
        {
            "schema": RUN_SCHEMA + "-confirmation-profile-binding",
            "schema_version": RUN_SCHEMA_VERSION,
            "selected_profile": dict(profile),
            "selected_profile_sha256": file_fingerprint(profile_path),
            "multiplicity_replay_gate_sha256": file_fingerprint(replay_path),
            "parent_registry_sha256": parent["artifact_registry_sha256"],
            "fresh_initialization_required": 1,
            "parent_checkpoint_loading_permitted": 0,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )


def _confirmation_sealed_binding(
    task_dir: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = dict(result.get("metrics", {}))
    nominee_step = int(metrics.get("nominee_step", 0))
    checkpoint_path = task_dir / "checkpoints" / f"step-{nominee_step:08d}.pt"
    sealed_path = task_dir / "sealed_panel_b.json"
    result_path = task_dir / "task_result.json"
    if not sealed_path.is_file() or not result_path.is_file():
        passed = False
        sealed: dict[str, Any] = {}
    else:
        sealed = _json_load(sealed_path)
        passed = _json_load(result_path) == dict(result)
    validations = [
        dict(value)
        for value in metrics.get("checkpoints", [])
        if isinstance(value, Mapping)
    ]
    rows = [value for value in validations if int(value.get("step", -1)) == nominee_step]
    recorded_b = (
        dict(dict(rows[0].get("panels", {})).get("b", {})) if len(rows) == 1 else {}
    )
    sealed_b = sealed.get("panel_record") if isinstance(sealed, Mapping) else None
    fingerprints = dict(result.get("fingerprints", {}))
    passed = bool(
        passed
        and nominee_step > 0
        and len(rows) == 1
        and checkpoint_path.is_file()
        and sealed.get("nominee_checkpoint_sha256") == file_fingerprint(checkpoint_path)
        and sealed.get("nominee_step") == nominee_step
        and sealed.get("fingerprints") == fingerprints
        and isinstance(sealed_b, Mapping)
        and dict(sealed_b) == recorded_b
    )
    return {
        "evaluation_status": "evaluated" if sealed_path.is_file() else "not_evaluated",
        "passed": int(passed),
        "task": result.get("task"),
        "model_seed": result.get("model_seed"),
        "nominee_step": nominee_step if nominee_step > 0 else None,
        "task_result_sha256": file_fingerprint(result_path) if result_path.is_file() else None,
        "sealed_panel_b_sha256": file_fingerprint(sealed_path) if sealed_path.is_file() else None,
        "nominee_checkpoint_sha256": (
            file_fingerprint(checkpoint_path) if checkpoint_path.is_file() else None
        ),
        "panel_fingerprint": (
            dict(sealed_b).get("panel_fingerprint")
            if isinstance(sealed_b, Mapping)
            else None
        ),
        "sealed_panel_b_evaluation_count": int(sealed_path.is_file()),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _confirmation_family_members(
    null_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    discovery: dict[str, dict[str, Any]] = {}
    confirmatory: dict[str, dict[str, Any]] = {}
    for raw in null_results:
        metrics = dict(raw.get("metrics", {}))
        model_seed = int(metrics.get("model_seed", raw.get("model_seed", -1)))
        nominee_step = int(metrics.get("nominee_step", 0))
        validations = [
            dict(value)
            for value in metrics.get("checkpoints", [])
            if isinstance(value, Mapping)
        ]
        nominee_rows = [
            value for value in validations if int(value.get("step", -1)) == nominee_step
        ]
        if len(nominee_rows) != 1:
            raise ArtifactCompatibilityError(
                f"null seed {model_seed} has no unique panel-A nominee"
            )
        for validation in validations:
            step = int(validation.get("step", -1))
            if step <= 0:
                continue
            panel_a = dict(dict(validation.get("panels", {})).get("a", {}))
            for scope in ("overall", "data_end"):
                name = f"seed-{model_seed}/step-{step:08d}/a/{scope}"
                key, member = _bootstrap_member(
                    panel_a,
                    scope=scope,
                    name=name,
                    block="confirmation-panel-a",
                    role="a",
                )
                discovery[key] = member
        panel_b = dict(dict(nominee_rows[0].get("panels", {})).get("b", {}))
        audits = dict(metrics.get("audit_panels", {}))
        for role, panel_record in (
            ("b", panel_b),
            ("c", dict(audits.get("c", {}))),
            ("d", dict(audits.get("d", {}))),
        ):
            for scope in ("overall", "data_end"):
                name = f"seed-{model_seed}/{role}/{scope}"
                key, member = _bootstrap_member(
                    panel_record,
                    scope=scope,
                    name=name,
                    block=f"confirmation-panel-{role}",
                    role=role,
                )
                confirmatory[key] = member
    return discovery, confirmatory


def _run_confirmation(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    parent: Mapping[str, Any],
    profile: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig,
    device: torch.device,
    stream_plan: DensityRatioStreamPlan,
    paired_stream_plan: PairedMixtureStreamPlan,
    thresholds: SealedNullThresholds,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    binding = _profile_binding(run_dir, profile=profile, parent=parent)
    panels, _ = _prepare_confirmation_panels(
        run_dir,
        args=args,
        manifest=manifest,
        parent=parent,
        stream_plan=stream_plan,
    )
    oracle_path = run_dir / "confirmation_oracle_feasibility.json"
    if oracle_path.is_file():
        oracle = _json_load(oracle_path)
        recomputed = _oracle_panel_bundle(panels["bounded_teacher"], args=args)
        if oracle != recomputed:
            raise ArtifactCompatibilityError("frozen oracle feasibility changed")
    else:
        if (run_dir / "confirmation").exists():
            raise ArtifactCompatibilityError(
                "optimizer artifacts exist before oracle feasibility was frozen"
            )
        oracle = _oracle_panel_bundle(panels["bounded_teacher"], args=args)
        _freeze_json(oracle_path, oracle)
    if not bool(int(oracle.get("passed", 0))):
        not_run = _not_evaluated_gate(
            "sealed_null_confirmation_family",
            "fixed confirmation panels failed exact-teacher oracle power",
        )
        _freeze_json(run_dir / "confirmation_null_family_gate.json", not_run)
        return [], [], oracle, not_run, _not_evaluated_gate(
            "selection_power_teacher_study", "confirmation training was skipped"
        )

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
                parent=parent,
                profile_binding=binding,
                oracle_path=oracle_path,
                task=task,
                model_seed=int(model_seed),
                stream_plan=stream_plan,
                paired_stream_plan=paired_stream_plan,
                panels=panels[task],
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
                        learning_rate=float(DEFAULTS["selected_learning_rate"]),
                        accumulation_level=int(DEFAULTS["accumulation_steps"]),
                    ),
                    device=device,
                    model_seed=int(model_seed),
                    learning_rate=float(DEFAULTS["selected_learning_rate"]),
                    accumulation_level=int(DEFAULTS["accumulation_steps"]),
                    stream_plan=stream_plan,
                    paired_stream_plan=paired_stream_plan,
                    fingerprints=fingerprints,
                    phase="multiplicity-confirmation",
                    thresholds=thresholds.selection_power.head,
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
        run_dir / "multiplicity_teacher_confirmation.json",
        {
            "task_results": teacher_results,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "multiplicity_null_confirmation.json",
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

    bindings = [
        _confirmation_sealed_binding(
            run_dir / "confirmation" / f"seed-{int(result.get('model_seed'))}" / "dirichlet_null",
            result,
        )
        for result in null_results
    ]
    _freeze_json(
        run_dir / "confirmation_sealed_b_bindings.json",
        {
            "bindings": bindings,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    try:
        discovery_members, family_members = _confirmation_family_members(null_results)
        discovery_record = studentized_whole_path_max_t(
            discovery_members,
            seed=base._derived_seed(int(args.root_seed), "confirmation-discovery-a"),
            confidence=float(args.familywise_confidence),
            reps=int(args.simultaneous_bootstrap_reps),
        )
        family_record = studentized_whole_path_max_t(
            family_members,
            seed=base._derived_seed(int(args.root_seed), "confirmation-b-c-d"),
            confidence=float(args.familywise_confidence),
            reps=int(args.simultaneous_bootstrap_reps),
        )
        expected_names = [
            f"seed-{int(seed)}/{role}/{scope}"
            for seed in args.confirm_model_seeds
            for role in ("b", "c", "d")
            for scope in ("overall", "data_end")
        ]
        max_t_gate = evaluate_max_t_null_family(
            family_record,
            expected_members=expected_names,
            expected_member_count=18,
            required_confidence=float(args.familywise_confidence),
            required_replicates=int(args.simultaneous_bootstrap_reps),
            gate_name="confirmation_b_c_d_simultaneous_null_family",
        )
        _freeze_json(run_dir / "confirmation_discovery_a_max_t.json", discovery_record)
        _freeze_json(run_dir / "confirmation_b_c_d_max_t.json", family_record)
        _write_family_csv(
            run_dir / "confirmation_discovery_a_simultaneous_bounds.csv",
            "confirmation-discovery-a",
            discovery_record,
        )
        _write_family_csv(
            run_dir / "confirmation_null_simultaneous_bounds.csv",
            "confirmation-b-c-d",
            family_record,
        )
    except (ArtifactCompatibilityError, ValueError) as exc:
        max_t_gate = _not_evaluated_gate(
            "confirmation_b_c_d_simultaneous_null_family",
            f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(
            run_dir / "confirmation_multiplicity_failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )
    null_family = evaluate_confirmation_null_family(
        null_results,
        max_t_family=max_t_gate,
        sealed_b_bindings=bindings,
        thresholds=thresholds,
    )
    teacher_study = evaluate_power_teacher_study(
        teacher_results, thresholds.selection_power
    )
    _freeze_json(run_dir / "confirmation_null_family_gate.json", null_family)
    _freeze_json(run_dir / "confirmation_teacher_study_gate.json", teacher_study)
    return teacher_results, null_results, oracle, null_family, teacher_study


def _load_report_inputs(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    def gate(filename: str, name: str) -> dict[str, Any]:
        path = run_dir / filename
        return _json_load(path) if path.is_file() else _not_evaluated_gate(
            name, f"{name} was not run"
        )

    return (
        gate("multiplicity_preflight_gate.json", "simultaneous_bootstrap_preflight"),
        gate("multiplicity_replay_gate.json", "sealed_null_parent_pilot_replay"),
        gate("confirmation_oracle_feasibility.json", "confirmation_oracle_panel_power"),
        gate("confirmation_teacher_study_gate.json", "selection_power_teacher_study"),
        gate("confirmation_null_family_gate.json", "sealed_null_confirmation_family"),
    )


def _workflow_report(
    *,
    provenance: Mapping[str, Any],
    preflight: Mapping[str, Any],
    replay: Mapping[str, Any],
    confirmation_power: Mapping[str, Any],
    teacher_study: Mapping[str, Any],
    null_family: Mapping[str, Any],
    require_gate: str,
    thresholds: SealedNullThresholds,
) -> dict[str, Any]:
    return evaluate_sealed_null_workflow(
        provenance=provenance,
        simultaneous_bootstrap=preflight,
        replay=replay,
        confirmation_panel_power=confirmation_power,
        teacher_study=teacher_study,
        null_family=null_family,
        require_gate=require_gate,
        thresholds=thresholds,
    )


def _save_report(run_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / "multiplicity_control_gate.json", report)
    atomic_write_json(
        run_dir / "multiplicity_decision.json", dict(report.get("decision", {}))
    )


def _write_task_summary_csv(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("task_result.json")):
        value = _json_load(path)
        metrics = dict(value.get("metrics", {}))
        selection = dict(metrics.get("selection", {}))
        confirmation = dict(selection.get("confirmation", {}))
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
                "sealed_b_accepted": confirmation.get("accepted"),
                "post_warmup_clip_fraction": metrics.get("post_warmup_clip_fraction"),
                "final_500_clip_fraction": metrics.get("final_500_clip_fraction"),
                "final_200_clip_fraction": metrics.get("final_200_clip_fraction"),
                "legacy_task_gate_pass_advisory": dict(value.get("gate", {})).get(
                    "passed"
                ),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
    if rows:
        atomic_write_csv(run_dir / "multiplicity_task_summary.csv", rows)


def _write_learning_plot(run_dir: Path) -> None:
    power._write_learning_plot(run_dir)
    source = run_dir / "selection_power_learning_curves.png"
    target = run_dir / "multiplicity_learning_curves.png"
    if source.is_file():
        atomic_copy_file(source, target)


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
        _write_task_summary_csv(run_dir)
        _write_learning_plot(run_dir)
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
            "implementation_error"
            if execution_failed
            else "complete"
            if required_pass
            else "gate_failed"
        ),
        phase=phase,
        stage=stage,
        required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=decision.get("decision", "controls_not_run"),
        recommended_next_action=decision.get("recommended_next_action"),
        h1_function_step_patch_authorized=int(
            decision.get("h1_function_step_patch_authorized", 0)
        ),
        physical_training_authorized=(
            int(decision.get("physical_training_authorized", 0))
            if not execution_failed
            else 0
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
        print(f"multiplicity density-ratio run directory: {run_dir.resolve()}", flush=True)
    thresholds = SealedNullThresholds(
        selection_power=SelectionPowerThresholds(head=HeadCoordinateThresholds()),
        confidence=float(args.familywise_confidence),
        bootstrap_replicates=int(args.simultaneous_bootstrap_reps),
    )
    mutation_started = False
    provenance: dict[str, Any] = _not_evaluated_gate(
        "multiplicity_parent_provenance", "parent provenance was not verified"
    )
    try:
        device = base._device(args.device)
        backend = configure_exact_torch_backend(device)
        runtime = base._runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
        source_hash, source_paths = _source_record()
        verified_parent = verify_parent_selection_power_run(
            args.parent_selection_power_run_dir
        )
        provenance = {
            **dict(verified_parent),
            "schema": RUN_SCHEMA + "-parent-provenance",
            "schema_version": RUN_SCHEMA_VERSION,
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

        scientific = _scientific_config(args, verified_parent, thresholds)
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
                "schema",
                "schema_version",
                "run_dir",
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
            preflight, replay, confirmation_power, teacher_study, null_family = (
                _load_report_inputs(run_dir)
            )
            report = _workflow_report(
                provenance=provenance,
                preflight=preflight,
                replay=replay,
                confirmation_power=confirmation_power,
                teacher_study=teacher_study,
                null_family=null_family,
                require_gate=str(args.require_gate),
                thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(run_dir, report=report, stage="report", phase="report")

        dynamics = base._make_dynamics(args)
        horizon = float(natural_horizon(dynamics))
        if not math.isclose(
            horizon,
            float(verified_parent.get("horizon", math.nan)),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ArtifactCompatibilityError("multiplicity horizon differs from parent")
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
        _freeze_json(
            run_dir / "multiplicity_stream_plans.json",
            {
                "schema": RUN_SCHEMA + "-stream-plans",
                "schema_version": RUN_SCHEMA_VERSION,
                "evaluation_stream": stream_plan_record(stream_plan),
                "paired_training_stream": paired_mixture_stream_plan_record(
                    paired_stream_plan
                ),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )

        _write_status(run_dir, status="running", phase="preflight")
        preflight = _run_preflight(
            run_dir,
            args=args,
            parent=verified_parent,
            thresholds=thresholds,
        )
        replay = _not_evaluated_gate(
            "sealed_null_parent_pilot_replay", "parent replay was not run"
        )
        confirmation_power = _not_evaluated_gate(
            "confirmation_oracle_panel_power", "confirmation panels were not frozen"
        )
        teacher_study = _not_evaluated_gate(
            "selection_power_teacher_study", "confirmation was not run"
        )
        null_family = _not_evaluated_gate(
            "sealed_null_confirmation_family", "confirmation was not run"
        )
        if args.stage == "preflight" or not bool(int(preflight.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance,
                preflight=preflight,
                replay=replay,
                confirmation_power=confirmation_power,
                teacher_study=teacher_study,
                null_family=null_family,
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
                    else [{"stage": "replay_and_confirmation", "reason": "preflight failed"}]
                ),
            )

        _write_status(run_dir, status="running", phase="parent-replay")
        replay, profile = _run_replay(
            run_dir,
            args=args,
            parent=verified_parent,
            thresholds=thresholds,
        )
        if args.stage == "replay" or not bool(int(replay.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance,
                preflight=preflight,
                replay=replay,
                confirmation_power=confirmation_power,
                teacher_study=teacher_study,
                null_family=null_family,
                require_gate=str(args.require_gate),
                thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(
                run_dir,
                report=report,
                stage=str(args.stage),
                phase="replay",
                skips=(
                    []
                    if bool(int(replay.get("passed", 0)))
                    else [{"stage": "confirmation", "reason": "profile replay failed"}]
                ),
            )

        _write_status(run_dir, status="running", phase="confirmation")
        _, _, confirmation_power, null_family, teacher_study = _run_confirmation(
            run_dir,
            args=args,
            manifest=manifest,
            parent=verified_parent,
            profile=profile,
            dynamics=dynamics,
            device=device,
            stream_plan=stream_plan,
            paired_stream_plan=paired_stream_plan,
            thresholds=thresholds,
        )
        report = _workflow_report(
            provenance=provenance,
            preflight=preflight,
            replay=replay,
            confirmation_power=confirmation_power,
            teacher_study=teacher_study,
            null_family=null_family,
            require_gate=str(args.require_gate),
            thresholds=thresholds,
        )
        _save_report(run_dir, report)
        return _finish(
            run_dir,
            report=report,
            stage=str(args.stage),
            phase="confirmation",
            skips=(
                []
                if bool(int(confirmation_power.get("passed", 0)))
                else [
                    {
                        "stage": "confirmation_training",
                        "reason": "fixed confirmation panels failed oracle power",
                    }
                ]
            ),
        )
    except Exception as exc:
        if resumed and not mutation_started:
            if not args.no_progress:
                print(
                    f"multiplicity resume rejected: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            return 2
        atomic_write_json(
            run_dir / "failure.json",
            {
                "schema": RUN_SCHEMA + "-failure",
                "schema_version": RUN_SCHEMA_VERSION,
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": str(args.stage),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )
        failed = _not_evaluated_gate("workflow", f"{type(exc).__name__}: {exc}")
        report = _workflow_report(
            provenance=provenance,
            preflight=failed,
            replay=failed,
            confirmation_power=failed,
            teacher_study=failed,
            null_family=failed,
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
                f"multiplicity control failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
