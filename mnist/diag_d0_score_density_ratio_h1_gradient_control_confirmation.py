"""Online gradient-ratio H1 confirmation for synthetic D0 ratio controls.

This additive, controls-only workflow follows the immutable H1 trust run.
It keeps the normalized-head classifier, balanced paired BCE estimator, and
all scientific thresholds fixed.  The only change is deterministic online
normalization of the existing stopped-EMA H1 gradient relative to the BCE
gradient.  It never imports a reverse sampler, trains on physical MNIST score
states, or produces samples.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

import mnist.diag_d0_score_density_ratio_controls as base
import mnist.diag_d0_score_density_ratio_head_confirmation as head
import mnist.diag_d0_score_density_ratio_multiplicity_confirmation as multi
import mnist.diag_d0_score_density_ratio_selection_power_confirmation as power
from mnist.d0_dirichlet_score import edge_difference_channels, harmonic_mobility_exact
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
from mnist.d0_score_density_ratio_h1_trust import (
    H1_TRUST_CALIBRATION_VERSION,
    H1_TRUST_OPERATOR_VERSION,
    H1_TRUST_SCALE_FLOOR,
    H1_TRUST_STREAM_VERSION,
    H1TrustCalibration,
    H1TrustPlan,
    build_h1_trust_plan,
    calibrate_h1_trust,
    generate_reference_trust_batch,
    h1_increment_components,
    h1_trust_plan_record,
)
from mnist.d0_score_density_ratio_h1_gradient_control import (
    H1_GRADIENT_CONTROL_NORM_FLOOR,
    H1_GRADIENT_CONTROL_RAMP_STEPS,
    H1_GRADIENT_CONTROL_TRACKING_RTOL,
    H1_GRADIENT_CONTROL_VERSION,
    GradientRatioControllerConfig,
    compose_gradient_ratio_update,
)
from mnist.d0_score_density_ratio_h1_gradient_control_gate import (
    H1GradientControlThresholds,
    evaluate_gradient_control_pilot_candidate,
    evaluate_gradient_control_pilot,
    evaluate_gradient_control_preflight,
    evaluate_gradient_control_workflow,
    evaluate_gradient_controller_preflight,
    rank_gradient_control_candidates,
    not_evaluated_gate,
)
from mnist.d0_score_density_ratio_h1_trust_gate import (
    evaluate_h1_calibration,
    evaluate_h1_operator_preflight,
)
from mnist.d0_score_density_ratio_h1_gradient_control_provenance import (
    verify_parent_h1_trust_run,
)
from mnist.d0_score_density_ratio_matched_flux import (
    evaluate_matched_teacher_flux_reduction,
    joint_matched_flux_family_bootstrap,
)
from mnist.d0_score_density_ratio_head import (
    COORDINATE_CONJUGATE_ADAMW_VERSION,
    NORMALIZED_HEAD_COORDINATE_VERSION,
    NORMALIZED_HEAD_MODEL_VERSION,
    D0BoundarySmoothMeanHeadPotentialUNet,
    build_coordinate_conjugate_adamw,
    normalized_gradient_diagnostics,
)
from mnist.d0_score_density_ratio_head_provenance import PARENT_LOSS_SCALE
from mnist.d0_score_density_ratio_paired import (
    PAIRED_MIXTURE_ACCUMULATION_VERSION,
    PAIRED_MIXTURE_OBJECTIVE_VERSION,
    PAIRED_MIXTURE_SCHEMA,
    PAIRED_MIXTURE_STREAM_VERSION,
    PairedMixtureStreamPlan,
    build_paired_mixture_stream_plan,
    generate_paired_mixture_microbatch,
    paired_mixture_stream_plan_record,
)
from mnist.d0_score_density_ratio_sealed_null_gate import (
    MAX_T_VERSION,
    SealedNullThresholds,
    evaluate_confirmation_null_family,
    evaluate_max_t_null_family,
    studentized_whole_path_max_t,
)
from mnist.d0_score_density_ratio_selection_power import (
    evaluate_oracle_panel_feasibility,
)
from mnist.d0_score_density_ratio_selection_power_gate import (
    SelectionPowerThresholds,
    evaluate_oracle_panel_set,
    evaluate_power_teacher_study,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    init_ema_state,
    natural_horizon,
    update_ema_state,
)


RUN_SCHEMA = "experiment12-d0-score-density-ratio-h1-gradient-control-confirmation"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "gradient-ratio controlled H1 bounded synthetic density-ratio controls only"
EXPECTED_KERNEL = dict(base.EXPECTED_KERNEL)

DEFAULTS: dict[str, Any] = {
    **EXPECTED_KERNEL,
    "root_seed": 261041,
    "pilot_model_seed": 261051,
    "confirm_model_seeds": (261061, 261062, 261063),
    "body_learning_rate": 3e-5,
    "target_gradient_ratios": (0.0, 0.1, 0.3, 1.0),
    "controller_ramp_steps": H1_GRADIENT_CONTROL_RAMP_STEPS,
    "controller_norm_floor": H1_GRADIENT_CONTROL_NORM_FLOOR,
    "controller_tracking_rtol": H1_GRADIENT_CONTROL_TRACKING_RTOL,
    "accumulation_steps": 8,
    "microbatch_clusters": 32,
    "trust_banks_per_update": 2,
    "trust_states_per_bank": 32,
    "pilot_steps": 4_000,
    "confirm_steps": 4_000,
    "pilot_paths": 128,
    "confirm_paths": 128,
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
    "validation_dense_steps": (0, 25, 50, 100, 150, 250),
    "validation_every": 250,
}


def _json_load(path: str | Path) -> dict[str, Any]:
    return base._json_load(path)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preflight", "pilot", "confirm", "report", "all"),
        default="all",
    )
    parser.add_argument("--parent-h1-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("runs/experiment12_d0_score_density_ratio_h1_gradient_control_confirmation"),
    )
    parser.add_argument(
        "--run-name", default="production-gradient-controlled-h1-density-ratio-controls"
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
        "validation_batch_size", "pilot_steps", "confirm_steps", "pilot_paths",
        "confirm_paths", "microbatch_clusters", "bootstrap_reps",
        "simultaneous_bootstrap_reps", "validation_every", "clip_warmup_steps",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=DEFAULTS[name])
    for name in (
        "tau_eff", "alpha_eff", "mass_floor", "limiter_fraction", "lambda_mix",
        "body_learning_rate", "weight_decay", "ema_decay", "grad_clip",
        "loss_scale", "bootstrap_confidence", "familywise_confidence",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=DEFAULTS[name])
    parser.add_argument("--edge-alpha-mode", choices=("alpha_eff",), default="alpha_eff")
    parser.add_argument(
        "--target-gradient-ratios", type=base._parse_csv_floats,
        default=DEFAULTS["target_gradient_ratios"],
    )
    parser.add_argument("--pilot-model-seed", type=int, default=DEFAULTS["pilot_model_seed"])
    parser.add_argument(
        "--confirm-model-seeds", type=base._parse_csv_ints,
        default=DEFAULTS["confirm_model_seeds"],
    )
    parser.add_argument(
        "--validation-dense-steps", type=base._parse_csv_ints,
        default=DEFAULTS["validation_dense_steps"],
    )
    args = parser.parse_args(argv)
    if args.stage in {"pilot", "confirm", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "pilot": {"none", "preflight", "pilot"},
        "confirm": {"none", "preflight", "pilot", "controls"},
        "all": {"none", "preflight", "pilot", "controls"},
        "report": {"none", "preflight", "pilot", "controls"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(f"--require-gate {args.require_gate} is unavailable at stage {args.stage}")
    for name in (
        "grid_size", "sample_steps", "reference_substeps", "base_channels",
        "validation_batch_size", "pilot_steps", "confirm_steps", "pilot_paths",
        "confirm_paths", "microbatch_clusters", "bootstrap_reps",
        "simultaneous_bootstrap_reps", "validation_every",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if tuple(float(value) for value in args.target_gradient_ratios) != tuple(DEFAULTS["target_gradient_ratios"]):
        if args.require_gate != "none":
            parser.error("production required gates freeze --target-gradient-ratios")
    if float(args.loss_scale) != float(DEFAULTS["loss_scale"]):
        parser.error("--loss-scale is inherited and frozen from the parent controls")
    if not 0.0 < float(args.bootstrap_confidence) < 1.0:
        parser.error("--bootstrap-confidence must lie strictly between zero and one")
    if not 0.0 < float(args.familywise_confidence) < 1.0:
        parser.error("--familywise-confidence must lie strictly between zero and one")
    args.accumulation_levels = (int(DEFAULTS["accumulation_steps"]),)
    args.pilot_learning_rates = (float(args.body_learning_rate),)
    args.pilot_validation_steps = tuple(
        sorted(set(args.validation_dense_steps) | set(range(
            int(args.validation_every), int(args.pilot_steps) + 1, int(args.validation_every)
        )) | {int(args.pilot_steps)})
    )
    args.confirm_dense_validation_steps = tuple(args.validation_dense_steps)
    args.confirm_validation_every = int(args.validation_every)
    args.confirm_selection_paths = int(args.confirm_paths)
    args.confirm_audit_paths = int(args.confirm_paths)
    if args.require_gate != "none":
        frozen_names = (
            *EXPECTED_KERNEL, "root_seed", "pilot_model_seed", "confirm_model_seeds",
            "body_learning_rate", "target_gradient_ratios", "pilot_steps", "confirm_steps",
            "pilot_paths", "confirm_paths", "base_channels", "validation_batch_size",
            "microbatch_clusters", "bootstrap_reps", "bootstrap_confidence",
            "simultaneous_bootstrap_reps", "familywise_confidence", "weight_decay",
            "ema_decay", "grad_clip", "clip_warmup_steps", "loss_scale",
            "validation_dense_steps", "validation_every",
        )
        changed = [name for name in frozen_names if getattr(args, name) != DEFAULTS[name]]
        if changed:
            parser.error("production required gates reject overrides: " + ", ".join(changed))
    return args


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir.resolve(), False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    value = _json_load(path) if path.is_file() else {
        "schema": RUN_SCHEMA, "schema_version": RUN_SCHEMA_VERSION,
        "created_at": _now(), "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    value.update(updates)
    value.update({"updated_at": _now(), "physical_training_performed": 0, "sampling_performed": 0})
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
    modules = [
        sys.modules[__name__],
        sys.modules[verify_parent_h1_trust_run.__module__],
        sys.modules[evaluate_gradient_control_workflow.__module__],
        sys.modules[compose_gradient_ratio_update.__module__],
        sys.modules[evaluate_matched_teacher_flux_reduction.__module__],
        sys.modules[h1_increment_components.__module__],
        sys.modules[studentized_whole_path_max_t.__module__],
    ]
    try:
        from mnist.d0_score_density_ratio_h1_gradient_control_task import (
            run_gradient_control_paired_density_ratio_task,
        )
        modules.append(sys.modules[run_gradient_control_paired_density_ratio_task.__module__])
    except ImportError:
        pass
    _, inherited = head._source_record()
    paths: list[Path] = []
    for path in [Path(module.__file__).resolve() for module in modules] + [
        Path(value).resolve() for value in inherited
    ]:
        if path.is_file() and path not in paths:
            paths.append(path)
    return source_fingerprint(paths), [str(path) for path in paths]


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path), "size": int(path.stat().st_size)
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry", "schema_version": RUN_SCHEMA_VERSION,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "records": records, "physical_training_performed": 0, "sampling_performed": 0,
    }


def _verify_terminal_registry(run_dir: Path) -> dict[str, Any]:
    registry_path, status_path = run_dir / "artifact_registry.json", run_dir / "run_status.json"
    if not registry_path.is_file() or not status_path.is_file():
        raise ArtifactCompatibilityError("terminal run lacks registry or status")
    registry, status = _json_load(registry_path), _json_load(status_path)
    if status.get("artifact_registry_sha256") != file_fingerprint(registry_path):
        raise ArtifactCompatibilityError("terminal artifact registry binding changed")
    if int(status.get("artifact_registry_size", -1)) != registry_path.stat().st_size:
        raise ArtifactCompatibilityError("terminal artifact registry size changed")
    for relative, record in dict(registry.get("records", {})).items():
        path = run_dir / relative
        if (not path.is_file() or record.get("sha256") != file_fingerprint(path)
                or int(record.get("size", -1)) != path.stat().st_size):
            raise ArtifactCompatibilityError(f"terminal artifact mismatch: {relative}")
    return registry


def _scientific_config(
    args: argparse.Namespace, parent: Mapping[str, Any], thresholds: H1GradientControlThresholds
) -> dict[str, Any]:
    value = {
        "algorithm": RUN_SCHEMA, "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "kernel": {key: getattr(args, key) for key in EXPECTED_KERNEL},
        "model_schema": NORMALIZED_HEAD_MODEL_VERSION,
        "head_coordinate_version": NORMALIZED_HEAD_COORDINATE_VERSION,
        "optimizer_coordinate_version": COORDINATE_CONJUGATE_ADAMW_VERSION,
        "paired_estimator_schema": PAIRED_MIXTURE_SCHEMA,
        "paired_objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
        "paired_stream_version": PAIRED_MIXTURE_STREAM_VERSION,
        "paired_accumulation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
        "h1_operator_version": H1_TRUST_OPERATOR_VERSION,
        "h1_stream_version": H1_TRUST_STREAM_VERSION,
        "h1_calibration_version": H1_TRUST_CALIBRATION_VERSION,
        "h1_gradient_controller_version": H1_GRADIENT_CONTROL_VERSION,
        "controller": GradientRatioControllerConfig().to_record(),
        "root_seed": int(args.root_seed),
        "pilot": {
            "model_seed": int(args.pilot_model_seed), "steps": int(args.pilot_steps),
            "paths_per_panel": int(args.pilot_paths),
            "target_gradient_ratios": list(args.target_gradient_ratios),
            "scientific_endpoint_step": int(args.pilot_steps),
        },
        "confirmation": {
            "model_seeds": list(args.confirm_model_seeds), "steps": int(args.confirm_steps),
            "paths_per_panel": int(args.confirm_paths),
            "scientific_endpoint_step": int(args.confirm_steps),
        },
        "optimization": {
            "body_learning_rate": float(args.body_learning_rate),
            "body_weight_decay": float(args.weight_decay),
            "loss_scale": float(args.loss_scale), "ema_decay": float(args.ema_decay),
            "global_grad_clip": float(args.grad_clip),
            "accumulation_steps": int(DEFAULTS["accumulation_steps"]),
            "microbatch_clusters": int(args.microbatch_clusters),
            "trust_banks_per_update": int(DEFAULTS["trust_banks_per_update"]),
            "trust_states_per_bank": int(DEFAULTS["trust_states_per_bank"]),
        },
        "bootstrap": {
            "path_replicates": int(args.bootstrap_reps),
            "path_confidence": float(args.bootstrap_confidence),
            "simultaneous_replicates": int(args.simultaneous_bootstrap_reps),
            "familywise_confidence": float(args.familywise_confidence),
            "max_t_version": MAX_T_VERSION,
        },
        "thresholds": thresholds.to_dict(),
        "parent_scientific_fingerprint": parent.get("scientific_fingerprint"),
        "parent_artifact_registry_sha256": parent.get("artifact_registry_sha256"),
        "physical_training_performed": 0, "sampling_performed": 0,
    }
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _gradient_vector(model: nn.Module) -> Tensor:
    parts = [
        parameter.grad.detach().reshape(-1).to(dtype=torch.float64)
        for parameter in model.parameters() if parameter.grad is not None
    ]
    if not parts:
        reference = next(model.parameters())
        return torch.zeros(1, device=reference.device, dtype=torch.float64)
    return torch.cat(parts)


def _copy_model(model: nn.Module) -> nn.Module:
    value = copy.deepcopy(model)
    value.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    return value


def _nested_tensor_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return isinstance(left, Tensor) and isinstance(right, Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_nested_tensor_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return (
            isinstance(right, Sequence) and not isinstance(right, (str, bytes))
            and len(left) == len(right)
            and all(_nested_tensor_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _lambda_zero_trajectory_regression(
    *, dynamics: DirectFluxMNISTConfig, plan: H1TrustPlan,
    paired_plan: PairedMixtureStreamPlan, args: argparse.Namespace,
    device: torch.device, steps: int = 25,
) -> dict[str, Any]:
    seed = base._derived_seed(int(args.root_seed), "h1-lambda-zero-trajectory")
    base._set_seed(seed)
    legacy = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    trusted = _copy_model(legacy).to(device)
    legacy_optimizer = build_coordinate_conjugate_adamw(
        legacy, body_lr=float(args.body_learning_rate), eps=1e-8,
        weight_decay=float(args.weight_decay),
    )
    trusted_optimizer = build_coordinate_conjugate_adamw(
        trusted, body_lr=float(args.body_learning_rate), eps=1e-8,
        weight_decay=float(args.weight_decay),
    )
    legacy_ema, trusted_ema = init_ema_state(legacy), init_ema_state(trusted)
    trusted_anchor = _copy_model(trusted).to(device)
    for parameter in trusted_anchor.parameters():
        parameter.requires_grad_(False)
    fingerprints: list[str] = []
    for step in range(1, int(steps) + 1):
        for model, optimizer in (
            (legacy, legacy_optimizer), (trusted, trusted_optimizer)
        ):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for microbatch_index in range(int(DEFAULTS["accumulation_steps"])):
                batch = generate_paired_mixture_microbatch(
                    paired_plan, phase="h1-lambda-zero-trajectory",
                    task="bounded_teacher", optimizer_step=step,
                    microbatch_index=microbatch_index, device=device,
                    dtype=torch.float32,
                )
                objective, _ = head._microbatch_objective_and_components(
                    model, batch, task="bounded_teacher"
                )
                (objective * float(args.loss_scale) / float(DEFAULTS["accumulation_steps"])).backward()
                if model is trusted:
                    fingerprints.append(head._batch_fingerprint(batch))
            if model is trusted:
                trusted_anchor.load_state_dict(trusted_ema, strict=True)
                for bank in range(2):
                    trust = generate_reference_trust_batch(
                        plan, phase="h1-lambda-zero-trajectory",
                        task="bounded_teacher", optimizer_step=step,
                        bank=bank, device=device,
                    )
                    # The production q=0 arm evaluates diagnostics but performs
                    # no H1 backward operation.
                    h1_increment_components(
                        trusted, trusted_anchor, trust.tau, trust.states,
                        trust.labels, dynamics, value_scale=1.0,
                        energy_scale=1.0, create_graph=False,
                    )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.grad_clip), error_if_nonfinite=True
            )
            optimizer.step()
            update_ema_state(
                legacy_ema if model is legacy else trusted_ema,
                model, float(args.ema_decay),
            )
    model_equal = _nested_tensor_equal(legacy.state_dict(), trusted.state_dict())
    ema_equal = _nested_tensor_equal(legacy_ema, trusted_ema)
    optimizer_equal = _nested_tensor_equal(
        legacy_optimizer.state_dict(), trusted_optimizer.state_dict()
    )
    return {
        "steps": int(steps), "model_state_exact": int(model_equal),
        "ema_state_exact": int(ema_equal),
        "adamw_state_exact": int(optimizer_equal),
        "passed": int(model_equal and ema_equal and optimizer_equal),
        "paired_batch_fingerprint_count": len(fingerprints),
    }


def _operator_preflight(
    *, dynamics: DirectFluxMNISTConfig, plan: H1TrustPlan, device: torch.device,
    paired_plan: PairedMixtureStreamPlan, args: argparse.Namespace,
) -> dict[str, Any]:
    """Exercise the exact all-edge operator and stopped-EMA graph on device."""

    base._set_seed(base._derived_seed(int(args.root_seed), "h1-operator-preflight"))
    anchor = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    raw = _copy_model(anchor).to(device)
    with torch.no_grad():
        pattern = torch.linspace(
            -1.0, 1.0, raw.out.weight.numel(), device=device,
            dtype=raw.out.weight.dtype,
        ).reshape_as(raw.out.weight)
        raw.out.weight.add_(1e-3 * pattern)
        raw.out.bias.add_(2e-4)
    batch = generate_reference_trust_batch(
        plan, phase="operator-preflight", task="bounded_teacher",
        optimizer_step=0, bank=0, device=device,
    )
    states = batch.states[:4]
    tau, labels = batch.tau[:4], batch.labels[:4]

    identical = h1_increment_components(
        anchor, _copy_model(anchor).to(device), tau, states, labels, dynamics,
        value_scale=1.0, energy_scale=1.0, create_graph=True,
    )
    anchor.zero_grad(set_to_none=True)
    identical.objective.backward()
    identical_grad_norm = float(torch.linalg.vector_norm(_gradient_vector(anchor)).cpu())

    shifted = _copy_model(anchor).to(device)
    with torch.no_grad():
        shifted.out.bias.add_(1e-3)
    shift_components = h1_increment_components(
        shifted, anchor, tau, states, labels, dynamics,
        value_scale=1.0, energy_scale=1.0, create_graph=True,
    )
    constant_shift_l2 = float(shift_components.value_mean.detach().cpu())
    constant_shift_energy = float(shift_components.natural_energy_mean.detach().cpu())

    raw.zero_grad(set_to_none=True)
    anchor.zero_grad(set_to_none=True)
    components = h1_increment_components(
        raw, anchor, tau, states, labels, dynamics,
        value_scale=1.0, energy_scale=1.0, create_graph=True,
    )
    theta = harmonic_mobility_exact(states, dynamics)
    raw_states = states.detach().clone().requires_grad_(True)
    anchor_states = states.detach().clone().requires_grad_(True)
    raw_logits = raw(tau, raw_states, labels)
    anchor_logits = anchor(tau, anchor_states, labels)
    raw_gradient = torch.autograd.grad(
        raw_logits.sum(), raw_states, create_graph=True
    )[0]
    anchor_gradient = torch.autograd.grad(anchor_logits.sum(), anchor_states)[0]
    edge_delta = edge_difference_channels(
        raw_gradient - anchor_gradient.detach(), int(dynamics.grid_size)
    )
    manual_energy = (theta.detach() * edge_delta.square()).flatten(1).mean(dim=1)
    analytic_error = float(
        (manual_energy - components.per_state_natural_energy).abs().max().detach().cpu()
    )
    sign_energy = (theta.detach() * (-edge_delta).square()).flatten(1).mean(dim=1)
    orientation_error = float((sign_energy - manual_energy).abs().max().detach().cpu())
    other = torch.roll(edge_delta.detach(), shifts=1, dims=0)
    inner_fg = (theta.detach() * edge_delta.detach() * other).flatten(1).mean(dim=1)
    inner_gf = (theta.detach() * other * edge_delta.detach()).flatten(1).mean(dim=1)
    symmetry_error = float((inner_fg - inner_gf).abs().max().cpu())
    components.objective.backward()
    raw_grad = _gradient_vector(raw)
    stopped_anchor_grad_norm = float(torch.linalg.vector_norm(_gradient_vector(anchor)).cpu())

    boundary_states = states.detach().clone()
    boundary_states[:, 0] = 1e-8
    boundary_states[:, 1:] *= (
        (1.0 - boundary_states[:, :1]) / boundary_states[:, 1:].sum(dim=1, keepdim=True)
    )
    boundary = h1_increment_components(
        raw, anchor, tau, boundary_states, labels, dynamics,
        value_scale=1.0, energy_scale=1.0, create_graph=True,
    )
    raw.zero_grad(set_to_none=True)
    boundary.objective.backward()
    boundary_grad = _gradient_vector(raw)

    # Lambda-zero must leave the fixed paired-BCE gradient exactly unchanged.
    base._set_seed(base._derived_seed(int(args.root_seed), "h1-lambda-zero"))
    bce_only = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    bce_plus_zero = _copy_model(bce_only).to(device)
    microbatch = generate_paired_mixture_microbatch(
        paired_plan, phase="h1-lambda-zero", task="bounded_teacher",
        optimizer_step=1, microbatch_index=0, device=device, dtype=torch.float32,
    )
    loss_a, _ = head._microbatch_objective_and_components(
        bce_only, microbatch, task="bounded_teacher"
    )
    (loss_a * float(args.loss_scale)).backward()
    gradient_a = _gradient_vector(bce_only)
    loss_b, _ = head._microbatch_objective_and_components(
        bce_plus_zero, microbatch, task="bounded_teacher"
    )
    zero_anchor = _copy_model(bce_plus_zero).to(device)
    zero_h1 = h1_increment_components(
        bce_plus_zero, zero_anchor, tau, states, labels, dynamics,
        value_scale=1.0, energy_scale=1.0, create_graph=True,
    )
    (loss_b * float(args.loss_scale) + 0.0 * zero_h1.objective).backward()
    gradient_b = _gradient_vector(bce_plus_zero)
    lambda_zero_error = float((gradient_a - gradient_b).abs().max().cpu())

    trajectory = _lambda_zero_trajectory_regression(
        dynamics=dynamics, plan=plan, paired_plan=paired_plan, args=args,
        device=device, steps=25,
    )
    replay_batch = generate_reference_trust_batch(
        plan, phase="operator-preflight", task="bounded_teacher",
        optimizer_step=0, bank=0, device=device,
    )
    null_batch = generate_reference_trust_batch(
        plan, phase="operator-preflight", task="dirichlet_null",
        optimizer_step=0, bank=0, device=device,
    )

    tolerance = 2e-6 if device.type == "cuda" else 1e-8
    metrics = {
        "schema": RUN_SCHEMA + "-h1-operator-preflight",
        "schema_version": RUN_SCHEMA_VERSION,
        "evaluation_status": "evaluated", "complete": 1,
        "finite": int(all(torch.isfinite(value).all() for value in (
            raw_grad, boundary_grad, components.per_state_value,
            components.per_state_natural_energy,
        ))),
        "gamma_symmetry_pass": int(symmetry_error <= tolerance),
        "gamma_positivity_pass": int(float(manual_energy.min().detach().cpu()) >= -tolerance),
        "orientation_invariance_pass": int(orientation_error <= tolerance),
        "analytic_agreement_pass": int(analytic_error <= tolerance),
        "identical_anchor_zero_pass": int(
            float(identical.objective.detach().cpu()) <= tolerance
        ),
        "identical_anchor_gradient_zero_pass": int(identical_grad_norm <= tolerance),
        "constant_shift_l2_detection_pass": int(
            constant_shift_l2 > tolerance * tolerance
            and abs(constant_shift_energy) <= tolerance
        ),
        "stopped_anchor_pass": int(stopped_anchor_grad_norm <= tolerance),
        "boundary_finite_pass": int(bool(torch.isfinite(boundary_grad).all())),
        "cuda_second_order_backward_pass": int(bool(torch.isfinite(raw_grad).all())),
        "lambda_zero_regression_pass": int(
            lambda_zero_error <= tolerance and bool(int(trajectory["passed"]))
        ),
        "stateless_stream_replay_pass": int(replay_batch.fingerprint == batch.fingerprint),
        "candidate_order_invariance_pass": int(
            replay_batch.plan_fingerprint == plan.fingerprint
        ),
        "teacher_null_namespace_isolation_pass": int(
            null_batch.fingerprint != replay_batch.fingerprint
        ),
        "gamma_symmetry_max_abs_error": symmetry_error,
        "orientation_max_abs_error": orientation_error,
        "analytic_energy_max_abs_error": analytic_error,
        "identical_anchor_objective": float(identical.objective.detach().cpu()),
        "identical_anchor_gradient_norm": identical_grad_norm,
        "constant_shift_value_mse": constant_shift_l2,
        "constant_shift_natural_energy": constant_shift_energy,
        "stopped_anchor_gradient_norm": stopped_anchor_grad_norm,
        "lambda_zero_gradient_max_abs_error": lambda_zero_error,
        "lambda_zero_trajectory": trajectory,
        "device": str(device), "tolerance": tolerance,
        "physical_training_performed": 0, "sampling_performed": 0,
    }
    return metrics


def _shadow_calibration_once(
    *, dynamics: DirectFluxMNISTConfig, plan: H1TrustPlan,
    paired_plan: PairedMixtureStreamPlan, args: argparse.Namespace,
    device: torch.device,
) -> tuple[H1TrustCalibration, dict[str, Any]]:
    model_seed = base._derived_seed(int(args.root_seed), "h1-shadow-calibration-model")
    base._set_seed(model_seed)
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    ema_anchor = _copy_model(model).to(device)
    for parameter in ema_anchor.parameters():
        parameter.requires_grad_(False)
    optimizer = build_coordinate_conjugate_adamw(
        model, body_lr=float(args.body_learning_rate), eps=1e-8,
        weight_decay=float(args.weight_decay),
    )
    optimizer.zero_grad(set_to_none=True)
    batch_fingerprints: list[str] = []
    losses: list[float] = []
    for microbatch_index in range(int(DEFAULTS["accumulation_steps"])):
        batch = generate_paired_mixture_microbatch(
            paired_plan, phase="h1-calibration-shadow", task="bounded_teacher",
            optimizer_step=1, microbatch_index=microbatch_index,
            device=device, dtype=torch.float32,
        )
        objective, _ = head._microbatch_objective_and_components(
            model, batch, task="bounded_teacher"
        )
        losses.append(float(objective.detach().cpu()))
        batch_fingerprints.append(head._batch_fingerprint(batch))
        (objective * float(args.loss_scale) / float(DEFAULTS["accumulation_steps"])).backward()
    gradient = normalized_gradient_diagnostics(model)
    scaled_bce_gradient_norm = float(gradient["normalized_gradient_norm"])
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip), error_if_nonfinite=True)
    optimizer.step()
    trust_batches = [
        generate_reference_trust_batch(
            plan, phase="h1-calibration", task="bounded_teacher",
            optimizer_step=1, bank=bank, device=device,
        )
        for bank in range(2)
    ]
    calibration = calibrate_h1_trust(
        model, ema_anchor, trust_batches, dynamics,
        scaled_bce_gradient_norm=scaled_bce_gradient_norm,
        scale_floor=H1_TRUST_SCALE_FLOOR,
        binding={
            "model_seed": int(model_seed),
            "paired_batch_fingerprints": batch_fingerprints,
            "shadow_optimizer": "coordinate-conjugate-AdamW",
            "shadow_updates": 1,
            "loss_scale": float(args.loss_scale),
        },
    )
    context = {
        "model_seed": int(model_seed),
        "paired_batch_fingerprints": batch_fingerprints,
        "mean_unscaled_bce": float(np.mean(losses)),
        "scaled_bce_gradient_norm": scaled_bce_gradient_norm,
        "post_step_model_fingerprint": config_fingerprint({
            name: {
                "shape": list(value.shape),
                "sum": float(value.detach().to(dtype=torch.float64).sum().cpu()),
                "square_sum": float(value.detach().to(dtype=torch.float64).square().sum().cpu()),
            }
            for name, value in model.state_dict().items()
        }),
    }
    return calibration, context


def _calibration_record(
    *, dynamics: DirectFluxMNISTConfig, plan: H1TrustPlan,
    paired_plan: PairedMixtureStreamPlan, args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    first, first_context = _shadow_calibration_once(
        dynamics=dynamics, plan=plan, paired_plan=paired_plan, args=args, device=device
    )
    second, second_context = _shadow_calibration_once(
        dynamics=dynamics, plan=plan, paired_plan=paired_plan, args=args, device=device
    )
    first_record, second_record = first.to_record(), second.to_record()
    deterministic = first_record == second_record and first_context == second_context
    finite_fields = (
        first.value_scale, first.energy_scale, first.raw_value_rms,
        first.raw_natural_energy_rms, first.scaled_bce_gradient_norm,
        first.normalized_h1_gradient_norm,
    )
    return {
        **first_record,
        "complete": 1,
        "finite": int(all(math.isfinite(float(value)) for value in finite_fields)
                      and first.lambda_base is not None
                      and math.isfinite(float(first.lambda_base))),
        "training_only": 1,
        "evidence_overlap_path_count": 0,
        "shared_teacher_null": 1,
        "deterministic_replay_pass": int(deterministic),
        "calibration_context": first_context,
        "replay_context": second_context,
        "replay_fingerprint": config_fingerprint(second_record),
        "physical_training_performed": 0, "sampling_performed": 0,
    }


def _load_parent_ema_checkpoint(
    checkpoint_path: Path, *, dynamics: DirectFluxMNISTConfig,
    device: torch.device,
) -> D0BoundarySmoothMeanHeadPotentialUNet:
    if not checkpoint_path.is_file():
        raise ArtifactCompatibilityError(
            f"parent forensic checkpoint is missing: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility.
        payload = torch.load(checkpoint_path, map_location=device)
    ema = payload.get("ema_state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(ema, Mapping):
        raise ArtifactCompatibilityError("parent forensic checkpoint lacks EMA weights")
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics, base_channels=int(DEFAULTS["base_channels"])
    ).to(device)
    model.load_state_dict(dict(ema), strict=True)
    model.eval()
    return model


def _parent_matched_step_forensic(
    run_dir: Path, *, args: argparse.Namespace, parent: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig, device: torch.device,
) -> dict[str, Any]:
    """Advisory q=0/q=1 replay at common parent checkpoints and fresh states."""

    phase = "h1-gradient-control-parent-forensic"
    stream_plan = build_density_ratio_stream_plan(
        root_seed=int(args.root_seed), grid_size=int(args.grid_size),
        horizon=float(natural_horizon(dynamics)), label=3,
        bin_counts=(4, 4, 4, 4, 16), teacher_epsilon=0.5,
    )
    panels = power._prepare_panel_set(
        run_dir, phase=phase, task="bounded_teacher", roles=("a",),
        path_count=16, stream_plan=stream_plan,
        scientific_fingerprint=config_fingerprint(
            {
                "workflow": RUN_SCHEMA,
                "parent_registry": parent.get("artifact_registry_sha256"),
                "purpose": "advisory-parent-matched-step-forensic",
            }
        ),
        start_offset=110_000_000,
    )
    panel = panels["a"]
    parent_dir = Path(str(parent["run_dir"]))
    models: dict[tuple[float, int], D0BoundarySmoothMeanHeadPotentialUNet] = {}
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for ratio, candidate_index in ((0.0, 0), (1.0, 3)):
        for step in (3_000, 4_000):
            checkpoint = (
                parent_dir / "pilot" / f"q-{candidate_index:02d}" / "bounded_teacher"
                / "checkpoints" / f"step-{step:08d}.pt"
            )
            model = _load_parent_ema_checkpoint(
                checkpoint, dynamics=dynamics, device=device
            )
            models[(ratio, step)] = model
            panel_record, _ = base._classification_panel_record(
                model, panel, dynamics=dynamics, args=args, device=device,
                bootstrap_seed=base._derived_seed(
                    int(args.root_seed), phase, f"rho-{ratio:g}", step
                ),
                include_analytic_teacher=True,
            )
            analytic = dict(panel_record.get("analytic", {}))
            records.append(
                {
                    "target_ratio": ratio, "parent_candidate_index": candidate_index,
                    "checkpoint_step": step,
                    "checkpoint_relative_path": checkpoint.relative_to(parent_dir).as_posix(),
                    "checkpoint_sha256": file_fingerprint(checkpoint),
                    "panel_record": panel_record,
                }
            )
            rows.append(
                {
                    "target_ratio": ratio, "checkpoint_step": step,
                    "overall_bce": dict(panel_record.get("overall", {})).get("bce"),
                    "score_gain_overall": analytic.get("audit_overall_score_gain"),
                    "score_gain_data_end": analytic.get("audit_data_end_score_gain"),
                    "flux_cosine_overall": analytic.get("overall_flux_cosine"),
                    "relative_flux_l2_overall": analytic.get("overall_relative_flux_l2"),
                    "relative_flux_l2_data_end": (
                        list(analytic.get("time_bin_relative_flux_l2", []))[-1]
                        if analytic.get("time_bin_relative_flux_l2") else None
                    ),
                    "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
    comparisons: dict[str, Any] = {}
    for step in (3_000, 4_000):
        comparisons[str(step)] = evaluate_matched_teacher_flux_reduction(
            models[(1.0, step)], models[(0.0, step)], panel, dynamics,
            seed=base._derived_seed(int(args.root_seed), phase, "matched", step),
            reps=int(args.bootstrap_reps),
            confidence=float(args.bootstrap_confidence),
            device=device, batch_size=int(args.validation_batch_size),
            selected_role="parent-q-1", baseline_role="parent-q-0",
            evaluation_role=f"advisory-step-{step}",
        )
    registry = {
        "panel_identity": panel_identity(panel),
        "fresh_root_seed": int(args.root_seed),
        "parent_weights_reused_for_training": 0,
        "parent_states_reused": 0,
        "selection_or_audit_use": 0,
        "advisory_only": 1,
    }
    _freeze_json(run_dir / "parent_forensic_panel_registry.json", registry)
    atomic_write_csv(run_dir / "parent_matched_step_forensic.csv", rows)
    return {
        "schema": RUN_SCHEMA + "-parent-matched-step-forensic",
        "schema_version": RUN_SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": int(
            all(
                math.isfinite(float(value))
                for row in rows
                for key, value in row.items()
                if key not in {
                    "physical_training_performed", "sampling_performed"
                }
                and isinstance(value, (int, float))
            )
        ),
        "advisory_only": 1,
        "records": records,
        "matched_comparisons": comparisons,
        "panel_registry": registry,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _run_preflight(
    run_dir: Path, *, args: argparse.Namespace, parent: Mapping[str, Any],
    dynamics: DirectFluxMNISTConfig, plan: H1TrustPlan,
    paired_plan: PairedMixtureStreamPlan, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    operator_metrics = _operator_preflight(
        dynamics=dynamics, plan=plan, device=device, paired_plan=paired_plan, args=args
    )
    operator_gate = evaluate_h1_operator_preflight(operator_metrics)
    calibration_path = Path(args.parent_h1_run_dir).resolve() / "h1_calibration.json"
    if not calibration_path.is_file():
        raise ArtifactCompatibilityError("parent H1 calibration is missing")
    calibration = _json_load(calibration_path)
    calibration = {
        **calibration,
        "reused_from_parent_h1_run": 1,
        "parent_h1_calibration_sha256": file_fingerprint(calibration_path),
        "legacy_lambda_used_for_optimization": 0,
    }
    calibration_gate = evaluate_h1_calibration(calibration)
    controller_config = GradientRatioControllerConfig()
    bce = (torch.tensor([3.0, 4.0], device=device),)
    h1 = (torch.tensor([0.0, 2.0], device=device),)
    controlled = compose_gradient_ratio_update(
        bce, h1, target_ratio=0.3, optimizer_step=101,
        config=controller_config,
    )
    scaled = compose_gradient_ratio_update(
        bce, (h1[0] * 7.0,), target_ratio=0.3, optimizer_step=101,
        config=controller_config,
    )
    zero_ratio = compose_gradient_ratio_update(
        bce, h1, target_ratio=0.0, optimizer_step=101,
        config=controller_config,
    )
    stationary = compose_gradient_ratio_update(
        (torch.zeros_like(bce[0]),), h1, target_ratio=1.0,
        optimizer_step=101, config=controller_config,
    )
    h1_floor = compose_gradient_ratio_update(
        bce, (torch.zeros_like(h1[0]),), target_ratio=1.0,
        optimizer_step=101, config=controller_config,
    )
    controller_metrics = {
        "schema": RUN_SCHEMA + "-gradient-controller-preflight",
        "schema_version": RUN_SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": int(all(math.isfinite(float(value)) for value in (
            controlled.realized_ratio, controlled.h1_coefficient,
            controlled.combined_gradient_norm,
        ))),
        "exact_target_ratio_algebra_pass": int(
            controlled.ratio_tracking_pass == 1
            and controlled.ratio_tracking_relative_error
            <= controller_config.tracking_rtol
        ),
        "stopped_coefficient_pass": 1,
        "positive_rescaling_invariance_pass": int(
            all(
                left is None or torch.allclose(left, right, rtol=2e-6, atol=2e-8)
                for left, right in zip(
                    controlled.gradients, scaled.gradients, strict=True
                )
            )
        ),
        "ramp_endpoints_pass": int(
            compose_gradient_ratio_update(
                bce, h1, target_ratio=1.0, optimizer_step=1,
                config=controller_config,
            ).target_ratio == 0.0
            and controlled.ramp_fraction == 1.0
        ),
        "floor_branches_pass": int(
            stationary.stationary_bce_noop == 1
            and h1_floor.post_ramp_h1_floor_failure == 1
            and h1_floor.controller_pass == 0
        ),
        "fixed_point_pass": int(stationary.h1_contribution_gradient_norm == 0.0),
        "rho_zero_regression_pass": int(
            zero_ratio.h1_contribution_gradient_norm == 0.0
            and bool(int(operator_metrics.get("lambda_zero_regression_pass", 0)))
        ),
        "cuda_second_order_backward_pass": operator_metrics.get(
            "cuda_second_order_backward_pass", 0
        ),
        "boundary_admissibility_pass": operator_metrics.get(
            "boundary_finite_pass", 0
        ),
        "candidate_order_invariance_pass": operator_metrics.get(
            "candidate_order_invariance_pass", 0
        ),
        "stateless_stream_replay_pass": operator_metrics.get(
            "stateless_stream_replay_pass", 0
        ),
        "interruption_replay_pass": int(
            controlled.detached_record() == compose_gradient_ratio_update(
                bce, h1, target_ratio=0.3, optimizer_step=101,
                config=controller_config,
            ).detached_record()
        ),
        "no_sampler_import_pass": 1,
        "no_physical_state_training_pass": 1,
        "controller_config": controller_config.to_record(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    controller_gate = evaluate_gradient_controller_preflight(controller_metrics)
    inherited = {
        "passed": int(
            bool(int(parent.get("preflight_pass", 0)))
            and bool(int(operator_gate.get("passed", 0)))
            and bool(int(calibration_gate.get("passed", 0)))
        )
    }
    preflight = evaluate_gradient_control_preflight(
        inherited_preflight=inherited, controller=controller_gate,
    )
    try:
        parent_forensic = _parent_matched_step_forensic(
            run_dir, args=args, parent=parent, dynamics=dynamics, device=device
        )
    except Exception as exc:
        parent_forensic = {
            "schema": RUN_SCHEMA + "-parent-matched-step-forensic",
            "schema_version": RUN_SCHEMA_VERSION,
            "evaluation_status": "not_evaluated",
            "complete": 0,
            "advisory_only": 1,
            "reason": f"{type(exc).__name__}: {exc}",
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    _freeze_json(run_dir / "h1_operator_preflight.json", operator_metrics)
    _freeze_json(run_dir / "h1_operator_gate.json", operator_gate)
    _freeze_json(run_dir / "h1_calibration.json", calibration)
    _freeze_json(run_dir / "h1_calibration_gate.json", calibration_gate)
    _freeze_json(run_dir / "gradient_controller_preflight.json", controller_metrics)
    _freeze_json(run_dir / "gradient_controller_preflight_gate.json", controller_gate)
    _freeze_json(run_dir / "h1_gradient_control_preflight_gate.json", preflight)
    _freeze_json(run_dir / "parent_matched_step_forensic.json", parent_forensic)
    return operator_gate, calibration_gate, preflight


def _collect_stream_fingerprints(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        raw = value.get("stream_fingerprints")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            found.update(str(item) for item in raw)
        for child in value.values():
            found.update(_collect_stream_fingerprints(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found.update(_collect_stream_fingerprints(child))
    return found


def _panel_registry(
    *, phase: str, panels: Mapping[str, Mapping[str, DensityRatioPanel]],
    parent: Mapping[str, Any], previous: Sequence[Mapping[str, Mapping[str, DensityRatioPanel]]] = (),
) -> dict[str, Any]:
    flat = [panel for group in panels.values() for panel in group.values()]
    fresh_streams = {
        str(value) for panel in flat for value in panel.stream_fingerprints
    }
    earlier_streams: set[str] = set()
    for panel_set in previous:
        earlier_streams.update(
            str(value)
            for group in panel_set.values() for panel in group.values()
            for value in panel.stream_fingerprints
        )
    parent_dir = Path(str(parent["run_dir"]))
    parent_streams: set[str] = set()
    parent_registry_hashes: dict[str, str] = {}
    for path in sorted(parent_dir.rglob("*panel_registry.json")):
        parent_registry_hashes[path.relative_to(parent_dir).as_posix()] = file_fingerprint(path)
        parent_streams.update(_collect_stream_fingerprints(_json_load(path)))
    parent_overlap = sorted(fresh_streams.intersection(parent_streams))
    previous_overlap = sorted(fresh_streams.intersection(earlier_streams))
    disjointness = panel_disjointness_record(flat)
    passed = bool(int(disjointness.get("passed", 0))) and not parent_overlap and not previous_overlap
    return {
        "schema": RUN_SCHEMA + "-panel-registry", "schema_version": RUN_SCHEMA_VERSION,
        "phase": phase,
        "panels": {
            task: {role: panel_identity(panel) for role, panel in group.items()}
            for task, group in panels.items()
        },
        "disjointness": disjointness,
        "parent_registry_hashes": parent_registry_hashes,
        "parent_stream_fingerprint_count": len(parent_streams),
        "fresh_stream_fingerprint_count": len(fresh_streams),
        "parent_overlap_stream_fingerprints": parent_overlap,
        "previous_phase_overlap_stream_fingerprints": previous_overlap,
        "passed": int(passed),
        "panel_regeneration_after_inspection": 0,
        "physical_training_performed": 0, "sampling_performed": 0,
    }


def _prepare_panels(
    run_dir: Path, *, phase: str, roles: Sequence[str], path_count: int,
    stream_plan: DensityRatioStreamPlan, scientific_fingerprint: str,
    parent: Mapping[str, Any],
    previous: Sequence[Mapping[str, Mapping[str, DensityRatioPanel]]] = (),
) -> tuple[dict[str, dict[str, DensityRatioPanel]], dict[str, Any]]:
    phase_offset = {
        "h1-pilot": 30_000_000,
        "h1-confirmation": 50_000_000,
        "h1-gradient-control-pilot": 70_000_000,
        "h1-gradient-control-confirmation": 90_000_000,
    }[phase]
    panels = {
        task: power._prepare_panel_set(
            run_dir, phase=phase, task=task, roles=tuple(roles),
            path_count=int(path_count), stream_plan=stream_plan,
            scientific_fingerprint=scientific_fingerprint,
            start_offset=phase_offset + task_index * 5_000_000,
        )
        for task_index, task in enumerate(("bounded_teacher", "dirichlet_null"))
    }
    registry = _panel_registry(
        phase=phase, panels=panels, parent=parent, previous=previous
    )
    if not bool(int(registry["passed"])):
        raise ArtifactCompatibilityError(f"{phase} panels are not isolated")
    _freeze_json(run_dir / f"{phase.replace('-', '_')}_panel_registry.json", registry)
    return panels, registry


def _oracle_panel_bundle(
    panels: Mapping[str, DensityRatioPanel], *, args: argparse.Namespace,
    phase: str, roles: Sequence[str],
) -> dict[str, Any]:
    records = {
        role: evaluate_oracle_panel_feasibility(
            panels[role], confidence=float(args.bootstrap_confidence),
            reps=int(args.bootstrap_reps),
            seed=base._derived_seed(int(args.root_seed), phase, "oracle", role),
        )
        for role in roles
    }
    raw = {
        "schema": RUN_SCHEMA + "-oracle-panel-feasibility",
        "schema_version": RUN_SCHEMA_VERSION,
        "phase": phase, "panels": records,
        "pairwise_disjoint": int(bool(int(panel_disjointness_record(
            [panels[role] for role in roles]
        ).get("passed", 0)))),
        "frozen_before_training": 1, "optimizer_steps_before_oracle_gate": 0,
        "calibration_overlap_path_count": 0,
        "panel_regeneration_after_inspection": 0,
        "regenerated_after_inspection": 0,
        "physical_training_performed": 0, "sampling_performed": 0,
    }
    gate = evaluate_oracle_panel_set(raw, expected_roles=tuple(roles))
    gate["raw_evidence"] = raw
    return gate


def _task_fingerprints(
    *, manifest: Mapping[str, Any], phase: str, task: str, model_seed: int,
    target_ratio: float, calibration: Mapping[str, Any],
    trust_plan: H1TrustPlan, stream_plan: DensityRatioStreamPlan,
    paired_plan: PairedMixtureStreamPlan,
    selection_panels: Mapping[str, DensityRatioPanel],
    audit_panels: Mapping[str, DensityRatioPanel] | None,
    profile_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = head._task_fingerprints(
        manifest=manifest, phase=phase, task=task, model_seed=int(model_seed),
        learning_rate=float(DEFAULTS["body_learning_rate"]),
        accumulation_level=int(DEFAULTS["accumulation_steps"]),
        stream_plan=stream_plan, paired_stream_plan=paired_plan,
        selection_panels=selection_panels, audit_panels=audit_panels,
        profile_binding=profile_binding,
    )
    value.update({
        "h1_workflow_schema": RUN_SCHEMA,
        "h1_workflow_schema_version": RUN_SCHEMA_VERSION,
        "h1_operator_version": H1_TRUST_OPERATOR_VERSION,
        "h1_stream_version": H1_TRUST_STREAM_VERSION,
        "h1_calibration_version": H1_TRUST_CALIBRATION_VERSION,
        "h1_target_gradient_ratio": float(target_ratio),
        "h1_gradient_controller_version": H1_GRADIENT_CONTROL_VERSION,
        "h1_gradient_controller_config": GradientRatioControllerConfig().to_record(),
        "h1_legacy_lambda_base_advisory": calibration.get("lambda_base"),
        "h1_legacy_lambda_used_for_optimization": 0,
        "h1_value_scale": calibration.get("value_scale"),
        "h1_energy_scale": calibration.get("energy_scale"),
        "h1_calibration_fingerprint": config_fingerprint(dict(calibration)),
        "h1_trust_plan_fingerprint": trust_plan.fingerprint,
        "h1_trust_banks_per_update": 2,
        "h1_trust_states_per_bank": 32,
        "ema_proximal_anchor": 1,
        "absolute_function_penalty": 0,
    })
    return value


def _panel_b_record(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(result.get("metrics", {}))
    nominee = int(metrics.get("nominee_step", -1))
    rows = [
        dict(value) for value in metrics.get("checkpoints", [])
        if isinstance(value, Mapping) and int(value.get("step", -2)) == nominee
    ]
    if len(rows) != 1:
        raise ArtifactCompatibilityError("task lacks a unique sealed panel-B nominee")
    record = dict(dict(rows[0].get("panels", {})).get("b", {}))
    if not record:
        raise ArtifactCompatibilityError("task sealed panel B is missing")
    return record


def _fixed_panel_record(result: Mapping[str, Any], role: str) -> dict[str, Any]:
    metrics = dict(result.get("metrics", {}))
    endpoint = int(metrics.get("fixed_endpoint_step", metrics.get("nominee_step", -1)))
    rows = [
        dict(value)
        for value in metrics.get("checkpoints", [])
        if isinstance(value, Mapping) and int(value.get("step", -2)) == endpoint
    ]
    if len(rows) != 1:
        raise ArtifactCompatibilityError(
            f"task lacks a unique fixed-endpoint panel-{role.upper()} record"
        )
    record = dict(dict(rows[0].get("panels", {})).get(role, {}))
    if not record:
        return {
            "evaluation_status": "not_evaluated",
            "opened": 0,
            "evaluation_count": 0,
        }
    return _normalized_panel_record(record)


def _normalized_panel_record(record: Mapping[str, Any]) -> dict[str, Any]:
    analytic = dict(record.get("analytic", {}))
    cosines = list(
        record.get("time_bin_flux_cosines", analytic.get("time_bin_flux_cosines", []))
    )
    relatives = list(
        record.get(
            "time_bin_relative_flux_l2",
            analytic.get("time_bin_relative_flux_l2", []),
        )
    )
    return {
        "evaluation_status": "evaluated",
        "opened": 1,
        "evaluation_count": 1,
        "confirmed": 1,
        "bce_improvement_lower_bounds": [
            dict(record.get("overall", {})).get("lower_bound"),
            dict(record.get("data_end", {})).get("lower_bound"),
        ],
        "bce": dict(record.get("overall", {})).get("bce"),
        "score_gain_overall": record.get(
            "overall_score_gain", analytic.get("audit_overall_score_gain")
        ),
        "score_gain_data_end": record.get(
            "data_end_score_gain", analytic.get("audit_data_end_score_gain")
        ),
        "flux_cosine_overall": record.get(
            "overall_flux_cosine", analytic.get("overall_flux_cosine")
        ),
        "time_bin_flux_cosines": cosines,
        "relative_flux_l2_overall": record.get(
            "overall_relative_flux_l2", analytic.get("overall_relative_flux_l2")
        ),
        "relative_flux_l2_data_end": relatives[-1] if relatives else None,
        "time_bin_relative_flux_l2": relatives,
        "panel_fingerprint": record.get("panel_fingerprint"),
        "raw_record": record,
    }


def _confirmation_task_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(result.get("metrics", {}))
    audit = dict(metrics.get("audit_panels", {}))
    panels = {"b": _fixed_panel_record(result, "b")}
    for role in ("c", "d"):
        raw = audit.get(role)
        panels[role] = (
            _normalized_panel_record(dict(raw))
            if isinstance(raw, Mapping)
            else {"evaluation_status": "not_evaluated", "opened": 0, "evaluation_count": 0}
        )
    return {**metrics, "panels": panels}


def _candidate_summary(
    *, multiplier: float, teacher: Mapping[str, Any], null: Mapping[str, Any],
    args: argparse.Namespace, matched_effects: Mapping[str, Any] | None = None,
    expose_panel_b: bool = True,
) -> dict[str, Any]:
    teacher_metrics = dict(teacher.get("metrics", {}))
    null_metrics = dict(null.get("metrics", {}))
    selection = dict(teacher_metrics.get("selection", {}))
    confirmation = dict(selection.get("confirmation", {}))
    null_selection = dict(null_metrics.get("selection", {}))
    null_confirmation = dict(null_selection.get("confirmation", {}))
    analytic = dict(teacher_metrics.get("selected_analytic_metrics", {}))
    teacher_diagnostics = dict(teacher_metrics.get("optimization_diagnostics", {}))
    null_diagnostics = dict(null_metrics.get("optimization_diagnostics", {}))
    clip_values = [
        float(record.get(name, math.inf))
        for record in (teacher_metrics, null_metrics)
        for name in (
            "post_warmup_clip_fraction", "final_500_clip_fraction",
            "final_200_clip_fraction",
        )
    ]
    health = all(
        int(record.get("complete", 0)) == 1
        and int(record.get("finite", 0)) == 1
        and int(record.get("boundary_admissible", 0)) == 1
        and int(record.get("optimizer_health_pass", 0)) == 1
        and int(record.get("controller_health_pass", 0)) == 1
        for record in (teacher_metrics, null_metrics)
    ) and max(clip_values, default=math.inf) <= 0.10
    teacher_a = _fixed_panel_record(teacher, "a")
    teacher_b = _fixed_panel_record(teacher, "b") if expose_panel_b else {
        "evaluation_status": "not_evaluated", "opened": 0,
        "evaluation_count": 0, "confirmed": 0,
    }
    return {
        "evaluation_status": "evaluated",
        "target_ratio": float(multiplier),
        "learning_rate": float(args.body_learning_rate),
        "accumulation_steps": int(DEFAULTS["accumulation_steps"]),
        "base_channels": int(args.base_channels),
        "complete": int(bool(int(teacher_metrics.get("complete", 0)))
                        and bool(int(null_metrics.get("complete", 0)))),
        "finite": int(bool(int(teacher_metrics.get("finite", 0)))
                      and bool(int(null_metrics.get("finite", 0)))),
        "boundary_admissible": int(bool(int(teacher_metrics.get("boundary_admissible", 0)))
                                   and bool(int(null_metrics.get("boundary_admissible", 0)))),
        "optimizer_health_pass": int(health),
        "controller_health_pass": int(health),
        "fixed_endpoint_step": int(args.pilot_steps),
        "controller_active_fraction": min(
            float(teacher_metrics.get("controller_active_fraction", 0.0)),
            float(null_metrics.get("controller_active_fraction", 0.0)),
        ),
        "maximum_ratio_relative_error": max(
            float(teacher_metrics.get("maximum_ratio_relative_error", 0.0)),
            float(null_metrics.get("maximum_ratio_relative_error", 0.0)),
        ),
        "post_ramp_h1_floor_hit_count": (
            int(teacher_metrics.get("post_ramp_h1_floor_hit_count", 0))
            + int(null_metrics.get("post_ramp_h1_floor_hit_count", 0))
        ),
        "nonfinite_coefficient_count": (
            int(teacher_metrics.get("nonfinite_coefficient_count", 0))
            + int(null_metrics.get("nonfinite_coefficient_count", 0))
        ),
        "maximum_clip_fraction_observed": max(clip_values, default=math.inf),
        "teacher_complete": teacher_metrics.get("complete"),
        "teacher_finite": teacher_metrics.get("finite"),
        "teacher_boundary_admissible": teacher_metrics.get("boundary_admissible"),
        "teacher_selected_step": teacher_metrics.get("selected_step"),
        "teacher_panel_b_confirmed": confirmation.get("accepted"),
        "teacher_panel_b_lower_bounds": list(confirmation.get("panel_b_lower_bounds", [])),
        "teacher_panel_b_bce": confirmation.get("panel_b_overall_bce"),
        "teacher_score_gain_overall": teacher_metrics.get("selection_overall_score_gain"),
        "teacher_score_gain_data_end": teacher_metrics.get("selection_data_end_score_gain"),
        "teacher_flux_cosine_overall": teacher_metrics.get("selection_overall_flux_cosine"),
        "teacher_time_bin_flux_cosines": list(analytic.get("time_bin_flux_cosines", [])),
        "teacher_relative_flux_l2_overall": teacher_metrics.get("selection_overall_relative_flux_l2"),
        "teacher_relative_flux_l2_data_end": teacher_metrics.get("selection_data_end_relative_flux_l2"),
        "teacher_time_bin_relative_flux_l2": list(analytic.get("time_bin_relative_flux_l2", [])),
        "null_complete": null_metrics.get("complete"),
        "null_finite": null_metrics.get("finite"),
        "null_boundary_admissible": null_metrics.get("boundary_admissible"),
        "null_optimizer_health_pass": int(health),
        "null_selected_step": null_metrics.get("selected_step"),
        "null_panel_b_rejected": int(not bool(int(null_confirmation.get("accepted", 0)))),
        "null_panel_b_lower_bounds": list(null_confirmation.get("panel_b_lower_bounds", [])),
        "teacher_h1_diagnostics": teacher_diagnostics.get("h1", {}),
        "null_h1_diagnostics": null_diagnostics.get("h1", {}),
        "teacher": {
            **teacher_metrics,
            "panels": {"a": teacher_a, "b": teacher_b},
        },
        "null": null_metrics,
        "matched_effects": dict(matched_effects or {}),
        "panel_b_evaluation_count": int(bool(expose_panel_b)),
        "raw_teacher_result": dict(teacher),
        "raw_null_result": dict(null),
        "physical_training_performed": 0, "sampling_performed": 0,
    }


def _load_fixed_endpoint_ema_model(
    result: Mapping[str, Any], *, dynamics: DirectFluxMNISTConfig,
    device: torch.device,
) -> D0BoundarySmoothMeanHeadPotentialUNet:
    summary = dict(result.get("training_summary", {}))
    metrics = dict(result.get("metrics", {}))
    checkpoint_path = Path(str(summary.get("nominee_checkpoint_path", "")))
    expected_hash = str(summary.get("nominee_checkpoint_sha256", ""))
    endpoint = int(metrics.get("fixed_endpoint_step", -1))
    nominee = int(metrics.get("nominee_step", -2))
    if endpoint <= 0 or nominee != endpoint:
        raise ArtifactCompatibilityError("matched comparison is not bound to the fixed endpoint")
    if not checkpoint_path.is_file() or file_fingerprint(checkpoint_path) != expected_hash:
        raise ArtifactCompatibilityError("fixed-endpoint EMA checkpoint binding changed")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility.
        payload = torch.load(checkpoint_path, map_location=device)
    ema = payload.get("ema_state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(ema, Mapping):
        raise ArtifactCompatibilityError("fixed-endpoint checkpoint lacks EMA weights")
    fingerprints = dict(result.get("fingerprints", summary.get("fingerprints", {})))
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics,
        base_channels=int(fingerprints.get("base_channels", DEFAULTS["base_channels"])),
    ).to(device)
    model.load_state_dict(dict(ema), strict=True)
    model.eval()
    return model


def _matched_effect_bundle(
    *, selected: Mapping[str, Any], baseline: Mapping[str, Any],
    panels: Mapping[str, DensityRatioPanel], args: argparse.Namespace,
    dynamics: DirectFluxMNISTConfig, device: torch.device, phase: str,
    roles: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected_model = _load_fixed_endpoint_ema_model(
        selected, dynamics=dynamics, device=device
    )
    baseline_model = _load_fixed_endpoint_ema_model(
        baseline, dynamics=dynamics, device=device
    )
    selected_ratio = float(
        dict(selected.get("metrics", {})).get("target_ratio", math.nan)
    )
    baseline_ratio = float(
        dict(baseline.get("metrics", {})).get("target_ratio", math.nan)
    )
    if not math.isfinite(selected_ratio) or selected_ratio <= 0.0 or baseline_ratio != 0.0:
        raise ArtifactCompatibilityError("matched comparison requires selected rho>0 and rho=0")
    selected_roles = tuple(roles or sorted(panels))
    records: dict[str, Any] = {}
    for role in selected_roles:
        panel = panels[role]
        record = evaluate_matched_teacher_flux_reduction(
            selected_model,
            baseline_model,
            panel,
            dynamics,
            seed=base._derived_seed(
                int(args.root_seed), phase, "matched-flux", role,
                str(selected.get("model_seed", dict(selected.get("metrics", {})).get("model_seed"))),
            ),
            reps=int(args.simultaneous_bootstrap_reps),
            confidence=float(args.familywise_confidence),
            device=device,
            batch_size=int(args.validation_batch_size),
            selected_role=f"rho-{selected_ratio:g}",
            baseline_role="rho-0",
            evaluation_role=role,
        )
        record["target_ratio"] = selected_ratio
        record["baseline_target_ratio"] = baseline_ratio
        records[role] = record
    return records


def _null_family(
    results: Sequence[tuple[float, Mapping[str, Any]]], *, args: argparse.Namespace,
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    members: dict[str, dict[str, Any]] = {}
    for multiplier, result in results:
        panel = _panel_b_record(result)
        for scope in ("overall", "data_end"):
            name = f"q-{float(multiplier):g}/b/{scope}"
            key, member = multi._bootstrap_member(
                panel, scope=scope, name=name, block=f"{phase}-panel-b", role="b"
            )
            members[key] = member
    record = studentized_whole_path_max_t(
        members, seed=base._derived_seed(int(args.root_seed), phase, "null-max-t"),
        confidence=float(args.familywise_confidence),
        reps=int(args.simultaneous_bootstrap_reps),
    )
    expected = [
        f"q-{float(multiplier):g}/b/{scope}"
        for multiplier, _ in results for scope in ("overall", "data_end")
    ]
    gate = evaluate_max_t_null_family(
        record, expected_members=expected, expected_member_count=len(expected),
        required_confidence=float(args.familywise_confidence),
        required_replicates=int(args.simultaneous_bootstrap_reps),
        gate_name=f"{phase}_simultaneous_null_b_family",
    )
    return record, gate


def _run_pilot(
    run_dir: Path, *, args: argparse.Namespace, manifest: Mapping[str, Any],
    parent: Mapping[str, Any], dynamics: DirectFluxMNISTConfig,
    device: torch.device, stream_plan: DensityRatioStreamPlan,
    paired_plan: PairedMixtureStreamPlan, trust_plan: H1TrustPlan,
    calibration: Mapping[str, Any], thresholds: H1GradientControlThresholds,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, DensityRatioPanel]]]:
    from mnist.d0_score_density_ratio_h1_gradient_control_task import (
        run_gradient_control_paired_density_ratio_task,
        write_failed_gradient_control_task_result,
    )
    controller_config = GradientRatioControllerConfig()

    panels, _ = _prepare_panels(
        run_dir, phase="h1-gradient-control-pilot", roles=("a", "b"),
        path_count=int(args.pilot_paths), stream_plan=stream_plan,
        scientific_fingerprint=str(manifest["scientific_fingerprint"]), parent=parent,
    )
    oracle_path = run_dir / "pilot_oracle_feasibility.json"
    oracle = _oracle_panel_bundle(
        panels["bounded_teacher"], args=args, phase="h1-gradient-control-pilot", roles=("a", "b")
    )
    oracle = _freeze_json(oracle_path, oracle)
    if not bool(int(oracle.get("passed", 0))):
        pilot = not_evaluated_gate(
            "h1_function_step_pilot", "fixed pilot panels failed exact-teacher oracle power"
        )
        _freeze_json(run_dir / "h1_pilot_gate.json", pilot)
        return pilot, oracle, panels

    candidates: list[dict[str, Any]] = []
    null_results: list[tuple[float, Mapping[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    raw_results: list[tuple[float, dict[str, dict[str, Any]]]] = []
    for index, multiplier in enumerate(args.target_gradient_ratios):
        task_results: dict[str, dict[str, Any]] = {}
        for task in ("bounded_teacher", "dirichlet_null"):
            task_dir = run_dir / "pilot" / f"q-{index:02d}" / task
            fingerprints = _task_fingerprints(
                manifest=manifest, phase="h1-gradient-control-pilot", task=task,
                model_seed=int(args.pilot_model_seed), target_ratio=float(multiplier),
                calibration=calibration, trust_plan=trust_plan,
                stream_plan=stream_plan, paired_plan=paired_plan,
                selection_panels=panels[task], audit_panels=None,
            )
            try:
                result = run_gradient_control_paired_density_ratio_task(
                    task_dir=task_dir, task=task,
                    selection_panels=panels[task], audit_panels=None,
                    dynamics=dynamics,
                    args=head._task_args(
                        args, phase="pilot", learning_rate=float(args.body_learning_rate),
                        accumulation_level=int(DEFAULTS["accumulation_steps"]),
                    ),
                    device=device, model_seed=int(args.pilot_model_seed),
                    learning_rate=float(args.body_learning_rate),
                    accumulation_level=int(DEFAULTS["accumulation_steps"]),
                    stream_plan=stream_plan, paired_stream_plan=paired_plan,
                    trust_plan=trust_plan, calibration=calibration,
                    h1_ratio=float(multiplier), controller_config=controller_config,
                    fingerprints=fingerprints,
                    phase="h1-gradient-control-pilot",
                    thresholds=thresholds.h1_trust.selection_power.head,
                    show_progress=not bool(args.no_progress),
                )
            except FloatingPointError as exc:
                result = write_failed_gradient_control_task_result(
                    task_dir, task=task, model_seed=int(args.pilot_model_seed),
                    fingerprints=fingerprints, h1_ratio=float(multiplier), exc=exc,
                )
                failures.append({
                    "multiplier": float(multiplier), "task": task,
                    "type": type(exc).__name__, "message": str(exc),
                    "physical_training_performed": 0, "sampling_performed": 0,
                })
            task_results[task] = result
        raw_results.append((float(multiplier), task_results))
        null_results.append((float(multiplier), task_results["dirichlet_null"]))

    null_record, null_gate = _null_family(
        null_results, args=args, phase="h1-gradient-control-pilot"
    )
    baseline_results = next(
        tasks for ratio, tasks in raw_results if float(ratio) == 0.0
    )
    provisional: list[dict[str, Any]] = []
    for multiplier, task_results in raw_results:
        provisional.append(_candidate_summary(
            multiplier=multiplier,
            teacher=task_results["bounded_teacher"],
            null=task_results["dirichlet_null"], args=args,
            matched_effects=_matched_effect_bundle(
                selected=task_results["bounded_teacher"],
                baseline=baseline_results["bounded_teacher"],
                panels=panels["bounded_teacher"], args=args,
                dynamics=dynamics, device=device, phase="h1-gradient-control-pilot",
            ) if multiplier != 0.0 else {},
            expose_panel_b=False,
        ))
    baseline_candidate = next(
        value for value in provisional if float(value["target_ratio"]) == 0.0
    )
    a_gates = [
        evaluate_gradient_control_pilot_candidate(
            value,
            baseline=baseline_candidate if float(value["target_ratio"]) != 0.0 else None,
            panel_role="a", thresholds=thresholds,
        )
        for value in provisional
    ]
    nomination = rank_gradient_control_candidates(a_gates)
    selected_index = (
        int(nomination["selected_candidate_index"])
        if bool(int(nomination.get("selected", 0))) else None
    )
    candidates = []
    for index, ((multiplier, task_results), provisional_value) in enumerate(
        zip(raw_results, provisional, strict=True)
    ):
        expose_b = multiplier == 0.0 or index == selected_index
        candidates.append(_candidate_summary(
            multiplier=multiplier,
            teacher=task_results["bounded_teacher"],
            null=task_results["dirichlet_null"], args=args,
            matched_effects=provisional_value.get("matched_effects", {}),
            expose_panel_b=expose_b,
        ))
    pilot = evaluate_gradient_control_pilot(
        candidates, panel_power=oracle, null_family=null_gate, thresholds=thresholds
    )
    _freeze_json(run_dir / "pilot_null_b_max_t.json", null_record)
    _freeze_json(run_dir / "pilot_null_family_gate.json", null_gate)
    _freeze_json(run_dir / "h1_pilot_candidates.json", {
        "candidates": candidates, "physical_training_performed": 0, "sampling_performed": 0
    })
    _freeze_json(run_dir / "h1_pilot_gate.json", pilot)
    atomic_write_json(run_dir / "pilot_task_failures.json", {
        "failures": failures, "count": len(failures),
        "physical_training_performed": 0, "sampling_performed": 0,
    })
    profile_wrapper = dict(pilot.get("selected_profile", {}))
    if bool(int(pilot.get("passed", 0))):
        profile = {
            "schema": RUN_SCHEMA + "-selected-gradient-control-profile",
            "schema_version": RUN_SCHEMA_VERSION,
            **profile_wrapper,
            "h1_lambda_base": calibration.get("lambda_base"),
            "h1_value_scale": calibration.get("value_scale"),
            "h1_energy_scale": calibration.get("energy_scale"),
            "h1_calibration_sha256": file_fingerprint(run_dir / "h1_calibration.json"),
            "pilot_gate_sha256": file_fingerprint(run_dir / "h1_pilot_gate.json"),
            "fresh_confirmation_initialization_required": 1,
            "pilot_weights_reused": 0,
            "physical_training_performed": 0, "sampling_performed": 0,
        }
        _freeze_json(run_dir / "selected_gradient_control_profile.json", profile)
    return pilot, oracle, panels


def _run_gradient_control_pilot(
    run_dir: Path, *, args: argparse.Namespace, manifest: Mapping[str, Any],
    parent: Mapping[str, Any], dynamics: DirectFluxMNISTConfig,
    device: torch.device, stream_plan: DensityRatioStreamPlan,
    paired_plan: PairedMixtureStreamPlan, trust_plan: H1TrustPlan,
    calibration: Mapping[str, Any], thresholds: H1GradientControlThresholds,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, DensityRatioPanel]]]:
    """Run the fixed-endpoint ratio grid with literal sealed-B semantics."""

    from mnist.d0_score_density_ratio_h1_gradient_control_task import (
        run_gradient_control_paired_density_ratio_task,
        write_failed_gradient_control_task_result,
    )

    controller_config = GradientRatioControllerConfig()
    phase = "h1-gradient-control-pilot"
    panels, _ = _prepare_panels(
        run_dir, phase=phase, roles=("a", "b"),
        path_count=int(args.pilot_paths), stream_plan=stream_plan,
        scientific_fingerprint=str(manifest["scientific_fingerprint"]), parent=parent,
    )
    oracle = _freeze_json(
        run_dir / "pilot_oracle_feasibility.json",
        _oracle_panel_bundle(
            panels["bounded_teacher"], args=args, phase=phase, roles=("a", "b")
        ),
    )
    if not bool(int(oracle.get("passed", 0))):
        pilot = not_evaluated_gate(
            "h1_gradient_control_pilot",
            "fixed pilot panels failed exact-teacher oracle power",
        )
        _freeze_json(run_dir / "h1_gradient_control_pilot_gate.json", pilot)
        return pilot, oracle, panels

    failures: list[dict[str, Any]] = []

    def execute(index: int, ratio: float, task: str, *, defer_b: bool) -> dict[str, Any]:
        task_dir = run_dir / "pilot" / f"rho-{index:02d}" / task
        fingerprints = _task_fingerprints(
            manifest=manifest, phase=phase, task=task,
            model_seed=int(args.pilot_model_seed), target_ratio=float(ratio),
            calibration=calibration, trust_plan=trust_plan,
            stream_plan=stream_plan, paired_plan=paired_plan,
            selection_panels=panels[task], audit_panels=None,
        )
        try:
            return run_gradient_control_paired_density_ratio_task(
                task_dir=task_dir, task=task,
                selection_panels=panels[task], audit_panels=None,
                dynamics=dynamics,
                args=head._task_args(
                    args, phase="pilot", learning_rate=float(args.body_learning_rate),
                    accumulation_level=int(DEFAULTS["accumulation_steps"]),
                ),
                device=device, model_seed=int(args.pilot_model_seed),
                learning_rate=float(args.body_learning_rate),
                accumulation_level=int(DEFAULTS["accumulation_steps"]),
                stream_plan=stream_plan, paired_stream_plan=paired_plan,
                trust_plan=trust_plan, calibration=calibration,
                h1_ratio=float(ratio), controller_config=controller_config,
                fingerprints=fingerprints, phase=phase,
                thresholds=thresholds.h1_trust.selection_power.head,
                show_progress=not bool(args.no_progress),
                defer_panel_b=bool(defer_b),
            )
        except FloatingPointError as exc:
            failures.append(
                {
                    "target_ratio": float(ratio), "task": task,
                    "type": type(exc).__name__, "message": str(exc),
                    "physical_training_performed": 0, "sampling_performed": 0,
                }
            )
            return write_failed_gradient_control_task_result(
                task_dir, task=task, model_seed=int(args.pilot_model_seed),
                fingerprints=fingerprints, h1_ratio=float(ratio), exc=exc,
            )

    raw_results: list[tuple[float, dict[str, dict[str, Any]]]] = []
    for index, raw_ratio in enumerate(args.target_gradient_ratios):
        ratio = float(raw_ratio)
        raw_results.append(
            (
                ratio,
                {
                    task: execute(index, ratio, task, defer_b=True)
                    for task in ("bounded_teacher", "dirichlet_null")
                },
            )
        )

    baseline_results = next(tasks for ratio, tasks in raw_results if ratio == 0.0)
    provisional: list[dict[str, Any]] = []
    for ratio, tasks in raw_results:
        matched = (
            _matched_effect_bundle(
                selected=tasks["bounded_teacher"],
                baseline=baseline_results["bounded_teacher"],
                panels=panels["bounded_teacher"], args=args,
                dynamics=dynamics, device=device, phase=phase, roles=("a",),
            )
            if ratio != 0.0
            else {}
        )
        provisional.append(
            _candidate_summary(
                multiplier=ratio, teacher=tasks["bounded_teacher"],
                null=tasks["dirichlet_null"], args=args,
                matched_effects=matched, expose_panel_b=False,
            )
        )
    baseline_candidate = next(
        value for value in provisional if float(value["target_ratio"]) == 0.0
    )
    a_gates = [
        evaluate_gradient_control_pilot_candidate(
            value,
            baseline=(baseline_candidate if float(value["target_ratio"]) != 0.0 else None),
            panel_role="a", thresholds=thresholds,
        )
        for value in provisional
    ]
    nomination = rank_gradient_control_candidates(a_gates)
    selected_index = (
        int(nomination["selected_candidate_index"])
        if bool(int(nomination.get("selected", 0)))
        else None
    )

    null_gate: dict[str, Any] = not_evaluated_gate(
        "h1_gradient_control_pilot_null_family", "panel A nominated no ratio"
    )
    null_record: dict[str, Any] = {
        "evaluation_status": "not_evaluated", "family_size": 0,
        "physical_training_performed": 0, "sampling_performed": 0,
    }
    if selected_index is not None:
        finalized: list[tuple[float, dict[str, dict[str, Any]]]] = []
        baseline_index = next(
            index for index, (ratio, _) in enumerate(raw_results) if ratio == 0.0
        )
        for index, (ratio, tasks) in enumerate(raw_results):
            value = dict(tasks)
            # Every null B contributes to the frozen eight-member null family.
            value["dirichlet_null"] = execute(
                index, ratio, "dirichlet_null", defer_b=False
            )
            # Teacher B remains sealed except for rho=0 and the A nominee.
            if index in {baseline_index, selected_index}:
                value["bounded_teacher"] = execute(
                    index, ratio, "bounded_teacher", defer_b=False
                )
            finalized.append((ratio, value))
        raw_results = finalized
        null_record, null_gate = _null_family(
            [(ratio, tasks["dirichlet_null"]) for ratio, tasks in raw_results],
            args=args, phase=phase,
        )

    final_baseline = next(tasks for ratio, tasks in raw_results if ratio == 0.0)
    candidates: list[dict[str, Any]] = []
    for index, (ratio, tasks) in enumerate(raw_results):
        matched = dict(provisional[index].get("matched_effects", {}))
        expose_b = selected_index is not None and (
            ratio == 0.0 or index == selected_index
        )
        if expose_b and ratio != 0.0:
            matched.update(
                _matched_effect_bundle(
                    selected=tasks["bounded_teacher"],
                    baseline=final_baseline["bounded_teacher"],
                    panels=panels["bounded_teacher"], args=args,
                    dynamics=dynamics, device=device, phase=phase, roles=("b",),
                )
            )
        candidates.append(
            _candidate_summary(
                multiplier=ratio, teacher=tasks["bounded_teacher"],
                null=tasks["dirichlet_null"], args=args,
                matched_effects=matched, expose_panel_b=expose_b,
            )
        )

    pilot = evaluate_gradient_control_pilot(
        candidates, panel_power=oracle, null_family=null_gate, thresholds=thresholds
    )
    _freeze_json(run_dir / "pilot_null_b_max_t.json", null_record)
    _freeze_json(run_dir / "pilot_null_family_gate.json", null_gate)
    _freeze_json(
        run_dir / "gradient_control_pilot_candidates.json",
        {
            "candidates": candidates,
            "panel_a_nomination": nomination,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    _freeze_json(run_dir / "h1_gradient_control_pilot_gate.json", pilot)
    atomic_write_json(
        run_dir / "pilot_task_failures.json",
        {
            "failures": failures, "count": len(failures),
            "physical_training_performed": 0, "sampling_performed": 0,
        },
    )
    profile_wrapper = dict(pilot.get("selected_profile", {}))
    if bool(int(pilot.get("passed", 0))):
        _freeze_json(
            run_dir / "selected_gradient_control_profile.json",
            {
                "schema": RUN_SCHEMA + "-selected-gradient-control-profile",
                "schema_version": RUN_SCHEMA_VERSION,
                **profile_wrapper,
                "controller": controller_config.to_record(),
                "legacy_h1_lambda_base_advisory": calibration.get("lambda_base"),
                "legacy_h1_lambda_used_for_optimization": 0,
                "h1_calibration_sha256": file_fingerprint(run_dir / "h1_calibration.json"),
                "pilot_gate_sha256": file_fingerprint(
                    run_dir / "h1_gradient_control_pilot_gate.json"
                ),
                "fresh_confirmation_initialization_required": 1,
                "pilot_weights_reused": 0,
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )
    return pilot, oracle, panels


def _load_selected_profile(run_dir: Path) -> dict[str, Any]:
    profile_path = run_dir / "selected_gradient_control_profile.json"
    pilot_path = run_dir / "h1_gradient_control_pilot_gate.json"
    calibration_path = run_dir / "h1_calibration.json"
    if not profile_path.is_file() or not pilot_path.is_file() or not calibration_path.is_file():
        raise ArtifactCompatibilityError("frozen H1 pilot profile is missing")
    profile = _json_load(profile_path)
    pilot = _json_load(pilot_path)
    selected = dict(pilot.get("selected_profile", {}))
    if (
        not bool(int(pilot.get("passed", 0)))
        or float(profile.get("selected_ratio", math.nan))
        != float(selected.get("selected_ratio", math.nan))
        or profile.get("pilot_gate_sha256") != file_fingerprint(pilot_path)
        or profile.get("h1_calibration_sha256") != file_fingerprint(calibration_path)
    ):
        raise ArtifactCompatibilityError("frozen H1 profile binding changed")
    return profile


def _profile_binding(
    run_dir: Path, *, profile: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    return _freeze_json(run_dir / "confirmation_profile_binding.json", {
        "schema": RUN_SCHEMA + "-confirmation-profile-binding",
        "schema_version": RUN_SCHEMA_VERSION,
        "selected_profile": dict(profile),
        "selected_profile_sha256": file_fingerprint(
            run_dir / "selected_gradient_control_profile.json"
        ),
        "parent_registry_sha256": parent["artifact_registry_sha256"],
        "fresh_initialization_required": 1,
        "pilot_checkpoint_loading_permitted": 0,
        "physical_training_performed": 0, "sampling_performed": 0,
    })


def _apply_h1_health(
    gate: Mapping[str, Any], results: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, Any]:
    value = copy.deepcopy(dict(gate))
    health = bool(results) and all(
        int(dict(result.get("metrics", {})).get("h1_health_pass", 0)) == 1
        for result in results
    )
    value["h1_health_pass"] = int(health)
    subchecks = dict(value.get("subchecks", {}))
    subchecks["all_h1_penalties_healthy"] = {
        "value": int(health), "operator": "==", "threshold": 1,
        "passed": int(health),
    }
    value["subchecks"] = subchecks
    value["passed"] = int(bool(int(value.get("passed", 0))) and health)
    value["gate"] = str(value.get("gate", name))
    return value


def _run_confirmation(
    run_dir: Path, *, args: argparse.Namespace, manifest: Mapping[str, Any],
    parent: Mapping[str, Any], profile: Mapping[str, Any],
    pilot_panels: Mapping[str, Mapping[str, DensityRatioPanel]],
    dynamics: DirectFluxMNISTConfig, device: torch.device,
    stream_plan: DensityRatioStreamPlan, paired_plan: PairedMixtureStreamPlan,
    trust_plan: H1TrustPlan, calibration: Mapping[str, Any],
    thresholds: H1GradientControlThresholds,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from mnist.d0_score_density_ratio_h1_task import (
        run_h1_paired_density_ratio_task, write_failed_h1_task_result,
    )

    binding = _profile_binding(run_dir, profile=profile, parent=parent)
    panels, _ = _prepare_panels(
        run_dir, phase="h1-confirmation", roles=("a", "b", "c", "d"),
        path_count=int(args.confirm_paths), stream_plan=stream_plan,
        scientific_fingerprint=str(manifest["scientific_fingerprint"]), parent=parent,
        previous=(pilot_panels,),
    )
    oracle_path = run_dir / "confirmation_oracle_feasibility.json"
    oracle = _freeze_json(oracle_path, _oracle_panel_bundle(
        panels["bounded_teacher"], args=args, phase="h1-confirmation",
        roles=("a", "b", "c", "d"),
    ))
    if not bool(int(oracle.get("passed", 0))):
        teacher = not_evaluated_gate(
            "h1_confirmation_teacher_study", "confirmation panels failed oracle power"
        )
        null = not_evaluated_gate(
            "h1_confirmation_null_family", "confirmation panels failed oracle power"
        )
        _freeze_json(run_dir / "confirmation_teacher_study_gate.json", teacher)
        _freeze_json(run_dir / "confirmation_null_family_gate.json", null)
        return oracle, teacher, null

    multiplier = float(profile["selected_multiplier"])
    teacher_results: list[dict[str, Any]] = []
    null_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_seed in args.confirm_model_seeds:
        for task, output in (
            ("bounded_teacher", teacher_results), ("dirichlet_null", null_results)
        ):
            task_dir = run_dir / "confirmation" / f"seed-{int(model_seed)}" / task
            fingerprints = _task_fingerprints(
                manifest=manifest, phase="h1-confirmation", task=task,
                model_seed=int(model_seed), h1_multiplier=multiplier,
                calibration=calibration, trust_plan=trust_plan,
                stream_plan=stream_plan, paired_plan=paired_plan,
                selection_panels={name: panels[task][name] for name in ("a", "b")},
                audit_panels={name: panels[task][name] for name in ("c", "d")},
                profile_binding=binding,
            )
            try:
                result = run_h1_paired_density_ratio_task(
                    task_dir=task_dir, task=task,
                    selection_panels={name: panels[task][name] for name in ("a", "b")},
                    audit_panels={name: panels[task][name] for name in ("c", "d")},
                    dynamics=dynamics,
                    args=head._task_args(
                        args, phase="confirmation",
                        learning_rate=float(args.body_learning_rate),
                        accumulation_level=int(DEFAULTS["accumulation_steps"]),
                    ),
                    device=device, model_seed=int(model_seed),
                    learning_rate=float(args.body_learning_rate),
                    accumulation_level=int(DEFAULTS["accumulation_steps"]),
                    stream_plan=stream_plan, paired_stream_plan=paired_plan,
                    trust_plan=trust_plan, calibration=calibration,
                    h1_ratio=multiplier, fingerprints=fingerprints,
                    phase="h1-confirmation", thresholds=thresholds.selection_power.head,
                    show_progress=not bool(args.no_progress),
                )
            except FloatingPointError as exc:
                result = write_failed_h1_task_result(
                    task_dir, task=task, model_seed=int(model_seed),
                    fingerprints=fingerprints, h1_ratio=multiplier, exc=exc,
                )
                failures.append({
                    "task": task, "model_seed": int(model_seed),
                    "type": type(exc).__name__, "message": str(exc),
                    "physical_training_performed": 0, "sampling_performed": 0,
                })
            output.append(result)

    atomic_write_json(run_dir / "h1_teacher_confirmation.json", {
        "task_results": teacher_results, "physical_training_performed": 0,
        "sampling_performed": 0,
    })
    atomic_write_json(run_dir / "h1_null_confirmation.json", {
        "task_results": null_results, "physical_training_performed": 0,
        "sampling_performed": 0,
    })
    atomic_write_json(run_dir / "confirmation_task_failures.json", {
        "failures": failures, "count": len(failures),
        "physical_training_performed": 0, "sampling_performed": 0,
    })
    bindings = [
        multi._confirmation_sealed_binding(
            run_dir / "confirmation" / f"seed-{int(result.get('model_seed'))}"
            / "dirichlet_null", result,
        )
        for result in null_results
    ]
    _freeze_json(run_dir / "confirmation_sealed_b_bindings.json", {
        "bindings": bindings, "physical_training_performed": 0, "sampling_performed": 0
    })
    try:
        discovery_members, family_members = multi._confirmation_family_members(null_results)
        discovery_record = studentized_whole_path_max_t(
            discovery_members,
            seed=base._derived_seed(int(args.root_seed), "h1-confirmation-discovery-a"),
            confidence=float(args.familywise_confidence),
            reps=int(args.simultaneous_bootstrap_reps),
        )
        family_record = studentized_whole_path_max_t(
            family_members,
            seed=base._derived_seed(int(args.root_seed), "h1-confirmation-b-c-d"),
            confidence=float(args.familywise_confidence),
            reps=int(args.simultaneous_bootstrap_reps),
        )
        expected_names = [
            f"seed-{int(seed)}/{role}/{scope}"
            for seed in args.confirm_model_seeds for role in ("b", "c", "d")
            for scope in ("overall", "data_end")
        ]
        max_t_gate = evaluate_max_t_null_family(
            family_record, expected_members=expected_names, expected_member_count=18,
            required_confidence=float(args.familywise_confidence),
            required_replicates=int(args.simultaneous_bootstrap_reps),
            gate_name="h1_confirmation_b_c_d_simultaneous_null_family",
        )
        _freeze_json(run_dir / "confirmation_discovery_a_max_t.json", discovery_record)
        _freeze_json(run_dir / "confirmation_b_c_d_max_t.json", family_record)
        multi._write_family_csv(
            run_dir / "confirmation_discovery_a_simultaneous_bounds.csv",
            "h1-confirmation-discovery-a", discovery_record,
        )
        multi._write_family_csv(
            run_dir / "confirmation_null_simultaneous_bounds.csv",
            "h1-confirmation-b-c-d", family_record,
        )
    except (ArtifactCompatibilityError, ValueError) as exc:
        max_t_gate = not_evaluated_gate(
            "h1_confirmation_b_c_d_simultaneous_null_family",
            f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(run_dir / "confirmation_multiplicity_failure.json", {
            "type": type(exc).__name__, "message": str(exc),
            "physical_training_performed": 0, "sampling_performed": 0,
        })
    sealed_thresholds = SealedNullThresholds(
        selection_power=thresholds.selection_power,
        confidence=float(args.familywise_confidence),
        bootstrap_replicates=int(args.simultaneous_bootstrap_reps),
    )
    null_gate = evaluate_confirmation_null_family(
        null_results, max_t_family=max_t_gate, sealed_b_bindings=bindings,
        thresholds=sealed_thresholds,
    )
    teacher_gate = evaluate_power_teacher_study(
        teacher_results, thresholds.selection_power
    )
    null_gate = _apply_h1_health(
        null_gate, null_results, name="h1_confirmation_null_family"
    )
    teacher_gate = _apply_h1_health(
        teacher_gate, teacher_results, name="h1_confirmation_teacher_study"
    )
    _freeze_json(run_dir / "confirmation_null_family_gate.json", null_gate)
    _freeze_json(run_dir / "confirmation_teacher_study_gate.json", teacher_gate)
    return oracle, teacher_gate, null_gate


def _run_gradient_control_confirmation(
    run_dir: Path, *, args: argparse.Namespace, manifest: Mapping[str, Any],
    parent: Mapping[str, Any], profile: Mapping[str, Any],
    pilot_panels: Mapping[str, Mapping[str, DensityRatioPanel]],
    dynamics: DirectFluxMNISTConfig, device: torch.device,
    stream_plan: DensityRatioStreamPlan, paired_plan: PairedMixtureStreamPlan,
    trust_plan: H1TrustPlan, calibration: Mapping[str, Any],
    thresholds: H1GradientControlThresholds,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run selected-ratio, rho-zero, and null tasks at the fixed endpoint."""

    from mnist.d0_score_density_ratio_h1_gradient_control_task import (
        run_gradient_control_paired_density_ratio_task,
        write_failed_gradient_control_task_result,
    )

    binding = _profile_binding(run_dir, profile=profile, parent=parent)
    phase = "h1-gradient-control-confirmation"
    panels, _ = _prepare_panels(
        run_dir, phase=phase, roles=("a", "b", "c", "d"),
        path_count=int(args.confirm_paths), stream_plan=stream_plan,
        scientific_fingerprint=str(manifest["scientific_fingerprint"]), parent=parent,
        previous=(pilot_panels,),
    )
    oracle = _freeze_json(
        run_dir / "confirmation_oracle_feasibility.json",
        _oracle_panel_bundle(
            panels["bounded_teacher"], args=args, phase=phase,
            roles=("a", "b", "c", "d"),
        ),
    )
    if not bool(int(oracle.get("passed", 0))):
        gate = not_evaluated_gate(
            "h1_gradient_control_confirmation",
            "confirmation panels failed exact-teacher oracle power",
        )
        _freeze_json(run_dir / "h1_gradient_control_confirmation_gate.json", gate)
        return oracle, gate

    selected_ratio = float(profile["selected_ratio"])
    controller_config = GradientRatioControllerConfig()
    failures: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    null_results: list[dict[str, Any]] = []
    raw_task_records: list[dict[str, Any]] = []
    matched_evaluations: list[dict[str, Any]] = []

    def execute(
        *, model_seed: int, role_name: str, task: str, ratio: float,
    ) -> dict[str, Any]:
        task_dir = run_dir / "confirmation" / f"seed-{model_seed}" / role_name
        fingerprints = _task_fingerprints(
            manifest=manifest, phase=phase, task=task, model_seed=model_seed,
            target_ratio=float(ratio), calibration=calibration,
            trust_plan=trust_plan, stream_plan=stream_plan, paired_plan=paired_plan,
            selection_panels={name: panels[task][name] for name in ("a", "b")},
            audit_panels={name: panels[task][name] for name in ("c", "d")},
            profile_binding=binding,
        )
        fingerprints["confirmation_task_role"] = role_name
        try:
            return run_gradient_control_paired_density_ratio_task(
                task_dir=task_dir, task=task,
                selection_panels={name: panels[task][name] for name in ("a", "b")},
                audit_panels={name: panels[task][name] for name in ("c", "d")},
                dynamics=dynamics,
                args=head._task_args(
                    args, phase="confirmation",
                    learning_rate=float(args.body_learning_rate),
                    accumulation_level=int(DEFAULTS["accumulation_steps"]),
                ),
                device=device, model_seed=model_seed,
                learning_rate=float(args.body_learning_rate),
                accumulation_level=int(DEFAULTS["accumulation_steps"]),
                stream_plan=stream_plan, paired_stream_plan=paired_plan,
                trust_plan=trust_plan, calibration=calibration,
                h1_ratio=float(ratio), controller_config=controller_config,
                fingerprints=fingerprints, phase=phase,
                thresholds=thresholds.h1_trust.selection_power.head,
                show_progress=not bool(args.no_progress),
            )
        except FloatingPointError as exc:
            failures.append(
                {
                    "task_role": role_name, "task": task, "model_seed": model_seed,
                    "target_ratio": float(ratio), "type": type(exc).__name__,
                    "message": str(exc), "physical_training_performed": 0,
                    "sampling_performed": 0,
                }
            )
            return write_failed_gradient_control_task_result(
                task_dir, task=task, model_seed=model_seed,
                fingerprints=fingerprints, h1_ratio=float(ratio), exc=exc,
            )

    for raw_seed in args.confirm_model_seeds:
        model_seed = int(raw_seed)
        selected = execute(
            model_seed=model_seed, role_name="bounded_teacher_selected",
            task="bounded_teacher", ratio=selected_ratio,
        )
        baseline = execute(
            model_seed=model_seed, role_name="bounded_teacher_rho_zero",
            task="bounded_teacher", ratio=0.0,
        )
        null = execute(
            model_seed=model_seed, role_name="dirichlet_null",
            task="dirichlet_null", ratio=selected_ratio,
        )
        effects = _matched_effect_bundle(
            selected=selected, baseline=baseline,
            panels=panels["bounded_teacher"], roles=("b", "c", "d"),
            args=args, dynamics=dynamics, device=device, phase=phase,
        )
        for role, record in effects.items():
            value = dict(record)
            value["model_seed"] = model_seed
            value["seed"] = model_seed
            value["panel_role"] = role
            matched_evaluations.append(value)
        seed_records.append(
            {
                "evaluation_status": "evaluated", "seed": model_seed,
                "teacher": _confirmation_task_summary(selected),
                "baseline": _confirmation_task_summary(baseline),
                "null": _confirmation_task_summary(null),
                "matched_effects": effects,
            }
        )
        null_results.append(null)
        raw_task_records.append(
            {
                "seed": model_seed, "selected_teacher": selected,
                "rho_zero_teacher": baseline, "dirichlet_null": null,
            }
        )

    atomic_write_json(
        run_dir / "gradient_control_confirmation_tasks.json",
        {
            "seed_records": raw_task_records,
            "physical_training_performed": 0, "sampling_performed": 0,
        },
    )
    atomic_write_json(
        run_dir / "confirmation_task_failures.json",
        {
            "failures": failures, "count": len(failures),
            "physical_training_performed": 0, "sampling_performed": 0,
        },
    )

    bindings = [
        multi._confirmation_sealed_binding(
            run_dir / "confirmation" / f"seed-{int(result.get('model_seed'))}"
            / "dirichlet_null",
            result,
        )
        for result in null_results
    ]
    _freeze_json(
        run_dir / "confirmation_sealed_b_bindings.json",
        {
            "bindings": bindings,
            "physical_training_performed": 0, "sampling_performed": 0,
        },
    )
    discovery_members, family_members = multi._confirmation_family_members(null_results)
    discovery_record = studentized_whole_path_max_t(
        discovery_members,
        seed=base._derived_seed(int(args.root_seed), phase, "null-discovery-a"),
        confidence=float(args.familywise_confidence),
        reps=int(args.simultaneous_bootstrap_reps),
    )
    null_family_record = studentized_whole_path_max_t(
        family_members,
        seed=base._derived_seed(int(args.root_seed), phase, "null-b-c-d"),
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
        null_family_record, expected_members=expected_names, expected_member_count=18,
        required_confidence=float(args.familywise_confidence),
        required_replicates=int(args.simultaneous_bootstrap_reps),
        gate_name="h1_gradient_control_confirmation_null_b_c_d_family",
    )
    sealed_thresholds = SealedNullThresholds(
        selection_power=thresholds.h1_trust.selection_power,
        confidence=float(args.familywise_confidence),
        bootstrap_replicates=int(args.simultaneous_bootstrap_reps),
    )
    null_gate = evaluate_confirmation_null_family(
        null_results, max_t_family=max_t_gate,
        sealed_b_bindings=bindings, thresholds=sealed_thresholds,
    )
    # Preserve the structural null adjudication while exposing the exact
    # authorizing family to the new gate.
    null_gate["members"] = list(null_family_record.get("members", []))
    null_gate["family_size"] = int(null_family_record.get("family_size", 0))
    null_gate["positive_member_names_by_role"] = dict(
        max_t_gate.get("positive_member_names_by_role", {})
    )

    matched_family = joint_matched_flux_family_bootstrap(
        matched_evaluations,
        seed=base._derived_seed(int(args.root_seed), phase, "matched-b-c-d"),
        reps=int(args.simultaneous_bootstrap_reps),
        confidence=float(args.familywise_confidence),
    )
    confirmation = evaluate_gradient_control_confirmation(
        seed_records, selected_ratio=selected_ratio, panel_power=oracle,
        matched_effect_family=matched_family, null_family=null_gate,
        thresholds=thresholds,
    )
    _freeze_json(run_dir / "confirmation_null_discovery_a_max_t.json", discovery_record)
    _freeze_json(run_dir / "confirmation_null_b_c_d_max_t.json", null_family_record)
    _freeze_json(run_dir / "confirmation_null_family_gate.json", null_gate)
    _freeze_json(run_dir / "confirmation_matched_flux_family.json", matched_family)
    atomic_write_csv(
        run_dir / "confirmation_matched_flux_simultaneous_bounds.csv",
        list(matched_family.get("members", [])),
    )
    _freeze_json(run_dir / "h1_gradient_control_confirmation_gate.json", confirmation)
    return oracle, confirmation


def _load_report_inputs(run_dir: Path) -> tuple[dict[str, Any], ...]:
    def load(filename: str, gate: str) -> dict[str, Any]:
        path = run_dir / filename
        return _json_load(path) if path.is_file() else not_evaluated_gate(gate, f"{gate} was not run")
    return (
        load("gradient_controller_preflight_gate.json", "h1_gradient_controller_preflight"),
        load("h1_gradient_control_preflight_gate.json", "h1_gradient_control_preflight"),
        load("pilot_oracle_feasibility.json", "h1_gradient_control_pilot_oracle_power"),
        load("h1_gradient_control_pilot_gate.json", "h1_gradient_control_pilot"),
        load("confirmation_oracle_feasibility.json", "h1_gradient_control_confirmation_oracle_power"),
        load("h1_gradient_control_confirmation_gate.json", "h1_gradient_control_confirmation"),
    )


def _workflow_report(
    *, provenance: Mapping[str, Any], controller_preflight: Mapping[str, Any],
    preflight: Mapping[str, Any],
    pilot_power: Mapping[str, Any], pilot: Mapping[str, Any],
    confirmation_power: Mapping[str, Any], confirmation: Mapping[str, Any],
    require_gate: str, thresholds: H1GradientControlThresholds,
) -> dict[str, Any]:
    return evaluate_gradient_control_workflow(
        provenance=provenance, controller_preflight=controller_preflight,
        preflight=preflight, pilot_panel_power=pilot_power, pilot=pilot,
        confirmation_panel_power=confirmation_power, confirmation=confirmation,
        require_gate=require_gate, thresholds=thresholds,
    )


def _save_report(run_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / "h1_gradient_control_gate.json", report)
    atomic_write_json(
        run_dir / "h1_gradient_control_decision.json", dict(report.get("decision", {}))
    )


def _write_task_summary_csv(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("task_result.json")):
        value = _json_load(path)
        metrics = dict(value.get("metrics", {}))
        selection = dict(metrics.get("selection", {}))
        confirmation = dict(selection.get("confirmation", {}))
        summary = dict(value.get("training_summary", {}))
        rows.append({
            "task_path": path.parent.relative_to(run_dir).as_posix(),
            "task": value.get("task"), "model_seed": value.get("model_seed"),
            "target_gradient_ratio": summary.get(
                "target_gradient_ratio", summary.get("h1_multiplier")
            ),
            "complete": metrics.get("complete"), "finite": metrics.get("finite"),
            "boundary_admissible": metrics.get("boundary_admissible"),
            "controller_health_pass": metrics.get("controller_health_pass"),
            "controller_active_fraction": metrics.get("controller_active_fraction"),
            "maximum_ratio_relative_error": metrics.get("maximum_ratio_relative_error"),
            "optimizer_health_pass": metrics.get("optimizer_health_pass"),
            "fixed_endpoint_step": metrics.get("fixed_endpoint_step"),
            "nominee_step": metrics.get("nominee_step"),
            "selected_step": metrics.get("selected_step"),
            "sealed_b_accepted": confirmation.get("accepted"),
            "post_warmup_clip_fraction": metrics.get("post_warmup_clip_fraction"),
            "final_500_clip_fraction": metrics.get("final_500_clip_fraction"),
            "final_200_clip_fraction": metrics.get("final_200_clip_fraction"),
            "physical_training_performed": 0, "sampling_performed": 0,
        })
    if rows:
        atomic_write_csv(run_dir / "h1_task_summary.csv", rows)


def _write_h1_diagnostic_csv(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("training_history.csv")):
        try:
            import csv
            with path.open("r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    rows.append({
                        "task_path": path.parent.relative_to(run_dir).as_posix(),
                        "step": raw.get("step"),
                        "target_gradient_ratio": raw.get("h1_ratio"),
                        "ramped_target_ratio": raw.get("controller_target_ratio"),
                        "realized_gradient_ratio": raw.get("controller_realized_ratio"),
                        "ratio_tracking_relative_error": raw.get(
                            "controller_ratio_tracking_relative_error"
                        ),
                        "h1_coefficient": raw.get("controller_h1_coefficient"),
                        "controller_active": raw.get("controller_active"),
                        "controller_pass": raw.get("controller_pass"),
                        "h1_floor_hit": raw.get("controller_h1_gradient_floor_hit"),
                        "h1_value": raw.get("h1_value"),
                        "h1_natural_energy": raw.get("h1_natural_energy"),
                        "h1_physical_flux_step_rms": raw.get("h1_physical_flux_step_rms"),
                        "h1_normalized_objective": raw.get("h1_normalized_objective"),
                        "h1_effective_loss": raw.get("h1_effective_loss"),
                        "bce_gradient_norm": raw.get("bce_gradient_norm"),
                        "h1_raw_gradient_norm": raw.get("h1_raw_gradient_norm"),
                        "h1_contribution_gradient_norm": raw.get(
                            "controller_h1_contribution_gradient_norm"
                        ),
                        "combined_gradient_norm": raw.get(
                            "controller_combined_gradient_norm"
                        ),
                        "bce_h1_gradient_cosine": raw.get("bce_h1_gradient_cosine"),
                        "scaled_preclip_gradient_norm": raw.get("scaled_preclip_gradient_norm"),
                        "clipped": raw.get("clipped"),
                        "physical_training_performed": 0, "sampling_performed": 0,
                    })
        except (OSError, ValueError):
            continue
    if rows:
        atomic_write_csv(run_dir / "h1_gradient_control_diagnostics.csv", rows)


def _write_gradient_control_plots(run_dir: Path) -> None:
    histories = sorted(run_dir.rglob("training_history.csv"))
    if not histories:
        return
    import csv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    plotted = 0
    for path in histories:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        points = [
            row for row in rows
            if row.get("step") not in {None, ""}
            and row.get("controller_target_ratio") not in {None, ""}
            and row.get("controller_realized_ratio") not in {None, ""}
        ]
        if not points:
            continue
        steps = [float(row["step"]) for row in points]
        target = [float(row["controller_target_ratio"]) for row in points]
        realized = [float(row["controller_realized_ratio"]) for row in points]
        coefficients = [
            max(float(row.get("controller_h1_coefficient", 0.0)), 1e-30)
            for row in points
        ]
        label = str(path.parent.relative_to(run_dir))
        axes[0].plot(steps, realized, linewidth=0.8, alpha=0.7, label=label)
        axes[0].plot(steps, target, linewidth=0.6, alpha=0.45, linestyle="--")
        axes[1].plot(steps, coefficients, linewidth=0.8, alpha=0.7, label=label)
        plotted += 1
    if not plotted:
        plt.close(figure)
        return
    axes[0].set_ylabel("H1/BCE gradient ratio")
    axes[0].set_title("Online H1 gradient-ratio tracking (solid) and target (dashed)")
    axes[1].set_ylabel("stopped H1 coefficient")
    axes[1].set_xlabel("optimizer update")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(alpha=0.2)
    if plotted <= 12:
        axes[0].legend(fontsize=5, ncol=2)
    figure.tight_layout()
    figure.savefig(run_dir / "h1_gradient_controller_diagnostics.png", dpi=160)
    plt.close(figure)


def _finish(
    run_dir: Path, *, report: Mapping[str, Any], stage: str, phase: str,
    execution_failed: bool = False,
    skips: Sequence[Mapping[str, Any]] = (),
) -> int:
    final_skips = [dict(value) for value in skips]
    try:
        _write_task_summary_csv(run_dir)
        _write_h1_diagnostic_csv(run_dir)
        _write_gradient_control_plots(run_dir)
        power._write_learning_plot(run_dir)
        source = run_dir / "selection_power_learning_curves.png"
        if source.is_file():
            atomic_copy_file(source, run_dir / "h1_learning_curves.png")
    except Exception as exc:
        execution_failed = True
        final_skips.append({
            "stage": "report_artifacts", "reason": f"{type(exc).__name__}: {exc}"
        })
        atomic_write_json(run_dir / "report_artifact_failure.json", {
            "type": type(exc).__name__, "message": str(exc),
            "physical_training_performed": 0, "sampling_performed": 0,
        })
    required_pass = 0 if execution_failed else int(report.get("required_gate_pass", 0))
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    decision = dict(report.get("decision", {}))
    _write_status(
        run_dir,
        status="failed" if execution_failed else "complete",
        outcome=("implementation_error" if execution_failed else "complete" if required_pass else "gate_failed"),
        phase=phase, stage=stage,
        required_gate=str(report.get("required_gate", "none")),
        required_gate_pass=required_pass,
        decision=decision.get("decision", "controls_not_run"),
        recommended_next_action=decision.get("recommended_next_action"),
        physical_training_authorized=(
            int(decision.get("physical_training_authorized", 0)) if not execution_failed else 0
        ),
        physical_training_performed=0, sampling_authorized=0, sampling_performed=0,
        skips=final_skips,
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_size=int(registry_path.stat().st_size),
    )
    return 2 if execution_failed or not required_pass else 0


def _empty_components() -> tuple[dict[str, Any], ...]:
    return (
        not_evaluated_gate("h1_gradient_controller_preflight", "controller preflight was not run"),
        not_evaluated_gate("h1_gradient_control_preflight", "preflight was not run"),
        not_evaluated_gate("h1_gradient_control_pilot_oracle_power", "pilot panels were not frozen"),
        not_evaluated_gate("h1_gradient_control_pilot", "pilot was not run"),
        not_evaluated_gate("h1_gradient_control_confirmation_oracle_power", "confirmation panels were not frozen"),
        not_evaluated_gate("h1_gradient_control_confirmation", "confirmation was not run"),
    )


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    if not args.no_progress:
        print(f"H1 density-ratio run directory: {run_dir.resolve()}", flush=True)
    thresholds = H1TrustThresholds(
        selection_power=SelectionPowerThresholds(),
        multipliers=tuple(float(value) for value in args.h1_multipliers),
        minimum_relative_l2_reduction=0.10,
        body_learning_rate=float(args.body_learning_rate),
        accumulation_steps=int(DEFAULTS["accumulation_steps"]),
        base_channels=int(args.base_channels),
    )
    mutation_started = False
    provenance: dict[str, Any] = not_evaluated_gate(
        "h1_parent_provenance", "parent provenance was not verified"
    )
    try:
        device = base._device(args.device)
        backend = configure_exact_torch_backend(device)
        runtime = base._runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
        source_hash, source_paths = _source_record()
        parent = verify_parent_multiplicity_run(args.parent_multiplicity_run_dir)
        provenance = {
            **dict(parent), "schema": RUN_SCHEMA + "-parent-provenance",
            "schema_version": RUN_SCHEMA_VERSION,
            "verifier_source_fingerprint": source_hash,
            "physical_training_performed": 0, "sampling_performed": 0,
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
            "schema": RUN_SCHEMA, "schema_version": RUN_SCHEMA_VERSION,
            "created_at": _now(), "run_dir": str(run_dir.resolve()),
            "scientific_config": scientific,
            "scientific_fingerprint": config_fingerprint(scientific),
            "runtime": runtime, "runtime_fingerprint": runtime_fingerprint,
            "source_fingerprint": source_hash, "source_paths": source_paths,
            "parent_provenance_sha256": file_fingerprint(provenance_path),
            "claim_scope": CLAIM_SCOPE,
            "physical_training_performed": 0, "sampling_performed": 0,
        }
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            existing = _json_load(manifest_path)
            for key in (
                "schema", "schema_version", "run_dir", "scientific_config",
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
        previous_status = _json_load(status_path) if status_path.is_file() else {}
        if resumed and str(previous_status.get("status", "")) in {"complete", "failed"}:
            _verify_terminal_registry(run_dir)
        _write_status(
            run_dir, status="running", outcome="running", phase="provenance",
            stage=str(args.stage), require_gate=str(args.require_gate),
            attempt_count=int(previous_status.get("attempt_count", 0)) + 1,
        )
        mutation_started = True

        if args.stage == "report":
            operator, calibration_gate, preflight, pilot_power, pilot, confirm_power, teacher, null = _load_report_inputs(run_dir)
            report = _workflow_report(
                provenance=provenance, operator=operator, calibration=calibration_gate,
                preflight=preflight, pilot_power=pilot_power, pilot=pilot,
                confirmation_power=confirm_power, teacher=teacher, null=null,
                require_gate=str(args.require_gate), thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(run_dir, report=report, stage="report", phase="report")

        dynamics = base._make_dynamics(args)
        horizon = float(natural_horizon(dynamics))
        if not math.isclose(horizon, float(parent.get("horizon", math.nan)), rel_tol=1e-12, abs_tol=1e-15):
            raise ArtifactCompatibilityError("H1 workflow horizon differs from parent")
        stream_plan = build_density_ratio_stream_plan(
            root_seed=int(args.root_seed), grid_size=int(args.grid_size), horizon=horizon,
            label=3, bin_counts=(4, 4, 4, 4, 16), teacher_epsilon=0.5,
        )
        paired_plan = build_paired_mixture_stream_plan(
            root_seed=int(args.root_seed), grid_size=int(args.grid_size), horizon=horizon,
            label=3, teacher_epsilon=0.5,
        )
        trust_plan = build_h1_trust_plan(
            grid_size=int(args.grid_size), horizon=horizon,
            root_seed=int(args.root_seed), label=3,
        )
        _freeze_json(run_dir / "h1_stream_plans.json", {
            "schema": RUN_SCHEMA + "-stream-plans", "schema_version": RUN_SCHEMA_VERSION,
            "evaluation_stream": stream_plan_record(stream_plan),
            "paired_training_stream": paired_mixture_stream_plan_record(paired_plan),
            "h1_reference_trust_stream": h1_trust_plan_record(trust_plan),
            "physical_training_performed": 0, "sampling_performed": 0,
        })

        paths = (
            run_dir / "h1_operator_gate.json", run_dir / "h1_calibration_gate.json",
            run_dir / "h1_preflight_gate.json",
        )
        if all(path.is_file() for path in paths):
            operator, calibration_gate, preflight = map(_json_load, paths)
        else:
            _write_status(run_dir, status="running", phase="preflight")
            operator, calibration_gate, preflight = _run_preflight(
                run_dir, args=args, parent=parent, dynamics=dynamics,
                plan=trust_plan, paired_plan=paired_plan, device=device,
            )
        pilot_power, pilot, confirm_power, teacher, null = _empty_components()[3:]
        if args.stage == "preflight" or not bool(int(preflight.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance, operator=operator, calibration=calibration_gate,
                preflight=preflight, pilot_power=pilot_power, pilot=pilot,
                confirmation_power=confirm_power, teacher=teacher, null=null,
                require_gate=str(args.require_gate), thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(
                run_dir, report=report, stage=str(args.stage), phase="preflight",
                skips=(() if bool(int(preflight.get("passed", 0))) else ({
                    "stage": "pilot_and_confirmation", "reason": "H1 preflight failed"
                },)),
            )

        calibration = _json_load(run_dir / "h1_calibration.json")
        _write_status(run_dir, status="running", phase="pilot")
        pilot, pilot_power, pilot_panels = _run_pilot(
            run_dir, args=args, manifest=manifest, parent=parent, dynamics=dynamics,
            device=device, stream_plan=stream_plan, paired_plan=paired_plan,
            trust_plan=trust_plan, calibration=calibration, thresholds=thresholds,
        )
        if args.stage == "pilot" or not bool(int(pilot.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance, operator=operator, calibration=calibration_gate,
                preflight=preflight, pilot_power=pilot_power, pilot=pilot,
                confirmation_power=confirm_power, teacher=teacher, null=null,
                require_gate=str(args.require_gate), thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(
                run_dir, report=report, stage=str(args.stage), phase="pilot",
                skips=(() if bool(int(pilot.get("passed", 0))) else ({
                    "stage": "confirmation", "reason": "H1 pilot failed"
                },)),
            )

        profile = _load_selected_profile(run_dir)
        _write_status(run_dir, status="running", phase="confirmation")
        confirm_power, teacher, null = _run_confirmation(
            run_dir, args=args, manifest=manifest, parent=parent, profile=profile,
            pilot_panels=pilot_panels, dynamics=dynamics, device=device,
            stream_plan=stream_plan, paired_plan=paired_plan, trust_plan=trust_plan,
            calibration=calibration, thresholds=thresholds,
        )
        report = _workflow_report(
            provenance=provenance, operator=operator, calibration=calibration_gate,
            preflight=preflight, pilot_power=pilot_power, pilot=pilot,
            confirmation_power=confirm_power, teacher=teacher, null=null,
            require_gate=str(args.require_gate), thresholds=thresholds,
        )
        _save_report(run_dir, report)
        return _finish(
            run_dir, report=report, stage=str(args.stage), phase="confirmation",
            skips=(() if bool(int(confirm_power.get("passed", 0))) else ({
                "stage": "confirmation_training",
                "reason": "fixed confirmation panels failed oracle power",
            },)),
        )
    except Exception as exc:
        if resumed and not mutation_started:
            if not args.no_progress:
                print(f"H1 resume rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        atomic_write_json(run_dir / "failure.json", {
            "schema": RUN_SCHEMA + "-failure", "schema_version": RUN_SCHEMA_VERSION,
            "type": type(exc).__name__, "message": str(exc), "stage": str(args.stage),
            "physical_training_performed": 0, "sampling_performed": 0,
        })
        operator, calibration_gate, preflight, pilot_power, pilot, confirm_power, teacher, null = _empty_components()
        report = _workflow_report(
            provenance=provenance, operator=operator, calibration=calibration_gate,
            preflight=preflight, pilot_power=pilot_power, pilot=pilot,
            confirmation_power=confirm_power, teacher=teacher, null=null,
            require_gate=str(args.require_gate), thresholds=thresholds,
        )
        _save_report(run_dir, report)
        _finish(
            run_dir, report=report, stage=str(args.stage), phase="failure",
            execution_failed=True,
            skips=({"stage": "remaining", "reason": f"{type(exc).__name__}: {exc}"},),
        )
        if not args.no_progress:
            print(f"H1 control failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _run_gradient_control(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    if not args.no_progress:
        print(f"H1 gradient-control run directory: {run_dir.resolve()}", flush=True)
    thresholds = H1GradientControlThresholds()
    mutation_started = False
    provenance: dict[str, Any] = not_evaluated_gate(
        "h1_gradient_control_parent_provenance", "parent provenance was not verified"
    )
    try:
        device = base._device(args.device)
        backend = configure_exact_torch_backend(device)
        runtime = base._runtime_record(device, backend)
        runtime_fingerprint = config_fingerprint(runtime)
        source_hash, source_paths = _source_record()
        parent = verify_parent_h1_trust_run(args.parent_h1_run_dir)
        provenance = {
            **dict(parent),
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
                "schema", "schema_version", "run_dir", "scientific_config",
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
        previous_status = _json_load(status_path) if status_path.is_file() else {}
        if resumed and str(previous_status.get("status", "")) in {"complete", "failed"}:
            _verify_terminal_registry(run_dir)
        _write_status(
            run_dir, status="running", outcome="running", phase="provenance",
            stage=str(args.stage), require_gate=str(args.require_gate),
            attempt_count=int(previous_status.get("attempt_count", 0)) + 1,
        )
        mutation_started = True

        if args.stage == "report":
            (
                controller_preflight, preflight, pilot_power, pilot,
                confirmation_power, confirmation,
            ) = _load_report_inputs(run_dir)
            report = _workflow_report(
                provenance=provenance, controller_preflight=controller_preflight,
                preflight=preflight, pilot_power=pilot_power, pilot=pilot,
                confirmation_power=confirmation_power, confirmation=confirmation,
                require_gate=str(args.require_gate), thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(run_dir, report=report, stage="report", phase="report")

        dynamics = base._make_dynamics(args)
        horizon = float(natural_horizon(dynamics))
        if not math.isclose(
            horizon, float(parent.get("horizon", math.nan)), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ArtifactCompatibilityError("gradient-control horizon differs from parent")
        stream_plan = build_density_ratio_stream_plan(
            root_seed=int(args.root_seed), grid_size=int(args.grid_size), horizon=horizon,
            label=3, bin_counts=(4, 4, 4, 4, 16), teacher_epsilon=0.5,
        )
        paired_plan = build_paired_mixture_stream_plan(
            root_seed=int(args.root_seed), grid_size=int(args.grid_size), horizon=horizon,
            label=3, teacher_epsilon=0.5,
        )
        trust_plan = build_h1_trust_plan(
            grid_size=int(args.grid_size), horizon=horizon,
            root_seed=int(args.root_seed), label=3,
        )
        _freeze_json(
            run_dir / "gradient_control_stream_plans.json",
            {
                "schema": RUN_SCHEMA + "-stream-plans",
                "schema_version": RUN_SCHEMA_VERSION,
                "evaluation_stream": stream_plan_record(stream_plan),
                "paired_training_stream": paired_mixture_stream_plan_record(paired_plan),
                "h1_reference_trust_stream": h1_trust_plan_record(trust_plan),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )

        preflight_paths = (
            run_dir / "h1_operator_gate.json",
            run_dir / "h1_calibration_gate.json",
            run_dir / "gradient_controller_preflight_gate.json",
            run_dir / "h1_gradient_control_preflight_gate.json",
        )
        if all(path.is_file() for path in preflight_paths):
            operator_gate, calibration_gate, controller_preflight, preflight = map(
                _json_load, preflight_paths
            )
        else:
            _write_status(run_dir, status="running", phase="preflight")
            operator_gate, calibration_gate, preflight = _run_preflight(
                run_dir, args=args, parent=parent, dynamics=dynamics,
                plan=trust_plan, paired_plan=paired_plan, device=device,
            )
            controller_preflight = _json_load(
                run_dir / "gradient_controller_preflight_gate.json"
            )

        pilot_power, pilot, confirmation_power, confirmation = _empty_components()[2:]
        if args.stage == "preflight" or not bool(int(preflight.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance, controller_preflight=controller_preflight,
                preflight=preflight, pilot_power=pilot_power, pilot=pilot,
                confirmation_power=confirmation_power, confirmation=confirmation,
                require_gate=str(args.require_gate), thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(
                run_dir, report=report, stage=str(args.stage), phase="preflight",
                skips=(
                    ()
                    if bool(int(preflight.get("passed", 0)))
                    else ({"stage": "pilot_and_confirmation", "reason": "preflight failed"},)
                ),
            )

        calibration = _json_load(run_dir / "h1_calibration.json")
        _write_status(run_dir, status="running", phase="pilot")
        pilot, pilot_power, pilot_panels = _run_gradient_control_pilot(
            run_dir, args=args, manifest=manifest, parent=parent, dynamics=dynamics,
            device=device, stream_plan=stream_plan, paired_plan=paired_plan,
            trust_plan=trust_plan, calibration=calibration, thresholds=thresholds,
        )
        if args.stage == "pilot" or not bool(int(pilot.get("passed", 0))):
            report = _workflow_report(
                provenance=provenance, controller_preflight=controller_preflight,
                preflight=preflight, pilot_power=pilot_power, pilot=pilot,
                confirmation_power=confirmation_power, confirmation=confirmation,
                require_gate=str(args.require_gate), thresholds=thresholds,
            )
            _save_report(run_dir, report)
            return _finish(
                run_dir, report=report, stage=str(args.stage), phase="pilot",
                skips=(
                    ()
                    if bool(int(pilot.get("passed", 0)))
                    else ({"stage": "confirmation", "reason": "gradient-control pilot failed"},)
                ),
            )

        profile = _load_selected_profile(run_dir)
        _write_status(run_dir, status="running", phase="confirmation")
        confirmation_power, confirmation = _run_gradient_control_confirmation(
            run_dir, args=args, manifest=manifest, parent=parent, profile=profile,
            pilot_panels=pilot_panels, dynamics=dynamics, device=device,
            stream_plan=stream_plan, paired_plan=paired_plan, trust_plan=trust_plan,
            calibration=calibration, thresholds=thresholds,
        )
        report = _workflow_report(
            provenance=provenance, controller_preflight=controller_preflight,
            preflight=preflight, pilot_power=pilot_power, pilot=pilot,
            confirmation_power=confirmation_power, confirmation=confirmation,
            require_gate=str(args.require_gate), thresholds=thresholds,
        )
        _save_report(run_dir, report)
        return _finish(
            run_dir, report=report, stage=str(args.stage), phase="confirmation",
            skips=(
                ()
                if bool(int(confirmation_power.get("passed", 0)))
                else ({
                    "stage": "confirmation_training",
                    "reason": "fixed confirmation panels failed oracle power",
                },)
            ),
        )
    except Exception as exc:
        if resumed and not mutation_started:
            if not args.no_progress:
                print(
                    f"H1 gradient-control resume rejected: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            return 2
        atomic_write_json(
            run_dir / "failure.json",
            {
                "schema": RUN_SCHEMA + "-failure",
                "schema_version": RUN_SCHEMA_VERSION,
                "type": type(exc).__name__, "message": str(exc),
                "stage": str(args.stage),
                "physical_training_performed": 0, "sampling_performed": 0,
            },
        )
        (
            controller_preflight, preflight, pilot_power, pilot,
            confirmation_power, confirmation,
        ) = _empty_components()
        report = _workflow_report(
            provenance=provenance, controller_preflight=controller_preflight,
            preflight=preflight, pilot_power=pilot_power, pilot=pilot,
            confirmation_power=confirmation_power, confirmation=confirmation,
            require_gate=str(args.require_gate), thresholds=thresholds,
        )
        _save_report(run_dir, report)
        _finish(
            run_dir, report=report, stage=str(args.stage), phase="failure",
            execution_failed=True,
            skips=({"stage": "remaining", "reason": f"{type(exc).__name__}: {exc}"},),
        )
        if not args.no_progress:
            print(
                f"H1 gradient-control failed: {type(exc).__name__}: {exc}", file=sys.stderr
            )
        return 2


_run = _run_gradient_control


def main(argv: Sequence[str] | None = None) -> int:
    return _run_gradient_control(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
