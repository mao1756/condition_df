"""Restartable gradient-ratio controlled H1 tasks for D0 density-ratio controls.

This module is deliberately additive.  It reuses the normalized-head paired
BCE task semantics while controlling the norm of the moving
``raw - stopgrad(EMA)`` H1 gradient at every optimizer update.  It never
imports or invokes a reverse sampler.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

import mnist.diag_d0_score_density_ratio_controls as base
import mnist.diag_d0_score_density_ratio_head_confirmation as head
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    atomic_copy_file,
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    capture_rng_state,
    config_fingerprint,
    file_fingerprint,
    restore_rng_state,
)
from mnist.d0_score_density_ratio import (
    DensityRatioPanel,
    DensityRatioStreamPlan,
    panel_identity,
    stream_plan_record,
)
from mnist.d0_score_density_ratio_h1_trust import (
    H1_TRUST_CALIBRATION_VERSION,
    H1_TRUST_OPERATOR_VERSION,
    H1_TRUST_SCHEMA,
    H1_TRUST_SCHEMA_VERSION,
    H1TrustCalibration,
    H1TrustPlan,
    generate_reference_trust_batch,
    h1_increment_components,
    h1_trust_plan_record,
)
from mnist.d0_score_density_ratio_h1_gradient_control import (
    H1_GRADIENT_CONTROL_VERSION,
    GradientRatioControllerConfig,
    assign_controlled_gradients,
    compose_gradient_ratio_update,
)
from mnist.d0_score_density_ratio_head import (
    COORDINATE_CONJUGATE_ADAMW_VERSION,
    NORMALIZED_HEAD_COORDINATE_VERSION,
    NORMALIZED_HEAD_MODEL_VERSION,
    D0BoundarySmoothMeanHeadPotentialUNet,
    build_coordinate_conjugate_adamw,
    coordinate_conjugate_adamw_record,
    normalized_gradient_diagnostics,
)
from mnist.d0_score_density_ratio_head_gate import (
    HeadCoordinateThresholds,
    evaluate_null_seed,
    evaluate_teacher_seed,
)
from mnist.d0_score_density_ratio_gate import confirm_nominee_on_b
from mnist.d0_score_density_ratio_head_provenance import PARENT_LOSS_SCALE
from mnist.d0_score_density_ratio_paired import (
    PAIRED_MIXTURE_ACCUMULATION_VERSION,
    PAIRED_MIXTURE_OBJECTIVE_VERSION,
    PAIRED_MIXTURE_SCHEMA,
    PAIRED_MIXTURE_STREAM_VERSION,
    PairedMixtureStreamPlan,
    generate_paired_mixture_microbatch,
    paired_mixture_stream_plan_record,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    init_ema_state,
    update_ema_state,
)


H1_TASK_SCHEMA = "experiment12-d0-score-density-ratio-h1-gradient-control-task"
H1_TASK_SCHEMA_VERSION = 2
H1_TASK_CHECKPOINT_SCHEMA_VERSION = 2
H1_TASK_TRAINING_VERSION = "d0-paired-bce-stopped-ema-h1-gradient-control-v1"

__all__ = [
    "H1_TASK_SCHEMA",
    "H1_TASK_SCHEMA_VERSION",
    "H1_TASK_CHECKPOINT_SCHEMA_VERSION",
    "H1_TASK_TRAINING_VERSION",
    "gradient_control_task_fingerprints",
    "write_failed_gradient_control_task_result",
    "run_gradient_control_paired_density_ratio_task",
]


def _json_load(path: str | Path) -> dict[str, Any]:
    return base._json_load(path)


def _not_evaluated_gate(name: str, reason: str) -> dict[str, Any]:
    return {
        "gate": str(name),
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "reason": str(reason),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _calibration_record(
    value: H1TrustCalibration | Mapping[str, Any],
) -> dict[str, Any]:
    record = value.to_record() if isinstance(value, H1TrustCalibration) else dict(value)
    if (
        record.get("schema") != H1_TRUST_SCHEMA + "-calibration"
        or int(record.get("schema_version", -1)) != H1_TRUST_SCHEMA_VERSION
        or record.get("calibration_version") != H1_TRUST_CALIBRATION_VERSION
        or int(record.get("passed", 0)) != 1
    ):
        raise ArtifactCompatibilityError("H1 trust calibration is not passing v1 evidence")
    for name in ("value_scale", "energy_scale", "lambda_base"):
        number = float(record.get(name, math.nan))
        if not math.isfinite(number) or number <= 0.0:
            raise ArtifactCompatibilityError(f"H1 trust calibration {name} is invalid")
    return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))


def gradient_control_task_fingerprints(
    base_fingerprints: Mapping[str, Any],
    *,
    trust_plan: H1TrustPlan,
    calibration: H1TrustCalibration | Mapping[str, Any],
    target_ratio: float,
    controller_config: GradientRatioControllerConfig,
) -> dict[str, Any]:
    """Bind one task to the immutable H1 geometry and online controller."""

    ratio = float(target_ratio)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("h1_ratio must be finite and nonnegative")
    calibration_record = _calibration_record(calibration)
    plan_record = h1_trust_plan_record(trust_plan)
    additions = {
        "h1_task_training_version": H1_TASK_TRAINING_VERSION,
        "h1_checkpoint_schema_version": H1_TASK_CHECKPOINT_SCHEMA_VERSION,
        "h1_operator_version": H1_TRUST_OPERATOR_VERSION,
        "h1_trust_plan_fingerprint": trust_plan.fingerprint,
        "h1_trust_plan_record_fingerprint": config_fingerprint(plan_record),
        "h1_calibration_fingerprint": config_fingerprint(calibration_record),
        "h1_target_gradient_ratio": ratio,
        "h1_controller_version": H1_GRADIENT_CONTROL_VERSION,
        "h1_controller_config": controller_config.to_record(),
        "h1_controller_config_fingerprint": controller_config.fingerprint,
        "h1_legacy_lambda_base_advisory": float(calibration_record["lambda_base"]),
        "h1_legacy_lambda_used_for_optimization": 0,
        "h1_value_scale": float(calibration_record["value_scale"]),
        "h1_energy_scale": float(calibration_record["energy_scale"]),
        "h1_reference_banks_per_update": int(trust_plan.banks_per_update),
        "h1_reference_states_per_bank": int(trust_plan.states_per_bank),
    }
    result = dict(base_fingerprints)
    for key, expected in additions.items():
        if key in result and result[key] != expected:
            raise ArtifactCompatibilityError(f"task fingerprint conflicts at {key}")
        result[key] = expected
    return result


def _optimizer_group_record(
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, Any]]:
    return [
        {
            "name": str(group.get("name", "")),
            "lr": float(group["lr"]),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "betas": [float(value) for value in group["betas"]],
            "amsgrad": int(bool(group["amsgrad"])),
            "parameter_count": len(group["params"]),
        }
        for group in optimizer.param_groups
    ]


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
    trust_plan: H1TrustPlan,
    calibration_record: Mapping[str, Any],
    h1_ratio: float,
    task: str,
    fingerprints: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    atomic_torch_save(
        path,
        {
            "schema": H1_TASK_SCHEMA + "-checkpoint",
            "schema_version": H1_TASK_CHECKPOINT_SCHEMA_VERSION,
            "training_version": H1_TASK_TRAINING_VERSION,
            "model_schema": NORMALIZED_HEAD_MODEL_VERSION,
            "head_coordinate_version": NORMALIZED_HEAD_COORDINATE_VERSION,
            "optimizer_coordinate_version": COORDINATE_CONJUGATE_ADAMW_VERSION,
            "optimizer_parameter_groups": _optimizer_group_record(optimizer),
            "task": str(task),
            "step": int(step),
            "stream_cursor": int(step),
            # Only a completed optimizer/EMA update is durable.  An interrupted
            # microbatch or trust bank is replayed from this committed cursor.
            "accumulation_cursor": 0,
            "trust_bank_cursor": 0,
            "paired_estimator_schema": PAIRED_MIXTURE_SCHEMA,
            "paired_objective_version": PAIRED_MIXTURE_OBJECTIVE_VERSION,
            "paired_stream_version": PAIRED_MIXTURE_STREAM_VERSION,
            "paired_accumulation_version": PAIRED_MIXTURE_ACCUMULATION_VERSION,
            "h1_operator_version": H1_TRUST_OPERATOR_VERSION,
            "stream_plan": stream_plan_record(stream_plan),
            "paired_stream_plan": paired_mixture_stream_plan_record(
                paired_stream_plan
            ),
            "h1_trust_plan": h1_trust_plan_record(trust_plan),
            "h1_calibration": copy.deepcopy(dict(calibration_record)),
            "h1_ratio": float(h1_ratio),
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
            "amp": False,
            "scaler_state_dict": None,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )


def _validate_optimizer_groups(
    payload: Mapping[str, Any], fingerprints: Mapping[str, Any]
) -> None:
    groups = [dict(value) for value in payload.get("optimizer_parameter_groups", [])]
    if len(groups) != 2 or [value.get("name") for value in groups] != [
        "body",
        "normalized_head",
    ]:
        raise ArtifactCompatibilityError("H1 checkpoint optimizer groups changed")
    body = groups[0]
    normalized = groups[1]
    factor = int(fingerprints.get("grid_cells", 0))
    if factor <= 0:
        body_lr = float(body.get("lr", math.nan))
        head_lr = float(normalized.get("lr", math.nan))
        factor = int(round(head_lr / body_lr)) if body_lr > 0.0 else -1
    expected = (
        (
            float(body.get("lr", math.nan)),
            float(body.get("eps", math.nan)),
            float(body.get("weight_decay", math.nan)),
        ),
        (
            factor * float(body.get("lr", math.nan)),
            float(body.get("eps", math.nan)) / factor,
            float(body.get("weight_decay", math.nan)) / factor,
        ),
    )
    for group, values in zip(groups, expected, strict=True):
        for key, target in zip(("lr", "eps", "weight_decay"), values, strict=True):
            if float(group.get(key, math.nan)) != float(target):
                raise ArtifactCompatibilityError(f"H1 optimizer group {key} changed")
        if tuple(float(item) for item in group.get("betas", ())) != (0.9, 0.999):
            raise ArtifactCompatibilityError("H1 optimizer betas changed")
        if int(group.get("amsgrad", -1)) != 0:
            raise ArtifactCompatibilityError("H1 optimizer AMSGrad changed")
    if int(body.get("parameter_count", -1)) <= 0 or int(
        normalized.get("parameter_count", -1)
    ) != 2:
        raise ArtifactCompatibilityError("H1 optimizer parameter partition changed")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ArtifactCompatibilityError("H1 optimizer state is invalid")
    state_groups = optimizer_state.get("param_groups")
    if not isinstance(state_groups, list) or len(state_groups) != len(groups):
        raise ArtifactCompatibilityError("H1 optimizer state groups changed")
    for state_group, recorded_group in zip(state_groups, groups, strict=True):
        if not isinstance(state_group, Mapping):
            raise ArtifactCompatibilityError("H1 optimizer state group is invalid")
        for key in ("name", "lr", "eps", "weight_decay", "betas", "amsgrad"):
            actual = state_group.get(key)
            expected_value = recorded_group.get(key)
            if key == "betas":
                actual = tuple(float(item) for item in (actual or ()))
                expected_value = tuple(float(item) for item in (expected_value or ()))
            elif key in {"lr", "eps", "weight_decay"}:
                actual = float(actual) if actual is not None else math.nan
                expected_value = (
                    float(expected_value)
                    if expected_value is not None
                    else math.nan
                )
            elif key == "amsgrad":
                actual = int(bool(actual))
                expected_value = int(bool(expected_value))
            if actual != expected_value:
                raise ArtifactCompatibilityError(
                    f"H1 optimizer state group {key} changed"
                )
        saved_parameters = state_group.get("params")
        if not isinstance(saved_parameters, list) or len(saved_parameters) != int(
            recorded_group["parameter_count"]
        ):
            raise ArtifactCompatibilityError(
                "H1 optimizer state parameter partition changed"
            )


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    task: str,
    fingerprints: Mapping[str, Any],
    trust_plan: H1TrustPlan,
    calibration_record: Mapping[str, Any],
    h1_ratio: float,
) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - old PyTorch
        value = torch.load(path, map_location=device)
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != H1_TASK_SCHEMA + "-checkpoint"
        or int(value.get("schema_version", -1))
        != H1_TASK_CHECKPOINT_SCHEMA_VERSION
        or value.get("training_version") != H1_TASK_TRAINING_VERSION
        or value.get("task") != task
        or dict(value.get("fingerprints", {})) != dict(fingerprints)
        or value.get("model_schema") != NORMALIZED_HEAD_MODEL_VERSION
        or value.get("head_coordinate_version") != NORMALIZED_HEAD_COORDINATE_VERSION
        or value.get("optimizer_coordinate_version")
        != COORDINATE_CONJUGATE_ADAMW_VERSION
        or value.get("h1_operator_version") != H1_TRUST_OPERATOR_VERSION
        or int(value.get("accumulation_cursor", -1)) != 0
        or int(value.get("trust_bank_cursor", -1)) != 0
        or dict(value.get("h1_trust_plan", {}))
        != h1_trust_plan_record(trust_plan)
        or dict(value.get("h1_calibration", {})) != dict(calibration_record)
        or float(value.get("h1_ratio", math.nan)) != float(h1_ratio)
    ):
        raise ArtifactCompatibilityError("legacy, foreign, or partial H1 checkpoint")
    required = {
        "step",
        "stream_cursor",
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "history",
        "validation_records",
        "checkpoint_selection",
        "rng_state",
        "stream_plan",
        "paired_stream_plan",
        "fingerprints",
        "optimizer_parameter_groups",
    }
    if not required.issubset(value):
        raise ArtifactCompatibilityError("H1 checkpoint is incomplete")
    _validate_optimizer_groups(value, fingerprints)
    if int(value["stream_cursor"]) != int(value["step"]):
        raise ArtifactCompatibilityError("H1 checkpoint cursor differs from step")
    return dict(value)


def _load_completed_task(
    task_dir: Path,
    *,
    device: torch.device,
    task: str,
    fingerprints: Mapping[str, Any],
    trust_plan: H1TrustPlan,
    calibration_record: Mapping[str, Any],
    h1_ratio: float,
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
    sealed_path = task_dir / "sealed_panel_b.json"
    if not latest_path.is_file() or not best_path.is_file() or not sealed_path.is_file():
        raise ArtifactCompatibilityError("completed H1 task lacks terminal pointers")
    latest, best = _json_load(latest_path), _json_load(best_path)
    files = {
        "latest": checkpoints / str(latest.get("filename", "")),
        "selected": checkpoints / str(best.get("selected_filename", "")),
        "nominee": checkpoints / str(best.get("nominee_filename", "")),
        "best_copy": checkpoints / str(best.get("best_ema_filename", "")),
        "nominee_copy": checkpoints / str(best.get("nominee_ema_filename", "")),
    }
    summary = dict(result.get("training_summary", {}))
    if (
        dict(result.get("fingerprints", {})) != dict(fingerprints)
        or dict(status.get("fingerprints", {})) != dict(fingerprints)
        or dict(latest.get("fingerprints", {})) != dict(fingerprints)
        or dict(best.get("fingerprints", {})) != dict(fingerprints)
        or status.get("task_result_sha256") != file_fingerprint(result_path)
        or status.get("sealed_panel_b_sha256") != file_fingerprint(sealed_path)
        or not all(path.is_file() for path in files.values())
        or latest.get("sha256") != file_fingerprint(files["latest"])
        or best.get("selected_sha256") != file_fingerprint(files["selected"])
        or best.get("nominee_sha256") != file_fingerprint(files["nominee"])
        or best.get("best_ema_sha256") != file_fingerprint(files["best_copy"])
        or best.get("nominee_ema_sha256")
        != file_fingerprint(files["nominee_copy"])
        or summary.get("selected_checkpoint_sha256")
        != file_fingerprint(files["best_copy"])
        or summary.get("nominee_checkpoint_sha256")
        != file_fingerprint(files["nominee_copy"])
        or summary.get("sealed_panel_b_sha256") != file_fingerprint(sealed_path)
    ):
        raise ArtifactCompatibilityError("completed H1 task hash chain mismatch")
    for name in ("latest", "selected", "nominee"):
        _load_checkpoint(
            files[name],
            device=device,
            task=task,
            fingerprints=fingerprints,
            trust_plan=trust_plan,
            calibration_record=calibration_record,
            h1_ratio=h1_ratio,
        )
    return result


def _copy_gradients(parameters: Sequence[nn.Parameter]) -> list[Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _gradient_geometry(
    left: Sequence[Tensor | None], right: Sequence[Tensor | None]
) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("gradient vectors differ in length")
    device = next(
        (value.device for value in (*left, *right) if value is not None),
        torch.device("cpu"),
    )
    left_sq = torch.zeros((), device=device)
    right_sq = torch.zeros((), device=device)
    dot = torch.zeros((), device=device)
    for lhs, rhs in zip(left, right, strict=True):
        if lhs is not None:
            left_sq = left_sq + lhs.square().sum()
        if rhs is not None:
            right_sq = right_sq + rhs.square().sum()
        if lhs is not None and rhs is not None:
            dot = dot + (lhs * rhs).sum()
    left_norm = torch.sqrt(left_sq)
    right_norm = torch.sqrt(right_sq)
    denominator = left_norm * right_norm
    cosine = torch.where(
        denominator > 0.0, dot / denominator, torch.zeros_like(denominator)
    )
    return {
        "left_norm": float(left_norm.detach().cpu()),
        "right_norm": float(right_norm.detach().cpu()),
        "dot": float(dot.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
    }


def _assign_sum_gradients(
    parameters: Sequence[nn.Parameter],
    first: Sequence[Tensor | None],
    second: Sequence[Tensor | None],
) -> None:
    for parameter, lhs, rhs in zip(parameters, first, second, strict=True):
        if lhs is None and rhs is None:
            parameter.grad = None
        elif lhs is None:
            parameter.grad = rhs.detach().clone()
        elif rhs is None:
            parameter.grad = lhs.detach().clone()
        else:
            parameter.grad = lhs + rhs


def _sync_ema_model(ema_model: nn.Module, ema_state: Mapping[str, Tensor]) -> None:
    ema_model.load_state_dict(ema_state, strict=True)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)


def _mean_records(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    names = sorted(
        name
        for name in set.intersection(*(set(record) for record in records))
        if all(
            isinstance(record[name], (int, float, np.integer, np.floating))
            and math.isfinite(float(record[name]))
            for record in records
        )
    )
    return {
        name: float(np.mean([float(record[name]) for record in records]))
        for name in names
    }


def _failed_task_result(
    task_dir: Path,
    *,
    task: str,
    model_seed: int,
    fingerprints: Mapping[str, Any],
    h1_ratio: float,
    exc: BaseException,
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    failure = {"type": type(exc).__name__, "message": str(exc)}
    result = {
        "schema": H1_TASK_SCHEMA + "-result",
        "schema_version": H1_TASK_SCHEMA_VERSION,
        "task": task,
        "model_seed": int(model_seed),
        "h1_ratio": float(h1_ratio),
        "metrics": {
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
        },
        "gate": _not_evaluated_gate("h1_density_ratio_task", str(exc)),
        "failure": failure,
        "fingerprints": dict(fingerprints),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(task_dir / "task_failure.json", result)
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "schema": H1_TASK_SCHEMA + "-status",
            "schema_version": H1_TASK_SCHEMA_VERSION,
            "status": "failed",
            "task": task,
            "model_seed": int(model_seed),
            "h1_ratio": float(h1_ratio),
            "failure": failure,
            "fingerprints": dict(fingerprints),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return result


def write_failed_gradient_control_task_result(
    task_dir: Path,
    *,
    task: str,
    model_seed: int,
    fingerprints: Mapping[str, Any],
    h1_ratio: float,
    exc: BaseException,
) -> dict[str, Any]:
    """Commit readable failure evidence for a task-level exception."""

    return _failed_task_result(
        task_dir,
        task=task,
        model_seed=model_seed,
        fingerprints=fingerprints,
        h1_ratio=h1_ratio,
        exc=exc,
    )


def run_gradient_control_paired_density_ratio_task(
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
    trust_plan: H1TrustPlan,
    calibration: H1TrustCalibration | Mapping[str, Any],
    h1_ratio: float,
    controller_config: GradientRatioControllerConfig,
    fingerprints: Mapping[str, Any],
    phase: str,
    thresholds: HeadCoordinateThresholds,
    show_progress: bool,
    interrupt_after_checkpoint_step: int | None = None,
    interrupt_during_accumulation: tuple[int, int] | None = None,
    interrupt_during_trust_bank: tuple[int, int] | None = None,
    interrupt_after_sealed_panel_b: bool = False,
    defer_panel_b: bool = False,
) -> dict[str, Any]:
    """Train one normalized-head paired-BCE task with EMA-proximal H1 trust.

    Checkpoints are written only after a complete BCE accumulation, both trust
    banks, one globally clipped AdamW update, and one EMA update.  All streams
    are stateless, so an uncommitted step is replayed byte-for-byte on resume.

    With ``defer_panel_b=True`` the task commits complete panel-A training
    evidence with status ``awaiting_panel_b`` and never reads panel B.  Calling
    this function again on the same directory with ``defer_panel_b=False``
    resumes from the frozen endpoint, opens B once, and finalizes the task.
    This is the pilot API used to keep nonselected teacher arms sealed.
    """

    if task not in {"bounded_teacher", "dirichlet_null"}:
        raise ValueError("unsupported H1 density-ratio task")
    if set(selection_panels) != {"a", "b"}:
        raise ValueError("selection panels must be exactly a and b")
    if audit_panels is not None and set(audit_panels) != {"c", "d"}:
        raise ValueError("audit panels must be exactly c and d")
    if bool(defer_panel_b) and audit_panels is not None:
        raise ValueError("panel B cannot be deferred when audit panels are supplied")
    if int(accumulation_level) != 8:
        raise ValueError("H1 production v1 freezes accumulation at 8")
    ratio = float(h1_ratio)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("h1_ratio must be finite and nonnegative")
    calibration_record = _calibration_record(calibration)
    bound_fingerprints = gradient_control_task_fingerprints(
        fingerprints,
        trust_plan=trust_plan,
        calibration=calibration_record,
        target_ratio=ratio,
        controller_config=controller_config,
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    completed_result = _load_completed_task(
        task_dir,
        device=device,
        task=task,
        fingerprints=bound_fingerprints,
        trust_plan=trust_plan,
        calibration_record=calibration_record,
        h1_ratio=ratio,
    )
    if completed_result is not None:
        return completed_result

    checkpoints = task_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints / "latest.json"
    best_pointer_path = checkpoints / "best.json"
    rng = base._set_seed(int(model_seed))
    model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    parameters = list(model.parameters())
    optimizer = build_coordinate_conjugate_adamw(
        model,
        body_lr=float(learning_rate),
        eps=1e-8,
        weight_decay=float(args.weight_decay),
    )
    ema_state = init_ema_state(model)
    ema_model = D0BoundarySmoothMeanHeadPotentialUNet(
        dynamics, base_channels=int(args.base_channels)
    ).to(device)
    _sync_ema_model(ema_model, ema_state)
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    completed = 0

    def load_payload(path: Path) -> dict[str, Any]:
        return _load_checkpoint(
            path,
            device=device,
            task=task,
            fingerprints=bound_fingerprints,
            trust_plan=trust_plan,
            calibration_record=calibration_record,
            h1_ratio=ratio,
        )

    if latest_path.is_file():
        latest = _json_load(latest_path)
        filename = str(latest.get("filename", ""))
        checkpoint_path = checkpoints / filename
        if (
            Path(filename).name != filename
            or not checkpoint_path.is_file()
            or latest.get("sha256") != file_fingerprint(checkpoint_path)
            or dict(latest.get("fingerprints", {})) != bound_fingerprints
        ):
            raise ArtifactCompatibilityError("H1 latest pointer is invalid")
        payload = load_payload(checkpoint_path)
        orphan = checkpoints / f"finalized-step-{int(payload['step']):08d}.pt"
        if checkpoint_path != orphan and orphan.is_file():
            recovered = load_payload(orphan)
            b_records = sum(
                "b" in dict(value.get("panels", {}))
                for value in recovered.get("validation_records", [])
            )
            if (
                int(recovered.get("step", -1)) == int(payload["step"])
                and b_records == 1
                and dict(recovered.get("checkpoint_selection", {})).get("gate")
                in {
                    "density_ratio_checkpoint_selection",
                    "density_ratio_fixed_endpoint_selection",
                }
            ):
                payload = recovered
                checkpoint_path = orphan
                atomic_write_json(
                    latest_path,
                    {
                        "schema": H1_TASK_SCHEMA + "-latest",
                        "schema_version": H1_TASK_SCHEMA_VERSION,
                        "filename": orphan.name,
                        "sha256": file_fingerprint(orphan),
                        "step": int(payload["step"]),
                        "stream_cursor": int(payload["step"]),
                        "accumulation_cursor": 0,
                        "trust_bank_cursor": 0,
                        "fingerprints": bound_fingerprints,
                        "recovered_orphan_finalization": 1,
                        "physical_training_performed": 0,
                        "sampling_performed": 0,
                    },
                )
        if dict(payload["stream_plan"]) != stream_plan_record(stream_plan):
            raise ArtifactCompatibilityError("H1 evaluation stream mismatch")
        if dict(payload["paired_stream_plan"]) != paired_mixture_stream_plan_record(
            paired_stream_plan
        ):
            raise ArtifactCompatibilityError("H1 paired stream mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        base._optimizer_to_device(optimizer, device)
        ema_state = {
            key: value.detach().clone().to(device)
            for key, value in dict(payload["ema_state_dict"]).items()
        }
        _sync_ema_model(ema_model, ema_state)
        history = [dict(value) for value in payload["history"]]
        validations = [dict(value) for value in payload["validation_records"]]
        selection = dict(payload["checkpoint_selection"])
        completed = int(payload["step"])
        restore_rng_state(payload["rng_state"], rng)

    def validate(step: int) -> dict[str, Any]:
        raw_state = copy.deepcopy(model.state_dict())
        try:
            model.load_state_dict(ema_state, strict=True)
            model.eval()
            record, _ = base._classification_panel_record(
                model,
                selection_panels["a"],
                dynamics=dynamics,
                args=args,
                device=device,
                bootstrap_seed=base._derived_seed(
                    int(args.root_seed), phase, task, "selection", "a", step
                ),
                include_analytic_teacher=(
                    task == "bounded_teacher" and int(step) == int(args.train_steps)
                ),
            )
        finally:
            model.load_state_dict(raw_state, strict=True)
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
            validations, thresholds.stability.density_ratio
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
            trust_plan=trust_plan,
            calibration_record=calibration_record,
            h1_ratio=ratio,
            task=task,
            fingerprints=bound_fingerprints,
            rng=rng,
        )
        atomic_write_json(
            latest_path,
            {
                "schema": H1_TASK_SCHEMA + "-latest",
                "schema_version": H1_TASK_SCHEMA_VERSION,
                "filename": checkpoint_path.name,
                "sha256": file_fingerprint(checkpoint_path),
                "step": int(step),
                "stream_cursor": int(step),
                "accumulation_cursor": 0,
                "trust_bank_cursor": 0,
                "fingerprints": bound_fingerprints,
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
    legacy_multiplier = float(calibration_record["lambda_base"]) * ratio
    value_scale = float(calibration_record["value_scale"])
    energy_scale = float(calibration_record["energy_scale"])

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
            objective, component_record = head._microbatch_objective_and_components(
                model, batch, task=task
            )
            if not bool(torch.isfinite(objective.detach())):
                raise FloatingPointError(
                    f"nonfinite paired BCE for {task} at step {step}"
                )
            microbatch_losses.append(float(objective.detach().cpu()))
            microbatch_fingerprints.append(head._batch_fingerprint(batch))
            microbatch_components.append(component_record)
            (
                objective
                * float(PARENT_LOSS_SCALE)
                / float(accumulation_level)
            ).backward()
            if interrupt_during_accumulation == (step, microbatch_index):
                raise RuntimeError("injected interruption during H1 BCE accumulation")

        bce_gradients = _copy_gradients(parameters)
        optimizer.zero_grad(set_to_none=True)
        _sync_ema_model(ema_model, ema_state)
        trust_records: list[dict[str, float]] = []
        trust_fingerprints: list[str] = []
        trust_objective_values: list[float] = []
        for bank_index in range(int(trust_plan.banks_per_update)):
            trust_batch = generate_reference_trust_batch(
                trust_plan,
                phase=phase,
                task=task,
                optimizer_step=int(step),
                bank=int(bank_index),
                device=device,
                dtype=torch.float32,
            )
            components = h1_increment_components(
                model,
                ema_model,
                trust_batch.tau,
                trust_batch.states,
                trust_batch.labels,
                dynamics,
                value_scale=value_scale,
                energy_scale=energy_scale,
                create_graph=bool(ratio > 0.0),
            )
            trust_records.append(components.detached_record())
            trust_fingerprints.append(str(trust_batch.fingerprint))
            trust_objective_values.append(float(components.objective.detach().cpu()))
            if ratio > 0.0:
                # Backward bank-by-bank to avoid retaining two full
                # second-order U-Net graphs.  Division here is exactly the
                # gradient of the prescribed two-bank arithmetic mean.
                (
                    components.objective / float(trust_plan.banks_per_update)
                ).backward()
            if interrupt_during_trust_bank == (step, bank_index):
                raise RuntimeError("injected interruption during H1 trust banks")
        if ratio > 0.0:
            trust_objective = torch.as_tensor(
                float(np.mean(trust_objective_values)), device=device
            )
            h1_gradients = _copy_gradients(parameters)
        else:
            trust_objective = torch.as_tensor(
                float(np.mean(trust_objective_values)),
                device=device,
            )
            h1_gradients = [None for _ in parameters]
        geometry = _gradient_geometry(bce_gradients, h1_gradients)
        controlled = compose_gradient_ratio_update(
            bce_gradients,
            h1_gradients,
            target_ratio=ratio,
            optimizer_step=step,
            config=controller_config,
        )
        controller_record = controlled.detached_record()
        assign_controlled_gradients(parameters, controlled)
        trust_loss = torch.as_tensor(
            float(controller_record["h1_coefficient"])
            * float(trust_objective.detach().cpu()),
            device=device,
        )
        if not math.isfinite(float(trust_loss.detach().cpu())):
            raise FloatingPointError(
                f"nonfinite gradient-controlled H1 loss for {task} at step {step}"
            )

        coordinate_gradients = normalized_gradient_diagnostics(model)
        scaled_preclip = float(coordinate_gradients["normalized_gradient_norm"])
        reconstructed_legacy_preclip = float(
            coordinate_gradients["reconstructed_legacy_gradient_norm"]
        )
        if not math.isfinite(scaled_preclip):
            raise FloatingPointError(
                f"nonfinite H1 accumulated gradient for {task} at step {step}"
            )
        bce_scaled_norm = float(geometry["left_norm"])
        h1_raw_norm = float(geometry["right_norm"])
        h1_scaled_norm = float(
            controller_record["h1_contribution_gradient_norm"]
        )
        raw_bce_norm = bce_scaled_norm / float(PARENT_LOSS_SCALE)
        clipped = int(scaled_preclip > float(args.grad_clip))
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(args.grad_clip), error_if_nonfinite=True
        )
        if not bool(torch.isfinite(clipped_norm)):
            raise FloatingPointError(
                f"nonfinite clipped H1 gradient for {task} at step {step}"
            )
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.step()
        update_sq = torch.zeros((), device=device)
        for old, parameter in zip(before, parameters, strict=True):
            update_sq = update_sq + (parameter.detach() - old).square().sum()
        if not bool(
            torch.isfinite(update_sq)
            and all(torch.isfinite(parameter.detach()).all() for parameter in parameters)
        ):
            raise FloatingPointError(
                f"nonfinite H1 optimizer update for {task} at step {step}"
            )
        ema_update_sq = torch.zeros((), device=device)
        for name, parameter in model.named_parameters():
            if name in ema_state:
                delta = (1.0 - float(args.ema_decay)) * (
                    parameter.detach() - ema_state[name]
                )
                ema_update_sq = ema_update_sq + delta.square().sum()
        update_ema_state(ema_state, model, float(args.ema_decay))
        _sync_ema_model(ema_model, ema_state)
        if not bool(torch.isfinite(ema_update_sq)) or not all(
            torch.isfinite(value).all() for value in ema_state.values()
        ):
            raise FloatingPointError(f"nonfinite H1 EMA state for {task} at step {step}")

        trust_mean = _mean_records(trust_records)
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
                        "base_positive",
                        "mixture_positive",
                        "reference_negative",
                        "total",
                    )
                },
                "h1_ratio": ratio,
                "h1_multiplier": ratio,
                "h1_lambda_base": float(calibration_record["lambda_base"]),
                "h1_effective_multiplier": float(
                    controller_record["h1_coefficient"]
                ),
                "h1_legacy_effective_multiplier_advisory": legacy_multiplier,
                "h1_legacy_multiplier_used_for_optimization": 0,
                "h1_value_scale": value_scale,
                "h1_energy_scale": energy_scale,
                "h1_reference_bank_fingerprints": trust_fingerprints,
                "h1_reference_banks": len(trust_records),
                "h1_value_mse": trust_mean.get("value_mse", math.nan),
                "h1_value_rms": trust_mean.get("value_rms", math.nan),
                "h1_natural_energy": trust_mean.get("natural_energy", math.nan),
                "h1_natural_energy_rms": trust_mean.get(
                    "natural_energy_rms", math.nan
                ),
                "h1_physical_flux_energy": trust_mean.get(
                    "physical_flux_energy", math.nan
                ),
                "h1_physical_flux_step_rms": trust_mean.get(
                    "physical_flux_step_rms", math.nan
                ),
                "h1_normalized_value": trust_mean.get(
                    "normalized_value", math.nan
                ),
                "h1_normalized_natural_energy": trust_mean.get(
                    "normalized_natural_energy", math.nan
                ),
                "h1_normalized_objective": float(trust_objective.detach().cpu()),
                "h1_scaled_optimizer_loss": float(trust_loss.detach().cpu()),
                # Stable flat aliases used by orchestration CSVs.  The more
                # descriptive fields above remain the canonical diagnostics.
                "h1_value": float(trust_objective.detach().cpu()),
                "h1_effective_loss": float(trust_loss.detach().cpu()),
                "bce_scaled_gradient_norm": bce_scaled_norm,
                "h1_scaled_gradient_norm": h1_scaled_norm,
                "bce_gradient_norm": bce_scaled_norm,
                "h1_gradient_norm": h1_scaled_norm,
                "h1_raw_gradient_norm": h1_raw_norm,
                "bce_h1_gradient_dot": float(geometry["dot"]),
                "bce_h1_gradient_cosine": float(geometry["cosine"]),
                **{
                    (name if name.startswith("controller_") else f"controller_{name}"): value
                    for name, value in controller_record.items()
                    if name
                    not in {
                        "physical_training_performed",
                        "sampling_performed",
                    }
                },
                "raw_accumulated_gradient_norm": raw_bce_norm,
                "raw_gradient_norm": raw_bce_norm,
                "scaled_preclip_gradient_norm": scaled_preclip,
                "grad_norm": scaled_preclip,
                "normalized_coordinate_gradient_norm": scaled_preclip,
                "normalized_body_gradient_norm": coordinate_gradients[
                    "body_gradient_norm"
                ],
                "normalized_head_gradient_norm": coordinate_gradients[
                    "normalized_head_gradient_norm"
                ],
                "normalized_head_squared_fraction": coordinate_gradients[
                    "normalized_head_squared_fraction"
                ],
                "reconstructed_legacy_preclip_gradient_norm": (
                    reconstructed_legacy_preclip
                ),
                "reconstructed_legacy_head_squared_fraction": (
                    coordinate_gradients[
                        "reconstructed_legacy_head_squared_fraction"
                    ]
                ),
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
                raise RuntimeError(f"injected interruption after H1 checkpoint {step}")
        if show_progress and (step % 50 == 0 or step == total_steps):
            elapsed = time.perf_counter() - started
            done = max(1, step - completed)
            eta = elapsed / done * max(0, total_steps - step)
            print(
                f"{phase}/{task}/h1-{ratio:g}: step {step}/{total_steps} "
                f"bce={history[-1]['scaled_loss']:.6g} "
                f"h1={history[-1]['h1_scaled_optimizer_loss']:.6g} "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    final_step = int(history[-1]["step"]) if history else completed
    if not latest_path.is_file() or int(_json_load(latest_path).get("step", -1)) != final_step:
        if not any(int(value.get("step", -1)) == final_step for value in validations):
            validations.append(validate(final_step))
        publish(final_step)

    nominee_step = int(total_steps)
    terminal_rows = [
        value for value in validations if int(value.get("step", -1)) == nominee_step
    ]
    if len(terminal_rows) != 1:
        raise ArtifactCompatibilityError("fixed endpoint validation is missing or ambiguous")
    terminal_a = dict(dict(terminal_rows[0].get("panels", {})).get("a", {}))
    nomination = {
        "gate": "density_ratio_fixed_endpoint_nomination",
        "evaluation_status": "evaluated",
        "passed": int(bool(int(terminal_rows[0].get("finite", 0)))),
        "nominee_step": nominee_step,
        "candidate_count": 1,
        "endpoint_policy": "ema-step-4000" if nominee_step == 4000 else f"ema-step-{nominee_step}",
        "panel_a_lower_bounds": [
            dict(terminal_a.get("overall", {})).get("lower_bound"),
            dict(terminal_a.get("data_end", {})).get("lower_bound"),
        ],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    nominee_path = checkpoints / f"step-{nominee_step:08d}.pt"
    if not nominee_path.is_file():
        raise ArtifactCompatibilityError("H1 panel-A nominee checkpoint is missing")
    nominee_rows = [
        value for value in validations if int(value.get("step", -1)) == nominee_step
    ]
    if len(nominee_rows) != 1:
        raise ArtifactCompatibilityError("H1 panel-A nominee is ambiguous")
    nominee_validation = nominee_rows[0]
    sealed_b_path = task_dir / "sealed_panel_b.json"
    sealed_binding = {
        "schema": H1_TASK_SCHEMA + "-sealed-panel-b",
        "schema_version": H1_TASK_SCHEMA_VERSION,
        "task": task,
        "phase": phase,
        "nominee_step": nominee_step,
        "nominee_checkpoint_sha256": file_fingerprint(nominee_path),
        "panel_identity": panel_identity(selection_panels["b"]),
        "fingerprints": bound_fingerprints,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    panel_b_deferred = bool(defer_panel_b)
    if panel_b_deferred and sealed_b_path.is_file():
        raise ArtifactCompatibilityError(
            "panel B is already sealed; resume with defer_panel_b=False"
        )
    if panel_b_deferred:
        confirmation = _not_evaluated_gate(
            "density_ratio_panel_b_confirmation",
            "panel B is deferred until the pilot selects this task",
        )
        confirmation.update(
            {
                "nominee_step": nominee_step,
                "accepted": 0,
                "selected_step": 0,
                "panel_b_lower_bounds": [],
                "panel_b_overall_bce": None,
                "panel_b_confidence": None,
            }
        )
    elif sealed_b_path.is_file():
        sealed = _json_load(sealed_b_path)
        for key, expected in sealed_binding.items():
            if sealed.get(key) != expected:
                raise ArtifactCompatibilityError(
                    f"H1 sealed panel-B binding mismatch for {key}"
                )
        panel_b = sealed.get("panel_record")
        if not isinstance(panel_b, Mapping):
            raise ArtifactCompatibilityError("H1 sealed panel-B record is missing")
        if "b" in dict(nominee_validation.get("panels", {})):
            if dict(nominee_validation["panels"]["b"]) != dict(panel_b):
                raise ArtifactCompatibilityError(
                    "H1 finalized checkpoint and sealed panel B disagree"
                )
        else:
            nominee_validation.setdefault("panels", {})["b"] = dict(panel_b)
            nominee_validation["finite"] = int(
                bool(int(nominee_validation.get("finite", 0)))
                and bool(int(dict(panel_b).get("finite", 0)))
            )
    elif "b" not in dict(nominee_validation.get("panels", {})):
        nominee_payload = load_payload(nominee_path)
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
        atomic_write_json(sealed_b_path, {**sealed_binding, "panel_record": panel_b})
        if interrupt_after_sealed_panel_b:
            raise RuntimeError("injected interruption after H1 sealed panel B")
        nominee_validation.setdefault("panels", {})["b"] = panel_b
        nominee_validation["finite"] = int(
            bool(int(nominee_validation.get("finite", 0)))
            and bool(int(panel_b.get("finite", 0)))
        )
    else:
        raise ArtifactCompatibilityError(
            "H1 checkpoint contains panel B without sealed evidence"
        )

    if not panel_b_deferred:
        confirmation = confirm_nominee_on_b(
            nomination, validations, thresholds.stability.density_ratio
        )
    selection = {
        "gate": (
            "density_ratio_fixed_endpoint_selection_pending_panel_b"
            if panel_b_deferred
            else "density_ratio_fixed_endpoint_selection"
        ),
        "evaluation_status": (
            "not_evaluated" if panel_b_deferred else "evaluated"
        ),
        "passed": (
            0 if panel_b_deferred else int(bool(int(nomination["passed"])))
        ),
        "selected_step": int(confirmation.get("selected_step", 0)),
        "nominee_step": nominee_step,
        "endpoint_step": nominee_step,
        "endpoint_policy": nomination["endpoint_policy"],
        "nomination": nomination,
        "confirmation": confirmation,
        "panel_b_deferred": int(panel_b_deferred),
        "panel_b_evaluation_count": 0 if panel_b_deferred else 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    selected_step = int(selection.get("selected_step", 0))
    selected_path = (
        None
        if panel_b_deferred
        else checkpoints / f"step-{selected_step:08d}.pt"
    )
    if selected_path is not None and not selected_path.is_file():
        raise ArtifactCompatibilityError("H1 selected checkpoint is missing")

    final_original = checkpoints / f"step-{final_step:08d}.pt"
    finalized = checkpoints / (
        f"a-only-finalized-step-{final_step:08d}.pt"
        if panel_b_deferred
        else f"finalized-step-{final_step:08d}.pt"
    )
    final_payload = load_payload(final_original)
    final_payload["validation_records"] = copy.deepcopy(validations)
    final_payload["checkpoint_selection"] = copy.deepcopy(selection)
    atomic_torch_save(finalized, final_payload)
    atomic_write_json(
        latest_path,
        {
            "schema": H1_TASK_SCHEMA + "-latest",
            "schema_version": H1_TASK_SCHEMA_VERSION,
            "filename": finalized.name,
            "sha256": file_fingerprint(finalized),
            "step": final_step,
            "stream_cursor": final_step,
            "accumulation_cursor": 0,
            "trust_bank_cursor": 0,
            "fingerprints": bound_fingerprints,
            "panel_b_deferred": int(panel_b_deferred),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    best_copy: Path | None = None
    nominee_copy: Path | None = None
    if not panel_b_deferred:
        assert selected_path is not None
        best_copy = checkpoints / "best_ema.pt"
        nominee_copy = checkpoints / "nominee_ema.pt"
        atomic_copy_file(selected_path, best_copy)
        atomic_copy_file(nominee_path, nominee_copy)
        atomic_write_json(
            best_pointer_path,
            {
                "schema": H1_TASK_SCHEMA + "-best",
                "schema_version": H1_TASK_SCHEMA_VERSION,
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
                "fingerprints": bound_fingerprints,
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
        )

    # The scientific endpoint is fixed independently of sealed-B acceptance.
    # B may reject the classifier, but C/D must still measure the step-4000
    # EMA function rather than silently falling back to analytic step zero.
    audit_step = nominee_step
    audit_path = checkpoints / f"step-{audit_step:08d}.pt"
    audited = load_payload(audit_path)
    model.load_state_dict(audited["ema_state_dict"], strict=True)
    model.eval()
    diagnostics = head._history_diagnostics(
        history,
        train_steps=total_steps,
        warmup_steps=int(args.clip_warmup_steps),
    )
    boundary = head._normalized_boundary_certificate(model)
    post_ramp_rows = [
        row
        for row in history
        if int(row.get("step", 0)) > int(controller_config.ramp_steps)
    ]
    active_fraction = (
        1.0
        if ratio == 0.0
        else float(
            np.mean(
                [float(row.get("controller_active", 0)) for row in post_ramp_rows]
            )
        )
        if post_ramp_rows
        else 0.0
    )
    tracking_errors = [
        float(row.get("controller_ratio_tracking_relative_error", math.inf))
        for row in post_ramp_rows
        if int(row.get("controller_active", 0)) == 1
    ]
    maximum_tracking_error = max(tracking_errors, default=0.0 if ratio == 0.0 else math.inf)
    post_ramp_h1_floor_hits = sum(
        int(row.get("controller_post_ramp_h1_floor_failure", 0))
        for row in post_ramp_rows
    )
    controller_health_pass = int(
        len(history) == total_steps
        and all(int(row.get("controller_pass", 0)) == 1 for row in history)
        and (
            ratio == 0.0
            or (
                active_fraction >= 0.99
                and maximum_tracking_error <= float(controller_config.tracking_rtol)
                and post_ramp_h1_floor_hits == 0
            )
        )
    )
    h1_diagnostics = {
        "h1_ratio": ratio,
        "lambda_base": float(calibration_record["lambda_base"]),
        "legacy_lambda_base_advisory": float(calibration_record["lambda_base"]),
        "legacy_lambda_used_for_optimization": 0,
        "target_gradient_ratio": ratio,
        "controller_version": H1_GRADIENT_CONTROL_VERSION,
        "controller_config": controller_config.to_record(),
        "controller_active_fraction_post_ramp": active_fraction,
        "controller_maximum_tracking_error_post_ramp": maximum_tracking_error,
        "controller_post_ramp_h1_floor_hits": post_ramp_h1_floor_hits,
        "controller_health_pass": controller_health_pass,
        "value_scale": value_scale,
        "energy_scale": energy_scale,
        "operator_version": H1_TRUST_OPERATOR_VERSION,
        "trust_plan_fingerprint": trust_plan.fingerprint,
        "mean_value_rms": float(
            np.mean([row["h1_value_rms"] for row in history])
        )
        if history
        else 0.0,
        "mean_natural_energy_rms": float(
            np.mean([row["h1_natural_energy_rms"] for row in history])
        )
        if history
        else 0.0,
        "mean_physical_flux_step_rms": float(
            np.mean([row["h1_physical_flux_step_rms"] for row in history])
        )
        if history
        else 0.0,
        "bce_h1_gradient_cosine_quantiles": head._quantiles(
            [float(row["bce_h1_gradient_cosine"]) for row in history]
        ),
        "h1_scaled_gradient_norm_quantiles": head._quantiles(
            [float(row["h1_scaled_gradient_norm"]) for row in history]
        ),
        "h1_raw_gradient_norm_quantiles": head._quantiles(
            [float(row["h1_raw_gradient_norm"]) for row in history]
        ),
        "controller_coefficient_quantiles": head._quantiles(
            [float(row["controller_h1_coefficient"]) for row in history]
        ),
        "controller_combined_gradient_norm_quantiles": head._quantiles(
            [float(row["controller_combined_gradient_norm"]) for row in history]
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    h1_health_pass = int(
        int(calibration_record.get("passed", 0)) == 1
        and controller_health_pass == 1
        and len(history) == total_steps
        and all(
            int(row.get("h1_reference_banks", -1))
            == int(trust_plan.banks_per_update)
            and len(list(row.get("h1_reference_bank_fingerprints", [])))
            == int(trust_plan.banks_per_update)
            and len(set(row.get("h1_reference_bank_fingerprints", [])))
            == int(trust_plan.banks_per_update)
            and all(
                math.isfinite(float(row.get(name, math.nan)))
                for name in (
                    "h1_value_mse",
                    "h1_natural_energy",
                    "h1_physical_flux_energy",
                    "h1_normalized_value",
                    "h1_normalized_natural_energy",
                    "h1_normalized_objective",
                    "h1_scaled_optimizer_loss",
                    "bce_scaled_gradient_norm",
                    "h1_scaled_gradient_norm",
                    "bce_h1_gradient_dot",
                    "bce_h1_gradient_cosine",
                )
            )
            for row in history
        )
    )
    h1_diagnostics["h1_health_pass"] = h1_health_pass
    # Preserve the existing optimization-diagnostics envelope while exposing
    # the H1 block directly for callers that consume task metrics.
    diagnostics["h1"] = h1_diagnostics
    metrics: dict[str, Any] = {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": int(
            all(
                bool(int(row.get("optimizer_finite", 0)))
                and all(
                    math.isfinite(float(row.get(name, math.nan)))
                    for name in (
                        "scaled_loss",
                        "h1_scaled_optimizer_loss",
                        "scaled_preclip_gradient_norm",
                        "optimizer_update_norm",
                        "ema_update_norm",
                    )
                )
                for row in history
            )
        ),
        "model_seed": int(model_seed),
        "h1_multiplier": ratio,
        "target_ratio": ratio,
        "fixed_endpoint_step": total_steps,
        "selected_step": selected_step,
        "nominee_step": nominee_step,
        "audit_step": audit_step,
        "panel_b_deferred": int(panel_b_deferred),
        "panel_b_evaluation_count": 0 if panel_b_deferred else 1,
        "selection": selection,
        "checkpoints": validations,
        "boundary_admissible": int(boundary["passed"]),
        "optimization_diagnostics": diagnostics,
        "h1_diagnostics": h1_diagnostics,
        "h1_health_pass": h1_health_pass,
        "controller_health_pass": controller_health_pass,
        "controller_active_fraction": active_fraction,
        "maximum_ratio_relative_error": maximum_tracking_error,
        "post_ramp_h1_floor_hit_count": post_ramp_h1_floor_hits,
        "nonfinite_coefficient_count": 0,
        "controller_active_fraction_post_ramp": active_fraction,
        "controller_maximum_tracking_error_post_ramp": maximum_tracking_error,
        "controller_post_ramp_h1_floor_hits": post_ramp_h1_floor_hits,
        "post_warmup_clip_fraction": diagnostics["post_warmup_clip_fraction"],
        "final_500_clip_fraction": diagnostics["final_500_clip_fraction"],
        "final_200_clip_fraction": diagnostics["final_200_clip_fraction"],
        "maximum_clip_fraction_observed": max(
            float(diagnostics["post_warmup_clip_fraction"]),
            float(diagnostics["final_500_clip_fraction"]),
            float(diagnostics["final_200_clip_fraction"]),
        ),
        "audit_panels": {},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    metrics["optimizer_health_pass"] = int(
        bool(metrics["finite"])
        and float(metrics["maximum_clip_fraction_observed"]) <= 0.10
    )
    panel_b_record = dict(dict(nominee_validation.get("panels", {})).get("b", {}))
    analytic_source = (
        terminal_a if panel_b_deferred else panel_b_record
    )
    if task == "bounded_teacher" and isinstance(
        analytic_source.get("analytic"), Mapping
    ):
        analytic = dict(analytic_source["analytic"])
        if panel_b_deferred:
            metrics["fixed_endpoint_panel_a_analytic_metrics"] = analytic
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
        metrics["selection_data_end_relative_flux_l2"] = (
            relatives[-1] if relatives else None
        )
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
            "gate": (
                "h1_density_ratio_pilot_task_awaiting_panel_b"
                if panel_b_deferred
                else "h1_density_ratio_pilot_task"
            ),
            "evaluation_status": "evaluated",
            "passed": int(bool(metrics["complete"]) and bool(metrics["finite"])),
            "panel_b_deferred": int(panel_b_deferred),
            "claim_scope": (
                "complete fixed-endpoint panel-A training evidence only"
                if panel_b_deferred
                else "complete fixed-endpoint pilot task with sealed panel B"
            ),
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
        "h1_ratio": ratio,
        "h1_multiplier": ratio,
        "target_gradient_ratio": ratio,
        "controller_config": controller_config.to_record(),
        "controller_health_pass": controller_health_pass,
        "h1_diagnostics": h1_diagnostics,
        "h1_health_pass": h1_health_pass,
        "selected_step": selected_step,
        "nominee_step": nominee_step,
        "audit_step": audit_step,
        "panel_b_deferred": int(panel_b_deferred),
        "panel_b_evaluation_count": 0 if panel_b_deferred else 1,
        "training_step": final_step,
        "target_training_steps": total_steps,
        "checkpoint_selection": selection,
        "optimization_diagnostics": diagnostics,
        "boundary_admissibility_certificate": boundary,
        "selected_checkpoint_path": (
            None if best_copy is None else str(best_copy.resolve())
        ),
        "selected_checkpoint_sha256": (
            None if best_copy is None else file_fingerprint(best_copy)
        ),
        "nominee_checkpoint_path": (
            str(nominee_path.resolve())
            if nominee_copy is None
            else str(nominee_copy.resolve())
        ),
        "nominee_checkpoint_sha256": file_fingerprint(
            nominee_path if nominee_copy is None else nominee_copy
        ),
        "sealed_panel_b_sha256": (
            None if panel_b_deferred else file_fingerprint(sealed_b_path)
        ),
        "stream_plan": stream_plan_record(stream_plan),
        "paired_stream_plan": paired_mixture_stream_plan_record(paired_stream_plan),
        "h1_trust_plan": h1_trust_plan_record(trust_plan),
        "h1_calibration": calibration_record,
        "model_schema": NORMALIZED_HEAD_MODEL_VERSION,
        "head_coordinate_version": NORMALIZED_HEAD_COORDINATE_VERSION,
        "optimizer_coordinate": coordinate_conjugate_adamw_record(
            model,
            body_lr=float(learning_rate),
            eps=1e-8,
            weight_decay=float(args.weight_decay),
        ),
        "fingerprints": bound_fingerprints,
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
                    "lower_bound_overall": dict(panel_record["overall"])[
                        "lower_bound"
                    ],
                    "bce_data_end": dict(panel_record["data_end"])["bce"],
                    "lower_bound_data_end": dict(panel_record["data_end"])[
                        "lower_bound"
                    ],
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
            analytic_bins = list(dict(panel_record.get("analytic", {})).get("time_bins", []))
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
        analytic_bins = list(dict(panel_record.get("analytic", {})).get("time_bins", []))
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
        "schema": H1_TASK_SCHEMA + "-result",
        "schema_version": H1_TASK_SCHEMA_VERSION,
        "task": task,
        "model_seed": int(model_seed),
        "h1_ratio": ratio,
        "target_gradient_ratio": ratio,
        "controller_config": controller_config.to_record(),
        "panel_b_deferred": int(panel_b_deferred),
        "panel_b_evaluation_count": 0 if panel_b_deferred else 1,
        "metrics": metrics,
        "gate": gate,
        "training_summary": summary,
        "fingerprints": bound_fingerprints,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result = json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
    result_path = task_dir / "task_result.json"
    atomic_write_json(result_path, result)
    atomic_write_json(
        task_dir / "task_status.json",
        {
            "schema": H1_TASK_SCHEMA + "-status",
            "schema_version": H1_TASK_SCHEMA_VERSION,
            "status": "awaiting_panel_b" if panel_b_deferred else "complete",
            "task": task,
            "model_seed": int(model_seed),
            "h1_ratio": ratio,
            "target_gradient_ratio": ratio,
            "controller_config_fingerprint": controller_config.fingerprint,
            "training_step": final_step,
            "selected_step": selected_step,
            "nominee_step": nominee_step,
            "panel_b_deferred": int(panel_b_deferred),
            "panel_b_evaluation_count": 0 if panel_b_deferred else 1,
            "fingerprints": bound_fingerprints,
            "task_result_sha256": file_fingerprint(result_path),
            "sealed_panel_b_sha256": (
                None if panel_b_deferred else file_fingerprint(sealed_b_path)
            ),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    return result
