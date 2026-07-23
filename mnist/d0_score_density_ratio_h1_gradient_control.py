"""Pure gradient-ratio controller for the D0 stopped-EMA H1 trust term.

The balanced BCE and stopped-EMA H1 objectives are differentiated
independently by the task runner.  This module combines their already-scaled
parameter gradients so that, before global clipping, the H1 contribution has
the prescribed norm relative to the BCE contribution::

    lambda_t = rho_t * ||g_bce||_2 / ||g_h1||_2
    g_total  = g_bce + stop_gradient(lambda_t) * g_h1

Norms, inner products, and ratio diagnostics are evaluated in float64 in the
model's normalized-head coordinates.  The controller deliberately does not
clip, step an optimizer, or own an autograd graph; callers must apply the one
global clip only after installing the returned combined gradients.

This is a finite-time optimizer controller, not the gradient of a fixed scalar
objective.  It contains no physical-training or sampler functionality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


H1_GRADIENT_CONTROL_SCHEMA = "experiment12-d0-h1-gradient-ratio-control"
H1_GRADIENT_CONTROL_SCHEMA_VERSION = 1
H1_GRADIENT_CONTROL_VERSION = "d0-stopped-ema-h1-gradient-ratio-v1"
H1_GRADIENT_CONTROL_COORDINATES = "normalized-head-v2"
H1_GRADIENT_CONTROL_RAMP_STEPS = 100
H1_GRADIENT_CONTROL_NORM_FLOOR = 1e-12
H1_GRADIENT_CONTROL_TRACKING_RTOL = 1e-4


__all__ = [
    "H1_GRADIENT_CONTROL_SCHEMA",
    "H1_GRADIENT_CONTROL_SCHEMA_VERSION",
    "H1_GRADIENT_CONTROL_VERSION",
    "H1_GRADIENT_CONTROL_COORDINATES",
    "H1_GRADIENT_CONTROL_RAMP_STEPS",
    "H1_GRADIENT_CONTROL_NORM_FLOOR",
    "H1_GRADIENT_CONTROL_TRACKING_RTOL",
    "GradientRatioControllerConfig",
    "GradientRatioControlResult",
    "gradient_ratio_ramp",
    "copy_parameter_gradients",
    "compose_gradient_ratio_update",
    "assign_controlled_gradients",
]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class GradientRatioControllerConfig:
    """Frozen numerical contract for the online gradient-ratio controller."""

    ramp_steps: int = H1_GRADIENT_CONTROL_RAMP_STEPS
    norm_floor: float = H1_GRADIENT_CONTROL_NORM_FLOOR
    tracking_rtol: float = H1_GRADIENT_CONTROL_TRACKING_RTOL
    coordinate_system: str = H1_GRADIENT_CONTROL_COORDINATES
    schema: str = H1_GRADIENT_CONTROL_SCHEMA + "-config"
    schema_version: int = H1_GRADIENT_CONTROL_SCHEMA_VERSION
    controller_version: str = H1_GRADIENT_CONTROL_VERSION

    def __post_init__(self) -> None:
        if int(self.ramp_steps) <= 0:
            raise ValueError("ramp_steps must be positive")
        if not math.isfinite(float(self.norm_floor)) or float(self.norm_floor) <= 0.0:
            raise ValueError("norm_floor must be finite and positive")
        if (
            not math.isfinite(float(self.tracking_rtol))
            or float(self.tracking_rtol) < 0.0
        ):
            raise ValueError("tracking_rtol must be finite and nonnegative")
        if not str(self.coordinate_system).strip():
            raise ValueError("coordinate_system must be nonempty")

    @property
    def fingerprint(self) -> str:
        return _canonical_fingerprint(self.to_record(include_fingerprint=False))

    def to_record(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "coefficient_semantics": "stop-gradient",
                "composition": "g_bce + lambda_t * g_h1",
                "global_clipping_owned_by_caller": 1,
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result


def gradient_ratio_ramp(
    optimizer_step: int,
    *,
    ramp_steps: int = H1_GRADIENT_CONTROL_RAMP_STEPS,
) -> float:
    """Return ``min(1, max(0, (step - 1) / ramp_steps))`` exactly.

    Optimizer steps are one-indexed.  Consequently the stopped-EMA fixed point
    at step one receives no H1 contribution and the ramp reaches one at step
    ``ramp_steps + 1``.
    """

    step = int(optimizer_step)
    width = int(ramp_steps)
    if step < 1:
        raise ValueError("optimizer_step must be at least one")
    if width <= 0:
        raise ValueError("ramp_steps must be positive")
    return min(1.0, max(0.0, float(step - 1) / float(width)))


def copy_parameter_gradients(
    parameters: Sequence[nn.Parameter],
) -> tuple[Tensor | None, ...]:
    """Take a detached snapshot of a parameter-gradient vector."""

    return tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )


def _validate_gradient_vectors(
    bce_gradients: Sequence[Tensor | None],
    h1_gradients: Sequence[Tensor | None],
) -> tuple[tuple[Tensor | None, ...], tuple[Tensor | None, ...], torch.device]:
    bce = tuple(bce_gradients)
    h1 = tuple(h1_gradients)
    if len(bce) != len(h1):
        raise ValueError("BCE and H1 gradient vectors differ in length")
    if not bce:
        raise ValueError("gradient vectors must contain at least one entry")

    device: torch.device | None = None
    for index, (left, right) in enumerate(zip(bce, h1, strict=True)):
        if left is not None and right is not None and left.shape != right.shape:
            raise ValueError(f"gradient shapes differ at index {index}")
        for name, value in (("BCE", left), ("H1", right)):
            if value is None:
                continue
            if value.is_sparse:
                raise ValueError(f"{name} gradient at index {index} is sparse")
            if not (value.is_floating_point() and not value.is_complex()):
                raise TypeError(f"{name} gradient at index {index} must be real floating")
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError("all gradients must be on one device")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(
                    f"{name} gradient at index {index} is nonfinite"
                )
    return bce, h1, device if device is not None else torch.device("cpu")


def _float64_geometry(
    left: Sequence[Tensor | None],
    right: Sequence[Tensor | None],
    *,
    device: torch.device,
) -> tuple[float, float, float, float]:
    left_sq = torch.zeros((), dtype=torch.float64, device=device)
    right_sq = torch.zeros((), dtype=torch.float64, device=device)
    dot = torch.zeros((), dtype=torch.float64, device=device)
    for lhs, rhs in zip(left, right, strict=True):
        if lhs is not None:
            lhs64 = lhs.detach().to(dtype=torch.float64)
            left_sq = left_sq + lhs64.square().sum()
        if rhs is not None:
            rhs64 = rhs.detach().to(dtype=torch.float64)
            right_sq = right_sq + rhs64.square().sum()
        if lhs is not None and rhs is not None:
            dot = dot + (
                lhs.detach().to(dtype=torch.float64)
                * rhs.detach().to(dtype=torch.float64)
            ).sum()
    left_norm_tensor = torch.sqrt(left_sq)
    right_norm_tensor = torch.sqrt(right_sq)
    denominator = left_norm_tensor * right_norm_tensor
    cosine_tensor = torch.where(
        denominator > 0.0, dot / denominator, torch.zeros_like(denominator)
    )
    values = tuple(
        float(value.detach().cpu())
        for value in (left_norm_tensor, right_norm_tensor, dot, cosine_tensor)
    )
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("gradient geometry is nonfinite")
    return values


def _scaled_gradients(
    gradients: Sequence[Tensor | None], coefficient: float
) -> tuple[Tensor | None, ...]:
    if not math.isfinite(float(coefficient)):
        raise FloatingPointError("H1 coefficient is nonfinite")
    if float(coefficient) == 0.0:
        return tuple(None for _ in gradients)
    result: list[Tensor | None] = []
    for index, gradient in enumerate(gradients):
        if gradient is None:
            result.append(None)
            continue
        contribution = gradient.detach() * float(coefficient)
        if not bool(torch.isfinite(contribution).all()):
            raise FloatingPointError(
                f"scaled H1 contribution at index {index} is nonfinite"
            )
        result.append(contribution)
    return tuple(result)


def _sum_gradients(
    first: Sequence[Tensor | None], second: Sequence[Tensor | None]
) -> tuple[Tensor | None, ...]:
    result: list[Tensor | None] = []
    for lhs, rhs in zip(first, second, strict=True):
        if lhs is None and rhs is None:
            result.append(None)
        elif lhs is None:
            result.append(rhs.detach().clone())
        elif rhs is None:
            result.append(lhs.detach().clone())
        else:
            value = lhs.detach() + rhs.detach()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("combined gradient is nonfinite")
            result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class GradientRatioControlResult:
    """Combined gradient vector and its fail-closed controller diagnostics."""

    gradients: tuple[Tensor | None, ...] = field(repr=False)
    optimizer_step: int
    requested_ratio: float
    ramp_fraction: float
    target_ratio: float
    bce_gradient_norm: float
    h1_gradient_norm: float
    bce_h1_gradient_dot: float
    bce_h1_gradient_cosine: float
    h1_coefficient: float
    h1_contribution_gradient_norm: float
    realized_ratio: float
    ratio_tracking_relative_error: float
    combined_gradient_norm: float
    norm_floor: float
    tracking_rtol: float
    bce_gradient_floor_hit: int
    h1_gradient_floor_hit: int
    stationary_bce_noop: int
    ramp_zero_noop: int
    controller_active: int
    ratio_tracking_pass: int
    post_ramp_h1_floor_failure: int
    controller_pass: int
    coordinate_system: str
    config_fingerprint: str
    schema: str = H1_GRADIENT_CONTROL_SCHEMA + "-result"
    schema_version: int = H1_GRADIENT_CONTROL_SCHEMA_VERSION
    controller_version: str = H1_GRADIENT_CONTROL_VERSION

    def detached_record(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("gradients")
        result.update(
            {
                "coefficient_semantics": "stop-gradient",
                "global_clipping_applied": 0,
                "physical_training_performed": 0,
                "sampling_performed": 0,
            }
        )
        return result


def compose_gradient_ratio_update(
    bce_gradients: Sequence[Tensor | None],
    h1_gradients: Sequence[Tensor | None],
    *,
    target_ratio: float,
    optimizer_step: int,
    config: GradientRatioControllerConfig | None = None,
) -> GradientRatioControlResult:
    """Compose one preclip BCE-plus-H1 parameter-gradient update.

    ``bce_gradients`` must already include the frozen BCE loss multiplier;
    ``h1_gradients`` must be the gradient of the unscaled normalized H1
    objective.  Both are interpreted in normalized-head coordinates.

    A BCE norm at or below the floor is an explicit stationary no-op for the
    H1 contribution.  An H1 floor hit once the ramp has completed is recorded
    as a fail-closed controller result.  During the ramp it remains reportable
    (with failed ratio tracking) so the task-level post-ramp health gate can
    adjudicate the prescribed active window.
    """

    cfg = config if config is not None else GradientRatioControllerConfig()
    requested = float(target_ratio)
    if not math.isfinite(requested) or requested < 0.0:
        raise ValueError("target_ratio must be finite and nonnegative")
    ramp = gradient_ratio_ramp(optimizer_step, ramp_steps=int(cfg.ramp_steps))
    instantaneous_target = requested * ramp
    if not math.isfinite(instantaneous_target):
        raise FloatingPointError("ramped target ratio is nonfinite")

    bce, h1, device = _validate_gradient_vectors(bce_gradients, h1_gradients)
    bce_norm, h1_norm, raw_dot, raw_cosine = _float64_geometry(
        bce, h1, device=device
    )
    floor = float(cfg.norm_floor)
    bce_floor_hit = int(bce_norm <= floor)
    h1_floor_hit = int(h1_norm <= floor)
    stationary = int(bool(bce_floor_hit))
    ramp_zero = int(instantaneous_target == 0.0)

    if stationary or ramp_zero:
        coefficient = 0.0
    elif h1_floor_hit:
        coefficient = 0.0
    else:
        coefficient = instantaneous_target * bce_norm / h1_norm
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise FloatingPointError("H1 coefficient is invalid")

    contribution = _scaled_gradients(h1, coefficient)
    combined = _sum_gradients(bce, contribution)
    contribution_norm = _float64_geometry(
        contribution,
        tuple(None for _ in contribution),
        device=device,
    )[0]
    combined_norm = _float64_geometry(
        combined,
        tuple(None for _ in combined),
        device=device,
    )[0]

    if bce_norm > floor:
        realized = contribution_norm / bce_norm
    else:
        realized = 0.0
    if instantaneous_target > 0.0 and not stationary:
        tracking_error = abs(realized - instantaneous_target) / instantaneous_target
    else:
        tracking_error = 0.0
    if not math.isfinite(realized) or not math.isfinite(tracking_error):
        raise FloatingPointError("gradient-ratio diagnostics are nonfinite")

    active = int(
        instantaneous_target > 0.0
        and not stationary
        and not h1_floor_hit
        and coefficient > 0.0
    )
    tracking_pass = int(
        stationary
        or instantaneous_target == 0.0
        or (
            active
            and tracking_error <= float(cfg.tracking_rtol)
        )
    )
    post_ramp_h1_failure = int(
        requested > 0.0
        and ramp >= 1.0
        and not stationary
        and bool(h1_floor_hit)
    )
    # A pre-completion H1 floor is recorded, but only a completed-ramp floor
    # is the plan's immediate fail-closed condition.  Ratio-tracking gates may
    # still reject repeated inactive ramp steps at the task level.
    controller_pass = int(
        not post_ramp_h1_failure
        and (
            tracking_pass
            or (ramp < 1.0 and bool(h1_floor_hit) and not stationary)
        )
    )

    return GradientRatioControlResult(
        gradients=combined,
        optimizer_step=int(optimizer_step),
        requested_ratio=requested,
        ramp_fraction=float(ramp),
        target_ratio=float(instantaneous_target),
        bce_gradient_norm=float(bce_norm),
        h1_gradient_norm=float(h1_norm),
        bce_h1_gradient_dot=float(raw_dot),
        bce_h1_gradient_cosine=float(raw_cosine),
        h1_coefficient=float(coefficient),
        h1_contribution_gradient_norm=float(contribution_norm),
        realized_ratio=float(realized),
        ratio_tracking_relative_error=float(tracking_error),
        combined_gradient_norm=float(combined_norm),
        norm_floor=floor,
        tracking_rtol=float(cfg.tracking_rtol),
        bce_gradient_floor_hit=bce_floor_hit,
        h1_gradient_floor_hit=h1_floor_hit,
        stationary_bce_noop=stationary,
        ramp_zero_noop=ramp_zero,
        controller_active=active,
        ratio_tracking_pass=tracking_pass,
        post_ramp_h1_floor_failure=post_ramp_h1_failure,
        controller_pass=controller_pass,
        coordinate_system=str(cfg.coordinate_system),
        config_fingerprint=cfg.fingerprint,
    )


def assign_controlled_gradients(
    parameters: Sequence[nn.Parameter],
    controlled: GradientRatioControlResult | Sequence[Tensor | None],
) -> None:
    """Install controlled gradients without clipping or stepping an optimizer."""

    gradients = controlled.gradients if isinstance(
        controlled, GradientRatioControlResult
    ) else tuple(controlled)
    values = tuple(parameters)
    if len(values) != len(gradients):
        raise ValueError("parameter and gradient vectors differ in length")
    for index, (parameter, gradient) in enumerate(zip(values, gradients, strict=True)):
        if gradient is None:
            parameter.grad = None
            continue
        if gradient.shape != parameter.shape:
            raise ValueError(f"gradient shape differs from parameter at index {index}")
        if gradient.device != parameter.device or gradient.dtype != parameter.dtype:
            raise ValueError(
                f"gradient dtype or device differs from parameter at index {index}"
            )
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"controlled gradient at index {index} is nonfinite")
        parameter.grad = gradient.detach().clone()

