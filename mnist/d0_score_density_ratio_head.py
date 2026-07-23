"""Function-equivalent normalized head for D0 density-ratio controls.

The boundary-smooth potential used by the first density-ratio controls reduces
its full-resolution energy map with a spatial sum.  Consequently, the final
``1 x 1`` convolution is expressed in coordinates whose gradients are larger
by the number of grid cells.  This additive module represents the same scalar
potential with a spatial mean and the exact coordinate change

``out_normalized = N * out_legacy``, where ``N = grid_size ** 2``.

The model topology and state-dict keys intentionally remain unchanged.  A
coordinate-conjugate AdamW constructor supplies the transformed learning rate,
epsilon, and decoupled weight decay for the final weight and bias.  No physical
training or sampler functionality belongs in this module.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .d0_score_boundary_controls import D0BoundarySmoothPotentialUNet


NORMALIZED_HEAD_MODEL_VERSION = "d0-boundary-smooth-potential-unet-mean-head-v2"
NORMALIZED_HEAD_COORDINATE_VERSION = "d0-spatial-sum-to-mean-head-coordinate-v1"
COORDINATE_CONJUGATE_ADAMW_VERSION = "d0-mean-head-coordinate-adamw-v1"

BODY_PARAMETER_GROUP_NAME = "body"
NORMALIZED_HEAD_PARAMETER_GROUP_NAME = "normalized_head"
HEAD_PARAMETER_NAMES = ("out.weight", "out.bias")


__all__ = [
    "NORMALIZED_HEAD_MODEL_VERSION",
    "NORMALIZED_HEAD_COORDINATE_VERSION",
    "COORDINATE_CONJUGATE_ADAMW_VERSION",
    "BODY_PARAMETER_GROUP_NAME",
    "NORMALIZED_HEAD_PARAMETER_GROUP_NAME",
    "HEAD_PARAMETER_NAMES",
    "D0BoundarySmoothMeanHeadPotentialUNet",
    "head_coordinate_factor",
    "legacy_state_dict_to_normalized_head",
    "normalized_state_dict_to_legacy_head",
    "legacy_ema_state_to_normalized_head",
    "normalized_ema_state_to_legacy_head",
    "legacy_gradient_dict_to_normalized_head",
    "normalized_gradient_dict_to_legacy_head",
    "coordinate_conjugate_parameter_groups",
    "build_coordinate_conjugate_adamw",
    "coordinate_conjugate_adamw_record",
    "normalized_gradient_diagnostics",
]


def _grid_size(value: nn.Module | int) -> int:
    if isinstance(value, nn.Module):
        config = getattr(value, "config", None)
        if config is None or not hasattr(config, "grid_size"):
            raise ValueError("model must expose config.grid_size")
        value = int(config.grid_size)
    n = int(value)
    if n <= 0:
        raise ValueError("grid_size must be positive")
    return n


def head_coordinate_factor(value: nn.Module | int) -> int:
    """Return the exact spatial sum-to-mean coordinate factor ``N``."""

    n = _grid_size(value)
    return n * n


class D0BoundarySmoothMeanHeadPotentialUNet(D0BoundarySmoothPotentialUNet):
    """Boundary-smooth D0 potential with a spatial-mean scalar head.

    If ``legacy`` is a :class:`D0BoundarySmoothPotentialUNet` and this model's
    ``out.weight`` and ``out.bias`` are ``N`` times the legacy values, both
    models represent exactly the same scalar function.  The inherited final
    layer remains zero-initialized, so fresh training still starts at the
    analytic zero potential.
    """

    model_version = NORMALIZED_HEAD_MODEL_VERSION
    head_coordinate_version = NORMALIZED_HEAD_COORDINATE_VERSION
    scalar_reduction = "spatial_mean"

    def forward(self, tau: Tensor | float, states: Tensor, labels: Tensor) -> Tensor:
        return self.potential_map(tau, states, labels).flatten(1).mean(dim=1)


def _copy_tensor_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    # ``deepcopy`` retains tensor dtype/device and also preserves non-tensor
    # checkpoint metadata should a caller pass a state-like mapping.
    return copy.deepcopy(dict(values))


def _transform_head_tensors(
    values: Mapping[str, Any],
    *,
    factor: float,
    head_parameter_names: tuple[str, str] = HEAD_PARAMETER_NAMES,
) -> dict[str, Any]:
    transformed = _copy_tensor_mapping(values)
    missing = [name for name in head_parameter_names if name not in transformed]
    if missing:
        raise KeyError(f"head tensor mapping is missing {missing}")
    for name in head_parameter_names:
        value = transformed[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
        transformed[name] = value * float(factor)
    return transformed


def legacy_state_dict_to_normalized_head(
    state_dict: Mapping[str, Any], grid_size: int
) -> dict[str, Any]:
    """Map legacy spatial-sum model/EMA tensors into mean-head coordinates."""

    return _transform_head_tensors(
        state_dict, factor=float(head_coordinate_factor(grid_size))
    )


def normalized_state_dict_to_legacy_head(
    state_dict: Mapping[str, Any], grid_size: int
) -> dict[str, Any]:
    """Map mean-head model/EMA tensors back into legacy sum coordinates."""

    return _transform_head_tensors(
        state_dict, factor=1.0 / float(head_coordinate_factor(grid_size))
    )


def legacy_ema_state_to_normalized_head(
    ema_state: Mapping[str, Any], grid_size: int
) -> dict[str, Any]:
    """Explicit EMA alias for the sum-to-mean state coordinate transform."""

    return legacy_state_dict_to_normalized_head(ema_state, grid_size)


def normalized_ema_state_to_legacy_head(
    ema_state: Mapping[str, Any], grid_size: int
) -> dict[str, Any]:
    """Explicit EMA alias for the mean-to-sum state coordinate transform."""

    return normalized_state_dict_to_legacy_head(ema_state, grid_size)


def legacy_gradient_dict_to_normalized_head(
    gradients: Mapping[str, Any], grid_size: int
) -> dict[str, Any]:
    """Convert legacy parameter gradients to mean-head gradients.

    For ``theta_normalized = N * theta_legacy``, the chain rule gives
    ``grad_normalized = grad_legacy / N`` for the two final-head tensors.
    """

    return _transform_head_tensors(
        gradients, factor=1.0 / float(head_coordinate_factor(grid_size))
    )


def normalized_gradient_dict_to_legacy_head(
    gradients: Mapping[str, Any], grid_size: int
) -> dict[str, Any]:
    """Reconstruct legacy-coordinate gradients from mean-head gradients."""

    return _transform_head_tensors(
        gradients, factor=float(head_coordinate_factor(grid_size))
    )


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def coordinate_conjugate_parameter_groups(
    model: D0BoundarySmoothMeanHeadPotentialUNet,
    *,
    body_lr: float,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
) -> list[dict[str, Any]]:
    """Return named AdamW groups conjugate to legacy sum-head coordinates."""

    if not isinstance(model, D0BoundarySmoothMeanHeadPotentialUNet):
        raise TypeError("model must use the normalized spatial-mean head")
    eta = _finite_positive(body_lr, name="body_lr")
    epsilon = _finite_positive(eps, name="eps")
    decay = _finite_nonnegative(weight_decay, name="weight_decay")
    factor = float(head_coordinate_factor(model))

    named = dict(model.named_parameters())
    missing = [name for name in HEAD_PARAMETER_NAMES if name not in named]
    if missing:
        raise ValueError(f"normalized model is missing head parameters {missing}")
    head_ids = {id(named[name]) for name in HEAD_PARAMETER_NAMES}
    body = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    head = [named[name] for name in HEAD_PARAMETER_NAMES]
    if not body or len(head) != len(HEAD_PARAMETER_NAMES):
        raise ValueError("normalized model parameter partition is invalid")
    if len({id(parameter) for parameter in body + head}) != len(body) + len(head):
        raise ValueError("normalized model parameter groups overlap")

    return [
        {
            "name": BODY_PARAMETER_GROUP_NAME,
            "params": body,
            "lr": eta,
            "eps": epsilon,
            "weight_decay": decay,
        },
        {
            "name": NORMALIZED_HEAD_PARAMETER_GROUP_NAME,
            "params": head,
            "lr": factor * eta,
            "eps": epsilon / factor,
            "weight_decay": decay / factor,
        },
    ]


def build_coordinate_conjugate_adamw(
    model: D0BoundarySmoothMeanHeadPotentialUNet,
    *,
    body_lr: float,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    amsgrad: bool = False,
) -> torch.optim.AdamW:
    """Construct AdamW whose head updates are conjugate to legacy AdamW."""

    beta1, beta2 = (float(betas[0]), float(betas[1]))
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("betas must lie in [0, 1)")
    groups = coordinate_conjugate_parameter_groups(
        model, body_lr=body_lr, eps=eps, weight_decay=weight_decay
    )
    return torch.optim.AdamW(groups, betas=(beta1, beta2), amsgrad=bool(amsgrad))


def coordinate_conjugate_adamw_record(
    model: D0BoundarySmoothMeanHeadPotentialUNet,
    *,
    body_lr: float,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
) -> dict[str, Any]:
    """Return the fingerprintable optimizer coordinate definition."""

    factor = head_coordinate_factor(model)
    eta = _finite_positive(body_lr, name="body_lr")
    epsilon = _finite_positive(eps, name="eps")
    decay = _finite_nonnegative(weight_decay, name="weight_decay")
    return {
        "version": COORDINATE_CONJUGATE_ADAMW_VERSION,
        "model_version": NORMALIZED_HEAD_MODEL_VERSION,
        "coordinate_version": NORMALIZED_HEAD_COORDINATE_VERSION,
        "grid_size": _grid_size(model),
        "coordinate_factor": factor,
        "betas": [float(betas[0]), float(betas[1])],
        "groups": [
            {
                "name": BODY_PARAMETER_GROUP_NAME,
                "parameter_names": [
                    name
                    for name, _ in model.named_parameters()
                    if name not in HEAD_PARAMETER_NAMES
                ],
                "lr": eta,
                "eps": epsilon,
                "weight_decay": decay,
            },
            {
                "name": NORMALIZED_HEAD_PARAMETER_GROUP_NAME,
                "parameter_names": list(HEAD_PARAMETER_NAMES),
                "lr": factor * eta,
                "eps": epsilon / factor,
                "weight_decay": decay / factor,
            },
        ],
    }


def normalized_gradient_diagnostics(
    model: D0BoundarySmoothMeanHeadPotentialUNet,
) -> dict[str, Any]:
    """Summarize gradients in normalized and reconstructed legacy coordinates."""

    if not isinstance(model, D0BoundarySmoothMeanHeadPotentialUNet):
        raise TypeError("model must use the normalized spatial-mean head")
    factor = float(head_coordinate_factor(model))
    body_sq: Tensor | None = None
    head_sq: Tensor | None = None
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        square = parameter.grad.detach().to(dtype=torch.float64).square().sum()
        if name in HEAD_PARAMETER_NAMES:
            head_sq = square if head_sq is None else head_sq + square
        else:
            body_sq = square if body_sq is None else body_sq + square

    reference = next(model.parameters())
    zero = torch.zeros((), device=reference.device, dtype=torch.float64)
    body_sq = zero if body_sq is None else body_sq
    head_sq = zero if head_sq is None else head_sq
    normalized_sq = body_sq + head_sq
    legacy_head_sq = head_sq * (factor * factor)
    legacy_sq = body_sq + legacy_head_sq
    finite = bool(
        torch.isfinite(body_sq)
        & torch.isfinite(head_sq)
        & torch.isfinite(normalized_sq)
        & torch.isfinite(legacy_sq)
    )

    def _root(value: Tensor) -> float:
        return float(torch.sqrt(value).detach().cpu())

    def _fraction(numerator: Tensor, denominator: Tensor) -> float:
        if not bool(denominator > 0.0):
            return 0.0
        return float((numerator / denominator).detach().cpu())

    return {
        "schema": "experiment12-d0-normalized-head-gradient-diagnostics",
        "schema_version": 1,
        "coordinate_version": NORMALIZED_HEAD_COORDINATE_VERSION,
        "coordinate_factor": int(factor),
        "finite": int(finite),
        "normalized_gradient_norm": _root(normalized_sq),
        "body_gradient_norm": _root(body_sq),
        "normalized_head_gradient_norm": _root(head_sq),
        "normalized_head_squared_fraction": _fraction(head_sq, normalized_sq),
        "reconstructed_legacy_gradient_norm": _root(legacy_sq),
        "reconstructed_legacy_head_gradient_norm": _root(legacy_head_sq),
        "reconstructed_legacy_head_squared_fraction": _fraction(
            legacy_head_sq, legacy_sq
        ),
        "missing_gradient_names": missing,
        "missing_gradient_count": len(missing),
    }
