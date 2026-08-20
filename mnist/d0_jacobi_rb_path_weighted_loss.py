"""Path-geometry weighted loss for Jacobi/Rao--Blackwell score training.

The boundary-tangent predictor returns the conormal score

    m_theta = mu(Y) q_theta,   mu(Y) = Y(1-Y).

For the reverse Jacobi SDE, a drift error is naturally measured relative to
its diffusion variance.  This yields the weighted square error

    (m_theta - Zbar)^2 / mu(Y).

The implementation never forms the unstable quotient ``Zbar / mu``.  A
strictly positive floor is applied only to the denominator, while exact
zero-mobility lanes are excluded after verifying that their stored target is
exactly zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE


PATH_WEIGHTED_LOSS_VERSION = "d0-jacobi-rb-path-weighted-loss-v1"
DEFAULT_MOBILITY_FLOOR = 1.0e-4


class PathWeightedLossError(ValueError):
    """The weighted-loss inputs or boundary contract were violated."""


@dataclass(frozen=True)
class PathWeightedLossConfig:
    """Frozen numerical choices for the path-weighted objective."""

    mobility_floor: float = DEFAULT_MOBILITY_FLOOR

    def __post_init__(self) -> None:
        floor = float(self.mobility_floor)
        if not math.isfinite(floor) or not 0.0 < floor <= 0.25:
            raise PathWeightedLossError(
                "mobility_floor must be finite and in (0, 0.25]"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PATH_WEIGHTED_LOSS_VERSION + "-config",
            **asdict(self),
            "target_quotient_formed": 0,
            "zero_mobility_lanes_excluded": 1,
        }


@dataclass(frozen=True)
class PathWeightedLossStatistics:
    """Descriptive statistics for one mobility tensor."""

    active_count: int
    zero_mobility_count: int
    floor_hit_count: int
    floor_hit_fraction: float
    minimum_positive_mobility: float
    maximum_weight: float
    weight_p50: float
    weight_p90: float
    weight_p99: float

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _validate_tensors(
    prediction: Tensor,
    exact_target: Tensor,
    mobility: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if (
        not isinstance(prediction, Tensor)
        or not isinstance(exact_target, Tensor)
        or not isinstance(mobility, Tensor)
        or prediction.shape != exact_target.shape
        or prediction.shape != mobility.shape
        or prediction.ndim != 2
        or prediction.shape[1] != EDGES_PER_PHASE
        or prediction.device != exact_target.device
        or prediction.device != mobility.device
        or not prediction.dtype.is_floating_point
        or not exact_target.dtype.is_floating_point
        or not mobility.dtype.is_floating_point
    ):
        raise PathWeightedLossError(
            "prediction, target, and mobility must be aligned floating [N,392] tensors"
        )
    prediction64 = prediction.to(dtype=torch.float64)
    target64 = exact_target.to(dtype=torch.float64)
    mobility64 = mobility.to(dtype=torch.float64)
    inactive = mobility64 == 0.0
    # CPU callers receive the full fail-closed value audit.  Production CUDA
    # batches come from a hash-verified cache whose values were checked while
    # computing the training-only scales; synchronizing here would add several
    # device-to-host barriers to every optimizer update.
    if prediction64.device.type == "cpu":
        if (
            not bool(torch.isfinite(prediction64).all())
            or not bool(torch.isfinite(target64).all())
            or not bool(torch.isfinite(mobility64).all())
            or bool(torch.any((mobility64 < 0.0) | (mobility64 > 0.25)))
        ):
            raise PathWeightedLossError("weighted-loss tensors are nonfinite or invalid")
        if bool(torch.any(target64[inactive] != 0.0)):
            raise PathWeightedLossError(
                "the exact Rao--Blackwell target must be zero on zero-mobility lanes"
            )
        if not bool(torch.any(~inactive)):
            raise PathWeightedLossError("weighted loss requires at least one active lane")
    return prediction64, target64, mobility64


def path_weighted_target_scale_squared(
    exact_target: Tensor,
    mobility: Tensor,
    *,
    config: PathWeightedLossConfig | None = None,
) -> Tensor:
    """Return ``mean(Zbar^2 / max(mu, floor))`` on active lanes.

    The returned scalar is the training-only normalization used by the
    normalized objective.  It is a squared scale, not an RMS.
    """

    active_config = config or PathWeightedLossConfig()
    zeros = torch.zeros_like(exact_target)
    _, target64, mobility64 = _validate_tensors(zeros, exact_target, mobility)
    active = mobility64 > 0.0
    denominator = mobility64.clamp_min(float(active_config.mobility_floor))
    scale = torch.mean(target64[active].square() / denominator[active])
    if not bool(torch.isfinite(scale)) or not bool(scale > 0.0):
        raise PathWeightedLossError(
            "path-weighted target scale must be finite and positive"
        )
    return scale


def path_weighted_raw_target_mse(
    prediction: Tensor,
    exact_target: Tensor,
    mobility: Tensor,
    *,
    target_scale_squared: Tensor | float,
    config: PathWeightedLossConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return normalized weighted loss, raw weighted loss, and unweighted MSE.

    No quotient target is constructed.  The Bayes minimizer remains
    ``E[Zbar | W]`` because the positive weight is a function of the permitted
    later-state information.
    """

    active_config = config or PathWeightedLossConfig()
    prediction64, target64, mobility64 = _validate_tensors(
        prediction, exact_target, mobility
    )
    if isinstance(target_scale_squared, Tensor):
        if target_scale_squared.numel() != 1:
            raise PathWeightedLossError(
                "target_scale_squared must be one finite positive scalar"
            )
        if target_scale_squared.device.type == "cpu":
            scale_value = float(target_scale_squared.detach().cpu())
            if not math.isfinite(scale_value) or scale_value <= 0.0:
                raise PathWeightedLossError(
                    "target_scale_squared must be one finite positive scalar"
                )
    else:
        scale_value = float(target_scale_squared)
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            raise PathWeightedLossError(
                "target_scale_squared must be one finite positive scalar"
            )
    scale = torch.as_tensor(
        target_scale_squared, dtype=torch.float64, device=prediction.device
    )
    active = mobility64 > 0.0
    denominator = mobility64.clamp_min(float(active_config.mobility_floor))
    residual_squared = (prediction64 - target64).square()
    raw_weighted = torch.mean(residual_squared[active] / denominator[active])
    raw_unweighted = torch.mean(residual_squared)
    return raw_weighted / scale, raw_weighted, raw_unweighted


def mobility_weight_statistics(
    mobility: Tensor | np.ndarray,
    *,
    config: PathWeightedLossConfig | None = None,
) -> PathWeightedLossStatistics:
    """Summarize the effective sample weights ``1/max(mu, floor)``."""

    active_config = config or PathWeightedLossConfig()
    values = (
        mobility.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(mobility, Tensor)
        else np.asarray(mobility, dtype=np.float64)
    )
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise PathWeightedLossError("mobility statistics require finite nonnegative data")
    positive = values[values > 0.0]
    if positive.size == 0:
        raise PathWeightedLossError("mobility statistics require a positive value")
    floor = float(active_config.mobility_floor)
    weights = 1.0 / np.maximum(positive, floor)
    floor_hits = int(np.count_nonzero(positive < floor))
    quantiles = np.quantile(weights, [0.50, 0.90, 0.99])
    return PathWeightedLossStatistics(
        active_count=int(positive.size),
        zero_mobility_count=int(values.size - positive.size),
        floor_hit_count=floor_hits,
        floor_hit_fraction=float(floor_hits / positive.size),
        minimum_positive_mobility=float(np.min(positive)),
        maximum_weight=float(np.max(weights)),
        weight_p50=float(quantiles[0]),
        weight_p90=float(quantiles[1]),
        weight_p99=float(quantiles[2]),
    )


__all__ = [
    "DEFAULT_MOBILITY_FLOOR",
    "PATH_WEIGHTED_LOSS_VERSION",
    "PathWeightedLossConfig",
    "PathWeightedLossError",
    "PathWeightedLossStatistics",
    "mobility_weight_statistics",
    "path_weighted_raw_target_mse",
    "path_weighted_target_scale_squared",
]
