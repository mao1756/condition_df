"""Optimizer-unit calibration for the D0 implicit-score controls.

The score and flux gates are evaluated in physical, unscaled units.  The
optimizer, however, should see a loss whose initial gradient has a predictable
size.  This module keeps those two concerns separate:

* :func:`calibrate_initial_loss_scale` measures an unscaled gradient on an
  explicitly identified training-only state set and returns a frozen positive
  multiplier; and
* :func:`scaled_backward_and_clip` applies that multiplier before backward and
  reports both the scaled pre-clip norm used by the optimizer gate and its
  implied unscaled norm.

The helpers deliberately contain no filesystem, experiment-CLI, or sampler
coupling.  Random initialization and probe plans are owned by the caller and
must be included in the calibration binding recorded in the artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


LOSS_SCALE_CALIBRATION_SCHEMA = "d0-score-initial-gradient-loss-scale"
LOSS_SCALE_CALIBRATION_SCHEMA_VERSION = 1


__all__ = [
    "LOSS_SCALE_CALIBRATION_SCHEMA",
    "LOSS_SCALE_CALIBRATION_SCHEMA_VERSION",
    "LossScaleCalibration",
    "ScaledGradientDiagnostics",
    "calibrate_initial_loss_scale",
    "derive_initial_loss_scale",
    "parameter_gradient_l2_norm",
    "scaled_backward_and_clip",
    "summarize_scaled_gradient_history",
]


ObjectiveBatchFactory = Callable[[], Iterable[tuple[Tensor, int]]]


def _finite_positive(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


def _parameters(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [parameter for parameter in parameters if parameter.requires_grad]


def parameter_gradient_l2_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    """Return the global L2 norm of the gradients currently on ``parameters``.

    The return tensor stays on the gradient device.  A parameter collection
    with no gradients has norm zero on the first parameter's device (or CPU
    when the collection itself is empty).
    """

    values = list(parameters)
    norms = [parameter.grad.detach().norm(2) for parameter in values if parameter.grad is not None]
    if norms:
        return torch.linalg.vector_norm(torch.stack(norms))
    if values:
        return values[0].detach().new_zeros(())
    return torch.tensor(0.0)


def derive_initial_loss_scale(raw_gradient_norm: float, target_gradient_norm: float) -> float:
    """Return ``min(1, target / raw)`` after fail-closed validation."""

    raw = float(raw_gradient_norm)
    target = float(target_gradient_norm)
    if not _finite_positive(raw):
        raise ValueError("raw_gradient_norm must be finite and positive")
    if not _finite_positive(target):
        raise ValueError("target_gradient_norm must be finite and positive")
    scale = min(1.0, target / raw)
    if not _finite_positive(scale):
        raise FloatingPointError("derived loss scale is zero or non-finite")
    return float(scale)


@dataclass(frozen=True)
class LossScaleCalibration:
    """Serializable result of a deterministic initial-gradient calibration."""

    objective_kind: str
    calibration_split: str
    calibration_state_count: int
    calibration_state_sha256: str
    target_initial_gradient_norm: float
    unscaled_initial_gradient_norm: float
    scaled_initial_gradient_norm: float
    loss_scale: float
    unscaled_objective: float
    binding: dict[str, Any]
    schema: str = LOSS_SCALE_CALIBRATION_SCHEMA
    schema_version: int = LOSS_SCALE_CALIBRATION_SCHEMA_VERSION
    training_only: int = 1
    scale_capped_at_one: int = 0
    sampling_performed: int = 0

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable calibration record."""

        return asdict(self)


def calibrate_initial_loss_scale(
    model: nn.Module,
    objective_batches: ObjectiveBatchFactory,
    *,
    objective_kind: str,
    calibration_state_sha256: str,
    binding: Mapping[str, Any],
    target_initial_gradient_norm: float = 0.10,
    calibration_state_count: int = 256,
    calibration_split: str = "train",
) -> LossScaleCalibration:
    """Calibrate a frozen loss multiplier on a fixed training-only state set.

    ``objective_batches`` must create a fresh iterable of ``(loss, count)``
    pairs.  Each loss is the *mean unscaled objective* for its batch.  The
    helper accumulates the correctly weighted full-set gradient without
    retaining all batch graphs at once.  Exactly ``calibration_state_count``
    states must be yielded; production callers use the default of 256.

    The caller owns deterministic model initialization and, for the implicit
    objective, deterministic probe construction.  Their seeds and hashes
    belong in ``binding`` so orchestration can fingerprint the resulting
    artifact.  Calibration does not update model parameters and clears the
    temporary gradients before returning.
    """

    kind = str(objective_kind).strip()
    if not kind:
        raise ValueError("objective_kind must be non-empty")
    split = str(calibration_split).strip().lower()
    if split != "train":
        raise ValueError("loss-scale calibration is restricted to the train split")
    state_hash = str(calibration_state_sha256).strip()
    if not state_hash:
        raise ValueError("calibration_state_sha256 must be non-empty")
    expected_count = int(calibration_state_count)
    if expected_count <= 0:
        raise ValueError("calibration_state_count must be positive")
    target = float(target_initial_gradient_norm)
    if not _finite_positive(target):
        raise ValueError("target_initial_gradient_norm must be finite and positive")

    parameters = _parameters(model.parameters())
    if not parameters:
        raise ValueError("model has no trainable parameters")
    model.zero_grad(set_to_none=True)
    seen = 0
    objective_value = 0.0
    try:
        for loss, batch_count_value in objective_batches():
            batch_count = int(batch_count_value)
            if batch_count <= 0:
                raise ValueError("calibration objective batch counts must be positive")
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise TypeError("each calibration objective must be a scalar tensor")
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("calibration objective is non-finite")
            if seen + batch_count > expected_count:
                raise ValueError("calibration objective yielded more states than declared")
            weight = float(batch_count) / float(expected_count)
            objective_value += float(loss.detach().cpu()) * weight
            (loss * weight).backward()
            seen += batch_count

        if seen != expected_count:
            raise ValueError(
                f"calibration objective yielded {seen} states; expected {expected_count}"
            )
        raw_norm = float(parameter_gradient_l2_norm(parameters).detach().cpu())
        if not _finite_positive(raw_norm):
            raise FloatingPointError(
                "initial unscaled calibration gradient is zero or non-finite"
            )
        loss_scale = derive_initial_loss_scale(raw_norm, target)
        scaled_norm = raw_norm * loss_scale
        if not (_finite_positive(scaled_norm) and scaled_norm <= target * (1.0 + 1e-12)):
            raise FloatingPointError("scaled calibration gradient is invalid")
        return LossScaleCalibration(
            objective_kind=kind,
            calibration_split=split,
            calibration_state_count=expected_count,
            calibration_state_sha256=state_hash,
            target_initial_gradient_norm=target,
            unscaled_initial_gradient_norm=raw_norm,
            scaled_initial_gradient_norm=scaled_norm,
            loss_scale=loss_scale,
            unscaled_objective=float(objective_value),
            binding=dict(binding),
            scale_capped_at_one=int(loss_scale == 1.0),
        )
    finally:
        model.zero_grad(set_to_none=True)


@dataclass(frozen=True)
class ScaledGradientDiagnostics:
    """Optimizer diagnostics for one scaled backward/clip operation."""

    unscaled_loss: float
    scaled_loss: float
    loss_scale: float
    raw_gradient_norm: float
    scaled_preclip_gradient_norm: float
    grad_clip: float | None
    clipped: int

    def to_record(self) -> dict[str, float | int | None]:
        return asdict(self)


def scaled_backward_and_clip(
    unscaled_loss: Tensor,
    parameters: Iterable[nn.Parameter],
    *,
    loss_scale: float,
    grad_clip: float | None,
) -> ScaledGradientDiagnostics:
    """Scale a scalar loss, backpropagate, clip, and return pre-clip norms.

    ``raw_gradient_norm`` is recovered as
    ``scaled_preclip_gradient_norm / loss_scale``.  The clipping decision and
    therefore optimizer-health gates use only the scaled pre-clip norm.
    Callers must zero gradients before invoking this helper and perform the
    optimizer step afterwards.
    """

    if not isinstance(unscaled_loss, Tensor) or unscaled_loss.numel() != 1:
        raise TypeError("unscaled_loss must be a scalar tensor")
    if not bool(torch.isfinite(unscaled_loss.detach())):
        raise FloatingPointError("unscaled loss is non-finite")
    scale = float(loss_scale)
    # The scale-repair calibration itself is capped at one.  The shared
    # backward helper remains compatible with legacy callers that may carry a
    # previously frozen positive multiplier greater than one.
    if not _finite_positive(scale):
        raise ValueError("loss_scale must be finite and positive")
    clip = None if grad_clip is None else float(grad_clip)
    if clip is not None and (not math.isfinite(clip) or clip <= 0.0):
        raise ValueError("grad_clip must be finite and positive, or None")

    trainable = _parameters(parameters)
    if not trainable:
        raise ValueError("parameters contains no trainable parameters")
    scaled_loss = unscaled_loss * scale
    if not bool(torch.isfinite(scaled_loss.detach())):
        raise FloatingPointError("scaled loss is non-finite")
    scaled_loss.backward()

    if clip is None:
        norm_tensor = parameter_gradient_l2_norm(trainable)
    else:
        norm_tensor = torch.nn.utils.clip_grad_norm_(
            trainable, max_norm=clip, error_if_nonfinite=True
        )
    scaled_norm = float(torch.as_tensor(norm_tensor).detach().cpu())
    if not math.isfinite(scaled_norm):
        raise FloatingPointError("scaled pre-clip gradient norm is non-finite")
    raw_norm = scaled_norm / scale
    return ScaledGradientDiagnostics(
        unscaled_loss=float(unscaled_loss.detach().cpu()),
        scaled_loss=float(scaled_loss.detach().cpu()),
        loss_scale=scale,
        raw_gradient_norm=float(raw_norm),
        scaled_preclip_gradient_norm=scaled_norm,
        grad_clip=clip,
        clipped=int(clip is not None and scaled_norm > clip),
    )


def summarize_scaled_gradient_history(
    history: Sequence[Mapping[str, Any]],
    *,
    warmup_steps: int,
    grad_clip: float,
) -> dict[str, Any]:
    """Summarize optimizer health using scaled pre-clip norms exclusively."""

    clip = float(grad_clip)
    if not _finite_positive(clip):
        raise ValueError("grad_clip must be finite and positive")
    after = [row for row in history if int(row.get("step", 0)) > int(warmup_steps)]
    if not after:
        after = list(history)

    def values(name: str) -> np.ndarray:
        result = np.asarray(
            [
                float(row[name])
                for row in after
                if name in row and math.isfinite(float(row[name]))
            ],
            dtype=np.float64,
        )
        return result

    def quantiles(name: str) -> dict[str, float | None]:
        result = values(name)
        names = ("q00", "q10", "q50", "q90", "q99", "q100")
        if result.size == 0:
            return {name: None for name in names}
        return {
            name: float(value)
            for name, value in zip(
                names, np.quantile(result, (0.0, 0.1, 0.5, 0.9, 0.99, 1.0))
            )
        }

    scaled = values("scaled_preclip_gradient_norm")
    if scaled.size != len(after):
        raise ValueError(
            "every post-warmup history row must contain a finite "
            "scaled_preclip_gradient_norm"
        )
    computed_clipped = scaled > clip
    recorded_clipped = np.asarray(
        [bool(int(row.get("clipped", int(value)))) for row, value in zip(after, computed_clipped)],
        dtype=bool,
    )
    mismatch_count = int(np.count_nonzero(computed_clipped != recorded_clipped))
    return {
        "gradient_norm_source": "scaled_preclip_gradient_norm",
        "post_warmup_steps": int(len(after)),
        "post_warmup_clip_count": int(np.count_nonzero(computed_clipped)),
        "post_warmup_clip_fraction": (
            float(np.mean(computed_clipped)) if computed_clipped.size else 0.0
        ),
        "recorded_clip_mismatch_count": mismatch_count,
        "recorded_clip_flags_consistent": int(mismatch_count == 0),
        "quantiles": {
            "raw_gradient_norm": quantiles("raw_gradient_norm"),
            "scaled_preclip_gradient_norm": quantiles(
                "scaled_preclip_gradient_norm"
            ),
            "unscaled_loss": quantiles("unscaled_loss"),
            "scaled_loss": quantiles("scaled_loss"),
        },
    }
