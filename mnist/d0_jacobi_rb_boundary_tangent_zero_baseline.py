"""Exact-zero boundary-tangent representation for Jacobi/RB v3.

The physical target remains the raw Rao--Blackwell label.  This module only
removes the fitted tangent table from the predictor coordinates:

``m_theta(W) = y(1-y) q_theta(W)``.

The conceptual baseline is exact zero and therefore has no model state and no
persisted array.  There is deliberately no training, cache, selection,
controller, or sampling code here.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_boundary_tangent import (
    BoundaryTangentContractError,
    TANGENT_BASELINE_SHAPE,
    edge_pair_geometry,
)
from mnist.d0_jacobi_rb_coarse_residual import zero_initialize_residual
from mnist.d0_jacobi_rb_learnability import (
    JacobiRBPhasePredictor,
    ModelInputs,
    call_model,
    configure_exact_synthetic_teacher,
)


ZERO_BASELINE_VERSION = "d0-jacobi-rb-boundary-tangent-zero-baseline-v3"
ZERO_BASELINE_CONTRACT_SCHEMA = ZERO_BASELINE_VERSION + "-contract"
ZERO_BASELINE_SHAPE = TANGENT_BASELINE_SHAPE
ZERO_BASELINE_BYTE_LENGTH = 702_464
ZERO_BASELINE_SHA256 = (
    "a0cfe4ce7c13acb57ced3803a69321b59b790ae5ec652a6c03476676d6204149"
)


def zero_baseline_contract() -> dict[str, Any]:
    """Return the immutable, stateless ``q_B := 0`` contract.

    The hash describes a conceptual C-order binary64 array.  The array is not
    constructed or retained by the predictor and must not be persisted by a
    v3 workflow.
    """

    byte_length = 8
    for extent in ZERO_BASELINE_SHAPE:
        byte_length *= int(extent)
    if byte_length != ZERO_BASELINE_BYTE_LENGTH:
        raise BoundaryTangentContractError(
            "conceptual zero-baseline byte length changed"
        )
    digest = hashlib.sha256(bytes(byte_length)).hexdigest()
    if digest != ZERO_BASELINE_SHA256:
        raise BoundaryTangentContractError("conceptual zero-baseline hash changed")
    return {
        "schema": ZERO_BASELINE_CONTRACT_SCHEMA,
        "schema_version": 1,
        "formula": "q_B := 0",
        "baseline_kind": "fixed_exact_zero",
        "fitted_parameter_count": 0,
        "baseline_array_persisted": 0,
        "training_labels_used": 0,
        "validation_labels_used": 0,
        "confirmation_labels_used": 0,
        "target_modified": 0,
        "conceptual_array_shape": list(ZERO_BASELINE_SHAPE),
        "conceptual_array_dtype": "float64",
        "conceptual_array_order": "C",
        "conceptual_array_byte_length": ZERO_BASELINE_BYTE_LENGTH,
        "conceptual_array_sha256": ZERO_BASELINE_SHA256,
    }


def exact_zero_baseline_prediction(inputs: ModelInputs) -> Tensor:
    """Return the exact stateless baseline prediction for diagnostics."""

    geometry = edge_pair_geometry(inputs)
    return torch.zeros_like(geometry.mobility)


class ZeroBaselineBoundaryTangentPredictor(nn.Module):
    """Unchanged width-32 score network in exact-zero tangent coordinates."""

    def __init__(
        self,
        residual_score: JacobiRBPhasePredictor | None = None,
        *,
        zero_residual: bool = True,
    ) -> None:
        super().__init__()
        active = (
            residual_score
            if residual_score is not None
            else JacobiRBPhasePredictor(width=32)
        )
        if not isinstance(active, JacobiRBPhasePredictor) or active.width != 32:
            raise BoundaryTangentContractError(
                "residual score must be the unchanged width-32 JacobiRBPhasePredictor"
            )
        self.residual_score = active
        if zero_residual:
            zero_initialize_residual(self.residual_score)

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        """Return exactly the residual network's finite score coefficient."""

        if type(inputs) is not ModelInputs:
            raise BoundaryTangentContractError(
                "predictor accepts only exact permitted ModelInputs"
            )
        score = call_model(self.residual_score, inputs).to(dtype=torch.float64)
        if not bool(torch.isfinite(score).all()):
            raise BoundaryTangentContractError("predicted tangent score is nonfinite")
        return score

    def forward(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise BoundaryTangentContractError(
                "predictor accepts only exact permitted ModelInputs"
            )
        geometry = edge_pair_geometry(inputs)
        prediction = geometry.mobility * self.score_prediction(inputs)
        return torch.where(
            geometry.mobility == 0.0,
            torch.zeros_like(prediction),
            prediction,
        )


def configure_exact_synthetic_zero_baseline_teacher(
    model: ZeroBaselineBoundaryTangentPredictor,
) -> None:
    """Configure the residual network as the exact analytic tangent teacher."""

    if not isinstance(model, ZeroBaselineBoundaryTangentPredictor):
        raise BoundaryTangentContractError("synthetic teacher model has wrong type")
    configure_exact_synthetic_teacher(model.residual_score)


__all__ = [
    "ZERO_BASELINE_BYTE_LENGTH",
    "ZERO_BASELINE_CONTRACT_SCHEMA",
    "ZERO_BASELINE_SHA256",
    "ZERO_BASELINE_SHAPE",
    "ZERO_BASELINE_VERSION",
    "ZeroBaselineBoundaryTangentPredictor",
    "configure_exact_synthetic_zero_baseline_teacher",
    "exact_zero_baseline_prediction",
    "zero_baseline_contract",
]
