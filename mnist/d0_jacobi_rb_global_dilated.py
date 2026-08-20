"""Global dilated score model for the Jacobi/RB boundary-tangent controller.

This module changes one scientific axis relative to the frequency-one v4
predictor: the spatial receptive field.  Four circular convolutions with
dilations 1, 2, 4, and 8 have a contiguous 31-pixel receptive field, spanning
the 28 by 28 torus.  The permitted ``ModelInputs`` firewall, density scaling,
metadata, frozen frequency-one coordinate field, color/head gather, local
affine branch, raw-score output, and boundary-tangent wrapper semantics are
otherwise retained.

The bare predictor returns the finite score coefficient ``q``.  The wrapper's
``score_prediction`` methods also return ``q``; only ``forward`` applies the
mobility once to return ``m = y(1-y)q`` for the Rao--Blackwell tangent loss.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mnist.d0_jacobi_rb_boundary_tangent import edge_pair_geometry
from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate import (
    FREQUENCY1_COORDINATE_SHA256,
    FREQUENCY1_COORDINATE_SHAPE,
    FrequencyOneCoordinateContractError,
    FrequencyOneCoordinateJacobiRBPhasePredictor,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_coarse_residual import zero_initialize_residual
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    GRID_SIZE,
    PHASE_COUNT,
    STATE_SIZE,
    LearnabilityContractError,
    ModelInputs,
    call_model,
)


GLOBAL_DILATED_VERSION = "d0-jacobi-rb-global-dilated-v1"
GLOBAL_DILATED_CONTRACT_SCHEMA = GLOBAL_DILATED_VERSION + "-architecture-contract"
GLOBAL_DILATED_WIDTH = 32
GLOBAL_DILATED_DILATIONS = (1, 2, 4, 8)
GLOBAL_DILATED_RECEPTIVE_FIELD = 31
GLOBAL_DILATED_PARAMETER_COUNT = 34_974


class GlobalDilatedContractError(FrequencyOneCoordinateContractError):
    """The frozen global-dilated model or wrapper contract was violated."""


def _circular_convolution(
    in_channels: int,
    out_channels: int,
    *,
    dilation: int,
    reference: Tensor,
) -> nn.Conv2d:
    """Construct one same-resolution circular 3x3 convolution."""

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
        padding_mode="circular",
        device=reference.device,
        dtype=reference.dtype,
    )


class GlobalDilatedJacobiRBPhasePredictor(
    FrequencyOneCoordinateJacobiRBPhasePredictor
):
    """Width-32 Jacobi/RB score model with a torus-spanning receptive field."""

    def __init__(self, *, width: int = 32, num_classes: int = 10) -> None:
        if width != GLOBAL_DILATED_WIDTH or num_classes != 10:
            raise GlobalDilatedContractError(
                "global-dilated production predictor requires width 32 and 10 classes"
            )
        super().__init__(width=width, num_classes=num_classes)

        # Conv1 from the inherited predictor already has circular dilation 1.
        # Replacing conv2/conv3 and appending conv4 is an architecture-only
        # intervention; no v4 parameter is transplanted.
        reference = self.conv1.weight
        self.conv2 = _circular_convolution(
            self.width, self.width, dilation=2, reference=reference
        )
        self.conv3 = _circular_convolution(
            self.width, self.width, dilation=4, reference=reference
        )
        self.conv4 = _circular_convolution(
            self.width, self.width, dilation=8, reference=reference
        )
        zero_initialize_global_dilated_residual(self)

    def _forward_from_metadata(self, inputs: ModelInputs, metadata: Tensor) -> Tensor:
        """Evaluate the preserved geometry with the four dilated convolutions."""

        state = inputs.later_full_state
        dtype = self.conv1.weight.dtype
        state = state.to(dtype=dtype)
        batch = inputs.batch_size
        density = state.reshape(batch, 1, GRID_SIZE, GRID_SIZE) * float(STATE_SIZE)
        metadata_planes = metadata[:, :, None, None].expand(
            batch, metadata.shape[1], GRID_SIZE, GRID_SIZE
        )
        first_preactivation = self.conv1(
            torch.cat([density, metadata_planes], dim=1)
        )
        coordinate = self.frequency1_coordinate.to(dtype=dtype).unsqueeze(0)
        coordinate_preactivation = F.conv2d(
            coordinate,
            self.coordinate_stem_weight,
            bias=None,
            stride=1,
            padding=0,
        )
        hidden = F.silu(
            first_preactivation
            + coordinate_preactivation.expand(batch, -1, -1, -1)
        )
        hidden = F.silu(self.conv2(hidden))
        hidden = F.silu(self.conv3(hidden))
        hidden = F.silu(self.conv4(hidden))
        spatial = self.spatial_output(hidden).reshape(batch, 4, STATE_SIZE)

        colors = inputs.color.to(dtype=torch.long)
        rows = torch.arange(batch, device=state.device)
        heads = self.head_indices[colors]
        tails = self.tail_indices[colors]
        active_spatial = spatial[rows, colors].gather(1, heads)
        head_mass = state.gather(1, heads) * float(STATE_SIZE)
        tail_mass = state.gather(1, tails) * float(STATE_SIZE)
        local_metadata = metadata[:, None, :].expand(
            batch, EDGES_PER_PHASE, metadata.shape[1]
        )
        local_features = torch.cat(
            [tail_mass[:, :, None], head_mass[:, :, None], local_metadata], dim=2
        )
        local = self.local_affine(local_features).squeeze(-1)
        return active_spatial + local

    def forward_prevalidated(self, inputs: ModelInputs) -> Tensor:
        """Return raw finite-score coordinates after boundary input validation."""

        return super().forward_prevalidated(inputs)

    def forward(self, inputs: ModelInputs) -> Tensor:
        """Return the raw finite score coefficient ``q_theta``."""

        return super().forward(inputs)


def zero_initialize_global_dilated_residual(
    model: GlobalDilatedJacobiRBPhasePredictor,
) -> None:
    """Zero both output paths and the coordinate stem, retaining hidden init."""

    if type(model) is not GlobalDilatedJacobiRBPhasePredictor:
        raise GlobalDilatedContractError(
            "zero initialization requires the exact global-dilated predictor"
        )
    zero_initialize_residual(model)
    with torch.no_grad():
        model.coordinate_stem_weight.zero_()


class GlobalDilatedZeroBaselinePredictor(ZeroBaselineBoundaryTangentPredictor):
    """Mobility-once wrapper around the global-dilated raw score model."""

    def __init__(
        self,
        residual_score: GlobalDilatedJacobiRBPhasePredictor | None = None,
        *,
        zero_residual: bool = True,
    ) -> None:
        active = (
            residual_score
            if residual_score is not None
            else GlobalDilatedJacobiRBPhasePredictor(width=GLOBAL_DILATED_WIDTH)
        )
        if (
            type(active) is not GlobalDilatedJacobiRBPhasePredictor
            or active.width != GLOBAL_DILATED_WIDTH
        ):
            raise GlobalDilatedContractError(
                "residual score must be the exact width-32 global-dilated predictor"
            )
        super().__init__(active, zero_residual=False)
        if zero_residual:
            zero_initialize_global_dilated_residual(active)

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        """Return ``q_theta`` without applying pair mobility."""

        if type(inputs) is not ModelInputs:
            raise GlobalDilatedContractError(
                "predictor accepts only exact permitted ModelInputs"
            )
        score = call_model(self.residual_score, inputs).to(dtype=torch.float64)
        if not bool(torch.isfinite(score).all()):
            raise GlobalDilatedContractError("predicted tangent score is nonfinite")
        return score

    def score_prediction_prevalidated(self, inputs: ModelInputs) -> Tensor:
        """Return one prevalidated ``q_theta`` without a hot-loop host sync."""

        if type(inputs) is not ModelInputs:
            raise GlobalDilatedContractError(
                "prevalidated score requires exact ModelInputs"
            )
        score = self.residual_score.forward_prevalidated(inputs).to(dtype=torch.float64)
        if (
            score.shape != (inputs.batch_size, EDGES_PER_PHASE)
            or score.device != inputs.later_full_state.device
            or not score.dtype.is_floating_point
        ):
            raise LearnabilityContractError(
                "model prediction must be floating [B,392] on the input device"
            )
        # The fused caller performs device-side nonfinite accounting and checks it
        # at the shard boundary.  A Python truth-value predicate here would force
        # one CUDA-to-host synchronization for every controller call.
        return score

    def forward(self, inputs: ModelInputs) -> Tensor:
        """Return ``m_theta = mobility * q_theta``, applying mobility once."""

        if type(inputs) is not ModelInputs:
            raise GlobalDilatedContractError(
                "predictor accepts only exact permitted ModelInputs"
            )
        geometry = edge_pair_geometry(inputs)
        prediction = geometry.mobility * self.score_prediction(inputs)
        return torch.where(
            geometry.mobility == 0.0,
            torch.zeros_like(prediction),
            prediction,
        )


def _canonical_model() -> GlobalDilatedJacobiRBPhasePredictor:
    """Construct a contract fixture without advancing the caller's CPU RNG."""

    with torch.random.fork_rng(devices=[], enabled=True):
        return GlobalDilatedJacobiRBPhasePredictor(width=GLOBAL_DILATED_WIDTH)


def global_dilated_architecture_contract() -> dict[str, Any]:
    """Return and enforce the frozen 34,974-parameter architecture contract."""

    model = _canonical_model()
    parameters = dict(model.named_parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters.values())
    convolutions = (model.conv1, model.conv2, model.conv3, model.conv4)
    dilations = tuple(int(layer.dilation[0]) for layer in convolutions)
    coordinate = model.frequency1_coordinate.detach().to(device="cpu")
    coordinate_bytes = (
        coordinate.contiguous()
        .numpy()
        .astype("<f8", copy=False)
        .tobytes(order="C")
    )
    metadata_channels = 1 + PHASE_COUNT + 4 + 1 + model.num_classes
    forbidden_module_types = (nn.BatchNorm2d, nn.LayerNorm, nn.Dropout, nn.MaxPool2d)
    checks = {
        "exact_model_width": int(model.width == GLOBAL_DILATED_WIDTH),
        "input_channels_density_plus_metadata": int(model.conv1.in_channels == 24),
        "metadata_channels_23": int(metadata_channels == 23),
        "four_circular_3x3_convolutions": int(
            all(
                layer.padding_mode == "circular"
                and tuple(layer.kernel_size) == (3, 3)
                and tuple(layer.padding) == tuple(layer.dilation)
                for layer in convolutions
            )
        ),
        "dilations_1_2_4_8": int(dilations == GLOBAL_DILATED_DILATIONS),
        "constant_width_32": int(
            model.conv1.out_channels == GLOBAL_DILATED_WIDTH
            and all(
                layer.in_channels == GLOBAL_DILATED_WIDTH
                and layer.out_channels == GLOBAL_DILATED_WIDTH
                for layer in convolutions[1:]
            )
        ),
        "spatial_output_1x1_32_to_4": int(
            model.spatial_output.in_channels == GLOBAL_DILATED_WIDTH
            and model.spatial_output.out_channels == 4
            and tuple(model.spatial_output.kernel_size) == (1, 1)
        ),
        "local_affine_25_to_1": int(
            model.local_affine.in_features == 25
            and model.local_affine.out_features == 1
        ),
        "coordinate_stem_4_to_32_bias_free": int(
            tuple(model.coordinate_stem_weight.shape) == (32, 4, 1, 1)
        ),
        "coordinate_buffer_shape": int(
            tuple(coordinate.shape) == FREQUENCY1_COORDINATE_SHAPE
        ),
        "coordinate_buffer_hash": int(
            hashlib.sha256(coordinate_bytes).hexdigest()
            == FREQUENCY1_COORDINATE_SHA256
        ),
        "coordinate_buffer_frozen": int(
            not model.frequency1_coordinate.requires_grad
            and "frequency1_coordinate" not in parameters
        ),
        "no_normalization_dropout_or_pooling": int(
            not any(
                isinstance(module, forbidden_module_types)
                for module in model.modules()
            )
        ),
        "zero_coordinate_stem": int(
            torch.count_nonzero(model.coordinate_stem_weight).item() == 0
        ),
        "zero_spatial_output": int(
            torch.count_nonzero(model.spatial_output.weight).item() == 0
            and torch.count_nonzero(model.spatial_output.bias).item() == 0
        ),
        "zero_local_affine": int(
            torch.count_nonzero(model.local_affine.weight).item() == 0
            and torch.count_nonzero(model.local_affine.bias).item() == 0
        ),
        "trainable_parameter_count_34974": int(
            parameter_count == GLOBAL_DILATED_PARAMETER_COUNT
        ),
    }
    return {
        "schema": GLOBAL_DILATED_CONTRACT_SCHEMA,
        "schema_version": 1,
        "version": GLOBAL_DILATED_VERSION,
        "grid_size": GRID_SIZE,
        "state_size": STATE_SIZE,
        "model_width": model.width,
        "metadata_channels": metadata_channels,
        "input_channels": model.conv1.in_channels,
        "dilations": list(dilations),
        "receptive_field": GLOBAL_DILATED_RECEPTIVE_FIELD,
        "contiguous_offset_range": [-15, 15],
        "spans_28x28_torus": 1,
        "activation": "silu_after_each_hidden_convolution",
        "density_scale": STATE_SIZE,
        "coordinate_channels": 4,
        "coordinate_sha256": FREQUENCY1_COORDINATE_SHA256,
        "spatial_output_channels": model.spatial_output.out_channels,
        "local_affine_features": model.local_affine.in_features,
        "trainable_parameter_count": parameter_count,
        "output_semantics": {
            "residual_score": "q_theta",
            "score_prediction": "q_theta",
            "wrapped_forward": "m_theta=mobility*q_theta",
        },
        "checks": checks,
        "passed": int(all(checks.values())),
    }


__all__ = [
    "GLOBAL_DILATED_CONTRACT_SCHEMA",
    "GLOBAL_DILATED_DILATIONS",
    "GLOBAL_DILATED_PARAMETER_COUNT",
    "GLOBAL_DILATED_RECEPTIVE_FIELD",
    "GLOBAL_DILATED_VERSION",
    "GLOBAL_DILATED_WIDTH",
    "GlobalDilatedContractError",
    "GlobalDilatedJacobiRBPhasePredictor",
    "GlobalDilatedZeroBaselinePredictor",
    "global_dilated_architecture_contract",
    "zero_initialize_global_dilated_residual",
]
