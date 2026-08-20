"""Large global Jacobi/Rao--Blackwell score controller.

The production small controller in :mod:`mnist.d0_jacobi_rb_global_dilated`
has 34,974 trainable parameters.  This module keeps the same permitted model
inputs and the same boundary-tangent output semantics, but replaces its
four-layer width-32 body with a width-128 residual stack.  The bare network
predicts the finite score coefficient ``q``; the wrapper applies the Jacobi
mobility exactly once and returns ``m = y(1-y)q``.

No normalization, dropout, pooling, clipping, projection, or target quotient
is used.  Every spatial convolution is circular so the architecture respects
the 28 by 28 torus.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mnist.d0_jacobi_rb_boundary_tangent import edge_pair_geometry
from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate import (
    FREQUENCY1_COORDINATE_SHA256,
    canonical_frequency1_coordinate_array,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    GRID_SIZE,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    LearnabilityContractError,
    ModelInputs,
    matching_indices,
)


LARGE_GLOBAL_VERSION = "d0-jacobi-rb-global-large-v1"
LARGE_GLOBAL_WIDTH = 128
LARGE_GLOBAL_DILATIONS = (1, 2, 4, 8, 1, 2, 4, 8)
LARGE_GLOBAL_RESIDUAL_BLOCKS = len(LARGE_GLOBAL_DILATIONS)
LARGE_GLOBAL_NUM_CLASSES = 10
LARGE_GLOBAL_PARAMETER_COUNT = 2_390_174


class LargeGlobalContractError(LearnabilityContractError):
    """The large-controller architecture or input contract was violated."""


def _circular_conv(
    in_channels: int,
    out_channels: int,
    *,
    dilation: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=int(dilation),
        dilation=int(dilation),
        padding_mode="circular",
        device=device,
        dtype=dtype,
    )


class _DilatedResidualBlock(nn.Module):
    """Two circular convolutions with a pre-activation residual connection."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.conv1 = _circular_conv(width, width, dilation=self.dilation)
        self.conv2 = _circular_conv(width, width, dilation=self.dilation)

    def forward(self, hidden: Tensor) -> Tensor:
        residual = self.conv2(F.silu(self.conv1(F.silu(hidden))))
        return hidden + residual / math.sqrt(2.0)


class LargeGlobalDilatedJacobiRBPhasePredictor(nn.Module):
    """Approximately 2.4M-parameter global residual predictor of ``q_theta``."""

    def __init__(
        self,
        *,
        width: int = LARGE_GLOBAL_WIDTH,
        num_classes: int = LARGE_GLOBAL_NUM_CLASSES,
    ) -> None:
        super().__init__()
        if width != LARGE_GLOBAL_WIDTH or num_classes != LARGE_GLOBAL_NUM_CLASSES:
            raise LargeGlobalContractError(
                "large production predictor requires width 128 and 10 classes"
            )
        self.width = int(width)
        self.num_classes = int(num_classes)
        metadata_channels = 1 + PHASE_COUNT + 4 + 1 + self.num_classes
        input_channels = 1 + metadata_channels
        self.stem = _circular_conv(input_channels, self.width, dilation=1)
        self.blocks = nn.ModuleList(
            [_DilatedResidualBlock(self.width, dilation) for dilation in LARGE_GLOBAL_DILATIONS]
        )
        self.spatial_output = nn.Conv2d(self.width, 4, kernel_size=1)
        self.local_affine = nn.Linear(2 + metadata_channels, 1)

        coordinate = torch.from_numpy(
            np.array(canonical_frequency1_coordinate_array(), copy=True, order="C")
        )
        self.register_buffer("frequency1_coordinate", coordinate, persistent=True)
        self.coordinate_stem_weight = nn.Parameter(
            torch.zeros((self.width, 4, 1, 1), dtype=self.stem.weight.dtype)
        )
        tails, heads = matching_indices()
        self.register_buffer("tail_indices", tails, persistent=True)
        self.register_buffer("head_indices", heads, persistent=True)
        zero_initialize_large_global_residual(self)

    def _validated_metadata(self, inputs: ModelInputs, dtype: torch.dtype) -> Tensor:
        phase = inputs.phase.to(dtype=torch.long)
        color = inputs.color.to(dtype=torch.long)
        label = inputs.label.to(dtype=torch.long)
        if (
            bool(torch.any((phase < 0) | (phase >= PHASE_COUNT)))
            or bool(torch.any((color < 0) | (color >= 4)))
            or bool(torch.any((label < 0) | (label >= self.num_classes)))
        ):
            raise LargeGlobalContractError("phase/color/label is outside its range")
        expected_color = torch.as_tensor(
            PHASE_MATCHINGS, dtype=torch.long, device=phase.device
        )[phase]
        expected_duration = torch.as_tensor(
            PHASE_DURATIONS, dtype=inputs.duration.dtype, device=phase.device
        )[phase]
        if not torch.equal(color, expected_color):
            raise LargeGlobalContractError("color does not match phase")
        if not torch.equal(inputs.duration, expected_duration):
            raise LargeGlobalContractError("duration does not match phase")
        return self._metadata_without_host_validation(inputs, dtype)

    def _metadata_without_host_validation(
        self, inputs: ModelInputs, dtype: torch.dtype
    ) -> Tensor:
        phase = inputs.phase.to(dtype=torch.long)
        color = inputs.color.to(dtype=torch.long)
        label = inputs.label.to(dtype=torch.long)
        return torch.cat(
            [
                inputs.reverse_time.to(dtype=dtype).reshape(-1, 1),
                F.one_hot(phase, num_classes=PHASE_COUNT).to(dtype=dtype),
                F.one_hot(color, num_classes=4).to(dtype=dtype),
                inputs.duration.to(dtype=dtype).reshape(-1, 1),
                F.one_hot(label, num_classes=self.num_classes).to(dtype=dtype),
            ],
            dim=1,
        )

    def _forward_from_metadata(self, inputs: ModelInputs, metadata: Tensor) -> Tensor:
        state = inputs.later_full_state.to(dtype=self.stem.weight.dtype)
        batch = inputs.batch_size
        density = state.reshape(batch, 1, GRID_SIZE, GRID_SIZE) * float(STATE_SIZE)
        metadata_planes = metadata[:, :, None, None].expand(
            batch, metadata.shape[1], GRID_SIZE, GRID_SIZE
        )
        hidden = self.stem(torch.cat([density, metadata_planes], dim=1))
        coordinate = self.frequency1_coordinate.to(
            device=state.device, dtype=self.stem.weight.dtype
        ).unsqueeze(0)
        coordinate_preactivation = F.conv2d(
            coordinate,
            self.coordinate_stem_weight,
            bias=None,
            stride=1,
            padding=0,
        )
        hidden = hidden + coordinate_preactivation.expand(batch, -1, -1, -1)
        for block in self.blocks:
            hidden = block(hidden)
        spatial = self.spatial_output(F.silu(hidden)).reshape(batch, 4, STATE_SIZE)

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
        """Hot-loop forward after a caller has validated schedule metadata."""

        if type(inputs) is not ModelInputs:
            raise LargeGlobalContractError(
                "prevalidated forward accepts only exact ModelInputs"
            )
        metadata = self._metadata_without_host_validation(inputs, self.stem.weight.dtype)
        return self._forward_from_metadata(inputs, metadata)

    def forward(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise LargeGlobalContractError("forward accepts only exact ModelInputs")
        metadata = self._validated_metadata(inputs, self.stem.weight.dtype)
        return self._forward_from_metadata(inputs, metadata)


def zero_initialize_large_global_residual(
    model: LargeGlobalDilatedJacobiRBPhasePredictor,
) -> None:
    """Make the initial learned controller exactly zero without zeroing the body."""

    if type(model) is not LargeGlobalDilatedJacobiRBPhasePredictor:
        raise LargeGlobalContractError(
            "zero initialization requires the exact large predictor"
        )
    with torch.no_grad():
        model.spatial_output.weight.zero_()
        model.spatial_output.bias.zero_()
        model.local_affine.weight.zero_()
        model.local_affine.bias.zero_()
        model.coordinate_stem_weight.zero_()


class LargeGlobalDilatedZeroBaselinePredictor(nn.Module):
    """Mobility-once wrapper exposing both ``q_theta`` and ``m_theta``."""

    def __init__(
        self,
        residual_score: LargeGlobalDilatedJacobiRBPhasePredictor | None = None,
        *,
        zero_residual: bool = True,
    ) -> None:
        super().__init__()
        active = residual_score or LargeGlobalDilatedJacobiRBPhasePredictor()
        if type(active) is not LargeGlobalDilatedJacobiRBPhasePredictor:
            raise LargeGlobalContractError("residual score has the wrong architecture")
        self.residual_score = active
        if zero_residual:
            zero_initialize_large_global_residual(active)

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise LargeGlobalContractError("predictor accepts only exact ModelInputs")
        score = self.residual_score(inputs).to(dtype=torch.float64)
        if not bool(torch.isfinite(score).all()):
            raise LargeGlobalContractError("predicted tangent score is nonfinite")
        return score

    def score_prediction_prevalidated(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise LargeGlobalContractError(
                "prevalidated score requires exact ModelInputs"
            )
        score = self.residual_score.forward_prevalidated(inputs).to(dtype=torch.float64)
        if (
            score.shape != (inputs.batch_size, EDGES_PER_PHASE)
            or score.device != inputs.later_full_state.device
            or not score.dtype.is_floating_point
        ):
            raise LargeGlobalContractError("model prediction must be floating [B,392]")
        return score

    def forward(self, inputs: ModelInputs) -> Tensor:
        geometry = edge_pair_geometry(inputs)
        prediction = geometry.mobility * self.score_prediction(inputs)
        return torch.where(
            geometry.mobility == 0.0,
            torch.zeros_like(prediction),
            prediction,
        )


class LargeEulerianJacobiDDPMModel(nn.Module):
    """Candidate-sampler-compatible model wrapper for the large predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = LargeGlobalDilatedZeroBaselinePredictor()

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        return self.predictor.score_prediction(inputs)

    def forward(self, inputs: ModelInputs) -> Tensor:
        return self.predictor(inputs)


def large_global_parameter_count() -> int:
    with torch.random.fork_rng(devices=[], enabled=True):
        model = LargeEulerianJacobiDDPMModel()
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True)
class LargeGlobalArchitectureContract:
    version: str
    width: int
    residual_blocks: int
    dilations: tuple[int, ...]
    parameter_count: int
    coordinate_sha256: str
    circular_convolutions: int
    normalization_layers: int
    dropout_layers: int
    pooling_layers: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def large_global_architecture_contract() -> dict[str, Any]:
    count = large_global_parameter_count()
    if count != LARGE_GLOBAL_PARAMETER_COUNT:
        raise LargeGlobalContractError(
            f"large parameter contract changed: expected {LARGE_GLOBAL_PARAMETER_COUNT}, got {count}"
        )
    with torch.random.fork_rng(devices=[], enabled=True):
        model = LargeEulerianJacobiDDPMModel()
    modules = tuple(model.modules())
    convolution_count = sum(isinstance(module, nn.Conv2d) for module in modules)
    normalization_count = sum(
        isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.GroupNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.LayerNorm,
            ),
        )
        for module in modules
    )
    dropout_count = sum(isinstance(module, nn.Dropout) for module in modules)
    pooling_count = sum(
        isinstance(
            module,
            (
                nn.AvgPool2d,
                nn.MaxPool2d,
                nn.AdaptiveAvgPool2d,
                nn.AdaptiveMaxPool2d,
            ),
        )
        for module in modules
    )
    for module in modules:
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            if module.padding_mode != "circular":
                raise LargeGlobalContractError("a spatial convolution is not circular")
    coordinate = model.predictor.residual_score.frequency1_coordinate
    digest = hashlib.sha256(
        coordinate.detach().cpu().numpy().astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()
    if digest != FREQUENCY1_COORDINATE_SHA256:
        raise LargeGlobalContractError("frozen coordinate field changed")
    contract = LargeGlobalArchitectureContract(
        version=LARGE_GLOBAL_VERSION,
        width=LARGE_GLOBAL_WIDTH,
        residual_blocks=LARGE_GLOBAL_RESIDUAL_BLOCKS,
        dilations=LARGE_GLOBAL_DILATIONS,
        parameter_count=count,
        coordinate_sha256=digest,
        circular_convolutions=convolution_count,
        normalization_layers=normalization_count,
        dropout_layers=dropout_count,
        pooling_layers=pooling_count,
    )
    if normalization_count or dropout_count or pooling_count:
        raise LargeGlobalContractError("large architecture gained normalization/dropout/pooling")
    return contract.to_record()


__all__ = [
    "LARGE_GLOBAL_DILATIONS",
    "LARGE_GLOBAL_PARAMETER_COUNT",
    "LARGE_GLOBAL_RESIDUAL_BLOCKS",
    "LARGE_GLOBAL_VERSION",
    "LARGE_GLOBAL_WIDTH",
    "LargeEulerianJacobiDDPMModel",
    "LargeGlobalArchitectureContract",
    "LargeGlobalContractError",
    "LargeGlobalDilatedJacobiRBPhasePredictor",
    "LargeGlobalDilatedZeroBaselinePredictor",
    "large_global_architecture_contract",
    "large_global_parameter_count",
    "zero_initialize_large_global_residual",
]
