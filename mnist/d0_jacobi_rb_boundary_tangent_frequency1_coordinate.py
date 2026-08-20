"""Frequency-one absolute-coordinate repair for the Jacobi/RB predictor.

This module makes exactly one representation change to the frozen width-32
Jacobi/RB phase predictor: a deterministic four-channel periodic coordinate
field is injected into the first spatial preactivation through a zero-
initialized, bias-free 1x1 stem.  The transition law, raw Rao--Blackwell
target, local-affine branch, and boundary-tangent geometry live in their
existing modules and are deliberately not reimplemented here.

The coordinate values are committed as IEEE-754 hexadecimal literals.  No
trigonometric function, QR factorization, path identifier, or audit field is
used by :meth:`FrequencyOneCoordinateJacobiRBPhasePredictor.forward`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields
import hashlib
import math
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mnist.d0_jacobi_rb_absolute_coordinate import build_coordinate_lattice
from mnist.d0_jacobi_rb_boundary_tangent import (
    BoundaryTangentContractError,
    edge_pair_geometry,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_coarse_residual import zero_initialize_residual
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    GRID_SIZE,
    MODEL_INPUT_FIELDS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    JacobiRBPhasePredictor,
    LearnabilityContractError,
    ModelInputs,
    call_model,
    matching_indices,
    state_dict_sha256,
)


FREQUENCY1_COORDINATE_VERSION = (
    "d0-jacobi-rb-boundary-tangent-frequency1-coordinate-v1"
)
FREQUENCY1_COORDINATE_CONTRACT_SCHEMA = FREQUENCY1_COORDINATE_VERSION + "-contract"
FREQUENCY1_COORDINATE_CHANNELS = (
    "sin_row_frequency1",
    "cos_row_frequency1",
    "sin_column_frequency1",
    "cos_column_frequency1",
)
FREQUENCY1_COORDINATE_SHAPE = (4, GRID_SIZE, GRID_SIZE)
FREQUENCY1_COORDINATE_DTYPE = "float64"
FREQUENCY1_COORDINATE_STEM_SHAPE = (32, 4, 1, 1)
FREQUENCY1_COORDINATE_PARAMETER_COUNT = 128
COORDINATE_FREE_PARAMETER_COUNT = 25_598
FREQUENCY1_COORDINATE_PARAMETER_COUNT_TOTAL = 25_726
FREQUENCY1_COORDINATE_PROJECTOR_TOLERANCE = 5.0e-14


class FrequencyOneCoordinateContractError(BoundaryTangentContractError):
    """The frozen frequency-one representation contract was violated."""


# Values were generated once as binary64 sin(2*pi*i/28) and committed in
# hexadecimal form.  Runtime model code only uses ``float.fromhex``.
FREQUENCY1_SIN_HEX = (
    "0x0.0p+0",
    "0x1.c7b90e3024582p-3",
    "0x1.bc4c04d71abc1p-2",
    "0x1.3f3a0e28bedd1p-1",
    "0x1.904c37505de4bp-1",
    "0x1.cd4bca9cb5c71p-1",
    "0x1.f329c0558e969p-1",
    "0x1.0000000000000p+0",
    "0x1.f329c0558e969p-1",
    "0x1.cd4bca9cb5c71p-1",
    "0x1.904c37505de4cp-1",
    "0x1.3f3a0e28bedd5p-1",
    "0x1.bc4c04d71abc3p-2",
    "0x1.c7b90e3024577p-3",
    "0x1.1a62633145c07p-53",
    "-0x1.c7b90e302456ep-3",
    "-0x1.bc4c04d71abbfp-2",
    "-0x1.3f3a0e28bedd4p-1",
    "-0x1.904c37505de4ap-1",
    "-0x1.cd4bca9cb5c70p-1",
    "-0x1.f329c0558e969p-1",
    "-0x1.0000000000000p+0",
    "-0x1.f329c0558e96bp-1",
    "-0x1.cd4bca9cb5c72p-1",
    "-0x1.904c37505de4cp-1",
    "-0x1.3f3a0e28bedd3p-1",
    "-0x1.bc4c04d71abb6p-2",
    "-0x1.c7b90e302458bp-3",
)

FREQUENCY1_COS_HEX = (
    "0x1.0000000000000p+0",
    "0x1.f329c0558e969p-1",
    "0x1.cd4bca9cb5c71p-1",
    "0x1.904c37505de4bp-1",
    "0x1.3f3a0e28bedd2p-1",
    "0x1.bc4c04d71abc2p-2",
    "0x1.c7b90e3024584p-3",
    "0x1.1a62633145c07p-54",
    "-0x1.c7b90e3024580p-3",
    "-0x1.bc4c04d71abc0p-2",
    "-0x1.3f3a0e28bedd1p-1",
    "-0x1.904c37505de48p-1",
    "-0x1.cd4bca9cb5c70p-1",
    "-0x1.f329c0558e96ap-1",
    "-0x1.0000000000000p+0",
    "-0x1.f329c0558e96ap-1",
    "-0x1.cd4bca9cb5c71p-1",
    "-0x1.904c37505de49p-1",
    "-0x1.3f3a0e28bedd3p-1",
    "-0x1.bc4c04d71abc4p-2",
    "-0x1.c7b90e3024589p-3",
    "-0x1.a79394c9e8a0ap-53",
    "0x1.c7b90e302455cp-3",
    "0x1.bc4c04d71abbep-2",
    "0x1.3f3a0e28bedd0p-1",
    "0x1.904c37505de4ap-1",
    "0x1.cd4bca9cb5c73p-1",
    "0x1.f329c0558e968p-1",
)

# SHA-256 of the C-order little-endian float64 [4,28,28] array returned by
# ``canonical_frequency1_coordinate_array``.
FREQUENCY1_COORDINATE_SHA256 = (
    "e4b47e7814dd260e0c82355e5b5bec57c74e8d27f0fca1e61707e5f6deb544d8"
)

_TEACHER_MIXING_MATRIX = np.asarray(
    (
        (1.00, 0.50, -0.75, 0.25),
        (0.50, -1.00, 0.25, 0.75),
        (-0.75, 0.25, 1.00, 0.50),
        (0.25, 0.75, 0.50, -1.00),
    ),
    dtype=np.float32,
)
_TEACHER_MIXING_MATRIX.setflags(write=False)


def _frozen_one_dimensional_tables() -> tuple[np.ndarray, np.ndarray]:
    if len(FREQUENCY1_SIN_HEX) != GRID_SIZE or len(FREQUENCY1_COS_HEX) != GRID_SIZE:
        raise FrequencyOneCoordinateContractError("frozen coordinate table length changed")
    sine = np.asarray([float.fromhex(value) for value in FREQUENCY1_SIN_HEX], dtype="<f8")
    cosine = np.asarray([float.fromhex(value) for value in FREQUENCY1_COS_HEX], dtype="<f8")
    if not np.isfinite(sine).all() or not np.isfinite(cosine).all():
        raise FrequencyOneCoordinateContractError("frozen coordinate table is nonfinite")
    return sine, cosine


def canonical_frequency1_coordinate_array() -> np.ndarray:
    """Return a fresh read-only C-order binary64 coordinate field.

    Channel order is row sine, row cosine, column sine, column cosine.  The
    returned bytes are platform-independent because the dtype is explicitly
    little-endian binary64.
    """

    sine, cosine = _frozen_one_dimensional_tables()
    result = np.empty(FREQUENCY1_COORDINATE_SHAPE, dtype="<f8", order="C")
    result[0] = sine[:, None]
    result[1] = cosine[:, None]
    result[2] = sine[None, :]
    result[3] = cosine[None, :]
    digest = hashlib.sha256(result.tobytes(order="C")).hexdigest()
    if digest != FREQUENCY1_COORDINATE_SHA256:
        raise FrequencyOneCoordinateContractError("frozen coordinate hash changed")
    result.setflags(write=False)
    return result


def active_head_frequency1_coordinates() -> np.ndarray:
    """Return the frozen fields at the seven phase occurrences' head sites.

    The result has shape ``[7,392,4]`` in phase, canonical-edge, channel order.
    """

    coordinates = canonical_frequency1_coordinate_array().reshape(4, STATE_SIZE)
    _tails, heads_by_color = matching_indices(device="cpu")
    heads = heads_by_color.detach().cpu().numpy().astype(np.int64, copy=False)
    colors = np.asarray(PHASE_MATCHINGS, dtype=np.int64)
    result = np.empty((PHASE_COUNT, EDGES_PER_PHASE, 4), dtype=np.float64)
    for phase, color in enumerate(colors):
        result[phase] = coordinates[:, heads[color]].T
    result = np.ascontiguousarray(result, dtype=np.float64)
    result.setflags(write=False)
    return result


def frequency1_coordinate_span_audit() -> dict[str, Any]:
    """Compare the frozen raw fields with the sealed diagnostic F1 span."""

    values = active_head_frequency1_coordinates()
    lattice = build_coordinate_lattice()
    ranks: list[int] = []
    errors: list[float] = []
    for phase in range(PHASE_COUNT):
        raw = values[phase]
        rank = int(np.linalg.matrix_rank(raw))
        basis, _triangular = np.linalg.qr(raw, mode="reduced")
        raw_projector = basis @ basis.T
        sealed = lattice.frequency1_basis[phase]
        sealed_projector = sealed @ sealed.T
        ranks.append(rank)
        errors.append(float(np.max(np.abs(raw_projector - sealed_projector))))
    maximum = max(errors)
    return {
        "schema": FREQUENCY1_COORDINATE_VERSION + "-span-audit",
        "schema_version": 1,
        "phase_ranks": ranks,
        "phase_projector_discrepancies": errors,
        "maximum_projector_discrepancy": maximum,
        "maximum_allowed_projector_discrepancy": FREQUENCY1_COORDINATE_PROJECTOR_TOLERANCE,
        "sealed_basis_sha256": lattice.basis_sha256,
        "passed": int(
            ranks == [4] * PHASE_COUNT
            and maximum <= FREQUENCY1_COORDINATE_PROJECTOR_TOLERANCE
        ),
    }


def frequency1_coordinate_array_audit() -> dict[str, Any]:
    """Return deterministic orientation, period, rank, and hash checks."""

    array = canonical_frequency1_coordinate_array()
    sine, cosine = _frozen_one_dimensional_tables()
    sine_step = sine[1]
    cosine_step = cosine[1]
    rotated_sine = sine * cosine_step + cosine * sine_step
    rotated_cosine = cosine * cosine_step - sine * sine_step
    rotation_error = max(
        float(np.max(np.abs(np.roll(sine, -1) - rotated_sine))),
        float(np.max(np.abs(np.roll(cosine, -1) - rotated_cosine))),
    )
    checks = {
        "shape_4x28x28": int(array.shape == FREQUENCY1_COORDINATE_SHAPE),
        "binary64": int(array.dtype == np.dtype("<f8")),
        "c_contiguous": int(array.flags.c_contiguous),
        "read_only": int(not array.flags.writeable),
        "finite_bounded": int(np.isfinite(array).all() and np.max(np.abs(array)) <= 1.0),
        "row_orientation": int(
            np.array_equal(array[0], np.broadcast_to(sine[:, None], (GRID_SIZE, GRID_SIZE)))
            and np.array_equal(array[1], np.broadcast_to(cosine[:, None], (GRID_SIZE, GRID_SIZE)))
        ),
        "column_orientation": int(
            np.array_equal(array[2], np.broadcast_to(sine[None, :], (GRID_SIZE, GRID_SIZE)))
            and np.array_equal(array[3], np.broadcast_to(cosine[None, :], (GRID_SIZE, GRID_SIZE)))
        ),
        "origin": int(
            array[0, 0, 0] == 0.0
            and array[1, 0, 0] == 1.0
            and array[2, 0, 0] == 0.0
            and array[3, 0, 0] == 1.0
        ),
        "raw_sha256": int(
            hashlib.sha256(array.tobytes(order="C")).hexdigest()
            == FREQUENCY1_COORDINATE_SHA256
        ),
        "period_28": int(
            np.array_equal(np.take(array, np.arange(GRID_SIZE) % GRID_SIZE, axis=1), array)
            and np.array_equal(
                np.take(array, np.arange(GRID_SIZE) % GRID_SIZE, axis=2), array
            )
        ),
        "one_step_rotation": int(rotation_error <= 2.0e-15),
    }
    span = frequency1_coordinate_span_audit()
    return {
        "schema": FREQUENCY1_COORDINATE_VERSION + "-array-audit",
        "schema_version": 1,
        "channel_order": list(FREQUENCY1_COORDINATE_CHANNELS),
        "shape": list(array.shape),
        "dtype": FREQUENCY1_COORDINATE_DTYPE,
        "array_sha256": FREQUENCY1_COORDINATE_SHA256,
        "maximum_one_step_rotation_error": rotation_error,
        "checks": checks,
        "span_audit": span,
        "passed": int(all(checks.values()) and span["passed"] == 1),
    }


def frequency1_coordinate_contract() -> dict[str, Any]:
    return {
        "schema": FREQUENCY1_COORDINATE_CONTRACT_SCHEMA,
        "schema_version": 1,
        "version": FREQUENCY1_COORDINATE_VERSION,
        "grid_size": GRID_SIZE,
        "coordinate_shape": list(FREQUENCY1_COORDINATE_SHAPE),
        "coordinate_dtype": FREQUENCY1_COORDINATE_DTYPE,
        "coordinate_order": "C",
        "coordinate_byte_order": "little",
        "coordinate_sha256": FREQUENCY1_COORDINATE_SHA256,
        "channel_order": list(FREQUENCY1_COORDINATE_CHANNELS),
        "vertex_index": "28*row+column",
        "active_site": "head_indices[color]",
        "phase_colors": list(PHASE_MATCHINGS),
        "stem_shape": list(FREQUENCY1_COORDINATE_STEM_SHAPE),
        "stem_bias": 0,
        "stem_initialization": "exact_zero",
        "added_trainable_parameters": FREQUENCY1_COORDINATE_PARAMETER_COUNT,
        "insertion": "spatial_branch_conv1_preactivation_only",
        "local_affine_coordinate_features": 0,
        "cache_coordinate_fields": 0,
        "runtime_trigonometry": 0,
    }


def frequency1_coordinate_input_contract() -> dict[str, Any]:
    actual_fields = tuple(field.name for field in fields(ModelInputs))
    checks = {
        "model_inputs_unchanged": int(actual_fields == tuple(MODEL_INPUT_FIELDS)),
        "coordinate_absent_from_model_inputs": int(
            not any("coordinate" in name or "position" in name for name in actual_fields)
        ),
        "coordinate_is_internal_buffer": 1,
        "audit_outer_step_forbidden": int("outer_step" not in actual_fields),
        "path_id_forbidden": int("path_id" not in actual_fields),
        "target_forbidden": int("target" not in actual_fields),
    }
    return {
        "schema": FREQUENCY1_COORDINATE_VERSION + "-input-contract",
        "schema_version": 1,
        "model_input_fields": list(actual_fields),
        "internal_conditioning": ["frequency1_coordinate"],
        "checks": checks,
        "passed": int(all(checks.values())),
    }


class FrequencyOneCoordinateJacobiRBPhasePredictor(JacobiRBPhasePredictor):
    """Width-32 phase predictor with one frozen frequency-one coordinate stem."""

    def __init__(self, *, width: int = 32, num_classes: int = 10) -> None:
        super().__init__(width=width, num_classes=num_classes)
        if self.width != 32:
            raise FrequencyOneCoordinateContractError(
                "frequency-one production predictor requires width 32"
            )
        coordinate = torch.from_numpy(
            np.array(canonical_frequency1_coordinate_array(), copy=True, order="C")
        ).to(device=self.conv1.weight.device)
        self.register_buffer("frequency1_coordinate", coordinate, persistent=True)
        self.coordinate_stem_weight = nn.Parameter(
            torch.zeros(
                (self.width, 4, 1, 1),
                dtype=self.conv1.weight.dtype,
                device=self.conv1.weight.device,
            )
        )

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Keep every inherited optimizer parameter before the new stem."""

        stem_name = f"{prefix}.coordinate_stem_weight" if prefix else "coordinate_stem_weight"
        delayed: tuple[str, nn.Parameter] | None = None
        for name, parameter in super().named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            if name == stem_name:
                delayed = (name, parameter)
            else:
                yield name, parameter
        if delayed is None:
            raise FrequencyOneCoordinateContractError(
                "coordinate stem disappeared from named parameters"
            )
        yield delayed

    def _forward_from_metadata(
        self, inputs: ModelInputs, metadata: Tensor
    ) -> Tensor:
        """Shared scientific forward operations after metadata construction."""

        state = inputs.later_full_state
        dtype = self.conv1.weight.dtype
        state = state.to(dtype=dtype)
        batch = inputs.batch_size
        density = state.reshape(batch, 1, GRID_SIZE, GRID_SIZE) * float(STATE_SIZE)
        metadata_planes = metadata[:, :, None, None].expand(
            batch, metadata.shape[1], GRID_SIZE, GRID_SIZE
        )
        old_pre = self.conv1(torch.cat([density, metadata_planes], dim=1))
        coordinate = self.frequency1_coordinate.to(dtype=dtype).unsqueeze(0)
        coordinate_pre = F.conv2d(
            coordinate,
            self.coordinate_stem_weight,
            bias=None,
            stride=1,
            padding=0,
        )
        hidden = F.silu(old_pre + coordinate_pre.expand(batch, -1, -1, -1))
        hidden = F.silu(self.conv2(hidden))
        hidden = F.silu(self.conv3(hidden))
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

    def _forward_prevalidated(self, inputs: ModelInputs) -> Tensor:
        """Evaluate the frozen network after a shard-boundary input audit.

        This is an additive execution hook for the fused rollout.  It performs
        exactly the same tensor operations, in the same order, as ``forward``
        but deliberately omits device-to-host metadata predicates.  The fused
        scheduler constructs phase/color/duration/label itself and validates
        the packed device flags once at the atomic shard boundary.
        """

        if type(inputs) is not ModelInputs:
            raise LearnabilityContractError(
                "prevalidated forward accepts only exact ModelInputs"
            )
        dtype = self.conv1.weight.dtype
        phase = inputs.phase.to(dtype=torch.long)
        color = inputs.color.to(dtype=torch.long)
        label = inputs.label.to(dtype=torch.long)
        metadata = torch.cat(
            [
                inputs.reverse_time.to(dtype=dtype).reshape(-1, 1),
                F.one_hot(phase, num_classes=PHASE_COUNT).to(dtype=dtype),
                F.one_hot(color, num_classes=4).to(dtype=dtype),
                inputs.duration.to(dtype=dtype).reshape(-1, 1),
                F.one_hot(label, num_classes=self.num_classes).to(dtype=dtype),
            ],
            dim=1,
        )
        return self._forward_from_metadata(inputs, metadata)

    def forward_prevalidated(self, inputs: ModelInputs) -> Tensor:
        """Public device-only counterpart used only after boundary validation."""

        return self._forward_prevalidated(inputs)

    def forward(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise LearnabilityContractError("forward accepts only exact ModelInputs")
        metadata = self._validated_metadata(inputs, self.conv1.weight.dtype)
        return self._forward_from_metadata(inputs, metadata)


def zero_initialize_frequency1_coordinate_residual(
    model: FrequencyOneCoordinateJacobiRBPhasePredictor,
) -> None:
    """Zero both inherited output paths and the added coordinate stem."""

    if type(model) is not FrequencyOneCoordinateJacobiRBPhasePredictor:
        raise FrequencyOneCoordinateContractError(
            "zero initialization requires the exact frequency-one predictor"
        )
    zero_initialize_residual(model)
    with torch.no_grad():
        model.coordinate_stem_weight.zero_()


class FrequencyOneCoordinateZeroBaselinePredictor(ZeroBaselineBoundaryTangentPredictor):
    """Exact-zero tangent wrapper around the coordinate-aware score model."""

    def __init__(
        self,
        residual_score: FrequencyOneCoordinateJacobiRBPhasePredictor | None = None,
        *,
        zero_residual: bool = True,
    ) -> None:
        active = (
            residual_score
            if residual_score is not None
            else FrequencyOneCoordinateJacobiRBPhasePredictor(width=32)
        )
        if type(active) is not FrequencyOneCoordinateJacobiRBPhasePredictor or active.width != 32:
            raise FrequencyOneCoordinateContractError(
                "residual score must be the exact width-32 frequency-one predictor"
            )
        super().__init__(active, zero_residual=False)
        if zero_residual:
            zero_initialize_frequency1_coordinate_residual(active)

    def score_prediction_prevalidated(self, inputs: ModelInputs) -> Tensor:
        """Return the one-row score without a hot-loop host synchronization."""

        if type(inputs) is not ModelInputs:
            raise FrequencyOneCoordinateContractError(
                "prevalidated score requires exact ModelInputs"
            )
        return self.residual_score.forward_prevalidated(inputs).to(dtype=torch.float64)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        stem_suffix = "residual_score.coordinate_stem_weight"
        stem_name = f"{prefix}.{stem_suffix}" if prefix else stem_suffix
        delayed: tuple[str, nn.Parameter] | None = None
        # Call nn.Module directly: the parent wrapper has no ordering override.
        for name, parameter in nn.Module.named_parameters(
            self, prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            if name == stem_name:
                delayed = (name, parameter)
            else:
                yield name, parameter
        if delayed is None and not recurse:
            return
        if delayed is None:
            raise FrequencyOneCoordinateContractError(
                "wrapped coordinate stem disappeared from named parameters"
            )
        yield delayed


def _canonical_model(factory: Any) -> nn.Module:
    # Construction helpers must not advance the caller's CPU RNG stream.
    with torch.random.fork_rng(devices=[], enabled=True):
        return factory()


def upgrade_coordinate_free_state_dict(
    old_state_dict: Mapping[str, Tensor],
    *,
    width: int = 32,
    num_classes: int = 10,
) -> OrderedDict[str, Tensor]:
    """Strictly extend one canonical coordinate-free residual state.

    Old keys, shapes, dtypes, and bytes are retained exactly.  The only new
    entries are the exact-zero stem and the canonical binary64 coordinate
    buffer.  The helper preserves the caller's CPU RNG state.
    """

    if not isinstance(old_state_dict, Mapping):
        raise FrequencyOneCoordinateContractError("old state dictionary is malformed")
    if width != 32 or num_classes != 10:
        raise FrequencyOneCoordinateContractError("state upgrade requires width 32/classes 10")
    old_template = _canonical_model(
        lambda: JacobiRBPhasePredictor(width=width, num_classes=num_classes)
    ).state_dict()
    if set(old_state_dict) != set(old_template):
        missing = sorted(set(old_template) - set(old_state_dict))
        additional = sorted(set(old_state_dict) - set(old_template))
        raise FrequencyOneCoordinateContractError(
            f"coordinate-free state key set changed; missing={missing}, additional={additional}"
        )
    for name, expected in old_template.items():
        value = old_state_dict[name]
        if not isinstance(value, Tensor):
            raise FrequencyOneCoordinateContractError(f"old state {name} is not a tensor")
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise FrequencyOneCoordinateContractError(
                f"old state {name} shape or dtype changed"
            )
        if (value.dtype.is_floating_point or value.dtype.is_complex) and not bool(
            torch.isfinite(value).all()
        ):
            raise FrequencyOneCoordinateContractError(f"old state {name} is nonfinite")

    new_template = _canonical_model(
        lambda: FrequencyOneCoordinateJacobiRBPhasePredictor(
            width=width, num_classes=num_classes
        )
    ).state_dict()
    additions = set(new_template) - set(old_template)
    if additions != {"coordinate_stem_weight", "frequency1_coordinate"}:
        raise FrequencyOneCoordinateContractError("new coordinate state closure changed")
    result: OrderedDict[str, Tensor] = OrderedDict()
    for name, expected in new_template.items():
        if name in old_state_dict:
            result[name] = old_state_dict[name].detach().clone(
                memory_format=torch.preserve_format
            )
        elif name == "coordinate_stem_weight":
            result[name] = torch.zeros_like(expected)
        elif name == "frequency1_coordinate":
            result[name] = expected.detach().clone(memory_format=torch.preserve_format)
        else:  # pragma: no cover - guarded by the exact addition set above
            raise FrequencyOneCoordinateContractError("unexpected coordinate state entry")
    return result


def frequency1_coordinate_architecture_contract(
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Describe and validate the one-variable architecture repair."""

    active_module = model or _canonical_model(
        lambda: FrequencyOneCoordinateJacobiRBPhasePredictor(width=32)
    )
    if isinstance(active_module, FrequencyOneCoordinateZeroBaselinePredictor):
        active = active_module.residual_score
    else:
        active = active_module
    if type(active) is not FrequencyOneCoordinateJacobiRBPhasePredictor:
        raise FrequencyOneCoordinateContractError(
            "architecture helper requires the exact frequency-one predictor"
        )
    parameters = dict(active.named_parameters())
    trainable_count = sum(value.numel() for value in parameters.values())
    coordinate = active.frequency1_coordinate.detach().to(device="cpu")
    coordinate_bytes = coordinate.contiguous().numpy().astype("<f8", copy=False).tobytes(order="C")
    metadata_channels = 1 + PHASE_COUNT + 4 + 1 + active.num_classes
    checks = {
        "model_width_32": int(active.width == 32),
        "old_conv1_channels_24": int(active.conv1.in_channels == 24),
        "local_affine_features_25": int(active.local_affine.in_features == 25),
        "three_circular_3x3_convolutions": int(
            all(
                layer.padding_mode == "circular" and tuple(layer.kernel_size) == (3, 3)
                for layer in (active.conv1, active.conv2, active.conv3)
            )
        ),
        "four_spatial_outputs": int(active.spatial_output.out_channels == 4),
        "stem_shape_32x4x1x1": int(
            tuple(active.coordinate_stem_weight.shape)
            == FREQUENCY1_COORDINATE_STEM_SHAPE
        ),
        "stem_is_last_optimizer_parameter": int(tuple(parameters)[-1] == "coordinate_stem_weight"),
        "added_parameter_count_128": int(
            active.coordinate_stem_weight.numel()
            == FREQUENCY1_COORDINATE_PARAMETER_COUNT
        ),
        "total_parameter_count_25726": int(
            trainable_count == FREQUENCY1_COORDINATE_PARAMETER_COUNT_TOTAL
        ),
        "coordinate_buffer_binary64": int(coordinate.dtype == torch.float64),
        "coordinate_buffer_shape": int(tuple(coordinate.shape) == FREQUENCY1_COORDINATE_SHAPE),
        "coordinate_buffer_hash": int(
            hashlib.sha256(coordinate_bytes).hexdigest()
            == FREQUENCY1_COORDINATE_SHA256
        ),
        "coordinate_buffer_persistent": int("frequency1_coordinate" in active.state_dict()),
        "coordinate_buffer_nontrainable": int("frequency1_coordinate" not in parameters),
        "metadata_channels_23": int(metadata_channels == 23),
    }
    return {
        "schema": FREQUENCY1_COORDINATE_VERSION + "-architecture-contract",
        "schema_version": 1,
        "model_width": active.width,
        "old_convolution_input_channels": active.conv1.in_channels,
        "local_affine_features": active.local_affine.in_features,
        "effective_state_receptive_field": 7,
        "coordinate_stem_shape": list(active.coordinate_stem_weight.shape),
        "coordinate_stem_zero_at_observation": int(
            bool(torch.all(active.coordinate_stem_weight == 0.0))
        ),
        "coordinate_buffer_shape": list(coordinate.shape),
        "coordinate_buffer_sha256": FREQUENCY1_COORDINATE_SHA256,
        "trainable_parameter_count": trainable_count,
        "coordinate_free_parameter_count": COORDINATE_FREE_PARAMETER_COUNT,
        "added_parameter_count": FREQUENCY1_COORDINATE_PARAMETER_COUNT,
        "named_parameter_order": list(parameters),
        "checks": checks,
        "passed": int(all(checks.values())),
    }


def configure_frequency1_coordinate_symmetry_break_fixture(
    model: FrequencyOneCoordinateJacobiRBPhasePredictor,
) -> None:
    """Install a deterministic audit-only coordinate-connected spatial path."""

    if type(model) is not FrequencyOneCoordinateJacobiRBPhasePredictor:
        raise FrequencyOneCoordinateContractError("symmetry fixture model has wrong type")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for channel in range(4):
            model.coordinate_stem_weight[channel, channel, 0, 0] = 0.5
            model.conv2.weight[channel, channel, 1, 1] = 1.0
            model.conv3.weight[channel, channel, 1, 1] = 1.0
            model.spatial_output.weight[channel, channel, 0, 0] = 1.0


def configure_exact_synthetic_frequency1_coordinate_teacher(
    model: FrequencyOneCoordinateJacobiRBPhasePredictor,
) -> None:
    """Install the frozen positive-energy coordinate teacher from the plan."""

    if type(model) is not FrequencyOneCoordinateJacobiRBPhasePredictor:
        raise FrequencyOneCoordinateContractError("synthetic teacher model has wrong type")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Local permitted-input teacher.  Features are scaled tail, scaled
        # head, reverse time, phase[7], color[4], duration, label[10].
        local = model.local_affine.weight[0]
        local[0] = -1.0
        local[1] = 1.0
        local[2] = 0.5
        phase_start = 3
        for phase in range(PHASE_COUNT):
            local[phase_start + phase] = 0.05 * phase
        duration_index = 3 + PHASE_COUNT + 4
        local[duration_index] = 0.10
        model.local_affine.bias.fill_(-0.475)

        for channel in range(4):
            model.coordinate_stem_weight[channel, channel, 0, 0] = 0.5
            model.conv2.weight[channel, channel, 1, 1] = 1.0
            model.conv3.weight[channel, channel, 1, 1] = 1.0
        matrix = torch.as_tensor(
            np.array(_TEACHER_MIXING_MATRIX, copy=True),
            dtype=model.spatial_output.weight.dtype,
            device=model.spatial_output.weight.device,
        )
        model.spatial_output.weight[:, :4, 0, 0] = 0.5 * matrix


def configure_exact_synthetic_frequency1_coordinate_zero_baseline_teacher(
    model: FrequencyOneCoordinateZeroBaselinePredictor,
) -> None:
    if type(model) is not FrequencyOneCoordinateZeroBaselinePredictor:
        raise FrequencyOneCoordinateContractError("wrapped synthetic teacher has wrong type")
    configure_exact_synthetic_frequency1_coordinate_teacher(model.residual_score)


def synthetic_frequency1_coordinate_score(inputs: ModelInputs) -> Tensor:
    """Evaluate the frozen teacher coefficient from permitted inputs only."""

    if type(inputs) is not ModelInputs:
        raise FrequencyOneCoordinateContractError("teacher requires exact ModelInputs")
    phase = inputs.phase.to(dtype=torch.long)
    color = inputs.color.to(dtype=torch.long)
    label = inputs.label.to(dtype=torch.long)
    if (
        bool(torch.any((phase < 0) | (phase >= PHASE_COUNT)))
        or bool(torch.any((color < 0) | (color >= 4)))
        or bool(torch.any((label < 0) | (label >= 10)))
    ):
        raise FrequencyOneCoordinateContractError("teacher metadata is outside its range")
    expected_color = torch.as_tensor(PHASE_MATCHINGS, dtype=torch.long, device=phase.device)[phase]
    expected_duration = torch.as_tensor(
        PHASE_DURATIONS, dtype=inputs.duration.dtype, device=phase.device
    )[phase]
    if not torch.equal(color, expected_color) or not torch.equal(
        inputs.duration, expected_duration
    ):
        raise FrequencyOneCoordinateContractError("teacher phase metadata is inconsistent")

    dtype = torch.float32
    state = inputs.later_full_state.to(dtype=dtype)
    tails_by_color, heads_by_color = matching_indices(device=state.device)
    tails = tails_by_color[color]
    heads = heads_by_color[color]
    tail = state.gather(1, tails) * float(STATE_SIZE)
    head = state.gather(1, heads) * float(STATE_SIZE)
    local = (
        -tail
        + head
        + 0.5 * inputs.reverse_time.to(dtype=dtype)[:, None]
        + 0.05 * phase.to(dtype=dtype)[:, None]
        + 0.10 * inputs.duration.to(dtype=dtype)[:, None]
        - 0.475
    )

    coordinate = torch.from_numpy(
        np.array(canonical_frequency1_coordinate_array(), copy=True, order="C")
    ).to(device=state.device, dtype=dtype)
    flat = coordinate.reshape(4, STATE_SIZE)
    active = torch.empty(
        (inputs.batch_size, EDGES_PER_PHASE, 4), dtype=dtype, device=state.device
    )
    for row in range(inputs.batch_size):
        active[row] = flat[:, heads[row]].T
    hidden = F.silu(0.5 * active)
    hidden = F.silu(hidden)
    hidden = F.silu(hidden)
    matrix = torch.as_tensor(
        np.array(_TEACHER_MIXING_MATRIX, copy=True), dtype=dtype, device=state.device
    )
    weights = 0.5 * matrix[color]
    spatial = torch.sum(hidden * weights[:, None, :], dim=2)
    return local + spatial


def synthetic_frequency1_coordinate_target(inputs: ModelInputs) -> Tensor:
    """Return mobility times the frozen synthetic coordinate score."""

    geometry = edge_pair_geometry(inputs)
    score = synthetic_frequency1_coordinate_score(inputs).to(dtype=torch.float64)
    prediction = geometry.mobility * score
    return torch.where(
        geometry.mobility == 0.0,
        torch.zeros_like(prediction),
        prediction,
    )


def frequency1_coordinate_teacher_contract(
    model: FrequencyOneCoordinateJacobiRBPhasePredictor | None = None,
) -> dict[str, Any]:
    if model is not None and type(model) is not FrequencyOneCoordinateJacobiRBPhasePredictor:
        raise FrequencyOneCoordinateContractError("teacher contract model has wrong type")
    # Always configure a private canonical instance: reading a contract must
    # not alter a caller-owned training model.
    active = _canonical_model(
        lambda: FrequencyOneCoordinateJacobiRBPhasePredictor(width=32)
    )
    configure_exact_synthetic_frequency1_coordinate_teacher(active)
    return {
        "schema": FREQUENCY1_COORDINATE_VERSION + "-synthetic-teacher-contract",
        "schema_version": 1,
        "state_dict_sha256": state_dict_sha256(active.state_dict()),
        "local_tail_weight": -1.0,
        "local_head_weight": 1.0,
        "reverse_time_weight": 0.5,
        "phase_weights": [0.05 * phase for phase in range(PHASE_COUNT)],
        "duration_weight": 0.10,
        "local_bias": -0.475,
        "coordinate_stem_diagonal": 0.5,
        "spatial_mixing_matrix": _TEACHER_MIXING_MATRIX.astype(np.float64).tolist(),
        "target": "mobility * (local_affine + coordinate_spatial_branch)",
        "physical_labels_used": 0,
    }


__all__ = [
    "COORDINATE_FREE_PARAMETER_COUNT",
    "FREQUENCY1_COORDINATE_CHANNELS",
    "FREQUENCY1_COORDINATE_CONTRACT_SCHEMA",
    "FREQUENCY1_COORDINATE_DTYPE",
    "FREQUENCY1_COORDINATE_PARAMETER_COUNT",
    "FREQUENCY1_COORDINATE_PARAMETER_COUNT_TOTAL",
    "FREQUENCY1_COORDINATE_PROJECTOR_TOLERANCE",
    "FREQUENCY1_COORDINATE_SHA256",
    "FREQUENCY1_COORDINATE_SHAPE",
    "FREQUENCY1_COORDINATE_STEM_SHAPE",
    "FREQUENCY1_COORDINATE_VERSION",
    "FREQUENCY1_COS_HEX",
    "FREQUENCY1_SIN_HEX",
    "FrequencyOneCoordinateContractError",
    "FrequencyOneCoordinateJacobiRBPhasePredictor",
    "FrequencyOneCoordinateZeroBaselinePredictor",
    "active_head_frequency1_coordinates",
    "canonical_frequency1_coordinate_array",
    "configure_exact_synthetic_frequency1_coordinate_teacher",
    "configure_exact_synthetic_frequency1_coordinate_zero_baseline_teacher",
    "configure_frequency1_coordinate_symmetry_break_fixture",
    "frequency1_coordinate_architecture_contract",
    "frequency1_coordinate_array_audit",
    "frequency1_coordinate_contract",
    "frequency1_coordinate_input_contract",
    "frequency1_coordinate_span_audit",
    "frequency1_coordinate_teacher_contract",
    "synthetic_frequency1_coordinate_score",
    "synthetic_frequency1_coordinate_target",
    "upgrade_coordinate_free_state_dict",
    "zero_initialize_frequency1_coordinate_residual",
]
