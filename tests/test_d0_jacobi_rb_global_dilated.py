from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_absolute_coordinate import (
    edge_translation_permutation,
    translate_model_inputs,
)
from mnist.d0_jacobi_rb_boundary_tangent import edge_pair_geometry
from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate import (
    FREQUENCY1_COORDINATE_SHA256,
    FrequencyOneCoordinateJacobiRBPhasePredictor,
)
from mnist.d0_jacobi_rb_global_dilated import (
    GLOBAL_DILATED_DILATIONS,
    GLOBAL_DILATED_PARAMETER_COUNT,
    GLOBAL_DILATED_RECEPTIVE_FIELD,
    GlobalDilatedContractError,
    GlobalDilatedJacobiRBPhasePredictor,
    GlobalDilatedZeroBaselinePredictor,
    global_dilated_architecture_contract,
    zero_initialize_global_dilated_residual,
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
    call_model,
)


def _inputs(*, batch: int = 7, phase: int | None = None) -> ModelInputs:
    phases = (
        torch.arange(batch, dtype=torch.long) % PHASE_COUNT
        if phase is None
        else torch.full((batch,), phase, dtype=torch.long)
    )
    colors = torch.as_tensor(PHASE_MATCHINGS, dtype=torch.long)[phases]
    durations = torch.as_tensor(PHASE_DURATIONS, dtype=torch.float64)[phases]
    state = torch.arange(
        1, batch * STATE_SIZE + 1, dtype=torch.float64
    ).reshape(batch, STATE_SIZE)
    state /= torch.sum(state, dim=1, keepdim=True)
    return ModelInputs(
        later_full_state=state,
        reverse_time=torch.linspace(0.1, 0.9, batch, dtype=torch.float64),
        phase=phases,
        color=colors,
        duration=durations,
        label=torch.full((batch,), 3, dtype=torch.long),
    )


class _FixedSpatialField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        field = torch.arange(
            4 * GRID_SIZE * GRID_SIZE, dtype=torch.float32
        ).reshape(1, 4, GRID_SIZE, GRID_SIZE)
        self.register_buffer("field", field)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.field.expand(hidden.shape[0], -1, -1, -1)


def test_architecture_contract_and_exact_parameter_count() -> None:
    before_rng = torch.random.get_rng_state().clone()
    contract = global_dilated_architecture_contract()
    after_rng = torch.random.get_rng_state().clone()
    model = GlobalDilatedJacobiRBPhasePredictor()
    convolutions = (model.conv1, model.conv2, model.conv3, model.conv4)

    assert torch.equal(before_rng, after_rng)
    assert contract["passed"] == 1
    assert contract["trainable_parameter_count"] == GLOBAL_DILATED_PARAMETER_COUNT
    assert sum(parameter.numel() for parameter in model.parameters()) == 34_974
    assert tuple(contract["dilations"]) == GLOBAL_DILATED_DILATIONS == (1, 2, 4, 8)
    assert contract["receptive_field"] == GLOBAL_DILATED_RECEPTIVE_FIELD == 31
    assert contract["contiguous_offset_range"] == [-15, 15]
    assert contract["spans_28x28_torus"] == 1
    assert all(layer.padding_mode == "circular" for layer in convolutions)
    assert all(tuple(layer.kernel_size) == (3, 3) for layer in convolutions)
    assert tuple(model.spatial_output.kernel_size) == (1, 1)
    assert model.spatial_output.out_channels == 4
    assert model.local_affine.in_features == 25
    assert tuple(model.coordinate_stem_weight.shape) == (32, 4, 1, 1)
    assert contract["coordinate_sha256"] == FREQUENCY1_COORDINATE_SHA256
    with pytest.raises(GlobalDilatedContractError, match="width 32 and 10 classes"):
        GlobalDilatedJacobiRBPhasePredictor(width=16)
    with pytest.raises(GlobalDilatedContractError, match="width 32 and 10 classes"):
        GlobalDilatedJacobiRBPhasePredictor(num_classes=5)


def test_exact_input_firewall_shape_finiteness_and_prevalidated_agreement() -> None:
    torch.manual_seed(261_401)
    model = GlobalDilatedJacobiRBPhasePredictor().eval()
    inputs = _inputs()
    with torch.no_grad():
        model.spatial_output.weight.copy_(
            torch.linspace(-0.1, 0.1, model.spatial_output.weight.numel()).reshape_as(
                model.spatial_output.weight
            )
        )
        model.spatial_output.bias.copy_(
            torch.linspace(-0.05, 0.05, model.spatial_output.bias.numel())
        )
        model.local_affine.weight.copy_(
            torch.linspace(-0.2, 0.2, model.local_affine.weight.numel()).reshape_as(
                model.local_affine.weight
            )
        )
        model.local_affine.bias.fill_(0.125)
        ordinary = model(inputs)
        prevalidated = model.forward_prevalidated(inputs)

    assert ordinary.shape == (7, EDGES_PER_PHASE)
    assert torch.isfinite(ordinary).all()
    assert torch.equal(ordinary, prevalidated)
    with pytest.raises(LearnabilityContractError, match="ModelInputs"):
        model(object())  # type: ignore[arg-type]

    invalid_phase = ModelInputs(
        later_full_state=inputs.later_full_state,
        reverse_time=inputs.reverse_time,
        phase=torch.full_like(inputs.phase, PHASE_COUNT),
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )
    with pytest.raises(LearnabilityContractError, match="outside"):
        model(invalid_phase)


def test_deterministic_far_pixel_at_torus_offset_14_14_changes_output() -> None:
    model = GlobalDilatedJacobiRBPhasePredictor().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # The selected output depends on state offset 0 + 2 + 4 + 8 = 14
        # in both axes.  SiLU is monotone on this positive fixture.
        model.conv1.weight[0, 0, 1, 1] = 1.0
        model.conv2.weight[0, 0, 2, 2] = 1.0
        model.conv3.weight[0, 0, 2, 2] = 1.0
        model.conv4.weight[0, 0, 2, 2] = 1.0

    inputs = _inputs(batch=1, phase=0)
    color = int(inputs.color.item())
    head = int(model.head_indices[color, 0].item())
    head_row, head_column = divmod(head, GRID_SIZE)
    far_row = (head_row + 14) % GRID_SIZE
    far_column = (head_column + 14) % GRID_SIZE
    far_vertex = far_row * GRID_SIZE + far_column
    baseline_state = torch.zeros_like(inputs.later_full_state)
    perturbed_state = baseline_state.clone()
    perturbed_state[0, far_vertex] = 1.0 / STATE_SIZE
    baseline = ModelInputs(
        later_full_state=baseline_state,
        reverse_time=inputs.reverse_time,
        phase=inputs.phase,
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )
    perturbed = ModelInputs(
        later_full_state=perturbed_state,
        reverse_time=inputs.reverse_time,
        phase=inputs.phase,
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )

    with torch.no_grad():
        model.spatial_output.weight[color, 0, 0, 0] = 1.0
        baseline_output = model(baseline)
        perturbed_output = model(perturbed)
    assert baseline_output[0, 0] == 0.0
    assert perturbed_output[0, 0] > baseline_output[0, 0]
    assert torch.count_nonzero(perturbed_output - baseline_output) > 0


def test_zero_coordinate_stem_retains_matching_aware_torus_translation() -> None:
    torch.manual_seed(261_402)
    model = GlobalDilatedJacobiRBPhasePredictor().eval()
    inputs = _inputs(batch=7)
    with torch.no_grad():
        assert torch.count_nonzero(model.coordinate_stem_weight) == 0
        model.spatial_output.weight.copy_(
            torch.linspace(-0.25, 0.25, model.spatial_output.weight.numel()).reshape_as(
                model.spatial_output.weight
            )
        )
        model.spatial_output.bias.zero_()
        model.local_affine.weight.zero_()
        model.local_affine.bias.zero_()
        original = model(inputs)
        shifted = model(
            translate_model_inputs(inputs, row_shift=2, column_shift=2)
        )

    expected = torch.empty_like(original)
    for row, color in enumerate(inputs.color.tolist()):
        permutation = edge_translation_permutation(int(color), 2, 2)
        expected[row] = original[row].index_select(
            0, torch.as_tensor(np.array(permutation, copy=True), dtype=torch.long)
        )
    torch.testing.assert_close(shifted, expected, rtol=0.0, atol=2.0e-8)


def test_output_gather_and_local_branch_match_v4_bit_for_bit() -> None:
    inputs = _inputs()
    v4 = FrequencyOneCoordinateJacobiRBPhasePredictor().eval()
    global_model = GlobalDilatedJacobiRBPhasePredictor().eval()
    fixed_spatial = _FixedSpatialField()
    v4.spatial_output = fixed_spatial
    global_model.spatial_output = _FixedSpatialField()
    with torch.no_grad():
        local_weight = torch.linspace(-0.3, 0.3, 25).reshape(1, 25)
        v4.local_affine.weight.copy_(local_weight)
        global_model.local_affine.weight.copy_(local_weight)
        v4.local_affine.bias.fill_(0.0625)
        global_model.local_affine.bias.fill_(0.0625)
        observed_v4 = v4(inputs)
        observed_global = global_model(inputs)
    assert torch.equal(observed_global, observed_v4)


def test_zero_initialization_and_mobility_once_wrapper_contract() -> None:
    model = GlobalDilatedJacobiRBPhasePredictor()
    hidden_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("conv")
    }
    assert any(torch.count_nonzero(value) > 0 for value in hidden_before.values())
    assert torch.count_nonzero(model.spatial_output.weight) == 0
    assert torch.count_nonzero(model.spatial_output.bias) == 0
    assert torch.count_nonzero(model.local_affine.weight) == 0
    assert torch.count_nonzero(model.local_affine.bias) == 0
    assert torch.count_nonzero(model.coordinate_stem_weight) == 0
    assert torch.equal(call_model(model, _inputs()), torch.zeros(7, EDGES_PER_PHASE))

    with torch.no_grad():
        model.spatial_output.weight.fill_(1.0)
        model.spatial_output.bias.fill_(1.0)
        model.local_affine.weight.fill_(1.0)
        model.local_affine.bias.fill_(1.0)
        model.coordinate_stem_weight.fill_(1.0)
    zero_initialize_global_dilated_residual(model)
    assert all(
        torch.equal(hidden_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in hidden_before
    )
    assert torch.equal(call_model(model, _inputs()), torch.zeros(7, EDGES_PER_PHASE))
    with pytest.raises(GlobalDilatedContractError, match="exact"):
        zero_initialize_global_dilated_residual(  # type: ignore[arg-type]
            FrequencyOneCoordinateJacobiRBPhasePredictor()
        )

    with torch.no_grad():
        model.local_affine.bias.fill_(2.0)
    wrapper = GlobalDilatedZeroBaselinePredictor(model, zero_residual=False).eval()
    uniform_inputs = _inputs(batch=1, phase=0)
    uniform_inputs = ModelInputs(
        later_full_state=torch.full_like(
            uniform_inputs.later_full_state, 1.0 / STATE_SIZE
        ),
        reverse_time=uniform_inputs.reverse_time,
        phase=uniform_inputs.phase,
        color=uniform_inputs.color,
        duration=uniform_inputs.duration,
        label=uniform_inputs.label,
    )
    score = wrapper.score_prediction(uniform_inputs)
    prevalidated_score = wrapper.score_prediction_prevalidated(uniform_inputs)
    mobility = edge_pair_geometry(uniform_inputs).mobility
    prediction = wrapper(uniform_inputs)
    assert torch.equal(score, torch.full_like(score, 2.0))
    assert torch.equal(score, prevalidated_score)
    assert torch.equal(mobility, torch.full_like(mobility, 0.25))
    assert torch.equal(prediction, mobility * score)
    assert torch.equal(prediction, torch.full_like(prediction, 0.5))

    boundary_state = torch.zeros_like(uniform_inputs.later_full_state)
    boundary_state[0, 0] = 1.0
    boundary_inputs = ModelInputs(
        later_full_state=boundary_state,
        reverse_time=uniform_inputs.reverse_time,
        phase=uniform_inputs.phase,
        color=uniform_inputs.color,
        duration=uniform_inputs.duration,
        label=uniform_inputs.label,
    )
    assert torch.equal(
        wrapper(boundary_inputs), torch.zeros(1, EDGES_PER_PHASE, dtype=torch.float64)
    )


def test_wrapped_training_objective_gradient_contains_exactly_one_mobility() -> None:
    model = GlobalDilatedJacobiRBPhasePredictor()
    wrapper = GlobalDilatedZeroBaselinePredictor(model, zero_residual=False)
    inputs = _inputs(batch=1, phase=0)
    inputs = ModelInputs(
        later_full_state=torch.full_like(inputs.later_full_state, 1.0 / STATE_SIZE),
        reverse_time=inputs.reverse_time,
        phase=inputs.phase,
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )
    bar_z = torch.ones((1, EDGES_PER_PHASE), dtype=torch.float64)

    wrapper.zero_grad(set_to_none=True)
    m_theta = wrapper(inputs)
    loss = torch.mean((m_theta - bar_z).square())
    loss.backward()

    assert torch.equal(wrapper.score_prediction(inputs), torch.zeros_like(bar_z))
    assert torch.equal(m_theta, torch.zeros_like(bar_z))
    # d mean((mobility*q - 1)^2) / d bias at q=0 and mobility=1/4
    # is -2*(1/4) = -1/2.  Training on raw q would instead produce -2.
    assert model.local_affine.bias.grad is not None
    torch.testing.assert_close(
        model.local_affine.bias.grad,
        torch.tensor([-0.5], dtype=model.local_affine.bias.dtype),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_wrapper_rejects_non_global_residual_and_non_model_inputs() -> None:
    with pytest.raises(GlobalDilatedContractError, match="exact width-32"):
        GlobalDilatedZeroBaselinePredictor(  # type: ignore[arg-type]
            FrequencyOneCoordinateJacobiRBPhasePredictor()
        )
    wrapper = GlobalDilatedZeroBaselinePredictor()
    with pytest.raises(GlobalDilatedContractError, match="ModelInputs"):
        wrapper.score_prediction(object())  # type: ignore[arg-type]
    with pytest.raises(GlobalDilatedContractError, match="ModelInputs"):
        wrapper.score_prediction_prevalidated(object())  # type: ignore[arg-type]
    with pytest.raises(GlobalDilatedContractError, match="ModelInputs"):
        wrapper(object())  # type: ignore[arg-type]
