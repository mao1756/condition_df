from __future__ import annotations

import torch
from torch import nn

from mnist.d0_jacobi_rb_boundary_tangent import edge_pair_geometry
from mnist.d0_jacobi_rb_global_large import (
    LARGE_GLOBAL_PARAMETER_COUNT,
    LargeEulerianJacobiDDPMModel,
    large_global_architecture_contract,
    large_global_parameter_count,
)
from mnist.d0_jacobi_rb_learnability import (
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    ModelInputs,
)


def _inputs(batch: int = 2) -> ModelInputs:
    state = torch.arange(1, batch * 784 + 1, dtype=torch.float32).reshape(batch, 784)
    state /= state.sum(dim=1, keepdim=True)
    phase = torch.tensor([0, 3][:batch], dtype=torch.long)
    return ModelInputs(
        later_full_state=state,
        reverse_time=torch.tensor([0.25, 0.75][:batch], dtype=torch.float32),
        phase=phase,
        color=torch.tensor([PHASE_MATCHINGS[int(value)] for value in phase]),
        duration=torch.tensor(
            [PHASE_DURATIONS[int(value)] for value in phase], dtype=torch.float32
        ),
        label=torch.tensor([5, 5][:batch], dtype=torch.long),
    )


def test_large_architecture_contract_and_parameter_count() -> None:
    assert large_global_parameter_count() == LARGE_GLOBAL_PARAMETER_COUNT == 2_390_174
    contract = large_global_architecture_contract()
    assert contract["width"] == 128
    assert contract["residual_blocks"] == 8
    assert contract["dilations"] == (1, 2, 4, 8, 1, 2, 4, 8)
    assert contract["normalization_layers"] == 0
    assert contract["dropout_layers"] == 0
    assert contract["pooling_layers"] == 0


def test_large_controller_starts_at_exact_zero_and_applies_mobility_once() -> None:
    model = LargeEulerianJacobiDDPMModel()
    inputs = _inputs()
    q = model.predictor.score_prediction(inputs)
    m = model(inputs)
    assert q.shape == m.shape == (2, 392)
    assert torch.equal(q, torch.zeros_like(q))
    assert torch.equal(m, torch.zeros_like(m))

    with torch.no_grad():
        model.predictor.residual_score.local_affine.bias.fill_(2.0)
    q = model.predictor.score_prediction(inputs)
    mobility = edge_pair_geometry(inputs).mobility
    m = model(inputs)
    assert torch.equal(q, torch.full_like(q, 2.0))
    assert torch.allclose(m, 2.0 * mobility, atol=0.0, rtol=0.0)


def test_every_nonpointwise_spatial_convolution_is_circular() -> None:
    model = LargeEulerianJacobiDDPMModel()
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            assert module.padding_mode == "circular"
