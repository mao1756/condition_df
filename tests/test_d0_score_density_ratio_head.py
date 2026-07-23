from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from mnist.d0_dirichlet_score import (
    edge_difference_channels,
    physical_flux_from_edge_score,
)
from mnist.d0_score_boundary_controls import D0BoundarySmoothPotentialUNet
from mnist.d0_score_density_ratio_head import (
    BODY_PARAMETER_GROUP_NAME,
    COORDINATE_CONJUGATE_ADAMW_VERSION,
    HEAD_PARAMETER_NAMES,
    NORMALIZED_HEAD_COORDINATE_VERSION,
    NORMALIZED_HEAD_MODEL_VERSION,
    NORMALIZED_HEAD_PARAMETER_GROUP_NAME,
    D0BoundarySmoothMeanHeadPotentialUNet,
    build_coordinate_conjugate_adamw,
    coordinate_conjugate_adamw_record,
    head_coordinate_factor,
    legacy_ema_state_to_normalized_head,
    legacy_gradient_dict_to_normalized_head,
    legacy_state_dict_to_normalized_head,
    normalized_ema_state_to_legacy_head,
    normalized_gradient_diagnostics,
    normalized_gradient_dict_to_legacy_head,
    normalized_state_dict_to_legacy_head,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    init_ema_state,
    natural_horizon,
    update_ema_state,
)


torch.set_num_threads(1)


def _config(grid_size: int = 4) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
        num_steps=8,
        source_lowfreq_size=2,
        ot_lowres_size=2,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-12,
        limiter_fraction=1.0,
        condition_on_source=False,
        flux_parameterization="edge",
    )


def _states(
    count: int = 4,
    grid_size: int = 4,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(260881)
    raw = torch.rand(
        (count, grid_size * grid_size), generator=generator, dtype=dtype
    ) + 0.2
    return (raw / raw.sum(dim=1, keepdim=True)).to(device)


def _equivalent_models(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[D0BoundarySmoothPotentialUNet, D0BoundarySmoothMeanHeadPotentialUNet]:
    torch.manual_seed(260882)
    config = _config()
    legacy = D0BoundarySmoothPotentialUNet(config, base_channels=2).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        legacy.out.weight.normal_(mean=0.0, std=0.025)
        legacy.out.bias.fill_(0.0075)
    normalized = D0BoundarySmoothMeanHeadPotentialUNet(
        config, base_channels=2
    ).to(device=device, dtype=dtype)
    normalized.load_state_dict(
        legacy_state_dict_to_normalized_head(legacy.state_dict(), config.grid_size),
        strict=True,
    )
    return legacy, normalized


def _assert_state_dict_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    atol: float = 2e-11,
    rtol: float = 2e-10,
) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol)


def test_normalized_head_versions_and_zero_initialization_are_frozen() -> None:
    model = D0BoundarySmoothMeanHeadPotentialUNet(_config(), base_channels=2)
    assert model.model_version == NORMALIZED_HEAD_MODEL_VERSION
    assert model.head_coordinate_version == NORMALIZED_HEAD_COORDINATE_VERSION
    assert COORDINATE_CONJUGATE_ADAMW_VERSION
    assert model.scalar_reduction == "spatial_mean"
    assert head_coordinate_factor(model) == 16
    assert torch.count_nonzero(model.out.weight) == 0
    assert torch.count_nonzero(model.out.bias) == 0


def test_state_and_gradient_coordinate_maps_round_trip_without_mutation() -> None:
    legacy, _ = _equivalent_models()
    original = copy.deepcopy(legacy.state_dict())
    normalized = legacy_state_dict_to_normalized_head(original, 4)
    round_trip = normalized_state_dict_to_legacy_head(normalized, 4)
    _assert_state_dict_close(round_trip, original, atol=0.0, rtol=0.0)
    _assert_state_dict_close(legacy.state_dict(), original, atol=0.0, rtol=0.0)
    for name in HEAD_PARAMETER_NAMES:
        torch.testing.assert_close(normalized[name], original[name] * 16.0)

    gradients = {
        name: torch.full_like(value, index + 1.0)
        for index, (name, value) in enumerate(legacy.named_parameters())
    }
    normalized_gradients = legacy_gradient_dict_to_normalized_head(gradients, 4)
    restored_gradients = normalized_gradient_dict_to_legacy_head(
        normalized_gradients, 4
    )
    _assert_state_dict_close(restored_gradients, gradients, atol=0.0, rtol=0.0)
    for name in HEAD_PARAMETER_NAMES:
        torch.testing.assert_close(
            normalized_gradients[name], gradients[name] / 16.0
        )

    normalized_ema = legacy_ema_state_to_normalized_head(original, 4)
    restored_ema = normalized_ema_state_to_legacy_head(normalized_ema, 4)
    _assert_state_dict_close(restored_ema, original, atol=0.0, rtol=0.0)


def test_logits_bce_state_gradients_edge_scores_and_flux_are_equivalent() -> None:
    legacy, normalized = _equivalent_models()
    states = _states()
    tau = torch.linspace(
        0.1 * natural_horizon(legacy.config),
        natural_horizon(legacy.config),
        states.shape[0],
        dtype=states.dtype,
    )
    labels = torch.tensor([3, 3, 3, 3], dtype=torch.long)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=states.dtype)

    legacy_states = states.detach().clone().requires_grad_(True)
    normalized_states = states.detach().clone().requires_grad_(True)
    legacy_logits = legacy(tau, legacy_states, labels)
    normalized_logits = normalized(tau, normalized_states, labels)
    torch.testing.assert_close(normalized_logits, legacy_logits, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        F.binary_cross_entropy_with_logits(normalized_logits, targets),
        F.binary_cross_entropy_with_logits(legacy_logits, targets),
        atol=2e-13,
        rtol=2e-13,
    )

    legacy_gradient = torch.autograd.grad(legacy_logits.sum(), legacy_states)[0]
    normalized_gradient = torch.autograd.grad(
        normalized_logits.sum(), normalized_states
    )[0]
    torch.testing.assert_close(
        normalized_gradient, legacy_gradient, atol=2e-11, rtol=2e-10
    )
    legacy_edge = edge_difference_channels(legacy_gradient, 4)
    normalized_edge = edge_difference_channels(normalized_gradient, 4)
    torch.testing.assert_close(normalized_edge, legacy_edge, atol=2e-11, rtol=2e-10)
    legacy_flux = physical_flux_from_edge_score(
        legacy_edge, states, legacy.config, time_change=1.0
    )
    normalized_flux = physical_flux_from_edge_score(
        normalized_edge, states, normalized.config, time_change=1.0
    )
    torch.testing.assert_close(
        normalized_flux, legacy_flux, atol=2e-11, rtol=2e-10
    )


def test_parameter_gradients_obey_exact_coordinate_chain_rule() -> None:
    legacy, normalized = _equivalent_models()
    states = _states()
    tau = torch.linspace(0.01, 0.1, states.shape[0], dtype=states.dtype)
    labels = torch.full((states.shape[0],), 3, dtype=torch.long)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=states.dtype)

    legacy.zero_grad(set_to_none=True)
    normalized.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(legacy(tau, states, labels), targets).backward()
    F.binary_cross_entropy_with_logits(
        normalized(tau, states, labels), targets
    ).backward()
    legacy_named = dict(legacy.named_parameters())
    normalized_named = dict(normalized.named_parameters())
    for name in legacy_named:
        assert legacy_named[name].grad is not None
        assert normalized_named[name].grad is not None
        expected = (
            legacy_named[name].grad / 16.0
            if name in HEAD_PARAMETER_NAMES
            else legacy_named[name].grad
        )
        torch.testing.assert_close(
            normalized_named[name].grad, expected, atol=2e-11, rtol=2e-9
        )

    diagnostics = normalized_gradient_diagnostics(normalized)
    assert diagnostics["finite"] == 1
    assert diagnostics["missing_gradient_count"] == 0
    legacy_sq = sum(
        parameter.grad.detach().double().square().sum()
        for parameter in legacy.parameters()
    )
    assert diagnostics["reconstructed_legacy_gradient_norm"] == pytest.approx(
        float(torch.sqrt(legacy_sq)), rel=2e-10, abs=2e-11
    )


def test_coordinate_conjugate_adamw_groups_are_named_and_scaled() -> None:
    _, model = _equivalent_models()
    optimizer = build_coordinate_conjugate_adamw(
        model,
        body_lr=3e-5,
        eps=1e-8,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {
        BODY_PARAMETER_GROUP_NAME,
        NORMALIZED_HEAD_PARAMETER_GROUP_NAME,
    }
    assert groups[BODY_PARAMETER_GROUP_NAME]["lr"] == pytest.approx(3e-5)
    assert groups[BODY_PARAMETER_GROUP_NAME]["eps"] == pytest.approx(1e-8)
    assert groups[BODY_PARAMETER_GROUP_NAME]["weight_decay"] == pytest.approx(1e-4)
    assert groups[NORMALIZED_HEAD_PARAMETER_GROUP_NAME]["lr"] == pytest.approx(
        16.0 * 3e-5
    )
    assert groups[NORMALIZED_HEAD_PARAMETER_GROUP_NAME]["eps"] == pytest.approx(
        1e-8 / 16.0
    )
    assert groups[NORMALIZED_HEAD_PARAMETER_GROUP_NAME][
        "weight_decay"
    ] == pytest.approx(1e-4 / 16.0)

    record = coordinate_conjugate_adamw_record(
        model, body_lr=3e-5, eps=1e-8, weight_decay=1e-4
    )
    assert record["version"] == COORDINATE_CONJUGATE_ADAMW_VERSION
    assert record["coordinate_factor"] == 16
    assert [group["name"] for group in record["groups"]] == [
        BODY_PARAMETER_GROUP_NAME,
        NORMALIZED_HEAD_PARAMETER_GROUP_NAME,
    ]
    assert record["groups"][1]["parameter_names"] == list(HEAD_PARAMETER_NAMES)


def test_multistep_adamw_and_ema_are_coordinate_conjugate() -> None:
    legacy, normalized = _equivalent_models()
    lr, eps, weight_decay = 7e-5, 3e-8, 2e-4
    betas = (0.85, 0.97)
    legacy_optimizer = torch.optim.AdamW(
        legacy.parameters(),
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
        betas=betas,
    )
    normalized_optimizer = build_coordinate_conjugate_adamw(
        normalized,
        body_lr=lr,
        eps=eps,
        weight_decay=weight_decay,
        betas=betas,
    )
    legacy_ema = init_ema_state(legacy)
    normalized_ema = init_ema_state(normalized)
    states = _states()
    tau = torch.linspace(0.01, 0.1, states.shape[0], dtype=states.dtype)
    labels = torch.full((states.shape[0],), 3, dtype=torch.long)

    for step in range(1, 5):
        targets = torch.tensor(
            [float(step % 2), float((step + 1) % 2), 0.0, 1.0],
            dtype=states.dtype,
        )
        legacy_optimizer.zero_grad(set_to_none=True)
        normalized_optimizer.zero_grad(set_to_none=True)
        F.binary_cross_entropy_with_logits(
            legacy(tau, states, labels), targets
        ).backward()
        F.binary_cross_entropy_with_logits(
            normalized(tau, states, labels), targets
        ).backward()
        legacy_optimizer.step()
        normalized_optimizer.step()
        update_ema_state(legacy_ema, legacy, 0.91)
        update_ema_state(normalized_ema, normalized, 0.91)

        _assert_state_dict_close(
            normalized_state_dict_to_legacy_head(normalized.state_dict(), 4),
            legacy.state_dict(),
        )
        _assert_state_dict_close(
            normalized_ema_state_to_legacy_head(normalized_ema, 4),
            legacy_ema,
        )


def test_equivalent_normalized_gradient_avoids_legacy_global_clipping_fixture() -> None:
    legacy, normalized = _equivalent_models()
    legacy.zero_grad(set_to_none=True)
    normalized.zero_grad(set_to_none=True)
    legacy_named = dict(legacy.named_parameters())
    normalized_named = dict(normalized.named_parameters())
    for name, parameter in legacy_named.items():
        parameter.grad = torch.zeros_like(parameter)
        normalized_named[name].grad = torch.zeros_like(normalized_named[name])
    for name in HEAD_PARAMETER_NAMES:
        legacy_named[name].grad.fill_(1.0)
        normalized_named[name].grad.copy_(legacy_named[name].grad / 16.0)

    diagnostics = normalized_gradient_diagnostics(normalized)
    legacy_norm = torch.nn.utils.clip_grad_norm_(legacy.parameters(), 1.0)
    normalized_norm = torch.nn.utils.clip_grad_norm_(normalized.parameters(), 1.0)
    assert float(legacy_norm) > 1.0
    assert float(normalized_norm) < 1.0
    assert diagnostics["normalized_gradient_norm"] < 1.0
    assert diagnostics["reconstructed_legacy_gradient_norm"] == pytest.approx(
        float(legacy_norm), rel=1e-12
    )
    assert diagnostics["reconstructed_legacy_head_squared_fraction"] == 1.0


def test_coordinate_helpers_reject_incompatible_inputs() -> None:
    legacy = D0BoundarySmoothPotentialUNet(_config(), base_channels=2)
    with pytest.raises(TypeError, match="normalized"):
        build_coordinate_conjugate_adamw(legacy, body_lr=1e-4)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="missing"):
        legacy_state_dict_to_normalized_head({"out.weight": torch.ones(1)}, 4)
    _, normalized = _equivalent_models()
    with pytest.raises(ValueError, match="body_lr"):
        build_coordinate_conjugate_adamw(normalized, body_lr=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float32_equivalence_meets_production_tolerance() -> None:
    legacy, normalized = _equivalent_models(dtype=torch.float32, device="cuda")
    states = _states(dtype=torch.float32, device="cuda")
    tau = torch.linspace(0.01, 0.1, states.shape[0], device="cuda")
    labels = torch.full((states.shape[0],), 3, device="cuda", dtype=torch.long)
    legacy_states = states.detach().clone().requires_grad_(True)
    normalized_states = states.detach().clone().requires_grad_(True)
    legacy_logits = legacy(tau, legacy_states, labels)
    normalized_logits = normalized(tau, normalized_states, labels)
    torch.testing.assert_close(normalized_logits, legacy_logits, atol=2e-6, rtol=2e-6)
    legacy_gradient = torch.autograd.grad(legacy_logits.sum(), legacy_states)[0]
    normalized_gradient = torch.autograd.grad(
        normalized_logits.sum(), normalized_states
    )[0]
    torch.testing.assert_close(
        normalized_gradient, legacy_gradient, atol=2e-6, rtol=2e-6
    )
