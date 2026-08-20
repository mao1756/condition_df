from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mnist.ddpm_eulerian_adapter import (
    DDPMEulerianAdapter,
    DDPMEulerianAdapterConfig,
    FROZEN_FLUX_PROJECTION,
    FROZEN_LATENT_POLICY,
    FROZEN_TIME_MAP,
    ddpm_denoised_mass,
    desired_mass_velocity,
    eulerian_flux_step_with_standard_normal_torch,
    load_bound_ddpm_generator,
    mass_to_ddpm_model_space,
    remaining_time_to_ddpm_timestep,
    velocity_to_periodic_controller_flux,
)
from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    eulerian_flux_step_torch,
    flux_divergence_torch,
    free_drift_flux_torch,
    natural_horizon,
)
from mnist.pixel_ddpm import (
    ClassConditionalUNet28,
    DDPMSchedule,
    make_linear_ddpm_schedule,
    predict_x0_from_epsilon,
    q_sample,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_bound_run(root: Path) -> str:
    (root / "training").mkdir(parents=True)
    (root / "controls").mkdir()
    model = ClassConditionalUNet28()
    checkpoint = root / "training" / "selected_checkpoint.pt"
    payload = {
        "state_dict": model.state_dict(),
        "selected_epoch": 7,
        "validation_mse": 0.125,
    }
    torch.save(payload, checkpoint)
    checkpoint_sha256 = _sha256(checkpoint)
    selection = {
        "checkpoint_sha256": checkpoint_sha256,
        "completed_epochs": 9,
        "learned_epsilon_rms": 0.75,
        "selected_epoch": 7,
        "validation_mse": 0.125,
        "zero_predictor_mse": 1.0,
    }
    (root / "training" / "selection.json").write_text(
        json.dumps(selection, sort_keys=True), encoding="utf-8"
    )
    config = {
        "schema": "pixel-ddpm-calibration-v1",
        "schedule": {"steps": 1000, "beta_start": 1e-4, "beta_end": 2e-2},
        "model": {"kind": "unet28", "parameter_count": 1_378_593},
    }
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    schedule = make_linear_ddpm_schedule()
    np.savez(
        root / "controls" / "schedule.npz",
        betas=schedule.betas.numpy(),
        alphas=schedule.alphas.numpy(),
        alpha_bars=schedule.alpha_bars.numpy(),
    )
    return checkpoint_sha256


class _ScaledEpsilon(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.125))
        self.calls = 0

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        offset = (timesteps.to(images.dtype) + labels.to(images.dtype)).reshape(-1, 1, 1, 1)
        return self.scale * images + 1e-4 * offset


def _adapter() -> tuple[DDPMEulerianAdapter, _ScaledEpsilon, DirectFluxMNISTConfig]:
    model = _ScaledEpsilon()
    config = DirectFluxMNISTConfig(
        grid_size=28,
        mass_floor=1e-8,
        free_weight=0.015,
        noise_weight=0.002,
    )
    adapter_config = DDPMEulerianAdapterConfig(
        num_ddpm_steps=8,
        beta_start=1e-4,
        beta_end=2e-2,
        mass_scale_numerator=25_471,
        mass_scale_denominator=255,
        min_tau_fraction=0.03,
        mass_floor=1e-8,
        time_map=FROZEN_TIME_MAP,
        latent_policy=FROZEN_LATENT_POLICY,
        flux_projection=FROZEN_FLUX_PROJECTION,
    )
    adapter = DDPMEulerianAdapter(
        model,
        make_linear_ddpm_schedule(8),
        config,
        adapter_config,
    )
    return adapter, model, config


def test_mass_rendering_is_exact_continuous_and_reports_clipping() -> None:
    masses = torch.zeros((2, 784), dtype=torch.float64)
    masses[0, 0] = 1.0
    masses[1] = 1.0 / 784.0
    scale = 25_471.0 / 255.0
    rendered, saturation = mass_to_ddpm_model_space(masses, scale=scale)

    expected = 2.0 * (masses * scale).clamp(0.0, 1.0) - 1.0
    assert rendered.shape == (2, 1, 28, 28)
    assert rendered.dtype == torch.float64
    assert torch.equal(rendered.reshape(2, -1), expected)
    assert torch.equal(saturation, torch.tensor([1.0 / 784.0, 0.0], dtype=torch.float64))

    with pytest.raises(ValueError, match="sum to one"):
        mass_to_ddpm_model_space(masses * 0.5, scale=scale)


def test_remaining_time_map_has_exact_endpoints_and_frozen_anchors() -> None:
    horizon = 2.5
    remaining = torch.tensor([horizon, 0.75 * horizon, 0.5 * horizon, 0.25 * horizon, 0.0])
    actual = remaining_time_to_ddpm_timestep(remaining, horizon=horizon, num_steps=1000)
    assert torch.equal(actual, torch.tensor([999, 749, 500, 250, 0]))
    clipped = remaining_time_to_ddpm_timestep(
        torch.tensor([-1.0, 10.0]), horizon=horizon, num_steps=1000
    )
    assert torch.equal(clipped, torch.tensor([0, 999]))


def test_forward_noising_epsilon_x0_and_mass_conversion_reuse_ddpm_math() -> None:
    schedule = make_linear_ddpm_schedule(8)
    masses = torch.arange(1, 785, dtype=torch.float32).reshape(1, -1)
    masses = masses / masses.sum(dim=1, keepdim=True)
    model_space, _ = mass_to_ddpm_model_space(masses, scale=25_471.0 / 255.0)
    latent = torch.linspace(-1.0, 1.0, 784).reshape(1, 1, 28, 28)
    timestep = torch.tensor([5])
    model = _ScaledEpsilon().eval()

    predicted_mass, telemetry = ddpm_denoised_mass(
        model,
        model_space,
        torch.tensor([3]),
        timestep,
        latent,
        schedule,
        mass_floor=1e-8,
    )
    expected_noisy = q_sample(model_space, timestep, latent, schedule)
    expected_epsilon = model(expected_noisy, timestep, torch.tensor([3]))
    expected_x0 = predict_x0_from_epsilon(expected_noisy, timestep, expected_epsilon, schedule)
    expected_positive = (expected_x0 + 1.0) * 0.5 + 1e-8
    expected_mass = expected_positive.flatten(1) / expected_positive.flatten(1).sum(dim=1, keepdim=True)

    assert torch.equal(telemetry["noisy_input"], expected_noisy)
    assert torch.equal(telemetry["epsilon_hat"], expected_epsilon)
    assert torch.equal(telemetry["x0_hat"], expected_x0)
    assert torch.equal(predicted_mass, expected_mass)
    assert bool((predicted_mass > 0).all())
    assert torch.allclose(predicted_mass.sum(dim=1), torch.ones(1), rtol=0.0, atol=1e-7)


def test_remaining_time_velocity_is_exact_and_endpoint_bounded() -> None:
    current = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]])
    target = torch.tensor([[0.3, 0.3, 0.4], [0.25, 0.25, 0.5]])
    remaining = torch.tensor([0.5, 0.0])
    velocity = desired_mass_velocity(
        current,
        target,
        remaining_time=remaining,
        minimum_time=0.1,
    )
    expected = (target - current) / torch.tensor([[0.5], [0.1]])
    expected = expected - expected.mean(dim=1, keepdim=True)
    assert torch.equal(velocity, expected)
    assert torch.allclose(velocity.sum(dim=1), torch.zeros(2), atol=1e-6, rtol=0.0)


def test_periodic_flux_projection_and_free_drift_cancellation() -> None:
    torch.manual_seed(11)
    current = torch.rand((2, 784)) + 0.1
    current = current / current.sum(dim=1, keepdim=True)
    target = torch.rand((2, 784)) + 0.1
    target = target / target.sum(dim=1, keepdim=True)
    velocity = desired_mass_velocity(
        current,
        target,
        remaining_time=torch.tensor([0.5, 0.25]),
        minimum_time=0.01,
    )
    config = DirectFluxMNISTConfig(grid_size=28, mass_floor=1e-8, free_weight=0.015)
    controller, residual = velocity_to_periodic_controller_flux(
        velocity,
        current,
        config,
        free_weight=0.015,
    )
    total = controller + 0.015 * free_drift_flux_torch(current, config)
    reconstructed = flux_divergence_torch(total).reshape_as(velocity)
    assert controller.shape == (2, 2, 28, 28)
    assert controller.dtype == current.dtype
    assert residual.dtype == torch.float64
    assert float(residual.max()) <= 2e-4
    assert torch.allclose(reconstructed, velocity, rtol=0.0, atol=2e-4)


def test_periodic_flux_projection_meets_gate_at_minimum_late_time_scale() -> None:
    torch.manual_seed(29)
    current = torch.rand((16, 784), dtype=torch.float32) + 0.01
    current = current / current.sum(dim=1, keepdim=True)
    target = torch.rand((16, 784), dtype=torch.float32) + 0.01
    target = target / target.sum(dim=1, keepdim=True)
    config = DirectFluxMNISTConfig(grid_size=28, mass_floor=1e-8, free_weight=0.015)
    minimum_time = 0.03 * natural_horizon(config)
    velocity = desired_mass_velocity(
        current,
        target,
        remaining_time=torch.zeros(16),
        minimum_time=minimum_time,
    )
    controller, residual = velocity_to_periodic_controller_flux(
        velocity,
        current,
        config,
        free_weight=0.015,
    )
    total = controller + 0.015 * free_drift_flux_torch(current, config)
    reconstructed = flux_divergence_torch(total).reshape_as(velocity)

    assert controller.dtype == current.dtype
    assert residual.dtype == torch.float64
    assert float(residual.max()) <= 2e-4
    assert torch.allclose(reconstructed, velocity, rtol=0.0, atol=2e-4)


def test_adapter_is_target_free_immutable_persistent_and_batch_local() -> None:
    adapter, model, config = _adapter()
    torch.manual_seed(17)
    masses = torch.rand((3, 784)) + 0.1
    masses = masses / masses.sum(dim=1, keepdim=True)
    labels = torch.tensor([2, 5, 9])
    remaining = natural_horizon(config) * torch.tensor([1.0, 0.5, 0.25])
    latent = torch.randn((3, 1, 28, 28))
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    training_before = model.training

    first = adapter.predict(masses, labels, remaining, latent)
    second = adapter.predict(masses, labels, remaining, latent)
    permutation = torch.tensor([2, 0, 1])
    permuted = adapter.predict(
        masses[permutation], labels[permutation], remaining[permutation], latent[permutation]
    )
    chunked = [
        adapter.predict(masses[index : index + 1], labels[index : index + 1], remaining[index : index + 1], latent[index : index + 1])
        for index in range(3)
    ]

    for field in first.__dataclass_fields__:
        first_value = getattr(first, field)
        assert torch.equal(first_value, getattr(second, field))
        assert torch.equal(first_value, getattr(permuted, field)[torch.argsort(permutation)])
        chunked_value = torch.cat([getattr(item, field) for item in chunked], dim=0)
        if first_value.is_floating_point():
            assert torch.allclose(first_value, chunked_value, rtol=1e-12, atol=1e-12)
        else:
            assert torch.equal(first_value, chunked_value)
    assert training_before is False and model.training is False
    assert all(parameter.requires_grad is False and parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, state_before[name]) for name, value in model.state_dict().items())
    assert "target" not in inspect.signature(adapter.predict).parameters
    with pytest.raises(TypeError):
        adapter.predict(masses, labels, remaining, latent, target=masses)  # type: ignore[call-arg]


def test_adapter_returns_plan_telemetry_and_zero_sum_mass_contract() -> None:
    adapter, model, config = _adapter()
    masses = torch.full((2, 784), 1.0 / 784.0)
    step = adapter.predict(
        masses,
        torch.tensor([0, 1]),
        torch.full((2,), natural_horizon(config)),
        torch.zeros((2, 1, 28, 28)),
    )
    assert set(step.__dataclass_fields__) == {
        "conditioning_flux",
        "predicted_mass",
        "desired_velocity",
        "ddpm_timestep",
        "epsilon_rms",
        "score_rms",
        "render_saturation_fraction",
        "x0_saturation_fraction",
        "divergence_residual_linf",
        "current_render_mean",
        "current_render_std",
        "current_render_min",
        "current_render_max",
        "noisy_input_mean",
        "noisy_input_std",
        "noisy_input_min",
        "noisy_input_max",
        "predicted_x0_rms",
    }
    assert torch.equal(step.ddpm_timestep, torch.tensor([7, 7]))
    assert model.calls == 1
    assert torch.allclose(step.predicted_mass.sum(dim=1), torch.ones(2), atol=1e-6, rtol=0.0)
    assert torch.allclose(step.desired_velocity.sum(dim=1), torch.zeros(2), atol=2e-4, rtol=0.0)
    assert float(step.divergence_residual_linf.max()) <= 2e-4
    for name in (
        "current_render_mean",
        "current_render_std",
        "current_render_min",
        "current_render_max",
        "noisy_input_mean",
        "noisy_input_std",
        "noisy_input_min",
        "noisy_input_max",
        "predicted_x0_rms",
    ):
        assert getattr(step, name).shape == (2,)
        assert bool(torch.isfinite(getattr(step, name)).all())


def test_bound_loader_records_and_freezes_checkpoint_schedule_and_model(tmp_path: Path) -> None:
    expected = _write_synthetic_bound_run(tmp_path)
    bound = load_bound_ddpm_generator(tmp_path, device=torch.device("cpu"), expected_sha256=expected)
    assert bound.checkpoint_sha256 == expected
    assert bound.selection_metadata["selected_epoch"] == 7
    assert bound.schedule.num_steps == 1000
    assert bound.parameter_count == 1_378_593
    assert len(bound.model_state_sha256) == 64
    assert bound.schedule_path.name == "schedule.npz"
    assert bound.schedule_bytes == bound.schedule_path.stat().st_size
    assert bound.schedule_sha256 == _sha256(bound.schedule_path)
    assert bound.model.training is False
    assert all(parameter.requires_grad is False for parameter in bound.model.parameters())
    with pytest.raises(TypeError):
        bound.selection_metadata["selected_epoch"] = 8  # type: ignore[index]


@pytest.mark.parametrize("fault", ["checkpoint", "selection", "config_schedule", "parameter_count", "schedule_receipt", "payload"])
def test_bound_loader_fails_closed_on_authority_mismatch(tmp_path: Path, fault: str) -> None:
    expected = _write_synthetic_bound_run(tmp_path)
    checkpoint = tmp_path / "training" / "selected_checkpoint.pt"
    selection_path = tmp_path / "training" / "selection.json"
    config_path = tmp_path / "config.json"
    schedule_path = tmp_path / "controls" / "schedule.npz"

    if fault == "checkpoint":
        expected = "0" * 64
    elif fault == "selection":
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["checkpoint_sha256"] = "1" * 64
        selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")
    elif fault == "config_schedule":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["schedule"]["steps"] = 999
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    elif fault == "parameter_count":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["model"]["parameter_count"] -= 1
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    elif fault == "schedule_receipt":
        schedule = make_linear_ddpm_schedule()
        betas = schedule.betas.numpy().copy()
        betas[500] = np.nextafter(betas[500], np.float32(1.0))
        np.savez(
            schedule_path,
            betas=betas,
            alphas=schedule.alphas.numpy(),
            alpha_bars=schedule.alpha_bars.numpy(),
        )
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["unexpected"] = 1
        torch.save(payload, checkpoint)
        expected = _sha256(checkpoint)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["checkpoint_sha256"] = expected
        selection_path.write_text(json.dumps(selection, sort_keys=True), encoding="utf-8")

    with pytest.raises((RuntimeError, ValueError)):
        load_bound_ddpm_generator(tmp_path, device=torch.device("cpu"), expected_sha256=expected)


def test_adapter_config_rejects_unplanned_bridge_modes() -> None:
    with pytest.raises(ValueError, match="time_map"):
        DDPMEulerianAdapterConfig(time_map="tuned")
    with pytest.raises(ValueError, match="latent_policy"):
        DDPMEulerianAdapterConfig(latent_policy="resampled")
    with pytest.raises(ValueError, match="flux_projection"):
        DDPMEulerianAdapterConfig(flux_projection="raw_score")

    schedule = make_linear_ddpm_schedule()
    altered_bars = schedule.alpha_bars.clone()
    altered_bars[500] = torch.nextafter(altered_bars[500], torch.tensor(1.0))
    altered = DDPMSchedule(schedule.betas, schedule.alphas, altered_bars)
    with pytest.raises(ValueError, match="schedule"):
        DDPMEulerianAdapter(
            _ScaledEpsilon(),
            altered,
            DirectFluxMNISTConfig(grid_size=28, mass_floor=1e-8),
            DDPMEulerianAdapterConfig(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_adapter_accepts_canonical_cpu_schedule_moved_to_cuda() -> None:
    schedule = make_linear_ddpm_schedule().to(torch.device("cuda:0"))
    adapter = DDPMEulerianAdapter(
        _ScaledEpsilon().to(torch.device("cuda:0")),
        schedule,
        DirectFluxMNISTConfig(grid_size=28, mass_floor=1e-8),
        DDPMEulerianAdapterConfig(),
    )

    assert adapter.schedule is schedule


def test_adapter_local_supplied_noise_is_exact_and_rng_neutral() -> None:
    config = DirectFluxMNISTConfig(
        grid_size=8,
        mass_floor=1e-8,
        limiter_fraction=1.0,
        free_weight=0.0,
        noise_weight=0.05,
        learned_weight=0.0,
    )
    states = torch.full((2, 64), 1.0 / 64.0)
    flux = torch.zeros((2, 2, 8, 8))
    standard_normal = torch.linspace(-0.5, 0.5, 256).reshape(2, 128)

    torch.manual_seed(917)
    before = torch.random.get_rng_state().clone()
    first = eulerian_flux_step_with_standard_normal_torch(
        states,
        flux,
        1e-5,
        config,
        standard_normal_flat=standard_normal,
    )
    after = torch.random.get_rng_state().clone()
    torch.manual_seed(3)
    second = eulerian_flux_step_with_standard_normal_torch(
        states,
        flux,
        1e-5,
        config,
        standard_normal_flat=standard_normal,
    )

    assert torch.equal(before, after)
    assert torch.equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_adapter_local_supplied_noise_uses_horizontal_then_vertical_order() -> None:
    config = DirectFluxMNISTConfig(
        grid_size=8,
        mass_floor=1e-8,
        limiter_fraction=1.0,
        free_weight=0.0,
        noise_weight=0.01,
        learned_weight=0.0,
    )
    states = torch.full((1, 64), 1.0 / 64.0)
    standard_normal = torch.zeros((1, 128))
    standard_normal[0, 0] = 1.0
    result, _, _ = eulerian_flux_step_with_standard_normal_torch(
        states,
        torch.zeros((1, 2, 8, 8)),
        1e-6,
        config,
        standard_normal_flat=standard_normal,
    )

    assert result[0, 0] < states[0, 0]
    assert result[0, 1] > states[0, 1]
    unchanged = torch.ones(64, dtype=torch.bool)
    unchanged[[0, 1]] = False
    assert torch.equal(result[0, unchanged], states[0, unchanged])


def test_adapter_local_none_delegates_to_frozen_core_exactly() -> None:
    config = DirectFluxMNISTConfig(grid_size=8, noise_weight=0.01)
    states = torch.full((2, 64), 1.0 / 64.0)
    flux = torch.zeros((2, 2, 8, 8))

    torch.manual_seed(101)
    expected = eulerian_flux_step_torch(states, flux, 1e-6, config)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(101)
    observed = eulerian_flux_step_with_standard_normal_torch(
        states,
        flux,
        1e-6,
        config,
        standard_normal_flat=None,
    )
    observed_rng = torch.random.get_rng_state().clone()

    assert torch.equal(expected[0], observed[0])
    assert expected[1:] == observed[1:]
    assert torch.equal(expected_rng, observed_rng)


def test_adapter_local_supplied_noise_validation_batch_order_and_zero_reference() -> None:
    config = DirectFluxMNISTConfig(grid_size=8, noise_weight=0.01)
    states = torch.tensor(
        [[1.0 + index for index in range(64)], [65.0 + index for index in range(64)]],
        dtype=torch.float32,
    )
    states = states / states.sum(dim=1, keepdim=True)
    flux = torch.zeros((2, 2, 8, 8))
    standard_normal = torch.linspace(-1.0, 1.0, 256).reshape(2, 128)
    direct = eulerian_flux_step_with_standard_normal_torch(
        states,
        flux,
        1e-6,
        config,
        standard_normal_flat=standard_normal,
    )[0]
    permutation = torch.tensor([1, 0])
    permuted = eulerian_flux_step_with_standard_normal_torch(
        states[permutation],
        flux[permutation],
        1e-6,
        config,
        standard_normal_flat=standard_normal[permutation],
    )[0]
    assert torch.equal(direct, permuted[permutation])

    zero_noise = eulerian_flux_step_with_standard_normal_torch(
        states,
        flux,
        1e-6,
        config,
        standard_normal_flat=torch.zeros_like(standard_normal),
    )
    deterministic = eulerian_flux_step_torch(
        states,
        flux,
        1e-6,
        config,
        deterministic=True,
    )
    assert torch.equal(zero_noise[0], deterministic[0])
    assert zero_noise[1:] == deterministic[1:]

    for invalid in (torch.zeros((2, 127)), torch.full((2, 128), float("nan"))):
        with pytest.raises(ValueError):
            eulerian_flux_step_with_standard_normal_torch(
                states,
                flux,
                1e-6,
                config,
                standard_normal_flat=invalid,
            )
    with pytest.raises(ValueError):
        eulerian_flux_step_with_standard_normal_torch(
            states,
            flux,
            1e-6,
            config,
            deterministic=True,
            standard_normal_flat=standard_normal,
        )
