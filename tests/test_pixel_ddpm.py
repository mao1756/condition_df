from __future__ import annotations

import copy

import torch
from torch import nn

from mnist.pixel_ddpm import (
    ClassConditionalUNet28,
    ddpm_step_from_epsilon,
    epsilon_from_x0,
    epsilon_prediction_loss,
    make_linear_ddpm_schedule,
    predict_x0_from_epsilon,
    q_sample,
    sample_reverse,
    sinusoidal_timestep_embedding,
    update_ema_,
)


def test_linear_schedule_and_forward_marginal_are_exact() -> None:
    schedule = make_linear_ddpm_schedule(7, 1e-4, 2e-2, dtype=torch.float64)
    expected_betas = torch.linspace(1e-4, 2e-2, 7, dtype=torch.float64)
    expected_alphas = 1.0 - expected_betas
    assert schedule.num_steps == 7
    assert schedule.betas.shape == schedule.alphas.shape == schedule.alpha_bars.shape == (7,)
    assert torch.equal(schedule.betas, expected_betas)
    assert torch.equal(schedule.alphas, expected_alphas)
    assert torch.equal(schedule.alpha_bars, torch.cumprod(expected_alphas, dim=0))
    assert torch.all(schedule.alpha_bars > 0)
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])

    x_0 = torch.linspace(-0.8, 0.8, 18, dtype=torch.float64).reshape(2, 1, 3, 3)
    noise = torch.linspace(0.7, -0.7, 18, dtype=torch.float64).reshape_as(x_0)
    timesteps = torch.tensor([0, 6])
    alpha_bar = schedule.alpha_bars[timesteps, None, None, None]
    expected = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * noise
    actual = q_sample(x_0, timesteps, noise, schedule)
    assert torch.equal(actual, expected)
    assert torch.allclose(epsilon_from_x0(actual, x_0, timesteps, schedule), noise)
    assert torch.allclose(predict_x0_from_epsilon(actual, timesteps, noise, schedule), x_0)


def test_epsilon_posterior_step_matches_direct_posterior_mean() -> None:
    schedule = make_linear_ddpm_schedule(8, dtype=torch.float64)
    x_0 = torch.tensor(
        [[[[0.2, -0.4], [0.6, -0.1]]], [[[0.3, 0.1], [-0.5, 0.7]]]],
        dtype=torch.float64,
    )
    forward_noise = torch.tensor(
        [[[[0.1, -0.3], [0.4, 0.2]]], [[[-0.2, 0.5], [-0.1, 0.3]]]],
        dtype=torch.float64,
    )
    timesteps = torch.tensor([2, 7])
    x_t = q_sample(x_0, timesteps, forward_noise, schedule)
    epsilon_hat = epsilon_from_x0(x_t, x_0, timesteps, schedule)
    actual = ddpm_step_from_epsilon(
        x_t, timesteps, epsilon_hat, schedule, torch.zeros_like(x_t)
    )

    beta = schedule.betas[timesteps, None, None, None]
    alpha = schedule.alphas[timesteps, None, None, None]
    alpha_bar = schedule.alpha_bars[timesteps, None, None, None]
    previous = torch.cat((torch.ones(1, dtype=torch.float64), schedule.alpha_bars[:-1]))
    alpha_bar_previous = previous[timesteps, None, None, None]
    expected = (
        beta * torch.sqrt(alpha_bar_previous) / (1.0 - alpha_bar) * x_0
        + (1.0 - alpha_bar_previous)
        * torch.sqrt(alpha)
        / (1.0 - alpha_bar)
        * x_t
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_t_zero_ignores_noise() -> None:
    schedule = make_linear_ddpm_schedule(4, dtype=torch.float64)
    x_0 = torch.tensor([[[[-0.7, 0.2], [0.4, 0.9]]]], dtype=torch.float64)
    forward_noise = torch.tensor([[[[0.5, -0.4], [0.1, -0.2]]]], dtype=torch.float64)
    x_t = q_sample(x_0, 0, forward_noise, schedule)
    epsilon_hat = epsilon_from_x0(x_t, x_0, 0, schedule)
    first = ddpm_step_from_epsilon(x_t, 0, epsilon_hat, schedule, torch.ones_like(x_t))
    second = ddpm_step_from_epsilon(x_t, 0, epsilon_hat, schedule, -torch.ones_like(x_t))
    assert torch.allclose(first, x_0, atol=1e-12, rtol=0)
    assert torch.equal(first, second)
    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()
    assert torch.equal(ddpm_step_from_epsilon(x_t, 0, epsilon_hat, schedule), first)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_analytic_oracle_reconstructs_short_and_complete_horizons_with_shared_noise() -> None:
    schedule = make_linear_ddpm_schedule(1000, dtype=torch.float64)
    x_0 = torch.linspace(-0.75, 0.75, 32, dtype=torch.float64).reshape(2, 1, 4, 4)
    forward_noise = torch.linspace(0.5, -0.5, 32, dtype=torch.float64).reshape_as(x_0)

    for start_t in (99, 999):
        state = q_sample(x_0, start_t, forward_noise, schedule)
        zero_state = state.clone()
        oracle_state = state.clone()
        generator = torch.Generator().manual_seed(1200 + start_t)
        for timestep in range(start_t, -1, -1):
            t_batch = torch.full((x_0.shape[0],), timestep, dtype=torch.long)
            shared_noise = (
                torch.randn(x_0.shape, generator=generator, dtype=x_0.dtype)
                if timestep > 0
                else torch.zeros_like(x_0)
            )
            zero_state = ddpm_step_from_epsilon(
                zero_state, t_batch, torch.zeros_like(zero_state), schedule, shared_noise
            )
            oracle_epsilon = epsilon_from_x0(oracle_state, x_0, t_batch, schedule)
            oracle_state = ddpm_step_from_epsilon(
                oracle_state, t_batch, oracle_epsilon, schedule, shared_noise
            )
        assert torch.max((oracle_state - x_0).square()).item() <= 1e-12
        assert not torch.equal(zero_state, oracle_state)


def test_frozen_unet_shape_gradients_label_conditioning_and_parameter_count(tmp_path) -> None:
    torch.manual_seed(17)
    model = ClassConditionalUNet28()
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 1_378_593

    images = torch.randn(2, 1, 28, 28, requires_grad=True)
    timesteps = torch.tensor([5, 900])
    labels = torch.tensor([1, 8])
    output = model(images, timesteps, labels)
    assert output.shape == images.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)

    with torch.no_grad():
        fixed_image = images[:1].detach()
        fixed_time = timesteps[:1]
        label_zero = model(fixed_image, fixed_time, torch.tensor([0]))
        label_one = model(fixed_image, fixed_time, torch.tensor([1]))
    assert not torch.equal(label_zero, label_one)

    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = ClassConditionalUNet28()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(original, loaded)


def test_timestep_embedding_loss_and_ema_update() -> None:
    embedding = sinusoidal_timestep_embedding(torch.tensor([0, 3, 7]), 128)
    assert embedding.shape == (3, 128)
    assert torch.isfinite(embedding).all()

    schedule = make_linear_ddpm_schedule(4)
    x_0 = torch.zeros(2, 1, 3, 3)
    noise = torch.arange(18, dtype=torch.float32).reshape_as(x_0) / 10.0
    timesteps = torch.tensor([0, 3])
    labels = torch.tensor([2, 4])

    def zero_model(images: torch.Tensor, _: torch.Tensor, __: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(images)

    loss = epsilon_prediction_loss(zero_model, x_0, timesteps, labels, noise, schedule)
    assert torch.equal(loss, noise.square().mean())

    model = nn.Linear(3, 2)
    ema = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in ema.parameters():
            parameter.zero_()
        for parameter in model.parameters():
            parameter.fill_(2.0)
    update_ema_(ema, model, 0.25)
    for parameter in ema.parameters():
        assert torch.equal(parameter, torch.full_like(parameter, 1.5))


class _ZeroEpsilon(nn.Module):
    def forward(self, images: torch.Tensor, _: torch.Tensor, __: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(images)


def test_reverse_sampling_is_seeded_and_saves_exact_sparse_anchors() -> None:
    schedule = make_linear_ddpm_schedule(6, dtype=torch.float64)
    initial = torch.zeros(2, 1, 4, 4, dtype=torch.float64)
    labels = torch.tensor([0, 1])
    model = _ZeroEpsilon()

    def run(seed: int) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        return sample_reverse(
            model,
            labels,
            initial,
            schedule,
            generator=torch.Generator().manual_seed(seed),
            anchor_steps=(0, 2, 6),
        )

    first, first_anchors = run(99)
    repeated, repeated_anchors = run(99)
    changed, _ = run(100)
    assert set(first_anchors) == {0, 2, 6}
    assert torch.equal(first_anchors[0], initial)
    assert torch.equal(first_anchors[6], first)
    assert torch.equal(first, repeated)
    assert all(torch.equal(first_anchors[key], repeated_anchors[key]) for key in first_anchors)
    assert not torch.equal(first, changed)
    assert torch.equal(initial, torch.zeros_like(initial))
