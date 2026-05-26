"""Smoke checks for Example 9 Eulerian image bridge helpers."""

from __future__ import annotations

import numpy as np
import torch

from examples.eulerian_image_bridge import (
    EulerianImageBridgeConfig,
    PositiveHeatPotentialCNN,
    make_synthetic_butterfly_image,
    make_synthetic_house_image,
    natural_horizon,
    normalize_image_to_measure,
    simulate_conditioned_image_bridge,
    simulate_free_rollout,
    terminal_potential_torch,
    terminal_preference_numpy,
)


def test_example9_terminal_score_and_rollouts_smoke() -> None:
    rng = np.random.default_rng(123)
    torch.manual_seed(123)
    config = EulerianImageBridgeConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=4,
        terminal_lambda=1.0,
        blur_sigmas=(1.0,),
        blur_weights=(1.0,),
    )
    house = normalize_image_to_measure(make_synthetic_house_image(config.grid_size))
    butterfly = normalize_image_to_measure(make_synthetic_butterfly_image(config.grid_size))

    score = terminal_preference_numpy(house, butterfly, config)
    assert np.isfinite(score)
    assert 0.0 < float(score) <= 1.0

    rollout = simulate_free_rollout(
        house,
        natural_horizon(config),
        config,
        rng=rng,
        num_steps=2,
        return_trajectory=True,
        deterministic=True,
    )
    assert rollout.trajectory.shape == (3, config.grid_size * config.grid_size)
    assert np.all(rollout.trajectory >= 0.0)
    assert np.allclose(rollout.final_state.sum(), 1.0)

    model = PositiveHeatPotentialCNN(config, hidden_channels=4, hidden_dim=8)
    masses = torch.tensor(house[None, :], dtype=torch.float32, requires_grad=True)
    target = torch.tensor(butterfly, dtype=torch.float32)
    phi = terminal_potential_torch(masses, target, config)
    assert phi.shape == (1,)
    assert torch.isfinite(phi).all()
    value = model(torch.tensor([natural_horizon(config)], dtype=torch.float32), masses, target)
    grad = torch.autograd.grad(value.sum(), masses)[0]
    assert value.shape == (1,)
    assert grad.shape == masses.shape
    assert torch.isfinite(grad).all()

    result = simulate_conditioned_image_bridge(
        model,
        house,
        butterfly,
        config,
        rng=rng,
        num_steps=2,
        save_every=1,
        deterministic=True,
    )
    assert result.trajectory.shape == (3, config.grid_size * config.grid_size)
    assert result.terminal_scores.shape == (3,)
    assert np.all(result.trajectory >= 0.0)
    assert np.allclose(result.trajectory.sum(axis=1), 1.0)
