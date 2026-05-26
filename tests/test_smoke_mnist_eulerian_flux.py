"""Smoke checks for Example 10 MNIST direct Eulerian edge-flux generation."""

from __future__ import annotations

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch only allows setting this once per process.
    pass

from mnist.eulerian_flux_mnist import (
    DirectFluxMNISTConfig,
    DirectFluxUNet,
    direct_flux_matching_loss,
    sample_flux_training_batch,
    simulate_direct_flux_generation,
    terminal_conditioning_flux_torch,
    train_direct_flux_model,
)
from mnist.weighted_point_cloud import normalize_images_to_measures


def _toy_digit_measures(num_samples: int = 20, grid_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    images = []
    labels = []
    for idx in range(num_samples):
        label = idx % 2
        cx = 0.35 + 0.30 * label
        cy = 0.50
        x = (xx + 0.5) / float(grid_size)
        y = (yy + 0.5) / float(grid_size)
        blob = np.exp(-45.0 * ((x - cx) ** 2 + (y - cy) ** 2))
        if label == 1:
            blob += 0.7 * np.exp(-60.0 * ((x - 0.35) ** 2 + (y - 0.35) ** 2))
        images.append(blob)
        labels.append(label)
    return normalize_images_to_measures(np.asarray(images)), np.asarray(labels, dtype=np.int64)


def test_direct_flux_teacher_model_and_sampler_smoke() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=4,
        terminal_lambda=1.0,
        blur_sigmas=(0.5,),
        blur_weights=(1.0,),
        state_jitter_weight=0.0,
        divergence_loss_weight=0.01,
        flux_scale=20.0,
    )
    images, labels = _toy_digit_measures(grid_size=config.grid_size)
    model = DirectFluxUNet(config, base_channels=4, num_classes=10)

    batch = sample_flux_training_batch(
        images,
        labels,
        config,
        batch_size=4,
        device="cpu",
        rng=rng,
    )
    flux = terminal_conditioning_flux_torch(batch.states, batch.targets, config)
    assert flux.shape == (4, 2, config.grid_size, config.grid_size)
    assert torch.isfinite(flux).all()

    loss, metrics = direct_flux_matching_loss(model, batch)
    assert torch.isfinite(loss)
    assert metrics["loss"] >= 0.0

    history = train_direct_flux_model(
        model,
        images,
        labels,
        train_steps=1,
        batch_size=4,
        lr=1e-3,
        device="cpu",
        seed=1,
        use_amp=False,
        show_progress=False,
    )
    assert len(history["loss"]) == 1

    result = simulate_direct_flux_generation(
        model,
        labels=[0, 1, 0],
        num_steps=2,
        save_every=1,
        deterministic=True,
        device="cpu",
        seed=2,
        use_amp=False,
        show_progress=False,
    )
    assert result.samples.shape == (3, config.grid_size * config.grid_size)
    assert result.trajectory is not None
    assert result.trajectory.shape == (3, 3, config.grid_size * config.grid_size)
    assert np.all(result.samples >= 0.0)
    assert np.allclose(result.samples.sum(axis=1), 1.0)
