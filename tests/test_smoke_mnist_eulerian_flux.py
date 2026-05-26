"""Smoke checks for Example 10b MNIST direct Eulerian edge-flux generation."""

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
    build_classwise_ot_cache,
    direct_flux_matching_loss,
    flux_divergence_torch,
    poisson_flux_from_velocity_torch,
    nearest_class_mean_metrics,
    sample_flux_training_batch,
    simulate_direct_flux_generation,
    simulate_teacher_flux_rollout,
    terminal_conditioning_flux_torch,
    train_direct_flux_model,
    training_target_flux_torch,
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


def test_poisson_flux_divergence_matches_velocity() -> None:
    torch.manual_seed(0)
    velocity = torch.randn(3, 8, 8)
    velocity = velocity - velocity.mean(dim=(1, 2), keepdim=True)
    flux = poisson_flux_from_velocity_torch(velocity)
    div = flux_divergence_torch(flux)
    assert flux.shape == (3, 2, 8, 8)
    assert torch.allclose(div, velocity, atol=2e-5, rtol=2e-5)



def test_poisson_ot_and_class_lowres_prior_smoke() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(123)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=3,
        target_mode="poisson-ot-flow",
        source_mode="class-lowres-prior",
        source_lowfreq_size=4,
        source_blur_sigma=0.5,
        ot_lowres_size=4,
        ot_blur_sigma=0.5,
        mean_flow_prob=0.0,
        mean_flow_warmup_prob=0.0,
        tau_sampling="endpoint-mixture",
        tau_source_prob=0.25,
        tau_data_prob=0.25,
        flux_scale=10.0,
    )
    images, labels = _toy_digit_measures(num_samples=30, grid_size=config.grid_size)
    cache = build_classwise_ot_cache(images, labels, config)
    batch = sample_flux_training_batch(
        images,
        labels,
        config,
        batch_size=6,
        device="cpu",
        rng=rng,
        class_means=cache.class_means,
        ot_cache=cache,
        step_index=10,
    )
    assert batch.sources.shape == (6, config.grid_size * config.grid_size)
    assert batch.targets.shape == batch.sources.shape
    assert torch.allclose(batch.sources.sum(dim=1), torch.ones(6), atol=1e-5)
    target_flux = training_target_flux_torch(batch, config)
    assert target_flux.shape == (6, 2, config.grid_size, config.grid_size)

    teacher = simulate_teacher_flux_rollout(batch.sources, batch.targets, config, num_steps=2, device="cpu")
    assert teacher.shape == batch.sources.shape
    assert torch.isfinite(teacher).all()
    assert torch.allclose(teacher.sum(dim=1), torch.ones(6), atol=1e-5)

    metrics = nearest_class_mean_metrics(
        teacher.detach().cpu().numpy(),
        batch.labels.detach().cpu().numpy(),
        cache.class_means,
    )
    assert set(metrics) == {"nearest_mean_acc", "correct_mean_dist", "wrong_mean_margin"}
    assert 0.0 <= metrics["nearest_mean_acc"] <= 1.0

def test_direct_flux_teacher_model_and_sampler_smoke() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=4,
        target_mode="poisson-flow",
        source_mode="lowfreq",
        source_lowfreq_size=4,
        source_blur_sigma=0.25,
        free_weight=0.0,
        noise_weight=0.0,
        learned_weight=1.0,
        terminal_lambda=1.0,
        blur_sigmas=(0.5,),
        blur_weights=(1.0,),
        state_jitter_weight=0.0,
        divergence_loss_weight=0.01,
        flux_scale=10.0,
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
    assert batch.sources.shape == batch.targets.shape == batch.states.shape
    target_flux = training_target_flux_torch(batch, config)
    assert target_flux.shape == (4, 2, config.grid_size, config.grid_size)
    assert torch.isfinite(target_flux).all()

    terminal_flux = terminal_conditioning_flux_torch(batch.states, batch.targets, config)
    assert terminal_flux.shape == (4, 2, config.grid_size, config.grid_size)
    assert torch.isfinite(terminal_flux).all()

    loss, metrics = direct_flux_matching_loss(model, batch)
    assert torch.isfinite(loss)
    assert metrics["loss"] >= 0.0
    assert "div_cos" in metrics

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
    assert len(history["div_cos"]) == 1

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
    assert result.sources is not None
    assert result.sources.shape == result.samples.shape
    assert result.trajectory is not None
    assert result.trajectory.shape == (3, 3, config.grid_size * config.grid_size)
    assert np.all(result.samples >= 0.0)
    assert np.allclose(result.samples.sum(axis=1), 1.0)
