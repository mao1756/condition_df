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
    EDGE_ALPHA_MODES,
    FluxTrainingBatch,
    _ot_coupled_target_indices,
    build_classwise_ot_cache,
    direct_flux_matching_loss,
    edge_alpha_value,
    free_drift_flux_torch,
    flux_divergence_torch,
    make_on_policy_training_batch,
    poisson_flux_from_velocity_torch,
    nearest_class_mean_metrics,
    step_component_rms_torch,
    sample_flux_training_batch,
    simulate_direct_flux_generation,
    simulate_teacher_flux_rollout,
    source_batch_diagnostics,
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




def test_free_aware_target_subtracts_free_drift_and_grid_alpha() -> None:
    rng = np.random.default_rng(11)
    config_plain = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=4,
        target_mode="poisson-flow",
        source_mode="lowfreq",
        source_lowfreq_size=4,
        source_blur_sigma=0.25,
        edge_alpha_mode="grid",
        beta=2.0,
        free_aware_target=False,
        sde_curriculum=True,
        target_free_weight=0.05,
        target_noise_weight=0.01,
        flux_scale=10.0,
    )
    assert EDGE_ALPHA_MODES == ("legacy", "grid")
    assert abs(edge_alpha_value(config_plain) - 2.0 / 64.0) < 1e-12
    config_aware = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=4,
        target_mode="poisson-flow",
        source_mode="lowfreq",
        source_lowfreq_size=4,
        source_blur_sigma=0.25,
        edge_alpha_mode="grid",
        beta=2.0,
        free_aware_target=True,
        sde_curriculum=True,
        target_free_weight=0.05,
        target_noise_weight=0.01,
        flux_scale=10.0,
    )
    images, labels = _toy_digit_measures(grid_size=config_plain.grid_size)
    batch_plain = sample_flux_training_batch(images, labels, config_plain, batch_size=4, device="cpu", rng=rng, step_index=999)
    batch_aware = FluxTrainingBatch(
        tau=batch_plain.tau,
        states=batch_plain.states,
        labels=batch_plain.labels,
        targets=batch_plain.targets,
        sources=batch_plain.sources,
        train_free_weight=batch_plain.train_free_weight,
        train_noise_weight=batch_plain.train_noise_weight,
    )
    plain_flux = training_target_flux_torch(batch_plain, config_plain)
    aware_flux = training_target_flux_torch(batch_aware, config_aware)
    free_flux = free_drift_flux_torch(batch_plain.states, config_plain)
    assert torch.allclose(aware_flux, plain_flux - batch_plain.train_free_weight * free_flux, atol=1e-5, rtol=1e-5)
    comp = step_component_rms_torch(
        batch_plain.states,
        plain_flux,
        0.01,
        config_plain,
        free_weight=batch_plain.train_free_weight,
        noise_weight=batch_plain.train_noise_weight,
    )
    assert comp["learned_step_rms"] > 0.0
    assert comp["free_step_rms"] >= 0.0
    assert comp["noise_step_rms"] >= 0.0


def test_nearest_ot_matching_is_stable_for_identical_sources() -> None:
    rng = np.random.default_rng(7)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        target_mode="poisson-ot-flow",
        source_mode="lowfreq",
        source_lowfreq_size=4,
        ot_match_mode="nearest",
        ot_lowres_size=4,
        ot_blur_sigma=0.5,
        flux_scale=10.0,
    )
    images, labels = _toy_digit_measures(num_samples=40, grid_size=config.grid_size)
    cache = build_classwise_ot_cache(images, labels, config)
    repeated_source_0 = np.repeat(images[0:1], 3, axis=0)
    repeated_source_1 = np.repeat(images[1:2], 3, axis=0)
    source_np = np.concatenate([repeated_source_0, repeated_source_1], axis=0)
    batch_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    assigned = _ot_coupled_target_indices(
        source_np,
        batch_labels,
        images,
        labels,
        config,
        rng=rng,
        ot_cache=cache,
    )
    assert np.unique(assigned[:3]).size == 1
    assert np.unique(assigned[3:]).size == 1
    assert np.all(labels[assigned[:3]] == 0)
    assert np.all(labels[assigned[3:]] == 1)



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



def test_target_lowres_prior_is_coupled_and_audited() -> None:
    rng = np.random.default_rng(321)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=3,
        target_mode="poisson-flow",
        source_mode="target-lowres-prior",
        source_lowfreq_size=4,
        source_blur_sigma=0.5,
        mean_flow_prob=0.0,
        mean_flow_warmup_prob=0.0,
        state_jitter_weight=0.0,
        velocity_target="residual",
        flux_scale=10.0,
    )
    images, labels = _toy_digit_measures(num_samples=40, grid_size=config.grid_size)
    batch = sample_flux_training_batch(
        images,
        labels,
        config,
        batch_size=8,
        device="cpu",
        rng=rng,
    )
    assert batch.source_indices is not None
    assert batch.target_indices is not None
    assert np.array_equal(batch.source_indices, batch.target_indices)
    assert batch.source_labels is not None
    assert np.array_equal(batch.source_labels, batch.labels.cpu().numpy())
    assert np.unique(batch.source_indices).size > 1
    diag = source_batch_diagnostics(
        batch.sources.cpu().numpy(),
        requested_labels=batch.labels.cpu().numpy(),
        source_indices=batch.source_indices,
        source_labels=batch.source_labels,
    )
    assert diag["source_unique_count"] > 1
    assert diag["source_diversity_l2"] > 0.0
    assert diag["source_label_match_rate"] == 1.0

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

    on_policy_batch = make_on_policy_training_batch(
        model,
        images,
        labels,
        config,
        batch_size=4,
        device="cpu",
        rng=rng,
    )
    assert on_policy_batch.target_velocity_mode == "residual"

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
