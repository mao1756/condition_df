r"""Smoke checks for MNIST-CP target-conditioned score matching utilities.

Run with
    .venv\Scripts\python.exe -m tests.test_smoke_mnist_target_conditioned_score
"""

from __future__ import annotations

import numpy as np
import torch

torch.set_num_threads(1)

from mnist.mnist_cp import uniform_point_cloud_masses
from mnist.target_conditioned_score import (
    LatentCritic,
    LatentGenerator,
    TargetConditionedScoreModel,
    encode_target_latents,
    empirical_mixture_scaled_score_target,
    evaluate_model_vs_mixture_oracle,
    evaluate_target_conditioned_score_model,
    fit_gaussian_latent_prior,
    make_sigma_tau_schedule,
    paired_chamfer_reconstruction_metrics,
    perturb_target_conditioned_positions,
    reconstruct_target_conditioned_point_clouds,
    sample_oracle_mixture_annealed_dynamics,
    sample_gaussian_latent_prior,
    sample_wgan_latent_prior,
    target_conditioned_score_matching_loss,
    train_latent_wgan_gp,
    train_target_conditioned_score_model,
)


def _toy_contours(num_samples: int = 6, num_points: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    positions = []
    labels = []
    for i in range(num_samples):
        label = i % 2
        center = np.asarray([0.45 + 0.08 * label, 0.50])
        radius_x = 0.16 + 0.02 * label
        radius_y = 0.12 + 0.01 * (i % 3)
        positions.append(center + np.column_stack([radius_x * np.cos(theta), radius_y * np.sin(theta)]))
        labels.append(label)
    masses = uniform_point_cloud_masses(num_samples, num_points, dtype=np.float64)
    return masses, np.asarray(positions, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _small_model(tau_min: float, tau_max: float) -> TargetConditionedScoreModel:
    return TargetConditionedScoreModel(
        latent_dim=16,
        target_encoder_hidden_dim=32,
        grid_size=16,
        base_channels=8,
        grid_feature_dim=12,
        set_feature_dim=16,
        set_hidden_dim=16,
        set_blocks=1,
        score_hidden_dim=32,
        score_residual_blocks=1,
        time_dim=16,
        context_dim=24,
        condition_on_label=True,
        tau_min=tau_min,
        tau_max=tau_max,
        dropout=0.0,
        use_image_field=False,
    )


def test_forward_loss_and_training_smoke() -> None:
    torch.manual_seed(0)
    masses, positions, labels = _toy_contours(num_samples=8, num_points=16)
    _, tau_levels = make_sigma_tau_schedule(num_points=16, num_levels=3, sigma_max=0.08, sigma_min=0.02)
    model = _small_model(float(np.min(tau_levels)), float(np.max(tau_levels)))

    batch_masses = torch.tensor(masses[:2], dtype=torch.float32)
    clean = torch.tensor(positions[:2], dtype=torch.float32)
    batch_labels = torch.tensor(labels[:2], dtype=torch.long)
    tau = torch.full((2,), float(tau_levels[0]), dtype=torch.float32)
    noisy, target_scaled, target_score = perturb_target_conditioned_positions(batch_masses, clean, tau)
    assert noisy.shape == clean.shape
    assert target_scaled.shape == clean.shape
    assert target_score.shape == clean.shape
    mixture_scaled = empirical_mixture_scaled_score_target(noisy, clean, batch_masses, tau)
    assert mixture_scaled.shape == clean.shape

    pred_scaled = model.predict_scaled_score(
        batch_masses,
        noisy,
        tau,
        target_positions=clean,
        target_masses=batch_masses,
        labels=batch_labels,
    )
    loss, metrics = target_conditioned_score_matching_loss(pred_scaled, target_scaled, batch_masses, tau)
    assert torch.isfinite(loss)
    assert metrics["loss"] >= 0.0

    history = train_target_conditioned_score_model(
        model,
        masses,
        positions,
        labels,
        val_masses=masses,
        val_positions=positions,
        val_labels=labels,
        tau_levels=tau_levels,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        direct_mixture_probability=0.5,
        direct_query_modes=("noised_target", "uniform"),
        device="cpu",
        verbose=False,
    )
    assert len(history["train_loss"]) == 1
    eval_metrics = evaluate_target_conditioned_score_model(
        model,
        masses,
        positions,
        labels,
        tau_levels=tau_levels,
        batch_size=4,
        direct_mixture_probability=0.5,
        direct_query_modes=("uniform",),
        device="cpu",
    )
    assert "loss_ratio_vs_zero" in eval_metrics
    oracle_rows = evaluate_model_vs_mixture_oracle(
        model,
        masses,
        positions,
        labels,
        tau_levels=tau_levels[:1],
        query_modes=("uniform",),
        max_samples=2,
        batch_size=2,
        device="cpu",
    )
    assert len(oracle_rows) == 1
    assert "relative_rmse" in oracle_rows[0]


def test_sampling_latent_prior_and_wgan_smoke() -> None:
    torch.manual_seed(1)
    masses, positions, labels = _toy_contours(num_samples=6, num_points=12)
    _, tau_levels = make_sigma_tau_schedule(num_points=12, num_levels=2, sigma_max=0.06, sigma_min=0.03)
    model = _small_model(float(np.min(tau_levels)), float(np.max(tau_levels)))
    latents = encode_target_latents(model, masses, positions, batch_size=3, device="cpu")
    assert latents.shape == (6, 16)

    prior = fit_gaussian_latent_prior(latents, labels, diagonal=True)
    z_sample, y_sample = sample_gaussian_latent_prior(prior, labels=np.asarray([0, 1, 0]), rng=np.random.default_rng(2))
    assert z_sample.shape == (3, 16)
    assert np.array_equal(y_sample, np.asarray([0, 1, 0]))

    generated = reconstruct_target_conditioned_point_clouds(
        model,
        masses[:2],
        positions[:2],
        labels[:2],
        tau_levels=tau_levels,
        steps_per_level=1,
        sampler_scheme="shape_gf_langevin",
        state_projection="none",
        langevin_alpha=1e-5,
        final_polish_steps=0,
        batch_size=2,
        device="cpu",
        rng=np.random.default_rng(3),
    )
    assert generated.positions.shape == positions[:2].shape
    metrics = paired_chamfer_reconstruction_metrics(generated.positions, positions[:2], labels[:2])
    assert "mean_chamfer" in metrics

    oracle_generated = sample_oracle_mixture_annealed_dynamics(
        target_masses=masses[:2],
        target_positions=positions[:2],
        labels=labels[:2],
        tau_levels=tau_levels,
        steps_per_level=1,
        final_polish_steps=0,
        state_projection="none",
        langevin_alpha=1e-5,
        batch_size=2,
        device="cpu",
        rng=np.random.default_rng(4),
    )
    assert oracle_generated.positions.shape == positions[:2].shape

    generator = LatentGenerator(noise_dim=8, latent_dim=16, hidden_dims=(16,), conditional=True)
    critic = LatentCritic(latent_dim=16, hidden_dims=(16,), conditional=True)
    gan_history = train_latent_wgan_gp(
        generator,
        critic,
        latents,
        labels,
        epochs=1,
        batch_size=3,
        critic_steps=1,
        device="cpu",
        verbose=False,
    )
    assert len(gan_history["critic_loss"]) == 1
    z_wgan, y_wgan = sample_wgan_latent_prior(generator, labels=np.asarray([0, 1]), device="cpu")
    assert z_wgan.shape == (2, 16)
    assert np.array_equal(y_wgan, np.asarray([0, 1]))


if __name__ == "__main__":
    test_forward_loss_and_training_smoke()
    test_sampling_latent_prior_and_wgan_smoke()
    print("All target-conditioned MNIST-CP score smoke tests passed.")
