r"""Smoke checks for MNIST-CP target-conditioned score matching utilities.

Run with
    .venv\Scripts\python.exe -m tests.test_smoke_mnist_target_conditioned_score
"""

from __future__ import annotations

import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch only allows setting this once per process.
    pass

from mnist.mnist_cp import uniform_point_cloud_masses
from mnist.target_conditioned_score import (
    LatentCritic,
    LatentGenerator,
    TargetConditionedScoreModel,
    encode_target_latents,
    empirical_mixture_scaled_score_target,
    evaluate_hybrid_oracle_neural_reconstruction,
    evaluate_model_vs_mixture_oracle,
    evaluate_target_conditioned_score_model,
    fit_gaussian_latent_prior,
    fit_pca_gmm_latent_prior,
    fit_pca_latent_prior,
    fit_pca_gmm_latent_prior,
    fit_score_calibration_against_mixture_oracle,
    latent_nearest_neighbor_diagnostics,
    latent_nearest_neighbor_summary,
    layernorm_project_latents,
    make_sigma_tau_schedule,
    paired_chamfer_reconstruction_metrics,
    perturb_target_conditioned_positions,
    reconstruct_target_conditioned_point_clouds,
    reconstruct_target_conditioned_from_latents,
    sample_empirical_latent_prior,
    sample_gaussian_latent_prior,
    sample_oracle_mixture_annealed_dynamics,
    sample_pca_latent_prior,
    sample_pca_gmm_latent_prior,
    sample_target_conditioned_annealed_dynamics,
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
        use_target_grid_conditioning=True,
        target_grid_feature_dim=8,
        target_grid_dropout_probability=0.10,
        measure_gate_init=-5.0,
        measure_gate_max=0.10,
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
        direct_mixture_probability=0.25,
        oracle_replay_probability=0.25,
        oracle_replay_weight=1.2,
        oracle_replay_steps_per_level=1,
        oracle_replay_max_levels=1,
        direct_query_modes=("noised_target", "uniform"),
        freeze_measure_branch_epochs=0,
        measure_gate_regularization=1e-3,
        device="cpu",
        verbose=False,
    )
    assert len(history["train_loss"]) == 1
    assert "train_replay_fraction" in history
    eval_metrics = evaluate_target_conditioned_score_model(
        model,
        masses,
        positions,
        labels,
        tau_levels=tau_levels[:1],
        batch_size=4,
        direct_mixture_probability=0.0,
        oracle_replay_probability=0.0,
        device="cpu",
    )
    assert "loss_ratio_vs_zero" in eval_metrics


def test_sampling_and_latent_priors_smoke() -> None:
    torch.manual_seed(1)
    masses, positions, labels = _toy_contours(num_samples=6, num_points=12)
    _, tau_levels = make_sigma_tau_schedule(num_points=12, num_levels=2, sigma_max=0.06, sigma_min=0.03)
    model = _small_model(float(np.min(tau_levels)), float(np.max(tau_levels)))
    latents = encode_target_latents(model, masses, positions, batch_size=3, device="cpu")
    assert latents.shape == (6, 16)
    calibration = None

    prior = fit_gaussian_latent_prior(latents, labels, diagonal=True)
    z_sample, y_sample = sample_gaussian_latent_prior(
        prior,
        labels=np.asarray([0, 1, 0]),
        covariance_scale=0.5,
        layernorm_project=True,
        rng=np.random.default_rng(2),
    )
    assert z_sample.shape == (3, 16)
    assert np.array_equal(y_sample, np.asarray([0, 1, 0]))
    assert layernorm_project_latents(z_sample).shape == z_sample.shape

    empirical_z, empirical_y = sample_empirical_latent_prior(
        latents,
        labels,
        requested_labels=np.asarray([0, 1, 0]),
        noise_scale=0.01,
        layernorm_project=True,
        rng=np.random.default_rng(22),
    )
    assert empirical_z.shape == (3, 16)
    assert np.array_equal(empirical_y, np.asarray([0, 1, 0]))

    pca_prior = fit_pca_latent_prior(
        latents,
        labels,
        pca_dim=4,
        shrink=0.25,
        layernorm_project=True,
    )
    pca_z, pca_y = sample_pca_latent_prior(
        pca_prior,
        labels=np.asarray([0, 1, 0]),
        layernorm_project=True,
        rng=np.random.default_rng(24),
    )
    assert pca_z.shape == (3, 16)
    assert np.array_equal(pca_y, np.asarray([0, 1, 0]))

    pca_gmm_prior = fit_pca_gmm_latent_prior(
        latents,
        labels,
        pca_dim=4,
        components_per_class=2,
        covariance_shrink=0.25,
        layernorm_project=True,
        rng=np.random.default_rng(23),
    )
    pca_gmm_z, pca_gmm_y = sample_pca_gmm_latent_prior(
        pca_gmm_prior,
        labels=np.asarray([0, 1, 0]),
        layernorm_project=True,
        rng=np.random.default_rng(26),
    )
    assert pca_gmm_z.shape == (3, 16)
    assert np.array_equal(pca_gmm_y, np.asarray([0, 1, 0]))

    nn_rows = latent_nearest_neighbor_diagnostics(
        latents,
        {"pca": pca_z, "pca_gmm": pca_gmm_z, "baseline": latents[:3]},
        reference_labels=labels,
        query_labels={"pca": pca_y, "pca_gmm": pca_gmm_y, "baseline": labels[:3]},
    )
    assert len(nn_rows) == 3
    assert "mean_nn_distance" in nn_rows[0]

    latent_only_generated = reconstruct_target_conditioned_from_latents(
        model,
        target_latents=latents[:2],
        labels=labels[:2],
        output_masses=masses[:2],
        tau_levels=tau_levels,
        steps_per_level=1,
        sampler_scheme="shape_gf_langevin",
        state_projection="none",
        langevin_alpha=1e-5,
        final_polish_steps=0,
        batch_size=2,
        device="cpu",
        rng=np.random.default_rng(25),
    )
    assert latent_only_generated.positions.shape == positions[:2].shape

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
        oracle_prefix_levels=1,
        batch_size=2,
        device="cpu",
        rng=np.random.default_rng(3),
    )
    assert generated.positions.shape == positions[:2].shape
    metrics = paired_chamfer_reconstruction_metrics(generated.positions, positions[:2], labels[:2])
    assert "mean_chamfer" in metrics
    hybrid_rows = evaluate_hybrid_oracle_neural_reconstruction(
        model,
        masses[:2],
        positions[:2],
        labels[:2],
        tau_levels=tau_levels,
        prefix_levels=(0, 1),
        max_samples=2,
        steps_per_level=1,
        langevin_alpha=1e-5,
        batch_size=2,
        device="cpu",
        rng=np.random.default_rng(5),
    )
    assert len(hybrid_rows) == 2

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
    test_sampling_and_latent_priors_smoke()
    print("All target-conditioned MNIST-CP score smoke tests passed.")
