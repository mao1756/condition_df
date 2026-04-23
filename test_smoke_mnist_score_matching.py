"""Smoke tests for the MNIST score-matching weighted point-cloud experiment.

Run with
    python test_smoke_mnist_score_matching.py
"""

from __future__ import annotations

import numpy as np
import torch

from mnist_weighted_point_cloud import images_to_weighted_point_clouds, normalize_images_to_measures
from mnist_score_matching import (
    ConditionalScoreSetNetwork,
    ConditionalScoreSetTransformer,
    add_forward_noise_to_positions,
    diagnose_score_prior_horizons,
    evaluate_score_model,
    evaluate_score_model_by_tau_bins,
    generate_balanced_score_matching_dataset,
    recommend_score_prior_horizon,
    perturb_weighted_point_cloud_positions,
    torus_heat_kernel_score_target,
    train_score_model,
)


def _make_toy_images(num_per_class: int = 10, image_size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    images = []
    labels = []
    xs = np.arange(image_size)
    for label in [0, 1]:
        for _ in range(num_per_class):
            image = np.zeros((image_size, image_size), dtype=np.float64)
            jitter = rng.normal(scale=0.6, size=image_size)
            if label == 0:
                ys = np.clip((0.20 * image_size + 0.18 * xs + jitter).astype(int), 0, image_size - 1)
            else:
                ys = np.clip((0.80 * image_size - 0.18 * xs + jitter).astype(int), 0, image_size - 1)
            image[ys, xs] = 1.0
            image += 0.05 * rng.random((image_size, image_size))
            images.append(image)
            labels.append(label)
    images = normalize_images_to_measures(np.asarray(images, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int64)
    return images, labels


def _test_perturbation_target_matches_formula() -> None:
    torch.manual_seed(0)
    masses = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    positions = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32)
    tau = torch.tensor([0.05], dtype=torch.float32)

    noisy, target, noise = perturb_weighted_point_cloud_positions(
        masses,
        positions,
        tau,
        projection="none",
    )
    sigma = torch.sqrt((2.0 * tau[:, None, None]) / masses.unsqueeze(-1))
    reconstructed = positions + sigma * noise
    expected_target = -(reconstructed - positions) / (2.0 * tau[:, None, None])

    assert torch.allclose(noisy, reconstructed)
    assert torch.allclose(target, expected_target, atol=1e-6, rtol=1e-5)



def _test_torus_score_target_smoke() -> None:
    masses = torch.tensor([[1.0, 0.25]], dtype=torch.float32)
    clean = torch.tensor([[[0.10, 0.20], [0.80, 0.40]]], dtype=torch.float32)
    wrapped = torch.remainder(clean + torch.tensor([[[0.03, -0.02], [0.10, 0.15]]]), 1.0)
    tau = torch.tensor([1e-4], dtype=torch.float32)

    target = torus_heat_kernel_score_target(masses, clean, wrapped, tau)
    expected_first = -(wrapped[:, :1] - clean[:, :1]) / (2.0 * tau[:, None, None])
    assert target.shape == clean.shape
    assert torch.all(torch.isfinite(target))
    assert torch.allclose(target[:, :1], expected_first, atol=1e-3, rtol=1e-3)

    large_tau_target = torus_heat_kernel_score_target(
        masses,
        clean,
        wrapped,
        torch.tensor([10.0], dtype=torch.float32),
    )
    assert torch.max(torch.abs(large_tau_target)).item() < 1e-5


def _test_deepsets_score_scaling_options() -> None:
    torch.manual_seed(1)
    masses = torch.tensor([[0.25, 1.0]], dtype=torch.float32)
    positions = torch.tensor([[[0.10, 0.20], [0.30, 0.40]]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.long)
    tau = torch.tensor([1e-3], dtype=torch.float32)

    common_kwargs = dict(
        point_feature_dim=12,
        hidden_dim=16,
        conditioning_dim=8,
        num_classes=2,
        condition_on_label=True,
        tau_min=1e-5,
        tau_max=1e-2,
        dropout=0.0,
        use_torus_features=True,
    )

    torch.manual_seed(2)
    tau_model = ConditionalScoreSetNetwork(**common_kwargs, score_output_scaling="tau")
    torch.manual_seed(2)
    tau_mass_model = ConditionalScoreSetNetwork(**common_kwargs, score_output_scaling="tau_mass")
    tau_mass_model.load_state_dict(tau_model.state_dict())

    out_tau = tau_model(masses, positions, tau, labels)
    out_tau_mass = tau_mass_model(masses, positions, tau, labels)

    assert out_tau.shape == positions.shape
    assert out_tau_mass.shape == positions.shape
    assert torch.all(torch.isfinite(out_tau_mass))

    expected_ratio = torch.rsqrt(masses)[:, :, None]
    assert torch.allclose(out_tau_mass, out_tau * expected_ratio, atol=1e-6, rtol=1e-5)

    raw_model = ConditionalScoreSetNetwork(**common_kwargs, score_output_scaling="none")
    raw_out = raw_model(masses, positions, tau, labels)
    assert raw_out.shape == positions.shape

    try:
        ConditionalScoreSetNetwork(**common_kwargs, score_output_scaling="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid score_output_scaling should raise ValueError")


def _test_training_and_generation_smoke() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    images, labels = _make_toy_images(num_per_class=8)
    point_clouds = images_to_weighted_point_clouds(
        images,
        labels=labels,
        top_k=8,
        mass_floor=1e-3,
    )

    model = ConditionalScoreSetTransformer(
        point_feature_dim=16,
        hidden_dim=32,
        conditioning_dim=16,
        num_classes=2,
        condition_on_label=True,
        tau_min=5e-4,
        tau_max=5e-3,
        num_attention_layers=1,
        num_attention_heads=4,
        feedforward_dim=32,
        use_torus_features=True,
        score_output_scaling="tau_mass",
    )

    history = train_score_model(
        model,
        point_clouds.masses,
        point_clouds.positions,
        point_clouds.labels,
        val_masses=point_clouds.masses,
        val_positions=point_clouds.positions,
        val_labels=point_clouds.labels,
        epochs=2,
        batch_size=8,
        lr=2e-3,
        tau_min=5e-4,
        tau_max=5e-3,
        tau_sampling="uniform",
        projection="wrap",
        device="cpu",
        verbose=False,
    )
    assert len(history["train_loss"]) == 2

    metrics = evaluate_score_model(
        model,
        point_clouds.masses,
        point_clouds.positions,
        point_clouds.labels,
        batch_size=8,
        tau_min=5e-4,
        tau_max=5e-3,
        tau_sampling="uniform",
        projection="wrap",
        device="cpu",
    )
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["sample_loss"])
    assert np.isfinite(metrics["zero_predictor_loss"])

    tau_rows = evaluate_score_model_by_tau_bins(
        model,
        point_clouds.masses,
        point_clouds.positions,
        point_clouds.labels,
        batch_size=8,
        tau_min=5e-4,
        tau_max=5e-3,
        num_bins=3,
        projection="wrap",
        device="cpu",
    )
    assert len(tau_rows) == 3
    assert {"tau", "loss", "zero_predictor_loss", "fraction_improved_over_zero"}.issubset(tau_rows[0])

    generated = generate_balanced_score_matching_dataset(
        model,
        point_clouds.masses,
        bank_labels=point_clouds.labels,
        num_points=point_clouds.num_points,
        num_per_class=3,
        mass_sampling_mode="bank",
        class_conditional_mass_sampling=True,
        horizon=5e-3,
        step_size=2.5e-3,
        initial_position_mode="uniform",
        state_projection="wrap",
        batch_size=3,
        rasterize=True,
        image_size=28,
        device="cpu",
        rng=np.random.default_rng(321),
    )

    assert generated.masses.shape == (6, point_clouds.num_points)
    assert generated.positions.shape == (6, point_clouds.num_points, 2)
    assert generated.images is not None and generated.images.shape == (6, 28, 28)
    assert np.all(np.isfinite(generated.positions))
    assert np.all(np.isfinite(generated.images))
    assert np.array_equal(generated.labels, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))

    generated_from_noised_bank = generate_balanced_score_matching_dataset(
        model,
        point_clouds.masses,
        bank_labels=point_clouds.labels,
        num_points=point_clouds.num_points,
        num_per_class=2,
        mass_sampling_mode="bank",
        class_conditional_mass_sampling=True,
        horizon=5e-3,
        step_size=2.5e-3,
        initial_position_mode="forward_noised_bank",
        initial_position_bank=point_clouds.positions,
        initial_position_bank_labels=point_clouds.labels,
        state_projection="wrap",
        batch_size=2,
        rasterize=True,
        image_size=28,
        device="cpu",
        rng=np.random.default_rng(654),
    )
    assert generated_from_noised_bank.positions.shape == (4, point_clouds.num_points, 2)
    assert np.all(np.isfinite(generated_from_noised_bank.positions))



def _test_bridge_sampler_smoke() -> None:
    images, labels = _make_toy_images(num_per_class=3)
    point_clouds = images_to_weighted_point_clouds(
        images,
        labels=labels,
        top_k=6,
        mass_floor=1e-3,
    )

    model = ConditionalScoreSetNetwork(
        point_feature_dim=12,
        hidden_dim=16,
        conditioning_dim=8,
        num_classes=2,
        condition_on_label=True,
        tau_min=5e-4,
        tau_max=5e-3,
        use_torus_features=True,
        score_output_scaling="tau_mass",
    )

    generated_bridge = generate_balanced_score_matching_dataset(
        model,
        point_clouds.masses,
        bank_labels=point_clouds.labels,
        num_points=point_clouds.num_points,
        num_per_class=2,
        mass_sampling_mode="bank",
        class_conditional_mass_sampling=True,
        horizon=5e-3,
        step_size=2.5e-3,
        initial_position_mode="forward_noised_bank",
        initial_position_bank=point_clouds.positions,
        initial_position_bank_labels=point_clouds.labels,
        state_projection="wrap",
        diffusion_temperature=1.0,
        score_scale=1.0,
        sampler_scheme="bridge",
        batch_size=2,
        rasterize=True,
        image_size=28,
        device="cpu",
        rng=np.random.default_rng(655),
    )
    assert generated_bridge.positions.shape == (4, point_clouds.num_points, 2)
    assert generated_bridge.images is not None and generated_bridge.images.shape == (4, 28, 28)
    assert np.all(np.isfinite(generated_bridge.positions))
    assert np.all(np.isfinite(generated_bridge.images))


def _test_horizon_diagnostic_smoke() -> None:
    images, labels = _make_toy_images(num_per_class=4)
    point_clouds = images_to_weighted_point_clouds(
        images,
        labels=labels,
        top_k=6,
        mass_floor=1e-3,
    )

    noised = add_forward_noise_to_positions(
        point_clouds.masses[:3],
        point_clouds.positions[:3],
        1e-3,
        projection="wrap",
        rng=np.random.default_rng(11),
    )
    assert noised.shape == point_clouds.positions[:3].shape
    assert np.all(np.isfinite(noised))

    diagnostics = diagnose_score_prior_horizons(
        point_clouds.masses,
        point_clouds.positions,
        point_clouds.labels,
        [5e-4, 2e-3],
        projection="wrap",
        max_samples=8,
        feature_grid_size=4,
        classifier_epochs=1,
        classifier_batch_size=4,
        device="cpu",
        rng=np.random.default_rng(12),
    )
    assert len(diagnostics) == 2
    assert {"horizon", "prior_accuracy", "label_accuracy", "weighted_marginal_tv"}.issubset(diagnostics[0])
    recommendation = recommend_score_prior_horizon(diagnostics, max_prior_accuracy=1.0, label_accuracy_slack=1.0)
    assert recommendation["horizon"] in {5e-4, 2e-3}


if __name__ == "__main__":
    _test_perturbation_target_matches_formula()
    _test_torus_score_target_smoke()
    _test_deepsets_score_scaling_options()
    _test_training_and_generation_smoke()
    _test_bridge_sampler_smoke()
    _test_horizon_diagnostic_smoke()
    print("All mnist_score_matching smoke tests passed.")
