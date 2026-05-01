"""Smoke tests for the MNIST weighted point-cloud experiment code.

Run with
    .venv\\Scripts\\python.exe -m tests.test_smoke_mnist_conditioned_diffusion
"""

from __future__ import annotations

import numpy as np
import torch

from mnist.weighted_point_cloud import (
    images_to_weighted_point_clouds,
    normalize_images_to_measures,
    rasterize_weighted_point_clouds,
)
import mnist.conditioned_diffusion as mnist_cd
import mnist.experiment6_fixes as mnist_fix
from mnist.conditioned_diffusion import (
    TerminalSetClassifier,
    confusion_matrix_from_predictions,
    evaluate_generation_metrics,
    generate_balanced_synthetic_dataset,
    generate_guided_point_clouds,
    train_terminal_set_classifier,
    terminal_g_accuracy,
)
from mnist.experiment6_fixes import (
    generate_balanced_synthetic_dataset_reparam,
    generate_guided_point_clouds_reparam,
    sample_truncated_poisson_dirichlet_masses,
)


def _make_toy_images(num_per_class: int = 12, image_size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    images = []
    labels = []
    xs = np.arange(image_size)
    for label in [0, 1]:
        for _ in range(num_per_class):
            image = np.zeros((image_size, image_size), dtype=np.float64)
            jitter = rng.normal(scale=0.6, size=image_size)
            if label == 0:
                ys = np.clip((0.25 * image_size + 0.15 * xs + jitter).astype(int), 0, image_size - 1)
            else:
                ys = np.clip((0.75 * image_size - 0.15 * xs + jitter).astype(int), 0, image_size - 1)
            image[ys, xs] = 1.0
            image += 0.05 * rng.random((image_size, image_size))
            images.append(image)
            labels.append(label)
    images = normalize_images_to_measures(np.asarray(images, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int64)
    return images, labels




def _make_toy_mass_bank() -> tuple[np.ndarray, np.ndarray]:
    masses = np.asarray(
        [
            [0.55, 0.45],
            [0.60, 0.40],
            [0.65, 0.35],
            [0.70, 0.30],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return masses, labels


def _make_toy_position_bank() -> np.ndarray:
    return np.asarray(
        [
            [[0.10, 0.15], [0.20, 0.25]],
            [[0.30, 0.15], [0.40, 0.25]],
            [[0.60, 0.65], [0.70, 0.75]],
            [[0.80, 0.65], [0.90, 0.75]],
        ],
        dtype=np.float64,
    )


def _regression_test_generation_updates_every_batch() -> None:
    torch.manual_seed(0)
    mass_bank, bank_labels = _make_toy_mass_bank()
    target_labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    model = TerminalSetClassifier(point_feature_dim=8, hidden_dim=16, num_classes=2)

    generated = generate_guided_point_clouds(
        model,
        mass_bank,
        target_labels,
        bank_labels=bank_labels,
        class_conditional_mass_sampling=True,
        horizon=0.01,
        step_size=0.005,
        terminal_mc_samples=4,
        guidance_scale=1.0,
        initial_position_scale=0.05,
        batch_size=2,
        return_trajectories=True,
        rasterize=False,
        device="cpu",
        rng=np.random.default_rng(123),
    )
    assert generated.trajectories is not None
    displacement = np.max(
        np.linalg.norm(generated.trajectories[-1] - generated.trajectories[0], axis=-1),
        axis=1,
    )
    assert np.all(displacement > 0.0), displacement


def _regression_test_generation_respects_mc_kwargs() -> None:
    torch.manual_seed(0)
    mass_bank, bank_labels = _make_toy_mass_bank()
    target_labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    model = TerminalSetClassifier(point_feature_dim=8, hidden_dim=16, num_classes=2)

    calls: list[tuple[int, float, str]] = []
    original = mnist_cd.estimate_monte_carlo_guided_drift

    @torch.enable_grad()
    def _spy(
        model,
        masses,
        positions,
        labels,
        tau,
        *,
        terminal_mc_samples=64,
        guidance_scale=1.0,
        terminal_projection="reflect",
    ) -> torch.Tensor:
        calls.append((int(terminal_mc_samples), float(guidance_scale), str(terminal_projection)))
        return torch.zeros_like(positions)

    mnist_cd.estimate_monte_carlo_guided_drift = _spy
    try:
        generate_guided_point_clouds(
            model,
            mass_bank,
            target_labels,
            bank_labels=bank_labels,
            class_conditional_mass_sampling=True,
            horizon=0.01,
            step_size=0.005,
            terminal_mc_samples=7,
            guidance_scale=1.25,
            initial_position_scale=0.05,
            terminal_projection="clip",
            drift_clip_norm=None,
            batch_size=2,
            rasterize=False,
            device="cpu",
            rng=np.random.default_rng(321),
        )
    finally:
        mnist_cd.estimate_monte_carlo_guided_drift = original

    assert calls
    assert set(calls) == {(7, 1.25, "clip")}


def _regression_test_joint_bank_sampling_keeps_mass_position_pairing() -> None:
    torch.manual_seed(0)
    mass_bank, bank_labels = _make_toy_mass_bank()
    position_bank = _make_toy_position_bank()
    target_labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    model = TerminalSetClassifier(point_feature_dim=8, hidden_dim=16, num_classes=2)

    original = mnist_fix.estimate_reparameterized_guided_drift

    @torch.enable_grad()
    def _zero_drift(
        model,
        masses,
        positions,
        labels,
        tau,
        *,
        terminal_mc_samples=128,
        guidance_scale=3.0,
        terminal_projection="reflect",
        antithetic=True,
    ) -> torch.Tensor:
        return torch.zeros_like(positions)

    mnist_fix.estimate_reparameterized_guided_drift = _zero_drift
    try:
        generated = generate_guided_point_clouds_reparam(
            model,
            mass_bank,
            target_labels,
            bank_labels=bank_labels,
            mass_sampling_mode="bank",
            class_conditional_mass_sampling=True,
            horizon=0.01,
            step_size=0.01,
            terminal_mc_samples=4,
            guidance_scale=1.0,
            initial_position_mode="bank",
            initial_position_bank=position_bank,
            initial_position_bank_labels=bank_labels,
            joint_bank_sampling=True,
            initial_position_jitter=0.0,
            state_projection="reflect",
            terminal_projection="reflect",
            diffusion_temperature=1e-12,
            drift_clip_norm=None,
            batch_size=2,
            return_trajectories=True,
            rasterize=False,
            device="cpu",
            rng=np.random.default_rng(123),
        )
    finally:
        mnist_fix.estimate_reparameterized_guided_drift = original

    assert generated.trajectories is not None
    initial_positions = generated.trajectories[0]
    for masses_row, positions_row, label in zip(generated.masses, initial_positions, target_labels):
        candidates = np.flatnonzero(bank_labels == label)
        assert any(
            np.allclose(masses_row, mass_bank[idx]) and np.allclose(positions_row, position_bank[idx])
            for idx in candidates
        )


def _regression_test_reparam_initial_positions_are_projected() -> None:
    torch.manual_seed(0)
    mass_bank, bank_labels = _make_toy_mass_bank()
    target_labels = np.asarray([0, 1], dtype=np.int64)
    model = TerminalSetClassifier(point_feature_dim=8, hidden_dim=16, num_classes=2)

    original = mnist_fix.estimate_reparameterized_guided_drift

    @torch.enable_grad()
    def _zero_drift(
        model,
        masses,
        positions,
        labels,
        tau,
        *,
        terminal_mc_samples=128,
        guidance_scale=3.0,
        terminal_projection="reflect",
        antithetic=True,
    ) -> torch.Tensor:
        return torch.zeros_like(positions)

    mnist_fix.estimate_reparameterized_guided_drift = _zero_drift
    try:
        generated = generate_guided_point_clouds_reparam(
            model,
            mass_bank,
            target_labels,
            bank_labels=bank_labels,
            class_conditional_mass_sampling=True,
            horizon=0.01,
            step_size=0.01,
            terminal_mc_samples=4,
            guidance_scale=1.0,
            initial_position_mode="centered_gaussian",
            initial_position_scale=5.0,
            state_projection="clip",
            terminal_projection="reflect",
            diffusion_temperature=1e-12,
            drift_clip_norm=None,
            batch_size=2,
            return_trajectories=True,
            rasterize=False,
            device="cpu",
            rng=np.random.default_rng(321),
        )
    finally:
        mnist_fix.estimate_reparameterized_guided_drift = original

    assert generated.trajectories is not None
    initial_positions = generated.trajectories[0]
    assert np.all(initial_positions >= 0.0)
    assert np.all(initial_positions <= 1.0)


def _regression_test_horizon_aware_drift_clip_total_displacement() -> None:
    torch.manual_seed(0)
    mass_bank, bank_labels = _make_toy_mass_bank()
    target_labels = np.asarray([0, 1], dtype=np.int64)
    model = TerminalSetClassifier(point_feature_dim=8, hidden_dim=16, num_classes=2)

    original = mnist_fix.estimate_reparameterized_guided_drift

    @torch.enable_grad()
    def _huge_drift(
        model,
        masses,
        positions,
        labels,
        tau,
        *,
        terminal_mc_samples=128,
        guidance_scale=3.0,
        terminal_projection="reflect",
        antithetic=True,
    ) -> torch.Tensor:
        drift = torch.zeros_like(positions)
        drift[..., 0] = 1000.0
        return drift

    mnist_fix.estimate_reparameterized_guided_drift = _huge_drift
    try:
        generated = generate_guided_point_clouds_reparam(
            model,
            mass_bank,
            target_labels,
            bank_labels=bank_labels,
            class_conditional_mass_sampling=True,
            horizon=0.2,
            step_size=0.2,
            terminal_mc_samples=4,
            guidance_scale=1.0,
            initial_position_mode="centered_gaussian",
            initial_position_scale=0.0,
            state_projection="none",
            terminal_projection="reflect",
            diffusion_temperature=1e-12,
            drift_clip_norm=None,
            drift_clip_total_displacement=0.1,
            batch_size=2,
            return_trajectories=True,
            rasterize=False,
            device="cpu",
            rng=np.random.default_rng(456),
        )
    finally:
        mnist_fix.estimate_reparameterized_guided_drift = original

    assert generated.trajectories is not None
    displacement = generated.trajectories[-1][..., 0] - generated.trajectories[0][..., 0]
    assert np.allclose(displacement, 0.1, atol=2e-3), displacement




def _regression_test_truncated_poisson_dirichlet_masses_are_valid() -> None:
    masses = sample_truncated_poisson_dirichlet_masses(
        16,
        10,
        beta=20.0,
        max_terms=256,
        rng=np.random.default_rng(0),
    )
    assert masses.shape == (16, 10)
    assert np.all(masses >= 0.0)
    assert np.allclose(masses.sum(axis=1), 1.0)
    assert np.all(masses[:, :-1] >= masses[:, 1:] - 1e-12)


def _regression_test_reparam_defaults_support_pd_uniform_start() -> None:
    torch.manual_seed(0)
    target_labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    model = TerminalSetClassifier(point_feature_dim=8, hidden_dim=16, num_classes=2)

    original = mnist_fix.estimate_reparameterized_guided_drift

    @torch.enable_grad()
    def _zero_drift(
        model,
        masses,
        positions,
        labels,
        tau,
        *,
        terminal_mc_samples=128,
        guidance_scale=3.0,
        terminal_projection="reflect",
        antithetic=True,
    ) -> torch.Tensor:
        return torch.zeros_like(positions)

    mnist_fix.estimate_reparameterized_guided_drift = _zero_drift
    try:
        generated = generate_balanced_synthetic_dataset_reparam(
            model,
            None,
            num_points=2,
            num_per_class=2,
            horizon=0.01,
            step_size=0.01,
            terminal_mc_samples=4,
            guidance_scale=1.0,
            diffusion_temperature=1e-12,
            drift_clip_norm=None,
            rasterize=False,
            device="cpu",
            rng=np.random.default_rng(123),
        )
    finally:
        mnist_fix.estimate_reparameterized_guided_drift = original

    assert generated.positions.shape == (4, 2, 2)
    assert np.allclose(generated.masses.sum(axis=1), 1.0)
    assert np.all(generated.masses[:, :-1] >= generated.masses[:, 1:] - 1e-12)
    assert np.all(generated.positions >= 0.0)
    assert np.all(generated.positions <= 1.0)


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)

    images, labels = _make_toy_images()
    point_clouds = images_to_weighted_point_clouds(images, labels=labels, top_k=10, mass_floor=1e-4)

    # Rasterization sanity check.
    recon = rasterize_weighted_point_clouds(point_clouds.masses[:2], point_clouds.positions[:2])
    assert recon.shape == (2, 28, 28)
    assert np.allclose(recon.reshape(2, -1).sum(axis=1), 1.0)

    train_idx = np.arange(0, len(labels), 2)
    val_idx = np.arange(1, len(labels), 2)

    model = TerminalSetClassifier(point_feature_dim=32, hidden_dim=64, num_classes=2)
    history = train_terminal_set_classifier(
        model,
        point_clouds.masses[train_idx],
        point_clouds.positions[train_idx],
        labels[train_idx],
        val_masses=point_clouds.masses[val_idx],
        val_positions=point_clouds.positions[val_idx],
        val_labels=labels[val_idx],
        epochs=2,
        batch_size=8,
        lr=5e-3,
        position_jitter_std=1e-3,
        device="cpu",
        verbose=False,
    )
    assert len(history["train_loss"]) == 2

    g_metrics = terminal_g_accuracy(
        model,
        point_clouds.masses[val_idx],
        point_clouds.positions[val_idx],
        labels[val_idx],
        device="cpu",
    )
    assert 0.0 <= g_metrics["accuracy"] <= 1.0
    assert 0.0 <= g_metrics["mean_target_probability"] <= 1.0

    _regression_test_generation_updates_every_batch()
    _regression_test_generation_respects_mc_kwargs()
    _regression_test_joint_bank_sampling_keeps_mass_position_pairing()
    _regression_test_reparam_initial_positions_are_projected()
    _regression_test_horizon_aware_drift_clip_total_displacement()
    _regression_test_truncated_poisson_dirichlet_masses_are_valid()
    _regression_test_reparam_defaults_support_pd_uniform_start()

    generated = generate_balanced_synthetic_dataset(
        model,
        point_clouds.masses[train_idx],
        bank_labels=labels[train_idx],
        num_per_class=2,
        class_conditional_mass_sampling=True,
        horizon=0.01,
        step_size=0.005,
        terminal_mc_samples=4,
        guidance_scale=1.0,
        initial_position_scale=0.05,
        batch_size=2,
        device="cpu",
        rng=np.random.default_rng(123),
    )
    assert generated.positions.shape == (4, 10, 2)
    assert generated.images is not None and generated.images.shape == (4, 28, 28)

    # Use the tiny toy set as both reference and test data for a smoke test only.
    metrics = evaluate_generation_metrics(
        model,
        generated,
        point_clouds,
        images,
        labels,
        cas_epochs=1,
        cas_batch_size=8,
        cas_lr=1e-3,
        sinkhorn_epsilon=0.05,
        sinkhorn_iterations=10,
        sinkhorn_subsample_per_class=2,
        device="cpu",
        rng=np.random.default_rng(0),
        verbose=False,
    )
    assert 0.0 <= metrics["g_accuracy"] <= 1.0
    assert 0.0 <= metrics["cas_accuracy"] <= 1.0
    assert 0.0 <= metrics["one_nn_accuracy_macro"] <= 1.0
    assert 0.0 <= metrics["coverage_macro"] <= 1.0

    print("MNIST weighted point-cloud smoke test passed.")


def _test_confusion_matrix_helper() -> None:
    true = np.asarray([0, 0, 1, 1, 1], dtype=np.int64)
    pred = np.asarray([0, 1, 1, 0, 1], dtype=np.int64)
    counts = confusion_matrix_from_predictions(true, pred, num_classes=2)
    normalized = confusion_matrix_from_predictions(true, pred, num_classes=2, normalize="true")
    assert counts.shape == (2, 2)
    assert np.array_equal(counts, np.asarray([[1.0, 1.0], [1.0, 2.0]]))
    assert np.allclose(normalized.sum(axis=1), 1.0)


if __name__ == "__main__":
    _test_confusion_matrix_helper()
    main()
