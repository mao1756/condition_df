"""Smoke test for the Example-6 hyperparameter-search helper."""

from __future__ import annotations

import numpy as np

from mnist.weighted_point_cloud import (
    images_to_weighted_point_clouds,
    normalize_images_to_measures,
)
from mnist.experiment6_hyperparameter_search import (
    Experiment6SearchConfig,
    PreparedMnistExperiment6Data,
    run_experiment6_random_search,
)


def _make_toy_images(num_per_class: int = 10, image_size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    images = []
    labels = []
    xs = np.arange(image_size)
    for label in [0, 1]:
        for _ in range(num_per_class):
            image = np.zeros((image_size, image_size), dtype=np.float64)
            jitter = rng.normal(scale=0.7, size=image_size)
            if label == 0:
                ys = np.clip((0.22 * image_size + 0.16 * xs + jitter).astype(int), 0, image_size - 1)
            else:
                ys = np.clip((0.78 * image_size - 0.16 * xs + jitter).astype(int), 0, image_size - 1)
            image[ys, xs] = 1.0
            image += 0.05 * rng.random((image_size, image_size))
            images.append(image)
            labels.append(label)
    images = normalize_images_to_measures(np.asarray(images, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int64)
    return images, labels


def _prepare_toy_data() -> PreparedMnistExperiment6Data:
    train_images, train_labels = _make_toy_images(num_per_class=12)
    val_images, val_labels = _make_toy_images(num_per_class=4)
    test_images, test_labels = _make_toy_images(num_per_class=4)
    train_pc = images_to_weighted_point_clouds(train_images, labels=train_labels, top_k=8, mass_floor=1e-4)
    val_pc = images_to_weighted_point_clouds(val_images, labels=val_labels, top_k=8, mass_floor=1e-4)
    test_pc = images_to_weighted_point_clouds(test_images, labels=test_labels, top_k=8, mass_floor=1e-4)
    return PreparedMnistExperiment6Data(
        train_pc=train_pc,
        val_pc=val_pc,
        test_pc=test_pc,
        real_test_images=test_images,
        real_test_labels=test_labels,
        top_k=8,
        mass_floor=1e-4,
    )


def main() -> None:
    prepared = _prepare_toy_data()
    config = Experiment6SearchConfig(
        seed=7,
        device="cpu",
        terminal_trials=1,
        keep_top_terminal=1,
        generation_trials_per_terminal=1,
        final_eval_top_k=1,
        synthetic_per_class_proxy=2,
        synthetic_per_class_final=2,
        terminal_noise_eval_repeats=1,
        cas_epochs_final=1,
        cas_batch_size=8,
        sinkhorn_subsample_per_class_final=2,
        sinkhorn_iterations_final=5,
        generation_batch_size=4,
        verbose=False,
    )
    terminal_space = {
        "point_feature_dim": [16],
        "hidden_dim": [32],
        "dropout": [0.0],
        "epochs": [1],
        "batch_size": [8],
        "lr": [1e-3],
        "weight_decay": [1e-5],
        "position_jitter_std": [1e-3],
        "max_tau_factor": [1.0],
        "tau_sampling": ["uniform"],
    }
    generation_space = {
        "horizon": [5e-5],
        "num_steps": [4],
        "terminal_mc_samples": [4],
        "guidance_scale": [2.0],
        "diffusion_temperature": [0.5],
        "drift_clip_pixels": [2.0],
        "poisson_dirichlet_beta_factor": [2.0],
        "poisson_dirichlet_max_terms_factor": [4],
        "mass_sampling_mode": "truncated_poisson_dirichlet",
        "class_conditional_mass_sampling": False,
        "initial_position_mode": "uniform",
        "initial_position_scale": 0.12,
        "initial_position_jitter": 0.0,
        "joint_bank_sampling": False,
        "state_projection": "reflect",
        "terminal_projection": "reflect",
        "num_points": 8,
    }
    result = run_experiment6_random_search(
        prepared_data=prepared,
        config=config,
        terminal_search_space=terminal_space,
        generation_search_space=generation_space,
        output_dir=None,
    )
    assert not result.terminal_trials.empty
    assert not result.generation_proxy_trials.empty
    assert not result.generation_final_trials.empty
    assert "g_accuracy" in result.best_generation_metrics
    assert int(result.best_generation_params["num_points"]) == 8
    print("test_smoke_mnist_experiment6_hyperparameter_search: OK")


if __name__ == "__main__":
    main()
