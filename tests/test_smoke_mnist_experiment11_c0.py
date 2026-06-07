"""Smoke checks for Experiment 11 C0 weighted innovation matching."""

from __future__ import annotations

import numpy as np
import torch

from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, DirectFluxUNet, build_classwise_ot_cache, natural_horizon
from mnist.experiment11_c0 import (
    Experiment11C0Config,
    build_c0_training_cache,
    c0_weighted_innovation_loss,
    make_experiment11_run_dir,
    simulate_c0_generation,
)
from mnist.weighted_point_cloud import normalize_images_to_measures


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _toy_digit_measures(num_samples: int = 20, grid_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    images = []
    labels = []
    for idx in range(num_samples):
        label = idx % 10
        cx = 0.20 + 0.06 * (label % 5)
        cy = 0.35 + 0.20 * (label // 5)
        x = (xx + 0.5) / float(grid_size)
        y = (yy + 0.5) / float(grid_size)
        blob = np.exp(-40.0 * ((x - cx) ** 2 + (y - cy) ** 2))
        images.append(blob)
        labels.append(label)
    return normalize_images_to_measures(np.asarray(images)), np.asarray(labels, dtype=np.int64)


def _toy_config() -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=8,
        num_steps=4,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        source_uniform_mix=0.10,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        ot_com_weight=0.10,
        condition_on_source=True,
        flux_parameterization="edge",
    )


def test_experiment11_run_dir_uses_timestamped_folder(tmp_path) -> None:
    out_dir, metadata = make_experiment11_run_dir(tmp_path / "runs" / "experiment11", "c0 smoke")
    assert out_dir.exists()
    assert out_dir.parent.name == "experiment11"
    assert "c0-smoke" in out_dir.name
    assert metadata["run_dir"] == str(out_dir)


def test_experiment11_cache_loss_and_generation_smoke() -> None:
    images, labels = _toy_digit_measures()
    config = _toy_config()
    c0 = Experiment11C0Config(
        cache_paths=4,
        cache_batch_size=2,
        time_slices_per_path=1,
        teacher_stride=2,
        sample_steps=4,
        num_samples=4,
        reference_free_weight=0.01,
        reference_noise_weight=0.002,
        terminal_epsilon=0.0,
        terminal_ess_target=0.5,
        base_channels=4,
        batch_size=4,
        proposal_mode="poisson-short",
        proposal_strength=0.2,
        proposal_eta_clip=2.0,
        hybrid_loss_weight=0.0,
    )
    rng = np.random.default_rng(123)
    device = torch.device("cpu")
    ot_cache = build_classwise_ot_cache(images, labels, config)
    cache = build_c0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        ot_cache=ot_cache,
        dynamics_config=config,
        c0_config=c0,
        device=device,
        rng=rng,
        show_progress=False,
    )
    assert cache.states.shape == (4, 64)
    assert cache.innovations.shape == (4, 2, 8, 8)
    assert cache.masks.float().mean() > 0.8
    assert cache.ess_fraction > 0.0
    assert cache.proposal_log_corrections is not None
    assert cache.proposal_log_corrections.shape == (4,)

    model = DirectFluxUNet(config, base_channels=4)
    batch = {
        "states": cache.states,
        "tau": cache.tau,
        "labels": cache.labels,
        "sources": cache.sources,
        "innovations": cache.innovations,
        "log_weights": cache.log_weights,
        "masks": cache.masks,
    }
    loss, diag = c0_weighted_innovation_loss(model, batch, config, c0)
    assert torch.isfinite(loss)
    assert diag["mask_fraction"] > 0.8
    assert natural_horizon(config) > 0.0

    result = simulate_c0_generation(
        model,
        np.arange(4) % 10,
        dynamics_config=config,
        c0_config=c0,
        device=device,
        seed=321,
        source_images=images,
        source_labels=labels,
        deterministic=True,
        show_progress=False,
    )
    samples = np.asarray(result["samples"])
    assert samples.shape == (4, 64)
    assert np.allclose(samples.sum(axis=1), 1.0, atol=1e-5)
    assert float(result["clipping_fraction"]) >= 0.0
