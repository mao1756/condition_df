from __future__ import annotations

import numpy as np
import torch

from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, build_classwise_ot_cache
from mnist.experiment11_c3_value import (
    Experiment11C3Config,
    ValuePotentialUNet,
    build_c3_value_cache,
    parse_args,
    simulate_value_generation,
    value_loss,
)
from mnist.weighted_point_cloud import normalize_images_to_measures


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _toy_digit_measures(num_samples: int = 30, grid_size: int = 8) -> tuple[np.ndarray, np.ndarray]:
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
        ot_match_mode="topk",
        ot_nearest_top_k=2,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        ot_com_weight=0.10,
        condition_on_source=True,
        flux_parameterization="edge",
    )


def test_experiment11_c3_value_cache_loss_generation_smoke() -> None:
    images, labels = _toy_digit_measures()
    config = _toy_config()
    c3 = Experiment11C3Config(
        cache_paths=3,
        cache_batch_size=3,
        branch_count=2,
        branch_batch_size=6,
        endpoint_count_per_state=2,
        sample_steps=3,
        batch_size=3,
        base_channels=4,
        terminal_epsilon=0.0,
        terminal_ess_target=0.5,
        terminal_feature_mode="lowres",
        reference_free_weight=0.01,
        reference_noise_weight=0.002,
        num_samples=3,
        sample_value_flux_scales="1",
        use_amp=False,
        ema_decay=0.0,
    )
    device = torch.device("cpu")
    rng = np.random.default_rng(123)
    ot_cache = build_classwise_ot_cache(images, labels, config)
    cache = build_c3_value_cache(
        dataset_images=images,
        dataset_labels=labels,
        ot_cache=ot_cache,
        dynamics_config=config,
        c3_config=c3,
        device=device,
        rng=rng,
        show_progress=False,
    )
    assert cache.states.shape == (3, 64)
    assert cache.value_targets.shape == (3,)
    assert cache.branch_ess_fraction.shape == (3,)
    assert np.all(cache.branch_ess_fraction > 0.0)
    assert cache.terminal_states.shape == (3, 8, 8)

    model = ValuePotentialUNet(config, base_channels=4)
    batch = {
        "states": cache.states,
        "tau": cache.tau,
        "labels": cache.labels,
        "sources": cache.sources,
        "value_targets": cache.value_targets,
        "log_weights": cache.log_weights,
    }
    loss, diag = value_loss(model, batch, c3)
    assert torch.isfinite(loss)
    assert "value_corr" in diag

    result = simulate_value_generation(
        model,
        np.arange(3) % 10,
        dynamics_config=config,
        c3_config=c3,
        device=device,
        source_images=images,
        source_labels=labels,
        seed=321,
        deterministic=True,
        show_progress=False,
    )
    samples = np.asarray(result["samples"])
    assert samples.shape == (3, 64)
    assert np.allclose(samples.sum(axis=1), 1.0, atol=1e-5)


def test_experiment11_c3_value_center_by_label_flag_alias() -> None:
    args = parse_args(["--value-center-by-label"])
    assert args.value_center_by_label is True
    args = parse_args(["--value-center-by-label", "--no-value-center-by-label"])
    assert args.value_center_by_label is False
