"""Smoke checks for standalone Experiment 12 D0."""

from __future__ import annotations

import numpy as np
import torch

from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, DirectFluxUNet, masked_reference_free_step_torch
from mnist.experiment12_d0 import (
    Experiment12D0Config,
    build_d0_training_cache,
    d0_unweighted_innovation_loss,
    effective_time_integral,
    make_rate_schedule,
    save_d0_cache_npz,
    simulate_d0_reverse_generation,
    synthetic_digit_measures,
)


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _toy_config() -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=8,
        num_steps=2,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        source_uniform_mix=0.10,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        ot_com_weight=0.10,
        condition_on_source=False,
        flux_parameterization="edge",
        limiter_fraction=0.5,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-8,
    )


def test_masked_reference_step_can_return_substep_states() -> None:
    cfg = _toy_config()
    state = torch.full((2, 64), 1.0 / 64.0)
    result = masked_reference_free_step_torch(
        state,
        1e-4,
        cfg,
        free_weight=1.0,
        noise_weight=0.0,
        substeps=3,
        deterministic=True,
        return_innovations=True,
        return_substep_states=True,
    )
    assert result.raw_innovations is not None and result.raw_innovations.shape == (3, 2, 2, 8, 8)
    assert result.valid_edge_mask is not None and result.valid_edge_mask.shape == (3, 2, 2, 8, 8)
    assert result.substep_states is not None and result.substep_states.shape == (3, 2, 64)
    assert torch.allclose(result.substep_states[-1], result.states)


def test_d0_cache_loss_and_reverse_smoke(tmp_path) -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=123)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=4,
        cache_batch_size=2,
        time_slices_per_path=2,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        lambda_mix=0.25,
        batch_size=4,
        base_channels=4,
        train_steps=0,
        num_samples=4,
        seed=7,
    )
    rng = np.random.default_rng(5)
    device = torch.device("cpu")
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=device,
        rng=rng,
        show_progress=False,
    )
    assert cache.states.shape == (8, 64)
    assert cache.tau.shape == (8,)
    assert cache.labels.shape == (8,)
    assert cache.innovations.shape == (8, 2, 8, 8)
    assert cache.masks.shape == (8, 2, 8, 8)
    assert cache.valid_innovation_fraction > 0.0
    assert np.isclose(effective_time_integral(cache.rate_schedule, horizon=cache.horizon), d0.tau_eff)

    model = DirectFluxUNet(cfg, base_channels=4)
    batch = {
        "states": cache.states[:4],
        "tau": cache.tau[:4],
        "labels": cache.labels[:4],
        "innovations": cache.innovations[:4],
        "masks": cache.masks[:4],
    }
    loss, diag = d0_unweighted_innovation_loss(model, batch, cfg, d0)
    assert torch.isfinite(loss)
    assert diag["mask_fraction"] > 0.0
    assert diag["batch_ess_fraction"] == 1.0

    prior_path = tmp_path / "prior_bank.npz"
    save_d0_cache_npz(cache, tmp_path / "cache.npz")
    np.savez_compressed(
        prior_path,
        terminal_states=cache.terminal_states.reshape(cache.terminal_states.shape[0], -1),
        labels=cache.requested_labels,
        rate_schedule=cache.rate_schedule,
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
    )
    for param in model.parameters():
        param.data.zero_()
    generated = simulate_d0_reverse_generation(
        model,
        np.arange(4) % 10,
        dynamics_config=cfg,
        d0_config=d0,
        prior_bank_path=prior_path,
        device=device,
        seed=9,
        deterministic=True,
        control_strength=0.0,
        show_progress=False,
    )
    assert generated.samples.shape == (4, 64)
    assert np.allclose(generated.samples.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(generated.samples >= -1e-7)


def test_d0_schedule_rate_mode() -> None:
    rates = make_rate_schedule(4, tau_eff=0.5, horizon=2.0, time_change_mode="integral")
    assert np.isclose(rates.mean(), 0.25)
    rates_legacy = make_rate_schedule(4, tau_eff=0.5, horizon=2.0, time_change_mode="rate")
    assert np.isclose(rates_legacy.mean(), 0.5)
