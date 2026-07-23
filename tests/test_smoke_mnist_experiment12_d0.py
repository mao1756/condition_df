"""Smoke checks for standalone Experiment 12 D0."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, DirectFluxUNet, masked_reference_free_step_torch
from mnist.experiment12_d0 import (
    Experiment12D0Config,
    build_d0_training_cache,
    cache_summary,
    d0_learned_block_reverse_diagnostic,
    d0_learned_rollout_diagnostic,
    d0_realized_target_replay_diagnostic,
    d0_unweighted_innovation_loss,
    effective_time_integral,
    make_rate_schedule,
    load_d0_cache_npz,
    sample_d0_cache_batch,
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
        return_realized_transfers=True,
    )
    assert result.raw_innovations is not None and result.raw_innovations.shape == (3, 2, 2, 8, 8)
    assert result.valid_edge_mask is not None and result.valid_edge_mask.shape == (3, 2, 2, 8, 8)
    assert result.realized_edge_transfers is not None and result.realized_edge_transfers.shape == (3, 2, 2, 8, 8)
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
    assert cache.earlier_states.shape == (8, 64)
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
    cache_path = tmp_path / "cache.npz"
    save_d0_cache_npz(cache, cache_path)
    loaded_cache = load_d0_cache_npz(cache_path, require_complete=True)
    assert torch.equal(loaded_cache.states, cache.states)
    assert torch.equal(loaded_cache.path_indices, cache.path_indices)
    assert loaded_cache.raw_limited_fraction == pytest.approx(cache.raw_limited_fraction)
    assert loaded_cache.floor_touched_pixels == cache.floor_touched_pixels
    np.savez_compressed(
        prior_path,
        terminal_states=cache.terminal_states.reshape(cache.terminal_states.shape[0], -1),
        labels=cache.requested_labels,
        rate_schedule=cache.rate_schedule,
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
        stride_substeps=np.asarray([cache.stride_substeps], dtype=np.int64),
        lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
        physical_target_scale=np.asarray([cache.physical_target_scale], dtype=np.float64),
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


def test_d0_overfit_outer_stride_rounding_requires_opt_in() -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=123)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=2,
        cache_batch_size=2,
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=1,
        tau_eff=1e-4,
        single_image_overfit=True,
        train_steps=0,
        num_samples=0,
    )
    with pytest.raises(ValueError, match="rounded teacher_stride_substeps"):
        build_d0_training_cache(
            dataset_images=images,
            dataset_labels=labels,
            dynamics_config=cfg,
            d0_config=d0,
            device=torch.device("cpu"),
            rng=np.random.default_rng(5),
            show_progress=False,
        )


def test_d0_prior_bank_mismatch_raises(tmp_path) -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=123)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=2,
        cache_batch_size=2,
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        train_steps=0,
        num_samples=2,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(6),
        show_progress=False,
    )
    bad_prior = tmp_path / "bad_prior.npz"
    np.savez_compressed(
        bad_prior,
        terminal_states=cache.terminal_states.reshape(cache.terminal_states.shape[0], -1),
        labels=cache.requested_labels,
        rate_schedule=cache.rate_schedule,
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([1], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
    )
    model = DirectFluxUNet(cfg, base_channels=4)
    with pytest.raises(ValueError, match="Incompatible D0 prior bank"):
        simulate_d0_reverse_generation(
            model,
            cache.requested_labels[:2],
            dynamics_config=cfg,
            d0_config=d0,
            prior_bank_path=bad_prior,
            device=torch.device("cpu"),
            seed=9,
            deterministic=True,
            control_strength=0.0,
            show_progress=False,
        )



def test_d0_projected_loss_reports_projection_diagnostics() -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=222)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=4,
        cache_batch_size=2,
        time_slices_per_path=2,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        d0_target_space="projected-edge",
        invalid_output_l2_weight=1e-3,
        curl_loss_weight=1e-3,
        train_steps=0,
        num_samples=0,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(222),
        show_progress=False,
    )
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
    assert diag["d0_target_space"] == "projected-edge"
    assert "projected_residual_rms" in diag
    assert "prediction_removed_curl_rms" in diag
    assert "invalid_output_l2" in diag
    summary = cache_summary(cache)
    assert sum(int(summary[f"cache_tau_bin{i}_count"]) for i in range(5)) == cache.size


def test_d0_learned_block_diagnostic_smoke() -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=333)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=4,
        cache_batch_size=2,
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        d0_target_space="projected-edge",
        sample_project_learned_mean=True,
        train_steps=0,
        num_samples=0,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(333),
        show_progress=False,
    )
    model = DirectFluxUNet(cfg, base_channels=4)
    diag = d0_learned_block_reverse_diagnostic(model, cache, cfg, d0, max_slices=2, device=torch.device("cpu"))
    assert diag["learned_block_slices"] == 2
    assert diag["learned_block_best"] in {"plus", "minus", "zero"}
    assert "learned_block_pred_projected_rms" in diag



def test_d0_physical_edge_loss_and_rollout_diagnostic_smoke() -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=444)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=4,
        cache_batch_size=2,
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        d0_target_space="physical-edge",
        edge_innovation_loss_weight=0.25,
        state_delta_loss_weight=1.0,
        rollout_loss_weight=0.25,
        rollout_loss_warmup_steps=0,
        rollout_loss_blocks=2,
        rollout_loss_batch_size=2,
        train_steps=0,
        num_samples=0,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(444),
        show_progress=False,
    )
    model = DirectFluxUNet(cfg, base_channels=4)
    batch = {
        "states": cache.states[:4],
        "tau": cache.tau[:4],
        "labels": cache.labels[:4],
        "innovations": cache.innovations[:4],
        "masks": cache.masks[:4],
        "earlier_states": cache.earlier_states[:4],
        "starts": cache.starts[:4],
    }
    loss, diag = d0_unweighted_innovation_loss(model, batch, cfg, d0, step=0)
    assert torch.isfinite(loss)
    assert diag["d0_target_space"] == "physical-edge"
    assert "state_delta_loss" in diag
    assert diag["rollout_loss_active"] == 1
    rollout = d0_learned_rollout_diagnostic(
        model,
        cache,
        cfg,
        d0,
        max_slices=2,
        block_counts=[1, 2],
        device=torch.device("cpu"),
    )
    assert rollout["learned_rollout_slices"] == 2
    assert "learned_rollout_depth1_l1_to_start" in rollout
    assert "learned_rollout_depth2_corr_to_start" in rollout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the cross-device rollout regression")
def test_d0_learned_rollout_diagnostic_keeps_indices_on_cuda() -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=445)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=4,
        cache_batch_size=2,
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        d0_target_space="realized-physical",
        train_steps=0,
        num_samples=0,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(445),
        show_progress=False,
    )
    device = torch.device("cuda")
    model = DirectFluxUNet(cfg, base_channels=4).to(device)
    rollout = d0_learned_rollout_diagnostic(
        model,
        cache,
        cfg,
        d0,
        max_slices=2,
        block_counts=[1, 2],
        device=device,
    )
    assert rollout["learned_rollout_slices"] == 2
    assert "learned_rollout_depth2_corr_to_start" in rollout


def test_d0_physical_total_and_residual_semantics_are_distinct(tmp_path) -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=555)
    cfg = _toy_config()
    base_kwargs = dict(
        cache_paths=4,
        cache_batch_size=2,
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        train_steps=0,
        num_samples=2,
    )
    d0_total = Experiment12D0Config(
        **base_kwargs,
        d0_target_space="physical-total",
        physical_sampler_noise_mode="none",
        edge_innovation_loss_weight=1.0,
        state_delta_loss_weight=1.0,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0_total,
        device=torch.device("cpu"),
        rng=np.random.default_rng(555),
        show_progress=False,
    )
    model = DirectFluxUNet(cfg, base_channels=4)
    batch = {
        "states": cache.states[:4],
        "tau": cache.tau[:4],
        "labels": cache.labels[:4],
        "innovations": cache.innovations[:4],
        "masks": cache.masks[:4],
        "earlier_states": cache.earlier_states[:4],
        "starts": cache.starts[:4],
        "stride_substeps": torch.full((4,), cache.stride_substeps, dtype=torch.long),
        "reference_substeps": torch.full((4,), cache.reference_substeps, dtype=torch.long),
        "dt_sub": torch.full((4,), cache.dt_sub, dtype=torch.float32),
        "rate_schedule": torch.as_tensor(cache.rate_schedule, dtype=torch.float32),
    }
    loss_total, diag_total = d0_unweighted_innovation_loss(model, batch, cfg, d0_total)
    assert torch.isfinite(loss_total)
    assert diag_total["d0_target_space_normalized"] == "physical-total"

    d0_residual = Experiment12D0Config(
        **base_kwargs,
        d0_target_space="physical-residual",
        physical_sampler_noise_mode="none",
        edge_innovation_loss_weight=1.0,
        state_delta_loss_weight=1.0,
    )
    loss_residual, diag_residual = d0_unweighted_innovation_loss(model, batch, cfg, d0_residual)
    assert torch.isfinite(loss_residual)
    assert diag_residual["d0_target_space_normalized"] == "physical-residual"

    prior_path = tmp_path / "prior_bank.npz"
    np.savez_compressed(
        prior_path,
        terminal_states=cache.terminal_states.reshape(cache.terminal_states.shape[0], -1),
        labels=cache.requested_labels,
        rate_schedule=cache.rate_schedule,
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
        stride_substeps=np.asarray([cache.stride_substeps], dtype=np.int64),
        lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
    )
    for param in model.parameters():
        param.data.zero_()
    total_sample = simulate_d0_reverse_generation(
        model,
        cache.requested_labels[:2],
        dynamics_config=cfg,
        d0_config=d0_total,
        prior_bank_path=prior_path,
        device=torch.device("cpu"),
        seed=556,
        deterministic=True,
        control_strength=0.0,
        show_progress=False,
    )
    residual_sample = simulate_d0_reverse_generation(
        model,
        cache.requested_labels[:2],
        dynamics_config=cfg,
        d0_config=d0_residual,
        prior_bank_path=prior_path,
        device=torch.device("cpu"),
        seed=557,
        deterministic=True,
        control_strength=0.0,
        show_progress=False,
    )
    assert total_sample.physical_sampler_mode == "physical-total"
    assert residual_sample.physical_sampler_mode == "physical-residual"
    assert total_sample.physical_sampler_noise_mode == "none"
    assert np.allclose(total_sample.samples.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(residual_sample.samples.sum(axis=1), 1.0, atol=1e-5)

    diag = d0_learned_block_reverse_diagnostic(model, cache, cfg, d0_total, max_slices=2, device=torch.device("cpu"))
    assert diag["learned_block_sampler_semantics"] == "physical-total"
    assert "learned_block_free_l1" in diag



def test_d0_realized_physical_target_smoke(tmp_path) -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=777)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=4,
        cache_batch_size=2,
        cache_build_mode="substep",
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        d0_target_space="realized-physical",
        physical_target_normalization="global-rms",
        physical_loss_mask="all",
        physical_sampler_noise_mode="none",
        edge_innovation_loss_weight=1.0,
        state_delta_loss_weight=1.0,
        train_steps=0,
        num_samples=2,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(777),
        show_progress=False,
    )
    assert cache.physical_transfers.shape == cache.innovations.shape
    assert torch.isfinite(cache.physical_transfers).all()
    replay = d0_realized_target_replay_diagnostic(cache, cfg, max_slices=2, device=torch.device("cpu"))
    assert replay["target_replay_slices"] == 2
    assert "target_replay_realized_l1" in replay
    assert "target_replay_poisson_l1" in replay

    model = DirectFluxUNet(cfg, base_channels=4)
    batch = {
        "states": cache.states[:4],
        "tau": cache.tau[:4],
        "labels": cache.labels[:4],
        "innovations": cache.innovations[:4],
        "masks": cache.masks[:4],
        "earlier_states": cache.earlier_states[:4],
        "physical_transfers": cache.physical_transfers[:4],
        "starts": cache.starts[:4],
        "stride_substeps": torch.full((4,), cache.stride_substeps, dtype=torch.long),
        "reference_substeps": torch.full((4,), cache.reference_substeps, dtype=torch.long),
        "dt_sub": torch.full((4,), cache.dt_sub, dtype=torch.float32),
        "rate_schedule": torch.as_tensor(cache.rate_schedule, dtype=torch.float32),
    }
    loss, diag = d0_unweighted_innovation_loss(model, batch, cfg, d0)
    assert torch.isfinite(loss)
    assert diag["d0_target_space_normalized"] == "realized-physical"
    assert diag["physical_target_source"] == "realized"
    assert diag["physical_target_normalization"] == "global-rms"
    assert diag["physical_loss_mask"] == "all"
    assert diag["physical_target_scale"] > 0.0

    # Realized physical mode trains a normalized transfer.  Even with an all-false
    # innovation mask, the physical all-edge mask should keep the loss finite.
    batch_all_edges = dict(batch)
    batch_all_edges["masks"] = torch.zeros_like(batch["masks"], dtype=torch.bool)
    loss_all, diag_all = d0_unweighted_innovation_loss(model, batch_all_edges, cfg, d0)
    assert torch.isfinite(loss_all)
    assert diag_all["mask_fraction"] == 0.0
    assert diag_all["physical_loss_mask"] == "all"

    prior = tmp_path / "prior_bank.npz"
    np.savez_compressed(
        prior,
        terminal_states=cache.terminal_states.reshape(cache.terminal_states.shape[0], -1),
        labels=cache.requested_labels,
        rate_schedule=cache.rate_schedule,
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
        stride_substeps=np.asarray([cache.stride_substeps], dtype=np.int64),
        lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
    )
    for param in model.parameters():
        param.data.zero_()
    sample = simulate_d0_reverse_generation(
        model,
        cache.requested_labels[:2],
        dynamics_config=cfg,
        d0_config=d0,
        prior_bank_path=prior,
        device=torch.device("cpu"),
        seed=778,
        deterministic=True,
        control_strength=0.0,
        show_progress=False,
    )
    assert sample.physical_sampler_mode == "realized-physical"
    assert sample.physical_sampler_noise_mode == "none"
    assert np.allclose(sample.samples.sum(axis=1), 1.0, atol=1e-5)


def test_d0_trajectory_window_cache_and_supervised_rollout_loss() -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=901)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=4,
        cache_batch_size=2,
        cache_build_mode="substep",
        time_slices_per_path=2,
        sample_steps=4,
        reference_substeps=2,
        teacher_stride_substeps=1,
        tau_eff=1e-4,
        d0_target_space="realized-physical",
        physical_target_normalization="global-rms",
        physical_loss_mask="all",
        cache_store_trajectory_windows=True,
        rollout_target_blocks="1,2,4",
        trajectory_rollout_loss_weight=0.25,
        trajectory_rollout_warmup_steps=0,
        trajectory_rollout_depths="1,2",
        trajectory_rollout_batch_size=4,
        train_steps=0,
        num_samples=0,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(901),
        show_progress=False,
    )
    assert cache.trajectory_window_states is not None
    assert cache.trajectory_window_valid is not None
    assert cache.trajectory_window_depths is not None
    assert cache.trajectory_window_states.shape == (cache.size, 3, 64)
    assert cache.trajectory_window_valid.shape == (cache.size, 3)
    assert cache.trajectory_window_depths.tolist() == [1, 2, 4]
    assert torch.all(cache.trajectory_window_valid[:, 0])
    assert torch.allclose(cache.trajectory_window_states[:, 0], cache.earlier_states)

    model = DirectFluxUNet(cfg, base_channels=4)
    batch = sample_d0_cache_batch(
        cache,
        4,
        device=torch.device("cpu"),
        rng=np.random.default_rng(902),
    )
    loss, diag = d0_unweighted_innovation_loss(model, batch, cfg, d0, step=1)
    assert torch.isfinite(loss)
    assert diag["trajectory_rollout_active"] == 1
    assert diag["trajectory_rollout_loss"] >= 0.0
    assert diag["trajectory_rollout_depth1_valid_fraction"] == 1.0
    assert "trajectory_rollout_depth2_corr" in diag
    summary = cache_summary(cache)
    assert summary["cache_trajectory_windows"] == 1
    assert summary["cache_trajectory_window_depths"] == "1,2,4"


def test_d0_trajectory_windows_require_exact_substep_cache() -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=903)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=2,
        cache_batch_size=2,
        cache_build_mode="outer",
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=2,
        tau_eff=1e-4,
        cache_store_trajectory_windows=True,
        train_steps=0,
        num_samples=0,
    )
    with pytest.raises(ValueError, match="requires --cache-build-mode substep"):
        build_d0_training_cache(
            dataset_images=images,
            dataset_labels=labels,
            dynamics_config=cfg,
            d0_config=d0,
            device=torch.device("cpu"),
            rng=np.random.default_rng(903),
            show_progress=False,
        )


def test_d0_reverse_sampler_can_stop_after_requested_blocks(tmp_path) -> None:
    images, labels = synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=904)
    cfg = _toy_config()
    d0 = Experiment12D0Config(
        cache_paths=2,
        cache_batch_size=2,
        cache_build_mode="substep",
        time_slices_per_path=1,
        sample_steps=2,
        reference_substeps=2,
        teacher_stride_substeps=1,
        tau_eff=1e-4,
        d0_target_space="realized-physical",
        physical_target_normalization="global-rms",
        train_steps=0,
        num_samples=2,
    )
    cache = build_d0_training_cache(
        dataset_images=images,
        dataset_labels=labels,
        dynamics_config=cfg,
        d0_config=d0,
        device=torch.device("cpu"),
        rng=np.random.default_rng(904),
        show_progress=False,
    )
    prior = tmp_path / "prior.npz"
    np.savez_compressed(
        prior,
        terminal_states=cache.terminal_states.reshape(cache.terminal_states.shape[0], -1),
        labels=cache.requested_labels,
        rate_schedule=cache.rate_schedule,
        sample_steps=np.asarray([cache.sample_steps], dtype=np.int64),
        substeps=np.asarray([cache.reference_substeps], dtype=np.int64),
        horizon=np.asarray([cache.horizon], dtype=np.float64),
        stride_substeps=np.asarray([cache.stride_substeps], dtype=np.int64),
        lambda_mix=np.asarray([cache.lambda_mix], dtype=np.float64),
        physical_target_scale=np.asarray([cache.physical_target_scale], dtype=np.float64),
    )
    model = DirectFluxUNet(cfg, base_channels=4)
    for parameter in model.parameters():
        parameter.data.zero_()
    result = simulate_d0_reverse_generation(
        model,
        cache.requested_labels[:2],
        dynamics_config=cfg,
        d0_config=d0,
        prior_bank_path=prior,
        device=torch.device("cpu"),
        seed=905,
        deterministic=True,
        control_strength=0.0,
        show_progress=False,
        max_reverse_blocks=2,
    )
    assert result.reverse_blocks_requested == 2
    assert result.reverse_blocks_executed == 2
