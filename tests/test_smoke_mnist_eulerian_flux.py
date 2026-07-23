"""Smoke checks for Example 10b MNIST direct Eulerian edge-flux generation."""

from __future__ import annotations

from contextlib import nullcontext

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
    TinyMNISTClassifier,
    EDGE_ALPHA_MODES,
    FLUX_PARAMETERIZATION_MODES,
    ON_POLICY_PREFIX_MODES,
    ON_POLICY_MODES,
    ON_POLICY_CACHE_MODES,
    ON_POLICY_TARGET_MODES,
    UPSAMPLE_MODES,
    SAMPLE_SELECTION_METRICS,
    CLASSIFIER_LOSS_MODES,
    FluxTrainingBatch,
    make_experiment10_run_dir,
    _ot_coupled_target_indices,
    masked_reference_free_step_torch,
    reference_step_substep_diagnostics_torch,
    choose_reference_substeps_torch,
    apply_flux_parameterization_torch,
    build_classwise_ot_cache,
    build_on_policy_replay_cache,
    checkerboard_energy_torch,
    direct_flux_matching_loss,
    direct_flux_rollout_consistency_loss,
    edge_alpha_value,
    free_drift_flux_torch,
    flux_curl_torch,
    flux_divergence_torch,
    image_total_variation,
    binary_cross_entropy_probs_autocast_safe,
    binary_cross_entropy_probs_per_sample_autocast_safe,
    make_on_policy_training_batch,
    poisson_flux_from_velocity_torch,
    sample_on_policy_replay_batch,
    save_diffusion_process_figure,
    nearest_class_mean_metrics,
    classifier_generation_metrics,
    compute_class_shape_statistics,
    compute_shape_statistics_np,
    local_shape_metrics_np,
    terminal_local_shape_loss_torch,
    write_goodbad_sample_report,
    select_generation_result_by_classifier,
    analyze_goodbad_annotations,
    step_component_rms_torch,
    sample_flux_training_batch,
    simulate_direct_flux_generation,
    simulate_teacher_flux_rollout,
    source_batch_diagnostics,
    terminal_conditioning_flux_torch,
    sample_terminal_flux_training_batch,
    _trajectory_snapshot_steps,
    natural_horizon,
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



def test_probability_bce_helper_is_autocast_safe_on_probabilities() -> None:
    pred = torch.tensor([[1e-6, 0.2, 0.8, 1.2]], dtype=torch.float16)
    target = torch.tensor([[0.0, 0.25, 0.75, 1.0]], dtype=torch.float16)
    context = torch.amp.autocast("cpu", enabled=True) if hasattr(torch, "amp") else nullcontext()
    with context:
        loss = binary_cross_entropy_probs_autocast_safe(pred, target)
    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32
    per_sample = binary_cross_entropy_probs_per_sample_autocast_safe(pred, target)
    assert per_sample.shape == (1,)
    assert torch.isfinite(per_sample).all()


def test_poisson_flux_divergence_matches_velocity() -> None:
    torch.manual_seed(0)
    velocity = torch.randn(3, 8, 8)
    velocity = velocity - velocity.mean(dim=(1, 2), keepdim=True)
    flux = poisson_flux_from_velocity_torch(velocity)
    div = flux_divergence_torch(flux)
    assert flux.shape == (3, 2, 8, 8)
    assert torch.allclose(div, velocity, atol=2e-5, rtol=2e-5)


def test_poisson_flux_half_precision_uses_float32_fft_on_28_grid() -> None:
    torch.manual_seed(1)
    velocity = torch.randn(2, 28, 28, dtype=torch.float16)
    velocity = (velocity.float() - velocity.float().mean(dim=(1, 2), keepdim=True)).to(torch.float16)
    velocity32 = velocity.float() - velocity.float().mean(dim=(1, 2), keepdim=True)
    flux = poisson_flux_from_velocity_torch(velocity)
    div = flux_divergence_torch(flux)
    assert flux.shape == (2, 2, 28, 28)
    assert flux.dtype == torch.float32
    assert torch.allclose(div, velocity32, atol=2e-4, rtol=2e-4)




def test_masked_reference_step_moves_mass_into_zero_endpoint() -> None:
    config = DirectFluxMNISTConfig(
        grid_size=4,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-12,
        limiter_fraction=1.0,
        source_lowfreq_size=2,
        ot_lowres_size=2,
    )
    state = torch.zeros(1, 16)
    # Put all mass at pixel 1.  Pixel 0 is the tail of the oriented horizontal
    # edge 0 -> 1 with a=0,b>0, so the free drift should move mass into pixel 0.
    state[0, 1] = 1.0
    result = masked_reference_free_step_torch(
        state,
        dt=1e-4,
        config=config,
        free_weight=1.0,
        noise_weight=0.0,
        substeps=1,
        stiffness_fraction=1.0,
        deterministic=True,
        return_innovations=True,
    )
    assert result.states[0, 1] < 1.0
    assert result.states[0].count_nonzero() > 1
    assert result.states[0, torch.arange(16) != 1].sum() > 0.0
    assert torch.all(result.states >= 0.0)
    assert torch.allclose(result.states.sum(dim=1), torch.ones(1), atol=1e-6)
    assert result.floor_correction_l1 == 0.0
    assert result.valid_edge_mask is not None
    assert result.raw_innovations is not None


def test_masked_reference_step_moves_mass_into_opposite_zero_endpoint() -> None:
    config = DirectFluxMNISTConfig(
        grid_size=4,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-12,
        limiter_fraction=1.0,
        source_lowfreq_size=2,
        ot_lowres_size=2,
    )
    state = torch.zeros(1, 16)
    # Now pixel 0 is the positive tail and pixel 1 is the zero head of edge
    # 0 -> 1.  The same boundary rule should move mass into pixel 1.
    state[0, 0] = 1.0
    result = masked_reference_free_step_torch(
        state,
        dt=1e-4,
        config=config,
        free_weight=1.0,
        noise_weight=0.0,
        substeps=1,
        stiffness_fraction=1.0,
        deterministic=True,
    )
    assert result.states[0, 0] < 1.0
    assert result.states[0, 1] > 0.0
    assert torch.all(result.states >= 0.0)
    assert torch.allclose(result.states.sum(dim=1), torch.ones(1), atol=1e-6)


def test_double_zero_edge_without_positive_neighbor_has_no_local_flux() -> None:
    config = DirectFluxMNISTConfig(
        grid_size=4,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-12,
        limiter_fraction=1.0,
        source_lowfreq_size=2,
        ot_lowres_size=2,
    )
    state = torch.zeros(1, 16)
    # Pixels 0 and 1 form a double-zero horizontal edge.  The only positive
    # mass is at pixel 10, which is not adjacent to either endpoint, so this
    # edge should not create mass locally in one deterministic step.
    state[0, 10] = 1.0
    result = masked_reference_free_step_torch(
        state,
        dt=1e-4,
        config=config,
        free_weight=1.0,
        noise_weight=0.0,
        substeps=1,
        stiffness_fraction=1.0,
        deterministic=True,
    )
    assert torch.allclose(result.states[0, [0, 1]], torch.zeros(2), atol=1e-10)
    assert torch.all(result.states >= 0.0)
    assert torch.allclose(result.states.sum(dim=1), torch.ones(1), atol=1e-6)


def test_reference_substep_diagnostics_and_adaptive_choice() -> None:
    config = DirectFluxMNISTConfig(
        grid_size=4,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-12,
        limiter_fraction=0.25,
        source_lowfreq_size=2,
        ot_lowres_size=2,
    )
    state = torch.full((1, 16), 1e-6)
    state[0, 1] = 1.0
    state = state / state.sum(dim=1, keepdim=True)
    diag = reference_step_substep_diagnostics_torch(
        state,
        dt=1e-3,
        config=config,
        free_weight=10.0,
        noise_weight=0.0,
        quantile=0.99,
    )
    chosen, adaptive = choose_reference_substeps_torch(
        state,
        dt=1e-3,
        config=config,
        free_weight=10.0,
        noise_weight=0.0,
        base_substeps=1,
        max_substeps=64,
        target_drift_ratio=0.05,
        target_noise_ratio=0.05,
        quantile=0.99,
    )
    assert diag["drift_ratio_q"] > 0.0
    assert 1 < chosen <= 64
    assert adaptive["required_substeps_unclipped"] >= chosen


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
    assert EDGE_ALPHA_MODES == ("legacy", "grid", "alpha_eff")
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



def test_anti_checkerboard_projected_flux_and_resize_conv_modes() -> None:
    torch.manual_seed(3)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=4,
        target_mode="poisson-flow",
        source_mode="lowfreq",
        source_lowfreq_size=4,
        upsample_mode="resize-conv",
        flux_parameterization="projected",
        velocity_target="mixed",
        late_residual_fraction=0.4,
        late_residual_prob=1.0,
        rollout_loss_weight=0.1,
        rollout_loss_steps=2,
        rollout_loss_batch_size=2,
        image_grad_loss_weight=0.0,
        rollout_image_grad_loss_weight=0.01,
        curl_loss_weight=0.01,
        checkerboard_loss_weight=0.001,
        flux_scale=10.0,
    )
    assert UPSAMPLE_MODES == ("transpose", "resize-conv")
    assert FLUX_PARAMETERIZATION_MODES == ("edge", "projected")
    assert "uniform" in ON_POLICY_PREFIX_MODES
    assert "replay" in ON_POLICY_MODES
    assert "trajectory" in ON_POLICY_CACHE_MODES
    assert "safe-residual" in ON_POLICY_TARGET_MODES
    assert "composite" in SAMPLE_SELECTION_METRICS
    assert "composite-local" in SAMPLE_SELECTION_METRICS
    assert "composite-gap" in SAMPLE_SELECTION_METRICS
    assert "low-confidence-terminal" in CLASSIFIER_LOSS_MODES
    images, labels = _toy_digit_measures(grid_size=config.grid_size)
    batch = sample_flux_training_batch(images, labels, config, batch_size=4, device="cpu", rng=np.random.default_rng(5), step_index=2000)
    model = DirectFluxUNet(config, base_channels=4, num_classes=10)
    raw = model(batch.tau, batch.states, batch.labels, batch.sources)
    projected = apply_flux_parameterization_torch(raw, batch.states, config)
    assert torch.allclose(flux_divergence_torch(projected), flux_divergence_torch(raw), atol=2e-5, rtol=2e-5)
    assert flux_curl_torch(projected).square().mean() <= flux_curl_torch(raw).square().mean() + 1e-7
    (
        rollout_loss,
        endpoint_l2,
        rollout_img,
        tv_loss,
        endpoint_bce,
        endpoint_tv,
        classifier_loss,
        classifier_conf_loss,
        terminal_active,
        terminal_tau_mean,
        terminal_scale,
        shape_loss,
        shape_entropy_loss,
        shape_tv_loss,
        shape_maxmass_loss,
        local_shape_loss,
        local_support_loss,
        local_edge_loss,
        negative_space_loss,
        gap_shape_loss,
        missing_support_loss,
        extra_support_loss,
        gap_loss,
        strict_negative_space_loss,
        foreground_recall_loss,
    ) = direct_flux_rollout_consistency_loss(model, batch, max_items=2, steps=1, return_extra=True)
    assert torch.isfinite(rollout_loss)
    assert torch.isfinite(endpoint_l2)
    assert torch.isfinite(rollout_img)
    assert torch.isfinite(tv_loss)
    assert torch.isfinite(endpoint_bce)
    assert torch.isfinite(endpoint_tv)
    assert torch.isfinite(classifier_loss)
    assert torch.isfinite(classifier_conf_loss)
    assert torch.isfinite(terminal_active)
    assert torch.isfinite(terminal_scale)
    assert torch.isfinite(shape_loss)
    assert torch.isfinite(shape_entropy_loss)
    assert torch.isfinite(shape_tv_loss)
    assert torch.isfinite(shape_maxmass_loss)
    assert torch.isfinite(local_shape_loss)
    assert torch.isfinite(local_support_loss)
    assert torch.isfinite(local_edge_loss)
    assert torch.isfinite(negative_space_loss)
    assert torch.isfinite(gap_shape_loss)
    assert torch.isfinite(missing_support_loss)
    assert torch.isfinite(extra_support_loss)
    assert torch.isfinite(gap_loss)
    assert torch.isfinite(strict_negative_space_loss)
    assert torch.isfinite(foreground_recall_loss)
    loss, metrics = direct_flux_matching_loss(model, batch)
    assert torch.isfinite(loss)
    for key in ["rollout_loss", "rollout_image_grad_loss", "target_tv_loss", "terminal_shape_loss", "terminal_gap_shape_loss", "terminal_gap_loss", "terminal_strict_negative_space_loss", "image_grad_loss", "curl_loss", "checkerboard_loss"]:
        assert key in metrics
    assert image_total_variation(batch.states, grid_size=config.grid_size) >= 0
    assert checkerboard_energy_torch(batch.states, grid_size=config.grid_size) >= 0


def test_replay_cache_and_process_figure_smoke(tmp_path) -> None:
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=4,
        target_mode="poisson-ot-flow",
        source_mode="lowfreq",
        source_lowfreq_size=4,
        ot_match_mode="nearest",
        on_policy_mode="replay",
        on_policy_cache_size=4,
        on_policy_cache_rollout_batch_size=2,
        on_policy_cache_mode="trajectory",
        on_policy_cache_snapshots_per_traj=2,
        on_policy_target_mode="safe-residual",
        on_policy_prefix_mode="short",
        on_policy_prefix_steps=2,
        rollout_loss_steps=1,
        rollout_loss_batch_size=2,
        rollout_loss_every=2,
        rollout_image_grad_loss_weight=0.01,
        project_main_loss=False,
        flux_scale=10.0,
    )
    images, labels = _toy_digit_measures(num_samples=20, grid_size=config.grid_size)
    cache = build_classwise_ot_cache(images, labels, config)
    model = DirectFluxUNet(config, base_channels=4, num_classes=10)
    replay = build_on_policy_replay_cache(
        model,
        images,
        labels,
        config,
        cache_size=4,
        rollout_batch_size=2,
        device=torch.device("cpu"),
        rng=rng,
        dtype=torch.float32,
        class_means=cache.class_means,
        ot_cache=cache,
        step_index=10,
    )
    assert replay.size == 4
    assert replay.mode == "trajectory"
    # In very short smoke-test horizons, the configured terminal tau window can
    # be unreachable after de-duplicating integer prefix steps.  The stable cache
    # builder should report that honestly rather than forcing duplicate terminal
    # snapshots.
    assert replay.terminal_snapshot_count == 0
    assert replay.regular_snapshot_count == 2
    assert abs(replay.terminal_requested_fraction - 0.0) < 1e-6
    assert abs(replay.terminal_actual_fraction - replay.terminal_fraction) < 1e-6
    assert 0.0 <= replay.tau_min <= replay.tau_mean <= replay.tau_max <= 1.0
    assert ON_POLICY_CACHE_MODES == ("independent", "trajectory")
    assert "safe-residual" in ON_POLICY_TARGET_MODES
    assert "composite" in SAMPLE_SELECTION_METRICS
    assert "low-confidence-terminal" in CLASSIFIER_LOSS_MODES
    batch = sample_on_policy_replay_batch(replay, batch_size=3, device=torch.device("cpu"), rng=rng, step_index=11)
    assert batch.states.shape[0] == 3
    assert batch.target_velocity_mode == "safe-residual"
    traj = np.stack([batch.sources.detach().numpy(), batch.states.detach().numpy(), batch.targets.detach().numpy()], axis=0)
    out = tmp_path / "process.png"
    save_diffusion_process_figure(traj, batch.labels.detach().numpy(), out, grid_size=config.grid_size, num_frames=3, max_samples=2)
    assert out.exists()



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
    assert on_policy_batch.target_velocity_mode == "safe-residual"

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


def test_terminal_classifier_metrics_and_goodbad_analysis(tmp_path) -> None:
    images, labels = _toy_digit_measures(num_samples=8, grid_size=8)
    clf = TinyMNISTClassifier(grid_size=8)
    metrics = classifier_generation_metrics(images.reshape(8, -1), labels, clf, grid_size=8, device="cpu")
    assert "classifier_acc" in metrics
    tmp = tmp_path / "samples_goodbad_smoke.txt"
    tmp.write_text("good bad good bad good bad good bad")
    analysis = analyze_goodbad_annotations(tmp, images.reshape(8, -1), labels, classifier_metrics=metrics)
    assert analysis["human_good_rate"] == 0.5
    assert analysis["human_bad_count_by_label"].shape == (10,)


def test_composite_selection_and_shape_statistics(tmp_path) -> None:
    images, labels = _toy_digit_measures(num_samples=8, grid_size=8)
    shape = compute_class_shape_statistics(images, labels, grid_size=8)
    assert "entropy_q75" in shape
    assert "local_support_mean" in shape
    local_metrics = local_shape_metrics_np(images.reshape(8, -1), labels, shape, grid_size=8)
    assert local_metrics["negative_space_mass"].shape == (8,)
    stats = compute_shape_statistics_np(images.reshape(8, -1), grid_size=8)
    assert stats["tv"].shape == (8,)
    clf = TinyMNISTClassifier(grid_size=8)
    config = DirectFluxMNISTConfig(grid_size=8, horizon_scale=0.2, num_steps=2)
    model = DirectFluxUNet(config, base_channels=4)
    result = simulate_direct_flux_generation(
        model,
        labels=[0, 0, 1, 1],
        num_steps=1,
        deterministic=True,
        device="cpu",
        seed=9,
        use_amp=False,
        show_progress=False,
    )
    selected = select_generation_result_by_classifier(
        result,
        np.asarray([0, 1], dtype=np.int64),
        factor=2,
        classifier=clf,
        grid_size=8,
        device="cpu",
        selection_metric="composite-local",
        shape_stats=shape,
        config=config,
        report_path=tmp_path / "selection.csv",
    )
    assert selected.samples.shape[0] == 2
    assert (tmp_path / "selection.csv").exists()
    goodbad = tmp_path / "samples_goodbad.txt"
    goodbad.write_text("goood bad")
    write_goodbad_sample_report(tmp_path / "goodbad.csv", goodbad, selected.samples, selected.labels, grid_size=8)
    assert (tmp_path / "goodbad.csv").exists()


def test_terminal_local_shape_loss_smoke() -> None:
    images, labels = _toy_digit_measures(num_samples=8, grid_size=8)
    shape = compute_class_shape_statistics(images, labels, grid_size=8, local_shape_size=4)
    shape_torch = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in shape.items()}
    config = DirectFluxMNISTConfig(
        grid_size=8,
        horizon_scale=0.2,
        num_steps=4,
        terminal_local_shape_size=4,
        terminal_local_shape_loss_weight=0.1,
    )
    states = torch.as_tensor(images.reshape(8, -1), dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    weights = torch.ones(8)
    total, support, edge, negative = terminal_local_shape_loss_torch(states, states, labels_t, shape_torch, config, weights=weights)
    assert torch.isfinite(total)
    assert support.item() >= 0.0
    assert edge.item() >= 0.0
    assert negative.item() >= 0.0


def test_terminal_snapshot_steps_include_late_tau() -> None:
    config = DirectFluxMNISTConfig(
        grid_size=8,
        num_steps=100,
        on_policy_prefix_mode="uniform",
        on_policy_min_prefix_fraction=0.05,
        on_policy_max_prefix_fraction=0.85,
        on_policy_cache_snapshots_per_traj=10,
        on_policy_cache_terminal_fraction=0.4,
        on_policy_cache_terminal_min_tau=0.02,
        on_policy_cache_terminal_max_tau=0.18,
    )
    from mnist.eulerian_flux_mnist import _trajectory_snapshot_steps
    steps = _trajectory_snapshot_steps(config)
    assert steps.max() >= 90
    assert steps.min() <= 10


def test_experiment10_run_dir_is_timestamped_unique_and_named(tmp_path) -> None:
    from datetime import datetime

    fixed = datetime(2026, 5, 28, 22, 39, 1)
    run_dir, meta = make_experiment10_run_dir(tmp_path / "runs" / "experiment10", "my first run!", now=fixed)
    assert run_dir.exists()
    assert run_dir.parent == tmp_path / "runs" / "experiment10"
    assert run_dir.name == "20260528-223901_my-first-run"
    assert meta["run_id"] == run_dir.name
    assert meta["run_name"] == "my-first-run"

    second_dir, second_meta = make_experiment10_run_dir(tmp_path / "runs" / "experiment10", "my first run!", now=fixed)
    assert second_dir.exists()
    assert second_dir.name == "20260528-223901_my-first-run_02"
    assert second_meta["run_id"] == second_dir.name

    timestamp_only, timestamp_meta = make_experiment10_run_dir(tmp_path / "runs" / "experiment10", "", now=fixed)
    assert timestamp_only.name == "20260528-223901"
    assert timestamp_meta["run_name"] == ""
