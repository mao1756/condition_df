"""Smoke checks for Experiment 12/D0 Phase 0 forward noising diagnostics."""

from __future__ import annotations

import json

import numpy as np
import torch

from mnist.diag_forward_noising import (
    _gate_summary,
    _reference_schedules,
    _synthetic_digit_measures,
    effective_time_integral,
    expected_symmetric_dirichlet_entropy,
    make_rate_schedule,
    parse_args,
    run_forward_noising_single,
    save_phase0_result,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def test_d0_phase0_forward_noising_smoke(tmp_path) -> None:
    images, labels = _synthetic_digit_measures(examples_per_class=2, grid_size=8, seed=123)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        num_steps=4,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        ot_com_weight=0.10,
        limiter_fraction=0.25,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-7,
    )
    result = run_forward_noising_single(
        images=images,
        labels=labels,
        config=config,
        lambda_mix=0.05,
        free_weight=0.0,
        noise_weight=0.002,
        sample_steps=4,
        num_paths=6,
        batch_size=3,
        theta_mask_min=1e-12,
        preview_fractions=[0.0, 0.5, 1.0],
        metric_bins=2,
        device=torch.device("cpu"),
        seed=11,
        show_progress=False,
    )
    assert result.initial_states.shape == (6, 64)
    assert result.terminal_states.shape == (6, 64)
    assert np.allclose(result.initial_states.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(result.terminal_states.sum(axis=1), 1.0, atol=1e-5)
    assert {0, 2, 4}.issubset(set(result.checkpoint_states))
    assert len(result.metrics) >= 3
    assert result.summary["cumulative_clip_fraction"] >= 0.0
    assert "mean_substeps" in result.summary
    assert "fraction_steps_at_max_substeps" in result.summary
    assert "cumulative_masked_edge_fraction" in result.summary
    assert "stationarity_pass" in result.summary
    assert "final_entropy_fraction_of_stationary" in result.summary

    gated = _gate_summary(
        result,
        max_final_corr=1.1,
        max_masked_edge_fraction=1.0,
        max_time_bin_masked_edge_fraction=1.0,
        max_frozen_edge_fraction=1.0,
        require_stationarity=False,
    )
    assert gated["gate_pass"] == 1

    result.summary = gated
    paths = save_phase0_result(result, tmp_path, preview_images=4, save_previews=True)
    assert (tmp_path / result.run_id / "metrics.csv").exists()
    assert (tmp_path / result.run_id / "prior_bank.npz").exists()
    assert (tmp_path / result.run_id / "forward_noising_panel.png").exists()
    assert "prior_bank_path" in paths
    summary = json.loads((tmp_path / result.run_id / "summary.json").read_text())
    assert summary["gate_pass"] == 1


def test_d0_phase0_faithful_reference_defaults() -> None:
    args = parse_args(["--sweep-tau-eff", "1e-6,4e-6"])
    assert args.reference_scale_mode == "faithful"
    assert args.edge_alpha_mode == "alpha_eff"
    assert args.time_change_mode == "integral"
    schedules = _reference_schedules(args)
    assert len(schedules) == 2

    cfg = DirectFluxMNISTConfig(grid_size=8, edge_alpha_mode="alpha_eff", alpha_eff=1.0)
    horizon = natural_horizon(cfg)
    dt = horizon / 8.0
    for sched in schedules:
        assert sched["reference_rate"] is None
        tau_eff = float(sched["tau_eff"])
        rate_schedule = make_rate_schedule(
            8,
            mode="faithful",
            tau_eff=tau_eff,
            constant_rate=None,
            ramp="none",
            ramp_ratio=1.0,
            rate_min=None,
            rate_max=None,
            horizon=horizon,
            time_change_mode=args.time_change_mode,
        )
        assert np.isclose(effective_time_integral(rate_schedule, dt=dt), tau_eff)
        assert np.isclose(float(rate_schedule.mean()), tau_eff / horizon)

    legacy_args = parse_args(["--sweep-tau-eff", "1e-6", "--time-change-mode", "rate"])
    legacy_schedule = make_rate_schedule(
        8,
        mode="faithful",
        tau_eff=1e-6,
        constant_rate=None,
        ramp="none",
        ramp_ratio=1.0,
        rate_min=None,
        rate_max=None,
        horizon=horizon,
        time_change_mode=legacy_args.time_change_mode,
    )
    assert np.isclose(float(legacy_schedule.mean()), 1e-6)

    expected_entropy = expected_symmetric_dirichlet_entropy(cfg)
    assert np.isfinite(expected_entropy)
    assert expected_entropy > 0.0


def test_d0_phase0_gate_modes_split_stationarity_from_data_gate() -> None:
    images, labels = _synthetic_digit_measures(examples_per_class=1, grid_size=8, seed=321)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        num_steps=2,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-7,
    )
    result = run_forward_noising_single(
        images=images,
        labels=labels,
        config=config,
        lambda_mix=0.10,
        free_weight=0.0,
        noise_weight=0.0,
        sample_steps=2,
        num_paths=4,
        batch_size=2,
        theta_mask_min=1e-12,
        preview_fractions=[0.0, 1.0],
        metric_bins=1,
        device=torch.device("cpu"),
        seed=12,
        show_progress=False,
    )
    # Force a clean D0-practical destruction/health record with stationarity
    # deliberately failing.  The data-start D0 gate must not require Dirichlet
    # terminal stationarity.
    result.summary.update(
        {
            "final_pixel_corr_mean": 0.0,
            "final_background_l1_mean": 1.0,
            "final_fraction_pixels_changed_above_floor": 1.0,
            "stationarity_pass": 0,
            "cumulative_mobility_weighted_masked_edge_fraction": 0.0,
            "max_time_bin_mobility_weighted_masked_edge_fraction": 0.0,
        }
    )
    gated = _gate_summary(
        result,
        max_final_corr=0.1,
        max_masked_edge_fraction=0.05,
        max_time_bin_masked_edge_fraction=0.10,
        max_frozen_edge_fraction=1.0,
        min_background_l1=1e-3,
        min_fraction_pixels_changed=0.01,
        phase0_gate_mode="d0-practical",
        limiter_health_metric="mobility_weighted",
        max_weighted_masked_edge_fraction=0.10,
        max_weighted_time_bin_masked_edge_fraction=0.20,
    )
    assert gated["gate_pass"] == 1
    assert gated["gate_require_stationarity"] == 0
    assert gated["stationarity_gate_pass"] == 0

    # Conversely, exact-stationary validation does not require digit destruction,
    # but it must be run from a Dirichlet initial law.
    result.summary.update({"init_law": "dirichlet", "stationarity_pass": 1, "final_pixel_corr_mean": 1.0})
    exact = _gate_summary(
        result,
        max_final_corr=0.1,
        max_masked_edge_fraction=0.05,
        max_time_bin_masked_edge_fraction=0.10,
        max_frozen_edge_fraction=1.0,
        phase0_gate_mode="exact-stationary",
        limiter_health_metric="mobility_weighted",
        max_weighted_masked_edge_fraction=0.10,
        max_weighted_time_bin_masked_edge_fraction=0.20,
    )
    assert exact["gate_pass"] == 1
    assert exact["destruction_gate_pass"] == 0
    assert exact["stationarity_gate_pass"] == 1


def test_d0_phase0_dirichlet_init_smoke() -> None:
    images = np.full((1, 8, 8), 1.0 / 64.0, dtype=np.float64)
    labels = np.zeros((1,), dtype=np.int64)
    config = DirectFluxMNISTConfig(
        grid_size=8,
        num_steps=2,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-7,
    )
    result = run_forward_noising_single(
        images=images,
        labels=labels,
        config=config,
        init_law="dirichlet",
        lambda_mix=0.0,
        free_weight=0.0,
        noise_weight=0.0,
        sample_steps=2,
        num_paths=5,
        batch_size=5,
        theta_mask_min=1e-12,
        preview_fractions=[0.0, 1.0],
        metric_bins=1,
        device=torch.device("cpu"),
        seed=13,
        show_progress=False,
    )
    assert result.init_law == "dirichlet"
    assert result.summary["init_law"] == "dirichlet"
    assert np.all(result.source_indices == -1)
    assert "initial_terminal_quantile_distance" in result.summary
