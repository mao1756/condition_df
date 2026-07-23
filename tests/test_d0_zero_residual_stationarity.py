"""Contracts for D0 zero-residual production-kernel validation."""

from __future__ import annotations

import json
import math
from types import ModuleType

import numpy as np
import pytest
import torch

import mnist.experiment12_d0 as d0_module
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, edge_alpha_value


def _zero_diag_module() -> ModuleType:
    import mnist.diag_d0_zero_residual as module

    return module


def _toy_dynamics(
    *,
    grid_size: int = 4,
    limiter_fraction: float = 1.0,
    mass_floor: float = 1e-10,
) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
        num_steps=1,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        source_uniform_mix=0.10,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        ot_com_weight=0.10,
        condition_on_source=False,
        flux_parameterization="edge",
        limiter_fraction=limiter_fraction,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=mass_floor,
    )


def _nonuniform_states(batch_size: int = 2) -> torch.Tensor:
    rows = torch.tensor(
        [
            [
                0.18,
                0.04,
                0.08,
                0.02,
                0.03,
                0.12,
                0.02,
                0.07,
                0.05,
                0.03,
                0.09,
                0.04,
                0.02,
                0.06,
                0.08,
                0.07,
            ],
            [
                0.03,
                0.11,
                0.02,
                0.05,
                0.14,
                0.03,
                0.04,
                0.06,
                0.02,
                0.08,
                0.12,
                0.04,
                0.07,
                0.03,
                0.09,
                0.07,
            ],
        ],
        dtype=torch.float64,
    )
    rows = rows / rows.sum(dim=1, keepdim=True)
    if batch_size <= rows.shape[0]:
        return rows[:batch_size].clone()
    return rows.repeat(math.ceil(batch_size / rows.shape[0]), 1)[:batch_size].clone()


def test_direct_substep_adds_positive_reference_drift_and_injected_noise_before_actual_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamics = _toy_dynamics()
    states = torch.full((1, 16), 1.0 / 16.0, dtype=torch.float64)
    learned = torch.zeros((1, 2, 4, 4), dtype=states.dtype)
    drift = torch.zeros_like(learned)
    noise_std = torch.zeros_like(learned)
    normal = torch.zeros_like(learned)
    learned[0, 0, 0, 0] = 0.005
    drift[0, 0, 0, 0] = 0.010
    noise_std[0, 0, 0, 0] = 0.005
    normal[0, 0, 0, 0] = 2.0

    monkeypatch.setattr(d0_module, "free_drift_flux_torch", lambda *_args, **_kwargs: drift)
    monkeypatch.setattr(d0_module, "edge_noise_std_channels", lambda *_args, **_kwargs: noise_std)

    rate = 4.0
    dt = 0.25
    expected_transfer = rate * drift * dt + learned + math.sqrt(rate) * noise_std * normal
    expected_states, _ = d0_module._apply_oriented_edge_transfer(
        states,
        expected_transfer,
        dynamics,
    )
    result = d0_module._direct_doob_reverse_substep(
        states,
        learned,
        rate=rate,
        dt=dt,
        dynamics_config=dynamics,
        standard_normal=normal,
        deterministic=False,
    )

    assert expected_transfer[0, 0, 0, 0].item() == pytest.approx(0.035)
    assert torch.equal(result.states, expected_states)
    assert torch.equal(result.free_delta + result.learned_delta + result.noise_delta, expected_transfer)
    assert result.diagnostics["proposed_edges"] == expected_transfer.numel()


def test_direct_substep_injected_noise_is_reproducible_and_bypasses_global_rng() -> None:
    dynamics = _toy_dynamics()
    states = _nonuniform_states()
    learned = torch.zeros((2, 2, 4, 4), dtype=states.dtype)
    generator = torch.Generator().manual_seed(731)
    normal = torch.randn(learned.shape, dtype=learned.dtype, generator=generator)

    torch.manual_seed(1)
    first = d0_module._direct_doob_reverse_substep(
        states,
        learned,
        rate=0.4,
        dt=2e-4,
        dynamics_config=dynamics,
        standard_normal=normal,
    )
    torch.manual_seed(999)
    second = d0_module._direct_doob_reverse_substep(
        states,
        learned,
        rate=0.4,
        dt=2e-4,
        dynamics_config=dynamics,
        standard_normal=normal.clone(),
    )
    reflected = d0_module._direct_doob_reverse_substep(
        states,
        learned,
        rate=0.4,
        dt=2e-4,
        dynamics_config=dynamics,
        standard_normal=-normal,
    )

    assert torch.equal(first.states, second.states)
    assert torch.equal(first.noise_delta, second.noise_delta)
    assert first.diagnostics == second.diagnostics
    assert not torch.equal(first.states, reflected.states)


def test_direct_substep_preserves_simplex_and_reports_complete_production_diagnostics() -> None:
    dynamics = _toy_dynamics(limiter_fraction=0.25, mass_floor=1e-7)
    states = _nonuniform_states()
    learned = torch.full((2, 2, 4, 4), 0.5, dtype=states.dtype)
    normal = torch.zeros_like(learned)
    normal[0, 1, 2, 3] = float("nan")

    result = d0_module._direct_doob_reverse_substep(
        states,
        learned,
        rate=0.3,
        dt=1e-3,
        dynamics_config=dynamics,
        standard_normal=normal,
    )

    expected_fields = {
        "limited_edges",
        "proposed_edges",
        "limiter_fraction",
        "mobility_weight_sum",
        "limited_mobility_weight_sum",
        "mobility_weighted_limiter_fraction",
        "noise_energy_sum",
        "limited_noise_energy_sum",
        "noise_energy_weighted_limiter_fraction",
        "nonfinite_edges",
        "floor_touched_pixels",
        "floor_proposed_pixels",
        "floor_correction_l1",
        "renorm_correction_l1",
        "max_simplex_mass_error",
    }
    diagnostics = result.diagnostics
    assert expected_fields <= diagnostics.keys()
    assert diagnostics["limited_edges"] > 0
    assert diagnostics["nonfinite_edges"] >= 1
    assert diagnostics["floor_proposed_pixels"] == states.numel()
    assert all(np.isfinite(float(diagnostics[key])) for key in expected_fields)
    assert torch.isfinite(result.states).all()
    assert (result.states >= 0.0).all()
    assert torch.allclose(
        result.states.sum(dim=1),
        torch.ones(result.states.shape[0], dtype=result.states.dtype),
    )
    assert diagnostics["max_simplex_mass_error"] <= 5e-15


def _passing_refinement_levels() -> list[dict[str, float | int]]:
    return [
        {
            "substeps": 2,
            "dt": 0.05,
            "stationarity_quantile_distance": 0.018,
            "stationarity_quantile_threshold": 0.02,
            "stationarity_quantile_ratio": 0.9,
            "stationarity_feature_mmd": 8e-4,
            "stationarity_feature_mmd_threshold": 1e-3,
            "stationarity_feature_mmd_ratio": 0.8,
            "entropy_standard_error_units": 1.1,
            "entropy_analytic_standard_error_units": 1.0,
            "entropy_paired_drift_standard_error_units": 0.9,
            "second_moment_standard_error_units": 0.9,
            "second_moment_analytic_standard_error_units": 0.8,
            "second_moment_paired_drift_standard_error_units": 0.7,
            "coupled_refinement_rms": float("nan"),
            "limiter_fraction": 0.02,
            "mobility_weighted_limiter_fraction": 0.012,
            "noise_energy_weighted_limiter_fraction": 0.010,
            "floor_correction_l1_per_path_substep": 8e-10,
            "renorm_correction_l1_per_path_substep": 8e-8,
            "max_simplex_mass_error": 2e-8,
            "nonfinite_edges": 0,
        },
        {
            "substeps": 4,
            "dt": 0.025,
            "stationarity_quantile_distance": 0.015,
            "stationarity_quantile_threshold": 0.02,
            "stationarity_quantile_ratio": 0.75,
            "stationarity_feature_mmd": 7e-4,
            "stationarity_feature_mmd_threshold": 1e-3,
            "stationarity_feature_mmd_ratio": 0.7,
            "entropy_standard_error_units": 0.8,
            "entropy_analytic_standard_error_units": 0.7,
            "entropy_paired_drift_standard_error_units": 0.6,
            "second_moment_standard_error_units": 0.7,
            "second_moment_analytic_standard_error_units": 0.6,
            "second_moment_paired_drift_standard_error_units": 0.5,
            "coupled_refinement_rms": 0.10,
            "limiter_fraction": 0.02,
            "mobility_weighted_limiter_fraction": 0.010,
            "noise_energy_weighted_limiter_fraction": 0.008,
            "floor_correction_l1_per_path_substep": 6e-10,
            "renorm_correction_l1_per_path_substep": 6e-8,
            "max_simplex_mass_error": 2e-8,
            "nonfinite_edges": 0,
        },
        {
            "substeps": 8,
            "dt": 0.0125,
            "stationarity_quantile_distance": 0.012,
            "stationarity_quantile_threshold": 0.02,
            "stationarity_quantile_ratio": 0.6,
            "stationarity_feature_mmd": 6e-4,
            "stationarity_feature_mmd_threshold": 1e-3,
            "stationarity_feature_mmd_ratio": 0.6,
            "entropy_standard_error_units": 0.6,
            "entropy_analytic_standard_error_units": 0.5,
            "entropy_paired_drift_standard_error_units": 0.4,
            "second_moment_standard_error_units": 0.5,
            "second_moment_analytic_standard_error_units": 0.4,
            "second_moment_paired_drift_standard_error_units": 0.3,
            "coupled_refinement_rms": 0.08,
            # The practical fixed-grid gate can pass even though the raw
            # intervention fraction has not converged toward zero.
            "limiter_fraction": 0.02,
            "mobility_weighted_limiter_fraction": 0.008,
            "noise_energy_weighted_limiter_fraction": 0.006,
            "floor_correction_l1_per_path_substep": 4e-10,
            "renorm_correction_l1_per_path_substep": 4e-8,
            "max_simplex_mass_error": 2e-8,
            "nonfinite_edges": 0,
        },
    ]


def test_refinement_gate_separates_fixed_grid_stationarity_from_strict_h_transform_limit() -> None:
    zero_diag = _zero_diag_module()
    levels = _passing_refinement_levels()
    gate = zero_diag.evaluate_refinement_gate(
        levels,
        max_simplex_mass_error=2e-6,
        max_floor_correction_l1=1e-8,
        max_renorm_correction_l1=1e-6,
        max_standardized_moment_error=3.0,
        refinement_contraction=0.9,
    )

    assert gate["gate_pass_stationarity"] == 1
    assert gate["gate_pass_coupled_refinement"] == 1
    assert gate["gate_pass_nonincreasing_interventions"] == 1
    assert gate["fixed_grid_stationarity_pass"] == 1
    assert gate["strict_h_transform_limit_supported"] == 0

    failing_levels = [dict(level) for level in levels]
    failing_levels[-1]["coupled_refinement_rms"] = 0.095
    failed = zero_diag.evaluate_refinement_gate(
        failing_levels,
        max_simplex_mass_error=2e-6,
        max_floor_correction_l1=1e-8,
        max_renorm_correction_l1=1e-6,
        max_standardized_moment_error=3.0,
        refinement_contraction=0.9,
    )
    assert failed["gate_pass_coupled_refinement"] == 0
    assert failed["fixed_grid_stationarity_pass"] == 0

    insufficient = zero_diag.evaluate_refinement_gate(levels[-2:])
    assert insufficient["gate_pass_coupled_refinement"] == 0
    assert insufficient["minimum_refinement_levels"] == 3
    assert insufficient["fixed_grid_stationarity_pass"] == 0

    nonfinite_levels = [dict(level) for level in levels]
    nonfinite_levels[0]["limiter_fraction"] = float("nan")
    nonfinite = zero_diag.evaluate_refinement_gate(nonfinite_levels)
    assert nonfinite["gate_pass_nonincreasing_interventions"] == 0
    assert nonfinite["fixed_grid_stationarity_pass"] == 0

    strict_levels = [dict(level) for level in levels]
    for row, raw, weighted in zip(
        strict_levels,
        (4e-3, 3e-3, 2e-3),
        (4e-4, 3e-4, 2e-4),
    ):
        row["limiter_fraction"] = raw
        row["mobility_weighted_limiter_fraction"] = weighted
        row["noise_energy_weighted_limiter_fraction"] = weighted
    strict = zero_diag.evaluate_refinement_gate(strict_levels)
    assert strict["gate_pass_strict_intervention_decay"] == 1
    assert strict["strict_h_transform_limit_supported"] == 1

    constant_levels = [dict(level) for level in strict_levels]
    for row in constant_levels:
        row["limiter_fraction"] = 2e-3
        row["mobility_weighted_limiter_fraction"] = 2e-4
        row["noise_energy_weighted_limiter_fraction"] = 2e-4
    constant = zero_diag.evaluate_refinement_gate(constant_levels)
    assert constant["gate_pass_nonincreasing_interventions"] == 1
    assert constant["gate_pass_strict_intervention_decay"] == 0
    assert constant["strict_h_transform_limit_supported"] == 0


def test_small_nonzero_reference_dynamics_dirichlet_stationarity_smoke() -> None:
    zero_diag = _zero_diag_module()
    dynamics = _toy_dynamics()
    result = zero_diag.run_zero_residual_refinement(
        dynamics_config=dynamics,
        num_paths=128,
        sample_steps=2,
        substeps=(1, 2, 4),
        horizon=2e-4,
        rate_schedule=np.asarray([0.5, 0.5], dtype=np.float64),
        seed=24680,
        device=torch.device("cpu"),
        calibration_reps=2,
    )

    assert result["reference_rate_integral"] == pytest.approx(1e-4)
    assert result["dirichlet_alpha"] == pytest.approx(edge_alpha_value(dynamics))
    levels = result["levels"]
    assert [int(level["substeps"]) for level in levels] == [1, 2, 4]
    assert np.isfinite([float(level["stationarity_quantile_distance"]) for level in levels]).all()
    assert np.isfinite([float(level["stationarity_feature_mmd"]) for level in levels]).all()
    assert len({float(level["stationarity_quantile_threshold"]) for level in levels}) == 1
    assert len({float(level["stationarity_feature_mmd_threshold"]) for level in levels}) == 1
    assert all(float(level["free_step_rms"]) > 0.0 for level in levels)
    assert all(float(level["noise_step_rms"]) > 0.0 for level in levels)
    assert all(int(level["nonfinite_edges"]) == 0 for level in levels)
    assert all(float(level["max_simplex_mass_error"]) <= 2e-6 for level in levels)
    exact_second = (edge_alpha_value(dynamics) + 1.0) / (
        16.0 * edge_alpha_value(dynamics) + 1.0
    )
    assert all(
        float(level["second_moment_analytic_expected"]) == pytest.approx(exact_second)
        for level in levels
    )
    assert all(
        np.isfinite(float(level[key]))
        for level in levels
        for key in (
            "entropy_analytic_standard_error_units",
            "entropy_paired_drift_standard_error_units",
            "second_moment_analytic_standard_error_units",
            "second_moment_paired_drift_standard_error_units",
        )
    )

    finest = np.asarray(result["states_finest"], dtype=np.float64)
    assert finest.shape == (128, 16)
    assert np.isfinite(finest).all()
    assert (finest >= 0.0).all()
    assert np.allclose(finest.sum(axis=1), 1.0, atol=2e-6, rtol=0.0)
    # Coupled refinements share the same initial bank and Brownian path.  Both
    # adjacent-level discrepancies must therefore be reported, independently
    # of whether this tiny smoke run is powerful enough to pass the science gate.
    assert np.isfinite(float(levels[1]["coupled_refinement_rms"]))
    assert np.isfinite(float(levels[2]["coupled_refinement_rms"]))
    assert {"fixed_grid_stationarity_pass", "strict_h_transform_limit_supported"} <= result[
        "gate"
    ].keys()


def test_refinement_parser_and_integrated_rate_schedule_contract() -> None:
    zero_diag = _zero_diag_module()

    assert zero_diag.parse_substep_levels("4,1,2,2") == (1, 2, 4)
    with pytest.raises(ValueError, match="divide the finest"):
        zero_diag.parse_substep_levels((3, 4))

    schedule = zero_diag.make_coupled_rate_schedule(
        4,
        horizon=0.2,
        tau_eff=0.01,
    )
    assert schedule.shape == (4,)
    assert np.isfinite(schedule).all()
    assert float(schedule.sum()) * 0.2 / 4.0 == pytest.approx(0.01)
    with pytest.raises(ValueError, match="tau_eff must be finite and positive"):
        zero_diag.ZeroResidualDiagnosticConfig(tau_eff=0.0)
    with pytest.raises(ValueError, match="tau_eff must be finite and positive"):
        zero_diag.make_coupled_rate_schedule(4, horizon=0.2, tau_eff=0.0)
    config = zero_diag.ZeroResidualDiagnosticConfig(
        num_paths=8,
        sample_steps=2,
        substep_levels=(1, 2, 4),
        horizon=0.2,
        tau_eff=0.01,
        calibration_reps=1,
    )
    with pytest.raises(ValueError, match="positive reference-rate integral"):
        zero_diag.run_zero_residual_diagnostic(
            dynamics_config=_toy_dynamics(),
            diagnostic_config=config,
            device=torch.device("cpu"),
            rate_schedule=np.zeros(2, dtype=np.float64),
        )


def test_saved_diagnostic_artifacts_include_gate_states_and_plot(tmp_path) -> None:
    zero_diag = _zero_diag_module()

    dynamics = _toy_dynamics()
    config = zero_diag.ZeroResidualDiagnosticConfig(
        num_paths=32,
        sample_steps=1,
        substep_levels=(1, 2),
        horizon=1e-4,
        tau_eff=1e-5,
        seed=97531,
        calibration_reps=1,
    )
    result = zero_diag.run_zero_residual_diagnostic(
        dynamics_config=dynamics,
        diagnostic_config=config,
        device=torch.device("cpu"),
    )
    paths = zero_diag.save_zero_residual_diagnostic(result, tmp_path)

    assert set(paths) == {"summary_path", "metrics_path", "states_path", "plot_path"}
    assert all(tmp_path.joinpath(name).exists() for name in (
        "summary.json",
        "refinement_metrics.csv",
        "states_finest.npz",
        "stationarity_refinement.png",
    ))
    summary = json.loads(tmp_path.joinpath("summary.json").read_text(encoding="utf-8"))
    assert summary["gate"]["claim_scope"] == "fixed-grid temporal refinement only"
    with np.load(tmp_path.joinpath("states_finest.npz")) as saved:
        assert saved["initial_states"].shape == (32, 16)
        assert saved["terminal_states"].shape == (32, 16)


def test_multi_seed_cli_saves_self_describing_aggregate(tmp_path) -> None:
    zero_diag = _zero_diag_module()
    zero_diag.main(
        [
            "--runs-root",
            str(tmp_path),
            "--run-name",
            "aggregate-smoke",
            "--device",
            "cpu",
            "--seeds",
            "41,42",
            "--num-paths",
            "8",
            "--grid-size",
            "4",
            "--sample-steps",
            "1",
            "--substeps",
            "1,2,4",
            "--tau-eff",
            "1e-5",
            "--calibration-reps",
            "1",
        ]
    )

    run_dirs = list(tmp_path.glob("*_aggregate-smoke"))
    assert len(run_dirs) == 1
    aggregate_path = run_dirs[0] / "aggregate_summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["seeds"] == [41, 42]
    assert aggregate["aggregate_summary_path"] == str(aggregate_path)
    assert {path.name for path in run_dirs[0].glob("seed-*")} == {"seed-41", "seed-42"}
