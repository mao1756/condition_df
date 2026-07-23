from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_denoising import (
    Alpha1SpectralConfig,
    SpectralConvergenceError,
    SpectralInverseCDFConfig,
    apply_matching_head_fractions,
    build_four_color_matchings,
    denoising_mean_to_mass_flux,
    evaluate_alpha1_spectral,
    evaluate_alpha1_spectral_torch_fixed_modes,
    jacobi_component_relative_score,
    jacobi_latent_label,
    jacobi_phase_exposure,
    linear_teacher_arrival_score,
    linear_teacher_denoising_mean,
    linear_teacher_relative_density,
    palindromic_strang_plan,
    sample_alpha1_spectral_inverse_cdf,
    validate_four_color_matchings,
)


def test_four_matchings_partition_even_periodic_grid_and_strang_is_palindromic() -> None:
    grid_size = 6
    matchings = build_four_color_matchings(grid_size)
    validate_four_color_matchings(grid_size, matchings)
    assert [matching.name for matching in matchings] == [
        "horizontal_even",
        "horizontal_odd",
        "vertical_even",
        "vertical_odd",
    ]
    assert [matching.edge_count for matching in matchings] == [18, 18, 18, 18]
    for matching in matchings:
        incident = np.concatenate([matching.tails, matching.heads])
        assert np.array_equal(np.sort(incident), np.arange(grid_size**2))

    plan = palindromic_strang_plan()
    assert [phase.matching_index for phase in plan] == [0, 1, 2, 3, 2, 1, 0]
    assert [phase.duration_fraction for phase in plan] == [
        0.5,
        0.5,
        0.5,
        1.0,
        0.5,
        0.5,
        0.5,
    ]
    assert [phase.matching_index for phase in plan] == [
        phase.matching_index for phase in reversed(plan)
    ]
    for matching_index in range(4):
        assert sum(
            phase.duration_fraction
            for phase in plan
            if phase.matching_index == matching_index
        ) == pytest.approx(1.0, abs=0.0)

    with pytest.raises(ValueError, match="even"):
        build_four_color_matchings(5)


def test_apply_matching_head_fractions_preserves_pair_totals_without_correction() -> None:
    rng = np.random.default_rng(260901)
    states = rng.dirichlet(np.ones(16), size=3)
    matching = build_four_color_matchings(4)[2]
    later = rng.uniform(0.0, 1.0, size=(3, matching.edge_count))
    before_pair_totals = states[:, matching.tails] + states[:, matching.heads]
    transformed = apply_matching_head_fractions(states, matching, later)
    after_pair_totals = (
        transformed[:, matching.tails] + transformed[:, matching.heads]
    )
    assert np.allclose(
        before_pair_totals, after_pair_totals, rtol=0.0, atol=7e-18
    )
    assert np.allclose(transformed.sum(axis=1), 1.0, rtol=0.0, atol=2e-16)
    assert np.all(transformed >= 0.0)
    assert np.allclose(
        transformed[:, matching.heads] / after_pair_totals,
        later,
        rtol=2e-15,
        atol=2e-16,
    )


def test_alpha1_spectral_density_normalizes_and_obeys_detailed_balance_semigroup() -> None:
    nodes, weights = np.polynomial.legendre.leggauss(192)
    z = 0.5 * (nodes + 1.0)
    uniform_weights = 0.5 * weights
    config = Alpha1SpectralConfig(
        absolute_tolerance=2e-14,
        relative_tolerance=2e-13,
        max_modes=512,
    )

    transition = evaluate_alpha1_spectral(
        0.27, z, 0.18, config=config
    )
    assert transition.diagnostics.converged
    assert float(np.dot(uniform_weights, transition.density)) == pytest.approx(
        1.0, abs=4e-13
    )
    assert np.all(transition.density > 0.0)

    xy = evaluate_alpha1_spectral(0.27, 0.73, 0.18, config=config)
    yx = evaluate_alpha1_spectral(0.73, 0.27, 0.18, config=config)
    assert float(xy.density) == pytest.approx(float(yx.density), abs=2e-14)

    first = evaluate_alpha1_spectral(0.27, z, 0.11, config=config)
    second = evaluate_alpha1_spectral(z, 0.73, 0.16, config=config)
    composed = float(np.dot(uniform_weights, first.density * second.density))
    direct = float(
        evaluate_alpha1_spectral(0.27, 0.73, 0.27, config=config).density
    )
    assert composed == pytest.approx(direct, abs=7e-13)


def test_alpha1_cdf_and_arrival_score_match_finite_differences_and_endpoints() -> None:
    config = Alpha1SpectralConfig(
        absolute_tolerance=1e-14,
        relative_tolerance=1e-13,
        max_modes=1024,
    )
    x = 0.31
    y = 0.58
    exposure = 0.22
    step = 2e-6
    center = evaluate_alpha1_spectral(x, y, exposure, config=config)
    left = evaluate_alpha1_spectral(x, y - step, exposure, config=config)
    right = evaluate_alpha1_spectral(x, y + step, exposure, config=config)
    cdf_derivative = (float(right.cdf) - float(left.cdf)) / (2.0 * step)
    log_density_derivative = (
        math.log(float(right.density)) - math.log(float(left.density))
    ) / (2.0 * step)
    assert cdf_derivative == pytest.approx(float(center.density), rel=2e-10)
    assert log_density_derivative == pytest.approx(
        float(center.arrival_score), rel=2e-9
    )

    endpoints = evaluate_alpha1_spectral(
        np.array([x, x]), np.array([0.0, 1.0]), exposure, config=config
    )
    assert np.array_equal(endpoints.cdf, np.array([0.0, 1.0]))
    assert endpoints.diagnostics.endpoint_cdf_count == 2


def test_spectral_mode_cap_fails_closed_and_never_clamps_negative_values() -> None:
    config = Alpha1SpectralConfig(
        absolute_tolerance=1e-14,
        relative_tolerance=0.0,
        max_modes=2,
    )
    with pytest.raises(SpectralConvergenceError) as captured:
        evaluate_alpha1_spectral(0.1, 0.9, 0.01, config=config)
    assert not captured.value.diagnostics.converged
    assert "mode cap" in captured.value.diagnostics.reason

    report = evaluate_alpha1_spectral(
        0.1, 0.9, 0.01, config=config, strict=False
    )
    assert not report.diagnostics.converged
    # This is the raw two-mode partial sum.  A clamp to zero would hide the
    # numerical failure and invalidate the gate.
    expected_partial_sum = 1.0 + 3.0 * math.exp(-0.02) * (-0.8) * 0.8
    assert float(report.density) == pytest.approx(expected_partial_sum, abs=1e-15)
    assert float(report.density) < 0.0
    assert report.diagnostics.negative_density_count == 1


def test_fixed_mode_torch_evaluator_matches_the_certified_float64_series() -> None:
    x = np.array([0.17, 0.43, 0.81], dtype=np.float64)
    y = np.array([0.22, 0.71, 0.64], dtype=np.float64)
    exposure = np.array([0.15, 0.38, 0.91], dtype=np.float64)
    numpy_result = evaluate_alpha1_spectral(x, y, exposure)
    torch_result = evaluate_alpha1_spectral_torch_fixed_modes(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(exposure),
        modes=numpy_result.diagnostics.modes_used,
    )
    assert torch.allclose(
        torch_result.density,
        torch.from_numpy(numpy_result.density),
        rtol=3e-15,
        atol=3e-15,
    )
    assert torch.allclose(
        torch_result.cdf,
        torch.from_numpy(numpy_result.cdf),
        rtol=3e-15,
        atol=3e-15,
    )
    assert torch.allclose(
        torch_result.arrival_score,
        torch.from_numpy(numpy_result.arrival_score),
        rtol=5e-15,
        atol=5e-15,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_spectral_evaluator_agrees_on_production_small_exposure() -> None:
    exposure = np.array([0.00011484375, 0.04501875, 0.91], dtype=np.float64)
    x = np.array([0.37, 0.43, 0.81], dtype=np.float64)
    y = np.array([0.3727, 0.483, 0.64], dtype=np.float64)
    oracle = evaluate_alpha1_spectral(x, y, exposure)
    result = evaluate_alpha1_spectral_torch_fixed_modes(
        torch.from_numpy(x).cuda(),
        torch.from_numpy(y).cuda(),
        torch.from_numpy(exposure).cuda(),
        modes=oracle.diagnostics.modes_used,
    )
    assert np.max(np.abs(result.density.cpu().numpy() - oracle.density) / np.maximum(1.0, np.abs(oracle.density))) <= 2e-6
    assert np.max(np.abs(result.arrival_score.cpu().numpy() - oracle.arrival_score) / np.maximum(1.0, np.abs(oracle.arrival_score))) <= 2e-5


def test_certified_inverse_cdf_replays_and_reports_no_latent_label() -> None:
    config = SpectralInverseCDFConfig(
        spectral=Alpha1SpectralConfig(
            absolute_tolerance=1e-13,
            relative_tolerance=1e-12,
            max_modes=512,
        ),
        cdf_residual_tolerance=2e-8,
        y_tolerance=1e-11,
        max_iterations=80,
    )
    x = np.linspace(0.15, 0.85, 12)
    exposure = np.full(12, 0.25)
    first = sample_alpha1_spectral_inverse_cdf(
        x, exposure, rng=np.random.default_rng(260902), config=config
    )
    second = sample_alpha1_spectral_inverse_cdf(
        x, exposure, rng=np.random.default_rng(260902), config=config
    )
    assert np.array_equal(first.uniforms, second.uniforms)
    assert np.array_equal(first.samples, second.samples)
    assert first.diagnostics.certified
    assert not first.diagnostics.latent_label_available
    assert (
        first.diagnostics.maximum_cdf_residual_bound
        <= config.cdf_residual_tolerance
    )
    checked = evaluate_alpha1_spectral(x, first.samples, exposure)
    assert np.max(np.abs(checked.cdf - first.uniforms)) < 2.1e-8


def test_latent_label_is_exact_component_score_and_validates_counts() -> None:
    m = np.array([5, 7, 0], dtype=np.int64)
    selected = np.array([2, 6, 0], dtype=np.int64)
    y = np.array([0.31, 0.72, 0.41])
    label = jacobi_latent_label(m, selected, y)
    expected = selected - m * y
    assert np.array_equal(label, expected)
    score = jacobi_component_relative_score(m, selected, y)
    assert np.allclose(y * (1.0 - y) * score, label, rtol=2e-16, atol=2e-16)

    step = 1e-7
    for count, successes, value, expected_score in zip(
        m[:2], selected[:2], y[:2], score[:2], strict=True
    ):
        def log_ratio(point: float) -> float:
            return float(
                successes * math.log(point)
                + (count - successes) * math.log1p(-point)
            )

        finite_difference = (log_ratio(value + step) - log_ratio(value - step)) / (
            2.0 * step
        )
        assert finite_difference == pytest.approx(expected_score, rel=2e-9)

    with pytest.raises(ValueError, match="0 <= L <= M"):
        jacobi_latent_label(2, 3, 0.5)
    with pytest.raises(ValueError, match="integers"):
        jacobi_latent_label(2.0, 1, 0.5)


def test_linear_teacher_identity_phase_exposure_and_flux_keep_head_orientation() -> None:
    y = np.array([0.13, 0.47, 0.82])
    exposure = np.array([0.07, 0.31, 0.8])
    density = linear_teacher_relative_density(y, exposure)
    score = linear_teacher_arrival_score(y, exposure)
    denoising_mean = linear_teacher_denoising_mean(y, exposure)
    assert np.all(density > 0.0)
    assert np.all(score > 0.0)
    assert np.allclose(
        denoising_mean,
        y * (1.0 - y) * score,
        rtol=2e-16,
        atol=2e-16,
    )

    step = 1e-7
    numeric_score = (
        np.log(linear_teacher_relative_density(y + step, exposure))
        - np.log(linear_teacher_relative_density(y - step, exposure))
    ) / (2.0 * step)
    assert np.allclose(numeric_score, score, rtol=5e-9, atol=7e-10)

    phase_exposure = jacobi_phase_exposure(
        np.array([0.25, 0.5]),
        2e-5,
        alpha=1.0,
        grid_spacing=1.0 / 28.0,
    )
    expected_exposure = 3.0 * 2e-5 / ((1.0 / 28.0) ** 2 * np.array([0.25, 0.5]))
    assert np.array_equal(phase_exposure, expected_exposure)

    flux = denoising_mean_to_mass_flux(
        denoising_mean,
        alpha=1.0,
        grid_spacing=1.0 / 28.0,
        schedule_value=0.4,
    )
    assert np.allclose(
        flux,
        6.0 * 0.4 * denoising_mean / (1.0 / 28.0) ** 2,
        rtol=2e-16,
        atol=0.0,
    )
    assert np.all(flux > 0.0)
