from __future__ import annotations

import numpy as np
import pytest
import torch

import mnist.d0_jacobi_rb_spectral as rb_spectral
from mnist.d0_jacobi_denoising import Alpha1SpectralConfig, evaluate_alpha1_spectral
from mnist.d0_jacobi_rb_spectral import (
    JacobiRBCertificationError,
    JacobiRBSpectralProfile,
    evaluate_alpha1_rb_torch_fixed_modes,
    philox_uniform_prefix,
    propose_alpha1_rb_transition_batch_torch_intervals,
    reconstruct_pair_masses,
    resolve_alpha1_pair_phase_inputs,
    sample_alpha1_rb_transition_batch,
    sample_alpha1_rb_transition_batch_torch,
)


@pytest.fixture(scope="module")
def strict_profile() -> JacobiRBSpectralProfile:
    return JacobiRBSpectralProfile()


def test_philox_prefix_replay_and_transition_local_refinement() -> None:
    key = (261121, "prefix-isolation")
    first = philox_uniform_prefix(key, sample_index=7, bits=64)
    assert first == philox_uniform_prefix(key, sample_index=7, bits=64)

    # Refining another transition must not advance or otherwise perturb the
    # transition-local Philox stream for sample seven.
    philox_uniform_prefix(key, sample_index=2, bits=1024)
    assert first == philox_uniform_prefix(key, sample_index=7, bits=64)

    refined_numerator, refined_bits, _ = philox_uniform_prefix(
        key, sample_index=7, bits=128
    )
    assert refined_bits == 128
    assert refined_numerator >> 64 == first[0]
    assert philox_uniform_prefix(key, sample_index=8, bits=64) != first


def test_strict_batch_replay_has_correct_rounding_certificate(
    strict_profile: JacobiRBSpectralProfile,
) -> None:
    x = np.asarray([0.37, 0.5], dtype=np.float64)
    u = np.asarray([0.5, 1.0], dtype=np.float64)
    first = sample_alpha1_rb_transition_batch(
        x, u, rng_key=(261121, "strict-replay"), profile=strict_profile
    )
    replay = sample_alpha1_rb_transition_batch(
        x, u, rng_key=(261121, "strict-replay"), profile=strict_profile
    )

    np.testing.assert_array_equal(first.later_head_fraction, replay.later_head_fraction)
    np.testing.assert_array_equal(first.denoising_target, replay.denoising_target)
    np.testing.assert_array_equal(first.certificate_codes, replay.certificate_codes)
    # Bit zero is the general certificate and bit three is the unique
    # binary64 round-to-nearest-even certificate.
    assert np.all((first.certificate_codes & np.uint8(0b1001)) == np.uint8(0b1001))
    assert first.diagnostics.certified
    assert first.diagnostics.correctly_rounded_count == first.diagnostics.active_count
    assert first.diagnostics.active_count == 2
    assert np.all(first.quantile_lower <= first.later_head_fraction)
    assert np.all(first.later_head_fraction <= first.quantile_upper)
    assert np.all(first.target_lower <= first.denoising_target)
    assert np.all(first.denoising_target <= first.target_upper)


def test_zero_duration_and_zero_pair_are_exact_noops(
    strict_profile: JacobiRBSpectralProfile,
) -> None:
    inputs = resolve_alpha1_pair_phase_inputs(
        tail_mass=np.asarray([0.0, 0.2, 0.3]),
        head_mass=np.asarray([0.0, 0.3, 0.2]),
        integrated_schedule_time=np.asarray([0.1, 0.0, 0.0]),
        grid_spacing=1.0 / 28.0,
    )
    np.testing.assert_array_equal(inputs.active_mask, [False, False, False])
    np.testing.assert_array_equal(inputs.exposure, 0.0)
    assert inputs.head_fraction[0] == 0.0

    batch = sample_alpha1_rb_transition_batch(
        inputs.head_fraction,
        inputs.exposure,
        rng_key="all-inactive",
        profile=strict_profile,
    )
    np.testing.assert_array_equal(batch.later_head_fraction, inputs.head_fraction)
    np.testing.assert_array_equal(batch.denoising_target, 0.0)
    np.testing.assert_array_equal(batch.certificate_codes, 0)
    np.testing.assert_array_equal(batch.prefix_bits, 0)
    assert batch.diagnostics.active_count == 0
    assert batch.diagnostics.zero_duration_count == 3
    assert batch.diagnostics.correctly_rounded_count == 0


def test_pair_reconstruction_conserves_each_total_exactly() -> None:
    totals = np.asarray([0.0, 1e-12, 0.025, 0.25, 1.0], dtype=np.float64)
    fractions = np.asarray([0.0, 1.0, 0.17, 0.5, 0.93], dtype=np.float64)
    tail, head = reconstruct_pair_masses(totals, fractions)
    np.testing.assert_array_equal(tail + head, totals)
    assert np.all(tail >= 0.0)
    assert np.all(head >= 0.0)


def test_torch_fixed_mode_evaluator_agrees_with_cpu_spectral_reference() -> None:
    x = np.asarray([0.2, 0.5, 0.8], dtype=np.float64)
    y = np.asarray([0.25, 0.55, 0.75], dtype=np.float64)
    u = np.asarray([0.75, 1.0, 0.75], dtype=np.float64)
    reference = evaluate_alpha1_spectral(
        x,
        y,
        u,
        config=Alpha1SpectralConfig(
            absolute_tolerance=1e-14,
            relative_tolerance=1e-13,
            max_modes=1024,
        ),
    )
    measured = evaluate_alpha1_rb_torch_fixed_modes(
        torch.as_tensor(x),
        torch.as_tensor(y),
        torch.as_tensor(u),
        modes=64,
    )
    np.testing.assert_allclose(measured.cdf.numpy(), reference.cdf, atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        measured.density.numpy(), reference.density, atol=2e-14, rtol=0
    )
    expected_target = y * (1.0 - y) * reference.arrival_score
    np.testing.assert_allclose(
        measured.denoising_target.numpy(), expected_target, atol=2e-14, rtol=0
    )


def test_float64_device_intervals_are_screening_only_and_fail_closed() -> None:
    profile = JacobiRBSpectralProfile(
        device_proposal_modes=16,
        device_bisection_steps=2,
        authorize_device_intervals=False,
    )
    proposal = propose_alpha1_rb_transition_batch_torch_intervals(
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        rng_key=(261121, "device-screening"),
        profile=profile,
    )
    assert not bool(proposal.certified_mask.any())
    assert bool(proposal.fallback_mask.all())
    assert torch.all(proposal.certificate_codes == (2 | 4))


def test_arb_tail_radii_dominate_explicit_omitted_sums() -> None:
    if rb_spectral._arb is None:
        pytest.skip("python-flint is unavailable")
    previous = int(rb_spectral._flint_ctx.prec)
    try:
        rb_spectral._flint_ctx.prec = 256
        exposure = rb_spectral._arb_exact(0.5)
        cdf_radius, density_radius, conormal_radius = (
            rb_spectral._arb_geometric_tail_radii(8, exposure)
        )
        assert cdf_radius is not None
        assert density_radius is not None
        assert conormal_radius is not None
        cdf_upper = float(cdf_radius.upper())
        density_upper = float(density_radius.upper())
        conormal_upper = float(conormal_radius.upper())
    finally:
        rb_spectral._flint_ctx.prec = previous

    degrees = np.arange(8, 256, dtype=np.float64)
    decay = np.exp(-degrees * (degrees + 1.0) * 0.5)
    assert cdf_upper >= float(np.sum(decay))
    assert density_upper >= float(np.sum((2.0 * degrees + 1.0) * decay))
    assert conormal_upper >= float(
        np.sum(degrees * (2.0 * degrees + 1.0) * decay)
    )


def test_float_tail_screen_cannot_authorize_an_underbounded_arb_tail(
    monkeypatch,
) -> None:
    if rb_spectral._arb is None:
        pytest.skip("python-flint is unavailable")
    # Simulate a catastrophically underbounded ordinary-float screen.  The
    # independent Arb radius must reject every attempted early stop and hit
    # the deliberately tiny mode cap.
    monkeypatch.setattr(rb_spectral, "_tail_bounds", lambda *_args: (0.0, 0.0, 0.0))
    profile = JacobiRBSpectralProfile(
        max_modes=4,
        arb_precision_bits=(128,),
        max_arb_precision_bits=128,
    )
    with pytest.raises(JacobiRBCertificationError) as caught:
        rb_spectral._spectral_intervals_arb(
            0.37,
            0.42,
            0.01,
            profile=profile,
            cdf_only=True,
            precision_bits=128,
        )
    assert caught.value.diagnostics["failure_kind"] == "arb_mode_cap"


def test_strict_scalar_path_never_uses_float_tail_authorization(monkeypatch) -> None:
    if rb_spectral._arb is None:
        pytest.skip("python-flint is unavailable")

    def forbidden_fast_path(*_args, **_kwargs):
        raise AssertionError("strict production decisions must go through Arb")

    monkeypatch.setattr(rb_spectral, "_spectral_intervals_fast", forbidden_fast_path)
    result = sample_alpha1_rb_transition_batch(
        0.37,
        0.5,
        rng_key=(261121, "arb-only-regression"),
        profile=JacobiRBSpectralProfile(),
    )
    assert int(result.certificate_codes) == 15
    assert result.diagnostics.correctly_rounded_count == 1


def test_hybrid_api_skips_unauthorized_device_proposal_and_replays_noops(
    monkeypatch,
) -> None:
    if rb_spectral._arb is None:
        pytest.skip("python-flint is unavailable")
    profile = JacobiRBSpectralProfile(authorize_device_intervals=False)

    def forbidden_proposal(*_args, **_kwargs):
        raise AssertionError("unauthorized device proposal should be skipped")

    monkeypatch.setattr(
        rb_spectral,
        "propose_alpha1_rb_transition_batch_torch_intervals",
        forbidden_proposal,
    )
    x = np.asarray([0.37, 0.2], dtype=np.float64)
    u = np.asarray([0.5, 0.0], dtype=np.float64)
    key = (261121, "hybrid-direct-arb")
    hybrid = sample_alpha1_rb_transition_batch_torch(
        torch.as_tensor(x), torch.as_tensor(u), rng_key=key, profile=profile
    )
    scalar = sample_alpha1_rb_transition_batch(x, u, rng_key=key, profile=profile)
    np.testing.assert_array_equal(hybrid.later_head_fraction, scalar.later_head_fraction)
    np.testing.assert_array_equal(hybrid.denoising_target, scalar.denoising_target)
    np.testing.assert_array_equal(hybrid.certificate_codes, scalar.certificate_codes)
    assert hybrid.later_head_fraction[1] == x[1]
    assert hybrid.denoising_target[1] == 0.0
    assert hybrid.certificate_codes[1] == 0
    assert hybrid.prefix_bits[1] == 0


def test_low_precision_unbounded_arb_ball_escalates_to_certificate() -> None:
    if rb_spectral._arb is None:
        pytest.skip("python-flint is unavailable")
    # This is the exact first transition from the production resource probe.
    # Its 128-bit CDF cancellation ball overflows binary64, but later Arb
    # precisions resolve it rigorously.
    result = sample_alpha1_rb_transition_batch(
        0.03,
        0.00011484375000000002,
        rng_key=(261121, "benchmark-probe"),
        profile=JacobiRBSpectralProfile(),
    )
    assert float(result.later_head_fraction) == 0.03335912509808867
    assert float(result.denoising_target) == -15.233012652698532
    assert int(result.certificate_codes) == 15
    assert result.diagnostics.correctly_rounded_count == 1
