from __future__ import annotations

from fractions import Fraction
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist import d0_jacobi_rb_cuda_controls as _controls
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_dynkin import (
    CompensatedDynkinAccumulator,
    DynkinAccumulatorState,
    _initial_observable_ball,
    compute_dynkin_phase_drift,
    run_dynkin_refinement_shard,
    run_dynkin_tower_phase,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    canonical_refinement_transition_ids,
    refinement_observable_spec,
    refinement_phase_exposure,
    run_refinement_shard,
)


class _RecordingSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __call__(
        self, x: torch.Tensor, exposure: torch.Tensor, **kwargs: object
    ) -> SimpleNamespace:
        ids = kwargs["transition_ids"]
        assert isinstance(ids, torch.Tensor)
        self.calls.append(
            (
                x.detach().clone(),
                exposure.detach().clone(),
                ids.detach().clone(),
            )
        )
        # A lane-local deterministic stand-in.  The Dynkin observer must not
        # change any value passed to or returned from this sampler.
        jitter = (
            torch.remainder(ids.to(torch.int64), 17).to(torch.float64)
            * 2.0**-48
        )
        later = torch.clamp(x + jitter, 0.0, 1.0)
        count = int(x.numel())
        zero_i64 = torch.zeros((), dtype=torch.int64, device=x.device)
        zero_f64 = torch.zeros((), dtype=torch.float64, device=x.device)
        return SimpleNamespace(
            later_head_fraction=later,
            denoising_target=later - x,
            certificate_codes=torch.full(
                (count,), 15, dtype=torch.uint8, device=x.device
            ),
            fallback_mask=torch.zeros(count, dtype=torch.bool, device=x.device),
            strengthened_mask=torch.zeros(
                count, dtype=torch.bool, device=x.device
            ),
            mode_counts=torch.full(
                (count,), 128, dtype=torch.int32, device=x.device
            ),
            prefix_bits=torch.full(
                (count,), 64, dtype=torch.int32, device=x.device
            ),
            arb_fallback_reason_codes=torch.zeros(
                count, dtype=torch.uint8, device=x.device
            ),
            diagnostics={
                "maximum_cuda_launch_lanes": torch.as_tensor(
                    count, dtype=torch.int64, device=x.device
                ),
                "fused_authorizer_launch_count": torch.ones(
                    (), dtype=torch.int64, device=x.device
                ),
                "arb_fallback_elapsed_seconds": zero_f64,
                "fused_authorizer_elapsed_seconds": zero_f64,
                "candidate_elapsed_seconds": zero_f64,
                "resource_cap_count": zero_i64,
                "invalid_density_count": zero_i64,
                "approximation_count": zero_i64,
                "correction_count": zero_i64,
                "floor_count": zero_i64,
                "limiter_count": zero_i64,
                "projection_count": zero_i64,
                "renormalization_count": zero_i64,
                "nonfinite_count": zero_i64,
            },
        )


def _states(path_count: int) -> torch.Tensor:
    values = np.random.Generator(np.random.Philox(261161)).dirichlet(
        np.ones(784), size=path_count
    )
    return torch.as_tensor(values, dtype=torch.float64).contiguous()


@pytest.mark.parametrize("matching_index", range(4))
def test_phase_drift_matches_direct_closed_form_and_encloses_it(
    matching_index: int,
) -> None:
    states = _states(3).numpy()
    tails, heads = _controls._matching_arrays()[matching_index]
    total_np = states[:, tails] + states[:, heads]
    fraction_np = states[:, heads] / total_np
    total = torch.as_tensor(total_np, dtype=torch.float64).contiguous()
    fraction = torch.as_tensor(fraction_np, dtype=torch.float64).contiguous()
    exposure = refinement_phase_exposure(
        total, sample_steps=512, duration_fraction=0.5
    )
    observed = compute_dynkin_phase_drift(
        total,
        fraction,
        exposure,
        matching_index=matching_index,
    )

    z = 2.0 * fraction_np - 1.0
    p2 = (3.0 * z * z - 1.0) / 2.0
    exp2 = np.expm1(-2.0 * exposure.numpy())
    exp6 = np.expm1(-6.0 * exposure.numpy())
    spec = refinement_observable_spec()
    weight_delta = spec.fourier_weights[:, heads] - spec.fourier_weights[:, tails]
    linear = np.sum(
        (
            0.5 * total_np * z * exp2
        )[:, None, :] * weight_delta[None, :, :],
        axis=2,
    )
    quadratic = np.sum(
        total_np**2 * p2 * exp6 / 3.0, axis=1, keepdims=True
    )
    cubic = np.sum(
        total_np**3 * p2 * exp6 / 2.0, axis=1, keepdims=True
    )
    expected = np.concatenate((linear, quadratic, cubic), axis=1)
    np.testing.assert_allclose(
        observed.center.numpy(), expected, rtol=0.0, atol=4.0e-16
    )
    assert np.all(observed.lower.numpy() <= expected)
    assert np.all(expected <= observed.upper.numpy())
    assert observed.diagnostics["uses_future_state"] == 0
    assert observed.diagnostics["fitted_coefficient_count"] == 0


def test_zero_pair_mass_or_duration_is_an_exact_zero_drift() -> None:
    total = torch.as_tensor(
        [[0.0] * 196 + [0.25] * 196], dtype=torch.float64
    ).contiguous()
    fraction = torch.full_like(total, 0.75)
    exposure = torch.as_tensor(
        [[1.0] * 196 + [0.0] * 196], dtype=torch.float64
    ).contiguous()
    result = compute_dynkin_phase_drift(
        total, fraction, exposure, matching_index=0
    )
    assert torch.equal(result.center, torch.zeros_like(result.center))
    assert torch.equal(result.error_radius, torch.zeros_like(result.error_radius))


def test_compensated_accumulator_is_deterministic_and_resumable() -> None:
    initial = torch.as_tensor(
        [[1.0e16, *([0.0] * 9)]], dtype=torch.float64
    ).contiguous()
    accumulator = CompensatedDynkinAccumulator.from_initial_observables(initial)
    from mnist.d0_jacobi_rb_dynkin import DynkinPhaseDriftBatch

    unit = torch.as_tensor([[1.0, *([0.0] * 9)]], dtype=torch.float64)
    zero = torch.zeros_like(unit)
    drift = DynkinPhaseDriftBatch(unit, zero, {})
    accumulator.add_(drift)
    saved = accumulator.state().clone()
    resumed = CompensatedDynkinAccumulator(saved)
    resumed.add_(drift)

    uninterrupted = CompensatedDynkinAccumulator.from_initial_observables(initial)
    uninterrupted.add_(drift)
    uninterrupted.add_(drift)
    expected = uninterrupted.state()
    actual = resumed.state()
    assert torch.equal(actual.center, expected.center)
    assert torch.equal(actual.compensation, expected.compensation)
    assert torch.equal(actual.error_radius, expected.error_radius)


def test_p2_root_rounding_is_enclosed_by_full_arithmetic_ball() -> None:
    total = torch.full((1, 392), 1.0e-3, dtype=torch.float64)
    root_fraction = (1.0 + float(1.0 / math.sqrt(3.0))) / 2.0
    fraction = torch.full_like(total, root_fraction)
    exposure = torch.full_like(total, 0.123)
    result = compute_dynkin_phase_drift(
        total.contiguous(),
        fraction.contiguous(),
        exposure.contiguous(),
        matching_index=0,
    )

    from mnist.diag_d0_jacobi_rb_dynkin_power_confirmation import (
        _independent_phase_formula_arb,
    )

    spec = refinement_observable_spec()
    tails, heads = _controls._matching_arrays()[0]
    _, oracle_lower, oracle_upper = _independent_phase_formula_arb(
        total.numpy(),
        fraction.numpy(),
        exposure.numpy(),
        tail_index=tails,
        head_index=heads,
        weights=spec.fourier_weights,
    )
    assert np.all(oracle_lower[:, 8:] >= result.lower.numpy()[:, 8:])
    assert np.all(oracle_upper[:, 8:] <= result.upper.numpy()[:, 8:])
    # The rounded centre lands exactly on P2=0 in this fixture.  A zero
    # radius would therefore repeat the bug this regression guards against.
    assert np.all(result.error_radius.numpy()[:, 8:] > 0.0)


def test_initial_observable_and_kahan_rounding_are_enclosed() -> None:
    states = _states(2)
    initial = _initial_observable_ball(
        states, spec=refinement_observable_spec()
    )
    assert torch.all(initial.radius > 0.0)

    raw = torch.as_tensor(
        [[1.0e16, *([0.0] * 9)]], dtype=torch.float64
    ).contiguous()
    accumulator = CompensatedDynkinAccumulator.from_initial_observables(raw)
    from mnist.d0_jacobi_rb_dynkin import DynkinPhaseDriftBatch

    unit = torch.as_tensor([[1.0, *([0.0] * 9)]], dtype=torch.float64)
    accumulator.add_(DynkinPhaseDriftBatch(unit, torch.zeros_like(unit), {}))
    state = accumulator.state()
    exact = Fraction.from_float(1.0e16) + Fraction(1, 1)
    lower = Fraction.from_float(
        float(state.center[0, 0] - state.error_radius[0, 0])
    )
    upper = Fraction.from_float(
        float(state.center[0, 0] + state.error_radius[0, 0])
    )
    assert lower <= exact <= upper


def test_dynkin_shard_preserves_parent_transition_and_state_hashes() -> None:
    states = _states(2)
    paths = (7, 3)
    plain_sampler = _RecordingSampler()
    observed_sampler = _RecordingSampler()
    plain = run_refinement_shard(
        states,
        path_ids=paths,
        sample_steps=128,
        start_step=0,
        root_seed=261161,
        panel_namespace="hash-preservation",
        profile=JacobiRBCudaProfile(),
        sampler=plain_sampler,
        checkpoint_steps=(8,),
        capture_phase_state_trace=True,
    )
    observed = run_dynkin_refinement_shard(
        states,
        path_ids=paths,
        sample_steps=128,
        start_step=0,
        root_seed=261161,
        panel_namespace="hash-preservation",
        profile=JacobiRBCudaProfile(),
        sampler=observed_sampler,
        checkpoint_steps=(8,),
        capture_phase_state_trace=True,
    )
    assert observed.batch_output_sha256 == plain.batch_output_sha256
    assert observed.batch_final_state_sha256 == plain.batch_final_state_sha256
    assert observed.batch_certificate_sha256 == plain.batch_certificate_sha256
    assert torch.equal(observed.final_states, plain.final_states)
    assert len(plain_sampler.calls) == len(observed_sampler.calls) == 56
    for plain_call, observed_call in zip(
        plain_sampler.calls, observed_sampler.calls, strict=True
    ):
        assert all(
            torch.equal(left, right)
            for left, right in zip(plain_call, observed_call, strict=True)
        )
    checkpoint = observed.observable_checkpoints[0]
    np.testing.assert_array_equal(
        checkpoint.raw_values, plain.observable_checkpoints[0].values
    )
    assert checkpoint.dynkin_values.shape == (2, 10)
    assert np.all(np.isfinite(checkpoint.dynkin_values))
    assert np.all(checkpoint.dynkin_error_radius >= 0.0)
    assert observed.diagnostics["dynkin_transition_hash_preserved"] == 1
    assert observed.diagnostics["dynkin_state_hash_preserved"] == 1


def test_dynkin_shard_resume_requires_and_replays_accumulator() -> None:
    states = _states(1)
    first = run_dynkin_refinement_shard(
        states,
        path_ids=(5,),
        sample_steps=128,
        start_step=0,
        root_seed=261161,
        panel_namespace="resume",
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
        checkpoint_steps=(8,),
    )
    with pytest.raises(ValueError, match="require accumulator_state"):
        run_dynkin_refinement_shard(
            first.final_states,
            path_ids=(5,),
            sample_steps=128,
            start_step=8,
            root_seed=261161,
            panel_namespace="resume",
            profile=JacobiRBCudaProfile(),
            sampler=_RecordingSampler(),
        )
    resumed = run_dynkin_refinement_shard(
        first.final_states,
        path_ids=(5,),
        sample_steps=128,
        start_step=8,
        root_seed=261161,
        panel_namespace="resume",
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
        checkpoint_steps=(16,),
        accumulator_state=first.accumulator_state,
    )
    repeated = run_dynkin_refinement_shard(
        first.final_states,
        path_ids=(5,),
        sample_steps=128,
        start_step=8,
        root_seed=261161,
        panel_namespace="resume",
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
        checkpoint_steps=(16,),
        accumulator_state=first.accumulator_state.clone(),
    )
    assert torch.equal(
        resumed.accumulator_state.center, repeated.accumulator_state.center
    )
    assert resumed.observable_checkpoints[0].dynkin_values_sha256 == (
        repeated.observable_checkpoints[0].dynkin_values_sha256
    )


def test_one_phase_control_exposes_raw_and_standardized_tower_values() -> None:
    states = _states(2)
    ids = canonical_refinement_transition_ids(
        (2, 9),
        sample_steps=512,
        outer_step=0,
        phase=0,
        device=torch.device("cpu"),
    )
    result = run_dynkin_tower_phase(
        states,
        matching_index=0,
        duration_fraction=0.5,
        sample_steps=512,
        rng_key=(261161, "tower"),
        transition_ids=ids,
        profile=JacobiRBCudaProfile(),
        sampler=_RecordingSampler(),
    )
    assert result.raw_before_values.shape == (2, 10)
    assert result.raw_after_values.shape == (2, 10)
    assert result.standardized_residual.shape == (2, 10)
    assert result.drift_center.shape == (2, 10)
    assert result.diagnostics["certified_count"] == 2 * 392
    assert result.to_record()["transition_output_sha256"] == (
        result.transition_output_sha256
    )
    assert torch.equal(result.before_values, result.raw_before_values)
    assert torch.equal(result.residual, result.standardized_residual)


def test_accumulator_state_shape_contract_fails_closed() -> None:
    bad = DynkinAccumulatorState(
        center=torch.zeros((1, 10), dtype=torch.float64),
        compensation=torch.zeros((1, 10), dtype=torch.float64),
        error_radius=torch.zeros((1, 10), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match=r"does not match"):
        run_dynkin_refinement_shard(
            _states(2),
            path_ids=(1, 2),
            sample_steps=128,
            start_step=8,
            root_seed=261161,
            panel_namespace="bad-accumulator",
            profile=JacobiRBCudaProfile(),
            sampler=_RecordingSampler(),
            accumulator_state=bad,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_directed_cuda_exp24_phase_drift_agrees_with_cpu() -> None:
    states = _states(2).numpy()
    tails, heads = _controls._matching_arrays()[3]
    total_cpu = torch.as_tensor(
        states[:, tails] + states[:, heads], dtype=torch.float64
    ).contiguous()
    fraction_cpu = torch.as_tensor(
        states[:, heads] / (states[:, tails] + states[:, heads]),
        dtype=torch.float64,
    ).contiguous()
    exposure_cpu = refinement_phase_exposure(
        total_cpu, sample_steps=1024, duration_fraction=1.0
    )
    expected = compute_dynkin_phase_drift(
        total_cpu, fraction_cpu, exposure_cpu, matching_index=3
    )
    observed = compute_dynkin_phase_drift(
        total_cpu.cuda().contiguous(),
        fraction_cpu.cuda().contiguous(),
        exposure_cpu.cuda().contiguous(),
        matching_index=3,
        cuda_profile=JacobiRBCudaProfile(),
    )
    np.testing.assert_allclose(
        observed.center.cpu().numpy(),
        expected.center.numpy(),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert observed.certificate_mask is not None
    assert bool(observed.certificate_mask.all().cpu())
    certificate = observed.diagnostics["cuda_exponential_certificate"]
    assert certificate["authorizing_directed_dd_exp24"] == 1
    assert certificate["libdevice_transcendental_authorization"] == 0
