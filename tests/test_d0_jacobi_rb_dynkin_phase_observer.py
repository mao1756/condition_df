from __future__ import annotations

import numpy as np
import pytest
import torch

from mnist import d0_jacobi_rb_cuda_controls as _controls
from mnist.d0_jacobi_rb_dynkin import compute_dynkin_phase_drift
from mnist.d0_jacobi_rb_dynkin_phase_observer import (
    combine_dynkin_phase_residual,
    compute_advisory_global_phase_increment,
    compute_dynkin_phase_observed_increment,
    compute_dynkin_phase_observed_increment_from_states,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    refinement_observable_spec,
    refinement_phase_exposure,
)
from mnist.d0_jacobi_rb_strang_refinement_gate import (
    whole_path_max_t_intervals,
)


def _phase_fixture(
    matching_index: int, path_count: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.Generator(
        np.random.Philox([261171, matching_index, path_count])
    )
    states = generator.dirichlet(np.ones(784), size=path_count)
    tails, heads = _controls._matching_arrays()[matching_index]
    pair_total = states[:, tails] + states[:, heads]
    earlier = states[:, heads] / pair_total
    later = np.clip(
        earlier + 0.15 * (generator.random(earlier.shape) - 0.5),
        0.0,
        1.0,
    )
    final = states.copy()
    final[:, tails] = pair_total * (1.0 - later)
    final[:, heads] = pair_total * later
    return states, final, pair_total, earlier, later


def _as_tensor(value: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.as_tensor(
        value, dtype=torch.float64, device=device
    ).contiguous()


@pytest.mark.parametrize("matching_index", range(4))
def test_phase_local_increment_matches_closed_form_and_encloses_it(
    matching_index: int,
) -> None:
    _, _, pair_total, earlier, later = _phase_fixture(matching_index)
    r = _as_tensor(pair_total)
    x = _as_tensor(earlier)
    y = _as_tensor(later)
    result = compute_dynkin_phase_observed_increment(
        r,
        x,
        y,
        matching_index=matching_index,
        quantile_lower=y,
        quantile_upper=y,
    )

    tails, heads = _controls._matching_arrays()[matching_index]
    weight_delta = (
        refinement_observable_spec().fourier_weights[:, heads]
        - refinement_observable_spec().fourier_weights[:, tails]
    )
    difference = later - earlier
    common = difference * (earlier + later - 1.0)
    expected = np.concatenate(
        (
            np.sum(
                weight_delta[None, :, :]
                * pair_total[:, None, :]
                * difference[:, None, :],
                axis=2,
            ),
            (
                2.0
                * np.sum(pair_total**2 * common, axis=1, keepdims=True)
            ),
            (
                3.0
                * np.sum(pair_total**3 * common, axis=1, keepdims=True)
            ),
        ),
        axis=1,
    )
    np.testing.assert_allclose(
        result.center.numpy(), expected, rtol=0.0, atol=3.0e-17
    )
    assert np.all(result.lower.numpy() <= expected)
    assert np.all(expected <= result.upper.numpy())
    assert result.diagnostics["deterministic_pairwise_reduction"] == 1
    assert result.diagnostics["tolerance_zeroing_used"] == 0


@pytest.mark.parametrize(
    ("matching_index", "expected"),
    (
        (0, (False, False, False, False, True, True, True, True)),
        (1, (False, False, False, False, True, True, True, True)),
        (2, (True, True, True, True, False, False, False, False)),
        (3, (True, True, True, True, False, False, False, False)),
    ),
)
def test_matching_invariants_are_bitwise_positive_zero(
    matching_index: int, expected: tuple[bool, ...]
) -> None:
    _, _, pair_total, earlier, later = _phase_fixture(matching_index)
    y = _as_tensor(later)
    result = compute_dynkin_phase_observed_increment(
        _as_tensor(pair_total),
        _as_tensor(earlier),
        y,
        matching_index=matching_index,
        quantile_lower=y,
        quantile_upper=y,
    )
    assert result.structural_zero_mask.tolist() == [*expected, False, False]
    invariant = result.structural_zero_mask
    assert torch.equal(
        result.center[:, invariant],
        torch.zeros_like(result.center[:, invariant]),
    )
    assert torch.equal(
        result.error_radius[:, invariant],
        torch.zeros_like(result.error_radius[:, invariant]),
    )
    assert not torch.signbit(result.center[:, invariant]).any()
    assert not torch.signbit(result.error_radius[:, invariant]).any()


def test_quantile_bounds_are_propagated_and_corruption_is_rejected() -> None:
    _, _, pair_total, earlier, later = _phase_fixture(0, path_count=2)
    r = _as_tensor(pair_total)
    x = _as_tensor(earlier)
    y = _as_tensor(later)
    lower = torch.nextafter(y, torch.zeros_like(y))
    upper = torch.nextafter(y, torch.ones_like(y))
    widened = compute_dynkin_phase_observed_increment(
        r,
        x,
        y,
        matching_index=0,
        quantile_lower=lower,
        quantile_upper=upper,
    )
    exact = compute_dynkin_phase_observed_increment(
        r,
        x,
        y,
        matching_index=0,
        quantile_lower=y,
        quantile_upper=y,
    )
    assert torch.all(widened.quantile_enclosure_valid)
    assert torch.all(widened.error_radius >= exact.error_radius)
    assert torch.any(widened.error_radius > exact.error_radius)

    bad_lower = lower.clone()
    bad_lower[0, 0] = torch.nextafter(y[0, 0], torch.ones_like(y[0, 0]))
    with pytest.raises(ValueError, match="quantile enclosures"):
        compute_dynkin_phase_observed_increment(
            r,
            x,
            y,
            matching_index=0,
            quantile_lower=bad_lower,
            quantile_upper=upper,
        )
    nonfinite = upper.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_dynkin_phase_observed_increment(
            r,
            x,
            y,
            matching_index=0,
            quantile_lower=lower,
            quantile_upper=nonfinite,
        )


def test_zero_mass_no_motion_and_zero_duration_are_exact_zero() -> None:
    total = torch.zeros((2, 392), dtype=torch.float64)
    earlier = torch.full_like(total, 0.25)
    later = torch.full_like(total, 0.75)
    zero_mass = compute_dynkin_phase_observed_increment(
        total,
        earlier,
        later,
        matching_index=0,
        quantile_lower=later,
        quantile_upper=later,
    )
    assert torch.equal(zero_mass.center, torch.zeros_like(zero_mass.center))
    assert torch.equal(
        zero_mass.error_radius, torch.zeros_like(zero_mass.error_radius)
    )

    positive = torch.full_like(total, 0.1)
    no_motion = compute_dynkin_phase_observed_increment(
        positive,
        earlier,
        earlier,
        matching_index=0,
        quantile_lower=earlier,
        quantile_upper=earlier,
    )
    assert torch.equal(no_motion.center, torch.zeros_like(no_motion.center))
    assert torch.equal(
        no_motion.error_radius, torch.zeros_like(no_motion.error_radius)
    )

    zero_duration = compute_dynkin_phase_observed_increment(
        positive,
        earlier,
        later,
        matching_index=0,
        quantile_lower=later,
        quantile_upper=later,
        duration_fraction=0.0,
    )
    assert torch.equal(
        zero_duration.center, torch.zeros_like(zero_duration.center)
    )
    assert torch.equal(
        zero_duration.error_radius,
        torch.zeros_like(zero_duration.error_radius),
    )


def test_residual_combiner_canonicalizes_invariant_analytic_drift() -> None:
    _, _, pair_total, earlier, later = _phase_fixture(0, path_count=3)
    r = _as_tensor(pair_total)
    x = _as_tensor(earlier)
    y = _as_tensor(later)
    observed = compute_dynkin_phase_observed_increment(
        r,
        x,
        y,
        matching_index=0,
        quantile_lower=y,
        quantile_upper=y,
    )
    exposure = refinement_phase_exposure(
        r, sample_steps=512, duration_fraction=0.5
    )
    drift = compute_dynkin_phase_drift(
        r, x, exposure, matching_index=0, standardized=False
    )
    residual = combine_dynkin_phase_residual(observed, drift)
    invariant = observed.structural_zero_mask
    assert torch.equal(
        residual.drift_center[:, invariant],
        torch.zeros_like(residual.drift_center[:, invariant]),
    )
    assert torch.equal(
        residual.drift_error_radius[:, invariant],
        torch.zeros_like(residual.drift_error_radius[:, invariant]),
    )
    assert torch.equal(
        residual.center[:, invariant],
        torch.zeros_like(residual.center[:, invariant]),
    )
    assert torch.equal(
        residual.error_radius[:, invariant],
        torch.zeros_like(residual.error_radius[:, invariant]),
    )
    assert (
        residual.diagnostics["maximum_standardized_error_radius"]
        == float(torch.max(residual.error_radius))
    )


def test_saved_global_subtraction_degeneracy_is_only_advisory() -> None:
    before, after, _, _, later = _phase_fixture(0, path_count=128)
    before_tensor = _as_tensor(before)
    after_tensor = _as_tensor(after)
    later_tensor = _as_tensor(later)
    local = compute_dynkin_phase_observed_increment_from_states(
        before_tensor,
        after_tensor,
        matching_index=0,
        quantile_lower=later_tensor,
        quantile_upper=later_tensor,
        later_head_fraction=later_tensor,
    )
    global_advisory = compute_advisory_global_phase_increment(
        before_tensor, after_tensor
    )
    invariant = local.structural_zero_mask
    assert torch.equal(
        local.center[:, invariant],
        torch.zeros_like(local.center[:, invariant]),
    )
    assert torch.count_nonzero(global_advisory.center[:, invariant]) > 0
    assert torch.all(global_advisory.lower[:, invariant] <= 0.0)
    assert torch.all(global_advisory.upper[:, invariant] >= 0.0)
    discrepancy = torch.abs(global_advisory.center - local.center)
    assert torch.all(
        discrepancy
        <= global_advisory.error_radius + local.error_radius
    )
    assert global_advisory.diagnostics["authorizing"] == 0


def test_exact_zero_members_pass_unchanged_max_t_degeneracy_guard() -> None:
    exact = np.zeros(128, dtype=np.float64)
    result = whole_path_max_t_intervals(
        {"horizontal_y_invariant": exact},
        seed=261172,
        confidence=0.99,
        reps=100,
    )
    member = result["members"][0]
    assert member["simultaneous_lower"] == 0.0
    assert member["simultaneous_upper"] == 0.0
    assert member["contains_zero"] == 1

    nonzero = np.full(128, 1.0e-14, dtype=np.float64)
    with pytest.raises(ValueError, match="nonzero degenerate"):
        whole_path_max_t_intervals(
            {"broken_invariant": nonzero},
            seed=261172,
            confidence=0.99,
            reps=100,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_observer_agrees_with_cpu_enclosures() -> None:
    _, _, pair_total, earlier, later = _phase_fixture(3, path_count=2)
    cpu_y = _as_tensor(later)
    cpu = compute_dynkin_phase_observed_increment(
        _as_tensor(pair_total),
        _as_tensor(earlier),
        cpu_y,
        matching_index=3,
        quantile_lower=cpu_y,
        quantile_upper=cpu_y,
    )
    cuda_y = _as_tensor(later, "cuda")
    cuda = compute_dynkin_phase_observed_increment(
        _as_tensor(pair_total, "cuda"),
        _as_tensor(earlier, "cuda"),
        cuda_y,
        matching_index=3,
        quantile_lower=cuda_y,
        quantile_upper=cuda_y,
    )
    measured = cuda.center.cpu()
    assert torch.all(measured >= cpu.lower)
    assert torch.all(measured <= cpu.upper)
    assert torch.equal(
        cuda.structural_zero_mask.cpu(), cpu.structural_zero_mask
    )
    assert torch.equal(
        cuda.center[:, cuda.structural_zero_mask].cpu(),
        torch.zeros_like(cpu.center[:, cpu.structural_zero_mask]),
    )
