from __future__ import annotations

import math

import pytest
import torch

from mnist.d0_dirichlet_score import (
    edge_difference_channels,
    edge_endpoint_channels,
    harmonic_mobility_exact,
    physical_flux_from_edge_score,
)
from mnist.d0_score_boundary_controls import (
    BOUNDED_TEACHER_VERSION,
    BOUNDARY_SMOOTH_MODEL_VERSION,
    ORTHOGONAL_HADAMARD_PROBE_VERSION,
    D0BoundarySmoothPotentialUNet,
    bounded_teacher_anchor_indices,
    bounded_teacher_cell_hessian,
    bounded_teacher_cell_score,
    bounded_teacher_density_ratio,
    bounded_teacher_edge_score,
    bounded_teacher_hessian_vector_product,
    bounded_teacher_log_relative_potential,
    bounded_teacher_physical_flux,
    bounded_teacher_weights,
    legacy_log_barrier_trace_drift_coefficient,
    orthogonal_hadamard_edge_probes,
    run_boundary_operator_preflight,
    run_boundary_model_facet_preflight,
    run_facet_ray_preflight,
    run_legacy_log_barrier_preflight,
    sample_bounded_teacher_mixture,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


torch.set_num_threads(1)


def _config(grid_size: int = 4) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
        num_steps=8,
        source_lowfreq_size=2,
        ot_lowres_size=2,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-12,
        limiter_fraction=1.0,
        condition_on_source=False,
        flux_parameterization="edge",
    )


def _interior_states(
    batch: int = 3,
    grid_size: int = 4,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(902)
    raw = torch.rand(
        (batch, grid_size * grid_size), generator=generator, dtype=dtype
    ) + 0.15
    return raw / raw.sum(dim=1, keepdim=True)


def test_boundary_control_versions_and_anchor_order_are_frozen() -> None:
    assert BOUNDARY_SMOOTH_MODEL_VERSION
    assert BOUNDED_TEACHER_VERSION
    assert ORTHOGONAL_HADAMARD_PROBE_VERSION
    anchors = bounded_teacher_anchor_indices(4)
    # Quarter-grid anchors in TL, TR, BL, BR order.
    assert torch.equal(anchors.cpu(), torch.tensor([5, 7, 13, 15]))
    assert torch.equal(
        bounded_teacher_anchor_indices(8).cpu(),
        torch.tensor([18, 22, 50, 54]),
    )


def test_bounded_teacher_weights_and_density_ratio_are_exactly_normalized() -> None:
    fractions = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float64)
    weights = bounded_teacher_weights(fractions)
    assert weights.shape == (4, 4)
    assert torch.all(weights >= 0.0)
    assert torch.all(weights[1:-1] > 0.0)
    assert torch.allclose(
        weights.sum(dim=1), torch.ones(4, dtype=torch.float64), atol=2e-15
    )
    # Time genuinely changes the mixture rather than being a decorative input.
    assert not torch.allclose(weights[0], weights[-1])

    uniform = torch.full((4, 16), 1.0 / 16.0, dtype=torch.float64)
    ratio = bounded_teacher_density_ratio(uniform, fractions, epsilon=0.5)
    # Each one-count tilted component has density ratio N*s_anchor, hence the
    # convex mixture is exactly normalized at the simplex barycenter.
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=2e-15, rtol=0.0)
    assert torch.all(ratio >= 0.5)

    # Monte Carlo integration under the uniform Dirichlet reference provides
    # an independent normalization check away from the barycenter.
    generator = torch.Generator().manual_seed(903)
    raw = torch._standard_gamma(
        torch.ones((32_768, 16), dtype=torch.float64), generator=generator
    )
    states = raw / raw.sum(dim=1, keepdim=True)
    repeated = torch.full((states.shape[0],), 0.6, dtype=torch.float64)
    ratio = bounded_teacher_density_ratio(states, repeated, epsilon=0.5)
    stderr = ratio.std(unbiased=True) / math.sqrt(float(ratio.numel()))
    assert float(ratio.mean()) == pytest.approx(1.0, abs=5.0 * float(stderr))
    assert float(ratio.min()) >= 0.5


def test_bounded_teacher_mixture_sampling_is_deterministic_and_matches_moments() -> None:
    count = 40_000
    fraction = 0.65
    fractions = torch.full((count,), fraction, dtype=torch.float64)
    first, tilted, choices = sample_bounded_teacher_mixture(
        fractions,
        4,
        seed=904,
        dtype=torch.float64,
        epsilon=0.5,
        return_components=True,
    )
    second = sample_bounded_teacher_mixture(
        fractions, 4, seed=904, dtype=torch.float64, epsilon=0.5
    )
    assert torch.equal(first, second)
    assert first.shape == (count, 16)
    assert torch.all(first > 0.0)
    assert torch.allclose(
        first.sum(dim=1), torch.ones(count, dtype=torch.float64), atol=2e-14
    )
    assert tilted.dtype == torch.bool and choices.dtype == torch.long
    assert torch.equal(choices[~tilted], torch.full_like(choices[~tilted], -1))
    assert bool(((choices[tilted] >= 0) & (choices[tilted] < 4)).all())
    assert float(tilted.double().mean()) == pytest.approx(0.5, abs=0.01)

    weights = bounded_teacher_weights(torch.tensor([fraction], dtype=torch.float64))[0]
    anchors = bounded_teacher_anchor_indices(4)
    expected = torch.full(
        (16,), (1.0 - 0.5) / 16.0 + 0.5 / 17.0, dtype=torch.float64
    )
    expected[anchors] += 0.5 * weights / 17.0
    empirical = first.mean(dim=0)
    assert torch.allclose(empirical, expected, atol=8e-4, rtol=0.0)
    empirical_choices = torch.bincount(choices[tilted], minlength=4).double()
    empirical_choices /= empirical_choices.sum()
    assert torch.allclose(empirical_choices, weights, atol=0.015, rtol=0.0)


def test_bounded_teacher_analytic_derivatives_hvp_edge_score_and_flux() -> None:
    config = _config()
    states = _interior_states(batch=3).requires_grad_(True)
    fractions = torch.tensor([0.15, 0.55, 0.95], dtype=torch.float64)
    potential = bounded_teacher_log_relative_potential(
        states, fractions, epsilon=0.5
    )
    gradient = torch.autograd.grad(
        potential.sum(), states, create_graph=True
    )[0]
    rows = [
        torch.autograd.grad(
            gradient[:, index].sum(), states, retain_graph=True
        )[0]
        for index in range(16)
    ]
    hessian = torch.stack(rows, dim=1)
    analytic_gradient = bounded_teacher_cell_score(
        states.detach(), fractions, epsilon=0.5
    )
    analytic_hessian = bounded_teacher_cell_hessian(
        states.detach(), fractions, epsilon=0.5
    )
    assert torch.allclose(gradient, analytic_gradient, atol=2e-12, rtol=2e-12)
    assert torch.allclose(hessian, analytic_hessian, atol=2e-11, rtol=2e-11)

    vectors = torch.randn(
        states.shape, generator=torch.Generator().manual_seed(905), dtype=torch.float64
    )
    expected_hvp = torch.einsum("bij,bj->bi", analytic_hessian, vectors)
    actual_hvp = bounded_teacher_hessian_vector_product(
        states.detach(), fractions, vectors, epsilon=0.5
    )
    assert torch.allclose(actual_hvp, expected_hvp, atol=2e-11, rtol=2e-11)

    edge = bounded_teacher_edge_score(states.detach(), fractions, epsilon=0.5)
    assert torch.allclose(
        edge,
        edge_difference_channels(analytic_gradient, 4),
        atol=2e-12,
        rtol=2e-12,
    )
    rates = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    flux = bounded_teacher_physical_flux(
        states.detach(), fractions, config, epsilon=0.5, time_change=rates
    )
    expected_flux = physical_flux_from_edge_score(
        edge, states.detach(), config, time_change=rates
    )
    assert torch.allclose(flux, expected_flux, atol=2e-11, rtol=2e-11)


def test_boundary_smooth_model_uses_log1p_and_has_bounded_face_derivatives() -> None:
    config = _config()
    model = D0BoundarySmoothPotentialUNet(config, base_channels=4).double()
    states = torch.zeros((1, 16), dtype=torch.float64)
    states[0, 1:] = 1.0 / 15.0
    inputs = model._inputs(  # noqa: SLF001 - the channel contract is scientific provenance.
        torch.tensor([0.5 * natural_horizon(config)], dtype=torch.float64),
        states,
        torch.tensor([3]),
    )
    density = inputs[:, 0]
    smooth_log = inputs[:, 1]
    assert torch.equal(density[0, 0, 0], torch.tensor(0.0, dtype=torch.float64))
    assert torch.equal(smooth_log[0, 0, 0], torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(smooth_log, torch.log1p(density), atol=0.0, rtol=0.0)
    assert torch.isfinite(inputs).all()

    with torch.no_grad():
        model.out.weight.fill_(0.1)
    gradients: list[float] = []
    incident_fluxes: list[float] = []
    for epsilon in (1e-5, 1e-8):
        state = torch.full(
            (1, 16), (1.0 - epsilon) / 15.0, dtype=torch.float64
        )
        state[0, 0] = epsilon
        state.requires_grad_(True)
        value = model(
            torch.tensor([0.5 * natural_horizon(config)], dtype=torch.float64),
            state,
            torch.tensor([3]),
        )
        gradient = torch.autograd.grad(value.sum(), state)[0]
        edge_flux = harmonic_mobility_exact(state.detach(), config) * edge_difference_channels(
            gradient, 4
        )
        incident = torch.stack(
            (
                edge_flux[0, 0, 0, 0],
                edge_flux[0, 1, 0, 0],
                edge_flux[0, 0, 0, 3],
                edge_flux[0, 1, 3, 0],
            )
        )
        gradients.append(float(gradient[0, 0].abs()))
        incident_fluxes.append(float(incident.abs().max()))
    assert all(math.isfinite(value) for value in gradients + incident_fluxes)
    assert gradients[1] <= 2.0 * max(gradients[0], 1e-12)
    assert incident_fluxes[1] <= 0.01 * max(incident_fluxes[0], 1e-30)


def test_orthogonal_hadamard_probes_are_rademacher_reproducible_and_exact() -> None:
    first = orthogonal_hadamard_edge_probes(
        32,
        2,
        4,
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(906),
    )
    second = orthogonal_hadamard_edge_probes(
        32,
        2,
        4,
        device="cpu",
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(906),
    )
    assert torch.equal(first, second)
    assert first.shape == (32, 2, 2, 4, 4)
    assert set(first.unique().tolist()) == {-1.0, 1.0}
    flat = first.flatten(2)
    expected = 32.0 * torch.eye(32, dtype=torch.float64)
    for batch in range(2):
        gram = flat[:, batch] @ flat[:, batch].T
        assert torch.equal(gram, expected)
        covariance = flat[:, batch].T @ flat[:, batch]
        assert torch.equal(covariance, expected)


def test_boundary_operator_preflight_rejects_legacy_barrier_and_passes_smooth_model() -> None:
    config = _config()
    facet = run_facet_ray_preflight(config)
    assert facet["passed"]
    assert facet["checks"]["conormal_log_log_slope"]["value"] >= 0.90
    assert facet["checks"]["conormal_four_decade_decay"]["value"] <= 1e-3
    model_facet = run_boundary_model_facet_preflight(config)
    assert model_facet["passed"]
    assert model_facet["model_version"] == BOUNDARY_SMOOTH_MODEL_VERSION
    assert model_facet["incident_flux_loglog_slope"] >= 0.90
    assert model_facet["incident_flux_endpoint_ratio"] <= 1e-3

    states = _interior_states(batch=3)
    coefficient = legacy_log_barrier_trace_drift_coefficient(states, config)
    tail, head = edge_endpoint_channels(states, 4)
    expected_coefficient = -(6.0 / 16.0) * (1.0 / (tail + head)).flatten(1).sum(dim=1)
    assert torch.allclose(coefficient, expected_coefficient, atol=2e-12, rtol=2e-12)
    assert torch.all(coefficient < 0.0)

    legacy = run_legacy_log_barrier_preflight(config, num_states=4096)
    assert legacy["passed"]
    assert legacy["admissible"] is False
    assert legacy["checks"]["fixture_rejected"]["passed"]
    assert legacy["empirical_relative_error"] <= 0.10

    report = run_boundary_operator_preflight(config, device="cpu")
    assert report["passed"]
    assert report["facet_ray"]["passed"]
    assert report["model_facet_ray"]["passed"]
    assert report["legacy_log_barrier"]["passed"]
    assert report["legacy_log_barrier"]["admissible"] is False
    assert report["operator"]["passed"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bounded_teacher_cpu_cuda_metrics_agree() -> None:
    config = _config()
    states = _interior_states(batch=4, dtype=torch.float64)
    fractions = torch.tensor([0.1, 0.3, 0.7, 0.9], dtype=torch.float64)
    cpu_score = bounded_teacher_edge_score(states, fractions)
    cpu_flux = bounded_teacher_physical_flux(states, fractions, config)
    cuda_states = states.cuda()
    cuda_fractions = fractions.cuda()
    cuda_score = bounded_teacher_edge_score(cuda_states, cuda_fractions).cpu()
    cuda_flux = bounded_teacher_physical_flux(
        cuda_states, cuda_fractions, config
    ).cpu()
    assert torch.allclose(cuda_score, cpu_score, atol=2e-12, rtol=2e-12)
    assert torch.allclose(cuda_flux, cpu_flux, atol=2e-12, rtol=2e-12)
