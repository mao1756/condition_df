from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from mnist.d0_dirichlet_score import (
    D0DirichletScorePotentialUNet,
    D0LinearSplinePotential,
    carre_du_champ_from_gradients,
    cubic_bspline_basis,
    dirichlet_score_objective,
    edge_difference_channels,
    edge_endpoint_channels,
    edge_incidence,
    edge_ratio_channels,
    exact_generator_from_derivatives,
    fit_linear_spline_baseline,
    harmonic_mobility_exact,
    physical_flux_from_edge_score,
    rademacher_edge_probes,
    run_operator_preflight,
    sample_teacher_dirichlet,
    stein_residual_from_derivatives,
    teacher_cell_hessian,
    teacher_cell_score,
    teacher_dirichlet_parameters,
    teacher_edge_score,
    teacher_fourier_pattern,
    teacher_log_relative_potential,
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


def _states(batch: int = 3, grid_size: int = 4, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(19)
    raw = torch.rand((batch, grid_size * grid_size), generator=generator, dtype=dtype) + 0.2
    return raw / raw.sum(dim=1, keepdim=True)


def test_edge_geometry_orientation_and_adjoint() -> None:
    state = torch.arange(1, 17, dtype=torch.float64).reshape(1, -1)
    tail, head = edge_endpoint_channels(state, 4)
    assert tail.shape == head.shape == (1, 2, 4, 4)
    assert head[0, 0, 0, 0] == state[0, 1]
    assert head[0, 0, 0, 3] == state[0, 0]
    assert head[0, 1, 0, 0] == state[0, 4]
    assert head[0, 1, 3, 0] == state[0, 0]

    cells = torch.randn((2, 16), dtype=torch.float64)
    edges = torch.randn((2, 2, 4, 4), dtype=torch.float64)
    lhs = (edge_difference_channels(cells, 4) * edges).sum()
    rhs = (cells * edge_incidence(edges)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-12, rtol=1e-12)


def test_mobility_and_ratio_match_closed_form() -> None:
    config = _config()
    states = _states(batch=2)
    tail, head = edge_endpoint_channels(states, 4)
    theta = harmonic_mobility_exact(states, config)
    ratio = edge_ratio_channels(states, 4)
    expected_theta = 3.0 * tail * head / (tail + head)
    assert torch.allclose(theta, expected_theta, atol=1e-14, rtol=1e-14)
    assert torch.allclose(ratio, (tail - head) / (tail + head), atol=1e-14, rtol=1e-14)

    zero = torch.zeros((1, 16), dtype=torch.float64)
    assert torch.equal(harmonic_mobility_exact(zero, config), torch.zeros((1, 2, 4, 4), dtype=torch.float64))
    assert torch.equal(edge_ratio_channels(zero, 4), torch.zeros((1, 2, 4, 4), dtype=torch.float64))


class _QuadraticPotential(nn.Module):
    def __init__(self, matrix: torch.Tensor, linear: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.register_buffer("linear", torch.zeros(matrix.shape[0], dtype=matrix.dtype) if linear is None else linear)

    def forward(self, tau, states, labels):
        del tau, labels
        return 0.5 * torch.einsum("bi,ij,bj->b", states, self.matrix, states) + states @ self.linear


def _exact_edge_basis_probes(batch: int, grid_size: int, dtype: torch.dtype) -> torch.Tensor:
    edge_count = 2 * grid_size * grid_size
    probes = torch.zeros((edge_count, batch, edge_count), dtype=dtype)
    diagonal = torch.arange(edge_count)
    probes[diagonal, :, diagonal] = math.sqrt(float(edge_count))
    return torch.stack(
        [
            probes[:, :, : grid_size * grid_size].reshape(edge_count, batch, grid_size, grid_size),
            probes[:, :, grid_size * grid_size :].reshape(edge_count, batch, grid_size, grid_size),
        ],
        dim=2,
    )


def test_hutchinson_objective_matches_dense_quadratic_generator() -> None:
    config = _config()
    states = _states(batch=2)
    generator = torch.Generator().manual_seed(23)
    raw = torch.randn((16, 16), generator=generator, dtype=torch.float64)
    matrix = (raw + raw.T) / 16.0
    linear = torch.randn((16,), generator=generator, dtype=torch.float64)
    model = _QuadraticPotential(matrix, linear)
    probes = _exact_edge_basis_probes(2, 4, torch.float64)
    result = dirichlet_score_objective(
        model,
        torch.zeros(2, dtype=torch.float64),
        states,
        torch.zeros(2, dtype=torch.long),
        config,
        probes,
        create_graph=False,
    )

    gradient = states @ matrix.T + linear
    hessian = matrix.expand(2, -1, -1)
    gamma = carre_du_champ_from_gradients(gradient, gradient, states, config)
    generator_value = exact_generator_from_derivatives(states, gradient, hessian, config)
    cell_count = 16.0
    expected = gamma / (2.0 * cell_count * cell_count) + generator_value / (cell_count * cell_count)
    assert torch.allclose(result.per_sample, expected, atol=2e-11, rtol=2e-11)


def test_objective_accepts_single_and_multi_probe_and_backpropagates() -> None:
    config = _config()
    states = _states(batch=2, dtype=torch.float32)
    labels = torch.tensor([3, 3])
    tau = torch.full((2,), 0.5 * natural_horizon(config))
    model = D0DirichletScorePotentialUNet(config, base_channels=4)
    single = rademacher_edge_probes(1, 2, 4, device="cpu", dtype=torch.float32, generator=torch.Generator().manual_seed(3))[0]
    result = dirichlet_score_objective(model, tau, states, labels, config, single)
    assert result.per_sample.shape == (2,)
    assert result.edge_score.shape == (2, 2, 4, 4)
    assert torch.equal(result.potential, torch.zeros_like(result.potential))
    assert torch.equal(result.per_sample, torch.zeros_like(result.per_sample))
    result.loss.backward()
    assert model.out.weight.grad is not None
    assert torch.isfinite(model.out.weight.grad).all()
    assert float(model.out.weight.grad.abs().sum()) > 0.0

    multi = single.unsqueeze(0).repeat(3, 1, 1, 1, 1)
    again = dirichlet_score_objective(model, tau, states, labels, config, multi, create_graph=False)
    assert torch.allclose(again.per_sample, result.per_sample.detach(), atol=1e-7, rtol=1e-7)


def test_model_uses_smooth_position_sensitive_energy_map() -> None:
    config = _config()
    model = D0DirichletScorePotentialUNet(config, base_channels=4)
    assert model.periodic_coordinates.shape == (1, 4, 4, 4)
    assert not any(isinstance(module, nn.ReLU) for module in model.modules())
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            assert module.padding_mode == "circular"

    # Move away from the deliberately zero baseline and verify first and
    # second state derivatives exist and are finite.
    with torch.no_grad():
        model.out.weight.fill_(0.1)
    states = _states(batch=1, dtype=torch.float64).requires_grad_(True)
    model = model.double()
    value = model(torch.tensor([0.4 * natural_horizon(config)], dtype=torch.float64), states, torch.tensor([3]))
    gradient = torch.autograd.grad(value.sum(), states, create_graph=True)[0]
    hvp = torch.autograd.grad((gradient * torch.randn_like(gradient)).sum(), states)[0]
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(hvp).all()


def test_teacher_helpers_match_autograd_and_flux_conversion() -> None:
    config = _config()
    fractions = torch.tensor([0.1, 0.9], dtype=torch.float64)
    states = sample_teacher_dirichlet(fractions, 4, seed=77, dtype=torch.float64)
    assert torch.allclose(states.sum(dim=1), torch.ones(2, dtype=torch.float64), atol=1e-14)
    params = teacher_dirichlet_parameters(fractions, 4)
    assert torch.all(params > 0.0)
    pattern = teacher_fourier_pattern(4)
    assert abs(float(pattern.mean())) <= 1e-15
    assert float(pattern.abs().max()) == pytest.approx(1.0)

    required = states.detach().clone().requires_grad_(True)
    potential = teacher_log_relative_potential(required, fractions)
    gradient = torch.autograd.grad(potential.sum(), required, create_graph=True)[0]
    rows = []
    for index in range(16):
        rows.append(torch.autograd.grad(gradient[:, index].sum(), required, retain_graph=True)[0])
    hessian = torch.stack(rows, dim=1)
    assert torch.allclose(gradient, teacher_cell_score(states, fractions), atol=1e-11, rtol=1e-11)
    assert torch.allclose(hessian, teacher_cell_hessian(states, fractions), atol=1e-10, rtol=1e-10)
    analytic_edge = teacher_edge_score(states, fractions)
    assert torch.allclose(analytic_edge, edge_difference_channels(gradient, 4), atol=1e-11, rtol=1e-11)
    flux = physical_flux_from_edge_score(analytic_edge, states, config, time_change=torch.tensor([1.0, 0.5], dtype=torch.float64))
    expected = 2.0 * 16.0 * torch.tensor([1.0, 0.5], dtype=torch.float64)[:, None, None, None] * harmonic_mobility_exact(states, config) * analytic_edge
    assert torch.allclose(flux, expected, atol=1e-11, rtol=1e-11)


def test_spline_basis_partition_and_frozen_baseline_gauge() -> None:
    config = _config()
    x = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    basis = cubic_bspline_basis(x)
    assert basis.shape == (101, 8)
    assert torch.all(basis >= 0.0)
    assert torch.allclose(basis.sum(dim=1), torch.ones(101, dtype=torch.float64), atol=2e-14)
    coefficients = torch.randn((8, 16), dtype=torch.float64)
    model = D0LinearSplinePotential(config, coefficients)
    assert torch.allclose(model.coefficients.mean(dim=1), torch.zeros(8, dtype=torch.float64), atol=1e-14)
    states = _states(batch=5)
    tau = x[:5] * natural_horizon(config)
    shift = torch.full_like(states, 0.25)
    # Total-mass shifts are a gauge because every coefficient row is zero-sum.
    assert torch.allclose(model(tau, states, None), model(tau, states + shift, None), atol=1e-13)


def test_matrix_free_linear_baseline_fit_satisfies_normal_equation() -> None:
    config = _config()
    states = _states(batch=48)
    tau = torch.linspace(0.01, 0.99, 48, dtype=torch.float64) * natural_horizon(config)
    fit = fit_linear_spline_baseline(states, tau, config, tolerance=1e-8, max_iterations=1000)
    assert fit.iterations > 0
    assert math.isfinite(fit.relative_residual)
    assert fit.relative_residual <= 1e-8
    assert fit.converged
    assert torch.isfinite(fit.model.coefficients).all()


def test_stein_residual_is_zero_for_analytic_teacher_in_expectation() -> None:
    config = _config()
    fractions = torch.linspace(0.05, 0.95, 8192, dtype=torch.float64)
    states = sample_teacher_dirichlet(fractions, 4, seed=1234, dtype=torch.float64)
    score_gradient = teacher_cell_score(states, fractions)
    generator = torch.Generator().manual_seed(51)
    witness_gradient = torch.randn((1, 16), generator=generator, dtype=torch.float64).expand_as(states)
    witness_gradient = witness_gradient - witness_gradient.mean(dim=1, keepdim=True)
    witness_hessian = torch.zeros((states.shape[0], 16, 16), dtype=torch.float64)
    residual = stein_residual_from_derivatives(
        score_gradient, witness_gradient, witness_hessian, states, config
    )
    stderr = residual.std(unbiased=True) / math.sqrt(float(residual.numel()))
    assert abs(float(residual.mean())) <= 5.0 * float(stderr) + 1e-8


def test_operator_preflight_reports_named_passing_checks() -> None:
    result = run_operator_preflight(_config(), hutchinson_probes=4096)
    assert result["grid_size"] == 4
    assert result["evaluation_device"] == "cpu"
    assert result["passed"]
    assert set(result["checks"]) == {
        "constant_gauge",
        "mass_gauge",
        "product_rule",
        "hutchinson_trace",
        "finite",
    }

