"""Small smoke tests for the manuscript algorithm implementations.

Run with
    python test_smoke_wasserstein_conditioning_algorithms.py
"""

from __future__ import annotations

import numpy as np

from wasserstein_conditioning_algorithms import (
    shortest_periodic_displacement,
    simulate_gaussian_terminal_em,
    simulate_nonlinear_cylinder_quadrature_em,
    simulate_wasserstein_mc_sinkhorn_em,
    sinkhorn_plan,
)


def main() -> None:
    rng = np.random.default_rng(123)

    # Utility sanity check.
    disp = shortest_periodic_displacement(np.array([[0.95]]), np.array([[0.05]]))
    assert np.allclose(disp, np.array([[-0.1]])), disp

    # Algorithm 1.
    sim1 = simulate_gaussian_terminal_em(
        masses=[0.6, 0.4],
        target_positions=[[0.25], [0.75]],
        lambda_=1.0,
        horizon=0.2,
        step_size=0.1,
        initial_positions=[[0.1], [0.9]],
        drift_mode="wrapped",
        rng=rng,
    )
    assert sim1.positions.shape == (3, 2, 1)
    assert np.all((0.0 <= sim1.positions) & (sim1.positions < 1.0))

    # Algorithm 2.
    phi1 = lambda x: np.cos(2 * np.pi * np.asarray(x)[..., 0])
    phi2 = lambda x: np.sin(2 * np.pi * np.asarray(x)[..., 0])
    nodes = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    weights = np.full(4, 0.25)
    sim2 = simulate_nonlinear_cylinder_quadrature_em(
        masses=[0.5, 0.5],
        observables=[phi1, phi2],
        target_vector=[0.2, -0.1],
        lambda_=0.5,
        horizon=0.2,
        step_size=0.1,
        initial_positions=[[0.1], [0.7]],
        quadrature_nodes=nodes,
        quadrature_weights=weights,
        grid_shape=16,
        rng=rng,
    )
    assert sim2.positions.shape == (3, 2, 1)
    assert np.all((0.0 <= sim2.positions) & (sim2.positions < 1.0))

    # Sinkhorn marginal check.
    cost = np.array([[0.1, 0.4], [0.3, 0.2]])
    plan = sinkhorn_plan(cost, [0.5, 0.5], [0.4, 0.6], epsilon=0.1, iterations=50)
    assert np.allclose(plan.sum(axis=1), np.array([0.5, 0.5]), atol=1e-4)
    assert np.allclose(plan.sum(axis=0), np.array([0.4, 0.6]), atol=1e-4)

    # Algorithm 3.
    sim3 = simulate_wasserstein_mc_sinkhorn_em(
        masses=[0.5, 0.5],
        target_positions=[[0.2], [0.8]],
        target_masses=[0.5, 0.5],
        lambda_=1.0,
        epsilon=0.05,
        horizon=0.2,
        step_size=0.1,
        initial_positions=[[0.05], [0.95]],
        terminal_samples=4,
        sinkhorn_iterations=20,
        rng=rng,
    )
    assert sim3.positions.shape == (3, 2, 1)
    assert np.all((0.0 <= sim3.positions) & (sim3.positions < 1.0))

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
