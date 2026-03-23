from __future__ import annotations

"""Numerical implementations of Algorithms 1--3 from the manuscript
"Conditioning a measure-valued diffusion on the Wasserstein space".

Coordinate convention
---------------------
The flat torus T^d is represented as the half-open cube [0, 1)^d, with
periodic wrapping performed componentwise modulo 1.

Implemented algorithms
----------------------
1. Euler--Maruyama for the Gaussian terminal condition.
2. Quadrature--Euler--Maruyama for the nonlinear cylinder terminal condition.
3. Monte Carlo--Sinkhorn Euler--Maruyama for the Wasserstein terminal
   condition.

The code is intentionally example-free so that it can be imported into a larger
project and wired to problem-specific targets later.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from conditioning_utils import (
    as_float_array as _as_float_array,
    build_time_grid as _build_time_grid,
    get_rng as _get_rng,
    image_shifts as _image_shifts,
    logsumexp as _logsumexp,
    validate_positions as _validate_positions,
    validate_probability_vector as _validate_probability_vector,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Observable = Callable[[ArrayLike], ArrayLike]

__all__ = [
    "ParticleSimulation",
    "TorusFourierHeatSemigroup",
    "wrap_torus",
    "shortest_periodic_displacement",
    "sinkhorn_plan",
    "regularized_ot_value",
    "simulate_gaussian_terminal_em",
    "simulate_nonlinear_cylinder_quadrature_em",
    "simulate_wasserstein_mc_sinkhorn_em",
]


@dataclass(frozen=True)
class ParticleSimulation:
    """Container for an atomic measure-valued particle simulation.

    Attributes
    ----------
    times:
        Time grid of shape ``(M + 1,)``.
    positions:
        Wrapped torus coordinates of shape ``(M + 1, n, d)``.
    masses:
        Frozen particle masses of shape ``(n,)``.
    drifts:
        Optional drift values on each time step, shape ``(M, n, d)``.
    lifted_positions:
        Optional unwrapped coordinates on the universal cover. This is populated
        by the Gaussian scheme when the Euclidean drift is used.
    """

    times: FloatArray
    positions: FloatArray
    masses: FloatArray
    drifts: Optional[FloatArray] = None
    lifted_positions: Optional[FloatArray] = None

    def atomic_configuration(self, step: int) -> Tuple[FloatArray, FloatArray]:
        """Return the masses and atom locations at a given time step."""
        if step < 0 or step >= len(self.times):
            raise IndexError(f"step {step} is outside [0, {len(self.times) - 1}]")
        return self.masses.copy(), self.positions[step].copy()


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def wrap_torus(x: ArrayLike) -> FloatArray:
    """Wrap coordinates onto the flat torus [0, 1)^d componentwise."""
    arr = np.asarray(x, dtype=np.float64)
    return np.mod(arr, 1.0)



def shortest_periodic_displacement(x: ArrayLike, y: ArrayLike) -> FloatArray:
    """Componentwise shortest displacement on the flat torus.

    The result lies in ``[-1/2, 1/2)`` in each component and broadcasts over the
    leading dimensions of ``x`` and ``y``.
    """
    dx = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return np.mod(dx + 0.5, 1.0) - 0.5



def _evaluate_observable(observable: Observable, points: FloatArray) -> FloatArray:
    """Evaluate a scalar observable on a collection of points.

    Observables may be vectorized (preferred) or pointwise callables.
    """
    try:
        values = np.asarray(observable(points), dtype=np.float64)
        if values.shape == (points.shape[0],):
            return values
        if values.shape == points.shape[:-1]:
            return values.reshape(points.shape[0])
        if values.size == points.shape[0]:
            return values.reshape(points.shape[0])
    except Exception:
        pass
    return np.array([float(np.asarray(observable(p), dtype=np.float64)) for p in points], dtype=np.float64)


# ---------------------------------------------------------------------------
# Algorithm 1: Gaussian terminal condition
# ---------------------------------------------------------------------------


def _periodized_gaussian_drift(
    x: FloatArray,
    target: FloatArray,
    masses: FloatArray,
    lambda_: float,
    tau: float,
    image_radius: int,
) -> FloatArray:
    """Approximate the wrapped Gaussian drift using a truncated image sum.

    This corresponds to the manuscript's suggestion to use a periodized terminal
    condition on the torus rather than the Euclidean closed form on the universal
    cover.
    """
    n, d = x.shape
    shifts = _image_shifts(image_radius, d)
    drift = np.empty_like(x)
    prefactor = -2.0 * lambda_ / (1.0 + 2.0 * lambda_ * tau)

    for i in range(n):
        deltas = x[i][None, :] - target[i][None, :] + shifts
        exponent = -0.5 * (lambda_ * masses[i] / (1.0 + 2.0 * lambda_ * tau)) * np.sum(deltas ** 2, axis=1)
        weights = np.exp(exponent - np.max(exponent))
        barycenter = np.sum(weights[:, None] * deltas, axis=0) / np.sum(weights)
        drift[i] = prefactor * barycenter
    return drift



def simulate_gaussian_terminal_em(
    masses: ArrayLike,
    target_positions: ArrayLike,
    lambda_: float,
    horizon: float,
    step_size: float,
    initial_positions: ArrayLike,
    *,
    drift_mode: str = "euclidean",
    image_radius: int = 1,
    rng: Optional[np.random.Generator] = None,
    store_drifts: bool = True,
) -> ParticleSimulation:
    """Algorithm 1: Euler--Maruyama for the Gaussian terminal condition.

    Parameters
    ----------
    masses:
        Frozen particle masses ``(s_1, ..., s_n)``.
    target_positions:
        Target atom locations ``\bar x_i``. For ``drift_mode='euclidean'`` these
        are interpreted on the universal cover; for ``drift_mode='wrapped'`` they
        are interpreted on the torus.
    lambda_:
        Terminal weight parameter from equation (3.32).
    horizon, step_size:
        Simulation horizon ``T`` and Euler step ``Δt``.
    initial_positions:
        Initial particle positions.
    drift_mode:
        ``'euclidean'`` uses the explicit drift from equation (3.38).
        ``'wrapped'`` uses a truncated torus image sum.
    image_radius:
        Radius of the image truncation for the wrapped drift.
    rng:
        Optional ``numpy.random.Generator``.
    store_drifts:
        Whether to retain the drift on every time step.
    """
    if lambda_ <= 0.0 or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be positive and finite")

    s = _validate_probability_vector(masses, name="masses")
    n = len(s)
    x0 = _validate_positions(initial_positions, n=n, name="initial_positions")
    target = _validate_positions(target_positions, n=n, name="target_positions")
    if x0.shape[1] != target.shape[1]:
        raise ValueError("initial_positions and target_positions must have the same dimension")

    rng = _get_rng(rng)
    m_steps, times = _build_time_grid(horizon, step_size)
    _, d = x0.shape

    wrapped_positions = np.empty((m_steps + 1, n, d), dtype=np.float64)
    wrapped_positions[0] = wrap_torus(x0)
    drift_history = np.empty((m_steps, n, d), dtype=np.float64) if store_drifts else None

    if drift_mode == "euclidean":
        state = x0.copy()
        lifted_positions = np.empty_like(wrapped_positions)
        lifted_positions[0] = state
    elif drift_mode == "wrapped":
        state = wrap_torus(x0)
        lifted_positions = None
        target = wrap_torus(target)
    else:
        raise ValueError("drift_mode must be either 'euclidean' or 'wrapped'")

    noise_scale = np.sqrt(2.0 * step_size / s)[:, None]

    for m in range(m_steps):
        tau = horizon - times[m]
        if drift_mode == "euclidean":
            drift = -2.0 * lambda_ * (state - target) / (1.0 + 2.0 * lambda_ * tau)
            state = state + drift * step_size + noise_scale * rng.normal(size=(n, d))
            wrapped_positions[m + 1] = wrap_torus(state)
            lifted_positions[m + 1] = state
        else:
            drift = _periodized_gaussian_drift(state, target, s, lambda_, tau, image_radius)
            state = wrap_torus(state + drift * step_size + noise_scale * rng.normal(size=(n, d)))
            wrapped_positions[m + 1] = state

        if drift_history is not None:
            drift_history[m] = drift

    return ParticleSimulation(
        times=times,
        positions=wrapped_positions,
        masses=s,
        drifts=drift_history,
        lifted_positions=lifted_positions,
    )


# ---------------------------------------------------------------------------
# Algorithm 2: nonlinear cylinder terminal condition
# ---------------------------------------------------------------------------


class TorusFourierHeatSemigroup:
    """Spectral heat-semigroup evaluator on the flat torus.

    The class approximates

        (P_r f)(x) = sum_k exp(-4π² |k|² r) hat_f(k) e^{2π i k·x}

    from samples of ``f`` on a regular grid. It is used for Algorithm 2 with
    ``f(x) = exp(i s_i η · Φ(x))``.
    """

    def __init__(
        self,
        observables: Sequence[Observable],
        *,
        dimension: int,
        grid_shape: int | Sequence[int] = 32,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if len(observables) == 0:
            raise ValueError("at least one observable is required")

        if isinstance(grid_shape, int):
            grid_shape = (grid_shape,) * dimension
        else:
            grid_shape = tuple(int(g) for g in grid_shape)
            if len(grid_shape) != dimension:
                raise ValueError("grid_shape must have one entry per torus dimension")

        if any(g <= 1 for g in grid_shape):
            raise ValueError("each grid dimension must be at least 2")

        self.dimension = int(dimension)
        self.observables = tuple(observables)
        self.grid_shape = grid_shape
        self.num_grid_points = int(np.prod(grid_shape))

        axes = [np.arange(g, dtype=np.float64) / g for g in grid_shape]
        mesh = np.meshgrid(*axes, indexing="ij")
        self.grid_points = np.stack([axis.ravel() for axis in mesh], axis=-1)

        values = [_evaluate_observable(phi, self.grid_points) for phi in self.observables]
        self.observable_values = np.stack(values, axis=1)
        self.num_observables = self.observable_values.shape[1]

        freq_axes = [np.fft.fftfreq(g, d=1.0 / g).astype(np.float64) for g in grid_shape]
        freq_mesh = np.meshgrid(*freq_axes, indexing="ij")
        self.k_vectors = np.stack([axis.ravel() for axis in freq_mesh], axis=-1)
        self.k_sq_norm = np.sum(self.k_vectors ** 2, axis=1)

        self._coefficients_cache: dict[tuple[float, ...], ComplexArray] = {}

    def _weight_key(self, weight: ArrayLike) -> tuple[float, ...]:
        arr = np.asarray(weight, dtype=np.float64).reshape(-1)
        if arr.shape != (self.num_observables,):
            raise ValueError(
                f"weight must have length {self.num_observables}; got shape {arr.shape}"
            )
        return tuple(np.round(arr, decimals=14))

    def _coefficients(self, weight: ArrayLike) -> ComplexArray:
        key = self._weight_key(weight)
        coeffs = self._coefficients_cache.get(key)
        if coeffs is not None:
            return coeffs

        weight_array = np.asarray(weight, dtype=np.float64).reshape(self.num_observables)
        phase = self.observable_values @ weight_array
        samples = np.exp(1j * phase).reshape(self.grid_shape)
        coeffs = np.fft.fftn(samples).reshape(-1) / self.num_grid_points
        self._coefficients_cache[key] = coeffs
        return coeffs

    def evaluate(self, weight: ArrayLike, time: float, x: ArrayLike) -> tuple[complex, ComplexArray]:
        """Return ``(P_t f)(x)`` and ``∇(P_t f)(x)`` for ``f(y)=exp(i weight·Φ(y))``."""
        if time < 0.0 or not np.isfinite(time):
            raise ValueError("time must be non-negative and finite")
        point = wrap_torus(np.asarray(x, dtype=np.float64).reshape(self.dimension))

        coeffs = self._coefficients(weight)
        decay = np.exp(-4.0 * np.pi ** 2 * self.k_sq_norm * time)
        phase = np.exp(2j * np.pi * (self.k_vectors @ point))
        terms = coeffs * decay * phase
        value = complex(np.sum(terms))
        gradient = np.sum((2j * np.pi * self.k_vectors) * terms[:, None], axis=0)
        return value, gradient.astype(np.complex128)



def _prefix_suffix_products(values: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    prefix = np.empty(len(values) + 1, dtype=np.complex128)
    suffix = np.empty(len(values) + 1, dtype=np.complex128)
    prefix[0] = 1.0 + 0.0j
    for i, value in enumerate(values):
        prefix[i + 1] = prefix[i] * value
    suffix[-1] = 1.0 + 0.0j
    for i in range(len(values) - 1, -1, -1):
        suffix[i] = suffix[i + 1] * values[i]
    return prefix, suffix



def simulate_nonlinear_cylinder_quadrature_em(
    masses: ArrayLike,
    observables: Sequence[Observable],
    target_vector: ArrayLike,
    lambda_: float,
    horizon: float,
    step_size: float,
    initial_positions: ArrayLike,
    quadrature_nodes: ArrayLike,
    quadrature_weights: ArrayLike,
    *,
    grid_shape: int | Sequence[int] = 32,
    rng: Optional[np.random.Generator] = None,
    store_drifts: bool = True,
) -> ParticleSimulation:
    """Algorithm 2: Quadrature--Euler--Maruyama for the cylinder target.

    Parameters
    ----------
    masses:
        Frozen masses ``(s_1, ..., s_n)``.
    observables:
        Smooth periodic observables ``φ_1, ..., φ_K`` on ``T^d``. Callables may
        be vectorized or pointwise.
    target_vector:
        Target vector ``a ∈ R^K``.
    quadrature_nodes, quadrature_weights:
        Nodes ``η_q`` and weights ``w_q`` approximating the Gaussian integral in
        equation (3.44). The weights should already include the Gaussian density
        factor.
    grid_shape:
        Fourier grid used to approximate the torus heat semigroup.
    """
    if lambda_ <= 0.0 or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be positive and finite")

    s = _validate_probability_vector(masses, name="masses")
    n = len(s)
    x0 = wrap_torus(_validate_positions(initial_positions, n=n, name="initial_positions"))
    _, d = x0.shape

    a = _as_float_array(target_vector, name="target_vector").reshape(-1)
    if len(observables) != len(a):
        raise ValueError(
            f"number of observables ({len(observables)}) must match len(target_vector) ({len(a)})"
        )

    eta = _as_float_array(quadrature_nodes, name="quadrature_nodes")
    if eta.ndim == 1:
        eta = eta[:, None]
    if eta.shape[1] != len(a):
        raise ValueError(
            f"quadrature_nodes must have shape (Q, {len(a)}) or (Q,)"
        )

    weights = _as_float_array(quadrature_weights, name="quadrature_weights").reshape(-1)
    if len(weights) != len(eta):
        raise ValueError("quadrature_nodes and quadrature_weights must have matching lengths")

    solver = TorusFourierHeatSemigroup(observables, dimension=d, grid_shape=grid_shape)
    rng = _get_rng(rng)
    m_steps, times = _build_time_grid(horizon, step_size)

    positions = np.empty((m_steps + 1, n, d), dtype=np.float64)
    positions[0] = x0
    drift_history = np.empty((m_steps, n, d), dtype=np.float64) if store_drifts else None
    noise_scale = np.sqrt(2.0 * step_size / s)[:, None]

    for m in range(m_steps):
        tau = horizon - times[m]
        x = positions[m]
        q_count = len(weights)

        h = np.empty((q_count, n), dtype=np.complex128)
        gamma = np.empty((q_count, n, d), dtype=np.complex128)

        for q in range(q_count):
            eta_q = eta[q]
            for i in range(n):
                value, grad = solver.evaluate(s[i] * eta_q, tau / s[i], x[i])
                h[q, i] = value
                gamma[q, i] = grad

        u_terms = np.empty(q_count, dtype=np.complex128)
        g_terms = np.empty((q_count, n, d), dtype=np.complex128)

        for q in range(q_count):
            prefactor = weights[q] * np.exp(-1j * float(np.dot(eta[q], a)))
            prefix, suffix = _prefix_suffix_products(h[q])
            u_terms[q] = prefactor * prefix[-1]
            for i in range(n):
                g_terms[q, i] = prefactor * gamma[q, i] * prefix[i] * suffix[i + 1]

        u_value = float(np.real(np.sum(u_terms)))
        if not np.isfinite(u_value) or u_value <= 1e-14:
            raise FloatingPointError(
                "quadrature approximation produced a near-zero denominator; "
                "increase quadrature accuracy or adjust the Fourier grid"
            )

        drift = np.empty((n, d), dtype=np.float64)
        g_sum = np.sum(g_terms, axis=0)
        for i in range(n):
            drift[i] = (2.0 / s[i]) * np.real(g_sum[i] / u_value)

        positions[m + 1] = wrap_torus(x + drift * step_size + noise_scale * rng.normal(size=(n, d)))
        if drift_history is not None:
            drift_history[m] = drift

    return ParticleSimulation(times=times, positions=positions, masses=s, drifts=drift_history)


# ---------------------------------------------------------------------------
# Algorithm 3: Wasserstein terminal condition
# ---------------------------------------------------------------------------


def sinkhorn_plan(
    cost_matrix: ArrayLike,
    source_masses: ArrayLike,
    target_masses: ArrayLike,
    epsilon: float,
    iterations: int,
    *,
    tol: Optional[float] = None,
) -> FloatArray:
    """Compute an entropically regularized transport plan with Sinkhorn iterations.

    The optimization problem matches equation (3.51). Because the manuscript's
    entropy is taken relative to ``s ⊗ \bar s`` and the marginals are fixed, the
    optimizer coincides with the usual entropic OT optimizer.
    """
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer")

    c = _as_float_array(cost_matrix, name="cost_matrix")
    if c.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    a = _validate_probability_vector(source_masses, name="source_masses")
    b = _validate_probability_vector(target_masses, name="target_masses")
    if c.shape != (len(a), len(b)):
        raise ValueError(
            f"cost_matrix must have shape ({len(a)}, {len(b)}); got {c.shape}"
        )

    log_a = np.log(a)
    log_b = np.log(b)
    log_k = -c / epsilon
    log_u = np.zeros_like(a)
    log_v = np.zeros_like(b)

    for _ in range(iterations):
        prev_u = log_u
        prev_v = log_v
        log_u = log_a - _logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_k + log_u[:, None], axis=0)
        if tol is not None:
            err = max(float(np.max(np.abs(log_u - prev_u))), float(np.max(np.abs(log_v - prev_v))))
            if err < tol:
                break

    log_pi = log_u[:, None] + log_k + log_v[None, :]
    pi = np.exp(log_pi)
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return pi



def regularized_ot_value(
    plan: ArrayLike,
    cost_matrix: ArrayLike,
    source_masses: ArrayLike,
    target_masses: ArrayLike,
    epsilon: float,
) -> float:
    """Evaluate the regularized OT objective from equation (3.51)."""
    pi = _as_float_array(plan, name="plan")
    c = _as_float_array(cost_matrix, name="cost_matrix")
    a = _validate_probability_vector(source_masses, name="source_masses")
    b = _validate_probability_vector(target_masses, name="target_masses")
    if pi.shape != c.shape or pi.shape != (len(a), len(b)):
        raise ValueError("plan, cost_matrix, and masses have incompatible shapes")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")

    cost_term = float(np.sum(pi * c))
    positive = pi > 0.0
    log_reference = np.log(a)[:, None] + np.log(b)[None, :]
    entropy_term = float(np.sum(pi[positive] * (np.log(pi[positive]) - log_reference[positive] - 1.0)))
    return cost_term + epsilon * entropy_term



def simulate_wasserstein_mc_sinkhorn_em(
    masses: ArrayLike,
    target_positions: ArrayLike,
    target_masses: ArrayLike,
    lambda_: float,
    epsilon: float,
    horizon: float,
    step_size: float,
    initial_positions: ArrayLike,
    terminal_samples: int,
    sinkhorn_iterations: int,
    *,
    rng: Optional[np.random.Generator] = None,
    sinkhorn_tol: Optional[float] = None,
    store_drifts: bool = True,
) -> ParticleSimulation:
    """Algorithm 3: Monte Carlo--Sinkhorn Euler--Maruyama.

    Parameters
    ----------
    masses:
        Source masses ``(s_1, ..., s_n)``.
    target_positions, target_masses:
        Target atomic measure ``\bar ν = Σ_j \bar s_j δ_{y_j}``.
    lambda_, epsilon:
        Terminal weight parameter and entropic regularization strength.
    terminal_samples:
        Number of Monte Carlo samples ``R`` used for the expectation in (3.57).
    sinkhorn_iterations:
        Number of Sinkhorn updates used to approximate ``π^*_ε``.
    """
    if lambda_ <= 0.0 or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be positive and finite")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    if terminal_samples <= 0:
        raise ValueError("terminal_samples must be a positive integer")
    if sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be a positive integer")

    s = _validate_probability_vector(masses, name="masses")
    n = len(s)
    x0 = wrap_torus(_validate_positions(initial_positions, n=n, name="initial_positions"))
    target = wrap_torus(_validate_positions(target_positions, n=None, name="target_positions"))
    target_s = _validate_probability_vector(target_masses, name="target_masses")

    if target.shape[0] != len(target_s):
        raise ValueError("target_positions and target_masses must have matching lengths")
    if target.shape[1] != x0.shape[1]:
        raise ValueError("initial_positions and target_positions must have the same dimension")

    rng = _get_rng(rng)
    m_steps, times = _build_time_grid(horizon, step_size)
    _, d = x0.shape

    positions = np.empty((m_steps + 1, n, d), dtype=np.float64)
    positions[0] = x0
    drift_history = np.empty((m_steps, n, d), dtype=np.float64) if store_drifts else None
    em_noise_scale = np.sqrt(2.0 * step_size / s)[:, None]

    for m in range(m_steps):
        tau = horizon - times[m]
        x = positions[m]

        log_weights = np.empty(terminal_samples, dtype=np.float64)
        gradients = np.empty((terminal_samples, n, d), dtype=np.float64)

        terminal_noise_scale = np.sqrt(2.0 * tau / s)[:, None]

        for r in range(terminal_samples):
            y = wrap_torus(x + terminal_noise_scale * rng.normal(size=(n, d)))
            displacements = shortest_periodic_displacement(y[:, None, :], target[None, :, :])
            cost_matrix = np.sum(displacements ** 2, axis=-1)

            plan = sinkhorn_plan(
                cost_matrix,
                s,
                target_s,
                epsilon,
                sinkhorn_iterations,
                tol=sinkhorn_tol,
            )
            ot_value = regularized_ot_value(plan, cost_matrix, s, target_s, epsilon)
            log_weights[r] = -0.5 * lambda_ * ot_value
            gradients[r] = -lambda_ * np.sum(plan[:, :, None] * displacements, axis=1)

        shift = float(np.max(log_weights))
        scaled_weights = np.exp(log_weights - shift)
        denominator = float(np.mean(scaled_weights))
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise FloatingPointError("Monte Carlo weights underflowed; increase epsilon or reduce lambda_")

        numerator = np.mean(scaled_weights[:, None, None] * gradients, axis=0)
        drift = (2.0 / s)[:, None] * (numerator / denominator)

        positions[m + 1] = wrap_torus(x + drift * step_size + em_noise_scale * rng.normal(size=(n, d)))
        if drift_history is not None:
            drift_history[m] = drift

    return ParticleSimulation(times=times, positions=positions, masses=s, drifts=drift_history)
