from __future__ import annotations

r"""Scalable factorized Gaussian-mixture h-transform for two-well conditioning.

This module targets the case where we do **not** care about exact left/right counts.
The terminal condition is the factorized positive function

    g(x_1, ..., x_n)
      = \prod_{i=1}^n \sum_{j=1}^J alpha_j exp(-kappa/2 * d(x_i, c_j)^2),

where ``c_j`` are allowed target centers and ``alpha_j`` are mixture weights.
For the free finite-dimensional generator

    L^(n)_s = \sum_i s_i^{-1} \Delta_{x_i},

the Euclidean heat evolution of each one-particle Gaussian is explicit, hence the
corresponding h-transform drift is explicit as well. The cost per Euler step is
O(n J), so this scales linearly in the number of particles for fixed number of
wells ``J``.

When ``J = 2`` and the centers are the two circle centers, this directly encodes
"every particle should end near one of the circles" without imposing an exact
split ratio.

Notes
-----
* ``distance_mode='euclidean'`` is the exact closed-form h-transform on the
  Euclidean lift. This is the mathematically clean mode when wrap-around is
  negligible on the time horizon.
* ``distance_mode='periodic'`` is a practical torus heuristic that replaces
  Euclidean displacements by shortest periodic displacements. It is convenient
  for experiments on [0, 1)^d but is not the exact torus heat evolution unless
  one also includes the full image sum.
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from core.conditioning_utils import (
    FloatArray,
    as_float_array as _as_float_array,
    build_time_grid as _build_time_grid,
    get_rng as _get_rng,
    validate_positions as _validate_positions,
    validate_probability_vector as _validate_probability_vector,
)

from core.wasserstein_conditioning_algorithms import (
    ParticleSimulation,
    shortest_periodic_displacement,
    wrap_torus,
)

__all__ = [
    "circle_membership_stats",
    "heuristic_equal_mass_parameters",
    "simulate_factorized_gaussian_mixture_em",
]


def heuristic_equal_mass_parameters(
    n_particles: int,
    *,
    horizon_constant: float = 0.5,
    kappa_per_particle: float = 25.6,
    steps: int = 1000,
) -> dict[str, float]:
    """Heuristic parameter regime for equal-mass many-particle splits.

    For equal masses ``s_i = 1/n``, the free noise scale in the Euler scheme is
    ``sqrt(2 / s_i) = sqrt(2n)``. A practical way to preserve localization as
    ``n`` grows is to shrink the horizon roughly like ``1/n`` and grow the well
    stiffness roughly like ``n``. The defaults match a robust regime on the toy
    two-circle problem.
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    horizon = float(horizon_constant) / float(n_particles)
    kappa = float(kappa_per_particle) * float(n_particles)
    step_size = horizon / float(steps)
    return {
        "kappa": kappa,
        "horizon": horizon,
        "step_size": step_size,
    }


def circle_membership_stats(
    final_positions: ArrayLike,
    centers: ArrayLike,
    radius: float,
    *,
    periodic: bool = True,
) -> dict[str, object]:
    """Summarize terminal membership in a collection of target circles."""
    pos = _validate_positions(final_positions, n=None, name="final_positions")
    ctr = _validate_positions(centers, n=None, name="centers")
    if pos.shape[1] != ctr.shape[1]:
        raise ValueError("final_positions and centers must have the same dimension")
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be positive and finite")

    if periodic:
        displacements = shortest_periodic_displacement(pos[:, None, :], ctr[None, :, :])
    else:
        displacements = pos[:, None, :] - ctr[None, :, :]
    distances = np.sqrt(np.sum(displacements ** 2, axis=-1))
    nearest = np.argmin(distances, axis=1)
    inside = np.any(distances <= radius, axis=1)
    return {
        "all_inside": bool(np.all(inside)),
        "inside_fraction": float(np.mean(inside)),
        "inside_counts": [int(np.sum(distances[:, j] <= radius)) for j in range(len(ctr))],
        "nearest_center_counts": [int(np.sum(nearest == j)) for j in range(len(ctr))],
    }


def _mixture_drift(
    state: FloatArray,
    masses: FloatArray,
    centers: FloatArray,
    mixture_weights: FloatArray,
    kappa: float,
    tau: float,
    *,
    distance_mode: str,
) -> FloatArray:
    """Explicit h-transform drift for the factorized mixture terminal."""
    if distance_mode == "euclidean":
        displacements = state[:, None, :] - centers[None, :, :]
    elif distance_mode == "periodic":
        displacements = shortest_periodic_displacement(state[:, None, :], centers[None, :, :])
    else:
        raise ValueError("distance_mode must be 'euclidean' or 'periodic'")

    sq_dist = np.sum(displacements ** 2, axis=-1)
    # One-particle heat evolution of exp(-kappa |x-c|^2 / 2) under diffusion 1/s_i.
    # The common Gaussian prefactor (1 + 2 kappa tau / s_i)^(-d/2) cancels within each
    # particle's mixture responsibilities, so only the exponent needs to be retained.
    a = kappa / (2.0 * (1.0 + 2.0 * kappa * tau / masses))
    log_weights = np.log(mixture_weights)[None, :] - a[:, None] * sq_dist
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    responsibilities = weights / np.sum(weights, axis=1, keepdims=True)

    # Weighted average of the center displacements.
    barycenter_disp = np.sum(responsibilities[:, :, None] * displacements, axis=1)
    factor = -2.0 * kappa / (masses + 2.0 * kappa * tau)
    return factor[:, None] * barycenter_disp


def simulate_factorized_gaussian_mixture_em(
    masses: ArrayLike,
    centers: ArrayLike,
    kappa: float,
    horizon: float,
    step_size: float,
    initial_positions: ArrayLike,
    *,
    mixture_weights: Optional[ArrayLike] = None,
    distance_mode: str = "euclidean",
    rng: Optional[np.random.Generator] = None,
    store_drifts: bool = True,
) -> ParticleSimulation:
    """Euler--Maruyama for the factorized Gaussian-mixture terminal h-transform.

    Parameters
    ----------
    masses:
        Frozen masses ``(s_1, ..., s_n)`` summing to 1.
    centers:
        Allowed target centers ``c_1, ..., c_J``. In the two-circle example,
        ``J = 2``.
    kappa:
        Well sharpness parameter in the terminal weight.
    horizon, step_size:
        Time horizon ``T`` and Euler step ``Δt``.
    initial_positions:
        Initial particle locations.
    mixture_weights:
        Positive center weights ``alpha_j``. Defaults to the uniform distribution.
    distance_mode:
        ``'euclidean'`` gives the exact explicit h-transform on the lift.
        ``'periodic'`` uses shortest periodic displacements as a torus heuristic.
    rng:
        Optional ``numpy.random.Generator``.
    store_drifts:
        Whether to keep the drift history.
    """
    if kappa <= 0.0 or not np.isfinite(kappa):
        raise ValueError("kappa must be positive and finite")

    s = _validate_probability_vector(masses, name="masses")
    n = len(s)
    x0 = _validate_positions(initial_positions, n=n, name="initial_positions")
    ctr = _validate_positions(centers, n=None, name="centers")
    if ctr.shape[1] != x0.shape[1]:
        raise ValueError("initial_positions and centers must have the same dimension")

    if mixture_weights is None:
        alpha = np.full(len(ctr), 1.0 / len(ctr), dtype=np.float64)
    else:
        alpha = _validate_probability_vector(mixture_weights, name="mixture_weights", normalize=True)
        if len(alpha) != len(ctr):
            raise ValueError("mixture_weights must have one entry per center")

    rng = _get_rng(rng)
    m_steps, times = _build_time_grid(horizon, step_size)
    _, d = x0.shape

    positions = np.empty((m_steps + 1, n, d), dtype=np.float64)
    drift_history = np.empty((m_steps, n, d), dtype=np.float64) if store_drifts else None
    noise_scale = np.sqrt(2.0 * step_size / s)[:, None]

    if distance_mode == "euclidean":
        state = np.array(x0, copy=True)
        lifted_positions = np.empty_like(positions)
        lifted_positions[0] = state
        positions[0] = wrap_torus(state)
    elif distance_mode == "periodic":
        state = wrap_torus(x0)
        lifted_positions = None
        positions[0] = state
    else:
        raise ValueError("distance_mode must be 'euclidean' or 'periodic'")

    for m in range(m_steps):
        tau = horizon - times[m]
        drift = _mixture_drift(
            state,
            s,
            ctr,
            alpha,
            kappa,
            tau,
            distance_mode=distance_mode,
        )

        state = state + drift * step_size + noise_scale * rng.normal(size=(n, d))
        if distance_mode == "periodic":
            state = wrap_torus(state)
            positions[m + 1] = state
        else:
            lifted_positions[m + 1] = state
            positions[m + 1] = wrap_torus(state)

        if drift_history is not None:
            drift_history[m] = drift

    return ParticleSimulation(
        times=times,
        positions=positions,
        masses=s,
        drifts=drift_history,
        lifted_positions=lifted_positions,
    )
