from __future__ import annotations

"""Gaussian-mixture h-transform utilities for robust multimodal splitting on the torus.

This module adds a small extension to the manuscript's Algorithm 1.

Instead of a single labelled Gaussian terminal weight,

    g(x) = exp(-lambda/2 * sum_i s_i |x_i - y_i|^2),

we use a positive finite mixture

    g(x) = sum_r alpha_r * exp(-lambda/2 * sum_i s_i |x_i - y_{r,i}|^2).

Because the backward heat equation is linear, the corresponding h-function is the
same mixture of the explicit Gaussian heat evolutes. The drift is therefore still
available in closed form: it is a responsibility-weighted average of the drifts of
all Gaussian components.

This is useful for *unlabelled* multimodal targets. In particular, if the target is
"three particles left, three particles right" but we do not want to commit in advance
which labels go left or right, we can sum over all left/right assignments.

The implementation below uses the Euclidean lift, exactly as the manuscript suggests
for the Gaussian target when wrap-around effects are negligible.
"""

from itertools import combinations
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from conditioning_utils import (
    FloatArray,
    as_float_array as _as_float_array,
    build_time_grid as _build_time_grid,
    get_rng as _get_rng,
    logsumexp as _logsumexp,
    validate_positions as _validate_positions,
    validate_probability_vector as _validate_probability_vector,
)

from wasserstein_conditioning_algorithms import ParticleSimulation, wrap_torus

__all__ = [
    "simulate_gaussian_mixture_terminal_em",
    "build_fixed_count_two_cluster_targets",
    "build_mass_split_two_cluster_targets",
]


# ---------------------------------------------------------------------------
# Mixture h-transform
# ---------------------------------------------------------------------------


def simulate_gaussian_mixture_terminal_em(
    masses: ArrayLike,
    target_configurations: ArrayLike,
    component_weights: ArrayLike,
    lambda_: float,
    horizon: float,
    step_size: float,
    initial_positions: ArrayLike,
    *,
    rng: Optional[np.random.Generator] = None,
    store_drifts: bool = True,
) -> ParticleSimulation:
    """Euler--Maruyama for a finite Gaussian-mixture terminal h-transform.

    The terminal function is

        g(x) = sum_{r=1}^R alpha_r * exp(-lambda_/2 * sum_i s_i |x_i - y_{r,i}|^2),

    where ``alpha_r > 0`` and the target configurations ``y_r`` are supplied in
    ``target_configurations``.

    By linearity of the backward heat equation, the corresponding h-function is
    a finite sum of the explicit Gaussian heat evolutes from Algorithm 1. The drift
    at time ``t`` can therefore be written in closed form as

        b_i(t, x) = - 2 lambda_ / (1 + 2 lambda_ (T - t))
                    * (x_i - sum_r pi_r(t, x) y_{r,i}),

    where ``pi_r`` is the normalized responsibility of component ``r``.

    Notes
    -----
    * This implementation uses the Euclidean lift, not the torus image sum.
      It is most accurate when the conditioned paths stay away from wrap-around.
    * For strong conditioning the time step must usually be much smaller than in
      the simple labelled Gaussian example. In the demo notebook, ``1e-4`` works
      well while ``5e-4`` is already visibly too coarse.
    """
    if lambda_ <= 0.0 or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be positive and finite")

    s = _validate_probability_vector(masses, name="masses")
    n = len(s)
    x0 = _validate_positions(initial_positions, n=n, name="initial_positions")
    targets = _as_float_array(target_configurations, name="target_configurations")
    if targets.ndim != 3:
        raise ValueError("target_configurations must have shape (R, n, d)")
    if targets.shape[1] != n:
        raise ValueError(
            f"target_configurations must have {n} particles in axis 1; got {targets.shape[1]}"
        )
    if targets.shape[2] != x0.shape[1]:
        raise ValueError(
            "target_configurations and initial_positions must have the same spatial dimension"
        )

    alpha = _validate_probability_vector(component_weights, name="component_weights")
    if len(alpha) != targets.shape[0]:
        raise ValueError(
            "component_weights and target_configurations must have matching first dimension"
        )

    rng = _get_rng(rng)
    m_steps, times = _build_time_grid(horizon, step_size)
    _, d = x0.shape
    num_components = targets.shape[0]

    wrapped_positions = np.empty((m_steps + 1, n, d), dtype=np.float64)
    wrapped_positions[0] = wrap_torus(x0)
    lifted_positions = np.empty_like(wrapped_positions)
    lifted_positions[0] = x0.copy()
    drift_history = np.empty((m_steps, n, d), dtype=np.float64) if store_drifts else None

    state = x0.copy()
    log_alpha = np.log(alpha)
    noise_scale = np.sqrt(2.0 * step_size / s)[:, None]
    mass_weights = s[None, :, None]

    for m in range(m_steps):
        tau = horizon - times[m]
        alpha_tau = lambda_ / (2.0 * (1.0 + 2.0 * lambda_ * tau))
        beta_tau = 2.0 * lambda_ / (1.0 + 2.0 * lambda_ * tau)

        deltas = state[None, :, :] - targets
        energies = np.sum(mass_weights * deltas * deltas, axis=(1, 2))

        log_resp = log_alpha - alpha_tau * energies
        lse = _logsumexp(log_resp)
        resp = np.exp(log_resp - lse)

        barycenter = np.tensordot(resp, targets, axes=(0, 0))
        drift = -beta_tau * (state - barycenter)

        state = state + drift * step_size + noise_scale * rng.normal(size=(n, d))
        wrapped_positions[m + 1] = wrap_torus(state)
        lifted_positions[m + 1] = state
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
# Target builders for the two-circle split example
# ---------------------------------------------------------------------------


def build_fixed_count_two_cluster_targets(
    num_particles: int,
    left_center: ArrayLike,
    right_center: ArrayLike,
    left_count: int,
) -> FloatArray:
    """Return all target configurations with exactly ``left_count`` particles on the left.

    Each component places the chosen labels at ``left_center`` and all remaining
    labels at ``right_center``. If ``left_count = 3`` and ``num_particles = 6``,
    the returned array has shape ``(20, 6, d)``.
    """
    if left_count < 0 or left_count > num_particles:
        raise ValueError("left_count must lie between 0 and num_particles")

    left = _as_float_array(left_center, name="left_center").reshape(1, -1)
    right = _as_float_array(right_center, name="right_center").reshape(1, -1)
    if left.shape != right.shape:
        raise ValueError("left_center and right_center must have the same dimension")

    d = left.shape[1]
    components = []
    for subset in combinations(range(num_particles), left_count):
        target = np.repeat(right, num_particles, axis=0)
        target[list(subset)] = left
        components.append(target)
    return np.asarray(components, dtype=np.float64).reshape(-1, num_particles, d)



def build_mass_split_two_cluster_targets(
    masses: ArrayLike,
    left_center: ArrayLike,
    right_center: ArrayLike,
    target_left_mass: float,
    *,
    atol: float = 1e-12,
) -> FloatArray:
    """Return all target configurations whose left-cluster mass matches ``target_left_mass``.

    This is useful when the desired split is a *mass* split rather than a fixed
    particle count split. For the notebook's masses

        [0.26, 0.20, 0.17, 0.14, 0.13, 0.10],

    the exact 0.5/0.5 mass split produces the two complementary assignments
    ``{0, 3, 5}`` and ``{1, 2, 4}``.
    """
    s = _validate_probability_vector(masses, name="masses")
    num_particles = len(s)
    left = _as_float_array(left_center, name="left_center").reshape(1, -1)
    right = _as_float_array(right_center, name="right_center").reshape(1, -1)
    if left.shape != right.shape:
        raise ValueError("left_center and right_center must have the same dimension")

    d = left.shape[1]
    components = []
    for r in range(num_particles + 1):
        for subset in combinations(range(num_particles), r):
            if np.isclose(np.sum(s[list(subset)]), target_left_mass, atol=atol, rtol=0.0):
                target = np.repeat(right, num_particles, axis=0)
                target[list(subset)] = left
                components.append(target)
    if not components:
        raise ValueError(
            "no subset of particle masses matches target_left_mass within the requested tolerance"
        )
    return np.asarray(components, dtype=np.float64).reshape(-1, num_particles, d)
