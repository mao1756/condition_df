
from __future__ import annotations

"""Exact fixed-count two-well h-transform for many-particle splitting.

This module implements a scalable version of the Gaussian-mixture idea.

Terminal condition
------------------
Fix two target centers c_L, c_R in R^d and an integer k in {0, ..., n}. Define

    g_k(s, x)
      = sum_{A subset {1,...,n}, |A| = k}
        exp(-lambda/2 * [sum_{i in A} s_i |x_i-c_L|^2
                         + sum_{i notin A} s_i |x_i-c_R|^2]).

This is exactly the same "sum over assignments" h-transform as the small-n
Gaussian-mixture construction, but naively it has binomial(n, k) terms.

Key observation
---------------
Let a_i(t, x_i) and b_i(t, x_i) be the heat-evolved single-particle Gaussian
factors corresponding to the left and right wells. Then the full h-function is
the coefficient of z^k in

    prod_{i=1}^n (b_i + a_i z).

Hence both u_t and the posterior probability that particle i is assigned to the
left well (under the exact k-left constraint) can be computed by dynamic
programming on polynomial coefficients in O(n k) time per Euler step, instead of
O(binomial(n, k)).

The resulting drift is explicit:
    b_i(t, x; s)
      = -2 lambda / (1 + 2 lambda (T-t))
        * [p_i^L(t, x) (x_i - c_L) + (1-p_i^L(t, x)) (x_i - c_R)],

where p_i^L is the exact conditional left-assignment probability under the
h-transform.

This implementation uses the Euclidean lift, exactly as suggested in the paper
for the Gaussian terminal condition when wrap-around effects are negligible.
"""

from typing import Optional

import math
import numpy as np
from numpy.typing import ArrayLike

from conditioning_utils import (
    FloatArray,
    as_float_array as _as_float_array,
    build_time_grid as _build_time_grid,
    get_rng as _get_rng,
    validate_positions as _validate_positions,
    validate_probability_vector as _validate_probability_vector,
)

from wasserstein_conditioning_algorithms import ParticleSimulation, wrap_torus

__all__ = [
    "simulate_fixed_count_two_well_terminal_em",
]


# ---------------------------------------------------------------------------
# Dynamic programming for exact fixed-count assignment
# ---------------------------------------------------------------------------


def _normalized_forward_coefficients(a: FloatArray, b: FloatArray, degree: int) -> tuple[FloatArray, FloatArray]:
    """Return normalized prefix coefficient tables for prod_i (b_i + a_i z).

    For each i = 0, ..., n, this returns a vector F[i, :] of degree <= degree and
    a log-scale lF[i] such that the raw coefficient vector for the first i factors is

        raw_F[i, :] = exp(lF[i]) * F[i, :].

    Each row of F is renormalized by its max entry for numerical stability.
    """
    n = len(a)
    F = np.zeros((n + 1, degree + 1), dtype=np.float64)
    log_scale = np.zeros(n + 1, dtype=np.float64)
    F[0, 0] = 1.0

    for i in range(1, n + 1):
        ai = float(a[i - 1])
        bi = float(b[i - 1])
        prev = F[i - 1]
        cur = np.zeros(degree + 1, dtype=np.float64)

        cur[0] = bi * prev[0]
        max_degree = min(i, degree)
        for m in range(1, max_degree + 1):
            cur[m] = bi * prev[m] + ai * prev[m - 1]

        scale = float(np.max(cur))
        if not np.isfinite(scale) or scale <= 0.0:
            raise FloatingPointError("prefix coefficient recursion became singular")
        F[i] = cur / scale
        log_scale[i] = log_scale[i - 1] + math.log(scale)

    return F, log_scale


def _normalized_backward_coefficients(a: FloatArray, b: FloatArray, degree: int) -> tuple[FloatArray, FloatArray]:
    """Return normalized suffix coefficient tables for prod_{j=i}^n (b_j + a_j z).

    The return arrays S and lS satisfy

        raw_S[i, :] = exp(lS[i]) * S[i, :],   i = 1, ..., n+1,

    with raw_S[n+1, 0] = 1.  The index convention is convenient for the
    leave-one-out convolution used below.
    """
    n = len(a)
    S = np.zeros((n + 2, degree + 1), dtype=np.float64)
    log_scale = np.zeros(n + 2, dtype=np.float64)
    S[n + 1, 0] = 1.0

    for i in range(n, 0, -1):
        ai = float(a[i - 1])
        bi = float(b[i - 1])
        prev = S[i + 1]
        cur = np.zeros(degree + 1, dtype=np.float64)

        cur[0] = bi * prev[0]
        max_degree = min(n - i + 1, degree)
        for m in range(1, max_degree + 1):
            cur[m] = bi * prev[m] + ai * prev[m - 1]

        scale = float(np.max(cur))
        if not np.isfinite(scale) or scale <= 0.0:
            raise FloatingPointError("suffix coefficient recursion became singular")
        S[i] = cur / scale
        log_scale[i] = log_scale[i + 1] + math.log(scale)

    return S, log_scale


def _left_assignment_probabilities(
    left_weights: FloatArray,
    right_weights: FloatArray,
    k_left: int,
) -> FloatArray:
    """Compute exact left-assignment probabilities under the k-left constraint.

    Parameters
    ----------
    left_weights, right_weights:
        Positive arrays a_i, b_i defining the polynomial
            prod_i (b_i + a_i z).
        They may already be shifted/rescaled by per-particle constants; only the
        relative weights matter for the posterior probabilities.
    k_left:
        Exact number of particles assigned to the left well.

    Returns
    -------
    probs:
        Array p_i^L in [0, 1] giving the exact posterior probability that the
        i-th particle is assigned to the left well under the h-transform.
    """
    n = len(left_weights)
    if not (0 <= k_left <= n):
        raise ValueError(f"k_left must lie in [0, n]; got {k_left} for n={n}")

    if k_left == 0:
        return np.zeros(n, dtype=np.float64)
    if k_left == n:
        return np.ones(n, dtype=np.float64)

    forward, log_forward = _normalized_forward_coefficients(left_weights, right_weights, k_left)
    backward, log_backward = _normalized_backward_coefficients(left_weights, right_weights, k_left)

    denom_norm = float(forward[n, k_left])
    if denom_norm <= 0.0 or not np.isfinite(denom_norm):
        raise FloatingPointError("global degree-k coefficient vanished numerically")
    log_denom = log_forward[n] + math.log(denom_norm)

    probs = np.empty(n, dtype=np.float64)

    for i in range(1, n + 1):
        # Leave particle i out, then take the degree-(k_left-1) coefficient of
        # the remaining polynomial by convolving prefix and suffix coefficients.
        conv_sum = float(np.dot(forward[i - 1, :k_left], backward[i + 1, k_left - 1 :: -1]))
        if conv_sum <= 0.0 or not np.isfinite(conv_sum):
            probs[i - 1] = 0.0
            continue

        log_num = (
            math.log(float(left_weights[i - 1]))
            + log_forward[i - 1]
            + log_backward[i + 1]
            + math.log(conv_sum)
        )
        p_left = math.exp(log_num - log_denom)
        probs[i - 1] = min(max(p_left, 0.0), 1.0)

    return probs


# ---------------------------------------------------------------------------
# Exact fixed-count two-well h-transform
# ---------------------------------------------------------------------------


def simulate_fixed_count_two_well_terminal_em(
    masses: ArrayLike,
    left_center: ArrayLike,
    right_center: ArrayLike,
    k_left: int,
    lambda_: float,
    horizon: float,
    step_size: float,
    initial_positions: ArrayLike,
    *,
    rng: Optional[np.random.Generator] = None,
    store_drifts: bool = True,
) -> ParticleSimulation:
    """Euler--Maruyama for the exact fixed-count two-well terminal condition.

    This is an exact h-transform solver for the terminal weight

        g_k(s, x)
          = sum_{|A|=k_left}
            exp(-lambda/2 * [sum_{i in A} s_i |x_i-c_L|^2
                             + sum_{i notin A} s_i |x_i-c_R|^2]).

    The naive binomial sum is replaced by O(n k_left) dynamic programming at each
    time step.

    Parameters
    ----------
    masses:
        Positive masses summing to 1.
    left_center, right_center:
        Two target centers in R^d. These are interpreted on the Euclidean lift.
    k_left:
        Exact number of particles assigned to the left well in the terminal
        mixture.
    lambda_:
        Terminal concentration parameter.
    horizon, step_size:
        Simulation horizon T and Euler step Δt.
    initial_positions:
        Initial particle locations on the Euclidean lift.
    rng:
        Optional random number generator.
    store_drifts:
        Whether to retain the drift at every Euler step.

    Returns
    -------
    ParticleSimulation
        Wrapped torus positions, the lifted trajectory, and optionally the drifts.
    """
    if lambda_ <= 0.0 or not np.isfinite(lambda_):
        raise ValueError("lambda_ must be positive and finite")

    s = _validate_probability_vector(masses, name="masses")
    n = len(s)
    if not (0 <= int(k_left) <= n):
        raise ValueError(f"k_left must lie in [0, n]; got {k_left} for n={n}")
    k_left = int(k_left)

    x0 = _validate_positions(initial_positions, n=n, name="initial_positions")
    left = _as_float_array(left_center, name="left_center").reshape(-1)
    right = _as_float_array(right_center, name="right_center").reshape(-1)
    if x0.shape[1] != left.shape[0] or x0.shape[1] != right.shape[0]:
        raise ValueError("initial_positions and centers must have the same dimension")

    rng = _get_rng(rng)
    m_steps, times = _build_time_grid(horizon, step_size)
    _, d = x0.shape

    wrapped_positions = np.empty((m_steps + 1, n, d), dtype=np.float64)
    lifted_positions = np.empty_like(wrapped_positions)
    wrapped_positions[0] = wrap_torus(x0)
    lifted_positions[0] = x0.copy()
    drift_history = np.empty((m_steps, n, d), dtype=np.float64) if store_drifts else None

    state = x0.copy()
    noise_scale = np.sqrt(2.0 * step_size / s)[:, None]

    for m in range(m_steps):
        tau = horizon - times[m]

        diff_left = state - left[None, :]
        diff_right = state - right[None, :]

        quadratic_prefactor = 0.5 * lambda_ * s / (1.0 + 2.0 * lambda_ * tau)
        log_left = -quadratic_prefactor * np.sum(diff_left ** 2, axis=1)
        log_right = -quadratic_prefactor * np.sum(diff_right ** 2, axis=1)

        # Particlewise shifting removes harmless common factors and keeps the DP
        # numerically well scaled.
        shift = np.maximum(log_left, log_right)
        left_weights = np.exp(log_left - shift)
        right_weights = np.exp(log_right - shift)

        p_left = _left_assignment_probabilities(left_weights, right_weights, k_left)
        drift_prefactor = -2.0 * lambda_ / (1.0 + 2.0 * lambda_ * tau)
        drift = drift_prefactor * (
            p_left[:, None] * diff_left + (1.0 - p_left)[:, None] * diff_right
        )

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
