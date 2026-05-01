from __future__ import annotations

"""Shared numerical utilities for the conditioning examples.

This module centralizes the small validation and numerical helper functions that
were previously duplicated across several example solvers.  Keeping them here
makes the core algorithms easier to audit and reduces the amount of repeated
boilerplate in the repository.
"""

from functools import lru_cache
from itertools import product
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "FloatArray",
    "as_float_array",
    "validate_probability_vector",
    "validate_positions",
    "build_time_grid",
    "get_rng",
    "logsumexp",
    "image_shifts",
]


def as_float_array(x: ArrayLike, *, name: str) -> FloatArray:
    """Convert *x* to a non-empty float64 array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr



def validate_probability_vector(
    x: ArrayLike,
    *,
    name: str,
    normalize: bool = False,
) -> FloatArray:
    """Validate a strictly positive probability vector.

    Parameters
    ----------
    x:
        Candidate vector.
    name:
        Name used in validation errors.
    normalize:
        If ``True``, normalize the vector instead of requiring an exact unit sum.
    """
    arr = as_float_array(x, name=name).reshape(-1)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must have strictly positive entries")
    total = float(arr.sum())
    if normalize:
        arr = arr / total
    elif not np.isclose(total, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to 1; got {total}")
    return arr



def validate_positions(x: ArrayLike, *, n: Optional[int], name: str) -> FloatArray:
    """Validate an array of positions with shape ``(n, d)`` or ``(n,)``."""
    arr = as_float_array(x, name=name)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d) or (n,)")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows; got {arr.shape[0]}")
    return arr



def build_time_grid(horizon: float, step_size: float) -> tuple[int, FloatArray]:
    """Return the number of steps and a uniform time grid."""
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be positive and finite")

    ratio = horizon / step_size
    m_steps = int(round(ratio))
    if m_steps <= 0 or not np.isclose(ratio, m_steps, atol=1e-12, rtol=1e-12):
        raise ValueError("horizon / step_size must be an integer")
    return m_steps, np.linspace(0.0, horizon, m_steps + 1, dtype=np.float64)



def get_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    """Return *rng* or create a default generator."""
    return np.random.default_rng() if rng is None else rng


@lru_cache(maxsize=None)
def image_shifts(radius: int, dimension: int) -> FloatArray:
    """Return periodic image shifts for a truncation radius and dimension."""
    if radius < 0:
        raise ValueError("image_radius must be non-negative")
    return np.array(
        list(product(range(-radius, radius + 1), repeat=dimension)),
        dtype=np.float64,
    )



def logsumexp(a: FloatArray, axis: Optional[int] = None) -> FloatArray:
    """Numerically stable log-sum-exp with optional axis reduction."""
    max_a = np.max(a, axis=axis, keepdims=True)
    shifted = np.exp(a - max_a)
    out = max_a + np.log(np.sum(shifted, axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(out.reshape(()), dtype=np.float64)
    return np.squeeze(out, axis=axis)
